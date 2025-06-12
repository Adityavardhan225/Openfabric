# event_driven_env.py

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import uuid
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

from order_book import OrderBook
from data_processor import L2BookDataProcessor # Import the new processor
from market_message import (
    MarketMessage, OrderSide, AgentAction, MessageType,
    SnapshotMessage, DeltaMessage, TradeMessage
)
from market_features import MarketFeatures

# A simple class to track our agent's orders internally
@dataclass
class AgentOrder:
    order_id: str
    side: OrderSide
    price: float
    size: float
    timestamp: int # When the order was created by the agent
    
@dataclass
class AgentTrade:
    timestamp: int
    price: float
    size: float
    side: str # 'buy' or 'sell'
    fee: float
    is_maker: bool


class L2EventDrivenOrderBookEnv(gym.Env):
    """
    An event-driven order book environment using L2 snapshot/delta data.
    The agent learns to place and cancel limit orders in a realistic,
    high-frequency setting.
    """
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, 
                 data_processor: L2BookDataProcessor,
                 levels: int = 10,
                 lookback_periods: List[int] = [10, 50, 100],
                 history_length: int = 20,
                 init_cash: float = 10000.0,
                 max_position: int = 50,
                 maker_fee: float = -0.0002,
                 taker_fee: float = 0.0007,
                 inventory_penalty: float = 0.05,
                 round_trip_profit_bonus: float = 10.0,
                 episode_length_seconds: Optional[int] = 1800, # Episode length in seconds
                 taker_penalty: float = 0.01,           # NEW: Penalty for aggressive orders
                 maker_reward: float = 0.01,            # NEW: Reward for passive fills
                 pnl_reward_multiplier: float = 5.0,    # NEW: Multiplier for realized PnL
                 seed: Optional[int] = None):
        
        super().__init__()
        
        # Core components
        self.data_processor = data_processor
        self.order_book = OrderBook()
        self.market_features = MarketFeatures(lookback_periods)
        
        # Environment parameters
        self.levels = levels
        self.history_length = history_length
        self.init_cash = init_cash
        self.max_position = max_position
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.inventory_penalty = inventory_penalty
        self.taker_penalty = taker_penalty
        self.maker_reward = maker_reward
        self.pnl_reward_multiplier = pnl_reward_multiplier
        self.round_trip_profit_bonus = round_trip_profit_bonus
        
        if episode_length_seconds:
            self.episode_length_ns = episode_length_seconds * 1_000_000_000
        else:
            self.episode_length_ns = None

        self.seed(seed)
        
        # Agent and episode state (will be reset)
        self.cash = 0.0
        self.inventory = 0
        self.avg_entry_price = 0.0
        self.agent_orders: Dict[str, AgentOrder] = {}
        self.agent_trades: List[AgentTrade] = []
        self.pnl_history = deque(maxlen=1000)
        self.state_history = deque(maxlen=history_length)
        
        self.current_timestamp = 0
        self.episode_start_timestamp = 0
        self.done = False
        self.num_steps = 0 
        self.pending_markouts = []
        # A list to store the calculated adverse selection costs for each trade
        self.adverse_selection_costs = []
        # How many steps into the future to check the price
        self.markout_delay_steps = 10 # Configurable: check price 10 steps later
        # --------------------------------
        # Metrics
        self._reset_metrics()
        
        # Observation and action spaces
        self._setup_observation_space()
        self._setup_action_space()
        
    def _reset_metrics(self):
        self.total_orders_placed = 0
        self.total_orders_filled = 0
        self.total_orders_canceled = 0
        self.maker_volume = 0.0
        self.taker_volume = 0.0
        self.fees_paid = 0.0

        self.pending_markouts.clear()
        self.adverse_selection_costs.clear()

    def _setup_observation_space(self):
        # Book levels: price, volume for each side
        book_features = self.levels * 4
        # Market features (from MarketFeatures class)
        # 5 base + 3 per lookback * num_lookbacks + 2 time
        num_lookbacks = len(self.market_features.lookback_periods)
        market_features_count = 5 + 3 * num_lookbacks + 2
        # Agent state features
        agent_features = 4  # cash, inventory, num_buy_orders, num_sell_orders
        
        # REMOVED: Queue position features (L3 concept)
        # ADDED: simpler order features
        order_features = 4  # avg_buy_price_offset, avg_sell_price_offset, total_buy_size, total_sell_size

        feature_dim = book_features + market_features_count + agent_features + order_features
        obs_dim = (self.history_length, feature_dim)
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_dim, dtype=np.float32
        )
        
    def _setup_action_space(self):
        # Using a simpler discrete action space is often more stable
        # 0: Do nothing
        # 1: Place LIMIT BUY at best bid
        # 2: Place LIMIT SELL at best ask
        # 3: Place MARKET BUY (cross spread)
        # 4: Place MARKET SELL (cross spread)
        # 5: Cancel all open BUY orders
        # 6: Cancel all open SELL orders
        self.action_space = spaces.Discrete(11)
        # For simplicity, we'll hardcode order size for now, e.g., 1 unit.
        # This can be parameterized later if needed.

    def seed(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
    
# In L2EventDrivenOrderBookEnv class

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self.seed(seed)
        
        # Reset data processor, potentially to a random starting point
        self.data_processor.reset(options)
        
        # Reset other components
        self.market_features = MarketFeatures(self.market_features.lookback_periods)
        self.cash = self.init_cash
        self.inventory = 0
        self.avg_entry_price = 0.0
        self.agent_orders.clear()
        self.agent_trades.clear()
        self.pnl_history.clear()
        self.state_history.clear()
        self.done = False
        self.num_steps = 0
        self._reset_metrics()
        
        # --- MODIFICATION IS HERE ---
        # Find the FIRST available snapshot from the current file position to initialize the book
        logging.info("Searching for the first snapshot to initialize episode...")
        msg = self.data_processor.get_next_message()
        
        # Keep reading until we find a snapshot or run out of data
        while msg and not isinstance(msg, SnapshotMessage):
            msg = self.data_processor.get_next_message()
            
        if not msg:
            # This can happen if we do a random start close to the end of the file
            logging.warning("Could not find a SnapshotMessage to initialize the environment. Data file might be exhausted.")
            # Return None to signal the training loop to try resetting again or stop.
            return None, None
        
        # We found a snapshot, initialize the book with it.
        self.order_book.process_snapshot(msg)
        self.current_timestamp = msg.timestamp
        self.episode_start_timestamp = self.current_timestamp
        logging.info(f"Episode initialized from snapshot at timestamp {self.current_timestamp}")

        # Populate initial state history
        if self.order_book.get_mid_price() is not None:
            self.market_features.update(self.order_book, self.current_timestamp)
            initial_state = self._build_state_features()
            for _ in range(self.history_length):
                self.state_history.append(initial_state)
        else: 
            zeros = np.zeros(self.observation_space.shape[1], dtype=np.float32)
            for _ in range(self.history_length):
                self.state_history.append(zeros)

        return self._get_observation(), self._get_info()

    def step(self, action: int):
        # 1. Execute agent action
        self.num_steps += 1 # Increment step counter at the beginning
        
        # 1. Process any markouts that are due from previous fills
        self._process_markouts()
        action_reward = self._take_action(action)
        
        # 2. Process next market data message
        msg = self.data_processor.get_next_message()
        if msg is None: # End of data
            self.done = True
            return self._get_observation(), 0.0, True, False, self._get_info()

        self._process_market_message(msg)

        # 3. Check for fills of our existing orders based on the new market state
        fill_reward = self._check_for_fills()
        total_reward = action_reward + fill_reward
        
        # 4. Apply inventory penalty
        total_reward -= self.inventory_penalty * (self.inventory / self.max_position)**2

        # 5. Update state and check for episode end
        self._update_state_history()
        if self.episode_length_ns and (self.current_timestamp - self.episode_start_timestamp) >= self.episode_length_ns:
            self.done = True

        return self._get_observation(), total_reward, self.done, False, self._get_info()
    
    def _process_markouts(self):
        """
        Processes pending markouts to calculate adverse selection.
        """
        # Avoid modifying the list while iterating
        remaining_markouts = []
        
        future_mid_price = self.order_book.get_mid_price()
        if future_mid_price is None:
            # If we can't get a price, keep all markouts pending
            self.pending_markouts.extend(remaining_markouts)
            return

        for check_step, fill_price, side in self.pending_markouts:
            if self.num_steps >= check_step:
                # This markout is due. Calculate the cost.
                if side == OrderSide.BUY:
                    # For a buy, adverse selection is when the future price is LOWER than our fill price.
                    # Cost is positive if adverse.
                    cost = fill_price - future_mid_price
                else: # SELL
                    # For a sell, adverse selection is when the future price is HIGHER than our fill price.
                    cost = future_mid_price - fill_price
                
                self.adverse_selection_costs.append(cost)
            else:
                # This markout is not due yet, keep it for the next step.
                remaining_markouts.append((check_step, fill_price, side))
        
        # Update the list of pending markouts
        self.pending_markouts = remaining_markouts

    def _process_market_message(self, msg: MarketMessage):
        """Processes a single message from the data feed."""
        self.current_timestamp = msg.timestamp
        if isinstance(msg, (SnapshotMessage, DeltaMessage)):
            self.order_book.process_snapshot(msg) if isinstance(msg, SnapshotMessage) else self.order_book.process_delta(msg)
        elif isinstance(msg, TradeMessage):
            self.market_features.add_trade(msg)
            self.order_book.last_trade_price = msg.price
        # Update features after every market message
        self.market_features.update(self.order_book, self.current_timestamp)
        
    def _take_action(self, action_type: int) -> float:
        """Processes agent action based on the new action space and returns an immediate reward/penalty."""
        if self.order_book.get_mid_price() is None: return 0.0

        order_size = 1.0  # Fixed size for simplicity
        tick_size = self.order_book.tick_size
        reward = 0.0

        def place_limit_order(side, price_offset_ticks):
            if side == OrderSide.BUY:
                base_price = self.order_book.get_best_bid()
                if base_price is None or self.inventory >= self.max_position: return
                price = base_price - (price_offset_ticks * tick_size)
            else: # SELL
                base_price = self.order_book.get_best_ask()
                if base_price is None or self.inventory <= -self.max_position: return
                price = base_price + (price_offset_ticks * tick_size)
            
            order_id = str(uuid.uuid4())
            self.agent_orders[order_id] = AgentOrder(order_id, side, price, order_size, self.current_timestamp)
            self.total_orders_placed += 1

        if action_type == 0:   pass
        elif action_type == 1: place_limit_order(OrderSide.BUY, 0)
        elif action_type == 2: place_limit_order(OrderSide.BUY, 1)
        elif action_type == 3: place_limit_order(OrderSide.BUY, 2)
        elif action_type == 4: place_limit_order(OrderSide.SELL, 0)
        elif action_type == 5: place_limit_order(OrderSide.SELL, 1)
        elif action_type == 6: place_limit_order(OrderSide.SELL, 2)
        
        elif action_type == 7: # MARKET BUY
            price = self.order_book.get_best_ask()
            if price and self.inventory < self.max_position:
                # _execute_trade does NOT return a reward in this case
                self._execute_trade(price, order_size, OrderSide.BUY, is_maker=False)
                reward -= self.taker_penalty # Apply penalty
        
        elif action_type == 8: # MARKET SELL
            price = self.order_book.get_best_bid()
            if price and self.inventory > -self.max_position:
                self._execute_trade(price, order_size, OrderSide.SELL, is_maker=False)
                reward -= self.taker_penalty # Apply penalty

        elif action_type == 9: # Cancel all BUY
            ids_to_cancel = [oid for oid, o in self.agent_orders.items() if o.side == OrderSide.BUY]
            for oid in ids_to_cancel: del self.agent_orders[oid]
            self.total_orders_canceled += len(ids_to_cancel)

        elif action_type == 10: # Cancel all SELL
            ids_to_cancel = [oid for oid, o in self.agent_orders.items() if o.side == OrderSide.SELL]
            for oid in ids_to_cancel: del self.agent_orders[oid]
            self.total_orders_canceled += len(ids_to_cancel)
            
        return reward

    def _check_for_fills(self) -> float:
        """
        Check if any agent orders were filled by the market moving.
        This is the new L2 matching engine logic.
        """
        best_bid = self.order_book.get_best_bid()
        best_ask = self.order_book.get_best_ask()
        total_reward = 0.0
        
        filled_order_ids = []
        for order_id, order in self.agent_orders.items():
            fill_price = None
            if order.side == OrderSide.BUY and best_ask is not None and order.price >= best_ask:
                # Our buy limit order is crossed by the ask side -> fill
                fill_price = order.price # Assume we get our price
            elif order.side == OrderSide.SELL and best_bid is not None and order.price <= best_bid:
                # Our sell limit order is crossed by the bid side -> fill
                fill_price = order.price

            if fill_price:
                reward = self._execute_trade(fill_price, order.size, order.side, is_maker=True)
                total_reward += reward
                filled_order_ids.append(order_id)

                markout_check_step = self.num_steps + self.markout_delay_steps
                self.pending_markouts.append((markout_check_step, fill_price, order.side))
        
        for oid in filled_order_ids:
            del self.agent_orders[oid]
            
        return total_reward

    # <<< REPLACE THIS ENTIRE METHOD >>>
    def _execute_trade(self, price: float, size: float, side: OrderSide, is_maker: bool) -> float:
        """
        Applies trade effects and calculates a comprehensive reward for MAKER fills.
        Taker penalties are handled in _take_action. Returns the calculated reward.
        """
        inventory_before = self.inventory
        reward = 0.0
        
        # 1. Realized PnL Reward (only calculated if the trade reduces/closes a position)
        is_closing_long = inventory_before > 0 and side == OrderSide.SELL
        is_closing_short = inventory_before < 0 and side == OrderSide.BUY

        if is_closing_long or is_closing_short:
            pnl_per_unit = (price - self.avg_entry_price) if is_closing_long else (self.avg_entry_price - price)
            reward += pnl_per_unit * size * self.pnl_reward_multiplier

        # 2. Maker Reward (only for passive fills)
        if is_maker:
            reward += self.maker_reward

        # --- State and Cost Basis Updates ---
        fee_rate = self.maker_fee if is_maker else self.taker_fee
        trade_value = price * size
        fee = trade_value * fee_rate
        self.cash -= fee
        self.fees_paid += fee
        
        if side == OrderSide.BUY:
            self.cash -= trade_value
            self.inventory += size
        else: # SELL
            self.cash += trade_value
            self.inventory -= size

        # Update Average Entry Price (Cost Basis)
        if (inventory_before > 0 and self.inventory <= 0) or (inventory_before < 0 and self.inventory >= 0):
            self.avg_entry_price = price if self.inventory != 0 else 0.0
        elif (inventory_before >= 0 and side == OrderSide.BUY) or (inventory_before <= 0 and side == OrderSide.SELL):
            if abs(self.inventory) > 1e-9: # Avoid division by zero if inventory becomes tiny
                self.avg_entry_price = ((self.avg_entry_price * abs(inventory_before)) + (price * size)) / abs(self.inventory)
        
        # --- Metrics Update ---
        if is_maker:
            self.maker_volume += trade_value
            self.total_orders_filled += 1
        else:
            self.taker_volume += trade_value
        self.agent_trades.append(AgentTrade(self.current_timestamp, price, size, side.name.lower(), fee, is_maker))
        
        return reward
    def _build_state_features(self):
        # ... This method needs to be implemented based on the new observation space ...
        # For now, a placeholder:
        # Get book levels
        bids, asks = self.order_book.get_book_levels(self.levels)
        
        # Flatten and normalize book data
        mid_price = self.order_book.get_mid_price()
        if mid_price is None or mid_price == 0: mid_price = 1.0 # Avoid division by zero
        
        book_features = []
        for p, s in bids: book_features.extend([p / mid_price, s])
        for p, s in asks: book_features.extend([p / mid_price, s])
        
        # Get market features
        market_features_dict = self.market_features.get_all_features(self.order_book, self.current_timestamp)
        # Ensure consistent order
        mkt_features_list = [v for k, v in sorted(market_features_dict.items())]

        # Agent state features
        buy_orders = [o for o in self.agent_orders.values() if o.side == OrderSide.BUY]
        sell_orders = [o for o in self.agent_orders.values() if o.side == OrderSide.SELL]
        agent_features = [
            self.cash / self.init_cash,
            self.inventory / self.max_position,
            len(buy_orders),
            len(sell_orders)
        ]

        # Working order features
        avg_buy_offset = np.mean([(o.price - mid_price) / mid_price for o in buy_orders]) if buy_orders else 0
        avg_sell_offset = np.mean([(o.price - mid_price) / mid_price for o in sell_orders]) if sell_orders else 0
        total_buy_size = sum(o.size for o in buy_orders) / self.max_position
        total_sell_size = sum(o.size for o in sell_orders) / self.max_position
        order_features = [avg_buy_offset, avg_sell_offset, total_buy_size, total_sell_size]

        state_vector = np.concatenate([
            book_features, mkt_features_list, agent_features, order_features
        ]).astype(np.float32)
        
        # Check for shape mismatch
        expected_len = self.observation_space.shape[1]
        if len(state_vector) != expected_len:
             # This is a critical error, pad or truncate as a temporary fix, but investigate the cause
            logging.warning(f"State vector length mismatch! Got {len(state_vector)}, expected {expected_len}. Padding/truncating.")
            padded_vector = np.zeros(expected_len, dtype=np.float32)
            copy_len = min(len(state_vector), expected_len)
            padded_vector[:copy_len] = state_vector[:copy_len]
            return padded_vector
            
        return state_vector

    def _get_observation(self):
        # Pad history if not full
        obs = list(self.state_history)
        while len(obs) < self.history_length:
            if obs:
                obs.insert(0, obs[0])
            else:
                zeros = np.zeros(self.observation_space.shape[1], dtype=np.float32)
                obs.append(zeros)
        return np.array(obs)
    
    def _update_state_history(self):
        if self.order_book.get_mid_price() is not None:
            state = self._build_state_features()
            self.state_history.append(state)

    def _get_info(self):
        mid_price = self.order_book.get_mid_price()

        pnl = (self.cash - self.init_cash) + self.inventory * (mid_price or 0)

        # Calculate adverse selection cost and ratio
        avg_adverse_selection_cost = np.mean(self.adverse_selection_costs) if self.adverse_selection_costs else 0.0
        num_adverse_trades = sum(1 for cost in self.adverse_selection_costs if cost > 0)
        adverse_selection_ratio = num_adverse_trades / max(1, self.total_orders_filled)

        return {
            "pnl": pnl,
            "cash": self.cash,
            "inventory": self.inventory,
            "mid_price": mid_price,
            "num_open_orders": len(self.agent_orders),
            "fill_ratio": self.total_orders_filled / max(1, self.total_orders_placed),
            "maker_volume": self.maker_volume,
            "taker_volume": self.taker_volume,
            "fees_paid": self.fees_paid,
            # FIXED KEY and providing both metrics for clarity
            "adverse_selection_ratio": adverse_selection_ratio,
            "adverse_selection_cost_avg": avg_adverse_selection_cost,
            "timestamp": self.current_timestamp
        }