
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional

# These imports are now correct based on our updated files
from order_book import OrderBook
from market_message import OrderSide, TradeMessage

class MarketFeatures:
    """Calculate market microstructure features from the L2 order book."""
    def __init__(self, lookback_periods: List[int] = [10, 50, 100]):
        self.lookback_periods = lookback_periods
        # These deques will store (timestamp, value) tuples
        self.mid_prices = deque(maxlen=max(lookback_periods) if lookback_periods else 100)
        self.trades = deque(maxlen=max(lookback_periods) if lookback_periods else 100)
        self.imbalances = deque(maxlen=max(lookback_periods) if lookback_periods else 100)
        
    def update(self, order_book: OrderBook, timestamp: int):
        """Update features based on the current order book state."""
        mid_price = order_book.get_mid_price()
        if mid_price is not None:
            self.mid_prices.append((timestamp, mid_price))
        
        imbalance = order_book.get_order_book_imbalance()
        self.imbalances.append((timestamp, imbalance))
        
    def add_trade(self, trade: TradeMessage):
        """Add a trade to the history for OFI calculation."""
        self.trades.append((trade.timestamp, trade.price, trade.size, trade.aggressor_side))

    # --- MISSING METHODS RESTORED ---

    def get_mid_price_volatility(self) -> Dict[int, float]:
        """Calculate realized volatility over different time periods."""
        result = {}
        for period in self.lookback_periods:
            if len(self.mid_prices) < period:
                result[period] = 0.0
                continue
            
            prices = [price for _, price in list(self.mid_prices)[-period:]]
            if not prices or len(prices) < 2:
                result[period] = 0.0
                continue
                
            returns = np.log(prices[1:]) - np.log(prices[:-1])
            vol = np.std(returns)
            result[period] = vol
            
        return result
    
    def get_order_flow_imbalance(self) -> Dict[int, float]:
        """Calculate order flow imbalance from recent trades."""
        result = {}
        for period in self.lookback_periods:
            recent_trades = [t for t in self.trades if t[0] >= (self.mid_prices[-1][0] - period * 100_000_000)] # Approx lookback
            if not recent_trades:
                result[period] = 0.0
                continue
                
            buy_volume = sum(size for _, _, size, side in recent_trades if side == OrderSide.BUY)
            sell_volume = sum(size for _, _, size, side in recent_trades if side == OrderSide.SELL)
            
            total_volume = buy_volume + sell_volume
            if total_volume == 0:
                result[period] = 0.0
                continue
                
            imbalance = (buy_volume - sell_volume) / total_volume
            result[period] = imbalance
            
        return result
    
    def get_volume_order_imbalance(self) -> Dict[int, float]:
        """Calculate the average order book imbalance over time."""
        result = {}
        for period in self.lookback_periods:
            if len(self.imbalances) < period:
                result[period] = 0.0
                continue
            
            recent_imbalances = [imb for _, imb in list(self.imbalances)[-period:]]
            result[period] = np.mean(recent_imbalances) if recent_imbalances else 0.0
            
        return result

    def get_micro_price(self, order_book: OrderBook) -> Optional[float]:
        """Calculate micro price. This version is L2-compatible."""
        best_bid = order_book.get_best_bid()
        best_ask = order_book.get_best_ask()
        
        if best_bid is None or best_ask is None:
            return order_book.get_mid_price()
            
        bid_volume = order_book.bids.get(best_bid, 0.0)
        ask_volume = order_book.asks.get(best_ask, 0.0)
        
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return (best_bid + best_ask) / 2
            
        micro_price = (best_bid * ask_volume + best_ask * bid_volume) / total_volume
        return micro_price
    
    def get_time_features(self, timestamp: int) -> Dict[str, float]:
        """Calculate cyclical time-based features."""
        seconds_since_midnight = (timestamp / 1_000_000_000) % 86400
        normalized_time = seconds_since_midnight / 86400.0
        
        return {
            "sin_time": np.sin(2 * np.pi * normalized_time),
            "cos_time": np.cos(2 * np.pi * normalized_time)
        }

    def get_all_features(self, order_book: OrderBook, timestamp: int) -> Dict[str, float]:
        """Get all market microstructure features."""
        mid_price = order_book.get_mid_price()
        if mid_price is None: mid_price = order_book.last_trade_price or 0
        
        features = {
            "mid_price": mid_price,
            "spread": order_book.get_spread() or 0,
            "micro_price": self.get_micro_price(order_book) or mid_price,
            "bid_depth": order_book.get_book_depth()[0],
            "ask_depth": order_book.get_book_depth()[1],
        }
        
        volatility = self.get_mid_price_volatility()
        for period, vol in volatility.items():
            features[f"vol_{period}"] = vol
        
        ofi = self.get_order_flow_imbalance()
        for period, imb in ofi.items():
            features[f"ofi_{period}"] = imb
            
        voi = self.get_volume_order_imbalance()
        for period, imb in voi.items():
            features[f"voi_{period}"] = imb
            
        features.update(self.get_time_features(timestamp))
        
        return features