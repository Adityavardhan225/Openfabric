# order_book.py

import logging
from typing import Dict, List, Tuple, Optional
from collections import deque
from sortedcontainers import SortedDict

# Import the new message types we will be processing
from market_message import SnapshotMessage, DeltaMessage

class OrderBook:
    """
    A high-performance L2 Limit Order Book.
    This implementation is designed to work with snapshot and delta updates.
    It does NOT track individual orders (L3), only aggregated price levels (L2).
    """
    def __init__(self, tick_size: float = 0.01):
        # Use SortedDict for performance. Bids are stored with negative prices
        # to make the 'largest' price the 'first' item (for efficient best_bid lookup).
        self.bids = SortedDict()  # price -> size
        self.asks = SortedDict()  # price -> size
        
        self.tick_size = tick_size
        self.last_trade_price = None # Can be updated by a separate trade feed

    def process_snapshot(self, msg: SnapshotMessage):
        """
        Initializes the book from a full snapshot message. This clears the
        existing book state entirely.
        """
        self.bids.clear()
        self.asks.clear()
        
        for price_str, size_str in msg.bids:
            self.bids[float(price_str)] = float(size_str)
        for price_str, size_str in msg.asks:
            self.asks[float(price_str)] = float(size_str)
        logging.debug(f"Order book initialized from snapshot at ts {msg.timestamp}.")

    def process_delta(self, msg: DeltaMessage):
        """
        Updates the book from an incremental delta message.
        """
        for price_str, size_str in msg.bids:
            price, size = float(price_str), float(size_str)
            if size == 0:
                self.bids.pop(price, None) # Use pop with default to avoid KeyError
            else:
                self.bids[price] = size
                
        for price_str, size_str in msg.asks:
            price, size = float(price_str), float(size_str)
            if size == 0:
                self.asks.pop(price, None)
            else:
                self.asks[price] = size
        logging.debug(f"Order book updated with delta at ts {msg.timestamp}.")

    def get_best_bid(self) -> Optional[float]:
        """Returns the highest bid price."""
        if not self.bids:
            return None
        # The last item in a sorted dict of bids is the highest price
        return self.bids.peekitem(-1)[0]

    def get_best_ask(self) -> Optional[float]:
        """Returns the lowest ask price."""
        if not self.asks:
            return None
        # The first item in a sorted dict of asks is the lowest price
        return self.asks.peekitem(0)[0]

    def get_mid_price(self) -> Optional[float]:
        """Returns the mid-price of the book."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2
        
        # Fallback to last known trade price if the book is one-sided
        if self.last_trade_price is not None:
            return self.last_trade_price
        
        return None

    def get_spread(self) -> Optional[float]:
        """Returns the spread between the best ask and best bid."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid is not None and best_ask is not None:
            return best_ask - best_bid
        return None

    def get_book_levels(self, levels: int) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        Gets the top N levels of the book for observation states.
        This is highly efficient with SortedDict.
        """
        # Get bids (highest prices first)
        # .items() gives a view, [-levels:] slices the last N items.
        num_bids = len(self.bids)
        start_index = max(0, num_bids - levels)
        top_bids = list(self.bids.items())[start_index:]
        top_bids.reverse()  # Reverse to have the best bid (highest price) at index 0

        # Get asks (lowest prices first)
        num_asks = len(self.asks)
        end_index = min(num_asks, levels)
        top_asks = list(self.asks.items())[:end_index]
        
        # Padding logic if not enough levels exist
        if not top_bids:
            # Use the best ask as a reference, or a default price if book is empty
            ref_price = self.get_best_ask() or self.last_trade_price or 1000.0
            last_price = ref_price - self.tick_size
        else:
            last_price = top_bids[-1][0]
            
        while len(top_bids) < levels:
            last_price -= self.tick_size
            top_bids.append((last_price, 0.0))

        if not top_asks:
            # Use the best bid as a reference, or a default price if book is empty
            ref_price = self.get_best_bid() or self.last_trade_price or 1000.0
            last_price = ref_price + self.tick_size
        else:
            last_price = top_asks[-1][0]
            
        while len(top_asks) < levels:
            last_price += self.tick_size
            top_asks.append((last_price, 0.0))
                
        return top_bids, top_asks


    def get_book_depth(self) -> Tuple[float, float]:
        """Returns the total volume on each side of the book."""
        bid_depth = sum(self.bids.values())
        ask_depth = sum(self.asks.values())
        return bid_depth, ask_depth

    def get_order_book_imbalance(self) -> float:
        """Calculates the volume imbalance of the book."""
        bid_volume, ask_volume = self.get_book_depth()
        
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return 0.0
        
        return (bid_volume - ask_volume) / total_volume