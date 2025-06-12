# market_message.py

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Tuple

# MODIFICATION: Added DELTA message type.
class MessageType(Enum):
    """Types of messages in the market data feed"""
    NEW_ORDER = 0      # For internal agent use
    CANCEL_ORDER = 1   # For internal agent use
    TRADE = 2          # For trade data streams
    SNAPSHOT = 3       # For a full L2 order book state
    DELTA = 4          # For an incremental L2 book update
    AGENT_ACTION = 5   # For internal environment use

class OrderSide(Enum):
    """Side of an order"""
    BUY = 0
    SELL = 1

@dataclass
class MarketMessage:
    """Base class for market data messages"""
    timestamp: int  # Nanosecond timestamp
    message_type: MessageType

# --- L2 MESSAGE TYPES ---
# These are the primary types we will read from the data file.

@dataclass
class SnapshotMessage(MarketMessage):
    """
    Message containing a full snapshot of the L2 order book.
    This represents the entire state at a point in time.
    """
    bids: List[Tuple[str, str]]  # List of ["price", "size"]
    asks: List[Tuple[str, str]]

    def __init__(self, timestamp: int, bids: List[Tuple[str, str]], asks: List[Tuple[str, str]]):
        # Convert timestamp from milliseconds (as seen in data) to nanoseconds
        super().__init__(timestamp * 1_000_000, MessageType.SNAPSHOT)
        self.bids = bids
        self.asks = asks

@dataclass
class DeltaMessage(MarketMessage):
    """
    Message containing an incremental update (a delta) to the L2 order book.
    A size of "0" means the price level should be removed.
    """
    bids: List[Tuple[str, str]]
    asks: List[Tuple[str, str]]

    def __init__(self, timestamp: int, bids: List[Tuple[str, str]], asks: List[Tuple[str, str]]):
        # Convert timestamp from milliseconds (as seen in data) to nanoseconds
        super().__init__(timestamp * 1_000_000, MessageType.DELTA)
        self.bids = bids
        self.asks = asks

# --- TRADE MESSAGE ---
# This might come from a separate data stream.
@dataclass
class TradeMessage(MarketMessage):
    """Message for a trade execution."""
    price: float
    size: float
    aggressor_side: OrderSide
    
    def __init__(self, timestamp: int, price: float, size: float, aggressor_side: OrderSide):
        super().__init__(timestamp * 1_000_000, MessageType.TRADE) # Also convert timestamp
        self.price = price
        self.size = size
        self.aggressor_side = aggressor_side


# --- AGENT ACTION CLASSES (Used internally by the Environment) ---
# No changes needed here.
@dataclass
class AgentAction:
    """Action taken by the agent."""
    action_type: str  # 'limit', 'market', 'cancel'
    side: Optional[OrderSide] = None
    price: Optional[float] = None
    size: Optional[float] = None
    order_id: Optional[str] = None

@dataclass
class AgentActionMessage(MarketMessage):
    """Message for an action taken by the agent, to be put in the event queue."""
    action: AgentAction
    execution_time: int  # When the action will be executed (timestamp + latency)
    
    def __init__(self, timestamp: int, action: AgentAction, latency: int):
        super().__init__(timestamp, MessageType.AGENT_ACTION)
        self.action = action
        self.execution_time = timestamp + latency

# --- LEGACY L3 MESSAGE TYPES ---
# These are no longer needed for data processing but can be kept for reference
# or if you ever mix L2/L3 simulations. I'll remove them for clarity.