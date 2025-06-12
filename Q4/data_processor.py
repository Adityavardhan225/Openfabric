# data_processor.py

import json
import logging
import os
import numpy as np
from typing import Optional, Dict

from market_message import MarketMessage, SnapshotMessage, DeltaMessage, TradeMessage 

class DataProcessor:
    """Base class for processing market data."""
    def reset(self, options: Optional[Dict] = None):
        raise NotImplementedError
        
    def get_next_message(self) -> Optional[MarketMessage]:
        raise NotImplementedError

class L2BookDataProcessor(DataProcessor):
    """
    High-performance processor for L2 order book data stored as line-delimited JSON.
    Handles 'snapshot' and 'delta' message types.
    """
    def __init__(self, filepath: str):
        self.filepath = filepath
        self._file = None
        logging.info(f"Initialized L2BookDataProcessor for file: {self.filepath}")
        self.reset()
        
    def reset(self, options: Optional[Dict] = None):
        """
        Resets the processor. If 'random_start' is True in options, it seeks to a
        random position in the file. Otherwise, it starts from the beginning.
        """
        if self._file and not self._file.closed:
            self._file.close()
            
        self._file = open(self.filepath, 'r')
        
        # --- MODIFICATION IS HERE ---
        if options and options.get('random_start', False):
            try:
                # Go to the end to get the file size
                self._file.seek(0, os.SEEK_END)
                file_size = self._file.tell()
                
                # Seek to a random position within the first 80% of the file
                # to ensure there's enough data left for an episode.
                random_position = np.random.randint(0, int(file_size * 0.8), dtype=np.int64)
                self._file.seek(random_position)
                
                # Read and discard the potentially partial line to align to the next full line
                self._file.readline()
                logging.info(f"Random start: seeking to ~{random_position / 1e6:.2f}MB in data file.")
                
            except Exception as e:
                logging.warning(f"Could not perform random seek, starting from beginning. Error: {e}")
                self._file.seek(0)
        else:
            # Default behavior: start from the beginning of the file.
            self._file.seek(0)
            logging.info("DataProcessor reset. File opened at the beginning.")

    # ... The rest of the L2BookDataProcessor class remains exactly the same ...
    def _parse_line(self, line: str) -> Optional[MarketMessage]:
        try:
            data = json.loads(line)
            msg_type = data.get('type')
            timestamp = data.get('ts')
            msg_data = data.get('data', {})
            if not all([msg_type, timestamp, msg_data]): return None
            bids = msg_data.get('b', [])
            asks = msg_data.get('a', [])
            if msg_type == 'snapshot':
                return SnapshotMessage(timestamp=timestamp, bids=bids, asks=asks)
            elif msg_type == 'delta':
                return DeltaMessage(timestamp=timestamp, bids=bids, asks=asks)
            return None
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

    def get_next_message(self) -> Optional[MarketMessage]:
        while True:
            line = self._file.readline()
            if not line:
                logging.info("End of data file reached.")
                return None
            message = self._parse_line(line)
            if message:
                return message

    def __del__(self):
        if self._file:
            self._file.close()