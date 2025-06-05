import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import matplotlib
import matplotlib.dates as mdates
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
# For advanced technical indicators


class MeanReversionBacktester:

    def __init__(
        self,
        # Signal parameters
        lookback_period: int = 20,
        z_entry: float = 1.5,       
        z_exit: float = 0.5,        
        rsi_lookback: int = 14,
        rsi_entry: float = 35.0,     
        rsi_exit: float = 55.0,     
        
        # ATR parameters for stops and sizing
        atr_lookback: int = 14,
        atr_stop_loss: float = 1.5, 
        atr_take_profit: float = 2.0, 
        
        # Position sizing and risk
        risk_per_trade: float = 0.01, 
        max_position_pct: float = 0.1, 
        
        # VWAP parameters
        vwap_entry_pct: float = 0.003, 
        
        # Filter parameters
        adx_lookback: int = 14,
        adx_threshold: float = 25.0, 
        min_volatility: float = 0.008, 
        max_volatility: float = 0.04, 
        
        # Trade management
        min_holding_period: int = 5,  
        trade_cooldown: int = 15,    
        
        # Execution parameters
        execution_delay: int = 1,     
        slippage_base_pct: float = 0.0001, 
        commission_per_share: float = 0.005, 
        
        # Other settings
        long_only: bool = True
    ):
        """Initialize advanced backtester with research-based parameters."""
        # Signal parameters
        self.lookback_period = lookback_period
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.rsi_lookback = rsi_lookback
        self.rsi_entry = rsi_entry
        self.rsi_exit = rsi_exit
        
        # ATR parameters
        self.atr_lookback = atr_lookback
        self.atr_stop_loss = atr_stop_loss
        self.atr_take_profit = atr_take_profit
        
        # Position sizing
        self.risk_per_trade = risk_per_trade  
        self.max_position_pct = max_position_pct
        
        # VWAP parameters
        self.vwap_entry_pct = vwap_entry_pct
        
        # Filter parameters
        self.adx_lookback = adx_lookback
        self.adx_threshold = adx_threshold
        self.min_volatility = min_volatility
        self.max_volatility = max_volatility
        
        # Trade management
        self.min_holding_period = min_holding_period
        self.trade_cooldown = trade_cooldown
        
        # Execution parameters
        self.execution_delay = execution_delay
        self.slippage_base_pct = slippage_base_pct
        self.commission_per_share = commission_per_share
        
        # Other settings
        self.long_only = long_only

        # Internal storage
        self.data = pd.DataFrame()
        self.results: Dict[str, any] = {}
        self.trades: List[Dict] = []

    def load_data(self, df: pd.DataFrame) -> None:
        """
        Load and validate minute‐level OHLCV DataFrame.
        DataFrame must have columns: ['Open', 'High', 'Low', 'Close', 'Volume']
        """
        required_cols = {"Open", "High", "Low", "Close", "Volume"}
        if not required_cols.issubset(set(df.columns)):
            missing = required_cols - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")

        # Ensure sorted by datetime ascending
        df = df.sort_index()
        self.data = df.copy().loc[~df.index.duplicated(keep="first")]
        
        # Make sure index is proper datetime
        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")
            
        # Add time-of-day columns
        self.data['hour'] = self.data.index.hour
        self.data['minute'] = self.data.index.minute
        
        # Add session day marker (to compute session VWAP)
        self.data['date'] = self.data.index.date
        
        print(f"[INFO] Loaded data: {len(self.data)} rows from {self.data.index[0]} to {self.data.index[-1]}")

    def _compute_indicators(self) -> None:
        """
        Compute all required indicators:
        • MA, STD, Z-score
        • RSI
        • Session VWAP
        • ATR for sizing and stops
        • ADX for trend filter
        """
        df = self.data
        print(f"Before computing indicators: {len(df)} rows")

        # 1) Z-score calculation
        df["MA"] = df["Close"].rolling(window=self.lookback_period, min_periods=1).mean()
        df["STD"] = df["Close"].rolling(window=self.lookback_period, min_periods=1).std()
        # Z-score = (price - MA) / STD
        df["z_score"] = (df["Close"] - df["MA"]) / df["STD"].replace(0, np.nan)
        df["z_score"] = df["z_score"].fillna(0)
        
        # 2) RSI calculation
        delta = df["Close"].diff()
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        avg_gain = pd.Series(gain).rolling(window=self.rsi_lookback, min_periods=1).mean()
        avg_loss = pd.Series(loss).rolling(window=self.rsi_lookback, min_periods=1).mean()
        # Avoid division by zero
        avg_loss = avg_loss.replace(0, 1e-10)
        rs = avg_gain / avg_loss
        df["RSI"] = 100.0 - (100.0 / (1.0 + rs))

        # 3) Session VWAP (resets each day)
        # Group by date and calculate cumulative PV and V
        df["PV"] = df["Close"] * df["Volume"]  
        df["cumPV"] = df.groupby("date")["PV"].cumsum()
        df["cumVol"] = df.groupby("date")["Volume"].cumsum()
        # Avoid division by zero
        df["cumVol"] = df["cumVol"].replace(0, 1e-10)
        df["session_VWAP"] = df["cumPV"] / df["cumVol"]
        
        # 4) ATR calculation
        high_low = df["High"] - df["Low"]
        high_close = np.abs(df["High"] - df["Close"].shift())
        low_close = np.abs(df["Low"] - df["Close"].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df["ATR"] = pd.Series(true_range).rolling(window=self.atr_lookback, min_periods=1).mean()
        
        # 5) ADX calculation (trend filter)
       
        # +DM, -DM
        df["plus_dm"] = df["High"].diff()
        df["minus_dm"] = df["Low"].diff().multiply(-1)
        
        # Set values that don't satisfy DM conditions to 0
        df.loc[(df["plus_dm"] <= 0) | (df["plus_dm"] < df["minus_dm"]), "plus_dm"] = 0.0
        df.loc[(df["minus_dm"] <= 0) | (df["minus_dm"] < df["plus_dm"]), "minus_dm"] = 0.0
        
        # Smoothed TR, +DM, -DM
        df["tr_smooth"] = df["ATR"] * self.adx_lookback  # Using ATR as TR already
        df["plus_dm_smooth"] = df["plus_dm"].rolling(window=self.adx_lookback).sum()
        df["minus_dm_smooth"] = df["minus_dm"].rolling(window=self.adx_lookback).sum()
        
        # +DI, -DI
        df["plus_di"] = 100 * df["plus_dm_smooth"] / df["tr_smooth"].replace(0, 1e-10)
        df["minus_di"] = 100 * df["minus_dm_smooth"] / df["tr_smooth"].replace(0, 1e-10)
        
        # DX and ADX
        df["dx"] = 100 * np.abs(df["plus_di"] - df["minus_di"]) / (df["plus_di"] + df["minus_di"]).replace(0, 1e-10)
        df["ADX"] = df["dx"].rolling(window=self.adx_lookback).mean()
        
        # 6) Minute returns for volatility filter
        df["ret_min"] = df["Close"].pct_change()
        df["rolling_vol_5min"] = df["ret_min"].rolling(window=5).std() * np.sqrt(252 * 390)
        
        # 7) Time-of-day filter (avoid first/last 15 min)
        df["avoid_time"] = ((df["hour"] == 9) & (df["minute"] < 45)) | ((df["hour"] == 15) & (df["minute"] >= 45))
        
        # Fill any remaining NaN values
        fill_columns = ["MA", "STD", "z_score", "RSI", "session_VWAP", "ATR", "ADX"]
        for col in fill_columns:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        self.data = df
        print(f"After computing indicators: {len(self.data)} rows")

    def _generate_raw_signals(self) -> None:
        """
        Build raw 'signal' series (0 or 1) with research-based criteria:
        1. Z-score < -z_entry (price significantly below MA)
        2. RSI < rsi_entry (oversold)
        3. Price < session_VWAP × (1 - vwap_entry_pct) (below VWAP)
        4. ADX < adx_threshold (not strongly trending)
        5. rolling_vol in [min_volatility, max_volatility] (good volatility regime)
        6. Not in avoid_time window (avoid market open/close)
        """
        df = self.data

        # Initialize raw signal
        df["signal_raw"] = 0

        # Entry conditions
        z_score_cond = df["z_score"] < -self.z_entry  # Price below MA by z_entry standard deviations
        rsi_cond = df["RSI"] < self.rsi_entry  # RSI oversold
        vwap_cond = df["Close"] < df["session_VWAP"] * (1 - self.vwap_entry_pct)  # Price below VWAP
        adx_cond = df["ADX"] < self.adx_threshold  # Not strongly trending
        vol_cond = (df["rolling_vol_5min"] > self.min_volatility) & (df["rolling_vol_5min"] < self.max_volatility)  # Good volatility
        time_cond = ~df["avoid_time"]  # Not during market open/close

        # Debug counts
        print(f"Z-score condition met: {z_score_cond.sum()} times")
        print(f"RSI condition met: {rsi_cond.sum()} times")
        print(f"VWAP condition met: {vwap_cond.sum()} times")
        print(f"ADX condition met: {adx_cond.sum()} times")
        print(f"Volatility condition met: {vol_cond.sum()} times")
        print(f"Time condition met: {time_cond.sum()} times")

        # CHANGED: More flexible signal logic - require z_score and either RSI or VWAP condition
        # plus time filter and relaxed trend/vol filters
        entry_cond = (
            z_score_cond &  # We always want price below MA
            (rsi_cond | vwap_cond) &  # Need either RSI oversold OR price below VWAP
            time_cond  # Always avoid market open/close
        )
        
        # Make trend and volatility filters optional if we're not getting enough signals
        if entry_cond.sum() < 5:
            print("WARNING: Few signals with all filters, relaxing trend and volatility filters")
            entry_cond = (
                z_score_cond & 
                (rsi_cond | vwap_cond) &
                time_cond
            )

        # Exit conditions
        z_score_exit = df["z_score"] > self.z_exit  # Price has reverted above exit threshold
        rsi_exit = df["RSI"] > self.rsi_exit  # RSI no longer oversold
        
        exit_cond = z_score_exit | rsi_exit
        
        # Set signal based on conditions
        df.loc[entry_cond, "signal_raw"] = 1
        df.loc[exit_cond, "signal_raw"] = 0

        # Shift raw signal by execution_delay to simulate latency
        df["signal"] = df["signal_raw"].shift(self.execution_delay).fillna(0).astype(int)
        
        # Debug info
        entry_signals = df["signal_raw"].sum()
        print(f"Entry signals generated: {entry_signals}")
        
        self.data = df

    def _apply_position_logic(self) -> None:
        """
        Convert 'signal' into 'final_signal' by enforcing:
          - ATR-based stop-loss and take-profit
          - Minimum holding period
          - Trade cooldown
        Also calculates entry price, exit reason, and other trade details.
        """
        df = self.data.copy()

        n = len(df)
        position = np.zeros(n, dtype=int)  # 1 if in a trade, else 0
        
        # Arrays for trade tracking
        df["entry_price"] = np.nan
        df["stop_price"] = np.nan
        df["target_price"] = np.nan
        df["exit_reason"] = ""
        df["trade_id"] = 0
        
        last_exit_idx = -self.trade_cooldown
        entry_idx = -self.trade_cooldown
        in_position = False
        current_trade_id = 0

        for i in range(n):
            sig = df["signal"].iat[i]
            
            if not in_position:
                # Can we enter?
                if (sig == 1) and ((i - last_exit_idx) >= self.trade_cooldown):
                    # Enter trade
                    entry_idx = i
                    current_trade_id += 1
                    
                    entry_price = df["Close"].iat[i]
                    atr = df["ATR"].iat[i]
                    
                    # Set stop and target based on ATR
                    stop_price = entry_price - (self.atr_stop_loss * atr)
                    target_price = entry_price + (self.atr_take_profit * atr)
                    
                    # Record trade details
                    df.at[df.index[i], "entry_price"] = entry_price
                    df.at[df.index[i], "stop_price"] = stop_price
                    df.at[df.index[i], "target_price"] = target_price
                    df.at[df.index[i], "trade_id"] = current_trade_id
                    
                    position[i] = 1
                    in_position = True
                else:
                    position[i] = 0

            else:
                # We're in a trade - carry forward entry details
                df.at[df.index[i], "entry_price"] = df["entry_price"].iat[i - 1]
                df.at[df.index[i], "stop_price"] = df["stop_price"].iat[i - 1]
                df.at[df.index[i], "target_price"] = df["target_price"].iat[i - 1]
                df.at[df.index[i], "trade_id"] = df["trade_id"].iat[i - 1]
                
                # Check stop-loss
                if df["Low"].iat[i] <= df["stop_price"].iat[i]:
                    # Stop loss hit
                    if (i - entry_idx) >= self.min_holding_period:
                        last_exit_idx = i
                        in_position = False
                        position[i] = 0
                        df.at[df.index[i], "exit_reason"] = "stop_loss"
                    else:
                        # Continue position until minimum hold time
                        position[i] = 1
                
                # Check take-profit
                elif df["High"].iat[i] >= df["target_price"].iat[i]:
                    # Take profit hit
                    if (i - entry_idx) >= self.min_holding_period:
                        last_exit_idx = i
                        in_position = False
                        position[i] = 0
                        df.at[df.index[i], "exit_reason"] = "take_profit"
                    else:
                        # Continue position until minimum hold time
                        position[i] = 1
                
                # Check normal exit conditions (signal = 0 + min hold time passed)
                elif (sig == 0) and ((i - entry_idx) >= self.min_holding_period):
                    last_exit_idx = i
                    in_position = False
                    position[i] = 0
                    df.at[df.index[i], "exit_reason"] = "signal"
                
                # Check z-score/RSI exit conditions
                elif (df["z_score"].iat[i] > self.z_exit or 
                      df["RSI"].iat[i] > self.rsi_exit) and \
                      ((i - entry_idx) >= self.min_holding_period):
                    last_exit_idx = i
                    in_position = False
                    position[i] = 0
                    df.at[df.index[i], "exit_reason"] = "indicator_exit"
                
                else:
                    # Continue holding
                    position[i] = 1

        df["final_signal"] = position
        
        # Count number of trade entries and exits
        num_entries = (df["final_signal"].diff() > 0).sum()
        num_exits = (df["final_signal"].diff() < 0).sum()
        print(f"Trades executed: {num_entries} entries, {num_exits} exits")
        
        # Collect trade information
        self._collect_trades(df)
        
        self.data = df

    def _collect_trades(self, df: pd.DataFrame) -> None:
        """Extract trade information for detailed analysis"""
        self.trades = []
        
        # Find all trade_ids
        trade_ids = df.loc[df["trade_id"] > 0, "trade_id"].unique()
        
        for trade_id in trade_ids:
            trade_data = df[df["trade_id"] == trade_id]
            
            # Skip if not complete (no exit)
            if trade_data["exit_reason"].isna().all():
                continue
            
            # First row = entry
            entry_idx = trade_data.index[0]
            entry_price = trade_data["entry_price"].iloc[0]
            
            # Find exit
            exit_rows = trade_data[trade_data["exit_reason"] != ""]
            if len(exit_rows) == 0:
                continue  # Skip trades with no exit
                
            exit_idx = exit_rows.index[0]
            exit_reason = exit_rows["exit_reason"].iloc[0]
            
            # Get executed prices (with slippage)
            if "executed_entry_price" in trade_data.columns:
                executed_entry_price = trade_data["executed_entry_price"].iloc[0]
                executed_exit_price = exit_rows["executed_exit_price"].iloc[0] \
                    if "executed_exit_price" in exit_rows else exit_rows["Close"].iloc[0]
            else:
                executed_entry_price = entry_price
                executed_exit_price = exit_rows["Close"].iloc[0]
            
            # Calculate trade P&L
            pnl_pct = (executed_exit_price / executed_entry_price) - 1.0
            
            # Get trade duration
            duration = (exit_idx - entry_idx).total_seconds() / 60  # minutes
            
            # Add to trades list
            self.trades.append({
                "trade_id": int(trade_id),
                "entry_time": entry_idx,
                "exit_time": exit_idx,
                "entry_price": entry_price,
                "exit_price": executed_exit_price,
                "pnl_pct": pnl_pct,
                "duration_minutes": duration,
                "exit_reason": exit_reason
            })

    def _calculate_position_size(self) -> None:
        """
        Calculate position size using ATR-based risk management:
        - Risk a fixed % of portfolio per trade (e.g., 1%)
        - Position size = RiskAmount / (ATR * ATR_stop_loss)
        - Cap position at max_position_pct
        """
        df = self.data.copy()
        
        # Fixed portfolio amount for % calculations (normalized to 1.0)
        portfolio_value = 1.0
        
        # Calculate position size based on ATR
        risk_amount = portfolio_value * self.risk_per_trade
        
        # Ensure we don't divide by zero
        df["ATR"] = df["ATR"].replace(0, 1e-10)
        
        # Calculate raw position size (# of shares)
        # We need to use the entry price and ATR at entry, propogated through the trade
        df["position_size_basis"] = np.nan
        
        # For each trade, calculate the position size once at entry and propagate
        for trade_id in df.loc[df["trade_id"] > 0, "trade_id"].unique():
            trade_rows = df["trade_id"] == trade_id
            if trade_rows.any():
                # First row of trade
                first_idx = df.index[trade_rows][0]
                
                # Calculate risk per share (ATR * stop_loss_multiple)
                risk_per_share = df.loc[first_idx, "ATR"] * self.atr_stop_loss
                
                # Position size = risk_amount / risk_per_share
                if risk_per_share > 0:
                    pos_size = risk_amount / risk_per_share
                else:
                    pos_size = 0
                
                # Cap at max position size
                entry_price = df.loc[first_idx, "entry_price"]
                max_shares = (portfolio_value * self.max_position_pct) / entry_price
                pos_size = min(pos_size, max_shares)
                
                # Store and propagate
                df.loc[trade_rows, "position_size_basis"] = pos_size

        # Convert to fraction of portfolio (position_size * price / portfolio = fraction)
        df["position"] = df["final_signal"] * (df["position_size_basis"] * df["Close"] / portfolio_value)
        
        # Fill NaNs with 0
        df["position"] = df["position"].fillna(0)
        
        # Position changes for cost calculation
        df["position_change"] = df["position"].diff().fillna(0)

        self.data = df

    def _apply_execution_costs(self) -> None:
        """
        Apply realistic execution costs:
        1. ATR-based slippage model
        2. Per-share commission
        """
        df = self.data

        # Get median ATR for slippage scaling
        median_atr = df["ATR"].median()
        if median_atr <= 0:
            median_atr = 1e-9  # Avoid division by zero
        
        # Slippage as % of price, scaled by ATR
        df["slippage_factor"] = self.slippage_base_pct * (1.0 + (df["ATR"] / median_atr))
        
        # Identify trade entries and exits
        entries = df["position_change"] > 0
        exits = df["position_change"] < 0
        
        # Calculate executed prices with slippage
        df["executed_price"] = df["Close"].copy()
        
        # For entries: pay more (slippage works against you)
        df.loc[entries, "executed_entry_price"] = df.loc[entries, "Close"] * (1.0 + df.loc[entries, "slippage_factor"])
        
        # For exits: receive less (slippage works against you)
        df.loc[exits, "executed_exit_price"] = df.loc[exits, "Close"] * (1.0 - df.loc[exits, "slippage_factor"])
        
        # Use appropriate price for calculations
        df.loc[entries, "executed_price"] = df.loc[entries, "executed_entry_price"]
        df.loc[exits, "executed_price"] = df.loc[exits, "executed_exit_price"]
        
        # Calculate position in shares (using portfolio_value = 1.0 as base)
        df["position_shares"] = df["position"] / df["Close"]
        df["shares_traded"] = np.abs(df["position_shares"].diff()).fillna(0)
        
        # Calculate transaction costs: commission per share + slippage
        # Commission (flat fee per share)
        df["commission_cost"] = df["shares_traded"] * self.commission_per_share
        
        # Total transaction cost in dollars
        df["transaction_cost"] = df["commission_cost"] + (df["shares_traded"] * df["Close"] * df["slippage_factor"])
        
        self.data = df

    def _calculate_returns(self) -> None:
        """
        Calculate strategy returns with transaction costs:
        - raw_returns: position exposure × price change
        - net_returns: raw_returns minus transaction costs
        - Calculate equity curve and drawdowns
        """
        df = self.data

        # Calculate raw returns based on positions
        df["ret_min"] = df["Close"].pct_change()
        df["ret_raw"] = df["position"].shift(1) * df["ret_min"]

        # Transaction cost impact
        prev_close = df["Close"].shift(1).replace(0, np.nan)
        df["cost_impact"] = (df["transaction_cost"] / prev_close).fillna(0)

        # Net returns after costs
        df["ret_net"] = df["ret_raw"] - df["cost_impact"]

        # Equity curve and drawdowns
        df["cum_returns"] = (1.0 + df["ret_net"].fillna(0)).cumprod()
        df["peak"] = df["cum_returns"].cummax()
        df["drawdown"] = (df["cum_returns"] - df["peak"]) / df["peak"]

        self.data = df

    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """
        Compute comprehensive performance metrics:
        - Total return, CAGR, Sharpe, Sortino, Calmar
        - Max drawdown, recovery time
        - Hit rate, profit factor, avg win/loss
        - Turnover and number of trades
        """
        df = self.data.copy()
        
        # Handle empty dataframe case
        if len(df) == 0 or df["cum_returns"].isna().all():
            print("WARNING: Empty DataFrame or no returns in performance metrics calculation")
            return {
                "total_return": 0.0, "cagr": 0.0, "sharpe": 0.0, "sortino": 0.0,
                "calmar": 0.0, "max_drawdown": 0.0, "hit_rate": 0.0,
                "avg_profit": 0.0, "profit_factor": 0.0, "turnover": 0.0, "num_trades": 0
            }
            
        # Clean returns for calculations
        rets = df["ret_net"].dropna()
        if len(rets) == 0:
            print("WARNING: No valid returns found")
            return {
                "total_return": 0.0, "cagr": 0.0, "sharpe": 0.0, "sortino": 0.0,
                "calmar": 0.0, "max_drawdown": 0.0, "hit_rate": 0.0,
                "avg_profit": 0.0, "profit_factor": 0.0, "turnover": 0.0, "num_trades": 0
            }

        # Time horizon in years
        duration_days = (df.index[-1] - df.index[0]).days
        years = duration_days / 365.25 if duration_days > 0 else 1e-9

        # Total and annualized return
        total_ret = df["cum_returns"].iloc[-1] - 1.0
        cagr = (1.0 + total_ret) ** (1.0 / years) - 1.0 if total_ret > -1 else -1.0

        # Annualization factor for minute data
        minutes_per_year = 252 * 390

        # Sharpe Ratio (risk‐free ~ 0)
        rets_std = rets.std()
        sharpe = (rets.mean() / rets_std * np.sqrt(minutes_per_year)) if rets_std > 0 else 0.0

        # Sortino Ratio
        downside = rets[rets < 0]
        sortino = (rets.mean() / downside.std() * np.sqrt(minutes_per_year)) if (len(downside) > 0 and downside.std() > 0) else 0.0

        # Drawdowns
        max_dd = df["drawdown"].min()
        calmar = (cagr / abs(max_dd)) if abs(max_dd) > 0 else 0.0

        # Trade-level metrics
        if len(self.trades) > 0:
            # Convert trades list to DataFrame for analysis
            trades_df = pd.DataFrame(self.trades)
            
            # Hit rate
            hit_rate = (trades_df["pnl_pct"] > 0).mean()
            
            # Average trade metrics
            avg_profit = trades_df["pnl_pct"].mean()
            avg_win = trades_df.loc[trades_df["pnl_pct"] > 0, "pnl_pct"].mean() if len(trades_df[trades_df["pnl_pct"] > 0]) > 0 else 0
            avg_loss = trades_df.loc[trades_df["pnl_pct"] < 0, "pnl_pct"].mean() if len(trades_df[trades_df["pnl_pct"] < 0]) > 0 else 0
            
            # Profit factor
            gross_profits = trades_df.loc[trades_df["pnl_pct"] > 0, "pnl_pct"].sum()
            gross_losses = abs(trades_df.loc[trades_df["pnl_pct"] < 0, "pnl_pct"].sum())
            profit_factor = gross_profits / gross_losses if gross_losses > 0 else np.inf
            
            # Metrics by exit reason
            stop_loss_rate = (trades_df["exit_reason"] == "stop_loss").mean()
            take_profit_rate = (trades_df["exit_reason"] == "take_profit").mean()
            
            # Average trade duration
            avg_duration = trades_df["duration_minutes"].mean()
            
            # Number of trades
            num_trades = len(trades_df)
        else:
            # No completed trades
            hit_rate = 0.0
            avg_profit = 0.0
            avg_win = 0.0
            avg_loss = 0.0
            profit_factor = 0.0
            stop_loss_rate = 0.0
            take_profit_rate = 0.0
            avg_duration = 0.0
            num_trades = 0

        # Turnover (trading activity)
        turnover = (df["position_change"].abs().sum() / 2.0) / years

        metrics = {
            "total_return": float(total_ret),
            "cagr": float(cagr),
            "sharpe": float(sharpe),
            "sortino": float(sortino),
            "calmar": float(calmar),
            "max_drawdown": float(max_dd),
            "hit_rate": float(hit_rate),
            "avg_profit": float(avg_profit),
            "avg_win": float(avg_win) if avg_win else 0.0,
            "avg_loss": float(avg_loss) if avg_loss else 0.0,
            "profit_factor": float(profit_factor),
            "turnover": float(turnover),
            "num_trades": int(num_trades),
            "stop_loss_rate": float(stop_loss_rate),
            "take_profit_rate": float(take_profit_rate), 
            "avg_duration": float(avg_duration)
        }
        return metrics

    def run_backtest(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Execute the complete backtest:
        1. Load data
        2. Compute all indicators and filters
        3. Generate signals
        4. Apply position logic with stops/targets
        5. Calculate position sizes
        6. Apply execution costs
        7. Calculate performance metrics
        """
        # 1) Load data
        self.load_data(df)

        # 2) Compute indicators: MA, STD, Z-score, RSI, VWAP, ATR, ADX
        self._compute_indicators()

        # 3) Generate signals with filters
        self._generate_raw_signals()

        # 4) Apply position logic with stop-loss & take-profit
        self._apply_position_logic()

        # 5) Calculate ATR-based position sizes
        self._calculate_position_size()

        # 6) Apply slippage and transaction costs
        self._apply_execution_costs()

        # 7) Calculate returns
        self._calculate_returns()

        # 8) Performance metrics
        metrics = self._calculate_performance_metrics()

        # Store results
        self.results = {
            "data": self.data.copy(),
            "equity_curve": self.data["cum_returns"].copy(),
            "returns": self.data["ret_net"].copy(),
            "metrics": metrics,
            "trades": self.trades.copy() if hasattr(self, 'trades') else []
        }
        return self.results


    def show_dashboard(self) -> None:
        """
        Display both performance and trade analytics dashboards in a tabbed interface
        with improved spacing and visual appeal.
        """
        if not self.results:
            raise ValueError("No backtest results to plot. Run the backtest first.")

        # Import required libraries
        import tkinter as tk
        from tkinter import ttk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

        # Create main window
        root = tk.Tk()
        root.title("Backtesting Results Dashboard")
        root.state('zoomed')  # Maximize window
        
        # Set background color
        root.configure(bg='#f5f5f5')
        
        # Create title frame
        title_frame = tk.Frame(root, bg='#f5f5f5')
        title_frame.pack(fill='x', padx=10, pady=5)
        
        # Add title label
        title_label = tk.Label(
            title_frame, 
            text="Trading Strategy Dashboard", 
            font=('Arial', 16, 'bold'),
            bg='#f5f5f5'
        )
        title_label.pack(pady=5)
        
        # Create notebook (tabs)
        style = ttk.Style()
        style.configure('TNotebook', background='#f5f5f5')
        style.configure('TNotebook.Tab', padding=[12, 6], font=('Arial', 10))
        
        notebook = ttk.Notebook(root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create frames for each tab
        tab1 = ttk.Frame(notebook)
        tab2 = ttk.Frame(notebook)
        
        notebook.add(tab1, text="Performance Dashboard")
        notebook.add(tab2, text="Trade Analytics")
        
        # Create performance dashboard figure
        perf_fig = self._create_performance_dashboard_fig()
        
        # Create trade analytics dashboard figure (if trades exist)
        trade_fig = None
        if hasattr(self, 'trades') and len(self.trades) > 0:
            trade_fig = self._create_trade_analytics_dashboard_fig()
        
        # Display performance dashboard in tab1
        canvas1 = FigureCanvasTkAgg(perf_fig, tab1)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar for performance dashboard
        toolbar1 = NavigationToolbar2Tk(canvas1, tab1)
        toolbar1.update()
        
        # Display trade analytics in tab2 if available
        if trade_fig:
            canvas2 = FigureCanvasTkAgg(trade_fig, tab2)
            canvas2.draw()
            canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add toolbar for trade analytics
            toolbar2 = NavigationToolbar2Tk(canvas2, tab2)
            toolbar2.update()
        else:
            # Show message if no trades
            no_trades_label = tk.Label(
                tab2, 
                text="No trades executed during this backtest period.", 
                font=('Arial', 14)
            )
            no_trades_label.pack(pady=50)
        
        # Add status bar
        status_bar = tk.Frame(root, bg='#e0e0e0', height=25)
        status_bar.pack(fill='x', side='bottom')
        
        status_text = tk.Label(
            status_bar, 
            text="© 2023 Backtesting Dashboard", 
            bg='#e0e0e0',
            font=('Arial', 8)
        )
        status_text.pack(side='right', padx=10)
        
        # Run the app
        root.mainloop()

    def _create_performance_dashboard_fig(self):
        """Creates performance dashboard figure with card-style visuals and improved spacing"""
        df = self.results["data"]
        metrics = self.results["metrics"]

        # Set nicer style with more readable fonts
        plt.style.use('ggplot')
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['font.size'] = 10
        
        # Create figure with white background for cleaner card appearance
        fig = plt.figure(figsize=(18, 20), facecolor='white')
        fig.suptitle('STRATEGY PERFORMANCE DASHBOARD', fontsize=24, y=0.95, fontweight='bold', 
                    fontfamily='monospace', color='#303030')
        
        # Increased spacing between plots
        plt.subplots_adjust(hspace=0.60, wspace=0.35, top=0.92, bottom=0.05, left=0.08, right=0.92)
        
        # Create grid with better proportions - keeping same structure
        gs = fig.add_gridspec(4, 2, height_ratios=[2, 1, 2, 2])
        
        # Card style parameters to be applied to all plots
        card_style = {
            'facecolor': '#f8f9fa', 
            'edgecolor': '#dee2e6', 
            'boxstyle': 'round,pad=1',
            'linewidth': 2,
            'alpha': 0.95
        }
        
        # ===== 1. EQUITY CURVE WITH BENCHMARK =====
        ax_equity = fig.add_subplot(gs[0, :])
        # Add card-like background
        ax_equity.patch.set_facecolor('#f8f9fa')
        for spine in ax_equity.spines.values():
            spine.set_visible(True)
            spine.set_color('#dee2e6')
            spine.set_linewidth(2)
        
        ax_equity.plot(df.index, df["cum_returns"], linewidth=2.5, color="#1f77b4", label="Strategy")
        
        # Add benchmark if available
        if "Close" in df.columns:
            benchmark_returns = df["Close"].pct_change()
            benchmark_curve = (1 + benchmark_returns.fillna(0)).cumprod()
            ax_equity.plot(df.index, benchmark_curve, linewidth=1.5, color="#ff7f0e", 
                        linestyle='--', alpha=0.7, label="Buy & Hold")
        
        ax_equity.set_title("Equity Growth", fontsize=18, pad=30, fontweight='bold', color='#303030')
        ax_equity.set_ylabel("Growth of $1", fontsize=14)
        ax_equity.grid(True, alpha=0.3, linestyle='--')
        ax_equity.legend(fontsize=12, loc='upper left')
        ax_equity.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
        
        # Create separate metrics text box - MOVED FURTHER DOWN to avoid overlap
        metrics_box = plt.axes([0.70, 0.70, 0.23, 0.13], frameon=True)
        metrics_box.axis('off')
        
        key_metrics_text = (
            f"PERFORMANCE METRICS\n"
            f"───────────────────────\n"
            f"CAGR: {metrics['cagr']:.2%}\n"
            f"Sharpe Ratio: {metrics['sharpe']:.2f}\n"
            f"Max Drawdown: {metrics['max_drawdown']:.2%}\n"
            f"Hit Rate: {metrics['hit_rate']:.2%}\n"
            f"Turnover: {metrics['turnover']:.2f}x"
        )
        
        metrics_box.text(
            0.05, 0.95, key_metrics_text,
            transform=metrics_box.transAxes, fontsize=13,
            bbox=dict(facecolor="#f8f9fa", edgecolor="#dee2e6", boxstyle="round,pad=0.8", alpha=0.95),
            verticalalignment="top", family="monospace", fontweight="bold"
        )
        
        # ===== 2. PERFORMANCE METRICS TABLE =====
        ax_table = fig.add_subplot(gs[2, 0])
        ax_table.axis('off')
        # Add card-like background to the table area
        ax_table.patch.set_facecolor('#f8f9fa')
        for spine in ax_table.spines.values():
            spine.set_visible(True)
            spine.set_color('#dee2e6')
            spine.set_linewidth(2)
        
        # Create table data
        table_data = [
            ["Metric", "Value"],
            ["Total Return", f"{metrics['total_return']:.2%}"],
            ["CAGR", f"{metrics['cagr']:.2%}"],
            ["Sharpe Ratio", f"{metrics['sharpe']:.2f}"],
            ["Sortino Ratio", f"{metrics['sortino']:.2f}"],
            ["Calmar Ratio", f"{metrics['calmar']:.2f}"],
            ["Max Drawdown", f"{metrics['max_drawdown']:.2%}"],
            ["Volatility", f"{df['ret_net'].std() * np.sqrt(252*390):.2%}"]
        ]
        
        table = ax_table.table(
            cellText=table_data, 
            loc='center', 
            cellLoc='center',
            colWidths=[0.4, 0.4],
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 2)
        
        # Add header style
        for key, cell in table.get_celld().items():
            if key[0] == 0:  # Header row
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#1f77b4')
            else:
                cell.set_facecolor('#f8f9fa' if key[0] % 2 == 0 else '#e9ecef')
                cell.set_edgecolor('#dee2e6')  # Add border to cells
        
        ax_table.set_title("Risk & Return Metrics", fontsize=18, pad=30, fontweight='bold', color='#303030')
        
        # ===== 3. TRADE STATISTICS =====
        ax_trade_stats = fig.add_subplot(gs[2, 1])
        ax_trade_stats.axis('off')
        # Add card-like background to the table area
        ax_trade_stats.patch.set_facecolor('#f8f9fa')
        for spine in ax_trade_stats.spines.values():
            spine.set_visible(True)
            spine.set_color('#dee2e6')
            spine.set_linewidth(2)
        
        # Create trade stats table
        trade_table_data = [
            ["Metric", "Value"],
            ["Number of Trades", f"{metrics['num_trades']}"],
            ["Win Rate", f"{metrics['hit_rate']:.2%}"],
            ["Profit Factor", f"{metrics['profit_factor']:.2f}"],
            ["Avg Profit per Trade", f"{metrics['avg_profit']:.4%}"],
            ["Avg Winner", f"{metrics.get('avg_win', 0):.4%}"],
            ["Avg Loser", f"{metrics.get('avg_loss', 0):.4%}"],
            ["Turnover", f"{metrics['turnover']:.2f}x"]
        ]
        
        trade_table = ax_trade_stats.table(
            cellText=trade_table_data, 
            loc='center', 
            cellLoc='center',
            colWidths=[0.4, 0.4]
        )
        
        # Style the trade table
        trade_table.auto_set_font_size(False)
        trade_table.set_fontsize(12)
        trade_table.scale(1.2, 2)
        
        # Add header style
        for key, cell in trade_table.get_celld().items():
            if key[0] == 0:  # Header row
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#2ca02c')
            else:
                cell.set_facecolor('#f8f9fa' if key[0] % 2 == 0 else '#e9ecef')
                cell.set_edgecolor('#dee2e6')  # Add border to cells
        
        ax_trade_stats.set_title("Trade Statistics", fontsize=18, pad=30, fontweight='bold', color='#303030')
        
        # ===== 4. ROLLING SHARPE RATIO =====
        ax_rolling = fig.add_subplot(gs[3, 1])
        # Add card-like background
        ax_rolling.patch.set_facecolor('#f8f9fa')
        for spine in ax_rolling.spines.values():
            spine.set_visible(True)
            spine.set_color('#dee2e6')
            spine.set_linewidth(2)
        
        # Calculate rolling metrics (30-period window)
        window = min(len(df) // 5, 390)  # ~1 trading day for minute data
        if window < 10:
            window = 10  # Minimum window size
        
        # Calculate rolling metrics if we have enough data
        if len(df) > window:
            roll_ret = df['ret_net'].fillna(0)
            roll_vol = roll_ret.rolling(window=window).std() * np.sqrt(252*390)  # Annualized
            
            # Replace zeros in volatility to avoid division by zero
            roll_vol = roll_vol.replace(0, np.nan)
            
            # Calculate rolling Sharpe
            roll_sharpe = (roll_ret.mean() * 252 * 390) / roll_vol
            
            ax_rolling.plot(df.index, roll_sharpe, linewidth=2, color='#ff7f0e')
            ax_rolling.axhline(y=1, color='#d62728', linestyle='--', alpha=0.7, label="Sharpe = 1")
            ax_rolling.axhline(y=2, color='#2ca02c', linestyle='--', alpha=0.7, label="Sharpe = 2")
            ax_rolling.set_title("Rolling Sharpe Ratio", fontsize=18, pad=30, fontweight='bold', color='#303030')
            ax_rolling.set_ylabel("Sharpe Ratio", fontsize=14)
            ax_rolling.legend(fontsize=10)
            ax_rolling.grid(True, alpha=0.3, linestyle='--')
            ax_rolling.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
        else:
            ax_rolling.text(0.5, 0.5, "Insufficient data for rolling calculations",
                        ha='center', va='center', fontsize=12)
            ax_rolling.set_title("Rolling Sharpe Ratio", fontsize=18, pad=30, fontweight='bold', color='#303030')

        # Add an empty plot with card styling for the bottom left position to maintain visual balance
        ax_empty = fig.add_subplot(gs[3, 0])
        ax_empty.axis('off')
        ax_empty.patch.set_facecolor('#f8f9fa')
        for spine in ax_empty.spines.values():
            spine.set_visible(True)
            spine.set_color('#dee2e6')
            spine.set_linewidth(2)

        # Add footer signature
        fig.text(0.98, 0.01, "Backtester v1.0", fontsize=8, color='#999999', ha='right')
        
        return fig

    def _create_trade_analytics_dashboard_fig(self):
        """Creates trade analytics dashboard figure with a clean, modern, card-like design."""
        trades_df = pd.DataFrame(self.trades)

        # --- Style Definitions ---
        FIG_BG_COLOR = '#f0f3f6'
        PLOT_BG_COLOR = 'white'
        PLOT_BORDER_COLOR = '#cccccc' # Slightly lighter border
        TITLE_COLOR = '#333333'
        LABEL_COLOR = '#555555'
        TICK_COLOR = '#777777'
        GRID_COLOR = '#e0e0e0'

        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['font.size'] = 9
        plt.rcParams['axes.labelcolor'] = LABEL_COLOR
        plt.rcParams['xtick.color'] = TICK_COLOR
        plt.rcParams['ytick.color'] = TICK_COLOR
        plt.rcParams['axes.titlepad'] = 18 # Increased default title padding
        plt.rcParams['axes.labelpad'] = 12 # Increased default label padding

        COLOR_PNL_POSITIVE = '#27ae60'
        COLOR_PNL_NEGATIVE = '#e74c3c'
        COLOR_PRIMARY_LINE = '#3498db'
        COLOR_SECONDARY_LINE = '#8e44ad' # Slightly different purple for KDE
        COLOR_TEXT_N_COUNT = '#666666'

        fig = plt.figure(figsize=(18, 24)) # Increased height for better vertical spacing
        fig.patch.set_facecolor(FIG_BG_COLOR)
        fig.suptitle('Trade Analytics Dashboard', fontsize=26, color=TITLE_COLOR, y=0.98, fontweight='bold')

        # Adjusted subplots_adjust for more vertical breathing room
        plt.subplots_adjust(hspace=0.75, wspace=0.25, top=0.93, bottom=0.04, left=0.07, right=0.95)
        
        
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.2])


        def style_plot(ax, title, title_fontsize=16): # Slightly larger default title
            ax.set_facecolor(PLOT_BG_COLOR)
            ax.patch.set_edgecolor(PLOT_BORDER_COLOR)
            ax.patch.set_linewidth(1.0) # Slightly thicker border for card effect
            ax.set_title(title, fontsize=title_fontsize, color=TITLE_COLOR, loc='left', fontweight='bold')
            ax.grid(True, color=GRID_COLOR, linestyle='--', linewidth=0.7, alpha=0.6)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color(PLOT_BORDER_COLOR)
            ax.spines['bottom'].set_color(PLOT_BORDER_COLOR)
            ax.spines['left'].set_linewidth(0.8)
            ax.spines['bottom'].set_linewidth(0.8)
            ax.tick_params(axis='both', which='major', labelsize=9.5, colors=TICK_COLOR, pad=6) # Increased tick label size and pad
            ax.xaxis.label.set_fontsize(11.5) # Increased axis label size
            ax.yaxis.label.set_fontsize(11.5)


        # ===== 1. TRADE P&L WATERFALL CHART (Row 0, Col 0, Span 2 cols) =====
        ax_waterfall = fig.add_subplot(gs[0, :]) # Spans both columns
        style_plot(ax_waterfall, "Cumulative Trade P&L")
        
        trades_df['pnl_pct_display'] = trades_df['pnl_pct'] * 100
        cumulative_pnl = np.cumsum(trades_df['pnl_pct_display'])
        
        ax_waterfall.plot(range(len(cumulative_pnl)), cumulative_pnl, 
                        color=COLOR_PRIMARY_LINE, linewidth=2.2, marker='o', markersize=4.5,
                        label="Cumulative P&L")
        
        for i in range(len(trades_df)):
            pnl = trades_df['pnl_pct_display'].iloc[i]
            y_start = 0 if i == 0 else cumulative_pnl[i-1]
            y_end = cumulative_pnl[i]
            color = COLOR_PNL_POSITIVE if pnl > 0 else COLOR_PNL_NEGATIVE
            ax_waterfall.plot([i, i], [y_start, y_end], color=color, linewidth=3.5)
        
        ax_waterfall.set_xlabel("Trade Number")
        ax_waterfall.set_ylabel("Cumulative Return (%)")
        ax_waterfall.axhline(y=0, color=PLOT_BORDER_COLOR, linestyle='-', linewidth=1.2, alpha=0.9)
        ax_waterfall.legend(loc='upper left', fontsize=9.5, frameon=True, facecolor=PLOT_BG_COLOR, edgecolor=PLOT_BORDER_COLOR, framealpha=0.9)

        # --- STATS BOX (as a "card" overlay) ---
        stats_ax_left = 0.70
        stats_ax_bottom = 0.85 # Relative to figure, adjust based on gs[0,:]'s actual top boundary
        stats_ax_width = 0.25
        stats_ax_height = 0.060

        stats_box_ax = plt.axes([stats_ax_left, stats_ax_bottom, stats_ax_width, stats_ax_height], frameon=True)
        stats_box_ax.set_facecolor(PLOT_BG_COLOR)
        stats_box_ax.patch.set_edgecolor(PLOT_BORDER_COLOR)
        stats_box_ax.patch.set_linewidth(1.0)
        stats_box_ax.axis('off')

        win_count = sum(trades_df['pnl_pct'] > 0)
        loss_count = sum(trades_df['pnl_pct'] <= 0)
        total_trades = len(trades_df)
        
        losers_text = "Losers: N/A"
        largest_win_text = "Largest Win: N/A"
        largest_loss_text = "Largest Loss: N/A"
        if total_trades > 0:
            losers_text = f"Losers: {loss_count} ({loss_count/total_trades:.1%})"
            if not trades_df['pnl_pct_display'].empty:
                largest_win_text = f"Largest Win: {trades_df['pnl_pct_display'].max():.2f}%"
                largest_loss_text = f"Largest Loss: {trades_df['pnl_pct_display'].min():.2f}%"

        stats_display_text = f"{losers_text}\n{largest_win_text}\n{largest_loss_text}"
        
        stats_box_ax.text(0.05, 0.90, stats_display_text,
            transform=stats_box_ax.transAxes, fontsize=9, color=LABEL_COLOR,
            verticalalignment="top", family="monospace", linespacing=1.4)

        # ===== 2. PROFIT DISTRIBUTION (Row 1, Col 0) =====
        ax_dist = fig.add_subplot(gs[1, 0])
        style_plot(ax_dist, "Trade Profit Distribution")
        
        if not trades_df.empty and 'pnl_pct_display' in trades_df.columns:
            sns.histplot(trades_df['pnl_pct_display'], bins=15, ax=ax_dist, 
                        color=COLOR_PRIMARY_LINE, stat="density", 
                        edgecolor=PLOT_BG_COLOR, linewidth=0.5, alpha=0.75)

            sns.kdeplot(trades_df['pnl_pct_display'], ax=ax_dist,
                        color=COLOR_SECONDARY_LINE, linewidth=1.8, 
                        warn_singular=False)
            
            mean_pnl = trades_df['pnl_pct_display'].mean()
            ax_dist.axvline(x=mean_pnl, color=COLOR_PNL_NEGATIVE, linestyle='-', linewidth=1.5, alpha=0.9,
                        label=f"Mean: {mean_pnl:.2f}%")
            ax_dist.legend(fontsize=9.5, frameon=True, facecolor=PLOT_BG_COLOR, edgecolor=PLOT_BORDER_COLOR, framealpha=0.9)
        else:
            ax_dist.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=10, color=LABEL_COLOR)

        ax_dist.set_xlabel("Profit (%)")
        ax_dist.set_ylabel("Density")
        ax_dist.axvline(x=0, color=PLOT_BORDER_COLOR, linestyle='--', linewidth=1.0, alpha=0.8)
        
        # ===== 3. PROFIT BY EXIT REASON (Row 1, Col 1) =====
        ax_exit = fig.add_subplot(gs[1, 1])
        style_plot(ax_exit, "Average Profit by Exit Reason")
        
        if 'exit_reason' in trades_df.columns and not trades_df.empty:
            exit_stats_agg = trades_df.groupby('exit_reason').agg(
                mean_pnl_display=('pnl_pct_display', 'mean'),
                count=('pnl_pct_display', 'count')
            ).reset_index().sort_values('count', ascending=False)
            
            bars = ax_exit.bar(exit_stats_agg['exit_reason'], exit_stats_agg['mean_pnl_display'], 
                               alpha=0.85, width=0.7)
            
            for i, bar in enumerate(bars):
                height = bar.get_height()
                count = exit_stats_agg['count'].iloc[i]
                bar.set_color(COLOR_PNL_POSITIVE if height > 0 else COLOR_PNL_NEGATIVE)
                
                y_range = ax_exit.get_ylim()[1] - ax_exit.get_ylim()[0] if ax_exit.get_ylim()[1] != ax_exit.get_ylim()[0] else 1
                offset = y_range * 0.04 # Increased offset
                y_pos = height + offset if height >= 0 else height - offset
                
                ax_exit.text(bar.get_x() + bar.get_width()/2., y_pos, f'n={count}',
                    ha='center', va='bottom' if height >=0 else 'top', fontsize=8.5, color=COLOR_TEXT_N_COUNT)
            
            plt.setp(ax_exit.get_xticklabels(), rotation=30, ha='right', fontsize=9)
        else:
            ax_exit.text(0.5, 0.5, "No exit reason data", ha='center', va='center', fontsize=10, color=LABEL_COLOR)

        ax_exit.set_xlabel("Exit Reason")
        ax_exit.set_ylabel("Average Profit (%)")
        ax_exit.axhline(y=0, color=PLOT_BORDER_COLOR, linestyle='-', linewidth=1.2, alpha=0.9)
        
        # 4. TRADE DURATION ANALYSIS (Row 2, Col 0) 
        ax_duration = fig.add_subplot(gs[2, 0]) # Changed from gs[2,0]
        style_plot(ax_duration, "Trade Duration Distribution")
        
        if 'duration_minutes' in trades_df.columns and not trades_df.empty:
            sns.histplot(trades_df['duration_minutes'], bins=15, kde=False, ax=ax_duration,
                        color=COLOR_SECONDARY_LINE, element="step", fill=True, alpha=0.65,
                        edgecolor=PLOT_BG_COLOR, linewidth=0.5)
            
            mean_duration = trades_df['duration_minutes'].mean()
            ax_duration.axvline(x=mean_duration, color=COLOR_PNL_NEGATIVE, linestyle='-', linewidth=1.5, alpha=0.9,
                            label=f"Mean: {mean_duration:.1f} min")
            ax_duration.legend(fontsize=9.5, frameon=True, facecolor=PLOT_BG_COLOR, edgecolor=PLOT_BORDER_COLOR, framealpha=0.9)
        else:
            ax_duration.text(0.5, 0.5, "No duration data", ha='center', va='center', fontsize=10, color=LABEL_COLOR)

        ax_duration.set_xlabel("Duration (minutes)")
        ax_duration.set_ylabel("Number of Trades")
            
        # ===== 5. TIME OF DAY ANALYSIS (Row 2, Col 1) =====
        ax_time = fig.add_subplot(gs[2, 1]) # Changed from gs[3,0]
        style_plot(ax_time, "Average Profit by Hour of Day")
        
        if 'entry_time' in trades_df.columns and not trades_df.empty:
            if not pd.api.types.is_datetime64_any_dtype(trades_df['entry_time']):
                 trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
            
            trades_df['hour'] = trades_df['entry_time'].dt.hour
            hour_stats_agg = trades_df.groupby('hour').agg(
                mean_pnl_display=('pnl_pct_display', 'mean'),
                count=('pnl_pct_display', 'count')
            ).reindex(range(24)).reset_index()
            
            bars = ax_time.bar(hour_stats_agg['hour'], hour_stats_agg['mean_pnl_display'].fillna(0), 
                               alpha=0.85, width=0.85)
            
            for i, bar in enumerate(bars):
                height = bar.get_height()
                count_val = hour_stats_agg['count'].iloc[i]
                if pd.isna(count_val) or count_val == 0 :
                    bar.set_color('#e0e0e0')
                    continue
                
                bar.set_color(COLOR_PNL_POSITIVE if height > 0 else (COLOR_PNL_NEGATIVE if height < 0 else '#bdc3c7'))
                
                y_range = ax_time.get_ylim()[1] - ax_time.get_ylim()[0] if ax_time.get_ylim()[1] != ax_time.get_ylim()[0] else 1
                offset = y_range * 0.04 # Increased offset
                y_pos = height + offset if height >=0 else height - offset

                ax_time.text(bar.get_x() + bar.get_width()/2., y_pos, f'n={int(count_val)}',
                    ha='center', va='bottom' if height >=0 else 'top', fontsize=8, color=COLOR_TEXT_N_COUNT)
            
            ax_time.set_xticks(range(0, 24, 2))
            ax_time.axvspan(8.75, 15.25, color=COLOR_PRIMARY_LINE, alpha=0.07, zorder=0) # More subtle shading
        else:
            ax_time.text(0.5, 0.5, "No entry time data", ha='center', va='center', fontsize=10, color=LABEL_COLOR)

        ax_time.set_xlabel("Hour (24h format)")
        ax_time.set_ylabel("Average Profit (%)")
        ax_time.axhline(y=0, color=PLOT_BORDER_COLOR, linestyle='-', linewidth=1.2, alpha=0.9)

     
        fig.clf() # Clear previous figure setup if any due to gs change
        gs = fig.add_gridspec(4, 2, height_ratios=[1.2, 1, 1, 1.5])
        plt.subplots_adjust(hspace=0.8, wspace=0.28, top=0.93, bottom=0.04, left=0.07, right=0.95) # More hspace for 4 rows

        # Re-plot with new gs
        ax_waterfall = fig.add_subplot(gs[0, :]); style_plot(ax_waterfall, "Cumulative Trade P&L")
        # ... (P&L plotting code from above) ...
        trades_df['pnl_pct_display'] = trades_df['pnl_pct'] * 100
        cumulative_pnl = np.cumsum(trades_df['pnl_pct_display'])
        ax_waterfall.plot(range(len(cumulative_pnl)), cumulative_pnl, color=COLOR_PRIMARY_LINE, linewidth=2.2, marker='o', markersize=4.5, label="Cumulative P&L")
        for i in range(len(trades_df)):
            pnl = trades_df['pnl_pct_display'].iloc[i]
            y_start = 0 if i == 0 else cumulative_pnl[i-1]; y_end = cumulative_pnl[i]
            color = COLOR_PNL_POSITIVE if pnl > 0 else COLOR_PNL_NEGATIVE
            ax_waterfall.plot([i, i], [y_start, y_end], color=color, linewidth=3.5)
        ax_waterfall.set_xlabel("Trade Number"); ax_waterfall.set_ylabel("Cumulative Return (%)")
        ax_waterfall.axhline(y=0, color=PLOT_BORDER_COLOR, linestyle='-', linewidth=1.2, alpha=0.9)
        ax_waterfall.legend(loc='upper left', fontsize=9.5, frameon=True, facecolor=PLOT_BG_COLOR, edgecolor=PLOT_BORDER_COLOR, framealpha=0.9)
        # Re-add stats box in the context of the new waterfall plot
        stats_box_ax_new = plt.axes([stats_ax_left, 0.82, stats_ax_width, stats_ax_height], frameon=True) # Adjusted y for 4 rows
        stats_box_ax_new.set_facecolor(PLOT_BG_COLOR); stats_box_ax_new.patch.set_edgecolor(PLOT_BORDER_COLOR); stats_box_ax_new.patch.set_linewidth(1.0); stats_box_ax_new.axis('off')
        stats_box_ax_new.text(0.05, 0.90, stats_display_text, transform=stats_box_ax_new.transAxes, fontsize=9, color=LABEL_COLOR, verticalalignment="top", family="monospace", linespacing=1.4)


        ax_dist = fig.add_subplot(gs[1, 0]); style_plot(ax_dist, "Trade Profit Distribution")
        # ... (Profit Distribution plotting code from above) ...
        if not trades_df.empty and 'pnl_pct_display' in trades_df.columns:
            sns.histplot(trades_df['pnl_pct_display'], bins=15, ax=ax_dist, color=COLOR_PRIMARY_LINE, stat="density", edgecolor=PLOT_BG_COLOR, linewidth=0.5, alpha=0.75)
            sns.kdeplot(trades_df['pnl_pct_display'], ax=ax_dist,color=COLOR_SECONDARY_LINE, linewidth=1.8, warn_singular=False)
            mean_pnl = trades_df['pnl_pct_display'].mean()
            ax_dist.axvline(x=mean_pnl, color=COLOR_PNL_NEGATIVE, linestyle='-', linewidth=1.5, alpha=0.9, label=f"Mean: {mean_pnl:.2f}%")
            ax_dist.legend(fontsize=9.5, frameon=True, facecolor=PLOT_BG_COLOR, edgecolor=PLOT_BORDER_COLOR, framealpha=0.9)
        else: ax_dist.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=10, color=LABEL_COLOR)
        ax_dist.set_xlabel("Profit (%)"); ax_dist.set_ylabel("Density")
        ax_dist.axvline(x=0, color=PLOT_BORDER_COLOR, linestyle='--', linewidth=1.0, alpha=0.8)

        ax_exit = fig.add_subplot(gs[1, 1]); style_plot(ax_exit, "Average Profit by Exit Reason")
        # ... (Profit by Exit Reason plotting code from above) ...
        if 'exit_reason' in trades_df.columns and not trades_df.empty:
            exit_stats_agg = trades_df.groupby('exit_reason').agg(mean_pnl_display=('pnl_pct_display', 'mean'), count=('pnl_pct_display', 'count')).reset_index().sort_values('count', ascending=False)
            bars = ax_exit.bar(exit_stats_agg['exit_reason'], exit_stats_agg['mean_pnl_display'], alpha=0.85, width=0.7)
            for i, bar in enumerate(bars):
                height = bar.get_height(); count = exit_stats_agg['count'].iloc[i]
                bar.set_color(COLOR_PNL_POSITIVE if height > 0 else COLOR_PNL_NEGATIVE)
                y_range = ax_exit.get_ylim()[1] - ax_exit.get_ylim()[0] if ax_exit.get_ylim()[1] != ax_exit.get_ylim()[0] else 1; offset = y_range * 0.04
                y_pos = height + offset if height >= 0 else height - offset
                ax_exit.text(bar.get_x() + bar.get_width()/2., y_pos, f'n={count}', ha='center', va='bottom' if height >=0 else 'top', fontsize=8.5, color=COLOR_TEXT_N_COUNT)
            plt.setp(ax_exit.get_xticklabels(), rotation=30, ha='right', fontsize=9)
        else: ax_exit.text(0.5, 0.5, "No exit reason data", ha='center', va='center', fontsize=10, color=LABEL_COLOR)
        ax_exit.set_xlabel("Exit Reason"); ax_exit.set_ylabel("Average Profit (%)")
        ax_exit.axhline(y=0, color=PLOT_BORDER_COLOR, linestyle='-', linewidth=1.2, alpha=0.9)


        ax_duration = fig.add_subplot(gs[2, 0]); style_plot(ax_duration, "Trade Duration Distribution")
        # ... (Trade Duration plotting code from above) ...
        if 'duration_minutes' in trades_df.columns and not trades_df.empty:
            sns.histplot(trades_df['duration_minutes'], bins=15, kde=False, ax=ax_duration, color=COLOR_SECONDARY_LINE, element="step", fill=True, alpha=0.65, edgecolor=PLOT_BG_COLOR, linewidth=0.5)
            mean_duration = trades_df['duration_minutes'].mean()
            ax_duration.axvline(x=mean_duration, color=COLOR_PNL_NEGATIVE, linestyle='-', linewidth=1.5, alpha=0.9, label=f"Mean: {mean_duration:.1f} min")
            ax_duration.legend(fontsize=9.5, frameon=True, facecolor=PLOT_BG_COLOR, edgecolor=PLOT_BORDER_COLOR, framealpha=0.9)
        else: ax_duration.text(0.5, 0.5, "No duration data", ha='center', va='center', fontsize=10, color=LABEL_COLOR)
        ax_duration.set_xlabel("Duration (minutes)"); ax_duration.set_ylabel("Number of Trades")

        ax_time = fig.add_subplot(gs[2, 1]); style_plot(ax_time, "Average Profit by Hour of Day")
        # ... (Time of Day plotting code from above) ...
        if 'entry_time' in trades_df.columns and not trades_df.empty:
            if not pd.api.types.is_datetime64_any_dtype(trades_df['entry_time']): trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
            trades_df['hour'] = trades_df['entry_time'].dt.hour
            hour_stats_agg = trades_df.groupby('hour').agg(mean_pnl_display=('pnl_pct_display', 'mean'), count=('pnl_pct_display', 'count')).reindex(range(24)).reset_index()
            bars = ax_time.bar(hour_stats_agg['hour'], hour_stats_agg['mean_pnl_display'].fillna(0), alpha=0.85, width=0.85)
            for i, bar in enumerate(bars):
                height = bar.get_height(); count_val = hour_stats_agg['count'].iloc[i]
                if pd.isna(count_val) or count_val == 0 : bar.set_color('#e0e0e0'); continue
                bar.set_color(COLOR_PNL_POSITIVE if height > 0 else (COLOR_PNL_NEGATIVE if height < 0 else '#bdc3c7'))
                y_range = ax_time.get_ylim()[1] - ax_time.get_ylim()[0] if ax_time.get_ylim()[1] != ax_time.get_ylim()[0] else 1; offset = y_range * 0.04
                y_pos = height + offset if height >=0 else height - offset
                ax_time.text(bar.get_x() + bar.get_width()/2., y_pos, f'n={int(count_val)}', ha='center', va='bottom' if height >=0 else 'top', fontsize=8, color=COLOR_TEXT_N_COUNT)
            ax_time.set_xticks(range(0, 24, 2)); ax_time.axvspan(8.75, 15.25, color=COLOR_PRIMARY_LINE, alpha=0.07, zorder=0)
        else: ax_time.text(0.5, 0.5, "No entry time data", ha='center', va='center', fontsize=10, color=LABEL_COLOR)
        ax_time.set_xlabel("Hour (24h format)"); ax_time.set_ylabel("Average Profit (%)")
        ax_time.axhline(y=0, color=PLOT_BORDER_COLOR, linestyle='-', linewidth=1.2, alpha=0.9)


        ax_timeline = fig.add_subplot(gs[3, :]); style_plot(ax_timeline, "Trade Timeline (size = magnitude of P&L)")
        # ... (Trade Timeline plotting code from above) ...
        if 'entry_time' in trades_df.columns and 'pnl_pct' in trades_df.columns and not trades_df.empty:
            if not pd.api.types.is_datetime64_any_dtype(trades_df['entry_time']): trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
            trades_df_sorted = trades_df.sort_values('entry_time'); win_mask = trades_df_sorted['pnl_pct'] > 0
            sizes = np.abs(trades_df_sorted['pnl_pct_display']) * 20 + 10 
            ax_timeline.axhspan(0.5, 1.5, color=COLOR_PNL_POSITIVE, alpha=0.05, zorder=0)
            ax_timeline.axhspan(-0.5, 0.5, color=COLOR_PNL_NEGATIVE, alpha=0.05, zorder=0)
            ax_timeline.scatter(trades_df_sorted.loc[win_mask, 'entry_time'], [1] * sum(win_mask), s=sizes[win_mask], color=COLOR_PNL_POSITIVE, alpha=0.7, marker='o', label='Winners', edgecolor=PLOT_BG_COLOR, linewidth=0.5)
            ax_timeline.scatter(trades_df_sorted.loc[~win_mask, 'entry_time'], [0] * sum(~win_mask), s=sizes[~win_mask], color=COLOR_PNL_NEGATIVE, alpha=0.7, marker='o', label='Losers', edgecolor=PLOT_BG_COLOR, linewidth=0.5)
            ax_timeline.set_yticks([0, 1]); ax_timeline.set_yticklabels(['Loss', 'Win'], fontsize=10)
            ax_timeline.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')); ax_timeline.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
            fig.autofmt_xdate(rotation=20, ha='right')
            ax_timeline.legend(fontsize=9.5, frameon=True, facecolor=PLOT_BG_COLOR, edgecolor=PLOT_BORDER_COLOR, framealpha=0.9, loc='upper right')
            ax_timeline.set_ylim(-0.6, 1.6)
        else: ax_timeline.text(0.5, 0.5, "Insufficient data", ha='center', va='center', fontsize=10, color=LABEL_COLOR)
        ax_timeline.set_xlabel("Date")
        
        return fig


if __name__ == "__main__":
    symbol = "SPY"
    lookback_days = 7
    
    print(f"Downloading {lookback_days} days of minute data for {symbol}...")
    
    try:
        # Download data
        data = yf.download(
            tickers=symbol,
            period=f"{lookback_days}d",
            interval="1m",
            progress=False
        ).dropna()

        if data.empty or len(data) < 100:
            raise RuntimeError("Insufficient minute data returned. Check symbol or date range.")
        
        # Handle multi-level columns properly
        if isinstance(data.columns, pd.MultiIndex):
            print("Detected multi-level columns, creating properly formatted dataframe...")
            flat_data = pd.DataFrame(index=data.index)
            
            # Extract data from the multi-level columns
            for target_col, source_cols in {
                "Open": ["Open", "open"],
                "High": ["High", "high"],
                "Low": ["Low", "low"],
                "Close": ["Close", "close", "Adj Close"],
                "Volume": ["Volume", "volume"]
            }.items():
                found = False
                # Try each possible source column
                for src in source_cols:
                    # Direct column access
                    if src in data.columns:
                        flat_data[target_col] = data[src]
                        found = True
                        break
                    
                    # Check in multi-level indices
                    for col in data.columns:
                        if isinstance(col, tuple) and src in col:
                            flat_data[target_col] = data[col]
                            found = True
                            break
                    
                    if found:
                        break
            
            data = flat_data
        else:
            # Simple column renaming for single-level columns
            data.columns = data.columns.str.capitalize()
        
        # Verify we have all required columns
        required_cols = {"Open", "High", "Low", "Close", "Volume"}
        if not required_cols.issubset(set(data.columns)):
            missing = required_cols - set(data.columns)
            print(f"WARNING: Missing required columns: {missing}")
            print(f"Available columns: {data.columns.tolist()}")
            
            # Try a last resort approach - rename based on position
            if len(data.columns) >= 5:
                print("Attempting to rename columns based on position...")
                col_mapping = {
                    data.columns[0]: "Open",
                    data.columns[1]: "High", 
                    data.columns[2]: "Low",
                    data.columns[3]: "Close",
                    data.columns[4]: "Volume"
                }
                data = data.rename(columns=col_mapping)
            else:
                raise ValueError("Cannot determine appropriate columns for OHLCV data")
        
        print("Final data structure:")
        print(data.head())
        print(f"Columns: {data.columns.tolist()}")
        print(f"Data shape: {data.shape}")
        
       
        backtester = MeanReversionBacktester(
            # Signal parameters
            lookback_period=20,
            z_entry=1.5,           
            z_exit=0.5,            
            rsi_lookback=14,
            rsi_entry=30.0,        
            rsi_exit=55.0,         
            
            # ATR parameters
            atr_lookback=14,
            atr_stop_loss=1.5,     
            atr_take_profit=2.0,   
            
            # Position sizing
            risk_per_trade=0.01,   
            max_position_pct=0.1,  
            
            # VWAP filter
            vwap_entry_pct=0.005,  
            
            # Filters
            adx_lookback=14,
            adx_threshold=20.0,    
            min_volatility=0.01,   
            max_volatility=0.035,  
            
            # Trade management
            min_holding_period=5,  
            trade_cooldown=15,     
            
            # Execution parameters
            execution_delay=1,
            slippage_base_pct=0.0001,
            commission_per_share=0.005,
            
            long_only=True
        )
        
        try:
            results = backtester.run_backtest(data)
            # Generate performance metrics dashboard
            backtester.show_dashboard()
            # Plot trade analytics if trades exist
            if len(backtester.trades) > 0:
                backtester.plot_trade_analytics()
            
            
            print("\nPerformance Metrics:")
            for key, val in results["metrics"].items():
                if key in ["cagr", "max_drawdown", "hit_rate", "avg_profit", "avg_win", "avg_loss", 
                           "stop_loss_rate", "take_profit_rate"]:
                    print(f"{key.capitalize():15}: {val:.2%}")
                elif key in ["sharpe", "sortino", "calmar", "profit_factor", "turnover", "avg_duration"]:
                    print(f"{key.capitalize():15}: {val:.2f}")
                else:
                    print(f"{key.capitalize():15}: {val}")
            
            
            if len(backtester.trades) > 0:
                print("\nTrade Summary:")
                trades_df = pd.DataFrame(backtester.trades)
                print(f"Total Trades: {len(trades_df)}")
                print(f"Winning Trades: {len(trades_df[trades_df['pnl_pct'] > 0])}")
                print(f"Losing Trades: {len(trades_df[trades_df['pnl_pct'] < 0])}")
                print(f"Stop-Loss Exits: {len(trades_df[trades_df['exit_reason'] == 'stop_loss'])}")
                print(f"Take-Profit Exits: {len(trades_df[trades_df['exit_reason'] == 'take_profit'])}")
                print(f"Signal Exits: {len(trades_df[trades_df['exit_reason'] == 'signal'])}")
                print(f"Indicator Exits: {len(trades_df[trades_df['exit_reason'] == 'indicator_exit'])}")
            else:
                print("\nNo trades executed during backtest period.")
                
        except Exception as e:
            print(f"Error during backtest execution: {str(e)}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"Error downloading or processing data: {str(e)}")
        import traceback
        traceback.print_exc()

