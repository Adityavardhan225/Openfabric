# utils.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_sharpe_ratio(pnl_history: List[float], risk_free_rate: float = 0.0, periods_per_year=252*24*60) -> float:
    """Calculate annualized Sharpe ratio. Assumes PnL is recorded frequently (e.g., per minute/step)."""
    if len(pnl_history) < 2: return 0.0
    returns = np.diff(pnl_history)
    if np.std(returns) == 0: return 0.0
    sharpe = (np.mean(returns) - risk_free_rate) / np.std(returns)
    return sharpe * np.sqrt(periods_per_year)

def calculate_sortino_ratio(pnl_history: List[float], risk_free_rate: float = 0.0, periods_per_year=252*24*60) -> float:
    """Calculate annualized Sortino ratio."""
    if len(pnl_history) < 2: return 0.0
    returns = np.diff(pnl_history)
    downside_returns = returns[returns < 0]
    if len(downside_returns) == 0 or np.std(downside_returns) == 0: return 10.0
    sortino = (np.mean(returns) - risk_free_rate) / np.std(downside_returns)
    return sortino * np.sqrt(periods_per_year)

def plot_training_metrics(
    episode_rewards: List[float],
    training_episode_metrics: List[Dict[str, Any]],
    output_path: str = None
):
    """
    Plot comprehensive training metrics, including rewards and key trading performance indicators.
    """
    if not episode_rewards:
        logging.warning("No episode rewards to plot.")
        return

    fig, axes = plt.subplots(3, 2, figsize=(18, 16))
    fig.suptitle('Training Performance Over Episodes', fontsize=18)

    # 1. Episode Rewards
    axes[0, 0].plot(episode_rewards, label='Episode Reward', alpha=0.8, color='c')
    axes[0, 0].plot(pd.Series(episode_rewards).rolling(50, min_periods=1).mean(), label='50-ep Rolling Avg', linestyle='--', color='b')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    if not training_episode_metrics:
        logging.warning("No detailed training metrics provided to plot.")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if output_path: plt.savefig(output_path)
        plt.close()
        return

    # Extract metrics from the list of dicts
    pnls = [m.get('pnl', 0) for m in training_episode_metrics]
    fill_ratios = [m.get('fill_ratio', 0) for m in training_episode_metrics]
    adverse_ratios = [m.get('adverse_selection_ratio', 0) for m in training_episode_metrics]
    # maker_ratios = [(m.get('maker_volume', 0) / (m.get('maker_volume', 0) + m.get('taker_volume', 1e-9))) for m in training_episode_metrics]
    maker_ratios = []
    for m in training_episode_metrics:
        maker_vol = m.get('maker_volume', 0.0)
        taker_vol = m.get('taker_volume', 0.0)
        total_volume = maker_vol + taker_vol
        
        if total_volume == 0:
            maker_ratios.append(0.0) # If no trades, the ratio is 0
        else:
            maker_ratios.append(maker_vol / total_volume)

    # 2. PnL per Episode
    axes[0, 1].plot(pnls, label='Episode PnL', color='g')
    axes[0, 1].plot(pd.Series(pnls).rolling(50, min_periods=1).mean(), label='50-ep Rolling Avg', linestyle='--', color='darkgreen')
    axes[0, 1].axhline(0, color='grey', linestyle='--', alpha=0.7)
    axes[0, 1].set_title('Profit and Loss (PnL) per Episode')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Final PnL ($)')
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    # 3. Fill Ratio per Episode
    axes[1, 0].plot(fill_ratios, label='Fill Ratio', color='m')
    axes[1, 0].plot(pd.Series(fill_ratios).rolling(50, min_periods=1).mean(), label='50-ep Rolling Avg', linestyle='--', color='purple')
    axes[1, 0].set_title('Fill Ratio per Episode')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Fill Ratio')
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    # 4. Adverse Selection Ratio per Episode
    axes[1, 1].plot(adverse_ratios, label='Adverse Selection Ratio', color='r')
    axes[1, 1].plot(pd.Series(adverse_ratios).rolling(50, min_periods=1).mean(), label='50-ep Rolling Avg', linestyle='--', color='darkred')
    axes[1, 1].set_title('Adverse Selection Ratio per Episode')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Adverse Selection Ratio')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    # 5. Maker Ratio per Episode
    axes[2, 0].plot(maker_ratios, label='Maker Ratio', color='orange')
    axes[2, 0].plot(pd.Series(maker_ratios).rolling(50, min_periods=1).mean(), label='50-ep Rolling Avg', linestyle='--', color='sienna')
    axes[2, 0].set_title('Maker Volume Ratio per Episode')
    axes[2, 0].set_xlabel('Episode')
    axes[2, 0].set_ylabel('Maker Volume / Total Volume')
    axes[2, 0].set_ylim(0, 1)
    axes[2, 0].grid(True)
    axes[2, 0].legend()
    
    # Hide the last empty subplot
    axes[2, 1].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if output_path:
        plt.savefig(output_path)
        logging.info(f"Saved training plot to {output_path}")
    else:
        plt.show()
    plt.close()

def plot_evaluation_summary(metrics_df: pd.DataFrame, output_path: str = None):
    """Create a summary plot of key evaluation metrics from a DataFrame."""
    if metrics_df.empty:
        logging.warning("Metrics DataFrame is empty, skipping evaluation summary plot.")
        return

    key_metrics = ['final_pnl', 'fill_ratio', 'adverse_selection_ratio', 'sharpe_ratio']
    plot_metrics = [m for m in key_metrics if m in metrics_df.columns]
    
    if not plot_metrics:
        logging.warning("No key metrics found in DataFrame for evaluation plot.")
        return

    num_metrics = len(plot_metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 5))
    axes = np.ravel(axes) # Ensure axes is always a 1D array
    
    fig.suptitle('Evaluation Metrics Summary (Distribution over Episodes)', fontsize=16)

    for i, metric in enumerate(plot_metrics):
        ax = axes[i]
        data = metrics_df[metric].dropna()
        if data.empty: continue

        ax.hist(data, bins=15, alpha=0.7, color='skyblue', density=False)
        mean_val = data.mean()
        ax.axvline(mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.3f}')
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(axis='y', alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    if output_path:
        plt.savefig(output_path)
        logging.info(f"Saved evaluation summary plot to {output_path}")
    else:
        plt.show()
    plt.close()

def plot_trades_and_pnl(pnl_history, inventory_history, mid_price_history, trades, output_path=None):
    """Create detailed visualization of trades and PnL for an L2 environment."""
    if not mid_price_history or not pnl_history:
        logging.warning("Cannot generate plot: history is empty.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})
    
    # Handle PnL history format
    pnl_timestamps = np.array([t[0] for t in pnl_history])
    pnl_values = np.array([t[1] for t in pnl_history])
    
    # Use PnL timestamps as the primary time axis
    time_axis = pnl_timestamps
    
    # 1. Price and Trades Plot
    axes[0].plot(time_axis, mid_price_history, label='Mid Price', color='grey', linestyle=':', alpha=0.8)
    axes[0].set_title('Market Mid Price and Agent Trades')
    axes[0].set_ylabel('Price ($)')
    axes[0].grid(True)

    if trades:
        buy_trades = [t for t in trades if t.side == 'buy']
        sell_trades = [t for t in trades if t.side == 'sell']
        
        if buy_trades:
            buy_ts = [t.timestamp for t in buy_trades]
            buy_prices = [t.price for t in buy_trades]
            buy_sizes = [t.size * 25 for t in buy_trades]
            axes[0].scatter(buy_ts, buy_prices, s=buy_sizes, color='green', alpha=0.7, marker='^', label='Buys')

        if sell_trades:
            sell_ts = [t.timestamp for t in sell_trades]
            sell_prices = [t.price for t in sell_trades]
            sell_sizes = [t.size * 25 for t in sell_trades]
            axes[0].scatter(sell_ts, sell_prices, s=sell_sizes, color='red', alpha=0.7, marker='v', label='Sells')
    
    axes[0].legend()

    # 2. Inventory Plot
    axes[1].plot(time_axis, inventory_history, color='purple')
    axes[1].set_title('Agent Inventory')
    axes[1].set_ylabel('Position Size')
    axes[1].grid(True)
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)

    # 3. PnL Plot
    axes[2].plot(time_axis, pnl_values, color='blue')
    axes[2].set_title('Agent Profit and Loss (PnL)')
    axes[2].set_xlabel('Time (Timestamp)')
    axes[2].set_ylabel('PnL ($)')
    axes[2].grid(True)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
        logging.info(f"Saved evaluation visualization to {output_path}")
    else:
        plt.show()
    plt.close()