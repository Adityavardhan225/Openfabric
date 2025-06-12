

import os
import sys
import time
import gc
import logging
import numpy as np
import pandas as pd
import dask
import dask.dataframe as dd
from dask.distributed import Client
import pyarrow as pa
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler('eth_feature_builder.log', mode='w')]
)
logger = logging.getLogger('ETHFeatureBuilder')

def convert_string_booleans(df):
    if df.empty: return df
    bool_map = {'True': True, 'true': True, 'False': False, 'false': False, '1': True, '0': False}
    if 'is_buyer_maker' in df.columns:
        df['is_buyer_maker'] = df['is_buyer_maker'].astype(str).map(bool_map).fillna(False).astype(bool)
    return df

class ETHFeatureBuilder:
    """
    Builds a comprehensive feature matrix from large tick-by-tick data using a robust,
    memory-efficient, and parallelized Dask workflow.
    """
    def __init__(self, delta_t=5):
        self.delta_t = delta_t
        self.client = None
        self.ddf = None
        logger.info(f"Initialized with: delta_t={self.delta_t}s")

    def set_dask_client(self, client: Client):
        self.client = client
        if self.client:
            n_workers = len(self.client.scheduler_info()['workers'])
            logger.info(f"Using {n_workers} workers. Dashboard: {self.client.dashboard_link}")

    def convert_csv_to_parquet(self, csv_path, parquet_dir):
        """Converts a large CSV to a partitioned Parquet dataset with a stable schema."""
        logger.info(f"Converting CSV '{csv_path}' to Parquet format...")
        start_time = time.time()
        if os.path.exists(parquet_dir):
            import shutil
            shutil.rmtree(parquet_dir)
        os.makedirs(parquet_dir, exist_ok=True)
        
        columns = ['trade_id', 'price', 'quantity', 'trade_value', 'timestamp', 'is_buyer_maker', 'is_best_match']
        dtypes = {'price': 'float32', 'quantity': 'float32', 'trade_value': 'float32', 'is_buyer_maker': 'str', 'is_best_match': 'str'}
        
        ddf = dd.read_csv(
            csv_path, header=None, names=columns, dtype=dtypes, 
            on_bad_lines='skip', usecols=columns, blocksize='64MB'
        )
        ddf['datetime'] = dd.to_datetime(ddf['timestamp'], unit='us', errors='coerce')
        ddf = ddf.dropna(subset=['datetime'])
        ddf['hour'] = ddf['datetime'].dt.strftime('%Y%m%d_%H')
        
        schema = {
            'trade_id': 'int64', 'price': 'float32', 'quantity': 'float32',
            'trade_value': 'float32', 'timestamp': 'int64',
            'is_buyer_maker': 'string', 'is_best_match': 'string',
            'datetime': pa.timestamp('ns'), 'hour': 'string'
        }
        ddf.to_parquet(
            parquet_dir, engine='pyarrow', compression='snappy',
            partition_on=['hour'], write_index=False, schema=schema
        )
        logger.info(f"CSV conversion completed in {time.time() - start_time:.1f} seconds.")
        return parquet_dir

    def load_tick_data(self, data_path):
        parquet_dir = data_path.replace('.csv', '') + "_parquet"
        if data_path.lower().endswith(".csv") and not os.path.exists(parquet_dir):
            self.convert_csv_to_parquet(data_path, parquet_dir)
        self.data_dir = parquet_dir if data_path.lower().endswith(".csv") else data_path
        self._create_dask_dataframe()
        return True

    def _create_dask_dataframe(self):
        self.ddf = dd.read_parquet(self.data_dir, engine='pyarrow')
        self.ddf = self.ddf.set_index('datetime', sorted=True)
        self.ddf = self.ddf.repartition(partition_size="256MB")
        logger.info(f"Dask DataFrame created with {self.ddf.npartitions} partitions.")
        logger.info("Sample data:\n" + str(self.ddf.head()))

    @staticmethod
    def _calculate_features(df, delta_t):
        """A single static method to prepare and calculate all advanced features on a pandas DataFrame."""
        if df.empty: return pd.DataFrame()

        # 1. Prepare Data
        df = df.loc[~df.index.duplicated(keep='first')]
        df = convert_string_booleans(df)
        df['buy_volume'] = df['quantity'].where(~df['is_buyer_maker'], 0)
        df['sell_volume'] = df['quantity'].where(df['is_buyer_maker'], 0)
        
        # 2. Resample to create the feature grid and `mid_price`
        grid_df = df.resample(f'{delta_t}S')['price'].mean().to_frame(name='mid_price')
        if grid_df.empty: return grid_df
        grid_df.dropna(inplace=True)
        grid_df = grid_df.loc[~grid_df.index.duplicated(keep='first')]

        # 3. Calculate "Primitive" Features from the prompt
        tfi_window = f'{delta_t}s'
        obi_window = '30s'
        
        grid_df['feature_tfi'] = ((df['buy_volume'].rolling(tfi_window).sum() - df['sell_volume'].rolling(tfi_window).sum()) / (df['buy_volume'].rolling(tfi_window).sum() + df['sell_volume'].rolling(tfi_window).sum() + 1e-9)).reindex(grid_df.index, method='ffill')
        grid_df['feature_obi'] = ((df['buy_volume'].rolling(obi_window).sum() - df['sell_volume'].rolling(obi_window).sum()) / (df['buy_volume'].rolling(obi_window).sum() + df['sell_volume'].rolling(obi_window).sum() + 1e-9)).reindex(grid_df.index, method='ffill')
        grid_df['feature_arrival_rate'] = (df['price'].rolling(tfi_window).count().reindex(grid_df.index, method='ffill') / delta_t)
        vol_1min = df['price'].pct_change().rolling('60s').std().reindex(grid_df.index, method='ffill')
        vol_10min = df['price'].pct_change().rolling('600s').std().reindex(grid_df.index, method='ffill')
        grid_df['feature_volatility_cluster'] = vol_1min / (vol_10min + 1e-9)

        # --- 4. Engineer Advanced Contextual Features (The "Best Approach") ---
        
        # "Second-Order" Features
        grid_df['feature_vol_of_vol_1m'] = grid_df['feature_volatility_cluster'].rolling(window=12).std()
        obi_ma_1min = grid_df['feature_obi'].rolling(window=12).mean()
        grid_df['feature_obi_accel'] = obi_ma_1min.pct_change(periods=12)

        # "Signature Plot" Inspired Features
        for p in [5, 10, 20, 40, 60, 120]:
            grid_df[f'feature_momentum_{p}'] = grid_df['mid_price'].pct_change(periods=p)
            grid_df[f'feature_volatility_{p}'] = grid_df['mid_price'].pct_change().rolling(window=p).std()

        for p in [30, 60, 90, 120]:
            mom_col = f'feature_momentum_{p}'
            acc_col = f'{mom_col}_acceleration'
            if mom_col in grid_df.columns:
                grid_df[acc_col] = grid_df[mom_col].pct_change(periods=12)

                # --- Feature Engineering 2.0 additions ---
        # Momentum Acceleration for 120-period normalized momentum
        if 'feature_momentum_120_norm' in grid_df.columns:
            grid_df['feature_momentum_120_acceleration'] = grid_df['feature_momentum_120_norm'].pct_change(periods=12)

        # Volatility Ratio: 20-period norm / 120-period norm
        if 'feature_volatility_20_norm' in grid_df.columns and 'feature_volatility_120_norm' in grid_df.columns:
            grid_df['feature_volatility_ratio_20_120'] = (
                grid_df['feature_volatility_20_norm'] / (grid_df['feature_volatility_120_norm'] + 1e-9)
            )


        # Interaction Features
        grid_df['feature_vol_weighted_obi'] = grid_df['feature_obi'] * grid_df['feature_volatility_cluster']
        grid_df['feature_arrival_weighted_tfi'] = grid_df['feature_arrival_rate'] * grid_df['feature_tfi']
        
        # --- Finalization ---
        final_column_order = list(grid_df.columns)
        
        final_df = grid_df.copy()
        final_df.fillna(0, inplace=True)
        final_df.replace([np.inf, -np.inf], 0, inplace=True)
        
        return final_df[final_column_order]

    def build_feature_matrix(self):
        """Builds the complete feature matrix using a memory-efficient, distributed approach."""
        logger.info("Building feature matrix using Dask `map_overlap`...")
        start_time = time.time()
        
        # Longest window from pct_change/rolling + buffer
        lookback = pd.Timedelta(seconds=600 + 60) 
        
        # Infer the schema automatically from a sample of the data. This is robust.
        dummy_df = self.ddf._meta_nonempty.copy()
        meta = ETHFeatureBuilder._calculate_features(dummy_df, self.delta_t)
        delta_t_local = self.delta_t
        feature_matrix_ddf = self.ddf.map_overlap(
            ETHFeatureBuilder._calculate_features,
            before=lookback, after=0, align_dataframes=False,
            meta=meta, delta_t=delta_t_local,
        )
        
        logger.info("Dask graph built. Now computing the final feature matrix...")
        feature_matrix = feature_matrix_ddf.compute()
        
        feature_matrix = feature_matrix.loc[~feature_matrix.index.duplicated(keep='first')]
        feature_matrix.reset_index(inplace=True)
        feature_matrix.rename(columns={'datetime': 'timestamp'}, inplace=True)
        
        logger.info(f"Feature matrix built with {len(feature_matrix):,} rows in {time.time() - start_time:.1f} seconds")
        
        self.feature_matrix = feature_matrix
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, "feature_matrix.parquet")
        feature_matrix.to_parquet(output_path, engine='pyarrow', compression='snappy')
        logger.info(f"Feature matrix saved to '{output_path}'.")
        return self.feature_matrix
    
    def save_feature_matrix(self, output_path):
        if not hasattr(self, 'feature_matrix') or self.feature_matrix.empty:
            raise ValueError("Feature matrix has not been built.")
        if not output_path.endswith('.parquet'):
            output_path += '.parquet'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.feature_matrix.to_parquet(output_path, engine='pyarrow', compression='snappy')
        logger.info(f"Feature matrix saved to '{output_path}'.")

    def cleanup(self):
        logger.info("Cleaning up ETHFeatureBuilder resources...")
        if hasattr(self, 'ddf'): del self.ddf
        if hasattr(self, 'feature_matrix'): del self.feature_matrix
        self.client = None
        gc.collect()