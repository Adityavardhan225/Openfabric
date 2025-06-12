import os
import sys
import time
import gc
import json
import logging
import traceback
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, 
    confusion_matrix, precision_recall_curve, roc_curve
)
import torch

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('price_direction_model.log')
    ]
)
logger = logging.getLogger('PriceDirectionPredictor')

class PriceDirectionPredictor:
    """
    Optimized price direction prediction model designed to work with feature-engineered data.
    Focuses on efficient handling of large datasets (90GB+) by:
    1. Processing data in manageable chunks
    2. Implementing memory-efficient operations
    3. Using streaming algorithms where possible
    """
    
    def __init__(self, 
                 forecast_horizon=60,
                 transaction_cost=0.0006,
                 test_size=0.2,
                 n_jobs=-1,
                 use_gpu=True):
        """
        Initialize the predictor with configuration parameters
        
        Parameters:
        -----------
        forecast_horizon : int
            Forecast horizon in seconds (default: 60 seconds = 1 minute)
        transaction_cost : float
            Transaction cost as a fraction (default: 0.0006 = 0.06%)
        test_size : float
            Proportion of data to use for testing (default: 0.2 = 20%)
        n_jobs : int
            Number of jobs for parallel processing (-1 for all cores)
        use_gpu : bool
            Whether to use GPU acceleration if available
        """
        logger.info("Initializing PriceDirectionPredictor")
        self.forecast_horizon = forecast_horizon
        self.transaction_cost = transaction_cost
        self.test_size = test_size
        self.n_jobs = n_jobs
        self.use_gpu = use_gpu
        
        # Model-related attributes
        self.model = None
        self.feature_cols = None
        self.optimal_threshold = {'long': 0.55, 'short': 0.45}
        self.feature_importance = None
        self.performance_metrics = None
        
        # Runtime statistics and normalization parameters
        self.feature_means = {}
        self.feature_stds = {}
        self.n_samples_seen = 0
        
        logger.info(f"Initialized with forecast_horizon={forecast_horizon}s, "
                   f"transaction_cost={transaction_cost}, test_size={test_size}")
        
        self.use_gpu = use_gpu and torch.cuda.is_available()
        if self.use_gpu:
            self.device = torch.device('cuda')
            print(f"Using GPU for prediction: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            print("Using CPU for prediction")
        
        # Auto-detect compute resources
        import psutil
        self.total_memory_gb = psutil.virtual_memory().total / (1024**3)
        self.total_cores = psutil.cpu_count(logical=False)
        logger.info(f"System has {self.total_cores} cores and {self.total_memory_gb:.1f}GB memory")
        
        # Configure memory usage - aim to use at most 75% of available memory
        self.chunk_size = self._calculate_optimal_chunk_size()
        logger.info(f"Using chunk size of {self.chunk_size} rows")
    
    def _calculate_optimal_chunk_size(self):
        """Calculate optimal chunk size based on available memory"""
        # Estimate ~1KB per row with features
        available_memory_bytes = self.total_memory_gb * 0.75 * 1e9  # 75% of available memory
        row_size_bytes = 1000  # Estimated size per row in bytes
        return int(available_memory_bytes / row_size_bytes)
    
    def process_features(self, feature_df):
        """
        Process feature-engineered data to prepare for model training/prediction
        
        Parameters:
        -----------
        feature_df : pandas.DataFrame
            Feature-engineered data from feature_eng.py
            
        Returns:
        --------
        pandas.DataFrame
            Processed data with target labels and normalized features
        """
        logger.info(f"Processing features dataframe with {len(feature_df)} rows")
        
        # Create a copy to avoid modifying the original
        df = feature_df.copy()
        
        # Store original feature columns
        exclude_cols = {'timestamp', 'future_timestamp', 'future_mid_price', 'label', 'return_pct', 'hour'}
        # self.feature_cols = [col for col in df.columns if col not in exclude_cols]
        self.feature_cols = [
            col for col in df.columns
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
        ]
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp to ensure correct sequence
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
            
        # Create future price timestamp
        df['future_timestamp'] = df['timestamp'] + pd.Timedelta(seconds=self.forecast_horizon)
        
        if 'mid_price' not in df.columns:
            logger.error("mid_price column missing from features. Please check feature engineering pipeline.")
            raise ValueError("mid_price column missing from features.")
        
        # Create reference series for future prices
        future_prices = df[['timestamp', 'mid_price']].copy()
        future_prices.columns = ['future_timestamp', 'future_mid_price']
        
        # Merge current data with future price using merge_asof
        result = pd.merge_asof(
            df,
            future_prices,
            left_on='future_timestamp',
            right_on='future_timestamp',
            direction='nearest'
        )
        
        # Calculate return and target label
        result['return_pct'] = (result['future_mid_price'] - result['mid_price']) / result['mid_price']
        result['label'] = (result['return_pct'] > 0).astype(int)
        
        # Update online statistics for normalization
        self._update_normalization_stats(result[self.feature_cols])
        
        # Normalize features
        result = self._normalize_features(result)
        
        # Return processed dataframe
        return result
    
    def _update_normalization_stats(self, features_df):
        """
        Update running statistics for online normalization
        
        Parameters:
        -----------
        features_df : pandas.DataFrame
            DataFrame containing feature columns
        """
        n_new = len(features_df)
        
        # If this is the first batch, initialize means and stds
        if self.n_samples_seen == 0:
            for col in features_df.columns:
                self.feature_means[col] = features_df[col].mean()
                self.feature_stds[col] = features_df[col].std()
        else:
            # Update means and stds using Welford's online algorithm
            for col in features_df.columns:
                if col not in self.feature_means:
                    self.feature_means[col] = features_df[col].mean()
                    self.feature_stds[col] = features_df[col].std()
                    continue
                
                old_mean = self.feature_means[col]
                new_mean = features_df[col].mean()
                
                # Update combined mean
                delta = new_mean - old_mean
                self.feature_means[col] = old_mean + delta * n_new / (self.n_samples_seen + n_new)
                
                # Update variance using the parallel algorithm
                old_s = self.feature_stds[col]**2 * self.n_samples_seen
                new_s = features_df[col].var() * n_new
                
                combined_var = (old_s + new_s) / (self.n_samples_seen + n_new)
                combined_var += (delta**2) * self.n_samples_seen * n_new / (self.n_samples_seen + n_new)**2
                
                self.feature_stds[col] = np.sqrt(combined_var)
        
        # Update total samples seen
        self.n_samples_seen += n_new
    
    def _normalize_features(self, df):
        """
        Normalize features using stored statistics
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame to normalize
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with normalized features
        """
        result = df.copy()
        
        # Create normalized versions of features
        for col in self.feature_cols:
            if col in result.columns:
                norm_col = f"{col}_norm"
                mean = self.feature_means.get(col, 0)
                std = self.feature_stds.get(col, 1)
                
                # Normalize with small epsilon to prevent division by zero
                result[norm_col] = (result[col] - mean) / (std + 1e-8)
                
                # Clip to reasonable bounds
                result[norm_col] = result[norm_col].clip(-5, 5)
        
        return result
    
    def train_model(self, train_data, val_data=None, params=None):
        """
        Train price direction prediction model
        
        Parameters:
        -----------
        train_data : pandas.DataFrame
            Training data with features and label
        val_data : pandas.DataFrame
            Validation data (if None, 20% of train_data will be used)
        params : dict
            LightGBM model parameters (if None, default parameters will be used)
            
        Returns:
        --------
        dict
            Training results including model and metrics
        """

        import lightgbm as lgb 
        logger.info(f"Training model on {len(train_data)} rows")
        
        # If validation data is not provided, split from training
        if val_data is None:
            if 'timestamp' in train_data.columns:
                # Use time-based split
                train_data = train_data.sort_values('timestamp')
            else:
                # Use random split
                train_data = train_data.sample(frac=1, random_state=42)
            
            split_idx = int(len(train_data) * (1 - self.test_size))
            val_data = train_data.iloc[split_idx:]
            train_data = train_data.iloc[:split_idx]
            
        logger.info(f"Training on {len(train_data)} samples, validating on {len(val_data)} samples")
        
        # Identify normalized feature columns
        norm_feature_cols = [col for col in train_data.columns if col.endswith('_norm')]
        if not norm_feature_cols:
            logger.warning("No normalized features found, normalizing now")
            train_data = self._normalize_features(train_data)
            val_data = self._normalize_features(val_data)
            norm_feature_cols = [col for col in train_data.columns if col.endswith('_norm')]
        
        self.norm_feature_cols = norm_feature_cols
        logger.info(f"Using {len(norm_feature_cols)} normalized features")
        
        # Prepare training data
        X_train = train_data[norm_feature_cols].values
        y_train = train_data['label'].values
        
        X_val = val_data[norm_feature_cols].values
        y_val = val_data['label'].values
        
        # Calculate class weights for imbalanced data
        pos_count = np.sum(y_train == 1)
        neg_count = np.sum(y_train == 0)
        scale_pos_weight = neg_count / max(1, pos_count)
        
        # Set default parameters if not provided
        if params is None:
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'verbosity': -1,
                'n_jobs': self.n_jobs,
                'max_depth': 6,
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_child_samples': 20,
                'scale_pos_weight': scale_pos_weight
            }
            
            # If GPU is available and requested, use GPU acceleration
            if self.use_gpu:
                try:
                    import lightgbm as lgb
                    if lgb.gpu_available():
                        params['device'] = 'gpu'
                        params['gpu_platform_id'] = 0
                        params['gpu_device_id'] = 0
                        logger.info("Using GPU acceleration")
                except:
                    logger.warning("GPU requested but not available")
        
        # Create LightGBM datasets
        train_dataset = lgb.Dataset(X_train, y_train)
        val_dataset = lgb.Dataset(X_val, y_val, reference=train_dataset)
        
        # Set up early stopping
        callbacks = [lgb.early_stopping(stopping_rounds=50)]
        
        # Train the model
        n_estimators = params.pop('n_estimators', 200) if isinstance(params, dict) else 200
        
        self.model = lgb.train(
            params=params,
            train_set=train_dataset,
            num_boost_round=n_estimators,
            valid_sets=[val_dataset],
            callbacks=callbacks
        )
        
        # Store best number of iterations
        self.best_iteration = self.model.best_iteration
        
        # Calculate feature importance
        self.feature_importance = pd.DataFrame({
            'Feature': norm_feature_cols,
            'Importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('Importance', ascending=False)
        
        # Log top features
        top_features = self.feature_importance.head(10)
        logger.info("Top 10 features by importance:")
        for _, row in top_features.iterrows():
            logger.info(f"  {row['Feature']}: {row['Importance']:.4f}")
        
        # Calculate metrics on validation data
        val_preds = self.model.predict(X_val)
        auc = roc_auc_score(y_val, val_preds)
        
        # Determine optimal threshold using validation data
        self.optimal_threshold = self._find_optimal_thresholds(val_data, val_preds)
        
        logger.info(f"Model training completed with validation AUC: {auc:.4f}")
        logger.info(f"Optimal thresholds - long: {self.optimal_threshold['long']:.4f}, "
                   f"short: {self.optimal_threshold['short']:.4f}")
        
        return {
            'model': self.model,
            'feature_importance': self.feature_importance,
            'auc': auc,
            'optimal_threshold': self.optimal_threshold
        }
    
    def _find_optimal_thresholds(self, val_data, val_preds):
        """
        Find optimal thresholds for trading signals
        
        Parameters:
        -----------
        val_data : pandas.DataFrame
            Validation data
        val_preds : numpy.ndarray
            Predicted probabilities
            
        Returns:
        --------
        dict
            Optimal thresholds for long and short signals
        """
        # Create results DataFrame for simulation
        results = val_data.copy()
        results['prediction'] = val_preds
        
        # Threshold ranges to test
        thresholds = np.linspace(0.51, 0.49, 21)  # Test from 0.45 to 0.65 in 0.01 steps
        
        # Store results for each threshold
        threshold_results = []
        
        # Long signal threshold optimization
        for long_threshold in thresholds:
            # Generate signals
            results['signal'] = (results['prediction'] > long_threshold).astype(int)
            
            # Skip if no trades
            if results['signal'].sum() == 0:
                continue
            
            # Calculate returns
            trade_returns = results.loc[results['signal'] == 1, 'return_pct'] - self.transaction_cost
            
            # Calculate metrics
            win_rate = (trade_returns > 0).mean() if len(trade_returns) > 0 else 0
            expected_return = trade_returns.mean() if len(trade_returns) > 0 else 0
            sharpe = trade_returns.mean() / trade_returns.std() if len(trade_returns) > 0 and trade_returns.std() > 0 else 0
            
            # Calculate Profit Factor
            profit_factor = 0
            if sum(trade_returns < 0) > 0:
                profit_factor = abs(sum(trade_returns[trade_returns > 0]) / 
                                  sum(trade_returns[trade_returns < 0]))
            
            # Calculate combined score
            score = expected_return * 0.4 + sharpe * 0.4 + profit_factor * 0.2
            
            # Store metrics
            threshold_results.append({
                'threshold': long_threshold,
                'signal_type': 'long',
                'n_trades': len(trade_returns),
                'win_rate': win_rate,
                'expected_return': expected_return,
                'profit_factor': profit_factor,
                'sharpe': sharpe,
                'score': score
            })
        
        # Short signal threshold optimization
        for short_threshold in thresholds:
            # Generate signals (short when prediction is less than threshold)
            results['signal'] = (results['prediction'] < short_threshold).astype(int)
            
            # Skip if no trades
            if results['signal'].sum() == 0:
                continue
            
            # Calculate returns (short returns are opposite of price movement)
            trade_returns = -results.loc[results['signal'] == 1, 'return_pct'] - self.transaction_cost
            
            # Calculate metrics
            win_rate = (trade_returns > 0).mean() if len(trade_returns) > 0 else 0
            expected_return = trade_returns.mean() if len(trade_returns) > 0 else 0
            sharpe = trade_returns.mean() / trade_returns.std() if len(trade_returns) > 0 and trade_returns.std() > 0 else 0
            
            # Calculate Profit Factor
            profit_factor = 0
            if sum(trade_returns < 0) > 0:
                profit_factor = abs(sum(trade_returns[trade_returns > 0]) / 
                                  sum(trade_returns[trade_returns < 0]))
            
            # Calculate combined score
            score = expected_return * 0.4 + sharpe * 0.4 + profit_factor * 0.2
            
            # Store metrics
            threshold_results.append({
                'threshold': short_threshold,
                'signal_type': 'short',
                'n_trades': len(trade_returns),
                'win_rate': win_rate,
                'expected_return': expected_return,
                'profit_factor': profit_factor,
                'sharpe': sharpe,
                'score': score
            })
        
        # Convert to DataFrame for analysis
        threshold_df = pd.DataFrame(threshold_results)
        
        # Find optimal thresholds for long and short signals
        optimal_thresholds = {}
        
        # Long signals
        long_thresholds = threshold_df[threshold_df['signal_type'] == 'long']
        if not long_thresholds.empty and long_thresholds['score'].max() > 0:
            best_long = long_thresholds.loc[long_thresholds['score'].idxmax()]
            optimal_thresholds['long'] = best_long['threshold']
            logger.info(f"Optimal long threshold: {best_long['threshold']:.3f}, "
                       f"Win rate: {best_long['win_rate']:.2%}, "
                       f"Expected return: {best_long['expected_return']:.6f}")
        else:
            optimal_thresholds['long'] = 0.55  # Default conservative threshold
            logger.info("Using default long threshold of 0.55")
        
        # Short signals
        short_thresholds = threshold_df[threshold_df['signal_type'] == 'short']
        if not short_thresholds.empty and short_thresholds['score'].max() > 0:
            best_short = short_thresholds.loc[short_thresholds['score'].idxmax()]
            optimal_thresholds['short'] = best_short['threshold']  # Store as lower threshold
            logger.info(f"Optimal short threshold: {best_short['threshold']:.3f}, "
                       f"Win rate: {best_short['win_rate']:.2%}, "
                       f"Expected return: {best_short['expected_return']:.6f}")
        else:
            optimal_thresholds['short'] = 0.45  # Default conservative threshold
            logger.info("Using default short threshold of 0.45")
        
        return optimal_thresholds
    

    def optimize_hyperparameters(self, train_data, val_data=None, n_trials=30, timeout=1200):
        """
        Optimize hyperparameters using Optuna
        
        Parameters:
        -----------
        train_data : pandas.DataFrame
            Training data with features and label
        val_data : pandas.DataFrame
            Validation data (if None, 20% of train_data will be used)
        n_trials : int
            Number of optimization trials
        timeout : int
            Timeout in seconds
            
        Returns:
        --------
        dict
            Best hyperparameters
        """
        import lightgbm as lgb 
        logger.info(f"Optimizing hyperparameters with {n_trials} trials (timeout: {timeout}s)")
        
        import optuna
        from optuna.pruners import MedianPruner
        from optuna.samplers import TPESampler
        
        # If validation data is not provided, split from training
        if val_data is None:
            if 'timestamp' in train_data.columns:
                # Use time-based split
                train_data = train_data.sort_values('timestamp')
            else:
                # Use random split
                train_data = train_data.sample(frac=1, random_state=42)
            
            split_idx = int(len(train_data) * (1 - self.test_size))
            val_data = train_data.iloc[split_idx:]
            train_data = train_data.iloc[:split_idx]
            
        # Identify normalized feature columns
        norm_feature_cols = [col for col in train_data.columns if col.endswith('_norm')]
        if not norm_feature_cols:
            logger.warning("No normalized features found, normalizing now")
            train_data = self._normalize_features(train_data)
            val_data = self._normalize_features(val_data)
            norm_feature_cols = [col for col in train_data.columns if col.endswith('_norm')]
        
        self.norm_feature_cols = norm_feature_cols
        
        # Prepare training data
        X_train = train_data[norm_feature_cols].values
        y_train = train_data['label'].values
        
        X_val = val_data[norm_feature_cols].values
        y_val = val_data['label'].values
        
        # Calculate class weights for imbalanced data
        pos_count = np.sum(y_train == 1)
        neg_count = np.sum(y_train == 0)
        scale_pos_weight = neg_count / max(1, pos_count)
        
        # Create LightGBM datasets
        train_dataset = lgb.Dataset(X_train, y_train)
        val_dataset = lgb.Dataset(X_val, y_val, reference=train_dataset)
        
        # Define Optuna objective function
        def objective(trial):
            # Define parameters to optimize
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'verbosity': -1,
                'n_jobs': self.n_jobs,
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'num_leaves': trial.suggest_int('num_leaves', 7, 127),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
                'scale_pos_weight': scale_pos_weight
            }
            
            # GPU acceleration if available and requested
            if self.use_gpu:
                try:
                    if lgb.gpu_available():
                        params['device'] = 'gpu'
                except:
                    pass
            
            # Number of estimators
            n_estimators = trial.suggest_int('n_estimators', 50, 500)
            
            # Set up early stopping callback
            callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
            
            # Train model with early stopping
            try:
                model = lgb.train(
                    params=params,
                    train_set=train_dataset,
                    num_boost_round=n_estimators,
                    valid_sets=[val_dataset],
                    callbacks=callbacks
                )
                
                # Use best iteration for prediction
                val_preds = model.predict(X_val, num_iteration=model.best_iteration)
                
                # Calculate AUC
                auc = roc_auc_score(y_val, val_preds)
                
                # Additional metrics for trading-specific optimization
                # Test different thresholds and calculate expected PnL
                thresholds = np.linspace(0.4, 0.6, num=11)
                profits = []
                
                for threshold in thresholds:
                    # Generate predictions
                    pred_long = (val_preds >= threshold).astype(int)
                    
                    # Calculate precision (win rate)
                    if np.sum(pred_long) > 0:
                        precision_pos = precision_score(y_val, pred_long, zero_division=0)
                        
                        # Calculate expected return based on validation data
                        expected_return = val_data.loc[pred_long == 1, 'return_pct'].mean() if 'return_pct' in val_data.columns else 0
                        
                        # Calculate expected PnL with transaction costs
                        expected_pnl = (2 * precision_pos - 1) * abs(expected_return) - self.transaction_cost
                        profits.append(expected_pnl)
                    else:
                        profits.append(-1)  # Penalty for no trades
                
                max_profit = max(profits) if profits else -1
                
                # Combine AUC and profit for final score (weighted)
                final_score = auc * 0.4 + max(0, max_profit) * 10 * 0.6
                
                return final_score
                
            except Exception as e:
                logger.error(f"Error in hyperparameter optimization trial: {str(e)}")
                return -1
        
        # Create Optuna study
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=30)
        sampler = TPESampler(seed=42)
        
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner
        )
        
        # Run optimization with error handling
        try:
            study.optimize(
                objective,
                n_trials=n_trials,
                timeout=timeout,
                show_progress_bar=True,
                catch=(Exception,)
            )
            
            # Get best parameters
            if study.best_trial:
                best_params = study.best_params
                
                # Extract best number of estimators
                n_estimators = best_params.pop('n_estimators', 200)
                
                # Rebuild fixed parameters
                fixed_params = {
                    'objective': 'binary',
                    'metric': 'auc',
                    'boosting_type': 'gbdt',
                    'verbosity': -1,
                    'n_jobs': self.n_jobs,
                    'scale_pos_weight': scale_pos_weight
                }
                
                # Combine with best parameters
                final_params = {**fixed_params, **best_params}
                
                # Store best parameters
                self.model_params = final_params
                self.model_params['n_estimators'] = n_estimators
                
                # Log results
                logger.info(f"Best trial: {study.best_trial.number}")
                logger.info(f"Best score: {study.best_value:.6f}")
                logger.info(f"Best parameters: {best_params}")
                
                # Train final model with best parameters
                logger.info("Training final model with best parameters")
                self.train_model(train_data, val_data, final_params)
                
                return final_params
            else:
                logger.warning("No best trial found, using default parameters")
                return None
                
        except Exception as e:
            logger.error(f"Error in hyperparameter optimization: {str(e)}")
            return None

    def process_stream(self, feature_generator, max_samples=None, batch_size=10000):
        """
        Process streaming data from feature engineering pipeline
        
        Parameters:
        -----------
        feature_generator : generator
            Generator yielding feature-engineered data
        max_samples : int
            Maximum number of samples to process (None for no limit)
        batch_size : int
            Number of samples to accumulate before processing
            
        Returns:
        --------
        dict
            Processing results
        """
        logger.info(f"Processing streaming data with batch size {batch_size}")
        
        # Track processing metrics
        metrics = {
            'samples_processed': 0,
            'batches_processed': 0,
            'training_completed': False,
            'prediction_samples': 0
        }
        
        try:
            # Initialize data containers
            all_train_data = pd.DataFrame()
            all_val_data = pd.DataFrame()
            current_batch = pd.DataFrame()
            sample_count = 0
            
            # Set batch collection mode initially to training
            mode = 'training'
            training_threshold = 100000  # Collect at least this many samples before training
            
            # Process data stream
            for features_df in feature_generator:
                if features_df is None or len(features_df) == 0:
                    continue
                
                # Append to current batch
                current_batch = pd.concat([current_batch, features_df])
                sample_count += len(features_df)
                metrics['samples_processed'] += len(features_df)
                
                # Process batch when it reaches batch_size
                if len(current_batch) >= batch_size:
                    logger.info(f"Processing batch of {len(current_batch)} samples (mode: {mode})")
                    
                    # Process features
                    processed_batch = self.process_features(current_batch)
                    
                    # Handle different modes
                    if mode == 'training' and not metrics['training_completed']:
                        # Split batch into train/val
                        split_idx = int(len(processed_batch) * 0.8)
                        if 'timestamp' in processed_batch.columns:
                            processed_batch = processed_batch.sort_values('timestamp')
                        train_batch = processed_batch.iloc[:split_idx]
                        val_batch = processed_batch.iloc[split_idx:]
                        
                        # Accumulate training data
                        all_train_data = pd.concat([all_train_data, train_batch])
                        all_val_data = pd.concat([all_val_data, val_batch])
                        
                        # If we have enough data, train the model
                        if len(all_train_data) >= training_threshold:
                            logger.info(f"Training model with {len(all_train_data)} samples")
                            self.train_model(all_train_data, all_val_data)
                            metrics['training_completed'] = True
                            
                            # Switch to prediction mode after training
                            mode = 'prediction'
                    
                    elif mode == 'prediction' or metrics['training_completed']:
                        # Make predictions on the batch
                        predictions = self.predict(processed_batch)
                        metrics['prediction_samples'] += len(predictions)
                    
                    # Reset current batch
                    current_batch = pd.DataFrame()
                    metrics['batches_processed'] += 1
                
                # Check if we've reached the maximum samples
                if max_samples is not None and sample_count >= max_samples:
                    logger.info(f"Reached maximum samples limit: {max_samples}")
                    break
            
            # Process any remaining data in final batch
            if len(current_batch) > 0:
                processed_batch = self.process_features(current_batch)
                
                if mode == 'training' and not metrics['training_completed']:
                    # Try to train with whatever data we have
                    all_train_data = pd.concat([all_train_data, processed_batch])
                    
                    # Split for validation if needed
                    if len(all_val_data) == 0:
                        split_idx = int(len(all_train_data) * 0.8)
                        if 'timestamp' in all_train_data.columns:
                            all_train_data = all_train_data.sort_values('timestamp')
                        all_val_data = all_train_data.iloc[split_idx:]
                        all_train_data = all_train_data.iloc[:split_idx]
                    
                    if len(all_train_data) > 100:  # Minimum samples for training
                        logger.info(f"Training final model with {len(all_train_data)} samples")
                        self.train_model(all_train_data, all_val_data)
                        metrics['training_completed'] = True
                
                elif mode == 'prediction' or metrics['training_completed']:
                    # Make predictions on the batch
                    predictions = self.predict(processed_batch)
                    metrics['prediction_samples'] += len(predictions)
                
                metrics['batches_processed'] += 1
            
            logger.info(f"Processing completed: {metrics['samples_processed']} samples in {metrics['batches_processed']} batches")
            if metrics['training_completed']:
                logger.info(f"Model trained successfully, made predictions on {metrics['prediction_samples']} samples")
            else:
                logger.warning(f"Model training not completed, insufficient data")
            
            return {
                'status': 'success',
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Error processing streaming data: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            return {
                'status': 'error',
                'error': str(e),
                'metrics': metrics
            }

    def predict(self, data):
        """
        Generate predictions using the trained model
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Data to predict on
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with predictions and signals
        """
        if not hasattr(self, 'model') or self.model is None:
            logger.error("No trained model available for prediction")
            return None
        
        try:
            # Clone the dataframe to avoid modifying the original
            result_df = data.copy()
            
            # Ensure features are normalized
            if not all(col.endswith('_norm') for col in self.norm_feature_cols):
                # Normalize the features
                result_df = self._normalize_features(result_df)
            
            # Prepare features for prediction
            X = result_df[self.norm_feature_cols].values
            
            # Generate raw predictions
            raw_preds = self.model.predict(X)
            result_df['prediction'] = raw_preds
            
            # Generate trading signals based on thresholds
            long_threshold = self.optimal_threshold.get('long', 0.55)
            short_threshold = self.optimal_threshold.get('short', 0.45)
            
            result_df['signal_long'] = (result_df['prediction'] >= long_threshold).astype(int)
            result_df['signal_short'] = (result_df['prediction'] <= short_threshold).astype(int)
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def backtest(self, test_data):
        """
        Backtest model performance on test data
        
        Parameters:
        -----------
        test_data : pandas.DataFrame
            Data to backtest on
            
        Returns:
        --------
        pandas.DataFrame
            Backtest results
        """
        logger.info(f"Running backtest on {len(test_data)} samples")
        
        try:
            # Generate predictions
            results = self.predict(test_data)
            
            if results is None:
                logger.error("Failed to generate predictions for backtest")
                return None
            
            # Calculate returns
            # For long signals: profit = return - transaction_cost
            results['long_return'] = np.where(
                results['signal_long'] == 1,
                results['return_pct'] - self.transaction_cost,
                0
            )
            
            # For short signals: profit = -return - transaction_cost
            results['short_return'] = np.where(
                results['signal_short'] == 1,
                -results['return_pct'] - self.transaction_cost,
                0
            )
            
            # Combine returns (avoid double counting if both signals are somehow active)
            results['trade_return'] = results['long_return'] + results['short_return']
            
            # Calculate cumulative P&L
            results['cum_pnl'] = results['trade_return'].cumsum()
            
            # Calculate drawdown
            results['peak'] = results['cum_pnl'].cummax()
            results['drawdown'] = results['peak'] - results['cum_pnl']
            
            # Track trade correctness for analysis
            results['correct_long'] = (
                (results['signal_long'] == 1) & 
                (results['return_pct'] > self.transaction_cost)
            ).astype(int)
            
            results['correct_short'] = (
                (results['signal_short'] == 1) & 
                (results['return_pct'] < -self.transaction_cost)
            ).astype(int)
            
            # Store results for analysis
            self.backtest_results = results
            
            # Calculate performance metrics
            self.backtest_metrics = self.calculate_metrics(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in backtest: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def calculate_metrics(self, results):
        """
        Calculate performance metrics from backtest results
        
        Parameters:
        -----------
        results : pandas.DataFrame
            Backtest results
            
        Returns:
        --------
        dict
            Performance metrics
        """
        metrics = {}
        
        try:
            # Trading metrics
            total_trades = results['signal_long'].sum() + results['signal_short'].sum()
            win_trades = results['correct_long'].sum() + results['correct_short'].sum()
            
            if total_trades > 0:
                metrics['total_trades'] = int(total_trades)
                metrics['win_rate'] = win_trades / total_trades
                metrics['avg_trade_return'] = results['trade_return'].mean()
                
                # Calculate profit from winning and losing trades
                winning_returns = results.loc[results['trade_return'] > 0, 'trade_return']
                losing_returns = results.loc[results['trade_return'] < 0, 'trade_return']
                
                if len(winning_returns) > 0:
                    metrics['avg_win'] = winning_returns.mean()
                else:
                    metrics['avg_win'] = 0
                    
                if len(losing_returns) > 0:
                    metrics['avg_loss'] = losing_returns.mean()
                else:
                    metrics['avg_loss'] = 0
                
                # Risk metrics
                if len(losing_returns) > 0 and abs(losing_returns.sum()) > 0:
                    metrics['profit_factor'] = abs(winning_returns.sum() / losing_returns.sum())
                else:
                    metrics['profit_factor'] = float('inf') if winning_returns.sum() > 0 else 0
                
                # P&L metrics
                metrics['final_pnl'] = results['cum_pnl'].iloc[-1]
                metrics['max_drawdown'] = results['drawdown'].max()
                
                # Sharpe ratio (annualized)
                if results['trade_return'].std() > 0:
                    # Assuming daily returns, annualize by multiplying by sqrt(252)
                    metrics['sharpe_ratio'] = (
                        results['trade_return'].mean() / 
                        results['trade_return'].std() * 
                        np.sqrt(252)
                    )
                else:
                    metrics['sharpe_ratio'] = 0
            else:
                logger.warning("No trades in backtest")
                metrics = {
                    'total_trades': 0,
                    'win_rate': 0,
                    'avg_trade_return': 0,
                    'profit_factor': 0,
                    'final_pnl': 0,
                    'max_drawdown': 0,
                    'sharpe_ratio': 0
                }
            
            # Classification metrics (if we have labels)
            if 'label' in results.columns:
                # Long prediction performance
                if results['signal_long'].sum() > 0:
                    metrics['long_precision'] = precision_score(
                        results.loc[results['signal_long'] == 1, 'label'],
                        np.ones(results['signal_long'].sum()),
                        zero_division=0
                    )
                else:
                    metrics['long_precision'] = 0
                
                # Short prediction performance (label=0 is success for shorts)
                if results['signal_short'].sum() > 0:
                    metrics['short_precision'] = precision_score(
                        1 - results.loc[results['signal_short'] == 1, 'label'],
                        np.ones(results['signal_short'].sum()),
                        zero_division=0
                    )
                else:
                    metrics['short_precision'] = 0
                
                # Overall prediction performance
                metrics['roc_auc'] = roc_auc_score(results['label'], results['prediction'])
            
            # Return yearly if timestamps are available
            if 'timestamp' in results.columns:
                # Convert timestamp to datetime if needed
                if not pd.api.types.is_datetime64_any_dtype(results['timestamp']):
                    results['timestamp'] = pd.to_datetime(results['timestamp'])
                
                # Calculate date range in days
                date_range_days = (results['timestamp'].max() - results['timestamp'].min()).total_seconds() / (24*3600)
                
                if date_range_days > 0:
                    # Annualize returns
                    metrics['annualized_return'] = (1 + metrics['final_pnl']) ** (365 / date_range_days) - 1
                else:
                    metrics['annualized_return'] = 0
            
            # Log key metrics
            logger.info("\n===== BACKTEST METRICS =====")
            logger.info(f"Total trades: {metrics.get('total_trades', 0)}")
            logger.info(f"Win rate: {metrics.get('win_rate', 0):.2%}")
            logger.info(f"Profit factor: {metrics.get('profit_factor', 0):.4f}")
            logger.info(f"Final P&L: {metrics.get('final_pnl', 0):.6f}")
            logger.info(f"Max drawdown: {metrics.get('max_drawdown', 0):.6f}")
            logger.info(f"Sharpe ratio: {metrics.get('sharpe_ratio', 0):.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_trade_return': 0,
                'profit_factor': 0,
                'final_pnl': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'error': str(e)
            }

    def plot_results(self, output_dir=None):
        """
        Create visualizations of backtest results
        
        Parameters:
        -----------
        output_dir : str
            Directory to save plots (if None, plots are displayed)
        """
        if not hasattr(self, 'backtest_results') or self.backtest_results is None:
            logger.warning("No backtest results available for plotting")
            return
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            results = self.backtest_results
            
            # Create output directory if needed
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
            
            # Set style
            sns.set(style="whitegrid")
            plt.rcParams['figure.figsize'] = (12, 8)
            
            # 1. Cumulative P&L plot
            plt.figure(figsize=(12, 6))
            plt.plot(range(len(results)), results['cum_pnl'], label='Cumulative P&L')
            
            # Add drawdown as shaded area
            plt.fill_between(
                range(len(results)),
                -results['drawdown'],
                0,
                alpha=0.3,
                color='red',
                label='Drawdown'
            )
            
            plt.title('Cumulative P&L and Drawdown')
            plt.xlabel('Trade #')
            plt.ylabel('P&L')
            plt.legend()
            plt.grid(True)
            
            # Save or display
            if output_dir:
                plt.savefig(os.path.join(output_dir, 'cumulative_pnl.png'))
                plt.close()
            else:
                plt.show()
            
            # 2. Return distribution
            plt.figure(figsize=(10, 6))
            trade_returns = results.loc[
                (results['signal_long'] == 1) | (results['signal_short'] == 1), 
                'trade_return'
            ]
            
            if len(trade_returns) > 0:
                sns.histplot(trade_returns, kde=True, bins=50)
                plt.axvline(0, color='r', linestyle='--')
                
                plt.title('Trade Return Distribution')
                plt.xlabel('Return')
                plt.ylabel('Frequency')
                
                # Add statistics
                mean_return = trade_returns.mean()
                std_return = trade_returns.std()
                win_rate = (trade_returns > 0).mean()
                
                stats_text = (
                    f"Mean: {mean_return:.6f}\n"
                    f"Std: {std_return:.6f}\n"
                    f"Win Rate: {win_rate:.2%}"
                )
                
                plt.annotate(
                    stats_text,
                    xy=(0.02, 0.95),
                    xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8)
                )
                
                # Save or display
                if output_dir:
                    plt.savefig(os.path.join(output_dir, 'return_distribution.png'))
                    plt.close()
                else:
                    plt.show()
            
            # 3. Feature importance
            if hasattr(self, 'feature_importance') and self.feature_importance is not None:
                plt.figure(figsize=(10, 8))
                
                # Get top 20 features
                top_features = self.feature_importance.head(20)
                
                # Create horizontal bar chart
                sns.barplot(
                    y='Feature',
                    x='Importance',
                    data=top_features,
                    palette='viridis'
                )
                
                plt.title('Top 20 Features by Importance')
                plt.xlabel('Importance')
                plt.ylabel('Feature')
                plt.tight_layout()
                
                # Save or display
                if output_dir:
                    plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
                    plt.close()
                else:
                    plt.show()
                    
            # 4. ROC curve if we have labels
            if 'label' in results.columns and 'prediction' in results.columns:
                plt.figure(figsize=(8, 8))
                
                y_true = results['label']
                y_score = results['prediction']
                
                fpr, tpr, _ = roc_curve(y_true, y_score)
                roc_auc = roc_auc_score(y_true, y_score)
                
                plt.plot(
                    fpr, tpr, 
                    lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.4f})'
                )
                
                plt.plot([0, 1], [0, 1], 'k--', lw=2)
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic')
                plt.legend(loc="lower right")
                
                # Save or display
                if output_dir:
                    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
                    plt.close()
                else:
                    plt.show()
                    
            # 5. Performance metrics summary
            if hasattr(self, 'backtest_metrics') and self.backtest_metrics:
                plt.figure(figsize=(10, 6))
                plt.axis('off')
                
                metrics = self.backtest_metrics
                
                key_metrics = {
                    'Total Trades': metrics.get('total_trades', 0),
                    'Win Rate': f"{metrics.get('win_rate', 0):.2%}",
                    'Profit Factor': f"{metrics.get('profit_factor', 0):.4f}",
                    'Sharpe Ratio': f"{metrics.get('sharpe_ratio', 0):.4f}",
                    'Max Drawdown': f"{metrics.get('max_drawdown', 0):.6f}",
                    'Final P&L': f"{metrics.get('final_pnl', 0):.6f}",
                    'Annualized Return': f"{metrics.get('annualized_return', 0):.2%}",
                }
                
                # Create table
                table_data = [[k, v] for k, v in key_metrics.items()]
                
                table = plt.table(
                    cellText=table_data,
                    colLabels=['Metric', 'Value'],
                    cellLoc='center',
                    loc='center',
                    bbox=[0.2, 0.2, 0.6, 0.6]
                )
                
                table.auto_set_font_size(False)
                table.set_fontsize(12)
                table.scale(1.2, 1.5)
                
                plt.title('Backtest Performance Metrics', fontsize=16, pad=20)
                
                # Save or display
                if output_dir:
                    plt.savefig(os.path.join(output_dir, 'performance_metrics.png'), 
                                dpi=120, bbox_inches='tight')
                    plt.close()
                else:
                    plt.show()
                    
            logger.info(f"Created visualization plots" + 
                      (f" in {output_dir}" if output_dir else ""))
                
        except Exception as e:
            logger.error(f"Error plotting results: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

    def save_model(self, output_dir):
        """
        Save trained model and parameters
        
        Parameters:
        -----------
        output_dir : str
            Directory to save model files
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Save LightGBM model
            if hasattr(self, 'model') and self.model is not None:
                model_path = os.path.join(output_dir, "model.txt")
                self.model.save_model(model_path)
                logger.info(f"Model saved to {model_path}")
                
                # Save feature importance
                if hasattr(self, 'feature_importance') and self.feature_importance is not None:
                    fi_path = os.path.join(output_dir, "feature_importance.csv")
                    self.feature_importance.to_csv(fi_path, index=False)
            
            # Save normalization parameters
            norm_params = {
                'means': self.feature_means,
                'stds': self.feature_stds,
                'feature_cols': self.feature_cols,
                'norm_feature_cols': self.norm_feature_cols if hasattr(self, 'norm_feature_cols') else []
            }
            
            norm_path = os.path.join(output_dir, "normalization_params.json")
            with open(norm_path, 'w') as f:
                # Convert to serializable format
                serializable_params = {
                    'means': {k: float(v) for k, v in norm_params['means'].items()},
                    'stds': {k: float(v) for k, v in norm_params['stds'].items()},
                    'feature_cols': norm_params['feature_cols'],
                    'norm_feature_cols': norm_params['norm_feature_cols']
                }
                json.dump(serializable_params, f, indent=2)
            
            # Save optimal thresholds
            threshold_path = os.path.join(output_dir, "optimal_thresholds.json")
            with open(threshold_path, 'w') as f:
                json.dump(self.optimal_threshold, f, indent=2)
            
            # Save configuration parameters
            config = {
                'forecast_horizon': self.forecast_horizon,
                'transaction_cost': self.transaction_cost,
                'model_params': self.model_params if hasattr(self, 'model_params') else {},
                'test_size': self.test_size
            }
            
            config_path = os.path.join(output_dir, "config.json")
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Save backtest metrics if available
            if hasattr(self, 'backtest_metrics') and self.backtest_metrics:
                metrics_path = os.path.join(output_dir, "backtest_metrics.json")
                with open(metrics_path, 'w') as f:
                    # Convert numpy types to Python types
                    metrics_dict = {}
                    for k, v in self.backtest_metrics.items():
                        if hasattr(v, 'item'):  # numpy type
                            metrics_dict[k] = v.item()
                        else:
                            metrics_dict[k] = v
                    
                    json.dump(metrics_dict, f, indent=2)
            
            logger.info(f"Model and parameters saved to {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
            
    def load_model(self, model_dir):
        """
        Load trained model and parameters
        
        Parameters:
        -----------
        model_dir : str
            Directory containing model files
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            # Load model
            model_path = os.path.join(model_dir, "model.txt")
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False
            
            self.model = lgb.Booster(model_file=model_path)
            logger.info(f"Model loaded from {model_path}")
            
            # Load configuration
            config_path = os.path.join(model_dir, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Update instance attributes
                self.forecast_horizon = config.get('forecast_horizon', self.forecast_horizon)
                self.transaction_cost = config.get('transaction_cost', self.transaction_cost)
                self.model_params = config.get('model_params', {})
            
            # Load normalization parameters
            norm_path = os.path.join(model_dir, "normalization_params.json")
            if os.path.exists(norm_path):
                with open(norm_path, 'r') as f:
                    norm_params = json.load(f)
                
                self.feature_means = norm_params.get('means', {})
                self.feature_stds = norm_params.get('stds', {})
                self.feature_cols = norm_params.get('feature_cols', [])
                self.norm_feature_cols = norm_params.get('norm_feature_cols', [])
            
            # Load optimal thresholds
            threshold_path = os.path.join(model_dir, "optimal_thresholds.json")
            if os.path.exists(threshold_path):
                with open(threshold_path, 'r') as f:
                    self.optimal_threshold = json.load(f)
            else:
                self.optimal_threshold = {'long': 0.55, 'short': 0.45}
            
            # Load feature importance if available
            fi_path = os.path.join(model_dir, "feature_importance.csv")
            if os.path.exists(fi_path):
                self.feature_importance = pd.read_csv(fi_path)
            
            logger.info(f"Model and parameters loaded successfully from {model_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def run_pipeline(self, feature_data, output_dir=None):
        """
        Run end-to-end pipeline on feature-engineered data
        
        Parameters:
        -----------
        feature_data : pandas.DataFrame
            Feature-engineered data
        output_dir : str
            Directory to save results (default: auto-generated)
            
        Returns:
        --------
        dict
            Pipeline results
        """
        start_time = time.time()
        
        # Create output directory if not provided
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"price_prediction_{timestamp}"
            os.makedirs(output_dir, exist_ok=True)
        
        try:
            logger.info(f"Starting pipeline with {len(feature_data)} samples")
            
            # 1. Process features
            logger.info("Processing features...")
            processed_data = self.process_features(feature_data)
            
            # 2. Split into train/validation/test
            logger.info("Splitting data...")
            if 'timestamp' in processed_data.columns:
                processed_data = processed_data.sort_values('timestamp')
            
            # Use 60/20/20 split
            train_idx = int(len(processed_data) * 0.6)
            val_idx = int(len(processed_data) * 0.8)
            
            train_data = processed_data.iloc[:train_idx]
            val_data = processed_data.iloc[train_idx:val_idx]
            test_data = processed_data.iloc[val_idx:]
            
            # 3. Optimize hyperparameters
            logger.info("Optimizing hyperparameters...")
            self.optimize_hyperparameters(train_data, val_data, n_trials=30, timeout=1200)
            
            # 4. Train final model
            logger.info("Training final model...")
            training_results = self.train_model(
                pd.concat([train_data, val_data]), 
                test_data.sample(frac=0.5) if len(test_data) > 1000 else test_data
            )
            
            # 5. Run backtest
            logger.info("Running backtest...")
            backtest_results = self.backtest(test_data)
            
            # 6. Create visualizations
            logger.info("Creating visualizations...")
            self.plot_results(output_dir)
            
            # 7. Save model and results
            logger.info("Saving model and results...")
            self.save_model(output_dir)
            
            # 8. Report completion
            end_time = time.time()
            duration = end_time - start_time
            
            logger.info(f"Pipeline completed in {int(duration)} seconds")
            logger.info(f"Results saved to {output_dir}")
            
            return {
                'status': 'success',
                'output_dir': output_dir,
                'duration_seconds': int(duration),
                'metrics': self.backtest_metrics if hasattr(self, 'backtest_metrics') else None
            }
            
        except Exception as e:
            logger.error(f"Error in pipeline: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            return {
                'status': 'error',
                'error': str(e),
                'output_dir': output_dir
            }
        
    def detailed_analysis(self, output_dir=None):
        """
        Save detailed analysis of backtest results, including classification metrics,
        confusion matrix, feature importance, and more. All plots are saved, not displayed.
        """
        if not hasattr(self, 'backtest_results') or self.backtest_results is None:
            logger.warning("No backtest results available for detailed analysis")
            return

        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        import pandas as pd
        import json
        from sklearn.metrics import (
            classification_report, confusion_matrix, precision_recall_curve, auc,
            roc_auc_score, f1_score, precision_score, recall_score, roc_curve
        )

        results = self.backtest_results

        # Only consider trades (where a signal was given)
        trade_mask = (results['signal_long'] == 1) | (results['signal_short'] == 1)
        y_true = results.loc[trade_mask, 'label']
        y_pred = ((results.loc[trade_mask, 'prediction'] >= self.optimal_threshold.get('long', 0.55)).astype(int))
        y_score = results.loc[trade_mask, 'prediction']

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        # --- Classification report ---
        report_dict = classification_report(y_true, y_pred, digits=4, output_dict=True)
        report_str = classification_report(y_true, y_pred, digits=4)
        with open(os.path.join(output_dir, "classification_report.json"), "w") as f:
            json.dump(report_dict, f, indent=2)
        with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
            f.write(report_str)

        # --- Confusion matrix ---
        cm = confusion_matrix(y_true, y_pred)
        pd.DataFrame(cm).to_csv(os.path.join(output_dir, "confusion_matrix.csv"), index=False)
        np.savetxt(os.path.join(output_dir, "confusion_matrix.txt"), cm, fmt='%d')
        # Save confusion matrix as PNG
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix (on trades)')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
        plt.close()

        # --- Precision, Recall, F1, Support (save as JSON) ---
        metrics_summary = {
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "support": int(len(y_true))
        }
        with open(os.path.join(output_dir, "main_classification_metrics.json"), "w") as f:
            json.dump(metrics_summary, f, indent=2)

        # --- Precision-Recall Curve and AUC ---
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        pr_auc = auc(recall, precision)
        pr_curve_df = pd.DataFrame({'precision': precision, 'recall': recall})
        pr_curve_df.to_csv(os.path.join(output_dir, "precision_recall_curve.csv"), index=False)
        with open(os.path.join(output_dir, "pr_auc.txt"), "w") as f:
            f.write(str(pr_auc))
        # Plot PR curve
        plt.figure()
        plt.plot(recall, precision, label=f'PR AUC={pr_auc:.4f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "precision_recall_curve.png"))
        plt.close()

        # --- ROC AUC ---
        roc_auc = roc_auc_score(y_true, y_score)
        with open(os.path.join(output_dir, "roc_auc.txt"), "w") as f:
            f.write(str(roc_auc))
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_score)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC AUC={roc_auc:.4f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "roc_curve.png"))
        plt.close()

        # --- Feature importance ---
        if hasattr(self, 'feature_importance') and self.feature_importance is not None:
            self.feature_importance.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)
            # Plot feature importance
            plt.figure(figsize=(10, 8))
            top_features = self.feature_importance.head(20)
            sns.barplot(y='Feature', x='Importance', data=top_features, palette='viridis')
            plt.title('Top 20 Features by Importance')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "feature_importance.png"))
            plt.close()

        # --- Trade return distribution ---
        trade_returns = results.loc[trade_mask, 'trade_return']
        if len(trade_returns) > 0:
            plt.figure(figsize=(10, 6))
            sns.histplot(trade_returns, kde=True, bins=50)
            plt.axvline(0, color='r', linestyle='--')
            plt.title('Trade Return Distribution')
            plt.xlabel('Return')
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(output_dir, "trade_return_distribution.png"))
            plt.close()

        # --- Cumulative P&L and Drawdown ---
        plt.figure(figsize=(12, 6))
        plt.plot(results['cum_pnl'], label='Cumulative P&L')
        plt.fill_between(range(len(results)), -results['drawdown'], 0, alpha=0.3, color='red', label='Drawdown')
        plt.title('Cumulative P&L and Drawdown')
        plt.xlabel('Trade #')
        plt.ylabel('P&L')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "cumulative_pnl_and_drawdown.png"))
        plt.close()

        # --- Save all main backtest metrics as JSON (convert numpy types) ---
        def convert_np(obj):
            if isinstance(obj, dict):
                return {k: convert_np(v) for k, v in obj.items()}
            elif isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            else:
                return obj

        if hasattr(self, "backtest_metrics") and self.backtest_metrics:
            with open(os.path.join(output_dir, "backtest_metrics.json"), "w") as f:
                json.dump(convert_np(self.backtest_metrics), f, indent=2)


# Example usage
if __name__ == "__main__":
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Price Direction Prediction Model")
    parser.add_argument("--data", type=str, required=True, help="Path to feature data CSV/Parquet file")
    parser.add_argument("--output", type=str, default=None, help="Output directory for results")
    parser.add_argument("--horizon", type=int, default=60, help="Forecast horizon in seconds")
    parser.add_argument("--cost", type=float, default=0.0006, help="Transaction cost fraction")
    parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration if available")
    
    args = parser.parse_args()
    
    # Load feature data
    try:
        if args.data.endswith('.parquet'):
            data = pd.read_parquet(args.data)
        elif args.data.endswith('.csv'):
            data = pd.read_csv(args.data)
        else:
            print(f"Unsupported file format: {args.data}")
            sys.exit(1)
            
        print(f"Loaded {len(data)} samples from {args.data}")
        
        # Create predictor instance
        predictor = PriceDirectionPredictor(
            forecast_horizon=args.horizon,
            transaction_cost=args.cost,
            use_gpu=args.gpu
        )
        
        # Run pipeline
        results = predictor.run_pipeline(data, args.output)
        
        # Print results
        if results['status'] == 'success':
            metrics = results.get('metrics', {})
            print("\n===== RESULTS =====")
            print(f"Total trades: {metrics.get('total_trades', 0)}")
            print(f"Win rate: {metrics.get('win_rate', 0):.2%}")
            print(f"Profit factor: {metrics.get('profit_factor', 0):.4f}")
            print(f"Final P&L: {metrics.get('final_pnl', 0):.6f}")
            print(f"Sharpe ratio: {metrics.get('sharpe_ratio', 0):.4f}")
            print(f"Results saved to: {results['output_dir']}")
        else:
            print(f"Pipeline failed: {results.get('error', 'Unknown error')}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
    
