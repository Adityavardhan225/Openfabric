

import os
import sys
import logging
import time
import pandas as pd
from datetime import datetime
import gc
import dask
import psutil
import signal
import threading
import argparse
import shutil
import json
from dask.distributed import Client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pipeline.log', mode='w')
    ]
)
logger = logging.getLogger('ETH_Pipeline')

from feature_eng import ETHFeatureBuilder
from prediction import PriceDirectionPredictor

monitoring_active = True

def monitor_memory(interval=30):
    while monitoring_active:
        try:
            mem = psutil.virtual_memory()
            if mem.percent > 85:
                logger.warning(f"HIGH MEMORY WARNING: System memory usage is at {mem.percent}%")
            swap = psutil.swap_memory()
            if swap.percent > 50:
                logger.warning(f"HIGH SWAP USAGE: {swap.percent}%. System performance will be degraded.")
        except:
            pass
        time.sleep(interval)

def kill_dask_processes():
    try:
        logger.info("Ensuring all lingering Dask processes are terminated...")
        killed_count = 0
        current_pid = os.getpid()
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            if proc.pid == current_pid: continue
            try:
                if proc.info['name'] and 'python' in proc.info['name'].lower():
                    cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                    if 'dask-worker' in cmdline or 'dask-scheduler' in cmdline:
                        logger.warning(f"Forcefully terminating lingering Dask process PID: {proc.info['pid']}")
                        proc.kill()
                        killed_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if killed_count > 0:
            logger.info(f"Terminated {killed_count} lingering Dask processes.")
            time.sleep(2)
    except Exception as e:
        logger.error(f"Error during final process cleanup: {str(e)}")

def batch_process_prediction(df, func, batch_size=50000):
    if df is None or df.empty:
        logger.error("Prediction step received an empty feature matrix. Aborting.")
        return {"status": "error", "error": "Empty feature matrix."}
    result_pieces = []
    total_rows = len(df)
    logger.info(f"Starting prediction phase on {total_rows} feature rows...")
    for start_idx in range(0, total_rows, batch_size):
        end_idx = min(start_idx + batch_size, total_rows)
        logger.info(f"Running prediction on batch: rows {start_idx} to {end_idx}")
        batch = df.iloc[start_idx:end_idx]
        result = func(batch)
        if result.get("status") == "success":
            result_pieces.append(result["predictions"])
        else:
            logger.error(f"Prediction failed for batch {start_idx}-{end_idx}. Details: {result.get('error')}")
        del batch
        gc.collect()
    if result_pieces:
        all_predictions = pd.concat(result_pieces, ignore_index=True)
        return {"status": "success", "predictions": all_predictions}
    else:
        return {"status": "error", "error": "All prediction batches failed."}

def run_pipeline(tick_data_path, output_dir, delta_t, forecast_horizon, transaction_cost, use_gpu):
    global monitoring_active
    start_time = time.time()
    logger.info("--- Starting ETH Price Prediction Pipeline ---")

    monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
    monitor_thread.start()

    temp_dir = '/tmp/dask_pipeline_temp'
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    logger.info(f"Forcing Dask temporary directory to: {temp_dir}")

    client = None

    try:
        logger.info("--- Step 1: Feature Engineering using Dask ---")
        n_workers = 4
        memory_limit_per_worker = "4GB"

        dask_config = {
            'distributed.worker.memory.target': 0.60,
            'distributed.worker.memory.spill': 0.75,
            'distributed.worker.memory.pause': 0.85,
            'temporary_directory': temp_dir,
            'distributed.worker.memory.terminate': False
        }

        logger.info(f"Using Dask config: {n_workers} workers, {memory_limit_per_worker}/worker.")

        with dask.config.set(dask_config):
            client = Client(
                n_workers=n_workers,
                threads_per_worker=1,
                memory_limit=memory_limit_per_worker,
                processes=True
            )

            feature_builder = ETHFeatureBuilder(delta_t=delta_t)
            feature_builder.set_dask_client(client)
            feature_builder.load_tick_data(tick_data_path)

            logger.info("Building feature matrix... This is a long-running process.")
            feature_matrix = feature_builder.build_feature_matrix()
            logger.info(f"Feature matrix built with shape: {feature_matrix.shape}")

            output_path = os.path.join(output_dir, "feature_matrix_after_eng.parquet")
            logger.info(f"Saving feature matrix directly to: {output_path}")
            feature_matrix.to_parquet(output_path, index=False)
            logger.info(f"File exists after save? {os.path.exists(output_path)}")

            if feature_matrix is None or feature_matrix.empty:
                raise RuntimeError("Feature engineering failed to produce any data.")

            logger.info(f"Feature engineering successful. Matrix shape: {feature_matrix.shape}")

        logger.info("--- Step 2: Price Direction Prediction ---")
        logger.info("Cleaning up Dask resources before starting prediction phase...")
        feature_matrix_copy = feature_matrix.copy()
        del feature_matrix

        feature_builder.cleanup()
        client.close()
        client = None
        del feature_builder

        gc.collect()
        kill_dask_processes()
        logger.info("Dask resources released. Pausing 5s for OS to stabilize...")
        time.sleep(5)

        predictor = PriceDirectionPredictor(
            forecast_horizon=forecast_horizon,
            transaction_cost=transaction_cost,
            use_gpu=use_gpu
        )
        # Step 1: Process features
        processed_data = predictor.process_features(feature_matrix_copy)
        train_data = processed_data.iloc[:int(0.8*len(processed_data))]
        test_data = processed_data.iloc[int(0.8*len(processed_data)):]

        # Step 2: Train initial model to get feature importance
        predictor.train_model(train_data, test_data)

        # Step 3: Select top N features by importance
        top_n = 15
        top_features = predictor.feature_importance['Feature'].iloc[:top_n].tolist()
        selected_cols = top_features + ['label', 'timestamp','return_pct']
        train_data_top = train_data[selected_cols]
        test_data_top = test_data[selected_cols]

        # Step 4: Update predictor to use only top N features
        predictor.norm_feature_cols = top_features

        # Step 5: Retrain model using only top N features
        predictor.train_model(train_data_top, test_data_top)

        # Save model
        model_path = os.path.join(output_dir, "price_direction_model.txt")
        if predictor.model is not None:
            predictor.model.save_model(model_path)
            logger.info(f"Model saved to {model_path}")

        # Predict and save predictions
        results = batch_process_prediction(
            test_data,
            lambda batch: {"status": "success", "predictions": predictor.predict(batch)},
            batch_size=50000
        )

        if results.get("status") == "success":
            predictions = results["predictions"]
            pred_path = os.path.join(output_dir, "predictions.parquet")
            predictions.to_parquet(pred_path, index=False)
            logger.info(f"Predictions saved to {pred_path}")

            # Backtest and save metrics
            backtest_results = predictor.backtest(test_data)
            import numpy as np

            metrics = predictor.backtest_metrics if hasattr(predictor, "backtest_metrics") else {}
            metrics_path = os.path.join(output_dir, "metrics.json")

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

            with open(metrics_path, "w") as f:
                json.dump(convert_np(metrics), f, indent=2)
            logger.info(f"Metrics saved to {metrics_path}")

            predictor.detailed_analysis(output_dir=output_dir)

        elapsed_time_hours = (time.time() - start_time) / 3600
        logger.info(f"--- Pipeline finished in {elapsed_time_hours:.2f} hours ---")
        logger.info(f"All models and results have been saved to: {output_dir}")

        return {"status": "success", "metrics": metrics}

    except Exception as e:
        logger.critical(f"A FATAL ERROR occurred in the pipeline: {e}", exc_info=True)
        return {"status": "error", "error": str(e)}

    finally:
        monitoring_active = False
        logger.info("Initiating final resource cleanup sequence...")
        if client:
            try:
                client.close()
            except Exception:
                pass
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"Successfully removed temporary directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Could not remove temp directory {temp_dir}: {e}")
        kill_dask_processes()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the end-to-end ETH price prediction pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data", type=str, required=True, help="Path to the large tick data CSV file.")
    parser.add_argument("--output", type=str, default=None, help="Output directory for results. If not set, a timestamped directory is created.")
    parser.add_argument("--delta_t", type=int, default=5, help="Feature sampling interval in seconds.")
    parser.add_argument("--horizon", type=int, default=60, help="Prediction forecast horizon in seconds.")
    parser.add_argument("--cost", type=float, default=0.0006, help="Backtesting transaction cost.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU-only mode for the entire pipeline.")

    args = parser.parse_args()

    try:
        p = psutil.Process(os.getpid())
        p.nice(psutil.NORMAL_PRIORITY_CLASS if os.name == 'nt' else 0)
    except Exception:
        pass

    def handle_signal(signum, frame):
        logger.critical(f"--- INTERRUPT SIGNAL {signum} RECEIVED. INITIATING EMERGENCY SHUTDOWN. ---")
        kill_dask_processes()
        sys.exit(1)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    output_directory = args.output
    if output_directory is None:
        output_directory = f"pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_directory, exist_ok=True)
    logger.info(f"Results will be saved in: {output_directory}")

    results = run_pipeline(
        tick_data_path=args.data,
        output_dir=output_directory,
        delta_t=args.delta_t,
        forecast_horizon=args.horizon,
        transaction_cost=0,
        use_gpu=(not args.cpu),
    )

    if results and results.get("status") == "success":
        metrics = results.get("metrics", {})
        print("\n" + "="*29)
        print("    ✅ PIPELINE SUCCESSFUL")
        print(f"Results saved in: {output_directory}")
        print(f"Metrics: {metrics}")
        sys.exit(0)
    else:
        logger.critical("Pipeline finished with an error. Please check 'pipeline.log' for details.")
        sys.exit(1)