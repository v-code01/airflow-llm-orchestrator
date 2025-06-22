"""
Examples demonstrating cost optimization capabilities
"""

import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from airflow_llm.decorators import cost_aware_execution, performance_monitor

logger = logging.getLogger(__name__)

default_args = {
    "owner": "airflow-llm",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "cost_optimization_examples",
    default_args=default_args,
    description="Examples of cost-aware task execution",
    schedule_interval=timedelta(hours=6),
    catchup=False,
    tags=["examples", "cost-optimization"],
)


@cost_aware_execution(
    max_cost_per_hour=2.0,
    prefer_spot_instances=True,
    multi_cloud=True,
    performance_weight=0.3,
)
@performance_monitor(
    track_memory=True,
    track_cpu=True,
    alert_thresholds={
        "max_execution_time": 3600,
        "max_memory_mb": 8192,
        "max_cpu_percent": 90,
    },
)
def batch_data_processing(**context):
    """
    Large batch processing job optimized for cost
    """
    logger.info("Starting cost-optimized batch processing")

    import numpy as np

    logger.info("Processing large dataset with cost optimization")

    data_size = 1000000
    data = np.random.random(data_size)

    logger.info(f"Generated dataset of size {data_size}")

    result = np.fft.fft(data)
    processed_data = np.abs(result)

    logger.info("FFT processing completed")

    summary_stats = {
        "mean": float(np.mean(processed_data)),
        "std": float(np.std(processed_data)),
        "max": float(np.max(processed_data)),
        "min": float(np.min(processed_data)),
    }

    logger.info(f"Processing complete: {summary_stats}")

    return {"status": "success", "data_size": data_size, "summary_stats": summary_stats}


@cost_aware_execution(
    max_cost_per_hour=10.0,
    prefer_spot_instances=True,
    multi_cloud=True,
    performance_weight=0.1,
)
def ml_model_training(**context):
    """
    ML model training with aggressive cost optimization
    """
    logger.info("Starting cost-optimized ML training")

    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    logger.info("Generating synthetic training data")

    n_samples = 50000
    n_features = 100

    X = np.random.random((n_samples, n_features))
    y = np.random.randint(0, 2, n_samples)

    logger.info(f"Generated dataset: {X.shape}, {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    logger.info("Training Random Forest model")

    model = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )

    model.fit(X_train, y_train)

    logger.info("Model training completed")

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    logger.info(f"Model accuracy: {accuracy:.4f}")

    return {
        "status": "success",
        "model_type": "RandomForestClassifier",
        "accuracy": accuracy,
        "training_samples": len(X_train),
        "test_samples": len(X_test),
    }


@cost_aware_execution(
    max_cost_per_hour=1.0,
    prefer_spot_instances=True,
    multi_cloud=True,
    performance_weight=0.5,
)
def data_validation_task(**context):
    """
    Data validation with strict cost constraints
    """
    logger.info("Starting cost-constrained data validation")

    import numpy as np
    import pandas as pd

    logger.info("Creating test dataset for validation")

    data = {
        "id": range(10000),
        "value1": np.random.normal(100, 15, 10000),
        "value2": np.random.exponential(2, 10000),
        "category": np.random.choice(["A", "B", "C"], 10000),
        "timestamp": pd.date_range("2024-01-01", periods=10000, freq="1min"),
    }

    df = pd.DataFrame(data)

    logger.info(f"Created dataset with shape: {df.shape}")

    validation_results = {
        "null_counts": df.isnull().sum().to_dict(),
        "data_types": df.dtypes.astype(str).to_dict(),
        "value_ranges": {
            "value1": {"min": df["value1"].min(), "max": df["value1"].max()},
            "value2": {"min": df["value2"].min(), "max": df["value2"].max()},
        },
        "category_counts": df["category"].value_counts().to_dict(),
    }

    logger.info("Data validation completed")
    logger.info(f"Validation results: {validation_results}")

    return {
        "status": "success",
        "dataset_shape": df.shape,
        "validation_results": validation_results,
    }


@cost_aware_execution(
    max_cost_per_hour=5.0,
    prefer_spot_instances=False,
    multi_cloud=True,
    performance_weight=0.8,
)
def real_time_inference(**context):
    """
    Real-time inference with performance priority over cost
    """
    logger.info("Starting performance-optimized inference")

    import time

    import numpy as np

    logger.info("Simulating real-time inference workload")

    batch_size = 1000
    num_batches = 50
    inference_times = []

    for batch_idx in range(num_batches):
        start_time = time.time()

        batch_data = np.random.random((batch_size, 50))

        result = np.sum(batch_data, axis=1)
        result > np.median(result)

        end_time = time.time()
        batch_time = end_time - start_time
        inference_times.append(batch_time)

        if batch_idx % 10 == 0:
            logger.info(f"Processed batch {batch_idx}, time: {batch_time:.4f}s")

    avg_inference_time = np.mean(inference_times)
    p95_inference_time = np.percentile(inference_times, 95)

    logger.info(
        f"Inference completed - Avg: {avg_inference_time:.4f}s, P95: {p95_inference_time:.4f}s"
    )

    return {
        "status": "success",
        "total_batches": num_batches,
        "batch_size": batch_size,
        "avg_inference_time": avg_inference_time,
        "p95_inference_time": p95_inference_time,
    }


batch_processing_task = PythonOperator(
    task_id="batch_data_processing",
    python_callable=batch_data_processing,
    dag=dag,
)

ml_training_task = PythonOperator(
    task_id="ml_model_training",
    python_callable=ml_model_training,
    dag=dag,
)

validation_task = PythonOperator(
    task_id="data_validation_task",
    python_callable=data_validation_task,
    dag=dag,
)

inference_task = PythonOperator(
    task_id="real_time_inference",
    python_callable=real_time_inference,
    dag=dag,
)

[batch_processing_task, validation_task] >> ml_training_task >> inference_task
