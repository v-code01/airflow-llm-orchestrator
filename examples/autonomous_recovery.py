"""
Examples demonstrating self-healing capabilities
"""

import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from airflow_llm.decorators import (
    cost_aware_execution,
    intelligent_retry,
    self_healing_task,
)

logger = logging.getLogger(__name__)

default_args = {
    "owner": "airflow-llm",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
}

dag = DAG(
    "self_healing_examples",
    default_args=default_args,
    description="Examples of self-healing tasks",
    schedule_interval=timedelta(hours=1),
    catchup=False,
    tags=["examples", "self-healing"],
)


@self_healing_task(retries=3, auto_fix=True, resource_scaling=True)
def memory_intensive_task(**context):
    """
    Task that may fail due to memory issues - will auto-scale
    """
    logger.info("Starting memory-intensive processing")

    import numpy as np

    large_array = np.random.random((10000, 10000))
    np.linalg.svd(large_array)

    logger.info("Memory-intensive task completed successfully")
    return {"status": "success", "array_shape": large_array.shape}


@self_healing_task(retries=5, auto_fix=True)
def dependency_error_task(**context):
    """
    Task that may fail due to missing dependencies - will auto-install
    """
    logger.info("Starting task with potential dependency issues")

    try:
        pass

        logger.info("All dependencies available")
        return {"status": "success", "dependencies": ["pandas", "scikit-learn"]}

    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        raise


@self_healing_task(retries=3, auto_fix=True)
@cost_aware_execution(max_cost_per_hour=5.0, prefer_spot_instances=True)
def api_connection_task(**context):
    """
    Task that may fail due to API connection issues - will implement backoff
    """
    logger.info("Attempting API connection")

    import requests

    response = requests.get("https://api.example.com/data", timeout=10)

    if response.status_code == 429:
        logger.warning("Rate limited, will be handled by self-healing")
        raise Exception("Rate limit exceeded")

    logger.info("API connection successful")
    return {"status": "success", "response_code": response.status_code}


@self_healing_task(retries=2, auto_fix=True, resource_scaling=True)
@intelligent_retry(max_retries=3, backoff_factor=2.0)
def gpu_computation_task(**context):
    """
    GPU computation that may need resource adjustment
    """
    logger.info("Starting GPU computation")

    import torch

    if not torch.cuda.is_available():
        logger.warning("CUDA not available, switching to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")

    tensor = torch.randn(1000, 1000, device=device)
    result = torch.matmul(tensor, tensor.T)

    logger.info("GPU computation completed")
    return {"status": "success", "device": str(device), "result_shape": result.shape}


@self_healing_task(retries=4, auto_fix=True)
def file_permission_task(**context):
    """
    Task that may fail due to file permission issues
    """
    logger.info("Starting file operations")

    import os
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("Test data for self-healing example")
        temp_file = f.name

    try:
        os.chmod(temp_file, 0o444)

        with open(temp_file, "a") as f:
            f.write("Additional data")

    except PermissionError as e:
        logger.error(f"Permission error: {e}")
        raise
    finally:
        os.unlink(temp_file)

    logger.info("File operations completed")
    return {"status": "success", "file_operations": "completed"}


memory_task = PythonOperator(
    task_id="memory_intensive_task",
    python_callable=memory_intensive_task,
    dag=dag,
)

dependency_task = PythonOperator(
    task_id="dependency_error_task",
    python_callable=dependency_error_task,
    dag=dag,
)

api_task = PythonOperator(
    task_id="api_connection_task",
    python_callable=api_connection_task,
    dag=dag,
)

gpu_task = PythonOperator(
    task_id="gpu_computation_task",
    python_callable=gpu_computation_task,
    dag=dag,
)

permission_task = PythonOperator(
    task_id="file_permission_task",
    python_callable=file_permission_task,
    dag=dag,
)

memory_task >> dependency_task >> api_task >> gpu_task >> permission_task
