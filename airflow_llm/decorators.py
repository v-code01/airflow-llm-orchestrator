"""
Decorators for natural language DAG generation and self-healing tasks
"""

import functools
import logging
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.utils.dates import days_ago

    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False
    DAG = None
    PythonOperator = None

    def days_ago(n):
        return datetime.now() - timedelta(days=n)


from .cost_optimizer import CostAwareScheduler
from .natural_language_processor import NaturalLanguageDAGGenerator
from .self_healer import SelfHealingAgent

logger = logging.getLogger(__name__)


def natural_language_dag(
    description: str,
    dag_id: str | None = None,
    schedule_interval: str = "@daily",
    start_date: datetime | None = None,
    catchup: bool = False,
    max_active_runs: int = 1,
    cost_optimization: bool = True,
    self_healing: bool = True,
    **dag_kwargs,
):
    """
    Decorator to generate DAG from natural language description
    """

    def decorator(func: Callable) -> DAG:
        if not AIRFLOW_AVAILABLE:
            raise RuntimeError(
                "Apache Airflow is required for DAG generation. Please install airflow: pip install apache-airflow"
            )

        logger.info(f"Generating DAG from description: {description[:100]}...")

        actual_dag_id = dag_id or func.__name__
        actual_start_date = start_date or days_ago(1)

        default_args = {
            "owner": "airflow-llm",
            "depends_on_past": False,
            "start_date": actual_start_date,
            "email_on_failure": True,
            "email_on_retry": False,
            "retries": 2 if self_healing else 1,
            "retry_delay": timedelta(minutes=5),
        }

        dag = DAG(
            actual_dag_id,
            default_args=default_args,
            description=f"AI-generated: {description}",
            schedule_interval=schedule_interval,
            catchup=catchup,
            max_active_runs=max_active_runs,
            tags=["ai-generated", "llm-orchestrated"],
            **dag_kwargs,
        )

        if cost_optimization:
            logger.debug("Cost optimization enabled for DAG")
            dag.cost_optimizer = CostAwareScheduler()

        if self_healing:
            logger.debug("Self-healing enabled for DAG")
            dag.self_healer = SelfHealingAgent()

        try:
            nl_generator = NaturalLanguageDAGGenerator(llm_client=None)
            dag_code = nl_generator.generate(description, actual_dag_id)

            logger.info(f"Successfully generated DAG: {actual_dag_id}")

            exec(dag_code, {"dag": dag})

        except Exception as e:
            logger.error(f"Failed to generate DAG from description: {e}")

            fallback_task = PythonOperator(
                task_id="fallback_task",
                python_callable=lambda: logger.info("Fallback task executed"),
                dag=dag,
            )

        return dag

    return decorator


def self_healing_task(
    retries: int = 3,
    retry_delay: timedelta = timedelta(minutes=5),
    auto_fix: bool = True,
    resource_scaling: bool = True,
    **task_kwargs,
):
    """
    Decorator to add self-healing capabilities to task functions
    """

    def decorator(func: Callable) -> Callable:
        logger.debug(f"Adding self-healing to task: {func.__name__}")

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            context = kwargs.get("context", {})
            task_instance = context.get("task_instance")

            healer = SelfHealingAgent(enable_auto_fix=auto_fix, max_retries=retries)

            attempt = 0
            while attempt < retries:
                try:
                    logger.info(
                        f"Executing task {func.__name__}, attempt {attempt + 1}"
                    )

                    result = func(*args, **kwargs)

                    logger.info(f"Task {func.__name__} completed successfully")
                    return result

                except Exception as e:
                    attempt += 1

                    logger.warning(
                        f"Task {func.__name__} failed on attempt {attempt}: {e}"
                    )

                    if attempt >= retries:
                        logger.error(f"Task {func.__name__} exhausted all retries")
                        raise

                    try:
                        analysis = healer.analyze_error(e, context)

                        if analysis.auto_fixable:
                            logger.info(
                                f"Attempting auto-fix for {func.__name__}: "
                                f"{analysis.suggested_fix}"
                            )

                            success = healer.attempt_fix(analysis, context)

                            if success:
                                logger.info(f"Auto-fix successful for {func.__name__}")
                                if resource_scaling and analysis.resource_adjustment:
                                    _apply_resource_scaling(
                                        task_instance, analysis.resource_adjustment
                                    )
                                continue
                            else:
                                logger.warning(f"Auto-fix failed for {func.__name__}")

                    except Exception as heal_error:
                        logger.error(f"Error during self-healing attempt: {heal_error}")

                    import time

                    time.sleep(retry_delay.total_seconds())

            raise RuntimeError(f"Task {func.__name__} failed after all retries")

        return wrapper

    return decorator


def cost_aware_execution(
    max_cost_per_hour: float = 10.0,
    prefer_spot_instances: bool = True,
    multi_cloud: bool = True,
    performance_weight: float = 0.3,
):
    """
    Decorator for cost-aware task execution
    """

    def decorator(func: Callable) -> Callable:
        logger.debug(f"Adding cost-aware execution to: {func.__name__}")

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            context = kwargs.get("context", {})

            scheduler = CostAwareScheduler(
                max_cost_per_hour=max_cost_per_hour,
                enable_spot_instances=prefer_spot_instances,
                enable_multi_cloud=multi_cloud,
                cost_vs_performance_weight=1.0 - performance_weight,
            )

            resource_requirements = _extract_resource_requirements(func, context)

            try:
                optimization = scheduler.optimize_resources(
                    resource_requirements=resource_requirements,
                    performance_targets={"max_latency": 300},
                    cost_constraints={"max_total_cost": max_cost_per_hour * 2},
                )

                logger.info(
                    f"Cost optimization for {func.__name__}: "
                    f"Selected {optimization.selected_quote.provider.value} "
                    f"({optimization.selected_quote.instance_type.value}), "
                    f"Cost: ${optimization.selected_quote.hourly_cost:.2f}/hr, "
                    f"Savings: {optimization.cost_savings:.1f}%"
                )

                _apply_optimized_resources(context, optimization.selected_quote)

            except Exception as e:
                logger.warning(f"Cost optimization failed: {e}, using defaults")

            return func(*args, **kwargs)

        return wrapper

    return decorator


def intelligent_retry(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    circuit_breaker: bool = True,
):
    """
    Decorator for intelligent retry with exponential backoff
    """

    def decorator(func: Callable) -> Callable:
        logger.debug(f"Adding intelligent retry to: {func.__name__}")

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import random
            import time

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)

                except Exception as e:
                    if attempt == max_retries:
                        logger.error(
                            f"Task {func.__name__} failed after "
                            f"{max_retries} attempts: {e}"
                        )
                        raise

                    delay = backoff_factor**attempt
                    if jitter:
                        delay *= 0.5 + random.random() * 0.5

                    logger.warning(
                        f"Task {func.__name__} attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.1f}s"
                    )

                    time.sleep(delay)

        return wrapper

    return decorator


def performance_monitor(
    track_memory: bool = True,
    track_cpu: bool = True,
    track_network: bool = False,
    alert_thresholds: dict[str, float] | None = None,
):
    """
    Decorator to monitor task performance metrics
    """

    def decorator(func: Callable) -> Callable:
        logger.debug(f"Adding performance monitoring to: {func.__name__}")

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import time

            import psutil

            start_time = time.time()
            start_memory = psutil.virtual_memory().used if track_memory else 0
            start_cpu = psutil.cpu_percent() if track_cpu else 0

            logger.info(f"Starting performance monitoring for {func.__name__}")

            try:
                result = func(*args, **kwargs)

                end_time = time.time()
                execution_time = end_time - start_time

                if track_memory:
                    memory_used = psutil.virtual_memory().used - start_memory
                    logger.info(
                        f"Task {func.__name__} memory usage: "
                        f"{memory_used / 1024 / 1024:.1f} MB"
                    )

                if track_cpu:
                    avg_cpu = (start_cpu + psutil.cpu_percent()) / 2
                    logger.info(f"Task {func.__name__} avg CPU: {avg_cpu:.1f}%")

                logger.info(f"Task {func.__name__} completed in {execution_time:.2f}s")

                if alert_thresholds:
                    _check_performance_thresholds(
                        func.__name__,
                        execution_time,
                        memory_used if track_memory else 0,
                        avg_cpu if track_cpu else 0,
                        alert_thresholds,
                    )

                return result

            except Exception as e:
                logger.error(f"Task {func.__name__} failed: {e}")
                raise

        return wrapper

    return decorator


def _apply_resource_scaling(task_instance, resource_adjustments: dict[str, Any]):
    """
    Apply resource scaling adjustments to task instance
    """
    logger.info(f"Applying resource scaling: {resource_adjustments}")

    if hasattr(task_instance, "executor_config"):
        config = task_instance.executor_config or {}
        config.update(resource_adjustments)
        task_instance.executor_config = config


def _extract_resource_requirements(func: Callable, context: dict) -> dict[str, Any]:
    """
    Extract resource requirements from function and context
    """
    requirements = {"cpu": 2, "memory": "4Gi", "estimated_runtime": 3600}

    if hasattr(func, "__resource_requirements__"):
        requirements.update(func.__resource_requirements__)

    task_instance = context.get("task_instance")
    if task_instance and hasattr(task_instance, "executor_config"):
        config = task_instance.executor_config or {}
        requirements.update(config)

    return requirements


def _apply_optimized_resources(context: dict, quote):
    """
    Apply optimized resource configuration
    """
    task_instance = context.get("task_instance")
    if task_instance:
        config = {
            "cpu": quote.cpu,
            "memory": quote.memory,
            "provider": quote.provider.value,
            "instance_type": quote.instance_type.value,
        }

        if quote.gpu:
            config["gpu"] = quote.gpu

        task_instance.executor_config = config


def _check_performance_thresholds(
    task_name: str,
    execution_time: float,
    memory_used: float,
    cpu_usage: float,
    thresholds: dict[str, float],
):
    """
    Check if performance metrics exceed thresholds
    """
    if "max_execution_time" in thresholds:
        if execution_time > thresholds["max_execution_time"]:
            logger.warning(
                f"Task {task_name} execution time ({execution_time:.2f}s) "
                f"exceeded threshold ({thresholds['max_execution_time']}s)"
            )

    if "max_memory_mb" in thresholds:
        memory_mb = memory_used / 1024 / 1024
        if memory_mb > thresholds["max_memory_mb"]:
            logger.warning(
                f"Task {task_name} memory usage ({memory_mb:.1f}MB) "
                f"exceeded threshold ({thresholds['max_memory_mb']}MB)"
            )

    if "max_cpu_percent" in thresholds:
        if cpu_usage > thresholds["max_cpu_percent"]:
            logger.warning(
                f"Task {task_name} CPU usage ({cpu_usage:.1f}%) "
                f"exceeded threshold ({thresholds['max_cpu_percent']}%)"
            )
