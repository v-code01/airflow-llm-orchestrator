"""
Unit tests for decorators
"""

import unittest.mock as mock

import pytest
from airflow import DAG

from airflow_llm.decorators import (
    cost_aware_execution,
    intelligent_retry,
    natural_language_dag,
    performance_monitor,
    self_healing_task,
)


class TestNaturalLanguageDAG:
    @mock.patch("airflow_llm.decorators.NaturalLanguageDAGGenerator")
    def test_natural_language_dag_decorator(self, mock_generator_class):
        mock_generator = mock.MagicMock()
        mock_generator.generate.return_value = "dag = DAG('test')"
        mock_generator_class.return_value = mock_generator

        @natural_language_dag("Process data and train model")
        def test_dag():
            pass

        assert isinstance(test_dag, DAG)
        assert test_dag.dag_id == "test_dag"
        assert "AI-generated" in test_dag.description
        assert "ai-generated" in test_dag.tags
        assert "llm-orchestrated" in test_dag.tags
        mock_generator.generate.assert_called_once()

    @mock.patch("airflow_llm.decorators.NaturalLanguageDAGGenerator")
    def test_natural_language_dag_with_custom_params(self, mock_generator_class):
        mock_generator = mock.MagicMock()
        mock_generator.generate.return_value = "dag = DAG('custom')"
        mock_generator_class.return_value = mock_generator

        @natural_language_dag(
            "Custom pipeline",
            dag_id="custom_dag",
            schedule_interval="@hourly",
            catchup=True,
            cost_optimization=False,
            self_healing=False,
        )
        def custom_dag():
            pass

        assert custom_dag.dag_id == "custom_dag"
        assert custom_dag.schedule_interval == "@hourly"
        assert custom_dag.catchup is True
        assert not hasattr(custom_dag, "cost_optimizer")
        assert not hasattr(custom_dag, "self_healer")

    @mock.patch("airflow_llm.decorators.NaturalLanguageDAGGenerator")
    def test_natural_language_dag_with_cost_optimization(self, mock_generator_class):
        mock_generator = mock.MagicMock()
        mock_generator.generate.return_value = "dag = DAG('test')"
        mock_generator_class.return_value = mock_generator

        @natural_language_dag("Test pipeline", cost_optimization=True)
        def test_dag():
            pass

        assert hasattr(test_dag, "cost_optimizer")

    @mock.patch("airflow_llm.decorators.NaturalLanguageDAGGenerator")
    def test_natural_language_dag_with_self_healing(self, mock_generator_class):
        mock_generator = mock.MagicMock()
        mock_generator.generate.return_value = "dag = DAG('test')"
        mock_generator_class.return_value = mock_generator

        @natural_language_dag("Test pipeline", self_healing=True)
        def test_dag():
            pass

        assert hasattr(test_dag, "self_healer")

    @mock.patch("airflow_llm.decorators.NaturalLanguageDAGGenerator")
    def test_natural_language_dag_generation_failure(self, mock_generator_class):
        mock_generator = mock.MagicMock()
        mock_generator.generate.side_effect = Exception("Generation failed")
        mock_generator_class.return_value = mock_generator

        @natural_language_dag("Failing pipeline")
        def failing_dag():
            pass

        assert isinstance(failing_dag, DAG)


class TestSelfHealingTask:
    def test_self_healing_task_decorator_success(self):
        @self_healing_task(retries=2, auto_fix=True)
        def successful_task(**context):
            return "success"

        result = successful_task(context={})
        assert result == "success"

    @mock.patch("airflow_llm.decorators.SelfHealingAgent")
    def test_self_healing_task_with_retry(self, mock_agent_class):
        mock_agent = mock.MagicMock()
        mock_analysis = mock.MagicMock()
        mock_analysis.auto_fixable = True
        mock_analysis.resource_adjustment = None
        mock_agent.analyze_error.return_value = mock_analysis
        mock_agent.attempt_fix.return_value = True
        mock_agent_class.return_value = mock_agent

        call_count = 0

        @self_healing_task(retries=3, auto_fix=True)
        def failing_then_succeeding_task(**context):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("First attempt fails")
            return "success"

        result = failing_then_succeeding_task(context={})

        assert result == "success"
        assert call_count == 2
        mock_agent.analyze_error.assert_called_once()
        mock_agent.attempt_fix.assert_called_once()

    @mock.patch("airflow_llm.decorators.SelfHealingAgent")
    def test_self_healing_task_exhausted_retries(self, mock_agent_class):
        mock_agent = mock.MagicMock()
        mock_analysis = mock.MagicMock()
        mock_analysis.auto_fixable = False
        mock_agent.analyze_error.return_value = mock_analysis
        mock_agent_class.return_value = mock_agent

        @self_healing_task(retries=2, auto_fix=True)
        def always_failing_task(**context):
            raise Exception("Always fails")

        with pytest.raises(RuntimeError, match="failed after all retries"):
            always_failing_task(context={})

    @mock.patch("airflow_llm.decorators.SelfHealingAgent")
    @mock.patch("airflow_llm.decorators._apply_resource_scaling")
    def test_self_healing_task_with_resource_scaling(
        self, mock_scaling, mock_agent_class
    ):
        mock_agent = mock.MagicMock()
        mock_analysis = mock.MagicMock()
        mock_analysis.auto_fixable = True
        mock_analysis.resource_adjustment = {"memory": "8Gi"}
        mock_agent.analyze_error.return_value = mock_analysis
        mock_agent.attempt_fix.return_value = True
        mock_agent_class.return_value = mock_agent

        call_count = 0

        @self_healing_task(retries=2, resource_scaling=True)
        def task_needing_scaling(**context):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise MemoryError("Out of memory")
            return "success"

        mock_task_instance = mock.MagicMock()
        context = {"task_instance": mock_task_instance}

        result = task_needing_scaling(context=context)

        assert result == "success"
        mock_scaling.assert_called_once_with(mock_task_instance, {"memory": "8Gi"})

    def test_self_healing_task_auto_fix_disabled(self):
        @self_healing_task(retries=1, auto_fix=False)
        def failing_task(**context):
            raise Exception("Task fails")

        with pytest.raises(RuntimeError, match="failed after all retries"):
            failing_task(context={})


class TestCostAwareExecution:
    @mock.patch("airflow_llm.decorators.CostAwareScheduler")
    def test_cost_aware_execution_decorator(self, mock_scheduler_class):
        mock_scheduler = mock.MagicMock()
        mock_optimization = mock.MagicMock()
        mock_optimization.selected_quote.provider.value = "aws"
        mock_optimization.selected_quote.instance_type.value = "spot"
        mock_optimization.selected_quote.hourly_cost = 1.50
        mock_optimization.cost_savings = 60.0
        mock_scheduler.optimize_resources.return_value = mock_optimization
        mock_scheduler_class.return_value = mock_scheduler

        @cost_aware_execution(max_cost_per_hour=5.0, prefer_spot_instances=True)
        def cost_optimized_task(**context):
            return "optimized"

        result = cost_optimized_task(context={})

        assert result == "optimized"
        mock_scheduler.optimize_resources.assert_called_once()

    @mock.patch("airflow_llm.decorators.CostAwareScheduler")
    def test_cost_aware_execution_optimization_failure(self, mock_scheduler_class):
        mock_scheduler = mock.MagicMock()
        mock_scheduler.optimize_resources.side_effect = Exception("Optimization failed")
        mock_scheduler_class.return_value = mock_scheduler

        @cost_aware_execution(max_cost_per_hour=5.0)
        def task_with_failed_optimization(**context):
            return "fallback"

        result = task_with_failed_optimization(context={})

        assert result == "fallback"

    @mock.patch("airflow_llm.decorators._extract_resource_requirements")
    @mock.patch("airflow_llm.decorators.CostAwareScheduler")
    def test_cost_aware_execution_resource_extraction(
        self, mock_scheduler_class, mock_extract
    ):
        mock_extract.return_value = {"cpu": 4, "memory": "8Gi"}
        mock_scheduler = mock.MagicMock()
        mock_optimization = mock.MagicMock()
        mock_scheduler.optimize_resources.return_value = mock_optimization
        mock_scheduler_class.return_value = mock_scheduler

        @cost_aware_execution()
        def task_with_resources(**context):
            return "success"

        task_with_resources(context={})

        mock_extract.assert_called_once()
        mock_scheduler.optimize_resources.assert_called_once()


class TestIntelligentRetry:
    def test_intelligent_retry_success_first_attempt(self):
        @intelligent_retry(max_retries=3)
        def successful_task():
            return "success"

        result = successful_task()
        assert result == "success"

    @mock.patch("time.sleep")
    def test_intelligent_retry_with_backoff(self, mock_sleep):
        call_count = 0

        @intelligent_retry(max_retries=2, backoff_factor=2.0, jitter=False)
        def failing_then_succeeding_task():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception(f"Attempt {call_count} fails")
            return "success"

        result = failing_then_succeeding_task()

        assert result == "success"
        assert call_count == 3
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(2.0)
        mock_sleep.assert_any_call(4.0)

    @mock.patch("time.sleep")
    @mock.patch("random.random", return_value=0.5)
    def test_intelligent_retry_with_jitter(self, mock_random, mock_sleep):
        call_count = 0

        @intelligent_retry(max_retries=1, backoff_factor=2.0, jitter=True)
        def failing_then_succeeding_task():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("First attempt fails")
            return "success"

        result = failing_then_succeeding_task()

        assert result == "success"
        expected_delay = 2.0 * 0.75
        mock_sleep.assert_called_once_with(expected_delay)

    def test_intelligent_retry_exhausted(self):
        @intelligent_retry(max_retries=2)
        def always_failing_task():
            raise Exception("Always fails")

        with pytest.raises(Exception, match="Always fails"):
            always_failing_task()


class TestPerformanceMonitor:
    @mock.patch("psutil.virtual_memory")
    @mock.patch("psutil.cpu_percent")
    @mock.patch("time.time")
    def test_performance_monitor_decorator(self, mock_time, mock_cpu, mock_memory):
        mock_time.side_effect = [0, 5]
        mock_memory.return_value.used = 1000000000
        mock_cpu.return_value = 50.0

        @performance_monitor(track_memory=True, track_cpu=True)
        def monitored_task():
            return "completed"

        result = monitored_task()

        assert result == "completed"
        assert mock_time.call_count >= 2
        mock_cpu.assert_called()
        mock_memory.assert_called()

    @mock.patch("psutil.virtual_memory")
    @mock.patch("psutil.cpu_percent")
    @mock.patch("time.time")
    @mock.patch("airflow_llm.decorators._check_performance_thresholds")
    def test_performance_monitor_with_thresholds(
        self, mock_check, mock_time, mock_cpu, mock_memory
    ):
        mock_time.side_effect = [0, 10]
        mock_memory.return_value.used = 2000000000
        mock_cpu.return_value = 80.0

        thresholds = {
            "max_execution_time": 5,
            "max_memory_mb": 1000,
            "max_cpu_percent": 70,
        }

        @performance_monitor(
            track_memory=True, track_cpu=True, alert_thresholds=thresholds
        )
        def monitored_task():
            return "completed"

        result = monitored_task()

        assert result == "completed"
        mock_check.assert_called_once()

    @mock.patch("psutil.virtual_memory")
    @mock.patch("psutil.cpu_percent")
    def test_performance_monitor_with_exception(self, mock_cpu, mock_memory):
        @performance_monitor()
        def failing_task():
            raise Exception("Task failed")

        with pytest.raises(Exception, match="Task failed"):
            failing_task()


class TestUtilityFunctions:
    def test_apply_resource_scaling(self):
        from airflow_llm.decorators import _apply_resource_scaling

        mock_task_instance = mock.MagicMock()
        mock_task_instance.executor_config = {"cpu": 2}

        resource_adjustments = {"memory": "8Gi", "cpu": 4}

        _apply_resource_scaling(mock_task_instance, resource_adjustments)

        expected_config = {"cpu": 4, "memory": "8Gi"}
        assert mock_task_instance.executor_config == expected_config

    def test_apply_resource_scaling_no_existing_config(self):
        from airflow_llm.decorators import _apply_resource_scaling

        mock_task_instance = mock.MagicMock()
        mock_task_instance.executor_config = None

        resource_adjustments = {"memory": "8Gi"}

        _apply_resource_scaling(mock_task_instance, resource_adjustments)

        assert mock_task_instance.executor_config == {"memory": "8Gi"}

    def test_extract_resource_requirements(self):
        from airflow_llm.decorators import _extract_resource_requirements

        def sample_func():
            pass

        sample_func.__resource_requirements__ = {"cpu": 4, "memory": "16Gi"}

        mock_task_instance = mock.MagicMock()
        mock_task_instance.executor_config = {"gpu": "nvidia-tesla-t4"}

        context = {"task_instance": mock_task_instance}

        result = _extract_resource_requirements(sample_func, context)

        assert result["cpu"] == 4
        assert result["memory"] == "16Gi"
        assert result["gpu"] == "nvidia-tesla-t4"
        assert "estimated_runtime" in result

    def test_extract_resource_requirements_defaults(self):
        from airflow_llm.decorators import _extract_resource_requirements

        def sample_func():
            pass

        result = _extract_resource_requirements(sample_func, {})

        assert result["cpu"] == 2
        assert result["memory"] == "4Gi"
        assert result["estimated_runtime"] == 3600

    def test_apply_optimized_resources(self):
        from airflow_llm.cost_optimizer import (
            CloudProvider,
            InstanceType,
            ResourceQuote,
        )
        from airflow_llm.decorators import _apply_optimized_resources

        mock_task_instance = mock.MagicMock()
        context = {"task_instance": mock_task_instance}

        quote = ResourceQuote(
            provider=CloudProvider.AWS,
            instance_type=InstanceType.SPOT,
            cpu=4,
            memory="8Gi",
            gpu="nvidia-tesla-t4",
            hourly_cost=2.50,
            availability_zone="us-west-2a",
            estimated_runtime=3600,
            total_cost=2.50,
            reliability_score=0.85,
        )

        _apply_optimized_resources(context, quote)

        expected_config = {
            "cpu": 4,
            "memory": "8Gi",
            "gpu": "nvidia-tesla-t4",
            "provider": "aws",
            "instance_type": "spot",
        }

        assert mock_task_instance.executor_config == expected_config

    def test_check_performance_thresholds(self):
        from airflow_llm.decorators import _check_performance_thresholds

        thresholds = {
            "max_execution_time": 5.0,
            "max_memory_mb": 1000.0,
            "max_cpu_percent": 80.0,
        }

        _check_performance_thresholds(
            "test_task",
            execution_time=10.0,
            memory_used=2000 * 1024 * 1024,
            cpu_usage=90.0,
            thresholds=thresholds,
        )


@pytest.mark.integration
class TestDecoratorsIntegration:
    @mock.patch("airflow_llm.decorators.NaturalLanguageDAGGenerator")
    @mock.patch("airflow_llm.decorators.SelfHealingAgent")
    @mock.patch("airflow_llm.decorators.CostAwareScheduler")
    def test_combined_decorators(
        self, mock_scheduler_class, mock_agent_class, mock_generator_class
    ):
        mock_generator = mock.MagicMock()
        mock_generator.generate.return_value = "dag = DAG('integrated')"
        mock_generator_class.return_value = mock_generator

        mock_agent = mock.MagicMock()
        mock_agent_class.return_value = mock_agent

        mock_scheduler = mock.MagicMock()
        mock_optimization = mock.MagicMock()
        mock_scheduler.optimize_resources.return_value = mock_optimization
        mock_scheduler_class.return_value = mock_scheduler

        @natural_language_dag(
            "Integrated pipeline with all features",
            cost_optimization=True,
            self_healing=True,
        )
        def integrated_dag():
            @self_healing_task(retries=3)
            @cost_aware_execution(max_cost_per_hour=10.0)
            @intelligent_retry(max_retries=2)
            @performance_monitor(track_memory=True, track_cpu=True)
            def complex_task(**context):
                return "integrated success"

            return complex_task

        assert isinstance(integrated_dag, DAG)
        assert hasattr(integrated_dag, "cost_optimizer")
        assert hasattr(integrated_dag, "self_healer")

        task_func = integrated_dag()
        result = task_func(context={})

        assert result == "integrated success"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
