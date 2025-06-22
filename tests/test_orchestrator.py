"""
Unit tests for LLMOrchestrator
"""

import unittest.mock as mock
from datetime import datetime, timedelta

import pytest

from airflow_llm.orchestrator import (
    CostTracker,
    LLMOrchestrator,
    ModelRouter,
    PipelineMetrics,
)


class TestLLMOrchestrator:
    @pytest.fixture
    def orchestrator(self):
        return LLMOrchestrator(
            models=["gpt-4", "gpt-3.5-turbo"],
            cost_optimization=True,
            self_healing=True,
            predictive_analytics=True,
        )

    def test_initialization(self, orchestrator):
        assert orchestrator.models == ["gpt-4", "gpt-3.5-turbo"]
        assert orchestrator.cost_optimization is True
        assert orchestrator.self_healing is True
        assert orchestrator.predictive_analytics is True
        assert isinstance(orchestrator.model_router, ModelRouter)
        assert isinstance(orchestrator.cost_tracker, CostTracker)

    @mock.patch("airflow_llm.orchestrator.LLMOrchestrator._parse_requirements")
    @mock.patch("airflow_llm.orchestrator.LLMOrchestrator._optimize_dependencies")
    @mock.patch("airflow_llm.orchestrator.LLMOrchestrator._create_dag")
    def test_generate_dag(
        self, mock_create_dag, mock_optimize_deps, mock_parse_req, orchestrator
    ):
        mock_parse_req.return_value = {"tasks": ["task1", "task2"]}
        mock_optimize_deps.return_value = {"optimized": True}
        mock_create_dag.return_value = mock.MagicMock()

        description = "Process data and train model"
        constraints = {"max_cost": 100}

        result = orchestrator.generate_dag(description, constraints)

        mock_parse_req.assert_called_once_with(description)
        mock_optimize_deps.assert_called_once()
        mock_create_dag.assert_called_once()
        assert result is not None

    @mock.patch("airflow_llm.orchestrator.DagRun.find")
    def test_analyze_execution_patterns(self, mock_dag_run_find, orchestrator):
        mock_run = mock.MagicMock()
        mock_run.state = "success"
        mock_run.start_date = datetime.now() - timedelta(hours=1)
        mock_run.end_date = datetime.now()
        mock_dag_run_find.return_value = [mock_run]

        orchestrator.cost_tracker.calculate_run_cost = mock.MagicMock(return_value=10.0)
        orchestrator._identify_bottlenecks = mock.MagicMock(return_value=["task1"])
        orchestrator._generate_suggestions = mock.MagicMock(
            return_value=["optimize memory"]
        )
        orchestrator._get_resource_utilization = mock.MagicMock(
            return_value={"cpu": 0.8}
        )

        result = orchestrator.analyze_execution_patterns("test_dag")

        assert isinstance(result, PipelineMetrics)
        assert result.success_rate == 1.0
        assert result.bottlenecks == ["task1"]
        assert result.optimization_suggestions == ["optimize memory"]

    @mock.patch("airflow_llm.orchestrator.LLMOrchestrator._analyze_task_patterns")
    def test_predict_bottlenecks(self, mock_analyze_patterns, orchestrator):
        mock_analyze_patterns.return_value = {
            "task1": {
                "failure_prob": 0.8,
                "expected_time": 3600,
                "metrics": {"cpu": 0.9},
            }
        }
        orchestrator._will_likely_fail = mock.MagicMock(return_value=True)
        orchestrator._get_prevention_strategy = mock.MagicMock(
            return_value="increase memory"
        )

        result = orchestrator.predict_bottlenecks("test_dag")

        assert len(result) == 1
        assert result[0]["task_id"] == "task1"
        assert result[0]["failure_probability"] == 0.8
        assert result[0]["recommendation"] == "increase memory"

    @mock.patch("airflow_llm.orchestrator.LLMOrchestrator._predict_resource_needs")
    @mock.patch(
        "airflow_llm.orchestrator.LLMOrchestrator._optimize_resource_allocation"
    )
    def test_auto_scale_resources(self, mock_optimize, mock_predict, orchestrator):
        mock_dag_run = mock.MagicMock()
        mock_predict.return_value = {"cpu": 4, "memory": "8Gi"}
        mock_optimize.return_value = {"cpu": 4, "memory": "8Gi", "gpu": 0}

        result = orchestrator.auto_scale_resources(mock_dag_run)

        mock_predict.assert_called_once_with(mock_dag_run)
        mock_optimize.assert_called_once()
        assert result["cpu"] == 4
        assert result["memory"] == "8Gi"

    def test_parse_requirements_integration(self, orchestrator):
        orchestrator.model_router.query = mock.MagicMock(
            return_value='{"tasks": [{"name": "extract", "type": "python"}]}'
        )

        result = orchestrator._parse_requirements("Extract data from S3")

        assert "tasks" in result
        assert len(result["tasks"]) == 1
        assert result["tasks"][0]["name"] == "extract"


class TestModelRouter:
    @pytest.fixture
    def router(self):
        return ModelRouter(["gpt-4", "gpt-3.5-turbo"])

    def test_initialization(self, router):
        assert router.models == ["gpt-4", "gpt-3.5-turbo"]
        assert "gpt-4" in router.performance_history
        assert "gpt-3.5-turbo" in router.performance_history

    @mock.patch("airflow_llm.orchestrator.ModelRouter._query_model")
    @mock.patch("airflow_llm.orchestrator.ModelRouter._select_model")
    def test_query_success(self, mock_select, mock_query_model, router):
        mock_select.return_value = "gpt-4"
        mock_query_model.return_value = "Generated response"

        result = router.query("Test prompt", "parsing")

        mock_select.assert_called_once_with("parsing")
        mock_query_model.assert_called_once_with("gpt-4", "Test prompt")
        assert result == "Generated response"

    @mock.patch("airflow_llm.orchestrator.ModelRouter._query_model")
    @mock.patch("airflow_llm.orchestrator.ModelRouter._fallback_query")
    @mock.patch("airflow_llm.orchestrator.ModelRouter._select_model")
    def test_query_with_fallback(
        self, mock_select, mock_fallback, mock_query_model, router
    ):
        mock_select.return_value = "gpt-4"
        mock_query_model.side_effect = Exception("API Error")
        mock_fallback.return_value = "Fallback response"

        result = router.query("Test prompt", "parsing")

        mock_fallback.assert_called_once_with(
            "Test prompt", "parsing", exclude=["gpt-4"]
        )
        assert result == "Fallback response"

    def test_select_model(self, router):
        result = router._select_model("parsing")
        assert result in router.models


class TestCostTracker:
    @pytest.fixture
    def tracker(self):
        return CostTracker()

    def test_get_gpu_rate(self, tracker):
        assert tracker._get_gpu_rate("a100") == 3.50
        assert tracker._get_gpu_rate("v100") == 2.20
        assert tracker._get_gpu_rate("t4") == 0.75
        assert tracker._get_gpu_rate("unknown") == 1.0

    def test_get_cpu_rate(self, tracker):
        assert tracker._get_cpu_rate(2) == 0.20
        assert tracker._get_cpu_rate(4) == 0.40
        assert tracker._get_cpu_rate(8) == 0.80

    @mock.patch("airflow_llm.orchestrator.DagRun")
    def test_calculate_run_cost(self, mock_dag_run, tracker):
        mock_ti = mock.MagicMock()
        mock_ti.start_date = datetime.now() - timedelta(hours=1)
        mock_ti.end_date = datetime.now()
        mock_ti.executor_config = {"cpu": 2}

        mock_dag_run.get_task_instances.return_value = [mock_ti]

        result = tracker.calculate_run_cost(mock_dag_run)

        assert result > 0
        assert isinstance(result, float)

    def test_calculate_run_cost_with_gpu(self, tracker):
        mock_dag_run = mock.MagicMock()
        mock_ti = mock.MagicMock()
        mock_ti.start_date = datetime.now() - timedelta(hours=1)
        mock_ti.end_date = datetime.now()
        mock_ti.executor_config = {"gpu": "a100", "cpu": 4}

        mock_dag_run.get_task_instances.return_value = [mock_ti]

        result = tracker.calculate_run_cost(mock_dag_run)

        assert result > 3.5
        assert isinstance(result, float)


class TestPipelineMetrics:
    def test_pipeline_metrics_creation(self):
        metrics = PipelineMetrics(
            avg_runtime=3600.0,
            success_rate=0.95,
            cost_per_run=25.50,
            resource_utilization={"cpu": 0.8, "memory": 0.6},
            bottlenecks=["task1", "task2"],
            optimization_suggestions=["increase memory", "use spot instances"],
        )

        assert metrics.avg_runtime == 3600.0
        assert metrics.success_rate == 0.95
        assert metrics.cost_per_run == 25.50
        assert len(metrics.bottlenecks) == 2
        assert len(metrics.optimization_suggestions) == 2
        assert metrics.resource_utilization["cpu"] == 0.8

    def test_pipeline_metrics_defaults(self):
        metrics = PipelineMetrics(
            avg_runtime=0.0,
            success_rate=0.0,
            cost_per_run=0.0,
            resource_utilization={},
            bottlenecks=[],
            optimization_suggestions=[],
        )

        assert metrics.avg_runtime == 0.0
        assert metrics.success_rate == 0.0
        assert len(metrics.bottlenecks) == 0


@pytest.mark.integration
class TestLLMOrchestratorIntegration:
    @pytest.fixture
    def orchestrator(self):
        return LLMOrchestrator(
            models=["gpt-4"], cost_optimization=True, self_healing=True
        )

    @mock.patch("airflow_llm.orchestrator.LLMOrchestrator._parse_requirements")
    @mock.patch("airflow_llm.orchestrator.LLMOrchestrator._optimize_dependencies")
    @mock.patch("airflow_llm.orchestrator.LLMOrchestrator._create_dag")
    def test_end_to_end_dag_generation(
        self, mock_create_dag, mock_optimize_deps, mock_parse_req, orchestrator
    ):
        mock_parse_req.return_value = {
            "tasks": [
                {"name": "extract", "type": "python", "dependencies": []},
                {"name": "transform", "type": "python", "dependencies": ["extract"]},
                {"name": "load", "type": "python", "dependencies": ["transform"]},
            ]
        }
        mock_optimize_deps.return_value = mock_parse_req.return_value
        mock_dag = mock.MagicMock()
        mock_create_dag.return_value = mock_dag

        description = "Extract data from S3, transform it, and load to warehouse"
        constraints = {"max_cost_per_run": 50.0, "max_runtime": 7200}

        result = orchestrator.generate_dag(description, constraints)

        assert result == mock_dag
        mock_parse_req.assert_called_once_with(description)
        mock_optimize_deps.assert_called_once()
        mock_create_dag.assert_called_once()

    def test_cost_optimization_integration(self, orchestrator):
        assert orchestrator.cost_optimization is True
        assert orchestrator.cost_tracker is not None

        mock_dag_run = mock.MagicMock()
        orchestrator._predict_resource_needs = mock.MagicMock(
            return_value={"cpu": 4, "memory": "8Gi"}
        )
        orchestrator._optimize_resource_allocation = mock.MagicMock(
            return_value={"cpu": 2, "memory": "4Gi", "cost_optimized": True}
        )

        result = orchestrator.auto_scale_resources(mock_dag_run)

        assert result["cost_optimized"] is True
        assert result["cpu"] == 2

    def test_self_healing_integration(self, orchestrator):
        assert orchestrator.self_healing is True

        mock_context = {
            "exception": Exception("Test error"),
            "task_instance": mock.MagicMock(),
        }

        orchestrator.model_router.query = mock.MagicMock(
            return_value="Increase memory allocation to 8GB"
        )
        orchestrator._apply_automatic_fix = mock.MagicMock(return_value=True)

        orchestrator._self_heal_callback(mock_context)

        orchestrator.model_router.query.assert_called_once()
        orchestrator._apply_automatic_fix.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
