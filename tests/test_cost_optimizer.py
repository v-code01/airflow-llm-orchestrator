"""
Unit tests for CostAwareScheduler
"""

import unittest.mock as mock
from datetime import datetime, timedelta

import pytest

from airflow_llm.cost_optimizer import (
    CloudProvider,
    CostAwareScheduler,
    InstanceType,
    OptimizationResult,
    ResourceQuote,
)


class TestCostAwareScheduler:
    @pytest.fixture
    def scheduler(self):
        return CostAwareScheduler(
            max_cost_per_hour=50.0,
            enable_spot_instances=True,
            enable_multi_cloud=True,
            cost_vs_performance_weight=0.7,
        )

    def test_initialization(self, scheduler):
        assert scheduler.max_cost_per_hour == 50.0
        assert scheduler.enable_spot_instances is True
        assert scheduler.enable_multi_cloud is True
        assert scheduler.cost_vs_performance_weight == 0.7
        assert isinstance(scheduler.pricing_cache, dict)
        assert isinstance(scheduler.performance_history, dict)

    @mock.patch("airflow_llm.cost_optimizer.CostAwareScheduler._gather_resource_quotes")
    @mock.patch("airflow_llm.cost_optimizer.CostAwareScheduler._filter_quotes")
    @mock.patch("airflow_llm.cost_optimizer.CostAwareScheduler._score_quotes")
    def test_optimize_resources(self, mock_score, mock_filter, mock_gather, scheduler):
        mock_quote = ResourceQuote(
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

        mock_gather.return_value = [mock_quote]
        mock_filter.return_value = [mock_quote]
        mock_score.return_value = [(mock_quote, 0.95)]

        requirements = {"cpu": 4, "memory": "8Gi", "gpu": "nvidia-tesla-t4"}
        performance_targets = {"max_latency": 300}
        cost_constraints = {"max_total_cost": 100.0}

        result = scheduler.optimize_resources(
            requirements, performance_targets, cost_constraints
        )

        assert isinstance(result, OptimizationResult)
        assert result.selected_quote == mock_quote
        assert result.cost_savings >= 0
        mock_gather.assert_called_once()
        mock_filter.assert_called_once()
        mock_score.assert_called_once()

    def test_optimize_resources_no_viable_quotes(self, scheduler):
        with mock.patch.object(scheduler, "_gather_resource_quotes", return_value=[]):
            with mock.patch.object(scheduler, "_filter_quotes", return_value=[]):
                with pytest.raises(ValueError, match="No viable resource options"):
                    scheduler.optimize_resources({}, {}, {})

    def test_gather_resource_quotes(self, scheduler):
        requirements = {"cpu": 2, "memory": "4Gi", "estimated_runtime": 3600}

        with mock.patch.object(scheduler, "_get_provider_quotes") as mock_get_quotes:
            mock_get_quotes.return_value = [
                ResourceQuote(
                    provider=CloudProvider.AWS,
                    instance_type=InstanceType.ON_DEMAND,
                    cpu=2,
                    memory="4Gi",
                    gpu=None,
                    hourly_cost=1.50,
                    availability_zone="us-west-2a",
                    estimated_runtime=3600,
                    total_cost=1.50,
                    reliability_score=0.95,
                )
            ]

            result = scheduler._gather_resource_quotes(requirements)

            assert len(result) >= 4
            assert mock_get_quotes.call_count >= 4

    def test_get_provider_quotes_aws(self, scheduler):
        requirements = {"cpu": 2, "memory": "4Gi", "estimated_runtime": 3600}

        with mock.patch.object(scheduler, "_get_hourly_cost", return_value=1.50):
            with mock.patch.object(
                scheduler, "_calculate_reliability_score", return_value=0.95
            ):
                result = scheduler._get_provider_quotes(CloudProvider.AWS, requirements)

                assert len(result) >= 1
                assert all(quote.provider == CloudProvider.AWS for quote in result)
                assert all(quote.cpu == 2 for quote in result)
                assert all(quote.memory == "4Gi" for quote in result)

    def test_get_hourly_cost_caching(self, scheduler):
        provider = CloudProvider.AWS
        instance_type = InstanceType.ON_DEMAND
        cpu = 2
        memory = "4Gi"
        gpu = None

        cost1 = scheduler._get_hourly_cost(provider, instance_type, cpu, memory, gpu)
        cost2 = scheduler._get_hourly_cost(provider, instance_type, cpu, memory, gpu)

        assert cost1 == cost2
        assert isinstance(cost1, float)
        assert cost1 > 0

    def test_calculate_base_cost_aws(self, scheduler):
        cost = scheduler._calculate_base_cost(CloudProvider.AWS, 4, "8Gi")
        expected = (4 * 0.05) + (8 * 0.01)
        assert cost == expected

    def test_calculate_base_cost_gcp(self, scheduler):
        cost = scheduler._calculate_base_cost(CloudProvider.GCP, 4, "8Gi")
        expected = (4 * 0.04) + (8 * 0.009)
        assert cost == expected

    def test_calculate_base_cost_coreweave(self, scheduler):
        cost = scheduler._calculate_base_cost(CloudProvider.COREWEAVE, 4, "8Gi")
        expected = (4 * 0.03) + (8 * 0.007)
        assert cost == expected

    def test_calculate_gpu_cost_aws(self, scheduler):
        cost = scheduler._calculate_gpu_cost(CloudProvider.AWS, "nvidia-tesla-t4")
        assert cost == 0.526

        cost = scheduler._calculate_gpu_cost(CloudProvider.AWS, "nvidia-tesla-v100")
        assert cost == 2.48

        cost = scheduler._calculate_gpu_cost(CloudProvider.AWS, "nvidia-a100")
        assert cost == 4.10

    def test_calculate_gpu_cost_coreweave(self, scheduler):
        cost = scheduler._calculate_gpu_cost(CloudProvider.COREWEAVE, "nvidia-h100")
        assert cost == 4.76

        cost = scheduler._calculate_gpu_cost(CloudProvider.COREWEAVE, "unknown-gpu")
        assert cost == 1.0

    def test_filter_quotes_by_cost(self, scheduler):
        quotes = [
            ResourceQuote(
                provider=CloudProvider.AWS,
                instance_type=InstanceType.ON_DEMAND,
                cpu=2,
                memory="4Gi",
                gpu=None,
                hourly_cost=1.50,
                availability_zone="us-west-2a",
                estimated_runtime=3600,
                total_cost=1.50,
                reliability_score=0.95,
            ),
            ResourceQuote(
                provider=CloudProvider.AWS,
                instance_type=InstanceType.ON_DEMAND,
                cpu=8,
                memory="32Gi",
                gpu="nvidia-a100",
                hourly_cost=10.00,
                availability_zone="us-west-2a",
                estimated_runtime=3600,
                total_cost=10.00,
                reliability_score=0.95,
            ),
        ]

        constraints = {"max_total_cost": 5.0}
        result = scheduler._filter_quotes(quotes, constraints)

        assert len(result) == 1
        assert result[0].total_cost <= 5.0

    def test_filter_quotes_by_reliability(self, scheduler):
        quotes = [
            ResourceQuote(
                provider=CloudProvider.AWS,
                instance_type=InstanceType.SPOT,
                cpu=2,
                memory="4Gi",
                gpu=None,
                hourly_cost=0.50,
                availability_zone="us-west-2a",
                estimated_runtime=3600,
                total_cost=0.50,
                reliability_score=0.70,
            ),
            ResourceQuote(
                provider=CloudProvider.AWS,
                instance_type=InstanceType.ON_DEMAND,
                cpu=2,
                memory="4Gi",
                gpu=None,
                hourly_cost=1.50,
                availability_zone="us-west-2a",
                estimated_runtime=3600,
                total_cost=1.50,
                reliability_score=0.95,
            ),
        ]

        constraints = {"min_reliability_score": 0.80}
        result = scheduler._filter_quotes(quotes, constraints)

        assert len(result) == 1
        assert result[0].reliability_score >= 0.80

    def test_score_quotes(self, scheduler):
        quotes = [
            ResourceQuote(
                provider=CloudProvider.AWS,
                instance_type=InstanceType.SPOT,
                cpu=2,
                memory="4Gi",
                gpu=None,
                hourly_cost=0.50,
                availability_zone="us-west-2a",
                estimated_runtime=3600,
                total_cost=0.50,
                reliability_score=0.80,
            ),
            ResourceQuote(
                provider=CloudProvider.AWS,
                instance_type=InstanceType.ON_DEMAND,
                cpu=2,
                memory="4Gi",
                gpu=None,
                hourly_cost=1.50,
                availability_zone="us-west-2a",
                estimated_runtime=3600,
                total_cost=1.50,
                reliability_score=0.95,
            ),
        ]

        performance_targets = {}
        cost_constraints = {}

        result = scheduler._score_quotes(quotes, performance_targets, cost_constraints)

        assert len(result) == 2
        assert all(isinstance(score, float) for _, score in result)
        assert all(0 <= score <= 1 for _, score in result)

    def test_parse_memory_to_gb(self, scheduler):
        assert scheduler._parse_memory_to_gb("8Gi") == 8.0
        assert scheduler._parse_memory_to_gb("16G") == 16.0
        assert scheduler._parse_memory_to_gb("2048Mi") == 2.0
        assert scheduler._parse_memory_to_gb("4") == 4.0

    def test_calculate_reliability_score(self, scheduler):
        aws_score = scheduler._calculate_reliability_score(
            CloudProvider.AWS, InstanceType.ON_DEMAND
        )
        assert aws_score == 0.95

        spot_score = scheduler._calculate_reliability_score(
            CloudProvider.AWS, InstanceType.SPOT
        )
        assert spot_score == 0.95 * 0.8

        coreweave_score = scheduler._calculate_reliability_score(
            CloudProvider.COREWEAVE, InstanceType.ON_DEMAND
        )
        assert coreweave_score == 0.90

    def test_predict_spot_interruption_no_data(self, scheduler):
        result = scheduler.predict_spot_interruption(
            CloudProvider.AWS, "t3.medium", "us-west-2a"
        )
        assert result == 0.3

    def test_predict_spot_interruption_with_data(self, scheduler):
        historical_data = [
            {"interrupted": True},
            {"interrupted": False},
            {"interrupted": False},
            {"interrupted": True},
        ]

        with mock.patch.object(
            scheduler, "_get_spot_history", return_value=historical_data
        ):
            result = scheduler.predict_spot_interruption(
                CloudProvider.AWS, "t3.medium", "us-west-2a"
            )
            assert result == 0.5

    def test_schedule_with_budget(self, scheduler):
        tasks = [
            {
                "id": "task1",
                "resource_requirements": {"cpu": 2},
                "estimated_runtime": 1800,
            },
            {
                "id": "task2",
                "resource_requirements": {"cpu": 4},
                "estimated_runtime": 3600,
            },
            {
                "id": "task3",
                "resource_requirements": {"cpu": 1},
                "estimated_runtime": 900,
            },
        ]

        deadline = datetime.now() + timedelta(hours=6)

        with mock.patch.object(scheduler, "_gather_resource_quotes") as mock_quotes:
            mock_quotes.return_value = [
                ResourceQuote(
                    provider=CloudProvider.AWS,
                    instance_type=InstanceType.SPOT,
                    cpu=2,
                    memory="4Gi",
                    gpu=None,
                    hourly_cost=1.0,
                    availability_zone="us-west-2a",
                    estimated_runtime=3600,
                    total_cost=5.0,
                    reliability_score=0.85,
                )
            ]

            result = scheduler.schedule_with_budget(tasks, 20.0, deadline)

            assert "scheduled_tasks" in result
            assert "total_cost" in result
            assert "completion_time" in result
            assert result["total_cost"] <= 20.0

    def test_calculate_baseline_cost(self, scheduler):
        requirements = {"cpu": 4, "memory": "8Gi", "estimated_runtime": 3600}

        cost = scheduler._calculate_baseline_cost(requirements)

        assert isinstance(cost, float)
        assert cost > 0

    def test_calculate_baseline_cost_with_gpu(self, scheduler):
        requirements = {
            "cpu": 4,
            "memory": "8Gi",
            "gpu": "nvidia-tesla-t4",
            "estimated_runtime": 3600,
        }

        cost = scheduler._calculate_baseline_cost(requirements)

        assert isinstance(cost, float)
        assert cost > 0.5

    def test_assess_risk_spot_instance(self, scheduler):
        quote = ResourceQuote(
            provider=CloudProvider.AWS,
            instance_type=InstanceType.SPOT,
            cpu=2,
            memory="4Gi",
            gpu=None,
            hourly_cost=0.50,
            availability_zone="us-west-2a",
            estimated_runtime=3600,
            total_cost=0.50,
            reliability_score=0.80,
        )

        risk = scheduler._assess_risk(quote)
        assert "Medium" in risk
        assert "Spot" in risk

    def test_assess_risk_low_reliability(self, scheduler):
        quote = ResourceQuote(
            provider=CloudProvider.AWS,
            instance_type=InstanceType.ON_DEMAND,
            cpu=2,
            memory="4Gi",
            gpu=None,
            hourly_cost=1.50,
            availability_zone="us-west-2a",
            estimated_runtime=3600,
            total_cost=1.50,
            reliability_score=0.85,
        )

        risk = scheduler._assess_risk(quote)
        assert "Medium" in risk
        assert "reliability" in risk

    def test_assess_risk_stable(self, scheduler):
        quote = ResourceQuote(
            provider=CloudProvider.AWS,
            instance_type=InstanceType.ON_DEMAND,
            cpu=2,
            memory="4Gi",
            gpu=None,
            hourly_cost=1.50,
            availability_zone="us-west-2a",
            estimated_runtime=3600,
            total_cost=1.50,
            reliability_score=0.95,
        )

        risk = scheduler._assess_risk(quote)
        assert "Low" in risk

    def test_determine_strategy(self, scheduler):
        spot_quote = ResourceQuote(
            provider=CloudProvider.AWS,
            instance_type=InstanceType.SPOT,
            cpu=2,
            memory="4Gi",
            gpu=None,
            hourly_cost=0.50,
            availability_zone="us-west-2a",
            estimated_runtime=3600,
            total_cost=0.50,
            reliability_score=0.80,
        )

        strategy = scheduler._determine_strategy(spot_quote)
        assert "Spot" in strategy

        coreweave_quote = ResourceQuote(
            provider=CloudProvider.COREWEAVE,
            instance_type=InstanceType.ON_DEMAND,
            cpu=2,
            memory="4Gi",
            gpu="nvidia-h100",
            hourly_cost=5.00,
            availability_zone="us-west-2a",
            estimated_runtime=3600,
            total_cost=5.00,
            reliability_score=0.90,
        )

        strategy = scheduler._determine_strategy(coreweave_quote)
        assert "GPU-specialized" in strategy


class TestResourceQuote:
    def test_resource_quote_creation(self):
        quote = ResourceQuote(
            provider=CloudProvider.AWS,
            instance_type=InstanceType.ON_DEMAND,
            cpu=4,
            memory="8Gi",
            gpu="nvidia-tesla-t4",
            hourly_cost=3.50,
            availability_zone="us-west-2a",
            estimated_runtime=7200,
            total_cost=7.00,
            reliability_score=0.95,
        )

        assert quote.provider == CloudProvider.AWS
        assert quote.instance_type == InstanceType.ON_DEMAND
        assert quote.cpu == 4
        assert quote.memory == "8Gi"
        assert quote.gpu == "nvidia-tesla-t4"
        assert quote.hourly_cost == 3.50
        assert quote.total_cost == 7.00
        assert quote.reliability_score == 0.95


class TestOptimizationResult:
    def test_optimization_result_creation(self):
        quote = ResourceQuote(
            provider=CloudProvider.AWS,
            instance_type=InstanceType.SPOT,
            cpu=2,
            memory="4Gi",
            gpu=None,
            hourly_cost=0.50,
            availability_zone="us-west-2a",
            estimated_runtime=3600,
            total_cost=0.50,
            reliability_score=0.80,
        )

        result = OptimizationResult(
            selected_quote=quote,
            cost_savings=65.0,
            risk_assessment="Medium - Spot instance may be interrupted",
            fallback_options=[],
            optimization_strategy="Spot instance optimization",
        )

        assert result.selected_quote == quote
        assert result.cost_savings == 65.0
        assert "Medium" in result.risk_assessment
        assert "Spot" in result.optimization_strategy


class TestEnums:
    def test_instance_type_enum(self):
        assert InstanceType.ON_DEMAND.value == "on_demand"
        assert InstanceType.SPOT.value == "spot"
        assert InstanceType.RESERVED.value == "reserved"
        assert InstanceType.PREEMPTIBLE.value == "preemptible"

    def test_cloud_provider_enum(self):
        assert CloudProvider.AWS.value == "aws"
        assert CloudProvider.GCP.value == "gcp"
        assert CloudProvider.AZURE.value == "azure"
        assert CloudProvider.COREWEAVE.value == "coreweave"


@pytest.mark.integration
class TestCostAwareSchedulerIntegration:
    @pytest.fixture
    def scheduler(self):
        return CostAwareScheduler(
            max_cost_per_hour=10.0, enable_spot_instances=True, enable_multi_cloud=True
        )

    def test_end_to_end_optimization(self, scheduler):
        requirements = {"cpu": 2, "memory": "4Gi", "estimated_runtime": 3600}
        performance_targets = {"max_latency": 300}
        cost_constraints = {"max_total_cost": 50.0}

        result = scheduler.optimize_resources(
            requirements, performance_targets, cost_constraints
        )

        assert isinstance(result, OptimizationResult)
        assert result.selected_quote.total_cost <= 50.0
        assert result.cost_savings >= 0
        assert len(result.fallback_options) <= 3

    def test_multi_cloud_comparison(self, scheduler):
        requirements = {
            "cpu": 4,
            "memory": "8Gi",
            "gpu": "nvidia-tesla-t4",
            "estimated_runtime": 7200,
        }

        quotes = scheduler._gather_resource_quotes(requirements)

        providers = {quote.provider for quote in quotes}
        assert CloudProvider.AWS in providers
        assert CloudProvider.GCP in providers
        assert CloudProvider.AZURE in providers
        assert CloudProvider.COREWEAVE in providers

    def test_spot_vs_ondemand_comparison(self, scheduler):
        requirements = {"cpu": 2, "memory": "4Gi", "estimated_runtime": 3600}

        quotes = scheduler._gather_resource_quotes(requirements)
        instance_types = {quote.instance_type for quote in quotes}

        assert InstanceType.ON_DEMAND in instance_types
        assert InstanceType.SPOT in instance_types

        spot_quotes = [q for q in quotes if q.instance_type == InstanceType.SPOT]
        ondemand_quotes = [
            q for q in quotes if q.instance_type == InstanceType.ON_DEMAND
        ]

        if spot_quotes and ondemand_quotes:
            avg_spot_cost = sum(q.hourly_cost for q in spot_quotes) / len(spot_quotes)
            avg_ondemand_cost = sum(q.hourly_cost for q in ondemand_quotes) / len(
                ondemand_quotes
            )

            assert avg_spot_cost < avg_ondemand_cost


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
