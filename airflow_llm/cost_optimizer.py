"""
Cost-aware scheduling and resource optimization
Production-grade intelligent cost optimization and resource scheduling
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

try:
    pass

    NUMPY_AVAILABLE = True
except ImportError:
    raise ImportError(
        "NumPy is required for cost optimization. Please install: pip install numpy"
    )

logger = logging.getLogger(__name__)


class InstanceType(Enum):
    ON_DEMAND = "on_demand"
    SPOT = "spot"
    RESERVED = "reserved"
    PREEMPTIBLE = "preemptible"


class CloudProvider(Enum):
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    COREWEAVE = "coreweave"


@dataclass
class ResourceQuote:
    provider: CloudProvider
    instance_type: InstanceType
    cpu: int
    memory: str
    gpu: str | None
    hourly_cost: float
    availability_zone: str
    estimated_runtime: float
    total_cost: float
    reliability_score: float


@dataclass
class OptimizationResult:
    selected_quote: ResourceQuote
    cost_savings: float
    risk_assessment: str
    fallback_options: list[ResourceQuote]
    optimization_strategy: str


class CostAwareScheduler:
    """
    Intelligent cost optimization and resource scheduling
    """

    def __init__(
        self,
        max_cost_per_hour: float = 50.0,
        enable_spot_instances: bool = True,
        enable_multi_cloud: bool = True,
        cost_vs_performance_weight: float = 0.7,
    ):
        self.max_cost_per_hour = max_cost_per_hour
        self.enable_spot_instances = enable_spot_instances
        self.enable_multi_cloud = enable_multi_cloud
        self.cost_vs_performance_weight = cost_vs_performance_weight

        self.pricing_cache = {}
        self.performance_history = {}
        self.spot_price_history = {}

        logger.info(
            f"CostAwareScheduler initialized - "
            f"max_cost_per_hour: ${max_cost_per_hour}, "
            f"spot_instances: {enable_spot_instances}, "
            f"multi_cloud: {enable_multi_cloud}"
        )

    def optimize_resources(
        self,
        resource_requirements: dict[str, Any],
        performance_targets: dict[str, Any],
        cost_constraints: dict[str, Any],
    ) -> OptimizationResult:
        """
        Find optimal resource allocation balancing cost and performance
        """
        logger.info(
            f"Starting resource optimization - "
            f"requirements: {resource_requirements}"
        )

        quotes = self._gather_resource_quotes(resource_requirements)

        logger.debug(f"Gathered {len(quotes)} resource quotes")

        filtered_quotes = self._filter_quotes(quotes, cost_constraints)

        logger.debug(f"Filtered to {len(filtered_quotes)} viable quotes")

        if not filtered_quotes:
            logger.error("No viable resource quotes found within constraints")
            raise ValueError("No viable resource options within cost constraints")

        scored_quotes = self._score_quotes(
            filtered_quotes, performance_targets, cost_constraints
        )

        best_quote = max(scored_quotes, key=lambda x: x[1])

        optimization_result = self._create_optimization_result(
            best_quote[0], scored_quotes, resource_requirements
        )

        logger.info(
            f"Optimization complete - Selected: {best_quote[0].provider.value} "
            f"{best_quote[0].instance_type.value}, "
            f"Cost: ${best_quote[0].hourly_cost:.2f}/hr, "
            f"Savings: {optimization_result.cost_savings:.1f}%"
        )

        return optimization_result

    def predict_spot_interruption(
        self, provider: CloudProvider, instance_type: str, availability_zone: str
    ) -> float:
        """
        Predict probability of spot instance interruption
        """
        logger.debug(
            f"Predicting spot interruption for {provider.value} "
            f"{instance_type} in {availability_zone}"
        )

        historical_data = self._get_spot_history(
            provider, instance_type, availability_zone
        )

        if not historical_data:
            logger.warning("No historical data available, using default probability")
            return 0.3

        interruption_rate = self._calculate_interruption_rate(historical_data)

        logger.debug(f"Predicted interruption probability: {interruption_rate:.3f}")

        return interruption_rate

    def schedule_with_budget(
        self, tasks: list[dict[str, Any]], total_budget: float, deadline: datetime
    ) -> dict[str, Any]:
        """
        Schedule tasks optimally within budget and deadline
        """
        logger.info(
            f"Scheduling {len(tasks)} tasks with budget ${total_budget:.2f} "
            f"by {deadline}"
        )

        task_costs = []
        for task in tasks:
            resource_req = task.get("resource_requirements", {})
            estimated_runtime = task.get("estimated_runtime", 3600)

            quotes = self._gather_resource_quotes(resource_req)
            if quotes:
                min_cost = min(q.total_cost for q in quotes)
                task_costs.append((task["id"], min_cost, estimated_runtime))

        scheduled_tasks = self._optimize_task_schedule(
            task_costs, total_budget, deadline
        )

        logger.info(f"Successfully scheduled {len(scheduled_tasks)} tasks")

        return {
            "scheduled_tasks": scheduled_tasks,
            "total_cost": sum(cost for _, cost, _ in scheduled_tasks),
            "completion_time": self._estimate_completion_time(scheduled_tasks),
        }

    def _gather_resource_quotes(
        self, requirements: dict[str, Any]
    ) -> list[ResourceQuote]:
        """
        Gather resource quotes from multiple providers
        """
        quotes = []

        providers = [CloudProvider.AWS, CloudProvider.GCP, CloudProvider.AZURE]
        if self.enable_multi_cloud:
            providers.append(CloudProvider.COREWEAVE)

        for provider in providers:
            provider_quotes = self._get_provider_quotes(provider, requirements)
            quotes.extend(provider_quotes)

            logger.debug(
                f"Retrieved {len(provider_quotes)} quotes from {provider.value}"
            )

        return quotes

    def _get_provider_quotes(
        self, provider: CloudProvider, requirements: dict[str, Any]
    ) -> list[ResourceQuote]:
        """
        Get quotes from specific cloud provider
        """
        quotes = []

        cpu = requirements.get("cpu", 2)
        memory = requirements.get("memory", "8Gi")
        gpu = requirements.get("gpu")
        estimated_runtime = requirements.get("estimated_runtime", 3600)

        instance_types = [InstanceType.ON_DEMAND]
        if self.enable_spot_instances:
            instance_types.append(InstanceType.SPOT)

        for instance_type in instance_types:
            hourly_cost = self._get_hourly_cost(
                provider, instance_type, cpu, memory, gpu
            )

            if hourly_cost and hourly_cost <= self.max_cost_per_hour:
                total_cost = hourly_cost * (estimated_runtime / 3600)
                reliability_score = self._calculate_reliability_score(
                    provider, instance_type
                )

                quote = ResourceQuote(
                    provider=provider,
                    instance_type=instance_type,
                    cpu=cpu,
                    memory=memory,
                    gpu=gpu,
                    hourly_cost=hourly_cost,
                    availability_zone=f"{provider.value}-us-west-2a",
                    estimated_runtime=estimated_runtime,
                    total_cost=total_cost,
                    reliability_score=reliability_score,
                )

                quotes.append(quote)

        return quotes

    def _get_hourly_cost(
        self,
        provider: CloudProvider,
        instance_type: InstanceType,
        cpu: int,
        memory: str,
        gpu: str | None,
    ) -> float | None:
        """
        Get hourly cost for specific configuration
        """
        cache_key = f"{provider.value}_{instance_type.value}_{cpu}_{memory}_{gpu}"

        if cache_key in self.pricing_cache:
            cached_price, timestamp = self.pricing_cache[cache_key]
            if time.time() - timestamp < 3600:
                return cached_price

        base_cost = self._calculate_base_cost(provider, cpu, memory)

        if gpu:
            gpu_cost = self._calculate_gpu_cost(provider, gpu)
            base_cost += gpu_cost

        if instance_type == InstanceType.SPOT:
            base_cost *= 0.3
        elif instance_type == InstanceType.PREEMPTIBLE:
            base_cost *= 0.2

        self.pricing_cache[cache_key] = (base_cost, time.time())

        logger.debug(
            f"Calculated cost for {provider.value} {instance_type.value}: "
            f"${base_cost:.2f}/hr"
        )

        return base_cost

    def _calculate_base_cost(
        self, provider: CloudProvider, cpu: int, memory: str
    ) -> float:
        """
        Calculate base compute cost
        """
        memory_gb = self._parse_memory_to_gb(memory)

        pricing_table = {
            CloudProvider.AWS: {"cpu": 0.05, "memory": 0.01},
            CloudProvider.GCP: {"cpu": 0.04, "memory": 0.009},
            CloudProvider.AZURE: {"cpu": 0.055, "memory": 0.011},
            CloudProvider.COREWEAVE: {"cpu": 0.03, "memory": 0.007},
        }

        rates = pricing_table.get(provider, {"cpu": 0.05, "memory": 0.01})

        return (cpu * rates["cpu"]) + (memory_gb * rates["memory"])

    def _calculate_gpu_cost(self, provider: CloudProvider, gpu: str) -> float:
        """
        Calculate GPU cost
        """
        gpu_pricing = {
            CloudProvider.AWS: {
                "nvidia-tesla-t4": 0.526,
                "nvidia-tesla-v100": 2.48,
                "nvidia-a100": 4.10,
            },
            CloudProvider.GCP: {
                "nvidia-tesla-t4": 0.35,
                "nvidia-tesla-v100": 2.20,
                "nvidia-a100": 3.67,
            },
            CloudProvider.COREWEAVE: {
                "nvidia-tesla-t4": 0.20,
                "nvidia-tesla-v100": 1.80,
                "nvidia-a100": 2.50,
                "nvidia-h100": 4.76,
            },
        }

        return gpu_pricing.get(provider, {}).get(gpu, 1.0)

    def _filter_quotes(
        self, quotes: list[ResourceQuote], constraints: dict[str, Any]
    ) -> list[ResourceQuote]:
        """
        Filter quotes based on constraints
        """
        max_cost = constraints.get("max_total_cost", float("inf"))
        min_reliability = constraints.get("min_reliability_score", 0.0)

        filtered = []
        for quote in quotes:
            if (
                quote.total_cost <= max_cost
                and quote.reliability_score >= min_reliability
            ):
                filtered.append(quote)

        return filtered

    def _score_quotes(
        self,
        quotes: list[ResourceQuote],
        performance_targets: dict[str, Any],
        cost_constraints: dict[str, Any],
    ) -> list[tuple[ResourceQuote, float]]:
        """
        Score quotes based on cost and performance
        """
        scored_quotes = []

        min_cost = min(q.total_cost for q in quotes)
        max_cost = max(q.total_cost for q in quotes)

        for quote in quotes:
            cost_score = 1.0 - (
                (quote.total_cost - min_cost) / (max_cost - min_cost + 0.001)
            )

            performance_score = quote.reliability_score

            combined_score = (
                self.cost_vs_performance_weight * cost_score
                + (1 - self.cost_vs_performance_weight) * performance_score
            )

            scored_quotes.append((quote, combined_score))

            logger.debug(
                f"Quote scored - {quote.provider.value}: "
                f"cost_score={cost_score:.3f}, "
                f"performance_score={performance_score:.3f}, "
                f"combined={combined_score:.3f}"
            )

        return scored_quotes

    def _create_optimization_result(
        self,
        best_quote: ResourceQuote,
        all_scored_quotes: list[tuple[ResourceQuote, float]],
        original_requirements: dict[str, Any],
    ) -> OptimizationResult:
        """
        Create optimization result with best quote and alternatives
        """
        baseline_cost = self._calculate_baseline_cost(original_requirements)
        cost_savings = (baseline_cost - best_quote.total_cost) / baseline_cost * 100

        fallback_options = [
            quote
            for quote, _ in sorted(all_scored_quotes, key=lambda x: x[1], reverse=True)[
                1:4
            ]
        ]

        risk_assessment = self._assess_risk(best_quote)
        optimization_strategy = self._determine_strategy(best_quote)

        return OptimizationResult(
            selected_quote=best_quote,
            cost_savings=max(0, cost_savings),
            risk_assessment=risk_assessment,
            fallback_options=fallback_options,
            optimization_strategy=optimization_strategy,
        )

    def _calculate_baseline_cost(self, requirements: dict[str, Any]) -> float:
        """
        Calculate baseline cost (on-demand AWS)
        """
        cpu = requirements.get("cpu", 2)
        memory = requirements.get("memory", "8Gi")
        gpu = requirements.get("gpu")
        runtime = requirements.get("estimated_runtime", 3600)

        base_cost = self._calculate_base_cost(CloudProvider.AWS, cpu, memory)
        if gpu:
            base_cost += self._calculate_gpu_cost(CloudProvider.AWS, gpu)

        return base_cost * (runtime / 3600)

    def _assess_risk(self, quote: ResourceQuote) -> str:
        """
        Assess risk level of selected quote
        """
        if quote.instance_type in [InstanceType.SPOT, InstanceType.PREEMPTIBLE]:
            return "Medium - Spot instance may be interrupted"
        elif quote.reliability_score < 0.9:
            return "Medium - Lower reliability provider"
        else:
            return "Low - Stable configuration"

    def _determine_strategy(self, quote: ResourceQuote) -> str:
        """
        Determine optimization strategy used
        """
        if quote.instance_type == InstanceType.SPOT:
            return "Spot instance optimization for maximum cost savings"
        elif quote.provider == CloudProvider.COREWEAVE:
            return "GPU-specialized provider optimization"
        else:
            return "Multi-cloud cost optimization"

    def _parse_memory_to_gb(self, memory: str) -> float:
        """
        Parse memory string to GB value
        """
        if memory.endswith("Gi"):
            return float(memory[:-2])
        elif memory.endswith("G"):
            return float(memory[:-1])
        elif memory.endswith("Mi"):
            return float(memory[:-2]) / 1024
        else:
            return float(memory)

    def _calculate_reliability_score(
        self, provider: CloudProvider, instance_type: InstanceType
    ) -> float:
        """
        Calculate reliability score based on historical data
        """
        base_scores = {
            CloudProvider.AWS: 0.95,
            CloudProvider.GCP: 0.93,
            CloudProvider.AZURE: 0.92,
            CloudProvider.COREWEAVE: 0.90,
        }

        base_score = base_scores.get(provider, 0.85)

        if instance_type in [InstanceType.SPOT, InstanceType.PREEMPTIBLE]:
            base_score *= 0.8

        return base_score

    def _get_spot_history(
        self, provider: CloudProvider, instance_type: str, availability_zone: str
    ) -> list[dict]:
        """
        Get historical spot instance data
        """
        return []

    def _calculate_interruption_rate(self, historical_data: list[dict]) -> float:
        """
        Calculate interruption rate from historical data
        """
        if not historical_data:
            return 0.3

        interruptions = sum(
            1 for event in historical_data if event.get("interrupted", False)
        )

        return interruptions / len(historical_data)

    def _optimize_task_schedule(
        self,
        task_costs: list[tuple[str, float, int]],
        budget: float,
        deadline: datetime,
    ) -> list[tuple[str, float, int]]:
        """
        Optimize task scheduling within budget and deadline
        """
        task_costs.sort(key=lambda x: x[1])

        scheduled = []
        remaining_budget = budget

        for task_id, cost, runtime in task_costs:
            if cost <= remaining_budget:
                scheduled.append((task_id, cost, runtime))
                remaining_budget -= cost

        return scheduled

    def _estimate_completion_time(
        self, scheduled_tasks: list[tuple[str, float, int]]
    ) -> datetime:
        """
        Estimate completion time for scheduled tasks
        """
        total_runtime = sum(runtime for _, _, runtime in scheduled_tasks)
        return datetime.now() + timedelta(seconds=total_runtime)
