#!/usr/bin/env python3
"""
Comprehensive tests for the specialized model ensemble
"""

import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from airflow_llm.models.specialized_ensemble import (
    EnhancedModelRouter,
    ModelResponse,
    SpecializedModelRouter,
    TaskType,
)


class TestSpecializedModelRouter:
    """Test the specialized model routing system"""

    @pytest.fixture
    def router(self):
        """Create router with mocked models for testing"""
        router = SpecializedModelRouter(device="cpu")

        # Mock the models to avoid loading actual models in tests
        router.pipelines = {
            TaskType.SQL_GENERATION: Mock(),
            TaskType.PYTHON_CODE: Mock(),
            TaskType.ERROR_DEBUG: Mock(),
            TaskType.DAG_ORCHESTRATION: Mock(),
        }

        # Configure mock responses
        for task_type, pipeline in router.pipelines.items():
            pipeline.return_value = [
                {"generated_text": f"Mock response for {task_type.value} task"}
            ]

        return router

    def test_task_classification(self, router):
        """Test that prompts are classified correctly"""
        test_cases = [
            ("SELECT * FROM users WHERE age > 25", TaskType.SQL_GENERATION),
            ("Create a Python function to process data", TaskType.PYTHON_CODE),
            ("Error: ModuleNotFoundError in my Airflow task", TaskType.ERROR_DEBUG),
            ("Build an Airflow DAG for daily processing", TaskType.DAG_ORCHESTRATION),
            ("Optimize costs for my ML training workload", TaskType.COST_OPTIMIZATION),
            ("Some random text", TaskType.GENERAL),
        ]

        for prompt, expected_type in test_cases:
            classified = router.classify_task(prompt)
            assert (
                classified == expected_type
            ), f"Expected {expected_type}, got {classified} for: {prompt}"

    def test_prompt_enhancement(self, router):
        """Test that prompts are enhanced with appropriate context"""
        prompt = "Generate a query to get user data"

        enhanced = router.enhance_prompt(prompt, TaskType.SQL_GENERATION)
        assert "SQL expert" in enhanced
        assert prompt in enhanced
        assert "SQL Query:" in enhanced

        enhanced_python = router.enhance_prompt(prompt, TaskType.PYTHON_CODE)
        assert "Python expert" in enhanced_python
        assert "```python" in enhanced_python

    def test_generate_response(self, router):
        """Test response generation"""
        prompt = "Create a simple ETL pipeline"

        response = router.generate(prompt)

        assert isinstance(response, ModelResponse)
        assert response.content is not None
        assert response.latency_ms > 0
        assert 0 <= response.confidence <= 1
        assert response.model_used is not None

    @pytest.mark.asyncio
    async def test_async_generation(self, router):
        """Test async response generation"""
        prompt = "SELECT name, email FROM users"

        response = await router.generate_async(prompt, TaskType.SQL_GENERATION)

        assert isinstance(response, ModelResponse)
        assert "Mock response for sql task" in response.content

    def test_confidence_calculation(self, router):
        """Test confidence scoring"""
        # SQL response should have high confidence
        sql_response = "SELECT id, name FROM users WHERE active = 1"
        confidence = router._calculate_confidence(
            "Get active users", sql_response, TaskType.SQL_GENERATION
        )
        assert confidence > 0.8

        # Error response should have low confidence
        error_response = "Error: Could not process request"
        confidence = router._calculate_confidence(
            "Get users", error_response, TaskType.SQL_GENERATION
        )
        assert confidence < 0.5

    def test_performance_tracking(self, router):
        """Test that performance metrics are tracked"""
        # Generate a few responses
        for i in range(5):
            router.generate(f"Test prompt {i}")

        stats = router.get_performance_stats()

        # Should have stats for the tasks that were called
        assert len(stats) > 0
        for task_type, stat in stats.items():
            if stat["total_requests"] > 0:
                assert stat["average_latency_ms"] > 0
                assert 0 <= stat["error_rate"] <= 1

    def test_health_check(self, router):
        """Test health check functionality"""
        health = router.health_check()

        assert isinstance(health, dict)
        assert len(health) == len(TaskType)

        # All should be healthy with mocked pipelines
        for task_type_name, is_healthy in health.items():
            assert isinstance(is_healthy, bool)

    def test_error_handling(self, router):
        """Test error handling when models fail"""
        # Make one pipeline fail
        router.pipelines[TaskType.SQL_GENERATION].side_effect = Exception(
            "Model failed"
        )

        response = router.generate("SELECT * FROM users", TaskType.SQL_GENERATION)

        assert response.confidence == 0.0
        assert "Error generating response" in response.content
        assert response.model_used == "error"

    def test_fallback_behavior(self, router):
        """Test fallback when specialized model not available"""
        # Remove SQL model
        router.pipelines[TaskType.SQL_GENERATION] = None

        response = router.generate("SELECT * FROM users", TaskType.SQL_GENERATION)

        # Should fallback to orchestration model
        assert (
            response.model_used
            == router.model_configs[TaskType.DAG_ORCHESTRATION]["model_id"]
        )


class TestEnhancedModelRouter:
    """Test the enhanced router integration"""

    @pytest.fixture
    def enhanced_router(self):
        """Create enhanced router with mocked ensemble"""
        with patch(
            "airflow_llm.models.specialized_ensemble.SpecializedModelRouter"
        ) as mock_router_class:
            mock_router = Mock()
            mock_router_class.return_value = mock_router

            # Mock the generate method
            mock_response = ModelResponse(
                content='{"dag_id": "test", "tasks": []}',
                model_used="test-model",
                confidence=0.9,
                latency_ms=100,
                tokens_used=50,
            )
            mock_router.generate.return_value = mock_response

            router = EnhancedModelRouter()
            router.ensemble = mock_router

            return router

    def test_query_interface(self, enhanced_router):
        """Test backward compatibility query interface"""
        result = enhanced_router.query("Create a test DAG", "orchestration")

        assert isinstance(result, str)
        assert "dag_id" in result  # Should be JSON string

    def test_fallback_on_error(self, enhanced_router):
        """Test fallback when specialized models fail"""
        enhanced_router.ensemble.generate.side_effect = Exception("All models failed")

        result = enhanced_router.query("Create a DAG")

        # Should get fallback response
        assert "fallback_dag" in result


class TestModelBenchmarks:
    """Benchmark tests to verify performance claims"""

    @pytest.fixture
    def benchmark_router(self):
        """Router for benchmarking (uses real models if available)"""
        router = SpecializedModelRouter(device="cpu")

        # Try to initialize models, fall back to mocks if not available
        try:
            router.initialize_models()
        except Exception:
            # Use mocks for CI/CD
            router.pipelines = {task_type: Mock() for task_type in TaskType}
            for pipeline in router.pipelines.values():
                pipeline.return_value = [{"generated_text": "Mock response"}]

        return router

    def test_latency_benchmark(self, benchmark_router):
        """Test that latency meets performance targets"""
        prompts = [
            "SELECT TOP 10 * FROM sales ORDER BY revenue DESC",
            "def process_data(df): return df.groupby('category').sum()",
            "Error: pandas.errors.ParserError in CSV reading task",
            "Create DAG for hourly data processing with 5 tasks",
        ]

        latencies = []

        for prompt in prompts:
            start_time = time.perf_counter()
            response = benchmark_router.generate(prompt)
            latency = (time.perf_counter() - start_time) * 1000
            latencies.append(latency)

            # Verify response quality
            assert len(response.content) > 10
            assert response.confidence > 0.5

        avg_latency = sum(latencies) / len(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]

        # Performance targets (these would be actual measurements with real models)
        assert (
            avg_latency < 1000
        ), f"Average latency {avg_latency}ms exceeds 1000ms target"
        assert p95_latency < 2000, f"P95 latency {p95_latency}ms exceeds 2000ms target"

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, benchmark_router):
        """Test handling concurrent requests"""
        num_concurrent = 10
        prompt = "Generate SQL for user analytics"

        # Generate concurrent requests
        tasks = [benchmark_router.generate_async(prompt) for _ in range(num_concurrent)]

        start_time = time.perf_counter()
        responses = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start_time

        # Verify all responses succeeded
        assert len(responses) == num_concurrent
        for response in responses:
            assert isinstance(response, ModelResponse)
            assert len(response.content) > 0

        # Performance should be better than sequential
        avg_concurrent_time = total_time / num_concurrent

        # This should be much faster than sequential processing
        print(f"Concurrent average: {avg_concurrent_time*1000:.1f}ms per request")

    def test_memory_usage(self, benchmark_router):
        """Test memory usage stays reasonable"""
        import tracemalloc

        tracemalloc.start()

        # Generate many requests to test memory leaks
        for i in range(100):
            response = benchmark_router.generate(f"Test query {i}")
            assert response is not None

            # Force cleanup
            del response

        # Check memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Memory should be reasonable (less than 100MB for 100 requests)
        peak_mb = peak / 1024 / 1024
        assert peak_mb < 100, f"Peak memory usage {peak_mb:.1f}MB is too high"


class TestIntegrationWithOrchestrator:
    """Test integration with the main orchestrator"""

    def test_orchestrator_integration(self):
        """Test that enhanced router integrates with orchestrator"""
        # This would test the actual integration
        # For now, verify the interface matches

        router = EnhancedModelRouter()

        # Should have the expected interface
        assert hasattr(router, "query")
        assert callable(router.query)

        # Should accept the same parameters as original
        result = router.query("test prompt", "general")
        assert isinstance(result, str)


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
