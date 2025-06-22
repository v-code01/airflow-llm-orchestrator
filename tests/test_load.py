#!/usr/bin/env python3
"""
Advanced Load Testing Framework for AirflowLLM

Tests system performance under realistic production loads:
- Concurrent DAG generation requests
- Stress testing with large DAGs (1000+ tasks)
- Memory usage monitoring
- Latency percentile analysis
- Error rate tracking
"""

import json
import logging
import multiprocessing
import random
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime

import psutil

sys.path.insert(0, "/Users/vanshverma/airflow-llm-orchestrator")
from airflow_llm.orchestrator import LLMOrchestrator
from airflow_llm.self_healer import SelfHealingAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LoadTestMetrics:
    """Performance metrics collected during load testing"""

    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    max_latency_ms: float
    requests_per_second: float
    peak_memory_mb: float
    peak_cpu_percent: float
    error_rate: float
    test_duration_seconds: float


class SystemMonitor:
    """Monitor system resources during testing"""

    def __init__(self):
        self.memory_usage = []
        self.cpu_usage = []
        self.monitoring = False
        self.monitor_thread = None

    def start_monitoring(self):
        """Start system monitoring in background thread"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)

    def _monitor_loop(self):
        """Background monitoring loop"""
        process = psutil.Process()
        while self.monitoring:
            try:
                memory_mb = process.memory_info().rss / (1024 * 1024)
                cpu_percent = process.cpu_percent()
                self.memory_usage.append(memory_mb)
                self.cpu_usage.append(cpu_percent)
                time.sleep(0.1)
            except Exception as e:
                logger.warning(f"Monitoring error: {e}")

    def get_peak_usage(self) -> tuple[float, float]:
        """Get peak memory and CPU usage"""
        peak_memory = max(self.memory_usage) if self.memory_usage else 0
        peak_cpu = max(self.cpu_usage) if self.cpu_usage else 0
        return peak_memory, peak_cpu


class MockModelRouter:
    """Mock model router for testing without API calls"""

    def __init__(self, latency_ms: int = 100):
        self.latency_ms = latency_ms
        self.call_count = 0

    def query(self, prompt: str, task_type: str = "general") -> str:
        """Simulate model query with configurable latency"""
        self.call_count += 1

        # Simulate network latency
        time.sleep(self.latency_ms / 1000.0)

        # Return mock response based on task type
        if "DAG" in prompt or "pipeline" in prompt:
            return self._generate_mock_dag_response()
        elif "error" in prompt or "fix" in prompt:
            return self._generate_mock_error_response()
        else:
            return '{"status": "success", "message": "Mock response"}'

    def _generate_mock_dag_response(self) -> str:
        """Generate realistic DAG structure"""
        num_tasks = random.randint(3, 15)
        tasks = []

        for i in range(num_tasks):
            task = {
                "id": f"task_{i}",
                "operator": random.choice(
                    [
                        "BashOperator",
                        "PythonOperator",
                        "SqlOperator",
                        "DockerOperator",
                        "KubernetesPodOperator",
                    ]
                ),
                "dependencies": [f"task_{j}" for j in range(max(0, i - 2), i)],
                "resources": {
                    "cpu": random.choice([1, 2, 4]),
                    "memory": random.choice([512, 1024, 2048]),
                },
            }
            tasks.append(task)

        return json.dumps(
            {
                "dag_id": f"generated_dag_{random.randint(1000, 9999)}",
                "description": "Auto-generated DAG for testing",
                "schedule_interval": "@daily",
                "tasks": tasks,
            }
        )

    def _generate_mock_error_response(self) -> str:
        """Generate mock error analysis"""
        return json.dumps(
            {
                "error_type": "ImportError",
                "confidence": 0.95,
                "suggested_fix": "pip install missing-package",
                "auto_fixable": True,
            }
        )


class LoadTestSuite:
    """Comprehensive load testing suite"""

    def __init__(self):
        self.monitor = SystemMonitor()
        self.results = []

    def run_concurrent_dag_generation_test(
        self,
        num_concurrent: int = 50,
        requests_per_thread: int = 10,
        model_latency_ms: int = 100,
    ) -> LoadTestMetrics:
        """Test concurrent DAG generation under load"""
        logger.info(
            f"Starting concurrent DAG generation test: {num_concurrent} threads, "
            f"{requests_per_thread} requests each"
        )

        self.monitor.start_monitoring()
        start_time = time.time()

        # Create orchestrator with mock model
        orchestrator = LLMOrchestrator(models=["mock-model"])
        orchestrator.model_router = MockModelRouter(latency_ms=model_latency_ms)

        latencies = []
        errors = []

        def worker_thread(thread_id: int) -> list[float]:
            """Worker thread that generates DAGs"""
            thread_latencies = []

            for i in range(requests_per_thread):
                request_start = time.time()
                try:
                    description = self._generate_random_pipeline_description()
                    orchestrator.generate_dag(description)
                    request_latency = (time.time() - request_start) * 1000
                    thread_latencies.append(request_latency)

                    if i % 5 == 0:
                        logger.debug(
                            f"Thread {thread_id}: completed {i+1}/{requests_per_thread}"
                        )

                except Exception as e:
                    errors.append(str(e))
                    logger.error(f"Thread {thread_id} error: {e}")

            return thread_latencies

        # Execute concurrent requests
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            future_to_thread = {
                executor.submit(worker_thread, i): i for i in range(num_concurrent)
            }

            for future in as_completed(future_to_thread):
                try:
                    thread_latencies = future.result()
                    latencies.extend(thread_latencies)
                except Exception as e:
                    logger.error(f"Thread execution error: {e}")
                    errors.append(str(e))

        end_time = time.time()
        self.monitor.stop_monitoring()

        return self._calculate_metrics(
            latencies,
            errors,
            start_time,
            end_time,
            num_concurrent * requests_per_thread,
        )

    def run_large_dag_stress_test(
        self, dag_sizes: list[int] = [100, 500, 1000, 2000], iterations: int = 5
    ) -> dict[int, LoadTestMetrics]:
        """Test performance with increasingly large DAGs"""
        logger.info(f"Starting large DAG stress test with sizes: {dag_sizes}")

        results = {}

        for dag_size in dag_sizes:
            logger.info(f"Testing DAG size: {dag_size} tasks")

            self.monitor.start_monitoring()
            start_time = time.time()

            orchestrator = LLMOrchestrator(models=["mock-model"])
            orchestrator.model_router = MockModelRouter(latency_ms=50)

            latencies = []
            errors = []

            for i in range(iterations):
                request_start = time.time()
                try:
                    description = self._generate_large_pipeline_description(dag_size)
                    orchestrator.generate_dag(description)
                    request_latency = (time.time() - request_start) * 1000
                    latencies.append(request_latency)

                except Exception as e:
                    errors.append(str(e))
                    logger.error(f"Large DAG error: {e}")

            end_time = time.time()
            self.monitor.stop_monitoring()

            results[dag_size] = self._calculate_metrics(
                latencies, errors, start_time, end_time, iterations
            )

        return results

    def run_memory_stress_test(
        self, duration_seconds: int = 300, request_rate: int = 10
    ) -> LoadTestMetrics:
        """Test memory usage under sustained load"""
        logger.info(
            f"Starting memory stress test: {duration_seconds}s at {request_rate} req/s"
        )

        self.monitor.start_monitoring()
        start_time = time.time()

        orchestrator = LLMOrchestrator(models=["mock-model"])
        orchestrator.model_router = MockModelRouter(latency_ms=50)

        latencies = []
        errors = []
        total_requests = 0

        while time.time() - start_time < duration_seconds:
            batch_start = time.time()

            # Generate batch of requests
            for _ in range(request_rate):
                request_start = time.time()
                try:
                    description = self._generate_random_pipeline_description()
                    orchestrator.generate_dag(description)
                    request_latency = (time.time() - request_start) * 1000
                    latencies.append(request_latency)
                    total_requests += 1

                except Exception as e:
                    errors.append(str(e))

            # Maintain request rate
            batch_duration = time.time() - batch_start
            if batch_duration < 1.0:
                time.sleep(1.0 - batch_duration)

        end_time = time.time()
        self.monitor.stop_monitoring()

        return self._calculate_metrics(
            latencies, errors, start_time, end_time, total_requests
        )

    def run_error_recovery_load_test(
        self, error_rate: float = 0.2, num_requests: int = 100
    ) -> LoadTestMetrics:
        """Test self-healing performance under error conditions"""
        logger.info(f"Starting error recovery test: {error_rate*100}% error rate")

        self.monitor.start_monitoring()
        start_time = time.time()

        healer = SelfHealingAgent()

        latencies = []
        errors = []

        for i in range(num_requests):
            request_start = time.time()
            try:
                # Simulate random errors
                if random.random() < error_rate:
                    error_type = random.choice(
                        [
                            "ImportError: No module named 'pandas'",
                            "MemoryError: Unable to allocate array",
                            "ConnectionError: Failed to connect to database",
                            "TimeoutError: Task execution timeout",
                        ]
                    )

                    analysis = healer.analyze_error(
                        Exception(error_type),
                        {"task_id": f"test_task_{i}", "dag_id": "test_dag"},
                    )

                    # Simulate fix application
                    if analysis.auto_fixable:
                        time.sleep(0.1)  # Simulate fix time

                request_latency = (time.time() - request_start) * 1000
                latencies.append(request_latency)

            except Exception as e:
                errors.append(str(e))

        end_time = time.time()
        self.monitor.stop_monitoring()

        return self._calculate_metrics(
            latencies, errors, start_time, end_time, num_requests
        )

    def _generate_random_pipeline_description(self) -> str:
        """Generate realistic pipeline descriptions"""
        templates = [
            "Extract data from PostgreSQL, transform with pandas, load to data warehouse",
            "Scrape web data, clean and validate, train ML model, deploy to production",
            "Process customer orders, update inventory, send notifications, generate reports",
            "Download files from S3, process with Spark, analyze results, store in database",
            "Read sensor data, detect anomalies, trigger alerts, archive historical data",
            "Fetch API data, enrich with external sources, apply business rules, export CSV",
            "Monitor system metrics, aggregate statistics, create dashboards, send summaries",
            "Import transaction data, calculate KPIs, generate financial reports, email stakeholders",
        ]
        return random.choice(templates)

    def _generate_large_pipeline_description(self, num_tasks: int) -> str:
        """Generate description for large DAG"""
        return (
            f"Complex data pipeline with {num_tasks} parallel processing tasks including "
            f"data extraction, transformation, validation, enrichment, analysis, and storage"
        )

    def _calculate_metrics(
        self,
        latencies: list[float],
        errors: list[str],
        start_time: float,
        end_time: float,
        total_requests: int,
    ) -> LoadTestMetrics:
        """Calculate comprehensive performance metrics"""

        peak_memory, peak_cpu = self.monitor.get_peak_usage()
        duration = end_time - start_time

        if latencies:
            avg_latency = statistics.mean(latencies)
            sorted_latencies = sorted(latencies)
            n = len(sorted_latencies)
            p50 = sorted_latencies[int(n * 0.5)]
            p95 = sorted_latencies[int(n * 0.95)]
            p99 = sorted_latencies[int(n * 0.99)]
            max_latency = max(sorted_latencies)
        else:
            avg_latency = p50 = p95 = p99 = max_latency = 0

        successful_requests = len(latencies)
        failed_requests = len(errors)
        requests_per_second = total_requests / duration if duration > 0 else 0
        error_rate = failed_requests / total_requests if total_requests > 0 else 0

        return LoadTestMetrics(
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            avg_latency_ms=avg_latency,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            max_latency_ms=max_latency,
            requests_per_second=requests_per_second,
            peak_memory_mb=peak_memory,
            peak_cpu_percent=peak_cpu,
            error_rate=error_rate,
            test_duration_seconds=duration,
        )


class LoadTestRunner:
    """Main test runner with reporting"""

    def __init__(self):
        self.suite = LoadTestSuite()

    def run_all_tests(self) -> dict[str, any]:
        """Run complete load testing suite"""
        logger.info("Starting comprehensive load testing suite")

        results = {}

        # Test 1: Concurrent DAG generation
        logger.info("\n=== Test 1: Concurrent DAG Generation ===")
        results["concurrent_light"] = self.suite.run_concurrent_dag_generation_test(
            num_concurrent=20, requests_per_thread=5, model_latency_ms=100
        )
        self._print_metrics("Concurrent Light Load", results["concurrent_light"])

        results["concurrent_heavy"] = self.suite.run_concurrent_dag_generation_test(
            num_concurrent=50, requests_per_thread=10, model_latency_ms=200
        )
        self._print_metrics("Concurrent Heavy Load", results["concurrent_heavy"])

        # Test 2: Large DAG stress test
        logger.info("\n=== Test 2: Large DAG Stress Test ===")
        results["large_dag"] = self.suite.run_large_dag_stress_test(
            dag_sizes=[50, 100, 500, 1000], iterations=3
        )
        for dag_size, metrics in results["large_dag"].items():
            self._print_metrics(f"DAG Size {dag_size}", metrics)

        # Test 3: Memory stress test
        logger.info("\n=== Test 3: Memory Stress Test ===")
        results["memory_stress"] = self.suite.run_memory_stress_test(
            duration_seconds=60, request_rate=5
        )
        self._print_metrics("Memory Stress Test", results["memory_stress"])

        # Test 4: Error recovery load test
        logger.info("\n=== Test 4: Error Recovery Load Test ===")
        results["error_recovery"] = self.suite.run_error_recovery_load_test(
            error_rate=0.3, num_requests=50
        )
        self._print_metrics("Error Recovery Test", results["error_recovery"])

        # Generate summary report
        self._generate_summary_report(results)

        return results

    def _print_metrics(self, test_name: str, metrics: LoadTestMetrics):
        """Print formatted test results"""
        print(f"\n{test_name} Results:")
        print(f"  Total Requests: {metrics.total_requests}")
        print(
            f"  Success Rate: {(metrics.successful_requests/metrics.total_requests)*100:.1f}%"
        )
        print(f"  Avg Latency: {metrics.avg_latency_ms:.1f}ms")
        print(f"  P95 Latency: {metrics.p95_latency_ms:.1f}ms")
        print(f"  P99 Latency: {metrics.p99_latency_ms:.1f}ms")
        print(f"  Requests/sec: {metrics.requests_per_second:.1f}")
        print(f"  Peak Memory: {metrics.peak_memory_mb:.1f} MB")
        print(f"  Peak CPU: {metrics.peak_cpu_percent:.1f}%")
        print(f"  Error Rate: {metrics.error_rate*100:.1f}%")

    def _generate_summary_report(self, results: dict[str, any]):
        """Generate comprehensive test report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"/Users/vanshverma/airflow-llm-orchestrator/load_test_report_{timestamp}.json"

        # Convert metrics to serializable format
        serializable_results = {}
        for test_name, result in results.items():
            if isinstance(result, dict):
                serializable_results[test_name] = {
                    str(k): vars(v) for k, v in result.items()
                }
            else:
                serializable_results[test_name] = vars(result)

        with open(report_file, "w") as f:
            json.dump(
                {
                    "timestamp": timestamp,
                    "system_info": {
                        "cpu_count": multiprocessing.cpu_count(),
                        "memory_gb": psutil.virtual_memory().total / (1024**3),
                        "python_version": sys.version,
                    },
                    "test_results": serializable_results,
                },
                f,
                indent=2,
            )

        logger.info(f"Load test report saved to: {report_file}")


def main():
    """Run load tests if executed directly"""
    runner = LoadTestRunner()
    results = runner.run_all_tests()

    # Performance assertions for CI/CD
    concurrent_metrics = results["concurrent_heavy"]
    assert (
        concurrent_metrics.error_rate < 0.05
    ), f"Error rate too high: {concurrent_metrics.error_rate}"
    assert (
        concurrent_metrics.p95_latency_ms < 5000
    ), f"P95 latency too high: {concurrent_metrics.p95_latency_ms}ms"
    assert (
        concurrent_metrics.requests_per_second > 10
    ), f"Throughput too low: {concurrent_metrics.requests_per_second} req/s"

    logger.info("All load tests passed!")


if __name__ == "__main__":
    main()
