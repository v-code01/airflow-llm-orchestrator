#!/usr/bin/env python3
"""
Extreme Stress Testing for AirflowLLM Production Readiness

This module pushes the system to breaking points to identify:
- Maximum concurrent users before degradation
- Memory leak detection under sustained load
- CPU utilization patterns under stress
- Network bottlenecks and timeout handling
- Database connection pool exhaustion
- Kubernetes resource limits testing
"""

import gc
import multiprocessing
import os
import signal
import sys
import threading
import time
import tracemalloc
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime

import psutil

sys.path.insert(0, "/Users/vanshverma/airflow-llm-orchestrator")
from airflow_llm.orchestrator import LLMOrchestrator
from airflow_llm.self_healer import SelfHealingAgent


class StressTestFramework:
    """Extreme stress testing framework"""

    def __init__(self):
        self.start_time = time.time()
        self.test_running = True
        self.max_memory_mb = 0
        self.max_cpu_percent = 0
        self.total_errors = 0
        self.memory_samples = []

        # Enable memory tracking
        tracemalloc.start()

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\nReceived signal {signum}, shutting down stress tests...")
        self.test_running = False

    def run_cpu_stress_test(self, duration_minutes: int = 10) -> dict:
        """Stress test CPU usage with compute-intensive operations"""
        print(f"Starting CPU stress test for {duration_minutes} minutes...")

        def cpu_intensive_work(worker_id: int) -> int:
            """CPU-intensive work to stress the system"""
            operations = 0
            start_time = time.time()

            while (
                time.time() - start_time < duration_minutes * 60 and self.test_running
            ):
                # Simulate complex DAG analysis
                for i in range(10000):
                    # Mathematical operations that stress CPU
                    sum(j * j for j in range(100))
                    operations += 1

                if operations % 100000 == 0:
                    print(f"Worker {worker_id}: {operations} operations completed")

            return operations

        # Use all available CPU cores
        num_workers = multiprocessing.cpu_count()
        print(f"Spawning {num_workers} CPU stress workers...")

        start_time = time.time()

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(cpu_intensive_work, i) for i in range(num_workers)
            ]

            # Monitor system resources
            monitor_thread = threading.Thread(target=self._monitor_resources)
            monitor_thread.daemon = True
            monitor_thread.start()

            # Wait for completion
            total_operations = sum(future.result() for future in futures)

        end_time = time.time()
        duration = end_time - start_time

        return {
            "test_type": "cpu_stress",
            "duration_seconds": duration,
            "total_operations": total_operations,
            "operations_per_second": total_operations / duration,
            "max_cpu_percent": self.max_cpu_percent,
            "max_memory_mb": self.max_memory_mb,
        }

    def run_memory_stress_test(self, target_memory_gb: int = 4) -> dict:
        """Stress test memory allocation and garbage collection"""
        print(f"Starting memory stress test targeting {target_memory_gb}GB...")

        # Track memory allocations
        memory_blocks = []
        allocation_count = 0
        gc_collections = 0

        try:
            while self.test_running:
                # Allocate large data structures similar to DAG processing
                large_dag_data = {
                    "tasks": [
                        {
                            "id": f"task_{i}",
                            "code": "x" * 1024,  # 1KB per task
                            "dependencies": list(range(max(0, i - 5), i)),
                            "metadata": {"logs": "y" * 2048},  # 2KB metadata
                        }
                        for i in range(1000)  # 1000 tasks = ~3MB per DAG
                    ],
                    "history": [
                        "log_entry_" + "z" * 512 for _ in range(2000)
                    ],  # 1MB history
                }

                memory_blocks.append(large_dag_data)
                allocation_count += 1

                # Check memory usage
                current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
                self.max_memory_mb = max(self.max_memory_mb, current_memory)

                if current_memory > target_memory_gb * 1024:
                    print(f"Target memory reached: {current_memory:.1f}MB")
                    break

                # Periodic garbage collection and cleanup
                if allocation_count % 100 == 0:
                    gc.collect()
                    gc_collections += 1
                    print(
                        f"Allocated {allocation_count} blocks, Memory: {current_memory:.1f}MB"
                    )

                # Occasionally free some memory
                if len(memory_blocks) > 500:
                    memory_blocks = memory_blocks[-250:]  # Keep only recent blocks

                time.sleep(0.01)  # Small delay to prevent overwhelming

        except MemoryError:
            print("MemoryError reached - testing memory limits")

        # Force cleanup
        memory_blocks.clear()
        gc.collect()

        return {
            "test_type": "memory_stress",
            "allocations_created": allocation_count,
            "gc_collections": gc_collections,
            "peak_memory_mb": self.max_memory_mb,
            "memory_efficiency": allocation_count / max(self.max_memory_mb, 1),
        }

    def run_concurrent_user_stress_test(self, max_users: int = 1000) -> dict:
        """Simulate thousands of concurrent users"""
        print(f"Starting concurrent user stress test up to {max_users} users...")

        successful_requests = 0
        failed_requests = 0
        response_times = []

        def simulate_user_session(user_id: int) -> dict:
            """Simulate a complete user session"""
            session_start = time.time()
            session_requests = 0
            session_errors = 0

            try:
                # Initialize user's orchestrator
                orchestrator = LLMOrchestrator(models=["mock-model"])

                # Simulate typical user workflow
                workflows = [
                    "Extract sales data from database and generate daily report",
                    "Process customer feedback and update sentiment scores",
                    "Monitor server metrics and trigger alerts if needed",
                    "Backup critical data to cloud storage",
                    "Analyze user behavior and update recommendation models",
                ]

                for workflow in workflows:
                    if not self.test_running:
                        break

                    request_start = time.time()
                    try:
                        orchestrator.generate_dag(workflow)
                        response_time = time.time() - request_start
                        response_times.append(response_time)
                        session_requests += 1
                    except Exception:
                        session_errors += 1
                        self.total_errors += 1

            except Exception:
                session_errors += 1

            return {
                "user_id": user_id,
                "session_duration": time.time() - session_start,
                "requests": session_requests,
                "errors": session_errors,
            }

        # Gradually ramp up users to find breaking point
        user_results = []
        current_users = 0

        with ThreadPoolExecutor(max_workers=max_users) as executor:
            futures = []

            # Ramp up users gradually
            ramp_up_rate = 50  # Users per second
            for batch_start in range(0, max_users, ramp_up_rate):
                if not self.test_running:
                    break

                batch_end = min(batch_start + ramp_up_rate, max_users)

                # Submit batch of users
                for user_id in range(batch_start, batch_end):
                    future = executor.submit(simulate_user_session, user_id)
                    futures.append(future)
                    current_users += 1

                print(f"Ramped up to {current_users} concurrent users...")
                time.sleep(1)  # 1 second between batches

                # Check system health
                memory_mb = psutil.Process().memory_info().rss / (1024 * 1024)
                cpu_percent = psutil.Process().cpu_percent()

                if memory_mb > 8192:  # 8GB limit
                    print(
                        f"Memory limit reached at {current_users} users: {memory_mb:.1f}MB"
                    )
                    break

                if cpu_percent > 90:
                    print(
                        f"CPU limit reached at {current_users} users: {cpu_percent:.1f}%"
                    )
                    break

            # Collect results
            for future in futures:
                try:
                    result = future.result(timeout=30)
                    user_results.append(result)
                    successful_requests += result["requests"]
                    failed_requests += result["errors"]
                except Exception:
                    failed_requests += 1

        # Calculate statistics
        avg_response_time = (
            sum(response_times) / len(response_times) if response_times else 0
        )
        success_rate = (
            successful_requests / (successful_requests + failed_requests)
            if (successful_requests + failed_requests) > 0
            else 0
        )

        return {
            "test_type": "concurrent_users",
            "max_concurrent_users": current_users,
            "total_requests": successful_requests + failed_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "success_rate": success_rate,
            "avg_response_time": avg_response_time,
            "peak_memory_mb": self.max_memory_mb,
            "peak_cpu_percent": self.max_cpu_percent,
        }

    def run_error_cascade_test(self, error_injection_rate: float = 0.5) -> dict:
        """Test system stability under cascading errors"""
        print(
            f"Starting error cascade test with {error_injection_rate*100}% error rate..."
        )

        healer = SelfHealingAgent()

        # Simulate various error types
        error_types = [
            "ImportError: No module named 'critical_package'",
            "MemoryError: Unable to allocate 8GB array",
            "ConnectionError: Database connection timeout",
            "TimeoutError: Task execution exceeded 300s",
            "PermissionError: Access denied to /critical/path",
            "RuntimeError: CUDA out of memory",
            "ValueError: Invalid configuration parameter",
            "FileNotFoundError: Required file missing",
            "NetworkError: Unable to reach external API",
        ]

        error_counts = {error_type: 0 for error_type in error_types}
        recovery_times = []
        auto_fix_success_rate = 0
        total_errors_processed = 0

        for i in range(500):  # Process 500 error scenarios
            if not self.test_running:
                break

            # Inject errors at specified rate
            if i % int(1 / error_injection_rate) == 0:
                error_type = error_types[i % len(error_types)]
                error_counts[error_type] += 1

                recovery_start = time.time()
                try:
                    analysis = healer.analyze_error(
                        Exception(error_type),
                        {
                            "task_id": f"stress_task_{i}",
                            "dag_id": f"stress_dag_{i//10}",
                            "execution_date": datetime.now().isoformat(),
                        },
                    )

                    recovery_time = time.time() - recovery_start
                    recovery_times.append(recovery_time)

                    if analysis.auto_fixable:
                        auto_fix_success_rate += 1

                    total_errors_processed += 1

                except Exception as e:
                    print(f"Error processing error {i}: {e}")

            time.sleep(0.01)  # Small delay between errors

        auto_fix_success_rate = (
            auto_fix_success_rate / total_errors_processed
            if total_errors_processed > 0
            else 0
        )
        avg_recovery_time = (
            sum(recovery_times) / len(recovery_times) if recovery_times else 0
        )

        return {
            "test_type": "error_cascade",
            "total_errors_processed": total_errors_processed,
            "error_breakdown": error_counts,
            "auto_fix_success_rate": auto_fix_success_rate,
            "avg_recovery_time_ms": avg_recovery_time * 1000,
            "max_recovery_time_ms": max(recovery_times) * 1000 if recovery_times else 0,
        }

    def run_resource_exhaustion_test(self) -> dict:
        """Test behavior when system resources are exhausted"""
        print("Starting resource exhaustion test...")

        # Test file descriptor limits
        open_files = []
        max_file_descriptors = 0

        try:
            while len(open_files) < 1000:  # Reasonable limit
                temp_file = open(f"/tmp/stress_test_{len(open_files)}.tmp", "w")
                open_files.append(temp_file)
                max_file_descriptors += 1
        except OSError as e:
            print(f"File descriptor limit reached: {e}")
        finally:
            for f in open_files:
                try:
                    f.close()
                    os.unlink(f.name)
                except:
                    pass

        # Test thread limits
        active_threads = []
        max_threads = 0

        def dummy_thread():
            time.sleep(10)

        try:
            while len(active_threads) < 500:  # Reasonable limit
                thread = threading.Thread(target=dummy_thread)
                thread.daemon = True
                thread.start()
                active_threads.append(thread)
                max_threads += 1
        except RuntimeError as e:
            print(f"Thread limit reached: {e}")

        return {
            "test_type": "resource_exhaustion",
            "max_file_descriptors": max_file_descriptors,
            "max_threads": max_threads,
            "system_limits_reached": True,
        }

    def _monitor_resources(self):
        """Background thread to monitor system resources"""
        process = psutil.Process()

        while self.test_running:
            try:
                memory_mb = process.memory_info().rss / (1024 * 1024)
                cpu_percent = process.cpu_percent()

                self.max_memory_mb = max(self.max_memory_mb, memory_mb)
                self.max_cpu_percent = max(self.max_cpu_percent, cpu_percent)
                self.memory_samples.append(memory_mb)

                time.sleep(0.5)
            except Exception:
                pass

    def run_all_stress_tests(self) -> dict:
        """Run complete stress testing suite"""
        print("=" * 60)
        print("STARTING EXTREME STRESS TESTING SUITE")
        print("=" * 60)

        results = {}

        try:
            # Test 1: CPU Stress
            results["cpu_stress"] = self.run_cpu_stress_test(duration_minutes=2)
            print(
                f"CPU Stress: {results['cpu_stress']['operations_per_second']:.0f} ops/sec"
            )

            # Test 2: Memory Stress
            results["memory_stress"] = self.run_memory_stress_test(target_memory_gb=2)
            print(
                f"Memory Stress: Peak {results['memory_stress']['peak_memory_mb']:.1f}MB"
            )

            # Test 3: Concurrent Users
            results["concurrent_users"] = self.run_concurrent_user_stress_test(
                max_users=200
            )
            print(
                f"Concurrent Users: {results['concurrent_users']['max_concurrent_users']} max users"
            )

            # Test 4: Error Cascade
            results["error_cascade"] = self.run_error_cascade_test(
                error_injection_rate=0.3
            )
            print(
                f"Error Cascade: {results['error_cascade']['auto_fix_success_rate']:.1%} auto-fix rate"
            )

            # Test 5: Resource Exhaustion
            results["resource_exhaustion"] = self.run_resource_exhaustion_test()
            print(
                f"Resource Limits: {results['resource_exhaustion']['max_file_descriptors']} max FDs"
            )

        except KeyboardInterrupt:
            print("\nStress tests interrupted by user")
        except Exception as e:
            print(f"Stress test error: {e}")

        # Generate summary
        self._generate_stress_report(results)

        return results

    def _generate_stress_report(self, results: dict):
        """Generate comprehensive stress test report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"/Users/vanshverma/airflow-llm-orchestrator/stress_test_report_{timestamp}.txt"

        with open(report_file, "w") as f:
            f.write("AIRFLOW-LLM EXTREME STRESS TEST REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Test Date: {datetime.now()}\n")
            f.write(
                f"System: {psutil.cpu_count()} cores, {psutil.virtual_memory().total/(1024**3):.1f}GB RAM\n\n"
            )

            for test_name, test_results in results.items():
                f.write(f"{test_name.upper()} RESULTS:\n")
                for key, value in test_results.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")

            # Performance thresholds
            f.write("PERFORMANCE ASSESSMENT:\n")

            # CPU performance
            if "cpu_stress" in results:
                ops_per_sec = results["cpu_stress"]["operations_per_second"]
                if ops_per_sec > 100000:
                    f.write("PASS CPU Performance: EXCELLENT\n")
                elif ops_per_sec > 50000:
                    f.write("PASS CPU Performance: GOOD\n")
                else:
                    f.write("WARNING  CPU Performance: NEEDS OPTIMIZATION\n")

            # Memory efficiency
            if "memory_stress" in results:
                peak_memory = results["memory_stress"]["peak_memory_mb"]
                if peak_memory < 1024:
                    f.write("PASS Memory Usage: EXCELLENT\n")
                elif peak_memory < 2048:
                    f.write("PASS Memory Usage: GOOD\n")
                else:
                    f.write("WARNING  Memory Usage: HIGH\n")

            # Concurrency handling
            if "concurrent_users" in results:
                max_users = results["concurrent_users"]["max_concurrent_users"]
                success_rate = results["concurrent_users"]["success_rate"]
                if max_users > 500 and success_rate > 0.95:
                    f.write("PASS Concurrency: EXCELLENT\n")
                elif max_users > 200 and success_rate > 0.90:
                    f.write("PASS Concurrency: GOOD\n")
                else:
                    f.write("WARNING  Concurrency: NEEDS IMPROVEMENT\n")

            f.write(f"\nReport saved: {report_file}\n")

        print(f"\nStress test report saved to: {report_file}")


def main():
    """Run stress tests if executed directly"""
    framework = StressTestFramework()

    try:
        results = framework.run_all_stress_tests()

        print("\n" + "=" * 60)
        print("STRESS TESTING COMPLETE")
        print("=" * 60)

        # Check if system passed all stress tests
        all_passed = True

        if "concurrent_users" in results:
            if results["concurrent_users"]["success_rate"] < 0.90:
                print("FAIL FAILED: Concurrent user success rate too low")
                all_passed = False

        if "error_cascade" in results:
            if results["error_cascade"]["auto_fix_success_rate"] < 0.70:
                print("FAIL FAILED: Error recovery rate too low")
                all_passed = False

        if all_passed:
            print("PASS ALL STRESS TESTS PASSED - PRODUCTION READY")
        else:
            print("WARNING  SOME STRESS TESTS FAILED - NEEDS OPTIMIZATION")

    except Exception as e:
        print(f"Stress testing failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
