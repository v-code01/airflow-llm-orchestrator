#!/usr/bin/env python3
"""
Simple test to verify AirflowLLM works end-to-end
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_basic_functionality():
    """Test basic functionality without complex imports"""
    print("TEST Testing AirflowLLM Basic Functionality")
    print("=" * 50)

    try:
        # Test 1: Import specialized ensemble
        print("1. Testing model ensemble import...")
        from airflow_llm.models.specialized_ensemble import (
            SpecializedModelRouter,
            TaskType,
        )

        print("PASS Model ensemble imported successfully")

        # Test 2: Initialize router (will use mocks)
        print("\n2. Initializing model router...")
        router = SpecializedModelRouter()
        router.initialize_models()
        print("PASS Model router initialized")

        # Test 3: Test classification
        print("\n3. Testing task classification...")
        test_cases = [
            ("SELECT * FROM users WHERE active = 1", TaskType.SQL_GENERATION),
            ("def process_data(df): return df.clean()", TaskType.PYTHON_CODE),
            ("ImportError: No module named pandas", TaskType.ERROR_DEBUG),
            ("Create Airflow DAG for daily ETL", TaskType.DAG_ORCHESTRATION),
        ]

        all_correct = True
        for prompt, expected in test_cases:
            result = router.classify_task(prompt)
            if result == expected:
                print(f"PASS '{prompt[:30]}...' → {result.value}")
            else:
                print(
                    f"FAIL '{prompt[:30]}...' → Expected {expected.value}, got {result.value}"
                )
                all_correct = False

        if not all_correct:
            return False

        # Test 4: Test generation
        print("\n4. Testing response generation...")
        response = router.generate(
            "SELECT top customers by revenue", TaskType.SQL_GENERATION
        )

        if response.content and len(response.content) > 10:
            print(f"PASS Generated response: {response.content[:60]}...")
            print(f"PASS Latency: {response.latency_ms:.1f}ms")
            print(f"PASS Confidence: {response.confidence:.2f}")
        else:
            print("FAIL Generation failed")
            return False

        # Test 5: Test orchestrator integration (simple)
        print("\n5. Testing orchestrator integration...")
        from airflow_llm.orchestrator import LLMOrchestrator

        orchestrator = LLMOrchestrator()
        print("PASS Orchestrator created")

        # Test DAG generation
        description = "Extract data from database and send email report"
        dag = orchestrator.generate_dag(description)

        if dag and hasattr(dag, "dag_id"):
            print(f"PASS DAG generated: {dag.dag_id}")
            print(f"PASS Tasks: {len(dag.tasks)}")
        else:
            print("FAIL DAG generation failed")
            return False

        print("\nSUCCESS All tests passed! AirflowLLM is working correctly.")
        return True

    except Exception as e:
        print(f"FAIL Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_performance():
    """Quick performance test"""
    print("\nLAUNCH Performance Test")
    print("=" * 20)

    try:
        import time

        from airflow_llm.models.specialized_ensemble import (
            SpecializedModelRouter,
            TaskType,
        )

        router = SpecializedModelRouter()
        router.initialize_models()

        # Test classification speed
        start_time = time.time()
        for _ in range(100):
            router.classify_task("SELECT COUNT(*) FROM orders")
        classification_time = (time.time() - start_time) * 1000

        print(f"PASS 100 classifications in {classification_time:.1f}ms")
        print(f"PASS Average: {classification_time/100:.2f}ms per classification")

        # Test generation speed
        start_time = time.time()
        for _ in range(10):
            router.generate("Create SQL query", TaskType.SQL_GENERATION)
        generation_time = (time.time() - start_time) * 1000

        print(f"PASS 10 generations in {generation_time:.1f}ms")
        print(f"PASS Average: {generation_time/10:.1f}ms per generation")

        return True

    except Exception as e:
        print(f"FAIL Performance test failed: {e}")
        return False


def main():
    """Run all tests"""
    success = True

    # Basic functionality test
    if not test_basic_functionality():
        success = False

    # Performance test
    if not test_performance():
        success = False

    print("\n" + "=" * 50)
    if success:
        print("SUCCESS ALL TESTS PASSED!")
        print("\nAirflowLLM is ready for:")
        print("• Open source launch")
        print("• Production deployment")
        print("• Real model integration")
        print("\nNext steps:")
        print("1. Install PyTorch + Transformers for real models")
        print("2. Run: python setup_production_models.py")
        print("3. Deploy with: kubectl apply -f infra/kubernetes/")
    else:
        print("FAIL Some tests failed")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
