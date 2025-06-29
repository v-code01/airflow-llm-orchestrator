#!/usr/bin/env python3
"""
AirflowLLM Working Demo
Real AI-powered DAG generation using Ollama
"""

import asyncio
import logging
import time
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


async def check_ollama():
    """Check if Ollama is available and suggest installation if not"""
    from airflow_llm.ollama_backend import ollama_backend

    async with ollama_backend as backend:
        if await backend.check_connection():
            models = await backend.list_models()
            logger.info(f"✅ Ollama is running with {len(models)} models")
            return True, models
        else:
            logger.error("❌ Ollama is not running")
            print("\n🚀 To install Ollama:")
            print("   curl -fsSL https://ollama.com/install.sh | sh")
            print("   ollama serve")
            print("   ollama pull phi3:mini")
            return False, []


async def ensure_model(model_name: str = "phi3:mini"):
    """Ensure we have a working model"""
    from airflow_llm.ollama_backend import ollama_backend

    async with ollama_backend as backend:
        models = await backend.list_models()

        if model_name in models:
            logger.info(f"✅ Model {model_name} is available")
            return True

        logger.info(f"📥 Pulling model {model_name}...")
        success = await backend.pull_model(model_name)

        if success:
            logger.info(f"✅ Successfully pulled {model_name}")
            return True
        else:
            logger.error(f"❌ Failed to pull {model_name}")
            return False


async def test_basic_generation():
    """Test basic model generation"""
    from airflow_llm.model_server import model_server

    logger.info("🧪 Testing basic model generation...")

    try:
        # Initialize with Ollama backend
        await model_server.initialize_models(["phi3-mini"], backend="ollama")

        # Test generation
        time.time()
        result = await model_server.generate(
            "Write a simple Python function that adds two numbers:",
            model_name="phi3-mini",
            max_tokens=100,
            temperature=0.7,
        )

        if result.success:
            logger.info(f"✅ Generation successful in {result.latency_ms:.1f}ms")
            logger.info(f"🎯 Tokens/sec: {result.tokens_per_second:.1f}")
            print(f"\n📝 Generated Code:\n{result.text}\n")
            return True
        else:
            logger.error(f"❌ Generation failed: {result.error}")
            return False

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False


async def generate_real_dag():
    """Generate a real DAG using AI"""
    import tempfile

    from airflow_llm.orchestrator import LLMOrchestrator

    logger.info("🏗️ Generating real DAG with AI...")

    try:
        # Initialize orchestrator with Ollama
        orchestrator = LLMOrchestrator(
            models=["phi3-mini"], cost_optimization=True, self_healing=True
        )

        # Simple but real DAG generation request
        description = """
        Create a daily data pipeline that:
        1. Extracts customer data from a PostgreSQL database
        2. Validates the data quality
        3. Transforms the data by calculating customer lifetime value
        4. Loads the results into a data warehouse
        5. Sends a Slack notification when complete
        """

        start_time = time.time()

        # This will use REAL AI generation (no templates!)
        try:
            dag_result = await orchestrator.generate_dag(description)
        except Exception as e:
            logger.error(f"DAG generation error: {e}")
            import traceback

            traceback.print_exc()
            raise

        generation_time = (time.time() - start_time) * 1000

        if hasattr(dag_result, "dag_id"):
            logger.info(f"✅ DAG generated in {generation_time:.1f}ms")
            logger.info(f"📁 DAG ID: {dag_result.dag_id}")

            # Save to temp file to show it's real
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(str(dag_result))
                temp_path = f.name

            file_size = Path(temp_path).stat().st_size
            logger.info(f"📄 Generated file: {temp_path} ({file_size:,} bytes)")

            # Show first few lines
            with open(temp_path) as f:
                lines = f.readlines()[:15]
                print("\n📜 Generated DAG Preview:")
                print("-" * 40)
                for i, line in enumerate(lines, 1):
                    print(f"{i:2d}│ {line.rstrip()}")
                if len(lines) >= 15:
                    print("   ... (truncated)")

            return True
        else:
            logger.error("❌ DAG generation failed - no DAG returned")
            return False

    except Exception as e:
        logger.error(f"❌ DAG generation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def run_performance_test():
    """Test performance with different prompts"""
    from airflow_llm.model_server import model_server

    logger.info("⚡ Running performance tests...")

    test_prompts = [
        "Generate SQL to find top 10 customers by revenue",
        "Write Python code to validate email addresses",
        "Create error handling for database connection failures",
        "Generate logging statements for a data pipeline",
    ]

    results = []

    for i, prompt in enumerate(test_prompts):
        try:
            time.time()
            result = await model_server.generate(
                prompt, model_name="phi3-mini", max_tokens=150, temperature=0.3
            )

            if result.success:
                results.append(
                    {
                        "prompt": prompt[:30] + "...",
                        "latency_ms": result.latency_ms,
                        "tokens_per_sec": result.tokens_per_second,
                        "success": True,
                    }
                )
                logger.info(
                    f"✅ Test {i+1}: {result.latency_ms:.1f}ms, {result.tokens_per_second:.1f} tok/s"
                )
            else:
                results.append(
                    {
                        "prompt": prompt[:30] + "...",
                        "latency_ms": 0,
                        "tokens_per_sec": 0,
                        "success": False,
                    }
                )
                logger.error(f"❌ Test {i+1} failed: {result.error}")

        except Exception as e:
            logger.error(f"❌ Test {i+1} crashed: {e}")
            results.append(
                {
                    "prompt": prompt[:30] + "...",
                    "latency_ms": 0,
                    "tokens_per_sec": 0,
                    "success": False,
                }
            )

    # Performance summary
    successful_results = [r for r in results if r["success"]]
    if successful_results:
        avg_latency = sum(r["latency_ms"] for r in successful_results) / len(
            successful_results
        )
        avg_throughput = sum(r["tokens_per_sec"] for r in successful_results) / len(
            successful_results
        )

        logger.info(f"📊 Performance Summary:")
        logger.info(f"   Average latency: {avg_latency:.1f}ms")
        logger.info(f"   Average throughput: {avg_throughput:.1f} tokens/sec")
        logger.info(
            f"   Success rate: {len(successful_results)}/{len(results)} ({len(successful_results)/len(results)*100:.0f}%)"
        )

    return len(successful_results) > 0


async def main():
    """Main demo function"""
    print("🚀 AirflowLLM Working Demo")
    print("=" * 50)
    print("Real AI-powered DAG generation")
    print("=" * 50)

    # Step 1: Check Ollama
    ollama_ok, models = await check_ollama()
    if not ollama_ok:
        return

    # Step 2: Ensure we have a model
    if not await ensure_model("phi3:mini"):
        logger.error("Cannot proceed without a working model")
        return

    # Step 3: Test basic generation
    if not await test_basic_generation():
        logger.error("Basic generation test failed")
        return

    # Step 4: Generate real DAG
    if not await generate_real_dag():
        logger.error("DAG generation failed")
        return

    # Step 5: Performance test
    if not await run_performance_test():
        logger.error("Performance test failed")
        return

    print("\n" + "=" * 50)
    print("🎉 SUCCESS: Real AI-powered DAG generation working!")
    print("=" * 50)
    print("✅ Models: Working with Ollama backend")
    print("✅ Generation: Real AI-generated code (no templates)")
    print("✅ Performance: Measured inference times")
    print("✅ Integration: End-to-end DAG creation")
    print("\n💡 This proves the architecture works with real models!")


if __name__ == "__main__":
    asyncio.run(main())
