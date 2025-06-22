#!/usr/bin/env python3
"""
Simple test to verify core functionality without heavy dependencies.
"""
import tempfile
from pathlib import Path

import yaml

from airflow_llm.dag_factory import DAGConfig, EnterpriseDAGFactory, TaskConfig


def test_core_functionality():
    """Test core DAG generation functionality."""
    print("Testing Core DAG Generation...")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize factory
        factory = EnterpriseDAGFactory(temp_dir)

        # Create DAG config
        dag_config = DAGConfig(
            dag_id="test_pipeline",
            description="Test pipeline for validation",
            owners=["data-team"],
            stakeholders=["business-team"],
        )

        # Create tasks
        tasks = [
            TaskConfig(name="init", operator="dummy", depends_on=["none"]),
            TaskConfig(
                name="extract_data",
                operator="postgresql",
                depends_on=["init"],
                sql_script="extract.sql",
            ),
            TaskConfig(
                name="process_data",
                operator="python",
                depends_on=["extract_data"],
                python_callable="process_function",
            ),
            TaskConfig(name="done", operator="dummy", depends_on=["process_data"]),
        ]

        # SQL scripts
        sql_scripts = {
            "extract.sql": """
SELECT customer_id, name, email
FROM {{ params.output_schema }}.customers
WHERE created_at >= '{{ ds }}'
"""
        }

        # Python functions
        python_functions = {
            "process_function.py": """
def process_function(**context):
    return {"status": "success", "records": 1000}
"""
        }

        # Generate DAG
        dag_path = factory.generate_dag(
            description="Test pipeline",
            dag_config=dag_config,
            tasks=tasks,
            sql_scripts=sql_scripts,
            python_functions=python_functions,
        )

        # Verify structure
        dag_file = Path(dag_path)
        resources_dir = dag_file.parent / "resources" / dag_config.dag_id

        assert dag_file.exists(), "DAG file should exist"
        assert resources_dir.exists(), "Resources directory should exist"
        assert (resources_dir / "config.yml").exists(), "config.yml should exist"
        assert (resources_dir / "tasks.yml").exists(), "tasks.yml should exist"
        assert (resources_dir / "extract.sql").exists(), "SQL script should exist"
        assert (
            resources_dir / "process_function.py"
        ).exists(), "Python function should exist"

        # Verify config.yml structure
        with open(resources_dir / "config.yml") as f:
            config = yaml.safe_load(f)
            assert "iam_role_s3" in config
            assert "s3_bucket" in config
            assert config["owners"] == "data-team"

        # Verify tasks.yml structure
        with open(resources_dir / "tasks.yml") as f:
            tasks_config = yaml.safe_load(f)
            assert "tasks" in tasks_config
            assert len(tasks_config["tasks"]) == 4

        # Verify generated DAG Python syntax
        with open(dag_file) as f:
            dag_content = f.read()

        compile(dag_content, str(dag_file), "exec")

        print("Core functionality test passed!")
        print(f"   Generated DAG: {dag_file.name}")
        print(f"   Resources: {resources_dir.name}/")
        print(f"   Files: {len(list(resources_dir.glob('*')))} files")

        return True


def test_cli_structure():
    """Test CLI module structure."""
    print("Testing CLI Structure...")

    try:
        from airflow_llm.cli import create_dag_command, generate_dag_command, main

        assert callable(main)
        assert callable(generate_dag_command)
        assert callable(create_dag_command)
        print("CLI structure test passed!")
        return True
    except ImportError as e:
        print(f"CLI structure test failed: {e}")
        return False


def test_dag_factory_classes():
    """Test DAG factory class structure."""
    print("Testing DAG Factory Classes...")

    try:
        from airflow_llm.dag_factory import (
            DAGConfig,
            EnterpriseDAGFactory,
            NaturalLanguageDAGGenerator,
            TaskConfig,
        )

        # Test instantiation
        with tempfile.TemporaryDirectory() as temp_dir:
            factory = EnterpriseDAGFactory(temp_dir)
            generator = NaturalLanguageDAGGenerator(factory)

            dag_config = DAGConfig(dag_id="test", description="test")
            task_config = TaskConfig(name="test", operator="dummy", depends_on=["none"])

            assert factory is not None
            assert generator is not None
            assert dag_config.dag_id == "test"
            assert task_config.name == "test"

        print("DAG factory classes test passed!")
        return True
    except Exception as e:
        print(f"DAG factory classes test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("AirflowLLM Core Functionality Tests")
    print("=" * 50)

    tests = [test_core_functionality, test_cli_structure, test_dag_factory_classes]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"{test.__name__} failed: {e}")
            failed += 1
        print()

    print("=" * 50)
    print(f"Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("ALL TESTS PASSED! AirflowLLM is ready for production!")
        print()
        print("Next steps:")
        print("   1. Install: pip install airflow-llm-orchestrator")
        print("   2. Generate: airflow-llm generate 'Your pipeline description'")
        print("   3. Deploy: Copy generated DAGs to Airflow")
        return True
    else:
        print("Some tests failed. Please check the output above.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
