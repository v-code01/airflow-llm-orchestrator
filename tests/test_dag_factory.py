"""Test suite for DAG factory functionality."""

import tempfile
from pathlib import Path

import yaml

from airflow_llm.dag_factory import (
    DAGConfig,
    EnterpriseDAGFactory,
    NaturalLanguageDAGGenerator,
    TaskConfig,
)


class TestEnterpriseDAGFactory:
    """Test enterprise DAG factory."""

    def test_basic_dag_generation(self):
        """Test basic DAG generation with datasci-rx structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            factory = EnterpriseDAGFactory(temp_dir)

            dag_config = DAGConfig(
                dag_id="test_pipeline",
                description="Test pipeline",
                owners=["test-team"],
                stakeholders=["business-team"],
            )

            tasks = [
                TaskConfig(name="start", operator="dummy", depends_on=["none"]),
                TaskConfig(
                    name="process",
                    operator="python",
                    depends_on=["start"],
                    python_callable="process_function",
                ),
                TaskConfig(name="finish", operator="dummy", depends_on=["process"]),
            ]

            dag_path = factory.generate_dag(
                description="Test pipeline", dag_config=dag_config, tasks=tasks
            )

            # Verify structure
            dag_file = Path(dag_path)
            assert dag_file.exists()

            resources_dir = dag_file.parent / "resources" / dag_config.dag_id
            assert resources_dir.exists()
            assert (resources_dir / "config.yml").exists()
            assert (resources_dir / "tasks.yml").exists()

    def test_config_yml_generation(self):
        """Test config.yml generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            factory = EnterpriseDAGFactory(temp_dir)

            dag_config = DAGConfig(
                dag_id="config_test",
                description="Config test",
                owners=["data-team"],
                stakeholders=["analytics-team"],
            )

            tasks = [TaskConfig(name="dummy", operator="dummy", depends_on=["none"])]

            factory.generate_dag("Test", dag_config, tasks)

            config_path = (
                Path(temp_dir) / "dags" / "resources" / "config_test" / "config.yml"
            )
            with open(config_path) as f:
                config = yaml.safe_load(f)

            assert "iam_role_s3" in config
            assert "s3_bucket" in config
            assert config["owners"] == "data-team"

    def test_tasks_yml_generation(self):
        """Test tasks.yml generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            factory = EnterpriseDAGFactory(temp_dir)

            dag_config = DAGConfig(dag_id="tasks_test", description="Tasks test")

            tasks = [
                TaskConfig(name="init", operator="dummy", depends_on=["none"]),
                TaskConfig(
                    name="sql_task",
                    operator="postgresql",
                    depends_on=["init"],
                    sql_script="query.sql",
                ),
            ]

            factory.generate_dag("Test", dag_config, tasks)

            tasks_path = (
                Path(temp_dir) / "dags" / "resources" / "tasks_test" / "tasks.yml"
            )
            with open(tasks_path) as f:
                tasks_config = yaml.safe_load(f)

            assert "tasks" in tasks_config
            assert len(tasks_config["tasks"]) == 2

            sql_task = next(t for t in tasks_config["tasks"] if t["name"] == "sql_task")
            assert sql_task["operator"] == "postgresql"
            assert sql_task["sql_script"] == "query.sql"


class TestNaturalLanguageDAGGenerator:
    """Test natural language DAG generator."""

    def test_dag_id_generation(self):
        """Test automatic DAG ID generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            factory = EnterpriseDAGFactory(temp_dir)
            generator = NaturalLanguageDAGGenerator(factory)

            dag_id = generator._generate_dag_id(
                "Extract customer data and process analytics"
            )

            assert "extract" in dag_id.lower()
            assert "customer" in dag_id.lower()
            assert "data" in dag_id.lower()

    def test_rule_based_parsing(self):
        """Test rule-based description parsing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            factory = EnterpriseDAGFactory(temp_dir)
            generator = NaturalLanguageDAGGenerator(factory)

            parsed = generator._rule_based_parse(
                "Extract data from database, transform it, and load to warehouse"
            )

            assert parsed["pipeline_type"] == "etl"
            assert len(parsed["tasks"]) > 0

            task_names = [task["name"] for task in parsed["tasks"]]
            assert any("extract" in name for name in task_names)
            assert any("transform" in name for name in task_names)
            assert any("load" in name for name in task_names)

    def test_sql_content_generation(self):
        """Test SQL content generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            factory = EnterpriseDAGFactory(temp_dir)
            generator = NaturalLanguageDAGGenerator(factory)

            task_info = {
                "name": "extract_sales",
                "type": "sql",
                "description": "Extract sales data",
            }

            sql_content = generator._generate_sql_content(task_info)

            assert "extract_sales" in sql_content
            assert "params.s3_bucket" in sql_content
            assert "params.iam_role_s3" in sql_content

    def test_python_content_generation(self):
        """Test Python function generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            factory = EnterpriseDAGFactory(temp_dir)
            generator = NaturalLanguageDAGGenerator(factory)

            task_info = {
                "name": "process_data",
                "type": "python",
                "description": "Process customer data",
            }

            python_content = generator._generate_python_content(task_info)

            assert "def process_data_function" in python_content
            assert "Process customer data" in python_content
            assert "**context" in python_content
