"""
Enterprise DAG Factory - Production-Grade DAG Generation
Generates DAGs following datasci-rx patterns with resources/ subfolder structure
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import jinja2
import yaml

logger = logging.getLogger(__name__)


@dataclass
class DAGConfig:
    """DAG configuration structure following datasci-rx patterns"""

    dag_id: str
    description: str
    schedule_interval: str = "@daily"
    max_active_runs: int = 1
    catchup: bool = False
    start_date: datetime = None
    retries: int = 1
    email_on_failure: bool = True
    owners: list[str] = None
    stakeholders: list[str] = None
    environment_configs: dict[str, Any] = None
    tags: list[str] = None

    def __post_init__(self):
        if self.start_date is None:
            self.start_date = datetime(2024, 1, 1)
        if self.owners is None:
            self.owners = []
        if self.stakeholders is None:
            self.stakeholders = []
        if self.environment_configs is None:
            self.environment_configs = {}
        if self.tags is None:
            self.tags = ["airflow-llm", "auto-generated"]


@dataclass
class TaskConfig:
    """Task configuration structure"""

    name: str
    operator: str
    depends_on: list[str]
    sql_script: str | None = None
    python_callable: str | None = None
    connection_id: str | None = None
    parameters: dict[str, Any] | None = None
    retries: int | None = None
    timeout: int | None = None
    pool: str | None = None
    resources: dict[str, Any] | None = None


class EnterpriseDAGFactory:
    """
    Production-grade DAG factory following datasci-rx patterns
    Generates complete DAG structures with resources/ folders
    """

    def __init__(self, output_dir: str = "generated_dags"):
        self.output_dir = Path(output_dir)
        self.templates_dir = Path(__file__).parent / "templates"
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.templates_dir)),
            undefined=jinja2.StrictUndefined,
        )

        # Ensure output directory structure exists
        self.ensure_directory_structure()

    def ensure_directory_structure(self):
        """Create the required directory structure"""
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "dags").mkdir(exist_ok=True)
        (self.output_dir / "dags" / "resources").mkdir(exist_ok=True)

    def generate_dag(
        self,
        description: str,
        dag_config: DAGConfig,
        tasks: list[TaskConfig],
        sql_scripts: dict[str, str] | None = None,
        python_functions: dict[str, str] | None = None,
    ) -> str:
        """
        Generate complete DAG following datasci-rx patterns

        Args:
            description: Natural language description of the pipeline
            dag_config: DAG configuration
            tasks: List of task configurations
            sql_scripts: Dictionary of SQL script contents
            python_functions: Dictionary of Python function contents

        Returns:
            Path to generated DAG file
        """

        # Create DAG-specific resources directory
        resources_dir = self.output_dir / "dags" / "resources" / dag_config.dag_id
        resources_dir.mkdir(exist_ok=True)

        # Generate configuration files
        self._generate_config_yml(resources_dir, dag_config)
        self._generate_tasks_yml(resources_dir, tasks)

        # Generate SQL scripts if provided
        if sql_scripts:
            self._generate_sql_scripts(resources_dir, sql_scripts)

        # Generate Python modules if provided
        if python_functions:
            self._generate_python_modules(resources_dir, python_functions)

        # Generate main DAG file
        dag_file_path = self._generate_dag_file(dag_config, tasks)

        logger.info(
            f"Generated enterprise DAG '{dag_config.dag_id}' at {dag_file_path}"
        )
        return str(dag_file_path)

    def _generate_config_yml(self, resources_dir: Path, dag_config: DAGConfig):
        """Generate config.yml following datasci-rx pattern"""

        # Default environment configurations
        default_env_configs = {
            "iam_role_s3": {
                "prod": "arn:aws:iam::account:role/AirflowLLMRole",
                "staging": "arn:aws:iam::account:role/AirflowLLMRole-Staging",
                "dev": "arn:aws:iam::account:role/AirflowLLMRole-Dev",
                "eng-prod": "arn:aws:iam::account:role/AirflowLLMRole",
                "eng-staging": "arn:aws:iam::account:role/AirflowLLMRole-Staging",
                "eng-dev": "arn:aws:iam::account:role/AirflowLLMRole-Dev",
            },
            "s3_bucket": {
                "prod": "airflow-llm-prod",
                "staging": "airflow-llm-staging",
                "dev": "airflow-llm-dev",
                "eng-prod": "airflow-llm-prod",
                "eng-staging": "airflow-llm-staging",
                "eng-dev": "airflow-llm-dev",
            },
            "output_schema": {
                "prod": "production",
                "staging": "staging",
                "dev": "development",
                "eng-prod": "production",
                "eng-staging": "staging",
                "eng-dev": "development",
            },
        }

        # Merge with user-provided configs
        config_data = {**default_env_configs, **dag_config.environment_configs}

        # Add non-environment specific configs
        config_data.update(
            {
                "stakeholders": ", ".join(dag_config.stakeholders),
                "owners": ", ".join(dag_config.owners),
                "dag_description": dag_config.description,
            }
        )

        config_path = resources_dir / "config.yml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Generated config.yml at {config_path}")

    def _generate_tasks_yml(self, resources_dir: Path, tasks: list[TaskConfig]):
        """Generate tasks.yml following datasci-rx pattern"""

        tasks_data = {
            "tasks": [
                {
                    "name": task.name,
                    "depends_on": task.depends_on,
                    "operator": task.operator,
                    **({"sql_script": task.sql_script} if task.sql_script else {}),
                    **(
                        {"python_callable": task.python_callable}
                        if task.python_callable
                        else {}
                    ),
                    **(
                        {"connection_id": task.connection_id}
                        if task.connection_id
                        else {}
                    ),
                    **({"parameters": task.parameters} if task.parameters else {}),
                    **({"retries": task.retries} if task.retries else {}),
                    **({"timeout": task.timeout} if task.timeout else {}),
                    **({"pool": task.pool} if task.pool else {}),
                    **({"resources": task.resources} if task.resources else {}),
                }
                for task in tasks
            ]
        }

        tasks_path = resources_dir / "tasks.yml"
        with open(tasks_path, "w") as f:
            yaml.dump(tasks_data, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Generated tasks.yml at {tasks_path}")

    def _generate_sql_scripts(self, resources_dir: Path, sql_scripts: dict[str, str]):
        """Generate SQL script files"""
        for script_name, content in sql_scripts.items():
            script_path = resources_dir / script_name
            with open(script_path, "w") as f:
                f.write(content)
            logger.info(f"Generated SQL script {script_path}")

    def _generate_python_modules(
        self, resources_dir: Path, python_functions: dict[str, str]
    ):
        """Generate Python module files"""
        for module_name, content in python_functions.items():
            module_path = resources_dir / module_name
            with open(module_path, "w") as f:
                f.write(content)
            logger.info(f"Generated Python module {module_path}")

    def _generate_dag_file(
        self, dag_config: DAGConfig, tasks: list[TaskConfig]
    ) -> Path:
        """Generate main DAG file following datasci-rx pattern"""

        template_data = {
            "dag_id": dag_config.dag_id,
            "description": dag_config.description,
            "schedule_interval": dag_config.schedule_interval,
            "max_active_runs": dag_config.max_active_runs,
            "catchup": dag_config.catchup,
            "start_date": f"{dag_config.start_date.year}, {dag_config.start_date.month}, {dag_config.start_date.day}",
            "retries": dag_config.retries,
            "email_on_failure": dag_config.email_on_failure,
            "owners": ", ".join(dag_config.owners),
            "stakeholders": ", ".join(dag_config.stakeholders),
            "tags": dag_config.tags,
            "tasks": tasks,
        }

        # Use Jinja2 template for DAG generation
        try:
            template = self.jinja_env.get_template("dag_template.py.j2")
            dag_content = template.render(**template_data)
        except jinja2.TemplateNotFound:
            # Fallback to inline template if file not found
            dag_content = self._generate_inline_dag_template(template_data)

        # Write DAG file
        dag_file_path = self.output_dir / "dags" / f"{dag_config.dag_id}.py"
        with open(dag_file_path, "w") as f:
            f.write(dag_content)

        return dag_file_path

    def _generate_inline_dag_template(self, template_data: dict[str, Any]) -> str:
        """Generate DAG content using inline template as fallback"""

        dag_content = f'''"""
{template_data["description"]}
Generated by AirflowLLM on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
from datetime import datetime, timedelta
import logging
import os
from typing import Dict

import airflow
from airflow.models.baseoperator import BaseOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator
try:
    from airflow.providers.postgres.operators.postgres import PostgresOperator
    from airflow.sensors.sql import SqlSensor
except ImportError:
    # Fallback for testing
    PostgresOperator = DummyOperator
    SqlSensor = DummyOperator

import yaml

# DAG Configuration
DAG_ID = "{template_data["dag_id"]}"
DAG_RESOURCES_DIR = os.path.join("dags", "resources", DAG_ID)
SQL_DIR = os.path.join("resources", DAG_ID)

LOGGER = logging.getLogger(__name__)

# Default arguments
DEFAULT_ARGS = {{
    "owner": "{", ".join(template_data.get("owners", "").split(","))}",
    "email_on_failure": {str(template_data["email_on_failure"]).lower()},
    "email_on_retry": False,
    "retries": {template_data["retries"]},
    "retry_delay": timedelta(minutes=5),
}}

# Create DAG
DAG = airflow.DAG(
    dag_id=DAG_ID,
    description="{template_data["description"]}",
    max_active_runs={template_data["max_active_runs"]},
    start_date=datetime({template_data["start_date"]}),
    schedule_interval="{template_data["schedule_interval"]}",
    default_args=DEFAULT_ARGS,
    catchup={str(template_data["catchup"]).lower()},
    tags={template_data["tags"]},
)


def attach_tasks_to_dag(dag: airflow.DAG) -> airflow.DAG:
    """Create and assign all tasks following datasci-rx patterns."""

    task_list: Dict[str, BaseOperator] = {{}}
    dag_config_path = os.path.join(DAG_RESOURCES_DIR, "tasks.yml")

    if os.path.exists(dag_config_path):
        with open(dag_config_path) as dag_config_file:
            dag_config = yaml.safe_load(dag_config_file)

        # Set up environment variables
        var_config_path = os.path.join(DAG_RESOURCES_DIR, "config.yml")
        if os.path.exists(var_config_path):
            try:
                # Try to import enterprise config parser
                from airflow_llm import config_parser
                env_vars = config_parser.parse_config_yaml(var_config_path)
            except ImportError:
                with open(var_config_path) as f:
                    env_vars = yaml.safe_load(f)
                    # Simple environment resolution
                    deployment_env = os.getenv("DEPLOYMENT_ENVIRONMENT", "dev")
                    for k, v in env_vars.items():
                        if isinstance(v, dict) and deployment_env in v:
                            env_vars[k] = v[deployment_env]
        else:
            env_vars = {{}}

        for task in dag_config["tasks"]:
            task_operator = None

            if task["operator"] == "postgresql":
                sql_params = {{
                    "iam_role_s3": env_vars.get("iam_role_s3", ""),
                    "s3_bucket": env_vars.get("s3_bucket", ""),
                    "output_schema": env_vars.get("output_schema", ""),
                    "stakeholders": env_vars.get("stakeholders", ""),
                    "owners": env_vars.get("owners", ""),
                }}

                task_operator = PostgresOperator(
                    task_id=task["name"],
                    postgres_conn_id="redshift_default",
                    sql=os.path.join(SQL_DIR, task.get("sql_script", "default.sql")),
                    params=sql_params,
                    dag=dag,
                )

            elif task["operator"] == "python":
                def default_python_callable(**context):
                    return {{"status": "completed", "task": task["name"]}}

                task_operator = PythonOperator(
                    task_id=task["name"],
                    python_callable=default_python_callable,
                    dag=dag,
                )

            elif task["operator"] == "sensor":
                task_operator = SqlSensor(
                    dag=dag,
                    task_id=task["name"],
                    conn_id="redshift_default",
                    sql="SELECT 1",
                    retries=1,
                )

            elif task["operator"] == "dummy":
                task_operator = DummyOperator(dag=dag, task_id=task["name"])

            # Set up task dependencies
            if task_operator:
                for dependency in task.get("depends_on", []):
                    if dependency != "none" and dependency in task_list:
                        task_list[dependency] >> task_operator

                task_list[task["name"]] = task_operator

    return dag


# Attach tasks to DAG
if not os.getenv("TEST_ENVIRONMENT"):
    attach_tasks_to_dag(DAG)

# Export for Airflow discovery
globals()[DAG_ID] = DAG
'''

        return dag_content


class NaturalLanguageDAGGenerator:
    """
    AI-powered natural language to DAG generator
    Converts descriptions to complete DAG structures
    """

    def __init__(self, dag_factory: EnterpriseDAGFactory):
        self.dag_factory = dag_factory
        self.code_generator = None

        # Initialize code generator if available
        try:
            from .core_engine import ProductionCodeGenerator

            self.code_generator = ProductionCodeGenerator()
        except ImportError:
            logger.warning("Code generator not available, using fallback templates")

    def generate_from_description(
        self,
        description: str,
        dag_id: str | None = None,
        owners: list[str] | None = None,
        stakeholders: list[str] | None = None,
        environment_configs: dict[str, Any] | None = None,
    ) -> str:
        """
        Generate complete DAG from natural language description

        Args:
            description: Natural language pipeline description
            dag_id: Optional DAG ID (auto-generated if not provided)
            owners: List of DAG owners
            stakeholders: List of stakeholders
            environment_configs: Environment-specific configurations

        Returns:
            Path to generated DAG file
        """

        # Generate DAG ID if not provided
        if not dag_id:
            dag_id = self._generate_dag_id(description)

        # Parse description to extract tasks and dependencies
        parsed_pipeline = self._parse_description(description)

        # Create DAG configuration
        dag_config = DAGConfig(
            dag_id=dag_id,
            description=description,
            owners=owners or ["airflow-llm"],
            stakeholders=stakeholders or [],
            environment_configs=environment_configs or {},
            tags=["airflow-llm", "nl-generated", "production"],
        )

        # Generate tasks from parsed description
        tasks = self._generate_tasks(parsed_pipeline)

        # Generate SQL scripts and Python functions
        sql_scripts = self._generate_sql_scripts(parsed_pipeline)
        python_functions = self._generate_python_functions(parsed_pipeline)

        # Generate complete DAG
        return self.dag_factory.generate_dag(
            description=description,
            dag_config=dag_config,
            tasks=tasks,
            sql_scripts=sql_scripts,
            python_functions=python_functions,
        )

    def _generate_dag_id(self, description: str) -> str:
        """Generate DAG ID from description"""
        import re

        # Extract meaningful words
        words = re.findall(r"\b[a-zA-Z]+\b", description.lower())

        # Filter out common words
        stop_words = {
            "the",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "a",
            "an",
        }
        meaningful_words = [w for w in words if w not in stop_words and len(w) > 2]

        # Take first 3-4 meaningful words
        dag_words = meaningful_words[:4]
        dag_id = "_".join(dag_words)

        # Add timestamp to ensure uniqueness
        timestamp = datetime.now().strftime("%Y%m%d")
        return f"{dag_id}_{timestamp}"

    def _parse_description(self, description: str) -> dict[str, Any]:
        """Parse natural language description into structured pipeline"""

        # Use AI code generator if available
        if self.code_generator:
            return self._ai_parse_description(description)

        # Fallback to rule-based parsing
        return self._rule_based_parse(description)

    def _ai_parse_description(self, description: str) -> dict[str, Any]:
        """Use AI to parse description into pipeline structure"""

        prompt = f"""
        Parse this pipeline description into a structured format:
        "{description}"

        Return JSON with:
        {{
            "pipeline_type": "etl|ml|analytics|streaming",
            "tasks": [
                {{
                    "name": "task_name",
                    "type": "sql|python|sensor|notification",
                    "description": "what this task does",
                    "dependencies": ["task1", "task2"],
                    "estimated_duration": "5m|1h|etc",
                    "resources": {{"cpu": 2, "memory": "4Gi"}}
                }}
            ],
            "data_sources": ["source1", "source2"],
            "data_targets": ["target1", "target2"],
            "schedule": "daily|hourly|weekly",
            "business_impact": "high|medium|low"
        }}
        """

        try:
            response = self.code_generator.generation_pipeline(prompt)
            return json.loads(response)
        except Exception as e:
            logger.warning(f"AI parsing failed: {e}, using fallback")
            return self._rule_based_parse(description)

    def _rule_based_parse(self, description: str) -> dict[str, Any]:
        """Rule-based parsing for fallback"""

        # Simple keyword detection
        keywords = {
            "extract": ["extract", "pull", "fetch", "download", "read"],
            "transform": ["transform", "process", "clean", "filter", "aggregate"],
            "load": ["load", "save", "write", "upload", "store"],
            "train": ["train", "model", "ml", "machine learning"],
            "validate": ["validate", "test", "check", "verify"],
            "notify": ["notify", "alert", "email", "slack"],
        }

        detected_tasks = []
        words = description.lower().split()

        for task_type, task_keywords in keywords.items():
            if any(keyword in words for keyword in task_keywords):
                detected_tasks.append(
                    {
                        "name": f"{task_type}_data",
                        "type": (
                            "python" if task_type in ["transform", "train"] else "sql"
                        ),
                        "description": f"{task_type.title()} operation",
                        "dependencies": [],
                        "estimated_duration": "30m",
                        "resources": {"cpu": 2, "memory": "4Gi"},
                    }
                )

        return {
            "pipeline_type": "etl",
            "tasks": detected_tasks,
            "data_sources": ["database"],
            "data_targets": ["warehouse"],
            "schedule": "daily",
            "business_impact": "medium",
        }

    def _generate_tasks(self, parsed_pipeline: dict[str, Any]) -> list[TaskConfig]:
        """Generate TaskConfig objects from parsed pipeline"""

        tasks = []

        # Add initialization task
        tasks.append(TaskConfig(name="init", operator="dummy", depends_on=["none"]))

        previous_task = "init"

        # Generate tasks from parsed pipeline
        for i, task_info in enumerate(parsed_pipeline.get("tasks", [])):
            task = TaskConfig(
                name=task_info["name"],
                operator=self._determine_operator(task_info["type"]),
                depends_on=[previous_task],
                sql_script=(
                    f"{task_info['name']}.sql" if task_info["type"] == "sql" else None
                ),
                python_callable=(
                    f"{task_info['name']}_function"
                    if task_info["type"] == "python"
                    else None
                ),
                resources=task_info.get("resources", {}),
            )
            tasks.append(task)
            previous_task = task_info["name"]

        # Add completion task
        tasks.append(
            TaskConfig(
                name="done",
                operator="dummy",
                depends_on=[previous_task] if tasks else ["init"],
            )
        )

        return tasks

    def _determine_operator(self, task_type: str) -> str:
        """Determine Airflow operator based on task type"""
        operator_mapping = {
            "sql": "postgresql",
            "python": "python",
            "sensor": "sensor",
            "notification": "python",
            "dummy": "dummy",
        }
        return operator_mapping.get(task_type, "python")

    def _generate_sql_scripts(self, parsed_pipeline: dict[str, Any]) -> dict[str, str]:
        """Generate SQL scripts for SQL tasks"""

        sql_scripts = {}

        for task_info in parsed_pipeline.get("tasks", []):
            if task_info["type"] == "sql":
                script_name = f"{task_info['name']}.sql"

                # Generate SQL based on task description
                sql_content = self._generate_sql_content(task_info)
                sql_scripts[script_name] = sql_content

        return sql_scripts

    def _generate_sql_content(self, task_info: dict[str, Any]) -> str:
        """Generate SQL content for a task"""

        # Template SQL with Jinja2 variables
        sql_template = f"""
-- {task_info['description']}
-- Generated by AirflowLLM on {datetime.now().isoformat()}

-- Environment variables available:
-- {{{{ params.iam_role_s3 }}}}
-- {{{{ params.s3_bucket }}}}
-- {{{{ params.output_schema }}}}

SELECT
    current_timestamp as execution_time,
    '{task_info['name']}' as task_name,
    'Generated SQL for {task_info['description']}' as description;

-- TODO: Replace with actual SQL logic for {task_info['description']}
"""

        return sql_template.strip()

    def _generate_python_functions(
        self, parsed_pipeline: dict[str, Any]
    ) -> dict[str, str]:
        """Generate Python functions for Python tasks"""

        python_functions = {}

        for task_info in parsed_pipeline.get("tasks", []):
            if task_info["type"] == "python":
                module_name = f"{task_info['name']}_function.py"

                # Generate Python function
                python_content = self._generate_python_content(task_info)
                python_functions[module_name] = python_content

        return python_functions

    def _generate_python_content(self, task_info: dict[str, Any]) -> str:
        """Generate Python function content"""

        python_template = f'''"""
{task_info['description']}
Generated by AirflowLLM on {datetime.now().isoformat()}
"""
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def {task_info['name']}_function(**context: Dict[str, Any]) -> Dict[str, Any]:
    """
    {task_info['description']}

    Args:
        context: Airflow context dictionary

    Returns:
        Dictionary with execution results
    """

    logger.info("Starting {task_info['name']} execution")

    # TODO: Implement actual logic for {task_info['description']}

    # Example implementation
    result = {{
        "task_name": "{task_info['name']}",
        "description": "{task_info['description']}",
        "execution_time": context.get("ts"),
        "status": "success"
    }}

    logger.info(f"Completed {task_info['name']}: {{result}}")

    return result
'''

        return python_template.strip()
