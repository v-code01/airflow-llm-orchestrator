"""
Natural Language to DAG Generator
"""

import json
from typing import Any

try:
    from airflow import DAG
    from airflow.operators.bash import BashOperator
    from airflow.operators.python import PythonOperator
    from airflow.providers.http.operators.http import SimpleHttpOperator

    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False

    # Mock classes for development
    class DAG:
        def __init__(self, dag_id, **kwargs):
            self.dag_id = dag_id
            self.tasks = []

    class BashOperator:
        def __init__(self, task_id, **kwargs):
            self.task_id = task_id

    class PythonOperator:
        def __init__(self, task_id, **kwargs):
            self.task_id = task_id

    class SimpleHttpOperator:
        def __init__(self, task_id, **kwargs):
            self.task_id = task_id


class NaturalLanguageDAGGenerator:
    """
    Convert natural language descriptions into executable Airflow DAGs
    """

    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.task_patterns = {
            "etl": ["extract", "transform", "load", "process", "clean"],
            "ml": ["train", "predict", "evaluate", "deploy", "model"],
            "data": ["read", "write", "query", "aggregate", "join"],
            "notification": ["email", "alert", "notify", "slack", "webhook"],
        }

    def generate(self, description: str, dag_id: str) -> str:
        """
        Generate complete DAG code from natural language
        """
        # Parse the description into tasks
        tasks = self._extract_tasks(description)

        # Determine dependencies
        dependencies = self._infer_dependencies(tasks)

        # Generate DAG code
        dag_code = self._generate_dag_code(dag_id, tasks, dependencies)

        return dag_code

    def _extract_tasks(self, description: str) -> list[dict[str, Any]]:
        """
        Extract tasks from natural language description
        """
        prompt = f"""
        Extract tasks from this pipeline description:
        "{description}"

        For each task, identify:
        1. Task name
        2. Task type (python, bash, http, etc.)
        3. Required resources (CPU, memory, GPU)
        4. Input/output data
        5. Any specific requirements

        Return as JSON array.
        """

        # Use the model router's generate method
        from .models.specialized_ensemble import TaskType

        response = self.llm_client.generate(prompt, TaskType.DAG_ORCHESTRATION)

        try:
            tasks = json.loads(response.content)
        except (json.JSONDecodeError, AttributeError):
            # Fallback: create simple tasks from description keywords
            tasks = self._create_fallback_tasks(description)

        # Enrich with task metadata
        for task in tasks:
            task["operator"] = self._determine_operator(task)
            task["resources"] = self._estimate_resources(task)

        return tasks

    def _infer_dependencies(self, tasks: list[dict]) -> dict[str, list[str]]:
        """
        Automatically infer task dependencies based on data flow
        """
        dependencies = {}

        for i, task in enumerate(tasks):
            task_deps = []

            # Check if this task's inputs match other tasks' outputs
            for j, other_task in enumerate(tasks[:i]):
                if self._tasks_connected(other_task, task):
                    task_deps.append(other_task["name"])

            dependencies[task["name"]] = task_deps

        return dependencies

    def _generate_dag_code(
        self, dag_id: str, tasks: list[dict], dependencies: dict[str, list[str]]
    ) -> str:
        """
        Generate executable DAG code
        """
        code = f"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.http.operators.http import SimpleHttpOperator
from datetime import datetime, timedelta
import logging

# Default arguments for the DAG
default_args = {{
    'owner': 'airflow-llm-production',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}}

# Create the DAG
dag = DAG(
    '{dag_id}',
    default_args=default_args,
    description='Production pipeline - NLP generated',
    schedule_interval=timedelta(days=1),
    catchup=False,
    tags=['production', 'nlp-generated'],
)

# Define task functions
"""

        # Generate task definitions
        task_vars = {}
        for task in tasks:
            task_code, var_name = self._generate_task_code(task)
            code += task_code + "\n"
            task_vars[task["name"]] = var_name

        # Add dependencies
        code += "\n# Set task dependencies\n"
        for task_name, deps in dependencies.items():
            if deps:
                dep_vars = [task_vars[dep] for dep in deps]
                code += f"{' >> '.join(dep_vars)} >> {task_vars[task_name]}\n"

        return code

    def _generate_task_code(self, task: dict) -> tuple[str, str]:
        """
        Generate code for a specific task
        """
        var_name = task["name"].lower().replace(" ", "_")

        if task["operator"] == "PythonOperator":
            code = f'''
def {var_name}_func(**context):
    """Auto-generated function for {task['name']}"""
    logging.info("Executing {task['name']}")
    # TODO: Implement task logic
    return {{'status': 'success'}}

{var_name} = PythonOperator(
    task_id='{var_name}',
    python_callable={var_name}_func,
    dag=dag,
)
'''
        elif task["operator"] == "BashOperator":
            code = f"""
{var_name} = BashOperator(
    task_id='{var_name}',
    bash_command='{task.get("command", "echo 'Task executed'")}',
    dag=dag,
)
"""
        else:
            # Default to Python operator
            code = self._generate_python_operator(task, var_name)

        return code, var_name

    def _determine_operator(self, task: dict) -> str:
        """
        Determine the appropriate Airflow operator for a task
        """
        task_type = task.get("type", "").lower()

        if "bash" in task_type or "shell" in task_type:
            return "BashOperator"
        elif "http" in task_type or "api" in task_type:
            return "SimpleHttpOperator"
        elif "sql" in task_type:
            return "SqlOperator"
        else:
            return "PythonOperator"

    def _estimate_resources(self, task: dict) -> dict[str, Any]:
        """
        Estimate resource requirements for a task
        """
        # Use ML model to predict resource needs based on task type
        task_type = task.get("type", "").lower()

        if any(ml_keyword in task_type for ml_keyword in ["train", "model", "gpu"]):
            return {"cpu": 4, "memory": "16Gi", "gpu": 1, "gpu_type": "nvidia-tesla-t4"}
        elif "etl" in task_type or "process" in task_type:
            return {"cpu": 2, "memory": "8Gi"}
        else:
            return {"cpu": 1, "memory": "2Gi"}

    def _tasks_connected(self, task1: dict, task2: dict) -> bool:
        """
        Determine if two tasks are connected based on data flow
        """
        # Check if task1's outputs match task2's inputs
        task1_outputs = set(task1.get("outputs", []))
        task2_inputs = set(task2.get("inputs", []))

        return bool(task1_outputs.intersection(task2_inputs))

    def _create_fallback_tasks(self, description: str) -> list[dict[str, Any]]:
        """
        Create simple task structure when JSON parsing fails
        """
        # Extract key action words
        words = description.lower().split()

        tasks = []
        task_id = 1

        # Look for common data engineering patterns
        if any(word in words for word in ["extract", "read", "fetch", "get"]):
            tasks.append(
                {
                    "name": f"extract_data_{task_id}",
                    "type": "data_extraction",
                    "description": "Extract data from source",
                }
            )
            task_id += 1

        if any(word in words for word in ["transform", "process", "clean", "validate"]):
            tasks.append(
                {
                    "name": f"process_data_{task_id}",
                    "type": "data_processing",
                    "description": "Process and transform data",
                }
            )
            task_id += 1

        if any(word in words for word in ["load", "save", "store", "insert"]):
            tasks.append(
                {
                    "name": f"load_data_{task_id}",
                    "type": "data_loading",
                    "description": "Load data to destination",
                }
            )
            task_id += 1

        if any(word in words for word in ["email", "notify", "alert", "send"]):
            tasks.append(
                {
                    "name": f"notify_{task_id}",
                    "type": "notification",
                    "description": "Send notification",
                }
            )
            task_id += 1

        # If no patterns found, create a generic task
        if not tasks:
            tasks.append(
                {
                    "name": "execute_pipeline",
                    "type": "generic",
                    "description": description,
                }
            )

        return tasks
