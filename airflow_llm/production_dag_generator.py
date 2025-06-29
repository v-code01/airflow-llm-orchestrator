"""
Production DAG Generator - Creates complete working DAGs with real implementations
"""

import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .core_engine import (
    CodeGenerationRequest,
    GeneratedImplementation,
    ProductionCodeGenerator,
    TaskType,
)

logger = logging.getLogger(__name__)


@dataclass
class DAGTask:
    task_id: str
    task_type: TaskType
    description: str
    implementation: GeneratedImplementation
    dependencies: list[str]
    connection_id: str | None = None
    table_schema: dict | None = None
    business_rules: list[str] | None = None


@dataclass
class ProductionDAG:
    dag_id: str
    description: str
    tasks: list[DAGTask]
    schedule_interval: str
    start_date: str
    default_args: dict[str, Any]
    full_dag_code: str
    estimated_cost: dict[str, float]
    performance_profile: dict[str, Any]


class ProductionDAGGenerator:
    """
    Generates complete production DAGs with working implementations
    """

    def __init__(self):
        self.code_generator = ProductionCodeGenerator()

        # Advanced task pattern recognition
        self.task_patterns = {
            # SQL Extraction patterns
            r"extract.*(?:from|postgresql|mysql|redshift|snowflake)": TaskType.SQL_EXTRACTION,
            r"(?:read|fetch|get).*(?:database|table|sql)": TaskType.SQL_EXTRACTION,
            r"query.*(?:database|table)": TaskType.SQL_EXTRACTION,
            # SQL Transformation patterns
            r"(?:transform|aggregate|join|calculate).*(?:sql|database)": TaskType.SQL_TRANSFORMATION,
            r"(?:group by|sum|count|avg)": TaskType.SQL_TRANSFORMATION,
            r"create.*(?:table|view|materialized)": TaskType.SQL_TRANSFORMATION,
            # Python Processing patterns
            r"(?:process|clean|filter|merge).*(?:data|dataframe|pandas)": TaskType.PYTHON_PROCESSING,
            r"(?:apply|transform).*(?:function|logic|rules)": TaskType.PYTHON_PROCESSING,
            r"feature.*(?:engineering|extraction)": TaskType.PYTHON_PROCESSING,
            # Data Validation patterns
            r"(?:validate|check|verify).*(?:data|quality|schema)": TaskType.DATA_VALIDATION,
            r"(?:test|assert).*(?:data|values)": TaskType.DATA_VALIDATION,
            r"quality.*(?:check|assurance)": TaskType.DATA_VALIDATION,
            # API Integration patterns
            r"(?:call|fetch|post).*(?:api|rest|endpoint)": TaskType.API_INTEGRATION,
            r"(?:http|webhook|service).*(?:request|call)": TaskType.API_INTEGRATION,
            r"integrate.*(?:external|third.party)": TaskType.API_INTEGRATION,
            # Notification patterns
            r"(?:send|notify|alert).*(?:email|slack|teams)": TaskType.NOTIFICATION,
            r"(?:report|dashboard|summary)": TaskType.NOTIFICATION,
            r"notification.*(?:success|failure|completion)": TaskType.NOTIFICATION,
            # ML Inference patterns
            r"(?:predict|score|inference).*(?:model|ml)": TaskType.ML_INFERENCE,
            r"(?:machine learning|neural network|algorithm)": TaskType.ML_INFERENCE,
            r"apply.*(?:model|prediction)": TaskType.ML_INFERENCE,
            # File Operations patterns
            r"(?:upload|download|copy|move).*(?:file|s3|gcs|azure)": TaskType.FILE_OPERATIONS,
            r"(?:csv|parquet|json|avro).*(?:read|write|convert)": TaskType.FILE_OPERATIONS,
            r"file.*(?:processing|handling|operations)": TaskType.FILE_OPERATIONS,
        }

        # Database connection mapping
        self.connection_mapping = {
            "postgresql": "postgres_default",
            "postgres": "postgres_default",
            "mysql": "mysql_default",
            "redshift": "redshift_default",
            "snowflake": "snowflake_default",
            "bigquery": "bigquery_default",
            "mongodb": "mongodb_default",
            "s3": "aws_default",
            "gcs": "google_cloud_default",
            "azure": "azure_default",
        }

    def generate_production_dag(
        self,
        description: str,
        dag_id: str | None = None,
        schedule_interval: str = "@daily",
        additional_context: dict | None = None,
    ) -> ProductionDAG:
        """
        Generate a complete production DAG with working implementations
        """
        logger.info(f"Generating production DAG for: {description}")

        if not dag_id:
            dag_id = f"generated_dag_{int(time.time())}"

        # Step 1: Parse description and extract tasks
        parsed_tasks = self._parse_pipeline_description(
            description, additional_context or {}
        )

        # Step 2: Generate implementations for each task
        dag_tasks = []
        for task_info in parsed_tasks:
            implementation = self._generate_task_implementation(task_info)
            dag_tasks.append(
                DAGTask(
                    task_id=task_info["task_id"],
                    task_type=task_info["task_type"],
                    description=task_info["description"],
                    implementation=implementation,
                    dependencies=task_info["dependencies"],
                    connection_id=task_info.get("connection_id"),
                    table_schema=task_info.get("table_schema"),
                    business_rules=task_info.get("business_rules"),
                )
            )

        # Step 3: Generate complete DAG code
        dag_code = self._generate_complete_dag_code(
            dag_id, dag_tasks, schedule_interval, description
        )

        # Step 4: Calculate performance and cost estimates
        performance_profile = self._calculate_performance_profile(dag_tasks)
        estimated_cost = self._estimate_execution_cost(dag_tasks)

        return ProductionDAG(
            dag_id=dag_id,
            description=description,
            tasks=dag_tasks,
            schedule_interval=schedule_interval,
            start_date=datetime.now().strftime("%Y-%m-%d"),
            default_args=self._get_production_default_args(),
            full_dag_code=dag_code,
            estimated_cost=estimated_cost,
            performance_profile=performance_profile,
        )

    def _parse_pipeline_description(
        self, description: str, context: dict
    ) -> list[dict]:
        """
        Parse natural language description into structured tasks
        """
        # Split description into logical steps
        sentences = re.split(r"[.;]\s+", description)
        tasks = []

        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue

            # Classify task type
            task_type = self._classify_task_type(sentence)

            # Extract entities (tables, databases, etc.)
            entities = self._extract_entities(sentence)

            # Generate task ID
            task_id = self._generate_task_id(sentence, i)

            # Determine dependencies
            dependencies = self._infer_dependencies(tasks, sentence, entities)

            task_info = {
                "task_id": task_id,
                "task_type": task_type,
                "description": sentence.strip(),
                "entities": entities,
                "dependencies": dependencies,
                "connection_id": self._determine_connection(entities),
                "table_schema": context.get("table_schemas", {}).get(
                    entities.get("table")
                ),
                "business_rules": context.get("business_rules", []),
            }

            tasks.append(task_info)

        return tasks

    def _classify_task_type(self, sentence: str) -> TaskType:
        """
        Classify sentence into specific task type using advanced pattern matching
        """
        sentence_lower = sentence.lower()

        # Score each task type
        scores = {}
        for pattern, task_type in self.task_patterns.items():
            if re.search(pattern, sentence_lower):
                scores[task_type] = scores.get(task_type, 0) + 1

        if scores:
            return max(scores, key=scores.get)

        # Fallback classification
        if any(
            word in sentence_lower for word in ["extract", "read", "fetch", "select"]
        ):
            return TaskType.SQL_EXTRACTION
        elif any(word in sentence_lower for word in ["process", "transform", "clean"]):
            return TaskType.PYTHON_PROCESSING
        elif any(word in sentence_lower for word in ["validate", "check", "verify"]):
            return TaskType.DATA_VALIDATION
        else:
            return TaskType.PYTHON_PROCESSING  # Default

    def _extract_entities(self, sentence: str) -> dict[str, str]:
        """
        Extract key entities like table names, databases, connections
        """
        entities = {}

        # Extract table names
        table_patterns = [
            r"(?:table|from)\s+([a-zA-Z_][a-zA-Z0-9_]*)",
            r"([a-zA-Z_][a-zA-Z0-9_]*)\s+table",
            r"(?:into|to)\s+([a-zA-Z_][a-zA-Z0-9_]*)",
        ]

        for pattern in table_patterns:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                entities["table"] = match.group(1)
                break

        # Extract database/connection types
        db_types = [
            "postgresql",
            "mysql",
            "redshift",
            "snowflake",
            "bigquery",
            "s3",
            "mongodb",
        ]
        for db_type in db_types:
            if db_type in sentence.lower():
                entities["database_type"] = db_type
                break

        # Extract file formats
        file_formats = ["csv", "parquet", "json", "avro", "orc"]
        for fmt in file_formats:
            if fmt in sentence.lower():
                entities["file_format"] = fmt
                break

        return entities

    def _generate_task_id(self, sentence: str, index: int) -> str:
        """
        Generate meaningful task IDs from sentence content
        """
        # Extract key verbs and nouns
        words = re.findall(r"\b[a-zA-Z]+\b", sentence.lower())

        # Priority words for task naming
        action_words = [
            "extract",
            "transform",
            "load",
            "validate",
            "process",
            "send",
            "notify",
        ]
        object_words = ["data", "table", "file", "report", "email", "api"]

        action = None
        obj = None

        for word in words:
            if word in action_words and not action:
                action = word
            elif word in object_words and not obj:
                obj = word

        if action and obj:
            return f"{action}_{obj}_{index + 1}"
        elif action:
            return f"{action}_data_{index + 1}"
        else:
            return f"task_{index + 1}"

    def _infer_dependencies(
        self, existing_tasks: list[dict], sentence: str, entities: dict
    ) -> list[str]:
        """
        Infer task dependencies based on data flow and logical sequence
        """
        dependencies = []

        # If this is not the first task, check for data dependencies
        if existing_tasks:
            sentence_lower = sentence.lower()

            # Check if this task consumes output from previous tasks
            for prev_task in existing_tasks:
                prev_entities = prev_task["entities"]

                # If previous task extracted from a table and this task processes data
                if prev_entities.get("table") and any(
                    word in sentence_lower
                    for word in ["process", "transform", "validate", "clean"]
                ):
                    dependencies.append(prev_task["task_id"])

                # If this is a notification task, it depends on all previous processing
                if any(
                    word in sentence_lower
                    for word in ["notify", "send", "email", "alert"]
                ):
                    dependencies.append(prev_task["task_id"])

        return dependencies

    def _determine_connection(self, entities: dict) -> str | None:
        """
        Determine Airflow connection ID based on entities
        """
        db_type = entities.get("database_type")
        if db_type and db_type in self.connection_mapping:
            return self.connection_mapping[db_type]
        return None

    def _generate_task_implementation(self, task_info: dict) -> GeneratedImplementation:
        """
        Generate complete implementation for a specific task
        """
        request = CodeGenerationRequest(
            task_type=task_info["task_type"],
            description=task_info["description"],
            table_schema=task_info.get("table_schema"),
            source_system=task_info["entities"].get("database_type"),
            target_system=task_info["entities"].get("target_system"),
            business_rules=task_info.get("business_rules"),
        )

        return self.code_generator.generate_implementation(request)

    def _generate_complete_dag_code(
        self,
        dag_id: str,
        tasks: list[DAGTask],
        schedule_interval: str,
        description: str,
    ) -> str:
        """
        Generate complete DAG code with all implementations
        """
        # Collect all imports
        all_imports = set()
        all_imports.update(
            [
                "from airflow import DAG",
                "from airflow.operators.python import PythonOperator",
                "from datetime import datetime, timedelta",
                "import logging",
                "import time",
                "import json",
            ]
        )

        for task in tasks:
            all_imports.update(task.implementation.imports)

        # Generate default args
        default_args = self._get_production_default_args()

        # Start building DAG code
        dag_code = f'''"""
Production DAG: {description}
Generated by AirflowLLM Production Engine
"""

{chr(10).join(sorted(all_imports))}

# Production-grade default arguments
default_args = {json.dumps(default_args, indent=4, default=str)}

# Create DAG with production settings
dag = DAG(
    dag_id='{dag_id}',
    default_args=default_args,
    description='{description}',
    schedule_interval='{schedule_interval}',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    max_active_tasks=10,
    tags=['production', 'airflow-llm', 'auto-generated'],
    doc_md="""
    # Production DAG: {dag_id}

    ## Description
    {description}

    ## Tasks
{chr(10).join(f"    - {task.task_id}: {task.description}" for task in tasks)}

    ## Performance Profile
    - Estimated runtime: Variable based on data volume
    - Resource requirements: Optimized for production workloads
    - Error handling: Comprehensive with automatic retries
    - Monitoring: Full observability with metrics and logging
    """
)

# Task implementations
'''

        # Add task function implementations
        for task in tasks:
            dag_code += f"\n# Task: {task.description}\n"
            dag_code += task.implementation.function_code + "\n\n"

        # Add task definitions
        dag_code += "\n# Task definitions with monitoring\n"

        for task in tasks:
            dag_code += f'''
{task.task_id} = PythonOperator(
    task_id='{task.task_id}',
    python_callable={task.implementation.function_code.split('def ')[1].split('(')[0]},
    dag=dag,
    retries=3,
    retry_delay=timedelta(minutes=5),
    execution_timeout=timedelta(hours=2),
    pool='default_pool',
    priority_weight=10,
    doc_md="""
    ## {task.task_id}

    **Description:** {task.description}

    **Type:** {task.task_type.value}

    **Implementation:** Production-grade with error handling and monitoring

    **Resources:** {task.implementation.resource_requirements}
    """
)

'''

        # Add task dependencies
        if tasks:
            dag_code += "\n# Task dependencies\n"
            for task in tasks:
                if task.dependencies:
                    deps = " >> ".join(task.dependencies)
                    dag_code += f"{deps} >> {task.task_id}\n"

        # Add monitoring and alerting
        dag_code += f'''
# Production monitoring setup
def dag_success_callback(context):
    """Called when DAG succeeds"""
    logging.info(f"DAG {{context['dag'].dag_id}} completed successfully")
    # Add your success notifications here

def dag_failure_callback(context):
    """Called when DAG fails"""
    logging.error(f"DAG {{context['dag'].dag_id}} failed")
    # Add your failure notifications here

dag.on_success_callback = dag_success_callback
dag.on_failure_callback = dag_failure_callback

# Set up task group for better organization if needed
if len([{", ".join(f"'{task.task_id}'" for task in tasks)}]) > 5:
    from airflow.utils.task_group import TaskGroup

    with TaskGroup("data_processing", dag=dag) as processing_group:
        # Organize tasks into logical groups
        pass
'''

        return dag_code

    def _get_production_default_args(self) -> dict[str, Any]:
        """
        Get production-grade default arguments
        """
        return {
            "owner": "airflow-llm-production",
            "depends_on_past": False,
            "start_date": "2024-01-01",
            "email_on_failure": True,
            "email_on_retry": False,
            "retries": 3,
            "retry_delay": 300,  # 5 minutes in seconds
            "execution_timeout": 7200,  # 2 hours in seconds
            "email": ["data-team@company.com"],
            "retry_exponential_backoff": True,
            "max_retry_delay": 3600,  # 1 hour max
            "sla": 14400,  # 4 hours SLA
        }

    def _calculate_performance_profile(self, tasks: list[DAGTask]) -> dict[str, Any]:
        """
        Calculate estimated performance profile for the DAG
        """
        total_cpu = sum(
            task.implementation.resource_requirements.get("cpu", 1) for task in tasks
        )
        total_memory = sum(
            int(
                task.implementation.resource_requirements.get("memory", "2Gi").replace(
                    "Gi", ""
                )
            )
            for task in tasks
        )

        return {
            "estimated_runtime_minutes": len(tasks) * 5,  # Conservative estimate
            "total_cpu_cores": total_cpu,
            "total_memory_gb": total_memory,
            "parallelizable_tasks": len([t for t in tasks if not t.dependencies]),
            "critical_path_length": self._calculate_critical_path(tasks),
            "complexity_score": len(tasks) * 10 + total_cpu * 5,
        }

    def _estimate_execution_cost(self, tasks: list[DAGTask]) -> dict[str, float]:
        """
        Estimate execution cost based on resource requirements
        """
        # Conservative cost estimates (AWS pricing)
        cpu_cost_per_hour = 0.10  # per vCPU
        memory_cost_per_hour = 0.02  # per GB

        total_cost = 0
        for task in tasks:
            cpu = task.implementation.resource_requirements.get("cpu", 1)
            memory_gb = int(
                task.implementation.resource_requirements.get("memory", "2Gi").replace(
                    "Gi", ""
                )
            )
            runtime_hours = 0.5  # Conservative estimate

            task_cost = (
                cpu * cpu_cost_per_hour + memory_gb * memory_cost_per_hour
            ) * runtime_hours
            total_cost += task_cost

        return {
            "estimated_cost_per_run": round(total_cost, 2),
            "estimated_monthly_cost": round(total_cost * 30, 2),  # Assuming daily runs
            "cost_breakdown": {
                "compute": round(total_cost * 0.7, 2),
                "storage": round(total_cost * 0.2, 2),
                "network": round(total_cost * 0.1, 2),
            },
        }

    def _calculate_critical_path(self, tasks: list[DAGTask]) -> int:
        """
        Calculate the critical path length (longest dependency chain)
        """
        # Simple implementation - could be enhanced with proper graph algorithms
        max_depth = 0

        def get_depth(task_id: str, visited: set) -> int:
            if task_id in visited:
                return 0

            visited.add(task_id)
            task = next((t for t in tasks if t.task_id == task_id), None)

            if not task or not task.dependencies:
                return 1

            return 1 + max(get_depth(dep, visited.copy()) for dep in task.dependencies)

        for task in tasks:
            depth = get_depth(task.task_id, set())
            max_depth = max(max_depth, depth)

        return max_depth
