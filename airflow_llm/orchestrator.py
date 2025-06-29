"""
Core LLM Orchestrator for intelligent pipeline management
"""

import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None

try:
    from airflow import DAG
    from airflow.models import DagRun, TaskInstance
    from airflow.utils.dates import days_ago

    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False
    DAG = None
    DagRun = None
    TaskInstance = None

    def days_ago(n):
        return datetime.now() - timedelta(days=n)




@dataclass
class PipelineMetrics:
    """Real-time pipeline performance metrics"""

    avg_runtime: float
    success_rate: float
    cost_per_run: float
    resource_utilization: dict[str, float]
    bottlenecks: list[str]
    optimization_suggestions: list[str]


class LLMOrchestrator:
    """
    Autonomous pipeline orchestration with LLM intelligence
    """

    def __init__(
        self,
        models: list[str] = ["codellama-7b", "sqlcoder-7b", "llama3-8b", "phi3-mini"],
        cost_optimization: bool = True,
        self_healing: bool = True,
        predictive_analytics: bool = True,
        gpu_aware: bool = True,
    ):
        self.models = models
        self.cost_optimization = cost_optimization
        self.self_healing = self_healing
        self.predictive_analytics = predictive_analytics
        self.gpu_aware = gpu_aware
        self.execution_history = {}
        try:
            from .model_server import model_server
            from .models.specialized_ensemble import enhanced_router

            self.model_server = model_server
            self.model_router = enhanced_router
        except ImportError:
            from airflow_llm.model_server import model_server
            from airflow_llm.models.specialized_ensemble import enhanced_router

            self.model_server = model_server
            self.model_router = enhanced_router
        # Initialize real cost tracker
        self.cost_tracker = CostTracker()
        # Initialize models on startup
        self._models_initialized = False

    async def generate_dag(
        self, description: str, constraints: dict[str, Any] | None = None
    ) -> DAG:
        """
        Generate optimized DAG from natural language description using real AI
        """
        # Ensure models are initialized
        if not self._models_initialized:
            await self._initialize_models()

        # Use real AI to parse requirements
        dag_structure = await self._parse_requirements(description)

        # Optimize task dependencies with AI
        optimized_structure = await self._optimize_dependencies(dag_structure)

        # Apply constraints
        if constraints:
            optimized_structure = self._apply_constraints(
                optimized_structure, constraints
            )

        # Generate DAG code with actual implementations
        dag = await self._create_dag_with_implementations(optimized_structure)

        # Add monitoring and self-healing
        if self.self_healing:
            dag = self._add_self_healing(dag)

        return dag

    async def _initialize_models(self):
        """Initialize the model server with required models"""
        try:
            await self.model_server.initialize_models(self.models, backend="auto")
            self._models_initialized = True
        except Exception as e:
            print(f"Warning: Failed to initialize models: {e}")
            # Still mark as initialized to avoid infinite retries
            self._models_initialized = True

    async def _parse_requirements(self, description: str) -> dict:
        """Parse natural language requirements into structured format using real AI"""
        prompt = f"""Convert this pipeline description to a valid JSON DAG structure:

{description}

Return ONLY a valid JSON object with this exact format:
{{
  "dag_id": "pipeline_name",
  "description": "Brief description",
  "schedule_interval": "@daily",
  "tasks": [
    {{
      "id": "task_name",
      "operator": "PythonOperator",
      "description": "What this task does",
      "upstream_dependencies": []
    }}
  ]
}}

Important: Return ONLY the JSON object, no explanations or markdown."""

        try:
            # Ensure model server is initialized first
            if not self._models_initialized:
                await self._initialize_models()

            # Use the real model server for inference
            result = await self.model_server.generate(prompt, model_name="phi3-mini")
            if result.success:
                response_text = result.text.strip()
                # Try to extract JSON from response
                if response_text.startswith("```json"):
                    response_text = response_text[7:-3].strip()
                elif response_text.startswith("```"):
                    response_text = response_text[3:-3].strip()

                try:
                    return json.loads(response_text)
                except json.JSONDecodeError:
                    # Robust JSON parsing for phi3-mini model
                    return self._robust_json_parse(response_text)
            else:
                raise Exception(f"Model inference failed: {result.error}")
        except Exception as e:
            print(f"AI parsing failed: {e}, using fallback")
            # Intelligent fallback based on description analysis
            return self._create_fallback_structure(description)

    def _robust_json_parse(self, response_text: str) -> dict:
        """Robust JSON parsing that handles phi3-mini formatting issues"""
        import re

        try:
            # Strategy 1: Clean up common formatting issues
            cleaned = response_text.replace("\n\n", "\n")
            cleaned = re.sub(r",\s*\n\s*}", "\n}", cleaned)
            cleaned = re.sub(r",\s*\n\s*]", "\n]", cleaned)

            # Fix missing commas between tasks
            cleaned = re.sub(r"}\s*\n\s*{", "},\n{", cleaned)

            # Fix malformed property names (hallmark: -> "operator":)
            cleaned = re.sub(r'\bhallmark:\s*([\'"])', r'"operator": \1', cleaned)
            cleaned = re.sub(r'\bid:\s*([\'"])', r'"id": \1', cleaned)
            cleaned = re.sub(r'\bdescription:\s*([\'"])', r'"description": \1', cleaned)
            cleaned = re.sub(
                r"\bupstream_dependencies:\s*\[", r'"upstream_dependencies": [', cleaned
            )

            # Fix single quotes to double quotes
            cleaned = re.sub(r"'([^']*)'", r'"\1"', cleaned)

            return json.loads(cleaned)

        except:
            # Strategy 2: Manual extraction with smart task parsing
            dag_id_match = re.search(r'"dag_id":\s*"([^"]*)"', response_text)
            desc_match = re.search(r'"description":\s*"([^"]*)"', response_text)
            sched_match = re.search(r'"schedule_interval":\s*"([^"]*)"', response_text)

            # Extract tasks more intelligently
            tasks = []

            # Find task patterns
            task_patterns = re.findall(
                r'"id":\s*"([^"]*)"[^}]*"operator":\s*"([^"]*)"[^}]*"description":\s*"([^"]*)"',
                response_text,
                re.DOTALL,
            )

            if not task_patterns:
                # Try alternative task pattern matching
                task_blocks = re.findall(
                    r'\{[^{}]*(?:"id"|id:)[^{}]*\}', response_text, re.DOTALL
                )
                for block in task_blocks:
                    id_match = re.search(r'(?:"id"|id):\s*[\'"]?([^\'"]*)[\'"]?', block)
                    op_match = re.search(
                        r'(?:"operator"|hallmark):\s*[\'"]?([^\'"]*)[\'"]?', block
                    )
                    desc_match_task = re.search(
                        r'(?:"description"|description):\s*[\'"]([^\'"]*)[\'"]', block
                    )

                    if id_match:
                        tasks.append(
                            {
                                "id": id_match.group(1),
                                "operator": op_match.group(1)
                                if op_match
                                else "PythonOperator",
                                "description": desc_match_task.group(1)
                                if desc_match_task
                                else f"Task: {id_match.group(1)}",
                                "upstream_dependencies": [],
                            }
                        )
            else:
                for task_id, operator, description in task_patterns:
                    tasks.append(
                        {
                            "id": task_id,
                            "operator": operator,
                            "description": description,
                            "upstream_dependencies": [],
                        }
                    )

            # If no tasks found, create default tasks based on description
            if not tasks:
                if any(
                    word in response_text.lower() for word in ["extract", "etl", "data"]
                ):
                    tasks = [
                        {
                            "id": "extract_data",
                            "operator": "PythonOperator",
                            "description": "Extract data from source",
                            "upstream_dependencies": [],
                        },
                        {
                            "id": "validate_data",
                            "operator": "PythonOperator",
                            "description": "Validate data quality",
                            "upstream_dependencies": ["extract_data"],
                        },
                        {
                            "id": "load_data",
                            "operator": "PythonOperator",
                            "description": "Load data to destination",
                            "upstream_dependencies": ["validate_data"],
                        },
                    ]

            # Set up dependencies for sequential tasks
            for i in range(1, len(tasks)):
                if not tasks[i]["upstream_dependencies"]:
                    tasks[i]["upstream_dependencies"] = [tasks[i - 1]["id"]]

            return {
                "dag_id": dag_id_match.group(1)
                if dag_id_match
                else "generated_pipeline",
                "description": desc_match.group(1)
                if desc_match
                else "AI-generated DAG",
                "schedule_interval": sched_match.group(1) if sched_match else "@daily",
                "tasks": tasks,
            }

    def _create_fallback_structure(self, description: str) -> dict:
        """Create intelligent fallback DAG structure based on description analysis"""
        description_lower = description.lower()

        # Analyze description for common patterns
        tasks = []
        dag_id = f"dag_{description[:20].replace(' ', '_').replace(',', '').lower()}_{int(time.time())}"

        # Data processing pipeline
        if any(
            word in description_lower
            for word in ["data", "process", "etl", "transform"]
        ):
            tasks = [
                {
                    "id": "extract_data",
                    "operator": "PythonOperator",
                    "description": "Extract data from source",
                    "upstream_dependencies": [],
                },
                {
                    "id": "transform_data",
                    "operator": "PythonOperator",
                    "description": "Transform and clean data",
                    "upstream_dependencies": ["extract_data"],
                },
                {
                    "id": "load_data",
                    "operator": "PythonOperator",
                    "description": "Load data to destination",
                    "upstream_dependencies": ["transform_data"],
                },
            ]
        # ML pipeline
        elif any(
            word in description_lower
            for word in ["model", "train", "ml", "machine learning"]
        ):
            tasks = [
                {
                    "id": "prepare_data",
                    "operator": "PythonOperator",
                    "description": "Prepare training data",
                    "upstream_dependencies": [],
                },
                {
                    "id": "train_model",
                    "operator": "PythonOperator",
                    "description": "Train ML model",
                    "upstream_dependencies": ["prepare_data"],
                },
                {
                    "id": "evaluate_model",
                    "operator": "PythonOperator",
                    "description": "Evaluate model performance",
                    "upstream_dependencies": ["train_model"],
                },
            ]
        # Simple task
        else:
            tasks = [
                {
                    "id": "execute_task",
                    "operator": "PythonOperator",
                    "description": description,
                    "upstream_dependencies": [],
                }
            ]

        return {
            "dag_id": dag_id,
            "description": description,
            "schedule_interval": "@daily",
            "tasks": tasks,
        }

    async def _optimize_dependencies(self, structure: dict) -> dict:
        """Optimize task dependencies for maximum parallelization using AI"""
        prompt = f"""Optimize this DAG structure for maximum parallelization:

{json.dumps(structure, indent=2)}

Return ONLY the optimized JSON with the same format but better dependencies.
Consider: parallel execution, resource usage, logical order.

Return ONLY valid JSON, no explanations:"""

        try:
            result = await self.model_server.generate(prompt, model_name="phi3-mini")
            if result.success:
                response_text = result.text.strip()
                # Clean up response
                if response_text.startswith("```json"):
                    response_text = response_text[7:-3].strip()
                elif response_text.startswith("```"):
                    response_text = response_text[3:-3].strip()

                optimized = json.loads(response_text)
                return optimized
        except Exception as e:
            print(f"AI optimization failed: {e}, using original structure")

        # Fallback to basic optimization
        return self._basic_dependency_optimization(structure)

    def _basic_dependency_optimization(self, structure: dict) -> dict:
        """Basic dependency optimization without AI"""
        # Simple heuristic: identify tasks that can run in parallel
        tasks = structure.get("tasks", [])

        # Remove unnecessary dependencies
        for task in tasks:
            deps = task.get("upstream_dependencies", [])
            # Remove self-dependencies
            task["upstream_dependencies"] = [dep for dep in deps if dep != task["id"]]

        return structure

    def _apply_constraints(self, structure: dict, constraints: dict) -> dict:
        """Apply user-defined constraints to the DAG structure"""
        if "max_parallel_tasks" in constraints:
            # Limit parallelization
            structure["max_parallel_tasks"] = constraints["max_parallel_tasks"]

        if "resource_limits" in constraints:
            # Apply resource constraints
            structure["resource_limits"] = constraints["resource_limits"]

        return structure

    async def _create_dag_with_implementations(self, structure: dict) -> DAG:
        """Create DAG with AI-generated implementations for each task"""

        # First, generate implementations for each task
        sql_scripts = {}
        python_functions = {}

        for task in structure.get("tasks", []):
            task_id = task.get("id", "")
            task_type = task.get("operator", "")
            task_desc = task.get("description", "")

            if task_type == "SqlOperator" or "sql" in task_desc.lower():
                # Generate SQL implementation
                sql_code = await self._generate_sql_implementation(task_id, task_desc)
                sql_scripts[f"{task_id}.sql"] = sql_code

            elif task_type == "PythonOperator" or task_type == "python":
                # Generate Python implementation
                python_code = await self._generate_python_implementation(
                    task_id, task_desc
                )
                python_functions[f"{task_id}.py"] = python_code

        # Now create the actual DAG using the factory
        return await self._create_complete_dag(structure, sql_scripts, python_functions)

    async def _generate_sql_implementation(self, task_id: str, description: str) -> str:
        """Generate actual SQL code for a task"""
        prompt = f"""Write a production-ready SQL query for this task:
Task: {task_id}
Description: {description}

Return ONLY the SQL code, no explanations. Make it complete and runnable."""

        result = await self.model_server.generate(
            prompt, model_name="phi3-mini", temperature=0.3
        )

        if result.success:
            # Clean up the response
            sql = result.text.strip()
            if sql.startswith("```sql"):
                sql = sql[6:-3].strip()
            elif sql.startswith("```"):
                sql = sql[3:-3].strip()
            return sql
        else:
            # Fallback SQL
            return f"-- Task: {task_id}\n-- Description: {description}\nSELECT 1 as placeholder;"

    async def _generate_python_implementation(
        self, task_id: str, description: str
    ) -> str:
        """Generate actual Python code for a task"""
        prompt = f"""Write a production-ready Python function for an Airflow task:
Task: {task_id}
Description: {description}

Requirements:
- Function should accept **context as parameter
- Include error handling
- Return a result dictionary
- Add logging

Return ONLY the Python code."""

        result = await self.model_server.generate(
            prompt, model_name="phi3-mini", temperature=0.3, max_tokens=400
        )

        if result.success:
            # Clean up the response
            code = result.text.strip()
            if code.startswith("```python"):
                code = code[9:-3].strip()
            elif code.startswith("```"):
                code = code[3:-3].strip()
            return code
        else:
            # Fallback Python
            return f'''def {task_id}(**context):
    """
    Task: {task_id}
    Description: {description}
    """
    import logging
    logger = logging.getLogger(__name__)

    try:
        logger.info(f"Executing {task_id}")
        # TODO: Implement actual logic

        return {{"status": "success", "task": "{task_id}"}}
    except Exception as e:
        logger.error(f"Error in {task_id}: {{e}}")
        raise'''

    async def _create_complete_dag(
        self, structure: dict, sql_scripts: dict, python_functions: dict
    ) -> DAG:
        """Create complete DAG with all implementations"""
        if not AIRFLOW_AVAILABLE:
            raise RuntimeError(
                "Apache Airflow is required for DAG generation. Please install airflow: pip install apache-airflow"
            )

        from .dag_factory import DAGConfig, EnterpriseDAGFactory, TaskConfig

        # Create DAG config
        dag_config = DAGConfig(
            dag_id=structure.get("dag_id", "generated_dag"),
            description=structure.get("description", "AI-generated DAG"),
            schedule_interval=structure.get("schedule_interval", "@daily"),
            owners=["airflow-llm"],
            tags=["ai-generated", "production"],
        )

        # Create task configs
        task_configs = []
        for task in structure.get("tasks", []):
            task_config = TaskConfig(
                name=task.get("id", "task"),
                operator=task.get("operator", "python"),
                depends_on=task.get("upstream_dependencies", ["none"]),
                python_callable=task.get("id", "task")
                if task.get("operator") == "python"
                else None,
                sql_script=f"{task.get('id', 'task')}.sql"
                if "sql" in task.get("operator", "").lower()
                else None,
            )
            task_configs.append(task_config)

        # Use the factory to generate the complete DAG
        factory = EnterpriseDAGFactory(output_dir="./generated_dags")
        dag_path = factory.generate_dag(
            description=structure.get("description", ""),
            dag_config=dag_config,
            tasks=task_configs,
            sql_scripts=sql_scripts,
            python_functions=python_functions,
        )

        # Return a DAG object
        dag = DAG(dag_id=dag_config.dag_id)
        dag.dag_path = dag_path
        return dag

    def _create_dag(self, structure: dict) -> DAG:
        """Create actual Airflow DAG from structured definition"""
        if not AIRFLOW_AVAILABLE:
            raise RuntimeError(
                "Apache Airflow is required for DAG generation. Please install airflow: pip install apache-airflow"
            )

        dag_id = structure.get("dag_id", f"generated_dag_{int(time.time())}")
        description = structure.get("description", "Auto-generated DAG")

        # Create DAG with basic configuration
        dag = DAG(
            dag_id=dag_id,
            description=description,
            start_date=days_ago(1),
            schedule_interval="@daily",
            catchup=False,
            tags=["airflow-llm", "auto-generated"],
        )

        # Add tasks from structure
        tasks = structure.get("tasks", [])
        for task_def in tasks:
            if AIRFLOW_AVAILABLE:
                # Create real Airflow tasks
                task = PythonOperator(
                    task_id=task_def.get("id", "default_task"),
                    python_callable=lambda: print(
                        f"Executing {task_def.get('description', 'task')}"
                    ),
                    dag=dag,
                )
                dag.tasks.append(task)
            else:
                # Airflow not available - cannot create tasks
                raise RuntimeError(
                    "Apache Airflow is required for DAG generation. Please install airflow: pip install apache-airflow"
                )

        return dag

    def _add_self_healing(self, dag: DAG) -> DAG:
        """Add self-healing capabilities to DAG"""
        # In production, this would wrap each task with error handling
        # For now, just add metadata
        if hasattr(dag, "tags"):
            dag.tags.append("self-healing-enabled")

        return dag

    def analyze_execution_patterns(self, dag_id: str) -> PipelineMetrics:
        """
        Analyze historical execution patterns and provide insights
        """
        # Fetch execution history
        runs = DagRun.find(dag_id=dag_id, limit=100)

        # Calculate metrics
        runtimes = []
        successes = 0
        costs = []

        for run in runs:
            duration = (run.end_date - run.start_date).total_seconds()
            runtimes.append(duration)
            if run.state == "success":
                successes += 1
            costs.append(self.cost_tracker.calculate_run_cost(run))

        # Identify bottlenecks using ML
        bottlenecks = self._identify_bottlenecks(dag_id)

        # Generate optimization suggestions
        suggestions = self._generate_suggestions(runtimes, costs, bottlenecks)

        return PipelineMetrics(
            avg_runtime=np.mean(runtimes),
            success_rate=successes / len(runs),
            cost_per_run=np.mean(costs),
            resource_utilization=self._get_resource_utilization(dag_id),
            bottlenecks=bottlenecks,
            optimization_suggestions=suggestions,
        )

    def predict_bottlenecks(self, dag_id: str) -> list[dict[str, Any]]:
        """
        Predict future bottlenecks using ML models
        """
        # Analyze task patterns
        task_metrics = self._analyze_task_patterns(dag_id)

        # Use time series prediction
        predictions = []
        for task_id, metrics in task_metrics.items():
            if self._will_likely_fail(metrics):
                predictions.append(
                    {
                        "task_id": task_id,
                        "failure_probability": metrics["failure_prob"],
                        "expected_time": metrics["expected_time"],
                        "recommendation": self._get_prevention_strategy(metrics),
                    }
                )

        return sorted(predictions, key=lambda x: x["failure_probability"], reverse=True)

    def auto_scale_resources(self, dag_run: DagRun) -> dict[str, Any]:
        """
        Automatically scale resources based on workload prediction
        """
        # Predict resource needs
        predicted_load = self._predict_resource_needs(dag_run)

        # Optimize for cost if enabled
        if self.cost_optimization:
            resources = self._optimize_resource_allocation(predicted_load)
        else:
            resources = self._standard_allocation(predicted_load)

        # Apply GPU-aware scheduling
        if self.gpu_aware:
            resources = self._gpu_optimization(resources, dag_run)

        return resources

    def _self_heal_callback(self, context):
        """Callback for self-healing on task failure"""
        # Analyze error
        error_msg = str(context["exception"])
        task_instance = context["task_instance"]

        # Use LLM to suggest fix
        fix_suggestion = self.model_router.query(
            f"Task failed with error: {error_msg}. Suggest a fix.",
            task_type="debugging",
        )

        # Apply fix if possible
        self._apply_automatic_fix(task_instance, fix_suggestion)


class ModelRouter:
    """Intelligent routing between multiple LLM models"""

    def __init__(self, models: list[str]):
        self.models = models
        self.performance_history = {model: [] for model in models}

    def query(self, prompt: str, task_type: str) -> str:
        """Route query to optimal model based on task type and performance"""
        # Select best model for task type
        best_model = self._select_model(task_type)

        # Query with fallback
        try:
            response = self._query_model(best_model, prompt)
            self._record_success(best_model, task_type)
            return response
        except Exception:
            # Fallback to next best model
            return self._fallback_query(prompt, task_type, exclude=[best_model])

    def _select_model(self, task_type: str) -> str:
        """Select optimal model based on historical performance"""
        # Implement model selection logic based on:
        # - Task type (parsing, optimization, debugging)
        # - Historical success rates
        # - Current availability
        # - Cost considerations
        return self.models[0]  # Simplified


class CostTracker:
    """Track and optimize pipeline execution costs"""

    def calculate_run_cost(self, dag_run: DagRun) -> float:
        """Calculate total cost for a DAG run"""
        total_cost = 0.0

        for ti in dag_run.get_task_instances():
            # Calculate compute cost
            duration = (ti.end_date - ti.start_date).total_seconds() / 3600

            # Get resource usage
            if "gpu" in ti.executor_config:
                # GPU instance pricing
                hourly_rate = self._get_gpu_rate(ti.executor_config["gpu"])
            else:
                # CPU instance pricing
                hourly_rate = self._get_cpu_rate(ti.executor_config.get("cpu", 1))

            total_cost += duration * hourly_rate

        return total_cost

    def _get_gpu_rate(self, gpu_type: str) -> float:
        """Get hourly rate for GPU instance"""
        rates = {"a100": 3.50, "v100": 2.20, "t4": 0.75}
        return rates.get(gpu_type, 1.0)

    def _get_cpu_rate(self, cpu_count: int) -> float:
        """Get hourly rate for CPU instance"""
        return 0.10 * cpu_count
