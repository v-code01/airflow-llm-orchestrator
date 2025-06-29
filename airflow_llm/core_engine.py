"""
Production-Grade DAG Generation Engine
Generates complete working implementations, not templates
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class TaskType(Enum):
    SQL_EXTRACTION = "sql_extraction"
    SQL_TRANSFORMATION = "sql_transformation"
    PYTHON_PROCESSING = "python_processing"
    DATA_VALIDATION = "data_validation"
    API_INTEGRATION = "api_integration"
    NOTIFICATION = "notification"
    ML_INFERENCE = "ml_inference"
    FILE_OPERATIONS = "file_operations"


@dataclass
class CodeGenerationRequest:
    task_type: TaskType
    description: str
    table_schema: dict | None = None
    source_system: str | None = None
    target_system: str | None = None
    business_rules: list[str] | None = None
    performance_requirements: dict | None = None


@dataclass
class GeneratedImplementation:
    function_code: str
    imports: list[str]
    airflow_operator: str
    dependencies: list[str]
    error_handling: str
    monitoring_code: str
    estimated_runtime: str
    resource_requirements: dict[str, Any]


class ProductionCodeGenerator:
    """
    Production-grade code generator using specialized prompts
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.generation_pipeline = None
        self._initialize_model()

        # S-tier prompt templates for each task type
        self.prompt_templates = {
            TaskType.SQL_EXTRACTION: self._sql_extraction_prompt,
            TaskType.SQL_TRANSFORMATION: self._sql_transformation_prompt,
            TaskType.PYTHON_PROCESSING: self._python_processing_prompt,
            TaskType.DATA_VALIDATION: self._data_validation_prompt,
            TaskType.API_INTEGRATION: self._api_integration_prompt,
            TaskType.NOTIFICATION: self._notification_prompt,
            TaskType.ML_INFERENCE: self._ml_inference_prompt,
            TaskType.FILE_OPERATIONS: self._file_operations_prompt,
        }

    def _initialize_model(self):
        """Initialize lightweight but powerful code generation model"""
        if not TORCH_AVAILABLE:
            logger.error(
                "PyTorch not available. Install with: pip install torch transformers"
            )
            return

        try:
            # Use CodeLlama which is proven for code generation
            model_name = "codellama/CodeLlama-7b-Python-hf"
            logger.info(f"Loading production model: {model_name}")

            # Try CodeLlama first, fallback to smaller models if needed
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=(
                        torch.float16 if torch.cuda.is_available() else torch.float32
                    ),
                    device_map="auto" if torch.cuda.is_available() else None,
                )
                logger.info("Loaded CodeLlama-7b-Python successfully")
            except Exception as e:
                logger.warning(f"CodeLlama failed: {e}, trying smaller model...")
                # Fallback to smaller model
                model_name = "microsoft/DialoGPT-medium"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
                logger.info("Loaded DialoGPT-medium as fallback")

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.generation_pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=2048,
                temperature=0.1,  # Low temperature for consistent code generation
                do_sample=True,
                device=0 if torch.cuda.is_available() else -1,
            )

            logger.info("Production model pipeline ready")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Fallback to OpenAI if available
            if os.getenv("OPENAI_API_KEY"):
                logger.info("Falling back to OpenAI API")
                self._use_openai_fallback()

    def _use_openai_fallback(self):
        """Fallback to OpenAI API for code generation"""
        try:
            import openai

            openai.api_key = os.getenv("OPENAI_API_KEY")
            self.generation_pipeline = self._openai_generate
            logger.info("OpenAI fallback initialized")
        except ImportError:
            logger.error("OpenAI not available. Install with: pip install openai")

    def _openai_generate(self, prompt: str) -> str:
        """Generate code using OpenAI API"""
        import openai

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert Airflow data engineer. Generate production-quality, complete function implementations.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=2000,
            temperature=0.1,
        )

        return response.choices[0].message.content

    def generate_implementation(
        self, request: CodeGenerationRequest
    ) -> GeneratedImplementation:
        """
        Generate complete working implementation for a task
        """
        if self.generation_pipeline is None:
            raise RuntimeError("No code generation model available")

        # Get specialized prompt for task type
        prompt_generator = self.prompt_templates[request.task_type]
        prompt = prompt_generator(request)

        # Generate code
        start_time = time.time()

        if callable(self.generation_pipeline):
            # OpenAI API
            generated_code = self.generation_pipeline(prompt)
        else:
            # Local model
            outputs = self.generation_pipeline(prompt)
            generated_code = outputs[0]["generated_text"]
            # Remove the prompt from output
            generated_code = generated_code[len(prompt) :].strip()

        generation_time = time.time() - start_time

        # Parse and structure the generated code
        implementation = self._parse_generated_code(generated_code, request.task_type)

        logger.info(f"Generated implementation in {generation_time:.2f}s")

        return implementation

    def _sql_extraction_prompt(self, request: CodeGenerationRequest) -> str:
        """S-tier prompt for SQL data extraction tasks"""
        schema_info = ""
        if request.table_schema:
            schema_info = f"Table schema: {json.dumps(request.table_schema, indent=2)}"

        return f"""
Generate a production-ready Airflow task function for SQL data extraction.

Requirements:
- Task: {request.description}
- Source: {request.source_system or 'Database'}
- {schema_info}
- Include proper error handling, logging, and data validation
- Use appropriate Airflow hooks (PostgresHook, MySqlHook, etc.)
- Implement incremental loading where applicable
- Add data quality checks
- Return meaningful metrics

Generate ONLY the Python function implementation with imports:

```python
def extract_data(**context):
    \"\"\"
    Production implementation for: {request.description}
    \"\"\"
    # Your implementation here
```

Requirements for S-tier code:
1. Proper parameterization using Airflow variables/context
2. Comprehensive error handling with specific exception types
3. Detailed logging for observability
4. Data validation and quality checks
5. Performance optimization (LIMIT, indexing hints)
6. Connection pooling and resource cleanup
7. Incremental loading logic with watermarks
8. XCom data passing with proper serialization
9. Retry logic for transient failures
10. Monitoring metrics and alerts
"""

    def _sql_transformation_prompt(self, request: CodeGenerationRequest) -> str:
        """S-tier prompt for SQL transformations"""
        return f"""
Generate a production-ready SQL transformation function.

Task: {request.description}
Source: {request.source_system}
Target: {request.target_system}
Business Rules: {request.business_rules or []}

Create a complete implementation that:
1. Handles complex SQL transformations
2. Implements business logic correctly
3. Includes data lineage tracking
4. Optimizes for performance
5. Handles data type conversions
6. Implements proper JOIN strategies
7. Includes aggregation logic
8. Validates transformation results

```python
def transform_data(**context):
    \"\"\"
    Production SQL transformation: {request.description}
    \"\"\"
    # Implementation with optimized SQL
```

Focus on:
- Window functions for analytics
- CTEs for readable complex logic
- Proper indexing strategies
- Batch processing for large datasets
- Data freshness validation
- Schema evolution handling
"""

    def _python_processing_prompt(self, request: CodeGenerationRequest) -> str:
        """S-tier prompt for Python data processing"""
        return f"""
Generate production-grade Python data processing code.

Task: {request.description}
Performance Requirements: {request.performance_requirements or {}}

Create efficient Python implementation with:
1. Pandas/Polars for data manipulation
2. Proper memory management
3. Vectorized operations
4. Error handling for edge cases
5. Progress tracking for long operations
6. Parallel processing where applicable
7. Data validation at each step
8. Resource monitoring

```python
def process_data(**context):
    \"\"\"
    Production data processing: {request.description}
    \"\"\"
    import pandas as pd
    import numpy as np
    from typing import Dict, List, Any
    import logging

    # Your implementation here
```

Requirements:
- Use efficient data structures
- Implement chunked processing for large datasets
- Include comprehensive data validation
- Add performance monitoring
- Handle memory constraints
- Implement proper error recovery
- Use type hints for maintainability
- Add detailed docstrings and comments
"""

    def _data_validation_prompt(self, request: CodeGenerationRequest) -> str:
        """S-tier prompt for data validation"""
        return f"""
Generate comprehensive data validation function.

Task: {request.description}
Business Rules: {request.business_rules or []}

Create validation logic that:
1. Checks data quality dimensions
2. Implements business rule validation
3. Provides detailed error reporting
4. Supports configurable thresholds
5. Generates data quality metrics
6. Handles schema validation
7. Detects anomalies and outliers
8. Provides remediation suggestions

```python
def validate_data(**context):
    \"\"\"
    Production data validation: {request.description}
    \"\"\"
    from great_expectations import DataContext
    import pandas as pd
    from typing import Dict, List, Tuple

    # Implementation here
```

Include:
- Statistical validation (nulls, duplicates, ranges)
- Business rule validation
- Schema compliance checking
- Data freshness validation
- Referential integrity checks
- Custom validation rules
- Detailed quality reports
- Alerting for quality issues
"""

    def _api_integration_prompt(self, request: CodeGenerationRequest) -> str:
        """S-tier prompt for API integration"""
        return f"""
Generate robust API integration code.

Task: {request.description}
Source API: {request.source_system}

Create production API client with:
1. Proper authentication handling
2. Rate limiting and backoff
3. Error handling and retries
4. Request/response validation
5. Pagination handling
6. Timeout management
7. Circuit breaker pattern
8. Comprehensive logging

```python
def api_integration(**context):
    \"\"\"
    Production API integration: {request.description}
    \"\"\"
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    import time
    import json
    from typing import Dict, List, Optional

    # Implementation here
```

Requirements:
- Implement exponential backoff
- Handle various HTTP status codes
- Parse and validate JSON responses
- Implement authentication (OAuth, API keys)
- Add request/response logging
- Handle large datasets with pagination
- Implement caching strategies
- Monitor API health and performance
"""

    def _notification_prompt(self, request: CodeGenerationRequest) -> str:
        """S-tier prompt for notifications"""
        return f"""
Generate production notification system.

Task: {request.description}
Notification Type: {request.target_system or 'Email/Slack'}

Create notification function with:
1. Multiple notification channels
2. Template-based messaging
3. Conditional notification logic
4. Error notification handling
5. Rich formatting support
6. Attachment handling
7. Notification throttling
8. Delivery confirmation

```python
def send_notification(**context):
    \"\"\"
    Production notification system: {request.description}
    \"\"\"
    from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
    from airflow.providers.email.operators.email import EmailOperator
    import json
    from typing import Dict, List, Optional

    # Implementation here
```

Include:
- Multi-channel delivery (email, Slack, PagerDuty)
- Dynamic message templating
- Conditional logic based on task status
- Rich formatting with charts/tables
- File attachments and screenshots
- Notification preferences
- Delivery tracking and confirmation
- Escalation policies
"""

    def _ml_inference_prompt(self, request: CodeGenerationRequest) -> str:
        """S-tier prompt for ML inference"""
        return f"""
Generate production ML inference pipeline.

Task: {request.description}
Model Type: {request.source_system or 'ML Model'}

Create ML inference function with:
1. Model loading and versioning
2. Input data preprocessing
3. Batch inference optimization
4. Output post-processing
5. Model performance monitoring
6. A/B testing support
7. Feature drift detection
8. Scalable inference architecture

```python
def ml_inference(**context):
    \"\"\"
    Production ML inference: {request.description}
    \"\"\"
    import joblib
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    import mlflow
    from typing import Dict, List, Any

    # Implementation here
```

Requirements:
- Efficient model loading and caching
- Robust preprocessing pipeline
- Batch processing for scalability
- Model version management
- Performance monitoring and alerting
- Feature engineering pipeline
- Output validation and formatting
- Integration with model registry
"""

    def _file_operations_prompt(self, request: CodeGenerationRequest) -> str:
        """S-tier prompt for file operations"""
        return f"""
Generate production file operations code.

Task: {request.description}
Source: {request.source_system}
Target: {request.target_system}

Create file processing function with:
1. Multiple format support (CSV, Parquet, JSON, Avro)
2. Large file handling with streaming
3. Compression and encryption
4. Data partitioning strategies
5. Schema evolution handling
6. File validation and checksums
7. Atomic operations
8. Cleanup and archival

```python
def file_operations(**context):
    \"\"\"
    Production file operations: {request.description}
    \"\"\"
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pathlib import Path
    import boto3
    from typing import Dict, List, Optional

    # Implementation here
```

Include:
- Multi-format reading/writing
- Streaming for large files
- Parallel processing
- Data compression optimization
- Schema validation
- File integrity checks
- Cloud storage integration (S3, GCS, Azure)
- Backup and recovery procedures
"""

    def _parse_generated_code(
        self, generated_code: str, task_type: TaskType
    ) -> GeneratedImplementation:
        """
        Parse generated code and extract components
        """
        # Extract function code
        function_match = re.search(
            r"def\s+\w+.*?(?=\n(?:def|\Z))", generated_code, re.DOTALL
        )
        function_code = function_match.group(0) if function_match else generated_code

        # Extract imports
        import_lines = re.findall(
            r"^(?:from .+ import .+|import .+)$", generated_code, re.MULTILINE
        )

        # Determine operator type based on task
        operator_mapping = {
            TaskType.SQL_EXTRACTION: "PythonOperator",
            TaskType.SQL_TRANSFORMATION: "PythonOperator",
            TaskType.PYTHON_PROCESSING: "PythonOperator",
            TaskType.DATA_VALIDATION: "PythonOperator",
            TaskType.API_INTEGRATION: "PythonOperator",
            TaskType.NOTIFICATION: "PythonOperator",
            TaskType.ML_INFERENCE: "PythonOperator",
            TaskType.FILE_OPERATIONS: "PythonOperator",
        }

        # Generate monitoring code
        monitoring_code = f"""
# Add monitoring for {task_type.value}
task_start_time = time.time()
try:
    result = {function_code.split('def ')[1].split('(')[0]}(**context)
    execution_time = time.time() - task_start_time

    # Log metrics
    logging.info(f"Task completed in {{execution_time:.2f}}s")
    context['ti'].xcom_push(key='execution_metrics', value={{
        'execution_time': execution_time,
        'status': 'success',
        'timestamp': time.time()
    }})

    return result
except Exception as e:
    execution_time = time.time() - task_start_time
    logging.error(f"Task failed after {{execution_time:.2f}}s: {{e}}")

    # Push failure metrics
    context['ti'].xcom_push(key='execution_metrics', value={{
        'execution_time': execution_time,
        'status': 'failed',
        'error': str(e),
        'timestamp': time.time()
    }})

    raise
"""

        return GeneratedImplementation(
            function_code=function_code,
            imports=import_lines,
            airflow_operator=operator_mapping[task_type],
            dependencies=[],
            error_handling="Comprehensive error handling included",
            monitoring_code=monitoring_code,
            estimated_runtime="Variable based on data size",
            resource_requirements={"cpu": 2, "memory": "4Gi", "timeout": "3600s"},
        )
