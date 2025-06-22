p# AirflowLLM Quick Start Guide

**Production DAG Generation from Natural Language**

## Installation

```bash
pip install airflow-llm-orchestrator
```

**Optional**: Configure OpenAI API for fallback generation:

```bash
export OPENAI_API_KEY="your-key"
```

## Basic Usage

### Basic DAG Generation

```bash
# Generate production DAG from natural language
airflow-llm generate "Extract customer data from PostgreSQL, validate it, and load to S3"

# Generated output structure:
# customer_data_20241222.py - Complete Airflow DAG
# resources/customer_data_20241222/ - Configuration and scripts
#   ├── config.yml - Environment-specific configuration
#   ├── tasks.yml - Task definitions and dependencies
#   ├── extract.sql - Generated SQL script
#   └── validate.py - Generated Python function
```

### Advanced Generation with Configuration

```bash
airflow-llm generate \
  "Train ML model on sales data and deploy if accuracy > 85%" \
  --dag-id ml_sales_pipeline \
  --owners "data-team,ml-team" \
  --stakeholders "business-team"
```

### Using Configuration Files

Create `pipeline_config.yml`:

```yaml
dag_id: "customer_analytics"
description: "Daily customer analytics pipeline"
owners: ["analytics-team"]
stakeholders: ["business-team", "product-team"]

tasks:
  - name: "extract_data"
    operator: "postgresql"
    depends_on: ["none"]
    sql_script: "extract.sql"

  - name: "transform_data"
    operator: "python"
    depends_on: ["extract_data"]
    python_callable: "transform_function"
```

```bash
airflow-llm create --config pipeline_config.yml
```

## Generated Structure

```
generated_dags/
├── dags/
│   ├── customer_analytics.py          # Main DAG file
│   └── resources/
│       └── customer_analytics/
│           ├── config.yml             # Environment configs
│           ├── tasks.yml              # Task definitions
│           ├── extract.sql            # SQL scripts
│           └── transform_function.py  # Python functions
```

## Environment Configuration

Set environment variables:

```bash
export DEPLOYMENT_ENVIRONMENT=prod  # or staging, dev
export OPENAI_API_KEY=your_key      # For AI features
```

## Validation

```bash
airflow-llm validate generated_dags/dags/customer_analytics.py
```

## Production Deployment

1. Copy generated files to Airflow DAGs directory
2. Update SQL scripts and Python functions as needed
3. Configure connections in Airflow UI
4. Enable and test the DAG

## Advanced Features

### Self-Healing Tasks

Generated DAGs include automatic error recovery:

- Dependency installation for ImportErrors
- Resource scaling for memory issues
- Retry logic for transient failures

### Cost Optimization

Automatic cloud cost optimization:

- Multi-cloud price comparison
- Spot instance scheduling
- GPU-aware allocation

### Enterprise Integration

Works with enterprise Airflow patterns:

- Datasci-rx compatible structure
- Environment-specific configurations
- Production monitoring integration

## Production Deployment

1. **Copy to Airflow**: Move generated files to your Airflow DAGs directory
2. **Configure connections**: Set up database and cloud storage connections in Airflow UI
3. **Test locally**: `airflow dags test your_dag_id`
4. **Deploy**: Enable DAG in production Airflow environment

## Enterprise Features

### High-Performance Models

```bash
# Use specialized models for sub-50ms generation
export AIRFLOW_LLM_ENDPOINT="https://enterprise.airflow-llm.dev"
airflow-llm generate "Complex ML pipeline" --enterprise
```

### Private Deployment

```bash
# Deploy models on your infrastructure
docker run -d -p 8080:8080 airflow-llm/enterprise:latest
export AIRFLOW_LLM_ENDPOINT="http://localhost:8080"
```

---

## Enterprise Deployment

Contact: enterprise@airflow-llm.dev

Documentation: https://docs.airflow-llm.dev

Source Code: https://github.com/v-code01/airflow-llm-orchestrator
