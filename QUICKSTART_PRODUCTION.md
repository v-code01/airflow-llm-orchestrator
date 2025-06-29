# AirflowLLM Production Quickstart

**Generate production Airflow DAGs from natural language in 5 minutes.**

## Real-World Demo: Customer Analytics Pipeline

This quickstart generates a complete customer analytics DAG with real SQL and Python implementations.

### Step 1: Environment Setup

```bash
# Install Ollama (local LLM runtime)
brew install ollama
ollama serve

# Download demo model (2.3GB)
ollama pull phi3:mini

# Clone and install AirflowLLM
git clone https://github.com/vanshverma/airflow-llm-orchestrator
cd airflow-llm-orchestrator
pip install -r requirements.txt
```

### Step 2: Generate Production DAG

```python
# generate_dag.py
from airflow_llm.orchestrator import LLMOrchestrator
import asyncio

async def main():
    # Initialize orchestrator
    orchestrator = LLMOrchestrator(
        models=["phi3:mini"],
        cost_optimization=False,  # Disable for demo
        self_healing=False
    )

    # Production pipeline specification
    description = """
    Create a daily customer analytics pipeline that:

    1. Extracts customer orders from PostgreSQL database
       - Include customer details, order amounts, and timestamps
       - Filter for orders in the last 30 days

    2. Validates data quality
       - Check order amounts are positive
       - Verify customer IDs exist in customer table
       - Remove duplicate orders

    3. Calculates daily revenue metrics
       - Daily revenue totals by customer
       - Customer lifetime value calculations
       - Top customers by revenue

    4. Loads aggregated results to data warehouse
       - Insert into customer_analytics table
       - Update customer_metrics table
       - Create daily summary records

    5. Sends success notification with metrics
       - Email notification with revenue totals
       - Slack alert for high-value customers
    """

    print("🔥 Generating production DAG with real AI implementations...")
    dag = await orchestrator.generate_dag(description)

    print(f"✅ Generated DAG: {dag.dag_id}")
    print(f"📁 DAG location: {dag.dag_path}")

if __name__ == "__main__":
    asyncio.run(main())
```

```bash
python generate_dag.py
```

**Expected Output:**

```
🔥 Generating production DAG with real AI implementations...
✅ Generated DAG: customer_analytics_pipeline
📁 DAG location: generated_dags/dags/customer_analytics_pipeline.py
```

### Step 3: Inspect Generated Files

```bash
ls -la generated_dags/dags/resources/customer_analytics_pipeline/
```

**Generated Structure:**

```
customer_analytics_pipeline/
├── config.yml                  # Environment configurations
├── tasks.yml                   # Task definitions
├── extract_orders.sql          # Real SQL: Customer order extraction
├── calculate_metrics.sql       # Real SQL: Revenue calculations
├── validate_data.py            # Real Python: Data quality checks
├── load_to_warehouse.py        # Real Python: Warehouse loading
└── send_notification.py        # Real Python: Success notifications
```

### Step 4: Review Generated Code

**AI-Generated SQL Example:**

```sql
-- extract_orders.sql (AI-generated)
SELECT
    c.customer_id,
    c.first_name,
    c.last_name,
    c.email,
    o.order_id,
    o.order_date,
    o.total_amount,
    oi.product_id,
    oi.quantity,
    oi.unit_price
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
JOIN order_items oi ON o.order_id = oi.order_id
WHERE o.order_date >= CURRENT_DATE - INTERVAL '30 days'
  AND o.status = 'completed'
ORDER BY o.order_date DESC;
```

**AI-Generated Python Example:**

```python
# validate_data.py (AI-generated)
import logging
from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults

class ValidateDataOperator(BaseOperator):
    """Validates customer order data quality"""

    @apply_defaults
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def execute(self, context):
        logger = logging.getLogger(__name__)

        try:
            # Validate positive order amounts
            negative_orders = self.check_negative_amounts()
            if negative_orders:
                logger.warning(f"Found {len(negative_orders)} orders with negative amounts")

            # Verify customer IDs exist
            invalid_customers = self.check_customer_ids()
            if invalid_customers:
                raise ValueError(f"Invalid customer IDs: {invalid_customers}")

            logger.info("Data validation completed successfully")
            return {"status": "success", "validated_records": True}

        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            raise
```

### Step 5: Production Deployment

**Option A: Local Airflow**

```bash
# Install Airflow (optional - for DAG object creation)
pip install apache-airflow

# Copy generated DAG to Airflow
cp generated_dags/dags/* $AIRFLOW_HOME/dags/
cp -r generated_dags/dags/resources/* $AIRFLOW_HOME/dags/resources/

# Start Airflow
airflow standalone
```

**Option B: Deploy DAG Files Only**

```bash
# The generated .py files are standard Airflow DAGs
# Deploy to your existing Airflow environment:
scp generated_dags/dags/* airflow-server:/opt/airflow/dags/
```

### Step 6: Production Model Upgrade

For production workloads, upgrade to specialized models:

```python
# Production configuration
orchestrator = LLMOrchestrator(
    models=[
        "codellama:7b",      # Better Python code generation
        "sqlcoder:7b",       # Optimized SQL generation
        "phi3:mini"          # Fallback
    ],
    backend="vllm",          # High-throughput inference
    cost_optimization=True,
    self_healing=True
)
```

```bash
# Download production models
ollama pull codellama:7b    # 3.8GB - Better Python code
ollama pull sqlcoder:7b     # 3.8GB - Optimized SQL queries
```

## Performance Expectations

| Component           | Demo (phi3:mini)  | Production (CodeLlama-7B) |
| ------------------- | ----------------- | ------------------------- |
| **Generation Time** | ~15-30 seconds    | ~8-15 seconds             |
| **Code Quality**    | Basic, functional | Production-grade          |
| **SQL Complexity**  | Simple queries    | Complex JOINs, CTEs       |
| **Python Features** | Basic functions   | Classes, error handling   |
| **Model Size**      | 2.3GB             | 3.8GB                     |

## Troubleshooting

### Common Issues

**Issue: "Ollama not running"**

```bash
# Start Ollama service
ollama serve
# In another terminal
ollama pull phi3:mini
```

**Issue: "JSON parsing failed"**

- Normal with phi3:mini - robust fallbacks handle this
- Upgrade to CodeLlama-7B for better output

**Issue: "Airflow import errors"**

- Generated DAGs work without Airflow installed
- Install Airflow only for direct DAG object creation

### Verification Tests

```bash
# Test core functionality
python demo.py

# Test complete pipeline
python test_direct_generation.py

# Production readiness test
python FINAL_PRODUCTION_TEST.py
```

## Next Steps

1. **Customize Templates**: Modify `airflow_llm/dag_factory.py` for your organization
2. **Add Models**: Download `codellama:7b` and `sqlcoder:7b` for production
3. **vLLM Setup**: Deploy with GPU acceleration for team usage
4. **CI/CD Integration**: Automate DAG generation in your pipeline

**Production models available in this repository.**

---

**🚀 You now have a working AI-powered DAG generation system!**
