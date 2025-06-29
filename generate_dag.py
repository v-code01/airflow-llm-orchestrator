# generate_dag.py
import asyncio

from airflow_llm.orchestrator import LLMOrchestrator


async def main():
    # Initialize orchestrator
    orchestrator = LLMOrchestrator(
        models=["phi3:mini"],
        cost_optimization=False,  # Disable for demo
        self_healing=False,
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
