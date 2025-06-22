"""
Example DAGs using natural language generation
"""

from datetime import timedelta

from airflow_llm import natural_language_dag


@natural_language_dag(
    "Extract customer data from S3, clean and validate it, train a churn prediction model, "
    "and deploy to production if accuracy exceeds 85%",
    dag_id="customer_churn_pipeline",
    schedule_interval="@daily",
    cost_optimization=True,
    self_healing=True,
    max_active_runs=1,
)
def customer_churn_pipeline():
    """
    AI-generated customer churn prediction pipeline
    """


@natural_language_dag(
    "Monitor real-time fraud transactions, aggregate features, score with ML model, "
    "and alert if fraud probability > 0.8 with < 100ms latency",
    dag_id="fraud_detection_realtime",
    schedule_interval=timedelta(minutes=5),
    cost_optimization=True,
    self_healing=True,
)
def fraud_detection_realtime():
    """
    Real-time fraud detection pipeline with sub-100ms requirements
    """


@natural_language_dag(
    "Process daily sales data from multiple sources, normalize formats, "
    "generate business intelligence reports, and update executive dashboard",
    dag_id="sales_bi_pipeline",
    schedule_interval="0 6 * * *",
    cost_optimization=True,
    self_healing=True,
)
def sales_bi_pipeline():
    """
    Sales business intelligence pipeline
    """


@natural_language_dag(
    "Ingest streaming IoT sensor data, detect anomalies using isolation forest, "
    "trigger maintenance alerts, and update predictive maintenance models",
    dag_id="iot_anomaly_detection",
    schedule_interval="@hourly",
    cost_optimization=True,
    self_healing=True,
)
def iot_anomaly_detection():
    """
    IoT anomaly detection and predictive maintenance
    """


@natural_language_dag(
    "Scrape financial news, perform sentiment analysis, calculate market indicators, "
    "train trading models, and execute trades if confidence > 90%",
    dag_id="algorithmic_trading",
    schedule_interval=timedelta(minutes=15),
    cost_optimization=True,
    self_healing=True,
)
def algorithmic_trading():
    """
    Algorithmic trading pipeline with news sentiment analysis
    """


@natural_language_dag(
    "Process medical imaging scans, run CNN diagnostic models, "
    "generate radiologist reports, and flag urgent cases for immediate review",
    dag_id="medical_imaging_analysis",
    schedule_interval="@continuous",
    cost_optimization=True,
    self_healing=True,
)
def medical_imaging_analysis():
    """
    Medical imaging analysis with AI diagnostics
    """
