"""
Complete ML pipeline example using all AirflowLLM features
"""

import logging

from airflow.operators.python import PythonOperator

from airflow_llm.decorators import (
    cost_aware_execution,
    intelligent_retry,
    natural_language_dag,
    performance_monitor,
    self_healing_task,
)

logger = logging.getLogger(__name__)


@natural_language_dag(
    "Build complete ML pipeline: extract features from data lake, "
    "perform feature engineering, train multiple models, "
    "evaluate performance, deploy best model, and monitor predictions",
    dag_id="complete_ml_pipeline",
    schedule_interval="@daily",
    cost_optimization=True,
    self_healing=True,
    max_active_runs=1,
)
def complete_ml_pipeline():
    """
    Comprehensive ML pipeline with all AirflowLLM features
    """

    @self_healing_task(retries=3, auto_fix=True, resource_scaling=True)
    @cost_aware_execution(max_cost_per_hour=3.0, prefer_spot_instances=True)
    @performance_monitor(track_memory=True, track_cpu=True)
    def extract_features(**context):
        """
        Extract and prepare features from data sources
        """
        logger.info("Starting feature extraction from data lake")

        import numpy as np
        import pandas as pd

        logger.info("Generating synthetic data lake extraction")

        n_samples = 100000
        features = {
            "customer_id": range(n_samples),
            "age": np.random.normal(35, 12, n_samples),
            "income": np.random.lognormal(10, 1, n_samples),
            "credit_score": np.random.normal(650, 100, n_samples),
            "account_balance": np.random.exponential(1000, n_samples),
            "transaction_count": np.random.poisson(15, n_samples),
            "days_since_last_login": np.random.geometric(0.1, n_samples),
            "product_usage_score": np.random.beta(2, 5, n_samples),
        }

        df = pd.DataFrame(features)

        logger.info(f"Extracted {len(df)} samples with {len(df.columns)} features")

        feature_stats = {
            "total_samples": len(df),
            "feature_count": len(df.columns),
            "missing_values": df.isnull().sum().sum(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
        }

        logger.info(f"Feature extraction stats: {feature_stats}")

        return feature_stats

    @self_healing_task(retries=2, auto_fix=True, resource_scaling=True)
    @cost_aware_execution(max_cost_per_hour=2.0, prefer_spot_instances=True)
    @intelligent_retry(max_retries=3, backoff_factor=2.0)
    def feature_engineering(**context):
        """
        Perform advanced feature engineering
        """
        logger.info("Starting feature engineering")

        import numpy as np
        import pandas as pd
        from sklearn.preprocessing import StandardScaler

        logger.info("Regenerating dataset for feature engineering")

        n_samples = 100000
        df = pd.DataFrame(
            {
                "age": np.random.normal(35, 12, n_samples),
                "income": np.random.lognormal(10, 1, n_samples),
                "credit_score": np.random.normal(650, 100, n_samples),
                "account_balance": np.random.exponential(1000, n_samples),
                "transaction_count": np.random.poisson(15, n_samples),
            }
        )

        logger.info("Creating engineered features")

        df["income_age_ratio"] = df["income"] / df["age"]
        df["credit_income_ratio"] = df["credit_score"] / df["income"] * 1000
        df["balance_transaction_ratio"] = df["account_balance"] / (
            df["transaction_count"] + 1
        )
        df["high_value_customer"] = (df["income"] > df["income"].quantile(0.8)).astype(
            int
        )
        df["credit_score_bucket"] = pd.cut(
            df["credit_score"],
            bins=5,
            labels=["Poor", "Fair", "Good", "Very Good", "Excellent"],
        )

        logger.info("Scaling numerical features")

        numerical_features = [
            "age",
            "income",
            "credit_score",
            "account_balance",
            "transaction_count",
            "income_age_ratio",
            "credit_income_ratio",
            "balance_transaction_ratio",
        ]

        scaler = StandardScaler()
        df_scaled = df[numerical_features].copy()
        df_scaled = pd.DataFrame(
            scaler.fit_transform(df_scaled), columns=numerical_features
        )

        engineering_stats = {
            "original_features": 5,
            "engineered_features": len(df.columns) - 5,
            "total_features": len(df.columns),
            "scaled_features": len(numerical_features),
        }

        logger.info(f"Feature engineering complete: {engineering_stats}")

        return engineering_stats

    @self_healing_task(retries=3, auto_fix=True, resource_scaling=True)
    @cost_aware_execution(max_cost_per_hour=8.0, prefer_spot_instances=True)
    @performance_monitor(
        track_memory=True,
        track_cpu=True,
        alert_thresholds={
            "max_execution_time": 3600,
            "max_memory_mb": 16384,
            "max_cpu_percent": 95,
        },
    )
    def train_models(**context):
        """
        Train multiple ML models and compare performance
        """
        logger.info("Starting model training phase")

        import numpy as np
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )
        from sklearn.model_selection import cross_val_score, train_test_split
        from sklearn.svm import SVC

        logger.info("Generating training dataset")

        n_samples = 50000
        n_features = 20

        X = np.random.random((n_samples, n_features))
        y = (
            X[:, 0] + X[:, 1] - X[:, 2] + np.random.normal(0, 0.1, n_samples) > 0.5
        ).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

        models = {
            "random_forest": RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1
            ),
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=100, random_state=42
            ),
            "logistic_regression": LogisticRegression(random_state=42, max_iter=1000),
            "svm": SVC(kernel="rbf", random_state=42, probability=True),
        }

        model_results = {}

        for model_name, model in models.items():
            logger.info(f"Training {model_name}")

            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            cv_scores = cross_val_score(
                model, X_train, y_train, cv=5, scoring="accuracy"
            )

            metrics = {
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(y_test, predictions),
                "recall": recall_score(y_test, predictions),
                "f1_score": f1_score(y_test, predictions),
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
            }

            model_results[model_name] = metrics

            logger.info(f"{model_name} results: {metrics}")

        best_model = max(
            model_results.keys(), key=lambda x: model_results[x]["f1_score"]
        )

        logger.info(
            f"Best model: {best_model} with F1 score: {model_results[best_model]['f1_score']:.4f}"
        )

        training_summary = {
            "models_trained": len(models),
            "best_model": best_model,
            "best_f1_score": model_results[best_model]["f1_score"],
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "feature_count": n_features,
        }

        return training_summary

    @self_healing_task(retries=2, auto_fix=True)
    @cost_aware_execution(max_cost_per_hour=1.0, prefer_spot_instances=True)
    def model_evaluation(**context):
        """
        Comprehensive model evaluation and validation
        """
        logger.info("Starting model evaluation")

        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report, roc_auc_score
        from sklearn.model_selection import train_test_split

        logger.info("Generating evaluation dataset")

        n_samples = 10000
        n_features = 20

        X = np.random.random((n_samples, n_features))
        y = (
            X[:, 0] + X[:, 1] - X[:, 2] + np.random.normal(0, 0.1, n_samples) > 0.5
        ).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        logger.info("Training model for evaluation")

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        prediction_probs = model.predict_proba(X_test)[:, 1]

        logger.info("Calculating evaluation metrics")

        roc_auc = roc_auc_score(y_test, prediction_probs)
        classification_rep = classification_report(
            y_test, predictions, output_dict=True
        )

        evaluation_results = {
            "roc_auc_score": roc_auc,
            "accuracy": classification_rep["accuracy"],
            "precision_class_1": classification_rep["1"]["precision"],
            "recall_class_1": classification_rep["1"]["recall"],
            "f1_score_class_1": classification_rep["1"]["f1-score"],
            "support_class_1": classification_rep["1"]["support"],
            "test_samples": len(X_test),
        }

        logger.info(f"Evaluation complete: {evaluation_results}")

        return evaluation_results

    @self_healing_task(retries=2, auto_fix=True)
    @cost_aware_execution(max_cost_per_hour=0.5, prefer_spot_instances=True)
    @intelligent_retry(max_retries=3)
    def deploy_model(**context):
        """
        Deploy model to production environment
        """
        logger.info("Starting model deployment")

        import time

        logger.info("Simulating model deployment process")

        deployment_steps = [
            "Validating model artifacts",
            "Creating deployment package",
            "Uploading to model registry",
            "Configuring inference endpoint",
            "Running deployment tests",
            "Enabling production traffic",
        ]

        for step in deployment_steps:
            logger.info(f"Deployment step: {step}")
            time.sleep(0.5)

        deployment_config = {
            "model_version": "1.0.0",
            "endpoint_url": "https://api.example.com/model/predict",
            "deployment_time": time.time(),
            "environment": "production",
            "auto_scaling_enabled": True,
            "max_instances": 10,
            "min_instances": 2,
        }

        logger.info("Model deployment completed successfully")
        logger.info(f"Deployment config: {deployment_config}")

        return deployment_config

    @self_healing_task(retries=2, auto_fix=True)
    @cost_aware_execution(max_cost_per_hour=0.3, prefer_spot_instances=True)
    def monitoring_setup(**context):
        """
        Set up model monitoring and alerting
        """
        logger.info("Setting up model monitoring")

        import time

        monitoring_components = [
            "Prediction accuracy tracking",
            "Data drift detection",
            "Model performance alerts",
            "Resource utilization monitoring",
            "Business metric correlation",
            "Automated retraining triggers",
        ]

        for component in monitoring_components:
            logger.info(f"Configuring: {component}")
            time.sleep(0.3)

        monitoring_config = {
            "accuracy_threshold": 0.85,
            "drift_detection_window": 7,
            "alert_channels": ["email", "slack"],
            "retraining_threshold": 0.80,
            "monitoring_frequency": "hourly",
            "dashboard_url": "https://monitoring.example.com/ml-dashboard",
        }

        logger.info("Monitoring setup completed")
        logger.info(f"Monitoring config: {monitoring_config}")

        return monitoring_config

    return {
        "extract_features": extract_features,
        "feature_engineering": feature_engineering,
        "train_models": train_models,
        "model_evaluation": model_evaluation,
        "deploy_model": deploy_model,
        "monitoring_setup": monitoring_setup,
    }


pipeline_functions = complete_ml_pipeline()

extract_task = PythonOperator(
    task_id="extract_features",
    python_callable=pipeline_functions["extract_features"],
)

engineering_task = PythonOperator(
    task_id="feature_engineering",
    python_callable=pipeline_functions["feature_engineering"],
)

training_task = PythonOperator(
    task_id="train_models",
    python_callable=pipeline_functions["train_models"],
)

evaluation_task = PythonOperator(
    task_id="model_evaluation",
    python_callable=pipeline_functions["model_evaluation"],
)

deployment_task = PythonOperator(
    task_id="deploy_model",
    python_callable=pipeline_functions["deploy_model"],
)

monitoring_task = PythonOperator(
    task_id="monitoring_setup",
    python_callable=pipeline_functions["monitoring_setup"],
)

(
    extract_task
    >> engineering_task
    >> training_task
    >> evaluation_task
    >> deployment_task
    >> monitoring_task
)
