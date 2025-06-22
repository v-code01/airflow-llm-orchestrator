"""Utility functions for AirflowLLM enterprise operations."""

from typing import Any

# Default executor configuration for enterprise deployments
EXEC_CONFIG = {
    "cpu_request": "100m",
    "cpu_limit": "1000m",
    "memory_request": "512Mi",
    "memory_limit": "2Gi",
}


def get_enterprise_config() -> dict[str, Any]:
    """Get enterprise-grade default configuration.

    Returns:
        Dictionary containing enterprise configuration defaults
    """
    return {
        "retry_delay_seconds": 300,
        "max_retries": 3,
        "execution_timeout_seconds": 3600,
        "pool": "default_pool",
        "priority_weight": 1,
        "queue": "default",
    }
