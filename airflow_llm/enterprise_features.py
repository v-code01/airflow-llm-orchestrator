"""
Enterprise-only features for AirflowLLM
Requires valid license key for activation
"""

import os


class EnterpriseFeatures:
    """Gate enterprise features behind license validation"""

    def __init__(self):
        self.license_key = os.getenv("AIRFLOW_LLM_LICENSE_KEY")
        self.is_enterprise = self._validate_license()

    def _validate_license(self) -> bool:
        """Validate enterprise license key"""
        if not self.license_key:
            return False

        # Simple validation for demo - in production this would
        # validate against license server
        return self.license_key.startswith("enterprise_")

    def require_enterprise(self, feature_name: str):
        """Decorator to gate enterprise features"""

        def decorator(func):
            def wrapper(*args, **kwargs):
                if not self.is_enterprise:
                    raise LicenseError(
                        f"{feature_name} is an enterprise feature. "
                        f"Upgrade to AirflowLLM Enterprise or use our managed service. "
                        f"Contact sales@airflowllm.com"
                    )
                return func(*args, **kwargs)

            return wrapper

        return decorator


class LicenseError(Exception):
    """Raised when enterprise feature is accessed without license"""


# Global instance
enterprise = EnterpriseFeatures()
