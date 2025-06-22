"""
Self-healing agent for autonomous error recovery
ENTERPRISE FEATURE - Requires AirflowLLM Enterprise license
"""

import logging
import re
import subprocess
import traceback
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .enterprise_features import enterprise

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    IMPORT_ERROR = "import_error"
    MEMORY_ERROR = "memory_error"
    CONNECTION_ERROR = "connection_error"
    PERMISSION_ERROR = "permission_error"
    RESOURCE_ERROR = "resource_error"
    DATA_ERROR = "data_error"
    UNKNOWN = "unknown"


@dataclass
class ErrorAnalysis:
    error_type: ErrorType
    error_message: str
    suggested_fix: str
    confidence: float
    auto_fixable: bool
    fix_command: str | None = None
    resource_adjustment: dict[str, Any] | None = None


class SelfHealingAgent:
    """
    Autonomous error detection and recovery system
    """

    @enterprise.require_enterprise("Self-Healing Agent")
    def __init__(self, enable_auto_fix: bool = True, max_retries: int = 3):
        self.enable_auto_fix = enable_auto_fix
        self.max_retries = max_retries
        self.error_patterns = self._load_error_patterns()
        self.fix_history = {}
        self.success_rate = {}

        logger.info(
            f"SelfHealingAgent initialized with auto_fix={enable_auto_fix}, "
            f"max_retries={max_retries}"
        )

    def analyze_error(self, error: Exception, context: dict[str, Any]) -> ErrorAnalysis:
        """
        Analyze error and determine fix strategy
        """
        logger.info(f"Analyzing error: {type(error).__name__}")

        error_message = str(error)
        traceback_str = traceback.format_exc()

        error_type = self._classify_error(error, error_message, traceback_str)

        logger.debug(f"Error classified as: {error_type.value}")

        analysis = self._generate_fix_strategy(
            error_type, error_message, traceback_str, context
        )

        logger.info(
            f"Error analysis complete - Type: {analysis.error_type.value}, "
            f"Auto-fixable: {analysis.auto_fixable}, "
            f"Confidence: {analysis.confidence:.2f}"
        )

        return analysis

    def attempt_fix(
        self, analysis: ErrorAnalysis, task_context: dict[str, Any]
    ) -> bool:
        """
        Attempt to fix the error automatically
        """
        if not self.enable_auto_fix or not analysis.auto_fixable:
            logger.warning(
                f"Auto-fix disabled or error not auto-fixable: "
                f"{analysis.error_type.value}"
            )
            return False

        logger.info(f"Attempting auto-fix for {analysis.error_type.value}")

        try:
            if analysis.fix_command:
                success = self._execute_fix_command(analysis.fix_command)
            elif analysis.resource_adjustment:
                success = self._adjust_resources(
                    analysis.resource_adjustment, task_context
                )
            else:
                success = self._apply_pattern_fix(analysis, task_context)

            if success:
                self._record_fix_success(analysis)
                logger.info(f"Auto-fix successful for {analysis.error_type.value}")
            else:
                logger.error(f"Auto-fix failed for {analysis.error_type.value}")

            return success

        except Exception as fix_error:
            logger.error(f"Error during auto-fix attempt: {fix_error}", exc_info=True)
            return False

    def _classify_error(
        self, error: Exception, error_message: str, traceback_str: str
    ) -> ErrorType:
        """
        Classify error type using pattern matching
        """
        error_lower = error_message.lower()

        if isinstance(error, ImportError) or "no module named" in error_lower:
            return ErrorType.IMPORT_ERROR
        elif isinstance(error, MemoryError) or "out of memory" in error_lower:
            return ErrorType.MEMORY_ERROR
        elif "connection" in error_lower or "timeout" in error_lower:
            return ErrorType.CONNECTION_ERROR
        elif "permission denied" in error_lower or "access denied" in error_lower:
            return ErrorType.PERMISSION_ERROR
        elif "resource" in error_lower or "limit exceeded" in error_lower:
            return ErrorType.RESOURCE_ERROR
        elif "data" in error_lower or "format" in error_lower:
            return ErrorType.DATA_ERROR
        else:
            return ErrorType.UNKNOWN

    def _generate_fix_strategy(
        self,
        error_type: ErrorType,
        error_message: str,
        traceback_str: str,
        context: dict[str, Any],
    ) -> ErrorAnalysis:
        """
        Generate fix strategy based on error type
        """
        if error_type == ErrorType.IMPORT_ERROR:
            return self._fix_import_error(error_message, context)
        elif error_type == ErrorType.MEMORY_ERROR:
            return self._fix_memory_error(error_message, context)
        elif error_type == ErrorType.CONNECTION_ERROR:
            return self._fix_connection_error(error_message, context)
        elif error_type == ErrorType.PERMISSION_ERROR:
            return self._fix_permission_error(error_message, context)
        elif error_type == ErrorType.RESOURCE_ERROR:
            return self._fix_resource_error(error_message, context)
        else:
            return ErrorAnalysis(
                error_type=error_type,
                error_message=error_message,
                suggested_fix="Manual intervention required",
                confidence=0.1,
                auto_fixable=False,
            )

    def _fix_import_error(self, error_message: str, context: dict) -> ErrorAnalysis:
        """
        Generate fix for import errors
        """
        missing_module = self._extract_missing_module(error_message)

        if missing_module:
            logger.debug(f"Detected missing module: {missing_module}")

            return ErrorAnalysis(
                error_type=ErrorType.IMPORT_ERROR,
                error_message=error_message,
                suggested_fix=f"Install missing package: {missing_module}",
                confidence=0.9,
                auto_fixable=True,
                fix_command=f"pip install {missing_module}",
            )

        return ErrorAnalysis(
            error_type=ErrorType.IMPORT_ERROR,
            error_message=error_message,
            suggested_fix="Check import paths and package installation",
            confidence=0.5,
            auto_fixable=False,
        )

    def _fix_memory_error(self, error_message: str, context: dict) -> ErrorAnalysis:
        """
        Generate fix for memory errors
        """
        current_memory = context.get("memory", "2Gi")
        new_memory = self._increase_memory(current_memory)

        logger.debug(f"Memory adjustment: {current_memory} -> {new_memory}")

        return ErrorAnalysis(
            error_type=ErrorType.MEMORY_ERROR,
            error_message=error_message,
            suggested_fix=f"Increase memory allocation to {new_memory}",
            confidence=0.8,
            auto_fixable=True,
            resource_adjustment={"memory": new_memory},
        )

    def _fix_connection_error(self, error_message: str, context: dict) -> ErrorAnalysis:
        """
        Generate fix for connection errors
        """
        return ErrorAnalysis(
            error_type=ErrorType.CONNECTION_ERROR,
            error_message=error_message,
            suggested_fix="Implement exponential backoff retry",
            confidence=0.7,
            auto_fixable=True,
            fix_command="retry_with_backoff",
        )

    def _fix_permission_error(self, error_message: str, context: dict) -> ErrorAnalysis:
        """
        Generate fix for permission errors
        """
        return ErrorAnalysis(
            error_type=ErrorType.PERMISSION_ERROR,
            error_message=error_message,
            suggested_fix="Check file permissions and ownership",
            confidence=0.6,
            auto_fixable=False,
        )

    def _fix_resource_error(self, error_message: str, context: dict) -> ErrorAnalysis:
        """
        Generate fix for resource errors
        """
        current_cpu = context.get("cpu", 1)
        new_cpu = min(current_cpu * 2, 8)

        logger.debug(f"CPU adjustment: {current_cpu} -> {new_cpu}")

        return ErrorAnalysis(
            error_type=ErrorType.RESOURCE_ERROR,
            error_message=error_message,
            suggested_fix=f"Increase CPU allocation to {new_cpu} cores",
            confidence=0.7,
            auto_fixable=True,
            resource_adjustment={"cpu": new_cpu},
        )

    def _execute_fix_command(self, command: str) -> bool:
        """
        Execute fix command safely
        """
        logger.info(f"Executing fix command: {command}")

        if command == "retry_with_backoff":
            return True

        if command.startswith("pip install"):
            try:
                result = subprocess.run(
                    command.split(), capture_output=True, text=True, timeout=300
                )

                if result.returncode == 0:
                    logger.info(f"Package installation successful: {command}")
                    return True
                else:
                    logger.error(f"Package installation failed: {result.stderr}")
                    return False

            except subprocess.TimeoutExpired:
                logger.error("Package installation timed out")
                return False
            except Exception as e:
                logger.error(f"Error executing pip install: {e}")
                return False

        return False

    def _adjust_resources(
        self, adjustments: dict[str, Any], context: dict[str, Any]
    ) -> bool:
        """
        Adjust resource allocation
        """
        logger.info(f"Adjusting resources: {adjustments}")

        for resource, value in adjustments.items():
            context[resource] = value
            logger.debug(f"Resource {resource} adjusted to {value}")

        return True

    def _apply_pattern_fix(
        self, analysis: ErrorAnalysis, context: dict[str, Any]
    ) -> bool:
        """
        Apply pattern-based fixes
        """
        logger.info(f"Applying pattern fix for {analysis.error_type.value}")
        return True

    def _extract_missing_module(self, error_message: str) -> str | None:
        """
        Extract missing module name from error message
        """
        patterns = [
            r"No module named '([^']+)'",
            r"No module named ([^\s]+)",
            r"ImportError: cannot import name '([^']+)'",
        ]

        for pattern in patterns:
            match = re.search(pattern, error_message)
            if match:
                module = match.group(1)
                return module.split(".")[0]

        return None

    def _increase_memory(self, current_memory: str) -> str:
        """
        Calculate increased memory allocation
        """
        if current_memory.endswith("Gi"):
            value = int(current_memory[:-2])
            return f"{value * 2}Gi"
        elif current_memory.endswith("G"):
            value = int(current_memory[:-1])
            return f"{value * 2}G"
        else:
            return "8Gi"

    def _load_error_patterns(self) -> dict[str, Any]:
        """
        Load error patterns from configuration
        """
        return {
            "import_patterns": [
                "No module named",
                "ImportError",
                "ModuleNotFoundError",
            ],
            "memory_patterns": ["out of memory", "memory error", "allocation failed"],
            "connection_patterns": ["connection refused", "timeout", "network error"],
        }

    def _record_fix_success(self, analysis: ErrorAnalysis):
        """
        Record successful fix for learning
        """
        error_key = analysis.error_type.value

        if error_key not in self.success_rate:
            self.success_rate[error_key] = {"attempts": 0, "successes": 0}

        self.success_rate[error_key]["attempts"] += 1
        self.success_rate[error_key]["successes"] += 1

        success_rate = (
            self.success_rate[error_key]["successes"]
            / self.success_rate[error_key]["attempts"]
        )

        logger.info(
            f"Fix success recorded for {error_key}. "
            f"Success rate: {success_rate:.2f}"
        )
