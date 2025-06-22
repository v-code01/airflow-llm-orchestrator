"""
Unit tests for SelfHealingAgent
"""

import unittest.mock as mock

import pytest

from airflow_llm.self_healer import ErrorAnalysis, ErrorType, SelfHealingAgent


class TestSelfHealingAgent:
    @pytest.fixture
    def agent(self):
        return SelfHealingAgent(enable_auto_fix=True, max_retries=3)

    def test_initialization(self, agent):
        assert agent.enable_auto_fix is True
        assert agent.max_retries == 3
        assert isinstance(agent.error_patterns, dict)
        assert isinstance(agent.fix_history, dict)
        assert isinstance(agent.success_rate, dict)

    def test_classify_import_error(self, agent):
        error = ImportError("No module named 'pandas'")
        result = agent._classify_error(error, str(error), "")
        assert result == ErrorType.IMPORT_ERROR

    def test_classify_memory_error(self, agent):
        error = MemoryError("Out of memory")
        result = agent._classify_error(error, str(error), "")
        assert result == ErrorType.MEMORY_ERROR

    def test_classify_connection_error(self, agent):
        error = Exception("Connection timeout")
        result = agent._classify_error(error, "connection failed", "")
        assert result == ErrorType.CONNECTION_ERROR

    def test_classify_permission_error(self, agent):
        error = Exception("Permission denied")
        result = agent._classify_error(error, "permission denied", "")
        assert result == ErrorType.PERMISSION_ERROR

    def test_classify_unknown_error(self, agent):
        error = Exception("Unknown error")
        result = agent._classify_error(error, "unknown error", "")
        assert result == ErrorType.UNKNOWN

    def test_analyze_import_error(self, agent):
        error = ImportError("No module named 'pandas'")
        context = {"task_id": "test_task"}

        result = agent.analyze_error(error, context)

        assert isinstance(result, ErrorAnalysis)
        assert result.error_type == ErrorType.IMPORT_ERROR
        assert result.auto_fixable is True
        assert "pandas" in result.suggested_fix
        assert result.fix_command == "pip install pandas"
        assert result.confidence > 0.8

    def test_analyze_memory_error(self, agent):
        error = MemoryError("Out of memory")
        context = {"memory": "4Gi"}

        result = agent.analyze_error(error, context)

        assert result.error_type == ErrorType.MEMORY_ERROR
        assert result.auto_fixable is True
        assert "8Gi" in result.suggested_fix
        assert result.resource_adjustment == {"memory": "8Gi"}

    def test_analyze_connection_error(self, agent):
        error = Exception("Connection timeout")
        context = {}

        result = agent.analyze_error(error, context)

        assert result.error_type == ErrorType.CONNECTION_ERROR
        assert result.auto_fixable is True
        assert "retry" in result.suggested_fix.lower()

    def test_extract_missing_module_single_quotes(self, agent):
        error_msg = "No module named 'pandas'"
        result = agent._extract_missing_module(error_msg)
        assert result == "pandas"

    def test_extract_missing_module_no_quotes(self, agent):
        error_msg = "No module named numpy"
        result = agent._extract_missing_module(error_msg)
        assert result == "numpy"

    def test_extract_missing_module_nested(self, agent):
        error_msg = "No module named 'sklearn.metrics'"
        result = agent._extract_missing_module(error_msg)
        assert result == "sklearn"

    def test_extract_missing_module_not_found(self, agent):
        error_msg = "Some other error"
        result = agent._extract_missing_module(error_msg)
        assert result is None

    def test_increase_memory_gi_format(self, agent):
        result = agent._increase_memory("4Gi")
        assert result == "8Gi"

    def test_increase_memory_g_format(self, agent):
        result = agent._increase_memory("4G")
        assert result == "8G"

    def test_increase_memory_unknown_format(self, agent):
        result = agent._increase_memory("unknown")
        assert result == "8Gi"

    @mock.patch("subprocess.run")
    def test_execute_fix_command_pip_success(self, mock_subprocess, agent):
        mock_result = mock.MagicMock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result

        result = agent._execute_fix_command("pip install pandas")

        assert result is True
        mock_subprocess.assert_called_once()

    @mock.patch("subprocess.run")
    def test_execute_fix_command_pip_failure(self, mock_subprocess, agent):
        mock_result = mock.MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Package not found"
        mock_subprocess.return_value = mock_result

        result = agent._execute_fix_command("pip install nonexistent")

        assert result is False

    def test_execute_fix_command_retry_with_backoff(self, agent):
        result = agent._execute_fix_command("retry_with_backoff")
        assert result is True

    @mock.patch("subprocess.run")
    def test_execute_fix_command_timeout(self, mock_subprocess, agent):
        import subprocess

        mock_subprocess.side_effect = subprocess.TimeoutExpired("pip", 300)

        result = agent._execute_fix_command("pip install slow-package")

        assert result is False

    def test_adjust_resources(self, agent):
        context = {"cpu": 2, "memory": "4Gi"}
        adjustments = {"memory": "8Gi", "cpu": 4}

        result = agent._adjust_resources(adjustments, context)

        assert result is True
        assert context["memory"] == "8Gi"
        assert context["cpu"] == 4

    @mock.patch("airflow_llm.self_healer.SelfHealingAgent._execute_fix_command")
    def test_attempt_fix_with_command(self, mock_execute, agent):
        mock_execute.return_value = True

        analysis = ErrorAnalysis(
            error_type=ErrorType.IMPORT_ERROR,
            error_message="No module named 'pandas'",
            suggested_fix="Install pandas",
            confidence=0.9,
            auto_fixable=True,
            fix_command="pip install pandas",
        )

        result = agent.attempt_fix(analysis, {})

        assert result is True
        mock_execute.assert_called_once_with("pip install pandas")

    def test_attempt_fix_with_resource_adjustment(self, agent):
        analysis = ErrorAnalysis(
            error_type=ErrorType.MEMORY_ERROR,
            error_message="Out of memory",
            suggested_fix="Increase memory",
            confidence=0.8,
            auto_fixable=True,
            resource_adjustment={"memory": "8Gi"},
        )

        context = {"memory": "4Gi"}
        result = agent.attempt_fix(analysis, context)

        assert result is True
        assert context["memory"] == "8Gi"

    def test_attempt_fix_not_auto_fixable(self, agent):
        analysis = ErrorAnalysis(
            error_type=ErrorType.PERMISSION_ERROR,
            error_message="Permission denied",
            suggested_fix="Check permissions",
            confidence=0.6,
            auto_fixable=False,
        )

        result = agent.attempt_fix(analysis, {})

        assert result is False

    def test_attempt_fix_disabled(self):
        agent = SelfHealingAgent(enable_auto_fix=False)

        analysis = ErrorAnalysis(
            error_type=ErrorType.IMPORT_ERROR,
            error_message="No module named 'pandas'",
            suggested_fix="Install pandas",
            confidence=0.9,
            auto_fixable=True,
            fix_command="pip install pandas",
        )

        result = agent.attempt_fix(analysis, {})

        assert result is False

    def test_record_fix_success_new_error_type(self, agent):
        analysis = ErrorAnalysis(
            error_type=ErrorType.IMPORT_ERROR,
            error_message="Test error",
            suggested_fix="Test fix",
            confidence=0.9,
            auto_fixable=True,
        )

        agent._record_fix_success(analysis)

        assert "import_error" in agent.success_rate
        assert agent.success_rate["import_error"]["attempts"] == 1
        assert agent.success_rate["import_error"]["successes"] == 1

    def test_record_fix_success_existing_error_type(self, agent):
        agent.success_rate["import_error"] = {"attempts": 2, "successes": 1}

        analysis = ErrorAnalysis(
            error_type=ErrorType.IMPORT_ERROR,
            error_message="Test error",
            suggested_fix="Test fix",
            confidence=0.9,
            auto_fixable=True,
        )

        agent._record_fix_success(analysis)

        assert agent.success_rate["import_error"]["attempts"] == 3
        assert agent.success_rate["import_error"]["successes"] == 2

    def test_load_error_patterns(self, agent):
        patterns = agent._load_error_patterns()

        assert "import_patterns" in patterns
        assert "memory_patterns" in patterns
        assert "connection_patterns" in patterns
        assert "No module named" in patterns["import_patterns"]
        assert "out of memory" in patterns["memory_patterns"]
        assert "connection refused" in patterns["connection_patterns"]


class TestErrorAnalysis:
    def test_error_analysis_creation(self):
        analysis = ErrorAnalysis(
            error_type=ErrorType.IMPORT_ERROR,
            error_message="No module named 'pandas'",
            suggested_fix="pip install pandas",
            confidence=0.9,
            auto_fixable=True,
            fix_command="pip install pandas",
        )

        assert analysis.error_type == ErrorType.IMPORT_ERROR
        assert "pandas" in analysis.error_message
        assert analysis.confidence == 0.9
        assert analysis.auto_fixable is True
        assert analysis.fix_command == "pip install pandas"

    def test_error_analysis_with_resource_adjustment(self):
        analysis = ErrorAnalysis(
            error_type=ErrorType.MEMORY_ERROR,
            error_message="Out of memory",
            suggested_fix="Increase memory to 8Gi",
            confidence=0.8,
            auto_fixable=True,
            resource_adjustment={"memory": "8Gi"},
        )

        assert analysis.error_type == ErrorType.MEMORY_ERROR
        assert analysis.resource_adjustment == {"memory": "8Gi"}
        assert analysis.fix_command is None

    def test_error_analysis_not_auto_fixable(self):
        analysis = ErrorAnalysis(
            error_type=ErrorType.UNKNOWN,
            error_message="Unknown error",
            suggested_fix="Manual intervention required",
            confidence=0.1,
            auto_fixable=False,
        )

        assert analysis.error_type == ErrorType.UNKNOWN
        assert analysis.auto_fixable is False
        assert analysis.confidence == 0.1


class TestErrorType:
    def test_error_type_values(self):
        assert ErrorType.IMPORT_ERROR.value == "import_error"
        assert ErrorType.MEMORY_ERROR.value == "memory_error"
        assert ErrorType.CONNECTION_ERROR.value == "connection_error"
        assert ErrorType.PERMISSION_ERROR.value == "permission_error"
        assert ErrorType.RESOURCE_ERROR.value == "resource_error"
        assert ErrorType.DATA_ERROR.value == "data_error"
        assert ErrorType.UNKNOWN.value == "unknown"

    def test_error_type_enum_membership(self):
        assert ErrorType.IMPORT_ERROR in ErrorType
        assert ErrorType.MEMORY_ERROR in ErrorType
        assert ErrorType.CONNECTION_ERROR in ErrorType
        assert ErrorType.PERMISSION_ERROR in ErrorType
        assert ErrorType.RESOURCE_ERROR in ErrorType
        assert ErrorType.DATA_ERROR in ErrorType
        assert ErrorType.UNKNOWN in ErrorType


@pytest.mark.integration
class TestSelfHealingAgentIntegration:
    @pytest.fixture
    def agent(self):
        return SelfHealingAgent(enable_auto_fix=True, max_retries=2)

    def test_end_to_end_import_error_healing(self, agent):
        error = ImportError("No module named 'requests'")
        context = {"task_id": "test_task"}

        with mock.patch("subprocess.run") as mock_subprocess:
            mock_result = mock.MagicMock()
            mock_result.returncode = 0
            mock_subprocess.return_value = mock_result

            analysis = agent.analyze_error(error, context)
            result = agent.attempt_fix(analysis, context)

            assert result is True
            assert analysis.error_type == ErrorType.IMPORT_ERROR
            assert "requests" in analysis.fix_command

    def test_end_to_end_memory_error_healing(self, agent):
        error = MemoryError("Cannot allocate memory")
        context = {"memory": "2Gi", "task_id": "memory_intensive_task"}

        analysis = agent.analyze_error(error, context)
        result = agent.attempt_fix(analysis, context)

        assert result is True
        assert analysis.error_type == ErrorType.MEMORY_ERROR
        assert context["memory"] == "4Gi"

    def test_healing_with_retry_logic(self, agent):
        errors = [
            ImportError("No module named 'pandas'"),
            MemoryError("Out of memory"),
            Exception("Connection failed"),
        ]

        for error in errors:
            context = {"memory": "2Gi"}
            analysis = agent.analyze_error(error, context)

            if analysis.auto_fixable:
                with mock.patch.object(
                    agent, "_execute_fix_command", return_value=True
                ):
                    result = agent.attempt_fix(analysis, context)
                    assert result is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
