"""Test suite for CLI functionality."""

import sys
from unittest.mock import patch

import pytest

from airflow_llm.cli import main


class TestCLI:
    """Test CLI functionality."""

    def test_cli_import(self):
        """Test that CLI module imports successfully."""
        from airflow_llm.cli import main

        assert callable(main)

    def test_argument_parsing(self):
        """Test CLI argument parsing."""
        # Test that the CLI can parse basic arguments
        with patch.object(sys, "argv", ["airflow-llm", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            # Help command should exit with code 0
            assert exc_info.value.code == 0

    def test_generate_command_structure(self):
        """Test generate command structure."""
        from airflow_llm.cli import generate_dag_command

        assert callable(generate_dag_command)

    def test_create_command_structure(self):
        """Test create command structure."""
        from airflow_llm.cli import create_dag_command

        assert callable(create_dag_command)

    def test_validate_command_structure(self):
        """Test validate command structure."""
        from airflow_llm.cli import validate_dag_command

        assert callable(validate_dag_command)

    def test_list_templates_command_structure(self):
        """Test list templates command structure."""
        from airflow_llm.cli import list_templates_command

        assert callable(list_templates_command)
