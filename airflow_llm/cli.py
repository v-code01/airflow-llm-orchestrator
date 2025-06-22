#!/usr/bin/env python3
"""
AirflowLLM Command Line Interface
Production-grade DAG generation from terminal
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml

from .dag_factory import (
    DAGConfig,
    EnterpriseDAGFactory,
    NaturalLanguageDAGGenerator,
    TaskConfig,
)


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("airflow_llm.log"),
        ],
    )


def generate_dag_command(args):
    """Generate DAG from natural language description"""

    # Initialize components
    output_dir = args.output_dir or os.getcwd()
    dag_factory = EnterpriseDAGFactory(output_dir)
    nl_generator = NaturalLanguageDAGGenerator(dag_factory)

    # Parse environment configs if provided
    env_configs = {}
    if args.config_file:
        with open(args.config_file) as f:
            if args.config_file.endswith(".json"):
                env_configs = json.load(f)
            else:
                env_configs = yaml.safe_load(f)

    # Generate DAG
    try:
        dag_path = nl_generator.generate_from_description(
            description=args.description,
            dag_id=args.dag_id,
            owners=args.owners.split(",") if args.owners else None,
            stakeholders=args.stakeholders.split(",") if args.stakeholders else None,
            environment_configs=env_configs,
        )

        print(f"✅ DAG generated successfully!")
        print(f"📁 Location: {dag_path}")
        print(f"📋 Resources: {Path(dag_path).parent / 'resources' / args.dag_id}")

        # Print usage instructions
        print("\n🚀 Next steps:")
        print(f"1. Copy generated files to your Airflow dags/ directory")
        print(f"2. Update SQL scripts in resources/{args.dag_id}/ folder")
        print(f"3. Configure environment variables in config.yml")
        print(f"4. Test the DAG: airflow dags test {args.dag_id}")

    except Exception as e:
        print(f"❌ DAG generation failed: {e}")
        logging.error(f"DAG generation error", exc_info=True)
        sys.exit(1)


def create_dag_command(args):
    """Create DAG from explicit configuration"""

    # Load configuration from file
    if not os.path.exists(args.config_file):
        print(f"❌ Configuration file not found: {args.config_file}")
        sys.exit(1)

    with open(args.config_file) as f:
        if args.config_file.endswith(".json"):
            config_data = json.load(f)
        else:
            config_data = yaml.safe_load(f)

    # Initialize DAG factory
    output_dir = args.output_dir or os.getcwd()
    dag_factory = EnterpriseDAGFactory(output_dir)

    # Create DAG config
    dag_config = DAGConfig(
        dag_id=config_data["dag_id"],
        description=config_data["description"],
        schedule_interval=config_data.get("schedule_interval", "@daily"),
        max_active_runs=config_data.get("max_active_runs", 1),
        owners=config_data.get("owners", []),
        stakeholders=config_data.get("stakeholders", []),
        environment_configs=config_data.get("environment_configs", {}),
    )

    # Create task configs
    tasks = []
    for task_data in config_data.get("tasks", []):
        task = TaskConfig(
            name=task_data["name"],
            operator=task_data["operator"],
            depends_on=task_data.get("depends_on", []),
            sql_script=task_data.get("sql_script"),
            python_callable=task_data.get("python_callable"),
            connection_id=task_data.get("connection_id"),
            parameters=task_data.get("parameters"),
            resources=task_data.get("resources"),
        )
        tasks.append(task)

    # Generate DAG
    try:
        dag_path = dag_factory.generate_dag(
            description=config_data["description"],
            dag_config=dag_config,
            tasks=tasks,
            sql_scripts=config_data.get("sql_scripts", {}),
            python_functions=config_data.get("python_functions", {}),
        )

        print(f"✅ DAG created successfully!")
        print(f"📁 Location: {dag_path}")

    except Exception as e:
        print(f"❌ DAG creation failed: {e}")
        logging.error(f"DAG creation error", exc_info=True)
        sys.exit(1)


def validate_dag_command(args):
    """Validate generated DAG structure"""

    dag_file = Path(args.dag_file)
    if not dag_file.exists():
        print(f"❌ DAG file not found: {dag_file}")
        sys.exit(1)

    # Extract DAG ID from filename
    dag_id = dag_file.stem

    # Check for required files
    base_dir = dag_file.parent
    resources_dir = base_dir / "resources" / dag_id

    required_files = [resources_dir / "config.yml", resources_dir / "tasks.yml"]

    validation_errors = []

    # Check file existence
    for required_file in required_files:
        if not required_file.exists():
            validation_errors.append(f"Missing required file: {required_file}")

    # Validate YAML syntax
    for yaml_file in [f for f in required_files if f.exists() and f.suffix == ".yml"]:
        try:
            with open(yaml_file) as f:
                yaml.safe_load(f)
        except yaml.YAMLError as e:
            validation_errors.append(f"Invalid YAML in {yaml_file}: {e}")

    # Validate Python syntax of DAG file
    try:
        with open(dag_file) as f:
            compile(f.read(), str(dag_file), "exec")
    except SyntaxError as e:
        validation_errors.append(f"Python syntax error in {dag_file}: {e}")

    # Report results
    if validation_errors:
        print(f"❌ DAG validation failed:")
        for error in validation_errors:
            print(f"   • {error}")
        sys.exit(1)
    else:
        print(f"✅ DAG validation passed!")
        print(f"📁 DAG: {dag_file}")
        print(f"📋 Resources: {resources_dir}")


def list_templates_command(args):
    """List available DAG templates"""

    templates_dir = Path(__file__).parent / "templates"

    print("📋 Available DAG templates:")
    print()

    if templates_dir.exists():
        for template_file in templates_dir.glob("*.j2"):
            template_name = template_file.stem
            print(f"   • {template_name}")

            # Try to read template description
            try:
                with open(template_file) as f:
                    first_line = f.readline().strip()
                    if first_line.startswith("{#"):
                        description = first_line[2:-2].strip()
                        print(f"     {description}")
            except:
                pass
            print()
    else:
        print("   No templates found")


def main():
    """Main CLI entry point"""

    parser = argparse.ArgumentParser(
        description="AirflowLLM - AI-Powered DAG Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate DAG from natural language
  airflow-llm generate "Extract sales data from PostgreSQL, clean it, and load to S3"

  # Generate with specific configuration
  airflow-llm generate "ML pipeline for customer churn" --dag-id customer_churn --owners "data-team"

  # Create DAG from configuration file
  airflow-llm create --config dag_config.yml

  # Validate generated DAG
  airflow-llm validate my_generated_dag.py

  # List available templates
  airflow-llm templates
        """,
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        help="Output directory for generated files (default: current directory)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Generate command
    generate_parser = subparsers.add_parser(
        "generate", help="Generate DAG from natural language description"
    )
    generate_parser.add_argument(
        "description", type=str, help="Natural language description of the pipeline"
    )
    generate_parser.add_argument(
        "--dag-id", type=str, help="Custom DAG ID (auto-generated if not provided)"
    )
    generate_parser.add_argument(
        "--owners", type=str, help="Comma-separated list of DAG owners"
    )
    generate_parser.add_argument(
        "--stakeholders", type=str, help="Comma-separated list of stakeholders"
    )
    generate_parser.add_argument(
        "--config-file", type=str, help="YAML/JSON file with environment configurations"
    )
    generate_parser.set_defaults(func=generate_dag_command)

    # Create command
    create_parser = subparsers.add_parser(
        "create", help="Create DAG from explicit configuration file"
    )
    create_parser.add_argument(
        "--config-file", type=str, required=True, help="YAML/JSON configuration file"
    )
    create_parser.set_defaults(func=create_dag_command)

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate", help="Validate generated DAG structure"
    )
    validate_parser.add_argument(
        "dag_file", type=str, help="Path to DAG file to validate"
    )
    validate_parser.set_defaults(func=validate_dag_command)

    # Templates command
    templates_parser = subparsers.add_parser(
        "templates", help="List available DAG templates"
    )
    templates_parser.set_defaults(func=list_templates_command)

    # Parse arguments
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Execute command
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
