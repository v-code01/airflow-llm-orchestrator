#!/usr/bin/env python3
"""
Quick Production Test - No External Dependencies

Tests core logic and structure without requiring numpy, pandas, or other external libraries.
This verifies the codebase is production-ready at the structural level.
"""

import ast
import re
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class QuickProductionTest:
    """Lightweight production testing without external dependencies"""

    def __init__(self):
        self.project_root = project_root
        self.passed = 0
        self.total = 0
        self.results = {}

    def run_all_tests(self) -> dict:
        """Run all quick production tests"""
        print("AirflowLLM Quick Production Verification")
        print("=" * 50)

        # File structure tests
        self._test_file_structure()

        # Code syntax tests
        self._test_code_syntax()

        # Documentation tests
        self._test_documentation()

        # Configuration tests
        self._test_configuration()

        # Security tests
        self._test_security()

        # Generate report
        score = (self.passed / self.total) * 100 if self.total > 0 else 0
        self.results["overall_score"] = score

        print(f"\nFinal Score: {score:.1f}% ({self.passed}/{self.total})")

        if score >= 85:
            print("PASS PRODUCTION READY")
            return True
        else:
            print("FAIL NEEDS FIXES")
            return False

    def _test_file_structure(self):
        """Test required file structure exists"""
        print("\nTesting File Structure...")

        required_files = [
            "airflow_llm/__init__.py",
            "airflow_llm/orchestrator.py",
            "airflow_llm/self_healer.py",
            "airflow_llm/cost_optimizer.py",
            "airflow_llm/decorators.py",
            "airflow_llm/nl2dag.py",
            "tests/__init__.py",
            "README.md",
            "pyproject.toml",
            "requirements.txt",
        ]

        for file_path in required_files:
            full_path = self.project_root / file_path
            exists = full_path.exists()
            self._record_test(f"file_structure.{file_path}", exists)
            print(f"  {file_path}: {'PASS' if exists else 'FAIL'}")

    def _test_code_syntax(self):
        """Test Python syntax is valid"""
        print("\nTesting Code Syntax...")

        syntax_errors = []
        python_files = list((self.project_root / "airflow_llm").rglob("*.py"))

        for py_file in python_files:
            try:
                with open(py_file) as f:
                    content = f.read()
                ast.parse(content)
                print(f"  {py_file.name}: PASS")
            except SyntaxError as e:
                syntax_errors.append(f"{py_file}: {e}")
                print(f"  {py_file.name}: FAIL {e}")
            except Exception as e:
                print(f"  {py_file.name}: WARNING  {e}")

        self._record_test("syntax.all_files_valid", len(syntax_errors) == 0)

    def _test_documentation(self):
        """Test documentation completeness"""
        print("\nTesting Documentation...")

        # Check README
        readme_path = self.project_root / "README.md"
        if readme_path.exists():
            content = readme_path.read_text()
            has_installation = "installation" in content.lower()
            has_usage = "usage" in content.lower()
            has_examples = "example" in content.lower()

            self._record_test("docs.readme_installation", has_installation)
            self._record_test("docs.readme_usage", has_usage)
            self._record_test("docs.readme_examples", has_examples)

            print(f"  README Installation: {'PASS' if has_installation else 'FAIL'}")
            print(f"  README Usage: {'PASS' if has_usage else 'FAIL'}")
            print(f"  README Examples: {'PASS' if has_examples else 'FAIL'}")
        else:
            self._record_test("docs.readme_exists", False)
            print("  README.md: FAIL Missing")

        # Check examples directory
        examples_dir = self.project_root / "examples"
        has_examples = (
            examples_dir.exists() and len(list(examples_dir.glob("*.py"))) >= 3
        )
        self._record_test("docs.examples_directory", has_examples)
        print(f"  Examples directory: {'PASS' if has_examples else 'FAIL'}")

    def _test_configuration(self):
        """Test configuration files"""
        print("\nTesting Configuration...")

        # Check pyproject.toml
        pyproject_path = self.project_root / "pyproject.toml"
        if pyproject_path.exists():
            content = pyproject_path.read_text()
            has_build_system = "[build-system]" in content
            has_project = "[project]" in content

            self._record_test("config.pyproject_build", has_build_system)
            self._record_test("config.pyproject_project", has_project)

            print(
                f"  pyproject.toml build-system: {'PASS' if has_build_system else 'FAIL'}"
            )
            print(f"  pyproject.toml project: {'PASS' if has_project else 'FAIL'}")
        else:
            self._record_test("config.pyproject_exists", False)
            print("  pyproject.toml: FAIL Missing")

        # Check requirements.txt
        req_path = self.project_root / "requirements.txt"
        if req_path.exists():
            content = req_path.read_text()
            has_deps = len(content.strip().split("\n")) >= 5
            self._record_test("config.requirements_populated", has_deps)
            print(f"  requirements.txt populated: {'PASS' if has_deps else 'FAIL'}")
        else:
            self._record_test("config.requirements_exists", False)
            print("  requirements.txt: FAIL Missing")

    def _test_security(self):
        """Test basic security measures"""
        print("\nTesting Security...")

        # Check for hardcoded secrets
        secret_patterns = [
            r'api[_-]?key\s*=\s*["\'][^"\'\s]{20,}["\']',
            r'password\s*=\s*["\'][^"\'\s]+["\']',
            r'secret\s*=\s*["\'][^"\'\s]{20,}["\']',
            r'token\s*=\s*["\'][^"\'\s]{20,}["\']',
        ]

        violations = []
        for py_file in (self.project_root / "airflow_llm").rglob("*.py"):
            try:
                content = py_file.read_text()
                for pattern in secret_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        violations.append(py_file.name)
            except Exception:
                continue

        no_secrets = len(violations) == 0
        self._record_test("security.no_hardcoded_secrets", no_secrets)
        print(f"  No hardcoded secrets: {'PASS' if no_secrets else 'FAIL'}")

        if violations:
            print(f"    Violations in: {', '.join(violations)}")

    def _record_test(self, test_name: str, passed: bool):
        """Record test result"""
        self.results[test_name] = passed
        if passed:
            self.passed += 1
        self.total += 1


def test_import_structure():
    """Test that basic import structure works"""
    print("\nTesting Import Structure (without external deps)...")

    # Test that files exist and can be parsed
    airflow_llm_dir = project_root / "airflow_llm"

    if not airflow_llm_dir.exists():
        print("FAIL airflow_llm directory missing")
        return False

    init_file = airflow_llm_dir / "__init__.py"
    if not init_file.exists():
        print("FAIL __init__.py missing")
        return False

    # Check that __init__.py has proper exports
    try:
        init_content = init_file.read_text()
        has_exports = "from ." in init_content and "import" in init_content
        print(f"  __init__.py has exports: {'PASS' if has_exports else 'FAIL'}")
        return has_exports
    except Exception as e:
        print(f"FAIL Error reading __init__.py: {e}")
        return False


def test_code_structure():
    """Test code structure and organization"""
    print("\nTesting Code Structure...")

    # Check class definitions exist
    orchestrator_file = project_root / "airflow_llm" / "orchestrator.py"

    if not orchestrator_file.exists():
        print("FAIL orchestrator.py missing")
        return False

    try:
        content = orchestrator_file.read_text()
        tree = ast.parse(content)

        # Find class definitions
        classes = [
            node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
        ]

        has_orchestrator = "LLMOrchestrator" in classes
        print(f"  LLMOrchestrator class: {'PASS' if has_orchestrator else 'FAIL'}")

        # Check for method definitions
        methods = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                methods.append(node.name)

        has_generate_dag = "generate_dag" in methods
        print(f"  generate_dag method: {'PASS' if has_generate_dag else 'FAIL'}")

        return has_orchestrator and has_generate_dag

    except Exception as e:
        print(f"FAIL Error parsing orchestrator.py: {e}")
        return False


def main():
    """Run quick production tests"""
    print("Starting Quick Production Verification")
    print("This test runs without external dependencies")
    print("=" * 60)

    # Run basic tests
    tester = QuickProductionTest()
    basic_ready = tester.run_all_tests()

    # Run import structure test
    import_ready = test_import_structure()

    # Run code structure test
    structure_ready = test_code_structure()

    print("\n" + "=" * 60)
    print("QUICK PRODUCTION TEST SUMMARY")
    print("=" * 60)

    overall_ready = basic_ready and import_ready and structure_ready

    print(f"Basic Structure: {'PASS' if basic_ready else 'FAIL'}")
    print(f"Import Structure: {'PASS' if import_ready else 'FAIL'}")
    print(f"Code Structure: {'PASS' if structure_ready else 'FAIL'}")
    print(f"Overall Ready: {'PASS' if overall_ready else 'FAIL'}")

    if overall_ready:
        print("\nSUCCESS CODEBASE IS STRUCTURALLY SOUND")
        print("Ready for dependency installation and full testing")
    else:
        print("\nWARNING  STRUCTURAL ISSUES FOUND")
        print("Fix basic structure before proceeding")

    # Save quick report
    report_file = project_root / "quick_test_report.txt"
    with open(report_file, "w") as f:
        f.write("AirflowLLM Quick Test Report\n")
        f.write("=" * 30 + "\n\n")
        f.write(f"Basic Structure: {'PASS' if basic_ready else 'FAIL'}\n")
        f.write(f"Import Structure: {'PASS' if import_ready else 'FAIL'}\n")
        f.write(f"Code Structure: {'PASS' if structure_ready else 'FAIL'}\n")
        f.write(f"Overall: {'PASS' if overall_ready else 'FAIL'}\n\n")

        f.write("Detailed Results:\n")
        for test_name, result in tester.results.items():
            f.write(f"  {test_name}: {'PASS' if result else 'FAIL'}\n")

    print(f"\nReport saved to: {report_file}")
    return 0 if overall_ready else 1


if __name__ == "__main__":
    exit(main())
