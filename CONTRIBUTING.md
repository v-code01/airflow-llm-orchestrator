# Contributing to AirflowLLM

Thank you for your interest in contributing to AirflowLLM! We're building the future of data pipeline development and your contributions make a difference.

## How to Contribute

### 🚀 Quick Start for Contributors

1. **Fork and Clone**

   ```bash
   git clone https://github.com/your-username/airflow-llm-orchestrator.git
   cd airflow-llm-orchestrator
   ```

2. **Setup Development Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev]"
   ```

3. **Run Tests**

   ```bash
   pytest tests/ -v
   python simple_test.py  # Core functionality test
   ```

4. **Make Your Changes**

   - Follow our coding standards (see below)
   - Add tests for new features
   - Update documentation as needed

5. **Submit Pull Request**
   - Create a descriptive PR title
   - Include tests and documentation
   - Link to any related issues

## Areas We Need Help

### 🎯 High Priority

- **Database Connectors**: MySQL, Oracle, SQL Server integrations
- **Cloud Providers**: Azure, GCP enhanced support
- **Performance**: Model optimization and caching
- **Documentation**: More examples and tutorials

### 🔧 Medium Priority

- **Testing**: Integration tests with real Airflow instances
- **CLI Enhancements**: Better error messages and debugging
- **Templates**: Industry-specific DAG templates
- **Monitoring**: Enhanced observability features

### 📚 Always Welcome

- **Bug Fixes**: Check our [Issues](https://github.com/v-code01/airflow-llm-orchestrator/issues)
- **Documentation**: Improve guides, add examples
- **Examples**: Real-world DAG templates
- **Community**: Help others in Discord/GitHub discussions

## Development Guidelines

### Code Quality Standards

- **Type Hints**: All functions must have type annotations
- **Documentation**: Docstrings for all public functions/classes
- **Testing**: Unit tests for new features (aim for 80%+ coverage)
- **Formatting**: Use Black for code formatting
- **Linting**: Pass flake8 and pylint checks

### Example Contribution

```python
def generate_sql_query(
    table_name: str,
    columns: List[str],
    where_clause: Optional[str] = None
) -> str:
    """Generate optimized SQL query with parameterization.

    Args:
        table_name: Name of the table to query
        columns: List of column names to select
        where_clause: Optional WHERE clause

    Returns:
        Formatted SQL query string

    Example:
        >>> generate_sql_query("users", ["id", "name"], "active = true")
        'SELECT id, name FROM users WHERE active = true'
    """
    # Implementation here
    pass
```

### Commit Message Format

```
type(scope): Brief description

Longer description if needed

- Changes made
- Why this change was necessary
- Any breaking changes

Closes #123
```

**Types**: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`

## Testing Guidelines

### Running Tests

```bash
# Unit tests
pytest tests/ -v

# Core functionality
python simple_test.py

# Specific test file
pytest tests/test_dag_factory.py -v

# With coverage
pytest tests/ --cov=airflow_llm --cov-report=html
```

### Writing Tests

- **Unit Tests**: Test individual functions/classes
- **Integration Tests**: Test component interactions
- **E2E Tests**: Test complete workflows
- **Performance Tests**: Benchmark critical paths

### Test Structure

```python
def test_feature_name():
    """Test description of what this validates."""
    # Arrange
    input_data = create_test_data()

    # Act
    result = function_under_test(input_data)

    # Assert
    assert result.is_valid()
    assert len(result.items) == expected_count
```

## Documentation

### Adding Documentation

- **Docstrings**: All public APIs need comprehensive docstrings
- **README Updates**: Keep README.md current with new features
- **Examples**: Add real-world usage examples
- **Tutorials**: Step-by-step guides for common use cases

### Documentation Standards

- **Clear Examples**: Show input and expected output
- **Error Handling**: Document error conditions and responses
- **Performance Notes**: Include performance characteristics
- **Version Info**: Note when features were added/changed

## Community

### Getting Help

- **Discord**: Join our [Discord server](https://discord.gg/airflow-llm) for real-time help
- **GitHub Discussions**: For longer-form discussions and Q&A
- **Issues**: For bug reports and feature requests

### Code of Conduct

- **Be Respectful**: Treat all community members with respect
- **Be Constructive**: Provide helpful feedback and suggestions
- **Be Inclusive**: Welcome contributors of all backgrounds
- **Be Professional**: Maintain professional communication

## Recognition

### Contributor Benefits

- **Credits**: Listed in CONTRIBUTORS.md and release notes
- **Swag**: AirflowLLM stickers and t-shirts for significant contributions
- **Enterprise**: Priority support and early access to enterprise features
- **Network**: Connect with other contributors and the core team

### Contribution Levels

- **🌟 First-time**: Made your first merged PR
- **🚀 Regular**: 5+ merged PRs
- **💎 Core**: 20+ merged PRs or major feature contributions
- **🏆 Maintainer**: Consistent high-quality contributions and community help

## Release Process

### Versioning

We follow [Semantic Versioning](https://semver.org/):

- **Major**: Breaking changes (1.0.0 → 2.0.0)
- **Minor**: New features, backward compatible (1.0.0 → 1.1.0)
- **Patch**: Bug fixes, backward compatible (1.0.0 → 1.0.1)

### Release Cycle

- **Patch Releases**: Weekly for bug fixes
- **Minor Releases**: Monthly for new features
- **Major Releases**: Quarterly for significant changes

## Getting Started Checklist

- [ ] Read this contributing guide
- [ ] Join our [Discord community](https://discord.gg/airflow-llm)
- [ ] Look through [good first issues](https://github.com/v-code01/airflow-llm-orchestrator/labels/good%20first%20issue)
- [ ] Set up development environment
- [ ] Run tests to ensure everything works
- [ ] Pick an issue or feature to work on
- [ ] Submit your first PR

## Questions?

Don't hesitate to ask! We're here to help:

- **Discord**: [discord.gg/airflow-llm](https://discord.gg/airflow-llm)
- **Email**: contributors@airflow-llm.dev
- **GitHub**: Create an issue with the "question" label

Thank you for contributing to AirflowLLM! Together, we're making data pipeline development 10x faster and more enjoyable.

---

**Happy Coding!** 🚀
