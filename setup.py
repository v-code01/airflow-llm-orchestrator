#!/usr/bin/env python3
"""
AirflowLLM - AI-powered Apache Airflow DAG generation
"""

from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", encoding="utf-8") as fh:
    requirements = [
        line.strip() for line in fh if line.strip() and not line.startswith("#")
    ]

setup(
    name="airflow-llm",
    version="1.0.0",
    author="Vansh Verma",
    description="AI-powered Apache Airflow DAG generation with real code implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vanshverma/airflow-llm-orchestrator",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Code Generators",
        "Topic :: System :: Distributed Computing",
    ],
    python_requires=">=3.9,<3.13",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "airflow-llm=airflow_llm.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "airflow_llm": ["templates/*.j2", "py.typed"],
    },
)
