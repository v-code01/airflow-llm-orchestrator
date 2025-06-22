# Multi-stage build for production-ready image
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements
COPY requirements.txt .
COPY requirements-dev.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Production stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create airflow user
RUN useradd -ms /bin/bash -u 1000 airflow

# Set working directory
WORKDIR /opt/airflow

# Copy application code
COPY --chown=airflow:airflow . .

# Install the package
RUN pip install -e .

# Switch to airflow user
USER airflow

# Expose ports
EXPOSE 8080 5555 8793

# Set environment variables
ENV AIRFLOW_HOME=/opt/airflow
ENV PYTHONPATH=/opt/airflow:$PYTHONPATH

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command
CMD ["airflow", "webserver"]
