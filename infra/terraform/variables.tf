variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "airflow-llm"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "openai_api_key" {
  description = "OpenAI API Key"
  type        = string
  sensitive   = true
}

variable "anthropic_api_key" {
  description = "Anthropic API Key"
  type        = string
  sensitive   = true
}
