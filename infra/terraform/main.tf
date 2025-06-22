terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# EKS Cluster for AirflowLLM
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name    = "${var.project_name}-cluster"
  cluster_version = "1.28"

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  # Enable IRSA
  enable_irsa = true

  # Node groups configuration
  eks_managed_node_groups = {
    # General compute nodes
    general = {
      desired_size = 3
      min_size     = 2
      max_size     = 10

      instance_types = ["t3.xlarge"]

      k8s_labels = {
        Environment = var.environment
        NodeType    = "general"
      }
    }

    # GPU nodes for ML workloads
    gpu = {
      desired_size = 1
      min_size     = 0
      max_size     = 5

      instance_types = ["g4dn.xlarge"]

      k8s_labels = {
        Environment = var.environment
        NodeType    = "gpu"
      }

      taints = [{
        key    = "nvidia.com/gpu"
        value  = "true"
        effect = "NO_SCHEDULE"
      }]
    }

    # Spot instances for cost optimization
    spot = {
      desired_size = 2
      min_size     = 1
      max_size     = 20

      instance_types = ["t3.large", "t3a.large", "t2.large"]
      capacity_type  = "SPOT"

      k8s_labels = {
        Environment = var.environment
        NodeType    = "spot"
      }

      taints = [{
        key    = "spot"
        value  = "true"
        effect = "NO_SCHEDULE"
      }]
    }
  }
}

# VPC Configuration
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "${var.project_name}-vpc"
  cidr = "10.0.0.0/16"

  azs             = data.aws_availability_zones.available.names
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]

  enable_nat_gateway   = true
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    "kubernetes.io/cluster/${var.project_name}-cluster" = "shared"
  }

  public_subnet_tags = {
    "kubernetes.io/cluster/${var.project_name}-cluster" = "shared"
    "kubernetes.io/role/elb"                             = "1"
  }

  private_subnet_tags = {
    "kubernetes.io/cluster/${var.project_name}-cluster" = "shared"
    "kubernetes.io/role/internal-elb"                    = "1"
  }
}

# S3 Buckets for Airflow
resource "aws_s3_bucket" "airflow_logs" {
  bucket = "${var.project_name}-logs-${var.environment}"
}

resource "aws_s3_bucket" "airflow_dags" {
  bucket = "${var.project_name}-dags-${var.environment}"
}

resource "aws_s3_bucket_versioning" "dags_versioning" {
  bucket = aws_s3_bucket.airflow_dags.id
  versioning_configuration {
    status = "Enabled"
  }
}

# RDS for Airflow Metadata
module "rds" {
  source  = "terraform-aws-modules/rds/aws"
  version = "~> 6.0"

  identifier = "${var.project_name}-metadata"

  engine               = "postgres"
  engine_version       = "15"
  instance_class       = "db.t3.large"
  allocated_storage    = 100
  storage_encrypted    = true

  db_name  = "airflow"
  username = "airflow"
  port     = "5432"

  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.airflow.name

  backup_retention_period = 30
  backup_window          = "03:00-06:00"
  maintenance_window     = "Mon:00:00-Mon:03:00"

  enabled_cloudwatch_logs_exports = ["postgresql"]
}

# ElastiCache for Celery Backend
resource "aws_elasticache_cluster" "redis" {
  cluster_id           = "${var.project_name}-redis"
  engine              = "redis"
  node_type           = "cache.t3.medium"
  num_cache_nodes     = 1
  parameter_group_name = "default.redis7"
  port                = 6379
  subnet_group_name   = aws_elasticache_subnet_group.airflow.name
  security_group_ids  = [aws_security_group.redis.id]
}

# IAM Role for Airflow Pods
resource "aws_iam_role" "airflow_pod_role" {
  name = "${var.project_name}-pod-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRoleWithWebIdentity"
      Effect = "Allow"
      Principal = {
        Federated = module.eks.oidc_provider_arn
      }
      Condition = {
        StringEquals = {
          "${replace(module.eks.cluster_oidc_issuer_url, "https://", "")}:sub" = "system:serviceaccount:airflow-llm:airflow-llm"
        }
      }
    }]
  })
}

# IAM Policy for S3 Access
resource "aws_iam_role_policy" "airflow_s3_policy" {
  name = "${var.project_name}-s3-policy"
  role = aws_iam_role.airflow_pod_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.airflow_logs.arn,
          "${aws_s3_bucket.airflow_logs.arn}/*",
          aws_s3_bucket.airflow_dags.arn,
          "${aws_s3_bucket.airflow_dags.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "ec2:DescribeInstances",
          "ec2:DescribeSpotPriceHistory",
          "pricing:GetProducts"
        ]
        Resource = "*"
      }
    ]
  })
}

# Secrets Manager for API Keys
resource "aws_secretsmanager_secret" "api_keys" {
  name = "${var.project_name}-api-keys"
}

resource "aws_secretsmanager_secret_version" "api_keys" {
  secret_id = aws_secretsmanager_secret.api_keys.id
  secret_string = jsonencode({
    openai_api_key    = var.openai_api_key
    anthropic_api_key = var.anthropic_api_key
  })
}

# Security Groups
resource "aws_security_group" "rds" {
  name_prefix = "${var.project_name}-rds-"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [module.vpc.vpc_cidr_block]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "redis" {
  name_prefix = "${var.project_name}-redis-"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = [module.vpc.vpc_cidr_block]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# Subnet Groups
resource "aws_db_subnet_group" "airflow" {
  name       = "${var.project_name}-db-subnet"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_elasticache_subnet_group" "airflow" {
  name       = "${var.project_name}-cache-subnet"
  subnet_ids = module.vpc.private_subnets
}

# Data Sources
data "aws_availability_zones" "available" {
  state = "available"
}

# Outputs
output "cluster_endpoint" {
  value = module.eks.cluster_endpoint
}

output "cluster_name" {
  value = module.eks.cluster_name
}

output "rds_endpoint" {
  value = module.rds.db_instance_endpoint
}

output "redis_endpoint" {
  value = aws_elasticache_cluster.redis.cache_nodes[0].address
}

output "s3_logs_bucket" {
  value = aws_s3_bucket.airflow_logs.id
}

output "s3_dags_bucket" {
  value = aws_s3_bucket.airflow_dags.id
}
