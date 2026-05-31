# Infrastructure

Terraform configuration for AWS deployment.

## Resources defined

- ECR repository for Docker images
- ECS cluster for container orchestration
- CloudWatch log group for observability
- IAM roles with least-privilege permissions

## Usage

```bash
cd infrastructure
terraform init
terraform plan -var aws_account_id=013200615679
terraform apply -var aws_account_id=013200615679
```

## Production additions (not in pilot scope)

- VPC with private subnets (replace default VPC)
- VPC peering to Acme on-prem network
- Application Load Balancer with ACM certificate
- RDS PostgreSQL instance
- Auto-scaling policies
