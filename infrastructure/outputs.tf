output "ecr_repository_url" {
  value = aws_ecr_repository.voice_agent.repository_url
}

output "ecs_cluster_name" {
  value = aws_ecs_cluster.voice_agent.name
}

output "execution_role_arn" {
  value = aws_iam_role.ecs_execution_role.arn
}

output "task_role_arn" {
  value = aws_iam_role.ecs_task_role.arn
}
