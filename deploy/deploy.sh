#!/usr/bin/env bash
set -euo pipefail

AWS_REGION="${AWS_REGION:-us-east-1}"
ECR_REPO="${ECR_REPO:-voice-agent-demo}"
ECS_CLUSTER="${ECS_CLUSTER:-voice-agent-demo}"
ECS_SERVICE="${ECS_SERVICE:-voice-agent-demo}"
TASK_FAMILY="${TASK_FAMILY:-voice-agent-demo}"
COMMIT_HASH="$(git rev-parse HEAD)"
AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}"

echo "${COMMIT_HASH}" > VERSION

aws ecr describe-repositories --repository-names "${ECR_REPO}" --region "${AWS_REGION}" >/dev/null 2>&1 || \
  aws ecr create-repository --repository-name "${ECR_REPO}" --region "${AWS_REGION}" >/dev/null

aws ecr get-login-password --region "${AWS_REGION}" | docker login --username AWS --password-stdin "${ECR_URI}"

docker build -t "${ECR_REPO}:${COMMIT_HASH}" .
docker tag "${ECR_REPO}:${COMMIT_HASH}" "${ECR_URI}:${COMMIT_HASH}"
docker tag "${ECR_REPO}:${COMMIT_HASH}" "${ECR_URI}:latest"
docker push "${ECR_URI}:${COMMIT_HASH}"
docker push "${ECR_URI}:latest"

TMP_TASKDEF="$(mktemp)"
sed \
  -e "s/<AWS_ACCOUNT_ID>/${AWS_ACCOUNT_ID}/g" \
  -e "s|voice-agent-demo:latest|${ECR_REPO}:${COMMIT_HASH}|g" \
  deploy/taskdef.json > "${TMP_TASKDEF}"

TASK_DEF_ARN="$(aws ecs register-task-definition --cli-input-json "file://${TMP_TASKDEF}" --region "${AWS_REGION}" --query 'taskDefinition.taskDefinitionArn' --output text)"

aws ecs update-service \
  --cluster "${ECS_CLUSTER}" \
  --service "${ECS_SERVICE}" \
  --task-definition "${TASK_DEF_ARN}" \
  --force-new-deployment \
  --region "${AWS_REGION}" >/dev/null

echo "Deployed ${ECS_SERVICE} with image ${ECR_URI}:${COMMIT_HASH}"
