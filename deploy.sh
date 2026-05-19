#!/usr/bin/env bash
# =============================================================
# deploy.sh — Build and deploy Hallucination Watchdog to Cloud Run
# Usage: ./deploy.sh
# Prerequisites: gcloud CLI authenticated, Docker available
# =============================================================
set -euo pipefail

# --- Config (edit these) ---
PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-your-gcp-project}"
REGION="${GOOGLE_CLOUD_LOCATION:-us-central1}"
SERVICE_NAME="hallucination-watchdog"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "🔨 Building image: ${IMAGE}"
docker build --platform linux/amd64 -t "${IMAGE}" .

echo "📤 Pushing to GCR..."
docker push "${IMAGE}"

echo "🚀 Deploying to Cloud Run..."
gcloud run deploy "${SERVICE_NAME}" \
  --image "${IMAGE}" \
  --platform managed \
  --region "${REGION}" \
  --project "${PROJECT_ID}" \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 2 \
  --concurrency 80 \
  --timeout 300 \
  --set-env-vars "PHOENIX_BASE_URL=${PHOENIX_BASE_URL:-https://app.phoenix.arize.com}" \
  --set-env-vars "PHOENIX_COLLECTOR_ENDPOINT=${PHOENIX_COLLECTOR_ENDPOINT:-https://app.phoenix.arize.com}" \
  --set-env-vars "GOOGLE_GENAI_USE_VERTEXAI=True" \
  --set-env-vars "GOOGLE_CLOUD_PROJECT=${PROJECT_ID}" \
  --set-env-vars "GOOGLE_CLOUD_LOCATION=${REGION}" \
  --set-env-vars "EVAL_MODEL=gemini-2.5-flash" \
  --update-secrets "PHOENIX_API_KEY=phoenix-api-key:latest" \
  --update-secrets "GOOGLE_API_KEY=google-api-key:latest"

echo ""
echo "✅ Deployed! Service URL:"
gcloud run services describe "${SERVICE_NAME}" \
  --region "${REGION}" \
  --project "${PROJECT_ID}" \
  --format "value(status.url)"
