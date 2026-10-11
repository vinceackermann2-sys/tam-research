#!/usr/bin/env bash
# Run once in the project owner's Cloud Shell after reviewing these permissions.
set -euo pipefail
PROJECT_ID=project-c40afe76-9188-4739-b66
REPOSITORY=vinceackermann2-sys/tam-research
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')
LAUNCHER="tam-github-launcher@$PROJECT_ID.iam.gserviceaccount.com"
RUNTIME="tam-batch-runtime@$PROJECT_ID.iam.gserviceaccount.com"
POOL=tam-github
PROVIDER=github

gcloud services enable batch.googleapis.com compute.googleapis.com logging.googleapis.com iam.googleapis.com iamcredentials.googleapis.com sts.googleapis.com cloudresourcemanager.googleapis.com --project="$PROJECT_ID"
for name in tam-github-launcher tam-batch-runtime; do
  if ! gcloud iam service-accounts describe "$name@$PROJECT_ID.iam.gserviceaccount.com" --project="$PROJECT_ID" >/dev/null 2>&1; then
    gcloud iam service-accounts create "$name" --project="$PROJECT_ID" --display-name="$name"
  fi
done
if ! gcloud iam workload-identity-pools describe "$POOL" --location=global --project="$PROJECT_ID" >/dev/null 2>&1; then
  gcloud iam workload-identity-pools create "$POOL" --location=global --project="$PROJECT_ID" --display-name='TAM GitHub Actions'
fi
MAPPING='google.subject=assertion.sub,attribute.repository_id=assertion.repository_id,attribute.repository_owner_id=assertion.repository_owner_id,attribute.ref=assertion.ref,attribute.workflow_ref=assertion.workflow_ref,attribute.actor_id=assertion.actor_id,attribute.event_name=assertion.event_name'
CONDITION="assertion.repository_id=='1341522112' && assertion.repository_owner_id=='232194416' && assertion.actor_id=='232194416' && assertion.ref=='refs/heads/main' && assertion.workflow_ref=='$REPOSITORY/.github/workflows/google-cloud-compute.yml@refs/heads/main' && assertion.event_name in ['issues','workflow_dispatch']"
if gcloud iam workload-identity-pools providers describe "$PROVIDER" --workload-identity-pool="$POOL" --location=global --project="$PROJECT_ID" >/dev/null 2>&1; then
  gcloud iam workload-identity-pools providers update-oidc "$PROVIDER" --workload-identity-pool="$POOL" --location=global --project="$PROJECT_ID" --attribute-mapping="$MAPPING" --attribute-condition="$CONDITION"
else
  gcloud iam workload-identity-pools providers create-oidc "$PROVIDER" --workload-identity-pool="$POOL" --location=global --project="$PROJECT_ID" --issuer-uri='https://token.actions.githubusercontent.com' --attribute-mapping="$MAPPING" --attribute-condition="$CONDITION"
fi
gcloud iam service-accounts add-iam-policy-binding "$LAUNCHER" --project="$PROJECT_ID" --role=roles/iam.workloadIdentityUser --member="principalSet://iam.googleapis.com/projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/$POOL/attribute.repository_id/1341522112"
for role in roles/batch.jobsEditor roles/logging.viewer; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" --member="serviceAccount:$LAUNCHER" --role="$role" --condition=None
done
gcloud iam service-accounts add-iam-policy-binding "$RUNTIME" --project="$PROJECT_ID" --member="serviceAccount:$LAUNCHER" --role=roles/iam.serviceAccountUser
for role in roles/batch.agentReporter roles/logging.logWriter; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" --member="serviceAccount:$RUNTIME" --role="$role" --condition=None
done

# These identifiers are public configuration, not credentials.
printf '\nGCP_PROJECT_ID=%s\nGCP_REGION=us-central1\nGCP_WORKLOAD_IDENTITY_PROVIDER=projects/%s/locations/global/workloadIdentityPools/%s/providers/%s\nGCP_LAUNCHER_ACCOUNT=%s\nGCP_RUNTIME_ACCOUNT=%s\n' "$PROJECT_ID" "$PROJECT_NUMBER" "$POOL" "$PROVIDER" "$LAUNCHER" "$RUNTIME"
