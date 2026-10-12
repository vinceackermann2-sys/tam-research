#!/usr/bin/env bash
# Run once in authorized Cloud Shell; no VM is created by this script.
set -Eeuo pipefail
PROJECT=project-c40afe76-9188-4739-b66
NUMBER=1009754963025
REPO=vinceackermann2-sys/tam-research
LAUNCHER=tam-gpu-launcher@$PROJECT.iam.gserviceaccount.com
WORKER=tam-gpu-worker@$PROJECT.iam.gserviceaccount.com
BUCKET=tam-research-compute-$NUMBER
POOL=tam-research-github
PROVIDER=github-main

gcloud services enable compute.googleapis.com iam.googleapis.com iamcredentials.googleapis.com sts.googleapis.com storage.googleapis.com --project="$PROJECT"
for name in tam-gpu-launcher tam-gpu-worker; do
  gcloud iam service-accounts describe "$name@$PROJECT.iam.gserviceaccount.com" --project="$PROJECT" >/dev/null 2>&1 || \
    gcloud iam service-accounts create "$name" --project="$PROJECT" --display-name="$name"
done
gcloud compute networks describe tam-research-compute --project="$PROJECT" >/dev/null 2>&1 || \
  gcloud compute networks create tam-research-compute --subnet-mode=custom --project="$PROJECT"
gcloud compute networks subnets describe tam-research-compute --region=us-central1 --project="$PROJECT" >/dev/null 2>&1 || \
  gcloud compute networks subnets create tam-research-compute --network=tam-research-compute --range=10.83.0.0/24 --region=us-central1 --project="$PROJECT"
# No ingress firewall rule: SSH and public application ports remain closed.
gcloud storage buckets describe "gs://$BUCKET" >/dev/null 2>&1 || \
  gcloud storage buckets create "gs://$BUCKET" --project="$PROJECT" --location=us-central1 --uniform-bucket-level-access --public-access-prevention
cat > /tmp/tam-gcp-lifecycle.json <<'JSON'
{"rule":[{"action":{"type":"Delete"},"condition":{"age":7,"matchesPrefix":["runs/"]}}]}
JSON
gcloud storage buckets update "gs://$BUCKET" --lifecycle-file=/tmp/tam-gcp-lifecycle.json
# Budget reservation objects never expire and cannot be removed by launcher or worker.
for account in "$LAUNCHER" "$WORKER"; do
  gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" --member="serviceAccount:$account" --role=roles/storage.objectViewer
done
gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" --member="serviceAccount:$LAUNCHER" --role=roles/storage.objectCreator
gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" --member="serviceAccount:$WORKER" --role=roles/storage.objectCreator \
  --condition="expression=resource.name.startsWith('projects/_/buckets/$BUCKET/objects/runs/'),title=write-results-only"
PERMISSIONS=compute.instances.create,compute.instances.get,compute.instances.list,compute.instances.delete,compute.instances.setMetadata,compute.instances.setServiceAccount,compute.instances.setLabels,compute.disks.create,compute.disks.use,compute.disks.setLabels,compute.images.useReadOnly,compute.networks.use,compute.subnetworks.use,compute.subnetworks.useExternalIp,compute.zoneOperations.get,compute.globalOperations.get,compute.zones.get,compute.machineTypes.get,compute.projects.get,compute.regions.get,compute.acceleratorTypes.get
gcloud iam roles describe tamGpuLauncher --project="$PROJECT" >/dev/null 2>&1 || \
  gcloud iam roles create tamGpuLauncher --project="$PROJECT" --title='TAM bounded GPU launcher' --permissions="$PERMISSIONS" --stage=GA
gcloud projects add-iam-policy-binding "$PROJECT" --member="serviceAccount:$LAUNCHER" --role="projects/$PROJECT/roles/tamGpuLauncher"
gcloud iam service-accounts add-iam-policy-binding "$WORKER" --project="$PROJECT" --member="serviceAccount:$LAUNCHER" --role=roles/iam.serviceAccountUser
gcloud iam workload-identity-pools describe "$POOL" --project="$PROJECT" --location=global >/dev/null 2>&1 || \
  gcloud iam workload-identity-pools create "$POOL" --project="$PROJECT" --location=global --display-name='TAM GitHub Actions'
gcloud iam workload-identity-pools providers describe "$PROVIDER" --project="$PROJECT" --location=global --workload-identity-pool="$POOL" >/dev/null 2>&1 || \
  gcloud iam workload-identity-pools providers create-oidc "$PROVIDER" --project="$PROJECT" --location=global --workload-identity-pool="$POOL" \
    --issuer-uri=https://token.actions.githubusercontent.com \
    --attribute-mapping='google.subject=assertion.sub,attribute.repository=assertion.repository' \
    --attribute-condition="assertion.repository=='$REPO' && assertion.ref=='refs/heads/main' && assertion.workflow_ref=='$REPO/.github/workflows/gcp-gpu.yml@refs/heads/main'"
gcloud iam service-accounts add-iam-policy-binding "$LAUNCHER" --project="$PROJECT" --role=roles/iam.workloadIdentityUser \
  --member="principalSet://iam.googleapis.com/projects/$NUMBER/locations/global/workloadIdentityPools/$POOL/attribute.repository/$REPO"
printf '\nGitHub configuration (identifiers only, no private key):\n'
printf 'GCP_PROJECT_ID=%s\nGCP_SERVICE_ACCOUNT=%s\nGCP_WORKER_SERVICE_ACCOUNT=%s\nGCP_RESULTS_BUCKET=%s\nGCP_WORKLOAD_IDENTITY_PROVIDER=projects/%s/locations/global/workloadIdentityPools/%s/providers/%s\n' "$PROJECT" "$LAUNCHER" "$WORKER" "$BUCKET" "$NUMBER" "$POOL" "$PROVIDER"
printf 'Leave GCP_GPU_ENABLED=false until billing, quota and budget checks are verified.\n'
