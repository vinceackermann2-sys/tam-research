from __future__ import annotations
import json
import modal

app = modal.App("tam-research-cpw-v2-secondary-h100-probe-1255")
image = modal.Image.debian_slim(python_version="3.11")

@app.function(image=image, gpu="H100!", cpu=0.125, memory=128, timeout=120, retries=0)
def probe() -> dict:
    return {"h100_admission_ok": True, "writes_performed": False, "scientific_seed_consumed": False}

@app.local_entrypoint()
def main() -> None:
    result = probe.remote()
    print("CPW_V2_SECONDARY_H100_PROBE=" + json.dumps(result, sort_keys=True), flush=True)
