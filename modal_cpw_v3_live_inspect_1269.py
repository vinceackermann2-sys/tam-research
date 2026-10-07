from __future__ import annotations

import json
from pathlib import Path
import modal

APP_NAME = "tam-research-cpw-v3-live-inspect-1269"
VOLUME_NAME = "tam-research-data"
app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11")

@app.function(image=image, cpu=0.125, memory=128, timeout=60, retries=0, volumes={'/vol': volume})
def inspect() -> dict:
    volume.reload()
    root = Path("/vol/cpw-v3/sparse-world-panel-v1")
    out = {
        "writes_performed": False,
        "gpu_requested": False,
        "root_exists": root.exists(),
        "attempt": (root / "ATTEMPT.json").exists(),
        "result": (root / "RESULT.json").exists(),
        "failure": (root / "FAILURE.json").exists(),
        "summaries": [],
    }
    for name in ("ATTEMPT.json", "RESULT.json", "FAILURE.json"):
        p = root / name
        if p.exists():
            try:
                out[name] = json.loads(p.read_text())
            except Exception as exc:
                out[name] = {"read_error": f"{type(exc).__name__}: {exc}"}
    if root.exists():
        for p in sorted(root.rglob("summary.json")):
            try:
                s = json.loads(p.read_text())
                out['summaries'].append({
                    'path': str(p.relative_to(root)),
                    'architecture': s.get('architecture'),
                    'seed': s.get('seed'),
                    'tokens_seen': s.get('tokens_seen'),
                    'nll': (s.get('final_eval') or {}).get('nll'),
                    'training_tps': s.get('training_tokens_per_second'),
                    'total_compute_seconds': s.get('total_compute_seconds'),
                })
            except Exception as exc:
                out['summaries'].append({'path': str(p.relative_to(root)), 'read_error': f'{type(exc).__name__}: {exc}'})
    print("CPW_V3_LIVE_INSPECT=" + json.dumps(out, sort_keys=True), flush=True)
    return out

@app.local_entrypoint()
def main() -> None:
    out = inspect.remote()
    print("CPW_V3_LIVE_INSPECT=" + json.dumps(out, sort_keys=True), flush=True)
