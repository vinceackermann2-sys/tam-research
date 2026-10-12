"""Verify the Google Cloud GPU and repo; never start research training."""
import json
import os
from pathlib import Path
import torch
import tam_research

assert torch.cuda.is_available(), "CUDA unavailable"
x = torch.eye(16, device="cuda")
assert torch.equal(x @ x, x)
report = {"status": "ok", "gpu": torch.cuda.get_device_name(0),
          "torch": torch.__version__, "cuda": torch.version.cuda,
          "training_started": False}
target = Path(os.environ["TAM_RESULTS_DIR"]) / "smoke.json"
target.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
