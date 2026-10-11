"""Verify a Colab checkout and compute device without launching training."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    import tam_research

    repo = Path(__file__).resolve().parents[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = torch.eye(8, device=device)
    assert torch.equal(a @ a, a), "Compute smoke test failed"
    report = {
        "status": "ok",
        "repository": "vinceackermann2-sys/tam-research",
        "commit": subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "tam_research": importlib.metadata.version("tam-research"),
        "package_path": str(Path(tam_research.__file__).resolve()),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "training_started": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Result saved to {args.output}")


if __name__ == "__main__":
    main()
