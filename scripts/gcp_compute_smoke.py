"""Dependency-free infrastructure check; no training, datasets, or scientific seeds."""

import json
import os
import platform


if __name__ == "__main__":
    checksum = sum(i * i for i in range(100_000))
    assert checksum == 333328333350000
    print(json.dumps({
        "result": "TAM_GCP_COMPUTE_OK",
        "repo_sha": os.environ.get("TAM_REPO_SHA"),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "checksum": checksum,
    }), flush=True)
