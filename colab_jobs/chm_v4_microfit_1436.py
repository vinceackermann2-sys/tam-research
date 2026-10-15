#!/usr/bin/env python3
"""Standalone exact-commit CPU bootstrap; submitted by the local issue worker."""
import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


COLAB_JOB_ID = 'chm-v4-1436-microfit-engineering-1436001-v1'

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-commit", required=True)
    p.add_argument("--issue", required=True, type=int)
    p.add_argument("--job-script", required=True)
    p.add_argument("--requested-gpu", required=True)
    p.add_argument("--script-sha256", required=True)
    a = p.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", a.repo_commit) or a.issue <= 0:
        p.error("invalid source/attempt identity")
    if a.requested_gpu != "CPU" or a.job_script != "colab_jobs/chm_v4_microfit_1436.py":
        p.error("job is CPU only and source-bound")
    # No credentials are transmitted; this is a public exact-SHA checkout.
    with tempfile.TemporaryDirectory(prefix="tam-colab-") as temporary:
        root = Path(temporary) / "repo"
        root.mkdir()
        subprocess.run(["git", "init", "--quiet", str(root)], check=True, timeout=15)
        subprocess.run(["git", "-C", str(root), "fetch", "--quiet", "--depth=1",
                        "https://github.com/vinceackermann2-sys/tam-research.git", a.repo_commit],
                       check=True, timeout=60)
        subprocess.run(["git", "-C", str(root), "checkout", "--quiet", "--detach", "FETCH_HEAD"],
                       check=True, timeout=15)
        actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        if actual != a.repo_commit:
            raise RuntimeError("Bootstrap did not retrieve exact commit")
        expected = (root / a.job_script).read_bytes()
        if hashlib.sha256(expected).hexdigest() != a.script_sha256:
            raise RuntimeError("Submitted standalone script differs from reviewed commit")
        subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                        "torch==2.10.0+cpu", "--index-url", "https://download.pytorch.org/whl/cpu"],
                       check=True, timeout=120)
        sys.path.insert(0, str(root))
        # Load stdlib helper by path: tam_research.__init__ imports torch.
        spec = importlib.util.spec_from_file_location("tam_colab_runtime", root / "tam_research/colab_job_runtime.py")
        runtime = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runtime)
        runtime.run_job(a, root, Path("/content/tam-results"), training=True)


if __name__ == "__main__":
    main()
