from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("tam-research-posttrain-inspect")
volume = modal.Volume.from_name("tam-research-data", create_if_missing=True)
github_secret = modal.Secret.from_name("github-secret")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("PyGithub>=2.3,<3")
)


def _comment(repo_full_name: str, issue_number: int, body: str) -> None:
    if not repo_full_name or not issue_number:
        return
    import os
    import github

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print(body, flush=True)
        return
    client = github.Github(auth=github.Auth.Token(token))
    client.get_repo(repo_full_name).get_issue(number=issue_number).create_comment(body)


@app.function(
    image=image,
    cpu=1,
    memory=512,
    timeout=60,
    volumes={"/vol": volume},
    secrets=[github_secret],
)
def inspect(repo_full_name: str = "", issue_number: int = 0) -> dict:
    volume.reload()
    root = Path("/vol/full-model-runs/TAM-v3-25M-Full-seed7400")
    pretrain = Path(
        "/vol/full-model-runs/pretrain/25m/ctx512-mb64-ga2/"
        "tamv3-25m-compiled-seed7400/latest.pt"
    )
    paths = {
        "pretrained_checkpoint": pretrain,
        "sft_checkpoint": root / "sft.pt",
        "sft_summary": root / "sft_summary.json",
        "dpo_checkpoint": root / "final_dpo.pt",
        "dpo_summary": root / "dpo_summary.json",
        "final_summary": root / "final_summary.json",
        "posttrain_summary": root / "posttrain_summary.json",
    }
    result = {name: path.exists() for name, path in paths.items()}
    summaries = {}
    for name in ("sft_summary", "dpo_summary", "final_summary", "posttrain_summary"):
        path = paths[name]
        if path.exists():
            try:
                summaries[name] = json.loads(path.read_text())
            except Exception as exc:
                summaries[name] = {"read_error": f"{type(exc).__name__}: {exc}"}
    result["summaries"] = summaries
    _comment(
        repo_full_name,
        issue_number,
        "🔎 **Post-training volume inspection**\n```json\n"
        + json.dumps(result, indent=2)[:12000]
        + "\n```",
    )
    print(json.dumps(result, indent=2), flush=True)
    return result


@app.local_entrypoint()
def main(repo_full_name: str = "", issue_number: int = 0):
    print(json.dumps(inspect.remote(repo_full_name, issue_number), indent=2))
