from __future__ import annotations

import json
import modal

APP_NAME = "tam-research-inspect"
VOLUME_NAME = "tam-research-data"
SEED = 7400

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
github_secret = modal.Secret.from_name("github-secret")

image = modal.Image.debian_slim(python_version="3.11").pip_install("PyGithub>=2.3,<3")


def _comment(repo_full_name: str, issue_number: int, body: str) -> None:
    if not repo_full_name or not issue_number:
        return
    import os
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print(body, flush=True)
        return
    try:
        import github
        client = github.Github(auth=github.Auth.Token(token))
        client.get_repo(repo_full_name).get_issue(number=issue_number).create_comment(body)
    except Exception as exc:
        print(f"[status-report-nonfatal] {type(exc).__name__}: {exc}; body={body}", flush=True)


@app.function(
    image=image,
    cpu=1,
    memory=512,
    timeout=90,
    volumes={"/vol": volume},
    secrets=[github_secret],
)
def inspect_posttrain(repo_full_name: str = "", issue_number: int = 0) -> dict:
    from pathlib import Path

    volume.reload()
    run_dir = Path(f"/vol/full-model-runs/TAM-v3-25M-Full-seed{SEED}")
    pretrain = Path(
        "/vol/full-model-runs/pretrain/25m/ctx512-mb64-ga2/"
        f"tamv3-25m-compiled-seed{SEED}/latest.pt"
    )
    names = [
        "sft.pt",
        "sft_summary.json",
        "final_dpo.pt",
        "dpo_summary.json",
        "posttrain_summary.json",
        "final_summary.json",
    ]
    artifacts = {}
    for name in names:
        path = run_dir / name
        entry = {"exists": path.exists()}
        if path.exists():
            entry["bytes"] = path.stat().st_size
            if path.suffix == ".json":
                try:
                    entry["summary"] = json.loads(path.read_text())
                except Exception as exc:
                    entry["json_error"] = f"{type(exc).__name__}: {exc}"
        artifacts[name] = entry

    result = {
        "pretrained_checkpoint": {"exists": pretrain.exists(), "bytes": pretrain.stat().st_size if pretrain.exists() else None},
        "run_dir_exists": run_dir.exists(),
        "artifacts": artifacts,
    }

    sft = artifacts["sft.pt"]["exists"] and artifacts["sft_summary.json"]["exists"]
    dpo = artifacts["final_dpo.pt"]["exists"] and artifacts["dpo_summary.json"]["exists"]
    final = artifacts["final_summary.json"]["exists"] or artifacts["posttrain_summary.json"]["exists"]
    status = f"base={'yes' if pretrain.exists() else 'NO'}, SFT={'yes' if sft else 'no'}, DPO={'yes' if dpo else 'no'}, final-summary={'yes' if final else 'no'}"
    details = []
    if artifacts["sft_summary.json"].get("summary"):
        s = artifacts["sft_summary.json"]["summary"]
        details.append(f"SFT assistant NLL={s.get('heldout', {}).get('assistant_nll', 'n/a')}, train={s.get('training_seconds', 'n/a')}s, compile={s.get('compile_seconds', 'n/a')}s")
    if artifacts["dpo_summary.json"].get("summary"):
        d = artifacts["dpo_summary.json"]["summary"]
        details.append(f"DPO implicit reward acc={d.get('heldout', {}).get('implicit_reward_accuracy', 'n/a')}, train={d.get('training_seconds', 'n/a')}s")
    body = "🔎 **Persistent Volume inspection** — " + status
    if details:
        body += "\n" + "\n".join(f"- {line}" for line in details)
    _comment(repo_full_name, issue_number, body)
    print(json.dumps(result, indent=2), flush=True)
    return result


@app.local_entrypoint()
def main(repo_full_name: str = "", issue_number: int = 0):
    result = inspect_posttrain.remote(repo_full_name, issue_number)
    print(json.dumps(result, indent=2))
