from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "tam-research-long-context-state-report"
VOLUME_NAME = "tam-research-data"
RESULT_PATH = "/vol/full100m-runs/tam-vs-transformer-long-context-state-seed8100.json"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
github_secret = modal.Secret.from_name("github-secret")
image = modal.Image.debian_slim(python_version="3.11").pip_install("PyGithub>=2.3,<3")


def _comment(repo_full_name: str, issue_number: int, body: str) -> None:
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
    memory=1024,
    timeout=120,
    volumes={"/vol": volume},
    secrets=[github_secret],
)
def report(repo_full_name: str = "", issue_number: int = 0) -> dict:
    volume.reload()
    path = Path(RESULT_PATH)
    if not path.exists():
        raise FileNotFoundError(RESULT_PATH)
    result = json.loads(path.read_text())
    cells = result["per_cell"]
    lengths = [int(x) for x in result["protocol"]["context_lengths"]]
    tasks = list(result["protocol"]["tasks"])

    def pooled(task_names, selected_lengths):
        tam_correct = trans_correct = total = 0.0
        for task in task_names:
            for length in selected_lengths:
                cell = cells[task][str(length)]
                n = int(cell["n"])
                tam_correct += float(cell["tam_accuracy"]) * n
                trans_correct += float(cell["transformer_accuracy"]) * n
                total += n
        return tam_correct / total, trans_correct / total

    by_task = {}
    for task in tasks:
        ta, tr = pooled([task], lengths)
        by_task[task] = {"tam": ta, "transformer": tr, "delta": ta - tr}

    in_lengths = [x for x in lengths if x <= 512]
    extra_lengths = [x for x in lengths if x > 512]
    tam_in, trans_in = pooled(tasks, in_lengths)
    tam_extra, trans_extra = pooled(tasks, extra_lengths)
    tam_all, trans_all = pooled(tasks, lengths)

    at_256 = {}
    for task in tasks:
        c = cells[task]["256"]
        at_256[task] = {
            "tam": float(c["tam_accuracy"]),
            "transformer": float(c["transformer_accuracy"]),
            "delta": float(c["tam_minus_transformer"]),
            "ci": c["paired_bootstrap_95ci"],
        }

    router = result.get("tam_router_stats") or {}
    mean_route = router.get("mean") or {}
    world = mean_route.get("world")
    attention = mean_route.get("attention")
    gates = router.get("per_layer_gate") or []

    lines = [
        "📚 **Long-context/state-memory task breakdown (zero GPU read)**",
        "",
        "**Pooled accuracy by task across all tested lengths**",
    ]
    for task in tasks:
        row = by_task[task]
        lines.append(
            f"- {task}: TAM={100*row['tam']:.2f}% vs Transformer={100*row['transformer']:.2f}% "
            f"(TAM-Transformer={100*row['delta']:+.2f} pp)"
        )
    lines += [
        "",
        f"- ≤512 pooled absolute accuracy: TAM={100*tam_in:.2f}% vs Transformer={100*trans_in:.2f}%",
        f"- >512 pooled absolute accuracy: TAM={100*tam_extra:.2f}% vs Transformer={100*trans_extra:.2f}%",
        f"- Overall absolute accuracy: TAM={100*tam_all:.2f}% vs Transformer={100*trans_all:.2f}%",
        "",
        "**256-token task breakdown (the length with the clear TAM aggregate win)**",
    ]
    for task in tasks:
        row = at_256[task]
        lines.append(
            f"- {task}: TAM={100*row['tam']:.1f}% vs Transformer={100*row['transformer']:.1f}% "
            f"({100*row['delta']:+.1f} pp; paired 95% CI {100*row['ci'][0]:+.1f} to {100*row['ci'][1]:+.1f})"
        )
    if world is not None:
        lines += [
            "",
            f"**TAM-v3 learned mixer gate:** attention≈{100*float(attention):.2f}%, world-state≈{100*float(world):.2f}%.",
        ]
        if gates:
            lines.append(
                f"Per-layer world-state gates: min={100*min(map(float,gates)):.2f}%, "
                f"max={100*max(map(float,gates)):.2f}% across {len(gates)} layers."
            )
    lines += ["", f"Source JSON: `{RESULT_PATH}`"]
    body = "\n".join(lines)
    if repo_full_name and issue_number:
        _comment(repo_full_name, issue_number, body)
    print(body, flush=True)
    return {
        "by_task": by_task,
        "at_256": at_256,
        "pooled": {
            "in_context": [tam_in, trans_in],
            "extrapolation": [tam_extra, trans_extra],
            "all": [tam_all, trans_all],
        },
        "router": router,
    }


@app.local_entrypoint()
def main(repo_full_name: str = "", issue_number: int = 0):
    print(json.dumps(report.remote(repo_full_name, issue_number), indent=2))
