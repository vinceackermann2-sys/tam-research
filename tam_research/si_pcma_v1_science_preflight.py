"""PCMA-v1 scientific preflight: read-only, fail-closed, NON-EXECUTABLE.

Nothing here reserves a seed, executes a model, tests a heldout world, creates
an experiment, or authorizes scientific compute. Its only purpose is to make
missing preregistration and reproducibility requirements explicit.

This is specific to original screenshot-inspired SI/PCMA, NOT the TAM v3
scaling protocol or any PGW/CPW/CHM/RLT/Conv7 experiment.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping


CANONICAL_ORIGINAL_HYBRID = "575c17dc2f39a29d930d82179aa0c96dcbdc7d0f"
CANONICAL_ORIGINAL_PATH = "tam_research/si_original_hybrid_v0.py"
SCHEMA = "si-pcma-v1-science-preflight-draft/1"

# These are confirmed known consumed identifiers, NOT an exhaustive inventory.
# A human-independent durable attempt ledger search must still be completed.
KNOWN_CONSUMED_SI = frozenset({
    117010, 117211, 117212, 117213, 117214,
    118010, 118211, 118212, 118213,
})
# Other project seeds are also NOT authorized for SI, even if no overlap is
# known. Do not infer global uniqueness from this finite exclusion list.

_CODE_PATHS = (
    CANONICAL_ORIGINAL_PATH,
    "tam_research/si_pcma_v1.py",
    "tam_research/si_pcma_v1_factual_learning.py",
    "tam_research/si_pcma_v1_ood_world.py",
    "tam_research/si_pcma_v1_ood_controls.py",
    "tam_research/si_pcma_v1_ood_development.py",
    "tam_research/si_pcma_v1_fair_comparison.py",
    "tam_research/si_pcma_v1_calibration.py",
    "tam_research/si_pcma_v1_probe_value.py",
    "tam_research/si_pcma_v1_probe_randomization.py",
    "tam_research/si_pcma_v1_neural_probe.py",
    "tam_research/si_pcma_v1_persistence.py",
    "tam_research/si_pcma_v1_science_preflight.py",
    "tam_research/si_pcma_v1_cost_profile.py",
    "tam_research/si_pcma_v1_attempt_audit.py",
    "tam_research/si_pcma_v1_world_worker.py",
    "tam_research/si_pcma_v1_world_ipc.py",
    "tam_research/si_pcma_v1_strong_controls.py",
    "tam_research/si_pcma_v1_history_world.py",
    "tam_research/si_pcma_v1_history_information.py",
    "tam_research/si_pcma_v1_published_memory.py",
    "tam_research/si_pcma_v1_popgym_neural_pilot.py",
    "tam_research/si_pcma_v1_popgym_factual_ablation.py",
    # Every original-SI / PCMA interface test is also pinned for review.
    "tests/test_si_pcma_v1.py",
    "tests/test_si_pcma_v1_factual_learning.py",
    "tests/test_si_pcma_v1_calibration.py",
    "tests/test_si_pcma_v1_ood_world.py",
    "tests/test_si_pcma_v1_ood_development.py",
    "tests/test_si_pcma_v1_fair_comparison.py",
    "tests/test_si_pcma_v1_probe_value.py",
    "tests/test_si_pcma_v1_probe_randomization.py",
    "tests/test_si_pcma_v1_neural_probe.py",
    "tests/test_si_pcma_v1_persistence.py",
    "tests/test_si_pcma_v1_science_preflight.py",
    "tests/test_si_pcma_v1_attempt_audit.py",
    "tests/test_si_pcma_v1_world_ipc.py",
    "tests/test_si_pcma_v1_strong_controls.py",
    "tests/test_si_pcma_v1_history_world.py",
    "tests/test_si_pcma_v1_history_information.py",
    "tests/test_si_pcma_v1_published_memory.py",
    "tests/test_si_pcma_v1_popgym_neural_pilot.py",
    "tests/test_si_pcma_v1_popgym_factual_ablation.py",
)
_TEST_PREFIX = "tests/test_si_pcma_v1_"
_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA64 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class PreflightVerdict:
    status: str
    blockers: tuple[str, ...]
    source_files_checked: int
    new_scientific_seeds_reserved: int
    paid_compute_authorized: bool

    @property
    def may_execute(self) -> bool:
        """Engineering review results NEVER provide execution authority."""
        return False


def git_blob(path: Path, *, repo_root: Path) -> str:
    """Only hash a file; never touch Git state or invoke jobs."""
    relative = path.relative_to(repo_root)
    if relative.is_absolute() or ".." in relative.parts or not path.is_file():
        raise ValueError("invalid tracked source path")
    result = subprocess.run(
        ["git", "-C", str(repo_root), "hash-object", "--", str(relative)],
        capture_output=True, text=True, check=True, timeout=10,
    )
    blob = result.stdout.strip()
    if not _SHA40.fullmatch(blob):
        raise ValueError("invalid local Git blob SHA")
    return blob


def draft_manifest(
    source_commit: str,
    source_blobs: Mapping[str, str],
) -> dict[str, Any]:
    """Prospective review scaffold; NEVER an executable preregistration.

    Required scientific items are deliberately UNSET, rather than inventing
    seeds, sample sizes, confidence intervals, compute, or effect thresholds.
    """
    if not _SHA40.fullmatch(source_commit):
        raise ValueError("source commit must be an exact 40-digit SHA")
    paths = dict(source_blobs)
    if (set(paths) != set(_CODE_PATHS) or
            any(not _SHA40.fullmatch(blob) for blob in paths.values()) or
            paths[CANONICAL_ORIGINAL_PATH] != CANONICAL_ORIGINAL_HYBRID):
        raise ValueError("source pin list incomplete or original source changed")
    return {
        "schema": SCHEMA,
        "research_lineage": "original-screenshot-SI/PCMA-v1",
        "source_commit": source_commit,
        "source_blobs": dict(sorted(paths.items())),
        "status": "DRAFT_UNAPPROVED",
        "scientific_seeds": [],
        "reserved_attempts": [],
        "explicit_authorization": False,
        "training_recipe": None,
        "development_data_distribution": None,
        "heldout_distribution": None,
        "evaluation_worlds_per_family": None,
        "compute_budget_cpu_seconds": None,
        "frozen_model_checkpoint_digests": None,
        "matched_active_compute_profile": None,
        "locked_classical_baselines": None,
        "clustered_uncertainty_method": None,
        "failure_and_termination_thresholds": None,
        "results_destination": None,
        "negative_evidence": {
            "original_v26": "NO_GO",
            "v25_seed_117214_duplicate": "QUARANTINED",
        },
    }


def validate_review_manifest(
    manifest: Mapping[str, Any],
    *,
    repo_root: Path | None = None,
) -> PreflightVerdict:
    """Read-only review. Missing or tampered fields fail closed.

    Does NOT accept an approval, seed reservation or positive result supplied
    by the caller; a future independently reviewed tool must own those writes.
    """
    blockers: list[str] = []
    if not isinstance(manifest, dict):
        return PreflightVerdict("REJECTED", ("not an object",), 0, 0, False)
    if manifest.get("schema") != SCHEMA:
        blockers.append("unsupported preflight schema")
    if manifest.get("research_lineage") != "original-screenshot-SI/PCMA-v1":
        blockers.append("wrong scientific architecture lineage")
    if manifest.get("status") != "DRAFT_UNAPPROVED":
        blockers.append("this review tool cannot approve scientific execution")
    if manifest.get("explicit_authorization") is not False:
        blockers.append("approval cannot be self-declared by this manifest")
    seeds = manifest.get("scientific_seeds")
    if seeds != []:
        blockers.append("science seeds cannot be declared or consumed here")
        if isinstance(seeds, list) and any(
            type(s) is int and s in KNOWN_CONSUMED_SI for s in seeds
        ):
            blockers.append("previously consumed original-SI seed forbidden")
    if manifest.get("reserved_attempts") != []:
        blockers.append("attempt reservations require separate durable ledger")
    blobs = manifest.get("source_blobs")
    checked = 0
    if (not isinstance(blobs, dict) or set(blobs) != set(_CODE_PATHS) or
            any(type(v) is not str or not _SHA40.fullmatch(v)
                for v in blobs.values())):
        blockers.append("source file pins are missing/invalid")
    else:
        if blobs[CANONICAL_ORIGINAL_PATH] != CANONICAL_ORIGINAL_HYBRID:
            blockers.append("original canonical SI source mismatch")
        if repo_root is not None:
            root = Path(repo_root).resolve()
            for name, sha in sorted(blobs.items()):
                try:
                    actual = git_blob(root / name, repo_root=root)
                    if sha != actual:
                        blockers.append(f"local source blob mismatch: {name}")
                    else:
                        checked += 1
                except (OSError, ValueError, subprocess.SubprocessError):
                    blockers.append(f"source absent/unreadable: {name}")
    commit = manifest.get("source_commit")
    if type(commit) is not str or not _SHA40.fullmatch(commit):
        blockers.append("source commit pin missing")
    elif repo_root is not None:
        try:
            proc = subprocess.run(
                ["git", "-C", str(Path(repo_root).resolve()), "rev-parse", "HEAD"],
                capture_output=True, text=True, check=True, timeout=10,
            )
            if proc.stdout.strip() != commit:
                blockers.append("working-tree HEAD does not match pinned source commit")
        except (OSError, subprocess.SubprocessError):
            blockers.append("cannot read pinned HEAD")
    evidence = manifest.get("negative_evidence")
    if evidence != {
        "original_v26": "NO_GO",
        "v25_seed_117214_duplicate": "QUARANTINED",
    }:
        blockers.append("negative scientific evidence overwritten")
    for name in (
        "training_recipe", "development_data_distribution",
        "heldout_distribution", "evaluation_worlds_per_family",
        "compute_budget_cpu_seconds", "frozen_model_checkpoint_digests",
        "matched_active_compute_profile", "locked_classical_baselines",
        "clustered_uncertainty_method", "failure_and_termination_thresholds",
        "results_destination",
    ):
        if name not in manifest:
            blockers.append(f"missing review requirement: {name}")
        elif manifest[name] is None:
            blockers.append(f"not yet preregistered: {name}")
        else:
            # The NONEXECUTABLE review scaffold forbids interpreting filled
            # requirements as a sealed, source-verified science protocol.
            blockers.append(f"unapproved field supplied: {name}")
    allowed = {
        "schema", "research_lineage", "source_commit", "source_blobs",
        "status", "scientific_seeds", "reserved_attempts",
        "explicit_authorization", "negative_evidence",
        "training_recipe", "development_data_distribution",
        "heldout_distribution", "evaluation_worlds_per_family",
        "compute_budget_cpu_seconds", "frozen_model_checkpoint_digests",
        "matched_active_compute_profile", "locked_classical_baselines",
        "clustered_uncertainty_method", "failure_and_termination_thresholds",
        "results_destination",
    }
    if set(manifest) != allowed:
        blockers.append("missing/unapproved extra top-level fields")
    # Always blocked by design even with all code hashes pinned; a separate
    # immutable protocol, attempt-ledger reservation and explicit approval
    # would be needed before actual science can start.
    if repo_root is not None:
        # The repository's own historical result/protocol/addendum scan is
        # informative only; a complete cross-branch issue/worker ledger is
        # still mandatory even when this local report has no missing anchors.
        from tam_research.si_pcma_v1_attempt_audit import audit_original_si_docs
        try:
            historical = audit_original_si_docs(Path(repo_root))
            if historical.missing_anchor_ids:
                blockers.append(
                    "original SI interrupted/consumed anchors missing from "
                    "document scan: " + ",".join(map(str, historical.missing_anchor_ids)))
            if historical.global_attempt_ledger_covered or historical.may_reserve_or_execute:
                blockers.append("invalid historical ledger authority claim")
        except (OSError, UnicodeError, ValueError):
            blockers.append("historical original SI attempt source unreadable")
    blockers.extend((
        "historical consumed-seed ledger not exhaustively audited",
        "approval and unique-attempt reservation not granted",
        "independent heldout evaluator and leakage isolation not frozen",
        "CPU/FLOPs and stronger classical controls not yet matched",
    ))
    return PreflightVerdict(
        "BLOCKED", tuple(blockers), checked, 0, False
    )


def seal_draft_document(manifest: Mapping[str, Any]) -> bytes:
    """Content-address a proposed *review document*, NOT a science attempt."""
    raw = json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode("utf-8")
    document = {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "manifest": manifest,
        "message": "DRAFT REVIEW ONLY — NO SCIENTIFIC EXECUTION AUTHORIZED",
    }
    return json.dumps(document, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")
