from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MIRROR = ROOT / ".github" / "workflows" / "modal-cortex-pair1-successor-preflight-storage-mirror-v1.yml"
RUNNER = ROOT / "modal_cortex_reduced_attention_pair1_successor_100m_2b_v2.py"
SCIENTIFIC_WORKFLOW = ROOT / ".github" / "workflows" / "modal-cortex-reduced-attention-pair1-successor-100m-2b-v2.yml"
SCIENTIFIC_HARNESS = ROOT / "architectures" / "cortex_s" / "tests" / "test_reduced_attention_pair1_successor_100m_2b_v2_harness.py"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_storage_mirror_v1_is_single_use_storage_only_and_non_authorizing() -> None:
    source = _text(MIRROR)
    assert "workflow_dispatch" not in source
    assert "[modal-cortex-pair1-successor-preflight-storage-mirror-v1]" in source
    assert 'test "$RUN_ATTEMPT" = "1"' in source
    assert "\n          modal run " not in source
    assert ".remote(" not in source
    assert "modal.Function" not in source
    assert 'gpu="H100!"' not in source
    assert "modal volume get" in source
    assert "modal volume put" in source
    assert "modal volume rm" not in source
    assert "--force" not in source
    assert "scientific trigger authority: NONE" in source
    assert "Pair-1 reservation: NONE" in source
    assert "seed 59231 consumption: NONE" in source
    assert "gh issue create" not in source


def test_storage_mirror_v1_processes_exactly_the_seven_frozen_preflight_files() -> None:
    source = _text(MIRROR)
    paths = (
        "/data/tam100m-2b-curated-v1/train.bin",
        "/data/tam100m-2b-curated-v1/val.bin",
        "/data/tam100m-2b-curated-v1/meta.json",
        "/cortex-s-v0/100m-2b/reduced-attention-calibration-v1/RESULT.json",
        "/cortex-s-v0/100m-2b/reduced-attention-pair1-v1/PAIR1_DISPATCH_RESERVED.json",
        "/cortex-s-v0/100m-2b/reduced-attention-pair1-v1/transformer/RESULT.json",
        "/cortex-s-v0/100m-2b/reduced-attention-pair1-v1/reduced_attention/RESULT.json",
    )
    for path in paths:
        assert source.count(path) == 1, path
    assert 'test "$(wc -l < mirror-work/hashes.txt)" = "7"' in source
    assert '"file_count":7' in source
    assert "reduced-attention-pair1-successor-v2" not in source.split(
        "- name: Mirror exactly seven frozen pre-reservation files with parity", 1
    )[1].split("- name: Record terminal storage-mirror success", 1)[0]


def test_storage_mirror_v1_freezes_corpus_hashes_sizes_and_semantics() -> None:
    source = _text(MIRROR)
    assert "93e9cb0b7076a4ddd855fc696f657ea62592b8a03a05220be99c402f9043265b" in source
    assert "ae0bc5adf36d0aa8e55e5e3903401d3f114b93037f43221944b4e88c5d1a5760" in source
    assert "14bbbcf0ab0b8cba374074ef8ccb80a04ecead06c74780f35a1becd5aef1b8f3" in source
    assert "4000000000" in source
    assert "10000000" in source
    for text in (
        '"assembly_version":3',
        '"train_tokens":2_000_000_000',
        '"val_tokens":5_000_000',
        '"seed":8100',
        '"tokenizer":"gpt2"',
        '"dtype":"uint16"',
    ):
        assert text in source


def test_storage_mirror_v1_validates_calibration_and_predecessor_evidence() -> None:
    source = _text(MIRROR)
    for text in (
        'payload.get("status")=="CALIBRATION_PASS"',
        'decision.get("classification")=="PAIR1_PREPARATION_ALLOWED"',
        'decision.get("pair1_preparation_allowed") is True',
        'payload.get("engineering_seed")==2_026_092_001',
        'payload.get("scientific_seeds_consumed") is False',
        'payload.get("status")=="PAIR1_DISPATCH_RESERVED"',
        'payload.get("pair_seed")==58_231',
        'payload.get("pair_seed_consumed") is True',
        'abs(nll-2.694100785255432) < 1e-12',
        'payload.get("classification")=="SCIENTIFIC_REDUCED_ATTENTION_PAIR1_ARCHITECTURE_ERROR_NO_RETRY"',
        'payload.get("automatic_retry_authorized") is False',
        'payload.get("resume_authorized") is False',
    ):
        assert text in source


def test_storage_mirror_v1_fails_closed_on_destination_mismatch_and_accepts_exact_parity() -> None:
    source = _text(MIRROR)
    assert 'if modal volume get tam-research-data "$remote" "$dst"' in source
    assert 'verify_hash_size "$dst" "$expected_sha" "$expected_bytes"' in source
    assert 'test "$(sha_file "$dst")" = "$source_sha"' in source
    assert 'test "$(stat -c %s "$dst")" = "$source_bytes"' in source
    assert 'modal volume put tam-research-data "$src" "$remote"' in source
    # Existing destination content is never overwritten: put occurs only in the get-failure branch.
    fixed = source.split("mirror_fixed() {", 1)[1].split("validate_json() {", 1)[0]
    dynamic = source.split("mirror_json() {", 1)[1].split("\n          use_primary\n          modal token info", 1)[0]
    assert fixed.index("else") < fixed.index('modal volume put tam-research-data "$src" "$remote"')
    assert dynamic.index("else") < dynamic.index('modal volume put tam-research-data "$src" "$remote"')


def test_storage_mirror_v1_is_drift_tolerant_only_when_protected_blobs_match() -> None:
    source = _text(MIRROR)
    assert 'git -C source merge-base --is-ancestor "$SOURCE_SHA" "$LIVE_MAIN"' in source
    for frozen in (
        "78a3133fd1d1b6a14117c751f68cb5746536fb6a",
        "02ece24c6873a70d1d3e9270332a6f51af585b30",
        "796af6832e7687eddba298e65fcdfbdffd7bd885",
        "adb979e2ecaa7cdf1c0ee36e7a4d929783e078d6",
        "440292942066d0b3d3a71d40ca8495092674ce6c",
        "04b1e9c610195b0896a209eb9d6ce3fd4f014fbc",
    ):
        assert frozen in source
    assert 'verify_blob "modal_cortex_reduced_attention_pair1_successor_100m_2b_v2.py"' in source
    assert 'verify_blob ".github/workflows/modal-cortex-reduced-attention-pair1-successor-100m-2b-v2.yml"' in source
    assert 'verify_blob "architectures/cortex_s/tests/test_reduced_attention_pair1_successor_100m_2b_v2_harness.py"' in source


def test_storage_mirror_v1_does_not_modify_scientific_runner_workflow_or_harness() -> None:
    runner = _text(RUNNER)
    workflow = _text(SCIENTIFIC_WORKFLOW)
    harness = _text(SCIENTIFIC_HARNESS)
    assert 'PAIR_SEED = 59_231' in runner
    assert 'TRIGGER_TITLE = "[modal-cortex-reduced-attention-pair1-successor-100m-2b-v2]"' in runner
    assert "modal run --detach --timestamps" in workflow
    assert 'test "$SELECTED_ACCOUNT" = "secondary"' in workflow
    assert "PAIR_SEED = 59_231" in harness
