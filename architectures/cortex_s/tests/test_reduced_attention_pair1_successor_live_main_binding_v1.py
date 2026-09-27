from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / ".github" / "workflows" / "modal-cortex-reduced-attention-pair1-successor-100m-2b-v2.yml"
RUNNER = ROOT / "modal_cortex_reduced_attention_pair1_successor_100m_2b_v2.py"
HARNESS = ROOT / "architectures" / "cortex_s" / "tests" / "test_reduced_attention_pair1_successor_100m_2b_v2_harness.py"
MODEL = ROOT / "architectures" / "cortex_s" / "reduced_attention_100m_v1.py"
PROTOCOL = ROOT / "architectures" / "cortex_s" / "reduced_attention_100m_v1_protocol.py"
MODELS = ROOT / "tam_research" / "models.py"
TRAIN = ROOT / "tam_research" / "train.py"
DATA = ROOT / "tam_research" / "data.py"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_pair1_successor_live_main_binding_is_ancestry_and_blob_based() -> None:
    source = _text(WORKFLOW)
    assert 'test "$(git rev-parse origin/main)" = "$SOURCE_SHA"' not in source
    assert 'LIVE_MAIN="$(git rev-parse origin/main)"' in source
    assert 'git merge-base --is-ancestor "$SOURCE_SHA" "$LIVE_MAIN"' in source
    assert 'git merge-base --is-ancestor "$HARNESS_SHA" "$SOURCE_SHA"' in source
    assert 'test "$(git rev-parse "$SOURCE_SHA:$path")" = "$(git rev-parse "$LIVE_MAIN:$path")"' in source


def test_pair1_successor_live_main_binding_checks_all_protected_paths() -> None:
    source = _text(WORKFLOW)
    protected = (
        "architectures/cortex_s/reduced_attention_100m_v1.py",
        "architectures/cortex_s/reduced_attention_100m_v1_protocol.py",
        "architectures/cortex_s/tests/test_reduced_attention_pair1_successor_100m_2b_v2_harness.py",
        "tam_research/models.py",
        "tam_research/train.py",
        "tam_research/data.py",
        "tam_research/modal_dual_account_v3.py",
        "scripts/modal_select_account_v3.py",
        "modal_runtime_admission_probe_1067_v1.py",
        "modal_cortex_reduced_attention_pair1_successor_100m_2b_v2.py",
        ".github/workflows/modal-cortex-reduced-attention-pair1-successor-100m-2b-v2.yml",
    )
    loop = source.split("for path in", 1)[1].split("; do", 1)[0]
    for path in protected:
        assert path in loop, path


def test_pair1_successor_harness_diff_still_freezes_scientific_files() -> None:
    source = _text(WORKFLOW)
    line = next(line for line in source.splitlines() if 'git diff --exit-code "$HARNESS_SHA" "$SOURCE_SHA"' in line)
    for path in (
        "architectures/cortex_s/reduced_attention_100m_v1.py",
        "architectures/cortex_s/reduced_attention_100m_v1_protocol.py",
        "architectures/cortex_s/tests/test_reduced_attention_pair1_successor_100m_2b_v2_harness.py",
        "tam_research/models.py",
        "tam_research/train.py",
        "tam_research/data.py",
        "tam_research/modal_dual_account_v3.py",
        "scripts/modal_select_account_v3.py",
        "modal_runtime_admission_probe_1067_v1.py",
        "modal_cortex_reduced_attention_pair1_successor_100m_2b_v2.py",
    ):
        assert path in line, path
    assert ".github/workflows/modal-cortex-reduced-attention-pair1-successor-100m-2b-v2.yml" not in line


def test_pair1_successor_scientific_contract_remains_frozen() -> None:
    runner = _text(RUNNER)
    harness = _text(HARNESS)
    model = _text(MODEL)
    protocol = _text(PROTOCOL)
    models = _text(MODELS)
    train = _text(TRAIN)
    data = _text(DATA)

    assert "PAIR_SEED = 59_231" in runner
    assert 'TRAIN_SHA256 = "93e9cb0b7076a4ddd855fc696f657ea62592b8a03a05220be99c402f9043265b"' in runner
    assert 'VAL_SHA256 = "ae0bc5adf36d0aa8e55e5e3903401d3f114b93037f43221944b4e88c5d1a5760"' in runner
    assert 'META_SHA256 = "14bbbcf0ab0b8cba374074ef8ccb80a04ecead06c74780f35a1becd5aef1b8f3"' in runner
    assert 'PAIR1_SCREEN_MAX_NLL_DELTA = 0.015' in runner
    assert "PAIR_SEED = 59_231" in harness
    assert model
    assert protocol
    assert models
    assert train
    assert data
