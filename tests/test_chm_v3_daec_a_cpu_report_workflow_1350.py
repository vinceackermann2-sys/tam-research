from __future__ import annotations

import ast
from pathlib import Path
import textwrap


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/chm-v3-daec-a-1350-cpu-report-v1.yml"
RUNNER = ROOT / "scripts/chm_v3_daec_a_stage_a_cpu_report_1350.py"


def test_cpu_report_workflow_is_one_shot_main_merge_and_no_paid_compute() -> None:
    source=WORKFLOW.read_text(encoding="utf-8")
    assert 'name: "CHM-v3 DAEC-A issue 1350 CPU toy quantitative report v1"' in source
    assert "branches: [main]" in source
    assert "paths:" in source
    assert "chm-v3-daec-a-1350-cpu-report-v1.yml" in source
    assert "workflow_dispatch" not in source
    assert "gpu:" not in source
    assert "modal run" not in source
    assert "MODAL_TOKEN" not in source
    assert "CUDA_VISIBLE_DEVICES: \"\"" in source
    assert "OMP_NUM_THREADS: \"1\"" in source
    assert "torch==2.10.0" in source
    assert "https://download.pytorch.org/whl/cpu" in source
    assert "upload-artifact@v4" in source
    assert "test \"$(git rev-parse HEAD:tam_research/chm_v3_100m_daec.py)\"" in source
    assert "77e9ccbff4383be40e6e2865503e1c3926bbda74" in source
    assert "b1a58e52b76ff7d06f14a569a7b6274afee8076c" in source
    assert "482e2c930fa02448e45327747f10571315fd41dc" in source
    assert "train_cpu_pointer_toy" in RUNNER.read_text(encoding="utf-8")
    assert source.count("python scripts/chm_v3_daec_a_stage_a_cpu_report_1350.py") == 1


def test_cpu_report_workflow_embedded_schema_code_is_valid_python() -> None:
    text=WORKFLOW.read_text(encoding="utf-8")
    lines=text.splitlines()
    begin=next(i for i,x in enumerate(lines) if "python - <<'PY'" in x)
    end=next(i for i,x in enumerate(lines) if i>begin and x.strip()=="PY")
    code=textwrap.dedent("\n".join(lines[begin+1:end]))
    tree=ast.parse(code)
    assert len(tree.body)>0
    assert "scientific_pass_claim" in code
    assert "CHM_V3_1350_CPU_TOY_REPORT_SCHEMA_PASS" in code


def test_frozen_cpu_report_is_never_a_scientific_trial() -> None:
    text=WORKFLOW.read_text(encoding="utf-8")
    assert 'r["historical_scientific_seed_2013161_consumed"] is True' in text
    assert 'r["prior_scientific_classification"]=="CHM_V3_100M_DAEC_STAGE_C_STOP"' in text
    assert 'r["gpu_allocated"] is False' in text
    assert 'r["scientific_pass_claim"] is False' in text
    assert 'r["further_gpu_authority"] is False' in text
