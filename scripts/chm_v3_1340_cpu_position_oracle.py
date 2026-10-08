from __future__ import annotations

"""Run CHM-v3 #1340 CPU-only, read-only historical position audit."""

import argparse
import json
from pathlib import Path

from tam_research.chm_v3_100m_daec_position_oracle import analyze_position_oracle


def main() -> None:
    parser = argparse.ArgumentParser(description="No model or GPU: frozen Stage-C memory-position audit")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    import tiktoken
    encoder = tiktoken.get_encoding("gpt2")
    if encoder.name != "gpt2":
        raise RuntimeError("#1340 requires exact GPT-2 encoding")
    archive = json.loads(args.archive.read_text(encoding="utf-8"))
    result = analyze_position_oracle(archive, encoder.encode, decode=encoder.decode)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print("CHM_V3_1340_CPU_POSITION_ORACLE_REPORT=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
