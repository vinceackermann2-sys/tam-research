from __future__ import annotations

"""CPU-only archived Stage-C result/candidate ID join entrypoint (#1337)."""

import argparse
import json
from pathlib import Path

from tam_research.chm_v3_100m_daec_copy_candidate_overlap import (
    analyze_archive,
    extract_archived_result,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archived-job-log",type=Path,required=True)
    parser.add_argument("--report",type=Path,required=True)
    args = parser.parse_args()
    # Scientific Stage-C uses the frozen GPT-2 encoder; no substitute encoders.
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    if enc.name != "gpt2":
        raise RuntimeError("#1337 tokenizer identity drift")
    text = args.archived_job_log.read_text(encoding="utf-8")
    result = extract_archived_result(text)
    report = analyze_archive(result,enc.encode)
    args.report.write_text(
        json.dumps(report,sort_keys=True,indent=2)+"\n",encoding="utf-8",
    )
    print("CHM_V3_1337_CPU_ONLY_CANDIDATE_REPORT="+json.dumps(report,sort_keys=True))


if __name__ == "__main__":
    main()
