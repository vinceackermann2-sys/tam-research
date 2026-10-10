from __future__ import annotations

"""#1419 fresh first-attempt, CPU-only CLI; old failed #1415 stays frozen."""

import sys

from tam_research.chm_v3_alias_vs_hash_one_shot_import_safe_1419 import (
    ARMS, CLASSIFICATION, REPORT_MARKER, TRAIN_STEPS, main,
)


def cli() -> None:
    if sys.argv[1:] == ["--preflight-only"]:
        # Installed-package import check; no model, dataset, training or score.
        if len(ARMS) != 4 or TRAIN_STEPS != 256 or not REPORT_MARKER.startswith(
            "CHM_V3_1419_"
        ):
            raise RuntimeError("frozen import-only preflight failed")
        if CLASSIFICATION != "CHM_V3_1419_ALIAS_VS_HASH_CPU_EXPLORATORY":
            raise RuntimeError("new classification drift")
        print("CHM_V3_1419_REAL_ENTRYPOINT_IMPORT_ONLY_PASS")
    elif len(sys.argv) == 1:
        main()
    else:
        raise SystemExit("unknown command; only --preflight-only supported")


if __name__ == "__main__":
    cli()
