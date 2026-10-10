from __future__ import annotations

"""Print exactly one source-bound, model-free TRAIN-only #1399 JSON audit."""

import json

from tam_research.chm_v3_identifier_composition_audit_1399 import (
    audit_training_inputs_only,
)

if __name__ == "__main__":
    print(
        "CHM_V3_1399_TRAIN_ID_AUDIT="
        + json.dumps(audit_training_inputs_only(), sort_keys=True, allow_nan=False)
    )
