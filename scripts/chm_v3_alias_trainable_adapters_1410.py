from __future__ import annotations

"""#1410: one-shot TRAIN-only adapter wiring and preprocessing overhead audit."""

import hashlib
import json
import time

from tam_research.chm_v3_alias_trainable_adapters_1410 import (
    CLASSIFICATION, alias_encoded_view,
    one_train_step_four_arm_smoke, verify_train_only_encoder_parity,
)
from tam_research.chm_v3_balanced_train_schedule_1391 import (
    TRAIN_STEPS, balanced_training_episode,
)
from tam_research.chm_v3_counterfactual_model_view_1365 import seal_model_view
from tam_research.chm_v3_matched_tiny_models_1377 import encode_sealed_view


def main() -> None:
    geometry = verify_train_only_encoder_parity()
    views = tuple(seal_model_view(balanced_training_episode(i))
                  for i in range(TRAIN_STEPS))
    # This is CPU Python preprocessing wall timing ONLY: NOT model learning
    # throughput, inference FLOPs or a portable benchmark.
    before=time.perf_counter_ns()
    for view in views:
        encode_sealed_view(view)
    hash_ns=time.perf_counter_ns()-before
    before=time.perf_counter_ns()
    for view in views:
        alias_encoded_view(view)
    alias_ns=time.perf_counter_ns()-before
    # Exactly one TRAIN-only shared gradient update on each tiny arm.
    smoke=one_train_step_four_arm_smoke()
    report = {
        "classification": CLASSIFICATION,
        "source_domain": "TRAIN-only 256 static views; one shared TRAIN optimizer step per arm",
        "geometry": geometry,
        "single_training_step_smoke": smoke,
        "preprocessor_time_ns_hash_256_train_views": hash_ns,
        "preprocessor_time_ns_alias_256_train_views": alias_ns,
        "preprocessor_timer_note": (
            "Single noisy CPU wall measurement, not statistically estimated "
            "latency; aliased adapter includes a second baseline-geometry "
            "verification pass by design. No model evaluation."
        ),
        "old_sci_stop": "CHM_V3_100M_DAEC_STAGE_C_STOP",
        "old_seed_consumed": 2013161,
        "new_scientific_attempt": False,
        "gpu_allocated": False,
        "heldout_scoring": False,
    }
    digest=json.dumps(report,sort_keys=True,separators=(",",":"),allow_nan=False)
    report["sha256_without_digest"] = hashlib.sha256(digest.encode()).hexdigest()
    print("CHM_V3_1410_TRAIN_ADAPTER_RESULT="+json.dumps(report,sort_keys=True,allow_nan=False))


if __name__=="__main__":
    main()
