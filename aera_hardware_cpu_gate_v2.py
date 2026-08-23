from __future__ import annotations

import aera_hardware_cpu_gate as gate
from tam_research.aera_hardware_core_v2 import HardwareAwareAERATextLMV2

# Reuse the exact preregistered v1 task, optimizer, metrics, and thresholds.
# Only the model implementation changes: cross-chunk stream state is now an
# unconditional residual. This avoids silently changing the gate after #176.
gate.HardwareAwareAERATextLM = HardwareAwareAERATextLMV2

if __name__ == "__main__":
    gate.main()
