from __future__ import annotations

import aera_hardware_cpu_gate as gate
from tam_research.aera_hardware_core_v3 import HardwareAwareAERATextLMV3

# Reuse the exact preregistered v1 task, optimizer, training steps, metrics, and
# thresholds. The model implementation adds a causal predictive-stream training
# objective and a direct observed-summary path into the recurrent state update.
# The >=90% carried-boundary gate is unchanged.
gate.HardwareAwareAERATextLM = HardwareAwareAERATextLMV3

if __name__ == "__main__":
    gate.main()
