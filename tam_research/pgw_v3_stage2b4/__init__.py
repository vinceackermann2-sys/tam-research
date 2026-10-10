"""PGW-v3 Stage-2B4 static CPU-only preflight; no training."""

from .preflight import (
    ARM_NAMES, DELAYS, ArmDerivativeReport, CpuDerivativeTiming,
    cpu_derivative_timing_preflight, gradient_activity_preflight,
)

__all__ = [
    "ARM_NAMES", "DELAYS", "ArmDerivativeReport", "CpuDerivativeTiming",
    "cpu_derivative_timing_preflight", "gradient_activity_preflight",
]
