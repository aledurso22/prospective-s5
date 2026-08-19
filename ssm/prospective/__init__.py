"""PROSPECTIVE ARM — the new work (treatment).

    scan.py   2x2 companion affine associative scan (second-order recurrence)
    layer.py  ProspectiveSSM: the Euler-discretized prospective update

Everything else (HiPPO init, block, classifier, training loop) is shared with
the baseline, so any measured difference comes from these two files.
"""

from .scan import (
    prospective_scan, prospective_scan_sequential,
    prospective_scan_flat, prospective_scan_companion_lax, select_scan,
)
from .layer import ProspectiveSSM, causal_conv1d_time, log_step_init_prospective

__all__ = [
    "prospective_scan", "prospective_scan_sequential",
    "prospective_scan_flat", "prospective_scan_companion_lax", "select_scan",
    "ProspectiveSSM", "causal_conv1d_time", "log_step_init_prospective",
]
