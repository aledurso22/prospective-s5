"""BASELINE ARM — standard S5 (Smith et al.), the control.

    scan.py   first-order elementwise associative scan
    layer.py  S5SSM: bilinear discretization + that scan

Nothing in here is new work; it is the reference the prospective arm is
compared against.
"""

from .scan import (
    elementwise_scan, elementwise_scan_sequential, elementwise_scan_lax,
    select_scan,
)
from .layer import S5SSM, discretize_bilinear

__all__ = [
    "elementwise_scan", "elementwise_scan_sequential", "elementwise_scan_lax",
    "select_scan", "S5SSM", "discretize_bilinear",
]
