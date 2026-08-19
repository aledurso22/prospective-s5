"""Prospective SSM on S5 (JAX).

Package layout — the split is BASELINE vs PROSPECTIVE vs SHARED:

    ssm/
    ├── shared/         identical in both arms
    │   ├── hippo.py    HiPPO-LegS + S5/DPLR diagonalization (Lambda, V, B~)
    │   ├── params.py   ComplexParam + the S5 initializers
    │   └── block.py    S5Block (LayerNorm -> SSM -> GLU -> residual)
    ├── baseline_s5/    the S5 control arm
    │   ├── scan.py     first-order elementwise associative scan
    │   └── layer.py    S5SSM (bilinear discretization)
    ├── prospective/    THE NEW WORK (treatment arm)
    │   ├── scan.py     2x2 companion affine associative scan
    │   └── layer.py    ProspectiveSSM (Euler-discretized prospective update)
    └── model.py        SequenceClassifier — the single arm switch

Everything outside ``prospective/`` is standard S5.
"""

from .shared.hippo import (
    make_hippo_legs, make_nplr_hippo, make_dplr_hippo,
    hippo_eig_direct, hippo_init,
)
from .shared.params import (
    ComplexParam, hippo_lambda_init, hippo_B_init, C_init, log_step_init,
)
from .shared.block import S5Block
from .baseline_s5.scan import (
    elementwise_scan, elementwise_scan_sequential, elementwise_scan_lax,
)
from .baseline_s5.layer import S5SSM, discretize_bilinear
from .prospective.scan import (
    prospective_scan, prospective_scan_sequential,
    prospective_scan_flat, prospective_scan_companion_lax,
)
from .prospective.layer import (
    ProspectiveSSM, causal_conv1d_time, log_step_init_prospective,
)
from .model import SequenceClassifier, count_params, build_model

__all__ = [
    # shared
    "make_hippo_legs", "make_nplr_hippo", "make_dplr_hippo",
    "hippo_eig_direct", "hippo_init",
    "ComplexParam", "hippo_lambda_init", "hippo_B_init", "C_init",
    "log_step_init", "S5Block",
    # baseline S5
    "elementwise_scan", "elementwise_scan_sequential", "elementwise_scan_lax",
    "S5SSM", "discretize_bilinear",
    # prospective
    "prospective_scan", "prospective_scan_sequential",
    "prospective_scan_flat", "prospective_scan_companion_lax",
    "ProspectiveSSM", "causal_conv1d_time", "log_step_init_prospective",
    # model
    "SequenceClassifier", "count_params", "build_model",
]
