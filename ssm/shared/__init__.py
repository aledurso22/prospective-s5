"""SHARED components — identical in both arms.

    hippo.py   HiPPO-LegS matrix and its S5/DPLR diagonalization (Lambda, V, B~)
    params.py  ComplexParam container + the S5 initializers
    block.py   S5Block: LayerNorm -> SSM -> dropout -> GLU -> dropout -> residual

Nothing here knows about the prospective update; both the baseline and the
prospective layer import from it unchanged.
"""

from .hippo import (
    make_hippo_legs, make_nplr_hippo, make_dplr_hippo,
    hippo_eig_direct, hippo_init,
)
from .params import (
    ComplexParam, hippo_lambda_init, hippo_B_init, C_init, log_step_init,
)
from .block import S5Block

__all__ = [
    "make_hippo_legs", "make_nplr_hippo", "make_dplr_hippo",
    "hippo_eig_direct", "hippo_init",
    "ComplexParam", "hippo_lambda_init", "hippo_B_init", "C_init",
    "log_step_init",
    "S5Block",
]
