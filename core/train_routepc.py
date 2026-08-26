"""Canonical RoutePC / PC0 entry point — toy delayed-copy rig.

Runs the frozen paired protocol: online baseline, routeA (exact-teacher
reference) and the PC arms; PC0 (beta=0, correction only) is the deployed
algorithm. Zero BPTT calls in the PC arms (asserted in-run).

Canonical implementation: toyrig/routepc.py (frozen; regression-gated
bitwise by tests/test_pc0_regression.py). Documentation:
core/README.md (structure) and README_ROUTEPC.md (algorithm).

Run from repo root:  python -m core.train_routepc
"""
from toyrig import routepc

if __name__ == "__main__":
    routepc.main()
