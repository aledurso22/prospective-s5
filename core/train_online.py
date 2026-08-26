"""Online baseline (Zucchet-style causal online gradient) — toy rig.

Runs the online arm of the canonical paired protocol (5 seeds) and prints
per-seed finals + median. This is the deployable reference RoutePC/PC0 is
measured against; it shares rig config, data streams and optimizer with
toyrig/routepc.py via routepc.setup().

Run from repo root:  python -m core.train_online
"""
import numpy as np

from toyrig import route_a as cvm
from toyrig import routepc as rp


def main() -> None:
    rp.setup()                       # canonical rig config (L=4, N=16, T=128,
                                     # D=50, batch=32, STEPS from train_cell)
    finals = {}
    for seed in rp.SEEDS:
        out = cvm.train_route("online", seed)
        finals[seed] = out["final_loss"]
        print(f"online s{seed}: final {out['final_loss']:.4f}  "
              f"finite {out['finite']}", flush=True)
    med = float(np.median(list(finals.values())))
    print(f"online median final loss: {med:.4f}  "
          f"(frozen reference: 0.0224, seeds {rp.SEEDS})")


if __name__ == "__main__":
    main()
