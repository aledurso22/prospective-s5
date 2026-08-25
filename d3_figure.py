"""D3 — the central figure, assembled from committed results (no new
training): static credit quality vs deployed training improvement.

x = static credit quality (cosine with exact credit, measured at fixed
params / alignment for deploy arms); y = deployed improvement as frac
of the online -> routeA gap. Sources: wiener_oracle, orient_wiener,
matched_phase, pac_deploy, pac_deploy2, pac_deploy4, factorize_w
summaries. routeA and frozen-learned-phase are marked adaptive/learned
(their static quality is not the object; annotated).

Run:  python d3_figure.py
"""
from __future__ import annotations

import json
import os

import numpy as np

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OUT = os.path.join(R, "d3_figure")


def load(name):
    with open(os.path.join(R, name, "summary.json")) as f:
        return json.load(f)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    points = []

    # Wiener full-filter deploys (static cos at trained params per K from
    # wiener_oracle wh probe, median across seeds/layers; y = deploy frac)
    wio = load("wiener_oracle")
    cos_by_K = {}
    for K in ["1", "4", "16", "32", "64", "96"]:
        vals = [wio["wh"][s][l][K]["cos"] for s in wio["wh"]
                for l in wio["wh"][s]]
        cos_by_K[K] = float(np.median(vals))
    points.append(dict(arm="wiener_full_K64", kind="wiener_full",
                       x=cos_by_K["64"], y=wio["frac"]))
    # orientation-only Wiener arms (deployed cosine from orient_wiener,
    # which is the honest in-loop quality)
    ori = load("orient_wiener")
    for K in [1, 4, 16, 32, 64, 96]:
        arm = f"orient{K}"
        points.append(dict(arm=arm, kind="wiener_orient",
                           x=float(np.median(ori["cosines"][arm])),
                           y=ori["fracs"][arm]))
    points.append(dict(arm="clipK64", kind="wiener_clip",
                       x=float(np.median(ori["cosines"]["clipK64"])),
                       y=ori["fracs"]["clipK64"]))
    # matched-class constant rotations (frozen/refresh/perbatch)
    mp = load("matched_phase")
    for arm, v in mp["fracs"].items():
        kind = ("perbatch" if arm.startswith("perbatch") else
                "refresh" if arm.startswith("refresh") else "frozen")
        points.append(dict(arm=arm, kind=f"matched_{kind}", x=None, y=v))
    # PAC derived laws (alignment from pac_deploy2)
    pd2 = load("pac_deploy2")
    for arm in ["pac_phase_oracle", "pac_phase_ema", "pac_full_oracle"]:
        points.append(dict(arm=arm, kind="pac_derived",
                       x=float(np.median(pd2["align"][arm])),
                       y=pd2["fracs"][arm]))
    # deploy4 family
    pd4 = load("pac_deploy4")
    for arm in ["c1_oracle", "c1_ema01", "c1_ema005", "c1_frozen200"]:
        points.append(dict(arm=arm, kind="pac_c1", x=None,
                           y=pd4["fracs"][arm]))
    # references
    points.append(dict(arm="online", kind="baseline", x=None, y=0.0))
    points.append(dict(arm="routeA_live", kind="learned", x=None, y=1.0))
    points.append(dict(arm="frozen_learned_phase", kind="learned_frozen",
                       x=None, y=1.13))
    points.append(dict(arm="bptt", kind="exact", x=1.0,
                       y=(0.0284 - 0.00003) / (0.0284 - 0.0015)))

    with open(os.path.join(OUT, "points.json"), "w") as f:
        json.dump(dict(points=points,
                       note="x = static/in-loop credit cosine; y = frac of "
                            "online->routeA gap closed when deployed"),
                  f, indent=2)

    print("D3 figure data (x = static credit quality, y = deployed frac):")
    print(f"  {'arm':<22s} {'kind':<16s} {'x':>6s} {'y':>8s}")
    for p in sorted(points, key=lambda p: (p["y"])):
        xs = f"{p['x']:.3f}" if p["x"] is not None else "  --"
        print(f"  {p['arm']:<22s} {p['kind']:<16s} {xs:>6s} {p['y']:>8.2f}")
    print("\nreading: the best static objects (wiener K=64/96) sit at the")
    print("BOTTOM of deployment; the best deployers (routeA, frozen-learned,")
    print("per-batch) have modest static quality. Static adjoint fidelity")
    print("does not predict learning performance.")


if __name__ == "__main__":
    main()
