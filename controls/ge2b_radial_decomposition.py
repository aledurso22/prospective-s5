"""GE2B — common vs relative decomposition of E2's radial meta-residual
(pre-freeze stored-log analysis; re-instrumented REPLAY of the frozen
e2action arm — no new arm, bitwise-gated against ge2_summary finals).

E2 keeps rho ~ 1 and sd(log|w|) ~ 0, yet carries a ~0.45 radial
residual share. The action-space R_A probe showed the COMMON radial
direction is nearly action-null (R_A ~ 0.11) while per-layer relative
radial is substantial (0.6-0.87). This decomposes the per-layer radial
residual vector r_alpha into

    r_common   = u u^T r_alpha,        u = 1/sqrt(m)  (all-ones dir)
    r_relative = (I - u u^T) r_alpha,

and reports the energy share ||r_common||^2 / ||r_alpha||^2 vs
||r_relative||^2 / ||r_alpha||^2 per layer, early/mid/late, pooled over
the 15 seeds. If the radial residual is predominantly common, E2's
apparent radial content is behaviorally inert by the action null — it
does not contradict sd(log|w|) ~ 0.

Run:  python -m controls.ge2b_radial_decomposition
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

from toyrig import routepc as rp
from controls.geometry_traj import setup, train_arm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "geometry_audit")


def main() -> None:
    setup()
    ge2 = json.load(open(os.path.join(OUT, "ge2_summary.json")))
    stored = {int(s): v for s, v in ge2["finals"].items()}
    audit0 = dict(rp.BPTT_CALLS)
    shares = {l: {"early": [], "mid": [], "late": [], "all": []}
              for l in range(4)}
    finals = {}
    for seed in range(15):
        out, traj = train_arm("e2action", seed, exact_probes=False)
        finals[seed] = out["final_loss"]
        assert finals[seed] == stored[seed], \
            f"replay gate failed s{seed}"
        ra = traj["ralpha"]                       # (STEPS, L, N)
        n = ra.shape[0]
        for l in range(4):
            r = ra[1:, l, :]                      # skip step 1 (no residual)
            m = r.shape[1]
            common = r.mean(axis=1, keepdims=True)
            e_c = (m * common[:, 0] ** 2)
            e_t = (r ** 2).sum(axis=1) + 1e-30
            share = e_c / e_t
            for name, sl in (("early", slice(0, n // 3)),
                             ("mid", slice(n // 3, 2 * n // 3)),
                             ("late", slice(2 * n // 3, n)),
                             ("all", slice(0, n))):
                shares[l][name].append(float(np.median(share[sl])))
        print(f"e2action s{seed} replay == stored ✓", flush=True)
    audit = {k: rp.BPTT_CALLS[k] - audit0[k] for k in rp.BPTT_CALLS}
    assert audit["exact_grad"] == 0 and audit["exact_lambda"] == 0

    print("-" * 78)
    print("E2 radial residual: COMMON energy share (median over seeds)")
    print(f"{'layer':<6s}{'early':>8s}{'mid':>8s}{'late':>8s}{'all':>8s}")
    out_tab = {}
    for l in range(4):
        row = {k: float(np.median(shares[l][k]))
               for k in ("early", "mid", "late", "all")}
        out_tab[l] = row
        print(f"L{l:<5d}{row['early']:>8.3f}{row['mid']:>8.3f}"
              f"{row['late']:>8.3f}{row['all']:>8.3f}")
    pooled = {k: float(np.median([shares[l][k][s]
                                  for l in range(4)
                                  for s in range(15)]))
              for k in ("early", "mid", "late", "all")}
    print(f"pooled: {pooled}")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git, per_layer=out_tab, pooled=pooled,
               probe_calls=audit)
    with open(os.path.join(OUT, "ge2b_summary.json"), "w") as fo:
        json.dump(doc, fo, indent=2, default=float)
    print("wrote ge2b_summary.json")


if __name__ == "__main__":
    main()
