"""GD — action-Jacobian RoutePC arms E1/E2 (refinement D).

Current PC0 uses J_n = d_w (M_w sg g_n) even though the actual model
action is M_w g -> global clip -> Adam, and clipping fires on 100% of
steps — the mismatch is a primary hypothesis. Two causal arms, rest of
the PC0 protocol fixed, zero BPTT:

  e1clip    clip-aware residual: r̂ = J^T D_clip h_n, with
            D_clip = (C/||x||)(I - xhat xhat^T) at the previous step's
            pre-clip flat gradient (active regime, symmetric).
  e2action  full one-step action-aware residual: r̂ = [D_w A]^T h_n with
            A = Adam-direction o clip; h_n transformed by D1 D_clip
            (D1 = d(Adam direction)/d(clipped grad), frozen previous
            Adam moments/step index). No backprop through g_n, history,
            or g_{n+1}.

AUDITS (asserted): step-1 forward trajectory identical to PC0 (w and
loss bitwise); BPTT/exact calls in the arms = 0. Also reported:
radial/tangential decomposition of the meta residual (old PC0 vs
E1/E2) — does respecting the clipping Jacobian suppress the runaway
radial component? — via |w| max and radial-residual share over
training.

REGISTERED rule (as G1): sane iff all 5 finite AND median <= median
(online); competitive iff median <= 1.5 x median(PC0) AND beats online
>= 4/5; competitive arms auto-extend to seeds 5..14.

Run:  python -m controls.gd_action_jacobian
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

from toyrig import routepc as rp
from controls.geometry_traj import setup, train_arm

SEEDS5 = [0, 1, 2, 3, 4]
ARMS = ["e1clip", "e2action"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "geometry_audit")


def stored():
    rp_ref = json.load(open(os.path.join(ROOT, "results", "route_pc",
                                         "summary.json")))
    c15 = json.load(open(os.path.join(ROOT, "results",
                                      "c1_phase_only_routepc",
                                      "summary_15seeds.json")))
    f = {"online": {}, "pc0": {}}
    for s in range(5):
        f["online"][s] = rp_ref["finals"]["online"][str(s)]
        f["pc0"][s] = rp_ref["finals"]["pc_b0.0"][str(s)]
    for s in range(5, 15):
        f["online"][s] = c15["finals"]["online"][str(s)]
        f["pc0"][s] = c15["finals"]["pc0"][str(s)]
    return f


def med(d, seeds):
    return float(np.median([d[s] for s in seeds]))


def main() -> None:
    setup()
    os.makedirs(OUT, exist_ok=True)
    f = stored()

    # ---- step-1 audit: pc0/e1/e2 share the identical first update ----
    import toyrig.train_cell as _tc
    import controls.geometry_traj as gt
    orig = gt.STEPS
    gt.STEPS = 1
    _, t_pc0 = gt.train_arm("pc0", 0)
    _, t_e1 = gt.train_arm("e1clip", 0)
    _, t_e2 = gt.train_arm("e2action", 0)
    gt.STEPS = orig
    assert np.array_equal(t_pc0["w"][-1], t_e1["w"][-1])
    assert np.array_equal(t_pc0["w"][-1], t_e2["w"][-1])
    assert t_pc0["losses"][-1] == t_e1["losses"][-1] == t_e2["losses"][-1]
    print("AUDIT step-1 forward identical to PC0 (w, loss bitwise): PASS")

    audit0 = dict(rp.BPTT_CALLS)
    doc = {"arms": {}}
    pc0_med5 = med(f["pc0"], SEEDS5)
    on_med5 = med(f["online"], SEEDS5)
    for arm in ARMS:
        print("=" * 70)
        finals, dyn = {}, {}
        for seed in SEEDS5:
            out, traj = train_arm(arm, seed, extra=True, exact_probes=False)
            finals[seed] = out["final_loss"]
            rho = np.abs(traj["w"])
            share = (traj["res_rad"][1:]
                     / (traj["res_rad"][1:] + traj["res_tan"][1:]
                        + 1e-30))
            dyn[seed] = dict(
                finite=out["finite"], rho_max=float(np.nanmax(rho)),
                rho_final_med=float(np.nanmedian(rho[-100:])),
                rad_share_med=float(np.nanmedian(share)))
            print(f"{arm} s{seed}: final {out['final_loss']:.4f}  "
                  f"finite {out['finite']}  rho max "
                  f"{dyn[seed]['rho_max']:.2f}  radial-res share "
                  f"{dyn[seed]['rad_share_med']:.3f}", flush=True)
            np.savez(os.path.join(OUT, f"traj_{arm}_s{seed}.npz"), **traj)
        all_fin = all(dyn[s]["finite"] for s in SEEDS5)
        m = med(finals, SEEDS5)
        beats = sum(finals[s] < f["online"][s] for s in SEEDS5)
        sane = all_fin and m <= on_med5
        comp = all_fin and m <= 1.5 * pc0_med5 and beats >= 4
        print(f"[{arm}] median {m:.4f}  sane {bool(sane)}  "
              f"competitive {bool(comp)}")
        row = dict(finals5={str(s): finals[s] for s in SEEDS5},
                   dyn5={str(s): dyn[s] for s in SEEDS5},
                   median5=m, sane=bool(sane), competitive=bool(comp))
        if comp:
            for seed in range(5, 15):
                out, traj = train_arm(arm, seed, extra=True, exact_probes=False)
                finals[seed] = out["final_loss"]
                np.savez(os.path.join(OUT, f"traj_{arm}_s{seed}.npz"),
                         **traj)
            m15 = med(finals, list(range(15)))
            fails = [s for s in range(15)
                     if finals[s] > f["online"][s]]
            row.update(finals15={str(s): finals[s] for s in range(15)},
                       median15=m15, fails15=fails)
            print(f"[{arm}] 15-seed median {m15:.4f}  fails {fails}")
        doc["arms"][arm] = row

    audit = {k: rp.BPTT_CALLS[k] - audit0[k] for k in rp.BPTT_CALLS}
    assert audit["exact_grad"] == 0 and audit["exact_lambda"] == 0, \
        f"BPTT in E1/E2 arms: {audit}"
    print(f"AUDIT BPTT calls in E1/E2 training: {audit} (0/0 required)")
    doc["probe_calls"] = audit
    doc["git"] = subprocess.run(["git", "rev-parse", "HEAD"],
                                capture_output=True,
                                text=True).stdout.strip()
    with open(os.path.join(OUT, "gd_summary.json"), "w") as fo:
        json.dump(doc, fo, indent=2, default=float)
    print("wrote gd_summary.json")


if __name__ == "__main__":
    main()
