"""D6 v2 — generality with a headroom gate (the directive's precondition).

D6 v1 at D=50 had no credit gap (bptt ~ online): capacity-limited, so
P3 was uninterpretable. This version first finds a regime WITH headroom.

Step 1 (gate): headroom sweep D in {10, 20, 30}, 1 seed, online vs bptt.
headroom = (online - bptt)/online. Proceed at the largest D with
headroom >= 0.3. If none has it, stop and report (the rot-RNN's credit
gap is the S5 pathology's absence — also a finding).

Step 2 (mechanism arms, paired seeds {0,1,2}, chosen D):
  online, bptt, routePhi (learned per-block rotation, meta-gradient),
  frozenPhi (the learned rotation deployed frozen from init),
  scalarGain (per-block scalar gain meta-learned — the scalar-only
  ablation), perbatchOracle (per-batch closed-form phase per block:
  phi_i = arg c*_i from the exact teacher on the current batch —
  2-vector complexified z = q0 + i q1).

REGISTERED BARS (fixed before running):
  HEADROOM: chosen regime has (online - bptt)/online >= 0.3.
  P3 (generality of the win): median routePhi <= 0.7 x median online,
  all finite (looser than S5's 0.5x — different rig, harder task).
  CONTROL: scalarGain should capture much less of the gap than
  routePhi (orientation > gain, as in the complex rig).

Run:  python rot_rnn_generality2.py
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

import rot_rnn as rr
from rot_rnn_generality import (rotate_G, drotate_G, adam, clip,
                                train_arm)

SEEDS = [0, 1, 2]
D_GRID = [10, 20, 30]
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results", "rot_rnn_generality2")


def headroom(delay, seed=0, steps=800):
    rr.DELAY = delay
    rr.STEPS = steps
    on = train_arm("online", seed)["final"]
    bp = train_arm("bptt", seed)["final"]
    rr.STEPS = 1500
    return on, bp, (on - bp) / max(on, 1e-300)


def train_scalar(seed):
    """Per-block scalar gain meta-learned (scalar-only ablation)."""
    params = rr.init_params(seed)
    rng = np.random.RandomState(1000 + seed)
    gain = [np.ones(rr.NB) for _ in range(rr.L)]
    flat = rr.flatten_params(params)
    m = np.zeros_like(flat)
    v = np.zeros_like(flat)
    losses = []
    for step in range(1, rr.STEPS + 1):
        x, y = rr.make_data(rng)
        loss, G, q, r, h = rr.batch_grad(params, x, y)
        losses.append(loss)
        Gs = dict(p=[gain[l] * G["p"][l] for l in range(rr.L)],
                  q=[gain[l] * G["q"][l] for l in range(rr.L)],
                  b=[gain[l][:, None, None] * G["b"][l] for l in range(rr.L)],
                  c=G["c"])
        g = clip(rr.flat(rr.param_grad_transform(Gs, params), params))
        flat, m, v = adam(flat, g, m, v, step)
        params_next = rr.pack_params(params, flat)
        _, Gn, _, _, _ = rr.batch_grad(params_next, x, y, exact=True)
        Gn_t = rr.param_grad_transform(Gn, params_next)
        for l in range(rr.L):
            th = params["theta"][l]
            u = rr.sig(params["rho"][l])
            sigp = u * (1 - u)
            # polar transform of block i's unscaled (Gp, Gq)
            Gr0 = sigp * (np.cos(th) * G["p"][l] + np.sin(th) * G["q"][l])
            Gt0 = u * (-np.sin(th) * G["p"][l] + np.cos(th) * G["q"][l])
            dg = (-rr.LR) * (Gn_t["rho"][l] * Gr0
                             + Gn_t["theta"][l] * Gt0
                             + (Gn_t["b"][l] * G["b"][l]).sum(axis=(1, 2)))
            gain[l] = gain[l] - 1e-3 * dg
        params = params_next
    losses = np.asarray(losses)
    return dict(final=float(losses[-100:].mean()),
                finite=bool(np.all(np.isfinite(losses))))


def train_perbatch(seed):
    """Per-batch closed-form phase per block from the exact teacher."""
    params = rr.init_params(seed)
    rng = np.random.RandomState(1000 + seed)
    flat = rr.flatten_params(params)
    m = np.zeros_like(flat)
    v = np.zeros_like(flat)
    losses = []
    for step in range(1, rr.STEPS + 1):
        x, y = rr.make_data(rng)
        loss, G, q, r, h = rr.batch_grad(params, x, y)
        losses.append(loss)
        lam = rr.exact_lambda(params, q)
        phi = []
        for l in range(rr.L):
            z_re = q[l][:, :, :, 0]
            z_im = q[l][:, :, :, 1]
            lz_re = lam[l][:, :, :, 0]
            lz_im = lam[l][:, :, :, 1]
            z = z_re + 1j * z_im
            lz = lz_re + 1j * lz_im
            num = np.mean(lz * np.conj(z), axis=(0, 1))
            den = np.mean(np.abs(z) ** 2, axis=(0, 1)) + 1e-300
            phi.append(np.angle(num / den))
        G = rotate_G(G, phi)
        g = clip(rr.flat(rr.param_grad_transform(G, params), params))
        flat, m, v = adam(flat, g, m, v, step)
        params = rr.pack_params(params, flat)
    losses = np.asarray(losses)
    return dict(final=float(losses[-100:].mean()),
                finite=bool(np.all(np.isfinite(losses))))


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("HEADROOM sweep:")
    chosen = None
    for D in D_GRID:
        on, bp, hr = headroom(D)
        print(f"  D={D}: online {on:.4f}  bptt {bp:.4f}  headroom {hr:.2f}",
              flush=True)
        if hr >= 0.3:
            chosen = D
    if chosen is None:
        print("NO REGIME WITH HEADROOM >= 0.3 — stopping per protocol")
        git = subprocess.run(["git", "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
        with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
            json.dump(dict(git=git, headroom=None, note="no headroom"),
                      f)
        return
    print(f"chosen delay D={chosen}")
    rr.DELAY = chosen
    table, phis = {}, {}
    for seed in SEEDS:
        for arm in ["online", "bptt", "routePhi"]:
            out = train_arm(arm, seed)
            table.setdefault(arm, []).append(out["final"])
            if arm == "routePhi":
                phis[seed] = [p.copy() for p in out["phi"]]
            print(f"  seed {seed} {arm:<9s} final {out['final']:.4f} "
                  f"finite {out['finite']}", flush=True)
        # frozen learned rotation: deploy phis[seed] from scratch
        params = rr.init_params(seed)
        rng = np.random.RandomState(1000 + seed)
        flat = rr.flatten_params(params)
        m = np.zeros_like(flat)
        v = np.zeros_like(flat)
        losses = []
        for step in range(1, rr.STEPS + 1):
            x, y = rr.make_data(rng)
            loss, G, q, r, h = rr.batch_grad(params, x, y)
            losses.append(loss)
            G = rotate_G(G, phis[seed])
            g = clip(rr.flat(rr.param_grad_transform(G, params), params))
            flat, m, v = adam(flat, g, m, v, step)
            params = rr.pack_params(params, flat)
        fl = float(np.mean(losses[-100:]))
        table.setdefault("frozenPhi", []).append(fl)
        print(f"  seed {seed} frozenPhi final {fl:.4f}", flush=True)
        out = train_scalar(seed)
        table.setdefault("scalarGain", []).append(out["final"])
        print(f"  seed {seed} scalarGain final {out['final']:.4f}",
              flush=True)
        out = train_perbatch(seed)
        table.setdefault("perbatchOracle", []).append(out["final"])
        print(f"  seed {seed} perbatchOracle final {out['final']:.4f}",
              flush=True)

    med = {a: float(np.median(v)) for a, v in table.items()}
    finite_all = all(np.isfinite(sum(table.values(), [])))
    p3 = med["routePhi"] <= 0.7 * med["online"] and finite_all
    print("-" * 70)
    print(f"medians (D={chosen}): { {k: round(v, 4) for k, v in med.items()} }")
    print(f"P3 (routePhi <= 0.7x online): {'PASS — GENERALIZES' if p3 else 'FAIL — S5-specific'}")
    print(f"orient > gain check: scalarGain {med.get('scalarGain', float('nan')):.4f} vs routePhi {med['routePhi']:.4f}")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git, delay=chosen, per_arm=table, medians=med,
               p3=bool(p3))
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(doc, f, indent=2)
    print("wrote summary.json")


if __name__ == "__main__":
    main()
