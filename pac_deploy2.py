"""PAC deployment v2 — deploy the PHASE, not the law (review amendment).

pac_deploy.py (running) deploys the full K = 1/(1 - conj(a) beta) as
primary. Review amendment, logged as a post-hoc arm-set change per
registration discipline:

  1. factorize_w already falsified the magnitude channel (frozen-full
     worse than frozen-phase, magnitude redundant with Adam); D^-1 =
     conj(D) |D|^-2 says the action supplies orientation and the
     optimizer supplies positive gain. So the PRIMARY arm is the unit
     modulus e^{i arg K}; full K is secondary. Bonus: |conj(a) beta| ->
     1 makes |K| blow up; unit modulus removes the hazard.
  2. oracle-beta arm added: EMA lag under non-stationarity is a
     separate failure mode from the law being wrong. Oracle passes +
     EMA fails => estimation problem; both fail => law wrong.
  3. convention settled by P1's sign (+0.93, not -0.93): w = c*, used
     directly.
  4. cosine alignment with the exact gradient reported alongside loss
     (variance-explained is the wrong yardstick: ~90-99% of exact
     credit is linearly unpredictable from the causal stream — an
     information ceiling, not a method weakness).

Arms (paired seeds {0,1,2}, same init/streams, 1500 steps):
  pac_phase_oracle  w = e^{i arg K}, beta from the full batch each step
  pac_phase_ema     w = e^{i arg K}, beta by EMA gamma=0.05
  pac_full_oracle   w = K (magnitude channel control)
References online/routeA read from results/pac_deploy/summary.json
(identical deterministic protocol).

REGISTERED BAR (P4, fixed before running): pac_phase_oracle closes
>= 50% of the online -> routeA gap on median final loss.
< 20% => directionally right, not load-bearing.

Run:  python pac_deploy2.py
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

import trained_credit_gains as tcg
import co_variational_metric as cvm
from depth_law import STEPS
from decompose_w_final import make_data

SEEDS = [0, 1, 2]
ARMS = ["pac_phase_oracle", "pac_phase_ema", "pac_full_oracle"]
EMA_GAMMA = 0.05
CLIP_RHO = 0.95
ALIGN_EVERY = 200
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results", "pac_deploy2")
PAC1_SUMMARY = os.path.join(os.path.dirname(RESULTS_DIR),
                            "pac_deploy", "summary.json")


def setup():
    tcg.L, tcg.N, tcg.T, tcg.DELAY, tcg.M_IN, tcg.BATCH = \
        4, 16, 128, 50, 1, 32


def rho1_of(q):
    num = np.mean(q[1:] * np.conj(q[:-1]), axis=(0, 1))
    den = np.mean(np.abs(q[:-1]) ** 2, axis=(0, 1)) + 1e-300
    return num / den


def k_of(params, beta):
    w_full, w_phase = [], []
    for l in range(tcg.L):
        z = np.conj(params["a"][l]) * beta[l]
        mag = np.abs(z)
        over = mag > CLIP_RHO
        z = z.copy()
        z[over] *= CLIP_RHO / mag[over]
        K = 1.0 / (1.0 - z)
        w_full.append(K)
        w_phase.append(np.exp(1j * np.angle(K)))
    return w_full, w_phase


def train_arm(arm, seed):
    params = tcg.init_params(seed)
    rng = np.random.RandomState(1000 + seed)
    flat = tcg.flatten(params)
    m = np.zeros_like(flat)
    v = np.zeros_like(flat)
    beta = [np.zeros(tcg.N, np.complex128) for _ in range(tcg.L)]
    losses, aligns = [], []
    for step in range(1, STEPS + 1):
        x, y = make_data(rng)
        loss, G, q, r, h = cvm.batch_grad(params, x, y)
        losses.append(loss)
        if "oracle" in arm:
            beta_now = [rho1_of(ql) for ql in q]
        else:
            beta = [(1 - EMA_GAMMA) * b + EMA_GAMMA * rho1_of(ql)
                    for b, ql in zip(beta, q)]
            beta_now = beta
        w_full, w_phase = k_of(params, beta_now)
        w = w_full if arm == "pac_full_oracle" else w_phase
        G_use = cvm.scale_by_w(G, w)
        g = cvm.clip(tcg.flat_grads(G_use, params))
        if step % ALIGN_EVERY == 0:
            G_ex = cvm.exact_grad(params, x, y)
            g_ex = tcg.flat_grads(G_ex, params)
            aligns.append(float(np.vdot(g_ex, g).real
                                / (np.linalg.norm(g_ex)
                                   * np.linalg.norm(g) + 1e-300)))
        flat, m, v = cvm.adam(flat, g, m, v, step)
        params = tcg.pack(params, flat)
    losses = np.asarray(losses)
    return dict(final=float(losses[-100:].mean()),
                finite=bool(np.all(np.isfinite(losses))),
                align_late=float(np.mean(aligns[len(aligns) // 2:]))
                if aligns else None)


def main() -> None:
    setup()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(PAC1_SUMMARY) as f:
        ref = json.load(f)["medians"]
    online_med, routeA_med = ref["online"], ref["routeA"]
    table, aligns = {}, {}
    for seed in SEEDS:
        for arm in ARMS:
            out = train_arm(arm, seed)
            table.setdefault(arm, []).append(out["final"])
            aligns.setdefault(arm, []).append(out["align_late"])
            print(f"  seed {seed} {arm:<17s} final {out['final']:.4f} "
                  f"finite {out['finite']} align {out['align_late']:.3f}",
                  flush=True)
    med = {arm: float(np.median(table[arm])) for arm in ARMS}
    gap = online_med - routeA_med
    fracs = {arm: (online_med - med[arm]) / gap for arm in ARMS}
    win = fracs["pac_phase_oracle"] >= 0.5
    print("-" * 70)
    print(f"refs: online {online_med:.4f}  routeA {routeA_med:.4f}  "
          f"gap {gap:.4f}")
    print(f"medians: { {k: round(v, 4) for k, v in med.items()} }")
    print(f"fracs of gap closed: { {k: round(v, 2) for k, v in fracs.items()} }")
    print(f"align(late): { {k: round(float(np.median(v)), 3) for k, v in aligns.items()} }")
    print("paired deltas (online - arm): "
          f"{ {a: [round(ref['online'] - med[a], 4)] for a in ARMS} } (med)")
    print(f"BAR P4: phase_oracle >= 50%  ->  "
          f"{'CAUSAL LAW HOLDS' if win else 'NO WIN'}")
    if not win and fracs["pac_phase_ema"] < 0.2:
        print("reading: both fail -> the LAW is wrong, not the estimator")
    elif win and fracs["pac_phase_ema"] < 0.5 * fracs["pac_phase_oracle"]:
        print("reading: oracle passes, EMA lags -> estimation problem, "
              "law stands")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git,
               config=dict(steps=STEPS, seeds=SEEDS, ema_gamma=EMA_GAMMA,
                           clip_rho=CLIP_RHO,
                           note="phase-primary arm set (post-hoc review "
                                "amendment, logged); full K secondary"),
               refs=ref, per_arm=table, medians=med, fracs=fracs,
               align=aligns, win=bool(win))
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(doc, f, indent=2)
    print("wrote summary.json")


if __name__ == "__main__":
    main()
