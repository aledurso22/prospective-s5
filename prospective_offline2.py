"""STAGE A v2 — corrected placement, four diagnostics, dispersion.

PLACEMENT/POLE AUDIT (why v1 is invalid). The layer-l gradient is
G_l = sum_t conj(lam_l,t) Sa_l(t) with lam_l = D_l^{-1}(up_l) and
up_l = route(lam_{l+1}). By the RTRL identity the D_l^{-1} part is
EXACTLY handled by the Sa_l eligibility (this is why the top layer is
exact), so the cotangent that needs correction is up_l, whose temporal
operator D_{l+1}^{-1} lives at the UPPER site with pole a_{l+1}, BEFORE
routing. v1 filtered the already-routed q_l with pole a_l — wrong site,
wrong pole. Record v1 only as: "the pole-only first-order narrowband
approximation, incorrectly placed, fails" — nothing stronger.

Corrected recursion (per probe batch):

    u_{L-1} = q_{L-1}
    lam^_m  = P_m(u_m),   u_{m-1} = Re(B_m conj(lam^_m)),  m = L-1 .. 1

with surrogate P_m approximating D_m^{-1} at site m with pole a_m.
Gradient err for layer l:  err_l = u_l (l < L-1),  err_{L-1} = q_{L-1}
(top untouched — already exact). P = exact D^{-1} (future sum) MUST
reproduce BPTT to ~1e-10: that is the factorization/placement test.

THE FOUR DIAGNOSTICS (per seed, 8 probe batches, routeA-trained params):
  D1  exact-D^{-1} oracle factorization accuracy (cos vs BPTT; must be
      ~1.0, else our placement is still wrong — stop building on it);
  D2  per-mode complex scalar ORACLE ceiling on the ACTUAL gradient
      blocks: one complex z_{l,j} shared by the block group {Ga, Gb}
      (scale_by_w's conj(w) convention), complex least squares; report
      best achievable cos/rel + the z DISTRIBUTION;
  D3  analytic cstat arm:  lam^_j = c^stat_j u_j,
      c^stat_j = sum_k conj(a_j)^k rho^_j(k)  — the exact least-squares
      scalar predictor of D^{-1}u from u at the SIGNAL level
      (bandwidth-free; no narrowband assumption; estimated on base
      site signals — noted);
  D4  (separate script, single_mode_control.py) routing-free
      single-mode bandwidth control.
Plus: arg w_learned vs arg cstat vs arg w_oracle (weighted MRL), and
per-seed DISPERSION with paired differences (no pooled-only gates).

Run:  python prospective_offline2.py
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
from prospective_ops import apply_operator

SEEDS = [0, 1, 2, 3, 4]
BATCHES = 8
ARMS = ["base", "gain", "raw", "ema0.99", "matched", "oppphase",
        "cstat", "exactD"]
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results", "prospective_offline2")


def setup():
    tcg.L, tcg.N, tcg.T, tcg.DELAY, tcg.M_IN, tcg.BATCH = \
        4, 16, 128, 50, 1, 32


def route(X, m, params):
    """route from layer m to m-1:  Re(B_m conj(X))."""
    B = params["b"][m]                     # (N, N_in)
    return np.einsum("jm,tbj->tbm", B, np.conj(X)).real


def future_filter(X, a):
    """exact D^{-1}:  lam_t = sum_k conj(a)^k X_{t+k}  (oracle — uses the
    future; placement validation only)."""
    lam = np.zeros_like(X, dtype=np.complex128)
    acc = np.zeros(X.shape[1:], np.complex128)
    for t in range(X.shape[0] - 1, -1, -1):
        acc = X[t] + np.conj(a)[None, :] * acc
        lam[t] = acc
    return lam


def build_err_v2(q, params, arm, rho_map=None):
    """Corrected site recursion. rho_map: callable (m) -> rho for ema."""
    L = tcg.L
    u = [None] * L
    u[L - 1] = np.asarray(q[L - 1], np.complex128)
    err = [None] * L
    err[L - 1] = np.asarray(q[L - 1], np.complex128)   # top untouched
    for m in range(L - 1, 0, -1):
        a_m = np.asarray(params["a"][m])
        if arm == "exactD":
            lam = future_filter(u[m], a_m)
        elif arm == "cstat":
            lam = rho_map["cstat"][m][None, None, :] * u[m]
        elif arm == "ema0.99":
            lam = apply_operator(u[m], a_m, "ema", 0.99)
        else:
            lam = apply_operator(u[m], a_m, arm, rho_map)
        u[m - 1] = route(lam, m, params)
        err[m - 1] = u[m - 1]
    return err


def blocks_vec(G, l):
    return np.concatenate([G["a"][l].ravel(), G["b"][l].ravel()])


def train_routeA(seed):
    """Registered routeA protocol; returns (params, w_learned)."""
    params = tcg.init_params(seed)
    rng = np.random.RandomState(1000 + seed)
    w = [np.ones(tcg.N, np.complex128) for _ in range(tcg.L)]
    flat = tcg.flatten(params)
    m = np.zeros_like(flat)
    v = np.zeros_like(flat)
    for step in range(1, STEPS + 1):
        x, y = make_data(rng)
        loss, G = cvm.batch_grad(params, x, y)[:2]
        G_use = cvm.scale_by_w(G, w)
        g = cvm.clip(tcg.flat_grads(G_use, params))
        flat, m, v = cvm.adam(flat, g, m, v, step)
        params_next = tcg.pack(params, flat)
        G_next = cvm.exact_grad(params_next, x, y)
        gN = tcg.flat_grads(G_next, params_next)
        off = 0
        for l in range(tcg.L):
            th = params["theta"][l]
            u_mode = tcg.sig(params["rho"][l])
            sigp = u_mode * (1 - u_mode)
            A = G["a"][l] * np.exp(1j * th)
            Gb = G["b"][l]
            M_ = Gb.shape[1]
            gN_rho = gN[off:off + tcg.N]
            gN_theta = gN[off + tcg.N:off + 2 * tcg.N]
            gN_bre = gN[off + 2 * tcg.N:
                        off + 2 * tcg.N + tcg.N * M_].reshape(tcg.N, M_)
            gN_bim = gN[off + 2 * tcg.N + tcg.N * M_:
                        off + 2 * tcg.N + 2 * tcg.N * M_].reshape(tcg.N, M_)
            off += 2 * tcg.N + 2 * tcg.N * M_
            du = (gN_rho * sigp * A.real
                  + gN_theta * (-u_mode) * A.imag
                  + (gN_bre * Gb.real - gN_bim * Gb.imag).sum(axis=1))
            dv = (gN_rho * sigp * A.imag
                  + gN_theta * (u_mode) * A.real
                  + (gN_bre * Gb.imag + gN_bim * Gb.real).sum(axis=1))
            w[l] = w[l] - cvm.LR_M * (-cvm.LR) * (du + 1j * dv)
        params = params_next
        if step % 500 == 0:
            print(f"    s{seed} step {step}: loss {loss:.4f}", flush=True)
    return params, w


def autocorr_cstat(q_site, a, K=60):
    """c^stat_j = sum_k conj(a_j)^k rho_j(k) on the site's signal
    (q_site: list of per-batch (T, B, N) arrays)."""
    N = q_site[0].shape[2]
    num = np.zeros((K + 1, N), np.complex128)
    den = np.zeros(N)
    for X in q_site:
        Xc = np.asarray(X, np.complex128)
        den += np.mean(np.abs(Xc) ** 2, axis=(0, 1))
        for k in range(K + 1):
            prod = (Xc if k == 0 else Xc[k:] * np.conj(Xc[:-k]))
            num[k] += prod.mean(axis=(0, 1))
    rho = num / (den[None, :] + 1e-30)
    ks = np.arange(K + 1)[:, None]
    return np.sum(np.conj(a)[None, :] ** ks * rho, axis=0)


def main() -> None:
    setup()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    L = tcg.L
    agg = {arm: [] for arm in ARMS}
    oracle_rows = []
    arg_rows = []
    placement_ok = True
    for seed in SEEDS:
        print(f"seed {seed}: routeA retrain...", flush=True)
        params, w_learned = train_routeA(seed)
        prng = np.random.RandomState(777000 + seed)
        # ---- pass 1: gather per-batch structures + cstat autocorrelation
        packs = []
        q_acc = [[] for _ in range(L)]
        for _ in range(BATCHES):
            x, y = make_data(prng)
            h, yhat = tcg.forward(params, x)
            r = yhat - y
            r[:tcg.DELAY] = 0.0
            q = tcg.spatial_q(params, h, r)
            Sa, Sb = tcg.sensitivities(params, h, x)
            lam = tcg.exact_lambda(params, q)
            G_ex = tcg.assemble(params, h, x, r, lam, Sa, Sb,
                                direct=True)
            packs.append((x, y, h, r, q, Sa, Sb, G_ex))
            for m in range(L):
                q_acc[m].append(np.asarray(q[m], np.complex128))
        cstat = {m: autocorr_cstat(q_acc[m], np.asarray(params["a"][m]))
                 for m in range(L)}
        rho_map = {"cstat": cstat}

        # ---- pass 2: arms
        for arm in ARMS:
            cos_f, rel_f, fin, amax = [], [], True, 0.
            per_layer_cs = [[] for _ in range(L)]
            for (x, y, h, r, q, Sa, Sb, G_ex) in packs:
                err = build_err_v2(q, params, arm, rho_map)
                G_sur = tcg.assemble(params, h, x, r, err, Sa, Sb)
                gs = np.concatenate([blocks_vec(G_sur, l)
                                     for l in range(L)])
                ge = np.concatenate([blocks_vec(G_ex, l)
                                     for l in range(L)])
                fin &= bool(np.all(np.isfinite(gs)))
                amax = max(amax, float(np.max(np.abs(gs))))
                for l in range(L):
                    gl = blocks_vec(G_sur, l)
                    el = blocks_vec(G_ex, l)
                    per_layer_cs[l].append(float(
                        np.abs(np.vdot(el, gl))
                        / (np.linalg.norm(el) * np.linalg.norm(gl)
                           + 1e-30)))
                cos_f.append(float(np.abs(np.vdot(ge, gs))
                                   / (np.linalg.norm(ge)
                                      * np.linalg.norm(gs) + 1e-30)))
                rel_f.append(float(np.linalg.norm(gs - ge)
                                   / (np.linalg.norm(ge) + 1e-30)))
            per_layer = [float(np.median(cs)) for cs in per_layer_cs]
            agg[arm].append(dict(
                cos=float(np.median(cos_f)),
                rel=float(np.median(rel_f)),
                per_layer=per_layer, finite=fin, amax=amax))

        # ---- placement/factorization assertion (exactD ~ BPTT)
        d1 = float(np.median([a["cos"] for a in agg["exactD"][-1:]]))
        d1rel = float(np.median([a["rel"] for a in agg["exactD"][-1:]]))
        print(f"  D1 exactD factorization: cos {d1:.6f}  rel {d1rel:.2e}",
              flush=True)
        if d1 < 0.999999:
            placement_ok = False
            print("  PLACE FACTORIZATION FAILED — placement still wrong",
                  flush=True)

        # ---- D2: per-mode scalar oracle ceiling (per batch)
        zs = []
        cos_o, rel_o = [], []
        cos_ol = [[] for _ in range(L)]
        for (x, y, h, r, q, Sa, Sb, G_ex) in packs:
            G_on = tcg.assemble(params, h, x, r, q, Sa, Sb)
            z_map = []
            G_or = dict(a=[], b=[], c=G_on["c"])
            for l in range(L):
                za = []
                for j in range(tcg.N):
                    go = np.concatenate([[G_on["a"][l][j]],
                                         G_on["b"][l][j].ravel()])
                    ge = np.concatenate([[G_ex["a"][l][j]],
                                         G_ex["b"][l][j].ravel()])
                    z = (np.vdot(go, ge)
                         / (np.vdot(go, go) + 1e-30))
                    za.append(z)
                    zs.append(z)
                za = np.asarray(za)
                z_map.append(za)
                G_or["a"].append(za * G_on["a"][l])
                G_or["b"].append(za[:, None] * G_on["b"][l])
            gs = np.concatenate([
                np.concatenate([G_or["a"][l].ravel(),
                                G_or["b"][l].ravel()]) for l in range(L)]
                + [np.ravel(G_or["c"])])
            ge = np.concatenate([blocks_vec(G_ex, l) for l in range(L)]
                                + [np.ravel(G_ex["c"])])
            cos_o.append(float(np.abs(np.vdot(ge, gs))
                               / (np.linalg.norm(ge) * np.linalg.norm(gs)
                                  + 1e-30)))
            rel_o.append(float(np.linalg.norm(gs - ge)
                               / (np.linalg.norm(ge) + 1e-30)))
            for l in range(L):
                gl = np.concatenate([G_or["a"][l].ravel(),
                                     G_or["b"][l].ravel()])
                el = blocks_vec(G_ex, l)
                cos_ol[l].append(float(np.abs(np.vdot(el, gl))
                                       / (np.linalg.norm(el)
                                          * np.linalg.norm(gl) + 1e-30)))
        zs = np.asarray(zs)
        oracle_rows.append(dict(
            seed=seed,
            cos=float(np.median(cos_o)),
            rel=float(np.median(rel_o)),
            per_layer=[float(np.median(c)) for c in cos_ol],
            z_abs_med=float(np.median(np.abs(zs))),
            z_abs_p90=float(np.percentile(np.abs(zs), 90)),
            z_arg_med=float(np.median(np.angle(zs))),
            z_arg_p90=float(np.percentile(np.abs(np.angle(zs)), 90))))
        # ---- arg comparisons (weighted MRL): learned vs cstat vs oracle
        for l in range(L):
            # oracle z aggregated per (l, j) over batches (median phasor)
            # cstat per (l, j): from cstat dict; learned: w_learned
            zw = w_learned[l] / np.maximum(np.abs(w_learned[l]), 1e-30)
            zc = cstat[l] / np.maximum(np.abs(cstat[l]), 1e-30)
            mrl_lc = float(np.abs(np.mean(zw * np.conj(zc))))
            arg_rows.append(dict(seed=seed, layer=l,
                                 mrl_learned_vs_cstat=mrl_lc))
        print(f"  arms (pooled cos): "
              + "  ".join(f"{a} {agg[a][-1]['cos']:.3f}" for a in ARMS),
              flush=True)
        print(f"  D2 oracle ceiling: cos {oracle_rows[-1]['cos']:.3f}  "
              f"rel {oracle_rows[-1]['rel']:.3f}  per-layer "
              f"{[round(c, 3) for c in oracle_rows[-1]['per_layer']]}",
              flush=True)

    # ---- aggregate with dispersion ----
    med = {}
    for arm in ARMS:
        med[arm] = dict(
            cos=[round(a["cos"], 4) for a in agg[arm]],
            cos_med=float(np.median([a["cos"] for a in agg[arm]])),
            rel_med=float(np.median([a["rel"] for a in agg[arm]])),
            per_layer=[float(np.median([a["per_layer"][l]
                                        for a in agg[arm]]))
                       for l in range(L)],
            finite=all(a["finite"] for a in agg[arm]))
    print("-" * 78)
    print("medians over seeds (pooled cos; per-seed in brackets):")
    for arm in ARMS:
        print(f"  {arm:<9s} {med[arm]['cos_med']:.3f}  {med[arm]['cos']}  "
              f"per-layer {[round(c, 3) for c in med[arm]['per_layer']]}")
    print("paired differences vs gain (per seed):")
    for arm in ARMS:
        if arm in ("base", "gain"):
            continue
        diffs = [agg[arm][s]["cos"] - agg["gain"][s]["cos"]
                 for s in range(len(SEEDS))]
        print(f"  {arm:<9s} {['%+.3f' % d for d in diffs]}")
    o_med = float(np.median([o["cos"] for o in oracle_rows]))
    print(f"D2 oracle ceiling (median pooled cos): {o_med:.3f}  "
          f"per-seed {[round(o['cos'], 3) for o in oracle_rows]}")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git,
               config=dict(steps=STEPS, seeds=SEEDS, batches=BATCHES),
               placement_ok=bool(placement_ok),
               per_arm=med, oracle=oracle_rows, arg_rows=arg_rows)
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(doc, f, indent=2)
    print("wrote summary.json")


if __name__ == "__main__":
    main()
