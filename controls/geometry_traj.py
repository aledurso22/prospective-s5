"""Shared trajectory-logging runner for the modal-geometry audit (G0–G8).

One loop, all causal arms, frozen PC0 protocol (delayed copy D=50/T=128,
L=4, N=16, batch=32, STEPS=1500, LR=LR_M=1e-3, Adam theta / clip 1.0).
Logging is read-only and does not change numerics: the pc0/pcphase/online
arms reproduce their stored finals BITWISE (gated in the callers).

Arms:
  online       w = 1 (reference)
  pc0          Cartesian complex w, plain-SGD meta (frozen PC0)
  pcphase      pc0 + per-step unit-modulus projection (frozen C1)
  real         per-mode real w, du only (frozen factorial semantics)
  polar        FREE log-polar w = exp(alpha + i phi): (alpha, phi)
               optimized directly via the chain-rule map from PC0's
               (du, dv): r_alpha = du*u + dv*v, r_phi = u*dv - v*du.
               No gauge fixing. Isolates the coordinate-conditioning
               correction (eta_eff = eta/rho^2 removed by construction).
  polar_gauge  polar + layerwise gauge fix: r_alpha demeaned per layer
               before the alpha update (sum_j alpha~_j = 0 preserved;
               init alpha = 0 satisfies it). phi updates identical.
  pc0_adam     Cartesian complex w EXACTLY as pc0, but the plain-SGD
               meta step is replaced by Adam on g_hat := (-LR)*c with
               lr = LR_M (b1 .9, b2 .999, eps 1e-8, mirroring cvm.adam;
               one fixed LR, no sweep).

Exact-teacher probes (offline diagnostic, audited): every CKPT_EVERY
steps, r_exact is built from the SAME stored previous blocks and the
EXACT gradient at the current params on the current batch; logs
cos(r_causal, r_exact) and ||eps|| = ||r_causal - r_exact|| per layer.
These exact_grad calls are counted (rp.BPTT_CALLS) and are diagnostic
only — they never enter any update.
"""
from __future__ import annotations

import numpy as np

from toyrig import ssm_rig as tcg
from toyrig import route_a as cvm
from toyrig.train_cell import STEPS
from toyrig.probes import make_data
from diagnostics.prospective_kappa import chain_c_stored

LR, LR_M = cvm.LR, cvm.LR_M
CKPT_EVERY = 50


def setup():
    tcg.L, tcg.N, tcg.T, tcg.DELAY, tcg.M_IN, tcg.BATCH = \
        4, 16, 128, 50, 1, 32


def _cos_real(a, b):
    """Real-embedded cosine for complex vectors."""
    return float(np.real(np.vdot(a, b))
                 / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def train_arm(arm, seed, clip=cvm.CLIP, exact_probes=True,
              ckpt_every=CKPT_EVERY):
    """Frozen-protocol run with full logging. Returns (out, traj) where
    traj holds per-step arrays; out holds final_loss/finite."""
    params = tcg.init_params(seed)
    rng = np.random.RandomState(1000 + seed)
    L, N = tcg.L, tcg.N
    w = [np.ones(N, np.complex128) for _ in range(L)]
    alpha = [np.zeros(N) for _ in range(L)]       # polar arms
    phi = [np.zeros(N) for _ in range(L)]
    mw = [np.zeros(N, np.complex128) for _ in range(L)]   # pc0_adam
    vwre = [np.zeros(N) for _ in range(L)]         # per-component 2nd
    vwim = [np.zeros(N) for _ in range(L)]         # moments (real Adam)
    prev = None
    flat = tcg.flatten(params)
    m = np.zeros_like(flat)
    v = np.zeros_like(flat)
    losses = np.empty(STEPS)
    w_tr = np.zeros((STEPS, L, N), np.complex128)
    gnorm = np.zeros((STEPS, L, N))
    rnorm = np.zeros((STEPS, L, N))
    clip_fire = np.zeros(STEPS, bool)
    preclip = np.zeros(STEPS)
    ex_cos, ex_eps = [], []
    for step in range(1, STEPS + 1):
        x, y = make_data(rng)
        loss, G = cvm.batch_grad(params, x, y)[:2]
        losses[step - 1] = loss
        h_n = tcg.flat_grads(G, params)

        if arm == "polar" or arm == "polar_gauge":
            w = [np.exp(al + 1j * ph) for al, ph in zip(alpha, phi)]

        if prev is not None and arm != "online":
            Gp, th_all, u_all, sig_all = prev
            c = chain_c_stored(Gp, th_all, u_all, sig_all, h_n)
            for l in range(L):
                rnorm[step - 1, l] = np.abs(c[l])
            if arm == "pc0":
                w = [wl - LR_M * (-LR) * cl for wl, cl in zip(w, c)]
            elif arm == "pcphase":
                w = [wl - LR_M * (-LR) * cl for wl, cl in zip(w, c)]
                w = [wl_ / np.maximum(np.abs(wl_), 1e-12) for wl_ in w]
            elif arm == "real":
                w = [np.real(wl - LR_M * (-LR) * cl.real) + 0j
                     for wl, cl in zip(w, c)]
            elif arm == "pc0_adam":
                t = step - 1                       # meta step index
                for l in range(L):
                    gl = (-LR) * c[l]              # the descended gradient
                    mw[l] = 0.9 * mw[l] + 0.1 * gl
                    vwre[l] = 0.999 * vwre[l] + 0.001 * gl.real ** 2
                    vwim[l] = 0.999 * vwim[l] + 0.001 * gl.imag ** 2
                    upd_re = (mw[l].real / (1 - 0.9 ** t)) / (
                        np.sqrt(vwre[l] / (1 - 0.999 ** t)) + 1e-8)
                    upd_im = (mw[l].imag / (1 - 0.9 ** t)) / (
                        np.sqrt(vwim[l] / (1 - 0.999 ** t)) + 1e-8)
                    w[l] = w[l] - LR_M * (upd_re + 1j * upd_im)
            else:                                # polar arms
                for l in range(L):
                    u_, v_ = w[l].real, w[l].imag
                    du, dv = c[l].real, c[l].imag
                    r_al = du * u_ + dv * v_
                    if arm == "polar_gauge":
                        r_al = r_al - r_al.mean()
                    r_ph = u_ * dv - v_ * du
                    alpha[l] = alpha[l] - LR_M * (-LR) * r_al
                    phi[l] = phi[l] - LR_M * (-LR) * r_ph
                w = [np.exp(al + 1j * ph) for al, ph in zip(alpha, phi)]

            # exact-teacher probe (diagnostic, audited)
            if exact_probes and (step % ckpt_every == 0):
                h_ex = tcg.flat_grads(cvm.exact_grad(params, x, y),
                                      params)
                c_ex = chain_c_stored(Gp, th_all, u_all, sig_all, h_ex)
                ex_cos.append([_cos_real(cl, ce)
                               for cl, ce in zip(c, c_ex)])
                ex_eps.append([float(np.linalg.norm(cl - ce))
                               for cl, ce in zip(c, c_ex)])

        for l in range(L):
            gnorm[step - 1, l] = np.sqrt(
                np.abs(G["a"][l]) ** 2
                + (np.abs(G["b"][l]) ** 2).sum(axis=1))
        w_tr[step - 1] = np.asarray(w)

        G_use = cvm.scale_by_w(G, w)
        g_flat = tcg.flat_grads(G_use, params)
        n0 = float(np.linalg.norm(g_flat))
        preclip[step - 1] = n0
        clip_fire[step - 1] = n0 > clip
        g = cvm.clip(g_flat) if clip <= 1e10 else g_flat * (clip / n0)
        flat, m, v = cvm.adam(flat, g, m, v, step)
        prev = (dict(a=[ga.copy() for ga in G["a"]],
                     b=[gb.copy() for gb in G["b"]]),
                [th.copy() for th in params["theta"]],
                [tcg.sig(params["rho"][l]) for l in range(L)],
                [tcg.sig(params["rho"][l]) * (1 - tcg.sig(params["rho"][l]))
                 for l in range(L)])
        params = tcg.pack(params, flat)
        if step % 500 == 0:
            print(f"    {arm} s{seed} step {step}: loss {loss:.4f}",
                  flush=True)

    traj = dict(losses=losses, w=w_tr, gnorm=gnorm, rnorm=rnorm,
                clip_fire=clip_fire, preclip=preclip,
                ex_cos=np.asarray(ex_cos), ex_eps=np.asarray(ex_eps),
                ckpt_every=ckpt_every)
    out = dict(final_loss=float(losses[-100:].mean()),
               finite=bool(np.all(np.isfinite(losses))))
    return out, traj


def polar_chain_fd_gate(tol=1e-4):
    """FD gate for the polar chain map. For random (params, batch, w):
    F(w) = <h, flat(scale_by_w(G, w))>; the analytic (du, dv) from
    chain_c_stored must satisfy, with w = u + i v_ = rho e^{i phi}:
        dF/dalpha = du*u + dv*v      (alpha = log rho)
        dF/dphi   = u*dv - v*du
    checked against central finite differences of F along alpha/phi.
    """
    setup()
    rng = np.random.RandomState(7)
    params = tcg.init_params(3)
    x, y = make_data(rng)
    loss, G = cvm.batch_grad(params, x, y)[:2]
    h_n = tcg.flat_grads(G, params)
    prevb = (dict(a=[ga.copy() for ga in G["a"]],
                  b=[gb.copy() for gb in G["b"]]),
             [th.copy() for th in params["theta"]],
             [tcg.sig(params["rho"][l]) for l in range(tcg.L)],
             [tcg.sig(params["rho"][l]) * (1 - tcg.sig(params["rho"][l]))
              for l in range(tcg.L)])
    c = chain_c_stored(*prevb, h_n)
    w0 = [1.3 * np.exp(0.4j * np.linspace(-1, 1, tcg.N))
          for _ in range(tcg.L)]

    def F(wlist):
        Gw = cvm.scale_by_w(G, wlist)
        return float(np.dot(h_n, tcg.flat_grads(Gw, params)))

    errs, ans = [], []
    eps = 1e-3      # F ~ 3e5 with dF/d(alpha,phi) spanning 1e-6..7e4:
                    # central FD is exact for the w-linear part; per-mode
                    # agreement is assessed relative to the MODE's own
                    # derivative magnitude (a floor of the median avoids
                    # meaningless rel-error blowups at near-zero modes)
    for l in range(tcg.L):
        u_, v_ = w0[l].real, w0[l].imag
        du, dv = c[l].real, c[l].imag
        al0, ph0 = np.log(np.abs(w0[l])), np.angle(w0[l])
        for j in range(tcg.N):
            for name, analytic in (
                    ("alpha", du[j] * u_[j] + dv[j] * v_[j]),
                    ("phi", u_[j] * dv[j] - v_[j] * du[j])):
                wp = [wl.copy() for wl in w0]
                wm = [wl.copy() for wl in w0]
                if name == "alpha":
                    wp[l][j] = np.exp(al0[j] + eps + 1j * ph0[j])
                    wm[l][j] = np.exp(al0[j] - eps + 1j * ph0[j])
                else:
                    wp[l][j] = np.exp(al0[j] + 1j * (ph0[j] + eps))
                    wm[l][j] = np.exp(al0[j] + 1j * (ph0[j] - eps))
                fd = (F(wp) - F(wm)) / (2 * eps)
                errs.append(abs(fd - analytic))
                ans.append(abs(analytic))
    floor = float(np.median(ans))
    worst = float(np.max(np.asarray(errs) / (np.asarray(ans) + floor)))
    print(f"polar chain FD gate: worst scaled rel {worst:.2e}  "
          f"{'PASS' if worst < tol else 'FAIL'}")
    return worst < tol
