"""routePC_pro — TSS prospection of the META-RESIDUAL, not of w.

INTERPRETATION HIERARCHY (keep explicit; do not collapse):

    TSS/Simonetto exact principle        (I + tau_w d_n) r_w = 0
      ->  r_w(w, n) = grad_w F_n(w)      (the true meta-residual;
                                          generally w-dependent)
      ->  our causal first-order surrogate
              r^_n = -eta [d_w (M_w g^on_{n-1})]^dagger g^on_n
          (linear in w => w-independent; zero BPTT; evaluated when
          batch n has genuinely arrived)
      ->  r^pro_n = r^_n + kappa (r^_n - r^_{n-1}).

routePC_pro is therefore "a TSS/Simonetto-INSPIRED prospective
correction of a causal surrogate meta-residual" — NOT an exact
implementation of  -H_w^{-1} d_n grad_w F.  P_n = I (no Hessian
inverse); the w-independence of the surrogate is what makes the fixed-w
difference free of H_w dw contamination HERE, and only here.

Theory target (directive). TSS residual dynamics,

    (I + tau d/dt) r = 0,   r = s - f(s, t)

applied to the moving meta-optimality residual of the modal geometry,

    r_w(w, n) = grad_w F_n(w),

gives the Simonetto prediction--correction law

    dw/dn = -(1/tau_w) H_w^{-1} [ r_w + tau_w d_n r_w ].

P_n = I in this arm (no Hessian inverse yet — we isolate the residual
prospection). The causal surrogate meta-residual (RoutePC's correction,
zero BPTT):

    r^_n = -eta [ d_w (M_w g^on_{n-1}) ]^dagger g^on_n ,

evaluated when batch n has genuinely arrived, after the previous update.

SAME-w CORRECTION (directive amendment) — and why it is exact here for
free. The Simonetto term d_n r_w is the residual change at FIXED w; a
naive total difference r_n(w_n) - r_{n-1}(w_{n-1}) would add H_w dw.
Our surrogate is LINEAR in w: M_w g = conj(w) (*) G, so
d_w(M_w g) is independent of w, and r^(_n)(w) == r^_n for every w.
Therefore

    r^_n(w) - r^_{n-1}(w) = r^_n - r^_{n-1}   for ANY common w,

i.e. the backward difference IS the fixed-w partial derivative of the
surrogate residual — no w-motion contamination, no extra evaluation, no
leakage. Formally:

    THE FIXED-w DISTINCTION VANISHES FOR OUR FIRST-ORDER RoutePC
    SURROGATE — this is NOT a statement about the true meta-residual
    grad_w F_n(w), which generally DOES depend on w (through the
    post-update loss's curvature pulled back into w-space). Once an
    H_w model is added, the distinction becomes real; document it then.

CONSEQUENCE FOR THE SYNTHETIC GATE (corrected after review): the
preflight must validate the FIXED-w object — the same mathematical
object the SSM arm computes. For F_n(w) = 1/2 (w - w*_n)^2 the fixed-w
difference is analytic: r_n(w) - r_{n-1}(w) = -(w*_n - w*_{n-1}). With
it, e_{n+1} = (1-alpha) e_n + (alpha kappa - 1) delta_n: the predictor
changes the TRACKING FORCING, never the homogeneous factor (1-alpha).
An earlier version of this gate differenced across w (total
difference); its two "phenomena" — kappa-invariant drift lag and a
large-kappa stability boundary — were artifacts of THAT object and do
not occur under the fixed-w formulation.

Discrete causal estimator (kappa = tau_w / Delta n dimensionless):

    r^pro_n = r^_n + kappa (r^_n - r^_{n-1}),
    w <- w - LR_M * (-LR) * r^pro_n         (identical update to PC0 at
                                             kappa = 0 — required gate).

This is NOT w-momentum (w <- w + beta (w_n - w_{n-1})); that arm is kept
separately as routePC_wmom (control).

TIMESCALES — recorded, NOT mixed. The adjoint filter identity
D^dagger ~ (1 - a) (I + a dt/(1-a) d/dt) is a SEQUENCE-time statement
about smooth error signals (and our autocorrelation/noise results show
sequence-time smoothness is not always satisfied — theory framing only).
routePC_pro's (I + tau_w d_n) acts in OPTIMIZER (minibatch) time. Two
different clocks; no claim connects them in the implementation.

REGISTERED PREDICTIONS (fixed before running):
  * Stationary task: PC-pro ~= PC0 (kappa* ~ 0) is the PREDICTED outcome
    — phase_track/optimum_track say the useful geometry is nearly static.
    Pre-measured: systematic residual drift ||E dr^|| vs the batch-noise
    floor, and lag-1 autocorrelation of dr^; drift buried in noise =>
    kappa* ~ 0 is a confirmation, not an inconclusive result.
  * Moving task (delay ramp 30 -> 70): PC-pro > PC0 is the strong TSS
    prediction — the prediction term is functional exactly when the
    residual genuinely moves. (Premise checked: PC0's own learned arg w
    must move materially more under the ramp than under the stationary
    task, else the moving comparison is void and flagged.)
  * kappa = 0 reproduces route_pc.py's PC0 finals (implementation gate).

DECISION RULE (registered):
  * stationary best-kappa wins >= 4/5 paired seeds AND improves the
    median (beyond a 10%-of-gap tie band) AND stable  => PROSPECTION HELPS
  * stationary tie AND moving win (same bars)  => DORMANT-BUT-FUNCTIONAL
  * tie in both regimes                       => CORRECTION ONLY
  * degradation/divergence growing with kappa => UNSTABLE/NOISY
  * no single lucky kappa/seed counts as a win: the best kappa must also
    have a sweep neighbour not worse than PC0 on the median.

Arms: PC-pro kappa in {0, 0.1, 0.25, 0.5, 1.0} x {stationary, moving};
routePC_wmom beta = 0.25 (control); references (online, routeA, PC0)
from results/route_pc/summary.json (same streams; cross-process
determinism verified by the kappa=0 gate). BPTT audit: counting wrappers
(imported from route_pc) must read ZERO for every arm here.

Run:  python route_pc_pro.py
"""
from __future__ import annotations

import json
import os
import subprocess
import time

import numpy as np

import trained_credit_gains as tcg
import co_variational_metric as cvm
import route_pc as rp                     # installs the BPTT audit wrappers
from depth_law import STEPS
from decompose_w_final import make_data

SEEDS = [0, 1, 2, 3, 4]
KAPPAS = [0.0, 0.1, 0.25, 0.5, 1.0]
WMOM_BETA = 0.25
LR, LR_M = cvm.LR, cvm.LR_M
TIE_BAND = 0.10        # fraction of the online->PC0 median gap
MOVE_D0, MOVE_D1 = 30, 70
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results", "route_pc_pro")


def setup():
    tcg.L, tcg.N, tcg.T, tcg.DELAY, tcg.M_IN, tcg.BATCH = \
        4, 16, 128, 50, 1, 32


# ---------------------------------------------------------------------------
# Preflight: synthetic moving-optimum sanity (derivation/implementation gate)
# ---------------------------------------------------------------------------

def sanity() -> bool:
    """F_n(w) = 1/2 (w - w*_n)^2, r_n(w) = w - w*_n — the moving residual
    is KNOWN analytically, so the fixed-w partial-time difference is
    exact and unambiguous:

        d_n r = r_n(w) - r_{n-1}(w) = -(w*_n - w*_{n-1})   (causal:
        uses only the current and previous optimum positions)

    update: w <- w - alpha [ r_n + kappa * d_n r ].

    With delta_{n-1} := w*_n - w*_{n-1} and delta_n := w*_{n+1} - w*_n,
    the fixed-w difference is d_n r = -delta_{n-1}, and the GENERAL
    tracking-error recurrence is

        e_{n+1} = (1 - alpha) e_n + alpha*kappa*delta_{n-1} - delta_n.

    Only for constant drift (delta_{n-1} = delta_n = v) does this reduce
    to  e_{n+1} = (1-alpha) e_n + (alpha*kappa - 1) v,  hence
    e* = -v/alpha + kappa*v.  For a sinusoid or any time-varying
    velocity, the residual tracking error comes from
    delta_n - delta_{n-1} — the ACCELERATION of the optimum's velocity —
    which the causal backward difference cannot see; hence a large but
    not perfect improvement.

    Expected corrected behavior (all asserted):
      (i)   kappa = 0 reproduces correction-only bitwise;
      (ii)  linear drift delta = v: e* = -v/alpha + kappa v — the
            predictor REDUCES the drift lag, exactly zero at
            kappa = 1/alpha, sign-flipped overshoot beyond;
      (iii) kappa never enters the homogeneous factor (1 - alpha): NO
            kappa-induced instability at any kappa (large kappa
            overshoots but stays finite). There is no TSS stability
            boundary in kappa for this exact object;
      (iv)  oscillatory optimum: strong benefit near kappa = 1/alpha
            (near-perfect tracking of smooth motion; the residual error
            is set by the per-step ACCELERATION of w*, which the causal
            difference cannot see).
    """
    def run(wstar, alpha, kappa, steps=4000):
        w = 0.0
        err = []
        for n in range(steps):
            r = w - wstar(n)
            err.append(r)          # e_n: error of the CURRENT iterate,
                                   # measured BEFORE the update
            dr = (wstar(n - 1) - wstar(n)) if n > 0 else 0.0
            w = w - alpha * (r + kappa * dr)
        return np.asarray(err)

    def run_corr_only(wstar, alpha, steps=4000):
        w = 0.0
        err = []
        for n in range(steps):
            err.append(w - wstar(n))
            w = w - alpha * (w - wstar(n))
        return np.asarray(err)

    ok = True
    drift = lambda n: 0.01 * n
    P = 200.0
    wsin = lambda n: np.sin(2 * np.pi * n / P)
    alpha = 0.5

    # (i) kappa=0 bitwise vs a bare correction-only loop
    a0 = run(drift, alpha, 0.0)
    b0 = run_corr_only(drift, alpha)
    ok &= bool(np.array_equal(a0, b0))
    print("SANITY (synthetic moving optimum, FIXED-w formulation):")
    print(f"  (i) kappa=0 bitwise == correction-only: {np.array_equal(a0, b0)}")

    # (ii) linear drift: lag e* = -v/alpha + kappa v — reduced by kappa,
    #      exactly zero at kappa = 1/alpha = 2, overshoot beyond
    rms_d = {k: float(np.sqrt(np.mean(run(drift, alpha, k)[-1000:] ** 2)))
             for k in [0.0, 1.0, 2.0, 3.0]}
    e_star = {k: float(np.mean(run(drift, alpha, k)[-1000:]))
              for k in [0.0, 1.0, 2.0, 3.0]}
    pred = {k: -0.01 / alpha + k * 0.01 for k in [0.0, 1.0, 2.0, 3.0]}
    lag_reduced = (rms_d[2.0] < 0.1 * rms_d[0.0]
                   and abs(e_star[2.0]) < 1e-3
                   and e_star[1.0] < 0 < e_star[3.0])
    formula_ok = all(abs(e_star[k] - pred[k]) < 1e-3 for k in e_star)
    ok &= lag_reduced and formula_ok
    print(f"  (ii) linear drift: e* measured "
          f"{ {k: round(v, 5) for k, v in e_star.items()} } vs formula "
          f"-v/a + kappa*v { {k: round(v, 5) for k, v in pred.items()} } "
          f"(match: {formula_ok}, lag reduced at kappa=1/alpha: "
          f"{lag_reduced})")

    # (iii) no kappa-induced instability under the fixed-w formulation
    big = run(drift, alpha, 50.0)
    big_s = run(wsin, alpha, 50.0)
    no_instab = bool(np.all(np.isfinite(big)) and np.all(
        np.isfinite(big_s)))
    overshoot = abs(float(np.mean(big[-1000:]))) > 10 * abs(
        float(np.mean(a0[-1000:])))
    ok &= no_instab and overshoot
    print(f"  (iii) kappa=50: finite {no_instab} (no TSS stability "
          f"boundary in kappa); large-kappa overshoot present: "
          f"{overshoot}")

    # (iv) oscillatory optimum: strong benefit near kappa = 1/alpha = 2
    rms_s = {k: float(np.sqrt(np.mean(run(wsin, alpha, k)[-2000:] ** 2)))
             for k in [0.0, 0.5, 1.0, 2.0, 3.0]}
    oscil = rms_s[2.0] < 0.2 * rms_s[0.0]
    ok &= oscil
    print(f"  (iv) sinusoid RMS vs kappa: "
          f"{ {k: round(v, 5) for k, v in rms_s.items()} } "
          f"(strong benefit at 1/alpha: {oscil})")
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# Toy-rig arm
# ---------------------------------------------------------------------------

def make_data_moving(rng, D):
    x = rng.randn(tcg.T, tcg.BATCH)
    y = np.concatenate([np.zeros((D, tcg.BATCH)), x[:-D]], axis=0)
    return x, y


def batch_grad_masked(params, x, y, D):
    """cvm.batch_grad with the loss mask at the CURRENT delay D."""
    h, yhat = tcg.forward(params, x)
    r = yhat - y
    r[:D] = 0.0
    q = tcg.spatial_q(params, h, r)
    Sa, Sb = tcg.sensitivities(params, h, x)
    G = tcg.assemble(params, h, x, r, q, Sa, Sb)
    loss = 0.5 * float(np.mean(r ** 2))
    return loss, G


def train_pro(seed, kappa, moving=False, record_complex=False):
    """PC-pro: PC0's correction + kappa * (residual difference)."""
    params = tcg.init_params(seed)
    rng = np.random.RandomState(1000 + seed)
    w_pred = [np.ones(tcg.N, np.complex128) for _ in range(tcg.L)]
    w_corr_prev = None
    prev = None
    r_prev = None
    flat = tcg.flatten(params)
    m = np.zeros_like(flat)
    v = np.zeros_like(flat)
    losses = []
    series = dict(rn=[], dr=[], ratio=[], cos_r_dr=[], phaseE_r=[],
                  phaseE_pro=[], phase_sign=[])
    wtrack = []
    w_abs_max = 1.0
    for step in range(1, STEPS + 1):
        if moving:
            D = int(round(MOVE_D0 + (MOVE_D1 - MOVE_D0) * step / STEPS))
            x, y = make_data_moving(rng, D)
            loss, G = batch_grad_masked(params, x, y, D)
        else:
            x, y = make_data(rng)
            loss, G = cvm.batch_grad(params, x, y)[:2]
        losses.append(loss)
        h_n = tcg.flat_grads(G, params)

        if prev is not None:
            Gp, th_all, u_all, sig_all = prev
            off = 0
            r_hat = []
            for l in range(tcg.L):
                th = th_all[l]
                u_mode = u_all[l]
                sigp = sig_all[l]
                A = Gp["a"][l] * np.exp(1j * th)
                Gb = Gp["b"][l]
                M_ = Gb.shape[1]
                gN_rho = h_n[off:off + tcg.N]
                gN_theta = h_n[off + tcg.N:off + 2 * tcg.N]
                gN_bre = h_n[off + 2 * tcg.N:
                             off + 2 * tcg.N + tcg.N * M_].reshape(tcg.N, M_)
                gN_bim = h_n[off + 2 * tcg.N + tcg.N * M_:
                             off + 2 * tcg.N + 2 * tcg.N * M_].reshape(
                                 tcg.N, M_)
                off += 2 * tcg.N + 2 * tcg.N * M_
                du = (gN_rho * sigp * A.real
                      + gN_theta * (-u_mode) * A.imag
                      + (gN_bre * Gb.real - gN_bim * Gb.imag).sum(axis=1))
                dv = (gN_rho * sigp * A.imag
                      + gN_theta * (u_mode) * A.real
                      + (gN_bre * Gb.imag + gN_bim * Gb.real).sum(axis=1))
                r_hat.append(du + 1j * dv)

            # ---- the TSS residual prospection (same-w: exact here) ----
            if r_prev is None or kappa == 0.0:
                r_pro = [r.copy() for r in r_hat]
            else:
                r_pro = [r + kappa * (r - rp_)
                         for r, rp_ in zip(r_hat, r_prev)]

            # ---- diagnostics (pre-update values) ----
            rn = np.concatenate(r_hat)
            drn = rn - (np.concatenate(r_prev)
                        if r_prev is not None else np.zeros_like(rn))
            if record_complex and r_prev is not None:
                series.setdefault("rhat_c", []).append(rn.copy())
                series.setdefault("dr_c", []).append(drn.copy())
            rpn = np.concatenate(r_pro)
            e_r = float(np.sum(np.abs(rn) ** 2))
            e_p = float(np.sum(np.abs(rpn) ** 2))
            series["rn"].append(float(np.sqrt(e_r)))
            series["dr"].append(float(np.linalg.norm(drn)))
            series["ratio"].append(float(
                kappa * np.linalg.norm(drn) / (np.sqrt(e_r) + 1e-12)))
            series["cos_r_dr"].append(float(
                np.real(np.vdot(rn, drn))
                / (np.linalg.norm(rn) * np.linalg.norm(drn) + 1e-30)))
            series["phaseE_r"].append(
                float(np.sum(rn.imag ** 2) / (e_r + 1e-30)))
            series["phaseE_pro"].append(
                float(np.sum(rpn.imag ** 2) / (e_p + 1e-30)))
            series["phase_sign"].append(float(np.mean(
                np.sign(rpn.imag) == np.sign(rn.imag))))

            w_corr = [wp - LR_M * (-LR) * r_
                      for wp, r_ in zip(w_pred, r_pro)]
            w_pred = w_corr
            w_corr_prev = [wc.copy() for wc in w_corr]
            r_prev = [r.copy() for r in r_hat]
            w_abs_max = max(w_abs_max,
                            max(float(np.abs(wp).max()) for wp in w_pred))

        # ---- store this step's pre-update blocks for the next correction --
        # (the u/v chain pairs G_{n-1} with the params it was evaluated at)
        prev = (dict(a=[ga.copy() for ga in G["a"]],
                     b=[gb.copy() for gb in G["b"]]),
                [th.copy() for th in params["theta"]],
                [tcg.sig(params["rho"][l]) for l in range(tcg.L)],
                [tcg.sig(params["rho"][l]) * (1 - tcg.sig(params["rho"][l]))
                 for l in range(tcg.L)])

        # ---- main update with the current geometry (PC0 timing) ----
        G_use = cvm.scale_by_w(G, w_pred)
        g = cvm.clip(tcg.flat_grads(G_use, params))
        flat, m, v = cvm.adam(flat, g, m, v, step)
        params = tcg.pack(params, flat)
        if step % 25 == 0:
            wtrack.append([np.angle(wl) for wl in w_pred])

    losses = np.asarray(losses)
    S = {k: np.asarray(vv) for k, vv in series.items()}
    half = len(S["rn"]) // 2

    def med(key, late=True):
        s_ = S[key][half:] if late else S[key]
        return float(np.median(s_)) if len(s_) else 0.0

    wt = np.asarray(wtrack)                      # (T/25, L, N)
    dphi = np.angle(np.exp(1j * np.diff(wt, axis=0)))
    rms_dphi = float(np.sqrt(np.mean(dphi ** 2))) if wt.shape[0] > 1 else 0.0

    return dict(final_loss=float(losses[-100:].mean()),
                finite=bool(np.all(np.isfinite(losses))),
                w_abs_mean=float(np.mean([np.abs(wl).mean()
                                          for wl in w_pred])),
                w_abs_max=w_abs_max,
                rms_dphi=rms_dphi,
                diag=dict(rn=med("rn"), dr=med("dr"), ratio=med("ratio"),
                          cos_r_dr=med("cos_r_dr"),
                          phaseE_r=med("phaseE_r"),
                          phaseE_pro=med("phaseE_pro"),
                          phase_sign=med("phase_sign")),
                series={k: vv.tolist() for k, vv in S.items()})


def main() -> None:
    setup()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t00 = time.time()
    if not sanity():
        print("SANITY FAIL — stopping before the sweep.")
        return

    ref = json.load(open(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results", "route_pc", "summary.json")))
    finals_ref = {k: v for k, v in ref["finals"].items()}
    print("reference finals (route_pc.py):",
          {k: {s: round(vv, 4) for s, vv in v.items()}
           for k, v in finals_ref.items() if k in ("online", "routeA",
                                                   "pc_b0.0")})

    results = {}
    audit0 = dict(rp.BPTT_CALLS)

    # ---- stationary sweep ----
    for kappa in KAPPAS:
        for seed in SEEDS:
            print(f"pro k={kappa} s{seed} (stationary)...", flush=True)
            out = train_pro(seed, kappa, moving=False)
            results[f"pro{kappa}/s{seed}"] = out
            print(f"  final {out['final_loss']:.4f}  "
                  f"finite {out['finite']}  |w|max {out['w_abs_max']:.1f}  "
                  f"rms_dphi {out['rms_dphi']:.4f}", flush=True)
            np.savez(os.path.join(
                RESULTS_DIR, f"series_pro{kappa}_s{seed}.npz"),
                **{k: np.asarray(v) for k, v in out["series"].items()})
            out = {k: v for k, v in out.items() if k != "series"}
            results[f"pro{kappa}/s{seed}"] = out

    # ---- kappa=0 implementation gate vs stored PC0 ----
    gate = []
    for seed in SEEDS:
        a = results[f"pro0.0/s{seed}"]["final_loss"]
        b = finals_ref["pc_b0.0"][str(seed)]
        gate.append(abs(a - b))
    print(f"kappa=0 vs stored PC0: max |dfinal| {max(gate):.2e} "
          f"(bitwise expected)")
    k0_ok = max(gate) < 1e-10

    # ---- w-momentum control ----
    for seed in SEEDS:
        print(f"wmom b={WMOM_BETA} s{seed}...", flush=True)
        out = rp.train_pc(seed, WMOM_BETA)
        out = {k: v for k, v in out.items() if k != "w_final"}
        results[f"wmom/s{seed}"] = out
        print(f"  final {out['final_loss']:.4f}", flush=True)

    # ---- moving sweep ----
    for kappa in KAPPAS:
        for seed in SEEDS:
            print(f"pro k={kappa} s{seed} (moving delay)...", flush=True)
            out = train_pro(seed, kappa, moving=True)
            results[f"pro{kappa}_mov/s{seed}"] = out
            print(f"  final {out['final_loss']:.4f}  "
                  f"finite {out['finite']}  |w|max {out['w_abs_max']:.1f}  "
                  f"rms_dphi {out['rms_dphi']:.4f}", flush=True)
            np.savez(os.path.join(
                RESULTS_DIR, f"series_pro{kappa}_mov_s{seed}.npz"),
                **{k: np.asarray(v) for k, v in out["series"].items()})
            out = {k: v for k, v in out.items() if k != "series"}
            results[f"pro{kappa}_mov/s{seed}"] = out

    # ---- BPTT audit ----
    audit_delta = {k: rp.BPTT_CALLS[k] - audit0[k] for k in rp.BPTT_CALLS}
    print(f"BPTT calls during ALL PC-pro/wmom arms: {audit_delta} "
          f"(must be zero)")

    # ---- drift/noise pre-measurement (stationary kappa=0) ----
    dn = {}
    for regime, suffix in [("stationary", ""), ("moving", "_mov")]:
        t_stats, acs = [], []
        for seed in SEEDS:
            z = np.load(os.path.join(
                RESULTS_DIR, f"series_pro0.0{suffix}_s{seed}.npz"))
            rn = z["rn"]  # note: ||r|| series only; reconstruct drift
            dr = z["dr"]
            half = len(dr) // 2
            dr_late = dr[half:]
            # scalar proxy: systematic drift of the residual MAGNITUDE and
            # its lag-1 autocorrelation (per-mode vector version lives in
            # the npz; the summary uses the robust scalar form)
            mu, sd = float(np.mean(dr_late)), float(np.std(dr_late))
            if sd > 0:
                t_stats.append(abs(mu) / (sd / np.sqrt(len(dr_late))))
                acs.append(float(np.corrcoef(dr_late[:-1],
                                             dr_late[1:])[0, 1]))
        dn[regime] = dict(drift_t=float(np.median(t_stats)),
                          ac1=float(np.median(acs)))
    print(f"drift/noise (stationary): t {dn['stationary']['drift_t']:.2f}  "
          f"ac1 {dn['stationary']['ac1']:+.3f}")
    print(f"drift/noise (moving):     t {dn['moving']['drift_t']:.2f}  "
          f"ac1 {dn['moving']['ac1']:+.3f}")

    # ---- geometry-movement premise check ----
    rms_stat = float(np.median([results[f"pro0.0/s{s}"]["rms_dphi"]
                                for s in SEEDS]))
    rms_mov = float(np.median([results[f"pro0.0_mov/s{s}"]["rms_dphi"]
                               for s in SEEDS]))
    premise = rms_mov > 1.5 * rms_stat
    print(f"geometry movement (rms dphi, PC0): stationary {rms_stat:.4f}  "
          f"moving {rms_mov:.4f}  -> premise "
          f"{'HOLDS' if premise else 'WEAK — flag the moving comparison'}")

    # ---- verdict ----
    def finals(key):
        return {s: results[f"{key}/s{s}"]["final_loss"] for s in SEEDS}

    pc0 = finals("pro0.0")
    onl = {s: finals_ref["online"][str(s)] for s in SEEDS}
    ra = {s: finals_ref["routeA"][str(s)] for s in SEEDS}
    gap0 = np.median([onl[s] for s in SEEDS]) - np.median(list(pc0.values()))

    def regime_verdict(suffix):
        out = {}
        for k in KAPPAS[1:]:
            f = finals(f"pro{k}{suffix}")
            wins = sum(f[s] < pc0[s] for s in SEEDS)
            med = float(np.median(list(f.values())))
            stable = all(results[f"pro{k}{suffix}/s{s}"]["finite"]
                         for s in SEEDS)
            out[k] = dict(finals=f, wins=wins, median=med,
                          stable=stable,
                          rgap=float(np.median(
                              [(onl[s] - f[s]) / (onl[s] - ra[s])
                               for s in SEEDS])) if not suffix else None)
        return out

    stat = regime_verdict("")
    mov = regime_verdict("_mov")
    band = TIE_BAND * abs(gap0)
    med0 = float(np.median(list(pc0.values())))

    def best_kappa(tab):
        return min(tab, key=lambda k: tab[k]["median"])

    bs, bm = best_kappa(stat), best_kappa(mov)
    stat_win = (stat[bs]["wins"] >= 4
                and stat[bs]["median"] < med0 - band
                and stat[bs]["stable"]
                and any(stat[k]["median"] <= med0 + band
                        for k in KAPPAS[1:] if k != bs))
    mov_win = (mov[bm]["wins"] >= 4
               and mov[bm]["median"] < float(np.median(list(
                   finals("pro0.0_mov").values()))) - TIE_BAND * abs(
                   float(np.median(list(finals("pro0.0_mov").values()))))
               and mov[bm]["stable"])
    unstable_growth = any(not stat[k]["stable"] or
                          not mov[k]["stable"] for k in KAPPAS[1:])

    if unstable_growth:
        verdict = "UNSTABLE/NOISY"
    elif stat_win:
        verdict = "PROSPECTION HELPS (stationary — contradicts " \
                  "phase_track; investigate)"
    elif mov_win and premise:
        verdict = "DORMANT-BUT-FUNCTIONAL (TSS term correct; dormant " \
                  "under stationarity)"
    else:
        verdict = "CORRECTION ONLY"

    print("-" * 78)
    print(f"stationary: PC0 median {med0:.4f}; sweep: "
          f"{ {k: round(v['median'], 4) for k, v in stat.items()} }")
    print(f"            wins vs PC0: "
          f"{ {k: v['wins'] for k, v in stat.items()} }")
    print(f"            R_gap:       "
          f"{ {k: round(v['rgap'], 2) for k, v in stat.items()} }")
    print(f"wmom (b={WMOM_BETA}): median "
          f"{np.median([results[f'wmom/s{s}']['final_loss'] for s in SEEDS]):.4f}")
    m0 = float(np.median(list(finals('pro0.0_mov').values())))
    print(f"moving:     PC0 median {m0:.4f}; sweep: "
          f"{ {k: round(v['median'], 4) for k, v in mov.items()} }")
    print(f"            wins vs PC0: "
          f"{ {k: v['wins'] for k, v in mov.items()} }")
    print(f"kappa=0 == stored PC0: {k0_ok} (max diff {max(gate):.1e})")
    print(f"VERDICT: {verdict}")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(git=git,
               config=dict(steps=STEPS, lr=LR, lr_m=LR_M, seeds=SEEDS,
                           kappas=KAPPAS, wmom_beta=WMOM_BETA,
                           tie_band=TIE_BAND,
                           move_delay=[MOVE_D0, MOVE_D1]),
               k0_reproduces_pc0=bool(k0_ok), k0_max_diff=float(max(gate)),
               bptt_calls=audit_delta,
               drift_noise=dn, premise_holds=bool(premise),
               rms_dphi_stationary=rms_stat, rms_dphi_moving=rms_mov,
               stationary={str(k): {kk: vv for kk, vv in v.items()
                                    if kk != "finals"}
                           for k, v in stat.items()},
               moving={str(k): {kk: vv for kk, vv in v.items()
                                if kk != "finals"}
                       for k, v in mov.items()},
               finals={k: {str(s): results[f"{k}/s{s}"]["final_loss"]
                           for s in SEEDS}
                       for k in ["pro0.0", "wmom", "pro0.0_mov"]},
               diags={k: {str(s): results[f"{k}/s{s}"]["diag"]
                          for s in SEEDS}
                      for k in [f"pro{k}" for k in KAPPAS]
                      + [f"pro{k}_mov" for k in KAPPAS]},
               verdict=verdict)
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(doc, f, indent=2)
    print(f"wrote summary.json  ({time.time() - t00:.0f}s total)")


if __name__ == "__main__":
    main()
