"""B38a section 5 -- ARM C: every-token carried-trace online updates.

theta_t -> theta_{t+1} after EVERY observed loss, eligibility carried across
parameter updates (never reset). Judged as an ONLINE LEARNING ALGORITHM, not as
a numerical reproduction of BPTT: once theta changes every step there is no
single fixed parameter vector that generated the history, so the carried trace
is the exact sensitivity under the fixed/path-shift interpretation but is not
the frozen-current replay gradient. The optimizer is never differentiated
through. Architecture and optimizer are unchanged from arms A/B."""
import json, time
import numpy as np
import jax
import jax.numpy as jnp

from credit_memory.b38a_train import (
    FAMILIES_ALL, LR_GRID, EVAL_SEEDS, make_teacher_norm, data_for, markov_err, T_SEQ)
from credit_memory.b37b_quotient_trainability import make_dataset
from credit_memory.b37c_productlocal_native import (
    spec_from_blocks, generic_init, to_jax, spectral_radius)
from credit_memory.b38a_engine import build_online_scan, build_eval

T_STREAM = 8192
BLOCKS = 32


def stream(teacher, seed, T):
    xs, ys = make_dataset(teacher, 1, T, 51000 + seed)
    return xs[0], ys[0]


def run_online(family, seed, lr, T=T_STREAM):
    t = make_teacher_norm(family, 8, seed)
    spec = spec_from_blocks(t["blocks"])
    xs, ys = stream(t, seed, T)
    p0 = to_jax(generic_init(spec, seed))
    run = build_online_scan(spec)
    s = time.perf_counter()
    out = run(p0, xs, ys, lr)
    jax.block_until_ready(out)
    wall = time.perf_counter() - s
    params, sq, gn = out
    sq = np.asarray(sq)
    ok = np.isfinite(sq)
    diverged = (not ok.all()) or float(np.nanmax(sq[ok])) > 1e12
    blk = np.array([np.mean(sq[i * (T // BLOCKS):(i + 1) * (T // BLOCKS)])
                    for i in range(BLOCKS)])
    yv = float(np.mean(np.asarray(ys) ** 2))
    return dict(params=params, spec=spec, teacher=t, wall=wall,
                online_nmse=float(np.mean(sq[ok])) / yv if ok.any() else float("inf"),
                blocks=(blk / yv).tolist(), diverged=bool(diverged),
                grad_norm=float(np.nanmedian(np.asarray(gn)[ok])))


rows = []
print(f"{'family':21s} {'s':>2s} {'lr':>7s} {'online NMSE':>12s} {'held-out NMSE':>13s} "
      f"{'markov':>10s} {'t_90%':>7s} {'rho':>6s} {'div':>4s} {'wall':>8s}")
for f in FAMILIES_ALL:
    for seed in EVAL_SEEDS:
        t = make_teacher_norm(f, 8, seed)
        _, (xva, yva), (xte, yte) = data_for(t, f, 8, seed)
        ev = build_eval(spec_from_blocks(t["blocks"]))
        best = None
        for lr in LR_GRID:
            o = run_online(f, seed, lr)
            vl = float(ev(o["params"], xva, yva))
            o["val"] = vl if np.isfinite(vl) else float("inf")
            if best is None or o["val"] < best["val"]:
                best = dict(o, lr=lr)
        ynorm = float(jnp.mean(yte ** 2))
        hn = float(ev(best["params"], xte, yte)) / ynorm
        mk = markov_err(best["params"], best["teacher"], best["spec"])
        b = np.array(best["blocks"])
        thr = b[0] * 0.1
        idx = np.where(b <= thr)[0]
        t90 = int(idx[0] * (T_STREAM // BLOCKS)) if len(idx) else -1
        rho = spectral_radius(best["params"]["u"], best["spec"])
        rows.append(dict(family=f, seed=seed, lr=best["lr"], online_nmse=best["online_nmse"],
                         heldout_nmse=hn, markov=mk, t90=t90, rho=rho,
                         diverged=best["diverged"], wall=best["wall"],
                         blocks=best["blocks"]))
        print(f"{f:21s} {seed:2d} {best['lr']:7.0e} {best['online_nmse']:12.3e} {hn:13.3e} "
              f"{mk:10.3e} {t90:7d} {rho:6.3f} {str(best['diverged']):>4s} "
              f"{best['wall']:7.2f}s", flush=True)
json.dump(rows, open("results/b38a/online.json", "w"), indent=1)
