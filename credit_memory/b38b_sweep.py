import json, time
import functools
import numpy as np
import jax
import jax.numpy as jnp
from credit_memory.b38b_train import (
    train, make_data, build_eval, LR_GRID, N_TEST, N_VAL, T_SEQ, M_IN, Q_BOTT,
    flat_all, unflat_all, adam, rtrl_grad_one)
from credit_memory.b38b_selective import init_L, fixed_R, nparams, tile_step_L

J, D = 4, 4
SEEDS = (0, 1, 2)
LS = (32, 128, 256)


@functools.lru_cache(maxsize=None)
def build_online(J, d):
    """ARM C: theta updated after EVERY token, eligibility CARRIED (never reset).
    Judged as an online algorithm, not a BPTT identity. Optimizer not
    differentiated through."""
    q = Q_BOTT

    def run(params, R, xs, ys, lr):
        th0 = flat_all(params, J, d)
        Ptau = th0.shape[1]

        def step(carry, inp):
            params, m, v, t, h, E = carry
            x, ystar = inp
            th = flat_all(params, J, d)
            xi = x[0]
            xl = jnp.einsum("jkm,m->jk", R, x)
            G = jax.vmap(lambda tt, hh, xx: jax.jacrev(
                lambda z: tile_step_L(z, jax.lax.stop_gradient(hh), xx, xi, d, q)[0]
            )(tt))(th, h, xl)
            hn, a = jax.vmap(lambda tt, hh, xx: tile_step_L(tt, hh, xx, xi, d, q))(th, h, xl)
            E = a[:, :, None] * E + G
            hf = hn.reshape(-1)
            err = jnp.dot(params["C"][0], hf) - ystar
            dldy = 2.0 * err
            qt = dldy * params["C"][0].reshape(J, d)
            g = unflat_all(jnp.einsum("jdp,jd->jp", E, qt), J, d)
            g["C"] = (dldy * hf).reshape(1, -1)
            params, m, v, t = adam(params, m, v, g, t, lr)
            return (params, m, v, t, hn, E), err ** 2
        m0 = jax.tree.map(jnp.zeros_like, params)
        carry = (params, m0, m0, jnp.array(0.0), jnp.zeros((J, d)),
                 jnp.zeros((J, d, Ptau)))
        (params, _, _, _, _, _), sq = jax.lax.scan(step, carry, (xs, ys))
        return params, sq
    return jax.jit(run)


rows, t0 = [], time.time()
print("=" * 112)
print("A/B matched training + in-training gradient check (Arm L)")
print("=" * 112)
print(f"{'seed':>4s} {'L':>4s} {'A nmse':>11s} {'B nmse':>11s} {'lr A':>6s} {'lr B':>6s} "
      f"{'|dNMSE|/NMSE':>13s} {'worst grad err (all families)':>30s}")
for seed in SEEDS:
    for L in LS:
        best = {}
        for arm in ("A", "B"):
            b = None
            for lr in LR_GRID:
                s = time.time(); o = train(J, D, seed, arm, L, lr); o["wall"] = time.time() - s
                o["lr"] = lr
                if b is None or o["val_loss"] < b["val_loss"]:
                    b = o
            best[arm] = b
        ck = train(J, D, seed, L=L, arm="B", lr=best["B"]["lr"], check=True)
        w = max(ck["worst_err"])
        dn = abs(best["A"]["test_nmse"] - best["B"]["test_nmse"]) / max(best["A"]["test_nmse"], 1e-30)
        rows.append(dict(kind="AB", seed=seed, L=L, A=best["A"]["test_nmse"],
                         B=best["B"]["test_nmse"], lrA=best["A"]["lr"], lrB=best["B"]["lr"],
                         dn=float(dn), worst=ck["worst_err"],
                         wallA=best["A"]["wall"], wallB=best["B"]["wall"]))
        print(f"{seed:4d} {L:4d} {best['A']['test_nmse']:11.4e} {best['B']['test_nmse']:11.4e} "
              f"{best['A']['lr']:6.0e} {best['B']['lr']:6.0e} {dn:13.2e} {w:30.3e}", flush=True)

print()
print("=" * 112)
print("Selectivity control: non-selective ablation of the SAME architecture (Delta, b input-independent)")
print("=" * 112)
for seed in SEEDS:
    b = None
    for lr in LR_GRID:
        o = train(J, D, seed, "B", 32, lr, ablate=True)
        if b is None or o["val_loss"] < b["val_loss"]:
            b = dict(o, lr=lr)
    sel = [r["B"] for r in rows if r["seed"] == seed and r["L"] == 32][0]
    rows.append(dict(kind="ablate", seed=seed, nmse=b["test_nmse"], sel=sel))
    print(f"  seed {seed}: selective {sel:.4e}   non-selective {b['test_nmse']:.4e}   "
          f"ratio {b['test_nmse']/sel:6.1f}x worse")

print()
print("=" * 112)
print("ARM C: every-token online updates (eligibility carried across parameter updates)")
print("=" * 112)
print(f"{'seed':>4s} {'lr':>7s} {'online NMSE':>12s} {'held-out NMSE':>14s} {'div':>5s} {'wall':>8s}")
for seed in SEEDS:
    R = fixed_R(J, M_IN)
    xs, ys = make_data(1, 8192, 9000 + seed)
    xva, yva = make_data(N_VAL, T_SEQ, 7000 + seed)
    xte, yte = make_data(N_TEST, T_SEQ, 8000 + seed)
    ev = build_eval(J, D, False)
    run = build_online(J, D)
    b = None
    for lr in LR_GRID:
        p0 = init_L(J, D, M_IN, seed)
        s = time.time(); pr, sq = run(p0, R, xs[0], ys[0], lr); jax.block_until_ready(sq)
        w = time.time() - s
        sq = np.asarray(sq)
        vl = float(ev(pr, R, xva, yva))
        o = dict(lr=lr, params=pr, val=vl if np.isfinite(vl) else np.inf,
                 online=float(np.mean(sq[np.isfinite(sq)])) / float(jnp.mean(ys ** 2)),
                 wall=w, diverged=bool(not np.isfinite(sq).all()))
        if b is None or o["val"] < b["val"]:
            b = o
    hn = float(ev(b["params"], R, xte, yte)) / float(jnp.mean(yte ** 2))
    rows.append(dict(kind="C", seed=seed, lr=b["lr"], online=b["online"], heldout=hn,
                     diverged=b["diverged"], wall=b["wall"]))
    print(f"{seed:4d} {b['lr']:7.0e} {b['online']:12.4e} {hn:14.4e} "
          f"{str(b['diverged']):>5s} {b['wall']:7.2f}s", flush=True)

json.dump(rows, open("results/b38b/train.json", "w"), indent=1)
ab = [r for r in rows if r["kind"] == "AB"]
print(f"\nSUMMARY  worst in-training gradient error over all families/seeds/L: "
      f"{max(max(r['worst']) for r in ab):.3e}")
print(f"         max |NMSE_A - NMSE_B| / NMSE_A: {max(r['dn'] for r in ab):.3e}")
print(f"         wall-clock (CPU-only) A {np.median([r['wallA'] for r in ab]):.2f}s "
      f"vs B {np.median([r['wallB'] for r in ab]):.2f}s")
print(f"done in {time.time()-t0:.1f}s")
