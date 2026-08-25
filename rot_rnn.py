"""D6 generality rig — real 2D-rotational RNN on delayed copy.

Real-arithmetic sibling of the complex S5 rig. Per block i:

    s_t^i = A_i s_{t-1}^i + B^i u_t,
    A_i = [[p, -q],[q, p]],  (p, q) = sig(r_i) (cos th_i, sin th_i)

Polar-constrained to the unit disk. The adjoint rotates the other way:
lam_t = q_t + A_i^T lam_{t+1}. The cartesian eigenvalue-gradient pair
(G_p, G_q) is the exact image of the complex rig's G_a.

Learned orientation: per block angle phi, applied as R(phi) to the
(G_p, G_q) pair and to the state-component index of G_b — the real
analog of conj(w) on the complex gradient blocks.

Gates: FD check of the exact (J-slot) gradient, rel err < 1e-4.

Run:  python rot_rnn.py
"""
from __future__ import annotations

import numpy as np

T, DELAY, BATCH = 128, 50, 32
L, NB = 2, 8
STEPS = 1500
LR, CLIP = 1e-3, 1.0
SEEDS = [0, 1, 2]


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def init_params(seed):
    rng = np.random.RandomState(seed)
    u0 = np.linspace(0.90, 0.995, NB)
    return dict(
        rho=[np.log(u0 / (1 - u0)) for _ in range(L)],
        theta=[rng.uniform(-np.pi, np.pi, NB) for _ in range(L)],
        b=[rng.randn(NB, 2, 1) * 0.5, rng.randn(NB, 2, 2 * NB) * 0.1],
        c=rng.randn(2 * NB) / np.sqrt(2 * NB),
    )


def a_mats(params, l):
    r = sig(params["rho"][l])
    th = params["theta"][l]
    p, q = r * np.cos(th), r * np.sin(th)
    A = np.zeros((NB, 2, 2))
    A[:, 0, 0], A[:, 0, 1] = p, -q
    A[:, 1, 0], A[:, 1, 1] = q, p
    return A


def da_mats(params, l):
    """Derivatives of A w.r.t. p and q."""
    dAp = np.zeros((NB, 2, 2))
    dAp[:, 0, 0], dAp[:, 1, 1] = 1.0, 1.0
    dAq = np.zeros((NB, 2, 2))
    dAq[:, 0, 1], dAq[:, 1, 0] = -1.0, 1.0
    return dAp, dAq


def forward(params, x):
    h = []
    inp = x[..., None]
    for l in range(L):
        A = a_mats(params, l)
        Bm = params["b"][l]
        hl = np.zeros((T, x.shape[1], NB, 2))
        sp = np.zeros((x.shape[1], NB, 2))
        for t in range(T):
            sp = np.einsum("nij,bnj->bni", A, sp) \
                + np.einsum("idm,bm->bid", Bm, inp[t])
            hl[t] = sp
        h.append(hl)
        inp = hl.reshape(T, x.shape[1], 2 * NB)
    yhat = np.einsum("n,tbn->tb", params["c"],
                     h[-1].reshape(T, x.shape[1], 2 * NB))
    return h, yhat


def spatial_q(params, r):
    qL = np.einsum("ie,tb->tbie", params["c"].reshape(NB, 2), r)
    B4 = params["b"][1].reshape(NB, 2, NB, 2)           # j,c,i,e
    q0 = np.einsum("jcie,tbjc->tbie", B4, qL)
    return [q0, qL]


def exact_lambda(params, q):
    lam = [np.zeros_like(ql) for ql in q]
    nxt = [np.zeros((q[0].shape[1], NB, 2)) for _ in range(L)]
    AT = [np.transpose(a_mats(params, l), (0, 2, 1)) for l in range(L)]
    B4 = params["b"][1].reshape(NB, 2, NB, 2)
    for t in range(T - 1, -1, -1):
        nxt[L - 1] = q[L - 1][t] + np.einsum("nij,bnj->bni",
                                             AT[L - 1], nxt[L - 1])
        for l in range(L - 2, -1, -1):
            up = np.einsum("jcie,bjc->bie", B4, nxt[l + 1])
            nxt[l] = up + np.einsum("nij,bnj->bni", AT[l], nxt[l])
        for l in range(L):
            lam[l][t] = nxt[l]
    return lam


def sensitivities(params, h, x):
    """S-slot sensitivities per layer: Sp, Sq (T,B,NB,2); Sb (T,B,NB,2,M)."""
    Sp, Sq, Sb = [], [], []
    inp = x[..., None]
    for l in range(L):
        A = a_mats(params, l)
        dAp, dAq = da_mats(params, l)
        M = params["b"][l].shape[2]
        sp = np.zeros((T, x.shape[1], NB, 2))
        sq = np.zeros((T, x.shape[1], NB, 2))
        sb = np.zeros((T, x.shape[1], NB, 2, 2, M))
        sprev = np.zeros((x.shape[1], NB, 2))
        sqrev = np.zeros((x.shape[1], NB, 2))
        sbrev = np.zeros((x.shape[1], NB, 2, 2, M))
        for t in range(T):
            s_prev = np.zeros((x.shape[1], NB, 2)) if t == 0 else h[l][t - 1]
            sprev = np.einsum("nce,bne->bnc", dAp, s_prev) \
                + np.einsum("nce,bne->bnc", A, sprev)
            sqrev = np.einsum("nce,bne->bnc", dAq, s_prev) \
                + np.einsum("nce,bne->bnc", A, sqrev)
            sbrev = np.einsum("nce,bnedm->bncdm", A, sbrev) \
                + np.eye(2)[None, None, :, :, None] \
                * inp[t][:, None, None, None, :]
            sp[t], sq[t], sb[t] = sprev, sqrev, sbrev
        Sp.append(sp)
        Sq.append(sq)
        Sb.append(sb)
        inp = h[l].reshape(T, x.shape[1], 2 * NB)
    return Sp, Sq, Sb


def assemble(params, h, x, r, err, S=None):
    """Gradient dict from credit err. S=(Sp,Sq,Sb) for the online
    S-slot; S=None gives the exact J-slot (direct sensitivities at the
    realized trajectory)."""
    Gp, Gq, Gb = [], [], []
    inp = x[..., None]
    for l in range(L):
        M = params["b"][l].shape[2]
        if S is not None:
            Sp, Sq, Sb = S
            Gp.append(np.einsum("tbie,tbie->i", err[l], Sp[l]))
            Gq.append(np.einsum("tbie,tbie->i", err[l], Sq[l]))
            Gb.append(np.einsum("tbie,tbiedm->idm", err[l], Sb[l]))
        else:
            dAp, dAq = da_mats(params, l)
            h_prev = np.concatenate([np.zeros((1, x.shape[1], NB, 2)),
                                     h[l][:-1]], axis=0)
            Gp.append(np.einsum("tbie,iec,tbic->i", err[l], dAp, h_prev))
            Gq.append(np.einsum("tbie,iec,tbic->i", err[l], dAq, h_prev))
            Gb.append(np.einsum("tbid,tbm->idm", err[l], inp))
        inp = h[l].reshape(T, x.shape[1], 2 * NB)
    Gc = np.einsum("tb,tbn->n", r, h[L - 1].reshape(T, x.shape[1],
                                                     2 * NB))
    return dict(p=Gp, q=Gq, b=Gb, c=Gc)


def flatten(G, params):
    parts = []
    for l in range(L):
        parts += [G["p"][l], G["q"][l], G["b"][l].ravel()]
    parts.append(G["c"])
    return np.concatenate(parts)


def pack_grad_shape(params):
    return sum(2 * NB + params["b"][l].size for l in range(L)) + 2 * NB


def make_data(rng):
    x = rng.randn(T, BATCH)
    y = np.concatenate([np.zeros((DELAY, BATCH)), x[:-DELAY]], axis=0)
    return x, y


def batch_grad(params, x, y, exact=False):
    h, yhat = forward(params, x)
    r = yhat - y
    r[:DELAY] = 0.0
    loss = 0.5 * float(np.mean(r ** 2))
    q = spatial_q(params, r)
    if exact:
        err = exact_lambda(params, q)
        G = assemble(params, h, x, r, err, S=None)
    else:
        S = sensitivities(params, h, x)
        G = assemble(params, h, x, r, q, S=S)
    return loss, G, q, r, h


def param_grad_transform(G, params):
    """(G_p, G_q) [cartesian] -> (G_rho, G_theta) via the polar chain."""
    Gr, Gt = [], []
    for l in range(L):
        r = params["rho"][l]
        th = params["theta"][l]
        u = sig(r)
        sigp = u * (1 - u)
        # p = u cos th, q = u sin th
        Gr.append(sigp * (np.cos(th) * G["p"][l] + np.sin(th) * G["q"][l]))
        Gt.append(u * (-np.sin(th) * G["p"][l] + np.cos(th) * G["q"][l]))
    return dict(rho=Gr, theta=Gt, b=G["b"], c=G["c"])


def flat(Gp, params):
    parts = []
    for l in range(L):
        parts += [Gp["rho"][l], Gp["theta"][l], Gp["b"][l].ravel()]
    parts.append(Gp["c"])
    return np.concatenate(parts)


def pack_params(params, vec):
    out = dict(rho=[], theta=[], b=[], c=None)
    i = 0
    for l in range(L):
        out["rho"].append(vec[i:i + NB].copy())
        out["theta"].append(vec[i + NB:i + 2 * NB].copy())
        i += 2 * NB
        sz = params["b"][l].size
        out["b"].append(vec[i:i + sz].reshape(params["b"][l].shape).copy())
        i += sz
    out["c"] = vec[i:i + 2 * NB].copy()
    return out


def flatten_params(params):
    parts = []
    for l in range(L):
        parts += [params["rho"][l], params["theta"][l],
                  params["b"][l].ravel()]
    parts.append(params["c"])
    return np.concatenate(parts)


def fd_gate():
    """FD check of the exact gradient (polar params), small config."""
    global T, DELAY, BATCH
    keep = (T, DELAY, BATCH)
    T, DELAY, BATCH = 12, 4, 2
    try:
        params = init_params(0)
        rng = np.random.RandomState(5)
        x, y = make_data(rng)
        loss, G, q, r, h = batch_grad(params, x, y, exact=True)
        g = flat(param_grad_transform(G, params), params) / (T * BATCH)
        vec = flatten_params(params)
        eps = 1e-6
        for idx in [0, NB, 2 * NB + 1, len(vec) // 2]:
            vp = vec.copy()
            vp[idx] += eps
            lp = batch_grad(pack_params(params, vp), x, y, exact=True)[0]
            vp[idx] -= 2 * eps
            lm = batch_grad(pack_params(params, vp), x, y, exact=True)[0]
            fd = (lp - lm) / (2 * eps)
            rel = abs(fd - g[idx]) / max(abs(g[idx]), 1e-12)
            print(f"  fd gate idx {idx}: fd {fd:.6e} vs exact {g[idx]:.6e} "
                  f"rel {rel:.2e}  {'PASS' if rel < 1e-4 else 'FAIL'}")
            assert rel < 1e-4
    finally:
        T, DELAY, BATCH = keep


if __name__ == "__main__":
    params = init_params(0)
    rng = np.random.RandomState(1000)
    x, y = make_data(rng)
    h, yhat = forward(params, x)
    print("forward:", [hl.shape for hl in h], yhat.shape,
          f"loss0 {0.5 * float(np.mean((yhat - y) ** 2)):.4f}")
    fd_gate()
    print("gates pass")
