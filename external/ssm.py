import numpy as np

# Two stacked diagonal complex SSM layers + real readout.
#   h1_t = a1 * h1_{t-1} + B1 x_t
#   h2_t = a2 * h2_{t-1} + C1 h1_t
#   y_t  = Re(C2 h2_t) + d
# Gradients are Wirtinger conjugate derivatives g = dL/d(conj z);
# descent is z <- z - eta*g (factor 2 absorbed into eta).


def init_params(P1, P2, din, dout, rng, r1=(0.90, 0.995), r2=(0.95, 0.999)):
    def poles(P, rlo, rhi):
        r = rng.uniform(rlo, rhi, P)
        th = rng.uniform(0.0, 0.4, P)
        return r * np.exp(1j * th)
    p = {}
    p['a1'] = poles(P1, *r1)
    p['a2'] = poles(P2, *r2)
    # LRU-style normalisation: scale inputs by sqrt(1-|a|^2) so states stay O(1)
    n1 = np.sqrt(1 - np.abs(p['a1']) ** 2)[:, None]
    n2 = np.sqrt(1 - np.abs(p['a2']) ** 2)[:, None]
    p['B1'] = (rng.normal(0, 1, (P1, din)) + 1j * rng.normal(0, 1, (P1, din))) / np.sqrt(2 * din) * n1
    p['C1'] = (rng.normal(0, 1, (P2, P1)) + 1j * rng.normal(0, 1, (P2, P1))) / np.sqrt(2 * P1) * n2
    p['C2'] = (rng.normal(0, 1, (dout, P2)) + 1j * rng.normal(0, 1, (dout, P2))) / np.sqrt(2 * P2)
    p['d'] = np.zeros(dout)
    return p


KEYS = ['a1', 'B1', 'a2', 'C1', 'C2', 'd']
L1KEYS = ['a1', 'B1']       # lower layer: online gradient is biased
L2KEYS = ['a2', 'C1']       # top recurrent layer: online gradient is exact


def forward(p, x):
    """x: (B,T,din) real. Returns y,(h1,h2)."""
    B, T, _ = x.shape
    P1, P2 = p['a1'].shape[0], p['a2'].shape[0]
    h1 = np.zeros((B, T, P1), complex)
    h2 = np.zeros((B, T, P2), complex)
    u = x @ p['B1'].T                       # (B,T,P1)
    s1 = np.zeros((B, P1), complex)
    s2 = np.zeros((B, P2), complex)
    for t in range(T):
        s1 = p['a1'] * s1 + u[:, t]
        s2 = p['a2'] * s2 + s1 @ p['C1'].T
        h1[:, t] = s1
        h2[:, t] = s2
    y = (h2 @ p['C2'].T).real + p['d']
    return y, (h1, h2)


def loss_and_resid(y, tgt):
    r = y - tgt
    B = y.shape[0]
    return 0.5 * np.sum(r ** 2) / B, r / B


def bptt_grads(p, x, tgt):
    """Exact gradients by reverse-time adjoint. Also returns q2, lam2 for diagnostics."""
    y, (h1, h2) = forward(p, x)
    L, r = loss_and_resid(y, tgt)
    B, T, _ = x.shape

    q2 = 0.5 * (r @ np.conj(p['C2']))                       # (B,T,P2)
    lam2 = np.zeros_like(q2)
    acc = np.zeros((B, q2.shape[2]), complex)
    for t in range(T - 1, -1, -1):
        acc = q2[:, t] + np.conj(p['a2']) * acc
        lam2[:, t] = acc

    mu1 = lam2 @ np.conj(p['C1'])                           # exact layer-1 error
    lam1 = np.zeros_like(mu1)
    acc = np.zeros((B, mu1.shape[2]), complex)
    for t in range(T - 1, -1, -1):
        acc = mu1[:, t] + np.conj(p['a1']) * acc
        lam1[:, t] = acc

    h1m = np.concatenate([np.zeros_like(h1[:, :1]), h1[:, :-1]], 1)
    h2m = np.concatenate([np.zeros_like(h2[:, :1]), h2[:, :-1]], 1)
    g = {}
    g['C2'] = 0.5 * np.einsum('bto,btp->op', r, np.conj(h2))
    g['d'] = r.sum((0, 1))
    g['a2'] = np.einsum('btp,btp->p', lam2, np.conj(h2m))
    g['C1'] = np.einsum('btp,btj->pj', lam2, np.conj(h1))
    g['a1'] = np.einsum('btp,btp->p', lam1, np.conj(h1m))
    g['B1'] = np.einsum('btp,bti->pi', lam1, x)
    return L, g, dict(q2=q2, lam2=lam2, mu1=mu1, h1=h1, h2=h2, r=r)


def _traces(p, x, h1, h2):
    """Forward eligibility traces (exact, causal, O(params) memory)."""
    B, T, din = x.shape
    P1, P2 = p['a1'].shape[0], p['a2'].shape[0]
    h1m = np.concatenate([np.zeros_like(h1[:, :1]), h1[:, :-1]], 1)
    h2m = np.concatenate([np.zeros_like(h2[:, :1]), h2[:, :-1]], 1)
    eB1 = np.zeros((B, T, P1, din), complex)
    ea1 = np.zeros((B, T, P1), complex)
    eC1 = np.zeros((B, T, P2, P1), complex)
    ea2 = np.zeros((B, T, P2), complex)
    sB1 = np.zeros((B, P1, din), complex); sa1 = np.zeros((B, P1), complex)
    sC1 = np.zeros((B, P2, P1), complex); sa2 = np.zeros((B, P2), complex)
    for t in range(T):
        sB1 = p['a1'][None, :, None] * sB1 + x[:, t][:, None, :]
        sa1 = p['a1'] * sa1 + h1m[:, t]
        sC1 = p['a2'][None, :, None] * sC1 + h1[:, t][:, None, :]
        sa2 = p['a2'] * sa2 + h2m[:, t]
        eB1[:, t] = sB1; ea1[:, t] = sa1; eC1[:, t] = sC1; ea2[:, t] = sa2
    return dict(B1=eB1, a1=ea1, C1=eC1, a2=ea2)


def online_grads(p, x, tgt, exact_layer1_error=False):
    """Zucchet-style online gradients: forward eligibility x instantaneous error.
    exact_layer1_error=True substitutes the true adjoint (the D1 restoration check)."""
    y, (h1, h2) = forward(p, x)
    L, r = loss_and_resid(y, tgt)
    q2 = 0.5 * (r @ np.conj(p['C2']))

    if exact_layer1_error:
        lam2 = np.zeros_like(q2)
        acc = np.zeros((q2.shape[0], q2.shape[2]), complex)
        for t in range(q2.shape[1] - 1, -1, -1):
            acc = q2[:, t] + np.conj(p['a2']) * acc
            lam2[:, t] = acc
        mu1 = lam2 @ np.conj(p['C1'])
    else:
        mu1 = q2 @ np.conj(p['C1'])          # <-- the Zucchet approximation

    e = _traces(p, x, h1, h2)
    g = {}
    g['C2'] = 0.5 * np.einsum('bto,btp->op', r, np.conj(h2))
    g['d'] = r.sum((0, 1))
    g['a2'] = np.einsum('btp,btp->p', q2, np.conj(e['a2']))
    g['C1'] = np.einsum('btp,btpj->pj', q2, np.conj(e['C1']))
    g['a1'] = np.einsum('btp,btp->p', mu1, np.conj(e['a1']))
    g['B1'] = np.einsum('btp,btpi->pi', mu1, np.conj(e['B1']))
    return L, g


# ---- gradient-space helpers -------------------------------------------------

def flat(g, keys=KEYS):
    """Real embedding of the (complex) gradient dict."""
    out = []
    for k in keys:
        v = np.asarray(g[k]).ravel()
        out.append(np.concatenate([v.real, v.imag]) if np.iscomplexobj(v) else v)
    return np.concatenate(out)


def cos(u, v):
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-30))


def mode_view(g):
    """Stack all layer-1 parameter gradients by their mode index p -> (P1, n)."""
    return np.concatenate([np.asarray(g['a1'])[:, None], np.asarray(g['B1'])], axis=1)


def apply_w(g, w):
    """Multiply every layer-1 gradient of mode p by complex scalar w[p]. Top layer untouched."""
    out = dict(g)
    out['a1'] = g['a1'] * w
    out['B1'] = g['B1'] * w[:, None]
    return out
