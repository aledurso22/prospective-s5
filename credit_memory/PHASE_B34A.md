# Phase B34a — nonlinear truncated-polynomial / jet-algebra exact RTRL: correctness

Branch `S5-CCM-scale-validation`. Code:
`credit_memory/b34a_jet_algebra_correctness.py` (`main()` reproduces
every number below; full logs at `/tmp/b34a_full2.log` this session).
An actual trainable nonlinear recurrent model — `J_t=D_hF_theta`,
`G_t=D_thetaF_theta` obtained via genuine autodiff — legitimate RTRL,
stronger than B33b-min: here `P=r` and **all** r trainable directions
of `theta` genuinely affect the recurrent transformation (not merely
an additive source term).

**Headline: on the jet algebra `A_r=R[eps]/(eps^r)`, rank(D_theta h_t)=r
generically at r∈{4,16,64} (confirmed via the mathematically exact
criterion, not just SVD), while the exact persistent eligibility state
is only r scalars — and the reduction dimension exactly saturates the
fixed-module lower bound (reduced eligibility dim = generated-algebra
dim = r).**

## 1. Model

`theta,h_t,a_t,b_t,kappa_t,c_t ∈ A_r` (length-r coefficient vectors);
algebra multiplication = truncated convolution, implemented via the
exact lower-triangular Toeplitz regular-representation matrix
`M_u` (`(M_u)_{k,i}=u_{k-i}` for `k≥i`, else 0) — direct float64, not
FFT, per instruction. `a_t,b_t,kappa_t,c_t` generated causally/
exogenously from an external input by a small FIXED FROZEN (never
trained) MLP; `kappa_t[0]` forced into `[0.2,0.8]` to guarantee
`kappa_t` is always a unit (an element of `A_r` is invertible iff its
constant term is nonzero — `A_r` is local with maximal ideal `(eps)`).

```
y_t = (a_t + kappa_t*theta)*h_t + b_t*theta + c_t
phi(y) = c1*y + c3*y^{*3}                 (c1=1.0, c3=0.05)
h_{t+1} = phi(y_t)
d_t = phi'(y_t) = c1*1 + 3c3*y_t^{*2}
```

Verified algebraically AND numerically: `J_t=M_{d_t*(a_t+kappa_t*theta)}`,
`G_t=M_{d_t*(kappa_t*h_t+b_t)}`; because `M_uM_v=M_{u*v}` exactly (the
regular representation is a ring homomorphism), induction gives
`S_t=M_{s_t}` for all t with the boxed reduced recursion
`s_{t+1}=d_t*[(a_t+kappa_t*theta)*s_t+kappa_t*h_t+b_t]` — exactly as
specified, now independently confirmed against autodiff.

## 2. Correctness (5 seeds × T∈{1,5,20,100,500}, r∈{4,16,64})

| r | worst full-vs-BPTT rel err | worst reduced-vs-BPTT rel err | worst S_recon err | max\|h_t\| (stability) |
|---|---|---|---|---|
| 4 | 2.852e-15 | 2.502e-15 | 5.551e-17 | 0.2061 |
| 16 | 1.951e-15 | 1.966e-15 | 2.776e-16 | 0.2900 |
| 64 | 1.369e-15 | 1.347e-15 | 1.066e-14 | 6.1470 |

`ALL < 1e-8: True` at every r. Gradients from BPTT, full RTRL, and the
reduced algebra recursion agree at machine precision throughout, and
`S_t` reconstructed from `s_t` alone matches full RTRL's materialized
`S_t` to the same precision. `T=500` at r=64 stayed bounded (max
`|h_t|`≈6.15, no blowup) confirming the conservative coefficient
scales (`c1=1,c3=0.05`, bounded `a_t,b_t,c_t,kappa_t`) keep the cubic
nonlinearity stable over long sequences.

## 3. The rank(S_t)=r diagnostic — an important correction made in-phase

**Naive SVD-based numerical rank badly under-reports "full rank" at
larger r — this is a conditioning artifact, not a real degeneracy, and
had to be diagnosed rather than taken at face value.**

| r | mean SVD-based frac rank=r | mean EXACT (diagonal-criterion) frac rank=r | median cond(S_t) sampled | min \|s_t[0]\| sampled |
|---|---|---|---|---|
| 4 | 1.0000 | 1.0000 | 1.6e+01 | 4.0e-04 |
| 16 | 0.7260 | **1.0000** | 2.6e+05 | 1.1e-04 |
| 64 | 0.2260 | **1.0000** | 7.6e+13 | 2.6e-04 |

`M_u` is lower-triangular Toeplitz with a **constant diagonal equal to
u's own constant term** (`u_0`, i.e. `s_t[0]` here), so
`det(M_{s_t})=s_t[0]^r` **exactly** — full rank iff `s_t[0]≠0`. Direct
inspection (verified numerically for a representative r=64 trajectory)
showed `s_t[0]` clearly and consistently nonzero (~0.01–0.07 in
magnitude, never near machine epsilon) at every sampled timestep, yet
the smallest singular value of the SAME matrix collapsed to
~1e-18–1e-19 (condition numbers routinely 1e17–1e20) — a naive
`sv > 1e-9·sv_max` SVD threshold therefore reports "rank<r" purely from
float64 roundoff on an extremely ill-conditioned nilpotent-plus-scalar
Jordan-like structure, not because the matrix is actually singular.
Using the mathematically exact criterion (`|s_t[0]|>1e-6`, the only
criterion this specific triangular structure actually requires): **100%
generic full rank at every r tested (4, 16, 64)**, exactly as the
theorem predicts. `rank(S_t)=r` is confirmed generically; it is
explicitly **not** described as low-rank.

## 4. Storage accounting

| | r=4 | r=16 | r=64 |
|---|---|---|---|
| Full RTRL persistent eligibility | r²=16 | r²=256 | r²=4,096 |
| Reduced algebra eligibility | r=4 | r=16 | r=64 |
| **Ratio** | **4x** | **16x** | **64x** |

Ratio is exactly `r` at every scale, as specified. `theta,h_t,a_t,b_t,
kappa_t,c_t` (all length-r) are forward/model state, not hidden from
the accounting; the claim concerns ADDITIONAL persistent eligibility
storage specifically, matching the same honest framing established in
B33a/B33b-min.

## 5. Structural algebra diagnostics (r=4, representative)

The regular-representation basis `{M_1,M_eps,...,M_{eps^{r-1}}}` spans
exactly an r-dimensional commutative subalgebra of `Mat_r(R)` by
construction (each `M_{eps^k}` is linearly independent — they are the
r distinct "shift-by-k" lower-triangular Toeplitz matrices). Sampled
Jacobians `J_t=M_{d_t*(a_t+kappa_t*theta)}` lie in this algebra by
construction (they are themselves regular-representation matrices of
an `A_r` element). Multiplication matrices commute (`M_uM_v=M_{u*v}=
M_{v*u}=M_vM_u`, since `A_r` is a commutative ring) — this is stated
plainly, not disguised as irreducibility: **the dynamics are
commutative and structurally reducible in the sense of a shared
eigenbasis-like flag** (`M_eps` is a single nilpotent Jordan chain of
length r, not a direct sum of smaller blocks). The interesting property
established here is **indecomposable/nonsemisimple coupling** — the
nilpotent generator ties all r coordinates into one Jordan chain, so
`theta`'s r components cannot be split into independent scalar/2×2
sub-blocks — not irreducibility in the representation-theoretic sense.

## 6. Status / next steps

B34a (correctness + rank diagnostic + storage + structural checks)
complete and archived here. B34b (teacher/student training) and B34c
(closure falsification) are the remaining pieces of this phase,
per the "critical recurrent-parameter diagnostic" (directional
derivative of `J_t` w.r.t. `theta` having rank r when `kappa_t` is a
unit) still to be added explicitly before B34b, if pursued in a
follow-up.

## 7. Commit hash

See the commit introducing this file.
