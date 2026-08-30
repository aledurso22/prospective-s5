# Phase B35h-online — direction-independent 2P Hessian carry

Branch `S5-CCM-scale-validation`. Diagnostic only: no training, no
tuning. Code: `credit_memory/b35h_online_2P_carry.py`
(`/tmp/b35h_online_2P.log`). C=64, seed=11, checkpoints t=(50,150,300),
5 random unit directions drawn PER checkpoint AFTER r_t was already
propagated.

## The problem B35h left open

B35h's `w_t` recursion required a direction Delta fixed from t=0 --
not usable online, since a real continual learner's update direction
changes every step and is only known after the current gradient.

## Derivation (by induction, using only associativity/bilinearity of the commutative algebra product, already established throughout this project)

Assume `w_k = alg_mult(r_k, Delta)` for some direction-independent
`r_k`. B35h's recursion `w_{k+1}=alg_mult(theta,w_k)+2*alg_mult(s_k,Delta)`
becomes:

```
w_{k+1} = alg_mult(theta, alg_mult(r_k,Delta)) + alg_mult(2*s_k, Delta)
        = alg_mult( alg_mult(theta,r_k) + 2*s_k , Delta )     [associativity + bilinearity]
=> r_{k+1} = alg_mult(theta, r_k, Q, d) + 2*s_k,   r_0 = 0.
```

**No Delta appears in r_t's own recursion at all** -- it can be carried
online, updated every step alongside s_t, with the actual optimizer
direction supplied only at the moment it's needed (after the current
gradient is known), via `H_t[Delta] = alg_mult(r_t, Delta)`.

One further associativity step gives the quadratic contraction:
```
H_t[Delta,Delta] = alg_mult(w_t, Delta) = alg_mult(alg_mult(r_t,Delta), Delta)
                  = alg_mult(r_t, alg_mult(Delta,Delta))  [associativity]
                  = alg_mult(r_t, Delta*Delta).
```

## Verification (r_t propagated ONCE per checkpoint; 5 directions drawn only afterward)

All 15 (checkpoint x direction) tests, both identities, against
brute-force `jax.jacobian(jax.jacobian(rollout))`:

| t | H_t[Delta] rel_err (worst of 5 directions) | H_t[Delta,Delta] rel_err (worst of 5) |
|---|---|---|
| 50 | 4.05e-16 | 5.00e-16 |
| 150 | 2.72e-16 | 3.34e-16 |
| 300 | 2.98e-16 | 3.89e-16 |

Worst across all 15 tests: H_t[Delta] rel_err=4.05e-16, H_t[Delta,Delta]
rel_err=5.00e-16. **Both identities verified to machine precision at
every checkpoint and every direction, with r_t computed BEFORE any
direction was chosen.**

`|r_t|` = P = 64 exactly at every checkpoint (independent of t, the
sequence length already processed). `s_t` (P) + `r_t` (P) = 2P = 128
total persistent scalars -- independent of sequence length AND
independent of any later-chosen optimizer direction.

## Conclusion

All identities hold. **The B35h correction is implementable as a
genuine online 2P carry**: `r_t` is direction-independent, satisfies
its own O(P)-cost recursion propagated alongside `s_t` with no
knowledge of future update directions, and reconstructs the exact
Hessian-vector product (both single and quadratic contraction) for
*any* direction supplied after the fact, to machine precision. This
resolves B35h's remaining objection (a fixed-direction HVP cannot be
the persistent online object) -- the fix is not a new mechanism but a
factorization already implied by the algebra's own associativity.

No training or tuning was performed here, per instruction.

## Commit hash

See the commit introducing this file.
