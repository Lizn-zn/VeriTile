---
title: "ApproxGeLU Midrange Strategy: Phi / Critical-Point Proof"
---

Goal:

```text
|approxGeLU(x) - exactGeLU(x)| <= 1e-3
```

where the implementation uses

```text
c0 = 7978845608028654 / 10000000000000000
k  = 44715 / 1000000
A(x) = c0 * (x + k*x^3)
g(x) = tanh(A(x)) - erf(x/sqrt(2))
D(x) = x/2 * g(x)
```

`D` is even, so it suffices to prove the positive half-line.

Existing coverage:

```text
0 <= x <= 0.83      Taylor-at-zero proof
3.8 <= x           separate tail proof
```

Promising midrange proof for `[0.83, 3.8]`:

Let

```text
alpha = sqrt(2/pi)
Phi(x) =
  log(c0 / alpha)
  + log(1 + 3*k*x^2)
  + x^2/2
  - 2 * log(cosh(A(x))).
```

Then

```text
g'(x)
  = c0*(1+3*k*x^2)*sech(A(x))^2 - alpha*exp(-x^2/2)
  = alpha*exp(-x^2/2) * (exp(Phi(x)) - 1).
```

This keeps the tanh/erf derivative cancellation inside `Phi`.

Numerical sign structure using real arithmetic:

```text
Phi zeros:
  alpha1 ~= 1.2420843929
  alpha2 ~= 2.5921448834

D' zeros:
  x1 ~= 1.3644977174
  x2 ~= 2.6989413864

D(0.83) ~= -9.3620e-5
D(x1)  ~= -2.3367e-4
D(x2)  ~=  4.7324e-4
D(3.8) ~=  9.8804e-5
```

For formalization we use slightly wider critical boxes to avoid tiny sign
margins at the numerical roots:

```text
box1 = [1.36449, 1.36451], center 1.3645
box2 = [2.69893, 2.69895], center 2.69894
```

If one proves the sign table

```text
D'(x) < 0 on [0.83, 1.36449)
D'(x) > 0 on (1.36451, 2.69893)
D'(x) < 0 on (2.69895, 3.8]
```

and proves the two center values

```text
|D(1.3645)| <= 1/2000
|D(2.69894)| <= 1/2000
```

then the Lean checker already proves the midrange. The boxes are small enough
that the crude global derivative bound `|D'| <= 10` on `[0,4]` controls all
points in each box.

Formalization plan:

1. Define `geluPhi`.
2. Prove the derivative cancellation identity:

   ```text
   geluHDerivExpr x (geluY x) (geluG x)
     = sqrt(2/pi) * exp(-x^2/2) * (exp(geluPhi x) - 1)
   ```

3. Prove or generate interval sign certificates for `geluPhi` and `D'`.
4. Use MVT/monotonicity to reduce `[0.83, 3.8]` to endpoint/critical-box
   value checks.

Important correction from informal derivations: the implemented GeLU uses the
decimal rational `c0`, not definitionally `sqrt(2/pi)`, so `log(c0/alpha)` is
required in `Phi`.

## ODE Affine-Tube Update

The later generated-certificate route bypasses the critical-point sign table:
prove a tube for `F = x/2 * H` directly on the midrange.

For each segment `[a,b]`, the candidate data uses an affine predictor

```text
f(x) ~= f0 + (x-a) * d0
```

with an initial error `E0` and derivative residual `R`:

```text
|f(a) - f0| <= E0
|f'(y) - d0| <= R
```

The Lean checker `affine_step_forall_mem` proves a scalar tube from those
facts. Numerically, interval-affine certificates with step `0.02` cover
`[0,3.8]`; the actual midrange `[0.83,3.8]` requires 149 rows. Using only the
rational bound `alpha in [79/100, 4/5]`, the scout reports

```text
step = 0.02
worst |F| tube ~= 5.23e-4
```

The formal caveat is important: residual proofs for `H'` and `F'` want to use
the same state tube being proved. A generated row cannot simply assume
`Y/G/H/F` are inside the tube to prove the residual; that would be circular.
The Lean-side abstraction now used for this is a self-consistent tube invariant
/ first-exit lemma:

```text
if
  the initial state is inside the tube,
  whenever the state is inside the tube the RHS maps into the derivative box,
  and the affine step maps that derivative box strictly inside the tube,
then
  the true state stays inside the tube on the whole segment.
```

Once this theorem exists, each generated row reduces to rational polynomial RHS
inclusions plus rational budget checks.

Current Lean hooks:

```text
Icc_subset_of_closed_local_right_extension
affine_state_step_F_tube_of_local_right_extension
affine_state_step_F_tube_of_strict_prefix
```

The generator should target `affine_state_step_F_tube_of_strict_prefix`: prove
the left endpoint state box, then under the prefix invariant `Icc a t` prove
eight strict slack inequalities for `Y/G/H/F` at every `t < b`. Continuity turns
those strict inequalities into the required right-neighborhood extension.
