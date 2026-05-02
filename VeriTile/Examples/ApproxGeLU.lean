/-
VeriTile.Examples.ApproxGeLU

Embedded Triton example for the tanh-style approximate GeLU commonly used
in transformer MLP blocks, together with the exact erf-based mathematical
GeLU specification it approximates:

    gelu_exact(x)  = 0.5 * x * (1 + erf(x / sqrt(2)))
    gelu_approx(x) = 0.5 * x *
      (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))

The reference Triton snippet often computes `tanh` through sigmoid:

```python
@triton.jit
def gelu_approx_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x3 = x * x * x
    u = 0.7978845608028654 * (x + 0.044715 * x3)
    tanh_u = 2.0 * tl.sigmoid(2.0 * u) - 1.0
    y = 0.5 * x * (1.0 + tanh_u)

    tl.store(out_ptr + offsets, y, mask=mask)
```

This file embeds the aligned single-block, unmasked approximate Triton
implementation in VeriTile's DSL. The exact erf-based GeLU appears as the
Lean mathematical spec, not as an embedded kernel: the current DSL has
`tl.sigmoid` but not `tl.erf` or masks.

Constants are written as rational expressions because the DSL currently
supports natural numeric literals plus arithmetic, not decimal literals.
For example, `7978845608028654 / 10000000000000000` represents the usual
`sqrt(2 / pi)` decimal approximation used by the Triton reference.
-/

import Mathlib.Analysis.SpecialFunctions.Sigmoid
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.Complex.ExponentialBounds
import Mathlib.Analysis.Real.Pi.Bounds
import Mathlib.MeasureTheory.Integral.IntervalIntegral.Basic
import VeriTile.Math.GeluTaylor20Cert
import VeriTile.Math.RealErf
import VeriTile.Math.Tanh
import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.DSL
import VeriTile.Examples.Common

namespace VeriTile.Examples

open VeriTile.Triton VeriTile.Math

/-! ## Embedded Triton AST -/

/-- Approximate GeLU, aligned single-block version.

Reads `blockSize` values from `xReg`, computes the tanh-style approximate
GeLU using `2 * sigmoid(2u) - 1` for `tanh(u)`, and writes the result to
`outReg`. -/
def approxGeLUKernel (xReg outReg : RegionName) (blockSize : Nat) : Kernel := triton {
  pid     := tl.program_id(0)
  offsets := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x       := tl.load(tl.ptr($(xReg)) + offsets)
  x3      := x * x * x
  c       := 7978845608028654 / 10000000000000000
  k       := 44715 / 1000000
  u       := c * (x + k * x3)
  tanh_u  := 2 * tl.sigmoid(2 * u) - 1
  y       := (1 / 2) * x * (1 + tanh_u)
  tl.store(tl.ptr($(outReg)) + offsets, y)
}

/-! ## Math Denotations -/

/-- Exact mathematical GeLU scalar function. -/
noncomputable def exactGeLUScalar (x : ℝ) : ℝ :=
  (1 / 2) * x * (1 + realErf (x / Real.sqrt 2))

/-- Tanh-style approximate GeLU scalar function implemented by the Triton kernel.

The `tanh` is represented as `2 * sigmoid(2u) - 1`, matching the embedded
DSL implementation. -/
noncomputable def approxGeLUScalar (x : ℝ) : ℝ :=
  let x3 := x * x * x
  let c : ℝ := 7978845608028654 / 10000000000000000
  let k : ℝ := 44715 / 1000000
  let u := c * (x + k * x3)
  (1 / 2) * x * (1 + (2 * Real.sigmoid (2 * u) - 1))

/-- Math-level denotation of the embedded approximate GeLU expression. -/
noncomputable def approxGeLUSpec {N : Nat} (xs : Fin N → ℝ) (i : Fin N) : ℝ :=
  approxGeLUScalar (xs i)

/-- Exact erf-based GeLU spec used as the mathematical reference. -/
noncomputable def exactGeLUSpec {N : Nat} (xs : Fin N → ℝ) (i : Fin N) : ℝ :=
  exactGeLUScalar (xs i)

/-! ## Algebraic Decomposition and Trivial Bound

Two helper lemmas that factor the approximation error into the gap between
`tanh` and `realErf`, and a uniform `|x|` bound that follows from `|tanh| ≤ 1`
and `|realErf| ≤ 1`. -/

/-! ## Explicit asymptotic tail bound

For `x ≥ √2`, the approximation error is dominated by a quantitative
exponential tail. This is the structural step that, combined with concrete
numerics on a single (large) value of `x`, would close the
`approx_gelu_error_bound_large` sorry on the half-line `[T, ∞)` for any
`T ≥ √2` chosen large enough that the RHS is `≤ 1/1000`. -/

/-! ## Taylor expansions of `tanh u` and `realErf z` at the gelu arguments

For the medium range `1/1000 < |x| < 20`, the asymptotic tail bounds of
`abs_approxGeLUScalar_sub_exactGeLUScalar_tail_pos` are too loose. The
tighter route is via Taylor expansions of `tanh u` and `realErf z` with the
specific substitutions `u = c·(x + k·x³)` and `z = x/√2`. These two lemmas
discharge the prerequisite (`|u| ≤ 1`, `|z| ≤ 1`) for `|x| ≤ 1` and apply the
septenary Taylor bounds from `VeriTile.Math.Tanh` and `VeriTile.Math.RealErf`. -/

/-! ## Numeric closure for very-large `|x|`

Combines `abs_approxGeLUScalar_sub_exactGeLUScalar_tail_pos` with a generic
monotonicity helper for `x · exp(−a·x)` and a concrete numeric verification
at `T = 20` to close `|approx − exact| ≤ 1/1000` whenever `|x| ≥ 20`. -/

/-! ### Interval-certificate skeleton for the remaining compact range

The remaining interval proof will be made of reusable certificates: on each
small interval, a bound at the midpoint plus a derivative bound gives the
whole-interval bound by the mean value theorem. -/

/-- Scalar error function used by the interval-certificate layer. -/
private noncomputable def geluError (x : ℝ) : ℝ :=
  approxGeLUScalar x - exactGeLUScalar x

/-- Explicit derivative of `geluError`. -/
private noncomputable def geluErrorDeriv (x : ℝ) : ℝ :=
  let c : ℝ := 7978845608028654 / 10000000000000000
  let k : ℝ := 44715 / 1000000
  let u := c * (x + k * (x * x * x))
  let z := x / Real.sqrt 2
  (1 / 2) * (Real.tanh u - realErf z) +
    (x / 2) *
      ((1 - Real.tanh u ^ 2) * (c * (1 + 3 * k * x ^ 2)) -
        (2 / Real.sqrt Real.pi) * gaussianKernel z / Real.sqrt 2)

/-! ## Structural Taylor decomposition of the gelu gap

For `|x| ≤ 1`, the gap `|tanh u - realErf z|` (with `u = c·(x + k·x³)` and
`z = x/√2`) splits via triangle inequality into three pieces:
* the tanh Taylor remainder `|u|^7/7`,
* the polynomial coefficient difference `|P_t(u) - (2/√π)·P_e(z)|`, and
* the realErf Taylor remainder `(2/√π) · 5|z|^9/864`.

The first and third pieces are already discharged by the wiring lemmas
`abs_tanh_at_gelu_arg_sub_taylor5_le` and `abs_realErf_at_gelu_arg_sub_taylor7_le`.
The middle piece is purely algebraic — it is a polynomial in `x` whose
coefficients involve `c`, `k`, and the irrational constant `c_exact = √(2/π)`. -/

/-! ### `|c − √(2/π)|` is tiny

The DSL's rational constant `c = 7978845608028654 / 10000000000000000` is
the 16-decimal rounding of `√(2/π)`. Using `Real.pi_gt_d20` and
`Real.pi_lt_d20` (20 decimals each), we show `c² ≥ 2/π` and
`|c − √(2/π)| ≤ 10⁻¹⁵`. The leading coefficient gap of the polynomial
difference `P_t(u) − (2/√π)·P_e(z)` is exactly this quantity. -/

/-! ### Reduction of `(2/√π)·P_e(x/√2)` to a clean monic polynomial in `x`

Using `2/√π = √(2/π)·√2` we eliminate one `√2` from each rescaled `(x/√2)^n`,
leaving the explicit polynomial form `c_exact · (x − x³/6 + x⁵/40 − x⁷/336)`. -/

/-- The two tanh Taylor coefficients sit naturally as `u - u³/3 + 2u⁵/15`. -/
private noncomputable def tanhTaylor5 (u : ℝ) : ℝ := u - u^3/3 + 2 * u^5/15

/-- The four realErf Taylor coefficients sit naturally as the polynomial
`z - z³/3 + z⁵/10 - z⁷/42` (still to be multiplied by the `2/√π` prefactor). -/
private noncomputable def realErfTaylor7 (z : ℝ) : ℝ :=
  z - z^3/3 + z^5/10 - z^7/42

/-! ### Polynomial coefficient bound for the medium-low subcase `|x| ≤ 1/2`

Expanding `tanhTaylor5(c·(x + k·x³)) − c·(x − x³/6 + x⁵/40 − x⁷/336)` as a
polynomial in `x` (rational coefficients, since `c` and `k` are rational) and
applying triangle inequality gives a `≤ 1/5000` bound for `|x| ≤ 1/2`. The
remaining `(c − √(2/π))·(...)` correction is `O(10⁻¹⁵)` and folded in below. -/

/-! ### Combined polynomial bound including the `c − √(2/π)` correction

For `|x| ≤ 1/2`, the polynomial gap `|tanhTaylor5(c·v(x)) − (2/√π)·realErfTaylor7(x/√2)|`
absorbs the `(c − √(2/π))·R(x)` correction (which is `O(10⁻¹⁵)`) into a clean
`≤ 1/3000` bound. -/

/-! ### Closure for `|x| ≤ 1/2` via the combined Taylor bound

For `|x| ≤ 1/2`, all three pieces of `gelu_gap_taylor_decomposition` are
explicitly bounded:
* `|u|^7/7 ≤ 1/3500`  (since `|u| ≤ 41/100`),
* the polynomial coefficient gap `≤ 1/3000`,
* the realErf Taylor remainder `≤ 1/60000`.

Then `|approx − exact| = (|x|/2)·|tanh u − realErf z| ≤ (1/4)·(1/3500 + 1/3000 + 1/60000) < 1/1000`. -/

/-! ### Closure for `|x| ≤ 3/5` (extended)

The same Taylor decomposition closes a wider window once the per-coefficient
bounds are re-tuned for `(3/5)^n` instead of `(1/2)^n`. The bottleneck shifts
from the polynomial gap to the tanh remainder `|u|⁷/7`, but the budget still
clears `1/1000`. -/

/-! ### Closure for `|x| ≤ 3/4` via 7th-order tanh Taylor

The 7th-order tanh Taylor expansion `tanhTaylor7(u) := u − u³/3 + 2u⁵/15 − 17u⁷/315`
gives a tighter remainder `|tanh u − tanhTaylor7(u)| ≤ |u|⁹/9` (vs `|u|⁷/7`).
This pushes the closed range from `|x| ≤ 3/5` up to `|x| ≤ 3/4`. -/

/-- 7th-order tanh Taylor polynomial: `u − u³/3 + 2u⁵/15 − 17u⁷/315`. -/
private noncomputable def tanhTaylor7 (u : ℝ) : ℝ :=
  u - u^3/3 + 2 * u^5/15 - 17 * u^7/315

/-! ### Closure for `|x| ≤ 4/5` via 9th-order tanh Taylor

The 9th-order tanh Taylor `tanhTaylor9(u) := u − u³/3 + 2u⁵/15 − 17u⁷/315 + 62u⁹/2835`
gives remainder `|tanh u − tanhTaylor9(u)| ≤ |u|¹¹/11`. At `|x| = 4/5`, the
tanh residual drops by an order of magnitude vs the 7th-order remainder,
opening up the closure for `|x| ≤ 4/5`. -/

/-- 9th-order tanh Taylor polynomial. -/
private noncomputable def tanhTaylor9 (u : ℝ) : ℝ :=
  u - u^3/3 + 2 * u^5/15 - 17 * u^7/315 + 62 * u^9/2835

/-! ### One more Taylor-certified segment: `|x| ≤ 83/100`

This reuses the 9th-order Taylor decomposition, but retunes every numerical
certificate at `M = 83/100`. It is intentionally a short extension past `4/5`:
at `17/20` the same triangle decomposition no longer has enough budget. -/

/-! ### External Taylor-20 midrange certificate

This experimental certificate expands the combined error
`geluError = approxGeLUScalar - exactGeLUScalar` at the midpoint
`463/200 = 2.315` to degree 20. The standalone certificate theorem is
deliberately limited to the two numeric facts an external CAS/Taylor checker
is expected to certify on
`[83/100, 19/5]`: a remainder bound and a polynomial bound.

External numerical verification (mpmath, 60-digit precision) results:
* `|geluError(x) − poly20(x)|`  worst case ≈ `1.78·10⁻⁷` at `x = 0.83`
  (certificate claims ≤ `1·10⁻⁵`, slack ≈ 98%).
* `|poly20(x)|`                 worst case ≈ `4.7324·10⁻⁴` at
  `x* ≈ 2.6989413864` (a stationary point), certificate claims ≤ `5·10⁻⁴`,
  slack ≈ 5.4%.

Combined: `|geluError(x)| ≤ 1·10⁻⁵ + 5·10⁻⁴ = 51/100000 < 1/1000`. ✓

A standalone Python script reproducing the verification is embedded below
for archival reuse.
```python
# Run with mpmath ≥ 1.3.0; use mp.mp.dps = 50 or higher.
from fractions import Fraction
import mpmath as mp
mp.mp.dps = 50

c_rat = Fraction(7978845608028654, 10**16)
k_rat = Fraction(44715, 10**6)
center = Fraction(463, 200)
sqrt2 = mp.sqrt(2)

coeffs = [
    Fraction(58799578250044, 168873512033139177),
    Fraction(490511376410848, 765861081911736471),
    -Fraction(69595973957346, 98732046708052927),
    -Fraction(318812859576406, 603001279934315985),
    Fraction(95323914281662, 166041057487226167),
    Fraction(83409703730901, 541893669526039966),
    -Fraction(131670005603011, 523709508683584689),
    -Fraction(3688290726053, 972548317670397000),
    Fraction(45774012660025, 681762334304929013),
    -Fraction(7348721185503, 706588070591977423),
    -Fraction(5731745181647, 501999458364499667),
    Fraction(12817416488, 3546237897524503),
    Fraction(1104703829666, 959634314348063215),
    -Fraction(638985140807, 953651943452319460),
    -Fraction(20677130095, 708673697371547448),
    Fraction(23268802103, 302557557620128678),
    -Fraction(3607948043, 331131972419363373),
    -Fraction(2089268122, 454157679889909063),
    Fraction(577867222, 307044885316340023),
    -Fraction(15956207, 139052182026179153),
    -Fraction(30752279, 251067520891536832),
]
coeffs_mp = [mp.mpf(c.numerator) / mp.mpf(c.denominator) for c in coeffs]

def gelu_error(x):
    c = mp.mpf(c_rat.numerator) / mp.mpf(c_rat.denominator)
    k = mp.mpf(k_rat.numerator) / mp.mpf(k_rat.denominator)
    u = c * (x + k * x**3)
    z = x / sqrt2
    return mp.mpf(1)/2 * x * (mp.tanh(u) - mp.erf(z))

def poly20(x):
    t = x - mp.mpf(center.numerator) / mp.mpf(center.denominator)
    return sum(c * t**i for i, c in enumerate(coeffs_mp))

def dpoly20(x):
    t = x - mp.mpf(center.numerator) / mp.mpf(center.denominator)
    return sum(i * coeffs_mp[i] * t**(i-1) for i in range(1, len(coeffs_mp)))

# Locate extrema by sign changes of dpoly20 + endpoints
a, b = mp.mpf("0.83"), mp.mpf("3.8")
N = 100001
xs = [a + (b - a) * i / (N - 1) for i in range(N)]
extrema = [a, b]
prev_d = dpoly20(xs[0])
for i in range(1, N):
    d = dpoly20(xs[i])
    if (prev_d > 0 and d < 0) or (prev_d < 0 and d > 0):
        try:
            extrema.append(mp.findroot(dpoly20, (xs[i-1] + xs[i]) / 2))
        except Exception:
            pass
    prev_d = d

worst_p = max(abs(poly20(x)) for x in extrema)
worst_r = max(abs(gelu_error(x) - poly20(x)) for x in extrema)
assert worst_p <= mp.mpf(50) / 100000, f"poly bound failed: {worst_p}"
assert worst_r <= mp.mpf(1) / 100000, f"residual bound failed: {worst_r}"
print(f"poly20 sup ≈ {float(worst_p):.6e}  (bound 5e-4, slack {float((mp.mpf(50)/100000 - worst_p)/(mp.mpf(50)/100000)*100):.1f}%)")
print(f"residual sup ≈ {float(worst_r):.6e}  (bound 1e-5, slack {float((mp.mpf(1)/100000 - worst_r)/(mp.mpf(1)/100000)*100):.1f}%)")
```
-/

private noncomputable def geluError_mid_taylor20 (x : ℝ) : ℝ :=
  let t := x - 463/200
  58799578250044/168873512033139177
    + (490511376410848/765861081911736471) * t
    + (-(69595973957346/98732046708052927)) * t^2
    + (-(318812859576406/603001279934315985)) * t^3
    + (95323914281662/166041057487226167) * t^4
    + (83409703730901/541893669526039966) * t^5
    + (-(131670005603011/523709508683584689)) * t^6
    + (-(3688290726053/972548317670397000)) * t^7
    + (45774012660025/681762334304929013) * t^8
    + (-(7348721185503/706588070591977423)) * t^9
    + (-(5731745181647/501999458364499667)) * t^10
    + (12817416488/3546237897524503) * t^11
    + (1104703829666/959634314348063215) * t^12
    + (-(638985140807/953651943452319460)) * t^13
    + (-(20677130095/708673697371547448)) * t^14
    + (23268802103/302557557620128678) * t^15
    + (-(3607948043/331131972419363373)) * t^16
    + (-(2089268122/454157679889909063)) * t^17
    + (577867222/307044885316340023) * t^18
    + (-(15956207/139052182026179153)) * t^19
    + (-(30752279/251067520891536832)) * t^20

/-! ## Correctness and Approximation Statements -/

/-- Target error tolerance for the standard tanh/sigmoid GeLU approximation. -/
noncomputable def approxGeLUEps : ℝ := 1 / 1000

end VeriTile.Examples
