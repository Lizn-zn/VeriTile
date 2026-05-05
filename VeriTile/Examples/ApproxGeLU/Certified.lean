/-
VeriTile.Examples.ApproxGeLU.Certified

Split-out support for the ApproxGeLU example.
-/

import VeriTile.Examples.ApproxGeLU.Taylor9

namespace VeriTile.Examples

open VeriTile.Triton VeriTile.Math

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

noncomputable def geluError_mid_taylor20 (x : ℝ) : ℝ :=
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

theorem geluError_mid_taylor20_cert :
    ∀ x ∈ Set.Icc (83/100 : ℝ) (19/5),
      |geluError x - geluError_mid_taylor20 x| ≤ 1/100000 ∧
      |geluError_mid_taylor20 x| ≤ 50/100000 := by
  intro x hx
  simpa [geluError, geluError_mid_taylor20,
    VeriTile.Math.geluErrorForCert,
    VeriTile.Math.approxGeLUScalarForCert,
    VeriTile.Math.exactGeLUScalarForCert,
    VeriTile.Math.geluError_mid_taylor20_forCert,
    approxGeLUScalar, exactGeLUScalar]
    using VeriTile.Math.geluError_mid_taylor20_cert_standalone x hx

theorem approx_gelu_error_bound_midrange_taylor20 {x : ℝ}
    (hx : x ∈ Set.Icc (83/100 : ℝ) (19/5)) :
    |geluError x| ≤ 1/1000 := by
  have hcert := geluError_mid_taylor20_cert x hx
  calc |geluError x|
      = |(geluError x - geluError_mid_taylor20 x) + geluError_mid_taylor20 x| := by
          ring_nf
    _ ≤ |geluError x - geluError_mid_taylor20 x| + |geluError_mid_taylor20 x| :=
          abs_add_le _ _
    _ ≤ 1/100000 + 50/100000 := add_le_add hcert.1 hcert.2
    _ ≤ 1/1000 := by norm_num

theorem approx_gelu_error_bound_medium_taylor20 {x : ℝ}
    (_hxlow : (1/1000 : ℝ) < |x|) (hxhigh : |x| < 19/5) :
    |approxGeLUScalar x - exactGeLUScalar x| ≤ 1/1000 := by
  rcases le_or_gt (|x|) (83/100) with h083 | hmid
  · exact approx_gelu_error_bound_083 h083
  · have hx_abs_mem : |x| ∈ Set.Icc (83/100 : ℝ) (19/5) :=
      ⟨le_of_lt hmid, le_of_lt hxhigh⟩
    have hpos := approx_gelu_error_bound_midrange_taylor20 hx_abs_mem
    change |geluError x| ≤ 1 / 1000
    rwa [geluError_abs_arg x] at hpos


end VeriTile.Examples
