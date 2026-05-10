/-
VeriTile.Triton.Math.Activation

Reusable mathematical specifications for elementwise activation kernels.

This is the pure-math layer. Definitions here only depend on Mathlib's
analytic primitives (`Real.sigmoid`, `Real.tanh`); they do not touch
`BlockState`, `RegionName`, or memory layout. Kernel transcriptions in
`bench/` and `Examples/` should reference these operators rather than
re-defining the same arithmetic.
-/

import Mathlib.Analysis.SpecialFunctions.Sigmoid
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic

namespace VeriTile.Triton

namespace TiledActivation

/-! ## Pointwise activations -/

/-- Rectified linear unit: `relu x = max 0 x`. -/
noncomputable def relu (x : ℝ) : ℝ := max 0 x

/-- Sigmoid-weighted linear unit (SiLU/Swish): `silu x = x * σ(x)`. -/
noncomputable def silu (x : ℝ) : ℝ := x * Real.sigmoid x

/-! ## SwiGLU -/

/-- SwiGLU activation: `swiglu a b = silu a · b = a · σ(a) · b`. -/
noncomputable def swiglu (a b : ℝ) : ℝ := silu a * b

/-- ∂(swiglu)/∂a · b combined with upstream gradient `dc`:
    `dc · (silu a · (1 − σ a) + σ a) · b`.
    Equivalent to `dc · σ(a) · (1 + a · (1 − σ a)) · b`. -/
noncomputable def swigluBwdA (dc a b : ℝ) : ℝ :=
  let sig := Real.sigmoid a
  dc * (silu a * (1 - sig) + sig) * b

/-- ∂(swiglu)/∂b combined with upstream gradient `dc`: `dc · silu a`. -/
noncomputable def swigluBwdB (dc a : ℝ) : ℝ := dc * silu a

/-! ## Tanh-approximation GELU and gated bilinear (gegluTanh) -/

/-- Inner argument of the tanh-approximation GELU:
    `√(2/π) · (a + 0.044715 · a³)`. -/
noncomputable def geluTanhArg (a : ℝ) : ℝ :=
  (0.7978845608028654 : ℝ) * (a + (0.044715 : ℝ) * (a * a * a))

/-- Tanh-approximation GELU core: `0.5 · a · (1 + tanh(geluTanhArg a))`. -/
noncomputable def geluTanhCore (a : ℝ) : ℝ :=
  (0.5 : ℝ) * a * (1 + Real.tanh (geluTanhArg a))

/-- Gated bilinear `geglu` with tanh-approximation GELU: `geluTanhCore a · b`. -/
noncomputable def geluTanhFwd (a b : ℝ) : ℝ := geluTanhCore a * b

/-- ∂(geluTanhFwd)/∂a combined with upstream gradient `dc` and partner `b`. -/
noncomputable def geluTanhBwdA (dc a b : ℝ) : ℝ :=
  let tanhResult := Real.tanh (geluTanhArg a)
  let term1 := (0.5 : ℝ) * (1 + tanhResult)
  let tanhSq := tanhResult * tanhResult
  let term2 := (0.5 : ℝ) * a * (1 - tanhSq) *
    ((0.7978845608028654 : ℝ) * (1 + (3.0 : ℝ) * (0.044715 : ℝ) * a * a))
  dc * b * (term1 + term2)

/-- ∂(geluTanhFwd)/∂b combined with upstream gradient `dc`:
    `dc · geluTanhCore a`. -/
noncomputable def geluTanhBwdB (dc a : ℝ) : ℝ :=
  dc * geluTanhCore a

/-! ## Fused SiLU residual gate -/

/-- Fused SiLU residual gate: `residual + silu (x · gate)`.
    Used by `fused_silu` style kernels that combine an elementwise gate, the
    SiLU activation, and a residual add into one fused store. -/
noncomputable def fusedSiLU (x gate residual : ℝ) : ℝ :=
  residual + silu (x * gate)

end TiledActivation

end VeriTile.Triton
