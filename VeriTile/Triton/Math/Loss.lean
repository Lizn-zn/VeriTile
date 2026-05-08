/-
VeriTile.Triton.Math.Loss

Reusable mathematical specifications for loss-style kernels.
-/

import Mathlib.Analysis.SpecialFunctions.Log.Basic

namespace VeriTile.Triton

namespace TiledLoss

/-- Elementwise KL-divergence lane expression `x * log (x / y)`. -/
noncomputable def klDivSpec (x y : ℝ) : ℝ :=
  x * Real.log (x / y)

end TiledLoss

end VeriTile.Triton
