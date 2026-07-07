/-
VeriTile.Math.RMSNorm

Reusable mathematical specifications for RMS normalization with an affine
scale. This is the pure-math layer: no `BlockState`, `RegionName`, or memory
layout appears here.
-/

import VeriTile.Math.L2Norm

namespace VeriTile

namespace TiledRMSNorm

/-- Mean square used by RMSNorm: `meanSq(xs, N) = (∑ xᵢ²) / N`.

The divisor is explicit because Triton kernels often reduce over a padded block
while dividing by the logical row length. -/
noncomputable def rmsMeanSq {B : Nat} (xs : Fin B → ℝ) (N : Nat) : ℝ :=
  TiledL2Norm.l2NormSqSum xs / (N : ℝ)

/-- Reciprocal RMS standard deviation with epsilon stabilizer. -/
noncomputable def rmsRstd {B : Nat} (xs : Fin B → ℝ) (N : Nat) (ε : ℝ) : ℝ :=
  1 / Real.sqrt (rmsMeanSq xs N + ε)

/-- RMS-normalized affine output at lane `i`: `xᵢ · rstd(xs, N, ε) · wᵢ`. -/
noncomputable def rmsAffine {B : Nat}
    (xs ws : Fin B → ℝ) (N : Nat) (ε : ℝ) (i : Fin B) : ℝ :=
  xs i * rmsRstd xs N ε * ws i

@[simp] theorem rmsMeanSq_eq_l2NormSqSum_div
    {B : Nat} (xs : Fin B → ℝ) (N : Nat) :
    rmsMeanSq xs N = TiledL2Norm.l2NormSqSum xs / (N : ℝ) := rfl

end TiledRMSNorm

end VeriTile
