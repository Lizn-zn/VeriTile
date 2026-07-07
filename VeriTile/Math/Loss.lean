/-
VeriTile.Math.Loss

Reusable mathematical specifications for loss-style kernels.
-/

import Mathlib.Analysis.SpecialFunctions.Log.Basic
import VeriTile.Math.LogSumExp

namespace VeriTile

namespace TiledLoss

open VeriTile.TiledLogSumExp

/-- Elementwise KL-divergence lane expression `x * log (x / y)`. -/
noncomputable def klDivSpec (x y : ℝ) : ℝ :=
  x * Real.log (x / y)

/-- Canonical textbook cross-entropy loss for a single example.

Given a length-`V` row of logits `xs` and a target class `t`, this is the
log-sum-exp of the logits minus the target logit:
`crossEntropyLoss xs t = log (∑ exp xs) − xs t`.  Equivalently, it is
`−log (softmax(xs) t)`.  The log-sum-exp term is written via the numerically
stable `stableLSE` (max-shifted) form with no per-row scaling; by
`stableLSE_eq_LSE` it equals the naive `log (∑ exp ·)`.  This is the shared
pure mathematical core that the `fast_ce_loss`, `cross_entropy1`,
`cross_entropy2`, and `cross_entropy_ops` kernels compute in their base
(no-softcap, no-scaling, no-smoothing) regime. -/
noncomputable def crossEntropyLoss {V : Nat} (xs : Fin V → ℝ) (t : Fin V)
    (hV : 0 < V) : ℝ :=
  stableLSE xs hV false 0 - xs t

/-- Canonical textbook cross-entropy loss with label smoothing.

With smoothing strength `ε`, the loss linearly interpolates between the hard
target loss and the uniform-target loss:
`(1−ε)·(LSE − xs t) + ε·(LSE − (∑ xs)/V)`, i.e. the smoothed target assigns
probability `1−ε` to the true class and spreads `ε` uniformly over all `V`
classes (so the soft cross-entropy is `LSE − ((1−ε)·xs t + ε·mean xs)`).  The
log-sum-exp uses the stable max-shifted form with no scaling, matching
`crossEntropyLoss`.  This is the shared pure core for the smoothed branch of
the `cross_entropy*` kernels. -/
noncomputable def crossEntropyLossSmoothed {V : Nat} (xs : Fin V → ℝ) (t : Fin V)
    (ε : ℝ) (hV : 0 < V) : ℝ :=
  (1 - ε) * (stableLSE xs hV false 0 - xs t)
    + ε * (stableLSE xs hV false 0 - (∑ i, xs i) / V)

end TiledLoss

end VeriTile
