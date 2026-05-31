/-
VeriTile.Triton.Math.Optimizer

Reusable mathematical specifications for optimizer-update kernels (Lion, and
forthcoming Adam/AdamW variants).

This is the pure-math layer. Definitions here only depend on Mathlib's
`Real.sign` and ordered-field arithmetic; they do not touch `BlockState`,
`RegionName`, or memory layout. Kernel transcriptions in `bench/` and
`Examples/` should bridge their per-lane carriers to these operators rather
than re-deriving the same formula inline.

Reference: Chen et al., "Symbolic Discovery of Optimization Algorithms"
(Lion); decoupled-weight-decay form as in the widely used `lion-pytorch`
implementation:

    update      = sign(β₁·m + (1−β₁)·g)
    p           ← (1 − lr·wd)·p − lr·update
    m (exp_avg) ← β₂·m + (1−β₂)·g
-/

import Mathlib.Data.Real.Sign
import Mathlib.Tactic.Ring

namespace VeriTile.Triton

namespace TiledOptimizer

/-! ## Lion optimizer -/

/-- Lion momentum (`exp_avg`) update: the exponential moving average
    `β₂·m + (1−β₂)·g`. -/
noncomputable def lionMomentum (m g β₂ : ℝ) : ℝ :=
  β₂ * m + (1 - β₂) * g

/-- Lion pre-sign update direction: the interpolation
    `β₁·m + (1−β₁)·g`. The realized parameter step is `−lr · sign` of this. -/
noncomputable def lionUpdateDir (m g β₁ : ℝ) : ℝ :=
  β₁ * m + (1 - β₁) * g

/-- Lion parameter update with decoupled weight decay:
    `(1 − lr·wd)·p − lr · sign(β₁·m + (1−β₁)·g)`. -/
noncomputable def lionParam (p m g lr wd β₁ : ℝ) : ℝ :=
  (1 - lr * wd) * p - lr * Real.sign (lionUpdateDir m g β₁)

/-! ## Lion repetition/frequency/presence penalty -/

/-- Lion's per-token logit penalty (repetition + frequency + presence): the
sign-based repetition rescale `logit/rep if logit>0 else logit·rep`, then
subtract `count·freq` and `pres`. -/
noncomputable def lionPenalty (logit count rep freq pres : ℝ) : ℝ :=
  (if logit > 0 then logit / rep else logit * rep) - count * freq - pres

/-! ## Bridging identities

Triton kernels commonly compute the momentum/direction in the factored form
`β·(m − g) + g` (one `tl.load` of the running average, one subtraction, one
fused multiply-add) and realize the sign step via a branch
`where(d > 0, −lr, lr)` gated by `d ≠ 0`. These identities connect those
forms to the definitions above. -/

/-- `β₂·m + (1−β₂)·g = β₂·(m − g) + g` — the factored momentum form emitted by
    kernels. -/
theorem lionMomentum_eq_factored (m g β₂ : ℝ) :
    lionMomentum m g β₂ = β₂ * (m - g) + g := by
  unfold lionMomentum; ring

/-- `β₁·m + (1−β₁)·g = β₁·(m − g) + g` — the factored direction form. -/
theorem lionUpdateDir_eq_factored (m g β₁ : ℝ) :
    lionUpdateDir m g β₁ = β₁ * (m - g) + g := by
  unfold lionUpdateDir; ring

/-- The realized Lion step `−lr · sign d` equals the kernel branch
    `if d = 0 then 0 else if 0 < d then −lr else lr`. -/
theorem lion_step_eq_branch (lr d : ℝ) :
    -lr * Real.sign d =
      (if d = 0 then 0 else if 0 < d then -lr else lr) := by
  rcases lt_trichotomy d 0 with h | h | h
  · rw [Real.sign_of_neg h, if_neg h.ne, if_neg (not_lt.mpr h.le)]; ring
  · subst h; rw [Real.sign_zero]; simp
  · rw [Real.sign_of_pos h, if_neg h.ne', if_pos h]; ring

end TiledOptimizer

end VeriTile.Triton
