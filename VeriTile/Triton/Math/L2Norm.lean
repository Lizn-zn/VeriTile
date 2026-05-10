/-
VeriTile.Triton.Math.L2Norm

Reusable mathematical specifications for row-wise L2 normalization (RMS-norm
without an affine scale): forward output, reciprocal standard deviation, and
the gradient w.r.t. the input.

This is the pure-math layer. Definitions here only depend on Mathlib's
analytic primitives and big-operators; they do not touch `BlockState`,
`RegionName`, or memory layout. Kernel transcriptions in
`bench/l2_norm_*` should bridge their `Tile.reduceSum`-based carriers to
these operators rather than re-deriving the same formula.
-/

import Mathlib.Data.Real.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.WithTop
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace VeriTile.Triton

namespace TiledL2Norm

/-- Squared 2-norm: `‖xs‖² = ∑ xᵢ²`. -/
noncomputable def l2NormSqSum {N : Nat} (xs : Fin N → ℝ) : ℝ :=
  ∑ i, xs i * xs i

/-- Reciprocal standard deviation with epsilon stabilizer:
    `rstd(xs, ε) = 1 / √(‖xs‖² + ε)`. -/
noncomputable def l2NormRstd {N : Nat} (xs : Fin N → ℝ) (ε : ℝ) : ℝ :=
  1 / Real.sqrt (l2NormSqSum xs + ε)

/-- L2-normalized output at lane `i`: `xᵢ · rstd(xs, ε)`. -/
noncomputable def l2Norm {N : Nat} (xs : Fin N → ℝ) (ε : ℝ) (i : Fin N) : ℝ :=
  xs i * l2NormRstd xs ε

/-- Cross dot product `⟨dy, x⟩ = ∑ dyᵢ · xᵢ`, used by the L2-norm backward. -/
noncomputable def l2NormDot {N : Nat} (xs ys : Fin N → ℝ) : ℝ :=
  ∑ i, xs i * ys i

/-- Gradient of L2-normalize w.r.t. input at lane `i`:
    `dy · rstd  −  ⟨dy, x⟩ / (‖xs‖² + ε) · rstd · xᵢ`. -/
noncomputable def l2NormBwd {N : Nat}
    (xs dys : Fin N → ℝ) (ε : ℝ) (i : Fin N) : ℝ :=
  let rstd := l2NormRstd xs ε
  let dot := l2NormDot dys xs
  dys i * rstd - dot * (1 / (l2NormSqSum xs + ε)) * rstd * xs i

/-! ## Carrier bridges

Single-rewrite reductions from the kernel-shape `Tile.reduceSum (Tile.bop mul …)`
carrier to the pure `∑` form. Decoupled from `l2InputTile` so any L2-style
kernel can plug in its own `(load, active)` pair.

These two lemmas exist because simp's bottom-up traversal of generic
`if c then some _ else some _` ⇒ `some (if c then _ else _)` did not fire
inside the kernel proof context (likely a `WithBot α` vs `Option α`
unification issue under `FloatDType.cast` wrappers). We close the whole
`∑ k, Option.map₂ (·*·) (if-shape) (if-shape)` reduction in a single
shape-matching theorem and let downstream simp handle the outer
`Option.map / WithBot.realSqrt / WithBot.unbotD` layer with existing
lemmas. -/

section CarrierBridges

variable {BLOCK_N : Nat}

/-- Bridge: kernel-shape sum of squares with masked tail = pure `∑` form.

The kernel produces, for each lane `k`, a `WithBot ℝ` value that is `xs k`
when `active k` and `0` otherwise — wrapped in two nested `if active k`s
because `tl.load(..., other=0.0)` and `tl.where(active k, _, 0)` both gate
on the same condition. -/
theorem reduceSum_masked_sq_eq_some_sum
    (load : Fin BLOCK_N → ℝ)
    (active : Fin BLOCK_N → Prop) [DecidablePred active] :
    @Finset.sum (Fin BLOCK_N) (WithBot ℝ) _ Finset.univ
        (fun k => Option.map₂ (· * ·)
          (if active k then
            (if active k then (some (load k) : WithBot ℝ)
             else (some (0.0 : ℝ) : WithBot ℝ))
           else (some (0.0 : ℝ) : WithBot ℝ))
          (if active k then
            (if active k then (some (load k) : WithBot ℝ)
             else (some (0.0 : ℝ) : WithBot ℝ))
           else (some (0.0 : ℝ) : WithBot ℝ)))
      = some (∑ k, if active k then load k * load k else 0) := by
  have hcongr :
      ∀ k, Option.map₂ (· * ·)
          (if active k then
            (if active k then (some (load k) : WithBot ℝ)
             else (some (0.0 : ℝ) : WithBot ℝ))
           else (some (0.0 : ℝ) : WithBot ℝ))
          (if active k then
            (if active k then (some (load k) : WithBot ℝ)
             else (some (0.0 : ℝ) : WithBot ℝ))
           else (some (0.0 : ℝ) : WithBot ℝ))
        = (((if active k then load k * load k else 0 : ℝ)) : WithBot ℝ) := by
    intro k
    by_cases h : active k
    · simp only [if_pos h]; rfl
    · simp only [if_neg h]
      show Option.map₂ (· * ·) (some (0.0 : ℝ)) (some 0.0) = ((0 : ℝ) : WithBot ℝ)
      have h0 : (0.0 : ℝ) * 0.0 = 0 := by norm_num
      rw [show Option.map₂ (· * ·) (some (0.0 : ℝ)) (some 0.0) = some ((0.0 : ℝ) * 0.0) from rfl,
          h0]
      rfl
  calc @Finset.sum (Fin BLOCK_N) (WithBot ℝ) _ Finset.univ _
      = ∑ k, (((if active k then load k * load k else 0 : ℝ)) : WithBot ℝ) :=
        Finset.sum_congr rfl (fun k _ => hcongr k)
    _ = ((∑ k, if active k then load k * load k else 0 : ℝ) : WithBot ℝ) :=
        (WithBot.coe_sum Finset.univ _).symm
    _ = some (∑ k, if active k then load k * load k else 0) := rfl

/-- Bridge: kernel-shape cross-product with masked tail = pure `∑` form.
Used by the L2-norm backward kernel for the `⟨dy, x⟩` dot. -/
theorem reduceSum_masked_dot_eq_some_sum
    (xs ys : Fin BLOCK_N → ℝ)
    (active : Fin BLOCK_N → Prop) [DecidablePred active] :
    @Finset.sum (Fin BLOCK_N) (WithBot ℝ) _ Finset.univ
        (fun k => Option.map₂ (· * ·)
          (if active k then
            (if active k then (some (xs k) : WithBot ℝ)
             else (some (0.0 : ℝ) : WithBot ℝ))
           else (some (0.0 : ℝ) : WithBot ℝ))
          (if active k then
            (if active k then (some (ys k) : WithBot ℝ)
             else (some (0.0 : ℝ) : WithBot ℝ))
           else (some (0.0 : ℝ) : WithBot ℝ)))
      = some (∑ k, if active k then xs k * ys k else 0) := by
  have hcongr :
      ∀ k, Option.map₂ (· * ·)
          (if active k then
            (if active k then (some (xs k) : WithBot ℝ)
             else (some (0.0 : ℝ) : WithBot ℝ))
           else (some (0.0 : ℝ) : WithBot ℝ))
          (if active k then
            (if active k then (some (ys k) : WithBot ℝ)
             else (some (0.0 : ℝ) : WithBot ℝ))
           else (some (0.0 : ℝ) : WithBot ℝ))
        = (((if active k then xs k * ys k else 0 : ℝ)) : WithBot ℝ) := by
    intro k
    by_cases h : active k
    · simp only [if_pos h]; rfl
    · simp only [if_neg h]
      show Option.map₂ (· * ·) (some (0.0 : ℝ)) (some 0.0) = ((0 : ℝ) : WithBot ℝ)
      have h0 : (0.0 : ℝ) * 0.0 = 0 := by norm_num
      rw [show Option.map₂ (· * ·) (some (0.0 : ℝ)) (some 0.0) = some ((0.0 : ℝ) * 0.0) from rfl,
          h0]
      rfl
  calc @Finset.sum (Fin BLOCK_N) (WithBot ℝ) _ Finset.univ _
      = ∑ k, (((if active k then xs k * ys k else 0 : ℝ)) : WithBot ℝ) :=
        Finset.sum_congr rfl (fun k _ => hcongr k)
    _ = ((∑ k, if active k then xs k * ys k else 0 : ℝ) : WithBot ℝ) :=
        (WithBot.coe_sum Finset.univ _).symm
    _ = some (∑ k, if active k then xs k * ys k else 0) := rfl

end CarrierBridges

end TiledL2Norm

end VeriTile.Triton
