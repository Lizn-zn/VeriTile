/-
VeriTile.Semantics.MaskedReduction

Reusable masked-lane reduction bridge lemmas. These connect Triton's
`WithBot`/masked tile carriers to finite real sums/suprema used by pure math
operators.
-/

import Mathlib.Data.Finset.Lattice.Fold
import Mathlib.Algebra.BigOperators.WithTop
import VeriTile.Semantics.TileOps
import VeriTile.Semantics.TiledIndexing
import VeriTile.Math.L2Norm

namespace VeriTile

namespace TiledLogSumExp

/-! ## Supremum and exponential-sum bridges -/

/-- `Finset.sup'` over a tile that mixes `some r`/`⊥` values equals `↑(sup' over
the valid lanes)`. This is the basic bridge for masked reductions with
`other = -inf`. -/
theorem withBot_sup'_partial {α : Type*} [LinearOrder α] {ι : Type*}
    (s : Finset ι) (h_ne : s.Nonempty)
    (p : ι → Prop) [DecidablePred p]
    (f : ι → α)
    (h_filter : (s.filter p).Nonempty) :
    s.sup' h_ne (fun i => if p i then ((f i : α) : WithBot α) else ⊥) =
    (((s.filter p).sup' h_filter f : α) : WithBot α) := by
  apply le_antisymm
  · apply Finset.sup'_le; intro i hi
    by_cases h : p i
    · rw [if_pos h]; apply WithBot.coe_le_coe.mpr
      apply Finset.le_sup'; rw [Finset.mem_filter]; exact ⟨hi, h⟩
    · rw [if_neg h]; exact bot_le
  · have h_mem : ∀ j ∈ s.filter p, (((f j : α)) : WithBot α) ≤
        s.sup' h_ne (fun i => if p i then ((f i : α) : WithBot α) else ⊥) := by
      intro j hj; rw [Finset.mem_filter] at hj
      have key := Finset.le_sup' (s := s)
        (f := fun i => if p i then ((f i : α) : WithBot α) else ⊥) hj.1
      rw [if_pos hj.2] at key; exact key
    have h_le := Finset.sup'_le h_filter (fun j => ((f j : α) : WithBot α)) h_mem
    rw [show ((((s.filter p).sup' h_filter f) : α) : WithBot α)
          = (s.filter p).sup' h_filter (fun j => ((f j : α) : WithBot α))
        from by
          rw [Finset.comp_sup'_eq_sup'_comp h_filter (fun (r : α) => ((r : WithBot α)))]
          · rfl
          · intros; rfl]
    exact h_le

/-- A single summand:
`exp(if valid then some(f i) - some m else none - some m)` simplifies to
`if valid then some(exp(f i - m)) else some 0`. -/
theorem summand_eq {n D i_d : Nat} (i : Fin (n+1)) (f : Fin (n+1) → ℝ) (m : ℝ) :
    WithBot.realExp (Option.map₂ (fun x y => x - y)
      (if i_d * (n+1) + i.val < D then (some (f i) : WithBot ℝ) else none)
      ((m : ℝ) : WithBot ℝ)) =
    if i_d * (n+1) + i.val < D
      then ((Real.exp (f i - m) : ℝ) : WithBot ℝ)
      else ((0 : ℝ) : WithBot ℝ) := by
  by_cases h : i_d * (n+1) + i.val < D
  · simp only [h, ↓reduceIte, Option.map₂_some_coe, WithBot.realExp_some]; rfl
  · simp only [h, ↓reduceIte]; rfl

/-- Sum of `exp`-of-masked-subtraction over a tile equals
`↑(∑ over valid lanes)`. -/
theorem sum_exp_masked_eq {n D i_d : Nat} (f : Fin (n+1) → ℝ) (m : ℝ) :
    (∑ i : Fin (n+1), WithBot.realExp (Option.map₂ (fun x y => x - y)
      (if i_d * (n+1) + i.val < D then (some (f i) : WithBot ℝ) else none)
      ((m : ℝ) : WithBot ℝ))) =
    ((∑ i ∈ validLanes n D i_d, Real.exp (f i - m) : ℝ) : WithBot ℝ) := by
  simp_rw [summand_eq]
  have key : ∀ i : Fin (n+1),
      (if i_d * (n+1) + i.val < D then ((Real.exp (f i - m) : ℝ) : WithBot ℝ)
        else ((0 : ℝ) : WithBot ℝ)) =
      ((if i_d * (n+1) + i.val < D then Real.exp (f i - m) else 0 : ℝ) : WithBot ℝ) :=
    fun i => by split_ifs <;> rfl
  simp_rw [key, ← WithBot.coe_sum]
  congr 1
  unfold validLanes; rw [← Finset.sum_filter]

/-- `sup'` over a mixed `some`/`none` tile equals `↑(sup' over valid lanes)`. -/
theorem sup'_masked_eq {n D i_d : Nat}
    (h_ne : (Finset.univ : Finset (Fin (n+1))).Nonempty)
    (h_filter : (validLanes n D i_d).Nonempty) (f : Fin (n+1) → ℝ) :
    @Finset.sup' (WithBot ℝ) (Fin (n+1)) _ Finset.univ h_ne
      (fun i => if i_d * (n+1) + i.val < D then (some (f i) : WithBot ℝ) else none) =
    (((validLanes n D i_d).sup' h_filter f : ℝ) : WithBot ℝ) :=
  withBot_sup'_partial Finset.univ h_ne _ f h_filter

/-- `sup'` over a tile of `Option.map g (if valid then some f else none)` equals
`↑(sup' over valid lanes of g ∘ f)`. -/
theorem sup'_masked_map_eq {n D i_d : Nat}
    (h_ne : (Finset.univ : Finset (Fin (n+1))).Nonempty)
    (h_filter : (validLanes n D i_d).Nonempty)
    (f : Fin (n+1) → ℝ) (g : ℝ → ℝ) :
    @Finset.sup' (WithBot ℝ) (Fin (n+1)) _ Finset.univ h_ne
      (fun i => Option.map g (if i_d * (n+1) + i.val < D then (some (f i) : WithBot ℝ) else none)) =
    (((validLanes n D i_d).sup' h_filter (fun i => g (f i)) : ℝ) : WithBot ℝ) := by
  have heq : ∀ i : Fin (n+1),
      Option.map g (if i_d * (n+1) + i.val < D then (some (f i) : WithBot ℝ) else none) =
      if i_d * (n+1) + i.val < D then (some (g (f i)) : WithBot ℝ) else none := by
    intro i; by_cases h : i_d * (n+1) + i.val < D <;> simp [h]
  have : @Finset.sup' (WithBot ℝ) (Fin (n+1)) _ Finset.univ h_ne
      (fun i => Option.map g (if i_d * (n+1) + i.val < D then (some (f i) : WithBot ℝ) else none)) =
      @Finset.sup' (WithBot ℝ) (Fin (n+1)) _ Finset.univ h_ne
      (fun i => if i_d * (n+1) + i.val < D then (some (g (f i)) : WithBot ℝ) else none) :=
    Finset.sup'_congr h_ne rfl fun i _ => heq i
  rw [this]
  erw [sup'_masked_eq h_ne h_filter (fun i => g (f i))]

/-- Sum of `exp`-of-masked-subtraction with `Option.map g` scaling equals
`↑(∑ over valid lanes)`. -/
theorem sum_exp_masked_map_eq {n D i_d : Nat} (f : Fin (n+1) → ℝ) (g : ℝ → ℝ) (m : ℝ) :
    (∑ i : Fin (n+1), WithBot.realExp (Option.map₂ (fun x y => x - y)
      (Option.map g (if i_d * (n+1) + i.val < D then (some (f i) : WithBot ℝ) else none))
      ((m : ℝ) : WithBot ℝ))) =
    ((∑ i ∈ validLanes n D i_d, Real.exp (g (f i) - m) : ℝ) : WithBot ℝ) := by
  have heq : ∀ i : Fin (n+1),
      Option.map g (if i_d * (n+1) + i.val < D then (some (f i) : WithBot ℝ) else none) =
      if i_d * (n+1) + i.val < D then (some (g (f i)) : WithBot ℝ) else none :=
    fun i => by by_cases h : i_d * (n+1) + i.val < D <;> simp [h]
  simp_rw [heq]
  exact sum_exp_masked_eq (fun i => g (f i)) m

end TiledLogSumExp

namespace TiledL2Norm

/-! ## Sum/product carrier bridges -/

/-- Bridge: kernel-shape sum of squares with masked tail = pure `∑` form. -/
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
      = some (∑ k, if active k then if active k then load k * load k else 0 else 0) := by
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
        = (((if active k then if active k then load k * load k else 0 else 0 : ℝ)) :
            WithBot ℝ) := by
    intro k
    by_cases h : active k
    · simp only [if_pos h]; rfl
    · simp only [if_neg h]
      show Option.map₂ (· * ·) (some (0.0 : ℝ)) (some 0.0) = ((0 : ℝ) : WithBot ℝ)
      have h0 : (0.0 : ℝ) * 0.0 = 0 := by norm_num
      rw [show Option.map₂ (· * ·) (some (0.0 : ℝ)) (some 0.0) =
          some ((0.0 : ℝ) * 0.0) from rfl, h0]
      rfl
  calc @Finset.sum (Fin BLOCK_N) (WithBot ℝ) _ Finset.univ _
      = ∑ k, (((if active k then if active k then load k * load k else 0 else 0 : ℝ)) :
            WithBot ℝ) :=
        Finset.sum_congr rfl (fun k _ => hcongr k)
    _ = ((∑ k, if active k then if active k then load k * load k else 0 else 0 : ℝ) :
          WithBot ℝ) :=
        (WithBot.coe_sum Finset.univ _).symm
    _ = some (∑ k, if active k then if active k then load k * load k else 0 else 0) := rfl

/-- Bridge: kernel-shape cross-product with masked tail = pure `∑` form. -/
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
      = some (∑ k, if active k then if active k then xs k * ys k else 0 else 0) := by
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
        = (((if active k then if active k then xs k * ys k else 0 else 0 : ℝ)) :
            WithBot ℝ) := by
    intro k
    by_cases h : active k
    · simp only [if_pos h]; rfl
    · simp only [if_neg h]
      show Option.map₂ (· * ·) (some (0.0 : ℝ)) (some 0.0) = ((0 : ℝ) : WithBot ℝ)
      have h0 : (0.0 : ℝ) * 0.0 = 0 := by norm_num
      rw [show Option.map₂ (· * ·) (some (0.0 : ℝ)) (some 0.0) =
          some ((0.0 : ℝ) * 0.0) from rfl, h0]
      rfl
  calc @Finset.sum (Fin BLOCK_N) (WithBot ℝ) _ Finset.univ _
      = ∑ k, (((if active k then if active k then xs k * ys k else 0 else 0 : ℝ)) :
            WithBot ℝ) :=
        Finset.sum_congr rfl (fun k _ => hcongr k)
    _ = ((∑ k, if active k then if active k then xs k * ys k else 0 else 0 : ℝ) :
          WithBot ℝ) :=
        (WithBot.coe_sum Finset.univ _).symm
    _ = some (∑ k, if active k then if active k then xs k * ys k else 0 else 0) := rfl

end TiledL2Norm

end VeriTile
