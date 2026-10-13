import VeriTile.Triton
import VeriTile.Triton.Semantics.MaskedReduction
import bench.tritonbench_g.softmax_triton2.Proofs.Real

namespace VeriTile.Bench.TritonBenchG.SoftmaxTriton3
open VeriTile.Triton

/-- Masked input row tile used by `softmax_kernel`: lane `j < n_cols` holds
`xs j`, masked lanes are `⊥`, matching `other=-float("inf")`. -/
noncomputable def softmaxInputTile (n_cols BLOCK_SIZE : Nat)
    (xs : Fin BLOCK_SIZE → ℝ) : Tile .real [BLOCK_SIZE] :=
  { data := fun idx => if idx.1.val < n_cols then some (xs idx.1) else none }

/-- Optional additive-mask row tile: lane `j < n_cols` holds `ms j`, inactive
lanes are `0`, matching `tl.load(..., other=0)` (the final store is still
masked by `n_cols`). -/
noncomputable def softmaxMaskTile (n_cols BLOCK_SIZE : Nat)
    (ms : Fin BLOCK_SIZE → ℝ) : Tile .real [BLOCK_SIZE] :=
  { data := fun idx => if idx.1.val < n_cols then some (ms idx.1) else some 0 }

/-- Exact stable-softmax value computed by `softmax_kernel` at lane `idx`, as a
pure function of the active row prefix `xs j`, `j < n_cols`, and the optional
additive-mask row `mask?` (`some ms` ↔ the `HAS_MASK` branch; masked input
lanes enter the reductions as `⊥`, masked mask lanes as `0`). -/
noncomputable def softmaxSpec (n_cols BLOCK_SIZE : Nat)
    (mask? : Option (Fin BLOCK_SIZE → ℝ)) (xs : Fin BLOCK_SIZE → ℝ)
    (idx : Fin BLOCK_SIZE) : ℝ :=
  let row := softmaxInputTile n_cols BLOCK_SIZE xs
  match Tile.reduceMax (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false row with
  | some rowMax =>
      let shifted := Tile.bop (NumericDType.sub .real) Broadcast.scalarR row rowMax
      let shifted :=
        match mask? with
        | some ms =>
            Tile.bop (NumericDType.add .real) (Broadcast.consSame Broadcast.nil)
              shifted (softmaxMaskTile n_cols BLOCK_SIZE ms)
        | none => shifted
      let numerator := Tile.uop WithBot.realExp shifted
      let denominator := Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false numerator
      WithBot.unbotD 0
        ((Tile.bop (NumericDType.div .real) Broadcast.scalarR numerator denominator).data
          (idx, PUnit.unit))
  | none => 0

/-- `softmaxSpec` only reads the active lanes of its input row: two rows
agreeing below `n_cols` yield the same softmax value (masked lanes are `⊥` in
the tile either way). -/
theorem softmaxSpec_congr (n_cols BLOCK_SIZE : Nat)
    (mask? : Option (Fin BLOCK_SIZE → ℝ)) (xs ys : Fin BLOCK_SIZE → ℝ)
    (h : ∀ j : Fin BLOCK_SIZE, j.val < n_cols → xs j = ys j) (i : Fin BLOCK_SIZE) :
    softmaxSpec n_cols BLOCK_SIZE mask? xs i
      = softmaxSpec n_cols BLOCK_SIZE mask? ys i := by
  have htile : softmaxInputTile n_cols BLOCK_SIZE xs
      = softmaxInputTile n_cols BLOCK_SIZE ys := by
    unfold softmaxInputTile
    congr 1
    funext idx
    by_cases hj : idx.1.val < n_cols
    · simp only [if_pos hj, h idx.1 hj]
    · simp only [if_neg hj]
  unfold softmaxSpec
  rw [htile]

/-- `softmaxSpec` only reads the active lanes of the additive-mask row: two
mask rows agreeing below `n_cols` yield the same softmax value (inactive mask
lanes are `0` in the tile either way). -/
theorem softmaxSpec_mask_congr (n_cols BLOCK_SIZE : Nat)
    (xs ms₁ ms₂ : Fin BLOCK_SIZE → ℝ)
    (h : ∀ j : Fin BLOCK_SIZE, j.val < n_cols → ms₁ j = ms₂ j) (i : Fin BLOCK_SIZE) :
    softmaxSpec n_cols BLOCK_SIZE (some ms₁) xs i
      = softmaxSpec n_cols BLOCK_SIZE (some ms₂) xs i := by
  have htile : softmaxMaskTile n_cols BLOCK_SIZE ms₁
      = softmaxMaskTile n_cols BLOCK_SIZE ms₂ := by
    unfold softmaxMaskTile
    congr 1
    funext idx
    by_cases hj : idx.1.val < n_cols
    · simp only [if_pos hj, h idx.1 hj]
    · simp only [if_neg hj]
  simp only [softmaxSpec, htile]

/-- Unmasked values agree with mathematical softmax on the active prefix. -/
theorem softmaxSpec_none_eq_formula (n B : Nat) (hB : 0 < B)
    (xs : Fin B → ℝ) (i : Fin B) (hi : i.val < n) :
    softmaxSpec n B none xs i =
      Real.exp (xs i) / ∑ j : Fin B, if j.val < n then Real.exp (xs j) else 0 := by
  exact SoftmaxTriton2.softmaxSpec_eq_formula n B hB xs i hi

/-- The additive mask is part of the logits; subtracting the original row's max
is a common shift and therefore cancels from the real normalization. -/
theorem softmaxSpec_mask_eq_formula (n B : Nat) (hB : 0 < B)
    (xs ms : Fin B → ℝ) (i : Fin B) (hi : i.val < n) :
    softmaxSpec n B (some ms) xs i =
      Real.exp (xs i + ms i) /
        ∑ j : Fin B, if j.val < n then Real.exp (xs j + ms j) else 0 := by
  classical
  obtain ⟨k, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hB.ne'
  have hn : ((Finset.univ : Finset (Fin (k + 1))).filter (fun j => j.val < n)).Nonempty :=
    ⟨i, by simp [hi]⟩
  have hm := TiledLogSumExp.withBot_sup'_partial (α := ℝ) Finset.univ
    Finset.univ_nonempty (fun j : Fin (k + 1) => j.val < n) xs hn
  simp [softmaxSpec, softmaxInputTile, softmaxMaskTile, Tile.reduceMax, Tile.reduceMaxDrop,
    Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
    TileShape.insertAxisIndex, Tile.bop, Tile.uop, NumericDType.sub,
    NumericDType.div, NumericDType.add, hi]
  have hm' : @Finset.sup' (WithBot ℝ) (Fin (k + 1)) _ Finset.univ Finset.univ_nonempty
      (fun j : Fin (k + 1) => if j.val < n then (some (xs j) : WithBot ℝ) else none) =
      (some ((Finset.univ.filter (fun j : Fin (k + 1) => j.val < n)).sup' hn xs) : WithBot ℝ) := hm
  erw [hm']
  let m := (Finset.univ.filter (fun j : Fin (k + 1) => j.val < n)).sup' hn xs
  have hsum := TiledLogSumExp.sum_exp_masked_eq (n := k) (D := n) (i_d := 0)
    (fun j => xs j + ms j) m
  simp only [Nat.zero_mul, Nat.zero_add] at hsum
  have hshift (j : Fin (k + 1)) :
      Option.map₂ (fun x y : ℝ => x + y)
        (Option.map₂ (fun x y : ℝ => x - y)
          (if j.val < n then some (xs j) else none) (some m))
        (if j.val < n then some (ms j) else some 0) =
      Option.map₂ (fun x y : ℝ => x - y)
        (if j.val < n then some (xs j + ms j) else none) (some m) := by
    by_cases hj : j.val < n
    · simp only [if_pos hj, Option.map₂_some_some]
      congr 1
      ring
    · simp [hj]
  change WithBot.unbotD 0
    (Option.map₂ (fun x y : ℝ => x / y)
      (WithBot.realExp (some (xs i - m + ms i)))
      (@Finset.sum (Fin (k + 1)) (WithBot ℝ) WithBot.addCommMonoid Finset.univ
        (fun j => WithBot.realExp (Option.map₂ (fun x y : ℝ => x + y)
          (Option.map₂ (fun x y : ℝ => x - y)
            (if j.val < n then some (xs j) else none) (some m))
          (if j.val < n then some (ms j) else some 0))))) = _
  simp_rw [hshift]
  erw [hsum]
  simp only [WithBot.realExp_some]
  rw [show xs i - m + ms i = xs i + ms i - m by ring]
  change Real.exp (xs i + ms i - m) /
      (∑ j ∈ TiledLogSumExp.validLanes k n 0, Real.exp (xs j + ms j - m)) = _
  simp only [TiledLogSumExp.validLanes, Nat.zero_mul, Nat.zero_add,
    Finset.sum_filter, Real.exp_sub]
  have hs : ∀ j : Fin (k + 1),
      (if j.val < n then Real.exp (xs j + ms j) / Real.exp m else 0) =
      (if j.val < n then Real.exp (xs j + ms j) else 0) / Real.exp m := by
    intro j
    split_ifs <;> simp
  simp_rw [hs]
  rw [← Finset.sum_div]
  exact div_div_div_cancel_right₀ (Real.exp_ne_zero m) _ _


end VeriTile.Bench.TritonBenchG.SoftmaxTriton3
