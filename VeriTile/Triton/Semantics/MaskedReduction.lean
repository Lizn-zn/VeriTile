/-
VeriTile.Triton.Semantics.MaskedReduction

Reusable masked-lane reduction bridge lemmas. These connect Triton's
`WithBot`/masked tile carriers to finite real sums/suprema used by pure math
operators.
-/

import Mathlib.Data.Finset.Lattice.Fold
import Mathlib.Algebra.BigOperators.WithTop
import VeriTile.Triton.Semantics.TileOps
import VeriTile.Triton.Semantics.TiledIndexing
import VeriTile.Triton.Math.L2Norm

namespace VeriTile.Triton

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

namespace TileShape

/-! ## `Fin`-literal axes of `insertAxisIndex`

`Tile.reduceSumDrop` presents the reduced lane of a `tl.sum(_, axis = j)` as
`TileShape.insertAxisIndex shape axis outIdx k`. The structural lemmas in
`Core/Shape.lean` (`insertAxisIndex_head` / `insertAxisIndex_succ`) match the
`Fin.mk` patterns `⟨0, _⟩` / `⟨k + 1, _⟩`, so they do **not** fire on the axis
terms that actually reach kernel proofs. Two different spellings occur, often
in one and the same goal:

* **Kernel (exec) side.** The DSL emits `⟨j, by simp⟩`; Mathlib's `Fin.mk_zero`
  / `Fin.mk_one` then rewrite that to an `OfNat` literal whose `Fin` type index
  is left in `?n + k` form, because it sits in a type argument of
  `OfNat.ofNat` that `simp` does not descend into. For a rank-2 tile this is
  `(0 : Fin (1 + 1))` at axis 0 and `(1 : Fin (0 + 2))` at axis 1.
* **Specification side.** A port's own spec writes
  `TileShape.insertAxisIndex [d₀, d₁] 1 …` in source, which elaborates at
  `Fin (d₀ :: d₁ :: rest).length`.

These `rfl` lemmas normalize every one of those spellings to plain lane
variables; they mirror the `Fin`-literal `dropInsertedIndex` lemmas in
`Core/Shape.lean`. They are deliberately **not** `@[simp]`: nothing can
currently simplify a literal-axis `insertAxisIndex`, so turning them on
globally would change the simp normal form of every reduction goal in the
library and in all bench ports. Consumers name them in their own simp set. -/

/-- Axis 0, source spelling `(0 : Fin (d :: rest).length)`. -/
theorem insertAxisIndex_zero_length {d : Nat} {rest : TileShape}
    (outIdx : TileIndex rest) (r : Fin d) :
    insertAxisIndex (d :: rest) (0 : Fin (d :: rest).length) outIdx r
      = (r, outIdx) := rfl

/-- Axis 1, source spelling `(1 : Fin (d₀ :: d₁ :: rest).length)`. -/
theorem insertAxisIndex_one_length {d₀ d₁ : Nat} {rest : TileShape}
    (outIdx : TileIndex (d₀ :: rest)) (r : Fin d₁) :
    insertAxisIndex (d₀ :: d₁ :: rest)
        (1 : Fin (d₀ :: d₁ :: rest).length) outIdx r
      = (outIdx.1, r, outIdx.2) := rfl

/-- Axis 0 of a rank-2 tile, post-`Fin.mk_zero` spelling `(0 : Fin (1 + 1))`.
This is the form `tl.sum(_, axis = 0)` on a `[d₀, d₁]` tile leaves behind. -/
theorem insertAxisIndex_zero_rank2 {d₀ d₁ : Nat}
    (outIdx : TileIndex [d₁]) (r : Fin d₀) :
    insertAxisIndex [d₀, d₁] (0 : Fin (1 + 1)) outIdx r = (r, outIdx) := rfl

/-- Axis 1 of a rank-2 tile, post-`Fin.mk_one` spelling `(1 : Fin (0 + 2))`.
This is the form `tl.sum(_, axis = 1)` on a `[d₀, d₁]` tile leaves behind. -/
theorem insertAxisIndex_one_rank2 {d₀ d₁ : Nat}
    (outIdx : TileIndex [d₀]) (r : Fin d₁) :
    insertAxisIndex [d₀, d₁] (1 : Fin (0 + 2)) outIdx r
      = (outIdx.1, r, outIdx.2) := rfl

end TileShape

namespace MaskedReduction

/-! ## Guarded multi-factor reductions

The bridges in `TiledLogSumExp` / `TiledL2Norm` above are shaped for the tiles
they were extracted from: one or two factors over a 1-D lane index, with the
mask spelled as a concrete `validLanes` / `active` predicate. A masked
*multi-factor 2-D* reduction — `tl.sum(prev * b_k[None, :] * …, axis = 1)`
where every operand is a `tl.load(…, mask = …, other = 0)` — does not fit
them: each factor contributes its own guard, the guards are on different axes,
and the reduced lane index is `TileShape.insertAxisIndex`-projected.

Two independent facts close that gap, and they compose with the existing
`@[simp]` `WithBot ℝ` helpers in `Semantics/TileOps.lean`
(`Option.map₂_some_some`, `WithBot.sum_someTerm_eq_some`,
`WithBot.unbotD_some`):

* `ite_some_some` pushes the `some` of a `other = 0` guarded load *outward*,
  which turns an arbitrarily deep tower of `Option.map₂` factors into
  `some` of a plain real — no matter how many factors, or what rank the tile
  has. This is what makes the multi-factor 2-D case behave like the
  two-factor 1-D one.
* `sum_guarded_eq_coe` and its `unbotD` forms transfer a `WithBot ℝ`
  reduction to a real reduction from a purely **lane-local** hypothesis, so
  the shape of the lane carrier is irrelevant to the bridge and the guard
  bookkeeping is discharged one lane at a time. -/

/-- A `tl.load(…, mask = m, other = z)` lane carrier is `some` of a plain
real: `if p then some a else some z = some (if p then a else z)`.

Pushing the `some` outward is what lets the existing `@[simp]` lemmas
`Option.map₂_some_some` / `WithBot.sum_someTerm_eq_some` /
`WithBot.unbotD_some` collapse a whole guarded factor tower, of any depth and
any tile rank, down to a real-valued expression.

Two spelling traps, both load-bearing:

* the `@ite (WithBot ℝ)` must be written **explicitly**. Writing
  `(if p then (some a : WithBot ℝ) else some z)` elaborates the `ite` at
  `Option ℝ` instead, and `simp` will then never match the `WithBot ℝ`-typed
  `ite` that kernel goals actually contain (`TileCarrier .real` reduces to
  `WithBot ℝ`, not to `Option ℝ`);
* this is not `@[simp]`, because masked-load carriers currently normalize to
  the `if … then some … else some …` form and every existing proof over a
  masked load is written against it. Consumers name it in their own simp set. -/
theorem ite_some_some {p : Prop} [inst : Decidable p] (a z : ℝ) :
    @ite (WithBot ℝ) p inst (some a) (some z)
      = (some (if p then a else z) : WithBot ℝ) := by
  split_ifs <;> rfl

/-- **Guarded reduction bridge.** A `WithBot ℝ` reduction whose lanes are
each (pointwise) a `some` of a real equals the coercion of the real reduction.

Unlike the bridges above this says nothing about the *shape* of the lane
carrier: `hlane` is a lane-local obligation, so it serves guarded
multi-factor 2-D tiles — including ones whose lane index is
`TileShape.insertAxisIndex`-projected and whose factors carry masks on
different axes — exactly as well as the two-factor 1-D case. -/
theorem sum_guarded_eq_coe {ι : Type*} [Fintype ι]
    (F : ι → WithBot ℝ) (g : ι → ℝ)
    (hlane : ∀ i, F i = (some (g i) : WithBot ℝ)) :
    @Finset.sum ι (WithBot ℝ) _ Finset.univ F = ((∑ i, g i : ℝ) : WithBot ℝ) := by
  rw [Finset.sum_congr rfl (fun i _ => hlane i)]
  exact WithBot.sum_someTerm_eq_some Finset.univ g

/-- `WithBot.unbotD`-facing form of `sum_guarded_eq_coe` — the shape a
`tl.sum` whose result is stored directly lands in. Stated so that
`refine unbotD_sum_guarded_eq _ _ ?_` reads `g` off the goal's own
right-hand side (which is the specification's `∑`), leaving only the
lane-local obligation. -/
theorem unbotD_sum_guarded_eq {ι : Type*} [Fintype ι]
    (F : ι → WithBot ℝ) (g : ι → ℝ)
    (hlane : ∀ i, F i = (some (g i) : WithBot ℝ)) :
    WithBot.unbotD 0 (@Finset.sum ι (WithBot ℝ) _ Finset.univ F) = ∑ i, g i := by
  rw [sum_guarded_eq_coe F g hlane]; rfl

/-- `unbotD` form for a reduction that is combined with a lane-wise scalar
before being stored — the `b_v -= tl.sum(…)` / `b_v *= …` shape. -/
theorem unbotD_map₂_sum_guarded_eq {ι : Type*} [Fintype ι]
    (op : ℝ → ℝ → ℝ) (c : ℝ) (F : ι → WithBot ℝ) (g : ι → ℝ)
    (hlane : ∀ i, F i = (some (g i) : WithBot ℝ)) :
    WithBot.unbotD 0 (Option.map₂ op (some c : WithBot ℝ)
        (@Finset.sum ι (WithBot ℝ) _ Finset.univ F))
      = op c (∑ i, g i) := by
  rw [sum_guarded_eq_coe F g hlane]; rfl

end MaskedReduction

end VeriTile.Triton
