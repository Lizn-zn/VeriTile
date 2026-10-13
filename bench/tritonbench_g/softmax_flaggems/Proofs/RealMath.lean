import VeriTile.Triton.Math.Softmax
import VeriTile.Triton.Semantics.TileOps
import Mathlib.Algebra.BigOperators.Group.Finset.Piecewise
import Mathlib.Algebra.BigOperators.Group.Finset.Sigma
import Mathlib.Tactic.Ring

/-! Independent mathematical targets for the complete FlagGems source family.
These definitions contain no interpreter or tile reduction. The execution bridge
is a separate obligation. Backward padding is explicit because source masked
loads omit `other`; their inactive values cannot silently be set to zero. -/
namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealMath
open VeriTile.Triton
open scoped BigOperators

noncomputable def forward {N : Nat} (x : Fin N → ℝ) (i : Fin N) : ℝ :=
  Real.exp (x i) / ∑ j, Real.exp (x j)

noncomputable def backward (N : Nat) (y dy : Nat → ℝ) (i : Nat) : ℝ :=
  y i * (dy i - ∑ j ∈ Finset.range N, y j * dy j)

theorem forward_positive {N : Nat} (x : Fin N → ℝ) (i : Fin N) :
    0 < forward x i := by
  apply div_pos (Real.exp_pos _)
  exact Finset.sum_pos (fun _ _ ↦ Real.exp_pos _) ⟨i, Finset.mem_univ i⟩

theorem forward_sum_one {N : Nat} (hN : 0 < N) (x : Fin N → ℝ) :
    (∑ i, forward x i) = 1 := by
  have hz : 0 < ∑ i, Real.exp (x i) :=
    Finset.sum_pos (fun _ _ ↦ Real.exp_pos _) ⟨⟨0, hN⟩, Finset.mem_univ _⟩
  simp only [forward, ← Finset.sum_div]
  exact div_self (ne_of_gt hz)

/-- The padded reduction actually used by the backward source, prior to masking stores. -/
noncomputable def backwardPadded (P : Nat) (y dy : Nat → ℝ) (i : Nat) : ℝ :=
  y i * (dy i - ∑ j ∈ Finset.range P, y j * dy j)

/-- Stable normalization has the independent full-row softmax value for every shift. -/
theorem forward_shift {N : Nat} (x : Fin N → ℝ) (m : ℝ) (i : Fin N) :
    Real.exp (x i - m) / (∑ j, Real.exp (x j - m)) = forward x i :=
  congrFun (VeriTile.Triton.TiledSoftmax.naive_eq_stable x m).symm i

/-- Exact contribution of arbitrary inactive padding to the backward result. -/
theorem backwardPadded_eq (N tail : Nat) (y dy : Nat → ℝ) (i : Nat) :
    backwardPadded (N + tail) y dy i = backward N y dy i -
      y i * (∑ j ∈ Finset.range tail, y (N + j) * dy (N + j)) := by
  unfold backwardPadded backward
  rw [Finset.sum_range_add]
  ring

/-- Zero products, rather than an implicit padding convention, recover the VJP. -/
theorem backwardPadded_eq_of_zero (N tail : Nat) (y dy : Nat → ℝ) (i : Nat)
    (hpad : ∀ j < tail, y (N + j) * dy (N + j) = 0) :
    backwardPadded (N + tail) y dy i = backward N y dy i := by
  rw [backwardPadded_eq]
  have hz : (∑ j ∈ Finset.range tail, y (N + j) * dy (N + j)) = 0 :=
    Finset.sum_eq_zero (fun j hj ↦ hpad j (Finset.mem_range.mp hj))
  rw [hz, mul_zero, sub_zero]

/-- Every active row admits a padding extension satisfying the explicit VJP domain. -/
theorem zero_padding_extension (N : Nat) (y dy : Fin N → ℝ) :
    ∃ y' dy' : Nat → ℝ,
      (∀ i : Fin N, y' i = y i ∧ dy' i = dy i) ∧
      (∀ j, N ≤ j → y' j * dy' j = 0) := by
  refine ⟨(fun j ↦ if h : j < N then y ⟨j, h⟩ else 0),
    (fun j ↦ if h : j < N then dy ⟨j, h⟩ else 0), ?_, ?_⟩
  · intro i
    simp [i.isLt]
  · intro j hj
    simp [Nat.not_lt.mpr hj]

/- The scalar recurrence and its invariant are reused from the existing
SoftmaxOptimize proof. FlagGems uses the same mathematical recurrence; source
execution and lane-partition correspondence remain separate proof obligations. -/
noncomputable def onlineStep (st : WithBot ℝ × ℝ) (x : ℝ) : WithBot ℝ × ℝ :=
  let m := st.1; let z := st.2
  let m' := m ⊔ ((x : ℝ) : WithBot ℝ)
  let α := (WithBot.realExp (WithBot.realSub m m')).unbotD 0
  (m', z * α + Real.exp (x - m'.unbotD 0))

/-- The running max after folding `onlineStep` over a list of scores: it is the
`WithBot`-supremum of the seed and the scores. -/
theorem online_maximum (xs : List ℝ) (m : WithBot ℝ) (z : ℝ) :
    (xs.foldl onlineStep (m, z)).1 = xs.foldl (fun a x => a ⊔ ((x : ℝ) : WithBot ℝ)) m := by
  induction xs generalizing m z with
  | nil => rfl
  | cons x xs ih => simpa [onlineStep] using ih (m ⊔ ((x : ℝ) : WithBot ℝ)) _

/-- **⊥-seeded denominator consistency.** Folding `onlineStep` from a state with
`z = κ(m)·L` (where `κ ⊥ = 0`, `κ (some r) = exp(-r)`, and `L` is the running
true denominator `Σ exp(x)`) keeps that invariant: the final
`z = κ(m_final)·(L + Σ_{x∈xs} exp x)`, with `m_final = ⊔ m xs`. -/
theorem online_denominator_invariant
    (xs : List ℝ) (m : WithBot ℝ) (z L : ℝ)
    (hz : z = (m.elim 0 (fun r => Real.exp (-r))) * L)
    (hmL : m = ⊥ → L = 0) :
    let st := xs.foldl onlineStep (m, z)
    let L' := L + (xs.map Real.exp).sum
    st.2 = (st.1.elim 0 (fun r => Real.exp (-r))) * L'
      ∧ (st.1 = ⊥ → L' = 0) := by
  induction xs generalizing m z L with
  | nil =>
    refine ⟨by simpa using hz, ?_⟩
    simpa using hmL
  | cons x xs ih =>
    simp only [List.foldl_cons, List.map_cons, List.sum_cons]
    -- one onlineStep then recurse
    set m' : WithBot ℝ := m ⊔ ((x : ℝ) : WithBot ℝ) with hm'
    -- m' is `some` (it is ≥ some x), so the new state is never ⊥ and L' > 0-side is fine
    have hm'def : m' = ((m.elim x (fun r => max r x) : ℝ) : WithBot ℝ) := by
      cases m with
      | bot => rw [hm']; simp [Option.elim]
      | coe r => rw [hm']; rfl
    have hz' : z * (WithBot.realExp (WithBot.realSub m m')).unbotD 0
          + Real.exp (x - m'.unbotD 0)
        = (m'.elim 0 (fun r => Real.exp (-r))) * (L + Real.exp x) := by
      cases m with
      | bot =>
        -- m' = some x, z = 0, L = 0
        have hL0 : L = 0 := hmL rfl
        rw [hz, hL0]
        rw [show m' = ((x : ℝ) : WithBot ℝ) from by rw [hm']; simp]
        simp only [WithBot.realSub, WithBot.unbotD, Option.elim, mul_zero, zero_mul,
          zero_add, WithBot.recBotCoe_coe, id_eq]
        rw [show x - x = (0:ℝ) from by ring, Real.exp_zero, Real.exp_neg, mul_comm,
          mul_inv_cancel₀ (Real.exp_ne_zero x)]
      | coe r =>
        have hmm : m' = ((max r x : ℝ) : WithBot ℝ) := by rw [hm']; rfl
        rw [hmm, hz]
        simp only [WithBot.realSub_coe_coe, WithBot.unbotD_coe]
        rw [show (((max r x : ℝ)) : WithBot ℝ).elim 0 (fun r => Real.exp (-r))
              = Real.exp (-(max r x)) from rfl]
        rw [show (((r : ℝ)) : WithBot ℝ).elim 0 (fun r => Real.exp (-r))
              = Real.exp (-r) from rfl]
        rw [show WithBot.realExp ((r - max r x : ℝ) : WithBot ℝ)
              = ((Real.exp (r - max r x) : ℝ) : WithBot ℝ) from rfl,
            WithBot.unbotD_coe]
        rw [mul_add]
        congr 1
        · rw [mul_comm (Real.exp (-r)), mul_assoc, ← Real.exp_add,
            show -r + (r - max r x) = -(max r x) from by ring, mul_comm]
        · rw [← Real.exp_add]; ring_nf
    have hm'L : m' = ⊥ → (L + Real.exp x) = 0 := by
      intro hbot; rw [hm'def] at hbot; exact absurd hbot (by simp)
    have hrec := ih m' (z * (WithBot.realExp (WithBot.realSub m m')).unbotD 0
          + Real.exp (x - m'.unbotD 0)) (L + Real.exp x) hz' hm'L
    simp only [onlineStep] at hrec ⊢
    have hassoc : L + (Real.exp x + (xs.map Real.exp).sum)
        = (L + Real.exp x) + (xs.map Real.exp).sum := by ring
    rw [hassoc]
    exact hrec

/-- Rescaling one online lane to a common finite maximum preserves its mass. -/
theorem online_lane_rescale (m : WithBot ℝ) (z L M : ℝ)
    (hz : z = m.elim 0 (fun r ↦ Real.exp (-r)) * L)
    (hmL : m = ⊥ → L = 0) :
    z * (WithBot.realExp (WithBot.realSub m (M : WithBot ℝ))).unbotD 0 =
      Real.exp (-M) * L := by
  cases m with
  | bot =>
    rw [hz, hmL rfl]
    simp
  | coe r =>
    change z * Real.exp (r - M) = Real.exp (-M) * L
    have hz' : z = Real.exp (-r) * L := hz
    rw [hz']
    calc
      (Real.exp (-r) * L) * Real.exp (r - M) =
          L * (Real.exp (-r) * Real.exp (r - M)) := by ring
      _ = L * Real.exp (-M) := by
        rw [← Real.exp_add]
        congr 2
        ring
      _ = Real.exp (-M) * L := mul_comm _ _

/-- The source's final lane reduction recovers the full accumulated denominator. -/
theorem online_lane_collapse {B : Nat} (m : Fin B → WithBot ℝ)
    (z L : Fin B → ℝ) (M : ℝ)
    (hz : ∀ i, z i = (m i).elim 0 (fun r ↦ Real.exp (-r)) * L i)
    (hmL : ∀ i, m i = ⊥ → L i = 0) :
    (∑ i, z i * (WithBot.realExp (WithBot.realSub (m i) (M : WithBot ℝ))).unbotD 0) =
      Real.exp (-M) * ∑ i, L i := by
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i _
  exact online_lane_rescale (m i) (z i) (L i) M (hz i) (hmL i)

/-- Full masked-lane update, including the source's negative-infinity padding. -/
noncomputable def maskedOnlineStep (st : WithBot ℝ × ℝ) (x : WithBot ℝ) :
    WithBot ℝ × ℝ :=
  let next := st.1 ⊔ x
  (next, st.2 * (WithBot.realExp (WithBot.realSub st.1 next)).unbotD 0 +
    (WithBot.realExp (WithBot.realSub x next)).unbotD 0)

theorem maskedOnlineStep_none (m : WithBot ℝ) (z : ℝ) (hz : m = ⊥ → z = 0) :
    maskedOnlineStep (m, z) ⊥ = (m, z) := by
  cases m with
  | bot =>
    rw [hz rfl]
    simp only [maskedOnlineStep, sup_bot_eq]
    change (⊥, (0 : ℝ) * 0 + 0) = (⊥, 0)
    simp
  | coe m =>
    simp only [maskedOnlineStep, sup_bot_eq]
    change ((m : WithBot ℝ), z * Real.exp (m - m) + 0) = ((m : WithBot ℝ), z)
    simp

theorem maskedOnlineStep_some (m : WithBot ℝ) (z x : ℝ) :
    maskedOnlineStep (m, z) (x : WithBot ℝ) = onlineStep (m, z) x := by
  cases m with
  | bot =>
    simp only [maskedOnlineStep, onlineStep, bot_sup_eq]
    rfl
  | coe m =>
    simp only [maskedOnlineStep, onlineStep, ← WithBot.coe_sup]
    rfl

/-- Dropping masked inputs is proved from the source recurrence, not assumed. -/
theorem maskedOnline_fold (xs : List (Option ℝ)) (m : WithBot ℝ) (z : ℝ)
    (hz : m = ⊥ → z = 0) :
    xs.foldl maskedOnlineStep (m, z) = (xs.filterMap id).foldl onlineStep (m, z) := by
  induction xs generalizing m z with
  | nil => rfl
  | cons x xs ih =>
    cases x with
    | none =>
      change xs.foldl maskedOnlineStep (maskedOnlineStep (m, z) ⊥) =
        (xs.filterMap id).foldl onlineStep (m, z)
      rw [maskedOnlineStep_none m z hz]
      exact ih m z hz
    | some x =>
      change xs.foldl maskedOnlineStep (maskedOnlineStep (m, z) (x : WithBot ℝ)) =
        (xs.filterMap id).foldl onlineStep (onlineStep (m, z) x)
      rw [maskedOnlineStep_some]
      apply ih
      intro hbot
      have hn : (m ⊔ (x : WithBot ℝ)) ≠ ⊥ := by cases m <;> simp
      exact (hn hbot).elim

/-- **Lane-major partition.** For full tile coverage (`N ≤ c·TILE_N`), summing a
column-indexed value `V` over `(lane, tile)` pairs in range recovers the sum over
all columns `j < N`. -/
theorem lane_tile_partition (N TILE_N : Nat) (hT : 0 < TILE_N) (c : Nat)
    (hc : N ≤ c * TILE_N) (V : Nat → ℝ) :
    ∑ ℓ : Fin TILE_N, ∑ t ∈ Finset.range c,
        (if t * TILE_N + ℓ.val < N then V (t * TILE_N + ℓ.val) else 0)
      = ∑ j ∈ Finset.range N, V j := by
  have step1 : ∀ ℓ : Fin TILE_N, ∑ t ∈ Finset.range c,
        (if t * TILE_N + ℓ.val < N then V (t * TILE_N + ℓ.val) else 0)
      = ∑ t ∈ (Finset.range c).filter (fun t => t * TILE_N + ℓ.val < N),
          V (t * TILE_N + ℓ.val) := by
    intro ℓ; rw [Finset.sum_filter]
  simp_rw [step1]
  rw [Finset.sum_sigma']
  apply Finset.sum_nbij'
    (i := fun p : (_ : Fin TILE_N) × Nat => p.2 * TILE_N + (p.1 : Nat))
    (j := fun j => (⟨⟨j % TILE_N, Nat.mod_lt _ hT⟩, j / TILE_N⟩ : (_ : Fin TILE_N) × Nat))
  · rintro ⟨ℓ, t⟩ hmem
    simp only [Finset.mem_sigma, Finset.mem_filter, Finset.mem_range] at hmem
    simp only [Finset.mem_range]; exact hmem.2.2
  · intro j hj
    simp only [Finset.mem_range] at hj
    simp only [Finset.mem_sigma, Finset.mem_filter, Finset.mem_range, Finset.mem_univ, true_and]
    constructor
    · rw [Nat.div_lt_iff_lt_mul hT]; omega
    · rw [Nat.div_add_mod' j TILE_N]; exact hj
  · rintro ⟨ℓ, t⟩ hmem
    have hℓ : ℓ.val < TILE_N := ℓ.isLt
    have hmod : (t * TILE_N + ℓ.val) % TILE_N = ℓ.val := Nat.mul_add_mod_of_lt hℓ
    have hdiv : (t * TILE_N + ℓ.val) / TILE_N = t := by
      rw [Nat.add_comm, Nat.add_mul_div_right _ _ hT, Nat.div_eq_of_lt hℓ, Nat.zero_add]
    apply Sigma.ext
    · apply Fin.ext; simpa using hmod
    · simpa using hdiv
  · intro j hj; rw [Nat.div_add_mod' j TILE_N]
  · rintro ⟨ℓ, t⟩ hmem; rfl

/-- Backward accumulation covers every padded lane; no masked product is dropped. -/
theorem lane_tile_partition_padded (B c : Nat) (hB : 0 < B) (V : Nat → ℝ) :
    (∑ i : Fin B, ∑ t ∈ Finset.range c, V (t * B + i.val)) =
      ∑ j ∈ Finset.range (c * B), V j := by
  rw [← lane_tile_partition (c * B) B hB c (Nat.le_refl _) V]
  apply Finset.sum_congr rfl
  intro i _
  apply Finset.sum_congr rfl
  intro t ht
  have ht' := Finset.mem_range.mp ht
  have hbound : t * B + i.val < c * B := by
    have hi := i.isLt
    have hm := Nat.mul_le_mul_right B (Nat.succ_le_of_lt ht')
    simp only [Nat.succ_mul] at hm
    omega
  rw [if_pos hbound]

/-- Combine the per-lane online invariant with exact coverage of the active row. -/
theorem tiled_online_denominator (N B c : Nat) (hB : 0 < B) (hc : N ≤ c * B)
    (x : Nat → ℝ) (m : Fin B → WithBot ℝ) (z : Fin B → ℝ) (M : ℝ)
    (hz : ∀ i, z i = (m i).elim 0 (fun r ↦ Real.exp (-r)) *
      ∑ t ∈ Finset.range c, if t * B + i.val < N then Real.exp (x (t * B + i.val)) else 0)
    (hmL : ∀ i, m i = ⊥ →
      (∑ t ∈ Finset.range c, if t * B + i.val < N then Real.exp (x (t * B + i.val)) else 0) = 0) :
    (∑ i, z i * (WithBot.realExp (WithBot.realSub (m i) (M : WithBot ℝ))).unbotD 0) =
      Real.exp (-M) * ∑ n ∈ Finset.range N, Real.exp (x n) := by
  rw [online_lane_collapse m z _ M hz hmL]
  rw [lane_tile_partition N B hB c hc (fun n ↦ Real.exp (x n))]

/-- Mathematical accumulator for the backward loop's lane-wise product updates. -/
noncomputable def backwardLaneAccumulator (B c : Nat) (V : Nat → ℝ) : Fin B → ℝ :=
  (List.range c).foldl (fun acc t i ↦ acc i + V (t * B + i.val)) (fun _ ↦ 0)

/-- Each accumulator lane retains every product visited by the source loop. -/
theorem backwardLaneAccumulator_eq (B c : Nat) (V : Nat → ℝ) (i : Fin B) :
    backwardLaneAccumulator B c V i =
      ∑ t ∈ Finset.range c, V (t * B + i.val) := by
  induction c with
  | zero => simp [backwardLaneAccumulator]
  | succ c ih =>
    simpa [backwardLaneAccumulator, List.range_succ, List.foldl_append,
      Finset.sum_range_succ] using congrArg (fun z ↦ z + V (c * B + i.val)) ih

/-- The final lane reduction contains exactly the padded row, including inactive lanes. -/
theorem backwardLaneAccumulator_reduce (B c : Nat) (hB : 0 < B) (V : Nat → ℝ) :
    (∑ i, backwardLaneAccumulator B c V i) =
      ∑ j ∈ Finset.range (c * B), V j := by
  simp only [backwardLaneAccumulator_eq]
  exact lane_tile_partition_padded B c hB V

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealMath
