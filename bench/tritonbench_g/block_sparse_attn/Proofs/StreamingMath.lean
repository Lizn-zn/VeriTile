import VeriTile.Triton.Semantics.TileOps
import VeriTile.Meta.StatementAudit

/-! Real row mathematics for block-sparse attention. Scores can be masked
bottom values; each source-style normalized update agrees with an independent
weighted sum. Empty rows and all-masked prefixes need no visibility premise.
This uses the project's total real division and masked exponential semantics.
Connection to the complete source execution and memory is a separate proof. -/
open VeriTile.Triton
open scoped BigOperators
namespace VeriTile.Bench.TritonBenchG.BlockSparseAttn.StreamingMath
noncomputable section

/-- Exponentiated masked score in the project's real semantics. -/
def weight (x : WithBot ℝ) : ℝ := (WithBot.realExp x).unbotD 0

/-- The actual score subtraction followed by exponentiation. -/
def shifted (x m : WithBot ℝ) : ℝ :=
  weight (WithBot.realSub x m)

@[simp] theorem weight_bot : weight ⊥ = 0 := rfl
@[simp] theorem weight_coe (x : ℝ) : weight (x : WithBot ℝ) = Real.exp x := rfl
@[simp] theorem shifted_bot_left (m : WithBot ℝ) : shifted ⊥ m = 0 := by
  cases m <;> rfl

/-- A masked lane stays zero, including an all-masked block. -/
theorem shifted_factor {x m : WithBot ℝ} (h : x ≤ m) :
    shifted x m = Real.exp (-m.unbotD 0) * weight x := by
  cases x with
  | bot => simp
  | coe x =>
    cases m with
    | bot => exact False.elim (WithBot.not_coe_le_bot x h)
    | coe m =>
      change Real.exp (x - m) = Real.exp (-m) * Real.exp x
      rw [← Real.exp_add]
      congr 1
      ring

theorem shift_rescale {m n : WithBot ℝ} (h : m ≤ n) (a : ℝ)
    (hz : m = ⊥ → a = 0) :
    shifted m n * (Real.exp (-m.unbotD 0) * a) =
      Real.exp (-n.unbotD 0) * a := by
  cases m with
  | bot => simp [hz rfl]
  | coe m =>
    cases n with
    | bot => exact False.elim (WithBot.not_coe_le_bot m h)
    | coe n =>
      change Real.exp (m - n) * (Real.exp (-m) * a) = Real.exp (-n) * a
      rw [← mul_assoc, ← Real.exp_add]
      congr 2
      ring

def blockMax {B : Nat} (score : Nat → Fin B → WithBot ℝ) (k : Nat) : WithBot ℝ :=
  Finset.univ.sup (score k)

def maximum {B : Nat} (score : Nat → Fin B → WithBot ℝ) : Nat → WithBot ℝ
  | 0 => ⊥
  | k + 1 => max (maximum score k) (blockMax score k)

def mass {B : Nat} (score : Nat → Fin B → WithBot ℝ) (k : Nat) : ℝ :=
  ∑ n ∈ Finset.range k, ∑ j : Fin B, weight (score n j)

def moment {B : Nat} (score : Nat → Fin B → WithBot ℝ)
    (value : Nat → Fin B → ℝ) (k : Nat) : ℝ :=
  ∑ n ∈ Finset.range k, ∑ j : Fin B, weight (score n j) * value n j

theorem score_le_blockMax {B : Nat} (score : Nat → Fin B → WithBot ℝ)
    (k : Nat) (j : Fin B) : score k j ≤ blockMax score k :=
  Finset.le_sup (Finset.mem_univ j)

theorem maximum_step {B : Nat} (score : Nat → Fin B → WithBot ℝ) (k : Nat) :
    maximum score k ≤ maximum score (k + 1) := le_max_left _ _

theorem score_le_maximum {B : Nat} (score : Nat → Fin B → WithBot ℝ)
    {n k : Nat} (h : n < k) (j : Fin B) : score n j ≤ maximum score k := by
  induction k with
  | zero => omega
  | succ k ih =>
    by_cases hn : n = k
    · subst n
      exact le_trans (score_le_blockMax score k j) (le_max_right _ _)
    · exact le_trans (ih (by omega)) (maximum_step score k)

theorem weight_zero_of_maximum_bot {B : Nat} (score : Nat → Fin B → WithBot ℝ)
    {n k : Nat} (h : n < k) (j : Fin B) (hm : maximum score k = ⊥) :
    weight (score n j) = 0 := by
  have hs := score_le_maximum score h j
  rw [hm] at hs
  rw [le_bot_iff.mp hs, weight_bot]

theorem mass_zero_of_maximum_bot {B : Nat} (score : Nat → Fin B → WithBot ℝ)
    {k : Nat} (hm : maximum score k = ⊥) : mass score k = 0 := by
  apply Finset.sum_eq_zero
  intro n hn
  apply Finset.sum_eq_zero
  intro j _
  exact weight_zero_of_maximum_bot score (Finset.mem_range.mp hn) j hm

theorem moment_zero_of_maximum_bot {B : Nat} (score : Nat → Fin B → WithBot ℝ)
    (value : Nat → Fin B → ℝ) {k : Nat} (hm : maximum score k = ⊥) :
    moment score value k = 0 := by
  apply Finset.sum_eq_zero
  intro n hn
  apply Finset.sum_eq_zero
  intro j _
  rw [weight_zero_of_maximum_bot score (Finset.mem_range.mp hn) j hm, zero_mul]

/-- Unnormalized accumulators shifted by the new running maximum. -/
def normalizer {B : Nat} (score : Nat → Fin B → WithBot ℝ) : Nat → ℝ
  | 0 => 0
  | k + 1 => shifted (maximum score k) (maximum score (k+1)) * normalizer score k +
      ∑ j : Fin B, shifted (score k j) (maximum score (k+1))

def numerator {B : Nat} (score : Nat → Fin B → WithBot ℝ)
    (value : Nat → Fin B → ℝ) : Nat → ℝ
  | 0 => 0
  | k + 1 => shifted (maximum score k) (maximum score (k+1)) * numerator score value k +
      ∑ j : Fin B, shifted (score k j) (maximum score (k+1)) * value k j

theorem normalizer_eq_mass {B : Nat} (score : Nat → Fin B → WithBot ℝ) (k : Nat) :
    normalizer score k = Real.exp (-(maximum score k).unbotD 0) * mass score k := by
  induction k with
  | zero => simp [normalizer, mass]
  | succ k ih =>
    rw [normalizer, ih, shift_rescale (maximum_step score k) _
      (mass_zero_of_maximum_bot score)]
    simp only [mass, Finset.sum_range_succ, mul_add, Finset.mul_sum]
    congr 1
    apply Finset.sum_congr rfl
    intro j _
    exact shifted_factor (score_le_maximum score (Nat.lt_succ_self k) j)

theorem numerator_eq_moment {B : Nat} (score : Nat → Fin B → WithBot ℝ)
    (value : Nat → Fin B → ℝ) (k : Nat) :
    numerator score value k =
      Real.exp (-(maximum score k).unbotD 0) * moment score value k := by
  induction k with
  | zero => simp [numerator, moment]
  | succ k ih =>
    rw [numerator, ih, shift_rescale (maximum_step score k) _
      (moment_zero_of_maximum_bot score value)]
    simp only [moment, Finset.sum_range_succ, mul_add, Finset.mul_sum]
    congr 1
    apply Finset.sum_congr rfl
    intro j _
    rw [shifted_factor (score_le_maximum score (Nat.lt_succ_self k) j), mul_assoc]

/-- Includes empty CSR rows, arbitrary masked prefixes and repeated blocks.
Real division is total; a zero denominator has the project's real value zero. -/
theorem streaming_ratio {B : Nat} (score : Nat → Fin B → WithBot ℝ)
    (value : Nat → Fin B → ℝ) (k : Nat) :
    numerator score value k / normalizer score k = moment score value k / mass score k := by
  rw [numerator_eq_moment, normalizer_eq_mass,
    mul_div_mul_left _ _ (Real.exp_ne_zero _)]


theorem weight_nonneg (x : WithBot ℝ) : 0 ≤ weight x := by
  cases x <;> simp [le_of_lt (Real.exp_pos _)]

theorem moment_zero_of_mass_zero {B : Nat} (score : Nat → Fin B → WithBot ℝ)
    (value : Nat → Fin B → ℝ) {k : Nat} (hm : mass score k = 0) :
    moment score value k = 0 := by
  have hn := (Finset.sum_eq_zero_iff_of_nonneg (fun n (_ : n ∈ Finset.range k) =>
    Finset.sum_nonneg (fun j _ => weight_nonneg (score n j)))).mp hm
  apply Finset.sum_eq_zero
  intro n hnmem
  have hj := (Finset.sum_eq_zero_iff_of_nonneg (fun j (_ : j ∈ Finset.univ) =>
    weight_nonneg (score n j))).mp (hn n hnmem)
  exact Finset.sum_eq_zero (fun j hjmem => by rw [hj j hjmem, zero_mul])

theorem numerator_zero_of_normalizer_zero {B : Nat}
    (score : Nat → Fin B → WithBot ℝ) (value : Nat → Fin B → ℝ)
    {k : Nat} (hl : normalizer score k = 0) : numerator score value k = 0 := by
  rw [normalizer_eq_mass, mul_eq_zero] at hl
  have hm := hl.resolve_left (Real.exp_ne_zero _)
  rw [numerator_eq_moment, moment_zero_of_mass_zero score value hm, mul_zero]

/-- Composition of the block shift and running shift, including masked blocks. -/
theorem shifted_compose {x m n : WithBot ℝ} (hxm : x ≤ m) (hmn : m ≤ n) :
    shifted m n * shifted x m = shifted x n := by
  cases x with
  | bot => simp
  | coe x =>
    cases m with
    | bot => exact False.elim (WithBot.not_coe_le_bot x hxm)
    | coe m =>
      cases n with
      | bot => exact False.elim (WithBot.not_coe_le_bot m hmn)
      | coe n =>
        change Real.exp (m - n) * Real.exp (x - m) = Real.exp (x - n)
        rw [← Real.exp_add]
        congr 1
        ring

/-- `beta * l_ij` is the same mass expressed at the new running maximum. -/
theorem block_normalizer {B : Nat} (score : Nat → Fin B → WithBot ℝ) (k : Nat) :
    shifted (maximum score k) (maximum score (k+1)) * normalizer score k +
      shifted (blockMax score k) (maximum score (k+1)) *
        (∑ j : Fin B, shifted (score k j) (blockMax score k)) = normalizer score (k+1) := by
  rw [normalizer, Finset.mul_sum]
  congr 1
  apply Finset.sum_congr rfl
  intro j _
  exact shifted_compose (score_le_blockMax score k j) (le_max_right _ _)

theorem block_numerator {B : Nat} (score : Nat → Fin B → WithBot ℝ)
    (value : Nat → Fin B → ℝ) (k : Nat) :
    shifted (maximum score k) (maximum score (k+1)) * numerator score value k +
      shifted (blockMax score k) (maximum score (k+1)) *
        (∑ j : Fin B, shifted (score k j) (blockMax score k) * value k j) =
      numerator score value (k+1) := by
  rw [numerator, Finset.mul_sum]
  congr 1
  apply Finset.sum_congr rfl
  intro j _
  rw [← mul_assoc]
  have hm : blockMax score k ≤ maximum score (k+1) := le_max_right _ _
  exact congrArg (· * value k j) (shifted_compose (score_le_blockMax score k j) hm)

/-- The actual `acc_scale` and `p_scale` update. A zero old normalizer is
allowed when its unnormalized numerator is zero, as for an all-masked prefix. -/
theorem normalized_update {B : Nat} (a l L alpha beta : ℝ)
    (p v : Fin B → ℝ) (hz : l = 0 → a = 0) :
    (a / l) * (l / L * alpha) + (∑ j : Fin B, (p j * (beta / L)) * v j) =
      (alpha * a + beta * ∑ j : Fin B, p j * v j) / L := by
  have hc : a / l * l = a := by
    by_cases hl : l = 0
    · simp [hl, hz hl]
    · exact div_mul_cancel₀ a hl
  have hs : (∑ j : Fin B, (p j * (beta / L)) * v j) =
      (beta / L) * ∑ j : Fin B, p j * v j := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro j _
    ring
  rw [hs]
  calc
    _ = (alpha * (a / l * l) + beta * ∑ j : Fin B, p j * v j) / L := by ring
    _ = _ := by rw [hc]

/-- Source-shaped recurrence: block maximum, block probabilities, new
normalizer, then scaled old accumulator plus normalized probability/value dot. -/
def accumulated {B : Nat} (score : Nat → Fin B → WithBot ℝ)
    (value : Nat → Fin B → ℝ) : Nat → ℝ
  | 0 => 0
  | k + 1 =>
    let mBlock := blockMax score k
    let mNew := maximum score (k+1)
    let p := fun j => shifted (score k j) mBlock
    let alpha := shifted (maximum score k) mNew
    let beta := shifted mBlock mNew
    let lNew := alpha * normalizer score k + beta * ∑ j : Fin B, p j
    accumulated score value k * (normalizer score k / lNew * alpha) +
      ∑ j : Fin B, (p j * (beta / lNew)) * value k j

theorem accumulated_eq_ratio {B : Nat} (score : Nat → Fin B → WithBot ℝ)
    (value : Nat → Fin B → ℝ) (k : Nat) :
    accumulated score value k = numerator score value k / normalizer score k := by
  induction k with
  | zero => simp [accumulated, numerator, normalizer]
  | succ k ih =>
    rw [accumulated]
    dsimp only
    rw [ih, normalized_update _ _ _ _ _ _ _ (numerator_zero_of_normalizer_zero score value),
      block_numerator, block_normalizer]

/-- Source-shaped normalized recurrence equals the independent weighted sum.
This mathematical bridge does not replace the separate source-execution proof. -/
theorem accumulated_eq_formula {B : Nat} (score : Nat → Fin B → WithBot ℝ)
    (value : Nat → Fin B → ℝ) (k : Nat) :
    accumulated score value k = moment score value k / mass score k := by
  rw [accumulated_eq_ratio, streaming_ratio]

#guard_msgs (drop info) in
#auditModuleAxioms

end
end VeriTile.Bench.TritonBenchG.BlockSparseAttn.StreamingMath
