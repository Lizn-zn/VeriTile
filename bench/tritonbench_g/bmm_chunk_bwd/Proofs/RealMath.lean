import VeriTile.Triton.Core
import VeriTile.Meta.StatementAudit
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Tactic

/-! The independent contraction and its masked block decomposition. The final
block may be partial; no divisibility assumption on the chunk length is used.
These are real identities, not admitted floating-point reassociations. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkBwd.RealMath
open scoped BigOperators
noncomputable section

def chunkLimit (seqlen chunk pid : Nat) : Nat := min chunk (seqlen - pid * chunk)

def blocks (length width : Nat) : Nat := (length + width - 1) / width

/-- One output entry, expressed only using the initial inputs in this chunk.
The Dout argument has already selected the transposed output-row coordinate. -/
def formula (length : Nat) (dout a : Nat → ℝ) (residual : ℝ) (hasResidual : Bool) : ℝ :=
  (∑ k ∈ Finset.range length, dout k * a k) + if hasResidual then residual else 0

def partialSum (length width count : Nat) (f : Nat → ℝ) : ℝ :=
  ∑ k ∈ Finset.range (count * width), if k < length then f k else 0

theorem tail_mask (length width count lane : Nat) :
    lane < length - count * width ↔ count * width + lane < length := by omega

theorem blocks_cover (length width : Nat) (hwidth : 0 < width) :
    length ≤ blocks length width * width := by
  have h := Nat.lt_mul_div_succ (length + width - 1) hwidth
  rw [Nat.mul_add, Nat.mul_one, Nat.mul_comm width] at h
  unfold blocks
  omega

theorem partial_zero (length width : Nat) (f : Nat → ℝ) :
    partialSum length width 0 f = 0 := by simp [partialSum]

/-- The exact tail predicate used by the source's two masked loads. -/
theorem partial_succ (length width count : Nat) (f : Nat → ℝ) :
    partialSum length width (count + 1) f = partialSum length width count f +
      ∑ lane : Fin width,
        if lane.val < length - count * width then f (count * width + lane.val) else 0 := by
  simp only [partialSum, Nat.add_mul, Nat.one_mul, Finset.sum_range_add]
  congr 1
  rw [Finset.sum_range]
  apply Finset.sum_congr rfl
  intro lane _
  simp only [tail_mask]

/-- Zero padding in the last dot block contributes no additional terms. -/
theorem partial_complete (length width : Nat) (hwidth : 0 < width) (f : Nat → ℝ) :
    partialSum length width (blocks length width) f = ∑ k ∈ Finset.range length, f k := by
  have hcover := blocks_cover length width hwidth
  unfold partialSum
  have hsubset : Finset.range length ⊆ Finset.range (blocks length width * width) :=
    Finset.range_mono hcover
  have hsum := Finset.sum_subset (f := fun k => if k < length then f k else 0) hsubset
    (by intro k _ hk; simp only [Finset.mem_range] at hk; simp [hk])
  rw [← hsum]
  apply Finset.sum_congr rfl
  intro k hk
  simp only [Finset.mem_range] at hk
  simp [hk]

/-- Both load masks share the contraction-tail predicate. Row/column masks
are independent of the contraction coordinate and factor out of the dot. -/
theorem masked_dot (length width count : Nat) (row col : Prop)
    [Decidable row] [Decidable col] (dout a : Nat → ℝ) :
    (∑ lane : Fin width,
      (if row ∧ lane.val < length - count * width then dout (count * width + lane.val) else 0) *
      (if lane.val < length - count * width ∧ col then a (count * width + lane.val) else 0)) =
    if row ∧ col then
      ∑ lane : Fin width, if lane.val < length - count * width then
        dout (count * width + lane.val) * a (count * width + lane.val) else 0
    else 0 := by
  by_cases hr : row <;> by_cases hc : col
  · simp only [hr, hc, true_and, and_true, ↓reduceIte]
    apply Finset.sum_congr rfl
    intro lane _
    split <;> simp_all
  all_goals simp [hr, hc]

theorem with_residual (length width : Nat) (hwidth : 0 < width)
    (dout a : Nat → ℝ) (residual : ℝ) (hasResidual : Bool) :
    partialSum length width (blocks length width) (fun k => dout k * a k) +
      (if hasResidual then residual else 0) = formula length dout a residual hasResidual := by
  rw [partial_complete length width hwidth]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmChunkBwd.RealMath
