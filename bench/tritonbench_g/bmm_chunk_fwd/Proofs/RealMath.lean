import bench.tritonbench_g.bmm_chunk_bwd.Proofs.RealMath

/-! Independent full-K contraction with actual row/column padding and optional
signed sequence filter. K need not be a multiple of the dot block width.
These are real identities; the FP proof never uses reassociation. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealMath
open scoped BigOperators
noncomputable section

def contraction (K : Nat) (a b : Nat → ℝ) : ℝ :=
  ∑ k ∈ Finset.range K, a k * b k

def partialSum (length row col K width count : Nat) (a b : Nat → ℝ) : ℝ :=
  if row < length ∧ col < length then
    BmmChunkBwd.RealMath.partialSum K width count (fun k => a k * b k) else 0

def formula (length row col K : Nat) (a b : Nat → ℝ)
    (rowSeq colSeq : Int) (hasSeq : Bool) : ℝ :=
  if row < length ∧ col < length ∧ (hasSeq = Bool.false ∨ rowSeq = colSeq)
  then contraction K a b else 0

theorem partial_zero (length row col K width : Nat) (a b : Nat → ℝ) :
    partialSum length row col K width 0 a b = 0 := by
  simp [partialSum, BmmChunkBwd.RealMath.partial_zero]

theorem partial_succ (length row col K width count : Nat) (a b : Nat → ℝ) :
    partialSum length row col K width (count + 1) a b =
      partialSum length row col K width count a b +
      (if row < length ∧ col < length then
        ∑ lane : Fin width, if lane.val < K - count * width then
          a (count * width + lane.val) * b (count * width + lane.val) else 0
       else 0) := by
  by_cases h : row < length ∧ col < length
  · simp only [partialSum, h]
    exact BmmChunkBwd.RealMath.partial_succ K width count (fun k => a k * b k)
  · simp [partialSum, h]

theorem partial_complete (length row col K width : Nat) (hwidth : 0 < width)
    (a b : Nat → ℝ) :
    partialSum length row col K width (BmmChunkBwd.RealMath.blocks K width) a b =
      if row < length ∧ col < length then contraction K a b else 0 := by
  simp only [partialSum, BmmChunkBwd.RealMath.partial_complete K width hwidth, contraction]

/-- The literal -1 and -2 sentinels are retained. Padding contributes zero even
when a real sequence ID happens to equal the opposite inactive sentinel. -/
theorem sequence_filter (length row col K : Nat) (a b : Nat → ℝ)
    (rowSeq colSeq : Int) (hasSeq : Bool) :
    (if hasSeq then
      if (if row < length then rowSeq else -1) = (if col < length then colSeq else -2)
      then (if row < length ∧ col < length then contraction K a b else 0) else 0
     else (if row < length ∧ col < length then contraction K a b else 0)) =
    formula length row col K a b rowSeq colSeq hasSeq := by
  cases hasSeq <;> by_cases hr : row < length <;> by_cases hc : col < length <;>
    simp [formula, hr, hc]

theorem formula_padded (length row col K : Nat) (a b : Nat → ℝ)
    (rowSeq colSeq : Int) (hasSeq : Bool) (h : length ≤ row ∨ length ≤ col) :
    formula length row col K a b rowSeq colSeq hasSeq = 0 := by
  rcases h with h | h <;> simp [formula, Nat.not_lt.mpr h]

theorem sequence_mismatch (length row col K : Nat) (a b : Nat → ℝ)
    (rowSeq colSeq : Int) (h : rowSeq ≠ colSeq) :
    formula length row col K a b rowSeq colSeq Bool.true = 0 := by
  simp [formula, h]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealMath
