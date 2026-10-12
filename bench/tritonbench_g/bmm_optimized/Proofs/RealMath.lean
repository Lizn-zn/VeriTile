import bench.tritonbench_g.bmm_chunk_bwd.Proofs.RealMath

/-! Independent matrix-product arithmetic. Mask elision on the contraction
axis is sound only under the host's divisibility condition. The source has
no explicit zero fill; the execution proof must supply its undef contract. -/
namespace VeriTile.Bench.TritonBenchG.BmmOptimized.RealMath
open scoped BigOperators
noncomputable section

def active (divisible : Bool) (index extent : Nat) : Prop :=
  divisible = Bool.true ∨ index < extent

instance (divisible : Bool) (index extent : Nat) : Decidable (active divisible index extent) :=
  inferInstanceAs (Decidable (divisible = Bool.true ∨ index < extent))

def contraction (K : Nat) (a b : Nat → ℝ) := ∑ k ∈ Finset.range K, a k * b k

def partialSum (M N row col K width count : Nat) (divM divN : Bool) (a b : Nat → ℝ) :=
  if active divM row M ∧ active divN col N then
    BmmChunkBwd.RealMath.partialSum K width count (fun k => a k * b k) else 0

theorem blocks_exact (K width : Nat) (hwidth : 0 < width) (hdiv : width ∣ K) :
    BmmChunkBwd.RealMath.blocks K width * width = K := by
  obtain ⟨q,rfl⟩ := hdiv
  rw [BmmChunkBwd.RealMath.blocks, Nat.add_sub_assoc hwidth,
    Nat.mul_add_div hwidth, Nat.div_eq_of_lt (show width - 1 < width by omega), Nat.add_zero, Nat.mul_comm]

theorem reached_lane (K width count : Nat) (hwidth : 0 < width) (hdiv : width ∣ K)
    (hcount : count < BmmChunkBwd.RealMath.blocks K width) (lane : Fin width) :
    count * width + lane.val < K := by
  have h := Nat.mul_le_mul_right width (Nat.succ_le_of_lt hcount)
  rw [Nat.succ_mul, blocks_exact K width hwidth hdiv] at h
  omega

/-- Every elided K-mask is true in every reached iteration when K is divisible. -/
theorem contraction_mask (K width count : Nat) (divK : Bool) (hwidth : 0 < width)
    (hdiv : divK = Bool.true → width ∣ K)
    (hcount : count < BmmChunkBwd.RealMath.blocks K width) (lane : Fin width) :
    active divK (count * width + lane.val) K ↔ count * width + lane.val < K := by
  cases hd : divK
  · simp [active]
  · have h := reached_lane K width count hwidth (hdiv hd) hcount lane
    simp [active, h]

theorem masked_dot (M N row col K width count : Nat) (divM divN divK : Bool)
    (hwidth : 0 < width) (hdiv : divK = Bool.true → width ∣ K)
    (hcount : count < BmmChunkBwd.RealMath.blocks K width) (a b : Nat → ℝ) :
    (∑ lane : Fin width,
      (if active divM row M ∧ active divK (count * width + lane.val) K then a (count * width + lane.val) else 0) *
      (if active divK (count * width + lane.val) K ∧ active divN col N then b (count * width + lane.val) else 0)) =
      if active divM row M ∧ active divN col N then
        ∑ lane : Fin width, if lane.val < K - count * width then a (count * width + lane.val) * b (count * width + lane.val) else 0
      else 0 := by
  simp_rw [contraction_mask K width count divK hwidth hdiv hcount,
    ← BmmChunkBwd.RealMath.tail_mask K width count]
  exact BmmChunkBwd.RealMath.masked_dot K width count (active divM row M) (active divN col N) a b

theorem partial_zero (M N row col K width : Nat) (divM divN : Bool) (a b : Nat → ℝ) :
    partialSum M N row col K width 0 divM divN a b = 0 := by
  simp [partialSum, BmmChunkBwd.RealMath.partial_zero]

theorem partial_succ (M N row col K width count : Nat) (divM divN : Bool) (a b : Nat → ℝ) :
    partialSum M N row col K width (count+1) divM divN a b =
      partialSum M N row col K width count divM divN a b +
      (if active divM row M ∧ active divN col N then
        ∑ lane : Fin width, if lane.val < K - count * width then a (count * width + lane.val) * b (count * width + lane.val) else 0
       else 0) := by
  by_cases h : active divM row M ∧ active divN col N
  · simp only [partialSum, h]
    exact BmmChunkBwd.RealMath.partial_succ K width count (fun k => a k * b k)
  · simp [partialSum, h]

theorem partial_complete (M N row col K width : Nat) (divM divN : Bool) (hwidth : 0 < width) (a b : Nat → ℝ) :
    partialSum M N row col K width (BmmChunkBwd.RealMath.blocks K width) divM divN a b =
      if active divM row M ∧ active divN col N then contraction K a b else 0 := by
  simp only [partialSum, BmmChunkBwd.RealMath.partial_complete K width hwidth, contraction]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmOptimized.RealMath
