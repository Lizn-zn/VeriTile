import Mathlib.Data.Int.Basic
import Mathlib.Tactic
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.VarLenCopy.Math

/-- Number of iterations of the signed range, for a positive block size. -/
def rounds (length : Int) (B : Nat) : Nat := (length.toNat + B - 1) / B

/-- Exact source footprint. The lane mask ignores the iteration index. -/
def touched (length : Int) (B n : Nat) : Prop :=
  ∃ c lane, c * B < length.toNat ∧ lane < B ∧ (lane : Int) < length ∧ n = c * B + lane

/-- Contiguous number of copied elements, including the source's final overcopy. -/
def copiedLength (length : Int) (B : Nat) : Nat :=
  if length.toNat ≤ B then length.toNat else rounds length B * B

 theorem rounds_coverage (length : Int) {B : Nat} (hB : 0 < B) :
    length.toNat ≤ rounds length B * B := by
  have hr := Nat.mod_lt (length.toNat + B - 1) hB
  have he := Nat.mod_add_div (length.toNat + B - 1) B
  rw [Nat.mul_comm B] at he
  unfold rounds
  omega

 theorem rounds_last (length : Int) {B : Nat} (hL : 0 < length) (hB : 0 < B) :
    (rounds length B - 1) * B < length.toNat := by
  have hq : 0 < rounds length B := by
    unfold rounds
    apply Nat.div_pos <;> omega
  have hd := Nat.div_mul_le_self (length.toNat + B - 1) B
  have hp : B ≤ rounds length B * B := by simpa using Nat.mul_le_mul_right B hq
  rw [Nat.sub_mul]
  unfold rounds at *
  omega

 theorem loop_test (length : Int) {B c : Nat} (hB : 0 < B) :
    (c * B : Nat) < length.toNat ↔ c < rounds length B := by
  by_cases hL : 0 < length
  · constructor
    · intro h
      have := rounds_coverage length hB
      by_contra hc
      have := Nat.mul_le_mul_right B (Nat.le_of_not_gt hc)
      omega
    · intro h
      have hm := Nat.mul_le_mul_right B (show c ≤ rounds length B - 1 by omega)
      have := rounds_last length hL hB
      omega
  · have hz : length.toNat = 0 := by omega
    have hdiv : (B - 1) / B = 0 := Nat.div_eq_of_lt (by omega)
    simp [rounds, hz, hdiv]

 theorem touched_iff (length : Int) {B n : Nat} (hB : 0 < B) :
    touched length B n ↔ n < copiedLength length B := by
  by_cases hsmall : length.toNat ≤ B
  · simp only [copiedLength, if_pos hsmall]
    constructor
    · rintro ⟨c,lane,hc,hl,hL,rfl⟩
      have hzero : c = 0 := by
        by_contra hn
        have hm := Nat.mul_le_mul_right B (show 1 ≤ c by omega)
        simp only [Nat.one_mul] at hm
        omega
      subst c
      omega
    · intro hn
      exact ⟨0,n,by omega,by omega,by omega,by omega⟩
  · simp only [copiedLength, if_neg hsmall]
    constructor
    · rintro ⟨c,lane,hc,hl,_,rfl⟩
      have hq := (loop_test length hB).mp hc
      have hm := Nat.mul_le_mul_right B hq
      simp only [Nat.add_mul, Nat.one_mul, Nat.succ_mul] at hm
      omega
    · intro hn
      refine ⟨n / B, n % B, ?_, Nat.mod_lt n hB, ?_, ?_⟩
      · apply (loop_test length hB).mpr
        exact (Nat.div_lt_iff_lt_mul hB).mpr hn
      · have := Nat.mod_lt n hB
        omega
      · have he := Nat.mod_add_div n B
        rw [Nat.mul_comm B] at he
        omega

 theorem copiedLength_nonpositive {length : Int} (h : length ≤ 0) (B : Nat) :
    copiedLength length B = 0 := by
  have hz : length.toNat = 0 := by omega
  simp [copiedLength, hz]

/-- This concrete discrepancy is the source mask, not an intended-length copy. -/
theorem three_by_two_overcopies : copiedLength 3 2 = 4 ∧ touched 3 2 3 := by
  constructor
  · decide
  · exact ⟨1,1,by decide,by decide,by decide,by decide⟩

end VeriTile.Bench.TritonBenchG.VarLenCopy.Math

#guard_msgs (drop info) in
#auditModuleAxioms
