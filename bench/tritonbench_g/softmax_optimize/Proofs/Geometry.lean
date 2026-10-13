import VeriTile.Triton
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.SoftmaxOptimize.FullSource

/-- The signed helper obtained by inlining pinned Triton's cdiv definition. -/
def prevMultipleInt (N B : Nat) : Int :=
  ((N : Int) + (B : Int) - 1) / (B : Int) * (B : Int) - (B : Int)

/-- Nat presentation used only after proving the signed result nonnegative. -/
def prevMultipleNat (N B : Nat) : Nat := (N+B-1)/B*B-B

theorem prevMultipleInt_zero (B : Nat) (hB : 0<B) :
    prevMultipleInt 0 B = -(B : Int) := by
  have hb : (0 : Int)<B := by exact_mod_cast hB
  have hz : ((B : Int)-1)/(B : Int)=0 := Int.ediv_eq_zero_of_lt (by omega) (by omega)
  simp [prevMultipleInt, hz]

/-- At N=0 the original masked tail starts at a negative multiple and its
`n_offsets < N` mask admits every lane of that tile. -/
theorem zero_tail_mask (B : Nat) (hB : 0<B) (i : Fin B) :
    prevMultipleInt 0 B + (i.val : Int) < 0 := by
  rw [prevMultipleInt_zero B hB]
  exact_mod_cast (show -(B : Int)+(i.val : Int)<0 by omega)

theorem prevMultipleInt_eq_nat (N B : Nat) (hN : 0<N) (hB : 0<B) :
    prevMultipleInt N B = (prevMultipleNat N B : Int) := by
  have hsum : 1≤N+B := by omega
  have hq : 0<(N+B-1)/B := Nat.div_pos (by omega) hB
  have hmul : B≤(N+B-1)/B*B := by nlinarith
  unfold prevMultipleInt prevMultipleNat
  rw [Int.ofNat_sub hmul, Int.natCast_mul, Int.natCast_ediv,
    Int.ofNat_sub hsum, Int.natCast_add]
  rfl

/-- Every accumulator lane starts with a finite input; this excludes the
never-active -infinity subtraction boundary of the IEEE source. -/
def NumericalDomain (N B : Nat) : Prop := 0<B ∧ B≤N

theorem numericalDomain_nonempty : NumericalDomain 3 2 := by
  constructor <;> norm_num

theorem numericalDomain_positive {N B : Nat} (h : NumericalDomain N B) : 0<N := by
  exact lt_of_lt_of_le h.1 h.2

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxOptimize.FullSource
