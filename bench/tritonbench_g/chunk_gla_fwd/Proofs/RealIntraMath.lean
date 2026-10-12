import VeriTile.Triton
import VeriTile.Meta.StatementAudit

/-! Independent real contraction for the intra-subblock producer. Matrix
query/gate inputs use timeStride; streamed key/gate inputs use K. The target
sums exactly K features. Unspecified matrix padding is kept in the internal
tile formula and eliminated only for valid rows, using the source's explicit
zero-masked key load. This is a real identity, not an FP admission. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealIntraMath
open scoped BigOperators
noncomputable section
structure Layout where
  headStride : Nat
  timeStride : Nat
  T : Nat
  K : Nat
  BT : Nat
  BC : Nat
  BK : Nat
structure Launch where
  chunk : Nat
  block : Nat
  head : Nat
structure Inputs where
  query : Nat → ℝ
  key : Nat → ℝ
  gate : Nat → ℝ

def rowStart (p : Layout) (l : Launch) := l.chunk*p.BT+l.block*p.BC
def columns (p : Layout) (l : Launch) := min p.BC (p.T-l.chunk*p.BT-l.block*p.BC)
def enabled (p : Layout) (l : Launch) : Prop := rowStart p l < p.T
def rowActive (p : Layout) (l : Launch) (i : Nat) : Prop := rowStart p l+i < p.T
def matrixAddress (p : Layout) (l : Launch) (i k : Nat) :=
  l.head*p.headStride+(rowStart p l+i)*p.timeStride+k
def keyAddress (p : Layout) (l : Launch) (j k : Nat) :=
  l.head*p.headStride+(rowStart p l+j)*p.K+k
def outputAddress (p : Layout) (l : Launch) (i j : Nat) :=
  l.head*p.T*p.BT+(rowStart p l+i)*p.BT+l.block*p.BC+j

def term (p : Layout) (l : Launch) (a : Inputs) (i j k : Nat) : ℝ :=
  a.query (matrixAddress p l i k) * a.key (keyAddress p l j k) *
    Real.exp (a.gate (matrixAddress p l i k)-a.gate (keyAddress p l j k))
def expected (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i j : Nat) : ℝ :=
  if j ≤ i then (∑ k ∈ Finset.range p.K, term p l a i j k)*scale else 0

instance (p : Layout) (l : Launch) (i : Nat) : Decidable (rowActive p l i) :=
  inferInstanceAs (Decidable (rowStart p l+i < p.T))

def matrixValue (p : Layout) (l : Launch) (a padding : Nat → ℝ) (i k : Nat) : ℝ :=
  if rowActive p l i ∧ k < p.K then a (matrixAddress p l i k)
  else padding (matrixAddress p l i k)
def keyValue (p : Layout) (l : Launch) (a : Nat → ℝ) (j k : Nat) : ℝ :=
  if k < p.K then a (keyAddress p l j k) else 0
def lane (p : Layout) (l : Launch) (a : Inputs) (qp gp : Nat → ℝ) (i j k : Nat) : ℝ :=
  matrixValue p l a.query qp i k * keyValue p l a.key j k *
    Real.exp (matrixValue p l a.gate gp i k-keyValue p l a.gate j k)
def reduced (p : Layout) (l : Launch) (a : Inputs) (qp gp : Nat → ℝ) (i j : Nat) : ℝ :=
  ∑ k : Fin p.BK, lane p l a qp gp i j k.val
def result (p : Layout) (l : Launch) (a : Inputs) (qp gp : Nat → ℝ) (scale : ℝ) (i j : Nat) : ℝ :=
  if j ≤ i then reduced p l a qp gp i j * scale else 0

theorem lane_valid (p : Layout) (l : Launch) (a : Inputs) (qp gp : Nat → ℝ) (i j k : Nat)
    (hi : rowActive p l i) (hk : k < p.K) :
    lane p l a qp gp i j k = term p l a i j k := by
  simp [lane,matrixValue,keyValue,hi,hk,term]

theorem lane_tail (p : Layout) (l : Launch) (a : Inputs) (qp gp : Nat → ℝ) (i j k : Nat)
    (hk : p.K ≤ k) : lane p l a qp gp i j k = 0 := by
  simp [lane,keyValue,show ¬k < p.K from Nat.not_lt.mpr hk]

theorem reduced_valid (p : Layout) (l : Launch) (a : Inputs) (qp gp : Nat → ℝ) (i j : Nat)
    (hi : rowActive p l i) (hK : p.K ≤ p.BK) :
    reduced p l a qp gp i j = ∑ k ∈ Finset.range p.K, term p l a i j k := by
  rw [reduced,Fin.sum_univ_eq_sum_range]
  calc
    _ = ∑ k ∈ Finset.range p.K, lane p l a qp gp i j k :=
      (Finset.sum_subset (Finset.range_mono hK) (by
        intro k _ hk
        exact lane_tail p l a qp gp i j k (by simpa using hk))).symm
    _ = _ := Finset.sum_congr rfl (fun k hk => lane_valid p l a qp gp i j k hi (Finset.mem_range.mp hk))

theorem result_valid (p : Layout) (l : Launch) (a : Inputs) (qp gp : Nat → ℝ)
    (scale : ℝ) (i j : Nat) (hi : rowActive p l i) (hK : p.K ≤ p.BK) :
    result p l a qp gp scale i j = expected p l a scale i j := by
  simp only [result,expected,reduced_valid p l a qp gp i j hi hK]

/-- At most one full output row is streamed per column range. This condition
prevents two different row/column stores from sharing the same address. -/
theorem output_injective (p : Layout) (l : Launch) (hcols : columns p l ≤ p.BT)
    (i i' j j' : Nat) (hj : j < columns p l) (hj' : j' < columns p l)
    (he : outputAddress p l i j = outputAddress p l i' j') : i = i' ∧ j = j' := by
  have hjT := Nat.lt_of_lt_of_le hj hcols
  have hjT' := Nat.lt_of_lt_of_le hj' hcols
  have h : i*p.BT+j = i'*p.BT+j' := by
    simp only [outputAddress,Nat.add_mul] at he
    omega
  have hm := congrArg (fun n => n % p.BT) h
  have hT : 0 < p.BT := Nat.zero_lt_of_lt hjT
  simp only [Nat.add_mod,Nat.mul_mod_left,zero_add,Nat.mod_eq_of_lt hjT,Nat.mod_eq_of_lt hjT'] at hm
  refine ⟨?_,hm⟩
  exact Nat.eq_of_mul_eq_mul_right hT (by omega)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealIntraMath
