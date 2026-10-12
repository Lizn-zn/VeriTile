import VeriTile.Triton
import VeriTile.Meta.StatementAudit

/-! Independent real contraction for the split intra-subblock producer. Matrix
query/gate inputs use timeStride; streamed key/gate inputs use K. The target
sums the active features of this launch's K tile. Unspecified matrix padding is kept in the internal
tile formula and eliminated only for valid rows, using the source's explicit
zero-masked key load. This is a real identity, not an FP admission. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealSplitMath
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
  NC : Nat
structure Launch where
  chunk : Nat
  block : Nat
  head : Nat
  feature : Nat
  heads : Nat
structure Inputs where
  query : Nat → ℝ
  key : Nat → ℝ
  gate : Nat → ℝ

def rowStart (p : Layout) (l : Launch) := l.chunk*p.BT+l.block*p.BC
def columns (p : Layout) (l : Launch) := min p.BC (p.T-l.chunk*p.BT-l.block*p.BC)
def enabled (p : Layout) (l : Launch) : Prop := rowStart p l < p.T
def rowActive (p : Layout) (l : Launch) (i : Nat) : Prop := rowStart p l+i < p.T
def featureIndex (p : Layout) (l : Launch) (e : Nat) := l.feature*p.BK+e
def matrixAddress (p : Layout) (l : Launch) (i k : Nat) :=
  l.head*p.headStride+(rowStart p l+i)*p.timeStride+featureIndex p l k
def keyAddress (p : Layout) (l : Launch) (j k : Nat) :=
  l.head*p.headStride+(rowStart p l+j)*p.K+featureIndex p l k
def outputAddress (p : Layout) (l : Launch) (i j : Nat) :=
  (l.head+l.feature*l.heads)*p.T*p.BC+(rowStart p l+i)*p.BC+j

def term (p : Layout) (l : Launch) (a : Inputs) (i j k : Nat) : ℝ :=
  a.query (matrixAddress p l i k) * a.key (keyAddress p l j k) *
    Real.exp (a.gate (matrixAddress p l i k)-a.gate (keyAddress p l j k))
def expected (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i j : Nat) : ℝ :=
  if j ≤ i then (∑ k : Fin p.BK, if featureIndex p l k.val < p.K then term p l a i j k.val else 0)*scale else 0

instance (p : Layout) (l : Launch) (i : Nat) : Decidable (rowActive p l i) :=
  inferInstanceAs (Decidable (rowStart p l+i < p.T))

def matrixValue (p : Layout) (l : Launch) (a padding : Nat → ℝ) (i k : Nat) : ℝ :=
  if rowActive p l i ∧ featureIndex p l k < p.K then a (matrixAddress p l i k)
  else padding (matrixAddress p l i k)
def keyValue (p : Layout) (l : Launch) (a : Nat → ℝ) (j k : Nat) : ℝ :=
  if featureIndex p l k < p.K then a (keyAddress p l j k) else 0
def lane (p : Layout) (l : Launch) (a : Inputs) (qp gp : Nat → ℝ) (i j k : Nat) : ℝ :=
  matrixValue p l a.query qp i k * keyValue p l a.key j k *
    Real.exp (matrixValue p l a.gate gp i k-keyValue p l a.gate j k)
def reduced (p : Layout) (l : Launch) (a : Inputs) (qp gp : Nat → ℝ) (i j : Nat) : ℝ :=
  ∑ k : Fin p.BK, lane p l a qp gp i j k.val
def result (p : Layout) (l : Launch) (a : Inputs) (qp gp : Nat → ℝ) (scale : ℝ) (i j : Nat) : ℝ :=
  if j ≤ i then reduced p l a qp gp i j * scale else 0

theorem lane_valid (p : Layout) (l : Launch) (a : Inputs) (qp gp : Nat → ℝ) (i j k : Nat)
    (hi : rowActive p l i) (hk : featureIndex p l k < p.K) :
    lane p l a qp gp i j k = term p l a i j k := by
  simp [lane,matrixValue,keyValue,hi,hk,term]

theorem lane_tail (p : Layout) (l : Launch) (a : Inputs) (qp gp : Nat → ℝ) (i j k : Nat)
    (hk : p.K ≤ featureIndex p l k) : lane p l a qp gp i j k = 0 := by
  simp [lane,keyValue,show ¬featureIndex p l k < p.K from Nat.not_lt.mpr hk]

theorem reduced_valid (p : Layout) (l : Launch) (a : Inputs) (qp gp : Nat → ℝ) (i j : Nat)
    (hi : rowActive p l i) :
    reduced p l a qp gp i j = ∑ k : Fin p.BK,
      if featureIndex p l k.val < p.K then term p l a i j k.val else 0 := by
  apply Finset.sum_congr rfl
  intro k _
  by_cases hk : featureIndex p l k.val < p.K
  · simp only [hk,if_true,lane_valid p l a qp gp i j k.val hi hk]
  · simp only [hk,if_false,lane_tail p l a qp gp i j k.val (Nat.le_of_not_gt hk)]

theorem result_valid (p : Layout) (l : Launch) (a : Inputs) (qp gp : Nat → ℝ)
    (scale : ℝ) (i j : Nat) (hi : rowActive p l i) :
    result p l a qp gp scale i j = expected p l a scale i j := by
  simp only [result,expected,reduced_valid p l a qp gp i j hi]

theorem columns_bound (p : Layout) (l : Launch) : columns p l ≤ p.BC := Nat.min_le_left _ _

/-- The column range fits one temporary row. This prevents different
row/column stores in this launch from sharing an address. -/
theorem output_injective (p : Layout) (l : Launch) (hcols : columns p l ≤ p.BC)
    (i i' j j' : Nat) (hj : j < columns p l) (hj' : j' < columns p l)
    (he : outputAddress p l i j = outputAddress p l i' j') : i = i' ∧ j = j' := by
  have hjT := Nat.lt_of_lt_of_le hj hcols
  have hjT' := Nat.lt_of_lt_of_le hj' hcols
  have h : i*p.BC+j = i'*p.BC+j' := by
    simp only [outputAddress,Nat.add_mul] at he
    omega
  have hm := congrArg (fun n => n % p.BC) h
  have hT : 0 < p.BC := Nat.zero_lt_of_lt hjT
  simp only [Nat.add_mod,Nat.mul_mod_left,zero_add,Nat.mod_eq_of_lt hjT,Nat.mod_eq_of_lt hjT'] at hm
  refine ⟨?_,hm⟩
  exact Nat.eq_of_mul_eq_mul_right hT (by omega)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealSplitMath
