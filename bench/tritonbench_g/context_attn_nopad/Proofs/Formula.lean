import VeriTile.Triton
import VeriTile.Meta.StatementAudit

/-! Independent initial-array formula for context_attn_nopad. The source uses
natural exponential weights with a causal negative-infinity mask. Masked weights
are zero; padded values are zero, and every actual visited key block is retained. -/
namespace VeriTile.Bench.TritonBenchG.ContextAttnNopad.Formula
open VeriTile.Triton
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 1000000

structure Layout where
  BM : Nat
  BN : Nat
  D : Nat
  qRow : Nat
  qHead : Nat
  qDim : Nat
  kHead : Nat
  kRow : Nat
  kDim : Nat
  vHead : Nat
  vRow : Nat
  vDim : Nat
  outRow : Nat
  outHead : Nat
  outDim : Nat

structure Launch where
  block : Nat
  batch : Nat
  head : Nat

structure Inputs where
  q : Nat → ℝ
  k : Nat → ℝ
  v : Nat → ℝ
  start : Nat → Int
  sequence : Nat → Int

def queryRow (p : Layout) (l : Launch) (i : Nat) := l.block*p.BM+i
def sequence (_p : Layout) (l : Launch) (a : Inputs) : Int := a.sequence l.batch
def limit (p : Layout) (l : Launch) (_a : Inputs) : Int := (l.block+1)*p.BM
def count (p : Layout) (l : Launch) (a : Inputs) : Nat :=
  if (l.block*p.BM : Int) < sequence p l a then
    ((l.block+1)*p.BM) / p.BN + if ((l.block+1)*p.BM) % p.BN = 0 then 0 else 1
  else 0

def query (p : Layout) (l : Launch) (a : Inputs) (i d : Nat) : ℝ :=
  if (queryRow p l i : Int) < sequence p l a then
    a.q ((a.start l.batch+queryRow p l i)*p.qRow+l.head*p.qHead+d*p.qDim).toNat
  else 0

def key (p : Layout) (l : Launch) (a : Inputs) (j d : Nat) : ℝ :=
  if (j : Int) < sequence p l a then
    a.k ((a.start l.batch+j)*p.kRow+l.head*p.kHead+d*p.kDim).toNat
  else 0

def value (p : Layout) (l : Launch) (a : Inputs) (j d : Nat) : ℝ :=
  if (j : Int) < sequence p l a then
    a.v ((a.start l.batch+j)*p.vRow+l.head*p.vHead+d*p.vDim).toNat
  else 0

def score (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i j : Nat) : ℝ :=
  (∑ e : Fin p.D, query p l a i e.val * key p l a j e.val) * scale

def weight (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i j : Nat) : ℝ :=
  if j ≤ queryRow p l i then Real.exp (score p l a scale i j) else 0

def denominator (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i c : Nat) : ℝ :=
  ∑ j : Fin (c*p.BN), weight p l a scale i j.val

def numerator (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i d c : Nat) : ℝ :=
  ∑ j : Fin (c*p.BN), weight p l a scale i j.val*value p l a j.val d

/-- Independent causal weighted sum over every key lane visited by the source. -/
def expected (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i d : Nat) : ℝ :=
  numerator p l a scale i d (count p l a) / denominator p l a scale i (count p l a)

theorem count_positive_of_active (p : Layout) (l : Launch) (a : Inputs)
    (i : Fin p.BM)
    (ha : (queryRow p l i.val : Int) < sequence p l a) :
    0 < count p l a := by
  have hBM : 0 < p.BM := Nat.pos_of_ne_zero (by intro hz; simpa [hz] using i.isLt)
  have hstart : (l.block*p.BM : Int) < sequence p l a := by
    unfold queryRow at ha
    exact lt_of_le_of_lt (by omega) ha
  have hn : 0 < (l.block+1)*p.BM := Nat.mul_pos (by omega) hBM
  rw [count,if_pos hstart]
  split_ifs with hmod
  · have hdiv := Nat.div_add_mod ((l.block+1)*p.BM) p.BN
    have hq : ((l.block+1)*p.BM) / p.BN ≠ 0 := by
      intro hz
      simp [hz,hmod] at hdiv
      omega
    simpa only [Nat.add_zero] using Nat.pos_of_ne_zero hq
  · exact Nat.zero_lt_succ _

theorem denominator_positive (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i c : Nat)
    (hBN : 0 < p.BN) (hc : 0 < c) :
    0 < denominator p l a scale i c := by
  have hz : 0 < c*p.BN := Nat.mul_pos hc hBN
  unfold denominator
  apply Finset.sum_pos'
  · intro j _; unfold weight; split_ifs <;> positivity
  · refine ⟨⟨0,hz⟩,Finset.mem_univ _,?_⟩
    simp only [weight,Nat.zero_le,if_true]
    exact Real.exp_pos _

theorem denominator_nonnegative (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i c : Nat) :
    0 ≤ denominator p l a scale i c := by
  apply Finset.sum_nonneg
  intro j _
  unfold weight
  split_ifs <;> positivity

theorem denominator_zero (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i : Nat) :
    denominator p l a scale i 0 = 0 := by
  unfold denominator
  apply Finset.sum_eq_zero
  intro j _
  have h := j.isLt
  simp at h
theorem numerator_zero (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i d : Nat) :
    numerator p l a scale i d 0 = 0 := by
  unfold numerator
  apply Finset.sum_eq_zero
  intro j _
  have h := j.isLt
  simp at h

theorem denominator_succ (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i c : Nat) :
    denominator p l a scale i (c+1) = denominator p l a scale i c +
      ∑ j : Fin p.BN, weight p l a scale i (c*p.BN+j.val) := by
  unfold denominator
  rw [show (c+1)*p.BN = c*p.BN+p.BN by ring,Fin.sum_univ_add]
  rfl

theorem numerator_succ (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i d c : Nat) :
    numerator p l a scale i d (c+1) = numerator p l a scale i d c +
      ∑ j : Fin p.BN, weight p l a scale i (c*p.BN+j.val)*value p l a (c*p.BN+j.val) d := by
  unfold numerator
  rw [show (c+1)*p.BN = c*p.BN+p.BN by ring,Fin.sum_univ_add]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnNopad.Formula
