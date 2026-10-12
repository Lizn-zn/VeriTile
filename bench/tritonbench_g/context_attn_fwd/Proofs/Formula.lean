import bench.tritonbench_g.context_attn_bloom.Proofs.Math
import VeriTile.Meta.StatementAudit

/-! Independent initial-array formula for context_attn_fwd. The source uses
base-2 weights and a finite causal sentinel; padded values are zero but their
finite scores still contribute to the denominator. -/
namespace VeriTile.Bench.TritonBenchG.ContextAttnFwd.Formula
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnBloom.Math
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 1000000

structure Layout where
  BM : Nat
  BN : Nat
  D : Nat
  heads : Nat
  kvGroup : Nat
  qRow : Nat
  qHead : Nat
  qDim : Nat
  kBatch : Nat
  kHead : Nat
  kRow : Nat
  kDim : Nat
  vBatch : Nat
  vHead : Nat
  vRow : Nat
  vDim : Nat
  outRow : Nat
  outHead : Nat
  outDim : Nat

structure Launch where
  block : Nat
  batchHead : Nat

structure Inputs where
  q : Nat → ℝ
  k : Nat → ℝ
  v : Nat → ℝ
  start : Nat → Int
  sequence : Nat → Int
  prompt : Nat → Int

def batch (p : Layout) (l : Launch) := l.batchHead / p.heads
def head (p : Layout) (l : Launch) := l.batchHead % p.heads
def queryRow (p : Layout) (l : Launch) (i : Nat) := l.block*p.BM+i
def sequence (p : Layout) (l : Launch) (a : Inputs) : Int :=
  a.sequence (batch p l)-a.prompt (batch p l)
def limit (p : Layout) (l : Launch) (a : Inputs) : Int :=
  min ((l.block+1)*p.BM+a.prompt (batch p l)) (a.sequence (batch p l))
def count (p : Layout) (l : Launch) (a : Inputs) : Nat :=
  if (l.block*p.BM : Int) < sequence p l a then
    (limit p l a).toNat / p.BN + if (limit p l a).toNat % p.BN = 0 then 0 else 1
  else 0

def query (p : Layout) (l : Launch) (a : Inputs) (i d : Nat) : ℝ :=
  if (queryRow p l i : Int) < sequence p l a then
    a.q ((a.start (batch p l)+queryRow p l i)*p.qRow+head p l*p.qHead+d*p.qDim).toNat
  else 0

def key (p : Layout) (l : Launch) (a : Inputs) (j d : Nat) : ℝ :=
  if (j : Int) < limit p l a then
    a.k (batch p l*p.kBatch+j*p.kRow+(head p l/p.kvGroup)*p.kHead+d*p.kDim)
  else 0

def value (p : Layout) (l : Launch) (a : Inputs) (j d : Nat) : ℝ :=
  if (j : Int) < limit p l a then
    a.v (batch p l*p.vBatch+j*p.vRow+(head p l/p.kvGroup)*p.vHead+d*p.vDim)
  else 0

def score (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i j : Nat) : ℝ :=
  if (j : Int) ≤ queryRow p l i+a.prompt (batch p l) then
    (∑ e : Fin p.D, query p l a i e.val * key p l a j e.val) * scale
  else -100000000

def denominator (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i c : Nat) : ℝ :=
  ∑ j : Fin (count p l a*p.BN), if j.val < c*p.BN then pow2 (score p l a scale i j.val) else 0

def numerator (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i d c : Nat) : ℝ :=
  ∑ j : Fin (count p l a*p.BN),
    if j.val < c*p.BN then pow2 (score p l a scale i j.val)*value p l a j.val d else 0

/-- Closed initial-array weighted sum, including finite future and padded scores. -/
def expected (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i d : Nat) : ℝ :=
  numerator p l a scale i d (count p l a) / denominator p l a scale i (count p l a)

def pairs (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i d : Nat) :
    Fin (count p l a*p.BN) → ℝ × ℝ :=
  fun j => (score p l a scale i j.val,value p l a j.val d)

theorem normalized_state (S hi : Nat) (g : Fin S → ℝ × ℝ) :
    (gStateBot S hi g).2.2 / (gStateBot S hi g).2.1 =
      ((gKeysUpto S hi g).map (fun x => pow2 x.1*x.2)).sum /
      ((gKeysUpto S hi g).map (fun x => pow2 x.1)).sum := by
  obtain ⟨hden,hnum⟩ := gStateBot_consistent S hi g
  rw [hden,hnum]
  cases hm : (gStateBot S hi g).1 with
  | bot =>
    have hn := gKeysUpto_map_sum_eq_zero_of_bot S hi g hm (fun x => pow2 x.1*x.2)
    simp [hn]
  | coe m =>
    exact mul_div_mul_left _ _ (ne_of_gt (pow2_pos _))

theorem formula_agrees (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i d : Nat) :
    let st := gStateBot (count p l a*p.BN) (count p l a*p.BN) (pairs p l a scale i d)
    st.2.2 / st.2.1 = expected p l a scale i d := by
  dsimp only
  rw [normalized_state]
  unfold gKeysUpto
  rw [g_filterMap_finRange_sum,g_filterMap_finRange_sum]
  rfl

theorem count_positive_of_active (p : Layout) (l : Launch) (a : Inputs)
    (i : Fin p.BM)
    (hp : 0 ≤ a.prompt (batch p l))
    (ha : (queryRow p l i.val : Int) < sequence p l a) :
    0 < count p l a := by
  have hBM : 0 < p.BM := Nat.pos_of_ne_zero (by intro hz; simpa [hz] using i.isLt)
  have hstart : (l.block*p.BM : Int) < sequence p l a := by
    unfold queryRow at ha
    exact lt_of_le_of_lt (by omega) ha
  have hlim : 0 < limit p l a := by
    unfold limit
    apply lt_min
    · have hnext : (0 : Int) < (l.block+1)*p.BM := by positivity
      omega
    · unfold sequence at hstart
      have hz : (0 : Int) ≤ l.block*p.BM := by positivity
      omega
  have hn : 0 < (limit p l a).toNat := by omega
  rw [count,if_pos hstart]
  split_ifs with hmod
  · have hdiv := Nat.div_add_mod (limit p l a).toNat p.BN
    have hq : (limit p l a).toNat / p.BN ≠ 0 := by
      intro hz
      simp [hz,hmod] at hdiv
      omega
    simpa only [Nat.add_zero] using Nat.pos_of_ne_zero hq
  · exact Nat.zero_lt_succ _

theorem denominator_positive (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i c : Nat)
    (hBN : 0 < p.BN) (hc : 0 < c) (hcount : c ≤ count p l a) :
    0 < denominator p l a scale i c := by
  have hs : 0 < count p l a*p.BN := Nat.mul_pos (by omega) hBN
  have hz : 0 < c*p.BN := Nat.mul_pos hc hBN
  unfold denominator
  apply Finset.sum_pos'
  · intro j _
    split_ifs
    · exact le_of_lt (pow2_pos _)
    · exact le_rfl
  · refine ⟨⟨0,hs⟩,Finset.mem_univ _,?_⟩
    simp only [hz,↓reduceIte]
    exact pow2_pos _

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnFwd.Formula
