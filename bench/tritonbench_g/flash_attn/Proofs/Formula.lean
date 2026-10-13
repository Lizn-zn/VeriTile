import bench.tritonbench_g.flash_attn.Proofs.Geometry
import bench.tritonbench_g.flash_attn.Proofs.RealMath
/-! Independent formulas over initial input memory, using the source's exact
address expressions, decimal scale and full final KV tile. No executed state
or online recurrence occurs in these definitions. -/
namespace VeriTile.Bench.TritonBenchG.FlashAttn.Formula
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 2000000

/-- The strides actually used by the source; its remaining stride parameters
are unused. In particular V's sequence stride is `stride_k_seqlen`. -/
structure Layout where
  stride_q_head : Nat
  stride_q_seqlen : Nat
  stride_q_dim : Nat
  stride_k_seqlen : Nat
  stride_k_dim : Nat
  stride_v_dim : Nat
  stride_o_seqlen : Nat
  stride_o_dim : Nat
  SEQLEN : Nat
  BLOCK_M : Nat
  DIM : Nat
  BLOCK_N : Nat
  causal : Bool
  sm_scale : ℝ

def base (p : Layout) (pid₁ : Nat) : Nat := pid₁*p.stride_q_head
def row (p : Layout) (pid₀ i : Nat) : Nat := pid₀*p.BLOCK_M+i
def qOffset (p : Layout) (pid₀ pid₁ i e : Nat) : Nat :=
  base p pid₁+row p pid₀ i*p.stride_q_seqlen+e*p.stride_q_dim
def kOffset (p : Layout) (pid₁ k e : Nat) : Nat :=
  base p pid₁+e*p.stride_k_dim+k*p.stride_k_seqlen
def vOffset (p : Layout) (pid₁ k d : Nat) : Nat :=
  base p pid₁+k*p.stride_k_seqlen+d*p.stride_v_dim
def oOffset (p : Layout) (pid₀ pid₁ i d : Nat) : Nat :=
  base p pid₁+row p pid₀ i*p.stride_o_seqlen+d*p.stride_o_dim
def lOffset (p : Layout) (pid₀ pid₁ i : Nat) : Nat := pid₁*p.SEQLEN+row p pid₀ i

def hi (p : Layout) (pid₀ : Nat) : Nat := Geometry.stop p.causal p.SEQLEN p.BLOCK_M pid₀
def keyCount (p : Layout) (pid₀ : Nat) : Nat := Geometry.visited (hi p pid₀) p.BLOCK_N

def keys (p : Layout) (pid₀ i : Nat) : List Nat :=
  (List.range (keyCount p pid₀)).filter (fun k => !p.causal || decide (k≤row p pid₀ i))

def score (p : Layout) (read : RegionName → Nat → ℝ) (Q K : RegionName)
    (pid₀ pid₁ i k : Nat) : ℝ :=
  Finset.univ.sum (fun e : Fin p.DIM =>
    (read Q (qOffset p pid₀ pid₁ i e.val)*(p.sm_scale*1.44269504)) *
      read K (kOffset p pid₁ k e.val))

def denominator (p : Layout) (read : RegionName → Nat → ℝ) (Q K : RegionName)
    (pid₀ pid₁ i : Nat) : ℝ :=
  ((keys p pid₀ i).map (fun k => pow2 (score p read Q K pid₀ pid₁ i k))).sum

def numerator (p : Layout) (read : RegionName → Nat → ℝ) (Q K V : RegionName)
    (pid₀ pid₁ i d : Nat) : ℝ :=
  ((keys p pid₀ i).map (fun k => pow2 (score p read Q K pid₀ pid₁ i k)*
    read V (vOffset p pid₁ k d))).sum

def output (p : Layout) (read : RegionName → Nat → ℝ) (Q K V : RegionName)
    (pid₀ pid₁ i d : Nat) : ℝ :=
  numerator p read Q K V pid₀ pid₁ i d / denominator p read Q K pid₀ pid₁ i

def lse (p : Layout) (read : RegionName → Nat → ℝ) (Q K : RegionName)
    (pid₀ pid₁ i : Nat) : ℝ := Real.log (denominator p read Q K pid₀ pid₁ i)/Real.log 2

/-- Every loaded key is included in the read footprint, even if the causal
predicate later replaces its score by negative infinity. -/
def readWindow (p : Layout) (Q K V : RegionName) (pid₀ pid₁ : Nat)
    (region : RegionName) (offset : Nat) : Prop :=
  (region=Q ∧ ∃ i : Fin p.BLOCK_M, ∃ e : Fin p.DIM, offset=qOffset p pid₀ pid₁ i.val e.val) ∨
  (region=K ∧ ∃ k : Fin (keyCount p pid₀), ∃ e : Fin p.DIM, offset=kOffset p pid₁ k.val e.val) ∨
  (region=V ∧ ∃ k : Fin (keyCount p pid₀), ∃ d : Fin p.DIM, offset=vOffset p pid₁ k.val d.val)

def writeWindow (p : Layout) (L O : RegionName) (pid₀ pid₁ : Nat)
    (region : RegionName) (offset : Nat) : Prop :=
  (region=L ∧ ∃ i : Fin p.BLOCK_M, offset=lOffset p pid₀ pid₁ i.val) ∨
  (region=O ∧ ∃ i : Fin p.BLOCK_M, ∃ d : Fin p.DIM, offset=oOffset p pid₀ pid₁ i.val d.val)

/-- The mathematical row as a list of score/value pairs, used only to connect
the independent sums to the generic online-softmax algebra. -/
def pairs (p : Layout) (read : RegionName → Nat → ℝ) (Q K V : RegionName)
    (pid₀ pid₁ i d : Nat) : List (ℝ × ℝ) :=
  (keys p pid₀ i).map (fun k => (score p read Q K pid₀ pid₁ i k,read V (vOffset p pid₁ k d)))

theorem zero_mem_keys (p : Layout) (pid₀ i : Nat) (hhi : 0<hi p pid₀) (hBN : 0<p.BLOCK_N) :
    0∈keys p pid₀ i := by
  have h := Geometry.visited_positive (hi p pid₀) p.BLOCK_N hhi hBN
  simp [keys,keyCount,List.mem_filter,List.mem_range,h]

theorem pairs_nonempty (p : Layout) (read : RegionName → Nat → ℝ) (Q K V : RegionName)
    (pid₀ pid₁ i d : Nat) (hhi : 0<hi p pid₀) (hBN : 0<p.BLOCK_N) :
    pairs p read Q K V pid₀ pid₁ i d≠[] := by
  intro h
  have hm := List.mem_map_of_mem (f:=fun k => (score p read Q K pid₀ pid₁ i k,read V (vOffset p pid₁ k d)))
    (zero_mem_keys p pid₀ i hhi hBN)
  change _∈pairs p read Q K V pid₀ pid₁ i d at hm
  rw [h] at hm
  exact List.not_mem_nil hm

theorem denominator_positive (p : Layout) (read : RegionName → Nat → ℝ) (Q K : RegionName)
    (pid₀ pid₁ i : Nat) (hhi : 0<hi p pid₀) (hBN : 0<p.BLOCK_N) :
    0<denominator p read Q K pid₀ pid₁ i := by
  simpa only [RealMath.denominator,pairs,List.map_map,Function.comp_apply,denominator] using
    RealMath.denominator_positive (pairs p read Q K Q pid₀ pid₁ i 0)
      (pairs_nonempty p read Q K Q pid₀ pid₁ i 0 hhi hBN)

theorem online_output (p : Layout) (read : RegionName → Nat → ℝ) (Q K V : RegionName)
    (pid₀ pid₁ i d : Nat) (hhi : 0<hi p pid₀) (hBN : 0<p.BLOCK_N) :
    let st := RealMath.running (pairs p read Q K V pid₀ pid₁ i d)
    st.2.2/st.2.1=output p read Q K V pid₀ pid₁ i d := by
  simpa only [output,numerator,denominator,RealMath.denominator,RealMath.numerator,pairs,List.map_map,Function.comp_apply] using
    RealMath.ratio_eq_reference (pairs p read Q K V pid₀ pid₁ i d)
      (pairs_nonempty p read Q K V pid₀ pid₁ i d hhi hBN)

theorem online_lse (p : Layout) (read : RegionName → Nat → ℝ) (Q K V : RegionName)
    (pid₀ pid₁ i d : Nat) (hhi : 0<hi p pid₀) (hBN : 0<p.BLOCK_N) :
    let xs := pairs p read Q K V pid₀ pid₁ i d
    (RealMath.maximum xs).unbotD 0+Real.log (RealMath.running xs).2.1/Real.log 2 =
      lse p read Q K pid₀ pid₁ i := by
  simpa only [lse,denominator,RealMath.denominator,pairs,List.map_map,Function.comp_apply] using
    RealMath.logsumexp_eq_reference (pairs p read Q K V pid₀ pid₁ i d)
      (pairs_nonempty p read Q K V pid₀ pid₁ i d hhi hBN)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FlashAttn.Formula
