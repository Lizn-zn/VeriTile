import bench.tritonbench_g.context_attn_bloom.Proofs.EpilogueResult
import VeriTile.Triton.Memory.TraceProofs
namespace VeriTile.Bench.TritonBenchG.ContextAttnBloom.AddressBounds
open VeriTile.Triton Math Program RealSetup
noncomputable section

def inside (a : Int) (extent : Nat) : Prop := 0 ≤ a ∧ a < (extent : Int)
theorem inside_toNat {a : Int} {n : Nat} (h : inside a n) : a.toNat < n := by
  unfold inside at h
  omega

def queryAddress (p : Layout) (l : Launch) (a : Inputs) (i d : Nat) : Int :=
  (a.start l.batch+queryRow p l i)*p.qRow+l.head*p.qHead+d*p.qDim

def tokenAddress (p : Layout) (l : Launch) (a : Inputs) (j : Nat) : Int :=
  p.requestRow*a.request l.batch+p.requestToken*j

def keyAddress (p : Layout) (l : Launch) (a : Inputs) (j d : Nat) : Int :=
  token p l a j*p.kRow+(l.head/p.kvGroup)*p.kHead+d*p.kDim

def valueAddress (p : Layout) (l : Launch) (a : Inputs) (j d : Nat) : Int :=
  token p l a j*p.vRow+(l.head/p.kvGroup)*p.vHead+d*p.vDim

def outputAddress (p : Layout) (l : Launch) (a : Inputs) (i d : Nat) : Int :=
  (a.start l.batch+queryRow p l i)*p.outRow+l.head*p.outHead+d*p.outDim

/-- Bounds describe only addresses actually accessed by this program. Gathered
K/V addresses depend on the initial signed page table. Inactive lanes require
no allocation, and negative active addresses are excluded before conversion. -/
structure Valid (p : Layout) (r : Regions) (l : Launch) (a : Inputs) (extent : RegionBounds) : Prop where
  start : l.batch < extent r.start
  prompt : l.batch < extent r.prompt
  sequence : l.batch < extent r.sequence
  request : l.batch < extent r.request
  query : ∀ (i : Fin p.BM) (d : Fin p.D), (queryRow p l i.val : Int) < Math.sequence a l →
    d.val < p.headDim → inside (queryAddress p l a i.val d.val) (extent r.query)
  tokens : ∀ c, c < count p l a → ∀ j : Fin p.BN, ((c*p.BN+j.val : Nat) : Int) < Math.limit p l a →
    inside (tokenAddress p l a (c*p.BN+j.val)) (extent r.tokens)
  key : ∀ c, c < count p l a → ∀ (j : Fin p.BN) (d : Fin p.D),
    ((c*p.BN+j.val : Nat) : Int) < Math.limit p l a → d.val < p.headDim →
    inside (keyAddress p l a (c*p.BN+j.val) d.val) (extent r.key)
  value : ∀ c, c < count p l a → ∀ (j : Fin p.BN) (d : Fin p.D),
    ((c*p.BN+j.val : Nat) : Int) < Math.limit p l a → d.val < p.headDim →
    inside (valueAddress p l a (c*p.BN+j.val) d.val) (extent r.value)
  output : ∀ (i : Fin p.BM) (d : Fin p.D), (queryRow p l i.val : Int) < Math.sequence a l →
    d.val < p.headDim → inside (outputAddress p l a i.val d.val) (extent r.out)

end
end VeriTile.Bench.TritonBenchG.ContextAttnBloom.AddressBounds
