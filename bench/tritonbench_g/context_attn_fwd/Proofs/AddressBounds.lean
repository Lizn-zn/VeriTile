import bench.tritonbench_g.context_attn_fwd.Proofs.EpilogueResult
import VeriTile.Triton.Memory.TraceProofs
namespace VeriTile.Bench.TritonBenchG.ContextAttnFwd.AddressBounds
open VeriTile.Triton Formula Program
noncomputable section

def inside (a : Int) (extent : Nat) : Prop := 0 ≤ a ∧ a < (extent : Int)
theorem inside_toNat {a : Int} {n : Nat} (h : inside a n) : a.toNat < n := by
  unfold inside at h
  omega

def queryAddress (p : Layout) (l : Launch) (a : Inputs) (i d : Nat) : Int :=
  (a.start (Formula.batch p l)+queryRow p l i)*p.qRow+(Formula.head p l)*p.qHead+d*p.qDim

def keyAddress (p : Layout) (l : Launch) (a : Inputs) (j d : Nat) : Int :=
  (Formula.batch p l*p.kBatch+j*p.kRow)+((Formula.head p l)/p.kvGroup)*p.kHead+d*p.kDim

def valueAddress (p : Layout) (l : Launch) (a : Inputs) (j d : Nat) : Int :=
  (Formula.batch p l*p.vBatch+j*p.vRow)+((Formula.head p l)/p.kvGroup)*p.vHead+d*p.vDim

def outputAddress (p : Layout) (l : Launch) (a : Inputs) (i d : Nat) : Int :=
  (a.start (Formula.batch p l)+queryRow p l i)*p.outRow+(Formula.head p l)*p.outHead+d*p.outDim

/-- Bounds cover the metadata and only active source loads/stores. All channel
lanes are used; negative active query/output addresses are excluded before toNat. -/
structure Valid (p : Layout) (r : Regions) (l : Launch) (a : Inputs) (extent : RegionBounds) : Prop where
  start : Formula.batch p l < extent r.start
  prompt : Formula.batch p l < extent r.prompt
  sequence : Formula.batch p l < extent r.sequence
  query : ∀ (i : Fin p.BM) (d : Fin p.D), (queryRow p l i.val : Int) < Formula.sequence p l a →
    inside (queryAddress p l a i.val d.val) (extent r.query)
  key : ∀ c, c < count p l a → ∀ (j : Fin p.BN) (d : Fin p.D),
    ((c*p.BN+j.val : Nat) : Int) < Formula.limit p l a →
    inside (keyAddress p l a (c*p.BN+j.val) d.val) (extent r.key)
  value : ∀ c, c < count p l a → ∀ (j : Fin p.BN) (d : Fin p.D),
    ((c*p.BN+j.val : Nat) : Int) < Formula.limit p l a →
    inside (valueAddress p l a (c*p.BN+j.val) d.val) (extent r.value)
  output : ∀ (i : Fin p.BM) (d : Fin p.D), (queryRow p l i.val : Int) < Formula.sequence p l a →
    inside (outputAddress p l a i.val d.val) (extent r.out)

end
end VeriTile.Bench.TritonBenchG.ContextAttnFwd.AddressBounds
