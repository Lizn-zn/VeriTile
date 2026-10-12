import bench.tritonbench_g.context_attn_mistral.Proofs.EpilogueResult
import VeriTile.Triton.Memory.TraceProofs
namespace VeriTile.Bench.TritonBenchG.ContextAttnMistral.AddressBounds
open VeriTile.Triton Formula Program
noncomputable section

def inside (a : Int) (extent : Nat) : Prop := 0 ≤ a ∧ a < (extent : Int)
theorem inside_toNat {a : Int} {n : Nat} (h : inside a n) : a.toNat < n := by
  unfold inside at h
  omega

def queryAddress (p : Layout) (l : Launch) (a : Inputs) (i d : Nat) : Int :=
  (a.start l.batch+queryRow p l i)*p.qRow+l.head*p.qHead+d*p.qDim

def keyAddress (p : Layout) (l : Launch) (a : Inputs) (j d : Nat) : Int :=
  ((a.start l.batch+j)*p.kRow)+(l.head/p.kvGroup)*p.kHead+d*p.kDim

def valueAddress (p : Layout) (l : Launch) (a : Inputs) (j d : Nat) : Int :=
  ((a.start l.batch+j)*p.vRow)+(l.head/p.kvGroup)*p.vHead+d*p.vDim

def outputAddress (p : Layout) (l : Launch) (a : Inputs) (i d : Nat) : Int :=
  (a.start l.batch+queryRow p l i)*p.outRow+l.head*p.outHead+d*p.outDim

/-- Bounds cover the metadata and only active source loads/stores. All channel
lanes are used; negative active query/output addresses are excluded before toNat. -/
structure Valid (p : Layout) (r : Regions) (l : Launch) (a : Inputs) (extent : RegionBounds) : Prop where
  advance : 0 ≤ a.start l.batch
  start : l.batch < extent r.start
  sequence : l.batch < extent r.sequence
  query : ∀ (i : Fin p.BM) (d : Fin p.D), (queryRow p l i.val : Int) < Formula.sequence p l a →
    inside (queryAddress p l a i.val d.val) (extent r.query)
  key : ∀ c, c < count p l a → ∀ (j : Fin p.BN) (d : Fin p.D),
    ((c*p.BN+j.val : Nat) : Int) < Formula.sequence p l a →
    inside (keyAddress p l a (c*p.BN+j.val) d.val) (extent r.key)
  value : ∀ c, c < count p l a → ∀ (j : Fin p.BN) (d : Fin p.D),
    ((c*p.BN+j.val : Nat) : Int) < Formula.sequence p l a →
    inside (valueAddress p l a (c*p.BN+j.val) d.val) (extent r.value)
  output : ∀ (i : Fin p.BM) (d : Fin p.D), (queryRow p l i.val : Int) < Formula.sequence p l a →
    inside (outputAddress p l a i.val d.val) (extent r.out)

end
end VeriTile.Bench.TritonBenchG.ContextAttnMistral.AddressBounds
