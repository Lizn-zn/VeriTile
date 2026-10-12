import VeriTile.Triton
import VeriTile.Meta.StatementAudit

/-! Independent real formulas for the complete H recurrence. Arrays refer to
initial memory, never to a materialized carry or a kernel execution result.
Time tails and checked-off endpoint reads below use zero; the eventual real
execution contract must require that behavior from omitted source padding. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealHMath
open scoped BigOperators
noncomputable section

structure Layout where
  keyHead : Nat
  keyTime : Nat
  keyFeature : Nat
  valueHead : Nat
  valueTime : Nat
  valueFeature : Nat
  historyHead : Nat
  historyKey : Nat
  historyValue : Nat
  T : Nat
  K : Nat
  V : Nat
  BT : Nat
  BK : Nat
  BV : Nat
  NT : Nat

structure Launch where
  value : Nat
  key : Nat
  head : Nat

structure Inputs where
  key : Nat → ℝ
  value : Nat → ℝ
  gate : Nat → ℝ
  seed : Nat → ℝ

def keyIndex (p : Layout) (l : Launch) (e : Nat) := l.key * p.BK + e
def valueIndex (p : Layout) (l : Launch) (j : Nat) := l.value * p.BV + j
def timeIndex (p : Layout) (c t : Nat) := c * p.BT + t

def keyAddress (p : Layout) (l : Launch) (k t : Nat) :=
  l.head * p.keyHead + k * p.keyFeature + t * p.keyTime

def valueAddress (p : Layout) (l : Launch) (t v : Nat) :=
  l.head * p.valueHead + t * p.valueTime + v * p.valueFeature

def stateAddress (p : Layout) (l : Launch) (k v : Nat) :=
  l.head * p.K * p.V + k * p.V + v

def historyAddress (p : Layout) (l : Launch) (c k v : Nat) :=
  l.head * p.historyHead + c * p.K * p.V + k * p.historyKey + v * p.historyValue

/-- Full chunk end, not the last valid time. The one-dimensional parent is
T*K or T*V, with the source's feature stride, including on the final chunk. -/
def endpointIndex (p : Layout) (gateK : Bool) (c index : Nat) :=
  (c * p.BT + p.BT - 1) * (if gateK then p.K else p.V) + index

def endpoint (p : Layout) (l : Launch) (a : Inputs) (gateK : Bool)
    (c k v : Nat) : ℝ :=
  let n := endpointIndex p gateK c (if gateK then k else v)
  if n < p.T * (if gateK then p.K else p.V) then
    a.gate (l.head * (if gateK then p.keyHead else p.valueHead) +
      n * (if gateK then p.keyFeature else p.valueFeature))
  else 0

def keyValue (p : Layout) (l : Launch) (a : Inputs) (k t : Nat) : ℝ :=
  if k < p.K ∧ t < p.T then a.key (keyAddress p l k t) else 0

def valueValue (p : Layout) (l : Launch) (a : Inputs) (t v : Nat) : ℝ :=
  if t < p.T ∧ v < p.V then a.value (valueAddress p l t v) else 0

def gateValue (p : Layout) (l : Launch) (a : Inputs) (gateK : Bool)
    (k t v : Nat) : ℝ :=
  if gateK then
    if k < p.K ∧ t < p.T then a.gate (keyAddress p l k t) else 0
  else if t < p.T ∧ v < p.V then a.gate (valueAddress p l t v) else 0

def seed (p : Layout) (l : Launch) (a : Inputs) (hasSeed : Bool) (k v : Nat) : ℝ :=
  if hasSeed ∧ k < p.K ∧ v < p.V then a.seed (stateAddress p l k v) else 0

def gateFactor (p : Layout) (l : Launch) (a : Inputs) (gateK : Bool)
    (c k v : Nat) : ℝ := Real.exp (endpoint p l a gateK c k v)

/-- One independently defined gated K/V contraction. This includes the whole
BT span with explicit zero extension; precision is erased only in this layer. -/
def contribution (p : Layout) (l : Launch) (a : Inputs) (gateK : Bool)
    (c k v : Nat) : ℝ :=
  ∑ lane : Fin p.BT,
    let t := timeIndex p c lane.val
    let weight := Real.exp (endpoint p l a gateK c k v - gateValue p l a gateK k t v)
    if gateK then (keyValue p l a k t * weight) * valueValue p l a t v
    else keyValue p l a k t * (valueValue p l a t v * weight)

/-- State entering chunk c; final state uses c=NT. Each earlier contraction
is multiplied by the intervening chunk gates, plus the gated initial seed. -/
def closed (p : Layout) (l : Launch) (a : Inputs) (gateK hasSeed : Bool)
    (c k v : Nat) : ℝ :=
  seed p l a hasSeed k v * (∏ u ∈ Finset.range c, gateFactor p l a gateK u k v) +
    ∑ t ∈ Finset.range c, contribution p l a gateK t k v *
      ∏ u ∈ Finset.Ico (t+1) c, gateFactor p l a gateK u k v

@[simp] theorem closed_zero (p : Layout) (l : Launch) (a : Inputs)
    (gateK hasSeed : Bool) (k v : Nat) :
    closed p l a gateK hasSeed 0 k v = seed p l a hasSeed k v := by simp [closed]

theorem closed_step (p : Layout) (l : Launch) (a : Inputs)
    (gateK hasSeed : Bool) (c k v : Nat) :
    closed p l a gateK hasSeed (c+1) k v =
      closed p l a gateK hasSeed c k v * gateFactor p l a gateK c k v +
        contribution p l a gateK c k v := by
  unfold closed
  rw [Finset.prod_range_succ, Finset.sum_range_succ]
  simp only [Finset.Ico_self, Finset.prod_empty, mul_one]
  have h : (∑ t ∈ Finset.range c, contribution p l a gateK t k v *
      ∏ u ∈ Finset.Ico (t+1) (c+1), gateFactor p l a gateK u k v) =
      (∑ t ∈ Finset.range c, contribution p l a gateK t k v *
      ∏ u ∈ Finset.Ico (t+1) c, gateFactor p l a gateK u k v) * gateFactor p l a gateK c k v := by
    rw [Finset.sum_mul]
    apply Finset.sum_congr rfl
    intro t ht
    rw [Finset.prod_Ico_succ_top (by simpa only [Finset.mem_range, Nat.succ_le_iff] using ht)]
    ring
  rw [h]
  ring

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealHMath
