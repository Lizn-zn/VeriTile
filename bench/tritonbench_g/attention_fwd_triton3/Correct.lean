import bench.tritonbench_g.attention_fwd_triton3.Kernels
import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealContract
import VeriTile.Meta.StatementAudit

/-! Real correctness of the complete original and reciprocal candidate.
Q/K/V and the optional M/L/Out carry are mathematical input arrays. The result
is a maximum and base-2 weighted sums over keys selected by the signed window.
The six windows below supply allocation bounds and input values to a contract
that proves successful flat execution, all three outputs and the memory frame.

Dimensions, independent strides, grouped heads, INIT/END, signed windows and
checked partial KV tails remain symbolic. M/Out stores cover all block rows,
including padding; L is written only for continuation. This is a per-program
real-semantics claim, with mathematical integer addresses. Host launch validity,
fixed-width integer overflow and IEEE NaN behavior are outside this statement.
-/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.Correctness
open VeriTile.Triton
open scoped BigOperators VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealIO
noncomputable section

/-- Source parameters: query/KV head counts; independent batch/head and
row/inner strides; Q/K/V/M/L/Out regions and context lengths; scale; INIT,
even-row flags and signed window selection. Every use is visible below. -/
abbrev Parameters := RealPrologue.Parameters
abbrev Inputs := RealInputs.Inputs

/-- Q is boundary checked unless IS_EVEN_M is set. K/V cover the global
context. Old carry values are read only for INIT=0. M/Out are always written;
L is written only for END=0. The same six allocations supply in-place carry. -/
def originalIO (BM BN BD Z finish : Nat) (p : Parameters) : RealIO.IO BM BD p.layout.kvContext where
  kernel := (TypedSource.original p.layout.Q p.layout.K p.layout.V p.layout.M p.layout.Out p.layout.L p.scale
    p.strides.qBatch p.strides.qHead p.layout.qRow p.layout.qDim
    p.strides.kBatch p.strides.kHead p.layout.kRow p.layout.kDim
    p.strides.vBatch p.strides.vHead p.layout.vRow p.layout.vDim
    p.strides.oBatch p.strides.oHead p.layout.oRow p.layout.oDim
    Z p.heads p.kvHeads p.layout.queryContext p.layout.roundedContext p.layout.kvContext
    p.evenM p.evenN BM BD BN finish p.init p.window p.complement p.offset p.size).eraseDType
  projection := RealSource.projection BM BN BD Z finish p Bool.false
  Q := p.layout.Q
  K := p.layout.K
  V := p.layout.V
  M := p.layout.M
  L := p.layout.L
  Out := p.layout.Out
  queryWindow := fun p₀ p₁ _ i =>
    p₁ / p.heads * p.strides.qBatch + p₁ % p.heads * p.strides.qHead +
      (p₀ * BM + i.1.val) * p.layout.qRow + i.2.1.val * p.layout.qDim
  keyWindow := fun _ p₁ _ i =>
    p₁ / p.heads * p.strides.kBatch + ((p₁ % p.heads) / (p.heads / p.kvHeads)) * p.strides.kHead +
      i.2.1.val * p.layout.kDim + i.1.val * p.layout.kRow
  valueWindow := fun _ p₁ _ i =>
    p₁ / p.heads * p.strides.vBatch + ((p₁ % p.heads) / (p.heads / p.kvHeads)) * p.strides.vHead +
      i.1.val * p.layout.vRow + i.2.1.val * p.layout.vDim
  centerWindow := fun p₀ p₁ _ i => p₁ * p.layout.roundedContext + p₀ * BM + i.1.val
  denominatorWindow := fun p₀ p₁ _ i => p₁ * p.layout.roundedContext + p₀ * BM + i.1.val
  outputWindow := fun p₀ p₁ _ i =>
    p₁ / p.heads * p.strides.oBatch + p₁ % p.heads * p.strides.oHead +
      (p₀ * BM + i.1.val) * p.layout.oRow + i.2.1.val * p.layout.oDim
  queryMask := fun p₀ _ _ i => p.evenM ≠ 0 ∨ p₀ * BM + i.1.val < p.layout.queryContext
  readCarry := p.init = 0
  writeDenominator := finish = 0


/-- The reciprocal candidate has exactly the same windows and masks. -/
def reciprocalIO (BM BN BD Z finish : Nat) (p : Parameters) : RealIO.IO BM BD p.layout.kvContext :=
  { originalIO BM BN BD Z finish p with
    kernel := (TypedSource.reciprocal p.layout.Q p.layout.K p.layout.V p.layout.M p.layout.Out p.layout.L p.scale
    p.strides.qBatch p.strides.qHead p.layout.qRow p.layout.qDim
    p.strides.kBatch p.strides.kHead p.layout.kRow p.layout.kDim
    p.strides.vBatch p.strides.vHead p.layout.vRow p.layout.vDim
    p.strides.oBatch p.strides.oHead p.layout.oRow p.layout.oDim
    Z p.heads p.kvHeads p.layout.queryContext p.layout.roundedContext p.layout.kvContext
    p.evenM p.evenN BM BD BN finish p.init p.window p.complement p.offset p.size).eraseDType
    projection := RealSource.projection BM BN BD Z finish p Bool.true }

/-- The exact decimal is the source's log2 scale conversion, retained in the
real projection. Inactive checked Q rows contribute zero. -/
def score (BM BD start : Nat) (p : Parameters) (xs : Inputs BM BD p.layout.kvContext)
    (r : Fin BM) (k : Fin p.layout.kvContext) : ℝ :=
  (∑ d : Fin BD,
    (if p.evenM ≠ 0 ∨ start * BM + r.val < p.layout.queryContext then xs.query r d else 0) *
      xs.key k d) * (p.scale * 1.4426950408889634)

def initial (BM BD : Nat) (p : Parameters) (xs : Inputs BM BD p.layout.kvContext) : RealLoopFormula.Carry BM BD where
  center := fun r => if p.init ≠ 0 then ⊥ else some (xs.center r)
  denominator := fun r => if p.init ≠ 0 then 1 else xs.denominator r
  numerator := fun i => if p.init ≠ 0 then 0 else xs.output i.1 i.2.1

/-- For selected (score,value) pairs, m is their maximum together with the
initial center. The denominator and numerator are rescaled initial carry plus
Σ 2^(score-m) and Σ 2^(score-m)*value. Bottom centers have weight zero. -/
def summarize (seed : RealFormula.RowState) (pairs : List (ℝ × ℝ)) : RealFormula.RowState :=
  let m := seed.center ⊔ RealFormula.maxScore pairs
  { center := m,
    denominator := seed.denominator * RealFormula.weight seed.center m + RealFormula.sumAt pairs m (fun _ => 1),
    numerator := seed.numerator * RealFormula.weight seed.center m + RealFormula.sumAt pairs m Prod.snd }

/-- Regular windows keep 0 ≤ query-key+offset < size; the complement arm
keeps query-key+offset ≥ size. No loop iterations preserve the initial state,
which differs from executing a block whose row has no kept keys. -/
def row (BM BN BD start : Nat) (p : Parameters) (xs : Inputs BM BD p.layout.kvContext)
    (r : Fin BM) (initialValue : ℝ) (v : Fin p.layout.kvContext → ℝ) : RealFormula.RowState :=
  let seed := RealLoopFormula.rowSeed (initial BM BD p xs) r initialValue
  if (RealKeys.lower start BM BN p.window p.complement p.offset p.size : Int) <
      RealWindow.upper start BM BN p.layout.kvContext p.window p.complement p.offset then
    summarize seed
      (((List.finRange p.layout.kvContext).filter
        (fun k => RealKeys.keep start BM r.val p.window p.complement p.offset p.size k.val)).map
        (fun k => (score BM BD start p xs r k, v k)))
  else seed

def carry (BM BN BD start : Nat) (p : Parameters) (xs : Inputs BM BD p.layout.kvContext) : RealLoopFormula.Carry BM BD where
  center := fun r => (row BM BN BD start p xs r 0 (fun _ => 0)).center
  denominator := fun r => (row BM BN BD start p xs r 0 (fun _ => 0)).denominator
  numerator := fun i => (row BM BN BD start p xs i.1
    ((initial BM BD p xs).numerator i) (fun k => xs.value k i.2.1)).numerator

/-- END stores M+log2(L) and Out/L, preserving L memory. Continuation stores
all three carry values. Real division/log are total; bottom stored M reads as
zero under the framework's finite real-memory view. -/
def expected (BM BN BD finish start : Nat) (p : Parameters)
    (xs : Inputs BM BD p.layout.kvContext) : RealIO.Outputs BM BD :=
  let c := carry BM BN BD start p xs
  { center := fun r => (c.center r).elim 0 (fun m =>
      if decide (finish ≠ 0) then m + Real.log (c.denominator r) / Real.log 2 else m),
    denominator := c.denominator,
    output := fun i => if finish ≠ 0 then c.numerator i / c.denominator i.1 else c.numerator i }

/-- Unchecked KV loads require whole key blocks. Positive inner stride and
sufficient row stride prevent output overlap; M/L/Out are distinct regions.
Active address bounds are supplied by the displayed IO windows. -/
specification attention_correctness (BM BN BD Z finish : Nat) (p : Parameters)
    (hN : 0 < BN) (hflag : p.evenN ≠ 0 → p.layout.kvContext % BN = 0)
    (hd : 0 < p.layout.oDim) (hr : BD * p.layout.oDim ≤ p.layout.oRow)
    (hm : p.layout.M ≠ p.layout.Out) (hlm : p.layout.L ≠ p.layout.M) (hlo : p.layout.L ≠ p.layout.Out) :
    Spec.Real (originalIO BM BN BD Z finish p ⊨ fun start _ xs => expected BM BN BD finish start p xs) := by
  exact RealContract.implements BM BN BD Z finish p Bool.false hN hflag hd hr hm hlm hlo

specification attention_reciprocal_correctness (BM BN BD Z finish : Nat) (p : Parameters)
    (hN : 0 < BN) (hflag : p.evenN ≠ 0 → p.layout.kvContext % BN = 0)
    (hd : 0 < p.layout.oDim) (hr : BD * p.layout.oDim ≤ p.layout.oRow)
    (hm : p.layout.M ≠ p.layout.Out) (hlm : p.layout.L ≠ p.layout.M) (hlo : p.layout.L ≠ p.layout.Out) :
    Spec.Real (reciprocalIO BM BN BD Z finish p ⊨ fun start _ xs => expected BM BN BD finish start p xs) := by
  exact RealContract.implements BM BN BD Z finish p Bool.true hN hflag hd hr hm hlm hlo

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.Correctness
