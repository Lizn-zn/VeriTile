import bench.tritonbench_g.attn_fwd_causal.Kernels
import bench.tritonbench_g.attn_fwd_causal.Proofs.RealContract
import VeriTile.Meta.StatementAudit

/-! Real correctness of both complete staged-attention sources in Kernels.
STAGE=3 visits the prefix then the diagonal; STAGE=2 visits the diagonal twice.
Future diagonal scores receive a finite -1e6 penalty, so their real weights
remain positive. Masked K/V inputs are zero; their weights still occur in the
denominator. This specification deliberately retains those source behaviors.

The IO contract includes successful flat-memory execution and preservation of
every cell outside active output writes. It requires disjoint allocations,
bounds for actual reads/writes, pinned input values, zero masked-undef values
and distinct addresses for active output lanes. Shapes and strides are
symbolic. Host launch composition and fixed-width integer overflow are outside
this per-program mathematical statement. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.Correctness
open VeriTile.Triton
open scoped VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealIO
noncomputable section

/-- Named Q/K/V/QS/KS/Out buffers, all source strides, batch/head/context
sizes and the legal outer stage. Source-unused strides remain present. -/
abbrev Parameters := RealContract.Parameters
abbrev Inputs := RealFormula.Inputs
abbrev Visit := RealFormula.Visit

/-- The contract uses these exact source addresses for tile origin `start`:
Q/Out = base+(startM*BM+row)*queryRow+d*queryDim;
K = base+d+j*keyRow+headDim*start;
V = base+j*queryRow+d*queryDim+headDim*start.
Q/Out require row<N and d<activeHead; K/V require start+j<N and d<activeHead.
QS[queryScaleBase+startM] and KS[keyScaleBase+start/BN] are unmasked reads,
including key scales for fully padded tiles. Only actual visits are pinned.
There is no scratch allocation. -/
def addresses (BM BN : Nat) (p : Parameters) (startM headBatch : Nat) : RealLoads.Config where
  Q := p.Q
  K := p.K
  V := p.V
  QS := p.QS
  KS := p.KS
  base := headBatch/p.heads*p.queryBatch+headBatch%p.heads*p.queryHead
  queryScaleBase := headBatch*((p.context+BM-1)/BM)
  keyScaleBase := headBatch*((p.context+BN-1)/BN)
  strideM := p.queryRow
  strideD := p.queryDim
  strideKN := p.keyRow
  headDim := p.headDim
  context := p.context
  activeHead := p.activeHead
  startM := startM

def originalIO (BM BN BD : Nat) (p : Parameters) : RealIO.IO BM BN BD where
  kernel := (TypedSource.original p.Q p.K p.V p.QS p.KS p.Out
    p.queryBatch p.queryHead p.queryRow p.queryDim p.keyBatch p.keyHead p.keyRow p.keyDim
    p.valueBatch p.valueHead p.valueRow p.valueDim p.outputBatch p.outputHead p.outputRow p.outputDim
    p.batches p.heads p.context p.headDim BM BN BD p.activeHead p.stage p.validStage).eraseDType
  projection := RealContract.projection BM BN BD p Bool.false
  configuration := addresses BM BN p
  out := p.Out
  stage := p.stage

def reciprocalIO (BM BN BD : Nat) (p : Parameters) : RealIO.IO BM BN BD :=
  { originalIO BM BN BD p with
    kernel := (TypedSource.reciprocal p.Q p.K p.V p.QS p.KS p.Out
      p.queryBatch p.queryHead p.queryRow p.queryDim p.keyBatch p.keyHead p.keyRow p.keyDim
      p.valueBatch p.valueHead p.valueRow p.valueDim p.outputBatch p.outputHead p.outputRow p.outputDim
      p.batches p.heads p.context p.headDim BM BN BD p.activeHead p.stage p.validStage).eraseDType
    projection := RealContract.projection BM BN BD p Bool.true }

def rangeVisits (lower upper BN : Nat) (diagonal : Bool) : List Visit :=
  (List.range ((upper-lower+BN-1)/BN)).map (fun j => ⟨lower+j*BN,diagonal⟩)

/-- Retain repeated visits and partial final tiles exactly as in the source. -/
def visits (BM BN startM : Nat) (p : Parameters) : List Visit :=
  (if p.stage = 2 then rangeVisits (startM*BM) ((startM+1)*BM) BN Bool.true
    else rangeVisits 0 (startM*BM) BN Bool.false) ++
    rangeVisits (startM*BM) ((startM+1)*BM) BN Bool.true

def score (BM BN BD startM : Nat) (p : Parameters) (xs : Inputs BM BN BD)
    (v : Visit) (row : Fin BM) (key : Fin BN) : ℝ :=
  (∑ d : Fin BD,
    (if startM*BM+row.val < p.context ∧ d.val < p.activeHead then xs.query row d else 0) *
    (if v.start+key.val < p.context ∧ d.val < p.activeHead then xs.key v.start key d else 0)) *
      xs.queryScale * xs.keyScale (v.start/BN)

def pairs (BM BN BD startM : Nat) (p : Parameters) (xs : Inputs BM BN BD)
    (row : Fin BM) (d : Fin BD) : List (ℝ × ℝ) :=
  ((visits BM BN startM p).map (fun v => (List.finRange BN).map (fun key =>
    (score BM BN BD startM p xs v row key +
      (if v.diagonal = Bool.true ∧ startM*BM+row.val < v.start+key.val then -1000000 else 0),
     if v.start+key.val < p.context ∧ d.val < p.activeHead then xs.value v.start key d else 0)))).flatten

/-- Independent normalized base-2 weighted sum of the visited input values. -/
def expected (BM BN BD startM : Nat) (p : Parameters) (xs : Inputs BM BN BD)
    (i : TileIndex [BM,BD]) : ℝ :=
  let values := pairs BM BN BD startM p xs i.1 i.2.1
  (values.map (fun sv => pow2 sv.1 * sv.2)).sum / (values.map (fun sv => pow2 sv.1)).sum

specification attention_correctness (BM BN BD : Nat) (p : Parameters) (hBN : 0 < BN) :
    Spec.Real (originalIO BM BN BD p ⊨ fun startM _ xs => expected BM BN BD startM p xs) := by
  exact RealContract.implements BM BN BD p Bool.false hBN

specification attention_reciprocal_correctness (BM BN BD : Nat) (p : Parameters) (hBN : 0 < BN) :
    Spec.Real (reciprocalIO BM BN BD p ⊨ fun startM _ xs => expected BM BN BD startM p xs) := by
  exact RealContract.implements BM BN BD p Bool.true hBN

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.Correctness
