import bench.tritonbench_g.attn_fwd_causal.Proofs.RealIO
import bench.tritonbench_g.attn_fwd_causal.Proofs.RealSafety
import bench.tritonbench_g.attn_fwd_causal.Proofs.RealFlatten

/-! Full real IO correctness for both complete source variants. The expected
value is the independent weighted sum over actual helper visits. The source's
unused stride arguments remain visible; Q strides still control V/Out access.
Physical allocation bounds and input pins are ordinary contract premises. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealContract
open VeriTile.Triton RealLoads
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000

structure Parameters where
  (Q K V QS KS Out : RegionName)
  (queryBatch queryHead queryRow queryDim : Nat)
  (keyBatch keyHead keyRow keyDim : Nat)
  (valueBatch valueHead valueRow valueDim : Nat)
  (outputBatch outputHead outputRow outputDim : Nat)
  (batches heads context headDim activeHead stage : Nat)
  validStage : stage = 2 ∨ stage = 3

def source (BM BN BD : Nat) (p : Parameters) (candidate : Bool) : ComputeKernel :=
  (if candidate then TypedSource.reciprocal else TypedSource.original)
    p.Q p.K p.V p.QS p.KS p.Out p.queryBatch p.queryHead p.queryRow p.queryDim
    p.keyBatch p.keyHead p.keyRow p.keyDim p.valueBatch p.valueHead p.valueRow p.valueDim
    p.outputBatch p.outputHead p.outputRow p.outputDim
    p.batches p.heads p.context p.headDim BM BN BD p.activeHead p.stage p.validStage

def kernel (BM BN BD : Nat) (p : Parameters) (candidate : Bool) : ComputeKernel :=
  (source BM BN BD p candidate).eraseDType

theorem source_projection (BM BN BD : Nat) (p : Parameters) (candidate : Bool) :
    (source BM BN BD p candidate).toAlgorithm? = .ok (source BM BN BD p candidate).toAlgKernel := by
  cases candidate <;> rfl

theorem projection (BM BN BD : Nat) (p : Parameters) (candidate : Bool) :
    (kernel BM BN BD p candidate).toAlgorithm? = .ok (kernel BM BN BD p candidate).toAlgKernel := by
  simp only [kernel, ComputeKernel.eraseDType, source_projection]
  simp [ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?_fromKernelBody]

def configuration (BM BN : Nat) (p : Parameters) (startM headBatch : Nat) : Config where
  Q := p.Q
  K := p.K
  V := p.V
  QS := p.QS
  KS := p.KS
  base := headBatch/p.heads*p.queryBatch+headBatch%p.heads*p.queryHead
  queryScaleBase := headBatch*RealCoordinates.ceil p.context BM
  keyScaleBase := headBatch*RealCoordinates.ceil p.context BN
  strideM := p.queryRow
  strideD := p.queryDim
  strideKN := p.keyRow
  headDim := p.headDim
  context := p.context
  activeHead := p.activeHead
  startM := startM

def io (BM BN BD : Nat) (p : Parameters) (candidate : Bool) : RealIO.IO BM BN BD where
  kernel := kernel BM BN BD p candidate
  projection := projection BM BN BD p candidate
  configuration := configuration BM BN p
  out := p.Out
  stage := p.stage

def expected (BM BN BD startM : Nat) (p : Parameters) (xs : RealFormula.Inputs BM BN BD)
    (i : TileIndex [BM,BD]) : ℝ :=
  RealFormula.expected xs startM p.context p.activeHead p.stage i.1 i.2.1

theorem implements (BM BN BD : Nat) (p : Parameters) (candidate : Bool) (hBN : 0 < BN) :
    (io BM BN BD p candidate).Implements (fun startM _ xs => expected BM BN BD startM p xs) := by
  apply RealIO.IO.Implements.intro
  · exact RealFlatten.supported p.Q p.K p.V p.QS p.KS p.Out
      p.queryBatch p.queryHead p.queryRow p.queryDim p.keyBatch p.keyHead p.keyRow p.keyDim
      p.valueBatch p.valueHead p.valueRow p.valueDim p.outputBatch p.outputHead p.outputRow p.outputDim
      p.batches p.heads p.context p.headDim BM BN BD p.activeHead p.stage p.validStage candidate
  · intro bounds s hu hb
    exact RealSafety.safe bounds s p.Q p.K p.V p.QS p.KS p.Out
      p.queryBatch p.queryHead p.queryRow p.queryDim p.keyBatch p.keyHead p.keyRow p.keyDim
      p.valueBatch p.valueHead p.valueRow p.valueDim p.outputBatch p.outputHead p.outputRow p.outputDim
      p.batches p.heads p.context p.headDim BM BN BD p.activeHead p.stage p.validStage candidate hBN hu hb
  · intro s xs hu hp hinj
    obtain ⟨t, he, hy, hf⟩ := RealExecution.executes s p.Q p.K p.V p.QS p.KS p.Out
      p.queryBatch p.queryHead p.queryRow p.queryDim p.keyBatch p.keyHead p.keyRow p.keyDim
      p.valueBatch p.valueHead p.valueRow p.valueDim p.outputBatch p.outputHead p.outputRow p.outputDim
      p.batches p.heads p.context p.headDim BM BN BD p.activeHead p.stage p.validStage candidate hBN hu
    refine ⟨t, he, ⟨?_, hf⟩⟩
    intro i hi
    exact (hy i hi hinj).trans
      (RealInputPins.formula_of_pins s (configuration BM BN p (s.pids 0) (s.pids 1)) p.stage xs hp i.1 i.2.1)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealContract
