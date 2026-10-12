import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealExecutionSource
import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealFlatten

/-! Bundle the source's existing parameters for IO assembly. The selected
program is the complete typed original or reciprocal candidate. The real
projection and every source decomposition are checked against those programs. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealSource
open VeriTile.Triton RealPrologue
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000

def source (BM BN BD Z finish : Nat) (p : Parameters) (reciprocal : Bool) : ComputeKernel :=
  if reciprocal then
    TypedSource.reciprocal p.layout.Q p.layout.K p.layout.V p.layout.M p.layout.Out p.layout.L p.scale
      p.strides.qBatch p.strides.qHead p.layout.qRow p.layout.qDim
      p.strides.kBatch p.strides.kHead p.layout.kRow p.layout.kDim
      p.strides.vBatch p.strides.vHead p.layout.vRow p.layout.vDim
      p.strides.oBatch p.strides.oHead p.layout.oRow p.layout.oDim
      Z p.heads p.kvHeads p.layout.queryContext p.layout.roundedContext p.layout.kvContext
      p.evenM p.evenN BM BD BN finish p.init p.window p.complement p.offset p.size
  else
    TypedSource.original p.layout.Q p.layout.K p.layout.V p.layout.M p.layout.Out p.layout.L p.scale
      p.strides.qBatch p.strides.qHead p.layout.qRow p.layout.qDim
      p.strides.kBatch p.strides.kHead p.layout.kRow p.layout.kDim
      p.strides.vBatch p.strides.vHead p.layout.vRow p.layout.vDim
      p.strides.oBatch p.strides.oHead p.layout.oRow p.layout.oDim
      Z p.heads p.kvHeads p.layout.queryContext p.layout.roundedContext p.layout.kvContext
      p.evenM p.evenN BM BD BN finish p.init p.window p.complement p.offset p.size

def kernel (BM BN BD Z finish : Nat) (p : Parameters) (reciprocal : Bool) : ComputeKernel :=
  (source BM BN BD Z finish p reciprocal).eraseDType

theorem source_projection (BM BN BD Z finish : Nat) (p : Parameters) (reciprocal : Bool) :
    (source BM BN BD Z finish p reciprocal).toAlgorithm? =
      .ok (source BM BN BD Z finish p reciprocal).toAlgKernel := by
  cases reciprocal <;> rfl

theorem projection (BM BN BD Z finish : Nat) (p : Parameters) (reciprocal : Bool) :
    (kernel BM BN BD Z finish p reciprocal).toAlgorithm? =
      .ok (kernel BM BN BD Z finish p reciprocal).toAlgKernel := by
  simp only [kernel, ComputeKernel.eraseDType, source_projection]
  simp [ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?_fromKernelBody]

theorem body (BM BN BD Z finish : Nat) (p : Parameters) (reciprocal : Bool) :
    (kernel BM BN BD Z finish p reciprocal).toAlgKernel.body =
      RealSafety.code BM BN BD finish p reciprocal := by
  cases reciprocal
  · exact RealSafetySource.body_split (TypedSource.original p.layout.Q p.layout.K p.layout.V p.layout.M p.layout.Out p.layout.L p.scale
      p.strides.qBatch p.strides.qHead p.layout.qRow p.layout.qDim
      p.strides.kBatch p.strides.kHead p.layout.kRow p.layout.kDim
      p.strides.vBatch p.strides.vHead p.layout.vRow p.layout.vDim
      p.strides.oBatch p.strides.oHead p.layout.oRow p.layout.oDim
      Z p.heads p.kvHeads p.layout.queryContext p.layout.roundedContext p.layout.kvContext
      p.evenM p.evenN BM BD BN finish p.init p.window p.complement p.offset p.size) BM BN BD finish p Bool.false (by rfl) (by rfl)
      (RealPrologueSource.original_prologue p.layout.Q p.layout.K p.layout.V p.layout.M p.layout.Out p.layout.L p.scale
      p.strides.qBatch p.strides.qHead p.layout.qRow p.layout.qDim
      p.strides.kBatch p.strides.kHead p.layout.kRow p.layout.kDim
      p.strides.vBatch p.strides.vHead p.layout.vRow p.layout.vDim
      p.strides.oBatch p.strides.oHead p.layout.oRow p.layout.oDim
      Z p.heads p.kvHeads p.layout.queryContext p.layout.roundedContext p.layout.kvContext
      p.evenM p.evenN BM BD BN finish p.init p.window p.complement p.offset p.size)
      (RealLoopSource.original_loop_source p.layout.Q p.layout.K p.layout.V p.layout.M p.layout.Out p.layout.L p.scale
      p.strides.qBatch p.strides.qHead p.layout.qRow p.layout.qDim
      p.strides.kBatch p.strides.kHead p.layout.kRow p.layout.kDim
      p.strides.vBatch p.strides.vHead p.layout.vRow p.layout.vDim
      p.strides.oBatch p.strides.oHead p.layout.oRow p.layout.oDim
      Z p.heads p.kvHeads p.layout.queryContext p.layout.roundedContext p.layout.kvContext
      p.evenM p.evenN BM BD BN finish p.init p.window p.complement p.offset p.size)
      (RealEpilogueSource.original_epilogue p.layout.Q p.layout.K p.layout.V p.layout.M p.layout.Out p.layout.L p.scale
      p.strides.qBatch p.strides.qHead p.layout.qRow p.layout.qDim
      p.strides.kBatch p.strides.kHead p.layout.kRow p.layout.kDim
      p.strides.vBatch p.strides.vHead p.layout.vRow p.layout.vDim
      p.strides.oBatch p.strides.oHead p.layout.oRow p.layout.oDim
      Z p.heads p.kvHeads p.layout.queryContext p.layout.roundedContext p.layout.kvContext
      p.evenM p.evenN BM BD BN finish p.init p.window p.complement p.offset p.size)
  · exact RealSafetySource.body_split (TypedSource.reciprocal p.layout.Q p.layout.K p.layout.V p.layout.M p.layout.Out p.layout.L p.scale
      p.strides.qBatch p.strides.qHead p.layout.qRow p.layout.qDim
      p.strides.kBatch p.strides.kHead p.layout.kRow p.layout.kDim
      p.strides.vBatch p.strides.vHead p.layout.vRow p.layout.vDim
      p.strides.oBatch p.strides.oHead p.layout.oRow p.layout.oDim
      Z p.heads p.kvHeads p.layout.queryContext p.layout.roundedContext p.layout.kvContext
      p.evenM p.evenN BM BD BN finish p.init p.window p.complement p.offset p.size) BM BN BD finish p Bool.true (by rfl) (by rfl)
      (RealPrologueSource.candidate_prologue p.layout.Q p.layout.K p.layout.V p.layout.M p.layout.Out p.layout.L p.scale
      p.strides.qBatch p.strides.qHead p.layout.qRow p.layout.qDim
      p.strides.kBatch p.strides.kHead p.layout.kRow p.layout.kDim
      p.strides.vBatch p.strides.vHead p.layout.vRow p.layout.vDim
      p.strides.oBatch p.strides.oHead p.layout.oRow p.layout.oDim
      Z p.heads p.kvHeads p.layout.queryContext p.layout.roundedContext p.layout.kvContext
      p.evenM p.evenN BM BD BN finish p.init p.window p.complement p.offset p.size)
      (RealLoopSource.candidate_loop_source p.layout.Q p.layout.K p.layout.V p.layout.M p.layout.Out p.layout.L p.scale
      p.strides.qBatch p.strides.qHead p.layout.qRow p.layout.qDim
      p.strides.kBatch p.strides.kHead p.layout.kRow p.layout.kDim
      p.strides.vBatch p.strides.vHead p.layout.vRow p.layout.vDim
      p.strides.oBatch p.strides.oHead p.layout.oRow p.layout.oDim
      Z p.heads p.kvHeads p.layout.queryContext p.layout.roundedContext p.layout.kvContext
      p.evenM p.evenN BM BD BN finish p.init p.window p.complement p.offset p.size)
      (RealEpilogueSource.candidate_epilogue p.layout.Q p.layout.K p.layout.V p.layout.M p.layout.Out p.layout.L p.scale
      p.strides.qBatch p.strides.qHead p.layout.qRow p.layout.qDim
      p.strides.kBatch p.strides.kHead p.layout.kRow p.layout.kDim
      p.strides.vBatch p.strides.vHead p.layout.vRow p.layout.vDim
      p.strides.oBatch p.strides.oHead p.layout.oRow p.layout.oDim
      Z p.heads p.kvHeads p.layout.queryContext p.layout.roundedContext p.layout.kvContext
      p.evenM p.evenN BM BD BN finish p.init p.window p.complement p.offset p.size)

theorem flatten_ok (BM BN BD Z finish : Nat) (p : Parameters) (reciprocal : Bool) :
    (kernel BM BN BD Z finish p reciprocal).toAlgKernel.FlattenOk := by
  unfold Kernel.FlattenOk
  rw [body]
  exact RealFlatten.code_ok BM BN BD finish p reciprocal

theorem safe (BM BN BD Z finish : Nat) (p : Parameters) (reciprocal : Bool)
    (s : BlockState) (bounds : RegionBounds) (hN : 0 < BN)
    (hflag : p.evenN ≠ 0 → p.layout.kvContext % BN = 0)
    (hb : RealSafety.Bounds BM BN BD finish p s bounds) :
    (kernel BM BN BD Z finish p reciprocal).toAlgKernel.TraceSafe bounds s := by
  unfold Kernel.TraceSafe
  rw [body]
  exact RealSafety.complete_safe BM BN BD finish p reciprocal s bounds hN hflag hb

theorem executes (BM BN BD Z finish : Nat) (p : Parameters) (reciprocal : Bool)
    (s : BlockState) (hN : 0 < BN) (hflag : p.evenN ≠ 0 → p.layout.kvContext % BN = 0)
    (hl : RealExecution.OutputLayout BM BD p s) :
    ∃ t, exec (kernel BM BN BD Z finish p reciprocal).toAlgKernel s = some t ∧
      RealExecution.Results BM BN BD finish p s t := by
  cases reciprocal
  · exact RealExecutionSource.original_executes p.layout.Q p.layout.K p.layout.V p.layout.M p.layout.Out p.layout.L p.scale
      p.strides.qBatch p.strides.qHead p.layout.qRow p.layout.qDim
      p.strides.kBatch p.strides.kHead p.layout.kRow p.layout.kDim
      p.strides.vBatch p.strides.vHead p.layout.vRow p.layout.vDim
      p.strides.oBatch p.strides.oHead p.layout.oRow p.layout.oDim
      Z p.heads p.kvHeads p.layout.queryContext p.layout.roundedContext p.layout.kvContext
      p.evenM p.evenN BM BD BN finish p.init p.window p.complement p.offset p.size s hN hflag hl
  · exact RealExecutionSource.candidate_executes p.layout.Q p.layout.K p.layout.V p.layout.M p.layout.Out p.layout.L p.scale
      p.strides.qBatch p.strides.qHead p.layout.qRow p.layout.qDim
      p.strides.kBatch p.strides.kHead p.layout.kRow p.layout.kDim
      p.strides.vBatch p.strides.vHead p.layout.vRow p.layout.vDim
      p.strides.oBatch p.strides.oHead p.layout.oRow p.layout.oDim
      Z p.heads p.kvHeads p.layout.queryContext p.layout.roundedContext p.layout.kvContext
      p.evenM p.evenN BM BD BN finish p.init p.window p.complement p.offset p.size s hN hflag hl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealSource
