import bench.tritonbench_g.rotary_emb.Kernels
import bench.tritonbench_g.rotary_emb.Proofs.Parameters
import bench.tritonbench_g.rotary_emb.Proofs.ProjectionLists
import bench.tritonbench_g.rotary_emb.Proofs.RealCandidate

namespace VeriTile.Bench.TritonBenchG.RotaryEmb.FullSource
open VeriTile.Triton FullMath
set_option maxHeartbeats 16000000
set_option maxRecDepth 16000

theorem template_supported (data cos sin : Storage) (p : Parameters)
    (Q K Cos Sin : RegionName) :
    (template data cos sin Q K Cos Sin p.qBatch p.qHead p.qDim p.kBatch p.kHead p.kDim
      p.cosBatch p.cosDim p.sinBatch p.sinDim p.length p.headsQ p.headsK
      p.blockHead p.blockSeq p.dimension).toAlgorithm? = .ok
    (template data cos sin Q K Cos Sin p.qBatch p.qHead p.qDim p.kBatch p.kHead p.kDim
      p.cosBatch p.cosDim p.sinBatch p.sinDim p.length p.headsQ p.headsK
      p.blockHead p.blockSeq p.dimension).toAlgKernel := by
  cases data <;> cases cos <;> cases sin <;> rfl

theorem template_math_body (data cos sin : Storage) (p : Parameters)
    (Q K Cos Sin : RegionName) :
    (template data cos sin Q K Cos Sin p.qBatch p.qHead p.qDim p.kBatch p.kHead p.kDim
      p.cosBatch p.cosDim p.sinBatch p.sinDim p.length p.headsQ p.headsK
      p.blockHead p.blockSeq p.dimension).eraseDType.toAlgKernel.body =
      (mathKernel p Q K Cos Sin).body := by
  simp only [ComputeKernel.eraseDType, template_supported,
    ComputeKernel.toAlgKernel_fromKernelBody]
  cases data <;> cases cos <;> cases sin
  all_goals simp only [template,
    template_fp32_fp32, template_fp32_fp64, template_fp32_fp16, template_fp32_bf16, template_fp32_f8e4, template_fp32_f8e5, template_fp64_fp32, template_fp64_fp64, template_fp64_fp16, template_fp64_bf16, template_fp64_f8e4, template_fp64_f8e5, template_fp16_fp32, template_fp16_fp64, template_fp16_fp16, template_fp16_bf16, template_fp16_f8e4, template_fp16_f8e5, template_bf16_fp32, template_bf16_fp64, template_bf16_fp16, template_bf16_bf16, template_bf16_f8e4, template_bf16_f8e5, template_f8e4_fp32, template_f8e4_fp64, template_f8e4_fp16, template_f8e4_bf16, template_f8e4_f8e4, template_f8e4_f8e5, template_f8e5_fp32, template_f8e5_fp64, template_f8e5_fp16, template_f8e5_bf16, template_f8e5_f8e4, template_f8e5_f8e5,
    mathKernel, rotary_kernel_surface,
    ComputeKernel.surfaceBody, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Except.bind, bind, Kernel.eraseDType, Stmt.eraseDTypeList,
    Stmt.eraseDType, Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    FloatDType.toTileDType, ComputeDType.eraseDType]
  all_goals rfl

theorem original_supported (q k cos sin : Storage) (p : Parameters)
    (Q K Cos Sin : RegionName) :
    (original q k cos sin Q K Cos Sin p.qBatch p.qHead p.qDim p.kBatch p.kHead p.kDim
      p.cosBatch p.cosDim p.sinBatch p.sinDim p.length p.headsQ p.headsK
      p.blockHead p.blockSeq p.dimension).toAlgorithm? = .ok
    (original q k cos sin Q K Cos Sin p.qBatch p.qHead p.qDim p.kBatch p.kHead p.kDim
      p.cosBatch p.cosDim p.sinBatch p.sinDim p.length p.headsQ p.headsK
      p.blockHead p.blockSeq p.dimension).toAlgKernel := by
  exact ProjectionLists.joined_supported _ _ [Q,K,Cos,Sin] [Q,K] 20
    (template_supported q cos sin p Q K Cos Sin)
    (template_supported k cos sin p Q K Cos Sin)

/-- Independent Q/K storage composition still projects to the complete single
source body, including the Q stores before all K-phase reloads. -/
theorem original_math_body (q k cos sin : Storage) (p : Parameters)
    (Q K Cos Sin : RegionName) :
    (original q k cos sin Q K Cos Sin p.qBatch p.qHead p.qDim p.kBatch p.kHead p.kDim
      p.cosBatch p.cosDim p.sinBatch p.sinDim p.length p.headsQ p.headsK
      p.blockHead p.blockSeq p.dimension).eraseDType.toAlgKernel.body =
      (mathKernel p Q K Cos Sin).body := by
  unfold original
  rw [ProjectionLists.joined_erased_body _ _ [Q,K,Cos,Sin] [Q,K] 20
    (template_supported q cos sin p Q K Cos Sin)
    (template_supported k cos sin p Q K Cos Sin)]
  rw [template_math_body q cos sin p Q K Cos Sin,
    template_math_body k cos sin p Q K Cos Sin]
  exact List.take_append_drop 20 _

theorem kernel_supported (q k cos sin : Storage) (v : Bool) (p : Parameters)
    (Q K Cos Sin : RegionName) :
    (kernel q k cos sin v Q K Cos Sin p.qBatch p.qHead p.qDim p.kBatch p.kHead p.kDim
      p.cosBatch p.cosDim p.sinBatch p.sinDim p.length p.headsQ p.headsK
      p.blockHead p.blockSeq p.dimension).toAlgorithm? = .ok
    (kernel q k cos sin v Q K Cos Sin p.qBatch p.qHead p.qDim p.kBatch p.kHead p.kDim
      p.cosBatch p.cosDim p.sinBatch p.sinDim p.length p.headsQ p.headsK
      p.blockHead p.blockSeq p.dimension).toAlgKernel := by
  cases v
  · exact original_supported q k cos sin p Q K Cos Sin
  · exact ProjectionLists.replace_supported _ _ _ 2 _
      (headStatement Bool.true p.blockHead)
      (original_supported q k cos sin p Q K Cos Sin) rfl

theorem kernel_math_body (q k cos sin : Storage) (v : Bool) (p : Parameters)
    (Q K Cos Sin : RegionName) :
    (kernel q k cos sin v Q K Cos Sin p.qBatch p.qHead p.qDim p.kBatch p.kHead p.kDim
      p.cosBatch p.cosDim p.sinBatch p.sinDim p.length p.headsQ p.headsK
      p.blockHead p.blockSeq p.dimension).eraseDType.toAlgKernel.body =
      (mathVariant v p Q K Cos Sin).body := by
  cases v
  · simpa only [kernel, Bool.false_eq_true, ↓reduceIte, math_original] using
      original_math_body q k cos sin p Q K Cos Sin
  · unfold kernel
    simp only [↓reduceIte]
    rw [ProjectionLists.replace_erased_body _ _ _ 2 _
      (headStatement Bool.true p.blockHead)
      (original_supported q k cos sin p Q K Cos Sin) rfl]
    rw [original_math_body q k cos sin p Q K Cos Sin]
    rfl

theorem typed_run (q k cos sin : Storage) (v : Bool) (p : Parameters)
    (Q K Cos Sin : RegionName) (s : BlockState) :
    exec (kernel q k cos sin v Q K Cos Sin p.qBatch p.qHead p.qDim p.kBatch p.kHead p.kDim
      p.cosBatch p.cosDim p.sinBatch p.sinDim p.length p.headsQ p.headsK
      p.blockHead p.blockSeq p.dimension).eraseDType.toAlgKernel s =
      exec (mathKernel p Q K Cos Sin) s := by
  unfold exec
  rw [kernel_math_body]
  cases v
  · rw [math_original]
  · exact math_candidate_run p Q K Cos Sin s

theorem typed_flat_run (A : FlatAlloc) (q k cos sin : Storage) (v : Bool) (p : Parameters)
    (Q K Cos Sin : RegionName) (s : BlockState) :
    exec (A.flattenKernel (kernel q k cos sin v Q K Cos Sin p.qBatch p.qHead p.qDim
      p.kBatch p.kHead p.kDim p.cosBatch p.cosDim p.sinBatch p.sinDim p.length
      p.headsQ p.headsK p.blockHead p.blockSeq p.dimension).eraseDType.toAlgKernel) s =
      exec (A.flattenKernel (mathKernel p Q K Cos Sin)) s := by
  change stepStmts (A.flattenStmts
      (kernel q k cos sin v Q K Cos Sin p.qBatch p.qHead p.qDim p.kBatch p.kHead p.kDim
        p.cosBatch p.cosDim p.sinBatch p.sinDim p.length p.headsQ p.headsK
        p.blockHead p.blockSeq p.dimension).eraseDType.toAlgKernel.body) s = _
  rw [kernel_math_body]
  cases v
  · simp only [math_original, exec, FlatAlloc.flattenKernel]
  · exact math_candidate_flat A p Q K Cos Sin s

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RotaryEmb.FullSource
