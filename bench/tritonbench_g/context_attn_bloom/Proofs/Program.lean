import bench.tritonbench_g.context_attn_bloom.Kernels
import bench.tritonbench_g.context_attn_bloom.Proofs.Math
import VeriTile.Triton.Float.Erasure

namespace VeriTile.Bench.TritonBenchG.ContextAttnBloom.Program
open VeriTile.Triton Math
set_option maxHeartbeats 5000000
set_option maxRecDepth 20000

structure Regions where
  query : RegionName
  key : RegionName
  value : RegionName
  start : RegionName
  sequence : RegionName
  out : RegionName
  tokens : RegionName
  request : RegionName
  prompt : RegionName

def kernel (p : Layout) (r : Regions) (storage : Storage) (swapped : Bool) (scale : ℝ) : ComputeKernel :=
  (if swapped then VeriTile.Bench.TritonBenchG.ContextAttnBloom.scaleFirst storage else VeriTile.Bench.TritonBenchG.ContextAttnBloom.original storage)
    r.query r.key r.value scale r.start r.sequence r.out r.tokens r.request r.prompt
    p.qRow p.qHead p.qDim p.kRow p.kHead p.kDim p.vRow p.vHead p.vDim
    p.outRow p.outHead p.outDim p.requestRow p.requestToken p.kvGroup p.headDim p.BM p.D p.BN

def mathematical (p : Layout) (r : Regions) (swapped : Bool) (scale : ℝ) : ComputeKernel :=
  (if swapped then contextBloomSource%[tl.float32,scaleFirst] else contextBloomSource%[tl.float32,original])
    r.query r.key r.value scale r.start r.sequence r.out r.tokens r.request r.prompt
    p.qRow p.qHead p.qDim p.kRow p.kHead p.kDim p.vRow p.vHead p.vDim
    p.outRow p.outHead p.outDim p.requestRow p.requestToken p.kvGroup p.headDim p.BM p.D p.BN

theorem projects (p : Layout) (r : Regions) (storage : Storage) (swapped : Bool) (scale : ℝ) :
    (kernel p r storage swapped scale).toAlgorithm? = .ok (kernel p r storage swapped scale).toAlgKernel := by
  cases swapped <;> cases storage <;> rfl

theorem storage_erasure (p : Layout) (r : Regions) (storage : Storage) (swapped : Bool) (scale : ℝ) :
    (kernel p r storage swapped scale).eraseDType.toAlgKernel.body =
      (mathematical p r swapped scale).eraseDType.toAlgKernel.body := by
  cases swapped <;> cases storage
  all_goals simp only [kernel,mathematical,VeriTile.Bench.TritonBenchG.ContextAttnBloom.original,VeriTile.Bench.TritonBenchG.ContextAttnBloom.scaleFirst,
    Bool.false_eq_true,↓reduceIte,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,List.map_cons,List.map_nil,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,Except.bind,bind,eraseDType_float]
  all_goals simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,FloatDType.toTileDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType]
  all_goals rfl

end VeriTile.Bench.TritonBenchG.ContextAttnBloom.Program
