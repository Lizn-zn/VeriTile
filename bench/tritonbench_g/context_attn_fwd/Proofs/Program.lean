import bench.tritonbench_g.context_attn_fwd.Kernels
import bench.tritonbench_g.context_attn_fwd.Proofs.Formula
import VeriTile.Triton.Float.Erasure
namespace VeriTile.Bench.TritonBenchG.ContextAttnFwd.Program
open VeriTile.Triton Formula
set_option maxHeartbeats 5000000
set_option maxRecDepth 20000

structure Regions where
  query : RegionName
  key : RegionName
  value : RegionName
  out : RegionName
  start : RegionName
  sequence : RegionName
  prompt : RegionName

def kernel (p : Layout) (r : Regions) (storage : Storage) (swapped : Bool) (scale : ℝ) : ComputeKernel :=
  (if swapped then VeriTile.Bench.TritonBenchG.ContextAttnFwd.scaleFirst storage else VeriTile.Bench.TritonBenchG.ContextAttnFwd.original storage)
    r.query r.key r.value scale r.out r.start r.sequence r.prompt
    p.qRow p.qHead p.qDim p.kBatch p.kHead p.kRow p.kDim
    p.vBatch p.vHead p.vRow p.vDim p.outRow p.outHead p.outDim p.kvGroup p.heads p.D p.BM p.BN

def mathematical (p : Layout) (r : Regions) (swapped : Bool) (scale : ℝ) : ComputeKernel :=
  kernel p r .fp32 swapped scale

theorem projects (p : Layout) (r : Regions) (storage : Storage) (swapped : Bool) (scale : ℝ) :
    (kernel p r storage swapped scale).toAlgorithm? = .ok (kernel p r storage swapped scale).toAlgKernel := by
  cases swapped <;> cases storage <;> rfl

theorem storage_erasure (p : Layout) (r : Regions) (storage : Storage) (swapped : Bool) (scale : ℝ) :
    (kernel p r storage swapped scale).eraseDType.toAlgKernel.body =
      (mathematical p r swapped scale).eraseDType.toAlgKernel.body := by
  cases swapped <;> cases storage
  all_goals simp only [kernel,mathematical,VeriTile.Bench.TritonBenchG.ContextAttnFwd.original,VeriTile.Bench.TritonBenchG.ContextAttnFwd.scaleFirst,
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

def parts (p : Layout) (r : Regions) (swapped : Bool) (scale : ℝ) : Source.IntRange :=
  match Source.intRange? (mathematical p r swapped scale).surfaceBody "start_n" with
  | some x => x
  | none => ⟨[],.constInt 0,.constInt 0,.constNat 1,[],[]⟩

def lower (code : List ComputeStmt) : List Stmt :=
  ((ComputeStmt.listToAlgorithm? code).toOption.getD []).map Stmt.eraseDType

def prelude (p : Layout) (r : Regions) : List Stmt :=
  lower (parts p r Bool.false 0).before

theorem prelude_shared (p : Layout) (r : Regions) (swapped : Bool) (scale : ℝ) :
    lower (parts p r swapped scale).before = prelude p r := by
  cases swapped <;> rfl

def launch (u : BlockState) : Launch := ⟨u.pids 0,u.pids 1⟩
def inputs (u : BlockState) (r : Regions) : Inputs where
  q := u.readMem r.query
  k := u.readMem r.key
  v := u.readMem r.value
  start := u.readMemValue .int r.start
  sequence := u.readMemValue .int r.sequence
  prompt := u.readMemValue .int r.prompt

end VeriTile.Bench.TritonBenchG.ContextAttnFwd.Program
