import bench.tritonbench_g.f8_conversion_utils.Kernels
import VeriTile.Triton.Float.Erasure
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.F8ConversionUtils.Erasure
open VeriTile.Triton TypedSource
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000

def toHalf (swapped : Bool) : sourceType :=
  match swapped with
  | Bool.false => fun Y X N B => (f8Forward%[tl.float64,tl.float64,original] Y X N B).eraseDType
  | Bool.true => fun Y X N B => (f8Forward%[tl.float64,tl.float64,indexCommuted] Y X N B).eraseDType

def toFP8 (swapped : Bool) : sourceType :=
  match swapped with
  | Bool.false => fun Y X N B => (f8Reverse%[tl.float64,tl.float64,original] Y X N B).eraseDType
  | Bool.true => fun Y X N B => (f8Reverse%[tl.float64,tl.float64,indexCommuted] Y X N B).eraseDType

theorem to_half_body (format : Format) (swapped : Bool) (Y X : RegionName) (N B : Nat) :
    (TypedSource.toHalf format swapped Y X N B).eraseDType.toAlgKernel.body =
      (toHalf swapped Y X N B).toAlgKernel.body := by
  cases format <;> cases swapped
  all_goals simp only [TypedSource.toHalf,toHalf,
    ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,List.map_cons,List.map_nil,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    Except.bind,bind]
  all_goals simp only [Op.eraseDType.eq_def,VeriTile.Triton.eraseDType,FloatDType.toTileDType]
  all_goals rfl

theorem to_fp8_body (input : Input) (format : Format) (swapped : Bool) (Y X : RegionName) (N B : Nat) :
    (TypedSource.toFP8 input format swapped Y X N B).eraseDType.toAlgKernel.body =
      (toFP8 swapped Y X N B).toAlgKernel.body := by
  cases input <;> cases format <;> cases swapped
  all_goals simp only [TypedSource.toFP8,toFP8,
    ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,List.map_cons,List.map_nil,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    Except.bind,bind]
  all_goals simp only [Op.eraseDType.eq_def,VeriTile.Triton.eraseDType,FloatDType.toTileDType]
  all_goals rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.F8ConversionUtils.Erasure
