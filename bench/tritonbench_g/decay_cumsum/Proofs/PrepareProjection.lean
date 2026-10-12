import bench.tritonbench_g.decay_cumsum.Kernels
import bench.tritonbench_g.decay_cumsum.Proofs.RealReference
import VeriTile.Triton.Float.Erasure
namespace VeriTile.Bench.TritonBenchG.DecayCumsum.PrepareProjection
open VeriTile.Triton TypedSource
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

/-- The complete source prelude includes the unmasked final gate load. -/
def prelude (q k g qg kg : RegionName) (sH DK BT BK : Nat) (scale : ℝ) : List Stmt :=
  ((prepare .fp32 .fp32 Bool.false q k g qg kg sH 0 0 0 0 0 BT BK DK scale).eraseDType.toAlgKernel.body).take 8

def canonicalBody (q k g qg kg : RegionName) (sH DK BT BK : Nat) (scale : ℝ) : List Stmt :=
  prelude q k g qg kg sH DK BT BK scale ++
    [.forRangeDyn "i" (.constNat 0) (.constNat BT) (.constNat 1)
      (RealReference.prepBody q k g qg kg sH DK BT BK scale)]

theorem original_body (source gate : Storage) (q k g qg kg : RegionName)
    (sH sT sD B H T BT BK DK : Nat) (scale : ℝ) :
    (prepare source gate Bool.false q k g qg kg sH sT sD B H T BT BK DK scale).eraseDType.toAlgKernel.body =
      canonicalBody q k g qg kg sH DK BT BK scale := by
  cases source <;> cases gate
  all_goals simp only [prepare,canonicalBody,prelude,RealReference.prepBody,
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

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DecayCumsum.PrepareProjection
