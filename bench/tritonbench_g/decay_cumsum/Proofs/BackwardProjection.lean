import bench.tritonbench_g.decay_cumsum.Kernels
import bench.tritonbench_g.decay_cumsum.Proofs.RealReference
import VeriTile.Triton.Float.Erasure
namespace VeriTile.Bench.TritonBenchG.DecayCumsum.BackwardProjection
open VeriTile.Triton TypedSource
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

/-- Complete source prelude, including all eight reverse pointers and both fp32 accumulators. -/
def prelude (dq_inner dq_inter dk_inner dk_inter q k g dg : RegionName) (sH DK BT BK : Nat) (scale : ℝ) : List Stmt :=
  ((backward .fp32 .fp32 Bool.false dq_inner dq_inter dk_inner dk_inter q k g dg sH 0 0 0 0 0 BT BK DK scale).eraseDType.toAlgKernel.body).take 12

def stop (BT : Nat) : Op .nat [] :=
  .add .nat .nil (.div .nat .nil (.sub .nat .nil (.constNat BT) (.constNat 1)) (.constNat 1)) (.constNat 1)

def canonicalBody (dq_inner dq_inter dk_inner dk_inter q k g dg : RegionName) (sH DK BT BK : Nat) (scale : ℝ) : List Stmt :=
  prelude dq_inner dq_inter dk_inner dk_inter q k g dg sH DK BT BK scale ++
    [.forRangeDyn "__rev_j" (.constNat 0) (stop BT) (.constNat 1)
      (RealReference.bwdIterBodyG DK BT BK)]

theorem original_body (source gate : Storage) (dq_inner dq_inter dk_inner dk_inter q k g dg : RegionName)
    (sH sT sD B H T BT BK DK : Nat) (scale : ℝ) :
    (backward source gate Bool.false dq_inner dq_inter dk_inner dk_inter q k g dg sH sT sD B H T BT BK DK scale).eraseDType.toAlgKernel.body =
      canonicalBody dq_inner dq_inter dk_inner dk_inter q k g dg sH DK BT BK scale := by
  cases source <;> cases gate
  all_goals simp only [backward,canonicalBody,prelude,RealReference.bwdIterBodyG,stop,
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
end VeriTile.Bench.TritonBenchG.DecayCumsum.BackwardProjection
