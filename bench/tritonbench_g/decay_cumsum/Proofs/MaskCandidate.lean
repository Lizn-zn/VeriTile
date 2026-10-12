import bench.tritonbench_g.decay_cumsum.Kernels
import bench.tritonbench_g.decay_cumsum.Proofs.ForwardCandidate
namespace VeriTile.Bench.TritonBenchG.DecayCumsum
open VeriTile.Triton TypedSource
set_option maxHeartbeats 8000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

namespace PrepareCandidate
/-- Only the exact natural-number product in the mask changes. -/
def body (swapped : Bool) (q k g qg kg : RegionName)
    (sH sT sD B H T BT BK DK : Nat) (scale : ℝ) : List Stmt :=
  let old := (prepare .fp32 .fp32 Bool.false q k g qg kg sH sT sD B H T BT BK DK scale).eraseDType.toAlgKernel.body
  old.take 6 ++ [ForwardCandidate.mask swapped BK DK] ++ old.drop 7

theorem source_body (source gate : Storage) (swapped : Bool) (q k g qg kg : RegionName)
    (sH sT sD B H T BT BK DK : Nat) (scale : ℝ) :
    (TypedSource.prepare source gate swapped q k g qg kg sH sT sD B H T BT BK DK scale).eraseDType.toAlgKernel.body =
      body swapped q k g qg kg sH sT sD B H T BT BK DK scale := by
  cases source <;> cases gate <;> cases swapped
  all_goals simp only [TypedSource.prepare,body,ForwardCandidate.mask,
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

theorem flat_exec_equal (A : FlatAlloc) (source gate : Storage) (q k g qg kg : RegionName)
    (sH sT sD B H T BT BK DK : Nat) (scale : ℝ) (u : BlockState) :
    exec (A.flattenKernel (TypedSource.prepare source gate Bool.false q k g qg kg sH sT sD B H T BT BK DK scale).eraseDType.toAlgKernel) u =
      exec (A.flattenKernel (TypedSource.prepare source gate Bool.true q k g qg kg sH sT sD B H T BT BK DK scale).eraseDType.toAlgKernel) u := by
  unfold exec FlatAlloc.flattenKernel
  rw [source_body,source_body]
  unfold body
  simp only [FlatAlloc.flattenStmts_append]
  exact ProgramRewrite.context _ _ _ _ (ForwardCandidate.flat_mask_equal A BK DK) u

end PrepareCandidate

namespace BackwardCandidate
/-- Only the exact natural-number product in the mask changes. -/
def body (swapped : Bool) (dq_inner dq_inter dk_inner dk_inter q k g dg : RegionName)
    (sH sT sD B H T BT BK DK : Nat) (scale : ℝ) : List Stmt :=
  let old := (backward .fp32 .fp32 Bool.false dq_inner dq_inter dk_inner dk_inter q k g dg sH sT sD B H T BT BK DK scale).eraseDType.toAlgKernel.body
  old.take 10 ++ [ForwardCandidate.mask swapped BK DK] ++ old.drop 11

theorem source_body (source gate : Storage) (swapped : Bool) (dq_inner dq_inter dk_inner dk_inter q k g dg : RegionName)
    (sH sT sD B H T BT BK DK : Nat) (scale : ℝ) :
    (TypedSource.backward source gate swapped dq_inner dq_inter dk_inner dk_inter q k g dg sH sT sD B H T BT BK DK scale).eraseDType.toAlgKernel.body =
      body swapped dq_inner dq_inter dk_inner dk_inter q k g dg sH sT sD B H T BT BK DK scale := by
  cases source <;> cases gate <;> cases swapped
  all_goals simp only [TypedSource.backward,body,ForwardCandidate.mask,
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

theorem flat_exec_equal (A : FlatAlloc) (source gate : Storage) (dq_inner dq_inter dk_inner dk_inter q k g dg : RegionName)
    (sH sT sD B H T BT BK DK : Nat) (scale : ℝ) (u : BlockState) :
    exec (A.flattenKernel (TypedSource.backward source gate Bool.false dq_inner dq_inter dk_inner dk_inter q k g dg sH sT sD B H T BT BK DK scale).eraseDType.toAlgKernel) u =
      exec (A.flattenKernel (TypedSource.backward source gate Bool.true dq_inner dq_inter dk_inner dk_inter q k g dg sH sT sD B H T BT BK DK scale).eraseDType.toAlgKernel) u := by
  unfold exec FlatAlloc.flattenKernel
  rw [source_body,source_body]
  unfold body
  simp only [FlatAlloc.flattenStmts_append]
  exact ProgramRewrite.context _ _ _ _ (ForwardCandidate.flat_mask_equal A BK DK) u

end BackwardCandidate

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DecayCumsum
