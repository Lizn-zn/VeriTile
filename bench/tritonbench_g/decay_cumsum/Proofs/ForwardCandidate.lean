import bench.tritonbench_g.decay_cumsum.Proofs.ForwardProjection
import VeriTile.Triton.Semantics.ProgramRewrite
namespace VeriTile.Bench.TritonBenchG.DecayCumsum.ForwardCandidate
open VeriTile.Triton TypedSource
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000

def mask (swapped : Bool) (BK DK : Nat) : Stmt :=
  .assign .bool [BK] "mask" (.lt .nat .scalarR
    (.add .nat .scalarL
      (if swapped then .mul .nat .nil (.constNat BK) (.ref .nat [] "i_k")
       else .mul .nat .nil (.ref .nat [] "i_k") (.constNat BK))
      (.arange BK)) (.constNat DK))

def body (swapped : Bool) (g go : RegionName)
    (sH sT sD B H T BT BK DK : Nat) (scale : ℝ) : List Stmt :=
  let old := ForwardProjection.canonicalBody g go sH sT sD B H T BT BK DK scale
  old.take 4 ++ [mask swapped BK DK] ++ old.drop 5

theorem source_body (storage : Storage) (swapped : Bool) (g go : RegionName)
    (sH sT sD B H T BT BK DK : Nat) (scale : ℝ) :
    (forward storage swapped g go sH sT sD B H T BT BK DK scale).eraseDType.toAlgKernel.body =
      body swapped g go sH sT sD B H T BT BK DK scale := by
  cases storage <;> cases swapped
  all_goals simp only [forward,inv_ln2,body,mask,ForwardProjection.canonicalBody,
    RealReference.fwdBody,RealReference.fwd_decay_cumsum_surface,
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

theorem mask_equal (BK DK : Nat) (u : BlockState) :
    stepStmts [mask Bool.false BK DK] u = stepStmts [mask Bool.true BK DK] u := by
  have hmul : evalOp (.mul .nat .nil (.ref .nat [] "i_k") (.constNat BK)) u =
      evalOp (.mul .nat .nil (.constNat BK) (.ref .nat [] "i_k")) u := by
    simp only [evalOp_mul,evalOp_constNat]
    cases h : evalOp (.ref .nat [] "i_k") u
    · rfl
    · simp only [bind,Option.bind_some]
      congr 1
      ext i
      exact Nat.mul_comm _ _
  simp only [mask,Bool.false_eq_true,↓reduceIte,stepStmts,stepStmt,
    evalOp_lt,evalOp_add,hmul]

theorem flat_mask_equal (A : FlatAlloc) (BK DK : Nat) (u : BlockState) :
    stepStmts (A.flattenStmts [mask Bool.false BK DK]) u =
      stepStmts (A.flattenStmts [mask Bool.true BK DK]) u := by
  simpa only [mask,Bool.false_eq_true,↓reduceIte,FlatAlloc.flattenStmts,
    FlatAlloc.flattenStmt,FlatAlloc.flattenOp] using mask_equal BK DK u

theorem flat_exec_equal (A : FlatAlloc) (storage : Storage) (g go : RegionName)
    (sH sT sD B H T BT BK DK : Nat) (scale : ℝ) (u : BlockState) :
    exec (A.flattenKernel (forward storage Bool.false g go sH sT sD B H T BT BK DK scale).eraseDType.toAlgKernel) u =
      exec (A.flattenKernel (forward storage Bool.true g go sH sT sD B H T BT BK DK scale).eraseDType.toAlgKernel) u := by
  simp only [exec,FlatAlloc.flattenKernel,source_body,body,FlatAlloc.flattenStmts_append]
  exact ProgramRewrite.context _ _ _ _ (flat_mask_equal A BK DK) u

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DecayCumsum.ForwardCandidate
