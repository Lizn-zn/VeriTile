import bench.tritonbench_g.decay_cumsum.Kernels
import bench.tritonbench_g.decay_cumsum.Proofs.RealReference
import VeriTile.Triton.Float.Erasure
namespace VeriTile.Bench.TritonBenchG.DecayCumsum.ForwardProjection
open VeriTile.Triton TypedSource
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000

/-- Tuple assignment is represented by an always-taken block; the one-argument
range uses constant dynamic bounds. Both are connected to the reference's
sequential prologue and static range by execution lemmas below. -/
def canonicalBody (g go : RegionName) (sH sT sD B H T BT BK DK : Nat) (scale : ℝ) : List Stmt :=
  let old := (RealReference.fwd_decay_cumsum_surface g go sH sT sD B H T scale BT BK DK).toAlgKernel.body
  [.ifThen (.constBool Bool.true) (old.take 3)] ++ (old.drop 3).take 4 ++
    [.forRangeDyn "i" (.constNat 0) (.constNat BT) (.constNat 1)
      (RealReference.fwdBody g go sH BT BK DK)]

theorem original_body (storage : Storage) (g go : RegionName)
    (sH sT sD B H T BT BK DK : Nat) (scale : ℝ) :
    (forward storage Bool.false g go sH sT sD B H T BT BK DK scale).eraseDType.toAlgKernel.body =
    canonicalBody g go sH sT sD B H T BT BK DK scale := by
  cases storage
  all_goals simp only [forward,inv_ln2,canonicalBody,RealReference.fwdBody,RealReference.fwd_decay_cumsum_surface,
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
theorem const_dynamic_step (idx : RegName) (start stop step : Nat)
    (body : List Stmt) (u : BlockState) :
    stepStmt (.forRangeDyn idx (.constNat start) (.constNat stop) (.constNat step) body) u =
      stepStmt (.forRange idx start stop step body) u := by
  simp only [stepForRangeAux.forRangeDyn_unfold,stepForRangeAux.forRange_unfold,
    evalOp_constNat,Option.bind_some,Tile.scalar_data]

theorem tuple_step (head : List Stmt) (u : BlockState) :
    stepStmt (.ifThen (.constBool Bool.true) head) u = stepStmts head u := by
  simp [stepStmt,evalOp]

theorem list_append (head tail : List Stmt) (u : BlockState) :
    stepStmts (head ++ tail) u = (stepStmts head u).bind (stepStmts tail) := by
  induction head generalizing u with
  | nil => simp [stepStmts]
  | cons a rest ih =>
    simp only [List.cons_append,stepStmts]
    cases h : stepStmt a u <;> simp [h,ih]

theorem control_bridge (head middle body : List Stmt) (BT : Nat) (u : BlockState) :
    stepStmts ([.ifThen (.constBool Bool.true) head] ++ middle ++
      [.forRangeDyn "i" (.constNat 0) (.constNat BT) (.constNat 1) body]) u =
    stepStmts (head ++ middle ++ [.forRange "i" 0 BT 1 body]) u := by
  simp only [list_append,stepStmts,tuple_step,const_dynamic_step,
    Option.bind_some,Option.bind_assoc]
  cases h : stepStmts head u <;> rfl

theorem reference_body (g go : RegionName) (sH sT sD B H T BT BK DK : Nat) (scale : ℝ) :
    let old := (RealReference.fwd_decay_cumsum_surface g go sH sT sD B H T scale BT BK DK).toAlgKernel.body
    old = old.take 3 ++ (old.drop 3).take 4 ++ [.forRange "i" 0 BT 1
      (RealReference.fwdBody g go sH BT BK DK)] := by rfl

theorem original_exec (storage : Storage) (g go : RegionName)
    (sH sT sD B H T BT BK DK : Nat) (scale : ℝ) (u : BlockState) :
    exec (forward storage Bool.false g go sH sT sD B H T BT BK DK scale).eraseDType.toAlgKernel u =
    exec (RealReference.fwd_decay_cumsum_surface g go sH sT sD B H T scale BT BK DK).toAlgKernel u := by
  unfold exec
  rw [original_body]
  unfold canonicalBody
  rw [control_bridge]
  rw [← reference_body]

theorem flatten_append (A : FlatAlloc) (head tail : List Stmt) :
    A.flattenStmts (head ++ tail) = A.flattenStmts head ++ A.flattenStmts tail := by
  induction head with
  | nil => rfl
  | cons a rest ih => simp [FlatAlloc.flattenStmts,ih]

theorem control_flat_bridge (A : FlatAlloc) (head middle body : List Stmt)
    (BT : Nat) (u : BlockState) :
    stepStmts (A.flattenStmts ([.ifThen (.constBool Bool.true) head] ++ middle ++
      [.forRangeDyn "i" (.constNat 0) (.constNat BT) (.constNat 1) body])) u =
    stepStmts (A.flattenStmts (head ++ middle ++ [.forRange "i" 0 BT 1 body])) u := by
  simp only [flatten_append,FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,FlatAlloc.flattenOp]
  exact control_bridge _ _ _ _ _

theorem original_flat_exec (storage : Storage) (g go : RegionName)
    (sH sT sD B H T BT BK DK : Nat) (scale : ℝ) (A : FlatAlloc) (u : BlockState) :
    exec (A.flattenKernel (forward storage Bool.false g go sH sT sD B H T BT BK DK scale).eraseDType.toAlgKernel) u =
    exec (A.flattenKernel (RealReference.fwd_decay_cumsum_surface g go sH sT sD B H T scale BT BK DK).toAlgKernel) u := by
  unfold exec FlatAlloc.flattenKernel
  rw [original_body]
  unfold canonicalBody
  rw [control_flat_bridge]
  rw [← reference_body]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DecayCumsum.ForwardProjection
