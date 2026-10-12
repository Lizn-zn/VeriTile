import bench.tritonbench_g.dequantize_rowwise.Kernels
import bench.tritonbench_g.dequantize_rowwise.Proofs.RealReference
import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.Semantics.ProgramRewrite
namespace VeriTile.Bench.TritonBenchG.DequantizeRowwise.Projection
open VeriTile.Triton TypedSource
set_option maxHeartbeats 4000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

def blockStart (swapped : Bool) (block : Nat) : Stmt :=
  .assign .nat [] "block_start"
    (if swapped then .mul .nat .nil (.constNat block) (.ref .nat [] "pid")
     else .mul .nat .nil (.ref .nat [] "pid") (.constNat block))

def body (swapped : Bool) (x scale out : RegionName) (inv : ℝ) (n block lanes : Nat) : List Stmt :=
  let old := (RealReference.dequantize_rowwise_kernel x scale out inv n block lanes).toAlgKernel.body
  old.take 1 ++ [blockStart swapped block] ++ old.drop 2

theorem source_body (swapped : Bool) (x scale out : RegionName) (inv : ℝ) (n block lanes : Nat) :
    ((if swapped then indexCommuted else original) x scale out inv n block lanes).eraseDType.toAlgKernel.body =
      body swapped x scale out inv n block lanes := by
  cases swapped
  all_goals simp only [original,indexCommuted,RealReference.dequantize_rowwise_kernel,body,blockStart,
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

theorem original_exec (A : FlatAlloc) (x scale out : RegionName) (inv : ℝ) (n block lanes : Nat) (s : BlockState) :
    exec (A.flattenKernel (original x scale out inv n block lanes).eraseDType.toAlgKernel) s =
      exec (A.flattenKernel (RealReference.dequantize_rowwise_kernel x scale out inv n block lanes).toAlgKernel) s := by
  change exec (A.flattenKernel ((if Bool.false then indexCommuted else original) x scale out inv n block lanes).eraseDType.toAlgKernel) s = _
  simp only [exec,FlatAlloc.flattenKernel,source_body]
  rfl

theorem index_equal (block : Nat) (u : BlockState) :
    stepStmts [blockStart Bool.false block] u = stepStmts [blockStart Bool.true block] u := by
  have h : evalOp (.mul .nat .nil (.ref .nat [] "pid") (.constNat block)) u =
      evalOp (.mul .nat .nil (.constNat block) (.ref .nat [] "pid")) u := by
    simp only [evalOp_mul,evalOp_constNat]
    cases h : evalOp (.ref .nat [] "pid") u
    · rfl
    · simp only [bind,Option.bind_some]
      congr 1
      ext i
      exact Nat.mul_comm _ _
  simp only [blockStart,Bool.false_eq_true,↓reduceIte,stepStmts,stepStmt,h]

theorem flat_index_equal (A : FlatAlloc) (block : Nat) (u : BlockState) :
    stepStmts (A.flattenStmts [blockStart Bool.false block]) u =
      stepStmts (A.flattenStmts [blockStart Bool.true block]) u := by
  simpa only [blockStart,Bool.false_eq_true,↓reduceIte,FlatAlloc.flattenStmts,
    FlatAlloc.flattenStmt,FlatAlloc.flattenOp] using index_equal block u

theorem candidate_exec (A : FlatAlloc) (x scale out : RegionName) (inv : ℝ) (n block lanes : Nat) (s : BlockState) :
    exec (A.flattenKernel (original x scale out inv n block lanes).eraseDType.toAlgKernel) s =
      exec (A.flattenKernel (indexCommuted x scale out inv n block lanes).eraseDType.toAlgKernel) s := by
  change exec (A.flattenKernel ((if Bool.false then indexCommuted else original) x scale out inv n block lanes).eraseDType.toAlgKernel) s =
    exec (A.flattenKernel ((if Bool.true then indexCommuted else original) x scale out inv n block lanes).eraseDType.toAlgKernel) s
  unfold exec FlatAlloc.flattenKernel
  rw [source_body Bool.false,source_body Bool.true]
  simp only [body,FlatAlloc.flattenStmts_append]
  exact ProgramRewrite.context _ _ _ _ (flat_index_equal A block) s

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DequantizeRowwise.Projection
