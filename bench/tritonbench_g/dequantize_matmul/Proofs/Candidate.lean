import bench.tritonbench_g.dequantize_matmul.Proofs.Projection
import VeriTile.Triton.Semantics.ProgramRewrite
namespace VeriTile.Bench.TritonBenchG.DequantizeMatmul.Candidate
open VeriTile.Triton TypedSource
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000

def mask (swapped : Bool) (BN N : Nat) : Stmt :=
  .assign .bool [1,BN] "n_mask" (.lt .nat .scalarR
    (.add .nat .scalarL
      (if swapped then .mul .nat .nil (.constNat BN) (.ref .nat [] "n_block_idx")
       else .mul .nat .nil (.ref .nat [] "n_block_idx") (.constNat BN))
      (.expandDim (shape := [BN]) ⟨0,by simp⟩ (.ref .nat [BN] "offs_n"))) (.constNat N))

def body (swapped : Bool) (b scale out : RegionName)
    (K N sbk sbn sok son BN BK : Nat) : List Stmt :=
  let old := (RealReference.dequantize_kernel b scale out K N sbk sbn sok son BN BK).toAlgKernel.body
  old.take 7 ++ [mask swapped BN N] ++ old.drop 8

theorem source_body (output : Storage) (swapped : Bool) (b scale out : RegionName)
    (K N sbk sbn sok son BN BK : Nat) :
    (kernel output swapped b scale out K N sbk sbn sok son BN BK).eraseDType.toAlgKernel.body =
      body swapped b scale out K N sbk sbn sok son BN BK := by
  cases output <;> cases swapped
  all_goals simp only [kernel,body,mask,RealReference.dequantize_kernel,
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

theorem mask_equal (BN N : Nat) (u : BlockState) :
    stepStmts [mask Bool.false BN N] u = stepStmts [mask Bool.true BN N] u := by
  have hmul : evalOp (.mul .nat .nil (.ref .nat [] "n_block_idx") (.constNat BN)) u =
      evalOp (.mul .nat .nil (.constNat BN) (.ref .nat [] "n_block_idx")) u := by
    simp only [evalOp_mul,evalOp_constNat]
    cases h : evalOp (.ref .nat [] "n_block_idx") u
    · rfl
    · simp only [bind,Option.bind_some]
      congr 1
      ext i
      exact Nat.mul_comm _ _
  simp only [mask,Bool.false_eq_true,↓reduceIte,stepStmts,stepStmt,
    evalOp_lt,evalOp_add,hmul]

theorem flat_mask_equal (A : FlatAlloc) (BN N : Nat) (u : BlockState) :
    stepStmts (A.flattenStmts [mask Bool.false BN N]) u =
      stepStmts (A.flattenStmts [mask Bool.true BN N]) u := by
  simpa only [mask,Bool.false_eq_true,↓reduceIte,FlatAlloc.flattenStmts,
    FlatAlloc.flattenStmt,FlatAlloc.flattenOp,FlatAlloc.flattenOp.eq_def] using mask_equal BN N u

theorem flat_exec_equal (A : FlatAlloc) (output : Storage) (b scale out : RegionName)
    (K N sbk sbn sok son BN BK : Nat) (u : BlockState) :
    exec (A.flattenKernel (kernel output Bool.false b scale out K N sbk sbn sok son BN BK).eraseDType.toAlgKernel) u =
      exec (A.flattenKernel (kernel output Bool.true b scale out K N sbk sbn sok son BN BK).eraseDType.toAlgKernel) u := by
  simp only [exec,FlatAlloc.flattenKernel,source_body,body,FlatAlloc.flattenStmts_append]
  exact ProgramRewrite.context _ _ _ _ (flat_mask_equal A BN N) u

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DequantizeMatmul.Candidate
