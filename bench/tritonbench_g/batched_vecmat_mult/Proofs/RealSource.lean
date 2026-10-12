import bench.tritonbench_g.batched_vecmat_mult.Kernels
import bench.tritonbench_g.batched_vecmat_mult.Proofs.RealFoundation
import VeriTile.Triton.Semantics.ProgramRewrite
import VeriTile.Meta.StatementAudit

/-! Exact real correspondence for the complete sources. The original erases
to the existing real execution foundation. Real addition commutes in every
loop state, so the candidate has the same full flat-memory execution result,
including failures. This bridge imports no FP admission or FP proof. -/
namespace VeriTile.Bench.TritonBenchG.BatchedVecmatMult.RealSource
open VeriTile.Triton
set_option maxHeartbeats 1800000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

section
variable (A B out : RegionName) (M N K BM BN BK : Nat)

def source (candidate : Bool) : ComputeKernel :=
  (if candidate then TypedSource.blockFirst else TypedSource.original) A B out M N K BM BN BK

def kernel (candidate : Bool) : ComputeKernel := (source A B out M N K BM BN BK candidate).eraseDType

theorem source_projection (candidate : Bool) :
    (source A B out M N K BM BN BK candidate).toAlgorithm? =
      .ok (source A B out M N K BM BN BK candidate).toAlgKernel := by
  cases candidate <;> rfl

theorem projection (candidate : Bool) :
    (kernel A B out M N K BM BN BK candidate).toAlgorithm? =
      .ok (kernel A B out M N K BM BN BK candidate).toAlgKernel := by
  simp only [kernel, ComputeKernel.eraseDType, source_projection]
  simp

theorem original_body :
    (kernel A B out M N K BM BN BK Bool.false).toAlgKernel.body =
      (batched_vecmat_surface A B out M N K BM BN BK).toAlgKernel.body := by
  simp only [kernel, source, Bool.false_eq_true, ↓reduceIte, TypedSource.original,
    batched_vecmat_surface, ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?,
    ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    IntegralDType.eraseDType, Except.bind, bind]
  rfl

end

def blockSum (BM BN BK : Nat) : Op .real [BM,BN] :=
  .transpose (batch := []) (.reduceSum ⟨2, by simp⟩ Bool.false
    (.mul .real (.consL (.consSame (.consSame .nil)))
      (.ref .real [1,BM,BK] "expanded_a") (.ref .real [BN,BM,BK] "b")))

def update (BM BN BK : Nat) (candidate : Bool) : Stmt :=
  .assign .real [BM,BN] "vecmat"
    (if candidate then .add .real (.consSame (.consSame .nil))
      (blockSum BM BN BK) (.ref .real [BM,BN] "vecmat")
    else .add .real (.consSame (.consSame .nil))
      (.ref .real [BM,BN] "vecmat") (blockSum BM BN BK))

def loop (A B : RegionName) (N K BM BN BK : Nat) (candidate : Bool) : List Stmt :=
  (vecmatLoopBody A B N K BM BN BK).take 5 ++ [update BM BN BK candidate]

private theorem loop_original (A B : RegionName) (N K BM BN BK : Nat) :
    loop A B N K BM BN BK Bool.false = vecmatLoopBody A B N K BM BN BK := rfl

theorem source_body (A B out : RegionName) (M N K BM BN BK : Nat) (candidate : Bool) :
    (kernel A B out M N K BM BN BK candidate).toAlgKernel.body =
      (batched_vecmat_surface A B out M N K BM BN BK).toAlgKernel.body.take 5 ++
        [.forRangeDyn "k_index" (.constNat 0) (.ref .nat [] "k_blocks") (.constNat 1)
          (loop A B N K BM BN BK candidate), vecmatStoreStmt out BM BN] := by
  cases candidate
  · rw [original_body, loop_original]
    exact vecmat_body_split A B out M N K BM BN BK
  · simp only [kernel, source, ↓reduceIte, TypedSource.blockFirst,
      batched_vecmat_surface, ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?,
      ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
      ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
      Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
      IntegralDType.eraseDType, Except.bind, bind]
    rfl

private theorem real_add_comm (a b : WithBot ℝ) : WithBot.realAdd a b = WithBot.realAdd b a := by
  cases a <;> cases b
  · rfl
  · rfl
  · rfl
  · simp [WithBot.realAdd, add_comm]

private theorem eval_add_comm (a b : Op .real [BM,BN]) (s : BlockState) :
    evalOp (.add .real (.consSame (.consSame .nil)) a b) s =
      evalOp (.add .real (.consSame (.consSame .nil)) b a) s := by
  simp only [evalOp_add]
  cases ha : evalOp a s <;> cases hb : evalOp b s
  all_goals simp only [Option.bind_eq_bind, Option.bind_none, Option.bind_some]
  rename_i va vb
  congr 1
  ext i
  exact real_add_comm _ _

private theorem flatten_append (alloc : FlatAlloc) (xs ys : List Stmt) :
    alloc.flattenStmts (xs ++ ys) = alloc.flattenStmts xs ++ alloc.flattenStmts ys := by
  induction xs with
  | nil => rfl
  | cons x xs ih => simp only [List.cons_append, FlatAlloc.flattenStmts, ih]

private theorem flatten_update (alloc : FlatAlloc) (BM BN BK : Nat) (candidate : Bool) :
    alloc.flattenStmt (update BM BN BK candidate) = update BM BN BK candidate := by
  cases candidate <;> simp [update, blockSum, FlatAlloc.flattenStmt, FlatAlloc.flattenOp.eq_def]

theorem flat_loop (alloc : FlatAlloc) (A B : RegionName) (N K BM BN BK : Nat) (s : BlockState) :
    stepStmts (alloc.flattenStmts (loop A B N K BM BN BK Bool.true)) s =
      stepStmts (alloc.flattenStmts (loop A B N K BM BN BK Bool.false)) s := by
  simp only [loop, flatten_append, FlatAlloc.flattenStmts, flatten_update, ProgramRewrite.append]
  congr 1
  funext t
  simp only [stepStmts, stepStmt, update, Bool.false_eq_true, ↓reduceIte,
    eval_add_comm (blockSum BM BN BK) (.ref .real [BM,BN] "vecmat")]

theorem candidate_flat_execution (alloc : FlatAlloc) (A B out : RegionName)
    (M N K BM BN BK : Nat) (s : BlockState) :
    exec (alloc.flattenKernel (kernel A B out M N K BM BN BK Bool.true).toAlgKernel) s =
      exec (alloc.flattenKernel (kernel A B out M N K BM BN BK Bool.false).toAlgKernel) s := by
  simp only [exec, FlatAlloc.flattenKernel, source_body, flatten_append, ProgramRewrite.append]
  congr 1
  funext t
  simp only [FlatAlloc.flattenStmts, FlatAlloc.flattenStmt, stepStmts,
    ProgramRewrite.dynamic_range_congr _ _ _ _ _ _ (flat_loop alloc A B N K BM BN BK)]

theorem original_flat_source (alloc : FlatAlloc) (A B out : RegionName) (M N K BM BN BK : Nat) :
    alloc.flattenKernel (kernel A B out M N K BM BN BK Bool.false).toAlgKernel =
      alloc.flattenKernel (batched_vecmat_surface A B out M N K BM BN BK).toAlgKernel := by
  simp only [FlatAlloc.flattenKernel, original_body]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.BatchedVecmatMult.RealSource
