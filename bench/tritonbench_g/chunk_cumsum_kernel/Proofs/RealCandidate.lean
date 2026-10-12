import bench.tritonbench_g.chunk_cumsum_kernel.Proofs.RealSource
import VeriTile.Triton.Semantics.ProgramRewrite

/-! A real addition rewrite lifted through the complete range and flat-memory
context. This proof is independent of FP admissions and keeps the final state. -/
namespace VeriTile.Bench.TritonBenchG.ChunkCumsumKernel.RealCandidate
open VeriTile.Triton
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

private def carry (swapped : Bool) : Stmt :=
  if swapped then .assign .real [] "b_z"
    (.add .real .nil (.ref .real [] "b_zz") (.ref .real [] "b_z"))
  else .assign .real [] "b_z"
    (.add .real .nil (.ref .real [] "b_z") (.ref .real [] "b_zz"))

def loopBody (swapped : Bool) (s o : RegionName) (T BT : Nat) : List Stmt :=
  (ccLoopBody s o T BT).take 5 ++ carry swapped :: (ccLoopBody s o T BT).drop 6

def body (swapped : Bool) (s o : RegionName) (T BT : Nat) : List Stmt :=
  [.assign .nat [] "i_bh" (.programId 0), .assign .real [] "b_z" (.full [] (.const 0)),
    .forRangeDyn "i_t" (.constNat 0) (ccStopOp T BT) (.constNat 1)
      (loopBody swapped s o T BT)]

theorem source_body (swapped : Bool) (dtype out_dtype : FloatDType)
    (s o : RegionName) (T BT : Nat) :
    (RealSource.source swapped dtype out_dtype s o T BT).eraseDType.toAlgKernel.body =
      body swapped s o T BT := by
  rw [RealSource.storage_erasure]
  cases swapped
  all_goals simp only [RealSource.source, Bool.false_eq_true, ↓reduceIte,
    TypedSource.original, TypedSource.sumLeft, ComputeKernel.surfaceBody,
    ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    List.map_cons, List.map_nil, Kernel.eraseDType, Stmt.eraseDTypeList,
    Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    NumericDType.eraseDType, Except.bind, bind, eraseDType_float]
  all_goals simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType, FloatDType.toTileDType]
  all_goals rfl

private theorem real_commute (a b : WithBot ℝ) :
    WithBot.realAdd a b = WithBot.realAdd b a := by
  cases a <;> cases b
  · rfl
  · rfl
  · rfl
  · simp [WithBot.realAdd, add_comm]

private theorem carry_region_equal (s : BlockState) :
    stepStmts [carry Bool.false] s = stepStmts [carry Bool.true] s := by
  simp only [carry, Bool.false_eq_true, ↓reduceIte, stepStmts, stepStmt, evalOp_add]
  cases ha : evalOp (.ref .real [] "b_z") s <;> cases hb : evalOp (.ref .real [] "b_zz") s
  all_goals simp only [bind, Option.bind_none, Option.bind_some]
  rename_i a b
  have hv : Tile.bop WithBot.realAdd .nil a b = Tile.bop WithBot.realAdd .nil b a := by
    ext i
    exact real_commute _ _
  simpa only [NumericDType.add] using congrArg (fun v => some (s.setReg "b_z" .real [] v)) hv

private theorem carry_equal (A : FlatAlloc) (s : BlockState) :
    stepStmts [A.flattenStmt (carry Bool.false)] s =
    stepStmts [A.flattenStmt (carry Bool.true)] s := by
  simpa only [carry, Bool.false_eq_true, ↓reduceIte, FlatAlloc.flattenStmt,
    FlatAlloc.flattenOp] using carry_region_equal s

theorem loop_region_equal (s o : RegionName) (T BT : Nat) (u : BlockState) :
    stepStmts (loopBody Bool.false s o T BT) u =
    stepStmts (loopBody Bool.true s o T BT) u := by
  unfold loopBody
  simpa only [List.append_assoc, List.singleton_append] using
    ProgramRewrite.context ((ccLoopBody s o T BT).take 5)
      ((ccLoopBody s o T BT).drop 6) _ _ carry_region_equal u

/-- The complete real executions agree for arbitrary initial unspecified values. -/
theorem region_exec_equal (dtype out_dtype : FloatDType)
    (s o : RegionName) (T BT : Nat) (u : BlockState) :
    exec (RealSource.source Bool.false dtype out_dtype s o T BT).eraseDType.toAlgKernel u =
    exec (RealSource.source Bool.true dtype out_dtype s o T BT).eraseDType.toAlgKernel u := by
  simp only [exec, source_body, body]
  have hr := ProgramRewrite.dynamic_range_congr "i_t" (.constNat 0)
    (ccStopOp T BT) (.constNat 1) _ _ (loop_region_equal s o T BT)
  simp only [stepStmts]
  simp only [hr]

theorem loop_equal (A : FlatAlloc) (s o : RegionName) (T BT : Nat) (u : BlockState) :
    stepStmts (A.flattenStmts (loopBody Bool.false s o T BT)) u =
    stepStmts (A.flattenStmts (loopBody Bool.true s o T BT)) u := by
  unfold loopBody
  simp only [FlatAlloc.flattenStmts_append, FlatAlloc.flattenStmts]
  simpa only [List.append_assoc, List.singleton_append] using
    ProgramRewrite.context (A.flattenStmts ((ccLoopBody s o T BT).take 5))
      (A.flattenStmts ((ccLoopBody s o T BT).drop 6)) _ _ (carry_equal A) u

theorem flat_exec_equal (A : FlatAlloc) (dtype out_dtype : FloatDType)
    (s o : RegionName) (T BT : Nat) (u : BlockState) :
    exec (A.flattenKernel (RealSource.source Bool.false dtype out_dtype s o T BT).eraseDType.toAlgKernel) u =
    exec (A.flattenKernel (RealSource.source Bool.true dtype out_dtype s o T BT).eraseDType.toAlgKernel) u := by
  simp only [exec, FlatAlloc.flattenKernel, source_body, body,
    FlatAlloc.flattenStmts, FlatAlloc.flattenStmt]
  have hr := ProgramRewrite.dynamic_range_congr "i_t" (A.flattenOp (.constNat 0))
    (A.flattenOp (ccStopOp T BT)) (A.flattenOp (.constNat 1)) _ _ (loop_equal A s o T BT)
  simp only [stepStmts]
  simp only [hr]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ChunkCumsumKernel.RealCandidate
