import bench.tritonbench_g.chunk_gate_recurrence.Proofs.RealForward

/-! The sum-left candidate has the same complete real execution, including
failure and full memory state. This uses real addition, independently of FP
admission. The rewrite is also preserved by the flat-memory translation. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealForwardCandidate
open VeriTile.Triton RealProgram RealForwardProgram
set_option maxRecDepth 16000
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false

def beforeUpdate (p : Layout) : List Stmt :=
  [.assign .real [] "d_i" (.load .real (.ptr (.ref .ptr [] "D")) .none),
   .assign .real [p.BK,p.BV] "S_i" (.load .real (.ptr (.ref .ptr [p.BK,p.BV] "S")) .none)]

def afterUpdate (p : Layout) : List Stmt :=
  [.store .real [p.BK,p.BV] (.ptr (.ref .ptr [p.BK,p.BV] "O")) (.ref .real [p.BK,p.BV] "acc") .none,
   .assign .ptr [] "D" (.ptrAdd .nil (.ref .ptr [] "D") (.constNat 1)),
   .assign .ptr [p.BK,p.BV] "S" (.ptrAdd .scalarR (.ref .ptr [p.BK,p.BV] "S") (stride p)),
   .assign .ptr [p.BK,p.BV] "O" (.ptrAdd .scalarR (.ref .ptr [p.BK,p.BV] "O") (stride p))]

def candidateUpdate (p : Layout) : Op .real [p.BK,p.BV] :=
  .add .real (.consSame (.consSame .nil)) (.ref .real [p.BK,p.BV] "S_i")
    (.mul .real .scalarR (.ref .real [p.BK,p.BV] "acc") (.ref .real [] "d_i"))

def candidateLoop (p : Layout) : List Stmt :=
  beforeUpdate p ++ [.assign .real [p.BK,p.BV] "acc" (candidateUpdate p)] ++ afterUpdate p

theorem original_loop_split (p : Layout) :
    loopBody p = beforeUpdate p ++ [.assign .real [p.BK,p.BV] "acc" (update p)] ++ afterUpdate p := rfl

theorem candidate_body (p : Layout) (S D O seed : RegionName) (hasSeed : Bool)
    (storage gate seedStorage : TypedSource.Storage) (H : Nat) :
    (TypedSource.forwardSumLeft storage gate seedStorage S D O seed H
      p.N p.K p.V p.BK p.BV hasSeed).eraseDType.toAlgKernel.body =
      initCode p S D O seed hasSeed ++
        [.forRangeDyn "_i" (.constNat 0) (stop p) (.constNat 1) (candidateLoop p)] := by
  rw [RealForwardInit.code_exact]
  cases storage <;> cases gate
  all_goals simp only [TypedSource.forwardSumLeft, TypedSource.specializeForward,
    ComputeKernel.surfaceBody, ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?,
    ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType, NumericDType.eraseDType,
    Except.bind, bind, List.map_cons, List.map_nil, eraseDType_float]
  all_goals simp only [VeriTile.Triton.eraseDType]
  all_goals simp [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def, eraseDType_float]
  all_goals cases seedStorage <;>
    simp [TypedSource.Storage.dtype, VeriTile.Triton.eraseDType, FloatDType.toTileDType,
      RealForwardInit.code, RealForwardInit.seedCode, RealForwardInit.zeroCode,
      RealForwardInit.pointerOp, RealForwardInit.headOp, RealForwardInit.gateOp,
      candidateLoop, candidateUpdate, beforeUpdate, afterUpdate, stop, stride]

private theorem real_commute (a b : WithBot ℝ) :
    WithBot.realAdd a b = WithBot.realAdd b a := by
  cases a <;> cases b
  · rfl
  · rfl
  · rfl
  · simp [WithBot.realAdd, add_comm]

private theorem add_equal (p : Layout) (a b : Op .real [p.BK,p.BV]) (u : BlockState) :
    stepStmts [.assign .real [p.BK,p.BV] "acc" (.add .real (.consSame (.consSame .nil)) a b)] u =
    stepStmts [.assign .real [p.BK,p.BV] "acc" (.add .real (.consSame (.consSame .nil)) b a)] u := by
  simp only [stepStmts, stepStmt, evalOp_add]
  cases ha : evalOp a u <;> cases hb : evalOp b u
  all_goals simp only [bind, Option.bind_none, Option.bind_some]
  rename_i va vb
  have hv : Tile.bop WithBot.realAdd (.consSame (.consSame .nil)) va vb =
      Tile.bop WithBot.realAdd (.consSame (.consSame .nil)) vb va := by
    ext i
    exact real_commute _ _
  simpa only [NumericDType.add] using congrArg (fun v => some (u.setReg "acc" .real [p.BK,p.BV] v)) hv

theorem loop_equal (p : Layout) (u : BlockState) :
    stepStmts (loopBody p) u = stepStmts (candidateLoop p) u := by
  rw [original_loop_split]
  exact ProgramRewrite.context (beforeUpdate p) (afterUpdate p) _ _ (add_equal p _ _) u

theorem region_exec_equal (p : Layout) (S D O seed : RegionName) (hasSeed : Bool)
    (storage gate seedStorage : TypedSource.Storage) (H : Nat) (u : BlockState) :
    exec (TypedSource.forward storage gate seedStorage S D O seed H
      p.N p.K p.V p.BK p.BV hasSeed).eraseDType.toAlgKernel u =
    exec (TypedSource.forwardSumLeft storage gate seedStorage S D O seed H
      p.N p.K p.V p.BK p.BV hasSeed).eraseDType.toAlgKernel u := by
  simp only [exec,source_body,candidate_body,statement,ProgramRewrite.append]
  have hr := ProgramRewrite.dynamic_range_congr "_i" (.constNat 0) (stop p)
    (.constNat 1) _ _ (loop_equal p)
  simp only [stepStmts,hr]

theorem flat_loop_equal (A : FlatAlloc) (p : Layout) (u : BlockState) :
    stepStmts (A.flattenStmts (loopBody p)) u = stepStmts (A.flattenStmts (candidateLoop p)) u := by
  rw [original_loop_split]
  unfold candidateLoop
  simp only [FlatAlloc.flattenStmts_append]
  apply ProgramRewrite.context
  intro v
  simp only [FlatAlloc.flattenStmts, FlatAlloc.flattenStmt, update, candidateUpdate, FlatAlloc.flattenOp]
  exact add_equal p _ _ v

theorem flat_exec_equal (A : FlatAlloc) (p : Layout) (S D O seed : RegionName) (hasSeed : Bool)
    (storage gate seedStorage : TypedSource.Storage) (H : Nat) (u : BlockState) :
    exec (A.flattenKernel (TypedSource.forward storage gate seedStorage S D O seed H
      p.N p.K p.V p.BK p.BV hasSeed).eraseDType.toAlgKernel) u =
    exec (A.flattenKernel (TypedSource.forwardSumLeft storage gate seedStorage S D O seed H
      p.N p.K p.V p.BK p.BV hasSeed).eraseDType.toAlgKernel) u := by
  simp only [exec,FlatAlloc.flattenKernel,source_body,candidate_body,statement,
    FlatAlloc.flattenStmts_append,FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,ProgramRewrite.append]
  have hr := ProgramRewrite.dynamic_range_congr "_i" (A.flattenOp (.constNat 0))
    (A.flattenOp (stop p)) (A.flattenOp (.constNat 1)) _ _ (flat_loop_equal A p)
  simp only [stepStmts,hr]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealForwardCandidate
