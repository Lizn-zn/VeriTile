import bench.tritonbench_g.chunk_gate_recurrence.Proofs.RealBackward

/-! Complete real correspondence for the sum-left reverse candidate. Both the
loop update and the DL postlude commute real addition. The two programs have
the same full state or failure, in region memory and after flat translation;
this result uses no FP admission rule. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealBackwardCandidate
open VeriTile.Triton RealProgram RealBackwardProgram
open RealForwardProgram (stride stop)
set_option maxRecDepth 16000
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false

def beforeUpdate (p : Layout) : List Stmt :=
  [.assign .real [p.BK,p.BV] "S_i" (.load .real (.ptr (.ref .ptr [p.BK,p.BV] "S")) .none),
   .assign .real [p.BK,p.BV] "DS_i" (.load .real (.ptr (.ref .ptr [p.BK,p.BV] "DS")) .none),
   .assign .real [] "d_i" (.load .real (.ptr (.ref .ptr [] "D")) .none)]

def afterUpdate (p : Layout) : List Stmt :=
  [.assign .real [] "DG_i" (gradient p),
   .store .real [] (.ptr (.ref .ptr [] "DG")) (.ref .real [] "DG_i") .none,
   .store .real [p.BK,p.BV] (.ptr (.ref .ptr [p.BK,p.BV] "DI")) (.ref .real [p.BK,p.BV] "Dacc") .none,
   .assign .ptr [p.BK,p.BV] "S" (.ptrSub .scalarR (.ref .ptr [p.BK,p.BV] "S") (stride p)),
   .assign .ptr [p.BK,p.BV] "DI" (.ptrSub .scalarR (.ref .ptr [p.BK,p.BV] "DI") (stride p)),
   .assign .ptr [p.BK,p.BV] "DS" (.ptrSub .scalarR (.ref .ptr [p.BK,p.BV] "DS") (stride p)),
   .assign .ptr [] "DG" (.ptrSub .nil (.ref .ptr [] "DG") gradientStride),
   .assign .ptr [] "D" (.ptrSub .nil (.ref .ptr [] "D") (.constNat 1))]

def candidateUpdate (p : Layout) : Op .real [p.BK,p.BV] :=
  .add .real (.consSame (.consSame .nil)) (.ref .real [p.BK,p.BV] "DS_i")
    (.mul .real .scalarR (.ref .real [p.BK,p.BV] "Dacc") (.ref .real [] "d_i"))

def candidateLoop (p : Layout) : List Stmt :=
  beforeUpdate p ++ [.assign .real [p.BK,p.BV] "Dacc" (candidateUpdate p)] ++ afterUpdate p

def beforePostUpdate (p : Layout) (DL : RegionName) : List Stmt :=
  [.assign .ptr [p.BK,p.BV] "DL" (RealBackwardInit.seedPointerOp p DL),
   .assign .real [p.BK,p.BV] "DS_i" (.load .real (.ptr (.ref .ptr [p.BK,p.BV] "DS")) .none),
   .assign .real [] "d_i" (.load .real (.ptr (.ref .ptr [] "D")) .none)]

def afterPostUpdate (p : Layout) : List Stmt :=
  [.store .real [p.BK,p.BV] (.ptr (.ref .ptr [p.BK,p.BV] "DL")) (.ref .real [p.BK,p.BV] "Dacc") .none]

def candidatePost (p : Layout) (DL : RegionName) : List Stmt :=
  beforePostUpdate p DL ++ [.assign .real [p.BK,p.BV] "Dacc" (candidateUpdate p)] ++ afterPostUpdate p

theorem original_loop_split (p : Layout) :
    loopBody p = beforeUpdate p ++ [.assign .real [p.BK,p.BV] "Dacc" (update p)] ++ afterUpdate p := rfl

theorem original_post_split (p : Layout) (DL : RegionName) :
    RealBackwardPost.code p DL = beforePostUpdate p DL ++
      [.assign .real [p.BK,p.BV] "Dacc" (update p)] ++ afterPostUpdate p := rfl

theorem candidate_body (p : Layout) (S D DI DG DL DS : RegionName)
    (storage gate incoming : TypedSource.Storage) (H : Nat) :
    (TypedSource.backwardSumLeft storage gate incoming S D DI DG DL DS H
      p.N p.K p.V p.BK p.BV).eraseDType.toAlgKernel.body =
      initCode p S D DI DG DL DS ++
        [.forRangeDyn "_i" (.constNat 0) (stop p) (.constNat 1) (candidateLoop p)] ++ candidatePost p DL := by
  rw [RealBackwardInit.code_exact]
  cases gate <;> cases incoming
  all_goals simp only [TypedSource.backwardSumLeft,TypedSource.specializeBackward,
    ComputeKernel.surfaceBody,ComputeKernel.eraseDType,ComputeKernel.toAlgorithm?,
    ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,NumericDType.eraseDType,
    Except.bind,bind,List.map_cons,List.map_nil,eraseDType_float]
  all_goals simp only [VeriTile.Triton.eraseDType]
  all_goals simp [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def,eraseDType_float]
  all_goals cases storage <;>
    simp [TypedSource.Storage.dtype,VeriTile.Triton.eraseDType,FloatDType.toTileDType,
      RealBackwardInit.prepareCode,RealBackwardInit.pointerCode,RealBackwardInit.pointerOp,
      RealBackwardInit.seedPointerOp,RealBackwardInit.offsetOp,RealBackwardInit.headOp,
      RealBackwardInit.gateOp,RealBackwardInit.gradientOp,RealBackwardInit.startOp,
      candidateLoop,candidatePost,candidateUpdate,beforeUpdate,afterUpdate,beforePostUpdate,
      afterPostUpdate,stop,stride,gradient,gradientStride,IntegralDType.eraseDType,NumericDType.eraseDType]

private theorem real_commute (a b : WithBot ℝ) :
    WithBot.realAdd a b = WithBot.realAdd b a := by
  cases a <;> cases b
  · rfl
  · rfl
  · rfl
  · simp [WithBot.realAdd, add_comm]

private theorem add_equal (p : Layout) (a b : Op .real [p.BK,p.BV]) (u : BlockState) :
    stepStmts [.assign .real [p.BK,p.BV] "Dacc" (.add .real (.consSame (.consSame .nil)) a b)] u =
    stepStmts [.assign .real [p.BK,p.BV] "Dacc" (.add .real (.consSame (.consSame .nil)) b a)] u := by
  simp only [stepStmts, stepStmt, evalOp_add]
  cases ha : evalOp a u <;> cases hb : evalOp b u
  all_goals simp only [bind, Option.bind_none, Option.bind_some]
  rename_i va vb
  have hv : Tile.bop WithBot.realAdd (.consSame (.consSame .nil)) va vb =
      Tile.bop WithBot.realAdd (.consSame (.consSame .nil)) vb va := by
    ext i
    exact real_commute _ _
  simpa only [NumericDType.add] using congrArg (fun v => some (u.setReg "Dacc" .real [p.BK,p.BV] v)) hv


theorem loop_equal (p : Layout) (u : BlockState) :
    stepStmts (loopBody p) u = stepStmts (candidateLoop p) u := by
  rw [original_loop_split]
  exact ProgramRewrite.context (beforeUpdate p) (afterUpdate p) _ _ (add_equal p _ _) u

theorem post_equal (p : Layout) (DL : RegionName) (u : BlockState) :
    stepStmts (RealBackwardPost.code p DL) u = stepStmts (candidatePost p DL) u := by
  rw [original_post_split]
  exact ProgramRewrite.context (beforePostUpdate p DL) (afterPostUpdate p) _ _ (add_equal p _ _) u

theorem region_exec_equal (p : Layout) (S D DI DG DL DS : RegionName)
    (storage gate incoming : TypedSource.Storage) (H : Nat) (u : BlockState) :
    exec (TypedSource.backward storage gate incoming S D DI DG DL DS H
      p.N p.K p.V p.BK p.BV).eraseDType.toAlgKernel u =
    exec (TypedSource.backwardSumLeft storage gate incoming S D DI DG DL DS H
      p.N p.K p.V p.BK p.BV).eraseDType.toAlgKernel u := by
  simp only [exec,source_body,candidate_body,statement,RealBackwardPost.code_exact,ProgramRewrite.append]
  have hr := ProgramRewrite.dynamic_range_congr "_i" (.constNat 0) (stop p)
    (.constNat 1) _ _ (loop_equal p)
  simp only [stepStmts,hr,post_equal]

theorem flat_loop_equal (A : FlatAlloc) (p : Layout) (u : BlockState) :
    stepStmts (A.flattenStmts (loopBody p)) u = stepStmts (A.flattenStmts (candidateLoop p)) u := by
  rw [original_loop_split]
  unfold candidateLoop
  simp only [FlatAlloc.flattenStmts_append]
  apply ProgramRewrite.context
  intro v
  simp only [FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,update,candidateUpdate,FlatAlloc.flattenOp]
  exact add_equal p _ _ v

theorem flat_post_equal (A : FlatAlloc) (p : Layout) (DL : RegionName) (u : BlockState) :
    stepStmts (A.flattenStmts (RealBackwardPost.code p DL)) u =
      stepStmts (A.flattenStmts (candidatePost p DL)) u := by
  rw [original_post_split]
  unfold candidatePost
  simp only [FlatAlloc.flattenStmts_append]
  apply ProgramRewrite.context
  intro v
  simp only [FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,update,candidateUpdate,FlatAlloc.flattenOp]
  exact add_equal p _ _ v

theorem flat_exec_equal (A : FlatAlloc) (p : Layout) (S D DI DG DL DS : RegionName)
    (storage gate incoming : TypedSource.Storage) (H : Nat) (u : BlockState) :
    exec (A.flattenKernel (TypedSource.backward storage gate incoming S D DI DG DL DS H
      p.N p.K p.V p.BK p.BV).eraseDType.toAlgKernel) u =
    exec (A.flattenKernel (TypedSource.backwardSumLeft storage gate incoming S D DI DG DL DS H
      p.N p.K p.V p.BK p.BV).eraseDType.toAlgKernel) u := by
  simp only [exec,FlatAlloc.flattenKernel,source_body,candidate_body,statement,RealBackwardPost.code_exact,
    FlatAlloc.flattenStmts_append,FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,ProgramRewrite.append]
  have hr := ProgramRewrite.dynamic_range_congr "_i" (A.flattenOp (.constNat 0))
    (A.flattenOp (stop p)) (A.flattenOp (.constNat 1)) _ _ (flat_loop_equal A p)
  simp only [stepStmts,hr,flat_post_equal]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealBackwardCandidate
