import bench.tritonbench_g.chunk_cumsum_vector.Proofs.RealProgram
import VeriTile.Triton.Semantics.ProgramRewrite

/-! Real commutation lifted through the branch, loop and flat-memory context.
This proof is independent of experimental FP admissions. -/
namespace VeriTile.Bench.TritonBenchG.ChunkCumsumVector.RealCandidate
open VeriTile.Triton RealProgram
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

private theorem real_commute (a b : WithBot ℝ) :
    WithBot.realAdd a b = WithBot.realAdd b a := by
  cases a <;> cases b
  · rfl
  · rfl
  · rfl
  · simp [WithBot.realAdd, add_comm]

private theorem add_equal (n : Nat) (a b : Op .real [n]) (u : BlockState) :
    stepStmts [.assign .real [n] "b_z" (.add .real (.consSame .nil) a b)] u =
    stepStmts [.assign .real [n] "b_z" (.add .real (.consSame .nil) b a)] u := by
  simp only [stepStmts, stepStmt, evalOp_add]
  cases ha : evalOp a u <;> cases hb : evalOp b u
  all_goals simp only [bind, Option.bind_none, Option.bind_some]
  rename_i a b
  have hv : Tile.bop WithBot.realAdd (.consSame .nil) a b =
      Tile.bop WithBot.realAdd (.consSame .nil) b a := by
    ext i
    exact real_commute _ _
  simpa only [NumericDType.add] using congrArg (fun v => some (u.setReg "b_z" .real [n] v)) hv

private theorem carry_equal (A : FlatAlloc) (p : Layout) (u : BlockState) :
    stepStmts [A.flattenStmt (.assign .real [p.BS] "b_z" (carryOp p Bool.false))] u =
    stepStmts [A.flattenStmt (.assign .real [p.BS] "b_z" (carryOp p Bool.true))] u := by
  simp only [carryOp, Bool.false_eq_true, ↓reduceIte, FlatAlloc.flattenStmt,
    FlatAlloc.flattenOp]
  exact add_equal p.BS _ _ u

private theorem branch_equal (A : FlatAlloc) (p : Layout) (u : BlockState) :
    stepStmts (A.flattenStmts [(.ifThen (.ge .nat .nil (.ref .nat [] "i_t") (.constNat 0))
      [.assign .real [p.BS] "b_z" (carryOp p Bool.false)])]) u =
    stepStmts (A.flattenStmts [(.ifThen (.ge .nat .nil (.ref .nat [] "i_t") (.constNat 0))
      [.assign .real [p.BS] "b_z" (carryOp p Bool.true)])]) u := by
  simp only [FlatAlloc.flattenStmts, FlatAlloc.flattenStmt]
  have hc := carry_equal A p u
  simp only [FlatAlloc.flattenStmt] at hc
  simp only [stepStmts, stepStmt, hc]

theorem loop_equal (A : FlatAlloc) (p : Layout) (s o : RegionName) (u : BlockState) :
    stepStmts (A.flattenStmts (loopBody p s o Bool.false)) u =
    stepStmts (A.flattenStmts (loopBody p s o Bool.true)) u := by
  have hsplit (b : Bool) : loopBody p s o b = (loopBody p s o Bool.false).take 5 ++
      [(.ifThen (.ge .nat .nil (.ref .nat [] "i_t") (.constNat 0))
        [.assign .real [p.BS] "b_z" (carryOp p b)])] := rfl
  rw [hsplit Bool.false, hsplit Bool.true]
  rw [FlatAlloc.flattenStmts_append, FlatAlloc.flattenStmts_append]
  exact ProgramRewrite.context (A.flattenStmts ((loopBody p s o Bool.false).take 5)) [] _ _
    (branch_equal A p) u

theorem flat_exec_equal (A : FlatAlloc) (p : Layout) (s o : RegionName)
    (dtype out_dtype : FloatDType) (u : BlockState) :
    exec (A.flattenKernel (RealSource.source Bool.false dtype out_dtype s o
      p.headStride p.timeStride p.channelStride p.T p.S p.BT p.BS).eraseDType.toAlgKernel) u =
    exec (A.flattenKernel (RealSource.source Bool.true dtype out_dtype s o
      p.headStride p.timeStride p.channelStride p.T p.S p.BT p.BS).eraseDType.toAlgKernel) u := by
  simp only [exec, FlatAlloc.flattenKernel, source_body, body, FlatAlloc.flattenStmts_append,
    FlatAlloc.flattenStmts, FlatAlloc.flattenStmt, ProgramRewrite.append]
  have hr := ProgramRewrite.dynamic_range_congr "i_t" (A.flattenOp (.constNat 0))
    (A.flattenOp (stopOp p)) (A.flattenOp (.constNat 1)) _ _ (loop_equal A p s o)
  simp only [stepStmts, hr]

private theorem carry_region_equal (p : Layout) (u : BlockState) :
    stepStmts [.assign .real [p.BS] "b_z" (carryOp p Bool.false)] u =
    stepStmts [.assign .real [p.BS] "b_z" (carryOp p Bool.true)] u := by
  simp only [carryOp, Bool.false_eq_true, ↓reduceIte]
  exact add_equal p.BS _ _ u

private theorem branch_region_equal (p : Layout) (u : BlockState) :
    stepStmts [.ifThen (.ge .nat .nil (.ref .nat [] "i_t") (.constNat 0))
      [.assign .real [p.BS] "b_z" (carryOp p Bool.false)]] u =
    stepStmts [.ifThen (.ge .nat .nil (.ref .nat [] "i_t") (.constNat 0))
      [.assign .real [p.BS] "b_z" (carryOp p Bool.true)]] u := by
  have hc := carry_region_equal p u
  simp only [stepStmts, stepStmt, hc]

theorem loop_region_equal (p : Layout) (s o : RegionName) (u : BlockState) :
    stepStmts (loopBody p s o Bool.false) u =
    stepStmts (loopBody p s o Bool.true) u := by
  have hsplit (b : Bool) : loopBody p s o b = (loopBody p s o Bool.false).take 5 ++
      [.ifThen (.ge .nat .nil (.ref .nat [] "i_t") (.constNat 0))
        [.assign .real [p.BS] "b_z" (carryOp p b)]] := rfl
  rw [hsplit Bool.false, hsplit Bool.true]
  exact ProgramRewrite.context ((loopBody p s o Bool.false).take 5) [] _ _
    (branch_region_equal p) u

/-- Full real state equality, retaining arbitrary initial unspecified values. -/
theorem region_exec_equal (p : Layout) (s o : RegionName)
    (dtype out_dtype : FloatDType) (u : BlockState) :
    exec (RealSource.source Bool.false dtype out_dtype s o
      p.headStride p.timeStride p.channelStride p.T p.S p.BT p.BS).eraseDType.toAlgKernel u =
    exec (RealSource.source Bool.true dtype out_dtype s o
      p.headStride p.timeStride p.channelStride p.T p.S p.BT p.BS).eraseDType.toAlgKernel u := by
  simp only [exec, source_body, body, ProgramRewrite.append]
  have hr := ProgramRewrite.dynamic_range_congr "i_t" (.constNat 0)
    (stopOp p) (.constNat 1) _ _ (loop_region_equal p s o)
  simp only [stepStmts, hr]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ChunkCumsumVector.RealCandidate
