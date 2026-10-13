import bench.tritonbench_g.kldiv_ops.Proofs.ForwardElementLoop
import bench.tritonbench_g.kldiv_ops.Proofs.Initialize
import bench.tritonbench_g.kldiv_ops.Proofs.Finish

namespace VeriTile.Bench.TritonBenchG.KldivOps.FullForward
open VeriTile.Triton FullAlgorithm

/-- Elementwise execution retains the six initial assignments, whole loop and final branch. -/
theorem element_source_decomposition (Y T O : RegionName) (sy st so N B : Nat)
    (eps : ℝ) (logTarget : Bool) :
    (forward Y T O sy st so N B eps logTarget 0).body =
      (forward Y T O sy st so N B eps logTarget 0).body.take 6 ++
        [.forRangeInt "i" (.constInt 0) (.constInt N) (.constNat B)
          (body Y T O sy st so N B eps logTarget 0)] ++
        (forward Y T O sy st so N B eps logTarget 0).body.drop 7 := by
  rfl

theorem element_initial_inv (s : BlockState) (Y T O : RegionName)
    (sy st so N B : Nat) (eps : ℝ) (logTarget : Bool) :
    ElementInv s Y T O (s.pids 0 * sy) (s.pids 0 * st) (s.pids 0 * so)
      N B 0 eps logTarget (forwardInitial s Y T O sy st so B) := by
  constructor
  · intro r k _; rfl
  · intro k; rfl
  · intro k; rfl
  · rfl
  · rfl
  · rfl
  · simp [forwardInitial]
  · simp [forwardInitial]
  · simp [forwardInitial]
  · simp [forwardInitial]
  · simp [forwardInitial]
  · intro t lane ht; omega

/-- Every active column receives its independent loss; all other raw memory is preserved. -/
theorem element_source_run (s : BlockState) (Y T O : RegionName)
    (sy st so N B : Nat) (eps : ℝ) (logTarget : Bool)
    (hB : 0 < B) (hYO : Y ≠ O) (hTO : T ≠ O) :
    ∃ final, exec (forward Y T O sy st so N B eps logTarget 0) s = some final ∧
      (∀ k, k < N → final.readMem O (s.pids 0 * so + k) =
        FullMath.loss logTarget eps (s.readMem Y (s.pids 0 * sy + k))
          (s.readMem T (s.pids 0 * st + k))) ∧
      BlockState.WriteWithin (elementFootprint O (s.pids 0 * so) N) s final := by
  let initial := forwardInitial s Y T O sy st so B
  have hi := element_initial_inv s Y T O sy st so N B eps logTarget
  obtain ⟨covered, final, hcover, he, hf⟩ :=
    element_loop_run s Y T O sy st so (s.pids 0 * sy) (s.pids 0 * st)
      (s.pids 0 * so) N B eps logTarget hB hYO hTO 0 initial hi
  have hloop : stepStmts
      [.forRangeInt "i" (.constInt 0) (.constInt N) (.constNat B)
        (body Y T O sy st so N B eps logTarget 0)] initial = some final := by
    simp only [Nat.zero_mul, Nat.cast_zero] at he
    simp only [stepStmts, stepStmt, evalOp.eq_def, Option.bind_eq_bind, Option.bind,
      Tile.scalar_data, he]
  refine ⟨final, ?_, hf.complete hB hcover, hf.frame⟩
  rw [exec, element_source_decomposition, List.append_assoc,
    stepStmts.append_some (forward_initialize s Y T O sy st so N B eps logTarget 0),
    stepStmts.append_some hloop]
  simpa only [ite_true] using
    forward_finish final Y T O sy st so N B (s.pids 0 * so) eps 0 logTarget 0 hf.optr hf.acc

end VeriTile.Bench.TritonBenchG.KldivOps.FullForward

#guard_msgs (drop info) in
#auditModuleAxioms
