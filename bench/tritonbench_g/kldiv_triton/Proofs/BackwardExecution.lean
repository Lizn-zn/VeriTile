import bench.tritonbench_g.kldiv_triton.Proofs.BackwardLoop
import bench.tritonbench_g.kldiv_triton.Proofs.Initialize

namespace VeriTile.Bench.TritonBenchG.KldivTriton.FullBackward
open VeriTile.Triton FullAlgorithm

/-- The actual backward source consists of its five initial assignments and complete loop. -/
theorem source_decomposition (T O : RegionName) (st so N B : Nat) (logTarget : Bool) :
    (backward T O st so N B logTarget).body =
      (backward T O st so N B logTarget).body.take 5 ++
        [.forRangeInt "i" (.constInt 0) (.constInt N) (.constNat B)
          (body T O st so N B logTarget)] := by
  cases logTarget <;> rfl

theorem initial_inv (s : BlockState) (T O : RegionName)
    (st so N B : Nat) (logTarget : Bool) :
    Inv s T O (s.pids 0 * st) (s.pids 0 * so) N B 0 logTarget
      (backwardInitial s T O st so N B) := by
  constructor
  · intro r k _; rfl
  · intro k; rfl
  · rfl
  · rfl
  · rfl
  · simp [backwardInitial]
  · simp [backwardInitial]
  · intro t lane ht; omega

/-- Complete backward algorithm execution, including initialization and all column blocks. -/
theorem source_run (s : BlockState) (T O : RegionName)
    (st so N B : Nat) (logTarget : Bool) (hB : 0 < B) (hTO : T ≠ O) :
    ∃ final, exec (backward T O st so N B logTarget) s = some final ∧
      (∀ k, k < N → final.readMem O (s.pids 0 * so + k) =
        gradient logTarget (s.readMem T (s.pids 0 * st + k))) ∧
      BlockState.WriteWithin (footprint O (s.pids 0 * so) N) s final := by
  let initial := backwardInitial s T O st so N B
  have hi := initial_inv s T O st so N B logTarget
  obtain ⟨covered, final, hcover, he, hf⟩ :=
    loop_run s T O st so (s.pids 0 * st) (s.pids 0 * so) N B logTarget hB hTO
      0 initial hi
  refine ⟨final, ?_, hf.complete hB hcover, hf.frame⟩
  rw [exec, source_decomposition,
    stepStmts.append_some (backward_initialize s T O st so N B logTarget)]
  simp only [initial, Nat.zero_mul, Nat.cast_zero] at he
  simp only [stepStmts, stepStmt, evalOp.eq_def, Option.bind_eq_bind, Option.bind,
    Tile.scalar_data, he]

end VeriTile.Bench.TritonBenchG.KldivTriton.FullBackward

#guard_msgs (drop info) in
#auditModuleAxioms
