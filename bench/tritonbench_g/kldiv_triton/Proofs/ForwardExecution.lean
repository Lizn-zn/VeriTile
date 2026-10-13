import bench.tritonbench_g.kldiv_triton.Proofs.ForwardLoop
import bench.tritonbench_g.kldiv_triton.Proofs.Initialize

namespace VeriTile.Bench.TritonBenchG.KldivTriton.FullForward
open VeriTile.Triton FullAlgorithm

/-- The actual five-statement initialization precedes the complete signed column loop. -/
theorem source_decomposition (Y T O : RegionName) (sy st so N B : Nat)
    (flag : Bool) (reduction : Nat) :
    (forward Y T O sy st so N B flag reduction).body =
      (forward Y T O sy st so N B flag reduction).body.take 5 ++
        [.forRangeInt "i" (.constInt 0) (.constInt N) (.constNat B)
          (body Y T O sy st so N B flag reduction)] := by
  rfl

theorem initial_inv (s : BlockState) (Y T O : RegionName) (sy st so N B : Nat)
    (flag : Bool) (reduction : Nat) :
    Inv s Y T O (s.pids 0 * sy) (s.pids 0 * st) (s.pids 0 * so)
      N B 0 reduction flag (forwardInitial s Y T O sy st so B) := by
  constructor
  · constructor
    · intro r k _; rfl
    · intro k; rfl
    · intro k; rfl
    · rfl
    · rfl
    · rfl
    · simp [forwardInitial]
    · simp [forwardInitial]
    · simp [forwardInitial, outputBase]
    · simp [forwardInitial]
  · intro _ t lane ht; omega
  · intro _ t ht; omega

/-- Full source execution, independent output formula and every raw memory cell's frame.
Reduced outputs are indexed by chunk; N=0 produces an empty write footprint. -/
theorem source_run (s : BlockState) (Y T O : RegionName) (sy st so N B : Nat)
    (flag : Bool) (reduction : Nat) (hB : 0 < B) (hYO : Y ≠ O) (hTO : T ≠ O) :
    ∃ final, exec (forward Y T O sy st so N B flag reduction) s = some final ∧
      (∀ k, k < FullMath.outputCount N B reduction →
        final.readMem O (s.pids 0 * so + k) =
          FullMath.forwardOutput flag (fun i ↦ s.readMem Y (s.pids 0 * sy + i))
            (fun i ↦ s.readMem T (s.pids 0 * st + i)) N B reduction k) ∧
      BlockState.WriteWithin (footprint O (s.pids 0 * so) N B reduction) s final := by
  let initial := forwardInitial s Y T O sy st so B
  have hi := initial_inv s Y T O sy st so N B flag reduction
  obtain ⟨covered, final, hcover, he, hf⟩ :=
    loop_run s Y T O sy st so (s.pids 0 * sy) (s.pids 0 * st) (s.pids 0 * so)
      N B reduction flag hB hYO hTO 0 initial hi
  refine ⟨final, ?_, hf.complete hB hcover, hf.context.frame⟩
  rw [exec, source_decomposition,
    stepStmts.append_some (forward_initialize s Y T O sy st so N B flag reduction)]
  simp only [initial, Nat.zero_mul, Nat.cast_zero] at he
  simp only [stepStmts, stepStmt, evalOp.eq_def, Option.bind_eq_bind, Option.bind,
    Tile.scalar_data, he]

end VeriTile.Bench.TritonBenchG.KldivTriton.FullForward
#guard_msgs (drop info) in
#auditModuleAxioms
