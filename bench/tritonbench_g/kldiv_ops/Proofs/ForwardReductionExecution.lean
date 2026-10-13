import bench.tritonbench_g.kldiv_ops.Proofs.ForwardReductionLoop
import bench.tritonbench_g.kldiv_ops.Proofs.Initialize
import bench.tritonbench_g.kldiv_ops.Proofs.Finish

namespace VeriTile.Bench.TritonBenchG.KldivOps.FullForward
open VeriTile.Triton FullAlgorithm

/-- Complete source decomposition retains initialization, every loop iteration and final store. -/
theorem source_decomposition (Y T O : RegionName) (sy st so N B : Nat)
    (eps : ℝ) (logTarget : Bool) (reduction : Nat) :
    (forward Y T O sy st so N B eps logTarget reduction).body =
      (forward Y T O sy st so N B eps logTarget reduction).body.take 6 ++
        [.forRangeInt "i" (.constInt 0) (.constInt N) (.constNat B)
          (body Y T O sy st so N B eps logTarget reduction)] ++
        (forward Y T O sy st so N B eps logTarget reduction).body.drop 7 := by
  rfl

theorem reduction_initial_inv (s : BlockState) (Y T O : RegionName)
    (sy st so N B : Nat) (eps : ℝ) (logTarget : Bool) :
    ReductionInv s Y T O (s.pids 0 * sy) (s.pids 0 * st) (s.pids 0 * so)
      N B 0 eps logTarget (forwardInitial s Y T O sy st so B) := by
  constructor
  · rfl
  · rfl
  · rfl
  · rfl
  · simp [forwardInitial]
  · simp [forwardInitial]
  · simp [forwardInitial]
  · simp [forwardInitial]
  · simp [forwardInitial, FullMath.prefixLoss_zero]

/-- All blocks are accumulated before the single scalar store, even when output aliases input. -/
theorem reduction_source_run (s : BlockState) (Y T O : RegionName)
    (sy st so N B : Nat) (eps : ℝ) (logTarget : Bool) (reduction : Nat)
    (hB : 0 < B) (hR : reduction ≠ 0) :
    ∃ final, exec (forward Y T O sy st so N B eps logTarget reduction) s = some final ∧
      final.readMem O (s.pids 0 * so) =
        FullMath.rowLoss logTarget eps (fun k ↦ s.readMem Y (s.pids 0 * sy + k))
          (fun k ↦ s.readMem T (s.pids 0 * st + k)) N ∧
      BlockState.WriteWithin (fun address ↦ address = (O, s.pids 0 * so)) s final := by
  let initial := forwardInitial s Y T O sy st so B
  let result := FullMath.rowLoss logTarget eps
    (fun k ↦ s.readMem Y (s.pids 0 * sy + k))
    (fun k ↦ s.readMem T (s.pids 0 * st + k)) N
  have hi := reduction_initial_inv s Y T O sy st so N B eps logTarget
  obtain ⟨covered, middle, hcover, he, hf⟩ :=
    reduction_loop_run s Y T O sy st so (s.pids 0 * sy) (s.pids 0 * st)
      (s.pids 0 * so) N B eps logTarget reduction hB hR 0 initial hi
  have hloop : stepStmts
      [.forRangeInt "i" (.constInt 0) (.constInt N) (.constNat B)
        (body Y T O sy st so N B eps logTarget reduction)] initial = some middle := by
    simp only [Nat.zero_mul, Nat.cast_zero] at he
    simp only [stepStmts, stepStmt, evalOp.eq_def, Option.bind_eq_bind, Option.bind,
      Tile.scalar_data, he]
  have hend := forward_finish middle Y T O sy st so N B (s.pids 0 * so)
    eps result logTarget reduction hf.optr (hf.complete hcover)
  simp only [if_neg hR] at hend
  refine ⟨middle.writeMem O (s.pids 0 * so) result, ?_, ?_, ?_⟩
  · rw [exec, source_decomposition, List.append_assoc,
      stepStmts.append_some
        (forward_initialize s Y T O sy st so N B eps logTarget reduction),
      stepStmts.append_some hloop]
    exact hend
  · simp [BlockState.writeMem_readMem, result]
  · have hw := BlockState.writeMem_writeWithin
      (P := fun address ↦ address = (O, s.pids 0 * so))
      (s := middle) O (s.pids 0 * so) result rfl
    intro r k hout
    exact (congrFun (congrFun hf.mem r) k).symm.trans (hw r k hout)

end VeriTile.Bench.TritonBenchG.KldivOps.FullForward

#guard_msgs (drop info) in
#auditModuleAxioms
