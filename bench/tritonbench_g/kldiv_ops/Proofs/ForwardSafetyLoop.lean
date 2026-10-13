import bench.tritonbench_g.kldiv_ops.Proofs.ForwardSafetyBlock

namespace VeriTile.Bench.TritonBenchG.KldivOps.FullForward
open VeriTile.Triton FullAlgorithm
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

/-- Reduction safety follows the actual signed loop without an output/input separation premise. -/
theorem reduction_loop_safe (bounds : RegionBounds) (initial s : BlockState)
    (Y T O : RegionName) (sy st so yBase tBase outputBase N B : Nat)
    (eps : ℝ) (logTarget : Bool) (reduction : Nat) (hR : reduction ≠ 0)
    (h : ReductionInv initial Y T O yBase tBase outputBase N B 0 eps logTarget s)
    (hby : ∀ k : Fin N, yBase + k.val < bounds Y)
    (hbt : ∀ k : Fin N, tBase + k.val < bounds T) :
    Stmt.TraceSafeR .triv bounds
      (.forRangeInt "i" (.constInt 0) (.constInt N) (.constNat B)
        (body Y T O sy st so N B eps logTarget reduction)) s := by
  simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, evalOpR_triv, evalOp.eq_def,
    Tile.scalar_data, Option.some.injEq, forall_eq', true_and]
  apply Stmt.forRangeIntTraceSafeR_inv .triv bounds "i" N B
    (body Y T O sy st so N B eps logTarget reduction)
    (fun current state ↦ ∃ c, current = (c * B : Nat) ∧
      ReductionInv initial Y T O yBase tBase outputBase N B c eps logTarget state)
    ?_ 0 s ⟨0, by simp, h⟩
  intro current state _ hs
  rcases hs with ⟨c, rfl, hs⟩
  refine ⟨block_safe bounds state Y T O sy st so yBase tBase outputBase N B (c * B)
    eps logTarget reduction hs.yptr hs.tptr hs.optr hs.base hby hbt
    (fun hz ↦ False.elim (hR hz)), ?_⟩
  obtain ⟨next, he, hn⟩ := reduction_invariant_step initial state Y T O
    sy st so yBase tBase outputBase N B c eps logTarget reduction hR hs
  refine ⟨next, ?_, c + 1, ?_, hn⟩
  · simpa only [stepStmtsR_triv] using he
  · simp [Nat.add_mul]

/-- Elementwise safety uses the same successful loop steps that preserve both source snapshots. -/
theorem element_loop_safe (bounds : RegionBounds) (initial s : BlockState)
    (Y T O : RegionName) (sy st so yBase tBase outputBase N B : Nat)
    (eps : ℝ) (logTarget : Bool) (hYO : Y ≠ O) (hTO : T ≠ O)
    (h : ElementInv initial Y T O yBase tBase outputBase N B 0 eps logTarget s)
    (hby : ∀ k : Fin N, yBase + k.val < bounds Y)
    (hbt : ∀ k : Fin N, tBase + k.val < bounds T)
    (hbo : ∀ k : Fin N, outputBase + k.val < bounds O) :
    Stmt.TraceSafeR .triv bounds
      (.forRangeInt "i" (.constInt 0) (.constInt N) (.constNat B)
        (body Y T O sy st so N B eps logTarget 0)) s := by
  simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, evalOpR_triv, evalOp.eq_def,
    Tile.scalar_data, Option.some.injEq, forall_eq', true_and]
  apply Stmt.forRangeIntTraceSafeR_inv .triv bounds "i" N B
    (body Y T O sy st so N B eps logTarget 0)
    (fun current state ↦ ∃ c, current = (c * B : Nat) ∧
      ElementInv initial Y T O yBase tBase outputBase N B c eps logTarget state)
    ?_ 0 s ⟨0, by simp, h⟩
  intro current state _ hs
  rcases hs with ⟨c, rfl, hs⟩
  refine ⟨block_safe bounds state Y T O sy st so yBase tBase outputBase N B (c * B)
    eps logTarget 0 hs.yptr hs.tptr hs.optr hs.base hby hbt (fun _ ↦ hbo), ?_⟩
  obtain ⟨next, he, hn⟩ := element_invariant_step initial state Y T O
    sy st so yBase tBase outputBase N B c eps logTarget hYO hTO hs
  refine ⟨next, ?_, c + 1, ?_, hn⟩
  · simpa only [stepStmtsR_triv] using he
  · simp [Nat.add_mul]

end VeriTile.Bench.TritonBenchG.KldivOps.FullForward
#guard_msgs (drop info) in
#auditModuleAxioms
