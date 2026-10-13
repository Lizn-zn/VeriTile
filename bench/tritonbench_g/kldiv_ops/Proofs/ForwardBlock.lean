import bench.tritonbench_g.kldiv_ops.Proofs.ForwardFinish

namespace VeriTile.Bench.TritonBenchG.KldivOps.FullForward
open VeriTile.Triton FullAlgorithm FullBackward
open scoped BigOperators

/-- The complete source block either scatters cached lanes or adds their sum to the accumulator. -/
theorem step (s : BlockState) (Y T O : RegionName)
    (sy st so N B yBase tBase outputBase : Nat) (current : Int) (eps acc : ℝ)
    (logTarget : Bool) (reduction : Nat)
    (hI : s.regs .int [] "i" = some (Tile.scalar current))
    (hY : s.regs .ptr [] "Y" = some (Tile.scalar (Y, yBase)))
    (hT : s.regs .ptr [] "T" = some (Tile.scalar (T, tBase)))
    (hO : s.regs .ptr [] "O" = some (Tile.scalar (O, outputBase)))
    (hBase : s.regs .nat [B] "base_offsets" = some (Tile.vec fun j ↦ j.val))
    (hAcc : s.regs .real [] "loss_sum" = some (Tile.scalar (some acc))) :
    let loaded := loadedState (B := B) s Y T yBase tBase N current eps logTarget
    let values := lossValue (B := B) s Y T yBase tBase N current eps logTarget
    stepStmts (body Y T O sy st so N B eps logTarget reduction) s =
      some (if reduction = 0 then scatter loaded O outputBase N current values
        else loaded.setReg "loss_sum" .real []
          (Tile.scalar (some (acc + ∑ j : Fin B, values (j, PUnit.unit))))) := by
  dsimp only
  rw [← List.take_append_drop 5 (body Y T O sy st so N B eps logTarget reduction)]
  rw [stepStmts.append_some
    (load_compute s Y T O sy st so N B yBase tBase current eps logTarget reduction
      hI hY hT hBase)]
  by_cases hR : reduction = 0
  · subst reduction
    simp only [ite_true]
    apply elementwise_store
    · simpa [loadedState] using hO
    · simp [loadedState]
    · simp [loadedState]
    · simp [loadedState]
  · rw [if_neg hR]
    apply accumulate _ Y T O sy st so N B eps acc logTarget reduction
      (fun j ↦ lossValue s Y T yBase tBase N current eps logTarget (j, PUnit.unit)) hR
    · simpa [loadedState] using hAcc
    · simp [loadedState]

end VeriTile.Bench.TritonBenchG.KldivOps.FullForward

#guard_msgs (drop info) in
#auditModuleAxioms
