import bench.tritonbench_g.kldiv_ops.Proofs.ForwardBlock
import bench.tritonbench_g.kldiv_ops.Proofs.ForwardMath

namespace VeriTile.Bench.TritonBenchG.KldivOps.FullForward
open VeriTile.Triton
open scoped BigOperators

/-- Reduction blocks leave all memory untouched and accumulate the independent row prefix. -/
structure ReductionInv (initial : BlockState) (Y T O : RegionName)
    (yBase tBase outputBase N B c : Nat) (eps : ℝ) (logTarget : Bool)
    (s : BlockState) : Prop where
  mem : s.mem = initial.mem
  pids : s.pids = initial.pids
  numPids : s.numPids = initial.numPids
  undef : s.undef = initial.undef
  yptr : s.regs .ptr [] "Y" = some (Tile.scalar (Y, yBase))
  tptr : s.regs .ptr [] "T" = some (Tile.scalar (T, tBase))
  optr : s.regs .ptr [] "O" = some (Tile.scalar (O, outputBase))
  base : s.regs .nat [B] "base_offsets" = some (Tile.vec fun j ↦ j.val)
  acc : s.regs .real [] "loss_sum" = some (Tile.scalar (some
    (FullMath.prefixLoss logTarget eps
      (fun k ↦ initial.readMem Y (yBase + k))
      (fun k ↦ initial.readMem T (tBase + k)) N B c)))

theorem ReductionInv.setIndex {initial s : BlockState} {Y T O : RegionName}
    {yBase tBase outputBase N B c : Nat} {eps : ℝ} {logTarget : Bool}
    (h : ReductionInv initial Y T O yBase tBase outputBase N B c eps logTarget s)
    (current : Int) :
    ReductionInv initial Y T O yBase tBase outputBase N B c eps logTarget
      (s.setReg "i" .int [] (Tile.scalar current)) := by
  exact ⟨h.mem, h.pids, h.numPids, h.undef,
    by simpa using h.yptr, by simpa using h.tptr, by simpa using h.optr,
    by simpa using h.base, by simpa using h.acc⟩

/-- One nonzero-mode source iteration adds exactly the next mathematical block loss. -/
theorem reduction_invariant_step (initial s : BlockState) (Y T O : RegionName)
    (sy st so yBase tBase outputBase N B c : Nat) (eps : ℝ) (logTarget : Bool)
    (reduction : Nat) (hR : reduction ≠ 0)
    (h : ReductionInv initial Y T O yBase tBase outputBase N B c eps logTarget s) :
    ∃ next, stepStmts (body Y T O sy st so N B eps logTarget reduction)
        (s.setReg "i" .int [] (Tile.scalar (c * B : Nat))) = some next ∧
      ReductionInv initial Y T O yBase tBase outputBase N B (c + 1) eps logTarget next := by
  let indexed := s.setReg "i" .int [] (Tile.scalar (c * B : Nat))
  have hi : ReductionInv initial Y T O yBase tBase outputBase N B c eps logTarget indexed :=
    h.setIndex _
  let loaded := loadedState (B := B) indexed Y T yBase tBase N (c * B : Nat) eps logTarget
  let f := fun j : Fin B ↦
    lossValue indexed Y T yBase tBase N (c * B : Nat) eps logTarget (j, PUnit.unit)
  let acc := FullMath.prefixLoss logTarget eps
    (fun k ↦ initial.readMem Y (yBase + k))
    (fun k ↦ initial.readMem T (tBase + k)) N B c
  let next := loaded.setReg "loss_sum" .real [] (Tile.scalar (some (acc + ∑ j : Fin B, f j)))
  have hread : indexed.readMem = initial.readMem := by
    funext r k
    simp only [BlockState.readMem, hi.mem]
  have hsum := block_total indexed Y T yBase tBase N B c eps logTarget
  simp only [hread] at hsum
  refine ⟨next, ?_, ?_⟩
  · simpa only [if_neg hR] using
      step indexed Y T O sy st so N B yBase tBase outputBase (c * B : Nat) eps acc
        logTarget reduction (by simp [indexed]) hi.yptr hi.tptr hi.optr hi.base hi.acc
  · constructor
    · exact hi.mem
    · exact hi.pids
    · exact hi.numPids
    · exact hi.undef
    · simpa [next, loaded, loadedState] using hi.yptr
    · simpa [next, loaded, loadedState] using hi.tptr
    · simpa [next, loaded, loadedState] using hi.optr
    · simpa [next, loaded, loadedState] using hi.base
    · simp only [next, BlockState.setReg_same]
      rw [FullMath.prefixLoss_succ]
      simp only [acc, f, hsum]

end VeriTile.Bench.TritonBenchG.KldivOps.FullForward

#guard_msgs (drop info) in
#auditModuleAxioms
