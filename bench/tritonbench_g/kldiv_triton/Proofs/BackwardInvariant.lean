import bench.tritonbench_g.kldiv_triton.Proofs.BackwardMemory
import bench.tritonbench_g.kldiv_triton.Proofs.BackwardState

namespace VeriTile.Bench.TritonBenchG.KldivTriton.FullBackward
open VeriTile.Triton

/-- The logical row is the entire allowed write footprint, including unvisited columns. -/
def footprint (O : RegionName) (base N : Nat) (address : RegionName × Nat) : Prop :=
  address.1 = O ∧ ∃ k, k < N ∧ address.2 = base + k

/-- After c chunks, all active visited lanes satisfy the initial-target gradient formula. -/
structure Inv (initial : BlockState) (T O : RegionName)
    (targetBase outputBase N B c : Nat) (logTarget : Bool) (s : BlockState) : Prop where
  frame : BlockState.WriteWithin (footprint O outputBase N) initial s
  input : ∀ k, s.mem T k = initial.mem T k
  pids : s.pids = initial.pids
  numPids : s.numPids = initial.numPids
  undef : s.undef = initial.undef
  targetReg : s.regs .ptr [] "T" = some (Tile.scalar (T, targetBase))
  outputReg : s.regs .ptr [] "O" = some (Tile.scalar (O, outputBase))
  values : ∀ t lane, t < c → lane < B → t * B + lane < N →
    s.readMem O (outputBase + (t * B + lane)) =
      gradient logTarget (initial.readMem T (targetBase + (t * B + lane)))

theorem Inv.setIndex {initial s : BlockState} {T O : RegionName}
    {targetBase outputBase N B c : Nat} {logTarget : Bool}
    (h : Inv initial T O targetBase outputBase N B c logTarget s) (index : Int) :
    Inv initial T O targetBase outputBase N B c logTarget
      (s.setReg "i" .int [] (Tile.scalar index)) := by
  exact ⟨h.frame, h.input, h.pids, h.numPids, h.undef,
    by simpa using h.targetReg, by simpa using h.outputReg, h.values⟩

theorem Inv.loaded {initial s : BlockState} {T O : RegionName}
    {targetBase outputBase N B c : Nat} {logTarget : Bool}
    (h : Inv initial T O targetBase outputBase N B c logTarget s) (current : Int) :
    Inv initial T O targetBase outputBase N B c logTarget
      (loadedState (B := B) s T targetBase N current logTarget) := by
  exact ⟨h.frame, h.input, h.pids, h.numPids, h.undef,
    by simpa [loadedState] using h.targetReg,
    by simpa [loadedState] using h.outputReg, h.values⟩

/-- One actual block extends the visited prefix and preserves the untouched input row. -/
theorem invariant_step (initial s : BlockState) (T O : RegionName)
    (st so targetBase outputBase N B c : Nat) (logTarget : Bool) (hTO : T ≠ O)
    (h : Inv initial T O targetBase outputBase N B c logTarget s) :
    ∃ next, stepStmts (body T O st so N B logTarget)
        (s.setReg "i" .int [] (Tile.scalar (c * B : Nat))) = some next ∧
      Inv initial T O targetBase outputBase N B (c + 1) logTarget next := by
  let indexed := s.setReg "i" .int [] (Tile.scalar (c * B : Nat))
  have hi : Inv initial T O targetBase outputBase N B c logTarget indexed := h.setIndex _
  let middle := loadedState (B := B) indexed T targetBase N (c * B : Nat) logTarget
  have hm : Inv initial T O targetBase outputBase N B c logTarget middle := hi.loaded _
  let vals := fun j : TileIndex [B] ↦
    gradient logTarget (targetValue indexed T targetBase N (c * B : Nat) j)
  let next := scatter middle O outputBase N (c * B : Nat) vals
  have hf := scatter_frame middle O outputBase N (c * B) vals
  have hs := scatter_state middle O outputBase N (c * B : Nat) vals
  refine ⟨next, ?_, ?_⟩
  · exact step indexed T O st so N B targetBase outputBase (c * B : Nat) logTarget
      (by simp [indexed]) hi.targetReg hi.outputReg
  · constructor
    · intro r k hout
      rw [hm.frame r k hout]
      apply hf r k
      rintro ⟨rfl, lane, _, hn, he⟩
      exact hout ⟨rfl, c * B + lane, hn, by omega⟩
    · intro k
      exact (hf T k (by rintro ⟨he, _⟩; exact hTO he)).symm.trans (hm.input k)
    · exact hs.2.1.trans hm.pids
    · exact hs.2.2.1.trans hm.numPids
    · exact hs.2.2.2.trans hm.undef
    · rw [hs.1]; exact hm.targetReg
    · rw [hs.1]; exact hm.outputReg
    · intro t lane ht hl hn
      by_cases he : t = c
      · subst t
        let j : TileIndex [B] := (⟨lane, hl⟩, PUnit.unit)
        have hv := scatter_value middle O outputBase N (c * B) vals j hn
        have ha : active N (c * B : Nat) j = Bool.true := by
          change decide ((c * B : Nat) + (lane : Int) < (N : Int)) = Bool.true
          apply decide_eq_true
          exact_mod_cast hn
        have hr : indexed.readMem T (targetBase + (c * B + lane)) =
            initial.readMem T (targetBase + (c * B + lane)) := by
          simp only [BlockState.readMem, hi.input]
        have hv' : next.readMem O (outputBase + (c * B + lane)) = vals j := by
          simpa only [next, j, Nat.add_assoc] using hv
        refine hv'.trans ?_
        simp only [vals, targetValue, ha, ite_true, address_nat]
        simpa only [j, Nat.add_assoc] using congrArg (gradient logTarget) hr
      · have htc : t < c := by omega
        have hmul := Nat.mul_le_mul_right B (show t + 1 ≤ c by omega)
        simp only [Nat.add_mul, Nat.one_mul] at hmul
        have hcell := hf O (outputBase + (t * B + lane)) (by
          rintro ⟨_, other, _, _, he⟩
          omega)
        have hr : next.readMem O (outputBase + (t * B + lane)) =
            middle.readMem O (outputBase + (t * B + lane)) := by
          simp only [next, BlockState.readMem, ← hcell]
        exact hr.trans (hm.values t lane htc hl hn)

end VeriTile.Bench.TritonBenchG.KldivTriton.FullBackward

#guard_msgs (drop info) in
#auditModuleAxioms
