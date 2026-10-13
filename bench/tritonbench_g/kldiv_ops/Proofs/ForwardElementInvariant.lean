import bench.tritonbench_g.kldiv_ops.Proofs.ForwardBlock
import bench.tritonbench_g.kldiv_ops.Proofs.BackwardMemory
import bench.tritonbench_g.kldiv_ops.Proofs.BackwardState

namespace VeriTile.Bench.TritonBenchG.KldivOps.FullForward
open VeriTile.Triton

def elementFootprint (O : RegionName) (base N : Nat) (address : RegionName × Nat) : Prop :=
  address.1 = O ∧ ∃ k, k < N ∧ address.2 = base + k

/-- Elementwise execution preserves both source regions and verifies every visited active lane. -/
structure ElementInv (initial : BlockState) (Y T O : RegionName)
    (yBase tBase outputBase N B c : Nat) (eps : ℝ) (logTarget : Bool)
    (s : BlockState) : Prop where
  frame : BlockState.WriteWithin (elementFootprint O outputBase N) initial s
  prediction : ∀ k, s.mem Y k = initial.mem Y k
  target : ∀ k, s.mem T k = initial.mem T k
  pids : s.pids = initial.pids
  numPids : s.numPids = initial.numPids
  undef : s.undef = initial.undef
  yptr : s.regs .ptr [] "Y" = some (Tile.scalar (Y, yBase))
  tptr : s.regs .ptr [] "T" = some (Tile.scalar (T, tBase))
  optr : s.regs .ptr [] "O" = some (Tile.scalar (O, outputBase))
  base : s.regs .nat [B] "base_offsets" = some (Tile.vec fun j ↦ j.val)
  acc : s.regs .real [] "loss_sum" = some (Tile.scalar (some 0))
  values : ∀ t lane, t < c → lane < B → t * B + lane < N →
    s.readMem O (outputBase + (t * B + lane)) = FullMath.loss logTarget eps
      (initial.readMem Y (yBase + (t * B + lane)))
      (initial.readMem T (tBase + (t * B + lane)))

theorem ElementInv.setIndex {initial s : BlockState} {Y T O : RegionName}
    {yBase tBase outputBase N B c : Nat} {eps : ℝ} {logTarget : Bool}
    (h : ElementInv initial Y T O yBase tBase outputBase N B c eps logTarget s)
    (current : Int) :
    ElementInv initial Y T O yBase tBase outputBase N B c eps logTarget
      (s.setReg "i" .int [] (Tile.scalar current)) := by
  exact ⟨h.frame, h.prediction, h.target, h.pids, h.numPids, h.undef,
    by simpa using h.yptr, by simpa using h.tptr, by simpa using h.optr,
    by simpa using h.base, by simpa using h.acc, h.values⟩

theorem ElementInv.loaded {initial s : BlockState} {Y T O : RegionName}
    {yBase tBase outputBase N B c : Nat} {eps : ℝ} {logTarget : Bool}
    (h : ElementInv initial Y T O yBase tBase outputBase N B c eps logTarget s)
    (current : Int) :
    ElementInv initial Y T O yBase tBase outputBase N B c eps logTarget
      (loadedState (B := B) s Y T yBase tBase N current eps logTarget) := by
  exact ⟨h.frame, h.prediction, h.target, h.pids, h.numPids, h.undef,
    by simpa [loadedState] using h.yptr, by simpa [loadedState] using h.tptr,
    by simpa [loadedState] using h.optr, by simpa [loadedState] using h.base,
    by simpa [loadedState] using h.acc, h.values⟩

/-- A complete elementwise source block extends the pointwise prefix without changing inputs. -/
theorem element_invariant_step (initial s : BlockState) (Y T O : RegionName)
    (sy st so yBase tBase outputBase N B c : Nat) (eps : ℝ) (logTarget : Bool)
    (hYO : Y ≠ O) (hTO : T ≠ O)
    (h : ElementInv initial Y T O yBase tBase outputBase N B c eps logTarget s) :
    ∃ next, stepStmts (body Y T O sy st so N B eps logTarget 0)
        (s.setReg "i" .int [] (Tile.scalar (c * B : Nat))) = some next ∧
      ElementInv initial Y T O yBase tBase outputBase N B (c + 1) eps logTarget next := by
  let indexed := s.setReg "i" .int [] (Tile.scalar (c * B : Nat))
  have hi : ElementInv initial Y T O yBase tBase outputBase N B c eps logTarget indexed :=
    h.setIndex _
  let middle := loadedState (B := B) indexed Y T yBase tBase N (c * B : Nat) eps logTarget
  have hm : ElementInv initial Y T O yBase tBase outputBase N B c eps logTarget middle :=
    hi.loaded _
  let vals := lossValue (B := B) indexed Y T yBase tBase N (c * B : Nat) eps logTarget
  let next := FullBackward.scatter middle O outputBase N (c * B : Nat) vals
  have hf := FullBackward.scatter_frame middle O outputBase N (c * B) vals
  have hs := FullBackward.scatter_state middle O outputBase N (c * B : Nat) vals
  refine ⟨next, ?_, ?_⟩
  · simpa only [ite_true] using
      step indexed Y T O sy st so N B yBase tBase outputBase (c * B : Nat) eps 0
        logTarget 0 (by simp [indexed]) hi.yptr hi.tptr hi.optr hi.base hi.acc
  · constructor
    · intro r k hout
      rw [hm.frame r k hout]
      apply hf r k
      rintro ⟨rfl, lane, _, hn, he⟩
      exact hout ⟨rfl, c * B + lane, hn, by omega⟩
    · intro k
      exact (hf Y k (by rintro ⟨he, _⟩; exact hYO he)).symm.trans (hm.prediction k)
    · intro k
      exact (hf T k (by rintro ⟨he, _⟩; exact hTO he)).symm.trans (hm.target k)
    · exact hs.2.1.trans hm.pids
    · exact hs.2.2.1.trans hm.numPids
    · exact hs.2.2.2.trans hm.undef
    · rw [hs.1]; exact hm.yptr
    · rw [hs.1]; exact hm.tptr
    · rw [hs.1]; exact hm.optr
    · rw [hs.1]; exact hm.base
    · rw [hs.1]; exact hm.acc
    · intro t lane ht hl hn
      by_cases he : t = c
      · subst t
        let j : TileIndex [B] := (⟨lane, hl⟩, PUnit.unit)
        have hv := FullBackward.scatter_value middle O outputBase N (c * B) vals j hn
        have ha : FullBackward.active N (c * B : Nat) j = Bool.true := by
          change decide ((c * B : Nat) + (lane : Int) < (N : Int)) = Bool.true
          apply decide_eq_true
          exact_mod_cast hn
        have hv' : next.readMem O (outputBase + (c * B + lane)) = vals j := by
          simpa only [next, j, Nat.add_assoc] using hv
        refine hv'.trans ?_
        simp only [vals, lossValue, value, ha, ite_true, FullBackward.address_nat]
        simp only [j, Nat.add_assoc, BlockState.readMem, hi.prediction, hi.target]
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

end VeriTile.Bench.TritonBenchG.KldivOps.FullForward

#guard_msgs (drop info) in
#auditModuleAxioms
