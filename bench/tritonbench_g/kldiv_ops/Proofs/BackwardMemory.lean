import bench.tritonbench_g.kldiv_ops.Proofs.BackwardStep

namespace VeriTile.Bench.TritonBenchG.KldivOps.FullBackward
open VeriTile.Triton

/-- Reachable loop counters are nonnegative, so pointer conversion loses no offset. -/
theorem address_nat {B : Nat} (base current : Nat) (j : TileIndex [B]) :
    base + (offset (current : Int) j).toNat = base + current + j.1.val := by
  simp only [offset, ← Int.natCast_add, Int.toNat_natCast, Nat.add_assoc]

theorem address_injective {B : Nat} (base current : Nat) :
    Function.Injective (fun j : TileIndex [B] ↦ base + (offset (current : Int) j).toNat) := by
  intro i j h
  simp only [address_nat] at h
  apply Prod.ext
  · exact Fin.ext (Nat.add_left_cancel h)
  · exact Subsingleton.elim _ _

/-- Every active lane reads back its independently specified value after the block store. -/
theorem scatter_value {B : Nat} (s : BlockState) (O : RegionName)
    (base N current : Nat) (values : TileIndex [B] → ℝ) (j : TileIndex [B])
    (hj : current + j.1.val < N) :
    (scatter s O base N current values).readMem O (base + current + j.1.val) = values j := by
  rw [← address_nat base current j]
  apply BlockState.scatter_readback_prop_masked_nd_of_true s
    (fun i ↦ base + (offset (current : Int) i).toNat) values
    (fun i ↦ active N (current : Int) i = Bool.true) j
    (by
      change decide ((current : Int) + j.1.val < (N : Int)) = Bool.true
      apply decide_eq_true
      exact_mod_cast hj)
  intro i _ he
  exact address_injective base current he

/-- The raw memory frame uses precisely the active block footprint. -/
theorem scatter_frame {B : Nat} (s : BlockState) (O : RegionName)
    (base N current : Nat) (values : TileIndex [B] → ℝ) :
    BlockState.WriteWithin
      (fun address ↦ address.1 = O ∧ ∃ lane, lane < B ∧ current + lane < N ∧
        address.2 = base + current + lane) s (scatter s O base N current values) := by
  apply BlockState.foldl_writeMem_prop_writeWithin O
    (fun i ↦ base + (offset (current : Int) i).toNat) values
    (fun i ↦ active N (current : Int) i = Bool.true) (TileShape.allIndices [B]) s
  intro i _ hi
  refine ⟨rfl, i.1.val, i.1.isLt, ?_, address_nat base current i⟩
  have h : (current : Int) + i.1.val < (N : Int) := of_decide_eq_true hi
  exact_mod_cast h

/-- A whole source block establishes the gradient formula and preserves every other raw cell. -/
theorem step_io (s : BlockState) (T O : RegionName)
    (st so N B targetBase outputBase current : Nat) (logTarget : Bool)
    (hI : s.regs .int [] "i" = some (Tile.scalar (current : Int)))
    (hT : s.regs .ptr [] "T" = some (Tile.scalar (T, targetBase)))
    (hO : s.regs .ptr [] "O" = some (Tile.scalar (O, outputBase))) :
    ∃ final, stepStmts (body T O st so N B logTarget) s = some final ∧
      (∀ j : TileIndex [B], current + j.1.val < N →
        final.readMem O (outputBase + current + j.1.val) =
          gradient logTarget (s.readMem T (targetBase + current + j.1.val))) ∧
      BlockState.WriteWithin
        (fun address ↦ address.1 = O ∧ ∃ lane, lane < B ∧ current + lane < N ∧
          address.2 = outputBase + current + lane) s final := by
  let loaded := loadedState (B := B) s T targetBase N current logTarget
  let values := fun j : TileIndex [B] ↦
    gradient logTarget (targetValue s T targetBase N current j)
  refine ⟨scatter loaded O outputBase N current values,
    step s T O st so N B targetBase outputBase current logTarget hI hT hO, ?_, ?_⟩
  · intro j hj
    rw [scatter_value loaded O outputBase N current values j hj]
    have hactive : active N (current : Int) j = Bool.true := by
      change decide ((current : Int) + j.1.val < (N : Int)) = Bool.true
      apply decide_eq_true
      exact_mod_cast hj
    simp [values, targetValue, hactive, address_nat]
  · exact scatter_frame loaded O outputBase N current values

end VeriTile.Bench.TritonBenchG.KldivOps.FullBackward

#guard_msgs (drop info) in
#auditModuleAxioms
