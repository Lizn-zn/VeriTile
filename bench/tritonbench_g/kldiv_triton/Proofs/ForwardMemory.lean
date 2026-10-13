import bench.tritonbench_g.kldiv_triton.Proofs.ForwardMath
import bench.tritonbench_g.kldiv_triton.Proofs.BackwardState

namespace VeriTile.Bench.TritonBenchG.KldivTriton.FullForward
open VeriTile.Triton FullBackward
open scoped BigOperators

/-- The current block's actual write footprint, including its scalar reduction alternative. -/
def blockFootprint (O : RegionName) (base N B current reduction : Nat)
    (address : RegionName × Nat) : Prop :=
  if reduction = 0 then
    address.1 = O ∧ ∃ lane, lane < B ∧ current + lane < N ∧
      address.2 = base + current + lane
  else address.1 = O ∧ address.2 = base

/-- The raw-cell frame includes every region and does not assume output/input separation. -/
theorem result_frame (s : BlockState) (Y T O : RegionName)
    (yBase tBase oBase N B current : Nat) (logTarget : Bool) (reduction : Nat) :
    BlockState.WriteWithin (blockFootprint O oBase N B current reduction) s
      (resultState (B := B) s Y T O yBase tBase oBase N current logTarget reduction) := by
  by_cases hr : reduction = 0
  · simp only [resultState, hr, if_true, blockFootprint]
    exact scatter_frame (loadedState (B := B) s Y T yBase tBase N current logTarget)
      O oBase N current _
  · intro r off hno
    simp only [blockFootprint, hr, if_false] at hno
    simp [resultState, hr, reducedState, loadedState, BlockState.writeMem_mem, hno]

/-- The reduced block stores the actual cached sum, then advances only its pointer register. -/
theorem reduced_value (s : BlockState) (Y T O : RegionName)
    (yBase tBase oBase N B chunk reduction : Nat) (logTarget : Bool) (hr : reduction ≠ 0) :
    (resultState (B := B) s Y T O yBase tBase oBase N (chunk * B : Nat) logTarget reduction).readMem O oBase =
      FullMath.chunkLoss logTarget (fun k ↦ s.readMem Y (yBase + k))
        (fun k ↦ s.readMem T (tBase + k)) N B chunk := by
  simp only [resultState, hr, if_false, reducedState, BlockState.setReg_readMem,
    BlockState.writeMem_readMem]
  exact block_total s Y T yBase tBase N B chunk logTarget

/-- Elementwise output is the independently stated loss at each active logical column. -/
theorem element_value (s : BlockState) (Y T O : RegionName)
    (yBase tBase oBase N B current : Nat) (logTarget : Bool) (j : TileIndex [B])
    (hj : current + j.1.val < N) :
    (resultState (B := B) s Y T O yBase tBase oBase N current logTarget 0).readMem O (oBase + current + j.1.val) =
      FullMath.loss logTarget (s.readMem Y (yBase + current + j.1.val))
        (s.readMem T (tBase + current + j.1.val)) := by
  simp only [resultState, if_true]
  rw [scatter_value _ O oBase N current _ j hj]
  have ha : active N (current : Int) j = Bool.true := by
    change decide ((current : Int) + j.1.val < (N : Int)) = Bool.true
    apply decide_eq_true
    exact_mod_cast hj
  simp only [lossValue, value, ha, ite_true, address_nat]

/-- The output pointer advances only in the scalar-per-chunk path. -/
theorem result_pointers (s : BlockState) (Y T O : RegionName)
    (yBase tBase oBase N B : Nat) (current : Int) (logTarget : Bool) (reduction : Nat)
    (hY : s.regs .ptr [] "Y" = some (Tile.scalar (Y, yBase)))
    (hT : s.regs .ptr [] "T" = some (Tile.scalar (T, tBase)))
    (hO : s.regs .ptr [] "O" = some (Tile.scalar (O, oBase)))
    (hBase : s.regs .nat [B] "base_offsets" = some (Tile.vec fun j ↦ j.val)) :
    let t := resultState (B := B) s Y T O yBase tBase oBase N current logTarget reduction
    t.regs .ptr [] "Y" = some (Tile.scalar (Y, yBase)) ∧
    t.regs .ptr [] "T" = some (Tile.scalar (T, tBase)) ∧
    t.regs .ptr [] "O" = some (Tile.scalar (O, if reduction = 0 then oBase else oBase + 1)) ∧
    t.regs .nat [B] "base_offsets" = some (Tile.vec fun j ↦ j.val) := by
  dsimp only
  by_cases hr : reduction = 0
  · simp only [resultState, hr, if_true]
    simp only [(scatter_state _ O oBase N current _).1]
    simpa [loadedState] using And.intro hY (And.intro hT (And.intro hO hBase))
  · simp [resultState, hr, reducedState, loadedState, hY, hT, hBase]

theorem result_metadata (s : BlockState) (Y T O : RegionName)
    (yBase tBase oBase N B : Nat) (current : Int) (logTarget : Bool) (reduction : Nat) :
    let t := resultState (B := B) s Y T O yBase tBase oBase N current logTarget reduction
    t.pids = s.pids ∧ t.numPids = s.numPids ∧ t.undef = s.undef := by
  dsimp only
  by_cases hr : reduction = 0
  · simp only [resultState, hr, if_true]
    exact (scatter_state _ O oBase N current _).2
  · simp [resultState, hr, reducedState, loadedState]
    rfl

end VeriTile.Bench.TritonBenchG.KldivTriton.FullForward
#guard_msgs (drop info) in
#auditModuleAxioms
