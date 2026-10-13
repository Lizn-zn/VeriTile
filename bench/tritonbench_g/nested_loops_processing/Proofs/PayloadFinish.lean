import bench.tritonbench_g.nested_loops_processing.Proofs.PayloadOuter
import bench.tritonbench_g.nested_loops_processing.Proofs.RealInitialize

namespace VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.PayloadReal
open VeriTile.Triton FullReal
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

theorem outer_registers (dtype : TileDType) (s : BlockState) (a c : PointerTile) (delta : Nat) :
    (outerState dtype s a c delta).regs .ptr [2,2] "a_ptrs" = some ⟨advances 7 a delta⟩ ∧
    (outerState dtype s a c delta).regs .ptr [2,2] "c_ptrs" = some ⟨advances 12 c delta⟩ := by
  have h := middleLoop_registers dtype (s.setReg "a1" dtype [2,2] ⟨loaded dtype s a⟩)
    a c (loaded dtype s a) delta (by simp)
  constructor
  · simp [outerState]
  · simpa [outerState] using h.2

noncomputable def outerLoopState (dtype : TileDType) (s : BlockState) (a c : PointerTile) (delta : Nat) : BlockState :=
  let t := outerState dtype (s.setReg "i" .nat [] (Tile.scalar 0)) a c delta
  outerState dtype (t.setReg "i" .nat [] (Tile.scalar 1)) (advances 7 a delta) (advances 12 c delta) delta

theorem outer_loop (dtype : TileDType) (src dst : RegionName) (sm sn : Int)
    (s : BlockState) (a c : PointerTile)
    (ha : s.regs .ptr [2,2] "a_ptrs" = some ⟨a⟩)
    (hc : s.regs .ptr [2,2] "c_ptrs" = some ⟨c⟩) :
    stepForRangeAux "i" 0 2 1 (PayloadAlgorithm.outer dtype src dst sm sn) s =
      some (outerLoopState dtype s a c (2 * sn).toNat) := by
  rw [range_two]
  have e0 := outer_step dtype src dst sm sn (s.setReg "i" .nat [] (Tile.scalar 0)) a c
    (by simpa using ha) (by simpa using hc)
  rw [e0]
  have regs := outer_registers dtype (s.setReg "i" .nat [] (Tile.scalar 0)) a c (2 * sn).toNat
  exact outer_step dtype src dst sm sn _ _ _ (by simpa using regs.1) (by simpa using regs.2)

noncomputable def finalState (dtype : TileDType) (s : BlockState) (src dst : RegionName) (sm sn : Int) : BlockState :=
  outerLoopState dtype (prefixState s src dst sm sn) (initialA src sm sn) (initialC dst sm sn)
    (2 * sn).toNat

/-- Execution of the actual source, with every cached load and all 24 stores retained. -/
theorem source_run (dtype : TileDType) (src dst : RegionName) (sm sn : Int) (s : BlockState) :
    stepStmts (PayloadAlgorithm.program dtype src dst sm sn).body s = some (finalState dtype s src dst sm sn) := by
  change stepStmts (prefixBody src dst sm sn ++
    [.forRange "i" 0 2 1 (PayloadAlgorithm.outer dtype src dst sm sn)]) s = _
  rw [stepStmts.append_some (prefix_step src dst sm sn s)]
  have h := outer_loop dtype src dst sm sn (prefixState s src dst sm sn)
    (initialA src sm sn) (initialC dst sm sn)
    (by simp [prefixState]) (by simp [prefixState])
  simpa only [stepStmts, stepStmt, h, finalState] using h

end VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.PayloadReal

#guard_msgs (drop info) in
#auditModuleAxioms
