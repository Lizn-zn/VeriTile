import bench.tritonbench_g.nested_loops_processing.Proofs.RealOuter
import bench.tritonbench_g.nested_loops_processing.Proofs.RealInitialize

namespace VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.FullReal
open VeriTile.Triton
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

theorem outer_registers (s : BlockState) (a c : PointerTile) (delta : Nat) :
    (outerState s a c delta).regs .ptr [2,2] "a_ptrs" = some ⟨advances 7 a delta⟩ ∧
    (outerState s a c delta).regs .ptr [2,2] "c_ptrs" = some ⟨advances 12 c delta⟩ := by
  have h := middleLoop_registers (s.setReg "a1" .int [2,2] ⟨loaded s a⟩)
    a c (loaded s a) delta (by simp)
  constructor
  · simp [outerState]
  · simpa [outerState] using h.2

noncomputable def outerLoopState (s : BlockState) (a c : PointerTile) (delta : Nat) : BlockState :=
  let t := outerState (s.setReg "i" .nat [] (Tile.scalar 0)) a c delta
  outerState (t.setReg "i" .nat [] (Tile.scalar 1)) (advances 7 a delta) (advances 12 c delta) delta

theorem outer_loop (src dst : RegionName) (sm sn : Int)
    (s : BlockState) (a c : PointerTile)
    (ha : s.regs .ptr [2,2] "a_ptrs" = some ⟨a⟩)
    (hc : s.regs .ptr [2,2] "c_ptrs" = some ⟨c⟩) :
    stepForRangeAux "i" 0 2 1 (outer src dst sm sn) s =
      some (outerLoopState s a c (2 * sn).toNat) := by
  rw [range_two]
  have e0 := outer_step src dst sm sn (s.setReg "i" .nat [] (Tile.scalar 0)) a c
    (by simpa using ha) (by simpa using hc)
  rw [e0]
  have regs := outer_registers (s.setReg "i" .nat [] (Tile.scalar 0)) a c (2 * sn).toNat
  exact outer_step src dst sm sn _ _ _ (by simpa using regs.1) (by simpa using regs.2)

noncomputable def finalState (s : BlockState) (src dst : RegionName) (sm sn : Int) : BlockState :=
  outerLoopState (prefixState s src dst sm sn) (initialA src sm sn) (initialC dst sm sn)
    (2 * sn).toNat

/-- Execution of the actual source, with every cached load and all 24 stores retained. -/
theorem source_run (src dst : RegionName) (sm sn : Int) (s : BlockState) :
    stepStmts (source src dst sm sn).body s = some (finalState s src dst sm sn) := by
  rw [source_decomposition, stepStmts.append_some (prefix_step src dst sm sn s)]
  have h := outer_loop src dst sm sn (prefixState s src dst sm sn)
    (initialA src sm sn) (initialC dst sm sn)
    (by simp [prefixState]) (by simp [prefixState])
  simpa only [stepStmts, stepStmt, h, finalState] using h

end VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.FullReal

#guard_msgs (drop info) in
#auditModuleAxioms
