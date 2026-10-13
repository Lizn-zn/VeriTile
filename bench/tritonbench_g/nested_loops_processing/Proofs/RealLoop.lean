import bench.tritonbench_g.nested_loops_processing.Proofs.RealStep

namespace VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.FullReal
open VeriTile.Triton
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

theorem range_two (idx : RegName) (body : List Stmt) (s : BlockState) :
    stepForRangeAux idx 0 2 1 body s =
      (stepStmts body (s.setReg idx .nat [] (Tile.scalar 0)) >>= fun t ↦
        stepStmts body (t.setReg idx .nat [] (Tile.scalar 1))) := by
  rw [stepForRangeAux]
  simp only [Nat.one_ne_zero, Nat.reduceLT, ↓reduceIte]
  cases h0 : stepStmts body (s.setReg idx .nat [] (Tile.scalar 0)) with
  | none => rfl
  | some t =>
    simp only [Option.bind_some]
    rw [stepForRangeAux]
    simp only [Nat.one_ne_zero, Nat.reduceLT, ↓reduceIte]
    cases h1 : stepStmts body (t.setReg idx .nat [] (Tile.scalar 1)) with
    | none => simp [h1]
    | some u => simp [stepForRangeAux, h1]

theorem inner_registers (s : BlockState) (a c : PointerTile) (v1 v2 : IntTile) (delta : Nat)
    (h1 : s.regs .int [2,2] "a1" = some ⟨v1⟩)
    (h2 : s.regs .int [2,2] "a2" = some ⟨v2⟩) :
    (innerState s a c v1 v2 delta).regs .ptr [2,2] "a_ptrs" = some ⟨bump a delta⟩ ∧
    (innerState s a c v1 v2 delta).regs .ptr [2,2] "c_ptrs" =
      some ⟨bump (bump (bump c delta) delta) delta⟩ ∧
    (innerState s a c v1 v2 delta).regs .int [2,2] "a1" = some ⟨v1⟩ ∧
    (innerState s a c v1 v2 delta).regs .int [2,2] "a2" = some ⟨v2⟩ := by
  simp [innerState, scatter, h1, h2]

noncomputable def innerLoopState (s : BlockState) (a c : PointerTile)
    (v1 v2 : IntTile) (delta : Nat) : BlockState :=
  let t := innerState (s.setReg "k" .nat [] (Tile.scalar 0)) a c v1 v2 delta
  innerState (t.setReg "k" .nat [] (Tile.scalar 1))
    (bump a delta) (bump (bump (bump c delta) delta) delta) v1 v2 delta

theorem inner_loop (src dst : RegionName) (sm sn : Int)
    (s : BlockState) (a c : PointerTile) (v1 v2 : IntTile)
    (ha : s.regs .ptr [2,2] "a_ptrs" = some ⟨a⟩)
    (hc : s.regs .ptr [2,2] "c_ptrs" = some ⟨c⟩)
    (h1 : s.regs .int [2,2] "a1" = some ⟨v1⟩)
    (h2 : s.regs .int [2,2] "a2" = some ⟨v2⟩) :
    stepForRangeAux "k" 0 2 1 (inner src dst sm sn) s =
      some (innerLoopState s a c v1 v2 (2 * sn).toNat) := by
  rw [range_two]
  have e0 := inner_step src dst sm sn (s.setReg "k" .nat [] (Tile.scalar 0)) a c v1 v2
    (by simpa using ha) (by simpa using hc) (by simpa using h1) (by simpa using h2)
  rw [e0]
  have regs := inner_registers (s.setReg "k" .nat [] (Tile.scalar 0)) a c v1 v2
    (2 * sn).toNat (by simpa using h1) (by simpa using h2)
  exact inner_step src dst sm sn _ _ _ _ _
    (by simpa using regs.1) (by simpa using regs.2.1)
    (by simpa using regs.2.2.1) (by simpa using regs.2.2.2)

end VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.FullReal

#guard_msgs (drop info) in
#auditModuleAxioms
