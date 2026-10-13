import bench.tritonbench_g.nested_loops_processing.Proofs.PayloadStep
import bench.tritonbench_g.nested_loops_processing.Proofs.RealLoop

namespace VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.PayloadReal
open VeriTile.Triton FullReal
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

theorem inner_registers (dtype : TileDType) (s : BlockState) (a c : PointerTile) (v1 v2 : Payload dtype) (delta : Nat)
    (h1 : s.regs dtype [2,2] "a1" = some ⟨v1⟩)
    (h2 : s.regs dtype [2,2] "a2" = some ⟨v2⟩) :
    (innerState dtype s a c v1 v2 delta).regs .ptr [2,2] "a_ptrs" = some ⟨bump a delta⟩ ∧
    (innerState dtype s a c v1 v2 delta).regs .ptr [2,2] "c_ptrs" =
      some ⟨bump (bump (bump c delta) delta) delta⟩ ∧
    (innerState dtype s a c v1 v2 delta).regs dtype [2,2] "a1" = some ⟨v1⟩ ∧
    (innerState dtype s a c v1 v2 delta).regs dtype [2,2] "a2" = some ⟨v2⟩ := by
  simp [innerState, scatter, h1, h2]

noncomputable def innerLoopState (dtype : TileDType) (s : BlockState) (a c : PointerTile)
    (v1 v2 : Payload dtype) (delta : Nat) : BlockState :=
  let t := innerState dtype (s.setReg "k" .nat [] (Tile.scalar 0)) a c v1 v2 delta
  innerState dtype (t.setReg "k" .nat [] (Tile.scalar 1))
    (bump a delta) (bump (bump (bump c delta) delta) delta) v1 v2 delta

theorem inner_loop (dtype : TileDType) (src dst : RegionName) (sm sn : Int)
    (s : BlockState) (a c : PointerTile) (v1 v2 : Payload dtype)
    (ha : s.regs .ptr [2,2] "a_ptrs" = some ⟨a⟩)
    (hc : s.regs .ptr [2,2] "c_ptrs" = some ⟨c⟩)
    (h1 : s.regs dtype [2,2] "a1" = some ⟨v1⟩)
    (h2 : s.regs dtype [2,2] "a2" = some ⟨v2⟩) :
    stepForRangeAux "k" 0 2 1 (PayloadAlgorithm.inner dtype src dst sm sn) s =
      some (innerLoopState dtype s a c v1 v2 (2 * sn).toNat) := by
  rw [range_two]
  have e0 := inner_step dtype src dst sm sn (s.setReg "k" .nat [] (Tile.scalar 0)) a c v1 v2
    (by simpa using ha) (by simpa using hc) (by simpa using h1) (by simpa using h2)
  rw [e0]
  have regs := inner_registers dtype (s.setReg "k" .nat [] (Tile.scalar 0)) a c v1 v2
    (2 * sn).toNat (by simpa using h1) (by simpa using h2)
  exact inner_step dtype src dst sm sn _ _ _ _ _
    (by simpa using regs.1) (by simpa using regs.2.1)
    (by simpa using regs.2.2.1) (by simpa using regs.2.2.2)

end VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.PayloadReal

#guard_msgs (drop info) in
#auditModuleAxioms
