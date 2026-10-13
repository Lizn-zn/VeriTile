import bench.tritonbench_g.nested_loops_processing.Proofs.PayloadSafetyStep

namespace VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.PayloadReal
open VeriTile.Triton FullReal
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

theorem inner_loop_safe (dtype : TileDType) (bounds : RegionBounds) (src dst : RegionName) (sm sn : Int)
    (s : BlockState) (a c : PointerTile) (v1 v2 : Payload dtype)
    (ha : s.regs .ptr [2,2] "a_ptrs" = some ⟨a⟩)
    (hc : s.regs .ptr [2,2] "c_ptrs" = some ⟨c⟩)
    (h1 : s.regs dtype [2,2] "a1" = some ⟨v1⟩)
    (h2 : s.regs dtype [2,2] "a2" = some ⟨v2⟩)
    (hin : ∀ lane n, n < 2 →
      (a lane).2 + (n + 1) * (2 * sn).toNat < bounds (a lane).1)
    (hout : ∀ lane n, n < 6 →
      (c lane).2 + n * (2 * sn).toNat < bounds (c lane).1) :
    Stmt.forRangeTraceSafe bounds "k" 0 2 1 (PayloadAlgorithm.inner dtype src dst sm sn) s := by
  let s0 := s.setReg "k" .nat [] (Tile.scalar 0)
  let t := innerState dtype s0 a c v1 v2 (2 * sn).toNat
  have e0 : stepStmts (PayloadAlgorithm.inner dtype src dst sm sn) s0 = some t :=
    inner_step dtype src dst sm sn s0 a c v1 v2
      (by simpa [s0] using ha) (by simpa [s0] using hc)
      (by simpa [s0] using h1) (by simpa [s0] using h2)
  refine range_two_safe bounds "k" (PayloadAlgorithm.inner dtype src dst sm sn) s t ?_ e0 ?_
  · apply inner_safe dtype bounds src dst sm sn s0 a c v1 v2
      (by simpa [s0] using ha) (by simpa [s0] using hc)
      (by simpa [s0] using h1) (by simpa [s0] using h2)
    · intro lane; have h := hin lane 0 (by decide); omega
    · intro lane; have h := hout lane 0 (by decide); omega
    · intro lane; have h := hout lane 1 (by decide); omega
    · intro lane; have h := hout lane 2 (by decide); omega
  · have regs := inner_registers dtype s0 a c v1 v2 (2 * sn).toNat
      (by simpa [s0] using h1) (by simpa [s0] using h2)
    apply inner_safe dtype bounds src dst sm sn (t.setReg "k" .nat [] (Tile.scalar 1))
      (bump a (2 * sn).toNat)
      (bump (bump (bump c (2 * sn).toNat) (2 * sn).toNat) (2 * sn).toNat) v1 v2
      (by simpa [t] using regs.1) (by simpa [t] using regs.2.1)
      (by simpa [t] using regs.2.2.1) (by simpa [t] using regs.2.2.2)
    · intro lane; have h := hin lane 1 (by decide); dsimp [bump]; omega
    · intro lane; have h := hout lane 3 (by decide); dsimp [bump]; omega
    · intro lane; have h := hout lane 4 (by decide); dsimp [bump]; omega
    · intro lane; have h := hout lane 5 (by decide); dsimp [bump]; omega

end VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.PayloadReal

#guard_msgs (drop info) in
#auditModuleAxioms
