import bench.tritonbench_g.softmax_optimize.Proofs.SafetyInput
import bench.tritonbench_g.softmax_optimize.Proofs.SafetyOutputBody
import bench.tritonbench_g.softmax_optimize.Proofs.SafetyBodyState
import bench.tritonbench_g.softmax_optimize.Proofs.SafetyLoop
import bench.tritonbench_g.softmax_optimize.Proofs.SafetyGeometry

namespace VeriTile.Bench.TritonBenchG.SoftmaxOptimize.Safety
open VeriTile.Triton SignedTransport FullSource

theorem input_full_range_safe (bounds : RegionBounds) (X : RegionName)
    (M N B p c : Nat) (s : BlockState) (hB : 0 < B)
    (hp : p < M) (hX : M * N ≤ bounds X) (hc : B ∣ c)
    (hpid : s.regs .nat [] "pid_m" = some (Tile.scalar p)) :
    Stmt.forRangeIntTraceSafe bounds "start_n" (c : Int) (prevMultipleNat N B : Int)
      B (inputFull X N B) s := by
  apply signed_loop_safe bounds "start_n" (prevMultipleNat N B) B (inputFull X N B)
    (fun i t ↦ B ∣ i ∧ t.regs .nat [] "pid_m" = some (Tile.scalar p)) hB
  · intro i t hi hinv
    apply input_full_safe bounds X N B i p t hinv.2
    intro lane
    exact row_address_bound p M N (i + lane.val) (bounds X) hp
      (full_lane N B i hB hi hinv.1 lane) hX
  · intro i t u hi hinv hrun
    refine ⟨Dvd.dvd.add hinv.1 (dvd_refl B), ?_⟩
    rw [source_bodies_preserve "pid_m" (Or.inl rfl) .nat [] X X N B
      (inputFull X N B) (by simp) _ u hrun]
    simpa using hinv.2
  · exact ⟨hc, hpid⟩

theorem input_tail_range_safe (bounds : RegionBounds) (X : RegionName)
    (M N B p c : Nat) (s : BlockState) (hB : 0 < B)
    (hp : p < M) (hX : M * N ≤ bounds X)
    (hpid : s.regs .nat [] "pid_m" = some (Tile.scalar p)) :
    Stmt.forRangeIntTraceSafe bounds "start_n" (c : Int) (N : Int)
      B (inputTail X N B) s := by
  apply signed_loop_safe bounds "start_n" N B (inputTail X N B)
    (fun _ t ↦ t.regs .nat [] "pid_m" = some (Tile.scalar p)) hB
  · intro i t hi hreg
    apply input_tail_safe bounds X N B i p t hreg
    intro lane hactive
    exact row_address_bound p M N (i + lane.val) (bounds X) hp hactive hX
  · intro i t u hi hreg hrun
    rw [source_bodies_preserve "pid_m" (Or.inl rfl) .nat [] X X N B
      (inputTail X N B) (by simp) _ u hrun]
    simpa using hreg
  · exact hpid

theorem output_full_range_safe (bounds : RegionBounds) (Y X : RegionName)
    (M N B p c : Nat) (s : BlockState) (hB : 0 < B)
    (hp : p < M) (hX : M * N ≤ bounds X) (hY : M * N ≤ bounds Y) (hc : B ∣ c)
    (hpid : s.regs .nat [] "pid_m" = some (Tile.scalar p)) :
    Stmt.forRangeIntTraceSafe bounds "start_n" (c : Int) (prevMultipleNat N B : Int)
      B (outputFull Y X N B) s := by
  apply signed_loop_safe bounds "start_n" (prevMultipleNat N B) B (outputFull Y X N B)
    (fun i t ↦ B ∣ i ∧ t.regs .nat [] "pid_m" = some (Tile.scalar p)) hB
  · intro i t hi hinv
    apply output_full_safe bounds Y X N B i p t hinv.2
    · intro lane
      exact row_address_bound p M N (i + lane.val) (bounds X) hp
        (full_lane N B i hB hi hinv.1 lane) hX
    · intro lane
      exact row_address_bound p M N (i + lane.val) (bounds Y) hp
        (full_lane N B i hB hi hinv.1 lane) hY
  · intro i t u hi hinv hrun
    refine ⟨Dvd.dvd.add hinv.1 (dvd_refl B), ?_⟩
    rw [source_bodies_preserve "pid_m" (Or.inl rfl) .nat [] Y X N B
      (outputFull Y X N B) (by simp) _ u hrun]
    simpa using hinv.2
  · exact ⟨hc, hpid⟩

theorem output_tail_range_safe (bounds : RegionBounds) (Y X : RegionName)
    (M N B p c : Nat) (s : BlockState) (hB : 0 < B)
    (hp : p < M) (hX : M * N ≤ bounds X) (hY : M * N ≤ bounds Y)
    (hpid : s.regs .nat [] "pid_m" = some (Tile.scalar p)) :
    Stmt.forRangeIntTraceSafe bounds "start_n" (c : Int) (N : Int)
      B (outputTail Y X N B) s := by
  apply signed_loop_safe bounds "start_n" N B (outputTail Y X N B)
    (fun _ t ↦ t.regs .nat [] "pid_m" = some (Tile.scalar p)) hB
  · intro i t hi hreg
    apply output_tail_safe bounds Y X N B i p t hreg
    · intro lane hactive
      exact row_address_bound p M N (i + lane.val) (bounds X) hp hactive hX
    · intro lane hactive
      exact row_address_bound p M N (i + lane.val) (bounds Y) hp hactive hY
  · intro i t u hi hreg hrun
    rw [source_bodies_preserve "pid_m" (Or.inl rfl) .nat [] Y X N B
      (outputTail Y X N B) (by simp) _ u hrun]
    simpa using hreg
  · exact hpid

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxOptimize.Safety
