import bench.tritonbench_g.nested_loops_processing.Proofs.PayloadSafetyOuter
import bench.tritonbench_g.nested_loops_processing.Proofs.RealSafetySource
import bench.tritonbench_g.nested_loops_processing.Proofs.PayloadOuterInvariant
import bench.tritonbench_g.nested_loops_processing.Proofs.RealBounds

namespace VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.PayloadReal
open VeriTile.Triton TypedSource FullReal
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

/-- Full nested-loop access safety. Bounds cover precisely the largest input and
output offsets, without requiring bounds for the unused final pointer cursors. -/
theorem source_safe (dtype : TileDType) (bounds : RegionBounds) (src dst : RegionName) (sm sn : Nat)
    (s : BlockState) (hsrc : sm + 27 * sn < bounds src)
    (hdst : sm + 47 * sn < bounds dst) :
    (PayloadAlgorithm.program dtype src dst sm sn).TraceSafe bounds s := by
  unfold Kernel.TraceSafe
  change Stmt.TraceSafeList bounds (prefixBody src dst sm sn ++
    [.forRange "i" 0 2 1 (PayloadAlgorithm.outer dtype src dst sm sn)]) s
  apply TraceProofs.append_safe
  · exact prefix_safe bounds src dst sm sn s
  · intro t ht
    have he := prefix_step src dst sm sn s
    have hteq : t = prefixState s src dst sm sn := Option.some.inj (ht.symm.trans he)
    subst t
    apply singleton_safe
    rw [Stmt.TraceSafe]
    apply outer_loop_safe dtype bounds src dst sm sn _ (initialA src sm sn) (initialC dst sm sn)
      (by simp [prefixState]) (by simp [prefixState])
    · intro lane n hn
      have hd : (2 * (sn : Int)).toNat = 2 * sn := by omega
      rw [initialA_nonnegative, hd]
      have h := tilePointer_bound bounds src sm sn (2 * n) 26 (by omega) hsrc lane
      dsimp [tilePointer] at h ⊢
      nlinarith
    · intro lane n hn
      have hd : (2 * (sn : Int)).toNat = 2 * sn := by omega
      rw [initialC_nonnegative, hd]
      have h := tilePointer_bound bounds dst sm sn (2 * n) 46 (by omega) hdst lane
      dsimp [tilePointer] at h ⊢
      nlinarith

end VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.PayloadReal

#guard_msgs (drop info) in
#auditModuleAxioms
