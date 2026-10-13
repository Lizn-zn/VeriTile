import bench.tritonbench_g.softmax_optimize.Proofs.Geometry
import bench.tritonbench_g.softmax_optimize.Proofs.LegacyReal

namespace VeriTile.Bench.TritonBenchG.SoftmaxOptimize.Safety
open VeriTile.Triton FullSource

/-- Every lane of an aligned unmasked tile precedes the logical row end. -/
theorem full_lane (N B c : Nat) (hB : 0 < B) (hc : c < prevMultipleNat N B)
    (haligned : B ∣ c) (i : Fin B) : c + i.val < N := by
  obtain ⟨k, rfl⟩ := haligned
  rw [Nat.mul_comm B k] at hc ⊢
  exact loopA_inRange N B hB k i hc

/-- The signed source casts the complete nonnegative address only after addition. -/
theorem address_cast (row N c lane : Nat) :
    (((row : Int) * (N : Int) + ((c : Int) + (lane : Int))).toNat) =
      row * N + c + lane := by
  have h : (row : Int) * (N : Int) + ((c : Int) + (lane : Int)) =
      ((row * N + c + lane : Nat) : Int) := by push_cast; ring
  rw [h]
  rfl

theorem row_address_bound (row M N column extent : Nat)
    (hrow : row < M) (hcolumn : column < N) (hextent : M * N ≤ extent) :
    row * N + column < extent := by
  have hrow' : (row + 1) * N ≤ M * N :=
    Nat.mul_le_mul_right N (by omega)
  rw [Nat.add_mul, Nat.one_mul] at hrow'
  omega

/-- A masked tail's signed comparison is sufficient for its active addresses. -/
theorem tail_address_bound (row M N c extent : Nat) (B : Nat) (i : Fin B)
    (hrow : row < M) (hextent : M * N ≤ extent)
    (hmask : (c : Int) + (i.val : Int) < (N : Int)) :
    (((row : Int) * (N : Int) + ((c : Int) + (i.val : Int))).toNat) < extent := by
  rw [address_cast]
  have hcolumn : c + i.val < N := by exact_mod_cast hmask
  simpa only [Nat.add_assoc] using
    row_address_bound row M N (c + i.val) extent hrow hcolumn hextent

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxOptimize.Safety
