import bench.tritonbench_g.chunk_cumsum_kernel.Proofs.RealCandidate

/-! The complete source programs compute valid prefixes for arbitrary real
unspecified values. The shared flat-memory IO adapter separately assumes zero
unspecified values; that adapter condition is not needed by this region proof. -/
namespace VeriTile.Bench.TritonBenchG.ChunkCumsumKernel.RealPadding
open VeriTile.Triton
open scoped BigOperators
noncomputable section

/-- Precision-erased source execution, with no condition on `u.undef`. The
final carry may contain padding; each valid output depends only on input cells. -/
theorem source_outputs (swapped : Bool) (dtype out_dtype : FloatDType)
    (s o : RegionName) (T BT : Nat) (u : BlockState)
    (hSO : o ≠ s) (hBT : 0 < BT) :
    ∃ v, exec (RealSource.source swapped dtype out_dtype s o T BT).eraseDType.toAlgKernel u =
      some v ∧ ∀ i, i < T → v.readMem o (u.pids 0 * T + i) =
        ∑ j ∈ (Finset.range T).filter (fun j => j ≤ i), u.readMem s (u.pids 0 * T + j) := by
  obtain ⟨v, hv, ho⟩ := chunk_cumsum_scalar_surface_global_cumsum s o T BT u hSO hBT
  have he : exec (RealSource.source Bool.false dtype out_dtype s o T BT).eraseDType.toAlgKernel u =
      some v := by
    simpa only [exec, RealSource.original_body] using hv
  refine ⟨v, ?_, ho⟩
  cases swapped
  · exact he
  · exact (RealCandidate.region_exec_equal dtype out_dtype s o T BT u).symm.trans he

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkCumsumKernel.RealPadding
