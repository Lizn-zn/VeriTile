import bench.tritonbench_g.nested_loops_processing.Proofs.RealFlat
import bench.tritonbench_g.nested_loops_processing.Proofs.RealCandidateFlat
import bench.tritonbench_g.nested_loops_processing.Proofs.RealSafetyCandidate

namespace VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.FullReal
open VeriTile.Triton TypedSource

/-- The candidate shares the complete ordered-write oracle and raw memory frame. -/
theorem candidate_io (s : BlockState) (src dst : RegionName) (sm sn : Nat)
    (hsep : src ≠ dst) :
    RegionCopyContract (RealCandidate.algorithm .int32 Bool.true src dst sm sn)
      s src dst sm sn := by
  obtain ⟨final, he, hf, hv⟩ := source_io s src dst sm sn hsep
  refine ⟨final, ?_, hf, hv⟩
  change stepStmts (RealCandidate.algorithm .int32 Bool.true src dst sm sn).body s = _
  rw [← RealCandidate.execution]
  exact he

/-- Full flat IO for the actual candidate, including every location outside its footprint. -/
theorem candidate_flat_io (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0) (s : BlockState)
    (src dst : RegionName) (sm sn : Nat) (hsep : src ≠ dst)
    (hsrc : sm + 27 * sn < A.extent src)
    (hdst : sm + 47 * sn < A.extent dst)
    (hu : s.undef = fun _ _ ↦ 0) :
    FlatCopyContract A (RealCandidate.algorithm .int32 Bool.true src dst sm sn)
      s src dst sm sn := by
  obtain ⟨final, he, hf, hv⟩ := source_flat_io A hd hcov s src dst sm sn hsep hsrc hdst hu
  refine ⟨final, ?_, hf, hv⟩
  rw [← RealCandidate.flat_execution]
  exact he

end VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.FullReal

#guard_msgs (drop info) in
#auditModuleAxioms
