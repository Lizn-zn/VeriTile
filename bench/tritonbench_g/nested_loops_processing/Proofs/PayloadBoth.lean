import bench.tritonbench_g.nested_loops_processing.Proofs.PayloadFlat
import bench.tritonbench_g.nested_loops_processing.Proofs.RealCandidateFlat
import bench.tritonbench_g.nested_loops_processing.Proofs.RealSafetyCandidate

namespace VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.PayloadReal
open VeriTile.Triton TypedSource FullReal

abbrev carrier (storage : Storage) : TileDType :=
  VeriTile.Triton.eraseDType (SourceChecks.dtype storage)

theorem original_io (storage : Storage) (s : BlockState)
    (src dst : RegionName) (sm sn : Nat) (hsep : src ≠ dst) :
    RegionCopyContract (carrier storage)
      (RealCandidate.algorithm storage Bool.false src dst sm sn) s src dst sm sn := by
  unfold RealCandidate.algorithm
  rw [PayloadAlgorithm.source_projection]
  exact source_io (carrier storage) s src dst sm sn hsep

theorem candidate_io (storage : Storage) (s : BlockState)
    (src dst : RegionName) (sm sn : Nat) (hsep : src ≠ dst) :
    RegionCopyContract (carrier storage)
      (RealCandidate.algorithm storage Bool.true src dst sm sn) s src dst sm sn := by
  obtain ⟨final, he, hf, hv⟩ := original_io storage s src dst sm sn hsep
  refine ⟨final, ?_, hf, hv⟩
  change stepStmts (RealCandidate.algorithm storage Bool.true src dst sm sn).body s = _
  rw [← RealCandidate.execution]
  exact he

theorem original_safe (storage : Storage) (bounds : RegionBounds)
    (src dst : RegionName) (sm sn : Nat) (s : BlockState)
    (hsrc : sm + 27 * sn < bounds src) (hdst : sm + 47 * sn < bounds dst) :
    (RealCandidate.algorithm storage Bool.false src dst sm sn).TraceSafe bounds s := by
  unfold RealCandidate.algorithm
  rw [PayloadAlgorithm.source_projection]
  exact source_safe (carrier storage) bounds src dst sm sn s hsrc hdst

theorem candidate_safe (storage : Storage) (bounds : RegionBounds)
    (src dst : RegionName) (sm sn : Nat) (s : BlockState)
    (hsrc : sm + 27 * sn < bounds src) (hdst : sm + 47 * sn < bounds dst) :
    (RealCandidate.algorithm storage Bool.true src dst sm sn).TraceSafe bounds s := by
  exact (RealCandidate.trace_equivalence bounds storage src dst sm sn s).mp
    (original_safe storage bounds src dst sm sn s hsrc hdst)

theorem original_flat_io (storage : Storage) (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0) (s : BlockState)
    (src dst : RegionName) (sm sn : Nat) (hsep : src ≠ dst)
    (hsrc : sm + 27 * sn < A.extent src) (hdst : sm + 47 * sn < A.extent dst)
    (hu : s.undef = fun _ _ ↦ 0) :
    FlatCopyContract (carrier storage) A
      (RealCandidate.algorithm storage Bool.false src dst sm sn) s src dst sm sn := by
  unfold RealCandidate.algorithm
  rw [PayloadAlgorithm.source_projection]
  exact source_flat_io (carrier storage) (by cases storage <;> decide)
    (by cases storage <;> decide) A hd hcov s src dst sm sn hsep hsrc hdst hu

theorem candidate_flat_io (storage : Storage) (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0) (s : BlockState)
    (src dst : RegionName) (sm sn : Nat) (hsep : src ≠ dst)
    (hsrc : sm + 27 * sn < A.extent src) (hdst : sm + 47 * sn < A.extent dst)
    (hu : s.undef = fun _ _ ↦ 0) :
    FlatCopyContract (carrier storage) A
      (RealCandidate.algorithm storage Bool.true src dst sm sn) s src dst sm sn := by
  obtain ⟨final, he, hf, hv⟩ := original_flat_io storage A hd hcov s src dst sm sn hsep hsrc hdst hu
  refine ⟨final, ?_, hf, hv⟩
  rw [← RealCandidate.flat_execution]
  exact he

end VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.PayloadReal

#guard_msgs (drop info) in
#auditModuleAxioms
