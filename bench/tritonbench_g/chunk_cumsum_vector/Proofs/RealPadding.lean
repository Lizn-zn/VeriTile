import bench.tritonbench_g.chunk_cumsum_vector.Proofs.RealInit
import bench.tritonbench_g.chunk_cumsum_vector.Proofs.RealCandidate

/-! The complete real source executions need no zero-padding premise.
Both time and feature tails retain arbitrary values; only active prefixes are
observed, and all other memory cells are preserved. -/
namespace VeriTile.Bench.TritonBenchG.ChunkCumsumVector.RealPadding
open VeriTile.Triton RealProgram
open scoped BigOperators
noncomputable section

/-- Both complete sources implement the independent column prefix formula for
arbitrary `u.undef`. The shared flat IO adapter separately requires a zero oracle. -/
theorem source_outputs (swapped : Bool) (p : Layout) (s o : RegionName)
    (dtype out_dtype : FloatDType) (u : BlockState) (hSO : s ≠ o) (hBT : 0 < p.BT)
    (hinj : RealLoop.NoCollision p (RealInit.launch u)) :
    ∃ v, exec (RealSource.source swapped dtype out_dtype s o
      p.headStride p.timeStride p.channelStride p.T p.S p.BT p.BS).eraseDType.toAlgKernel u =
        some v ∧
      (∀ (t : Nat) (j : Fin p.BS), RealStep.active p (RealInit.launch u) t j.val →
        v.readMem o (RealStep.address p (RealInit.launch u) t j.val) =
          ∑ k ∈ (Finset.range p.T).filter (fun k => k ≤ t),
            u.readMem s (RealStep.address p (RealInit.launch u) k j.val)) ∧
      BlockState.WriteWithin (RealLoop.footprint p (RealInit.launch u) o) u v := by
  obtain ⟨v, he, ho, hf⟩ := RealInit.original_executes p u s o dtype out_dtype
    (u.readMem s) (fun _ => rfl) hSO hBT hinj
  refine ⟨v, ?_, ?_, hf⟩
  · cases swapped
    · exact he
    · exact (RealCandidate.region_exec_equal p s o dtype out_dtype u).symm.trans he
  · intro t j ha
    simpa only [RealLoop.expected, RealMath.prefixValue, RealStep.column, ha.2, ↓reduceIte]
      using ho t j ha

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkCumsumVector.RealPadding
