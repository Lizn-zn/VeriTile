import bench.tritonbench_g.flash_attn.Proofs.SourceBridge
import bench.tritonbench_g.flash_attn.Proofs.RealSafety
import bench.tritonbench_g.flash_attn.Proofs.RealStructural

namespace VeriTile.Bench.TritonBenchG.FlashAttn
open VeriTile.Triton Formula RealExecution NormalizedSource
noncomputable section

/-- Full flat execution transports the independently proved two-store result.
The exact source address windows, including rounded-up KV tiles, must fit the allocation. -/
theorem source_flat_correct (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (swapped : Bool) (Q K V L O : RegionName) (p : Layout) (s : BlockState)
    (hb : AddressBounds p s Q K V L O A.extent)
    (hundef : s.undef = fun _ _ => 0) (hBN : 0 < p.BLOCK_N)
    (hhi : 0 < hi p (s.pids 0)) :
    ∃ t, exec (A.flattenKernel (Candidate.source swapped Q K V L O p))
        (A.flattenState s) = some (A.flattenState t) ∧
      t.mem = (referenceState p s Q K V L O).mem ∧
      ∀ r o, ¬writeWindow p L O (s.pids 0) (s.pids 1) r o → t.mem r o = s.mem r o := by
  obtain ⟨t,ht,hm,hframe⟩ := algorithm_correct_frame Q K V L O p s hBN hhi
  have hflat := A.exec_flatten hd hcov (algorithm Q K V L O p) s
    (algorithm_safe Q K V L O p s A.extent hb hBN) (flattenOk Q K V L O p) hundef
  rw [ht] at hflat
  have horiginal : exec (A.flattenKernel (Candidate.source Bool.false Q K V L O p))
      (A.flattenState s) = some (A.flattenState t) := by
    change exec (A.flattenKernel (original Q K V L O p)) (A.flattenState s) = _
    rw [original_eq_algorithm]
    exact hflat
  refine ⟨t,?_,hm,hframe⟩
  cases swapped
  · exact horiginal
  · rw [←Candidate.flat_exec]
    exact horiginal
end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.FlashAttn
