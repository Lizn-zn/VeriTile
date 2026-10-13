import bench.tritonbench_g.flash_attn.Proofs.Erasure
import bench.tritonbench_g.flash_attn.Proofs.Candidate
import bench.tritonbench_g.flash_attn.Proofs.RealComplete

/-! All source parameters and supported storage types reduce to the same
real algorithm. Only parameters unused by the Python kernel are discarded. -/
namespace VeriTile.Bench.TritonBenchG.FlashAttn
open VeriTile.Triton TypedSource Formula
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 30000

theorem full_source_erasure (input output : Storage) (swapped : Bool)
    (Q K V L O : RegionName) (p : Layout)
    (stride_q_bs stride_k_bs stride_k_head stride_v_bs stride_v_head stride_v_seqlen
      stride_o_bs stride_o_head BS HEAD : Nat) :
    (kernel input output swapped Q K V L O p.sm_scale
      stride_q_bs p.stride_q_head p.stride_q_seqlen p.stride_q_dim
      stride_k_bs stride_k_head p.stride_k_seqlen p.stride_k_dim
      stride_v_bs stride_v_head stride_v_seqlen p.stride_v_dim
      stride_o_bs stride_o_head p.stride_o_seqlen p.stride_o_dim
      BS HEAD p.SEQLEN p.BLOCK_M p.DIM p.BLOCK_N p.causal).eraseDType.toAlgKernel =
    Candidate.source swapped Q K V L O p := by
  rw [Erasure.body]
  cases swapped <;> rfl

theorem source_real_correct (swapped : Bool) (Q K V L O : RegionName)
    (p : Layout) (s : BlockState) (hBN : 0 < p.BLOCK_N)
    (hhi : 0 < hi p (s.pids 0)) :
    ∃ t, exec (Candidate.source swapped Q K V L O p) s = some t ∧
      t.mem = (referenceState p s Q K V L O).mem ∧
      ∀ r o, ¬writeWindow p L O (s.pids 0) (s.pids 1) r o → t.mem r o = s.mem r o := by
  obtain ⟨t,ht,hm,hframe⟩ := RealExecution.algorithm_correct_frame Q K V L O p s hBN hhi
  refine ⟨t,?_,hm,hframe⟩
  have horiginal : exec (Candidate.source Bool.false Q K V L O p) s = some t := by
    change exec (NormalizedSource.original Q K V L O p) s = some t
    rw [NormalizedSource.original_eq_algorithm]
    exact ht
  cases swapped
  · exact horiginal
  · rw [←Candidate.source_exec]
    exact horiginal

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.FlashAttn
