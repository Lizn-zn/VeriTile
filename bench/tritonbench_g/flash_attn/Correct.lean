import bench.tritonbench_g.flash_attn.Proofs.RealWitness
import bench.tritonbench_g.flash_attn.Proofs.LegacyReal

/-! Full-source real contracts for both typed programs. The independent
formula uses the source's exact decimal scale, source-used strides, causal
mask, and rounded-up physical KV window. The two-output reference scatters L
then O in source lane order, so arbitrary output aliasing remains covered.
Legacy theorem signatures and their historical hypotheses are imported above. -/
namespace VeriTile.Bench.TritonBenchG.FlashAttn.Correctness
open VeriTile.Triton TypedSource Formula RealExecution
noncomputable section

specification original_correctness (input output : Storage) (Q K V L O : RegionName) (p : Layout)
    (stride_q_bs stride_k_bs stride_k_head stride_v_bs stride_v_head stride_v_seqlen
      stride_o_bs stride_o_head BS HEAD : Nat)
    (s : BlockState) (hBN : 0 < p.BLOCK_N) (hhi : 0 < hi p (s.pids 0)) :
    ∃ t, exec (kernel input output Bool.false Q K V L O p.sm_scale
      stride_q_bs p.stride_q_head p.stride_q_seqlen p.stride_q_dim
      stride_k_bs stride_k_head p.stride_k_seqlen p.stride_k_dim
      stride_v_bs stride_v_head stride_v_seqlen p.stride_v_dim
      stride_o_bs stride_o_head p.stride_o_seqlen p.stride_o_dim
      BS HEAD p.SEQLEN p.BLOCK_M p.DIM p.BLOCK_N p.causal).eraseDType.toAlgKernel s = some t ∧
      t.mem = (referenceState p s Q K V L O).mem ∧
      ∀ r o, ¬writeWindow p L O (s.pids 0) (s.pids 1) r o → t.mem r o = s.mem r o := by
  rw [full_source_erasure]
  exact source_real_correct Bool.false Q K V L O p s hBN hhi

specification candidate_correctness (input output : Storage) (Q K V L O : RegionName) (p : Layout)
    (stride_q_bs stride_k_bs stride_k_head stride_v_bs stride_v_head stride_v_seqlen
      stride_o_bs stride_o_head BS HEAD : Nat)
    (s : BlockState) (hBN : 0 < p.BLOCK_N) (hhi : 0 < hi p (s.pids 0)) :
    ∃ t, exec (kernel input output Bool.true Q K V L O p.sm_scale
      stride_q_bs p.stride_q_head p.stride_q_seqlen p.stride_q_dim
      stride_k_bs stride_k_head p.stride_k_seqlen p.stride_k_dim
      stride_v_bs stride_v_head stride_v_seqlen p.stride_v_dim
      stride_o_bs stride_o_head p.stride_o_seqlen p.stride_o_dim
      BS HEAD p.SEQLEN p.BLOCK_M p.DIM p.BLOCK_N p.causal).eraseDType.toAlgKernel s = some t ∧
      t.mem = (referenceState p s Q K V L O).mem ∧
      ∀ r o, ¬writeWindow p L O (s.pids 0) (s.pids 1) r o → t.mem r o = s.mem r o := by
  rw [full_source_erasure]
  exact source_real_correct Bool.true Q K V L O p s hBN hhi

specification flat_correctness (swapped : Bool) (input output : Storage) (Q K V L O : RegionName) (p : Layout)
    (stride_q_bs stride_k_bs stride_k_head stride_v_bs stride_v_head stride_v_seqlen
      stride_o_bs stride_o_head BS HEAD : Nat)
    (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (s : BlockState) (hb : AddressBounds p s Q K V L O A.extent)
    (hundef : s.undef = fun _ _ => 0)
    (hBN : 0 < p.BLOCK_N) (hhi : 0 < hi p (s.pids 0)) :
    ∃ t, exec (A.flattenKernel (kernel input output swapped Q K V L O p.sm_scale
      stride_q_bs p.stride_q_head p.stride_q_seqlen p.stride_q_dim
      stride_k_bs stride_k_head p.stride_k_seqlen p.stride_k_dim
      stride_v_bs stride_v_head stride_v_seqlen p.stride_v_dim
      stride_o_bs stride_o_head p.stride_o_seqlen p.stride_o_dim
      BS HEAD p.SEQLEN p.BLOCK_M p.DIM p.BLOCK_N p.causal).eraseDType.toAlgKernel) (A.flattenState s) =
        some (A.flattenState t) ∧
      t.mem = (referenceState p s Q K V L O).mem ∧
      ∀ r o, ¬writeWindow p L O (s.pids 0) (s.pids 1) r o → t.mem r o = s.mem r o := by
  rw [full_source_erasure]
  exact source_flat_correct A hd hcov swapped Q K V L O p s hb hundef hBN hhi

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.FlashAttn.Correctness
