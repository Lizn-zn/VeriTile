import bench.tritonbench_g.destindex_copy.Proofs.RealIO
import bench.tritonbench_g.destindex_copy.Proofs.Candidate
namespace VeriTile.Bench.TritonBenchG.DestindexCopy.SourceIO
open VeriTile.Triton
open scoped RealIO
noncomputable section
set_option maxHeartbeats 2000000

def sourceIO (p : Layout) (nopeType ropeType : TypedSource.Storage) (swapped : Bool) : RealIO.IO p :=
  { RealIO.referenceIO p with
    kernel := (p.source nopeType ropeType swapped).eraseDType
    projection := by
      have h : (p.source nopeType ropeType swapped).toAlgorithm? =
          .ok (p.source nopeType ropeType swapped).toAlgKernel := by
        cases nopeType <;> cases ropeType <;> cases swapped <;> rfl
      simp only [ComputeKernel.eraseDType,h,ComputeKernel.toAlgorithm?_fromKernelBody,
        ComputeKernel.toAlgKernel_fromKernelBody] }

theorem correct (p : Layout) (nopeType ropeType : TypedSource.Storage) (swapped : Bool)
    (hOut : p.O_nope ≠ p.O_rope)
    (hNope : Function.Injective (fun j : Fin p.BLOCK_DMODEL_NOPE => p.stride_o_nope_d*j.val))
    (hRope : Function.Injective (fun j : Fin p.BLOCK_DMODEL_ROPE => p.stride_o_rope_d*j.val)) :
    sourceIO p nopeType ropeType swapped ⊨ fun xs => (xs.nope,xs.rope) := by
  intro A hd hregs hc s xs hr hu hb hp
  obtain ⟨t,he,hn,hro,hf⟩ := RealIO.reference_correct p hOut hNope hRope A hd hregs hc s xs hr hu ⟨hb.metadata,hb.nope,hb.rope,hb.outNope,hb.outRope⟩ ⟨hp.metadata,hp.nope,hp.rope⟩
  refine ⟨t,?_,hn,hro,hf⟩
  have ho := Projection.original_exec p.KV_nope p.KV_rope p.Dest_loc p.O_nope p.O_rope p.stride_kv_nope_bs p.stride_kv_nope_h p.stride_kv_nope_d p.stride_kv_rope_bs p.stride_kv_rope_h p.stride_kv_rope_d p.stride_o_nope_bs p.stride_o_nope_h p.stride_o_nope_d p.stride_o_rope_bs p.stride_o_rope_h p.stride_o_rope_d p.kv_nope_head_num p.kv_rope_head_num p.BLOCK_DMODEL_NOPE p.BLOCK_DMODEL_ROPE A nopeType ropeType (A.flattenState s)
  cases swapped
  · exact ho.trans he
  · have heq := Candidate.flat_exec_equal A nopeType ropeType p.KV_nope p.KV_rope p.Dest_loc p.O_nope p.O_rope p.stride_kv_nope_bs p.stride_kv_nope_h p.stride_kv_nope_d p.stride_kv_rope_bs p.stride_kv_rope_h p.stride_kv_rope_d p.stride_o_nope_bs p.stride_o_nope_h p.stride_o_nope_d p.stride_o_rope_bs p.stride_o_rope_h p.stride_o_rope_d p.kv_nope_head_num p.kv_rope_head_num p.BLOCK_DMODEL_NOPE p.BLOCK_DMODEL_ROPE (A.flattenState s)
    exact heq.symm.trans (ho.trans he)
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DestindexCopy.SourceIO
