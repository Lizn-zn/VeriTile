import bench.tritonbench_g.destindex_copy_kv2.Proofs.RealIO
import bench.tritonbench_g.destindex_copy_kv2.Proofs.Candidate
namespace VeriTile.Bench.TritonBenchG.DestindexCopyKv2.SourceIO
open VeriTile.Triton
open scoped RealIO
noncomputable section
set_option maxHeartbeats 2000000

def sourceIO (p : Layout) (input output : TypedSource.Storage) (swapped : Bool) : RealIO.IO p :=
  { RealIO.referenceIO p with
    kernel := (p.source input output swapped).eraseDType
    projection := by
      have h : (p.source input output swapped).toAlgorithm? = .ok (p.source input output swapped).toAlgKernel := by
        cases input <;> cases output <;> cases swapped <;> rfl
      simp only [ComputeKernel.eraseDType,h,ComputeKernel.toAlgorithm?_fromKernelBody,
        ComputeKernel.toAlgKernel_fromKernelBody] }

theorem correct (p : Layout) (input output : TypedSource.Storage) (swapped : Bool)
    (hLayout : Function.Injective (fun j : TileIndex [p.BLOCK_HEAD,p.BLOCK_DMODEL] =>
      p.stride_o_h*j.1.val + p.stride_o_d*j.2.1.val)) :
    sourceIO p input output swapped ⊨ fun xs => xs.values := by
  intro A hd hregs hc s xs hr hu hb hp
  obtain ⟨t,he,hv,hf⟩ := RealIO.reference_correct p hLayout A hd hregs hc s xs hr hu
    ⟨hb.metadata,hb.input,hb.output⟩ ⟨hp.metadata,hp.input⟩
  refine ⟨t,?_,hv,hf⟩
  have ho := Projection.original_exec p.K p.Dest_loc p.Out p.stride_k_bs p.stride_k_h p.stride_k_d p.stride_o_bs p.stride_o_h p.stride_o_d p.head_num p.BLOCK_DMODEL p.BLOCK_HEAD A input output (A.flattenState s)
  cases swapped
  · exact ho.trans he
  · have heq := Candidate.flat_exec_equal A input output p.K p.Dest_loc p.Out p.stride_k_bs p.stride_k_h p.stride_k_d p.stride_o_bs p.stride_o_h p.stride_o_d p.head_num p.BLOCK_DMODEL p.BLOCK_HEAD (A.flattenState s)
    exact heq.symm.trans (ho.trans he)
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DestindexCopyKv2.SourceIO
