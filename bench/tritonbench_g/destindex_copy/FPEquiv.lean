import bench.tritonbench_g.destindex_copy.Proofs.FP
import bench.tritonbench_g.destindex_copy.Proofs.Layout
/-! Whole-program FP equivalence of the two complete sources in Kernels.lean.
Only the exact natural-number source-row product is commuted. Both independent
payload precisions, signed destination arithmetic, all unmasked loads/stores,
complete final states and execution failure are preserved. No numerical rule
or domain premise is used; this is not a host memory-safety or GPU race proof. -/
namespace VeriTile.Bench.TritonBenchG.DestindexCopy.Equivalence
open VeriTile Triton FP.ProfiledRewrite
open scoped VeriTile.Spec

def original (p : Layout) (nopeType ropeType : TypedSource.Storage) : Program where
  kernel := p.source nopeType ropeType Bool.false
  domain := []

def indexCommuted (p : Layout) (nopeType ropeType : TypedSource.Storage) : Program where
  kernel := p.source nopeType ropeType Bool.true
  domain := []

def assumptions : Spec.Assumptions FP.GuardedFragment := []

specification index_commuted_equiv (p : Layout) (nopeType ropeType : TypedSource.Storage) :
    original p nopeType ropeType ≡[assumptions] indexCommuted p nopeType ropeType := by
  exact FPProof.equivalent nopeType ropeType p.KV_nope p.KV_rope p.Dest_loc p.O_nope p.O_rope p.stride_kv_nope_bs p.stride_kv_nope_h p.stride_kv_nope_d p.stride_kv_rope_bs p.stride_kv_rope_h p.stride_kv_rope_d p.stride_o_nope_bs p.stride_o_nope_h p.stride_o_nope_d p.stride_o_rope_bs p.stride_o_rope_h p.stride_o_rope_d p.kv_nope_head_num p.kv_rope_head_num p.BLOCK_DMODEL_NOPE p.BLOCK_DMODEL_ROPE

#print_fp_assumptions index_commuted_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DestindexCopy.Equivalence
