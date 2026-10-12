import bench.tritonbench_g.destindex_copy_kv2.Proofs.FP
import bench.tritonbench_g.destindex_copy_kv2.Proofs.Layout
/-! Whole-program equivalence of the complete typed sources in Kernels.lean.
Only the exact natural-number source-row product is commuted. Input/output
precision, the destination cast, signed metadata, the head mask, zero load default,
complete final states and failure are preserved. No numerical atom or domain
premise is required. This is not a host memory-safety or concurrent-launch proof. -/
namespace VeriTile.Bench.TritonBenchG.DestindexCopyKv2.Equivalence
open VeriTile Triton FP.ProfiledRewrite
open scoped VeriTile.Spec

def original (p : Layout) (input output : TypedSource.Storage) : Program where
  kernel := p.source input output Bool.false
  domain := []

def indexCommuted (p : Layout) (input output : TypedSource.Storage) : Program where
  kernel := p.source input output Bool.true
  domain := []

def assumptions : Spec.Assumptions FP.GuardedFragment := []

specification index_commuted_equiv (p : Layout) (input output : TypedSource.Storage) :
    original p input output ≡[assumptions] indexCommuted p input output := by
  exact FPProof.equivalent input output p.K p.Dest_loc p.Out p.stride_k_bs p.stride_k_h p.stride_k_d p.stride_o_bs p.stride_o_h p.stride_o_d p.head_num p.BLOCK_DMODEL p.BLOCK_HEAD

#print_fp_assumptions index_commuted_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DestindexCopyKv2.Equivalence
