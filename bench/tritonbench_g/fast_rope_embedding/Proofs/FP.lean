import bench.tritonbench_g.fast_rope_embedding.Kernels
import bench.tritonbench_g.destindex_copy.Proofs.IndexRewrite
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.FastRopeEmbedding.FPProof
open VeriTile Triton FP.ProfiledRewrite TypedSource
open scoped VeriTile.Spec
set_option maxHeartbeats 10000000
set_option maxRecDepth 25000

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def program (input cosine sine : Storage) (swapped : Bool) (Q C S : RegionName)
    (QR CR SR seq H heads B : Nat) (backwards : Bool) : Program where
  kernel := rope input cosine sine swapped Q C S QR CR SR seq H heads B backwards
  domain := []

theorem equivalent (input cosine sine : Storage) (Q C S : RegionName)
    (QR CR SR seq H heads B : Nat) (backwards : Bool) :
    program input cosine sine Bool.false Q C S QR CR SR seq H heads B backwards ≡[assumptions]
    program input cosine sine Bool.true Q C S QR CR SR seq H heads B backwards := by
  cases input <;> cases cosine <;> cases sine <;> cases backwards <;> equiv_decompose <;> fp_prove

#print_fp_assumptions equivalent
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.FastRopeEmbedding.FPProof
