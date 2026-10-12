import bench.tritonbench_g.attention_forward_triton.Kernels
import VeriTile.Triton.Float.BroadcastRewrite
import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Meta.StatementAudit

/-! FP proof for the complete typed source and reciprocal candidate. This is
shared by the public FPEquiv specification and its execution witness.
The prefix includes the complete
attention loop, casts, reductions and both typed dot products. -/
namespace VeriTile.Bench.TritonBenchG.AttentionForwardTriton.FPEquivProofs
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec

abbrev Rules := FP.ScalarArithmetic.Rules

section
variable (q k v qScale kScale out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE : Nat)

/-- Check actual fp32 accumulators and row denominators after the full loop.
Every evaluated lane is finite and every row denominator is nonzero. This adds
no positivity, shape restriction, output equation or dot/reduction identity. -/
def divisionSite : Site :=
  .divBroadcast (Source.prefixBefore
    (TypedSource.original q k v qScale kScale out
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE).surfaceBody "acc" 1)
    [BLOCK_M, BLOCK_DMODEL] BLOCK_M [1]
    (.ref .real [BLOCK_M, BLOCK_DMODEL] "acc")
    (.expandDim ⟨1, by simp⟩ (.ref .real [BLOCK_M] "l_i"))

def original : Program where
  kernel := TypedSource.original q k v qScale kScale out
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE
  domain := [divisionSite q k v qScale kScale out
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE]

def reciprocal : Program :=
  { original q k v qScale kScale out
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE with
    kernel := TypedSource.reciprocal q k v qScale kScale out
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE }

theorem attention_reciprocal_equiv (R : Rules) :
    original q k v qScale kScale out
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE ≡[R]
    reciprocal q k v qScale kScale out
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE := by
  equiv_decompose
  all_goals fp_prove

end

#print_fp_assumptions attention_reciprocal_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionForwardTriton.FPEquivProofs
