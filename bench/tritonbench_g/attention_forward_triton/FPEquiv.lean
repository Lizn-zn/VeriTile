import bench.tritonbench_g.attention_forward_triton.Kernels
import bench.tritonbench_g.attention_forward_triton.Proofs.FP
import VeriTile.Triton.Float.BroadcastRewrite
import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Meta.StatementAudit

/-! Conditional FP equivalence of the complete sources in Kernels.lean.
Q/K/V/Out use fp16 storage; both scale buffers use fp32. The second dot has
explicit fp16 output and is promoted into the fp32 accumulator. The common
loop, exp2, reduction order, casts, mask and final store are preserved.

The normalization site checks the actual fp32 accumulator and denominators
after the loop. This proof is conditional on admitted scalar div_mul_rcp;
it makes no bitwise IEEE or GPU performance claim. -/
namespace VeriTile.Bench.TritonBenchG.AttentionForwardTriton.FPEquivalence
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

specification attention_reciprocal_equiv (R : Rules) :
    original q k v qScale kScale out
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE ≡[R]
    reciprocal q k v qScale kScale out
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE := by
  exact FPEquivProofs.attention_reciprocal_equiv q k v qScale kScale out
    stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
    stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
    Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE R

end

#print_fp_assumptions attention_reciprocal_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionForwardTriton.FPEquivalence
