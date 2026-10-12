import bench.tritonbench_g.attn_fwd_causal.Kernels
import bench.tritonbench_g.attn_fwd_causal.Proofs.FP
import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Meta.StatementAudit

/-! Full-source FP equivalence for the final division-to-reciprocal rewrite.
Both kernels are in Kernels.lean, with fp16 Q/K/V/Out, fp32 scales and
accumulators, the fp16 probability cast and fp16 value-dot result retained.
The domain checks actual fp32 accumulator/denominator lanes after both helper
loops, including masked lanes evaluated by the normalization. It requires
finite operands and nonzero denominators; no input-sign restriction, fixed
experiment shape or output equation is assumed. TypedExecution supplies a
consistent-model execution witness; it is not a new GPU experiment. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.FPEquivalence
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
    (hStage : STAGE = 2 ∨ STAGE = 3)

/-- Check actual fp32 accumulators and row denominators after the full loop.
Every evaluated lane is finite and every row denominator is nonzero. This adds
no positivity, shape restriction, output equation or dot/reduction identity. -/
def divisionSite : Site :=
  .divBroadcast (Source.prefixBefore
    (TypedSource.original q k v qScale kScale out
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE hStage).surfaceBody "acc" 1)
    [BLOCK_M, BLOCK_DMODEL] BLOCK_M [1]
    (.ref .real [BLOCK_M, BLOCK_DMODEL] "acc")
    (.expandDim ⟨1, by simp⟩ (.ref .real [BLOCK_M] "l_i"))

def original : Program where
  kernel := TypedSource.original q k v qScale kScale out
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE hStage
  domain := [divisionSite q k v qScale kScale out
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE hStage]

def reciprocal : Program :=
  { original q k v qScale kScale out
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE hStage with
    kernel := TypedSource.reciprocal q k v qScale kScale out
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE hStage }

specification attention_reciprocal_equiv (R : Rules) :
    original q k v qScale kScale out
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE hStage ≡[R]
    reciprocal q k v qScale kScale out
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE hStage := by
  exact FPEquivProofs.attention_reciprocal_equiv q k v qScale kScale out
    stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
    stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
    Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE hStage R

end

#print_fp_assumptions attention_reciprocal_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.FPEquivalence
