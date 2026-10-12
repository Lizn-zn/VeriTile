import bench.tritonbench_g.attention_kernel.Kernels
import bench.tritonbench_g.attention_kernel.Proofs.FP
import VeriTile.Triton.Float.BroadcastRewrite
import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Meta.StatementAudit

/-! Complete-source normalization rewrite. Bias evaluation, typed dot operations,
all casts and memory effects are common context. Only div_mul_rcp is used. -/
namespace VeriTile.Bench.TritonBenchG.AttentionKernel.FPEquivalence
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec

abbrev Rules := FP.ScalarArithmetic.Rules

section
variable (storage : TypedSource.Storage)
    (q k v bias out : RegionName) (sm_scale : ℝ)
    (stride_qh stride_qm stride_qk stride_kh stride_kn stride_kk
      stride_vh stride_vk stride_vn stride_oh stride_om stride_on
      stride_b0h stride_b0m Z H N_CTX P_SEQ BIAS_LAST_SIZE B0_NUMEL
      BLOCK_DMODEL BLOCK_M BLOCK_N : Nat)

/-- At the actual normalization after the complete loop, every accumulator lane
and row denominator is finite, and each row denominator is nonzero. This checks
intermediate operands, not an output equation or a restriction to an experiment shape. -/
def divisionSite : Site :=
  .divBroadcast (Source.prefixBefore
    (TypedSource.original storage q k v bias out sm_scale
      stride_qh stride_qm stride_qk stride_kh stride_kn stride_kk
      stride_vh stride_vk stride_vn stride_oh stride_om stride_on
      stride_b0h stride_b0m Z H N_CTX P_SEQ BIAS_LAST_SIZE B0_NUMEL
      BLOCK_DMODEL BLOCK_M BLOCK_N).surfaceBody "acc" 1)
    [BLOCK_M, BLOCK_DMODEL] BLOCK_M [1]
    (.ref .real [BLOCK_M, BLOCK_DMODEL] "acc")
    (.expandDim ⟨1, by simp⟩ (.ref .real [BLOCK_M] "l_i"))

def original : Program where
  kernel := TypedSource.original storage q k v bias out sm_scale
      stride_qh stride_qm stride_qk stride_kh stride_kn stride_kk
      stride_vh stride_vk stride_vn stride_oh stride_om stride_on
      stride_b0h stride_b0m Z H N_CTX P_SEQ BIAS_LAST_SIZE B0_NUMEL
      BLOCK_DMODEL BLOCK_M BLOCK_N
  domain := [divisionSite storage q k v bias out sm_scale
      stride_qh stride_qm stride_qk stride_kh stride_kn stride_kk
      stride_vh stride_vk stride_vn stride_oh stride_om stride_on
      stride_b0h stride_b0m Z H N_CTX P_SEQ BIAS_LAST_SIZE B0_NUMEL
      BLOCK_DMODEL BLOCK_M BLOCK_N]

def reciprocal : Program :=
  { original storage q k v bias out sm_scale
      stride_qh stride_qm stride_qk stride_kh stride_kn stride_kk
      stride_vh stride_vk stride_vn stride_oh stride_om stride_on
      stride_b0h stride_b0m Z H N_CTX P_SEQ BIAS_LAST_SIZE B0_NUMEL
      BLOCK_DMODEL BLOCK_M BLOCK_N with
    kernel := TypedSource.reciprocal storage q k v bias out sm_scale
      stride_qh stride_qm stride_qk stride_kh stride_kn stride_kk
      stride_vh stride_vk stride_vn stride_oh stride_om stride_on
      stride_b0h stride_b0m Z H N_CTX P_SEQ BIAS_LAST_SIZE B0_NUMEL
      BLOCK_DMODEL BLOCK_M BLOCK_N }

specification attention_reciprocal_equiv (R : Rules) :
    original storage q k v bias out sm_scale
      stride_qh stride_qm stride_qk stride_kh stride_kn stride_kk
      stride_vh stride_vk stride_vn stride_oh stride_om stride_on
      stride_b0h stride_b0m Z H N_CTX P_SEQ BIAS_LAST_SIZE B0_NUMEL
      BLOCK_DMODEL BLOCK_M BLOCK_N ≡[R]
    reciprocal storage q k v bias out sm_scale
      stride_qh stride_qm stride_qk stride_kh stride_kn stride_kk
      stride_vh stride_vk stride_vn stride_oh stride_om stride_on
      stride_b0h stride_b0m Z H N_CTX P_SEQ BIAS_LAST_SIZE B0_NUMEL
      BLOCK_DMODEL BLOCK_M BLOCK_N := by
  exact FPEquivProofs.attention_reciprocal_equiv storage q k v bias out sm_scale
    stride_qh stride_qm stride_qk stride_kh stride_kn stride_kk
    stride_vh stride_vk stride_vn stride_oh stride_om stride_on
    stride_b0h stride_b0m Z H N_CTX P_SEQ BIAS_LAST_SIZE B0_NUMEL
    BLOCK_DMODEL BLOCK_M BLOCK_N R

end

#print_fp_assumptions attention_reciprocal_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionKernel.FPEquivalence
