import bench.tritonbench_g.attention_fwd_triton3.Kernels
import VeriTile.Triton.Float.BroadcastRewrite
import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Meta.StatementAudit

/-! Equivalence of the full typed source and reciprocal candidate. The sole rewrite is in the END arm. Its guards concern the actual
fp32 accumulator and row denominator after that arm's log2 update. No guard
is requested on the continuation arm. TypedExecution.lean supplies successful
execution and domain witnesses; real correctness and source IO safety are proved in the separate real layer. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.FPEquivProofs
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec
abbrev Rules := FP.ScalarArithmetic.Rules

/-- Select the final normalization by its operation, not a statement offset. -/
def isNormalization : ComputeStmt → Bool
  | .assign _ _ "acc" (.compute (.alg .fp32 (.div .real _ _ _))) => Bool.true
  | _ => Bool.false

/-- Checks each actual accumulator lane and denominator lane for finiteness,
and each denominator for nonzero, only when END selects normalization. -/
def normalizationSites (code : List ComputeStmt) (BM DM : Nat) : List Site :=
  match Source.conditional? code (fun yes => yes.any isNormalization) with
  | none => []
  | some branch => [.branch branch.before branch.condition Bool.true
      (.divBroadcast (Source.prefixBefore branch.yes "acc") [BM, DM] BM [1]
        (.ref .real [BM, DM] "acc")
        (.expandDim ⟨1, by simp⟩ (.ref .real [BM] "l_i")))]

section
variable (Q K V M Out L : RegionName) (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX
      IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N END INIT
      SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW : Nat)
    (sliding_window_offset sliding_window_size : Int)

def original : Program where
  kernel := TypedSource.original Q K V M Out L sm_scale
    stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
    stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
    Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
    END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW sliding_window_offset sliding_window_size
  domain := normalizationSites (TypedSource.original Q K V M Out L sm_scale
    stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
    stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
    Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
    END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW sliding_window_offset sliding_window_size).surfaceBody
    BLOCK_M BLOCK_DMODEL

def reciprocal : Program :=
  { original Q K V M Out L sm_scale
    stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
    stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
    Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
    END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW sliding_window_offset sliding_window_size with
    kernel := TypedSource.reciprocal Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW sliding_window_offset sliding_window_size }

theorem attention_reciprocal_equiv (R : Rules) :
    original Q K V M Out L sm_scale
    stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
    stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
    Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
    END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW sliding_window_offset sliding_window_size ≡[R]
    reciprocal Q K V M Out L sm_scale
    stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
    stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
    Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
    END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW sliding_window_offset sliding_window_size := by
  equiv_decompose
  all_goals fp_prove

end

#print_fp_assumptions attention_reciprocal_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.FPEquivProofs
