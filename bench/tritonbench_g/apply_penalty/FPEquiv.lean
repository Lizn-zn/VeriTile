import bench.tritonbench_g.apply_penalty.Kernels
import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Triton.Float.ExpressionRewrite
import VeriTile.Meta.StatementAudit

/-! Complete fp32 program equivalence of the sources in Kernels.lean.
The original and reciprocal programs use the same gather/scatter addresses,
metadata slots, count promotion, comparison, masks and in-place store.
Program selects default fp32 precision. The proof includes execution failure;
both tl.where arms are evaluated, and either comparison outcome is allowed. -/
namespace VeriTile.Bench.TritonBenchG.ApplyPenalty.FPEquiv
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.ApplyPenalty
open scoped VeriTile.Spec

abbrev Rules := FP.ScalarArithmetic.Rules

/-- Finite actual logits (including evaluated masked lanes, loaded as zero) and
a finite nonzero loaded repetition scalar at the division site. Negative logits
and negative repetition penalties are allowed; no sign-branch result is assumed. -/
def divisionSite (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b stride_logit_s BLOCK_P : Nat) : Site :=
  .divScalar (Source.prefixBefore (apply_penalty Logits presence_penalty freqency_penalty
      repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
      stride_logit_b stride_logit_s BLOCK_P).surfaceBody
    "rep_logits") BLOCK_P [] (.ref .real [BLOCK_P] "cur_logits")
    (.ref .real [] "cur_repetition")

def original (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b stride_logit_s BLOCK_P : Nat) : Program where
  kernel := apply_penalty Logits presence_penalty freqency_penalty
      repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
      stride_logit_b stride_logit_s BLOCK_P
  domain := [divisionSite Logits presence_penalty freqency_penalty
      repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
      stride_logit_b stride_logit_s BLOCK_P]

def reciprocal (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b stride_logit_s BLOCK_P : Nat) : Program :=
  { original Logits presence_penalty freqency_penalty
      repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
      stride_logit_b stride_logit_s BLOCK_P with
    kernel := apply_penalty_reciprocal Logits presence_penalty freqency_penalty
      repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
      stride_logit_b stride_logit_s BLOCK_P }

/-- Admitted div_mul_rcp inside tl.where preserves the whole computation.
The FP contextual equality also preserves repeated-index store order; the real
independent per-lane output formula separately requires active-index injectivity. -/
specification apply_penalty_reciprocal_equiv
    (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b stride_logit_s BLOCK_P : Nat) (R : Rules) :
    original Logits presence_penalty freqency_penalty
      repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
      stride_logit_b stride_logit_s BLOCK_P ≡[R]
    reciprocal Logits presence_penalty freqency_penalty
      repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
      stride_logit_b stride_logit_s BLOCK_P := by
  equiv_decompose
  all_goals fp_prove

#print_fp_assumptions apply_penalty_reciprocal_equiv
#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.TritonBenchG.ApplyPenalty.FPEquiv
