import bench.tritonbench_g.fast_ce_loss.Kernels
import bench.tritonbench_g.destindex_copy.Proofs.IndexRewrite
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.FastCeLoss.FPProof
open VeriTile Triton FP.ProfiledRewrite TypedSource
open scoped VeriTile.Spec
set_option maxHeartbeats 9000000
set_option maxRecDepth 25000

def assumptions : Spec.Assumptions FP.GuardedFragment := []
section
variable (storage : Storage) (cap scale : Bool)
    (logits_ptr loss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
    (V R B : Nat) (SOFTCAP LOGIT_SCALE : ℝ)

def forwardProgram (swapped : Bool) : Program where
  kernel := forward storage cap scale swapped logits_ptr loss_ptr logsumexp_ptr labels_ptr V R B SOFTCAP LOGIT_SCALE
  domain := []

theorem forward_equivalent :
    forwardProgram storage cap scale logits_ptr loss_ptr logsumexp_ptr labels_ptr V R B SOFTCAP LOGIT_SCALE Bool.false ≡[assumptions]
    forwardProgram storage cap scale logits_ptr loss_ptr logsumexp_ptr labels_ptr V R B SOFTCAP LOGIT_SCALE Bool.true := by
  cases storage <;> cases cap <;> cases scale <;> equiv_decompose <;> fp_prove

variable (C : Nat)
def chunkedProgram (swapped : Bool) : Program where
  kernel := chunked storage cap scale swapped logits_ptr loss_ptr logsumexp_ptr labels_ptr V C R B SOFTCAP LOGIT_SCALE
  domain := []

theorem chunked_equivalent :
    chunkedProgram storage cap scale logits_ptr loss_ptr logsumexp_ptr labels_ptr V R B SOFTCAP LOGIT_SCALE C Bool.false ≡[assumptions]
    chunkedProgram storage cap scale logits_ptr loss_ptr logsumexp_ptr labels_ptr V R B SOFTCAP LOGIT_SCALE C Bool.true := by
  cases storage <;> cases cap <;> cases scale <;> equiv_decompose <;> fp_prove

variable (DR : Nat)
def backwardProgram (swapped : Bool) : Program where
  kernel := backward storage cap scale swapped logits_ptr loss_ptr logsumexp_ptr labels_ptr V R DR B SOFTCAP LOGIT_SCALE
  domain := []

theorem backward_equivalent :
    backwardProgram storage cap scale logits_ptr loss_ptr logsumexp_ptr labels_ptr V R B SOFTCAP LOGIT_SCALE DR Bool.false ≡[assumptions]
    backwardProgram storage cap scale logits_ptr loss_ptr logsumexp_ptr labels_ptr V R B SOFTCAP LOGIT_SCALE DR Bool.true := by
  cases storage <;> cases cap <;> cases scale <;> equiv_decompose <;> fp_prove
end
#print_fp_assumptions forward_equivalent
#print_fp_assumptions chunked_equivalent
#print_fp_assumptions backward_equivalent
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.FastCeLoss.FPProof
