import bench.tritonbench_g.bmm_optimized.Kernels
import bench.tritonbench_g.bmm_optimized.Proofs.FP
import VeriTile.Meta.StatementAudit

/-! Whole-program FP equivalence of the original and dot-first candidate.
Only each reached fp32 `o + dot(a,b)` becomes `dot(a,b) + o`. A/B/O retain
their common fp32/fp16/bf16 storage, fp32 dot accumulation, allow_tf32=false,
implicit output cast, all mask choices, omitted `other` and CTA reordering.

The domain checks actual finite carry/dot operands in every reached K
iteration, after its loads and pointer updates. Unlike the real product
specification, this equivalence does not require zero unspecified loads:
both programs consume the same values. The only numerical atom is
add_commute. Full state and failure are retained under the admitted rule;
no dot reassociation or end-to-end error bound is asserted. -/
namespace VeriTile.Bench.TritonBenchG.BmmOptimized.Equivalence
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec

section
variable (dtype : TypedSource.Storage) (divM divN divK : Bool)
    (A B O : RegionName) (M N K BM BN BK GROUP_M : Nat)

def original : Program where
  kernel := TypedSource.original dtype divM divN divK A B O M N K BM BN BK GROUP_M
  domain := FPEquivProofs.requirementsFor
    (TypedSource.original dtype divM divN divK A B O M N K BM BN BK GROUP_M) dtype BM BN BK

def dotFirst : Program :=
  { original dtype divM divN divK A B O M N K BM BN BK GROUP_M with
    kernel := TypedSource.dotFirst dtype divM divN divK A B O M N K BM BN BK GROUP_M }

specification bmm_dot_first_equiv (R : FP.ScalarArithmetic.Rules) :
    original dtype divM divN divK A B O M N K BM BN BK GROUP_M ≡[R]
    dotFirst dtype divM divN divK A B O M N K BM BN BK GROUP_M := by
  exact FPEquivProofs.dot_first_equiv dtype divM divN divK A B O M N K BM BN BK GROUP_M R

end
#print_fp_assumptions bmm_dot_first_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.BmmOptimized.Equivalence
