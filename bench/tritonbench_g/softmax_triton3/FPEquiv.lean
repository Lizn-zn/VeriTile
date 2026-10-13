import bench.tritonbench_g.softmax_triton3.Kernels
import bench.tritonbench_g.softmax_triton3.Proofs.FP
import VeriTile.Meta.StatementAudit

/-! Complete typed softmax equivalence. Input/output storage and optional mask
storage range independently over fp32, fp16 and bf16. All fp32 conversions,
opaque exp, reduction order, padding, masks and stores remain unchanged. -/
namespace VeriTile.Bench.TritonBenchG.SoftmaxTriton3.FPEquiv
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec

/-- Finiteness of actual computed numerators and finiteness/nonzero of the actual
row sum, including all evaluated inactive lanes. The prefix includes the optional
additive mask and its cast. No intrinsic exponential identity is assumed. -/
def divisionSite (inputType maskType : Storage) (out inp mask : RegionName)
    (stride n B : Nat) (hasMask : Bool) : Site :=
  .divScalar (Source.prefixBefore
    (typedKernel inputType maskType Bool.false out inp mask stride n B hasMask).surfaceBody
    "softmax_output") B [] (.ref .real [B] "numerator") (.ref .real [] "denominator")

def original (inputType maskType : Storage) (out inp mask : RegionName)
    (stride n B : Nat) (hasMask : Bool) : Program where
  kernel := typedKernel inputType maskType Bool.false out inp mask stride n B hasMask
  domain := [divisionSite inputType maskType out inp mask stride n B hasMask]

def reciprocal (inputType maskType : Storage) (out inp mask : RegionName)
    (stride n B : Nat) (hasMask : Bool) : Program :=
  { original inputType maskType out inp mask stride n B hasMask with
    kernel := typedKernel inputType maskType Bool.true out inp mask stride n B hasMask }

specification softmax_reciprocal_equiv (inputType maskType : Storage)
    (out inp mask : RegionName) (stride n B : Nat) (hasMask : Bool)
    (R : FP.ScalarArithmetic.Rules) :
    original inputType maskType out inp mask stride n B hasMask ≡[R]
      reciprocal inputType maskType out inp mask stride n B hasMask := by
  cases inputType <;> cases maskType <;> cases hasMask <;> equiv_decompose
  all_goals fp_prove

#print_fp_assumptions softmax_reciprocal_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxTriton3.FPEquiv
