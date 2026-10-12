import bench.tritonbench_g.bgmv_expand_slice.Kernels
import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Meta.StatementAudit

/-! Equivalence of the complete typed programs, for every storage/cast choice.
The domain follows the actual non-sentinel branch, each reached output-loop
iteration and the ADD_INPUTS arm. Only its actual fp32 addition is commuted;
typed products, conversions, reduction, addresses, masks and store remain.
The independent real IO contract is proved separately in Correct.lean. -/
namespace VeriTile.Bench.TritonBenchG.BgmvExpandSlice.FPEquivProofs
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec
set_option maxHeartbeats 2200000
set_option maxRecDepth 10000

/-- Extract exactly the scalar-addition site supported by the admitted rule.
The symbolic source check below rejects missing sites in every specialization.
Both operands keep the precision context of the actual assignment. -/
def additionSite? (code : List ComputeStmt) : Option Site :=
  let before := Source.prefixBefore code "accumulator"
  match (code.drop before.length).head? with
  | some (.assign _ _ _ (.alg (.add .real (.consSame .nil) left right))) =>
      some (.add before _ left right)
  | some (.assign _ _ _ (.compute (.alg .fp32 (.add .real (.consSame .nil) left right)))) =>
      some (.addFP32 before _ left right)
  | _ => none

def requirementsFor (kernel : ComputeKernel) : List Site :=
  (do
    let active ← Source.conditional? kernel.surfaceBody
      (fun body => (Source.dynamicRange? body "n").isSome)
    let loop ← Source.dynamicRange? active.yes "n"
    let accumulation ← Source.conditional? loop.body (fun _ => Bool.true)
    let addition ← additionSite? accumulation.yes
    return Site.branch active.before active.condition Bool.true
      (.range loop.before "n" loop.start loop.stop loop.stride loop.body
        (.branch accumulation.before accumulation.condition Bool.true addition))).toList

section
variable (input : TypedSource.InputStorage) (weight : TypedSource.WeightStorage)
    (CAST_TYPE : Bool) (out_dtype : FloatDType)
    (x w out : RegionName) (N K : Nat) (indices : Region .int)
    (sx sk sw0 swn swk sy syn slice BN BK splits : Nat) (EVEN_K ADD_INPUTS : Bool)

def original : Program where
  kernel := TypedSource.original input weight CAST_TYPE out_dtype x w out N K indices
    sx sk sw0 swn swk sy syn slice BN BK splits EVEN_K ADD_INPUTS
  domain := requirementsFor (TypedSource.original input weight CAST_TYPE out_dtype x w out N K indices
    sx sk sw0 swn swk sy syn slice BN BK splits EVEN_K ADD_INPUTS)

def outputFirst : Program :=
  { original input weight CAST_TYPE out_dtype x w out N K indices sx sk sw0 swn swk sy syn slice BN BK splits EVEN_K ADD_INPUTS with
    kernel := TypedSource.outputFirst input weight CAST_TYPE out_dtype x w out N K indices
      sx sk sw0 swn swk sy syn slice BN BK splits EVEN_K ADD_INPUTS }

/-- A failed source locator cannot silently remove the required domain. -/
theorem requirements_present :
    (original input weight CAST_TYPE out_dtype x w out N K indices
      sx sk sw0 swn swk sy syn slice BN BK splits EVEN_K ADD_INPUTS).domain.length = 1 := by
  cases input <;> cases weight <;> cases CAST_TYPE <;> rfl

theorem output_first_equiv (R : FP.ScalarArithmetic.Rules) :
    original input weight CAST_TYPE out_dtype x w out N K indices
      sx sk sw0 swn swk sy syn slice BN BK splits EVEN_K ADD_INPUTS ≡[R]
    outputFirst input weight CAST_TYPE out_dtype x w out N K indices
      sx sk sw0 swn swk sy syn slice BN BK splits EVEN_K ADD_INPUTS := by
  cases input <;> cases weight <;> cases CAST_TYPE
  all_goals equiv_decompose
  all_goals fp_prove

end
#print_fp_assumptions output_first_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.BgmvExpandSlice.FPEquivProofs
