/-
TritonBench vector_addition: correctness is real; equivalence uses admitted
floating-point atom assumptions. The original real proof is imported unchanged.

The original kernel and the variant below differ only at `output = x + y`.
The ADD-COMMUTE row checks this local rewrite under its numerical configuration;
Lean then lifts the admitted assumption through the unchanged loads and store.
N=98432, BLOCK_SIZE=1024 is the original first test's fp32 instance.
No two-gates experiment has run; the theorem requires a model `R` supplying
the admitted ADD-COMMUTE atom. It does not construct such a model.
-/
import bench.tritonbench_g.vector_addition.VectorAddition
import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Float.Equivalence

namespace VeriTile.Bench.Examples.TritonBenchVectorAdditionFP

open VeriTile Triton
open VeriTile.Bench.TritonBenchG.VectorAddition
open scoped VeriTile.Spec

def nElements : Nat := 98432
def blockSize : Nat := 1024

def originalKernel : ComputeKernel :=
  add_kernel "x" "y" "output" nElements blockSize

/-- The existing TritonBench kernel with just the addition operands exchanged. -/
def optimizedKernel : ComputeKernel :=
  let x_ptr : RegionName := "x"
  let y_ptr : RegionName := "y"
  let output_ptr : RegionName := "output"
  triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(blockSize)
  offsets = block_start + tl.arange(0, $(blockSize))
  mask = offsets < $(nElements)
  x = tl.load(x_ptr + offsets, mask=mask)
  y = tl.load(y_ptr + offsets, mask=mask)
  output = y + x
  tl.store(output_ptr + offsets, output, mask=mask)
}

abbrev body := Spec.ProgramSyntax.body (Program := ComputeKernel)

/-- These are the actual single-assignment fragments from the two kernels. -/
def originalAdd : List ComputeStmt := (body originalKernel).drop 6 |>.take 1
def optimizedAdd : List ComputeStmt := (body optimizedKernel).drop 6 |>.take 1

def beforeAdd : List ComputeStmt := (body originalKernel).take 6
def afterAdd : List ComputeStmt := (body originalKernel).drop 7

/-- This exact syntax decomposition checks that only the one atom changed. -/
theorem original_decomposition :
    body originalKernel = beforeAdd ++ originalAdd ++ afterAdd := rfl

theorem optimized_decomposition :
    body optimizedKernel = beforeAdd ++ optimizedAdd ++ afterAdd := rfl

/-- Numerical configuration is supplied by the rule-table entry. It must bind
this ADD-COMMUTE instance, fp32 input/operation/output, shape and mask, the
operand distribution at this use site, actual backend, and full gate protocol.
The configuration JSON and key must agree with the fixed rule ID; validation
and replay are required by `AcceptedAtom`. No PASS data is manufactured here. -/
@[spec_rule] def addCommute (experiment : Spec.Contract) (evidence : Spec.Evidence) :
    Spec.RuleEntry ComputeStmt where
  rule := {
    lhs := originalAdd
    rhs := optimizedAdd
    contract := { experiment with ruleID := "ADD-COMMUTE" } }
  evidence := evidence

/-- The rule model used by this example. Experiment bookkeeping belongs here,
not in the public specification. Constructing a model requires atom admission;
this file neither supplies unchecked evidence nor declares a global axiom. -/
structure Rules where
  contract : Spec.Contract
  evidence : Spec.Evidence
  add_comm : Spec.AcceptedAtom (addCommute contract evidence)

def Rules.assumptions (R : Rules) : Spec.Assumptions ComputeStmt :=
  [addCommute R.contract R.evidence]

instance : Coe Rules (Spec.Assumptions (Spec.ProgramSyntax.Statement ComputeKernel)) :=
  ⟨Rules.assumptions⟩

/-- The atom is used here, not a mathematical `add_comm` lemma and not an
assumed whole-kernel result. Both gates and evidence validation are required
by `R.add_comm`; `#print_spec` exposes this model assumption. -/
@[spec_rule] theorem admitted_add_commute (R : Rules) :
    Spec.Derivation R.assumptions originalAdd optimizedAdd := by
  exact .atom (addCommute R.contract R.evidence) (by simp [Rules.assumptions]) R.add_comm

/-- Public implementation equivalence UNDER the two-gates-admitted atom table.
Correctness against the real addition formula remains `add_kernel_correctness`
in the imported TritonBench file. This conclusion is an equational derivation,
not an IEEE bit-equality theorem or a whole-kernel experiment result. -/
specification vector_addition_fp_equiv (R : Rules) :
    originalKernel ≡[R] optimizedKernel := by
  refine ⟨rfl, ?_⟩
  change Spec.Derivation R.assumptions (body originalKernel) (body optimizedKernel)
  rw [original_decomposition, optimized_decomposition]
  exact .frame beforeAdd afterAdd (admitted_add_commute R)

#print_spec vector_addition_fp_equiv
#axiomsClean vector_addition_fp_equiv
#auditModuleAxioms

end VeriTile.Bench.Examples.TritonBenchVectorAdditionFP
