/-
TritonBench vector_addition, instantiated at the PR #9 numerical profile:
4096 x 4096 elements, block 1024, fp32 input/compute/output, Normal(1,1).
The source kernel is specialized to typed fp32 regions; only x+y changes to y+x.
The real correctness theorem remains in the imported TritonBench source.

The numerical model R trusts the published ADD-COMMUTE row for the exact
fragments below. This is the declared atomic assumption, not an outstanding
whole-kernel proof or an assertion of IEEE equality. No new global axiom.
-/
import bench.tritonbench_g.vector_addition.VectorAddition
import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Float.Equivalence
import VeriTile.Triton.Float.ReportedAdmission

namespace VeriTile.Bench.Examples.TritonBenchVectorAdditionFP

open VeriTile Triton
open scoped VeriTile.Spec

abbrev admitted := FP.ReportedAdmission.fp32_add_commute

def rows : Nat := 4096
def columns : Nat := 4096
def nElements : Nat := rows * columns
def blockSize : Nat := 1024

/-- Same vector_addition body, with the experiment's fp32 region types. -/
def originalKernel : ComputeKernel :=
  let x_ptr : Region .fp32 := ⟨"x"⟩
  let y_ptr : Region .fp32 := ⟨"y"⟩
  let output_ptr : Region .fp32 := ⟨"output"⟩
  triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(blockSize)
  offsets = block_start + tl.arange(0, $(blockSize))
  mask = offsets < $(nElements)
  x = tl.load(x_ptr + offsets, mask=mask)
  y = tl.load(y_ptr + offsets, mask=mask)
  output = x + y
  tl.store(output_ptr + offsets, output, mask=mask)
}

/-- Erasing numerical precision recovers the existing TritonBench real kernel.
Its mathematical correctness proof can therefore be reused unchanged. -/
theorem real_projection :
    originalKernel.toAlgorithm? =
      (VeriTile.Bench.TritonBenchG.VectorAddition.add_kernel
        "x" "y" "output" nElements blockSize).toAlgorithm? := rfl

/-- The sole rewrite is output = y + x. Addresses/masks are unchanged. -/
def optimizedKernel : ComputeKernel :=
  let x_ptr : Region .fp32 := ⟨"x"⟩
  let y_ptr : Region .fp32 := ⟨"y"⟩
  let output_ptr : Region .fp32 := ⟨"output"⟩
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

/-- Parameterized local fp32 addition: register renaming is explicit. -/
def addFragment (out x y : RegName) : List ComputeStmt :=
  [.assign .real [blockSize] out
    (.compute (.alg .fp32 (.add .real (.consSame .nil)
      (.ref .real [blockSize] x) (.ref .real [blockSize] y))))]

def originalAdd := addFragment "output" "x" "y"
def optimizedAdd := addFragment "output" "y" "x"
def beforeAdd : List ComputeStmt := (body originalKernel).take 6
def afterAdd : List ComputeStmt := (body originalKernel).drop 7

theorem original_decomposition :
    body originalKernel = beforeAdd ++ originalAdd ++ afterAdd := rfl

theorem optimized_decomposition :
    body optimizedKernel = beforeAdd ++ optimizedAdd ++ afterAdd := rfl

/-- Changes to shape, launch or dtype invalidate this binding at compile time.
Distribution/backend/protocol and source hashes remain in admitted.configuration.
Flattening the contiguous matrix uses exactly rows*columns elements. -/
theorem report_matches :
    admitted.ruleID = "ADD-COMMUTE" ∧ admitted.shape = [rows, columns] ∧
    admitted.block = blockSize ∧ admitted.input = "fp32" ∧
    admitted.compute = "fp32" ∧ admitted.accumulator = "fp32" ∧
    admitted.output = "fp32" := by decide

/-- Only this frozen accepted row is bound, never a user-supplied PASS label. -/
def addCommute : Spec.RuleEntry ComputeStmt :=
  admitted.bind originalAdd optimizedAdd

/-- Trust in the published numerical result and its use-site correspondence.
The record is concrete: the caller cannot choose the rule, gates or contract.
This scoped premise is exactly the numerical assumption exposed by #print_spec. -/
structure Rules where
  add_comm : Spec.EvidenceValidated addCommute.rule addCommute.evidence

def Rules.assumptions (_ : Rules) : Spec.Assumptions ComputeStmt := [addCommute]

instance : Coe Rules (Spec.Assumptions (Spec.ProgramSyntax.Statement ComputeKernel)) :=
  ⟨Rules.assumptions⟩

@[spec_rule] theorem admitted_add_commute (R : Rules) :
    Spec.Derivation R.assumptions originalAdd optimizedAdd := by
  have _ := report_matches
  exact .atom addCommute (by simp [Rules.assumptions])
    (admitted.admit originalAdd optimizedAdd R.add_comm)

/-- Public specification: a kernel equivalence derived from one accepted atom. -/
specification vector_addition_equiv (R : Rules) :
    originalKernel ≡[R] optimizedKernel := by
  refine ⟨rfl, ?_⟩
  change Spec.Derivation R.assumptions (body originalKernel) (body optimizedKernel)
  rw [original_decomposition, optimized_decomposition]
  exact .frame beforeAdd afterAdd (admitted_add_commute R)

#print_spec vector_addition_equiv
#axiomsClean vector_addition_equiv
#auditModuleAxioms

end VeriTile.Bench.Examples.TritonBenchVectorAdditionFP
