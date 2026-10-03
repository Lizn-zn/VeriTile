/- Scheduled public IO objects for the original Welford pair. Their domain
is computed syntax containing only value checks. The run comparison remains
conditional on scalar count conversion; WelfordFPEquiv discharges those
premises from the accepted bounded atoms. -/
import bench.examples.support.WelfordComparison
import VeriTile.Triton.Float.WelfordConditions
import VeriTile.Triton.Float.ScheduledIO

namespace VeriTile.Bench.Examples.WelfordFPContract
open VeriTile Triton FP.Structural FP.Guarded FP.ScalarArithmetic
open FP.GuardExpression
open _root_.VeriTile.Triton.FP.Equational (ReductionPlan Schedules)
open WelfordFPExecution WelfordFPComparison

def rowExpressions (x : RegionName) (stride : Nat) (i : Nat) : Expr MemoryInput :=
  .input ⟨x, fun pid => pid * stride + i, .real⟩

def rowValues {α : Type} [Inhabited α] (s : State α) (x : RegionName)
    (stride i : Nat) : α := (s.mem x (s.pids 0 * stride + i)).read .real

@[simp] theorem eval_row {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (x : RegionName) (stride i : Nat) :
    (rowExpressions x stride i).eval M (MemoryInput.read s) = rowValues s x stride i := rfl

/-- The numerical proof domain contains initialization, each iteration, and
both normalization paths for all three statistics. Dimensions stay symbolic. -/
def requirements (x : RegionName) (N stride : Nat) (empty : ReductionPlan 0)
    (plans : Schedules) : Condition :=
  FP.WelfordConditions.complete (FP.GuardExpression.algebra MemoryInput)
    (rowExpressions x stride) empty (rowPlan plans N)

/-- Reification neither drops a guard nor introduces an equation premise. -/
theorem requirements_holds {α : Type} [Inhabited α] (M : Algebra α) (D : Domain α)
    (s : State α) (x : RegionName) (N stride : Nat) (empty : ReductionPlan 0) (plans : Schedules) :
    (requirements x N stride empty plans).Holds M D s ↔
      FP.WelfordInduction.IterationDomain M D (rowValues s x stride) empty.tree N ∧
      FP.WelfordSchedule.StatisticsDomain M D
        (FP.WelfordInduction.rowPrefix (rowValues s x stride) N)
        (FP.WelfordInduction.plan empty N) (rowPlan plans N) := by
  unfold requirements Condition.Holds
  rw [← Requirements.holds_map]
  simp only [FP.WelfordConditions.map_complete, eval_row, FP.WelfordConditions.complete_holds]

def online (x mean variance : RegionName) (N stride : Nat) (empty : ReductionPlan 0) :
    FP.Scheduled.IO₁ₓ₂ :=
  ⟨onlineIO x mean variance N stride, FP.Scheduled.fp32, requirements x N stride empty⟩

def twopass (x mean variance : RegionName) (N stride : Nat) (empty : ReductionPlan 0) :
    FP.Scheduled.IO₁ₓ₂ :=
  ⟨twopassIO x mean variance N stride, FP.Scheduled.fp32, requirements x N stride empty⟩

theorem same_signature (x mean variance : RegionName) (N stride : Nat) (empty : ReductionPlan 0) :
    Spec.ProgramSyntax.signature (online x mean variance N stride empty) =
      Spec.ProgramSyntax.signature (twopass x mean variance N stride empty) := rfl

/-- Both original runs, both typed output cells, and both memory frames under
the syntactic contract. This is a support theorem, not a completed `≡[R]`
specification: CountConversion is supplied by the admitted public theorem within its bound. -/
theorem original_runs_under_count {α : Type} [Inhabited α] (R : Rules) (M : Algebra α)
    (D : Domain α) (hM : Models R.assumptions M D) (s : State α) (plans : Schedules)
    (x mean variance : RegionName) (N stride : Nat) (hN : 0 < N) (hdistinct : mean ≠ variance)
    (empty : ReductionPlan 0) (hc : FP.WelfordInduction.CountConversion M N)
    (hd : (requirements x N stride empty plans).Holds M D s) :
    IO₁ₓ₂PrivateScratch (online x mean variance N stride empty).io ∧
    IO₁ₓ₂PrivateScratch (twopass x mean variance N stride empty).io ∧
    ∃ a b,
      FP.Structural.exec ((online x mean variance N stride empty).profile.algebra M plans)
        (online x mean variance N stride empty).io.kernel s = some a ∧
      FP.Structural.exec ((twopass x mean variance N stride empty).profile.algebra M plans)
        (twopass x mean variance N stride empty).io.kernel s = some b ∧
      IO₁ₓ₂Outputs (online x mean variance N stride empty).io
        (twopass x mean variance N stride empty).io s a b ∧
      IO₁ₓ₂Frame (online x mean variance N stride empty).io s a ∧
      IO₁ₓ₂Frame (twopass x mean variance N stride empty).io s b := by
  obtain ⟨hi, hs⟩ := (requirements_holds M D s x N stride empty plans).mp hd
  exact original_runs R M D hM s plans x mean variance N stride hN hdistinct
    (rowValues s x stride) empty (fun _ => rfl) hc hi hs

end VeriTile.Bench.Examples.WelfordFPContract
