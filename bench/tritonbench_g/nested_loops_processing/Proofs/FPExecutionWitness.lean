import bench.tritonbench_g.nested_loops_processing.Proofs.FPExecutionFinish
import bench.tritonbench_g.nested_loops_processing.Proofs.NumericModel

namespace VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.NumericExecution
open VeriTile Triton TypedSource FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000

def initial : State (WithBot ℝ) where
  mem := fun r n ↦ if r = "src" then .mk .int (n : Int) else .mk .int (-1)
  regs := fun _ _ _ ↦ none
  pids := fun _ ↦ 0
  numPids := fun _ ↦ 1
  undef := fun d _ _ ↦ defaultValue d

def finalState : State (WithBot ℝ) :=
  TypedExecution.finalState (engine model) .int32 initial "src" "dst" 8 1

/-- Successful complete typed execution in a nonempty backend, preserving source
loop structure and all 24 stores. The carrier is real-valued, not a unit witness. -/
theorem original_run :
    FP.Structural.exec (engine model) (kernel .int32 Bool.false "src" "dst" 8 1) initial =
      some finalState := TypedExecution.source_run (engine model) .int32 "src" "dst" 8 1 initial

theorem candidate_run :
    FP.Structural.exec (engine model) (kernel .int32 Bool.true "src" "dst" 8 1) initial =
      some finalState := TypedExecution.candidate_run (engine model) .int32 "src" "dst" 8 1 initial

/-- Typed integer payloads exhibit the same overwrite and untouched sentinel as
independent real IO; no output values are premises of the execution theorem. -/
theorem final_cells :
    finalState.mem "dst" 8 = Cell.mk .int 2 ∧
    finalState.mem "dst" 47 = Cell.mk .int 27 ∧
    finalState.mem "dst" 55 = Cell.mk .int 35 ∧
    finalState.mem "dst" 56 = Cell.mk .int (-1) := by
  exact ⟨rfl, rfl, rfl, rfl⟩

/-- A second successful execution actually uses the real-valued floating backend
and fp64 cells, rather than merely instantiating a backend with integer payloads. -/
def floatingInitial : State (WithBot ℝ) where
  mem := fun r n ↦ if r = "src" then .mk .real (some (n : ℝ))
    else .mk .real (some (-1 : ℝ))
  regs := fun _ _ _ ↦ none
  pids := fun _ ↦ 0
  numPids := fun _ ↦ 1
  undef := fun d _ _ ↦ defaultValue d

def floatingFinal : State (WithBot ℝ) :=
  TypedExecution.finalState (engine model) .fp64 floatingInitial "src" "dst" 8 1

theorem floating_original_run :
    FP.Structural.exec (engine model) (kernel .fp64 Bool.false "src" "dst" 8 1) floatingInitial =
      some floatingFinal :=
  TypedExecution.source_run (engine model) .fp64 "src" "dst" 8 1 floatingInitial

theorem floating_candidate_run :
    FP.Structural.exec (engine model) (kernel .fp64 Bool.true "src" "dst" 8 1) floatingInitial =
      some floatingFinal :=
  TypedExecution.candidate_run (engine model) .fp64 "src" "dst" 8 1 floatingInitial

theorem floating_cells :
    floatingFinal.mem "dst" 8 = Cell.mk .real (some (2 : ℝ)) ∧
    floatingFinal.mem "dst" 47 = Cell.mk .real (some (27 : ℝ)) ∧
    floatingFinal.mem "dst" 55 = Cell.mk .real (some (35 : ℝ)) ∧
    floatingFinal.mem "dst" 56 = Cell.mk .real (some (-1 : ℝ)) := by
  exact ⟨rfl, rfl, rfl, rfl⟩

end
end VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.NumericExecution

#guard_msgs (drop info) in
#auditModuleAxioms
