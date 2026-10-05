import bench.examples.StableLogSumExp.Kernels
/- Logsumexp source contracts and scalar-derived sum recovery.
The conditional optimized is proved from admitted tl.log rewrites. The older
unconditional shift retains explicit, unadmitted primitive log premises below;
those lemmas do not certify the unconditional transformation. -/
import bench.examples.StableLogSumExp.Execution
import bench.examples.SoftmaxStable.Contract
import VeriTile.Triton.Float.LogSumExpShift
import VeriTile.Triton.Float.LogSumExpRules
import VeriTile.Triton.Float.ConditionalIO

namespace VeriTile.Bench.Examples.StableLogSumExpFPContract
open VeriTile.Bench.Examples.StableLogSumExp.Kernels
open VeriTile Triton FP.Structural FP.Guarded FP.ScalarArithmetic
open FP.GuardExpression
open _root_.VeriTile.Triton.FP.Equational (Schedules)
open StableLogSumExpFPExecution
open SoftmaxStableFPContract (engine rowPlan center rowValues rowExpressions)

def requirements (x : RegionName) (B : Nat) (plans : Schedules) : Condition :=
  let M := FP.GuardExpression.algebra MemoryInput
  FP.LogSumExpShift.checks M (rowExpressions x B) (center M (rowExpressions x B)) (rowPlan plans B).tree

theorem requirements_holds {α : Type} [Inhabited α] (M : Algebra α) (D : Domain α)
    (s : State α) (x : RegionName) (B : Nat) (plans : Schedules) :
    (requirements x B plans).Holds M D s ↔
      FP.LogSumExpShift.ShiftDomain M D (rowValues s x B) (center M (rowValues s x B))
        (rowPlan plans B).tree := by
  unfold requirements Condition.Holds
  rw [← Requirements.holds_map]
  simp only [FP.LogSumExpShift.map_checks, SoftmaxStableFPContract.eval_center,
    SoftmaxStableFPContract.eval_row, FP.LogSumExpShift.checks_holds]

def stable (x y : RegionName) (B : Nat) : FP.Scheduled.IO₁ :=
  ⟨stableIO x y B, FP.Scheduled.fp32, requirements x B⟩

def direct (x y : RegionName) (B : Nat) : FP.Scheduled.IO₁ :=
  ⟨directIO x y B, FP.Scheduled.fp32, requirements x B⟩

theorem same_signature (x y : RegionName) (B : Nat) :
    Spec.ProgramSyntax.signature (stable x y B) = Spec.ProgramSyntax.signature (direct x y B) := rfl

theorem original_values {α : Type} [Inhabited α] (R : Rules) (M : Algebra α)
    (D : Domain α) (hM : Models R.assumptions M D) (s : State α) (plans : Schedules)
    (hExp : FP.SoftmaxShift.LibdeviceExpSub M D)
    (hMul : FP.LogSumExpShift.IntrinsicLogMul M D) (hLogExp : FP.LogSumExpShift.LogLibdeviceExp M D)
    (xs : Fin B → α)
    (hd : FP.LogSumExpShift.ShiftDomain M D xs (center M xs) (rowPlan plans B).tree) :
    stableValue (engine M plans) xs = directValue (engine M plans) xs := by
  unfold stableValue directValue
  rw [SoftmaxStableFPContract.sum_value, SoftmaxStableFPContract.sum_value]
  exact congrArg (M.cast (some .fp32) .real .bf16)
    (FP.LogSumExpShift.shifted_result R M D hM s hExp hMul hLogExp xs
      (center M xs) (rowPlan plans B).tree hd)

/-- Both original kernels succeed and agree at the original scalar bf16
output, preserving all other cells. Only primitive log obligations remain;
this conditional theorem is not a completed admitted FP specification. -/
theorem original_runs_under_log {α : Type} [Inhabited α] (R : FP.Exponential.Rules) (M : Algebra α)
    (D : Domain α) (hM : Models R.assumptions M D) (s : State α) (plans : Schedules)
    (x y : RegionName) (B : Nat) (hB : 0 < B)
    (hMul : FP.LogSumExpShift.IntrinsicLogMul M D) (hLogExp : FP.LogSumExpShift.LogLibdeviceExp M D)
    (hd : (requirements x B plans).Holds M D s) :
    IO₁PrivateScratch (stable x y B).io ∧ IO₁PrivateScratch (direct x y B).io ∧
    ∃ a b,
      FP.Structural.exec ((stable x y B).profile.algebra M plans) (stable x y B).io.kernel s = some a ∧
      FP.Structural.exec ((direct x y B).profile.algebra M plans) (direct x y B).io.kernel s = some b ∧
      a.mem y (s.pids 0) = b.mem y (s.pids 0) ∧
      IO₁Frame (stable x y B).io s a ∧ IO₁Frame (direct x y B).io s b := by
  let xs := rowValues s x B
  have hd' := (requirements_holds M D s x B plans).mp hd
  obtain ⟨a, ha, hva, hfa⟩ := stable_run (engine M plans) x y B hB xs s (fun _ => rfl)
  obtain ⟨b, hb, hvb, hfb⟩ := direct_run (engine M plans) x y B xs s (fun _ => rfl)
  refine ⟨by simp [stable, IO₁PrivateScratch, stableIO, directIO],
    by simp [direct, IO₁PrivateScratch, directIO], a, b, ha, hb, ?_, ?_, ?_⟩
  · exact hva.trans ((congrArg (Cell.mk .bf16)
      (original_values R.arithmetic M D (FP.Exponential.arithmetic_models R M D hM) s plans
        (FP.Exponential.exp_sub R M D hM s) hMul hLogExp xs hd')).trans hvb.symm)
  · intro r o ho _
    apply hfa r o
    simpa only [stable, stableIO, directIO, Fin.forall_fin_one, Fin.val_zero, Nat.add_zero] using ho
  · intro r o ho _
    apply hfb r o
    simpa only [direct, directIO, Fin.forall_fin_one, Fin.val_zero, Nat.add_zero] using ho


noncomputable section Candidate

/-- The optimized and reference share numeric-domain checks and require only
availability of the fp32 comparisons used by the optimized. Either result of
each comparison is allowed. No kernel-output equation is a precondition. -/
def optimized (x y : RegionName) (B : Nat) : FP.Scheduled.ConditionalIO₁ where
  io := optimizedIO x y B
  profile := FP.Scheduled.fp32
  domain := requirements x B
  comparisons := [.lt .fp32 .real, .le .fp32 .real]

def original (x y : RegionName) (B : Nat) : FP.Scheduled.ConditionalIO₁ :=
  { optimized x y B with io := directIO x y B }

/-- Recover the direct sum from exp-sub and arithmetic atoms, then compose
conditional product splitting and log-exp cancellation, both using tl.log.
The bf16 conversion is preserved by congruence, never cancelled. -/
theorem candidate_values {α : Type} [Inhabited α] (R : FP.LogSumExp.Rules)
    (M : Algebra α) (D : Domain α) (hM : Models R.assumptions M D)
    (s : State α) (plans : Schedules) (lt le : α → α → Bool)
    (hlt : M.compareLt (some .fp32) .real = some lt)
    (hle : M.compareLe (some .fp32) .real = some le)
    (xs : Fin B → α)
    (hd : FP.LogSumExpShift.ShiftDomain M D xs (center M xs) (rowPlan plans B).tree) :
    candidateValue M plans lt le xs = directValue (engine M plans) xs := by
  have he := FP.LogSumExp.exponential_models R M D hM
  have hl := FP.LogSumExp.logarithm_models R M D hM
  dsimp only [candidateValue, directValue]
  change M.cast (some .fp32) .real .bf16
    (FP.LogSumExpCandidate.finish M lt le
      (SoftmaxStableFPExecution.rowSum (engine M plans)
        (SoftmaxStableFPExecution.shifted (engine M plans) xs)) (center M xs)) =
    M.cast (some .fp32) .real .bf16
      (M.unary (some .fp32) .log (SoftmaxStableFPExecution.rowSum (engine M plans)
        (SoftmaxStableFPExecution.exponentials (engine M plans) xs)))
  rw [SoftmaxStableFPContract.sum_value, SoftmaxStableFPContract.sum_value]
  apply congrArg (M.cast (some .fp32) .real .bf16)
  change FP.LogSumExpCandidate.finish M lt le
    (FP.ScalarReduction.value M (FP.SoftmaxShift.shifted M xs (center M xs))
      (zero M) (rowPlan plans B).tree) (center M xs) = _
  rw [FP.LogSumExpCandidate.finish_eq R.logarithm M D hl s lt le hlt hle
    _ _ hd.shiftedSum hd.shiftedSumPositive hd.center hd.centerExp hd.centerExpPositive]
  exact congrArg (M.unary (some .fp32) .log)
    (FP.LogSumExpShift.recover_sum R.exponential.arithmetic M D
      (FP.Exponential.arithmetic_models R.exponential M D he) s
      (FP.Exponential.exp_sub R.exponential M D he s) xs
      (center M xs) (rowPlan plans B).tree hd)

/-- Successful executions, the scalar bf16 observation and the memory frame
for the actual two sources. The original unconditional shift remains separate. -/
theorem candidate_runs {α : Type} [Inhabited α] (R : FP.LogSumExp.Rules)
    (M : Algebra α) (D : Domain α) (hM : Models R.assumptions M D)
    (s : State α) (plans : Schedules) (lt le : α → α → Bool)
    (hlt : M.compareLt (some .fp32) .real = some lt)
    (hle : M.compareLe (some .fp32) .real = some le)
    (x y : RegionName) (B : Nat) (hB : 0 < B)
    (hd : (requirements x B plans).Holds M D s) :
    ∃ a b,
      FP.Structural.exec (engine M plans) (optimizedLSEKernel x y B) s = some a ∧
      FP.Structural.exec (engine M plans) (directLSEKernel x y B) s = some b ∧
      a.mem y (s.pids 0) = b.mem y (s.pids 0) ∧
      IO₁Frame (optimizedIO x y B) s a ∧ IO₁Frame (directIO x y B) s b := by
  let xs := rowValues s x B
  have hd' := (requirements_holds M D s x B plans).mp hd
  obtain ⟨a, ha, hva, hfa⟩ := candidate_run M plans lt le hlt hle x y B hB xs s (fun _ => rfl)
  obtain ⟨b, hb, hvb, hfb⟩ := direct_run (engine M plans) x y B xs s (fun _ => rfl)
  refine ⟨a, b, ha, hb, ?_, ?_, ?_⟩
  · exact hva.trans ((congrArg (Cell.mk .bf16)
      (candidate_values R M D hM s plans lt le hlt hle xs hd')).trans hvb.symm)
  · intro r o ho _
    apply hfa r o
    simpa only [optimizedIO, directIO, Fin.forall_fin_one, Fin.val_zero, Nat.add_zero] using ho
  · intro r o ho _
    apply hfb r o
    simpa only [directIO, Fin.forall_fin_one, Fin.val_zero, Nat.add_zero] using ho

end Candidate
end VeriTile.Bench.Examples.StableLogSumExpFPContract
