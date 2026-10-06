import bench.examples.StableLogSumExp.Kernels
import bench.examples.SoftmaxStable.Proofs.FP
import VeriTile.Triton.Float.ScheduledIO
import VeriTile.Triton.Float.LogSumExpCandidate
import VeriTile.Triton.Float.LogSumExpShift
import VeriTile.Triton.Float.LogSumExpRules
import VeriTile.Triton.Float.ConditionalIO

/-! FP proof support for StableLogSumExp. The public specification and atomic
assumption report are in ../FPEquiv.lean; the shared sources are in ../Kernels.lean.
Execution, intermediate-value domains and their composition are kept together. -/

/- Use libdevice.exp for exp-sub rewrites: the measured fp32 tl.exp relation
has B = 0.1608954387 ULP > 0.05 under the configured Normal(1,1) probe.
That intrinsic relation failed admission; the libdevice EXP-SUB instance passed. -/
/- Original direct and stable logsumexp execution under opaque FP operations.
The source's scalar output address and bf16 conversion remain explicit. -/

namespace VeriTile.Bench.Examples.StableLogSumExpFPExecution
open VeriTile.Bench.Examples.StableLogSumExp.Kernels
open VeriTile Triton FP.Structural
open SoftmaxStableFPExecution (maximum exponentials shifted rowSum)

set_option maxHeartbeats 1600000


def directValue {α : Type} (M : Algebra α) (xs : Fin B → α) : α :=
  M.cast none .real .bf16 (M.unary none .log (rowSum M (exponentials M xs)))

def stableValue {α : Type} (M : Algebra α) (xs : Fin B → α) : α :=
  M.cast none .real .bf16 (M.binary none .real .add (maximum M xs)
    (M.unary none .log (rowSum M (shifted M xs))))

theorem direct_run {α : Type} [Inhabited α] (M : Algebra α)
    (x y : RegionName) (B : Nat) (xs : Fin B → α) (s : State α)
    (hx : ∀ i : Fin B, (s.mem x (s.pids 0 * B + i.val)).read .real = xs i) :
    ∃ t, FP.Structural.exec M (directLSEKernel x y B) s = some t ∧
      t.mem y (s.pids 0) = .mk .bf16 (directValue M xs) ∧
      (∀ r o, (r ≠ y ∨ o ≠ s.pids 0) → t.mem r o = s.mem r o) := by
  simp [directLSEKernel, FP.Structural.exec, run, step, evalExpr, evalOp_unfold,
    numeric, FP.Structural.bop, store, State.write, TileShape.allIndices,
    TileShape.eraseAxis, Region.cast, ofFloat, toFloat, hx, directValue, exponentials, rowSum]
  refine ⟨rfl, ?_⟩
  intro r o hmiss hr ho
  exact (hmiss.elim (fun h => h hr) (fun h => h ho)).elim

theorem stable_run {α : Type} [Inhabited α] (M : Algebra α)
    (x y : RegionName) (B : Nat) (hB : 0 < B) (xs : Fin B → α) (s : State α)
    (hx : ∀ i : Fin B, (s.mem x (s.pids 0 * B + i.val)).read .real = xs i) :
    ∃ t, FP.Structural.exec M (stableLSEKernel x y B) s = some t ∧
      t.mem y (s.pids 0) = .mk .bf16 (stableValue M xs) ∧
      (∀ r o, (r ≠ y ∨ o ≠ s.pids 0) → t.mem r o = s.mem r o) := by
  simp [stableLSEKernel, FP.Structural.exec, run, step, evalExpr, evalOp_unfold,
    numeric, FP.Structural.bop, store, State.write, TileShape.allIndices,
    TileShape.axisDim, TileShape.eraseAxis, hB, Region.cast, ofFloat, toFloat,
    hx, stableValue, shifted, maximum, rowSum]
  refine ⟨rfl, ?_⟩
  intro r o hmiss hr ho
  exact (hmiss.elim (fun h => h hr) (fun h => h ho)).elim

def directIO (x y : RegionName) (B : Nat) : KernelIO₁ where
  kernel := directLSEKernel x y B
  projection := by rfl
  inp := x
  out := y
  Bin := B
  Bout := 1
  read := fun pid => pid * B
  write := fun pid => pid

def stableIO (x y : RegionName) (B : Nat) : KernelIO₁ :=
  { directIO x y B with kernel := stableLSEKernel x y B, projection := by rfl }

theorem same_signature (x y : RegionName) (B : Nat) :
    io₁Signature (directIO x y B) = io₁Signature (stableIO x y B) := rfl

/- Execution of the conditional candidate under the same fp32 reduction
profile. Comparison support is explicit; neither branch outcome is assumed. -/
noncomputable section Candidate
open _root_.VeriTile.Triton.FP.Equational (Schedules)
set_option maxHeartbeats 2400000
set_option maxRecDepth 8000

def candidateValue {α : Type} (M : Algebra α) (plans : Schedules)
    (lt le : α → α → Bool) (xs : Fin B → α) : α :=
  let engine := FP.Scheduled.fp32.algebra M plans
  M.cast (some .fp32) .real .bf16 (FP.LogSumExpCandidate.finish M lt le
    (rowSum engine (shifted engine xs)) (maximum engine xs))

private theorem half_eq : (0.5 : ℝ) = 1 / 2 := by norm_num
private theorem two_eq : (2.0 : ℝ) = 2 := by norm_num
private theorem upper_eq : (80.0 : ℝ) = 80 := by norm_num
private theorem zero_eq : (0.0 : ℝ) = 0 := by norm_num
private theorem one_eq : (1.0 : ℝ) = 1 := by norm_num

theorem candidate_run {α : Type} [Inhabited α] (M : Algebra α)
    (plans : Schedules) (lt le : α → α → Bool)
    (hlt : M.compareLt (some .fp32) .real = some lt)
    (hle : M.compareLe (some .fp32) .real = some le)
    (x y : RegionName) (B : Nat) (hB : 0 < B) (xs : Fin B → α) (s : State α)
    (hx : ∀ i : Fin B, (s.mem x (s.pids 0 * B + i.val)).read .real = xs i) :
    ∃ t, FP.Structural.exec (FP.Scheduled.fp32.algebra M plans) (optimizedLSEKernel x y B) s = some t ∧
      t.mem y (s.pids 0) = .mk .bf16 (candidateValue M plans lt le xs) ∧
      (∀ r o, (r ≠ y ∨ o ≠ s.pids 0) → t.mem r o = s.mem r o) := by
  let E := FP.Scheduled.fp32.algebra M plans
  have hlt' : E.compareLt none .real = some lt := hlt
  have hle' : E.compareLe none .real = some le := hle
  change ∃ t, FP.Structural.exec E (optimizedLSEKernel x y B) s = some t ∧ _
  simp [optimizedLSEKernel, FP.Structural.exec, run, step, evalExpr, evalOp_unfold,
    numeric, numericLt, numericLe, FP.Structural.bop, store, State.write, TileShape.allIndices,
    TileShape.axisDim, TileShape.eraseAxis, hB, Region.cast, ofFloat, toFloat,
    hx, hlt', hle']
  refine ⟨?_, ?_⟩
  · simp [candidateValue, FP.LogSumExpCandidate.finish, FP.LogExp.value, FP.LogExp.Backend.logOp,
      E, FP.Scheduled.Profile.algebra, FP.Scheduled.fp32, FP.ScalarReduction.algebra,
      Algebra.withDefaultPrecision, resolvePrecision, shifted, maximum, rowSum,
      FP.Structural.bop, Function.comp_def,
      half_eq, two_eq, upper_eq, zero_eq, one_eq]
  · intro r o hmiss hr ho
    exact (hmiss.elim (fun h => h hr) (fun h => h ho)).elim

def optimizedIO (x y : RegionName) (B : Nat) : KernelIO₁ :=
  { directIO x y B with kernel := optimizedLSEKernel x y B, projection := by rfl }

end Candidate

end VeriTile.Bench.Examples.StableLogSumExpFPExecution

/- Logsumexp source contracts and scalar-derived sum recovery.
The conditional optimized is proved from admitted tl.log rewrites. The older
unconditional shift retains explicit, unadmitted primitive log premises below;
those lemmas do not certify the unconditional transformation. -/

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
