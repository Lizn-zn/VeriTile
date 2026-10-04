/- Conditional comparison of the original batch output with the normalized
values computed from the original online loop's final m/l registers. This
does not claim that the online source stores an output array. -/
import bench.examples.support.OnlineSoftmaxComparison
import bench.examples.support.OnlineSoftmaxBatch
import bench.examples.support.SoftmaxStableContract
import VeriTile.Triton.Float.ExponentialLaws
import VeriTile.Triton.Float.ObservedRow

namespace VeriTile.Bench.Examples.OnlineSoftmaxFPContract
open VeriTile Triton FP.Structural FP.Guarded FP.ScalarArithmetic FP.GuardExpression
open _root_.VeriTile.Triton.FP.Equational (Schedules ReductionPlan)
open OnlineSoftmaxFPComparison (rowValues rowExpressions)
open SoftmaxStableFPContract (rowPlan center)

def prefixPlan (N : Nat) : ReductionPlan N := FP.WelfordInduction.plan (.withSeed 0) N

@[simp] theorem prefixPlan_tree (N : Nat) :
    (prefixPlan N).tree = FP.WelfordInduction.tree .zero N :=
  FP.WelfordInduction.plan_tree (.withSeed 0) N

structure ComparisonDomain {α : Type} (M : Algebra α) (D : Domain α)
    (xs : Nat → α) (N : Nat) (plans : Schedules) : Prop where
  iterations : FP.OnlineSoftmax.IterationDomain M D xs N
  online : FP.SoftmaxShift.ShiftDomain M D (fun i : Fin N => xs i.val)
    (FP.OnlineSoftmax.state M xs N).1 (prefixPlan N).tree
  onlineTotal : D .finite (FP.OnlineSoftmax.state M xs N).2
  batch : FP.SoftmaxShift.ShiftDomain M D (fun i : Fin N => xs i.val)
    (center M (fun i : Fin N => xs i.val)) (rowPlan plans N).tree
  schedule : FP.ReductionSchedule.ScheduleDomain M D
    (FP.SoftmaxShift.exponentials M (fun i : Fin N => xs i.val)) (prefixPlan N) (rowPlan plans N)

theorem normalized_values {α : Type} [Inhabited α] (R : Rules) (M : Algebra α)
    (D : Domain α) (hM : Models R.assumptions M D) (s : State α)
    (xs : Nat → α) (N : Nat) (hN : 0 < N) (plans : Schedules)
    (hExp : FP.SoftmaxShift.LibdeviceExpSub M D) (hd : ComparisonDomain M D xs N plans) (i : Fin N) :
    OnlineSoftmaxFPBatch.outputValue (SoftmaxStableFPContract.engine M plans) (fun i : Fin N => xs i.val) i =
      div M (FP.SoftmaxShift.exp M (sub M (xs i.val) (FP.OnlineSoftmax.state M xs N).1))
        (FP.OnlineSoftmax.state M xs N).2 := by
  have honline := FP.OnlineSoftmax.normalized_prefix R M D hM s hExp xs N hN
    hd.iterations (by simpa only [prefixPlan_tree] using hd.online) hd.onlineTotal i
  have hbatch := FP.SoftmaxShift.normalized R M D hM s hExp (fun i : Fin N => xs i.val)
    (center M (fun i : Fin N => xs i.val)) (rowPlan plans N).tree hd.batch i
  have hs := FP.ReductionSchedule.plans_value R M D hM s _ (prefixPlan N) (rowPlan plans N) hd.schedule
  rw [prefixPlan_tree] at hs
  change FP.ScalarReduction.value M (fun i : Fin N => FP.SoftmaxShift.exp M (xs i.val)) (zero M)
      (FP.WelfordInduction.tree .zero N) =
    FP.ScalarReduction.value M (fun i : Fin N => FP.SoftmaxShift.exp M (xs i.val)) (zero M)
      (rowPlan plans N).tree at hs
  rw [FP.OnlineSoftmax.prefixSum_tree, hs] at honline
  exact hbatch.trans honline.symm

def requirements (x : RegionName) (N : Nat) (plans : Schedules) : Condition :=
  let M := FP.GuardExpression.algebra MemoryInput
  let xs := rowExpressions x N
  let row := fun i : Fin N => xs i.val
  .all [FP.OnlineSoftmaxConditions.iterations M xs N,
    FP.SoftmaxShift.checks M row (FP.OnlineSoftmax.state M xs N).1 (prefixPlan N).tree,
    .guard .finite (FP.OnlineSoftmax.state M xs N).2,
    FP.SoftmaxShift.checks M row (center M row) (rowPlan plans N).tree,
    FP.WelfordConditions.schedule M (FP.SoftmaxShift.exponentials M row) (prefixPlan N) (rowPlan plans N)]

theorem requirements_holds {α : Type} [Inhabited α] (M : Algebra α) (D : Domain α)
    (s : State α) (x : RegionName) (N : Nat) (plans : Schedules) :
    (requirements x N plans).Holds M D s ↔ ComparisonDomain M D (rowValues s x N) N plans := by
  have hstate := FP.OnlineSoftmaxConditions.eval_state M (MemoryInput.read s) (rowExpressions x N) N
  simp only [OnlineSoftmaxFPComparison.eval_row] at hstate
  have hm := congrArg Prod.fst hstate
  have hl := congrArg Prod.snd hstate
  dsimp only at hm hl
  unfold requirements Condition.Holds
  rw [← Requirements.holds_map]
  simp only [Requirements.map_all, List.map_cons, List.map_nil, Requirements.map,
    FP.OnlineSoftmaxConditions.map_iterations, FP.SoftmaxShift.map_checks,
    FP.WelfordConditions.map_schedule, SoftmaxStableFPContract.eval_center,
    OnlineSoftmaxFPComparison.eval_row, hm, hl, FP.SoftmaxShift.exponentials, FP.SoftmaxShift.eval_exp,
    Requirements.holds_all, List.mem_cons, List.not_mem_nil, forall_eq_or_imp,
    false_implies, implies_true, and_true, Requirements.holds_guard,
    FP.OnlineSoftmaxConditions.iterations_holds, FP.SoftmaxShift.checks_holds,
    FP.WelfordConditions.schedule_holds]
  constructor
  · rintro ⟨h0, h1, h2, h3, h4⟩
    exact ⟨h0, h1, h2, h3, h4⟩
  · intro h
    exact ⟨h.iterations, h.online, h.onlineTotal, h.batch, h.schedule⟩

/-- Preserve both original executions. The batch writes its original real
cells and frames other memory; the online loop only computes registers and
preserves all memory. The comparison observes their normalized numerical values. -/
theorem original_normalization_runs {α : Type} [Inhabited α] (R : FP.Exponential.Rules) (M : Algebra α)
    (D : Domain α) (hM : Models R.assumptions M D) (s : State α) (plans : Schedules)
    (x y : RegionName) (N : Nat) (hN : 0 < N)
    (hd : (requirements x N plans).Holds M D s) :
    ∃ batch online m l,
      FP.Structural.exec (SoftmaxStableFPContract.engine M plans)
        (OnlineSoftmaxFPBatch.stableSoftmaxKernel x y N) s = some batch ∧
      FP.Structural.exec (OnlineSoftmaxFPComparison.engine M)
        (OnlineSoftmaxFPExecution.onlineSoftmaxKernel x y N) s = some online ∧
      online.regs .real [] "m" = some (fun _ => m) ∧ online.regs .real [] "l" = some (fun _ => l) ∧
      (∀ i : Fin N, batch.mem y (s.pids 0 * N + i.val) = .mk .real
        (div M (FP.SoftmaxShift.exp M (sub M (rowValues s x N i.val) m)) l)) ∧
      (∀ r o, (r ≠ y ∨ ∀ i : Fin N, o ≠ s.pids 0 * N + i.val) → batch.mem r o = s.mem r o) ∧
      online.mem = s.mem ∧ online.pids = s.pids := by
  let xs := rowValues s x N
  obtain ⟨a, ha, hva, hfa⟩ := OnlineSoftmaxFPBatch.batch_run (SoftmaxStableFPContract.engine M plans)
    x y N hN (fun i : Fin N => xs i.val) s (fun _ => rfl)
  obtain ⟨b, hb, hm, hl, hmem, hpids⟩ := OnlineSoftmaxFPExecution.online_run
    (OnlineSoftmaxFPComparison.engine M) x y (fun i : Fin N => xs i.val) s (fun _ => rfl)
  rw [OnlineSoftmaxFPComparison.recurrence_state M xs N N le_rfl] at hm hl
  refine ⟨a, b, (FP.OnlineSoftmax.state M xs N).1, (FP.OnlineSoftmax.state M xs N).2,
    ha, hb, hm, hl, ?_, hfa, hmem, hpids⟩
  intro i
  exact (hva i).trans (congrArg (Cell.mk .real)
    (normalized_values R.arithmetic M D (FP.Exponential.arithmetic_models R M D hM) s xs N hN plans
      (FP.Exponential.exp_sub R M D hM s) ((requirements_holds M D s x N plans).mp hd) i))

/-- The online source has no reduction operation. Expanding the batch sum
schedule therefore leaves its scalar recurrence unchanged. -/
theorem scheduled_recurrence {α : Type} (M : Algebra α) (plans : Schedules)
    (xs : Nat → α) (N i : Nat) (hi : i ≤ N) :
    OnlineSoftmaxFPExecution.recurrence (SoftmaxStableFPContract.engine M plans)
      (fun k : Fin N => xs k.val) i = FP.OnlineSoftmax.state M xs i := by
  induction i with
  | zero => rfl
  | succ i ih =>
    rw [OnlineSoftmaxFPExecution.recurrence, dif_pos (by omega), ih (by omega)]
    rfl

def loadRow (r : RegionName) (N pid i : Nat) : Op .real [] :=
  .load .real (.region r (.constNat (pid * N + i))) .none

/-- Observe the stored batch result, at its original real dtype. -/
def batchOutput (x y : RegionName) (N : Nat) : FP.ObservedRow.Program where
  input := x
  output := y
  size := N
  offset := fun pid => pid * N
  kernel := OnlineSoftmaxFPBatch.stableSoftmaxKernel x y N
  observe := fun pid i => loadRow y N pid i.val
  writesOutput := Bool.true
  profile := FP.Scheduled.fp32
  domain := requirements x N

/-- Read the original online source's final m/l, then evaluate the normalized
value. This is the same observation scope as OnlineSoftmaxCorrect; it is a
read-only expression, not an additional store in the online kernel. -/
def normalizedOnline (x y : RegionName) (N : Nat) : FP.ObservedRow.Program where
  input := x
  output := y
  size := N
  offset := fun pid => pid * N
  kernel := OnlineSoftmaxFPExecution.onlineSoftmaxKernel x y N
  observe := fun pid i => .div .real .nil
    (.libdeviceExp (.sub .real .nil (loadRow x N pid i.val) (.ref .real [] "m")))
    (.ref .real [] "l")
  writesOutput := Bool.false
  profile := FP.Scheduled.fp32
  domain := requirements x N

set_option maxHeartbeats 1600000 in
theorem observed_equivalent (R : FP.Exponential.Rules) (x y : RegionName)
    (N : Nat) (hN : 0 < N) :
    FP.ObservedRow.Equivalent R.assumptions (batchOutput x y N) (normalizedOnline x y N) := by
  intro α _ M D hM plans s hd
  let xs := rowValues s x N
  obtain ⟨a, ha, hva, hfa⟩ := OnlineSoftmaxFPBatch.batch_run (SoftmaxStableFPContract.engine M plans)
    x y N hN (fun i : Fin N => xs i.val) s (fun _ => rfl)
  obtain ⟨b, hb, hm, hl, hmem, _⟩ := OnlineSoftmaxFPExecution.online_run
    (SoftmaxStableFPContract.engine M plans) x y (fun i : Fin N => xs i.val) s (fun _ => rfl)
  rw [scheduled_recurrence M plans xs N N le_rfl] at hm hl
  let values := fun i => div M
    (FP.SoftmaxShift.exp M (sub M (xs i) (FP.OnlineSoftmax.state M xs N).1))
    (FP.OnlineSoftmax.state M xs N).2
  refine ⟨a, b, values, ha, hb, ?_, ?_, ?_, ?_⟩
  · intro i
    have hv := (hva i).trans (congrArg (Cell.mk .real)
      (normalized_values R.arithmetic M D (FP.Exponential.arithmetic_models R M D hM) s
        xs N hN plans (FP.Exponential.exp_sub R M D hM s)
        ((requirements_holds M D s x N plans).mp hd) i))
    simp [batchOutput, loadRow, evalOp_unfold, Region.cast, hv, values]
  · intro i
    simp [normalizedOnline, loadRow, evalOp_unfold, numeric,
      hm, hl, hmem, Region.cast, values, xs, rowValues,
      FP.SoftmaxShift.exp, sub, div,
      FP.Scheduled.fp32, FP.Scheduled.Profile.algebra, Algebra.withDefaultPrecision,
      FP.ScalarReduction.algebra]
    rfl
  · intro r o ho
    apply hfa r o
    simpa [batchOutput] using ho
  · intro r o _
    exact congrFun (congrFun hmem r) o

end VeriTile.Bench.Examples.OnlineSoftmaxFPContract
