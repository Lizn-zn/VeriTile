import bench.examples.OnlineSoftmax.Kernels
import VeriTile.Triton.DSL
import VeriTile.Triton.Float.Control
import bench.examples.SoftmaxStable.Proofs.FP
import VeriTile.Triton.Float.OnlineSoftmaxConditions
import VeriTile.Triton.Float.ExecutionProfile
import VeriTile.Triton.Float.ExponentialLaws
import VeriTile.Triton.Float.ObservedRow

/-! FP proof support for OnlineSoftmax. The public specification and atomic
assumption report are in ../FPEquiv.lean; the shared sources are in ../Kernels.lean.
Execution, intermediate-value domains and their composition are kept together. -/

/- Use libdevice.exp for exp-sub rewrites: the measured fp32 tl.exp relation
has B = 0.1608954387 ULP > 0.05 under the configured Normal(1,1) probe.
That intrinsic relation failed admission; the libdevice EXP-SUB instance passed. -/
/- Online execution. The first-pass normalizer leaves its result in m/l;
the complete two-pass kernel then stores the output row. Numerical comparisons
are derived below from the admitted scalar atoms. -/

namespace VeriTile.Bench.Examples.OnlineSoftmaxFPExecution
open VeriTile.Bench.Examples.OnlineSoftmax.Kernels
open VeriTile Triton FP.Structural

set_option maxHeartbeats 1600000


private def ref (n : RegName) : Op .real [] := .ref .real [] n

def body (xReg : RegionName) (N : Nat) : List ComputeStmt :=
  [.assign .real [] "xi" (.alg (.load .real (.region xReg
      (.add .nat .nil (.mul .nat .nil (.ref .nat [] "pid") (.constNat N))
        (.ref .nat [] "i"))) .none)),
   .assign .real [] "m_new" (.alg (.max2 .nil (ref "m") (ref "xi"))),
   .assign .real [] "l" (.alg (.add .real .nil
      (.mul .real .nil (.libdeviceExp (.sub .real .nil (ref "m") (ref "m_new"))) (ref "l"))
      (.libdeviceExp (.sub .real .nil (ref "xi") (ref "m_new"))))),
   .assign .real [] "m" (.alg (ref "m_new"))]

def initialCode : List ComputeStmt :=
  [.assign .nat [] "pid" (.alg (.programId 0)),
   .assign .real [] "m" (.alg .negInf),
   .assign .real [] "l" (.alg (.const 0))]

theorem kernel_body (x y : RegionName) (N : Nat) :
    (onlineNormalizerKernel x y N).surfaceBody =
      initialCode ++ [.forLoop "i" N (body x N)] := rfl

def update {α : Type} (M : Algebra α) (x : α) (acc : α × α) : α × α :=
  let m := M.binary none .real .max acc.1 x
  (m, M.binary none .real .add
    (M.binary none .real .mul
      (M.unary none .libdeviceExp (M.binary none .real .sub acc.1 m)) acc.2)
    (M.unary none .libdeviceExp (M.binary none .real .sub x m)))

def recurrence {α : Type} (M : Algebra α) (xs : Fin N → α) : Nat → α × α
  | 0 => (M.negInf, M.literal none .real 0)
  | i + 1 => if h : i < N then update M (xs ⟨i, h⟩) (recurrence M xs i)
      else recurrence M xs i

private def Invariant {α : Type} [Inhabited α] (M : Algebra α) (xs : Fin N → α)
    (origin : State α) (i : Nat) (s : State α) : Prop :=
  s.regs .real [] "m" = some (fun _ => (recurrence M xs i).1) ∧
  s.regs .real [] "l" = some (fun _ => (recurrence M xs i).2) ∧
  s.regs .nat [] "pid" = some (fun _ => origin.pids 0) ∧
  s.mem = origin.mem ∧ s.pids = origin.pids

private theorem iteration {α : Type} [Inhabited α] (M : Algebra α)
    (xReg : RegionName) (xs : Fin N → α) (origin : State α)
    (hx : ∀ i : Fin N, (origin.mem xReg (origin.pids 0 * N + i.val)).read .real = xs i)
    (i : Nat) (s : State α) (hi : i < N) (hs : Invariant M xs origin i s) :
    ∃ t, run M (body xReg N) (s.setReg "i" .nat [] (fun _ => i)) = some t ∧
      Invariant M xs origin (i + 1) t := by
  rcases hs with ⟨hm, hl, hpid, hmem, hpids⟩
  have hxi := hx ⟨i, hi⟩
  simp [body, run, step, evalExpr, evalOp_unfold, numeric, FP.Structural.bop,
    ref, State.setReg, hmem, hm, hl, hpid, hxi, Region.cast,
    Invariant, recurrence, hi, update, hpids]
  exact ⟨rfl, rfl⟩

/-- For any floating interpretation, the original source terminates with its
recurrence in m/l. Even -inf and exp remain opaque; there is no hidden identity
for initialization, no Real projection and no invented output store. -/
theorem online_run {α : Type} [Inhabited α] (M : Algebra α)
    (xReg yReg : RegionName) (xs : Fin N → α) (origin : State α)
    (hx : ∀ i : Fin N, (origin.mem xReg (origin.pids 0 * N + i.val)).read .real = xs i) :
    ∃ t, FP.Structural.exec M (onlineNormalizerKernel xReg yReg N) origin = some t ∧
      t.regs .real [] "m" = some (fun _ => (recurrence M xs N).1) ∧
      t.regs .real [] "l" = some (fun _ => (recurrence M xs N).2) ∧
      t.mem = origin.mem ∧ t.pids = origin.pids := by
  let s := ((origin.setReg "pid" .nat [] (fun _ => origin.pids 0)).setReg "m" .real []
    (fun _ => M.negInf)).setReg "l" .real [] (fun _ => M.literal none .real 0)
  have hs : Invariant M xs origin 0 s := by simp [Invariant, recurrence, s, State.setReg]
  obtain ⟨t, ht, hm, hl, _, hmem, hpids⟩ := forLoop_invariant M hs (iteration M xReg xs origin hx)
  refine ⟨t, ?_, hm, hl, hmem, hpids⟩
  have he : FP.Structural.exec M (onlineNormalizerKernel xReg yReg N) origin =
      run M (initialCode ++ [.forLoop "i" N (body xReg N)]) origin :=
    congrArg (fun code => run M code origin) (kernel_body xReg yReg N)
  rw [he, run_append]
  simpa [initialCode, run, step, evalExpr, evalOp_unfold, s] using ht

/-- The source's second pass, isolated only to compose its execution proof. -/
def normalizationBody (x y : RegionName) (N : Nat) : List ComputeStmt :=
  (onlineSoftmaxKernel x y N).surfaceBody.drop 4

theorem full_kernel_body (x y : RegionName) (N : Nat) :
    (onlineSoftmaxKernel x y N).surfaceBody =
      (onlineNormalizerKernel x y N).surfaceBody ++ normalizationBody x y N := rfl

def normalizedValue {α : Type} (M : Algebra α) (x m l : α) : α :=
  M.binary none .real .div (M.unary none .libdeviceExp
    (M.binary none .real .sub x m)) l

/-- Successful execution of the actual vector load and store. The load
precedes the store, including when x and y name the same region. -/
theorem normalization_run {α : Type} [Inhabited α] (M : Algebra α)
    (x y : RegionName) (N : Nat) (xs : Fin N → α) (s : State α) (m l : α)
    (hm : s.regs .real [] "m" = some (fun _ => m))
    (hl : s.regs .real [] "l" = some (fun _ => l))
    (hx : ∀ i : Fin N, (s.mem x (s.pids 0 * N + i.val)).read .real = xs i) :
    ∃ t, run M (normalizationBody x y N) s = some t ∧
      (∀ i : Fin N, t.mem y (s.pids 0 * N + i.val) =
        .mk .real (normalizedValue M (xs i) m l)) ∧
      (∀ r o, (r ≠ y ∨ ∀ i : Fin N, o ≠ s.pids 0 * N + i.val) → t.mem r o = s.mem r o) := by
  have hinj : Function.Injective (fun i : TileIndex [N] => s.pids 0 * N + i.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [normalizationBody, onlineSoftmaxKernel, ComputeKernel.surfaceBody, List.drop, run, step, evalExpr, evalOp_unfold,
    numeric, FP.Structural.bop, store, hm, hl, hx, Region.cast, normalizedValue]
  refine ⟨fun i => ?_, fun r o hmiss => ?_⟩
  · rw [State.scatter_readback _ _ _ _ (i, PUnit.unit) hinj]
  · apply (State.scatter_frame _ _ _ _ r o ?_ _).trans rfl
    rcases hmiss with hr | ho
    · exact Or.inl hr
    · exact Or.inr fun k _ => ho k.1

/-- Complete online kernel: final output cells and the memory frame. -/
theorem online_output_run {α : Type} [Inhabited α] (M : Algebra α)
    (x y : RegionName) (N : Nat) (xs : Fin N → α) (s : State α)
    (hx : ∀ i : Fin N, (s.mem x (s.pids 0 * N + i.val)).read .real = xs i) :
    ∃ t, FP.Structural.exec M (onlineSoftmaxKernel x y N) s = some t ∧
      (∀ i : Fin N, t.mem y (s.pids 0 * N + i.val) =
        .mk .real (normalizedValue M (xs i) (recurrence M xs N).1 (recurrence M xs N).2)) ∧
      (∀ r o, (r ≠ y ∨ ∀ i : Fin N, o ≠ s.pids 0 * N + i.val) → t.mem r o = s.mem r o) := by
  obtain ⟨a, ha, hm, hl, hmem, hpids⟩ := online_run M x y xs s hx
  obtain ⟨b, hb, hout, hframe⟩ := normalization_run M x y N xs a _ _ hm hl
    (by simpa [hmem, hpids] using hx)
  refine ⟨b, ?_, ?_, ?_⟩
  · change run M (onlineSoftmaxKernel x y N).surfaceBody s = some b
    rw [full_kernel_body, run_append]
    change (FP.Structural.exec M (onlineNormalizerKernel x y N) s).bind _ = _
    rw [ha]
    exact hb
  · simpa [hpids] using hout
  · simpa [hpids, hmem] using hframe

end VeriTile.Bench.Examples.OnlineSoftmaxFPExecution

/- Use libdevice.exp for exp-sub rewrites: the measured fp32 tl.exp relation
has B = 0.1608954387 ULP > 0.05 under the configured Normal(1,1) probe.
That intrinsic relation failed admission; the libdevice EXP-SUB instance passed. -/
/- Batch execution. The reference stores real-typed values, matching the
complete online source's output dtype. -/

namespace VeriTile.Bench.Examples.OnlineSoftmaxFPBatch
open VeriTile.Bench.Examples.OnlineSoftmax.Kernels
open VeriTile Triton FP.Structural
open SoftmaxStableFPExecution (maximum shifted rowSum)

set_option maxHeartbeats 1600000


def outputValue {α : Type} (M : Algebra α) (xs : Fin N → α) (i : Fin N) : α :=
  M.binary none .real .div (shifted M xs i) (rowSum M (shifted M xs))

theorem batch_run {α : Type} [Inhabited α] (M : Algebra α)
    (x y : RegionName) (N : Nat) (hN : 0 < N) (xs : Fin N → α) (s : State α)
    (hx : ∀ i : Fin N, (s.mem x (s.pids 0 * N + i.val)).read .real = xs i) :
    ∃ t, FP.Structural.exec M (stableSoftmaxKernel x y N) s = some t ∧
      (∀ i : Fin N, t.mem y (s.pids 0 * N + i.val) = .mk .real (outputValue M xs i)) ∧
      (∀ r o, (r ≠ y ∨ ∀ i : Fin N, o ≠ s.pids 0 * N + i.val) → t.mem r o = s.mem r o) := by
  have hinj : Function.Injective (fun i : TileIndex [N] => s.pids 0 * N + i.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [stableSoftmaxKernel, FP.Structural.exec, run, step, evalExpr, evalOp_unfold,
    numeric, FP.Structural.bop, store, TileShape.axisDim, TileShape.eraseAxis, hN,
    Region.cast, hx, outputValue, shifted, maximum, rowSum]
  refine ⟨fun i => ?_, fun r o hmiss => ?_⟩
  · rw [State.scatter_readback _ _ _ _ (i, PUnit.unit) hinj]
    rfl
  · apply (State.scatter_frame _ _ _ _ r o ?_ _).trans rfl
    rcases hmiss with hr | ho
    · exact Or.inl hr
    · exact Or.inr fun k _ => ho k.1

end VeriTile.Bench.Examples.OnlineSoftmaxFPBatch

/- Recurrence comparison. Connect the scalar-derived prefix invariant to the
first-pass normalizer's m/l registers; its memory remains unchanged. -/

namespace VeriTile.Bench.Examples.OnlineSoftmaxFPComparison
open VeriTile.Bench.Examples.OnlineSoftmax.Kernels
open VeriTile Triton FP.Structural FP.Guarded FP.ScalarArithmetic FP.GuardExpression
open OnlineSoftmaxFPExecution

def engine {α : Type} (M : Algebra α) : Algebra α := M.withDefaultPrecision .fp32

theorem recurrence_state {α : Type} (M : Algebra α) (xs : Nat → α) (N i : Nat) (hi : i ≤ N) :
    recurrence (engine M) (fun k : Fin N => xs k.val) i = FP.OnlineSoftmax.state M xs i := by
  induction i with
  | zero => rfl
  | succ i ih =>
    rw [recurrence, dif_pos (by omega), ih (by omega)]
    rfl

def rowExpressions (x : RegionName) (N i : Nat) : Expr MemoryInput :=
  .input ⟨x, fun pid => pid * N + i, .real⟩

def rowValues {α : Type} [Inhabited α] (s : State α) (x : RegionName) (N i : Nat) : α :=
  (s.mem x (s.pids 0 * N + i)).read .real

@[simp] theorem eval_row {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (x : RegionName) (N i : Nat) :
    (rowExpressions x N i).eval M (MemoryInput.read s) = rowValues s x N i := rfl

def requirements (x : RegionName) (N : Nat) : Condition :=
  FP.OnlineSoftmaxConditions.iterations (FP.GuardExpression.algebra MemoryInput) (rowExpressions x N) N

theorem requirements_holds {α : Type} [Inhabited α] (M : Algebra α) (D : Domain α)
    (s : State α) (x : RegionName) (N : Nat) :
    (requirements x N).Holds M D s ↔ FP.OnlineSoftmax.IterationDomain M D (rowValues s x N) N := by
  unfold requirements Condition.Holds
  rw [← Requirements.holds_map]
  simp only [FP.OnlineSoftmaxConditions.map_iterations, eval_row, FP.OnlineSoftmaxConditions.iterations_holds]

/-- The actual final registers recover the unshifted prefix sum, with the
original loop's full memory preservation and unchanged program IDs. No output
store, maximum equality or whole-recurrence numerical assumption is supplied. -/
theorem original_recovery_run {α : Type} [Inhabited α] (R : Rules) (M : Algebra α)
    (D : Domain α) (hM : Models R.assumptions M D) (s : State α)
    (x y : RegionName) (N : Nat) (hN : 0 < N) (hExp : FP.SoftmaxShift.LibdeviceExpSub M D)
    (hd : (requirements x N).Holds M D s) :
    ∃ t m l, FP.Structural.exec (engine M) (onlineNormalizerKernel x y N) s = some t ∧
      t.regs .real [] "m" = some (fun _ => m) ∧ t.regs .real [] "l" = some (fun _ => l) ∧
      mul M l (FP.SoftmaxShift.exp M m) = FP.OnlineSoftmax.prefixSum M (rowValues s x N) N ∧
      t.mem = s.mem ∧ t.pids = s.pids := by
  let xs := rowValues s x N
  obtain ⟨t, ht, hm, hl, hmem, hpids⟩ := online_run (engine M) x y
    (fun i : Fin N => xs i.val) s (fun _ => rfl)
  rw [recurrence_state M xs N N le_rfl] at hm hl
  refine ⟨t, (FP.OnlineSoftmax.state M xs N).1, (FP.OnlineSoftmax.state M xs N).2,
    ht, hm, hl, ?_, hmem, hpids⟩
  exact FP.OnlineSoftmax.state_recovery R M D hM s hExp xs N hN
    ((requirements_holds M D s x N).mp hd)

end VeriTile.Bench.Examples.OnlineSoftmaxFPComparison

/- Output contract. Derive normalization from the final m/l registers, then
connect both complete kernels to successful stored outputs and memory frames.
The intermediate register-observation theorem is retained as a helper. -/

namespace VeriTile.Bench.Examples.OnlineSoftmaxFPContract
open VeriTile.Bench.Examples.OnlineSoftmax.Kernels
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
        (OnlineSoftmax.Kernels.stableSoftmaxKernel x y N) s = some batch ∧
      FP.Structural.exec (OnlineSoftmaxFPComparison.engine M)
        (OnlineSoftmax.Kernels.onlineNormalizerKernel x y N) s = some online ∧
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
  kernel := OnlineSoftmax.Kernels.stableSoftmaxKernel x y N
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
  kernel := OnlineSoftmax.Kernels.onlineNormalizerKernel x y N
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

/-- Both public programs now write the same output row. -/
def batch (x y : RegionName) (N : Nat) : FP.Scheduled.IO₁ where
  io := {
    kernel := stableSoftmaxKernel x y N
    inp := x
    out := y
    Bin := N
    Bout := N
    read := fun pid => pid * N
    write := fun pid => pid * N }
  profile := FP.Scheduled.fp32
  domain := requirements x N

def online (x y : RegionName) (N : Nat) : FP.Scheduled.IO₁ :=
  { batch x y N with io := { (batch x y N).io with kernel := onlineSoftmaxKernel x y N, projection := rfl } }

/-- The two successful kernel executions agree on stored cells and preserve
all memory outside the output row. No readback expression supplies missing work. -/
theorem output_equivalent (R : FP.Exponential.Rules) (x y : RegionName)
    (N : Nat) (hN : 0 < N) :
    FP.Scheduled.Equivalent₁ R.assumptions (batch x y N) (online x y N) := by
  refine ⟨by simp [batch, IO₁PrivateScratch], by simp [online, batch, IO₁PrivateScratch], ?_⟩
  intro α _ M D hM plans s hd
  let xs := rowValues s x N
  obtain ⟨a, ha, hva, hfa⟩ := OnlineSoftmaxFPBatch.batch_run (SoftmaxStableFPContract.engine M plans)
    x y N hN (fun i : Fin N => xs i.val) s (fun _ => rfl)
  obtain ⟨b, hb, hvb, hfb⟩ := OnlineSoftmaxFPExecution.online_output_run
    (SoftmaxStableFPContract.engine M plans) x y N (fun i : Fin N => xs i.val) s (fun _ => rfl)
  rw [scheduled_recurrence M plans xs N N le_rfl] at hvb
  refine ⟨a, b, ha, hb, ?_, ?_, ?_⟩
  · intro i
    exact (hva i).trans ((congrArg (Cell.mk .real)
      (normalized_values R.arithmetic M D (FP.Exponential.arithmetic_models R M D hM) s
        xs N hN plans (FP.Exponential.exp_sub R M D hM s)
        ((requirements_holds M D s x N plans).mp hd) i)).trans (hvb i).symm)
  · intro r o ho _
    exact hfa r o ho
  · intro r o ho _
    exact hfb r o ho

end VeriTile.Bench.Examples.OnlineSoftmaxFPContract
