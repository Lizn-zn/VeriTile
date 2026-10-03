/- Connect the scalar-derived prefix invariant to the original stored m/l
registers. The original loop still performs no output store. -/
import bench.examples.support.OnlineSoftmaxExecution
import VeriTile.Triton.Float.OnlineSoftmaxConditions
import VeriTile.Triton.Float.ExecutionProfile

namespace VeriTile.Bench.Examples.OnlineSoftmaxFPComparison
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
    ∃ t m l, FP.Structural.exec (engine M) (onlineSoftmaxKernel x y N) s = some t ∧
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
