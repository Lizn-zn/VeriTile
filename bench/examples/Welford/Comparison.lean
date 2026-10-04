import bench.examples.Welford.Kernels
/- Conditional comparison of the two original Welford kernels under explicit
fp32 reduction schedules. The scalar count-conversion premises remain explicit here;
WelfordFPEquiv supplies them from the admitted bounded count atoms. -/
import bench.examples.Welford.Execution
import VeriTile.Triton.Float.WelfordSchedule

namespace VeriTile.Bench.Examples.WelfordFPComparison
open VeriTile.Bench.Examples.Welford.Kernels
open VeriTile Triton FP.Structural FP.Guarded FP.ScalarArithmetic FP.ScalarReduction
open FP.WelfordReduction FP.WelfordInduction FP.WelfordSchedule
open _root_.VeriTile.Triton.FP.Equational (ReductionPlan Schedules)
open WelfordFPExecution

/-- Resolve the original default compute precision and expose its sum trees.
All casts, index conversions and non-sum operations retain their meanings. -/
def engine {α : Type} (M : Algebra α) (plans : Schedules) : Algebra α :=
  (FP.ScalarReduction.algebra M plans).withDefaultPrecision .fp32

def rowPlan (plans : Schedules) (N : Nat) : ReductionPlan N :=
  plans (some .fp32) [N] ⟨0, by simp⟩ Bool.false PUnit.unit

theorem sum_value {α : Type} (M : Algebra α) (plans : Schedules) (xs : Fin N → α) :
    sumValue (engine M plans) xs = value M xs (zero M) (rowPlan plans N).tree := by
  rfl

theorem recurrence_value {α : Type} (M : Algebra α) (plans : Schedules) (xs : Nat → α)
    (N k : Nat) (hk : k ≤ N) :
    recurrence (engine M plans) (rowPrefix xs N) k = state M xs k := by
  induction k with
  | zero => rfl
  | succ k ih =>
    rw [recurrence, dif_pos (by omega), ih (by omega)]
    rfl

theorem twopass_mean {α : Type} (M : Algebra α) (plans : Schedules) (xs : Fin N → α) :
    twopassMean (engine M plans) xs =
      div M (value M xs (zero M) (rowPlan plans N).tree) (M.fromNat (some .fp32) N) := by
  change div M (sumValue (engine M plans) xs) (M.fromNat (some .fp32) N) = _
  rw [sum_value]

theorem twopass_variance {α : Type} (M : Algebra α) (plans : Schedules) (xs : Fin N → α) :
    twopassVariance (engine M plans) xs =
      div M (value M (FP.Welford.deviationSquares M xs (twopassMean (engine M plans) xs))
        (zero M) (rowPlan plans N).tree) (M.fromNat (some .fp32) N) := by
  change div M (sumValue (engine M plans)
    (FP.Welford.deviationSquares M xs (twopassMean (engine M plans) xs)))
      (M.fromNat (some .fp32) N) = _
  rw [sum_value]

/-- Equality of the statistics before any output cast. A consumer such as
LayerNorm needs these values, not merely equality after bf16 rounding. -/
theorem original_statistics {α : Type} [Inhabited α] (R : Rules) (M : Algebra α)
    (D : Domain α) (hM : Models R.assumptions M D) (s : State α) (plans : Schedules)
    (xs : Nat → α) (empty : ReductionPlan 0) (N : Nat) (hN : 0 < N)
    (hc : CountConversion M N) (hi : IterationDomain M D xs empty.tree N)
    (hs : StatisticsDomain M D (rowPrefix xs N) (plan empty N) (rowPlan plans N)) :
    ((recurrence (engine M plans) (rowPrefix xs N) N).1,
      (engine M plans).binary none .real .div
        (recurrence (engine M plans) (rowPrefix xs N) N).2 ((engine M plans).fromNat none N)) =
      (twopassMean (engine M plans) (rowPrefix xs N),
        twopassVariance (engine M plans) (rowPrefix xs N)) := by
  have ht := state_batch_statistics R M D hM s xs empty (rowPlan plans N) hN hc hi hs
  have hn := converted_batch_count R M D hM s empty (rowPlan plans N) hc hi.initial.zero hs.counts
  have hm : twopassMean (engine M plans) (rowPrefix xs N) =
      FP.WelfordReduction.mean M (rowPrefix xs N) (rowPlan plans N).tree := by
    rw [twopass_mean, hn]
    rfl
  have hv : twopassVariance (engine M plans) (rowPrefix xs N) =
      div M (statistics M (rowPrefix xs N) (rowPlan plans N).tree).2
        (count M (rowPlan plans N).tree) := by
    rw [twopass_variance, hm, hn]
    rfl
  change ((recurrence (engine M plans) (rowPrefix xs N) N).1,
      div M (recurrence (engine M plans) (rowPrefix xs N) N).2 (M.fromNat (some .fp32) N)) = _
  rw [recurrence_value M plans xs N N le_rfl, ht, hn, hm, hv]
  rfl

/-- Both original bf16 output values agree after deriving the recurrence,
count binding and all three schedule changes. No output equality is a premise. -/
theorem original_values {α : Type} [Inhabited α] (R : Rules) (M : Algebra α)
    (D : Domain α) (hM : Models R.assumptions M D) (s : State α) (plans : Schedules)
    (xs : Nat → α) (empty : ReductionPlan 0) (N : Nat) (hN : 0 < N)
    (hc : CountConversion M N) (hi : IterationDomain M D xs empty.tree N)
    (hs : StatisticsDomain M D (rowPrefix xs N) (plan empty N) (rowPlan plans N)) :
    (meanValue (engine M plans) (rowPrefix xs N), varianceValue (engine M plans) (rowPrefix xs N)) =
      ((engine M plans).cast none .real .bf16 (twopassMean (engine M plans) (rowPrefix xs N)),
        (engine M plans).cast none .real .bf16 (twopassVariance (engine M plans) (rowPrefix xs N))) := by
  exact congrArg (fun values : α × α =>
    ((engine M plans).cast none .real .bf16 values.1,
      (engine M plans).cast none .real .bf16 values.2))
    (original_statistics R M D hM s plans xs empty N hN hc hi hs)

/-- Successful executions of both original kernels, both complete output
windows, and both memory frames. This remains conditional on count admission
and rewrite domains; it is not a new whole-kernel numerical assumption. -/
theorem original_runs {α : Type} [Inhabited α] (R : Rules) (M : Algebra α)
    (D : Domain α) (hM : Models R.assumptions M D) (s : State α) (plans : Schedules)
    (x mean variance : RegionName) (N stride : Nat) (hN : 0 < N) (hdistinct : mean ≠ variance)
    (xs : Nat → α) (empty : ReductionPlan 0)
    (hx : ∀ i : Fin N, (s.mem x (s.pids 0 * stride + i.val)).read .real = xs i.val)
    (hc : CountConversion M N) (hi : IterationDomain M D xs empty.tree N)
    (hs : StatisticsDomain M D (rowPrefix xs N) (plan empty N) (rowPlan plans N)) :
    IO₁ₓ₂PrivateScratch (onlineIO x mean variance N stride) ∧
    IO₁ₓ₂PrivateScratch (twopassIO x mean variance N stride) ∧
    ∃ a b, FP.Structural.exec (engine M plans) (onlineIO x mean variance N stride).kernel s = some a ∧
      FP.Structural.exec (engine M plans) (twopassIO x mean variance N stride).kernel s = some b ∧
      IO₁ₓ₂Outputs (onlineIO x mean variance N stride) (twopassIO x mean variance N stride) s a b ∧
      IO₁ₓ₂Frame (onlineIO x mean variance N stride) s a ∧
      IO₁ₓ₂Frame (twopassIO x mean variance N stride) s b := by
  obtain ⟨hpa, a, ha, hma, hva, hfa⟩ := online_io_run (engine M plans)
    x mean variance stride (rowPrefix xs N) s hdistinct hx
  obtain ⟨hpb, b, hb, hmb, hvb, hfb⟩ := twopass_io_run (engine M plans)
    x mean variance stride (rowPrefix xs N) s hdistinct hx
  have hv := original_values R M D hM s plans xs empty N hN hc hi hs
  refine ⟨hpa, hpb, a, b, ha, hb, ⟨?_, ?_⟩, hfa, hfb⟩
  · intro i
    have hi : i.val = 0 := by have := i.isLt; change i.val < 1 at this; omega
    simpa only [onlineIO, twopassIO, hi, Nat.add_zero] using
      hma.trans ((congrArg (Cell.mk .bf16) (congrArg Prod.fst hv)).trans hmb.symm)
  · intro i
    have hi : i.val = 0 := by have := i.isLt; change i.val < 1 at this; omega
    simpa only [onlineIO, twopassIO, hi, Nat.add_zero] using
      hva.trans ((congrArg (Cell.mk .bf16) (congrArg Prod.snd hv)).trans hvb.symm)

end VeriTile.Bench.Examples.WelfordFPComparison
