/- Transfer Welford's prefix statistics to an arbitrary valid batch reduction
schedule. All reordering is derived through the scalar rewrite paths. -/
import VeriTile.Triton.Float.WelfordInduction
import VeriTile.Triton.Float.ReductionSchedule

namespace VeriTile.Triton.FP.WelfordSchedule
open Structural Guarded ScalarArithmetic ScalarReduction WelfordReduction
open Equational (ReductionPlan)
open ReductionSchedule (ScheduleDomain plans_value)

/-- The three numerical rows whose schedules change: input values, count
ones, and squared deviations about the already-defined prefix mean. -/
structure StatisticsDomain {α : Type} (M : Algebra α) (D : Domain α)
    (xs : Fin n → α) (a b : ReductionPlan n) : Prop where
  inputs : ScheduleDomain M D xs a b
  counts : ScheduleDomain M D (fun _ => one M) a b
  squares : ScheduleDomain M D (Welford.deviationSquares M xs (mean M xs a.tree)) a b

section Transfer
variable {α : Type} [Inhabited α] (R : Rules) (M : Algebra α) (D : Domain α)
  (hM : Models R.assumptions M D) (s : State α)
include R hM s

theorem statistics_schedule (xs : Fin n → α) (a b : ReductionPlan n)
    (hd : StatisticsDomain M D xs a b) :
    WelfordInduction.statistics M xs a.tree = WelfordInduction.statistics M xs b.tree := by
  have hsum := plans_value R M D hM s xs a b hd.inputs
  have hcount := plans_value R M D hM s (fun _ => one M) a b hd.counts
  have hmean : mean M xs a.tree = mean M xs b.tree := congrArg₂ (div M) hsum hcount
  apply Prod.ext hmean
  change value M (Welford.deviationSquares M xs (mean M xs a.tree)) (zero M) a.tree =
    value M (Welford.deviationSquares M xs (mean M xs b.tree)) (zero M) b.tree
  rw [← hmean]
  exact plans_value R M D hM s _ a b hd.squares

/-- Only primitive conversion laws are premises; equality between integer
conversion and the batch tree's count is derived, including padding changes. -/
theorem converted_batch_count (empty : ReductionPlan 0) (batch : ReductionPlan N)
    (hc : WelfordInduction.CountConversion M N) (hz : D .finite (zero M))
    (hd : ScheduleDomain M D (fun _ => one M) (WelfordInduction.plan empty N) batch) :
    M.fromNat (some .fp32) N = count M batch.tree := by
  have hp := WelfordInduction.converted_count R M D hM s empty.tree N hc hz N le_rfl
  rw [← WelfordInduction.plan_tree empty N] at hp
  exact hp.trans (plans_value R M D hM s _ _ batch hd)

theorem state_batch_statistics (xs : Nat → α) (empty : ReductionPlan 0)
    (batch : ReductionPlan N) (hN : 0 < N) (hc : WelfordInduction.CountConversion M N)
    (hi : WelfordInduction.IterationDomain M D xs empty.tree N)
    (hs : StatisticsDomain M D (WelfordInduction.rowPrefix xs N)
      (WelfordInduction.plan empty N) batch) :
    WelfordInduction.state M xs N =
      WelfordInduction.statistics M (WelfordInduction.rowPrefix xs N) batch.tree := by
  have hp := WelfordInduction.state_statistics R M D hM s xs empty.tree N hN hc hi
  rw [← WelfordInduction.plan_tree empty N] at hp
  exact hp.trans (statistics_schedule R M D hM s _ _ batch hs)

/-- The final division still uses the original converted row length. Both
mean and variance are related to an arbitrary batch schedule by derivation. -/
theorem state_batch_normalized (xs : Nat → α) (empty : ReductionPlan 0)
    (batch : ReductionPlan N) (hN : 0 < N) (hc : WelfordInduction.CountConversion M N)
    (hi : WelfordInduction.IterationDomain M D xs empty.tree N)
    (hs : StatisticsDomain M D (WelfordInduction.rowPrefix xs N)
      (WelfordInduction.plan empty N) batch) :
    ((WelfordInduction.state M xs N).1,
      div M (WelfordInduction.state M xs N).2 (M.fromNat (some .fp32) N)) =
    (mean M (WelfordInduction.rowPrefix xs N) batch.tree,
      div M (WelfordInduction.statistics M (WelfordInduction.rowPrefix xs N) batch.tree).2
        (count M batch.tree)) := by
  rw [state_batch_statistics R M D hM s xs empty batch hN hc hi hs,
    converted_batch_count R M D hM s empty batch hc hi.initial.zero hs.counts]
  rfl

end Transfer
end VeriTile.Triton.FP.WelfordSchedule
