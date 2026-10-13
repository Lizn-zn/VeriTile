import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.Order.Floor.Div
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.KldivTriton.FullMath
open scoped BigOperators
noncomputable section

/-- Independent source formula: this kernel has no epsilon clamp. -/
def loss (logTarget : Bool) (prediction target : ℝ) : ℝ :=
  if logTarget then Real.exp target * (target - prediction)
  else target * (Real.log target - prediction)

/-- The backward kernel precedes the wrapper's optional gradient scaling. -/
def gradient (logTarget : Bool) (target : ℝ) : ℝ :=
  if logTarget then -Real.exp target else -target

/-- Real padding vanishes; this theorem makes no claim about IEEE 0*log(0). -/
theorem loss_zero (logTarget : Bool) : loss logTarget 0 0 = 0 := by
  cases logTarget <;> simp [loss]

def paddedLoss (logTarget : Bool) (prediction target : Nat → ℝ) (N k : Nat) : ℝ :=
  loss logTarget (if k < N then prediction k else 0) (if k < N then target k else 0)

theorem paddedLoss_eq (logTarget : Bool) (prediction target : Nat → ℝ) (N k : Nat) :
    paddedLoss logTarget prediction target N k =
      if k < N then loss logTarget (prediction k) (target k) else 0 := by
  by_cases h : k < N <;> simp [paddedLoss, h, loss_zero]

/-- Each nonzero-reduction store contains this block's sum, not a whole-row sum. -/
def chunkLoss (logTarget : Bool) (prediction target : Nat → ℝ) (N B chunk : Nat) : ℝ :=
  ∑ lane : Fin B, paddedLoss logTarget prediction target N (chunk * B + lane.val)

/-- Actual number of positive-stride loop iterations. -/
def rounds (N B : Nat) : Nat := (N + B - 1) / B

theorem rounds_cover (N B : Nat) (hB : 0 < B) : N ≤ rounds N B * B := by
  have h := (ceilDiv_le_iff_le_mul (a := B) (b := N) hB).mp
    (show N ⌈/⌉ B ≤ N ⌈/⌉ B from le_rfl)
  simpa only [rounds, Nat.ceilDiv_eq_add_pred_div, Nat.mul_comm] using h

theorem rounds_le_iff (N B c : Nat) (hB : 0 < B) :
    rounds N B ≤ c ↔ N ≤ c * B := by
  simpa only [rounds, Nat.ceilDiv_eq_add_pred_div, Nat.mul_comm] using
    (ceilDiv_le_iff_le_mul (a := B) (b := N) (c := c) hB)

theorem lt_rounds_iff (N B c : Nat) (hB : 0 < B) :
    c < rounds N B ↔ c * B < N := by
  have h := rounds_le_iff N B c hB
  omega

@[simp] theorem rounds_zero (B : Nat) : rounds 0 B = 0 := by
  cases B with
  | zero => simp [rounds]
  | succ b => simp [rounds, Nat.div_eq_of_lt (Nat.lt_succ_self b)]

/-- Actual forward footprint length. An empty reduced row has no final scalar store. -/
def outputCount (N B reduction : Nat) : Nat := if reduction = 0 then N else rounds N B

@[simp] theorem outputCount_zero (B reduction : Nat) : outputCount 0 B reduction = 0 := by
  simp [outputCount]

/-- The independently specified output sequence, before any host post-reduction. -/
def forwardOutput (logTarget : Bool) (prediction target : Nat → ℝ)
    (N B reduction index : Nat) : ℝ :=
  if reduction = 0 then loss logTarget (prediction index) (target index)
  else chunkLoss logTarget prediction target N B index

/-- Two reduced chunks occupy two different cells even though the wrapper allocates
only one scalar per row. This concrete arithmetic fact guards against row-sum substitution. -/
theorem multiblock_footprint : outputCount 3 2 1 = 2 := by decide

/-- Zero-padding produces different per-chunk results in this nonempty example. -/
theorem multiblock_values :
    forwardOutput Bool.true (fun _ ↦ 2) (fun _ ↦ 0) 3 2 1 0 = -4 ∧
    forwardOutput Bool.true (fun _ ↦ 2) (fun _ ↦ 0) 3 2 1 1 = -2 := by
  norm_num [forwardOutput, chunkLoss, paddedLoss, loss, Fin.sum_univ_succ]

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.KldivTriton.FullMath
