import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealFormula

/-! Independent input formula for the actual two helper invocations.
Each visited key lane contributes a base-2 weight, including a finite penalty
on future keys in diagonal stages. Masked K/V loads become zero inputs; their
scores are still included in the denominator. No kernel execution or computed
output is used to define the expected value.

Inputs name tiles by their starting key index. This retains independent source
strides and visits whose origins are not multiples of BLOCK_N. Repeated visits
are retained: STAGE=2 processes the same diagonal span twice. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealFormula
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealFormula
  (summarize initial block_updates_eq_closed initialized_result_normalized unshiftedSum)
noncomputable section

structure Inputs (BM BN BD : Nat) where
  query : Fin BM → Fin BD → ℝ
  key : Nat → Fin BN → Fin BD → ℝ
  value : Nat → Fin BN → Fin BD → ℝ
  queryScale : ℝ
  keyScale : Nat → ℝ

structure Visit where
  start : Nat
  diagonal : Bool
  deriving DecidableEq

def tripCount (lower upper BN : Nat) : Nat := (upper - lower + BN - 1) / BN

def visits (lower upper BN : Nat) (diagonal : Bool) : List Visit :=
  (List.range (tripCount lower upper BN)).map (fun j => ⟨lower+j*BN, diagonal⟩)

def firstVisits (startM BM BN stage : Nat) : List Visit :=
  if stage = 2 then visits (startM*BM) ((startM+1)*BM) BN Bool.true
  else visits 0 (startM*BM) BN Bool.false

def secondVisits (startM BM BN : Nat) : List Visit :=
  visits (startM*BM) ((startM+1)*BM) BN Bool.true

def allVisits (startM BM BN stage : Nat) : List Visit :=
  firstVisits startM BM BN stage ++ secondVisits startM BM BN

theorem stage_two_repeats_diagonal (startM BM BN : Nat) :
    allVisits startM BM BN 2 = secondVisits startM BM BN ++ secondVisits startM BM BN := by
  simp [allVisits, firstVisits, secondVisits]

theorem stage_three_prefix_then_diagonal (startM BM BN : Nat) :
    allVisits startM BM BN 3 = visits 0 (startM*BM) BN Bool.false ++ secondVisits startM BM BN := by
  simp [allVisits, firstVisits]

def score {BM BN BD : Nat} (xs : Inputs BM BN BD) (startM N HA : Nat)
    (v : Visit) (row : Fin BM) (key : Fin BN) : ℝ :=
  (∑ d : Fin BD,
    (if startM*BM+row.val < N ∧ d.val < HA then xs.query row d else 0) *
    (if v.start+key.val < N ∧ d.val < HA then xs.key v.start key d else 0)) *
      xs.queryScale * xs.keyScale (v.start / BN)

/-- This is an additive finite score penalty. It is not a negative-infinity
score and does not turn the resulting real exponential into zero. -/
def penalty (diagonal : Bool) (row key : Nat) : ℝ :=
  if diagonal = Bool.true ∧ row < key then -1000000 else 0

def logit {BM BN BD : Nat} (xs : Inputs BM BN BD) (startM N HA : Nat)
    (v : Visit) (row : Fin BM) (key : Fin BN) : ℝ :=
  score xs startM N HA v row key + penalty v.diagonal (startM*BM+row.val) (v.start+key.val)

def loadedValue {BM BN BD : Nat} (xs : Inputs BM BN BD) (N HA : Nat)
    (v : Visit) (key : Fin BN) (d : Fin BD) : ℝ :=
  if v.start+key.val < N ∧ d.val < HA then xs.value v.start key d else 0

def blockPairs {BM BN BD : Nat} (xs : Inputs BM BN BD) (startM N HA : Nat)
    (v : Visit) (row : Fin BM) (d : Fin BD) : List (ℝ × ℝ) :=
  (List.finRange BN).map (fun key =>
    (logit xs startM N HA v row key, loadedValue xs N HA v key d))

def blocks {BM BN BD : Nat} (xs : Inputs BM BN BD) (startM N HA stage : Nat)
    (row : Fin BM) (d : Fin BD) : List (List (ℝ × ℝ)) :=
  (allVisits startM BM BN stage).map (fun v => blockPairs xs startM N HA v row d)

def expected {BM BN BD : Nat} (xs : Inputs BM BN BD) (startM N HA stage : Nat)
    (row : Fin BM) (d : Fin BD) : ℝ :=
  let pairs := (blocks xs startM N HA stage row d).flatten
  (pairs.map (fun sv => pow2 sv.1 * sv.2)).sum / (pairs.map (fun sv => pow2 sv.1)).sum

/-- Pure mathematical block updates compose to the independent batch formula.
Source execution, address safety and IO pinning remain separate obligations. -/
theorem online_updates_normalized {BM BN BD : Nat} (xs : Inputs BM BN BD)
    (startM N HA stage : Nat) (row : Fin BM) (d : Fin BD) :
    let state := (blocks xs startM N HA stage row d).foldl summarize initial
    state.numerator / state.denominator = expected xs startM N HA stage row d := by
  dsimp only
  rw [block_updates_eq_closed, initialized_result_normalized]
  simp [unshiftedSum, expected]

theorem every_lane_has_positive_weight {BM BN BD : Nat} (xs : Inputs BM BN BD)
    (startM N HA : Nat) (v : Visit) (row : Fin BM) (key : Fin BN) :
    0 < pow2 (logit xs startM N HA v row key) := pow2_pos _

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealFormula
