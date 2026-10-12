import VeriTile.Triton

/-! Independent input-array formulas for both chunk recurrences. These are
real formulas, with no execution state, materialized carry buffer or FP rule.
The forward output is exclusive: chunk zero stores the seed. Reverse DI at
chunk t stores the suffix beginning at t+1, while DL stores suffix zero.
DG is the complete BK×BV tile contraction, before the host's split reduction. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealMath
open scoped BigOperators
noncomputable section

structure Inputs (BK BV : Nat) where
  source : Nat → Fin BK → Fin BV → ℝ
  gate : Nat → ℝ
  seed : Fin BK → Fin BV → ℝ
  incoming : Nat → Fin BK → Fin BV → ℝ

def initial (a : Inputs BK BV) (hasSeed : Bool) (e : Fin BK) (j : Fin BV) : ℝ :=
  if hasSeed then a.seed e j else 0

/-- Seed times all earlier gates, plus each earlier source tile times the gates
strictly between its chunk and the output chunk. -/
def forward (a : Inputs BK BV) (hasSeed : Bool) (c : Nat) (e : Fin BK) (j : Fin BV) : ℝ :=
  initial a hasSeed e j * (∏ t ∈ Finset.range c, a.gate t) +
    ∑ t ∈ Finset.range c, a.source t e j * ∏ u ∈ Finset.Ico (t+1) c, a.gate u

@[simp] theorem forward_zero (a : Inputs BK BV) (hasSeed : Bool) (e : Fin BK) (j : Fin BV) :
    forward a hasSeed 0 e j = initial a hasSeed e j := by
  simp [forward]

theorem forward_step (a : Inputs BK BV) (hasSeed : Bool) (c : Nat) (e : Fin BK) (j : Fin BV) :
    forward a hasSeed (c+1) e j = forward a hasSeed c e j * a.gate c + a.source c e j := by
  unfold forward
  rw [Finset.prod_range_succ, Finset.sum_range_succ]
  simp only [Finset.Ico_self, Finset.prod_empty, mul_one]
  have h : (∑ t ∈ Finset.range c,
      a.source t e j * ∏ u ∈ Finset.Ico (t+1) (c+1), a.gate u) =
      (∑ t ∈ Finset.range c, a.source t e j * ∏ u ∈ Finset.Ico (t+1) c, a.gate u) * a.gate c := by
    rw [Finset.sum_mul]
    apply Finset.sum_congr rfl
    intro t ht
    rw [Finset.prod_Ico_succ_top (by simpa only [Finset.mem_range, Nat.succ_le_iff] using ht)]
    ring
  rw [h]
  ring

/-- Suffix formula over incoming DS and gates. The gate at u itself does not
multiply DS[u]; it multiplies the carry from later chunks. -/
def reverse (a : Inputs BK BV) (N c : Nat) (e : Fin BK) (j : Fin BV) : ℝ :=
  ∑ u ∈ Finset.Ico c N, a.incoming u e j * ∏ w ∈ Finset.Ico c u, a.gate w

@[simp] theorem reverse_end (a : Inputs BK BV) (N : Nat) (e : Fin BK) (j : Fin BV) :
    reverse a N N e j = 0 := by
  simp [reverse]

theorem reverse_step (a : Inputs BK BV) (N c : Nat) (hc : c < N) (e : Fin BK) (j : Fin BV) :
    reverse a N c e j = reverse a N (c+1) e j * a.gate c + a.incoming c e j := by
  unfold reverse
  rw [Finset.sum_eq_sum_Ico_succ_bot hc, Finset.Ico_self, Finset.prod_empty,
    mul_one, Finset.sum_mul]
  have h : ∀ u ∈ Finset.Ico (c+1) N,
      a.incoming u e j * (∏ w ∈ Finset.Ico c u, a.gate w) =
      (a.incoming u e j * ∏ w ∈ Finset.Ico (c+1) u, a.gate w) * a.gate c := by
    intro u hu
    rw [Finset.prod_eq_prod_Ico_succ_bot (by have := (Finset.mem_Ico.mp hu).1; omega)]
    ring
  rw [Finset.sum_congr rfl h]
  ring

def gateGradient (a : Inputs BK BV) (N c : Nat) : ℝ :=
  ∑ e : Fin BK, ∑ j : Fin BV, reverse a N (c+1) e j * a.source c e j

/-- An arbitrary count of forward updates implements the closed form. This
arithmetic fold is later connected to the actual source loop and its stores. -/
theorem forward_fold (a : Inputs BK BV) (hasSeed : Bool) (count : Nat)
    (e : Fin BK) (j : Fin BV) :
    (List.range count).foldl (fun carry c => carry * a.gate c + a.source c e j)
      (initial a hasSeed e j) = forward a hasSeed count e j := by
  induction count with
  | zero => simp
  | succ c ih =>
    rw [List.range_succ, List.foldl_append]
    simpa only [List.foldl_cons, List.foldl_nil, ih] using (forward_step a hasSeed c e j).symm

/-- The state after k reverse iterations is suffix N-k. In particular the
source's N-1 loop leaves suffix one for the final DL multiply-add. -/
theorem reverse_fold (a : Inputs BK BV) (N count : Nat) (hc : count ≤ N)
    (e : Fin BK) (j : Fin BV) :
    (List.range count).foldl
      (fun carry step => carry * a.gate (N-1-step) + a.incoming (N-1-step) e j)
      0 = reverse a N (N-count) e j := by
  induction count with
  | zero => simp
  | succ count ih =>
    rw [List.range_succ, List.foldl_append]
    simp only [List.foldl_cons, List.foldl_nil]
    rw [ih (by omega)]
    have hindex : N - (count+1) = N-1-count := by omega
    have hsucc : N - count = (N-1-count)+1 := by omega
    rw [hindex, hsucc]
    exact (reverse_step a N (N-1-count) (by omega) e j).symm

/-- No reverse-loop store exists for N=1. The separate DL step still returns
DS[0], not the initial zero carry. -/
theorem reverse_single (a : Inputs BK BV) (e : Fin BK) (j : Fin BV) :
    reverse a 1 0 e j = a.incoming 0 e j := by
  rw [reverse_step a 1 0 (by omega) e j]
  simp

end
end VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealMath
