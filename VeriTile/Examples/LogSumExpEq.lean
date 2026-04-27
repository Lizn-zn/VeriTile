/-
VeriTile.Examples.LogSumExpEq

Worked equivalence example: direct log-sum-exp kernel vs shift-trick
log-sum-exp kernel. The math identity `log_sum_exp_shift_invariant` is the
load-bearing fact; the kernel-level theorem composes operational walk-throughs
of both kernels with this identity.

Status (Phase A Task 1.1.A): infrastructure laid; math lemma and kernel
theorems sorry'd, to be closed by subsequent subagents (1.1.B, 1.1.C).
-/

import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.DSL
import VeriTile.Examples.Common
import VeriTile.Examples.SoftmaxEq

namespace VeriTile.Examples

open VeriTile.Triton

/-- Direct log-sum-exp kernel: y = log(Σ exp(x)). -/
def directLSEKernel (xReg yReg : RegionName) (N : Nat) : Kernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(N) + tl.arange($(N))
  x    := tl.load($(xReg), offs)
  e    := tl.exp(x)
  s    := tl.sum(e)
  y    := tl.log(s)
  tl.store($(yReg), pid, y)
}

/-- Shift-trick LSE kernel: y = m + log(Σ exp(x - m)) where m = max(x). -/
def stableLSEKernel (xReg yReg : RegionName) (N : Nat) : Kernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(N) + tl.arange($(N))
  x    := tl.load($(xReg), offs)
  m    := tl.max(x)
  e    := tl.exp(x - m)
  s    := tl.sum(e)
  y    := m + tl.log(s)
  tl.store($(yReg), pid, y)
}

/-- The load-bearing math identity for `log_sum_exp_refinement`.
    Closed by Phase A Task 1.1.B. -/
theorem log_sum_exp_shift_invariant {n : Nat} (hn : 0 < n) (x : Fin n → ℝ) (m : ℝ) :
    Real.log (∑ i, Real.exp (x i)) = m + Real.log (∑ i, Real.exp (x i - m)) := by
  have h_factor : ∀ i : Fin n, Real.exp (x i) = Real.exp m * Real.exp (x i - m) := by
    intro i
    rw [← Real.exp_add]
    ring_nf
  rw [show (∑ i, Real.exp (x i)) = ∑ i, Real.exp m * Real.exp (x i - m) from
        Finset.sum_congr rfl (fun i _ => h_factor i)]
  rw [← Finset.mul_sum]
  have h_em_pos : 0 < Real.exp m := Real.exp_pos m
  have h_sum_pos : 0 < ∑ i, Real.exp (x i - m) := by
    apply Finset.sum_pos
    · intro i _; exact Real.exp_pos _
    · exact ⟨⟨0, hn⟩, Finset.mem_univ _⟩
  rw [Real.log_mul (ne_of_gt h_em_pos) (ne_of_gt h_sum_pos)]
  rw [Real.log_exp]

/-! ## Kernel-level refinement -/

/-- Read region `region` at offset `basePid` (single scalar per program_id).
    Unlike softmax which writes a tile, LSE writes one cell per pid. -/
noncomputable def observeLSE
    (sf : Option BlockState) (region : RegionName) (basePid : Nat) : Option ℝ :=
  sf.map (·.readMem region basePid)

/-- What `directLSEKernel` writes at `Y[pid]`. -/
noncomputable def directLSESpec {N : Nat} (xs : Fin N → ℝ) : ℝ :=
  Real.log (∑ j, Real.exp (xs j))

/-- What `stableLSEKernel` writes at `Y[pid]`. -/
noncomputable def stableLSESpec {N : Nat} (xs : Fin N → ℝ) (m : ℝ) : ℝ :=
  m + Real.log (∑ j, Real.exp (xs j - m))

/-- **Direct LSE kernel correctness.** Single-cell observation at Y[pid]. -/
theorem direct_lse_correct
    (xReg yReg : RegionName)
    (N : Nat) (_hN : 0 < N) (s : BlockState) (xs : Fin N → ℝ)
    (_h_x : InputLoadedAt s xReg N xs) :
    observeLSE (exec (directLSEKernel xReg yReg N) s) yReg s.pid
      = some (directLSESpec xs) := by
  -- Post-RP2: address arithmetic stays in `Nat`, no `hcast` needed.
  simp [observeLSE, exec, directLSEKernel, stepStmts, stepStmt, evalOp,
        Value.bop, Value.uop, Value.reduceSum,
        BlockState.setReg, BlockState.readMem, BlockState.writeMem,
        directLSESpec]
  unfold InputLoadedAt at _h_x
  simp_rw [_h_x]

/-- **Stable LSE kernel correctness.** Single-cell observation; closed form
    is `m + log(Σ exp(x - m))` where `m = tileMax xs`. -/
theorem stable_lse_correct
    (xReg yReg : RegionName)
    (N : Nat) (hN : 0 < N) (s : BlockState) (xs : Fin N → ℝ)
    (_h_x : InputLoadedAt s xReg N xs) :
    observeLSE (exec (stableLSEKernel xReg yReg N) s) yReg s.pid
      = some (stableLSESpec xs (tileMax hN xs)) := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hN.ne'
  -- Post-RP2: address arithmetic stays in `Nat`, no `hcast` needed.
  simp [observeLSE, exec, stableLSEKernel, stepStmts, stepStmt, evalOp,
        Value.bop, Value.uop, Value.reduceSum, Value.reduceMax,
        BlockState.setReg, BlockState.readMem, BlockState.writeMem,
        stableLSESpec, tileMax]
  unfold InputLoadedAt at _h_x
  simp_rw [_h_x]

/-- **Refinement: `directLSEKernel` and `stableLSEKernel` produce the same
    `Y[pid]` value.** Composes the two correctness lemmas via the math
    identity `log_sum_exp_shift_invariant`. -/
theorem log_sum_exp_refinement
    (xReg yReg : RegionName)
    (N : Nat) (hN : 0 < N) (s : BlockState) (xs : Fin N → ℝ)
    (h_x : InputLoadedAt s xReg N xs) :
    observeLSE (exec (directLSEKernel xReg yReg N) s) yReg s.pid =
    observeLSE (exec (stableLSEKernel xReg yReg N) s) yReg s.pid := by
  rw [direct_lse_correct xReg yReg N hN s xs h_x,
      stable_lse_correct xReg yReg N hN s xs h_x]
  congr 1
  unfold directLSESpec stableLSESpec
  exact log_sum_exp_shift_invariant hN xs (tileMax hN xs)

end VeriTile.Examples
