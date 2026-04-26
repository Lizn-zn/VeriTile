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
import VeriTile.Examples.SoftmaxEq

namespace VeriTile.Examples

open VeriTile.Triton

/-- Direct log-sum-exp kernel: y = log(Σ exp(x)). -/
def directLSEKernel (N : Nat) : Kernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(N) + tl.arange($(N))
  x    := tl.load(X, offs)
  e    := tl.exp(x)
  s    := tl.sum(e)
  y    := tl.log(s)
  tl.store(Y, pid, y)
}

/-- Shift-trick LSE kernel: y = m + log(Σ exp(x - m)) where m = max(x). -/
def stableLSEKernel (N : Nat) : Kernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(N) + tl.arange($(N))
  x    := tl.load(X, offs)
  m    := tl.max(x)
  e    := tl.exp(x - m)
  s    := tl.sum(e)
  y    := m + tl.log(s)
  tl.store(Y, pid, y)
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

/-- The `Y` region at offset `basePid` after running an LSE kernel.
    Unlike softmax, LSE writes a single scalar per program_id (at offset = pid),
    so the observation reads one cell. -/
noncomputable def observeLSE
    (sf : Option BlockState) (basePid : Nat) : Option ℝ :=
  sf.map (·.readMem "Y" basePid)

/-- What `directLSEKernel` writes at `Y[pid]`. -/
noncomputable def directLSESpec {N : Nat} (xs : Fin N → ℝ) : ℝ :=
  Real.log (∑ j, Real.exp (xs j))

/-- What `stableLSEKernel` writes at `Y[pid]`. -/
noncomputable def stableLSESpec {N : Nat} (xs : Fin N → ℝ) (m : ℝ) : ℝ :=
  m + Real.log (∑ j, Real.exp (xs j - m))

/-- **Direct LSE kernel correctness.** Single-cell observation at Y[pid]. -/
theorem direct_lse_correct
    (N : Nat) (_hN : 0 < N) (s : BlockState) (xs : Fin N → ℝ)
    (_h_x : InputLoaded s N xs) :
    observeLSE (exec (directLSEKernel N) s) s.pid
      = some (directLSESpec xs) := by
  have hcast :
      ∀ k : Fin N,
        realToNat ((↑s.pid : ℝ) * (↑N : ℝ) + (↑(↑k : ℕ) : ℝ)) = s.pid * N + k.val := by
    intro k
    unfold realToNat
    have heq :
        ((↑s.pid : ℝ) * (↑N : ℝ) + (↑(↑k : ℕ) : ℝ)) = ((s.pid * N + k.val : ℕ) : ℝ) := by
      push_cast; ring
    rw [heq]; exact Nat.floor_natCast _
  have hpid : realToNat ((↑s.pid : ℝ)) = s.pid := by
    unfold realToNat
    exact Nat.floor_natCast _
  simp [observeLSE, exec, directLSEKernel, stepStmts, stepStmt, evalOp,
        Value.bop, Value.uop, Value.reduceSum,
        BlockState.setReg, BlockState.readMem, BlockState.writeMem,
        directLSESpec]
  unfold InputLoaded at _h_x
  simp_rw [hcast, _h_x, hpid]
  simp

/-- **Stable LSE kernel correctness.** Single-cell observation; closed form
    is `m + log(Σ exp(x - m))` where `m = tileMax xs`. -/
theorem stable_lse_correct
    (N : Nat) (hN : 0 < N) (s : BlockState) (xs : Fin N → ℝ)
    (_h_x : InputLoaded s N xs) :
    observeLSE (exec (stableLSEKernel N) s) s.pid
      = some (stableLSESpec xs (tileMax hN xs)) := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hN.ne'
  have hcast :
      ∀ k : Fin (n + 1),
        realToNat ((↑s.pid : ℝ) * (↑(n + 1) : ℝ) + (↑(↑k : ℕ) : ℝ))
          = s.pid * (n + 1) + k.val := by
    intro k
    unfold realToNat
    have heq :
        ((↑s.pid : ℝ) * (↑(n + 1) : ℝ) + (↑(↑k : ℕ) : ℝ))
          = ((s.pid * (n + 1) + k.val : ℕ) : ℝ) := by
      push_cast; ring
    rw [heq]; exact Nat.floor_natCast _
  have hpid : realToNat ((↑s.pid : ℝ)) = s.pid := by
    unfold realToNat
    exact Nat.floor_natCast _
  simp [observeLSE, exec, stableLSEKernel, stepStmts, stepStmt, evalOp,
        Value.bop, Value.uop, Value.reduceSum, Value.reduceMax,
        BlockState.setReg, BlockState.readMem, BlockState.writeMem,
        stableLSESpec, tileMax]
  simp only [show ((n : ℝ) + 1) = ((n + 1 : ℕ) : ℝ) by push_cast; ring]
  unfold InputLoaded at _h_x
  simp_rw [hcast, _h_x, hpid]
  simp

/-- **Refinement: `directLSEKernel` and `stableLSEKernel` produce the same
    `Y[pid]` value.** Composes the two correctness lemmas via the math
    identity `log_sum_exp_shift_invariant`. -/
theorem log_sum_exp_refinement
    (N : Nat) (hN : 0 < N) (s : BlockState) (xs : Fin N → ℝ)
    (h_x : InputLoaded s N xs) :
    observeLSE (exec (directLSEKernel N) s) s.pid =
    observeLSE (exec (stableLSEKernel N) s) s.pid := by
  rw [direct_lse_correct N hN s xs h_x,
      stable_lse_correct N hN s xs h_x]
  congr 1
  unfold directLSESpec stableLSESpec
  exact log_sum_exp_shift_invariant hN xs (tileMax hN xs)

end VeriTile.Examples
