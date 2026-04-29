/-
VeriTile.Examples.OnlineSoftmax

Tier 2 kernel-pair (PAPER CENTERPIECE): online softmax recurrence ≡ batch
softmax. The streaming form

  m_0 = -∞,  l_0 = 0
  m_{k+1} = max(m_k, x_k)
  l_{k+1} = exp(m_k − m_{k+1}) · l_k + exp(x_k − m_{k+1})

produces the same (m, l) as the one-shot batch form

  m = max(x_0, ..., x_{N-1})
  l = Σ exp(x_i − m)

This is the algorithmic core of FlashAttention; Phase C will reuse the
recurrence at the kernel level.

For Phase B we use a degenerate "block size 1" form so that the loop
iterates over single elements, exposing the recurrence directly. Phase C's
FA kernel will instantiate this with full Bk-size blocks.
-/

import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.DSL
import VeriTile.Triton.LoopInvariant
import VeriTile.Examples.Common
import VeriTile.Examples.SoftmaxEq  -- reuse stableSoftmaxKernel + tileMax
import VeriTile.Examples.WelfordMath  -- reuse castFin helper

namespace VeriTile.Examples

open VeriTile.Triton

/-- Batch softmax kernel — same as the existing `stableSoftmaxKernel` from
    `Examples/SoftmaxEq.lean`. Restated here for clarity / direct reference. -/
def batchSoftmaxKernel (xReg yReg : RegionName) (N : Nat) : Kernel :=
  stableSoftmaxKernel xReg yReg N

/-- Online softmax kernel: maintains `(m, l)` registers across single elements
    of the input tile. The Phase C FA kernel will lift this to KV-blocks. -/
def onlineSoftmaxKernel (xReg yReg : RegionName) (N : Nat) : Kernel := triton {
  pid := tl.program_id(0)
  m   := -inf    -- ℝ-channel sentinel (Op.negInf), finite stand-in for -∞
  l   := 0
  tl.for i in $(N) {
    xi    := tl.load($(xReg) + (pid * $(N) + i))
    m_new := tl.max(m, xi)
    l     := tl.exp(m - m_new) * l + tl.exp(xi - m_new)
    m     := m_new
  }
  -- Phase B focuses on (m, l) at the end. Phase C will add the per-element
  -- Y[i] = exp(x_i - m) / l divide.
}

/-- Online softmax math: streaming `m_k` after k iterations. -/
noncomputable def onlineSoftmaxM {N : Nat} (xs : Fin N → ℝ) : Nat → ℝ
  | 0     => -1e38
  | k + 1 =>
      if h : k < N then max (onlineSoftmaxM xs k) (xs ⟨k, h⟩)
      else onlineSoftmaxM xs k

/-- Online softmax math: streaming `l_k` after k iterations. -/
noncomputable def onlineSoftmaxL {N : Nat} (xs : Fin N → ℝ) : Nat → ℝ
  | 0     => 0
  | k + 1 =>
      if h : k < N then
        let m_old := onlineSoftmaxM xs k
        let m_new := onlineSoftmaxM xs (k + 1)
        Real.exp (m_old - m_new) * onlineSoftmaxL xs k + Real.exp (xs ⟨k, h⟩ - m_new)
      else onlineSoftmaxL xs k

/-- Batch softmax math. -/
noncomputable def batchSoftmaxM {N : Nat} (hN : 0 < N) (xs : Fin N → ℝ) : ℝ :=
  tileMax hN xs  -- defined in SoftmaxEq.lean

noncomputable def batchSoftmaxL {N : Nat} (hN : 0 < N) (xs : Fin N → ℝ) : ℝ :=
  ∑ i, Real.exp (xs i - batchSoftmaxM hN xs)

/-! ### Prefix lemmas (induction on `k`) -/

/-- Prefix form of the **max** recurrence: for any prefix length `j+1 ≤ N`,
    the running `onlineSoftmaxM xs (j+1)` matches the batch max over the
    first `j+1` inputs.

    The hypothesis `h_lo` is required because the seed value `M_0 = -1e38`
    is a finite stand-in for `-∞` (see `Op.negInf` in `Triton.Semantics`).
    Without it, e.g. at `k = 1` the recurrence would give
    `max (-1e38) (xs 0)` which only equals `xs 0` when `xs 0 ≥ -1e38`. -/
private theorem online_softmax_prefix_M {N : Nat} (xs : Fin N → ℝ)
    (h_lo : ∀ i, (-1e38 : ℝ) ≤ xs i) :
    ∀ j : Nat, ∀ (hk : j+1 ≤ N),
      onlineSoftmaxM xs (j+1) = (Finset.univ : Finset (Fin (j+1))).sup'
        Finset.univ_nonempty
        (fun i => xs (castFin hk i)) := by
  intro j
  induction j with
  | zero =>
    intro hk
    have hj_lt : 0 < N := hk
    have hM : onlineSoftmaxM xs 1 = max (-1e38) (xs ⟨0, hj_lt⟩) := by
      show (if h : 0 < N then max (onlineSoftmaxM xs 0) (xs ⟨0, h⟩) else _) = _
      simp [hj_lt, onlineSoftmaxM]
    rw [hM]
    have hsup : (Finset.univ : Finset (Fin 1)).sup' Finset.univ_nonempty
        (fun i => xs (castFin hk i)) = xs ⟨0, hj_lt⟩ := by rfl
    rw [hsup]
    exact max_eq_right (h_lo ⟨0, hj_lt⟩)
  | succ j ih =>
    intro hk
    have hj : j+1 ≤ N := Nat.le_of_succ_le hk
    have hj_lt : j+1 < N := hk
    have hM : onlineSoftmaxM xs (j+1+1) =
        max (onlineSoftmaxM xs (j+1)) (xs ⟨j+1, hj_lt⟩) := by
      show (if h : j+1 < N then max (onlineSoftmaxM xs (j+1)) (xs ⟨j+1, h⟩) else _) = _
      simp [hj_lt]
    rw [hM, ih hj]
    rw [Finset.sup'_congr Finset.univ_nonempty
          (Fin.univ_castSuccEmb (j+1)) (fun _ _ => rfl)]
    rw [Finset.sup'_cons (H := by simp)]
    rw [Finset.sup'_map]
    simp only [Function.comp, Fin.castSuccEmb_apply, max_comm]
    rfl

/-- Prefix form of the **sum** recurrence: for any prefix length `k ≤ N`,
    the running `onlineSoftmaxL xs k` equals the batch-style sum
    `∑ i, exp(xs i − onlineSoftmaxM xs k)` restricted to the first `k`
    inputs.  No hypothesis on the magnitude of `xs` is needed here — only
    the M-side identity at `k = N` requires it. -/
private theorem online_softmax_prefix_L {N : Nat} (xs : Fin N → ℝ) :
    ∀ k : Nat, ∀ (hk : k ≤ N),
      onlineSoftmaxL xs k =
        ∑ i : Fin k, Real.exp (xs (castFin hk i) - onlineSoftmaxM xs k) := by
  intro k
  induction k with
  | zero =>
    intro _
    show (0 : ℝ) = _
    simp
  | succ j ih =>
    intro hk
    have hj : j ≤ N := Nat.le_of_succ_le hk
    have hj_lt : j < N := hk
    have ih' := ih hj
    have hL : onlineSoftmaxL xs (j+1) =
        Real.exp (onlineSoftmaxM xs j - onlineSoftmaxM xs (j+1)) * onlineSoftmaxL xs j
        + Real.exp (xs ⟨j, hj_lt⟩ - onlineSoftmaxM xs (j+1)) := by
      show (if h : j < N then
          Real.exp (onlineSoftmaxM xs j - onlineSoftmaxM xs (j+1)) * onlineSoftmaxL xs j
          + Real.exp (xs ⟨j, h⟩ - onlineSoftmaxM xs (j+1)) else _) = _
      simp [hj_lt]
    rw [hL, ih']
    rw [Finset.mul_sum]
    rw [Fin.sum_univ_castSucc (n := j)
          (f := fun i => Real.exp (xs (castFin hk i) - onlineSoftmaxM xs (j+1)))]
    have h1 : ∀ i : Fin j,
        castFin hk i.castSucc = castFin hj i := fun _ => rfl
    have h2 : (castFin hk (Fin.last j) : Fin N) = ⟨j, hj_lt⟩ := rfl
    simp_rw [h1, h2]
    congr 1
    apply Finset.sum_congr rfl
    intros i _
    rw [← Real.exp_add]
    ring_nf

/-- **The math identity (paper centerpiece)**: the online recurrence at depth
    `N` produces the same `(m, l)` as the batch form.

    The hypothesis `h_lo : ∀ i, -1e38 ≤ xs i` is necessary because Phase B's
    `Op.negInf` is the finite stand-in `-1e38` rather than the IEEE `-∞`.
    Without this assumption, the recurrence's first step
    `M_1 = max(-1e38, xs 0)` could disagree with the batch `M = xs 0`
    when `xs 0 < -1e38`.  Phase C will replace `negInf` with a proper
    `⊥` sentinel and drop the hypothesis. -/
theorem online_softmax_recurrence_eq_batch
    {N : Nat} (hN : 0 < N) (xs : Fin N → ℝ)
    (h_lo : ∀ i, (-1e38 : ℝ) ≤ xs i) :
    onlineSoftmaxM xs N = batchSoftmaxM hN xs ∧
    onlineSoftmaxL xs N = batchSoftmaxL hN xs := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hN.ne'
  have hcastFin_id : ∀ i : Fin (n+1),
      castFin (le_refl (n+1)) i = i := fun _ => rfl
  have hM_eq : onlineSoftmaxM xs (n+1) = batchSoftmaxM hN xs := by
    rw [online_softmax_prefix_M xs h_lo n (le_refl _)]
    simp_rw [hcastFin_id]
    rfl
  refine ⟨hM_eq, ?_⟩
  rw [online_softmax_prefix_L xs (n+1) (le_refl _), hM_eq]
  simp_rw [hcastFin_id]
  rfl

/-- Loop invariant for `onlineSoftmaxKernel`: after `k` body iterations,
    register `m` holds `onlineSoftmaxM xs k`, register `l` holds
    `onlineSoftmaxL xs k`, register `pid` holds the original program-id
    (which is also `s.pid`), and the input region `xReg` still holds the
    input tile `xs`. -/
private def P_online_softmax {N : Nat} (xs : Fin N → ℝ) (xReg : RegionName)
    (orig_pid : Nat) (k : Nat) (s : BlockState) : Prop :=
  s.regs "m" = some (Value.scalar (onlineSoftmaxM xs k))
  ∧ s.regs "l" = some (Value.scalar (onlineSoftmaxL xs k))
  ∧ s.regs "pid" = some (Value.scalarNat orig_pid)
  ∧ s.pid = orig_pid
  ∧ InputLoadedAt s xReg N xs

/-- One online-softmax-loop iteration preserves the invariant: given a state
    at depth `k` with the index register `i` rebound to `Value.scalarNat k`,
    executing the body produces a state at depth `k+1`. -/
private theorem online_softmax_step
    (xReg : RegionName) (N : Nat) (xs : Fin N → ℝ) (orig_pid : Nat) :
    ∀ k s, k < N → P_online_softmax xs xReg orig_pid k s →
      ∃ s',
        stepStmts
          [ .assign "xi"    (.load xReg (.add (.mul (.ref "pid")
                                                    (.constNat N))
                                              (.ref "i")))
          , .assign "m_new" (.max2 (.ref "m") (.ref "xi"))
          , .assign "l"     (.add (.mul (.exp (.sub (.ref "m") (.ref "m_new")))
                                        (.ref "l"))
                                  (.exp (.sub (.ref "xi") (.ref "m_new"))))
          , .assign "m"     (.ref "m_new")
          ]
          (s.setReg "i" (Value.scalarNat k)) = some s'
        ∧ P_online_softmax xs xReg orig_pid (k + 1) s' := by
  intro k s hk hP
  obtain ⟨hM, hL, hpid_reg, hpid_eq, hmem⟩ := hP
  -- Bind shorthand for the iteration's input value and recurrence values.
  set xk : ℝ := xs ⟨k, hk⟩ with hxk_def
  set Mk : ℝ := onlineSoftmaxM xs k with hMk_def
  set Lk : ℝ := onlineSoftmaxL xs k with hLk_def
  set Mk1 : ℝ := max Mk xk with hMk1_def
  set Lk1 : ℝ :=
    Real.exp (Mk - Mk1) * Lk + Real.exp (xk - Mk1) with hLk1_def
  -- The explicit final state: chain of 4 setRegs on `s.setReg "i" ...`.
  set s_after_i : BlockState := s.setReg "i" (Value.scalarNat k) with hsi_def
  set s1 : BlockState := s_after_i.setReg "xi" (Value.scalar xk) with hs1_def
  set s2 : BlockState := s1.setReg "m_new" (Value.scalar Mk1) with hs2_def
  set s3 : BlockState := s2.setReg "l" (Value.scalar Lk1) with hs3_def
  set s4 : BlockState := s3.setReg "m" (Value.scalar Mk1) with hs4_def
  -- Pre-step: registers / memory / pid in `s_after_i`.
  have hM' : s_after_i.regs "m" = some (Value.scalar Mk) := by
    simp [hsi_def, BlockState.setReg, hM]
  have hL' : s_after_i.regs "l" = some (Value.scalar Lk) := by
    simp [hsi_def, BlockState.setReg, hL]
  have hpid' : s_after_i.regs "pid" = some (Value.scalarNat orig_pid) := by
    simp [hsi_def, BlockState.setReg, hpid_reg]
  have hi_reg : s_after_i.regs "i" = some (Value.scalarNat k) := by
    simp [hsi_def, BlockState.setReg]
  have hpid_after : s_after_i.pid = orig_pid := by
    simpa [hsi_def, BlockState.setReg] using hpid_eq
  have hload_at : s_after_i.mem xReg (orig_pid * N + k) = xk := by
    have := hmem ⟨k, hk⟩
    show s.mem xReg (orig_pid * N + k) = xk
    rw [← hpid_eq]
    simpa [hxk_def] using this
  -- Provide the witness `s4` and split the conjunction.
  refine ⟨s4, ?_, ?_⟩
  · -- 1. Body reduces to s4.
    -- Step 1: assign "xi" := load(...)
    have hstep1 : stepStmt
        (.assign "xi" (.load xReg (.add (.mul (.ref "pid")
                                              (.constNat N))
                                        (.ref "i"))))
        s_after_i = some s1 := by
      simp [stepStmt, evalOp, hpid', hi_reg, Value.bop, hs1_def,
            BlockState.readMem, hload_at]
    -- Step 2: assign "m_new" := max(m, xi)
    have hxi1 : s1.regs "xi" = some (Value.scalar xk) := by
      simp [hs1_def, BlockState.setReg]
    have hM1 : s1.regs "m" = some (Value.scalar Mk) := by
      simp [hs1_def, BlockState.setReg, hM']
    have hstep2 : stepStmt (.assign "m_new" (.max2 (.ref "m") (.ref "xi"))) s1
                  = some s2 := by
      simp [stepStmt, evalOp, hM1, hxi1, Value.bop, hs2_def, hMk1_def]
    -- Step 3: assign "l" := exp(m - m_new) * l + exp(xi - m_new)
    have hM2 : s2.regs "m" = some (Value.scalar Mk) := by
      simp [hs2_def, BlockState.setReg, hM1]
    have hMnew2 : s2.regs "m_new" = some (Value.scalar Mk1) := by
      simp [hs2_def, BlockState.setReg]
    have hxi2 : s2.regs "xi" = some (Value.scalar xk) := by
      simp [hs2_def, BlockState.setReg, hxi1]
    have hL2 : s2.regs "l" = some (Value.scalar Lk) := by
      simp [hs2_def, BlockState.setReg, hs1_def, BlockState.setReg, hL']
    have hstep3 : stepStmt
        (.assign "l" (.add (.mul (.exp (.sub (.ref "m") (.ref "m_new")))
                                 (.ref "l"))
                           (.exp (.sub (.ref "xi") (.ref "m_new")))))
        s2 = some s3 := by
      simp [stepStmt, evalOp, hM2, hMnew2, hxi2, hL2, Value.bop, Value.uop,
            hs3_def, hLk1_def]
    -- Step 4: assign "m" := m_new
    have hMnew3 : s3.regs "m_new" = some (Value.scalar Mk1) := by
      simp [hs3_def, BlockState.setReg, hMnew2]
    have hstep4 : stepStmt (.assign "m" (.ref "m_new")) s3 = some s4 := by
      simp [stepStmt, evalOp, hMnew3, hs4_def]
    -- Chain the 4 steps via stepStmts.
    show stepStmts _ s_after_i = some s4
    simp [stepStmts, hstep1, hstep2, hstep3, hstep4]
  · -- 2. P_online_softmax xs xReg orig_pid (k+1) s4.
    -- Compute the recurrences at depth (k+1):
    have hM_rec : onlineSoftmaxM xs (k + 1) = Mk1 := by
      show (if h : k < N then max (onlineSoftmaxM xs k) (xs ⟨k, h⟩)
            else onlineSoftmaxM xs k) = Mk1
      simp [hk, hMk_def, hxk_def, hMk1_def]
    have hL_rec : onlineSoftmaxL xs (k + 1) = Lk1 := by
      have : onlineSoftmaxL xs (k + 1) =
          Real.exp (onlineSoftmaxM xs k - onlineSoftmaxM xs (k+1)) *
            onlineSoftmaxL xs k
          + Real.exp (xs ⟨k, hk⟩ - onlineSoftmaxM xs (k+1)) := by
        show (if h : k < N then
          Real.exp (onlineSoftmaxM xs k - onlineSoftmaxM xs (k+1)) *
            onlineSoftmaxL xs k
          + Real.exp (xs ⟨k, h⟩ - onlineSoftmaxM xs (k+1))
            else onlineSoftmaxL xs k) = _
        simp [hk]
      rw [this, hM_rec, hLk1_def, hMk_def, hxk_def, hLk_def]
    refine ⟨?_, ?_, ?_, ?_, ?_⟩
    · -- s4.regs "m" = onlineSoftmaxM xs (k+1)
      simp [hs4_def, BlockState.setReg, hM_rec]
    · -- s4.regs "l" = onlineSoftmaxL xs (k+1)
      simp [hs4_def, BlockState.setReg, hs3_def, BlockState.setReg, hL_rec]
    · -- s4.regs "pid" preserved.
      simp [hs4_def, hs3_def, hs2_def, hs1_def, BlockState.setReg, hpid']
    · -- s4.pid = orig_pid (setReg never touches pid).
      simpa [hs4_def, hs3_def, hs2_def, hs1_def, BlockState.setReg]
        using hpid_after
    · -- InputLoadedAt s4: memory unchanged across all setRegs.
      intro i
      have hi := hmem i
      have hpidS4 : s4.pid = orig_pid := by
        simpa [hs4_def, hs3_def, hs2_def, hs1_def, BlockState.setReg]
          using hpid_after
      show s4.mem xReg (s4.pid * N + i.val) = xs i
      rw [hpidS4, ← hpid_eq]
      simpa [hs4_def, hs3_def, hs2_def, hs1_def, BlockState.setReg]
        using hi

/-- Operational correctness: the online softmax kernel computes (m, l)
    matching `onlineSoftmaxM xs N` and `onlineSoftmaxL xs N`. -/
theorem online_softmax_correct
    (xReg yReg : RegionName) (N : Nat) (hN : 0 < N)
    (s : BlockState) (xs : Fin N → ℝ)
    (h_x : InputLoadedAt s xReg N xs) :
    let final := exec (onlineSoftmaxKernel xReg yReg N) s
    final.bind (fun s' => s'.regs "m" >>= Value.asScalar)
        = some (onlineSoftmaxM xs N)
    ∧ final.bind (fun s' => s'.regs "l" >>= Value.asScalar)
        = some (onlineSoftmaxL xs N) := by
  -- The kernel is: 3 pre-loop assigns, 1 forLoop, no post-loop stmts.
  -- Reduce the prefix manually, apply forLoop_inv, then conclude.
  -- Pre-loop state: pid := s.pid, m := -1e38, l := 0.
  set s0 : BlockState :=
    ((s.setReg "pid" (Value.scalarNat s.pid)).setReg "m"
      (Value.scalar (-1e38))).setReg "l" (Value.scalar 0) with hs0_def
  -- s0 satisfies P_online_softmax at depth 0.
  have hP0 : P_online_softmax xs xReg s.pid 0 s0 := by
    refine ⟨?_, ?_, ?_, ?_, ?_⟩
    · simp [hs0_def, BlockState.setReg, onlineSoftmaxM]
    · simp [hs0_def, BlockState.setReg, onlineSoftmaxL]
    · simp [hs0_def, BlockState.setReg]
    · simp [hs0_def, BlockState.setReg]
    · intro i
      have := h_x i
      simpa [hs0_def, BlockState.setReg] using this
  -- Apply forLoop_inv to obtain a final state s_final after the loop.
  obtain ⟨s_final, h_loop_eq, hPn⟩ :=
    forLoop_inv (P := P_online_softmax xs xReg s.pid) (s_init := s0) hP0
      (fun k s' h_lt hP' =>
        online_softmax_step xReg N xs s.pid k s' h_lt hP')
  obtain ⟨hMfin, hLfin, _, _, _⟩ := hPn
  -- Reduce the full kernel exec by composing prefix → s0 → forLoop → s_final.
  have stmts_cons : ∀ (st : Stmt) (rest : List Stmt) (s s' : BlockState),
      stepStmt st s = some s' →
      stepStmts (st :: rest) s = stepStmts rest s' := by
    intro st rest s s' h
    conv_lhs => unfold stepStmts
    rw [h]
  have stmts_nil : ∀ (s : BlockState), stepStmts [] s = some s := by
    intro s
    conv_lhs => unfold stepStmts
  -- Statement-level reductions for the 3 pre-loop statements.
  have h_pid : stepStmt (.assign "pid" Op.programId) s
                = some (s.setReg "pid" (Value.scalarNat s.pid)) := by
    simp [stepStmt, evalOp]
  have h_m0 : stepStmt (.assign "m" Op.negInf)
              (s.setReg "pid" (Value.scalarNat s.pid))
            = some ((s.setReg "pid" (Value.scalarNat s.pid)).setReg "m"
                    (Value.scalar (-1e38))) := by
    simp [stepStmt, evalOp]
  have h_l0 : stepStmt (.assign "l" (Op.const 0))
              ((s.setReg "pid" (Value.scalarNat s.pid)).setReg "m"
                (Value.scalar (-1e38))) = some s0 := by
    simp [stepStmt, evalOp, hs0_def]
  -- The full stepStmts evaluation, threaded through all 4 statements.
  have h_kernel : stepStmts
        (onlineSoftmaxKernel xReg yReg N).body s
      = some s_final := by
    -- Unfold the kernel so the body is an explicit cons-list.
    show stepStmts
        [ .assign "pid" .programId
        , .assign "m" .negInf
        , .assign "l" (.const 0)
        , .forLoop "i" N
            [ .assign "xi" (.load xReg (.add (.mul (.ref "pid")
                                                    (.constNat N))
                                              (.ref "i")))
            , .assign "m_new" (.max2 (.ref "m") (.ref "xi"))
            , .assign "l" (.add (.mul (.exp (.sub (.ref "m") (.ref "m_new")))
                                      (.ref "l"))
                                (.exp (.sub (.ref "xi") (.ref "m_new"))))
            , .assign "m" (.ref "m_new")
            ]
        ] s = some s_final
    rw [stmts_cons _ _ _ _ h_pid]
    rw [stmts_cons _ _ _ _ h_m0]
    rw [stmts_cons _ _ _ _ h_l0]
    rw [stmts_cons _ _ _ _ h_loop_eq]
    exact stmts_nil _
  -- Plug into the conclusion.
  simp only [exec, h_kernel, Option.bind_some]
  refine ⟨?_, ?_⟩
  · -- m readback.
    show (s_final.regs "m" >>= Value.asScalar) = some (onlineSoftmaxM xs N)
    rw [hMfin]
    rfl
  · -- l readback.
    show (s_final.regs "l" >>= Value.asScalar) = some (onlineSoftmaxL xs N)
    rw [hLfin]
    rfl

end VeriTile.Examples
