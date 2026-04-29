/-
VeriTile.Examples.WelfordKernels

Tier 2 kernel-pair: Welford online recurrence kernel ≡ two-pass variance
kernel. Lifts `welford_eq_two_pass` (Examples/WelfordMath.lean:204) to the
operational level using `forLoop_inv`.

Output spec: each output cell holds (μ, var) packed somehow — for this
kernel-pair the simplest form is two separate output regions `meanReg` and
`varReg`, each storing a single scalar at offset `0`. (Production
LayerNorm uses the same pattern.)

Skeleton only — the three correctness theorems are sorry'd here and will
be filled in by tasks T4.2 / T4.3 / T4.4.
-/

import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.DSL
import VeriTile.Triton.LoopInvariant
import VeriTile.Examples.Common
import VeriTile.Examples.WelfordMath

namespace VeriTile.Examples

open VeriTile.Triton

/-- Two-pass variance kernel: `tl.sum` twice, then per-element variance. -/
def twopassWelfordKernel (xReg meanReg varReg : RegionName)
    (blockSize : Nat) : Kernel := triton {
  pid    := tl.program_id(0)
  offs   := pid * $(blockSize) + tl.arange($(blockSize))
  x      := tl.load($(xReg) + offs)
  s_x    := tl.sum(x)
  μ      := s_x / tl.toReal($(blockSize))
  d      := x - μ
  s_d2   := tl.sum(d * d)
  v      := s_d2 / tl.toReal($(blockSize))
  tl.store($(meanReg), μ)
  tl.store($(varReg), v)
}

/-- Online Welford kernel: `forLoop` maintains `(M, S)` per iteration. -/
def onlineWelfordKernel (xReg meanReg varReg : RegionName)
    (blockSize : Nat) : Kernel := triton {
  pid := tl.program_id(0)
  M   := 0
  S   := 0
  tl.for i in $(blockSize) {
    xi     := tl.load($(xReg) + (pid * $(blockSize) + i))
    -- Standard Welford update:
    --   M' = M + (xi - M) / (i + 1)
    --   S' = S + (xi - M) * (xi - M')
    delta  := xi - M
    M      := M + delta / (tl.toReal(i) + 1)
    delta2 := xi - M
    S      := S + delta * delta2
  }
  tl.store($(meanReg), M)
  tl.store($(varReg), S / tl.toReal($(blockSize)))
}

/-- Shared spec: both kernels compute the two-pass mean / variance. -/
noncomputable def welfordMeanSpec {N : Nat} (xs : Fin N → ℝ) : ℝ :=
  twoPassMean xs

noncomputable def welfordVarSpec {N : Nat} (xs : Fin N → ℝ) : ℝ :=
  twoPassS xs / N

theorem twopass_welford_correct
    (xReg meanReg varReg : RegionName) (blockSize : Nat) (_hN : 0 < blockSize)
    (s : BlockState) (xs : Fin blockSize → ℝ)
    (h_x : InputLoadedAt s xReg blockSize xs)
    (h_mv : meanReg ≠ varReg) :
    let final := exec (twopassWelfordKernel xReg meanReg varReg blockSize) s
    final.bind (fun s' => some (s'.readMem meanReg 0))
        = some (welfordMeanSpec xs)
    ∧ final.bind (fun s' => some (s'.readMem varReg 0))
        = some (welfordVarSpec xs) := by
  -- Operational walk-through: simp through `exec → stepStmts → stepStmt`
  -- collapses the 10 statements to a closed form; `h_x` substitutes the
  -- loaded inputs; `h_mv` resolves the `meanReg = varReg` conditional left
  -- behind by `BlockState.writeMem`.
  simp [exec, twopassWelfordKernel, stepStmts, stepStmt, evalOp,
        Value.bop, Value.reduceSum,
        BlockState.setReg, BlockState.readMem, BlockState.writeMem,
        welfordMeanSpec, welfordVarSpec, twoPassMean, twoPassS]
  unfold InputLoadedAt at h_x
  simp_rw [h_x, if_neg h_mv]
  -- Remaining goal: variance side must rewrite `(x - μ) * (x - μ)` to
  -- `(x - μ)^2` term-wise inside the sum.
  refine ⟨trivial, ?_⟩
  congr 1
  apply Finset.sum_congr rfl
  intro i _
  ring

/-- Loop invariant for `onlineWelfordKernel`: after `k` body iterations,
    register `M` holds `welfordMean xs k`, register `S` holds `welfordS xs k`,
    register `pid` holds the original program-id (which is also `s.pid`),
    and the input region `xReg` still holds the input tile `xs`. -/
private def P_welford {N : Nat} (xs : Fin N → ℝ) (xReg : RegionName)
    (orig_pid : Nat) (k : Nat) (s : BlockState) : Prop :=
  s.regs "M" = some (Value.scalar (welfordMean xs k))
  ∧ s.regs "S" = some (Value.scalar (welfordS xs k))
  ∧ s.regs "pid" = some (Value.scalarNat orig_pid)
  ∧ s.pid = orig_pid
  ∧ InputLoadedAt s xReg N xs

/-- One Welford-loop iteration preserves the invariant: given a state at depth
    `k` with the index register `i` rebound to `Value.scalarNat k`, executing
    the body produces a state at depth `k+1`. -/
private theorem online_welford_step
    (xReg : RegionName) (blockSize : Nat) (xs : Fin blockSize → ℝ)
    (orig_pid : Nat) :
    ∀ k s, k < blockSize → P_welford xs xReg orig_pid k s →
      ∃ s',
        stepStmts
          [ .assign "xi"     (.load xReg (.add (.mul (.ref "pid")
                                                    (.constNat blockSize))
                                              (.ref "i")))
          , .assign "delta"  (.sub (.ref "xi") (.ref "M"))
          , .assign "M"      (.add (.ref "M")
                                  (.div (.ref "delta")
                                        (.add (.natToReal (.ref "i"))
                                              (.const 1))))
          , .assign "delta2" (.sub (.ref "xi") (.ref "M"))
          , .assign "S"      (.add (.ref "S")
                                  (.mul (.ref "delta") (.ref "delta2")))
          ]
          (s.setReg "i" (Value.scalarNat k)) = some s'
        ∧ P_welford xs xReg orig_pid (k + 1) s' := by
  intro k s hk hP
  obtain ⟨hM, hS, hpid_reg, hpid_eq, hmem⟩ := hP
  -- Bind shorthand for the iteration's input value.
  set xk : ℝ := xs ⟨k, hk⟩ with hxk_def
  set Mk : ℝ := welfordMean xs k with hMk_def
  set Sk : ℝ := welfordS xs k with hSk_def
  set Mk1 : ℝ := Mk + (xk - Mk) / ((k : ℝ) + 1) with hMk1_def
  -- The explicit final state: chain of 5 setRegs on `s.setReg "i" ...`.
  set s_after_i : BlockState := s.setReg "i" (Value.scalarNat k) with hsi_def
  set s1 : BlockState := s_after_i.setReg "xi" (Value.scalar xk) with hs1_def
  set s2 : BlockState := s1.setReg "delta" (Value.scalar (xk - Mk)) with hs2_def
  set s3 : BlockState := s2.setReg "M" (Value.scalar Mk1) with hs3_def
  set s4 : BlockState := s3.setReg "delta2" (Value.scalar (xk - Mk1)) with hs4_def
  set s5 : BlockState := s4.setReg "S"
            (Value.scalar (Sk + (xk - Mk) * (xk - Mk1))) with hs5_def
  -- Pre-step: the registers / memory / pid in `s_after_i`.
  have hM' : s_after_i.regs "M" = some (Value.scalar Mk) := by
    simp [hsi_def, BlockState.setReg, hM]
  have hS' : s_after_i.regs "S" = some (Value.scalar Sk) := by
    simp [hsi_def, BlockState.setReg, hS]
  have hpid' : s_after_i.regs "pid" = some (Value.scalarNat orig_pid) := by
    simp [hsi_def, BlockState.setReg, hpid_reg]
  have hi_reg : s_after_i.regs "i" = some (Value.scalarNat k) := by
    simp [hsi_def, BlockState.setReg]
  have hpid_after : s_after_i.pid = orig_pid := by
    simpa [hsi_def, BlockState.setReg] using hpid_eq
  have hload_at : s_after_i.mem xReg (orig_pid * blockSize + k) = xk := by
    have := hmem ⟨k, hk⟩
    show s.mem xReg (orig_pid * blockSize + k) = xk
    rw [← hpid_eq]
    simpa [hxk_def] using this
  -- Provide the witness `s5` and split the conjunction.
  refine ⟨s5, ?_, ?_⟩
  · -- 1. Body reduces to s5.
    -- Step 1: assign "xi" := load(...)
    have hstep1 : stepStmt
        (.assign "xi" (.load xReg (.add (.mul (.ref "pid")
                                              (.constNat blockSize))
                                        (.ref "i"))))
        s_after_i = some s1 := by
      simp [stepStmt, evalOp, hpid', hi_reg, Value.bop, hs1_def,
            BlockState.readMem, hload_at]
    -- Step 2: assign "delta" := xi - M
    have hxi1 : s1.regs "xi" = some (Value.scalar xk) := by
      simp [hs1_def, BlockState.setReg]
    have hM1 : s1.regs "M" = some (Value.scalar Mk) := by
      simp [hs1_def, BlockState.setReg, hM']
    have hstep2 : stepStmt (.assign "delta" (.sub (.ref "xi") (.ref "M"))) s1
                  = some s2 := by
      simp [stepStmt, evalOp, hxi1, hM1, Value.bop, hs2_def]
    -- Step 3: assign "M" := M + delta / (toReal(i) + 1)
    have hM2 : s2.regs "M" = some (Value.scalar Mk) := by
      simp [hs2_def, BlockState.setReg, hM1]
    have hdelta2reg : s2.regs "delta" = some (Value.scalar (xk - Mk)) := by
      simp [hs2_def, BlockState.setReg]
    have hi2 : s2.regs "i" = some (Value.scalarNat k) := by
      simp [hs2_def, hs1_def, BlockState.setReg, hi_reg]
    have hstep3 : stepStmt
        (.assign "M" (.add (.ref "M")
                          (.div (.ref "delta")
                                (.add (.natToReal (.ref "i")) (.const 1)))))
        s2 = some s3 := by
      simp [stepStmt, evalOp, hM2, hdelta2reg, hi2, Value.bop,
            hs3_def, hMk1_def]
    -- Step 4: assign "delta2" := xi - M
    have hxi3 : s3.regs "xi" = some (Value.scalar xk) := by
      simp [hs3_def, BlockState.setReg, hs2_def, hxi1]
    have hM3 : s3.regs "M" = some (Value.scalar Mk1) := by
      simp [hs3_def, BlockState.setReg]
    have hstep4 : stepStmt (.assign "delta2" (.sub (.ref "xi") (.ref "M"))) s3
                  = some s4 := by
      simp [stepStmt, evalOp, hxi3, hM3, Value.bop, hs4_def]
    -- Step 5: assign "S" := S + delta * delta2
    have hS1 : s1.regs "S" = some (Value.scalar Sk) := by
      simp [hs1_def, BlockState.setReg, hS']
    have hS2 : s2.regs "S" = some (Value.scalar Sk) := by
      simp [hs2_def, BlockState.setReg, hS1]
    have hS3 : s3.regs "S" = some (Value.scalar Sk) := by
      simp [hs3_def, BlockState.setReg, hS2]
    have hS4 : s4.regs "S" = some (Value.scalar Sk) := by
      simp [hs4_def, BlockState.setReg, hS3]
    have hdelta3 : s3.regs "delta" = some (Value.scalar (xk - Mk)) := by
      simp [hs3_def, BlockState.setReg, hdelta2reg]
    have hdelta4 : s4.regs "delta"
                    = some (Value.scalar (xk - Mk)) := by
      simp [hs4_def, BlockState.setReg, hdelta3]
    have hdelta2reg4 : s4.regs "delta2"
                        = some (Value.scalar (xk - Mk1)) := by
      simp [hs4_def, BlockState.setReg]
    have hstep5 : stepStmt
        (.assign "S" (.add (.ref "S") (.mul (.ref "delta") (.ref "delta2"))))
        s4 = some s5 := by
      simp [stepStmt, evalOp, hS4, hdelta4, hdelta2reg4, Value.bop, hs5_def]
    -- Chain the 5 steps via stepStmts.
    show stepStmts _ s_after_i = some s5
    simp [stepStmts, hstep1, hstep2, hstep3, hstep4, hstep5]
  · -- 2. P_welford xs xReg orig_pid (k+1) s5.
    -- Compute the recurrences at depth (k+1):
    have hM_rec : welfordMean xs (k + 1) = Mk1 := by
      simp [welfordMean, hk, hMk_def, hxk_def, hMk1_def]
    have hS_rec : welfordS xs (k + 1) = Sk + (xk - Mk) * (xk - Mk1) := by
      have : welfordS xs (k + 1) = Sk + (xs ⟨k, hk⟩ -
              welfordMean xs k) * (xs ⟨k, hk⟩ - welfordMean xs (k+1)) := by
        simp [welfordS, hk, hSk_def]
      rw [this, hM_rec]
    refine ⟨?_, ?_, ?_, ?_, ?_⟩
    · -- s5.regs "M" = welfordMean xs (k+1)
      simp [hs5_def, BlockState.setReg, hs4_def, BlockState.setReg, hs3_def,
            BlockState.setReg, hM_rec]
    · -- s5.regs "S" = welfordS xs (k+1)
      simp [hs5_def, BlockState.setReg, hS_rec]
    · -- s5.regs "pid" preserved.
      simp [hs5_def, hs4_def, hs3_def, hs2_def, hs1_def, BlockState.setReg,
            hpid']
    · -- s5.pid = orig_pid (setReg never touches pid).
      simpa [hs5_def, hs4_def, hs3_def, hs2_def, hs1_def, BlockState.setReg]
        using hpid_after
    · -- InputLoadedAt s5: memory unchanged across all setRegs.
      intro i
      have hi := hmem i
      have hpidS5 : s5.pid = orig_pid := by
        simpa [hs5_def, hs4_def, hs3_def, hs2_def, hs1_def, BlockState.setReg]
          using hpid_after
      show s5.mem xReg (s5.pid * blockSize + i.val) = xs i
      rw [hpidS5, ← hpid_eq]
      simpa [hs5_def, hs4_def, hs3_def, hs2_def, hs1_def, BlockState.setReg]
        using hi

theorem online_welford_correct
    (xReg meanReg varReg : RegionName) (blockSize : Nat) (hN : 0 < blockSize)
    (s : BlockState) (xs : Fin blockSize → ℝ)
    (h_x : InputLoadedAt s xReg blockSize xs)
    (h_mv : meanReg ≠ varReg) :
    let final := exec (onlineWelfordKernel xReg meanReg varReg blockSize) s
    final.bind (fun s' => some (s'.readMem meanReg 0))
        = some (welfordMeanSpec xs)
    ∧ final.bind (fun s' => some (s'.readMem varReg 0))
        = some (welfordVarSpec xs) := by
  -- The kernel is: 3 pre-loop assigns, 1 forLoop, 2 post-loop stores.
  -- Reduce the prefix manually, apply forLoop_inv, then reduce the suffix.
  -- Pre-loop state: pid := s.pid, M := 0, S := 0.
  set s0 : BlockState :=
    ((s.setReg "pid" (Value.scalarNat s.pid)).setReg "M" (Value.scalar 0)).setReg
      "S" (Value.scalar 0) with hs0_def
  -- s0 satisfies P_welford at depth 0.
  have hP0 : P_welford xs xReg s.pid 0 s0 := by
    refine ⟨?_, ?_, ?_, ?_, ?_⟩
    · simp [hs0_def, BlockState.setReg, welfordMean]
    · simp [hs0_def, BlockState.setReg, welfordS]
    · simp [hs0_def, BlockState.setReg]
    · simp [hs0_def, BlockState.setReg]
    · intro i
      have := h_x i
      simpa [hs0_def, BlockState.setReg] using this
  -- Apply forLoop_inv to obtain a final state s_final after the loop.
  obtain ⟨s_final, h_loop_eq, hPn⟩ :=
    forLoop_inv (P := P_welford xs xReg s.pid) (s_init := s0) hP0
      (fun k s' h_lt hP' =>
        online_welford_step xReg blockSize xs s.pid k s' h_lt hP')
  obtain ⟨hMfin, hSfin, _, _, _⟩ := hPn
  -- Mean / variance specs come from welford_eq_two_pass.
  obtain ⟨hMeanEq, hSEq⟩ := welford_eq_two_pass hN xs
  -- Reduce the full kernel exec by composing: prefix (3 assigns) → s0,
  -- then loop = some s_final via h_loop_eq, then 2 stores.
  -- We compute the kernel run as a chain of stepStmt's.
  -- Step 1: the kernel's 3 assigns reduce to s0.
  have h_prefix : stepStmts
        [ .assign "pid" .programId
        , .assign "M" (.const 0)
        , .assign "S" (.const 0) ]
        s = some s0 := by
    simp [stepStmts, stepStmt, evalOp, hs0_def]
  -- Step 2: the post-loop stores. After the loop ends in `s_final`, the
  -- two stores write to meanReg/varReg offset 0.
  have h_store1 : stepStmt
        (.store meanReg (.constNat 0) (.ref "M")) s_final
        = some (s_final.writeMem meanReg 0 (welfordMean xs blockSize)) := by
    simp [stepStmt, evalOp, hMfin]
  -- The mean store changes meanReg's value but leaves S register intact.
  set s_after_meanstore : BlockState :=
    s_final.writeMem meanReg 0 (welfordMean xs blockSize)
  have hS_after : s_after_meanstore.regs "S"
                  = some (Value.scalar (welfordS xs blockSize)) := by
    show s_final.regs "S" = _
    exact hSfin
  have h_store2 : stepStmt
        (.store varReg (.constNat 0)
              (.div (.ref "S") (.natToReal (.constNat blockSize))))
        s_after_meanstore
        = some (s_after_meanstore.writeMem varReg 0
                  (welfordS xs blockSize / blockSize)) := by
    simp [stepStmt, evalOp, hS_after, Value.bop]
  -- Combine: full kernel exec = prefix-assigns ++ forLoop ++ 2-stores.
  -- We use the equation `stepStmts (st :: rest) s = stepStmts rest s'` when
  -- `stepStmt st s = some s'`.
  have stmts_cons : ∀ (st : Stmt) (rest : List Stmt) (s s' : BlockState),
      stepStmt st s = some s' →
      stepStmts (st :: rest) s = stepStmts rest s' := by
    intro st rest s s' h
    conv_lhs => unfold stepStmts
    rw [h]
  have stmts_nil : ∀ (s : BlockState), stepStmts [] s = some s := by
    intro s
    conv_lhs => unfold stepStmts
  -- Statement-level reductions for the 6 body statements.
  have h_pid : stepStmt (.assign "pid" Op.programId) s
                = some (s.setReg "pid" (Value.scalarNat s.pid)) := by
    simp [stepStmt, evalOp]
  have h_M0 : stepStmt (.assign "M" (Op.const 0))
              (s.setReg "pid" (Value.scalarNat s.pid))
            = some ((s.setReg "pid" (Value.scalarNat s.pid)).setReg "M"
                    (Value.scalar 0)) := by simp [stepStmt, evalOp]
  have h_S0 : stepStmt (.assign "S" (Op.const 0))
              ((s.setReg "pid" (Value.scalarNat s.pid)).setReg "M"
                (Value.scalar 0)) = some s0 := by
    simp [stepStmt, evalOp, hs0_def]
  -- The full stepStmts evaluation, threaded through all 6 statements.
  set s_final2 : BlockState :=
    s_after_meanstore.writeMem varReg 0 (welfordS xs blockSize / blockSize)
    with hsf2_def
  have h_kernel : stepStmts
        (onlineWelfordKernel xReg meanReg varReg blockSize).body s
      = some s_final2 := by
    -- Unfold the kernel so the body is an explicit cons-list.
    show stepStmts
        [ .assign "pid" .programId
        , .assign "M" (.const 0)
        , .assign "S" (.const 0)
        , .forLoop "i" blockSize
            [ .assign "xi" (.load xReg (.add (.mul (.ref "pid")
                                                   (.constNat blockSize))
                                              (.ref "i")))
            , .assign "delta" (.sub (.ref "xi") (.ref "M"))
            , .assign "M" (.add (.ref "M")
                                (.div (.ref "delta")
                                      (.add (.natToReal (.ref "i"))
                                            (.const 1))))
            , .assign "delta2" (.sub (.ref "xi") (.ref "M"))
            , .assign "S" (.add (.ref "S")
                                (.mul (.ref "delta") (.ref "delta2")))
            ]
        , .store meanReg (.constNat 0) (.ref "M")
        , .store varReg (.constNat 0)
            (.div (.ref "S") (.natToReal (.constNat blockSize)))
        ] s = some s_final2
    rw [stmts_cons _ _ _ _ h_pid]
    rw [stmts_cons _ _ _ _ h_M0]
    rw [stmts_cons _ _ _ _ h_S0]
    rw [stmts_cons _ _ _ _ h_loop_eq]
    rw [stmts_cons _ _ _ _ h_store1]
    rw [stmts_cons _ _ _ _ h_store2]
    exact stmts_nil _
  -- Plug into the conclusion.
  simp only [exec, h_kernel, Option.bind_some, hsf2_def]
  refine ⟨?_, ?_⟩
  · -- Mean readback.
    show some (((s_final.writeMem meanReg 0 (welfordMean xs blockSize)).writeMem
                  varReg 0 (welfordS xs blockSize / blockSize)).readMem
                meanReg 0)
        = some (welfordMeanSpec xs)
    simp [BlockState.writeMem, BlockState.readMem,
          welfordMeanSpec, hMeanEq, fun h : meanReg = varReg => h_mv h]
  · -- Variance readback.
    show some (((s_final.writeMem meanReg 0 (welfordMean xs blockSize)).writeMem
                  varReg 0 (welfordS xs blockSize / blockSize)).readMem
                varReg 0)
        = some (welfordVarSpec xs)
    simp [BlockState.writeMem, BlockState.readMem,
          welfordVarSpec, hSEq]

theorem welford_kernels_refinement
    (xReg meanReg varReg : RegionName) (blockSize : Nat) (_hN : 0 < blockSize)
    (s : BlockState) (xs : Fin blockSize → ℝ)
    (_h_x : InputLoadedAt s xReg blockSize xs) :
    let final_2p := exec (twopassWelfordKernel xReg meanReg varReg blockSize) s
    let final_on := exec (onlineWelfordKernel xReg meanReg varReg blockSize) s
    final_2p.bind (fun s' => some (s'.readMem meanReg 0))
        = final_on.bind (fun s' => some (s'.readMem meanReg 0))
    ∧ final_2p.bind (fun s' => some (s'.readMem varReg 0))
        = final_on.bind (fun s' => some (s'.readMem varReg 0)) := by
  sorry

end VeriTile.Examples
