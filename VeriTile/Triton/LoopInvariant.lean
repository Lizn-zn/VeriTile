/-
VeriTile.Triton.LoopInvariant

The forLoop induction lemma family. Master lemma `forLoop_inv` (spec §4.1)
plus ergonomics corollaries `forLoop_readout_scalar` (§4.2) and
`forLoop_readout_tile` (§4.3). See Notes/2026-04-29-forloop-inv-design.md
for the full interface design.
-/

import VeriTile.Triton.Core
import VeriTile.Triton.Semantics

namespace VeriTile.Triton

/-- Auxiliary form of `forLoop_inv` quantified over the starting index `i` and
    state `s`. The master `forLoop_inv` is the `i = 0` instance. -/
theorem forLoopAux_inv
    {idx : RegName} {n : Nat} {body : List Stmt}
    {P : Nat → BlockState → Prop}
    (h_step :
      ∀ i s, i < n → P i s →
        ∃ s',
          stepStmts body (s.setReg idx .nat [] (Tile.scalar i)) = some s' ∧
          P (i + 1) s') :
    ∀ i, i ≤ n → ∀ s, P i s →
      ∃ s_final,
        stepForLoopAux idx i n body s = some s_final ∧ P n s_final := by
  -- Induction on `n - i`. Equivalent to `Nat.le_induction` from `i = n`
  -- downward, but easier to phrase here.
  intro i hi s hPs
  -- Pull out the measure once.
  have key : ∀ k i, n - i = k → i ≤ n → ∀ s, P i s →
      ∃ s_final, stepForLoopAux idx i n body s = some s_final ∧ P n s_final := by
    intro k
    induction k with
    | zero =>
        -- n - i = 0 means i ≥ n; combined with i ≤ n we get i = n.
        intro i heq hi_le s hP
        have hi_eq : i = n := by omega
        subst hi_eq
        refine ⟨s, ?_, hP⟩
        exact stepForLoopAux.step_eq_self s
    | succ k ih =>
        -- n - i = k+1 ⇒ i < n.
        intro i heq hi_le s hP
        have hi_lt : i < n := by omega
        obtain ⟨s', h_body, hP'⟩ := h_step i s hi_lt hP
        have heq' : n - (i + 1) = k := by omega
        have hi'_le : i + 1 ≤ n := hi_lt
        obtain ⟨s_final, h_aux, hP_n⟩ := ih (i + 1) heq' hi'_le s' hP'
        refine ⟨s_final, ?_, hP_n⟩
        rw [stepForLoopAux.step_lt hi_lt]
        rw [h_body]
        simpa using h_aux
  exact key (n - i) i rfl hi s hPs
/-- **Master loop-induction lemma.** For any `forLoop` of body length `n`
    over register `idx`, given an entry invariant `P 0 s_init` and a step
    obligation showing each iteration preserves `P` (and does not error),
    the final state satisfies `P n`.

    Spec: `Notes/2026-04-29-forloop-inv-design.md` §4.1 (Form 1). -/
theorem forLoop_inv
    {idx : RegName} {n : Nat} {body : List Stmt}
    {P : Nat → BlockState → Prop} {s_init : BlockState}
    (h_init : P 0 s_init)
    (h_step :
      ∀ i s, i < n → P i s →
        ∃ s',
          stepStmts body (s.setReg idx .nat [] (Tile.scalar i)) = some s' ∧
          P (i + 1) s') :
    ∃ s_final,
      stepStmt (.forLoop idx n body) s_init = some s_final ∧
      P n s_final := by
  obtain ⟨s_final, h_aux, hP⟩ :=
    forLoopAux_inv (P := P) h_step 0 (Nat.zero_le _) s_init h_init
  refine ⟨s_final, ?_, hP⟩
  -- stepStmt (forLoop ...) = stepForLoopAux ... 0 n body s_init by definition.
  simpa [stepForLoopAux.forLoop_unfold] using h_aux

/-- **Scalar-register readout corollary.** Combines `forLoop_inv` with a
    proof that some output register holds a target scalar value when `P n`
    holds; the conclusion is the standard "kernel correctness reads
    register `outReg` and finds `f n`" form used throughout Tier 2 / 3-A. -/
theorem forLoop_readout_scalar
    {idx outReg : RegName} {n : Nat} {body : List Stmt}
    {P : Nat → BlockState → Prop} {s_init : BlockState}
    {f : Nat → ℝ}
    (h_init : P 0 s_init)
    (h_step :
      ∀ i s, i < n → P i s →
        ∃ s',
          stepStmts body (s.setReg idx .nat [] (Tile.scalar i)) = some s' ∧
          P (i + 1) s')
    (h_readout :
      ∀ s, P n s → s.regs .real [] outReg = some (Tile.scalar (f n))) :
    ∃ s_final,
      stepStmt (.forLoop idx n body) s_init = some s_final ∧
      s_final.regs .real [] outReg = some (Tile.scalar (f n)) := by
  obtain ⟨s_final, h_eq, hP⟩ := forLoop_inv h_init h_step
  exact ⟨s_final, h_eq, h_readout _ hP⟩

/-- **Tile-register readout corollary.** Same as `forLoop_readout_scalar` but
    for an output register containing a real vector tile; used by FA-1
    forward (`O` register) and any kernel writing a tile-valued accumulator.

    The readout is in `WithBot ℝ` to match `TileCarrier .real`. Callers whose
    `f` is `ℝ`-valued can compose with `some` at the call site. -/
theorem forLoop_readout_tile
    {idx outReg : RegName} {n : Nat} {body : List Stmt}
    {P : Nat → BlockState → Prop} {s_init : BlockState}
    {len : Nat} {f : Nat → Fin len → WithBot ℝ}
    (h_init : P 0 s_init)
    (h_step :
      ∀ i s, i < n → P i s →
        ∃ s',
          stepStmts body (s.setReg idx .nat [] (Tile.scalar i)) = some s' ∧
          P (i + 1) s')
    (h_readout :
      ∀ s, P n s → s.regs .real [len] outReg = some (Tile.vec (f n))) :
    ∃ s_final,
      stepStmt (.forLoop idx n body) s_init = some s_final ∧
      s_final.regs .real [len] outReg = some (Tile.vec (f n)) := by
  obtain ⟨s_final, h_eq, hP⟩ := forLoop_inv h_init h_step
  exact ⟨s_final, h_eq, h_readout _ hP⟩
/-! ### Sanity check

A trivial 1-statement-body forLoop counter. We use `Nat`-channel arithmetic
because the loop index `idx` is a typed Nat scalar. -/

private def counterBody : List Stmt :=
  [.assign .nat [] "cnt"
    (.add NumericDType.nat Broadcast.nil
      (.ref .nat [] "cnt") (.constNat 1))]

private theorem counter_invariant_step
    (i : Nat) (s : BlockState)
    (_h_lt : i < 5)
    (hP : s.regs .nat [] "cnt" = some (Tile.scalar i)) :
    ∃ s', stepStmts counterBody (s.setReg "i" .nat [] (Tile.scalar i)) = some s'
        ∧ s'.regs .nat [] "cnt" = some (Tile.scalar (i + 1)) := by
  -- Body is a single assign; step it manually. The resulting state writes
  -- "cnt" := Nat scalar (i + 1) on top of `s.setReg "i" ...`.
  refine ⟨((s.setReg "i" .nat [] (Tile.scalar i)).setReg "cnt" .nat []
      (Tile.bop (NumericDType.add NumericDType.nat) Broadcast.nil
        (Tile.scalar i) (Tile.scalar 1))), ?_, ?_⟩
  · simp [stepStmts, stepStmt, evalOp, counterBody, BlockState.setReg, hP,
          Tile.bop, NumericDType.add]
  · rw [BlockState.setReg_same]
    rfl

example
    (s_init : BlockState)
    (h_cnt0 : s_init.regs .nat [] "cnt" = some (Tile.scalar 0)) :
    ∃ s_final,
      stepStmt (.forLoop "i" 5 counterBody) s_init = some s_final ∧
      s_final.regs .nat [] "cnt" = some (Tile.scalar 5) := by
  -- Note: the `i` register is unset in s_init; that's fine — `setReg "i"`
  -- in stepForLoopAux defines it on first use. The invariant talks about
  -- "cnt" only.
  have h_init : s_init.regs .nat [] "cnt" = some (Tile.scalar 0) := h_cnt0
  obtain ⟨s_final, h_eq, hP⟩ :=
    forLoop_inv (P := fun k s => s.regs .nat [] "cnt" = some (Tile.scalar k))
      h_init
      (fun i s h_lt hP => counter_invariant_step i s h_lt hP)
  exact ⟨s_final, h_eq, hP⟩

/-! ### Nested-loop sanity check

This toy kernel checks that `forLoop_inv` composes when an outer loop body
contains another `Stmt.forLoop`. The inner loop increments a Nat counter `N`
times; the outer loop runs that inner loop `M` times. The final counter is
therefore `M * N`.
-/

private def nestedCounterInnerBody : List Stmt :=
  [.assign .nat [] "cnt"
    (.add NumericDType.nat Broadcast.nil
      (.ref .nat [] "cnt") (.constNat 1))]

private def nestedCounterOuterBody (N : Nat) : List Stmt :=
  [.forLoop "j" N nestedCounterInnerBody]

private def nestedCounterKernel (M N : Nat) : Kernel :=
  { inputs := []
  , outputs := []
  , body :=
      [.assign .nat [] "cnt" (.constNat 0),
       .forLoop "i" M (nestedCounterOuterBody N)] }

private theorem nestedCounter_inner_step
    (base j : Nat) (s : BlockState)
    (_h_lt : j < N)
    (hP : s.regs .nat [] "cnt" = some (Tile.scalar (base + j))) :
    ∃ s',
      stepStmts nestedCounterInnerBody (s.setReg "j" .nat [] (Tile.scalar j)) =
        some s' ∧
      s'.regs .nat [] "cnt" = some (Tile.scalar (base + (j + 1))) := by
  let sj := s.setReg "j" .nat [] (Tile.scalar j)
  let s' := sj.setReg "cnt" .nat []
    (Tile.bop (NumericDType.add NumericDType.nat) Broadcast.nil
      (Tile.scalar (base + j)) (Tile.scalar 1))
  refine ⟨s', ?_, ?_⟩
  · simp [nestedCounterInnerBody, stepStmts, stepStmt, evalOp, sj, s',
      BlockState.setReg, hP, Tile.bop, NumericDType.add]
  · rw [BlockState.setReg_same]
    simp [Tile.bop, NumericDType.add]
    rw [Nat.add_assoc]

private theorem nestedCounter_inner_correct
    (N base : Nat) (s : BlockState)
    (hP : s.regs .nat [] "cnt" = some (Tile.scalar base)) :
    ∃ s',
      stepStmt (.forLoop "j" N nestedCounterInnerBody) s = some s' ∧
      s'.regs .nat [] "cnt" = some (Tile.scalar (base + N)) := by
  have hInit :
      s.regs .nat [] "cnt" = some (Tile.scalar (base + 0)) := by
    simpa using hP
  obtain ⟨s', hStep, hFinal⟩ :=
    forLoop_inv
      (P := fun j st =>
        st.regs .nat [] "cnt" = some (Tile.scalar (base + j)))
      hInit
      (fun j st hlt hPj => nestedCounter_inner_step (N := N) base j st hlt hPj)
  exact ⟨s', hStep, hFinal⟩

private theorem nestedCounter_outer_step
    (M N i : Nat) (s : BlockState)
    (_h_lt : i < M)
    (hP : s.regs .nat [] "cnt" = some (Tile.scalar (i * N))) :
    ∃ s',
      stepStmts (nestedCounterOuterBody N) (s.setReg "i" .nat [] (Tile.scalar i)) =
        some s' ∧
      s'.regs .nat [] "cnt" = some (Tile.scalar ((i + 1) * N)) := by
  obtain ⟨s', hInner, hCnt⟩ :=
    nestedCounter_inner_correct N (i * N) (s.setReg "i" .nat [] (Tile.scalar i))
      (by simpa [BlockState.setReg_ne_name, show "cnt" ≠ "i" by decide] using hP)
  refine ⟨s', ?_, ?_⟩
  · simp [nestedCounterOuterBody, stepStmts, hInner]
  · simpa [Nat.succ_mul] using hCnt

theorem nestedCounter_correct
    (M N : Nat) (s_init : BlockState)
    (h_cnt0 : s_init.regs .nat [] "cnt" = some (Tile.scalar 0)) :
    ∃ s_final,
      stepStmt (.forLoop "i" M (nestedCounterOuterBody N)) s_init = some s_final ∧
      s_final.regs .nat [] "cnt" = some (Tile.scalar (M * N)) := by
  have hInit :
      s_init.regs .nat [] "cnt" = some (Tile.scalar (0 * N)) := by
    simpa using h_cnt0
  obtain ⟨s_final, hLoop, hFinal⟩ :=
    forLoop_inv
      (P := fun i st =>
        st.regs .nat [] "cnt" = some (Tile.scalar (i * N)))
      hInit
      (fun i st hlt hP => nestedCounter_outer_step M N i st hlt hP)
  exact ⟨s_final, hLoop, hFinal⟩

theorem nestedCounterKernel_correct
    (M N : Nat) (s_init : BlockState) :
    ∃ s_final,
      exec (nestedCounterKernel M N) s_init = some s_final ∧
      s_final.regs .nat [] "cnt" = some (Tile.scalar (M * N)) := by
  let s0 := s_init.setReg "cnt" .nat [] (Tile.scalar 0)
  have hInit : s0.regs .nat [] "cnt" = some (Tile.scalar 0) := by
    rw [BlockState.setReg_same]
  obtain ⟨s_final, hLoop, hFinal⟩ :=
    nestedCounter_correct M N s0 hInit
  refine ⟨s_final, ?_, hFinal⟩
  simp [nestedCounterKernel, exec, stepStmts, stepStmt, evalOp, s0, hLoop]

end VeriTile.Triton
