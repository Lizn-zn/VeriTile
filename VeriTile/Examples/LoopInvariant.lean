/-
VeriTile.Examples.LoopInvariant

Small sanity proofs for the Triton `forLoop_inv` API.
-/

import VeriTile.Triton.LoopInvariant

namespace VeriTile.Examples.LoopInvariant

open VeriTile.Triton

/-! ### Single-loop sanity check

A trivial 1-statement-body forLoop counter. We use `Nat`-channel arithmetic
because the loop index is a typed Nat scalar.
-/

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

end VeriTile.Examples.LoopInvariant
