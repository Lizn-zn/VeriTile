/-
VeriTile.Triton.Semantics.AtomicReduction

Kernel-agnostic atomic-RMW reduction discipline.

The two `foldl_atomicAddAt_*` lemmas formalize the load-bearing fact that a
sequence of masked atomic adds — the form Triton's `tl.atomic_add` reduces
to in `stepStmt` — preserves the kernel's `pid` axes and any register that
isn't being written. Kernel proofs that contain an atomic step followed by
ordinary tail stores (e.g. FA-1's `fa1BackwardAtomicDQ` family, FA-2's
analogous backward kernels) lift their pre-atomic register hypotheses
through the atomic step using these lemmas.

`stepStmt_atomicAdd_regs` is the step-level specialization that combines
the masked-foldl with `stepStmt`'s `Stmt.atomicAdd` case analysis. It lives
in `VeriTile.Triton.Semantics.Step` because it depends on `stepStmt`
itself (which Step.lean defines), but its proof is just specialized
foldl + case analysis from this file.
-/

import VeriTile.Triton.Semantics.State

namespace VeriTile.Triton

/-- A masked atomic-add fold over a list of indices preserves
`BlockState.pid`. The `regionFn` / `offsetFn` / `valueFn` / `active`
parameters generalize over the four shapes of `Stmt.atomicAdd` (region
+ offset, ptr-tile, blockPtr-tile, with or without mask), so this single
lemma covers all atomic-add kernel patterns. -/
theorem foldl_atomicAddAt_pid {α : Type}
    (dtype : TileDType) (h : NumericDType dtype)
    (regionFn : α → RegionName) (offsetFn : α → Nat)
    (valueFn : BlockState → α → TileCarrier dtype) (active : α → Bool)
    (l : List α) (s : BlockState) :
    (l.foldl
      (fun acc i =>
        if active i then
          acc.writeMemTyped dtype (regionFn i) (offsetFn i)
            (h.add (acc.readMemValue dtype (regionFn i) (offsetFn i)) (valueFn acc i))
        else acc) s).pid = s.pid := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons]
      by_cases hactive : active hd = true
      · simp only [hactive, if_true]
        rw [ih]
        simp
      · have hfalse : active hd = false := by
          cases h' : active hd
          · rfl
          · exact False.elim (hactive h')
        simp only [hfalse]
        exact ih s

/-- A masked atomic-add fold preserves any register `(dtype', shape', name)`.
This is the load-bearing lemma that lets kernel proofs lift pre-atomic
register hypotheses through an atomic step: the atomic only writes memory,
never the registers that the subsequent ordinary stores read.

Used by `stepStmt_atomicAdd_regs` (Step.lean) and indirectly by the
gridLaunched-atomic correctness theorems for FA-1 / FA-2 backward. -/
theorem foldl_atomicAddAt_regs {α : Type}
    (dtype : TileDType) (h : NumericDType dtype)
    (regionFn : α → RegionName) (offsetFn : α → Nat)
    (valueFn : BlockState → α → TileCarrier dtype) (active : α → Bool)
    (l : List α) (s : BlockState)
    (dtype' : TileDType) (shape' : TileShape) (name : RegName) :
    (l.foldl
      (fun acc i =>
        if active i then
          acc.writeMemTyped dtype (regionFn i) (offsetFn i)
            (h.add (acc.readMemValue dtype (regionFn i) (offsetFn i)) (valueFn acc i))
        else acc) s).regs dtype' shape' name = s.regs dtype' shape' name := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons]
      by_cases hactive : active hd = true
      · simp only [hactive, if_true]
        rw [ih]
        simp
      · have hfalse : active hd = false := by
          cases h' : active hd
          · rfl
          · exact False.elim (hactive h')
        simp only [hfalse]
        exact ih s

end VeriTile.Triton
