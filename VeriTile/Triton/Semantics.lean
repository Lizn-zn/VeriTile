/-
VeriTile.Triton.Semantics

Operational semantics for the P1 Triton subset.

We give a small-step state-threaded interpreter:
* `evalOp : Op → BlockState → Option Value` -- pure expression evaluation.
* `stepStmt : Stmt → BlockState → Option BlockState` -- statement execution.
* `exec : Kernel → BlockState → Option BlockState` -- full kernel run.

The `Option` is for runtime errors (undefined ref, shape mismatch, etc.).

Floating-point is modelled in `ℝ`. This is a deliberate simplification;
IEEE-754 fidelity is out of scope for VeriTile (covered by differential
testing in P5+).

Several helpers below carry explicit `sorry` to indicate where future
P1 work will land. Each is annotated with `-- TODO(P1):`.
-/

import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Algebra.Order.Floor.Defs
import Mathlib.Algebra.Order.Floor.Semifield
import VeriTile.Triton.Core

namespace VeriTile.Triton

/--
Coerce a `ℝ` to `ℕ` by flooring; negative values become 0.

Used to model memory address arithmetic: Triton offsets are integer-valued at
runtime, but our `Value.scalar` packs them as `ℝ`. This helper bridges. We
expect the input to be a non-negative integer-valued `ℝ` in normal kernel
operation; the floor is a defensive default.

`noncomputable` because it depends on `Nat.floor` over `ℝ`.

TODO(P1 polish): bifurcate `Value` into `.scalarReal` / `.scalarNat` to avoid
this cast at the semantics level.
-/
noncomputable def realToNat (c : ℝ) : Nat := ⌊c⌋₊

/--
A Triton runtime value. Either a scalar `ℝ`, or a tile of length `n`
represented as `Fin n → ℝ`. Tile lengths are existential (packed in
the constructor) for simplicity; type-level shape tracking is a later
extension if needed.
-/
inductive Value : Type where
  | scalar : ℝ → Value
  | tile   : (n : Nat) → (Fin n → ℝ) → Value

instance : Inhabited Value := ⟨.scalar 0⟩

/--
A block-level execution state.

* `mem` is the memory abstraction: each region (`String`) has an
  offset-indexed (`Nat`) map to `ℝ`. Regions are disjoint by name.
* `regs` is the named register file. `none` means the register is
  not currently defined.
* `pid` is the value of `tl.program_id(0)` for this block.
-/
structure BlockState where
  mem  : RegionName → Nat → ℝ
  regs : RegName → Option Value
  pid  : Nat

instance : Inhabited BlockState :=
  ⟨{ mem  := fun _ _ => 0
   , regs := fun _ => none
   , pid  := 0 }⟩

namespace BlockState

/-- Update a single register. -/
def setReg (s : BlockState) (name : RegName) (v : Value) : BlockState :=
  { s with regs := fun n => if n = name then some v else s.regs n }

/-- Write a single scalar to memory. -/
def writeMem (s : BlockState) (region : RegionName) (offset : Nat) (v : ℝ) : BlockState :=
  { s with mem := fun r o =>
      if r = region ∧ o = offset then v else s.mem r o }

/-- Read a single scalar from memory. -/
@[inline] def readMem (s : BlockState) (region : RegionName) (offset : Nat) : ℝ :=
  s.mem region offset

/-- A `foldl` of `writeMem` writes preserves any cell whose offset is missed by
    every step. The workhorse lemma behind `scatter_readback`. -/
private theorem foldl_writeMem_preserves {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → ℝ) (o : Nat) (l : List α) :
    ∀ (s : BlockState), (∀ k ∈ l, offsetFn k ≠ o) →
      ((l.foldl (fun acc k => acc.writeMem region (offsetFn k) (valueFn k)) s).mem
        region o)
      = s.mem region o := by
  induction l with
  | nil => intros; rfl
  | cons hd tl ih =>
    intro s h
    have hhd : offsetFn hd ≠ o := h hd (List.mem_cons_self)
    have htl : ∀ k ∈ tl, offsetFn k ≠ o :=
      fun k hk => h k (List.mem_cons_of_mem hd hk)
    rw [List.foldl_cons, ih _ htl]
    show (if region = region ∧ o = offsetFn hd then valueFn hd else s.mem region o)
        = s.mem region o
    rw [if_neg]
    rintro ⟨_, h_eq⟩
    exact hhd h_eq.symm

/-- After writing `valueFn k` to address `offsetFn k` for each `k ∈ Fin n`,
    reading at `offsetFn i` returns `valueFn i`, provided the offset function
    is injective (so writes don't shadow each other). The "readback" property
    of the scatter store.

    Proof: split `List.finRange n = l₁ ++ i :: l₂` (since `i ∈ List.finRange n`).
    The head-of-tail write puts `valueFn i` at `offsetFn i`. Subsequent writes
    in `l₂` have offsets `offsetFn k` with `k ≠ i` (because `Nodup` forces
    `i ∉ l₂`), so by injectivity `offsetFn k ≠ offsetFn i`, and
    `foldl_writeMem_preserves` shows they leave the cell alone. -/
theorem scatter_readback {region : RegionName} {n : Nat}
    (s : BlockState) (offsetFn : Fin n → Nat) (valueFn : Fin n → ℝ)
    (h_inj : Function.Injective offsetFn) (i : Fin n) :
    ((List.finRange n).foldl
       (fun acc k => acc.writeMem region (offsetFn k) (valueFn k)) s).mem
        region (offsetFn i)
    = valueFn i := by
  have h_nodup : (List.finRange n).Nodup := List.nodup_finRange n
  obtain ⟨l₁, l₂, hl⟩ := List.append_of_mem (List.mem_finRange i)
  rw [hl] at h_nodup
  rw [List.nodup_append, List.nodup_cons] at h_nodup
  obtain ⟨_, ⟨hi_notin_l2, _⟩, _⟩ := h_nodup
  rw [hl, List.foldl_append, List.foldl_cons]
  have h_not_in : ∀ k ∈ l₂, offsetFn k ≠ offsetFn i := fun k hk heq => by
    have hki : k = i := h_inj heq
    subst hki
    exact hi_notin_l2 hk
  rw [foldl_writeMem_preserves offsetFn valueFn (offsetFn i) l₂ _ h_not_in]
  show (if region = region ∧ offsetFn i = offsetFn i then valueFn i else _)
      = valueFn i
  exact if_pos ⟨rfl, rfl⟩

end BlockState

namespace Value

/-- Coerce a Value to a scalar; fails (`none`) on tiles. -/
def asScalar : Value → Option ℝ
  | .scalar a => some a
  | _         => none

/-- Pointwise lift of a binary scalar op to `Value`s.
    Scalar/scalar -> scalar, scalar/tile -> tile (broadcasting), etc.
    Tile/tile of mismatched lengths -> `none`. -/
def bop (op : ℝ → ℝ → ℝ) : Value → Value → Option Value
  | .scalar a, .scalar b => some (.scalar (op a b))
  | .scalar a, .tile n f => some (.tile n (fun i => op a (f i)))
  | .tile n f, .scalar b => some (.tile n (fun i => op (f i) b))
  | .tile n f, .tile m g =>
      if h : n = m then
        some (.tile n (fun i => op (f i) (g (Fin.cast h i))))
      else none

/-- Pointwise lift of a unary scalar op to a `Value`. -/
def uop (op : ℝ → ℝ) : Value → Value
  | .scalar a => .scalar (op a)
  | .tile n f => .tile n (fun i => op (f i))

/-- Reduce a tile to a scalar with a binary op and identity element.
    Reducing a `scalar` is undefined (returns `none`). Used for legacy /
    diagnostic purposes; production reductions go through `reduceSum` and
    `reduceMax` below, which use Mathlib `Finset` forms directly so that
    proofs about them connect to standard math lemmas. -/
def reduce (op : ℝ → ℝ → ℝ) (e : ℝ) : Value → Option Value
  | .scalar _   => none
  | .tile n f =>
      let acc := (List.finRange n).foldl (fun a i => op a (f i)) e
      some (.scalar acc)

/-- `tl.sum(x, axis=0)` semantics: sum over the tile via Mathlib `Finset.sum`. -/
noncomputable def reduceSum : Value → Option Value
  | .scalar _ => none
  | .tile _ f => some (.scalar (∑ i, f i))

/-- `tl.max(x, axis=0)` semantics: max over a non-empty tile via `Finset.sup'`.
    Returns `none` on empty tiles (no well-defined max). -/
noncomputable def reduceMax : Value → Option Value
  | .scalar _ => none
  | .tile 0 _ => none
  | .tile (n+1) f =>
      some (.scalar ((Finset.univ : Finset (Fin (n+1))).sup' Finset.univ_nonempty f))

end Value

/--
Evaluate an `Op` against a `BlockState`.

Returns `none` on type / shape errors (e.g. dividing tiles of different
lengths, dereferencing an undefined register). Returns `some v` otherwise.

`noncomputable` because `Real` arithmetic (`/`, `Real.exp`) is noncomputable.
Lean accepts this as structural recursion on `Op`.
-/
noncomputable def evalOp : Op → BlockState → Option Value
  | .const c, _      => some (.scalar c)
  | .negInf, _       =>
      -- TODO(P1): replace with a proper `⊥`/`-∞` sentinel that interacts
      -- correctly with `max`. -1e38 is a finite stand-in.
      some (.scalar (-1e38))
  | .programId, s    => some (.scalar (s.pid : ℝ))
  | .ref name, s     => s.regs name
  | .arange n, _     => some (.tile n (fun i => (i.val : ℝ)))
  | .broadcast e n, s =>
      match evalOp e s with
      | some v => match v.asScalar with
                  | some c => some (.tile n (fun _ => c))
                  | none   => none
      | none => none
  | .full n e, s =>
      match evalOp e s with
      | some v => match v.asScalar with
                  | some c => some (.tile n (fun _ => c))
                  | none   => none
      | none => none
  | .add a b, s =>
      match evalOp a s, evalOp b s with
      | some va, some vb => va.bop (· + ·) vb
      | _, _ => none
  | .sub a b, s =>
      match evalOp a s, evalOp b s with
      | some va, some vb => va.bop (· - ·) vb
      | _, _ => none
  | .mul a b, s =>
      match evalOp a s, evalOp b s with
      | some va, some vb => va.bop (· * ·) vb
      | _, _ => none
  | .div a b, s =>
      match evalOp a s, evalOp b s with
      | some va, some vb => va.bop (· / ·) vb
      | _, _ => none
  | .exp a, s =>
      match evalOp a s with
      | some va => some (va.uop Real.exp)
      | none => none
  | .log a, s =>
      match evalOp a s with
      | some va => some (va.uop Real.log)
      | none => none
  | .max2 a b, s =>
      match evalOp a s, evalOp b s with
      | some va, some vb => va.bop max vb
      | _, _ => none
  | .reduceMax a, s =>
      match evalOp a s with
      | some va => va.reduceMax
      | none => none
  | .reduceSum a, s =>
      match evalOp a s with
      | some va => va.reduceSum
      | none => none
  | .load region off, s =>
      match evalOp off s with
      | some (.scalar c) =>
          -- Scalar offset: single-cell read.
          some (.scalar (s.readMem region (realToNat c)))
      | some (.tile n f) =>
          -- Tile-valued offset: gather. Each output cell `i` reads the cell
          -- at memory address `realToNat (f i)`.
          some (.tile n (fun i => s.readMem region (realToNat (f i))))
      | none => none

-- Execute one statement.
--
-- `assign` evaluates RHS and stores the value in the named register.
-- `store` writes a scalar or contiguous tile to memory.
-- `forLoop` is a known gap in P1: bounded-loop operational semantics requires
-- mutual recursion with `stepStmts` and a non-trivial termination measure
-- combining list size and the loop counter. We leave it as `sorry` for P1
-- skeleton and complete it as the first piece of P2 work.
noncomputable def stepStmt : Stmt → BlockState → Option BlockState
  | .assign name e, s =>
      match evalOp e s with
      | some v => some (s.setReg name v)
      | none   => none
  | .store region off val, s =>
      match evalOp off s, evalOp val s with
      | some voff, some vval =>
          match voff with
          | .scalar coff =>
              -- Scalar offset: contiguous store starting at `coff`.
              match vval with
              | .scalar c =>
                  some (s.writeMem region (realToNat coff) c)
              | .tile n f =>
                  some ((List.finRange n).foldl
                          (fun acc i =>
                            acc.writeMem region (realToNat coff + i.val) (f i))
                          s)
          | .tile n offs =>
              -- Tile-valued offset: scatter. Iteration `i` writes `vals i` to
              -- address `realToNat (offs i)`. Requires the value tile to have
              -- the same length as the offset tile.
              match vval with
              | .scalar _ => none
              | .tile m vals =>
                  if h : n = m then
                    some ((List.finRange n).foldl
                            (fun acc i =>
                              acc.writeMem region (realToNat (offs i))
                                          (vals (Fin.cast h i)))
                            s)
                  else none
      | _, _ => none
  | .forLoop _idx _n _body, _s =>
      -- TODO(P2): bounded-loop operational semantics. Plan: turn `Stmt` into
      -- a `mutual` block with `stepStmts : List Stmt → ...` and
      -- `stepForLoopAux : ... → Nat → ... → Option BlockState`, with explicit
      -- lex `termination_by` measure `(sizeOf body + 1, n - i)`. See
      -- Notes/MacroOptions.md for the worked-out measure.
      -- For now we return `none` ("this case is not supported yet") rather
      -- than `sorry` so the operational semantics remains a total function
      -- and downstream proofs about non-loop kernels do not transitively
      -- inherit a `sorry` dependency.
      none

/-- Sequence a list of statements, threading state. Structural on `List`. -/
noncomputable def stepStmts : List Stmt → BlockState → Option BlockState
  | [], s => some s
  | st :: rest, s =>
      match stepStmt st s with
      | some s' => stepStmts rest s'
      | none    => none

/-- Execute a kernel from an initial state to a final state (or `none` on error). -/
noncomputable def exec (k : Kernel) (s : BlockState) : Option BlockState :=
  stepStmts k.body s

-- ────────────── Sanity checks ──────────────
-- These are intentionally tiny: they confirm the framework is alive
-- (definitions reduce, simp/norm_num apply, register-binding works)
-- before we attempt anything substantive in P2.

-- Pure structural reduction: a constant op evaluates to its constant.
example : evalOp (.const 5) default = some (Value.scalar 5) := by
  unfold evalOp; rfl

-- `programId` reads `BlockState.pid`; default state has `pid = 0`.
example : evalOp .programId default = some (Value.scalar 0) := by
  unfold evalOp
  show some (Value.scalar ((0 : Nat) : ℝ)) = some (Value.scalar 0)
  norm_num

-- Trivial constant arithmetic: `(1 + 2 : ℝ) = 3` lifts through `evalOp`.
example : evalOp (.add (.const 1) (.const 2)) default = some (Value.scalar 3) := by
  show some (Value.scalar ((1 : ℝ) + 2)) = some (Value.scalar 3)
  norm_num

-- `assign` writes to the register file.
example (s : BlockState) :
    stepStmt (.assign "x" (.const 7)) s
      = some (s.setReg "x" (Value.scalar 7)) := by
  unfold stepStmt evalOp; rfl

end VeriTile.Triton
