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
import Mathlib.Analysis.SpecialFunctions.Sigmoid
import VeriTile.Triton.Core

namespace VeriTile.Triton

/--
A Triton runtime value, bifurcated into `ℝ` (data) and `Nat` (address)
channels. Tile lengths are existential (packed in the constructor) for
simplicity; type-level shape tracking is a later extension if needed.

* `scalar`     — `ℝ` data scalar (e.g. an `exp`/`div` result, an accumulator).
* `scalarNat`  — `Nat` address scalar (e.g. `pid`, a tile size, a stride).
* `tile`       — `ℝ` data tile (e.g. `tl.exp(x)`, `tl.load(...)` from a data buffer).
* `tileNat`    — `Nat` offset tile (e.g. `tl.arange(N)`, computed offsets).

The bifurcation eliminates the prior `realToNat` round-trip cast and
removes one class of permissive typing (`exp(pid)` no longer evaluates).
See RP2 (`Notes/research_problem_address_typing.md`).
-/
inductive Value : Type where
  | scalar    : ℝ   → Value
  | scalarNat : Nat → Value
  | tile      : (n : Nat) → (Fin n → ℝ)   → Value
  | tileNat   : (n : Nat) → (Fin n → Nat) → Value

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

/-- Coerce a Value to an `ℝ` scalar; fails (`none`) on tiles or `Nat` values. -/
def asScalar : Value → Option ℝ
  | .scalar a => some a
  | _         => none

/-- Coerce a Value to a `Nat` scalar; fails (`none`) on tiles or `ℝ` values. -/
def asScalarNat : Value → Option Nat
  | .scalarNat n => some n
  | _            => none

/-- Pointwise lift of a binary scalar op, dispatched by carrier type.

    * Two `ℝ` operands (any combination of `scalar`/`tile`) → `opR`
    * Two `Nat` operands (any combination of `scalarNat`/`tileNat`) → `opN`
    * Mixed `ℝ`/`Nat` operands → `none` (semantic error)

    Tile/tile of mismatched lengths → `none`. -/
def bop (opR : ℝ → ℝ → ℝ) (opN : Nat → Nat → Nat) :
    Value → Value → Option Value
  -- ℝ × ℝ
  | .scalar a, .scalar b      => some (.scalar (opR a b))
  | .scalar a, .tile n f      => some (.tile n (fun i => opR a (f i)))
  | .tile n f, .scalar b      => some (.tile n (fun i => opR (f i) b))
  | .tile n f, .tile m g      =>
      if h : n = m then
        some (.tile n (fun i => opR (f i) (g (Fin.cast h i))))
      else none
  -- Nat × Nat
  | .scalarNat a, .scalarNat b   => some (.scalarNat (opN a b))
  | .scalarNat a, .tileNat n f   => some (.tileNat n (fun i => opN a (f i)))
  | .tileNat n f, .scalarNat b   => some (.tileNat n (fun i => opN (f i) b))
  | .tileNat n f, .tileNat m g   =>
      if h : n = m then
        some (.tileNat n (fun i => opN (f i) (g (Fin.cast h i))))
      else none
  -- mixed ℝ / Nat → semantic error
  | _, _ => none

/-- Pointwise lift of a unary `ℝ` op (e.g. `Real.exp`, `Real.log`) to a
    `Value`. Returns `none` on `Nat` carriers — applying `exp`/`log` to an
    address value is rejected at the semantics layer. -/
def uop (op : ℝ → ℝ) : Value → Option Value
  | .scalar a   => some (.scalar (op a))
  | .tile n f   => some (.tile n (fun i => op (f i)))
  | _           => none

/-- `tl.sum(x, axis=0)` semantics: sum over the tile via Mathlib `Finset.sum`.
    Defined only on `ℝ` tiles; `Nat` tiles or scalars return `none`. -/
noncomputable def reduceSum : Value → Option Value
  | .tile _ f => some (.scalar (∑ i, f i))
  | _         => none

/-- `tl.max(x, axis=0)` semantics: max over a non-empty `ℝ` tile via
    `Finset.sup'`. Empty tiles, `Nat` tiles, and scalars all return `none`. -/
noncomputable def reduceMax : Value → Option Value
  | .tile (n+1) f =>
      some (.scalar ((Finset.univ : Finset (Fin (n+1))).sup' Finset.univ_nonempty f))
  | _ => none

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
  | .constNat n, _   => some (.scalarNat n)
  | .negInf, _       =>
      -- TODO(P1): replace with a proper `⊥`/`-∞` sentinel that interacts
      -- correctly with `max`. -1e38 is a finite stand-in.
      some (.scalar (-1e38))
  | .programId, s    => some (.scalarNat s.pid)
  | .ref name, s     => s.regs name
  | .arange n, _     => some (.tileNat n (fun i => i.val))
  | .broadcast e n, s =>
      match evalOp e s with
      | some (.scalar c)    => some (.tile n (fun _ => c))
      | some (.scalarNat c) => some (.tileNat n (fun _ => c))
      | _ => none
  | .full n e, s =>
      match evalOp e s with
      | some (.scalar c)    => some (.tile n (fun _ => c))
      | some (.scalarNat c) => some (.tileNat n (fun _ => c))
      | _ => none
  | .add a b, s =>
      match evalOp a s, evalOp b s with
      | some va, some vb => va.bop (· + ·) (· + ·) vb
      | _, _ => none
  | .sub a b, s =>
      match evalOp a s, evalOp b s with
      | some va, some vb => va.bop (· - ·) (· - ·) vb
      | _, _ => none
  | .mul a b, s =>
      match evalOp a s, evalOp b s with
      | some va, some vb => va.bop (· * ·) (· * ·) vb
      | _, _ => none
  | .div a b, s =>
      match evalOp a s, evalOp b s with
      | some va, some vb => va.bop (· / ·) (· / ·) vb
      | _, _ => none
  | .exp a, s =>
      match evalOp a s with
      | some va => va.uop Real.exp
      | none => none
  | .log a, s =>
      match evalOp a s with
      | some va => va.uop Real.log
      | none => none
  | .sigmoid a, s =>
      match evalOp a s with
      | some va => va.uop Real.sigmoid
      | none => none
  | .max2 a b, s =>
      match evalOp a s, evalOp b s with
      | some va, some vb => va.bop max max vb
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
      | some (.scalarNat n) =>
          -- Scalar offset: single-cell read at the given Nat address.
          some (.scalar (s.readMem region n))
      | some (.tileNat n f) =>
          -- Tile-valued offset: gather. Each output cell `i` reads the cell
          -- at memory address `f i`.
          some (.tile n (fun i => s.readMem region (f i)))
      | _ => none

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
          | .scalarNat coff =>
              -- Scalar offset (Nat): contiguous store starting at `coff`.
              match vval with
              | .scalar c =>
                  some (s.writeMem region coff c)
              | .tile n f =>
                  some ((List.finRange n).foldl
                          (fun acc i =>
                            acc.writeMem region (coff + i.val) (f i))
                          s)
              | _ => none
          | .tileNat n offs =>
              -- Tile-valued offset (Nat): scatter. Iteration `i` writes
              -- `vals i` to address `offs i`. Requires the value tile to
              -- have the same length as the offset tile.
              match vval with
              | .tile m vals =>
                  if h : n = m then
                    some ((List.finRange n).foldl
                            (fun acc i =>
                              acc.writeMem region (offs i)
                                          (vals (Fin.cast h i)))
                            s)
                  else none
              | _ => none
          | _ => none  -- ℝ-valued offset is rejected
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

-- Pure structural reduction: an `ℝ` constant evaluates to a `Value.scalar`.
example : evalOp (.const 5) default = some (Value.scalar 5) := by
  unfold evalOp; rfl

-- A `Nat` constant evaluates to a `Value.scalarNat`.
example : evalOp (.constNat 7) default = some (Value.scalarNat 7) := by
  unfold evalOp; rfl

-- `programId` reads `BlockState.pid` as a `Nat`; default state has `pid = 0`.
example : evalOp .programId default = some (Value.scalarNat 0) := by
  unfold evalOp; rfl

-- Trivial `ℝ` arithmetic: `(1 + 2 : ℝ) = 3` lifts through `evalOp`.
example : evalOp (.add (.const 1) (.const 2)) default = some (Value.scalar 3) := by
  show some (Value.scalar ((1 : ℝ) + 2)) = some (Value.scalar 3)
  norm_num

-- Trivial `Nat` address arithmetic: `(2 + 3 : Nat) = 5` stays `Nat`.
example : evalOp (.add (.constNat 2) (.constNat 3)) default
            = some (Value.scalarNat 5) := by
  unfold evalOp Value.bop; rfl

-- Mixing ℝ and Nat in arithmetic is a semantic error.
example : evalOp (.add (.const 1) (.constNat 2)) default = none := by
  unfold evalOp Value.bop; rfl

-- `exp` on a `Nat` value is rejected.
example : evalOp (.exp .programId) default = none := by
  unfold evalOp Value.uop; rfl

-- `assign` writes to the register file.
example (s : BlockState) :
    stepStmt (.assign "x" (.const 7)) s
      = some (s.setReg "x" (Value.scalar 7)) := by
  unfold stepStmt evalOp; rfl

end VeriTile.Triton
