/-
VeriTile.Triton.Core

Data types for the embedded Triton subset (Phase 1 scope).

Scope decisions for P1:
* Single-block, single program_id, deterministic execution.
* 1-D tiles only (vectors).
* Floating-point arithmetic modelled in `ℝ` (Mathlib `Real`).
* Excluded: `tl.atomic_*`, `tl.dot`, async copy, multi-block coordination,
  Hopper/Blackwell-specific ops (TMA, WGMMA).

Operational semantics live in `VeriTile.Triton.Semantics`.
-/

import Mathlib.Data.Real.Basic

namespace VeriTile.Triton

/-- Symbolic name for a memory region (input/output buffer). -/
abbrev RegionName := String

/-- Symbolic name for a register (scalar or tile variable). -/
abbrev RegName := String

/--
P1 Triton expressions.

Each constructor models one Triton expression or block-level reduction.
Statement-level constructs (assignment, control flow, memory writes) live
in `Stmt`.

Notes on individual constructors:
* `negInf` is a sentinel for `tl.full((), -inf)` used in `tl.full(... -float('inf'))`.
* `programId` returns the current `tl.program_id(axis=0)` as a scalar.
* `arange n` produces a length-`n` tile `[0, 1, ..., n-1]`.
* `broadcast e n` lifts a scalar to a length-`n` tile.
* `full n e` fills a length-`n` tile with the scalar value of `e`.
* `reduceMax`/`reduceSum` are block-level `axis=0` reductions on a tile.
* `load region offset` evaluates `offset` (must be a scalar after eval) and
  reads the corresponding cell from `region`. Tile-valued offsets (gather)
  are NOT modelled via this constructor; use a `forLoop` to fill a tile.
-/
inductive Op : Type where
  | const     : ℝ → Op
  | negInf    : Op
  | programId : Op
  | ref       : RegName → Op
  | arange    : Nat → Op
  | broadcast : Op → Nat → Op
  | full      : Nat → Op → Op
  | add       : Op → Op → Op
  | sub       : Op → Op → Op
  | mul       : Op → Op → Op
  | div       : Op → Op → Op
  | exp       : Op → Op
  | log       : Op → Op
  | max2      : Op → Op → Op
  | reduceMax : Op → Op
  | reduceSum : Op → Op
  | load      : (region : RegionName) → (offset : Op) → Op
  deriving Inhabited

/--
P1 Triton statements (mutating constructs).

* `assign name e` defines or updates the register `name` to the value of `e`.
* `store region offset value` writes `value` to `region` at `offset`. If
  `value` is a tile and `offset` is a scalar, the tile is written contiguously
  starting at `offset`.
* `forLoop i n body` runs `body` `n` times, with the scalar register `i` bound
  to the iteration index.
-/
inductive Stmt : Type where
  | assign  : RegName → Op → Stmt
  | store   : (region : RegionName) → (offset : Op) → (value : Op) → Stmt
  | forLoop : (idx : RegName) → (n : Nat) → (body : List Stmt) → Stmt
  deriving Inhabited

/--
A complete Triton kernel.

* `inputs` / `outputs` are the names of memory regions referenced by the kernel.
  These are metadata only; the operational semantics treats memory as a single
  global region map.
* `body` is the sequence of statements executed for each `program_id`.
-/
structure Kernel where
  inputs  : List RegionName
  outputs : List RegionName
  body    : List Stmt
  deriving Inhabited

end VeriTile.Triton
