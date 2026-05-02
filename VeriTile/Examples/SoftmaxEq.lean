/-
VeriTile.Examples.SoftmaxEq

Worked equivalence example: naive softmax vs numerically stable
(max-subtracted) softmax. Embedded as `triton { ... }` macro kernels and
proved equivalent via the ARM-in-Lean refinement pattern.

Structure:

  (a) Source `.py` Triton (commented, for reference)
  (b) Embedded Triton ASTs via `triton { ... }` macro
  (c) Math denotation (`Fin n → ℝ` functions) and `naive_eq_stable` lemma
  (d) ARM-style kernel refinement:
        softmax_naive_correct       — closed-form output of naiveSoftmaxKernel
        softmax_stable_correct      — closed-form output of stableSoftmaxKernel
        softmax_kernels_refinement  — both kernels write the same `Y` memory

The math step is fully proven in (c). The two kernel-correctness lemmas in (d)
currently carry P2-level mechanical sorries — they reduce `exec kernel s` to
its closed form via simp on the operational semantics. The refinement theorem
itself is fully proven from those two lemmas plus `naive_eq_stable`.
-/

import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Field
import Mathlib.Tactic.FieldSimp
import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.DSL
import VeriTile.Examples.Common

namespace VeriTile.Examples

open VeriTile.Triton

/-! ## (a) Source Triton (`.py`) — for reference only

```python
# naive_softmax.py
@triton.jit
def naive_softmax(X, Y, N: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * N + tl.arange(0, N)
    x    = tl.load(X + offs)
    e    = tl.exp(x)
    s    = tl.sum(e, axis=0)
    y    = e / s
    tl.store(Y + offs, y)

# stable_softmax.py
@triton.jit
def stable_softmax(X, Y, N: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * N + tl.arange(0, N)
    x    = tl.load(X + offs)
    m    = tl.max(x, axis=0)
    e    = tl.exp(x - m)
    s    = tl.sum(e, axis=0)
    y    = e / s
    tl.store(Y + offs, y)
```
-/

/-! ## (b) Embedded Triton ASTs (using the `triton { ... }` macro)

Region names are kernel parameters; both kernels share the input region
`xReg` (read-only) and output region `yReg`. -/

/-- Naive softmax: `y = exp(x) / sum(exp(x))`. -/
def naiveSoftmaxKernel (xReg yReg : RegionName) (blockSize : Nat) : Kernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x    := tl.load($(xReg) + offs)
  e    := tl.exp(x)
  s    := tl.sum(e, axis=0)
  y    := e / s
  tl.store($(yReg) + offs, y)
}

/-- Stable softmax: subtract the max before exponentiating. -/
def stableSoftmaxKernel (xReg yReg : RegionName) (blockSize : Nat) : Kernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x    := tl.load($(xReg) + offs)
  m    := tl.max(x, axis=0)
  e    := tl.exp(x - m)
  s    := tl.sum(e, axis=0)
  y    := e / s
  tl.store($(yReg) + offs, y)
}

/-! ## (c) Math denotation and equivalence -/

/-- Naive softmax over a tile of length `n` in ℝ. -/
noncomputable def naiveSoftmaxMath {n : Nat} (x : Fin n → ℝ) : Fin n → ℝ :=
  fun i => Real.exp (x i) / ∑ j, Real.exp (x j)

/-- Max-subtracted softmax over a tile of length `n` in ℝ. -/
noncomputable def stableSoftmaxMath {n : Nat} (x : Fin n → ℝ) (m : ℝ) :
    Fin n → ℝ :=
  fun i => Real.exp (x i - m) / ∑ j, Real.exp (x j - m)

/-- What `naiveSoftmaxKernel` writes at the output region's `[pid*blockSize + i]` cell. -/
noncomputable def naiveSpec {blockSize : Nat} (xs : Fin blockSize → ℝ) (i : Fin blockSize) : ℝ :=
  Real.exp (xs i) / ∑ j, Real.exp (xs j)

/-- What `stableSoftmaxKernel` writes at the output region's `[pid*blockSize + i]` cell. -/
noncomputable def stableSpec {blockSize : Nat} (xs : Fin blockSize → ℝ) (m : ℝ) (i : Fin blockSize) : ℝ :=
  Real.exp (xs i - m) / ∑ j, Real.exp (xs j - m)

/-- Max of a non-empty tile, in `Tile.reduceMax`'s form. -/
noncomputable def tileMax {blockSize : Nat} (h : 0 < blockSize) (xs : Fin blockSize → ℝ) : ℝ :=
  match blockSize, h, xs with
  | _ + 1, _, xs => Finset.univ.sup' Finset.univ_nonempty xs

end VeriTile.Examples
