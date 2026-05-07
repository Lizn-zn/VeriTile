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

import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.Math.Softmax
import VeriTile.Examples.Common

namespace VeriTile.Examples

open VeriTile.Triton VeriTile.Triton.TiledSoftmax

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
def naiveSoftmaxKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x    := tl.load($(xReg) + offs)
  e    := tl.exp(x)
  s    := tl.sum(e, axis=0)
  y    := e / s
  tl.store($(yReg) + offs, y)
}

/-- Stable softmax: subtract the max before exponentiating. -/
def stableSoftmaxKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x    := tl.load($(xReg) + offs)
  m    := tl.max(x, axis=0)
  e    := tl.exp(x - m)
  s    := tl.sum(e, axis=0)
  y    := e / s
  tl.store($(yReg) + offs, y)
}

/-! ## (c) Math denotation and equivalence

The reusable math denotation (`naiveSoftmaxMath`, `stableSoftmaxMath`,
`naive_eq_stable`, `naiveSpec`, `stableSpec`, `tileMax`) lives in
`VeriTile.Triton.Math.Softmax`. -/

/-! ## (d) kernel refinement

The refinement pattern from arm-in-lean's `tnum_const_refinement`:

```lean
theorem tnum_const_O3_correct (s : ArmState) ... :
    let sf := exec 2 tnum_const_O3 s
    readReg sf x0 = readReg s x0 ∧ readReg sf x1 = 0 ∧ ...

theorem tnum_const_refinement ... :
    let r3 := exec 2 tnum_const_O3 s3
    let r0 := exec 9 tnum_const_O0 s0
    readReg r3 x0 = readReg r0 x0 ∧ readReg r3 x1 = readReg r0 x1 := by
  obtain ⟨h3x0, h3x1, _⟩ := tnum_const_O3_correct s3 ...
  obtain ⟨h0x0, h0x1, _⟩ := tnum_const_O0_correct s0 ...
  simp only [...]
  trivial
```

We mirror it for Triton:
* `exec` runs the kernel,
* `readMem ... "Y" addr` plays the role of `readReg ... x0`,
* the two `softmax_*_correct` lemmas play the role of `tnum_const_O*_correct`,
* the refinement theorem composes them via `naive_eq_stable`.
-/

/-- **Naive softmax kernel correctness.**

After running `naiveSoftmaxKernel blockSize` on a state with input properly loaded
in region `"X"`, the `Y` region at observable offsets equals the naive softmax
of the input.

Proof outline: the kernel has 7 statements
(`assign × 6` then `store × 1`); each `assign` updates one register, the final
`store` does a tile-tile scatter to `"Y"`. Walking through them with `simp` on
`exec / stepStmts / stepStmt / evalOp / writeMem` yields the closed form.
**Math content here: zero** — that's all isolated in `naive_eq_stable`. -/
theorem softmax_naive_correct
    (xReg yReg : RegionName)
    (blockSize : Nat) (_hN : 0 < blockSize) (s : BlockState) (xs : Fin blockSize → ℝ)
    (_h_x : InputLoadedAt s xReg blockSize xs) :
    ∀ i : Fin blockSize,
      observeAt (exec (naiveSoftmaxKernel xReg yReg blockSize) s) yReg blockSize s.pid i
        = some (naiveSpec xs i) := by
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [blockSize] => s.pid * blockSize + idx.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [observeAt, exec, naiveSoftmaxKernel, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.uop, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        NumericDType.add, NumericDType.mul, NumericDType.div, naiveSpec]
  unfold InputLoadedAt at _h_x
  rw [BlockState.scatter_readback_nd _ _ _ h_inj (i, PUnit.unit)]
  simp [_h_x]
  rfl

/-- **Stable softmax kernel correctness.** Same scheme as above with the
    max-shift formula as the closed form. P2 polish. -/
theorem softmax_stable_correct
    (xReg yReg : RegionName)
    (blockSize : Nat) (hN : 0 < blockSize) (s : BlockState) (xs : Fin blockSize → ℝ)
    (_h_x : InputLoadedAt s xReg blockSize xs) :
    ∀ i : Fin blockSize,
      observeAt (exec (stableSoftmaxKernel xReg yReg blockSize) s) yReg blockSize s.pid i
        = some (stableSpec xs (tileMax hN xs) i) := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hN.ne'
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [n + 1] => s.pid * (n + 1) + idx.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [observeAt, exec, stableSoftmaxKernel, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.uop, Tile.reduceSum, Tile.reduceSumDrop,
        Tile.reduceMax, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div, stableSpec, tileMax]
  unfold InputLoadedAt at _h_x
  rw [BlockState.scatter_readback_nd _ _ _ h_inj (i, PUnit.unit)]
  simp [_h_x]
  rfl

/-- **Refinement: `naiveSoftmaxKernel` and `stableSoftmaxKernel` produce the
    same `Y`-region memory.**

This is the ARM-in-Lean-style theorem: take the optional final state from each
kernel's `exec`, read off the relevant `Y` cells, and assert pointwise equality.
The proof composes the two correctness lemmas with `naive_eq_stable`, exactly
mirroring `tnum_const_refinement`'s `obtain ... + simp + trivial`. -/
theorem softmax_kernels_refinement
    (xReg yReg : RegionName)
    (blockSize : Nat) (hN : 0 < blockSize) (s : BlockState) (xs : Fin blockSize → ℝ)
    (h_x : InputLoadedAt s xReg blockSize xs) :
    ∀ i : Fin blockSize,
      observeAt (exec (naiveSoftmaxKernel  xReg yReg blockSize) s) yReg blockSize s.pid i =
      observeAt (exec (stableSoftmaxKernel xReg yReg blockSize) s) yReg blockSize s.pid i := by
  intro i
  -- Discharge each side via its correctness lemma (the ARM-style obtain).
  rw [softmax_naive_correct  xReg yReg blockSize hN s xs h_x i,
      softmax_stable_correct xReg yReg blockSize hN s xs h_x i]
  -- Now: `some (naiveSpec xs i) = some (stableSpec xs (tileMax hN xs) i)`.
  -- Reduce to the math identity and apply `naive_eq_stable`.
  congr 1
  unfold naiveSpec stableSpec
  have h := naive_eq_stable xs (tileMax hN xs)
  exact congrFun h i

/-- View-level surface for `softmax_kernels_refinement`. -/
theorem softmax_kernels_refinement_exec_view
    (xReg yReg : RegionName)
    (blockSize : Nat) (hN : 0 < blockSize) (s : BlockState) (xs : Fin blockSize → ℝ)
    (h_x : TensorView.loaded s (programTileView s xReg blockSize)
      (fun idx : TileIndex [blockSize] => xs idx.1)) :
    ∀ idx : TileIndex [blockSize],
      TensorView.observe (exec (naiveSoftmaxKernel  xReg yReg blockSize) s)
          (programTileView s yReg blockSize) idx =
      TensorView.observe (exec (stableSoftmaxKernel xReg yReg blockSize) s)
          (programTileView s yReg blockSize) idx := by
  intro idx
  have hx := inputLoadedAt_of_programTileView_loaded (s := s) (region := xReg)
    (N := blockSize) (xs := xs) h_x
  simpa [TensorView.observe, observeTileAt, programTileView,
         TensorView.offset, Offset.strided, observeAt]
    using softmax_kernels_refinement xReg yReg blockSize hN s xs hx idx.1

/-- Compute-facing view-level surface for `softmax_kernels_refinement`. -/
theorem softmax_kernels_refinement_view
    (xReg yReg : RegionName)
    (blockSize : Nat) (hN : 0 < blockSize) (s : BlockState) (xs : Fin blockSize → ℝ)
    (h_x : TensorView.loaded s (programTileView s xReg blockSize)
      (fun idx : TileIndex [blockSize] => xs idx.1)) :
    ComputeRefine.General
      ((naiveSoftmaxKernel xReg yReg blockSize))
      ((stableSoftmaxKernel xReg yReg blockSize))
      (fun s0 lhs' rhs' =>
        s0 = s →
        ∀ idx : TileIndex [blockSize],
          TensorView.observe (some lhs')
              (programTileView s yReg blockSize) idx =
          TensorView.observe (some rhs')
              (programTileView s yReg blockSize) idx) := by
  apply ComputeKernel.computeRefine_of_toAlgKernel rfl rfl
  intro s0 lhs' rhs' hL hR hs0
  subst s0
  intro idx
  have hview := softmax_kernels_refinement_exec_view xReg yReg blockSize hN s xs h_x idx
  rw [hL, hR] at hview
  simpa using hview

end VeriTile.Examples
