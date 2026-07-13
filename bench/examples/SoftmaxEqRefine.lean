import VeriTile.Triton
import VeriTile.Examples.Common
import VeriTile.Meta.StatementAudit

/-!
# Softmax: naive vs stable — rounding-invariant writes-equality

Self-contained showcase, read top to bottom: **kernels** first, the
**supporting lemma** in the middle (`private` plumbing — a bf16-scatter
congruence), the **theorem** last (one public headline
`softmax_kernels_refinement_view`), then a compile-time **trust audit**. The
three real sections are `Softmax.kernels`, `Softmax.lemmas`, `Softmax.theorems`.

Two block-parallel softmax kernels compute `y = exp(x) / Σ exp(x)` per row and
**store the result rounded to bf16** (`(y).to(tl.bfloat16)`):
`naiveSoftmaxKernel` exponentiates `x` raw; `stableSoftmaxKernel` subtracts the
row max first. The reductions run in ℝ (no intermediate rounding), so both
kernels produce the **same** per-lane ℝ value — the shift-cancellation of
softmax (`naive_eq_stable`) — and the only rounding is the shared bf16 output
store, which quantizes equal values identically.

## The public result (bottom of file)

The single public headline is **`softmax_kernels_refinement_view`** — a
kernel-vs-kernel refinement on `ComputeRefine.Refines R` (the rounding-model
surface): for the rounding model `R`, from the same state the naive and stable
kernels perform the same writes (no scratch, no input precondition). The
compositional rounding pattern is `bench/examples/FusedSwiglu.lean`.

## Exact-ℝ companions (bottom of file)

The file also carries the exact-ℝ (raw-store, no bf16 rounding) variants
`naiveSoftmaxKernelReal` / `stableSoftmaxKernelReal` together with their
ARM-in-Lean-style story: per-kernel closed forms (`softmax_naive_correct`,
`softmax_stable_correct`), the pointwise `Y`-memory refinement
(`softmax_kernels_refinement`), and its `TensorView` surface
(`softmax_kernels_refinement_exec_view`).
-/

namespace VeriTile.Bench.Examples.Softmax

open VeriTile.Triton VeriTile.Triton.TiledSoftmax

/-! ## Kernels -/
section Softmax.kernels

/-- Naive softmax: `y = exp(x) / Σ exp(x)`, exponentiated raw, stored bf16. -/
def naiveSoftmaxKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x    := tl.load($(xReg) + offs)
  e    := tl.exp(x)
  s    := tl.sum(e, axis=0)
  y    := e / s
  tl.store($(yReg) + offs, (y).to(tl.bfloat16))
}

/-- Numerically-stable softmax: subtract the row max first,
`y = exp(x − m) / Σ exp(x − m)`, stored bf16. -/
def stableSoftmaxKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x    := tl.load($(xReg) + offs)
  m    := tl.max(x, axis=0)
  e    := tl.exp(x - m)
  s    := tl.sum(e, axis=0)
  y    := e / s
  tl.store($(yReg) + offs, (y).to(tl.bfloat16))
}

/-! ### Exact-ℝ variants (raw store, no rounding)

Term-identical to the bf16 kernels above except that the final store writes the
raw ℝ value `y` instead of `(y).to(tl.bfloat16)`. These are the kernels of the
exact-ℝ companion theorems at the bottom of the file.

Source Triton (`.py`), for reference:

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

/-- Naive softmax, exact-ℝ store: `y = exp(x) / sum(exp(x))`. -/
def naiveSoftmaxKernelReal (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x    := tl.load($(xReg) + offs)
  e    := tl.exp(x)
  s    := tl.sum(e, axis=0)
  y    := e / s
  tl.store($(yReg) + offs, y)
}

/-- Stable softmax, exact-ℝ store: subtract the max before exponentiating. -/
def stableSoftmaxKernelReal (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x    := tl.load($(xReg) + offs)
  m    := tl.max(x, axis=0)
  e    := tl.exp(x - m)
  s    := tl.sum(e, axis=0)
  y    := e / s
  tl.store($(yReg) + offs, y)
}

end Softmax.kernels

/-! ## Supporting lemma (private plumbing) -/
section Softmax.lemmas

/-- Two `writeMemAsR` scatters over the same offsets agree cell-by-cell when
their per-lane values agree — the rounding-store analogue of
`BlockState.foldl_writeMem_mem_congr`. -/
private theorem foldl_writeMemAsR_mem_congr {α : Type} (R : RoundingModel)
    (dtype : FloatDType) {region : RegionName} (l : List α) (offsetFn : α → Nat)
    (vL vR : α → TileCarrier dtype.toTileDType)
    (hv : ∀ k ∈ l, vL k = vR k) (r : RegionName) (o : Nat) :
    ∀ sL sR : BlockState, sL.mem r o = sR.mem r o →
      (l.foldl (fun acc k => acc.writeMemAsR R dtype region (offsetFn k) (vL k)) sL).mem r o
        = (l.foldl (fun acc k => acc.writeMemAsR R dtype region (offsetFn k) (vR k)) sR).mem r o := by
  induction l with
  | nil => intro sL sR h; exact h
  | cons hd tl ih =>
      intro sL sR h
      refine ih (fun k hk => hv k (List.mem_cons_of_mem _ hk)) _ _ ?_
      rw [BlockState.writeMemAsR_mem, BlockState.writeMemAsR_mem,
          hv hd List.mem_cons_self]
      by_cases hc : r = region ∧ o = offsetFn hd
      · rw [if_pos hc, if_pos hc]
      · rw [if_neg hc, if_neg hc]; exact h

end Softmax.lemmas

/-! ## The headline theorem -/
section Softmax.theorems

/- Shared parameters of the headline: the two regions, the block size
(nonempty), the state, and the rounding model `R`. -/
variable (xReg yReg : RegionName) (blockSize : Nat) (hN : 0 < blockSize) (s : BlockState)
variable (R : RoundingModel)

include hN in
/-- **naive refines stable** (`ComputeRefine.Refines R`, no scratch): for the
rounding model `R`, from the same initial state the naive and stable softmax
kernels perform the same writes. Both compute the same per-lane ℝ softmax value
(shift-cancellation) and round it at the same bf16 store. -/
specification softmax_kernels_refinement_view :
    ComputeRefine.Refines R
      (naiveSoftmaxKernel xReg yReg blockSize)
      (stableSoftmaxKernel xReg yReg blockSize) s [] := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hN.ne'
  apply ComputeKernel.computeRefineR_of_toAlgKernel rfl rfl
  intro s0 lhs' rhs' hL hR hs0
  subst s0
  intro r hr o
  simp [execR, naiveSoftmaxKernel, stepStmtsR, stepStmtR, evalOpR.eq_def,
        Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul, NumericDType.div,
        ComputeExpr.toAlgorithm?] at hL
  simp [Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex] at hL
  simp [execR, stableSoftmaxKernel, stepStmtsR, stepStmtR, evalOpR.eq_def,
        Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul, NumericDType.sub,
        NumericDType.div, ComputeExpr.toAlgorithm?] at hR
  simp [Tile.reduceSumDrop, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex] at hR
  subst lhs'
  subst rhs'
  refine foldl_writeMemAsR_mem_congr R .bf16 _ _ _ _ ?_ r o _ _ rfl
  intro k _
  exact congrArg (fun v : ℝ => R.cast FloatDType.real FloatDType.bf16 (some v))
    (congrFun (naive_eq_stable
      (fun i : Fin (n + 1) => s.readMem xReg (s.pids 0 * (n + 1) + i.val)) _) k.1)

/-! ## Trust audit (compile-time gate)

If either gate fails (a smuggled axiom / `sorry`, or a foreign constant in the
trusted statement) the file stops compiling. See `VeriTile.Meta.StatementAudit`. -/

-- (1) No `sorry`, no smuggled axiom, in the public theorem's transitive proof.
#axiomsClean softmax_kernels_refinement_view

-- (2) The headline is a *kernel-vs-kernel* refinement: its statement may mention
-- ONLY the two kernels, the rounding-model surface, and the state/region types.
#stmtSurfaceSubset softmax_kernels_refinement_view ⊆
  [naiveSoftmaxKernel, stableSoftmaxKernel, ComputeRefine.Refines, RoundingModel,
   BlockState, RegionName]

end Softmax.theorems

/-! ## Exact-ℝ companion theorems

The ARM-in-Lean refinement pattern for the raw-store kernels, mirroring
arm-in-lean's `tnum_const_refinement`:

* `exec` runs the kernel,
* `readMem ... yReg addr` plays the role of `readReg ... x0`,
* the two `softmax_*_correct` lemmas play the role of `tnum_const_O*_correct`,
* the refinement theorem composes them via `naive_eq_stable`.

The reusable math denotation (`naiveSoftmaxMath`, `stableSoftmaxMath`,
`naive_eq_stable`, `naiveSpec`, `stableSpec`, `tileMax`) lives in
`VeriTile.Triton.Math.Softmax`; the observation vocabulary (`InputLoadedAt`,
`observeAt`, `programTileView`) in `VeriTile.Examples.Common`. -/
section Softmax.theoremsReal

open VeriTile.Examples

/-- **Naive softmax kernel correctness (exact ℝ).**

After running `naiveSoftmaxKernelReal blockSize` on a state with input properly
loaded in region `xReg`, the `yReg` region at observable offsets equals the
naive softmax of the input.

Proof outline: the kernel has 7 statements
(`assign × 6` then `store × 1`); each `assign` updates one register, the final
`store` does a tile-tile scatter to `yReg`. Walking through them with `simp` on
`exec / stepStmts / stepStmt / evalOp / writeMem` yields the closed form.
**Math content here: zero** — that's all isolated in `naive_eq_stable`. -/
theorem softmax_naive_correct
    (xReg yReg : RegionName)
    (blockSize : Nat) (_hN : 0 < blockSize) (s : BlockState) (xs : Fin blockSize → ℝ)
    (_h_x : InputLoadedAt s xReg blockSize xs) :
    ∀ i : Fin blockSize,
      observeAt (exec (naiveSoftmaxKernelReal xReg yReg blockSize) s) yReg blockSize s.pid i
        = some (naiveSpec xs i) := by
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [blockSize] => s.pid * blockSize + idx.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [observeAt, exec, naiveSoftmaxKernelReal, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.uop, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        NumericDType.add, NumericDType.mul, NumericDType.div, naiveSpec]
  repeat unfold evalOp
  simp [observeAt, exec, naiveSoftmaxKernelReal, stepStmts, stepStmt,
        Tile.bop, Tile.uop, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        NumericDType.add, NumericDType.mul, NumericDType.div, naiveSpec]
  rw [BlockState.scatter_readback_nd _ _ _ h_inj (i, PUnit.unit)]
  unfold InputLoadedAt at _h_x
  simp [_h_x]
  rfl

/-- **Stable softmax kernel correctness (exact ℝ).** Same scheme as above with
    the max-shift formula as the closed form. -/
theorem softmax_stable_correct
    (xReg yReg : RegionName)
    (blockSize : Nat) (hN : 0 < blockSize) (s : BlockState) (xs : Fin blockSize → ℝ)
    (_h_x : InputLoadedAt s xReg blockSize xs) :
    ∀ i : Fin blockSize,
      observeAt (exec (stableSoftmaxKernelReal xReg yReg blockSize) s) yReg blockSize s.pid i
        = some (stableSpec xs (tileMax hN xs) i) := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hN.ne'
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [n + 1] => s.pid * (n + 1) + idx.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [observeAt, exec, stableSoftmaxKernelReal, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.uop, Tile.reduceSum, Tile.reduceSumDrop,
        Tile.reduceMax, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div, stableSpec, tileMax]
  repeat unfold evalOp
  simp [observeAt, exec, stableSoftmaxKernelReal, stepStmts, stepStmt,
        Tile.bop, Tile.uop, Tile.reduceSum, Tile.reduceSumDrop,
        Tile.reduceMax, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div, stableSpec, tileMax]
  rw [BlockState.scatter_readback_nd _ _ _ h_inj (i, PUnit.unit)]
  unfold InputLoadedAt at _h_x
  simp [_h_x]
  rfl

/-- **Refinement: `naiveSoftmaxKernelReal` and `stableSoftmaxKernelReal`
    produce the same `Y`-region memory.**

This is the ARM-in-Lean-style theorem: take the optional final state from each
kernel's `exec`, read off the relevant `Y` cells, and assert pointwise equality.
The proof composes the two correctness lemmas with `naive_eq_stable`, exactly
mirroring `tnum_const_refinement`'s `obtain ... + simp + trivial`. -/
theorem softmax_kernels_refinement
    (xReg yReg : RegionName)
    (blockSize : Nat) (hN : 0 < blockSize) (s : BlockState) (xs : Fin blockSize → ℝ)
    (h_x : InputLoadedAt s xReg blockSize xs) :
    ∀ i : Fin blockSize,
      observeAt (exec (naiveSoftmaxKernelReal  xReg yReg blockSize) s) yReg blockSize s.pid i =
      observeAt (exec (stableSoftmaxKernelReal xReg yReg blockSize) s) yReg blockSize s.pid i := by
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
      TensorView.observe (exec (naiveSoftmaxKernelReal  xReg yReg blockSize) s)
          (programTileView s yReg blockSize) idx =
      TensorView.observe (exec (stableSoftmaxKernelReal xReg yReg blockSize) s)
          (programTileView s yReg blockSize) idx := by
  intro idx
  have hx := inputLoadedAt_of_programTileView_loaded (s := s) (region := xReg)
    (N := blockSize) (xs := xs) h_x
  simpa [TensorView.observe, observeTileAt, programTileView,
         TensorView.offset, Offset.strided, observeAt]
    using softmax_kernels_refinement xReg yReg blockSize hN s xs hx idx.1

/-! ### Trust audit for the exact-ℝ companions -/

#axiomsClean softmax_naive_correct
#axiomsClean softmax_stable_correct
#axiomsClean softmax_kernels_refinement
#axiomsClean softmax_kernels_refinement_exec_view

end Softmax.theoremsReal

end VeriTile.Bench.Examples.Softmax
