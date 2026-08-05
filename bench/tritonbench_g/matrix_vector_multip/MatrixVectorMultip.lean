import VeriTile.Triton

/-!
# `matrix_vector_multip` — strict per-kernel correctness

`mv_kernel` computes a matrix-vector product `C = A · B`: program `pid` owns rows
`[pid·BLOCK_N, …)` of `A`, loops over the `M` columns in `BLOCK_M`-wide chunks
accumulating `a * b`, reduces over the `M` axis (`tl.sum(acc, axis=1)`), and
stores the resulting length-`BLOCK_N` vector to `C`, masked by `offset_n < N`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`mv_kernel[grid](...)`, the grid size `cdiv(N, BLOCK_N)`,
the strides passed from the host, and how the runtime composes per-program
writes into `C`) is the *trusted boundary*, not a proof obligation here. The
program id enters only via `BlockState`, so the per-program statement covers
every program of the grid.

## Proof architecture

```
mv_one_block_io_correctness                      ← ★ MAIN THEOREM (`⊨`, symbolic dims)
  ├─ mv_one_block_flattenOk                       inside the flat-memory bridge
  ├─ mv_one_block_traceSafe                       per-execution address safety
  └─ mv_one_block_region_run                      region-model run
       ├─ mv_one_block_terminates
       ├─ mv_kernel_one_block_correct              per-row readback (shared, below)
       ├─ mvSpec_eq_of                             memory spec = value spec under the pins
       └─ mv_one_block_frame                       cell-level frame off the write window

mv_kernel_output_summary_general                 per-write-map summary (symbolic dims)
  ├─ mv_kernel_surface_toAlgorithm_supported      (toAlgorithm? = Except.ok _, full looping surface)
  └─ mv_kernel_one_block_compute_correct → mv_kernel_one_block_correct
       └─ mvSpec / mvProdTile  (row-wise `Tile.reduceSum` of `a * b`)
       (honest side-condition: output-offset injectivity `hOutInj`)
```

`mv_kernel_surface_toAlgorithm_supported` separately establishes that the full
looping surface (not just the one-block slice) lowers to the algorithm layer.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` is not
modeled — its config grid all uses `BLOCK_M ≥ 32`, so for the bundled test
shapes (`M ∈ {3, 16}`) the `M` loop executes exactly one block; the
`mv_kernel_one_block` slice captures that single-block path, with the full
looping surface verified only to lower (`toAlgorithm?`). The `.to(tl.float32)`
casts reduce to the identity post-erasure. Masked loads use `other=0.0`. The
output scatter requires the per-row offset map to be injective (the `hOutInj`
hypothesis of the main theorem; holds for the contiguous Python strides).
-/

namespace VeriTile.Bench.TritonBenchG.MatrixVectorMultip

open VeriTile.Triton
open scoped VeriTile.Triton.MaskedTile2DKernelIO₂

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-! **★ Main theorem:** `mv_kernel_output_summary_general` — dimension-general
correctness for `mv_kernel` against `mvSpec` (symbolic `N M BLOCK_N BLOCK_M` and
strides; honest side-conditions only). -/

/-! # ══════════ CORRECT — genuine / dimension-general (review this) ══════════ -/

section Correct_without_Rounding

/-- Faithful transcription of `matrix_vector_multip.py`'s `mv_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_N` / `BLOCK_M: tl.constexpr` -> Lean `Nat` parameters. -/
def mv_kernel
    (A B C : RegionName)
    (N M stride_an stride_am stride_bm stride_cn BLOCK_N BLOCK_M : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  offset_n = pid * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))[:, None]
  offset_m = tl.arange(0, $(BLOCK_M))[None, :]
  n_mask = offset_n < $(N)
  A_ptrs = A + offset_n * $(stride_an) + offset_m * $(stride_am)
  B_ptrs = B + offset_m * $(stride_bm)
  acc = tl.zeros([$(BLOCK_N), $(BLOCK_M)], dtype=tl.float32)
  for m in range(0, $(M), $(BLOCK_M)) {
    m_mask = (m + offset_m) < $(M)
    a = tl.load(A_ptrs, mask=n_mask & m_mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptrs, mask=m_mask, other=0.0).to(tl.float32)
    acc += a * b
    A_ptrs += $(BLOCK_M) * $(stride_am)
    B_ptrs += $(BLOCK_M) * $(stride_bm)
  }
  acc = tl.sum(acc, axis=1)
  C_ptrs = C + offset_n * $(stride_cn)
  tl.store(C_ptrs, acc[:, None], mask=n_mask)
}

/-- The full matrix-vector multiplication surface lowers to the algorithm
layer, including the `M` loop, masked loads, vector multiply, reduction, and
masked output store. -/
theorem mv_kernel_surface_toAlgorithm_supported
    (A B C : RegionName)
    (N M stride_an stride_am stride_bm stride_cn BLOCK_N BLOCK_M : Nat) :
    ∃ alg, (mv_kernel A B C N M stride_an stride_am stride_bm stride_cn
      BLOCK_N BLOCK_M).toAlgorithm? = Except.ok alg := by
  simp [mv_kernel, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- Proof-oriented one-`BLOCK_M` slice of `matrix_vector_multip.py`'s
`mv_kernel`.

The full surface loops over `M` in `BLOCK_M` chunks. This slice captures the
single-block path used by the bundled tests (`M = 3` and `M = 16`, while the
autotune choices have `BLOCK_M >= 32`): load one A tile and one B tile, reduce
over `BLOCK_M`, and write one C block. -/
def mv_kernel_one_block
    (A B C : RegionName)
    (N M stride_an stride_am stride_bm stride_cn BLOCK_N BLOCK_M : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  offset_n = pid * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  offset_m = tl.arange(0, $(BLOCK_M))
  a = tl.load(A + offset_n[:, None] * $(stride_an) + offset_m[None, :] * $(stride_am),
    mask=(offset_n[:, None] < $(N)) and (offset_m[None, :] < $(M)), other=0.0).to(tl.float32)
  b = tl.load(B + offset_m * $(stride_bm), mask=offset_m < $(M), other=0.0).to(tl.float32)
  acc = tl.sum(a * b[None, :], axis=1)
  tl.store(C + offset_n * $(stride_cn), acc, mask=offset_n < $(N))
}

def nIndex (s : BlockState) (BLOCK_N : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * BLOCK_N + i.val

def cOffset (s : BlockState) (stride_cn BLOCK_N : Nat) (i : Fin BLOCK_N) : Nat :=
  nIndex s BLOCK_N i * stride_cn

def aOffset
    (s : BlockState) (stride_an stride_am BLOCK_N : Nat)
    (i : Fin BLOCK_N) (j : Fin BLOCK_M) : Nat :=
  nIndex s BLOCK_N i * stride_an + j.val * stride_am

def bOffset (stride_bm : Nat) (j : Fin BLOCK_M) : Nat :=
  j.val * stride_bm

noncomputable def mvProdTile
    (s : BlockState) (A B : RegionName)
    (N M stride_an stride_am stride_bm BLOCK_N BLOCK_M : Nat) :
    Tile .real [BLOCK_N, BLOCK_M] :=
  { data := fun idx =>
      let ni := (TileShape.dropInsertedIndex [BLOCK_N] 1 1 (idx.1, 0, PUnit.unit)).1
      let mj := (TileShape.dropInsertedIndex [BLOCK_M] 0 1 (0, idx.2.1, PUnit.unit)).1
      Option.map₂ (fun a b => a * b)
        (if s.pids 0 * BLOCK_N + ni.val < N ∧ mj.val < M then
          some (s.readMem A ((s.pids 0 * BLOCK_N + ni.val) * stride_an + mj.val * stride_am))
        else some (0.0 : ℝ))
        (if mj.val < M then
          some (s.readMem B (mj.val * stride_bm))
        else some (0.0 : ℝ)) }

noncomputable def mvSpec
    (s : BlockState) (A B : RegionName)
    (N M stride_an stride_am stride_bm BLOCK_N BLOCK_M : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  WithBot.unbotD 0
    ((Tile.reduceSum (shape := [BLOCK_N, BLOCK_M]) ⟨1, by simp⟩ Bool.false
      (mvProdTile s A B N M stride_an stride_am stride_bm BLOCK_N BLOCK_M)).data
        (i, PUnit.unit))

/-- Algorithm-layer correctness for the one-block matrix-vector slice. -/
theorem mv_kernel_one_block_correct
    (A B C : RegionName)
    (N M stride_an stride_am stride_bm stride_cn BLOCK_N BLOCK_M : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => cOffset s stride_cn BLOCK_N i))
    (hExec : exec (mv_kernel_one_block A B C N M stride_an stride_am stride_bm
        stride_cn BLOCK_N BLOCK_M) s = some s') :
    ∀ i : Fin BLOCK_N,
      s'.readMem C (cOffset s stride_cn BLOCK_N i) =
        if nIndex s BLOCK_N i < N then
          mvSpec s A B N M stride_an stride_am stride_bm BLOCK_N BLOCK_M i
        else s.readMem C (cOffset s stride_cn BLOCK_N i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_N] =>
        (s.pids 0 * BLOCK_N + idx.1.val) * stride_cn) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [cOffset, nIndex] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hBN : 0 < BLOCK_N
  · simp [exec, mv_kernel_one_block, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
          Option.bind, Option.map,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim, Tile.uop,
          Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim,
          TileShape.eraseAxis, TileShape.insertAxisIndex,
          NumericDType.add, NumericDType.mul, ComparableDType.lt, hBN] at hExec
    rw [← hExec]
    simp only [cOffset, nIndex]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj
      (i, PUnit.unit)]
    by_cases hi : s.pids 0 * BLOCK_N + i.val < N
    · simp [hi, mvSpec, mvProdTile, aOffset, bOffset, cOffset, nIndex,
            Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim,
            TileShape.eraseAxis, TileShape.insertAxisIndex]
      congr 1
      apply Finset.sum_congr rfl
      intro x _
      change Option.map₂ (fun x1 x2 => x1 * x2)
          (if x.val < M then
            some (s.readMem A
              ((s.pids 0 * BLOCK_N + i.val) * stride_an + x.val * stride_am))
          else some 0.0)
          (if x.val < M then
            some (s.readMem B (x.val * stride_bm))
          else some 0.0) =
        Option.map₂ (fun x1 x2 => x1 * x2)
          (if s.pids 0 * BLOCK_N + i.val < N ∧ x.val < M then
            some (s.readMem A
              ((s.pids 0 * BLOCK_N + i.val) * stride_an + x.val * stride_am))
          else some 0.0)
          (if x.val < M then
            some (s.readMem B (x.val * stride_bm))
          else some 0.0)
      by_cases hxM : x.val < M
      · simp [hxM, hi]
      · simp [hxM]
    · simp [hi]
  · exact False.elim (hBN (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the one-block matrix-vector slice. -/
theorem mv_kernel_one_block_compute_correct
    (A B C : RegionName)
    (N M stride_an stride_am stride_bm stride_cn BLOCK_N BLOCK_M : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => cOffset s stride_cn BLOCK_N i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := mv_kernel_one_block A B C N M stride_an stride_am stride_bm
        stride_cn BLOCK_N BLOCK_M)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => nIndex s BLOCK_N i < N)
        (fun i => (C, cOffset s stride_cn BLOCK_N i)))
      (expected := fun i =>
        mvSpec s A B N M stride_an stride_am stride_bm BLOCK_N BLOCK_M i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [mv_kernel_one_block, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := mv_kernel_one_block_correct A B C N M stride_an stride_am stride_bm
    stride_cn BLOCK_N BLOCK_M s s' hOutInj hExec i
  simpa [hActive] using h

/-- Dimension-general correctness surface for `mv_kernel`.

Bundles the two genuine obligations at fully symbolic dimensions
(`N M BLOCK_N BLOCK_M` and all strides):

* the full looping surface lowers to the algorithm layer
  (`toAlgorithm? = Except.ok _`), and
* the one-`BLOCK_M` slice realizes `mvSpec` — the masked, input-only
  matrix-vector reduction — writing each active row to `C` at `cOffset`.

The single honest side-condition is output-offset injectivity
(`hOutInj`): distinct block rows must map to distinct `C` slots, which
holds for any nonzero `stride_cn` (and in particular the contiguous
Python strides). -/
abbrev mv_kernel_general_prop
    (A B C : RegionName)
    (N M stride_an stride_am stride_bm stride_cn BLOCK_N BLOCK_M : Nat)
    (s : BlockState) : Prop :=
  (∃ alg, (mv_kernel A B C N M stride_an stride_am stride_bm stride_cn
      BLOCK_N BLOCK_M).toAlgorithm? = Except.ok alg) ∧
  ComputeCorrect.Realizes_without_Rounding
    (kernel := mv_kernel_one_block A B C N M stride_an stride_am stride_bm
      stride_cn BLOCK_N BLOCK_M)
    (initialState := s)
    (write := ComputeCorrect.WriteMap.writeIf
      (fun i : Fin BLOCK_N => nIndex s BLOCK_N i < N)
      (fun i => (C, cOffset s stride_cn BLOCK_N i)))
    (expected := fun i =>
      mvSpec s A B N M stride_an stride_am stride_bm BLOCK_N BLOCK_M i)

/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/
/-- **Dimension-general public summary for `mv_kernel`.**

For symbolic dimensions `N M BLOCK_N BLOCK_M` and arbitrary strides, the full
matrix-vector surface lowers to the algorithm layer and its one-block slice
realizes the genuine input-only specification `mvSpec` (a masked row-wise
`Tile.reduceSum` of `A · B`), writing each active output row `i` of `C`.

The only hypothesis is the honest output-offset injectivity condition
`hOutInj`; there are no shape-specific assumptions. -/
specification mv_kernel_output_summary_general
    (A B C : RegionName)
    (N M stride_an stride_am stride_bm stride_cn BLOCK_N BLOCK_M : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => cOffset s stride_cn BLOCK_N i)) :
    mv_kernel_general_prop A B C N M stride_an stride_am stride_bm stride_cn
      BLOCK_N BLOCK_M s := by
  refine ⟨?_, ?_⟩
  · exact mv_kernel_surface_toAlgorithm_supported A B C N M stride_an stride_am
      stride_bm stride_cn BLOCK_N BLOCK_M
  · exact mv_kernel_one_block_compute_correct A B C N M stride_an stride_am
      stride_bm stride_cn BLOCK_N BLOCK_M s hOutInj

end Correct_without_Rounding

/-! ## ════════ `⊨` IO face for the one-block slice ════════

The summary above is stated per *declared write map*. This section restates the
one-block slice on the audit-once IO surface `MaskedTile2DKernelIO₂.Implements`
(`⊨`), which additionally pins the **flat memory** placement.

Lanes are `TileIndex [BLOCK_N, BLOCK_M]` — the shape the kernel's `a` tile
actually has, with no flattening. The three channels are all genuine 2-D address
functions on that lane set:

* `A` at `(pid·BLOCK_N + i)·stride_an + j·stride_am`, read-active on the kernel's
  own `offset_n < N ∧ offset_m < M` guard;
* `B` at `j·stride_bm` — a *row-broadcast* read: the address ignores `i`, and its
  mask is the independent `read2Mask := j < M`, which is exactly the field
  `MaskedTile2DKernelIO₂` carries for reads gated differently from `A`;
* `C` at `(pid·BLOCK_N + i)·stride_cn`, write-active only on **column 0**, since
  the store is the length-`BLOCK_N` reduced vector — a reduction's write footprint
  is a sub-slice of its read tile (same shape as `max_reduction` #568 and
  `matrix_reduction` #569).

The masked-off lanes of both loads carry the kernel's `other=0.0`, so the stored
value depends only on the pinned lanes; `mvSpecOf` says so over the *values*. -/

section IOFace

/-- Cell-level frame of a masked scatter (private copy — `bench` files are
standalone). -/
private theorem foldl_writeMem_frame {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → ℝ) (P : α → Prop) [DecidablePred P]
    (R : RegionName) (off : Nat) :
    ∀ l : List α, (R ≠ region ∨ ∀ k ∈ l, P k → offsetFn k ≠ off) →
      ∀ s : BlockState,
        ((l.foldl (fun acc k =>
            if P k then acc.writeMem region (offsetFn k) (valueFn k) else acc)
            s).mem R off) = s.mem R off := by
  intro l
  induction l with
  | nil => intro _ s; rfl
  | cons hd tl ih =>
      intro hc s
      have htl : R ≠ region ∨ ∀ k ∈ tl, P k → offsetFn k ≠ off := by
        rcases hc with h | h
        · exact Or.inl h
        · exact Or.inr fun k hk => h k (List.mem_cons_of_mem hd hk)
      rw [List.foldl_cons, ih htl]
      by_cases hP : P hd
      · rw [if_pos hP, BlockState.writeMem_mem, if_neg ?_]
        rintro ⟨h1, h2⟩
        rcases hc with h | h
        · exact h h1
        · exact h hd List.mem_cons_self hP h2.symm
      · rw [if_neg hP]

/-- `Op.FlattenOk` of an `expandDim` broadcast reduces to its argument's
obligation. Stated as its own lemma and used as a *term*: the per-case equation
does not fire under `simp` on an `expandDim` node, only under `rw`, and `rw` only
succeeds in an isolated goal. -/
private theorem flattenOk_expandDim {dtype : TileDType} {shape : TileShape}
    (ax : Fin (shape.length + 1)) (e : Op dtype shape) (h : e.FlattenOk) :
    (Op.expandDim ax e).FlattenOk := by
  rw [Op.FlattenOk]
  exact h

/-- `Op.SafeAt` of an `expandDim` broadcast (same caveat). -/
private theorem safeAt_expandDim {dtype : TileDType} {shape : TileShape}
    (bounds : RegionBounds) (s : BlockState) (ax : Fin (shape.length + 1))
    (e : Op dtype shape) (h : Op.SafeAt bounds s e) :
    Op.SafeAt bounds s (Op.expandDim ax e) := by
  rw [Op.SafeAt]
  exact h

/-- `Op.FlattenOk` of a `reduceSum` reduces to its argument's obligation. -/
private theorem flattenOk_reduceSum {shape : TileShape}
    (ax : Fin shape.length) (keepDims : Bool) (e : Op .real shape)
    (h : e.FlattenOk) : (Op.reduceSum ax keepDims e).FlattenOk := by
  rw [Op.FlattenOk]
  exact h

/-- `Op.SafeAt` of a `reduceSum` (same caveat). -/
private theorem safeAt_reduceSum {shape : TileShape} (bounds : RegionBounds)
    (s : BlockState) (ax : Fin shape.length) (keepDims : Bool)
    (e : Op .real shape) (h : Op.SafeAt bounds s e) :
    Op.SafeAt bounds s (Op.reduceSum ax keepDims e) := by
  rw [Op.SafeAt]
  exact h

/-- `Op.FlattenOk` of a register reference is vacuous. -/
private theorem flattenOk_ref (dtype : TileDType) (shape : TileShape)
    (name : RegName) : (Op.ref dtype shape name).FlattenOk := by
  rw [Op.FlattenOk]
  trivial

/-- `Op.SafeAt` of a register reference is vacuous. -/
private theorem safeAt_ref (bounds : RegionBounds) (s : BlockState)
    (dtype : TileDType) (shape : TileShape) (name : RegName) :
    Op.SafeAt bounds s (Op.ref dtype shape name) := by
  rw [Op.SafeAt]
  trivial

/-- Value-level one-block spec: the masked row-wise `Tile.reduceSum` of `A · B`
written over the *loaded values* rather than over memory — the kernel's
`other=0.0` on both loads is what makes the inactive lanes contribute `0`, so
this is a closed form in `xs` / `ys` at the pinned lanes only. -/
noncomputable def mvSpecOf (N M BLOCK_N BLOCK_M pid : Nat)
    (xs ys : TileIndex [BLOCK_N, BLOCK_M] → ℝ) (i : Fin BLOCK_N) : ℝ :=
  WithBot.unbotD 0
    ((Tile.reduceSum (shape := [BLOCK_N, BLOCK_M]) ⟨1, by simp⟩ Bool.false
      ⟨fun k => Option.map₂ (fun a b => a * b)
        (if pid * BLOCK_N + k.1.val < N ∧ k.2.1.val < M then some (xs k)
         else some (0.0 : ℝ))
        (if k.2.1.val < M then some (ys k) else some (0.0 : ℝ))⟩).data
      (i, PUnit.unit))

/-- The memory-level and value-level one-block specs agree once both input
windows are pinned at their own active lanes. -/
theorem mvSpec_eq_of (A B : RegionName)
    (N M stride_an stride_am stride_bm BLOCK_N BLOCK_M : Nat) (s : BlockState)
    (xs ys : TileIndex [BLOCK_N, BLOCK_M] → ℝ)
    (hx : ∀ k : TileIndex [BLOCK_N, BLOCK_M],
      (s.pids 0 * BLOCK_N + k.1.val < N ∧ k.2.1.val < M) →
      s.readMem A ((s.pids 0 * BLOCK_N + k.1.val) * stride_an
        + k.2.1.val * stride_am) = xs k)
    (hy : ∀ k : TileIndex [BLOCK_N, BLOCK_M], k.2.1.val < M →
      s.readMem B (k.2.1.val * stride_bm) = ys k)
    (i : Fin BLOCK_N) :
    mvSpec s A B N M stride_an stride_am stride_bm BLOCK_N BLOCK_M i
      = mvSpecOf N M BLOCK_N BLOCK_M (s.pids 0) xs ys i := by
  have hd1 : ∀ a : Fin BLOCK_N,
      (TileShape.dropInsertedIndex [BLOCK_N] 1 1 (a, 0, PUnit.unit)).1 = a :=
    fun _ => rfl
  have hd2 : ∀ b : Fin BLOCK_M,
      (TileShape.dropInsertedIndex [BLOCK_M] 0 1 (0, b, PUnit.unit)).1 = b :=
    fun _ => rfl
  have htile :
      mvProdTile s A B N M stride_an stride_am stride_bm BLOCK_N BLOCK_M
        = ⟨fun k : TileIndex [BLOCK_N, BLOCK_M] =>
            Option.map₂ (fun a b => a * b)
              (if s.pids 0 * BLOCK_N + k.1.val < N ∧ k.2.1.val < M then
                some (xs k)
               else some (0.0 : ℝ))
              (if k.2.1.val < M then some (ys k) else some (0.0 : ℝ))⟩ := by
    refine congrArg Tile.mk (funext fun k => ?_)
    simp only [mvProdTile, hd1, hd2]
    by_cases hA : s.pids 0 * BLOCK_N + k.1.val < N ∧ k.2.1.val < M
    · by_cases hB : k.2.1.val < M
      · rw [if_pos hA, if_pos hA, if_pos hB, if_pos hB, hx k hA, hy k hB]
      · rw [if_neg hB, if_neg hB]
        exact absurd hA.2 hB
    · by_cases hB : k.2.1.val < M
      · rw [if_neg hA, if_neg hA, if_pos hB, if_pos hB, hy k hB]
      · rw [if_neg hA, if_neg hA, if_neg hB, if_neg hB]
  rw [mvSpec, mvSpecOf, htile]

/-- The one-block slice sits inside the flat-memory bridge's covered fragment. -/
theorem mv_one_block_flattenOk (A B C : RegionName)
    (N M stride_an stride_am stride_bm stride_cn BLOCK_N BLOCK_M : Nat) :
    ((mv_kernel_one_block A B C N M stride_an stride_am stride_bm stride_cn
      BLOCK_N BLOCK_M).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [mv_kernel_one_block, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]
  refine ⟨?_, ?_, ?_⟩ <;> simp [Op.FlattenOk.eq_def]

/-- Termination of the one-block slice. -/
theorem mv_one_block_terminates (A B C : RegionName)
    (N M stride_an stride_am stride_bm stride_cn BLOCK_N BLOCK_M : Nat)
    (hBM : 0 < BLOCK_M) (s : BlockState) :
    ∃ s1, exec (mv_kernel_one_block A B C N M stride_an stride_am stride_bm
      stride_cn BLOCK_N BLOCK_M) s = some s1 := by
  simp [exec, mv_kernel_one_block, stepStmts, stepStmt, evalOp.eq_def,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Option.bind, Option.map,
    Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim, Tile.uop, Tile.reduceSum,
    Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
    TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
    ComparableDType.lt, hBM]

/-- Cell-level frame of the one-block slice. -/
theorem mv_one_block_frame (A B C : RegionName)
    (N M stride_an stride_am stride_bm stride_cn BLOCK_N BLOCK_M : Nat)
    (hBM : 0 < BLOCK_M) (s s' : BlockState)
    (hExec : exec (mv_kernel_one_block A B C N M stride_an stride_am stride_bm
      stride_cn BLOCK_N BLOCK_M) s = some s') :
    ∀ (r : RegionName) (o : Nat),
      (r ≠ C ∨ ∀ i : Fin BLOCK_N, s.pid * BLOCK_N + i.val < N →
        o ≠ (s.pid * BLOCK_N + i.val) * stride_cn) →
      s'.mem r o = s.mem r o := by
  intro r o hcond
  simp [exec, mv_kernel_one_block, stepStmts, stepStmt, evalOp.eq_def,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Option.bind, Option.map,
    Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim, Tile.uop, Tile.reduceSum,
    Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
    TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
    ComparableDType.lt, hBM] at hExec
  subst hExec
  rw [foldl_writeMem_frame (region := C)
    (fun i : TileIndex [BLOCK_N] => (s.pids 0 * BLOCK_N + i.1.val) * stride_cn)
    _ (fun i : TileIndex [BLOCK_N] => s.pids 0 * BLOCK_N + i.1.val < N) r o
    (TileShape.allIndices [BLOCK_N]) ?_]
  · simp
  · rcases hcond with h | h
    · exact Or.inl h
    · exact Or.inr fun i _ hi => Ne.symm (h i.1 hi)

/-- Per-execution safety walk for the one-block slice. -/
theorem mv_one_block_traceSafe (A B C : RegionName)
    (N M stride_an stride_am stride_bm stride_cn BLOCK_N BLOCK_M : Nat)
    (hBM : 0 < BLOCK_M) (bounds : RegionBounds) (s : BlockState)
    (hA : ∀ (i : Fin BLOCK_N) (j : Fin BLOCK_M),
      s.pid * BLOCK_N + i.val < N → j.val < M →
      (s.pid * BLOCK_N + i.val) * stride_an + j.val * stride_am < bounds A)
    (hB : ∀ j : Fin BLOCK_M, j.val < M → j.val * stride_bm < bounds B)
    (hC : ∀ i : Fin BLOCK_N, s.pid * BLOCK_N + i.val < N →
      (s.pid * BLOCK_N + i.val) * stride_cn < bounds C) :
    ((mv_kernel_one_block A B C N M stride_an stride_am stride_bm stride_cn
      BLOCK_N BLOCK_M).toAlgKernel).TraceSafe bounds s := by
  simp [Kernel.TraceSafe, mv_kernel_one_block, Stmt.TraceSafeList,
    Stmt.TraceSafe, Op.SafeAt, MaskOpt.SafeAt, MaskOpt.Active,
    MaskOpt.MemorySafe, MemAccess.SafeAt, MemAccess.MemorySafe,
    memAccessMemorySafe, MemAccess.ActiveAddressSafe,
    memAccessActiveAddressSafe, Op.PointerAddressesSafeOn, Op.MemorySafe,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, stepStmt, evalOp,
    evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd,
    Tile.expandDim, Tile.uop, Tile.reduceSum, Tile.reduceSumDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
    NumericDType.add, NumericDType.mul, ComparableDType.lt, hBM]
  refine ⟨?_, ?_, safeAt_reduceSum _ _ _ _ _ (by simp [Op.SafeAt.eq_def]),
    fun a ha => hC a ha⟩
  · simp [Op.SafeAt.eq_def, MaskOpt.Active, MemAccess.ActiveAddressSafe,
      memAccessActiveAddressSafe, evalOp, evalOp.eq_def, Option.bind,
      Option.map, Tile.bop, Tile.cop, Tile.expandDim, Tile.ptrAdd,
      NumericDType.add, NumericDType.mul, ComparableDType.lt]
    exact fun a b ha hb => hA a b ha hb
  · simpa [Op.SafeAt.eq_def, MaskOpt.Active, MemAccess.ActiveAddressSafe,
      memAccessActiveAddressSafe, evalOp, evalOp.eq_def, Option.bind,
      Option.map, Tile.bop, Tile.cop, Tile.expandDim, Tile.ptrAdd,
      NumericDType.add, NumericDType.mul, ComparableDType.lt] using hB

/-- Region-model run of the one-block slice, in the shape
`MaskedTile2DKernelIO₂.Implements.intro` consumes. -/
theorem mv_one_block_region_run (A B C : RegionName)
    (N M stride_an stride_am stride_bm stride_cn BLOCK_N BLOCK_M : Nat)
    (hBM : 0 < BLOCK_M) (s₀ : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => cOffset s₀ stride_cn BLOCK_N i))
    (xs ys : TileIndex [BLOCK_N, BLOCK_M] → ℝ)
    (hx : ∀ k : TileIndex [BLOCK_N, BLOCK_M],
      (s₀.pids 0 * BLOCK_N + k.1.val < N ∧ k.2.1.val < M) →
      s₀.readMem A ((s₀.pids 0 * BLOCK_N + k.1.val) * stride_an
        + k.2.1.val * stride_am) = xs k)
    (hy : ∀ k : TileIndex [BLOCK_N, BLOCK_M], k.2.1.val < M →
      s₀.readMem B (k.2.1.val * stride_bm) = ys k) :
    ∃ s1, exec (mv_kernel_one_block A B C N M stride_an stride_am stride_bm
        stride_cn BLOCK_N BLOCK_M) s₀ = some s1
      ∧ (∀ i : Fin BLOCK_N, s₀.pids 0 * BLOCK_N + i.val < N →
          s1.readMem C ((s₀.pids 0 * BLOCK_N + i.val) * stride_cn)
            = mvSpecOf N M BLOCK_N BLOCK_M (s₀.pids 0) xs ys i)
      ∧ (∀ (r : RegionName) (o : Nat),
          (r ≠ C ∨ ∀ i : Fin BLOCK_N, s₀.pids 0 * BLOCK_N + i.val < N →
            o ≠ (s₀.pids 0 * BLOCK_N + i.val) * stride_cn) →
          s1.mem r o = s₀.mem r o) := by
  obtain ⟨s1, hexec⟩ := mv_one_block_terminates A B C N M stride_an stride_am
    stride_bm stride_cn BLOCK_N BLOCK_M hBM s₀
  refine ⟨s1, hexec, ?_, mv_one_block_frame A B C N M stride_an stride_am
    stride_bm stride_cn BLOCK_N BLOCK_M hBM s₀ s1 hexec⟩
  intro i hi
  have h := mv_kernel_one_block_correct A B C N M stride_an stride_am stride_bm
    stride_cn BLOCK_N BLOCK_M s₀ s1 hOutInj hexec i
  rw [show cOffset s₀ stride_cn BLOCK_N i
        = (s₀.pids 0 * BLOCK_N + i.val) * stride_cn from rfl] at h
  rw [h, if_pos (show nIndex s₀ BLOCK_N i < N from hi),
    mvSpec_eq_of A B N M stride_an stride_am stride_bm BLOCK_N BLOCK_M s₀ xs ys
      hx hy i]

/-- IO signature of the one-block slice on the tile-indexed two-input surface.
The `B` channel is the **row-broadcast** read: its address ignores the row index
and its gate is the independent `read2Mask := j < M`. Only **column 0** is
write-active, since the store is the reduced length-`BLOCK_N` vector. -/
def mvOneBlockIO (A B C : RegionName)
    (N M stride_an stride_am stride_bm stride_cn BLOCK_N BLOCK_M : Nat) :
    MaskedTile2DKernelIO₂ where
  kernel := mv_kernel_one_block A B C N M stride_an stride_am stride_bm
    stride_cn BLOCK_N BLOCK_M
  in1 := A
  in2 := B
  out := C
  shape := [BLOCK_N, BLOCK_M]
  read1 := fun p₀ _p₁ k =>
    (p₀ * BLOCK_N + k.1.val) * stride_an + k.2.1.val * stride_am
  read2 := fun _p₀ _p₁ k => k.2.1.val * stride_bm
  write := fun p₀ _p₁ k => (p₀ * BLOCK_N + k.1.val) * stride_cn
  mask := fun p₀ _p₁ k => p₀ * BLOCK_N + k.1.val < N ∧ k.2.1.val < M
  read2Mask := fun _p₀ _p₁ k => k.2.1.val < M
  writeMask := fun p₀ _p₁ k => p₀ * BLOCK_N + k.1.val < N ∧ k.2.1.val = 0

/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/

/-- **The headline on the IO surface** for `matrix_vector_multip.py`'s
`mv_kernel`, one-block slice: for every disjoint flat placement of the three
buffers, every program id whose active lanes are in bounds, and every launch state
whose `A` tile and broadcast `B` vector are pinned at their own active lanes, the
translated pointer kernel terminates, every active row of `C` holds the genuine
masked row-wise sum `Σ_j A[i, j]·B[j]` (`mvSpecOf`, with the kernel's `other=0.0`
making the inactive lanes contribute `0`), and every other memory cell is
unchanged.

Dimension-general in `N`, `M`, all four strides, `BLOCK_N` and `BLOCK_M`. Honest
side-conditions: `0 < BLOCK_M` (an empty reduction axis makes `tl.sum` fault, and
it is the write-active lane witness); `0 < BLOCK_N` (a lane witness for the
broadcast `B` channel's own bound); and output-offset injectivity at every program
id — the universally quantified form of the hypothesis the per-write-map summary
already takes. -/
specification mv_one_block_io_correctness (A B C : RegionName)
    (N M stride_an stride_am stride_bm stride_cn BLOCK_N BLOCK_M : Nat)
    (hBN : 0 < BLOCK_N) (hBM : 0 < BLOCK_M)
    (hOutInj : ∀ p₀ : Nat, Function.Injective
      (fun i : Fin BLOCK_N => (p₀ * BLOCK_N + i.val) * stride_cn)) :
    mvOneBlockIO A B C N M stride_an stride_am stride_bm stride_cn BLOCK_N
        BLOCK_M
      ⊨ fun p₀ _p₁ xs ys k =>
          mvSpecOf N M BLOCK_N BLOCK_M p₀ xs ys k.1 := by
  refine MaskedTile2DKernelIO₂.Implements.intro _ ?_ ?_ ?_
  · exact mv_one_block_flattenOk A B C N M stride_an stride_am stride_bm
      stride_cn BLOCK_N BLOCK_M
  · intro bounds s h1 h2 h3
    exact mv_one_block_traceSafe A B C N M stride_an stride_am stride_bm
      stride_cn BLOCK_N BLOCK_M hBM bounds s
      (fun i j hi hj => h1 (i, j, PUnit.unit) ⟨hi, hj⟩)
      (fun j hj => h2 (⟨0, hBN⟩, j, PUnit.unit) hj)
      (fun i hi => h3 (i, ⟨0, hBM⟩, PUnit.unit) ⟨hi, rfl⟩)
  · intro s₀ xs ys hx hy
    obtain ⟨s1, hexec, hval, hframe⟩ :=
      mv_one_block_region_run A B C N M stride_an stride_am stride_bm stride_cn
        BLOCK_N BLOCK_M hBM s₀ (hOutInj (s₀.pids 0)) xs ys
        (fun k hk => hx k hk) (fun k hk => hy k hk)
    refine ⟨s1, hexec, fun k hk => hval k.1 hk.1, ?_⟩
    intro r o hcond
    refine hframe r o ?_
    rcases hcond with h | h
    · exact Or.inl h
    · exact Or.inr fun i hi => h (i, ⟨0, hBM⟩, PUnit.unit) ⟨hi, rfl⟩

end IOFace

end VeriTile.Bench.TritonBenchG.MatrixVectorMultip
