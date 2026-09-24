# Spec sheet — `bench/tritonbench_g/matrix_vector_multip/MatrixVectorMultip.lean`

**Python source:** `bench/tritonbench_g/matrix_vector_multip/matrix_vector_multip.py`

## Public theorem: `mv_kernel_output_summary_general`

<details><summary>docstring</summary>

```
/-- **Dimension-general public summary for `mv_kernel`.**

For symbolic dimensions `N M BLOCK_N BLOCK_M` and arbitrary strides, the full
matrix-vector surface lowers to the algorithm layer and its one-block slice
realizes the genuine input-only specification `mvSpec` (a masked row-wise
`Tile.reduceSum` of `A · B`), writing each active output row `i` of `C`.

The only hypothesis is the honest output-offset injectivity condition
`hOutInj`; there are no shape-specific assumptions. -/
```
</details>

**Statement:**
```lean
specification mv_kernel_output_summary_general
    (A B C : RegionName)
    (N M stride_an stride_am stride_bm stride_cn BLOCK_N BLOCK_M : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => cOffset s stride_cn BLOCK_N i)) :
    mv_kernel_general_prop A B C N M stride_an stride_am stride_bm stride_cn
      BLOCK_N BLOCK_M s
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => cOffset s stride_cn BLOCK_N i)`

**Closed-form spec defs (transitive):** `cOffset`, `mv_kernel_general_prop`, `nIndex`, `mv_kernel`, `mv_kernel_one_block`, `mvSpec`, `mvProdTile`

<details><summary><code>cOffset</code></summary>

```lean
def cOffset (s : BlockState) (stride_cn BLOCK_N : Nat) (i : Fin BLOCK_N) : Nat :=
  nIndex s BLOCK_N i * stride_cn
```
</details>

<details><summary><code>mv_kernel_general_prop</code></summary>

```
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
```
```lean
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
```
</details>

<details><summary><code>nIndex</code></summary>

```lean
def nIndex (s : BlockState) (BLOCK_N : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * BLOCK_N + i.val
```
</details>

<details><summary><code>mv_kernel</code></summary>

```
/-- Faithful transcription of `matrix_vector_multip.py`'s `mv_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_N` / `BLOCK_M: tl.constexpr` -> Lean `Nat` parameters. -/
```
```lean
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
```
</details>

<details><summary><code>mv_kernel_one_block</code></summary>

```
/-- Proof-oriented one-`BLOCK_M` slice of `matrix_vector_multip.py`'s
`mv_kernel`.

The full surface loops over `M` in `BLOCK_M` chunks. This slice captures the
single-block path used by the bundled tests (`M = 3` and `M = 16`, while the
autotune choices have `BLOCK_M >= 32`): load one A tile and one B tile, reduce
over `BLOCK_M`, and write one C block. -/
```
```lean
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
```
</details>

<details><summary><code>mvSpec</code></summary>

```lean
noncomputable def mvSpec
    (s : BlockState) (A B : RegionName)
    (N M stride_an stride_am stride_bm BLOCK_N BLOCK_M : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  WithBot.unbotD 0
    ((Tile.reduceSum (shape := [BLOCK_N, BLOCK_M]) ⟨1, by simp⟩ Bool.false
      (mvProdTile s A B N M stride_an stride_am stride_bm BLOCK_N BLOCK_M)).data
        (i, PUnit.unit))
```
</details>

<details><summary><code>mvProdTile</code></summary>

```lean
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
```
</details>

## Public theorem: `mv_one_block_io_correctness`

<details><summary>docstring</summary>

```
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
```
</details>

**Statement:**
```lean
specification mv_one_block_io_correctness (A B C : RegionName)
    (N M stride_an stride_am stride_bm stride_cn BLOCK_N BLOCK_M : Nat)
    (hBN : 0 < BLOCK_N) (hBM : 0 < BLOCK_M)
    (hOutInj : ∀ p₀ : Nat, Function.Injective
      (fun i : Fin BLOCK_N => (p₀ * BLOCK_N + i.val) * stride_cn)) :
    mvOneBlockIO A B C N M stride_an stride_am stride_bm stride_cn BLOCK_N
        BLOCK_M
      ⊨ fun p₀ _p₁ xs ys k =>
          mvSpecOf N M BLOCK_N BLOCK_M p₀ xs ys k.1
```

**Assumptions / layout contracts:**
- `hBN : 0 < BLOCK_N`
- `hBM : 0 < BLOCK_M`

**Closed-form spec defs (transitive):** `mvOneBlockIO`, `mvSpecOf`, `mv_kernel_one_block`

<details><summary><code>mvOneBlockIO</code></summary>

```
/-- IO signature of the one-block slice on the tile-indexed two-input surface.
The `B` channel is the **row-broadcast** read: its address ignores the row index
and its gate is the independent `read2Mask := j < M`. Only **column 0** is
write-active, since the store is the reduced length-`BLOCK_N` vector. -/
```
```lean
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
```
</details>

<details><summary><code>mvSpecOf</code></summary>

```
/-- Value-level one-block spec: the masked row-wise `Tile.reduceSum` of `A · B`
written over the *loaded values* rather than over memory — the kernel's
`other=0.0` on both loads is what makes the inactive lanes contribute `0`, so
this is a closed form in `xs` / `ys` at the pinned lanes only. -/
```
```lean
noncomputable def mvSpecOf (N M BLOCK_N BLOCK_M pid : Nat)
    (xs ys : TileIndex [BLOCK_N, BLOCK_M] → ℝ) (i : Fin BLOCK_N) : ℝ :=
  WithBot.unbotD 0
    ((Tile.reduceSum (shape := [BLOCK_N, BLOCK_M]) ⟨1, by simp⟩ Bool.false
      ⟨fun k => Option.map₂ (fun a b => a * b)
        (if pid * BLOCK_N + k.1.val < N ∧ k.2.1.val < M then some (xs k)
         else some (0.0 : ℝ))
        (if k.2.1.val < M then some (ys k) else some (0.0 : ℝ))⟩).data
      (i, PUnit.unit))
```
</details>

<details><summary><code>mv_kernel_one_block</code></summary>

```
/-- Proof-oriented one-`BLOCK_M` slice of `matrix_vector_multip.py`'s
`mv_kernel`.

The full surface loops over `M` in `BLOCK_M` chunks. This slice captures the
single-block path used by the bundled tests (`M = 3` and `M = 16`, while the
autotune choices have `BLOCK_M >= 32`): load one A tile and one B tile, reduce
over `BLOCK_M`, and write one C block. -/
```
```lean
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
```
</details>

## Also present (pinned special-case summaries)
- `mv_kernel_one_block_compute_correct`
