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
theorem mv_kernel_output_summary_general
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

## Also present (pinned special-case summaries)
- `mv_kernel_one_block_compute_correct`
