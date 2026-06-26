# Spec sheet — `bench/tritonbench_g/chunk_cumsum_vector/ChunkCumsumVector.lean`

**Python source:** `bench/tritonbench_g/chunk_cumsum_vector/chunk_cumsum_vector.py`

## Public theorem: `chunk_cumsum_vector_output_summary_general`

<details><summary>docstring</summary>

```
/-- **Dimension-general single-chunk output summary.** Subsumes the per-shape
`chunk_cumsum_vector_python_case{1,2,3}_output_summary` (which differ only in the
concrete `s_s_h s_s_t s_s_d T S BT BS` numerals) at fully symbolic dimensions.
For any single-Python-chunk shape (`T ≤ BT`, so the loop runs once with carry
`= 0`), the single-block `S → Z` surface realizes the genuine per-column global
prefix sum `singleBlockCumsumVectorClosed` — a standalone `Finset.sum`, never a
read-back of the kernel's own output. The active-lane collision-freedom of the
block address map is the explicit hypothesis. -/
```
</details>

**Statement:**
```lean
theorem chunk_cumsum_vector_output_summary_general
    (SReg Z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat) (s : BlockState)
    (hNoCollision : ∀ idx : TileIndex [BT, BS], singleBlockActive s T S BS idx →
      ∀ k : TileIndex [BT, BS], singleBlockActive s T S BS k →
        singleBlockTileOffset s s_s_h s_s_t s_s_d BS k =
          singleBlockTileOffset s s_s_h s_s_t s_s_d BS idx → k = idx) :
    ComputeCorrect.Realizes
      (kernel
```

**Assumptions / layout contracts:**
- `hNoCollision : ∀ idx : TileIndex [BT, BS], singleBlockActive s T S BS idx →
      ∀ k : TileIndex [BT, BS], singleBlockActive s T S BS k →
        singleBlockTileOffset s s_s_h s_s_t s_s_d BS k =
          singleBlockTileOffset s s_s_h s_s_t s_s_d BS idx → k = idx`

**Closed-form spec defs (transitive):** `singleBlockActive`, `singleBlockTileOffset`, `sIndex`

<details><summary><code>singleBlockActive</code></summary>

```lean
def singleBlockActive (s : BlockState) (T S BS : Nat)
    (idx : TileIndex [BT, BS]) : Prop :=
  idx.1.val < T ∧ sIndex s BS idx.2.1 < S
```
</details>

<details><summary><code>singleBlockTileOffset</code></summary>

```lean
def singleBlockTileOffset (s : BlockState) (s_s_h s_s_t s_s_d BS : Nat)
    (idx : TileIndex [BT, BS]) : Nat :=
  s.pids 1 * s_s_h + idx.1.val * s_s_t + sIndex s BS idx.2.1 * s_s_d
```
</details>

<details><summary><code>sIndex</code></summary>

```lean
def sIndex (s : BlockState) (BS : Nat) (j : Fin BS) : Nat :=
  s.pids 0 * BS + j.val
```
</details>

## Also present (pinned special-case summaries)
- `chunk_cumsum_vector_store_slice_compute_correct`
- `chunk_cumsum_vector_single_block_surface_compute_correct`
- `chunk_cumsum_vector_single_block_surface_active_compute_correct`
- `chunk_cumsum_vector_cumsum_slice_compute_correct`
- `chunk_cumsum_vector_store_slice_active_compute_correct`
- `chunk_cumsum_vector_cumsum_slice_active_compute_correct`
- `chunk_cumsum_vector_single_block_python_case1_compute_correct`
- `chunk_cumsum_vector_single_block_python_case2_compute_correct`
- `chunk_cumsum_vector_single_block_python_case3_compute_correct`
- `chunk_cumsum_vector_store_python_case1_compute_correct`
- `chunk_cumsum_vector_cumsum_python_case1_compute_correct`
- `chunk_cumsum_vector_store_python_case2_compute_correct`
- `chunk_cumsum_vector_cumsum_python_case2_compute_correct`
- `chunk_cumsum_vector_store_python_case3_compute_correct`
- `chunk_cumsum_vector_cumsum_python_case3_compute_correct`
- `chunk_cumsum_vector_python_case1_slice_summary`
- `chunk_cumsum_vector_python_case2_slice_summary`
- `chunk_cumsum_vector_python_case3_slice_summary`
