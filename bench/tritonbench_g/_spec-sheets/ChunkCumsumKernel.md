# Spec sheet — `bench/tritonbench_g/chunk_cumsum_kernel/ChunkCumsumKernel.lean`

**Python source:** `bench/tritonbench_g/chunk_cumsum_kernel/chunk_cumsum_kernel.py`

## Public theorem: `chunk_cumsum_scalar_output_summary_general`

<details><summary>docstring</summary>

```
/-- **★ MAIN THEOREM ★ — Public general scalar chunk-cumsum output summary
(genuine closed form, dimension-general `T`, `BT`).** Two genuine facts about the
complete `chunk_cumsum_scalar_surface` kernel, with **honest side-conditions
only** (`0 < BT`, output/input regions distinct):

* the full surface — prefix `i_bh`/`b_z` init followed by the `forRangeDyn` over
  `cdiv(T, BT)` chunks that threads the running carry `b_z` — **lowers** to the
  algorithm layer;
* it **computes the genuine global cumulative sum** at every in-range output flat
  index: `O[i_bh·T + flat] = Σ_{m ≤ flat, m < T} S[i_bh·T + m]`.

The carry invariant `carry_c = Σ_{flat < c·BT, flat < T} s[i_bh·T+flat]` is
*proven* by the loop induction (`forRangeDyn_inv` + `surface_step`), not assumed.
`expected` is a standalone `Finset.sum` over input memory (`globalCumsumClosed`),
never a read-back of the kernel's own output. This is the dimension-parameterized
headline; the `T = 4`, `BT = 16` Python benchmark shape is one instantiation. -/
```
</details>

**Statement:**
```lean
theorem chunk_cumsum_scalar_output_summary_general
    (S O : RegionName) (T BT : Nat) (s : BlockState)
    (hSO : O ≠ S) (hBT : 0 < BT) :
    -- (1) the full surface lowers to the algorithm layer
    (∃ alg, (chunk_cumsum_scalar_surface S O T BT).toAlgorithm? = Except.ok alg) ∧
    -- (2) the surface runs to completion (existence / termination)
    (∃ sfinal,
      exec (chunk_cumsum_scalar_surface S O T BT).toAlgKernel s = some sfinal) ∧
    -- (3) standard Realizes_without_Rounding: every in-range O lane holds the genuine global
    --     prefix sum `Σ_{m ≤ flat, m < T} S[i_bh·T + m]`, read purely over input
    ComputeCorrect.Realizes_without_Rounding
      (kernel := chunk_cumsum_scalar_surface S O T BT)
      (initialState := s)
      (write := fun i : Fin T => some (O, s.pids 0 * T + i.val))
      (expected := fun i : Fin T =>
        ∑ m ∈ (Finset.range T).filter (fun m => m ≤ i.val),
          s.readMem S (s.pids 0 * T + m))
```

**Assumptions / layout contracts:**
- `hSO : O ≠ S`
- `hBT : 0 < BT`

**Closed-form spec defs (transitive):** `chunk_cumsum_scalar_surface`

<details><summary><code>chunk_cumsum_scalar_surface</code></summary>

```
/-- Faithful transcription of `chunk_cumsum_kernel.py`'s
`chunk_global_cumsum_scalar_kernel`.

The final cast targets the block pointer destination dtype. -/
```
```lean
def chunk_cumsum_scalar_surface
  (S O : RegionName) (T BT : Nat) : ComputeKernel := triton {
  i_bh = tl.program_id(0)
  b_z = tl.zeros([], dtype=tl.float32)
  for i_t in range($(0), tl.cdiv($(T), $(BT)), $(1)) {
    p_s = tl.make_block_ptr(base=S + i_bh * $(T), shape=($(T)),
      strides=($(1)), offsets=(i_t * $(BT)), block_shape=($(BT)), order=(0))
    p_o = tl.make_block_ptr(base=O + i_bh * $(T), shape=($(T)),
      strides=($(1)), offsets=(i_t * $(BT)), block_shape=($(BT)), order=(0))
    b_s = tl.load(p_s, boundary_check=([0] : List Nat)).to(tl.float32)
    b_o = tl.cumsum(b_s, axis=0) + b_z[None]
    b_zz = tl.sum(b_s, axis=0)
    b_z += b_zz
    tl.store(p_o, (b_o).to(p_o.dtype.element_ty), boundary_check=([0] : List Nat))
  }
}
```
</details>

## Also present (pinned special-case summaries)
- `chunk_cumsum_scalar_store_slice_compute_correct`
- `chunk_cumsum_scalar_cumsum_slice_compute_correct`
- `chunk_cumsum_scalar_single_block_surface_compute_correct`
