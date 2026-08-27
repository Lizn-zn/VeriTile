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
specification chunk_cumsum_scalar_output_summary_general
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

## Public theorem: `chunk_cumsum_scalar_io_correctness`

<details><summary>docstring</summary>

```
/-- **The `⊨[R]` streaming headline (wave-5 S3 per-step emit genre).** For
every rounding model `R`, the faithful `chunk_cumsum_scalar_surface` — the
whole per-program kernel, prologue plus the `forRangeDyn` over
`cdiv(T, BT)` chunks that threads the running carry `b_z` — implements, on
its `StreamEmitMasked2DKernelIO₁` signature, the **ideal ℝ global cumulative
sum** over the streamed chunks: emitted window `(t, j)` holds the guarded
prefix sum `Σ_{u·BT+e ≤ t·BT+j, u·BT+e < T} xs[u, e]` of the entire stream.
The spec `f` is exact real arithmetic and is a standalone `Finset.sum` over
the *input* stream — never a read-back of the kernel's own output.

Rounding story: this kernel has **zero rounding events**. The lowered body
carries no `Op.castFloat` at all (both Python `.to(...)`s erase), its load is
`Op.load TileDType.real` and its in-loop store is
`Stmt.store TileDType.real … MaskOpt.none`, so `outDType := .real` (the skin
default) is the honest grid: the readback contract's `R.round .real` is the
identity by the model's defining `round_real`, and `writeMemTypedR R .real`
is definitionally the exact write. The ∀-`R` face therefore holds via the
`RoundingModel` `.real` identity fields, not as a `.triv` special case.

Layer map: the body is cast-free, so under `execR R` the whole kernel
collapses verbatim onto the exact stepper (`ccStepList_castFree` +
`stepForRangeAuxR_castFree` through the `forRangeDyn`), and the proven
`surfaceInv` / `surface_step` / `forRangeDyn_inv` stack above is reused
unchanged; the `⊨[R]` face adds the block-pointer `TraceSafeR` walk
(`cc_traceSafeR`, whose load/store obligations are discharged from the
`boundary_check=(0,)` guard `BlockPtr.inBounds_1d`), the per-cell memory
frame (`cc_body_frame`, the `mem` twin of `surface_step`), and the
stream-lane spec bridge (`ccGuardedSum_eq_flat`, re-blocking the global flat
index `m ↔ (m / BT, m % BT)`).

Both hypotheses are truth-forced — they are exactly the retained exact
headline `chunk_cumsum_scalar_output_summary_general`'s side conditions:

* `hSO : O ≠ S` — chunk `c+1` **loads** `s` after chunk `c` has **stored**
  into `o`; if the output aliased the input, later chunks would re-read
  already-overwritten values and the closed form would be false. It is the
  hypothesis `surface_step` consumes (via `surfaceScatter_readMem_ne`) to
  keep the streamed input pinned across the in-loop store.
* `hBT : 0 < BT` — the chunk width. It is the loop's block size: at
  `BT = 0` the trip count `cdiv(T, 0)` is meaningless, `surface_carry_succ`
  fails, and the step index `m / BT` used by the stream re-blocking is
  undefined. It holds for every real launch (the autotune set is
  `BT ∈ {16, 32, 64}`).

Relation to the exact surface: the exact headline
`chunk_cumsum_scalar_output_summary_general`
(`Realizes_without_Rounding`) above is retained unchanged; this `⊨[R]` face
restates the same global-cumsum content on the streaming emit skin, for
every `R` at once (at the `.real` grid the two faces carry the same exact
cell). Both faces are kept per the rounding-as-default doctrine. -/
```
</details>

**Statement:**
```lean
specification chunk_cumsum_scalar_io_correctness (R : RoundingModel)
    (S O : RegionName) (NT BT : Nat) (hSO : O ≠ S) (hBT : 0 < BT) :
    chunkCumsumKernelIO S O NT BT ⊨[R]
      fun _ _ xs t j => ccStreamSpec NT BT xs t j
```

**Assumptions / layout contracts:**
- `hSO : O ≠ S`
- `hBT : 0 < BT`

**Closed-form spec defs (transitive):** `chunkCumsumKernelIO`, `ccStreamSpec`, `chunk_cumsum_scalar_surface`, `ccNumChunks`

<details><summary><code>chunkCumsumKernelIO</code></summary>

```
/-- **Streaming IO signature** of `chunk_cumsum_scalar_surface` on the
single-stream per-step emit skin (S3: in-loop store). Step `t` (the loop
counter `i_t`) reads the `BT`-lane `s` chunk through the block pointer `p_s`
and stores the `BT`-lane output chunk through `p_o` at the **`.real`** grid
(`outDType` default — the lowered store is `Stmt.store TileDType.real`, so
the per-step stores carry no quantization event). The windows transcribe the
kernel's block-pointer arithmetic verbatim (base `i_bh·T`, stride `1`,
offsets `i_t·BT`, block shape `BT`):

* `read1` step `t`, lane `j`: `pid₀·T + (t·BT + j)`.
* `write` step `t`, lane `j`: `pid₀·T + (t·BT + j)`.

Both masks are the block pointers' `boundary_check=(0,)` guard,
`t·BT + j < T`. The second pid is unused (the launch grid is 1-D). -/
```
```lean
def chunkCumsumKernelIO (S O : RegionName) (NT BT : Nat) :
    StreamEmitMasked2DKernelIO₁ where
  kernel := chunk_cumsum_scalar_surface S O NT BT
  inp1 := S
  out := O
  T := ccNumChunks NT BT
  B1 := BT
  C := BT
  read1 := fun p₀ _ t j => p₀ * NT + (t.val * BT + j.val)
  write := fun p₀ _ t j => p₀ * NT + (t.val * BT + j.val)
  mask1 := fun _ _ t j => t.val * BT + j.val < NT
  writeMask := fun _ _ t j => t.val * BT + j.val < NT
```
</details>

<details><summary><code>ccStreamSpec</code></summary>

```
/-- The stream-level global-cumsum spec (the genre's *scan* shape): output
window `(t, j)` holds the **guarded prefix sum** of the whole curried stream
— every stream cell `(u, e)` whose flat index `u·BT + e` is at most the
window's flat index `t·BT + j` and lies inside the row (`< T`). The guard is
the skin's `mask1`, so the spec never reads an unmasked lane (the contract
pins `xs` only where `mask1` holds). -/
```
```lean
noncomputable def ccStreamSpec (NT BT : Nat)
    (xs : Fin (ccNumChunks NT BT) → Fin BT → ℝ)
    (t : Fin (ccNumChunks NT BT)) (j : Fin BT) : ℝ :=
  ∑ u : Fin (ccNumChunks NT BT), ∑ e : Fin BT,
    if u.val * BT + e.val ≤ t.val * BT + j.val ∧ u.val * BT + e.val < NT then
      xs u e else 0
```
</details>

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

<details><summary><code>ccNumChunks</code></summary>

```
/-- Trip count of `for i_t in range(0, tl.cdiv(T, BT), 1)`: `⌈T / BT⌉`. -/
```
```lean
def ccNumChunks (NT BT : Nat) : Nat := (NT + BT - 1) / BT
```
</details>

## Also present (pinned special-case summaries)
- `chunk_cumsum_scalar_store_slice_compute_correct`
- `chunk_cumsum_scalar_cumsum_slice_compute_correct`
- `chunk_cumsum_scalar_single_block_surface_compute_correct`
