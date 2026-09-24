# Spec sheet — `bench/tritonbench_g/sgmv_expand_slice/SgmvExpandSlice.lean`

**Python source:** `bench/tritonbench_g/sgmv_expand_slice/sgmv_expand_slice.py`

## Public theorem: `sgmv_expand_slice_one_row_block_output_summary`

<details><summary>docstring</summary>

```
/-- **Public per-kernel output summary** (replaces the former slice/self-ref
proof-gap spec with a genuine contraction): the full SGMV expand-slice surface
(with the K-loop `tl.dot` accumulator and the masked store) lowers to the
algorithm layer **and** is compute-correct against the *genuine* rank-`K`
contraction `sgmvSpec` on every active output lane — under the
no-duplicate-destination hypothesis `hInj`. This is the closed-form GEMV
reference `Σ_{k<K} input·loraB`, not the kernel's own emitted value. -/
```
</details>

**Statement:**
```lean
specification sgmv_expand_slice_one_row_block_output_summary
    (input_ptr lora_ptr out_ptr : RegionName)
    (b_seq_start_loc seq_lens lora_indices : Region .nat) (s : BlockState)
    (N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride slice_offset BLOCK_M BLOCK_N BLOCK_K numKBlocks : Nat) (hBK : 0 < BLOCK_K)
    (hInj : Function.Injective (cOffset s b_seq_start_loc cm_stride cn_stride slice_offset BLOCK_M BLOCK_N))
    (hundef : ∀ rg o, s.undef rg o = 0) :
    (∃ alg, (sgmv_expand_slice_surface input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens
        lora_indices N (BLOCK_K * numKBlocks) xm_stride xk_stride l0_stride lora_k_stride
        lora_n_stride cm_stride cn_stride slice_offset BLOCK_M BLOCK_N BLOCK_K).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel
```

**Assumptions / layout contracts:**
- `hBK : 0 < BLOCK_K`
- `hInj : Function.Injective (cOffset s b_seq_start_loc cm_stride cn_stride slice_offset BLOCK_M BLOCK_N)`
- `hundef : ∀ rg o, s.undef rg o = 0`

**Closed-form spec defs (transitive):** `cOffset`, `sgmv_expand_slice_surface`, `seqStart`, `rowG`, `colG`

<details><summary><code>cOffset</code></summary>

```
/-- The output store address for tile lane `(i,j)`:
`(cur_seq_start + offset_m)·cm + (offset_n + slice_offset)·cn`. -/
```
```lean
def cOffset (s : BlockState) (b_seq_start_loc : Region .nat)
    (cm_stride cn_stride slice_offset BLOCK_M BLOCK_N : Nat) (idx : TileIndex [BLOCK_M, BLOCK_N]) : Nat :=
  (seqStart s b_seq_start_loc + rowG s BLOCK_M idx.1) * cm_stride
    + (colG s BLOCK_N idx.2.1 + slice_offset) * cn_stride
```
</details>

<details><summary><code>sgmv_expand_slice_surface</code></summary>

```
/-- Faithful transcription of `sgmv_expand_slice.py`'s `_sgmv_expand_slice_kernel`
core (the `ADD_INPUTS = false`, `CAST_TYPE = false`, `EVEN_K` numeric path).

Program ids: `pid_m` (axis 0), `pid_n` (axis 1), `cur_batch` (axis 2) — the host
grid linearizes `(pid, cur_batch) ↦ (pid_m, pid_n)`; that linearization and the
`pid_m·BLOCK_M > M` / `lora_index == -1` early returns are the trusted boundary.

The metadata loads, the `ram = offset_m % M` / `rbn = offset_n % N` gathers
(`tl.max_contiguous`/`tl.multiple_of` are layout hints erased to the same value),
the K-block `tl.dot` accumulation loop, and the final masked store are all
transcribed. -/
```
```lean
def sgmv_expand_slice_surface
    (input_ptr lora_ptr out_ptr : RegionName)
    (b_seq_start_loc seq_lens lora_indices : Region .nat)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride slice_offset BLOCK_M BLOCK_N BLOCK_K : Nat) :
    ComputeKernel := triton {
  pid_m = tl.program_id(0)
  pid_n = tl.program_id(1)
  cur_batch = tl.program_id(2)
  M = tl.load(seq_lens + cur_batch)
  cur_seq_start = tl.load(b_seq_start_loc + cur_batch)
  lora_index = tl.load(lora_indices + cur_batch)
  offset_m = tl.arange(0, $(BLOCK_M)) + pid_m * $(BLOCK_M)
  offset_n = tl.arange(0, $(BLOCK_N)) + pid_n * $(BLOCK_N)
  offset_k = tl.arange(0, $(BLOCK_K))
  ram = offset_m % M
  rbn = offset_n % $(N)
  a_ptr = input_ptr + cur_seq_start * $(xm_stride) +
    ram[:, None] * $(xm_stride) + offset_k[None, :] * $(xk_stride)
  b_ptr = lora_ptr + $(l0_stride) * lora_index +
    offset_k[:, None] * $(lora_n_stride) + rbn[None, :] * $(lora_k_stride)
  accumulator = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
  for kk in range($(0), $(K), $(BLOCK_K)) {
    tiled_a = tl.load(a_ptr)
    tiled_b = tl.load(b_ptr)
    accumulator += tl.dot(tiled_a, tiled_b)
    a_ptr += $(BLOCK_K) * $(xk_stride)
    b_ptr += $(BLOCK_K) * $(lora_n_stride)
  }
  offset_cm = cur_seq_start + offset_m
  offset_cn = offset_n + $(slice_offset)
  c_ptr = out_ptr + offset_cm[:, None] * $(cm_stride) +
    offset_cn[None, :] * $(cn_stride)
  c_mask = (offset_m[:, None] < M) & (offset_n[None, :] < $(N))
  tl.store(c_ptr, accumulator, mask=c_mask)
}
```
</details>

<details><summary><code>seqStart</code></summary>

```
/-- `cur_seq_start = b_seq_start_loc[cur_batch]` (this program's token offset). -/
```
```lean
def seqStart (s : BlockState) (b_seq_start_loc : Region .nat) : Nat :=
  s.readMemValue .nat b_seq_start_loc.cast (s.pids 2)
```
</details>

<details><summary><code>rowG</code></summary>

```
/-- Global row index `offset_m = pid_m·BLOCK_M + i` of tile lane `i`. -/
```
```lean
def rowG (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val
```
</details>

<details><summary><code>colG</code></summary>

```
/-- Global col index `offset_n = pid_n·BLOCK_N + j` of tile lane `j`. -/
```
```lean
def colG (s : BlockState) (BLOCK_N : Nat) (j : Fin BLOCK_N) : Nat :=
  s.pids 1 * BLOCK_N + j.val
```
</details>

## Public theorem: `sgmv_expand_slice_io_correctness`

<details><summary>docstring</summary>

```
/-- **The `⊨[R]` streaming metadata headline (wave-5 S1 fold genre).** For
every rounding model `R`, the faithful `sgmv_expand_slice` surface
implements, on its `StreamMetaMasked3DKernelIO₂` signature, the **ideal ℝ
SGMV fold** over the streamed tiles: every write-active output lane
`l = (i, n)` holds `∑ t, ∑ e, a-tile[t](i,e) · b-tile[t](e,n)` — exact real
arithmetic; the slot vector `m` enters the spec only through the gather
geometry of the windows (`% m 0` row gather, `m 1` token offset, `m 2`
adapter bank) and the sentinel-shaped `writeMask`. The kernel has **no
rounding events** (`.nat` metadata loads, `.real` loads/dot, `.real` masked
terminal store), so the skin's boundary quantization degenerates: the
readback's `R.round .real` is the identity (`round_real`), and the `.real`
store is exact under `execR R` (`RoundingModel.storeValue_real`).

Layer map: the prologue and the whole K-loop are cast-free, so under
`execR R` they collapse verbatim onto the exact stepper and the proven
`sgmv_preLoop` / `sgmv_step` / `forRange_inv` stack above is reused
unchanged; only the masked terminal store is re-proved on the `R` side
(`sgmv_postLoopR`).

Both hypotheses are truth-forced:

* `hBK : 0 < BLOCK_K` — the surface's K-loop steps by `BLOCK_K`
  (`range(0, K, BLOCK_K)`); at `BLOCK_K = 0` the loop never advances and the
  block-index arithmetic `i / BLOCK_K` is meaningless — the same hypothesis
  the exact headline carries. It holds for every real launch.
* `hInj` — output-offset injectivity of the write window
  `(start + pid₀·BLOCK_M + i)·cm + ((pid₁·BLOCK_N + n) + slice)·cn`,
  the pid/slot-parametrized spelling of the exact headline's open side
  condition `hInj : Function.Injective (cOffset …)` (the skin quantifies the
  launch state internally, so the state-indexed spelling is not expressible
  here); with colliding output lanes the per-lane readback would be
  last-writer-wins and the statement false. As in the exact headline it is
  carried as an open side condition, not discharged.

Inherited modeling boundary (unchanged from the exact surface, see the file
docstring): `lora_indices` is Python `int32` with a `-1` skip sentinel; this
port's surface has always erased it to `.nat` and delegated the two host
guards (`pid_m·BLOCK_M > M` early return and the `-1` sentinel skip) to the
trusted launch boundary, so the slot channel here is `.nat` and the sentinel
skip is *not* expressed as an empty `writeMask` — the statement covers the
programs the host actually launches, exactly as the exact headline does.

Relation to the exact surface: the exact headline
`sgmv_expand_slice_one_row_block_output_summary`
(`Realizes_without_Rounding`) above is retained unchanged; this `⊨[R]` face
restates the same SGMV contraction on the streaming metadata skin, for every
`R` at once (at the `.real` grid the two faces carry the same exact cell).
Both faces are kept per the rounding-as-default doctrine. -/
```
</details>

**Statement:**
```lean
specification sgmv_expand_slice_io_correctness (R : RoundingModel)
    (input_ptr lora_ptr out_ptr : RegionName)
    (b_seq_start_loc seq_lens lora_indices : Region .nat)
    (N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride slice_offset BLOCK_M BLOCK_N BLOCK_K numKBlocks : Nat)
    (hBK : 0 < BLOCK_K)
    (hInj : ∀ pid₀ pid₁ start : Nat,
      Function.Injective (fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
        (start + (pid₀ * BLOCK_M + idx.1.val)) * cm_stride
          + ((pid₁ * BLOCK_N + idx.2.1.val) + slice_offset) * cn_stride)) :
    sgmvExpandSliceIO input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
        N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
        cm_stride cn_stride slice_offset BLOCK_M BLOCK_N BLOCK_K numKBlocks ⊨[R]
      fun _ _ _ _ xs ys l =>
        ∑ t : Fin numKBlocks, ∑ e : Fin BLOCK_K,
          xs t (aLane BLOCK_M BLOCK_N BLOCK_K l e)
            * ys t (bLane BLOCK_M BLOCK_N BLOCK_K l e)
```

**Assumptions / layout contracts:**
- `hBK : 0 < BLOCK_K`

**Closed-form spec defs (transitive):** `sgmvExpandSliceIO`, `aLane`, `bLane`, `sgmv_expand_slice_surface`, `sgmvMetaBuf`

<details><summary><code>sgmvExpandSliceIO</code></summary>

```
/-- **Streaming metadata IO signature** of `sgmv_expand_slice` on the
metadata-parametrized two-stream fold skin (S1: fold + terminal masked
store, 3-D pid grid). The three `.nat` metadata slots are the kernel's own
per-batch scalars, all loaded at cell `cur_batch = pid₂` of their own
regions (`sgmvMetaBuf`; no chained slot indirection). Step `c` of the K-loop
reads the `[BLOCK_M, BLOCK_K]` `input`-tile and the `[BLOCK_K, BLOCK_N]`
`loraB`-tile **unmasked** (the transcribed path is the `EVEN_K` shape,
`K = BLOCK_K · numKBlocks` baked into the loop structure, so `mask1`/`mask2`
are `True`); after the loop one `[BLOCK_M, BLOCK_N]` output tile is
masked-stored at the **`.real`** grid (`outDType` default — the store is an
untyped `tl.store` at `.real`, no quantization event). The windows
transcribe the kernel's pointer arithmetic exactly, with the loaded slot
vector `m` in place of the in-state metadata reads
(`m 0 = M`, `m 1 = cur_seq_start`, `m 2 = lora_index`):

* `read1` lane `j = (i, e)` (row-major over `[BLOCK_M, BLOCK_K]`), step `t`:
  `m 1·xm + ((pid₀·BLOCK_M + i) % m 0)·xm + (t·BLOCK_K + e)·xk` — the
  invariant's `a_ptr` cell after `t` advances (the `ram = offset_m % M`
  gather geometry, slot-parametrized).
* `read2` lane `j = (e, n)` (row-major over `[BLOCK_K, BLOCK_N]`), step `t`:
  `l0·m 2 + (t·BLOCK_K + e)·ln + ((pid₁·BLOCK_N + n) % N)·lk` — the `b_ptr`
  cell (the `rbn = offset_n % N` gather and the `lora_index` bank select).
* `write` lane `j = (i, n)`:
  `(m 1 + (pid₀·BLOCK_M + i))·cm + ((pid₁·BLOCK_N + n) + slice_offset)·cn`
  — the kernel's `c_ptr` (= `cOffset` in pid/slot form).
* `writeMask` lane `j = (i, n)`:
  `pid₀·BLOCK_M + i < m 0 ∧ pid₁·BLOCK_N + n < N` — the kernel's `c_mask`
  (`offset_m < M & offset_n < N`), slot-parametrized. -/
```
```lean
def sgmvExpandSliceIO (input_ptr lora_ptr out_ptr : RegionName)
    (b_seq_start_loc seq_lens lora_indices : Region .nat)
    (N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride slice_offset BLOCK_M BLOCK_N BLOCK_K numKBlocks : Nat) :
    StreamMetaMasked3DKernelIO₂ where
  kernel := sgmv_expand_slice_surface input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens
    lora_indices N (BLOCK_K * numKBlocks) xm_stride xk_stride l0_stride lora_k_stride
    lora_n_stride cm_stride cn_stride slice_offset BLOCK_M BLOCK_N BLOCK_K
  inp1 := input_ptr
  inp2 := lora_ptr
  out := out_ptr
  nMeta := 3
  sty := fun _ => ChanTy.nat
  mbuf := sgmvMetaBuf b_seq_start_loc seq_lens lora_indices
  mwin := fun _ _ _ pid₂ => pid₂
  T := numKBlocks
  B1 := BLOCK_M * BLOCK_K
  B2 := BLOCK_K * BLOCK_N
  C := BLOCK_M * BLOCK_N
  read1 := fun pid₀ _ _ m t j =>
    m (⟨1, by omega⟩ : Fin 3) * xm_stride
      + ((pid₀ * BLOCK_M + j.val / BLOCK_K) % m (⟨0, by omega⟩ : Fin 3)) * xm_stride
      + (t.val * BLOCK_K + j.val % BLOCK_K) * xk_stride
  read2 := fun _ pid₁ _ m t j =>
    l0_stride * m (⟨2, by omega⟩ : Fin 3)
      + (t.val * BLOCK_K + j.val / BLOCK_N) * lora_n_stride
      + ((pid₁ * BLOCK_N + j.val % BLOCK_N) % N) * lora_k_stride
  write := fun pid₀ pid₁ _ m j =>
    (m (⟨1, by omega⟩ : Fin 3) + (pid₀ * BLOCK_M + j.val / BLOCK_N)) * cm_stride
      + ((pid₁ * BLOCK_N + j.val % BLOCK_N) + slice_offset) * cn_stride
  mask1 := fun _ _ _ _ _ _ => True
  mask2 := fun _ _ _ _ _ _ => True
  writeMask := fun pid₀ pid₁ _ m j =>
    pid₀ * BLOCK_M + j.val / BLOCK_N < m (⟨0, by omega⟩ : Fin 3)
      ∧ pid₁ * BLOCK_N + j.val % BLOCK_N < N
```
</details>

<details><summary><code>aLane</code></summary>

```
/-- The `input`-stream lane feeding output lane `l` at inner key `e`: the row
of `l` (row-major over the `[BLOCK_M, BLOCK_N]` output tile) paired with `e`
over the `[BLOCK_M, BLOCK_K]` per-step `a`-tile, via the shared `Lane2D`
bridge. -/
```
```lean
def aLane (BLOCK_M BLOCK_N BLOCK_K : Nat) (l : Fin (BLOCK_M * BLOCK_N))
    (e : Fin BLOCK_K) : Fin (BLOCK_M * BLOCK_K) :=
  Lane2D.encode ((Lane2D.decode l).1, e, PUnit.unit)
```
</details>

<details><summary><code>bLane</code></summary>

```
/-- The `loraB`-stream lane feeding output lane `l` at inner key `e`: `e`
paired with the column of `l` over the `[BLOCK_K, BLOCK_N]` per-step
`b`-tile. -/
```
```lean
def bLane (BLOCK_M BLOCK_N BLOCK_K : Nat) (l : Fin (BLOCK_M * BLOCK_N))
    (e : Fin BLOCK_K) : Fin (BLOCK_K * BLOCK_N) :=
  Lane2D.encode (e, (Lane2D.decode l).2.1, PUnit.unit)
```
</details>

<details><summary><code>sgmv_expand_slice_surface</code></summary>

```
/-- Faithful transcription of `sgmv_expand_slice.py`'s `_sgmv_expand_slice_kernel`
core (the `ADD_INPUTS = false`, `CAST_TYPE = false`, `EVEN_K` numeric path).

Program ids: `pid_m` (axis 0), `pid_n` (axis 1), `cur_batch` (axis 2) — the host
grid linearizes `(pid, cur_batch) ↦ (pid_m, pid_n)`; that linearization and the
`pid_m·BLOCK_M > M` / `lora_index == -1` early returns are the trusted boundary.

The metadata loads, the `ram = offset_m % M` / `rbn = offset_n % N` gathers
(`tl.max_contiguous`/`tl.multiple_of` are layout hints erased to the same value),
the K-block `tl.dot` accumulation loop, and the final masked store are all
transcribed. -/
```
```lean
def sgmv_expand_slice_surface
    (input_ptr lora_ptr out_ptr : RegionName)
    (b_seq_start_loc seq_lens lora_indices : Region .nat)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride slice_offset BLOCK_M BLOCK_N BLOCK_K : Nat) :
    ComputeKernel := triton {
  pid_m = tl.program_id(0)
  pid_n = tl.program_id(1)
  cur_batch = tl.program_id(2)
  M = tl.load(seq_lens + cur_batch)
  cur_seq_start = tl.load(b_seq_start_loc + cur_batch)
  lora_index = tl.load(lora_indices + cur_batch)
  offset_m = tl.arange(0, $(BLOCK_M)) + pid_m * $(BLOCK_M)
  offset_n = tl.arange(0, $(BLOCK_N)) + pid_n * $(BLOCK_N)
  offset_k = tl.arange(0, $(BLOCK_K))
  ram = offset_m % M
  rbn = offset_n % $(N)
  a_ptr = input_ptr + cur_seq_start * $(xm_stride) +
    ram[:, None] * $(xm_stride) + offset_k[None, :] * $(xk_stride)
  b_ptr = lora_ptr + $(l0_stride) * lora_index +
    offset_k[:, None] * $(lora_n_stride) + rbn[None, :] * $(lora_k_stride)
  accumulator = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
  for kk in range($(0), $(K), $(BLOCK_K)) {
    tiled_a = tl.load(a_ptr)
    tiled_b = tl.load(b_ptr)
    accumulator += tl.dot(tiled_a, tiled_b)
    a_ptr += $(BLOCK_K) * $(xk_stride)
    b_ptr += $(BLOCK_K) * $(lora_n_stride)
  }
  offset_cm = cur_seq_start + offset_m
  offset_cn = offset_n + $(slice_offset)
  c_ptr = out_ptr + offset_cm[:, None] * $(cm_stride) +
    offset_cn[None, :] * $(cn_stride)
  c_mask = (offset_m[:, None] < M) & (offset_n[None, :] < $(N))
  tl.store(c_ptr, accumulator, mask=c_mask)
}
```
</details>

<details><summary><code>sgmvMetaBuf</code></summary>

```
/-- Slot-region table of the three per-batch metadata slots, in the kernel's
own load order: slot `0` = `seq_lens` (the sequence length `M`), slot `1` =
`b_seq_start_loc` (the token offset `cur_seq_start`), slot `2` =
`lora_indices` (the adapter slot `lora_index`). A shared def, never an
inline `match` in a window/spec position. -/
```
```lean
def sgmvMetaBuf (b_seq_start_loc seq_lens lora_indices : Region .nat) : Fin 3 → RegionName
  | ⟨0, _⟩ => seq_lens.cast
  | ⟨1, _⟩ => b_seq_start_loc.cast
  | ⟨_ + 2, _⟩ => lora_indices.cast
```
</details>

## Also present (pinned special-case summaries)
- `sgmv_expand_slice_closed_form_correct`
