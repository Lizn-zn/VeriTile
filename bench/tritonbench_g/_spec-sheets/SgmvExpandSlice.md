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
theorem sgmv_expand_slice_one_row_block_output_summary
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

## Also present (pinned special-case summaries)
- `sgmv_expand_slice_closed_form_correct`
