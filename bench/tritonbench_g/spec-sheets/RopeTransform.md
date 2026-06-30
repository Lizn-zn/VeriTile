# Spec sheet — `bench/tritonbench_g/rope_transform/RopeTransform.lean`

**Python source:** `bench/tritonbench_g/rope_transform/rope_transform.py`

## Public theorem: `rope_transform_output_summary_general`

<details><summary>docstring</summary>

```
/-- **Dimension-general public Python forward summary for `rope_transform.py`.**
For arbitrary symbolic
row strides, sequence length, head counts, head dim, and `next_power_of_2`
padding, the full `BACKWARD_PASS = false` surface lowers to the algorithm layer
and each of the four Python-observable forward stores — Q/K first and second
halves — reads back, on every active lane, to the genuine rotary closed form
(`ropeForwardKernel{Q0,Q1,K0,K1}Spec`), NOT the kernel's own executed value.
The general building-block lemmas (`rope_transform_q0/q1/k0/k1_forward_correct`)
are already symbolic, so the bundle is a plain anonymous constructor. -/
```
</details>

**Statement:**
```lean
theorem rope_transform_output_summary_general
    (Q K COS SIN : RegionName)
    (q_row_stride k_row_stride cos_row_stride sin_row_stride
      sl bs n_qh n_kh hd pad_n_qh pad_n_kh pad_hd BLOCK_SIZE : Nat)
    (s : BlockState) (hundef : ∀ rg o, s.undef rg o = 0) (hqk : Q ≠ K) :
    (∃ alg, (triton_rope_surface Q K COS SIN
      q_row_stride k_row_stride cos_row_stride sin_row_stride
      sl bs n_qh n_kh hd pad_n_qh pad_n_kh pad_hd BLOCK_SIZE
      Bool.false).toAlgorithm? = Except.ok alg) ∧
    (∀ idx : TileIndex [pad_n_qh, pad_hd/2],
      activeQFull (pad_n_qh
```

**Assumptions / layout contracts:**
- `hundef : ∀ rg o, s.undef rg o = 0`
- `hqk : Q ≠ K`

**Closed-form spec defs (transitive):** `triton_rope_surface`, `activeQFull`

<details><summary><code>triton_rope_surface</code></summary>

```
/-- Faithful transcription of `rope_transform.py`'s `_triton_rope`. -/
```
```lean
def triton_rope_surface
    (q_ptr k_ptr cos sin : RegionName)
    (q_row_stride k_row_stride cos_row_stride sin_row_stride
      sl bs n_qh n_kh hd pad_n_qh pad_n_kh pad_hd BLOCK_SIZE : Nat)
    (BACKWARD_PASS : Bool) :
    ComputeKernel := triton {
    pid = tl.program_id(0)
    q_ptr = q_ptr + pid * $(q_row_stride)
    k_ptr = k_ptr + pid * $(k_row_stride)
    cos_row_idx = pid % $(sl)
    cos = cos + cos_row_idx * $(cos_row_stride)
    sin = sin + cos_row_idx * $(sin_row_stride)
    cos_offsets = tl.arange(0, $(pad_hd) // $(2))
    cos_mask = cos_offsets < $(hd) // $(2)
    cos_row = tl.load(cos + cos_offsets, mask=cos_mask, other=0)
    sin_row = tl.load(sin + cos_offsets, mask=cos_mask, other=0)
    first_half_q_offsets = tl.arange(0, $(pad_n_qh))[:, None] * $(hd) +
      tl.arange(0, $(pad_hd) // $(2))[None, :]
    first_half_k_offsets = tl.arange(0, $(pad_n_kh))[:, None] * $(hd) +
      tl.arange(0, $(pad_hd) // $(2))[None, :]
    first_q_mask = (tl.arange(0, $(pad_n_qh))[:, None] < $(n_qh)) &
      (tl.arange(0, $(pad_hd) // $(2))[None, :] < $(hd) // $(2))
    first_k_mask = (tl.arange(0, $(pad_n_kh))[:, None] < $(n_kh)) &
      (tl.arange(0, $(pad_hd) // $(2))[None, :] < $(hd) // $(2))
    q_tile_1 = tl.load(q_ptr + first_half_q_offsets, mask=first_q_mask,
      other=0).to(sin_row.dtype)
    k_tile_1 = tl.load(k_ptr + first_half_k_offsets, mask=first_k_mask,
      other=0).to(sin_row.dtype)
    second_half_q_offsets = first_half_q_offsets + $(hd) // $(2)
    second_half_k_offsets = first_half_k_offsets + $(hd) // $(2)
    second_q_mask = first_q_mask
    second_k_mask = first_k_mask
    q_tile_2 = tl.load(q_ptr + second_half_q_offsets, mask=second_q_mask,
      other=0).to(sin_row.dtype)
    k_tile_2 = tl.load(k_ptr + second_half_k_offsets, mask=second_k_mask,
      other=0).to(sin_row.dtype)
    if not BACKWARD_PASS {
    new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
    tl.store(q_ptr + first_half_q_offsets, new_q_tile_1, mask=first_q_mask)
    new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row
    tl.store(q_ptr + second_half_q_offsets, new_q_tile_2, mask=second_q_mask)
    new_k_tile_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
    tl.store(k_ptr + first_half_k_offsets, new_k_tile_1, mask=first_k_mask)
    new_k_tile_2 = k_tile_2 * cos_row + k_tile_1 * sin_row
    tl.store(k_ptr + second_half_k_offsets, new_k_tile_2, mask=second_k_mask)
    } else {
    new_q_tile_1 = q_tile_1 * cos_row + q_tile_2 * sin_row
    tl.store(q_ptr + first_half_q_offsets, new_q_tile_1, mask=first_q_mask)
    new_q_tile_2 = q_tile_2 * cos_row - q_tile_1 * sin_row
    tl.store(q_ptr + second_half_q_offsets, new_q_tile_2, mask=second_q_mask)
    new_k_tile_1 = k_tile_1 * cos_row + k_tile_2 * sin_row
    tl.store(k_ptr + first_half_k_offsets, new_k_tile_1, mask=first_k_mask)
    new_k_tile_2 = k_tile_2 * cos_row - k_tile_1 * sin_row
    tl.store(k_ptr + second_half_k_offsets, new_k_tile_2, mask=second_k_mask)
    }
}
```
</details>

<details><summary><code>activeQFull</code></summary>

```
/-- Active predicate for the Q-side stores of the full kernel. -/
```
```lean
def activeQFull (n_qh hd : Nat)
    (idx : TileIndex [pad_n_qh, pad_hd_half]) : Prop :=
  idx.1.val < n_qh ∧ idx.2.1.val < hd / 2
```
</details>

## Also present (pinned special-case summaries)
- `rope_transform_q0_head_compute_correct`
- `rope_transform_q1_head_compute_correct`
- `rope_transform_k0_head_compute_correct`
- `rope_transform_k1_head_compute_correct`
- `rope_kernel_o0o1_row_o0_compute_correct`
- `rope_kernel_o0o1_row_o1_compute_correct`
- `rope_kernel_o0o1_row_all_outputs_compute_correct`
