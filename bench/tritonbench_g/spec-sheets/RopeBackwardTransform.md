# Spec sheet — `bench/tritonbench_g/rope_backward_transform/RopeBackwardTransform.lean`

**Python source:** `bench/tritonbench_g/rope_backward_transform/rope_backward_transform.py`

## Public theorem: `rope_backward_python_backward_output_summary`

<details><summary>docstring</summary>

```
/-- Public Python-path backward summary for `test_rope_backward` (genuine,
full-tile). The checked Python test fixes `batch_size = 2`, `seq_len = 4`,
`n_q_head = n_kv_head = 8`, `head_dim = 16`; after transpose+contiguous both
Q/K rows have stride `8 * 16 = 128` and cos/sin rows stride `8`. Every active
lane of each of the four `BACKWARD_PASS = true` Python-observable stores (Q/K
first and second halves) reads back to the genuine rotary-backward closed form,
NOT the kernel's own executed value, against the real `triton_rope_surface`
kernel (no head-slice gap). The host launch / `next_power_of_2` padding and
`q_ptr ≠ k_ptr` (distinct Q/K buffers) remain the trusted boundary. -/
```
</details>

**Statement:**
```lean
theorem rope_backward_python_backward_output_summary
    (Q K COS SIN : RegionName)
    (s : BlockState) (hundef : ∀ rg o, s.undef rg o = 0) (hqk : Q ≠ K) :
    (∃ alg, (triton_rope_surface Q K COS SIN
      128 128 8 8 4 2 8 8 16 8 8 16 8 Bool.true).toAlgorithm? =
        Except.ok alg) ∧
    (∀ idx : TileIndex [8, 16/2], activeQFull (pad_n_qh := 8) (pad_hd_half := 16/2) 8 16 idx →
      (match exec (triton_rope_surface Q K COS SIN 128 128 8 8 4 2 8 8 16 8 8 16 8 Bool.true) s with
        | some s' => s'.readMem Q (qFullFirstOffset (pad_n_qh := 8) (pad_hd_half := 16/2) s 128 16 idx)
        | none => (0.0 : ℝ)) =
        ropeBackwardKernelQ0Spec (pad_n_qh := 8) (pad_hd_half := 16/2) s Q COS SIN 128 4 8 8 16 idx) ∧
    (∀ idx : TileIndex [8, 16/2], activeQFull (pad_n_qh := 8) (pad_hd_half := 16/2) 8 16 idx →
      (match exec (triton_rope_surface Q K COS SIN 128 128 8 8 4 2 8 8 16 8 8 16 8 Bool.true) s with
        | some s' => s'.readMem Q (qFullSecondOffset (pad_n_qh := 8) (pad_hd_half := 16/2) s 128 16 idx)
        | none => (0.0 : ℝ)) =
        ropeBackwardKernelQ1Spec (pad_n_qh := 8) (pad_hd_half := 16/2) s Q COS SIN 128 4 8 8 16 idx) ∧
    (∀ idx : TileIndex [8, 16/2], activeKFull (pad_n_kh := 8) (pad_hd_half := 16/2) 8 16 idx →
      (match exec (triton_rope_surface Q K COS SIN 128 128 8 8 4 2 8 8 16 8 8 16 8 Bool.true) s with
        | some s' => s'.readMem K (kFullFirstOffset (pad_n_kh := 8) (pad_hd_half := 16/2) s 128 16 idx)
        | none => (0.0 : ℝ)) =
        ropeBackwardKernelK0Spec (pad_n_kh := 8) (pad_hd_half := 16/2) s K COS SIN 128 4 8 8 16 idx) ∧
    (∀ idx : TileIndex [8, 16/2], activeKFull (pad_n_kh := 8) (pad_hd_half := 16/2) 8 16 idx →
      (match exec (triton_rope_surface Q K COS SIN 128 128 8 8 4 2 8 8 16 8 8 16 8 Bool.true) s with
        | some s' => s'.readMem K (kFullSecondOffset (pad_n_kh := 8) (pad_hd_half := 16/2) s 128 16 idx)
        | none => (0.0 : ℝ)) =
        ropeBackwardKernelK1Spec (pad_n_kh := 8) (pad_hd_half := 16/2) s K COS SIN 128 4 8 8 16 idx)
```

**Assumptions / layout contracts:**
- `hundef : ∀ rg o, s.undef rg o = 0`
- `hqk : Q ≠ K`

**Closed-form spec defs (transitive):** `triton_rope_surface`, `activeQFull`, `qFullFirstOffset`, `ropeBackwardKernelQ0Spec`, `qFullSecondOffset`, `ropeBackwardKernelQ1Spec`, `activeKFull`, `kFullFirstOffset`, `ropeBackwardKernelK0Spec`, `kFullSecondOffset`, `ropeBackwardKernelK1Spec`, `cosFullFirstOffset`, `sinFullFirstOffset`

<details><summary><code>triton_rope_surface</code></summary>

```
/-- Faithful transcription of `rope_backward_transform.py`'s `_triton_rope`. -/
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
/-- Active predicate for the Q first-half store of `triton_rope_surface`. -/
```
```lean
def activeQFull (n_qh hd : Nat)
    (idx : TileIndex [pad_n_qh, pad_hd_half]) : Prop :=
  idx.1.val < n_qh ∧ idx.2.1.val < hd / 2
```
</details>

<details><summary><code>qFullFirstOffset</code></summary>

```
/-- Tile-level Q first-half offset in the full `triton_rope_surface` kernel.
Tile shape is `[pad_n_qh, pad_hd / 2]`. -/
```
```lean
def qFullFirstOffset
    (s : BlockState) (q_row_stride hd : Nat)
    (idx : TileIndex [pad_n_qh, pad_hd_half]) : Nat :=
  s.pids 0 * q_row_stride + idx.1.val * hd + idx.2.1.val
```
</details>

<details><summary><code>ropeBackwardKernelQ0Spec</code></summary>

```
/-- Specification for the full kernel's Q first-half output, under
`BACKWARD_PASS = true` (i.e. `new_q_tile_1 = q_tile_1 * cos + q_tile_2 * sin`). -/
```
```lean
noncomputable def ropeBackwardKernelQ0Spec
    (s : BlockState) (q_ptr cos sin : RegionName)
    (q_row_stride sl cos_row_stride sin_row_stride hd : Nat)
    (idx : TileIndex [pad_n_qh, pad_hd_half]) : ℝ :=
  s.readMem q_ptr (qFullFirstOffset s q_row_stride hd idx) *
    s.readMem cos (cosFullFirstOffset s sl cos_row_stride (idx.2.1, idx.2.2)) +
  s.readMem q_ptr (qFullSecondOffset s q_row_stride hd idx) *
    s.readMem sin (sinFullFirstOffset s sl sin_row_stride (idx.2.1, idx.2.2))
```
</details>

<details><summary><code>qFullSecondOffset</code></summary>

```
/-- Tile-level Q second-half offset (writes go here in store #2). -/
```
```lean
def qFullSecondOffset
    (s : BlockState) (q_row_stride hd : Nat)
    (idx : TileIndex [pad_n_qh, pad_hd_half]) : Nat :=
  s.pids 0 * q_row_stride + idx.1.val * hd + idx.2.1.val + hd / 2
```
</details>

<details><summary><code>ropeBackwardKernelQ1Spec</code></summary>

```
/-- Spec for the full kernel's Q second-half output under
`BACKWARD_PASS = true`: `new_q_tile_2 = q_tile_2 * cos - q_tile_1 * sin`. -/
```
```lean
noncomputable def ropeBackwardKernelQ1Spec
    (s : BlockState) (q_ptr cos sin : RegionName)
    (q_row_stride sl cos_row_stride sin_row_stride hd : Nat)
    (idx : TileIndex [pad_n_qh, pad_hd_half]) : ℝ :=
  s.readMem q_ptr (qFullSecondOffset s q_row_stride hd idx) *
    s.readMem cos (cosFullFirstOffset s sl cos_row_stride (idx.2.1, idx.2.2)) -
  s.readMem q_ptr (qFullFirstOffset s q_row_stride hd idx) *
    s.readMem sin (sinFullFirstOffset s sl sin_row_stride (idx.2.1, idx.2.2))
```
</details>

<details><summary><code>activeKFull</code></summary>

```
/-- Active predicate for the K-side stores. -/
```
```lean
def activeKFull (n_kh hd : Nat)
    (idx : TileIndex [pad_n_kh, pad_hd_half]) : Prop :=
  idx.1.val < n_kh ∧ idx.2.1.val < hd / 2
```
</details>

<details><summary><code>kFullFirstOffset</code></summary>

```
/-- Tile-level K first-half offset (target of store #3). -/
```
```lean
def kFullFirstOffset
    (s : BlockState) (k_row_stride hd : Nat)
    (idx : TileIndex [pad_n_kh, pad_hd_half]) : Nat :=
  s.pids 0 * k_row_stride + idx.1.val * hd + idx.2.1.val
```
</details>

<details><summary><code>ropeBackwardKernelK0Spec</code></summary>

```
/-- Spec for the full kernel's K first-half output, under
`BACKWARD_PASS = true`: `new_k_tile_1 = k_tile_1 * cos + k_tile_2 * sin`. -/
```
```lean
noncomputable def ropeBackwardKernelK0Spec
    (s : BlockState) (k_ptr cos sin : RegionName)
    (k_row_stride sl cos_row_stride sin_row_stride hd : Nat)
    (idx : TileIndex [pad_n_kh, pad_hd_half]) : ℝ :=
  s.readMem k_ptr (kFullFirstOffset s k_row_stride hd idx) *
    s.readMem cos (cosFullFirstOffset s sl cos_row_stride (idx.2.1, idx.2.2)) +
  s.readMem k_ptr (kFullSecondOffset s k_row_stride hd idx) *
    s.readMem sin (sinFullFirstOffset s sl sin_row_stride (idx.2.1, idx.2.2))
```
</details>

<details><summary><code>kFullSecondOffset</code></summary>

```
/-- Tile-level K second-half offset (target of store #4). -/
```
```lean
def kFullSecondOffset
    (s : BlockState) (k_row_stride hd : Nat)
    (idx : TileIndex [pad_n_kh, pad_hd_half]) : Nat :=
  s.pids 0 * k_row_stride + idx.1.val * hd + idx.2.1.val + hd / 2
```
</details>

<details><summary><code>ropeBackwardKernelK1Spec</code></summary>

```
/-- Spec for the full kernel's K second-half output, under
`BACKWARD_PASS = true`: `new_k_tile_2 = k_tile_2 * cos - k_tile_1 * sin`. -/
```
```lean
noncomputable def ropeBackwardKernelK1Spec
    (s : BlockState) (k_ptr cos sin : RegionName)
    (k_row_stride sl cos_row_stride sin_row_stride hd : Nat)
    (idx : TileIndex [pad_n_kh, pad_hd_half]) : ℝ :=
  s.readMem k_ptr (kFullSecondOffset s k_row_stride hd idx) *
    s.readMem cos (cosFullFirstOffset s sl cos_row_stride (idx.2.1, idx.2.2)) -
  s.readMem k_ptr (kFullFirstOffset s k_row_stride hd idx) *
    s.readMem sin (sinFullFirstOffset s sl sin_row_stride (idx.2.1, idx.2.2))
```
</details>

<details><summary><code>cosFullFirstOffset</code></summary>

```
/-- Cos offset for the full kernel's first-half Q store. -/
```
```lean
def cosFullFirstOffset
    (s : BlockState) (sl cos_row_stride : Nat)
    (idx : TileIndex [pad_hd_half]) : Nat :=
  s.pids 0 % sl * cos_row_stride + idx.1.val
```
</details>

<details><summary><code>sinFullFirstOffset</code></summary>

```
/-- Sin offset for the full kernel's first-half Q store. -/
```
```lean
def sinFullFirstOffset
    (s : BlockState) (sl sin_row_stride : Nat)
    (idx : TileIndex [pad_hd_half]) : Nat :=
  s.pids 0 % sl * sin_row_stride + idx.1.val
```
</details>

## Also present (pinned special-case summaries)
- `rope_backward_q0_head_compute_correct`
- `rope_backward_q1_head_compute_correct`
- `rope_backward_k0_head_compute_correct`
- `rope_backward_k1_head_compute_correct`
