# Spec sheet — `bench/tritonbench_g/cache_transform_triton/CacheTransformTriton.lean`

**Python source:** `bench/tritonbench_g/cache_transform_triton/cache_transform_triton.py`

## Public theorem: `decoding_cache_correctness`

**Statement:**
```lean
specification decoding_cache_correctness
    (cos_cache sin_cache : RegionName) (lengths : Region .nat)
    (cos_output sin_output : RegionName)
    (cache_stride hidden_stride HIDDEN_DIM NUM_SEQS BLOCK_H : Nat)
    (hRegion : cos_output ≠ sin_output) :
    decodingCacheIO cos_cache sin_cache lengths cos_output sin_output
        cache_stride hidden_stride HIDDEN_DIM NUM_SEQS BLOCK_H
      ⊨ fun _ _ _ xs ys => (xs, ys)
```

**Assumptions / layout contracts:**
- `hRegion : cos_output ≠ sin_output`

**Closed-form spec defs (transitive):** `decodingCacheIO`, `decoding_cache_one_seq_block`

<details><summary><code>decodingCacheIO</code></summary>

```lean
def decodingCacheIO
    (cos_cache sin_cache : RegionName) (lengths : Region .nat)
    (cos_output sin_output : RegionName)
    (cache_stride hidden_stride HIDDEN_DIM NUM_SEQS BLOCK_H : Nat) :
    GatherMasked2DKernelIO₂ₓ₂ where
  kernel := decoding_cache_one_seq_block cos_cache sin_cache lengths cos_output
    sin_output cache_stride hidden_stride HIDDEN_DIM NUM_SEQS BLOCK_H
  in1 := cos_cache
  in2 := sin_cache
  idxbuf := lengths
  out1 := cos_output
  out2 := sin_output
  B := BLOCK_H
  N := 1
  readx := fun pid₀ _ _ => pid₀
  read := fun _ pid₁ ids i =>
    ids 0 * cache_stride + (pid₁ * BLOCK_H + i.val) * hidden_stride
  write1 := fun pid₀ pid₁ _ i =>
    pid₀ * cache_stride + (pid₁ * BLOCK_H + i.val) * hidden_stride
  write2 := fun pid₀ pid₁ _ i =>
    pid₀ * cache_stride + (pid₁ * BLOCK_H + i.val) * hidden_stride
  mask := fun pid₀ _ _ => pid₀ < NUM_SEQS
  readMask := fun pid₀ pid₁ _ i =>
    pid₀ < NUM_SEQS ∧ pid₁ * BLOCK_H + i.val < HIDDEN_DIM
```
</details>

<details><summary><code>decoding_cache_one_seq_block</code></summary>

```
/-- Proof-oriented one-sequence, one-hidden-block slice of
`cache_transform_triton.py`'s `decoding_cache_kernel`.

This models the decoding branch: load the source cache row from `lengths[seq]`
and copy the same hidden block from cos/sin caches into cos/sin outputs. -/
```
```lean
def decoding_cache_one_seq_block
    (cos_cache sin_cache : RegionName) (lengths : Region .nat) (cos_output sin_output : RegionName)
    (cache_stride hidden_stride HIDDEN_DIM NUM_SEQS BLOCK_H : Nat) :
    ComputeKernel := triton {
  seq = tl.program_id(0)
  hid_block = tl.program_id(1)
  offs = tl.arange(0, $(BLOCK_H))
  hid = hid_block * $(BLOCK_H) + offs
  ori_seq_idx = tl.load(lengths + seq, mask=seq < $(NUM_SEQS),
    other=$(0))
  cos_part = tl.load(cos_cache + ori_seq_idx * $(cache_stride) +
      hid * $(hidden_stride),
    mask=(seq < $(NUM_SEQS)) and (hid < $(HIDDEN_DIM)), other=0.0)
  sin_part = tl.load(sin_cache + ori_seq_idx * $(cache_stride) +
      hid * $(hidden_stride),
    mask=(seq < $(NUM_SEQS)) and (hid < $(HIDDEN_DIM)), other=0.0)
  tl.store(cos_output + seq * $(cache_stride) + hid * $(hidden_stride),
    cos_part, mask=(seq < $(NUM_SEQS)) and (hid < $(HIDDEN_DIM)))
  tl.store(sin_output + seq * $(cache_stride) + hid * $(hidden_stride),
    sin_part, mask=(seq < $(NUM_SEQS)) and (hid < $(HIDDEN_DIM)))
}
```
</details>

## Public theorem: `prefill_cache_correctness`

**Statement:**
```lean
specification prefill_cache_correctness
    (cos_cache sin_cache : RegionName) (cumsum_lengths : Region .nat)
    (cos_output sin_output : RegionName)
    (cache_stride hidden_stride total_length HIDDEN_DIM N_ELEMENTS BLOCK_SIZE : Nat)
    (hRegion : cos_output ≠ sin_output) (hN : 0 < N_ELEMENTS) :
    prefillCacheIO cos_cache sin_cache cumsum_lengths cos_output sin_output
        cache_stride hidden_stride total_length HIDDEN_DIM N_ELEMENTS BLOCK_SIZE
      ⊨ fun _ _ _ xs ys => (xs, ys)
```

**Assumptions / layout contracts:**
- `hRegion : cos_output ≠ sin_output`
- `hN : 0 < N_ELEMENTS`

**Closed-form spec defs (transitive):** `prefillCacheIO`, `prefill_cache_kernel`, `prefillOriSeqIdxOfIds`

<details><summary><code>prefillCacheIO</code></summary>

```lean
noncomputable def prefillCacheIO
    (cos_cache sin_cache : RegionName) (cumsum_lengths : Region .nat)
    (cos_output sin_output : RegionName)
    (cache_stride hidden_stride total_length HIDDEN_DIM N_ELEMENTS BLOCK_SIZE : Nat) :
    GatherMasked2DKernelIO₂ₓ₂ where
  kernel := prefill_cache_kernel cos_cache sin_cache cumsum_lengths cos_output
    sin_output cache_stride hidden_stride total_length HIDDEN_DIM N_ELEMENTS BLOCK_SIZE
  in1 := cos_cache
  in2 := sin_cache
  idxbuf := cumsum_lengths
  out1 := cos_output
  out2 := sin_output
  B := HIDDEN_DIM
  N := N_ELEMENTS
  readx := fun _ _ j => j.val
  read := fun pid₀ pid₁ ids i =>
    prefillOriSeqIdxOfIds (pid₀ * BLOCK_SIZE + pid₁) N_ELEMENTS ids * cache_stride
      + i.val * hidden_stride
  write1 := fun pid₀ pid₁ _ i =>
    (pid₀ * BLOCK_SIZE + pid₁) * cache_stride + i.val * hidden_stride
  write2 := fun pid₀ pid₁ _ i =>
    (pid₀ * BLOCK_SIZE + pid₁) * cache_stride + i.val * hidden_stride
  mask := fun _ _ _ => True
  readMask := fun pid₀ pid₁ _ _ => pid₀ * BLOCK_SIZE + pid₁ < total_length
```
</details>

<details><summary><code>prefill_cache_kernel</code></summary>

```
/-- Faithful transcription of `cache_transform_triton.py`'s
`prefill_cache_kernel`. -/
```
```lean
def prefill_cache_kernel
    (cos_cache sin_cache : RegionName) (cumsum_lengths : Region .nat)
    (cos_output sin_output : RegionName)
    (cache_stride hidden_stride total_length HIDDEN_DIM N_ELEMENTS BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  idx0 = tl.program_id(axis=0)
  idx1 = tl.program_id(axis=1)
  idx = idx0 * $(BLOCK_SIZE) + idx1
  cumsum_lens = tl.load(cumsum_lengths + tl.arange(0, $(N_ELEMENTS)))
  ori_seq_idx = idx - tl.max(tl.where(cumsum_lens <= idx, cumsum_lens, $(0)))
  cos_cache_part = tl.load(
    cos_cache + ori_seq_idx * $(cache_stride) +
      tl.arange(0, $(HIDDEN_DIM)) * $(hidden_stride),
    mask=idx < $(total_length))
  sin_cache_part = tl.load(
    sin_cache + ori_seq_idx * $(cache_stride) +
      tl.arange(0, $(HIDDEN_DIM)) * $(hidden_stride),
    mask=idx < $(total_length))
  tl.store(
    cos_output + idx * $(cache_stride) +
      tl.arange(0, $(HIDDEN_DIM)) * $(hidden_stride),
    cos_cache_part,
    mask=idx < $(total_length))
  tl.store(
    sin_output + idx * $(cache_stride) +
      tl.arange(0, $(HIDDEN_DIM)) * $(hidden_stride),
    sin_cache_part,
    mask=idx < $(total_length))
}
```
</details>

<details><summary><code>prefillOriSeqIdxOfIds</code></summary>

```
/-- Pure source-row index for the prefill kernel, phrased over the loaded index
vector `ids` (rather than reading it back from memory as `prefillOriSeqIdx`).
Mirrors `prefillOriSeqIdx`'s `reduceMaxNatDrop` reduction so the `⊨` read window
stays total. -/
```
```lean
noncomputable def prefillOriSeqIdxOfIds
    (idx N_ELEMENTS : Nat) (ids : Fin N_ELEMENTS → Nat) : Nat :=
  let tile : Tile .nat [N_ELEMENTS] := ⟨fun i => ids i.1⟩
  let cond : Tile .bool [N_ELEMENTS] :=
    ⟨fun i => decide (tile.data i ≤ idx)⟩
  let zero : Tile .nat [N_ELEMENTS] := ⟨fun _ => 0⟩
  let masked : Tile .nat [N_ELEMENTS] := Tile.select cond tile zero
  let reduced : Option (Tile .nat (TileShape.eraseAxis [N_ELEMENTS] ⟨0, by simp⟩)) :=
    Tile.reduceMaxNatDrop (shape := [N_ELEMENTS]) ⟨0, by simp⟩ masked
  let maxv : Nat := match reduced with
    | some t => t.data PUnit.unit
    | none => 0
  idx - maxv
```
</details>

## Also present (pinned special-case summaries)
- `decoding_cache_kernel_compute_correct`
- `decoding_cache_kernel_sin_compute_correct`
- `decoding_cache_one_seq_block_compute_correct`
- `prefill_cache_cos_store_slice_compute_correct`
- `prefill_cache_sin_store_slice_compute_correct`
- `prefill_cache_kernel_compute_correct`
