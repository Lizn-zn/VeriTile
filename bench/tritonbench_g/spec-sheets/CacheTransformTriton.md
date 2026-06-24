# Spec sheet — `bench/tritonbench_g/cache_transform_triton/CacheTransformTriton.lean`

**Python source:** `bench/tritonbench_g/cache_transform_triton/cache_transform_triton.py`

## Public theorem: `prefill_cache_kernel_output_summary`

<details><summary>docstring</summary>

```
/-- Per-kernel output summary for `prefill_cache_kernel`: the DSL surface lowers
to the algorithm layer (including the `where`/`max` source-row reduction), and
both masked cos/sin scatters are compute-correct — every active cell holds the
matching cos/sin cache cell at the derived source row, inactive cells are
preserved. -/
```
</details>

**Statement:**
```lean
theorem prefill_cache_kernel_output_summary
    (cos_cache sin_cache : RegionName) (cumsum_lengths : Region .nat)
    (cos_output sin_output : RegionName)
    (cache_stride hidden_stride total_length HIDDEN_DIM N_ELEMENTS BLOCK_SIZE : Nat)
    (s : BlockState)
    (hRegion : cos_output ≠ sin_output)
    (hN : 0 < N_ELEMENTS)
    (hOutInj : Function.Injective
      (fun i : Fin HIDDEN_DIM =>
        prefillOutOffset s cache_stride hidden_stride BLOCK_SIZE i)) :
    (∃ alg, (prefill_cache_kernel cos_cache sin_cache cumsum_lengths cos_output
        sin_output cache_stride hidden_stride total_length HIDDEN_DIM N_ELEMENTS
        BLOCK_SIZE).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := prefill_cache_kernel cos_cache sin_cache cumsum_lengths cos_output
        sin_output cache_stride hidden_stride total_length HIDDEN_DIM N_ELEMENTS
        BLOCK_SIZE)
      (initialState := s)
      (write := fun i : Sum (Fin HIDDEN_DIM) (Fin HIDDEN_DIM) =>
        match i with
        | .inl idx =>
            if prefillActive s total_length BLOCK_SIZE then
              some (cos_output, prefillOutOffset s cache_stride hidden_stride
                BLOCK_SIZE idx)
            else none
        | .inr idx =>
            if prefillActive s total_length BLOCK_SIZE then
              some (sin_output, prefillOutOffset s cache_stride hidden_stride
                BLOCK_SIZE idx)
            else none)
      (expected := fun i =>
        match i with
        | .inl idx =>
            s.readMem cos_cache
              (prefillCacheOffset s cumsum_lengths cache_stride hidden_stride
                BLOCK_SIZE N_ELEMENTS idx)
        | .inr idx =>
            s.readMem sin_cache
              (prefillCacheOffset s cumsum_lengths cache_stride hidden_stride
                BLOCK_SIZE N_ELEMENTS idx))
```

**Assumptions / layout contracts:**
- `hRegion : cos_output ≠ sin_output`
- `hN : 0 < N_ELEMENTS`
- `hOutInj : Function.Injective
      (fun i : Fin HIDDEN_DIM =>
        prefillOutOffset s cache_stride hidden_stride BLOCK_SIZE i)`
- `kernel : = prefill_cache_kernel cos_cache sin_cache cumsum_lengths cos_output
        sin_output cache_stride hidden_stride total_length HIDDEN_DIM N_ELEMENTS
        BLOCK_SIZE`
- `initialState : = s`
- `write : = fun i : Sum (Fin HIDDEN_DIM) (Fin HIDDEN_DIM) =>
        match i with
        | .inl idx =>
            if prefillActive s total_length BLOCK_SIZE then
              some (cos_output, prefillOutOffset s cache_stride hidden_stride
                BLOCK_SIZE idx)
            else none
        | .inr idx =>
            if prefillActive s total_length BLOCK_SIZE then
              some (sin_output, prefillOutOffset s cache_stride hidden_stride
                BLOCK_SIZE idx)
            else none`
- `expected : = fun i =>
        match i with
        | .inl idx =>
            s.readMem cos_cache
              (prefillCacheOffset s cumsum_lengths cache_stride hidden_stride
                BLOCK_SIZE N_ELEMENTS idx)
        | .inr idx =>
            s.readMem sin_cache
              (prefillCacheOffset s cumsum_lengths cache_stride hidden_stride
                BLOCK_SIZE N_ELEMENTS idx)`

**Closed-form spec defs (transitive):** `prefillOutOffset`, `prefill_cache_kernel`, `prefillActive`, `prefillCacheOffset`, `prefillIdx`, `prefillOriSeqIdx`

<details><summary><code>prefillOutOffset</code></summary>

```lean
def prefillOutOffset
    (s : BlockState) (cache_stride hidden_stride BLOCK_SIZE : Nat)
    (i : Fin HIDDEN_DIM) : Nat :=
  prefillIdx s BLOCK_SIZE * cache_stride + i.val * hidden_stride
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

<details><summary><code>prefillActive</code></summary>

```lean
def prefillActive (s : BlockState) (total_length BLOCK_SIZE : Nat) : Prop :=
  prefillIdx s BLOCK_SIZE < total_length
```
</details>

<details><summary><code>prefillCacheOffset</code></summary>

```
/-- Cache offset for the full prefill kernel using the data-dependent source row
`ori_seq_idx`. -/
```
```lean
noncomputable def prefillCacheOffset
    (s : BlockState) (cumsum_lengths : Region .nat)
    (cache_stride hidden_stride BLOCK_SIZE N_ELEMENTS : Nat) (i : Fin HIDDEN_DIM) : Nat :=
  prefillOriSeqIdx s cumsum_lengths BLOCK_SIZE N_ELEMENTS * cache_stride +
    i.val * hidden_stride
```
</details>

<details><summary><code>prefillIdx</code></summary>

```lean
def prefillIdx (s : BlockState) (BLOCK_SIZE : Nat) : Nat :=
  s.pids 0 * BLOCK_SIZE + s.pids 1
```
</details>

<details><summary><code>prefillOriSeqIdx</code></summary>

```
/-- Source row index `ori_seq_idx` computed by the full prefill kernel:
`idx - tl.max(tl.where(cumsum_lens <= idx, cumsum_lens, 0))`. The reduction
is `reduceMaxNatDrop` over the `[N_ELEMENTS]` tile loaded from `cumsum_lengths`.
When `N_ELEMENTS = 0` the reduction returns `none`; we treat that case as `0`
so the spec is total. -/
```
```lean
noncomputable def prefillOriSeqIdx
    (s : BlockState) (cumsum_lengths : Region .nat) (BLOCK_SIZE N_ELEMENTS : Nat) : Nat :=
  let idx := prefillIdx s BLOCK_SIZE
  let tile : Tile .nat [N_ELEMENTS] :=
    ⟨fun i => s.readMemValue .nat (Region.cast cumsum_lengths) i.1.val⟩
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

## Public theorem: `decoding_cache_kernel_output_summary`

<details><summary>docstring</summary>

```
/-- Per-kernel output summary for `decoding_cache_kernel`: the DSL surface lowers
to the algorithm layer, and both masked cos/sin scatters are compute-correct —
every active cell holds the matching cos/sin cache cell at the `lengths`-selected
source row, inactive cells are preserved. -/
```
</details>

**Statement:**
```lean
theorem decoding_cache_kernel_output_summary
    (cos_cache sin_cache : RegionName) (lengths : Region .nat) (cos_output sin_output : RegionName)
    (cache_stride hidden_stride HIDDEN_DIM NUM_SEQS BLOCK_SIZE : Nat)
    (s : BlockState)
    (hRegion : cos_output ≠ sin_output)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE, HIDDEN_DIM] =>
        decodeOutOffset s cache_stride hidden_stride BLOCK_SIZE idx)) :
    (∃ alg, (decoding_cache_kernel cos_cache sin_cache lengths cos_output
        sin_output cache_stride hidden_stride HIDDEN_DIM NUM_SEQS
        BLOCK_SIZE).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := decoding_cache_kernel cos_cache sin_cache lengths cos_output
        sin_output cache_stride hidden_stride HIDDEN_DIM NUM_SEQS BLOCK_SIZE)
      (initialState := s)
      (write := fun i : Sum (TileIndex [BLOCK_SIZE, HIDDEN_DIM])
          (TileIndex [BLOCK_SIZE, HIDDEN_DIM]) =>
        match i with
        | .inl idx =>
            if decodeActive s NUM_SEQS BLOCK_SIZE idx then
              some (cos_output, decodeOutOffset s cache_stride hidden_stride
                BLOCK_SIZE idx)
            else none
        | .inr idx =>
            if decodeActive s NUM_SEQS BLOCK_SIZE idx then
              some (sin_output, decodeOutOffset s cache_stride hidden_stride
                BLOCK_SIZE idx)
            else none)
      (expected := fun i =>
        match i with
        | .inl idx =>
            s.readMem cos_cache
              (decodeCacheOffset s lengths cache_stride hidden_stride BLOCK_SIZE idx)
        | .inr idx =>
            s.readMem sin_cache
              (decodeCacheOffset s lengths cache_stride hidden_stride BLOCK_SIZE idx))
```

**Assumptions / layout contracts:**
- `hRegion : cos_output ≠ sin_output`
- `hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE, HIDDEN_DIM] =>
        decodeOutOffset s cache_stride hidden_stride BLOCK_SIZE idx)`
- `kernel : = decoding_cache_kernel cos_cache sin_cache lengths cos_output
        sin_output cache_stride hidden_stride HIDDEN_DIM NUM_SEQS BLOCK_SIZE`
- `initialState : = s`
- `write : = fun i : Sum (TileIndex [BLOCK_SIZE, HIDDEN_DIM])
          (TileIndex [BLOCK_SIZE, HIDDEN_DIM]) =>
        match i with
        | .inl idx =>
            if decodeActive s NUM_SEQS BLOCK_SIZE idx then
              some (cos_output, decodeOutOffset s cache_stride hidden_stride
                BLOCK_SIZE idx)
            else none
        | .inr idx =>
            if decodeActive s NUM_SEQS BLOCK_SIZE idx then
              some (sin_output, decodeOutOffset s cache_stride hidden_stride
                BLOCK_SIZE idx)
            else none`
- `expected : = fun i =>
        match i with
        | .inl idx =>
            s.readMem cos_cache
              (decodeCacheOffset s lengths cache_stride hidden_stride BLOCK_SIZE idx)
        | .inr idx =>
            s.readMem sin_cache
              (decodeCacheOffset s lengths cache_stride hidden_stride BLOCK_SIZE idx)`

**Closed-form spec defs (transitive):** `decodeOutOffset`, `decoding_cache_kernel`, `decodeActive`, `decodeCacheOffset`, `rowIndex`

<details><summary><code>decodeOutOffset</code></summary>

```lean
def decodeOutOffset
    (s : BlockState) (cache_stride hidden_stride BLOCK_SIZE : Nat)
    (idx : TileIndex [BLOCK_SIZE, HIDDEN_DIM]) : Nat :=
  rowIndex s BLOCK_SIZE idx.1 * cache_stride + idx.2.1.val * hidden_stride
```
</details>

<details><summary><code>decoding_cache_kernel</code></summary>

```
/-- Faithful transcription of `cache_transform_triton.py`'s
`decoding_cache_kernel`.

Allowed mechanical Lean-syntax-only changes:
- `lengths` is a typed Lean Nat region so its `tl.load` call does not need an
  extra `dtype=` kwarg. -/
```
```lean
def decoding_cache_kernel
    (cos_cache sin_cache : RegionName) (lengths : Region .nat)
    (cos_output sin_output : RegionName)
    (cache_stride hidden_stride HIDDEN_DIM NUM_SEQS BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  idx = tl.program_id(0) * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  ori_seq_idx = tl.load(lengths + idx,
    mask=idx < $(NUM_SEQS), other=None)
  cos_cache_part = tl.load(cos_cache + ori_seq_idx[:, None] * $(cache_stride) +
      tl.arange(0, $(HIDDEN_DIM))[None, :] * $(hidden_stride),
    mask=idx[:, None] < $(NUM_SEQS))
  sin_cache_part = tl.load(sin_cache + ori_seq_idx[:, None] * $(cache_stride) +
      tl.arange(0, $(HIDDEN_DIM))[None, :] * $(hidden_stride),
    mask=idx[:, None] < $(NUM_SEQS))
  tl.store(cos_output + idx[:, None] * $(cache_stride) +
      tl.arange(0, $(HIDDEN_DIM))[None, :] * $(hidden_stride),
    cos_cache_part, mask=idx[:, None] < $(NUM_SEQS))
  tl.store(sin_output + idx[:, None] * $(cache_stride) +
      tl.arange(0, $(HIDDEN_DIM))[None, :] * $(hidden_stride),
    sin_cache_part, mask=idx[:, None] < $(NUM_SEQS))
}
```
</details>

<details><summary><code>decodeActive</code></summary>

```lean
def decodeActive (s : BlockState) (NUM_SEQS BLOCK_SIZE : Nat)
    (idx : TileIndex [BLOCK_SIZE, HIDDEN_DIM]) : Prop :=
  rowIndex s BLOCK_SIZE idx.1 < NUM_SEQS
```
</details>

<details><summary><code>decodeCacheOffset</code></summary>

```lean
def decodeCacheOffset
    (s : BlockState) (lengths : RegionName)
    (cache_stride hidden_stride BLOCK_SIZE : Nat)
    (idx : TileIndex [BLOCK_SIZE, HIDDEN_DIM]) : Nat :=
  s.readMemValue .nat lengths (rowIndex s BLOCK_SIZE idx.1) * cache_stride +
    idx.2.1.val * hidden_stride
```
</details>

<details><summary><code>rowIndex</code></summary>

```lean
def rowIndex (s : BlockState) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 0 * BLOCK_SIZE + i.val
```
</details>

## Also present (pinned special-case summaries)
- `decoding_cache_kernel_compute_correct`
- `decoding_cache_kernel_sin_compute_correct`
- `decoding_cache_one_seq_block_compute_correct`
- `prefill_cache_cos_store_slice_compute_correct`
- `prefill_cache_sin_store_slice_compute_correct`
- `prefill_cache_kernel_compute_correct`
