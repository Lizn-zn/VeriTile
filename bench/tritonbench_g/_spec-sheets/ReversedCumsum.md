# Spec sheet — `bench/tritonbench_g/reversed_cumsum/ReversedCumsum.lean`

**Python source:** `bench/tritonbench_g/reversed_cumsum/reversed_cumsum.py`

## Public theorem: `reversed_cumsum_output_summary_general`

<details><summary>docstring</summary>

```
/-- **Dimension-general single-chunk output summary.** Holds at fully symbolic
dimensions (arbitrary `s_s_h s_s_t s_s_d T S BT BS`).
For any single-chunk shape (`T ≤ BT`, so the reverse loop runs once with carry
`b_z = 0`), the full reverse-traversal surface lowers to the algorithm layer, and
the faithful single-chunk surface writes into every active lane `(i, j)` the
genuine reversed cumulative sum `Σ_{k ≥ i, k < T} x[k, j]` (`reversedCumsumClosed`,
a standalone `Finset.sum`, not a read-back of the kernel output). The active-lane
collision-freedom of the block address map is the explicit hypothesis. -/
```
</details>

**Statement:**
```lean
specification reversed_cumsum_output_summary_general
    (SReg Z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat) (s : BlockState)
    (hNoCollision : ∀ idx : TileIndex [BT, BS], singleBlockActive s T S BS idx →
      ∀ k : TileIndex [BT, BS], singleBlockActive s T S BS k →
        singleBlockTileOffset s s_s_h s_s_t s_s_d BS k =
          singleBlockTileOffset s s_s_h s_s_t s_s_d BS idx → k = idx) :
    (∃ alg, (reversed_cumsum_surface SReg Z s_s_h s_s_t s_s_d T S BT BS).toAlgorithm? =
      Except.ok alg) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := reversed_cumsum_single_block_surface SReg Z s_s_h s_s_t s_s_d
        T S BT BS)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BT, BS] => singleBlockActive s T S BS idx)
        (fun idx : TileIndex [BT, BS] =>
          (Z, singleBlockTileOffset s s_s_h s_s_t s_s_d BS idx)))
      (expected := fun idx : TileIndex [BT, BS] =>
        reversedCumsumClosed s SReg s_s_h s_s_t s_s_d T S BT BS idx))
```

**Assumptions / layout contracts:**
- `hNoCollision : ∀ idx : TileIndex [BT, BS], singleBlockActive s T S BS idx →
      ∀ k : TileIndex [BT, BS], singleBlockActive s T S BS k →
        singleBlockTileOffset s s_s_h s_s_t s_s_d BS k =
          singleBlockTileOffset s s_s_h s_s_t s_s_d BS idx → k = idx`
- `fun idx : TileIndex [BT, BS] => singleBlockActive s T S BS idx`
- `fun idx : TileIndex [BT, BS] =>
          (Z, singleBlockTileOffset s s_s_h s_s_t s_s_d BS idx)`

**Closed-form spec defs (transitive):** `singleBlockActive`, `singleBlockTileOffset`, `reversed_cumsum_surface`, `reversed_cumsum_single_block_surface`, `reversedCumsumClosed`, `sIndex`

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

<details><summary><code>reversed_cumsum_surface</code></summary>

```
/-- Faithful transcription of `reversed_cumsum.py`'s
`chunk_global_reversed_cumsum_vector_kernel`.

The source traverses chunks in reverse. This surface preserves the reverse
range and carries `b_z` across chunks. -/
```
```lean
def reversed_cumsum_surface
    (s z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat) :
    ComputeKernel := triton {
  i_s = tl.program_id(0)
  i_bh = tl.program_id(1)
  o_i = tl.arange(0, $(BT))
  m_s = tl.where(o_i[:, None] <= o_i[None, :], 1.0, 0.0)
  b_z = tl.zeros([$(BS)], dtype=tl.float32)
  for i_t in range(tl.cdiv($(T), $(BT)) - $(1), -$(1), -$(1)) {
    p_s = tl.make_block_ptr(base=s + i_bh * $(s_s_h),
      shape=($(T), $(S)), strides=($(s_s_t), $(s_s_d)),
      offsets=(i_t * $(BT), i_s * $(BS)), block_shape=($(BT), $(BS)), order=(1, 0))
    p_z = tl.make_block_ptr(base=z + i_bh * $(s_s_h),
      shape=($(T), $(S)), strides=($(s_s_t), $(s_s_d)),
      offsets=(i_t * $(BT), i_s * $(BS)), block_shape=($(BT), $(BS)), order=(1, 0))
    b_s = tl.load(p_s, boundary_check=([0, 1] : List Nat)).to(tl.float32)
    b_c = b_z[None, :] + tl.dot(m_s, b_s, allow_tf32=false)
    tl.store(p_z, (b_c).to(p_z.dtype.element_ty), boundary_check=([0, 1] : List Nat))
    if i_t >= $(0) {
      b_z += tl.sum(b_s, 0)
    }
  }
}
```
</details>

<details><summary><code>reversed_cumsum_single_block_surface</code></summary>

```
/-- Specialized transcription of `reversed_cumsum.py`'s
`chunk_global_reversed_cumsum_vector_kernel` for the single-`BT` block path.

When the time dimension fits in one `BT` tile, the Python reverse traversal has
one iteration with `b_z = 0`. This surface preserves the triangular mask
`o_i[:, None] <= o_i[None, :]`, the boundary-checked input load, the
`tl.dot(m_s, b_s, allow_tf32=false)` reversed cumulative sum, and the
boundary-checked store. -/
```
```lean
def reversed_cumsum_single_block_surface
    (SReg Z : RegionName) (s_s_h s_s_t s_s_d T SSize BT BS : Nat) :
    ComputeKernel := triton {
  i_s = tl.program_id(0)
  i_bh = tl.program_id(1)
  o_i = tl.arange(0, $(BT))
  m_s = tl.where(o_i[:, None] <= o_i[None, :], 1.0, 0.0)
  p_s = tl.make_block_ptr(base=SReg + i_bh * $(s_s_h),
    shape=($(T), $(SSize)), strides=($(s_s_t), $(s_s_d)),
    offsets=($(0), i_s * $(BS)), block_shape=($(BT), $(BS)), order=(1, 0))
  p_z = tl.make_block_ptr(base=Z + i_bh * $(s_s_h),
    shape=($(T), $(SSize)), strides=($(s_s_t), $(s_s_d)),
    offsets=($(0), i_s * $(BS)), block_shape=($(BT), $(BS)), order=(1, 0))
  b_s = tl.load(p_s, boundary_check=([0, 1] : List Nat)).to(tl.float32)
  b_c = tl.dot(m_s, b_s, allow_tf32=false)
  tl.store(p_z, (b_c).to(p_z.dtype.element_ty), boundary_check=([0, 1] : List Nat))
}
```
</details>

<details><summary><code>reversedCumsumClosed</code></summary>

```
/-! ## Genuine reversed-cumsum closed form

The Triton kernel computes, for output row `i` and feature column `j`, the
*reversed* cumulative sum along the time axis: the sum of the source value at
every row `k ≥ i` (that is still inside the tensor, `k < T`) in the same
feature column. The matrix product `tl.dot(m_s, b_s)` with the upper-triangular
mask `m_s[i,k] = [i ≤ k]` realizes exactly this directional scan.

`reversedCumsumClosed` is the genuine mathematical specification — a `Finset.sum`
over `{k : k ≥ i ∧ k < T}` of the loaded source value. It is *not* a read-back
of the kernel's own output, so realizing it is a true correctness statement. -/
```
```lean
noncomputable def reversedCumsumClosed
    (s : BlockState) (SReg : RegionName)
    (s_s_h s_s_t s_s_d T S BT BS : Nat)
    (idx : TileIndex [BT, BS]) : ℝ :=
  ∑ k : Fin BT,
    if idx.1.val ≤ k.val ∧ k.val < T ∧ sIndex s BS idx.2.1 < S then
      s.readMem SReg (singleBlockTileOffset s s_s_h s_s_t s_s_d BS
        ((k, idx.2.1, PUnit.unit) : TileIndex [BT, BS]))
    else 0
```
</details>

<details><summary><code>sIndex</code></summary>

```lean
def sIndex (s : BlockState) (BS : Nat) (j : Fin BS) : Nat :=
  s.pids 0 * BS + j.val
```
</details>

## Public theorem: `reversed_cumsum_block_store_io_correctness`

<details><summary>docstring</summary>

```
/-- **The headline on the IO surface** for `reversed_cumsum.py`'s masked block
store: for every disjoint flat placement of `BC` / `Z`, every program coordinate
whose active lanes are in bounds, and every launch state whose `BC` block holds `xs`
at the active lanes, the translated pointer kernel terminates, every active lane of
the `Z` block holds `xs idx`, and every other memory cell is unchanged.

The block address is built from **all three** program axes (`i_s`, `i_bh`, `i_t`) —
the shape the three-axis tile skin exists for. Dimension-general in the three
strides, `T`, `S`, `BT`, `BS`. Honest side-condition: output-address injectivity at
every program coordinate, the same hypothesis the per-write-map summary takes. -/
```
</details>

**Statement:**
```lean
specification reversed_cumsum_block_store_io_correctness (BC Z : RegionName)
    (s_s_h s_s_t s_s_d T S BT BS : Nat)
    (hOutInj : ∀ p₀ p₁ p₂ : Nat, Function.Injective
      (fun idx : TileIndex [BT, BS] =>
        p₁ * s_s_h + (p₂ * BT + idx.1.val) * s_s_t
          + (p₀ * BS + idx.2.1.val) * s_s_d)) :
    blockStoreIO BC Z s_s_h s_s_t s_s_d T S BT BS
      ⊨ fun _p₀ _p₁ xs idx => xs idx
```

**Assumptions / layout contracts:**
- `fun idx : TileIndex [BT, BS] =>
        p₁ * s_s_h + (p₂ * BT + idx.1.val) * s_s_t
          + (p₀ * BS + idx.2.1.val) * s_s_d`

**Closed-form spec defs (transitive):** `blockStoreIO`, `reversed_cumsum_store_slice`

<details><summary><code>blockStoreIO</code></summary>

```
/-- IO signature of the masked block store on the three-axis tile surface: the
`[BT, BS]` block's one address is built from all three program axes, and the same
address serves the read and the write. -/
```
```lean
def blockStoreIO (BC Z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat) :
    Masked3DTileKernelIO₁ where
  kernel := reversed_cumsum_store_slice BC Z s_s_h s_s_t s_s_d T S BT BS
  inp := BC
  out := Z
  shape := [BT, BS]
  read := fun p₀ p₁ p₂ idx =>
    p₁ * s_s_h + (p₂ * BT + idx.1.val) * s_s_t + (p₀ * BS + idx.2.1.val) * s_s_d
  write := fun p₀ p₁ p₂ idx =>
    p₁ * s_s_h + (p₂ * BT + idx.1.val) * s_s_t + (p₀ * BS + idx.2.1.val) * s_s_d
  mask := fun p₀ _p₁ p₂ idx =>
    p₂ * BT + idx.1.val < T ∧ p₀ * BS + idx.2.1.val < S
```
</details>

<details><summary><code>reversed_cumsum_store_slice</code></summary>

```
/-- Proof-oriented block store surface slice of `reversed_cumsum.py`'s
`chunk_global_reversed_cumsum_vector_kernel`.

The full kernel computes a per-feature reversed chunk cumsum tile. This slice
starts from a precomputed `BC` tile for one `(i_s, i_bh, i_t)` block and proves
the boundary-checked writeback into `Z`. -/
```
```lean
def reversed_cumsum_store_slice
    (BC Z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat) :
    ComputeKernel := triton {
  i_s = tl.program_id(0)
  i_bh = tl.program_id(1)
  i_t = tl.program_id(2)
  offs_t = i_t * $(BT) + tl.arange(0, $(BT))
  offs_s = i_s * $(BS) + tl.arange(0, $(BS))
  mask = (offs_t[:, None] < $(T)) & (offs_s[None, :] < $(S))
  b_c = tl.load(BC + i_bh * $(s_s_h) + offs_t[:, None] * $(s_s_t) +
      offs_s[None, :] * $(s_s_d), mask=mask, other=0.0)
  tl.store(Z + i_bh * $(s_s_h) + offs_t[:, None] * $(s_s_t) +
      offs_s[None, :] * $(s_s_d), (b_c).to(Z.dtype.element_ty), mask=mask)
}
```
</details>

## Public theorem: `reversed_cumsum_block_store_io_correctnessR`

<details><summary>docstring</summary>

```
/-- **The `⊨[R]` headline** for `reversed_cumsum.py`'s masked block store: for
**every** rounding model `R`, the same masked Hoare triple as
`reversed_cumsum_block_store_io_correctness`, but run under `execR R` and read
back as `.real`-typed cells holding `R.round .real (xs idx)`.

The store is a pure copy — no arithmetic, and the `.to(Z.dtype.element_ty)`
erases to `.real` — so the slice is cast-free and the exact run transports
verbatim. The content of the rounding face here is exactly that: *this kernel
introduces no rounding event of its own*, at any `R`. -/
```
</details>

**Statement:**
```lean
specification reversed_cumsum_block_store_io_correctnessR (R : RoundingModel)
    (BC Z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat)
    (hOutInj : ∀ p₀ p₁ p₂ : Nat, Function.Injective
      (fun idx : TileIndex [BT, BS] =>
        p₁ * s_s_h + (p₂ * BT + idx.1.val) * s_s_t
          + (p₀ * BS + idx.2.1.val) * s_s_d)) :
    blockStoreIO BC Z s_s_h s_s_t s_s_d T S BT BS
      ⊨[R, FloatDType.real] fun _p₀ _p₁ xs idx => xs idx
```

**Assumptions / layout contracts:**
- `fun idx : TileIndex [BT, BS] =>
        p₁ * s_s_h + (p₂ * BT + idx.1.val) * s_s_t
          + (p₀ * BS + idx.2.1.val) * s_s_d`

**Closed-form spec defs (transitive):** `blockStoreIO`, `reversed_cumsum_store_slice`

<details><summary><code>blockStoreIO</code></summary>

```
/-- IO signature of the masked block store on the three-axis tile surface: the
`[BT, BS]` block's one address is built from all three program axes, and the same
address serves the read and the write. -/
```
```lean
def blockStoreIO (BC Z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat) :
    Masked3DTileKernelIO₁ where
  kernel := reversed_cumsum_store_slice BC Z s_s_h s_s_t s_s_d T S BT BS
  inp := BC
  out := Z
  shape := [BT, BS]
  read := fun p₀ p₁ p₂ idx =>
    p₁ * s_s_h + (p₂ * BT + idx.1.val) * s_s_t + (p₀ * BS + idx.2.1.val) * s_s_d
  write := fun p₀ p₁ p₂ idx =>
    p₁ * s_s_h + (p₂ * BT + idx.1.val) * s_s_t + (p₀ * BS + idx.2.1.val) * s_s_d
  mask := fun p₀ _p₁ p₂ idx =>
    p₂ * BT + idx.1.val < T ∧ p₀ * BS + idx.2.1.val < S
```
</details>

<details><summary><code>reversed_cumsum_store_slice</code></summary>

```
/-- Proof-oriented block store surface slice of `reversed_cumsum.py`'s
`chunk_global_reversed_cumsum_vector_kernel`.

The full kernel computes a per-feature reversed chunk cumsum tile. This slice
starts from a precomputed `BC` tile for one `(i_s, i_bh, i_t)` block and proves
the boundary-checked writeback into `Z`. -/
```
```lean
def reversed_cumsum_store_slice
    (BC Z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat) :
    ComputeKernel := triton {
  i_s = tl.program_id(0)
  i_bh = tl.program_id(1)
  i_t = tl.program_id(2)
  offs_t = i_t * $(BT) + tl.arange(0, $(BT))
  offs_s = i_s * $(BS) + tl.arange(0, $(BS))
  mask = (offs_t[:, None] < $(T)) & (offs_s[None, :] < $(S))
  b_c = tl.load(BC + i_bh * $(s_s_h) + offs_t[:, None] * $(s_s_t) +
      offs_s[None, :] * $(s_s_d), mask=mask, other=0.0)
  tl.store(Z + i_bh * $(s_s_h) + offs_t[:, None] * $(s_s_t) +
      offs_s[None, :] * $(s_s_d), (b_c).to(Z.dtype.element_ty), mask=mask)
}
```
</details>

## Also present (pinned special-case summaries)
- `reversed_cumsum_store_slice_compute_correct`
- `reversed_cumsum_single_block_surface_compute_correct`
- `reversed_cumsum_cumsum_slice_compute_correct`
- `reversed_cumsum_store_slice_active_compute_correct`
- `reversed_cumsum_cumsum_slice_active_compute_correct`
