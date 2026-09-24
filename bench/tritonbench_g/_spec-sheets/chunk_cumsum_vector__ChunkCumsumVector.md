# Spec sheet — `bench/tritonbench_g/chunk_cumsum_vector/ChunkCumsumVector.lean`

**Python source:** `bench/tritonbench_g/chunk_cumsum_vector/chunk_cumsum_vector.py`

## Public theorem: `chunk_cumsum_vector_output_summary_general`

<details><summary>docstring</summary>

```
/-- **Dimension-general single-chunk output summary.** Subsumes the former
per-shape Python-case summaries (which differed only in the
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
specification chunk_cumsum_vector_output_summary_general
    (SReg Z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat) (s : BlockState)
    (hNoCollision : ∀ idx : TileIndex [BT, BS], singleBlockActive s T S BS idx →
      ∀ k : TileIndex [BT, BS], singleBlockActive s T S BS k →
        singleBlockTileOffset s s_s_h s_s_t s_s_d BS k =
          singleBlockTileOffset s s_s_h s_s_t s_s_d BS idx → k = idx) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := chunk_cumsum_vector_single_block_surface SReg Z s_s_h s_s_t
        s_s_d T S BT BS)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BT, BS] => singleBlockActive s T S BS idx)
        (fun idx : TileIndex [BT, BS] =>
          (Z, singleBlockTileOffset s s_s_h s_s_t s_s_d BS idx)))
      (expected := fun idx : TileIndex [BT, BS] =>
        singleBlockCumsumVectorClosed s SReg s_s_h s_s_t s_s_d T S BS idx)
```

**Assumptions / layout contracts:**
- `hNoCollision : ∀ idx : TileIndex [BT, BS], singleBlockActive s T S BS idx →
      ∀ k : TileIndex [BT, BS], singleBlockActive s T S BS k →
        singleBlockTileOffset s s_s_h s_s_t s_s_d BS k =
          singleBlockTileOffset s s_s_h s_s_t s_s_d BS idx → k = idx`

**Closed-form spec defs (transitive):** `singleBlockActive`, `singleBlockTileOffset`, `chunk_cumsum_vector_single_block_surface`, `singleBlockCumsumVectorClosed`, `sIndex`

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

<details><summary><code>chunk_cumsum_vector_single_block_surface</code></summary>

```
/-- Single-iteration surface for Python cases where `T <= BT`.

The checked cases are covered by the autotuned `BT = 16` configuration. In this
path the loop executes once, `b_z` is the initial zero vector, and the observable
output is the block-pointer load followed by the lower-triangular dot and
boundary-checked block-pointer store. -/
```
```lean
def chunk_cumsum_vector_single_block_surface
    (S Z : RegionName) (s_s_h s_s_t s_s_d T SSize BT BS : Nat) :
    ComputeKernel := triton {
  i_s = tl.program_id(0)
  i_bh = tl.program_id(1)
  o_i = tl.arange(0, $(BT))
  m_s = tl.where(o_i[:, None] >= o_i[None, :], 1.0, 0.0)
  p_s = tl.make_block_ptr(base=S + i_bh * $(s_s_h), shape=($(T), $(SSize)),
    strides=($(s_s_t), $(s_s_d)), offsets=($(0), i_s * $(BS)),
    block_shape=($(BT), $(BS)), order=(1, 0))
  p_z = tl.make_block_ptr(base=Z + i_bh * $(s_s_h), shape=($(T), $(SSize)),
    strides=($(s_s_t), $(s_s_d)), offsets=($(0), i_s * $(BS)),
    block_shape=($(BT), $(BS)), order=(1, 0))
  b_s = tl.load(p_s, boundary_check=([0, 1] : List Nat)).to(tl.float32)
  b_c = tl.dot(m_s, b_s, allow_tf32=false)
  tl.store(p_z, (b_c).to(p_z.dtype.element_ty), boundary_check=([0, 1] : List Nat))
}
```
</details>

<details><summary><code>singleBlockCumsumVectorClosed</code></summary>

```
/-- Genuine closed form for the single-Python-chunk path (`i_t = 0`, carry `= 0`):
for each feature column `j`, the prefix sum of all source entries up to and
including flat index `i`. -/
```
```lean
noncomputable def singleBlockCumsumVectorClosed
    (s : BlockState) (SReg : RegionName) (s_s_h s_s_t s_s_d T S BS : Nat)
    (idx : TileIndex [BT, BS]) : ℝ :=
  if sIndex s BS idx.2.1 < S then
    ∑ flat ∈ (Finset.range T).filter (fun flat => flat ≤ idx.1.val),
      s.readMem SReg (s.pids 1 * s_s_h + flat * s_s_t +
        sIndex s BS idx.2.1 * s_s_d)
  else 0
```
</details>

<details><summary><code>sIndex</code></summary>

```lean
def sIndex (s : BlockState) (BS : Nat) (j : Fin BS) : Nat :=
  s.pids 0 * BS + j.val
```
</details>

## Public theorem: `chunk_cumsum_vector_block_store_io_correctness`

<details><summary>docstring</summary>

```
/-- **The headline on the IO surface** for `chunk_cumsum_vector.py`'s masked block
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
specification chunk_cumsum_vector_block_store_io_correctness (BC Z : RegionName)
    (s_s_h s_s_t s_s_d T S BT BS : Nat)
    (hOutInj : ∀ p₀ p₁ p₂ : Nat, Function.Injective
      (fun idx : TileIndex [BT, BS] =>
        p₁ * s_s_h + (p₂ * BT + idx.1.val) * s_s_t
          + (p₀ * BS + idx.2.1.val) * s_s_d)) :
    blockStoreIO BC Z s_s_h s_s_t s_s_d T S BT BS
      ⊨ fun _p₀ _p₁ xs idx => xs idx
```

**Closed-form spec defs (transitive):** `blockStoreIO`, `chunk_cumsum_vector_store_slice`

<details><summary><code>blockStoreIO</code></summary>

```
/-- IO signature of the masked block store on the three-axis tile surface: the
`[BT, BS]` block's one address is built from all three program axes, and the same
address serves the read and the write. -/
```
```lean
def blockStoreIO (BC Z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat) :
    Masked3DTileKernelIO₁ where
  kernel := chunk_cumsum_vector_store_slice BC Z s_s_h s_s_t s_s_d T S BT BS
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

<details><summary><code>chunk_cumsum_vector_store_slice</code></summary>

```
/-- Proof-oriented block store slice of `chunk_cumsum_vector.py`'s
`chunk_global_cumsum_vector_kernel`.

The full kernel computes a per-feature chunk cumsum tile. This slice starts from
a precomputed `BC` tile for one `(i_s, i_bh, i_t)` block and proves the
boundary-checked writeback into `Z`. -/
```
```lean
def chunk_cumsum_vector_store_slice
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
      offs_s[None, :] * $(s_s_d), b_c, mask=mask)
}
```
</details>

## Public theorem: `chunk_cumsum_vector_block_store_io_correctnessR`

<details><summary>docstring</summary>

```
/-- **The `⊨[R]` headline** for `chunk_cumsum_vector.py`'s masked block store:
for **every** rounding model `R`, the same masked Hoare triple as
`chunk_cumsum_vector_block_store_io_correctness`, but run under `execR R` and
read back as `.real`-typed cells holding `R.round .real (xs idx)`.

The store is a pure copy and carries no `.to(...)`, so the slice is cast-free and
the exact run transports verbatim. The content of the rounding face here is
exactly that: *this kernel introduces no rounding event of its own*, at any
`R`. -/
```
</details>

**Statement:**
```lean
specification chunk_cumsum_vector_block_store_io_correctnessR (R : RoundingModel)
    (BC Z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat)
    (hOutInj : ∀ p₀ p₁ p₂ : Nat, Function.Injective
      (fun idx : TileIndex [BT, BS] =>
        p₁ * s_s_h + (p₂ * BT + idx.1.val) * s_s_t
          + (p₀ * BS + idx.2.1.val) * s_s_d)) :
    blockStoreIO BC Z s_s_h s_s_t s_s_d T S BT BS
      ⊨[R, FloatDType.real] fun _p₀ _p₁ xs idx => xs idx
```

**Closed-form spec defs (transitive):** `blockStoreIO`, `chunk_cumsum_vector_store_slice`

<details><summary><code>blockStoreIO</code></summary>

```
/-- IO signature of the masked block store on the three-axis tile surface: the
`[BT, BS]` block's one address is built from all three program axes, and the same
address serves the read and the write. -/
```
```lean
def blockStoreIO (BC Z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat) :
    Masked3DTileKernelIO₁ where
  kernel := chunk_cumsum_vector_store_slice BC Z s_s_h s_s_t s_s_d T S BT BS
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

<details><summary><code>chunk_cumsum_vector_store_slice</code></summary>

```
/-- Proof-oriented block store slice of `chunk_cumsum_vector.py`'s
`chunk_global_cumsum_vector_kernel`.

The full kernel computes a per-feature chunk cumsum tile. This slice starts from
a precomputed `BC` tile for one `(i_s, i_bh, i_t)` block and proves the
boundary-checked writeback into `Z`. -/
```
```lean
def chunk_cumsum_vector_store_slice
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
      offs_s[None, :] * $(s_s_d), b_c, mask=mask)
}
```
</details>

## Also present (pinned special-case summaries)
- `chunk_cumsum_vector_store_slice_compute_correct`
- `chunk_cumsum_vector_single_block_surface_compute_correct`
- `chunk_cumsum_vector_single_block_surface_active_compute_correct`
- `chunk_cumsum_vector_cumsum_slice_compute_correct`
- `chunk_cumsum_vector_store_slice_active_compute_correct`
- `chunk_cumsum_vector_cumsum_slice_active_compute_correct`
