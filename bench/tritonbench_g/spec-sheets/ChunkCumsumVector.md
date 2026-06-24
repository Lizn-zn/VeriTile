# Spec sheet — `bench/tritonbench_g/chunk_cumsum_vector/ChunkCumsumVector.lean`

**Python source:** `bench/tritonbench_g/chunk_cumsum_vector/chunk_cumsum_vector.py`

## Public theorem: `chunk_cumsum_vector_python_case1_slice_summary`

<details><summary>docstring</summary>

```
/-- **Public Python case 1 summary (`B = 2`, `H = 3`, `T = 4`, `S = 5`).** The
full vector-cumsum surface lowers to the algorithm layer; the single-Python-chunk
surface (the `S → Z` path, carry `= 0`) realizes the genuine per-column global
prefix sum `singleBlockCumsumVectorClosed`; the boundary store slice passes a
precomputed tile through; and the carry-fold slice realizes the genuine global
cumulative sum `globalCumsumVectorClosed` under the per-column carry invariant.
Every `expected` is a standalone `Finset.sum` — never a read-back of the
kernel's own output. -/
```
</details>

**Statement:**
```lean
theorem chunk_cumsum_vector_python_case1_slice_summary
    (BC SReg Carry Z : RegionName) (s : BlockState)
    (hcarry : ∀ idx : TileIndex [16, 32], sIndex s 32 idx.2.1 < 5 →
      s.readMem Carry (s.pids 1 * 20 + sIndex s 32 idx.2.1 * 1)
        = ∑ flat ∈ (Finset.range 4).filter (fun flat => flat < s.pids 2 * 16),
            s.readMem SReg (s.pids 1 * 20 + flat * 5 + sIndex s 32 idx.2.1 * 1)) :
    (∃ alg, (chunk_cumsum_vector_surface SReg Z 20 5 1 4 5 16 32).toAlgorithm? =
      Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := chunk_cumsum_vector_single_block_surface SReg Z 20 5 1 4 5 16 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 32] => singleBlockActive s 4 5 32 idx)
        (fun idx => (Z, singleBlockTileOffset s 20 5 1 32 idx)))
      (expected := fun idx : TileIndex [16, 32] =>
        singleBlockCumsumVectorClosed s SReg 20 5 1 4 5 32 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := chunk_cumsum_vector_store_slice BC Z 20 5 1 4 5 16 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 32] => active s 4 5 16 32 idx)
        (fun idx => (Z, tileOffset s 20 5 1 16 32 idx)))
      (expected := fun idx : TileIndex [16, 32] =>
        storeValue s BC 20 5 1 4 5 16 32 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := chunk_cumsum_vector_cumsum_slice SReg Carry Z 20 5 1 4 5 16 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 32] => active s 4 5 16 32 idx)
        (fun idx => (Z, tileOffset s 20 5 1 16 32 idx)))
      (expected := fun idx : TileIndex [16, 32] =>
        globalCumsumVectorClosed s SReg 20 5 1 4 5 16 32 idx))
```

**Assumptions / layout contracts:**
- `hcarry : ∀ idx : TileIndex [16, 32], sIndex s 32 idx.2.1 < 5 →
      s.readMem Carry (s.pids 1 * 20 + sIndex s 32 idx.2.1 * 1)
        = ∑ flat ∈ (Finset.range 4).filter (fun flat => flat < s.pids 2 * 16),
            s.readMem SReg (s.pids 1 * 20 + flat * 5 + sIndex s 32 idx.2.1 * 1)`
- `fun idx : TileIndex [16, 32] => singleBlockActive s 4 5 32 idx`
- `fun idx : TileIndex [16, 32] => active s 4 5 16 32 idx`
- `fun idx : TileIndex [16, 32] => active s 4 5 16 32 idx`

**Closed-form spec defs (transitive):** `sIndex`, `chunk_cumsum_vector_surface`, `chunk_cumsum_vector_single_block_surface`, `singleBlockActive`, `singleBlockTileOffset`, `singleBlockCumsumVectorClosed`, `chunk_cumsum_vector_store_slice`, `active`, `tileOffset`, `storeValue`, `chunk_cumsum_vector_cumsum_slice`, `globalCumsumVectorClosed`, `tIndex`

<details><summary><code>sIndex</code></summary>

```lean
def sIndex (s : BlockState) (BS : Nat) (j : Fin BS) : Nat :=
  s.pids 0 * BS + j.val
```
</details>

<details><summary><code>chunk_cumsum_vector_surface</code></summary>

```
/-- Faithful transcription of `chunk_cumsum_vector.py`'s
`chunk_global_cumsum_vector_kernel`.

The final cast targets the block pointer destination dtype in Python. -/
```
```lean
def chunk_cumsum_vector_surface
    (S Z : RegionName) (s_s_h s_s_t s_s_d T SSize BT BS : Nat) :
    ComputeKernel := triton {
  i_s = tl.program_id(0)
  i_bh = tl.program_id(1)
  o_i = tl.arange(0, $(BT))
  m_s = tl.where(o_i[:, None] >= o_i[None, :], 1.0, 0.0)
  b_z = tl.zeros([$(BS)], dtype=tl.float32)
  for i_t in range($(0), tl.cdiv($(T), $(BT)), $(1)) {
    p_s = tl.make_block_ptr(base=S + i_bh * $(s_s_h), shape=($(T), $(SSize)),
      strides=($(s_s_t), $(s_s_d)), offsets=(i_t * $(BT), i_s * $(BS)),
      block_shape=($(BT), $(BS)), order=(1, 0))
    p_z = tl.make_block_ptr(base=Z + i_bh * $(s_s_h), shape=($(T), $(SSize)),
      strides=($(s_s_t), $(s_s_d)), offsets=(i_t * $(BT), i_s * $(BS)),
      block_shape=($(BT), $(BS)), order=(1, 0))
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

<details><summary><code>active</code></summary>

```lean
def active (s : BlockState) (T S BT BS : Nat) (idx : TileIndex [BT, BS]) : Prop :=
  tIndex s BT idx.1 < T ∧ sIndex s BS idx.2.1 < S
```
</details>

<details><summary><code>tileOffset</code></summary>

```lean
def tileOffset (s : BlockState) (s_s_h s_s_t s_s_d BT BS : Nat)
    (idx : TileIndex [BT, BS]) : Nat :=
  s.pids 1 * s_s_h + tIndex s BT idx.1 * s_s_t +
    sIndex s BS idx.2.1 * s_s_d
```
</details>

<details><summary><code>storeValue</code></summary>

```lean
noncomputable def storeValue (s : BlockState) (BC : RegionName)
    (s_s_h s_s_t s_s_d T S BT BS : Nat) (idx : TileIndex [BT, BS]) : ℝ :=
  WithBot.unbotD 0
    (if active s T S BT BS idx then
      some (s.readMem BC (tileOffset s s_s_h s_s_t s_s_d BT BS idx))
    else some (0.0 : ℝ))
```
</details>

<details><summary><code>chunk_cumsum_vector_cumsum_slice</code></summary>

```
/-! ## Computed per-chunk cumsum slice

The store slice above starts from a precomputed `BC` tile. This slice models one
loop iteration of the Python kernel after the carry vector `b_z` has been
materialized in `Carry`: it loads the source block, computes
`b_z[None, :] + tl.dot(m_s, b_s, allow_tf32=false)`, and stores the masked
result. -/
```
```lean
def chunk_cumsum_vector_cumsum_slice
    (SReg Carry Z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat) :
    ComputeKernel := triton {
  i_s = tl.program_id(0)
  i_bh = tl.program_id(1)
  i_t = tl.program_id(2)
  offs_t = i_t * $(BT) + tl.arange(0, $(BT))
  offs_s = i_s * $(BS) + tl.arange(0, $(BS))
  o_i = tl.arange(0, $(BT))
  m_s = tl.where(o_i[:, None] >= o_i[None, :], 1.0, 0.0)
  mask = (offs_t[:, None] < $(T)) & (offs_s[None, :] < $(S))
  b_s = tl.load(SReg + i_bh * $(s_s_h) + offs_t[:, None] * $(s_s_t) +
      offs_s[None, :] * $(s_s_d), mask=mask, other=0.0)
  b_z = tl.load(Carry + i_bh * $(s_s_h) + offs_s * $(s_s_d),
    mask=offs_s < $(S), other=0.0)
  b_c = b_z[None, :] + tl.dot(m_s, b_s, allow_tf32=false)
  tl.store(Z + i_bh * $(s_s_h) + offs_t[:, None] * $(s_s_t) +
      offs_s[None, :] * $(s_s_d), b_c, mask=mask)
}
```
</details>

<details><summary><code>globalCumsumVectorClosed</code></summary>

```
/-! ## Genuine chunked-cumsum closed form (vector / per-column)

`globalCumsumVectorClosed` is the genuine mathematical specification — for each
feature column `j` it is a `Finset.sum` over all *flat* time indices
`flat ≤ i_t·BT + i` (with `flat < T`) of the source value at the 2D address
`i_bh·s_s_h + flat·s_s_t + (i_s·BS + j)·s_s_d`. It is **not** a read-back of the
kernel's own output, so realizing it is a true correctness statement. Padded
feature lanes (`i_s·BS + j ≥ S`) hold `0`, matching the masked store. This is
the 2D analogue of the scalar `globalCumsumClosed`, carrying the column index
`j` through the prefix sum. -/
```
```lean
noncomputable def globalCumsumVectorClosed
    (s : BlockState) (SReg : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat)
    (idx : TileIndex [BT, BS]) : ℝ :=
  if sIndex s BS idx.2.1 < S then
    ∑ flat ∈ (Finset.range T).filter
        (fun flat => flat ≤ s.pids 2 * BT + idx.1.val),
      s.readMem SReg (s.pids 1 * s_s_h + flat * s_s_t +
        sIndex s BS idx.2.1 * s_s_d)
  else 0
```
</details>

<details><summary><code>tIndex</code></summary>

```lean
def tIndex (s : BlockState) (BT : Nat) (i : Fin BT) : Nat :=
  s.pids 2 * BT + i.val
```
</details>

## Public theorem: `chunk_cumsum_vector_python_case2_slice_summary`

<details><summary>docstring</summary>

```
/-- **Public Python case 2 summary (`B = 2`, `H = 3`, `T = 8`, `S = 10`).** See
`chunk_cumsum_vector_python_case1_slice_summary`. -/
```
</details>

**Statement:**
```lean
theorem chunk_cumsum_vector_python_case2_slice_summary
    (BC SReg Carry Z : RegionName) (s : BlockState)
    (hcarry : ∀ idx : TileIndex [16, 32], sIndex s 32 idx.2.1 < 10 →
      s.readMem Carry (s.pids 1 * 80 + sIndex s 32 idx.2.1 * 1)
        = ∑ flat ∈ (Finset.range 8).filter (fun flat => flat < s.pids 2 * 16),
            s.readMem SReg (s.pids 1 * 80 + flat * 10 + sIndex s 32 idx.2.1 * 1)) :
    (∃ alg, (chunk_cumsum_vector_surface SReg Z 80 10 1 8 10 16 32).toAlgorithm? =
      Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := chunk_cumsum_vector_single_block_surface SReg Z 80 10 1 8 10 16 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 32] => singleBlockActive s 8 10 32 idx)
        (fun idx => (Z, singleBlockTileOffset s 80 10 1 32 idx)))
      (expected := fun idx : TileIndex [16, 32] =>
        singleBlockCumsumVectorClosed s SReg 80 10 1 8 10 32 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := chunk_cumsum_vector_store_slice BC Z 80 10 1 8 10 16 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 32] => active s 8 10 16 32 idx)
        (fun idx => (Z, tileOffset s 80 10 1 16 32 idx)))
      (expected := fun idx : TileIndex [16, 32] =>
        storeValue s BC 80 10 1 8 10 16 32 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := chunk_cumsum_vector_cumsum_slice SReg Carry Z 80 10 1 8 10 16 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 32] => active s 8 10 16 32 idx)
        (fun idx => (Z, tileOffset s 80 10 1 16 32 idx)))
      (expected := fun idx : TileIndex [16, 32] =>
        globalCumsumVectorClosed s SReg 80 10 1 8 10 16 32 idx))
```

**Assumptions / layout contracts:**
- `hcarry : ∀ idx : TileIndex [16, 32], sIndex s 32 idx.2.1 < 10 →
      s.readMem Carry (s.pids 1 * 80 + sIndex s 32 idx.2.1 * 1)
        = ∑ flat ∈ (Finset.range 8).filter (fun flat => flat < s.pids 2 * 16),
            s.readMem SReg (s.pids 1 * 80 + flat * 10 + sIndex s 32 idx.2.1 * 1)`
- `fun idx : TileIndex [16, 32] => singleBlockActive s 8 10 32 idx`
- `fun idx : TileIndex [16, 32] => active s 8 10 16 32 idx`
- `fun idx : TileIndex [16, 32] => active s 8 10 16 32 idx`

**Closed-form spec defs (transitive):** `sIndex`, `chunk_cumsum_vector_surface`, `chunk_cumsum_vector_single_block_surface`, `singleBlockActive`, `singleBlockTileOffset`, `singleBlockCumsumVectorClosed`, `chunk_cumsum_vector_store_slice`, `active`, `tileOffset`, `storeValue`, `chunk_cumsum_vector_cumsum_slice`, `globalCumsumVectorClosed`, `tIndex`

<details><summary><code>sIndex</code></summary>

```lean
def sIndex (s : BlockState) (BS : Nat) (j : Fin BS) : Nat :=
  s.pids 0 * BS + j.val
```
</details>

<details><summary><code>chunk_cumsum_vector_surface</code></summary>

```
/-- Faithful transcription of `chunk_cumsum_vector.py`'s
`chunk_global_cumsum_vector_kernel`.

The final cast targets the block pointer destination dtype in Python. -/
```
```lean
def chunk_cumsum_vector_surface
    (S Z : RegionName) (s_s_h s_s_t s_s_d T SSize BT BS : Nat) :
    ComputeKernel := triton {
  i_s = tl.program_id(0)
  i_bh = tl.program_id(1)
  o_i = tl.arange(0, $(BT))
  m_s = tl.where(o_i[:, None] >= o_i[None, :], 1.0, 0.0)
  b_z = tl.zeros([$(BS)], dtype=tl.float32)
  for i_t in range($(0), tl.cdiv($(T), $(BT)), $(1)) {
    p_s = tl.make_block_ptr(base=S + i_bh * $(s_s_h), shape=($(T), $(SSize)),
      strides=($(s_s_t), $(s_s_d)), offsets=(i_t * $(BT), i_s * $(BS)),
      block_shape=($(BT), $(BS)), order=(1, 0))
    p_z = tl.make_block_ptr(base=Z + i_bh * $(s_s_h), shape=($(T), $(SSize)),
      strides=($(s_s_t), $(s_s_d)), offsets=(i_t * $(BT), i_s * $(BS)),
      block_shape=($(BT), $(BS)), order=(1, 0))
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

<details><summary><code>active</code></summary>

```lean
def active (s : BlockState) (T S BT BS : Nat) (idx : TileIndex [BT, BS]) : Prop :=
  tIndex s BT idx.1 < T ∧ sIndex s BS idx.2.1 < S
```
</details>

<details><summary><code>tileOffset</code></summary>

```lean
def tileOffset (s : BlockState) (s_s_h s_s_t s_s_d BT BS : Nat)
    (idx : TileIndex [BT, BS]) : Nat :=
  s.pids 1 * s_s_h + tIndex s BT idx.1 * s_s_t +
    sIndex s BS idx.2.1 * s_s_d
```
</details>

<details><summary><code>storeValue</code></summary>

```lean
noncomputable def storeValue (s : BlockState) (BC : RegionName)
    (s_s_h s_s_t s_s_d T S BT BS : Nat) (idx : TileIndex [BT, BS]) : ℝ :=
  WithBot.unbotD 0
    (if active s T S BT BS idx then
      some (s.readMem BC (tileOffset s s_s_h s_s_t s_s_d BT BS idx))
    else some (0.0 : ℝ))
```
</details>

<details><summary><code>chunk_cumsum_vector_cumsum_slice</code></summary>

```
/-! ## Computed per-chunk cumsum slice

The store slice above starts from a precomputed `BC` tile. This slice models one
loop iteration of the Python kernel after the carry vector `b_z` has been
materialized in `Carry`: it loads the source block, computes
`b_z[None, :] + tl.dot(m_s, b_s, allow_tf32=false)`, and stores the masked
result. -/
```
```lean
def chunk_cumsum_vector_cumsum_slice
    (SReg Carry Z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat) :
    ComputeKernel := triton {
  i_s = tl.program_id(0)
  i_bh = tl.program_id(1)
  i_t = tl.program_id(2)
  offs_t = i_t * $(BT) + tl.arange(0, $(BT))
  offs_s = i_s * $(BS) + tl.arange(0, $(BS))
  o_i = tl.arange(0, $(BT))
  m_s = tl.where(o_i[:, None] >= o_i[None, :], 1.0, 0.0)
  mask = (offs_t[:, None] < $(T)) & (offs_s[None, :] < $(S))
  b_s = tl.load(SReg + i_bh * $(s_s_h) + offs_t[:, None] * $(s_s_t) +
      offs_s[None, :] * $(s_s_d), mask=mask, other=0.0)
  b_z = tl.load(Carry + i_bh * $(s_s_h) + offs_s * $(s_s_d),
    mask=offs_s < $(S), other=0.0)
  b_c = b_z[None, :] + tl.dot(m_s, b_s, allow_tf32=false)
  tl.store(Z + i_bh * $(s_s_h) + offs_t[:, None] * $(s_s_t) +
      offs_s[None, :] * $(s_s_d), b_c, mask=mask)
}
```
</details>

<details><summary><code>globalCumsumVectorClosed</code></summary>

```
/-! ## Genuine chunked-cumsum closed form (vector / per-column)

`globalCumsumVectorClosed` is the genuine mathematical specification — for each
feature column `j` it is a `Finset.sum` over all *flat* time indices
`flat ≤ i_t·BT + i` (with `flat < T`) of the source value at the 2D address
`i_bh·s_s_h + flat·s_s_t + (i_s·BS + j)·s_s_d`. It is **not** a read-back of the
kernel's own output, so realizing it is a true correctness statement. Padded
feature lanes (`i_s·BS + j ≥ S`) hold `0`, matching the masked store. This is
the 2D analogue of the scalar `globalCumsumClosed`, carrying the column index
`j` through the prefix sum. -/
```
```lean
noncomputable def globalCumsumVectorClosed
    (s : BlockState) (SReg : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat)
    (idx : TileIndex [BT, BS]) : ℝ :=
  if sIndex s BS idx.2.1 < S then
    ∑ flat ∈ (Finset.range T).filter
        (fun flat => flat ≤ s.pids 2 * BT + idx.1.val),
      s.readMem SReg (s.pids 1 * s_s_h + flat * s_s_t +
        sIndex s BS idx.2.1 * s_s_d)
  else 0
```
</details>

<details><summary><code>tIndex</code></summary>

```lean
def tIndex (s : BlockState) (BT : Nat) (i : Fin BT) : Nat :=
  s.pids 2 * BT + i.val
```
</details>

## Public theorem: `chunk_cumsum_vector_python_case3_slice_summary`

<details><summary>docstring</summary>

```
/-- **Public Python case 3 summary (`B = 2`, `H = 3`, `T = 1`, `S = 5`).** See
`chunk_cumsum_vector_python_case1_slice_summary`. -/
```
</details>

**Statement:**
```lean
theorem chunk_cumsum_vector_python_case3_slice_summary
    (BC SReg Carry Z : RegionName) (s : BlockState)
    (hcarry : ∀ idx : TileIndex [16, 32], sIndex s 32 idx.2.1 < 5 →
      s.readMem Carry (s.pids 1 * 5 + sIndex s 32 idx.2.1 * 1)
        = ∑ flat ∈ (Finset.range 1).filter (fun flat => flat < s.pids 2 * 16),
            s.readMem SReg (s.pids 1 * 5 + flat * 5 + sIndex s 32 idx.2.1 * 1)) :
    (∃ alg, (chunk_cumsum_vector_surface SReg Z 5 5 1 1 5 16 32).toAlgorithm? =
      Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := chunk_cumsum_vector_single_block_surface SReg Z 5 5 1 1 5 16 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 32] => singleBlockActive s 1 5 32 idx)
        (fun idx => (Z, singleBlockTileOffset s 5 5 1 32 idx)))
      (expected := fun idx : TileIndex [16, 32] =>
        singleBlockCumsumVectorClosed s SReg 5 5 1 1 5 32 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := chunk_cumsum_vector_store_slice BC Z 5 5 1 1 5 16 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 32] => active s 1 5 16 32 idx)
        (fun idx => (Z, tileOffset s 5 5 1 16 32 idx)))
      (expected := fun idx : TileIndex [16, 32] =>
        storeValue s BC 5 5 1 1 5 16 32 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := chunk_cumsum_vector_cumsum_slice SReg Carry Z 5 5 1 1 5 16 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 32] => active s 1 5 16 32 idx)
        (fun idx => (Z, tileOffset s 5 5 1 16 32 idx)))
      (expected := fun idx : TileIndex [16, 32] =>
        globalCumsumVectorClosed s SReg 5 5 1 1 5 16 32 idx))
```

**Assumptions / layout contracts:**
- `hcarry : ∀ idx : TileIndex [16, 32], sIndex s 32 idx.2.1 < 5 →
      s.readMem Carry (s.pids 1 * 5 + sIndex s 32 idx.2.1 * 1)
        = ∑ flat ∈ (Finset.range 1).filter (fun flat => flat < s.pids 2 * 16),
            s.readMem SReg (s.pids 1 * 5 + flat * 5 + sIndex s 32 idx.2.1 * 1)`
- `fun idx : TileIndex [16, 32] => singleBlockActive s 1 5 32 idx`
- `fun idx : TileIndex [16, 32] => active s 1 5 16 32 idx`
- `fun idx : TileIndex [16, 32] => active s 1 5 16 32 idx`

**Closed-form spec defs (transitive):** `sIndex`, `chunk_cumsum_vector_surface`, `chunk_cumsum_vector_single_block_surface`, `singleBlockActive`, `singleBlockTileOffset`, `singleBlockCumsumVectorClosed`, `chunk_cumsum_vector_store_slice`, `active`, `tileOffset`, `storeValue`, `chunk_cumsum_vector_cumsum_slice`, `globalCumsumVectorClosed`, `tIndex`

<details><summary><code>sIndex</code></summary>

```lean
def sIndex (s : BlockState) (BS : Nat) (j : Fin BS) : Nat :=
  s.pids 0 * BS + j.val
```
</details>

<details><summary><code>chunk_cumsum_vector_surface</code></summary>

```
/-- Faithful transcription of `chunk_cumsum_vector.py`'s
`chunk_global_cumsum_vector_kernel`.

The final cast targets the block pointer destination dtype in Python. -/
```
```lean
def chunk_cumsum_vector_surface
    (S Z : RegionName) (s_s_h s_s_t s_s_d T SSize BT BS : Nat) :
    ComputeKernel := triton {
  i_s = tl.program_id(0)
  i_bh = tl.program_id(1)
  o_i = tl.arange(0, $(BT))
  m_s = tl.where(o_i[:, None] >= o_i[None, :], 1.0, 0.0)
  b_z = tl.zeros([$(BS)], dtype=tl.float32)
  for i_t in range($(0), tl.cdiv($(T), $(BT)), $(1)) {
    p_s = tl.make_block_ptr(base=S + i_bh * $(s_s_h), shape=($(T), $(SSize)),
      strides=($(s_s_t), $(s_s_d)), offsets=(i_t * $(BT), i_s * $(BS)),
      block_shape=($(BT), $(BS)), order=(1, 0))
    p_z = tl.make_block_ptr(base=Z + i_bh * $(s_s_h), shape=($(T), $(SSize)),
      strides=($(s_s_t), $(s_s_d)), offsets=(i_t * $(BT), i_s * $(BS)),
      block_shape=($(BT), $(BS)), order=(1, 0))
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

<details><summary><code>active</code></summary>

```lean
def active (s : BlockState) (T S BT BS : Nat) (idx : TileIndex [BT, BS]) : Prop :=
  tIndex s BT idx.1 < T ∧ sIndex s BS idx.2.1 < S
```
</details>

<details><summary><code>tileOffset</code></summary>

```lean
def tileOffset (s : BlockState) (s_s_h s_s_t s_s_d BT BS : Nat)
    (idx : TileIndex [BT, BS]) : Nat :=
  s.pids 1 * s_s_h + tIndex s BT idx.1 * s_s_t +
    sIndex s BS idx.2.1 * s_s_d
```
</details>

<details><summary><code>storeValue</code></summary>

```lean
noncomputable def storeValue (s : BlockState) (BC : RegionName)
    (s_s_h s_s_t s_s_d T S BT BS : Nat) (idx : TileIndex [BT, BS]) : ℝ :=
  WithBot.unbotD 0
    (if active s T S BT BS idx then
      some (s.readMem BC (tileOffset s s_s_h s_s_t s_s_d BT BS idx))
    else some (0.0 : ℝ))
```
</details>

<details><summary><code>chunk_cumsum_vector_cumsum_slice</code></summary>

```
/-! ## Computed per-chunk cumsum slice

The store slice above starts from a precomputed `BC` tile. This slice models one
loop iteration of the Python kernel after the carry vector `b_z` has been
materialized in `Carry`: it loads the source block, computes
`b_z[None, :] + tl.dot(m_s, b_s, allow_tf32=false)`, and stores the masked
result. -/
```
```lean
def chunk_cumsum_vector_cumsum_slice
    (SReg Carry Z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat) :
    ComputeKernel := triton {
  i_s = tl.program_id(0)
  i_bh = tl.program_id(1)
  i_t = tl.program_id(2)
  offs_t = i_t * $(BT) + tl.arange(0, $(BT))
  offs_s = i_s * $(BS) + tl.arange(0, $(BS))
  o_i = tl.arange(0, $(BT))
  m_s = tl.where(o_i[:, None] >= o_i[None, :], 1.0, 0.0)
  mask = (offs_t[:, None] < $(T)) & (offs_s[None, :] < $(S))
  b_s = tl.load(SReg + i_bh * $(s_s_h) + offs_t[:, None] * $(s_s_t) +
      offs_s[None, :] * $(s_s_d), mask=mask, other=0.0)
  b_z = tl.load(Carry + i_bh * $(s_s_h) + offs_s * $(s_s_d),
    mask=offs_s < $(S), other=0.0)
  b_c = b_z[None, :] + tl.dot(m_s, b_s, allow_tf32=false)
  tl.store(Z + i_bh * $(s_s_h) + offs_t[:, None] * $(s_s_t) +
      offs_s[None, :] * $(s_s_d), b_c, mask=mask)
}
```
</details>

<details><summary><code>globalCumsumVectorClosed</code></summary>

```
/-! ## Genuine chunked-cumsum closed form (vector / per-column)

`globalCumsumVectorClosed` is the genuine mathematical specification — for each
feature column `j` it is a `Finset.sum` over all *flat* time indices
`flat ≤ i_t·BT + i` (with `flat < T`) of the source value at the 2D address
`i_bh·s_s_h + flat·s_s_t + (i_s·BS + j)·s_s_d`. It is **not** a read-back of the
kernel's own output, so realizing it is a true correctness statement. Padded
feature lanes (`i_s·BS + j ≥ S`) hold `0`, matching the masked store. This is
the 2D analogue of the scalar `globalCumsumClosed`, carrying the column index
`j` through the prefix sum. -/
```
```lean
noncomputable def globalCumsumVectorClosed
    (s : BlockState) (SReg : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat)
    (idx : TileIndex [BT, BS]) : ℝ :=
  if sIndex s BS idx.2.1 < S then
    ∑ flat ∈ (Finset.range T).filter
        (fun flat => flat ≤ s.pids 2 * BT + idx.1.val),
      s.readMem SReg (s.pids 1 * s_s_h + flat * s_s_t +
        sIndex s BS idx.2.1 * s_s_d)
  else 0
```
</details>

<details><summary><code>tIndex</code></summary>

```lean
def tIndex (s : BlockState) (BT : Nat) (i : Fin BT) : Nat :=
  s.pids 2 * BT + i.val
```
</details>

## Also present (pinned special-case summaries)
- `chunk_cumsum_vector_store_slice_compute_correct`
- `chunk_cumsum_vector_single_block_surface_compute_correct`
- `chunk_cumsum_vector_single_block_surface_active_compute_correct`
- `chunk_cumsum_vector_cumsum_slice_compute_correct`
- `chunk_cumsum_vector_store_slice_active_compute_correct`
- `chunk_cumsum_vector_cumsum_slice_active_compute_correct`
- `chunk_cumsum_vector_single_block_python_case1_compute_correct`
- `chunk_cumsum_vector_single_block_python_case2_compute_correct`
- `chunk_cumsum_vector_single_block_python_case3_compute_correct`
- `chunk_cumsum_vector_store_python_case1_compute_correct`
- `chunk_cumsum_vector_cumsum_python_case1_compute_correct`
- `chunk_cumsum_vector_store_python_case2_compute_correct`
- `chunk_cumsum_vector_cumsum_python_case2_compute_correct`
- `chunk_cumsum_vector_store_python_case3_compute_correct`
- `chunk_cumsum_vector_cumsum_python_case3_compute_correct`
