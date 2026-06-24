# Spec sheet — `bench/tritonbench_g/reversed_cumsum/ReversedCumsum.lean`

**Python source:** `bench/tritonbench_g/reversed_cumsum/reversed_cumsum.py`

## Public theorem: `reversed_cumsum_python_case1_store_summary`

<details><summary>docstring</summary>

```
/-- Public Python case 1 summary: the full reversed-cumsum surface lowers and
both store/cumsum output slices are checked for the contiguous `T = 4`,
`S = 5` shape. -/
```
</details>

**Statement:**
```lean
theorem reversed_cumsum_python_case1_store_summary
    (BC SReg Carry Z : RegionName) (s : BlockState) :
    (∃ alg, (reversed_cumsum_surface SReg Z 20 5 1 4 5 16 32).toAlgorithm? =
      Except.ok alg) ∧
    ((ComputeCorrect.Realizes
      (kernel := reversed_cumsum_store_slice BC Z 20 5 1 4 5 16 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 32] => active s 4 5 16 32 idx)
        (fun idx => (Z, tileOffset s 20 5 1 16 32 idx)))
      (expected := fun idx : TileIndex [16, 32] =>
        storeValue s BC 20 5 1 4 5 16 32 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := reversed_cumsum_cumsum_slice SReg Carry Z 20 5 1 4 5 16 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 32] => active s 4 5 16 32 idx)
        (fun idx => (Z, tileOffset s 20 5 1 16 32 idx)))
      (expected := fun idx : TileIndex [16, 32] =>
        cumsumStoreValue s SReg Carry 20 5 1 4 5 16 32 idx)))
```

**Assumptions / layout contracts:**
- `kernel : = reversed_cumsum_store_slice BC Z 20 5 1 4 5 16 32`
- `initialState : = s`
- `fun idx : TileIndex [16, 32] => active s 4 5 16 32 idx`
- `expected : = fun idx : TileIndex [16, 32] =>
        storeValue s BC 20 5 1 4 5 16 32 idx`
- `kernel : = reversed_cumsum_cumsum_slice SReg Carry Z 20 5 1 4 5 16 32`
- `initialState : = s`
- `fun idx : TileIndex [16, 32] => active s 4 5 16 32 idx`
- `expected : = fun idx : TileIndex [16, 32] =>
        cumsumStoreValue s SReg Carry 20 5 1 4 5 16 32 idx`

**Closed-form spec defs (transitive):** `reversed_cumsum_surface`, `reversed_cumsum_store_slice`, `active`, `tileOffset`, `storeValue`, `reversed_cumsum_cumsum_slice`, `cumsumStoreValue`, `tIndex`, `sIndex`, `carryValue`, `upperTriTile`, `sourceTile`

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

<details><summary><code>reversed_cumsum_cumsum_slice</code></summary>

```
/-! ## Computed per-chunk reversed-cumsum slice

This slice models one reverse loop iteration after the carry vector `b_z` has
been materialized in `Carry`: it loads the source block, computes
`b_z[None, :] + tl.dot(m_s, b_s, allow_tf32=false)` with the upper-triangular
reverse-cumsum mask, and stores the masked result. -/
```
```lean
def reversed_cumsum_cumsum_slice
    (SReg Carry Z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat) :
    ComputeKernel := triton {
  i_s = tl.program_id(0)
  i_bh = tl.program_id(1)
  i_t = tl.program_id(2)
  offs_t = i_t * $(BT) + tl.arange(0, $(BT))
  offs_s = i_s * $(BS) + tl.arange(0, $(BS))
  o_i = tl.arange(0, $(BT))
  m_s = tl.where(o_i[:, None] <= o_i[None, :], 1.0, 0.0)
  mask = (offs_t[:, None] < $(T)) & (offs_s[None, :] < $(S))
  b_s = tl.load(SReg + i_bh * $(s_s_h) + offs_t[:, None] * $(s_s_t) +
      offs_s[None, :] * $(s_s_d), mask=mask, other=0.0)
  b_z = tl.load(Carry + i_bh * $(s_s_h) + offs_s * $(s_s_d),
    mask=offs_s < $(S), other=0.0)
  b_c = b_z[None, :] + tl.dot(m_s, b_s, allow_tf32=false)
  tl.store(Z + i_bh * $(s_s_h) + offs_t[:, None] * $(s_s_t) +
      offs_s[None, :] * $(s_s_d), (b_c).to(Z.dtype.element_ty), mask=mask)
}
```
</details>

<details><summary><code>cumsumStoreValue</code></summary>

```lean
noncomputable def cumsumStoreValue
    (s : BlockState) (SReg Carry : RegionName)
    (s_s_h s_s_t s_s_d T S BT BS : Nat)
    (idx : TileIndex [BT, BS]) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun carry dot : ℝ => carry + dot)
      (carryValue s Carry s_s_h s_s_d S BS idx.2.1)
      ((Tile.dot [] (upperTriTile BT)
        (sourceTile s SReg s_s_h s_s_t s_s_d T S BT BS)).data
          (idx.1, idx.2.1, PUnit.unit)))
```
</details>

<details><summary><code>tIndex</code></summary>

```lean
def tIndex (s : BlockState) (BT : Nat) (i : Fin BT) : Nat :=
  s.pids 2 * BT + i.val
```
</details>

<details><summary><code>sIndex</code></summary>

```lean
def sIndex (s : BlockState) (BS : Nat) (j : Fin BS) : Nat :=
  s.pids 0 * BS + j.val
```
</details>

<details><summary><code>carryValue</code></summary>

```lean
noncomputable def carryValue
    (s : BlockState) (Carry : RegionName) (s_s_h s_s_d S BS : Nat)
    (j : Fin BS) : WithBot ℝ :=
  if sIndex s BS j < S then
    some (s.readMem Carry (s.pids 1 * s_s_h + sIndex s BS j * s_s_d))
  else some (0.0 : ℝ)
```
</details>

<details><summary><code>upperTriTile</code></summary>

```lean
noncomputable def upperTriTile (BT : Nat) : Tile .real [BT, BT] :=
  { data := fun idx =>
      if idx.1.val <= idx.2.1.val then some (1.0 : ℝ) else some (0.0 : ℝ) }
```
</details>

<details><summary><code>sourceTile</code></summary>

```lean
noncomputable def sourceTile
    (s : BlockState) (SReg : RegionName)
    (s_s_h s_s_t s_s_d T S BT BS : Nat) :
    Tile .real [BT, BS] :=
  { data := fun idx =>
      if active s T S BT BS idx then
        some (s.readMem SReg (tileOffset s s_s_h s_s_t s_s_d BT BS idx))
      else some (0.0 : ℝ) }
```
</details>

## Public theorem: `reversed_cumsum_python_case1_output_summary`

<details><summary>docstring</summary>

```
/-- **Genuine Python case 1 summary** (`B=2, H=3, T=4, S=5`, strides
`(20,5,1)`). The full reverse-traversal surface lowers, and — since `T = 4 ≤ BT`
the loop runs a single chunk with carry `b_z = 0` — the (faithful single-chunk)
surface writes into every active lane `(i, j)` the genuine reversed cumulative
sum `Σ_{k ≥ i, k < T} x[k, j]`. The `expected` value is `reversedCumsumClosed`,
a standalone `Finset.sum` specification, not a read-back of the kernel output. -/
```
</details>

**Statement:**
```lean
theorem reversed_cumsum_python_case1_output_summary
    (SReg Z : RegionName) (s : BlockState) :
    (∃ alg, (reversed_cumsum_surface SReg Z 20 5 1 4 5 16 32).toAlgorithm? =
      Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := reversed_cumsum_single_block_surface SReg Z 20 5 1 4 5 16 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 32] => singleBlockActive s 4 5 32 idx)
        (fun idx : TileIndex [16, 32] =>
          (Z, singleBlockTileOffset s 20 5 1 32 idx)))
      (expected := fun idx : TileIndex [16, 32] =>
        reversedCumsumClosed s SReg 20 5 1 4 5 16 32 idx))
```

**Assumptions / layout contracts:**
- `kernel : = reversed_cumsum_single_block_surface SReg Z 20 5 1 4 5 16 32`
- `initialState : = s`
- `fun idx : TileIndex [16, 32] => singleBlockActive s 4 5 32 idx`
- `fun idx : TileIndex [16, 32] =>
          (Z, singleBlockTileOffset s 20 5 1 32 idx)`
- `expected : = fun idx : TileIndex [16, 32] =>
        reversedCumsumClosed s SReg 20 5 1 4 5 16 32 idx`

**Closed-form spec defs (transitive):** `reversed_cumsum_surface`, `reversed_cumsum_single_block_surface`, `singleBlockActive`, `singleBlockTileOffset`, `reversedCumsumClosed`, `sIndex`

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

## Public theorem: `reversed_cumsum_python_case2_store_summary`

<details><summary>docstring</summary>

```
/-- Public Python case 2 summary: the full reversed-cumsum surface lowers and
both store/cumsum output slices are checked for the contiguous `T = 8`,
`S = 8` shape. -/
```
</details>

**Statement:**
```lean
theorem reversed_cumsum_python_case2_store_summary
    (BC SReg Carry Z : RegionName) (s : BlockState) :
    (∃ alg, (reversed_cumsum_surface SReg Z 64 8 1 8 8 16 32).toAlgorithm? =
      Except.ok alg) ∧
    ((ComputeCorrect.Realizes
      (kernel := reversed_cumsum_store_slice BC Z 64 8 1 8 8 16 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 32] => active s 8 8 16 32 idx)
        (fun idx => (Z, tileOffset s 64 8 1 16 32 idx)))
      (expected := fun idx : TileIndex [16, 32] =>
        storeValue s BC 64 8 1 8 8 16 32 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := reversed_cumsum_cumsum_slice SReg Carry Z 64 8 1 8 8 16 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 32] => active s 8 8 16 32 idx)
        (fun idx => (Z, tileOffset s 64 8 1 16 32 idx)))
      (expected := fun idx : TileIndex [16, 32] =>
        cumsumStoreValue s SReg Carry 64 8 1 8 8 16 32 idx)))
```

**Assumptions / layout contracts:**
- `kernel : = reversed_cumsum_store_slice BC Z 64 8 1 8 8 16 32`
- `initialState : = s`
- `fun idx : TileIndex [16, 32] => active s 8 8 16 32 idx`
- `expected : = fun idx : TileIndex [16, 32] =>
        storeValue s BC 64 8 1 8 8 16 32 idx`
- `kernel : = reversed_cumsum_cumsum_slice SReg Carry Z 64 8 1 8 8 16 32`
- `initialState : = s`
- `fun idx : TileIndex [16, 32] => active s 8 8 16 32 idx`
- `expected : = fun idx : TileIndex [16, 32] =>
        cumsumStoreValue s SReg Carry 64 8 1 8 8 16 32 idx`

**Closed-form spec defs (transitive):** `reversed_cumsum_surface`, `reversed_cumsum_store_slice`, `active`, `tileOffset`, `storeValue`, `reversed_cumsum_cumsum_slice`, `cumsumStoreValue`, `tIndex`, `sIndex`, `carryValue`, `upperTriTile`, `sourceTile`

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

<details><summary><code>reversed_cumsum_cumsum_slice</code></summary>

```
/-! ## Computed per-chunk reversed-cumsum slice

This slice models one reverse loop iteration after the carry vector `b_z` has
been materialized in `Carry`: it loads the source block, computes
`b_z[None, :] + tl.dot(m_s, b_s, allow_tf32=false)` with the upper-triangular
reverse-cumsum mask, and stores the masked result. -/
```
```lean
def reversed_cumsum_cumsum_slice
    (SReg Carry Z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat) :
    ComputeKernel := triton {
  i_s = tl.program_id(0)
  i_bh = tl.program_id(1)
  i_t = tl.program_id(2)
  offs_t = i_t * $(BT) + tl.arange(0, $(BT))
  offs_s = i_s * $(BS) + tl.arange(0, $(BS))
  o_i = tl.arange(0, $(BT))
  m_s = tl.where(o_i[:, None] <= o_i[None, :], 1.0, 0.0)
  mask = (offs_t[:, None] < $(T)) & (offs_s[None, :] < $(S))
  b_s = tl.load(SReg + i_bh * $(s_s_h) + offs_t[:, None] * $(s_s_t) +
      offs_s[None, :] * $(s_s_d), mask=mask, other=0.0)
  b_z = tl.load(Carry + i_bh * $(s_s_h) + offs_s * $(s_s_d),
    mask=offs_s < $(S), other=0.0)
  b_c = b_z[None, :] + tl.dot(m_s, b_s, allow_tf32=false)
  tl.store(Z + i_bh * $(s_s_h) + offs_t[:, None] * $(s_s_t) +
      offs_s[None, :] * $(s_s_d), (b_c).to(Z.dtype.element_ty), mask=mask)
}
```
</details>

<details><summary><code>cumsumStoreValue</code></summary>

```lean
noncomputable def cumsumStoreValue
    (s : BlockState) (SReg Carry : RegionName)
    (s_s_h s_s_t s_s_d T S BT BS : Nat)
    (idx : TileIndex [BT, BS]) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun carry dot : ℝ => carry + dot)
      (carryValue s Carry s_s_h s_s_d S BS idx.2.1)
      ((Tile.dot [] (upperTriTile BT)
        (sourceTile s SReg s_s_h s_s_t s_s_d T S BT BS)).data
          (idx.1, idx.2.1, PUnit.unit)))
```
</details>

<details><summary><code>tIndex</code></summary>

```lean
def tIndex (s : BlockState) (BT : Nat) (i : Fin BT) : Nat :=
  s.pids 2 * BT + i.val
```
</details>

<details><summary><code>sIndex</code></summary>

```lean
def sIndex (s : BlockState) (BS : Nat) (j : Fin BS) : Nat :=
  s.pids 0 * BS + j.val
```
</details>

<details><summary><code>carryValue</code></summary>

```lean
noncomputable def carryValue
    (s : BlockState) (Carry : RegionName) (s_s_h s_s_d S BS : Nat)
    (j : Fin BS) : WithBot ℝ :=
  if sIndex s BS j < S then
    some (s.readMem Carry (s.pids 1 * s_s_h + sIndex s BS j * s_s_d))
  else some (0.0 : ℝ)
```
</details>

<details><summary><code>upperTriTile</code></summary>

```lean
noncomputable def upperTriTile (BT : Nat) : Tile .real [BT, BT] :=
  { data := fun idx =>
      if idx.1.val <= idx.2.1.val then some (1.0 : ℝ) else some (0.0 : ℝ) }
```
</details>

<details><summary><code>sourceTile</code></summary>

```lean
noncomputable def sourceTile
    (s : BlockState) (SReg : RegionName)
    (s_s_h s_s_t s_s_d T S BT BS : Nat) :
    Tile .real [BT, BS] :=
  { data := fun idx =>
      if active s T S BT BS idx then
        some (s.readMem SReg (tileOffset s s_s_h s_s_t s_s_d BT BS idx))
      else some (0.0 : ℝ) }
```
</details>

## Public theorem: `reversed_cumsum_python_case2_output_summary`

<details><summary>docstring</summary>

```
/-- **Genuine Python case 2 summary** (`B=H=1, T=8, S=8`, strides `(64,8,1)`).
Full surface lowers; single-chunk (`T = 8 ≤ BT`) surface realizes the genuine
reversed cumulative sum `Σ_{k ≥ i, k < T} x[k, j]`. -/
```
</details>

**Statement:**
```lean
theorem reversed_cumsum_python_case2_output_summary
    (SReg Z : RegionName) (s : BlockState) :
    (∃ alg, (reversed_cumsum_surface SReg Z 64 8 1 8 8 16 32).toAlgorithm? =
      Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := reversed_cumsum_single_block_surface SReg Z 64 8 1 8 8 16 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 32] => singleBlockActive s 8 8 32 idx)
        (fun idx : TileIndex [16, 32] =>
          (Z, singleBlockTileOffset s 64 8 1 32 idx)))
      (expected := fun idx : TileIndex [16, 32] =>
        reversedCumsumClosed s SReg 64 8 1 8 8 16 32 idx))
```

**Assumptions / layout contracts:**
- `kernel : = reversed_cumsum_single_block_surface SReg Z 64 8 1 8 8 16 32`
- `initialState : = s`
- `fun idx : TileIndex [16, 32] => singleBlockActive s 8 8 32 idx`
- `fun idx : TileIndex [16, 32] =>
          (Z, singleBlockTileOffset s 64 8 1 32 idx)`
- `expected : = fun idx : TileIndex [16, 32] =>
        reversedCumsumClosed s SReg 64 8 1 8 8 16 32 idx`

**Closed-form spec defs (transitive):** `reversed_cumsum_surface`, `reversed_cumsum_single_block_surface`, `singleBlockActive`, `singleBlockTileOffset`, `reversedCumsumClosed`, `sIndex`

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

## Public theorem: `reversed_cumsum_python_case3_store_summary`

<details><summary>docstring</summary>

```
/-- Public Python case 3 summary: the full reversed-cumsum surface lowers and
both store/cumsum output slices are checked for the contiguous `T = 16`,
`S = 16` shape. -/
```
</details>

**Statement:**
```lean
theorem reversed_cumsum_python_case3_store_summary
    (BC SReg Carry Z : RegionName) (s : BlockState) :
    (∃ alg, (reversed_cumsum_surface SReg Z 256 16 1 16 16 16 32).toAlgorithm? =
      Except.ok alg) ∧
    ((ComputeCorrect.Realizes
      (kernel := reversed_cumsum_store_slice BC Z 256 16 1 16 16 16 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 32] => active s 16 16 16 32 idx)
        (fun idx => (Z, tileOffset s 256 16 1 16 32 idx)))
      (expected := fun idx : TileIndex [16, 32] =>
        storeValue s BC 256 16 1 16 16 16 32 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := reversed_cumsum_cumsum_slice SReg Carry Z 256 16 1 16 16 16 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 32] => active s 16 16 16 32 idx)
        (fun idx => (Z, tileOffset s 256 16 1 16 32 idx)))
      (expected := fun idx : TileIndex [16, 32] =>
        cumsumStoreValue s SReg Carry 256 16 1 16 16 16 32 idx)))
```

**Assumptions / layout contracts:**
- `kernel : = reversed_cumsum_store_slice BC Z 256 16 1 16 16 16 32`
- `initialState : = s`
- `fun idx : TileIndex [16, 32] => active s 16 16 16 32 idx`
- `expected : = fun idx : TileIndex [16, 32] =>
        storeValue s BC 256 16 1 16 16 16 32 idx`
- `kernel : = reversed_cumsum_cumsum_slice SReg Carry Z 256 16 1 16 16 16 32`
- `initialState : = s`
- `fun idx : TileIndex [16, 32] => active s 16 16 16 32 idx`
- `expected : = fun idx : TileIndex [16, 32] =>
        cumsumStoreValue s SReg Carry 256 16 1 16 16 16 32 idx`

**Closed-form spec defs (transitive):** `reversed_cumsum_surface`, `reversed_cumsum_store_slice`, `active`, `tileOffset`, `storeValue`, `reversed_cumsum_cumsum_slice`, `cumsumStoreValue`, `tIndex`, `sIndex`, `carryValue`, `upperTriTile`, `sourceTile`

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

<details><summary><code>reversed_cumsum_cumsum_slice</code></summary>

```
/-! ## Computed per-chunk reversed-cumsum slice

This slice models one reverse loop iteration after the carry vector `b_z` has
been materialized in `Carry`: it loads the source block, computes
`b_z[None, :] + tl.dot(m_s, b_s, allow_tf32=false)` with the upper-triangular
reverse-cumsum mask, and stores the masked result. -/
```
```lean
def reversed_cumsum_cumsum_slice
    (SReg Carry Z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat) :
    ComputeKernel := triton {
  i_s = tl.program_id(0)
  i_bh = tl.program_id(1)
  i_t = tl.program_id(2)
  offs_t = i_t * $(BT) + tl.arange(0, $(BT))
  offs_s = i_s * $(BS) + tl.arange(0, $(BS))
  o_i = tl.arange(0, $(BT))
  m_s = tl.where(o_i[:, None] <= o_i[None, :], 1.0, 0.0)
  mask = (offs_t[:, None] < $(T)) & (offs_s[None, :] < $(S))
  b_s = tl.load(SReg + i_bh * $(s_s_h) + offs_t[:, None] * $(s_s_t) +
      offs_s[None, :] * $(s_s_d), mask=mask, other=0.0)
  b_z = tl.load(Carry + i_bh * $(s_s_h) + offs_s * $(s_s_d),
    mask=offs_s < $(S), other=0.0)
  b_c = b_z[None, :] + tl.dot(m_s, b_s, allow_tf32=false)
  tl.store(Z + i_bh * $(s_s_h) + offs_t[:, None] * $(s_s_t) +
      offs_s[None, :] * $(s_s_d), (b_c).to(Z.dtype.element_ty), mask=mask)
}
```
</details>

<details><summary><code>cumsumStoreValue</code></summary>

```lean
noncomputable def cumsumStoreValue
    (s : BlockState) (SReg Carry : RegionName)
    (s_s_h s_s_t s_s_d T S BT BS : Nat)
    (idx : TileIndex [BT, BS]) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun carry dot : ℝ => carry + dot)
      (carryValue s Carry s_s_h s_s_d S BS idx.2.1)
      ((Tile.dot [] (upperTriTile BT)
        (sourceTile s SReg s_s_h s_s_t s_s_d T S BT BS)).data
          (idx.1, idx.2.1, PUnit.unit)))
```
</details>

<details><summary><code>tIndex</code></summary>

```lean
def tIndex (s : BlockState) (BT : Nat) (i : Fin BT) : Nat :=
  s.pids 2 * BT + i.val
```
</details>

<details><summary><code>sIndex</code></summary>

```lean
def sIndex (s : BlockState) (BS : Nat) (j : Fin BS) : Nat :=
  s.pids 0 * BS + j.val
```
</details>

<details><summary><code>carryValue</code></summary>

```lean
noncomputable def carryValue
    (s : BlockState) (Carry : RegionName) (s_s_h s_s_d S BS : Nat)
    (j : Fin BS) : WithBot ℝ :=
  if sIndex s BS j < S then
    some (s.readMem Carry (s.pids 1 * s_s_h + sIndex s BS j * s_s_d))
  else some (0.0 : ℝ)
```
</details>

<details><summary><code>upperTriTile</code></summary>

```lean
noncomputable def upperTriTile (BT : Nat) : Tile .real [BT, BT] :=
  { data := fun idx =>
      if idx.1.val <= idx.2.1.val then some (1.0 : ℝ) else some (0.0 : ℝ) }
```
</details>

<details><summary><code>sourceTile</code></summary>

```lean
noncomputable def sourceTile
    (s : BlockState) (SReg : RegionName)
    (s_s_h s_s_t s_s_d T S BT BS : Nat) :
    Tile .real [BT, BS] :=
  { data := fun idx =>
      if active s T S BT BS idx then
        some (s.readMem SReg (tileOffset s s_s_h s_s_t s_s_d BT BS idx))
      else some (0.0 : ℝ) }
```
</details>

## Public theorem: `reversed_cumsum_python_case3_output_summary`

<details><summary>docstring</summary>

```
/-- **Genuine Python case 3 summary** (`B=4, H=2, T=16, S=16`, strides
`(256,16,1)`). Full surface lowers; single-chunk (`T = 16 ≤ BT`) surface
realizes the genuine reversed cumulative sum `Σ_{k ≥ i, k < T} x[k, j]`. -/
```
</details>

**Statement:**
```lean
theorem reversed_cumsum_python_case3_output_summary
    (SReg Z : RegionName) (s : BlockState) :
    (∃ alg, (reversed_cumsum_surface SReg Z 256 16 1 16 16 16 32).toAlgorithm? =
      Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := reversed_cumsum_single_block_surface SReg Z 256 16 1 16 16 16 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 32] => singleBlockActive s 16 16 32 idx)
        (fun idx : TileIndex [16, 32] =>
          (Z, singleBlockTileOffset s 256 16 1 32 idx)))
      (expected := fun idx : TileIndex [16, 32] =>
        reversedCumsumClosed s SReg 256 16 1 16 16 16 32 idx))
```

**Assumptions / layout contracts:**
- `kernel : = reversed_cumsum_single_block_surface SReg Z 256 16 1 16 16 16 32`
- `initialState : = s`
- `fun idx : TileIndex [16, 32] => singleBlockActive s 16 16 32 idx`
- `fun idx : TileIndex [16, 32] =>
          (Z, singleBlockTileOffset s 256 16 1 32 idx)`
- `expected : = fun idx : TileIndex [16, 32] =>
        reversedCumsumClosed s SReg 256 16 1 16 16 16 32 idx`

**Closed-form spec defs (transitive):** `reversed_cumsum_surface`, `reversed_cumsum_single_block_surface`, `singleBlockActive`, `singleBlockTileOffset`, `reversedCumsumClosed`, `sIndex`

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

## Public theorem: `reversed_cumsum_python_case4_store_summary`

<details><summary>docstring</summary>

```
/-- Public Python case 4 summary: the full reversed-cumsum surface lowers and
both store/cumsum output slices are checked for the two-chunk `T = 32`,
`S = 32` shape. -/
```
</details>

**Statement:**
```lean
theorem reversed_cumsum_python_case4_store_summary
    (BC SReg Carry Z : RegionName) (s : BlockState) :
    (∃ alg, (reversed_cumsum_surface SReg Z 1024 32 1 32 32 16 32).toAlgorithm? =
      Except.ok alg) ∧
    ((ComputeCorrect.Realizes
      (kernel := reversed_cumsum_store_slice BC Z 1024 32 1 32 32 16 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 32] => active s 32 32 16 32 idx)
        (fun idx : TileIndex [16, 32] => (Z, tileOffset s 1024 32 1 16 32 idx)))
      (expected := fun idx : TileIndex [16, 32] =>
        storeValue s BC 1024 32 1 32 32 16 32 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := reversed_cumsum_cumsum_slice SReg Carry Z 1024 32 1 32 32 16 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 32] => active s 32 32 16 32 idx)
        (fun idx : TileIndex [16, 32] => (Z, tileOffset s 1024 32 1 16 32 idx)))
      (expected := fun idx : TileIndex [16, 32] =>
        cumsumStoreValue s SReg Carry 1024 32 1 32 32 16 32 idx)))
```

**Assumptions / layout contracts:**
- `kernel : = reversed_cumsum_store_slice BC Z 1024 32 1 32 32 16 32`
- `initialState : = s`
- `fun idx : TileIndex [16, 32] => active s 32 32 16 32 idx`
- `fun idx : TileIndex [16, 32] => (Z, tileOffset s 1024 32 1 16 32 idx)`
- `expected : = fun idx : TileIndex [16, 32] =>
        storeValue s BC 1024 32 1 32 32 16 32 idx`
- `kernel : = reversed_cumsum_cumsum_slice SReg Carry Z 1024 32 1 32 32 16 32`
- `initialState : = s`
- `fun idx : TileIndex [16, 32] => active s 32 32 16 32 idx`
- `fun idx : TileIndex [16, 32] => (Z, tileOffset s 1024 32 1 16 32 idx)`
- `expected : = fun idx : TileIndex [16, 32] =>
        cumsumStoreValue s SReg Carry 1024 32 1 32 32 16 32 idx`

**Closed-form spec defs (transitive):** `reversed_cumsum_surface`, `reversed_cumsum_store_slice`, `active`, `tileOffset`, `storeValue`, `reversed_cumsum_cumsum_slice`, `cumsumStoreValue`, `tIndex`, `sIndex`, `carryValue`, `upperTriTile`, `sourceTile`

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

<details><summary><code>reversed_cumsum_cumsum_slice</code></summary>

```
/-! ## Computed per-chunk reversed-cumsum slice

This slice models one reverse loop iteration after the carry vector `b_z` has
been materialized in `Carry`: it loads the source block, computes
`b_z[None, :] + tl.dot(m_s, b_s, allow_tf32=false)` with the upper-triangular
reverse-cumsum mask, and stores the masked result. -/
```
```lean
def reversed_cumsum_cumsum_slice
    (SReg Carry Z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat) :
    ComputeKernel := triton {
  i_s = tl.program_id(0)
  i_bh = tl.program_id(1)
  i_t = tl.program_id(2)
  offs_t = i_t * $(BT) + tl.arange(0, $(BT))
  offs_s = i_s * $(BS) + tl.arange(0, $(BS))
  o_i = tl.arange(0, $(BT))
  m_s = tl.where(o_i[:, None] <= o_i[None, :], 1.0, 0.0)
  mask = (offs_t[:, None] < $(T)) & (offs_s[None, :] < $(S))
  b_s = tl.load(SReg + i_bh * $(s_s_h) + offs_t[:, None] * $(s_s_t) +
      offs_s[None, :] * $(s_s_d), mask=mask, other=0.0)
  b_z = tl.load(Carry + i_bh * $(s_s_h) + offs_s * $(s_s_d),
    mask=offs_s < $(S), other=0.0)
  b_c = b_z[None, :] + tl.dot(m_s, b_s, allow_tf32=false)
  tl.store(Z + i_bh * $(s_s_h) + offs_t[:, None] * $(s_s_t) +
      offs_s[None, :] * $(s_s_d), (b_c).to(Z.dtype.element_ty), mask=mask)
}
```
</details>

<details><summary><code>cumsumStoreValue</code></summary>

```lean
noncomputable def cumsumStoreValue
    (s : BlockState) (SReg Carry : RegionName)
    (s_s_h s_s_t s_s_d T S BT BS : Nat)
    (idx : TileIndex [BT, BS]) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun carry dot : ℝ => carry + dot)
      (carryValue s Carry s_s_h s_s_d S BS idx.2.1)
      ((Tile.dot [] (upperTriTile BT)
        (sourceTile s SReg s_s_h s_s_t s_s_d T S BT BS)).data
          (idx.1, idx.2.1, PUnit.unit)))
```
</details>

<details><summary><code>tIndex</code></summary>

```lean
def tIndex (s : BlockState) (BT : Nat) (i : Fin BT) : Nat :=
  s.pids 2 * BT + i.val
```
</details>

<details><summary><code>sIndex</code></summary>

```lean
def sIndex (s : BlockState) (BS : Nat) (j : Fin BS) : Nat :=
  s.pids 0 * BS + j.val
```
</details>

<details><summary><code>carryValue</code></summary>

```lean
noncomputable def carryValue
    (s : BlockState) (Carry : RegionName) (s_s_h s_s_d S BS : Nat)
    (j : Fin BS) : WithBot ℝ :=
  if sIndex s BS j < S then
    some (s.readMem Carry (s.pids 1 * s_s_h + sIndex s BS j * s_s_d))
  else some (0.0 : ℝ)
```
</details>

<details><summary><code>upperTriTile</code></summary>

```lean
noncomputable def upperTriTile (BT : Nat) : Tile .real [BT, BT] :=
  { data := fun idx =>
      if idx.1.val <= idx.2.1.val then some (1.0 : ℝ) else some (0.0 : ℝ) }
```
</details>

<details><summary><code>sourceTile</code></summary>

```lean
noncomputable def sourceTile
    (s : BlockState) (SReg : RegionName)
    (s_s_h s_s_t s_s_d T S BT BS : Nat) :
    Tile .real [BT, BS] :=
  { data := fun idx =>
      if active s T S BT BS idx then
        some (s.readMem SReg (tileOffset s s_s_h s_s_t s_s_d BT BS idx))
      else some (0.0 : ℝ) }
```
</details>

## Public theorem: `reversed_cumsum_python_case4_output_summary`

<details><summary>docstring</summary>

```
/-- **Genuine Python case 4 summary** (`B=3, H=3, T=32, S=32`, strides
`(1024,32,1)`). This is the only bundled shape with two reverse chunks
(`cdiv(32,16) = 2`), so it carries `b_z` across chunks. The summary records
genuine, non-self-referential content:

* the full two-chunk reverse-traversal surface lowers to the algorithm layer;
* each per-chunk store realizes the carried reversed prefix sum
  `b_z[j] + Σ_{k ≥ i, k < T} x[k, j]` (`cumsumStoreValue`, the value computed by
  `b_z[None,:] + tl.dot(m_s, b_s)`), with the fully injective `S = 32` block map;
* the within-chunk matrix product `tl.dot(m_s, b_s)` is exactly the reversed
  cumulative sum `Σ_{k ≥ i, k < T} x[k, j]`
  (`singleBlockStoreValue_eq_reversedCumsumClosed`), the mathematical core of the
  directional scan. -/
```
</details>

**Statement:**
```lean
theorem reversed_cumsum_python_case4_output_summary
    (SReg Carry Z : RegionName) (s : BlockState) :
    (∃ alg, (reversed_cumsum_surface SReg Z 1024 32 1 32 32 16 32).toAlgorithm? =
      Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := reversed_cumsum_cumsum_slice SReg Carry Z 1024 32 1 32 32 16 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 32] => active s 32 32 16 32 idx)
        (fun idx : TileIndex [16, 32] => (Z, tileOffset s 1024 32 1 16 32 idx)))
      (expected := fun idx : TileIndex [16, 32] =>
        cumsumStoreValue s SReg Carry 1024 32 1 32 32 16 32 idx)) ∧
    (∀ idx : TileIndex [16, 32],
      singleBlockStoreValue s SReg 1024 32 1 32 32 16 32 idx
        = reversedCumsumClosed s SReg 1024 32 1 32 32 16 32 idx)
```

**Assumptions / layout contracts:**
- `kernel : = reversed_cumsum_cumsum_slice SReg Carry Z 1024 32 1 32 32 16 32`
- `initialState : = s`
- `fun idx : TileIndex [16, 32] => active s 32 32 16 32 idx`
- `fun idx : TileIndex [16, 32] => (Z, tileOffset s 1024 32 1 16 32 idx)`
- `expected : = fun idx : TileIndex [16, 32] =>
        cumsumStoreValue s SReg Carry 1024 32 1 32 32 16 32 idx`
- `∀ idx : TileIndex [16, 32],
      singleBlockStoreValue s SReg 1024 32 1 32 32 16 32 idx
        = reversedCumsumClosed s SReg 1024 32 1 32 32 16 32 idx`

**Closed-form spec defs (transitive):** `reversed_cumsum_surface`, `reversed_cumsum_cumsum_slice`, `active`, `tileOffset`, `cumsumStoreValue`, `singleBlockStoreValue`, `reversedCumsumClosed`, `tIndex`, `sIndex`, `carryValue`, `upperTriTile`, `sourceTile`, `singleBlockSourceTile`, `singleBlockTileOffset`, `singleBlockActive`

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

<details><summary><code>reversed_cumsum_cumsum_slice</code></summary>

```
/-! ## Computed per-chunk reversed-cumsum slice

This slice models one reverse loop iteration after the carry vector `b_z` has
been materialized in `Carry`: it loads the source block, computes
`b_z[None, :] + tl.dot(m_s, b_s, allow_tf32=false)` with the upper-triangular
reverse-cumsum mask, and stores the masked result. -/
```
```lean
def reversed_cumsum_cumsum_slice
    (SReg Carry Z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat) :
    ComputeKernel := triton {
  i_s = tl.program_id(0)
  i_bh = tl.program_id(1)
  i_t = tl.program_id(2)
  offs_t = i_t * $(BT) + tl.arange(0, $(BT))
  offs_s = i_s * $(BS) + tl.arange(0, $(BS))
  o_i = tl.arange(0, $(BT))
  m_s = tl.where(o_i[:, None] <= o_i[None, :], 1.0, 0.0)
  mask = (offs_t[:, None] < $(T)) & (offs_s[None, :] < $(S))
  b_s = tl.load(SReg + i_bh * $(s_s_h) + offs_t[:, None] * $(s_s_t) +
      offs_s[None, :] * $(s_s_d), mask=mask, other=0.0)
  b_z = tl.load(Carry + i_bh * $(s_s_h) + offs_s * $(s_s_d),
    mask=offs_s < $(S), other=0.0)
  b_c = b_z[None, :] + tl.dot(m_s, b_s, allow_tf32=false)
  tl.store(Z + i_bh * $(s_s_h) + offs_t[:, None] * $(s_s_t) +
      offs_s[None, :] * $(s_s_d), (b_c).to(Z.dtype.element_ty), mask=mask)
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

<details><summary><code>cumsumStoreValue</code></summary>

```lean
noncomputable def cumsumStoreValue
    (s : BlockState) (SReg Carry : RegionName)
    (s_s_h s_s_t s_s_d T S BT BS : Nat)
    (idx : TileIndex [BT, BS]) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun carry dot : ℝ => carry + dot)
      (carryValue s Carry s_s_h s_s_d S BS idx.2.1)
      ((Tile.dot [] (upperTriTile BT)
        (sourceTile s SReg s_s_h s_s_t s_s_d T S BT BS)).data
          (idx.1, idx.2.1, PUnit.unit)))
```
</details>

<details><summary><code>singleBlockStoreValue</code></summary>

```lean
noncomputable def singleBlockStoreValue
    (s : BlockState) (SReg : RegionName)
    (s_s_h s_s_t s_s_d T S BT BS : Nat)
    (idx : TileIndex [BT, BS]) : ℝ :=
  WithBot.unbotD 0
    ((Tile.dot [] (upperTriTile BT)
      (singleBlockSourceTile s SReg s_s_h s_s_t s_s_d T S BT BS)).data
        (idx.1, idx.2.1, PUnit.unit))
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

<details><summary><code>tIndex</code></summary>

```lean
def tIndex (s : BlockState) (BT : Nat) (i : Fin BT) : Nat :=
  s.pids 2 * BT + i.val
```
</details>

<details><summary><code>sIndex</code></summary>

```lean
def sIndex (s : BlockState) (BS : Nat) (j : Fin BS) : Nat :=
  s.pids 0 * BS + j.val
```
</details>

<details><summary><code>carryValue</code></summary>

```lean
noncomputable def carryValue
    (s : BlockState) (Carry : RegionName) (s_s_h s_s_d S BS : Nat)
    (j : Fin BS) : WithBot ℝ :=
  if sIndex s BS j < S then
    some (s.readMem Carry (s.pids 1 * s_s_h + sIndex s BS j * s_s_d))
  else some (0.0 : ℝ)
```
</details>

<details><summary><code>upperTriTile</code></summary>

```lean
noncomputable def upperTriTile (BT : Nat) : Tile .real [BT, BT] :=
  { data := fun idx =>
      if idx.1.val <= idx.2.1.val then some (1.0 : ℝ) else some (0.0 : ℝ) }
```
</details>

<details><summary><code>sourceTile</code></summary>

```lean
noncomputable def sourceTile
    (s : BlockState) (SReg : RegionName)
    (s_s_h s_s_t s_s_d T S BT BS : Nat) :
    Tile .real [BT, BS] :=
  { data := fun idx =>
      if active s T S BT BS idx then
        some (s.readMem SReg (tileOffset s s_s_h s_s_t s_s_d BT BS idx))
      else some (0.0 : ℝ) }
```
</details>

<details><summary><code>singleBlockSourceTile</code></summary>

```lean
noncomputable def singleBlockSourceTile
    (s : BlockState) (SReg : RegionName)
    (s_s_h s_s_t s_s_d T S BT BS : Nat) :
    Tile .real [BT, BS] :=
  { data := fun idx =>
      if singleBlockActive s T S BS idx then
        some (s.readMem SReg (singleBlockTileOffset s s_s_h s_s_t s_s_d BS idx))
      else some (0.0 : ℝ) }
```
</details>

<details><summary><code>singleBlockTileOffset</code></summary>

```lean
def singleBlockTileOffset (s : BlockState) (s_s_h s_s_t s_s_d BS : Nat)
    (idx : TileIndex [BT, BS]) : Nat :=
  s.pids 1 * s_s_h + idx.1.val * s_s_t + sIndex s BS idx.2.1 * s_s_d
```
</details>

<details><summary><code>singleBlockActive</code></summary>

```lean
def singleBlockActive (s : BlockState) (T S BS : Nat)
    (idx : TileIndex [BT, BS]) : Prop :=
  idx.1.val < T ∧ sIndex s BS idx.2.1 < S
```
</details>

## Also present (pinned special-case summaries)
- `reversed_cumsum_store_slice_compute_correct`
- `reversed_cumsum_single_block_surface_compute_correct`
- `reversed_cumsum_cumsum_slice_compute_correct`
- `reversed_cumsum_store_slice_active_compute_correct`
- `reversed_cumsum_cumsum_slice_active_compute_correct`
- `reversed_cumsum_store_python_case1_compute_correct`
- `reversed_cumsum_cumsum_python_case1_compute_correct`
- `reversed_cumsum_store_python_case2_compute_correct`
- `reversed_cumsum_cumsum_python_case2_compute_correct`
- `reversed_cumsum_store_python_case3_compute_correct`
- `reversed_cumsum_cumsum_python_case3_compute_correct`
- `reversed_cumsum_store_python_case4_compute_correct`
- `reversed_cumsum_cumsum_python_case4_compute_correct`
- `reversed_cumsum_python_case1_all_outputs_compute_correct`
- `reversed_cumsum_python_case2_all_outputs_compute_correct`
- `reversed_cumsum_python_case3_all_outputs_compute_correct`
- `reversed_cumsum_python_case4_all_outputs_compute_correct`
