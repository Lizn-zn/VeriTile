# Spec sheet — `bench/tritonbench_g/attention_fwd_triton1/Correct.lean`

**Python source:** `bench/tritonbench_g/attention_fwd_triton1/attention_fwd_triton1.py`

## Public theorem: `linear_attention_correctness`

<details><summary>docstring</summary>

```
/-- The original implements the two independent formulas for all flag combinations. -/
```
</details>

**Statement:**
```lean
specification linear_attention_correctness (Q K V H O : RegionName) (s_qh s_hh s_ht : Nat)
    (scale : ℝ) (BT BD NT : Nat) (STORE IFCOND : Bool)
    (hBT : 0 < BT) (hstride : BD ≤ s_ht) :
    Spec.Real (originalIO Q K V H O s_qh s_hh s_ht scale BT BD NT STORE IFCOND ⊨
      fun _ _ _ xs ys zs => expected scale NT BT BD xs ys zs)
```

**Assumptions / layout contracts:**
- `hBT : 0 < BT`
- `hstride : BD ≤ s_ht`

**Closed-form spec defs (transitive):** `originalIO`, `expected`, `original`, `expectedOutput`, `expectedCarry`

<details><summary><code>originalIO</code></summary>

```
/-- Three input streams and two aggregate output windows, indexed by chunk
and matrix cell. H's write window is active only when STORE is enabled;
otherwise the frame preserves all H cells. -/
```
```lean
def originalIO (Q K V H O : RegionName) (s_qh s_hh s_ht : Nat)
    (scale : ℝ) (BT BD NT : Nat) (STORE IFCOND : Bool) : StreamMasked3DKernelIO₃ₓ₂ where
  kernel := (TypedSource.original Q K V H O s_qh BD 1 s_hh s_ht (NT * BT)
    scale BT BD NT STORE IFCOND).eraseDType
  projection := RealProjection.original_projection_supported ..
  inp1 := Q
  inp2 := K
  inp3 := V
  out1 := O
  out2 := H
  T := NT
  B1 := BT * BD
  B2 := BD * BT
  B3 := BT * BD
  C1 := NT * (BT * BD)
  C2 := NT * (BD * BD)
  out1DType := .real
  out2DType := .real
  pre := fun _ _ _ => Q ≠ O ∧ K ≠ O ∧ V ≠ O ∧ Q ≠ H ∧ K ≠ H ∧ V ≠ H ∧ H ≠ O
  read1 := fun p _ _ c j => p * s_qh + (c.val * BT + j.val / BD) * BD + j.val % BD
  read2 := fun p _ _ c j => p * s_qh + j.val / BT + (c.val * BT + j.val % BT) * BD
  read3 := fun p _ _ c j => p * s_qh + (c.val * BT + j.val / BD) * BD + j.val % BD
  write1 := fun p _ _ j =>
    let chunk := Lane2D.decode j
    let cell := Lane2D.decode chunk.2.1
    p * s_qh + (chunk.1.val * BT + cell.1.val) * BD + cell.2.1.val
  write2 := fun p _ _ j =>
    let chunk := Lane2D.decode j
    let cell := Lane2D.decode chunk.2.1
    p * s_hh + (chunk.1.val * BD + cell.1.val) * s_ht + cell.2.1.val
  mask1 := fun _ _ _ _ _ => True
  mask2 := fun _ _ _ _ _ => True
  mask3 := fun _ _ _ _ _ => True
  writeMask1 := fun _ _ _ _ => True
  writeMask2 := fun _ _ _ _ => STORE = Bool.true
```
</details>

<details><summary><code>expected</code></summary>

```lean
noncomputable def expected (scale : ℝ) (NT BT BD : Nat)
    (xs : Fin NT → Fin (BT * BD) → ℝ) (ys : Fin NT → Fin (BD * BT) → ℝ)
    (zs : Fin NT → Fin (BT * BD) → ℝ) :
    (Fin (NT * (BT * BD)) → ℝ) × (Fin (NT * (BD * BD)) → ℝ) :=
  (fun lane =>
    let chunk := Lane2D.decode lane
    let cell := Lane2D.decode chunk.2.1
    expectedOutput scale NT BT BD xs ys zs chunk.1 cell.1 cell.2.1,
   fun lane =>
    let chunk := Lane2D.decode lane
    let cell := Lane2D.decode chunk.2.1
    expectedCarry NT BT BD ys zs chunk.1 cell.1 cell.2.1)
```
</details>

<details><summary><code>original</code></summary>

```lean
def original
    (q k v h o : RegionName)
    (s_qh s_qt s_qd s_hh s_ht T : Nat) (scale : ℝ)
    (BT BD NT : Nat) (STORE IFCOND : Bool) :
    ComputeKernel := triton {
  tl.region q = tl.float32, k = tl.float32, v = tl.float32, h = tl.float32, o = tl.float32
  i_bh = tl.program_id(0)
  b_h = tl.zeros([$(BD), $(BD)], dtype=tl.float32)
  for i in range($(0), tl.cdiv($(T), $(BT))) {
    p_q = tl.make_block_ptr(base=q + i_bh * $(s_qh),
      shape=($(T), $(BD)), strides=($(s_qt), $(s_qd)),
      offsets=(i * $(BT), 0), block_shape=($(BT), $(BD)), order=(1, 0))
    p_k = tl.make_block_ptr(base=k + i_bh * $(s_qh),
      shape=($(BD), $(T)), strides=($(s_qd), $(s_qt)),
      offsets=(0, i * $(BT)), block_shape=($(BD), $(BT)), order=(0, 1))
    p_v = tl.make_block_ptr(base=v + i_bh * $(s_qh),
      shape=($(T), $(BD)), strides=($(s_qt), $(s_qd)),
      offsets=(i * $(BT), 0), block_shape=($(BT), $(BD)), order=(1, 0))
    p_h = tl.make_block_ptr(base=h + i_bh * $(s_hh),
      shape=($((NT * BD : Nat)), $(BD)), strides=($(s_ht), $(s_qd)),
      offsets=(i * $(BD), 0), block_shape=($(BD), $(BD)), order=(1, 0))
    p_o = tl.make_block_ptr(base=o + i_bh * $(s_qh),
      shape=($(T), $(BD)), strides=($(s_qt), $(s_qd)),
      offsets=(i * $(BT), 0), block_shape=($(BT), $(BD)), order=(1, 0))

    if STORE {
      tl.store(p_h, (b_h).to(tl.float32))
    }
    b_q = tl.load(p_q)
    b_q = (b_q * $((scale : ℝ))).to(tl.float32)
    b_k = tl.load(p_k)
    b_v = tl.load(p_v)

    b_s = tl.dot(b_q, b_k, allow_tf32=false)
    b_o = tl.dot((b_s).to(tl.float32), b_v, allow_tf32=false)
    if IFCOND {
      if i == $(0) {
        b_h = tl.dot(b_k, b_v, allow_tf32=false)
      } else {
        b_o += tl.dot(b_q, (b_h).to(tl.float32), allow_tf32=false)
        b_h += tl.dot(b_k, b_v, allow_tf32=false)
      }
    } else {
      b_o += tl.dot(b_q, (b_h).to(tl.float32), allow_tf32=false)
      b_h += tl.dot(b_k, b_v, allow_tf32=false)
    }

    tl.store(p_o, (b_o).to(tl.float32))
  }
}
```
</details>

<details><summary><code>expectedOutput</code></summary>

```
/-- O is the local contribution plus Q times the pre-update carry. -/
```
```lean
noncomputable def expectedOutput (scale : ℝ) (NT BT BD : Nat)
    (xs : Fin NT → Fin (BT * BD) → ℝ) (ys : Fin NT → Fin (BD * BT) → ℝ)
    (zs : Fin NT → Fin (BT * BD) → ℝ) (c : Fin NT) (i : Fin BT) (j : Fin BD) : ℝ :=
  (∑ t : Fin BT,
    (∑ e : Fin BD, (xs c (Lane2D.encode (i, e, PUnit.unit)) * scale) *
      ys c (Lane2D.encode (e, t, PUnit.unit))) * zs c (Lane2D.encode (t, j, PUnit.unit))) +
  ∑ e : Fin BD, (xs c (Lane2D.encode (i, e, PUnit.unit)) * scale) *
    expectedCarry NT BT BD ys zs c e j
```
</details>

<details><summary><code>expectedCarry</code></summary>

```
/-- H contains the sum of KᵀV from chunks strictly before this one. -/
```
```lean
noncomputable def expectedCarry (NT BT BD : Nat)
    (ys : Fin NT → Fin (BD * BT) → ℝ) (zs : Fin NT → Fin (BT * BD) → ℝ)
    (c : Fin NT) (i j : Fin BD) : ℝ :=
  ∑ past : Fin c.val, ∑ t : Fin BT,
    ys (Fin.castLE (Nat.le_of_lt c.isLt) past) (Lane2D.encode (i, t, PUnit.unit)) *
      zs (Fin.castLE (Nat.le_of_lt c.isLt) past) (Lane2D.encode (t, j, PUnit.unit))
```
</details>

## Public theorem: `linear_attention_scalar_left_correctness`

<details><summary>docstring</summary>

```
/-- The scalar-left candidate implements the same formulas and output masks. -/
```
</details>

**Statement:**
```lean
specification linear_attention_scalar_left_correctness (Q K V H O : RegionName) (s_qh s_hh s_ht : Nat)
    (scale : ℝ) (BT BD NT : Nat) (STORE IFCOND : Bool)
    (hBT : 0 < BT) (hstride : BD ≤ s_ht) :
    Spec.Real (scalarLeftIO Q K V H O s_qh s_hh s_ht scale BT BD NT STORE IFCOND ⊨
      fun _ _ _ xs ys zs => expected scale NT BT BD xs ys zs)
```

**Assumptions / layout contracts:**
- `hBT : 0 < BT`
- `hstride : BD ≤ s_ht`

**Closed-form spec defs (transitive):** `scalarLeftIO`, `expected`, `originalIO`, `scalarLeft`, `expectedOutput`, `expectedCarry`, `original`

<details><summary><code>scalarLeftIO</code></summary>

```lean
def scalarLeftIO (Q K V H O : RegionName) (s_qh s_hh s_ht : Nat)
    (scale : ℝ) (BT BD NT : Nat) (STORE IFCOND : Bool) : StreamMasked3DKernelIO₃ₓ₂ :=
  { originalIO Q K V H O s_qh s_hh s_ht scale BT BD NT STORE IFCOND with
    kernel := (TypedSource.scalarLeft Q K V H O s_qh BD 1 s_hh s_ht (NT * BT)
      scale BT BD NT STORE IFCOND).eraseDType
    projection := RealProjection.scalarLeft_projection_supported .. }
```
</details>

<details><summary><code>expected</code></summary>

```lean
noncomputable def expected (scale : ℝ) (NT BT BD : Nat)
    (xs : Fin NT → Fin (BT * BD) → ℝ) (ys : Fin NT → Fin (BD * BT) → ℝ)
    (zs : Fin NT → Fin (BT * BD) → ℝ) :
    (Fin (NT * (BT * BD)) → ℝ) × (Fin (NT * (BD * BD)) → ℝ) :=
  (fun lane =>
    let chunk := Lane2D.decode lane
    let cell := Lane2D.decode chunk.2.1
    expectedOutput scale NT BT BD xs ys zs chunk.1 cell.1 cell.2.1,
   fun lane =>
    let chunk := Lane2D.decode lane
    let cell := Lane2D.decode chunk.2.1
    expectedCarry NT BT BD ys zs chunk.1 cell.1 cell.2.1)
```
</details>

<details><summary><code>originalIO</code></summary>

```
/-- Three input streams and two aggregate output windows, indexed by chunk
and matrix cell. H's write window is active only when STORE is enabled;
otherwise the frame preserves all H cells. -/
```
```lean
def originalIO (Q K V H O : RegionName) (s_qh s_hh s_ht : Nat)
    (scale : ℝ) (BT BD NT : Nat) (STORE IFCOND : Bool) : StreamMasked3DKernelIO₃ₓ₂ where
  kernel := (TypedSource.original Q K V H O s_qh BD 1 s_hh s_ht (NT * BT)
    scale BT BD NT STORE IFCOND).eraseDType
  projection := RealProjection.original_projection_supported ..
  inp1 := Q
  inp2 := K
  inp3 := V
  out1 := O
  out2 := H
  T := NT
  B1 := BT * BD
  B2 := BD * BT
  B3 := BT * BD
  C1 := NT * (BT * BD)
  C2 := NT * (BD * BD)
  out1DType := .real
  out2DType := .real
  pre := fun _ _ _ => Q ≠ O ∧ K ≠ O ∧ V ≠ O ∧ Q ≠ H ∧ K ≠ H ∧ V ≠ H ∧ H ≠ O
  read1 := fun p _ _ c j => p * s_qh + (c.val * BT + j.val / BD) * BD + j.val % BD
  read2 := fun p _ _ c j => p * s_qh + j.val / BT + (c.val * BT + j.val % BT) * BD
  read3 := fun p _ _ c j => p * s_qh + (c.val * BT + j.val / BD) * BD + j.val % BD
  write1 := fun p _ _ j =>
    let chunk := Lane2D.decode j
    let cell := Lane2D.decode chunk.2.1
    p * s_qh + (chunk.1.val * BT + cell.1.val) * BD + cell.2.1.val
  write2 := fun p _ _ j =>
    let chunk := Lane2D.decode j
    let cell := Lane2D.decode chunk.2.1
    p * s_hh + (chunk.1.val * BD + cell.1.val) * s_ht + cell.2.1.val
  mask1 := fun _ _ _ _ _ => True
  mask2 := fun _ _ _ _ _ => True
  mask3 := fun _ _ _ _ _ => True
  writeMask1 := fun _ _ _ _ => True
  writeMask2 := fun _ _ _ _ => STORE = Bool.true
```
</details>

<details><summary><code>scalarLeft</code></summary>

```
/-- The loop-local rewrite requires finite actual operands at each iteration. -/
```
```lean
def scalarLeft
    (q k v h o : RegionName)
    (s_qh s_qt s_qd s_hh s_ht T : Nat) (scale : ℝ)
    (BT BD NT : Nat) (STORE IFCOND : Bool) :
    ComputeKernel := triton {
  tl.region q = tl.float32, k = tl.float32, v = tl.float32, h = tl.float32, o = tl.float32
  i_bh = tl.program_id(0)
  b_h = tl.zeros([$(BD), $(BD)], dtype=tl.float32)
  for i in range($(0), tl.cdiv($(T), $(BT))) {
    p_q = tl.make_block_ptr(base=q + i_bh * $(s_qh),
      shape=($(T), $(BD)), strides=($(s_qt), $(s_qd)),
      offsets=(i * $(BT), 0), block_shape=($(BT), $(BD)), order=(1, 0))
    p_k = tl.make_block_ptr(base=k + i_bh * $(s_qh),
      shape=($(BD), $(T)), strides=($(s_qd), $(s_qt)),
      offsets=(0, i * $(BT)), block_shape=($(BD), $(BT)), order=(0, 1))
    p_v = tl.make_block_ptr(base=v + i_bh * $(s_qh),
      shape=($(T), $(BD)), strides=($(s_qt), $(s_qd)),
      offsets=(i * $(BT), 0), block_shape=($(BT), $(BD)), order=(1, 0))
    p_h = tl.make_block_ptr(base=h + i_bh * $(s_hh),
      shape=($((NT * BD : Nat)), $(BD)), strides=($(s_ht), $(s_qd)),
      offsets=(i * $(BD), 0), block_shape=($(BD), $(BD)), order=(1, 0))
    p_o = tl.make_block_ptr(base=o + i_bh * $(s_qh),
      shape=($(T), $(BD)), strides=($(s_qt), $(s_qd)),
      offsets=(i * $(BT), 0), block_shape=($(BT), $(BD)), order=(1, 0))

    if STORE {
      tl.store(p_h, (b_h).to(tl.float32))
    }
    b_q = tl.load(p_q)
    b_q = ($((scale : ℝ)) * b_q).to(tl.float32)
    b_k = tl.load(p_k)
    b_v = tl.load(p_v)

    b_s = tl.dot(b_q, b_k, allow_tf32=false)
    b_o = tl.dot((b_s).to(tl.float32), b_v, allow_tf32=false)
    if IFCOND {
      if i == $(0) {
        b_h = tl.dot(b_k, b_v, allow_tf32=false)
      } else {
        b_o += tl.dot(b_q, (b_h).to(tl.float32), allow_tf32=false)
        b_h += tl.dot(b_k, b_v, allow_tf32=false)
      }
    } else {
      b_o += tl.dot(b_q, (b_h).to(tl.float32), allow_tf32=false)
      b_h += tl.dot(b_k, b_v, allow_tf32=false)
    }

    tl.store(p_o, (b_o).to(tl.float32))
  }
}
```
</details>

<details><summary><code>expectedOutput</code></summary>

```
/-- O is the local contribution plus Q times the pre-update carry. -/
```
```lean
noncomputable def expectedOutput (scale : ℝ) (NT BT BD : Nat)
    (xs : Fin NT → Fin (BT * BD) → ℝ) (ys : Fin NT → Fin (BD * BT) → ℝ)
    (zs : Fin NT → Fin (BT * BD) → ℝ) (c : Fin NT) (i : Fin BT) (j : Fin BD) : ℝ :=
  (∑ t : Fin BT,
    (∑ e : Fin BD, (xs c (Lane2D.encode (i, e, PUnit.unit)) * scale) *
      ys c (Lane2D.encode (e, t, PUnit.unit))) * zs c (Lane2D.encode (t, j, PUnit.unit))) +
  ∑ e : Fin BD, (xs c (Lane2D.encode (i, e, PUnit.unit)) * scale) *
    expectedCarry NT BT BD ys zs c e j
```
</details>

<details><summary><code>expectedCarry</code></summary>

```
/-- H contains the sum of KᵀV from chunks strictly before this one. -/
```
```lean
noncomputable def expectedCarry (NT BT BD : Nat)
    (ys : Fin NT → Fin (BD * BT) → ℝ) (zs : Fin NT → Fin (BT * BD) → ℝ)
    (c : Fin NT) (i j : Fin BD) : ℝ :=
  ∑ past : Fin c.val, ∑ t : Fin BT,
    ys (Fin.castLE (Nat.le_of_lt c.isLt) past) (Lane2D.encode (i, t, PUnit.unit)) *
      zs (Fin.castLE (Nat.le_of_lt c.isLt) past) (Lane2D.encode (t, j, PUnit.unit))
```
</details>

<details><summary><code>original</code></summary>

```lean
def original
    (q k v h o : RegionName)
    (s_qh s_qt s_qd s_hh s_ht T : Nat) (scale : ℝ)
    (BT BD NT : Nat) (STORE IFCOND : Bool) :
    ComputeKernel := triton {
  tl.region q = tl.float32, k = tl.float32, v = tl.float32, h = tl.float32, o = tl.float32
  i_bh = tl.program_id(0)
  b_h = tl.zeros([$(BD), $(BD)], dtype=tl.float32)
  for i in range($(0), tl.cdiv($(T), $(BT))) {
    p_q = tl.make_block_ptr(base=q + i_bh * $(s_qh),
      shape=($(T), $(BD)), strides=($(s_qt), $(s_qd)),
      offsets=(i * $(BT), 0), block_shape=($(BT), $(BD)), order=(1, 0))
    p_k = tl.make_block_ptr(base=k + i_bh * $(s_qh),
      shape=($(BD), $(T)), strides=($(s_qd), $(s_qt)),
      offsets=(0, i * $(BT)), block_shape=($(BD), $(BT)), order=(0, 1))
    p_v = tl.make_block_ptr(base=v + i_bh * $(s_qh),
      shape=($(T), $(BD)), strides=($(s_qt), $(s_qd)),
      offsets=(i * $(BT), 0), block_shape=($(BT), $(BD)), order=(1, 0))
    p_h = tl.make_block_ptr(base=h + i_bh * $(s_hh),
      shape=($((NT * BD : Nat)), $(BD)), strides=($(s_ht), $(s_qd)),
      offsets=(i * $(BD), 0), block_shape=($(BD), $(BD)), order=(1, 0))
    p_o = tl.make_block_ptr(base=o + i_bh * $(s_qh),
      shape=($(T), $(BD)), strides=($(s_qt), $(s_qd)),
      offsets=(i * $(BT), 0), block_shape=($(BT), $(BD)), order=(1, 0))

    if STORE {
      tl.store(p_h, (b_h).to(tl.float32))
    }
    b_q = tl.load(p_q)
    b_q = (b_q * $((scale : ℝ))).to(tl.float32)
    b_k = tl.load(p_k)
    b_v = tl.load(p_v)

    b_s = tl.dot(b_q, b_k, allow_tf32=false)
    b_o = tl.dot((b_s).to(tl.float32), b_v, allow_tf32=false)
    if IFCOND {
      if i == $(0) {
        b_h = tl.dot(b_k, b_v, allow_tf32=false)
      } else {
        b_o += tl.dot(b_q, (b_h).to(tl.float32), allow_tf32=false)
        b_h += tl.dot(b_k, b_v, allow_tf32=false)
      }
    } else {
      b_o += tl.dot(b_q, (b_h).to(tl.float32), allow_tf32=false)
      b_h += tl.dot(b_k, b_v, allow_tf32=false)
    }

    tl.store(p_o, (b_o).to(tl.float32))
  }
}
```
</details>
