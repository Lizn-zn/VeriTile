# Spec sheet — `bench/tritonbench_g/rotary_emb/RotaryEmb.lean`

**Python source:** `bench/tritonbench_g/rotary_emb/rotary_emb.py`

## Public theorem: `rotary_emb_python_case1_output_summary`

<details><summary>docstring</summary>

```
/-- Public Python case-1 summary for `rotary_emb_fwd`.

The full `_rotary_kernel` surface is instantiated with contiguous
`(total_len, 8, 64)` Q/K tensors, contiguous `(total_len, 64)` cos/sin tensors,
`BLOCK_HEAD = 4`, `BLOCK_SEQ = 16`, and `BLOCK_DMODEL = 64`.  The output
conjunct covers the four Python-observable stores: Q even/odd and K even/odd. -/
```
</details>

**Statement:**
```lean
theorem rotary_emb_python_case1_output_summary
    (Q K Cos Sin : RegionName) (s : BlockState) :
    (∃ alg, (rotary_kernel_surface Q K Cos Sin
      512 64 1 512 64 1 64 1 64 1 32 8 8 4 16 64).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := rotary_emb_q0_block Q Cos Sin 512 64 1 64 1 64 1 32 8 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin 32 => active s 32 8)
        (fun i => (Q, qOffset s 512 64 1 (dimEven i))))
      (expected := fun i => rotaryQ0Spec s Q Cos Sin 512 64 1 64 1 64 1 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := rotary_emb_q1_block Q Cos Sin 512 64 1 64 1 64 1 32 8 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin 32 => active s 32 8)
        (fun i => (Q, qOffset s 512 64 1 (dimOdd i))))
      (expected := fun i => rotaryQ1Spec s Q Cos Sin 512 64 1 64 1 64 1 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := rotary_emb_k0_block K Cos Sin 512 64 1 64 1 64 1 32 8 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin 32 => activeK s 32 8)
        (fun i => (K, kOffset s 512 64 1 (dimEven i))))
      (expected := fun i => rotaryK0Spec s K Cos Sin 512 64 1 64 1 64 1 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := rotary_emb_k1_block K Cos Sin 512 64 1 64 1 64 1 32 8 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin 32 => activeK s 32 8)
        (fun i => (K, kOffset s 512 64 1 (dimOdd i))))
      (expected := fun i => rotaryK1Spec s K Cos Sin 512 64 1 64 1 64 1 i))
```

**Assumptions / layout contracts:**
- `fun _ : Fin 32 => active s 32 8`
- `fun _ : Fin 32 => active s 32 8`
- `fun _ : Fin 32 => activeK s 32 8`
- `fun _ : Fin 32 => activeK s 32 8`

**Closed-form spec defs (transitive):** `rotary_kernel_surface`, `rotary_emb_q0_block`, `active`, `qOffset`, `dimEven`, `rotaryQ0Spec`, `rotary_emb_q1_block`, `dimOdd`, `rotaryQ1Spec`, `rotary_emb_k0_block`, `activeK`, `kOffset`, `rotaryK0Spec`, `rotary_emb_k1_block`, `rotaryK1Spec`, `seqIndex`, `headIndex`, `cosOffset`, `sinOffset`

<details><summary><code>rotary_kernel_surface</code></summary>

```
/-- Faithful transcription of `rotary_emb.py`'s `_rotary_kernel`.

This keeps the block-shaped sequence/head/dimension tiles and writes both Q and
K rotary pairs over the full `[BLOCK_SEQ, BLOCK_HEAD, BLOCK_DMODEL / 2]`
surface. -/
```
```lean
def rotary_kernel_surface
    (Q K Cos Sin : RegionName)
    (stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd
      stride_cosbs stride_cosd stride_sinbs stride_sind max_total_len
      HEAD_Q HEAD_K BLOCK_HEAD BLOCK_SEQ BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  cur_head_index = tl.program_id(0)
  cur_seq_index = tl.program_id(1)

  cur_head_range = cur_head_index * $(BLOCK_HEAD) + tl.arange(0, $(BLOCK_HEAD))
  cur_seq_range = cur_seq_index * $(BLOCK_SEQ) + tl.arange(0, $(BLOCK_SEQ))

  dim_range0 = tl.arange(0, $(BLOCK_DMODEL) // $(2)) * $(2)
  dim_range1 = tl.arange(0, $(BLOCK_DMODEL) // $(2)) * $(2) + $(1)

  off_q0 = cur_seq_range[:, None, None] * $(stride_qbs) +
    cur_head_range[None, :, None] * $(stride_qh) +
    dim_range0[None, None, :] * $(stride_qd)
  off_q1 = cur_seq_range[:, None, None] * $(stride_qbs) +
    cur_head_range[None, :, None] * $(stride_qh) +
    dim_range1[None, None, :] * $(stride_qd)

  off_dimcos_sin0 = cur_seq_range[:, None, None] * $(stride_cosbs) +
    dim_range0[None, None, :] * $(stride_cosd)
  off_dimcos_sin1 = cur_seq_range[:, None, None] * $(stride_cosbs) +
    dim_range1[None, None, :] * $(stride_cosd)

  q0 = tl.load(Q + off_q0,
    mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
      (cur_head_range[None, :, None] < $(HEAD_Q)),
    other=0.0)
  q1 = tl.load(Q + off_q1,
    mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
      (cur_head_range[None, :, None] < $(HEAD_Q)),
    other=0.0)

  cos0 = tl.load(Cos + off_dimcos_sin0,
    mask=cur_seq_range[:, None, None] < $(max_total_len), other=0.0)
  sin0 = tl.load(Sin + off_dimcos_sin0,
    mask=cur_seq_range[:, None, None] < $(max_total_len), other=0.0)

  cos1 = tl.load(Cos + off_dimcos_sin1,
    mask=cur_seq_range[:, None, None] < $(max_total_len), other=0.0)
  sin1 = tl.load(Sin + off_dimcos_sin1,
    mask=cur_seq_range[:, None, None] < $(max_total_len), other=0.0)

  out0 = q0 * cos0 - q1 * sin0
  out1 = q0 * sin1 + q1 * cos1

  tl.store(Q + off_q0, out0,
    mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
      (cur_head_range[None, :, None] < $(HEAD_Q)))
  tl.store(Q + off_q1, out1,
    mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
      (cur_head_range[None, :, None] < $(HEAD_Q)))

  off_k0 = cur_seq_range[:, None, None] * $(stride_kbs) +
    cur_head_range[None, :, None] * $(stride_kh) +
    dim_range0[None, None, :] * $(stride_kd)
  off_k1 = cur_seq_range[:, None, None] * $(stride_kbs) +
    cur_head_range[None, :, None] * $(stride_kh) +
    dim_range1[None, None, :] * $(stride_kd)

  off_dimcos_sin0 = cur_seq_range[:, None, None] * $(stride_cosbs) +
    dim_range0[None, None, :] * $(stride_cosd)
  off_dimcos_sin1 = cur_seq_range[:, None, None] * $(stride_cosbs) +
    dim_range1[None, None, :] * $(stride_cosd)

  k0 = tl.load(K + off_k0,
    mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
      (cur_head_range[None, :, None] < $(HEAD_K)),
    other=0.0)
  k1 = tl.load(K + off_k1,
    mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
      (cur_head_range[None, :, None] < $(HEAD_K)),
    other=0.0)

  cos0 = tl.load(Cos + off_dimcos_sin0,
    mask=cur_seq_range[:, None, None] < $(max_total_len), other=0.0)
  sin0 = tl.load(Sin + off_dimcos_sin0,
    mask=cur_seq_range[:, None, None] < $(max_total_len), other=0.0)

  cos1 = tl.load(Cos + off_dimcos_sin1,
    mask=cur_seq_range[:, None, None] < $(max_total_len), other=0.0)
  sin1 = tl.load(Sin + off_dimcos_sin1,
    mask=cur_seq_range[:, None, None] < $(max_total_len), other=0.0)

  out_k0 = k0 * cos0 - k1 * sin0
  out_k1 = k0 * sin1 + k1 * cos1

  tl.store(K + off_k0, out_k0,
    mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
      (cur_head_range[None, :, None] < $(HEAD_K)))
  tl.store(K + off_k1, out_k1,
    mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
      (cur_head_range[None, :, None] < $(HEAD_K)))
}
```
</details>

<details><summary><code>rotary_emb_q0_block</code></summary>

```
/-- Proof-oriented Q-even-dimension slice of `rotary_emb.py`'s `_rotary_kernel`.

This models the first Q store for one sequence/head program tile:
`out0 = q0 * cos0 - q1 * sin0`, where even dimensions are paired with the
following odd dimension. -/
```
```lean
def rotary_emb_q0_block
    (Q Cos Sin : RegionName)
    (stride_qbs stride_qh stride_qd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_Q BLOCK_HALF : Nat) :
    ComputeKernel := triton {
  cur_head_index = tl.program_id(0)
  cur_seq_index = tl.program_id(1)
  dim = tl.arange(0, $(BLOCK_HALF))
  dim0 = dim * $(2)
  dim1 = dim * $(2) + $(1)
  q0 = tl.load(Q + cur_seq_index * $(stride_qbs) +
      cur_head_index * $(stride_qh) + dim0 * $(stride_qd),
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_Q)),
    other=0.0)
  q1 = tl.load(Q + cur_seq_index * $(stride_qbs) +
      cur_head_index * $(stride_qh) + dim1 * $(stride_qd),
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_Q)),
    other=0.0)
  cos0 = tl.load(Cos + cur_seq_index * $(stride_cosbs) +
      dim0 * $(stride_cosd),
    mask=cur_seq_index < $(max_total_len), other=0.0)
  sin0 = tl.load(Sin + cur_seq_index * $(stride_sinbs) +
      dim0 * $(stride_sind),
    mask=cur_seq_index < $(max_total_len), other=0.0)
  out0 = q0 * cos0 - q1 * sin0
  tl.store(Q + cur_seq_index * $(stride_qbs) +
      cur_head_index * $(stride_qh) + dim0 * $(stride_qd),
    out0,
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_Q)))
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active (s : BlockState) (max_total_len HEAD_Q : Nat) : Prop :=
  seqIndex s < max_total_len ∧ headIndex s < HEAD_Q
```
</details>

<details><summary><code>qOffset</code></summary>

```lean
def qOffset
    (s : BlockState) (stride_qbs stride_qh stride_qd : Nat) (dim : Nat) : Nat :=
  seqIndex s * stride_qbs + headIndex s * stride_qh + dim * stride_qd
```
</details>

<details><summary><code>dimEven</code></summary>

```lean
def dimEven (i : Fin BLOCK_HALF) : Nat :=
  i.val * 2
```
</details>

<details><summary><code>rotaryQ0Spec</code></summary>

```lean
noncomputable def rotaryQ0Spec
    (s : BlockState) (Q Cos Sin : RegionName)
    (stride_qbs stride_qh stride_qd stride_cosbs stride_cosd stride_sinbs
      stride_sind : Nat) (i : Fin BLOCK_HALF) : ℝ :=
  s.readMem Q (qOffset s stride_qbs stride_qh stride_qd (dimEven i)) *
    s.readMem Cos (cosOffset s stride_cosbs stride_cosd (dimEven i)) -
  s.readMem Q (qOffset s stride_qbs stride_qh stride_qd (dimOdd i)) *
    s.readMem Sin (sinOffset s stride_sinbs stride_sind (dimEven i))
```
</details>

<details><summary><code>rotary_emb_q1_block</code></summary>

```
/-- Proof-oriented Q-odd-dimension slice of `rotary_emb.py`'s `_rotary_kernel`.

This models the second Q store for one sequence/head program tile:
`out1 = q0 * sin0 + q1 * cos0`, written at the odd-dimension offset. -/
```
```lean
def rotary_emb_q1_block
    (Q Cos Sin : RegionName)
    (stride_qbs stride_qh stride_qd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_Q BLOCK_HALF : Nat) :
    ComputeKernel := triton {
  cur_head_index = tl.program_id(0)
  cur_seq_index = tl.program_id(1)
  dim = tl.arange(0, $(BLOCK_HALF))
  dim0 = dim * $(2)
  dim1 = dim * $(2) + $(1)
  q0 = tl.load(Q + cur_seq_index * $(stride_qbs) +
      cur_head_index * $(stride_qh) + dim0 * $(stride_qd),
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_Q)),
    other=0.0)
  q1 = tl.load(Q + cur_seq_index * $(stride_qbs) +
      cur_head_index * $(stride_qh) + dim1 * $(stride_qd),
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_Q)),
    other=0.0)
  cos0 = tl.load(Cos + cur_seq_index * $(stride_cosbs) +
      dim0 * $(stride_cosd),
    mask=cur_seq_index < $(max_total_len), other=0.0)
  sin0 = tl.load(Sin + cur_seq_index * $(stride_sinbs) +
      dim0 * $(stride_sind),
    mask=cur_seq_index < $(max_total_len), other=0.0)
  out1 = q0 * sin0 + q1 * cos0
  tl.store(Q + cur_seq_index * $(stride_qbs) +
      cur_head_index * $(stride_qh) + dim1 * $(stride_qd),
    out1,
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_Q)))
}
```
</details>

<details><summary><code>dimOdd</code></summary>

```lean
def dimOdd (i : Fin BLOCK_HALF) : Nat :=
  i.val * 2 + 1
```
</details>

<details><summary><code>rotaryQ1Spec</code></summary>

```lean
noncomputable def rotaryQ1Spec
    (s : BlockState) (Q Cos Sin : RegionName)
    (stride_qbs stride_qh stride_qd stride_cosbs stride_cosd stride_sinbs
      stride_sind : Nat) (i : Fin BLOCK_HALF) : ℝ :=
  s.readMem Q (qOffset s stride_qbs stride_qh stride_qd (dimEven i)) *
    s.readMem Sin (sinOffset s stride_sinbs stride_sind (dimEven i)) +
  s.readMem Q (qOffset s stride_qbs stride_qh stride_qd (dimOdd i)) *
    s.readMem Cos (cosOffset s stride_cosbs stride_cosd (dimEven i))
```
</details>

<details><summary><code>rotary_emb_k0_block</code></summary>

```
/-- Proof-oriented K-even-dimension slice of `rotary_emb.py`'s `_rotary_kernel`.

Mirrors the Q0 slice for the K buffer with `HEAD_K` head bound. -/
```
```lean
def rotary_emb_k0_block
    (K Cos Sin : RegionName)
    (stride_kbs stride_kh stride_kd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_K BLOCK_HALF : Nat) :
    ComputeKernel := triton {
  cur_head_index = tl.program_id(0)
  cur_seq_index = tl.program_id(1)
  dim = tl.arange(0, $(BLOCK_HALF))
  dim0 = dim * $(2)
  dim1 = dim * $(2) + $(1)
  k0 = tl.load(K + cur_seq_index * $(stride_kbs) +
      cur_head_index * $(stride_kh) + dim0 * $(stride_kd),
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_K)),
    other=0.0)
  k1 = tl.load(K + cur_seq_index * $(stride_kbs) +
      cur_head_index * $(stride_kh) + dim1 * $(stride_kd),
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_K)),
    other=0.0)
  cos0 = tl.load(Cos + cur_seq_index * $(stride_cosbs) +
      dim0 * $(stride_cosd),
    mask=cur_seq_index < $(max_total_len), other=0.0)
  sin0 = tl.load(Sin + cur_seq_index * $(stride_sinbs) +
      dim0 * $(stride_sind),
    mask=cur_seq_index < $(max_total_len), other=0.0)
  outK0 = k0 * cos0 - k1 * sin0
  tl.store(K + cur_seq_index * $(stride_kbs) +
      cur_head_index * $(stride_kh) + dim0 * $(stride_kd),
    outK0,
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_K)))
}
```
</details>

<details><summary><code>activeK</code></summary>

```lean
def activeK (s : BlockState) (max_total_len HEAD_K : Nat) : Prop :=
  seqIndex s < max_total_len ∧ headIndex s < HEAD_K
```
</details>

<details><summary><code>kOffset</code></summary>

```lean
def kOffset
    (s : BlockState) (stride_kbs stride_kh stride_kd : Nat) (dim : Nat) : Nat :=
  seqIndex s * stride_kbs + headIndex s * stride_kh + dim * stride_kd
```
</details>

<details><summary><code>rotaryK0Spec</code></summary>

```lean
noncomputable def rotaryK0Spec
    (s : BlockState) (K Cos Sin : RegionName)
    (stride_kbs stride_kh stride_kd stride_cosbs stride_cosd stride_sinbs
      stride_sind : Nat) (i : Fin BLOCK_HALF) : ℝ :=
  s.readMem K (kOffset s stride_kbs stride_kh stride_kd (dimEven i)) *
    s.readMem Cos (cosOffset s stride_cosbs stride_cosd (dimEven i)) -
  s.readMem K (kOffset s stride_kbs stride_kh stride_kd (dimOdd i)) *
    s.readMem Sin (sinOffset s stride_sinbs stride_sind (dimEven i))
```
</details>

<details><summary><code>rotary_emb_k1_block</code></summary>

```
/-- Proof-oriented K-odd-dimension slice of `rotary_emb.py`'s `_rotary_kernel`. -/
```
```lean
def rotary_emb_k1_block
    (K Cos Sin : RegionName)
    (stride_kbs stride_kh stride_kd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_K BLOCK_HALF : Nat) :
    ComputeKernel := triton {
  cur_head_index = tl.program_id(0)
  cur_seq_index = tl.program_id(1)
  dim = tl.arange(0, $(BLOCK_HALF))
  dim0 = dim * $(2)
  dim1 = dim * $(2) + $(1)
  k0 = tl.load(K + cur_seq_index * $(stride_kbs) +
      cur_head_index * $(stride_kh) + dim0 * $(stride_kd),
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_K)),
    other=0.0)
  k1 = tl.load(K + cur_seq_index * $(stride_kbs) +
      cur_head_index * $(stride_kh) + dim1 * $(stride_kd),
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_K)),
    other=0.0)
  cos0 = tl.load(Cos + cur_seq_index * $(stride_cosbs) +
      dim0 * $(stride_cosd),
    mask=cur_seq_index < $(max_total_len), other=0.0)
  sin0 = tl.load(Sin + cur_seq_index * $(stride_sinbs) +
      dim0 * $(stride_sind),
    mask=cur_seq_index < $(max_total_len), other=0.0)
  outK1 = k0 * sin0 + k1 * cos0
  tl.store(K + cur_seq_index * $(stride_kbs) +
      cur_head_index * $(stride_kh) + dim1 * $(stride_kd),
    outK1,
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_K)))
}
```
</details>

<details><summary><code>rotaryK1Spec</code></summary>

```lean
noncomputable def rotaryK1Spec
    (s : BlockState) (K Cos Sin : RegionName)
    (stride_kbs stride_kh stride_kd stride_cosbs stride_cosd stride_sinbs
      stride_sind : Nat) (i : Fin BLOCK_HALF) : ℝ :=
  s.readMem K (kOffset s stride_kbs stride_kh stride_kd (dimEven i)) *
    s.readMem Sin (sinOffset s stride_sinbs stride_sind (dimEven i)) +
  s.readMem K (kOffset s stride_kbs stride_kh stride_kd (dimOdd i)) *
    s.readMem Cos (cosOffset s stride_cosbs stride_cosd (dimEven i))
```
</details>

<details><summary><code>seqIndex</code></summary>

```lean
def seqIndex (s : BlockState) : Nat :=
  s.pids 1
```
</details>

<details><summary><code>headIndex</code></summary>

```lean
def headIndex (s : BlockState) : Nat :=
  s.pids 0
```
</details>

<details><summary><code>cosOffset</code></summary>

```lean
def cosOffset
    (s : BlockState) (stride_cosbs stride_cosd : Nat) (dim : Nat) : Nat :=
  seqIndex s * stride_cosbs + dim * stride_cosd
```
</details>

<details><summary><code>sinOffset</code></summary>

```lean
def sinOffset
    (s : BlockState) (stride_sinbs stride_sind : Nat) (dim : Nat) : Nat :=
  seqIndex s * stride_sinbs + dim * stride_sind
```
</details>

## Public theorem: `rotary_emb_python_case2_output_summary`

<details><summary>docstring</summary>

```
/-- Public Python case-2 summary: same shape as case 1 except
`head_dim = BLOCK_DMODEL = 128`, so contiguous row/head strides are `1024`
and `128`. -/
```
</details>

**Statement:**
```lean
theorem rotary_emb_python_case2_output_summary
    (Q K Cos Sin : RegionName) (s : BlockState) :
    (∃ alg, (rotary_kernel_surface Q K Cos Sin
      1024 128 1 1024 128 1 128 1 128 1 32 8 8 4 16 128).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := rotary_emb_q0_block Q Cos Sin 1024 128 1 128 1 128 1 32 8 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin 64 => active s 32 8)
        (fun i => (Q, qOffset s 1024 128 1 (dimEven i))))
      (expected := fun i => rotaryQ0Spec s Q Cos Sin 1024 128 1 128 1 128 1 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := rotary_emb_q1_block Q Cos Sin 1024 128 1 128 1 128 1 32 8 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin 64 => active s 32 8)
        (fun i => (Q, qOffset s 1024 128 1 (dimOdd i))))
      (expected := fun i => rotaryQ1Spec s Q Cos Sin 1024 128 1 128 1 128 1 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := rotary_emb_k0_block K Cos Sin 1024 128 1 128 1 128 1 32 8 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin 64 => activeK s 32 8)
        (fun i => (K, kOffset s 1024 128 1 (dimEven i))))
      (expected := fun i => rotaryK0Spec s K Cos Sin 1024 128 1 128 1 128 1 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := rotary_emb_k1_block K Cos Sin 1024 128 1 128 1 128 1 32 8 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin 64 => activeK s 32 8)
        (fun i => (K, kOffset s 1024 128 1 (dimOdd i))))
      (expected := fun i => rotaryK1Spec s K Cos Sin 1024 128 1 128 1 128 1 i))
```

**Assumptions / layout contracts:**
- `fun _ : Fin 64 => active s 32 8`
- `fun _ : Fin 64 => active s 32 8`
- `fun _ : Fin 64 => activeK s 32 8`
- `fun _ : Fin 64 => activeK s 32 8`

**Closed-form spec defs (transitive):** `rotary_kernel_surface`, `rotary_emb_q0_block`, `active`, `qOffset`, `dimEven`, `rotaryQ0Spec`, `rotary_emb_q1_block`, `dimOdd`, `rotaryQ1Spec`, `rotary_emb_k0_block`, `activeK`, `kOffset`, `rotaryK0Spec`, `rotary_emb_k1_block`, `rotaryK1Spec`, `seqIndex`, `headIndex`, `cosOffset`, `sinOffset`

<details><summary><code>rotary_kernel_surface</code></summary>

```
/-- Faithful transcription of `rotary_emb.py`'s `_rotary_kernel`.

This keeps the block-shaped sequence/head/dimension tiles and writes both Q and
K rotary pairs over the full `[BLOCK_SEQ, BLOCK_HEAD, BLOCK_DMODEL / 2]`
surface. -/
```
```lean
def rotary_kernel_surface
    (Q K Cos Sin : RegionName)
    (stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd
      stride_cosbs stride_cosd stride_sinbs stride_sind max_total_len
      HEAD_Q HEAD_K BLOCK_HEAD BLOCK_SEQ BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  cur_head_index = tl.program_id(0)
  cur_seq_index = tl.program_id(1)

  cur_head_range = cur_head_index * $(BLOCK_HEAD) + tl.arange(0, $(BLOCK_HEAD))
  cur_seq_range = cur_seq_index * $(BLOCK_SEQ) + tl.arange(0, $(BLOCK_SEQ))

  dim_range0 = tl.arange(0, $(BLOCK_DMODEL) // $(2)) * $(2)
  dim_range1 = tl.arange(0, $(BLOCK_DMODEL) // $(2)) * $(2) + $(1)

  off_q0 = cur_seq_range[:, None, None] * $(stride_qbs) +
    cur_head_range[None, :, None] * $(stride_qh) +
    dim_range0[None, None, :] * $(stride_qd)
  off_q1 = cur_seq_range[:, None, None] * $(stride_qbs) +
    cur_head_range[None, :, None] * $(stride_qh) +
    dim_range1[None, None, :] * $(stride_qd)

  off_dimcos_sin0 = cur_seq_range[:, None, None] * $(stride_cosbs) +
    dim_range0[None, None, :] * $(stride_cosd)
  off_dimcos_sin1 = cur_seq_range[:, None, None] * $(stride_cosbs) +
    dim_range1[None, None, :] * $(stride_cosd)

  q0 = tl.load(Q + off_q0,
    mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
      (cur_head_range[None, :, None] < $(HEAD_Q)),
    other=0.0)
  q1 = tl.load(Q + off_q1,
    mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
      (cur_head_range[None, :, None] < $(HEAD_Q)),
    other=0.0)

  cos0 = tl.load(Cos + off_dimcos_sin0,
    mask=cur_seq_range[:, None, None] < $(max_total_len), other=0.0)
  sin0 = tl.load(Sin + off_dimcos_sin0,
    mask=cur_seq_range[:, None, None] < $(max_total_len), other=0.0)

  cos1 = tl.load(Cos + off_dimcos_sin1,
    mask=cur_seq_range[:, None, None] < $(max_total_len), other=0.0)
  sin1 = tl.load(Sin + off_dimcos_sin1,
    mask=cur_seq_range[:, None, None] < $(max_total_len), other=0.0)

  out0 = q0 * cos0 - q1 * sin0
  out1 = q0 * sin1 + q1 * cos1

  tl.store(Q + off_q0, out0,
    mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
      (cur_head_range[None, :, None] < $(HEAD_Q)))
  tl.store(Q + off_q1, out1,
    mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
      (cur_head_range[None, :, None] < $(HEAD_Q)))

  off_k0 = cur_seq_range[:, None, None] * $(stride_kbs) +
    cur_head_range[None, :, None] * $(stride_kh) +
    dim_range0[None, None, :] * $(stride_kd)
  off_k1 = cur_seq_range[:, None, None] * $(stride_kbs) +
    cur_head_range[None, :, None] * $(stride_kh) +
    dim_range1[None, None, :] * $(stride_kd)

  off_dimcos_sin0 = cur_seq_range[:, None, None] * $(stride_cosbs) +
    dim_range0[None, None, :] * $(stride_cosd)
  off_dimcos_sin1 = cur_seq_range[:, None, None] * $(stride_cosbs) +
    dim_range1[None, None, :] * $(stride_cosd)

  k0 = tl.load(K + off_k0,
    mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
      (cur_head_range[None, :, None] < $(HEAD_K)),
    other=0.0)
  k1 = tl.load(K + off_k1,
    mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
      (cur_head_range[None, :, None] < $(HEAD_K)),
    other=0.0)

  cos0 = tl.load(Cos + off_dimcos_sin0,
    mask=cur_seq_range[:, None, None] < $(max_total_len), other=0.0)
  sin0 = tl.load(Sin + off_dimcos_sin0,
    mask=cur_seq_range[:, None, None] < $(max_total_len), other=0.0)

  cos1 = tl.load(Cos + off_dimcos_sin1,
    mask=cur_seq_range[:, None, None] < $(max_total_len), other=0.0)
  sin1 = tl.load(Sin + off_dimcos_sin1,
    mask=cur_seq_range[:, None, None] < $(max_total_len), other=0.0)

  out_k0 = k0 * cos0 - k1 * sin0
  out_k1 = k0 * sin1 + k1 * cos1

  tl.store(K + off_k0, out_k0,
    mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
      (cur_head_range[None, :, None] < $(HEAD_K)))
  tl.store(K + off_k1, out_k1,
    mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
      (cur_head_range[None, :, None] < $(HEAD_K)))
}
```
</details>

<details><summary><code>rotary_emb_q0_block</code></summary>

```
/-- Proof-oriented Q-even-dimension slice of `rotary_emb.py`'s `_rotary_kernel`.

This models the first Q store for one sequence/head program tile:
`out0 = q0 * cos0 - q1 * sin0`, where even dimensions are paired with the
following odd dimension. -/
```
```lean
def rotary_emb_q0_block
    (Q Cos Sin : RegionName)
    (stride_qbs stride_qh stride_qd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_Q BLOCK_HALF : Nat) :
    ComputeKernel := triton {
  cur_head_index = tl.program_id(0)
  cur_seq_index = tl.program_id(1)
  dim = tl.arange(0, $(BLOCK_HALF))
  dim0 = dim * $(2)
  dim1 = dim * $(2) + $(1)
  q0 = tl.load(Q + cur_seq_index * $(stride_qbs) +
      cur_head_index * $(stride_qh) + dim0 * $(stride_qd),
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_Q)),
    other=0.0)
  q1 = tl.load(Q + cur_seq_index * $(stride_qbs) +
      cur_head_index * $(stride_qh) + dim1 * $(stride_qd),
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_Q)),
    other=0.0)
  cos0 = tl.load(Cos + cur_seq_index * $(stride_cosbs) +
      dim0 * $(stride_cosd),
    mask=cur_seq_index < $(max_total_len), other=0.0)
  sin0 = tl.load(Sin + cur_seq_index * $(stride_sinbs) +
      dim0 * $(stride_sind),
    mask=cur_seq_index < $(max_total_len), other=0.0)
  out0 = q0 * cos0 - q1 * sin0
  tl.store(Q + cur_seq_index * $(stride_qbs) +
      cur_head_index * $(stride_qh) + dim0 * $(stride_qd),
    out0,
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_Q)))
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active (s : BlockState) (max_total_len HEAD_Q : Nat) : Prop :=
  seqIndex s < max_total_len ∧ headIndex s < HEAD_Q
```
</details>

<details><summary><code>qOffset</code></summary>

```lean
def qOffset
    (s : BlockState) (stride_qbs stride_qh stride_qd : Nat) (dim : Nat) : Nat :=
  seqIndex s * stride_qbs + headIndex s * stride_qh + dim * stride_qd
```
</details>

<details><summary><code>dimEven</code></summary>

```lean
def dimEven (i : Fin BLOCK_HALF) : Nat :=
  i.val * 2
```
</details>

<details><summary><code>rotaryQ0Spec</code></summary>

```lean
noncomputable def rotaryQ0Spec
    (s : BlockState) (Q Cos Sin : RegionName)
    (stride_qbs stride_qh stride_qd stride_cosbs stride_cosd stride_sinbs
      stride_sind : Nat) (i : Fin BLOCK_HALF) : ℝ :=
  s.readMem Q (qOffset s stride_qbs stride_qh stride_qd (dimEven i)) *
    s.readMem Cos (cosOffset s stride_cosbs stride_cosd (dimEven i)) -
  s.readMem Q (qOffset s stride_qbs stride_qh stride_qd (dimOdd i)) *
    s.readMem Sin (sinOffset s stride_sinbs stride_sind (dimEven i))
```
</details>

<details><summary><code>rotary_emb_q1_block</code></summary>

```
/-- Proof-oriented Q-odd-dimension slice of `rotary_emb.py`'s `_rotary_kernel`.

This models the second Q store for one sequence/head program tile:
`out1 = q0 * sin0 + q1 * cos0`, written at the odd-dimension offset. -/
```
```lean
def rotary_emb_q1_block
    (Q Cos Sin : RegionName)
    (stride_qbs stride_qh stride_qd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_Q BLOCK_HALF : Nat) :
    ComputeKernel := triton {
  cur_head_index = tl.program_id(0)
  cur_seq_index = tl.program_id(1)
  dim = tl.arange(0, $(BLOCK_HALF))
  dim0 = dim * $(2)
  dim1 = dim * $(2) + $(1)
  q0 = tl.load(Q + cur_seq_index * $(stride_qbs) +
      cur_head_index * $(stride_qh) + dim0 * $(stride_qd),
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_Q)),
    other=0.0)
  q1 = tl.load(Q + cur_seq_index * $(stride_qbs) +
      cur_head_index * $(stride_qh) + dim1 * $(stride_qd),
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_Q)),
    other=0.0)
  cos0 = tl.load(Cos + cur_seq_index * $(stride_cosbs) +
      dim0 * $(stride_cosd),
    mask=cur_seq_index < $(max_total_len), other=0.0)
  sin0 = tl.load(Sin + cur_seq_index * $(stride_sinbs) +
      dim0 * $(stride_sind),
    mask=cur_seq_index < $(max_total_len), other=0.0)
  out1 = q0 * sin0 + q1 * cos0
  tl.store(Q + cur_seq_index * $(stride_qbs) +
      cur_head_index * $(stride_qh) + dim1 * $(stride_qd),
    out1,
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_Q)))
}
```
</details>

<details><summary><code>dimOdd</code></summary>

```lean
def dimOdd (i : Fin BLOCK_HALF) : Nat :=
  i.val * 2 + 1
```
</details>

<details><summary><code>rotaryQ1Spec</code></summary>

```lean
noncomputable def rotaryQ1Spec
    (s : BlockState) (Q Cos Sin : RegionName)
    (stride_qbs stride_qh stride_qd stride_cosbs stride_cosd stride_sinbs
      stride_sind : Nat) (i : Fin BLOCK_HALF) : ℝ :=
  s.readMem Q (qOffset s stride_qbs stride_qh stride_qd (dimEven i)) *
    s.readMem Sin (sinOffset s stride_sinbs stride_sind (dimEven i)) +
  s.readMem Q (qOffset s stride_qbs stride_qh stride_qd (dimOdd i)) *
    s.readMem Cos (cosOffset s stride_cosbs stride_cosd (dimEven i))
```
</details>

<details><summary><code>rotary_emb_k0_block</code></summary>

```
/-- Proof-oriented K-even-dimension slice of `rotary_emb.py`'s `_rotary_kernel`.

Mirrors the Q0 slice for the K buffer with `HEAD_K` head bound. -/
```
```lean
def rotary_emb_k0_block
    (K Cos Sin : RegionName)
    (stride_kbs stride_kh stride_kd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_K BLOCK_HALF : Nat) :
    ComputeKernel := triton {
  cur_head_index = tl.program_id(0)
  cur_seq_index = tl.program_id(1)
  dim = tl.arange(0, $(BLOCK_HALF))
  dim0 = dim * $(2)
  dim1 = dim * $(2) + $(1)
  k0 = tl.load(K + cur_seq_index * $(stride_kbs) +
      cur_head_index * $(stride_kh) + dim0 * $(stride_kd),
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_K)),
    other=0.0)
  k1 = tl.load(K + cur_seq_index * $(stride_kbs) +
      cur_head_index * $(stride_kh) + dim1 * $(stride_kd),
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_K)),
    other=0.0)
  cos0 = tl.load(Cos + cur_seq_index * $(stride_cosbs) +
      dim0 * $(stride_cosd),
    mask=cur_seq_index < $(max_total_len), other=0.0)
  sin0 = tl.load(Sin + cur_seq_index * $(stride_sinbs) +
      dim0 * $(stride_sind),
    mask=cur_seq_index < $(max_total_len), other=0.0)
  outK0 = k0 * cos0 - k1 * sin0
  tl.store(K + cur_seq_index * $(stride_kbs) +
      cur_head_index * $(stride_kh) + dim0 * $(stride_kd),
    outK0,
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_K)))
}
```
</details>

<details><summary><code>activeK</code></summary>

```lean
def activeK (s : BlockState) (max_total_len HEAD_K : Nat) : Prop :=
  seqIndex s < max_total_len ∧ headIndex s < HEAD_K
```
</details>

<details><summary><code>kOffset</code></summary>

```lean
def kOffset
    (s : BlockState) (stride_kbs stride_kh stride_kd : Nat) (dim : Nat) : Nat :=
  seqIndex s * stride_kbs + headIndex s * stride_kh + dim * stride_kd
```
</details>

<details><summary><code>rotaryK0Spec</code></summary>

```lean
noncomputable def rotaryK0Spec
    (s : BlockState) (K Cos Sin : RegionName)
    (stride_kbs stride_kh stride_kd stride_cosbs stride_cosd stride_sinbs
      stride_sind : Nat) (i : Fin BLOCK_HALF) : ℝ :=
  s.readMem K (kOffset s stride_kbs stride_kh stride_kd (dimEven i)) *
    s.readMem Cos (cosOffset s stride_cosbs stride_cosd (dimEven i)) -
  s.readMem K (kOffset s stride_kbs stride_kh stride_kd (dimOdd i)) *
    s.readMem Sin (sinOffset s stride_sinbs stride_sind (dimEven i))
```
</details>

<details><summary><code>rotary_emb_k1_block</code></summary>

```
/-- Proof-oriented K-odd-dimension slice of `rotary_emb.py`'s `_rotary_kernel`. -/
```
```lean
def rotary_emb_k1_block
    (K Cos Sin : RegionName)
    (stride_kbs stride_kh stride_kd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_K BLOCK_HALF : Nat) :
    ComputeKernel := triton {
  cur_head_index = tl.program_id(0)
  cur_seq_index = tl.program_id(1)
  dim = tl.arange(0, $(BLOCK_HALF))
  dim0 = dim * $(2)
  dim1 = dim * $(2) + $(1)
  k0 = tl.load(K + cur_seq_index * $(stride_kbs) +
      cur_head_index * $(stride_kh) + dim0 * $(stride_kd),
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_K)),
    other=0.0)
  k1 = tl.load(K + cur_seq_index * $(stride_kbs) +
      cur_head_index * $(stride_kh) + dim1 * $(stride_kd),
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_K)),
    other=0.0)
  cos0 = tl.load(Cos + cur_seq_index * $(stride_cosbs) +
      dim0 * $(stride_cosd),
    mask=cur_seq_index < $(max_total_len), other=0.0)
  sin0 = tl.load(Sin + cur_seq_index * $(stride_sinbs) +
      dim0 * $(stride_sind),
    mask=cur_seq_index < $(max_total_len), other=0.0)
  outK1 = k0 * sin0 + k1 * cos0
  tl.store(K + cur_seq_index * $(stride_kbs) +
      cur_head_index * $(stride_kh) + dim1 * $(stride_kd),
    outK1,
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_K)))
}
```
</details>

<details><summary><code>rotaryK1Spec</code></summary>

```lean
noncomputable def rotaryK1Spec
    (s : BlockState) (K Cos Sin : RegionName)
    (stride_kbs stride_kh stride_kd stride_cosbs stride_cosd stride_sinbs
      stride_sind : Nat) (i : Fin BLOCK_HALF) : ℝ :=
  s.readMem K (kOffset s stride_kbs stride_kh stride_kd (dimEven i)) *
    s.readMem Sin (sinOffset s stride_sinbs stride_sind (dimEven i)) +
  s.readMem K (kOffset s stride_kbs stride_kh stride_kd (dimOdd i)) *
    s.readMem Cos (cosOffset s stride_cosbs stride_cosd (dimEven i))
```
</details>

<details><summary><code>seqIndex</code></summary>

```lean
def seqIndex (s : BlockState) : Nat :=
  s.pids 1
```
</details>

<details><summary><code>headIndex</code></summary>

```lean
def headIndex (s : BlockState) : Nat :=
  s.pids 0
```
</details>

<details><summary><code>cosOffset</code></summary>

```lean
def cosOffset
    (s : BlockState) (stride_cosbs stride_cosd : Nat) (dim : Nat) : Nat :=
  seqIndex s * stride_cosbs + dim * stride_cosd
```
</details>

<details><summary><code>sinOffset</code></summary>

```lean
def sinOffset
    (s : BlockState) (stride_sinbs stride_sind : Nat) (dim : Nat) : Nat :=
  seqIndex s * stride_sinbs + dim * stride_sind
```
</details>

## Public theorem: `rotary_emb_python_case3_output_summary`

<details><summary>docstring</summary>

```
/-- Public Python case-3 summary: the source tensors are contiguous
`(32, 8, 64)`, but `partial_rotary_factor = 0.5` makes the launched
`BLOCK_DMODEL = 32`. -/
```
</details>

**Statement:**
```lean
theorem rotary_emb_python_case3_output_summary
    (Q K Cos Sin : RegionName) (s : BlockState) :
    (∃ alg, (rotary_kernel_surface Q K Cos Sin
      512 64 1 512 64 1 64 1 64 1 32 8 8 4 16 32).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := rotary_emb_q0_block Q Cos Sin 512 64 1 64 1 64 1 32 8 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin 16 => active s 32 8)
        (fun i => (Q, qOffset s 512 64 1 (dimEven i))))
      (expected := fun i => rotaryQ0Spec s Q Cos Sin 512 64 1 64 1 64 1 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := rotary_emb_q1_block Q Cos Sin 512 64 1 64 1 64 1 32 8 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin 16 => active s 32 8)
        (fun i => (Q, qOffset s 512 64 1 (dimOdd i))))
      (expected := fun i => rotaryQ1Spec s Q Cos Sin 512 64 1 64 1 64 1 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := rotary_emb_k0_block K Cos Sin 512 64 1 64 1 64 1 32 8 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin 16 => activeK s 32 8)
        (fun i => (K, kOffset s 512 64 1 (dimEven i))))
      (expected := fun i => rotaryK0Spec s K Cos Sin 512 64 1 64 1 64 1 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := rotary_emb_k1_block K Cos Sin 512 64 1 64 1 64 1 32 8 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin 16 => activeK s 32 8)
        (fun i => (K, kOffset s 512 64 1 (dimOdd i))))
      (expected := fun i => rotaryK1Spec s K Cos Sin 512 64 1 64 1 64 1 i))
```

**Assumptions / layout contracts:**
- `fun _ : Fin 16 => active s 32 8`
- `fun _ : Fin 16 => active s 32 8`
- `fun _ : Fin 16 => activeK s 32 8`
- `fun _ : Fin 16 => activeK s 32 8`

**Closed-form spec defs (transitive):** `rotary_kernel_surface`, `rotary_emb_q0_block`, `active`, `qOffset`, `dimEven`, `rotaryQ0Spec`, `rotary_emb_q1_block`, `dimOdd`, `rotaryQ1Spec`, `rotary_emb_k0_block`, `activeK`, `kOffset`, `rotaryK0Spec`, `rotary_emb_k1_block`, `rotaryK1Spec`, `seqIndex`, `headIndex`, `cosOffset`, `sinOffset`

<details><summary><code>rotary_kernel_surface</code></summary>

```
/-- Faithful transcription of `rotary_emb.py`'s `_rotary_kernel`.

This keeps the block-shaped sequence/head/dimension tiles and writes both Q and
K rotary pairs over the full `[BLOCK_SEQ, BLOCK_HEAD, BLOCK_DMODEL / 2]`
surface. -/
```
```lean
def rotary_kernel_surface
    (Q K Cos Sin : RegionName)
    (stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd
      stride_cosbs stride_cosd stride_sinbs stride_sind max_total_len
      HEAD_Q HEAD_K BLOCK_HEAD BLOCK_SEQ BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  cur_head_index = tl.program_id(0)
  cur_seq_index = tl.program_id(1)

  cur_head_range = cur_head_index * $(BLOCK_HEAD) + tl.arange(0, $(BLOCK_HEAD))
  cur_seq_range = cur_seq_index * $(BLOCK_SEQ) + tl.arange(0, $(BLOCK_SEQ))

  dim_range0 = tl.arange(0, $(BLOCK_DMODEL) // $(2)) * $(2)
  dim_range1 = tl.arange(0, $(BLOCK_DMODEL) // $(2)) * $(2) + $(1)

  off_q0 = cur_seq_range[:, None, None] * $(stride_qbs) +
    cur_head_range[None, :, None] * $(stride_qh) +
    dim_range0[None, None, :] * $(stride_qd)
  off_q1 = cur_seq_range[:, None, None] * $(stride_qbs) +
    cur_head_range[None, :, None] * $(stride_qh) +
    dim_range1[None, None, :] * $(stride_qd)

  off_dimcos_sin0 = cur_seq_range[:, None, None] * $(stride_cosbs) +
    dim_range0[None, None, :] * $(stride_cosd)
  off_dimcos_sin1 = cur_seq_range[:, None, None] * $(stride_cosbs) +
    dim_range1[None, None, :] * $(stride_cosd)

  q0 = tl.load(Q + off_q0,
    mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
      (cur_head_range[None, :, None] < $(HEAD_Q)),
    other=0.0)
  q1 = tl.load(Q + off_q1,
    mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
      (cur_head_range[None, :, None] < $(HEAD_Q)),
    other=0.0)

  cos0 = tl.load(Cos + off_dimcos_sin0,
    mask=cur_seq_range[:, None, None] < $(max_total_len), other=0.0)
  sin0 = tl.load(Sin + off_dimcos_sin0,
    mask=cur_seq_range[:, None, None] < $(max_total_len), other=0.0)

  cos1 = tl.load(Cos + off_dimcos_sin1,
    mask=cur_seq_range[:, None, None] < $(max_total_len), other=0.0)
  sin1 = tl.load(Sin + off_dimcos_sin1,
    mask=cur_seq_range[:, None, None] < $(max_total_len), other=0.0)

  out0 = q0 * cos0 - q1 * sin0
  out1 = q0 * sin1 + q1 * cos1

  tl.store(Q + off_q0, out0,
    mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
      (cur_head_range[None, :, None] < $(HEAD_Q)))
  tl.store(Q + off_q1, out1,
    mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
      (cur_head_range[None, :, None] < $(HEAD_Q)))

  off_k0 = cur_seq_range[:, None, None] * $(stride_kbs) +
    cur_head_range[None, :, None] * $(stride_kh) +
    dim_range0[None, None, :] * $(stride_kd)
  off_k1 = cur_seq_range[:, None, None] * $(stride_kbs) +
    cur_head_range[None, :, None] * $(stride_kh) +
    dim_range1[None, None, :] * $(stride_kd)

  off_dimcos_sin0 = cur_seq_range[:, None, None] * $(stride_cosbs) +
    dim_range0[None, None, :] * $(stride_cosd)
  off_dimcos_sin1 = cur_seq_range[:, None, None] * $(stride_cosbs) +
    dim_range1[None, None, :] * $(stride_cosd)

  k0 = tl.load(K + off_k0,
    mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
      (cur_head_range[None, :, None] < $(HEAD_K)),
    other=0.0)
  k1 = tl.load(K + off_k1,
    mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
      (cur_head_range[None, :, None] < $(HEAD_K)),
    other=0.0)

  cos0 = tl.load(Cos + off_dimcos_sin0,
    mask=cur_seq_range[:, None, None] < $(max_total_len), other=0.0)
  sin0 = tl.load(Sin + off_dimcos_sin0,
    mask=cur_seq_range[:, None, None] < $(max_total_len), other=0.0)

  cos1 = tl.load(Cos + off_dimcos_sin1,
    mask=cur_seq_range[:, None, None] < $(max_total_len), other=0.0)
  sin1 = tl.load(Sin + off_dimcos_sin1,
    mask=cur_seq_range[:, None, None] < $(max_total_len), other=0.0)

  out_k0 = k0 * cos0 - k1 * sin0
  out_k1 = k0 * sin1 + k1 * cos1

  tl.store(K + off_k0, out_k0,
    mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
      (cur_head_range[None, :, None] < $(HEAD_K)))
  tl.store(K + off_k1, out_k1,
    mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
      (cur_head_range[None, :, None] < $(HEAD_K)))
}
```
</details>

<details><summary><code>rotary_emb_q0_block</code></summary>

```
/-- Proof-oriented Q-even-dimension slice of `rotary_emb.py`'s `_rotary_kernel`.

This models the first Q store for one sequence/head program tile:
`out0 = q0 * cos0 - q1 * sin0`, where even dimensions are paired with the
following odd dimension. -/
```
```lean
def rotary_emb_q0_block
    (Q Cos Sin : RegionName)
    (stride_qbs stride_qh stride_qd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_Q BLOCK_HALF : Nat) :
    ComputeKernel := triton {
  cur_head_index = tl.program_id(0)
  cur_seq_index = tl.program_id(1)
  dim = tl.arange(0, $(BLOCK_HALF))
  dim0 = dim * $(2)
  dim1 = dim * $(2) + $(1)
  q0 = tl.load(Q + cur_seq_index * $(stride_qbs) +
      cur_head_index * $(stride_qh) + dim0 * $(stride_qd),
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_Q)),
    other=0.0)
  q1 = tl.load(Q + cur_seq_index * $(stride_qbs) +
      cur_head_index * $(stride_qh) + dim1 * $(stride_qd),
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_Q)),
    other=0.0)
  cos0 = tl.load(Cos + cur_seq_index * $(stride_cosbs) +
      dim0 * $(stride_cosd),
    mask=cur_seq_index < $(max_total_len), other=0.0)
  sin0 = tl.load(Sin + cur_seq_index * $(stride_sinbs) +
      dim0 * $(stride_sind),
    mask=cur_seq_index < $(max_total_len), other=0.0)
  out0 = q0 * cos0 - q1 * sin0
  tl.store(Q + cur_seq_index * $(stride_qbs) +
      cur_head_index * $(stride_qh) + dim0 * $(stride_qd),
    out0,
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_Q)))
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active (s : BlockState) (max_total_len HEAD_Q : Nat) : Prop :=
  seqIndex s < max_total_len ∧ headIndex s < HEAD_Q
```
</details>

<details><summary><code>qOffset</code></summary>

```lean
def qOffset
    (s : BlockState) (stride_qbs stride_qh stride_qd : Nat) (dim : Nat) : Nat :=
  seqIndex s * stride_qbs + headIndex s * stride_qh + dim * stride_qd
```
</details>

<details><summary><code>dimEven</code></summary>

```lean
def dimEven (i : Fin BLOCK_HALF) : Nat :=
  i.val * 2
```
</details>

<details><summary><code>rotaryQ0Spec</code></summary>

```lean
noncomputable def rotaryQ0Spec
    (s : BlockState) (Q Cos Sin : RegionName)
    (stride_qbs stride_qh stride_qd stride_cosbs stride_cosd stride_sinbs
      stride_sind : Nat) (i : Fin BLOCK_HALF) : ℝ :=
  s.readMem Q (qOffset s stride_qbs stride_qh stride_qd (dimEven i)) *
    s.readMem Cos (cosOffset s stride_cosbs stride_cosd (dimEven i)) -
  s.readMem Q (qOffset s stride_qbs stride_qh stride_qd (dimOdd i)) *
    s.readMem Sin (sinOffset s stride_sinbs stride_sind (dimEven i))
```
</details>

<details><summary><code>rotary_emb_q1_block</code></summary>

```
/-- Proof-oriented Q-odd-dimension slice of `rotary_emb.py`'s `_rotary_kernel`.

This models the second Q store for one sequence/head program tile:
`out1 = q0 * sin0 + q1 * cos0`, written at the odd-dimension offset. -/
```
```lean
def rotary_emb_q1_block
    (Q Cos Sin : RegionName)
    (stride_qbs stride_qh stride_qd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_Q BLOCK_HALF : Nat) :
    ComputeKernel := triton {
  cur_head_index = tl.program_id(0)
  cur_seq_index = tl.program_id(1)
  dim = tl.arange(0, $(BLOCK_HALF))
  dim0 = dim * $(2)
  dim1 = dim * $(2) + $(1)
  q0 = tl.load(Q + cur_seq_index * $(stride_qbs) +
      cur_head_index * $(stride_qh) + dim0 * $(stride_qd),
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_Q)),
    other=0.0)
  q1 = tl.load(Q + cur_seq_index * $(stride_qbs) +
      cur_head_index * $(stride_qh) + dim1 * $(stride_qd),
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_Q)),
    other=0.0)
  cos0 = tl.load(Cos + cur_seq_index * $(stride_cosbs) +
      dim0 * $(stride_cosd),
    mask=cur_seq_index < $(max_total_len), other=0.0)
  sin0 = tl.load(Sin + cur_seq_index * $(stride_sinbs) +
      dim0 * $(stride_sind),
    mask=cur_seq_index < $(max_total_len), other=0.0)
  out1 = q0 * sin0 + q1 * cos0
  tl.store(Q + cur_seq_index * $(stride_qbs) +
      cur_head_index * $(stride_qh) + dim1 * $(stride_qd),
    out1,
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_Q)))
}
```
</details>

<details><summary><code>dimOdd</code></summary>

```lean
def dimOdd (i : Fin BLOCK_HALF) : Nat :=
  i.val * 2 + 1
```
</details>

<details><summary><code>rotaryQ1Spec</code></summary>

```lean
noncomputable def rotaryQ1Spec
    (s : BlockState) (Q Cos Sin : RegionName)
    (stride_qbs stride_qh stride_qd stride_cosbs stride_cosd stride_sinbs
      stride_sind : Nat) (i : Fin BLOCK_HALF) : ℝ :=
  s.readMem Q (qOffset s stride_qbs stride_qh stride_qd (dimEven i)) *
    s.readMem Sin (sinOffset s stride_sinbs stride_sind (dimEven i)) +
  s.readMem Q (qOffset s stride_qbs stride_qh stride_qd (dimOdd i)) *
    s.readMem Cos (cosOffset s stride_cosbs stride_cosd (dimEven i))
```
</details>

<details><summary><code>rotary_emb_k0_block</code></summary>

```
/-- Proof-oriented K-even-dimension slice of `rotary_emb.py`'s `_rotary_kernel`.

Mirrors the Q0 slice for the K buffer with `HEAD_K` head bound. -/
```
```lean
def rotary_emb_k0_block
    (K Cos Sin : RegionName)
    (stride_kbs stride_kh stride_kd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_K BLOCK_HALF : Nat) :
    ComputeKernel := triton {
  cur_head_index = tl.program_id(0)
  cur_seq_index = tl.program_id(1)
  dim = tl.arange(0, $(BLOCK_HALF))
  dim0 = dim * $(2)
  dim1 = dim * $(2) + $(1)
  k0 = tl.load(K + cur_seq_index * $(stride_kbs) +
      cur_head_index * $(stride_kh) + dim0 * $(stride_kd),
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_K)),
    other=0.0)
  k1 = tl.load(K + cur_seq_index * $(stride_kbs) +
      cur_head_index * $(stride_kh) + dim1 * $(stride_kd),
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_K)),
    other=0.0)
  cos0 = tl.load(Cos + cur_seq_index * $(stride_cosbs) +
      dim0 * $(stride_cosd),
    mask=cur_seq_index < $(max_total_len), other=0.0)
  sin0 = tl.load(Sin + cur_seq_index * $(stride_sinbs) +
      dim0 * $(stride_sind),
    mask=cur_seq_index < $(max_total_len), other=0.0)
  outK0 = k0 * cos0 - k1 * sin0
  tl.store(K + cur_seq_index * $(stride_kbs) +
      cur_head_index * $(stride_kh) + dim0 * $(stride_kd),
    outK0,
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_K)))
}
```
</details>

<details><summary><code>activeK</code></summary>

```lean
def activeK (s : BlockState) (max_total_len HEAD_K : Nat) : Prop :=
  seqIndex s < max_total_len ∧ headIndex s < HEAD_K
```
</details>

<details><summary><code>kOffset</code></summary>

```lean
def kOffset
    (s : BlockState) (stride_kbs stride_kh stride_kd : Nat) (dim : Nat) : Nat :=
  seqIndex s * stride_kbs + headIndex s * stride_kh + dim * stride_kd
```
</details>

<details><summary><code>rotaryK0Spec</code></summary>

```lean
noncomputable def rotaryK0Spec
    (s : BlockState) (K Cos Sin : RegionName)
    (stride_kbs stride_kh stride_kd stride_cosbs stride_cosd stride_sinbs
      stride_sind : Nat) (i : Fin BLOCK_HALF) : ℝ :=
  s.readMem K (kOffset s stride_kbs stride_kh stride_kd (dimEven i)) *
    s.readMem Cos (cosOffset s stride_cosbs stride_cosd (dimEven i)) -
  s.readMem K (kOffset s stride_kbs stride_kh stride_kd (dimOdd i)) *
    s.readMem Sin (sinOffset s stride_sinbs stride_sind (dimEven i))
```
</details>

<details><summary><code>rotary_emb_k1_block</code></summary>

```
/-- Proof-oriented K-odd-dimension slice of `rotary_emb.py`'s `_rotary_kernel`. -/
```
```lean
def rotary_emb_k1_block
    (K Cos Sin : RegionName)
    (stride_kbs stride_kh stride_kd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_K BLOCK_HALF : Nat) :
    ComputeKernel := triton {
  cur_head_index = tl.program_id(0)
  cur_seq_index = tl.program_id(1)
  dim = tl.arange(0, $(BLOCK_HALF))
  dim0 = dim * $(2)
  dim1 = dim * $(2) + $(1)
  k0 = tl.load(K + cur_seq_index * $(stride_kbs) +
      cur_head_index * $(stride_kh) + dim0 * $(stride_kd),
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_K)),
    other=0.0)
  k1 = tl.load(K + cur_seq_index * $(stride_kbs) +
      cur_head_index * $(stride_kh) + dim1 * $(stride_kd),
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_K)),
    other=0.0)
  cos0 = tl.load(Cos + cur_seq_index * $(stride_cosbs) +
      dim0 * $(stride_cosd),
    mask=cur_seq_index < $(max_total_len), other=0.0)
  sin0 = tl.load(Sin + cur_seq_index * $(stride_sinbs) +
      dim0 * $(stride_sind),
    mask=cur_seq_index < $(max_total_len), other=0.0)
  outK1 = k0 * sin0 + k1 * cos0
  tl.store(K + cur_seq_index * $(stride_kbs) +
      cur_head_index * $(stride_kh) + dim1 * $(stride_kd),
    outK1,
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_K)))
}
```
</details>

<details><summary><code>rotaryK1Spec</code></summary>

```lean
noncomputable def rotaryK1Spec
    (s : BlockState) (K Cos Sin : RegionName)
    (stride_kbs stride_kh stride_kd stride_cosbs stride_cosd stride_sinbs
      stride_sind : Nat) (i : Fin BLOCK_HALF) : ℝ :=
  s.readMem K (kOffset s stride_kbs stride_kh stride_kd (dimEven i)) *
    s.readMem Sin (sinOffset s stride_sinbs stride_sind (dimEven i)) +
  s.readMem K (kOffset s stride_kbs stride_kh stride_kd (dimOdd i)) *
    s.readMem Cos (cosOffset s stride_cosbs stride_cosd (dimEven i))
```
</details>

<details><summary><code>seqIndex</code></summary>

```lean
def seqIndex (s : BlockState) : Nat :=
  s.pids 1
```
</details>

<details><summary><code>headIndex</code></summary>

```lean
def headIndex (s : BlockState) : Nat :=
  s.pids 0
```
</details>

<details><summary><code>cosOffset</code></summary>

```lean
def cosOffset
    (s : BlockState) (stride_cosbs stride_cosd : Nat) (dim : Nat) : Nat :=
  seqIndex s * stride_cosbs + dim * stride_cosd
```
</details>

<details><summary><code>sinOffset</code></summary>

```lean
def sinOffset
    (s : BlockState) (stride_sinbs stride_sind : Nat) (dim : Nat) : Nat :=
  seqIndex s * stride_sinbs + dim * stride_sind
```
</details>

## Public theorem: `rotary_emb_python_case4_output_summary`

<details><summary>docstring</summary>

```
/-- Public Python case-4 summary: `total_len = 64`, `head_dim = 64`, and
the same contiguous Q/K and cos/sin strides as case 1. -/
```
</details>

**Statement:**
```lean
theorem rotary_emb_python_case4_output_summary
    (Q K Cos Sin : RegionName) (s : BlockState) :
    (∃ alg, (rotary_kernel_surface Q K Cos Sin
      512 64 1 512 64 1 64 1 64 1 64 8 8 4 16 64).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := rotary_emb_q0_block Q Cos Sin 512 64 1 64 1 64 1 64 8 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin 32 => active s 64 8)
        (fun i => (Q, qOffset s 512 64 1 (dimEven i))))
      (expected := fun i => rotaryQ0Spec s Q Cos Sin 512 64 1 64 1 64 1 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := rotary_emb_q1_block Q Cos Sin 512 64 1 64 1 64 1 64 8 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin 32 => active s 64 8)
        (fun i => (Q, qOffset s 512 64 1 (dimOdd i))))
      (expected := fun i => rotaryQ1Spec s Q Cos Sin 512 64 1 64 1 64 1 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := rotary_emb_k0_block K Cos Sin 512 64 1 64 1 64 1 64 8 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin 32 => activeK s 64 8)
        (fun i => (K, kOffset s 512 64 1 (dimEven i))))
      (expected := fun i => rotaryK0Spec s K Cos Sin 512 64 1 64 1 64 1 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := rotary_emb_k1_block K Cos Sin 512 64 1 64 1 64 1 64 8 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin 32 => activeK s 64 8)
        (fun i => (K, kOffset s 512 64 1 (dimOdd i))))
      (expected := fun i => rotaryK1Spec s K Cos Sin 512 64 1 64 1 64 1 i))
```

**Assumptions / layout contracts:**
- `fun _ : Fin 32 => active s 64 8`
- `fun _ : Fin 32 => active s 64 8`
- `fun _ : Fin 32 => activeK s 64 8`
- `fun _ : Fin 32 => activeK s 64 8`

**Closed-form spec defs (transitive):** `rotary_kernel_surface`, `rotary_emb_q0_block`, `active`, `qOffset`, `dimEven`, `rotaryQ0Spec`, `rotary_emb_q1_block`, `dimOdd`, `rotaryQ1Spec`, `rotary_emb_k0_block`, `activeK`, `kOffset`, `rotaryK0Spec`, `rotary_emb_k1_block`, `rotaryK1Spec`, `seqIndex`, `headIndex`, `cosOffset`, `sinOffset`

<details><summary><code>rotary_kernel_surface</code></summary>

```
/-- Faithful transcription of `rotary_emb.py`'s `_rotary_kernel`.

This keeps the block-shaped sequence/head/dimension tiles and writes both Q and
K rotary pairs over the full `[BLOCK_SEQ, BLOCK_HEAD, BLOCK_DMODEL / 2]`
surface. -/
```
```lean
def rotary_kernel_surface
    (Q K Cos Sin : RegionName)
    (stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd
      stride_cosbs stride_cosd stride_sinbs stride_sind max_total_len
      HEAD_Q HEAD_K BLOCK_HEAD BLOCK_SEQ BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  cur_head_index = tl.program_id(0)
  cur_seq_index = tl.program_id(1)

  cur_head_range = cur_head_index * $(BLOCK_HEAD) + tl.arange(0, $(BLOCK_HEAD))
  cur_seq_range = cur_seq_index * $(BLOCK_SEQ) + tl.arange(0, $(BLOCK_SEQ))

  dim_range0 = tl.arange(0, $(BLOCK_DMODEL) // $(2)) * $(2)
  dim_range1 = tl.arange(0, $(BLOCK_DMODEL) // $(2)) * $(2) + $(1)

  off_q0 = cur_seq_range[:, None, None] * $(stride_qbs) +
    cur_head_range[None, :, None] * $(stride_qh) +
    dim_range0[None, None, :] * $(stride_qd)
  off_q1 = cur_seq_range[:, None, None] * $(stride_qbs) +
    cur_head_range[None, :, None] * $(stride_qh) +
    dim_range1[None, None, :] * $(stride_qd)

  off_dimcos_sin0 = cur_seq_range[:, None, None] * $(stride_cosbs) +
    dim_range0[None, None, :] * $(stride_cosd)
  off_dimcos_sin1 = cur_seq_range[:, None, None] * $(stride_cosbs) +
    dim_range1[None, None, :] * $(stride_cosd)

  q0 = tl.load(Q + off_q0,
    mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
      (cur_head_range[None, :, None] < $(HEAD_Q)),
    other=0.0)
  q1 = tl.load(Q + off_q1,
    mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
      (cur_head_range[None, :, None] < $(HEAD_Q)),
    other=0.0)

  cos0 = tl.load(Cos + off_dimcos_sin0,
    mask=cur_seq_range[:, None, None] < $(max_total_len), other=0.0)
  sin0 = tl.load(Sin + off_dimcos_sin0,
    mask=cur_seq_range[:, None, None] < $(max_total_len), other=0.0)

  cos1 = tl.load(Cos + off_dimcos_sin1,
    mask=cur_seq_range[:, None, None] < $(max_total_len), other=0.0)
  sin1 = tl.load(Sin + off_dimcos_sin1,
    mask=cur_seq_range[:, None, None] < $(max_total_len), other=0.0)

  out0 = q0 * cos0 - q1 * sin0
  out1 = q0 * sin1 + q1 * cos1

  tl.store(Q + off_q0, out0,
    mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
      (cur_head_range[None, :, None] < $(HEAD_Q)))
  tl.store(Q + off_q1, out1,
    mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
      (cur_head_range[None, :, None] < $(HEAD_Q)))

  off_k0 = cur_seq_range[:, None, None] * $(stride_kbs) +
    cur_head_range[None, :, None] * $(stride_kh) +
    dim_range0[None, None, :] * $(stride_kd)
  off_k1 = cur_seq_range[:, None, None] * $(stride_kbs) +
    cur_head_range[None, :, None] * $(stride_kh) +
    dim_range1[None, None, :] * $(stride_kd)

  off_dimcos_sin0 = cur_seq_range[:, None, None] * $(stride_cosbs) +
    dim_range0[None, None, :] * $(stride_cosd)
  off_dimcos_sin1 = cur_seq_range[:, None, None] * $(stride_cosbs) +
    dim_range1[None, None, :] * $(stride_cosd)

  k0 = tl.load(K + off_k0,
    mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
      (cur_head_range[None, :, None] < $(HEAD_K)),
    other=0.0)
  k1 = tl.load(K + off_k1,
    mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
      (cur_head_range[None, :, None] < $(HEAD_K)),
    other=0.0)

  cos0 = tl.load(Cos + off_dimcos_sin0,
    mask=cur_seq_range[:, None, None] < $(max_total_len), other=0.0)
  sin0 = tl.load(Sin + off_dimcos_sin0,
    mask=cur_seq_range[:, None, None] < $(max_total_len), other=0.0)

  cos1 = tl.load(Cos + off_dimcos_sin1,
    mask=cur_seq_range[:, None, None] < $(max_total_len), other=0.0)
  sin1 = tl.load(Sin + off_dimcos_sin1,
    mask=cur_seq_range[:, None, None] < $(max_total_len), other=0.0)

  out_k0 = k0 * cos0 - k1 * sin0
  out_k1 = k0 * sin1 + k1 * cos1

  tl.store(K + off_k0, out_k0,
    mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
      (cur_head_range[None, :, None] < $(HEAD_K)))
  tl.store(K + off_k1, out_k1,
    mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
      (cur_head_range[None, :, None] < $(HEAD_K)))
}
```
</details>

<details><summary><code>rotary_emb_q0_block</code></summary>

```
/-- Proof-oriented Q-even-dimension slice of `rotary_emb.py`'s `_rotary_kernel`.

This models the first Q store for one sequence/head program tile:
`out0 = q0 * cos0 - q1 * sin0`, where even dimensions are paired with the
following odd dimension. -/
```
```lean
def rotary_emb_q0_block
    (Q Cos Sin : RegionName)
    (stride_qbs stride_qh stride_qd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_Q BLOCK_HALF : Nat) :
    ComputeKernel := triton {
  cur_head_index = tl.program_id(0)
  cur_seq_index = tl.program_id(1)
  dim = tl.arange(0, $(BLOCK_HALF))
  dim0 = dim * $(2)
  dim1 = dim * $(2) + $(1)
  q0 = tl.load(Q + cur_seq_index * $(stride_qbs) +
      cur_head_index * $(stride_qh) + dim0 * $(stride_qd),
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_Q)),
    other=0.0)
  q1 = tl.load(Q + cur_seq_index * $(stride_qbs) +
      cur_head_index * $(stride_qh) + dim1 * $(stride_qd),
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_Q)),
    other=0.0)
  cos0 = tl.load(Cos + cur_seq_index * $(stride_cosbs) +
      dim0 * $(stride_cosd),
    mask=cur_seq_index < $(max_total_len), other=0.0)
  sin0 = tl.load(Sin + cur_seq_index * $(stride_sinbs) +
      dim0 * $(stride_sind),
    mask=cur_seq_index < $(max_total_len), other=0.0)
  out0 = q0 * cos0 - q1 * sin0
  tl.store(Q + cur_seq_index * $(stride_qbs) +
      cur_head_index * $(stride_qh) + dim0 * $(stride_qd),
    out0,
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_Q)))
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active (s : BlockState) (max_total_len HEAD_Q : Nat) : Prop :=
  seqIndex s < max_total_len ∧ headIndex s < HEAD_Q
```
</details>

<details><summary><code>qOffset</code></summary>

```lean
def qOffset
    (s : BlockState) (stride_qbs stride_qh stride_qd : Nat) (dim : Nat) : Nat :=
  seqIndex s * stride_qbs + headIndex s * stride_qh + dim * stride_qd
```
</details>

<details><summary><code>dimEven</code></summary>

```lean
def dimEven (i : Fin BLOCK_HALF) : Nat :=
  i.val * 2
```
</details>

<details><summary><code>rotaryQ0Spec</code></summary>

```lean
noncomputable def rotaryQ0Spec
    (s : BlockState) (Q Cos Sin : RegionName)
    (stride_qbs stride_qh stride_qd stride_cosbs stride_cosd stride_sinbs
      stride_sind : Nat) (i : Fin BLOCK_HALF) : ℝ :=
  s.readMem Q (qOffset s stride_qbs stride_qh stride_qd (dimEven i)) *
    s.readMem Cos (cosOffset s stride_cosbs stride_cosd (dimEven i)) -
  s.readMem Q (qOffset s stride_qbs stride_qh stride_qd (dimOdd i)) *
    s.readMem Sin (sinOffset s stride_sinbs stride_sind (dimEven i))
```
</details>

<details><summary><code>rotary_emb_q1_block</code></summary>

```
/-- Proof-oriented Q-odd-dimension slice of `rotary_emb.py`'s `_rotary_kernel`.

This models the second Q store for one sequence/head program tile:
`out1 = q0 * sin0 + q1 * cos0`, written at the odd-dimension offset. -/
```
```lean
def rotary_emb_q1_block
    (Q Cos Sin : RegionName)
    (stride_qbs stride_qh stride_qd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_Q BLOCK_HALF : Nat) :
    ComputeKernel := triton {
  cur_head_index = tl.program_id(0)
  cur_seq_index = tl.program_id(1)
  dim = tl.arange(0, $(BLOCK_HALF))
  dim0 = dim * $(2)
  dim1 = dim * $(2) + $(1)
  q0 = tl.load(Q + cur_seq_index * $(stride_qbs) +
      cur_head_index * $(stride_qh) + dim0 * $(stride_qd),
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_Q)),
    other=0.0)
  q1 = tl.load(Q + cur_seq_index * $(stride_qbs) +
      cur_head_index * $(stride_qh) + dim1 * $(stride_qd),
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_Q)),
    other=0.0)
  cos0 = tl.load(Cos + cur_seq_index * $(stride_cosbs) +
      dim0 * $(stride_cosd),
    mask=cur_seq_index < $(max_total_len), other=0.0)
  sin0 = tl.load(Sin + cur_seq_index * $(stride_sinbs) +
      dim0 * $(stride_sind),
    mask=cur_seq_index < $(max_total_len), other=0.0)
  out1 = q0 * sin0 + q1 * cos0
  tl.store(Q + cur_seq_index * $(stride_qbs) +
      cur_head_index * $(stride_qh) + dim1 * $(stride_qd),
    out1,
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_Q)))
}
```
</details>

<details><summary><code>dimOdd</code></summary>

```lean
def dimOdd (i : Fin BLOCK_HALF) : Nat :=
  i.val * 2 + 1
```
</details>

<details><summary><code>rotaryQ1Spec</code></summary>

```lean
noncomputable def rotaryQ1Spec
    (s : BlockState) (Q Cos Sin : RegionName)
    (stride_qbs stride_qh stride_qd stride_cosbs stride_cosd stride_sinbs
      stride_sind : Nat) (i : Fin BLOCK_HALF) : ℝ :=
  s.readMem Q (qOffset s stride_qbs stride_qh stride_qd (dimEven i)) *
    s.readMem Sin (sinOffset s stride_sinbs stride_sind (dimEven i)) +
  s.readMem Q (qOffset s stride_qbs stride_qh stride_qd (dimOdd i)) *
    s.readMem Cos (cosOffset s stride_cosbs stride_cosd (dimEven i))
```
</details>

<details><summary><code>rotary_emb_k0_block</code></summary>

```
/-- Proof-oriented K-even-dimension slice of `rotary_emb.py`'s `_rotary_kernel`.

Mirrors the Q0 slice for the K buffer with `HEAD_K` head bound. -/
```
```lean
def rotary_emb_k0_block
    (K Cos Sin : RegionName)
    (stride_kbs stride_kh stride_kd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_K BLOCK_HALF : Nat) :
    ComputeKernel := triton {
  cur_head_index = tl.program_id(0)
  cur_seq_index = tl.program_id(1)
  dim = tl.arange(0, $(BLOCK_HALF))
  dim0 = dim * $(2)
  dim1 = dim * $(2) + $(1)
  k0 = tl.load(K + cur_seq_index * $(stride_kbs) +
      cur_head_index * $(stride_kh) + dim0 * $(stride_kd),
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_K)),
    other=0.0)
  k1 = tl.load(K + cur_seq_index * $(stride_kbs) +
      cur_head_index * $(stride_kh) + dim1 * $(stride_kd),
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_K)),
    other=0.0)
  cos0 = tl.load(Cos + cur_seq_index * $(stride_cosbs) +
      dim0 * $(stride_cosd),
    mask=cur_seq_index < $(max_total_len), other=0.0)
  sin0 = tl.load(Sin + cur_seq_index * $(stride_sinbs) +
      dim0 * $(stride_sind),
    mask=cur_seq_index < $(max_total_len), other=0.0)
  outK0 = k0 * cos0 - k1 * sin0
  tl.store(K + cur_seq_index * $(stride_kbs) +
      cur_head_index * $(stride_kh) + dim0 * $(stride_kd),
    outK0,
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_K)))
}
```
</details>

<details><summary><code>activeK</code></summary>

```lean
def activeK (s : BlockState) (max_total_len HEAD_K : Nat) : Prop :=
  seqIndex s < max_total_len ∧ headIndex s < HEAD_K
```
</details>

<details><summary><code>kOffset</code></summary>

```lean
def kOffset
    (s : BlockState) (stride_kbs stride_kh stride_kd : Nat) (dim : Nat) : Nat :=
  seqIndex s * stride_kbs + headIndex s * stride_kh + dim * stride_kd
```
</details>

<details><summary><code>rotaryK0Spec</code></summary>

```lean
noncomputable def rotaryK0Spec
    (s : BlockState) (K Cos Sin : RegionName)
    (stride_kbs stride_kh stride_kd stride_cosbs stride_cosd stride_sinbs
      stride_sind : Nat) (i : Fin BLOCK_HALF) : ℝ :=
  s.readMem K (kOffset s stride_kbs stride_kh stride_kd (dimEven i)) *
    s.readMem Cos (cosOffset s stride_cosbs stride_cosd (dimEven i)) -
  s.readMem K (kOffset s stride_kbs stride_kh stride_kd (dimOdd i)) *
    s.readMem Sin (sinOffset s stride_sinbs stride_sind (dimEven i))
```
</details>

<details><summary><code>rotary_emb_k1_block</code></summary>

```
/-- Proof-oriented K-odd-dimension slice of `rotary_emb.py`'s `_rotary_kernel`. -/
```
```lean
def rotary_emb_k1_block
    (K Cos Sin : RegionName)
    (stride_kbs stride_kh stride_kd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_K BLOCK_HALF : Nat) :
    ComputeKernel := triton {
  cur_head_index = tl.program_id(0)
  cur_seq_index = tl.program_id(1)
  dim = tl.arange(0, $(BLOCK_HALF))
  dim0 = dim * $(2)
  dim1 = dim * $(2) + $(1)
  k0 = tl.load(K + cur_seq_index * $(stride_kbs) +
      cur_head_index * $(stride_kh) + dim0 * $(stride_kd),
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_K)),
    other=0.0)
  k1 = tl.load(K + cur_seq_index * $(stride_kbs) +
      cur_head_index * $(stride_kh) + dim1 * $(stride_kd),
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_K)),
    other=0.0)
  cos0 = tl.load(Cos + cur_seq_index * $(stride_cosbs) +
      dim0 * $(stride_cosd),
    mask=cur_seq_index < $(max_total_len), other=0.0)
  sin0 = tl.load(Sin + cur_seq_index * $(stride_sinbs) +
      dim0 * $(stride_sind),
    mask=cur_seq_index < $(max_total_len), other=0.0)
  outK1 = k0 * sin0 + k1 * cos0
  tl.store(K + cur_seq_index * $(stride_kbs) +
      cur_head_index * $(stride_kh) + dim1 * $(stride_kd),
    outK1,
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_K)))
}
```
</details>

<details><summary><code>rotaryK1Spec</code></summary>

```lean
noncomputable def rotaryK1Spec
    (s : BlockState) (K Cos Sin : RegionName)
    (stride_kbs stride_kh stride_kd stride_cosbs stride_cosd stride_sinbs
      stride_sind : Nat) (i : Fin BLOCK_HALF) : ℝ :=
  s.readMem K (kOffset s stride_kbs stride_kh stride_kd (dimEven i)) *
    s.readMem Sin (sinOffset s stride_sinbs stride_sind (dimEven i)) +
  s.readMem K (kOffset s stride_kbs stride_kh stride_kd (dimOdd i)) *
    s.readMem Cos (cosOffset s stride_cosbs stride_cosd (dimEven i))
```
</details>

<details><summary><code>seqIndex</code></summary>

```lean
def seqIndex (s : BlockState) : Nat :=
  s.pids 1
```
</details>

<details><summary><code>headIndex</code></summary>

```lean
def headIndex (s : BlockState) : Nat :=
  s.pids 0
```
</details>

<details><summary><code>cosOffset</code></summary>

```lean
def cosOffset
    (s : BlockState) (stride_cosbs stride_cosd : Nat) (dim : Nat) : Nat :=
  seqIndex s * stride_cosbs + dim * stride_cosd
```
</details>

<details><summary><code>sinOffset</code></summary>

```lean
def sinOffset
    (s : BlockState) (stride_sinbs stride_sind : Nat) (dim : Nat) : Nat :=
  seqIndex s * stride_sinbs + dim * stride_sind
```
</details>

## Also present (pinned special-case summaries)
- `rotary_emb_q0_block_compute_correct`
- `rotary_emb_q1_block_compute_correct`
- `rotary_emb_k0_block_compute_correct`
- `rotary_emb_k1_block_compute_correct`
- `rotary_emb_python_shape_all_outputs_compute_correct`
- `rotary_emb_python_case1_all_outputs_compute_correct`
- `rotary_emb_python_case2_all_outputs_compute_correct`
- `rotary_emb_python_case3_all_outputs_compute_correct`
- `rotary_emb_python_case4_all_outputs_compute_correct`
