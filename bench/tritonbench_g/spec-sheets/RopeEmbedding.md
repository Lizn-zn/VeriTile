# Spec sheet — `bench/tritonbench_g/rope_embedding/RopeEmbedding.lean`

**Python source:** `bench/tritonbench_g/rope_embedding/rope_embedding.py`

## Public theorem: `rope_embedding_python_case1_forward_backward_summary`

<details><summary>docstring</summary>

```
/-- Public Python case 1 coverage summary: both forward/backward full surfaces
lower, and both rotary half stores realize the checked outputs for `Q_out` and
`dY_out`. -/
```
</details>

**Statement:**
```lean
theorem rope_embedding_python_case1_forward_backward_summary
    (Q dY cos sin : RegionName) (sQ sDY : BlockState) :
    (∃ alg, (rope_embedding_surface Q 128 cos 16 sin 16 8 32 4
      Bool.false 16 4).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := rope_embedding_forward_first_half Q cos sin
        128 16 16 8 32 4 4 16)
      (initialState := sQ)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 16 => active sQ 32 4 4 16 i)
        (fun i => (Q, qFirstOffset sQ 128 32 4 i)))
      (expected := fun i => ropeFirstSpec sQ Q cos sin 128 16 16 8 32 4 16 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := rope_embedding_forward_second_half Q cos sin
        128 16 16 8 32 4 4 16)
      (initialState := sQ)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 16 => active sQ 32 4 4 16 i)
        (fun i => (Q, qSecondOffset sQ 128 32 4 i)))
      (expected := fun i => ropeSecondSpec sQ Q cos sin 128 16 16 8 32 4 16 i)) ∧
    (∃ alg, (rope_embedding_surface dY 128 cos 16 sin 16 8 32 4
      Bool.true 16 4).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := rope_embedding_backward_first_half dY cos sin
        128 16 16 8 32 4 4 16)
      (initialState := sDY)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 16 => active sDY 32 4 4 16 i)
        (fun i => (dY, qFirstOffset sDY 128 32 4 i)))
      (expected := fun i =>
        ropeBackwardFirstSpec sDY dY cos sin 128 16 16 8 32 4 16 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := rope_embedding_backward_second_half dY cos sin
        128 16 16 8 32 4 4 16)
      (initialState := sDY)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 16 => active sDY 32 4 4 16 i)
        (fun i => (dY, qSecondOffset sDY 128 32 4 i)))
      (expected := fun i =>
        ropeBackwardSecondSpec sDY dY cos sin 128 16 16 8 32 4 16 i))
```

**Assumptions / layout contracts:**
- `fun i : Fin 16 => active sQ 32 4 4 16 i`
- `fun i : Fin 16 => active sQ 32 4 4 16 i`
- `fun i : Fin 16 => active sDY 32 4 4 16 i`
- `fun i : Fin 16 => active sDY 32 4 4 16 i`

**Closed-form spec defs (transitive):** `rope_embedding_surface`, `rope_embedding_forward_first_half`, `active`, `qFirstOffset`, `ropeFirstSpec`, `rope_embedding_forward_second_half`, `qSecondOffset`, `ropeSecondSpec`, `rope_embedding_backward_first_half`, `ropeBackwardFirstSpec`, `rope_embedding_backward_second_half`, `ropeBackwardSecondSpec`, `colIndex`, `headStart`, `cosOffset`, `sinOffset`, `rowMod`

<details><summary><code>rope_embedding_surface</code></summary>

```
/-- Faithful transcription of `rope_embedding.py`'s `_rope_embedding`.

The body preserves the group-head loop, the `BACKWARD_PASS` path, and both
rotary-pair stores. -/
```
```lean
def rope_embedding_surface
    (Q : RegionName) (Q_row_stride : Nat)
    (cos : RegionName) (cos_row_stride : Nat)
    (sin : RegionName) (sin_row_stride seqlen head_dim n_heads : Nat)
    (BACKWARD_PASS : Bool) (BLOCK_SIZE ROPE_GROUP_SIZE : Nat) :
    ComputeKernel := triton {
  row_position = tl.program_id(0)
  group_head_position = tl.program_id(1)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  half_head_dim = $(head_dim / 2)
  mask = col_offsets < $(head_dim / 2)
  sin1 = tl.load(sin + (row_position % $(seqlen)) * $(sin_row_stride) +
    half_head_dim * $(0) + col_offsets, mask=mask, other=0)
  cos1 = tl.load(cos + (row_position % $(seqlen)) * $(cos_row_stride) +
    half_head_dim * $(0) + col_offsets, mask=mask, other=0)
  if BACKWARD_PASS {
    sin1 = -sin1
  }
  head_start = group_head_position * $(ROPE_GROUP_SIZE)
  head_end = min(head_start + $(ROPE_GROUP_SIZE), $(n_heads))
  for k in range(head_start, head_end, $(1)) {
    offs_q1 = row_position * $(Q_row_stride) + k * $(head_dim) + col_offsets
    offs_q2 = row_position * $(Q_row_stride) + k * $(head_dim) +
      col_offsets + half_head_dim
    Q1 = tl.load(Q + offs_q1, mask=mask, other=0).to(sin1.dtype)
    Q2 = tl.load(Q + offs_q2, mask=mask, other=0).to(sin1.dtype)
    tl.store(Q + offs_q1, Q1 * cos1 - Q2 * sin1, mask=mask)
    tl.store(Q + offs_q2, Q2 * cos1 + Q1 * sin1, mask=mask)
  }
}
```
</details>

<details><summary><code>rope_embedding_forward_first_half</code></summary>

```
/-- Proof-oriented forward first-half slice of `rope_embedding.py`'s
`_rope_embedding`.

The full kernel loops over `ROPE_GROUP_SIZE` heads and writes both halves of
the rotary pair. This slice captures one group's first head and first-half
store: `Q1 * cos - Q2 * sin`. -/
```
```lean
def rope_embedding_forward_first_half
    (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  row_position = tl.program_id(0)
  group_head_position = tl.program_id(1)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  half_head_dim = $(head_dim / 2)
  mask = col_offsets < $(head_dim / 2)
  sin1 = tl.load(sin + (row_position % $(seqlen)) * $(sin_row_stride) + col_offsets,
    mask=mask, other=0)
  cos1 = tl.load(cos + (row_position % $(seqlen)) * $(cos_row_stride) + col_offsets,
    mask=mask, other=0)
  head_start = group_head_position * $(ROPE_GROUP_SIZE)
  offs_q1 = row_position * $(Q_row_stride) + head_start * $(head_dim) + col_offsets
  offs_q2 = row_position * $(Q_row_stride) + head_start * $(head_dim) +
    col_offsets + $(head_dim / 2)
  Q1 = tl.load(Q + offs_q1, mask=mask, other=0).to(sin1.dtype)
  Q2 = tl.load(Q + offs_q2, mask=mask, other=0).to(sin1.dtype)
  out = Q1 * cos1 - Q2 * sin1
  tl.store(Q + offs_q1, out, mask=mask and head_start < $(n_heads))
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active (s : BlockState) (head_dim n_heads ROPE_GROUP_SIZE BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : Prop :=
  colIndex i < head_dim / 2 ∧ headStart s ROPE_GROUP_SIZE < n_heads
```
</details>

<details><summary><code>qFirstOffset</code></summary>

```lean
def qFirstOffset
    (s : BlockState) (Q_row_stride head_dim ROPE_GROUP_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 0 * Q_row_stride + headStart s ROPE_GROUP_SIZE * head_dim + colIndex i
```
</details>

<details><summary><code>ropeFirstSpec</code></summary>

```lean
noncomputable def ropeFirstSpec
    (s : BlockState) (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim ROPE_GROUP_SIZE
      BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : ℝ :=
  s.readMem Q (qFirstOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i) *
    s.readMem cos (cosOffset s seqlen cos_row_stride i) -
  s.readMem Q (qSecondOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i) *
    s.readMem sin (sinOffset s seqlen sin_row_stride i)
```
</details>

<details><summary><code>rope_embedding_forward_second_half</code></summary>

```
/-- Proof-oriented second-half slice of `_rope_embedding` (forward).
Captures the companion second-half writeback `out = Q2 * cos + Q1 * sin`
to offset `offs_q2`. -/
```
```lean
def rope_embedding_forward_second_half
    (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  row_position = tl.program_id(0)
  group_head_position = tl.program_id(1)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  half_head_dim = $(head_dim / 2)
  mask = col_offsets < $(head_dim / 2)
  sin1 = tl.load(sin + (row_position % $(seqlen)) * $(sin_row_stride) + col_offsets,
    mask=mask, other=0)
  cos1 = tl.load(cos + (row_position % $(seqlen)) * $(cos_row_stride) + col_offsets,
    mask=mask, other=0)
  head_start = group_head_position * $(ROPE_GROUP_SIZE)
  offs_q1 = row_position * $(Q_row_stride) + head_start * $(head_dim) + col_offsets
  offs_q2 = row_position * $(Q_row_stride) + head_start * $(head_dim) +
    col_offsets + $(head_dim / 2)
  Q1 = tl.load(Q + offs_q1, mask=mask, other=0).to(sin1.dtype)
  Q2 = tl.load(Q + offs_q2, mask=mask, other=0).to(sin1.dtype)
  out = Q2 * cos1 + Q1 * sin1
  tl.store(Q + offs_q2, out, mask=mask and head_start < $(n_heads))
}
```
</details>

<details><summary><code>qSecondOffset</code></summary>

```lean
def qSecondOffset
    (s : BlockState) (Q_row_stride head_dim ROPE_GROUP_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 0 * Q_row_stride + headStart s ROPE_GROUP_SIZE * head_dim +
    colIndex i + head_dim / 2
```
</details>

<details><summary><code>ropeSecondSpec</code></summary>

```lean
noncomputable def ropeSecondSpec
    (s : BlockState) (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim ROPE_GROUP_SIZE
      BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : ℝ :=
  s.readMem Q (qSecondOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i) *
    s.readMem cos (cosOffset s seqlen cos_row_stride i) +
  s.readMem Q (qFirstOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i) *
    s.readMem sin (sinOffset s seqlen sin_row_stride i)
```
</details>

<details><summary><code>rope_embedding_backward_first_half</code></summary>

```
/-- Proof-oriented first-half slice of `_rope_embedding` for
`BACKWARD_PASS = true`. The surface flips `sin1 = -sin1`, so the first-half
write becomes `Q1 * cos + Q2 * sin`. -/
```
```lean
def rope_embedding_backward_first_half
    (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  row_position = tl.program_id(0)
  group_head_position = tl.program_id(1)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  half_head_dim = $(head_dim / 2)
  mask = col_offsets < $(head_dim / 2)
  sin1 = tl.load(sin + (row_position % $(seqlen)) * $(sin_row_stride) + col_offsets,
    mask=mask, other=0)
  cos1 = tl.load(cos + (row_position % $(seqlen)) * $(cos_row_stride) + col_offsets,
    mask=mask, other=0)
  sin1 = -sin1
  head_start = group_head_position * $(ROPE_GROUP_SIZE)
  offs_q1 = row_position * $(Q_row_stride) + head_start * $(head_dim) + col_offsets
  offs_q2 = row_position * $(Q_row_stride) + head_start * $(head_dim) +
    col_offsets + $(head_dim / 2)
  Q1 = tl.load(Q + offs_q1, mask=mask, other=0).to(sin1.dtype)
  Q2 = tl.load(Q + offs_q2, mask=mask, other=0).to(sin1.dtype)
  out = Q1 * cos1 - Q2 * sin1
  tl.store(Q + offs_q1, out, mask=mask and head_start < $(n_heads))
}
```
</details>

<details><summary><code>ropeBackwardFirstSpec</code></summary>

```lean
noncomputable def ropeBackwardFirstSpec
    (s : BlockState) (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim ROPE_GROUP_SIZE
      BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : ℝ :=
  s.readMem Q (qFirstOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i) *
    s.readMem cos (cosOffset s seqlen cos_row_stride i) +
  0 - s.readMem Q (qSecondOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i) *
    ((0.0 : ℝ) - s.readMem sin (sinOffset s seqlen sin_row_stride i))
```
</details>

<details><summary><code>rope_embedding_backward_second_half</code></summary>

```
/-- Proof-oriented second-half slice of `_rope_embedding` for
`BACKWARD_PASS = true`. The surface flips `sin1 = -sin1`, so the second-half
write becomes `Q2 * cos - Q1 * sin`. -/
```
```lean
def rope_embedding_backward_second_half
    (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  row_position = tl.program_id(0)
  group_head_position = tl.program_id(1)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  half_head_dim = $(head_dim / 2)
  mask = col_offsets < $(head_dim / 2)
  sin1 = tl.load(sin + (row_position % $(seqlen)) * $(sin_row_stride) + col_offsets,
    mask=mask, other=0)
  cos1 = tl.load(cos + (row_position % $(seqlen)) * $(cos_row_stride) + col_offsets,
    mask=mask, other=0)
  sin1 = -sin1
  head_start = group_head_position * $(ROPE_GROUP_SIZE)
  offs_q1 = row_position * $(Q_row_stride) + head_start * $(head_dim) + col_offsets
  offs_q2 = row_position * $(Q_row_stride) + head_start * $(head_dim) +
    col_offsets + $(head_dim / 2)
  Q1 = tl.load(Q + offs_q1, mask=mask, other=0).to(sin1.dtype)
  Q2 = tl.load(Q + offs_q2, mask=mask, other=0).to(sin1.dtype)
  out = Q2 * cos1 + Q1 * sin1
  tl.store(Q + offs_q2, out, mask=mask and head_start < $(n_heads))
}
```
</details>

<details><summary><code>ropeBackwardSecondSpec</code></summary>

```lean
noncomputable def ropeBackwardSecondSpec
    (s : BlockState) (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim ROPE_GROUP_SIZE
      BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : ℝ :=
  s.readMem Q (qSecondOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i) *
    s.readMem cos (cosOffset s seqlen cos_row_stride i) -
  0 + s.readMem Q (qFirstOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i) *
    ((0.0 : ℝ) - s.readMem sin (sinOffset s seqlen sin_row_stride i))
```
</details>

<details><summary><code>colIndex</code></summary>

```lean
def colIndex (i : Fin BLOCK_SIZE) : Nat :=
  i.val
```
</details>

<details><summary><code>headStart</code></summary>

```lean
def headStart (s : BlockState) (ROPE_GROUP_SIZE : Nat) : Nat :=
  s.pids 1 * ROPE_GROUP_SIZE
```
</details>

<details><summary><code>cosOffset</code></summary>

```lean
def cosOffset
    (s : BlockState) (seqlen cos_row_stride : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  rowMod s seqlen * cos_row_stride + colIndex i
```
</details>

<details><summary><code>sinOffset</code></summary>

```lean
def sinOffset
    (s : BlockState) (seqlen sin_row_stride : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  rowMod s seqlen * sin_row_stride + colIndex i
```
</details>

<details><summary><code>rowMod</code></summary>

```lean
def rowMod (s : BlockState) (seqlen : Nat) : Nat :=
  IntegralDType.nat.mod (s.pids 0) seqlen
```
</details>

## Public theorem: `rope_embedding_python_case2_forward_backward_summary`

<details><summary>docstring</summary>

```
/-- Public Python case 2 coverage summary. -/
```
</details>

**Statement:**
```lean
theorem rope_embedding_python_case2_forward_backward_summary
    (Q dY cos sin : RegionName) (sQ sDY : BlockState) :
    (∃ alg, (rope_embedding_surface Q 2048 cos 64 sin 64 32 128 16
      Bool.false 64 4).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := rope_embedding_forward_first_half Q cos sin
        2048 64 64 32 128 16 4 64)
      (initialState := sQ)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 64 => active sQ 128 16 4 64 i)
        (fun i => (Q, qFirstOffset sQ 2048 128 4 i)))
      (expected := fun i => ropeFirstSpec sQ Q cos sin 2048 64 64 32 128 4 64 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := rope_embedding_forward_second_half Q cos sin
        2048 64 64 32 128 16 4 64)
      (initialState := sQ)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 64 => active sQ 128 16 4 64 i)
        (fun i => (Q, qSecondOffset sQ 2048 128 4 i)))
      (expected := fun i => ropeSecondSpec sQ Q cos sin 2048 64 64 32 128 4 64 i)) ∧
    (∃ alg, (rope_embedding_surface dY 2048 cos 64 sin 64 32 128 16
      Bool.true 64 4).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := rope_embedding_backward_first_half dY cos sin
        2048 64 64 32 128 16 4 64)
      (initialState := sDY)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 64 => active sDY 128 16 4 64 i)
        (fun i => (dY, qFirstOffset sDY 2048 128 4 i)))
      (expected := fun i =>
        ropeBackwardFirstSpec sDY dY cos sin 2048 64 64 32 128 4 64 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := rope_embedding_backward_second_half dY cos sin
        2048 64 64 32 128 16 4 64)
      (initialState := sDY)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 64 => active sDY 128 16 4 64 i)
        (fun i => (dY, qSecondOffset sDY 2048 128 4 i)))
      (expected := fun i =>
        ropeBackwardSecondSpec sDY dY cos sin 2048 64 64 32 128 4 64 i))
```

**Assumptions / layout contracts:**
- `fun i : Fin 64 => active sQ 128 16 4 64 i`
- `fun i : Fin 64 => active sQ 128 16 4 64 i`
- `fun i : Fin 64 => active sDY 128 16 4 64 i`
- `fun i : Fin 64 => active sDY 128 16 4 64 i`

**Closed-form spec defs (transitive):** `rope_embedding_surface`, `rope_embedding_forward_first_half`, `active`, `qFirstOffset`, `ropeFirstSpec`, `rope_embedding_forward_second_half`, `qSecondOffset`, `ropeSecondSpec`, `rope_embedding_backward_first_half`, `ropeBackwardFirstSpec`, `rope_embedding_backward_second_half`, `ropeBackwardSecondSpec`, `colIndex`, `headStart`, `cosOffset`, `sinOffset`, `rowMod`

<details><summary><code>rope_embedding_surface</code></summary>

```
/-- Faithful transcription of `rope_embedding.py`'s `_rope_embedding`.

The body preserves the group-head loop, the `BACKWARD_PASS` path, and both
rotary-pair stores. -/
```
```lean
def rope_embedding_surface
    (Q : RegionName) (Q_row_stride : Nat)
    (cos : RegionName) (cos_row_stride : Nat)
    (sin : RegionName) (sin_row_stride seqlen head_dim n_heads : Nat)
    (BACKWARD_PASS : Bool) (BLOCK_SIZE ROPE_GROUP_SIZE : Nat) :
    ComputeKernel := triton {
  row_position = tl.program_id(0)
  group_head_position = tl.program_id(1)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  half_head_dim = $(head_dim / 2)
  mask = col_offsets < $(head_dim / 2)
  sin1 = tl.load(sin + (row_position % $(seqlen)) * $(sin_row_stride) +
    half_head_dim * $(0) + col_offsets, mask=mask, other=0)
  cos1 = tl.load(cos + (row_position % $(seqlen)) * $(cos_row_stride) +
    half_head_dim * $(0) + col_offsets, mask=mask, other=0)
  if BACKWARD_PASS {
    sin1 = -sin1
  }
  head_start = group_head_position * $(ROPE_GROUP_SIZE)
  head_end = min(head_start + $(ROPE_GROUP_SIZE), $(n_heads))
  for k in range(head_start, head_end, $(1)) {
    offs_q1 = row_position * $(Q_row_stride) + k * $(head_dim) + col_offsets
    offs_q2 = row_position * $(Q_row_stride) + k * $(head_dim) +
      col_offsets + half_head_dim
    Q1 = tl.load(Q + offs_q1, mask=mask, other=0).to(sin1.dtype)
    Q2 = tl.load(Q + offs_q2, mask=mask, other=0).to(sin1.dtype)
    tl.store(Q + offs_q1, Q1 * cos1 - Q2 * sin1, mask=mask)
    tl.store(Q + offs_q2, Q2 * cos1 + Q1 * sin1, mask=mask)
  }
}
```
</details>

<details><summary><code>rope_embedding_forward_first_half</code></summary>

```
/-- Proof-oriented forward first-half slice of `rope_embedding.py`'s
`_rope_embedding`.

The full kernel loops over `ROPE_GROUP_SIZE` heads and writes both halves of
the rotary pair. This slice captures one group's first head and first-half
store: `Q1 * cos - Q2 * sin`. -/
```
```lean
def rope_embedding_forward_first_half
    (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  row_position = tl.program_id(0)
  group_head_position = tl.program_id(1)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  half_head_dim = $(head_dim / 2)
  mask = col_offsets < $(head_dim / 2)
  sin1 = tl.load(sin + (row_position % $(seqlen)) * $(sin_row_stride) + col_offsets,
    mask=mask, other=0)
  cos1 = tl.load(cos + (row_position % $(seqlen)) * $(cos_row_stride) + col_offsets,
    mask=mask, other=0)
  head_start = group_head_position * $(ROPE_GROUP_SIZE)
  offs_q1 = row_position * $(Q_row_stride) + head_start * $(head_dim) + col_offsets
  offs_q2 = row_position * $(Q_row_stride) + head_start * $(head_dim) +
    col_offsets + $(head_dim / 2)
  Q1 = tl.load(Q + offs_q1, mask=mask, other=0).to(sin1.dtype)
  Q2 = tl.load(Q + offs_q2, mask=mask, other=0).to(sin1.dtype)
  out = Q1 * cos1 - Q2 * sin1
  tl.store(Q + offs_q1, out, mask=mask and head_start < $(n_heads))
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active (s : BlockState) (head_dim n_heads ROPE_GROUP_SIZE BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : Prop :=
  colIndex i < head_dim / 2 ∧ headStart s ROPE_GROUP_SIZE < n_heads
```
</details>

<details><summary><code>qFirstOffset</code></summary>

```lean
def qFirstOffset
    (s : BlockState) (Q_row_stride head_dim ROPE_GROUP_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 0 * Q_row_stride + headStart s ROPE_GROUP_SIZE * head_dim + colIndex i
```
</details>

<details><summary><code>ropeFirstSpec</code></summary>

```lean
noncomputable def ropeFirstSpec
    (s : BlockState) (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim ROPE_GROUP_SIZE
      BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : ℝ :=
  s.readMem Q (qFirstOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i) *
    s.readMem cos (cosOffset s seqlen cos_row_stride i) -
  s.readMem Q (qSecondOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i) *
    s.readMem sin (sinOffset s seqlen sin_row_stride i)
```
</details>

<details><summary><code>rope_embedding_forward_second_half</code></summary>

```
/-- Proof-oriented second-half slice of `_rope_embedding` (forward).
Captures the companion second-half writeback `out = Q2 * cos + Q1 * sin`
to offset `offs_q2`. -/
```
```lean
def rope_embedding_forward_second_half
    (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  row_position = tl.program_id(0)
  group_head_position = tl.program_id(1)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  half_head_dim = $(head_dim / 2)
  mask = col_offsets < $(head_dim / 2)
  sin1 = tl.load(sin + (row_position % $(seqlen)) * $(sin_row_stride) + col_offsets,
    mask=mask, other=0)
  cos1 = tl.load(cos + (row_position % $(seqlen)) * $(cos_row_stride) + col_offsets,
    mask=mask, other=0)
  head_start = group_head_position * $(ROPE_GROUP_SIZE)
  offs_q1 = row_position * $(Q_row_stride) + head_start * $(head_dim) + col_offsets
  offs_q2 = row_position * $(Q_row_stride) + head_start * $(head_dim) +
    col_offsets + $(head_dim / 2)
  Q1 = tl.load(Q + offs_q1, mask=mask, other=0).to(sin1.dtype)
  Q2 = tl.load(Q + offs_q2, mask=mask, other=0).to(sin1.dtype)
  out = Q2 * cos1 + Q1 * sin1
  tl.store(Q + offs_q2, out, mask=mask and head_start < $(n_heads))
}
```
</details>

<details><summary><code>qSecondOffset</code></summary>

```lean
def qSecondOffset
    (s : BlockState) (Q_row_stride head_dim ROPE_GROUP_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 0 * Q_row_stride + headStart s ROPE_GROUP_SIZE * head_dim +
    colIndex i + head_dim / 2
```
</details>

<details><summary><code>ropeSecondSpec</code></summary>

```lean
noncomputable def ropeSecondSpec
    (s : BlockState) (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim ROPE_GROUP_SIZE
      BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : ℝ :=
  s.readMem Q (qSecondOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i) *
    s.readMem cos (cosOffset s seqlen cos_row_stride i) +
  s.readMem Q (qFirstOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i) *
    s.readMem sin (sinOffset s seqlen sin_row_stride i)
```
</details>

<details><summary><code>rope_embedding_backward_first_half</code></summary>

```
/-- Proof-oriented first-half slice of `_rope_embedding` for
`BACKWARD_PASS = true`. The surface flips `sin1 = -sin1`, so the first-half
write becomes `Q1 * cos + Q2 * sin`. -/
```
```lean
def rope_embedding_backward_first_half
    (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  row_position = tl.program_id(0)
  group_head_position = tl.program_id(1)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  half_head_dim = $(head_dim / 2)
  mask = col_offsets < $(head_dim / 2)
  sin1 = tl.load(sin + (row_position % $(seqlen)) * $(sin_row_stride) + col_offsets,
    mask=mask, other=0)
  cos1 = tl.load(cos + (row_position % $(seqlen)) * $(cos_row_stride) + col_offsets,
    mask=mask, other=0)
  sin1 = -sin1
  head_start = group_head_position * $(ROPE_GROUP_SIZE)
  offs_q1 = row_position * $(Q_row_stride) + head_start * $(head_dim) + col_offsets
  offs_q2 = row_position * $(Q_row_stride) + head_start * $(head_dim) +
    col_offsets + $(head_dim / 2)
  Q1 = tl.load(Q + offs_q1, mask=mask, other=0).to(sin1.dtype)
  Q2 = tl.load(Q + offs_q2, mask=mask, other=0).to(sin1.dtype)
  out = Q1 * cos1 - Q2 * sin1
  tl.store(Q + offs_q1, out, mask=mask and head_start < $(n_heads))
}
```
</details>

<details><summary><code>ropeBackwardFirstSpec</code></summary>

```lean
noncomputable def ropeBackwardFirstSpec
    (s : BlockState) (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim ROPE_GROUP_SIZE
      BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : ℝ :=
  s.readMem Q (qFirstOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i) *
    s.readMem cos (cosOffset s seqlen cos_row_stride i) +
  0 - s.readMem Q (qSecondOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i) *
    ((0.0 : ℝ) - s.readMem sin (sinOffset s seqlen sin_row_stride i))
```
</details>

<details><summary><code>rope_embedding_backward_second_half</code></summary>

```
/-- Proof-oriented second-half slice of `_rope_embedding` for
`BACKWARD_PASS = true`. The surface flips `sin1 = -sin1`, so the second-half
write becomes `Q2 * cos - Q1 * sin`. -/
```
```lean
def rope_embedding_backward_second_half
    (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  row_position = tl.program_id(0)
  group_head_position = tl.program_id(1)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  half_head_dim = $(head_dim / 2)
  mask = col_offsets < $(head_dim / 2)
  sin1 = tl.load(sin + (row_position % $(seqlen)) * $(sin_row_stride) + col_offsets,
    mask=mask, other=0)
  cos1 = tl.load(cos + (row_position % $(seqlen)) * $(cos_row_stride) + col_offsets,
    mask=mask, other=0)
  sin1 = -sin1
  head_start = group_head_position * $(ROPE_GROUP_SIZE)
  offs_q1 = row_position * $(Q_row_stride) + head_start * $(head_dim) + col_offsets
  offs_q2 = row_position * $(Q_row_stride) + head_start * $(head_dim) +
    col_offsets + $(head_dim / 2)
  Q1 = tl.load(Q + offs_q1, mask=mask, other=0).to(sin1.dtype)
  Q2 = tl.load(Q + offs_q2, mask=mask, other=0).to(sin1.dtype)
  out = Q2 * cos1 + Q1 * sin1
  tl.store(Q + offs_q2, out, mask=mask and head_start < $(n_heads))
}
```
</details>

<details><summary><code>ropeBackwardSecondSpec</code></summary>

```lean
noncomputable def ropeBackwardSecondSpec
    (s : BlockState) (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim ROPE_GROUP_SIZE
      BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : ℝ :=
  s.readMem Q (qSecondOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i) *
    s.readMem cos (cosOffset s seqlen cos_row_stride i) -
  0 + s.readMem Q (qFirstOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i) *
    ((0.0 : ℝ) - s.readMem sin (sinOffset s seqlen sin_row_stride i))
```
</details>

<details><summary><code>colIndex</code></summary>

```lean
def colIndex (i : Fin BLOCK_SIZE) : Nat :=
  i.val
```
</details>

<details><summary><code>headStart</code></summary>

```lean
def headStart (s : BlockState) (ROPE_GROUP_SIZE : Nat) : Nat :=
  s.pids 1 * ROPE_GROUP_SIZE
```
</details>

<details><summary><code>cosOffset</code></summary>

```lean
def cosOffset
    (s : BlockState) (seqlen cos_row_stride : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  rowMod s seqlen * cos_row_stride + colIndex i
```
</details>

<details><summary><code>sinOffset</code></summary>

```lean
def sinOffset
    (s : BlockState) (seqlen sin_row_stride : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  rowMod s seqlen * sin_row_stride + colIndex i
```
</details>

<details><summary><code>rowMod</code></summary>

```lean
def rowMod (s : BlockState) (seqlen : Nat) : Nat :=
  IntegralDType.nat.mod (s.pids 0) seqlen
```
</details>

## Public theorem: `rope_embedding_python_case3_forward_backward_summary`

<details><summary>docstring</summary>

```
/-- Public Python case 3 coverage summary. -/
```
</details>

**Statement:**
```lean
theorem rope_embedding_python_case3_forward_backward_summary
    (Q dY cos sin : RegionName) (sQ sDY : BlockState) :
    (∃ alg, (rope_embedding_surface Q 8192 cos 128 sin 128 64 256 32
      Bool.false 128 4).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := rope_embedding_forward_first_half Q cos sin
        8192 128 128 64 256 32 4 128)
      (initialState := sQ)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 128 => active sQ 256 32 4 128 i)
        (fun i => (Q, qFirstOffset sQ 8192 256 4 i)))
      (expected := fun i => ropeFirstSpec sQ Q cos sin 8192 128 128 64 256 4 128 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := rope_embedding_forward_second_half Q cos sin
        8192 128 128 64 256 32 4 128)
      (initialState := sQ)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 128 => active sQ 256 32 4 128 i)
        (fun i => (Q, qSecondOffset sQ 8192 256 4 i)))
      (expected := fun i => ropeSecondSpec sQ Q cos sin 8192 128 128 64 256 4 128 i)) ∧
    (∃ alg, (rope_embedding_surface dY 8192 cos 128 sin 128 64 256 32
      Bool.true 128 4).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := rope_embedding_backward_first_half dY cos sin
        8192 128 128 64 256 32 4 128)
      (initialState := sDY)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 128 => active sDY 256 32 4 128 i)
        (fun i => (dY, qFirstOffset sDY 8192 256 4 i)))
      (expected := fun i =>
        ropeBackwardFirstSpec sDY dY cos sin 8192 128 128 64 256 4 128 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := rope_embedding_backward_second_half dY cos sin
        8192 128 128 64 256 32 4 128)
      (initialState := sDY)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 128 => active sDY 256 32 4 128 i)
        (fun i => (dY, qSecondOffset sDY 8192 256 4 i)))
      (expected := fun i =>
        ropeBackwardSecondSpec sDY dY cos sin 8192 128 128 64 256 4 128 i))
```

**Assumptions / layout contracts:**
- `fun i : Fin 128 => active sQ 256 32 4 128 i`
- `fun i : Fin 128 => active sQ 256 32 4 128 i`
- `fun i : Fin 128 => active sDY 256 32 4 128 i`
- `fun i : Fin 128 => active sDY 256 32 4 128 i`

**Closed-form spec defs (transitive):** `rope_embedding_surface`, `rope_embedding_forward_first_half`, `active`, `qFirstOffset`, `ropeFirstSpec`, `rope_embedding_forward_second_half`, `qSecondOffset`, `ropeSecondSpec`, `rope_embedding_backward_first_half`, `ropeBackwardFirstSpec`, `rope_embedding_backward_second_half`, `ropeBackwardSecondSpec`, `colIndex`, `headStart`, `cosOffset`, `sinOffset`, `rowMod`

<details><summary><code>rope_embedding_surface</code></summary>

```
/-- Faithful transcription of `rope_embedding.py`'s `_rope_embedding`.

The body preserves the group-head loop, the `BACKWARD_PASS` path, and both
rotary-pair stores. -/
```
```lean
def rope_embedding_surface
    (Q : RegionName) (Q_row_stride : Nat)
    (cos : RegionName) (cos_row_stride : Nat)
    (sin : RegionName) (sin_row_stride seqlen head_dim n_heads : Nat)
    (BACKWARD_PASS : Bool) (BLOCK_SIZE ROPE_GROUP_SIZE : Nat) :
    ComputeKernel := triton {
  row_position = tl.program_id(0)
  group_head_position = tl.program_id(1)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  half_head_dim = $(head_dim / 2)
  mask = col_offsets < $(head_dim / 2)
  sin1 = tl.load(sin + (row_position % $(seqlen)) * $(sin_row_stride) +
    half_head_dim * $(0) + col_offsets, mask=mask, other=0)
  cos1 = tl.load(cos + (row_position % $(seqlen)) * $(cos_row_stride) +
    half_head_dim * $(0) + col_offsets, mask=mask, other=0)
  if BACKWARD_PASS {
    sin1 = -sin1
  }
  head_start = group_head_position * $(ROPE_GROUP_SIZE)
  head_end = min(head_start + $(ROPE_GROUP_SIZE), $(n_heads))
  for k in range(head_start, head_end, $(1)) {
    offs_q1 = row_position * $(Q_row_stride) + k * $(head_dim) + col_offsets
    offs_q2 = row_position * $(Q_row_stride) + k * $(head_dim) +
      col_offsets + half_head_dim
    Q1 = tl.load(Q + offs_q1, mask=mask, other=0).to(sin1.dtype)
    Q2 = tl.load(Q + offs_q2, mask=mask, other=0).to(sin1.dtype)
    tl.store(Q + offs_q1, Q1 * cos1 - Q2 * sin1, mask=mask)
    tl.store(Q + offs_q2, Q2 * cos1 + Q1 * sin1, mask=mask)
  }
}
```
</details>

<details><summary><code>rope_embedding_forward_first_half</code></summary>

```
/-- Proof-oriented forward first-half slice of `rope_embedding.py`'s
`_rope_embedding`.

The full kernel loops over `ROPE_GROUP_SIZE` heads and writes both halves of
the rotary pair. This slice captures one group's first head and first-half
store: `Q1 * cos - Q2 * sin`. -/
```
```lean
def rope_embedding_forward_first_half
    (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  row_position = tl.program_id(0)
  group_head_position = tl.program_id(1)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  half_head_dim = $(head_dim / 2)
  mask = col_offsets < $(head_dim / 2)
  sin1 = tl.load(sin + (row_position % $(seqlen)) * $(sin_row_stride) + col_offsets,
    mask=mask, other=0)
  cos1 = tl.load(cos + (row_position % $(seqlen)) * $(cos_row_stride) + col_offsets,
    mask=mask, other=0)
  head_start = group_head_position * $(ROPE_GROUP_SIZE)
  offs_q1 = row_position * $(Q_row_stride) + head_start * $(head_dim) + col_offsets
  offs_q2 = row_position * $(Q_row_stride) + head_start * $(head_dim) +
    col_offsets + $(head_dim / 2)
  Q1 = tl.load(Q + offs_q1, mask=mask, other=0).to(sin1.dtype)
  Q2 = tl.load(Q + offs_q2, mask=mask, other=0).to(sin1.dtype)
  out = Q1 * cos1 - Q2 * sin1
  tl.store(Q + offs_q1, out, mask=mask and head_start < $(n_heads))
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active (s : BlockState) (head_dim n_heads ROPE_GROUP_SIZE BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : Prop :=
  colIndex i < head_dim / 2 ∧ headStart s ROPE_GROUP_SIZE < n_heads
```
</details>

<details><summary><code>qFirstOffset</code></summary>

```lean
def qFirstOffset
    (s : BlockState) (Q_row_stride head_dim ROPE_GROUP_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 0 * Q_row_stride + headStart s ROPE_GROUP_SIZE * head_dim + colIndex i
```
</details>

<details><summary><code>ropeFirstSpec</code></summary>

```lean
noncomputable def ropeFirstSpec
    (s : BlockState) (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim ROPE_GROUP_SIZE
      BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : ℝ :=
  s.readMem Q (qFirstOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i) *
    s.readMem cos (cosOffset s seqlen cos_row_stride i) -
  s.readMem Q (qSecondOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i) *
    s.readMem sin (sinOffset s seqlen sin_row_stride i)
```
</details>

<details><summary><code>rope_embedding_forward_second_half</code></summary>

```
/-- Proof-oriented second-half slice of `_rope_embedding` (forward).
Captures the companion second-half writeback `out = Q2 * cos + Q1 * sin`
to offset `offs_q2`. -/
```
```lean
def rope_embedding_forward_second_half
    (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  row_position = tl.program_id(0)
  group_head_position = tl.program_id(1)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  half_head_dim = $(head_dim / 2)
  mask = col_offsets < $(head_dim / 2)
  sin1 = tl.load(sin + (row_position % $(seqlen)) * $(sin_row_stride) + col_offsets,
    mask=mask, other=0)
  cos1 = tl.load(cos + (row_position % $(seqlen)) * $(cos_row_stride) + col_offsets,
    mask=mask, other=0)
  head_start = group_head_position * $(ROPE_GROUP_SIZE)
  offs_q1 = row_position * $(Q_row_stride) + head_start * $(head_dim) + col_offsets
  offs_q2 = row_position * $(Q_row_stride) + head_start * $(head_dim) +
    col_offsets + $(head_dim / 2)
  Q1 = tl.load(Q + offs_q1, mask=mask, other=0).to(sin1.dtype)
  Q2 = tl.load(Q + offs_q2, mask=mask, other=0).to(sin1.dtype)
  out = Q2 * cos1 + Q1 * sin1
  tl.store(Q + offs_q2, out, mask=mask and head_start < $(n_heads))
}
```
</details>

<details><summary><code>qSecondOffset</code></summary>

```lean
def qSecondOffset
    (s : BlockState) (Q_row_stride head_dim ROPE_GROUP_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 0 * Q_row_stride + headStart s ROPE_GROUP_SIZE * head_dim +
    colIndex i + head_dim / 2
```
</details>

<details><summary><code>ropeSecondSpec</code></summary>

```lean
noncomputable def ropeSecondSpec
    (s : BlockState) (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim ROPE_GROUP_SIZE
      BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : ℝ :=
  s.readMem Q (qSecondOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i) *
    s.readMem cos (cosOffset s seqlen cos_row_stride i) +
  s.readMem Q (qFirstOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i) *
    s.readMem sin (sinOffset s seqlen sin_row_stride i)
```
</details>

<details><summary><code>rope_embedding_backward_first_half</code></summary>

```
/-- Proof-oriented first-half slice of `_rope_embedding` for
`BACKWARD_PASS = true`. The surface flips `sin1 = -sin1`, so the first-half
write becomes `Q1 * cos + Q2 * sin`. -/
```
```lean
def rope_embedding_backward_first_half
    (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  row_position = tl.program_id(0)
  group_head_position = tl.program_id(1)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  half_head_dim = $(head_dim / 2)
  mask = col_offsets < $(head_dim / 2)
  sin1 = tl.load(sin + (row_position % $(seqlen)) * $(sin_row_stride) + col_offsets,
    mask=mask, other=0)
  cos1 = tl.load(cos + (row_position % $(seqlen)) * $(cos_row_stride) + col_offsets,
    mask=mask, other=0)
  sin1 = -sin1
  head_start = group_head_position * $(ROPE_GROUP_SIZE)
  offs_q1 = row_position * $(Q_row_stride) + head_start * $(head_dim) + col_offsets
  offs_q2 = row_position * $(Q_row_stride) + head_start * $(head_dim) +
    col_offsets + $(head_dim / 2)
  Q1 = tl.load(Q + offs_q1, mask=mask, other=0).to(sin1.dtype)
  Q2 = tl.load(Q + offs_q2, mask=mask, other=0).to(sin1.dtype)
  out = Q1 * cos1 - Q2 * sin1
  tl.store(Q + offs_q1, out, mask=mask and head_start < $(n_heads))
}
```
</details>

<details><summary><code>ropeBackwardFirstSpec</code></summary>

```lean
noncomputable def ropeBackwardFirstSpec
    (s : BlockState) (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim ROPE_GROUP_SIZE
      BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : ℝ :=
  s.readMem Q (qFirstOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i) *
    s.readMem cos (cosOffset s seqlen cos_row_stride i) +
  0 - s.readMem Q (qSecondOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i) *
    ((0.0 : ℝ) - s.readMem sin (sinOffset s seqlen sin_row_stride i))
```
</details>

<details><summary><code>rope_embedding_backward_second_half</code></summary>

```
/-- Proof-oriented second-half slice of `_rope_embedding` for
`BACKWARD_PASS = true`. The surface flips `sin1 = -sin1`, so the second-half
write becomes `Q2 * cos - Q1 * sin`. -/
```
```lean
def rope_embedding_backward_second_half
    (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  row_position = tl.program_id(0)
  group_head_position = tl.program_id(1)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  half_head_dim = $(head_dim / 2)
  mask = col_offsets < $(head_dim / 2)
  sin1 = tl.load(sin + (row_position % $(seqlen)) * $(sin_row_stride) + col_offsets,
    mask=mask, other=0)
  cos1 = tl.load(cos + (row_position % $(seqlen)) * $(cos_row_stride) + col_offsets,
    mask=mask, other=0)
  sin1 = -sin1
  head_start = group_head_position * $(ROPE_GROUP_SIZE)
  offs_q1 = row_position * $(Q_row_stride) + head_start * $(head_dim) + col_offsets
  offs_q2 = row_position * $(Q_row_stride) + head_start * $(head_dim) +
    col_offsets + $(head_dim / 2)
  Q1 = tl.load(Q + offs_q1, mask=mask, other=0).to(sin1.dtype)
  Q2 = tl.load(Q + offs_q2, mask=mask, other=0).to(sin1.dtype)
  out = Q2 * cos1 + Q1 * sin1
  tl.store(Q + offs_q2, out, mask=mask and head_start < $(n_heads))
}
```
</details>

<details><summary><code>ropeBackwardSecondSpec</code></summary>

```lean
noncomputable def ropeBackwardSecondSpec
    (s : BlockState) (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim ROPE_GROUP_SIZE
      BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : ℝ :=
  s.readMem Q (qSecondOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i) *
    s.readMem cos (cosOffset s seqlen cos_row_stride i) -
  0 + s.readMem Q (qFirstOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i) *
    ((0.0 : ℝ) - s.readMem sin (sinOffset s seqlen sin_row_stride i))
```
</details>

<details><summary><code>colIndex</code></summary>

```lean
def colIndex (i : Fin BLOCK_SIZE) : Nat :=
  i.val
```
</details>

<details><summary><code>headStart</code></summary>

```lean
def headStart (s : BlockState) (ROPE_GROUP_SIZE : Nat) : Nat :=
  s.pids 1 * ROPE_GROUP_SIZE
```
</details>

<details><summary><code>cosOffset</code></summary>

```lean
def cosOffset
    (s : BlockState) (seqlen cos_row_stride : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  rowMod s seqlen * cos_row_stride + colIndex i
```
</details>

<details><summary><code>sinOffset</code></summary>

```lean
def sinOffset
    (s : BlockState) (seqlen sin_row_stride : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  rowMod s seqlen * sin_row_stride + colIndex i
```
</details>

<details><summary><code>rowMod</code></summary>

```lean
def rowMod (s : BlockState) (seqlen : Nat) : Nat :=
  IntegralDType.nat.mod (s.pids 0) seqlen
```
</details>

## Public theorem: `rope_embedding_python_case4_forward_backward_summary`

<details><summary>docstring</summary>

```
/-- Public Python case 4 coverage summary. -/
```
</details>

**Statement:**
```lean
theorem rope_embedding_python_case4_forward_backward_summary
    (Q dY cos sin : RegionName) (sQ sDY : BlockState) :
    (∃ alg, (rope_embedding_surface Q 32768 cos 256 sin 256 128 512 64
      Bool.false 256 4).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := rope_embedding_forward_first_half Q cos sin
        32768 256 256 128 512 64 4 256)
      (initialState := sQ)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 256 => active sQ 512 64 4 256 i)
        (fun i => (Q, qFirstOffset sQ 32768 512 4 i)))
      (expected := fun i => ropeFirstSpec sQ Q cos sin 32768 256 256 128 512 4 256 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := rope_embedding_forward_second_half Q cos sin
        32768 256 256 128 512 64 4 256)
      (initialState := sQ)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 256 => active sQ 512 64 4 256 i)
        (fun i => (Q, qSecondOffset sQ 32768 512 4 i)))
      (expected := fun i => ropeSecondSpec sQ Q cos sin 32768 256 256 128 512 4 256 i)) ∧
    (∃ alg, (rope_embedding_surface dY 32768 cos 256 sin 256 128 512 64
      Bool.true 256 4).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := rope_embedding_backward_first_half dY cos sin
        32768 256 256 128 512 64 4 256)
      (initialState := sDY)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 256 => active sDY 512 64 4 256 i)
        (fun i => (dY, qFirstOffset sDY 32768 512 4 i)))
      (expected := fun i =>
        ropeBackwardFirstSpec sDY dY cos sin 32768 256 256 128 512 4 256 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := rope_embedding_backward_second_half dY cos sin
        32768 256 256 128 512 64 4 256)
      (initialState := sDY)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 256 => active sDY 512 64 4 256 i)
        (fun i => (dY, qSecondOffset sDY 32768 512 4 i)))
      (expected := fun i =>
        ropeBackwardSecondSpec sDY dY cos sin 32768 256 256 128 512 4 256 i))
```

**Assumptions / layout contracts:**
- `fun i : Fin 256 => active sQ 512 64 4 256 i`
- `fun i : Fin 256 => active sQ 512 64 4 256 i`
- `fun i : Fin 256 => active sDY 512 64 4 256 i`
- `fun i : Fin 256 => active sDY 512 64 4 256 i`

**Closed-form spec defs (transitive):** `rope_embedding_surface`, `rope_embedding_forward_first_half`, `active`, `qFirstOffset`, `ropeFirstSpec`, `rope_embedding_forward_second_half`, `qSecondOffset`, `ropeSecondSpec`, `rope_embedding_backward_first_half`, `ropeBackwardFirstSpec`, `rope_embedding_backward_second_half`, `ropeBackwardSecondSpec`, `colIndex`, `headStart`, `cosOffset`, `sinOffset`, `rowMod`

<details><summary><code>rope_embedding_surface</code></summary>

```
/-- Faithful transcription of `rope_embedding.py`'s `_rope_embedding`.

The body preserves the group-head loop, the `BACKWARD_PASS` path, and both
rotary-pair stores. -/
```
```lean
def rope_embedding_surface
    (Q : RegionName) (Q_row_stride : Nat)
    (cos : RegionName) (cos_row_stride : Nat)
    (sin : RegionName) (sin_row_stride seqlen head_dim n_heads : Nat)
    (BACKWARD_PASS : Bool) (BLOCK_SIZE ROPE_GROUP_SIZE : Nat) :
    ComputeKernel := triton {
  row_position = tl.program_id(0)
  group_head_position = tl.program_id(1)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  half_head_dim = $(head_dim / 2)
  mask = col_offsets < $(head_dim / 2)
  sin1 = tl.load(sin + (row_position % $(seqlen)) * $(sin_row_stride) +
    half_head_dim * $(0) + col_offsets, mask=mask, other=0)
  cos1 = tl.load(cos + (row_position % $(seqlen)) * $(cos_row_stride) +
    half_head_dim * $(0) + col_offsets, mask=mask, other=0)
  if BACKWARD_PASS {
    sin1 = -sin1
  }
  head_start = group_head_position * $(ROPE_GROUP_SIZE)
  head_end = min(head_start + $(ROPE_GROUP_SIZE), $(n_heads))
  for k in range(head_start, head_end, $(1)) {
    offs_q1 = row_position * $(Q_row_stride) + k * $(head_dim) + col_offsets
    offs_q2 = row_position * $(Q_row_stride) + k * $(head_dim) +
      col_offsets + half_head_dim
    Q1 = tl.load(Q + offs_q1, mask=mask, other=0).to(sin1.dtype)
    Q2 = tl.load(Q + offs_q2, mask=mask, other=0).to(sin1.dtype)
    tl.store(Q + offs_q1, Q1 * cos1 - Q2 * sin1, mask=mask)
    tl.store(Q + offs_q2, Q2 * cos1 + Q1 * sin1, mask=mask)
  }
}
```
</details>

<details><summary><code>rope_embedding_forward_first_half</code></summary>

```
/-- Proof-oriented forward first-half slice of `rope_embedding.py`'s
`_rope_embedding`.

The full kernel loops over `ROPE_GROUP_SIZE` heads and writes both halves of
the rotary pair. This slice captures one group's first head and first-half
store: `Q1 * cos - Q2 * sin`. -/
```
```lean
def rope_embedding_forward_first_half
    (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  row_position = tl.program_id(0)
  group_head_position = tl.program_id(1)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  half_head_dim = $(head_dim / 2)
  mask = col_offsets < $(head_dim / 2)
  sin1 = tl.load(sin + (row_position % $(seqlen)) * $(sin_row_stride) + col_offsets,
    mask=mask, other=0)
  cos1 = tl.load(cos + (row_position % $(seqlen)) * $(cos_row_stride) + col_offsets,
    mask=mask, other=0)
  head_start = group_head_position * $(ROPE_GROUP_SIZE)
  offs_q1 = row_position * $(Q_row_stride) + head_start * $(head_dim) + col_offsets
  offs_q2 = row_position * $(Q_row_stride) + head_start * $(head_dim) +
    col_offsets + $(head_dim / 2)
  Q1 = tl.load(Q + offs_q1, mask=mask, other=0).to(sin1.dtype)
  Q2 = tl.load(Q + offs_q2, mask=mask, other=0).to(sin1.dtype)
  out = Q1 * cos1 - Q2 * sin1
  tl.store(Q + offs_q1, out, mask=mask and head_start < $(n_heads))
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active (s : BlockState) (head_dim n_heads ROPE_GROUP_SIZE BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : Prop :=
  colIndex i < head_dim / 2 ∧ headStart s ROPE_GROUP_SIZE < n_heads
```
</details>

<details><summary><code>qFirstOffset</code></summary>

```lean
def qFirstOffset
    (s : BlockState) (Q_row_stride head_dim ROPE_GROUP_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 0 * Q_row_stride + headStart s ROPE_GROUP_SIZE * head_dim + colIndex i
```
</details>

<details><summary><code>ropeFirstSpec</code></summary>

```lean
noncomputable def ropeFirstSpec
    (s : BlockState) (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim ROPE_GROUP_SIZE
      BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : ℝ :=
  s.readMem Q (qFirstOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i) *
    s.readMem cos (cosOffset s seqlen cos_row_stride i) -
  s.readMem Q (qSecondOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i) *
    s.readMem sin (sinOffset s seqlen sin_row_stride i)
```
</details>

<details><summary><code>rope_embedding_forward_second_half</code></summary>

```
/-- Proof-oriented second-half slice of `_rope_embedding` (forward).
Captures the companion second-half writeback `out = Q2 * cos + Q1 * sin`
to offset `offs_q2`. -/
```
```lean
def rope_embedding_forward_second_half
    (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  row_position = tl.program_id(0)
  group_head_position = tl.program_id(1)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  half_head_dim = $(head_dim / 2)
  mask = col_offsets < $(head_dim / 2)
  sin1 = tl.load(sin + (row_position % $(seqlen)) * $(sin_row_stride) + col_offsets,
    mask=mask, other=0)
  cos1 = tl.load(cos + (row_position % $(seqlen)) * $(cos_row_stride) + col_offsets,
    mask=mask, other=0)
  head_start = group_head_position * $(ROPE_GROUP_SIZE)
  offs_q1 = row_position * $(Q_row_stride) + head_start * $(head_dim) + col_offsets
  offs_q2 = row_position * $(Q_row_stride) + head_start * $(head_dim) +
    col_offsets + $(head_dim / 2)
  Q1 = tl.load(Q + offs_q1, mask=mask, other=0).to(sin1.dtype)
  Q2 = tl.load(Q + offs_q2, mask=mask, other=0).to(sin1.dtype)
  out = Q2 * cos1 + Q1 * sin1
  tl.store(Q + offs_q2, out, mask=mask and head_start < $(n_heads))
}
```
</details>

<details><summary><code>qSecondOffset</code></summary>

```lean
def qSecondOffset
    (s : BlockState) (Q_row_stride head_dim ROPE_GROUP_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 0 * Q_row_stride + headStart s ROPE_GROUP_SIZE * head_dim +
    colIndex i + head_dim / 2
```
</details>

<details><summary><code>ropeSecondSpec</code></summary>

```lean
noncomputable def ropeSecondSpec
    (s : BlockState) (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim ROPE_GROUP_SIZE
      BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : ℝ :=
  s.readMem Q (qSecondOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i) *
    s.readMem cos (cosOffset s seqlen cos_row_stride i) +
  s.readMem Q (qFirstOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i) *
    s.readMem sin (sinOffset s seqlen sin_row_stride i)
```
</details>

<details><summary><code>rope_embedding_backward_first_half</code></summary>

```
/-- Proof-oriented first-half slice of `_rope_embedding` for
`BACKWARD_PASS = true`. The surface flips `sin1 = -sin1`, so the first-half
write becomes `Q1 * cos + Q2 * sin`. -/
```
```lean
def rope_embedding_backward_first_half
    (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  row_position = tl.program_id(0)
  group_head_position = tl.program_id(1)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  half_head_dim = $(head_dim / 2)
  mask = col_offsets < $(head_dim / 2)
  sin1 = tl.load(sin + (row_position % $(seqlen)) * $(sin_row_stride) + col_offsets,
    mask=mask, other=0)
  cos1 = tl.load(cos + (row_position % $(seqlen)) * $(cos_row_stride) + col_offsets,
    mask=mask, other=0)
  sin1 = -sin1
  head_start = group_head_position * $(ROPE_GROUP_SIZE)
  offs_q1 = row_position * $(Q_row_stride) + head_start * $(head_dim) + col_offsets
  offs_q2 = row_position * $(Q_row_stride) + head_start * $(head_dim) +
    col_offsets + $(head_dim / 2)
  Q1 = tl.load(Q + offs_q1, mask=mask, other=0).to(sin1.dtype)
  Q2 = tl.load(Q + offs_q2, mask=mask, other=0).to(sin1.dtype)
  out = Q1 * cos1 - Q2 * sin1
  tl.store(Q + offs_q1, out, mask=mask and head_start < $(n_heads))
}
```
</details>

<details><summary><code>ropeBackwardFirstSpec</code></summary>

```lean
noncomputable def ropeBackwardFirstSpec
    (s : BlockState) (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim ROPE_GROUP_SIZE
      BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : ℝ :=
  s.readMem Q (qFirstOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i) *
    s.readMem cos (cosOffset s seqlen cos_row_stride i) +
  0 - s.readMem Q (qSecondOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i) *
    ((0.0 : ℝ) - s.readMem sin (sinOffset s seqlen sin_row_stride i))
```
</details>

<details><summary><code>rope_embedding_backward_second_half</code></summary>

```
/-- Proof-oriented second-half slice of `_rope_embedding` for
`BACKWARD_PASS = true`. The surface flips `sin1 = -sin1`, so the second-half
write becomes `Q2 * cos - Q1 * sin`. -/
```
```lean
def rope_embedding_backward_second_half
    (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  row_position = tl.program_id(0)
  group_head_position = tl.program_id(1)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  half_head_dim = $(head_dim / 2)
  mask = col_offsets < $(head_dim / 2)
  sin1 = tl.load(sin + (row_position % $(seqlen)) * $(sin_row_stride) + col_offsets,
    mask=mask, other=0)
  cos1 = tl.load(cos + (row_position % $(seqlen)) * $(cos_row_stride) + col_offsets,
    mask=mask, other=0)
  sin1 = -sin1
  head_start = group_head_position * $(ROPE_GROUP_SIZE)
  offs_q1 = row_position * $(Q_row_stride) + head_start * $(head_dim) + col_offsets
  offs_q2 = row_position * $(Q_row_stride) + head_start * $(head_dim) +
    col_offsets + $(head_dim / 2)
  Q1 = tl.load(Q + offs_q1, mask=mask, other=0).to(sin1.dtype)
  Q2 = tl.load(Q + offs_q2, mask=mask, other=0).to(sin1.dtype)
  out = Q2 * cos1 + Q1 * sin1
  tl.store(Q + offs_q2, out, mask=mask and head_start < $(n_heads))
}
```
</details>

<details><summary><code>ropeBackwardSecondSpec</code></summary>

```lean
noncomputable def ropeBackwardSecondSpec
    (s : BlockState) (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim ROPE_GROUP_SIZE
      BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : ℝ :=
  s.readMem Q (qSecondOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i) *
    s.readMem cos (cosOffset s seqlen cos_row_stride i) -
  0 + s.readMem Q (qFirstOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i) *
    ((0.0 : ℝ) - s.readMem sin (sinOffset s seqlen sin_row_stride i))
```
</details>

<details><summary><code>colIndex</code></summary>

```lean
def colIndex (i : Fin BLOCK_SIZE) : Nat :=
  i.val
```
</details>

<details><summary><code>headStart</code></summary>

```lean
def headStart (s : BlockState) (ROPE_GROUP_SIZE : Nat) : Nat :=
  s.pids 1 * ROPE_GROUP_SIZE
```
</details>

<details><summary><code>cosOffset</code></summary>

```lean
def cosOffset
    (s : BlockState) (seqlen cos_row_stride : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  rowMod s seqlen * cos_row_stride + colIndex i
```
</details>

<details><summary><code>sinOffset</code></summary>

```lean
def sinOffset
    (s : BlockState) (seqlen sin_row_stride : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  rowMod s seqlen * sin_row_stride + colIndex i
```
</details>

<details><summary><code>rowMod</code></summary>

```lean
def rowMod (s : BlockState) (seqlen : Nat) : Nat :=
  IntegralDType.nat.mod (s.pids 0) seqlen
```
</details>

## Also present (pinned special-case summaries)
- `rope_embedding_forward_first_half_compute_correct`
- `rope_embedding_forward_second_half_compute_correct`
- `rope_embedding_backward_first_half_compute_correct`
- `rope_embedding_backward_second_half_compute_correct`
- `rope_embedding_python_base_first_half_compute_correct`
- `rope_embedding_python_base_second_half_compute_correct`
- `rope_embedding_python_case1_first_half_compute_correct`
- `rope_embedding_python_case1_second_half_compute_correct`
- `rope_embedding_python_case2_first_half_compute_correct`
- `rope_embedding_python_case2_second_half_compute_correct`
- `rope_embedding_python_case3_first_half_compute_correct`
- `rope_embedding_python_case3_second_half_compute_correct`
- `rope_embedding_python_case4_first_half_compute_correct`
- `rope_embedding_python_case4_second_half_compute_correct`
- `rope_embedding_python_base_backward_first_half_compute_correct`
- `rope_embedding_python_base_backward_second_half_compute_correct`
- `rope_embedding_python_case1_backward_first_half_compute_correct`
- `rope_embedding_python_case1_backward_second_half_compute_correct`
- `rope_embedding_python_case2_backward_first_half_compute_correct`
- `rope_embedding_python_case2_backward_second_half_compute_correct`
- `rope_embedding_python_case3_backward_first_half_compute_correct`
- `rope_embedding_python_case3_backward_second_half_compute_correct`
- `rope_embedding_python_case4_backward_first_half_compute_correct`
- `rope_embedding_python_case4_backward_second_half_compute_correct`
