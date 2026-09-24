# Spec sheet — `bench/tritonbench_g/rope_embedding/RopeEmbedding.lean`

**Python source:** `bench/tritonbench_g/rope_embedding/rope_embedding.py`

## Public theorem: `rope_embedding_forward_backward_summary_general`

<details><summary>docstring</summary>

```
/-- **Dimension-general forward+backward output summary.** For arbitrary strides,
`seqlen`/`head_dim`/`n_heads`/`ROPE_GROUP_SIZE`/`BLOCK_SIZE` (and any program ids in
`sQ`/`sDY`), the forward surface lowers and both forward half-kernels realize the
genuine rotary specs `ropeFirst/SecondSpec` on `Q`, and symmetrically the backward
surface lowers and both backward half-kernels realize `ropeBackwardFirst/SecondSpec`
on `dY` — under the honest offset-injectivity side conditions. -/
```
</details>

**Statement:**
```lean
specification rope_embedding_forward_backward_summary_general
    (Q dY cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat)
    (sQ sDY : BlockState)
    (hQF : Function.Injective
      (fun i : Fin BLOCK_SIZE => qFirstOffset sQ Q_row_stride head_dim ROPE_GROUP_SIZE i))
    (hQS : Function.Injective
      (fun i : Fin BLOCK_SIZE => qSecondOffset sQ Q_row_stride head_dim ROPE_GROUP_SIZE i))
    (hDF : Function.Injective
      (fun i : Fin BLOCK_SIZE => qFirstOffset sDY Q_row_stride head_dim ROPE_GROUP_SIZE i))
    (hDS : Function.Injective
      (fun i : Fin BLOCK_SIZE => qSecondOffset sDY Q_row_stride head_dim ROPE_GROUP_SIZE i)) :
    (∃ alg, (rope_embedding_surface Q Q_row_stride cos cos_row_stride sin
      sin_row_stride seqlen head_dim n_heads Bool.false BLOCK_SIZE ROPE_GROUP_SIZE).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := rope_embedding_forward_first_half Q cos sin Q_row_stride
        cos_row_stride sin_row_stride seqlen head_dim n_heads ROPE_GROUP_SIZE BLOCK_SIZE)
      (initialState := sQ)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => active sQ head_dim n_heads ROPE_GROUP_SIZE BLOCK_SIZE i)
        (fun i => (Q, qFirstOffset sQ Q_row_stride head_dim ROPE_GROUP_SIZE i)))
      (expected := fun i => ropeFirstSpec sQ Q cos sin Q_row_stride cos_row_stride
        sin_row_stride seqlen head_dim ROPE_GROUP_SIZE BLOCK_SIZE i) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := rope_embedding_forward_second_half Q cos sin Q_row_stride
        cos_row_stride sin_row_stride seqlen head_dim n_heads ROPE_GROUP_SIZE BLOCK_SIZE)
      (initialState := sQ)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => active sQ head_dim n_heads ROPE_GROUP_SIZE BLOCK_SIZE i)
        (fun i => (Q, qSecondOffset sQ Q_row_stride head_dim ROPE_GROUP_SIZE i)))
      (expected := fun i => ropeSecondSpec sQ Q cos sin Q_row_stride cos_row_stride
        sin_row_stride seqlen head_dim ROPE_GROUP_SIZE BLOCK_SIZE i) ∧
    (∃ alg, (rope_embedding_surface dY Q_row_stride cos cos_row_stride sin
      sin_row_stride seqlen head_dim n_heads Bool.true BLOCK_SIZE ROPE_GROUP_SIZE).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := rope_embedding_backward_first_half dY cos sin Q_row_stride
        cos_row_stride sin_row_stride seqlen head_dim n_heads ROPE_GROUP_SIZE BLOCK_SIZE)
      (initialState := sDY)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => active sDY head_dim n_heads ROPE_GROUP_SIZE BLOCK_SIZE i)
        (fun i => (dY, qFirstOffset sDY Q_row_stride head_dim ROPE_GROUP_SIZE i)))
      (expected := fun i => ropeBackwardFirstSpec sDY dY cos sin Q_row_stride
        cos_row_stride sin_row_stride seqlen head_dim ROPE_GROUP_SIZE BLOCK_SIZE i) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := rope_embedding_backward_second_half dY cos sin Q_row_stride
        cos_row_stride sin_row_stride seqlen head_dim n_heads ROPE_GROUP_SIZE BLOCK_SIZE)
      (initialState := sDY)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => active sDY head_dim n_heads ROPE_GROUP_SIZE BLOCK_SIZE i)
        (fun i => (dY, qSecondOffset sDY Q_row_stride head_dim ROPE_GROUP_SIZE i)))
      (expected := fun i => ropeBackwardSecondSpec sDY dY cos sin Q_row_stride
        cos_row_stride sin_row_stride seqlen head_dim ROPE_GROUP_SIZE BLOCK_SIZE i)
```

**Assumptions / layout contracts:**
- `hQF : Function.Injective
      (fun i : Fin BLOCK_SIZE => qFirstOffset sQ Q_row_stride head_dim ROPE_GROUP_SIZE i)`
- `hQS : Function.Injective
      (fun i : Fin BLOCK_SIZE => qSecondOffset sQ Q_row_stride head_dim ROPE_GROUP_SIZE i)`
- `hDF : Function.Injective
      (fun i : Fin BLOCK_SIZE => qFirstOffset sDY Q_row_stride head_dim ROPE_GROUP_SIZE i)`
- `hDS : Function.Injective
      (fun i : Fin BLOCK_SIZE => qSecondOffset sDY Q_row_stride head_dim ROPE_GROUP_SIZE i)`

**Closed-form spec defs (transitive):** `qFirstOffset`, `qSecondOffset`, `rope_embedding_surface`, `rope_embedding_forward_first_half`, `active`, `ropeFirstSpec`, `rope_embedding_forward_second_half`, `ropeSecondSpec`, `rope_embedding_backward_first_half`, `ropeBackwardFirstSpec`, `rope_embedding_backward_second_half`, `ropeBackwardSecondSpec`, `headStart`, `colIndex`, `cosOffset`, `sinOffset`, `rowMod`

<details><summary><code>qFirstOffset</code></summary>

```lean
def qFirstOffset
    (s : BlockState) (Q_row_stride head_dim ROPE_GROUP_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 0 * Q_row_stride + headStart s ROPE_GROUP_SIZE * head_dim + colIndex i
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

<details><summary><code>headStart</code></summary>

```lean
def headStart (s : BlockState) (ROPE_GROUP_SIZE : Nat) : Nat :=
  s.pids 1 * ROPE_GROUP_SIZE
```
</details>

<details><summary><code>colIndex</code></summary>

```lean
def colIndex (i : Fin BLOCK_SIZE) : Nat :=
  i.val
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

## Public theorem: `rope_embedding_io_correctness`

<details><summary>docstring</summary>

```
/-- **The headline on the IO surface** for `rope_embedding.py`'s
`_rope_embedding`, all four slices (forward / backward × first / second half): for
every disjoint flat placement of `Q`/`cos`/`sin`, every program coordinate whose
active lanes are in bounds, and every launch state whose two `Q` windows and the
`cos`/`sin` windows are pinned, each slice terminates, every write-active lane of
`Q` holds the genuine rotary value, and every other memory cell is unchanged.

The four spec arguments are the **pre-state** lane values — the honest reading of
an in-place update. Dimension-general in every stride, `seqlen`, `head_dim`,
`n_heads`, `ROPE_GROUP_SIZE` and `BLOCK_SIZE`, with **no** side-condition: the
closed-form readbacks' output-injectivity precondition is discharged outright by
`qFirst_inj` / `qSecond_inj`. -/
```
</details>

**Statement:**
```lean
specification rope_embedding_io_correctness (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat) :
    (fwd_firstIO Q cos sin Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE
      ⊨ fun _p₀ _p₁ q1 q2 c1 s1 i => ropeFirstSpecOf q1 q2 c1 s1 i) ∧
    (fwd_secondIO Q cos sin Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE
      ⊨ fun _p₀ _p₁ q1 q2 c1 s1 i => ropeSecondSpecOf q1 q2 c1 s1 i) ∧
    (bwd_firstIO Q cos sin Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE
      ⊨ fun _p₀ _p₁ q1 q2 c1 s1 i => ropeBackwardFirstSpecOf q1 q2 c1 s1 i) ∧
    (bwd_secondIO Q cos sin Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE
      ⊨ fun _p₀ _p₁ q1 q2 c1 s1 i => ropeBackwardSecondSpecOf q1 q2 c1 s1 i)
```

**Closed-form spec defs (transitive):** `fwd_firstIO`, `ropeFirstSpecOf`, `fwd_secondIO`, `ropeSecondSpecOf`, `bwd_firstIO`, `ropeBackwardFirstSpecOf`, `bwd_secondIO`, `ropeBackwardSecondSpecOf`, `rope_embedding_forward_first_half`, `rope_embedding_forward_second_half`, `rope_embedding_backward_first_half`, `rope_embedding_backward_second_half`

<details><summary><code>fwd_firstIO</code></summary>

```
/-- IO signature of `rope_embedding_forward_first_half` on the in-place tile surface. -/
```
```lean
def fwd_firstIO (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat) : InPlaceMaskedTileKernelIO where
  kernel := rope_embedding_forward_first_half Q cos sin Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
    ROPE_GROUP_SIZE BLOCK_SIZE
  main := Q
  aux1 := cos
  aux2 := sin
  shape := [BLOCK_SIZE]
  readMain1 := fun p₀ p₁ i =>
    p₀ * Q_row_stride + p₁ * ROPE_GROUP_SIZE * head_dim + i.1.val
  readMain2 := fun p₀ p₁ i =>
    p₀ * Q_row_stride + p₁ * ROPE_GROUP_SIZE * head_dim + i.1.val + head_dim / 2
  readAux1 := fun p₀ _p₁ i =>
    IntegralDType.nat.mod p₀ seqlen * cos_row_stride + i.1.val
  readAux2 := fun p₀ _p₁ i =>
    IntegralDType.nat.mod p₀ seqlen * sin_row_stride + i.1.val
  write := fun p₀ p₁ i =>
    p₀ * Q_row_stride + p₁ * ROPE_GROUP_SIZE * head_dim + i.1.val
  mask := fun _p₀ _p₁ i => i.1.val < head_dim / 2
  writeMask := fun _p₀ p₁ i =>
    i.1.val < head_dim / 2 ∧ p₁ * ROPE_GROUP_SIZE < n_heads
```
</details>

<details><summary><code>ropeFirstSpecOf</code></summary>

```
/-- Value-level spec of `rope_embedding_forward_first_half`, over the loaded values. -/
```
```lean
noncomputable def ropeFirstSpecOf {BLOCK_SIZE : Nat}
    (q1 q2 c1 s1 : TileIndex [BLOCK_SIZE] → ℝ)
    (i : TileIndex [BLOCK_SIZE]) : ℝ :=
  q1 i * c1 i - q2 i * s1 i
```
</details>

<details><summary><code>fwd_secondIO</code></summary>

```
/-- IO signature of `rope_embedding_forward_second_half` on the in-place tile surface. -/
```
```lean
def fwd_secondIO (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat) : InPlaceMaskedTileKernelIO where
  kernel := rope_embedding_forward_second_half Q cos sin Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
    ROPE_GROUP_SIZE BLOCK_SIZE
  main := Q
  aux1 := cos
  aux2 := sin
  shape := [BLOCK_SIZE]
  readMain1 := fun p₀ p₁ i =>
    p₀ * Q_row_stride + p₁ * ROPE_GROUP_SIZE * head_dim + i.1.val
  readMain2 := fun p₀ p₁ i =>
    p₀ * Q_row_stride + p₁ * ROPE_GROUP_SIZE * head_dim + i.1.val + head_dim / 2
  readAux1 := fun p₀ _p₁ i =>
    IntegralDType.nat.mod p₀ seqlen * cos_row_stride + i.1.val
  readAux2 := fun p₀ _p₁ i =>
    IntegralDType.nat.mod p₀ seqlen * sin_row_stride + i.1.val
  write := fun p₀ p₁ i =>
    p₀ * Q_row_stride + p₁ * ROPE_GROUP_SIZE * head_dim + i.1.val + head_dim / 2
  mask := fun _p₀ _p₁ i => i.1.val < head_dim / 2
  writeMask := fun _p₀ p₁ i =>
    i.1.val < head_dim / 2 ∧ p₁ * ROPE_GROUP_SIZE < n_heads
```
</details>

<details><summary><code>ropeSecondSpecOf</code></summary>

```
/-- Value-level spec of `rope_embedding_forward_second_half`, over the loaded values. -/
```
```lean
noncomputable def ropeSecondSpecOf {BLOCK_SIZE : Nat}
    (q1 q2 c1 s1 : TileIndex [BLOCK_SIZE] → ℝ)
    (i : TileIndex [BLOCK_SIZE]) : ℝ :=
  q2 i * c1 i + q1 i * s1 i
```
</details>

<details><summary><code>bwd_firstIO</code></summary>

```
/-- IO signature of `rope_embedding_backward_first_half` on the in-place tile surface. -/
```
```lean
def bwd_firstIO (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat) : InPlaceMaskedTileKernelIO where
  kernel := rope_embedding_backward_first_half Q cos sin Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
    ROPE_GROUP_SIZE BLOCK_SIZE
  main := Q
  aux1 := cos
  aux2 := sin
  shape := [BLOCK_SIZE]
  readMain1 := fun p₀ p₁ i =>
    p₀ * Q_row_stride + p₁ * ROPE_GROUP_SIZE * head_dim + i.1.val
  readMain2 := fun p₀ p₁ i =>
    p₀ * Q_row_stride + p₁ * ROPE_GROUP_SIZE * head_dim + i.1.val + head_dim / 2
  readAux1 := fun p₀ _p₁ i =>
    IntegralDType.nat.mod p₀ seqlen * cos_row_stride + i.1.val
  readAux2 := fun p₀ _p₁ i =>
    IntegralDType.nat.mod p₀ seqlen * sin_row_stride + i.1.val
  write := fun p₀ p₁ i =>
    p₀ * Q_row_stride + p₁ * ROPE_GROUP_SIZE * head_dim + i.1.val
  mask := fun _p₀ _p₁ i => i.1.val < head_dim / 2
  writeMask := fun _p₀ p₁ i =>
    i.1.val < head_dim / 2 ∧ p₁ * ROPE_GROUP_SIZE < n_heads
```
</details>

<details><summary><code>ropeBackwardFirstSpecOf</code></summary>

```
/-- Value-level spec of `rope_embedding_backward_first_half`, over the loaded values. -/
```
```lean
noncomputable def ropeBackwardFirstSpecOf {BLOCK_SIZE : Nat}
    (q1 q2 c1 s1 : TileIndex [BLOCK_SIZE] → ℝ)
    (i : TileIndex [BLOCK_SIZE]) : ℝ :=
  q1 i * c1 i + 0 - q2 i * ((0.0 : ℝ) - s1 i)
```
</details>

<details><summary><code>bwd_secondIO</code></summary>

```
/-- IO signature of `rope_embedding_backward_second_half` on the in-place tile surface. -/
```
```lean
def bwd_secondIO (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat) : InPlaceMaskedTileKernelIO where
  kernel := rope_embedding_backward_second_half Q cos sin Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
    ROPE_GROUP_SIZE BLOCK_SIZE
  main := Q
  aux1 := cos
  aux2 := sin
  shape := [BLOCK_SIZE]
  readMain1 := fun p₀ p₁ i =>
    p₀ * Q_row_stride + p₁ * ROPE_GROUP_SIZE * head_dim + i.1.val
  readMain2 := fun p₀ p₁ i =>
    p₀ * Q_row_stride + p₁ * ROPE_GROUP_SIZE * head_dim + i.1.val + head_dim / 2
  readAux1 := fun p₀ _p₁ i =>
    IntegralDType.nat.mod p₀ seqlen * cos_row_stride + i.1.val
  readAux2 := fun p₀ _p₁ i =>
    IntegralDType.nat.mod p₀ seqlen * sin_row_stride + i.1.val
  write := fun p₀ p₁ i =>
    p₀ * Q_row_stride + p₁ * ROPE_GROUP_SIZE * head_dim + i.1.val + head_dim / 2
  mask := fun _p₀ _p₁ i => i.1.val < head_dim / 2
  writeMask := fun _p₀ p₁ i =>
    i.1.val < head_dim / 2 ∧ p₁ * ROPE_GROUP_SIZE < n_heads
```
</details>

<details><summary><code>ropeBackwardSecondSpecOf</code></summary>

```
/-- Value-level spec of `rope_embedding_backward_second_half`, over the loaded values. -/
```
```lean
noncomputable def ropeBackwardSecondSpecOf {BLOCK_SIZE : Nat}
    (q1 q2 c1 s1 : TileIndex [BLOCK_SIZE] → ℝ)
    (i : TileIndex [BLOCK_SIZE]) : ℝ :=
  q2 i * c1 i - 0 + q1 i * ((0.0 : ℝ) - s1 i)
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

## Also present (pinned special-case summaries)
- `rope_embedding_forward_first_half_compute_correct`
- `rope_embedding_forward_second_half_compute_correct`
- `rope_embedding_backward_first_half_compute_correct`
- `rope_embedding_backward_second_half_compute_correct`
