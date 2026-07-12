# Spec sheet — `bench/tritonbench_g/fast_rope_embedding/FastRopeEmbedding.lean`

**Python source:** `bench/tritonbench_g/fast_rope_embedding/fast_rope_embedding.py`

## Public theorem: `rope_embedding_output_summary_general`

<details><summary>docstring</summary>

```
/-- Dimension-general public summary for `fast_rope_embedding`.

Symbolic in every shape parameter (`seqlen`, `head_dim`, `n_heads`,
`BLOCK_SIZE`, and the `Q`/`cos`/`sin` row strides) and in the regions, this is
the genuine compute-correctness statement that the pinned Python-test-shape
summary is merely one instance of:

* the forward surfaces for `Q` and `K` lower to the algorithm layer,
* both forward half stores for `Q` and for `K` realize the rotary writeback
  against `ropeFirstSpec` / `ropeSecondSpec` (pure functions of input memory),
* the backward gradient surfaces for `QGrad` and `KGrad` lower.

The only side conditions are honest store-offset injectivity hypotheses (the
per-region first/second-half offsets are injective on the active lanes); these
are exactly what the readback lemmas need and hold for any contiguous,
in-range layout. `BLOCK_SIZE > 0` is *not* required as a hypothesis: it is
discharged internally from the lane index. -/
```
</details>

**Statement:**
```lean
specification rope_embedding_output_summary_general
    (Q K QGrad KGrad cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      BLOCK_SIZE : Nat)
    (sQ sK : BlockState)
    (hQFirstInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => qFirstOffset sQ Q_row_stride head_dim i))
    (hQSecondInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => qSecondOffset sQ Q_row_stride head_dim i))
    (hKFirstInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => qFirstOffset sK Q_row_stride head_dim i))
    (hKSecondInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => qSecondOffset sK Q_row_stride head_dim i)) :
    (∃ alg, (rope_embedding_surface Q cos sin Q_row_stride cos_row_stride
      sin_row_stride seqlen head_dim n_heads BLOCK_SIZE
      Bool.false).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := rope_embedding_q_first_half Q cos sin Q_row_stride
        cos_row_stride sin_row_stride seqlen head_dim n_heads BLOCK_SIZE)
      (initialState := sQ)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => active sQ head_dim n_heads BLOCK_SIZE i)
        (fun i => (Q, qFirstOffset sQ Q_row_stride head_dim i)))
      (expected := fun i =>
        ropeFirstSpec sQ Q cos sin Q_row_stride cos_row_stride sin_row_stride
          seqlen head_dim BLOCK_SIZE i)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := rope_embedding_q_second_half Q cos sin Q_row_stride
        cos_row_stride sin_row_stride seqlen head_dim n_heads BLOCK_SIZE)
      (initialState := sQ)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => active sQ head_dim n_heads BLOCK_SIZE i)
        (fun i => (Q, qSecondOffset sQ Q_row_stride head_dim i)))
      (expected := fun i =>
        ropeSecondSpec sQ Q cos sin Q_row_stride cos_row_stride sin_row_stride
          seqlen head_dim BLOCK_SIZE i)) ∧
    (∃ alg, (rope_embedding_surface K cos sin Q_row_stride cos_row_stride
      sin_row_stride seqlen head_dim n_heads BLOCK_SIZE
      Bool.false).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := rope_embedding_q_first_half K cos sin Q_row_stride
        cos_row_stride sin_row_stride seqlen head_dim n_heads BLOCK_SIZE)
      (initialState := sK)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => active sK head_dim n_heads BLOCK_SIZE i)
        (fun i => (K, qFirstOffset sK Q_row_stride head_dim i)))
      (expected := fun i =>
        ropeFirstSpec sK K cos sin Q_row_stride cos_row_stride sin_row_stride
          seqlen head_dim BLOCK_SIZE i)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := rope_embedding_q_second_half K cos sin Q_row_stride
        cos_row_stride sin_row_stride seqlen head_dim n_heads BLOCK_SIZE)
      (initialState := sK)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => active sK head_dim n_heads BLOCK_SIZE i)
        (fun i => (K, qSecondOffset sK Q_row_stride head_dim i)))
      (expected := fun i =>
        ropeSecondSpec sK K cos sin Q_row_stride cos_row_stride sin_row_stride
          seqlen head_dim BLOCK_SIZE i)) ∧
    (∃ alg, (rope_embedding_surface QGrad cos sin Q_row_stride cos_row_stride
      sin_row_stride seqlen head_dim n_heads BLOCK_SIZE
      Bool.true).toAlgorithm? = Except.ok alg) ∧
    (∃ alg, (rope_embedding_surface KGrad cos sin Q_row_stride cos_row_stride
      sin_row_stride seqlen head_dim n_heads BLOCK_SIZE
      Bool.true).toAlgorithm? = Except.ok alg)
```

**Assumptions / layout contracts:**
- `hQFirstInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => qFirstOffset sQ Q_row_stride head_dim i)`
- `hQSecondInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => qSecondOffset sQ Q_row_stride head_dim i)`
- `hKFirstInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => qFirstOffset sK Q_row_stride head_dim i)`
- `hKSecondInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => qSecondOffset sK Q_row_stride head_dim i)`
- `fun i : Fin BLOCK_SIZE => active sQ head_dim n_heads BLOCK_SIZE i`
- `fun i : Fin BLOCK_SIZE => active sQ head_dim n_heads BLOCK_SIZE i`
- `fun i : Fin BLOCK_SIZE => active sK head_dim n_heads BLOCK_SIZE i`
- `fun i : Fin BLOCK_SIZE => active sK head_dim n_heads BLOCK_SIZE i`

**Closed-form spec defs (transitive):** `qFirstOffset`, `qSecondOffset`, `rope_embedding_surface`, `rope_embedding_q_first_half`, `active`, `ropeFirstSpec`, `rope_embedding_q_second_half`, `ropeSecondSpec`, `headStart`, `colIndex`, `cosOffset`, `sinOffset`, `rowMod`

<details><summary><code>qFirstOffset</code></summary>

```lean
def qFirstOffset
    (s : BlockState) (Q_row_stride head_dim : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 0 * Q_row_stride + headStart s * head_dim + colIndex i
```
</details>

<details><summary><code>qSecondOffset</code></summary>

```lean
def qSecondOffset
    (s : BlockState) (Q_row_stride head_dim : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 0 * Q_row_stride + headStart s * head_dim + colIndex i + head_dim / 2
```
</details>

<details><summary><code>rope_embedding_surface</code></summary>

```
/-- Faithful transcription of `fast_rope_embedding.py`'s `_rope_embedding`.

The body preserves the fixed four-head group loop, the `BACKWARD_PASS` case,
and both rotary-pair stores. -/
```
```lean
def rope_embedding_surface
    (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      BLOCK_SIZE : Nat) (BACKWARD_PASS : Bool) :
    ComputeKernel := triton {
  ROPE_GROUP_SIZE = $(4)
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
  head_start = group_head_position * ROPE_GROUP_SIZE
  head_end = min(head_start + ROPE_GROUP_SIZE, $(n_heads))
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

<details><summary><code>rope_embedding_q_first_half</code></summary>

```
/-- Proof-oriented forward first-half slice of `fast_rope_embedding.py`'s
`_rope_embedding`.

The full kernel loops over a small group of heads and writes both halves of the
rotary pair. This slice captures one head's first-half update:
`Q1 * cos - Q2 * sin`. -/
```
```lean
def rope_embedding_q_first_half
    (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      BLOCK_SIZE : Nat) :
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
  head_start = group_head_position * $(4)
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
def active (s : BlockState) (head_dim n_heads BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : Prop :=
  colIndex i < head_dim / 2 ∧ headStart s < n_heads
```
</details>

<details><summary><code>ropeFirstSpec</code></summary>

```lean
noncomputable def ropeFirstSpec
    (s : BlockState) (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : ℝ :=
  s.readMem Q (qFirstOffset s Q_row_stride head_dim i) *
    s.readMem cos (cosOffset s seqlen cos_row_stride i) -
  s.readMem Q (qSecondOffset s Q_row_stride head_dim i) *
    s.readMem sin (sinOffset s seqlen sin_row_stride i)
```
</details>

<details><summary><code>rope_embedding_q_second_half</code></summary>

```
/-- Proof-oriented second-half slice of `_rope_embedding`. Captures the
companion second-half writeback `out = Q2 * cos + Q1 * sin` to offset
`offs_q2`. -/
```
```lean
def rope_embedding_q_second_half
    (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      BLOCK_SIZE : Nat) :
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
  head_start = group_head_position * $(4)
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
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : ℝ :=
  s.readMem Q (qSecondOffset s Q_row_stride head_dim i) *
    s.readMem cos (cosOffset s seqlen cos_row_stride i) +
  s.readMem Q (qFirstOffset s Q_row_stride head_dim i) *
    s.readMem sin (sinOffset s seqlen sin_row_stride i)
```
</details>

<details><summary><code>headStart</code></summary>

```lean
def headStart (s : BlockState) : Nat :=
  s.pids 1 * 4
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

## Also present (pinned special-case summaries)
- `rope_embedding_q_first_half_compute_correct`
- `rope_embedding_q_second_half_compute_correct`
