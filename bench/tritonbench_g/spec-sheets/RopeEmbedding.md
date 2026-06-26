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
on `dY` — under the honest offset-injectivity side conditions. The pinned
`..._python_case{1,2,3,4}_*` summaries are concrete instantiations of this. -/
```
</details>

**Statement:**
```lean
theorem rope_embedding_forward_backward_summary_general
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
    ComputeCorrect.Realizes
      (kernel
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

**Closed-form spec defs (transitive):** `qFirstOffset`, `qSecondOffset`, `rope_embedding_surface`, `headStart`, `colIndex`

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
- `rope_embedding_python_case1_forward_backward_summary`
- `rope_embedding_python_case2_forward_backward_summary`
- `rope_embedding_python_case3_forward_backward_summary`
- `rope_embedding_python_case4_forward_backward_summary`
