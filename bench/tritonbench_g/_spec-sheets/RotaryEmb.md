# Spec sheet — `bench/tritonbench_g/rotary_emb/RotaryEmb.lean`

**Python source:** `bench/tritonbench_g/rotary_emb/rotary_emb.py`

## Public theorem: `rotary_emb_output_summary_general`

<details><summary>docstring</summary>

```
/-- **Dimension-general public Python summary for `rotary_emb_fwd`.**
For arbitrary symbolic strides,
sequence length, head counts, and block sizes, the full `_rotary_kernel` surface
lowers to the algorithm layer and each of the four Python-observable stores
(Q even/odd, K even/odd) reads back, on every active lane, to the genuine
interleaved rotary closed form (`rotary{Q0,Q1,K0,K1}Spec`). The general
block-level building blocks (`rotary_emb_q0/q1/k0/k1_block_compute_correct`)
consume within-family offset injectivity, taken here as hypotheses. -/
```
</details>

**Statement:**
```lean
theorem rotary_emb_output_summary_general
    (Q K Cos Sin : RegionName)
    (stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd
      stride_cosbs stride_cosd stride_sinbs stride_sind max_total_len
      HEAD_Q HEAD_K BLOCK_HEAD BLOCK_SEQ BLOCK_DMODEL BLOCK_HALF : Nat)
    (s : BlockState)
    (hQEven : Function.Injective
      (fun i : Fin BLOCK_HALF => qOffset s stride_qbs stride_qh stride_qd (dimEven i)))
    (hQOdd : Function.Injective
      (fun i : Fin BLOCK_HALF => qOffset s stride_qbs stride_qh stride_qd (dimOdd i)))
    (hKEven : Function.Injective
      (fun i : Fin BLOCK_HALF => kOffset s stride_kbs stride_kh stride_kd (dimEven i)))
    (hKOdd : Function.Injective
      (fun i : Fin BLOCK_HALF => kOffset s stride_kbs stride_kh stride_kd (dimOdd i))) :
    (∃ alg, (rotary_kernel_surface Q K Cos Sin stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_Q HEAD_K BLOCK_HEAD BLOCK_SEQ
      BLOCK_DMODEL).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel
```

**Assumptions / layout contracts:**
- `fun i : Fin BLOCK_HALF => qOffset s stride_qbs stride_qh stride_qd (dimEven i)`
- `fun i : Fin BLOCK_HALF => qOffset s stride_qbs stride_qh stride_qd (dimOdd i)`
- `fun i : Fin BLOCK_HALF => kOffset s stride_kbs stride_kh stride_kd (dimEven i)`
- `fun i : Fin BLOCK_HALF => kOffset s stride_kbs stride_kh stride_kd (dimOdd i)`

**Closed-form spec defs (transitive):** `qOffset`, `dimEven`, `dimOdd`, `kOffset`, `rotary_kernel_surface`, `seqIndex`, `headIndex`

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

<details><summary><code>dimOdd</code></summary>

```lean
def dimOdd (i : Fin BLOCK_HALF) : Nat :=
  i.val * 2 + 1
```
</details>

<details><summary><code>kOffset</code></summary>

```lean
def kOffset
    (s : BlockState) (stride_kbs stride_kh stride_kd : Nat) (dim : Nat) : Nat :=
  seqIndex s * stride_kbs + headIndex s * stride_kh + dim * stride_kd
```
</details>

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

## Also present (pinned special-case summaries)
- `rotary_emb_q0_block_compute_correct`
- `rotary_emb_q1_block_compute_correct`
- `rotary_emb_k0_block_compute_correct`
- `rotary_emb_k1_block_compute_correct`
