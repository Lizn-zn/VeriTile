# Spec sheet — `bench/tritonbench_g/attention_fwd_triton2/AttentionFwdTriton2.lean`

**Python source:** `bench/tritonbench_g/attention_fwd_triton2/attention_fwd_triton2.py`

## Public theorem: `attention_fwd_triton2_output_summary_general`

<details><summary>docstring</summary>

```
/-- **Dimension-general output summary for `attention_fwd_triton2` (no test-shape pin).**

Mirrors the reference `attention_forward_triton_closed_form_correct`: over
*symbolic* batch/head strides, head count `H`, block sizes `BLOCK_M`/`BLOCK_N`,
KV-block count `numKVBlocks` (so `N_CTX = BLOCK_N · numKVBlocks`), head/active
dimensions and arbitrary `q_scale`/`k_scale`, this combines

* the checked full-surface lowering to the algorithm layer
  (`attention_fwd_triton2_surface_toAlgorithm_supported`), and
* the genuine closed-form value of every active `Out` lane
  (`attention_fwd_triton2_closed_form_correct`):
  `attentionRealBase2PerKeyScale` of the loaded Q/K/V tiles under the per-block
  key scale — the base-2, per-key-scaled attention output reading INPUT Q/K/V
  memory, NOT the kernel's own executed value.

The only layout assumptions are the contiguity contracts the kernel relies on
(`stride_qm = stride_kn = HEAD_DIM`, head stride `1`), `0 < BLOCK_N`,
`HEAD_ACTIVE ≤ BLOCK_DMODEL`, `HEAD_ACTIVE ≤ HEAD_DIM`, and a clean initial
`undef`. The Python test case (`B=2, H=4, N_CTX=128, HEAD_DIM=128, BLOCK_M=128,
BLOCK_N=64, HEAD_ACTIVE=96, numKVBlocks=2`) is the special case. -/
```
</details>

**Statement:**
```lean
specification attention_fwd_triton2_output_summary_general
    (Q K V Q_scale K_scale Out : RegionName) (s : BlockState)
    (stride_qz stride_qh Z H BLOCK_M BLOCK_N numKVBlocks
      HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE STAGE : Nat)
    (hBN : 0 < BLOCK_N) (hActiveLe : HEAD_ACTIVE ≤ BLOCK_DMODEL)
    (hHD : HEAD_ACTIVE ≤ HEAD_DIM) (hundef : ∀ rg o, s.undef rg o = 0) :
    (∃ alg, (attention_fwd_triton2_surface Q K V Q_scale K_scale Out
      stride_qz stride_qh HEAD_DIM 1
      stride_qz stride_qh HEAD_DIM 1
      stride_qz stride_qh HEAD_DIM 1
      stride_qz stride_qh HEAD_DIM 1
      Z H (BLOCK_N * numKVBlocks) HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL
      HEAD_ACTIVE STAGE).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel
```

**Assumptions / layout contracts:**
- `hBN : 0 < BLOCK_N`
- `hActiveLe : HEAD_ACTIVE ≤ BLOCK_DMODEL`
- `hHD : HEAD_ACTIVE ≤ HEAD_DIM`
- `hundef : ∀ rg o, s.undef rg o = 0`

**Closed-form spec defs (transitive):** `attention_fwd_triton2_surface`, `cdiv`

<details><summary><code>attention_fwd_triton2_surface</code></summary>

```
/-- Full Lean port of `attention_fwd_triton2.py`'s `_attn_fwd`.

The upstream kernel calls a separate `@triton.jit` helper `_attn_fwd_inner` to
run the K/V streaming-softmax loop. The DSL has no function-call surface, so the
helper body is inlined verbatim into the outer kernel; semantically the two
forms are identical for this fixed-stage path. The upstream `v.to(tl.float16)`
dot-input cast and the `bfloat16` output cast erase to the identity over `ℝ`, so
this surface is the same inlined online-softmax loop verified for
`attention_forward_triton`.

The literal `128` and `96` in the upstream kernel correspond to the
`BLOCK_DMODEL` / `HEAD_ACTIVE` parameters threaded through the bundled tests
(`head_dim = 128`, with the inner dot using only the first 96 lanes of the head
dimension). They appear here as explicit Lean parameters. -/
```
```lean
def attention_fwd_triton2_surface
    (Q K V Q_scale K_scale Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      _stride_kz _stride_kh stride_kn _stride_kk
      _stride_vz _stride_vh _stride_vk _stride_vn
      _stride_oz _stride_oh _stride_om _stride_on
      _Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE _STAGE : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)

  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  qvk_offset = (off_z).to(tl.int64) * $(stride_qz) + (off_h).to(tl.int64) * $(stride_qh)
  vk_offset = qvk_offset // $(stride_qm)
  q_scale_offset = off_hz * tl.cdiv($(N_CTX), $(BLOCK_M))
  k_scale_offset = off_hz * tl.cdiv($(N_CTX), $(BLOCK_N))

  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_n = tl.arange(0, $(BLOCK_N))
  offs_k = tl.arange(0, $(BLOCK_DMODEL))
  Q_ptrs = Q + qvk_offset + offs_m[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk)
  Q_scale_ptr = Q_scale + q_scale_offset + start_m
  K_ptrs = K + qvk_offset + offs_k[:, None] + offs_n[None, :] * $(stride_kn)
  K_scale_ptr = K_scale + k_scale_offset
  V_ptrs = V + qvk_offset + offs_n[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk)
  O_block_ptr = Out + qvk_offset + offs_m[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk)
  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) + 1.0
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  q = tl.load(Q_ptrs,
    mask=(offs_m[:, None] < $(N_CTX)) & (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[None, :])
  q_scale = tl.load(Q_scale_ptr)
  for start_n in range(0, $(N_CTX), $(BLOCK_N)) {
    start_n = tl.multiple_of(start_n, $(BLOCK_N))
    k_mask = (offs_n[None, :] < ($(N_CTX) - start_n)) &
      (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[:, None]
    k = tl.load(K_ptrs, mask=k_mask)
    k_scale = tl.load(K_scale_ptr)
    qk = (tl.dot(q, k)).to(tl.float32) * q_scale * k_scale
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk = qk - m_ij[:, None]
    p = tl.math.exp2(qk)
    l_ij = tl.sum(p, 1)
    alpha = tl.math.exp2(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    v = tl.load(V_ptrs,
      mask=(offs_n[:, None] < ($(N_CTX) - start_n)) &
        (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[None, :])
    p = (p).to(tl.float16)
    acc += tl.dot(p, v, out_dtype=tl.float16)
    m_i = m_ij
    K_ptrs += $(BLOCK_N) * $(HEAD_DIM)
    K_scale_ptr += $(1)
    V_ptrs += $(BLOCK_N) * $(HEAD_DIM)
  }
  acc = acc / l_i[:, None]
  tl.store(O_block_ptr, (acc).to(Out.type.element_ty),
    mask=(offs_m[:, None] < $(N_CTX)) & (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[None, :])
}
```
</details>

<details><summary><code>cdiv</code></summary>

```
/-- Ceiling division `⌈a / b⌉`, matching Triton's `tl.cdiv`. -/
```
```lean
def cdiv (a b : Nat) : Nat := (a + b - 1) / b
```
</details>

## Also present (pinned special-case summaries)
- `attention_fwd_triton2_final_store_slice_compute_correct`
- `attention_fwd_triton2_closed_form_correct`
