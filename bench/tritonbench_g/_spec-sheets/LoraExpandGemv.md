# Spec sheet — `bench/tritonbench_g/lora_expand_gemv/LoraExpandGemv.lean`

**Python source:** `bench/tritonbench_g/lora_expand_gemv/lora_expand_gemv.py`

## Public theorem: `gemv_full_output_summary`

<details><summary>docstring</summary>

```
/-- **Full output summary**: the full LoRA expand GEMV surface lowers to the
algorithm layer, and the masked store realizes the genuine matrix-vector product
`out[m] = Σ_{k<K} x[k]·W[m,k]` at every active global lane `m < split_n_length`
(general `⌈split_n_length/BLOCK_N⌉`-block loop). -/
```
</details>

**Statement:**
```lean
specification gemv_full_output_summary
    (input_ptr lora_ptr out_ptr : RegionName) (lora_indices : Region .nat)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride BLOCK_N BLOCK_K : Nat)
    (s : BlockState) (hBN : 0 < BLOCK_N) (hKB : K ≤ BLOCK_K) (hol : out_ptr ≠ lora_ptr)
    (hundef : ∀ rg o, s.undef rg o = 0)
    (hcn : 0 < cn_stride) :
    (∃ alg, (bgmv_loop_surface input_ptr lora_ptr out_ptr lora_indices K
        split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
        cm_stride cn_stride BLOCK_N BLOCK_K).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel
```

**Assumptions / layout contracts:**
- `hBN : 0 < BLOCK_N`
- `hKB : K ≤ BLOCK_K`
- `hol : out_ptr ≠ lora_ptr`
- `hundef : ∀ rg o, s.undef rg o = 0`
- `hcn : 0 < cn_stride`

**Closed-form spec defs (transitive):** `bgmv_loop_surface`

<details><summary><code>bgmv_loop_surface</code></summary>

```
/-- The full loop surface: signed-sentinel guard elided (the host only launches
the active body when `lora_index ≠ -1`), `RegionName`-typed `lora_indices` so the
selected base is read back via `readMemValue .nat`, `ADD_INPUTS = false`,
`CAST_TYPE = false`, `EVEN_K = false` (masked input load). The `for n` loop runs
`⌈split_n_length / BLOCK_N⌉` blocks. -/
```
```lean
def bgmv_loop_surface
    (input_ptr lora_ptr out_ptr : RegionName) (lora_indices : Region .nat)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride
      lora_n_stride cm_stride cn_stride BLOCK_N BLOCK_K : Nat) :
    ComputeKernel := triton {
  pid_sn = tl.program_id(axis=0)
  cur_batch = tl.program_id(axis=1)
  lora_index = tl.load(lora_indices + cur_batch)
  offset_k = tl.arange(0, $(BLOCK_K))
  offset_n = tl.arange(0, $(BLOCK_N))
  tiled_a = tl.load(input_ptr + cur_batch * $(xm_stride) + offset_k * $(xk_stride),
    mask=offset_k < $(K), other=0.0)
  b_ptr = lora_ptr + $(l0_stride) * lora_index +
    pid_sn * $(split_n_length) * $(lora_k_stride)
  c_ptr = out_ptr + cur_batch * $(cm_stride) + pid_sn * $(split_n_length)
  for n in range($(0), $(split_n_length), $(BLOCK_N)) {
    current_n = n + offset_n
    tiled_b = tl.load(
      b_ptr + current_n[:, None] * $(lora_k_stride) +
        offset_k[None, :] * $(lora_n_stride),
      mask=(current_n[:, None] < $(split_n_length)) and (offset_k[None, :] < $(K)),
      other=0.0)
    accumulator = tl.sum(tiled_a * tiled_b, 1)
    tl.store(c_ptr + current_n * $(cn_stride), accumulator,
      mask=current_n < $(split_n_length))
  }
}
```
</details>

## Also present (pinned special-case summaries)
- `gemv_compute_correct`
