# Spec sheet — `bench/tritonbench_g/bgmv_expand_slice/BgmvExpandSlice.lean`

**Python source:** `bench/tritonbench_g/bgmv_expand_slice/bgmv_expand_slice.py`

## Public theorem: `bgmv_full_output_summary`

<details><summary>docstring</summary>

```
/-- **Full output summary** for the general `bgmv_expand_slice` kernel (arbitrary
`split_n_length`, multi-block `for n` loop, signed `-1` sentinel guard): the DSL
surface lowers to the algorithm layer, and the masked GEMV store to the
slice-offset output realizes the genuine rank-`K` reduction
`bgmvFullSpec g = Σ_k (k<K ? A[k] : 0)·(g<split_n_length ∧ k<K ? B[g,k] : 0)` at
every output lane `g < split_n_length`. Requires the active LoRA index
(`lora_index = Int.ofNat li ≥ 0`, so the `-1` early return is not taken),
out-of-place output (`out ≠ input`, `out ≠ lora`), and per-lane output-offset
injectivity. -/
```
</details>

**Statement:**
```lean
theorem bgmv_full_output_summary
    (input_ptr lora_ptr out_ptr : RegionName) (lora_indices : Region .int)
    (li K split_n_length xm_stride xk_stride l0_stride lora_k_stride
      lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K : Nat)
    (s : BlockState) (hBN : 0 < BLOCK_N)
    (hoi : out_ptr ≠ input_ptr) (hol : out_ptr ≠ lora_ptr)
    (hcn : 0 < cn_stride)
    (hlx : s.readMemValue .int (Region.cast lora_indices) (s.pids 1) = Int.ofNat li) :
    (∃ alg, (bgmv_full input_ptr lora_ptr out_ptr lora_indices K split_n_length xm_stride xk_stride
        l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := bgmv_full input_ptr lora_ptr out_ptr lora_indices K split_n_length xm_stride xk_stride
        l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin split_n_length => True)
        (fun g => (out_ptr, cOff s split_n_length cm_stride cn_stride slice_offset g.val)))
      (expected := fun g : Fin split_n_length =>
        bgmvFullSpec s input_ptr lora_ptr li K split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride BLOCK_K g.val)
```

**Assumptions / layout contracts:**
- `hBN : 0 < BLOCK_N`
- `hoi : out_ptr ≠ input_ptr`
- `hol : out_ptr ≠ lora_ptr`
- `hcn : 0 < cn_stride`
- `hlx : s.readMemValue .int (Region.cast lora_indices) (s.pids 1) = Int.ofNat li`
- `fun _ : Fin split_n_length => True`

**Closed-form spec defs (transitive):** `bgmv_full`, `cOff`, `bgmvFullSpec`, `prodGK`, `aElem`, `bElem`

<details><summary><code>bgmv_full</code></summary>

```lean
def bgmv_full
    (input_ptr lora_ptr out_ptr : RegionName) (lora_indices : Region .int)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride
      lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K : Nat) :
    ComputeKernel := triton {
  pid_sn = tl.program_id(axis=0)
  cur_batch = tl.program_id(axis=1)
  lora_index = tl.load(lora_indices + cur_batch)
  if lora_index != $((-1 : Int)) {
    offset_k = tl.arange(0, $(BLOCK_K))
    offset_n = tl.arange(0, $(BLOCK_N))
    tiled_a = tl.load(input_ptr + cur_batch * $(xm_stride) + offset_k * $(xk_stride),
      mask=offset_k < $(K), other=0)
    b_ptr = lora_ptr + $(l0_stride) * lora_index +
      pid_sn * $(split_n_length) * $(lora_k_stride)
    c_ptr = out_ptr + cur_batch * $(cm_stride) + pid_sn * $(split_n_length) +
      $(slice_offset) * $(cn_stride)
    for n in range($(0), $(split_n_length), $(BLOCK_N)) {
      current_n = n + offset_n
      b_ptr_mask = (current_n[:, None] < $(split_n_length)) & (offset_k[None, :] < $(K))
      c_mask = current_n < $(split_n_length)
      tiled_b = tl.load(
        b_ptr + current_n[:, None] * $(lora_k_stride) +
          offset_k[None, :] * $(lora_n_stride),
        mask=b_ptr_mask, other=0.0)
      accumulator = tl.sum(tiled_a * tiled_b, 1)
      tl.store(c_ptr + current_n * $(cn_stride), accumulator, mask=c_mask)
    }
  }
}
```
</details>

<details><summary><code>cOff</code></summary>

```lean
def cOff (s0 : BlockState) (split_n_length cm_stride cn_stride slice_offset : Nat) (g : Nat) : Nat :=
  s0.pids 1 * cm_stride + s0.pids 0 * split_n_length + slice_offset * cn_stride + g * cn_stride

-- masked product for global lane g, key k (over Fin BLOCK_K)
```
</details>

<details><summary><code>bgmvFullSpec</code></summary>

```lean
noncomputable def bgmvFullSpec (s0 : BlockState) (input_ptr lora_ptr : RegionName) (li : Nat)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride BLOCK_K : Nat) (g : Nat) : ℝ :=
  ∑ k : Fin BLOCK_K, prodGK s0 input_ptr lora_ptr li K split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride g k.val

set_option maxHeartbeats 4000000
set_option maxRecDepth 8000
set_option linter.unusedSimpArgs false

-- matmul-style expandDim helpers
```
</details>

<details><summary><code>prodGK</code></summary>

```lean
noncomputable def prodGK (s0 : BlockState) (input_ptr lora_ptr : RegionName) (li : Nat)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride : Nat) (g k : Nat) : ℝ :=
  (if k < K then aElem s0 input_ptr xm_stride xk_stride k else 0) *
  (if g < split_n_length ∧ k < K then bElem s0 lora_ptr li split_n_length l0_stride lora_k_stride lora_n_stride g k else 0)

-- full spec for global lane g : sum over k : Fin BLOCK_K of prodGK
```
</details>

<details><summary><code>aElem</code></summary>

```lean
noncomputable def aElem (s0 : BlockState) (input_ptr : RegionName) (xm_stride xk_stride : Nat) (k : Nat) : ℝ :=
  s0.readMem input_ptr (s0.pids 1 * xm_stride + k * xk_stride)

-- read lora B[g,k] for global lane g
```
</details>

<details><summary><code>bElem</code></summary>

```lean
noncomputable def bElem (s0 : BlockState) (lora_ptr : RegionName) (li : Nat)
    (split_n_length l0_stride lora_k_stride lora_n_stride : Nat) (g k : Nat) : ℝ :=
  s0.readMem lora_ptr (l0_stride * li + s0.pids 0 * split_n_length * lora_k_stride + g * lora_k_stride + k * lora_n_stride)

-- output offset for global lane g (relative to out_ptr region)
```
</details>

## Also present (pinned special-case summaries)
- `bgmv_full_compute_correct`
