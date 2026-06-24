# Spec sheet — `bench/tritonbench_g/token_attn_mistral/TokenAttnMistral.lean`

**Python source:** `bench/tritonbench_g/token_attn_mistral/token_attn_mistral.py`

## Public theorem: `token_attn_mistral_output_summary_general`

<details><summary>docstring</summary>

```
/-! ### ════════ ★ MAIN THEOREM ★ ════════

**Dimension-general headline.** For *symbolic* output width `BLOCK_DMODEL`, block
size `BLOCK_N`, sliding window `sliding_window`, and *all* strides, the full
sliding-window reduce-V surface kernel (1) lowers to the algorithm layer, and (2)
realizes the genuine, self-reference-free PV-reduction closed form
`tokenAttnMistralClosedForm` (which incorporates `sliding_window` symbolically via
`startIndex`/`vActive`) at every `[BLOCK_DMODEL]` output lane.

Honest side-conditions only: `0 < BLOCK_N`, the contiguous
layout hyps `stride_pbs = 1` / `stride_req_to_tokens_s = 1` (faithful to the
checked test's contiguous `Prob`/`Req_to_tokens`), output-offset injectivity
`hOutInj`, and a clean `undef` state `hundef`. The four `python_case*` theorems
below are thin corollaries instantiating the concrete (dims, sliding_window,
stride_req_to_tokens_b). -/
```
</details>

**Statement:**
```lean
theorem token_attn_mistral_output_summary_general
    (Prob V Out : RegionName)
    (Req_to_tokens B_req_idx : Region .nat) (B_Start_Loc : RegionName)
    (B_Seqlen B_Att_Start_Loc B_Att_Seqlen : Region .nat)
    (stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      kv_group_num sliding_window BLOCK_DMODEL BLOCK_N : Nat)
    (hpbs : stride_pbs = 1) (hrts : stride_req_to_tokens_s = 1) (hBN : 0 < BLOCK_N)
    (s : BlockState) (hundef : ∀ rg o, s.undef rg o = 0)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_DMODEL => outOffset s stride_obs stride_oh stride_od i)) :
    (∃ alg, (token_attn_mistral_surface Prob V Out Req_to_tokens B_req_idx
      B_Start_Loc B_Seqlen B_Att_Start_Loc B_Att_Seqlen
      stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      kv_group_num sliding_window BLOCK_DMODEL BLOCK_N).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := token_attn_mistral_surface Prob V Out Req_to_tokens B_req_idx
        B_Start_Loc B_Seqlen B_Att_Start_Loc B_Att_Seqlen
        stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
        stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
        kv_group_num sliding_window BLOCK_DMODEL BLOCK_N)
      (initialState := s)
      (write := fun i : Fin BLOCK_DMODEL =>
        some (Out, outOffset s stride_obs stride_oh stride_od i))
      (expected := fun i : Fin BLOCK_DMODEL =>
        tokenAttnMistralClosedForm s Prob V Req_to_tokens B_req_idx B_Att_Start_Loc
          B_Seqlen B_Att_Seqlen stride_req_to_tokens_b stride_req_to_tokens_s
          stride_ph stride_pbs stride_vbs stride_vh stride_vd kv_group_num
          sliding_window BLOCK_DMODEL i))
```

**Assumptions / layout contracts:**
- `hpbs : stride_pbs = 1`
- `hrts : stride_req_to_tokens_s = 1`
- `hBN : 0 < BLOCK_N`
- `hundef : ∀ rg o, s.undef rg o = 0`
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_DMODEL => outOffset s stride_obs stride_oh stride_od i)`
- `kernel : = token_attn_mistral_surface Prob V Out Req_to_tokens B_req_idx
        B_Start_Loc B_Seqlen B_Att_Start_Loc B_Att_Seqlen
        stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
        stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
        kv_group_num sliding_window BLOCK_DMODEL BLOCK_N`
- `initialState : = s`
- `write : = fun i : Fin BLOCK_DMODEL =>
        some (Out, outOffset s stride_obs stride_oh stride_od i)`
- `expected : = fun i : Fin BLOCK_DMODEL =>
        tokenAttnMistralClosedForm s Prob V Req_to_tokens B_req_idx B_Att_Start_Loc
          B_Seqlen B_Att_Seqlen stride_req_to_tokens_b stride_req_to_tokens_s
          stride_ph stride_pbs stride_vbs stride_vh stride_vd kv_group_num
          sliding_window BLOCK_DMODEL i`

**Closed-form spec defs (transitive):** `outOffset`, `token_attn_mistral_surface`, `tokenAttnMistralClosedForm`, `dIndex`, `tokenAttnMistralPVValue`, `attSeqLen`, `pOffset`, `vActive`, `vOffset`, `attStartLoc`, `startIndex`, `batchSeqLen`, `vLoc`, `reqIdx`

<details><summary><code>outOffset</code></summary>

```lean
def outOffset
    (s : BlockState) (stride_obs stride_oh stride_od : Nat)
    (i : Fin BLOCK_DMODEL) : Nat :=
  s.pids 0 * stride_obs + s.pids 1 * stride_oh + dIndex s i * stride_od
```
</details>

<details><summary><code>token_attn_mistral_surface</code></summary>

```
/-- Faithful transcription of `token_attn_mistral.py`'s
`_fwd_kernel_token_att2`.

Typed-region note: metadata/gather buffers are `Region .nat`, matching their
index role without adding source-level `dtype=` kwargs. -/
```
```lean
def token_attn_mistral_surface
    (Prob V Out : RegionName)
    (Req_to_tokens B_req_idx : Region .nat) (_B_Start_Loc : RegionName)
    (B_Seqlen B_Att_Start_Loc B_Att_Seqlen : Region .nat)
    (stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      kv_group_num sliding_window BLOCK_DMODEL BLOCK_N : Nat) : ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  cur_kv_head = cur_head // $(kv_group_num)
  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
  cur_batch_start_index = tl.maximum(cur_batch_seq_len - $(sliding_window), $(0))
  cur_batch_in_all_start_index = tl.load(B_Att_Start_Loc + cur_batch)
  cur_batch_req_idx = tl.load(B_req_idx + cur_batch)
  cur_att_seq_len = tl.load(B_Att_Seqlen + cur_batch)
  v_loc_off = cur_batch_req_idx * $(stride_req_to_tokens_b) +
    (cur_batch_start_index + offs_n) * $(stride_req_to_tokens_s)
  p_offs = cur_head * $(stride_ph) +
    (cur_batch_in_all_start_index + offs_n) * $(stride_pbs)
  v_offs = cur_kv_head * $(stride_vh) + offs_d[None, :] * $(stride_vd)
  acc = tl.zeros([$(BLOCK_DMODEL)], dtype=tl.float32)
  for start_n in range($(0), cur_att_seq_len, $(BLOCK_N)) {
    start_n = tl.multiple_of(start_n, $(BLOCK_N))
    p_value = tl.load(Prob + p_offs + start_n,
      mask=(start_n + offs_n) < cur_att_seq_len, other=0.0)
    v_loc = tl.load(Req_to_tokens + v_loc_off +
      start_n * $(stride_req_to_tokens_s),
      mask=(start_n + offs_n + cur_batch_start_index) < cur_batch_seq_len,
      other=0.0)
    v_value = tl.load(V + v_offs + v_loc[:, None] * $(stride_vbs),
      mask=(start_n + offs_n[:, None] + cur_batch_start_index) < cur_batch_seq_len,
      other=0.0)
    acc += tl.sum(p_value[:, None] * v_value, 0)
  }
  acc = (acc).to(Out.dtype.element_ty)
  off_o = cur_batch * $(stride_obs) + cur_head * $(stride_oh) + offs_d * $(stride_od)
  out_ptrs = Out + off_o
  tl.store(out_ptrs, acc)
}
```
</details>

<details><summary><code>tokenAttnMistralClosedForm</code></summary>

```
/-- Genuine closed-form value written to `Out[outOffset d]`. The store is
unmasked over the full `[BLOCK_DMODEL]` vector, so every lane holds the
PV-accumulator `tokenAttnMistralPVValue` for its head-dim `d = dIndex s i`. -/
```
```lean
noncomputable def tokenAttnMistralClosedForm
    (s : BlockState) (Prob V : RegionName)
    (Req_to_tokens B_req_idx B_Att_Start_Loc B_Seqlen B_Att_Seqlen : RegionName)
    (stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
      stride_vbs stride_vh stride_vd kv_group_num sliding_window BLOCK_DMODEL : Nat)
    (i : Fin BLOCK_DMODEL) : ℝ :=
  tokenAttnMistralPVValue s Prob V Req_to_tokens B_req_idx B_Att_Start_Loc
    B_Seqlen B_Att_Seqlen stride_req_to_tokens_b stride_req_to_tokens_s
    stride_ph stride_pbs stride_vbs stride_vh stride_vd kv_group_num
    sliding_window (dIndex s i)
```
</details>

<details><summary><code>dIndex</code></summary>

```lean
def dIndex (_s : BlockState) (i : Fin BLOCK_DMODEL) : Nat :=
  i.val
```
</details>

<details><summary><code>tokenAttnMistralPVValue</code></summary>

```
/-- The genuine closed-form accumulator for output head-dim `d`:
`Σ_{n < cur_att_seq_len} p[n] · v[v_loc[n], d]`, where an out-of-window token
(`¬ vActive`) contributes `0` because its masked `V` load reads `0`. -/
```
```lean
noncomputable def tokenAttnMistralPVValue
    (s : BlockState) (Prob V : RegionName)
    (Req_to_tokens B_req_idx B_Att_Start_Loc B_Seqlen B_Att_Seqlen : RegionName)
    (stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
      stride_vbs stride_vh stride_vd kv_group_num sliding_window : Nat)
    (d : Nat) : ℝ :=
  ∑ n ∈ Finset.range (attSeqLen s B_Att_Seqlen),
    s.readMem Prob (pOffset s B_Att_Start_Loc stride_ph stride_pbs n) *
      (if vActive s B_Seqlen sliding_window n then
        s.readMem V (vOffset s Req_to_tokens B_req_idx B_Seqlen
          stride_req_to_tokens_b stride_req_to_tokens_s stride_vbs stride_vh
          stride_vd kv_group_num sliding_window n d)
      else 0.0)
```
</details>

<details><summary><code>attSeqLen</code></summary>

```
/-! ## Genuine closed-form PV-reduction spec

The Triton `_fwd_kernel_token_att2` accumulates, per output head-dim `d`, the
sliding-window probability-weighted value sum

```
acc[d] = Σ_{n < cur_att_seq_len}  p[n] · v[v_loc[n], d]
```

where, with `start_index = max(cur_batch_seq_len - sliding_window, 0)` and
`cur_kv_head = cur_head / kv_group_num`:

* the per-token probability
  `p[n] = Prob[cur_head·stride_ph + (att_start_loc + n)·stride_pbs]`, masked by
  `n < cur_att_seq_len` (`other = 0`);
* the gathered KV page index
  `v_loc[n] = Req_to_tokens[req_idx·stride_req_to_tokens_b +
    (start_index + n)·stride_req_to_tokens_s]`, masked by
  `start_index + n < cur_batch_seq_len`;
* the value row
  `v[v_loc[n], d] = V[cur_kv_head·stride_vh + d·stride_vd +
    v_loc[n]·stride_vbs]`, masked by `start_index + n < cur_batch_seq_len`
  (`other = 0`), so an out-of-window token contributes `0` to every `d`.

The final store of the whole `[BLOCK_DMODEL]` accumulator is **unmasked**, with a
`.to(Out.dtype)` cast that is the identity at the algorithm (ℝ) layer.

These definitions are a *genuine closed form* — they never execute the kernel —
and (via `token_attn_mistral_closed_form_correct`) fully replace the former
self-referential surface-value spec.

`attSeqLen`/`batchSeqLen`/`reqIdx`/`attStartLoc` are the metadata loads of the
prelude; `startIndex` is the sliding-window left edge. -/
```
```lean
def attSeqLen (s : BlockState) (B_Att_Seqlen : RegionName) : Nat :=
  s.readMemValue .nat B_Att_Seqlen (s.pids 0)
```
</details>

<details><summary><code>pOffset</code></summary>

```
/-- Per-token probability load offset:
`cur_head·stride_ph + (att_start_loc + n)·stride_pbs`. -/
```
```lean
def pOffset
    (s : BlockState) (B_Att_Start_Loc : RegionName)
    (stride_ph stride_pbs : Nat) (n : Nat) : Nat :=
  s.pids 1 * stride_ph + (attStartLoc s B_Att_Start_Loc + n) * stride_pbs
```
</details>

<details><summary><code>vActive</code></summary>

```
/-- A window token `n` whose V-gather is in range: `start_index + n <
cur_batch_seq_len`. Out-of-range tokens read `0` from both the `Req_to_tokens`
gather and the masked `V` load, hence contribute `0`. -/
```
```lean
def vActive
    (s : BlockState) (B_Seqlen : RegionName) (sliding_window : Nat) (n : Nat) : Prop :=
  startIndex s B_Seqlen sliding_window + n < batchSeqLen s B_Seqlen
```
</details>

<details><summary><code>vOffset</code></summary>

```
/-- Value-row load offset for window token `n`, head-dim `d`:
`v_loc[n]·stride_vbs + cur_kv_head·stride_vh + d·stride_vd`, with
`cur_kv_head = cur_head / kv_group_num`. -/
```
```lean
def vOffset
    (s : BlockState) (Req_to_tokens B_req_idx B_Seqlen : RegionName)
    (stride_req_to_tokens_b stride_req_to_tokens_s stride_vbs stride_vh stride_vd
      kv_group_num sliding_window : Nat) (n d : Nat) : Nat :=
  vLoc s Req_to_tokens B_req_idx B_Seqlen stride_req_to_tokens_b
      stride_req_to_tokens_s sliding_window n * stride_vbs +
    (s.pids 1 / kv_group_num) * stride_vh + d * stride_vd
```
</details>

<details><summary><code>attStartLoc</code></summary>

```lean
def attStartLoc (s : BlockState) (B_Att_Start_Loc : RegionName) : Nat :=
  s.readMemValue .nat B_Att_Start_Loc (s.pids 0)
```
</details>

<details><summary><code>startIndex</code></summary>

```
/-- Sliding-window left edge `cur_batch_start_index = max(cur_batch_seq_len -
sliding_window, 0)`. The Nat subtraction already truncates at `0`, so the
`tl.maximum(·, 0)` is the identity. -/
```
```lean
def startIndex (s : BlockState) (B_Seqlen : RegionName) (sliding_window : Nat) : Nat :=
  batchSeqLen s B_Seqlen - sliding_window
```
</details>

<details><summary><code>batchSeqLen</code></summary>

```lean
def batchSeqLen (s : BlockState) (B_Seqlen : RegionName) : Nat :=
  s.readMemValue .nat B_Seqlen (s.pids 0)
```
</details>

<details><summary><code>vLoc</code></summary>

```
/-- Gathered KV page index for window token `n`:
`Req_to_tokens[req_idx·stride_req_to_tokens_b +
  (start_index + n)·stride_req_to_tokens_s]`. -/
```
```lean
def vLoc
    (s : BlockState) (Req_to_tokens B_req_idx B_Seqlen : RegionName)
    (stride_req_to_tokens_b stride_req_to_tokens_s sliding_window : Nat)
    (n : Nat) : Nat :=
  s.readMemValue .nat Req_to_tokens
    (reqIdx s B_req_idx * stride_req_to_tokens_b +
      (startIndex s B_Seqlen sliding_window + n) * stride_req_to_tokens_s)
```
</details>

<details><summary><code>reqIdx</code></summary>

```lean
def reqIdx (s : BlockState) (B_req_idx : RegionName) : Nat :=
  s.readMemValue .nat B_req_idx (s.pids 0)
```
</details>

## Also present (pinned special-case summaries)
- `token_attn_mistral_final_store_slice_compute_correct`
- `token_attn_mistral_final_store_python_test_shape_compute_correct`
- `token_attn_mistral_closed_form_correct`
- `token_attn_mistral_closed_form_compute_correct`
- `token_attn_mistral_python_case1_output_surface_summary`
- `token_attn_mistral_python_case2_output_surface_summary`
- `token_attn_mistral_python_case3_output_surface_summary`
- `token_attn_mistral_python_case4_output_surface_summary`
- `token_attn_mistral_python_case1_output_summary`
- `token_attn_mistral_python_case2_output_summary`
- `token_attn_mistral_python_case3_output_summary`
- `token_attn_mistral_python_case4_output_summary`
