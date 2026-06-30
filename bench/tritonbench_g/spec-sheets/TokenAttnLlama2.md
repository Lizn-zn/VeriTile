# Spec sheet — `bench/tritonbench_g/token_attn_llama2/TokenAttnLlama2.lean`

**Python source:** `bench/tritonbench_g/token_attn_llama2/token_attn_llama2.py`

## Public theorem: `token_attn_llama2_output_summary_general`

<details><summary>docstring</summary>

```
/-! ### ════════ ★ MAIN THEOREM ★ ════════

**Dimension-general headline.** For *symbolic* head width `BLOCK_DMODEL`, block
size `BLOCK_N`, `sm_scale`, and *all* strides, the full Llama2 token-decode
QK-score surface kernel (1) lowers to the algorithm layer, and (2) realizes the
genuine, self-reference-free QK-dot closed form `tokenAttnLlama2ClosedForm` at
every active output lane of the masked store: each active lane
(`offs_n_new < max_input_len`) of an active `block_mask = 1` block holds the score
`sm_scale · Σ_d q[d]·k[i,d]` (with the `B_Loc` gather and varlen `start_index`
offset folded in), and an inactive lane (or any lane of an inactive block) is
preserved.

Honest side-conditions only: a clean `undef` state `hundef` and output-offset
injectivity `hOutInj`. The headline is dimension-general, so the concrete Python
test shapes are recovered as instances. -/
```
</details>

**Statement:**
```lean
theorem token_attn_llama2_output_summary_general
    (Q K : RegionName) (sm_scale : ℝ)
    (B_Loc B_Start_Loc B_Seqlen : Region .nat) (Att_Out : RegionName)
    (max_input_len stride_b_loc_b stride_b_loc_s stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd att_stride_h att_stride_bs kv_group_num
      BLOCK_DMODEL BLOCK_N : Nat)
    (s : BlockState) (hundef : ∀ rg o, s.undef rg o = 0)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => outOffset s B_Start_Loc att_stride_h att_stride_bs BLOCK_N i)) :
    (∃ alg, (token_attn_llama2_surface Q K sm_scale B_Loc B_Start_Loc B_Seqlen
      Att_Out max_input_len stride_b_loc_b stride_b_loc_s stride_qbs stride_qh
      stride_qd stride_kbs stride_kh stride_kd att_stride_h att_stride_bs
      kv_group_num BLOCK_DMODEL BLOCK_N).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := token_attn_llama2_surface Q K sm_scale B_Loc B_Start_Loc
        B_Seqlen Att_Out max_input_len stride_b_loc_b stride_b_loc_s stride_qbs
        stride_qh stride_qd stride_kbs stride_kh stride_kd att_stride_h
        att_stride_bs kv_group_num BLOCK_DMODEL BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => active s B_Seqlen max_input_len BLOCK_N i)
        (fun i => (Att_Out, outOffset s B_Start_Loc att_stride_h att_stride_bs BLOCK_N i)))
      (expected := fun i : Fin BLOCK_N =>
        tokenAttnLlama2ClosedForm s Q K sm_scale B_Loc B_Start_Loc B_Seqlen Att_Out
          max_input_len stride_b_loc_b stride_b_loc_s stride_qbs stride_qh stride_qd
          stride_kbs stride_kh stride_kd att_stride_h att_stride_bs kv_group_num
          BLOCK_DMODEL BLOCK_N i))
```

**Assumptions / layout contracts:**
- `hundef : ∀ rg o, s.undef rg o = 0`
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => outOffset s B_Start_Loc att_stride_h att_stride_bs BLOCK_N i)`
- `fun i : Fin BLOCK_N => active s B_Seqlen max_input_len BLOCK_N i`

**Closed-form spec defs (transitive):** `outOffset`, `token_attn_llama2_surface`, `active`, `tokenAttnLlama2ClosedForm`, `startLoc`, `blockOffset`, `seqLen`, `blockActive`, `tokenAttnLlama2DotScore`, `qOffset`, `kOffset`, `kLoc`

<details><summary><code>outOffset</code></summary>

```lean
def outOffset
    (s : BlockState) (B_Start_Loc : RegionName)
    (att_stride_h att_stride_bs BLOCK_N : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pids 1 * att_stride_h +
    (startLoc s B_Start_Loc + blockOffset s BLOCK_N i) * att_stride_bs
```
</details>

<details><summary><code>token_attn_llama2_surface</code></summary>

```
/-- Faithful transcription of `token_attn_llama2.py`'s
`_fwd_kernel_token_att1`.

Typed-region note: metadata/gather buffers are `Region .nat`, matching their
index role without adding source-level `dtype=` kwargs. -/
```
```lean
def token_attn_llama2_surface
    (Q K : RegionName) (sm_scale : ℝ) (B_Loc B_Start_Loc B_Seqlen : Region .nat)
    (Att_Out : RegionName)
    (max_input_len stride_b_loc_b stride_b_loc_s stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd att_stride_h att_stride_bs kv_group_num
      BLOCK_DMODEL BLOCK_N : Nat) : ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  start_n = tl.program_id(2)
  cur_kv_head = cur_head // $(kv_group_num)
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
  cur_batch_start_index = $(max_input_len) - cur_batch_seq_len
  cur_batch_end_index = $(max_input_len)
  off_q = cur_batch * $(stride_qbs) + cur_head * $(stride_qh) + offs_d * $(stride_qd)
  offs_n = start_n * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  block_stard_index = start_n * $(BLOCK_N)
  block_mask = tl.where(block_stard_index < cur_batch_seq_len, $(1), $(0))
  for start_mark in range($(0), block_mask, $(1)) {
    q = tl.load(Q + off_q + start_mark)
    offs_n_new = cur_batch_start_index + offs_n
    k_loc = tl.load(B_Loc + $(stride_b_loc_b) * cur_batch +
      $(stride_b_loc_s) * offs_n_new,
      mask=offs_n_new < cur_batch_end_index, other=$(0))
    off_k = k_loc[:, None] * $(stride_kbs) + cur_kv_head * $(stride_kh) +
      offs_d[None, :] * $(stride_kd)
    k = tl.load(K + off_k, mask=offs_n_new[:, None] < cur_batch_end_index, other=0.0)
    att_value = tl.sum(q[None, :] * k, 1)
    att_value *= $((sm_scale : ℝ))
    off_o = cur_head * $(att_stride_h) +
      (cur_batch_in_all_start_index + offs_n) * $(att_stride_bs)
    tl.store(Att_Out + off_o, att_value, mask=offs_n_new < cur_batch_end_index)
  }
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active
    (s : BlockState) (B_Seqlen : RegionName) (max_input_len BLOCK_N : Nat)
    (i : Fin BLOCK_N) : Prop :=
  max_input_len - seqLen s B_Seqlen + blockOffset s BLOCK_N i < max_input_len
```
</details>

<details><summary><code>tokenAttnLlama2ClosedForm</code></summary>

```
/-- Genuine closed-form value written to `Att_Out` for lane `i`. For an active
lane in an active block it is the QK score; otherwise (inactive lane, or an
inactive `block_mask = 0` block in which the kernel stores nothing) the original
`Att_Out` cell at `offset` is preserved. -/
```
```lean
noncomputable def tokenAttnLlama2ClosedForm
    (s : BlockState) (Q K : RegionName) (sm_scale : ℝ)
    (B_Loc B_Start_Loc B_Seqlen : RegionName) (Att_Out : RegionName)
    (max_input_len stride_b_loc_b stride_b_loc_s stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd att_stride_h att_stride_bs kv_group_num
      BLOCK_DMODEL BLOCK_N : Nat) (i : Fin BLOCK_N) : ℝ :=
  if blockActive s B_Seqlen BLOCK_N ∧ active s B_Seqlen max_input_len BLOCK_N i then
    tokenAttnLlama2DotScore s Q K sm_scale B_Loc B_Seqlen max_input_len
      stride_b_loc_b stride_b_loc_s stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd kv_group_num BLOCK_DMODEL BLOCK_N i
  else
    s.readMem Att_Out (outOffset s B_Start_Loc att_stride_h att_stride_bs BLOCK_N i)
```
</details>

<details><summary><code>startLoc</code></summary>

```lean
def startLoc (s : BlockState) (B_Start_Loc : RegionName) : Nat :=
  s.readMemValue .nat B_Start_Loc (s.pids 0)
```
</details>

<details><summary><code>blockOffset</code></summary>

```lean
def blockOffset (s : BlockState) (BLOCK_N : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pids 2 * BLOCK_N + i.val
```
</details>

<details><summary><code>seqLen</code></summary>

```lean
def seqLen (s : BlockState) (B_Seqlen : RegionName) : Nat :=
  s.readMemValue .nat B_Seqlen (s.pids 0)
```
</details>

<details><summary><code>blockActive</code></summary>

```
/-- `block_mask = 1` predicate: the `start_n` block has at least one in-range
key token (`start_n·BLOCK_N < cur_batch_seq_len`). When false the kernel's
`range(0, block_mask, 1)` loop is empty and nothing is stored. -/
```
```lean
def blockActive (s : BlockState) (B_Seqlen : RegionName) (BLOCK_N : Nat) : Prop :=
  s.pids 2 * BLOCK_N < seqLen s B_Seqlen
```
</details>

<details><summary><code>tokenAttnLlama2DotScore</code></summary>

```
/-- The genuine per-lane closed-form QK score for an *active* lane `i` in an
*active* block: `sm_scale · Σ_{d < BLOCK_DMODEL} Q[qOffset d] · K[kOffset i d]`. -/
```
```lean
noncomputable def tokenAttnLlama2DotScore
    (s : BlockState) (Q K : RegionName) (sm_scale : ℝ)
    (B_Loc B_Seqlen : RegionName)
    (max_input_len stride_b_loc_b stride_b_loc_s stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd kv_group_num
      BLOCK_DMODEL BLOCK_N : Nat) (i : Fin BLOCK_N) : ℝ :=
  (∑ d : Fin BLOCK_DMODEL,
      s.readMem Q (qOffset s stride_qbs stride_qh stride_qd d.val) *
      s.readMem K (kOffset s B_Loc B_Seqlen max_input_len stride_b_loc_b stride_b_loc_s
        stride_kbs stride_kh stride_kd kv_group_num BLOCK_N i d.val)) * sm_scale
```
</details>

<details><summary><code>qOffset</code></summary>

```
/-- The query-load offset for head dim `d`: `cur_batch·stride_qbs +
cur_head·stride_qh + d·stride_qd` (the `start_mark = 0` loop index folds away). -/
```
```lean
def qOffset (s : BlockState) (stride_qbs stride_qh stride_qd : Nat) (d : Nat) : Nat :=
  s.pids 0 * stride_qbs + s.pids 1 * stride_qh + d * stride_qd
```
</details>

<details><summary><code>kOffset</code></summary>

```
/-- The key-load offset for lane `i`, head dim `d`:
`k_loc[i]·stride_kbs + cur_kv_head·stride_kh + d·stride_kd`,
with `cur_kv_head = cur_head / kv_group_num`. -/
```
```lean
def kOffset
    (s : BlockState) (B_Loc B_Seqlen : RegionName)
    (max_input_len stride_b_loc_b stride_b_loc_s stride_kbs stride_kh stride_kd
      kv_group_num BLOCK_N : Nat) (i : Fin BLOCK_N) (d : Nat) : Nat :=
  kLoc s B_Loc B_Seqlen max_input_len stride_b_loc_b stride_b_loc_s BLOCK_N i * stride_kbs +
    (s.pids 1 / kv_group_num) * stride_kh + d * stride_kd
```
</details>

<details><summary><code>kLoc</code></summary>

```
/-- The gathered KV page index for lane `i`:
`B_Loc[stride_b_loc_b·cur_batch + stride_b_loc_s·offs_n_new[i]]`. -/
```
```lean
def kLoc
    (s : BlockState) (B_Loc B_Seqlen : RegionName)
    (max_input_len stride_b_loc_b stride_b_loc_s BLOCK_N : Nat) (i : Fin BLOCK_N) : Nat :=
  s.readMemValue .nat B_Loc
    (stride_b_loc_b * s.pids 0 +
      stride_b_loc_s * (max_input_len - seqLen s B_Seqlen + blockOffset s BLOCK_N i))
```
</details>

## Also present (pinned special-case summaries)
- `token_attn_llama2_score_store_slice_compute_correct`
- `token_attn_llama2_closed_form_correct`
- `token_attn_llama2_surface_output_compute_correct`
