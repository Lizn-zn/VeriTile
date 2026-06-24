# Spec sheet — `bench/tritonbench_g/token_attn_reduceV/TokenAttnReduceV.lean`

**Python source:** `bench/tritonbench_g/token_attn_reduceV/token_attn_reduceV.py`

## Public theorem: `token_attn_reducev_output_summary_general`

<details><summary>docstring</summary>

```
/-! ### ════════ ★ MAIN THEOREM ★ ════════

`token_attn_reducev_output_summary_general` is the headline: a **single
dimension-general** statement over symbolic `(BLOCK_DMODEL, BLOCK_N)` and all
strides. It bundles the surface→algorithm lowering with the genuine
PV-reduction compute-correctness (`tokenAttnReduceVClosedForm`), under honest
side-conditions only:

* `0 < BLOCK_DMODEL`, `0 < BLOCK_N` — non-degenerate output / loop block;
* `stride_pbs = 1`, `stride_req_to_tokens_s = 1` — the contiguous `Prob` /
  `Req_to_tokens` layouts the kernel's per-lane address arithmetic genuinely
  assumes;
* `hOutInj` — output-offset injectivity (no aliasing across the store);
* `hundef` — clean input (`undef = 0`).

No shape-specific numeric cheats; `expected` is the self-reference-free closed
form over input memory, never the kernel's executed value. The Python test
cases below are thin corollaries instantiating this at `BLOCK_DMODEL = 64`,
`BLOCK_N = 128` and the checked `stride_req_to_tokens_b` gather-stride
(`128` for cases 1/3, `64` for case 2). That stride feeds `vLoc`/`vMasked`, so
it genuinely enters the PV-reduction value spec; the cases are distinct
instantiations, not a single collapsed one. -/
```
</details>

**Statement:**
```lean
theorem token_attn_reducev_output_summary_general
    (Prob V Out : RegionName)
    (Req_to_tokens B_req_idx B_Start_Loc B_Seqlen : Region .nat)
    (stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      kv_group_num BLOCK_DMODEL BLOCK_N : Nat)
    (hBD : 0 < BLOCK_DMODEL) (hBN : 0 < BLOCK_N)
    (hpbs : stride_pbs = 1) (hrts : stride_req_to_tokens_s = 1)
    (s : BlockState) (hundef : ∀ rg o, s.undef rg o = 0)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_DMODEL => outOffset s stride_obs stride_oh stride_od i)) :
    (∃ alg, (token_attn_reducev_surface Prob V Out Req_to_tokens B_req_idx
      B_Start_Loc B_Seqlen stride_req_to_tokens_b stride_req_to_tokens_s stride_ph
      stride_pbs stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      kv_group_num BLOCK_DMODEL BLOCK_N).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := token_attn_reducev_surface Prob V Out Req_to_tokens B_req_idx
        B_Start_Loc B_Seqlen stride_req_to_tokens_b stride_req_to_tokens_s stride_ph
        stride_pbs stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
        kv_group_num BLOCK_DMODEL BLOCK_N)
      (initialState := s)
      (write := fun i : Fin BLOCK_DMODEL =>
        some (Out, outOffset s stride_obs stride_oh stride_od i))
      (expected := fun i : Fin BLOCK_DMODEL =>
        tokenAttnReduceVClosedForm s Prob V Req_to_tokens B_req_idx
          B_Start_Loc B_Seqlen stride_req_to_tokens_b stride_req_to_tokens_s stride_ph
          stride_pbs stride_vbs stride_vh stride_vd kv_group_num BLOCK_DMODEL i))
```

**Assumptions / layout contracts:**
- `hBD : 0 < BLOCK_DMODEL`
- `hBN : 0 < BLOCK_N`
- `hpbs : stride_pbs = 1`
- `hrts : stride_req_to_tokens_s = 1`
- `hundef : ∀ rg o, s.undef rg o = 0`
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_DMODEL => outOffset s stride_obs stride_oh stride_od i)`

**Closed-form spec defs (transitive):** `outOffset`, `token_attn_reducev_surface`, `tokenAttnReduceVClosedForm`, `dIndex`, `tokenAttnReduceVPVValue`, `batchSeqLen`, `pOffset`, `vOffset`, `inAllStartLoc`, `vLoc`, `reqIdx`

<details><summary><code>outOffset</code></summary>

```lean
def outOffset
    (s : BlockState) (stride_obs stride_oh stride_od : Nat)
    (i : Fin BLOCK_DMODEL) : Nat :=
  s.pids 0 * stride_obs + s.pids 1 * stride_oh + dIndex s i * stride_od
```
</details>

<details><summary><code>token_attn_reducev_surface</code></summary>

```
/-- Faithful transcription of `token_attn_reduceV.py`'s
`_fwd_kernel_token_att2`.

Typed-region note: metadata/gather buffers are `Region .nat`, matching their
index role without adding source-level `dtype=` kwargs. -/
```
```lean
def token_attn_reducev_surface
    (Prob V Out : RegionName)
    (Req_to_tokens B_req_idx B_Start_Loc B_Seqlen : Region .nat)
    (stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      kv_group_num BLOCK_DMODEL BLOCK_N : Nat) : ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  cur_kv_head = cur_head // $(kv_group_num)
  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
  cur_batch_start_index = 0
  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
  cur_batch_req_idx = tl.load(B_req_idx + cur_batch)
  v_loc_off = cur_batch_req_idx * $(stride_req_to_tokens_b) +
    (cur_batch_start_index + offs_n) * $(stride_req_to_tokens_s)
  p_offs = cur_head * $(stride_ph) +
    (cur_batch_in_all_start_index + offs_n) * $(stride_pbs)
  v_offs = cur_kv_head * $(stride_vh) + offs_d[None, :] * $(stride_vd)
  acc = tl.zeros([$(BLOCK_DMODEL)], dtype=tl.float32)
  for start_n in range($(0), cur_batch_seq_len, $(BLOCK_N)) {
    start_n = tl.multiple_of(start_n, $(BLOCK_N))
    p_value = tl.load(Prob + p_offs + start_n,
      mask=(start_n + offs_n) < cur_batch_seq_len, other=0.0)
    v_loc = tl.load(Req_to_tokens + v_loc_off +
      start_n * $(stride_req_to_tokens_s),
      mask=(start_n + offs_n) < cur_batch_seq_len, other=0.0)
    v_value = tl.load(V + v_offs + v_loc[:, None] * $(stride_vbs),
      mask=(start_n + offs_n[:, None]) < cur_batch_seq_len, other=0.0)
    acc += tl.sum(p_value[:, None] * v_value, 0)
  }
  acc = (acc).to(Out.dtype.element_ty)
  off_o = cur_batch * $(stride_obs) + cur_head * $(stride_oh) + offs_d * $(stride_od)
  out_ptrs = Out + off_o
  tl.store(out_ptrs, acc)
}
```
</details>

<details><summary><code>tokenAttnReduceVClosedForm</code></summary>

```
/-- Genuine closed-form value written to `Out[outOffset d]`. The store is
unmasked over the full `[BLOCK_DMODEL]` vector, so every lane holds the
PV-accumulator `tokenAttnReduceVPVValue` for its head-dim `d = dIndex s i`. -/
```
```lean
noncomputable def tokenAttnReduceVClosedForm
    (s : BlockState) (Prob V : RegionName)
    (Req_to_tokens B_req_idx B_Start_Loc B_Seqlen : RegionName)
    (stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
      stride_vbs stride_vh stride_vd kv_group_num BLOCK_DMODEL : Nat)
    (i : Fin BLOCK_DMODEL) : ℝ :=
  tokenAttnReduceVPVValue s Prob V Req_to_tokens B_req_idx B_Start_Loc
    B_Seqlen stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
    stride_vbs stride_vh stride_vd kv_group_num (dIndex s i)
```
</details>

<details><summary><code>dIndex</code></summary>

```lean
def dIndex (_s : BlockState) (i : Fin BLOCK_DMODEL) : Nat :=
  i.val
```
</details>

<details><summary><code>tokenAttnReduceVPVValue</code></summary>

```
/-- The genuine closed-form accumulator for output head-dim `d`:
`Σ_{n < cur_batch_seq_len} p[n] · v[v_loc[n], d]`. The sum range is exactly the
masked token window, so no `if`-guard is needed — every padding lane
(`n ≥ cur_batch_seq_len`) is excluded from `Finset.range` and contributes `0`,
mirroring the `other = 0` masked loads. -/
```
```lean
noncomputable def tokenAttnReduceVPVValue
    (s : BlockState) (Prob V : RegionName)
    (Req_to_tokens B_req_idx B_Start_Loc B_Seqlen : RegionName)
    (stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
      stride_vbs stride_vh stride_vd kv_group_num : Nat)
    (d : Nat) : ℝ :=
  ∑ n ∈ Finset.range (batchSeqLen s B_Seqlen),
    s.readMem Prob (pOffset s B_Start_Loc stride_ph stride_pbs n) *
      s.readMem V (vOffset s Req_to_tokens B_req_idx
        stride_req_to_tokens_b stride_req_to_tokens_s stride_vbs stride_vh
        stride_vd kv_group_num n d)
```
</details>

<details><summary><code>batchSeqLen</code></summary>

```
/-- `cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)`: the loop bound and the
mask threshold for every per-token load. -/
```
```lean
def batchSeqLen (s : BlockState) (B_Seqlen : RegionName) : Nat :=
  s.readMemValue .nat B_Seqlen (s.pids 0)
```
</details>

<details><summary><code>pOffset</code></summary>

```
/-- Per-token probability load offset:
`cur_head·stride_ph + (in_all_start_index + n)·stride_pbs`. -/
```
```lean
def pOffset
    (s : BlockState) (B_Start_Loc : RegionName)
    (stride_ph stride_pbs : Nat) (n : Nat) : Nat :=
  s.pids 1 * stride_ph + (inAllStartLoc s B_Start_Loc + n) * stride_pbs
```
</details>

<details><summary><code>vOffset</code></summary>

```
/-- Value-row load offset for token `n`, head-dim `d`:
`v_loc[n]·stride_vbs + cur_kv_head·stride_vh + d·stride_vd`, with
`cur_kv_head = cur_head / kv_group_num`. -/
```
```lean
def vOffset
    (s : BlockState) (Req_to_tokens B_req_idx : RegionName)
    (stride_req_to_tokens_b stride_req_to_tokens_s stride_vbs stride_vh stride_vd
      kv_group_num : Nat) (n d : Nat) : Nat :=
  vLoc s Req_to_tokens B_req_idx stride_req_to_tokens_b
      stride_req_to_tokens_s n * stride_vbs +
    (s.pids 1 / kv_group_num) * stride_vh + d * stride_vd
```
</details>

<details><summary><code>inAllStartLoc</code></summary>

```
/-- `cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)`: the
flattened start offset folded into the `Prob` load address. -/
```
```lean
def inAllStartLoc (s : BlockState) (B_Start_Loc : RegionName) : Nat :=
  s.readMemValue .nat B_Start_Loc (s.pids 0)
```
</details>

<details><summary><code>vLoc</code></summary>

```
/-- Gathered KV page index for token `n` (`cur_batch_start_index = 0`):
`Req_to_tokens[req_idx·stride_req_to_tokens_b + n·stride_req_to_tokens_s]`. -/
```
```lean
def vLoc
    (s : BlockState) (Req_to_tokens B_req_idx : RegionName)
    (stride_req_to_tokens_b stride_req_to_tokens_s : Nat)
    (n : Nat) : Nat :=
  s.readMemValue .nat Req_to_tokens
    (reqIdx s B_req_idx * stride_req_to_tokens_b +
      n * stride_req_to_tokens_s)
```
</details>

<details><summary><code>reqIdx</code></summary>

```
/-- `cur_batch_req_idx = tl.load(B_req_idx + cur_batch)`: the request row used to
index `Req_to_tokens`. -/
```
```lean
def reqIdx (s : BlockState) (B_req_idx : RegionName) : Nat :=
  s.readMemValue .nat B_req_idx (s.pids 0)
```
</details>

## Also present (pinned special-case summaries)
- `token_attn_reducev_final_store_slice_compute_correct`
- `token_attn_reducev_final_store_python_test_shape_compute_correct`
- `token_attn_reducev_closed_form_correct`
- `token_attn_reducev_closed_form_compute_correct`
- `token_attn_reducev_python_case1_output_surface_summary`
- `token_attn_reducev_python_case2_output_surface_summary`
- `token_attn_reducev_python_case3_output_surface_summary`
- `token_attn_reducev_python_case1_output_summary`
- `token_attn_reducev_python_case2_output_summary`
- `token_attn_reducev_python_case3_output_summary`
