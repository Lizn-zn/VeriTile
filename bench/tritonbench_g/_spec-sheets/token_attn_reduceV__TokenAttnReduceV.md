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
specification token_attn_reducev_output_summary_general
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
    (ComputeCorrect.Realizes_without_Rounding
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

## Public theorem: `token_attn_reducev_io_correctness`

<details><summary>docstring</summary>

```
/-- **The `⊨[R]` gather-skin headline** — `token_attn_reduceV` on
`StreamMetaGatherMasked3DKernelIO₂`, at fully symbolic per-axis strides. For
every rounding model `R`, the faithful surface implements, on its
gather-indexed signature, the streamed closed form
`tokenAttnReduceVIOSpec`: every output lane `d` holds
`Σ_{n < m 0} p[n] · v[v_loc[n], d]` over the batch's live tokens, read off
the two pinned streams. The kernel has **zero rounding events** (three `.nat`
slot loads, a `.nat` page-table gather, `other = 0`-defaulted `.real` loads,
`.real` in-loop arithmetic, and an `acc.to(Out.dtype.element_ty)` that lowers
to a self-assign, not an `Op.castFloat`), so the skin's boundary quantization
degenerates: the readback's `R.round .real` is the identity by the model's
defining `round_real`.

**The gather channel.** `Req_to_tokens` enters as the skin's index channel
(`gty = .nat`, `gother = 0`), and the `V` window `read2` eats the gathered
tile: `G t jL · stride_vbs + …`. Unlike the softmax_reducev exemplar, the
Python `V` load carries **the gather's own mask**
(`mask2 = gmask` on the row coordinate — this kernel has a *single* window
predicate `t·BLOCK_N + jL < m 0`, because `cur_batch_start_index = 0`), so a
masked-off lane never dereferences the substituted `other=` address and this
port needs **no hypothesis at all** on `gother`; only the gather pin's
*active* leg is used.

**Launch legality (`pre` = the trusted-launch boundary).** The surface takes
no `max_input_len`-style host argument, so the skin's pid-free step budget
`T` cannot be derived from the kernel's own parameters: `T` is a **new
io-level parameter** and the triple is guarded by
`io.pre = (m 0 ≤ T · BLOCK_N)`, i.e. the host promises that the dynamic trip
count `cur_batch_seq_len` fits the budget. This is a **disclosed launch
restriction** — the honest cost of putting a data-dependent trip count on a
fixed-`T` streaming skin — not a derived fact, and the `⊨[R]` triple says
nothing about launches outside it.

**Hypothesis provenance**: `stride_pbs = 1` and `stride_req_to_tokens_s = 1`
are the exact headline's contiguous-layout side conditions (the per-lane
address arithmetic `p_offs + start_n` / `v_loc_off + start_n·stride_s` folds
into the closed form's `(base + n)·stride` only at unit stride); `0 < BLOCK_N`
is the exact headline's nonempty-block condition (and what makes the step
budget citable); `hOutInj` restates the exact headline's **open**
output-offset injectivity side condition in ∀-pids form (per-axis strides are
symbolic, so no contiguity discharge is available). The exact headline's
`hundef` is **not** a hypothesis: the skin's Hoare triple carries the `undef`
pin itself, and `0 < BLOCK_DMODEL` is not needed here (nothing in the fold
reduces over an empty head-dim tile).

Relation to the exact surface: the `Realizes_without_Rounding` headline
`token_attn_reducev_output_summary_general` above is retained unchanged; this
`⊨[R]` face restates the same genuine closed form on the gather skin, for
every `R` at once. -/
```
</details>

**Statement:**
```lean
specification token_attn_reducev_io_correctness (R : RoundingModel)
    (Prob V Out : RegionName)
    (Req_to_tokens B_req_idx B_Start_Loc B_Seqlen : Region .nat)
    (srtb srts sph spbs svbs svh svd sobs soh sod kvg BLOCK_DMODEL BLOCK_N T : Nat)
    (hpbs : spbs = 1) (hrts : srts = 1) (hBN : 0 < BLOCK_N)
    (hOutInj : ∀ pid₀ pid₁ : Nat, Function.Injective
      (fun i : Fin BLOCK_DMODEL => pid₀ * sobs + pid₁ * soh + i.val * sod)) :
    tokenAttnReduceVIO Prob V Out Req_to_tokens B_req_idx B_Start_Loc B_Seqlen srtb srts
        sph spbs svbs svh svd sobs soh sod kvg BLOCK_DMODEL BLOCK_N T ⊨[R]
      fun _ _ _ m xs ys j =>
        tokenAttnReduceVIOSpec BLOCK_N BLOCK_DMODEL T (m (⟨0, by omega⟩ : Fin 3))
          hBN xs ys j
```

**Assumptions / layout contracts:**
- `hpbs : spbs = 1`
- `hrts : srts = 1`
- `hBN : 0 < BLOCK_N`
- `hOutInj : ∀ pid₀ pid₁ : Nat, Function.Injective
      (fun i : Fin BLOCK_DMODEL => pid₀ * sobs + pid₁ * soh + i.val * sod)`

**Closed-form spec defs (transitive):** `tokenAttnReduceVIO`, `tokenAttnReduceVIOSpec`, `token_attn_reducev_surface`, `rvIOMetaBuf`, `rvIOprob`, `rvIOval`

<details><summary><code>tokenAttnReduceVIO</code></summary>

```
/-- **Gather-indexed IO signature** of `token_attn_reduceV` on the
gather-indexed two-stream fold skin (S1: PV-accumulation fold + terminal
store, 2-D pid grid `(cur_batch, cur_head)`), at fully **symbolic per-axis
strides**.

Windows transcribe the kernel's pointer arithmetic VERBATIM, with the loaded
slot vector `m` in place of the in-state metadata reads:

* `gread` (`Req_to_tokens`, the page table): lane `jL` of step `t` reads
  `m 2 · srtb + (t·BN + jL) · srts` (the `cur_batch_start_index = 0` base
  contributes nothing); `gmask` is `t·BN + jL < m 0` and `gother = 0`.
* `read1` (`Prob`): lane `jL` of step `t` reads
  `pid₁ · sph + (m 1 + (t·BN + jL)) · spbs`, masked by the same predicate.
* `read2` (`V`, the **gather-addressed** value rows, lane `j = (jL, d)`
  row-major over `[BLOCK_N, BLOCK_DMODEL]`) reads
  `G t jL · svbs + (pid₁ / kvg) · svh + d · svd`. `mask2` repeats the
  gather's predicate on the row coordinate — the Python `V` load carries
  exactly the gather's mask, so dead lanes are never dereferenced.
* `write` (`Out`, the terminal store): lane `i` writes
  `pid₀ · sobs + pid₁ · soh + i · sod`, `writeMask ≡ True` (the store is
  unmasked over the whole `[BLOCK_DMODEL]` vector).

`pre` is the launch-legality field: `m 0 ≤ T · BLOCK_N`. The surface takes no
`max_input_len` argument, so the pid-free step budget `T` cannot be derived
from a host parameter; it is a **new io-level parameter** and `pre` is the
honest disclosure of the launch restriction it imposes.

`outDType` is the `.real` default: the terminal `tl.store` is untyped and the
`acc.to(Out.dtype.element_ty)` cast lowers to a **self-assign**
(`reducevPostlude`'s first statement is `Op.ref .real [BD] "acc"`, not an
`Op.castFloat`), so there is no quantization event anywhere in the port. -/
```
```lean
def tokenAttnReduceVIO (Prob V Out : RegionName)
    (Req_to_tokens B_req_idx B_Start_Loc B_Seqlen : Region .nat)
    (srtb srts sph spbs svbs svh svd sobs soh sod kvg BLOCK_DMODEL BLOCK_N T : Nat) :
    StreamMetaGatherMasked3DKernelIO₂ where
  kernel := token_attn_reducev_surface Prob V Out Req_to_tokens B_req_idx B_Start_Loc
    B_Seqlen srtb srts sph spbs svbs svh svd sobs soh sod kvg BLOCK_DMODEL BLOCK_N
  inp1 := Prob
  inp2 := V
  out := Out
  nMeta := 3
  sty := fun _ => ChanTy.nat
  mbuf := rvIOMetaBuf B_Seqlen B_Start_Loc B_req_idx
  mwin := fun _ pid₀ _ _ => pid₀
  gbuf := Req_to_tokens.cast
  gty := ChanTy.nat
  Bg := BLOCK_N
  gother := 0
  T := T
  B1 := BLOCK_N
  B2 := BLOCK_N * BLOCK_DMODEL
  C := BLOCK_DMODEL
  outDType := .real
  pre := fun _ _ _ m => m (⟨0, by omega⟩ : Fin 3) ≤ T * BLOCK_N
  gread := fun _ _ _ m t jL =>
    m (⟨2, by omega⟩ : Fin 3) * srtb + (t.val * BLOCK_N + jL.val) * srts
  gmask := fun _ _ _ m t jL => t.val * BLOCK_N + jL.val < m (⟨0, by omega⟩ : Fin 3)
  read1 := fun _ pid₁ _ m t jL =>
    pid₁ * sph + (m (⟨1, by omega⟩ : Fin 3) + (t.val * BLOCK_N + jL.val)) * spbs
  mask1 := fun _ _ _ m t jL => t.val * BLOCK_N + jL.val < m (⟨0, by omega⟩ : Fin 3)
  read2 := fun _ pid₁ _ _ G t j =>
    G t (Lane2D.decode j).1 * svbs + (pid₁ / kvg) * svh
      + (Lane2D.decode j).2.1.val * svd
  mask2 := fun _ _ _ m t j =>
    t.val * BLOCK_N + (Lane2D.decode j).1.val < m (⟨0, by omega⟩ : Fin 3)
  write := fun pid₀ pid₁ _ _ i => pid₀ * sobs + pid₁ * soh + i.val * sod
  writeMask := fun _ _ _ _ _ => True
```
</details>

<details><summary><code>tokenAttnReduceVIOSpec</code></summary>

```
/-- **The streamed closed form**: `tokenAttnReduceVPVValue` restated over the
two streamed tiles — `out[d] = Σ_{n < m 0} p[n] · v[v_loc[n], d]`. The sum
range is exactly the live-token window, so no per-term guard is needed. -/
```
```lean
noncomputable def tokenAttnReduceVIOSpec (BLOCK_N BLOCK_DMODEL T S : Nat)
    (hBN : 0 < BLOCK_N) (xs : Fin T → Fin BLOCK_N → ℝ)
    (ys : Fin T → Fin (BLOCK_N * BLOCK_DMODEL) → ℝ) (d : Fin BLOCK_DMODEL) : ℝ :=
  ∑ n ∈ Finset.range S,
    rvIOprob BLOCK_N T hBN xs n * rvIOval BLOCK_N BLOCK_DMODEL T hBN ys n d
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

<details><summary><code>rvIOMetaBuf</code></summary>

```
/-- Slot-region table of the three per-batch metadata slots, in the kernel's
own load order (`B_Seqlen`, `B_Start_Loc`, `B_req_idx`). A shared def, never
an inline `match` in a window position. -/
```
```lean
def rvIOMetaBuf (B_Seqlen B_Start_Loc B_req_idx : Region .nat) : Fin 3 → RegionName
  | ⟨0, _⟩ => B_Seqlen.cast
  | ⟨1, _⟩ => B_Start_Loc.cast
  | ⟨_ + 2, _⟩ => B_req_idx.cast
```
</details>

<details><summary><code>rvIOprob</code></summary>

```
/-- The probability of live token `n` read off the first stream: step
`n / BLOCK_N`, lane `n % BLOCK_N` (`0` past the step budget — unreachable at
any `pre`-legal launch). -/
```
```lean
noncomputable def rvIOprob (BLOCK_N T : Nat) (hBN : 0 < BLOCK_N)
    (xs : Fin T → Fin BLOCK_N → ℝ) (n : Nat) : ℝ :=
  if h : n / BLOCK_N < T then
    xs ⟨n / BLOCK_N, h⟩ ⟨n % BLOCK_N, Nat.mod_lt _ hBN⟩
  else 0
```
</details>

<details><summary><code>rvIOval</code></summary>

```
/-- The gathered value row of live token `n`, channel `d`, read off the
second stream at lane `(n % BLOCK_N, d)`. The page indirection lives in the
*window* (`read2` eats `G`), so the stream cell is already the gathered
row. -/
```
```lean
noncomputable def rvIOval (BLOCK_N BLOCK_DMODEL T : Nat) (hBN : 0 < BLOCK_N)
    (ys : Fin T → Fin (BLOCK_N * BLOCK_DMODEL) → ℝ) (n : Nat)
    (d : Fin BLOCK_DMODEL) : ℝ :=
  if h : n / BLOCK_N < T then
    ys ⟨n / BLOCK_N, h⟩
      (Lane2D.encode (⟨n % BLOCK_N, Nat.mod_lt _ hBN⟩, d, PUnit.unit))
  else 0
```
</details>

## Also present (pinned special-case summaries)
- `token_attn_reducev_final_store_slice_compute_correct`
- `token_attn_reducev_closed_form_correct`
- `token_attn_reducev_closed_form_compute_correct`
