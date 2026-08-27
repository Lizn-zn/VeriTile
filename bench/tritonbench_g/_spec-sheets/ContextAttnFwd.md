# Spec sheet — `bench/tritonbench_g/context_attn_fwd/ContextAttnFwd.lean`

**Python source:** `bench/tritonbench_g/context_attn_fwd/context_attn_fwd.py`

## Public theorem: `context_attn_fwd_surface_compute_correct_general`

<details><summary>docstring</summary>

```
/-- **General surface compute-correctness** for `context_attn_fwd.py` over symbolic
`BLOCK_M`/`BLOCK_N`/`BLOCK_DMODEL`/`H` and the per-axis strides (`kv_group_num = 1`).
Every active observable `Out` write holds the genuine boundary-masked causal-softmax
closed form `ctxFwdGenuineOutValueG` of the loaded Q/K/V memory — a pure function of
memory, NOT the kernel's executed readback. Side conditions: `0 < BLOCK_DMODEL`,
`0 < BLOCK_N`, and output-offset injectivity. -/
```
</details>

**Statement:**
```lean
specification context_attn_fwd_surface_compute_correct_general
    (Q K V Out B_Start_Loc B_Seqlen B_Prompt_Cache_Len : RegionName)
    (sm_scale : ℝ)
    (stride_qbs stride_qh stride_qd stride_kb stride_kh stride_ks stride_kd
      stride_vb stride_vh stride_vs stride_vd stride_obs stride_oh stride_od
      H BLOCK_DMODEL BLOCK_M BLOCK_N : Nat) (hD : 0 < BLOCK_DMODEL) (hBN : 0 < BLOCK_N)
    (s : BlockState)
    (hOInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => outOffset s H B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx))
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := context_attn_fwd_kernel_int8kv_surface Q K V sm_scale Out
        B_Start_Loc B_Seqlen B_Prompt_Cache_Len
        stride_qbs stride_qh stride_qd stride_kb stride_kh stride_ks stride_kd
        stride_vb stride_vh stride_vs stride_vd stride_obs stride_oh stride_od
        1 H BLOCK_DMODEL BLOCK_M BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          active s H B_Seqlen B_Prompt_Cache_Len BLOCK_M idx)
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          (Out, outOffset s H B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx)))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        ctxFwdGenuineOutValueG s Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len
          sm_scale H stride_qbs stride_qh stride_qd stride_kb stride_ks stride_kh stride_kd
          stride_vb stride_vs stride_vh stride_vd BLOCK_DMODEL BLOCK_M BLOCK_N idx)
```

**Assumptions / layout contracts:**
- `hD : 0 < BLOCK_DMODEL`
- `hBN : 0 < BLOCK_N`
- `hOInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => outOffset s H B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx)`
- `hundef : ∀ rg o, s.undef rg o = 0`
- `fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          active s H B_Seqlen B_Prompt_Cache_Len BLOCK_M idx`
- `fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          (Out, outOffset s H B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx)`

**Closed-form spec defs (transitive):** `outOffset`, `context_attn_fwd_kernel_int8kv_surface`, `active`, `ctxFwdGenuineOutValueG`, `startLoc`, `mIndex`, `curHead`, `dIndex`, `seqLen`, `contextAttnExactFoldMG`, `ctxFwdWindowG`, `ctxFwdBelG`, `curBatch`, `promptLen`, `gStateBot`, `ctxKVMG`, `gKeysUpto`, `osStepBot`, `ctxQTileMG`, `ctxKTileMG`, `ctxVTileMG`, `ctxQTileG`, `ctxKTileG`, `ctxVTileG`

<details><summary><code>outOffset</code></summary>

```lean
def outOffset
    (s : BlockState) (H : Nat) (B_Start_Loc : RegionName)
    (stride_obs stride_oh stride_od BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  (startLoc s H B_Start_Loc + mIndex s BLOCK_M idx.1) * stride_obs +
    curHead s H * stride_oh + dIndex idx * stride_od
```
</details>

<details><summary><code>context_attn_fwd_kernel_int8kv_surface</code></summary>

```
/-- Faithful DSL port of `context_attn_fwd.py`'s `_fwd_kernel_int8kv`. -/
```
```lean
def context_attn_fwd_kernel_int8kv_surface
    (Q K V : RegionName) (sm_scale : ℝ) (Out : RegionName)
    (B_Start_Loc B_Seqlen b_prompt_cache_len : Region .nat)
    (stride_qbs stride_qh stride_qd
      stride_kb stride_kh stride_ks stride_kd
      stride_vb stride_vh stride_vs stride_vd
      stride_obs stride_oh stride_od
      kv_group_num H BLOCK_DMODEL BLOCK_M BLOCK_N : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  cur_bh = tl.program_id(1)
  cur_batch = cur_bh // $(H)
  cur_head = cur_bh % $(H)

  cur_kv_head = cur_head // $(kv_group_num)
  prompt_cache_len = tl.load(b_prompt_cache_len + cur_batch)
  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch) - prompt_cache_len

  block_start_loc = $(BLOCK_M) * start_m

  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  offs_m = block_start_loc + tl.arange(0, $(BLOCK_M))
  off_q = (cur_batch_in_all_start_index + offs_m[:, None]) * $(stride_qbs) +
    cur_head * $(stride_qh) + offs_d[None, :] * $(stride_qd)
  q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)

  block_mask = tl.where(block_start_loc < cur_batch_seq_len, $(1), $(0))
  block_end_loc = tl.minimum(block_start_loc + $(BLOCK_M) + prompt_cache_len,
    cur_batch_seq_len + prompt_cache_len)
  for start_n in range($(0), block_mask * block_end_loc, $(BLOCK_N)) {
    start_n = tl.multiple_of(start_n, $(BLOCK_N))
    off_k = cur_batch * $(stride_kb) + (start_n + offs_n[None, :]) * $(stride_ks) +
      cur_kv_head * $(stride_kh) + offs_d[:, None] * $(stride_kd)
    k = tl.load(K + off_k,
      mask=(start_n + offs_n[None, :]) < block_end_loc,
      other=0.0)

    qk = tl.dot(q, k)
    mask = (offs_m[:, None] + prompt_cache_len) >= (start_n + offs_n[None, :])
    qk = tl.where(mask, qk * $((sm_scale : ℝ)), -1.0e8)
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk -= m_ij[:, None]
    p = tl.math.exp2(qk)
    l_ij = tl.sum(p, 1)

    alpha = tl.math.exp2(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    off_v = cur_batch * $(stride_vb) + (start_n + offs_n[:, None]) * $(stride_vs) +
      cur_kv_head * $(stride_vh) + offs_d[None, :] * $(stride_vd)
    v = tl.load(V + off_v,
      mask=(start_n + offs_n[:, None]) < block_end_loc,
      other=0.0)

    p = (p).to(v.dtype)
    acc = tl.dot(p, v, acc)
    m_i = m_ij
  }

  acc = acc / l_i[:, None]
  off_o = (cur_batch_in_all_start_index + offs_m[:, None]) * $(stride_obs) +
    cur_head * $(stride_oh) + offs_d[None, :] * $(stride_od)
  out_ptrs = Out + off_o
  tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_batch_seq_len)
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active
    (s : BlockState) (H : Nat) (B_Seqlen B_Prompt_Cache_Len : RegionName)
    (BLOCK_M : Nat) (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Prop :=
  mIndex s BLOCK_M idx.1 < seqLen s H B_Seqlen B_Prompt_Cache_Len
```
</details>

<details><summary><code>ctxFwdGenuineOutValueG</code></summary>

```
/-- **General genuine closed-form output value.** -/
```
```lean
noncomputable def ctxFwdGenuineOutValueG
    (s : BlockState) (Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len : RegionName)
    (sm_scale : ℝ)
    (H stride_qbs stride_qh stride_qd stride_kb stride_ks stride_kh stride_kd
      stride_vb stride_vs stride_vh stride_vd BLOCK_DMODEL BLOCK_M BLOCK_N : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : ℝ :=
  contextAttnExactFoldMG s Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len sm_scale
    H stride_qbs stride_qh stride_qd stride_kb stride_ks stride_kh stride_kd
    stride_vb stride_vs stride_vh stride_vd BLOCK_DMODEL BLOCK_M
    (ctxFwdWindowG s B_Seqlen B_Prompt_Cache_Len H BLOCK_M BLOCK_N)
    (ctxFwdBelG s B_Seqlen B_Prompt_Cache_Len H BLOCK_M) idx
```
</details>

<details><summary><code>startLoc</code></summary>

```lean
def startLoc (s : BlockState) (H : Nat) (B_Start_Loc : RegionName) : Nat :=
  s.readMemValue .nat B_Start_Loc (curBatch s H)
```
</details>

<details><summary><code>mIndex</code></summary>

```lean
def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val
```
</details>

<details><summary><code>curHead</code></summary>

```lean
def curHead (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 % H
```
</details>

<details><summary><code>dIndex</code></summary>

```lean
def dIndex (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  idx.2.1.val
```
</details>

<details><summary><code>seqLen</code></summary>

```lean
def seqLen
    (s : BlockState) (H : Nat) (B_Seqlen B_Prompt_Cache_Len : RegionName) : Nat :=
  s.readMemValue .nat B_Seqlen (curBatch s H) -
    promptLen s H B_Prompt_Cache_Len
```
</details>

<details><summary><code>contextAttnExactFoldMG</code></summary>

```
/-- **General faithful kernel value** at output lane `(i, d)`: `acc/l` of the
⊥-seeded online-softmax fold over `ctxKVMG` for the full streamed window `[0, S)`.
A pure function of `Q`/`K`/`V` memory. -/
```
```lean
noncomputable def contextAttnExactFoldMG
    (s : BlockState) (Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len : RegionName)
    (sm_scale : ℝ)
    (H stride_qbs stride_qh stride_qd stride_kb stride_ks stride_kh stride_kd
      stride_vb stride_vs stride_vh stride_vd BLOCK_DMODEL BLOCK_M S bel : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : ℝ :=
  let st := gStateBot S S (ctxKVMG s Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len sm_scale
      H stride_qbs stride_qh stride_qd stride_kb stride_ks stride_kh stride_kd
      stride_vb stride_vs stride_vh stride_vd BLOCK_DMODEL BLOCK_M S bel idx.1 idx.2.1.val)
  st.2.2 / st.2.1
```
</details>

<details><summary><code>ctxFwdWindowG</code></summary>

```
/-- General kernel-decoded streamed window `S = ceil_{BN}(block_mask·block_end_loc)`. -/
```
```lean
def ctxFwdWindowG (s : BlockState) (B_Seqlen B_Prompt_Cache_Len : RegionName) (H BLOCK_M BLOCK_N : Nat) : Nat :=
  let plen := s.readMemValue .nat B_Prompt_Cache_Len (s.pids 1 / H)
  let sl := s.readMemValue .nat B_Seqlen (s.pids 1 / H) - plen
  let bel := ctxFwdBelG s B_Seqlen B_Prompt_Cache_Len H BLOCK_M
  let bm := if BLOCK_M * s.pids 0 < sl then 1 else 0
  BLOCK_N * ((bm * bel + (BLOCK_N - 1)) / BLOCK_N)
```
</details>

<details><summary><code>ctxFwdBelG</code></summary>

```
/-- General kernel-decoded `block_end_loc = min(BM·start_m + BM + plen, seq_len + plen)`. -/
```
```lean
def ctxFwdBelG (s : BlockState) (B_Seqlen B_Prompt_Cache_Len : RegionName) (H BLOCK_M : Nat) : Nat :=
  let plen := s.readMemValue .nat B_Prompt_Cache_Len (s.pids 1 / H)
  let sl := s.readMemValue .nat B_Seqlen (s.pids 1 / H) - plen
  let a := BLOCK_M * s.pids 0 + BLOCK_M + plen
  let b := sl + plen
  if a < b then a else b
```
</details>

<details><summary><code>curBatch</code></summary>

```lean
def curBatch (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 / H
```
</details>

<details><summary><code>promptLen</code></summary>

```lean
def promptLen (s : BlockState) (H : Nat) (B_Prompt_Cache_Len : RegionName) : Nat :=
  s.readMemValue .nat B_Prompt_Cache_Len (curBatch s H)
```
</details>

<details><summary><code>gStateBot</code></summary>

```
/-- Generic ⊥-seeded running `(max, l, acc)` after streaming `[0, hi)`. -/
```
```lean
noncomputable def gStateBot (S hi : Nat) (g : Fin S → ℝ × ℝ) : WithBot ℝ × ℝ × ℝ :=
  (gKeysUpto S hi g).foldl osStepBot (⊥, 0, 0)
```
</details>

<details><summary><code>ctxKVMG</code></summary>

```
/-- General faithful per-key `(score, value)` the loop folds, channel dim
`BLOCK_DMODEL`. Active causal lane (`j ≤ gi+plen`): `sm·Σ_{e<BLOCK_DMODEL}
ctxQTileMG(i,e)·ctxKTileMG(j,e)`; future lane: the `-1e8` sentinel; value:
`ctxVTileMG`. -/
```
```lean
noncomputable def ctxKVMG
    (s : BlockState) (Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len : RegionName) (sm_scale : ℝ)
    (H stride_qbs stride_qh stride_qd stride_kb stride_ks stride_kh stride_kd
      stride_vb stride_vs stride_vh stride_vd BLOCK_DMODEL BLOCK_M S bel : Nat)
    (i : Fin BLOCK_M) (d : Nat) (j : Fin S) : ℝ × ℝ :=
  (if j.val ≤ s.pids 0 * BLOCK_M + i.val + promptLen s H B_Prompt_Cache_Len then
      sm_scale * Finset.univ.sum (fun e : Fin BLOCK_DMODEL =>
        ctxQTileMG s Q B_Start_Loc B_Seqlen B_Prompt_Cache_Len H stride_qbs stride_qh stride_qd BLOCK_M i e.val
          * ctxKTileMG s K H stride_kb stride_ks stride_kh stride_kd S bel j e.val)
    else (0.0 - 10e7 : ℝ),
    ctxVTileMG s V H stride_vb stride_vs stride_vh stride_vd S bel j d)
```
</details>

<details><summary><code>gKeysUpto</code></summary>

```
/-- Generic windowed key list `[0, hi)` over an abstract per-key `g`. -/
```
```lean
noncomputable def gKeysUpto (S hi : Nat) (g : Fin S → ℝ × ℝ) : List (ℝ × ℝ) :=
  (List.finRange S).filterMap (fun j : Fin S => if j.val < hi then some (g j) else none)
```
</details>

<details><summary><code>osStepBot</code></summary>

```
/-- One ⊥-seeded online-softmax step: running max in `WithBot ℝ` (seeded `⊥`), so
`α = realExp2(m ⊖ m')` is `0` on the first block — faithful to the kernel's
`m_i = tl.zeros − inf` and `l_i`/`acc = 0`. -/
```
```lean
noncomputable def osStepBot (st : WithBot ℝ × ℝ × ℝ) (sv : ℝ × ℝ) : WithBot ℝ × ℝ × ℝ :=
  let m := st.1; let l := st.2.1; let acc := st.2.2
  let sc := sv.1; let v := sv.2
  let m' := m ⊔ ((sc : ℝ) : WithBot ℝ)
  let α := (WithBot.realExp2 (WithBot.realSub m m')).unbotD 0
  let p := pow2 (sc - m'.unbotD 0)
  (m', l * α + p, acc * α + p * v)
```
</details>

<details><summary><code>ctxQTileMG</code></summary>

```
/-- General row-masked query tile: `ctxQTileG` for active rows
(`pids0·BLOCK_M + i < seq_len`), else `0`. -/
```
```lean
noncomputable def ctxQTileMG
    (s : BlockState) (Q B_Start_Loc B_Seqlen B_Prompt_Cache_Len : RegionName)
    (H stride_qbs stride_qh stride_qd BLOCK_M : Nat) (i : Fin BLOCK_M) (e : Nat) : ℝ :=
  if s.pids 0 * BLOCK_M + i.val < seqLen s H B_Seqlen B_Prompt_Cache_Len then
    ctxQTileG s Q B_Start_Loc H stride_qbs stride_qh stride_qd BLOCK_M i e
  else 0
```
</details>

<details><summary><code>ctxKTileMG</code></summary>

```
/-- General `block_end_loc`-masked key tile: `ctxKTileG` for `j < bel`, else `0`. -/
```
```lean
noncomputable def ctxKTileMG (s : BlockState) (K : RegionName)
    (H stride_kb stride_ks stride_kh stride_kd S bel : Nat)
    (j : Fin S) (e : Nat) : ℝ :=
  if j.val < bel then ctxKTileG s K H stride_kb stride_ks stride_kh stride_kd S j e else 0
```
</details>

<details><summary><code>ctxVTileMG</code></summary>

```
/-- General `block_end_loc`-masked value tile: `ctxVTileG` for `j < bel`, else `0`. -/
```
```lean
noncomputable def ctxVTileMG (s : BlockState) (V : RegionName)
    (H stride_vb stride_vs stride_vh stride_vd S bel : Nat)
    (j : Fin S) (d : Nat) : ℝ :=
  if j.val < bel then ctxVTileG s V H stride_vb stride_vs stride_vh stride_vd S j d else 0
```
</details>

<details><summary><code>ctxQTileG</code></summary>

```
/-- General coordinate-faithful query tile `Q[gi, e]` at head/stride parameters
(`gi = pids0·BLOCK_M + i`, offset by `cur_batch_in_all_start_index`). -/
```
```lean
noncomputable def ctxQTileG
    (s : BlockState) (Q B_Start_Loc : RegionName)
    (H stride_qbs stride_qh stride_qd BLOCK_M : Nat)
    (i : Fin BLOCK_M) (e : Nat) : ℝ :=
  s.readMem Q
    ((s.readMemValue .nat B_Start_Loc (curBatch s H) + (s.pids 0 * BLOCK_M + i.val))
        * stride_qbs + curHead s H * stride_qh + e * stride_qd)
```
</details>

<details><summary><code>ctxKTileG</code></summary>

```
/-- General coordinate-faithful key tile `K[cur_batch, j, cur_head, e]`
(`kv_group_num = 1` so `cur_kv_head = cur_head`). -/
```
```lean
noncomputable def ctxKTileG (s : BlockState) (K : RegionName)
    (H stride_kb stride_ks stride_kh stride_kd S : Nat)
    (j : Fin S) (e : Nat) : ℝ :=
  s.readMem K (curBatch s H * stride_kb + j.val * stride_ks
    + curHead s H * stride_kh + e * stride_kd)
```
</details>

<details><summary><code>ctxVTileG</code></summary>

```
/-- General coordinate-faithful value tile `V[cur_batch, j, cur_head, d]`. -/
```
```lean
noncomputable def ctxVTileG (s : BlockState) (V : RegionName)
    (H stride_vb stride_vs stride_vh stride_vd S : Nat)
    (j : Fin S) (d : Nat) : ℝ :=
  s.readMem V (curBatch s H * stride_vb + j.val * stride_vs
    + curHead s H * stride_vh + d * stride_vd)
```
</details>

## Public theorem: `context_attn_fwd_io_correctness`

<details><summary>docstring</summary>

```
/-- **The `⊨[R]` streaming-metadata headline** — `context_attn_fwd` on
`StreamMetaMasked3DKernelIO₃`, the context-attention trio's skin, at fully
symbolic per-axis strides. For every rounding model `R`, the faithful
surface implements, on its metadata three-stream signature, the streamed
closed form `contextAttnFwdIOSpec`: every write-active output lane
`j = (i, e)` holds the `acc/l` readback of the ⊥-seeded online-softmax
`exp2` fold over the live streamed window, with the prompt-cache-offset
causal boundary and the **finite `-1e8` sentinel** — masked keys genuinely
carry weight `exp2(-1e8 − m)`, so the spec transcribes the kernel's fold
verbatim rather than a cleaned-up causal softmax (the `-1e8` honesty is
inherited from the exact headline, not repaired here). The three `.nat`
slots enter only through the windows/masks (`m 0 = prompt_cache_len`,
`m 1 = cur_batch_in_all_start_index`, `m 2` = the **raw** `B_Seqlen` load —
the kernel subtracts in-register, so masks carry the ℕ-truncated
`m 2 - m 0` verbatim). The kernel has **zero rounding events** (`.nat` slot
loads, `other=0.0`-masked `.real` loads, `.real` in-loop arithmetic, untyped
terminal store), so the skin's boundary quantization degenerates: the
readback's `R.round .real` is the identity by the model's defining
`round_real`.

**Launch legality (`pre` = the trusted-launch boundary).** The triple is
guarded by `io.pre pid₀ m = (pid₀ < NT ∧ m 2 ≤ NT·BLOCK_M)` — exactly the
port's documented trusted boundary (see the file docstring's Scope section):
the host launches `grid = (cdiv(max_input_len, BLOCK_M), batch·head)` with
`NT` the first grid dimension and every raw `B_Seqlen[b] ≤ max_input_len ≤
NT·BLOCK_M`. The live trip count `block_mask·block_end_loc` has no pid-free
bound; under `pre` it is `≤ NT·BLOCK_M ≤ BLOCK_N·T`
(`ctxFwdIOBel_le`/`ctxFwdIOT_mul_le`), which is what makes the
`T = ⌈NT·BLOCK_M/BLOCK_N⌉`-step window citable in both the safety walk and
the value bridge. The `⊨[R]` triple says nothing about launches outside this
boundary.

**Hypothesis provenance**: `0 < BLOCK_DMODEL`, `0 < BLOCK_N` are the exact
headline's side conditions (nonempty tiles; `reduceMax` totality);
`0 < BLOCK_M` and `0 < NT` are truth-forced by the static `Q` stream's
step-`0` read (`0 < T`); `hOInj` restates the exact headline's **open**
output-offset injectivity side condition in ∀-pids/∀-slot form (per-axis
strides are symbolic here — unlike `context_attn_nopad`'s contiguous pin, no
`DM ≤ rs` discharge is available). `kv_group_num = 1` is inherited from the
exact headline's pin. The exact headline's `hundef` is **not** a hypothesis
here — the skin's Hoare triple carries the `undef` pin itself.

Relation to the exact surface: the `Realizes_without_Rounding` headline
above is retained unchanged; this `⊨[R]` face restates the same genuine
closed form (`ctxFwdGenuineOutValueG`) on the streaming metadata skin, for
every `R` at once. -/
```
</details>

**Statement:**
```lean
specification context_attn_fwd_io_correctness (R : RoundingModel)
    (Q K V Out : RegionName)
    (B_Start_Loc B_Seqlen b_prompt_cache_len : Region .nat)
    (sm_scale : ℝ)
    (stride_qbs stride_qh stride_qd stride_kb stride_kh stride_ks stride_kd
      stride_vb stride_vh stride_vs stride_vd stride_obs stride_oh stride_od
      H BLOCK_DMODEL BLOCK_M BLOCK_N NT : Nat)
    (hD : 0 < BLOCK_DMODEL) (hBN : 0 < BLOCK_N) (hBM : 0 < BLOCK_M) (hNT : 0 < NT)
    (hOInj : ∀ pid₀ pid₁ base : Nat, Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        (base + (pid₀ * BLOCK_M + idx.1.val)) * stride_obs + pid₁ % H * stride_oh
          + idx.2.1.val * stride_od)) :
    contextAttnFwdIO Q K V Out B_Start_Loc B_Seqlen b_prompt_cache_len sm_scale
        stride_qbs stride_qh stride_qd stride_kb stride_kh stride_ks stride_kd
        stride_vb stride_vh stride_vs stride_vd stride_obs stride_oh stride_od
        H BLOCK_DMODEL BLOCK_M BLOCK_N NT ⊨[R]
      fun pid₀ _ _ m xs ys zs j =>
        contextAttnFwdIOSpec BLOCK_M BLOCK_DMODEL BLOCK_N NT hBN
          (ctxFwdIOT_pos NT BLOCK_M BLOCK_N hBM hBN hNT) sm_scale pid₀
          (m (⟨0, by omega⟩ : Fin 3)) (m (⟨2, by omega⟩ : Fin 3)) xs ys zs j
```

**Assumptions / layout contracts:**
- `hD : 0 < BLOCK_DMODEL`
- `hBN : 0 < BLOCK_N`
- `hBM : 0 < BLOCK_M`
- `hNT : 0 < NT`

**Closed-form spec defs (transitive):** `contextAttnFwdIO`, `contextAttnFwdIOSpec`, `context_attn_fwd_kernel_int8kv_surface`, `ctxFwdIOMetaBuf`, `ctxFwdIOT`, `ctxFwdIOBel`, `gStateBot`, `ctxFwdIOqM`, `ctxFwdIOkM`, `ctxFwdIOvM`, `gKeysUpto`, `osStepBot`

<details><summary><code>contextAttnFwdIO</code></summary>

```
/-- **Streaming metadata IO signature** of `context_attn_fwd` on the
metadata-parametrized three-stream fold skin (S1: online-softmax fold +
terminal masked store, 2-D pid grid `(start_m, cur_bh)`), at the existing
headline's `kv_group_num = 1` pin (so `cur_kv_head = cur_head`) and fully
**symbolic per-axis strides**.

The kernel's `forRangeDyn` trip count `block_mask·block_end_loc` grows with
`pid₀` × the loaded slots, so the walk has **no pid-free step bound**:
`T := ⌈NT·BLOCK_M / BLOCK_N⌉` for a new `Nat` parameter `NT` (the host
grid's **first** dimension `cdiv(max_input_len, BLOCK_M)`), and the skin's
launch-legality field is

`pre := pid₀ < NT ∧ B_Seqlen[pid₁/H] ≤ NT·BLOCK_M`

— exactly the port's documented **trusted boundary** (see the file
docstring's Scope section): the host launches
`grid = (cdiv(max_input_len, BLOCK_M), batch·head)`, so every real program
has `start_m < NT`, and every raw `B_Seqlen[b] ≤ max_input_len ≤
NT·BLOCK_M`. The `⊨[R]` triple says nothing about launches outside this
boundary.

Windows transcribe the kernel's pointer arithmetic exactly, with the loaded
slot vector `m` in place of the in-state metadata reads (`m 0 =
prompt_cache_len`, `m 1 = cur_batch_in_all_start_index`, `m 2 = raw
B_Seqlen[pid₁/H]`; the register `cur_batch_seq_len` is `m 2 - m 0`):

* `read1` (`Q`, the **static** stream — the window ignores `t`): lane
  `j = (i, e)` row-major over `[BLOCK_M, BLOCK_DMODEL]` reads
  `(m 1 + (pid₀·BM + i))·stride_qbs + (pid₁%H)·stride_qh + e·stride_qd`;
  `mask1` is the row guard `pid₀·BM + i < m 2 − m 0`.
* `read2` (`K`, slot-free **batch-strided** window, shifted by `t·BN`
  columns per step): lane `j = (e, jL)` over `[BLOCK_DMODEL, BLOCK_N]` reads
  `(pid₁/H)·stride_kb + (t·BN + jL)·stride_ks + (pid₁%H)·stride_kh +
  e·stride_kd`; `mask2` is `t·BN + jL < block_end_loc(pid₀, m)` — the
  **slot-eating step mask** that keeps the dead tail steps unpinned.
* `read3` (`V`, mirror at `[BLOCK_N, BLOCK_DMODEL]`): lane `j = (jL, e)`
  reads `(pid₁/H)·stride_vb + (t·BN + jL)·stride_vs + (pid₁%H)·stride_vh +
  e·stride_vd`; `mask3` is `t·BN + jL < block_end_loc(pid₀, m)`.
* `write` (`Out`, the terminal store after the post-loop divide): lane
  `j = (i, e)` writes `(m 1 + (pid₀·BM + i))·stride_obs +
  (pid₁%H)·stride_oh + e·stride_od`; `writeMask` is the row guard.

`outDType` is the `.real` default: the terminal `tl.store` is untyped, so
there is no quantization event. -/
```
```lean
def contextAttnFwdIO (Q K V Out : RegionName)
    (B_Start_Loc B_Seqlen b_prompt_cache_len : Region .nat) (sm_scale : ℝ)
    (stride_qbs stride_qh stride_qd stride_kb stride_kh stride_ks stride_kd
      stride_vb stride_vh stride_vs stride_vd stride_obs stride_oh stride_od
      H BLOCK_DMODEL BLOCK_M BLOCK_N NT : Nat) :
    StreamMetaMasked3DKernelIO₃ where
  kernel := context_attn_fwd_kernel_int8kv_surface Q K V sm_scale Out
    B_Start_Loc B_Seqlen b_prompt_cache_len
    stride_qbs stride_qh stride_qd stride_kb stride_kh stride_ks stride_kd
    stride_vb stride_vh stride_vs stride_vd stride_obs stride_oh stride_od
    1 H BLOCK_DMODEL BLOCK_M BLOCK_N
  inp1 := Q
  inp2 := K
  inp3 := V
  out := Out
  nMeta := 3
  sty := fun _ => ChanTy.nat
  mbuf := ctxFwdIOMetaBuf B_Start_Loc B_Seqlen b_prompt_cache_len
  mwin := fun _ _ pid₁ _ => pid₁ / H
  T := ctxFwdIOT NT BLOCK_M BLOCK_N
  B1 := BLOCK_M * BLOCK_DMODEL
  B2 := BLOCK_DMODEL * BLOCK_N
  B3 := BLOCK_N * BLOCK_DMODEL
  C := BLOCK_M * BLOCK_DMODEL
  pre := fun pid₀ _ _ m =>
    pid₀ < NT ∧ m (⟨2, by omega⟩ : Fin 3) ≤ NT * BLOCK_M
  read1 := fun pid₀ pid₁ _ m _ j =>
    (m (⟨1, by omega⟩ : Fin 3) + (pid₀ * BLOCK_M + j.val / BLOCK_DMODEL)) * stride_qbs
      + pid₁ % H * stride_qh + j.val % BLOCK_DMODEL * stride_qd
  read2 := fun _ pid₁ _ _ t j =>
    pid₁ / H * stride_kb + (t.val * BLOCK_N + j.val % BLOCK_N) * stride_ks
      + pid₁ % H * stride_kh + j.val / BLOCK_N * stride_kd
  read3 := fun _ pid₁ _ _ t j =>
    pid₁ / H * stride_vb + (t.val * BLOCK_N + j.val / BLOCK_DMODEL) * stride_vs
      + pid₁ % H * stride_vh + j.val % BLOCK_DMODEL * stride_vd
  write := fun pid₀ pid₁ _ m j =>
    (m (⟨1, by omega⟩ : Fin 3) + (pid₀ * BLOCK_M + j.val / BLOCK_DMODEL)) * stride_obs
      + pid₁ % H * stride_oh + j.val % BLOCK_DMODEL * stride_od
  mask1 := fun pid₀ _ _ m _ j =>
    pid₀ * BLOCK_M + j.val / BLOCK_DMODEL
      < m (⟨2, by omega⟩ : Fin 3) - m (⟨0, by omega⟩ : Fin 3)
  mask2 := fun pid₀ _ _ m t j =>
    t.val * BLOCK_N + j.val % BLOCK_N
      < ctxFwdIOBel BLOCK_M pid₀ (m (⟨0, by omega⟩ : Fin 3)) (m (⟨2, by omega⟩ : Fin 3))
  mask3 := fun pid₀ _ _ m t j =>
    t.val * BLOCK_N + j.val / BLOCK_DMODEL
      < ctxFwdIOBel BLOCK_M pid₀ (m (⟨0, by omega⟩ : Fin 3)) (m (⟨2, by omega⟩ : Fin 3))
  writeMask := fun pid₀ _ _ m j =>
    pid₀ * BLOCK_M + j.val / BLOCK_DMODEL
      < m (⟨2, by omega⟩ : Fin 3) - m (⟨0, by omega⟩ : Fin 3)
```
</details>

<details><summary><code>contextAttnFwdIOSpec</code></summary>

```
/-- **The streamed closed form**: `ctxFwdGenuineOutValueG` restated VERBATIM
over the three streamed tiles — the `acc/l` readback of the ⊥-seeded
online-softmax `exp2` fold over the kernel's live streamed window
`S = BLOCK_N·⌈bm·bel / BLOCK_N⌉` (a function of `pid₀` and the slots), with
the prompt-cache-offset causal boundary and the **finite `-1e8` sentinel**
(`0.0 - 10e7`) on future lanes — masked keys genuinely carry weight
`exp2(-1e8 - m)`, so this is NOT cleaned up into a pure causal softmax.
Output lane `j = (i, e)` row-major over `[BLOCK_M, BLOCK_DMODEL]`. -/
```
```lean
noncomputable def contextAttnFwdIOSpec (BLOCK_M BLOCK_DMODEL BLOCK_N NT : Nat)
    (hBN : 0 < BLOCK_N) (hT : 0 < ctxFwdIOT NT BLOCK_M BLOCK_N) (sm_scale : ℝ)
    (pid₀ plen rawsl : Nat)
    (xs : Fin (ctxFwdIOT NT BLOCK_M BLOCK_N) → Fin (BLOCK_M * BLOCK_DMODEL) → ℝ)
    (ys : Fin (ctxFwdIOT NT BLOCK_M BLOCK_N) → Fin (BLOCK_DMODEL * BLOCK_N) → ℝ)
    (zs : Fin (ctxFwdIOT NT BLOCK_M BLOCK_N) → Fin (BLOCK_N * BLOCK_DMODEL) → ℝ)
    (j : Fin (BLOCK_M * BLOCK_DMODEL)) : ℝ :=
  let st := gStateBot
    (BLOCK_N * (((if BLOCK_M * pid₀ < rawsl - plen then 1 else 0)
        * ctxFwdIOBel BLOCK_M pid₀ plen rawsl + (BLOCK_N - 1)) / BLOCK_N))
    (BLOCK_N * (((if BLOCK_M * pid₀ < rawsl - plen then 1 else 0)
        * ctxFwdIOBel BLOCK_M pid₀ plen rawsl + (BLOCK_N - 1)) / BLOCK_N))
    (fun jg =>
      (if jg.val ≤ pid₀ * BLOCK_M + (Lane2D.decode j).1.val + plen then
          sm_scale * Finset.univ.sum (fun e : Fin BLOCK_DMODEL =>
            ctxFwdIOqM BLOCK_M BLOCK_DMODEL (ctxFwdIOT NT BLOCK_M BLOCK_N) hT
                pid₀ (rawsl - plen) xs (Lane2D.decode j).1 e
              * ctxFwdIOkM BLOCK_DMODEL BLOCK_N (ctxFwdIOT NT BLOCK_M BLOCK_N) hBN
                  (ctxFwdIOBel BLOCK_M pid₀ plen rawsl) ys jg.val e)
        else (0.0 - 10e7 : ℝ),
       ctxFwdIOvM BLOCK_N BLOCK_DMODEL (ctxFwdIOT NT BLOCK_M BLOCK_N) hBN
         (ctxFwdIOBel BLOCK_M pid₀ plen rawsl) zs jg.val (Lane2D.decode j).2.1))
  st.2.2 / st.2.1
```
</details>

<details><summary><code>context_attn_fwd_kernel_int8kv_surface</code></summary>

```
/-- Faithful DSL port of `context_attn_fwd.py`'s `_fwd_kernel_int8kv`. -/
```
```lean
def context_attn_fwd_kernel_int8kv_surface
    (Q K V : RegionName) (sm_scale : ℝ) (Out : RegionName)
    (B_Start_Loc B_Seqlen b_prompt_cache_len : Region .nat)
    (stride_qbs stride_qh stride_qd
      stride_kb stride_kh stride_ks stride_kd
      stride_vb stride_vh stride_vs stride_vd
      stride_obs stride_oh stride_od
      kv_group_num H BLOCK_DMODEL BLOCK_M BLOCK_N : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  cur_bh = tl.program_id(1)
  cur_batch = cur_bh // $(H)
  cur_head = cur_bh % $(H)

  cur_kv_head = cur_head // $(kv_group_num)
  prompt_cache_len = tl.load(b_prompt_cache_len + cur_batch)
  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch) - prompt_cache_len

  block_start_loc = $(BLOCK_M) * start_m

  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  offs_m = block_start_loc + tl.arange(0, $(BLOCK_M))
  off_q = (cur_batch_in_all_start_index + offs_m[:, None]) * $(stride_qbs) +
    cur_head * $(stride_qh) + offs_d[None, :] * $(stride_qd)
  q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)

  block_mask = tl.where(block_start_loc < cur_batch_seq_len, $(1), $(0))
  block_end_loc = tl.minimum(block_start_loc + $(BLOCK_M) + prompt_cache_len,
    cur_batch_seq_len + prompt_cache_len)
  for start_n in range($(0), block_mask * block_end_loc, $(BLOCK_N)) {
    start_n = tl.multiple_of(start_n, $(BLOCK_N))
    off_k = cur_batch * $(stride_kb) + (start_n + offs_n[None, :]) * $(stride_ks) +
      cur_kv_head * $(stride_kh) + offs_d[:, None] * $(stride_kd)
    k = tl.load(K + off_k,
      mask=(start_n + offs_n[None, :]) < block_end_loc,
      other=0.0)

    qk = tl.dot(q, k)
    mask = (offs_m[:, None] + prompt_cache_len) >= (start_n + offs_n[None, :])
    qk = tl.where(mask, qk * $((sm_scale : ℝ)), -1.0e8)
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk -= m_ij[:, None]
    p = tl.math.exp2(qk)
    l_ij = tl.sum(p, 1)

    alpha = tl.math.exp2(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    off_v = cur_batch * $(stride_vb) + (start_n + offs_n[:, None]) * $(stride_vs) +
      cur_kv_head * $(stride_vh) + offs_d[None, :] * $(stride_vd)
    v = tl.load(V + off_v,
      mask=(start_n + offs_n[:, None]) < block_end_loc,
      other=0.0)

    p = (p).to(v.dtype)
    acc = tl.dot(p, v, acc)
    m_i = m_ij
  }

  acc = acc / l_i[:, None]
  off_o = (cur_batch_in_all_start_index + offs_m[:, None]) * $(stride_obs) +
    cur_head * $(stride_oh) + offs_d[None, :] * $(stride_od)
  out_ptrs = Out + off_o
  tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_batch_seq_len)
}
```
</details>

<details><summary><code>ctxFwdIOMetaBuf</code></summary>

```
/-- Slot-region table of the three per-batch metadata slots, in the kernel's
own load order. A shared def, never an inline `match` in a window/spec
position. -/
```
```lean
def ctxFwdIOMetaBuf (B_Start_Loc B_Seqlen b_prompt_cache_len : Region .nat) :
    Fin 3 → RegionName
  | ⟨0, _⟩ => b_prompt_cache_len.cast
  | ⟨1, _⟩ => B_Start_Loc.cast
  | ⟨_ + 2, _⟩ => B_Seqlen.cast
```
</details>

<details><summary><code>ctxFwdIOT</code></summary>

```
/-- The pid-free step budget `T = ⌈NT·BLOCK_M / BLOCK_N⌉`: at a `pre`-legal
launch the live trip count `block_mask·block_end_loc ≤ B_Seqlen[b] ≤
NT·BLOCK_M` never outruns `T·BLOCK_N` (`ctxFwdIOBel_le` below). -/
```
```lean
def ctxFwdIOT (NT BLOCK_M BLOCK_N : Nat) : Nat :=
  (NT * BLOCK_M + (BLOCK_N - 1)) / BLOCK_N
```
</details>

<details><summary><code>ctxFwdIOBel</code></summary>

```
/-- The kernel-decoded `block_end_loc = min(BLOCK_M·pid₀ + BLOCK_M + plen,
(rawsl − plen) + plen)`, transcribed VERBATIM on the slot values (ℕ-truncated
subtraction and all) — `ctxFwdBelG` freed from its state anchor (they are
definitionally equal at the slot reads). -/
```
```lean
def ctxFwdIOBel (BLOCK_M pid₀ plen rawsl : Nat) : Nat :=
  let sl := rawsl - plen
  let a := BLOCK_M * pid₀ + BLOCK_M + plen
  let b := sl + plen
  if a < b then a else b
```
</details>

<details><summary><code>gStateBot</code></summary>

```
/-- Generic ⊥-seeded running `(max, l, acc)` after streaming `[0, hi)`. -/
```
```lean
noncomputable def gStateBot (S hi : Nat) (g : Fin S → ℝ × ℝ) : WithBot ℝ × ℝ × ℝ :=
  (gKeysUpto S hi g).foldl osStepBot (⊥, 0, 0)
```
</details>

<details><summary><code>ctxFwdIOqM</code></summary>

```
/-- The `Q` cell read off the (static) first stream, row-guarded like the
kernel's masked `q` load (`offs_m < cur_batch_seq_len` on the slot values
`sl = rawsl − plen`); the window ignores `t`, so the step-`0` slice carries
the whole `[BLOCK_M, BLOCK_DMODEL]` tile. -/
```
```lean
noncomputable def ctxFwdIOqM (BLOCK_M BLOCK_DMODEL T : Nat) (hT : 0 < T)
    (pid₀ sl : Nat) (xs : Fin T → Fin (BLOCK_M * BLOCK_DMODEL) → ℝ)
    (i : Fin BLOCK_M) (e : Fin BLOCK_DMODEL) : ℝ :=
  if pid₀ * BLOCK_M + i.val < sl then
    xs ⟨0, hT⟩ (Lane2D.encode (i, e, PUnit.unit))
  else 0
```
</details>

<details><summary><code>ctxFwdIOkM</code></summary>

```
/-- The `block_end_loc`-masked `K` cell at global key `jg` off the second
stream: step `jg / BLOCK_N`, block column `jg % BLOCK_N`, stream lane
`(e, jg % BLOCK_N)`; `0` at or beyond `bel` (the kernel's `other=0.0`) and
beyond the `T`-step budget (unreachable at any `pre`-legal launch). -/
```
```lean
noncomputable def ctxFwdIOkM (BLOCK_DMODEL BLOCK_N T : Nat) (hBN : 0 < BLOCK_N)
    (bel : Nat) (ys : Fin T → Fin (BLOCK_DMODEL * BLOCK_N) → ℝ)
    (jg : Nat) (e : Fin BLOCK_DMODEL) : ℝ :=
  if jg < bel then
    if h : jg / BLOCK_N < T then
      ys ⟨jg / BLOCK_N, h⟩
        (Lane2D.encode (e, ⟨jg % BLOCK_N, Nat.mod_lt _ hBN⟩, PUnit.unit))
    else 0
  else 0
```
</details>

<details><summary><code>ctxFwdIOvM</code></summary>

```
/-- The `block_end_loc`-masked `V` cell at global key `jg` off the third
stream, at stream lane `(jg % BLOCK_N, d)`. -/
```
```lean
noncomputable def ctxFwdIOvM (BLOCK_N BLOCK_DMODEL T : Nat) (hBN : 0 < BLOCK_N)
    (bel : Nat) (zs : Fin T → Fin (BLOCK_N * BLOCK_DMODEL) → ℝ)
    (jg : Nat) (d : Fin BLOCK_DMODEL) : ℝ :=
  if jg < bel then
    if h : jg / BLOCK_N < T then
      zs ⟨jg / BLOCK_N, h⟩
        (Lane2D.encode (⟨jg % BLOCK_N, Nat.mod_lt _ hBN⟩, d, PUnit.unit))
    else 0
  else 0
```
</details>

<details><summary><code>gKeysUpto</code></summary>

```
/-- Generic windowed key list `[0, hi)` over an abstract per-key `g`. -/
```
```lean
noncomputable def gKeysUpto (S hi : Nat) (g : Fin S → ℝ × ℝ) : List (ℝ × ℝ) :=
  (List.finRange S).filterMap (fun j : Fin S => if j.val < hi then some (g j) else none)
```
</details>

<details><summary><code>osStepBot</code></summary>

```
/-- One ⊥-seeded online-softmax step: running max in `WithBot ℝ` (seeded `⊥`), so
`α = realExp2(m ⊖ m')` is `0` on the first block — faithful to the kernel's
`m_i = tl.zeros − inf` and `l_i`/`acc = 0`. -/
```
```lean
noncomputable def osStepBot (st : WithBot ℝ × ℝ × ℝ) (sv : ℝ × ℝ) : WithBot ℝ × ℝ × ℝ :=
  let m := st.1; let l := st.2.1; let acc := st.2.2
  let sc := sv.1; let v := sv.2
  let m' := m ⊔ ((sc : ℝ) : WithBot ℝ)
  let α := (WithBot.realExp2 (WithBot.realSub m m')).unbotD 0
  let p := pow2 (sc - m'.unbotD 0)
  (m', l * α + p, acc * α + p * v)
```
</details>
