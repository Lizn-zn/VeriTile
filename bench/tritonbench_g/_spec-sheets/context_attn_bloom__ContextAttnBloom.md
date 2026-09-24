# Spec sheet — `bench/tritonbench_g/context_attn_bloom/ContextAttnBloom.lean`

**Python source:** `bench/tritonbench_g/context_attn_bloom/context_attn_bloom.py`

## Public theorem: `context_attn_bloom_surface_compute_correct_general`

<details><summary>docstring</summary>

```
/-- **General surface compute-correctness** for `context_attn_bloom.py` over symbolic
`BLOCK_M`/`BLOCK_N`/`BLOCK_DMODEL`/`head_dim`, the per-axis K/V/Q/O strides and the
`Req_to_tokens` gather strides (`kv_group_num = 1`). Every active observable `Out`
write holds the genuine boundary-masked natural-exp in-loop-normalized causal-softmax
closed form `bloomFwdGenuineOutValueG` of the BLOOM gathered Q/K/V memory — a pure
function of memory, NOT the kernel's executed readback. The `-1e8` causal sentinel is
kept exactly. Side conditions: `0 < BLOCK_DMODEL`, `0 < BLOCK_N`, output-offset
injectivity. -/
```
</details>

**Statement:**
```lean
specification context_attn_bloom_surface_compute_correct_general
    (Q K V Out B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len : RegionName)
    (sm_scale : ℝ)
    (stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      head_dim BLOCK_DMODEL BLOCK_M BLOCK_N : Nat) (hD : 0 < BLOCK_DMODEL) (hBN : 0 < BLOCK_N)
    (s : BlockState)
    (hOInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => outOffset s B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx))
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := context_attn_bloom_fwd_kernel_surface Q K V sm_scale
        B_Start_Loc B_Seqlen Out Req_to_tokens B_req_idx B_Prompt_Cache_Len
        stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd
        stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
        stride_req_b stride_req_s 1 head_dim BLOCK_M BLOCK_DMODEL BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          active s B_Seqlen B_Prompt_Cache_Len head_dim BLOCK_M idx)
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          (Out, outOffset s B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx)))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bloomFwdGenuineOutValueG s Q K V B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len
          sm_scale stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
          stride_vbs stride_vh stride_vd head_dim BLOCK_DMODEL BLOCK_N BLOCK_M idx)
```

**Assumptions / layout contracts:**
- `hD : 0 < BLOCK_DMODEL`
- `hBN : 0 < BLOCK_N`
- `hOInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => outOffset s B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx)`
- `hundef : ∀ rg o, s.undef rg o = 0`
- `fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          active s B_Seqlen B_Prompt_Cache_Len head_dim BLOCK_M idx`
- `fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          (Out, outOffset s B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx)`

**Closed-form spec defs (transitive):** `outOffset`, `context_attn_bloom_fwd_kernel_surface`, `active`, `bloomFwdGenuineOutValueG`, `startLoc`, `mIndex`, `dIndex`, `seqLen`, `contextAttnBloomExactFoldMG`, `bloomFwdWindowG`, `bloomFwdBel`, `promptLen`, `gAccN`, `bloomKVMG`, `gStateBot`, `bloomQTileMG`, `bloomKTileMG`, `bloomVTileMG`, `gKeysUpto`, `osStepBot`, `bloomQTileG`, `bloomKTileG`, `bloomVTileG`, `curHead`, `bloomKvLocG`, `reqIdx`

<details><summary><code>outOffset</code></summary>

```lean
def outOffset
    (s : BlockState) (B_Start_Loc : RegionName)
    (stride_obs stride_oh stride_od BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  (startLoc s B_Start_Loc + mIndex s BLOCK_M idx.1) * stride_obs +
    s.pids 1 * stride_oh + dIndex idx * stride_od
```
</details>

<details><summary><code>context_attn_bloom_fwd_kernel_surface</code></summary>

```
/-- Faithful DSL port of `context_attn_bloom.py`'s `_fwd_kernel`. -/
```
```lean
def context_attn_bloom_fwd_kernel_surface
    (Q K V : RegionName) (sm_scale : ℝ)
    (B_Start_Loc B_Seqlen : Region .nat) (Out : RegionName)
    (Req_to_tokens B_req_idx b_prompt_cache_len : Region .nat)
    (stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd
      stride_obs stride_oh stride_od
      stride_req_to_tokens_b stride_req_to_tokens_s
      kv_group_num head_dim BLOCK_M BLOCK_DMODEL BLOCK_N : Nat) :
    ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  start_m = tl.program_id(2)

  cur_kv_head = cur_head // $(kv_group_num)

  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
  prompt_cache_len = tl.load(b_prompt_cache_len + cur_batch)
  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch) - prompt_cache_len
  cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

  block_start_loc = $(BLOCK_M) * start_m

  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  off_q = (cur_batch_in_all_start_index + offs_m[:, None]) * $(stride_qbs) +
    cur_head * $(stride_qh) + offs_d[None, :] * $(stride_qd)

  q = tl.load(Q + off_q,
    mask=(offs_m[:, None] < cur_batch_seq_len) & (offs_d[None, :] < $(head_dim)),
    other=0.0)

  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)

  block_mask = tl.where(block_start_loc < cur_batch_seq_len, $(1), $(0))
  block_end_loc = tl.minimum((start_m + $(1)) * $(BLOCK_M) + prompt_cache_len,
    cur_batch_seq_len + prompt_cache_len)

  for start_n in range($(0), block_mask * block_end_loc, $(BLOCK_N)) {
    start_n = tl.multiple_of(start_n, $(BLOCK_N))
    kv_loc = tl.load(Req_to_tokens + $(stride_req_to_tokens_b) * cur_batch_req_idx +
      $(stride_req_to_tokens_s) * (start_n + offs_n),
      mask=(start_n + offs_n) < block_end_loc,
      other=0)
    off_k = kv_loc[None, :] * $(stride_kbs) + cur_kv_head * $(stride_kh) +
      offs_d[:, None] * $(stride_kd)
    k = tl.load(K + off_k,
      mask=((start_n + offs_n[None, :]) < block_end_loc) &
        (offs_d[:, None] < $(head_dim)),
      other=0.0)

    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, k)
    qk *= $((sm_scale : ℝ))
    qk = tl.where(offs_m[:, None] + prompt_cache_len >= start_n + offs_n[None, :],
      qk, -100000000.0)

    m_ij = tl.max(qk, 1)
    p = tl.exp(qk - m_ij[:, None])
    l_ij = tl.sum(p, 1)
    m_i_new = tl.maximum(m_i, m_ij)
    alpha = tl.exp(m_i - m_i_new)
    beta = tl.exp(m_ij - m_i_new)
    l_i_new = alpha * l_i + beta * l_ij
    p_scale = beta / l_i_new
    p = p * p_scale[:, None]
    acc_scale = l_i / l_i_new * alpha
    acc_scale = tl.where(offs_m + prompt_cache_len >= start_n, acc_scale, 1.0)
    acc = acc * acc_scale[:, None]
    off_v = kv_loc[:, None] * $(stride_vbs) + cur_kv_head * $(stride_vh) +
      offs_d[None, :] * $(stride_vd)
    v = tl.load(V + off_v,
      mask=((start_n + offs_n[:, None]) < block_end_loc) &
        (offs_d[None, :] < $(head_dim)),
      other=0.0)
    p = (p).to(v.dtype)
    acc += tl.dot(p, v)
    l_i = l_i_new
    m_i = m_i_new
  }
  off_o = (cur_batch_in_all_start_index + offs_m[:, None]) * $(stride_obs) +
    cur_head * $(stride_oh) + offs_d[None, :] * $(stride_od)
  out_ptrs = Out + off_o
  tl.store(out_ptrs, acc,
    mask=(offs_m[:, None] < cur_batch_seq_len) & (offs_d[None, :] < $(head_dim)))
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active
    (s : BlockState) (B_Seqlen B_Prompt_Cache_Len : RegionName)
    (head_dim BLOCK_M : Nat) (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Prop :=
  mIndex s BLOCK_M idx.1 < seqLen s B_Seqlen B_Prompt_Cache_Len ∧
    dIndex idx < head_dim
```
</details>

<details><summary><code>bloomFwdGenuineOutValueG</code></summary>

```
/-- **General genuine closed-form `Out` value**: the block-causal-guarded boundary-masked
in-loop-normalized online-softmax fold `contextAttnBloomExactFoldMG` of the gathered
Q/K/V memory — a pure function of memory, not the kernel's executed readback. -/
```
```lean
noncomputable def bloomFwdGenuineOutValueG
    (s : BlockState) (Q K V B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len : RegionName)
    (sm_scale : ℝ)
    (stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd head_dim BLOCK_DMODEL BLOCK_N BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : ℝ :=
  contextAttnBloomExactFoldMG s Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len Req_to_tokens B_req_idx sm_scale
    stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
    stride_vbs stride_vh stride_vd head_dim BLOCK_DMODEL BLOCK_N BLOCK_M
    (bloomFwdWindowG s B_Seqlen B_Prompt_Cache_Len BLOCK_M BLOCK_N)
    (bloomFwdBel s B_Seqlen B_Prompt_Cache_Len BLOCK_M) idx
```
</details>

<details><summary><code>startLoc</code></summary>

```lean
def startLoc (s : BlockState) (B_Start_Loc : RegionName) : Nat :=
  s.readMemValue .nat B_Start_Loc (s.pids 0)
```
</details>

<details><summary><code>mIndex</code></summary>

```lean
def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 2 * BLOCK_M + i.val
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
def seqLen (s : BlockState) (B_Seqlen B_Prompt_Cache_Len : RegionName) : Nat :=
  s.readMemValue .nat B_Seqlen (s.pids 0) - promptLen s B_Prompt_Cache_Len
```
</details>

<details><summary><code>contextAttnBloomExactFoldMG</code></summary>

```
/-- **General faithful kernel value** at output lane `(i,d)`: the block-causal-guarded
normalized accumulator `gAccN` of the ⊥-seeded online-softmax fold over `bloomKVMG`
for the full streamed window `[0, S)`. A pure function of `Q`/`K`/`V` memory. -/
```
```lean
noncomputable def contextAttnBloomExactFoldMG
    (s : BlockState) (Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len Req_to_tokens B_req_idx : RegionName)
    (sm_scale : ℝ)
    (stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd head_dim BLOCK_DMODEL BLOCK_N BLOCK_M S bel : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : ℝ :=
  gAccN S BLOCK_N (s.pids 2 * BLOCK_M + idx.1.val + promptLen s B_Prompt_Cache_Len)
    (bloomKVMG s Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len Req_to_tokens B_req_idx sm_scale
      stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd head_dim BLOCK_DMODEL BLOCK_M S bel idx.1 idx.2.1.val)
    (S / BLOCK_N)
```
</details>

<details><summary><code>bloomFwdWindowG</code></summary>

```
/-- General kernel-decoded streamed window `S = BLOCK_N·⌈(block_mask·block_end_loc)/BLOCK_N⌉`
(loop step `BLOCK_N`; `block_end_loc` uses query block size `BLOCK_M`). -/
```
```lean
def bloomFwdWindowG (s : BlockState) (B_Seqlen B_Prompt_Cache_Len : RegionName) (BLOCK_M BLOCK_N : Nat) : Nat :=
  let plen := promptLen s B_Prompt_Cache_Len
  let sl := seqLen s B_Seqlen B_Prompt_Cache_Len
  let bel := bloomFwdBel s B_Seqlen B_Prompt_Cache_Len BLOCK_M
  let bm := if BLOCK_M * s.pids 2 < sl then 1 else 0
  BLOCK_N * ((bm * bel + (BLOCK_N - 1)) / BLOCK_N)
```
</details>

<details><summary><code>bloomFwdBel</code></summary>

```
/-- `block_end_loc = min((start_m+1)·BLOCK_M + plen, cur_batch_seq_len + plen)`. -/
```
```lean
def bloomFwdBel (s : BlockState) (B_Seqlen B_Prompt_Cache_Len : RegionName) (BLOCK_M : Nat) : Nat :=
  let plen := promptLen s B_Prompt_Cache_Len
  let sl := seqLen s B_Seqlen B_Prompt_Cache_Len
  let a := (s.pids 2 + 1) * BLOCK_M + plen
  let b := sl + plen
  if a < b then a else b
```
</details>

<details><summary><code>promptLen</code></summary>

```lean
def promptLen (s : BlockState) (B_Prompt_Cache_Len : RegionName) : Nat :=
  s.readMemValue .nat B_Prompt_Cache_Len (s.pids 0)
```
</details>

<details><summary><code>gAccN</code></summary>

```
/-- **Faithful normalized accumulator** after `c` blocks for a row with causal
limit `qpos`. Mirrors the kernel's `acc` register: `acc_new = acc·acc_scale +
dot(p,v)`, with `acc_scale = (lᵢ/lᵢⁿᵉʷ)·α` on guard-pass blocks (`c·BN ≤ qpos`)
and `1` on guard-fail blocks, and `dot(p,v) = (numerⁿᵉʷ − numer·α)/lᵢⁿᵉʷ`. -/
```
```lean
noncomputable def gAccN (S BN qpos : Nat) (g : Fin S → ℝ × ℝ) : Nat → ℝ
  | 0 => 0
  | c + 1 =>
    let st := gStateBot S (c * BN) g
    let stn := gStateBot S ((c + 1) * BN) g
    let α := (WithBot.realExp2 (WithBot.realSub st.1 stn.1)).unbotD 0
    let accScale := if c * BN ≤ qpos then (st.2.1 / stn.2.1) * α else 1
    gAccN S BN qpos g c * accScale + (stn.2.2 - st.2.2 * α) / stn.2.1
```
</details>

<details><summary><code>bloomKVMG</code></summary>

```
/-- General faithful per-key `(base-2 score, value)` the loop folds, with the
`-1e8` sentinel kept and the `block_end_loc`/channel load masks; score fed
`/ log 2` so `pow2 score = exp (natural kernel score)`. -/
```
```lean
noncomputable def bloomKVMG
    (s : BlockState) (Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len Req_to_tokens B_req_idx : RegionName)
    (sm_scale : ℝ)
    (stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd head_dim BLOCK_DMODEL BLOCK_M S bel : Nat)
    (i : Fin BLOCK_M) (d : Nat) (j : Fin S) : ℝ × ℝ :=
  ((if j.val ≤ s.pids 2 * BLOCK_M + i.val + promptLen s B_Prompt_Cache_Len then
      sm_scale * Finset.univ.sum (fun e : Fin BLOCK_DMODEL =>
        bloomQTileMG s Q B_Start_Loc B_Seqlen B_Prompt_Cache_Len stride_qbs stride_qh stride_qd head_dim BLOCK_M i e.val
          * bloomKTileMG s K Req_to_tokens B_req_idx stride_req_b stride_req_s stride_kbs stride_kh stride_kd head_dim S bel j e.val)
    else (0.0 - 10e7 : ℝ)) / Real.log 2,
    bloomVTileMG s V Req_to_tokens B_req_idx stride_req_b stride_req_s stride_vbs stride_vh stride_vd head_dim S bel j d)
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

<details><summary><code>bloomQTileMG</code></summary>

```
/-- General row/channel-masked query tile: genuine `bloomQTileG` on active rows
(`gi < seq_len`) and channel `e < head_dim`, else `0`. -/
```
```lean
noncomputable def bloomQTileMG
    (s : BlockState) (Q B_Start_Loc B_Seqlen B_Prompt_Cache_Len : RegionName)
    (stride_qbs stride_qh stride_qd head_dim BLOCK_M : Nat) (i : Fin BLOCK_M) (e : Nat) : ℝ :=
  if (s.pids 2 * BLOCK_M + i.val < seqLen s B_Seqlen B_Prompt_Cache_Len) ∧ (e < head_dim) then
    bloomQTileG s Q B_Start_Loc stride_qbs stride_qh stride_qd BLOCK_M i e
  else 0
```
</details>

<details><summary><code>bloomKTileMG</code></summary>

```
/-- General `block_end_loc`/channel-masked key tile: genuine `bloomKTileG` for
`j < bel` and channel `e < head_dim`, else `0`. -/
```
```lean
noncomputable def bloomKTileMG (s : BlockState) (K Req_to_tokens B_req_idx : RegionName)
    (stride_req_b stride_req_s stride_kbs stride_kh stride_kd head_dim S bel : Nat)
    (j : Fin S) (e : Nat) : ℝ :=
  if (j.val < bel) ∧ (e < head_dim) then
    bloomKTileG s K Req_to_tokens B_req_idx stride_req_b stride_req_s stride_kbs stride_kh stride_kd S j e
  else 0
```
</details>

<details><summary><code>bloomVTileMG</code></summary>

```
/-- General `block_end_loc`/channel-masked value tile. -/
```
```lean
noncomputable def bloomVTileMG (s : BlockState) (V Req_to_tokens B_req_idx : RegionName)
    (stride_req_b stride_req_s stride_vbs stride_vh stride_vd head_dim S bel : Nat)
    (j : Fin S) (d : Nat) : ℝ :=
  if (j.val < bel) ∧ (d < head_dim) then
    bloomVTileG s V Req_to_tokens B_req_idx stride_req_b stride_req_s stride_vbs stride_vh stride_vd S j d
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
`α = realExp2(m ⊖ m')` is `0` on the first block. -/
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

<details><summary><code>bloomQTileG</code></summary>

```
/-- General coordinate-faithful query tile `Q[gi, e]` at head/stride parameters. -/
```
```lean
noncomputable def bloomQTileG
    (s : BlockState) (Q B_Start_Loc : RegionName)
    (stride_qbs stride_qh stride_qd BLOCK_M : Nat) (i : Fin BLOCK_M) (e : Nat) : ℝ :=
  s.readMem Q
    ((startLoc s B_Start_Loc + (s.pids 2 * BLOCK_M + i.val)) * stride_qbs
      + curHead s * stride_qh + e * stride_qd)
```
</details>

<details><summary><code>bloomKTileG</code></summary>

```
/-- General coordinate-faithful key tile `K[kv_loc(j), cur_head, e]`. -/
```
```lean
noncomputable def bloomKTileG (s : BlockState) (K Req_to_tokens B_req_idx : RegionName)
    (stride_req_b stride_req_s stride_kbs stride_kh stride_kd S : Nat) (j : Fin S) (e : Nat) : ℝ :=
  s.readMem K (bloomKvLocG s Req_to_tokens B_req_idx stride_req_b stride_req_s j.val * stride_kbs
    + curHead s * stride_kh + e * stride_kd)
```
</details>

<details><summary><code>bloomVTileG</code></summary>

```
/-- General coordinate-faithful value tile `V[kv_loc(j), cur_head, d]`. -/
```
```lean
noncomputable def bloomVTileG (s : BlockState) (V Req_to_tokens B_req_idx : RegionName)
    (stride_req_b stride_req_s stride_vbs stride_vh stride_vd S : Nat) (j : Fin S) (d : Nat) : ℝ :=
  s.readMem V (bloomKvLocG s Req_to_tokens B_req_idx stride_req_b stride_req_s j.val * stride_vbs
    + curHead s * stride_vh + d * stride_vd)
```
</details>

<details><summary><code>curHead</code></summary>

```
/-- Head index of this kernel's program (`cur_head = pids 1`, `kv_group_num = 1`
so `cur_kv_head = cur_head`). -/
```
```lean
def curHead (s : BlockState) : Nat := s.pids 1
```
</details>

<details><summary><code>bloomKvLocG</code></summary>

```
/-- General `Req_to_tokens` gather: physical token slot for streamed key index
`j`, gather strides free:
`kv_loc(j) = Req_to_tokens[stride_req_b·req_idx + stride_req_s·j]`. -/
```
```lean
noncomputable def bloomKvLocG
    (s : BlockState) (Req_to_tokens B_req_idx : RegionName)
    (stride_req_b stride_req_s j : Nat) : Nat :=
  s.readMemValue .nat Req_to_tokens (reqIdx s B_req_idx * stride_req_b + stride_req_s * j)
```
</details>

<details><summary><code>reqIdx</code></summary>

```
/-- Request index for this batch: `cur_batch_req_idx = B_req_idx[cur_batch]`. -/
```
```lean
def reqIdx (s : BlockState) (B_req_idx : RegionName) : Nat :=
  s.readMemValue .nat B_req_idx (s.pids 0)
```
</details>

## Public theorem: `context_attn_bloom_io_correctness`

<details><summary>docstring</summary>

```
/-- **The `⊨[R]` gather-skin headline** — `context_attn_bloom` on
`StreamMetaGatherMasked3DKernelIO₃`, the paged-KV gather skin, at fully
symbolic per-axis strides and symbolic `head_dim`. For every rounding model
`R`, the faithful surface implements, on its metadata + gather + three-stream
signature, the streamed closed form `contextAttnBloomIOSpec`: every
write-active output lane `j = (i, e)` holds the **block-causal-guarded
normalized accumulator** `gAccN` of the ⊥-seeded online-softmax fold over the
live streamed window, with the prompt-cache-offset causal boundary and the
**finite `-1e8` sentinel** — masked keys genuinely carry weight
`exp(-1e8 − m)`, so the spec transcribes the kernel's fold verbatim rather
than a cleaned-up causal softmax (the `-1e8` honesty is inherited from the
exact headline, not repaired here).

**BLOOM's shape, versus its `context_attn_llama` sibling.** The grid is
genuinely 3-D (`cur_batch = pid₀`, `cur_head = pid₁`, `start_m = pid₂`), so
all four `.nat` slots are read at cell `pid₀` and `cur_head` enters the
windows as the bare `pid₁` (no `pid / H` decode). Normalization is **in
loop** (`p_scale = β/lᵢⁿᵉʷ`, `acc_scale = tl.where(offs_m+plen ≥ start_n,
(lᵢ/lᵢⁿᵉʷ)·α, 1.0)`) with **no** post-loop divide, so the spec is the `gAccN`
shape rather than llama's `acc/l` ratio; the block-causal guard on
`acc_scale` is modelled exactly. The kernel uses natural `tl.exp` with
`sm_scale = 1/√D`; the per-key score is divided by `Real.log 2` exactly as
the port's `bloomKVMG` does, so the shared `pow2`/`gStateBot` machinery
expresses it. The slots are `m 0 = cur_batch_in_all_start_index`,
`m 1 = prompt_cache_len`, `m 2` = the **raw** `B_Seqlen` load — the kernel
subtracts in-register, so masks carry the ℕ-truncated `m 2 - m 1` verbatim —
and `m 3 = cur_batch_req_idx`, the page-table row.

**The `head_dim` delta.** Every float load and the terminal store carry the
extra channel guard `offs_d < head_dim` (`BLOCK_DMODEL` is the padded tile
width), so `mask1`/`mask2`/`mask3`/`writeMask` are conjunctions. `gmask` is
*not*: the `Req_to_tokens` tile is a `[BLOCK_N]` vector with only the
`block_end_loc` liveness guard.

**The gather layer.** `Req_to_tokens` is the skin's index channel: a
`[BLOCK_N]` `.nat` tile `G t ·` per step, read at
`stride_req_b·m 3 + stride_req_s·(t·BN + jL)` under the same `block_end_loc`
liveness the `K`/`V` loads carry, `other = 0` on dead lanes. `G` is
universally quantified beside the slot vector and pinned with the skin's two
legs; the `K`/`V` windows eat it (`read2`/`read3`), while the spec `f` does
**not** — the gathered index moves addresses only, and the data pins already
deliver the gathered cells as the ordinary streams
(`bloomFwdIOSpec_eq_genuine` is the single place `bloomKvLocG ↦ G` is
rewritten, on the pin's active leg). Consequently the in-bounds-ness of the
*gathered values* is a hypothesis of the triple (`hbr2`/`hbr3`, quantified
over `G`), exactly like the slot values: host page-table contents are trusted
input.

The kernel has **zero rounding events** (`.nat` slot/gather loads,
`other=0.0`-masked `.real` loads, `.real` in-loop arithmetic, untyped terminal
store), so the skin's boundary quantization degenerates: the readback's
`R.round .real` is the identity by the model's defining `round_real`.

**Launch legality (`pre` = the trusted-launch boundary).** The triple is
guarded by `io.pre pid₂ m = (pid₂ < NT ∧ m 2 ≤ NT·BLOCK_M)` — the gather adds
**nothing** to `pre` — exactly the port's documented trusted boundary (see the
file docstring's Scope section): the host launches
`grid = (batch, head, cdiv(max_input_len, BLOCK))` with `NT` the third grid
dimension and every raw `B_Seqlen[b] ≤ max_input_len ≤ NT·BLOCK_M`. The live
trip count `block_mask·block_end_loc` has no pid-free bound; under `pre` it is
`≤ NT·BLOCK_M ≤ BLOCK_N·T` (`bloomFwdIOBel_le`/`bloomFwdIOT_mul_le`), which is
what makes the `T = ⌈NT·BLOCK_M/BLOCK_N⌉`-step window citable in both the
safety walk and the value bridge. The `⊨[R]` triple says nothing about
launches outside this boundary.

**Hypothesis provenance**: `0 < BLOCK_DMODEL`, `0 < BLOCK_N` are the exact
headline's side conditions (nonempty tiles; `reduceMax` totality);
`0 < BLOCK_M` and `0 < NT` are truth-forced by the static `Q` stream's
step-`0` read (`0 < T`); `hOInj` restates the exact headline's **open**
output-offset injectivity side condition in ∀-pids/∀-base form (per-axis
strides are symbolic here — unlike `context_attn_nopad`'s contiguous pin, no
`DM ≤ rs` discharge is available). `kv_group_num = 1` is inherited from the
exact headline's pin. The exact headline's `hundef` is **not** a hypothesis
here — the skin's Hoare triple carries the `undef` pin itself.

Relation to the exact surface: the `Realizes_without_Rounding` headline above
is retained unchanged; this `⊨[R]` face restates the same genuine closed form
(`bloomFwdGenuineOutValueG`) on the gather skin, for every `R` at once. -/
```
</details>

**Statement:**
```lean
specification context_attn_bloom_io_correctness (R : RoundingModel)
    (Q K V Out : RegionName)
    (B_Start_Loc B_Seqlen Req_to_tokens B_req_idx b_prompt_cache_len : Region .nat)
    (sm_scale : ℝ)
    (stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      head_dim BLOCK_DMODEL BLOCK_M BLOCK_N NT : Nat)
    (hD : 0 < BLOCK_DMODEL) (hBN : 0 < BLOCK_N) (hBM : 0 < BLOCK_M) (hNT : 0 < NT)
    (hOInj : ∀ pid₁ pid₂ base : Nat, Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        (base + (pid₂ * BLOCK_M + idx.1.val)) * stride_obs + pid₁ * stride_oh
          + idx.2.1.val * stride_od)) :
    contextAttnBloomIO Q K V Out B_Start_Loc B_Seqlen Req_to_tokens B_req_idx
        b_prompt_cache_len sm_scale
        stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
        stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
        head_dim BLOCK_DMODEL BLOCK_M BLOCK_N NT ⊨[R]
      fun _ _ pid₂ m xs ys zs j =>
        contextAttnBloomIOSpec BLOCK_M BLOCK_DMODEL BLOCK_N head_dim NT hBN
          (bloomFwdIOT_pos NT BLOCK_M BLOCK_N hBM hBN hNT) sm_scale pid₂
          (m (⟨1, by omega⟩ : Fin 4)) (m (⟨2, by omega⟩ : Fin 4)) xs ys zs j
```

**Assumptions / layout contracts:**
- `hD : 0 < BLOCK_DMODEL`
- `hBN : 0 < BLOCK_N`
- `hBM : 0 < BLOCK_M`
- `hNT : 0 < NT`

**Closed-form spec defs (transitive):** `contextAttnBloomIO`, `contextAttnBloomIOSpec`, `context_attn_bloom_fwd_kernel_surface`, `bloomIOMetaBuf`, `bloomFwdIOT`, `bloomFwdIOBel`, `gAccN`, `bloomFwdIOWindow`, `bloomFwdIOqM`, `bloomFwdIOkM`, `bloomFwdIOvM`, `gStateBot`, `gKeysUpto`, `osStepBot`

<details><summary><code>contextAttnBloomIO</code></summary>

```
/-- **Gather-indexed streaming metadata IO signature** of
`context_attn_bloom` on the gather-widened three-stream fold skin (S1:
online-softmax fold + terminal masked store, **3-D** pid grid
`(cur_batch, cur_head, start_m)`), at the existing headline's
`kv_group_num = 1` pin (so `cur_kv_head = cur_head`) and fully **symbolic
per-axis strides** and symbolic `head_dim`.

The kernel's `forRangeDyn` trip count `block_mask·block_end_loc` grows with
`pid₂` × the loaded slots, so the walk has **no pid-free step bound**:
`T := ⌈NT·BLOCK_M / BLOCK_N⌉` for a new `Nat` parameter `NT` (the host grid's
**third** dimension `cdiv(max_input_len, BLOCK)`), and the skin's
launch-legality field is

`pre := pid₂ < NT ∧ B_Seqlen[pid₀] ≤ NT·BLOCK_M`

— exactly the port's documented **trusted boundary** (see the file
docstring's Scope section): the host launches
`grid = (batch, head, cdiv(max_input_len, BLOCK))`, so every real program has
`start_m < NT`, and every raw `B_Seqlen[b] ≤ max_input_len ≤ NT·BLOCK_M`. The
`⊨[R]` triple says nothing about launches outside this boundary. The gather
adds **nothing** to `pre`.

Windows transcribe the kernel's pointer arithmetic exactly, with the loaded
slot vector `m` in place of the in-state metadata reads (`m 0 =
cur_batch_in_all_start_index`, `m 1 = prompt_cache_len`, `m 2 = raw
B_Seqlen[pid₀]`, `m 3 = cur_batch_req_idx`; the register `cur_batch_seq_len`
is `m 2 - m 1`) and the gathered tile `G` in place of the in-state `kv_loc`
register:

* `gread` (`Req_to_tokens`, the page table — `Bg = BLOCK_N`, `gty = .nat`,
  `gother = 0` from the load's `other=0`): lane `jL` at step `t` reads
  `stride_req_b·m 3 + stride_req_s·(t·BN + jL)`; `gmask` is the K/V liveness
  `t·BN + jL < block_end_loc(pid₂, m)` — the **same** guard the data loads
  carry, so dead lanes are never dereferenced. **No `head_dim` conjunct**
  (the page-table tile is 1-D).
* `read1` (`Q`, the **static** stream — the window ignores `t`): lane
  `j = (i, e)` row-major over `[BLOCK_M, BLOCK_DMODEL]` reads
  `(m 0 + (pid₂·BM + i))·stride_qbs + pid₁·stride_qh + e·stride_qd`;
  `mask1` is the row guard `pid₂·BM + i < m 2 − m 1` **∧** the channel guard
  `e < head_dim`.
* `read2` (`K`, **gather-addressed**, `kv_loc[None,:]` so the gathered lane
  sits on axis 1): lane `j = (e, jL)` over `[BLOCK_DMODEL, BLOCK_N]` reads
  `G t jL·stride_kbs + pid₁·stride_kh + e·stride_kd`; `mask2` is
  `t·BN + jL < block_end_loc(pid₂, m)` ∧ `e < head_dim`.
* `read3` (`V`, mirror at `[BLOCK_N, BLOCK_DMODEL]`, `kv_loc[:,None]` so the
  gathered lane sits on axis 0): lane `j = (jL, e)` reads
  `G t jL·stride_vbs + pid₁·stride_vh + e·stride_vd`; `mask3` is
  `t·BN + jL < block_end_loc(pid₂, m)` ∧ `e < head_dim`.
* `write` (`Out`, the terminal store — BLOOM normalizes **in loop**, so there
  is no post-loop divide): lane `j = (i, e)` writes
  `(m 0 + (pid₂·BM + i))·stride_obs + pid₁·stride_oh + e·stride_od`;
  `writeMask` is the row guard ∧ the channel guard.

The gathered *values* are host page-table contents — a trusted input, like
the slot values — so their in-bounds-ness is an `hbr2`/`hbr3` hypothesis of
the triple (quantified over `G`), not a `pre` fact.

`outDType` is the `.real` default: the terminal `tl.store` is untyped, so
there is no quantization event. -/
```
```lean
def contextAttnBloomIO (Q K V Out : RegionName)
    (B_Start_Loc B_Seqlen Req_to_tokens B_req_idx b_prompt_cache_len : Region .nat) (sm_scale : ℝ)
    (stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      head_dim BLOCK_DMODEL BLOCK_M BLOCK_N NT : Nat) :
    StreamMetaGatherMasked3DKernelIO₃ where
  kernel := context_attn_bloom_fwd_kernel_surface Q K V sm_scale B_Start_Loc B_Seqlen Out
    Req_to_tokens B_req_idx b_prompt_cache_len
    stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd
    stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
    stride_req_b stride_req_s 1 head_dim BLOCK_M BLOCK_DMODEL BLOCK_N
  inp1 := Q
  inp2 := K
  inp3 := V
  out := Out
  nMeta := 4
  sty := fun _ => ChanTy.nat
  mbuf := bloomIOMetaBuf B_Start_Loc B_Seqlen B_req_idx b_prompt_cache_len
  mwin := fun _ pid₀ _ _ => pid₀
  gbuf := Req_to_tokens
  gty := ChanTy.nat
  Bg := BLOCK_N
  gother := 0
  T := bloomFwdIOT NT BLOCK_M BLOCK_N
  B1 := BLOCK_M * BLOCK_DMODEL
  B2 := BLOCK_DMODEL * BLOCK_N
  B3 := BLOCK_N * BLOCK_DMODEL
  C := BLOCK_M * BLOCK_DMODEL
  pre := fun _ _ pid₂ m =>
    pid₂ < NT ∧ m (⟨2, by omega⟩ : Fin 4) ≤ NT * BLOCK_M
  gread := fun _ _ _ m t j =>
    stride_req_b * m (⟨3, by omega⟩ : Fin 4) + stride_req_s * (t.val * BLOCK_N + j.val)
  gmask := fun _ _ pid₂ m t j =>
    t.val * BLOCK_N + j.val
      < bloomFwdIOBel BLOCK_M pid₂ (m (⟨1, by omega⟩ : Fin 4)) (m (⟨2, by omega⟩ : Fin 4))
  read1 := fun _ pid₁ pid₂ m _ j =>
    (m (⟨0, by omega⟩ : Fin 4) + (pid₂ * BLOCK_M + j.val / BLOCK_DMODEL)) * stride_qbs
      + pid₁ * stride_qh + j.val % BLOCK_DMODEL * stride_qd
  read2 := fun _ pid₁ _ _ G t j =>
    G t (Lane2D.decode j).2.1 * stride_kbs
      + pid₁ * stride_kh + j.val / BLOCK_N * stride_kd
  read3 := fun _ pid₁ _ _ G t j =>
    G t (Lane2D.decode j).1 * stride_vbs
      + pid₁ * stride_vh + j.val % BLOCK_DMODEL * stride_vd
  write := fun _ pid₁ pid₂ m j =>
    (m (⟨0, by omega⟩ : Fin 4) + (pid₂ * BLOCK_M + j.val / BLOCK_DMODEL)) * stride_obs
      + pid₁ * stride_oh + j.val % BLOCK_DMODEL * stride_od
  mask1 := fun _ _ pid₂ m _ j =>
    pid₂ * BLOCK_M + j.val / BLOCK_DMODEL
        < m (⟨2, by omega⟩ : Fin 4) - m (⟨1, by omega⟩ : Fin 4)
      ∧ j.val % BLOCK_DMODEL < head_dim
  mask2 := fun _ _ pid₂ m t j =>
    t.val * BLOCK_N + j.val % BLOCK_N
        < bloomFwdIOBel BLOCK_M pid₂ (m (⟨1, by omega⟩ : Fin 4)) (m (⟨2, by omega⟩ : Fin 4))
      ∧ j.val / BLOCK_N < head_dim
  mask3 := fun _ _ pid₂ m t j =>
    t.val * BLOCK_N + j.val / BLOCK_DMODEL
        < bloomFwdIOBel BLOCK_M pid₂ (m (⟨1, by omega⟩ : Fin 4)) (m (⟨2, by omega⟩ : Fin 4))
      ∧ j.val % BLOCK_DMODEL < head_dim
  writeMask := fun _ _ pid₂ m j =>
    pid₂ * BLOCK_M + j.val / BLOCK_DMODEL
        < m (⟨2, by omega⟩ : Fin 4) - m (⟨1, by omega⟩ : Fin 4)
      ∧ j.val % BLOCK_DMODEL < head_dim
```
</details>

<details><summary><code>contextAttnBloomIOSpec</code></summary>

```
/-- **The streamed closed form**: `bloomFwdGenuineOutValueG` restated VERBATIM
over the three streamed tiles — the **block-causal-guarded normalized
accumulator** `gAccN` (BLOOM normalizes *in loop*: `p_scale = β/lᵢⁿᵉʷ`,
`acc_scale = tl.where(offs_m+plen ≥ start_n, (lᵢ/lᵢⁿᵉʷ)·α, 1.0)`, and there
is **no** post-loop divide) of the ⊥-seeded online-softmax fold over the
kernel's live streamed window `S = BLOCK_N·⌈bm·bel / BLOCK_N⌉` (a function of
`pid₂` and the slots), with the prompt-cache-offset causal boundary and the
**finite `-1e8` sentinel** on future lanes — masked keys genuinely carry
weight `exp(-1e8 - m)`, so this is NOT cleaned up into a pure causal softmax.

The per-key score is divided by `Real.log 2` exactly as the port's
`bloomKVMG` does, so the shared `pow2`/`gStateBot` machinery expresses the
kernel's **natural** `tl.exp` with `sm_scale = 1/√D`. Output lane
`j = (i, e)` row-major over `[BLOCK_M, BLOCK_DMODEL]`. -/
```
```lean
noncomputable def contextAttnBloomIOSpec
    (BLOCK_M BLOCK_DMODEL BLOCK_N head_dim NT : Nat)
    (hBN : 0 < BLOCK_N) (hT : 0 < bloomFwdIOT NT BLOCK_M BLOCK_N) (sm_scale : ℝ)
    (pid₂ plen rawsl : Nat)
    (xs : Fin (bloomFwdIOT NT BLOCK_M BLOCK_N) → Fin (BLOCK_M * BLOCK_DMODEL) → ℝ)
    (ys : Fin (bloomFwdIOT NT BLOCK_M BLOCK_N) → Fin (BLOCK_DMODEL * BLOCK_N) → ℝ)
    (zs : Fin (bloomFwdIOT NT BLOCK_M BLOCK_N) → Fin (BLOCK_N * BLOCK_DMODEL) → ℝ)
    (j : Fin (BLOCK_M * BLOCK_DMODEL)) : ℝ :=
  gAccN (bloomFwdIOWindow BLOCK_M BLOCK_N pid₂ plen rawsl) BLOCK_N
    (pid₂ * BLOCK_M + (Lane2D.decode j).1.val + plen)
    (fun jg =>
      ((if jg.val ≤ pid₂ * BLOCK_M + (Lane2D.decode j).1.val + plen then
          sm_scale * Finset.univ.sum (fun e : Fin BLOCK_DMODEL =>
            bloomFwdIOqM BLOCK_M BLOCK_DMODEL (bloomFwdIOT NT BLOCK_M BLOCK_N) head_dim hT
                pid₂ (rawsl - plen) xs (Lane2D.decode j).1 e
              * bloomFwdIOkM BLOCK_DMODEL BLOCK_N (bloomFwdIOT NT BLOCK_M BLOCK_N) head_dim hBN
                  (bloomFwdIOBel BLOCK_M pid₂ plen rawsl) ys jg.val e)
        else (0.0 - 10e7 : ℝ)) / Real.log 2,
       bloomFwdIOvM BLOCK_N BLOCK_DMODEL (bloomFwdIOT NT BLOCK_M BLOCK_N) head_dim hBN
         (bloomFwdIOBel BLOCK_M pid₂ plen rawsl) zs jg.val (Lane2D.decode j).2.1))
    (bloomFwdIOWindow BLOCK_M BLOCK_N pid₂ plen rawsl / BLOCK_N)
```
</details>

<details><summary><code>context_attn_bloom_fwd_kernel_surface</code></summary>

```
/-- Faithful DSL port of `context_attn_bloom.py`'s `_fwd_kernel`. -/
```
```lean
def context_attn_bloom_fwd_kernel_surface
    (Q K V : RegionName) (sm_scale : ℝ)
    (B_Start_Loc B_Seqlen : Region .nat) (Out : RegionName)
    (Req_to_tokens B_req_idx b_prompt_cache_len : Region .nat)
    (stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd
      stride_obs stride_oh stride_od
      stride_req_to_tokens_b stride_req_to_tokens_s
      kv_group_num head_dim BLOCK_M BLOCK_DMODEL BLOCK_N : Nat) :
    ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  start_m = tl.program_id(2)

  cur_kv_head = cur_head // $(kv_group_num)

  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
  prompt_cache_len = tl.load(b_prompt_cache_len + cur_batch)
  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch) - prompt_cache_len
  cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

  block_start_loc = $(BLOCK_M) * start_m

  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  off_q = (cur_batch_in_all_start_index + offs_m[:, None]) * $(stride_qbs) +
    cur_head * $(stride_qh) + offs_d[None, :] * $(stride_qd)

  q = tl.load(Q + off_q,
    mask=(offs_m[:, None] < cur_batch_seq_len) & (offs_d[None, :] < $(head_dim)),
    other=0.0)

  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)

  block_mask = tl.where(block_start_loc < cur_batch_seq_len, $(1), $(0))
  block_end_loc = tl.minimum((start_m + $(1)) * $(BLOCK_M) + prompt_cache_len,
    cur_batch_seq_len + prompt_cache_len)

  for start_n in range($(0), block_mask * block_end_loc, $(BLOCK_N)) {
    start_n = tl.multiple_of(start_n, $(BLOCK_N))
    kv_loc = tl.load(Req_to_tokens + $(stride_req_to_tokens_b) * cur_batch_req_idx +
      $(stride_req_to_tokens_s) * (start_n + offs_n),
      mask=(start_n + offs_n) < block_end_loc,
      other=0)
    off_k = kv_loc[None, :] * $(stride_kbs) + cur_kv_head * $(stride_kh) +
      offs_d[:, None] * $(stride_kd)
    k = tl.load(K + off_k,
      mask=((start_n + offs_n[None, :]) < block_end_loc) &
        (offs_d[:, None] < $(head_dim)),
      other=0.0)

    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, k)
    qk *= $((sm_scale : ℝ))
    qk = tl.where(offs_m[:, None] + prompt_cache_len >= start_n + offs_n[None, :],
      qk, -100000000.0)

    m_ij = tl.max(qk, 1)
    p = tl.exp(qk - m_ij[:, None])
    l_ij = tl.sum(p, 1)
    m_i_new = tl.maximum(m_i, m_ij)
    alpha = tl.exp(m_i - m_i_new)
    beta = tl.exp(m_ij - m_i_new)
    l_i_new = alpha * l_i + beta * l_ij
    p_scale = beta / l_i_new
    p = p * p_scale[:, None]
    acc_scale = l_i / l_i_new * alpha
    acc_scale = tl.where(offs_m + prompt_cache_len >= start_n, acc_scale, 1.0)
    acc = acc * acc_scale[:, None]
    off_v = kv_loc[:, None] * $(stride_vbs) + cur_kv_head * $(stride_vh) +
      offs_d[None, :] * $(stride_vd)
    v = tl.load(V + off_v,
      mask=((start_n + offs_n[:, None]) < block_end_loc) &
        (offs_d[None, :] < $(head_dim)),
      other=0.0)
    p = (p).to(v.dtype)
    acc += tl.dot(p, v)
    l_i = l_i_new
    m_i = m_i_new
  }
  off_o = (cur_batch_in_all_start_index + offs_m[:, None]) * $(stride_obs) +
    cur_head * $(stride_oh) + offs_d[None, :] * $(stride_od)
  out_ptrs = Out + off_o
  tl.store(out_ptrs, acc,
    mask=(offs_m[:, None] < cur_batch_seq_len) & (offs_d[None, :] < $(head_dim)))
}
```
</details>

<details><summary><code>bloomIOMetaBuf</code></summary>

```
/-- Slot-region table of the four per-batch metadata slots, in the kernel's
own load order. A shared def, never an inline `match` in a window/spec
position. -/
```
```lean
def bloomIOMetaBuf (B_Start_Loc B_Seqlen B_req_idx b_prompt_cache_len : Region .nat) :
    Fin 4 → RegionName
  | ⟨0, _⟩ => B_Start_Loc.cast
  | ⟨1, _⟩ => b_prompt_cache_len.cast
  | ⟨2, _⟩ => B_Seqlen.cast
  | ⟨_ + 3, _⟩ => B_req_idx.cast
```
</details>

<details><summary><code>bloomFwdIOT</code></summary>

```
/-- The pid-free step budget `T = ⌈NT·BLOCK_M / BLOCK_N⌉`: at a `pre`-legal
launch the live trip count `block_mask·block_end_loc ≤ B_Seqlen[b] ≤
NT·BLOCK_M` never outruns `T·BLOCK_N` (`bloomFwdIOBel_le` below). -/
```
```lean
def bloomFwdIOT (NT BLOCK_M BLOCK_N : Nat) : Nat :=
  (NT * BLOCK_M + (BLOCK_N - 1)) / BLOCK_N
```
</details>

<details><summary><code>bloomFwdIOBel</code></summary>

```
/-- The kernel-decoded `block_end_loc = min((pid₂+1)·BLOCK_M + plen,
(rawsl − plen) + plen)`, transcribed VERBATIM on the slot values
(ℕ-truncated subtraction and all) — `bloomFwdBel` freed from its state anchor
(they are definitionally equal at the slot reads). -/
```
```lean
def bloomFwdIOBel (BLOCK_M pid₂ plen rawsl : Nat) : Nat :=
  let sl := rawsl - plen
  let a := (pid₂ + 1) * BLOCK_M + plen
  let b := sl + plen
  if a < b then a else b
```
</details>

<details><summary><code>gAccN</code></summary>

```
/-- **Faithful normalized accumulator** after `c` blocks for a row with causal
limit `qpos`. Mirrors the kernel's `acc` register: `acc_new = acc·acc_scale +
dot(p,v)`, with `acc_scale = (lᵢ/lᵢⁿᵉʷ)·α` on guard-pass blocks (`c·BN ≤ qpos`)
and `1` on guard-fail blocks, and `dot(p,v) = (numerⁿᵉʷ − numer·α)/lᵢⁿᵉʷ`. -/
```
```lean
noncomputable def gAccN (S BN qpos : Nat) (g : Fin S → ℝ × ℝ) : Nat → ℝ
  | 0 => 0
  | c + 1 =>
    let st := gStateBot S (c * BN) g
    let stn := gStateBot S ((c + 1) * BN) g
    let α := (WithBot.realExp2 (WithBot.realSub st.1 stn.1)).unbotD 0
    let accScale := if c * BN ≤ qpos then (st.2.1 / stn.2.1) * α else 1
    gAccN S BN qpos g c * accScale + (stn.2.2 - st.2.2 * α) / stn.2.1
```
</details>

<details><summary><code>bloomFwdIOWindow</code></summary>

```
/-- The kernel-decoded streamed window `S = BLOCK_N·⌈bm·bel / BLOCK_N⌉`
(`bloomFwdWindowG` freed from its state anchor). -/
```
```lean
def bloomFwdIOWindow (BLOCK_M BLOCK_N pid₂ plen rawsl : Nat) : Nat :=
  BLOCK_N * (((if BLOCK_M * pid₂ < rawsl - plen then 1 else 0)
      * bloomFwdIOBel BLOCK_M pid₂ plen rawsl + (BLOCK_N - 1)) / BLOCK_N)
```
</details>

<details><summary><code>bloomFwdIOqM</code></summary>

```
/-- The `Q` cell read off the (static) first stream, row/channel-guarded like
the kernel's masked `q` load (`offs_m < cur_batch_seq_len` on the slot values
`sl = rawsl − plen`, `offs_d < head_dim`); the window ignores `t`, so the
step-`0` slice carries the whole `[BLOCK_M, BLOCK_DMODEL]` tile. -/
```
```lean
noncomputable def bloomFwdIOqM (BLOCK_M BLOCK_DMODEL T head_dim : Nat) (hT : 0 < T)
    (pid₂ sl : Nat) (xs : Fin T → Fin (BLOCK_M * BLOCK_DMODEL) → ℝ)
    (i : Fin BLOCK_M) (e : Fin BLOCK_DMODEL) : ℝ :=
  if pid₂ * BLOCK_M + i.val < sl ∧ e.val < head_dim then
    xs ⟨0, hT⟩ (Lane2D.encode (i, e, PUnit.unit))
  else 0
```
</details>

<details><summary><code>bloomFwdIOkM</code></summary>

```
/-- The `block_end_loc`/channel-masked `K` cell at global key `jg` off the
second stream: step `jg / BLOCK_N`, block column `jg % BLOCK_N`, stream lane
`(e, jg % BLOCK_N)`; `0` at or beyond `bel`, off-channel (`e ≥ head_dim`)
— the kernel's `other=0.0` — and beyond the `T`-step budget (unreachable at
any `pre`-legal launch). -/
```
```lean
noncomputable def bloomFwdIOkM (BLOCK_DMODEL BLOCK_N T head_dim : Nat) (hBN : 0 < BLOCK_N)
    (bel : Nat) (ys : Fin T → Fin (BLOCK_DMODEL * BLOCK_N) → ℝ)
    (jg : Nat) (e : Fin BLOCK_DMODEL) : ℝ :=
  if jg < bel ∧ e.val < head_dim then
    if h : jg / BLOCK_N < T then
      ys ⟨jg / BLOCK_N, h⟩
        (Lane2D.encode (e, ⟨jg % BLOCK_N, Nat.mod_lt _ hBN⟩, PUnit.unit))
    else 0
  else 0
```
</details>

<details><summary><code>bloomFwdIOvM</code></summary>

```
/-- The `block_end_loc`/channel-masked `V` cell at global key `jg` off the
third stream, at stream lane `(jg % BLOCK_N, d)`. -/
```
```lean
noncomputable def bloomFwdIOvM (BLOCK_N BLOCK_DMODEL T head_dim : Nat) (hBN : 0 < BLOCK_N)
    (bel : Nat) (zs : Fin T → Fin (BLOCK_N * BLOCK_DMODEL) → ℝ)
    (jg : Nat) (d : Fin BLOCK_DMODEL) : ℝ :=
  if jg < bel ∧ d.val < head_dim then
    if h : jg / BLOCK_N < T then
      zs ⟨jg / BLOCK_N, h⟩
        (Lane2D.encode (⟨jg % BLOCK_N, Nat.mod_lt _ hBN⟩, d, PUnit.unit))
    else 0
  else 0
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
`α = realExp2(m ⊖ m')` is `0` on the first block. -/
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
