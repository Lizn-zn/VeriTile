# Spec sheet — `bench/tritonbench_g/context_attn_mistral/ContextAttnMistral.lean`

**Python source:** `bench/tritonbench_g/context_attn_mistral/context_attn_mistral.py`

## Public theorem: `context_attn_mistral_genuine_output_summary_general`

<details><summary>docstring</summary>

```
/-- **General Public summary for `context_attn_mistral.py`.**

The full faithful `_fwd_kernel` surface (preLoop + streaming-softmax `forRangeDyn`
loop + masked store) *realizes* the genuine sliding-window causal-softmax closed
form `mistralGenuineOutValueG` — the boundary-masked sliding-window-softmax fold of
the loaded Q/K/V memory with the `-1e9` sentinel kept — at every active output
lane, at the **dimension-parameterized** contiguous layout `(stride_*bs, stride_*h,
stride_*d) = (rs, hs, 1)`, `kv_group_num = 1`, with `BLOCK_M = BLOCK_N = BLK`,
`BLOCK_DMODEL = DM`, `sliding_window = sw`, and a free `sm_scale`. NOT a
self-referential executed value: the streaming `m_i`/`l_i`/`acc` recurrence (two
`where` masks, the `-1e9` block-max guard, and the `l_i_new` denominator guard) is
decoded statement-by-statement and proven to collapse to the closed form. Side
conditions: `0 < BLK`, `0 < DM`, `DM ≤ rs` (output-offset injectivity), `hundef`.
Instantiating `BLK = DM = 128`, `rs = 768`, `hs = 128`, `sm_scale = (√128)⁻¹`,
`sw ∈ {10, 20}` recovers the Python test-shape summary. -/
```
</details>

**Statement:**
```lean
theorem context_attn_mistral_genuine_output_summary_general
    (Q K V : RegionName) (B_Start_Loc B_Seqlen : Region .nat)
    (Out : RegionName) (sm_scale : ℝ) (rs hs BLK DM sw : Nat)
    (hBLK : 0 < BLK) (hDM : 0 < DM) (hDMrs : DM ≤ rs)
    (s : BlockState) (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes
      (kernel := context_attn_mistral_fwd_kernel_surface Q K V sm_scale B_Start_Loc B_Seqlen Out
        rs hs 1 rs hs 1 rs hs 1 rs hs 1 1 sw BLK DM BLK)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLK, DM] => mistralActiveG s B_Seqlen BLK DM idx)
        (fun idx : TileIndex [BLK, DM] => (Out, mistralOutOffsetG s B_Start_Loc rs hs BLK DM idx)))
      (expected := fun idx : TileIndex [BLK, DM] =>
        mistralGenuineOutValueG s Q K V B_Start_Loc B_Seqlen sm_scale rs hs BLK DM sw idx)
```

**Assumptions / layout contracts:**
- `hBLK : 0 < BLK`
- `hDM : 0 < DM`
- `hDMrs : DM ≤ rs`
- `hundef : ∀ rg o, s.undef rg o = 0`
- `kernel : = context_attn_mistral_fwd_kernel_surface Q K V sm_scale B_Start_Loc B_Seqlen Out
        rs hs 1 rs hs 1 rs hs 1 rs hs 1 1 sw BLK DM BLK`
- `initialState : = s`
- `fun idx : TileIndex [BLK, DM] => mistralActiveG s B_Seqlen BLK DM idx`
- `fun idx : TileIndex [BLK, DM] => (Out, mistralOutOffsetG s B_Start_Loc rs hs BLK DM idx)`
- `expected : = fun idx : TileIndex [BLK, DM] =>
        mistralGenuineOutValueG s Q K V B_Start_Loc B_Seqlen sm_scale rs hs BLK DM sw idx`

**Closed-form spec defs (transitive):** `context_attn_mistral_fwd_kernel_surface`, `mistralActiveG`, `mistralOutOffsetG`, `mistralGenuineOutValueG`, `seqLen`, `startLoc`, `contextAttnMistralExactFoldMG`, `ctxMistralWindowG`, `ctxMistralBel`, `mistralScore`, `ctxVTileMG`, `mistralActive`, `ctxQTileG`, `ctxKTileMG`, `ctxVTileG`, `ctxKTileG`

<details><summary><code>context_attn_mistral_fwd_kernel_surface</code></summary>

```
/-- Faithful DSL port of `context_attn_mistral.py`'s `_fwd_kernel`. -/
```
```lean
def context_attn_mistral_fwd_kernel_surface
    (Q K V : RegionName) (sm_scale : ℝ)
    (B_Start_Loc B_Seqlen : Region .nat) (Out : RegionName)
    (stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd
      stride_obs stride_oh stride_od
      kv_group_num sliding_window BLOCK_M BLOCK_DMODEL BLOCK_N : Nat) :
    ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  start_m = tl.program_id(2)

  cur_kv_head = cur_head // $(kv_group_num)

  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

  block_start_loc = $(BLOCK_M) * start_m

  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  off_q = (cur_batch_in_all_start_index + offs_m[:, None]) * $(stride_qbs) +
    cur_head * $(stride_qh) + offs_d[None, :] * $(stride_qd)
  off_k = offs_n[None, :] * $(stride_kbs) + cur_kv_head * $(stride_kh) +
    offs_d[:, None] * $(stride_kd)
  off_v = offs_n[:, None] * $(stride_vbs) + cur_kv_head * $(stride_vh) +
    offs_d[None, :] * $(stride_vd)

  q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

  k_ptrs = K + off_k
  v_ptrs = V + off_v

  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)

  block_mask = tl.where(block_start_loc < cur_batch_seq_len, $(1), $(0))

  for start_n in range($(0), block_mask * (start_m + $(1)) * $(BLOCK_M), $(BLOCK_N)) {
    start_n = tl.multiple_of(start_n, $(BLOCK_N))
    k = tl.load(k_ptrs + (cur_batch_in_all_start_index + start_n) * $(stride_kbs),
      mask=(start_n + offs_n[None, :]) < cur_batch_seq_len,
      other=0.0)

    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, k)
    qk *= $((sm_scale : ℝ))
    qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, -1e9)
    qk = tl.where((start_n + offs_n[None, :]) > (offs_m[:, None] - $(sliding_window)),
      qk, -1e9)

    m_ij = tl.max(qk, 1)
    m_ij = tl.where(m_ij == -1e9, 0.0, m_ij)
    p = tl.exp(qk - m_ij[:, None])
    l_ij = tl.sum(p, 1)

    m_i_new = tl.maximum(m_i, m_ij)
    alpha = tl.exp(m_i - m_i_new)
    beta = tl.exp(m_ij - m_i_new)
    l_i_new = alpha * l_i + beta * l_ij
    l_i_new = tl.where(l_i_new == 0.0, 1e-9, l_i_new)

    p_scale = beta / l_i_new
    p = p * p_scale[:, None]
    acc_scale = l_i / l_i_new * alpha
    acc = acc * acc_scale[:, None]
    v = tl.load(v_ptrs + (cur_batch_in_all_start_index + start_n) * $(stride_vbs),
      mask=(start_n + offs_n[:, None]) < cur_batch_seq_len,
      other=0.0)

    p = (p).to(v.dtype)
    acc += tl.dot(p, v)
    l_i = l_i_new
    m_i = m_i_new
  }
  off_o = (cur_batch_in_all_start_index + offs_m[:, None]) * $(stride_obs) +
    cur_head * $(stride_oh) + offs_d[None, :] * $(stride_od)
  out_ptrs = Out + off_o
  tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_batch_seq_len)
}
```
</details>

<details><summary><code>mistralActiveG</code></summary>

```
/-- General active-output predicate. -/
```
```lean
def mistralActiveG (s : BlockState) (B_Seqlen : RegionName) (BLK DM : Nat)
    (idx : TileIndex [BLK, DM]) : Prop :=
  s.pids 2 * BLK + idx.1.val < seqLen s B_Seqlen
```
</details>

<details><summary><code>mistralOutOffsetG</code></summary>

```
/-- General output offset (contiguous layout, head stride `hs`). -/
```
```lean
def mistralOutOffsetG
    (s : BlockState) (B_Start_Loc : RegionName)
    (rs hs BLK DM : Nat) (idx : TileIndex [BLK, DM]) : Nat :=
  (startLoc s B_Start_Loc + (s.pids 2 * BLK + idx.1.val)) * rs
    + s.pids 1 * hs + idx.2.1.val
```
</details>

<details><summary><code>mistralGenuineOutValueG</code></summary>

```
/-- General genuine closed-form output value (boundary-masked sliding-window-softmax
fold), with the `-1e9` sentinel kept on masked cells. -/
```
```lean
noncomputable def mistralGenuineOutValueG
    (s : BlockState) (Q K V B_Start_Loc B_Seqlen : RegionName)
    (sm_scale : ℝ) (rs hs BLK DM sw : Nat) (idx : TileIndex [BLK, DM]) : ℝ :=
  contextAttnMistralExactFoldMG s Q K V B_Start_Loc sm_scale rs hs BLK DM
    (ctxMistralWindowG s B_Seqlen BLK) (ctxMistralBel s B_Seqlen) sw idx
```
</details>

<details><summary><code>seqLen</code></summary>

```lean
def seqLen (s : BlockState) (B_Seqlen : RegionName) : Nat :=
  s.readMemValue .nat B_Seqlen (s.pids 0)
```
</details>

<details><summary><code>startLoc</code></summary>

```lean
def startLoc (s : BlockState) (B_Start_Loc : RegionName) : Nat :=
  s.readMemValue .nat B_Start_Loc (s.pids 0)
```
</details>

<details><summary><code>contextAttnMistralExactFoldMG</code></summary>

```
/-- General boundary-masked sliding-window-softmax fold (the faithful kernel value).
Every key in `Fin S` contributes weight `exp(score j)` (sentinel kept), so the
denominator is always positive. -/
```
```lean
noncomputable def contextAttnMistralExactFoldMG
    (s : BlockState) (Q K V B_Start_Loc : RegionName)
    (sm_scale : ℝ) (rs hs BLK DM S bel sw : Nat) (idx : TileIndex [BLK, DM]) : ℝ :=
  let i := idx.1
  let d := idx.2.1
  let weight := fun j : Fin S => Real.exp (mistralScore s Q K B_Start_Loc sm_scale rs hs BLK DM S bel sw i j)
  let denom := Finset.univ.sum (fun j : Fin S => weight j)
  let numer := Finset.univ.sum (fun j : Fin S =>
    weight j * ctxVTileMG s V B_Start_Loc rs hs S DM bel (j, d, PUnit.unit))
  numer / denom
```
</details>

<details><summary><code>ctxMistralWindowG</code></summary>

```
/-- General kernel-decoded streamed window `S = block_mask·(start_m+1)·BLK`. -/
```
```lean
def ctxMistralWindowG (s : BlockState) (B_Seqlen : RegionName) (BLK : Nat) : Nat :=
  let sl := seqLen s B_Seqlen
  let bm := if BLK * s.pids 2 < sl then 1 else 0
  bm * (s.pids 2 + 1) * BLK
```
</details>

<details><summary><code>ctxMistralBel</code></summary>

```
/-- General k/v load boundary `bel = cur_batch_seq_len`. -/
```
```lean
def ctxMistralBel (s : BlockState) (B_Seqlen : RegionName) : Nat :=
  seqLen s B_Seqlen
```
</details>

<details><summary><code>mistralScore</code></summary>

```
/-- The per-cell sentinel-kept score: real scaled dot on active cells, the finite
`-1e9` sentinel on masked cells. -/
```
```lean
noncomputable def mistralScore
    (s : BlockState) (Q K B_Start_Loc : RegionName) (sm_scale : ℝ)
    (rs hs BLK DM S bel sw : Nat) (i : Fin BLK) (j : Fin S) : ℝ :=
  if mistralActive (s.pids 2 * BLK + i.val) sw j.val then
    sm_scale * Finset.univ.sum (fun e : Fin DM =>
      ctxQTileG s Q B_Start_Loc rs hs BLK DM (i, e, PUnit.unit)
        * ctxKTileMG s K B_Start_Loc rs hs S DM bel (j, e, PUnit.unit))
  else (-1e9 : ℝ)
```
</details>

<details><summary><code>ctxVTileMG</code></summary>

```
/-- General sequence-length-masked value tile. -/
```
```lean
noncomputable def ctxVTileMG
    (s : BlockState) (V B_Start_Loc : RegionName) (rs hs S DM bel : Nat) :
    TileIndex [S, DM] → ℝ :=
  fun (j, d, u) => if j.val < bel then ctxVTileG s V B_Start_Loc rs hs S DM (j, d, u) else 0
```
</details>

<details><summary><code>mistralActive</code></summary>

```
/-- The Mistral sliding-window+causal "active" predicate for cell `(gi, j)`:
`j ≤ gi` (causal) AND `gi ∸ sliding_window < j` (sliding window band). On inactive
cells the kernel keeps the finite `-1e9` sentinel score. -/
```
```lean
def mistralActive (gi sw j : Nat) : Prop := j ≤ gi ∧ gi - sw < j

instance (gi sw j : Nat) : Decidable (mistralActive gi sw j) := by
  unfold mistralActive; infer_instance
```
</details>

<details><summary><code>ctxQTileG</code></summary>

```
/-- General coordinate-faithful query tile (contiguous strides `(rs, hs, 1)`). -/
```
```lean
noncomputable def ctxQTileG
    (s : BlockState) (Q B_Start_Loc : RegionName) (rs hs BLK DM : Nat) :
    TileIndex [BLK, DM] → ℝ :=
  fun (i, e, _) =>
    s.readMem Q
      ((startLoc s B_Start_Loc + (s.pids 2 * BLK + i.val)) * rs
        + s.pids 1 * hs + e.val)
```
</details>

<details><summary><code>ctxKTileMG</code></summary>

```
/-- General sequence-length-masked key tile. -/
```
```lean
noncomputable def ctxKTileMG
    (s : BlockState) (K B_Start_Loc : RegionName) (rs hs S DM bel : Nat) :
    TileIndex [S, DM] → ℝ :=
  fun (j, e, u) => if j.val < bel then ctxKTileG s K B_Start_Loc rs hs S DM (j, e, u) else 0
```
</details>

<details><summary><code>ctxVTileG</code></summary>

```
/-- General coordinate-faithful value tile. -/
```
```lean
noncomputable def ctxVTileG
    (s : BlockState) (V B_Start_Loc : RegionName) (rs hs S DM : Nat) :
    TileIndex [S, DM] → ℝ :=
  fun (j, d, _) =>
    s.readMem V
      ((startLoc s B_Start_Loc + j.val) * rs + s.pids 1 * hs + d.val)
```
</details>

<details><summary><code>ctxKTileG</code></summary>

```
/-- General coordinate-faithful key tile. -/
```
```lean
noncomputable def ctxKTileG
    (s : BlockState) (K B_Start_Loc : RegionName) (rs hs S DM : Nat) :
    TileIndex [S, DM] → ℝ :=
  fun (j, e, _) =>
    s.readMem K
      ((startLoc s B_Start_Loc + j.val) * rs + s.pids 1 * hs + e.val)
```
</details>

## Also present (pinned special-case summaries)
- `context_attn_mistral_final_store_slice_compute_correct`
- `context_attn_mistral_genuine_output_summary`
