# Spec sheet — `bench/tritonbench_g/context_attn_nopad/ContextAttnNopad.lean`

**Python source:** `bench/tritonbench_g/context_attn_nopad/context_attn_nopad.py`

## Public theorem: `context_attn_nopad_output_summary_general`

<details><summary>docstring</summary>

```
/-- **General Public summary for `context_attn_nopad.py`.**

The full faithful `_fwd_kernel` surface (preLoop + streaming-softmax `forRangeDyn`
loop + masked store) *realizes* the genuine causal-softmax closed form
`ctxNopadGenuineOutValueG` — the boundary-masked causal-softmax fold of the loaded
Q/K/V memory — at every active output lane, at the **dimension-parameterized**
contiguous layout `(stride_*bs, stride_*h, stride_*d) = (rs, hs, 1)` with
`BLOCK_M = BLOCK_N = BLK`, `BLOCK_DMODEL = DM`. NOT a self-referential executed
value: the streaming `m_i`/`l_i`/`acc` recurrence is decoded statement-by-statement
and proven to collapse to the closed form. Side conditions: `0 < BLK`, `0 < DM`,
`DM ≤ rs` (output-offset injectivity; contiguous layout has `rs = H·DM ≥ DM`),
`hundef`. Instantiating `BLK = DM = 128`, `rs = 768`, `hs = 128` recovers the
concrete Python test shape. -/
```
</details>

**Statement:**
```lean
theorem context_attn_nopad_output_summary_general
    (Q K V : RegionName) (B_Start_Loc B_Seqlen : Region .nat)
    (Out : RegionName) (sm_scale : ℝ) (rs hs BLK DM : Nat)
    (hBLK : 0 < BLK) (hDM : 0 < DM) (hDMrs : DM ≤ rs)
    (s : BlockState) (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes
      (kernel := context_attn_nopad_fwd_kernel_surface Q K V sm_scale B_Start_Loc B_Seqlen Out
        rs hs 1 rs hs 1 rs hs 1 rs hs 1 BLK DM BLK)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLK, DM] => activeG s B_Seqlen BLK idx)
        (fun idx : TileIndex [BLK, DM] =>
          (Out, outOffsetG s B_Start_Loc rs hs BLK DM idx)))
      (expected := fun idx : TileIndex [BLK, DM] =>
        ctxNopadGenuineOutValueG s Q K V B_Start_Loc B_Seqlen sm_scale rs hs BLK DM idx)
```

**Assumptions / layout contracts:**
- `hBLK : 0 < BLK`
- `hDM : 0 < DM`
- `hDMrs : DM ≤ rs`
- `hundef : ∀ rg o, s.undef rg o = 0`
- `kernel : = context_attn_nopad_fwd_kernel_surface Q K V sm_scale B_Start_Loc B_Seqlen Out
        rs hs 1 rs hs 1 rs hs 1 rs hs 1 BLK DM BLK`
- `initialState : = s`
- `fun idx : TileIndex [BLK, DM] => activeG s B_Seqlen BLK idx`
- `fun idx : TileIndex [BLK, DM] =>
          (Out, outOffsetG s B_Start_Loc rs hs BLK DM idx)`
- `expected : = fun idx : TileIndex [BLK, DM] =>
        ctxNopadGenuineOutValueG s Q K V B_Start_Loc B_Seqlen sm_scale rs hs BLK DM idx`

**Closed-form spec defs (transitive):** `context_attn_nopad_fwd_kernel_surface`, `activeG`, `outOffsetG`, `ctxNopadGenuineOutValueG`, `seqLen`, `startLoc`, `contextAttnNopadExactFoldMG`, `ctxNopadWindowG`, `ctxNopadBel`, `ctxQTileG`, `ctxKTileMG`, `ctxVTileMG`, `ctxKTileG`, `ctxVTileG`

<details><summary><code>context_attn_nopad_fwd_kernel_surface</code></summary>

```
/-- Faithful DSL port of `context_attn_nopad.py`'s `_fwd_kernel`. -/
```
```lean
def context_attn_nopad_fwd_kernel_surface
    (Q K V : RegionName) (sm_scale : ℝ)
    (B_Start_Loc B_Seqlen : Region .nat) (Out : RegionName)
    (stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd
      stride_obs stride_oh stride_od
      BLOCK_M BLOCK_DMODEL BLOCK_N : Nat) :
    ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  start_m = tl.program_id(2)

  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

  block_start_loc = $(BLOCK_M) * start_m

  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  off_q = (cur_batch_in_all_start_index + offs_m[:, None]) * $(stride_qbs) +
    cur_head * $(stride_qh) + offs_d[None, :] * $(stride_qd)
  off_k = offs_n[None, :] * $(stride_kbs) + cur_head * $(stride_kh) +
    offs_d[:, None] * $(stride_kd)
  off_v = offs_n[:, None] * $(stride_vbs) + cur_head * $(stride_vh) +
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
      mask=(start_n + offs_n[None, :]) < cur_batch_seq_len, other=0.0)

    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, k)
    qk *= $((sm_scale : ℝ))
    qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))

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
    acc = acc * acc_scale[:, None]
    v = tl.load(v_ptrs + (cur_batch_in_all_start_index + start_n) * $(stride_vbs),
      mask=(start_n + offs_n[:, None]) < cur_batch_seq_len, other=0.0)

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

<details><summary><code>activeG</code></summary>

```
/-- General active-output predicate. -/
```
```lean
def activeG (s : BlockState) (B_Seqlen : RegionName) (BLK : Nat)
    (idx : TileIndex [BLK, DM]) : Prop :=
  s.pids 2 * BLK + idx.1.val < seqLen s B_Seqlen
```
</details>

<details><summary><code>outOffsetG</code></summary>

```
/-- General output offset (contiguous layout, head stride `hs`). -/
```
```lean
def outOffsetG
    (s : BlockState) (B_Start_Loc : RegionName)
    (rs hs BLK DM : Nat) (idx : TileIndex [BLK, DM]) : Nat :=
  (startLoc s B_Start_Loc + (s.pids 2 * BLK + idx.1.val)) * rs
    + s.pids 1 * hs + idx.2.1.val
```
</details>

<details><summary><code>ctxNopadGenuineOutValueG</code></summary>

```
/-- General genuine closed-form output value (boundary-masked causal-softmax fold). -/
```
```lean
noncomputable def ctxNopadGenuineOutValueG
    (s : BlockState) (Q K V B_Start_Loc B_Seqlen : RegionName)
    (sm_scale : ℝ) (rs hs BLK DM : Nat) (idx : TileIndex [BLK, DM]) : ℝ :=
  contextAttnNopadExactFoldMG s Q K V B_Start_Loc sm_scale rs hs BLK DM
    (ctxNopadWindowG s B_Seqlen BLK) (ctxNopadBel s B_Seqlen) idx
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

<details><summary><code>contextAttnNopadExactFoldMG</code></summary>

```
/-- General boundary-masked causal-softmax fold (the faithful kernel value). -/
```
```lean
noncomputable def contextAttnNopadExactFoldMG
    (s : BlockState) (Q K V B_Start_Loc : RegionName)
    (sm_scale : ℝ) (rs hs BLK DM S bel : Nat) (idx : TileIndex [BLK, DM]) : ℝ :=
  let i := idx.1
  let d := idx.2.1
  let gi := s.pids 2 * BLK + i.val
  let raw := fun j : Fin S =>
    Finset.univ.sum (fun e : Fin DM =>
      ctxQTileG s Q B_Start_Loc rs hs BLK DM (i, e, PUnit.unit)
        * ctxKTileMG s K B_Start_Loc rs hs S DM bel (j, e, PUnit.unit))
  let weight := fun j : Fin S =>
    if j.val ≤ gi then Real.exp (sm_scale * raw j) else 0
  let denom := Finset.univ.sum (fun j : Fin S => weight j)
  let numer := Finset.univ.sum (fun j : Fin S =>
    weight j * ctxVTileMG s V B_Start_Loc rs hs S DM bel (j, d, PUnit.unit))
  numer / denom
```
</details>

<details><summary><code>ctxNopadWindowG</code></summary>

```
/-- General kernel-decoded streamed window `S = block_mask·(start_m+1)·BLK`. -/
```
```lean
def ctxNopadWindowG (s : BlockState) (B_Seqlen : RegionName) (BLK : Nat) : Nat :=
  let sl := seqLen s B_Seqlen
  let bm := if BLK * s.pids 2 < sl then 1 else 0
  bm * (s.pids 2 + 1) * BLK
```
</details>

<details><summary><code>ctxNopadBel</code></summary>

```
/-- Kernel-decoded k/v load boundary `bel = cur_batch_seq_len`. -/
```
```lean
def ctxNopadBel (s : BlockState) (B_Seqlen : RegionName) : Nat :=
  seqLen s B_Seqlen
```
</details>

<details><summary><code>ctxQTileG</code></summary>

```
/-- General coordinate-faithful query tile: row `i` is the global packed row
`B_Start_Loc[cur_batch] + start_m·BLK + i`, channel `e`, contiguous strides
`(rs, hs, 1)` (= `(H·DM, DM, 1)`). -/
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

## Also present (pinned special-case summaries)
- `context_attn_nopad_final_store_slice_compute_correct`
