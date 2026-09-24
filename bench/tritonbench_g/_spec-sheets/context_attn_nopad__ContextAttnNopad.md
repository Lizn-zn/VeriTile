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
specification context_attn_nopad_output_summary_general
    (Q K V : RegionName) (B_Start_Loc B_Seqlen : Region .nat)
    (Out : RegionName) (sm_scale : ℝ) (rs hs BLK DM : Nat)
    (hBLK : 0 < BLK) (hDM : 0 < DM) (hDMrs : DM ≤ rs)
    (s : BlockState) (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes_without_Rounding
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

## Public theorem: `context_attn_nopad_io_correctness`

<details><summary>docstring</summary>

```
/-- **The `⊨[R]` streaming-metadata headline (wave-5) — the exemplar
consumer of `StreamMetaMasked3DKernelIO₃`, the context-attention trio's
skin.** For every rounding model `R`, the faithful `context_attn_nopad`
surface implements, on its metadata three-stream signature, the **honest
causal softmax** over the streamed `Q`/`K`/`V` tiles: every write-active
output lane `j = (i, e)` holds

`Σ_{jg ≤ pid₂·BLK+i} exp(sm_scale·raw jg)·V[jg,e] / Σ_{jg ≤ pid₂·BLK+i} exp(sm_scale·raw jg)`

(`contextAttnNopadIOSpec` — exactly the port's genuine closed form
`contextAttnNopadExactFoldMG` restated on the streams: the `float("-inf")`
causal mask genuinely drops future keys, the `K`/`V` boundary masks zero
out-of-sequence keys, and the in-loop-normalized `acc` needs no
post-divide). The two `.nat` slots enter only through the windows/masks
(`m 0 = B_Seqlen[pid₀]` the boundary, `m 1 = B_Start_Loc[pid₀]` the packed
row offset). The kernel has **zero rounding events** (`.nat` slot loads,
`other=0.0`-masked `.real` loads, `.real` in-loop arithmetic, untyped
terminal store), so the skin's boundary quantization degenerates: the
readback's `R.round .real` is the identity by the model's defining
`round_real`.

**Launch legality (the skin's genre novelty).** The triple is guarded by

`io.pre pid₂ m = (pid₂ < NT ∧ m 0 ≤ NT·BLK)`

— the port's documented **trusted boundary** (see the file docstring's
Scope section): the host launches
`grid = (batch, head, cdiv(max_input_len, BLOCK_M))` with `NT` the third
dimension and every `B_Seqlen[b] ≤ max_input_len ≤ NT·BLOCK_M`. The live
trip count `block_mask·(pid₂+1)` has no pid-free bound, so `pid₂ < NT` is
what makes the `T = NT`-step window citable at all — it is threaded into
both the safety walk (every live step `c ≤ pid₂ < NT` indexes `Fin NT`)
and the value bridge (`gi < (pid₂+1)·BLK ≤ NT·BLK` makes the causal window
extension inert).

**Hypothesis provenance** (all truth-forced, inherited from the exact
headline `context_attn_nopad_output_summary_general`): `0 < BLK`, `0 < DM`
(nonempty tiles); `DM ≤ rs` (output-offset injectivity — the contiguous
layout has `rs = H·DM ≥ DM`, so no open `hInj` side condition); `0 < NT`
(the host grid's third dimension is `cdiv(max_input_len, BLOCK_M) ≥ 1`;
also forced by the static `Q` stream's step-`0` read). The exact headline's
`hundef` is **not** a hypothesis here — the skin's Hoare triple carries the
`undef` pin itself.

Relation to the exact surface: the `Realizes_without_Rounding` headline
above is retained unchanged; this `⊨[R]` face restates the same causal
softmax on the streaming metadata skin, for every `R` at once. -/
```
</details>

**Statement:**
```lean
specification context_attn_nopad_io_correctness (R : RoundingModel)
    (Q K V : RegionName) (B_Start_Loc B_Seqlen : Region .nat) (Out : RegionName)
    (sm_scale : ℝ) (rs hs BLK DM NT : Nat)
    (hBLK : 0 < BLK) (hDM : 0 < DM) (hDMrs : DM ≤ rs) (hNT : 0 < NT) :
    contextAttnNopadIO Q K V B_Start_Loc B_Seqlen Out sm_scale rs hs BLK DM NT ⊨[R]
      fun _ _ pid₂ m xs ys zs j =>
        contextAttnNopadIOSpec BLK DM NT (m (⟨0, by omega⟩ : Fin 2)) hBLK hNT sm_scale
          pid₂ xs ys zs j
```

**Assumptions / layout contracts:**
- `hBLK : 0 < BLK`
- `hDM : 0 < DM`
- `hDMrs : DM ≤ rs`
- `hNT : 0 < NT`

**Closed-form spec defs (transitive):** `contextAttnNopadIO`, `contextAttnNopadIOSpec`, `context_attn_nopad_fwd_kernel_surface`, `ctxNopadMetaBuf`, `ctxNopadIOqT`, `ctxNopadIOkT`, `ctxNopadIOvT`

<details><summary><code>contextAttnNopadIO</code></summary>

```
/-- **Streaming metadata IO signature** of `context_attn_nopad` on the
metadata-parametrized three-stream fold skin (S1: online-softmax fold +
terminal masked store, 3-D pid grid `(cur_batch, cur_head, start_m)`).

The kernel's `forRangeDyn` trip count `block_mask·(start_m+1)·BLOCK_M/BLOCK_N`
grows with `pid₂`, so the walk has **no pid-free step bound**: `T := NT`, a
new `Nat` parameter (the host grid's third dimension
`cdiv(max_input_len, BLOCK_M)`), and the skin's launch-legality field is

`pre := pid₂ < NT ∧ B_Seqlen[pid₀] ≤ NT·BLOCK_M`

— exactly the port's documented **trusted boundary** (see the file
docstring's Scope section): the host launches
`grid = (batch, head, cdiv(max_input_len, BLOCK_M))`, so every real program
has `start_m < NT`, and every `B_Seqlen[b] ≤ max_input_len ≤ NT·BLOCK_M`.
The `⊨[R]` triple says nothing about launches outside this boundary.

Windows transcribe the kernel's pointer arithmetic exactly, with the loaded
slot vector `m` in place of the in-state metadata reads
(`m 0 = cur_batch_seq_len`, `m 1 = cur_batch_in_all_start_index`), at the
contiguous strides `(rs, hs, 1)` and `BLOCK_M = BLOCK_N = BLK`,
`BLOCK_DMODEL = DM`:

* `read1` (`Q`, the **static** stream — the window ignores `t`): lane
  `j = (i, e)` row-major over `[BLK, DM]` reads
  `(m 1 + (pid₂·BLK + i))·rs + pid₁·hs + e` (the `off_q` cell);
  `mask1` is the row guard `pid₂·BLK + i < m 0` (`offs_m < seq_len`).
* `read2` (`K`, slot-shifted by `t·BLK` **columns** per step): lane
  `j = (e, jL)` over `[DM, BLK]` reads `(m 1 + (t·BLK + jL))·rs + pid₁·hs + e`;
  `mask2` is `t·BLK + jL < m 0` — the **slot-eating step mask** that keeps
  the dead tail steps (`t·BLK ≥` the live window) unpinned.
* `read3` (`V`, mirror at `[BLK, DM]`): lane `j = (jL, e)` reads
  `(m 1 + (t·BLK + jL))·rs + pid₁·hs + e`; `mask3` is `t·BLK + jL < m 0`.
* `write` (`Out`, the terminal store): lane `j = (i, e)` writes
  `(m 1 + (pid₂·BLK + i))·rs + pid₁·hs + e`; `writeMask` is the row guard
  (the kernel's `offs_m < cur_batch_seq_len` store mask).

`outDType` is the `.real` default: the loop normalizes in place and the
terminal `tl.store` is untyped, so there is no quantization event. -/
```
```lean
def contextAttnNopadIO (Q K V : RegionName) (B_Start_Loc B_Seqlen : Region .nat)
    (Out : RegionName) (sm_scale : ℝ) (rs hs BLK DM NT : Nat) :
    StreamMetaMasked3DKernelIO₃ where
  kernel := context_attn_nopad_fwd_kernel_surface Q K V sm_scale B_Start_Loc B_Seqlen Out
    rs hs 1 rs hs 1 rs hs 1 rs hs 1 BLK DM BLK
  inp1 := Q
  inp2 := K
  inp3 := V
  out := Out
  nMeta := 2
  sty := fun _ => ChanTy.nat
  mbuf := ctxNopadMetaBuf B_Start_Loc B_Seqlen
  mwin := fun _ pid₀ _ _ => pid₀
  T := NT
  B1 := BLK * DM
  B2 := DM * BLK
  B3 := BLK * DM
  C := BLK * DM
  pre := fun _ _ pid₂ m => pid₂ < NT ∧ m (⟨0, by omega⟩ : Fin 2) ≤ NT * BLK
  read1 := fun _ pid₁ pid₂ m _ j =>
    (m (⟨1, by omega⟩ : Fin 2) + (pid₂ * BLK + j.val / DM)) * rs + pid₁ * hs + j.val % DM
  read2 := fun _ pid₁ _ m t j =>
    (m (⟨1, by omega⟩ : Fin 2) + (t.val * BLK + j.val % BLK)) * rs + pid₁ * hs + j.val / BLK
  read3 := fun _ pid₁ _ m t j =>
    (m (⟨1, by omega⟩ : Fin 2) + (t.val * BLK + j.val / DM)) * rs + pid₁ * hs + j.val % DM
  write := fun _ pid₁ pid₂ m j =>
    (m (⟨1, by omega⟩ : Fin 2) + (pid₂ * BLK + j.val / DM)) * rs + pid₁ * hs + j.val % DM
  mask1 := fun _ _ pid₂ m _ j => pid₂ * BLK + j.val / DM < m (⟨0, by omega⟩ : Fin 2)
  mask2 := fun _ _ _ m t j => t.val * BLK + j.val % BLK < m (⟨0, by omega⟩ : Fin 2)
  mask3 := fun _ _ _ m t j => t.val * BLK + j.val / DM < m (⟨0, by omega⟩ : Fin 2)
  writeMask := fun _ _ pid₂ m j => pid₂ * BLK + j.val / DM < m (⟨0, by omega⟩ : Fin 2)
```
</details>

<details><summary><code>contextAttnNopadIOSpec</code></summary>

```
/-- **The streamed closed form**: `contextAttnNopadExactFoldMG` — the honest
causal softmax `Σ_{jg ≤ gi} exp(sm_scale·raw)·v / Σ_{jg ≤ gi} exp(sm_scale·raw)`
with boundary-masked `K`/`V` — restated over the three streamed tiles on the
`pre`-legal global-key window `[0, NT·BLK)` (which covers the kernel's live
window `(pid₂+1)·BLK` for every legal launch; causality `jg ≤ gi` cuts the
extension to the same keys). Output lane `j = (i, e)` row-major over
`[BLK, DM]`. -/
```
```lean
noncomputable def contextAttnNopadIOSpec (BLK DM NT m0 : Nat) (hBLK : 0 < BLK)
    (hNT : 0 < NT) (sm_scale : ℝ) (pid₂ : Nat)
    (xs : Fin NT → Fin (BLK * DM) → ℝ) (ys : Fin NT → Fin (DM * BLK) → ℝ)
    (zs : Fin NT → Fin (BLK * DM) → ℝ) (j : Fin (BLK * DM)) : ℝ :=
  let i := (Lane2D.decode j).1
  let d := (Lane2D.decode j).2.1
  let gi := pid₂ * BLK + i.val
  let raw := fun jg : Fin (NT * BLK) =>
    Finset.univ.sum (fun e : Fin DM =>
      ctxNopadIOqT BLK DM NT hNT xs (i, e, PUnit.unit)
        * ctxNopadIOkT BLK DM NT m0 hBLK ys (jg, e, PUnit.unit))
  let weight := fun jg : Fin (NT * BLK) =>
    if jg.val ≤ gi then Real.exp (sm_scale * raw jg) else 0
  let denom := Finset.univ.sum (fun jg : Fin (NT * BLK) => weight jg)
  let numer := Finset.univ.sum (fun jg : Fin (NT * BLK) =>
    weight jg * ctxNopadIOvT BLK DM NT m0 hBLK zs (jg, d, PUnit.unit))
  numer / denom
```
</details>

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

<details><summary><code>ctxNopadMetaBuf</code></summary>

```
/-- Slot-region table of the two per-batch metadata slots, in the kernel's
own load order: slot `0` = `B_Seqlen` (`cur_batch_seq_len`), slot `1` =
`B_Start_Loc` (`cur_batch_in_all_start_index`). A shared def, never an
inline `match` in a window/spec position. -/
```
```lean
def ctxNopadMetaBuf (B_Start_Loc B_Seqlen : Region .nat) : Fin 2 → RegionName
  | ⟨0, _⟩ => B_Seqlen.cast
  | ⟨_ + 1, _⟩ => B_Start_Loc.cast
```
</details>

<details><summary><code>ctxNopadIOqT</code></summary>

```
/-- The `Q` tile read off the (static) first stream: the window ignores `t`,
so the step-`0` slice carries the whole `[BLK, DM]` tile. -/
```
```lean
noncomputable def ctxNopadIOqT (BLK DM NT : Nat) (hNT : 0 < NT)
    (xs : Fin NT → Fin (BLK * DM) → ℝ) : TileIndex [BLK, DM] → ℝ :=
  fun idx => xs ⟨0, hNT⟩ (Lane2D.encode idx)
```
</details>

<details><summary><code>ctxNopadIOkT</code></summary>

```
/-- The boundary-masked global `K` tile read off the second stream: global
key `jg` lives in step `jg / BLK`, block-local column `jg % BLK`, at stream
lane `(e, jg % BLK)`; keys at or beyond the sequence boundary `m0` are `0`
(the kernel's `other=0.0` load mask). -/
```
```lean
noncomputable def ctxNopadIOkT (BLK DM NT m0 : Nat) (hBLK : 0 < BLK)
    (ys : Fin NT → Fin (DM * BLK) → ℝ) : TileIndex [NT * BLK, DM] → ℝ :=
  fun idx =>
    if idx.1.val < m0 then
      ys ⟨idx.1.val / BLK, (Nat.div_lt_iff_lt_mul hBLK).mpr idx.1.isLt⟩
        (Lane2D.encode (idx.2.1, ⟨idx.1.val % BLK, Nat.mod_lt _ hBLK⟩, PUnit.unit))
    else 0
```
</details>

<details><summary><code>ctxNopadIOvT</code></summary>

```
/-- The boundary-masked global `V` tile read off the third stream: global
key `jg` at stream lane `(jg % BLK, e)`. -/
```
```lean
noncomputable def ctxNopadIOvT (BLK DM NT m0 : Nat) (hBLK : 0 < BLK)
    (zs : Fin NT → Fin (BLK * DM) → ℝ) : TileIndex [NT * BLK, DM] → ℝ :=
  fun idx =>
    if idx.1.val < m0 then
      zs ⟨idx.1.val / BLK, (Nat.div_lt_iff_lt_mul hBLK).mpr idx.1.isLt⟩
        (Lane2D.encode (⟨idx.1.val % BLK, Nat.mod_lt _ hBLK⟩, idx.2.1, PUnit.unit))
    else 0
```
</details>
