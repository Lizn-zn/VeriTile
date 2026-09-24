# Spec sheet — `bench/tritonbench_g/chunk_gated_attention/ChunkGatedAttention.lean`

**Python source:** `bench/tritonbench_g/chunk_gated_attention/chunk_gated_attention.py`

## Public theorem: `chunk_gated_attention_cum_slice_output_summary_general`

<details><summary>docstring</summary>

```
/-- **SCOPE — this covers `chunk_gated_abc_fwd_kernel_cum` only.** The headline
is a single `Realizes` fact about the hand-cut slice
`chunk_gated_attention_cum_compute_slice`, which reproduces the whole body of the
`fwd_pre` cumsum kernel (masked tile load, lower-triangular `tl.dot`, masked
store) at symbolic `T S BT BS` and strides, with the launch's own pid assignment
`i_s, i_t, i_bh = pid 0, pid 1, pid 2`.

`chunk_gated_abc_fwd_kernel_h` — the gated chunk recurrence, which is the
substance of this benchmark — is now covered **at the level of one loop body per
`GATEK` branch** (conjuncts 2 and 3). Its two *former* conjuncts were masked
memcpys (identical load and store addresses) whose entire content was the
`hBufState` / `hBufFinal` assumptions; those stay out of the headline and remain
labelled `*_memcpy_transports_hClosed` above. What replaces them computes: each
new conjunct realizes `hClosed (m+1)` — gate-times-previous plus the gated
`b_k @ b_v` accumulation, read off the launched `K`, `V`, `G` — so `GATEK` is now
exercised by the headline in **both** settings. `USE_INITIAL_STATE` flows through
`hClosed`'s seed term; `STORE_FINAL_STATE` is still exercised by no conjunct.

The recurrence's **base case** is conjunct 4 (`hClosed 0 = hSeed`) and its **step**
is `hClosed_succ`, used inside conjuncts 2 and 3. The **induction** joining them
is `chunk_gated_attention_h_state_carry_fold`, which is *not* a conjunct here:
this headline keeps the per-chunk shape, so `hPrevV`/`hPrevK` still constrain
`HPrev` while the conjuncts write `HOut`. `HPrev`/`HOut` are fiction regions
(the Python `b_h` is a register) — see the step-slice section's scope list, which
also records that the step slices read `b_k`/`b_v`/`b_g`/`b_gn` unmasked.

Honest side conditions: offset injectivity of both store footprints (`hCumInj`,
`hStateInj`) — no-aliasing hypotheses on the strides, which hold for the bench
strides `(s_s_t, s_s_d) = (S, 1)` and `(VSize, 1)`; and the two carry invariants
`hPrevV`/`hPrevK`, which are *assumptions* carrying the cross-chunk fold. No
dimension is pinned. No region-distinctness hypothesis is needed: every slice
performs its loads before its single store and every `expected` reads the initial
state, so aliasing cannot falsify a conjunct.

(The `cum_slice` in this declaration's name is historical — it now bundles the
gated-recurrence bodies too. The name is kept because the checked-in
`proof_gap_manifest.tsv` keys on it.) -/
```
</details>

**Statement:**
```lean
specification chunk_gated_attention_cum_slice_output_summary_general
    (SReg GCum K V G H0 HPrev HOut : RegionName) (USE_INITIAL_STATE : Bool)
    (s_s_h s_s_t s_s_d T S BT BS
      m s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BK BV : Nat)
    (s : BlockState)
    (hCumInj : Function.Injective
      (fun idx : TileIndex [BT, BS] => tileOffset s s_s_h s_s_t s_s_d BT BS idx))
    (hStateInj : Function.Injective
      (fun idx : TileIndex [BK, BV] => finalStateOffset s KSize VSize BK BV idx))
    (hPrevV : ∀ idx : TileIndex [BK, BV],
      s.readMem HPrev (finalStateOffset s KSize VSize BK BV idx)
        = hClosed s K V G H0 Bool.false USE_INITIAL_STATE
            s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV m idx)
    (hPrevK : ∀ idx : TileIndex [BK, BV],
      s.readMem HPrev (finalStateOffset s KSize VSize BK BV idx)
        = hClosed s K V G H0 Bool.true USE_INITIAL_STATE
            s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV m idx) :
    -- (1) the `fwd_pre` cumsum body realizes the causal intra-chunk cumsum
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := chunk_gated_attention_cum_compute_slice SReg GCum s_s_h s_s_t
        s_s_d T S BT BS)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BT, BS] => cumSurfaceActive s T S BT BS idx)
        (fun idx : TileIndex [BT, BS] =>
          (GCum, tileOffset s s_s_h s_s_t s_s_d BT BS idx)))
      (expected := fun idx : TileIndex [BT, BS] =>
        cumComputeStoreValue s SReg s_s_h s_s_t s_s_d T S BT BS idx)) ∧
    -- (2) one `GATEK = false` recurrence body realizes `hClosed (m+1)`
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := chunk_gated_attention_h_step_gatev_slice K V G HPrev HOut
        m s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV)
      (initialState := s)
      (write := fun idx : TileIndex [BK, BV] =>
        some (HOut, finalStateOffset s KSize VSize BK BV idx))
      (expected := fun idx : TileIndex [BK, BV] =>
        hClosed s K V G H0 Bool.false USE_INITIAL_STATE
          s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV (m + 1) idx)) ∧
    -- (3) one `GATEK = true` recurrence body realizes `hClosed (m+1)`
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := chunk_gated_attention_h_step_gatek_slice K V G HPrev HOut
        m s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV)
      (initialState := s)
      (write := fun idx : TileIndex [BK, BV] =>
        some (HOut, finalStateOffset s KSize VSize BK BV idx))
      (expected := fun idx : TileIndex [BK, BV] =>
        hClosed s K V G H0 Bool.true USE_INITIAL_STATE
          s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV (m + 1) idx)) ∧
    -- (4) the recurrence's base case
    (∀ (GATEK : Bool) (idx : TileIndex [BK, BV]),
      hClosed s K V G H0 GATEK USE_INITIAL_STATE
          s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV 0 idx
        = hSeed s H0 USE_INITIAL_STATE KSize VSize BK BV idx)
```

**Assumptions / layout contracts:**
- `hCumInj : Function.Injective
      (fun idx : TileIndex [BT, BS] => tileOffset s s_s_h s_s_t s_s_d BT BS idx)`
- `hStateInj : Function.Injective
      (fun idx : TileIndex [BK, BV] => finalStateOffset s KSize VSize BK BV idx)`
- `hPrevV : ∀ idx : TileIndex [BK, BV],
      s.readMem HPrev (finalStateOffset s KSize VSize BK BV idx)
        = hClosed s K V G H0 Bool.false USE_INITIAL_STATE
            s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV m idx`
- `hPrevK : ∀ idx : TileIndex [BK, BV],
      s.readMem HPrev (finalStateOffset s KSize VSize BK BV idx)
        = hClosed s K V G H0 Bool.true USE_INITIAL_STATE
            s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV m idx`

**Closed-form spec defs (transitive):** `tileOffset`, `finalStateOffset`, `hClosed`, `chunk_gated_attention_cum_compute_slice`, `cumSurfaceActive`, `cumComputeStoreValue`, `chunk_gated_attention_h_step_gatev_slice`, `chunk_gated_attention_h_step_gatek_slice`, `hSeed`, `cumSurfaceTIndex`, `cumSurfaceSIndex`, `kIndexFinal`, `vIndexFinal`, `hGate`, `hStepTerm`, `lowerTriTile`, `sourceTile`, `chunkLastTime`, `h0Elem`, `kIndexState`, `vIndexState`, `gnkElem`, `gnvElem`, `ktElem`, `tvElem`

<details><summary><code>tileOffset</code></summary>

```
/-- Flat address of tile lane `idx` in the `[T, S]` block-pointer footprint
`base = R + i_bh * s_s_h`, `strides = (s_s_t, s_s_d)`, with `i_bh = pid 2`. -/
```
```lean
def tileOffset (s : BlockState) (s_s_h s_s_t s_s_d BT BS : Nat)
    (idx : TileIndex [BT, BS]) : Nat :=
  s.pids 2 * s_s_h + cumSurfaceTIndex s BT idx.1 * s_s_t +
    cumSurfaceSIndex s BS idx.2.1 * s_s_d
```
</details>

<details><summary><code>finalStateOffset</code></summary>

```lean
def finalStateOffset (s : BlockState) (KSize VSize BK BV : Nat)
    (idx : TileIndex [BK, BV]) : Nat :=
  s.pids 2 * KSize * VSize +
    kIndexFinal s BK idx.1 * VSize + vIndexFinal s BV idx.2.1
```
</details>

<details><summary><code>hClosed</code></summary>

```
/-- **Genuine closed form** for the folded gated-recurrence state at chunk `m`
(the value the kernel stores into `h` at loop row `m`). Standalone over the
input regions `K`, `V`, `G`, `H0`. -/
```
```lean
noncomputable def hClosed
    (s : BlockState) (K V G H0 : RegionName)
    (GATEK USE_INITIAL_STATE : Bool)
    (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV m : Nat)
    (idx : TileIndex [BK, BV]) : ℝ :=
  hSeed s H0 USE_INITIAL_STATE KSize VSize BK BV idx *
      (∏ j ∈ Finset.range m,
        hGate s G GATEK s_k_h s_k_d s_v_h s_v_d KSize VSize BT BK BV j idx) +
    ∑ t ∈ Finset.range m,
      hStepTerm s K V G GATEK s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d
          KSize VSize BT BK BV t idx *
        (∏ j ∈ Finset.Ico (t + 1) m,
          hGate s G GATEK s_k_h s_k_d s_v_h s_v_d KSize VSize BT BK BV j idx)
```
</details>

<details><summary><code>chunk_gated_attention_cum_compute_slice</code></summary>

```
/-! ## Computed cumulative-normalizer slice

The `fwd_pre` Python path computes `b_o = tl.dot(m_s, b_s)` with a
lower-triangular mask before storing. This slice proves that computation and
writeback directly, rather than starting from a precomputed `BC` tile. -/
```
```lean
def chunk_gated_attention_cum_compute_slice
    (SReg Z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat) :
    ComputeKernel := triton {
  i_s = tl.program_id(0)
  i_t = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_t = i_t * $(BT) + tl.arange(0, $(BT))
  offs_s = i_s * $(BS) + tl.arange(0, $(BS))
  o_i = tl.arange(0, $(BT))
  m_s = tl.where(o_i[:, None] >= o_i[None, :], 1.0, 0.0)
  mask = (offs_t[:, None] < $(T)) & (offs_s[None, :] < $(S))
  b_s = tl.load(SReg + i_bh * $(s_s_h) + offs_t[:, None] * $(s_s_t) +
      offs_s[None, :] * $(s_s_d), mask=mask, other=0.0)
  b_o = tl.dot(m_s, b_s, allow_tf32=false)
  tl.store(Z + i_bh * $(s_s_h) + offs_t[:, None] * $(s_s_t) +
      offs_s[None, :] * $(s_s_d), b_o, mask=mask)
}
```
</details>

<details><summary><code>cumSurfaceActive</code></summary>

```
/-- The `boundary_check=(0, 1)` in-range predicate of the `[BT, BS]` tile. -/
```
```lean
def cumSurfaceActive (s : BlockState) (T S BT BS : Nat)
    (idx : TileIndex [BT, BS]) : Prop :=
  cumSurfaceTIndex s BT idx.1 < T ∧ cumSurfaceSIndex s BS idx.2.1 < S
```
</details>

<details><summary><code>cumComputeStoreValue</code></summary>

```lean
noncomputable def cumComputeStoreValue
    (s : BlockState) (SReg : RegionName)
    (s_s_h s_s_t s_s_d T S BT BS : Nat)
    (idx : TileIndex [BT, BS]) : ℝ :=
  WithBot.unbotD 0
    ((Tile.dot [] (lowerTriTile BT)
      (sourceTile s SReg s_s_h s_s_t s_s_d T S BT BS)).data
        (idx.1, idx.2.1, PUnit.unit))
```
</details>

<details><summary><code>chunk_gated_attention_h_step_gatev_slice</code></summary>

```
/-- `GATEK = false` branch: the gate rides the **value columns**, and `b_v` is
pre-gated by `exp(b_gn[v] - b_g[t,v])`. -/
```
```lean
def chunk_gated_attention_h_step_gatev_slice
    (K V G HPrev HOut : RegionName)
    (m s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV : Nat) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = tl.arange(0, $(BK))
  offs_v = tl.arange(0, $(BV))
  offs_t = tl.arange(0, $(BT))
  b_h = tl.load(HPrev + i_bh * $(KSize) * $(VSize) +
    (i_k * $(BK) + offs_k[:, None]) * $(VSize) + (i_v * $(BV) + offs_v[None, :]))
  b_gn = tl.load(G + i_bh * $(s_v_h) +
    ($(chunkLastTime BT m) * $(VSize) + (i_v * $(BV) + offs_v)) * $(s_v_d))
  b_g = tl.load(G + i_bh * $(s_v_h) + ($(m) * $(BT) + offs_t[:, None]) * $(s_v_t) +
    (i_v * $(BV) + offs_v[None, :]) * $(s_v_d))
  b_k = tl.load(K + i_bh * $(s_k_h) + (i_k * $(BK) + offs_k[:, None]) * $(s_k_d) +
    ($(m) * $(BT) + offs_t[None, :]) * $(s_k_t))
  b_v = tl.load(V + i_bh * $(s_v_h) + ($(m) * $(BT) + offs_t[:, None]) * $(s_v_t) +
    (i_v * $(BV) + offs_v[None, :]) * $(s_v_d))
  b_h = b_h * tl.exp(b_gn)[None, :]
  b_v = b_v * tl.exp(b_gn[None, :] - b_g)
  b_h = b_h + tl.dot(b_k, b_v, allow_tf32=false)
  tl.store(HOut + i_bh * $(KSize) * $(VSize) +
    (i_k * $(BK) + offs_k[:, None]) * $(VSize) + (i_v * $(BV) + offs_v[None, :]),
    (b_h).to(HOut.dtype.element_ty))
}
```
</details>

<details><summary><code>chunk_gated_attention_h_step_gatek_slice</code></summary>

```
/-- `GATEK = true` branch: the gate rides the **key rows**, and `b_k` is
pre-gated by `exp(b_gn[k] - b_g[k,t])`. -/
```
```lean
def chunk_gated_attention_h_step_gatek_slice
    (K V G HPrev HOut : RegionName)
    (m s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV : Nat) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = tl.arange(0, $(BK))
  offs_v = tl.arange(0, $(BV))
  offs_t = tl.arange(0, $(BT))
  b_h = tl.load(HPrev + i_bh * $(KSize) * $(VSize) +
    (i_k * $(BK) + offs_k[:, None]) * $(VSize) + (i_v * $(BV) + offs_v[None, :]))
  b_gn = tl.load(G + i_bh * $(s_k_h) +
    ($(chunkLastTime BT m) * $(KSize) + (i_k * $(BK) + offs_k)) * $(s_k_d))
  b_g = tl.load(G + i_bh * $(s_k_h) + (i_k * $(BK) + offs_k[:, None]) * $(s_k_d) +
    ($(m) * $(BT) + offs_t[None, :]) * $(s_k_t))
  b_k = tl.load(K + i_bh * $(s_k_h) + (i_k * $(BK) + offs_k[:, None]) * $(s_k_d) +
    ($(m) * $(BT) + offs_t[None, :]) * $(s_k_t))
  b_v = tl.load(V + i_bh * $(s_v_h) + ($(m) * $(BT) + offs_t[:, None]) * $(s_v_t) +
    (i_v * $(BV) + offs_v[None, :]) * $(s_v_d))
  b_h = b_h * tl.exp(b_gn)[:, None]
  b_k = b_k * tl.exp(b_gn[:, None] - b_g)
  b_h = b_h + tl.dot(b_k, b_v, allow_tf32=false)
  tl.store(HOut + i_bh * $(KSize) * $(VSize) +
    (i_k * $(BK) + offs_k[:, None]) * $(VSize) + (i_v * $(BV) + offs_v[None, :]),
    (b_h).to(HOut.dtype.element_ty))
}
```
</details>

<details><summary><code>hSeed</code></summary>

```
/-- Seed state `b_h^(0)[k,v]`: `h0[k,v]` when `USE_INITIAL_STATE`, else `0`. -/
```
```lean
noncomputable def hSeed
    (s : BlockState) (H0 : RegionName) (USE_INITIAL_STATE : Bool)
    (KSize VSize BK BV : Nat) (idx : TileIndex [BK, BV]) : ℝ :=
  if USE_INITIAL_STATE then
    h0Elem s H0 KSize VSize (kIndexState s BK idx.1) (vIndexState s BV idx.2.1)
  else 0
```
</details>

<details><summary><code>cumSurfaceTIndex</code></summary>

```
/-- Global time row of tile lane `i`: `i_t * BT + i` with `i_t = pid 1`. -/
```
```lean
def cumSurfaceTIndex (s : BlockState) (BT : Nat) (i : Fin BT) : Nat :=
  s.pids 1 * BT + i.val
```
</details>

<details><summary><code>cumSurfaceSIndex</code></summary>

```
/-- Global feature column of tile lane `j`: `i_s * BS + j` with `i_s = pid 0`. -/
```
```lean
def cumSurfaceSIndex (s : BlockState) (BS : Nat) (j : Fin BS) : Nat :=
  s.pids 0 * BS + j.val
```
</details>

<details><summary><code>kIndexFinal</code></summary>

```lean
def kIndexFinal (s : BlockState) (BK : Nat) (i : Fin BK) : Nat :=
  s.pids 1 * BK + i.val
```
</details>

<details><summary><code>vIndexFinal</code></summary>

```lean
def vIndexFinal (s : BlockState) (BV : Nat) (j : Fin BV) : Nat :=
  s.pids 0 * BV + j.val
```
</details>

<details><summary><code>hGate</code></summary>

```
/-- Per-chunk gate factor `G_m[k,v]` at the bench shape. `GATEK` ⇒ per-key-row
`exp(b_gn_m[k])`; otherwise per-value-column `exp(b_gn_m[v])`. `b_gn` is the
last-row (`t = BT-1`) cumulative gate of chunk `m`. -/
```
```lean
noncomputable def hGate
    (s : BlockState) (G : RegionName) (GATEK : Bool)
    (s_k_h s_k_d s_v_h s_v_d KSize VSize BT BK BV m : Nat)
    (idx : TileIndex [BK, BV]) : ℝ :=
  if GATEK then
    Real.exp (gnkElem s G s_k_h s_k_d KSize (chunkLastTime BT m)
      (kIndexState s BK idx.1))
  else
    Real.exp (gnvElem s G s_v_h s_v_d VSize (chunkLastTime BT m)
      (vIndexState s BV idx.2.1))
```
</details>

<details><summary><code>hStepTerm</code></summary>

```
/-- Per-chunk gated matmul accumulation `S_m[k,v] = (gated b_k) @ (gated b_v)`,
summed over the `BT` intra-chunk time lanes. `GATEK` gates `b_k` by
`exp(b_gn_m[k] - b_g_m[k,t])`; otherwise gates `b_v` by `exp(b_gn_m[v] - b_g_m[t,v])`. -/
```
```lean
noncomputable def hStepTerm
    (s : BlockState) (K V G : RegionName) (GATEK : Bool)
    (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV m : Nat)
    (idx : TileIndex [BK, BV]) : ℝ :=
  ∑ t : Fin BT,
    let kVal := ktElem s K s_k_h s_k_t s_k_d
      (kIndexState s BK idx.1) (m * BT + t.val)
    let vVal := tvElem s V s_v_h s_v_t s_v_d
      (m * BT + t.val) (vIndexState s BV idx.2.1)
    if GATEK then
      let gnVal := gnkElem s G s_k_h s_k_d KSize (chunkLastTime BT m)
        (kIndexState s BK idx.1)
      let gVal := ktElem s G s_k_h s_k_t s_k_d
        (kIndexState s BK idx.1) (m * BT + t.val)
      (kVal * Real.exp (gnVal - gVal)) * vVal
    else
      let gnVal := gnvElem s G s_v_h s_v_d VSize (chunkLastTime BT m)
        (vIndexState s BV idx.2.1)
      let gVal := tvElem s G s_v_h s_v_t s_v_d
        (m * BT + t.val) (vIndexState s BV idx.2.1)
      kVal * (vVal * Real.exp (gnVal - gVal))
```
</details>

<details><summary><code>lowerTriTile</code></summary>

```lean
noncomputable def lowerTriTile (BT : Nat) : Tile .real [BT, BT] :=
  { data := fun idx =>
      if idx.1.val >= idx.2.1.val then some (1.0 : ℝ) else some (0.0 : ℝ) }
```
</details>

<details><summary><code>sourceTile</code></summary>

```lean
noncomputable def sourceTile
    (s : BlockState) (SReg : RegionName)
    (s_s_h s_s_t s_s_d T S BT BS : Nat) :
    Tile .real [BT, BS] :=
  { data := fun idx =>
      if cumSurfaceActive s T S BT BS idx then
        some (s.readMem SReg (tileOffset s s_s_h s_s_t s_s_d BT BS idx))
      else some (0.0 : ℝ) }
```
</details>

<details><summary><code>chunkLastTime</code></summary>

```
/-- Global time index of the **last lane of chunk `m`** (`i_t*BT + BT - 1`) —
the row whose cumulative gate `b_gn` normalizes the whole chunk. -/
```
```lean
def chunkLastTime (BT m : Nat) : Nat := m * BT + BT - 1
```
</details>

<details><summary><code>h0Elem</code></summary>

```
/-- Initial-state element `h0[i_bh][k, v]`: mirrors the `h0` block pointer
(`base = h0 + i_bh*K*V`, `shape = (K, V)`, `strides = (V, 1)`). -/
```
```lean
noncomputable def h0Elem (s : BlockState) (H0 : RegionName)
    (KSize VSize k v : Nat) : ℝ :=
  s.readMem H0 (s.pids 2 * KSize * VSize + k * VSize + v)
```
</details>

<details><summary><code>kIndexState</code></summary>

```lean
def kIndexState (s : BlockState) (BK : Nat) (i : Fin BK) : Nat :=
  s.pids 1 * BK + i.val
```
</details>

<details><summary><code>vIndexState</code></summary>

```lean
def vIndexState (s : BlockState) (BV : Nat) (j : Fin BV) : Nat :=
  s.pids 0 * BV + j.val
```
</details>

<details><summary><code>gnkElem</code></summary>

```
/-- Last-lane cumulative gate `b_gn[k]` under `GATEK`: lane `tLast*KSize + k` of
`p_gn`'s flattened `(T*K,)` view of `G` at batch-head `s.pids 2`, at the block
pointer's own element stride `s_k_d` (`chunk_gated_attention.py:90`). -/
```
```lean
noncomputable def gnkElem (s : BlockState) (G : RegionName)
    (s_k_h s_k_d KSize tLast k : Nat) : ℝ :=
  s.readMem G (s.pids 2 * s_k_h + (tLast * KSize + k) * s_k_d)
```
</details>

<details><summary><code>gnvElem</code></summary>

```
/-- Last-lane cumulative gate `b_gn[v]` under `¬GATEK`: lane `tLast*VSize + v` of
`p_gn`'s flattened `(T*V,)` view of `G` at batch-head `s.pids 2`, at the block
pointer's own element stride `s_v_d` (`chunk_gated_attention.py:100`). -/
```
```lean
noncomputable def gnvElem (s : BlockState) (G : RegionName)
    (s_v_h s_v_d VSize tLast v : Nat) : ℝ :=
  s.readMem G (s.pids 2 * s_v_h + (tLast * VSize + v) * s_v_d)
```
</details>

<details><summary><code>ktElem</code></summary>

```
/-- `[k, t]`-layout element `R[i_bh][k, t]` at batch-head `s.pids 2`: mirrors
the `p_k`-shaped block pointers (`base = R + i_bh*s_k_h`, `shape = (K, T)`,
`strides = (s_k_d, s_k_t)`) — used for `K` itself and, under `GATEK`, for the
per-time gate read `b_g` on `G`. -/
```
```lean
noncomputable def ktElem (s : BlockState) (R : RegionName)
    (s_k_h s_k_t s_k_d k t : Nat) : ℝ :=
  s.readMem R (s.pids 2 * s_k_h + k * s_k_d + t * s_k_t)
```
</details>

<details><summary><code>tvElem</code></summary>

```
/-- `[t, v]`-layout element `R[i_bh][t, v]` at batch-head `s.pids 2`: mirrors
the `p_v`-shaped block pointers (`base = R + i_bh*s_v_h`, `shape = (T, V)`,
`strides = (s_v_t, s_v_d)`) — used for `V` itself and, under `¬GATEK`, for the
per-time gate read `b_g` on `G`. -/
```
```lean
noncomputable def tvElem (s : BlockState) (R : RegionName)
    (s_v_h s_v_t s_v_d t v : Nat) : ℝ :=
  s.readMem R (s.pids 2 * s_v_h + t * s_v_t + v * s_v_d)
```
</details>

## Public theorem: `chunk_gated_attention_state_stores_io_correctness`

<details><summary>docstring</summary>

```
/-- **The headline on the IO surface** for `chunk_gated_attention.py`'s two masked
state writebacks (the per-chunk `H` store and the `STORE_FINAL_STATE` `Ht` store):
for every disjoint flat placement of the source and destination buffers, every
program coordinate whose active lanes are in bounds, and every launch state whose
source block holds `xs` at the active lanes, each slice terminates, every active lane
of the destination holds `xs idx`, and every other memory cell is unchanged.

Both are masked `[BK, BV]` memcpys whose single address is built from all three
program axes (`i_v`, `i_k`, `i_bh`) — no new library surface. Dimension-general in
`i_t`, the three `s_h_*` strides, `KSize`, `VSize`, `BK` and `BV`. Honest
side-condition: address injectivity at every program coordinate, the same hypothesis
the per-write-map summaries take. -/
```
</details>

**Statement:**
```lean
specification chunk_gated_attention_state_stores_io_correctness
    (BH H BHFinal Ht : RegionName)
    (i_t s_h_h s_h_t s_h_d KSize VSize BK BV : Nat)
    (hInj1 : ∀ p₀ p₁ p₂ : Nat, Function.Injective
      (fun idx : TileIndex [BK, BV] =>
        p₂ * s_h_h + i_t * KSize * VSize + (p₁ * BK + idx.1.val) * s_h_t
          + (p₀ * BV + idx.2.1.val) * s_h_d))
    (hInj2 : ∀ p₀ p₁ p₂ : Nat, Function.Injective
      (fun idx : TileIndex [BK, BV] =>
        p₂ * KSize * VSize + (p₁ * BK + idx.1.val) * VSize
          + (p₀ * BV + idx.2.1.val))) :
    (h_stateIO BH H i_t s_h_h s_h_t s_h_d KSize VSize BK BV
      ⊨ fun _p₀ _p₁ xs idx => xs idx) ∧
    (final_stateIO BHFinal Ht KSize VSize BK BV
      ⊨ fun _p₀ _p₁ xs idx => xs idx)
```

**Closed-form spec defs (transitive):** `h_stateIO`, `final_stateIO`, `chunk_gated_attention_h_state_store_slice`, `chunk_gated_attention_final_state_store_slice`

<details><summary><code>h_stateIO</code></summary>

```
/-- IO signature of `chunk_gated_attention_h_state_store_slice` on the three-axis tile surface. -/
```
```lean
def h_stateIO (BH H : RegionName) (i_t s_h_h s_h_t s_h_d KSize VSize BK BV : Nat) :
    Masked3DTileKernelIO₁ where
  kernel := chunk_gated_attention_h_state_store_slice BH H i_t s_h_h s_h_t s_h_d KSize VSize BK BV
  inp := BH
  out := H
  shape := [BK, BV]
  read := fun p₀ p₁ p₂ idx => p₂ * s_h_h + i_t * KSize * VSize + (p₁ * BK + idx.1.val) * s_h_t
      + (p₀ * BV + idx.2.1.val) * s_h_d
  write := fun p₀ p₁ p₂ idx => p₂ * s_h_h + i_t * KSize * VSize + (p₁ * BK + idx.1.val) * s_h_t
      + (p₀ * BV + idx.2.1.val) * s_h_d
  mask := fun p₀ p₁ _p₂ idx =>
    p₁ * BK + idx.1.val < KSize ∧ p₀ * BV + idx.2.1.val < VSize
```
</details>

<details><summary><code>final_stateIO</code></summary>

```
/-- IO signature of `chunk_gated_attention_final_state_store_slice` on the three-axis tile surface. -/
```
```lean
def final_stateIO (BHFinal Ht : RegionName) (KSize VSize BK BV : Nat) :
    Masked3DTileKernelIO₁ where
  kernel := chunk_gated_attention_final_state_store_slice BHFinal Ht KSize VSize BK BV
  inp := BHFinal
  out := Ht
  shape := [BK, BV]
  read := fun p₀ p₁ p₂ idx => p₂ * KSize * VSize + (p₁ * BK + idx.1.val) * VSize
      + (p₀ * BV + idx.2.1.val)
  write := fun p₀ p₁ p₂ idx => p₂ * KSize * VSize + (p₁ * BK + idx.1.val) * VSize
      + (p₀ * BV + idx.2.1.val)
  mask := fun p₀ p₁ _p₂ idx =>
    p₁ * BK + idx.1.val < KSize ∧ p₀ * BV + idx.2.1.val < VSize
```
</details>

<details><summary><code>chunk_gated_attention_h_state_store_slice</code></summary>

```
/-- **Masked memcpy** modelling the intermediate-state writeback of
`chunk_gated_abc_fwd_kernel_h`.

At each `i_t`, the Python kernel stores the current recurrent state `b_h` into
`H + i_bh * s_h_h + i_t * KSize * VSize` before applying the chunk update. This
slice starts from a precomputed `BH` tile — and its **load and store addresses
are character-identical** under the same mask, so it computes nothing: it
transports whatever `BH` holds into `H`. The gating and the `b_k @ b_v`
accumulation that actually produce `b_h` are *not* modeled anywhere in this
file. Consequently no theorem built on this slice is a claim about the
recurrence; see the module docstring. -/
```
```lean
def chunk_gated_attention_h_state_store_slice
    (BH H : RegionName) (i_t s_h_h s_h_t s_h_d KSize VSize BK BV : Nat) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = i_k * $(BK) + tl.arange(0, $(BK))
  offs_v = i_v * $(BV) + tl.arange(0, $(BV))
  mask = (offs_k[:, None] < $(KSize)) & (offs_v[None, :] < $(VSize))
  b_h = tl.load(BH + i_bh * $(s_h_h) + $(i_t) * $(KSize) * $(VSize) +
      offs_k[:, None] * $(s_h_t) + offs_v[None, :] * $(s_h_d),
    mask=mask, other=0.0)
  tl.store(H + i_bh * $(s_h_h) + $(i_t) * $(KSize) * $(VSize) +
      offs_k[:, None] * $(s_h_t) + offs_v[None, :] * $(s_h_d),
    b_h, mask=mask)
}
```
</details>

<details><summary><code>chunk_gated_attention_final_state_store_slice</code></summary>

```
/-- **Masked memcpy** modelling the `STORE_FINAL_STATE` writeback of
`chunk_gated_attention.py`'s `chunk_gated_abc_fwd_kernel_h`: it moves a
precomputed final-state `BHFinal` `[BK, BV]` tile into `Ht` after the
`NT`-iteration chunk loop. Load and store addresses are character-identical
under the same mask, so this slice computes nothing — it transports whatever
`BHFinal` holds. -/
```
```lean
def chunk_gated_attention_final_state_store_slice
    (BHFinal Ht : RegionName) (KSize VSize BK BV : Nat) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = i_k * $(BK) + tl.arange(0, $(BK))
  offs_v = i_v * $(BV) + tl.arange(0, $(BV))
  mask = (offs_k[:, None] < $(KSize)) & (offs_v[None, :] < $(VSize))
  b_h = tl.load(BHFinal + i_bh * $(KSize) * $(VSize) +
      offs_k[:, None] * $(VSize) + offs_v[None, :],
    mask=mask, other=0.0)
  tl.store(Ht + i_bh * $(KSize) * $(VSize) +
      offs_k[:, None] * $(VSize) + offs_v[None, :], b_h, mask=mask)
}
```
</details>

## Public theorem: `chunk_gated_attention_state_stores_io_correctnessR`

<details><summary>docstring</summary>

```
/-- **The `⊨[R]` headline** for `chunk_gated_attention.py`'s two masked state
writebacks: for **every** rounding model `R`, the same pair of masked Hoare triples
as `chunk_gated_attention_state_stores_io_correctness`, but run under `execR R` and
read back as `.real`-typed cells holding `R.round .real (xs idx)`.

Both are pure copies carrying no `.to(...)`, so both slices are cast-free and the
exact runs transport verbatim. The content of the rounding face here is exactly
that: *neither kernel introduces a rounding event of its own*, at any `R`. -/
```
</details>

**Statement:**
```lean
specification chunk_gated_attention_state_stores_io_correctnessR
    (R : RoundingModel) (BH H BHFinal Ht : RegionName)
    (i_t s_h_h s_h_t s_h_d KSize VSize BK BV : Nat)
    (hInj1 : ∀ p₀ p₁ p₂ : Nat, Function.Injective
      (fun idx : TileIndex [BK, BV] =>
        p₂ * s_h_h + i_t * KSize * VSize + (p₁ * BK + idx.1.val) * s_h_t
          + (p₀ * BV + idx.2.1.val) * s_h_d))
    (hInj2 : ∀ p₀ p₁ p₂ : Nat, Function.Injective
      (fun idx : TileIndex [BK, BV] =>
        p₂ * KSize * VSize + (p₁ * BK + idx.1.val) * VSize
          + (p₀ * BV + idx.2.1.val))) :
    (h_stateIO BH H i_t s_h_h s_h_t s_h_d KSize VSize BK BV
      ⊨[R, FloatDType.real] fun _p₀ _p₁ xs idx => xs idx) ∧
    (final_stateIO BHFinal Ht KSize VSize BK BV
      ⊨[R, FloatDType.real] fun _p₀ _p₁ xs idx => xs idx)
```

**Closed-form spec defs (transitive):** `h_stateIO`, `final_stateIO`, `chunk_gated_attention_h_state_store_slice`, `chunk_gated_attention_final_state_store_slice`

<details><summary><code>h_stateIO</code></summary>

```
/-- IO signature of `chunk_gated_attention_h_state_store_slice` on the three-axis tile surface. -/
```
```lean
def h_stateIO (BH H : RegionName) (i_t s_h_h s_h_t s_h_d KSize VSize BK BV : Nat) :
    Masked3DTileKernelIO₁ where
  kernel := chunk_gated_attention_h_state_store_slice BH H i_t s_h_h s_h_t s_h_d KSize VSize BK BV
  inp := BH
  out := H
  shape := [BK, BV]
  read := fun p₀ p₁ p₂ idx => p₂ * s_h_h + i_t * KSize * VSize + (p₁ * BK + idx.1.val) * s_h_t
      + (p₀ * BV + idx.2.1.val) * s_h_d
  write := fun p₀ p₁ p₂ idx => p₂ * s_h_h + i_t * KSize * VSize + (p₁ * BK + idx.1.val) * s_h_t
      + (p₀ * BV + idx.2.1.val) * s_h_d
  mask := fun p₀ p₁ _p₂ idx =>
    p₁ * BK + idx.1.val < KSize ∧ p₀ * BV + idx.2.1.val < VSize
```
</details>

<details><summary><code>final_stateIO</code></summary>

```
/-- IO signature of `chunk_gated_attention_final_state_store_slice` on the three-axis tile surface. -/
```
```lean
def final_stateIO (BHFinal Ht : RegionName) (KSize VSize BK BV : Nat) :
    Masked3DTileKernelIO₁ where
  kernel := chunk_gated_attention_final_state_store_slice BHFinal Ht KSize VSize BK BV
  inp := BHFinal
  out := Ht
  shape := [BK, BV]
  read := fun p₀ p₁ p₂ idx => p₂ * KSize * VSize + (p₁ * BK + idx.1.val) * VSize
      + (p₀ * BV + idx.2.1.val)
  write := fun p₀ p₁ p₂ idx => p₂ * KSize * VSize + (p₁ * BK + idx.1.val) * VSize
      + (p₀ * BV + idx.2.1.val)
  mask := fun p₀ p₁ _p₂ idx =>
    p₁ * BK + idx.1.val < KSize ∧ p₀ * BV + idx.2.1.val < VSize
```
</details>

<details><summary><code>chunk_gated_attention_h_state_store_slice</code></summary>

```
/-- **Masked memcpy** modelling the intermediate-state writeback of
`chunk_gated_abc_fwd_kernel_h`.

At each `i_t`, the Python kernel stores the current recurrent state `b_h` into
`H + i_bh * s_h_h + i_t * KSize * VSize` before applying the chunk update. This
slice starts from a precomputed `BH` tile — and its **load and store addresses
are character-identical** under the same mask, so it computes nothing: it
transports whatever `BH` holds into `H`. The gating and the `b_k @ b_v`
accumulation that actually produce `b_h` are *not* modeled anywhere in this
file. Consequently no theorem built on this slice is a claim about the
recurrence; see the module docstring. -/
```
```lean
def chunk_gated_attention_h_state_store_slice
    (BH H : RegionName) (i_t s_h_h s_h_t s_h_d KSize VSize BK BV : Nat) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = i_k * $(BK) + tl.arange(0, $(BK))
  offs_v = i_v * $(BV) + tl.arange(0, $(BV))
  mask = (offs_k[:, None] < $(KSize)) & (offs_v[None, :] < $(VSize))
  b_h = tl.load(BH + i_bh * $(s_h_h) + $(i_t) * $(KSize) * $(VSize) +
      offs_k[:, None] * $(s_h_t) + offs_v[None, :] * $(s_h_d),
    mask=mask, other=0.0)
  tl.store(H + i_bh * $(s_h_h) + $(i_t) * $(KSize) * $(VSize) +
      offs_k[:, None] * $(s_h_t) + offs_v[None, :] * $(s_h_d),
    b_h, mask=mask)
}
```
</details>

<details><summary><code>chunk_gated_attention_final_state_store_slice</code></summary>

```
/-- **Masked memcpy** modelling the `STORE_FINAL_STATE` writeback of
`chunk_gated_attention.py`'s `chunk_gated_abc_fwd_kernel_h`: it moves a
precomputed final-state `BHFinal` `[BK, BV]` tile into `Ht` after the
`NT`-iteration chunk loop. Load and store addresses are character-identical
under the same mask, so this slice computes nothing — it transports whatever
`BHFinal` holds. -/
```
```lean
def chunk_gated_attention_final_state_store_slice
    (BHFinal Ht : RegionName) (KSize VSize BK BV : Nat) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = i_k * $(BK) + tl.arange(0, $(BK))
  offs_v = i_v * $(BV) + tl.arange(0, $(BV))
  mask = (offs_k[:, None] < $(KSize)) & (offs_v[None, :] < $(VSize))
  b_h = tl.load(BHFinal + i_bh * $(KSize) * $(VSize) +
      offs_k[:, None] * $(VSize) + offs_v[None, :],
    mask=mask, other=0.0)
  tl.store(Ht + i_bh * $(KSize) * $(VSize) +
      offs_k[:, None] * $(VSize) + offs_v[None, :], b_h, mask=mask)
}
```
</details>

## Also present (pinned special-case summaries)
- `chunk_gated_attention_cum_compute_slice_compute_correct`
- `chunk_gated_attention_h_state_store_slice_compute_correct`
- `chunk_gated_attention_final_state_store_slice_compute_correct`
