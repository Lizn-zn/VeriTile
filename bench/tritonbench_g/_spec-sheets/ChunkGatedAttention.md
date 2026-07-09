# Spec sheet — `bench/tritonbench_g/chunk_gated_attention/ChunkGatedAttention.lean`

**Python source:** `bench/tritonbench_g/chunk_gated_attention/chunk_gated_attention.py`

## Public theorem: `chunk_gated_attention_output_summary_general`

<details><summary>docstring</summary>

```
/-- **Dimension-general** correctness summary for `chunk_gated_attention.py`,
against the **genuine closed forms** (no self-referential read-back), for
**arbitrary** chunk shapes `T S KSize VSize BT BS BK BV NT` and strides.

This is the headline: it packages, for symbolic dimensions,

* `fwd_pre` cumulative normalizer: the `b_o = m_s @ b_s` store realizes the
  genuine causal-cumsum `cumComputeStoreValue` (`lowerTri @ source`);
* the gated `fwd_inner` recurrence (all `(GATEK, USE_INITIAL_STATE,
  STORE_FINAL_STATE)` flag settings): each `h` loop-row store at chunk `i_t`
  realizes the genuine folded gated state `hClosed i_t`, and the final `ht` store
  realizes the fully-folded `hClosed NT`.

Honest side-conditions only: offset injectivity of each store footprint (a
contiguity/aliasing-freedom hypothesis on the strides) and the
buffer-carries-`hClosed` hypotheses (the cross-chunk loop scheduling that threads
the carried `b_h` register, whose algebra is the `hClosed` carry-fold, is the
trusted runtime boundary, as in #290). No dimension is pinned. -/
```
</details>

**Statement:**
```lean
theorem chunk_gated_attention_output_summary_general
    (SReg GCum K V G H H0 Ht BH BHFinal : RegionName)
    (GATEK USE_INITIAL_STATE _STORE_FINAL_STATE : Bool)
    (i_t NT : Nat)
    (s_s_h s_s_t s_s_d s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d
      s_h_h s_h_t s_h_d T S KSize VSize BT BS BK BV : Nat) (s : BlockState)
    (hCumInj : Function.Injective
      (fun idx : TileIndex [BT, BS] => tileOffset s s_s_h s_s_t s_s_d BT BS idx))
    (hStateInj : Function.Injective
      (fun idx : TileIndex [BK, BV] =>
        hStateOffset s i_t s_h_h s_h_t s_h_d KSize VSize BK BV idx))
    (hFinalInj : Function.Injective
      (fun idx : TileIndex [BK, BV] =>
        finalStateOffset s KSize VSize BK BV idx))
    (hBufState : ∀ idx : TileIndex [BK, BV],
      s.readMem BH (hStateOffset s i_t s_h_h s_h_t s_h_d KSize VSize BK BV idx)
        = hClosed s K V G H0 GATEK USE_INITIAL_STATE
            s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV i_t idx)
    (hBufFinal : ∀ idx : TileIndex [BK, BV],
      s.readMem BHFinal (finalStateOffset s KSize VSize BK BV idx)
        = hClosed s K V G H0 GATEK USE_INITIAL_STATE
            s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV NT idx) :
    -- (1) `fwd_pre` cumulative normalizer realizes the genuine causal cumsum.
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := chunk_gated_attention_cum_compute_slice SReg GCum s_s_h s_s_t
        s_s_d T S BT BS)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BT, BS] => active s T S BT BS idx)
        (fun idx : TileIndex [BT, BS] =>
          (GCum, tileOffset s s_s_h s_s_t s_s_d BT BS idx)))
      (expected := fun idx : TileIndex [BT, BS] =>
        cumComputeStoreValue s SReg s_s_h s_s_t s_s_d T S BT BS idx)) ∧
    -- (2) `h` loop-row store at chunk `i_t` realizes the genuine folded state.
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := chunk_gated_attention_h_state_store_slice BH H i_t
        s_h_h s_h_t s_h_d KSize VSize BK BV)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BK, BV] => stateActive s KSize VSize BK BV idx)
        (fun idx : TileIndex [BK, BV] =>
          (H, hStateOffset s i_t s_h_h s_h_t s_h_d KSize VSize BK BV idx)))
      (expected := fun idx : TileIndex [BK, BV] =>
        hClosed s K V G H0 GATEK USE_INITIAL_STATE
          s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV i_t idx)) ∧
    -- (3) final `ht` store realizes the genuine fully-folded state `hClosed NT`.
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := chunk_gated_attention_final_state_store_slice BHFinal Ht
        KSize VSize BK BV)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BK, BV] => finalActive s KSize VSize BK BV idx)
        (fun idx : TileIndex [BK, BV] =>
          (Ht, finalStateOffset s KSize VSize BK BV idx)))
      (expected := fun idx : TileIndex [BK, BV] =>
        hClosed s K V G H0 GATEK USE_INITIAL_STATE
          s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV NT idx))
```

**Assumptions / layout contracts:**
- `hCumInj : Function.Injective
      (fun idx : TileIndex [BT, BS] => tileOffset s s_s_h s_s_t s_s_d BT BS idx)`
- `hStateInj : Function.Injective
      (fun idx : TileIndex [BK, BV] =>
        hStateOffset s i_t s_h_h s_h_t s_h_d KSize VSize BK BV idx)`
- `hFinalInj : Function.Injective
      (fun idx : TileIndex [BK, BV] =>
        finalStateOffset s KSize VSize BK BV idx)`
- `hBufState : ∀ idx : TileIndex [BK, BV],
      s.readMem BH (hStateOffset s i_t s_h_h s_h_t s_h_d KSize VSize BK BV idx)
        = hClosed s K V G H0 GATEK USE_INITIAL_STATE
            s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV i_t idx`
- `hBufFinal : ∀ idx : TileIndex [BK, BV],
      s.readMem BHFinal (finalStateOffset s KSize VSize BK BV idx)
        = hClosed s K V G H0 GATEK USE_INITIAL_STATE
            s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV NT idx`
- `fun idx : TileIndex [BT, BS] => active s T S BT BS idx`
- `fun idx : TileIndex [BT, BS] =>
          (GCum, tileOffset s s_s_h s_s_t s_s_d BT BS idx)`
- `fun idx : TileIndex [BK, BV] => stateActive s KSize VSize BK BV idx`
- `fun idx : TileIndex [BK, BV] =>
          (H, hStateOffset s i_t s_h_h s_h_t s_h_d KSize VSize BK BV idx)`
- `fun idx : TileIndex [BK, BV] => finalActive s KSize VSize BK BV idx`
- `fun idx : TileIndex [BK, BV] =>
          (Ht, finalStateOffset s KSize VSize BK BV idx)`

**Closed-form spec defs (transitive):** `tileOffset`, `hStateOffset`, `finalStateOffset`, `hClosed`, `chunk_gated_attention_cum_compute_slice`, `active`, `cumComputeStoreValue`, `chunk_gated_attention_h_state_store_slice`, `stateActive`, `chunk_gated_attention_final_state_store_slice`, `finalActive`, `tIndex`, `sIndex`, `kIndexState`, `vIndexState`, `kIndexFinal`, `vIndexFinal`, `hSeed`, `hGate`, `hStepTerm`, `lowerTriTile`, `sourceTile`, `h0Elem`, `gnkElem`, `chunkLastTime`, `gnvElem`, `ktElem`, `tvElem`

<details><summary><code>tileOffset</code></summary>

```lean
def tileOffset (s : BlockState) (s_s_h s_s_t s_s_d BT BS : Nat)
    (idx : TileIndex [BT, BS]) : Nat :=
  s.pids 1 * s_s_h + tIndex s BT idx.1 * s_s_t +
    sIndex s BS idx.2.1 * s_s_d
```
</details>

<details><summary><code>hStateOffset</code></summary>

```lean
def hStateOffset (s : BlockState) (i_t s_h_h s_h_t s_h_d KSize VSize BK BV : Nat)
    (idx : TileIndex [BK, BV]) : Nat :=
  s.pids 2 * s_h_h + i_t * KSize * VSize +
    kIndexState s BK idx.1 * s_h_t + vIndexState s BV idx.2.1 * s_h_d
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
        hGate s G GATEK s_k_h s_v_h KSize VSize BT BK BV j idx) +
    ∑ t ∈ Finset.range m,
      hStepTerm s K V G GATEK s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d
          KSize VSize BT BK BV t idx *
        (∏ j ∈ Finset.Ico (t + 1) m,
          hGate s G GATEK s_k_h s_v_h KSize VSize BT BK BV j idx)
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
  i_bh = tl.program_id(1)
  i_t = tl.program_id(2)
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

<details><summary><code>active</code></summary>

```lean
def active (s : BlockState) (T S BT BS : Nat) (idx : TileIndex [BT, BS]) : Prop :=
  tIndex s BT idx.1 < T ∧ sIndex s BS idx.2.1 < S
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

<details><summary><code>chunk_gated_attention_h_state_store_slice</code></summary>

```
/-- Proof-oriented intermediate-state store slice of
`chunk_gated_abc_fwd_kernel_h`.

At each `i_t`, the Python kernel stores the current recurrent state `b_h` into
`H + i_bh * s_h_h + i_t * KSize * VSize` before applying the chunk update.
This slice starts from a precomputed `BH` tile and proves that masked writeback. -/
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

<details><summary><code>stateActive</code></summary>

```lean
def stateActive (s : BlockState) (KSize VSize BK BV : Nat)
    (idx : TileIndex [BK, BV]) : Prop :=
  kIndexState s BK idx.1 < KSize ∧ vIndexState s BV idx.2.1 < VSize
```
</details>

<details><summary><code>chunk_gated_attention_final_state_store_slice</code></summary>

```
/-- Proof-oriented final-state store slice of `chunk_gated_attention.py`'s
`chunk_gated_abc_fwd_kernel_h`. Writes a precomputed final-state `BHFinal`
[BK, BV] tile into `Ht` after the NT-iteration chunk loop completes
(STORE_FINAL_STATE=True branch). -/
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

<details><summary><code>finalActive</code></summary>

```lean
def finalActive (s : BlockState) (KSize VSize BK BV : Nat)
    (idx : TileIndex [BK, BV]) : Prop :=
  kIndexFinal s BK idx.1 < KSize ∧ vIndexFinal s BV idx.2.1 < VSize
```
</details>

<details><summary><code>tIndex</code></summary>

```lean
def tIndex (s : BlockState) (BT : Nat) (i : Fin BT) : Nat :=
  s.pids 2 * BT + i.val
```
</details>

<details><summary><code>sIndex</code></summary>

```lean
def sIndex (s : BlockState) (BS : Nat) (j : Fin BS) : Nat :=
  s.pids 0 * BS + j.val
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

<details><summary><code>hGate</code></summary>

```
/-- Per-chunk gate factor `G_m[k,v]` at the bench shape. `GATEK` ⇒ per-key-row
`exp(b_gn_m[k])`; otherwise per-value-column `exp(b_gn_m[v])`. `b_gn` is the
last-row (`t = BT-1`) cumulative gate of chunk `m`. -/
```
```lean
noncomputable def hGate
    (s : BlockState) (G : RegionName) (GATEK : Bool)
    (s_k_h s_v_h KSize VSize BT BK BV m : Nat)
    (idx : TileIndex [BK, BV]) : ℝ :=
  if GATEK then
    Real.exp (gnkElem s G s_k_h KSize (chunkLastTime BT m)
      (kIndexState s BK idx.1))
  else
    Real.exp (gnvElem s G s_v_h VSize (chunkLastTime BT m)
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
      let gnVal := gnkElem s G s_k_h KSize (chunkLastTime BT m)
        (kIndexState s BK idx.1)
      let gVal := ktElem s G s_k_h s_k_t s_k_d
        (kIndexState s BK idx.1) (m * BT + t.val)
      (kVal * Real.exp (gnVal - gVal)) * vVal
    else
      let gnVal := gnvElem s G s_v_h VSize (chunkLastTime BT m)
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
      if active s T S BT BS idx then
        some (s.readMem SReg (tileOffset s s_s_h s_s_t s_s_d BT BS idx))
      else some (0.0 : ℝ) }
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

<details><summary><code>gnkElem</code></summary>

```
/-- Last-lane cumulative gate `b_gn[k]` under `GATEK`: lane `tLast*KSize + k` of
`p_gn`'s flattened `(T*K,)` view of `G` at batch-head `s.pids 2`. NOTE: the
port's `p_gn` carries element stride `s_k_d`; this closed form reads the lane at
**unit element stride** (i.e. it is the `s_k_d = 1` footprint). -/
```
```lean
noncomputable def gnkElem (s : BlockState) (G : RegionName)
    (s_k_h KSize tLast k : Nat) : ℝ :=
  s.readMem G (s.pids 2 * s_k_h + (tLast * KSize + k))
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

<details><summary><code>gnvElem</code></summary>

```
/-- Last-lane cumulative gate `b_gn[v]` under `¬GATEK`: lane `tLast*VSize + v` of
`p_gn`'s flattened `(T*V,)` view of `G` at batch-head `s.pids 2`. NOTE: the
port's `p_gn` carries element stride `s_v_d`; this closed form reads the lane at
**unit element stride** (i.e. it is the `s_v_d = 1` footprint). -/
```
```lean
noncomputable def gnvElem (s : BlockState) (G : RegionName)
    (s_v_h VSize tLast v : Nat) : ℝ :=
  s.readMem G (s.pids 2 * s_v_h + (tLast * VSize + v))
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

## Also present (pinned special-case summaries)
- `chunk_gated_attention_store_slice_compute_correct`
- `chunk_gated_attention_cum_compute_slice_compute_correct`
- `chunk_gated_attention_h_state_store_slice_compute_correct`
- `chunk_gated_attention_final_state_store_slice_compute_correct`
