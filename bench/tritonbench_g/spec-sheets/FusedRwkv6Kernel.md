# Spec sheet — `bench/tritonbench_g/fused_rwkv6_kernel/FusedRwkv6Kernel.lean`

**Python source:** `bench/tritonbench_g/fused_rwkv6_kernel/fused_rwkv6_kernel.py`

## Public theorem: `fused_recurrent_rwkv6_output_summary_general`

<details><summary>docstring</summary>

```
/-! ### ════════ ★ MAIN THEOREM ★ ════════

**Genuine, dimension-general RWKV6 compute-correctness.** Parameterized over the
symbolic head strides `s_k_h s_v_h`, batch/head/time `B H T`, key/value extents
`K V`, tile sizes `BK BV`, the real `scale`, the step index `m`, the final time
`T`, and **both** flags `USE_INITIAL_STATE STORE_FINAL_STATE`. It bundles the
four genuine faces, each realized against the closed forms `stateClosed` /
`outputClosed` over the *input* regions (never a read-back of the kernel's own
output):

1. the full RWKV6 forward surface lowers to the algorithm layer;
2. one **output** body realizes `outputClosed(m)` (the key-axis reduction);
3. one **state-update** body realizes `stateClosed(m+1)` (the per-channel decay
   carry-fold), given the carry invariant `BHPrev = stateClosed(m)`;
4. the **final-state** writeback realizes `stateClosed(T)` (masked), given
   `BHFinal = stateClosed(T)`.

Honest structural side conditions only: `BK ≤ K`, `BV ≤ V` (the tile fits the
logical extents, giving offset injectivity for the state face) and `0 < BV`
(contiguous output lanes, giving injectivity for the output face). The flags flow
through verbatim; clauses 2–4 hold for every flag setting, and each Python case
is recovered by projecting the subset of clauses its `STORE_FINAL_STATE` /
`USE_INITIAL_STATE` configuration exercises. -/
```
</details>

**Statement:**
```lean
theorem fused_recurrent_rwkv6_output_summary_general
    (q k v w u o h0 ht BHPrev BHOut BHFinal : RegionName)
    (USE_INITIAL_STATE STORE_FINAL_STATE : Bool)
    (s_k_h s_v_h B H T K V BK BV m : Nat) (scale : ℝ) (s : BlockState)
    (hBV : BV ≤ V) (hBK : BK ≤ K) (hBVpos : 0 < BV)
    (hPrev : ∀ idx : TileIndex [BV, BK],
      s.readMem BHPrev (finalStateOffset s K V BK BV idx)
        = stateClosed s k v w h0 USE_INITIAL_STATE s_k_h s_v_h K V BK BV m idx)
    (hFinal : ∀ idx : TileIndex [BV, BK],
      s.readMem BHFinal (finalStateOffset s K V BK BV idx)
        = stateClosed s k v w h0 USE_INITIAL_STATE s_k_h s_v_h K V BK BV T idx) :
    -- (1) the full surface lowers to the algorithm layer
    (∃ alg, (fused_recurrent_rwkv6_fwd_surface q k v w u o h0 ht
      s_k_h s_v_h B H T K V BK BV scale USE_INITIAL_STATE STORE_FINAL_STATE
      Bool.false).toAlgorithm? = Except.ok alg) ∧
    -- (2) the output body realizes the genuine `outputClosed(m)`
    (ComputeCorrect.Realizes
      (kernel := fused_recurrent_rwkv6_output_step_slice BHPrev q k v u o
        m s_k_h s_v_h B H T K V BK BV scale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun jv : Fin BV => active s V BV jv)
        (fun jv => (o, outStepOffset s m s_v_h B H V BV jv)))
      (expected := fun jv : Fin BV =>
        outputClosed s q k v w u h0 USE_INITIAL_STATE s_k_h s_v_h H K V BK BV
          scale m jv)) ∧
    -- (3) the state-update body realizes the genuine `stateClosed(m+1)`
    (ComputeCorrect.Realizes
      (kernel := fused_recurrent_rwkv6_state_step_slice BHPrev k v w BHOut
        m s_k_h s_v_h K V BK BV)
      (initialState := s)
      (write := fun idx : TileIndex [BV, BK] =>
        some (BHOut, finalStateOffset s K V BK BV idx))
      (expected := fun idx =>
        stateClosed s k v w h0 USE_INITIAL_STATE s_k_h s_v_h K V BK BV
          (m + 1) idx)) ∧
    -- (4) the final-state writeback realizes the genuine `stateClosed(T)` (masked)
    (ComputeCorrect.Realizes
      (kernel := fused_recurrent_rwkv6_final_state_store_slice BHFinal ht K V BK BV)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BV, BK] => finalActive s K V BK BV idx)
        (fun idx : TileIndex [BV, BK] => (ht, finalStateOffset s K V BK BV idx)))
      (expected := fun idx : TileIndex [BV, BK] =>
        if finalActive s K V BK BV idx then
          stateClosed s k v w h0 USE_INITIAL_STATE s_k_h s_v_h K V BK BV T idx
        else 0))
```

**Assumptions / layout contracts:**
- `hBV : BV ≤ V`
- `hBK : BK ≤ K`
- `hBVpos : 0 < BV`
- `hPrev : ∀ idx : TileIndex [BV, BK],
      s.readMem BHPrev (finalStateOffset s K V BK BV idx)
        = stateClosed s k v w h0 USE_INITIAL_STATE s_k_h s_v_h K V BK BV m idx`
- `hFinal : ∀ idx : TileIndex [BV, BK],
      s.readMem BHFinal (finalStateOffset s K V BK BV idx)
        = stateClosed s k v w h0 USE_INITIAL_STATE s_k_h s_v_h K V BK BV T idx`
- `kernel : = fused_recurrent_rwkv6_output_step_slice BHPrev q k v u o
        m s_k_h s_v_h B H T K V BK BV scale`
- `initialState : = s`
- `fun jv : Fin BV => active s V BV jv`
- `expected : = fun jv : Fin BV =>
        outputClosed s q k v w u h0 USE_INITIAL_STATE s_k_h s_v_h H K V BK BV
          scale m jv`
- `kernel : = fused_recurrent_rwkv6_state_step_slice BHPrev k v w BHOut
        m s_k_h s_v_h K V BK BV`
- `initialState : = s`
- `write : = fun idx : TileIndex [BV, BK] =>
        some (BHOut, finalStateOffset s K V BK BV idx)`
- `expected : = fun idx =>
        stateClosed s k v w h0 USE_INITIAL_STATE s_k_h s_v_h K V BK BV
          (m + 1) idx`
- `kernel : = fused_recurrent_rwkv6_final_state_store_slice BHFinal ht K V BK BV`
- `initialState : = s`
- `fun idx : TileIndex [BV, BK] => finalActive s K V BK BV idx`
- `fun idx : TileIndex [BV, BK] => (ht, finalStateOffset s K V BK BV idx)`
- `expected : = fun idx : TileIndex [BV, BK] =>
        if finalActive s K V BK BV idx then
          stateClosed s k v w h0 USE_INITIAL_STATE s_k_h s_v_h K V BK BV T idx
        else 0`

**Closed-form spec defs (transitive):** `finalStateOffset`, `stateClosed`, `fused_recurrent_rwkv6_fwd_surface`, `fused_recurrent_rwkv6_output_step_slice`, `active`, `outStepOffset`, `outputClosed`, `fused_recurrent_rwkv6_state_step_slice`, `fused_recurrent_rwkv6_final_state_store_slice`, `finalActive`, `kIndex`, `vIndex`, `stateSeed`, `decay`, `kVal`, `vVal`, `uVal`, `qVal`, `h0Val`

<details><summary><code>finalStateOffset</code></summary>

```
/-- Flattened final-state / `h0` address (row-major `i_bh·K·V + j_k·V + j_v`). -/
```
```lean
def finalStateOffset (s : BlockState) (K V BK BV : Nat)
    (idx : TileIndex [BV, BK]) : Nat :=
  s.pids 2 * K * V + kIndex s BK idx.2.1 * V + vIndex s BV idx.1
```
</details>

<details><summary><code>stateClosed</code></summary>

```
/-- **Genuine closed form for the state after `m` steps**, tile element `idx`:
`seed · ∏_{j<m} exp(w_j) + Σ_{t<m} (k_t·v_t) · ∏_{t<j<m} exp(w_j)`. This is a
standalone specification over the input regions `k,v,w,h0` — never a read-back
of the kernel's own output. -/
```
```lean
noncomputable def stateClosed
    (s : BlockState) (k v w h0 : RegionName) (USE_INITIAL_STATE : Bool)
    (s_k_h s_v_h K V BK BV m : Nat) (idx : TileIndex [BV, BK]) : ℝ :=
  stateSeed s h0 USE_INITIAL_STATE K V BK BV idx *
      (∏ j ∈ Finset.range m, decay s w s_k_h K BK j idx.2.1) +
    ∑ t ∈ Finset.range m,
      (kVal s k s_k_h K BK t idx.2.1 * vVal s v s_v_h V BV t idx.1) *
        (∏ j ∈ Finset.Ico (t + 1) m, decay s w s_k_h K BK j idx.2.1)
```
</details>

<details><summary><code>fused_recurrent_rwkv6_fwd_surface</code></summary>

```
/-- Faithful transcription of `fused_rwkv6_kernel.py`'s
`fused_recurrent_rwkv6_fwd_kernel` as used by the exported benchmark helper.

The exported helper always calls the autograd entry point with its default
`reverse = false`, so pointer movement is modeled in the forward direction. The
`REVERSE` parameter is retained to match the source signature. -/
```
```lean
def fused_recurrent_rwkv6_fwd_surface
    (q k v w u o h0 ht : RegionName)
    (s_k_h s_v_h B H T K V BK BV : Nat) (scale : ℝ)
    (USE_INITIAL_STATE STORE_FINAL_STATE REVERSE : Bool) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  i_h = i_bh % $(H)
  p_q = q + i_bh * $(s_k_h) + i_k * $(BK) + tl.arange(0, $(BK))
  p_k = k + i_bh * $(s_k_h) + i_k * $(BK) + tl.arange(0, $(BK))
  p_v = v + i_bh * $(s_v_h) + i_v * $(BV) + tl.arange(0, $(BV))
  p_o = o + (i_bh + i_k * $(B) * $(H)) * $(s_v_h) +
    i_v * $(BV) + tl.arange(0, $(BV))
  p_w = w + i_bh * $(s_k_h) + i_k * $(BK) + tl.arange(0, $(BK))
  p_u = u + i_h * $(K) + tl.arange(0, $(BK)) + i_k * $(BK)
  mask_bk = (i_k * $(BK) + tl.arange(0, $(BK))) < $(K)
  mask_bv = (i_v * $(BV) + tl.arange(0, $(BV))) < $(V)
  mask_kv = mask_bv[:, None] & mask_bk[None, :]
  b_h = tl.zeros([$(BV), $(BK)], dtype=tl.float32)
  if USE_INITIAL_STATE {
    p_h0 = h0 + i_bh * $(K) * $(V) +
      (i_k * $(BK) + tl.arange(0, $(BK))[None, :]) * $(V) +
      (i_v * $(BV) + tl.arange(0, $(BV))[:, None])
    b_h += tl.load(p_h0, mask=mask_kv, other=0).to(tl.float32)
  }
  b_u = tl.load(p_u, mask=mask_bk, other=0).to(tl.float32)
  for _i in range($(0), $(T), $(1)) {
    b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
    b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
    b_q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * $(scale)
    b_w = tl.load(p_w, mask=mask_bk, other=0).to(tl.float32)
    b_w = tl.exp(b_w)
    b_kv = b_k[None, :] * b_v[:, None]
    b_o = (b_h + b_kv * b_u[None, :]) * b_q[None, :]
    b_o = tl.sum(b_o, axis=1)
    b_h = b_h * b_w[None, :]
    b_h += b_kv
    tl.store(p_o, (b_o).to(p_o.dtype.element_ty), mask=mask_bv)
    p_q += $(K)
    p_k += $(K)
    p_o += $(V)
    p_v += $(V)
    p_w += $(K)
  }
  if STORE_FINAL_STATE {
    p_ht = ht + i_bh * $(K) * $(V) +
      (i_k * $(BK) + tl.arange(0, $(BK))[None, :]) * $(V) +
      (i_v * $(BV) + tl.arange(0, $(BV))[:, None])
    tl.store(p_ht, (b_h).to(p_ht.dtype.element_ty), mask=mask_kv)
  }
}
```
</details>

<details><summary><code>fused_recurrent_rwkv6_output_step_slice</code></summary>

```
/-! ## Output-reduction step slice (the per-step output `o_t`)

This isolates the Python loop body's output computation. With the *pre-update*
state tile `BHPrev`, it loads `k_t/v_t/q_t/u`, forms
`(BHPrev + (k⊗v)·u)·(scale·q)`, reduces over the key axis (`tl.sum(_, axis=1)`),
and masked-stores the resulting `[BV]` row into `o` at time row `t`. -/
```
```lean
def fused_recurrent_rwkv6_output_step_slice
    (BHPrev q k v u o : RegionName)
    (t s_k_h s_v_h B H T K V BK BV : Nat) (scale : ℝ) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  i_h = i_bh % $(H)
  offs_k = tl.arange(0, $(BK))
  offs_v = tl.arange(0, $(BV))
  mask_bv = (i_v * $(BV) + offs_v) < $(V)
  prev = tl.load(BHPrev + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[None, :]) * $(V) + (i_v * $(BV) + offs_v[:, None]))
  b_k = tl.load(k + i_bh * $(s_k_h) + i_k * $(BK) + offs_k + $(t) * $(K))
  b_v = tl.load(v + i_bh * $(s_v_h) + i_v * $(BV) + offs_v + $(t) * $(V))
  b_q = tl.load(q + i_bh * $(s_k_h) + i_k * $(BK) + offs_k + $(t) * $(K)) * $(scale)
  b_u = tl.load(u + i_h * $(K) + offs_k + i_k * $(BK))
  b_kv = b_k[None, :] * b_v[:, None]
  b_o = (prev + b_kv * b_u[None, :]) * b_q[None, :]
  b_o = tl.sum(b_o, axis=1)
  tl.store(o + (i_bh + i_k * $(B) * $(H)) * $(s_v_h) + i_v * $(BV) + offs_v +
    $(t) * $(V), (b_o).to(o.dtype.element_ty), mask=mask_bv)
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active (s : BlockState) (V BV : Nat) (jv : Fin BV) : Prop :=
  vIndex s BV jv < V
```
</details>

<details><summary><code>outStepOffset</code></summary>

```
/-- The masked output address at lane `j_v` for time row `t` — the kernel's exact
`o + (i_bh + i_k·B·H)·s_v_h + i_v·BV + j_v + t·V` layout. -/
```
```lean
def outStepOffset (s : BlockState) (t s_v_h B H V BV : Nat) (jv : Fin BV) : Nat :=
  (s.pids 2 + s.pids 1 * B * H) * s_v_h + s.pids 0 * BV + jv.val + t * V
```
</details>

<details><summary><code>outputClosed</code></summary>

```
/-- **Genuine closed form for output row `m`, lane `j_v`** — the per-step
reduction over key channels:
`o_m[j_v] = Σ_{j_k} (b_h^(m)[j_v,j_k] + k_m[j_k]·v_m[j_v]·u[j_k]) · scale·q_m[j_k]`,
reading the *pre-update* state `stateClosed(m)`. -/
```
```lean
noncomputable def outputClosed
    (s : BlockState) (q k v w u h0 : RegionName) (USE_INITIAL_STATE : Bool)
    (s_k_h s_v_h H K V BK BV : Nat) (scale : ℝ) (m : Nat) (jv : Fin BV) : ℝ :=
  ∑ jk : Fin BK,
    (stateClosed s k v w h0 USE_INITIAL_STATE s_k_h s_v_h K V BK BV m
        (TileShape.insertAxisIndex [BV, BK] 1
          (TileShape.insertAxisIndex [BV] 0 PUnit.unit jv) jk)
      + (kVal s k s_k_h K BK m jk * vVal s v s_v_h V BV m jv) *
          uVal s u H K BK jk)
    * qVal s q s_k_h K BK scale m jk
```
</details>

<details><summary><code>fused_recurrent_rwkv6_state_step_slice</code></summary>

```
/-! ## State-update step slice (the per-channel decay carry-fold body)

This isolates the Python loop body's state update from the cross-step loop
induction. It loads the materialized previous-state tile `BHPrev`, the time-row
`k_t/v_t/w_t`, computes the outer product `b_kv = k ⊗ v`, the per-channel decay
`exp(w_t)`, and stores `b_h·exp(w_t) + b_kv` into a state buffer `BHOut` at the
canonical `[BV,BK]` layout. -/
```
```lean
def fused_recurrent_rwkv6_state_step_slice
    (BHPrev k v w BHOut : RegionName)
    (t s_k_h s_v_h K V BK BV : Nat) : ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = tl.arange(0, $(BK))
  offs_v = tl.arange(0, $(BV))
  prev = tl.load(BHPrev + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[None, :]) * $(V) + (i_v * $(BV) + offs_v[:, None]))
  b_k = tl.load(k + i_bh * $(s_k_h) + i_k * $(BK) + offs_k + $(t) * $(K))
  b_v = tl.load(v + i_bh * $(s_v_h) + i_v * $(BV) + offs_v + $(t) * $(V))
  b_w = tl.load(w + i_bh * $(s_k_h) + i_k * $(BK) + offs_k + $(t) * $(K))
  b_w = tl.exp(b_w)
  b_kv = b_k[None, :] * b_v[:, None]
  acc = prev * b_w[None, :] + b_kv
  tl.store(BHOut + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[None, :]) * $(V) + (i_v * $(BV) + offs_v[:, None]),
    (acc).to(BHOut.dtype.element_ty))
}
```
</details>

<details><summary><code>fused_recurrent_rwkv6_final_state_store_slice</code></summary>

```
/-! ## Final-state store slice (the `STORE_FINAL_STATE` branch)

After the loop, the kernel masked-stores the final state tile `b_h` into `ht`.
This slice models that writeback exactly, reading the materialized final-state
tile `BHFinal` and writing the masked `[BV,BK]` face into `Ht`. -/
```
```lean
def fused_recurrent_rwkv6_final_state_store_slice
    (BHFinal Ht : RegionName) (K V BK BV : Nat) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = i_k * $(BK) + tl.arange(0, $(BK))
  offs_v = i_v * $(BV) + tl.arange(0, $(BV))
  mask_kv = (offs_v[:, None] < $(V)) & (offs_k[None, :] < $(K))
  b_h = tl.load(BHFinal + i_bh * $(K) * $(V) +
      offs_k[None, :] * $(V) + offs_v[:, None],
    mask=mask_kv, other=0.0)
  tl.store(Ht + i_bh * $(K) * $(V) +
      offs_k[None, :] * $(V) + offs_v[:, None],
    b_h, mask=mask_kv)
}
```
</details>

<details><summary><code>finalActive</code></summary>

```lean
def finalActive (s : BlockState) (K V BK BV : Nat)
    (idx : TileIndex [BV, BK]) : Prop :=
  vIndex s BV idx.1 < V ∧ kIndex s BK idx.2.1 < K
```
</details>

<details><summary><code>kIndex</code></summary>

```lean
def kIndex (s : BlockState) (BK : Nat) (jk : Fin BK) : Nat :=
  s.pids 1 * BK + jk.val
```
</details>

<details><summary><code>vIndex</code></summary>

```
/-! ## Index / offset helpers

`s.pids 0 = i_v`, `s.pids 1 = i_k`, `s.pids 2 = i_bh`. A `[BV, BK]` state tile is
indexed by `idx : TileIndex [BV, BK]` with `idx.1` the value (`j_v`) axis and
`idx.2.1` the key (`j_k`) axis. -/
```
```lean
def vIndex (s : BlockState) (BV : Nat) (jv : Fin BV) : Nat :=
  s.pids 0 * BV + jv.val
```
</details>

<details><summary><code>stateSeed</code></summary>

```
/-- Seeded initial state `b_h^(0)`: `h0` if `USE_INITIAL_STATE` else `0`. -/
```
```lean
noncomputable def stateSeed (s : BlockState) (h0 : RegionName)
    (USE_INITIAL_STATE : Bool) (K V BK BV : Nat)
    (idx : TileIndex [BV, BK]) : ℝ :=
  if USE_INITIAL_STATE then h0Val s h0 K V BK BV idx else 0
```
</details>

<details><summary><code>decay</code></summary>

```
/-- Per-channel decay gate at time `t`, key channel `j_k`: `exp(w_t[j_k])`. -/
```
```lean
noncomputable def decay (s : BlockState) (w : RegionName)
    (s_k_h K BK : Nat) (t : Nat) (jk : Fin BK) : ℝ :=
  Real.exp (s.readMem w (s.pids 2 * s_k_h + s.pids 1 * BK + jk.val + t * K))
```
</details>

<details><summary><code>kVal</code></summary>

```lean
noncomputable def kVal (s : BlockState) (k : RegionName)
    (s_k_h K BK : Nat) (t : Nat) (jk : Fin BK) : ℝ :=
  s.readMem k (s.pids 2 * s_k_h + s.pids 1 * BK + jk.val + t * K)
```
</details>

<details><summary><code>vVal</code></summary>

```lean
noncomputable def vVal (s : BlockState) (v : RegionName)
    (s_v_h V BV : Nat) (t : Nat) (jv : Fin BV) : ℝ :=
  s.readMem v (s.pids 2 * s_v_h + s.pids 0 * BV + jv.val + t * V)
```
</details>

<details><summary><code>uVal</code></summary>

```lean
noncomputable def uVal (s : BlockState) (u : RegionName)
    (H K BK : Nat) (jk : Fin BK) : ℝ :=
  s.readMem u ((s.pids 2 % H) * K + jk.val + s.pids 1 * BK)
```
</details>

<details><summary><code>qVal</code></summary>

```
/-- `q[t][j_k]·scale` — the kernel multiplies the loaded `q` row by `scale`. -/
```
```lean
noncomputable def qVal (s : BlockState) (q : RegionName)
    (s_k_h K BK : Nat) (scale : ℝ) (t : Nat) (jk : Fin BK) : ℝ :=
  scale * s.readMem q (s.pids 2 * s_k_h + s.pids 1 * BK + jk.val + t * K)
```
</details>

<details><summary><code>h0Val</code></summary>

```lean
noncomputable def h0Val (s : BlockState) (h0 : RegionName)
    (K V BK BV : Nat) (idx : TileIndex [BV, BK]) : ℝ :=
  s.readMem h0 (finalStateOffset s K V BK BV idx)
```
</details>

## Also present (pinned special-case summaries)
- `fused_recurrent_rwkv6_state_step_slice_compute_correct`
- `fused_recurrent_rwkv6_output_step_slice_compute_correct`
- `fused_recurrent_rwkv6_final_state_store_slice_compute_correct`
- `fused_recurrent_rwkv6_python_test_case1_output_summary`
- `fused_recurrent_rwkv6_python_test_case2_output_summary`
- `fused_recurrent_rwkv6_python_test_case3_output_summary`
- `fused_recurrent_rwkv6_python_test_case4_output_summary`
