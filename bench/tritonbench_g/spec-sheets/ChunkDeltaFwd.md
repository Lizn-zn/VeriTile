# Spec sheet — `bench/tritonbench_g/chunk_delta_fwd/ChunkDeltaFwd.lean`

**Python source:** `bench/tritonbench_g/chunk_delta_fwd/chunk_delta_fwd.py`

## Public theorem: `chunk_delta_fwd_output_summary_general`

<details><summary>docstring</summary>

```
/-- **Public dimension-general output summary.** For **arbitrary** symbolic
dimensions `T BT BC BK BV NT`, key/value strides
`s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d`, state strides `s_h_h s_h_t`, and
batch/head counts `_H K V`, the full chunk-delta forward surface

* lowers to the algorithm layer (`(...).toAlgorithm? = Except.ok _`); and
* under the genuine **producer hypotheses** `hBH`/`hBVN`/`hBHF` — asserting that
  the within-kernel cross-chunk fold materialized the genuine closed forms
  `hValue` (chunk-start state `H_{i_t}`), `vNewValue` (the corrected value
  `v − d·H_{i_t}`) and `finalValue` (`H_{NT}`) into the producer buffers
  `BH`/`BVN`/`BHFinal` — together with output-offset injectivity, each masked
  store face **realizes the genuine delta-rule recurrence** at every active lane:
  the `h[i_t]` store realizes `hValue i_t`, the `v_new[i_t]` store (single inner
  chunk, `BC = BT`) realizes `vNewValue i_t`, and the `final_state` store realizes
  `finalValue NT`.

The genuine recurrence specs (`stateValue`, `vNewValue`, `hValue`, `finalValue`)
are the closed forms over the **input** memory `k`/`v`/`d`/`initial_state`, never
an exec-readback. The producer hypotheses are honest explicit hypotheses on the
producer buffers (the same KIND of assumption as the `chunk_cumsum` carry
invariant); for the two checked Python shapes they are *discharged end-to-end
from the kernel `exec` with no producer hypotheses* by
`chunk_delta_fwd_python_case{1,2}_output_summary` below (which call
`chunk_delta_fwd_exec_genuine`). This headline statement carries **no concrete
dimension literals**: it is the genuine dimension-generalization of the recurrence
store faces. -/
```
</details>

**Statement:**
```lean
theorem chunk_delta_fwd_output_summary_general
    (k v d v_new h initial_state final_state BH BVN BHFinal : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t
      _H T K V BT BC BK BV NT : Nat)
    (USE_INITIAL_STATE STORE_FINAL_STATE : Bool)
    (s : BlockState)
    -- offset injectivity (per active store face) — honest side conditions
    (hInjH : ∀ i_t : Nat, Function.Injective
      (fun idx : TileIndex [BK, BV] => hOffset s i_t s_h_h s_h_t K V BK BV idx))
    (hInjV : ∀ i_t : Nat, Function.Injective
      (fun idx : TileIndex [BT, BV] => vNewOffset s i_t 0 s_vo_h s_vo_t s_vo_d BT BT BV idx))
    (hInjF : Function.Injective
      (fun idx : TileIndex [BK, BV] => finalStateOffset s K V BK BV idx))
    -- genuine producer hypotheses (symbolic dims) — the cross-chunk fold landed
    -- the genuine closed forms into the producer buffers
    (hBH : ∀ i_t : Nat, ∀ idx : TileIndex [BK, BV], active s K V BK BV idx →
        s.readMem BH (hOffset s i_t s_h_h s_h_t K V BK BV idx)
          = hValue s k v d initial_state s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
              K V BT BV BK USE_INITIAL_STATE i_t idx)
    (hBVN : ∀ i_t : Nat, ∀ idx : TileIndex [BT, BV], vNewActive s i_t 0 T V BT BT BV idx →
        s.readMem BVN (vNewOffset s i_t 0 s_vo_h s_vo_t s_vo_d BT BT BV idx)
          = vNewSpec s k v d initial_state s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
              K V BT BV BK USE_INITIAL_STATE i_t idx)
    (hBHF : ∀ idx : TileIndex [BK, BV], active s K V BK BV idx →
        s.readMem BHFinal (finalStateOffset s K V BK BV idx)
          = finalValue s k v d initial_state s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
              K V BT BV BK USE_INITIAL_STATE NT idx) :
    -- (i) the full surface lowers
    (∃ alg, (chunk_delta_rule_fwd_h_surface k v d v_new h initial_state final_state
        s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t
        _H T K V BT BC BK BV NT USE_INITIAL_STATE STORE_FINAL_STATE).toAlgorithm?
          = Except.ok alg)
    -- (ii) the state-store face realizes the genuine chunk-start state recurrence
    ∧ (∀ i_t : Nat, ComputeCorrect.Realizes
        (kernel := chunk_delta_fwd_h_store_slice BH h i_t s_h_h s_h_t K V BK BV)
        (initialState := s)
        (write := ComputeCorrect.WriteMap.writeIf
          (fun idx : TileIndex [BK, BV] => active s K V BK BV idx)
          (fun idx : TileIndex [BK, BV] => (h, hOffset s i_t s_h_h s_h_t K V BK BV idx)))
        (expected := fun idx : TileIndex [BK, BV] =>
          hValue s k v d initial_state s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
            K V BT BV BK USE_INITIAL_STATE i_t idx))
    -- (iii) the corrected-value store face realizes the genuine `vNewValue`
    ∧ (∀ i_t : Nat, ComputeCorrect.Realizes
        (kernel := chunk_delta_fwd_v_new_store_slice BVN v_new i_t 0
          s_vo_h s_vo_t s_vo_d T V BT BT BV)
        (initialState := s)
        (write := ComputeCorrect.WriteMap.writeIf
          (fun idx : TileIndex [BT, BV] => vNewActive s i_t 0 T V BT BT BV idx)
          (fun idx : TileIndex [BT, BV] =>
            (v_new, vNewOffset s i_t 0 s_vo_h s_vo_t s_vo_d BT BT BV idx)))
        (expected := fun idx : TileIndex [BT, BV] =>
          vNewSpec s k v d initial_state s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
            K V BT BV BK USE_INITIAL_STATE i_t idx))
    -- (iv) the final-state store face realizes `H_{NT}`
    ∧ ComputeCorrect.Realizes
        (kernel := chunk_delta_fwd_final_state_store_slice BHFinal final_state K V BK BV)
        (initialState := s)
        (write := ComputeCorrect.WriteMap.writeIf
          (fun idx : TileIndex [BK, BV] => active s K V BK BV idx)
          (fun idx : TileIndex [BK, BV] => (final_state, finalStateOffset s K V BK BV idx)))
        (expected := fun idx : TileIndex [BK, BV] =>
          finalValue s k v d initial_state s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
            K V BT BV BK USE_INITIAL_STATE NT idx)
```

**Assumptions / layout contracts:**
- `hInjH : ∀ i_t : Nat, Function.Injective
      (fun idx : TileIndex [BK, BV] => hOffset s i_t s_h_h s_h_t K V BK BV idx)`
- `hInjV : ∀ i_t : Nat, Function.Injective
      (fun idx : TileIndex [BT, BV] => vNewOffset s i_t 0 s_vo_h s_vo_t s_vo_d BT BT BV idx)`
- `hInjF : Function.Injective
      (fun idx : TileIndex [BK, BV] => finalStateOffset s K V BK BV idx)`
- `hBH : ∀ i_t : Nat, ∀ idx : TileIndex [BK, BV], active s K V BK BV idx →
        s.readMem BH (hOffset s i_t s_h_h s_h_t K V BK BV idx)
          = hValue s k v d initial_state s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
              K V BT BV BK USE_INITIAL_STATE i_t idx`
- `hBVN : ∀ i_t : Nat, ∀ idx : TileIndex [BT, BV], vNewActive s i_t 0 T V BT BT BV idx →
        s.readMem BVN (vNewOffset s i_t 0 s_vo_h s_vo_t s_vo_d BT BT BV idx)
          = vNewSpec s k v d initial_state s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
              K V BT BV BK USE_INITIAL_STATE i_t idx`
- `hBHF : ∀ idx : TileIndex [BK, BV], active s K V BK BV idx →
        s.readMem BHFinal (finalStateOffset s K V BK BV idx)
          = finalValue s k v d initial_state s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
              K V BT BV BK USE_INITIAL_STATE NT idx`
- `kernel : = chunk_delta_fwd_h_store_slice BH h i_t s_h_h s_h_t K V BK BV`
- `initialState : = s`
- `fun idx : TileIndex [BK, BV] => active s K V BK BV idx`
- `fun idx : TileIndex [BK, BV] => (h, hOffset s i_t s_h_h s_h_t K V BK BV idx)`
- `expected : = fun idx : TileIndex [BK, BV] =>
          hValue s k v d initial_state s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
            K V BT BV BK USE_INITIAL_STATE i_t idx`
- `kernel : = chunk_delta_fwd_v_new_store_slice BVN v_new i_t 0
          s_vo_h s_vo_t s_vo_d T V BT BT BV`
- `initialState : = s`
- `fun idx : TileIndex [BT, BV] => vNewActive s i_t 0 T V BT BT BV idx`
- `fun idx : TileIndex [BT, BV] =>
            (v_new, vNewOffset s i_t 0 s_vo_h s_vo_t s_vo_d BT BT BV idx)`
- `expected : = fun idx : TileIndex [BT, BV] =>
          vNewSpec s k v d initial_state s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
            K V BT BV BK USE_INITIAL_STATE i_t idx`
- `kernel : = chunk_delta_fwd_final_state_store_slice BHFinal final_state K V BK BV`
- `initialState : = s`
- `fun idx : TileIndex [BK, BV] => active s K V BK BV idx`
- `fun idx : TileIndex [BK, BV] => (final_state, finalStateOffset s K V BK BV idx)`
- `expected : = fun idx : TileIndex [BK, BV] =>
          finalValue s k v d initial_state s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
            K V BT BV BK USE_INITIAL_STATE NT idx`

**Closed-form spec defs (transitive):** `hOffset`, `vNewOffset`, `finalStateOffset`, `active`, `hValue`, `vNewActive`, `vNewSpec`, `finalValue`, `chunk_delta_rule_fwd_h_surface`, `chunk_delta_fwd_h_store_slice`, `chunk_delta_fwd_v_new_store_slice`, `chunk_delta_fwd_final_state_store_slice`, `kIndex`, `vIndex`, `cIndex`, `stateValue`, `vNewValue`, `initElem`, `kElem`, `vElem`, `dElem`

<details><summary><code>hOffset</code></summary>

```lean
def hOffset (s : BlockState) (i_t s_h_h s_h_t K V BK BV : Nat)
    (idx : TileIndex [BK, BV]) : Nat :=
  s.pids 2 * s_h_h + i_t * K * V +
    kIndex s BK idx.1 * s_h_t + vIndex s BV idx.2.1
```
</details>

<details><summary><code>vNewOffset</code></summary>

```lean
def vNewOffset (s : BlockState) (i_t i_c s_vo_h s_vo_t s_vo_d BT BC BV : Nat)
    (idx : TileIndex [BC, BV]) : Nat :=
  s.pids 2 * s_vo_h +
    (i_t * BT + i_c * BC + cIndex BC idx.1) * s_vo_t +
    vIndex s BV idx.2.1 * s_vo_d
```
</details>

<details><summary><code>finalStateOffset</code></summary>

```lean
def finalStateOffset (s : BlockState) (K V BK BV : Nat)
    (idx : TileIndex [BK, BV]) : Nat :=
  s.pids 2 * K * V + kIndex s BK idx.1 * V + vIndex s BV idx.2.1
```
</details>

<details><summary><code>active</code></summary>

```lean
def active (s : BlockState) (K V BK BV : Nat) (idx : TileIndex [BK, BV]) : Prop :=
  kIndex s BK idx.1 < K ∧ vIndex s BV idx.2.1 < V
```
</details>

<details><summary><code>hValue</code></summary>

```
/-- Stored `h[i_t]` tile lane `(e,p)`: the state `H_{i_t}[e,p]` at chunk start. -/
```
```lean
noncomputable def hValue (s : BlockState)
    (k v d initial_state : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d K V BT BV BK : Nat)
    (USE_INITIAL_STATE : Bool)
    (i_t : Nat) (idx : TileIndex [BK, BV]) : ℝ :=
  stateValue s k v d initial_state s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
    K V BT BV BK USE_INITIAL_STATE i_t (kIndex s BK idx.1) idx.2.1.val
```
</details>

<details><summary><code>vNewActive</code></summary>

```lean
def vNewActive (s : BlockState) (i_t i_c T V BT BC BV : Nat)
    (idx : TileIndex [BC, BV]) : Prop :=
  i_t * BT + i_c * BC + cIndex BC idx.1 < T ∧ vIndex s BV idx.2.1 < V
```
</details>

<details><summary><code>vNewSpec</code></summary>

```
/-- The corrected value tile lane `(c,p)` for inner chunk `i_c = 0` (the
single-inner-chunk regime `BC = BT`): the genuine `vNewValue`. -/
```
```lean
noncomputable def vNewSpec (s : BlockState)
    (k v d initial_state : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d K V BT BV BK : Nat)
    (USE_INITIAL_STATE : Bool)
    (i_t : Nat) (idx : TileIndex [BT, BV]) : ℝ :=
  vNewValue s k v d initial_state s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
    K V BT BV BK USE_INITIAL_STATE i_t idx.1.val idx.2.1.val
```
</details>

<details><summary><code>finalValue</code></summary>

```
/-- Final state tile lane `(e,p)`: `H_{NT}[e,p]`. -/
```
```lean
noncomputable def finalValue (s : BlockState)
    (k v d initial_state : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d K V BT BV BK : Nat)
    (USE_INITIAL_STATE : Bool)
    (NT : Nat) (idx : TileIndex [BK, BV]) : ℝ :=
  stateValue s k v d initial_state s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
    K V BT BV BK USE_INITIAL_STATE NT (kIndex s BK idx.1) idx.2.1.val
```
</details>

<details><summary><code>chunk_delta_rule_fwd_h_surface</code></summary>

```
/-- Faithful transcription of `chunk_delta_fwd.py`'s
`chunk_delta_rule_fwd_kernel_h`.

The source uses dynamic tile-dtype casts around the two dot products and
block-pointer element dtype casts on stores; this surface preserves those forms
alongside the nested `NT`/`ceil(BT/BC)` loop structure and optional
initial/final state paths. -/
```
```lean
def chunk_delta_rule_fwd_h_surface
    (k v d v_new h initial_state final_state : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t
      _H T K V BT BC BK BV NT : Nat)
    (USE_INITIAL_STATE STORE_FINAL_STATE : Bool) :
    ComputeKernel := triton {
  i_k = tl.program_id(0)
  i_v = tl.program_id(1)
  i_bh = tl.program_id(2)
  b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
  if USE_INITIAL_STATE {
    p_h0 = tl.make_block_ptr(base=initial_state + i_bh * $(K) * $(V),
      shape=($(K), $(V)), strides=($(V), $(1)),
      offsets=(i_k * $(BK), i_v * $(BV)), block_shape=($(BK), $(BV)), order=(1, 0))
    b_h = tl.load(p_h0, boundary_check=([0, 1] : List Nat)).to(tl.float32)
  }
  for i_t in range($(0), $(NT), $(1)) {
    p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V),
      shape=($(K), $(V)), strides=($(s_h_t), $(1)),
      offsets=(i_k * $(BK), i_v * $(BV)), block_shape=($(BK), $(BV)), order=(1, 0))
    tl.store(p_h, (b_h).to(p_h.dtype.element_ty), boundary_check=([0, 1] : List Nat))
    b_h_cumsum = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
    for i_c in range($(0), tl.cdiv($(BT), $(BC)), $(1)) {
      p_k = tl.make_block_ptr(base=k + i_bh * $(s_qk_h),
        shape=($(K), $(T)), strides=($(s_qk_d), $(s_qk_t)),
        offsets=(i_k * $(BK), i_t * $(BT) + i_c * $(BC)),
        block_shape=($(BK), $(BC)), order=(0, 1))
      p_d = tl.make_block_ptr(base=d + i_bh * $(s_qk_h),
        shape=($(T), $(K)), strides=($(s_qk_t), $(s_qk_d)),
        offsets=(i_t * $(BT) + i_c * $(BC), i_k * $(BK)),
        block_shape=($(BC), $(BK)), order=(1, 0))
      p_v = tl.make_block_ptr(base=v + i_bh * $(s_vo_h),
        shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)),
        offsets=(i_t * $(BT) + i_c * $(BC), i_v * $(BV)),
        block_shape=($(BC), $(BV)), order=(1, 0))
      p_v_new = tl.make_block_ptr(base=v_new + i_bh * $(s_vo_h),
        shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)),
        offsets=(i_t * $(BT) + i_c * $(BC), i_v * $(BV)),
        block_shape=($(BC), $(BV)), order=(1, 0))
      b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
      b_d = tl.load(p_d, boundary_check=([0, 1] : List Nat))
      b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
      b_v -= tl.dot(b_d, (b_h).to(b_k.dtype), allow_tf32=false)
      tl.store(p_v_new, (b_v).to(p_v_new.dtype.element_ty),
        boundary_check=([0, 1] : List Nat))
      b_h_cumsum += tl.dot(b_k, (b_v).to(b_k.dtype), allow_tf32=false)
    }
    b_h += b_h_cumsum
  }
  if STORE_FINAL_STATE {
    p_ht = tl.make_block_ptr(base=final_state + i_bh * $(K) * $(V),
      shape=($(K), $(V)), strides=($(V), $(1)),
      offsets=(i_k * $(BK), i_v * $(BV)), block_shape=($(BK), $(BV)), order=(1, 0))
    tl.store(p_ht, (b_h).to(p_ht.dtype.element_ty),
      boundary_check=([0, 1] : List Nat))
  }
}
```
</details>

<details><summary><code>chunk_delta_fwd_h_store_slice</code></summary>

```
/-- Proof-oriented state-store slice of `chunk_delta_fwd.py`'s
`chunk_delta_rule_fwd_kernel_h`. Models one `i_t` store from a precomputed `BH`
tile into `HOut`, preserving the source K/V block offsets and boundary checks. -/
```
```lean
def chunk_delta_fwd_h_store_slice
    (BH HOut : RegionName)
    (i_t s_h_h s_h_t K V BK BV : Nat) :
    ComputeKernel := triton {
  i_k = tl.program_id(0)
  i_v = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = i_k * $(BK) + tl.arange(0, $(BK))
  offs_v = i_v * $(BV) + tl.arange(0, $(BV))
  mask = (offs_k[:, None] < $(K)) & (offs_v[None, :] < $(V))
  b_h = tl.load(BH + i_bh * $(s_h_h) + $(i_t) * $(K) * $(V) +
      offs_k[:, None] * $(s_h_t) + offs_v[None, :], mask=mask, other=0.0)
  tl.store(HOut + i_bh * $(s_h_h) + $(i_t) * $(K) * $(V) +
      offs_k[:, None] * $(s_h_t) + offs_v[None, :], b_h, mask=mask)
}
```
</details>

<details><summary><code>chunk_delta_fwd_v_new_store_slice</code></summary>

```
/-- Proof-oriented v_new-store slice. Writes a precomputed `BVN` tile into `VNew`
at the per-iteration `(i_t, i_c)` chunk offsets. -/
```
```lean
def chunk_delta_fwd_v_new_store_slice
    (BVN VNew : RegionName)
    (i_t i_c s_vo_h s_vo_t s_vo_d T V BT BC BV : Nat) :
    ComputeKernel := triton {
  i_k = tl.program_id(0)
  i_v = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_c = tl.arange(0, $(BC))
  offs_v = i_v * $(BV) + tl.arange(0, $(BV))
  c_pos = $(i_t) * $(BT) + $(i_c) * $(BC) + offs_c[:, None]
  mask = (c_pos < $(T)) & (offs_v[None, :] < $(V))
  b_v = tl.load(BVN + i_bh * $(s_vo_h) + c_pos * $(s_vo_t) +
      offs_v[None, :] * $(s_vo_d), mask=mask, other=0.0)
  tl.store(VNew + i_bh * $(s_vo_h) + c_pos * $(s_vo_t) +
      offs_v[None, :] * $(s_vo_d), b_v, mask=mask)
}
```
</details>

<details><summary><code>chunk_delta_fwd_final_state_store_slice</code></summary>

```
/-- Proof-oriented final-state store slice. Writes a precomputed final-state
`BHFinal` tile into `FinalState` after the loop completes
(`STORE_FINAL_STATE = True`). -/
```
```lean
def chunk_delta_fwd_final_state_store_slice
    (BHFinal FinalState : RegionName) (K V BK BV : Nat) :
    ComputeKernel := triton {
  i_k = tl.program_id(0)
  i_v = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = i_k * $(BK) + tl.arange(0, $(BK))
  offs_v = i_v * $(BV) + tl.arange(0, $(BV))
  mask = (offs_k[:, None] < $(K)) & (offs_v[None, :] < $(V))
  b_h = tl.load(BHFinal + i_bh * $(K) * $(V) +
      offs_k[:, None] * $(V) + offs_v[None, :], mask=mask, other=0.0)
  tl.store(FinalState + i_bh * $(K) * $(V) +
      offs_k[:, None] * $(V) + offs_v[None, :], b_h, mask=mask)
}
```
</details>

<details><summary><code>kIndex</code></summary>

```
/-! ## Tile-lane index helpers and active region -/
```
```lean
def kIndex (s : BlockState) (BK : Nat) (i : Fin BK) : Nat :=
  s.pids 0 * BK + i.val
```
</details>

<details><summary><code>vIndex</code></summary>

```lean
def vIndex (s : BlockState) (BV : Nat) (j : Fin BV) : Nat :=
  s.pids 1 * BV + j.val
```
</details>

<details><summary><code>cIndex</code></summary>

```lean
def cIndex (BC : Nat) (i : Fin BC) : Nat :=
  i.val
```
</details>

<details><summary><code>stateValue</code></summary>

```
/-- Genuine closed form for the chunk-delta state recurrence in the
single-inner-chunk regime (`BC = BT`).

`stateValue i_t e p` is the state `H_{i_t}[e,p]` carried into chunk `i_t`
(`= h[i_t][e,p]`, the stored state). `H_0` is the seed (`initElem` when
`USE_INITIAL_STATE`, else `0`). The state advances by

```
  H_{i_t+1}[e,p] = H_{i_t}[e,p] + Σ_c k_{i_t}[e,c] · vNew_{i_t}[c,p]
```

where the corrected value `vNew_{i_t}[c,p] = v_{i_t}[c,p] − Σ_e d_{i_t}[c,e] ·
H_{i_t}[e,p]` is inlined into the advance step (the kernel computes it from the
*same* chunk-start state `H_{i_t}`, so the dependency is well-founded on `i_t`). -/
```
```lean
noncomputable def stateValue (s : BlockState)
    (k v d initial_state : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d K V BT BV BK : Nat)
    (USE_INITIAL_STATE : Bool) :
    Nat → Nat → Nat → ℝ
  | 0, e, p =>
      if USE_INITIAL_STATE then initElem s initial_state K V BV e p else 0
  | i_t + 1, e, p =>
      stateValue s k v d initial_state s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
          K V BT BV BK USE_INITIAL_STATE i_t e p
        + Finset.univ.sum (fun c : Fin BT =>
            kElem s k s_qk_h s_qk_t s_qk_d BT i_t e c.val
              * (vElem s v s_vo_h s_vo_t s_vo_d BT BV i_t c.val p
                  - Finset.univ.sum (fun e' : Fin BK =>
                      dElem s d s_qk_h s_qk_t s_qk_d BT i_t c.val e'.val
                        * stateValue s k v d initial_state s_qk_h s_qk_t s_qk_d
                            s_vo_h s_vo_t s_vo_d K V BT BV BK USE_INITIAL_STATE
                            i_t e'.val p)))
```
</details>

<details><summary><code>vNewValue</code></summary>

```
/-- The corrected value `v_new[i_t][c,p] = v − d · H_{i_t}` (a non-recursive
wrapper over the chunk-start state). -/
```
```lean
noncomputable def vNewValue (s : BlockState)
    (k v d initial_state : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d K V BT BV BK : Nat)
    (USE_INITIAL_STATE : Bool)
    (i_t c p : Nat) : ℝ :=
  vElem s v s_vo_h s_vo_t s_vo_d BT BV i_t c p
    - Finset.univ.sum (fun e : Fin BK =>
        dElem s d s_qk_h s_qk_t s_qk_d BT i_t c e.val
          * stateValue s k v d initial_state s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t
              s_vo_d K V BT BV BK USE_INITIAL_STATE i_t e.val p)
```
</details>

<details><summary><code>initElem</code></summary>

```
/-- `initial_state[e, p]` element (block ptr `(K,V)` strides `(V,1)`, offsets
`(0, i_v·BV)`): `initial_state` at `i_bh·K·V + e·V + (i_v·BV + p)`. -/
```
```lean
noncomputable def initElem (s : BlockState) (initial_state : RegionName)
    (K V BV : Nat) (e p : Nat) : ℝ :=
  s.readMem initial_state (s.pids 2 * K * V + e * V + (s.pids 1 * BV + p))
```
</details>

<details><summary><code>kElem</code></summary>

```
/-- `k[e, c]` element (block ptr `(K,T)` strides `(s_qk_d, s_qk_t)`, offsets
`(0, i_t·BT)`): `k` at `i_bh·s_qk_h + e·s_qk_d + (i_t·BT + c)·s_qk_t`. -/
```
```lean
noncomputable def kElem (s : BlockState) (k : RegionName)
    (s_qk_h s_qk_t s_qk_d BT : Nat) (i_t e c : Nat) : ℝ :=
  s.readMem k (s.pids 2 * s_qk_h + e * s_qk_d + (i_t * BT + c) * s_qk_t)
```
</details>

<details><summary><code>vElem</code></summary>

```
/-- `v[c, p]` element (block ptr `(T,V)` strides `(s_vo_t, s_vo_d)`, offsets
`(i_t·BT, i_v·BV)`): `v` at `i_bh·s_vo_h + (i_t·BT + c)·s_vo_t + (i_v·BV + p)·s_vo_d`. -/
```
```lean
noncomputable def vElem (s : BlockState) (v : RegionName)
    (s_vo_h s_vo_t s_vo_d BT BV : Nat) (i_t c p : Nat) : ℝ :=
  s.readMem v (s.pids 2 * s_vo_h + (i_t * BT + c) * s_vo_t + (s.pids 1 * BV + p) * s_vo_d)
```
</details>

<details><summary><code>dElem</code></summary>

```
/-- `d[c, e]` element (block ptr `(T,K)` strides `(s_qk_t, s_qk_d)`, offsets
`(i_t·BT, 0)`): `d` at `i_bh·s_qk_h + (i_t·BT + c)·s_qk_t + e·s_qk_d`. -/
```
```lean
noncomputable def dElem (s : BlockState) (d : RegionName)
    (s_qk_h s_qk_t s_qk_d BT : Nat) (i_t c e : Nat) : ℝ :=
  s.readMem d (s.pids 2 * s_qk_h + (i_t * BT + c) * s_qk_t + e * s_qk_d)
```
</details>

## Also present (pinned special-case summaries)
- `chunk_delta_fwd_python_case1_output_summary`
- `chunk_delta_fwd_python_case2_output_summary`
