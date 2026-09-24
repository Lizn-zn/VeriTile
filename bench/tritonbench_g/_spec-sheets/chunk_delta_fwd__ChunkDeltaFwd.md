# Spec sheet — `bench/tritonbench_g/chunk_delta_fwd/ChunkDeltaFwd.lean`

**Python source:** `bench/tritonbench_g/chunk_delta_fwd/chunk_delta_fwd.py`

## Public theorem: `chunk_delta_fwd_exec_genuine`

<details><summary>docstring</summary>

```
/-- **Genuine and end-to-end over the launched surface, dimension-general in every
stride and block size.** Executing the entire `chunk_delta_rule_fwd_h_surface`
(prologue + outer `forRange` over `NT` + epilogue) writes the genuine delta-rule
closed form into the output buffers — `h[j]` holds the chunk-start state
`hValue j`, `v_new[j]` the corrected `vNewSpec j`, and, when `STORE_FINAL_STATE`,
`final_state` holds `finalValue NT` — at every active lane, with the whole
cross-chunk carry fold *derived from the kernel `exec`* rather than assumed, and
with **no producer hypotheses at all.** The closed forms `hValue` / `vNewSpec` /
`finalValue` are over the input memory `k`/`v`/`d`/`initial_state`; none is an
exec read-back.

Every dimension and stride (`s_qk_h … s_h_t`, `T`, `K`, `V`, `BT`, `BC`, `BK`,
`BV`, `NT`) is a free variable. Five **regime** hypotheses replace what used to be
concrete literals; each is satisfied by the launcher and each is stated rather
than baked in:

* `hpids0 : s.pids 0 = 0` — the key-axis program id, guaranteed by the host's
  `assert NK == 1` but not proven here;
* `hBC : BC = BT` with `hBT : 0 < BT` — the single-inner-chunk regime
  (`ceil(BT/BC) = 1`); the launcher sets `BC = min(BT, 64)` for `BK ≤ 64`, so this
  holds whenever `BT ≤ 64`;
* `hBK : BK ≤ K` — no key-axis boundary lane inside the block; with the `NK == 1`
  assertion giving `K ≤ BK`, in practice `BK = K`;
* `hTNT : NT * BT ≤ T` — no partial trailing time chunk (`BT ∣ T` under the
  launcher's `NT = cdiv(T, BT)`);
* `hVod : 0 < s_vo_d`, `hHBlock : (BK - 1) * s_h_t + BV ≤ K * V` and
  `hVBlock : BV * s_vo_d ≤ s_vo_t` — the *block-fit* conditions that make distinct
  time chunks write disjoint memory; the launcher's contiguous layouts
  (`h.stride(2) = V`, `v_new.stride(3) = 1`) satisfy them with equality.

Also honest: twelve **region-distinctness** (`≠`) hypotheses — the outputs
`v_new`, `h`, `final_state` must not alias each other or the inputs, which the
store-order-sensitive multi-store fold genuinely needs — and three output-offset
injectivity hypotheses. The `USE_INITIAL_STATE` and `STORE_FINAL_STATE` flags flow
through symbolically.

Not covered: the host launch (the 3-D grid, autotuned warp counts, host-computed
`BK/BV/BC/NT`, and the `NK == 1` assertion), and the multi-inner-chunk regime
`BC < BT`. -/
```
</details>

**Statement:**
```lean
specification chunk_delta_fwd_exec_genuine
    (k v d v_new h initial_state final_state : RegionName)
    (USE_INITIAL_STATE STORE_FINAL_STATE : Bool)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t
      H T K V BT BC BK BV NT : Nat)
    (s : BlockState) (hpids0 : s.pids 0 = 0)
    (hBC : BC = BT) (hBT : 0 < BT) (hBK : BK ≤ K) (hTNT : NT * BT ≤ T)
    (hVod : 0 < s_vo_d)
    (hHBlock : (BK - 1) * s_h_t + BV ≤ K * V) (hVBlock : BV * s_vo_d ≤ s_vo_t)
    (hVk : v_new ≠ k) (hVv : v_new ≠ v) (hVd : v_new ≠ d) (hHv : h ≠ v_new)
    (hHk : h ≠ k) (hHv2 : h ≠ v) (hHd : h ≠ d)
    (hFh : final_state ≠ h) (hFv : final_state ≠ v_new) (hFk : final_state ≠ k)
    (hFv2 : final_state ≠ v) (hFd : final_state ≠ d)
    (hInjV : ∀ i_t : Fin NT, Function.Injective
      (fun idx : TileIndex [BC, BV] =>
        cdfVNewAddr s s_vo_h s_vo_t s_vo_d BT BC BV i_t.val idx))
    (hInjH : ∀ i_t : Fin NT, Function.Injective
      (fun idx : TileIndex [BK, BV] => hOffset s i_t.val s_h_h s_h_t K V BK BV idx))
    (hInjF : Function.Injective
      (fun idx : TileIndex [BK, BV] => finalStateOffset s K V BK BV idx)) :
    ∃ sF, exec (chunk_delta_rule_fwd_h_surface k v d v_new h initial_state final_state
        s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t H T K V BT BC BK BV NT
        USE_INITIAL_STATE STORE_FINAL_STATE).toAlgKernel s = some sF
      ∧ (∀ j : Fin NT, ∀ idx : TileIndex [BK, BV], active s K V BK BV idx →
          sF.readMem h (hOffset s j.val s_h_h s_h_t K V BK BV idx)
            = hValue s k v d initial_state s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
                K V BT BV BK USE_INITIAL_STATE j.val idx)
      ∧ (∀ j : Fin NT, ∀ idx : TileIndex [BC, BV],
          vNewActive s j.val 0 T V BT BC BV idx →
          sF.readMem v_new (cdfVNewAddr s s_vo_h s_vo_t s_vo_d BT BC BV j.val idx)
            = vNewSpec s k v d initial_state s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
                K V BT BV BK BC USE_INITIAL_STATE j.val idx)
      ∧ (STORE_FINAL_STATE = Bool.true →
          ∀ idx : TileIndex [BK, BV], active s K V BK BV idx →
            sF.readMem final_state (finalStateOffset s K V BK BV idx)
              = finalValue s k v d initial_state s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
                  K V BT BV BK USE_INITIAL_STATE NT idx)
```

**Assumptions / layout contracts:**
- `hpids0 : s.pids 0 = 0`
- `hBC : BC = BT`
- `hBT : 0 < BT`
- `hBK : BK ≤ K`
- `hTNT : NT * BT ≤ T`
- `hVod : 0 < s_vo_d`
- `hHBlock : (BK - 1) * s_h_t + BV ≤ K * V`
- `hVBlock : BV * s_vo_d ≤ s_vo_t`
- `hVk : v_new ≠ k`
- `hVv : v_new ≠ v`
- `hVd : v_new ≠ d`
- `hHv : h ≠ v_new`
- `hHk : h ≠ k`
- `hHv2 : h ≠ v`
- `hHd : h ≠ d`
- `hFh : final_state ≠ h`
- `hFv : final_state ≠ v_new`
- `hFk : final_state ≠ k`
- `hFv2 : final_state ≠ v`
- `hFd : final_state ≠ d`
- `hInjV : ∀ i_t : Fin NT, Function.Injective
      (fun idx : TileIndex [BC, BV] =>
        cdfVNewAddr s s_vo_h s_vo_t s_vo_d BT BC BV i_t.val idx)`
- `hInjH : ∀ i_t : Fin NT, Function.Injective
      (fun idx : TileIndex [BK, BV] => hOffset s i_t.val s_h_h s_h_t K V BK BV idx)`
- `hInjF : Function.Injective
      (fun idx : TileIndex [BK, BV] => finalStateOffset s K V BK BV idx)`

**Closed-form spec defs (transitive):** `cdfVNewAddr`, `hOffset`, `finalStateOffset`, `chunk_delta_rule_fwd_h_surface`, `active`, `hValue`, `vNewActive`, `vNewSpec`, `finalValue`, `kIndex`, `vIndex`, `stateValue`, `cIndex`, `vNewValue`, `initElem`, `kElem`, `vElem`, `dElem`

<details><summary><code>cdfVNewAddr</code></summary>

```
/-- The block-ptr `v_new` store offset at lane `(c,p)` (inner chunk `i_c=0`). -/
```
```lean
def cdfVNewAddr (s : BlockState) (s_vo_h s_vo_t s_vo_d BT BC BV : Nat)
    (i_t : Nat) (idx : TileIndex [BC, BV]) : Nat :=
  s.pids 2 * s_vo_h + (i_t * BT + 0 * BC + idx.1.val) * s_vo_t
    + (s.pids 1 * BV + idx.2.1.val) * s_vo_d
```
</details>

<details><summary><code>hOffset</code></summary>

```lean
def hOffset (s : BlockState) (i_t s_h_h s_h_t K V BK BV : Nat)
    (idx : TileIndex [BK, BV]) : Nat :=
  s.pids 2 * s_h_h + i_t * K * V +
    kIndex s BK idx.1 * s_h_t + vIndex s BV idx.2.1
```
</details>

<details><summary><code>finalStateOffset</code></summary>

```
/-- **Corrected-value store face realizes the genuine recurrence.** Under `hBVN`
(the producer materialized `vNewValue` into `BVN`) and offset injectivity, the
kernel's `v_new` store realizes `vNewValue` at every active lane (inner chunk
`i_c = 0`, `BC = BT`). -/
```
```lean
def finalStateOffset (s : BlockState) (K V BK BV : Nat)
    (idx : TileIndex [BK, BV]) : Nat :=
  s.pids 2 * K * V + kIndex s BK idx.1 * V + vIndex s BV idx.2.1
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

```lean
noncomputable def vNewSpec (s : BlockState)
    (k v d initial_state : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d K V BT BV BK BC : Nat)
    (USE_INITIAL_STATE : Bool)
    (i_t : Nat) (idx : TileIndex [BC, BV]) : ℝ :=
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

<details><summary><code>cIndex</code></summary>

```lean
def cIndex (BC : Nat) (i : Fin BC) : Nat :=
  i.val
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

## Public theorem: `chunk_delta_h_state_store_io_correctness`

<details><summary>docstring</summary>

```
/-- **The headline on the IO surface** for `chunk_delta_rule_fwd_h_surface`'s
per-chunk `h` writeback: for every disjoint flat placement of `HPre` / `h`, every
program coordinate whose active lanes are in bounds, and every launch state whose
`HPre` block holds `xs` at the active lanes, the writeback terminates, every
active lane of `h` holds `xs idx`, and every other memory cell is unchanged.

**Scope.** `HPre` is a **fiction region**: it stands for the kernel's carry
register `b_h`, not for a Python tensor — the same idiom as
`fused_recurrent_retention`'s `HSeed`. The delta-rule recurrence that produces the
carry keeps its own closed-form summary above; this face is the writeback's
contract, and the two meet at the carry. The block pointer's
`boundary_check = [0, 1]` is transcribed as the equivalent lane mask, exactly as
in the twin `chunk_gated_attention` surface.

Dimension-general in `i_t`, `s_h_h`, `s_h_t`, `K`, `V`, `BK` and `BV`. Honest
side-condition: address injectivity at every program coordinate, the same
hypothesis the per-write-map summaries take. -/
```
</details>

**Statement:**
```lean
specification chunk_delta_h_state_store_io_correctness
    (HPre h : RegionName) (i_t s_h_h s_h_t K V BK BV : Nat)
    (hInj : ∀ p₀ p₁ p₂ : Nat, Function.Injective
      (fun idx : TileIndex [BK, BV] =>
        p₂ * s_h_h + i_t * K * V + (p₀ * BK + idx.1.val) * s_h_t
          + (p₁ * BV + idx.2.1.val))) :
    h_stateIO HPre h i_t s_h_h s_h_t K V BK BV
      ⊨ fun _p₀ _p₁ xs idx => xs idx
```

**Closed-form spec defs (transitive):** `h_stateIO`, `chunk_delta_h_state_store_slice`

<details><summary><code>h_stateIO</code></summary>

```
/-- IO signature of the per-chunk `h` writeback on the three-axis tile surface. -/
```
```lean
def h_stateIO (HPre h : RegionName) (i_t s_h_h s_h_t K V BK BV : Nat) :
    Masked3DTileKernelIO₁ where
  kernel := chunk_delta_h_state_store_slice HPre h i_t s_h_h s_h_t K V BK BV
  inp := HPre
  out := h
  shape := [BK, BV]
  read := fun p₀ p₁ p₂ idx =>
    p₂ * s_h_h + i_t * K * V + (p₀ * BK + idx.1.val) * s_h_t
      + (p₁ * BV + idx.2.1.val)
  write := fun p₀ p₁ p₂ idx =>
    p₂ * s_h_h + i_t * K * V + (p₀ * BK + idx.1.val) * s_h_t
      + (p₁ * BV + idx.2.1.val)
  mask := fun p₀ p₁ _p₂ idx =>
    p₀ * BK + idx.1.val < K ∧ p₁ * BV + idx.2.1.val < V
```
</details>

<details><summary><code>chunk_delta_h_state_store_slice</code></summary>

```
/-- The per-chunk `h` writeback of `chunk_delta_rule_fwd_h_surface`, with the
carry register materialized into `HPre` and the loop variable `i_t` a parameter. -/
```
```lean
def chunk_delta_h_state_store_slice
    (HPre h : RegionName) (i_t s_h_h s_h_t K V BK BV : Nat) :
    ComputeKernel := triton {
  i_k = tl.program_id(0)
  i_v = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = i_k * $(BK) + tl.arange(0, $(BK))
  offs_v = i_v * $(BV) + tl.arange(0, $(BV))
  mask = (offs_k[:, None] < $(K)) & (offs_v[None, :] < $(V))
  b_h = tl.load(HPre + i_bh * $(s_h_h) + $(i_t) * $(K) * $(V) +
      offs_k[:, None] * $(s_h_t) + offs_v[None, :],
    mask=mask, other=0.0)
  tl.store(h + i_bh * $(s_h_h) + $(i_t) * $(K) * $(V) +
      offs_k[:, None] * $(s_h_t) + offs_v[None, :],
    (b_h).to(h.dtype.element_ty), mask=mask)
}
```
</details>

## Public theorem: `chunk_delta_h_state_store_io_correctnessR`

<details><summary>docstring</summary>

```
/-- **The `⊨[R]` headline** for the per-chunk `h` writeback: for **every**
rounding model `R`, the same masked Hoare triple as
`chunk_delta_h_state_store_io_correctness`, run under `execR R` and read back as
`.real`-typed cells holding `R.round .real (xs idx)`.

The writeback is a pure copy — no arithmetic, and the store's `.to(...)` erases to
`.real` — so the slice is cast-free and the exact run transports verbatim. The
content of the rounding face is exactly that: *the writeback introduces no
rounding event of its own*, at any `R`. Same scope caveat as the exact face:
`HPre` is a fiction region standing for the carry register `b_h`. -/
```
</details>

**Statement:**
```lean
specification chunk_delta_h_state_store_io_correctnessR
    (R : RoundingModel) (HPre h : RegionName) (i_t s_h_h s_h_t K V BK BV : Nat)
    (hInj : ∀ p₀ p₁ p₂ : Nat, Function.Injective
      (fun idx : TileIndex [BK, BV] =>
        p₂ * s_h_h + i_t * K * V + (p₀ * BK + idx.1.val) * s_h_t
          + (p₁ * BV + idx.2.1.val))) :
    h_stateIO HPre h i_t s_h_h s_h_t K V BK BV
      ⊨[R, FloatDType.real] fun _p₀ _p₁ xs idx => xs idx
```

**Closed-form spec defs (transitive):** `h_stateIO`, `chunk_delta_h_state_store_slice`

<details><summary><code>h_stateIO</code></summary>

```
/-- IO signature of the per-chunk `h` writeback on the three-axis tile surface. -/
```
```lean
def h_stateIO (HPre h : RegionName) (i_t s_h_h s_h_t K V BK BV : Nat) :
    Masked3DTileKernelIO₁ where
  kernel := chunk_delta_h_state_store_slice HPre h i_t s_h_h s_h_t K V BK BV
  inp := HPre
  out := h
  shape := [BK, BV]
  read := fun p₀ p₁ p₂ idx =>
    p₂ * s_h_h + i_t * K * V + (p₀ * BK + idx.1.val) * s_h_t
      + (p₁ * BV + idx.2.1.val)
  write := fun p₀ p₁ p₂ idx =>
    p₂ * s_h_h + i_t * K * V + (p₀ * BK + idx.1.val) * s_h_t
      + (p₁ * BV + idx.2.1.val)
  mask := fun p₀ p₁ _p₂ idx =>
    p₀ * BK + idx.1.val < K ∧ p₁ * BV + idx.2.1.val < V
```
</details>

<details><summary><code>chunk_delta_h_state_store_slice</code></summary>

```
/-- The per-chunk `h` writeback of `chunk_delta_rule_fwd_h_surface`, with the
carry register materialized into `HPre` and the loop variable `i_t` a parameter. -/
```
```lean
def chunk_delta_h_state_store_slice
    (HPre h : RegionName) (i_t s_h_h s_h_t K V BK BV : Nat) :
    ComputeKernel := triton {
  i_k = tl.program_id(0)
  i_v = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = i_k * $(BK) + tl.arange(0, $(BK))
  offs_v = i_v * $(BV) + tl.arange(0, $(BV))
  mask = (offs_k[:, None] < $(K)) & (offs_v[None, :] < $(V))
  b_h = tl.load(HPre + i_bh * $(s_h_h) + $(i_t) * $(K) * $(V) +
      offs_k[:, None] * $(s_h_t) + offs_v[None, :],
    mask=mask, other=0.0)
  tl.store(h + i_bh * $(s_h_h) + $(i_t) * $(K) * $(V) +
      offs_k[:, None] * $(s_h_t) + offs_v[None, :],
    (b_h).to(h.dtype.element_ty), mask=mask)
}
```
</details>
