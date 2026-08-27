# Spec sheet — `bench/tritonbench_g/fused_rwkv6_kernel/FusedRwkv6Kernel.lean`

**Python source:** `bench/tritonbench_g/fused_rwkv6_kernel/fused_rwkv6_kernel.py`

## Public theorem: `fused_recurrent_rwkv6_output_summary_general`

<details><summary>docstring</summary>

```
/-! ### ════════ ★ MAIN THEOREM ★ ════════

**SCOPE — this is a claim about two hand-cut single-step slices, not about the
launched kernel.** Clauses 2 and 3 are `Realizes` facts about
`fused_recurrent_rwkv6_output_step_slice` and
`fused_recurrent_rwkv6_state_step_slice`; the launched surface
`fused_recurrent_rwkv6_fwd_surface` appears only in clause 1, which says nothing
more than "it lowers to the algorithm layer". The `range(0, T)` fold, the
`STORE_FINAL_STATE` writeback, and the host-side `o.sum(0)` over `NK` are all
outside this theorem (see the module docstring's modeling boundary).

Parameterized over the symbolic head strides `s_k_h s_v_h`, batch/head/time
`B H T`, key/value extents `K V`, tile sizes `BK BV`, the real `scale`, the step
index `m`, and both flags `USE_INITIAL_STATE STORE_FINAL_STATE`. The two step
faces are realized against the closed forms `stateClosed` / `outputClosed` over
the *input* regions (never a read-back of the kernel's own output):

1. the full RWKV6 forward surface lowers to the algorithm layer;
2. one **output** body realizes `outputClosed(m)` (the key-axis reduction),
   given the *assumed* carry invariant `BHPrev = stateClosed(m)` — **masked
   faithfully** (`mask_kv` on `prev`, `mask_bk` on `b_k`/`b_q`/`b_u`, `mask_bv`
   on `b_v` and on the store), with the key-axis summand correspondingly
   `activeK`-guarded, so this clause holds for partial tiles too;
3. one **state-update** body realizes `stateClosed(m+1)` (the per-channel decay
   step), given the same assumed carry invariant — **masked faithfully**
   (`mask_bk`/`mask_bv`/`mask_kv` loads and the `mask_kv` store), as a
   `writeIf (activeKV …)` statement about the in-range lanes;
4. the **cross-step carry fold**: chaining `T` state slices through one shared
   carry region `C` reaches `stateClosed(T)`, and leaves everything outside `C`
   untouched. This clause assumes *no* pinned carry — only that the initial
   buffer holds the seed on write-active lanes.

The pinned carry `BHPrev = stateClosed(m)` is a **clause-local antecedent** of
clauses 2 and 3 rather than a hypothesis of the theorem, precisely so that it
cannot weaken clause 4, whose whole point is not needing it.

Side conditions, all honest and all necessary:

* `BK ≤ K`, `BV ≤ V` — the tile fits the logical extents, giving offset
  injectivity for the state face and for the fold;
* `0 < BV` — contiguous output lanes, giving injectivity for the output face;
* inside clause 4 only: `k ≠ C`, `v ≠ C`, `w ≠ C` (a step must not overwrite its
  own inputs, or the closed form — which reads the *initial* state — would stop
  describing what later steps compute), the seed assumption on the initial
  buffer, and that the chain runs to completion (postcondition style: nothing
  here proves a stage terminates).

No clause carries a full-tile hypothesis: both slices mask exactly where Python
masks, and both `expected` closed forms are guarded to match. -/
```
</details>

**Statement:**
```lean
specification fused_recurrent_rwkv6_output_summary_general
    (q k v w u o h0 ht BHPrev BHOut C : RegionName)
    (USE_INITIAL_STATE STORE_FINAL_STATE : Bool)
    (s_k_h s_v_h B H T K V BK BV m : Nat) (scale : ℝ) (s sFinal : BlockState)
    (hBV : BV ≤ V) (hBK : BK ≤ K) (hBVpos : 0 < BV) :
    -- (1) the full surface lowers to the algorithm layer
    (∃ alg, (fused_recurrent_rwkv6_fwd_surface q k v w u o h0 ht
      s_k_h s_v_h B H T K V BK BV scale USE_INITIAL_STATE STORE_FINAL_STATE
      Bool.false).toAlgorithm? = Except.ok alg) ∧
    -- (2) the output body realizes the genuine `outputClosed(m)` — mask-faithful,
    --     so this holds for partial key/value tiles as well. The pinned carry is a
    --     clause-local antecedent, so it cannot weaken clause 4.
    ((∀ idx : TileIndex [BV, BK],
        s.readMem BHPrev (finalStateOffset s K V BK BV idx)
          = stateClosed s k v w h0 USE_INITIAL_STATE s_k_h s_v_h K V BK BV m idx) →
      ComputeCorrect.Realizes_without_Rounding
        (kernel := fused_recurrent_rwkv6_output_step_slice BHPrev q k v u o
          m s_k_h s_v_h B H T K V BK BV scale)
        (initialState := s)
        (write := ComputeCorrect.WriteMap.writeIf
          (fun jv : Fin BV => active s V BV jv)
          (fun jv => (o, outStepOffset s m s_v_h B H V BV jv)))
        (expected := fun jv : Fin BV =>
          outputClosed s q k v w u h0 USE_INITIAL_STATE s_k_h s_v_h H K V BK BV
            scale m jv)) ∧
    -- (3) the state-update body realizes the genuine `stateClosed(m+1)`, under the
    --     same clause-local pinned carry
    ((∀ idx : TileIndex [BV, BK],
        s.readMem BHPrev (finalStateOffset s K V BK BV idx)
          = stateClosed s k v w h0 USE_INITIAL_STATE s_k_h s_v_h K V BK BV m idx) →
      ComputeCorrect.Realizes_without_Rounding
        (kernel := fused_recurrent_rwkv6_state_step_slice BHPrev k v w BHOut
          m s_k_h s_v_h K V BK BV)
        (initialState := s)
        (write := ComputeCorrect.WriteMap.writeIf
          (fun idx : TileIndex [BV, BK] => activeKV s K V BK BV idx)
          (fun idx => (BHOut, finalStateOffset s K V BK BV idx)))
        (expected := fun idx =>
          stateClosed s k v w h0 USE_INITIAL_STATE s_k_h s_v_h K V BK BV
            (m + 1) idx)) ∧
    -- (4) the **cross-step carry fold**: chaining `T` state slices through one
    --     shared carry region `C` reaches `stateClosed(T)` from a *single*
    --     assumption about the initial buffer — no per-step pinned carry at all.
    --     Its own antecedents keep clauses 2 and 3 free of them.
    (k ≠ C → v ≠ C → w ≠ C →
      (∀ idx : TileIndex [BV, BK], activeKV s K V BK BV idx →
        s.readMem C (finalStateOffset s K V BK BV idx)
          = stateSeed s h0 USE_INITIAL_STATE K V BK BV idx) →
      execChain (foldStages
        (fun j => fused_recurrent_rwkv6_state_step_slice C k v w C
          j s_k_h s_v_h K V BK BV) T) s = some sFinal →
      (AgreeOutsideRegion C s sFinal ∧
        ∀ idx : TileIndex [BV, BK], activeKV s K V BK BV idx →
          sFinal.readMem C (finalStateOffset s K V BK BV idx)
            = stateClosed s k v w h0 USE_INITIAL_STATE s_k_h s_v_h K V BK BV T idx))
```

**Assumptions / layout contracts:**
- `hBV : BV ≤ V`
- `hBK : BK ≤ K`
- `hBVpos : 0 < BV`
- `∀ idx : TileIndex [BV, BK],
        s.readMem BHPrev (finalStateOffset s K V BK BV idx)
          = stateClosed s k v w h0 USE_INITIAL_STATE s_k_h s_v_h K V BK BV m idx`
- `fun jv : Fin BV => active s V BV jv`
- `∀ idx : TileIndex [BV, BK],
        s.readMem BHPrev (finalStateOffset s K V BK BV idx)
          = stateClosed s k v w h0 USE_INITIAL_STATE s_k_h s_v_h K V BK BV m idx`
- `fun idx : TileIndex [BV, BK] => activeKV s K V BK BV idx`
- `∀ idx : TileIndex [BV, BK], activeKV s K V BK BV idx →
        s.readMem C (finalStateOffset s K V BK BV idx)
          = stateSeed s h0 USE_INITIAL_STATE K V BK BV idx`
- `AgreeOutsideRegion C s sFinal ∧
        ∀ idx : TileIndex [BV, BK], activeKV s K V BK BV idx →
          sFinal.readMem C (finalStateOffset s K V BK BV idx)
            = stateClosed s k v w h0 USE_INITIAL_STATE s_k_h s_v_h K V BK BV T idx`

**Closed-form spec defs (transitive):** `fused_recurrent_rwkv6_fwd_surface`, `finalStateOffset`, `stateClosed`, `fused_recurrent_rwkv6_output_step_slice`, `active`, `outStepOffset`, `outputClosed`, `fused_recurrent_rwkv6_state_step_slice`, `activeKV`, `stateSeed`, `kIndex`, `vIndex`, `decay`, `kVal`, `vVal`, `activeK`, `uVal`, `qVal`, `h0Val`

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

<details><summary><code>finalStateOffset</code></summary>

```
/-- Flattened state-tile / `h0` address (row-major `i_bh·K·V + j_k·V + j_v`). -/
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
  mask_bk = (i_k * $(BK) + offs_k) < $(K)
  mask_bv = (i_v * $(BV) + offs_v) < $(V)
  mask_kv = mask_bv[:, None] & mask_bk[None, :]
  prev = tl.load(BHPrev + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[None, :]) * $(V) + (i_v * $(BV) + offs_v[:, None]),
    mask=mask_kv, other=0.0)
  b_k = tl.load(k + i_bh * $(s_k_h) + i_k * $(BK) + offs_k + $(t) * $(K),
    mask=mask_bk, other=0.0)
  b_v = tl.load(v + i_bh * $(s_v_h) + i_v * $(BV) + offs_v + $(t) * $(V),
    mask=mask_bv, other=0.0)
  b_q = tl.load(q + i_bh * $(s_k_h) + i_k * $(BK) + offs_k + $(t) * $(K),
    mask=mask_bk, other=0.0) * $(scale)
  b_u = tl.load(u + i_h * $(K) + offs_k + i_k * $(BK), mask=mask_bk, other=0.0)
  b_kv = b_k[None, :] * b_v[:, None]
  b_o = (prev + b_kv * b_u[None, :]) * b_q[None, :]
  b_o = tl.sum(b_o, axis=1)
  tl.store(o + (i_bh + i_k * $(B) * $(H)) * $(s_v_h) + i_v * $(BV) + offs_v +
    $(t) * $(V), (b_o).to(o.dtype.element_ty), mask=mask_bv)
}
```
</details>

<details><summary><code>active</code></summary>

```
/-- Python's `mask_bv` lane predicate (`i_v·BV + j_v < V`). -/
```
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

-- Raised over the pre-masking budget: the masked slice carries a guard on every
-- one of the five loads, so the executed register tower — and every defeq check
-- against it — is materially larger than the unmasked one used to be.
```
</details>

<details><summary><code>outputClosed</code></summary>

```
/-- **Genuine closed form for output row `m`, lane `j_v`** — the per-step
reduction over the *in-range* key channels (`mask_bk`, i.e. `activeK`):
`o_m[j_v] = Σ_{j_k ∈ mask_bk} (b_h^(m)[j_v,j_k] + k_m[j_k]·v_m[j_v]·u[j_k]) · scale·q_m[j_k]`,
reading the *pre-update* state `stateClosed(m)`. The `activeK` guard mirrors
Python's `other=0` loads: an out-of-range key column contributes nothing to
`tl.sum(_, axis=1)`. -/
```
```lean
noncomputable def outputClosed
    (s : BlockState) (q k v w u h0 : RegionName) (USE_INITIAL_STATE : Bool)
    (s_k_h s_v_h H K V BK BV : Nat) (scale : ℝ) (m : Nat) (jv : Fin BV) : ℝ :=
  ∑ jk : Fin BK,
    if activeK s K BK jk then
      (stateClosed s k v w h0 USE_INITIAL_STATE s_k_h s_v_h K V BK BV m
          (TileShape.insertAxisIndex [BV, BK] 1
            (TileShape.insertAxisIndex [BV] 0 PUnit.unit jv) jk)
        + (kVal s k s_k_h K BK m jk * vVal s v s_v_h V BV m jv) *
            uVal s u H K BK jk)
      * qVal s q s_k_h K BK scale m jk
    else 0
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
  mask_bk = (i_k * $(BK) + offs_k) < $(K)
  mask_bv = (i_v * $(BV) + offs_v) < $(V)
  mask_kv = mask_bv[:, None] & mask_bk[None, :]
  prev = tl.load(BHPrev + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[None, :]) * $(V) + (i_v * $(BV) + offs_v[:, None]),
    mask=mask_kv, other=0.0)
  b_k = tl.load(k + i_bh * $(s_k_h) + i_k * $(BK) + offs_k + $(t) * $(K),
    mask=mask_bk, other=0.0)
  b_v = tl.load(v + i_bh * $(s_v_h) + i_v * $(BV) + offs_v + $(t) * $(V),
    mask=mask_bv, other=0.0)
  b_w = tl.load(w + i_bh * $(s_k_h) + i_k * $(BK) + offs_k + $(t) * $(K),
    mask=mask_bk, other=0.0)
  b_w = tl.exp(b_w)
  b_kv = b_k[None, :] * b_v[:, None]
  acc = prev * b_w[None, :] + b_kv
  tl.store(BHOut + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[None, :]) * $(V) + (i_v * $(BV) + offs_v[:, None]),
    (acc).to(BHOut.dtype.element_ty), mask=mask_kv)
}
```
</details>

<details><summary><code>activeKV</code></summary>

```
/-- Python's `mask_kv = mask_bv[:, None] & mask_bk[None, :]` tile predicate. -/
```
```lean
def activeKV (s : BlockState) (K V BK BV : Nat) (idx : TileIndex [BV, BK]) : Prop :=
  active s V BV idx.1 ∧ activeK s K BK idx.2.1
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

<details><summary><code>decay</code></summary>

```
/-- Per-channel decay gate at time `t`, key channel `j_k`: `exp(w_t[j_k])`
(the `w` row read through the shared k-layout). -/
```
```lean
noncomputable def decay (s : BlockState) (w : RegionName)
    (s_k_h K BK : Nat) (t : Nat) (jk : Fin BK) : ℝ :=
  Real.exp (kVal s w s_k_h K BK t jk)
```
</details>

<details><summary><code>kVal</code></summary>

```
/-- Element `R[i_bh][t, j_k]` of the shared `[T, K]` **k-layout** at time row
`t`, key channel `j_k` (offset `i_bh·s_k_h + t·K + (i_k·BK + j_k)`). The `q`,
`k`, and `w` block pointers all use this layout — `qVal` and `decay` read
through it. -/
```
```lean
noncomputable def kVal (s : BlockState) (k : RegionName)
    (s_k_h K BK : Nat) (t : Nat) (jk : Fin BK) : ℝ :=
  s.readMem k (s.pids 2 * s_k_h + s.pids 1 * BK + jk.val + t * K)
```
</details>

<details><summary><code>vVal</code></summary>

```
/-- Element `v[i_bh][t, j_v]` of the `[T, V]` **v-layout** at time row `t`,
value channel `j_v` (offset `i_bh·s_v_h + t·V + (i_v·BV + j_v)`). -/
```
```lean
noncomputable def vVal (s : BlockState) (v : RegionName)
    (s_v_h V BV : Nat) (t : Nat) (jv : Fin BV) : ℝ :=
  s.readMem v (s.pids 2 * s_v_h + s.pids 0 * BV + jv.val + t * V)
```
</details>

<details><summary><code>activeK</code></summary>

```
/-- Python's `mask_bk` lane predicate (`i_k·BK + j_k < K`). -/
```
```lean
def activeK (s : BlockState) (K BK : Nat) (jk : Fin BK) : Prop :=
  kIndex s BK jk < K
```
</details>

<details><summary><code>uVal</code></summary>

```
/-- Per-head bonus row element `u[i_bh % H][j_k]` of the `[H, K]` row-major
`u` matrix (time-independent). -/
```
```lean
noncomputable def uVal (s : BlockState) (u : RegionName)
    (H K BK : Nat) (jk : Fin BK) : ℝ :=
  s.readMem u ((s.pids 2 % H) * K + jk.val + s.pids 1 * BK)
```
</details>

<details><summary><code>qVal</code></summary>

```
/-- `q[t][j_k]·scale` — the kernel multiplies the loaded `q` row (k-layout)
by `scale`. -/
```
```lean
noncomputable def qVal (s : BlockState) (q : RegionName)
    (s_k_h K BK : Nat) (scale : ℝ) (t : Nat) (jk : Fin BK) : ℝ :=
  scale * kVal s q s_k_h K BK t jk
```
</details>

<details><summary><code>h0Val</code></summary>

```lean
noncomputable def h0Val (s : BlockState) (h0 : RegionName)
    (K V BK BV : Nat) (idx : TileIndex [BV, BK]) : ℝ :=
  s.readMem h0 (finalStateOffset s K V BK BV idx)
```
</details>

## Public theorem: `fused_recurrent_rwkv6_state_step_io_correctness`

<details><summary>docstring</summary>

```
/-- **The headline on the IO surface** for `fused_recurrent_rwkv6.py`'s state
update: for every disjoint flat placement of `BHPrev` / `k` / `v` / `w` / `BHOut`,
every program coordinate whose active lanes are in bounds, and every launch state
whose four channels hold `xs`, `ks`, `vs`, `ws` on their active lanes, the kernel
terminates, every write-active lane of the next state tile holds

    xs[j_v, j_k] · exp(ws[j_k]) + ks[j_k] · vs[j_v]

and every other memory cell is unchanged. The decay's `tl.exp` and the `k ⊗ v`
outer product live entirely in the spec function; the four channels have three
different tile shapes, which is what `Masked3DTileShaped4KernelIO` exists for.

Dimension-general in `K`, `V`, `BK`, `BV` and both head strides. Honest
side-condition: state-address injectivity at every program coordinate, the same
hypothesis the per-write-map summary takes. -/
```
</details>

**Statement:**
```lean
specification fused_recurrent_rwkv6_state_step_io_correctness
    (BHPrev k v w BHOut : RegionName) (t s_k_h s_v_h K V BK BV : Nat)
    (hOutInj : ∀ p₀ p₁ p₂ : Nat, Function.Injective
      (fun idx : TileIndex [BV, BK] =>
        p₂ * K * V + (p₁ * BK + idx.2.1.val) * V + (p₀ * BV + idx.1.val))) :
    stateStepIO BHPrev k v w BHOut t s_k_h s_v_h K V BK BV
      ⊨ fun _p₀ _p₁ xs ks vs ws idx =>
          xs idx * Real.exp (ws (idx.2.1, PUnit.unit))
            + ks (idx.2.1, PUnit.unit) * vs (idx.1, PUnit.unit)
```

**Assumptions / layout contracts:**
- `fun idx : TileIndex [BV, BK] =>
        p₂ * K * V + (p₁ * BK + idx.2.1.val) * V + (p₀ * BV + idx.1.val)`

**Closed-form spec defs (transitive):** `stateStepIO`, `fused_recurrent_rwkv6_state_step_slice`

<details><summary><code>stateStepIO</code></summary>

```
/-- IO signature of the state step: the `[BV, BK]` state tile, the `[BK]` key and
decay rows and the `[BV]` value row produce the next `[BV, BK]` state tile. -/
```
```lean
noncomputable def stateStepIO (BHPrev k v w BHOut : RegionName)
    (t s_k_h s_v_h K V BK BV : Nat) : Masked3DTileShaped4KernelIO where
  kernel := fused_recurrent_rwkv6_state_step_slice BHPrev k v w BHOut t s_k_h
    s_v_h K V BK BV
  in1 := BHPrev
  in2 := k
  in3 := v
  in4 := w
  out := BHOut
  shape1 := [BV, BK]
  shape2 := [BK]
  shape3 := [BV]
  shape4 := [BK]
  shapeOut := [BV, BK]
  read1 := fun p₀ p₁ p₂ idx =>
    p₂ * K * V + (p₁ * BK + idx.2.1.val) * V + (p₀ * BV + idx.1.val)
  read2 := fun _p₀ p₁ p₂ jk => p₂ * s_k_h + p₁ * BK + jk.1.val + t * K
  read3 := fun p₀ _p₁ p₂ jv => p₂ * s_v_h + p₀ * BV + jv.1.val + t * V
  read4 := fun _p₀ p₁ p₂ jk => p₂ * s_k_h + p₁ * BK + jk.1.val + t * K
  write := fun p₀ p₁ p₂ idx =>
    p₂ * K * V + (p₁ * BK + idx.2.1.val) * V + (p₀ * BV + idx.1.val)
  mask1 := fun p₀ p₁ _p₂ idx =>
    p₀ * BV + idx.1.val < V ∧ p₁ * BK + idx.2.1.val < K
  mask2 := fun _p₀ p₁ _p₂ jk => p₁ * BK + jk.1.val < K
  mask3 := fun p₀ _p₁ _p₂ jv => p₀ * BV + jv.1.val < V
  mask4 := fun _p₀ p₁ _p₂ jk => p₁ * BK + jk.1.val < K
  writeMask := fun p₀ p₁ _p₂ idx =>
    p₀ * BV + idx.1.val < V ∧ p₁ * BK + idx.2.1.val < K
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
  mask_bk = (i_k * $(BK) + offs_k) < $(K)
  mask_bv = (i_v * $(BV) + offs_v) < $(V)
  mask_kv = mask_bv[:, None] & mask_bk[None, :]
  prev = tl.load(BHPrev + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[None, :]) * $(V) + (i_v * $(BV) + offs_v[:, None]),
    mask=mask_kv, other=0.0)
  b_k = tl.load(k + i_bh * $(s_k_h) + i_k * $(BK) + offs_k + $(t) * $(K),
    mask=mask_bk, other=0.0)
  b_v = tl.load(v + i_bh * $(s_v_h) + i_v * $(BV) + offs_v + $(t) * $(V),
    mask=mask_bv, other=0.0)
  b_w = tl.load(w + i_bh * $(s_k_h) + i_k * $(BK) + offs_k + $(t) * $(K),
    mask=mask_bk, other=0.0)
  b_w = tl.exp(b_w)
  b_kv = b_k[None, :] * b_v[:, None]
  acc = prev * b_w[None, :] + b_kv
  tl.store(BHOut + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[None, :]) * $(V) + (i_v * $(BV) + offs_v[:, None]),
    (acc).to(BHOut.dtype.element_ty), mask=mask_kv)
}
```
</details>

## Also present (pinned special-case summaries)
- `fused_recurrent_rwkv6_state_step_slice_compute_correct`
- `fused_recurrent_rwkv6_output_step_slice_compute_correct`
