# Spec sheet — `bench/tritonbench_g/chunk_linear_attn/ChunkLinearAttn.lean`

**Python source:** `bench/tritonbench_g/chunk_linear_attn/chunk_linear_attn.py`

## Public theorem: `cla_fwd_h_exec_genuine`

<details><summary>docstring</summary>

```
/-- **Genuine, dimension-general correctness** of the file's first kernel,
`chunk_linear_attn_fwd_kernel_h`. For every launch state the kernel runs to
completion; every active lane of every chunk block `h[·,·,t]` (`t < NT`) holds
the **pre-chunk** state `claHState t` — the gated `h0` seed plus every chunk
before `t` — and, when `STORE_FINAL_STATE` is set, `ht` holds the post-loop
state `claHState NT`. One theorem covers all four gate configurations, with
every dimension, stride, and the chunk count symbolic.

The three layout hypotheses are equalities/immediate under the launcher's
contiguous `h` (`s_h_t = V`): `BV ≤ s_h_t` makes one block's store lanes
injective, and the block-fit `(K-1)·s_h_t + V ≤ K·V` makes distinct chunks'
active lanes land in disjoint `K·V` blocks. -/
```
</details>

**Statement:**
```lean
specification cla_fwd_h_exec_genuine
    (k v h h0 ht : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat)
    (T K V BT BK BV NT : Nat) (USE_INITIAL_STATE STORE_FINAL_STATE : Bool)
    (s : BlockState)
    (hHk : h ≠ k) (hHv : h ≠ v) (hHtH : ht ≠ h)
    (hσ : BV ≤ s_h_t) (hFit : (K - 1) * s_h_t + V ≤ K * V) (hBVV : BV ≤ V) :
    ∃ sF, exec (cla_fwd_h_surface k v h h0 ht s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t
        s_vo_d s_h_h s_h_t T K V BT BK BV NT
        USE_INITIAL_STATE STORE_FINAL_STATE).toAlgKernel s = some sF
      ∧ (∀ t (idx : TileIndex [BK, BV]), t < NT → claActive s K V BK BV idx →
          sF.readMem h (claHOffset s s_h_h s_h_t K V BK BV t idx)
            = claHState s k v h0 USE_INITIAL_STATE s_qk_h s_qk_t s_qk_d s_vo_h
                s_vo_t s_vo_d T K V BT BK BV t idx.1.val idx.2.1.val)
      ∧ (STORE_FINAL_STATE = Bool.true →
          ∀ idx : TileIndex [BK, BV], claActive s K V BK BV idx →
          sF.readMem ht (claHtOffset s K V BK BV idx)
            = claHState s k v h0 USE_INITIAL_STATE s_qk_h s_qk_t s_qk_d s_vo_h
                s_vo_t s_vo_d T K V BT BK BV NT idx.1.val idx.2.1.val)
```

**Assumptions / layout contracts:**
- `hHk : h ≠ k`
- `hHv : h ≠ v`
- `hHtH : ht ≠ h`
- `hσ : BV ≤ s_h_t`
- `hFit : (K - 1) * s_h_t + V ≤ K * V`
- `hBVV : BV ≤ V`
- `STORE_FINAL_STATE = Bool.true →
          ∀ idx : TileIndex [BK, BV], claActive s K V BK BV idx →
          sF.readMem ht (claHtOffset s K V BK BV idx)
            = claHState s k v h0 USE_INITIAL_STATE s_qk_h s_qk_t s_qk_d s_vo_h
                s_vo_t s_vo_d T K V BT BK BV NT idx.1.val idx.2.1.val`

**Closed-form spec defs (transitive):** `cla_fwd_h_surface`, `claActive`, `claHOffset`, `claHState`, `claHtOffset`, `h0Guarded`, `claContrib`, `h0Elem`, `kGuarded`, `vGuarded`, `kElem`, `vElem`

<details><summary><code>cla_fwd_h_surface</code></summary>

```
/-- Faithful transcription of `chunk_linear_attn_fwd_kernel_h`. -/
```
```lean
def cla_fwd_h_surface
    (k v h h0 ht : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat)
    (T K V BT BK BV NT : Nat)
    (USE_INITIAL_STATE STORE_FINAL_STATE : Bool) :
    ComputeKernel := triton {
  i_k = tl.program_id(0)
  i_v = tl.program_id(1)
  i_bh = tl.program_id(2)
  b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
  if USE_INITIAL_STATE {
    p_h0 = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V),
      shape=($(K), $(V)), strides=($(V), $(1)),
      offsets=(i_k * $(BK), i_v * $(BV)), block_shape=($(BK), $(BV)), order=(1, 0))
    b_h = (tl.load(p_h0, boundary_check=([0, 1] : List Nat))).to(tl.float32)
  }
  for i_t in range($(0), $(NT), $(1)) {
    p_k = tl.make_block_ptr(base=k + i_bh * $(s_qk_h),
      shape=($(K), $(T)), strides=($(s_qk_d), $(s_qk_t)),
      offsets=(i_k * $(BK), i_t * $(BT)), block_shape=($(BK), $(BT)), order=(0, 1))
    p_v = tl.make_block_ptr(base=v + i_bh * $(s_vo_h),
      shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)),
      offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
    p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V),
      shape=($(K), $(V)), strides=($(s_h_t), $(1)),
      offsets=(i_k * $(BK), i_v * $(BV)), block_shape=($(BK), $(BV)), order=(1, 0))
    tl.store(p_h, (b_h).to(p_h.dtype.element_ty), boundary_check=([0, 1] : List Nat))
    b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
    b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
    b_h += tl.dot(b_k, b_v, allow_tf32=false)
  }
  if STORE_FINAL_STATE {
    p_ht = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V),
      shape=($(K), $(V)), strides=($(V), $(1)),
      offsets=(i_k * $(BK), i_v * $(BV)), block_shape=($(BK), $(BV)), order=(1, 0))
    tl.store(p_ht, (b_h).to(p_ht.dtype.element_ty), boundary_check=([0, 1] : List Nat))
  }
}
```
</details>

<details><summary><code>claActive</code></summary>

```
/-- A state lane is *active* when it maps inside the `K × V` window. -/
```
```lean
def claActive (s : BlockState) (K V BK BV : Nat)
    (idx : TileIndex [BK, BV]) : Prop :=
  s.pids 0 * BK + idx.1.val < K ∧ s.pids 1 * BV + idx.2.1.val < V
```
</details>

<details><summary><code>claHOffset</code></summary>

```
/-- The `h` / `dh` chunk-`t` store address for lane `(e, p)`. -/
```
```lean
def claHOffset (s : BlockState) (s_h_h s_h_t K V BK BV : Nat) (t : Nat)
    (idx : TileIndex [BK, BV]) : Nat :=
  s.pids 2 * s_h_h + t * K * V + (s.pids 0 * BK + idx.1.val) * s_h_t
    + (s.pids 1 * BV + idx.2.1.val) * 1
```
</details>

<details><summary><code>claHState</code></summary>

```
/-- **The forward state at chunk `t`** — what the kernel stores into `h[·,·,t]`:
the (gated) initial state plus every chunk *before* `t`. -/
```
```lean
noncomputable def claHState (s : BlockState) (k v h0 : RegionName)
    (UIS : Bool) (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat)
    (T K V BT BK BV : Nat) (t e p : Nat) : ℝ :=
  (if UIS then h0Guarded s h0 K V BK BV e p else 0)
    + ∑ u : Fin t, claContrib s k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
        T K V BT BK BV u.val e p
```
</details>

<details><summary><code>claHtOffset</code></summary>

```
/-- The `ht` store address for lane `(e, p)`. -/
```
```lean
def claHtOffset (s : BlockState) (K V BK BV : Nat)
    (idx : TileIndex [BK, BV]) : Nat :=
  s.pids 2 * K * V + (s.pids 0 * BK + idx.1.val) * V
    + (s.pids 1 * BV + idx.2.1.val) * 1
```
</details>

<details><summary><code>h0Guarded</code></summary>

```
/-- The guarded initial state, as the `USE_INITIAL_STATE` branch leaves `b_h`. -/
```
```lean
noncomputable def h0Guarded (s : BlockState) (h0 : RegionName)
    (K V BK BV : Nat) (e p : Nat) : ℝ :=
  if s.pids 0 * BK + e < K ∧ s.pids 1 * BV + p < V then
    h0Elem s h0 K V BK BV e p
  else 0
```
</details>

<details><summary><code>claContrib</code></summary>

```
/-- One chunk's contribution to state lane `(e, p)`: `Σ_c k[e, c] · v[c, p]`. -/
```
```lean
noncomputable def claContrib (s : BlockState) (k v : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat)
    (T K V BT BK BV : Nat) (t e p : Nat) : ℝ :=
  ∑ c : Fin BT,
    kGuarded s k s_qk_h s_qk_t s_qk_d T K BT BK t c.val e
      * vGuarded s v s_vo_h s_vo_t s_vo_d T V BT BV t c.val p
```
</details>

<details><summary><code>h0Elem</code></summary>

```
/-- `h0[i_k·BK + e, i_v·BV + p]` (parent `(K, V)`, strides `(V, 1)`). -/
```
```lean
noncomputable def h0Elem (s : BlockState) (h0 : RegionName)
    (K V BK BV : Nat) (e p : Nat) : ℝ :=
  s.readMem h0 (s.pids 2 * K * V + (s.pids 0 * BK + e) * V
    + (s.pids 1 * BV + p) * 1)
```
</details>

<details><summary><code>kGuarded</code></summary>

```
/-- The guarded `k` lane, as `b_k` holds it. -/
```
```lean
noncomputable def kGuarded (s : BlockState) (k : RegionName)
    (s_qk_h s_qk_t s_qk_d : Nat) (T K BT BK : Nat) (t c e : Nat) : ℝ :=
  if s.pids 0 * BK + e < K ∧ t * BT + c < T then
    kElem s k s_qk_h s_qk_t s_qk_d BT BK t c e
  else 0
```
</details>

<details><summary><code>vGuarded</code></summary>

```
/-- The guarded `v` lane, as `b_v` holds it. -/
```
```lean
noncomputable def vGuarded (s : BlockState) (v : RegionName)
    (s_vo_h s_vo_t s_vo_d : Nat) (T V BT BV : Nat) (t c p : Nat) : ℝ :=
  if t * BT + c < T ∧ s.pids 1 * BV + p < V then
    vElem s v s_vo_h s_vo_t s_vo_d BT BV t c p
  else 0
```
</details>

<details><summary><code>kElem</code></summary>

```
/-- `k[i_k·BK + e, t·BT + c]` (parent `(K, T)`, strides `(s_qk_d, s_qk_t)`). -/
```
```lean
noncomputable def kElem (s : BlockState) (k : RegionName)
    (s_qk_h s_qk_t s_qk_d BT BK : Nat) (t c e : Nat) : ℝ :=
  s.readMem k (s.pids 2 * s_qk_h + (s.pids 0 * BK + e) * s_qk_d
    + (t * BT + c) * s_qk_t)
```
</details>

<details><summary><code>vElem</code></summary>

```
/-- `v[t·BT + c, i_v·BV + p]` (parent `(T, V)`, strides `(s_vo_t, s_vo_d)`). -/
```
```lean
noncomputable def vElem (s : BlockState) (v : RegionName)
    (s_vo_h s_vo_t s_vo_d BT BV : Nat) (t c p : Nat) : ℝ :=
  s.readMem v (s.pids 2 * s_vo_h + (t * BT + c) * s_vo_t
    + (s.pids 1 * BV + p) * s_vo_d)
```
</details>

## Public theorem: `cla_bwd_dh_exec_genuine`

<details><summary>docstring</summary>

```
/-- **Genuine, dimension-general correctness** of `chunk_linear_attn_bwd_kernel_dh`
(the descending loop, verified through its ascending change of variable). For
every launch state the kernel runs to completion and every active lane of every
chunk block `dh[·,·,t]` (`t < NT`) holds `claDhState t`: the sum of
`(scale·q_u)ᵀ·do_u` over the strictly-later chunks `u > t` — the state *before*
chunk `t`'s own contribution, exactly as the descending store-then-accumulate
order produces. -/
```
</details>

**Statement:**
```lean
specification cla_bwd_dh_exec_genuine
    (q do_ dh : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV NT : Nat) (s : BlockState)
    (hDq : dh ≠ q) (hDdo : dh ≠ do_)
    (hσ : BV ≤ s_h_t) (hFit : (K - 1) * s_h_t + V ≤ K * V) :
    ∃ sF, exec (cla_bwd_dh_surface q do_ dh s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t
        s_vo_d s_h_h s_h_t scale T K V BT BK BV NT).toAlgKernel s = some sF
      ∧ ∀ t (idx : TileIndex [BK, BV]), t < NT → claActive s K V BK BV idx →
          sF.readMem dh (claHOffset s s_h_h s_h_t K V BK BV t idx)
            = claDhState s q do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
                T K V BT BK BV NT t idx.1.val idx.2.1.val
```

**Assumptions / layout contracts:**
- `hDq : dh ≠ q`
- `hDdo : dh ≠ do_`
- `hσ : BV ≤ s_h_t`
- `hFit : (K - 1) * s_h_t + V ≤ K * V`

**Closed-form spec defs (transitive):** `cla_bwd_dh_surface`, `claActive`, `claHOffset`, `claDhState`, `claBContrib`, `kGuarded`, `vGuarded`, `kElem`, `vElem`

<details><summary><code>cla_bwd_dh_surface</code></summary>

```
/-- Faithful transcription of `chunk_linear_attn_bwd_kernel_dh`, with the
descending loop spelled as its ascending change of variable (see the preamble). -/
```
```lean
def cla_bwd_dh_surface
    (q do_ dh : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV NT : Nat) :
    ComputeKernel := triton {
  i_k = tl.program_id(0)
  i_v = tl.program_id(1)
  i_bh = tl.program_id(2)
  b_dh = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
  for j in range($(0), $(NT), $(1)) {
    i_t = $(NT) - $(1) - j
    p_q = tl.make_block_ptr(base=q + i_bh * $(s_qk_h),
      shape=($(K), $(T)), strides=($(s_qk_d), $(s_qk_t)),
      offsets=(i_k * $(BK), i_t * $(BT)), block_shape=($(BK), $(BT)), order=(0, 1))
    p_do = tl.make_block_ptr(base=do_ + i_bh * $(s_vo_h),
      shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)),
      offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
    p_dh = tl.make_block_ptr(base=dh + i_bh * $(s_h_h) + i_t * $(K) * $(V),
      shape=($(K), $(V)), strides=($(s_h_t), $(1)),
      offsets=(i_k * $(BK), i_v * $(BV)), block_shape=($(BK), $(BV)), order=(1, 0))
    tl.store(p_dh, (b_dh).to(p_dh.dtype.element_ty), boundary_check=([0, 1] : List Nat))
    b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat))
    b_q = (b_q * $(scale)).to(b_q.dtype)
    b_do = tl.load(p_do, boundary_check=([0, 1] : List Nat))
    b_dh += tl.dot(b_q, (b_do).to(b_q.dtype), allow_tf32=false)
  }
}
```
</details>

<details><summary><code>claActive</code></summary>

```
/-- A state lane is *active* when it maps inside the `K × V` window. -/
```
```lean
def claActive (s : BlockState) (K V BK BV : Nat)
    (idx : TileIndex [BK, BV]) : Prop :=
  s.pids 0 * BK + idx.1.val < K ∧ s.pids 1 * BV + idx.2.1.val < V
```
</details>

<details><summary><code>claHOffset</code></summary>

```
/-- The `h` / `dh` chunk-`t` store address for lane `(e, p)`. -/
```
```lean
def claHOffset (s : BlockState) (s_h_h s_h_t K V BK BV : Nat) (t : Nat)
    (idx : TileIndex [BK, BV]) : Nat :=
  s.pids 2 * s_h_h + t * K * V + (s.pids 0 * BK + idx.1.val) * s_h_t
    + (s.pids 1 * BV + idx.2.1.val) * 1
```
</details>

<details><summary><code>claDhState</code></summary>

```
/-- **The backward state at chunk `t`** — what the kernel stores into `dh[·,·,t]`:
every chunk *after* `t` (the loop descends, and each chunk is stored before it is
accumulated). -/
```
```lean
noncomputable def claDhState (s : BlockState) (q do_ : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (T K V BT BK BV NT : Nat) (t e p : Nat) : ℝ :=
  ∑ u : Fin NT, if t < u.val then
    claBContrib s q do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
      T K V BT BK BV u.val e p
  else 0
```
</details>

<details><summary><code>claBContrib</code></summary>

```
/-- One chunk's contribution to the backward state lane `(e, p)`:
`Σ_c (scale · q)[e, c] · do[c, p]` — `q` shares `k`'s layout, `do` shares `v`'s. -/
```
```lean
noncomputable def claBContrib (s : BlockState) (q do_ : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (T K V BT BK BV : Nat) (t e p : Nat) : ℝ :=
  ∑ c : Fin BT,
    (kGuarded s q s_qk_h s_qk_t s_qk_d T K BT BK t c.val e * scale)
      * vGuarded s do_ s_vo_h s_vo_t s_vo_d T V BT BV t c.val p
```
</details>

<details><summary><code>kGuarded</code></summary>

```
/-- The guarded `k` lane, as `b_k` holds it. -/
```
```lean
noncomputable def kGuarded (s : BlockState) (k : RegionName)
    (s_qk_h s_qk_t s_qk_d : Nat) (T K BT BK : Nat) (t c e : Nat) : ℝ :=
  if s.pids 0 * BK + e < K ∧ t * BT + c < T then
    kElem s k s_qk_h s_qk_t s_qk_d BT BK t c e
  else 0
```
</details>

<details><summary><code>vGuarded</code></summary>

```
/-- The guarded `v` lane, as `b_v` holds it. -/
```
```lean
noncomputable def vGuarded (s : BlockState) (v : RegionName)
    (s_vo_h s_vo_t s_vo_d : Nat) (T V BT BV : Nat) (t c p : Nat) : ℝ :=
  if t * BT + c < T ∧ s.pids 1 * BV + p < V then
    vElem s v s_vo_h s_vo_t s_vo_d BT BV t c p
  else 0
```
</details>

<details><summary><code>kElem</code></summary>

```
/-- `k[i_k·BK + e, t·BT + c]` (parent `(K, T)`, strides `(s_qk_d, s_qk_t)`). -/
```
```lean
noncomputable def kElem (s : BlockState) (k : RegionName)
    (s_qk_h s_qk_t s_qk_d BT BK : Nat) (t c e : Nat) : ℝ :=
  s.readMem k (s.pids 2 * s_qk_h + (s.pids 0 * BK + e) * s_qk_d
    + (t * BT + c) * s_qk_t)
```
</details>

<details><summary><code>vElem</code></summary>

```
/-- `v[t·BT + c, i_v·BV + p]` (parent `(T, V)`, strides `(s_vo_t, s_vo_d)`). -/
```
```lean
noncomputable def vElem (s : BlockState) (v : RegionName)
    (s_vo_h s_vo_t s_vo_d BT BV : Nat) (t c p : Nat) : ℝ :=
  s.readMem v (s.pids 2 * s_vo_h + (t * BT + c) * s_vo_t
    + (s.pids 1 * BV + p) * s_vo_d)
```
</details>
