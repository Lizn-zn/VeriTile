# Spec sheet — `bench/tritonbench_g/chunk_retention_ops/ChunkRetentionOps.lean`

**Python source:** `bench/tritonbench_g/chunk_retention_ops/chunk_retention_ops.py`

## Public theorem: `cro_fwd_h_exec_genuine`

<details><summary>docstring</summary>

```
/-- **Genuine, dimension-general correctness** of the file's first kernel,
`chunk_retention_fwd_kernel_h`. For every launch state the kernel runs to
completion; every active lane of every chunk block `h[·,·,t]` (`t < NT`) holds
the **pre-chunk** state `croState t` of the decayed recurrence
`H_{t+1} = d_b(t)·H_t + k_tᵀ·(v_t ⊙ d_i(t))` — including the ragged last
chunk, whose in-loop `d_b`/`d_i` rebind gives the final step its own decay
length `T % BT` — and, when `STORE_FINAL_STATE` is set, `ht` holds
`croState NT`. One theorem covers all four gate configurations; every
dimension, stride, the head count `H`, and the chunk count `NT` stay
symbolic. -/
```
</details>

**Statement:**
```lean
specification cro_fwd_h_exec_genuine
    (k v h h0 ht : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat)
    (H T K V BT BK BV NT : Nat) (USE_INITIAL_STATE STORE_FINAL_STATE : Bool)
    (s : BlockState)
    (hHk : h ≠ k) (hHv : h ≠ v) (hHtH : ht ≠ h)
    (hσ : BV ≤ s_h_t) (hFit : (K - 1) * s_h_t + V ≤ K * V) (hBVV : BV ≤ V) :
    ∃ sF, exec (cro_fwd_h_surface k v h h0 ht s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t
        s_vo_d s_h_h s_h_t H T K V BT BK BV NT
        USE_INITIAL_STATE STORE_FINAL_STATE).toAlgKernel s = some sF
      ∧ (∀ t (idx : TileIndex [BK, BV]), t < NT → croActive s K V BK BV idx →
          sF.readMem h (croHOffset s s_h_h s_h_t K V BK BV t idx)
            = croState s k v h0 USE_INITIAL_STATE s_qk_h s_qk_t s_qk_d s_vo_h
                s_vo_t s_vo_d H T K V BT BK BV NT t idx.1.val idx.2.1.val)
      ∧ (STORE_FINAL_STATE = Bool.true →
          ∀ idx : TileIndex [BK, BV], croActive s K V BK BV idx →
          sF.readMem ht (croHtOffset s K V BK BV idx)
            = croState s k v h0 USE_INITIAL_STATE s_qk_h s_qk_t s_qk_d s_vo_h
                s_vo_t s_vo_d H T K V BT BK BV NT NT idx.1.val idx.2.1.val)
```

**Assumptions / layout contracts:**
- `hHk : h ≠ k`
- `hHv : h ≠ v`
- `hHtH : ht ≠ h`
- `hσ : BV ≤ s_h_t`
- `hFit : (K - 1) * s_h_t + V ≤ K * V`
- `hBVV : BV ≤ V`

**Closed-form spec defs (transitive):** `cro_fwd_h_surface`, `croActive`, `croHOffset`, `croState`, `croHtOffset`, `croH0Guarded`, `croDb`, `croKGuarded`, `croVGuarded`, `croDi`, `croH0Elem`, `croLen`, `croBeta`, `croKElem`, `croVElem`

<details><summary><code>cro_fwd_h_surface</code></summary>

```
/-- Faithful transcription of `chunk_retention_fwd_kernel_h`. -/
```
```lean
def cro_fwd_h_surface
    (k v h h0 ht : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat)
    (H T K V BT BK BV NT : Nat)
    (USE_INITIAL_STATE STORE_FINAL_STATE : Bool) :
    ComputeKernel := triton {
  i_k = tl.program_id(0)
  i_v = tl.program_id(1)
  i_bh = tl.program_id(2)
  i_h = i_bh % $(H)
  b_b = tl.math.log2(1.0 - tl.math.exp2(0.0 - 5.0 - tl.toReal(i_h) * 1.0))
  o_i = tl.arange(0, $(BT))
  d_b = tl.math.exp2(tl.toReal($(BT)) * b_b)
  d_i = tl.math.exp2(tl.toReal($(BT) - o_i - $(1)) * b_b)
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
    if i_t == $(NT) - $(1) and $(T) % $(BT) != $(0) {
      d_b = tl.math.exp2(tl.toReal($(T) % $(BT)) * b_b)
      d_i = tl.math.exp2(tl.toReal($(T) % $(BT) - o_i - $(1)) * b_b)
    }
    b_h = d_b * b_h + tl.dot(b_k, (b_v * d_i[:, None]).to(b_k.dtype), allow_tf32=false)
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

<details><summary><code>croActive</code></summary>

```
/-- A state lane is *active* when it maps inside the `K × V` window. -/
```
```lean
def croActive (s : BlockState) (K V BK BV : Nat)
    (idx : TileIndex [BK, BV]) : Prop :=
  s.pids 0 * BK + idx.1.val < K ∧ s.pids 1 * BV + idx.2.1.val < V
```
</details>

<details><summary><code>croHOffset</code></summary>

```
/-- The `h` / `ht` / `dh` chunk-store address at lane `(e, p)`. -/
```
```lean
def croHOffset (s : BlockState) (s_h_h s_h_t K V BK BV : Nat) (t : Nat)
    (idx : TileIndex [BK, BV]) : Nat :=
  s.pids 2 * s_h_h + t * K * V + (s.pids 0 * BK + idx.1.val) * s_h_t
    + (s.pids 1 * BV + idx.2.1.val) * 1
```
</details>

<details><summary><code>croState</code></summary>

```
/-- **The forward state at chunk `t`** — what the kernel stores into `h[·,·,t]`
before chunk `t` runs: the decayed recurrence
`H_{t+1} = d_b(t)·H_t + Σ_c k[e,c]·(v[c,p]·d_i(t)[c])` seeded with the gated
`h0`. Recursive (not a power closed form): the ragged last chunk gives the
final step its own decay length. -/
```
```lean
noncomputable def croState (s : BlockState) (k v h0 : RegionName) (UIS : Bool)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat)
    (H T K V BT BK BV NT : Nat) : Nat → Nat → Nat → ℝ
  | 0 => fun e p => if UIS then croH0Guarded s h0 K V BK BV e p else 0
  | t + 1 => fun e p =>
      croDb s H T BT NT t
          * croState s k v h0 UIS s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
              H T K V BT BK BV NT t e p
        + ∑ c : Fin BT,
            croKGuarded s k s_qk_h s_qk_t s_qk_d T K BT BK t c.val e
              * (croVGuarded s v s_vo_h s_vo_t s_vo_d T V BT BV t c.val p
                  * croDi s H T BT NT t c.val)
```
</details>

<details><summary><code>croHtOffset</code></summary>

```
/-- The `ht` store address at lane `(e, p)`. -/
```
```lean
def croHtOffset (s : BlockState) (K V BK BV : Nat)
    (idx : TileIndex [BK, BV]) : Nat :=
  s.pids 2 * K * V + (s.pids 0 * BK + idx.1.val) * V
    + (s.pids 1 * BV + idx.2.1.val) * 1
```
</details>

<details><summary><code>croH0Guarded</code></summary>

```
/-- The guarded initial state, as the `USE_INITIAL_STATE` branch leaves `b_h`. -/
```
```lean
noncomputable def croH0Guarded (s : BlockState) (h0 : RegionName)
    (K V BK BV : Nat) (e p : Nat) : ℝ :=
  if s.pids 0 * BK + e < K ∧ s.pids 1 * BV + p < V then
    croH0Elem s h0 K V BK BV e p
  else 0
```
</details>

<details><summary><code>croDb</code></summary>

```
/-- The inter-chunk decay factor `d_b(t) = 2^(len(t)·b_b)`. -/
```
```lean
noncomputable def croDb (s : BlockState) (H T BT NT t : Nat) : ℝ :=
  Real.exp (((croLen T BT NT t : Nat) : ℝ) * croBeta s H * Real.log 2)
```
</details>

<details><summary><code>croKGuarded</code></summary>

```
/-- The guarded `k` lane, as `b_k` holds it. -/
```
```lean
noncomputable def croKGuarded (s : BlockState) (k : RegionName)
    (s_qk_h s_qk_t s_qk_d : Nat) (T K BT BK : Nat) (t c e : Nat) : ℝ :=
  if s.pids 0 * BK + e < K ∧ t * BT + c < T then
    croKElem s k s_qk_h s_qk_t s_qk_d BT BK t c e
  else 0
```
</details>

<details><summary><code>croVGuarded</code></summary>

```
/-- The guarded `v` (or `do`) lane, as `b_v` / `b_o` holds it. -/
```
```lean
noncomputable def croVGuarded (s : BlockState) (v : RegionName)
    (s_vo_h s_vo_t s_vo_d : Nat) (T V BT BV : Nat) (t c p : Nat) : ℝ :=
  if t * BT + c < T ∧ s.pids 1 * BV + p < V then
    croVElem s v s_vo_h s_vo_t s_vo_d BT BV t c p
  else 0
```
</details>

<details><summary><code>croDi</code></summary>

```
/-- The intra-chunk decay weights `d_i(t)[c] = 2^((len(t) - c - 1)·b_b)` — with
the `.nat`-truncated tail lanes (see the preamble; unobservable through the
boundary-checked `v`). -/
```
```lean
noncomputable def croDi (s : BlockState) (H T BT NT t c : Nat) : ℝ :=
  Real.exp (((croLen T BT NT t - c - 1 : Nat) : ℝ) * croBeta s H * Real.log 2)
```
</details>

<details><summary><code>croH0Elem</code></summary>

```
/-- `h0[i_k·BK + e, i_v·BV + p]` (parent `(K, V)`, strides `(V, 1)`). -/
```
```lean
noncomputable def croH0Elem (s : BlockState) (h0 : RegionName)
    (K V BK BV : Nat) (e p : Nat) : ℝ :=
  s.readMem h0 (s.pids 2 * K * V + (s.pids 0 * BK + e) * V
    + (s.pids 1 * BV + p) * 1)
```
</details>

<details><summary><code>croLen</code></summary>

```
/-- Chunk `t`'s effective length: `T % BT` on a ragged last chunk, else `BT`. -/
```
```lean
def croLen (T BT NT t : Nat) : Nat :=
  if t = NT - 1 ∧ T % BT ≠ 0 then T % BT else BT
```
</details>

<details><summary><code>croBeta</code></summary>

```
/-- The per-head decay exponent `b_b = log2(1 - 2^(-5 - i_h))`, exactly as the
walk computes it. -/
```
```lean
noncomputable def croBeta (s : BlockState) (H : Nat) : ℝ :=
  Real.log ((1.0 : ℝ) - Real.exp ((((0.0 : ℝ) - 5.0)
      - ((s.pids 2 % H : Nat) : ℝ) * 1.0) * Real.log 2)) / Real.log 2
```
</details>

<details><summary><code>croKElem</code></summary>

```
/-- `k[i_k·BK + e, t·BT + c]` (parent `(K, T)`, strides `(s_qk_d, s_qk_t)`). -/
```
```lean
noncomputable def croKElem (s : BlockState) (k : RegionName)
    (s_qk_h s_qk_t s_qk_d BT BK : Nat) (t c e : Nat) : ℝ :=
  s.readMem k (s.pids 2 * s_qk_h + (s.pids 0 * BK + e) * s_qk_d
    + (t * BT + c) * s_qk_t)
```
</details>

<details><summary><code>croVElem</code></summary>

```
/-- `v[t·BT + c, i_v·BV + p]` (parent `(T, V)`, strides `(s_vo_t, s_vo_d)`). -/
```
```lean
noncomputable def croVElem (s : BlockState) (v : RegionName)
    (s_vo_h s_vo_t s_vo_d BT BV : Nat) (t c p : Nat) : ℝ :=
  s.readMem v (s.pids 2 * s_vo_h + (t * BT + c) * s_vo_t
    + (s.pids 1 * BV + p) * s_vo_d)
```
</details>

## Public theorem: `cro_bwd_dh_exec_genuine`

<details><summary>docstring</summary>

```
/-- **Genuine, dimension-general correctness** of
`chunk_retention_ops.py`'s `chunk_retention_bwd_kernel_dh` (the descending
loop, verified through its ascending change of variable). For every launch
state the kernel runs to completion and every active lane of every chunk
block `dh[·,·,t]` (`t < NT`) holds the **pre-chunk** descending carry
`croDhCarry (NT-1-t)` of the decayed recurrence
`D_next = d_b·D + (scale·q)ᵀ·(do ⊙ d_i)` — unlike the `chunk_retention`
sibling's backward, this one is fully dimension-general. -/
```
</details>

**Statement:**
```lean
specification cro_bwd_dh_exec_genuine
    (q do_ dh : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat)
    (scale : ℝ) (H T K V BT BK BV NT : Nat) (s : BlockState)
    (hDq : dh ≠ q) (hDdo : dh ≠ do_)
    (hσ : BV ≤ s_h_t) (hFit : (K - 1) * s_h_t + V ≤ K * V) :
    ∃ sF, exec (cro_bwd_dh_surface q do_ dh s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t
        s_vo_d s_h_h s_h_t scale H T K V BT BK BV NT).toAlgKernel s = some sF
      ∧ ∀ t (idx : TileIndex [BK, BV]), t < NT → croActive s K V BK BV idx →
          sF.readMem dh (croHOffset s s_h_h s_h_t K V BK BV t idx)
            = croDhCarry s q do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
                scale H T K V BT BK BV NT (NT - 1 - t) idx.1.val idx.2.1.val
```

**Assumptions / layout contracts:**
- `hDq : dh ≠ q`
- `hDdo : dh ≠ do_`
- `hσ : BV ≤ s_h_t`
- `hFit : (K - 1) * s_h_t + V ≤ K * V`

**Closed-form spec defs (transitive):** `cro_bwd_dh_surface`, `croActive`, `croHOffset`, `croDhCarry`, `croDbBwd`, `croBContrib`, `croBeta`, `croKGuarded`, `croVGuarded`, `croDiBwd`, `croKElem`, `croVElem`

<details><summary><code>cro_bwd_dh_surface</code></summary>

```
/-- Faithful transcription of `chunk_retention_bwd_kernel_dh`, with the
descending loop spelled as its ascending change of variable (see the
preamble). -/
```
```lean
def cro_bwd_dh_surface
    (q do_ dh : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat)
    (scale : ℝ) (H T K V BT BK BV NT : Nat) :
    ComputeKernel := triton {
  i_k = tl.program_id(0)
  i_v = tl.program_id(1)
  i_bh = tl.program_id(2)
  i_h = i_bh % $(H)
  b_b = tl.math.log2(1.0 - tl.math.exp2(0.0 - 5.0 - tl.toReal(i_h) * 1.0))
  o_i = tl.arange(0, $(BT))
  d_b = tl.math.exp2(tl.toReal($(BT)) * b_b)
  d_i = tl.math.exp2(tl.toReal(o_i + $(1)) * b_b)
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
    b_dh = d_b * b_dh + tl.dot(b_q, (b_do * d_i[:, None]).to(b_q.dtype), allow_tf32=false)
  }
}
```
</details>

<details><summary><code>croActive</code></summary>

```
/-- A state lane is *active* when it maps inside the `K × V` window. -/
```
```lean
def croActive (s : BlockState) (K V BK BV : Nat)
    (idx : TileIndex [BK, BV]) : Prop :=
  s.pids 0 * BK + idx.1.val < K ∧ s.pids 1 * BV + idx.2.1.val < V
```
</details>

<details><summary><code>croHOffset</code></summary>

```
/-- The `h` / `ht` / `dh` chunk-store address at lane `(e, p)`. -/
```
```lean
def croHOffset (s : BlockState) (s_h_h s_h_t K V BK BV : Nat) (t : Nat)
    (idx : TileIndex [BK, BV]) : Nat :=
  s.pids 2 * s_h_h + t * K * V + (s.pids 0 * BK + idx.1.val) * s_h_t
    + (s.pids 1 * BV + idx.2.1.val) * 1
```
</details>

<details><summary><code>croDhCarry</code></summary>

```
/-- **The backward carry after `c` descending iterations** — recursive, like
`croState`: `D_0 = 0`, `D_{c+1} = d_b·D_c + contrib(NT-1-c)`. What the kernel
stores into chunk `NT-1-c`'s block is `D_c` (the pre-chunk value). -/
```
```lean
noncomputable def croDhCarry (s : BlockState) (q do_ : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (H T K V BT BK BV NT : Nat) : Nat → Nat → Nat → ℝ
  | 0 => fun _ _ => 0
  | c + 1 => fun e p =>
      croDbBwd s H BT
          * croDhCarry s q do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
              H T K V BT BK BV NT c e p
        + croBContrib s q do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
            H T K V BT BK BV (NT - 1 - c) e p
```
</details>

<details><summary><code>croDbBwd</code></summary>

```
/-- The backward outer factor `d_b = 2^(BT·b_b)` (never rebound). -/
```
```lean
noncomputable def croDbBwd (s : BlockState) (H BT : Nat) : ℝ :=
  Real.exp (((BT : Nat) : ℝ) * croBeta s H * Real.log 2)
```
</details>

<details><summary><code>croBContrib</code></summary>

```
/-- One backward chunk's contribution at lane `(e, p)`:
`Σ_c (scale·q[e,c]) · (do[c,p]·d_i[c])`. -/
```
```lean
noncomputable def croBContrib (s : BlockState) (q do_ : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (H T K V BT BK BV : Nat) (t e p : Nat) : ℝ :=
  ∑ c : Fin BT,
    (croKGuarded s q s_qk_h s_qk_t s_qk_d T K BT BK t c.val e * scale)
      * (croVGuarded s do_ s_vo_h s_vo_t s_vo_d T V BT BV t c.val p
          * croDiBwd s H c.val)
```
</details>

<details><summary><code>croBeta</code></summary>

```
/-- The per-head decay exponent `b_b = log2(1 - 2^(-5 - i_h))`, exactly as the
walk computes it. -/
```
```lean
noncomputable def croBeta (s : BlockState) (H : Nat) : ℝ :=
  Real.log ((1.0 : ℝ) - Real.exp ((((0.0 : ℝ) - 5.0)
      - ((s.pids 2 % H : Nat) : ℝ) * 1.0) * Real.log 2)) / Real.log 2
```
</details>

<details><summary><code>croKGuarded</code></summary>

```
/-- The guarded `k` lane, as `b_k` holds it. -/
```
```lean
noncomputable def croKGuarded (s : BlockState) (k : RegionName)
    (s_qk_h s_qk_t s_qk_d : Nat) (T K BT BK : Nat) (t c e : Nat) : ℝ :=
  if s.pids 0 * BK + e < K ∧ t * BT + c < T then
    croKElem s k s_qk_h s_qk_t s_qk_d BT BK t c e
  else 0
```
</details>

<details><summary><code>croVGuarded</code></summary>

```
/-- The guarded `v` (or `do`) lane, as `b_v` / `b_o` holds it. -/
```
```lean
noncomputable def croVGuarded (s : BlockState) (v : RegionName)
    (s_vo_h s_vo_t s_vo_d : Nat) (T V BT BV : Nat) (t c p : Nat) : ℝ :=
  if t * BT + c < T ∧ s.pids 1 * BV + p < V then
    croVElem s v s_vo_h s_vo_t s_vo_d BT BV t c p
  else 0
```
</details>

<details><summary><code>croDiBwd</code></summary>

```
/-- The backward weights `d_i[c] = 2^((c + 1)·b_b)` (never rebound). -/
```
```lean
noncomputable def croDiBwd (s : BlockState) (H c : Nat) : ℝ :=
  Real.exp (((c + 1 : Nat) : ℝ) * croBeta s H * Real.log 2)
```
</details>

<details><summary><code>croKElem</code></summary>

```
/-- `k[i_k·BK + e, t·BT + c]` (parent `(K, T)`, strides `(s_qk_d, s_qk_t)`). -/
```
```lean
noncomputable def croKElem (s : BlockState) (k : RegionName)
    (s_qk_h s_qk_t s_qk_d BT BK : Nat) (t c e : Nat) : ℝ :=
  s.readMem k (s.pids 2 * s_qk_h + (s.pids 0 * BK + e) * s_qk_d
    + (t * BT + c) * s_qk_t)
```
</details>

<details><summary><code>croVElem</code></summary>

```
/-- `v[t·BT + c, i_v·BV + p]` (parent `(T, V)`, strides `(s_vo_t, s_vo_d)`). -/
```
```lean
noncomputable def croVElem (s : BlockState) (v : RegionName)
    (s_vo_h s_vo_t s_vo_d BT BV : Nat) (t c p : Nat) : ℝ :=
  s.readMem v (s.pids 2 * s_vo_h + (t * BT + c) * s_vo_t
    + (s.pids 1 * BV + p) * s_vo_d)
```
</details>
