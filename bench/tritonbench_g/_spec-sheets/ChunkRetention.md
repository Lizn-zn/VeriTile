# Spec sheet — `bench/tritonbench_g/chunk_retention/ChunkRetention.lean`

**Python source:** `bench/tritonbench_g/chunk_retention/chunk_retention.py`

## Public theorem: `crh_fwd_h_exec_genuine`

<details><summary>docstring</summary>

```
/-- **Genuine, dimension-general correctness** of the file's first kernel,
`chunk_retention_fwd_kernel_h`. For every launch state the kernel runs to
completion; every active lane of every chunk block `h[·,·,t]` (`t < NT`) holds
the **pre-chunk** state `crhState t` of the decayed recurrence
`H_{t+1} = d_b(t)·H_t + k_tᵀ·(v_t ⊙ d_i(t))` — including the ragged last
chunk, whose in-loop `d_b`/`d_i` rebind gives the final step its own decay
length `T % BT` — and, when `STORE_FINAL_STATE` is set, `ht` holds
`crhState NT`. One theorem covers all four gate configurations; every
dimension, stride, the head count `H`, and the chunk count `NT` stay
symbolic. -/
```
</details>

**Statement:**
```lean
specification crh_fwd_h_exec_genuine
    (k v h h0 ht : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat)
    (H T K V BT BK BV NT : Nat) (USE_INITIAL_STATE STORE_FINAL_STATE : Bool)
    (s : BlockState)
    (hHk : h ≠ k) (hHv : h ≠ v) (hHtH : ht ≠ h)
    (hσ : BV ≤ s_h_t) (hFit : (K - 1) * s_h_t + V ≤ K * V) (hBVV : BV ≤ V) :
    ∃ sF, exec (crh_fwd_h_surface k v h h0 ht s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t
        s_vo_d s_h_h s_h_t H T K V BT BK BV NT
        USE_INITIAL_STATE STORE_FINAL_STATE).toAlgKernel s = some sF
      ∧ (∀ t (idx : TileIndex [BK, BV]), t < NT → crhActive s K V BK BV idx →
          sF.readMem h (crhHOffset s s_h_h s_h_t K V BK BV t idx)
            = crhState s k v h0 USE_INITIAL_STATE s_qk_h s_qk_t s_qk_d s_vo_h
                s_vo_t s_vo_d H T K V BT BK BV NT t idx.1.val idx.2.1.val)
      ∧ (STORE_FINAL_STATE = Bool.true →
          ∀ idx : TileIndex [BK, BV], crhActive s K V BK BV idx →
          sF.readMem ht (crhHtOffset s K V BK BV idx)
            = crhState s k v h0 USE_INITIAL_STATE s_qk_h s_qk_t s_qk_d s_vo_h
                s_vo_t s_vo_d H T K V BT BK BV NT NT idx.1.val idx.2.1.val)
```

**Assumptions / layout contracts:**
- `hHk : h ≠ k`
- `hHv : h ≠ v`
- `hHtH : ht ≠ h`
- `hσ : BV ≤ s_h_t`
- `hFit : (K - 1) * s_h_t + V ≤ K * V`
- `hBVV : BV ≤ V`
- `STORE_FINAL_STATE = Bool.true →
          ∀ idx : TileIndex [BK, BV], crhActive s K V BK BV idx →
          sF.readMem ht (crhHtOffset s K V BK BV idx)
            = crhState s k v h0 USE_INITIAL_STATE s_qk_h s_qk_t s_qk_d s_vo_h
                s_vo_t s_vo_d H T K V BT BK BV NT NT idx.1.val idx.2.1.val`

**Closed-form spec defs (transitive):** `crh_fwd_h_surface`, `crhActive`, `crhHOffset`, `crhState`, `crhHtOffset`, `crhH0Guarded`, `crhDb`, `crhKGuarded`, `crhVGuarded`, `crhDi`, `crhH0Elem`, `crhLen`, `crhBeta`, `crhKElem`, `crhVElem`

<details><summary><code>crh_fwd_h_surface</code></summary>

```
/-- Faithful transcription of `chunk_retention_fwd_kernel_h`. -/
```
```lean
def crh_fwd_h_surface
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

<details><summary><code>crhActive</code></summary>

```
/-- A forward state lane is *active* when it maps inside the `K × V` window. -/
```
```lean
def crhActive (s : BlockState) (K V BK BV : Nat)
    (idx : TileIndex [BK, BV]) : Prop :=
  s.pids 0 * BK + idx.1.val < K ∧ s.pids 1 * BV + idx.2.1.val < V
```
</details>

<details><summary><code>crhHOffset</code></summary>

```
/-- The `h` / `ht` chunk-store address at lane `(e, p)` (same layout as
`chunk_linear_attn`). -/
```
```lean
def crhHOffset (s : BlockState) (s_h_h s_h_t K V BK BV : Nat) (t : Nat)
    (idx : TileIndex [BK, BV]) : Nat :=
  s.pids 2 * s_h_h + t * K * V + (s.pids 0 * BK + idx.1.val) * s_h_t
    + (s.pids 1 * BV + idx.2.1.val) * 1
```
</details>

<details><summary><code>crhState</code></summary>

```
/-- **The forward state at chunk `t`** — what the kernel stores into `h[·,·,t]`
before chunk `t` runs: the decayed recurrence
`H_{t+1} = d_b(t)·H_t + Σ_c k[e,c]·(v[c,p]·d_i(t)[c])` seeded with the gated
`h0`. Recursive (not a power closed form): the ragged last chunk gives the
final step its own decay length. -/
```
```lean
noncomputable def crhState (s : BlockState) (k v h0 : RegionName) (UIS : Bool)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat)
    (H T K V BT BK BV NT : Nat) : Nat → Nat → Nat → ℝ
  | 0 => fun e p => if UIS then crhH0Guarded s h0 K V BK BV e p else 0
  | t + 1 => fun e p =>
      crhDb s H T BT NT t
          * crhState s k v h0 UIS s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
              H T K V BT BK BV NT t e p
        + ∑ c : Fin BT,
            crhKGuarded s k s_qk_h s_qk_t s_qk_d T K BT BK t c.val e
              * (crhVGuarded s v s_vo_h s_vo_t s_vo_d T V BT BV t c.val p
                  * crhDi s H T BT NT t c.val)
```
</details>

<details><summary><code>crhHtOffset</code></summary>

```
/-- The `ht` store address at lane `(e, p)`. -/
```
```lean
def crhHtOffset (s : BlockState) (K V BK BV : Nat)
    (idx : TileIndex [BK, BV]) : Nat :=
  s.pids 2 * K * V + (s.pids 0 * BK + idx.1.val) * V
    + (s.pids 1 * BV + idx.2.1.val) * 1
```
</details>

<details><summary><code>crhH0Guarded</code></summary>

```
/-- The guarded initial state, as the `USE_INITIAL_STATE` branch leaves `b_h`. -/
```
```lean
noncomputable def crhH0Guarded (s : BlockState) (h0 : RegionName)
    (K V BK BV : Nat) (e p : Nat) : ℝ :=
  if s.pids 0 * BK + e < K ∧ s.pids 1 * BV + p < V then
    crhH0Elem s h0 K V BK BV e p
  else 0
```
</details>

<details><summary><code>crhDb</code></summary>

```
/-- The inter-chunk decay factor `d_b(t) = 2^(len(t)·b_b)`. -/
```
```lean
noncomputable def crhDb (s : BlockState) (H T BT NT t : Nat) : ℝ :=
  Real.exp (((crhLen T BT NT t : Nat) : ℝ) * crhBeta s H * Real.log 2)
```
</details>

<details><summary><code>crhKGuarded</code></summary>

```
/-- The guarded `k` lane, as `b_k` holds it. -/
```
```lean
noncomputable def crhKGuarded (s : BlockState) (k : RegionName)
    (s_qk_h s_qk_t s_qk_d : Nat) (T K BT BK : Nat) (t c e : Nat) : ℝ :=
  if s.pids 0 * BK + e < K ∧ t * BT + c < T then
    crhKElem s k s_qk_h s_qk_t s_qk_d BT BK t c e
  else 0
```
</details>

<details><summary><code>crhVGuarded</code></summary>

```
/-- The guarded `v` (or `do`) lane, as `b_v` / `b_o` holds it. -/
```
```lean
noncomputable def crhVGuarded (s : BlockState) (v : RegionName)
    (s_vo_h s_vo_t s_vo_d : Nat) (T V BT BV : Nat) (t c p : Nat) : ℝ :=
  if t * BT + c < T ∧ s.pids 1 * BV + p < V then
    crhVElem s v s_vo_h s_vo_t s_vo_d BT BV t c p
  else 0
```
</details>

<details><summary><code>crhDi</code></summary>

```
/-- The intra-chunk decay weights `d_i(t)[c] = 2^((len(t) - c - 1)·b_b)` — with
the `.nat`-truncated tail lanes (see the preamble; unobservable through the
boundary-checked `v`). -/
```
```lean
noncomputable def crhDi (s : BlockState) (H T BT NT t c : Nat) : ℝ :=
  Real.exp (((crhLen T BT NT t - c - 1 : Nat) : ℝ) * crhBeta s H * Real.log 2)
```
</details>

<details><summary><code>crhH0Elem</code></summary>

```
/-- `h0[i_k·BK + e, i_v·BV + p]` (parent `(K, V)`, strides `(V, 1)`). -/
```
```lean
noncomputable def crhH0Elem (s : BlockState) (h0 : RegionName)
    (K V BK BV : Nat) (e p : Nat) : ℝ :=
  s.readMem h0 (s.pids 2 * K * V + (s.pids 0 * BK + e) * V
    + (s.pids 1 * BV + p) * 1)
```
</details>

<details><summary><code>crhLen</code></summary>

```
/-- Chunk `t`'s effective length: `T % BT` on a ragged last chunk, else `BT`. -/
```
```lean
def crhLen (T BT NT t : Nat) : Nat :=
  if t = NT - 1 ∧ T % BT ≠ 0 then T % BT else BT
```
</details>

<details><summary><code>crhBeta</code></summary>

```
/-- The per-head decay exponent `b_b = log2(1 - 2^(-5 - i_h))`, exactly as the
walk computes it. -/
```
```lean
noncomputable def crhBeta (s : BlockState) (H : Nat) : ℝ :=
  Real.log ((1.0 : ℝ) - Real.exp ((((0.0 : ℝ) - 5.0)
      - ((s.pids 2 % H : Nat) : ℝ) * 1.0) * Real.log 2)) / Real.log 2
```
</details>

<details><summary><code>crhKElem</code></summary>

```
/-- `k[i_k·BK + e, t·BT + c]` (parent `(K, T)`, strides `(s_qk_d, s_qk_t)`). -/
```
```lean
noncomputable def crhKElem (s : BlockState) (k : RegionName)
    (s_qk_h s_qk_t s_qk_d BT BK : Nat) (t c e : Nat) : ℝ :=
  s.readMem k (s.pids 2 * s_qk_h + (s.pids 0 * BK + e) * s_qk_d
    + (t * BT + c) * s_qk_t)
```
</details>

<details><summary><code>crhVElem</code></summary>

```
/-- `v[t·BT + c, i_v·BV + p]` (parent `(T, V)`, strides `(s_vo_t, s_vo_d)`). -/
```
```lean
noncomputable def crhVElem (s : BlockState) (v : RegionName)
    (s_vo_h s_vo_t s_vo_d BT BV : Nat) (t c p : Nat) : ℝ :=
  s.readMem v (s.pids 2 * s_vo_h + (t * BT + c) * s_vo_t
    + (s.pids 1 * BV + p) * s_vo_d)
```
</details>

## Public theorem: `crh_bwd_dh_exec_genuine`

<details><summary>docstring</summary>

```
/-- **Genuine correctness** of `chunk_retention_bwd_kernel_dh`, exactly as the
upstream kernel computes (see the preamble's faithfulness notes: the square
shape is forced by its `tl.dot`, the contraction is the scrambled one, the
final store reuses the post-loop `i_t = 0`, and the `dh` loads are dead).
Every active lane of the single `dh` store block reads back
`crhDhOut = d_b · Σ_{all chunks} (do ⊙ d_i)·v`. -/
```
</details>

**Statement:**
```lean
specification crh_bwd_dh_exec_genuine
    (v do_ dh : RegionName)
    (s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat)
    (H T K V BT NT : Nat) (s : BlockState)
    (hσ : BT ≤ s_h_t) :
    ∃ sF, exec (crh_bwd_dh_surface v do_ dh s_vo_h s_vo_t s_vo_d s_h_h s_h_t
        H T K V BT NT).toAlgKernel s = some sF
      ∧ ∀ idx : TileIndex [BT, BT], crhDhActive s K V BT idx →
          sF.readMem dh (crhDhOffset s s_h_h s_h_t K V BT idx)
            = crhDhOut s v do_ s_vo_h s_vo_t s_vo_d H T V BT NT
                idx.1.val idx.2.1.val
```

**Assumptions / layout contracts:**
- `hσ : BT ≤ s_h_t`

**Closed-form spec defs (transitive):** `crh_bwd_dh_surface`, `crhDhActive`, `crhDhOffset`, `crhDhOut`, `crhDhAcc`, `crhDbBwd`, `crhBContrib`, `crhBeta`, `crhVGuarded`, `crhDiBwd`, `crhVElem`

<details><summary><code>crh_bwd_dh_surface</code></summary>

```
/-- Faithful transcription of `chunk_retention_bwd_kernel_dh`, with the
descending loop spelled as its ascending change of variable, and the single
shared block size (forced by its `tl.dot` shapes — see the preamble) bound as
`BT` where the Python spells `BK`/`BV`. -/
```
```lean
def crh_bwd_dh_surface
    (v do_ dh : RegionName)
    (s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat)
    (H T K V BT NT : Nat) :
    ComputeKernel := triton {
  i_k = tl.program_id(0)
  i_v = tl.program_id(1)
  i_bh = tl.program_id(2)
  i_h = i_bh % $(H)
  b_b = tl.math.log2(1.0 - tl.math.exp2(0.0 - 5.0 - tl.toReal(i_h) * 1.0))
  o_i = tl.arange(0, $(BT))
  d_b = tl.math.exp2(tl.toReal($(BT)) * b_b)
  d_i = tl.math.exp2(tl.toReal(o_i + $(1)) * b_b)
  b_dh = tl.zeros([$(BT), $(BT)], dtype=tl.float32)
  i_t = $(0)
  for j in range($(0), $(NT), $(1)) {
    i_t = $(NT) - $(1) - j
    p_o = tl.make_block_ptr(base=do_ + i_bh * $(s_vo_h),
      shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)),
      offsets=(i_t * $(BT), i_v * $(BT)), block_shape=($(BT), $(BT)), order=(1, 0))
    p_v = tl.make_block_ptr(base=v + i_bh * $(s_vo_h),
      shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)),
      offsets=(i_t * $(BT), i_v * $(BT)), block_shape=($(BT), $(BT)), order=(1, 0))
    p_h = tl.make_block_ptr(base=dh + i_bh * $(s_h_h) + i_t * $(K) * $(V),
      shape=($(K), $(V)), strides=($(s_h_t), $(1)),
      offsets=(i_k * $(BT), i_v * $(BT)), block_shape=($(BT), $(BT)), order=(1, 0))
    b_o = tl.load(p_o, boundary_check=([0, 1] : List Nat))
    b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
    b_h = tl.load(p_h, boundary_check=([0, 1] : List Nat))
    b_dh += tl.dot((b_o * d_i[:, None]).to(b_o.dtype), b_v, allow_tf32=false)
  }
  b_dh *= d_b
  p_dh = tl.make_block_ptr(base=dh + i_bh * $(s_h_h) + i_k * $(K) * $(V),
    shape=($(K), $(V)), strides=($(s_h_t), $(1)),
    offsets=(i_v * $(BT), i_t * $(BT)), block_shape=($(BT), $(BT)), order=(1, 0))
  tl.store(p_dh, (b_dh).to(p_dh.dtype.element_ty), boundary_check=([0, 1] : List Nat))
}
```
</details>

<details><summary><code>crhDhActive</code></summary>

```
/-- A backward store lane is *active* under its own (scrambled) window. -/
```
```lean
def crhDhActive (s : BlockState) (K V BT : Nat)
    (idx : TileIndex [BT, BT]) : Prop :=
  s.pids 1 * BT + idx.1.val < K ∧ 0 * BT + idx.2.1.val < V
```
</details>

<details><summary><code>crhDhOffset</code></summary>

```
/-- The backward single-store address at lane `(x, y)`: base
`i_bh·s_h_h + i_k·K·V`, offsets `(i_v·BT, i_t·BT)` with the post-loop
`i_t = 0`. -/
```
```lean
def crhDhOffset (s : BlockState) (s_h_h s_h_t K V BT : Nat)
    (idx : TileIndex [BT, BT]) : Nat :=
  s.pids 2 * s_h_h + s.pids 0 * K * V + (s.pids 1 * BT + idx.1.val) * s_h_t
    + (0 * BT + idx.2.1.val) * 1
```
</details>

<details><summary><code>crhDhOut</code></summary>

```
/-- **The stored backward value**: `d_b · Σ_{all chunks} contrib`. -/
```
```lean
noncomputable def crhDhOut (s : BlockState) (v do_ : RegionName)
    (s_vo_h s_vo_t s_vo_d : Nat) (H T V BT NT : Nat) (x y : Nat) : ℝ :=
  crhDhAcc s v do_ s_vo_h s_vo_t s_vo_d H T V BT NT NT x y * crhDbBwd s H BT
```
</details>

<details><summary><code>crhDhAcc</code></summary>

```
/-- The backward accumulator after `c` descending iterations: every chunk from
`NT - c` up (the same indicator-sum carry as `chunk_linear_attn`). -/
```
```lean
noncomputable def crhDhAcc (s : BlockState) (v do_ : RegionName)
    (s_vo_h s_vo_t s_vo_d : Nat) (H T V BT NT : Nat) (c x y : Nat) : ℝ :=
  ∑ u : Fin NT, if NT - c ≤ u.val then
    crhBContrib s v do_ s_vo_h s_vo_t s_vo_d H T V BT u.val x y
  else 0
```
</details>

<details><summary><code>crhDbBwd</code></summary>

```
/-- The backward outer factor `d_b = 2^(BT·b_b)` (never rebound). -/
```
```lean
noncomputable def crhDbBwd (s : BlockState) (H BT : Nat) : ℝ :=
  Real.exp (((BT : Nat) : ℝ) * crhBeta s H * Real.log 2)
```
</details>

<details><summary><code>crhBContrib</code></summary>

```
/-- One backward chunk's contribution at lane `(x, y)`:
`Σ_r (do[x, r]·d_i[x]) · v[r, y]` — the scrambled contraction the upstream
kernel actually performs (both operands are `[BT, BT]` blocks of the `(T, V)`
parents; see the preamble). -/
```
```lean
noncomputable def crhBContrib (s : BlockState) (v do_ : RegionName)
    (s_vo_h s_vo_t s_vo_d : Nat) (H T V BT : Nat) (t x y : Nat) : ℝ :=
  ∑ r : Fin BT,
    (crhVGuarded s do_ s_vo_h s_vo_t s_vo_d T V BT BT t x r.val
        * crhDiBwd s H x)
      * crhVGuarded s v s_vo_h s_vo_t s_vo_d T V BT BT t r.val y
```
</details>

<details><summary><code>crhBeta</code></summary>

```
/-- The per-head decay exponent `b_b = log2(1 - 2^(-5 - i_h))`, exactly as the
walk computes it. -/
```
```lean
noncomputable def crhBeta (s : BlockState) (H : Nat) : ℝ :=
  Real.log ((1.0 : ℝ) - Real.exp ((((0.0 : ℝ) - 5.0)
      - ((s.pids 2 % H : Nat) : ℝ) * 1.0) * Real.log 2)) / Real.log 2
```
</details>

<details><summary><code>crhVGuarded</code></summary>

```
/-- The guarded `v` (or `do`) lane, as `b_v` / `b_o` holds it. -/
```
```lean
noncomputable def crhVGuarded (s : BlockState) (v : RegionName)
    (s_vo_h s_vo_t s_vo_d : Nat) (T V BT BV : Nat) (t c p : Nat) : ℝ :=
  if t * BT + c < T ∧ s.pids 1 * BV + p < V then
    crhVElem s v s_vo_h s_vo_t s_vo_d BT BV t c p
  else 0
```
</details>

<details><summary><code>crhDiBwd</code></summary>

```
/-- The backward weights `d_i[c] = 2^((c + 1)·b_b)` (never rebound). -/
```
```lean
noncomputable def crhDiBwd (s : BlockState) (H c : Nat) : ℝ :=
  Real.exp (((c + 1 : Nat) : ℝ) * crhBeta s H * Real.log 2)
```
</details>

<details><summary><code>crhVElem</code></summary>

```
/-- `v[t·BT + c, i_v·BV + p]` (parent `(T, V)`, strides `(s_vo_t, s_vo_d)`). -/
```
```lean
noncomputable def crhVElem (s : BlockState) (v : RegionName)
    (s_vo_h s_vo_t s_vo_d BT BV : Nat) (t c p : Nat) : ℝ :=
  s.readMem v (s.pids 2 * s_vo_h + (t * BT + c) * s_vo_t
    + (s.pids 1 * BV + p) * s_vo_d)
```
</details>
