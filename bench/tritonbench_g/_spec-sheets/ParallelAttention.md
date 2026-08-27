# Spec sheet — `bench/tritonbench_g/parallel_attention/ParallelAttention.lean`

**Python source:** `bench/tritonbench_g/parallel_attention/parallel_attention.py`

## Public theorem: `pa_fwd_o_exec_genuine`

<details><summary>docstring</summary>

```
/-- **★ Forward main theorem: the `o` and `z` stores are the genuine causal
rebased-attention closed forms.**

For every program `(i_kv, i_c, i_bh)` (universally quantified through
`s.pids`), executing the full forward surface succeeds, the `o` block store
holds `Σ_{t ≤ i_c·BTL + a} score(a,t)² · v[t,p]` at every in-window lane
`(a, p)`, and the `z` masked store holds the matching normalizer
`Σ_{t ≤ i_c·BTL + a} score(a,t)²` at every in-range row.

Side conditions: `o ≠ z` (the two output regions are distinct buffers),
the host's contiguous last-dim stride `s_vo_d = 1` with `BV ≤ s_vo_t`
(store-lane injectivity), and the host's own `assert BTL % BTS == 0`. The
loads are boundary-checked, so **no** divisibility hypothesis on `T` is
needed: ragged tails are exact. -/
```
</details>

**Statement:**
```lean
specification pa_fwd_o_exec_genuine
    (s : BlockState) (q k v o z : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (B H T K V BTL BTS BK BV : Nat)
    (hOZ : o ≠ z) (hSd : s_vo_d = 1) (hσ : BV ≤ s_vo_t)
    (hBTS : BTL % BTS = 0) (hBTSpos : 0 < BTS) :
    ∃ sF, exec (pa_fwd_surface q k v o z s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t
        s_vo_d scale B H T K V BTL BTS BK BV).toAlgKernel s = some sF
      ∧ (∀ idx : TileIndex [BTL, BV], paOActive s T V BV BTL idx →
          sF.readMem o (paOOffset s B H s_vo_h s_vo_t s_vo_d V BV BTL idx)
            = paOOut s q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
                T K V BTL BK BV idx.1.val idx.2.1.val)
      ∧ (∀ a : Fin BTL, s.pids 1 * BTL + a.val < T →
          sF.readMem z (paZOffset s B H T V BV BTL a)
            = paZOut s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV a.val)
```

**Assumptions / layout contracts:**
- `hOZ : o ≠ z`
- `hSd : s_vo_d = 1`
- `hσ : BV ≤ s_vo_t`
- `hBTS : BTL % BTS = 0`
- `hBTSpos : 0 < BTS`
- `∀ idx : TileIndex [BTL, BV], paOActive s T V BV BTL idx →
          sF.readMem o (paOOffset s B H s_vo_h s_vo_t s_vo_d V BV BTL idx)
            = paOOut s q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
                T K V BTL BK BV idx.1.val idx.2.1.val`
- `∀ a : Fin BTL, s.pids 1 * BTL + a.val < T →
          sF.readMem z (paZOffset s B H T V BV BTL a)
            = paZOut s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV a.val`

**Closed-form spec defs (transitive):** `pa_fwd_surface`, `paOActive`, `paOOffset`, `paOOut`, `paZOffset`, `paZOut`, `paIv`, `paOBase`, `paOAcc`, `paODiag`, `paIk`, `paZAcc`, `paZDiag`, `paNV`, `paScore`, `paVGuarded`, `paQGuarded`, `paKGuarded`

<details><summary><code>pa_fwd_surface</code></summary>

```
/-- Faithful transcription of `parallel_rebased_fwd_kernel`. -/
```
```lean
def pa_fwd_surface
    (q k v o z : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (B H T K V BTL BTS BK BV : Nat) :
    ComputeKernel := triton {
  i_kv = tl.program_id(0)
  i_c = tl.program_id(1)
  i_bh = tl.program_id(2)
  NV = tl.cdiv($(V), $(BV))
  i_k = i_kv // NV
  i_v = i_kv % NV
  p_q = tl.make_block_ptr(base=q + i_bh * $(s_qk_h),
    shape=($(T), $(K)), strides=($(s_qk_t), $(s_qk_d)),
    offsets=(i_c * $(BTL), i_k * $(BK)), block_shape=($(BTL), $(BK)), order=(1, 0))
  p_k = tl.make_block_ptr(base=k + i_bh * $(s_qk_h),
    shape=($(K), $(T)), strides=($(s_qk_d), $(s_qk_t)),
    offsets=(i_k * $(BK), 0), block_shape=($(BK), $(BTS)), order=(0, 1))
  p_v = tl.make_block_ptr(base=v + i_bh * $(s_vo_h),
    shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)),
    offsets=(0, i_v * $(BV)), block_shape=($(BTS), $(BV)), order=(1, 0))
  b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat))
  b_q = (b_q * $((scale : ℝ))).to(b_q.dtype)
  b_o = tl.zeros([$(BTL), $(BV)], dtype=tl.float32)
  b_z = tl.zeros([$(BTL)], dtype=tl.float32)
  for _i in range($(0), i_c * $(BTL), $(BTS)) {
    b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
    b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
    b_s = tl.dot(b_q, b_k, allow_tf32=false)
    b_s = b_s * b_s
    b_z += tl.sum(b_s, axis=1)
    b_o = b_o + tl.dot((b_s).to(b_v.dtype), b_v, allow_tf32=false)
    p_k = tl.advance(p_k, [$(0), $(BTS)])
    p_v = tl.advance(p_v, [$(BTS), $(0)])
  }
  tl.debug_barrier()
  o_q = tl.arange(0, $(BTL))
  o_k = tl.arange(0, $(BTS))
  p_k = tl.make_block_ptr(base=k + i_bh * $(s_qk_h),
    shape=($(K), $(T)), strides=($(s_qk_d), $(s_qk_t)),
    offsets=(i_k * $(BK), i_c * $(BTL)), block_shape=($(BK), $(BTS)), order=(0, 1))
  p_v = tl.make_block_ptr(base=v + i_bh * $(s_vo_h),
    shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)),
    offsets=(i_c * $(BTL), i_v * $(BV)), block_shape=($(BTS), $(BV)), order=(1, 0))
  for _i in range(i_c * $(BTL), (i_c + $(1)) * $(BTL), $(BTS)) {
    b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
    b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
    m_s = o_q[:, None] >= o_k[None, :]
    b_s = tl.dot(b_q, b_k, allow_tf32=false)
    b_s = b_s * b_s
    b_s = tl.where(m_s, b_s, 0.0)
    b_z += tl.sum(b_s, axis=1)
    b_o += tl.dot((b_s).to(b_q.dtype), b_v, allow_tf32=false)
    p_k = tl.advance(p_k, [$(0), $(BTS)])
    p_v = tl.advance(p_v, [$(BTS), $(0)])
    o_k += $(BTS)
  }
  p_o = tl.make_block_ptr(base=o + (i_bh + $(B) * $(H) * i_k) * $(s_vo_h),
    shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)),
    offsets=(i_c * $(BTL), i_v * $(BV)), block_shape=($(BTL), $(BV)), order=(1, 0))
  p_z = z + (i_bh + $(B) * $(H) * i_k) * $(T) + i_c * $(BTL) + tl.arange(0, $(BTL))
  tl.store(p_o, (b_o).to(p_o.dtype.element_ty), boundary_check=([0, 1] : List Nat))
  tl.store(p_z, (b_z).to(p_z.dtype.element_ty),
    mask=((i_c * $(BTL) + tl.arange(0, $(BTL))) < $(T)))
}
```
</details>

<details><summary><code>paOActive</code></summary>

```
/-- An `o` store lane is *active* when it maps inside the `T × V` window. -/
```
```lean
def paOActive (s : BlockState) (T V BV BTL : Nat)
    (idx : TileIndex [BTL, BV]) : Prop :=
  s.pids 1 * BTL + idx.1.val < T ∧ paIv s V BV * BV + idx.2.1.val < V
```
</details>

<details><summary><code>paOOffset</code></summary>

```
/-- The `o` store address at lane `(a, p)` (strides `(s_vo_t, 1)` after the
`s_vo_d = 1` substitution). -/
```
```lean
def paOOffset (s : BlockState) (B H s_vo_h s_vo_t s_vo_d V BV BTL : Nat)
    (idx : TileIndex [BTL, BV]) : Nat :=
  paOBase s B H s_vo_h V BV + (s.pids 1 * BTL + idx.1.val) * s_vo_t
    + (paIv s V BV * BV + idx.2.1.val) * s_vo_d
```
</details>

<details><summary><code>paOOut</code></summary>

```
/-- **The stored `o` lane** — non-diagonal sum plus the masked diagonal
block: causal rebased attention `Σ_{t ≤ i_c·BTL + a} score² · v`. -/
```
```lean
noncomputable def paOOut (s : BlockState) (q k v : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (T K V BTL BK BV : Nat) (a p : Nat) : ℝ :=
  paOAcc s q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
      T K V BTL BK BV (s.pids 1 * BTL) a p
    + paODiag s q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
        T K V BTL BK BV ((s.pids 1 + 1) * BTL) a p
```
</details>

<details><summary><code>paZOffset</code></summary>

```
/-- The `z` store address at lane `a`. -/
```
```lean
def paZOffset (s : BlockState) (B H T V BV BTL : Nat) (a : Fin BTL) : Nat :=
  (s.pids 2 + B * H * paIk s V BV) * T + s.pids 1 * BTL + a.val
```
</details>

<details><summary><code>paZOut</code></summary>

```
/-- **The stored `z` lane** — the matching normalizer. -/
```
```lean
noncomputable def paZOut (s : BlockState) (q k : RegionName)
    (s_qk_h s_qk_t s_qk_d : Nat) (scale : ℝ) (T K V BTL BK BV : Nat)
    (a : Nat) : ℝ :=
  paZAcc s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV
      (s.pids 1 * BTL) a
    + paZDiag s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV
        ((s.pids 1 + 1) * BTL) a
```
</details>

<details><summary><code>paIv</code></summary>

```
/-- `i_v = i_kv % NV`. -/
```
```lean
def paIv (s : BlockState) (V BV : Nat) : Nat := s.pids 0 % paNV V BV
```
</details>

<details><summary><code>paOBase</code></summary>

```
/-- The `o` store base `(i_bh + B·H·i_k) · s_vo_h`. -/
```
```lean
def paOBase (s : BlockState) (B H s_vo_h V BV : Nat) : Nat :=
  (s.pids 2 + B * H * paIk s V BV) * s_vo_h
```
</details>

<details><summary><code>paOAcc</code></summary>

```
/-- The unmasked accumulator after the non-diagonal loop has consumed keys
`[0, n)`: `Σ_t score² · v`. -/
```
```lean
noncomputable def paOAcc (s : BlockState) (q k v : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (T K V BTL BK BV : Nat) (n a p : Nat) : ℝ :=
  ∑ t ∈ Finset.range n,
    paScore s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV a t
      * paScore s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV a t
      * paVGuarded s v s_vo_h s_vo_t s_vo_d T V BV t p
```
</details>

<details><summary><code>paODiag</code></summary>

```
/-- The diagonal (causally masked) accumulator over keys
`[i_c·BTL, i)`: kept iff `t ≤ i_c·BTL + a`. -/
```
```lean
noncomputable def paODiag (s : BlockState) (q k v : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (T K V BTL BK BV : Nat) (i a p : Nat) : ℝ :=
  ∑ t ∈ Finset.Ico (s.pids 1 * BTL) i,
    if t ≤ s.pids 1 * BTL + a then
      paScore s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV a t
        * paScore s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV a t
        * paVGuarded s v s_vo_h s_vo_t s_vo_d T V BV t p
    else 0
```
</details>

<details><summary><code>paIk</code></summary>

```
/-- `i_k = i_kv // NV`. -/
```
```lean
def paIk (s : BlockState) (V BV : Nat) : Nat := s.pids 0 / paNV V BV
```
</details>

<details><summary><code>paZAcc</code></summary>

```
/-- The unmasked normalizer after keys `[0, n)`: `Σ_t score²`. -/
```
```lean
noncomputable def paZAcc (s : BlockState) (q k : RegionName)
    (s_qk_h s_qk_t s_qk_d : Nat) (scale : ℝ) (T K V BTL BK BV : Nat)
    (n a : Nat) : ℝ :=
  ∑ t ∈ Finset.range n,
    paScore s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV a t
      * paScore s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV a t
```
</details>

<details><summary><code>paZDiag</code></summary>

```
/-- The diagonal normalizer over keys `[i_c·BTL, i)`. -/
```
```lean
noncomputable def paZDiag (s : BlockState) (q k : RegionName)
    (s_qk_h s_qk_t s_qk_d : Nat) (scale : ℝ) (T K V BTL BK BV : Nat)
    (i a : Nat) : ℝ :=
  ∑ t ∈ Finset.Ico (s.pids 1 * BTL) i,
    if t ≤ s.pids 1 * BTL + a then
      paScore s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV a t
        * paScore s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV a t
    else 0
```
</details>

<details><summary><code>paNV</code></summary>

```
/-- `NV = tl.cdiv(V, BV)` as the prologue computes it. -/
```
```lean
def paNV (V BV : Nat) : Nat := (V + BV - 1) / BV
```
</details>

<details><summary><code>paScore</code></summary>

```
/-- The rebased score at `(row a, key t)`: `(scale·q) · k` over the `BK`
head window. -/
```
```lean
noncomputable def paScore (s : BlockState) (q k : RegionName)
    (s_qk_h s_qk_t s_qk_d : Nat) (scale : ℝ) (T K V BTL BK BV : Nat)
    (a t : Nat) : ℝ :=
  ∑ e : Fin BK,
    paQGuarded s q s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV a e.val
      * paKGuarded s k s_qk_h s_qk_t s_qk_d T K V BK BV e.val t
```
</details>

<details><summary><code>paVGuarded</code></summary>

```
/-- The guarded `v` lane `(t, p)` at absolute key `t`. -/
```
```lean
noncomputable def paVGuarded (s : BlockState) (v : RegionName)
    (s_vo_h s_vo_t s_vo_d : Nat) (T V BV : Nat) (t p : Nat) : ℝ :=
  if t < T ∧ paIv s V BV * BV + p < V then
    s.readMem v (s.pids 2 * s_vo_h + t * s_vo_t
      + (paIv s V BV * BV + p) * s_vo_d)
  else 0
```
</details>

<details><summary><code>paQGuarded</code></summary>

```
/-- The scaled, guarded `b_q` lane `(a, e)`: row `i_c·BTL + a` of the `(T, K)`
parent, column `i_k·BK + e`, times `scale`; `0` outside the window. -/
```
```lean
noncomputable def paQGuarded (s : BlockState) (q : RegionName)
    (s_qk_h s_qk_t s_qk_d : Nat) (scale : ℝ) (T K V BTL BK BV : Nat)
    (a e : Nat) : ℝ :=
  if s.pids 1 * BTL + a < T ∧ paIk s V BV * BK + e < K then
    s.readMem q (s.pids 2 * s_qk_h + (s.pids 1 * BTL + a) * s_qk_t
      + (paIk s V BV * BK + e) * s_qk_d) * scale
  else 0
```
</details>

<details><summary><code>paKGuarded</code></summary>

```
/-- The guarded transposed `k` lane `(e, t)` at absolute key `t` (the `(K, T)`
parent read with strides `(s_qk_d, s_qk_t)`). -/
```
```lean
noncomputable def paKGuarded (s : BlockState) (k : RegionName)
    (s_qk_h s_qk_t s_qk_d : Nat) (T K V BK BV : Nat) (e t : Nat) : ℝ :=
  if paIk s V BV * BK + e < K ∧ t < T then
    s.readMem k (s.pids 2 * s_qk_h + (paIk s V BV * BK + e) * s_qk_d
      + t * s_qk_t)
  else 0
```
</details>

## Public theorem: `pa_bwd_dkv_exec_genuine`

<details><summary>docstring</summary>

```
/-- **★ Backward dk/dv main theorem: the `dk` and `dv` stores are the genuine
causal rebased-attention gradient closed forms.**

For every scalar-argument tuple `(i_bh, i_c, i_k, i_v, i_h)` the shell would
pass (universally quantified binders; `i_h` is unused by the helper),
executing the full backward dk/dv surface succeeds, the `dk` block store
holds `Σ_{t ≥ i_c·BTL + l} 2·ds(l,t)·score(l,t)·q[e,t]` and the `dv` block
store holds `Σ_{t ≥ i_c·BTL + l} score(l,t)²·do[p,t]` at every in-window
lane — the causal key sweep from the diagonal row to the streamed top
`cdiv(T,BTS)·BTS` (loads beyond `T` read as zero, so ragged tails are
exact), with the `i_v = 0` gate folding the normalizer gradient `dz` into
`ds`.

Side conditions: `dk ≠ dv` (distinct output buffers), the host's contiguous
last-dim strides `s_k_d = 1` / `s_v_d = 1` with `BK ≤ s_k_t` / `BV ≤ s_v_t`
(store-lane injectivity), the host's own `assert BTL % BTS == 0`, and a
clean-input state (`undef ≡ 0`, the masked `dz` load's off-lanes). -/
```
</details>

**Statement:**
```lean
specification pa_bwd_dkv_exec_genuine
    (s : BlockState) (q k v do_ dz dk dv : RegionName)
    (i_bh i_c i_k i_v i_h : Nat)
    (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d : Nat) (scale : ℝ)
    (B H T K V BTL BTS BK BV : Nat)
    (hDkDv : dk ≠ dv)
    (hSkd : s_k_d = 1) (hSvd : s_v_d = 1)
    (hσk : BK ≤ s_k_t) (hσv : BV ≤ s_v_t)
    (hBTS : BTL % BTS = 0) (hBTSpos : 0 < BTS)
    (hundef : ∀ rg off, s.undef rg off = 0) :
    ∃ sF, exec (pa_bwd_dkv_surface q k v do_ dz dk dv i_bh i_c i_k i_v i_h
        s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d scale
        B H T K V BTL BTS BK BV).toAlgKernel s = some sF
      ∧ (∀ idx : TileIndex [BTL, BK], pbDkActive i_c i_k T K BTL BK idx →
          sF.readMem dk
              (pbDkOffset i_bh i_c i_k i_v B H s_k_h s_k_t s_k_d BTL BK idx)
            = pbDkOut s q k v do_ dz s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d i_bh
                i_c i_k i_v scale T K V BTL BTS BK BV idx.1.val idx.2.1.val)
      ∧ (∀ idx : TileIndex [BTL, BV], pbDvActive i_c i_v T V BTL BV idx →
          sF.readMem dv
              (pbDvOffset i_bh i_c i_k i_v B H s_v_h s_v_t s_v_d BTL BV idx)
            = pbDvOut s q k do_ s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d i_bh i_c
                i_k i_v scale T K V BTL BTS BK BV idx.1.val idx.2.1.val)
```

**Assumptions / layout contracts:**
- `hDkDv : dk ≠ dv`
- `hSkd : s_k_d = 1`
- `hSvd : s_v_d = 1`
- `hσk : BK ≤ s_k_t`
- `hσv : BV ≤ s_v_t`
- `hBTS : BTL % BTS = 0`
- `hBTSpos : 0 < BTS`
- `hundef : ∀ rg off, s.undef rg off = 0`
- `∀ idx : TileIndex [BTL, BK], pbDkActive i_c i_k T K BTL BK idx →
          sF.readMem dk
              (pbDkOffset i_bh i_c i_k i_v B H s_k_h s_k_t s_k_d BTL BK idx)
            = pbDkOut s q k v do_ dz s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d i_bh
                i_c i_k i_v scale T K V BTL BTS BK BV idx.1.val idx.2.1.val`
- `∀ idx : TileIndex [BTL, BV], pbDvActive i_c i_v T V BTL BV idx →
          sF.readMem dv
              (pbDvOffset i_bh i_c i_k i_v B H s_v_h s_v_t s_v_d BTL BV idx)
            = pbDvOut s q k do_ s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d i_bh i_c
                i_k i_v scale T K V BTL BTS BK BV idx.1.val idx.2.1.val`

**Closed-form spec defs (transitive):** `pa_bwd_dkv_surface`, `pbDkActive`, `pbDkOffset`, `pbDkOut`, `pbDvActive`, `pbDvOffset`, `pbDvOut`, `pbDkPart`, `pbNB`, `pbDvPart`, `pbDsVal`, `pbSVal`, `pbQGuarded`, `pbDoGuarded`, `pbVGuarded`, `pbDzGuarded`, `pbKGuarded`

<details><summary><code>pa_bwd_dkv_surface</code></summary>

```
/-- Faithful transcription of `_parallel_rebased_bwd_dkv`, with the helper's
scalar arguments `i_bh, i_c, i_k, i_v, i_h` as universally-quantified
binders, the descending loop spelled as its ascending change of variable
(see the preamble), and the trailing bare `return` dropped. `i_h` is unused
(rebased attention has no per-head decay); it is kept as a binder to match
the Python signature. -/
```
```lean
def pa_bwd_dkv_surface
    (q k v do_ dz dk dv : RegionName)
    (i_bh i_c i_k i_v i_h : Nat)
    (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d : Nat) (scale : ℝ)
    (B H T K V BTL BTS BK BV : Nat) :
    ComputeKernel := triton {
  p_k = tl.make_block_ptr(base=k + $(i_bh) * $(s_k_h),
    shape=($(T), $(K)), strides=($(s_k_t), $(s_k_d)),
    offsets=($(i_c) * $(BTL), $(i_k) * $(BK)), block_shape=($(BTL), $(BK)), order=(1, 0))
  p_v = tl.make_block_ptr(base=v + $(i_bh) * $(s_v_h),
    shape=($(T), $(V)), strides=($(s_v_t), $(s_v_d)),
    offsets=($(i_c) * $(BTL), $(i_v) * $(BV)), block_shape=($(BTL), $(BV)), order=(1, 0))
  b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
  b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
  b_dk = tl.zeros([$(BTL), $(BK)], dtype=tl.float32)
  b_dv = tl.zeros([$(BTL), $(BV)], dtype=tl.float32)
  for j in range($(0),
      tl.cdiv(tl.cdiv($(T), $(BTS)) * $(BTS) - $(((i_c + 1) * BTL : Nat)), $(BTS)),
      $(1)) {
    i = tl.cdiv($(T), $(BTS)) * $(BTS) - $(BTS) - j * $(BTS)
    p_q = tl.make_block_ptr(base=q + $(i_bh) * $(s_k_h),
      shape=($(K), $(T)), strides=($(s_k_d), $(s_k_t)),
      offsets=($(i_k) * $(BK), i), block_shape=($(BK), $(BTS)), order=(0, 1))
    p_do = tl.make_block_ptr(base=do_ + $(i_bh) * $(s_v_h),
      shape=($(V), $(T)), strides=($(s_v_d), $(s_v_t)),
      offsets=($(i_v) * $(BV), i), block_shape=($(BV), $(BTS)), order=(0, 1))
    p_dz = dz + $(i_bh) * $(T) + i + tl.arange(0, $(BTS))
    b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat))
    b_do = (tl.load(p_do, boundary_check=([0, 1] : List Nat))).to(b_q.dtype)
    b_dz = tl.load(p_dz, mask=((i + tl.arange(0, $(BTS))) < $(T)))
    b_s = tl.dot((b_k).to(b_q.dtype), b_q, allow_tf32=false) * $((scale : ℝ))
    b_s2 = b_s * b_s
    b_dv += tl.dot((b_s2).to(b_q.dtype), tl.trans(b_do), allow_tf32=false)
    b_ds = tl.dot(b_v, b_do, allow_tf32=false) * $((scale : ℝ))
    if $(i_v) == $(0) {
      b_ds += b_dz[None, :] * $((scale : ℝ))
    } else {
      b_ds = b_ds
    }
    b_dk += tl.dot((2.0 * b_ds * b_s).to(b_q.dtype), tl.trans(b_q), allow_tf32=false)
  }
  tl.debug_barrier()
  o_q = tl.arange(0, $(BTS))
  o_k = tl.arange(0, $(BTL))
  for i in range($(i_c) * $(BTL), $(((i_c + 1) * BTL : Nat)), $(BTS)) {
    p_q = tl.make_block_ptr(base=q + $(i_bh) * $(s_k_h),
      shape=($(K), $(T)), strides=($(s_k_d), $(s_k_t)),
      offsets=($(i_k) * $(BK), i), block_shape=($(BK), $(BTS)), order=(0, 1))
    p_do = tl.make_block_ptr(base=do_ + $(i_bh) * $(s_v_h),
      shape=($(V), $(T)), strides=($(s_v_d), $(s_v_t)),
      offsets=($(i_v) * $(BV), i), block_shape=($(BV), $(BTS)), order=(0, 1))
    p_dz = dz + $(i_bh) * $(T) + i + tl.arange(0, $(BTS))
    b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat))
    b_do = (tl.load(p_do, boundary_check=([0, 1] : List Nat))).to(b_q.dtype)
    b_dz = tl.load(p_dz, mask=((i + tl.arange(0, $(BTS))) < $(T)))
    m_s = o_k[:, None] <= o_q[None, :]
    b_s = tl.dot(b_k, b_q, allow_tf32=false) * $((scale : ℝ))
    b_s2 = b_s * b_s
    b_s = tl.where(m_s, b_s, 0.0)
    b_s2 = tl.where(m_s, b_s2, 0.0)
    b_ds = tl.dot(b_v, b_do, allow_tf32=false)
    if $(i_v) == $(0) {
      b_ds += b_dz[None, :]
    } else {
      b_ds = b_ds
    }
    b_ds = tl.where(m_s, b_ds, 0.0) * $((scale : ℝ))
    b_dv += tl.dot((b_s2).to(b_q.dtype), tl.trans(b_do), allow_tf32=false)
    b_dk += tl.dot((2.0 * b_ds * b_s).to(b_q.dtype), tl.trans(b_q), allow_tf32=false)
    o_q += $(BTS)
  }
  p_dk = tl.make_block_ptr(base=dk + $(((i_bh + B * H * i_v) * s_k_h : Nat)),
    shape=($(T), $(K)), strides=($(s_k_t), $(s_k_d)),
    offsets=($(i_c) * $(BTL), $(i_k) * $(BK)), block_shape=($(BTL), $(BK)), order=(1, 0))
  p_dv = tl.make_block_ptr(base=dv + $(((i_bh + B * H * i_k) * s_v_h : Nat)),
    shape=($(T), $(V)), strides=($(s_v_t), $(s_v_d)),
    offsets=($(i_c) * $(BTL), $(i_v) * $(BV)), block_shape=($(BTL), $(BV)), order=(1, 0))
  tl.store(p_dk, (b_dk).to(p_dk.dtype.element_ty), boundary_check=([0, 1] : List Nat))
  tl.store(p_dv, (b_dv).to(p_dv.dtype.element_ty), boundary_check=([0, 1] : List Nat))
}
```
</details>

<details><summary><code>pbDkActive</code></summary>

```
/-- A `dk` store lane is *active* when it maps inside the `T × K` window. -/
```
```lean
def pbDkActive (i_c i_k T K BTL BK : Nat) (idx : TileIndex [BTL, BK]) : Prop :=
  i_c * BTL + idx.1.val < T ∧ i_k * BK + idx.2.1.val < K
```
</details>

<details><summary><code>pbDkOffset</code></summary>

```
/-- The `dk` store address at lane `(l, e)`. -/
```
```lean
def pbDkOffset (i_bh i_c i_k i_v B H s_k_h s_k_t s_k_d BTL BK : Nat)
    (idx : TileIndex [BTL, BK]) : Nat :=
  (i_bh + B * H * i_v) * s_k_h + (i_c * BTL + idx.1.val) * s_k_t
    + (i_k * BK + idx.2.1.val) * s_k_d
```
</details>

<details><summary><code>pbDkOut</code></summary>

```
/-- **The stored `dk` lane**: the same key sweep (up to the larger of the
streamed top and the diagonal end; the kernel loads beyond `T` read as zero)
with the gate term. -/
```
```lean
noncomputable def pbDkOut (s : BlockState) (q k v do_ dz : RegionName)
    (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d : Nat) (i_bh i_c i_k i_v : Nat)
    (scale : ℝ) (T K V BTL BTS BK BV : Nat) (l e : Nat) : ℝ :=
  pbDkPart s q k v do_ dz s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d i_bh i_c i_k i_v
    scale T K V BTL BTS BK BV (i_c * BTL) (max (pbNB T BTS) ((i_c + 1) * BTL)) l e
```
</details>

<details><summary><code>pbDvActive</code></summary>

```
/-- A `dv` store lane is *active* when it maps inside the `T × V` window. -/
```
```lean
def pbDvActive (i_c i_v T V BTL BV : Nat) (idx : TileIndex [BTL, BV]) : Prop :=
  i_c * BTL + idx.1.val < T ∧ i_v * BV + idx.2.1.val < V
```
</details>

<details><summary><code>pbDvOffset</code></summary>

```
/-- The `dv` store address at lane `(l, p)`. -/
```
```lean
def pbDvOffset (i_bh i_c i_k i_v B H s_v_h s_v_t s_v_d BTL BV : Nat)
    (idx : TileIndex [BTL, BV]) : Nat :=
  (i_bh + B * H * i_k) * s_v_h + (i_c * BTL + idx.1.val) * s_v_t
    + (i_v * BV + idx.2.1.val) * s_v_d
```
</details>

<details><summary><code>pbDvOut</code></summary>

```
/-- **The stored `dv` lane**: the key sweep from `i_c·BTL` up to the larger of
the streamed top `cdiv(T,BTS)·BTS` and the diagonal end `(i_c+1)·BTL` (the
kernel loads beyond `T` read as zero, so the tail summands vanish). -/
```
```lean
noncomputable def pbDvOut (s : BlockState) (q k do_ : RegionName)
    (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d : Nat) (i_bh i_c i_k i_v : Nat)
    (scale : ℝ) (T K V BTL BTS BK BV : Nat) (l p : Nat) : ℝ :=
  pbDvPart s q k do_ s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d i_bh i_c i_k i_v
    scale T K V BTL BTS BK BV (i_c * BTL) (max (pbNB T BTS) ((i_c + 1) * BTL)) l p
```
</details>

<details><summary><code>pbDkPart</code></summary>

```
/-- The `b_dk` accumulator over keys `[lo, hi)` with the causal keep. -/
```
```lean
noncomputable def pbDkPart (s : BlockState) (q k v do_ dz : RegionName)
    (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d : Nat) (i_bh i_c i_k i_v : Nat)
    (scale : ℝ) (T K V BTL BTS BK BV : Nat) (lo hi : Nat) (l e : Nat) : ℝ :=
  ∑ t ∈ Finset.Ico lo hi,
    if i_c * BTL + l ≤ t then
      2 * pbDsVal s v do_ dz s_v_h s_v_t s_v_d i_bh i_c i_v scale T V BTL BV l t
        * pbSVal s q k s_k_h s_k_t s_k_d i_bh i_c i_k scale T K BTL BK l t
        * pbQGuarded s q s_k_h s_k_t s_k_d i_bh i_k T K BK e t
    else 0
```
</details>

<details><summary><code>pbNB</code></summary>

```
/-- The top of the streamed key range: `cdiv(T, BTS)·BTS`. -/
```
```lean
def pbNB (T BTS : Nat) : Nat := (T + BTS - 1) / BTS * BTS
```
</details>

<details><summary><code>pbDvPart</code></summary>

```
/-- The `b_dv` accumulator over keys `[lo, hi)` with the causal keep
`i_c·BTL + l ≤ t` (identically true above the diagonal chunk). -/
```
```lean
noncomputable def pbDvPart (s : BlockState) (q k do_ : RegionName)
    (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d : Nat) (i_bh i_c i_k i_v : Nat)
    (scale : ℝ) (T K V BTL BTS BK BV : Nat) (lo hi : Nat) (l p : Nat) : ℝ :=
  ∑ t ∈ Finset.Ico lo hi,
    if i_c * BTL + l ≤ t then
      pbSVal s q k s_k_h s_k_t s_k_d i_bh i_c i_k scale T K BTL BK l t
        * pbSVal s q k s_k_h s_k_t s_k_d i_bh i_c i_k scale T K BTL BK l t
        * pbDoGuarded s do_ s_v_h s_v_t s_v_d i_bh i_v T V BV p t
    else 0
```
</details>

<details><summary><code>pbDsVal</code></summary>

```
/-- The scaled backward dscore `b_ds[l, t] = (b_v · b_do + [i_v = 0]·dz)·scale`. -/
```
```lean
noncomputable def pbDsVal (s : BlockState) (v do_ dz : RegionName)
    (s_v_h s_v_t s_v_d : Nat) (i_bh i_c i_v : Nat) (scale : ℝ)
    (T V BTL BV : Nat) (l t : Nat) : ℝ :=
  ((∑ p : Fin BV,
      pbVGuarded s v s_v_h s_v_t s_v_d i_bh i_c i_v T V BTL BV l p.val
        * pbDoGuarded s do_ s_v_h s_v_t s_v_d i_bh i_v T V BV p.val t)
    + (if i_v = 0 then pbDzGuarded s dz i_bh T t else 0)) * scale
```
</details>

<details><summary><code>pbSVal</code></summary>

```
/-- The scaled backward score `b_s[l, t] = (b_k · b_q)·scale`. -/
```
```lean
noncomputable def pbSVal (s : BlockState) (q k : RegionName)
    (s_k_h s_k_t s_k_d : Nat) (i_bh i_c i_k : Nat) (scale : ℝ)
    (T K BTL BK : Nat) (l t : Nat) : ℝ :=
  (∑ e : Fin BK,
      pbKGuarded s k s_k_h s_k_t s_k_d i_bh i_c i_k T K BTL BK l e.val
        * pbQGuarded s q s_k_h s_k_t s_k_d i_bh i_k T K BK e.val t) * scale
```
</details>

<details><summary><code>pbQGuarded</code></summary>

```
/-- The guarded transposed `q` lane `(e, t)` at absolute key `t`. -/
```
```lean
noncomputable def pbQGuarded (s : BlockState) (q : RegionName)
    (s_k_h s_k_t s_k_d : Nat) (i_bh i_k : Nat) (T K BK : Nat)
    (e t : Nat) : ℝ :=
  if i_k * BK + e < K ∧ t < T then
    s.readMem q (i_bh * s_k_h + (i_k * BK + e) * s_k_d + t * s_k_t)
  else 0
```
</details>

<details><summary><code>pbDoGuarded</code></summary>

```
/-- The guarded transposed `do` lane `(p, t)` at absolute key `t`. -/
```
```lean
noncomputable def pbDoGuarded (s : BlockState) (do_ : RegionName)
    (s_v_h s_v_t s_v_d : Nat) (i_bh i_v : Nat) (T V BV : Nat)
    (p t : Nat) : ℝ :=
  if i_v * BV + p < V ∧ t < T then
    s.readMem do_ (i_bh * s_v_h + (i_v * BV + p) * s_v_d + t * s_v_t)
  else 0
```
</details>

<details><summary><code>pbVGuarded</code></summary>

```
/-- The guarded `b_v` lane `(l, p)` of the fixed chunk block. -/
```
```lean
noncomputable def pbVGuarded (s : BlockState) (v : RegionName)
    (s_v_h s_v_t s_v_d : Nat) (i_bh i_c i_v : Nat) (T V BTL BV : Nat)
    (l p : Nat) : ℝ :=
  if i_c * BTL + l < T ∧ i_v * BV + p < V then
    s.readMem v (i_bh * s_v_h + (i_c * BTL + l) * s_v_t
      + (i_v * BV + p) * s_v_d)
  else 0
```
</details>

<details><summary><code>pbDzGuarded</code></summary>

```
/-- The mask-guarded `dz` lane at absolute key `t` (masked-off lanes read the
clean-input `undef ≡ 0` oracle). -/
```
```lean
noncomputable def pbDzGuarded (s : BlockState) (dz : RegionName)
    (i_bh T : Nat) (t : Nat) : ℝ :=
  if t < T then s.readMem dz (i_bh * T + t) else 0
```
</details>

<details><summary><code>pbKGuarded</code></summary>

```
/-- The guarded `b_k` lane `(l, e)` of the fixed chunk block. -/
```
```lean
noncomputable def pbKGuarded (s : BlockState) (k : RegionName)
    (s_k_h s_k_t s_k_d : Nat) (i_bh i_c i_k : Nat) (T K BTL BK : Nat)
    (l e : Nat) : ℝ :=
  if i_c * BTL + l < T ∧ i_k * BK + e < K then
    s.readMem k (i_bh * s_k_h + (i_c * BTL + l) * s_k_t
      + (i_k * BK + e) * s_k_d)
  else 0
```
</details>
