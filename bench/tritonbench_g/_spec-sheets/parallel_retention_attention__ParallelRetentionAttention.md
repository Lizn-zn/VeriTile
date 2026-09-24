# Spec sheet — `bench/tritonbench_g/parallel_retention_attention/ParallelRetentionAttention.lean`

**Python source:** `bench/tritonbench_g/parallel_retention_attention/parallel_retention_attention.py`

## Public theorem: `pra_fwd_o_exec_genuine`

<details><summary>docstring</summary>

```
/-- **★ Forward main theorem: the `o` store is the genuine causal retention
attention closed form.**

For every program `(i_kv, i_c, i_bh)` (universally quantified through
`s.pids`), executing the full forward surface succeeds and the `o` block
store holds `Σ_{t ≤ i_c·BTL + a} 2^((i_c·BTL + a − t)·b_b) · score(a,t) ·
v[t,p]` at every in-window lane `(a, p)`, with the per-head decay
`b_b = log2(1 − 2^(−5 − i_bh % H))` and `score(a,t) = (scale·q[a]) · k[t]`
over the `BK` head window.

Side conditions: the host's contiguous last-dim stride (`s_vo_d = 1`,
`BV ≤ s_vo_t`, store-lane injectivity) and the host's own
`assert BTL % BTS == 0`. The loads are boundary-checked, so **no**
divisibility hypothesis on `T` is needed: ragged tails are exact. -/
```
</details>

**Statement:**
```lean
specification pra_fwd_o_exec_genuine
    (s : BlockState) (q k v o : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (B H T K V BTL BTS BK BV : Nat)
    (hSd : s_vo_d = 1) (hσ : BV ≤ s_vo_t)
    (hBTS : BTL % BTS = 0) (hBTSpos : 0 < BTS) :
    ∃ sF, exec (pra_fwd_surface q k v o s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t
        s_vo_d scale B H T K V BTL BTS BK BV).toAlgKernel s = some sF
      ∧ (∀ idx : TileIndex [BTL, BV], praOActive s T V BV BTL idx →
          sF.readMem o (praOOffset s B H s_vo_h s_vo_t s_vo_d V BV BTL idx)
            = praOOut s q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
                H T K V BTL BK BV idx.1.val idx.2.1.val)
```

**Assumptions / layout contracts:**
- `hSd : s_vo_d = 1`
- `hσ : BV ≤ s_vo_t`
- `hBTS : BTL % BTS = 0`
- `hBTSpos : 0 < BTS`
- `∀ idx : TileIndex [BTL, BV], praOActive s T V BV BTL idx →
          sF.readMem o (praOOffset s B H s_vo_h s_vo_t s_vo_d V BV BTL idx)
            = praOOut s q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
                H T K V BTL BK BV idx.1.val idx.2.1.val`

**Closed-form spec defs (transitive):** `pra_fwd_surface`, `praOActive`, `praOOffset`, `praOOut`, `praIv`, `praOBase`, `praW`, `praBeta`, `praScore`, `praVGuarded`, `praNV`, `praIk`, `praQGuarded`, `praKGuarded`

<details><summary><code>pra_fwd_surface</code></summary>

```
/-- Faithful transcription of `parallel_retention_fwd_kernel`. -/
```
```lean
def pra_fwd_surface
    (q k v o : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (B H T K V BTL BTS BK BV : Nat) :
    ComputeKernel := triton {
  i_kv = tl.program_id(0)
  i_c = tl.program_id(1)
  i_bh = tl.program_id(2)
  NV = tl.cdiv($(V), $(BV))
  i_k = i_kv // NV
  i_v = i_kv % NV
  i_h = i_bh % $(H)
  b_b = tl.math.log2(1.0 - tl.math.exp2(0.0 - 5.0 - tl.toReal(i_h) * 1.0))
  o_k = tl.arange(0, $(BTS))
  d_h = tl.math.exp2(tl.toReal($(BTS) - o_k) * b_b)
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
  for _i in range($(0), i_c * $(BTL), $(BTS)) {
    b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
    b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
    b_s = tl.dot(b_q, b_k, allow_tf32=false) * d_h[None, :]
    b_o = b_o * tl.math.exp2(b_b * tl.toReal($(BTS)))
    b_o = b_o + tl.dot((b_s).to(b_v.dtype), b_v, allow_tf32=false)
    p_k = tl.advance(p_k, [$(0), $(BTS)])
    p_v = tl.advance(p_v, [$(BTS), $(0)])
  }
  tl.debug_barrier()
  o_q = tl.arange(0, $(BTL))
  d_q = tl.math.exp2(tl.toReal(tl.arange(0, $(BTL))) * b_b)
  b_o *= d_q[:, None]
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
    d_s = tl.where(m_s, tl.math.exp2(tl.toReal(o_q[:, None] - o_k[None, :]) * b_b), 0.0)
    b_s = tl.dot(b_q, b_k, allow_tf32=false) * d_s
    b_o += tl.dot((b_s).to(b_q.dtype), b_v, allow_tf32=false)
    p_k = tl.advance(p_k, [$(0), $(BTS)])
    p_v = tl.advance(p_v, [$(BTS), $(0)])
    o_k += $(BTS)
  }
  p_o = tl.make_block_ptr(base=o + (i_bh + $(B) * $(H) * i_k) * $(s_vo_h),
    shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)),
    offsets=(i_c * $(BTL), i_v * $(BV)), block_shape=($(BTL), $(BV)), order=(1, 0))
  tl.store(p_o, (b_o).to(p_o.dtype.element_ty), boundary_check=([0, 1] : List Nat))
}
```
</details>

<details><summary><code>praOActive</code></summary>

```
/-- An `o` store lane is *active* when it maps inside the `T × V` window. -/
```
```lean
def praOActive (s : BlockState) (T V BV BTL : Nat)
    (idx : TileIndex [BTL, BV]) : Prop :=
  s.pids 1 * BTL + idx.1.val < T ∧ praIv s V BV * BV + idx.2.1.val < V
```
</details>

<details><summary><code>praOOffset</code></summary>

```
/-- The `o` store address at lane `(a, p)` (strides `(s_vo_t, 1)` after the
`s_vo_d = 1` substitution). -/
```
```lean
def praOOffset (s : BlockState) (B H s_vo_h s_vo_t s_vo_d V BV BTL : Nat)
    (idx : TileIndex [BTL, BV]) : Nat :=
  praOBase s B H s_vo_h V BV + (s.pids 1 * BTL + idx.1.val) * s_vo_t
    + (praIv s V BV * BV + idx.2.1.val) * s_vo_d
```
</details>

<details><summary><code>praOOut</code></summary>

```
/-- **The stored `o` lane** — causal retention attention as one sum:
`Σ_{t ≤ i_c·BTL + a} 2^((i_c·BTL + a − t)·b_b) · score(a,t) · v[t,p]` over
the keys `[0, (i_c+1)·BTL)` the program consumes. -/
```
```lean
noncomputable def praOOut (s : BlockState) (q k v : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (H T K V BTL BK BV : Nat) (a p : Nat) : ℝ :=
  ∑ t ∈ Finset.range ((s.pids 1 + 1) * BTL),
    if t ≤ s.pids 1 * BTL + a then
      praW (praBeta s H) (s.pids 1 * BTL + a - t)
        * praScore s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV a t
        * praVGuarded s v s_vo_h s_vo_t s_vo_d T V BV t p
    else 0
```
</details>

<details><summary><code>praIv</code></summary>

```
/-- `i_v = i_kv % NV`. -/
```
```lean
def praIv (s : BlockState) (V BV : Nat) : Nat := s.pids 0 % praNV V BV
```
</details>

<details><summary><code>praOBase</code></summary>

```
/-- The `o` store base `(i_bh + B·H·i_k) · s_vo_h`. -/
```
```lean
def praOBase (s : BlockState) (B H s_vo_h V BV : Nat) : Nat :=
  (s.pids 2 + B * H * praIk s V BV) * s_vo_h
```
</details>

<details><summary><code>praW</code></summary>

```
/-- The retention decay weight `γ^n = 2^(n·b_b)`, in the walk's exact form. -/
```
```lean
noncomputable def praW (β : ℝ) (n : Nat) : ℝ :=
  Real.exp ((n : ℝ) * β * Real.log 2)
```
</details>

<details><summary><code>praBeta</code></summary>

```
/-- The per-head decay exponent `b_b = log2(1 - 2^(-5 - i_h))` with
`i_h = i_bh % H`, exactly as the walk computes it. -/
```
```lean
noncomputable def praBeta (s : BlockState) (H : Nat) : ℝ :=
  Real.log ((1.0 : ℝ) - Real.exp ((((0.0 : ℝ) - 5.0)
      - ((s.pids 2 % H : Nat) : ℝ) * 1.0) * Real.log 2)) / Real.log 2
```
</details>

<details><summary><code>praScore</code></summary>

```
/-- The retention score at `(row a, key t)`: `(scale·q) · k` over the `BK`
head window. -/
```
```lean
noncomputable def praScore (s : BlockState) (q k : RegionName)
    (s_qk_h s_qk_t s_qk_d : Nat) (scale : ℝ) (T K V BTL BK BV : Nat)
    (a t : Nat) : ℝ :=
  ∑ e : Fin BK,
    praQGuarded s q s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV a e.val
      * praKGuarded s k s_qk_h s_qk_t s_qk_d T K V BK BV e.val t
```
</details>

<details><summary><code>praVGuarded</code></summary>

```
/-- The guarded `v` lane `(t, p)` at absolute key `t`. -/
```
```lean
noncomputable def praVGuarded (s : BlockState) (v : RegionName)
    (s_vo_h s_vo_t s_vo_d : Nat) (T V BV : Nat) (t p : Nat) : ℝ :=
  if t < T ∧ praIv s V BV * BV + p < V then
    s.readMem v (s.pids 2 * s_vo_h + t * s_vo_t
      + (praIv s V BV * BV + p) * s_vo_d)
  else 0
```
</details>

<details><summary><code>praNV</code></summary>

```
/-- `NV = tl.cdiv(V, BV)` as the prologue computes it. -/
```
```lean
def praNV (V BV : Nat) : Nat := (V + BV - 1) / BV
```
</details>

<details><summary><code>praIk</code></summary>

```
/-- `i_k = i_kv // NV`. -/
```
```lean
def praIk (s : BlockState) (V BV : Nat) : Nat := s.pids 0 / praNV V BV
```
</details>

<details><summary><code>praQGuarded</code></summary>

```
/-- The scaled, guarded `b_q` lane `(a, e)`: row `i_c·BTL + a` of the `(T, K)`
parent, column `i_k·BK + e`, times `scale`; `0` outside the window. -/
```
```lean
noncomputable def praQGuarded (s : BlockState) (q : RegionName)
    (s_qk_h s_qk_t s_qk_d : Nat) (scale : ℝ) (T K V BTL BK BV : Nat)
    (a e : Nat) : ℝ :=
  if s.pids 1 * BTL + a < T ∧ praIk s V BV * BK + e < K then
    s.readMem q (s.pids 2 * s_qk_h + (s.pids 1 * BTL + a) * s_qk_t
      + (praIk s V BV * BK + e) * s_qk_d) * scale
  else 0
```
</details>

<details><summary><code>praKGuarded</code></summary>

```
/-- The guarded transposed `k` lane `(e, t)` at absolute key `t` (the `(K, T)`
parent read with strides `(s_qk_d, s_qk_t)`). -/
```
```lean
noncomputable def praKGuarded (s : BlockState) (k : RegionName)
    (s_qk_h s_qk_t s_qk_d : Nat) (T K V BK BV : Nat) (e t : Nat) : ℝ :=
  if praIk s V BV * BK + e < K ∧ t < T then
    s.readMem k (s.pids 2 * s_qk_h + (praIk s V BV * BK + e) * s_qk_d
      + t * s_qk_t)
  else 0
```
</details>

## Public theorem: `pra_bwd_dkv_exec_genuine`

<details><summary>docstring</summary>

```
/-- **★ Backward dk/dv main theorem: the `dk` and `dv` stores are the genuine
retention-gradient closed forms.**

For every scalar-argument tuple `(i_bh, i_c, i_k, i_v, i_h)` the shell would
pass (universally quantified binders), executing the full backward dk/dv
surface succeeds, the `dk` block store holds
`Σ_{t ≥ i_c·BTL + r} 2^((t − (i_c·BTL+r))·b_b) · scale · ds(r,t) · qᵀ[e,t]`
and the `dv` block store holds
`Σ_{t ≥ i_c·BTL + r} 2^((t − (i_c·BTL+r))·b_b) · scale · s(r,t) · doᵀ[p,t]`
at every in-window lane — the causally kept key sweep from the diagonal row
to the streamed top `cdiv(T,BTS)·BTS` (loads beyond `T` read as zero, so
ragged tails are exact), with the per-head decay
`b_b = log2(1 − 2^(−5 − i_h))`, `s(r,t) = k[r]·qᵀ[t]` the raw retention
score, and `ds(r,t) = v[r]·doᵀ[t]` the raw output-gradient contraction.

Side conditions: `dk ≠ dv` (distinct output buffers), the host's contiguous
last-dim strides `s_qk_d = 1` / `s_vo_d = 1` with `BK ≤ s_qk_t` /
`BV ≤ s_vo_t` (store-lane injectivity), and the host's own
`assert BTL % BTS == 0`. -/
```
</details>

**Statement:**
```lean
specification pra_bwd_dkv_exec_genuine
    (s : BlockState) (q k v do_ dk dv : RegionName)
    (i_bh i_c i_k i_v i_h : Nat)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (B H T K V BTL BTS BK BV : Nat)
    (hDkDv : dk ≠ dv) (hSkd : s_qk_d = 1) (hSvd : s_vo_d = 1)
    (hσk : BK ≤ s_qk_t) (hσv : BV ≤ s_vo_t)
    (hBTS : BTL % BTS = 0) (hBTSpos : 0 < BTS) :
    ∃ sF, exec (pra_bwd_dkv_surface q k v do_ dk dv i_bh i_c i_k i_v i_h
        s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
        B H T K V BTL BTS BK BV).toAlgKernel s = some sF
      ∧ (∀ idx : TileIndex [BTL, BK], prbDkActive i_c i_k T K BTL BK idx →
          sF.readMem dk
              (prbDkOffset i_bh i_c i_k i_v B H s_qk_h s_qk_t s_qk_d BTL BK idx)
            = prbDkOut s q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
                i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
                idx.1.val idx.2.1.val)
      ∧ (∀ idx : TileIndex [BTL, BV], prbDvActive i_c i_v T V BTL BV idx →
          sF.readMem dv
              (prbDvOffset i_bh i_c i_k i_v B H s_vo_h s_vo_t s_vo_d BTL BV idx)
            = prbDvOut s q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
                i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
                idx.1.val idx.2.1.val)
```

**Assumptions / layout contracts:**
- `hDkDv : dk ≠ dv`
- `hSkd : s_qk_d = 1`
- `hSvd : s_vo_d = 1`
- `hσk : BK ≤ s_qk_t`
- `hσv : BV ≤ s_vo_t`
- `hBTS : BTL % BTS = 0`
- `hBTSpos : 0 < BTS`
- `∀ idx : TileIndex [BTL, BK], prbDkActive i_c i_k T K BTL BK idx →
          sF.readMem dk
              (prbDkOffset i_bh i_c i_k i_v B H s_qk_h s_qk_t s_qk_d BTL BK idx)
            = prbDkOut s q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
                i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
                idx.1.val idx.2.1.val`
- `∀ idx : TileIndex [BTL, BV], prbDvActive i_c i_v T V BTL BV idx →
          sF.readMem dv
              (prbDvOffset i_bh i_c i_k i_v B H s_vo_h s_vo_t s_vo_d BTL BV idx)
            = prbDvOut s q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
                i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
                idx.1.val idx.2.1.val`

**Closed-form spec defs (transitive):** `pra_bwd_dkv_surface`, `prbDkActive`, `prbDkOffset`, `prbDkOut`, `prbDvActive`, `prbDvOffset`, `prbDvOut`, `prbNB`, `praW`, `prbBeta`, `prbDsVal`, `prbQGuarded`, `prbSVal`, `prbDoGuarded`, `prbVGuarded`, `prbKGuarded`

<details><summary><code>pra_bwd_dkv_surface</code></summary>

```
/-- Faithful transcription of `_parallel_retention_bwd_dkv`, with the
helper's scalar arguments `i_bh, i_c, i_k, i_v, i_h` as
universally-quantified binders, the descending loop spelled as its ascending
change of variable, the diagonal unary-minus decay respelled as a
subtraction (see the preamble), and the trailing bare `return` dropped. -/
```
```lean
def pra_bwd_dkv_surface
    (q k v do_ dk dv : RegionName)
    (i_bh i_c i_k i_v i_h : Nat)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (B H T K V BTL BTS BK BV : Nat) :
    ComputeKernel := triton {
  b_b = tl.math.log2(1.0 - tl.math.exp2(0.0 - 5.0 - tl.toReal($(i_h)) * 1.0))
  d_b = tl.math.exp2(b_b * tl.toReal($(BTS)))
  p_k = tl.make_block_ptr(base=k + $(i_bh) * $(s_qk_h),
    shape=($(T), $(K)), strides=($(s_qk_t), $(s_qk_d)),
    offsets=($(i_c) * $(BTL), $(i_k) * $(BK)), block_shape=($(BTL), $(BK)), order=(1, 0))
  p_v = tl.make_block_ptr(base=v + $(i_bh) * $(s_vo_h),
    shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)),
    offsets=($(i_c) * $(BTL), $(i_v) * $(BV)), block_shape=($(BTL), $(BV)), order=(1, 0))
  b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
  b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
  b_dk = tl.zeros([$(BTL), $(BK)], dtype=tl.float32)
  b_dv = tl.zeros([$(BTL), $(BV)], dtype=tl.float32)
  d_h = tl.math.exp2(tl.toReal($(BTL) - tl.arange(0, $(BTL))) * b_b)
  b_kd = (b_k * d_h[:, None]).to(b_k.dtype)
  d_q = tl.math.exp2(tl.toReal(tl.arange(0, $(BTS))) * b_b)
  for j in range($(0),
      tl.cdiv(tl.cdiv($(T), $(BTS)) * $(BTS) - $(((i_c + 1) * BTL : Nat)), $(BTS)),
      $(1)) {
    i = tl.cdiv($(T), $(BTS)) * $(BTS) - $(BTS) - j * $(BTS)
    p_q = tl.make_block_ptr(base=q + $(i_bh) * $(s_qk_h),
      shape=($(K), $(T)), strides=($(s_qk_d), $(s_qk_t)),
      offsets=($(i_k) * $(BK), i), block_shape=($(BK), $(BTS)), order=(0, 1))
    p_do = tl.make_block_ptr(base=do_ + $(i_bh) * $(s_vo_h),
      shape=($(V), $(T)), strides=($(s_vo_d), $(s_vo_t)),
      offsets=($(i_v) * $(BV), i), block_shape=($(BV), $(BTS)), order=(0, 1))
    b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat))
    b_do = tl.load(p_do, boundary_check=([0, 1] : List Nat))
    b_do = (b_do * d_q[None, :]).to(b_do.dtype)
    b_dv *= d_b
    b_s = tl.dot((b_kd).to(b_q.dtype), b_q, allow_tf32=false)
    b_dv += tl.dot((b_s).to(b_q.dtype), tl.trans(b_do), allow_tf32=false)
    b_dk *= d_b
    b_ds = tl.dot(b_v, b_do, allow_tf32=false)
    b_dk += tl.dot((b_ds).to(b_q.dtype), tl.trans(b_q), allow_tf32=false)
  }
  b_dk *= d_h[:, None] * $((scale : ℝ))
  b_dv *= $((scale : ℝ))
  tl.debug_barrier()
  o_q = tl.arange(0, $(BTS))
  o_k = tl.arange(0, $(BTL))
  for i in range($(i_c) * $(BTL), $(((i_c + 1) * BTL : Nat)), $(BTS)) {
    p_q = tl.make_block_ptr(base=q + $(i_bh) * $(s_qk_h),
      shape=($(K), $(T)), strides=($(s_qk_d), $(s_qk_t)),
      offsets=($(i_k) * $(BK), i), block_shape=($(BK), $(BTS)), order=(0, 1))
    p_do = tl.make_block_ptr(base=do_ + $(i_bh) * $(s_vo_h),
      shape=($(V), $(T)), strides=($(s_vo_d), $(s_vo_t)),
      offsets=($(i_v) * $(BV), i), block_shape=($(BV), $(BTS)), order=(0, 1))
    b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat))
    b_do = tl.load(p_do, boundary_check=([0, 1] : List Nat))
    m_s = o_k[:, None] <= o_q[None, :]
    d_s = tl.where(m_s,
      tl.math.exp2(tl.toReal(o_q[None, :] - o_k[:, None]) * (b_b).to(tl.float32)),
      0.0) * $((scale : ℝ))
    b_s = tl.dot(b_k, b_q, allow_tf32=false) * d_s
    b_ds = tl.dot(b_v, b_do, allow_tf32=false) * d_s
    b_dk += tl.dot((b_ds).to(b_q.dtype), tl.trans(b_q), allow_tf32=false)
    b_dv += tl.dot((b_s).to(b_q.dtype), tl.trans(b_do), allow_tf32=false)
    o_q += $(BTS)
  }
  p_dk = tl.make_block_ptr(base=dk + $(((i_bh + B * H * i_v) * s_qk_h : Nat)),
    shape=($(T), $(K)), strides=($(s_qk_t), $(s_qk_d)),
    offsets=($(i_c) * $(BTL), $(i_k) * $(BK)), block_shape=($(BTL), $(BK)), order=(1, 0))
  p_dv = tl.make_block_ptr(base=dv + $(((i_bh + B * H * i_k) * s_vo_h : Nat)),
    shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)),
    offsets=($(i_c) * $(BTL), $(i_v) * $(BV)), block_shape=($(BTL), $(BV)), order=(1, 0))
  tl.store(p_dk, (b_dk).to(p_dk.dtype.element_ty), boundary_check=([0, 1] : List Nat))
  tl.store(p_dv, (b_dv).to(p_dv.dtype.element_ty), boundary_check=([0, 1] : List Nat))
}
```
</details>

<details><summary><code>prbDkActive</code></summary>

```
/-- A `dk` store lane is *active* when it maps inside the `T × K` window. -/
```
```lean
def prbDkActive (i_c i_k T K BTL BK : Nat) (idx : TileIndex [BTL, BK]) : Prop :=
  i_c * BTL + idx.1.val < T ∧ i_k * BK + idx.2.1.val < K
```
</details>

<details><summary><code>prbDkOffset</code></summary>

```
/-- The `dk` store address at lane `(r, e)`. -/
```
```lean
def prbDkOffset (i_bh i_c i_k i_v B H s_qk_h s_qk_t s_qk_d BTL BK : Nat)
    (idx : TileIndex [BTL, BK]) : Nat :=
  (i_bh + B * H * i_v) * s_qk_h + (i_c * BTL + idx.1.val) * s_qk_t
    + (i_k * BK + idx.2.1.val) * s_qk_d
```
</details>

<details><summary><code>prbDkOut</code></summary>

```
/-- **The stored `dk` lane** — the retention key-gradient closed form. Over
the full key sweep (up to the larger of the streamed top `cdiv(T,BTS)·BTS`
and the diagonal end `(i_c+1)·BTL`; loads beyond `T` read as zero, so the
tail summands vanish), every causally kept key `t ≥ i_c·BTL + r` contributes
its decayed, scaled dscore times the transposed-`q` lane:
`Σ_t 2^((t − (i_c·BTL+r))·b_b) · scale · (v[r]·doᵀ[t]) · q[e, t]`. -/
```
```lean
noncomputable def prbDkOut (s : BlockState) (q k v do_ : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat)
    (i_bh i_c i_k i_v i_h : Nat) (scale : ℝ) (T K V BTL BTS BK BV : Nat)
    (r e : Nat) : ℝ :=
  ∑ t ∈ Finset.range (max (prbNB T BTS) ((i_c + 1) * BTL)),
    if i_c * BTL + r ≤ t then
      praW (prbBeta i_h) (t - (i_c * BTL + r)) * scale
        * prbDsVal s v do_ s_vo_h s_vo_t s_vo_d i_bh i_c i_v T V BTL BV r t
        * prbQGuarded s q s_qk_h s_qk_t s_qk_d i_bh i_k T K BK e t
    else 0
```
</details>

<details><summary><code>prbDvActive</code></summary>

```
/-- A `dv` store lane is *active* when it maps inside the `T × V` window. -/
```
```lean
def prbDvActive (i_c i_v T V BTL BV : Nat) (idx : TileIndex [BTL, BV]) : Prop :=
  i_c * BTL + idx.1.val < T ∧ i_v * BV + idx.2.1.val < V
```
</details>

<details><summary><code>prbDvOffset</code></summary>

```
/-- The `dv` store address at lane `(r, p)`. -/
```
```lean
def prbDvOffset (i_bh i_c i_k i_v B H s_vo_h s_vo_t s_vo_d BTL BV : Nat)
    (idx : TileIndex [BTL, BV]) : Nat :=
  (i_bh + B * H * i_k) * s_vo_h + (i_c * BTL + idx.1.val) * s_vo_t
    + (i_v * BV + idx.2.1.val) * s_vo_d
```
</details>

<details><summary><code>prbDvOut</code></summary>

```
/-- **The stored `dv` lane** — the retention value-gradient closed form:
`Σ_t 2^((t − (i_c·BTL+r))·b_b) · scale · (k[r]·qᵀ[t]) · do[p, t]` over the
same causally kept key sweep. -/
```
```lean
noncomputable def prbDvOut (s : BlockState) (q k v do_ : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat)
    (i_bh i_c i_k i_v i_h : Nat) (scale : ℝ) (T K V BTL BTS BK BV : Nat)
    (r p : Nat) : ℝ :=
  ∑ t ∈ Finset.range (max (prbNB T BTS) ((i_c + 1) * BTL)),
    if i_c * BTL + r ≤ t then
      praW (prbBeta i_h) (t - (i_c * BTL + r)) * scale
        * prbSVal s q k s_qk_h s_qk_t s_qk_d i_bh i_c i_k T K BTL BK r t
        * prbDoGuarded s do_ s_vo_h s_vo_t s_vo_d i_bh i_v T V BV p t
    else 0
```
</details>

<details><summary><code>prbNB</code></summary>

```
/-- The top of the streamed key range: `cdiv(T, BTS)·BTS`. -/
```
```lean
def prbNB (T BTS : Nat) : Nat := (T + BTS - 1) / BTS * BTS
```
</details>

<details><summary><code>praW</code></summary>

```
/-- The retention decay weight `γ^n = 2^(n·b_b)`, in the walk's exact form. -/
```
```lean
noncomputable def praW (β : ℝ) (n : Nat) : ℝ :=
  Real.exp ((n : ℝ) * β * Real.log 2)
```
</details>

<details><summary><code>prbBeta</code></summary>

```
/-- The per-head decay exponent `b_b = log2(1 − 2^(−5 − i_h))` on the spliced
`i_h` constant, exactly as the walk computes it. -/
```
```lean
noncomputable def prbBeta (i_h : Nat) : ℝ :=
  Real.log ((1.0 : ℝ) - Real.exp ((((0.0 : ℝ) - 5.0) - ((i_h : Nat) : ℝ) * 1.0)
      * Real.log 2)) / Real.log 2
```
</details>

<details><summary><code>prbDsVal</code></summary>

```
/-- The (unscaled) backward dscore core at `(row r, key t)`: `v[r] · doᵀ[t]`
over the `BV` value window. -/
```
```lean
noncomputable def prbDsVal (s : BlockState) (v do_ : RegionName)
    (s_vo_h s_vo_t s_vo_d : Nat) (i_bh i_c i_v : Nat) (T V BTL BV : Nat)
    (r t : Nat) : ℝ :=
  ∑ p : Fin BV,
    prbVGuarded s v s_vo_h s_vo_t s_vo_d i_bh i_c i_v T V BTL BV r p.val
      * prbDoGuarded s do_ s_vo_h s_vo_t s_vo_d i_bh i_v T V BV p.val t
```
</details>

<details><summary><code>prbQGuarded</code></summary>

```
/-- The guarded transposed `q` lane `(e, t)` at absolute key `t` (the `(K, T)`
parent read with strides `(s_qk_d, s_qk_t)`). -/
```
```lean
noncomputable def prbQGuarded (s : BlockState) (q : RegionName)
    (s_qk_h s_qk_t s_qk_d : Nat) (i_bh i_k : Nat) (T K BK : Nat)
    (e t : Nat) : ℝ :=
  if i_k * BK + e < K ∧ t < T then
    s.readMem q (i_bh * s_qk_h + (i_k * BK + e) * s_qk_d + t * s_qk_t)
  else 0
```
</details>

<details><summary><code>prbSVal</code></summary>

```
/-- The (unscaled) backward score core at `(row r, key t)`: `k[r] · qᵀ[t]`
over the `BK` head window. -/
```
```lean
noncomputable def prbSVal (s : BlockState) (q k : RegionName)
    (s_qk_h s_qk_t s_qk_d : Nat) (i_bh i_c i_k : Nat) (T K BTL BK : Nat)
    (r t : Nat) : ℝ :=
  ∑ e : Fin BK,
    prbKGuarded s k s_qk_h s_qk_t s_qk_d i_bh i_c i_k T K BTL BK r e.val
      * prbQGuarded s q s_qk_h s_qk_t s_qk_d i_bh i_k T K BK e.val t
```
</details>

<details><summary><code>prbDoGuarded</code></summary>

```
/-- The guarded transposed `do` lane `(p, t)` at absolute key `t`. -/
```
```lean
noncomputable def prbDoGuarded (s : BlockState) (do_ : RegionName)
    (s_vo_h s_vo_t s_vo_d : Nat) (i_bh i_v : Nat) (T V BV : Nat)
    (p t : Nat) : ℝ :=
  if i_v * BV + p < V ∧ t < T then
    s.readMem do_ (i_bh * s_vo_h + (i_v * BV + p) * s_vo_d + t * s_vo_t)
  else 0
```
</details>

<details><summary><code>prbVGuarded</code></summary>

```
/-- The guarded `b_v` lane `(r, p)` of the fixed chunk block. -/
```
```lean
noncomputable def prbVGuarded (s : BlockState) (v : RegionName)
    (s_vo_h s_vo_t s_vo_d : Nat) (i_bh i_c i_v : Nat) (T V BTL BV : Nat)
    (r p : Nat) : ℝ :=
  if i_c * BTL + r < T ∧ i_v * BV + p < V then
    s.readMem v (i_bh * s_vo_h + (i_c * BTL + r) * s_vo_t
      + (i_v * BV + p) * s_vo_d)
  else 0
```
</details>

<details><summary><code>prbKGuarded</code></summary>

```
/-- The guarded `b_k` lane `(r, e)` of the fixed chunk block. -/
```
```lean
noncomputable def prbKGuarded (s : BlockState) (k : RegionName)
    (s_qk_h s_qk_t s_qk_d : Nat) (i_bh i_c i_k : Nat) (T K BTL BK : Nat)
    (r e : Nat) : ℝ :=
  if i_c * BTL + r < T ∧ i_k * BK + e < K then
    s.readMem k (i_bh * s_qk_h + (i_c * BTL + r) * s_qk_t
      + (i_k * BK + e) * s_qk_d)
  else 0
```
</details>
