# Spec sheet — `bench/tritonbench_g/chunk_gla_fwd/ChunkGlaFwd.lean`

**Python source:** `bench/tritonbench_g/chunk_gla_fwd/chunk_gla_fwd.py`

## Public theorem: `chunk_gla_fwd_o_exec_genuine`

<details><summary>docstring</summary>

```
/-- **Genuine, dimension-general correctness** of `chunk_gla_fwd_kernel_o`. For
every launch state, the kernel runs to completion and every in-window output lane
holds `cgfOutput`: the inter-chunk sum over **all** `cdiv(K, BK)` K blocks of the
scaled, gated query against the chunk state, plus the causally-masked intra-chunk
`A · V` term.

`hInj` says distinct output lanes get distinct `o` addresses — the standard
row-major side condition, exactly as in the ported sibling `chunk_gla_simple`.
Unlike that sibling, there is **no** `K = BK` hypothesis: the K loop is verified in
full multi-block generality. -/
```
</details>

**Statement:**
```lean
specification chunk_gla_fwd_o_exec_genuine
    (q v g h o A : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV : Nat) (s : BlockState)
    (hInj : Function.Injective
      (fun idx : TileIndex [BT, BV] => outOffset s s_v_h s_v_t BT BV idx)) :
    ∃ sF, exec (chunk_gla_fwd_o_surface q v g h o A s_k_h s_k_t s_v_h s_v_t
        s_h_h s_h_t scale T K V BT BK BV).toAlgKernel s = some sF
      ∧ ∀ idx : TileIndex [BT, BV],
          (tIndex s BT idx.1.val < T ∧ vIndex s BV idx.2.1.val < V) →
          sF.readMem o (outOffset s s_v_h s_v_t BT BV idx)
            = cgfOutput s q v g h A s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale
                T K V BT BK BV idx.1.val idx.2.1.val
```

**Assumptions / layout contracts:**
- `hInj : Function.Injective
      (fun idx : TileIndex [BT, BV] => outOffset s s_v_h s_v_t BT BV idx)`

**Closed-form spec defs (transitive):** `outOffset`, `chunk_gla_fwd_o_surface`, `tIndex`, `vIndex`, `cgfOutput`, `numKB`, `qgElem`, `hGuarded`, `aMasked`, `vGuarded`, `kIndex`, `qElem`, `gElem`, `hElem`, `aElem`, `vElem`

<details><summary><code>outOffset</code></summary>

```
/-- The output store address for lane `(i, p)` — `p_o` shares `p_v`'s layout. -/
```
```lean
def outOffset (s : BlockState) (s_v_h s_v_t BT BV : Nat)
    (idx : TileIndex [BT, BV]) : Nat :=
  s.pids 2 * s_v_h + tIndex s BT idx.1.val * s_v_t + vIndex s BV idx.2.1.val * 1
```
</details>

<details><summary><code>chunk_gla_fwd_o_surface</code></summary>

```lean
def chunk_gla_fwd_o_surface
    (q v g h o A : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV : Nat) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_t = tl.program_id(1)
  i_bh = tl.program_id(2)
  m_s = tl.arange(0, $(BT))[:, None] >= tl.arange(0, $(BT))[None, :]
  b_o = tl.zeros([$(BT), $(BV)], dtype=tl.float32)
  for i_k in range($(0), tl.cdiv($(K), $(BK)), $(1)) {
    p_q = tl.make_block_ptr(base=q + i_bh * $(s_k_h),
      shape=($(T), $(K)), strides=($(s_k_t), $(1)),
      offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
    p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h),
      shape=($(T), $(K)), strides=($(s_k_t), $(1)),
      offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
    p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V),
      shape=($(K), $(V)), strides=($(s_h_t), $(1)),
      offsets=(i_k * $(BK), i_v * $(BV)), block_shape=($(BK), $(BV)), order=(1, 0))
    b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat))
    b_q = (b_q * $(scale)).to(b_q.dtype)
    b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
    b_qg = (b_q * tl.exp(b_g)).to(b_q.dtype)
    b_h = tl.load(p_h, boundary_check=([0, 1] : List Nat))
    if i_k >= $(0) {
      b_o += tl.dot(b_qg, (b_h).to(b_qg.dtype))
    }
  }
  p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h),
    shape=($(T), $(V)), strides=($(s_v_t), $(1)),
    offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
  p_o = tl.make_block_ptr(base=o + i_bh * $(s_v_h),
    shape=($(T), $(V)), strides=($(s_v_t), $(1)),
    offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
  p_A = tl.make_block_ptr(base=A + i_bh * $(T) * $(BT),
    shape=($(T), $(BT)), strides=($(BT), $(1)),
    offsets=(i_t * $(BT), $(0)), block_shape=($(BT), $(BT)), order=(1, 0))
  b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
  b_A = tl.load(p_A, boundary_check=([0, 1] : List Nat))
  b_A = (tl.where(m_s, b_A, 0.0)).to(b_v.dtype)
  b_o += tl.dot(b_A, b_v, allow_tf32=false)
  tl.store(p_o, (b_o).to(p_o.dtype.element_ty), boundary_check=([0, 1] : List Nat))
}
```
</details>

<details><summary><code>tIndex</code></summary>

```
/-- Global row (time) index of tile lane `i`: `i_t · BT + i`. -/
```
```lean
def tIndex (s : BlockState) (BT : Nat) (i : Nat) : Nat := s.pids 1 * BT + i
```
</details>

<details><summary><code>vIndex</code></summary>

```
/-- Global value (column) index of tile lane `p`: `i_v · BV + p`. -/
```
```lean
def vIndex (s : BlockState) (BV : Nat) (p : Nat) : Nat := s.pids 0 * BV + p
```
</details>

<details><summary><code>cgfOutput</code></summary>

```
/-- **The stored value.** Lane `(i, p)` of `b_o` after the K loop and the intra-chunk
`tl.dot`: the inter-chunk sum over every K block, plus the causally-masked `A · V`. -/
```
```lean
noncomputable def cgfOutput (s : BlockState) (q v g h A : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat) (scale : ℝ)
    (T K V BT BK BV : Nat) (i p : Nat) : ℝ :=
  (∑ kb : Fin (numKB K BK), ∑ e : Fin BK,
      qgElem s q g s_k_h s_k_t scale T K BT BK kb.val i e.val
        * hGuarded s h s_h_h s_h_t K V BV kb.val BK e.val p)
    + ∑ j : Fin BT,
        aMasked s A T BT i j.val * vGuarded s v s_v_h s_v_t T V BT BV j.val p
```
</details>

<details><summary><code>numKB</code></summary>

```
/-- `cdiv(K, BK)` — the K-block trip count. -/
```
```lean
def numKB (K BK : Nat) : Nat := (K + BK - 1) / BK
```
</details>

<details><summary><code>qgElem</code></summary>

```
/-- The gated query lane, as the loop body's `b_qg` holds it:
`(q · scale) · exp(g)`, or `0` outside the `(T, K)` parent window. `q` and `g` share
one block-pointer geometry, so one guard covers both — and an out-of-bounds lane is
`0` regardless of the gate, since the load returns `0` and `0 · exp(0) = 0`. -/
```
```lean
noncomputable def qgElem (s : BlockState) (q g : RegionName)
    (s_k_h s_k_t : Nat) (scale : ℝ) (T K BT BK : Nat) (kb i e : Nat) : ℝ :=
  if tIndex s BT i < T ∧ kIndex BK kb e < K then
    qElem s q s_k_h s_k_t BT BK kb i e * scale
      * Real.exp (gElem s g s_k_h s_k_t BT BK kb i e)
  else 0
```
</details>

<details><summary><code>hGuarded</code></summary>

```
/-- The chunk-state lane, as `b_h` holds it. -/
```
```lean
noncomputable def hGuarded (s : BlockState) (h : RegionName)
    (s_h_h s_h_t : Nat) (K V BV : Nat) (kb BK e p : Nat) : ℝ :=
  if kIndex BK kb e < K ∧ vIndex s BV p < V then
    hElem s h s_h_h s_h_t K V BV kb BK e p
  else 0
```
</details>

<details><summary><code>aMasked</code></summary>

```
/-- The intra-chunk matrix lane, as `b_A` holds it after `tl.where(m_s, b_A, 0.)`:
the causal mask `i ≥ j` on top of the row boundary check. The **column** check is
vacuous — `p_A`'s parent shape is `(T, BT)` and the lane index `j` is a `Fin BT` — so
only the row guard bites. -/
```
```lean
noncomputable def aMasked (s : BlockState) (A : RegionName) (T BT : Nat)
    (i j : Nat) : ℝ :=
  if j ≤ i then (if tIndex s BT i < T then aElem s A T BT i j else 0) else 0
```
</details>

<details><summary><code>vGuarded</code></summary>

```
/-- The value lane, as `b_v` holds it. -/
```
```lean
noncomputable def vGuarded (s : BlockState) (v : RegionName)
    (s_v_h s_v_t : Nat) (T V BT BV : Nat) (j p : Nat) : ℝ :=
  if tIndex s BT j < T ∧ vIndex s BV p < V then
    vElem s v s_v_h s_v_t BT BV j p
  else 0
```
</details>

<details><summary><code>kIndex</code></summary>

```
/-- Global key index of lane `e` in K block `kb`: `kb · BK + e`. -/
```
```lean
def kIndex (BK kb e : Nat) : Nat := kb * BK + e
```
</details>

<details><summary><code>qElem</code></summary>

```
/-- `q[i, kIndex kb e]`, at the address `p_q` computes. -/
```
```lean
noncomputable def qElem (s : BlockState) (q : RegionName) (s_k_h s_k_t BT BK : Nat)
    (kb : Nat) (i e : Nat) : ℝ :=
  s.readMem q (s.pids 2 * s_k_h + tIndex s BT i * s_k_t + kIndex BK kb e * 1)
```
</details>

<details><summary><code>gElem</code></summary>

```
/-- `g[i, kIndex kb e]` — the **2-D** gate, on `q`'s own layout (same base stride
`s_k_h`, same parent shape `(T, K)`). This is what separates this kernel from
`chunk_gla_simple`, whose gate is a `[T]` vector read once after the loop. -/
```
```lean
noncomputable def gElem (s : BlockState) (g : RegionName) (s_k_h s_k_t BT BK : Nat)
    (kb : Nat) (i e : Nat) : ℝ :=
  s.readMem g (s.pids 2 * s_k_h + tIndex s BT i * s_k_t + kIndex BK kb e * 1)
```
</details>

<details><summary><code>hElem</code></summary>

```
/-- `h[kIndex kb e, vIndex p]`, the chunk state at base `h + i_bh·s_h_h + i_t·K·V`. -/
```
```lean
noncomputable def hElem (s : BlockState) (h : RegionName) (s_h_h s_h_t K V BV : Nat)
    (kb BK : Nat) (e p : Nat) : ℝ :=
  s.readMem h (s.pids 2 * s_h_h + s.pids 1 * K * V + kIndex BK kb e * s_h_t
    + vIndex s BV p * 1)
```
</details>

<details><summary><code>aElem</code></summary>

```
/-- `A[i, j]`, the intra-chunk attention matrix the other four kernels built, at base
`A + i_bh·T·BT` with parent shape `(T, BT)` and strides `(BT, 1)`. -/
```
```lean
noncomputable def aElem (s : BlockState) (A : RegionName) (T BT : Nat)
    (i j : Nat) : ℝ :=
  s.readMem A (s.pids 2 * T * BT + tIndex s BT i * BT + j * 1)
```
</details>

<details><summary><code>vElem</code></summary>

```
/-- `v[j, vIndex p]`. -/
```
```lean
noncomputable def vElem (s : BlockState) (v : RegionName) (s_v_h s_v_t BT BV : Nat)
    (j p : Nat) : ℝ :=
  s.readMem v (s.pids 2 * s_v_h + tIndex s BT j * s_v_t + vIndex s BV p * 1)
```
</details>
