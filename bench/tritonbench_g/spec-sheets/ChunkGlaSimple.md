# Spec sheet — `bench/tritonbench_g/chunk_gla_simple/ChunkGlaSimple.lean`

**Python source:** `bench/tritonbench_g/chunk_gla_simple/chunk_gla_simple.py`

## Public theorem: `chunk_gla_simple_python_case1_output_summary`

<details><summary>docstring</summary>

```
/-- Public Python case 1 summary (`B=2,H=4,T=128,K=BK=64,V=64,BT=32`, `scale=0.1`). -/
```
</details>

**Statement:**
```lean
theorem chunk_gla_simple_python_case1_output_summary
    (q k v h g o : RegionName) (s : BlockState)
    (hundef : ∀ rg off, s.undef rg off = 0) :
    (∃ alg, (chunk_gla_simple_fwd_surface q k v h g o
      8192 64 8192 64 4096 64 (0.1 : ℝ) 128 64 64 32 64 64).toAlgorithm? = Except.ok alg) ∧
    (∀ idx : TileIndex [32, 64], active s 128 64 32 64 idx →
      (exec (chunk_gla_simple_fwd_surface q k v h g o
          8192 64 8192 64 4096 64 (0.1 : ℝ) 128 64 64 32 64 64) s).map
          (·.readMem o (outOffset s 8192 64 32 64 idx))
        = some (glaOutput s q k v h g 8192 64 8192 64 4096 64 (0.1 : ℝ)
            128 64 64 32 64 64 idx.1 idx.2.1))
```

**Assumptions / layout contracts:**
- `hundef : ∀ rg off, s.undef rg off = 0`

**Closed-form spec defs (transitive):** `chunk_gla_simple_fwd_surface`, `active`, `outOffset`, `glaOutput`, `tIndex`, `vIndex`, `interTerm`, `scoreTerm`, `vElem`, `qElem`, `hElem`, `gElem`, `kElem`

<details><summary><code>chunk_gla_simple_fwd_surface</code></summary>

```
/-- Faithful transcription of `chunk_gla_simple.py`'s
`chunk_simple_gla_fwd_kernel_o`. -/
```
```lean
def chunk_gla_simple_fwd_surface
    (q k v h g o : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV : Nat) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_t = tl.program_id(1)
  i_bh = tl.program_id(2)
  o_i = tl.arange(0, $(BT))
  m_s = o_i[:, None] >= o_i[None, :]
  b_o = tl.zeros([$(BT), $(BV)], dtype=tl.float32)
  b_s = tl.zeros([$(BT), $(BT)], dtype=tl.float32)
  for i_k in range($(0), tl.cdiv($(K), $(BK)), $(1)) {
    p_q = tl.make_block_ptr(base=q + i_bh * $(s_k_h),
      shape=($(T), $(K)), strides=($(s_k_t), $(1)),
      offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
    p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h),
      shape=($(K), $(T)), strides=($(1), $(s_k_t)),
      offsets=(i_k * $(BK), i_t * $(BT)), block_shape=($(BK), $(BT)), order=(0, 1))
    p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V),
      shape=($(K), $(V)), strides=($(s_h_t), $(1)),
      offsets=(i_k * $(BK), i_v * $(BV)), block_shape=($(BK), $(BV)), order=(1, 0))
    b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat))
    b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
    b_h = tl.load(p_h, boundary_check=([0, 1] : List Nat))
    b_o += tl.dot(b_q, b_h, allow_tf32=false)
    b_s += tl.dot(b_q, b_k, allow_tf32=false)
  }
  p_g = tl.make_block_ptr(base=g + i_bh * $(T), shape=($(T)),
    strides=($(1)), offsets=(i_t * $(BT)), block_shape=($(BT)), order=(0))
  b_g = tl.load(p_g, boundary_check=([0] : List Nat))
  b_o = b_o * tl.exp(b_g)[:, None]
  b_s = b_s * tl.exp(b_g[:, None] - b_g[None, :])
  b_s = tl.where(m_s, b_s, 0.0)
  p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h),
    shape=($(T), $(V)), strides=($(s_v_t), $(1)),
    offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
  b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
  b_o = (b_o + tl.dot((b_s).to(b_v.dtype), b_v, allow_tf32=false)) * $(scale)
  p_o = tl.make_block_ptr(base=o + i_bh * $(s_v_h),
    shape=($(T), $(V)), strides=($(s_v_t), $(1)),
    offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
  tl.store(p_o, (b_o).to(p_o.dtype.element_ty), boundary_check=([0, 1] : List Nat))
}
```
</details>

<details><summary><code>active</code></summary>

```
/-- A tile lane is *active* when it maps inside the `T × V` output window. -/
```
```lean
def active (s : BlockState) (T V BT BV : Nat) (idx : TileIndex [BT, BV]) : Prop :=
  tIndex s BT idx.1 < T ∧ vIndex s BV idx.2.1 < V
```
</details>

<details><summary><code>outOffset</code></summary>

```
/-- The output store address for lane `(i, p)`:
`i_bh·s_v_h + (i_t·BT + i)·s_v_t + (i_v·BV + p)`. -/
```
```lean
def outOffset (s : BlockState) (s_v_h s_v_t BT BV : Nat)
    (idx : TileIndex [BT, BV]) : Nat :=
  s.pids 2 * s_v_h + tIndex s BT idx.1 * s_v_t + vIndex s BV idx.2.1 * 1
```
</details>

<details><summary><code>glaOutput</code></summary>

```
/-- **Genuine GLA output closed form** for lane `(i, p)`. -/
```
```lean
noncomputable def glaOutput
    (s : BlockState) (q k v h g : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV : Nat) (i : Fin BT) (p : Fin BV) : ℝ :=
  (interTerm s q h s_k_h s_k_t s_h_h s_h_t T K V BT BV BK g i p
    + Finset.univ.sum fun j : Fin BT =>
        scoreTerm s q k s_k_h s_k_t T BT BK g i j
          * vElem s v s_v_h s_v_t BT BV j p) * scale
```
</details>

<details><summary><code>tIndex</code></summary>

```
/-- Global time (row) index of tile lane `i`: `i_t · BT + i`. -/
```
```lean
def tIndex (s : BlockState) (BT : Nat) (i : Fin BT) : Nat := s.pids 1 * BT + i.val
```
</details>

<details><summary><code>vIndex</code></summary>

```
/-- Global value (column) index of tile lane `p`: `i_v · BV + p`. -/
```
```lean
def vIndex (s : BlockState) (BV : Nat) (p : Fin BV) : Nat := s.pids 0 * BV + p.val
```
</details>

<details><summary><code>interTerm</code></summary>

```
/-- Inter-chunk term lane `(i,p)`: `(Σ_e q[i,e]·h[e,p]) · exp(g_i)`. -/
```
```lean
noncomputable def interTerm (s : BlockState) (q h : RegionName)
    (s_k_h s_k_t s_h_h s_h_t : Nat) (T K V BT BV BK : Nat)
    (g : RegionName) (i : Fin BT) (p : Fin BV) : ℝ :=
  (Finset.univ.sum fun e : Fin BK => qElem s q s_k_h s_k_t BT i e.val
      * hElem s h s_h_h s_h_t K V BV p e.val)
    * Real.exp (gElem s g T BT i)
```
</details>

<details><summary><code>scoreTerm</code></summary>

```
/-- Masked, decayed score lane `(i,j)`: `if i ≥ j then (Σ_e q·k) · exp(g_i−g_j) else 0`. -/
```
```lean
noncomputable def scoreTerm (s : BlockState) (q k : RegionName)
    (s_k_h s_k_t : Nat) (T BT BK : Nat)
    (g : RegionName) (i j : Fin BT) : ℝ :=
  if (j.val ≤ i.val) then
    (Finset.univ.sum fun e : Fin BK => qElem s q s_k_h s_k_t BT i e.val
        * kElem s k s_k_h s_k_t BT j e.val)
      * Real.exp (gElem s g T BT i - gElem s g T BT j)
  else 0
```
</details>

<details><summary><code>vElem</code></summary>

```
/-- `v[j, p]` element: `v` at `i_bh·s_v_h + (i_t·BT + j)·s_v_t + (i_v·BV + p)`. -/
```
```lean
noncomputable def vElem (s : BlockState) (v : RegionName) (s_v_h s_v_t BT BV : Nat)
    (j : Fin BT) (p : Fin BV) : ℝ :=
  s.readMem v (s.pids 2 * s_v_h + (s.pids 1 * BT + j.val) * s_v_t + (s.pids 0 * BV + p.val) * 1)
```
</details>

<details><summary><code>qElem</code></summary>

```
/-- `q[i, e]` element: `q` at `i_bh·s_k_h + (i_t·BT + i)·s_k_t + e`. -/
```
```lean
noncomputable def qElem (s : BlockState) (q : RegionName) (s_k_h s_k_t BT : Nat)
    (i : Fin BT) (e : Nat) : ℝ :=
  s.readMem q (s.pids 2 * s_k_h + (s.pids 1 * BT + i.val) * s_k_t + e)
```
</details>

<details><summary><code>hElem</code></summary>

```
/-- `h[e, p]` element (chunk state, base `h + i_bh·s_h_h + i_t·K·V`):
`h` at `i_bh·s_h_h + i_t·K·V + e·s_h_t + (i_v·BV + p)`. -/
```
```lean
noncomputable def hElem (s : BlockState) (h : RegionName) (s_h_h s_h_t K V BV : Nat)
    (p : Fin BV) (e : Nat) : ℝ :=
  s.readMem h (s.pids 2 * s_h_h + s.pids 1 * K * V + e * s_h_t + (s.pids 0 * BV + p.val) * 1)
```
</details>

<details><summary><code>gElem</code></summary>

```
/-- `g[i]` element (gate): `g` at `i_bh·T + (i_t·BT + i)`. -/
```
```lean
noncomputable def gElem (s : BlockState) (g : RegionName) (T BT : Nat) (i : Fin BT) : ℝ :=
  s.readMem g (s.pids 2 * T + (s.pids 1 * BT + i.val) * 1)
```
</details>

<details><summary><code>kElem</code></summary>

```
/-- `k[e, j]` element (block ptr layout `(K,T)` strides `(1, s_k_t)`):
`k` at `i_bh·s_k_h + e + (i_t·BT + j)·s_k_t`. -/
```
```lean
noncomputable def kElem (s : BlockState) (k : RegionName) (s_k_h s_k_t BT : Nat)
    (j : Fin BT) (e : Nat) : ℝ :=
  s.readMem k (s.pids 2 * s_k_h + e * 1 + (s.pids 1 * BT + j.val) * s_k_t)
```
</details>

## Public theorem: `chunk_gla_simple_python_case2_output_summary`

<details><summary>docstring</summary>

```
/-- Public Python case 2 summary (`BT=64`, `scale=0.1`). -/
```
</details>

**Statement:**
```lean
theorem chunk_gla_simple_python_case2_output_summary
    (q k v h g o : RegionName) (s : BlockState)
    (hundef : ∀ rg off, s.undef rg off = 0) :
    (∃ alg, (chunk_gla_simple_fwd_surface q k v h g o
      8192 64 8192 64 4096 64 (0.1 : ℝ) 128 64 64 64 64 64).toAlgorithm? = Except.ok alg) ∧
    (∀ idx : TileIndex [64, 64], active s 128 64 64 64 idx →
      (exec (chunk_gla_simple_fwd_surface q k v h g o
          8192 64 8192 64 4096 64 (0.1 : ℝ) 128 64 64 64 64 64) s).map
          (·.readMem o (outOffset s 8192 64 64 64 idx))
        = some (glaOutput s q k v h g 8192 64 8192 64 4096 64 (0.1 : ℝ)
            128 64 64 64 64 64 idx.1 idx.2.1))
```

**Assumptions / layout contracts:**
- `hundef : ∀ rg off, s.undef rg off = 0`

**Closed-form spec defs (transitive):** `chunk_gla_simple_fwd_surface`, `active`, `outOffset`, `glaOutput`, `tIndex`, `vIndex`, `interTerm`, `scoreTerm`, `vElem`, `qElem`, `hElem`, `gElem`, `kElem`

<details><summary><code>chunk_gla_simple_fwd_surface</code></summary>

```
/-- Faithful transcription of `chunk_gla_simple.py`'s
`chunk_simple_gla_fwd_kernel_o`. -/
```
```lean
def chunk_gla_simple_fwd_surface
    (q k v h g o : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV : Nat) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_t = tl.program_id(1)
  i_bh = tl.program_id(2)
  o_i = tl.arange(0, $(BT))
  m_s = o_i[:, None] >= o_i[None, :]
  b_o = tl.zeros([$(BT), $(BV)], dtype=tl.float32)
  b_s = tl.zeros([$(BT), $(BT)], dtype=tl.float32)
  for i_k in range($(0), tl.cdiv($(K), $(BK)), $(1)) {
    p_q = tl.make_block_ptr(base=q + i_bh * $(s_k_h),
      shape=($(T), $(K)), strides=($(s_k_t), $(1)),
      offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
    p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h),
      shape=($(K), $(T)), strides=($(1), $(s_k_t)),
      offsets=(i_k * $(BK), i_t * $(BT)), block_shape=($(BK), $(BT)), order=(0, 1))
    p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V),
      shape=($(K), $(V)), strides=($(s_h_t), $(1)),
      offsets=(i_k * $(BK), i_v * $(BV)), block_shape=($(BK), $(BV)), order=(1, 0))
    b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat))
    b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
    b_h = tl.load(p_h, boundary_check=([0, 1] : List Nat))
    b_o += tl.dot(b_q, b_h, allow_tf32=false)
    b_s += tl.dot(b_q, b_k, allow_tf32=false)
  }
  p_g = tl.make_block_ptr(base=g + i_bh * $(T), shape=($(T)),
    strides=($(1)), offsets=(i_t * $(BT)), block_shape=($(BT)), order=(0))
  b_g = tl.load(p_g, boundary_check=([0] : List Nat))
  b_o = b_o * tl.exp(b_g)[:, None]
  b_s = b_s * tl.exp(b_g[:, None] - b_g[None, :])
  b_s = tl.where(m_s, b_s, 0.0)
  p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h),
    shape=($(T), $(V)), strides=($(s_v_t), $(1)),
    offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
  b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
  b_o = (b_o + tl.dot((b_s).to(b_v.dtype), b_v, allow_tf32=false)) * $(scale)
  p_o = tl.make_block_ptr(base=o + i_bh * $(s_v_h),
    shape=($(T), $(V)), strides=($(s_v_t), $(1)),
    offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
  tl.store(p_o, (b_o).to(p_o.dtype.element_ty), boundary_check=([0, 1] : List Nat))
}
```
</details>

<details><summary><code>active</code></summary>

```
/-- A tile lane is *active* when it maps inside the `T × V` output window. -/
```
```lean
def active (s : BlockState) (T V BT BV : Nat) (idx : TileIndex [BT, BV]) : Prop :=
  tIndex s BT idx.1 < T ∧ vIndex s BV idx.2.1 < V
```
</details>

<details><summary><code>outOffset</code></summary>

```
/-- The output store address for lane `(i, p)`:
`i_bh·s_v_h + (i_t·BT + i)·s_v_t + (i_v·BV + p)`. -/
```
```lean
def outOffset (s : BlockState) (s_v_h s_v_t BT BV : Nat)
    (idx : TileIndex [BT, BV]) : Nat :=
  s.pids 2 * s_v_h + tIndex s BT idx.1 * s_v_t + vIndex s BV idx.2.1 * 1
```
</details>

<details><summary><code>glaOutput</code></summary>

```
/-- **Genuine GLA output closed form** for lane `(i, p)`. -/
```
```lean
noncomputable def glaOutput
    (s : BlockState) (q k v h g : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV : Nat) (i : Fin BT) (p : Fin BV) : ℝ :=
  (interTerm s q h s_k_h s_k_t s_h_h s_h_t T K V BT BV BK g i p
    + Finset.univ.sum fun j : Fin BT =>
        scoreTerm s q k s_k_h s_k_t T BT BK g i j
          * vElem s v s_v_h s_v_t BT BV j p) * scale
```
</details>

<details><summary><code>tIndex</code></summary>

```
/-- Global time (row) index of tile lane `i`: `i_t · BT + i`. -/
```
```lean
def tIndex (s : BlockState) (BT : Nat) (i : Fin BT) : Nat := s.pids 1 * BT + i.val
```
</details>

<details><summary><code>vIndex</code></summary>

```
/-- Global value (column) index of tile lane `p`: `i_v · BV + p`. -/
```
```lean
def vIndex (s : BlockState) (BV : Nat) (p : Fin BV) : Nat := s.pids 0 * BV + p.val
```
</details>

<details><summary><code>interTerm</code></summary>

```
/-- Inter-chunk term lane `(i,p)`: `(Σ_e q[i,e]·h[e,p]) · exp(g_i)`. -/
```
```lean
noncomputable def interTerm (s : BlockState) (q h : RegionName)
    (s_k_h s_k_t s_h_h s_h_t : Nat) (T K V BT BV BK : Nat)
    (g : RegionName) (i : Fin BT) (p : Fin BV) : ℝ :=
  (Finset.univ.sum fun e : Fin BK => qElem s q s_k_h s_k_t BT i e.val
      * hElem s h s_h_h s_h_t K V BV p e.val)
    * Real.exp (gElem s g T BT i)
```
</details>

<details><summary><code>scoreTerm</code></summary>

```
/-- Masked, decayed score lane `(i,j)`: `if i ≥ j then (Σ_e q·k) · exp(g_i−g_j) else 0`. -/
```
```lean
noncomputable def scoreTerm (s : BlockState) (q k : RegionName)
    (s_k_h s_k_t : Nat) (T BT BK : Nat)
    (g : RegionName) (i j : Fin BT) : ℝ :=
  if (j.val ≤ i.val) then
    (Finset.univ.sum fun e : Fin BK => qElem s q s_k_h s_k_t BT i e.val
        * kElem s k s_k_h s_k_t BT j e.val)
      * Real.exp (gElem s g T BT i - gElem s g T BT j)
  else 0
```
</details>

<details><summary><code>vElem</code></summary>

```
/-- `v[j, p]` element: `v` at `i_bh·s_v_h + (i_t·BT + j)·s_v_t + (i_v·BV + p)`. -/
```
```lean
noncomputable def vElem (s : BlockState) (v : RegionName) (s_v_h s_v_t BT BV : Nat)
    (j : Fin BT) (p : Fin BV) : ℝ :=
  s.readMem v (s.pids 2 * s_v_h + (s.pids 1 * BT + j.val) * s_v_t + (s.pids 0 * BV + p.val) * 1)
```
</details>

<details><summary><code>qElem</code></summary>

```
/-- `q[i, e]` element: `q` at `i_bh·s_k_h + (i_t·BT + i)·s_k_t + e`. -/
```
```lean
noncomputable def qElem (s : BlockState) (q : RegionName) (s_k_h s_k_t BT : Nat)
    (i : Fin BT) (e : Nat) : ℝ :=
  s.readMem q (s.pids 2 * s_k_h + (s.pids 1 * BT + i.val) * s_k_t + e)
```
</details>

<details><summary><code>hElem</code></summary>

```
/-- `h[e, p]` element (chunk state, base `h + i_bh·s_h_h + i_t·K·V`):
`h` at `i_bh·s_h_h + i_t·K·V + e·s_h_t + (i_v·BV + p)`. -/
```
```lean
noncomputable def hElem (s : BlockState) (h : RegionName) (s_h_h s_h_t K V BV : Nat)
    (p : Fin BV) (e : Nat) : ℝ :=
  s.readMem h (s.pids 2 * s_h_h + s.pids 1 * K * V + e * s_h_t + (s.pids 0 * BV + p.val) * 1)
```
</details>

<details><summary><code>gElem</code></summary>

```
/-- `g[i]` element (gate): `g` at `i_bh·T + (i_t·BT + i)`. -/
```
```lean
noncomputable def gElem (s : BlockState) (g : RegionName) (T BT : Nat) (i : Fin BT) : ℝ :=
  s.readMem g (s.pids 2 * T + (s.pids 1 * BT + i.val) * 1)
```
</details>

<details><summary><code>kElem</code></summary>

```
/-- `k[e, j]` element (block ptr layout `(K,T)` strides `(1, s_k_t)`):
`k` at `i_bh·s_k_h + e + (i_t·BT + j)·s_k_t`. -/
```
```lean
noncomputable def kElem (s : BlockState) (k : RegionName) (s_k_h s_k_t BT : Nat)
    (j : Fin BT) (e : Nat) : ℝ :=
  s.readMem k (s.pids 2 * s_k_h + e * 1 + (s.pids 1 * BT + j.val) * s_k_t)
```
</details>

## Public theorem: `chunk_gla_simple_python_case3_output_summary`

<details><summary>docstring</summary>

```
/-- Public Python case 3 summary (`BT=64`, `scale=0.2`). -/
```
</details>

**Statement:**
```lean
theorem chunk_gla_simple_python_case3_output_summary
    (q k v h g o : RegionName) (s : BlockState)
    (hundef : ∀ rg off, s.undef rg off = 0) :
    (∃ alg, (chunk_gla_simple_fwd_surface q k v h g o
      8192 64 8192 64 4096 64 (0.2 : ℝ) 128 64 64 64 64 64).toAlgorithm? = Except.ok alg) ∧
    (∀ idx : TileIndex [64, 64], active s 128 64 64 64 idx →
      (exec (chunk_gla_simple_fwd_surface q k v h g o
          8192 64 8192 64 4096 64 (0.2 : ℝ) 128 64 64 64 64 64) s).map
          (·.readMem o (outOffset s 8192 64 64 64 idx))
        = some (glaOutput s q k v h g 8192 64 8192 64 4096 64 (0.2 : ℝ)
            128 64 64 64 64 64 idx.1 idx.2.1))
```

**Assumptions / layout contracts:**
- `hundef : ∀ rg off, s.undef rg off = 0`

**Closed-form spec defs (transitive):** `chunk_gla_simple_fwd_surface`, `active`, `outOffset`, `glaOutput`, `tIndex`, `vIndex`, `interTerm`, `scoreTerm`, `vElem`, `qElem`, `hElem`, `gElem`, `kElem`

<details><summary><code>chunk_gla_simple_fwd_surface</code></summary>

```
/-- Faithful transcription of `chunk_gla_simple.py`'s
`chunk_simple_gla_fwd_kernel_o`. -/
```
```lean
def chunk_gla_simple_fwd_surface
    (q k v h g o : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV : Nat) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_t = tl.program_id(1)
  i_bh = tl.program_id(2)
  o_i = tl.arange(0, $(BT))
  m_s = o_i[:, None] >= o_i[None, :]
  b_o = tl.zeros([$(BT), $(BV)], dtype=tl.float32)
  b_s = tl.zeros([$(BT), $(BT)], dtype=tl.float32)
  for i_k in range($(0), tl.cdiv($(K), $(BK)), $(1)) {
    p_q = tl.make_block_ptr(base=q + i_bh * $(s_k_h),
      shape=($(T), $(K)), strides=($(s_k_t), $(1)),
      offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
    p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h),
      shape=($(K), $(T)), strides=($(1), $(s_k_t)),
      offsets=(i_k * $(BK), i_t * $(BT)), block_shape=($(BK), $(BT)), order=(0, 1))
    p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V),
      shape=($(K), $(V)), strides=($(s_h_t), $(1)),
      offsets=(i_k * $(BK), i_v * $(BV)), block_shape=($(BK), $(BV)), order=(1, 0))
    b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat))
    b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
    b_h = tl.load(p_h, boundary_check=([0, 1] : List Nat))
    b_o += tl.dot(b_q, b_h, allow_tf32=false)
    b_s += tl.dot(b_q, b_k, allow_tf32=false)
  }
  p_g = tl.make_block_ptr(base=g + i_bh * $(T), shape=($(T)),
    strides=($(1)), offsets=(i_t * $(BT)), block_shape=($(BT)), order=(0))
  b_g = tl.load(p_g, boundary_check=([0] : List Nat))
  b_o = b_o * tl.exp(b_g)[:, None]
  b_s = b_s * tl.exp(b_g[:, None] - b_g[None, :])
  b_s = tl.where(m_s, b_s, 0.0)
  p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h),
    shape=($(T), $(V)), strides=($(s_v_t), $(1)),
    offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
  b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
  b_o = (b_o + tl.dot((b_s).to(b_v.dtype), b_v, allow_tf32=false)) * $(scale)
  p_o = tl.make_block_ptr(base=o + i_bh * $(s_v_h),
    shape=($(T), $(V)), strides=($(s_v_t), $(1)),
    offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
  tl.store(p_o, (b_o).to(p_o.dtype.element_ty), boundary_check=([0, 1] : List Nat))
}
```
</details>

<details><summary><code>active</code></summary>

```
/-- A tile lane is *active* when it maps inside the `T × V` output window. -/
```
```lean
def active (s : BlockState) (T V BT BV : Nat) (idx : TileIndex [BT, BV]) : Prop :=
  tIndex s BT idx.1 < T ∧ vIndex s BV idx.2.1 < V
```
</details>

<details><summary><code>outOffset</code></summary>

```
/-- The output store address for lane `(i, p)`:
`i_bh·s_v_h + (i_t·BT + i)·s_v_t + (i_v·BV + p)`. -/
```
```lean
def outOffset (s : BlockState) (s_v_h s_v_t BT BV : Nat)
    (idx : TileIndex [BT, BV]) : Nat :=
  s.pids 2 * s_v_h + tIndex s BT idx.1 * s_v_t + vIndex s BV idx.2.1 * 1
```
</details>

<details><summary><code>glaOutput</code></summary>

```
/-- **Genuine GLA output closed form** for lane `(i, p)`. -/
```
```lean
noncomputable def glaOutput
    (s : BlockState) (q k v h g : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV : Nat) (i : Fin BT) (p : Fin BV) : ℝ :=
  (interTerm s q h s_k_h s_k_t s_h_h s_h_t T K V BT BV BK g i p
    + Finset.univ.sum fun j : Fin BT =>
        scoreTerm s q k s_k_h s_k_t T BT BK g i j
          * vElem s v s_v_h s_v_t BT BV j p) * scale
```
</details>

<details><summary><code>tIndex</code></summary>

```
/-- Global time (row) index of tile lane `i`: `i_t · BT + i`. -/
```
```lean
def tIndex (s : BlockState) (BT : Nat) (i : Fin BT) : Nat := s.pids 1 * BT + i.val
```
</details>

<details><summary><code>vIndex</code></summary>

```
/-- Global value (column) index of tile lane `p`: `i_v · BV + p`. -/
```
```lean
def vIndex (s : BlockState) (BV : Nat) (p : Fin BV) : Nat := s.pids 0 * BV + p.val
```
</details>

<details><summary><code>interTerm</code></summary>

```
/-- Inter-chunk term lane `(i,p)`: `(Σ_e q[i,e]·h[e,p]) · exp(g_i)`. -/
```
```lean
noncomputable def interTerm (s : BlockState) (q h : RegionName)
    (s_k_h s_k_t s_h_h s_h_t : Nat) (T K V BT BV BK : Nat)
    (g : RegionName) (i : Fin BT) (p : Fin BV) : ℝ :=
  (Finset.univ.sum fun e : Fin BK => qElem s q s_k_h s_k_t BT i e.val
      * hElem s h s_h_h s_h_t K V BV p e.val)
    * Real.exp (gElem s g T BT i)
```
</details>

<details><summary><code>scoreTerm</code></summary>

```
/-- Masked, decayed score lane `(i,j)`: `if i ≥ j then (Σ_e q·k) · exp(g_i−g_j) else 0`. -/
```
```lean
noncomputable def scoreTerm (s : BlockState) (q k : RegionName)
    (s_k_h s_k_t : Nat) (T BT BK : Nat)
    (g : RegionName) (i j : Fin BT) : ℝ :=
  if (j.val ≤ i.val) then
    (Finset.univ.sum fun e : Fin BK => qElem s q s_k_h s_k_t BT i e.val
        * kElem s k s_k_h s_k_t BT j e.val)
      * Real.exp (gElem s g T BT i - gElem s g T BT j)
  else 0
```
</details>

<details><summary><code>vElem</code></summary>

```
/-- `v[j, p]` element: `v` at `i_bh·s_v_h + (i_t·BT + j)·s_v_t + (i_v·BV + p)`. -/
```
```lean
noncomputable def vElem (s : BlockState) (v : RegionName) (s_v_h s_v_t BT BV : Nat)
    (j : Fin BT) (p : Fin BV) : ℝ :=
  s.readMem v (s.pids 2 * s_v_h + (s.pids 1 * BT + j.val) * s_v_t + (s.pids 0 * BV + p.val) * 1)
```
</details>

<details><summary><code>qElem</code></summary>

```
/-- `q[i, e]` element: `q` at `i_bh·s_k_h + (i_t·BT + i)·s_k_t + e`. -/
```
```lean
noncomputable def qElem (s : BlockState) (q : RegionName) (s_k_h s_k_t BT : Nat)
    (i : Fin BT) (e : Nat) : ℝ :=
  s.readMem q (s.pids 2 * s_k_h + (s.pids 1 * BT + i.val) * s_k_t + e)
```
</details>

<details><summary><code>hElem</code></summary>

```
/-- `h[e, p]` element (chunk state, base `h + i_bh·s_h_h + i_t·K·V`):
`h` at `i_bh·s_h_h + i_t·K·V + e·s_h_t + (i_v·BV + p)`. -/
```
```lean
noncomputable def hElem (s : BlockState) (h : RegionName) (s_h_h s_h_t K V BV : Nat)
    (p : Fin BV) (e : Nat) : ℝ :=
  s.readMem h (s.pids 2 * s_h_h + s.pids 1 * K * V + e * s_h_t + (s.pids 0 * BV + p.val) * 1)
```
</details>

<details><summary><code>gElem</code></summary>

```
/-- `g[i]` element (gate): `g` at `i_bh·T + (i_t·BT + i)`. -/
```
```lean
noncomputable def gElem (s : BlockState) (g : RegionName) (T BT : Nat) (i : Fin BT) : ℝ :=
  s.readMem g (s.pids 2 * T + (s.pids 1 * BT + i.val) * 1)
```
</details>

<details><summary><code>kElem</code></summary>

```
/-- `k[e, j]` element (block ptr layout `(K,T)` strides `(1, s_k_t)`):
`k` at `i_bh·s_k_h + e + (i_t·BT + j)·s_k_t`. -/
```
```lean
noncomputable def kElem (s : BlockState) (k : RegionName) (s_k_h s_k_t BT : Nat)
    (j : Fin BT) (e : Nat) : ℝ :=
  s.readMem k (s.pids 2 * s_k_h + e * 1 + (s.pids 1 * BT + j.val) * s_k_t)
```
</details>

## Public theorem: `chunk_gla_simple_python_case4_output_summary`

<details><summary>docstring</summary>

```
/-- Public Python case 4 summary (`B=1,H=2,T=64,K=BK=32,V=32,BT=64`, `scale=0.2`). -/
```
</details>

**Statement:**
```lean
theorem chunk_gla_simple_python_case4_output_summary
    (q k v h g o : RegionName) (s : BlockState)
    (hundef : ∀ rg off, s.undef rg off = 0) :
    (∃ alg, (chunk_gla_simple_fwd_surface q k v h g o
      2048 32 2048 32 1024 32 (0.2 : ℝ) 64 32 32 64 32 32).toAlgorithm? = Except.ok alg) ∧
    (∀ idx : TileIndex [64, 32], active s 64 32 64 32 idx →
      (exec (chunk_gla_simple_fwd_surface q k v h g o
          2048 32 2048 32 1024 32 (0.2 : ℝ) 64 32 32 64 32 32) s).map
          (·.readMem o (outOffset s 2048 32 64 32 idx))
        = some (glaOutput s q k v h g 2048 32 2048 32 1024 32 (0.2 : ℝ)
            64 32 32 64 32 32 idx.1 idx.2.1))
```

**Assumptions / layout contracts:**
- `hundef : ∀ rg off, s.undef rg off = 0`

**Closed-form spec defs (transitive):** `chunk_gla_simple_fwd_surface`, `active`, `outOffset`, `glaOutput`, `tIndex`, `vIndex`, `interTerm`, `scoreTerm`, `vElem`, `qElem`, `hElem`, `gElem`, `kElem`

<details><summary><code>chunk_gla_simple_fwd_surface</code></summary>

```
/-- Faithful transcription of `chunk_gla_simple.py`'s
`chunk_simple_gla_fwd_kernel_o`. -/
```
```lean
def chunk_gla_simple_fwd_surface
    (q k v h g o : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV : Nat) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_t = tl.program_id(1)
  i_bh = tl.program_id(2)
  o_i = tl.arange(0, $(BT))
  m_s = o_i[:, None] >= o_i[None, :]
  b_o = tl.zeros([$(BT), $(BV)], dtype=tl.float32)
  b_s = tl.zeros([$(BT), $(BT)], dtype=tl.float32)
  for i_k in range($(0), tl.cdiv($(K), $(BK)), $(1)) {
    p_q = tl.make_block_ptr(base=q + i_bh * $(s_k_h),
      shape=($(T), $(K)), strides=($(s_k_t), $(1)),
      offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
    p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h),
      shape=($(K), $(T)), strides=($(1), $(s_k_t)),
      offsets=(i_k * $(BK), i_t * $(BT)), block_shape=($(BK), $(BT)), order=(0, 1))
    p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V),
      shape=($(K), $(V)), strides=($(s_h_t), $(1)),
      offsets=(i_k * $(BK), i_v * $(BV)), block_shape=($(BK), $(BV)), order=(1, 0))
    b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat))
    b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
    b_h = tl.load(p_h, boundary_check=([0, 1] : List Nat))
    b_o += tl.dot(b_q, b_h, allow_tf32=false)
    b_s += tl.dot(b_q, b_k, allow_tf32=false)
  }
  p_g = tl.make_block_ptr(base=g + i_bh * $(T), shape=($(T)),
    strides=($(1)), offsets=(i_t * $(BT)), block_shape=($(BT)), order=(0))
  b_g = tl.load(p_g, boundary_check=([0] : List Nat))
  b_o = b_o * tl.exp(b_g)[:, None]
  b_s = b_s * tl.exp(b_g[:, None] - b_g[None, :])
  b_s = tl.where(m_s, b_s, 0.0)
  p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h),
    shape=($(T), $(V)), strides=($(s_v_t), $(1)),
    offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
  b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
  b_o = (b_o + tl.dot((b_s).to(b_v.dtype), b_v, allow_tf32=false)) * $(scale)
  p_o = tl.make_block_ptr(base=o + i_bh * $(s_v_h),
    shape=($(T), $(V)), strides=($(s_v_t), $(1)),
    offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
  tl.store(p_o, (b_o).to(p_o.dtype.element_ty), boundary_check=([0, 1] : List Nat))
}
```
</details>

<details><summary><code>active</code></summary>

```
/-- A tile lane is *active* when it maps inside the `T × V` output window. -/
```
```lean
def active (s : BlockState) (T V BT BV : Nat) (idx : TileIndex [BT, BV]) : Prop :=
  tIndex s BT idx.1 < T ∧ vIndex s BV idx.2.1 < V
```
</details>

<details><summary><code>outOffset</code></summary>

```
/-- The output store address for lane `(i, p)`:
`i_bh·s_v_h + (i_t·BT + i)·s_v_t + (i_v·BV + p)`. -/
```
```lean
def outOffset (s : BlockState) (s_v_h s_v_t BT BV : Nat)
    (idx : TileIndex [BT, BV]) : Nat :=
  s.pids 2 * s_v_h + tIndex s BT idx.1 * s_v_t + vIndex s BV idx.2.1 * 1
```
</details>

<details><summary><code>glaOutput</code></summary>

```
/-- **Genuine GLA output closed form** for lane `(i, p)`. -/
```
```lean
noncomputable def glaOutput
    (s : BlockState) (q k v h g : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV : Nat) (i : Fin BT) (p : Fin BV) : ℝ :=
  (interTerm s q h s_k_h s_k_t s_h_h s_h_t T K V BT BV BK g i p
    + Finset.univ.sum fun j : Fin BT =>
        scoreTerm s q k s_k_h s_k_t T BT BK g i j
          * vElem s v s_v_h s_v_t BT BV j p) * scale
```
</details>

<details><summary><code>tIndex</code></summary>

```
/-- Global time (row) index of tile lane `i`: `i_t · BT + i`. -/
```
```lean
def tIndex (s : BlockState) (BT : Nat) (i : Fin BT) : Nat := s.pids 1 * BT + i.val
```
</details>

<details><summary><code>vIndex</code></summary>

```
/-- Global value (column) index of tile lane `p`: `i_v · BV + p`. -/
```
```lean
def vIndex (s : BlockState) (BV : Nat) (p : Fin BV) : Nat := s.pids 0 * BV + p.val
```
</details>

<details><summary><code>interTerm</code></summary>

```
/-- Inter-chunk term lane `(i,p)`: `(Σ_e q[i,e]·h[e,p]) · exp(g_i)`. -/
```
```lean
noncomputable def interTerm (s : BlockState) (q h : RegionName)
    (s_k_h s_k_t s_h_h s_h_t : Nat) (T K V BT BV BK : Nat)
    (g : RegionName) (i : Fin BT) (p : Fin BV) : ℝ :=
  (Finset.univ.sum fun e : Fin BK => qElem s q s_k_h s_k_t BT i e.val
      * hElem s h s_h_h s_h_t K V BV p e.val)
    * Real.exp (gElem s g T BT i)
```
</details>

<details><summary><code>scoreTerm</code></summary>

```
/-- Masked, decayed score lane `(i,j)`: `if i ≥ j then (Σ_e q·k) · exp(g_i−g_j) else 0`. -/
```
```lean
noncomputable def scoreTerm (s : BlockState) (q k : RegionName)
    (s_k_h s_k_t : Nat) (T BT BK : Nat)
    (g : RegionName) (i j : Fin BT) : ℝ :=
  if (j.val ≤ i.val) then
    (Finset.univ.sum fun e : Fin BK => qElem s q s_k_h s_k_t BT i e.val
        * kElem s k s_k_h s_k_t BT j e.val)
      * Real.exp (gElem s g T BT i - gElem s g T BT j)
  else 0
```
</details>

<details><summary><code>vElem</code></summary>

```
/-- `v[j, p]` element: `v` at `i_bh·s_v_h + (i_t·BT + j)·s_v_t + (i_v·BV + p)`. -/
```
```lean
noncomputable def vElem (s : BlockState) (v : RegionName) (s_v_h s_v_t BT BV : Nat)
    (j : Fin BT) (p : Fin BV) : ℝ :=
  s.readMem v (s.pids 2 * s_v_h + (s.pids 1 * BT + j.val) * s_v_t + (s.pids 0 * BV + p.val) * 1)
```
</details>

<details><summary><code>qElem</code></summary>

```
/-- `q[i, e]` element: `q` at `i_bh·s_k_h + (i_t·BT + i)·s_k_t + e`. -/
```
```lean
noncomputable def qElem (s : BlockState) (q : RegionName) (s_k_h s_k_t BT : Nat)
    (i : Fin BT) (e : Nat) : ℝ :=
  s.readMem q (s.pids 2 * s_k_h + (s.pids 1 * BT + i.val) * s_k_t + e)
```
</details>

<details><summary><code>hElem</code></summary>

```
/-- `h[e, p]` element (chunk state, base `h + i_bh·s_h_h + i_t·K·V`):
`h` at `i_bh·s_h_h + i_t·K·V + e·s_h_t + (i_v·BV + p)`. -/
```
```lean
noncomputable def hElem (s : BlockState) (h : RegionName) (s_h_h s_h_t K V BV : Nat)
    (p : Fin BV) (e : Nat) : ℝ :=
  s.readMem h (s.pids 2 * s_h_h + s.pids 1 * K * V + e * s_h_t + (s.pids 0 * BV + p.val) * 1)
```
</details>

<details><summary><code>gElem</code></summary>

```
/-- `g[i]` element (gate): `g` at `i_bh·T + (i_t·BT + i)`. -/
```
```lean
noncomputable def gElem (s : BlockState) (g : RegionName) (T BT : Nat) (i : Fin BT) : ℝ :=
  s.readMem g (s.pids 2 * T + (s.pids 1 * BT + i.val) * 1)
```
</details>

<details><summary><code>kElem</code></summary>

```
/-- `k[e, j]` element (block ptr layout `(K,T)` strides `(1, s_k_t)`):
`k` at `i_bh·s_k_h + e + (i_t·BT + j)·s_k_t`. -/
```
```lean
noncomputable def kElem (s : BlockState) (k : RegionName) (s_k_h s_k_t BT : Nat)
    (j : Fin BT) (e : Nat) : ℝ :=
  s.readMem k (s.pids 2 * s_k_h + e * 1 + (s.pids 1 * BT + j.val) * s_k_t)
```
</details>
