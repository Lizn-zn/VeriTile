# Spec sheet — `bench/tritonbench_g/chunk_gla_simple/ChunkGlaSimple.lean`

**Python source:** `bench/tritonbench_g/chunk_gla_simple/chunk_gla_simple.py`

## Public theorem: `chunk_gla_simple_output_summary_general`

<details><summary>docstring</summary>

```
/-- **Public dimension-general output summary.** Over *symbolic* strides, scale,
and dimensions `T K V BT BK BV` (with `K = BK`, `BK,BT > 0`, the `undef`-free and
output-offset-injective side conditions), the full simple-GLA forward surface

* lowers to the algorithm layer, and
* executes so that every active output lane of `o` equals the genuine GLA closed
  form `glaOutput` (read off the kernel's actual store; the `glaOutput` spec reads
  the *input* memory `q/k/v/h/g`, NOT a self-referential exec read-back).

The masked write map matches the kernel's store mask (`active`), and every active
output lane of `o` equals the genuine GLA closed form `glaOutput` (read off the
kernel's actual store; the `glaOutput` spec reads the *input* memory `q/k/v/h/g`,
NOT a self-referential exec read-back). Discharged via
`chunk_gla_simple_fwd_surface_toAlgorithm_supported` (lowering) and
`chunk_gla_simple_exec_glaOutput` (per-lane readback). The pinned per-Python-case
summaries are specializations of this theorem at their literal dimensions. -/
```
</details>

**Statement:**
```lean
specification chunk_gla_simple_output_summary_general
    (q k v h g o : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV : Nat) (s : BlockState)
    (hKBK : K = BK) (hBK : 0 < BK) (hBT : 0 < BT)
    (hundef : ∀ rg off, s.undef rg off = 0)
    (hInj : Function.Injective
      (fun idx : TileIndex [BT, BV] => outOffset s s_v_h s_v_t BT BV idx)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := chunk_gla_simple_fwd_surface q k v h g o s_k_h s_k_t s_v_h s_v_t
        s_h_h s_h_t scale T K V BT BK BV)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BT, BV] => active s T V BT BV idx)
        (fun idx => (o, outOffset s s_v_h s_v_t BT BV idx)))
      (expected := fun idx : TileIndex [BT, BV] =>
        glaOutput s q k v h g s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t
          scale T K V BT BK BV idx.1 idx.2.1)
```

**Assumptions / layout contracts:**
- `hKBK : K = BK`
- `hBK : 0 < BK`
- `hBT : 0 < BT`
- `hundef : ∀ rg off, s.undef rg off = 0`
- `hInj : Function.Injective
      (fun idx : TileIndex [BT, BV] => outOffset s s_v_h s_v_t BT BV idx)`
- `fun idx : TileIndex [BT, BV] => active s T V BT BV idx`

**Closed-form spec defs (transitive):** `outOffset`, `chunk_gla_simple_fwd_surface`, `active`, `glaOutput`, `tIndex`, `vIndex`, `interTerm`, `scoreTerm`, `vElem`, `qElem`, `hElem`, `gElem`, `kElem`

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

## Public theorem: `chunk_gla_simple_io_correctness`

<details><summary>docstring</summary>

```
/-- **The `chunk_gla_simple` `⊨[R]` io headline.** For every rounding model
`R`, `chunk_simple_gla_fwd_kernel_o` implements, on its
`StreamMasked3DKernelIO₅` signature, the **ideal-ℝ chunked gated-linear-attention
output fold** over its five streamed input channels: output lane `j = (i, p)`
of `o` holds

```
  ( (Σ_e q[i,e]·h[e,p]) · exp(g_i)
    + Σ_jj (if jj ≤ i then (Σ_e q[i,e]·k[e,jj]) · exp(g_i − g_jj) else 0) · v[jj,p]
  ) · scale
```

(`chunkGlaSimpleIOOutSpec` = the exact stack's `glaOutput` closed form restated
on the streams), and every flat cell outside the write window is untouched.

The output grid is the `.real` default: the surface's
`tl.store(p_o, b_o.to(p_o.dtype.element_ty), …)` cast **erases at translation**
(the lowered statement is `chunkGlaStoreStmt = Stmt.store .real …`, matched to
the surface by `chunk_gla_simple_body_split`'s `rfl`), and so does the pre-store
`b_s.to(b_v.dtype)`. The whole body is therefore cast-free — machine-checked by
`cgsIO_execR_eq : execR R … = exec …` — so **no `R.round .fp16 = id` modeling
boundary is carried**, and at every `R` the terminal cells hold the exact fold
values.

**Hypothesis provenance** (all truth-forced, inherited from the exact headline
`chunk_gla_simple_output_summary_general`):

* `hKBK : K = BK` — the single-key-block regime: it makes the surface's
  `for i_k in range(cdiv(K, BK))` a one-iteration loop, which is what the skin's
  step count `T = 1` records (and what every checked Python case runs:
  `K = BK = 64` for cases 1–3, `K = BK = 32` for case 4).
* `hBK : 0 < BK` — needed to evaluate the compiled `(K + BK − 1) / BK` trip
  count; a zero-wide key block is not a launch.
* `hBT : 0 < BT` — nonempty time block, as on the exact stack.
* `hInj` — the terminal store is a masked scatter, so its per-lane readback is
  only well defined when distinct output lanes hit distinct offsets; this is the
  ∀-pids form of the exact headline's per-program `hInj`.

The exact headline's `hundef` is **not** a hypothesis here — the skin's Hoare
triple carries the `undef` pin itself.

**Scope inherited from the port**: arithmetic is over `ℝ` (not bit-accurate
IEEE); `@triton.autotune` / `num_warps` are not modeled; the host launch (the
3-D grid and the host-computed `BK`/`BV`) is the trusted boundary, and the
statement is universally quantified over the launch state, so it covers every
program of the grid. -/
```
</details>

**Statement:**
```lean
specification chunk_gla_simple_io_correctness (R : RoundingModel)
    (q k v h g o : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat) (scale : ℝ)
    (Tt K V BT BK BV : Nat)
    (hKBK : K = BK) (hBK : 0 < BK) (hBT : 0 < BT)
    (hInj : ∀ p₀ p₁ p₂ : Nat, Function.Injective
      (fun idx : TileIndex [BT, BV] =>
        p₂ * s_v_h + (p₁ * BT + idx.1.val) * s_v_t + (p₀ * BV + idx.2.1.val) * 1)) :
    chunkGlaSimpleIO q k v h g o s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale
      Tt K V BT BK BV ⊨[R]
      fun _ _ _ xs ys zs ws vs j =>
        chunkGlaSimpleIOOutSpec scale BT BK BV xs ys zs ws vs j
```

**Assumptions / layout contracts:**
- `hKBK : K = BK`
- `hBK : 0 < BK`
- `hBT : 0 < BT`
- `fun idx : TileIndex [BT, BV] =>
        p₂ * s_v_h + (p₁ * BT + idx.1.val) * s_v_t + (p₀ * BV + idx.2.1.val) * 1`

**Closed-form spec defs (transitive):** `chunkGlaSimpleIO`, `chunkGlaSimpleIOOutSpec`, `chunk_gla_simple_fwd_surface`, `cgsIOqT`, `cgsIOhT`, `cgsIOgT`, `cgsIOScoreTerm`, `cgsIOvT`, `cgsIOkT`

<details><summary><code>chunkGlaSimpleIO</code></summary>

```
/-- **Streaming IO signature** of `chunk_simple_gla_fwd_kernel_o` on the
five-stream single-output fold skin. The step count is `1`: in the file's
declared `K = BK` regime the key-block loop `for i_k in range(cdiv(K, BK))`
runs exactly once, so all five channels are static one-step streams. -/
```
```lean
def chunkGlaSimpleIO (q k v h g o : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
    (scale : ℝ) (Tt K V BT BK BV : Nat) : StreamMasked3DKernelIO₅ where
  kernel := chunk_gla_simple_fwd_surface q k v h g o s_k_h s_k_t s_v_h s_v_t
    s_h_h s_h_t scale Tt K V BT BK BV
  inp1 := q
  inp2 := k
  inp3 := h
  inp4 := g
  inp5 := v
  out := o
  T := 1
  B1 := BT * BK
  B2 := BK * BT
  B3 := BK * BV
  B4 := BT
  B5 := BT * BV
  C := BT * BV
  outDType := .real
  read1 := fun _ p₁ p₂ _ j => p₂ * s_k_h + (p₁ * BT + j.val / BK) * s_k_t + j.val % BK
  read2 := fun _ p₁ p₂ _ j => p₂ * s_k_h + (j.val / BT) * 1 + (p₁ * BT + j.val % BT) * s_k_t
  read3 := fun p₀ p₁ p₂ _ j =>
    p₂ * s_h_h + p₁ * K * V + (j.val / BV) * s_h_t + (p₀ * BV + j.val % BV) * 1
  read4 := fun _ p₁ p₂ _ j => p₂ * Tt + (p₁ * BT + j.val) * 1
  read5 := fun p₀ p₁ p₂ _ j =>
    p₂ * s_v_h + (p₁ * BT + j.val / BV) * s_v_t + (p₀ * BV + j.val % BV) * 1
  write := fun p₀ p₁ p₂ j =>
    p₂ * s_v_h + (p₁ * BT + j.val / BV) * s_v_t + (p₀ * BV + j.val % BV) * 1
  mask1 := fun _ p₁ _ _ j => p₁ * BT + j.val / BK < Tt ∧ j.val % BK < K
  mask2 := fun _ p₁ _ _ j => j.val / BT < K ∧ p₁ * BT + j.val % BT < Tt
  mask3 := fun p₀ _ _ _ j => j.val / BV < K ∧ p₀ * BV + j.val % BV < V
  mask4 := fun _ p₁ _ _ j => p₁ * BT + j.val < Tt
  mask5 := fun p₀ p₁ _ _ j => p₁ * BT + j.val / BV < Tt ∧ p₀ * BV + j.val % BV < V
  writeMask := fun p₀ p₁ _ j => p₁ * BT + j.val / BV < Tt ∧ p₀ * BV + j.val % BV < V
```
</details>

<details><summary><code>chunkGlaSimpleIOOutSpec</code></summary>

```
/-- **The GLA closed form on the streams** — `glaOutput` restated over the
five streamed tiles, at output lane `j = (i, p)` row-major over `[BT, BV]`. -/
```
```lean
noncomputable def chunkGlaSimpleIOOutSpec (scale : ℝ) (BT BK BV : Nat)
    (xs : Fin 1 → Fin (BT * BK) → ℝ) (ys : Fin 1 → Fin (BK * BT) → ℝ)
    (zs : Fin 1 → Fin (BK * BV) → ℝ) (ws : Fin 1 → Fin BT → ℝ)
    (vs : Fin 1 → Fin (BT * BV) → ℝ) (j : Fin (BT * BV)) : ℝ :=
  ((Finset.univ.sum fun e : Fin BK =>
        cgsIOqT BT BK xs (Lane2D.decode j).1 e
          * cgsIOhT BK BV zs e (Lane2D.decode j).2.1)
      * Real.exp (cgsIOgT BT ws (Lane2D.decode j).1)
    + Finset.univ.sum fun jj : Fin BT =>
        cgsIOScoreTerm BT BK xs ys ws (Lane2D.decode j).1 jj
          * cgsIOvT BT BV vs jj (Lane2D.decode j).2.1) * scale
```
</details>

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

<details><summary><code>cgsIOqT</code></summary>

```
/-- `q[i, e]` off the first stream. -/
```
```lean
noncomputable def cgsIOqT (BT BK : Nat) (xs : Fin 1 → Fin (BT * BK) → ℝ)
    (i : Fin BT) (e : Fin BK) : ℝ := xs 0 (Lane2D.encode (i, e, PUnit.unit))
```
</details>

<details><summary><code>cgsIOhT</code></summary>

```
/-- `h[e, p]` off the third stream. -/
```
```lean
noncomputable def cgsIOhT (BK BV : Nat) (zs : Fin 1 → Fin (BK * BV) → ℝ)
    (e : Fin BK) (p : Fin BV) : ℝ := zs 0 (Lane2D.encode (e, p, PUnit.unit))
```
</details>

<details><summary><code>cgsIOgT</code></summary>

```
/-- `g[i]` off the fourth stream. -/
```
```lean
noncomputable def cgsIOgT (BT : Nat) (ws : Fin 1 → Fin BT → ℝ) (i : Fin BT) : ℝ := ws 0 i
```
</details>

<details><summary><code>cgsIOScoreTerm</code></summary>

```
/-- Masked, decayed intra-chunk score lane `(i, jj)` on the streams — the
stream restatement of `scoreTerm`. -/
```
```lean
noncomputable def cgsIOScoreTerm (BT BK : Nat) (xs : Fin 1 → Fin (BT * BK) → ℝ)
    (ys : Fin 1 → Fin (BK * BT) → ℝ) (ws : Fin 1 → Fin BT → ℝ)
    (i jj : Fin BT) : ℝ :=
  if jj.val ≤ i.val then
    (Finset.univ.sum fun e : Fin BK => cgsIOqT BT BK xs i e * cgsIOkT BT BK ys e jj)
      * Real.exp (cgsIOgT BT ws i - cgsIOgT BT ws jj)
  else 0
```
</details>

<details><summary><code>cgsIOvT</code></summary>

```
/-- `v[jj, p]` off the fifth stream. -/
```
```lean
noncomputable def cgsIOvT (BT BV : Nat) (vs : Fin 1 → Fin (BT * BV) → ℝ)
    (jj : Fin BT) (p : Fin BV) : ℝ := vs 0 (Lane2D.encode (jj, p, PUnit.unit))
```
</details>

<details><summary><code>cgsIOkT</code></summary>

```
/-- `k[e, jj]` off the second stream. -/
```
```lean
noncomputable def cgsIOkT (BT BK : Nat) (ys : Fin 1 → Fin (BK * BT) → ℝ)
    (e : Fin BK) (jj : Fin BT) : ℝ := ys 0 (Lane2D.encode (e, jj, PUnit.unit))
```
</details>
