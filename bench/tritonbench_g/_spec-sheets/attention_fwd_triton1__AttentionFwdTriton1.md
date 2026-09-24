# Spec sheet — `bench/tritonbench_g/attention_fwd_triton1/AttentionFwdTriton1.lean`

**Python source:** `bench/tritonbench_g/attention_fwd_triton1/attention_fwd_triton1.py`

## Public theorem: `attention_fwd_triton1_output_summary_general`

<details><summary>docstring</summary>

```
/-- **Dimension-general `output_summary` for `attention_fwd_triton1`.**

For arbitrary batch-head stride `s_qh`, chunk size `BT > 0`, head dimension `BD`,
chunk count `NT`, scale `scale : ℝ`, and recurrent-state strides `s_hh`/`s_ht`,
executing the full `attention_fwd_kernel_surface` writes the genuine
`outputClosedForm` (= `localTerm` intra-chunk `(scale·Q·K·V)` + `recurrentTerm`
cross-chunk `(scale·Q·b_h_c)`, read purely over INPUT memory, **no self-reference**)
into `O` at every chunk `c < NT` and lane `(t, d)`.

Stated through the standard `ComputeCorrect.Realizes_without_Rounding` surface (like every other
kernel summary), so the `exec`/`Except`/`toAlgorithm?` plumbing stays out of the
statement: the output write is the `WriteMap` over the streamed index
`(chunk, row, col) : Fin NT × Fin BT × Fin BD`, and `expected` is the genuine
`outputClosedForm`. Three conjuncts:
* **(1)** all four `STORE`/`IFCOND` branch surfaces lower faithfully to the
  algorithm layer;
* **(2)** the kernel runs to completion (the default branch's projected algorithm
  kernel produces a final state) — strictly stronger than `Realizes_without_Rounding` alone, which
  is conditional on execution succeeding;
* **(3)** `Realizes_without_Rounding`: every streamed `O` lane holds the genuine `outputClosedForm`.

The only layout contracts are the contiguity ones the kernel genuinely relies
on: Q/V/O block strides `(BD, 1)`, K block strides `(1, BD)`, recurrent stride
`(s_ht, 1)`, and the dynamic bound `cdiv(NT·BT, BT) = NT` (needs `0 < BT`). The
Python test shape (`s_qh = 131072`, `s_hh = 524288`, `s_ht = 128`, `BT = 32`,
`BD = 128`, `NT = 32`, `scale = 1/√128`) is the special case. -/
```
</details>

**Statement:**
```lean
specification attention_fwd_triton1_output_summary_general
    (Q K V H O : RegionName) (s_qh s_hh s_ht : Nat) (scale : ℝ)
    (BT BD NT : Nat) (hBT : 0 < BT) (s : BlockState)
    (hOQ : O ≠ Q) (hOK : O ≠ K) (hOV : O ≠ V) (hOH : O ≠ H) :
    -- (1) all four STORE/IFCOND branch surfaces lower to the algorithm layer
    ((∃ alg, (attention_fwd_kernel_surface Q K V H O
      s_qh BD 1 s_hh s_ht (NT * BT) scale BT BD NT Bool.false Bool.false).toAlgorithm?
        = Except.ok alg) ∧
     (∃ alg, (attention_fwd_kernel_surface Q K V H O
      s_qh BD 1 s_hh s_ht (NT * BT) scale BT BD NT Bool.true Bool.false).toAlgorithm?
        = Except.ok alg) ∧
     (∃ alg, (attention_fwd_kernel_surface Q K V H O
      s_qh BD 1 s_hh s_ht (NT * BT) scale BT BD NT Bool.false Bool.true).toAlgorithm?
        = Except.ok alg) ∧
     (∃ alg, (attention_fwd_kernel_surface Q K V H O
      s_qh BD 1 s_hh s_ht (NT * BT) scale BT BD NT Bool.true Bool.true).toAlgorithm?
        = Except.ok alg)) ∧
    -- (2) the default branch runs to completion (existence / termination)
    (∃ sF, exec (attention_fwd_kernel_surface Q K V H O
        s_qh BD 1 s_hh s_ht (NT * BT) scale BT BD NT Bool.false Bool.false).toAlgKernel s
          = some sF) ∧
    -- (3) standard Realizes_without_Rounding: every streamed O lane holds the genuine closed form
    ComputeCorrect.Realizes_without_Rounding
      (kernel := attention_fwd_kernel_surface Q K V H O
        s_qh BD 1 s_hh s_ht (NT * BT) scale BT BD NT Bool.false Bool.false)
      (initialState := s)
      (write := fun i : Fin NT × Fin BT × Fin BD =>
        some (O, s.pids 0 * s_qh + (i.1.val * BT + i.2.1.val) * BD + i.2.2.val))
      (expected := fun i : Fin NT × Fin BT × Fin BD =>
        outputClosedForm s Q K V scale BT BD
          (aft1QAddrG s s_qh BT BD) (aft1KAddrG s s_qh BT BD)
          (aft1QAddrG s s_qh BT BD) i.1.val i.2.1 i.2.2)
```

**Assumptions / layout contracts:**
- `hBT : 0 < BT`
- `hOQ : O ≠ Q`
- `hOK : O ≠ K`
- `hOV : O ≠ V`
- `hOH : O ≠ H`

**Closed-form spec defs (transitive):** `attention_fwd_kernel_surface`, `outputClosedForm`, `aft1QAddrG`, `aft1KAddrG`, `localTerm`, `recurrentTerm`, `recurrentState`

<details><summary><code>attention_fwd_kernel_surface</code></summary>

```
/-- Faithful DSL port of `attention_fwd_triton1.py`'s
`attention_fwd_kernel`.

The Python kernel uses block pointers plus two constexpr gates, `STORE` and
`IFCOND`. The `order` metadata is accepted by the DSL and erased into the same
block-pointer AST. -/
```
```lean
def attention_fwd_kernel_surface
    (q k v h o : RegionName)
    (s_qh s_qt s_qd s_hh s_ht T : Nat) (scale : ℝ)
    (BT BD NT : Nat) (STORE IFCOND : Bool) :
    ComputeKernel := triton {
  i_bh = tl.program_id(0)
  b_h = tl.zeros([$(BD), $(BD)], dtype=tl.float32)
  for i in range($(0), tl.cdiv($(T), $(BT))) {
    p_q = tl.make_block_ptr(base=q + i_bh * $(s_qh),
      shape=($(T), $(BD)), strides=($(s_qt), $(s_qd)),
      offsets=(i * $(BT), 0), block_shape=($(BT), $(BD)), order=(1, 0))
    p_k = tl.make_block_ptr(base=k + i_bh * $(s_qh),
      shape=($(BD), $(T)), strides=($(s_qd), $(s_qt)),
      offsets=(0, i * $(BT)), block_shape=($(BD), $(BT)), order=(0, 1))
    p_v = tl.make_block_ptr(base=v + i_bh * $(s_qh),
      shape=($(T), $(BD)), strides=($(s_qt), $(s_qd)),
      offsets=(i * $(BT), 0), block_shape=($(BT), $(BD)), order=(1, 0))
    p_h = tl.make_block_ptr(base=h + i_bh * $(s_hh),
      shape=($((NT * BD : Nat)), $(BD)), strides=($(s_ht), $(s_qd)),
      offsets=(i * $(BD), 0), block_shape=($(BD), $(BD)), order=(1, 0))
    p_o = tl.make_block_ptr(base=o + i_bh * $(s_qh),
      shape=($(T), $(BD)), strides=($(s_qt), $(s_qd)),
      offsets=(i * $(BT), 0), block_shape=($(BT), $(BD)), order=(1, 0))

    if STORE {
      tl.store(p_h, (b_h).to(p_h.dtype.element_ty))
    }
    b_q = tl.load(p_q)
    b_q = (b_q * $((scale : ℝ))).to(b_q.dtype)
    b_k = tl.load(p_k)
    b_v = tl.load(p_v)

    b_s = tl.dot(b_q, b_k, allow_tf32=false)
    b_o = tl.dot((b_s).to(b_q.dtype), b_v, allow_tf32=false)
    if IFCOND {
      if i == $(0) {
        b_h = tl.dot(b_k, b_v, allow_tf32=false)
      } else {
        b_o += tl.dot(b_q, (b_h).to(b_q.dtype), allow_tf32=false)
        b_h += tl.dot(b_k, b_v, allow_tf32=false)
      }
    } else {
      b_o += tl.dot(b_q, (b_h).to(b_q.dtype), allow_tf32=false)
      b_h += tl.dot(b_k, b_v, allow_tf32=false)
    }

    tl.store(p_o, (b_o).to(p_o.dtype.element_ty))
  }
}
```
</details>

<details><summary><code>outputClosedForm</code></summary>

```
/-- Genuine closed-form output of chunk `chunk`, position `(t, d)`:
local term plus recurrent term. No self-reference to the executed kernel. -/
```
```lean
noncomputable def outputClosedForm
    (s : BlockState) (Q K V : RegionName) (scale : ℝ) (BT BD : Nat)
    (qAddr kAddr vAddr : Nat → Nat → Nat)
    (chunk : Nat) (t : Fin BT) (d : Fin BD) : ℝ :=
  localTerm s Q K V scale BT BD qAddr kAddr vAddr chunk t d +
    recurrentTerm s Q K V scale BT BD qAddr kAddr vAddr chunk t d
```
</details>

<details><summary><code>aft1QAddrG</code></summary>

```
/-- General `Q`/`V`/`O` accessor: chunk `c` offset `off ↦ base + c·(BT·BD) + off`. -/
```
```lean
def aft1QAddrG (s : BlockState) (s_qh BT BD : Nat) (c off : Nat) : Nat :=
  s.pids 0 * s_qh + c * (BT * BD) + off
```
</details>

<details><summary><code>aft1KAddrG</code></summary>

```
/-- General `K` accessor: `off = BT·d' + tk ↦ base + d' + (c·BT + tk)·BD`. -/
```
```lean
def aft1KAddrG (s : BlockState) (s_qh BT BD : Nat) (c off : Nat) : Nat :=
  s.pids 0 * s_qh + off / BT + (c * BT + off % BT) * BD
```
</details>

<details><summary><code>localTerm</code></summary>

```
/-- One chunk's local (`(scale·Q)·K·V`) closed-form entry, stated directly over
input memory via per-chunk row/col offset accessors. -/
```
```lean
noncomputable def localTerm
    (s : BlockState) (Q K V : RegionName) (scale : ℝ) (BT BD : Nat)
    (qAddr kAddr vAddr : Nat → Nat → Nat)
    (chunk : Nat) (t : Fin BT) (d : Fin BD) : ℝ :=
  ∑ tk : Fin BT,
    (∑ dd : Fin BD,
      (s.readMem Q (qAddr chunk (BD * t.val + dd.val)) * scale) *
        s.readMem K (kAddr chunk (BT * dd.val + tk.val))) *
      s.readMem V (vAddr chunk (BD * tk.val + d.val))
```
</details>

<details><summary><code>recurrentTerm</code></summary>

```
/-- The recurrent output contribution `((scale·Qᵢ)·b_h_i)[t, d]`. -/
```
```lean
noncomputable def recurrentTerm
    (s : BlockState) (Q K V : RegionName) (scale : ℝ) (BT BD : Nat)
    (qAddr kAddr vAddr : Nat → Nat → Nat)
    (chunk : Nat) (t : Fin BT) (d : Fin BD) : ℝ :=
  ∑ d' : Fin BD,
    (s.readMem Q (qAddr chunk (BD * t.val + d'.val)) * scale) *
      recurrentState s K V BT BD kAddr vAddr chunk d' d
```
</details>

<details><summary><code>recurrentState</code></summary>

```
/-- The recurrent state matrix `b_h_i[d', d] = Σ_{j < i} (Kⱼᵀ·Vⱼ)[d', d]`,
genuine closed form over input memory. -/
```
```lean
noncomputable def recurrentState
    (s : BlockState) (K V : RegionName) (BT BD : Nat)
    (kAddr vAddr : Nat → Nat → Nat)
    (chunk : Nat) (d' d : Fin BD) : ℝ :=
  ∑ j ∈ Finset.range chunk,
    ∑ tk : Fin BT,
      s.readMem K (kAddr j (BT * d'.val + tk.val)) *
        s.readMem V (vAddr j (BD * tk.val + d.val))
```
</details>

## Public theorem: `attention_fwd_triton1_io_correctness`

<details><summary>docstring</summary>

```
/-- **The `⊨[R]` streaming headline (wave-5 S3 per-step emit genre, matrix
carry).** For every rounding model `R`, the faithful `attention_fwd_kernel`
surface in its **verified default configuration** (`STORE=false`,
`IFCOND=false`) implements, on its `StreamEmitMasked2DKernelIO₃` signature,
the **ideal ℝ linear-attention forward**: emitted lane `j` of chunk `c` holds
`((scale·Qc)·Kc·Vc + (scale·Qc)·Σ_{u<c} Kᵤᵀ·Vᵤ)[j/BD, j%BD]` — the spec `f`
is exact real arithmetic (no softmax; the carry is the `[BD, BD]` matrix
state, and the emitted value uses the **pre-update** carry). The kernel has
**zero rounding events** (the erased `(b_q·scale).to(...)` /
`(b_o).to(...)` casts are `.real → .real`, all loads and the in-loop store
are `.real`), so the skin's boundary quantization degenerates: the readback
contract's `R.round .real` is the identity by the model's defining
`round_real` — the ∀-`R` face holds via the `RoundingModel` `.real` identity
fields, not as a `.triv` special case.

Layer map: the loop body is cast-free, so under `execR R` it collapses
statement-by-statement onto the exact stepper and the proven `aft1InvG` carry
stack above (`aft1_prologue_invG` / `aft1InvG_step` / `forRangeDyn_inv`) is
reused unchanged; the `⊨[R]` face adds the `TraceSafeR` walk (the wave's
first over a `forRangeDyn` loop — `aft1_stopOp_eval` resolves the dynamic
`cdiv` bound to `NT` once, then `Stmt.forRangeTraceSafeR_inv` runs over
`aft1InvG` with the per-chunk block pointers recomputed through the `i_bh`
pin), the per-cell memory frame (`aft1_body_memFrame`, the raw-`mem` twin of
the iteration lemma), and the `Lane2D` stream-lane spec bridge
(`aft1_streamSpec_eq_outG`).

All hypotheses are inherited from the exact headline
`attention_fwd_triton1_output_summary_general`'s side conditions: `hBT` pins
the `cdiv(NT·BT, BT) = NT` bound resolution, and the four region-distinctness
hypotheses keep the per-chunk `O` stores from clobbering the streamed inputs
(and the dead `h` buffer) between iterations. The skin quantifies a 2-D pid
grid; the kernel is a 1-D grid (`i_bh = pid₀`), so every window ignores
`pid₁`.

Configuration scope: this io face covers exactly the branch the exact
closed-form stack verifies. The `STORE=true` / `IFCOND=true` configurations
stay on the existing coverage — conjunct (1) of
`attention_fwd_triton1_output_summary_general` (all four `STORE`/`IFCOND`
surfaces lower faithfully to the algorithm layer); their `⊨[R]` faces are
future work, not regressions.

Relation to the exact surface: the exact headline above is retained
unchanged; this `⊨[R]` face restates the same linear-attention content on the
streaming emit skin, for every `R` at once (at the `.real` grid the two faces
carry the same exact cell). Both faces are kept per the rounding-as-default
doctrine. -/
```
</details>

**Statement:**
```lean
specification attention_fwd_triton1_io_correctness (R : RoundingModel)
    (Q K V H O : RegionName) (s_qh s_hh s_ht : Nat) (scale : ℝ)
    (BT BD NT : Nat) (hBT : 0 < BT)
    (hOQ : O ≠ Q) (hOK : O ≠ K) (hOV : O ≠ V) (hOH : O ≠ H) :
    attentionFwdTriton1KernelIO Q K V H O s_qh s_hh s_ht scale BT BD NT ⊨[R]
      fun _ _ qs ks vs t j => aft1StreamSpec scale NT BT BD qs ks vs t j
```

**Assumptions / layout contracts:**
- `hBT : 0 < BT`
- `hOQ : O ≠ Q`
- `hOK : O ≠ K`
- `hOV : O ≠ V`
- `hOH : O ≠ H`

**Closed-form spec defs (transitive):** `attentionFwdTriton1KernelIO`, `aft1StreamSpec`, `attention_fwd_kernel_surface`

<details><summary><code>attentionFwdTriton1KernelIO</code></summary>

```
/-- **Streaming IO signature** of the `STORE=false, IFCOND=false` branch of
`attention_fwd_kernel` on the three-stream per-step emit skin. Step `t` of the
`cdiv`-bounded chunk loop reads the `[BT, BD]` `Q` tile, the `[BD, BT]` `K`
tile and the `[BT, BD]` `V` tile through the per-chunk `make_block_ptr`s, and
stores the `[BT, BD]` output tile through `p_o` at the **`.real`** grid
(`outDType` default — the store's `.to(p_o.dtype.element_ty)` cast erases, so
the per-step stores have no quantization event). The windows transcribe the
block pointers' effective 2-D addresses verbatim in the standard row-major
`Lane2D` div/mod spelling (`pid₀ = i_bh`; the kernel is a 1-D grid, so `pid₁`
is ignored):

* `read1` step `t`, lane `j`: `i_bh·s_qh + (t·BT + j/BD)·BD + j%BD` — `Q` tile
  cell `(j/BD, j%BD)` of chunk `t` (strides `(BD, 1)`).
* `read2` step `t`, lane `j`: `i_bh·s_qh + j/BT + (t·BT + j%BT)·BD` — `K` tile
  cell `(j/BT, j%BT)` of chunk `t` (strides `(1, BD)`).
* `read3` / `write`: the same `(BD, 1)`-strided window into `V` / `O`.

All loads and the store are unmasked with empty boundary check, so every mask
is `True`. -/
```
```lean
def attentionFwdTriton1KernelIO (Q K V H O : RegionName)
    (s_qh s_hh s_ht : Nat) (scale : ℝ) (BT BD NT : Nat) :
    StreamEmitMasked2DKernelIO₃ where
  kernel := attention_fwd_kernel_surface Q K V H O s_qh BD 1 s_hh s_ht (NT * BT)
    scale BT BD NT Bool.false Bool.false
  inp1 := Q
  inp2 := K
  inp3 := V
  out := O
  T := NT
  B1 := BT * BD
  B2 := BD * BT
  B3 := BT * BD
  C := BT * BD
  read1 := fun p₀ _ t j => p₀ * s_qh + (t.val * BT + j.val / BD) * BD + j.val % BD
  read2 := fun p₀ _ t j => p₀ * s_qh + j.val / BT + (t.val * BT + j.val % BT) * BD
  read3 := fun p₀ _ t j => p₀ * s_qh + (t.val * BT + j.val / BD) * BD + j.val % BD
  write := fun p₀ _ t j => p₀ * s_qh + (t.val * BT + j.val / BD) * BD + j.val % BD
  mask1 := fun _ _ _ _ => True
  mask2 := fun _ _ _ _ => True
  mask3 := fun _ _ _ _ => True
  writeMask := fun _ _ _ _ => True
```
</details>

<details><summary><code>aft1StreamSpec</code></summary>

```
/-- The stream-level linear-attention spec (the genre's *scan* shape with a
matrix carry): output lane `j` (tile cell `(j/BD, j%BD)` by `Lane2D`) of
chunk `c` holds the local term `((scale·Qc)·Kc·Vc)[j/BD, j%BD]` plus the
recurrent term `((scale·Qc) · Σ_{u<c} Kᵤᵀ·Vᵤ)[j/BD, j%BD]` — the emitted
`b_o` uses the **pre-update** carry `b_h_c = Σ_{u<c} Kᵤᵀ·Vᵤ` (the kernel adds
`dot(b_q, b_h)` *before* `b_h += dot(b_k, b_v)`). Pure real arithmetic over
the three streams; the algebra is `aft1OutG` restated on the curried lanes. -/
```
```lean
noncomputable def aft1StreamSpec (scale : ℝ) (NT BT BD : Nat)
    (qs : Fin NT → Fin (BT * BD) → ℝ) (ks : Fin NT → Fin (BD * BT) → ℝ)
    (vs : Fin NT → Fin (BT * BD) → ℝ) (c : Fin NT) (j : Fin (BT * BD)) : ℝ :=
  (∑ tk : Fin BT,
    (∑ e : Fin BD,
        (qs c (Lane2D.encode ((Lane2D.decode j).1, e, PUnit.unit)) * scale) *
          ks c (Lane2D.encode (e, tk, PUnit.unit))) *
      vs c (Lane2D.encode (tk, (Lane2D.decode j).2.1, PUnit.unit)))
  + ∑ d' : Fin BD,
      (qs c (Lane2D.encode ((Lane2D.decode j).1, d', PUnit.unit)) * scale) *
        ∑ u : Fin c.val, ∑ tk : Fin BT,
          ks (Fin.castLE (Nat.le_of_lt c.isLt) u)
              (Lane2D.encode (d', tk, PUnit.unit)) *
            vs (Fin.castLE (Nat.le_of_lt c.isLt) u)
              (Lane2D.encode (tk, (Lane2D.decode j).2.1, PUnit.unit))
```
</details>

<details><summary><code>attention_fwd_kernel_surface</code></summary>

```
/-- Faithful DSL port of `attention_fwd_triton1.py`'s
`attention_fwd_kernel`.

The Python kernel uses block pointers plus two constexpr gates, `STORE` and
`IFCOND`. The `order` metadata is accepted by the DSL and erased into the same
block-pointer AST. -/
```
```lean
def attention_fwd_kernel_surface
    (q k v h o : RegionName)
    (s_qh s_qt s_qd s_hh s_ht T : Nat) (scale : ℝ)
    (BT BD NT : Nat) (STORE IFCOND : Bool) :
    ComputeKernel := triton {
  i_bh = tl.program_id(0)
  b_h = tl.zeros([$(BD), $(BD)], dtype=tl.float32)
  for i in range($(0), tl.cdiv($(T), $(BT))) {
    p_q = tl.make_block_ptr(base=q + i_bh * $(s_qh),
      shape=($(T), $(BD)), strides=($(s_qt), $(s_qd)),
      offsets=(i * $(BT), 0), block_shape=($(BT), $(BD)), order=(1, 0))
    p_k = tl.make_block_ptr(base=k + i_bh * $(s_qh),
      shape=($(BD), $(T)), strides=($(s_qd), $(s_qt)),
      offsets=(0, i * $(BT)), block_shape=($(BD), $(BT)), order=(0, 1))
    p_v = tl.make_block_ptr(base=v + i_bh * $(s_qh),
      shape=($(T), $(BD)), strides=($(s_qt), $(s_qd)),
      offsets=(i * $(BT), 0), block_shape=($(BT), $(BD)), order=(1, 0))
    p_h = tl.make_block_ptr(base=h + i_bh * $(s_hh),
      shape=($((NT * BD : Nat)), $(BD)), strides=($(s_ht), $(s_qd)),
      offsets=(i * $(BD), 0), block_shape=($(BD), $(BD)), order=(1, 0))
    p_o = tl.make_block_ptr(base=o + i_bh * $(s_qh),
      shape=($(T), $(BD)), strides=($(s_qt), $(s_qd)),
      offsets=(i * $(BT), 0), block_shape=($(BT), $(BD)), order=(1, 0))

    if STORE {
      tl.store(p_h, (b_h).to(p_h.dtype.element_ty))
    }
    b_q = tl.load(p_q)
    b_q = (b_q * $((scale : ℝ))).to(b_q.dtype)
    b_k = tl.load(p_k)
    b_v = tl.load(p_v)

    b_s = tl.dot(b_q, b_k, allow_tf32=false)
    b_o = tl.dot((b_s).to(b_q.dtype), b_v, allow_tf32=false)
    if IFCOND {
      if i == $(0) {
        b_h = tl.dot(b_k, b_v, allow_tf32=false)
      } else {
        b_o += tl.dot(b_q, (b_h).to(b_q.dtype), allow_tf32=false)
        b_h += tl.dot(b_k, b_v, allow_tf32=false)
      }
    } else {
      b_o += tl.dot(b_q, (b_h).to(b_q.dtype), allow_tf32=false)
      b_h += tl.dot(b_k, b_v, allow_tf32=false)
    }

    tl.store(p_o, (b_o).to(p_o.dtype.element_ty))
  }
}
```
</details>
