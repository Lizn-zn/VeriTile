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
into `O` at every chunk `c < NT` and lane `(t, d)`. All four `STORE`/`IFCOND`
branch surfaces also lower faithfully to the algorithm layer. The only layout
contracts are the contiguity ones the kernel genuinely relies on: Q/V/O block
strides `(BD, 1)`, K block strides `(1, BD)`, recurrent stride `(s_ht, 1)`, and the
dynamic bound `cdiv(NT·BT, BT) = NT` (needs `0 < BT`). The Python test shape
(`s_qh = 131072`, `s_hh = 524288`, `s_ht = 128`, `BT = 32`, `BD = 128`, `NT = 32`,
`scale = 1/√128`) is the special case. -/
```
</details>

**Statement:**
```lean
theorem attention_fwd_triton1_output_summary_general
    (Q K V H O : RegionName) (s_qh s_hh s_ht : Nat) (scale : ℝ)
    (BT BD NT : Nat) (hBT : 0 < BT) (s : BlockState)
    (hOQ : O ≠ Q) (hOK : O ≠ K) (hOV : O ≠ V) (hOH : O ≠ H) :
    -- (1) genuine closed-form output of the executed kernel
    (∃ sF, exec (attention_fwd_kernel_surface Q K V H O
        s_qh BD 1 s_hh s_ht (NT * BT) scale BT BD NT Bool.false Bool.false).toAlgKernel s = some sF
      ∧ ∀ (c : Nat), c < NT → ∀ (t : Fin BT) (d : Fin BD),
          sF.readMem O (s.pids 0 * s_qh + (c * BT + t.val) * BD + d.val)
            = outputClosedForm s Q K V scale BT BD
                (aft1QAddrG s s_qh BT BD) (aft1KAddrG s s_qh BT BD)
                (aft1QAddrG s s_qh BT BD) c t d) ∧
    -- (2) all four STORE/IFCOND branch surfaces lower to the algorithm layer
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
        = Except.ok alg))
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
