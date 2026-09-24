# Spec sheet — `bench/tritonbench_g/lightning_attention/LightningAttention.lean`

**Python source:** `bench/tritonbench_g/lightning_attention/lightning_attention.py`

## Public theorem: `lightning_attention_output_summary_general`

<details><summary>docstring</summary>

```
/-- **SCOPE — this is a claim about one block index `m` of the forward loop, not
about the whole launched kernel.** The full launched surfaces appear only in the
lowering clauses; the `NUM_BLOCK` fold that threads the carry across blocks is
outside every clause. The loop *body* — including `o_intra`, the causal mask, and
the masked `Out` writeback — is inside it (clause 4).

**Genuine, shape-general forward step summary** for the lightning-attention
kernel. Symbolic in every dimension (`d e BLOCK NUM_BLOCK BLOCK_MODEL`) and over
an arbitrary loop index `m` and materialized carry buffer; no dimension is
pinned. Exposes:

1. **Surface lowering** of the forward and both backward kernels (every
   dimension symbolic). Lowering is well-formedness, not correctness.
2. **`kv` carry-fold body — genuine closed form.** One `kv += tl.dot(k_trans,
   v)` body realizes its spec `kvStepSpec`, and under the *assumed* loop-carry
   invariant (`KVPrev = kvClosed m`, and `KTrans`/`VTile` staging block `m` of the
   launched `K`/`V`) that spec equals exactly `kvClosed (m+1)` — the running
   `Σ kᵀ·v` over the first `m+1` key blocks. This is a genuine standalone closed
   form over the launched input regions `K` and `V`, never an `exec` read-back.
3. **`o_inter = tl.dot(q, kv)` inter-block producer** realizes its spec
   `oInterDotSpec` — `Σ_a q[r,a]·kv[a,c]` against the carried state — which is
   **one of the two halves** of the causal linear-attention output row.
4. **★ The whole loop body → the launched `Out`.** The causal-masked intra-block
   product `o_intra = tl.dot(where(j ≤ r, qk, 0), v)`, the inter-block product
   `o_inter`, their sum, and the masked `tl.store` into `Out` at the launch's real
   address realize exactly `fwdOutClosed` — the causal linear-attention row
   `o[t,c] = Σ_{s ≤ t}(Σ_a q[t,a]·k[s,a])·v[s,c]` at `t = m·BLOCK + r` — over the
   input regions `Q`, `K`, `V`. The causal mask supplies the in-block keys and the
   carried state (via `hPrev`) supplies all earlier ones. The write map is
   `writeIf (m·BLOCK + r < n)`, so this clause covers the partial tail block
   too; its only side condition is `hBM` (see the module docstring).

What this theorem does **not** say: nothing about the
`NUM_BLOCK` loop that threads the carry, and nothing about the backward
gradients beyond lowering. See the module docstring's modeling boundary.

Side conditions. The staged-tile offsets are injective unconditionally from the
`Fin` bounds; the launched `Out` offset needs `hBM : BLOCK_MODEL ≤ e`, and clause 4
is stated per written lane, covering partial tail blocks as well (both with launch
provenance on their binders). **Load-bearing:** `hPrev`/`hK`/`hV` are *assumed*
loop-carry / staging hypotheses, not lemmas — `hPrev` is what clause 4 needs and
what the unmodeled `NUM_BLOCK` driver would have to establish. -/
```
</details>

**Statement:**
```lean
specification lightning_attention_output_summary_general
    (Q K V Out DO DQ DK DV KVPrev KTrans VTile KVOut OInter : RegionName)
    (s : BlockState)
    (_b h n d e BLOCK NUM_BLOCK BLOCK_MODEL m : Nat)
    -- block `m` lies inside the sequence; `NUM_BLOCK = cdiv(n, BLOCK)`
    -- (`lightning_attention.py:378`) so only a partial tail block can violate it
    -- the channel block fits the value width; the launch sets
    -- `BLOCK_MODEL = min(next_power_of_2(e), 32)` (`lightning_attention.py:380`)
    (hBM : BLOCK_MODEL ≤ e)
    (hPrev : ∀ idx : TileIndex [d, BLOCK_MODEL],
      s.readMem KVPrev (kvOffset BLOCK_MODEL idx)
        = kvClosed s K V n d e BLOCK BLOCK_MODEL m idx.1.val idx.2.1.val)
    (hK : ∀ (idx : TileIndex [d, BLOCK_MODEL]) (j : Fin BLOCK),
      s.readMem KTrans (idx.1.val * BLOCK + j.val)
        = fwdKVal s K n d idx.1.val (m * BLOCK + j.val))
    (hV : ∀ (idx : TileIndex [d, BLOCK_MODEL]) (j : Fin BLOCK),
      s.readMem VTile (j.val * BLOCK_MODEL + idx.2.1.val)
        = fwdVVal s V n e BLOCK_MODEL idx.2.1.val (m * BLOCK + j.val)) :
    -- (1) surface lowering of forward + both backward kernels (symbolic dims)
    (∃ alg, (lightning_attention_forward_surface Q K V Out _b h n d e BLOCK
      NUM_BLOCK BLOCK_MODEL).toAlgorithm? = Except.ok alg) ∧
    (∃ alg, (lightning_attention_bwd_intra_surface Q K V DO DQ DK DV
      _b h n d e BLOCK NUM_BLOCK BLOCK NUM_BLOCK).toAlgorithm? = Except.ok alg) ∧
    (∃ alg, (lightning_attention_bwd_inter_surface Q K V DO DQ DK DV
      _b h n d e BLOCK NUM_BLOCK BLOCK NUM_BLOCK).toAlgorithm? = Except.ok alg) ∧
    -- (2) kv carry-fold body realizes kvStepSpec ...
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := lightning_attention_forward_kv_step_slice KVPrev KTrans VTile KVOut
        d BLOCK BLOCK_MODEL)
      (initialState := s)
      (write := fun idx : TileIndex [d, BLOCK_MODEL] =>
        some (KVOut, kvOffset BLOCK_MODEL idx))
      (expected := fun idx : TileIndex [d, BLOCK_MODEL] =>
        kvStepSpec s KVPrev KTrans VTile d BLOCK BLOCK_MODEL idx)) ∧
    -- ... and that spec is the genuine closed-form kvClosed (m+1) under the carry invariant
    (∀ idx : TileIndex [d, BLOCK_MODEL],
      kvStepSpec s KVPrev KTrans VTile d BLOCK BLOCK_MODEL idx
        = kvClosed s K V n d e BLOCK BLOCK_MODEL (m + 1) idx.1.val idx.2.1.val) ∧
    -- (3) o_inter producer realizes its genuine spec oInterDotSpec
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := lightning_attention_forward_o_inter_dot_slice Q KVPrev OInter
        BLOCK d BLOCK_MODEL)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK, BLOCK_MODEL] =>
        some (OInter, oInterOffset BLOCK_MODEL idx))
      (expected := fun idx : TileIndex [BLOCK, BLOCK_MODEL] =>
        oInterDotSpec s Q KVPrev BLOCK d BLOCK_MODEL idx)) ∧
    -- (4) ★ the WHOLE loop body — the causal-masked `o_intra`, `o_inter`, their
    --     sum, and the masked `tl.store` into the launched `Out` — realizes the
    --     genuine causal linear-attention output row
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := lightning_attention_forward_body_slice Q K V KVPrev Out
        m n d e BLOCK BLOCK_MODEL)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK, BLOCK_MODEL] => m * BLOCK + idx.1.val < n)
        (fun idx => (Out, bodyOutOffset s m n e BLOCK BLOCK_MODEL idx)))
      (expected := fun idx : TileIndex [BLOCK, BLOCK_MODEL] =>
        fwdOutClosed s Q K V n d e BLOCK_MODEL
          (m * BLOCK + idx.1.val) idx.2.1.val))
```

**Assumptions / layout contracts:**
- `hBM : BLOCK_MODEL ≤ e`
- `hPrev : ∀ idx : TileIndex [d, BLOCK_MODEL],
      s.readMem KVPrev (kvOffset BLOCK_MODEL idx)
        = kvClosed s K V n d e BLOCK BLOCK_MODEL m idx.1.val idx.2.1.val`
- `hK : ∀ (idx : TileIndex [d, BLOCK_MODEL]) (j : Fin BLOCK),
      s.readMem KTrans (idx.1.val * BLOCK + j.val)
        = fwdKVal s K n d idx.1.val (m * BLOCK + j.val)`
- `hV : ∀ (idx : TileIndex [d, BLOCK_MODEL]) (j : Fin BLOCK),
      s.readMem VTile (j.val * BLOCK_MODEL + idx.2.1.val)
        = fwdVVal s V n e BLOCK_MODEL idx.2.1.val (m * BLOCK + j.val)`
- `∀ idx : TileIndex [d, BLOCK_MODEL],
      kvStepSpec s KVPrev KTrans VTile d BLOCK BLOCK_MODEL idx
        = kvClosed s K V n d e BLOCK BLOCK_MODEL (m + 1) idx.1.val idx.2.1.val`
- `fun idx : TileIndex [BLOCK, BLOCK_MODEL] => m * BLOCK + idx.1.val < n`

**Closed-form spec defs (transitive):** `kvOffset`, `kvClosed`, `fwdKVal`, `fwdVVal`, `lightning_attention_forward_surface`, `lightning_attention_bwd_intra_surface`, `lightning_attention_bwd_inter_surface`, `lightning_attention_forward_kv_step_slice`, `kvStepSpec`, `lightning_attention_forward_o_inter_dot_slice`, `oInterOffset`, `oInterDotSpec`, `lightning_attention_forward_body_slice`, `bodyOutOffset`, `fwdOutClosed`, `tileElem`

<details><summary><code>kvOffset</code></summary>

```
/-- Flat `[d, BLOCK_MODEL]` tile offset `a · BLOCK_MODEL + c`. -/
```
```lean
def kvOffset (BLOCK_MODEL : Nat) (idx : TileIndex [D, BLOCK_MODEL]) : Nat :=
  idx.1.val * BLOCK_MODEL + idx.2.1.val
```
</details>

<details><summary><code>kvClosed</code></summary>

```
/-- **Genuine closed form for the `kv` state entering block `m`**, element
`(a, c)`: `Σ_{s < m·BLOCK} K[s,a]·V[s,c]`, the running sum of `kᵀ·v` over the
first `m` key blocks. This is the carry state the Python loop accumulates with
`kv += tl.dot(k_trans, v)` *after* each block's output store. -/
```
```lean
noncomputable def kvClosed (s : BlockState) (K V : RegionName)
    (n d e BLOCK BLOCK_MODEL : Nat) (m : Nat) (a c : Nat) : ℝ :=
  ∑ keyRow ∈ Finset.range (m * BLOCK),
    fwdKVal s K n d a keyRow * fwdVVal s V n e BLOCK_MODEL c keyRow
```
</details>

<details><summary><code>fwdKVal</code></summary>

```
/-- `K[qk_offset + s·d + a]`: the key entry at global key row `s`, head channel
`a`, for the program `off_bh`. -/
```
```lean
noncomputable def fwdKVal (s : BlockState) (K : RegionName)
    (n d : Nat) (a : Nat) (keyRow : Nat) : ℝ :=
  s.readMem K (s.pids 0 * n * d + keyRow * d + a)
```
</details>

<details><summary><code>fwdVVal</code></summary>

```
/-- `V[v_offset + e_offset + s·e + c]`: the value entry at global key row `s`,
value channel `c` (within the `off_e` channel block), for the program. -/
```
```lean
noncomputable def fwdVVal (s : BlockState) (V : RegionName)
    (n e BLOCK_MODEL : Nat) (c : Nat) (keyRow : Nat) : ℝ :=
  s.readMem V (s.pids 0 * n * e + keyRow * e + s.pids 1 * BLOCK_MODEL + c)
```
</details>

<details><summary><code>lightning_attention_forward_surface</code></summary>

```
/-- Faithful transcription of `lightning_attention.py`'s `_fwd_kernel`.

This covers the full forward recurrent tile loop. -/
```
```lean
def lightning_attention_forward_surface
    (Q K V Out : RegionName)
    (_b h n d e BLOCK NUM_BLOCK BLOCK_MODEL : Nat) :
    ComputeKernel := triton {
  off_bh = tl.program_id(0)
  off_bh % $(h)
  off_e = tl.program_id(1)
  qk_offset = off_bh * $(n) * $(d)
  v_offset = off_bh * $(n) * $(e)
  o_offset = off_bh * $(n) * $(e)
  e_offset = off_e * $(BLOCK_MODEL)
  Q_block_ptr = Q + qk_offset + tl.arange(0, $(d))[None, :]
  K_trans_block_ptr = K + qk_offset + tl.arange(0, $(d))[:, None]
  V_block_ptr = V + v_offset + e_offset + tl.arange(0, $(BLOCK_MODEL))[None, :]
  O_block_ptr = Out + o_offset + e_offset + tl.arange(0, $(BLOCK_MODEL))[None, :]
  off_block = tl.arange(0, $(BLOCK))
  index = off_block[:, None] - off_block[None, :]
  kv = tl.zeros([$(d), $(BLOCK_MODEL)], dtype=tl.float32)
  for i in range($(0), $(NUM_BLOCK), $(1)) {
    q = tl.load(Q_block_ptr + off_block[:, None] * $(d),
      mask=off_block[:, None] < $(n), other=0.0).to(tl.float32)
    k_trans = tl.load(K_trans_block_ptr + off_block[None, :] * $(d),
      mask=off_block[None, :] < $(n), other=0.0).to(tl.float32)
    v = tl.load(V_block_ptr + off_block[:, None] * $(e),
      mask=off_block[:, None] < $(n), other=0.0).to(tl.float32)
    qk = tl.dot(q, k_trans)
    qk = tl.where(index >= 0, qk, 0)
    o_intra = tl.dot(qk, v)
    o_inter = tl.dot(q, kv)
    o = o_intra + o_inter
    tl.store(O_block_ptr + off_block[:, None] * $(e),
      (o).to(O_block_ptr.dtype.element_ty), mask=off_block[:, None] < $(n))
    kv += tl.dot(k_trans, v)
    off_block += $(BLOCK)
  }
}
```
</details>

<details><summary><code>lightning_attention_bwd_intra_surface</code></summary>

```
/-- Faithful transcription of `lightning_attention.py`'s `_bwd_intra_kernel`.

This records the intra-block backward path: diagonal causal masks, `DQ`/`DK`/
`DV` stores, and the Python test's block layout. -/
```
```lean
def lightning_attention_bwd_intra_surface
    (Q K V DO DQ DK DV : RegionName)
    (_b h n d e BLOCK NUM_BLOCK CBLOCK NUM_CBLOCK : Nat) :
    ComputeKernel := triton {
  off_bh = tl.program_id(0)
  off_block = tl.program_id(1)
  off_bh % $(h)
  qk_offset = off_bh * $(n) * $(d)
  v_offset = off_bh * $(n) * $(e)
  o_offset = off_bh * $(n) * $(e)
  block_offset = off_block * $(BLOCK) + tl.arange(0, $(BLOCK))
  Q_trans_block_ptr =
    Q + qk_offset + block_offset[None, :] * $(d) + tl.arange(0, $(d))[:, None]
  K_block_ptr =
    K + qk_offset + block_offset[:, None] * $(d) + tl.arange(0, $(d))[None, :]
  V_trans_block_ptr =
    V + v_offset + block_offset[None, :] * $(e) + tl.arange(0, $(e))[:, None]
  DQ_block_ptr =
    DQ + qk_offset + block_offset[:, None] * $(d) + tl.arange(0, $(d))[None, :]
  DK_trans_block_ptr =
    DK + qk_offset + block_offset[None, :] * $(d) + tl.arange(0, $(d))[:, None]
  DV_block_ptr =
    DV + v_offset + block_offset[:, None] * $(e) + tl.arange(0, $(e))[None, :]
  DO_block_ptr =
    DO + o_offset + block_offset[:, None] * $(e) + tl.arange(0, $(e))[None, :]
  array = tl.arange(0, $(BLOCK))
  index = array[:, None] - array[None, :]
  k = tl.load(K_block_ptr, mask=block_offset[:, None] < $(n),
    other=0.0).to(tl.float32)
  v_trans = tl.load(V_trans_block_ptr, mask=block_offset[None, :] < $(n),
    other=0.0).to(tl.float32)
  b_do = tl.load(DO_block_ptr, mask=block_offset[:, None] < $(n),
    other=0.0).to(tl.float32)
  q_trans = tl.load(Q_trans_block_ptr, mask=block_offset[None, :] < $(n),
    other=0.0).to(tl.float32)
  dqk = tl.dot(b_do, v_trans)
  dqk = tl.where(index >= 0, dqk, 0)
  dq_intra = tl.dot(dqk, k)
  dk_intra_trans = tl.dot(q_trans, dqk)
  qk_trans = tl.dot(k, q_trans)
  qk_trans = tl.where(index <= 0, qk_trans, 0)
  dv_intra = tl.dot(qk_trans, b_do)
  dq = dq_intra
  dk_trans = dk_intra_trans
  dv = dv_intra
  tl.store(DQ_block_ptr, (dq).to(DQ_block_ptr.dtype.element_ty),
    mask=block_offset[:, None] < $(n))
  tl.store(DK_trans_block_ptr, (dk_trans).to(DK_trans_block_ptr.dtype.element_ty),
    mask=block_offset[None, :] < $(n))
  tl.store(DV_block_ptr, (dv).to(DV_block_ptr.dtype.element_ty),
    mask=block_offset[:, None] < $(n))
}
```
</details>

<details><summary><code>lightning_attention_bwd_inter_surface</code></summary>

```
/-- Faithful transcription of `lightning_attention.py`'s `_bwd_inter_kernel`.

The surface preserves both Python loop nests: the forward scan that accumulates
and writes inter-block `DQ`, and the reverse scan that accumulates and writes
inter-block `DK`/`DV`. -/
```
```lean
def lightning_attention_bwd_inter_surface
    (Q K V DO DQ DK DV : RegionName)
    (_b h n d e BLOCK NUM_BLOCK CBLOCK NUM_CBLOCK : Nat) :
    ComputeKernel := triton {
  off_bh = tl.program_id(0)
  off_bh % $(h)
  qk_offset = off_bh * $(n) * $(d)
  v_offset = off_bh * $(n) * $(e)
  o_offset = off_bh * $(n) * $(e)
  DQ_block_ptr =
    DQ + qk_offset + tl.arange(0, $(CBLOCK))[:, None] * $(d) +
      tl.arange(0, $(d))[None, :]
  K_block_ptr =
    K + qk_offset + tl.arange(0, $(CBLOCK))[:, None] * $(d) +
      tl.arange(0, $(d))[None, :]
  V_trans_block_ptr =
    V + v_offset + tl.arange(0, $(CBLOCK))[None, :] * $(e) +
      tl.arange(0, $(e))[:, None]
  DO_block_ptr =
    DO + o_offset + tl.arange(0, $(CBLOCK))[:, None] * $(e) +
      tl.arange(0, $(e))[None, :]
  off_block1 = tl.arange(0, $(CBLOCK))
  off_block2 = tl.arange(0, $(CBLOCK))
  kv_trans = tl.zeros([$(e), $(d)], dtype=tl.float32)
  for i in range($(0), $(NUM_BLOCK), $(1)) {
    for j in range($(0), $(NUM_CBLOCK), $(1)) {
      if i > 0 {
        b_do = tl.load(DO_block_ptr, mask=off_block1[:, None] < $(n),
          other=0.0).to(tl.float32)
        dq_inter = tl.dot(b_do, kv_trans)
        dq = dq_inter + tl.load(DQ_block_ptr,
          mask=off_block1[:, None] < $(n), other=0.0)
        tl.store(DQ_block_ptr, (dq).to(DQ_block_ptr.dtype.element_ty),
          mask=off_block1[:, None] < $(n))
      }
      DQ_block_ptr += $(CBLOCK) * $(d)
      DO_block_ptr += $(CBLOCK) * $(e)
      off_block1 += $(CBLOCK)
    }
    kv_trans_current = tl.zeros([$(e), $(d)], dtype=tl.float32)
    for j in range($(0), $(NUM_CBLOCK), $(1)) {
      v_trans = tl.load(V_trans_block_ptr, mask=off_block2[None, :] < $(n),
        other=0.0).to(tl.float32)
      k = tl.load(K_block_ptr, mask=off_block2[:, None] < $(n),
        other=0.0).to(tl.float32)
      kv_trans_current += tl.dot(v_trans, k)
      K_block_ptr += $(CBLOCK) * $(d)
      V_trans_block_ptr += $(CBLOCK) * $(e)
      off_block2 += $(CBLOCK)
    }
    kv_trans += kv_trans_current
  }
  m = $(NUM_BLOCK) * $(BLOCK)
  off_block1 = m + tl.arange(0, $(CBLOCK))
  off_block2 = m + tl.arange(0, $(CBLOCK))
  Q_trans_block_ptr =
    Q + qk_offset + m * $(d) + tl.arange(0, $(CBLOCK))[None, :] * $(d) +
      tl.arange(0, $(d))[:, None]
  K_block_ptr =
    K + qk_offset + m * $(d) + tl.arange(0, $(CBLOCK))[:, None] * $(d) +
      tl.arange(0, $(d))[None, :]
  V_trans_block_ptr =
    V + v_offset + m * $(e) + tl.arange(0, $(CBLOCK))[None, :] * $(e) +
      tl.arange(0, $(e))[:, None]
  DK_trans_block_ptr =
    DK + qk_offset + m * $(d) + tl.arange(0, $(CBLOCK))[None, :] * $(d) +
      tl.arange(0, $(d))[:, None]
  DV_block_ptr =
    DV + v_offset + m * $(e) + tl.arange(0, $(CBLOCK))[:, None] * $(e) +
      tl.arange(0, $(e))[None, :]
  DO_block_ptr =
    DO + o_offset + m * $(e) + tl.arange(0, $(CBLOCK))[:, None] * $(e) +
      tl.arange(0, $(e))[None, :]
  dkv = tl.zeros([$(d), $(e)], dtype=tl.float32)
  for i in range($(NUM_BLOCK) - $(1), -$(1), -$(1)) {
    for j in range($(NUM_CBLOCK) - $(1), -$(1), -$(1)) {
      K_block_ptr -= $(CBLOCK) * $(d)
      V_trans_block_ptr -= $(CBLOCK) * $(e)
      DK_trans_block_ptr -= $(CBLOCK) * $(d)
      DV_block_ptr -= $(CBLOCK) * $(e)
      off_block1 -= $(CBLOCK)
      if i < $(NUM_BLOCK) - $(1) {
        k = tl.load(K_block_ptr, mask=off_block1[:, None] < $(n),
          other=0.0).to(tl.float32)
        v_trans = tl.load(V_trans_block_ptr, mask=off_block1[None, :] < $(n),
          other=0.0).to(tl.float32)
        dk_inter_trans = tl.dot(dkv, v_trans)
        dv_inter = tl.dot(k, dkv)
        dk_trans = dk_inter_trans + tl.load(DK_trans_block_ptr,
          mask=off_block1[None, :] < $(n), other=0.0)
        dv = dv_inter + tl.load(DV_block_ptr,
          mask=off_block1[:, None] < $(n), other=0.0)
        tl.store(DK_trans_block_ptr, (dk_trans).to(DK_trans_block_ptr.dtype.element_ty),
          mask=off_block1[None, :] < $(n))
        tl.store(DV_block_ptr, (dv).to(DV_block_ptr.dtype.element_ty),
          mask=off_block1[:, None] < $(n))
      }
    }
    dkv_current = tl.zeros([$(d), $(e)], dtype=tl.float32)
    for j in range($(NUM_CBLOCK) - $(1), -$(1), -$(1)) {
      DO_block_ptr -= $(CBLOCK) * $(e)
      Q_trans_block_ptr -= $(CBLOCK) * $(d)
      off_block2 -= $(CBLOCK)
      b_do = tl.load(DO_block_ptr, mask=off_block2[:, None] < $(n),
        other=0.0).to(tl.float32)
      q_trans = tl.load(Q_trans_block_ptr, mask=off_block2[None, :] < $(n),
        other=0.0).to(tl.float32)
      dkv_current += tl.dot(q_trans, b_do)
    }
    dkv += dkv_current
  }
}
```
</details>

<details><summary><code>lightning_attention_forward_kv_step_slice</code></summary>

```
/-! ### `kv`-update step slice (the per-block `kv += tl.dot(k_trans, v)` body)

This isolates the Python loop body's `kv` update from the cross-block loop
induction. It loads the materialized previous-state tile `KVPrev`, the
block's `k_trans`/`v` tiles, forms the per-block outer product `tl.dot(k_trans,
v)`, and stores `KVPrev + tl.dot(k_trans, v)` into a state buffer `KVOut` at the
canonical `[d, BLOCK_MODEL]` layout. -/
```
```lean
def lightning_attention_forward_kv_step_slice
    (KVPrev KTrans V KVOut : RegionName) (D BLOCK BLOCK_MODEL : Nat) :
    ComputeKernel := triton {
  offs_a = tl.arange(0, $(D))
  offs_j = tl.arange(0, $(BLOCK))
  offs_c = tl.arange(0, $(BLOCK_MODEL))
  prev = tl.load(KVPrev + offs_a[:, None] * $(BLOCK_MODEL) + offs_c[None, :])
  k_trans = tl.load(KTrans + offs_a[:, None] * $(BLOCK) + offs_j[None, :])
  v = tl.load(V + offs_j[:, None] * $(BLOCK_MODEL) + offs_c[None, :])
  kv_update = tl.dot(k_trans, v)
  kv = prev + kv_update
  tl.store(KVOut + offs_a[:, None] * $(BLOCK_MODEL) + offs_c[None, :], kv)
}
```
</details>

<details><summary><code>kvStepSpec</code></summary>

```
/-- The arithmetic spec of one `kv`-update body: `KVPrev[a,c] + Σ_j
k_trans[a,j]·v[j,c]`, i.e. the materialized previous state plus the block's
`tl.dot(k_trans, v)` outer product. -/
```
```lean
noncomputable def kvStepSpec (s : BlockState) (KVPrev KTrans V : RegionName)
    (D BLOCK BLOCK_MODEL : Nat) (idx : TileIndex [D, BLOCK_MODEL]) : ℝ :=
  s.readMem KVPrev (kvOffset BLOCK_MODEL idx) +
    ∑ j : Fin BLOCK,
      tileElem s KTrans BLOCK idx.1.val j.val *
        tileElem s V BLOCK_MODEL j.val idx.2.1.val
```
</details>

<details><summary><code>lightning_attention_forward_o_inter_dot_slice</code></summary>

```
/-! ### `o_inter` producer slice (the per-block `o_inter = tl.dot(q, kv)` body)

The inter-block contribution to the output: with the carried state `kv` loaded
from a materialized `KVPrev` tile, `o_inter[r, c] = Σ_a q[r,a]·kv[a,c]`. This is
the `tl.dot(q, kv)` producer; under the carry invariant `KVPrev = kvClosed m`
its value is exactly `Σ_a q[r,a]·(Σ_{s<m·BLOCK} k[s,a]·v[s,c])`, the inter-block
half of the causal linear-attention output row. -/
```
```lean
def lightning_attention_forward_o_inter_dot_slice
    (Q KVPrev OInter : RegionName) (BLOCK D BLOCK_MODEL : Nat) :
    ComputeKernel := triton {
  offs_r = tl.arange(0, $(BLOCK))
  offs_a = tl.arange(0, $(D))
  offs_c = tl.arange(0, $(BLOCK_MODEL))
  q = tl.load(Q + offs_r[:, None] * $(D) + offs_a[None, :])
  kv = tl.load(KVPrev + offs_a[:, None] * $(BLOCK_MODEL) + offs_c[None, :])
  o_inter = tl.dot(q, kv)
  tl.store(OInter + offs_r[:, None] * $(BLOCK_MODEL) + offs_c[None, :], o_inter)
}
```
</details>

<details><summary><code>oInterOffset</code></summary>

```
/-- Flat `[BLOCK, BLOCK_MODEL]` output-tile offset `r · BLOCK_MODEL + c`. -/
```
```lean
def oInterOffset (BLOCK_MODEL : Nat) (idx : TileIndex [BLOCK, BLOCK_MODEL]) : Nat :=
  idx.1.val * BLOCK_MODEL + idx.2.1.val
```
</details>

<details><summary><code>oInterDotSpec</code></summary>

```
/-- The genuine arithmetic spec of `o_inter = tl.dot(q, kv)`, reading the
materialized previous-state tile `KVPrev`: `Σ_a q[r,a]·KVPrev[a,c]`. -/
```
```lean
noncomputable def oInterDotSpec (s : BlockState) (Q KVPrev : RegionName)
    (BLOCK D BLOCK_MODEL : Nat) (idx : TileIndex [BLOCK, BLOCK_MODEL]) : ℝ :=
  ∑ a : Fin D,
    tileElem s Q D idx.1.val a.val *
      tileElem s KVPrev BLOCK_MODEL a.val idx.2.1.val
```
</details>

<details><summary><code>lightning_attention_forward_body_slice</code></summary>

```
/-! ### ════════ The full forward loop body (causal mask + `Out` writeback) ════════

This is the whole `_fwd_kernel` loop body at a fixed block index `m`, over the
**launched** regions `Q`, `K`, `V`, `Out` at their real addresses: the three
masked tile loads, `qk = tl.dot(q, k_trans)`, the causal
`qk = tl.where(index >= 0, qk, 0)`, `o_intra = tl.dot(qk, v)`,
`o_inter = tl.dot(q, kv)`, `o = o_intra + o_inter`, and the masked
`tl.store` into `Out`. Only the carried `kv` register is presented as a
materialized buffer `KVPrev` (it is a register, so it has no address).

Two modelling points that must be read before trusting the face below.

* **The causal predicate is re-spelled, not re-interpreted.** Python computes
  `index = off_block[:, None] - off_block[None, :]` in *signed* int32 and tests
  `index >= 0`. This model's `-` on `.nat` tiles truncates at zero, which would
  make `index >= 0` a tautology — so the slice spells the same predicate as the
  direct comparison `offs_r[:, None] >= offs_r[None, :]`, which is equal to
  `r - j ≥ 0` over ℤ for the non-negative lane indices involved. (Note that
  `lightning_attention_forward_surface` keeps the *literal* Python spelling, and
  therefore models that mask as a no-op; the surface carries only the lowering
  claim, never a semantic one, but a reader comparing the two should know why they
  differ. Note also that Python computes `index` **once, before** the loop, from
  `off_block = tl.arange(0, BLOCK)`, so the predicate is block-relative `j ≤ r`
  and does not drift as `off_block += BLOCK`.)
* **Every block, tail included.** The face is stated per written lane
  (`writeIf (m·BLOCK + r < n)`), so it covers the partial tail block too. On a
  written row the three load masks are satisfied wherever the value depends on
  them: `j ≤ r` forces `m·BLOCK + j ≤ m·BLOCK + r < n`, and the columns where a
  mask could fire have already been zeroed by `tl.where`. Rows outside the
  sequence are not written at all, and the face says nothing about them —
  which is exactly what the kernel does. -/
```
```lean
def lightning_attention_forward_body_slice
    (Q K V KVPrev Out : RegionName)
    (m n d e BLOCK BLOCK_MODEL : Nat) : ComputeKernel := triton {
  off_bh = tl.program_id(0)
  off_e = tl.program_id(1)
  qk_offset = off_bh * $(n) * $(d)
  v_offset = off_bh * $(n) * $(e)
  o_offset = off_bh * $(n) * $(e)
  e_offset = off_e * $(BLOCK_MODEL)
  offs_a = tl.arange(0, $(d))
  offs_c = tl.arange(0, $(BLOCK_MODEL))
  offs_r = tl.arange(0, $(BLOCK))
  off_block = $(m) * $(BLOCK) + tl.arange(0, $(BLOCK))
  q = tl.load(Q + qk_offset + off_block[:, None] * $(d) + offs_a[None, :],
    mask=off_block[:, None] < $(n), other=0.0)
  k_trans = tl.load(K + qk_offset + off_block[None, :] * $(d) + offs_a[:, None],
    mask=off_block[None, :] < $(n), other=0.0)
  v = tl.load(V + v_offset + off_block[:, None] * $(e) + e_offset + offs_c[None, :],
    mask=off_block[:, None] < $(n), other=0.0)
  kv = tl.load(KVPrev + offs_a[:, None] * $(BLOCK_MODEL) + offs_c[None, :])
  qk = tl.dot(q, k_trans)
  qk = tl.where(offs_r[:, None] >= offs_r[None, :], qk, 0.0)
  o_intra = tl.dot(qk, v)
  o_inter = tl.dot(q, kv)
  o = o_intra + o_inter
  tl.store(Out + o_offset + off_block[:, None] * $(e) + e_offset + offs_c[None, :],
    (o).to(Out.dtype.element_ty), mask=off_block[:, None] < $(n))
}
```
</details>

<details><summary><code>bodyOutOffset</code></summary>

```
/-- The kernel's store address for block-`m` lane `(r, c)`:
`o_offset + (m·BLOCK + r)·e + e_offset + c`, i.e. global time row `m·BLOCK + r`,
value channel `off_e·BLOCK_MODEL + c`. -/
```
```lean
def bodyOutOffset (s : BlockState) (m n e BLOCK BLOCK_MODEL : Nat)
    (idx : TileIndex [BLOCK, BLOCK_MODEL]) : Nat :=
  s.pids 0 * n * e + (m * BLOCK + idx.1.val) * e + s.pids 1 * BLOCK_MODEL +
    idx.2.1.val
```
</details>

<details><summary><code>fwdOutClosed</code></summary>

```
/-- **Genuine closed form for the launched output row** — the causal linear
attention `_fwd_kernel` computes, at global time row `t` and value channel
`off_e·BLOCK_MODEL + c`:

```
o[t, c] = Σ_{s ≤ t} (Σ_a q[t,a]·k[s,a]) · v[s,c].
```

A standalone specification over the input regions `Q`, `K`, `V` — never a
read-back of the kernel's own output. -/
```
```lean
noncomputable def fwdOutClosed (s : BlockState) (Q K V : RegionName)
    (n d e BLOCK_MODEL : Nat) (t : Nat) (c : Nat) : ℝ :=
  ∑ keyRow ∈ Finset.range (t + 1),
    (∑ a : Fin d, fwdKVal s Q n d a.val t * fwdKVal s K n d a.val keyRow) *
      fwdVVal s V n e BLOCK_MODEL c keyRow
```
</details>

<details><summary><code>tileElem</code></summary>

```
/-- Row-major staged-tile element `R[i, j]` of a flat `[rows, width]` scratch
region (offset `i·width + j`): the layout of every materialized per-block tile
(`Q`, `KTrans`, `V`, `KVPrev`) that the proof slices exchange. -/
```
```lean
noncomputable def tileElem (s : BlockState) (R : RegionName)
    (width i j : Nat) : ℝ :=
  s.readMem R (i * width + j)
```
</details>

## Public theorem: `lightning_attention_bwd_grad_store_io_correctness`

<details><summary>docstring</summary>

```
/-- **The headline on the IO surface** for `lightning_attention.py`'s backward
gradient writeback: for every disjoint flat placement of `GradPre` / `Out`, every
program coordinate whose active lanes are in bounds, and every launch state whose
`GradPre` block holds `xs` on the row-active lanes, the kernel terminates, every
active lane of `Out` holds `xs idx`, and every other memory cell is unchanged.

Dimension-general in `n`, `width`, `BLOCK` and `WIDTH`. Honest side-condition:
address injectivity at every program coordinate, the same hypothesis the
per-write-map summary takes (the store's flat address is not injective for every
choice of `n`/`width`/`BLOCK`, so it cannot be discharged inline). -/
```
</details>

**Statement:**
```lean
specification lightning_attention_bwd_grad_store_io_correctness
    (GradPre Out : RegionName) (n width BLOCK WIDTH : Nat)
    (hOutInj : ∀ p₀ p₁ : Nat, Function.Injective
      (fun idx : TileIndex [BLOCK, WIDTH] =>
        p₀ * n * width + (p₁ * BLOCK + idx.1.val) * width + idx.2.1.val)) :
    gradStoreIO GradPre Out n width BLOCK WIDTH
      ⊨ fun _p₀ _p₁ xs idx => xs idx
```

**Assumptions / layout contracts:**
- `fun idx : TileIndex [BLOCK, WIDTH] =>
        p₀ * n * width + (p₁ * BLOCK + idx.1.val) * width + idx.2.1.val`

**Closed-form spec defs (transitive):** `gradStoreIO`, `lightning_attention_bwd_grad_store_slice`

<details><summary><code>gradStoreIO</code></summary>

```
/-- IO signature of the backward gradient store. `WIDTH` is the full column
extent, so the column half of the Python mask is vacuous and only the row bound
survives into `mask`. The third program axis is unused. -/
```
```lean
def gradStoreIO (GradPre Out : RegionName) (n width BLOCK WIDTH : Nat) :
    Masked3DTileKernelIO₁ where
  kernel := lightning_attention_bwd_grad_store_slice GradPre Out n width BLOCK
    WIDTH
  inp := GradPre
  out := Out
  shape := [BLOCK, WIDTH]
  read := fun p₀ p₁ _p₂ idx =>
    p₀ * n * width + (p₁ * BLOCK + idx.1.val) * width + idx.2.1.val
  write := fun p₀ p₁ _p₂ idx =>
    p₀ * n * width + (p₁ * BLOCK + idx.1.val) * width + idx.2.1.val
  mask := fun _p₀ p₁ _p₂ idx => p₁ * BLOCK + idx.1.val < n
```
</details>

<details><summary><code>lightning_attention_bwd_grad_store_slice</code></summary>

```
/-- Proof-oriented backward gradient tile-store slice of
`lightning_attention.py`.

The Python backward path writes `DQ`, `DK`, and `DV` from intra- and inter-block
accumulators. This generic slice fixes one block row and proves the masked
writeback from a precomputed gradient tile into any of those output regions. -/
```
```lean
def lightning_attention_bwd_grad_store_slice
    (GradPre Out : RegionName) (n width BLOCK WIDTH : Nat) :
    ComputeKernel := triton {
  off_bh = tl.program_id(0)
  off_block = tl.program_id(1) * $(BLOCK) + tl.arange(0, $(BLOCK))
  offs_w = tl.arange(0, $(WIDTH))
  mask = (off_block[:, None] < $(n)) & (offs_w[None, :] < $(WIDTH))
  grad = tl.load(GradPre + off_bh * $(n) * $(width) +
      off_block[:, None] * $(width) + offs_w[None, :],
      mask=mask, other=0.0)
  tl.store(Out + off_bh * $(n) * $(width) +
      off_block[:, None] * $(width) + offs_w[None, :],
      (grad).to(Out.dtype.element_ty), mask=mask)
}
```
</details>

## Public theorem: `lightning_attention_bwd_grad_store_io_correctnessR`

<details><summary>docstring</summary>

```
/-- **The `⊨[R]` headline** for the backward gradient store: for **every**
rounding model `R`, the same masked Hoare triple as
`lightning_attention_bwd_grad_store_io_correctness`, but run under `execR R` and
read back as `.real`-typed cells holding `R.round .real (xs idx)`.

The writeback is a pure copy — no arithmetic, and the store's `.to(...)` erases to
`.real` — so the slice is cast-free and the exact run transports verbatim. The
content of the rounding face here is exactly that: *this kernel introduces no
rounding event of its own*, at any `R`. -/
```
</details>

**Statement:**
```lean
specification lightning_attention_bwd_grad_store_io_correctnessR
    (R : RoundingModel) (GradPre Out : RegionName) (n width BLOCK WIDTH : Nat)
    (hOutInj : ∀ p₀ p₁ : Nat, Function.Injective
      (fun idx : TileIndex [BLOCK, WIDTH] =>
        p₀ * n * width + (p₁ * BLOCK + idx.1.val) * width + idx.2.1.val)) :
    gradStoreIO GradPre Out n width BLOCK WIDTH
      ⊨[R, FloatDType.real] fun _p₀ _p₁ xs idx => xs idx
```

**Assumptions / layout contracts:**
- `fun idx : TileIndex [BLOCK, WIDTH] =>
        p₀ * n * width + (p₁ * BLOCK + idx.1.val) * width + idx.2.1.val`

**Closed-form spec defs (transitive):** `gradStoreIO`, `lightning_attention_bwd_grad_store_slice`

<details><summary><code>gradStoreIO</code></summary>

```
/-- IO signature of the backward gradient store. `WIDTH` is the full column
extent, so the column half of the Python mask is vacuous and only the row bound
survives into `mask`. The third program axis is unused. -/
```
```lean
def gradStoreIO (GradPre Out : RegionName) (n width BLOCK WIDTH : Nat) :
    Masked3DTileKernelIO₁ where
  kernel := lightning_attention_bwd_grad_store_slice GradPre Out n width BLOCK
    WIDTH
  inp := GradPre
  out := Out
  shape := [BLOCK, WIDTH]
  read := fun p₀ p₁ _p₂ idx =>
    p₀ * n * width + (p₁ * BLOCK + idx.1.val) * width + idx.2.1.val
  write := fun p₀ p₁ _p₂ idx =>
    p₀ * n * width + (p₁ * BLOCK + idx.1.val) * width + idx.2.1.val
  mask := fun _p₀ p₁ _p₂ idx => p₁ * BLOCK + idx.1.val < n
```
</details>

<details><summary><code>lightning_attention_bwd_grad_store_slice</code></summary>

```
/-- Proof-oriented backward gradient tile-store slice of
`lightning_attention.py`.

The Python backward path writes `DQ`, `DK`, and `DV` from intra- and inter-block
accumulators. This generic slice fixes one block row and proves the masked
writeback from a precomputed gradient tile into any of those output regions. -/
```
```lean
def lightning_attention_bwd_grad_store_slice
    (GradPre Out : RegionName) (n width BLOCK WIDTH : Nat) :
    ComputeKernel := triton {
  off_bh = tl.program_id(0)
  off_block = tl.program_id(1) * $(BLOCK) + tl.arange(0, $(BLOCK))
  offs_w = tl.arange(0, $(WIDTH))
  mask = (off_block[:, None] < $(n)) & (offs_w[None, :] < $(WIDTH))
  grad = tl.load(GradPre + off_bh * $(n) * $(width) +
      off_block[:, None] * $(width) + offs_w[None, :],
      mask=mask, other=0.0)
  tl.store(Out + off_bh * $(n) * $(width) +
      off_block[:, None] * $(width) + offs_w[None, :],
      (grad).to(Out.dtype.element_ty), mask=mask)
}
```
</details>

## Also present (pinned special-case summaries)
- `lightning_attention_forward_kv_step_slice_compute_correct`
- `lightning_attention_forward_o_inter_dot_slice_compute_correct`
- `lightning_attention_bwd_grad_store_slice_compute_correct`
- `lightning_attention_bwd_dq_accum_store_slice_compute_correct`
- `lightning_attention_bwd_dq_inter_dot_slice_compute_correct`
- `lightning_attention_bwd_dq_store_slice_compute_correct`
- `lightning_attention_bwd_dk_store_slice_compute_correct`
- `lightning_attention_bwd_dv_store_slice_compute_correct`
