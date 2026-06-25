# Spec sheet — `bench/tritonbench_g/triton_attention/TritonAttention.lean`

**Python source:** `bench/tritonbench_g/triton_attention/triton_attention.py`

## Public theorem: `triton_attention_bwd_preprocess_genuine_output_summary_general`

<details><summary>docstring</summary>

```
/-- **Genuine faithful backward-preprocess summary (dimension-general).** For
*symbolic* `BLOCK_M`, `D_HEAD` and any program id, every output lane of the full
`_bwd_preprocess` surface holds its genuine Python closed form, stated purely
over the **input** regions `Out`/`DO`/`L`:

* `NewDO[i,d] = DO[i,d] / L[i]` (`bwdPreprocessNewDOSpecG`);
* `Delta[i]  = Σ_{d} O[i,d] · (DO[i,d] / L[i])` (`bwdPreprocessDeltaSpecG`).

Honest side conditions only: `NewDO ≠ Delta` (the two stores hit distinct
regions, so the `Delta` store cannot clobber the `NewDO` readback) and
injectivity of the `NewDO` tile offset map (`hOutInj`). The expected values are
genuine input-memory closed forms, **not** a self-referential executed readback. -/
```
</details>

**Statement:**
```lean
theorem triton_attention_bwd_preprocess_genuine_output_summary_general
    (Out DO L NewDO Delta : RegionName) (BLOCK_M D_HEAD : Nat) (s : BlockState)
    (hND : NewDO ≠ Delta)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, D_HEAD] =>
        newdoOffset s BLOCK_M D_HEAD idx)) :
    (∃ alg, (triton_attention_bwd_preprocess Out DO L NewDO Delta
      BLOCK_M D_HEAD).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_preprocess Out DO L NewDO Delta
        BLOCK_M D_HEAD)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, D_HEAD] =>
        some (NewDO, newdoOffset s BLOCK_M D_HEAD idx))
      (expected := fun idx : TileIndex [BLOCK_M, D_HEAD] =>
        bwdPreprocessNewDOSpecG s Out DO L BLOCK_M D_HEAD idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_preprocess Out DO L NewDO Delta
        BLOCK_M D_HEAD)
      (initialState := s)
      (write := fun i : Fin BLOCK_M => some (Delta, deltaOffset s BLOCK_M i))
      (expected := fun i : Fin BLOCK_M =>
        bwdPreprocessDeltaSpecG s Out DO L BLOCK_M D_HEAD i))
```

**Assumptions / layout contracts:**
- `hND : NewDO ≠ Delta`
- `hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, D_HEAD] =>
        newdoOffset s BLOCK_M D_HEAD idx)`

**Closed-form spec defs (transitive):** `newdoOffset`, `triton_attention_bwd_preprocess`, `bwdPreprocessNewDOSpecG`, `deltaOffset`, `bwdPreprocessDeltaSpecG`, `newdoMIndex`, `newdoNIndex`

<details><summary><code>newdoOffset</code></summary>

```lean
def newdoOffset (s : BlockState) (BLOCK_M D_HEAD : Nat)
    (idx : TileIndex [BLOCK_M, D_HEAD]) : Nat :=
  newdoMIndex s BLOCK_M idx.1 * D_HEAD + newdoNIndex idx
```
</details>

<details><summary><code>triton_attention_bwd_preprocess</code></summary>

```
/-- DSL port of `triton_attention.py`'s `_bwd_preprocess`. -/
```
```lean
def triton_attention_bwd_preprocess
    (Out DO L NewDO Delta : RegionName) (BLOCK_M D_HEAD : Nat) :
    ComputeKernel := triton {
  off_m = tl.program_id(0) * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  off_n = tl.arange(0, $(D_HEAD))
  o = (tl.load(Out + off_m[:, None] * $(D_HEAD) + off_n[None, :])).to(tl.float32)
  do_val = (tl.load(DO + off_m[:, None] * $(D_HEAD) + off_n[None, :])).to(tl.float32)
  denom = (tl.load(L + off_m)).to(tl.float32)
  do_val = do_val / denom[:, None]
  delta = tl.sum(o * do_val, axis=1)
  tl.store(NewDO + off_m[:, None] * $(D_HEAD) + off_n[None, :], do_val)
  tl.store(Delta + off_m, delta)
}
```
</details>

<details><summary><code>bwdPreprocessNewDOSpecG</code></summary>

```lean
noncomputable def bwdPreprocessNewDOSpecG (s : BlockState) (_O DO L : RegionName)
    (BLOCK_M D_HEAD : Nat) (idx : TileIndex [BLOCK_M, D_HEAD]) : ℝ :=
  s.readMem DO (newdoOffset s BLOCK_M D_HEAD idx) /
    s.readMem L (newdoMIndex s BLOCK_M idx.1)
```
</details>

<details><summary><code>deltaOffset</code></summary>

```lean
def deltaOffset (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val
```
</details>

<details><summary><code>bwdPreprocessDeltaSpecG</code></summary>

```lean
noncomputable def bwdPreprocessDeltaSpecG (s : BlockState) (O DO L : RegionName)
    (BLOCK_M D_HEAD : Nat) (i : Fin BLOCK_M) : ℝ :=
  ∑ j : Fin D_HEAD,
    let idx : TileIndex [BLOCK_M, D_HEAD] :=
      TileShape.insertAxisIndex [BLOCK_M, D_HEAD] 1
        (TileShape.insertAxisIndex [BLOCK_M] 0 PUnit.unit i) j
    s.readMem O (newdoOffset s BLOCK_M D_HEAD idx) *
      (s.readMem DO (newdoOffset s BLOCK_M D_HEAD idx) /
        s.readMem L (newdoMIndex s BLOCK_M idx.1))
```
</details>

<details><summary><code>newdoMIndex</code></summary>

```lean
def newdoMIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val
```
</details>

<details><summary><code>newdoNIndex</code></summary>

```lean
def newdoNIndex (idx : TileIndex [BLOCK_M, D_HEAD]) : Nat :=
  idx.2.1.val
```
</details>

## Public theorem: `triton_attention_forward_output_summary_general`

<details><summary>docstring</summary>

```
/-- **★ MAIN (forward, dimension-general).** Public symbolic-dimension forward
output summary for `triton_attention.py`'s `_fwd_kernel`. For arbitrary symbolic
`N_CTX`/`D_HEAD = BLOCK_DMODEL`/`BLOCK_M`/`BLOCK_N` and a contiguous layout, the
lowered kernel's `Out`/`L`/`M` writes Realize the genuine general closed-form
specs `fwdOutSpecG`/`fwdLSpecG`/`fwdMSpecG` — defined purely over the **input**
`Q`/`K`/`V` memory (causal natural-exp attention over the streaming KV span
`SEQ = (pids0+1)·BLOCK_M`), never over the kernel's own readback. Honest side
conditions: positive block dims, `BLOCK_N ∣ (pids0+1)·BLOCK_M`, contiguous
strides, output-offset injectivity, and the boundary
`pids1·(stride_qh/BLOCK_DMODEL) + (pids0+1)·BLOCK_M ≤ D0`. -/
```
</details>

**Statement:**
```lean
theorem triton_attention_forward_output_summary_general
    (Q K V L M Out : RegionName) (s : BlockState) (sc : ℝ)
    (stride_qz stride_qh Z H N_CTX D0 BLOCK_M BLOCK_DMODEL BLOCK_N : Nat)
    (hBM : 0 < BLOCK_M) (hBN : 0 < BLOCK_N) (hBD : 0 < BLOCK_DMODEL)
    (hdvd : BLOCK_N ∣ (s.pids 0 + 1) * BLOCK_M)
    (hbound : s.pids 1 * (stride_qh / BLOCK_DMODEL) + (s.pids 0 + 1) * BLOCK_M ≤ D0)
    (hLOut : L ≠ Out) (hMOut : M ≠ Out) (hLM : M ≠ L)
    (houtinj : Function.Injective (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        (s.pids 1 * (stride_qh / BLOCK_DMODEL) + s.pids 0 * BLOCK_M + idx.1.val) * BLOCK_DMODEL + idx.2.1.val * 1))
    (hundef : ∀ rg o, s.undef rg o = 0) :
    (∃ alg, (triton_attention_fwd_kernel Q K V L M Out sc
      stride_qz stride_qh BLOCK_DMODEL 1 stride_qz stride_qh BLOCK_DMODEL 1
      stride_qz stride_qh BLOCK_DMODEL 1 stride_qz stride_qh BLOCK_DMODEL 1
      Z H N_CTX D0 BLOCK_M BLOCK_DMODEL BLOCK_N).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_fwd_kernel Q K V L M Out sc
        stride_qz stride_qh BLOCK_DMODEL 1 stride_qz stride_qh BLOCK_DMODEL 1
        stride_qz stride_qh BLOCK_DMODEL 1 stride_qz stride_qh BLOCK_DMODEL 1
        Z H N_CTX D0 BLOCK_M BLOCK_DMODEL BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          active s (s.pids 1 * (stride_qh / BLOCK_DMODEL)) D0 BLOCK_M idx)
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          (Out, outOffset s (s.pids 1 * (stride_qh / BLOCK_DMODEL)) BLOCK_DMODEL 1 BLOCK_M idx)))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16
          (some (fwdOutSpecG s Q K V (stride_qh / BLOCK_DMODEL) BLOCK_DMODEL
              ((s.pids 0 + 1) * BLOCK_M) BLOCK_M BLOCK_DMODEL sc idx))))) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_fwd_kernel Q K V L M Out sc
        stride_qz stride_qh BLOCK_DMODEL 1 stride_qz stride_qh BLOCK_DMODEL 1
        stride_qz stride_qh BLOCK_DMODEL 1 stride_qz stride_qh BLOCK_DMODEL 1
        Z H N_CTX D0 BLOCK_M BLOCK_DMODEL BLOCK_N)
      (initialState := s)
      (write := fun i : Fin BLOCK_M => some (L, lRowOffset s (s.pids 1) N_CTX BLOCK_M i))
      (expected := fun i : Fin BLOCK_M =>
        fwdLSpecG s Q K (stride_qh / BLOCK_DMODEL) BLOCK_DMODEL ((s.pids 0 + 1) * BLOCK_M) BLOCK_M BLOCK_DMODEL sc
          (Nat.mul_pos (Nat.succ_pos _) hBM) i)) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_fwd_kernel Q K V L M Out sc
        stride_qz stride_qh BLOCK_DMODEL 1 stride_qz stride_qh BLOCK_DMODEL 1
        stride_qz stride_qh BLOCK_DMODEL 1 stride_qz stride_qh BLOCK_DMODEL 1
        Z H N_CTX D0 BLOCK_M BLOCK_DMODEL BLOCK_N)
      (initialState := s)
      (write := fun i : Fin BLOCK_M => some (M, lRowOffset s (s.pids 1) N_CTX BLOCK_M i))
      (expected := fun i : Fin BLOCK_M =>
        fwdMSpecG s Q K (stride_qh / BLOCK_DMODEL) BLOCK_DMODEL ((s.pids 0 + 1) * BLOCK_M) BLOCK_M BLOCK_DMODEL sc
          (Nat.mul_pos (Nat.succ_pos _) hBM) i))
```

**Assumptions / layout contracts:**
- `hBM : 0 < BLOCK_M`
- `hBN : 0 < BLOCK_N`
- `hBD : 0 < BLOCK_DMODEL`
- `hdvd : BLOCK_N ∣ (s.pids 0 + 1) * BLOCK_M`
- `hbound : s.pids 1 * (stride_qh / BLOCK_DMODEL) + (s.pids 0 + 1) * BLOCK_M ≤ D0`
- `hLOut : L ≠ Out`
- `hMOut : M ≠ Out`
- `hLM : M ≠ L`
- `hundef : ∀ rg o, s.undef rg o = 0`

**Closed-form spec defs (transitive):** `triton_attention_fwd_kernel`, `active`, `outOffset`, `fwdOutSpecG`, `lRowOffset`, `fwdLSpecG`, `fwdMSpecG`, `rowIndex`, `dIndex`, `fwdQTileG`, `fwdKTileG`, `fwdVTileG`, `fwdCausalSetG`

<details><summary><code>triton_attention_fwd_kernel</code></summary>

```
/-- DSL port of `triton_attention.py`'s `_fwd_kernel`. -/
```
```lean
def triton_attention_fwd_kernel
    (Q K V L M Out : RegionName)
    (sm_scale : ℝ)
    (_stride_qz stride_qh stride_qm stride_qk
      _stride_kz _stride_kh stride_kn stride_kk
      _stride_vz _stride_vh stride_vk stride_vn
      _stride_oz _stride_oh stride_om stride_on
      _Z _H N_CTX D0 BLOCK_M BLOCK_DMODEL BLOCK_N : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)

  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_n = tl.arange(0, $(BLOCK_N))
  m_prev = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_prev = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)

  stride_qh_2d = $(stride_qh) // $(stride_qm) // $(stride_qk)

  q_tile_ptr = tl.make_block_ptr(base=Q,
    shape=($(D0), $(BLOCK_DMODEL)),
    strides=($(stride_qm), $(stride_qk)),
    offsets=(off_hz * stride_qh_2d + start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
    order=(1, 0))
  k_tile_ptr = tl.make_block_ptr(base=K,
    shape=($(D0), $(BLOCK_DMODEL)),
    strides=($(stride_kn), $(stride_kk)),
    offsets=(off_hz * stride_qh_2d, 0),
    block_shape=($(BLOCK_N), $(BLOCK_DMODEL)),
    order=(1, 0))
  v_tile_ptr = tl.make_block_ptr(base=V,
    shape=($(D0), $(BLOCK_DMODEL)),
    strides=($(stride_vk), $(stride_vn)),
    offsets=(off_hz * stride_qh_2d, 0),
    block_shape=($(BLOCK_N), $(BLOCK_DMODEL)),
    order=(1, 0))
  out_tile_ptr = tl.make_block_ptr(base=Out,
    shape=($(D0), $(BLOCK_DMODEL)),
    strides=($(stride_om), $(stride_on)),
    offsets=(off_hz * stride_qh_2d + start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
    order=(1, 0))
  q = tl.load(q_tile_ptr)

  for start_n in range($(0), (start_m + $(1)) * $(BLOCK_M), $(BLOCK_N)) {
    k = tl.load(k_tile_ptr, boundary_check=(0, 1))
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, tl.trans(k))
    qk *= $((sm_scale : ℝ))
    qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
    m_curr = tl.maximum(tl.max(qk, 1), m_prev)
    l_prev *= tl.exp(m_prev - m_curr)
    p = tl.exp(qk - m_curr[:, None])
    l_curr = tl.sum(p, 1) + l_prev
    l_rcp = 1.0 / l_curr
    p *= l_rcp[:, None]
    acc *= (l_prev * l_rcp)[:, None]
    p = (p).to(tl.float16)
    v = tl.load(v_tile_ptr, boundary_check=(0, 1))
    acc += tl.dot(p, v)
    l_prev = l_curr
    m_prev = m_curr
    k_tile_ptr = tl.advance(k_tile_ptr, [$(BLOCK_N), $(0)])
    v_tile_ptr = tl.advance(v_tile_ptr, [$(BLOCK_N), $(0)])
  }
  start_m = tl.program_id(0)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  l_ptrs = L + off_hz * $(N_CTX) + offs_m
  m_ptrs = M + off_hz * $(N_CTX) + offs_m
  tl.store(l_ptrs, l_prev)
  tl.store(m_ptrs, m_prev)

  acc = (acc).to(tl.float16)
  tl.store(out_tile_ptr, acc, boundary_check=(0, 1))
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active (s : BlockState) (hzRowOffset D0 BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Prop :=
  rowIndex s BLOCK_M idx.1 + hzRowOffset < D0
```
</details>

<details><summary><code>outOffset</code></summary>

```lean
def outOffset (s : BlockState) (hzRowOffset stride_om stride_on BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  (rowIndex s BLOCK_M idx.1 + hzRowOffset) * stride_om + dIndex idx * stride_on
```
</details>

<details><summary><code>fwdOutSpecG</code></summary>

```
/-- General genuine closed-form forward `Out`: the natural-exp causal attention
block at query start `pids0·BLOCK_M`, scale `sc`, over the loaded key span `SEQ`.
Expressed via `attentionRealCausalBlock`, independent of the kernel `exec`. -/
```
```lean
noncomputable def fwdOutSpecG
    (s : BlockState) (Q K V : RegionName)
    (stride_hz_2d HEAD_DIM SEQ BLOCK_M BLOCK_DMODEL : Nat)
    (sc : ℝ) (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : ℝ :=
  attentionRealCausalBlock (s.pids 0 * BLOCK_M)
    (fwdQTileG s Q stride_hz_2d HEAD_DIM BLOCK_M BLOCK_DMODEL)
    (fwdKTileG s K stride_hz_2d HEAD_DIM SEQ BLOCK_DMODEL)
    (fwdVTileG s V stride_hz_2d HEAD_DIM SEQ BLOCK_DMODEL)
    sc idx
```
</details>

<details><summary><code>lRowOffset</code></summary>

```lean
def lRowOffset (s : BlockState) (off_hz N_CTX BLOCK_M : Nat)
    (i : Fin BLOCK_M) : Nat :=
  off_hz * N_CTX + (s.pids 0 * BLOCK_M + i.val)
```
</details>

<details><summary><code>fwdLSpecG</code></summary>

```
/-- General genuine closed-form forward `L` for query row `i`: the **m-shifted**
causal softmax normalizer `Σ_{j ≤ pids0·BLOCK_M + i} exp(score i j − M_row)` over
the loaded span `SEQ`, with `M_row = fwdMSpecG`. This is exactly the value the
kernel stores (the running m-shifted `l_prev`); the un-shifted log-sum-exp is
recovered with the separately stored `M`. -/
```
```lean
noncomputable def fwdLSpecG
    (s : BlockState) (Q K : RegionName)
    (stride_hz_2d HEAD_DIM SEQ BLOCK_M BLOCK_DMODEL : Nat) (sc : ℝ)
    (hSEQ : 0 < SEQ) (i : Fin BLOCK_M) : ℝ :=
  Finset.univ.sum (fun j : Fin SEQ =>
    if j.val ≤ s.pids 0 * BLOCK_M + i.val then
      Real.exp (scaledScore (fwdQTileG s Q stride_hz_2d HEAD_DIM BLOCK_M BLOCK_DMODEL)
        (fwdKTileG s K stride_hz_2d HEAD_DIM SEQ BLOCK_DMODEL) sc i j
        - fwdMSpecG s Q K stride_hz_2d HEAD_DIM SEQ BLOCK_M BLOCK_DMODEL sc hSEQ i)
    else 0)
```
</details>

<details><summary><code>fwdMSpecG</code></summary>

```
/-- General genuine closed-form forward `M` for query row `i`: the per-row maximum
causal score `max_{j ≤ pids0·BLOCK_M + i} score i j` over the (nonempty) causal
key set within the loaded span `SEQ`. -/
```
```lean
noncomputable def fwdMSpecG
    (s : BlockState) (Q K : RegionName)
    (stride_hz_2d HEAD_DIM SEQ BLOCK_M BLOCK_DMODEL : Nat) (sc : ℝ)
    (hSEQ : 0 < SEQ) (i : Fin BLOCK_M) : ℝ :=
  (fwdCausalSetG s SEQ BLOCK_M i).sup' (fwdCausalSetG_nonempty s SEQ BLOCK_M hSEQ i)
    (fun j : Fin SEQ =>
      scaledScore (fwdQTileG s Q stride_hz_2d HEAD_DIM BLOCK_M BLOCK_DMODEL)
        (fwdKTileG s K stride_hz_2d HEAD_DIM SEQ BLOCK_DMODEL) sc i j)
```
</details>

<details><summary><code>rowIndex</code></summary>

```lean
def rowIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val
```
</details>

<details><summary><code>dIndex</code></summary>

```lean
def dIndex (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  idx.2.1.val
```
</details>

<details><summary><code>fwdQTileG</code></summary>

```
/-- General Q tile: output tile-row `i`, head channel `e` reads
`(pids1·stride_hz_2d + pids0·BLOCK_M + i)·HEAD_DIM + e` — the kernel's
`make_block_ptr` Q address with `stride_qm = HEAD_DIM`, `stride_qk = 1`,
offset `off_hz·stride_qh_2d + start_m·BLOCK_M`. -/
```
```lean
noncomputable def fwdQTileG (s : BlockState) (Q : RegionName)
    (stride_hz_2d HEAD_DIM BLOCK_M BLOCK_DMODEL : Nat) :
    TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ :=
  fun (i, e, _) =>
    s.readMem Q ((s.pids 1 * stride_hz_2d + s.pids 0 * BLOCK_M + i.val) * HEAD_DIM + e.val)
```
</details>

<details><summary><code>fwdKTileG</code></summary>

```
/-- General K tile: key row `j` (global, over the loaded key span `SEQ`),
channel `e` reads `(pids1·stride_hz_2d + j)·HEAD_DIM + e`. -/
```
```lean
noncomputable def fwdKTileG (s : BlockState) (K : RegionName)
    (stride_hz_2d HEAD_DIM SEQ BLOCK_DMODEL : Nat) :
    TileIndex [SEQ, BLOCK_DMODEL] → ℝ :=
  fun (j, e, _) =>
    s.readMem K ((s.pids 1 * stride_hz_2d + j.val) * HEAD_DIM + e.val)
```
</details>

<details><summary><code>fwdVTileG</code></summary>

```
/-- General V tile: value row `j`, channel `e` reads
`(pids1·stride_hz_2d + j)·HEAD_DIM + e`. -/
```
```lean
noncomputable def fwdVTileG (s : BlockState) (V : RegionName)
    (stride_hz_2d HEAD_DIM SEQ BLOCK_DMODEL : Nat) :
    TileIndex [SEQ, BLOCK_DMODEL] → ℝ :=
  fun (j, e, _) =>
    s.readMem V ((s.pids 1 * stride_hz_2d + j.val) * HEAD_DIM + e.val)
```
</details>

<details><summary><code>fwdCausalSetG</code></summary>

```
/-- General causal key set for query row `i`: keys `j ≤ pids0·BLOCK_M + i` within
the loaded span `SEQ`. Nonempty because `j = 0` always qualifies (needs
`0 < SEQ`). -/
```
```lean
def fwdCausalSetG (s : BlockState) (SEQ BLOCK_M : Nat) (i : Fin BLOCK_M) :
    Finset (Fin SEQ) :=
  Finset.univ.filter (fun j : Fin SEQ => j.val ≤ s.pids 0 * BLOCK_M + i.val)
```
</details>

## Public theorem: `triton_attention_bwd_grads_genuine_output_summary_general`

<details><summary>docstring</summary>

```
/-- **★ MAIN (backward gradients, multi-block general).** Public symbolic-dimension
backward-gradient summary for `triton_attention.py`'s `_bwd_kernel` over a full
`N_CTX = BLOCK_M · num_block` sequence (the multi-block KV/Q streaming loop).
For symbolic `num_block`/`N_CTX`/`BLOCK_M`/`BLOCK_DMODEL` with contiguous strides:

* `DQ` (stored `.real`) reads back as a **real** equal to the genuine general
  `bwdKernelDQSpecG` (`priorDQ + Σ_J fp16(ds)·k`, summed over **all** key rows);
* `DV`/`DK` are stored `tl.float16`, so read back at the **fp16 `MemCell`** level as
  `MemCell.of fp16 (real.cast fp16 (some <genuine real column sum>))` — the raw
  column sums `DV[J,e] = Σ_I fp16(p[I,J])·do[I,e]`,
  `DK[J,e] = Σ_I fp16(ds[I,J])·q[I,e]` over all query rows `I ∈ Fin (BLOCK_M·num_block)`.

Honest side conditions: positive block dims and `num_block`, `BD ∣ bwdKBase`, the
streaming boundary `bwdKBase/BD + num_block·BLOCK_M ≤ D0`, the index/stride
arithmetic `hbase`, input/output region disjointness, and the honest pids grid. All
specs are defined purely over the **input** `Q`/`K`/`V`/`DO`/`M`/`Delta`/`DQ`
memory — never over the kernel's own `exec` readback. -/
```
</details>

**Statement:**
```lean
theorem triton_attention_bwd_grads_genuine_output_summary_general
    (Q K V Out DO DQ DK DV L M Delta : RegionName) (s : BlockState) (sc : ℝ)
    (BM BD D0 nb : Nat)
    (hBM : 0 < BM) (hBD : 0 < BD) (hnb : 0 < nb) (hbdvd : BD ∣ bwdKBase s)
    (hbound : bwdKBase s / BD + nb * BM ≤ D0)
    (hbase : (s.pids 0 / 4) * (32768 / BD) + (s.pids 0 % 4) * (8192 / BD) = bwdKBase s / BD)
    (hQDQ : Q ≠ DQ) (hKDQ : K ≠ DQ) (hVDQ : V ≠ DQ) (hDODQ : DO ≠ DQ)
    (hMDQ : M ≠ DQ) (hDeDQ : Delta ≠ DQ)
    (hDVDQ : DV ≠ DQ) (hDKDQ : DK ≠ DQ) (hDVDK : DV ≠ DK) (hDKDV : DK ≠ DV)
    (hin : ∀ R : RegionName, R = Q ∨ R = K ∨ R = V ∨ R = DO ∨ R = M ∨ R = Delta →
        R ≠ DV ∧ R ≠ DK ∧ R ≠ DQ)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    (∃ alg, (triton_attention_bwd_kernel Q K V Out DO DQ DK DV L M Delta sc
        32768 8192 BD 1 32768 8192 BD 1 32768 8192 BD 1
        2 4 (BM * nb) D0 nb BM BD BM).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_kernel Q K V Out DO DQ DK DV L M Delta sc
        32768 8192 BD 1 32768 8192 BD 1 32768 8192 BD 1
        2 4 (BM * nb) D0 nb BM BD BM)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BM * nb, BD] => idx.1.val < nb * BM)
        (fun idx : TileIndex [BM * nb, BD] => (DQ, bwdKBase s + idx.1.val * BD + idx.2.1.val)))
      (expected := fun idx : TileIndex [BM * nb, BD] =>
        bwdKernelDQSpecG s Q K V DO M Delta DQ BD (BM * nb) sc idx.1.val idx.2.1.val)) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_kernel Q K V Out DO DQ DK DV L M Delta sc
        32768 8192 BD 1 32768 8192 BD 1 32768 8192 BD 1
        2 4 (BM * nb) D0 nb BM BD BM)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BM * nb, BD] => idx.1.val < nb * BM)
        (fun idx : TileIndex [BM * nb, BD] => (DV, bwdKBase s + idx.1.val * BD + idx.2.1.val)))
      (expected := fun idx : TileIndex [BM * nb, BD] =>
        MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16
          (some (∑ I : Fin (BM * nb),
            bwdFp16 (bwdKernelPG s Q K M BD (BM * nb) sc I.val idx.1.val) *
              bwdKernelDOG s DO BD I.val idx.2.1.val))))) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_kernel Q K V Out DO DQ DK DV L M Delta sc
        32768 8192 BD 1 32768 8192 BD 1 32768 8192 BD 1
        2 4 (BM * nb) D0 nb BM BD BM)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BM * nb, BD] => idx.1.val < nb * BM)
        (fun idx : TileIndex [BM * nb, BD] => (DK, bwdKBase s + idx.1.val * BD + idx.2.1.val)))
      (expected := fun idx : TileIndex [BM * nb, BD] =>
        MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16
          (some (∑ I : Fin (BM * nb),
            bwdFp16 (bwdKernelDSG s Q K V DO M Delta BD (BM * nb) sc I.val idx.1.val) *
              bwdKernelQG s Q BD I.val idx.2.1.val)))))
```

**Assumptions / layout contracts:**
- `hBM : 0 < BM`
- `hBD : 0 < BD`
- `hnb : 0 < nb`
- `hbdvd : BD ∣ bwdKBase s`
- `hbound : bwdKBase s / BD + nb * BM ≤ D0`
- `hbase : (s.pids 0 / 4) * (32768 / BD) + (s.pids 0 % 4) * (8192 / BD) = bwdKBase s / BD`
- `hQDQ : Q ≠ DQ`
- `hKDQ : K ≠ DQ`
- `hVDQ : V ≠ DQ`
- `hDODQ : DO ≠ DQ`
- `hMDQ : M ≠ DQ`
- `hDeDQ : Delta ≠ DQ`
- `hDVDQ : DV ≠ DQ`
- `hDKDQ : DK ≠ DQ`
- `hDVDK : DV ≠ DK`
- `hDKDV : DK ≠ DV`
- `hin : ∀ R : RegionName, R = Q ∨ R = K ∨ R = V ∨ R = DO ∨ R = M ∨ R = Delta →
        R ≠ DV ∧ R ≠ DK ∧ R ≠ DQ`
- `hundef : ∀ rg o, s.undef rg o = 0`
- `fun idx : TileIndex [BM * nb, BD] => idx.1.val < nb * BM`
- `fun idx : TileIndex [BM * nb, BD] => (DQ, bwdKBase s + idx.1.val * BD + idx.2.1.val)`
- `fun idx : TileIndex [BM * nb, BD] => idx.1.val < nb * BM`
- `fun idx : TileIndex [BM * nb, BD] => (DV, bwdKBase s + idx.1.val * BD + idx.2.1.val)`
- `fun idx : TileIndex [BM * nb, BD] => idx.1.val < nb * BM`
- `fun idx : TileIndex [BM * nb, BD] => (DK, bwdKBase s + idx.1.val * BD + idx.2.1.val)`

**Closed-form spec defs (transitive):** `bwdKBase`, `triton_attention_bwd_kernel`, `bwdKernelDQSpecG`, `bwdFp16`, `bwdKernelPG`, `bwdKernelDOG`, `bwdKernelDSG`, `bwdKernelQG`, `bwdKernelKG`, `storeValue`, `bwdKernelQKG`, `bwdKernelMG`, `bwdKernelDPG`, `active`, `accOffset`, `bwdKernelVG`, `bwdKernelDiG`, `rowIndex`, `dIndex`

<details><summary><code>bwdKBase</code></summary>

```
/-- Block-ptr base offset for the program: `off_z·32768 + off_h·8192`. -/
```
```lean
def bwdKBase (s : BlockState) : Nat :=
  s.pids 0 / 4 * 32768 + s.pids 0 % 4 * 8192
```
</details>

<details><summary><code>triton_attention_bwd_kernel</code></summary>

```
/-- DSL port of `triton_attention.py`'s main `_bwd_kernel`.

The real Python `_bwd_kernel` rewinds the `q`/`do`/`dq` block pointers after the
inner loop with a *signed*, runtime-dependent `tl.advance(_, [lo + (1 - num_block)
* BLOCK_M, 0])` (it depends on the outer-loop counter `start_n`). That negative,
loop-counter-dependent rewind is not expressible as a static `tl.advance` delta.

Following the repo precedent (`attention_fwd_triton1` and the forward
`triton_attention` kernel), this surface re-models `q`/`do`/`dq` with a *dynamic
offset* referencing the loop counter: the three pointers are reconstructed via
`tl.make_block_ptr` at the start of each outer (`start_n`) iteration with
`offsets = (… + lo, 0)` where `lo = start_n * BLOCK_M`, then advanced
`[BLOCK_M, 0]` per inner step. The memory addresses accessed are byte-identical
to the Python advance+rewind flow, so this surface is *faithful at the observable
(memory) level for arbitrary `num_block`*. The `k`/`v`/`dk`/`dv` pointers advance
monotonically `[BLOCK_M, 0]` once per outer iteration and are unchanged. -/
```
```lean
def triton_attention_bwd_kernel
    (Q K V _Out DO DQ DK DV _L M Delta : RegionName)
    (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk
      _stride_kz _stride_kh stride_kn stride_kk
      _stride_vz _stride_vh stride_vk stride_vn
      _Z H N_CTX D0 num_block BLOCK_M BLOCK_DMODEL _BLOCK_N : Nat) :
    ComputeKernel := triton {
  off_hz = tl.program_id(0)
  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  stride_qz_2d = $(stride_qz) // $(stride_qm) // $(stride_qk)
  stride_qh_2d = $(stride_qh) // $(stride_qm) // $(stride_qk)
  k_tile_ptr = tl.make_block_ptr(base=K,
    shape=($(D0), $(BLOCK_DMODEL)),
    strides=($(stride_kn), $(stride_kk)),
    offsets=(off_z * stride_qz_2d + off_h * stride_qh_2d, 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
    order=(1, 0))
  v_tile_ptr = tl.make_block_ptr(base=V,
    shape=($(D0), $(BLOCK_DMODEL)),
    strides=($(stride_vk), $(stride_vn)),
    offsets=(off_z * stride_qz_2d + off_h * stride_qh_2d, 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
    order=(1, 0))
  dk_tile_ptr = tl.make_block_ptr(base=DK,
    shape=($(D0), $(BLOCK_DMODEL)),
    strides=($(stride_qm), $(stride_qk)),
    offsets=(off_z * stride_qz_2d + off_h * stride_qh_2d, 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
    order=(1, 0))
  dv_tile_ptr = tl.make_block_ptr(base=DV,
    shape=($(D0), $(BLOCK_DMODEL)),
    strides=($(stride_qm), $(stride_qk)),
    offsets=(off_z * stride_qz_2d + off_h * stride_qh_2d, 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
    order=(1, 0))
  for start_n in range($(0), $(num_block), $(1)) {
    lo = start_n * $(BLOCK_M)
    q_tile_ptr = tl.make_block_ptr(base=Q,
      shape=($(D0), $(BLOCK_DMODEL)),
      strides=($(stride_qm), $(stride_qk)),
      offsets=(off_z * stride_qz_2d + off_h * stride_qh_2d + lo, 0),
      block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
      order=(1, 0))
    do_tile_ptr = tl.make_block_ptr(base=DO,
      shape=($(D0), $(BLOCK_DMODEL)),
      strides=($(stride_qm), $(stride_qk)),
      offsets=(off_z * stride_qz_2d + off_h * stride_qh_2d + lo, 0),
      block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
      order=(1, 0))
    dq_tile_ptr = tl.make_block_ptr(base=DQ,
      shape=($(D0), $(BLOCK_DMODEL)),
      strides=($(stride_qm), $(stride_qk)),
      offsets=(off_z * stride_qz_2d + off_h * stride_qh_2d + lo, 0),
      block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
      order=(1, 0))
    DQ = DQ + off_z * $(stride_qz) + off_h * $(stride_qh)
    offs_qm = lo + tl.arange(0, $(BLOCK_M))
    offs_n = start_n * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
    offs_m = tl.arange(0, $(BLOCK_M))
    offs_k = tl.arange(0, $(BLOCK_DMODEL))
    dq_ptrs = DQ + (offs_qm[:, None] * $(stride_qm) +
      offs_k[None, :] * $(stride_qk))
    D_ptrs = Delta + off_hz * $(N_CTX)
    m_ptrs = M + off_hz * $(N_CTX)
    dv = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
    dk = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
    k = tl.load(k_tile_ptr, boundary_check=(0, 1))
    v = tl.load(v_tile_ptr, boundary_check=(0, 1))
    for start_m in range(lo, $(num_block) * $(BLOCK_M), $(BLOCK_M)) {
      offs_m_curr = start_m + offs_m
      q = tl.load(q_tile_ptr, boundary_check=(0, 1))
      qk = tl.dot(q, tl.trans(k))
      qk = tl.where(offs_m_curr[:, None] >= (offs_n[None, :]), qk,
        float("-inf"))
      m = tl.load(m_ptrs + offs_m_curr)
      p = tl.exp(qk * $((sm_scale : ℝ)) - m[:, None])
      do_val = tl.load(do_tile_ptr, boundary_check=(0, 1))
      dv += tl.dot(tl.trans((p).to(tl.float16)), do_val)
      Di = tl.load(D_ptrs + offs_m_curr)
      dp = tl.zeros([$(BLOCK_M), $(BLOCK_M)], dtype=tl.float32) - Di[:, None]
      dp += tl.dot(do_val, tl.trans(v))
      ds = p * dp * $((sm_scale : ℝ))
      dk += tl.dot(tl.trans((ds).to(tl.float16)), q)
      dq = tl.load(dq_tile_ptr)
      dq += tl.dot((ds).to(tl.float16), k)
      tl.store(dq_tile_ptr, dq)
      dq_ptrs += $(BLOCK_M) * $(stride_qm)
      q_tile_ptr = tl.advance(q_tile_ptr, [$(BLOCK_M), $(0)])
      do_tile_ptr = tl.advance(do_tile_ptr, [$(BLOCK_M), $(0)])
      dq_tile_ptr = tl.advance(dq_tile_ptr, [$(BLOCK_M), $(0)])
    }
    k_tile_ptr = tl.advance(k_tile_ptr, [$(BLOCK_M), $(0)])
    v_tile_ptr = tl.advance(v_tile_ptr, [$(BLOCK_M), $(0)])
    tl.store(dv_tile_ptr, (dv).to(tl.float16), boundary_check=(0, 1))
    tl.store(dk_tile_ptr, (dk).to(tl.float16), boundary_check=(0, 1))
    dv_tile_ptr = tl.advance(dv_tile_ptr, [$(BLOCK_M), $(0)])
    dk_tile_ptr = tl.advance(dk_tile_ptr, [$(BLOCK_M), $(0)])
  }
}
```
</details>

<details><summary><code>bwdKernelDQSpecG</code></summary>

```
/-- **Genuine general `DQ` value.** `dq[I,e] = priorDQ[I,e] + Σ_J fp16(ds[I,J])·k[J,e]`,
summed over **all** global key rows `J ∈ [0, N_CTX)` (causal `p ⇒ ds` zeroes
`J>I`), stored real (no fp16 cast on the `DQ` store). -/
```
```lean
noncomputable def bwdKernelDQSpecG
    (s : BlockState) (Q K V DO M Delta DQ : RegionName) (BD NCTX : Nat) (sc : ℝ)
    (I e : Nat) : ℝ :=
  s.readMem DQ (bwdKBase s + I * BD + e) +
    ∑ J : Fin NCTX,
      bwdFp16 (bwdKernelDSG s Q K V DO M Delta BD NCTX sc I J.val) *
        bwdKernelKG s K BD J.val e
```
</details>

<details><summary><code>bwdFp16</code></summary>

```
/-- fp16 round-trip on a real value, as performed by `tl.…to(tl.float16)`. -/
```
```lean
noncomputable def bwdFp16 (x : ℝ) : ℝ :=
  FloatDType.fp16.storeValue (FloatDType.real.cast FloatDType.fp16 (some x))
```
</details>

<details><summary><code>bwdKernelPG</code></summary>

```
/-- `p[I,J] = exp(qk·sm_scale − m[I])` with causal masking (`J ≤ I` keeps the
score, else `0`); mirrors the kernel `tl.where` / `tl.exp`. -/
```
```lean
noncomputable def bwdKernelPG (s : BlockState) (Q K M : RegionName) (BD NCTX : Nat)
    (sc : ℝ) (I J : Nat) : ℝ :=
  if J ≤ I then
    Real.exp (bwdKernelQKG s Q K BD I J * sc - bwdKernelMG s M NCTX I)
  else 0
```
</details>

<details><summary><code>bwdKernelDOG</code></summary>

```
/-- Loaded `do[I,e] = DO[base + I·BD + e]`. -/
```
```lean
noncomputable def bwdKernelDOG (s : BlockState) (DO : RegionName) (BD : Nat)
    (I : Nat) (e : Nat) : ℝ :=
  s.readMem DO (bwdKBase s + I * BD + e)
```
</details>

<details><summary><code>bwdKernelDSG</code></summary>

```
/-- `ds[I,J] = p[I,J]·dp[I,J]·sm_scale`. -/
```
```lean
noncomputable def bwdKernelDSG (s : BlockState) (Q K V DO M Delta : RegionName)
    (BD NCTX : Nat) (sc : ℝ) (I J : Nat) : ℝ :=
  bwdKernelPG s Q K M BD NCTX sc I J * bwdKernelDPG s V DO Delta BD NCTX I J * sc
```
</details>

<details><summary><code>bwdKernelQG</code></summary>

```
/-- Loaded `q[I,e] = Q[base + I·BD + e]` at global query row `I`. -/
```
```lean
noncomputable def bwdKernelQG (s : BlockState) (Q : RegionName) (BD : Nat)
    (I : Nat) (e : Nat) : ℝ :=
  s.readMem Q (bwdKBase s + I * BD + e)
```
</details>

<details><summary><code>bwdKernelKG</code></summary>

```
/-- Loaded `k[J,e] = K[base + J·BD + e]` at global key row `J`. -/
```
```lean
noncomputable def bwdKernelKG (s : BlockState) (K : RegionName) (BD : Nat)
    (J : Nat) (e : Nat) : ℝ :=
  s.readMem K (bwdKBase s + J * BD + e)
```
</details>

<details><summary><code>storeValue</code></summary>

```lean
noncomputable def storeValue (s : BlockState) (Acc : RegionName)
    (hzRowOffset D0 BLOCK_M BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : ℝ :=
  WithBot.unbotD 0
    (if active s hzRowOffset D0 BLOCK_M idx then
      some (s.readMem Acc (accOffset s BLOCK_M BLOCK_DMODEL idx))
    else some (0.0 : ℝ))
```
</details>

<details><summary><code>bwdKernelQKG</code></summary>

```
/-- `qk[I,J] = Σ_e q[I,e]·k[J,e]` (the `tl.dot(q, trans(k))` score). -/
```
```lean
noncomputable def bwdKernelQKG (s : BlockState) (Q K : RegionName) (BD : Nat)
    (I J : Nat) : ℝ :=
  ∑ e : Fin BD, bwdKernelQG s Q BD I e.val * bwdKernelKG s K BD J e.val
```
</details>

<details><summary><code>bwdKernelMG</code></summary>

```
/-- Loaded `m[I] = M[off_hz·N_CTX + I]` (`N_CTX = nb·BM`). -/
```
```lean
noncomputable def bwdKernelMG (s : BlockState) (M : RegionName) (NCTX : Nat)
    (I : Nat) : ℝ :=
  s.readMem M (s.pids 0 * NCTX + I)
```
</details>

<details><summary><code>bwdKernelDPG</code></summary>

```
/-- `dp[I,J] = (Σ_e do[I,e]·v[J,e]) − Di[I]`. -/
```
```lean
noncomputable def bwdKernelDPG (s : BlockState) (V DO Delta : RegionName) (BD NCTX : Nat)
    (I J : Nat) : ℝ :=
  (∑ e : Fin BD, bwdKernelDOG s DO BD I e.val * bwdKernelVG s V BD J e.val)
    - bwdKernelDiG s Delta NCTX I
```
</details>

<details><summary><code>active</code></summary>

```lean
def active (s : BlockState) (hzRowOffset D0 BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Prop :=
  rowIndex s BLOCK_M idx.1 + hzRowOffset < D0
```
</details>

<details><summary><code>accOffset</code></summary>

```lean
def accOffset (s : BlockState) (BLOCK_M BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  rowIndex s BLOCK_M idx.1 * BLOCK_DMODEL + dIndex idx
```
</details>

<details><summary><code>bwdKernelVG</code></summary>

```
/-- Loaded `v[J,e] = V[base + J·BD + e]`. -/
```
```lean
noncomputable def bwdKernelVG (s : BlockState) (V : RegionName) (BD : Nat)
    (J : Nat) (e : Nat) : ℝ :=
  s.readMem V (bwdKBase s + J * BD + e)
```
</details>

<details><summary><code>bwdKernelDiG</code></summary>

```
/-- Loaded `Di[I] = Delta[off_hz·N_CTX + I]`. -/
```
```lean
noncomputable def bwdKernelDiG (s : BlockState) (Delta : RegionName) (NCTX : Nat)
    (I : Nat) : ℝ :=
  s.readMem Delta (s.pids 0 * NCTX + I)
```
</details>

<details><summary><code>rowIndex</code></summary>

```lean
def rowIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val
```
</details>

<details><summary><code>dIndex</code></summary>

```lean
def dIndex (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  idx.2.1.val
```
</details>

## Also present (pinned special-case summaries)
- `triton_attention_forward_output_store_slice_compute_correct`
- `triton_attention_forward_l_store_slice_compute_correct`
- `triton_attention_forward_m_store_slice_compute_correct`
- `triton_attention_bwd_preprocess_newdo_store_slice_compute_correct`
- `triton_attention_bwd_preprocess_newdo_formula_slice_compute_correct`
- `triton_attention_bwd_preprocess_delta_formula_slice_compute_correct`
- `triton_attention_bwd_preprocess_delta_store_slice_compute_correct`
- `triton_attention_bwd_preprocess_newdo_genuine_compute_correct`
- `triton_attention_bwd_preprocess_delta_genuine_compute_correct`
- `triton_attention_bwd_score_p_formula_slice_compute_correct`
- `triton_attention_bwd_score_ds_formula_slice_compute_correct`
- `triton_attention_bwd_dq_dot_step_slice_compute_correct`
- `triton_attention_bwd_trans_dot_step_slice_compute_correct`
- `triton_attention_bwd_dv_dot_step_slice_compute_correct`
- `triton_attention_bwd_dk_dot_step_slice_compute_correct`
- `triton_attention_bwd_dq_store_slice_compute_correct`
- `triton_attention_bwd_dkdv_store_slice_compute_correct`
- `triton_attention_bwd_dk_store_slice_compute_correct`
- `triton_attention_bwd_dv_store_slice_compute_correct`
- `triton_attention_forward_output_store_python_test_shape_compute_correct`
- `triton_attention_forward_l_store_python_test_shape_compute_correct`
- `triton_attention_forward_m_store_python_test_shape_compute_correct`
- `triton_attention_bwd_preprocess_newdo_store_python_test_shape_compute_correct`
- `triton_attention_bwd_preprocess_newdo_formula_python_test_shape_compute_correct`
- `triton_attention_bwd_preprocess_delta_formula_python_test_shape_compute_correct`
- `triton_attention_bwd_preprocess_delta_store_python_test_shape_compute_correct`
- `triton_attention_bwd_dq_store_python_test_shape_compute_correct`
- `triton_attention_bwd_dk_store_python_test_shape_compute_correct`
- `triton_attention_bwd_dv_store_python_test_shape_compute_correct`
- `triton_attention_bwd_score_p_formula_python_test_shape_compute_correct`
- `triton_attention_bwd_score_ds_formula_python_test_shape_compute_correct`
- `triton_attention_forward_surface_out_python_test_shape_compute_correct`
- `triton_attention_forward_surface_l_python_test_shape_compute_correct`
- `triton_attention_forward_surface_m_python_test_shape_compute_correct`
- `triton_attention_forward_python_test_shape_all_outputs_compute_correct`
- `triton_attention_bwd_score_python_test_shape_all_outputs_compute_correct`
- `triton_attention_bwd_score_python_test_shape_formula_summary`
- `triton_attention_forward_python_test_shape_output_summary`
