# Spec sheet — `bench/tritonbench_g/triton_linear_activation/TritonLinearActivation.lean`

**Python source:** `bench/tritonbench_g/triton_linear_activation/triton_linear_activation.py`

## Public theorem: `triton_linear_activation_output_summary_general`

<details><summary>docstring</summary>

```
/-- **Dimension-general bundled correctness summary for
`triton_linear_activation.py`'s `kernel_fma`.**

For arbitrary shapes `M`/`N`, contracted dimension `K = BLOCK_K · numKBlocks`
(the `K_LOAD_MASK_NEEDED` heuristics arm — exact-multiple `K`), strides, tile
dims `BLOCK_M`/`BLOCK_N`, L2-group size `GROUP_M`, linear program id, constexpr
flags `HAS_BIAS` / `SHOULD_SAVE_ACT_INPUTS`, and **any** `ACTIVATION` string,
it packages:

* the surface lowers to the algorithm layer;
* the `C` output: every cell of the computed tile equals
  `applyActivation(ACTIVATION, bias[n_offs j] + Σ_{k<K} A[m_offs i, k]·B[k, n_offs j])`
  — the genuine linear form over **input** memory (a `gemmSum` `Finset.sum`)
  composed with the exact real activation (`Real.tanh`, erf-GELU via
  `VeriTile.Math.realErf`, tanh fast-GELU, `max 0 ·` ReLU, identity for every
  other string, e.g. the benchmark's `""`), never a read-back of the kernel's
  own output;
* the `ACT_INPUTS` output (`SHOULD_SAVE_ACT_INPUTS = true`, with its own
  layout side-conditions and `ACT_INPUTS ≠ C`): every cell equals the
  **un-activated** `bias + Σ_k A·B` pre-activation value.

Honest side-conditions: `hFitM`/`hFitN` — this program's tile fits the output
(`block_m_idx·BLOCK_M + BLOCK_M ≤ M`, `block_n_idx·BLOCK_N + BLOCK_N ≤ N`, true
for every program of the launch grid whenever `BLOCK_M ∣ M ∧ BLOCK_N ∣ N`,
e.g. all benchmark shapes); without them the kernel's `%`-wrapped store offsets
genuinely collide (the store mask tests the already-wrapped offsets, so it
never masks an overhanging lane). Unit minor stride and `BLOCK_N ≤` major
stride (`output_n_stride = 1`, `BLOCK_N ≤ output_m_stride`; the wrapper's
contiguous row-major outputs) give store-footprint injectivity. -/
```
</details>

**Statement:**
```lean
theorem triton_linear_activation_output_summary_general
    (C ACT_INPUTS A B bias : RegionName) (s : BlockState)
    (M N output_m_stride output_n_stride act_inputs_m_stride act_inputs_n_stride
      a_m_stride a_k_stride b_n_stride b_k_stride
      BLOCK_M GROUP_M BLOCK_N BLOCK_K numKBlocks : Nat)
    (HAS_BIAS SHOULD_SAVE_ACT_INPUTS : Bool) (ACTIVATION : String)
    (hFitM : blockMIdx (s.pids 0) M N BLOCK_M BLOCK_N GROUP_M * BLOCK_M + BLOCK_M ≤ M)
    (hFitN : blockNIdx (s.pids 0) M N BLOCK_M BLOCK_N GROUP_M * BLOCK_N + BLOCK_N ≤ N)
    (hsno : output_n_stride = 1) (hble : BLOCK_N ≤ output_m_stride) :
    -- (1) the surface lowers to the algorithm layer
    (∃ alg, (kernel_fma_surface C ACT_INPUTS A B bias M N output_m_stride output_n_stride
      act_inputs_m_stride act_inputs_n_stride a_m_stride a_k_stride b_n_stride b_k_stride
      BLOCK_M GROUP_M BLOCK_N BLOCK_K numKBlocks
      HAS_BIAS SHOULD_SAVE_ACT_INPUTS ACTIVATION).toAlgorithm? = Except.ok alg) ∧
    -- (2) C: genuine fused linear + activation
    ComputeCorrect.Realizes_without_Rounding
      (kernel := kernel_fma_surface C ACT_INPUTS A B bias M N output_m_stride output_n_stride
        act_inputs_m_stride act_inputs_n_stride a_m_stride a_k_stride b_n_stride b_k_stride
        BLOCK_M GROUP_M BLOCK_N BLOCK_K numKBlocks
        HAS_BIAS SHOULD_SAVE_ACT_INPUTS ACTIVATION)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
        some (C, cOffset s M N BLOCK_M BLOCK_N GROUP_M output_m_stride output_n_stride idx))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
        applyActivation ACTIVATION
          (linearSpec s A B bias M N BLOCK_M BLOCK_N GROUP_M a_m_stride a_k_stride
            b_k_stride b_n_stride BLOCK_K numKBlocks HAS_BIAS idx.1 idx.2.1)) ∧
    -- (3) ACT_INPUTS (when saving): the genuine un-activated pre-activation values
    (SHOULD_SAVE_ACT_INPUTS = Bool.true → ACT_INPUTS ≠ C →
      act_inputs_n_stride = 1 → BLOCK_N ≤ act_inputs_m_stride →
      ComputeCorrect.Realizes_without_Rounding
        (kernel := kernel_fma_surface C ACT_INPUTS A B bias M N output_m_stride output_n_stride
          act_inputs_m_stride act_inputs_n_stride a_m_stride a_k_stride b_n_stride b_k_stride
          BLOCK_M GROUP_M BLOCK_N BLOCK_K numKBlocks
          HAS_BIAS SHOULD_SAVE_ACT_INPUTS ACTIVATION)
        (initialState := s)
        (write := fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
          some (ACT_INPUTS, actOffset s M N BLOCK_M BLOCK_N GROUP_M
            act_inputs_m_stride act_inputs_n_stride idx))
        (expected := fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
          linearSpec s A B bias M N BLOCK_M BLOCK_N GROUP_M a_m_stride a_k_stride
            b_k_stride b_n_stride BLOCK_K numKBlocks HAS_BIAS idx.1 idx.2.1))
```

**Assumptions / layout contracts:**
- `hFitM : blockMIdx (s.pids 0) M N BLOCK_M BLOCK_N GROUP_M * BLOCK_M + BLOCK_M ≤ M`
- `hFitN : blockNIdx (s.pids 0) M N BLOCK_M BLOCK_N GROUP_M * BLOCK_N + BLOCK_N ≤ N`
- `hsno : output_n_stride = 1`
- `hble : BLOCK_N ≤ output_m_stride`

**Closed-form spec defs (transitive):** `blockMIdx`, `blockNIdx`, `kernel_fma_surface`, `cOffset`, `applyActivation`, `linearSpec`, `actOffset`, `kernelMin`, `rowIndex`, `colIndex`, `geluRef`, `fastGeluRef`, `biasBase`, `aElem`, `bElem`, `rowGlobal`, `colGlobal`

<details><summary><code>blockMIdx</code></summary>

```
/-- The kernel's `block_m_idx` derivation from the linear `program_idx`. -/
```
```lean
def blockMIdx (pid M N BM BN GM : Nat) : Nat :=
  let grid_m := (M + BM - 1) / BM
  let grid_n := (N + BN - 1) / BN
  let width := GM * grid_n
  let group_idx := pid / width
  let group_size := kernelMin (grid_m - group_idx * GM) GM
  group_idx * GM + pid % group_size
```
</details>

<details><summary><code>blockNIdx</code></summary>

```
/-- The kernel's `block_n_idx` derivation from the linear `program_idx`. -/
```
```lean
def blockNIdx (pid M N BM BN GM : Nat) : Nat :=
  let grid_m := (M + BM - 1) / BM
  let grid_n := (N + BN - 1) / BN
  let width := GM * grid_n
  let group_idx := pid / width
  let group_size := kernelMin (grid_m - group_idx * GM) GM
  (pid % width) / group_size
```
</details>

<details><summary><code>kernel_fma_surface</code></summary>

```
/-- Faithful transcription of `triton_linear_activation.py`'s `kernel_fma`
(the `K_LOAD_MASK_NEEDED = True` heuristics arm; see the module docstring's
translation-surface blocker for the presented loop bound and inlined helpers). -/
```
```lean
noncomputable def kernel_fma_surface
    (C ACT_INPUTS A B bias : RegionName)
    (M N output_m_stride output_n_stride act_inputs_m_stride act_inputs_n_stride
      a_m_stride a_k_stride b_n_stride b_k_stride
      BLOCK_M GROUP_M BLOCK_N BLOCK_K numKBlocks : Nat)
    (HAS_BIAS SHOULD_SAVE_ACT_INPUTS : Bool) (ACTIVATION : String) :
    ComputeKernel := triton {
  program_idx = tl.program_id(axis=0)
  grid_m = ($((M : Nat)) + $(BLOCK_M) - $(1)) // $(BLOCK_M)
  grid_n = ($((N : Nat)) + $(BLOCK_N) - $(1)) // $(BLOCK_N)
  width = $(GROUP_M) * grid_n
  group_idx = program_idx // width
  group_size = min(grid_m - group_idx * $(GROUP_M), $(GROUP_M))
  block_m_idx = group_idx * $(GROUP_M) + (program_idx % group_size)
  block_n_idx = (program_idx % width) // group_size
  m_offs_untagged = block_m_idx * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  n_offs_untagged = block_n_idx * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  m_offs = tl.max_contiguous(tl.multiple_of(m_offs_untagged % $(M), $(BLOCK_M)), $(BLOCK_M))
  n_offs = tl.max_contiguous(tl.multiple_of(n_offs_untagged % $(N), $(BLOCK_N)), $(BLOCK_N))
  k_range_offs = tl.arange(0, $(BLOCK_K))
  A = A + (m_offs[:, None] * $(a_m_stride) + k_range_offs[None, :] * $(a_k_stride))
  B = B + (k_range_offs[:, None] * $(b_k_stride) + n_offs[None, :] * $(b_n_stride))
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
  if HAS_BIAS {
    bias = tl.load(bias + n_offs, mask=n_offs < $(N), other=0.0).to(tl.float32)
    acc += bias[None, :]
  }
  for k in range($(0), $(numKBlocks), $(1)) {
    a = tl.load(A)
    b = tl.load(B)
    acc += tl.dot(a, b)
    A += $(BLOCK_K) * $(a_k_stride)
    B += $(BLOCK_K) * $(b_k_stride)
  }
  if SHOULD_SAVE_ACT_INPUTS {
    act_in_ptrs = ACT_INPUTS + m_offs[:, None] * $(act_inputs_m_stride) + n_offs[None, :] * $(act_inputs_n_stride)
    tl.store(act_in_ptrs, acc)
  }
  if ACTIVATION == "tanh" {
    acc = tanh(acc)
  }
  if ACTIVATION == "gelu" {
    acc = acc * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(acc / $((Real.sqrt 2 : ℝ))))
  }
  if ACTIVATION == "fast_gelu" {
    acc = 0.5 * acc * (1 + tanh($((Real.sqrt (2.0 / Real.pi) : ℝ)) * (acc + 0.044715 * acc * acc * acc)))
  }
  if ACTIVATION == "relu" {
    acc = tl.maximum(0, acc)
  }
  C = C + m_offs[:, None] * $(output_m_stride) + n_offs[None, :] * $(output_n_stride)
  c_ptr_mask = (m_offs < $(M))[:, None] & (n_offs < $(N))[None, :]
  tl.store(C, acc, mask=c_ptr_mask)
}
```
</details>

<details><summary><code>cOffset</code></summary>

```
/-- The `C` store address of tile lane `(i,j)`:
`m_offs i · output_m_stride + n_offs j · output_n_stride` (the kernel stores
through the **`%`-wrapped** `m_offs`/`n_offs`). -/
```
```lean
def cOffset (s : BlockState) (M N BM BN GM smo sno : Nat) (idx : TileIndex [BM, BN]) : Nat :=
  rowIndex s M N BM BN GM idx.1 * smo + colIndex s M N BM BN GM idx.2.1 * sno
```
</details>

<details><summary><code>applyActivation</code></summary>

```
/-- The activation tail exactly as the kernel applies it: the four sequential
`ACTIVATION == "…"` constexpr gates (mutually exclusive for any one string;
every other string — including the benchmark's `""` — is the identity). -/
```
```lean
noncomputable def applyActivation (ACTIVATION : String) (x : ℝ) : ℝ :=
  let x1 := if ACTIVATION == "tanh" then Real.tanh x else x
  let x2 := if ACTIVATION == "gelu" then geluRef x1 else x1
  let x3 := if ACTIVATION == "fast_gelu" then fastGeluRef x2 else x2
  if ACTIVATION == "relu" then max 0 x3 else x3
```
</details>

<details><summary><code>linearSpec</code></summary>

```
/-- **Genuine pre-activation linear spec** (over ℝ):
`bias[j] + Σ_{k < BLOCK_K·numKBlocks} A[i,k] · B[k,j]`. -/
```
```lean
noncomputable def linearSpec (s : BlockState) (A B biasR : RegionName)
    (M N BM BN GM sam sak sbk sbn BLOCK_K numKBlocks : Nat) (HAS_BIAS : Bool)
    (i : Fin BM) (j : Fin BN) : ℝ :=
  biasBase s biasR M N BM BN GM HAS_BIAS j
    + gemmSum (aElem s A M N BM BN GM sam sak i) (bElem s B M N BM BN GM sbk sbn j)
        (BLOCK_K * numKBlocks)
```
</details>

<details><summary><code>actOffset</code></summary>

```
/-- The `ACT_INPUTS` store address of tile lane `(i,j)`. -/
```
```lean
def actOffset (s : BlockState) (M N BM BN GM saim sain : Nat) (idx : TileIndex [BM, BN]) : Nat :=
  rowIndex s M N BM BN GM idx.1 * saim + colIndex s M N BM BN GM idx.2.1 * sain
```
</details>

<details><summary><code>kernelMin</code></summary>

```
/-- `min` as the kernel's `tl.where(a < b, a, b)` spells it. -/
```
```lean
def kernelMin (a b : Nat) : Nat := if a < b then a else b
```
</details>

<details><summary><code>rowIndex</code></summary>

```
/-- The `% M`-wrapped row index of tile lane `i` (the kernel's `m_offs`). -/
```
```lean
def rowIndex (s : BlockState) (M N BM BN GM : Nat) (i : Fin BM) : Nat :=
  rowGlobal s M N BM BN GM i % M
```
</details>

<details><summary><code>colIndex</code></summary>

```
/-- The `% N`-wrapped column index of tile lane `j` (the kernel's `n_offs`). -/
```
```lean
def colIndex (s : BlockState) (M N BM BN GM : Nat) (j : Fin BN) : Nat :=
  colGlobal s M N BM BN GM j % N
```
</details>

<details><summary><code>geluRef</code></summary>

```
/-- Real GELU as the kernel's inlined `gelu` helper spells it:
`x · 0.5 · (1 + erf(x / √2))` with the exact real error function. -/
```
```lean
noncomputable def geluRef (x : ℝ) : ℝ :=
  x * 0.5 * (1.0 + VeriTile.Math.realErf (x / Real.sqrt 2))
```
</details>

<details><summary><code>fastGeluRef</code></summary>

```
/-- Real fast-GELU as the kernel's inlined `fast_gelu` helper spells it:
`0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))`. -/
```
```lean
noncomputable def fastGeluRef (x : ℝ) : ℝ :=
  0.5 * x * (1 + Real.tanh (Real.sqrt (2.0 / Real.pi) * (x + 0.044715 * x * x * x)))
```
</details>

<details><summary><code>biasBase</code></summary>

```
/-- The bias contribution of column lane `j`: the masked bias-row load when
`HAS_BIAS` (`bias[n_offs j]` if `n_offs j < N`, else the load's `other=0.0`),
`0` otherwise. -/
```
```lean
noncomputable def biasBase (s : BlockState) (biasR : RegionName) (M N BM BN GM : Nat)
    (HAS_BIAS : Bool) (j : Fin BN) : ℝ :=
  if HAS_BIAS then
    (if colIndex s M N BM BN GM j < N then s.readMem biasR (colIndex s M N BM BN GM j) else 0)
  else 0
```
</details>

<details><summary><code>aElem</code></summary>

```
/-- `A[i, k] = readMem A (m_offs i · a_m_stride + k · a_k_stride)`. -/
```
```lean
noncomputable def aElem (s : BlockState) (A : RegionName) (M N BM BN GM sam sak : Nat)
    (i : Fin BM) (k : Nat) : ℝ :=
  s.readMem A (rowIndex s M N BM BN GM i * sam + k * sak)
```
</details>

<details><summary><code>bElem</code></summary>

```
/-- `B[k, j] = readMem B (k · b_k_stride + n_offs j · b_n_stride)`. -/
```
```lean
noncomputable def bElem (s : BlockState) (B : RegionName) (M N BM BN GM sbk sbn : Nat)
    (j : Fin BN) (k : Nat) : ℝ :=
  s.readMem B (k * sbk + colIndex s M N BM BN GM j * sbn)
```
</details>

<details><summary><code>rowGlobal</code></summary>

```
/-- Global output row of tile lane `i`: `block_m_idx · BLOCK_M + i`, **before**
the `% M` wrap (the kernel's `m_offs_untagged`). -/
```
```lean
def rowGlobal (s : BlockState) (M N BM BN GM : Nat) (i : Fin BM) : Nat :=
  blockMIdx (s.pids 0) M N BM BN GM * BM + i.val
```
</details>

<details><summary><code>colGlobal</code></summary>

```
/-- Global output column of tile lane `j`, before the `% N` wrap. -/
```
```lean
def colGlobal (s : BlockState) (M N BM BN GM : Nat) (j : Fin BN) : Nat :=
  blockNIdx (s.pids 0) M N BM BN GM * BN + j.val
```
</details>

## Also present (pinned special-case summaries)
- `kernel_fma_C_compute_correct`
- `kernel_fma_act_inputs_compute_correct`
