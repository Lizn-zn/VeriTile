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
specification triton_linear_activation_output_summary_general
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

## Public theorem: `triton_linear_activation_io_correctness`

<details><summary>docstring</summary>

```
/-- **The `⊨[R]` io headline for `triton_linear_activation.py`'s `kernel_fma`**
(round-12; the second `StreamMasked3DKernelIO₃ₓ₂`-with-`pre` consumer, and
the genre's first *cast-free GEMM* face).

For **every** rounding model `R` — with no hypothesis on `R` whatsoever, the
kernel being entirely cast-free (no `Op.castFloat`, both terminal stores
`Stmt.store .real`) — the faithful `kernel_fma` surface implements, on its
`StreamMasked3DKernelIO₃ₓ₂` signature, the **ideal-ℝ fused linear +
activation** over the three streamed channels:

* `C` lane `l = (i, j)` holds
  `applyActivation ACTIVATION (bias-seed + ∑ t, ∑ e, A-tile[t](i,e) · B-tile[t](e,j))`;
* `ACT_INPUTS` lane `l` holds the **un-activated** same value (the claim is
  vacuous unless `SHOULD_SAVE_ACT_INPUTS`, which `writeMask2` carries).

The whole constexpr mix stays **parametric**, exactly as the port has it: the
`String` parameter `ACTIVATION` enters `f` through `applyActivation` — itself
the same four *sequential, non-exclusive* `ACTIVATION == "…"` gates the kernel
runs, so even pathological strings are modelled faithfully and every
unrecognised string (e.g. the benchmark's `""`) is the identity; the `Bool`
parameter `HAS_BIAS` enters through `tlaBiasIO`'s outer guard and through
`mask3`; the `Bool` parameter `SHOULD_SAVE_ACT_INPUTS` enters through
`writeMask2`. Nothing is specialised to a configuration.

**`pre` = the trusted-launch boundary.** `io.pre pid₀ _ _` is instantiated
with exactly the exact headline's `hFitM ∧ hFitN` (this program's tile fits
the output). It cannot be demoted into `writeMask1`: a non-fitting program
still *writes*, at `%`-wrapped addresses, so the skin's frame clause would be
false. Under `pre` the kernel's `c_ptr_mask` is all-true, which is why
`writeMask1 = True`.

**Hypothesis provenance** (all inherited from the exact stack, all
truth-forced):
* `hT : 0 < numKBlocks` — the `bias` row is a *degenerate static stream* read
  before the loop; with `T = 0` there is no step for the skin to carry it on.
* `hsno`/`hble` (`output_n_stride = 1`, `BLOCK_N ≤ output_m_stride`) — the
  exact headline's store-footprint injectivity for `C`
  (`cOffset_injective_of_fit`); `hsain`/`hale` are the same facts for
  `ACT_INPUTS`, and `hne : ACT_INPUTS ≠ C` keeps the `C` store from
  clobbering the saved pre-activation tile. These are the wrapper's
  contiguous row-major layouts.

**Disclosed scope.** (1) The io face is stated for a **5-region disjoint**
allocation `[A, B, bias, C, ACT_INPUTS]`; the real launcher aliases
`bias := x` when `bias is None` and passes `ACT_INPUTS := None` when not
saving, so those two flag-off configurations fall outside this allocation
shape (their claims are still available from the exact headline, which needs
no allocation). (2) The ℕ-truncated `grid_m - group_idx·GROUP_M` of the
L2 schedule is transcribed as the kernel computes it: for a `program_idx`
past the grid the truncation can make `group_size = 0`, and Lean's
`pid % 0 = pid` / `_ / 0 = 0` then silently diverge from Triton's UB. This is
*not* repaired here — `blockMIdx`/`blockNIdx` are the kernel's own arithmetic,
`pre` merely scopes the claim to whatever tile those produce. (3) `.real`
output grids: the port models no narrow-float store, so at every `R` the two
terminal cells carry the exact fold values.

The exact headline `triton_linear_activation_output_summary_general` above is
retained unchanged; this face strictly restates its content on the streaming
skin (at `R := .triv` the readback contract degenerates to it). -/
```
</details>

**Statement:**
```lean
specification triton_linear_activation_io_correctness (R : RoundingModel)
    (C ACT_INPUTS A B bias : RegionName)
    (M N output_m_stride output_n_stride act_inputs_m_stride act_inputs_n_stride
      a_m_stride a_k_stride b_n_stride b_k_stride
      BLOCK_M GROUP_M BLOCK_N BLOCK_K numKBlocks : Nat)
    (HAS_BIAS SHOULD_SAVE_ACT_INPUTS : Bool) (ACTIVATION : String)
    (hT : 0 < numKBlocks)
    (hsno : output_n_stride = 1) (hble : BLOCK_N ≤ output_m_stride)
    (hsain : act_inputs_n_stride = 1) (hale : BLOCK_N ≤ act_inputs_m_stride)
    (hne : ACT_INPUTS ≠ C) :
    tlaIO C ACT_INPUTS A B bias M N output_m_stride output_n_stride
        act_inputs_m_stride act_inputs_n_stride a_m_stride a_k_stride b_n_stride b_k_stride
        BLOCK_M GROUP_M BLOCK_N BLOCK_K numKBlocks
        HAS_BIAS SHOULD_SAVE_ACT_INPUTS ACTIVATION ⊨[R]
      fun p₀ _ _ xs ys zs =>
        (fun l => applyActivation ACTIVATION
            (tlaLinIO p₀ M N BLOCK_M BLOCK_N GROUP_M BLOCK_K numKBlocks HAS_BIAS xs ys zs l),
         fun l => tlaLinIO p₀ M N BLOCK_M BLOCK_N GROUP_M BLOCK_K numKBlocks HAS_BIAS xs ys zs l)
```

**Assumptions / layout contracts:**
- `hT : 0 < numKBlocks`
- `hsno : output_n_stride = 1`
- `hble : BLOCK_N ≤ output_m_stride`
- `hsain : act_inputs_n_stride = 1`
- `hale : BLOCK_N ≤ act_inputs_m_stride`
- `hne : ACT_INPUTS ≠ C`

**Closed-form spec defs (transitive):** `tlaIO`, `applyActivation`, `tlaLinIO`, `kernel_fma_surface`, `blockMIdx`, `blockNIdx`, `tlaRowIdx`, `tlaColIdx`, `geluRef`, `fastGeluRef`, `tlaBiasIO`, `tlaALane`, `tlaBLane`, `kernelMin`

<details><summary><code>tlaIO</code></summary>

```
/-- **Streaming IO signature** of `kernel_fma` on the three-stream,
two-terminal-store fold skin.

Channels: `inp1 = A` and `inp2 = B` are the genuine per-step streams (step
`t` reads the `[BLOCK_M, BLOCK_K]` `A`-tile and the `[BLOCK_K, BLOCK_N]`
`B`-tile at the invariant's advanced pointers); `inp3 = bias` is the genre's
**degenerate static stream** — its window ignores `t` and reads the
`[BLOCK_N]` bias row at the wrapped column indices. Outputs: `out1 = C`
(activated) and `out2 = ACT_INPUTS` (pre-activation), both `[BLOCK_M ·
BLOCK_N]` lanes on the `.real` grid — the port models no narrow-float store
(both statements are `Stmt.store .real`; Python's `.to(tl.float32)` on the
bias load and the host's fp16 output dtype are not modeled), so declaring
anything but `.real` here would be false against this surface.

Masks: the two GEMM streams are unmasked (`K_LOAD_MASK_NEEDED = True` arm);
`mask3` carries **both** of the bias load's guards — the `HAS_BIAS` constexpr
gate and the load's own `n_offs < N` mask. `writeMask1 = True` (under `pre`
the `c_ptr_mask` is all-true); `writeMask2` carries the
`SHOULD_SAVE_ACT_INPUTS` constexpr gate, so the `ACT_INPUTS` claim is vacuous
when the flag is off.

`pre` is the port's **trusted-launch boundary**: this program's tile fits the
output (`hFitM`/`hFitN`). It cannot be folded into `writeMask1` — a
non-fitting program still *writes*, at `%`-wrapped addresses, so the skin's
frame clause ("every cell outside the write-active window is untouched")
would be false. -/
```
```lean
noncomputable def tlaIO (C ACT_INPUTS A B biasR : RegionName)
    (M N smo sno saim sain sam sak sbn sbk BM GM BN BK numKBlocks : Nat)
    (HAS_BIAS SHOULD_SAVE_ACT_INPUTS : Bool) (ACTIVATION : String) :
    StreamMasked3DKernelIO₃ₓ₂ where
  kernel := kernel_fma_surface C ACT_INPUTS A B biasR M N smo sno saim sain
    sam sak sbn sbk BM GM BN BK numKBlocks HAS_BIAS SHOULD_SAVE_ACT_INPUTS ACTIVATION
  inp1 := A
  inp2 := B
  inp3 := biasR
  out1 := C
  out2 := ACT_INPUTS
  T := numKBlocks
  B1 := BM * BK
  B2 := BK * BN
  B3 := BN
  C1 := BM * BN
  C2 := BM * BN
  out1DType := .real
  out2DType := .real
  pre := fun p₀ _ _ =>
    blockMIdx p₀ M N BM BN GM * BM + BM ≤ M ∧ blockNIdx p₀ M N BM BN GM * BN + BN ≤ N
  read1 := fun p₀ _ _ t l => tlaRowIdx p₀ M N BM BN GM (l.val / BK) * sam + l.val % BK * sak + t.val * BK * sak
  read2 := fun p₀ _ _ t l => l.val / BN * sbk + tlaColIdx p₀ M N BM BN GM (l.val % BN) * sbn + t.val * BK * sbk
  read3 := fun p₀ _ _ _ j => tlaColIdx p₀ M N BM BN GM j.val
  write1 := fun p₀ _ _ l => tlaRowIdx p₀ M N BM BN GM (l.val / BN) * smo + tlaColIdx p₀ M N BM BN GM (l.val % BN) * sno
  write2 := fun p₀ _ _ l => tlaRowIdx p₀ M N BM BN GM (l.val / BN) * saim + tlaColIdx p₀ M N BM BN GM (l.val % BN) * sain
  mask1 := fun _ _ _ _ _ => True
  mask2 := fun _ _ _ _ _ => True
  mask3 := fun p₀ _ _ _ j => HAS_BIAS = Bool.true ∧ tlaColIdx p₀ M N BM BN GM j.val < N
  writeMask1 := fun _ _ _ _ => True
  writeMask2 := fun _ _ _ _ => SHOULD_SAVE_ACT_INPUTS = Bool.true
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

<details><summary><code>tlaLinIO</code></summary>

```
/-- The **pre-activation** streaming spec of output lane `l`: the bias seed
plus the ideal-ℝ double fold `∑ t, ∑ e, A-tile[t](i,e) · B-tile[t](e,j)`
(`linearSpec` restated on the streams). -/
```
```lean
noncomputable def tlaLinIO (p₀ M N BM BN GM BK T : Nat) (HAS_BIAS : Bool)
    (xs : Fin T → Fin (BM * BK) → ℝ) (ys : Fin T → Fin (BK * BN) → ℝ)
    (zs : Fin T → Fin BN → ℝ) (l : Fin (BM * BN)) : ℝ :=
  tlaBiasIO p₀ M N BM BN GM T HAS_BIAS zs l
    + ∑ t : Fin T, ∑ e : Fin BK,
        xs t (tlaALane BM BN BK l e) * ys t (tlaBLane BM BN BK l e)
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

<details><summary><code>tlaRowIdx</code></summary>

```
/-- Pid form of `rowIndex`: the `% M`-wrapped global row of tile lane `i`
(`(block_m_idx·BLOCK_M + i) % M`). Definitionally `rowIndex` at
`pid₀ = s.pids 0`. -/
```
```lean
def tlaRowIdx (p₀ M N BM BN GM i : Nat) : Nat := (blockMIdx p₀ M N BM BN GM * BM + i) % M
```
</details>

<details><summary><code>tlaColIdx</code></summary>

```
/-- Pid form of `colIndex`: the `% N`-wrapped global column of tile lane `j`. -/
```
```lean
def tlaColIdx (p₀ M N BM BN GM j : Nat) : Nat := (blockNIdx p₀ M N BM BN GM * BN + j) % N
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

<details><summary><code>tlaBiasIO</code></summary>

```
/-- The bias seed of output lane `l` read off the **degenerate static
stream** `zs` (the `bias` channel ignores `t`, so only step `0` is ever
consulted; `T = 0` has no step to read and the pre-loop load is unreachable
from the streams — the headline therefore assumes `0 < numKBlocks`). The two
guards transcribe `biasBase` verbatim: the `HAS_BIAS` constexpr gate and the
load's own `mask = n_offs < N, other = 0.0`. -/
```
```lean
noncomputable def tlaBiasIO (p₀ M N BM BN GM T : Nat) (HAS_BIAS : Bool)
    (zs : Fin T → Fin BN → ℝ) (l : Fin (BM * BN)) : ℝ :=
  if HAS_BIAS = Bool.true then
    (if tlaColIdx p₀ M N BM BN GM (Lane2D.decode l).2.1.val < N then
      (if h : 0 < T then zs ⟨0, h⟩ (Lane2D.decode l).2.1 else 0)
     else 0)
  else 0
```
</details>

<details><summary><code>tlaALane</code></summary>

```
/-- The `A`-stream lane feeding output lane `l` at inner key `e`: the row of
`l` (row-major over the `[BLOCK_M, BLOCK_N]` output tile) paired with `e`
over the `[BLOCK_M, BLOCK_K]` per-step `A`-tile. -/
```
```lean
def tlaALane (BM BN BK : Nat) (l : Fin (BM * BN)) (e : Fin BK) : Fin (BM * BK) :=
  Lane2D.encode ((Lane2D.decode l).1, e, PUnit.unit)
```
</details>

<details><summary><code>tlaBLane</code></summary>

```
/-- The `B`-stream lane feeding output lane `l` at inner key `e`: `e` paired
with the column of `l` over the `[BLOCK_K, BLOCK_N]` per-step `B`-tile. -/
```
```lean
def tlaBLane (BM BN BK : Nat) (l : Fin (BM * BN)) (e : Fin BK) : Fin (BK * BN) :=
  Lane2D.encode (e, (Lane2D.decode l).2.1, PUnit.unit)
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

## Also present (pinned special-case summaries)
- `kernel_fma_C_compute_correct`
- `kernel_fma_act_inputs_compute_correct`
