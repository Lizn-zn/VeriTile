# Spec sheet — `bench/tritonbench_g/rmsnorm_implementation/RmsnormImplementation.lean`

**Python source:** `bench/tritonbench_g/rmsnorm_implementation/rmsnorm_implementation.py`

## Public theorem: `rmsnorm_implementation_output_summary`

<details><summary>docstring</summary>

```
/-- Per-kernel output summary for `rmsnorm_triton`: the DSL surface lowers to the
algorithm layer, and the masked store to `out_ptr` is compute-correct for
arbitrary `N_SIZE` — every output column holds the full-`N` RMSNorm spec
`rmsnormWeightedYFullNSpec`. Built on the multi-block `*_compute_fullN_correct`
result; requires `0 < BLOCK_N_SIZE`, `0 < stride_out_k`, and output/input
disjointness. -/
```
</details>

**Statement:**
```lean
specification rmsnorm_implementation_output_summary
    (x_ptr rms_w_ptr out_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE : Nat)
    (eps : ℝ) (s : BlockState)
    (hBlockPos : 0 < BLOCK_N_SIZE)
    (hStrideOutKPos : 0 < stride_out_k)
    (hXOutNe : x_ptr ≠ out_ptr)
    (hWOutNe : rms_w_ptr ≠ out_ptr) :
    (∃ alg, (rmsnorm_implementation x_ptr rms_w_ptr out_ptr
        stride_x_batch stride_x_m stride_x_k stride_rms_w
        stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE
        eps).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := rmsnorm_implementation x_ptr rms_w_ptr out_ptr
        stride_x_batch stride_x_m stride_x_k stride_rms_w
        stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin N_SIZE => True)
        (fun i => (out_ptr,
          outColOffset s stride_out_batch stride_out_m stride_out_k i.val)))
      (expected := fun i =>
        rmsnormWeightedYFullNSpec s x_ptr rms_w_ptr stride_x_batch stride_x_m
          stride_x_k stride_rms_w N_SIZE BLOCK_N_SIZE eps i)
```

**Assumptions / layout contracts:**
- `hBlockPos : 0 < BLOCK_N_SIZE`
- `hStrideOutKPos : 0 < stride_out_k`
- `hXOutNe : x_ptr ≠ out_ptr`
- `hWOutNe : rms_w_ptr ≠ out_ptr`
- `fun _ : Fin N_SIZE => True`

**Closed-form spec defs (transitive):** `rmsnorm_implementation`, `outColOffset`, `rmsnormWeightedYFullNSpec`, `rmsnormYFullNSpec`, `rmsInvVarFullN`, `rmsVarFullNCarrier`, `xColOffset`

<details><summary><code>rmsnorm_implementation</code></summary>

```
/-- Faithful transcription of `rmsnorm_implementation.py`'s `rmsnorm_triton`.

Allowed mechanical Lean-syntax-only changes:
- Python `N_SIZE: tl.constexpr` / `eps: tl.constexpr` / `BLOCK_N_SIZE: tl.constexpr`
  -> Lean `Nat` / `ℝ` parameters. -/
```
```lean
def rmsnorm_implementation
    (x_ptr rms_w_ptr out_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k : Nat)
    (N_SIZE BLOCK_N_SIZE : Nat) (eps : ℝ) :
    ComputeKernel := triton {
  pid_batch = tl.program_id(0)
  pid_m = tl.program_id(1)
  offset_m = pid_batch * $(stride_x_batch) + pid_m * $(stride_x_m)
  block_n_size = tl.arange(0, $(BLOCK_N_SIZE))
  var = tl.zeros([$(BLOCK_N_SIZE)], tl.float32)
  for block_n_strart_ptr in range(0, $(N_SIZE), $(BLOCK_N_SIZE)) {
    offset_n = block_n_strart_ptr + block_n_size
    x_ptr_mask = offset_n < $(N_SIZE)
    x = tl.load(x_ptr + offset_m + offset_n * $(stride_x_k), mask=x_ptr_mask, other=0.0)
    xf = (x).to(tl.float32)
    var += xf * xf
  }
  var = tl.sum(var, axis=0) / $(N_SIZE)
  std = tl.sqrt(var + $(eps))
  for block_n_strart_ptr in range(0, $(N_SIZE), $(BLOCK_N_SIZE)) {
    offset_n = block_n_strart_ptr + block_n_size
    x_ptr_mask = offset_n < $(N_SIZE)
    rms_w_offset = tl.load(rms_w_ptr + offset_n * $(stride_rms_w), mask=x_ptr_mask)
    x = tl.load(x_ptr + offset_m + offset_n * $(stride_x_k), mask=x_ptr_mask, other=0.0)
    x_new = x / std
    out = x_new * rms_w_offset
    out_offset = pid_batch * $(stride_out_batch) + pid_m * $(stride_out_m) +
      offset_n * $(stride_out_k)
    tl.store(out_ptr + out_offset, out, mask=x_ptr_mask)
  }
}
```
</details>

<details><summary><code>outColOffset</code></summary>

```lean
def outColOffset
    (s : BlockState) (stride_out_batch stride_out_m stride_out_k col : Nat) : Nat :=
  s.pids 0 * stride_out_batch + s.pids 1 * stride_out_m + col * stride_out_k
```
</details>

<details><summary><code>rmsnormWeightedYFullNSpec</code></summary>

```
/-- Full-N RMSNorm output including the learned RMS weight. This is the
Python-observable value written by `rmsnorm_triton` for column `i`. -/
```
```lean
noncomputable def rmsnormWeightedYFullNSpec
    (s : BlockState) (x_ptr rms_w_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w N_SIZE BLOCK_N_SIZE : Nat)
    (eps : ℝ) (i : Fin N_SIZE) : ℝ :=
  rmsnormYFullNSpec s x_ptr stride_x_batch stride_x_m stride_x_k
      N_SIZE BLOCK_N_SIZE eps i *
    s.readMem rms_w_ptr (i.val * stride_rms_w)
```
</details>

<details><summary><code>rmsnormYFullNSpec</code></summary>

```
/-- Multi-block full-N output spec: `x[i] * rmsInvVarFullN` for each
`i < N_SIZE`, expressed against the algebraic ground truth. -/
```
```lean
noncomputable def rmsnormYFullNSpec
    (s : BlockState) (x_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k N_SIZE BLOCK_N_SIZE : Nat)
    (eps : ℝ) (i : Fin N_SIZE) : ℝ :=
  s.readMem x_ptr
      (s.pids 0 * stride_x_batch + s.pids 1 * stride_x_m + i.val * stride_x_k) *
    rmsInvVarFullN s x_ptr stride_x_batch stride_x_m stride_x_k
      N_SIZE BLOCK_N_SIZE eps
```
</details>

<details><summary><code>rmsInvVarFullN</code></summary>

```
/-- Multi-block full-N reciprocal-standard-deviation:
`1 / sqrt(Σ x_j² / N_SIZE + eps)`. -/
```
```lean
noncomputable def rmsInvVarFullN
    (s : BlockState) (x_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k N_SIZE BLOCK_N_SIZE : Nat)
    (eps : ℝ) : ℝ :=
  1 / Real.sqrt
    (rmsVarFullNCarrier s x_ptr stride_x_batch stride_x_m stride_x_k
      N_SIZE BLOCK_N_SIZE / (N_SIZE : ℝ) + eps)
```
</details>

<details><summary><code>rmsVarFullNCarrier</code></summary>

```
/-- Multi-block full-N variance carrier: the algebraic ground truth for
`Σ_{j < N_SIZE} (x[j])²`, independent of any block decomposition. -/
```
```lean
noncomputable def rmsVarFullNCarrier
    (s : BlockState) (x_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k N_SIZE _BLOCK_N_SIZE : Nat) : ℝ :=
  ∑ j : Fin N_SIZE,
    (s.readMem x_ptr
        (xColOffset s stride_x_batch stride_x_m stride_x_k j.val))^2
```
</details>

<details><summary><code>xColOffset</code></summary>

```
/-- Algebraic offset into `x_ptr` for an arbitrary column index `col : Nat`:
row `(batch, m) = (pids 0, pids 1)` at strides `(stride_x_batch, stride_x_m)`,
column `col` at stride `stride_x_k`. -/
```
```lean
def xColOffset
    (s : BlockState) (stride_x_batch stride_x_m stride_x_k col : Nat) : Nat :=
  s.pids 0 * stride_x_batch + s.pids 1 * stride_x_m + col * stride_x_k
```
</details>

## Public theorem: `rmsnorm_implementation_io_correctness`

<details><summary>docstring</summary>

```
/-- **The `⊨[R]` streaming headline (wave-5 S3 per-step emit genre).** For
every rounding model `R`, the faithful `rmsnorm_implementation` surface
implements, on its `StreamEmitMasked2DKernelIO₂` signature, the **ideal ℝ
two-pass RMS-norm** over the streamed tiles: emitted window `(t, j)` holds
`x[t,j] / √(Σ_guarded x²/N_SIZE + eps) · w[t,j]`, where the guarded double
sum folds the *entire* `x` stream — the spec `f` is exact real arithmetic
in the kernel's own division spelling. The kernel has **zero rounding
events** (loads, both passes' arithmetic and the per-step stores are all at
`.real`; the erased `.to(tl.float32)` is `.real → .real`), so the skin's
boundary quantization degenerates: the readback contract's `R.round .real`
is the identity by the model's defining `round_real`, and the `.real`
in-loop stores are exact under `execR R` — the ∀-`R` face holds via the
`RoundingModel` `.real` identity fields, not as a `.triv` special case.

Layer map: both passes are cast-free, so under `execR R` they collapse
verbatim onto the exact stepper and the proven
`rmsVarLoopContextInvariant` / `rmsStdPostLoop_step_to_out_init` /
`rmsOutLoopContextInvariant` invariant stack above is reused unchanged; the
`⊨[R]` face adds the `TraceSafeR` walk, the per-cell memory frame
(`rmsImplOutBody_step_frame`, the `mem` twin of
`rmsOutLoopBody_step_preserves_old_output`), and the stream-lane spec
bridge (`rmsImpl_sum_sq_mean` re-blocking `k ↔ (k/B, k%B)`).

All four hypotheses are truth-forced (exactly the exact headline
`rmsnorm_implementation_output_summary`'s side conditions):

* `hBlockPos : 0 < BLOCK_N_SIZE` — both loops step by `BLOCK_N_SIZE`
  (`range(0, N_SIZE, BLOCK_N_SIZE)`); at `BLOCK_N_SIZE = 0` and
  `N_SIZE > 0` neither `forRange` advances, `execR` cannot terminate the
  way the invariant stack requires, and the step index `i / B` is
  meaningless. It holds for every real launch.
* `hXOutNe : x_ptr ≠ out_ptr`, `hWOutNe : rms_w_ptr ≠ out_ptr` — pass 2
  stores into `out` **between** its re-reads of `x` and `rms_w`; if `out`
  aliased either input, later blocks would re-read already-overwritten
  values and the closed form would be false.
* `hStrideOutKPos : 0 < stride_out_k` — the per-lane write window
  `pid₀·sob + pid₁·som + k·sok` is injective over global lanes only when
  the output column stride is nonzero
  (`outColOffset_fin_injective_of_stride_out_k_pos`); with
  `stride_out_k = 0` all lanes collide and the per-lane readback would be
  last-writer-wins. Always true for a real tensor.

Relation to the exact surface: the exact headline
`rmsnorm_implementation_output_summary` (`Realizes_without_Rounding`) above
is retained unchanged; this `⊨[R]` face restates the same RMS-norm content
on the streaming emit skin, for every `R` at once (at the `.real` grid the
two faces carry the same exact cell). Both faces are kept per the
rounding-as-default doctrine. -/
```
</details>

**Statement:**
```lean
specification rmsnorm_implementation_io_correctness (R : RoundingModel)
    (x_ptr rms_w_ptr out_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE : Nat)
    (eps : ℝ)
    (hBlockPos : 0 < BLOCK_N_SIZE)
    (hXOutNe : x_ptr ≠ out_ptr) (hWOutNe : rms_w_ptr ≠ out_ptr)
    (hStrideOutKPos : 0 < stride_out_k) :
    rmsnormImplementationKernelIO x_ptr rms_w_ptr out_ptr stride_x_batch
        stride_x_m stride_x_k stride_rms_w stride_out_batch stride_out_m
        stride_out_k N_SIZE BLOCK_N_SIZE eps ⊨[R]
      fun _ _ xs ws t j =>
        rmsImplStreamSpec N_SIZE BLOCK_N_SIZE eps xs ws t j
```

**Assumptions / layout contracts:**
- `hBlockPos : 0 < BLOCK_N_SIZE`
- `hXOutNe : x_ptr ≠ out_ptr`
- `hWOutNe : rms_w_ptr ≠ out_ptr`
- `hStrideOutKPos : 0 < stride_out_k`

**Closed-form spec defs (transitive):** `rmsnormImplementationKernelIO`, `rmsImplStreamSpec`, `rmsnorm_implementation`, `rmsNumSteps`, `rmsImplStreamSumSq`

<details><summary><code>rmsnormImplementationKernelIO</code></summary>

```
/-- **Streaming IO signature** of `rmsnorm_implementation` on the two-stream
per-step emit skin (S3: in-loop store). Step `t` of either pass (at
`block_n_strart_ptr = t·BLOCK_N_SIZE`) reads the `BLOCK_N_SIZE`-lane `x`
tile (`read1`, both passes read the same addresses) and the `rms_w` tile
(`read2`, pass 2 only); step `t` of pass 2 stores the `BLOCK_N_SIZE`-lane
output window (`write`) at the **`.real`** grid (`outDType` default — the
kernel's store is untyped `tl.store(out_ptr + out_offset, out)` at `.real`,
so the per-step stores have no quantization event). The windows transcribe
the kernel's pointer arithmetic verbatim
(`offset_n = t·BLOCK_N_SIZE + j`):

* `read1` step `t`, lane `j`:
  `pid₀·stride_x_batch + pid₁·stride_x_m + (t·BLOCK_N_SIZE + j)·stride_x_k`
  — the kernel's `x_ptr + offset_m + offset_n * stride_x_k`.
* `read2` step `t`, lane `j`: `(t·BLOCK_N_SIZE + j)·stride_rms_w` — the
  kernel's `rms_w_ptr + offset_n * stride_rms_w`.
* `write` step `t`, lane `j`:
  `pid₀·stride_out_batch + pid₁·stride_out_m + (t·BLOCK_N_SIZE + j)·stride_out_k`
  — the kernel's `out_offset`.

All three masks are the kernel's single `x_ptr_mask`:
`t·BLOCK_N_SIZE + j < N_SIZE`. -/
```
```lean
def rmsnormImplementationKernelIO (x w o : RegionName)
    (sxb sxm sxk srw sob som sok N B : Nat) (eps : ℝ) :
    StreamEmitMasked2DKernelIO₂ where
  kernel := rmsnorm_implementation x w o sxb sxm sxk srw sob som sok N B eps
  inp1 := x
  inp2 := w
  out := o
  T := rmsNumSteps N B
  B1 := B
  B2 := B
  C := B
  read1 := fun p₀ p₁ t j => p₀ * sxb + p₁ * sxm + (t.val * B + j.val) * sxk
  read2 := fun _ _ t j => (t.val * B + j.val) * srw
  write := fun p₀ p₁ t j => p₀ * sob + p₁ * som + (t.val * B + j.val) * sok
  mask1 := fun _ _ t j => t.val * B + j.val < N
  mask2 := fun _ _ t j => t.val * B + j.val < N
  writeMask := fun _ _ t j => t.val * B + j.val < N
```
</details>

<details><summary><code>rmsImplStreamSpec</code></summary>

```
/-- The stream-level RMS-norm spec (the genre's two-pass shape, in this
kernel's **division** spelling): output window `(t, j)` holds the step-`t`
`x` value divided by `std = √(Σ x²/N_SIZE + eps)` — the fold over the
*entire* stream — times the step-`t` `rms_w` value. Algebraically
`rmsnormWeightedYFullNSpec` with the `Fin N_SIZE` sum re-blocked to the
guarded stream double sum. -/
```
```lean
noncomputable def rmsImplStreamSpec (N B : Nat) (eps : ℝ)
    (xs ws : Fin (rmsNumSteps N B) → Fin B → ℝ)
    (t : Fin (rmsNumSteps N B)) (j : Fin B) : ℝ :=
  xs t j / Real.sqrt (rmsImplStreamSumSq N B xs / (N : ℝ) + eps) * ws t j
```
</details>

<details><summary><code>rmsnorm_implementation</code></summary>

```
/-- Faithful transcription of `rmsnorm_implementation.py`'s `rmsnorm_triton`.

Allowed mechanical Lean-syntax-only changes:
- Python `N_SIZE: tl.constexpr` / `eps: tl.constexpr` / `BLOCK_N_SIZE: tl.constexpr`
  -> Lean `Nat` / `ℝ` parameters. -/
```
```lean
def rmsnorm_implementation
    (x_ptr rms_w_ptr out_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k : Nat)
    (N_SIZE BLOCK_N_SIZE : Nat) (eps : ℝ) :
    ComputeKernel := triton {
  pid_batch = tl.program_id(0)
  pid_m = tl.program_id(1)
  offset_m = pid_batch * $(stride_x_batch) + pid_m * $(stride_x_m)
  block_n_size = tl.arange(0, $(BLOCK_N_SIZE))
  var = tl.zeros([$(BLOCK_N_SIZE)], tl.float32)
  for block_n_strart_ptr in range(0, $(N_SIZE), $(BLOCK_N_SIZE)) {
    offset_n = block_n_strart_ptr + block_n_size
    x_ptr_mask = offset_n < $(N_SIZE)
    x = tl.load(x_ptr + offset_m + offset_n * $(stride_x_k), mask=x_ptr_mask, other=0.0)
    xf = (x).to(tl.float32)
    var += xf * xf
  }
  var = tl.sum(var, axis=0) / $(N_SIZE)
  std = tl.sqrt(var + $(eps))
  for block_n_strart_ptr in range(0, $(N_SIZE), $(BLOCK_N_SIZE)) {
    offset_n = block_n_strart_ptr + block_n_size
    x_ptr_mask = offset_n < $(N_SIZE)
    rms_w_offset = tl.load(rms_w_ptr + offset_n * $(stride_rms_w), mask=x_ptr_mask)
    x = tl.load(x_ptr + offset_m + offset_n * $(stride_x_k), mask=x_ptr_mask, other=0.0)
    x_new = x / std
    out = x_new * rms_w_offset
    out_offset = pid_batch * $(stride_out_batch) + pid_m * $(stride_out_m) +
      offset_n * $(stride_out_k)
    tl.store(out_ptr + out_offset, out, mask=x_ptr_mask)
  }
}
```
</details>

<details><summary><code>rmsNumSteps</code></summary>

```
/-- Trip count of both `for block_n_strart_ptr in range(0, N_SIZE,
BLOCK_N_SIZE)` passes: `⌈N_SIZE / BLOCK_N_SIZE⌉`. -/
```
```lean
def rmsNumSteps (N B : Nat) : Nat := (N + B - 1) / B
```
</details>

<details><summary><code>rmsImplStreamSumSq</code></summary>

```
/-- The guarded stream-level sum of squares: the pass-1 fold `var += xf*xf`
over the whole curried `x` stream, guarded by the kernel's window
(`t·B + e < N`) — the contract only pins `xs` on masked lanes, so the spec
must not read unmasked lanes. -/
```
```lean
noncomputable def rmsImplStreamSumSq (N B : Nat)
    (xs : Fin (rmsNumSteps N B) → Fin B → ℝ) : ℝ :=
  ∑ u : Fin (rmsNumSteps N B), ∑ e : Fin B,
    if u.val * B + e.val < N then xs u e ^ 2 else 0
```
</details>

## Also present (pinned special-case summaries)
- `rmsnorm_implementation_compute_correct`
