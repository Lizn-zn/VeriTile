# Spec sheet — `bench/tritonbench_g/rmsnorm_triton/RmsnormTriton.lean`

**Python source:** `bench/tritonbench_g/rmsnorm_triton/rmsnorm_triton.py`

## Public theorem: `rmsnorm_full_output_summary`

<details><summary>docstring</summary>

```
/-- **Full output summary**: the surface lowers to the algorithm layer, and the
masked store realizes the genuine multi-block RMS-norm closed form at every
global lane `k < N_SIZE`. Holds for arbitrary `N_SIZE` — no
`N_SIZE ≤ BLOCK_N_SIZE` hypothesis. -/
```
</details>

**Statement:**
```lean
specification rmsnorm_full_output_summary
    (x w o : RegionName) (sxb sxm sxk srw sob som sok N B : Nat) (eps : ℝ)
    (s : BlockState) (hB : 0 < B) (hNpos : 0 < N)
    (hox : o ≠ x) (how : o ≠ w)
    (hsok : 0 < sok) :
    (∃ alg, (VeriTile.Bench.TritonBenchG.RmsnormTriton.rmsnorm_triton x w o
        sxb sxm sxk srw sob som sok N B eps).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := VeriTile.Bench.TritonBenchG.RmsnormTriton.rmsnorm_triton x w o
        sxb sxm sxk srw sob som sok N B eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin N => True)
        (fun k => (o, outOff s sob som sok k.val)))
      (expected := fun k : Fin N => rmsSpecFull s x w sxb sxm sxk srw N eps k.val)
```

**Assumptions / layout contracts:**
- `hB : 0 < B`
- `hNpos : 0 < N`
- `hox : o ≠ x`
- `how : o ≠ w`
- `hsok : 0 < sok`

**Closed-form spec defs (transitive):** `rmsnorm_triton`, `outOff`, `rmsSpecFull`, `xOff`, `meanSq`

<details><summary><code>rmsnorm_triton</code></summary>

```
/-- Faithful transcription of `rmsnorm_triton.py`'s `rmsnorm_triton`.

Allowed mechanical Lean-syntax-only changes:
- Python `N_SIZE: tl.constexpr` / `eps: tl.constexpr` / `BLOCK_N_SIZE: tl.constexpr`
  -> Lean `Nat` / `ℝ` parameters. -/
```
```lean
def rmsnorm_triton
    (x_ptr rms_w_ptr output_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k : Nat)
    (N_SIZE BLOCK_N_SIZE : Nat) (eps : ℝ) :
    ComputeKernel := triton {
  pid_batch = tl.program_id(0)
  pid_m = tl.program_id(1)
  offs_m = pid_batch * $(stride_x_batch) + pid_m * $(stride_x_m)
  block_N = tl.arange(0, $(BLOCK_N_SIZE))
  var = tl.zeros([$(BLOCK_N_SIZE)], tl.float32)
  for block_n_start_idx in range(0, $(N_SIZE), $(BLOCK_N_SIZE)) {
    offs_n = block_n_start_idx + block_N
    x_ptr_mask = offs_n < $(N_SIZE)
    x = tl.load(x_ptr + offs_m + offs_n * $(stride_x_k), mask=x_ptr_mask, other=0.0)
    var += tl.extra.cuda.libdevice.pow((x).to(tl.float32), 2)
  }
  var = tl.sum(var, axis=0) / $(N_SIZE)
  rstd = tl.math.rsqrt(var + $(eps))
  for block_n_start_idx in range(0, $(N_SIZE), $(BLOCK_N_SIZE)) {
    offs_n = block_n_start_idx + block_N
    x_ptr_mask = offs_n < $(N_SIZE)
    rms_w = tl.load(rms_w_ptr + offs_n * $(stride_rms_w), mask=x_ptr_mask)
    x = tl.load(x_ptr + offs_m + offs_n * $(stride_x_k), mask=x_ptr_mask, other=0.0).to(tl.float32)
    x_hat = x * rstd
    out = x_hat * rms_w
    out_off = pid_batch * $(stride_out_batch) + pid_m * $(stride_out_m) +
      offs_n * $(stride_out_k)
    tl.store(output_ptr + out_off, out, mask=x_ptr_mask)
  }
}
```
</details>

<details><summary><code>outOff</code></summary>

```lean
def outOff (s : BlockState) (sob som sok : Nat) (k : Nat) : Nat :=
  s.pids 0 * sob + s.pids 1 * som + k * sok

end Wb

namespace VarLoop
open ScratchRms

-- var-loop invariant
```
</details>

<details><summary><code>rmsSpecFull</code></summary>

```lean
noncomputable def rmsSpecFull (s : BlockState) (x w : RegionName) (sxb sxm sxk srw N : Nat) (eps : ℝ) (k : Nat) : ℝ :=
  s.readMem x (xOff s sxb sxm sxk k) * (WithBot.unbotD 0 (WithBot.realRsqrt (some (meanSq s x sxb sxm sxk N + eps)))) * s.readMem w (k*srw)
```
</details>

<details><summary><code>xOff</code></summary>

```lean
def xOff (s : BlockState) (sxb sxm sxk : Nat) (k : Nat) : Nat :=
  s.pids 0 * sxb + s.pids 1 * sxm + k * sxk
```
</details>

<details><summary><code>meanSq</code></summary>

```lean
noncomputable def meanSq (s : BlockState) (x : RegionName) (sxb sxm sxk N : Nat) : ℝ :=
  (∑ k : Fin N, (s.readMem x (xOff s sxb sxm sxk k.val))^2) / (N:ℝ)
```
</details>

## Public theorem: `rmsnorm_full_io_correctness`

<details><summary>docstring</summary>

```
/-- **The `⊨[R]` streaming headline (wave-5 S3 per-step emit genre).** For
every rounding model `R`, the faithful `rmsnorm_triton` surface implements,
on its `StreamEmitMasked2DKernelIO₂` signature, the **ideal ℝ two-pass
RMS-norm** over the streamed tiles: emitted window `(t, j)` holds
`x[t,j] · (√(Σ_guarded x²/N_SIZE + eps))⁻¹ · w[t,j]`, where the guarded
double sum folds the *entire* `x` stream — the spec `f` is exact real
arithmetic. The kernel has **zero rounding events** (loads, both passes'
arithmetic and the per-step stores are all at `.real`; the erased
`.to(tl.float32)`s are `.real → .real`), so the skin's boundary
quantization degenerates: the readback contract's `R.round .real` is the
identity by the model's defining `round_real`, and the `.real` in-loop
stores are exact under `execR R` — the ∀-`R` face holds via the
`RoundingModel` `.real` identity fields, not as a `.triv` special case.

Layer map: both passes are cast-free, so under `execR R` they collapse
verbatim onto the exact stepper and the proven
`preLoop` / `varInv` / `postStep` / `wbInv` invariant stack above is reused
unchanged; the `⊨[R]` face adds the `TraceSafeR` walk, the per-cell memory
frame (`rms_wbBody_step_frame`, the `mem` twin of `wbStep`), and the
stream-lane spec bridge (`sum_sq_mean` re-blocking `k ↔ (k/B, k%B)`).

All four hypotheses are truth-forced (exactly the exact headline
`rmsnorm_full_output_summary`'s side conditions):

* `hB : 0 < BLOCK_N_SIZE` — both loops step by `BLOCK_N_SIZE`
  (`range(0, N_SIZE, BLOCK_N_SIZE)`); at `BLOCK_N_SIZE = 0` and
  `N_SIZE > 0` neither `forRange` advances, `execR` cannot terminate the
  way the invariant stack requires, and the step index `i / B` is
  meaningless. It holds for every real launch.
* `hox : o ≠ x`, `how : o ≠ w` — pass 2 stores into `out` **between** its
  re-reads of `x` and `rms_w`; if `out` aliased either input, later blocks
  would re-read already-overwritten values and the closed form would be
  false.
* `hsok : 0 < stride_out_k` — the per-lane write window
  `pid₀·sob + pid₁·som + k·sok` is injective over global lanes only when
  the output column stride is nonzero (`outOff_injective_of_pos`); with
  `sok = 0` all lanes collide and the per-lane readback would be
  last-writer-wins. Always true for a real tensor.

Relation to the exact surface: the exact headline
`rmsnorm_full_output_summary` (`Realizes_without_Rounding`) above is
retained unchanged; this `⊨[R]` face restates the same RMS-norm content on
the streaming emit skin, for every `R` at once (at the `.real` grid the two
faces carry the same exact cell). Both faces are kept per the
rounding-as-default doctrine. -/
```
</details>

**Statement:**
```lean
specification rmsnorm_full_io_correctness (R : RoundingModel)
    (x w o : RegionName) (sxb sxm sxk srw sob som sok N B : Nat) (eps : ℝ)
    (hB : 0 < B) (hox : o ≠ x) (how : o ≠ w) (hsok : 0 < sok) :
    rmsnormKernelIO x w o sxb sxm sxk srw sob som sok N B eps ⊨[R]
      fun _ _ xs ws t j => rmsStreamSpec N B eps xs ws t j
```

**Assumptions / layout contracts:**
- `hB : 0 < B`
- `hox : o ≠ x`
- `how : o ≠ w`
- `hsok : 0 < sok`

**Closed-form spec defs (transitive):** `rmsnormKernelIO`, `rmsStreamSpec`, `rmsnorm_triton`, `rmsNumSteps`, `rmsStreamSumSq`

<details><summary><code>rmsnormKernelIO</code></summary>

```
/-- **Streaming IO signature** of `rmsnorm_triton` on the two-stream
per-step emit skin (S3: in-loop store). Step `t` of either pass (at
`block_n_start_idx = t·BLOCK_N_SIZE`) reads the `BLOCK_N_SIZE`-lane `x`
tile (`read1`, both passes read the same addresses) and the `rms_w` tile
(`read2`, pass 2 only); step `t` of pass 2 stores the `BLOCK_N_SIZE`-lane
output window (`write`) at the **`.real`** grid (`outDType` default — the
kernel's store is untyped `tl.store(output_ptr + out_off, out)` at `.real`,
so the per-step stores have no quantization event). The windows transcribe
the kernel's pointer arithmetic verbatim (`offs_n = t·BLOCK_N_SIZE + j`):

* `read1` step `t`, lane `j`:
  `pid₀·stride_x_batch + pid₁·stride_x_m + (t·BLOCK_N_SIZE + j)·stride_x_k`
  — the kernel's `x_ptr + offs_m + offs_n * stride_x_k`.
* `read2` step `t`, lane `j`: `(t·BLOCK_N_SIZE + j)·stride_rms_w` — the
  kernel's `rms_w_ptr + offs_n * stride_rms_w`.
* `write` step `t`, lane `j`:
  `pid₀·stride_out_batch + pid₁·stride_out_m + (t·BLOCK_N_SIZE + j)·stride_out_k`
  — the kernel's `out_off`.

All three masks are the kernel's single `x_ptr_mask`:
`t·BLOCK_N_SIZE + j < N_SIZE`. -/
```
```lean
def rmsnormKernelIO (x w o : RegionName)
    (sxb sxm sxk srw sob som sok N B : Nat) (eps : ℝ) :
    StreamEmitMasked2DKernelIO₂ where
  kernel := rmsnorm_triton x w o sxb sxm sxk srw sob som sok N B eps
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

<details><summary><code>rmsStreamSpec</code></summary>

```
/-- The stream-level RMS-norm spec (the genre's two-pass shape): output
window `(t, j)` holds the step-`t` `x` value times
`rsqrt(Σ x²/N_SIZE + eps)` — the fold over the *entire* stream — times the
step-`t` `rms_w` value. Algebraically `rmsSpecFull` with the `Fin N_SIZE`
sum re-blocked to the guarded stream double sum. -/
```
```lean
noncomputable def rmsStreamSpec (N B : Nat) (eps : ℝ)
    (xs ws : Fin (rmsNumSteps N B) → Fin B → ℝ)
    (t : Fin (rmsNumSteps N B)) (j : Fin B) : ℝ :=
  xs t j * (Real.sqrt (rmsStreamSumSq N B xs / (N : ℝ) + eps))⁻¹ * ws t j
```
</details>

<details><summary><code>rmsnorm_triton</code></summary>

```
/-- Faithful transcription of `rmsnorm_triton.py`'s `rmsnorm_triton`.

Allowed mechanical Lean-syntax-only changes:
- Python `N_SIZE: tl.constexpr` / `eps: tl.constexpr` / `BLOCK_N_SIZE: tl.constexpr`
  -> Lean `Nat` / `ℝ` parameters. -/
```
```lean
def rmsnorm_triton
    (x_ptr rms_w_ptr output_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k : Nat)
    (N_SIZE BLOCK_N_SIZE : Nat) (eps : ℝ) :
    ComputeKernel := triton {
  pid_batch = tl.program_id(0)
  pid_m = tl.program_id(1)
  offs_m = pid_batch * $(stride_x_batch) + pid_m * $(stride_x_m)
  block_N = tl.arange(0, $(BLOCK_N_SIZE))
  var = tl.zeros([$(BLOCK_N_SIZE)], tl.float32)
  for block_n_start_idx in range(0, $(N_SIZE), $(BLOCK_N_SIZE)) {
    offs_n = block_n_start_idx + block_N
    x_ptr_mask = offs_n < $(N_SIZE)
    x = tl.load(x_ptr + offs_m + offs_n * $(stride_x_k), mask=x_ptr_mask, other=0.0)
    var += tl.extra.cuda.libdevice.pow((x).to(tl.float32), 2)
  }
  var = tl.sum(var, axis=0) / $(N_SIZE)
  rstd = tl.math.rsqrt(var + $(eps))
  for block_n_start_idx in range(0, $(N_SIZE), $(BLOCK_N_SIZE)) {
    offs_n = block_n_start_idx + block_N
    x_ptr_mask = offs_n < $(N_SIZE)
    rms_w = tl.load(rms_w_ptr + offs_n * $(stride_rms_w), mask=x_ptr_mask)
    x = tl.load(x_ptr + offs_m + offs_n * $(stride_x_k), mask=x_ptr_mask, other=0.0).to(tl.float32)
    x_hat = x * rstd
    out = x_hat * rms_w
    out_off = pid_batch * $(stride_out_batch) + pid_m * $(stride_out_m) +
      offs_n * $(stride_out_k)
    tl.store(output_ptr + out_off, out, mask=x_ptr_mask)
  }
}
```
</details>

<details><summary><code>rmsNumSteps</code></summary>

```
/-- Trip count of both `for block_n_start_idx in range(0, N_SIZE,
BLOCK_N_SIZE)` passes: `⌈N_SIZE / BLOCK_N_SIZE⌉`. -/
```
```lean
def rmsNumSteps (N B : Nat) : Nat := (N + B - 1) / B
```
</details>

<details><summary><code>rmsStreamSumSq</code></summary>

```
/-- The guarded stream-level sum of squares: the pass-1 fold
`var += pow(x, 2)` over the whole curried `x` stream, guarded by the
kernel's window (`t·B + e < N`) — the contract only pins `xs` on masked
lanes, so the spec must not read unmasked lanes. -/
```
```lean
noncomputable def rmsStreamSumSq (N B : Nat)
    (xs : Fin (rmsNumSteps N B) → Fin B → ℝ) : ℝ :=
  ∑ u : Fin (rmsNumSteps N B), ∑ e : Fin B,
    if u.val * B + e.val < N then xs u e ^ 2 else 0
```
</details>

## Also present (pinned special-case summaries)
- `rmsnorm_full_compute_correct`
