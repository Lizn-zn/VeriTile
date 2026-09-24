# Spec sheet — `bench/tritonbench_g/rmsnorm_fused_llama/RmsnormFusedLlama.lean`

**Python source:** `bench/tritonbench_g/rmsnorm_fused_llama/rmsnorm_fused_llama.py`

## Public theorem: `rms_norm_fwd_fused_llama_output_summary`

<details><summary>docstring</summary>

```
/-- Per-kernel output summary for `_rms_norm_fwd_fused` (Llama): the DSL surface
lowers to the algorithm layer, and the masked fp16 store to `Y` is
compute-correct — every active lane (`i.val < N`) holds the fp16-cast RMSNorm
spec, out-of-bounds lanes are preserved. Stated under the `0 < N ≤ BLOCK_SIZE`
single-block launch precondition chosen by the Python wrapper. -/
```
</details>

**Statement:**
```lean
specification rms_norm_fwd_fused_llama_output_summary
    (X Y W : RegionName) (stride N BLOCK_SIZE : Nat) (eps : ℝ)
    (s : BlockState)
    (hNpos : 0 < N) (hNle : N ≤ BLOCK_SIZE)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => yOffset s stride i)) :
    (∃ alg, (rms_norm_fwd_fused_llama X Y W stride N BLOCK_SIZE eps).toAlgorithm? =
        Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := rms_norm_fwd_fused_llama X Y W stride N BLOCK_SIZE eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < N)
        (fun i => (Y, yOffset s stride i)))
      (expected := fun i =>
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (rmsnormSpec s X W stride N BLOCK_SIZE eps i))))
```

**Assumptions / layout contracts:**
- `hNpos : 0 < N`
- `hNle : N ≤ BLOCK_SIZE`
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => yOffset s stride i)`
- `fun i : Fin BLOCK_SIZE => i.val < N`

**Closed-form spec defs (transitive):** `yOffset`, `rms_norm_fwd_fused_llama`, `rmsnormSpec`, `rmsnormCarrierSpec`, `xOffset`, `rmsInvCarrier`, `rmsVarCarrier`, `rmsInputTile`

<details><summary><code>yOffset</code></summary>

```lean
def yOffset (s : BlockState) (stride : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * stride + i.val
```
</details>

<details><summary><code>rms_norm_fwd_fused_llama</code></summary>

```
/-- Faithful `forRange` transcription of `rmsnorm_fused_llama.py`'s
`_rms_norm_fwd_fused`.

The Python wrapper fixes `BLOCK_SIZE = 16384` after checking `N <= BLOCK_SIZE`,
so the correctness theorem below proves the loop-shaped kernel under that
runtime precondition. Under the precondition both `range(0, N, BLOCK_SIZE)`
loops execute exactly the `off = 0` iteration.

Allowed mechanical Lean-syntax-only changes:
- Python `N` / `BLOCK_SIZE: tl.constexpr` -> Lean `Nat` parameters. -/
```
```lean
def rms_norm_fwd_fused_llama
    (X Y W : RegionName) (stride N BLOCK_SIZE : Nat) (eps : ℝ) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  Y += row * $(stride)
  X += row * $(stride)
  _var = tl.zeros([$(BLOCK_SIZE)], dtype=tl.float32)
  for off in range(0, $(N), $(BLOCK_SIZE)) {
    cols = off + tl.arange(0, $(BLOCK_SIZE))
    x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
    _var += x * x
  }
  var = tl.sum(_var, axis=0) / $(N)
  rstd = 1 / tl.sqrt(var + $(eps))
  for off in range(0, $(N), $(BLOCK_SIZE)) {
    cols = off + tl.arange(0, $(BLOCK_SIZE))
    mask = cols < $(N)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    x_hat = x * rstd
    y = x_hat * w
    tl.store(Y + cols, (y).to(tl.float16), mask=mask)
  }
}
```
</details>

<details><summary><code>rmsnormSpec</code></summary>

```lean
noncomputable def rmsnormSpec
    (s : BlockState) (X W : RegionName)
    (stride N BLOCK_SIZE : Nat) (eps : ℝ) (i : Fin BLOCK_SIZE) : ℝ :=
  rmsnormCarrierSpec s X W stride N BLOCK_SIZE eps i
```
</details>

<details><summary><code>rmsnormCarrierSpec</code></summary>

```lean
noncomputable def rmsnormCarrierSpec
    (s : BlockState) (X W : RegionName)
    (stride N BLOCK_SIZE : Nat) (eps : ℝ) (i : Fin BLOCK_SIZE) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun scaled w => scaled * w)
      (Option.map₂ (fun x inv => x * inv)
        (some (s.readMem X (xOffset s stride i)))
        (rmsInvCarrier s X stride N BLOCK_SIZE eps))
      (some (s.readMem W i.val)))
```
</details>

<details><summary><code>xOffset</code></summary>

```lean
def xOffset (s : BlockState) (stride : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * stride + i.val
```
</details>

<details><summary><code>rmsInvCarrier</code></summary>

```lean
noncomputable def rmsInvCarrier
    (s : BlockState) (X : RegionName) (stride N BLOCK_SIZE : Nat)
    (eps : ℝ) : WithBot ℝ :=
  Option.map (fun b => b⁻¹)
    (WithBot.realSqrt
      (Option.map (fun a => a + eps)
        (rmsVarCarrier s X stride N BLOCK_SIZE)))
```
</details>

<details><summary><code>rmsVarCarrier</code></summary>

```lean
noncomputable def rmsVarCarrier
    (s : BlockState) (X : RegionName) (stride N BLOCK_SIZE : Nat) :
    WithBot ℝ :=
  Option.map₂ (fun a n => a / n)
    ((Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false
      (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
        (rmsInputTile s X stride N BLOCK_SIZE)
        (rmsInputTile s X stride N BLOCK_SIZE))).data PUnit.unit)
    ((Tile.scalar (dtype := .real) (some (N : ℝ) : WithBot ℝ)).data PUnit.unit)
```
</details>

<details><summary><code>rmsInputTile</code></summary>

```lean
noncomputable def rmsInputTile
    (s : BlockState) (X : RegionName) (stride N BLOCK_SIZE : Nat) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      if idx.1.val < N then
        some (s.readMem X (xOffset s stride idx.1))
      else some (0.0 : ℝ) }
```
</details>

## Public theorem: `rms_norm_fwd_fused_llama_io_correctness`

<details><summary>docstring</summary>

```
/-- **The `⊨[R]` streaming headline (wave-5 S3 per-step emit genre).** For
every rounding model `R`, the faithful `_rms_norm_fwd_fused` (Llama) surface
implements, on its `StreamEmitMasked2DKernelIO₂` signature, the **ideal ℝ
two-pass RMS-norm** over the streamed tiles, quantized **once** at the fp16
grid: emitted window `(t, j)` reads back through `readMemAs .fp16` as
`FloatDType.fp16.ofReal (R.round .fp16 (x[t,j] · (√(Σ_guarded x²/N + eps))⁻¹ ·
w[t,j]))`, where the guarded double sum folds the *entire* `x` stream. The
full Hoare triple is the skin's: for every disjoint flat allocation of
`[X, W, Y]`, every pid pair, and every launch state with both masked input
streams pinned, `execR R` terminates, every write-active emitted cell reads
back as above, and every flat cell outside the active output window is
untouched.

Rounding story — the honest part of this port. The kernel's store is
`tl.store(Y + cols, (y).to(tl.float16), mask=mask)`, which lowers to
`Stmt.store TileDType.fp16` carrying an
`Op.castFloat FloatDType.real FloatDType.fp16` (verified by
`llama_body_decomp`, which is `rfl`), so `outDType := .real` would be
*false* here. Under `execR R` the lane value crosses **two** rounding sites
at the same grid — `evalOpR`'s `R.cast .real .fp16` and the typed store's
`R.storeValue .fp16` — and the model's *defining* `round_idem` field
collapses them to a single `R.round .fp16`
(`RoundingModel.storeValue_cast`, packaged as
`llama_storeValue_cast_fp16`). That is exactly one rounding event per
emitted cell, which is what the skin's `outDType := .fp16` contract asserts.
**No hypothesis on `R` is used**: there is no `R.round .fp16 = id` pin, and
the face holds for every rounding model, including genuinely lossy ones.
The two `.to(tl.float32)` casts, by contrast, are erased outright by the
compute-to-algorithm lowering and are not rounding events in this model.

Layer map: the exact `exec`-level stack above cannot be reused verbatim
(pass 2 is *not* cast-free — it is the rounding pass), so the `execR R` run
is redone directly: the loops are unrolled through the `R` mirrors
`stepForRangeAuxR_step_lt` / `_step_ge` (single iteration each, by the
launch regime), termination is `rms_norm_fwd_fused_llama_terminatesR`, the
rounded per-cell readback is `rms_norm_fwd_fused_llama_readbackR` (masked
`writeMemAsR` scatter readback + the single-round collapse), the per-cell
frame is `rms_norm_fwd_fused_llama_frameR`, safety is
`rms_norm_fwd_fused_llama_traceSafeR`, and the exact stack's carrier-level
`rmsnormSpec` is evaluated into real arithmetic by `llama_spec_eq` before
the stream-lane bridge `rmsLlamaStreamSpec_eq_rmsnormSpec`.

Both hypotheses are the *exact* headline
`rms_norm_fwd_fused_llama_output_summary`'s own launch precondition, with
the same provenance:

* `hNpos : 0 < N`, `hNle : N ≤ BLOCK_SIZE` — the Python wrapper fixes
  `BLOCK_SIZE = 16384` after checking `N ≤ BLOCK_SIZE`, so every real launch
  is in this single-block regime; there both `range(0, N, BLOCK_SIZE)`
  passes execute exactly the `off = 0` iteration, which is what makes the
  stream one step wide (`T := 1`) and the closed form available.

The exact headline's third side condition `hOutInj` is **not** carried here:
the write window `pid₀·stride + i` is injective in `i` outright, and this
face discharges it.

Relation to the exact surface: the exact headline
`rms_norm_fwd_fused_llama_output_summary` (`Realizes_without_Rounding`,
whose fp16 store is modeled by the *exact* `FloatDType.cast .real .fp16`)
above is retained unchanged; this `⊨[R]` face restates the same RMS-norm
content on the streaming emit skin with the cast replaced by the abstract
rounding model, for every `R` at once — at `R := .triv` the two faces carry
the same cell. Both faces are kept per the rounding-as-default doctrine. -/
```
</details>

**Statement:**
```lean
specification rms_norm_fwd_fused_llama_io_correctness (R : RoundingModel)
    (X Y W : RegionName) (stride N BLOCK_SIZE : Nat) (eps : ℝ)
    (hNpos : 0 < N) (hNle : N ≤ BLOCK_SIZE) :
    rmsnormFusedLlamaKernelIO X Y W stride N BLOCK_SIZE eps ⊨[R]
      fun _ _ xs ws t j => rmsLlamaStreamSpec N BLOCK_SIZE eps xs ws t j
```

**Assumptions / layout contracts:**
- `hNpos : 0 < N`
- `hNle : N ≤ BLOCK_SIZE`

**Closed-form spec defs (transitive):** `rmsnormFusedLlamaKernelIO`, `rmsLlamaStreamSpec`, `rms_norm_fwd_fused_llama`, `rmsLlamaStreamSumSq`

<details><summary><code>rmsnormFusedLlamaKernelIO</code></summary>

```
/-- **Streaming IO signature** of `_rms_norm_fwd_fused` (Llama) on the
two-stream per-step emit skin (S3: in-loop store), in the Python wrapper's
single-block launch regime `0 < N ≤ BLOCK_SIZE`: both
`range(0, N, BLOCK_SIZE)` passes run exactly the `off = 0` iteration, so the
stream has `T := 1` step of `BLOCK_SIZE` lanes. Step `t` reads the
`BLOCK_SIZE`-lane `x` tile (`read1`, both passes read the same addresses)
and the `w` tile (`read2`, pass 2 only); step `t` of pass 2 stores the
`BLOCK_SIZE`-lane output window (`write`) at the **`.fp16`** grid — the
kernel's store is `tl.store(Y + cols, (y).to(tl.float16), mask=mask)`, which
lowers to `Stmt.store TileDType.fp16` with an
`Op.castFloat FloatDType.real FloatDType.fp16` value (see `llamaWbBody`), so
every emitted cell carries exactly one quantization event. The windows
transcribe the kernel's pointer arithmetic verbatim
(`cols = off + tl.arange(0, BLOCK_SIZE)`, i.e. `t·BLOCK_SIZE + j`, against
the row-bumped `X`/`Y` pointers):

* `read1` step `t`, lane `j`: `pid₀·stride + (t·BLOCK_SIZE + j)` — the
  kernel's `X + cols` after `X += row * stride`.
* `read2` step `t`, lane `j`: `t·BLOCK_SIZE + j` — the kernel's `W + cols`
  (`W` is not row-bumped).
* `write` step `t`, lane `j`: `pid₀·stride + (t·BLOCK_SIZE + j)` — the
  kernel's `Y + cols` after `Y += row * stride`.

All three masks are the kernel's single `cols < N`. The second pid is inert
(the launch grid is 1-D). -/
```
```lean
def rmsnormFusedLlamaKernelIO (X Y W : RegionName) (stride N BLOCK_SIZE : Nat)
    (eps : ℝ) : StreamEmitMasked2DKernelIO₂ where
  kernel := rms_norm_fwd_fused_llama X Y W stride N BLOCK_SIZE eps
  inp1 := X
  inp2 := W
  out := Y
  T := 1
  B1 := BLOCK_SIZE
  B2 := BLOCK_SIZE
  C := BLOCK_SIZE
  outDType := .fp16
  read1 := fun p₀ _ t j => p₀ * stride + (t.val * BLOCK_SIZE + j.val)
  read2 := fun _ _ t j => t.val * BLOCK_SIZE + j.val
  write := fun p₀ _ t j => p₀ * stride + (t.val * BLOCK_SIZE + j.val)
  mask1 := fun _ _ t j => t.val * BLOCK_SIZE + j.val < N
  mask2 := fun _ _ t j => t.val * BLOCK_SIZE + j.val < N
  writeMask := fun _ _ t j => t.val * BLOCK_SIZE + j.val < N
```
</details>

<details><summary><code>rmsLlamaStreamSpec</code></summary>

```
/-- The stream-level RMS-norm spec (the genre's two-pass shape): output
window `(t, j)` holds the step-`t` `x` value times `rsqrt(Σ x²/N + eps)` —
the fold over the *entire* stream — times the step-`t` `w` value. This is
the **ideal ℝ** value; the fp16 quantization lives in the skin's readback
contract, not in `f`. -/
```
```lean
noncomputable def rmsLlamaStreamSpec (N B : Nat) (eps : ℝ)
    (xs ws : Fin 1 → Fin B → ℝ) (t : Fin 1) (j : Fin B) : ℝ :=
  xs t j * (Real.sqrt (rmsLlamaStreamSumSq N B xs / (N : ℝ) + eps))⁻¹ * ws t j
```
</details>

<details><summary><code>rms_norm_fwd_fused_llama</code></summary>

```
/-- Faithful `forRange` transcription of `rmsnorm_fused_llama.py`'s
`_rms_norm_fwd_fused`.

The Python wrapper fixes `BLOCK_SIZE = 16384` after checking `N <= BLOCK_SIZE`,
so the correctness theorem below proves the loop-shaped kernel under that
runtime precondition. Under the precondition both `range(0, N, BLOCK_SIZE)`
loops execute exactly the `off = 0` iteration.

Allowed mechanical Lean-syntax-only changes:
- Python `N` / `BLOCK_SIZE: tl.constexpr` -> Lean `Nat` parameters. -/
```
```lean
def rms_norm_fwd_fused_llama
    (X Y W : RegionName) (stride N BLOCK_SIZE : Nat) (eps : ℝ) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  Y += row * $(stride)
  X += row * $(stride)
  _var = tl.zeros([$(BLOCK_SIZE)], dtype=tl.float32)
  for off in range(0, $(N), $(BLOCK_SIZE)) {
    cols = off + tl.arange(0, $(BLOCK_SIZE))
    x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
    _var += x * x
  }
  var = tl.sum(_var, axis=0) / $(N)
  rstd = 1 / tl.sqrt(var + $(eps))
  for off in range(0, $(N), $(BLOCK_SIZE)) {
    cols = off + tl.arange(0, $(BLOCK_SIZE))
    mask = cols < $(N)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    x_hat = x * rstd
    y = x_hat * w
    tl.store(Y + cols, (y).to(tl.float16), mask=mask)
  }
}
```
</details>

<details><summary><code>rmsLlamaStreamSumSq</code></summary>

```
/-- The guarded stream-level sum of squares: the pass-1 fold `_var += x * x`
over the whole curried `x` stream, guarded by the kernel's window
(`t·BLOCK_SIZE + e < N`) — the skin's contract only pins `xs` on masked
lanes, so the spec must not read unmasked lanes. -/
```
```lean
noncomputable def rmsLlamaStreamSumSq (N B : Nat) (xs : Fin 1 → Fin B → ℝ) : ℝ :=
  ∑ u : Fin 1, ∑ e : Fin B, if u.val * B + e.val < N then xs u e ^ 2 else 0
```
</details>

## Also present (pinned special-case summaries)
- `rms_norm_fwd_fused_llama_compute_correct`
