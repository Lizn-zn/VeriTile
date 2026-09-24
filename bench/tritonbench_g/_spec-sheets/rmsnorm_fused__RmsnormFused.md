# Spec sheet — `bench/tritonbench_g/rmsnorm_fused/RmsnormFused.lean`

**Python source:** `bench/tritonbench_g/rmsnorm_fused/rmsnorm_fused.py`

## Public theorem: `rms_norm_fwd_fused_output_summary`

<details><summary>docstring</summary>

```
/-- Per-kernel output summary for `rms_norm_fwd_fused`: the DSL surface lowers to
the algorithm layer, and the masked store to `Y` is compute-correct — every
active lane (`i.val < N`) holds the RMSNorm spec `rmsnormSpec`, out-of-bounds
lanes are preserved. Stated under the `0 < N ≤ BLOCK_SIZE` single-block launch
precondition chosen by the Python wrapper. -/
```
</details>

**Statement:**
```lean
specification rms_norm_fwd_fused_output_summary
    (X Y W : RegionName) (stride N BLOCK_SIZE : Nat) (eps : ℝ)
    (s : BlockState)
    (hNpos : 0 < N) (hNle : N ≤ BLOCK_SIZE)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => yOffset s stride i)) :
    (∃ alg, (rms_norm_fwd_fused X Y W stride N BLOCK_SIZE eps).toAlgorithm? =
        Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := rms_norm_fwd_fused X Y W stride N BLOCK_SIZE eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < N)
        (fun i => (Y, yOffset s stride i)))
      (expected := fun i => rmsnormSpec s X W stride N BLOCK_SIZE eps i)
```

**Assumptions / layout contracts:**
- `hNpos : 0 < N`
- `hNle : N ≤ BLOCK_SIZE`
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => yOffset s stride i)`
- `fun i : Fin BLOCK_SIZE => i.val < N`

**Closed-form spec defs (transitive):** `yOffset`, `rms_norm_fwd_fused`, `rmsnormSpec`, `rmsLoad`, `rmsWeight`, `xOffset`

<details><summary><code>yOffset</code></summary>

```lean
def yOffset (s : BlockState) (stride : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * stride + i.val
```
</details>

<details><summary><code>rms_norm_fwd_fused</code></summary>

```
/-- Faithful `forRange` transcription of `rmsnorm_fused.py`'s
`rms_norm_fwd_fused`.

The Python wrapper chooses `BLOCK_SIZE >= N` and raises otherwise, so the
correctness theorem below proves the full loop-shaped kernel under that
runtime precondition. Under the precondition both `range(0, N, BLOCK_SIZE)`
loops execute exactly the `off = 0` iteration.

Allowed mechanical Lean-syntax-only changes:
- Python `N` / `BLOCK_SIZE: tl.constexpr` -> Lean `Nat` parameters. -/
```
```lean
def rms_norm_fwd_fused
    (X Y W : RegionName) (stride N BLOCK_SIZE : Nat) (eps : ℝ) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  Y += row * $(stride)
  X += row * $(stride)
  _var = tl.zeros([$(BLOCK_SIZE)], dtype=tl.float32)
  for off in range(0, $(N), $(BLOCK_SIZE)) {
    cols = off + tl.arange(0, $(BLOCK_SIZE))
    x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
    x = tl.where(cols < $(N), x, 0.0)
    _var += x * x
  }
  var = tl.sum(_var, axis=0) / $(N)
  rstd = 1 / tl.sqrt(var + $(eps))
  for off in range(0, $(N), $(BLOCK_SIZE)) {
    cols = off + tl.arange(0, $(BLOCK_SIZE))
    mask = cols < $(N)
    w = tl.load(W + cols, mask=mask)
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    x_hat = x * rstd
    y = x_hat * w
    tl.store(Y + cols, y, mask=mask)
  }
}
```
</details>

<details><summary><code>rmsnormSpec</code></summary>

```lean
noncomputable def rmsnormSpec
    (s : BlockState) (X W : RegionName)
    (stride N BLOCK_SIZE : Nat) (eps : ℝ) (i : Fin BLOCK_SIZE) : ℝ :=
  TiledRMSNorm.rmsAffine
    (rmsLoad s X stride N BLOCK_SIZE)
    (rmsWeight s W)
    N eps i
```
</details>

<details><summary><code>rmsLoad</code></summary>

```lean
noncomputable def rmsLoad
    (s : BlockState) (X : RegionName) (stride N BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : ℝ :=
  Tile.maskedRowLoad
    (fun k : Fin BLOCK_SIZE => s.readMem X (xOffset s stride k))
    (fun k : Fin BLOCK_SIZE => k.val < N)
    0
    i
```
</details>

<details><summary><code>rmsWeight</code></summary>

```lean
noncomputable def rmsWeight
    (s : BlockState) (W : RegionName) (i : Fin BLOCK_SIZE) : ℝ :=
  s.readMem W i.val
```
</details>

<details><summary><code>xOffset</code></summary>

```lean
def xOffset (s : BlockState) (stride : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * stride + i.val
```
</details>

## Public theorem: `rms_norm_fwd_fused_io_correctness`

<details><summary>docstring</summary>

```
/-- **The `⊨[R]` streaming headline (wave-5 S3 per-step emit genre).** For
every rounding model `R`, the faithful `rms_norm_fwd_fused` surface
implements, on its `StreamEmitMasked2DKernelIO₂` signature, the **ideal ℝ
two-pass RMS-norm** over the streamed tiles: emitted window `(t, j)` holds
`x[t,j] · (√(Σ_guarded x²/N + eps))⁻¹ · w[t,j]`, where the guarded double sum
folds the *entire* `x` stream — the spec `f` is exact real arithmetic. The
full Hoare triple is the skin's: for every disjoint flat allocation of
`[X, W, Y]`, every pid pair, and every launch state with both masked input
streams pinned, `execR R` terminates, every write-active emitted cell reads
back through `readMemAs .real` as the ideal value, and every flat cell
outside the active output window is untouched.

Rounding story: the kernel has **zero rounding events**. Both masked loads,
both passes' arithmetic and the in-loop store are at `.real`, and the two
`.to(tl.float32)` casts are erased outright by the compute-to-algorithm
lowering (`fused_body_decomp` is `rfl` onto a body with no `Op.castFloat`).
The skin's boundary quantization therefore degenerates: the readback
contract's `R.round .real` is the identity by the model's defining
`round_real`, and the `.real` in-loop stores are exact under `execR R` — the
∀-`R` face holds via the `RoundingModel` `.real` identity fields, not as a
`.triv` special case.

Layer map: `fused_castFree` collapses `execR R` onto the exact `exec`
statement by statement (the two `forRange` clauses lift their cast-free
bodies through `stepForRangeAuxR_castFree`), so the proven exact readback
`rms_norm_fwd_fused_correct` is reused unchanged; the `⊨[R]` face adds the
`TraceSafeR` walk (`rms_norm_fwd_fused_traceSafeR`), termination
(`rms_norm_fwd_fused_terminates`), the per-cell memory frame
(`rms_norm_fwd_fused_frame`), and the stream-lane spec bridge
(`rmsFusedStreamSpec_eq_rmsnormSpec`).

Both hypotheses are the *exact* headline `rms_norm_fwd_fused_output_summary`'s
own launch precondition, with the same provenance:

* `hNpos : 0 < N`, `hNle : N ≤ BLOCK_SIZE` — the Python wrapper picks
  `BLOCK_SIZE ≥ N` and raises otherwise, so every real launch is in this
  single-block regime; there both `range(0, N, BLOCK_SIZE)` passes execute
  exactly the `off = 0` iteration, which is what makes the stream one step
  wide (`T := 1`) and the closed form available.

The exact headline's third side condition `hOutInj` is **not** carried here:
the write window `pid₀·stride + i` is injective in `i` outright, and this
face discharges it.

Relation to the exact surface: the exact headline
`rms_norm_fwd_fused_output_summary` (`Realizes_without_Rounding`) above is
retained unchanged; this `⊨[R]` face restates the same RMS-norm content on
the streaming emit skin, for every `R` at once (at the `.real` grid the two
faces carry the same exact cell). Both faces are kept per the
rounding-as-default doctrine. -/
```
</details>

**Statement:**
```lean
specification rms_norm_fwd_fused_io_correctness (R : RoundingModel)
    (X Y W : RegionName) (stride N BLOCK_SIZE : Nat) (eps : ℝ)
    (hNpos : 0 < N) (hNle : N ≤ BLOCK_SIZE) :
    rmsnormFusedKernelIO X Y W stride N BLOCK_SIZE eps ⊨[R]
      fun _ _ xs ws t j => rmsFusedStreamSpec N BLOCK_SIZE eps xs ws t j
```

**Assumptions / layout contracts:**
- `hNpos : 0 < N`
- `hNle : N ≤ BLOCK_SIZE`

**Closed-form spec defs (transitive):** `rmsnormFusedKernelIO`, `rmsFusedStreamSpec`, `rms_norm_fwd_fused`, `rmsFusedStreamSumSq`

<details><summary><code>rmsnormFusedKernelIO</code></summary>

```
/-- **Streaming IO signature** of `rms_norm_fwd_fused` on the two-stream
per-step emit skin (S3: in-loop store), in the Python wrapper's single-block
launch regime `0 < N ≤ BLOCK_SIZE`: both `range(0, N, BLOCK_SIZE)` passes run
exactly the `off = 0` iteration, so the stream has `T := 1` step of
`BLOCK_SIZE` lanes. Step `t` reads the `BLOCK_SIZE`-lane `x` tile (`read1`,
both passes read the same addresses) and the `w` tile (`read2`, pass 2 only);
step `t` of pass 2 stores the `BLOCK_SIZE`-lane output window (`write`) at
the **`.real`** grid — the kernel's store is the untyped
`tl.store(Y + cols, y, mask=mask)`, which lowers to
`Stmt.store TileDType.real` (see `fusedWbBody`), so the per-step stores carry
no quantization event. The windows transcribe the kernel's pointer arithmetic
verbatim (`cols = off + tl.arange(0, BLOCK_SIZE)`, i.e.
`t·BLOCK_SIZE + j`, against the row-bumped `X`/`Y` pointers):

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
def rmsnormFusedKernelIO (X Y W : RegionName) (stride N BLOCK_SIZE : Nat)
    (eps : ℝ) : StreamEmitMasked2DKernelIO₂ where
  kernel := rms_norm_fwd_fused X Y W stride N BLOCK_SIZE eps
  inp1 := X
  inp2 := W
  out := Y
  T := 1
  B1 := BLOCK_SIZE
  B2 := BLOCK_SIZE
  C := BLOCK_SIZE
  outDType := .real
  read1 := fun p₀ _ t j => p₀ * stride + (t.val * BLOCK_SIZE + j.val)
  read2 := fun _ _ t j => t.val * BLOCK_SIZE + j.val
  write := fun p₀ _ t j => p₀ * stride + (t.val * BLOCK_SIZE + j.val)
  mask1 := fun _ _ t j => t.val * BLOCK_SIZE + j.val < N
  mask2 := fun _ _ t j => t.val * BLOCK_SIZE + j.val < N
  writeMask := fun _ _ t j => t.val * BLOCK_SIZE + j.val < N
```
</details>

<details><summary><code>rmsFusedStreamSpec</code></summary>

```
/-- The stream-level RMS-norm spec (the genre's two-pass shape): output
window `(t, j)` holds the step-`t` `x` value times
`rsqrt(Σ x²/N + eps)` — the fold over the *entire* stream — times the
step-`t` `w` value. Algebraically `TiledRMSNorm.rmsAffine` with the
`Fin BLOCK_SIZE` guarded sum re-read as the stream double sum. -/
```
```lean
noncomputable def rmsFusedStreamSpec (N B : Nat) (eps : ℝ)
    (xs ws : Fin 1 → Fin B → ℝ) (t : Fin 1) (j : Fin B) : ℝ :=
  xs t j * (Real.sqrt (rmsFusedStreamSumSq N B xs / (N : ℝ) + eps))⁻¹ * ws t j
```
</details>

<details><summary><code>rms_norm_fwd_fused</code></summary>

```
/-- Faithful `forRange` transcription of `rmsnorm_fused.py`'s
`rms_norm_fwd_fused`.

The Python wrapper chooses `BLOCK_SIZE >= N` and raises otherwise, so the
correctness theorem below proves the full loop-shaped kernel under that
runtime precondition. Under the precondition both `range(0, N, BLOCK_SIZE)`
loops execute exactly the `off = 0` iteration.

Allowed mechanical Lean-syntax-only changes:
- Python `N` / `BLOCK_SIZE: tl.constexpr` -> Lean `Nat` parameters. -/
```
```lean
def rms_norm_fwd_fused
    (X Y W : RegionName) (stride N BLOCK_SIZE : Nat) (eps : ℝ) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  Y += row * $(stride)
  X += row * $(stride)
  _var = tl.zeros([$(BLOCK_SIZE)], dtype=tl.float32)
  for off in range(0, $(N), $(BLOCK_SIZE)) {
    cols = off + tl.arange(0, $(BLOCK_SIZE))
    x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
    x = tl.where(cols < $(N), x, 0.0)
    _var += x * x
  }
  var = tl.sum(_var, axis=0) / $(N)
  rstd = 1 / tl.sqrt(var + $(eps))
  for off in range(0, $(N), $(BLOCK_SIZE)) {
    cols = off + tl.arange(0, $(BLOCK_SIZE))
    mask = cols < $(N)
    w = tl.load(W + cols, mask=mask)
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    x_hat = x * rstd
    y = x_hat * w
    tl.store(Y + cols, y, mask=mask)
  }
}
```
</details>

<details><summary><code>rmsFusedStreamSumSq</code></summary>

```
/-- The guarded stream-level sum of squares: the pass-1 fold `_var += x * x`
over the whole curried `x` stream, guarded by the kernel's window
(`t·BLOCK_SIZE + e < N`) — the skin's contract only pins `xs` on masked
lanes, so the spec must not read unmasked lanes. -/
```
```lean
noncomputable def rmsFusedStreamSumSq (N B : Nat) (xs : Fin 1 → Fin B → ℝ) : ℝ :=
  ∑ u : Fin 1, ∑ e : Fin B, if u.val * B + e.val < N then xs u e ^ 2 else 0
```
</details>

## Also present (pinned special-case summaries)
- `rms_norm_fwd_fused_compute_correct`
