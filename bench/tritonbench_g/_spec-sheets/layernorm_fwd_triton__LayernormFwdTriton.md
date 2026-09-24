# Spec sheet — `bench/tritonbench_g/layernorm_fwd_triton/LayernormFwdTriton.lean`

**Python source:** `bench/tritonbench_g/layernorm_fwd_triton/layernorm_fwd_triton.py`

## Public theorem: `layernorm_fwd_triton_output_summary`

<details><summary>docstring</summary>

```
/-- Per-kernel output summary for `_layer_norm_fwd_kernel`: the DSL surface
lowers to the algorithm layer, and the masked store to `Y` is compute-correct
for arbitrary `N` — every output column holds the full-`N` LayerNorm spec
`layernormYFullNSpec`. Built on the multi-block `*_compute_fullN_correct`
result; requires only `0 < BLOCK_SIZE` and output/input disjointness. -/
```
</details>

**Statement:**
```lean
specification layernorm_fwd_triton_output_summary
    (X W Y : RegionName)
    (stride_x_N stride_x_hn stride_x_hd
      stride_y_N stride_y_hn stride_y_hd stride_w_hn stride_w_hd
      N BLOCK_SIZE : Nat)
    (eps : ℝ) (s : BlockState)
    (hBlockPos : 0 < BLOCK_SIZE)
    (hXYNe : X ≠ Y)
    (hWYNe : W ≠ Y) :
    (∃ alg, (layernorm_fwd_triton X W Y
        stride_x_N stride_x_hn stride_x_hd
        stride_y_N stride_y_hn stride_y_hd stride_w_hn stride_w_hd
        N BLOCK_SIZE eps).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := layernorm_fwd_triton X W Y
        stride_x_N stride_x_hn stride_x_hd
        stride_y_N stride_y_hn stride_y_hd stride_w_hn stride_w_hd
        N BLOCK_SIZE eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin N => True)
        (fun i => (Y, yColOffset s stride_y_N stride_y_hn i.val)))
      (expected := fun i =>
        layernormYFullNSpec s X W stride_x_N stride_x_hn stride_w_hn
          N BLOCK_SIZE eps i)
```

**Assumptions / layout contracts:**
- `hBlockPos : 0 < BLOCK_SIZE`
- `hXYNe : X ≠ Y`
- `hWYNe : W ≠ Y`

**Closed-form spec defs (transitive):** `layernorm_fwd_triton`, `yColOffset`, `layernormYFullNSpec`, `xColOffset`, `layernormMeanFullNSpec`, `layernormRstdFullNSpec`, `wColOffset`, `layernormVarFullNSpec`

<details><summary><code>layernorm_fwd_triton</code></summary>

```
/-- Documented transcription of `layernorm_fwd_triton.py`'s
`_layer_norm_fwd_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `N` / `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameters.
- The Python `stride_x_hd`, `stride_y_hd`, and `stride_w_hd` parameters are kept
  as unused Lean parameters because the source kernel body does not use them. -/
```
```lean
def layernorm_fwd_triton
    (X W Y : RegionName)
    (stride_x_N stride_x_hn _stride_x_hd
      stride_y_N stride_y_hn _stride_y_hd
      stride_w_hn _stride_w_hd : Nat)
    (N BLOCK_SIZE : Nat) (eps : ℝ) :
  ComputeKernel := triton {
  Seq = tl.program_id(0)
  H = tl.program_id(1)
  X += Seq * $(stride_x_N) + H * $(stride_x_hn)
  Y += Seq * $(stride_y_N) + H * $(stride_y_hn)
  W += H * $(stride_w_hn)
  _mean = tl.zeros([$(BLOCK_SIZE)], dtype=tl.float32)
  for off in range(0, $(N), $(BLOCK_SIZE)) {
    cols = off + tl.arange(0, $(BLOCK_SIZE))
    a = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
    _mean += a
  }
  mean = tl.sum(_mean, axis=0) / $(N)
  _var = tl.zeros([$(BLOCK_SIZE)], dtype=tl.float32)
  for off in range(0, $(N), $(BLOCK_SIZE)) {
    cols = off + tl.arange(0, $(BLOCK_SIZE))
    x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
    x = tl.where(cols < $(N), x - mean, 0.0)
    _var += x * x
  }
  var = tl.sum(_var, axis=0) / $(N)
  rstd = 1 / tl.sqrt(var + $(eps))
  for off in range(0, $(N), $(BLOCK_SIZE)) {
    cols = off + tl.arange(0, $(BLOCK_SIZE))
    mask = cols < $(N)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    x_hat = (x - mean) * rstd
    y = x_hat * w
    tl.store(Y + cols, (y).to(X.dtype.element_ty), mask=mask)
  }
}
```
</details>

<details><summary><code>yColOffset</code></summary>

```lean
def yColOffset
    (s : BlockState) (stride_y_N stride_y_hn : Nat) (col : Nat) : Nat :=
  s.pids 0 * stride_y_N + s.pids 1 * stride_y_hn + col
```
</details>

<details><summary><code>layernormYFullNSpec</code></summary>

```
/-- Full-N output spec for every Python-observable output column. -/
```
```lean
noncomputable def layernormYFullNSpec
    (s : BlockState) (X W : RegionName)
    (stride_x_N stride_x_hn stride_w_hn N BLOCK_SIZE : Nat) (eps : ℝ)
    (i : Fin N) : ℝ :=
  ((s.readMem X (xColOffset s stride_x_N stride_x_hn i.val) -
      layernormMeanFullNSpec s X stride_x_N stride_x_hn N BLOCK_SIZE) *
    layernormRstdFullNSpec s X stride_x_N stride_x_hn N BLOCK_SIZE eps) *
    s.readMem W (wColOffset s stride_w_hn i.val)
```
</details>

<details><summary><code>xColOffset</code></summary>

```lean
def xColOffset
    (s : BlockState) (stride_x_N stride_x_hn : Nat) (col : Nat) : Nat :=
  s.pids 0 * stride_x_N + s.pids 1 * stride_x_hn + col
```
</details>

<details><summary><code>layernormMeanFullNSpec</code></summary>

```
/-- Full-N mean used by the Python `for off in range(0, N, BLOCK_SIZE)` path. -/
```
```lean
noncomputable def layernormMeanFullNSpec
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N _BLOCK_SIZE : Nat) : ℝ :=
  (∑ j : Fin N, s.readMem X (xColOffset s stride_x_N stride_x_hn j.val)) /
    (N : ℝ)
```
</details>

<details><summary><code>layernormRstdFullNSpec</code></summary>

```lean
noncomputable def layernormRstdFullNSpec
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE : Nat) (eps : ℝ) : ℝ :=
  (Real.sqrt
    (layernormVarFullNSpec s X stride_x_N stride_x_hn N BLOCK_SIZE + eps))⁻¹
```
</details>

<details><summary><code>wColOffset</code></summary>

```lean
def wColOffset (s : BlockState) (stride_w_hn : Nat) (col : Nat) : Nat :=
  s.pids 1 * stride_w_hn + col
```
</details>

<details><summary><code>layernormVarFullNSpec</code></summary>

```
/-- Full-N variance after subtracting the full-N mean. -/
```
```lean
noncomputable def layernormVarFullNSpec
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE : Nat) : ℝ :=
  (∑ j : Fin N,
      (s.readMem X (xColOffset s stride_x_N stride_x_hn j.val) -
        layernormMeanFullNSpec s X stride_x_N stride_x_hn N BLOCK_SIZE)^2) /
    (N : ℝ)
```
</details>

## Public theorem: `layernorm_fwd_triton_io_correctness`

<details><summary>docstring</summary>

```
/-- **The `⊨[R]` streaming headline (wave-5 S3 per-step emit genre).** For
every rounding model `R`, the faithful `layernorm_fwd_triton` surface
implements, on its `StreamEmitMasked2DKernelIO₂` signature, the **ideal ℝ
three-pass LayerNorm** over the streamed tiles: emitted window `(t, j)`
holds `((x[t,j] − mean) · (√(var + eps))⁻¹) · w[t,j]`, where
`mean = (Σ_guarded x)/N` and `var = (Σ_guarded (x − mean)²)/N` are guarded
double sums folding the *entire* `x` stream — the spec `f` is exact real
arithmetic. The kernel has **zero rounding events**: every load, all three
passes' arithmetic and the per-step stores are at `.real`, and the Python
`.to(tl.float32)` input casts and the store cast `(y).to(X.dtype.element_ty)`
are erased by the lowering (`layernormOutLoopBody`'s emitted statement is a
plain `Stmt.store .real`, so the `outDType := .real` default is the honest
grid, not a modelling shortcut). The skin's boundary quantization therefore
degenerates: the readback contract's `R.round .real` is the identity by the
model's defining `round_real`, and the `.real` in-loop stores are exact
under `execR R` — the ∀-`R` face holds via the `RoundingModel` `.real`
identity fields, not as a `.triv` special case.

Layer map: all three passes are cast-free (`lnMeanBody_castFree`,
`lnVarBody_castFree`, `lnOutBody_castFree` and the three register-only
segments), so under `execR R` they collapse verbatim onto the exact stepper
and the proven mean / variance / output invariant stack above
(`layernormMeanLoopContextInvariant`, `layernormVarLoopContextInvariant`,
`layernormOutLoopContextInvariant`, closed by
`layernorm_fwd_triton_staged_fullN_correct_from_preloop`) is reused
unchanged; the `⊨[R]` face adds only the `TraceSafeR` walk
(`layernorm_traceSafeR`), the per-cell memory frame
(`ln_outBody_step_frame`, the `mem` twin of
`layernormOutLoopBody_step_preserves_old_output`), and the stream-lane spec
bridge (`ln_reblock` re-blocking `k ↔ (k/B, k%B)`).

All three hypotheses are truth-forced — they are exactly the exact
headline `layernorm_fwd_triton_output_summary`'s side conditions:

* `hBlockPos : 0 < BLOCK_SIZE` — all three loops step by `BLOCK_SIZE`
  (`range(0, N, BLOCK_SIZE)`); at `BLOCK_SIZE = 0` and `N > 0` no `forRange`
  advances, `execR` cannot terminate the way the invariant stack requires,
  and the step index `off / BLOCK_SIZE` is meaningless. It holds for every
  real launch.
* `hXYNe : X ≠ Y`, `hWYNe : W ≠ Y` — the output pass stores into `Y`
  **between** its re-reads of `X` and `W` (each iteration reloads both from
  the *same* base pointers); if `Y` aliased either input, later blocks would
  re-read already-overwritten values and the closed form would be false.

No further hypothesis is needed. In particular the emit window is
injective in the global lane with **no** side condition: `write` is
`pid₀·stride_y_N + pid₁·stride_y_hn + k`, whose dependence on `k` is a bare
`+ k` (the file's `yColOffset_injective`), so a nonzero column stride is not
required here — unlike the strided-output members of this genre.

Relation to the exact surface: the exact headline
`layernorm_fwd_triton_output_summary` (`Realizes_without_Rounding`) above is
retained unchanged; this `⊨[R]` face restates the same LayerNorm content on
the streaming emit skin, for every `R` at once (at the `.real` grid the two
faces carry the same exact cell). Both faces are kept per the
rounding-as-default doctrine. -/
```
</details>

**Statement:**
```lean
specification layernorm_fwd_triton_io_correctness (R : RoundingModel)
    (X W Y : RegionName)
    (stride_x_N stride_x_hn stride_x_hd
      stride_y_N stride_y_hn stride_y_hd stride_w_hn stride_w_hd
      N BLOCK_SIZE : Nat) (eps : ℝ)
    (hBlockPos : 0 < BLOCK_SIZE) (hXYNe : X ≠ Y) (hWYNe : W ≠ Y) :
    layernormKernelIO X W Y stride_x_N stride_x_hn stride_x_hd
      stride_y_N stride_y_hn stride_y_hd stride_w_hn stride_w_hd
      N BLOCK_SIZE eps ⊨[R]
      fun _ _ xs ws t j => lnStreamSpec N BLOCK_SIZE eps xs ws t j
```

**Assumptions / layout contracts:**
- `hBlockPos : 0 < BLOCK_SIZE`
- `hXYNe : X ≠ Y`
- `hWYNe : W ≠ Y`

**Closed-form spec defs (transitive):** `layernormKernelIO`, `lnStreamSpec`, `layernorm_fwd_triton`, `lnNumSteps`, `lnStreamMean`, `lnStreamVar`, `lnStreamSum`

<details><summary><code>layernormKernelIO</code></summary>

```
/-- **Streaming IO signature** of `layernorm_fwd_triton` on the two-stream
per-step emit skin (S3: in-loop store). Step `t` of any of the three passes
(at `off = t·BLOCK_SIZE`) addresses the `BLOCK_SIZE`-lane column window
`cols = t·BLOCK_SIZE + j`; the mean and variance passes read the `X` window
(`read1`), the third pass re-reads that same `X` window plus the `W` window
(`read2`) and stores the `BLOCK_SIZE`-lane output window (`write`) at the
**`.real`** grid (`outDType` default — `layernormOutLoopBody`'s emitted
statement is a plain `Stmt.store .real`, the Python
`(y).to(X.dtype.element_ty)` cast having been erased by the lowering, so the
per-step stores carry no quantization event). The windows transcribe the
kernel's pointer arithmetic verbatim (`X += Seq*stride_x_N + H*stride_x_hn`,
`W += H*stride_w_hn`, `Y += Seq*stride_y_N + H*stride_y_hn`, all indexed by
`cols`):

* `read1` step `t`, lane `j`:
  `pid₀·stride_x_N + pid₁·stride_x_hn + (t·BLOCK_SIZE + j)` — the file's
  `xColOffset`.
* `read2` step `t`, lane `j`: `pid₁·stride_w_hn + (t·BLOCK_SIZE + j)` — the
  file's `wColOffset`.
* `write` step `t`, lane `j`:
  `pid₀·stride_y_N + pid₁·stride_y_hn + (t·BLOCK_SIZE + j)` — the file's
  `yColOffset`.

All three masks are the kernel's single `cols < N`. -/
```
```lean
def layernormKernelIO (X W Y : RegionName)
    (stride_x_N stride_x_hn stride_x_hd
      stride_y_N stride_y_hn stride_y_hd
      stride_w_hn stride_w_hd N BLOCK_SIZE : Nat) (eps : ℝ) :
    StreamEmitMasked2DKernelIO₂ where
  kernel := layernorm_fwd_triton X W Y stride_x_N stride_x_hn stride_x_hd
    stride_y_N stride_y_hn stride_y_hd stride_w_hn stride_w_hd N BLOCK_SIZE eps
  inp1 := X
  inp2 := W
  out := Y
  T := lnNumSteps N BLOCK_SIZE
  B1 := BLOCK_SIZE
  B2 := BLOCK_SIZE
  C := BLOCK_SIZE
  read1 := fun p₀ p₁ t j =>
    p₀ * stride_x_N + p₁ * stride_x_hn + (t.val * BLOCK_SIZE + j.val)
  read2 := fun _ p₁ t j => p₁ * stride_w_hn + (t.val * BLOCK_SIZE + j.val)
  write := fun p₀ p₁ t j =>
    p₀ * stride_y_N + p₁ * stride_y_hn + (t.val * BLOCK_SIZE + j.val)
  mask1 := fun _ _ t j => t.val * BLOCK_SIZE + j.val < N
  mask2 := fun _ _ t j => t.val * BLOCK_SIZE + j.val < N
  writeMask := fun _ _ t j => t.val * BLOCK_SIZE + j.val < N
```
</details>

<details><summary><code>lnStreamSpec</code></summary>

```
/-- The stream-level LayerNorm spec (the genre's three-pass shape): emitted
window `(t, j)` holds `((x[t,j] - mean) · rstd) · w[t,j]`, where `mean` and
`rstd = 1/√(var + eps)` are folds over the *entire* `x` stream.
Algebraically `layernormYFullNSpec` with the `Fin N` sums re-blocked to the
guarded stream double sums. -/
```
```lean
noncomputable def lnStreamSpec (N B : Nat) (eps : ℝ)
    (xs ws : Fin (lnNumSteps N B) → Fin B → ℝ)
    (t : Fin (lnNumSteps N B)) (j : Fin B) : ℝ :=
  ((xs t j - lnStreamMean N B xs) *
      (Real.sqrt (lnStreamVar N B xs + eps))⁻¹) * ws t j
```
</details>

<details><summary><code>layernorm_fwd_triton</code></summary>

```
/-- Documented transcription of `layernorm_fwd_triton.py`'s
`_layer_norm_fwd_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `N` / `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameters.
- The Python `stride_x_hd`, `stride_y_hd`, and `stride_w_hd` parameters are kept
  as unused Lean parameters because the source kernel body does not use them. -/
```
```lean
def layernorm_fwd_triton
    (X W Y : RegionName)
    (stride_x_N stride_x_hn _stride_x_hd
      stride_y_N stride_y_hn _stride_y_hd
      stride_w_hn _stride_w_hd : Nat)
    (N BLOCK_SIZE : Nat) (eps : ℝ) :
  ComputeKernel := triton {
  Seq = tl.program_id(0)
  H = tl.program_id(1)
  X += Seq * $(stride_x_N) + H * $(stride_x_hn)
  Y += Seq * $(stride_y_N) + H * $(stride_y_hn)
  W += H * $(stride_w_hn)
  _mean = tl.zeros([$(BLOCK_SIZE)], dtype=tl.float32)
  for off in range(0, $(N), $(BLOCK_SIZE)) {
    cols = off + tl.arange(0, $(BLOCK_SIZE))
    a = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
    _mean += a
  }
  mean = tl.sum(_mean, axis=0) / $(N)
  _var = tl.zeros([$(BLOCK_SIZE)], dtype=tl.float32)
  for off in range(0, $(N), $(BLOCK_SIZE)) {
    cols = off + tl.arange(0, $(BLOCK_SIZE))
    x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
    x = tl.where(cols < $(N), x - mean, 0.0)
    _var += x * x
  }
  var = tl.sum(_var, axis=0) / $(N)
  rstd = 1 / tl.sqrt(var + $(eps))
  for off in range(0, $(N), $(BLOCK_SIZE)) {
    cols = off + tl.arange(0, $(BLOCK_SIZE))
    mask = cols < $(N)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    x_hat = (x - mean) * rstd
    y = x_hat * w
    tl.store(Y + cols, (y).to(X.dtype.element_ty), mask=mask)
  }
}
```
</details>

<details><summary><code>lnNumSteps</code></summary>

```
/-- Trip count of all three `for off in range(0, N, BLOCK_SIZE)` passes:
`⌈N / BLOCK_SIZE⌉`. -/
```
```lean
def lnNumSteps (N B : Nat) : Nat := (N + B - 1) / B
```
</details>

<details><summary><code>lnStreamMean</code></summary>

```
/-- The stream-level mean: `tl.sum(_mean, axis=0) / N`. -/
```
```lean
noncomputable def lnStreamMean (N B : Nat)
    (xs : Fin (lnNumSteps N B) → Fin B → ℝ) : ℝ :=
  lnStreamSum N B xs / (N : ℝ)
```
</details>

<details><summary><code>lnStreamVar</code></summary>

```
/-- The stream-level variance: the second pass's `_var += x*x` fold over the
whole stream, centred at `lnStreamMean` and guarded by the same window (the
kernel's `tl.where(cols < N, x - mean, 0.0)`), divided by `N`. -/
```
```lean
noncomputable def lnStreamVar (N B : Nat)
    (xs : Fin (lnNumSteps N B) → Fin B → ℝ) : ℝ :=
  (∑ u : Fin (lnNumSteps N B), ∑ e : Fin B,
      if u.val * B + e.val < N then (xs u e - lnStreamMean N B xs) ^ 2 else 0)
    / (N : ℝ)
```
</details>

<details><summary><code>lnStreamSum</code></summary>

```
/-- The guarded stream-level sum of the `X` stream: the mean pass's
`_mean += a` fold over the whole curried stream, guarded by the kernel's
window (`t·B + e < N`) — the contract only pins `xs` on masked lanes, so
the spec must not read unmasked lanes. -/
```
```lean
noncomputable def lnStreamSum (N B : Nat)
    (xs : Fin (lnNumSteps N B) → Fin B → ℝ) : ℝ :=
  ∑ u : Fin (lnNumSteps N B), ∑ e : Fin B,
    if u.val * B + e.val < N then xs u e else 0
```
</details>

## Also present (pinned special-case summaries)
- `layernorm_fwd_triton_compute_correct`
