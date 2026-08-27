# Spec sheet — `bench/tritonbench_g/diag_ssm_triton/DiagSsmTriton.lean`

**Python source:** `bench/tritonbench_g/diag_ssm_triton/diag_ssm_triton.py`

## Public theorem: `diag_ssm_forward_kernel_output_summary`

<details><summary>docstring</summary>

```
/-- Per-kernel output summary for `diag_ssm_forward_kernel`: the DSL surface
lowers to the algorithm layer, and the time-step stores to `y_ptr` are
compute-correct — after the `0..length` recurrent scan every active output
offset holds the diagonal-SSM spec value `diagSsmForwardSpecAt`, and inactive
lanes are preserved. Mirrors `add_kernel_output_summary`. -/
```
</details>

**Statement:**
```lean
specification diag_ssm_forward_kernel_output_summary
    (s_ptr x_ptr lambda_ptr y_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [length, BLOCK_SIZE] =>
        diagSsmForwardOutOffset s batch_size dim BLOCK_SIZE idx))
    (hXOutNe : x_ptr ≠ y_ptr) :
    (∃ alg, (diag_ssm_forward_kernel s_ptr x_ptr lambda_ptr y_ptr
        length batch_size dim BLOCK_SIZE).toAlgorithm? = Except.ok alg) ∧
    diag_ssm_forward_kernel_correct_target s_ptr x_ptr lambda_ptr y_ptr
      length batch_size dim BLOCK_SIZE s
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun idx : TileIndex [length, BLOCK_SIZE] =>
        diagSsmForwardOutOffset s batch_size dim BLOCK_SIZE idx)`
- `hXOutNe : x_ptr ≠ y_ptr`

**Closed-form spec defs (transitive):** `diagSsmForwardOutOffset`, `diag_ssm_forward_kernel`, `diag_ssm_forward_kernel_correct_target`, `timeOffset`, `diagSsmForwardActive`, `diagSsmForwardSpecAt`, `colOffset`, `active`, `diagSsmForwardSpec`, `diagSsmStateAfter`

<details><summary><code>diagSsmForwardOutOffset</code></summary>

```lean
def diagSsmForwardOutOffset
    {length : Nat}
    (st : BlockState) (batch_size dim BLOCK_SIZE : Nat)
    (idx : TileIndex [length, BLOCK_SIZE]) : Nat :=
  timeOffset st batch_size dim BLOCK_SIZE idx.1.val idx.2.1
```
</details>

<details><summary><code>diag_ssm_forward_kernel</code></summary>

```
/-- Faithful transcription of `diag_ssm_triton.py`'s
`diag_ssm_forward_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameter.
- Python `length`, `batch_size`, `dim` → Lean `Nat` parameters.

The proof below connects the recurrence invariant across `tl.for t in length`
to `ComputeCorrect.Realizes_without_Rounding` under the stated no-collision/no-alias
hypotheses. -/
```
```lean
def diag_ssm_forward_kernel
    (s_ptr x_ptr lambda_ptr y_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  col_idx = tl.program_id(0) * $(BLOCK_SIZE)
  col_offsets = col_idx + tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(batch_size * dim)
  s = tl.load(s_ptr + col_offsets, mask=mask, other=0)
  Lambda = tl.load(lambda_ptr + col_offsets % $(dim), mask=mask, other=0)
  tl.for t in $(length) {
    offsets = t * $(batch_size * dim) + col_offsets
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    s = s * Lambda + x
    tl.store(y_ptr + offsets, s, mask=mask)
  }
}
```
</details>

<details><summary><code>diag_ssm_forward_kernel_correct_target</code></summary>

```lean
def diag_ssm_forward_kernel_correct_target
    (s_ptr x_ptr lambda_ptr y_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE : Nat) (s : BlockState) : Prop :=
  ComputeCorrect.Realizes_without_Rounding
    (kernel := diag_ssm_forward_kernel s_ptr x_ptr lambda_ptr y_ptr
      length batch_size dim BLOCK_SIZE)
    (initialState := s)
    (write := ComputeCorrect.WriteMap.writeIf
      (fun idx : TileIndex [length, BLOCK_SIZE] =>
        diagSsmForwardActive s batch_size dim BLOCK_SIZE idx)
      (fun idx : TileIndex [length, BLOCK_SIZE] =>
        (y_ptr, diagSsmForwardOutOffset s batch_size dim BLOCK_SIZE idx)))
    (expected := fun idx : TileIndex [length, BLOCK_SIZE] =>
      diagSsmForwardSpecAt s s_ptr x_ptr lambda_ptr batch_size dim BLOCK_SIZE idx)
```
</details>

<details><summary><code>timeOffset</code></summary>

```lean
def timeOffset
    (st : BlockState) (batch_size dim BLOCK_SIZE t : Nat)
    (i : Fin BLOCK_SIZE) : Nat :=
  t * (batch_size * dim) + colOffset st BLOCK_SIZE i
```
</details>

<details><summary><code>diagSsmForwardActive</code></summary>

```lean
def diagSsmForwardActive
    {length : Nat}
    (st : BlockState) (batch_size dim BLOCK_SIZE : Nat)
    (idx : TileIndex [length, BLOCK_SIZE]) : Prop :=
  active st batch_size dim BLOCK_SIZE idx.2.1
```
</details>

<details><summary><code>diagSsmForwardSpecAt</code></summary>

```lean
noncomputable def diagSsmForwardSpecAt
    {length : Nat}
    (st : BlockState) (s_ptr x_ptr lambda_ptr : RegionName)
    (batch_size dim BLOCK_SIZE : Nat)
    (idx : TileIndex [length, BLOCK_SIZE]) : ℝ :=
  diagSsmForwardSpec st s_ptr x_ptr lambda_ptr batch_size dim BLOCK_SIZE
    idx.1.val idx.2.1
```
</details>

<details><summary><code>colOffset</code></summary>

```lean
def colOffset (st : BlockState) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  st.pids 0 * BLOCK_SIZE + i.val
```
</details>

<details><summary><code>active</code></summary>

```lean
def active (st : BlockState) (batch_size dim BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : Prop :=
  colOffset st BLOCK_SIZE i < batch_size * dim
```
</details>

<details><summary><code>diagSsmForwardSpec</code></summary>

```lean
noncomputable def diagSsmForwardSpec
    (st : BlockState) (s_ptr x_ptr lambda_ptr : RegionName)
    (batch_size dim BLOCK_SIZE : Nat) (t : Nat) (i : Fin BLOCK_SIZE) : ℝ :=
  diagSsmStateAfter st s_ptr x_ptr lambda_ptr batch_size dim BLOCK_SIZE i (t + 1)
```
</details>

<details><summary><code>diagSsmStateAfter</code></summary>

```lean
noncomputable def diagSsmStateAfter
    (st : BlockState) (s_ptr x_ptr lambda_ptr : RegionName)
    (batch_size dim BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : Nat → ℝ
  | 0 => st.readMem s_ptr (colOffset st BLOCK_SIZE i)
  | t + 1 =>
      diagSsmStateAfter st s_ptr x_ptr lambda_ptr batch_size dim BLOCK_SIZE i t *
          st.readMem lambda_ptr (IntegralDType.nat.mod (colOffset st BLOCK_SIZE i) dim) +
        st.readMem x_ptr (timeOffset st batch_size dim BLOCK_SIZE t i)
```
</details>

## Public theorem: `diag_ssm_forward_io_correctness`

<details><summary>docstring</summary>

```
/-- **The `⊨[R]` streaming headline (wave-5 S3 per-step emit genre).** For
every rounding model `R`, the faithful `diag_ssm_forward_kernel` surface
implements, on its `StreamEmitMasked2DKernelIO₃` signature, the **ideal ℝ
diagonal-SSM scan** over the streamed tiles: emitted window `(t, j)` holds
the state after `t + 1` steps of `s ← s·Λ + x_u` (`u ≤ t`), seeded with
the static initial-state tile and folded with the static decay tile —
`diagSsmStreamState` is exact real arithmetic. The `s`/`Λ` channels are
the skin's degenerate static streams (windows ignore `t`); the grid is
1-D, so every window also ignores `pid₁` and the headline is ∀-`pid₁`
(the `bgmv_shrink` precedent, one grid rank down). The kernel has **zero
rounding events** (all loads, the scan arithmetic and the per-step raw
stores are at `.real`; there is no `castFloat`), so the skin's boundary
quantization degenerates: the readback's `R.round .real` is the identity
by the model's defining `round_real` — the ∀-`R` face holds via the
`RoundingModel` `.real` identity fields, not as a `.triv` special case.

Layer map: the surface is cast-free, so under `execR R` it collapses
verbatim onto the exact stepper and the proven
`diagSsmForwardLoopContextInvariant` stack above is reused unchanged; the
`⊨[R]` face adds the first bench `TraceSafeR` walk over a `forLoop` (a
private mirror of the library's `forRangeTraceSafeR_inv`), the per-cell
memory frame (`dssm_body_step_frame`), and the stream-lane spec bridge
(`dssm_streamState_eq_stateAfter`).

All three hypotheses are truth-forced, with provenance:

* `hT : 0 < length` — the static `s`/`Λ` windows are step-indexed
  (`Fin length`), so the skin's `read1`/`read2` bound groups are
  non-vacuous only when a step exists; at `length = 0` the kernel still
  issues the two pre-loop loads but the contract would supply no bounds
  for them. A 0-step scan has an empty output anyway.
* `hBS : BLOCK_SIZE ≤ batch_size·dim` — the tiling-geometry form of the
  exact headline `diag_ssm_forward_kernel_output_summary`'s `hOutInj`
  (full-grid output-window injectivity, which the loop invariant carries
  every previously-emitted window through later scatters with): with the
  lane offset below the timestep pitch, distinct `(t, j)` windows never
  collide. It holds for every real launch (the grid is
  `⌈batch_size·dim / BLOCK_SIZE⌉` programs over a `batch_size·dim`-wide
  lane space).
* `hXOutNe : x_ptr ≠ y_ptr` — inherited verbatim from the exact headline:
  step `t` stores into `y_ptr` **before** step `t+1` re-reads `x_ptr`; if
  the two aliased, later steps would stream already-overwritten inputs
  and the closed form would be false.

No `s_ptr`/`lambda_ptr` disalias is needed: both tiles are loaded into
registers before the loop, and the spec is pinned on the initial state.

Relation to the exact surface: the exact headline
`diag_ssm_forward_kernel_output_summary` (`Realizes_without_Rounding`)
above is retained unchanged; this `⊨[R]` face restates the same scan
content on the streaming emit skin, for every `R` at once (at the `.real`
grid the two faces carry the same exact cell). Both faces are kept per
the rounding-as-default doctrine. -/
```
</details>

**Statement:**
```lean
specification diag_ssm_forward_io_correctness (R : RoundingModel)
    (s_ptr x_ptr lambda_ptr y_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE : Nat)
    (hT : 0 < length) (hBS : BLOCK_SIZE ≤ batch_size * dim)
    (hXOutNe : x_ptr ≠ y_ptr) :
    diagSsmForwardKernelIO s_ptr x_ptr lambda_ptr y_ptr length batch_size dim
        BLOCK_SIZE ⊨[R]
      fun _ _ ss ls xs t j =>
        diagSsmStreamState (ss t j) (ls t j) (fun u => xs u j) (t.val + 1)
```

**Assumptions / layout contracts:**
- `hT : 0 < length`
- `hBS : BLOCK_SIZE ≤ batch_size * dim`
- `hXOutNe : x_ptr ≠ y_ptr`

**Closed-form spec defs (transitive):** `diagSsmForwardKernelIO`, `diagSsmStreamState`, `diag_ssm_forward_kernel`

<details><summary><code>diagSsmForwardKernelIO</code></summary>

```
/-- **Streaming IO signature** of `diag_ssm_forward_kernel` on the
three-stream per-step emit skin (S3: in-loop store). Step `t`, lane `j`
of program `pid₀` (the grid is 1-D; `pid₁` is unused and every window
ignores it) transcribes the kernel's pointer arithmetic verbatim
(`col_offsets = pid₀·BLOCK_SIZE + j`):

* `read1` (the `s_ptr` initial-state tile, **static**: ignores `t`):
  `pid₀·BLOCK_SIZE + j` — the kernel's pre-loop `s_ptr + col_offsets`.
* `read2` (the `lambda_ptr` decay tile, **static**: ignores `t`):
  `(pid₀·BLOCK_SIZE + j) % dim` — the kernel's pre-loop
  `lambda_ptr + col_offsets % dim` (the `%` is just the window function;
  reads need no injectivity).
* `read3` (the `x_ptr` input, **streamed**):
  `t·(batch_size·dim) + pid₀·BLOCK_SIZE + j` — the kernel's in-loop
  `offsets = t * batch_size * dim + col_offsets`.
* `write` (the `y_ptr` output): the same in-loop `offsets` window.

All four masks are the kernel's single `mask`:
`pid₀·BLOCK_SIZE + j < batch_size·dim` (`t`-independent). The store is the
raw `tl.store(y_ptr + offsets, s, mask)` at `.real`, so `outDType` keeps
the skin's `.real` default — no quantization event. -/
```
```lean
def diagSsmForwardKernelIO (s_ptr x_ptr lambda_ptr y_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE : Nat) :
    StreamEmitMasked2DKernelIO₃ where
  kernel := diag_ssm_forward_kernel s_ptr x_ptr lambda_ptr y_ptr length
    batch_size dim BLOCK_SIZE
  inp1 := s_ptr
  inp2 := lambda_ptr
  inp3 := x_ptr
  out := y_ptr
  T := length
  B1 := BLOCK_SIZE
  B2 := BLOCK_SIZE
  B3 := BLOCK_SIZE
  C := BLOCK_SIZE
  read1 := fun p₀ _ _ j => p₀ * BLOCK_SIZE + j.val
  read2 := fun p₀ _ _ j => (p₀ * BLOCK_SIZE + j.val) % dim
  read3 := fun p₀ _ t j => t.val * (batch_size * dim) + (p₀ * BLOCK_SIZE + j.val)
  write := fun p₀ _ t j => t.val * (batch_size * dim) + (p₀ * BLOCK_SIZE + j.val)
  mask1 := fun p₀ _ _ j => p₀ * BLOCK_SIZE + j.val < batch_size * dim
  mask2 := fun p₀ _ _ j => p₀ * BLOCK_SIZE + j.val < batch_size * dim
  mask3 := fun p₀ _ _ j => p₀ * BLOCK_SIZE + j.val < batch_size * dim
  writeMask := fun p₀ _ _ j => p₀ * BLOCK_SIZE + j.val < batch_size * dim
```
</details>

<details><summary><code>diagSsmStreamState</code></summary>

```
/-- The stream-side diagonal-SSM scan: state after `u` steps of
`s ← s·lam + xs u`, seeded with `init`. The step input is guarded by the
stream length (`u < T`); the guard is always live on the bridge below
(only prefixes `u ≤ T` are ever evaluated). Mirrors the memory-side
recurrence `diagSsmStateAfter` with the three memory channels replaced by
the skin's per-lane stream values. -/
```
```lean
noncomputable def diagSsmStreamState {T : Nat} (init lam : ℝ)
    (xs : Fin T → ℝ) : Nat → ℝ
  | 0 => init
  | u + 1 =>
      diagSsmStreamState init lam xs u * lam +
        (if h : u < T then xs ⟨u, h⟩ else 0)
```
</details>

<details><summary><code>diag_ssm_forward_kernel</code></summary>

```
/-- Faithful transcription of `diag_ssm_triton.py`'s
`diag_ssm_forward_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameter.
- Python `length`, `batch_size`, `dim` → Lean `Nat` parameters.

The proof below connects the recurrence invariant across `tl.for t in length`
to `ComputeCorrect.Realizes_without_Rounding` under the stated no-collision/no-alias
hypotheses. -/
```
```lean
def diag_ssm_forward_kernel
    (s_ptr x_ptr lambda_ptr y_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  col_idx = tl.program_id(0) * $(BLOCK_SIZE)
  col_offsets = col_idx + tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(batch_size * dim)
  s = tl.load(s_ptr + col_offsets, mask=mask, other=0)
  Lambda = tl.load(lambda_ptr + col_offsets % $(dim), mask=mask, other=0)
  tl.for t in $(length) {
    offsets = t * $(batch_size * dim) + col_offsets
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    s = s * Lambda + x
    tl.store(y_ptr + offsets, s, mask=mask)
  }
}
```
</details>

## Public theorem: `diag_ssm_backward_io_correctness`

<details><summary>docstring</summary>

```
/-- **The backward `⊨[R]` streaming headline (wave-5 S3 grouped emit
genre).** For every rounding model `R`, the faithful
`diag_ssm_backward_kernel` surface implements, on its grouped
`StreamGroupedEmitMasked3DKernelIO` signature, the **ideal ℝ reverse-time
gradient scan** over the streamed tiles, one closed form per output
channel (`diagSsmBackwardStreamSpec`, the shared constructor-head
matcher):

* `grad_x` at step `t` — the running `grad_s` after absorbing
  `grad_y[t]`: the upstream gradient at `t` plus the reverse **suffix**
  fold of `(gs ← (grad_y[T-1-u] + gs)·Λ)` over the `T-1-t` iterations
  strictly before `t` is processed;
* `grad_s` (terminal, designated step `t = 0`) — the full `T`-iteration
  reverse fold;
* `grad_Λ` (terminal) — the full accumulator
  `Σ (grad_y[u] + gs)·prev u`, where `prev` resolves the kernel's
  `if t > 0` branch on the streams (`y[t-1]` vs the initial state `s`).

The grid is 1-D, so every window ignores `pid₁`/`pid₂` and the headline
is ∀-both. The kernel has **zero rounding events** (all loads, the scan
arithmetic and all three raw stores are at `.real`; there is no
`castFloat`), so the skin's boundary quantization degenerates: the
readback's `R.round .real` is the identity by the model's defining
`round_real` field — the ∀-`R` face holds via the `RoundingModel` `.real`
identity fields, not as a `.triv` special case.

Layer map: the surface is cast-free, so under `execR R` it collapses
verbatim onto the exact stepper and the proven
`diagSsmBackwardLoopContextInvariant` / `forRange_inv` /
terminal-scatter stack above is reused unchanged; the `⊨[R]` face adds
the `TraceSafeR` walk (via the library's `forRangeTraceSafeR_inv`; the
runtime `if t > 0` branch is walked per iteration, both arms taken
depending on the counter), the per-cell memory frames
(`dssmb_body_step_frame` / `dssmb_postLoop_frame`), and the stream-lane
spec bridges (`dssmb_streamGradS_eq` / `dssmb_streamGradLambda_eq`).

All hypotheses are truth-forced, with provenance:

* `hT : 0 < length` — truth-forced twice. (1) The step-independent
  windows (`s`/`Λ` reads, the two terminal stores) are step-indexed
  (`Fin length`), so their bound groups are non-vacuous only when a step
  exists, yet the kernel issues the pre-loop `Λ` load and both terminal
  stores even for a 0-step scan (the wave-5 round-5 lesson). (2) At
  `length = 0` the two terminal stores still execute while the
  `Fin 0`-gated `writeMask` declares an **empty** write set, so the
  skin's frame condition would be outright false.
* `hBS : BLOCK_SIZE ≤ batch_size·dim` — the tiling-geometry form of the
  exact stack's `hOutInj` (`grad_x` full-grid window injectivity, which
  the loop invariant needs to carry previously-emitted windows through
  later scatters). Holds for every real launch (the grid is
  `⌈batch_size·dim / BLOCK_SIZE⌉` programs over a `batch_size·dim`-wide
  lane space).
* `hGradXSNe`/`hGradXYNe`/`hGradXGradYNe`/`hGradXLambdaNe` — inherited
  verbatim from the exact backward stack: the in-loop `grad_x` scatter
  must not clobber the four regions the scan keeps reading (`s`, `y`,
  `grad_y`, `λ`); e.g. if `grad_x_ptr = y_ptr`, iteration `k`'s store
  would corrupt the `y[t-1]` cells iteration `k+1` reads and the closed
  form would be false (`y_ptr` readback stability).
* `hGradSGradXNe`/`hGradLambdaGradXNe` — the terminal stores must not
  clobber the already-emitted `grad_x` windows
  (`diagSsmBackwardPostLoop_preserve_gradX`).
* `hGradLambdaGradSNe` — the second terminal scatter must not clobber
  the first one's cells (`diagSsmBackwardPostLoop_readback`).

Relation to the exact surface: the exact headline
`diag_ssm_backward_kernel_compute_correct` above is retained unchanged;
this `⊨[R]` face restates the same gradient content on the grouped
streaming emit skin, for every `R` at once (at the `.real` grid the two
faces carry the same exact cells). Both faces are kept per the
rounding-as-default doctrine. -/
```
</details>

**Statement:**
```lean
specification diag_ssm_backward_io_correctness (R : RoundingModel)
    (s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr
      grad_y_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE : Nat)
    (hT : 0 < length) (hBS : BLOCK_SIZE ≤ batch_size * dim)
    (hGradXSNe : grad_x_ptr ≠ s_ptr) (hGradXYNe : grad_x_ptr ≠ y_ptr)
    (hGradXGradYNe : grad_x_ptr ≠ grad_y_ptr)
    (hGradXLambdaNe : grad_x_ptr ≠ lambda_ptr)
    (hGradSGradXNe : grad_s_ptr ≠ grad_x_ptr)
    (hGradLambdaGradXNe : grad_lambda_ptr ≠ grad_x_ptr)
    (hGradLambdaGradSNe : grad_lambda_ptr ≠ grad_s_ptr) :
    diagSsmBackwardKernelIO s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr
        grad_lambda_ptr grad_y_ptr length batch_size dim BLOCK_SIZE ⊨[R]
      fun _ _ _ xs o t j => diagSsmBackwardStreamSpec xs o t j
```

**Assumptions / layout contracts:**
- `hT : 0 < length`
- `hBS : BLOCK_SIZE ≤ batch_size * dim`
- `hGradXSNe : grad_x_ptr ≠ s_ptr`
- `hGradXYNe : grad_x_ptr ≠ y_ptr`
- `hGradXGradYNe : grad_x_ptr ≠ grad_y_ptr`
- `hGradXLambdaNe : grad_x_ptr ≠ lambda_ptr`
- `hGradSGradXNe : grad_s_ptr ≠ grad_x_ptr`
- `hGradLambdaGradXNe : grad_lambda_ptr ≠ grad_x_ptr`
- `hGradLambdaGradSNe : grad_lambda_ptr ≠ grad_s_ptr`

**Closed-form spec defs (transitive):** `diagSsmBackwardKernelIO`, `diagSsmBackwardStreamSpec`, `diag_ssm_backward_kernel`, `dssmbStreamGradS`, `dssmbStreamGradLambda`, `reverseTime`, `dssmbStreamPrev`

<details><summary><code>diagSsmBackwardKernelIO</code></summary>

```
/-- **Streaming IO signature** of `diag_ssm_backward_kernel` on the grouped
per-step emit skin (S3; the grid is 1-D, so every window ignores
`pid₁`/`pid₂` and the headline is ∀-both). Step `t`, lane `j` of program
`pid₀` transcribes the kernel's pointer arithmetic verbatim
(`col_offsets = pid₀·BLOCK_SIZE + j`,
`offsets = t·(batch_size·dim) + col_offsets`):

Input channels (`Fin 4`):

* `0` — `grad_y_ptr` (streamed): window `offsets`, the per-iteration
  upstream-gradient load.
* `1` — `y_ptr` (the `t > 0` arm of the previous-state branch): window
  `(t-1)·(batch_size·dim) + col_offsets` (the kernel's
  `offsets - batch_size·dim`), read-active only when `0 < t`.
* `2` — `s_ptr` (the `t = 0` arm): window `col_offsets`, read-active only
  at `t = 0`.
* `3` — `lambda_ptr` (**static**: window ignores `t`): the pre-loop
  `lambda_ptr + col_offsets % dim` register-cached decay tile.

Output channels (`Fin 3`):

* `0` — `grad_x_ptr` (per-step): window `offsets`, the in-loop store.
* `1` — `grad_s_ptr` (**terminal**): window `col_offsets` (ignores `t`),
  write-active only at the designated step `t = 0`.
* `2` — `grad_lambda_ptr` (**terminal**): same designated-step
  convention, window `col_offsets`.

Every step-independent mask conjunct is the kernel's single `mask`:
`pid₀·BLOCK_SIZE + j < batch_size·dim`. All three stores are raw `.real`
stores, so `outDType` keeps the skin's `.real` default — no quantization
event. `bufs` lists every region the kernel touches exactly once, in
kernel-argument order. -/
```
```lean
def diagSsmBackwardKernelIO
    (s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr
      grad_y_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE : Nat) :
    StreamGroupedEmitMasked3DKernelIO where
  kernel := diag_ssm_backward_kernel s_ptr lambda_ptr y_ptr grad_s_ptr
    grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim BLOCK_SIZE
  nIn := 4
  nOut := 3
  bufs := [s_ptr, lambda_ptr, y_ptr, grad_s_ptr, grad_x_ptr,
    grad_lambda_ptr, grad_y_ptr]
  inp := fun i => match i with
    | ⟨0, _⟩ => grad_y_ptr
    | ⟨1, _⟩ => y_ptr
    | ⟨2, _⟩ => s_ptr
    | ⟨_ + 3, _⟩ => lambda_ptr
  out := fun o => match o with
    | ⟨0, _⟩ => grad_x_ptr
    | ⟨1, _⟩ => grad_s_ptr
    | ⟨_ + 2, _⟩ => grad_lambda_ptr
  T := length
  B := BLOCK_SIZE
  read := fun i p₀ _ _ t j => match i with
    | ⟨0, _⟩ => t.val * (batch_size * dim) + (p₀ * BLOCK_SIZE + j.val)
    | ⟨1, _⟩ => (t.val - 1) * (batch_size * dim) + (p₀ * BLOCK_SIZE + j.val)
    | ⟨2, _⟩ => p₀ * BLOCK_SIZE + j.val
    | ⟨_ + 3, _⟩ => (p₀ * BLOCK_SIZE + j.val) % dim
  readMask := fun i p₀ _ _ t j => match i with
    | ⟨0, _⟩ => p₀ * BLOCK_SIZE + j.val < batch_size * dim
    | ⟨1, _⟩ => 0 < t.val ∧ p₀ * BLOCK_SIZE + j.val < batch_size * dim
    | ⟨2, _⟩ => t.val = 0 ∧ p₀ * BLOCK_SIZE + j.val < batch_size * dim
    | ⟨_ + 3, _⟩ => p₀ * BLOCK_SIZE + j.val < batch_size * dim
  write := fun o p₀ _ _ t j => match o with
    | ⟨0, _⟩ => t.val * (batch_size * dim) + (p₀ * BLOCK_SIZE + j.val)
    | ⟨1, _⟩ => p₀ * BLOCK_SIZE + j.val
    | ⟨_ + 2, _⟩ => p₀ * BLOCK_SIZE + j.val
  writeMask := fun o p₀ _ _ t j => match o with
    | ⟨0, _⟩ => p₀ * BLOCK_SIZE + j.val < batch_size * dim
    | ⟨1, _⟩ => t.val = 0 ∧ p₀ * BLOCK_SIZE + j.val < batch_size * dim
    | ⟨_ + 2, _⟩ => t.val = 0 ∧ p₀ * BLOCK_SIZE + j.val < batch_size * dim
```
</details>

<details><summary><code>diagSsmBackwardStreamSpec</code></summary>

```
/-- The shared three-channel spec matcher of the backward headline (the
grouped skin's `f`, one constructor-head arm per output channel — the
single match point, no inline per-site matchers):

* channel `0` (`grad_x`, step `t`): the running `grad_s` after absorbing
  `grad_y[t]` — the upstream gradient at `t` plus the reverse **suffix**
  fold over times `> t` (`T-1-t` reverse iterations);
* channel `1` (`grad_s`, terminal): the full `T`-iteration reverse fold;
* channel `2` (`grad_Λ`, terminal): the full `T`-iteration accumulator
  over the `grad_y`/`y`/`s` streams.

The static decay channel is read at the ambient step (`xs 3 t j`; its
window ignores `t`, so every pinned step holds the same cell). -/
```
```lean
private noncomputable def diagSsmBackwardStreamSpec {T B : Nat}
    (xs : Fin 4 → Fin T → Fin B → ℝ) (o : Fin 3) (t : Fin T) (j : Fin B) :
    ℝ :=
  match o with
  | ⟨0, _⟩ =>
      xs 0 t j +
        dssmbStreamGradS (fun u => xs 0 u j) (xs 3 t j) (T - 1 - t.val)
  | ⟨1, _⟩ => dssmbStreamGradS (fun u => xs 0 u j) (xs 3 t j) T
  | ⟨_ + 2, _⟩ =>
      dssmbStreamGradLambda (fun u => xs 0 u j) (fun u => xs 1 u j)
        (fun u => xs 2 u j) (xs 3 t j) T
```
</details>

<details><summary><code>diag_ssm_backward_kernel</code></summary>

```
/-- Faithful transcription of `diag_ssm_triton.py`'s
`diag_ssm_backward_kernel` for the real-valued path.

The source rewrites reverse traversal as `for i in range(length); t = length -
1 - i`, which is preserved here. -/
```
```lean
def diag_ssm_backward_kernel
    (s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr :
      RegionName)
    (length batch_size dim BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  col_idx = tl.program_id(0) * $(BLOCK_SIZE)
  col_offsets = col_idx + tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(batch_size * dim)
  Lambda = tl.load(lambda_ptr + col_offsets % $(dim), mask=mask, other=0)
  grad_s = tl.zeros_like(Lambda)
  grad_Lambda = tl.zeros_like(Lambda)
  for i in range(0, $(length), $(1)) {
    t = $(length) - $(1) - i
    offsets = t * $(batch_size * dim) + col_offsets
    grad_y = tl.load(grad_y_ptr + offsets, mask=mask, other=0)
    if t > 0 {
      s = tl.load(y_ptr + (offsets - $(batch_size * dim)), mask=mask, other=0)
    } else {
      s = tl.load(s_ptr + col_offsets, mask=mask, other=0)
    }
    grad_s = grad_y + grad_s
    grad_x = grad_s
    grad_Lambda += grad_s * s
    grad_s = grad_s * Lambda
    tl.store(grad_x_ptr + offsets, grad_x, mask=mask)
  }
  tl.store(grad_s_ptr + col_offsets, grad_s, mask=mask)
  tl.store(grad_lambda_ptr + col_offsets, grad_Lambda, mask=mask)
}
```
</details>

<details><summary><code>dssmbStreamGradS</code></summary>

```
/-- Stream-side reverse-scan gradient carry: `grad_s` after `k` reverse
iterations (iteration `u` processes time `reverseTime T u = T-1-u`),
folded over the `grad_y` stream with the static decay value `lam`
(`gs ← (grad_y[T-1-u] + gs)·lam`, seeded with `0`). The step input is
guarded by the stream length; the guard is always live on the bridge
below (only prefixes `k ≤ T` are ever evaluated). Mirrors the memory-side
recurrence `diagSsmBackwardGradSAfter` with the memory channels replaced
by the skin's per-lane stream values. -/
```
```lean
private noncomputable def dssmbStreamGradS {T : Nat}
    (gys : Fin T → ℝ) (lam : ℝ) : Nat → ℝ
  | 0 => 0
  | k + 1 =>
      ((if h : reverseTime T k < T then gys ⟨reverseTime T k, h⟩ else 0) +
        dssmbStreamGradS gys lam k) * lam
```
</details>

<details><summary><code>dssmbStreamGradLambda</code></summary>

```
/-- Stream-side `grad_Λ` accumulator after `k` reverse iterations
(`gΛ ← gΛ + (grad_y[T-1-u] + gs u)·prev u`) — the stream mirror of
`diagSsmBackwardGradLambdaAfter`. -/
```
```lean
private noncomputable def dssmbStreamGradLambda {T : Nat}
    (gys ys ss : Fin T → ℝ) (lam : ℝ) : Nat → ℝ
  | 0 => 0
  | k + 1 =>
      dssmbStreamGradLambda gys ys ss lam k +
        ((if h : reverseTime T k < T then gys ⟨reverseTime T k, h⟩ else 0) +
          dssmbStreamGradS gys lam k) * dssmbStreamPrev ys ss k
```
</details>

<details><summary><code>reverseTime</code></summary>

```lean
def reverseTime (length k : Nat) : Nat :=
  length - 1 - k
```
</details>

<details><summary><code>dssmbStreamPrev</code></summary>

```
/-- Stream-side previous forward state consumed by reverse iteration `k`
(time `t = reverseTime T k`): the `y` stream at step `t` when `0 < t`
(the kernel's `y[t-1]` window is pinned at step `t`), else the `s` stream
at the designated step `t = 0` — the stream mirror of
`diagSsmBackwardPrevState`, resolving the kernel's `if t > 0` branch. -/
```
```lean
private noncomputable def dssmbStreamPrev {T : Nat}
    (ys ss : Fin T → ℝ) (k : Nat) : ℝ :=
  if h : reverseTime T k < T then
    (if 0 < reverseTime T k then ys ⟨reverseTime T k, h⟩
     else ss ⟨reverseTime T k, h⟩)
  else 0
```
</details>

## Also present (pinned special-case summaries)
- `diag_ssm_backward_kernel_compute_correct`
- `diag_ssm_forward_kernel_compute_correct`
