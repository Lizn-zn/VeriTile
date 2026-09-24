# Spec sheet — `bench/tritonbench_g/log_softmax/LogSoftmax.lean`

**Python source:** `bench/tritonbench_g/log_softmax/log_softmax.py`

## Public theorem: `log_softmax_backward_kernel_correctness`

<details><summary>docstring</summary>

```
/-- **The backward headline**: `log_softmax_backward_kernel` implements the
log-softmax VJP `in_grad = out_grad − exp(out) · (Σ_row out_grad)` on its masked
IO signature — for every disjoint flat placement of the three buffers, every
program-id pair whose active lanes are in bounds, and every launch state whose
input windows hold `xs`/`ys` at the active lanes, the translated pointer kernel
terminates, every active output lane `j` holds `logSoftmaxBackwardOf` applied to
the `out`/`out_grad` tiles rebuilt from `xs`/`ys`, and every other memory cell,
including the masked-off lanes of the output window, is unchanged.

The kernel's two loads are masked **without** an `other=` default, so at the
masked-off lanes the register tiles carry `s₀.undef` — which
`Masked2DKernelIO₂.Implements` pins to `0`. The row reduction
`scale = tl.sum(out_grad, 1)` genuinely consumes those pinned zeros, so the
honest per-lane spec sums the tile with masked-off lanes set to `0`
(`logSoftmaxBackwardPure{Out,Grad}Tile`). Because the pin is a hypothesis of the
`⊨` obligation, this headline is discharged via
`Masked2DKernelIO₂.Implements.intro_undef` (the `undef`-threading sibling of
`intro`), not the plain `intro`.

One honest, **truth-required** side hypothesis: `hOutInj` (distinct lanes of a
program's tile must not alias the same output cell — otherwise only the last
store survives). Unlike the forward headline, **no** `0 < BLOCK_N` is needed:
the backward kernel's only reduction is `tl.sum`, which is total on an empty
axis (`= 0`), so termination never fails. -/
```
</details>

**Statement:**
```lean
specification log_softmax_backward_kernel_correctness
    (out_ptr out_grad_ptr in_grad_ptr : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat)
    (hOutInj : ∀ p₀ p₁ : Nat, Function.Injective
      (fun j : Fin (BLOCK_M * BLOCK_N) =>
        outOffset p₀ p₁ N K BLOCK_M (laneIdx BLOCK_M BLOCK_N j))) :
    Masked2DKernelIO₂.Implements
      (logSoftmaxBackwardIO out_ptr out_grad_ptr in_grad_ptr M N K BLOCK_M BLOCK_N)
      (fun p₀ _ xs ys j =>
          logSoftmaxBackwardOf BLOCK_M BLOCK_N
            (logSoftmaxBackwardPureOutTile p₀ M N BLOCK_M BLOCK_N xs)
            (logSoftmaxBackwardPureGradTile p₀ M N BLOCK_M BLOCK_N ys)
            (laneIdx BLOCK_M BLOCK_N j))
```

**Closed-form spec defs (transitive):** `outOffset`, `laneIdx`, `logSoftmaxBackwardIO`, `logSoftmaxBackwardOf`, `logSoftmaxBackwardPureOutTile`, `logSoftmaxBackwardPureGradTile`, `log_softmax_backward_kernel`, `active`, `laneOf`

<details><summary><code>outOffset</code></summary>

```
/-- The shared load/store address of tile cell `idx` for program `(p₀, p₁)`. -/
```
```lean
def outOffset (p₀ p₁ N K BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : Nat :=
  (p₀ * BLOCK_M + idx.1.val) * N * K + idx.2.1.val * K + p₁
```
</details>

<details><summary><code>laneIdx</code></summary>

```
/-- Lane `j` of the row-major flattening of an `R × C` tile. -/
```
```lean
def laneIdx (R C : Nat) (j : Fin (R * C)) : TileIndex [R, C] :=
  (⟨j.val / C, Nat.div_lt_of_lt_mul
      (Nat.lt_of_lt_of_le j.isLt (Nat.le_of_eq (Nat.mul_comm R C)))⟩,
   ⟨j.val % C, Nat.mod_lt _ (Nat.pos_of_ne_zero (by
      rintro rfl
      exact absurd j.isLt (by simp)))⟩,
   PUnit.unit)
```
</details>

<details><summary><code>logSoftmaxBackwardIO</code></summary>

```
/-- `log_softmax_backward_kernel`'s masked **IO signature** — the two-input
sibling of `logSoftmaxIO`:

* `in1`/`in2`/`out` — `out_ptr`, `out_grad_ptr`, `in_grad_ptr`;
* `B = BLOCK_M * BLOCK_N` — the row-major flattening (`laneIdx`);
* `read1 = read2 = write` — the shared window
  `(pid₀·BLOCK_M + m)·N·K + n·K + pid₁` (the two loads and the store use the
  same offsets in the three buffers);
* `mask` — the kernel's `m_offset < M ∧ n_offset < N`; `read2Mask`/`writeMask`
  keep the struct default (both loads and the store carry the *same* mask). -/
```
```lean
def logSoftmaxBackwardIO (out_ptr out_grad_ptr in_grad_ptr : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat) : Masked2DKernelIO₂ where
  kernel := log_softmax_backward_kernel out_ptr out_grad_ptr in_grad_ptr M N K BLOCK_M BLOCK_N
  in1 := out_ptr
  in2 := out_grad_ptr
  out := in_grad_ptr
  B := BLOCK_M * BLOCK_N
  read1 := fun p₀ p₁ j => outOffset p₀ p₁ N K BLOCK_M (laneIdx BLOCK_M BLOCK_N j)
  read2 := fun p₀ p₁ j => outOffset p₀ p₁ N K BLOCK_M (laneIdx BLOCK_M BLOCK_N j)
  write := fun p₀ p₁ j => outOffset p₀ p₁ N K BLOCK_M (laneIdx BLOCK_M BLOCK_N j)
  mask := fun p₀ _ j => active p₀ M N BLOCK_M (laneIdx BLOCK_M BLOCK_N j)
```
</details>

<details><summary><code>logSoftmaxBackwardOf</code></summary>

```
/-- The backward kernel's pointwise VJP core as a function of the two *loaded
tiles* (`out`, `outGrad`): `outGrad − exp(out) · (Σ_row outGrad)` along axis 1.
This is the pure content of the `⊨` headline; the tile-builders below feed it
either the memory contents (readback lemma) or the `⊨` interface's `xs`/`ys`
(headline). -/
```
```lean
noncomputable def logSoftmaxBackwardOf (BLOCK_M BLOCK_N : Nat)
    (out outGrad : Tile .real [BLOCK_M, BLOCK_N])
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : ℝ :=
  let scale := Tile.reduceSum (shape := [BLOCK_M, BLOCK_N]) ⟨1, by simp⟩ Bool.true outGrad
  let rowBroadcast : Broadcast [BLOCK_M, BLOCK_N] [BLOCK_M, 1] [BLOCK_M, BLOCK_N] :=
    Broadcast.consSame (Broadcast.consR Broadcast.nil)
  let sameBroadcast : Broadcast [BLOCK_M, BLOCK_N] [BLOCK_M, BLOCK_N] [BLOCK_M, BLOCK_N] :=
    Broadcast.consSame (Broadcast.consSame Broadcast.nil)
  WithBot.unbotD 0
    ((Tile.bop (NumericDType.sub .real) sameBroadcast
      outGrad
      (Tile.bop (NumericDType.mul .real) rowBroadcast
        (Tile.uop WithBot.realExp out) scale)).data idx)
```
</details>

<details><summary><code>logSoftmaxBackwardPureOutTile</code></summary>

```
/-- The `out`/`outGrad` tiles rebuilt from the `⊨` interface's per-lane inputs
`xs`/`ys`, with the masked-off lanes carrying the pinned `undef = 0` (the
backward loads have no `other=` default, so masked lanes enter as `s.undef`,
which `Masked2DKernelIO₂.Implements` pins to `0`). -/
```
```lean
noncomputable def logSoftmaxBackwardPureOutTile
    (p₀ M N BLOCK_M BLOCK_N : Nat) (xs : Fin (BLOCK_M * BLOCK_N) → ℝ) :
    Tile .real [BLOCK_M, BLOCK_N] :=
  { data := fun idx =>
      if active p₀ M N BLOCK_M idx then
        some (xs (laneOf BLOCK_M BLOCK_N idx))
      else some 0 }
```
</details>

<details><summary><code>logSoftmaxBackwardPureGradTile</code></summary>

```
/-- Companion of `logSoftmaxBackwardPureOutTile` for the second input. -/
```
```lean
noncomputable def logSoftmaxBackwardPureGradTile
    (p₀ M N BLOCK_M BLOCK_N : Nat) (ys : Fin (BLOCK_M * BLOCK_N) → ℝ) :
    Tile .real [BLOCK_M, BLOCK_N] :=
  { data := fun idx =>
      if active p₀ M N BLOCK_M idx then
        some (ys (laneOf BLOCK_M BLOCK_N idx))
      else some 0 }
```
</details>

<details><summary><code>log_softmax_backward_kernel</code></summary>

```
/-- Faithful transcription of `log_softmax.py`'s
`log_softmax_backward_kernel`. -/
```
```lean
def log_softmax_backward_kernel
    (out_ptr out_grad_ptr in_grad_ptr : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat) :
    ComputeKernel := triton {
  pid_m = tl.program_id(0)
  pid_k = tl.program_id(1)
  m_offset = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  n_offset = tl.arange(0, $(BLOCK_N))
  offsets = m_offset[:, None] * $(N) * $(K) + n_offset[None, :] * $(K) + pid_k
  mask = m_offset[:, None] < $(M) and n_offset[None, :] < $(N)
  out_ptrs = out_ptr + offsets
  out = tl.load(out_ptrs, mask=mask).to(tl.float32)
  out_grad_ptrs = out_grad_ptr + offsets
  out_grad = tl.load(out_grad_ptrs, mask=mask).to(tl.float32)
  scale = tl.sum(out_grad, 1)
  in_grad = out_grad - tl.exp((out).to(tl.float32)) * scale[:, None]
  in_grad_ptrs = in_grad_ptr + offsets
  tl.store(in_grad_ptrs, in_grad, mask=mask)
}
```
</details>

<details><summary><code>active</code></summary>

```
/-- The kernel's mask `m_offset < M ∧ n_offset < N`. -/
```
```lean
def active (p₀ M N BLOCK_M : Nat) (idx : TileIndex [BLOCK_M, BLOCK_N]) : Prop :=
  p₀ * BLOCK_M + idx.1.val < M ∧ idx.2.1.val < N
```
</details>

<details><summary><code>laneOf</code></summary>

```
/-- The lane of a logical `R × C` tile index — inverse of `laneIdx`. -/
```
```lean
def laneOf (R C : Nat) (idx : TileIndex [R, C]) : Fin (R * C) :=
  ⟨idx.1.val * C + idx.2.1.val, by
    have h1 : idx.1.val + 1 ≤ R := idx.1.isLt
    have h2 : idx.1.val * C + idx.2.1.val < idx.1.val * C + C :=
      Nat.add_lt_add_left idx.2.1.isLt _
    refine Nat.lt_of_lt_of_le h2 ?_
    rw [← Nat.succ_mul]
    exact Nat.mul_le_mul_right C h1⟩
```
</details>

## Public theorem: `log_softmax_kernel_correctness`

<details><summary>docstring</summary>

```
/-- **The headline**: `log_softmax_kernel` implements the row-wise log-softmax
on its masked IO signature — for every disjoint flat placement of
`input_ptr`/`output_ptr`, every program-id pair whose active lanes are in
bounds, and every launch state whose input window holds the tile `xs` at the
active lanes, the translated pointer kernel terminates, every active output
lane `j` holds `log(exp(x − rowmax) / ∑ exp(x − rowmax))` — the reduction core
`logSoftmaxOf` applied to the tile rebuilt from `xs` (with the kernel's own
`other = -inf` padding on the masked-off lanes, which is what makes the
row reduction independent of memory outside the window) — and every other
memory cell, including the masked-off lanes of the output window, is
unchanged.

Two honest, **truth-required** side hypotheses: `hOutInj` (distinct lanes of a
program's tile must not alias the same output cell — otherwise only the last
store survives) and `hBN : 0 < BLOCK_N` (`tl.max` over a zero-length reduction
axis has no value and `exec` returns `none`, so termination genuinely fails). -/
```
</details>

**Statement:**
```lean
specification log_softmax_kernel_correctness
    (output_ptr input_ptr : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat) (hBN : 0 < BLOCK_N)
    (hOutInj : ∀ p₀ p₁ : Nat, Function.Injective
      (fun j : Fin (BLOCK_M * BLOCK_N) =>
        outOffset p₀ p₁ N K BLOCK_M (laneIdx BLOCK_M BLOCK_N j))) :
    logSoftmaxIO output_ptr input_ptr M N K BLOCK_M BLOCK_N
      ⊨ fun p₀ _ xs j =>
          logSoftmaxOf BLOCK_M BLOCK_N
            (logSoftmaxPureTile p₀ M N BLOCK_M BLOCK_N xs)
            (laneIdx BLOCK_M BLOCK_N j)
```

**Assumptions / layout contracts:**
- `hBN : 0 < BLOCK_N`

**Closed-form spec defs (transitive):** `outOffset`, `laneIdx`, `logSoftmaxIO`, `logSoftmaxOf`, `logSoftmaxPureTile`, `log_softmax_kernel`, `active`, `laneOf`

<details><summary><code>outOffset</code></summary>

```
/-- The shared load/store address of tile cell `idx` for program `(p₀, p₁)`. -/
```
```lean
def outOffset (p₀ p₁ N K BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : Nat :=
  (p₀ * BLOCK_M + idx.1.val) * N * K + idx.2.1.val * K + p₁
```
</details>

<details><summary><code>laneIdx</code></summary>

```
/-- Lane `j` of the row-major flattening of an `R × C` tile. -/
```
```lean
def laneIdx (R C : Nat) (j : Fin (R * C)) : TileIndex [R, C] :=
  (⟨j.val / C, Nat.div_lt_of_lt_mul
      (Nat.lt_of_lt_of_le j.isLt (Nat.le_of_eq (Nat.mul_comm R C)))⟩,
   ⟨j.val % C, Nat.mod_lt _ (Nat.pos_of_ne_zero (by
      rintro rfl
      exact absurd j.isLt (by simp)))⟩,
   PUnit.unit)
```
</details>

<details><summary><code>logSoftmaxIO</code></summary>

```
/-- `log_softmax_kernel`'s masked **IO signature** — the whole kernel-specific
audit surface of the `⊨` headline:

* `inp`/`out` — which buffer is which argument (`input_ptr`, `output_ptr`);
* `B = BLOCK_M * BLOCK_N` — the tile flattened row-major into lanes, lane `j` =
  tile cell `(j / BLOCK_N, j % BLOCK_N)` (`laneIdx`);
* `read = write` — the shared window
  `(pid₀·BLOCK_M + m)·N·K + n·K + pid₁` (the kernel reads and writes the same
  offsets in two different buffers);
* `mask` — the kernel's `m_offset < M ∧ n_offset < N`; `writeMask` keeps the
  struct default (the load and the store carry the *same* mask).

Both program-id axes are used (`pid_m` tiles the `M` axis, `pid_k` is the
innermost coordinate). -/
```
```lean
def logSoftmaxIO (output_ptr input_ptr : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat) : Masked2DKernelIO₁ where
  kernel := log_softmax_kernel output_ptr input_ptr M N K BLOCK_M BLOCK_N
  inp := input_ptr
  out := output_ptr
  B := BLOCK_M * BLOCK_N
  read := fun p₀ p₁ j => outOffset p₀ p₁ N K BLOCK_M (laneIdx BLOCK_M BLOCK_N j)
  write := fun p₀ p₁ j => outOffset p₀ p₁ N K BLOCK_M (laneIdx BLOCK_M BLOCK_N j)
  mask := fun p₀ _ j => active p₀ M N BLOCK_M (laneIdx BLOCK_M BLOCK_N j)
```
</details>

<details><summary><code>logSoftmaxOf</code></summary>

```
/-- The kernel's reduction core as a function of the *loaded tile*:
`log(exp(inp − rowmax) / ∑ exp(inp − rowmax))` along axis 1. This is the pure
content of the `⊨` headline; the two tile-builders below feed it either the
memory contents (readback lemma) or the `⊨` interface's `xs` (headline). -/
```
```lean
noncomputable def logSoftmaxOf (BLOCK_M BLOCK_N : Nat)
    (inp : Tile .real [BLOCK_M, BLOCK_N])
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : ℝ :=
  match Tile.reduceMax (shape := [BLOCK_M, BLOCK_N]) ⟨1, by simp⟩ Bool.true inp with
  | some rowMax =>
      let rowBroadcast : Broadcast [BLOCK_M, BLOCK_N] [BLOCK_M, 1] [BLOCK_M, BLOCK_N] :=
        Broadcast.consSame (Broadcast.consR Broadcast.nil)
      let shifted := Tile.bop (NumericDType.sub .real) rowBroadcast inp rowMax
      let numerator := Tile.uop WithBot.realExp shifted
      let denominator := Tile.reduceSum (shape := [BLOCK_M, BLOCK_N]) ⟨1, by simp⟩ Bool.true numerator
      WithBot.unbotD 0
        ((Tile.uop WithBot.realLog
          (Tile.bop (NumericDType.div .real) rowBroadcast numerator denominator)).data idx)
  | none => 0
```
</details>

<details><summary><code>logSoftmaxPureTile</code></summary>

```
/-- The same tile rebuilt from the `⊨` interface's per-lane input `xs`. -/
```
```lean
noncomputable def logSoftmaxPureTile
    (p₀ M N BLOCK_M BLOCK_N : Nat) (xs : Fin (BLOCK_M * BLOCK_N) → ℝ) :
    Tile .real [BLOCK_M, BLOCK_N] :=
  { data := fun idx =>
      if active p₀ M N BLOCK_M idx then
        some (xs (laneOf BLOCK_M BLOCK_N idx))
      else none }
```
</details>

<details><summary><code>log_softmax_kernel</code></summary>

```
/-- Faithful transcription of `log_softmax.py`'s `log_softmax_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_M: tl.constexpr` / `BLOCK_N: tl.constexpr` -> Lean `Nat`
  parameters. -/
```
```lean
def log_softmax_kernel
    (output_ptr input_ptr : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat) :
    ComputeKernel := triton {
  pid_m = tl.program_id(0)
  pid_k = tl.program_id(1)
  m_offset = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  n_offset = tl.arange(0, $(BLOCK_N))
  offset = m_offset[:, None] * $(N) * $(K) + n_offset[None, :] * $(K) + pid_k
  mask = m_offset[:, None] < $(M) and n_offset[None, :] < $(N)
  input_ptrs = input_ptr + offset
  inp = (tl.load(input_ptrs, mask=mask, other=-float("inf"))).to(tl.float32)
  row_minus_max = inp - tl.max(inp, axis=1)[:, None]
  numerator = tl.exp(row_minus_max)
  denominator = tl.sum(numerator, axis=1)[:, None]
  softmax_output = tl.log(numerator / denominator)
  output_ptrs = output_ptr + offset
  tl.store(output_ptrs, softmax_output, mask=mask)
}
```
</details>

<details><summary><code>active</code></summary>

```
/-- The kernel's mask `m_offset < M ∧ n_offset < N`. -/
```
```lean
def active (p₀ M N BLOCK_M : Nat) (idx : TileIndex [BLOCK_M, BLOCK_N]) : Prop :=
  p₀ * BLOCK_M + idx.1.val < M ∧ idx.2.1.val < N
```
</details>

<details><summary><code>laneOf</code></summary>

```
/-- The lane of a logical `R × C` tile index — inverse of `laneIdx`. -/
```
```lean
def laneOf (R C : Nat) (idx : TileIndex [R, C]) : Fin (R * C) :=
  ⟨idx.1.val * C + idx.2.1.val, by
    have h1 : idx.1.val + 1 ≤ R := idx.1.isLt
    have h2 : idx.1.val * C + idx.2.1.val < idx.1.val * C + C :=
      Nat.add_lt_add_left idx.2.1.isLt _
    refine Nat.lt_of_lt_of_le h2 ?_
    rw [← Nat.succ_mul]
    exact Nat.mul_le_mul_right C h1⟩
```
</details>
