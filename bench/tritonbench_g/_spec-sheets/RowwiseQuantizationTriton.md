# Spec sheet — `bench/tritonbench_g/rowwise_quantization_triton/RowwiseQuantizationTriton.lean`

**Python source:** `bench/tritonbench_g/rowwise_quantization_triton/rowwise_quantization_triton.py`

## Public theorem: `quantize_rowwise_correctness`

<details><summary>docstring</summary>

```
/-- **The headline**: `_quantize_rowwise` (with the CUDA `llrint` rounding
erased — see `quantize_rowwise_real_surface_toAlgorithm_blocked` for the honest
blocker) implements the exact rowwise-quantization pair on its masked
two-output IO signature. For every disjoint flat placement of the buffers,
every program id whose active lanes and scalar max cell are in bounds, and
every launch state whose active `x`-row lanes hold `xs`, the translated pointer
kernel terminates, every active output lane `j` holds
`qrOutSpec … = 127 · (xs j / max_val)`, the cell `output_maxs[pid]` holds
`qrMaxSpec … = max_val`, and every other memory cell is unchanged. The row
maximum `max_val` is **computed by the verified kernel**: it is the `sup'` of
`tl.where(row_mask, |x|, 0)` over the whole `P2` tile.

Side conditions, both genuinely forced: `0 < P2` (the `max` reduce, like
`Finset.sup'`, is only defined on a non-empty tile, and the unmasked scalar
store's bound and frame exclusion are carried by the lane-`0` gate
`writeMask2`, which needs a lane) and `output_ptr ≠ output_maxs` (the unmasked
scalar max store must not alias the masked row store). Proof:
`Masked2DKernelIO₂ₓ₂.Implements.intro` assembles the region-model masked triple
with the flat-memory bridge side conditions. -/
```
</details>

**Statement:**
```lean
specification quantize_rowwise_correctness
    (x_ptr inert output_ptr output_maxs : RegionName)
    (n_elements BLOCK_SIZE P2 : Nat) (hP : 0 < P2)
    (hRegions : output_ptr ≠ output_maxs) :
    quantizeRowwiseIO x_ptr inert output_ptr output_maxs n_elements
        BLOCK_SIZE P2 ⊨
      fun _ _ xs _ =>
        (fun i => qrOutSpec BLOCK_SIZE P2 xs i,
         fun _ => qrMaxSpec BLOCK_SIZE P2 xs)
```

**Assumptions / layout contracts:**
- `hP : 0 < P2`
- `hRegions : output_ptr ≠ output_maxs`

**Closed-form spec defs (transitive):** `quantizeRowwiseIO`, `qrOutSpec`, `qrMaxSpec`, `quantize_rowwise_scaled`, `qrMaxCarrier`, `qrAbsTile`

<details><summary><code>quantizeRowwiseIO</code></summary>

```
/-- `_quantize_rowwise`'s masked two-output **IO signature** — the whole
kernel-specific audit surface of the `⊨` headline:

* `in1`/`out1`/`out2` — the input row buffer `x`, the quantized row buffer
  `output`, and the per-row maxima vector `output_maxs`;
* `in2` — a **declared-inert allocation slot**. The Python kernel has a single
  input; this two-input skin's second read channel is switched off
  (`read2Mask := fun _ _ _ => False`), so it carries no obligation on either
  side — the headline merely also quantifies over a fourth, untouched buffer;
* `B = P2` — the padded row window each program owns
  (`P2 = next_power_of_2(n_cols)` on the host side);
* `read1`/`write1` — program `pid` reads and writes its row at
  `pid · BLOCK_SIZE + j` (the host-side one-program-per-row launch);
* `write2` — the **scalar** cell `output_maxs[pid]`, the same for every lane;
* `mask` — the active lanes `j < BLOCK_SIZE`, i.e. the Python `row_mask`; the
  load mask and the row-store mask coincide, so `writeMask1` keeps its `mask`
  default;
* `writeMask2` — lane `0` carries the scalar row maximum; the other lanes are
  write-inactive.

The grid is 1-D, so the second program-id axis is an unused parameter: windows
and masks are constant in `pid₁`. The windows and masks are declared, not
parsed from the kernel; the headline **proves** the kernel's actual addressing
and masking match them. -/
```
```lean
def quantizeRowwiseIO (x_ptr inert output_ptr output_maxs : RegionName)
    (n_elements BLOCK_SIZE P2 : Nat) : Masked2DKernelIO₂ₓ₂ where
  kernel := quantize_rowwise_scaled x_ptr output_ptr output_maxs n_elements
    BLOCK_SIZE P2
  in1 := x_ptr
  in2 := inert
  out1 := output_ptr
  out2 := output_maxs
  B := P2
  read1 := fun pid _ j => pid * BLOCK_SIZE + j.val
  read2 := fun _ _ _ => 0
  write1 := fun pid _ j => pid * BLOCK_SIZE + j.val
  write2 := fun pid _ _ => pid
  mask := fun _ _ j => j.val < BLOCK_SIZE
  read2Mask := fun _ _ _ => False
  writeMask2 := fun _ _ j => j.val = 0
```
</details>

<details><summary><code>qrOutSpec</code></summary>

```
/-- The value written to the active output row lane `i`: `127 · (x i / max)`. -/
```
```lean
noncomputable def qrOutSpec (BLOCK_SIZE P2 : Nat) (xs : Fin P2 → ℝ)
    (i : Fin P2) : ℝ :=
  match qrMaxCarrier BLOCK_SIZE P2 xs with
  | some m =>
      WithBot.unbotD 0
        (Option.map (fun b => 127.0 * (xs i / b)) (m.data PUnit.unit))
  | none => 0
```
</details>

<details><summary><code>qrMaxSpec</code></summary>

```
/-- The scalar written to `output_maxs[pid]`. -/
```
```lean
noncomputable def qrMaxSpec (BLOCK_SIZE P2 : Nat) (xs : Fin P2 → ℝ) : ℝ :=
  match qrMaxCarrier BLOCK_SIZE P2 xs with
  | some m => WithBot.unbotD 0 (m.data PUnit.unit)
  | none => 0
```
</details>

<details><summary><code>quantize_rowwise_scaled</code></summary>

```
/-- The verified kernel: `_quantize_rowwise` with the CUDA `llrint` rounding
(and the implicit int8 cast) erased. Every other statement — the row
addressing, the row mask, `tl.abs`, the masked max reduction, the scaled row
store and the scalar `output_maxs` store — is transcribed 1:1 from the Python
kernel. -/
```
```lean
def quantize_rowwise_scaled
    (x_ptr output_ptr output_maxs : RegionName)
    (_n_elements BLOCK_SIZE P2 : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  arange = tl.arange(0, $(P2))
  offsets = block_start + arange
  row_mask = arange < $(BLOCK_SIZE)
  x = tl.load(x_ptr + offsets, mask=row_mask)
  abs_x = tl.abs(x)
  max_val = tl.max(tl.where(row_mask, abs_x, 0.0), axis=0)
  output = 127.0 * (x / max_val)
  tl.store(output_ptr + offsets, output, mask=row_mask)
  tl.store(output_maxs + pid, max_val)
}
```
</details>

<details><summary><code>qrMaxCarrier</code></summary>

```
/-- The row maximum carrier `max_val = tl.max(tl.where(row_mask, |x|, 0))`. -/
```
```lean
noncomputable def qrMaxCarrier (BLOCK_SIZE P2 : Nat) (xs : Fin P2 → ℝ) :
    Option (Tile .real (TileShape.eraseAxis [P2] ⟨0, by simp⟩)) :=
  Tile.reduceMaxDrop (shape := [P2]) ⟨0, by simp⟩ (qrAbsTile BLOCK_SIZE P2 xs)
```
</details>

<details><summary><code>qrAbsTile</code></summary>

```
/-- The reduced tile `tl.where(row_mask, tl.abs(x), 0.0)`: active lanes hold
`|xs j|`, the padded lanes hold `0`. -/
```
```lean
noncomputable def qrAbsTile (BLOCK_SIZE P2 : Nat) (xs : Fin P2 → ℝ) :
    Tile .real [P2] :=
  { data := fun idx =>
      if idx.1.val < BLOCK_SIZE then
        some (if xs idx.1 < 0 then -xs idx.1 else xs idx.1)
      else some (0 : ℝ) }
```
</details>
