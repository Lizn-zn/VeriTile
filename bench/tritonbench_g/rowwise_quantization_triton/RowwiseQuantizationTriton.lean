import VeriTile.Triton

/-!
# `rowwise_quantization_triton` — strict per-kernel correctness

`_quantize_rowwise` is a per-row int8 quantizer: program `pid` loads one row of
`x` (a `P2`-wide tile masked to the first `BLOCK_SIZE` lanes), computes
`max_val = max(|x|)` over the row, rounds `127.0 · (x / max_val)` to int8 via
CUDA `llrint`, stores the row to `output_ptr`, and writes `max_val` to
`output_maxs + pid`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`_quantize_rowwise[grid](...)`, the grid size
`x.shape[0]`, the host-side `BLOCK_SIZE`/`P2` choice, and how the runtime
composes per-program writes into one buffer) is the *trusted boundary*, not a
proof obligation here. Because `pid` is universally quantified over `s`, the
per-program statement covers every program of the grid.

## Proof architecture

```
quantize_rowwise_blocked_output_summary_general                ← TOP THEOREM (dimension-general)
  ├─ quantize_rowwise_real_surface_toAlgorithm_blocked         faithful surface blocked at erasure (llrint)
  ├─ quantize_rowwise_scaled_store_slice_compute_correct       ← ComputeCorrect over the masked row store (scaled row output)
  │    └─ quantize_rowwise_scaled_store_slice_correct          ← algorithm-layer readback per lane
  └─ quantize_rowwise_max_store_slice_compute_correct          ← ComputeCorrect over the output_maxs scalar store (per-row max)
```

The pre-rounding spec is `scale127 · (x i / max_val)` per lane
(`quantizeRowwiseScaledSpec`) and `max_val` for the scalar
(`quantizeRowwiseMaxSpec`).

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` /
`num_warps` / `num_stages` are not modeled. The **faithful surface**
(`quantize_rowwise_real_surface`) is *blocked* at algorithm erasure: the CUDA
`tl.extra.cuda.libdevice.llrint` rounding and int8 cast are outside VeriTile's
real-tile arithmetic layer, so `toAlgorithm?` returns `Except.error`. What is
positively verified is the pre-rounding *scaled-store slice*: starting from a
precomputed per-row maximum `MaxVals`, the masked row writeback of
`scale127 · (x / max_val)`, plus the `output_maxs` scalar writeback. The
`max(|x|)` reduction is therefore taken as the precomputed `MaxVals` input here,
and the `llrint` rounding / int8 cast is the honest blocker recorded in each
summary.
-/

namespace VeriTile.Bench.TritonBenchG.RowwiseQuantizationTriton

open VeriTile.Triton

/-- Real-valued surface of `rowwise_quantization_triton.py`'s
`_quantize_rowwise`.

This preserves row addressing, `tl.abs`, masked max reduction, the `output_maxs`
store, CUDA `llrint` surface operation, and the scaled output expression. The
algorithm carrier records the pre-cast real value. -/
def quantize_rowwise_real_surface
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
  output = tl.extra.cuda.libdevice.llrint(127.0 * (x / max_val))
  tl.store(output_ptr + offsets, output, mask=row_mask)
  tl.store(output_maxs + pid, max_val)
}

/-- The faithful rowwise quantization surface is blocked at algorithm erasure by
the CUDA `llrint` rounding operation. The scaled-store slice below proves the
real-valued expression before backend-specific rounding/cast. -/
theorem quantize_rowwise_real_surface_toAlgorithm_blocked
    (x_ptr output_ptr output_maxs : RegionName)
    (_n_elements BLOCK_SIZE P2 : Nat) :
    ∃ err,
      (quantize_rowwise_real_surface x_ptr output_ptr output_maxs _n_elements
        BLOCK_SIZE P2).toAlgorithm? = Except.error err := by
  simp [quantize_rowwise_real_surface, ComputeExpr.toAlgorithm?]

/-- Proof-oriented scaled-output store slice of `rowwise_quantization_triton.py`'s
`_quantize_rowwise`.

The upstream kernel computes `max_val = max(abs(x))`, stores it in
`output_maxs`, rounds `127.0 * (x / max_val)` with CUDA `llrint`, and stores an
int8 row. VeriTile's current arithmetic layer models real tiles, so this slice
starts from a precomputed per-row maximum `MaxVals` and proves the masked
scaled output writeback before the backend-specific rounding/cast step. -/
def quantize_rowwise_scaled_store_slice
    (x_ptr output_ptr MaxVals : RegionName)
    (_n_elements BLOCK_SIZE P2 : Nat) (scale127 : ℝ) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  arange = tl.arange(0, $(P2))
  offsets = block_start + arange
  row_mask = arange < $(BLOCK_SIZE)
  x = tl.load(x_ptr + offsets, mask=row_mask)
  max_val = tl.load(MaxVals + pid)
  output = $(scale127) * (x / max_val)
  tl.store(output_ptr + offsets, output, mask=row_mask)
}

/-- Store slice for the per-row maxima produced by `_quantize_rowwise`.

The full kernel computes `max(abs(x))`; this proof slice starts from a
precomputed `MaxVals` scalar per row and proves the Python `output_maxs + pid`
writeback. -/
def quantize_rowwise_max_store_slice
    (MaxVals output_maxs : RegionName) : ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  max_val = tl.load(MaxVals + pid)
  tl.store(output_maxs + pid, max_val)
}

def offset (s : BlockState) (BLOCK_SIZE : Nat) (i : Fin P2) : Nat :=
  s.pid * BLOCK_SIZE + i.val

def maxOffset (s : BlockState) : Nat :=
  s.pid

noncomputable def quantizeRowwiseScaledSpec
    (s : BlockState) (x_ptr MaxVals : RegionName)
    (BLOCK_SIZE : Nat) (scale127 : ℝ) (i : Fin P2) : ℝ :=
  scale127 * (s.readMem x_ptr (offset s BLOCK_SIZE i) /
    s.readMem MaxVals s.pid)

noncomputable def quantizeRowwiseMaxSpec (s : BlockState) (MaxVals : RegionName) : ℝ :=
  s.readMem MaxVals (maxOffset s)

/-- Algorithm-layer correctness for the rowwise scaled-output store slice. -/
theorem quantize_rowwise_scaled_store_slice_correct
    (x_ptr output_ptr MaxVals : RegionName)
    (n_elements BLOCK_SIZE P2 : Nat) (scale127 : ℝ)
    (s : BlockState) :
    ∀ i : Fin P2,
      let outAddr := offset s BLOCK_SIZE i
      (exec (quantize_rowwise_scaled_store_slice x_ptr output_ptr MaxVals
            n_elements BLOCK_SIZE P2 scale127) s).map (·.readMem output_ptr outAddr)
        = some (if i.val < BLOCK_SIZE then
            quantizeRowwiseScaledSpec s x_ptr MaxVals BLOCK_SIZE scale127 i
          else s.readMem output_ptr outAddr) := by
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [P2] => s.pid * BLOCK_SIZE + idx.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [exec, quantize_rowwise_scaled_store_slice, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul, NumericDType.div,
        ComparableDType.lt, offset]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
  by_cases hi : i.val < BLOCK_SIZE
  · simp [hi, quantizeRowwiseScaledSpec, offset]
  · simp [hi]

/-- Compute-facing correctness for the rowwise scaled-output store slice. -/
theorem quantize_rowwise_scaled_store_slice_compute_correct
    (x_ptr output_ptr MaxVals : RegionName)
    (n_elements BLOCK_SIZE P2 : Nat) (scale127 : ℝ)
    (s : BlockState) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := quantize_rowwise_scaled_store_slice x_ptr output_ptr MaxVals
        n_elements BLOCK_SIZE P2 scale127)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin P2 => i.val < BLOCK_SIZE)
          (fun i => (output_ptr, offset s BLOCK_SIZE i)))
      (expected := fun i =>
        quantizeRowwiseScaledSpec s x_ptr MaxVals BLOCK_SIZE scale127 i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [quantize_rowwise_scaled_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have hi := quantize_rowwise_scaled_store_slice_correct x_ptr output_ptr MaxVals
    n_elements BLOCK_SIZE P2 scale127 s i
  rw [hExec] at hi
  simpa [hActive] using Option.some.inj hi

/-- Compute-facing correctness for the per-row max writeback slice. -/
theorem quantize_rowwise_max_store_slice_compute_correct
    (MaxVals output_maxs : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := quantize_rowwise_max_store_slice MaxVals output_maxs)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.scalar output_maxs (maxOffset s))
      (expected := fun _ : PUnit => quantizeRowwiseMaxSpec s MaxVals) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [quantize_rowwise_max_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i
  simp [exec, quantize_rowwise_max_store_slice, stepStmts, stepStmt, evalOp] at hExec
  subst s'
  simp [ComputeCorrect.WriteMap.scalar, maxOffset, quantizeRowwiseMaxSpec]

/-- **Dimension-general blocked output summary.** For arbitrary `_n_elements`,
`BLOCK_SIZE`, padded width `P2`, and real scale `scale127` (and any program id in
`s`), the faithful full surface is recorded as **blocked** at algorithm erasure
(it stores CUDA `llrint`/int8 results, not the real-valued pre-rounding
expression), while the checked proof slices realize the genuine pre-rounding
scaled row output `scale127 * (x / max_val)` (`quantizeRowwiseScaledSpec`) at
every in-range lane and the per-row `output_maxs` writeback
(`quantizeRowwiseMaxSpec`). This holds over arbitrary (symbolic) dimensions and
`scale127`. The `max(|x|)` reduction is
taken as the precomputed `MaxVals` input; the `llrint` rounding / int8 cast
remain the honest, unmodeled blocker. -/
specification quantize_rowwise_blocked_output_summary_general
    (x_ptr output_ptr MaxVals output_maxs : RegionName)
    (_n_elements BLOCK_SIZE P2 : Nat) (scale127 : ℝ) (s : BlockState) :
    (∃ err,
      (quantize_rowwise_real_surface x_ptr output_ptr output_maxs
        _n_elements BLOCK_SIZE P2).toAlgorithm? = Except.error err) ∧
    ((ComputeCorrect.Realizes_without_Rounding
      (kernel := quantize_rowwise_scaled_store_slice x_ptr output_ptr MaxVals
        _n_elements BLOCK_SIZE P2 scale127)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin P2 => i.val < BLOCK_SIZE)
          (fun i => (output_ptr, offset s BLOCK_SIZE i)))
      (expected := fun i =>
        quantizeRowwiseScaledSpec s x_ptr MaxVals BLOCK_SIZE scale127 i)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := quantize_rowwise_max_store_slice MaxVals output_maxs)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.scalar output_maxs (maxOffset s))
      (expected := fun _ : PUnit => quantizeRowwiseMaxSpec s MaxVals))) :=
  ⟨quantize_rowwise_real_surface_toAlgorithm_blocked x_ptr output_ptr output_maxs
      _n_elements BLOCK_SIZE P2,
   quantize_rowwise_scaled_store_slice_compute_correct x_ptr output_ptr MaxVals
      _n_elements BLOCK_SIZE P2 scale127 s,
   quantize_rowwise_max_store_slice_compute_correct MaxVals output_maxs s⟩

end VeriTile.Bench.TritonBenchG.RowwiseQuantizationTriton
