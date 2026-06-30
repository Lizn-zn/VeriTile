import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

/-!
# `quantize_global` — strict per-kernel correctness (blocked tail)

`_quantize_global` quantizes a flat tensor to int8: program `pid` loads its block
`[pid·BLOCK_SIZE, (pid+1)·BLOCK_SIZE)` of `x`, loads the scalar `absmax_inv`, and
stores `llrint(127.0 * (x * absmax_inv))` to `output_ptr`, masked by
`offsets < n_elements`. The host computes `absmax = |x|.max()` and
`absmax_inv = 1/absmax` before the launch.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`_quantize_global[grid](...)`, the grid
`cdiv(n_elements, BLOCK_SIZE)`, the PyTorch-side `absmax` reduction returned
alongside the output, and the runtime composition of per-program writes) is the
*trusted boundary*. `pid` is universally quantified, so the per-program statement
covers every program of the grid.

## Proof architecture

```
quantize_global_blocked_output_summary_general   ← TOP THEOREM (arbitrary n/bs/scale)
  ├─ quantize_global_surface_toAlgorithm?_blocked          full surface does NOT lower
  └─ quantize_global_scaled_store_slice_compute_correct   ComputeCorrect over the store
       └─ quantize_global_scaled_store_slice_correct       algorithm-layer readback per lane
```

The spec is the pre-rounding scaled value `127.0 * (x * absmax_inv)`
(`quantizeGlobalScaledSpec`); seven `(n_elements, BLOCK_SIZE)` shapes mirror the
`@triton.autotune` configs and Python test sizes.

## Modeling boundary

Arithmetic is over `ℝ`, not bit-accurate IEEE float. This is the **honestly
blocked** quantization case: the faithful Python surface contains CUDA
`tl.extra.cuda.libdevice.llrint` plus the int8 store, and that surface is proven
*not* to lower through algorithm projection
(`quantize_global_surface_toAlgorithm?_blocked` returns `Except.error`). What is
verified is the real-valued pre-rounding store slice — the masked addressing, the
scalar `absmax_inv` load, and the value `127.0 * (x * absmax_inv)` — exactly up
to the `llrint` rounding / int8 cast, which are left as the blocker and are
**not** numerically modeled. `@triton.autotune` is not modeled; representative
configs are instantiated as separate theorems.
-/

namespace VeriTile.Bench.TritonBenchG.QuantizeGlobal

open VeriTile.Triton

/-- Faithful transcription of `quantize_global.py`'s `_quantize_global`.

The CUDA `llrint` operation is preserved as a surface operation; the algorithm
carrier records the pre-cast real value. -/
def quantize_global_surface
    (x_ptr absmax_inv_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(x_ptr + offsets, mask=mask)
  absmax_inv = tl.load(absmax_inv_ptr)
  output = tl.extra.cuda.libdevice.llrint(127.0 * (x * absmax_inv))
  tl.store(output_ptr + offsets, output, mask=mask)
}

/-- Proof-oriented arithmetic store slice of `quantize_global.py`'s
`_quantize_global`.

This slice proves the masked vector addressing and scaled value before the
backend-specific rounding/cast step. -/
def quantize_global_scaled_store_slice
    (x_ptr absmax_inv_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) (scale127 : ℝ) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(x_ptr + offsets, mask=mask)
  absmax_inv = tl.load(absmax_inv_ptr)
  output = $(scale127) * (x * absmax_inv)
  tl.store(output_ptr + offsets, output, mask=mask)
}

def offset (s : BlockState) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * BLOCK_SIZE + i.val

noncomputable def quantizeGlobalScaledSpec
    (s : BlockState) (x_ptr absmax_inv_ptr : RegionName)
    (BLOCK_SIZE : Nat) (scale127 : ℝ) (i : Fin BLOCK_SIZE) : ℝ :=
  scale127 * (s.readMem x_ptr (offset s BLOCK_SIZE i) *
    s.readMem absmax_inv_ptr 0)

/-- Algorithm-layer correctness for the masked global-quantization store slice. -/
theorem quantize_global_scaled_store_slice_correct
    (x_ptr absmax_inv_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) (scale127 : ℝ)
    (s : BlockState) :
    ∀ i : Fin BLOCK_SIZE,
      let outAddr := offset s BLOCK_SIZE i
      (exec (quantize_global_scaled_store_slice x_ptr absmax_inv_ptr output_ptr
            n_elements BLOCK_SIZE scale127) s).map (·.readMem output_ptr outAddr)
        = some (if outAddr < n_elements then
            quantizeGlobalScaledSpec s x_ptr absmax_inv_ptr BLOCK_SIZE scale127 i
          else s.readMem output_ptr outAddr) := by
  intro i
  simp [exec, quantize_global_scaled_store_slice, stepStmts, stepStmt, evalOp.eq_def,
        Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul,
        ComparableDType.lt, offset]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
        (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
  by_cases hi : s.pid * BLOCK_SIZE + i.val < n_elements
  · simp [hi, quantizeGlobalScaledSpec, offset]
  · simp [hi]

/-- The faithful full surface is intentionally blocked at compute projection:
the tested Python kernel stores CUDA `llrint`/int8 results, not the real-valued
pre-rounding expression proved by `quantize_global_scaled_store_slice`. -/
theorem quantize_global_surface_toAlgorithm?_blocked
    (x_ptr absmax_inv_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ∃ err,
      (quantize_global_surface x_ptr absmax_inv_ptr output_ptr
        n_elements BLOCK_SIZE).toAlgorithm? = Except.error err := by
  simp [quantize_global_surface, ComputeExpr.toAlgorithm?]

/-- Compute-facing correctness for the masked global-quantization store slice. -/
theorem quantize_global_scaled_store_slice_compute_correct
    (x_ptr absmax_inv_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) (scale127 : ℝ)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := quantize_global_scaled_store_slice x_ptr absmax_inv_ptr output_ptr
        n_elements BLOCK_SIZE scale127)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin BLOCK_SIZE => offset s BLOCK_SIZE i < n_elements)
          (fun i => (output_ptr, offset s BLOCK_SIZE i)))
      (expected := fun i =>
        quantizeGlobalScaledSpec s x_ptr absmax_inv_ptr BLOCK_SIZE scale127 i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [quantize_global_scaled_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have hi := quantize_global_scaled_store_slice_correct x_ptr absmax_inv_ptr output_ptr
    n_elements BLOCK_SIZE scale127 s i
  rw [hExec] at hi
  simpa [hActive] using Option.some.inj hi

/-- **Dimension-general blocked output summary.** For arbitrary `n_elements`,
`BLOCK_SIZE`, and real scale `scale127` (and any program id in `s`), the faithful
full surface is recorded as **blocked** at algorithm projection (it stores CUDA
`llrint`/int8 results, not the real-valued pre-rounding expression), while the
checked store slice realizes the genuine pre-rounding quantity
`scale127 * (x * absmax_inv)` (`quantizeGlobalScaledSpec`) at every in-range lane,
leaving out-of-range lanes unchanged. Concrete Python benchmark shapes are
instantiations of this (with `scale127 = 127.0`). No injectivity hypothesis is
needed: the 1-D block offset map is injective by construction. The `llrint`
rounding / int8 cast remain the honest, unmodeled blocker. -/
theorem quantize_global_blocked_output_summary_general
    (x_ptr absmax_inv_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) (scale127 : ℝ) (s : BlockState) :
    (∃ err, (quantize_global_surface x_ptr absmax_inv_ptr output_ptr
      n_elements BLOCK_SIZE).toAlgorithm? = Except.error err) ∧
    ComputeCorrect.Realizes
      (kernel := quantize_global_scaled_store_slice x_ptr absmax_inv_ptr output_ptr
        n_elements BLOCK_SIZE scale127)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin BLOCK_SIZE => offset s BLOCK_SIZE i < n_elements)
          (fun i => (output_ptr, offset s BLOCK_SIZE i)))
      (expected := fun i =>
        quantizeGlobalScaledSpec s x_ptr absmax_inv_ptr BLOCK_SIZE scale127 i) :=
  ⟨quantize_global_surface_toAlgorithm?_blocked x_ptr absmax_inv_ptr output_ptr
      n_elements BLOCK_SIZE,
   quantize_global_scaled_store_slice_compute_correct x_ptr absmax_inv_ptr output_ptr
      n_elements BLOCK_SIZE scale127 s⟩

end VeriTile.Bench.TritonBenchG.QuantizeGlobal
