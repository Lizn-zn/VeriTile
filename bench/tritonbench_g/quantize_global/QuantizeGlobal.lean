import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.QuantizeGlobal

open VeriTile.Triton

/-- Surface transcription of `quantize_global.py`'s `_quantize_global`.

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
  have h_inj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE] => s.pid * BLOCK_SIZE + idx.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [exec, quantize_global_scaled_store_slice, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul,
        ComparableDType.lt, offset]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
  by_cases hi : s.pid * BLOCK_SIZE + i.val < n_elements
  · simp [hi, quantizeGlobalScaledSpec, offset]
  · simp [hi]

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

end VeriTile.Bench.TritonBenchG.QuantizeGlobal
