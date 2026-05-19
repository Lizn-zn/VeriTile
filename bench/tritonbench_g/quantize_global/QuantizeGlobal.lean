import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

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
  simp [exec, quantize_global_scaled_store_slice, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul,
        ComparableDType.lt, offset]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
        (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
  by_cases hi : s.pid * BLOCK_SIZE + i.val < n_elements
  · simp [hi, quantizeGlobalScaledSpec, offset]
  · simp [hi]

/-- Surface-level pre-rounding expression: `127.0 * (x * absmax_inv)` at lane
`i`. The full surface does not currently erase to an algorithm kernel because
CUDA `llrint` is compute-only. -/
noncomputable def quantizeGlobalSurfaceSpec
    (s : BlockState) (x_ptr absmax_inv_ptr : RegionName)
    (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : ℝ :=
  127.0 * (s.readMem x_ptr (offset s BLOCK_SIZE i) *
    s.readMem absmax_inv_ptr 0)

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

theorem quantize_global_python_n1024_bs1024_compute_correct
    (x_ptr absmax_inv_ptr output_ptr : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := quantize_global_scaled_store_slice x_ptr absmax_inv_ptr output_ptr
        1024 1024 127.0)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin 1024 => offset s 1024 i < 1024)
          (fun i => (output_ptr, offset s 1024 i)))
      (expected := fun i =>
        quantizeGlobalScaledSpec s x_ptr absmax_inv_ptr 1024 127.0 i) := by
  exact quantize_global_scaled_store_slice_compute_correct x_ptr absmax_inv_ptr
    output_ptr 1024 1024 127.0 s

theorem quantize_global_python_n2048_bs1024_compute_correct
    (x_ptr absmax_inv_ptr output_ptr : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := quantize_global_scaled_store_slice x_ptr absmax_inv_ptr output_ptr
        2048 1024 127.0)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin 1024 => offset s 1024 i < 2048)
          (fun i => (output_ptr, offset s 1024 i)))
      (expected := fun i =>
        quantizeGlobalScaledSpec s x_ptr absmax_inv_ptr 1024 127.0 i) := by
  exact quantize_global_scaled_store_slice_compute_correct x_ptr absmax_inv_ptr
    output_ptr 2048 1024 127.0 s

theorem quantize_global_python_n2048_bs2048_compute_correct
    (x_ptr absmax_inv_ptr output_ptr : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := quantize_global_scaled_store_slice x_ptr absmax_inv_ptr output_ptr
        2048 2048 127.0)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin 2048 => offset s 2048 i < 2048)
          (fun i => (output_ptr, offset s 2048 i)))
      (expected := fun i =>
        quantizeGlobalScaledSpec s x_ptr absmax_inv_ptr 2048 127.0 i) := by
  exact quantize_global_scaled_store_slice_compute_correct x_ptr absmax_inv_ptr
    output_ptr 2048 2048 127.0 s

theorem quantize_global_python_n3072_bs1024_compute_correct
    (x_ptr absmax_inv_ptr output_ptr : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := quantize_global_scaled_store_slice x_ptr absmax_inv_ptr output_ptr
        3072 1024 127.0)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin 1024 => offset s 1024 i < 3072)
          (fun i => (output_ptr, offset s 1024 i)))
      (expected := fun i =>
        quantizeGlobalScaledSpec s x_ptr absmax_inv_ptr 1024 127.0 i) := by
  exact quantize_global_scaled_store_slice_compute_correct x_ptr absmax_inv_ptr
    output_ptr 3072 1024 127.0 s

theorem quantize_global_python_n3072_bs2048_compute_correct
    (x_ptr absmax_inv_ptr output_ptr : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := quantize_global_scaled_store_slice x_ptr absmax_inv_ptr output_ptr
        3072 2048 127.0)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin 2048 => offset s 2048 i < 3072)
          (fun i => (output_ptr, offset s 2048 i)))
      (expected := fun i =>
        quantizeGlobalScaledSpec s x_ptr absmax_inv_ptr 2048 127.0 i) := by
  exact quantize_global_scaled_store_slice_compute_correct x_ptr absmax_inv_ptr
    output_ptr 3072 2048 127.0 s

theorem quantize_global_python_n4096_bs1024_compute_correct
    (x_ptr absmax_inv_ptr output_ptr : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := quantize_global_scaled_store_slice x_ptr absmax_inv_ptr output_ptr
        4096 1024 127.0)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin 1024 => offset s 1024 i < 4096)
          (fun i => (output_ptr, offset s 1024 i)))
      (expected := fun i =>
        quantizeGlobalScaledSpec s x_ptr absmax_inv_ptr 1024 127.0 i) := by
  exact quantize_global_scaled_store_slice_compute_correct x_ptr absmax_inv_ptr
    output_ptr 4096 1024 127.0 s

theorem quantize_global_python_n4096_bs2048_compute_correct
    (x_ptr absmax_inv_ptr output_ptr : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := quantize_global_scaled_store_slice x_ptr absmax_inv_ptr output_ptr
        4096 2048 127.0)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin 2048 => offset s 2048 i < 4096)
          (fun i => (output_ptr, offset s 2048 i)))
      (expected := fun i =>
        quantizeGlobalScaledSpec s x_ptr absmax_inv_ptr 2048 127.0 i) := by
  exact quantize_global_scaled_store_slice_compute_correct x_ptr absmax_inv_ptr
    output_ptr 4096 2048 127.0 s

end VeriTile.Bench.TritonBenchG.QuantizeGlobal
