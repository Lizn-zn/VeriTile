import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

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
    ComputeCorrect.Realizes
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
    ComputeCorrect.Realizes
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

/-- Python case 1: `x.shape=(2,3)`, `BLOCK_SIZE=3`, `P2=4`,
`n_elements=6`. This proves the real-valued scaled row output before CUDA
`llrint`/int8 casting. -/
theorem quantize_rowwise_python_case1_scaled_output_compute_correct
    (x_ptr output_ptr MaxVals : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := quantize_rowwise_scaled_store_slice x_ptr output_ptr MaxVals
        6 3 4 127.0)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin 4 => i.val < 3)
          (fun i => (output_ptr, offset s 3 i)))
      (expected := fun i =>
        quantizeRowwiseScaledSpec s x_ptr MaxVals 3 127.0 i) := by
  exact quantize_rowwise_scaled_store_slice_compute_correct x_ptr output_ptr
    MaxVals 6 3 4 127.0 s

/-- Python case 3: `x.shape=(2,5)`, `BLOCK_SIZE=5`, `P2=8`,
`n_elements=10`. -/
theorem quantize_rowwise_python_case3_scaled_output_compute_correct
    (x_ptr output_ptr MaxVals : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := quantize_rowwise_scaled_store_slice x_ptr output_ptr MaxVals
        10 5 8 127.0)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin 8 => i.val < 5)
          (fun i => (output_ptr, offset s 5 i)))
      (expected := fun i =>
        quantizeRowwiseScaledSpec s x_ptr MaxVals 5 127.0 i) := by
  exact quantize_rowwise_scaled_store_slice_compute_correct x_ptr output_ptr
    MaxVals 10 5 8 127.0 s

theorem quantize_rowwise_python_output_maxs_compute_correct
    (MaxVals output_maxs : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := quantize_rowwise_max_store_slice MaxVals output_maxs)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.scalar output_maxs (maxOffset s))
      (expected := fun _ : PUnit => quantizeRowwiseMaxSpec s MaxVals) := by
  exact quantize_rowwise_max_store_slice_compute_correct MaxVals output_maxs s

/-- Python case 1 all-output coverage: scaled row output plus the per-row
`output_maxs` writeback. -/
theorem quantize_rowwise_python_case1_all_outputs_compute_correct
    (x_ptr output_ptr MaxVals output_maxs : RegionName) (s : BlockState) :
    (ComputeCorrect.Realizes
      (kernel := quantize_rowwise_scaled_store_slice x_ptr output_ptr MaxVals
        6 3 4 127.0)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin 4 => i.val < 3)
          (fun i => (output_ptr, offset s 3 i)))
      (expected := fun i =>
        quantizeRowwiseScaledSpec s x_ptr MaxVals 3 127.0 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := quantize_rowwise_max_store_slice MaxVals output_maxs)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.scalar output_maxs (maxOffset s))
      (expected := fun _ : PUnit => quantizeRowwiseMaxSpec s MaxVals)) := by
  constructor
  · exact quantize_rowwise_python_case1_scaled_output_compute_correct
      x_ptr output_ptr MaxVals s
  · exact quantize_rowwise_python_output_maxs_compute_correct
      MaxVals output_maxs s

/-- Python case 3 all-output coverage: scaled row output plus the per-row
`output_maxs` writeback. -/
theorem quantize_rowwise_python_case3_all_outputs_compute_correct
    (x_ptr output_ptr MaxVals output_maxs : RegionName) (s : BlockState) :
    (ComputeCorrect.Realizes
      (kernel := quantize_rowwise_scaled_store_slice x_ptr output_ptr MaxVals
        10 5 8 127.0)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin 8 => i.val < 5)
          (fun i => (output_ptr, offset s 5 i)))
      (expected := fun i =>
        quantizeRowwiseScaledSpec s x_ptr MaxVals 5 127.0 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := quantize_rowwise_max_store_slice MaxVals output_maxs)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.scalar output_maxs (maxOffset s))
      (expected := fun _ : PUnit => quantizeRowwiseMaxSpec s MaxVals)) := by
  constructor
  · exact quantize_rowwise_python_case3_scaled_output_compute_correct
      x_ptr output_ptr MaxVals s
  · exact quantize_rowwise_python_output_maxs_compute_correct
      MaxVals output_maxs s

/-- Public Python case 1 summary for rowwise quantization.

The faithful surface is intentionally recorded as blocked at algorithm erasure
by CUDA `llrint`. The accompanying proof slices cover the real-valued scaled
output before backend rounding/cast and the per-row `output_maxs` writeback. -/
theorem quantize_rowwise_python_case1_blocked_output_summary
    (x_ptr output_ptr MaxVals output_maxs : RegionName) (s : BlockState) :
    (∃ err,
      (quantize_rowwise_real_surface x_ptr output_ptr output_maxs
        6 3 4).toAlgorithm? = Except.error err) ∧
    ((ComputeCorrect.Realizes
      (kernel := quantize_rowwise_scaled_store_slice x_ptr output_ptr MaxVals
        6 3 4 127.0)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin 4 => i.val < 3)
          (fun i => (output_ptr, offset s 3 i)))
      (expected := fun i =>
        quantizeRowwiseScaledSpec s x_ptr MaxVals 3 127.0 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := quantize_rowwise_max_store_slice MaxVals output_maxs)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.scalar output_maxs (maxOffset s))
      (expected := fun _ : PUnit => quantizeRowwiseMaxSpec s MaxVals))) := by
  constructor
  · exact quantize_rowwise_real_surface_toAlgorithm_blocked
      x_ptr output_ptr output_maxs 6 3 4
  · exact quantize_rowwise_python_case1_all_outputs_compute_correct
      x_ptr output_ptr MaxVals output_maxs s

/-- Public Python case 3 summary for rowwise quantization.

As in case 1, this keeps the faithful `llrint` blocker explicit while exposing
the checked real-valued scaled output and `output_maxs` proof slices. -/
theorem quantize_rowwise_python_case3_blocked_output_summary
    (x_ptr output_ptr MaxVals output_maxs : RegionName) (s : BlockState) :
    (∃ err,
      (quantize_rowwise_real_surface x_ptr output_ptr output_maxs
        10 5 8).toAlgorithm? = Except.error err) ∧
    ((ComputeCorrect.Realizes
      (kernel := quantize_rowwise_scaled_store_slice x_ptr output_ptr MaxVals
        10 5 8 127.0)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin 8 => i.val < 5)
          (fun i => (output_ptr, offset s 5 i)))
      (expected := fun i =>
        quantizeRowwiseScaledSpec s x_ptr MaxVals 5 127.0 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := quantize_rowwise_max_store_slice MaxVals output_maxs)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.scalar output_maxs (maxOffset s))
      (expected := fun _ : PUnit => quantizeRowwiseMaxSpec s MaxVals))) := by
  constructor
  · exact quantize_rowwise_real_surface_toAlgorithm_blocked
      x_ptr output_ptr output_maxs 10 5 8
  · exact quantize_rowwise_python_case3_all_outputs_compute_correct
      x_ptr output_ptr MaxVals output_maxs s

end VeriTile.Bench.TritonBenchG.RowwiseQuantizationTriton
