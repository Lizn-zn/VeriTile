import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.MatmulTriton1

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Faithful transcription of `matmul_triton1.py`'s `matmul_kernel`.

Python passes `m_size` but the kernel body does not use it; this surface keeps
the signature position as `_m_size`. -/
def matmul_triton1_surface
    (X Y Z : RegionName)
    (_m_size k_size n_size m_block_size k_block_size n_block_size : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  num_n_blocks = tl.cdiv($(n_size), $(n_block_size))
  m_block = pid // num_n_blocks
  n_block = pid % num_n_blocks
  m_offsets = tl.arange(0, $(m_block_size)) + m_block * $(m_block_size)
  n_offsets = tl.arange(0, $(n_block_size)) + n_block * $(n_block_size)
  k_offsets = tl.arange(0, $(k_block_size))
  x_ptrs = X + m_offsets[:, None] * $(k_size) + k_offsets[None, :]
  y_ptrs = Y + k_offsets[:, None] * $(n_size) + n_offsets[None, :]
  z_ptrs = Z + m_offsets[:, None] * $(n_size) + n_offsets[None, :]
  z = tl.zeros([$(m_block_size), $(n_block_size)], dtype=tl.float32)
  for kk in range($(0), $(k_size), $(k_block_size)) {
    x_sub = tl.load(x_ptrs)
    y_sub = tl.load(y_ptrs)
    z += tl.dot(x_sub, y_sub, allow_tf32=false)
    x_ptrs += $(k_block_size)
    y_ptrs += $(k_block_size) * $(n_size)
  }
  tl.store(z_ptrs, z)
}

/-- The full `matmul_triton1` surface lowers to the algorithm layer. -/
theorem matmul_triton1_surface_toAlgorithm_supported
    (X Y Z : RegionName)
    (m_size k_size n_size m_block_size k_block_size n_block_size : Nat) :
    ∃ alg, (matmul_triton1_surface X Y Z m_size k_size n_size m_block_size
      k_block_size n_block_size).toAlgorithm? = Except.ok alg := by
  simp [matmul_triton1_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- Proof-oriented output-store slice of `matmul_triton1.py`'s `matmul_kernel`.

The full kernel maps a linear program id to an M/N tile, computes `z` in a dot loop, and stores the tile. This slice starts from a precomputed `Acc` tile
and proves the final 2D writeback into `C`. -/
def matmul_output_store_slice
    (C Acc : RegionName)
    (stride_cm stride_cn stride_accm stride_accn BLOCK_SIZE_M BLOCK_SIZE_N : Nat) :
    ComputeKernel := triton {
  pid_m = tl.program_id(0)
  pid_n = tl.program_id(1)
  offs_cm = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_cn = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  acc = tl.load(Acc + $(stride_accm) * offs_cm[:, None] +
    $(stride_accn) * offs_cn[None, :])
  tl.store(C + $(stride_cm) * offs_cm[:, None] + $(stride_cn) * offs_cn[None, :],
    acc)
}

def rowIndex (s : BlockState) (BLOCK_SIZE_M : Nat) (i : Fin BLOCK_SIZE_M) : Nat :=
  s.pids 0 * BLOCK_SIZE_M + i.val

def colIndex (s : BlockState) (BLOCK_SIZE_N : Nat) (j : Fin BLOCK_SIZE_N) : Nat :=
  s.pids 1 * BLOCK_SIZE_N + j.val

def cOffset
    (s : BlockState) (stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N : Nat)
    (idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N]) : Nat :=
  stride_cm * rowIndex s BLOCK_SIZE_M idx.1 +
    stride_cn * colIndex s BLOCK_SIZE_N idx.2.1

def accOffset
    (s : BlockState) (stride_accm stride_accn BLOCK_SIZE_M BLOCK_SIZE_N : Nat)
    (idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N]) : Nat :=
  stride_accm * rowIndex s BLOCK_SIZE_M idx.1 +
    stride_accn * colIndex s BLOCK_SIZE_N idx.2.1

private noncomputable def preStoreState
    (s : BlockState) (Acc : RegionName)
    (stride_accm stride_accn BLOCK_SIZE_M BLOCK_SIZE_N : Nat) : BlockState :=
  s.setReg "pid_m" TileDType.nat [] (Tile.scalar (s.pids 0))
    |>.setReg "pid_n" TileDType.nat [] (Tile.scalar (s.pids 1))
    |>.setReg "offs_cm" TileDType.nat [BLOCK_SIZE_M]
      (Tile.vec fun i => s.pids 0 * BLOCK_SIZE_M + i.val)
    |>.setReg "offs_cn" TileDType.nat [BLOCK_SIZE_N]
      (Tile.vec fun i => s.pids 1 * BLOCK_SIZE_N + i.val)
    |>.setReg "acc" TileDType.real [BLOCK_SIZE_M, BLOCK_SIZE_N]
      { data := fun idx =>
        some (s.readMem Acc
          (stride_accm * (s.pids 0 * BLOCK_SIZE_M + idx.1.val) +
            stride_accn * (s.pids 1 * BLOCK_SIZE_N + idx.2.1.val))) }

/-- Algorithm-layer correctness for the 2D output tile store. -/
theorem matmul_output_store_slice_correct
    (C Acc : RegionName)
    (stride_cm stride_cn stride_accm stride_accn BLOCK_SIZE_M BLOCK_SIZE_N : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (cOffset s stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N))
    (hExec : exec (matmul_output_store_slice C Acc stride_cm stride_cn
        stride_accm stride_accn BLOCK_SIZE_M BLOCK_SIZE_N) s = some s') :
    ∀ idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N],
      s'.readMem C (cOffset s stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N idx) =
        s.readMem Acc
          (accOffset s stride_accm stride_accn BLOCK_SIZE_M BLOCK_SIZE_N idx) := by
  intro idx
  simp [exec, matmul_output_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.expandDim, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul, TileShape.dropInsertedIndex] at hExec
  rw [← hExec]
  let offsetFn : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N] → Nat :=
    fun idx =>
      stride_cm * (s.pids 0 * BLOCK_SIZE_M + idx.1.val) +
        stride_cn * (s.pids 1 * BLOCK_SIZE_N + idx.2.1.val)
  let valueFn : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N] → ℝ :=
    fun idx =>
      s.readMem Acc
        (stride_accm * (s.pids 0 * BLOCK_SIZE_M + idx.1.val) +
          stride_accn * (s.pids 1 * BLOCK_SIZE_N + idx.2.1.val))
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, cOffset, rowIndex, colIndex] using hOutInj
  have hscatter := BlockState.scatter_readback_nd
    (region := C)
    (s := preStoreState s Acc stride_accm stride_accn BLOCK_SIZE_M BLOCK_SIZE_N)
    (offsetFn := offsetFn) (valueFn := valueFn) hOffsetInj idx
  simpa [offsetFn, valueFn, cOffset, accOffset, rowIndex, colIndex,
    preStoreState, TileShape.dropInsertedIndex] using hscatter

/-- Compute-facing correctness for the 2D output tile store. -/
theorem matmul_output_store_slice_compute_correct
    (C Acc : RegionName)
    (stride_cm stride_cn stride_accm stride_accn BLOCK_SIZE_M BLOCK_SIZE_N : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (cOffset s stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N)) :
    ComputeCorrect.Realizes
      (kernel := matmul_output_store_slice C Acc stride_cm stride_cn
        stride_accm stride_accn BLOCK_SIZE_M BLOCK_SIZE_N)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N] =>
        some (C, cOffset s stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N idx))
      (expected := fun idx =>
        s.readMem Acc
          (accOffset s stride_accm stride_accn BLOCK_SIZE_M BLOCK_SIZE_N idx)) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [matmul_output_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  exact matmul_output_store_slice_correct C Acc stride_cm stride_cn
    stride_accm stride_accn BLOCK_SIZE_M BLOCK_SIZE_N s s' hOutInj hExec idx

/-- Python test-shape output offsets are injective for the single
`16 × 16` matmul tile. -/
theorem matmul_triton1_python_test_shape_offset_injective
    (s : BlockState) :
    Function.Injective (cOffset s 16 1 16 16) := by
  intro a b h
  simp [cOffset, rowIndex, colIndex] at h
  ext <;> omega

/-- Python test-shape full surface lowering for `x.shape = y.shape =
(16, 16)`, with block sizes equal to the matrix dimensions. -/
theorem matmul_triton1_python_test_shape_surface_toAlgorithm_supported
    (X Y Z : RegionName) :
    ∃ alg, (matmul_triton1_surface X Y Z 16 16 16 16 16 16).toAlgorithm? =
        Except.ok alg := by
  exact matmul_triton1_surface_toAlgorithm_supported X Y Z 16 16 16 16 16 16

/-- Public Python test-shape coverage summary: the full matmul surface lowers,
and the final `16 × 16` output tile store realizes the checked result tensor. -/
theorem matmul_triton1_python_test_shape_output_summary
    (X Y Z Acc : RegionName) (s : BlockState) :
    (∃ alg, (matmul_triton1_surface X Y Z 16 16 16 16 16 16).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := matmul_output_store_slice Z Acc 16 1 16 1 16 16)
      (initialState := s)
      (write := fun idx : TileIndex [16, 16] =>
        some (Z, cOffset s 16 1 16 16 idx))
      (expected := fun idx =>
        s.readMem Acc (accOffset s 16 1 16 16 idx))) := by
  constructor
  · exact matmul_triton1_python_test_shape_surface_toAlgorithm_supported X Y Z
  · exact matmul_output_store_slice_compute_correct Z Acc 16 1 16 1 16 16 s
      (matmul_triton1_python_test_shape_offset_injective s)

end VeriTile.Bench.TritonBenchG.MatmulTriton1
