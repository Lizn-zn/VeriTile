import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.MatmulTma

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Surface transcription of `matmul_tma.py`'s `matmul_tma_load_store` with
`OUTPUT_F16 = false`.

The Python wrapper's transpose cases are represented by the strides passed to
the same kernel, so no separate transpose-specific surface is needed. The TMA
`order` tuple is scheduling metadata and is not represented by the current
block-pointer DSL. -/
def matmul_tma_load_store_surface
    (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_M BLOCK_N BLOCK_K : Nat) :
    ComputeKernel := triton {
  a_block_ptr = tl.make_block_ptr(A, base=$(0), shape=[$(M), $(K)],
    strides=[$(stride_am), $(stride_ak)], offsets=[$(0), $(0)],
    block_shape=[$(BLOCK_M), $(BLOCK_K)])
  b_block_ptr = tl.make_block_ptr(B, base=$(0), shape=[$(K), $(N)],
    strides=[$(stride_bk), $(stride_bn)], offsets=[$(0), $(0)],
    block_shape=[$(BLOCK_K), $(BLOCK_N)])
  c_block_ptr = tl.make_block_ptr(C, base=$(0), shape=[$(M), $(N)],
    strides=[$(stride_cm), $(stride_cn)], offsets=[$(0), $(0)],
    block_shape=[$(BLOCK_M), $(BLOCK_N)])
  a = tl.load(a_block_ptr)
  b = tl.load(b_block_ptr)
  c = (tl.dot(a, b)).to(C.dtype.element_ty)
  tl.store(c_block_ptr, c, boundary_check=([0, 1] : List Nat))
}

/-- Surface transcription of `matmul_tma.py`'s `matmul_tma_load_store` with
`OUTPUT_F16 = true`. -/
def matmul_tma_load_store_f16_surface
    (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_M BLOCK_N BLOCK_K : Nat) :
    ComputeKernel := triton {
  a_block_ptr = tl.make_block_ptr(A, base=$(0), shape=[$(M), $(K)],
    strides=[$(stride_am), $(stride_ak)], offsets=[$(0), $(0)],
    block_shape=[$(BLOCK_M), $(BLOCK_K)])
  b_block_ptr = tl.make_block_ptr(B, base=$(0), shape=[$(K), $(N)],
    strides=[$(stride_bk), $(stride_bn)], offsets=[$(0), $(0)],
    block_shape=[$(BLOCK_K), $(BLOCK_N)])
  c_block_ptr = tl.make_block_ptr(C, base=$(0), shape=[$(M), $(N)],
    strides=[$(stride_cm), $(stride_cn)], offsets=[$(0), $(0)],
    block_shape=[$(BLOCK_M), $(BLOCK_N)])
  a = tl.load(a_block_ptr)
  b = tl.load(b_block_ptr)
  c = (tl.dot(a, b)).to(C.dtype.element_ty)
  tl.store(c_block_ptr, c, boundary_check=([0, 1] : List Nat))
}

/-- Proof-oriented output-store slice of `matmul_tma.py`'s `matmul_kernel`.

The full kernel uses Triton block pointers/TMA loads, computes the dot product, optionally converts the result, and stores the tile. This slice starts from a precomputed `Acc` tile
and proves the final 2D writeback into `C`. -/
def matmul_output_store_slice
    (C Acc : RegionName)
    (stride_cm stride_cn stride_accm stride_accn BLOCK_SIZE_M BLOCK_SIZE_N : Nat) :
    ComputeKernel := triton {
  pid_m = tl.program_id(axis=0)
  pid_n = tl.program_id(axis=1)
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
  simp [exec, matmul_output_store_slice, stepStmts, stepStmt, evalOp,
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

end VeriTile.Bench.TritonBenchG.MatmulTma
