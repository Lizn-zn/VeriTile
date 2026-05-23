import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.MatmulTma

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Faithful transcription of `matmul_tma.py`'s `matmul_tma_load_store`.

The Python wrapper's transpose cases are represented by the strides passed to
the same kernel, so no separate transpose-specific surface is needed. The TMA
`order` tuple is scheduling metadata; the DSL accepts it at the surface and
erases it into the same block-pointer AST. -/
def matmul_tma_load_store_surface
    (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_M BLOCK_N BLOCK_K : Nat)
    (OUTPUT_F16 : Bool) :
    ComputeKernel := triton {
  a_block_ptr = tl.make_block_ptr(base=A, shape=($(M), $(K)),
    strides=($(stride_am), $(stride_ak)), offsets=($(0), $(0)),
    block_shape=($(BLOCK_M), $(BLOCK_K)), order=(1, 0))
  b_block_ptr = tl.make_block_ptr(base=B, shape=($(K), $(N)),
    strides=($(stride_bk), $(stride_bn)), offsets=($(0), $(0)),
    block_shape=($(BLOCK_K), $(BLOCK_N)), order=(0, 1))
  c_block_ptr = tl.make_block_ptr(base=C, shape=($(M), $(N)),
    strides=($(stride_cm), $(stride_cn)), offsets=($(0), $(0)),
    block_shape=($(BLOCK_M), $(BLOCK_N)), order=(1, 0))
  a = tl.load(a_block_ptr)
  b = tl.load(b_block_ptr)
  c = tl.dot(a, b)
  if OUTPUT_F16 {
    c = (c).to(tl.float16)
  }
  tl.store(c_block_ptr, c)
}

/-- The full TMA load/store matmul surface lowers to the algorithm layer. -/
theorem matmul_tma_load_store_surface_toAlgorithm_supported
    (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_M BLOCK_N BLOCK_K : Nat)
    (OUTPUT_F16 : Bool) :
    ∃ alg, (matmul_tma_load_store_surface A B C M N K stride_am stride_ak
      stride_bk stride_bn stride_cm stride_cn BLOCK_M BLOCK_N BLOCK_K
      OUTPUT_F16).toAlgorithm? = Except.ok alg := by
  simp [matmul_tma_load_store_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- Surface transcription of `matmul_tma.py`'s `matmul_tma_load_store` with
`OUTPUT_F16 = true`. -/
def matmul_tma_load_store_f16_surface
    (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_M BLOCK_N BLOCK_K : Nat) :
    ComputeKernel := triton {
  a_block_ptr = tl.make_block_ptr(base=A, shape=($(M), $(K)),
    strides=($(stride_am), $(stride_ak)), offsets=($(0), $(0)),
    block_shape=($(BLOCK_M), $(BLOCK_K)), order=(1, 0))
  b_block_ptr = tl.make_block_ptr(base=B, shape=($(K), $(N)),
    strides=($(stride_bk), $(stride_bn)), offsets=($(0), $(0)),
    block_shape=($(BLOCK_K), $(BLOCK_N)), order=(0, 1))
  c_block_ptr = tl.make_block_ptr(base=C, shape=($(M), $(N)),
    strides=($(stride_cm), $(stride_cn)), offsets=($(0), $(0)),
    block_shape=($(BLOCK_M), $(BLOCK_N)), order=(1, 0))
  a = tl.load(a_block_ptr)
  b = tl.load(b_block_ptr)
  c = (tl.dot(a, b)).to(tl.float16)
  tl.store(c_block_ptr, c)
}

/-- The fp16-specialized TMA load/store surface lowers to the algorithm layer. -/
theorem matmul_tma_load_store_f16_surface_toAlgorithm_supported
    (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_M BLOCK_N BLOCK_K : Nat) :
    ∃ alg, (matmul_tma_load_store_f16_surface A B C M N K stride_am stride_ak
      stride_bk stride_bn stride_cm stride_cn BLOCK_M BLOCK_N BLOCK_K).toAlgorithm?
        = Except.ok alg := by
  simp [matmul_tma_load_store_f16_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

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
    (acc).to(C.dtype.element_ty))
}

/-- Proof-oriented fp16 output-store slice for `OUTPUT_F16 = true`. -/
def matmul_output_store_f16_slice
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
    (acc).to(tl.float16))
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

private theorem foldl_writeMemTyped_fp16_preserves {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → TileCarrier TileDType.fp16)
    (o : Nat) (l : List α) :
    ∀ s : BlockState,
      (∀ k ∈ l, offsetFn k ≠ o) →
        ((l.foldl
          (fun acc k => acc.writeMemTyped .fp16 region (offsetFn k) (valueFn k))
          s).mem region o) = s.mem region o := by
  induction l with
  | nil =>
      intro s _h
      rfl
  | cons hd tl ih =>
      intro s h
      rw [List.foldl_cons]
      have htl : ∀ k ∈ tl, offsetFn k ≠ o :=
        fun k hk => h k (List.mem_cons_of_mem hd hk)
      have hhd : offsetFn hd ≠ o := h hd (List.mem_cons_self)
      rw [ih _ htl]
      unfold BlockState.writeMemTyped BlockState.writeMemAs
      change
        (if region = region ∧ o = offsetFn hd then
          MemCell.of .fp16 (FloatDType.fp16.ofReal (FloatDType.fp16.storeValue (valueFn hd)))
        else
          s.mem region o) = s.mem region o
      rw [if_neg (by
        intro hsame
        exact hhd hsame.2.symm)]

private theorem scatter_memcell_fp16_nd {region : RegionName} {shape : TileShape}
    (s : BlockState) (offsetFn : TileIndex shape → Nat)
    (valueFn : TileIndex shape → TileCarrier TileDType.fp16)
    (h_inj : Function.Injective offsetFn) (i : TileIndex shape) :
    ((TileShape.allIndices shape).foldl
       (fun acc k => acc.writeMemTyped .fp16 region (offsetFn k) (valueFn k))
       s).mem region (offsetFn i)
    = MemCell.of .fp16
        (FloatDType.fp16.ofReal (FloatDType.fp16.storeValue (valueFn i))) := by
  let l := TileShape.allIndices shape
  obtain ⟨l₁, l₂, hl⟩ := List.append_of_mem (TileShape.mem_allIndices shape i)
  have h_nodup := TileShape.allIndices_nodup shape
  change ((l.foldl
       (fun acc k => acc.writeMemTyped .fp16 region (offsetFn k) (valueFn k))
       s).mem region (offsetFn i))
    = MemCell.of .fp16
        (FloatDType.fp16.ofReal (FloatDType.fp16.storeValue (valueFn i)))
  rw [hl] at h_nodup
  rw [List.nodup_append, List.nodup_cons] at h_nodup
  obtain ⟨_, ⟨hi_notin_l2, _⟩, _⟩ := h_nodup
  have hl' : l = l₁ ++ i :: l₂ := by
    simpa [l] using hl
  rw [hl', List.foldl_append, List.foldl_cons]
  have h_l2_not_in : ∀ k ∈ l₂, offsetFn k ≠ offsetFn i := by
    intro k hk heq
    have hki : k = i := h_inj heq
    subst hki
    exact hi_notin_l2 hk
  rw [foldl_writeMemTyped_fp16_preserves offsetFn valueFn (offsetFn i) l₂ _ h_l2_not_in]
  unfold BlockState.writeMemTyped BlockState.writeMemAs
  change
    (if region = region ∧ offsetFn i = offsetFn i then
      MemCell.of .fp16 (FloatDType.fp16.ofReal (FloatDType.fp16.storeValue (valueFn i)))
    else
      (List.foldl
        (fun acc k => acc.writeMemTyped .fp16 region (offsetFn k) (valueFn k))
        s l₁).mem region (offsetFn i))
      =
      MemCell.of .fp16 (FloatDType.fp16.ofReal (FloatDType.fp16.storeValue (valueFn i)))
  rw [if_pos ⟨rfl, rfl⟩]

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

/-- Algorithm-layer correctness for the fp16 2D output tile store. -/
theorem matmul_output_store_f16_slice_correct
    (C Acc : RegionName)
    (stride_cm stride_cn stride_accm stride_accn BLOCK_SIZE_M BLOCK_SIZE_N : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (cOffset s stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N))
    (hExec : exec (matmul_output_store_f16_slice C Acc stride_cm stride_cn
        stride_accm stride_accn BLOCK_SIZE_M BLOCK_SIZE_N) s = some s') :
    ∀ idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N],
      s'.mem C (cOffset s stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N idx) =
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (s.readMem Acc
              (accOffset s stride_accm stride_accn BLOCK_SIZE_M BLOCK_SIZE_N idx)))) := by
  intro idx
  simp [exec, matmul_output_store_f16_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.expandDim, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul, TileShape.dropInsertedIndex] at hExec
  rw [← hExec]
  let offsetFn : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N] → Nat :=
    fun idx =>
      stride_cm * (s.pids 0 * BLOCK_SIZE_M + idx.1.val) +
        stride_cn * (s.pids 1 * BLOCK_SIZE_N + idx.2.1.val)
  let valueFn : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N] → TileCarrier TileDType.fp16 :=
    fun idx =>
      FloatDType.real.cast FloatDType.fp16
        (some (s.readMem Acc
          (stride_accm * (s.pids 0 * BLOCK_SIZE_M + idx.1.val) +
            stride_accn * (s.pids 1 * BLOCK_SIZE_N + idx.2.1.val))))
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, cOffset, rowIndex, colIndex] using hOutInj
  have hscatter := scatter_memcell_fp16_nd
    (region := C)
    (s := preStoreState s Acc stride_accm stride_accn BLOCK_SIZE_M BLOCK_SIZE_N)
    (offsetFn := offsetFn) (valueFn := valueFn) hOffsetInj idx
  simpa [offsetFn, valueFn, cOffset, accOffset, rowIndex, colIndex,
    preStoreState, TileShape.dropInsertedIndex] using hscatter

/-- Compute-facing correctness for the fp16 2D output tile store. -/
theorem matmul_output_store_f16_slice_compute_correct
    (C Acc : RegionName)
    (stride_cm stride_cn stride_accm stride_accn BLOCK_SIZE_M BLOCK_SIZE_N : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (cOffset s stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N)) :
    ComputeCorrect.Realizes
      (kernel := matmul_output_store_f16_slice C Acc stride_cm stride_cn
        stride_accm stride_accn BLOCK_SIZE_M BLOCK_SIZE_N)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N] =>
        some (C, cOffset s stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N idx))
      (expected := fun idx =>
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (s.readMem Acc
              (accOffset s stride_accm stride_accn BLOCK_SIZE_M BLOCK_SIZE_N idx))))) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [matmul_output_store_f16_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  exact matmul_output_store_f16_slice_correct C Acc stride_cm stride_cn
    stride_accm stride_accn BLOCK_SIZE_M BLOCK_SIZE_N s s' hOutInj hExec idx

/-- Contiguous `128 × 128` output tiles in the Python TMA tests have injective
addresses for the single full-tile launch. -/
theorem matmul_tma_python_test_output_offset_injective
    (s : BlockState) :
    Function.Injective (cOffset s 128 1 128 128) := by
  intro a b h
  simp [cOffset, rowIndex, colIndex] at h
  ext <;> omega

/-- Python case 1: no transposition, float32 output. -/
theorem matmul_tma_python_case1_surface_toAlgorithm_supported
    (A B C : RegionName) :
    ∃ alg, (matmul_tma_load_store_surface A B C
      128 128 128 128 1 128 1 128 1 128 128 128 Bool.false).toAlgorithm? =
        Except.ok alg := by
  exact matmul_tma_load_store_surface_toAlgorithm_supported A B C
    128 128 128 128 1 128 1 128 1 128 128 128 Bool.false

/-- Python case 2: transposed A strides, float32 output. -/
theorem matmul_tma_python_case2_surface_toAlgorithm_supported
    (A B C : RegionName) :
    ∃ alg, (matmul_tma_load_store_surface A B C
      128 128 128 1 128 128 1 128 1 128 128 128 Bool.false).toAlgorithm? =
        Except.ok alg := by
  exact matmul_tma_load_store_surface_toAlgorithm_supported A B C
    128 128 128 1 128 128 1 128 1 128 128 128 Bool.false

/-- Python case 3: transposed B strides, float32 output. -/
theorem matmul_tma_python_case3_surface_toAlgorithm_supported
    (A B C : RegionName) :
    ∃ alg, (matmul_tma_load_store_surface A B C
      128 128 128 128 1 1 128 128 1 128 128 128 Bool.false).toAlgorithm? =
        Except.ok alg := by
  exact matmul_tma_load_store_surface_toAlgorithm_supported A B C
    128 128 128 128 1 1 128 128 1 128 128 128 Bool.false

/-- Python case 4: transposed A and B strides, float32 output. -/
theorem matmul_tma_python_case4_surface_toAlgorithm_supported
    (A B C : RegionName) :
    ∃ alg, (matmul_tma_load_store_surface A B C
      128 128 128 1 128 1 128 128 1 128 128 128 Bool.false).toAlgorithm? =
        Except.ok alg := by
  exact matmul_tma_load_store_surface_toAlgorithm_supported A B C
    128 128 128 1 128 1 128 128 1 128 128 128 Bool.false

/-- Python case 5: no transposition, fp16 output. -/
theorem matmul_tma_python_case5_surface_toAlgorithm_supported
    (A B C : RegionName) :
    ∃ alg, (matmul_tma_load_store_surface A B C
      128 128 128 128 1 128 1 128 1 128 128 128 Bool.true).toAlgorithm? =
        Except.ok alg := by
  exact matmul_tma_load_store_surface_toAlgorithm_supported A B C
    128 128 128 128 1 128 1 128 1 128 128 128 Bool.true

/-- Python case 6: transposed A strides, fp16 output. -/
theorem matmul_tma_python_case6_surface_toAlgorithm_supported
    (A B C : RegionName) :
    ∃ alg, (matmul_tma_load_store_surface A B C
      128 128 128 1 128 128 1 128 1 128 128 128 Bool.true).toAlgorithm? =
        Except.ok alg := by
  exact matmul_tma_load_store_surface_toAlgorithm_supported A B C
    128 128 128 1 128 128 1 128 1 128 128 128 Bool.true

/-- Python case 7: transposed B strides, fp16 output. -/
theorem matmul_tma_python_case7_surface_toAlgorithm_supported
    (A B C : RegionName) :
    ∃ alg, (matmul_tma_load_store_surface A B C
      128 128 128 128 1 1 128 128 1 128 128 128 Bool.true).toAlgorithm? =
        Except.ok alg := by
  exact matmul_tma_load_store_surface_toAlgorithm_supported A B C
    128 128 128 128 1 1 128 128 1 128 128 128 Bool.true

/-- Python case 8: transposed A and B strides, fp16 output. -/
theorem matmul_tma_python_case8_surface_toAlgorithm_supported
    (A B C : RegionName) :
    ∃ alg, (matmul_tma_load_store_surface A B C
      128 128 128 1 128 1 128 128 1 128 128 128 Bool.true).toAlgorithm? =
        Except.ok alg := by
  exact matmul_tma_load_store_surface_toAlgorithm_supported A B C
    128 128 128 1 128 1 128 128 1 128 128 128 Bool.true

/-- Public Python float32-output summary for a concrete TMA test branch. -/
theorem matmul_tma_python_f32_output_summary
    (A B C Acc : RegionName) (s : BlockState)
    (hSurface :
      ∃ alg, (matmul_tma_load_store_surface A B C
        128 128 128 128 1 128 1 128 1 128 128 128 Bool.false).toAlgorithm? =
          Except.ok alg) :
    (∃ alg, (matmul_tma_load_store_surface A B C
      128 128 128 128 1 128 1 128 1 128 128 128 Bool.false).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := matmul_output_store_slice C Acc 128 1 128 1 128 128)
      (initialState := s)
      (write := fun idx : TileIndex [128, 128] =>
        some (C, cOffset s 128 1 128 128 idx))
      (expected := fun idx =>
        s.readMem Acc (accOffset s 128 1 128 128 idx))) := by
  constructor
  · exact hSurface
  · exact matmul_output_store_slice_compute_correct C Acc
      128 1 128 1 128 128 s
      (matmul_tma_python_test_output_offset_injective s)

/-- Public Python float32-output summary for any of the transpose-stride TMA
test branches. The input strides identify the branch; the observed output
layout is always contiguous `128 × 128`. -/
theorem matmul_tma_python_f32_branch_output_summary
    (A B C Acc : RegionName) (s : BlockState)
    (stride_am stride_ak stride_bk stride_bn : Nat)
    (hSurface :
      ∃ alg, (matmul_tma_load_store_surface A B C
        128 128 128 stride_am stride_ak stride_bk stride_bn
        128 1 128 128 128 Bool.false).toAlgorithm? = Except.ok alg) :
    (∃ alg, (matmul_tma_load_store_surface A B C
      128 128 128 stride_am stride_ak stride_bk stride_bn
      128 1 128 128 128 Bool.false).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := matmul_output_store_slice C Acc 128 1 128 1 128 128)
      (initialState := s)
      (write := fun idx : TileIndex [128, 128] =>
        some (C, cOffset s 128 1 128 128 idx))
      (expected := fun idx =>
        s.readMem Acc (accOffset s 128 1 128 128 idx))) := by
  constructor
  · exact hSurface
  · exact matmul_output_store_slice_compute_correct C Acc
      128 1 128 1 128 128 s
      (matmul_tma_python_test_output_offset_injective s)

/-- Public Python fp16-output summary for a concrete TMA test branch. -/
theorem matmul_tma_python_f16_output_summary
    (A B C Acc : RegionName) (s : BlockState)
    (hSurface :
      ∃ alg, (matmul_tma_load_store_surface A B C
        128 128 128 128 1 128 1 128 1 128 128 128 Bool.true).toAlgorithm? =
          Except.ok alg) :
    (∃ alg, (matmul_tma_load_store_surface A B C
      128 128 128 128 1 128 1 128 1 128 128 128 Bool.true).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := matmul_output_store_f16_slice C Acc 128 1 128 1 128 128)
      (initialState := s)
      (write := fun idx : TileIndex [128, 128] =>
        some (C, cOffset s 128 1 128 128 idx))
      (expected := fun idx =>
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (s.readMem Acc (accOffset s 128 1 128 128 idx)))))) := by
  constructor
  · exact hSurface
  · exact matmul_output_store_f16_slice_compute_correct C Acc
      128 1 128 1 128 128 s
      (matmul_tma_python_test_output_offset_injective s)

/-- Public Python fp16-output summary for any of the transpose-stride TMA test
branches. -/
theorem matmul_tma_python_f16_branch_output_summary
    (A B C Acc : RegionName) (s : BlockState)
    (stride_am stride_ak stride_bk stride_bn : Nat)
    (hSurface :
      ∃ alg, (matmul_tma_load_store_surface A B C
        128 128 128 stride_am stride_ak stride_bk stride_bn
        128 1 128 128 128 Bool.true).toAlgorithm? = Except.ok alg) :
    (∃ alg, (matmul_tma_load_store_surface A B C
      128 128 128 stride_am stride_ak stride_bk stride_bn
      128 1 128 128 128 Bool.true).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := matmul_output_store_f16_slice C Acc 128 1 128 1 128 128)
      (initialState := s)
      (write := fun idx : TileIndex [128, 128] =>
        some (C, cOffset s 128 1 128 128 idx))
      (expected := fun idx =>
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (s.readMem Acc (accOffset s 128 1 128 128 idx)))))) := by
  constructor
  · exact hSurface
  · exact matmul_output_store_f16_slice_compute_correct C Acc
      128 1 128 1 128 128 s
      (matmul_tma_python_test_output_offset_injective s)

end VeriTile.Bench.TritonBenchG.MatmulTma
