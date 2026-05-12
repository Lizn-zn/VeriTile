import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.MatmulKernel

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Surface transcription of `matmul_kernel.py`'s `matmul_kernel`.

The Python kernel hard-codes `M = N = K = 4096` and the corresponding
contiguous strides; this surface keeps those constants and the fused
`tl.dot(a, b, accumulator)` loop. -/
def matmul_kernel_surface
    (C A B : RegionName) (BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat) :
    ComputeKernel := triton {
  pid_m = tl.program_id(0)
  pid_n = tl.program_id(1)
  offs_am = (pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))) % $(4096)
  offs_bn = (pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))) % $(4096)
  offs_k = tl.arange(0, $(BLOCK_SIZE_K))
  a_ptrs = A + offs_am[:, None] * $(4096) + offs_k[None, :] * $(1)
  b_ptrs = B + offs_k[:, None] * $(4096) + offs_bn[None, :] * $(1)
  accumulator = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
  for k in range($(0), tl.cdiv($(4096), $(BLOCK_SIZE_K)), $(1)) {
    a = tl.load(a_ptrs)
    b = tl.load(b_ptrs)
    accumulator = tl.dot(a, b, accumulator)
    a_ptrs += $(BLOCK_SIZE_K) * $(1)
    b_ptrs += $(BLOCK_SIZE_K) * $(4096)
  }
  c = (accumulator).to(tl.float16)
  offs_cm = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_cn = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  c_ptrs = C + $(4096) * offs_cm[:, None] + $(1) * offs_cn[None, :]
  tl.store(c_ptrs, c)
}

/-- Proof-oriented output-store slice of `matmul_kernel.py`'s `matmul_kernel`.

The full kernel computes `accumulator = dot(A, B)` in a loop, casts it to
float16, and stores the tile. This slice starts from a precomputed `Acc` tile
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
      s'.mem C (cOffset s stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N idx) =
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (s.readMem Acc
              (accOffset s stride_accm stride_accn BLOCK_SIZE_M BLOCK_SIZE_N idx)))) := by
  intro idx
  simp [exec, matmul_output_store_slice, stepStmts, stepStmt, evalOp,
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
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (s.readMem Acc
              (accOffset s stride_accm stride_accn BLOCK_SIZE_M BLOCK_SIZE_N idx))))) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [matmul_output_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  exact matmul_output_store_slice_correct C Acc stride_cm stride_cn
    stride_accm stride_accn BLOCK_SIZE_M BLOCK_SIZE_N s s' hOutInj hExec idx

end VeriTile.Bench.TritonBenchG.MatmulKernel
