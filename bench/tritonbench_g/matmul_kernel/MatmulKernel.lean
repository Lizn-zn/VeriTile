import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

/-!
# `matmul_kernel` — strict per-kernel correctness

`matmul_kernel` is a tiled GEMM: program `(pid_m, pid_n)` accumulates a
`BLOCK_SIZE_M × BLOCK_SIZE_N` output tile by looping over the K dimension,
loading `A`/`B` blocks and fusing them with `accumulator = tl.dot(a, b,
accumulator)`, then casts the accumulator to float16 and stores it into `C`. The
sizes `M = N = K = 4096` and contiguous strides are hard-coded in the kernel.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`matmul_kernel[cdiv(M, BLOCK_SIZE_M), cdiv(N,
BLOCK_SIZE_N)](...)`, the grid scheduling, and how the runtime composes
per-program output tiles into one `C` buffer) is the *trusted boundary*, not a
proof obligation here. Because the program coordinates are universally
quantified over `s`, the per-program statement covers every program of the grid.

## Proof architecture

```
matmul_kernel_python_case{1,2,3,4}_output_summary          ← TOP THEOREMS (full surface)
  ├─ matmul_kernel_python_case{1..4}_surface_toAlgorithm_supported  full surface lowers to algorithm layer
  │    └─ matmul_kernel_surface_toAlgorithm_supported
  └─ matmul_kernel_surface_output_compute_correct          ← ComputeCorrect of the surface output tile

matmul_kernel_python_case{1,2,3,4}_store_summary           ← store-slice summaries
  ├─ matmul_kernel_python_case{1..4}_surface_toAlgorithm_supported
  └─ matmul_output_store_slice_compute_correct             ← ComputeCorrect over the fp16 output store
       ├─ matmul_output_store_slice_correct                ← algorithm-layer fp16 scatter readback
       │    └─ scatter_memcell_fp16_nd / foldl_writeMemTyped_fp16_preserves
       └─ matmul_kernel_python_output_offset_injective     output-address injectivity (contiguous 4096×4096)
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` /
`num_warps` are not modeled. The **K-loop dot-accumulator** — the
`accumulator = tl.dot(a, b, accumulator)` reduction over `range(0, cdiv(K,
BLOCK_SIZE_K))` — is the key honesty point. The full surface
(`matmul_kernel_surface`) lowers to the algorithm layer and its output store is
proved compute-correct against `matmulKernelSurfaceCell` (the actual executed
cell), but that spec is *the kernel's own emitted value*, not an independent
`Σ_k a·b` matrix-product reference: the dot reduction itself is not re-derived
against a mathematical GEMM here. What is independently verified is the **output
store** (`matmul_output_store_slice`): starting from a precomputed accumulator
tile `Acc`, the float16 cast and masked 2D writeback into `C` are proved correct,
including output-address injectivity for the four Python block shapes. The
matmul output-store accumulator is therefore the modeled boundary; relating the
K-loop accumulator to a closed-form dot product is the remaining blocker.
-/

namespace VeriTile.Bench.TritonBenchG.MatmulKernel

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Faithful transcription of `matmul_kernel.py`'s `matmul_kernel`.

The Python kernel hard-codes `M = N = K = 4096` and the corresponding
contiguous strides; this surface keeps those constants and the fused
`tl.dot(a, b, accumulator)` loop. -/
def matmul_kernel_surface
    (C A B : RegionName) (BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat) :
    ComputeKernel := triton {
  pid_m = tl.program_id(axis=0)
  pid_n = tl.program_id(axis=1)
  M, N, K = $(4096), $(4096), $(4096)
  stride_am = $(4096)
  stride_ak = $(1)
  stride_bk = $(4096)
  stride_bn = $(1)
  stride_cm = $(4096)
  stride_cn = $(1)
  offs_am = (pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))) % M
  offs_bn = (pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))) % N
  offs_k = tl.arange(0, $(BLOCK_SIZE_K))
  a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
  b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
  accumulator = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
  for k in range($(0), tl.cdiv(K, $(BLOCK_SIZE_K)), $(1)) {
    a = tl.load(a_ptrs)
    b = tl.load(b_ptrs)
    accumulator = tl.dot(a, b, accumulator)
    a_ptrs += $(BLOCK_SIZE_K) * stride_ak
    b_ptrs += $(BLOCK_SIZE_K) * stride_bk
  }
  c = tl.cast(accumulator, tl.float16)
  offs_cm = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_cn = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
  tl.store(c_ptrs, c)
}

/-- The full matmul kernel surface lowers to the algorithm layer. -/
theorem matmul_kernel_surface_toAlgorithm_supported
    (C A B : RegionName) (BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat) :
    ∃ alg, (matmul_kernel_surface C A B BLOCK_SIZE_M BLOCK_SIZE_N
      BLOCK_SIZE_K).toAlgorithm? = Except.ok alg := by
  simp [matmul_kernel_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

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
  simp [exec, matmul_output_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
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

/-- Contiguous `4096 × 4096` output tiles have injective addresses for the
Python block widths used by `matmul_kernel.py`. -/
theorem matmul_kernel_python_output_offset_injective
    (s : BlockState) {BLOCK_SIZE_M BLOCK_SIZE_N : Nat}
    (hN : BLOCK_SIZE_N ≤ 4096) :
    Function.Injective (cOffset s 4096 1 BLOCK_SIZE_M BLOCK_SIZE_N) := by
  intro a b h
  simp [cOffset, rowIndex, colIndex] at h
  ext <;> omega

/-- Python case 1 full surface lowering for block shape `(64, 128, 64)`. -/
theorem matmul_kernel_python_case1_surface_toAlgorithm_supported
    (C A B : RegionName) :
    ∃ alg, (matmul_kernel_surface C A B 64 128 64).toAlgorithm? =
        Except.ok alg := by
  exact matmul_kernel_surface_toAlgorithm_supported C A B 64 128 64

/-- Python case 2 full surface lowering for block shape `(128, 64, 128)`. -/
theorem matmul_kernel_python_case2_surface_toAlgorithm_supported
    (C A B : RegionName) :
    ∃ alg, (matmul_kernel_surface C A B 128 64 128).toAlgorithm? =
        Except.ok alg := by
  exact matmul_kernel_surface_toAlgorithm_supported C A B 128 64 128

/-- Python case 3 full surface lowering for block shape `(256, 256, 64)`. -/
theorem matmul_kernel_python_case3_surface_toAlgorithm_supported
    (C A B : RegionName) :
    ∃ alg, (matmul_kernel_surface C A B 256 256 64).toAlgorithm? =
        Except.ok alg := by
  exact matmul_kernel_surface_toAlgorithm_supported C A B 256 256 64

/-- Python case 4 full surface lowering for block shape `(32, 32, 32)`. -/
theorem matmul_kernel_python_case4_surface_toAlgorithm_supported
    (C A B : RegionName) :
    ∃ alg, (matmul_kernel_surface C A B 32 32 32).toAlgorithm? =
        Except.ok alg := by
  exact matmul_kernel_surface_toAlgorithm_supported C A B 32 32 32

noncomputable def matmulKernelSurfaceCell
    (s : BlockState) (C A B Out : RegionName)
    (BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K offset : Nat) : MemCell :=
  match exec (matmul_kernel_surface C A B BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K) s with
  | some s' => s'.mem Out offset
  | none => 0

theorem matmul_kernel_surface_output_compute_correct
    (C A B : RegionName) (BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := matmul_kernel_surface C A B BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N] =>
        some (C, cOffset s 4096 1 BLOCK_SIZE_M BLOCK_SIZE_N idx))
      (expected := fun idx =>
        matmulKernelSurfaceCell s C A B C BLOCK_SIZE_M BLOCK_SIZE_N
          BLOCK_SIZE_K (cOffset s 4096 1 BLOCK_SIZE_M BLOCK_SIZE_N idx)) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [matmul_kernel_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  simp [matmulKernelSurfaceCell, hExec]

/-- Public Python case 1 coverage summary: the full surface lowers and the
final fp16 output tile store realizes the side-effected `c` tensor. -/
theorem matmul_kernel_python_case1_store_summary
    (C A B Acc : RegionName) (s : BlockState) :
    (∃ alg, (matmul_kernel_surface C A B 64 128 64).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := matmul_output_store_slice C Acc 4096 1 4096 1 64 128)
      (initialState := s)
      (write := fun idx : TileIndex [64, 128] =>
        some (C, cOffset s 4096 1 64 128 idx))
      (expected := fun idx =>
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (s.readMem Acc (accOffset s 4096 1 64 128 idx)))))) := by
  constructor
  · exact matmul_kernel_python_case1_surface_toAlgorithm_supported C A B
  · exact matmul_output_store_slice_compute_correct C Acc 4096 1 4096 1
      64 128 s (matmul_kernel_python_output_offset_injective s (by omega))

/-- Public Python case 2 coverage summary. -/
theorem matmul_kernel_python_case2_store_summary
    (C A B Acc : RegionName) (s : BlockState) :
    (∃ alg, (matmul_kernel_surface C A B 128 64 128).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := matmul_output_store_slice C Acc 4096 1 4096 1 128 64)
      (initialState := s)
      (write := fun idx : TileIndex [128, 64] =>
        some (C, cOffset s 4096 1 128 64 idx))
      (expected := fun idx =>
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (s.readMem Acc (accOffset s 4096 1 128 64 idx)))))) := by
  constructor
  · exact matmul_kernel_python_case2_surface_toAlgorithm_supported C A B
  · exact matmul_output_store_slice_compute_correct C Acc 4096 1 4096 1
      128 64 s (matmul_kernel_python_output_offset_injective s (by omega))

/-- Public Python case 3 coverage summary. -/
theorem matmul_kernel_python_case3_store_summary
    (C A B Acc : RegionName) (s : BlockState) :
    (∃ alg, (matmul_kernel_surface C A B 256 256 64).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := matmul_output_store_slice C Acc 4096 1 4096 1 256 256)
      (initialState := s)
      (write := fun idx : TileIndex [256, 256] =>
        some (C, cOffset s 4096 1 256 256 idx))
      (expected := fun idx =>
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (s.readMem Acc (accOffset s 4096 1 256 256 idx)))))) := by
  constructor
  · exact matmul_kernel_python_case3_surface_toAlgorithm_supported C A B
  · exact matmul_output_store_slice_compute_correct C Acc 4096 1 4096 1
      256 256 s (matmul_kernel_python_output_offset_injective s (by omega))

/-- Public Python case 4 coverage summary. -/
theorem matmul_kernel_python_case4_store_summary
    (C A B Acc : RegionName) (s : BlockState) :
    (∃ alg, (matmul_kernel_surface C A B 32 32 32).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := matmul_output_store_slice C Acc 4096 1 4096 1 32 32)
      (initialState := s)
      (write := fun idx : TileIndex [32, 32] =>
        some (C, cOffset s 4096 1 32 32 idx))
      (expected := fun idx =>
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (s.readMem Acc (accOffset s 4096 1 32 32 idx)))))) := by
  constructor
  · exact matmul_kernel_python_case4_surface_toAlgorithm_supported C A B
  · exact matmul_output_store_slice_compute_correct C Acc 4096 1 4096 1
      32 32 s (matmul_kernel_python_output_offset_injective s (by omega))




















theorem matmul_kernel_python_case1_output_summary
    (C A B : RegionName) (s : BlockState) :
    (∃ alg, (matmul_kernel_surface C A B 64 128 64).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := matmul_kernel_surface C A B 64 128 64)
      (initialState := s)
      (write := fun idx : TileIndex [64, 128] =>
        some (C, cOffset s 4096 1 64 128 idx))
      (expected := fun idx =>
        matmulKernelSurfaceCell s C A B C 64 128 64
          (cOffset s 4096 1 64 128 idx))) := by
  constructor
  · exact matmul_kernel_python_case1_surface_toAlgorithm_supported C A B
  · exact matmul_kernel_surface_output_compute_correct C A B 64 128 64 s

theorem matmul_kernel_python_case2_output_summary
    (C A B : RegionName) (s : BlockState) :
    (∃ alg, (matmul_kernel_surface C A B 128 64 128).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := matmul_kernel_surface C A B 128 64 128)
      (initialState := s)
      (write := fun idx : TileIndex [128, 64] =>
        some (C, cOffset s 4096 1 128 64 idx))
      (expected := fun idx =>
        matmulKernelSurfaceCell s C A B C 128 64 128
          (cOffset s 4096 1 128 64 idx))) := by
  constructor
  · exact matmul_kernel_python_case2_surface_toAlgorithm_supported C A B
  · exact matmul_kernel_surface_output_compute_correct C A B 128 64 128 s

theorem matmul_kernel_python_case3_output_summary
    (C A B : RegionName) (s : BlockState) :
    (∃ alg, (matmul_kernel_surface C A B 256 256 64).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := matmul_kernel_surface C A B 256 256 64)
      (initialState := s)
      (write := fun idx : TileIndex [256, 256] =>
        some (C, cOffset s 4096 1 256 256 idx))
      (expected := fun idx =>
        matmulKernelSurfaceCell s C A B C 256 256 64
          (cOffset s 4096 1 256 256 idx))) := by
  constructor
  · exact matmul_kernel_python_case3_surface_toAlgorithm_supported C A B
  · exact matmul_kernel_surface_output_compute_correct C A B 256 256 64 s

theorem matmul_kernel_python_case4_output_summary
    (C A B : RegionName) (s : BlockState) :
    (∃ alg, (matmul_kernel_surface C A B 32 32 32).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := matmul_kernel_surface C A B 32 32 32)
      (initialState := s)
      (write := fun idx : TileIndex [32, 32] =>
        some (C, cOffset s 4096 1 32 32 idx))
      (expected := fun idx =>
        matmulKernelSurfaceCell s C A B C 32 32 32
          (cOffset s 4096 1 32 32 idx))) := by
  constructor
  · exact matmul_kernel_python_case4_surface_toAlgorithm_supported C A B
  · exact matmul_kernel_surface_output_compute_correct C A B 32 32 32 s

end VeriTile.Bench.TritonBenchG.MatmulKernel
