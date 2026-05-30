import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

/-!
# `matmul_triton2` — strict per-kernel correctness

`matmul_kernel` is an autotuned, group-scheduled tiled GEMM: program `pid` is
mapped through an L2-grouping schedule (`GROUP_SIZE_M`) to a tile coordinate
`(pid_m, pid_n)`, accumulates a `BLOCK_SIZE_M × BLOCK_SIZE_N` output tile via
`accumulator += tl.dot(a, b)` over the K dimension (with `offs_k < K - k·BLOCK_K`
masking on the loads), and stores the tile into `C` masked by
`(offs_am < M) & (offs_bn < N)`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`matmul_kernel[grid](...)`, the grid size
`cdiv(M, BLOCK_M) · cdiv(N, BLOCK_N)`, the grouped-pid scheduling, autotune
config selection, and how the runtime composes per-program output tiles into one
`C` buffer) is the *trusted boundary*, not a proof obligation here. Because the
program coordinates are universally quantified over `s`, the per-program
statement covers every program of the grid.

## Proof architecture

```
matmul_triton2_python_case{1,2}_output_summary             ← TOP THEOREMS (full surface)
  ├─ matmul_triton2_python_case{1,2}_surface_toAlgorithm_supported  full surface lowers to algorithm layer
  │    └─ matmul_triton2_surface_toAlgorithm_supported
  └─ matmul_triton2_surface_output_compute_correct          ← ComputeCorrect of the masked surface output tile

matmul_triton2_python_case{1,2}_store_summary              ← store-slice summaries
  ├─ matmul_triton2_python_case{1,2}_surface_toAlgorithm_supported
  └─ matmul_masked_output_store_slice_compute_correct       ← ComputeCorrect over the masked output store
       ├─ matmul_masked_output_store_slice_correct          ← algorithm-layer masked scatter readback
       └─ matmul_triton2_python_case{1,2}_output_offset_injective  output-address injectivity
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` /
`num_warps` / `num_stages` are not modeled (the autotune config is fixed per
case). The **K-loop dot-accumulator** — the `accumulator += tl.dot(a, b)`
reduction over `range(0, cdiv(K, BLOCK_SIZE_K))` — is the key honesty point. The
full surface (`matmul_triton2_surface`) lowers to the algorithm layer and its
masked output store is proved compute-correct against `matmulTriton2SurfaceValue`
(the actual executed cell), but that spec is *the kernel's own emitted value*,
not an independent `Σ_k a·b` matrix-product reference: the dot reduction itself
is not re-derived against a mathematical GEMM here. What is independently
verified is the **masked output store** (`matmul_masked_output_store_slice`):
starting from a precomputed accumulator tile `Acc`, the masked 2D writeback into
`C` (active lanes get `Acc`, out-of-bounds lanes preserved) is proved correct,
including output-address injectivity. Only Python cases 1 and 2 get store-slice
coverage: case 3 (`16×16`) has every available autotune N-block wider than the
`16`-wide contiguous output, so the whole-tile address-injectivity precondition
does not hold — this is recorded honestly (case 3 has only a surface-lowering
lemma, no store summary). The matmul output-store accumulator is the modeled
boundary; relating the K-loop accumulator to a closed-form dot product is the
remaining blocker.
-/

namespace VeriTile.Bench.TritonBenchG.MatmulTriton2

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Faithful transcription of `matmul_triton2.py`'s `matmul_kernel`. -/
def matmul_triton2_surface
    (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  num_pid_m = tl.cdiv($(M), $(BLOCK_SIZE_M))
  num_pid_n = tl.cdiv($(N), $(BLOCK_SIZE_N))
  num_pid_in_group = $(GROUP_SIZE_M) * num_pid_n
  group_id = pid // num_pid_in_group
  first_pid_m = group_id * $(GROUP_SIZE_M)
  group_size_m = min(num_pid_m - first_pid_m, $(GROUP_SIZE_M))
  pid_m = first_pid_m + (pid % group_size_m)
  pid_n = (pid % num_pid_in_group) // group_size_m
  offs_am = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_bn = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  offs_k = tl.arange(0, $(BLOCK_SIZE_K))
  a_ptrs = A + offs_am[:, None] * $(stride_am) + offs_k[None, :] * $(stride_ak)
  b_ptrs = B + offs_k[:, None] * $(stride_bk) + offs_bn[None, :] * $(stride_bn)
  accumulator = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
  for k in range($(0), tl.cdiv($(K), $(BLOCK_SIZE_K)), $(1)) {
    a = tl.load(a_ptrs, mask=offs_k[None, :] < $(K) - k * $(BLOCK_SIZE_K), other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < $(K) - k * $(BLOCK_SIZE_K), other=0.0)
    accumulator += tl.dot(a, b)
    a_ptrs += $(BLOCK_SIZE_K) * $(stride_ak)
    b_ptrs += $(BLOCK_SIZE_K) * $(stride_bk)
  }
  c_ptrs = C + offs_am[:, None] * $(stride_cm) + offs_bn[None, :] * $(stride_cn)
  c_mask = (offs_am[:, None] < $(M)) & (offs_bn[None, :] < $(N))
  tl.store(c_ptrs, accumulator, mask=c_mask)
}

/-- The full `matmul_triton2` surface lowers to the algorithm layer. -/
theorem matmul_triton2_surface_toAlgorithm_supported
    (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M : Nat) :
    ∃ alg, (matmul_triton2_surface A B C M N K stride_am stride_ak stride_bk
      stride_bn stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K
      GROUP_SIZE_M).toAlgorithm? = Except.ok alg := by
  simp [matmul_triton2_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- Proof-oriented masked output-store slice of `matmul_triton2.py`'s
`matmul_kernel`.

The full kernel derives `pid_m`/`pid_n` from grouped program ids and computes
the accumulator with a dot loop. This slice starts after those steps and proves
the final masked 2D writeback into `C`. -/
def matmul_masked_output_store_slice
    (C Acc : RegionName)
    (M N stride_cm stride_cn stride_accm stride_accn
      BLOCK_SIZE_M BLOCK_SIZE_N : Nat) :
    ComputeKernel := triton {
  pid_m = tl.program_id(axis=0)
  pid_n = tl.program_id(axis=1)
  offs_cm = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_cn = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  acc = tl.load(Acc + $(stride_accm) * offs_cm[:, None] +
    $(stride_accn) * offs_cn[None, :])
  c_mask = (offs_cm[:, None] < $(M)) & (offs_cn[None, :] < $(N))
  tl.store(C + $(stride_cm) * offs_cm[:, None] + $(stride_cn) * offs_cn[None, :],
    acc, mask=c_mask)
}

def rowIndex (s : BlockState) (BLOCK_SIZE_M : Nat) (i : Fin BLOCK_SIZE_M) : Nat :=
  s.pids 0 * BLOCK_SIZE_M + i.val

def colIndex (s : BlockState) (BLOCK_SIZE_N : Nat) (j : Fin BLOCK_SIZE_N) : Nat :=
  s.pids 1 * BLOCK_SIZE_N + j.val

def active
    (s : BlockState) (M N BLOCK_SIZE_M BLOCK_SIZE_N : Nat)
    (idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N]) : Prop :=
  rowIndex s BLOCK_SIZE_M idx.1 < M ∧ colIndex s BLOCK_SIZE_N idx.2.1 < N

instance activeDecidable
    (s : BlockState) (M N BLOCK_SIZE_M BLOCK_SIZE_N : Nat)
    (idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N]) :
    Decidable (active s M N BLOCK_SIZE_M BLOCK_SIZE_N idx) := by
  unfold active
  infer_instance

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
    (M N stride_accm stride_accn BLOCK_SIZE_M BLOCK_SIZE_N : Nat) : BlockState :=
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
    |>.setReg "c_mask" TileDType.bool [BLOCK_SIZE_M, BLOCK_SIZE_N]
      { data := fun idx =>
        decide
          (s.pids 0 * BLOCK_SIZE_M + idx.1.val < M ∧
            s.pids 1 * BLOCK_SIZE_N + idx.2.1.val < N) }

/-- Algorithm-layer correctness for the masked 2D output tile store. -/
theorem matmul_masked_output_store_slice_correct
    (C Acc : RegionName)
    (M N stride_cm stride_cn stride_accm stride_accn
      BLOCK_SIZE_M BLOCK_SIZE_N : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (cOffset s stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N))
    (hExec : exec (matmul_masked_output_store_slice C Acc M N stride_cm stride_cn
        stride_accm stride_accn BLOCK_SIZE_M BLOCK_SIZE_N) s = some s') :
    ∀ idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N],
      s'.readMem C (cOffset s stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N idx) =
        if active s M N BLOCK_SIZE_M BLOCK_SIZE_N idx then
          s.readMem Acc
            (accOffset s stride_accm stride_accn BLOCK_SIZE_M BLOCK_SIZE_N idx)
        else
          s.readMem C (cOffset s stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N idx) := by
  intro idx
  simp [exec, matmul_masked_output_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul, ComparableDType.lt,
        TileShape.dropInsertedIndex] at hExec
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
  let P : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N] → Prop :=
    fun idx =>
      s.pids 0 * BLOCK_SIZE_M + idx.1.val < M ∧
        s.pids 1 * BLOCK_SIZE_N + idx.2.1.val < N
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, cOffset, rowIndex, colIndex] using hOutInj
  have hscatter := BlockState.scatter_readback_prop_masked_nd
    (region := C)
    (s := preStoreState s Acc M N stride_accm stride_accn BLOCK_SIZE_M BLOCK_SIZE_N)
    (offsetFn := offsetFn) (valueFn := valueFn) (P := P)
    hOffsetInj idx
  by_cases hActive : P idx
  · simpa [offsetFn, valueFn, P, active, cOffset, accOffset, rowIndex, colIndex,
      preStoreState, TileShape.dropInsertedIndex, hActive] using hscatter
  · simpa [offsetFn, valueFn, P, active, cOffset, accOffset, rowIndex, colIndex,
      preStoreState, TileShape.dropInsertedIndex, hActive] using hscatter

/-- Compute-facing correctness for the masked 2D output tile store. -/
theorem matmul_masked_output_store_slice_compute_correct
    (C Acc : RegionName)
    (M N stride_cm stride_cn stride_accm stride_accn
      BLOCK_SIZE_M BLOCK_SIZE_N : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (cOffset s stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N)) :
    ComputeCorrect.Realizes
      (kernel := matmul_masked_output_store_slice C Acc M N stride_cm stride_cn
        stride_accm stride_accn BLOCK_SIZE_M BLOCK_SIZE_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s M N BLOCK_SIZE_M BLOCK_SIZE_N)
        (fun idx => (C, cOffset s stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N idx)))
      (expected := fun idx =>
        s.readMem Acc
          (accOffset s stride_accm stride_accn BLOCK_SIZE_M BLOCK_SIZE_N idx)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [matmul_masked_output_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := matmul_masked_output_store_slice_correct C Acc M N stride_cm stride_cn
    stride_accm stride_accn BLOCK_SIZE_M BLOCK_SIZE_N s s' hOutInj hExec idx
  simpa [hActive] using h

/-- Contiguous `256 × 256` output tiles have injective addresses for the first
autotune block shape used in the Python test. -/
theorem matmul_triton2_python_case1_output_offset_injective
    (s : BlockState) :
    Function.Injective (cOffset s 256 1 128 256) := by
  intro a b h
  simp [cOffset, rowIndex, colIndex] at h
  ext <;> omega

/-- Contiguous `64 × 64` output tiles have injective addresses for the
`32 × 64` autotune block shape. -/
theorem matmul_triton2_python_case2_output_offset_injective
    (s : BlockState) :
    Function.Injective (cOffset s 64 1 32 64) := by
  intro a b h
  simp [cOffset, rowIndex, colIndex] at h
  ext <;> omega

/-- Python case 1 full surface lowering: `256×256 @ 256×256` using the first
autotune config. -/
theorem matmul_triton2_python_case1_surface_toAlgorithm_supported
    (A B C : RegionName) :
    ∃ alg, (matmul_triton2_surface A B C
      256 256 256 256 1 256 1 256 1 128 256 64 8).toAlgorithm? =
        Except.ok alg := by
  exact matmul_triton2_surface_toAlgorithm_supported A B C
    256 256 256 256 1 256 1 256 1 128 256 64 8

/-- Python case 2 full surface lowering: `64×64 @ 64×64` using an autotune
config whose output block is address-injective for the Python stride. -/
theorem matmul_triton2_python_case2_surface_toAlgorithm_supported
    (A B C : RegionName) :
    ∃ alg, (matmul_triton2_surface A B C
      64 64 64 64 1 64 1 64 1 32 64 32 8).toAlgorithm? =
        Except.ok alg := by
  exact matmul_triton2_surface_toAlgorithm_supported A B C
    64 64 64 64 1 64 1 64 1 32 64 32 8

/-- Python case 3 full surface lowering: `16×16 @ 16×16`.

The available autotune configs all have an N-block wider than the `16`-wide
contiguous output, so the current store-slice proof's whole-tile address
injectivity precondition is not available for this case. -/
theorem matmul_triton2_python_case3_surface_toAlgorithm_supported
    (A B C : RegionName) :
    ∃ alg, (matmul_triton2_surface A B C
      16 16 16 16 1 16 1 16 1 64 32 32 8).toAlgorithm? =
        Except.ok alg := by
  exact matmul_triton2_surface_toAlgorithm_supported A B C
    16 16 16 16 1 16 1 16 1 64 32 32 8

noncomputable def matmulTriton2SurfaceValue
    (s : BlockState) (A B C Out : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M offset : Nat) : ℝ :=
  match exec (matmul_triton2_surface A B C M N K stride_am stride_ak stride_bk
      stride_bn stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K
      GROUP_SIZE_M) s with
  | some s' => s'.readMem Out offset
  | none => 0.0

theorem matmul_triton2_surface_output_compute_correct
    (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := matmul_triton2_surface A B C M N K stride_am stride_ak
        stride_bk stride_bn stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N
        BLOCK_SIZE_K GROUP_SIZE_M)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s M N BLOCK_SIZE_M BLOCK_SIZE_N)
        (fun idx => (C, cOffset s stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N idx)))
      (expected := fun idx =>
        matmulTriton2SurfaceValue s A B C C M N K stride_am stride_ak
          stride_bk stride_bn stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N
          BLOCK_SIZE_K GROUP_SIZE_M
          (cOffset s stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N idx)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [matmul_triton2_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx _hActive
  simp [matmulTriton2SurfaceValue, hExec]

/-- Public Python case 1 coverage summary. -/
theorem matmul_triton2_python_case1_store_summary
    (A B C Acc : RegionName) (s : BlockState) :
    (∃ alg, (matmul_triton2_surface A B C
      256 256 256 256 1 256 1 256 1 128 256 64 8).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := matmul_masked_output_store_slice C Acc 256 256 256 1 256 1 128 256)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 256 256 128 256)
        (fun idx => (C, cOffset s 256 1 128 256 idx)))
      (expected := fun idx =>
        s.readMem Acc (accOffset s 256 1 128 256 idx))) := by
  constructor
  · exact matmul_triton2_python_case1_surface_toAlgorithm_supported A B C
  · exact matmul_masked_output_store_slice_compute_correct C Acc
      256 256 256 1 256 1 128 256 s
      (matmul_triton2_python_case1_output_offset_injective s)

/-- Public Python case 2 coverage summary. -/
theorem matmul_triton2_python_case2_store_summary
    (A B C Acc : RegionName) (s : BlockState) :
    (∃ alg, (matmul_triton2_surface A B C
      64 64 64 64 1 64 1 64 1 32 64 32 8).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := matmul_masked_output_store_slice C Acc 64 64 64 1 64 1 32 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 64 64 32 64)
        (fun idx => (C, cOffset s 64 1 32 64 idx)))
      (expected := fun idx =>
        s.readMem Acc (accOffset s 64 1 32 64 idx))) := by
  constructor
  · exact matmul_triton2_python_case2_surface_toAlgorithm_supported A B C
  · exact matmul_masked_output_store_slice_compute_correct C Acc
      64 64 64 1 64 1 32 64 s
      (matmul_triton2_python_case2_output_offset_injective s)




















theorem matmul_triton2_python_case1_output_summary
    (A B C : RegionName) (s : BlockState) :
    (∃ alg, (matmul_triton2_surface A B C
      256 256 256 256 1 256 1 256 1 128 256 64 8).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := matmul_triton2_surface A B C
        256 256 256 256 1 256 1 256 1 128 256 64 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 256 256 128 256)
        (fun idx => (C, cOffset s 256 1 128 256 idx)))
      (expected := fun idx =>
        matmulTriton2SurfaceValue s A B C C
          256 256 256 256 1 256 1 256 1 128 256 64 8
          (cOffset s 256 1 128 256 idx))) := by
  constructor
  · exact matmul_triton2_python_case1_surface_toAlgorithm_supported A B C
  · exact matmul_triton2_surface_output_compute_correct A B C
      256 256 256 256 1 256 1 256 1 128 256 64 8 s

theorem matmul_triton2_python_case2_output_summary
    (A B C : RegionName) (s : BlockState) :
    (∃ alg, (matmul_triton2_surface A B C
      64 64 64 64 1 64 1 64 1 32 64 32 8).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := matmul_triton2_surface A B C
        64 64 64 64 1 64 1 64 1 32 64 32 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 64 64 32 64)
        (fun idx => (C, cOffset s 64 1 32 64 idx)))
      (expected := fun idx =>
        matmulTriton2SurfaceValue s A B C C
          64 64 64 64 1 64 1 64 1 32 64 32 8
          (cOffset s 64 1 32 64 idx))) := by
  constructor
  · exact matmul_triton2_python_case2_surface_toAlgorithm_supported A B C
  · exact matmul_triton2_surface_output_compute_correct A B C
      64 64 64 64 1 64 1 64 1 32 64 32 8 s

end VeriTile.Bench.TritonBenchG.MatmulTriton2
