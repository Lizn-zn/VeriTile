import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

/-!
# `dequantize_rowwise` — strict per-kernel correctness

`_dequantize_rowwise` dequantizes one row per program: program `pid` loads its
row from `x_ptr` (a `P2`-wide power-of-two arange masked by `arange < BLOCK_SIZE`),
loads the per-row max `state_x[pid]`, and stores `max_val * x * inv_127` to
`output_ptr`, masked by the same row mask.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`_dequantize_rowwise[grid](...)`, the grid `(x.shape[0],)`
of one program per row, the host-side `P2 = 2^ceil(log2(cols))` and `BLOCK_SIZE
= cols` choices, the `inv_127 = 1/127` constant, and the runtime composition of
per-program writes) is the *trusted boundary*. `pid` is universally quantified,
so the per-program statement covers every row of the grid.

## Proof architecture

```
dequantize_rowwise_python_caseN_output_summary    ← TOP THEOREMS (4 cases)
  ├─ (toAlgorithm? = Except.ok _)                  surface lowers to the algorithm layer
  └─ dequantize_rowwise_python_caseN_compute_correct
       └─ dequantize_rowwise_kernel_compute_correct   ComputeCorrect over the masked store
            └─ dequantize_rowwise_kernel_correct       algorithm-layer readback per lane
```

The four cases mirror the Python test shapes. The spec is the elementwise
dequantize `state_x[pid] * x * inv_127` (`dequantizeRowwiseSpec`); the active
mask is `arange < BLOCK_SIZE` (the kernel's `n_elements` argument is unused by
the body and kept as `_n_elements`).

## Modeling boundary

Arithmetic is over `ℝ`, not bit-accurate IEEE float. The quantization semantics
here are the **dequantization direction only**: `x` is the loaded int8 value
taken as a real number and the output is the exact real product `max_val * x *
inv_127`. There is no rounding or int cast on the output side, so this kernel has
no `to(tl.int8)` / `llrint` blocker and the full surface lowers through algorithm
erasure. The output dtype is `float16` upstream, modeled as `ℝ` (post-erasure all
dtypes unify). `@triton.autotune` is not present here.
-/

namespace VeriTile.Bench.TritonBenchG.DequantizeRowwise

open VeriTile.Triton

/-- Faithful 1:1 transcription of `dequantize_rowwise.py`'s `_dequantize_rowwise`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` / `P2: tl.constexpr` → Lean `Nat`
  parameters.
- `n_elements` is kept as `_n_elements`: the upstream Triton kernel accepts it
  but does not use it in the body. -/
def dequantize_rowwise_kernel
    (x_ptr state_x output_ptr : RegionName)
    (inv_127 : ℝ) (_n_elements BLOCK_SIZE P2 : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  arange = tl.arange(0, $(P2))
  offsets = block_start + arange
  row_mask = arange < $(BLOCK_SIZE)
  x = tl.load(x_ptr + offsets, mask=row_mask)
  max_val = tl.load(state_x + pid)
  output = max_val * x * $(inv_127)
  tl.store(output_ptr + offsets, output, mask=row_mask)
}

/-- The Python `_dequantize_rowwise` kernel surface lowers to the algorithm
layer, including the masked row load, per-row scale load, multiply by
`inv_127`, and masked output store. -/
theorem dequantize_rowwise_kernel_surface_toAlgorithm_supported
    (x_ptr state_x output_ptr : RegionName)
    (inv_127 : ℝ) (n_elements BLOCK_SIZE P2 : Nat) :
    ∃ alg, (dequantize_rowwise_kernel x_ptr state_x output_ptr inv_127
      n_elements BLOCK_SIZE P2).toAlgorithm? = Except.ok alg := by
  simp [dequantize_rowwise_kernel]

/-- Exact dequantized value written at active lane `i`. -/
noncomputable def dequantizeRowwiseSpec
    (s : BlockState) (x_ptr state_x : RegionName)
    (BLOCK_SIZE : Nat) (inv_127 : ℝ) (i : Fin P2) : ℝ :=
  s.readMem state_x s.pid *
    s.readMem x_ptr (s.pid * BLOCK_SIZE + i.val) * inv_127

/-- Algorithm-layer correctness for `_dequantize_rowwise`.

The upstream kernel's active mask is `arange < BLOCK_SIZE`; `n_elements` is not
used by the kernel body. -/
theorem dequantize_rowwise_kernel_correct
    (x_ptr state_x output_ptr : RegionName)
    (inv_127 : ℝ) (n_elements BLOCK_SIZE P2 : Nat)
    (s : BlockState) :
    ∀ i : Fin P2,
      let outAddr := s.pid * BLOCK_SIZE + i.val
      (exec (dequantize_rowwise_kernel x_ptr state_x output_ptr inv_127
            n_elements BLOCK_SIZE P2) s).map (·.readMem output_ptr outAddr)
        = some (if i.val < BLOCK_SIZE then
            dequantizeRowwiseSpec s x_ptr state_x BLOCK_SIZE inv_127 i
          else s.readMem output_ptr outAddr) := by
  intro i
  simp [exec, dequantize_rowwise_kernel, stepStmts, stepStmt, evalOp.eq_def,
        Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul,
        ComparableDType.lt]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
        (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
  by_cases hi : i.val < BLOCK_SIZE
  · simp [hi, dequantizeRowwiseSpec, BlockState.pid_eq, mul_assoc]
  · simp [hi]

/-- Compute-facing correctness for `_dequantize_rowwise`. -/
theorem dequantize_rowwise_kernel_compute_correct
    (x_ptr state_x output_ptr : RegionName)
    (inv_127 : ℝ) (n_elements BLOCK_SIZE P2 : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := dequantize_rowwise_kernel x_ptr state_x output_ptr inv_127
        n_elements BLOCK_SIZE P2)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin P2 => i.val < BLOCK_SIZE)
          (fun i => (output_ptr, s.pid * BLOCK_SIZE + i.val)))
      (expected := fun i => dequantizeRowwiseSpec s x_ptr state_x BLOCK_SIZE inv_127 i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have hi := dequantize_rowwise_kernel_correct x_ptr state_x output_ptr inv_127
    n_elements BLOCK_SIZE P2 s i
  rw [hExec] at hi
  simpa [hActive] using Option.some.inj hi

theorem dequantize_rowwise_python_case1_compute_correct
    (x_ptr state_x output_ptr : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := dequantize_rowwise_kernel x_ptr state_x output_ptr (1.0 / 127)
        8 4 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin 4 => i.val < 4)
          (fun i => (output_ptr, s.pid * 4 + i.val)))
      (expected := fun i => dequantizeRowwiseSpec s x_ptr state_x 4 (1.0 / 127) i) := by
  exact dequantize_rowwise_kernel_compute_correct x_ptr state_x output_ptr
    (1.0 / 127) 8 4 4 s

theorem dequantize_rowwise_python_case2_compute_correct
    (x_ptr state_x output_ptr : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := dequantize_rowwise_kernel x_ptr state_x output_ptr (1.0 / 127)
        160 16 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin 16 => i.val < 16)
          (fun i => (output_ptr, s.pid * 16 + i.val)))
      (expected := fun i => dequantizeRowwiseSpec s x_ptr state_x 16 (1.0 / 127) i) := by
  exact dequantize_rowwise_kernel_compute_correct x_ptr state_x output_ptr
    (1.0 / 127) 160 16 16 s

theorem dequantize_rowwise_python_case3_compute_correct
    (x_ptr state_x output_ptr : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := dequantize_rowwise_kernel x_ptr state_x output_ptr (1.0 / 127)
        40 8 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin 8 => i.val < 8)
          (fun i => (output_ptr, s.pid * 8 + i.val)))
      (expected := fun i => dequantizeRowwiseSpec s x_ptr state_x 8 (1.0 / 127) i) := by
  exact dequantize_rowwise_kernel_compute_correct x_ptr state_x output_ptr
    (1.0 / 127) 40 8 8 s

theorem dequantize_rowwise_python_case4_compute_correct
    (x_ptr state_x output_ptr : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := dequantize_rowwise_kernel x_ptr state_x output_ptr (1.0 / 127)
        96 32 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin 32 => i.val < 32)
          (fun i => (output_ptr, s.pid * 32 + i.val)))
      (expected := fun i => dequantizeRowwiseSpec s x_ptr state_x 32 (1.0 / 127) i) := by
  exact dequantize_rowwise_kernel_compute_correct x_ptr state_x output_ptr
    (1.0 / 127) 96 32 32 s

/-- Public Python case-1 summary for `dequantize_rowwise`.

The bundled case has shape `(2, 4)`, so each program row writes the four active
lanes of one output row. -/
theorem dequantize_rowwise_python_case1_output_summary
    (x_ptr state_x output_ptr : RegionName) (s : BlockState) :
    (∃ alg, (dequantize_rowwise_kernel x_ptr state_x output_ptr (1.0 / 127)
      8 4 4).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := dequantize_rowwise_kernel x_ptr state_x output_ptr (1.0 / 127)
        8 4 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin 4 => i.val < 4)
          (fun i => (output_ptr, s.pid * 4 + i.val)))
      (expected := fun i => dequantizeRowwiseSpec s x_ptr state_x 4 (1.0 / 127) i) := by
  constructor
  · exact dequantize_rowwise_kernel_surface_toAlgorithm_supported
      x_ptr state_x output_ptr (1.0 / 127) 8 4 4
  · exact dequantize_rowwise_python_case1_compute_correct x_ptr state_x output_ptr s

/-- Public Python case-2 summary for `dequantize_rowwise`, shape `(10, 16)`. -/
theorem dequantize_rowwise_python_case2_output_summary
    (x_ptr state_x output_ptr : RegionName) (s : BlockState) :
    (∃ alg, (dequantize_rowwise_kernel x_ptr state_x output_ptr (1.0 / 127)
      160 16 16).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := dequantize_rowwise_kernel x_ptr state_x output_ptr (1.0 / 127)
        160 16 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin 16 => i.val < 16)
          (fun i => (output_ptr, s.pid * 16 + i.val)))
      (expected := fun i => dequantizeRowwiseSpec s x_ptr state_x 16 (1.0 / 127) i) := by
  constructor
  · exact dequantize_rowwise_kernel_surface_toAlgorithm_supported
      x_ptr state_x output_ptr (1.0 / 127) 160 16 16
  · exact dequantize_rowwise_python_case2_compute_correct x_ptr state_x output_ptr s

/-- Public Python case-3 summary for `dequantize_rowwise`, shape `(5, 8)`. -/
theorem dequantize_rowwise_python_case3_output_summary
    (x_ptr state_x output_ptr : RegionName) (s : BlockState) :
    (∃ alg, (dequantize_rowwise_kernel x_ptr state_x output_ptr (1.0 / 127)
      40 8 8).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := dequantize_rowwise_kernel x_ptr state_x output_ptr (1.0 / 127)
        40 8 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin 8 => i.val < 8)
          (fun i => (output_ptr, s.pid * 8 + i.val)))
      (expected := fun i => dequantizeRowwiseSpec s x_ptr state_x 8 (1.0 / 127) i) := by
  constructor
  · exact dequantize_rowwise_kernel_surface_toAlgorithm_supported
      x_ptr state_x output_ptr (1.0 / 127) 40 8 8
  · exact dequantize_rowwise_python_case3_compute_correct x_ptr state_x output_ptr s

/-- Public Python case-4 summary for `dequantize_rowwise`, shape `(3, 32)`. -/
theorem dequantize_rowwise_python_case4_output_summary
    (x_ptr state_x output_ptr : RegionName) (s : BlockState) :
    (∃ alg, (dequantize_rowwise_kernel x_ptr state_x output_ptr (1.0 / 127)
      96 32 32).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := dequantize_rowwise_kernel x_ptr state_x output_ptr (1.0 / 127)
        96 32 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin 32 => i.val < 32)
          (fun i => (output_ptr, s.pid * 32 + i.val)))
      (expected := fun i => dequantizeRowwiseSpec s x_ptr state_x 32 (1.0 / 127) i) := by
  constructor
  · exact dequantize_rowwise_kernel_surface_toAlgorithm_supported
      x_ptr state_x output_ptr (1.0 / 127) 96 32 32
  · exact dequantize_rowwise_python_case4_compute_correct x_ptr state_x output_ptr s

end VeriTile.Bench.TritonBenchG.DequantizeRowwise
