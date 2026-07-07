import VeriTile.Core
import VeriTile.Semantics
import VeriTile.Float
import VeriTile.Frontend.Triton.DSL

/-!
# `matrix_reduction` — strict per-kernel correctness

`load_reduce_kernel` loads a `BLOCK_M × BLOCK_N` tile of `x_ptr` through a
block pointer, computes the row-wise maximum `tl.max(x, axis=1)`, and stores the
resulting length-`BLOCK_M` vector to `y_ptr + tl.arange(0, BLOCK_M)`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`load_reduce_kernel[(1,)](...)`, the grid size, the
strides chosen on the host, and how the runtime composes per-program writes into
one buffer) is the *trusted boundary*, not a proof obligation here. Because the
program id `pid` is universally quantified (it enters only via `BlockState`), the
per-program statement covers every program of the grid.

## Proof architecture

```
load_reduce_kernel_output_summary               ← TOP THEOREM
  ├─ (toAlgorithm? = Except.ok _)               surface lowers to the algorithm layer
  └─ load_reduce_kernel_compute_correct         ← ComputeCorrect over the row-vector store
       └─ load_reduce_kernel_correct            ← algorithm-layer readback per row lane
            └─ matrixReduceSpec / matrixReduceInputTile (row-wise `Tile.reduceMax`)
```

The spec `matrixReduceSpec` is the row-wise `Tile.reduceMax` of the input tile
read by the block pointer.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float). The Python test mixes `float16`/`float32` and passes
`check_dtype=False`; post-erasure all dtypes unify to `ℝ`, so the cast is the
identity here. The unused `stride_y` argument is kept as `_stride_y`. The store
target `y_ptr` is taken disjoint enough that the row-vector scatter is injective
(`BlockState.tileIndex1d_offset_injective`).
-/

namespace VeriTile.Bench.TritonBenchG.MatrixReduction

open VeriTile

/-- Faithful 1:1 transcription of `matrix_reduction.py`'s `load_reduce_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_M: tl.constexpr` / `BLOCK_N: tl.constexpr` → Lean `Nat`
  parameters.
- `stride_y` is kept as `_stride_y`: the upstream Triton kernel accepts it but
  stores to `y_ptr + tl.arange(0, BLOCK_M)` and does not use it. -/
def load_reduce_kernel
    (x_ptr y_ptr : RegionName)
    (stride_xm stride_xn _stride_y BLOCK_M BLOCK_N : Nat) :
    ComputeKernel := triton {
  x_ptr = tl.make_block_ptr(base=x_ptr,
    shape=($(BLOCK_M), $(BLOCK_N)),
    strides=($(stride_xm), $(stride_xn)),
    offsets=($(0), $(0)),
    block_shape=($(BLOCK_M), $(BLOCK_N)),
    order=(1, 0))
  x = tl.load(x_ptr)
  y = tl.max(x, axis=1)
  tl.store(y_ptr + tl.arange(0, $(BLOCK_M)), y)
}

/-- Input tile read by the block pointer in `load_reduce_kernel`. -/
noncomputable def matrixReduceInputTile
    (s : BlockState) (x_ptr : RegionName)
    (stride_xm stride_xn BLOCK_M BLOCK_N : Nat) :
    Tile .real [BLOCK_M, BLOCK_N] :=
  { data := fun idx =>
      let bp : BlockPtr :=
        { region := x_ptr, baseOffset := 0, parentShape := [BLOCK_M, BLOCK_N],
          blockShape := [BLOCK_M, BLOCK_N], strides := [stride_xm, stride_xn],
          offsets := [0, 0] }
      some (s.readMem x_ptr (bp.address (TileShape.indexToList [BLOCK_M, BLOCK_N] idx))) }

/-- Exact row-wise max written by `load_reduce_kernel` at row lane `i`. -/
noncomputable def matrixReduceSpec
    (s : BlockState) (x_ptr : RegionName)
    (stride_xm stride_xn BLOCK_M BLOCK_N : Nat) (i : Fin BLOCK_M) : ℝ :=
  match Tile.reduceMax (shape := [BLOCK_M, BLOCK_N]) ⟨1, by simp⟩ Bool.false
      (matrixReduceInputTile s x_ptr stride_xm stride_xn BLOCK_M BLOCK_N) with
  | some out => WithBot.unbotD 0 (out.data (i, PUnit.unit))
  | none => 0

/-- Algorithm-layer row-wise max correctness for `load_reduce_kernel`. -/
theorem load_reduce_kernel_correct
    (x_ptr y_ptr : RegionName)
    (stride_xm stride_xn stride_y BLOCK_M BLOCK_N : Nat)
    (s s' : BlockState)
    (hExec : exec (load_reduce_kernel x_ptr y_ptr stride_xm stride_xn stride_y
          BLOCK_M BLOCK_N) s = some s') :
    ∀ i : Fin BLOCK_M,
      s'.readMem y_ptr i.val =
        matrixReduceSpec s x_ptr stride_xm stride_xn BLOCK_M BLOCK_N i := by
  intro i
  have h_inj := BlockState.tileIndex1d_offset_injective (BLOCK := BLOCK_M)
  by_cases hBM : 0 < BLOCK_M
  · by_cases hBN : 0 < BLOCK_N
    · simp [exec, load_reduce_kernel, stepStmts, stepStmt, evalOp, evalOp.eq_def,
            Tile.reduceMax, Tile.reduceMaxDrop,
            TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
            BlockPtr.address, BlockPtr.inBounds, hBN] at hExec
      subst s'
      rw [BlockState.scatter_readback_nd _ _ _ h_inj (i, PUnit.unit)]
      simp [matrixReduceSpec, matrixReduceInputTile, Tile.reduceMax, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        BlockPtr.address, hBN]
      congr with x
    · simp [exec, load_reduce_kernel, stepStmts, stepStmt, evalOp, evalOp.eq_def,
            Tile.reduceMax, Tile.reduceMaxDrop,
            TileShape.axisDim, TileShape.eraseAxis, hBN] at hExec
  · exact False.elim (hBM (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing row-wise max correctness for `load_reduce_kernel`. -/
theorem load_reduce_kernel_compute_correct
    (x_ptr y_ptr : RegionName)
    (stride_xm stride_xn stride_y BLOCK_M BLOCK_N : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := load_reduce_kernel x_ptr y_ptr stride_xm stride_xn stride_y
        BLOCK_M BLOCK_N)
      (initialState := s)
      (write := fun i : Fin BLOCK_M => some (y_ptr, i.val))
      (expected := fun i => matrixReduceSpec s x_ptr stride_xm stride_xn BLOCK_M BLOCK_N i) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  intro i
  exact load_reduce_kernel_correct x_ptr y_ptr stride_xm stride_xn stride_y
    BLOCK_M BLOCK_N s s' hExec i

/-- Per-kernel output summary for `load_reduce_kernel`: the DSL surface lowers to
the algorithm layer, and the row-vector store to `y_ptr` is compute-correct —
every row lane `i` holds the row-wise maximum `matrixReduceSpec`. -/
theorem load_reduce_kernel_output_summary
    (x_ptr y_ptr : RegionName)
    (stride_xm stride_xn stride_y BLOCK_M BLOCK_N : Nat)
    (s : BlockState) :
    (∃ alg, (load_reduce_kernel x_ptr y_ptr stride_xm stride_xn stride_y
        BLOCK_M BLOCK_N).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := load_reduce_kernel x_ptr y_ptr stride_xm stride_xn stride_y
        BLOCK_M BLOCK_N)
      (initialState := s)
      (write := fun i : Fin BLOCK_M => some (y_ptr, i.val))
      (expected := fun i =>
        matrixReduceSpec s x_ptr stride_xm stride_xn BLOCK_M BLOCK_N i) := by
  refine ⟨⟨_, rfl⟩, ?_⟩
  exact load_reduce_kernel_compute_correct x_ptr y_ptr stride_xm stride_xn
    stride_y BLOCK_M BLOCK_N s

end VeriTile.Bench.TritonBenchG.MatrixReduction
