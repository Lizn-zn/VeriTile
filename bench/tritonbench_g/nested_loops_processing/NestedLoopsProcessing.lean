import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.NestedLoopsProcessing

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Faithful transcription of `nested_loops_processing.py`'s `nested3`.

Allowed mechanical Lean-syntax-only changes:
- Python literal loop bounds `range(0, 2)` are written as
  `range(0, 2, 1)`.
- Scalar constants in pointer increments are antiquoted so they are inferred
  in the Nat offset channel. -/
def nested3 (in_ptr out_ptr : RegionName)
    (stride_m stride_n : Nat) : ComputeKernel := triton {
  offs_am = tl.arange(0, 2)
  offs_an = tl.arange(0, 2)
  a_ptrs = in_ptr + (offs_am[:, None] * $(stride_m) +
    offs_an[None, :] * $(stride_n))
  offs_cm = tl.arange(0, 2)
  offs_cn = tl.arange(0, 2)
  c_ptrs = out_ptr + $(stride_m) * offs_cm[:, None] +
    $(stride_n) * offs_cn[None, :]
  for i in range(0, $(2), $(1)) {
    a1 = tl.load(a_ptrs)
    for j in range(0, $(2), $(1)) {
      a_ptrs += $(2) * $(stride_n)
      a2 = tl.load(a_ptrs)
      for k in range(0, $(2), $(1)) {
        a_ptrs += $(2) * $(stride_n)
        a3 = tl.load(a_ptrs)
        tl.store(c_ptrs, a1)
        c_ptrs += $(2) * $(stride_n)
        tl.store(c_ptrs, a2)
        c_ptrs += $(2) * $(stride_n)
        tl.store(c_ptrs, a3)
        c_ptrs += $(2) * $(stride_n)
      }
    }
    a_ptrs += $(2) * $(stride_n)
  }
}

/-- Proof-oriented first 2x2 transfer slice of `nested_loops_processing.py`'s
`nested3`.

The full kernel repeatedly advances `a_ptrs` and `c_ptrs` through nested static
loops. This slice captures the initial `a1 = tl.load(a_ptrs)` and first
`tl.store(c_ptrs, a1)` block. -/
def nested3_first_a1_store (in_ptr out_ptr : RegionName)
    (stride_m stride_n : Nat) : ComputeKernel := triton {
  offs_am = tl.arange(0, 2)
  offs_an = tl.arange(0, 2)
  a = tl.load(in_ptr + offs_am[:, None] * $(stride_m) +
    offs_an[None, :] * $(stride_n))
  tl.store(out_ptr + offs_am[:, None] * $(stride_m) +
    offs_an[None, :] * $(stride_n), a)
}

def rowIndex (idx : TileIndex [2, 2]) : Nat :=
  idx.1.val

def colIndex (idx : TileIndex [2, 2]) : Nat :=
  idx.2.1.val

def matrixOffset (stride_m stride_n : Nat) (idx : TileIndex [2, 2]) : Nat :=
  rowIndex idx * stride_m + colIndex idx * stride_n

/-- Algorithm-layer correctness for the initial `a1` 2x2 store. -/
theorem nested3_first_a1_store_correct
    (in_ptr out_ptr : RegionName)
    (stride_m stride_n : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffset stride_m stride_n idx))
    (hExec : exec (nested3_first_a1_store in_ptr out_ptr stride_m stride_n) s =
      some s') :
    ∀ idx : TileIndex [2, 2],
      s'.readMem out_ptr (matrixOffset stride_m stride_n idx) =
        s.readMem in_ptr (matrixOffset stride_m stride_n idx) := by
  intro idx
  simp [exec, nested3_first_a1_store, stepStmts, stepStmt, evalOp,
        Option.bind, Option.map, Tile.bop, Tile.ptrAdd, Tile.expandDim,
        NumericDType.add, NumericDType.mul,
        TileShape.insertAxis, TileShape.dropInsertedIndex] at hExec
  rw [← hExec]
  simp [matrixOffset, rowIndex, colIndex]
  simpa [matrixOffset, rowIndex, colIndex,
         TileShape.insertAxis, TileShape.dropInsertedIndex] using
    (BlockState.scatter_readback_nd
      (region := out_ptr)
      (shape := [2, 2])
      (s := ((s.setReg "offs_am" TileDType.nat [2] (Tile.vec fun i => i.val))
        |>.setReg "offs_an" TileDType.nat [2] (Tile.vec fun i => i.val)
        |>.setReg "a" TileDType.real [2, 2]
          { data := fun i =>
            some (s.readMem in_ptr (i.1.val * stride_m + i.2.1.val * stride_n)) }))
      (offsetFn := fun i : TileIndex [2, 2] =>
        i.1.val * stride_m + i.2.1.val * stride_n)
      (valueFn := fun i : TileIndex [2, 2] =>
        s.readMem in_ptr (i.1.val * stride_m + i.2.1.val * stride_n))
      hOutInj idx)

/-- Compute-facing correctness for the initial `a1` 2x2 store. -/
theorem nested3_first_a1_store_compute_correct
    (in_ptr out_ptr : RegionName)
    (stride_m stride_n : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffset stride_m stride_n idx)) :
    ComputeCorrect.Realizes
      (kernel := nested3_first_a1_store in_ptr out_ptr stride_m stride_n)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffset stride_m stride_n idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffset stride_m stride_n idx)) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [nested3_first_a1_store]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  exact nested3_first_a1_store_correct in_ptr out_ptr stride_m stride_n
    s s' hOutInj hExec idx

end VeriTile.Bench.TritonBenchG.NestedLoopsProcessing
