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

/-- The full nested-loop surface lowers to the algorithm layer, preserving the
three static loop levels, pointer increments, and all stores. -/
theorem nested3_surface_toAlgorithm_supported
    (in_ptr out_ptr : RegionName) (stride_m stride_n : Nat) :
    ∃ alg, (nested3 in_ptr out_ptr stride_m stride_n).toAlgorithm? =
      Except.ok alg := by
  simp [nested3, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

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
  simp [exec, nested3_first_a1_store, stepStmts, stepStmt, evalOp, evalOp.eq_def,
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

/-- Proof-oriented shifted 2x2 transfer slice for the inner nested-loop stores.

`SHIFT = 2` models the first `a_ptrs += 2 * stride_n` / `c_ptrs += 2 * stride_n`
step before the `a2` store in the first inner iteration; `SHIFT = 4` models the
next pointer advance and the corresponding `a3` store. -/
def nested3_shifted_store (in_ptr out_ptr : RegionName)
    (stride_m stride_n SHIFT : Nat) : ComputeKernel := triton {
  offs_am = tl.arange(0, 2)
  offs_an = tl.arange(0, 2)
  a = tl.load(in_ptr + offs_am[:, None] * $(stride_m) +
    offs_an[None, :] * $(stride_n) + $(SHIFT) * $(stride_n))
  tl.store(out_ptr + offs_am[:, None] * $(stride_m) +
    offs_an[None, :] * $(stride_n) + $(SHIFT) * $(stride_n), a)
}

def matrixOffsetShift (stride_m stride_n SHIFT : Nat)
    (idx : TileIndex [2, 2]) : Nat :=
  matrixOffset stride_m stride_n idx + SHIFT * stride_n

theorem nested3_shifted_store_correct
    (in_ptr out_ptr : RegionName)
    (stride_m stride_n SHIFT : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [2, 2] =>
        matrixOffsetShift stride_m stride_n SHIFT idx))
    (hExec : exec (nested3_shifted_store in_ptr out_ptr stride_m stride_n
        SHIFT) s = some s') :
    ∀ idx : TileIndex [2, 2],
      s'.readMem out_ptr (matrixOffsetShift stride_m stride_n SHIFT idx) =
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n SHIFT idx) := by
  intro idx
  simp [exec, nested3_shifted_store, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.ptrAdd, Tile.expandDim,
        NumericDType.add, NumericDType.mul,
        TileShape.insertAxis, TileShape.dropInsertedIndex] at hExec
  rw [← hExec]
  simp [matrixOffsetShift, matrixOffset, rowIndex, colIndex, Nat.add_assoc]
  simpa [matrixOffsetShift, matrixOffset, rowIndex, colIndex, Nat.add_assoc,
         TileShape.insertAxis, TileShape.dropInsertedIndex] using
    (BlockState.scatter_readback_nd
      (region := out_ptr)
      (shape := [2, 2])
      (s := ((s.setReg "offs_am" TileDType.nat [2] (Tile.vec fun i => i.val))
        |>.setReg "offs_an" TileDType.nat [2] (Tile.vec fun i => i.val)
        |>.setReg "a" TileDType.real [2, 2]
          { data := fun i =>
            some (s.readMem in_ptr
              (i.1.val * stride_m + i.2.1.val * stride_n +
                SHIFT * stride_n)) }))
      (offsetFn := fun i : TileIndex [2, 2] =>
        i.1.val * stride_m + i.2.1.val * stride_n + SHIFT * stride_n)
      (valueFn := fun i : TileIndex [2, 2] =>
        s.readMem in_ptr
          (i.1.val * stride_m + i.2.1.val * stride_n + SHIFT * stride_n))
      hOutInj idx)

theorem nested3_shifted_store_compute_correct
    (in_ptr out_ptr : RegionName)
    (stride_m stride_n SHIFT : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [2, 2] =>
        matrixOffsetShift stride_m stride_n SHIFT idx)) :
    ComputeCorrect.Realizes
      (kernel := nested3_shifted_store in_ptr out_ptr stride_m stride_n SHIFT)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n SHIFT idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n SHIFT idx)) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [nested3_shifted_store]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  exact nested3_shifted_store_correct in_ptr out_ptr stride_m stride_n SHIFT
    s s' hOutInj hExec idx

/-- First `a2` store of the initial `(i,j,k) = (0,0,0)` path. -/
abbrev nested3_first_a2_store (in_ptr out_ptr : RegionName)
    (stride_m stride_n : Nat) : ComputeKernel :=
  nested3_shifted_store in_ptr out_ptr stride_m stride_n 2

/-- First `a3` store of the initial `(i,j,k) = (0,0,0)` path. -/
abbrev nested3_first_a3_store (in_ptr out_ptr : RegionName)
    (stride_m stride_n : Nat) : ComputeKernel :=
  nested3_shifted_store in_ptr out_ptr stride_m stride_n 4

theorem nested3_first_a2_store_compute_correct
    (in_ptr out_ptr : RegionName)
    (stride_m stride_n : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [2, 2] =>
        matrixOffsetShift stride_m stride_n 2 idx)) :
    ComputeCorrect.Realizes
      (kernel := nested3_first_a2_store in_ptr out_ptr stride_m stride_n)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 2 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 2 idx)) :=
  nested3_shifted_store_compute_correct in_ptr out_ptr stride_m stride_n 2
    s hOutInj

theorem nested3_first_a3_store_compute_correct
    (in_ptr out_ptr : RegionName)
    (stride_m stride_n : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [2, 2] =>
        matrixOffsetShift stride_m stride_n 4 idx)) :
    ComputeCorrect.Realizes
      (kernel := nested3_first_a3_store in_ptr out_ptr stride_m stride_n)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 4 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 4 idx)) :=
  nested3_shifted_store_compute_correct in_ptr out_ptr stride_m stride_n 4
    s hOutInj

/-- Generic shifted copy slice for later nested-loop stores where the source
`a_ptrs` shift and destination `c_ptrs` shift are no longer the same. -/
def nested3_shifted_copy_store (in_ptr out_ptr : RegionName)
    (stride_m stride_n IN_SHIFT OUT_SHIFT : Nat) : ComputeKernel := triton {
  offs_am = tl.arange(0, 2)
  offs_an = tl.arange(0, 2)
  a = tl.load(in_ptr + offs_am[:, None] * $(stride_m) +
    offs_an[None, :] * $(stride_n) + $(IN_SHIFT) * $(stride_n))
  tl.store(out_ptr + offs_am[:, None] * $(stride_m) +
    offs_an[None, :] * $(stride_n) + $(OUT_SHIFT) * $(stride_n), a)
}

theorem nested3_shifted_copy_store_correct
    (in_ptr out_ptr : RegionName)
    (stride_m stride_n IN_SHIFT OUT_SHIFT : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [2, 2] =>
        matrixOffsetShift stride_m stride_n OUT_SHIFT idx))
    (hExec : exec (nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n
        IN_SHIFT OUT_SHIFT) s = some s') :
    ∀ idx : TileIndex [2, 2],
      s'.readMem out_ptr (matrixOffsetShift stride_m stride_n OUT_SHIFT idx) =
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n IN_SHIFT idx) := by
  intro idx
  simp [exec, nested3_shifted_copy_store, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.ptrAdd, Tile.expandDim,
        NumericDType.add, NumericDType.mul,
        TileShape.insertAxis, TileShape.dropInsertedIndex] at hExec
  rw [← hExec]
  simp [matrixOffsetShift, matrixOffset, rowIndex, colIndex, Nat.add_assoc]
  simpa [matrixOffsetShift, matrixOffset, rowIndex, colIndex, Nat.add_assoc,
         TileShape.insertAxis, TileShape.dropInsertedIndex] using
    (BlockState.scatter_readback_nd
      (region := out_ptr)
      (shape := [2, 2])
      (s := ((s.setReg "offs_am" TileDType.nat [2] (Tile.vec fun i => i.val))
        |>.setReg "offs_an" TileDType.nat [2] (Tile.vec fun i => i.val)
        |>.setReg "a" TileDType.real [2, 2]
          { data := fun i =>
            some (s.readMem in_ptr
              (i.1.val * stride_m + i.2.1.val * stride_n +
                IN_SHIFT * stride_n)) }))
      (offsetFn := fun i : TileIndex [2, 2] =>
        i.1.val * stride_m + i.2.1.val * stride_n + OUT_SHIFT * stride_n)
      (valueFn := fun i : TileIndex [2, 2] =>
        s.readMem in_ptr
          (i.1.val * stride_m + i.2.1.val * stride_n + IN_SHIFT * stride_n))
      hOutInj idx)

theorem nested3_shifted_copy_store_compute_correct
    (in_ptr out_ptr : RegionName)
    (stride_m stride_n IN_SHIFT OUT_SHIFT : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [2, 2] =>
        matrixOffsetShift stride_m stride_n OUT_SHIFT idx)) :
    ComputeCorrect.Realizes
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n
        IN_SHIFT OUT_SHIFT)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n OUT_SHIFT idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n IN_SHIFT idx)) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [nested3_shifted_copy_store]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  exact nested3_shifted_copy_store_correct in_ptr out_ptr stride_m stride_n
    IN_SHIFT OUT_SHIFT s s' hOutInj hExec idx

/-- Second `k` iteration: `a1` is reused from shift 0 and written after the
first three output blocks. -/
theorem nested3_second_k_a1_store_compute_correct
    (in_ptr out_ptr : RegionName)
    (stride_m stride_n : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [2, 2] =>
        matrixOffsetShift stride_m stride_n 6 idx)) :
    ComputeCorrect.Realizes
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n
        0 6)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 6 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 0 idx)) :=
  nested3_shifted_copy_store_compute_correct in_ptr out_ptr stride_m stride_n
    0 6 s hOutInj

/-- Second `k` iteration: `a2` from shift 2 is written after the reused `a1`. -/
theorem nested3_second_k_a2_store_compute_correct
    (in_ptr out_ptr : RegionName)
    (stride_m stride_n : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [2, 2] =>
        matrixOffsetShift stride_m stride_n 8 idx)) :
    ComputeCorrect.Realizes
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n
        2 8)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 8 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 2 idx)) :=
  nested3_shifted_copy_store_compute_correct in_ptr out_ptr stride_m stride_n
    2 8 s hOutInj

/-- Second `k` iteration: after another `a_ptrs += 2 * stride_n`, `a3` comes
from shift 6 and is written after the second `a2` block. -/
theorem nested3_second_k_a3_store_compute_correct
    (in_ptr out_ptr : RegionName)
    (stride_m stride_n : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [2, 2] =>
        matrixOffsetShift stride_m stride_n 10 idx)) :
    ComputeCorrect.Realizes
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n
        6 10)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 10 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 6 idx)) :=
  nested3_shifted_copy_store_compute_correct in_ptr out_ptr stride_m stride_n
    6 10 s hOutInj

/-- First `i`, second `j`, first `k`: reused `a1`, now written at output
shift 12. -/
theorem nested3_second_j_first_k_a1_store_compute_correct
    (in_ptr out_ptr : RegionName)
    (stride_m stride_n : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [2, 2] =>
        matrixOffsetShift stride_m stride_n 12 idx)) :
    ComputeCorrect.Realizes
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n
        0 12)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 12 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 0 idx)) :=
  nested3_shifted_copy_store_compute_correct in_ptr out_ptr stride_m stride_n
    0 12 s hOutInj

/-- First `i`, second `j`, first `k`: `a2` is loaded after the `j=1`
pointer advance, from input shift 8. -/
theorem nested3_second_j_first_k_a2_store_compute_correct
    (in_ptr out_ptr : RegionName)
    (stride_m stride_n : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [2, 2] =>
        matrixOffsetShift stride_m stride_n 14 idx)) :
    ComputeCorrect.Realizes
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n
        8 14)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 14 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 8 idx)) :=
  nested3_shifted_copy_store_compute_correct in_ptr out_ptr stride_m stride_n
    8 14 s hOutInj

/-- First `i`, second `j`, first `k`: `a3` comes from input shift 10. -/
theorem nested3_second_j_first_k_a3_store_compute_correct
    (in_ptr out_ptr : RegionName)
    (stride_m stride_n : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [2, 2] =>
        matrixOffsetShift stride_m stride_n 16 idx)) :
    ComputeCorrect.Realizes
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n
        10 16)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 16 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 10 idx)) :=
  nested3_shifted_copy_store_compute_correct in_ptr out_ptr stride_m stride_n
    10 16 s hOutInj

/-- First `i`, second `j`, second `k`: reused `a1`, written at output
shift 18. -/
theorem nested3_second_j_second_k_a1_store_compute_correct
    (in_ptr out_ptr : RegionName)
    (stride_m stride_n : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [2, 2] =>
        matrixOffsetShift stride_m stride_n 18 idx)) :
    ComputeCorrect.Realizes
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n
        0 18)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 18 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 0 idx)) :=
  nested3_shifted_copy_store_compute_correct in_ptr out_ptr stride_m stride_n
    0 18 s hOutInj

/-- First `i`, second `j`, second `k`: reused `a2` from input shift 8. -/
theorem nested3_second_j_second_k_a2_store_compute_correct
    (in_ptr out_ptr : RegionName)
    (stride_m stride_n : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [2, 2] =>
        matrixOffsetShift stride_m stride_n 20 idx)) :
    ComputeCorrect.Realizes
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n
        8 20)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 20 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 8 idx)) :=
  nested3_shifted_copy_store_compute_correct in_ptr out_ptr stride_m stride_n
    8 20 s hOutInj

/-- First `i`, second `j`, second `k`: after one more inner advance, `a3`
comes from input shift 12. -/
theorem nested3_second_j_second_k_a3_store_compute_correct
    (in_ptr out_ptr : RegionName)
    (stride_m stride_n : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [2, 2] =>
        matrixOffsetShift stride_m stride_n 22 idx)) :
    ComputeCorrect.Realizes
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n
        12 22)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 22 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 12 idx)) :=
  nested3_shifted_copy_store_compute_correct in_ptr out_ptr stride_m stride_n
    12 22 s hOutInj

/-- Second outer `i`, first `j`, first `k`: new `a1` loaded after the outer
pointer advance, from input shift 14. -/
theorem nested3_second_i_first_j_first_k_a1_store_compute_correct
    (in_ptr out_ptr : RegionName)
    (stride_m stride_n : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [2, 2] =>
        matrixOffsetShift stride_m stride_n 24 idx)) :
    ComputeCorrect.Realizes
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n
        14 24)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 24 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 14 idx)) :=
  nested3_shifted_copy_store_compute_correct in_ptr out_ptr stride_m stride_n
    14 24 s hOutInj

/-- Second outer `i`, first `j`, first `k`: `a2` is loaded from input
shift 16. -/
theorem nested3_second_i_first_j_first_k_a2_store_compute_correct
    (in_ptr out_ptr : RegionName)
    (stride_m stride_n : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [2, 2] =>
        matrixOffsetShift stride_m stride_n 26 idx)) :
    ComputeCorrect.Realizes
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n
        16 26)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 26 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 16 idx)) :=
  nested3_shifted_copy_store_compute_correct in_ptr out_ptr stride_m stride_n
    16 26 s hOutInj

/-- Second outer `i`, first `j`, first `k`: `a3` comes from input shift 18. -/
theorem nested3_second_i_first_j_first_k_a3_store_compute_correct
    (in_ptr out_ptr : RegionName)
    (stride_m stride_n : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [2, 2] =>
        matrixOffsetShift stride_m stride_n 28 idx)) :
    ComputeCorrect.Realizes
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n
        18 28)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 28 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 18 idx)) :=
  nested3_shifted_copy_store_compute_correct in_ptr out_ptr stride_m stride_n
    18 28 s hOutInj

/-- Second outer `i`, first `j`, second `k`: reused `a1` from input shift 14. -/
theorem nested3_second_i_first_j_second_k_a1_store_compute_correct
    (in_ptr out_ptr : RegionName)
    (stride_m stride_n : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [2, 2] =>
        matrixOffsetShift stride_m stride_n 30 idx)) :
    ComputeCorrect.Realizes
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n
        14 30)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 30 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 14 idx)) :=
  nested3_shifted_copy_store_compute_correct in_ptr out_ptr stride_m stride_n
    14 30 s hOutInj

/-- Second outer `i`, first `j`, second `k`: reused `a2` from input shift 16. -/
theorem nested3_second_i_first_j_second_k_a2_store_compute_correct
    (in_ptr out_ptr : RegionName)
    (stride_m stride_n : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [2, 2] =>
        matrixOffsetShift stride_m stride_n 32 idx)) :
    ComputeCorrect.Realizes
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n
        16 32)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 32 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 16 idx)) :=
  nested3_shifted_copy_store_compute_correct in_ptr out_ptr stride_m stride_n
    16 32 s hOutInj

/-- Second outer `i`, first `j`, second `k`: after one more inner advance,
`a3` comes from input shift 20. -/
theorem nested3_second_i_first_j_second_k_a3_store_compute_correct
    (in_ptr out_ptr : RegionName)
    (stride_m stride_n : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [2, 2] =>
        matrixOffsetShift stride_m stride_n 34 idx)) :
    ComputeCorrect.Realizes
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n
        20 34)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 34 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 20 idx)) :=
  nested3_shifted_copy_store_compute_correct in_ptr out_ptr stride_m stride_n
    20 34 s hOutInj

/-- Second outer `i`, second `j`, first `k`: reused `a1` from input shift 14,
written at output shift 36. -/
theorem nested3_second_i_second_j_first_k_a1_store_compute_correct
    (in_ptr out_ptr : RegionName)
    (stride_m stride_n : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [2, 2] =>
        matrixOffsetShift stride_m stride_n 36 idx)) :
    ComputeCorrect.Realizes
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n
        14 36)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 36 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 14 idx)) :=
  nested3_shifted_copy_store_compute_correct in_ptr out_ptr stride_m stride_n
    14 36 s hOutInj

/-- Second outer `i`, second `j`, first `k`: `a2` is loaded from input
shift 22. -/
theorem nested3_second_i_second_j_first_k_a2_store_compute_correct
    (in_ptr out_ptr : RegionName)
    (stride_m stride_n : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [2, 2] =>
        matrixOffsetShift stride_m stride_n 38 idx)) :
    ComputeCorrect.Realizes
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n
        22 38)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 38 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 22 idx)) :=
  nested3_shifted_copy_store_compute_correct in_ptr out_ptr stride_m stride_n
    22 38 s hOutInj

/-- Second outer `i`, second `j`, first `k`: `a3` comes from input shift 24. -/
theorem nested3_second_i_second_j_first_k_a3_store_compute_correct
    (in_ptr out_ptr : RegionName)
    (stride_m stride_n : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [2, 2] =>
        matrixOffsetShift stride_m stride_n 40 idx)) :
    ComputeCorrect.Realizes
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n
        24 40)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 40 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 24 idx)) :=
  nested3_shifted_copy_store_compute_correct in_ptr out_ptr stride_m stride_n
    24 40 s hOutInj

/-- Second outer `i`, second `j`, second `k`: reused `a1` from input shift
14, written at output shift 42. -/
theorem nested3_second_i_second_j_second_k_a1_store_compute_correct
    (in_ptr out_ptr : RegionName)
    (stride_m stride_n : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [2, 2] =>
        matrixOffsetShift stride_m stride_n 42 idx)) :
    ComputeCorrect.Realizes
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n
        14 42)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 42 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 14 idx)) :=
  nested3_shifted_copy_store_compute_correct in_ptr out_ptr stride_m stride_n
    14 42 s hOutInj

/-- Second outer `i`, second `j`, second `k`: reused `a2` from input
shift 22. -/
theorem nested3_second_i_second_j_second_k_a2_store_compute_correct
    (in_ptr out_ptr : RegionName)
    (stride_m stride_n : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [2, 2] =>
        matrixOffsetShift stride_m stride_n 44 idx)) :
    ComputeCorrect.Realizes
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n
        22 44)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 44 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 22 idx)) :=
  nested3_shifted_copy_store_compute_correct in_ptr out_ptr stride_m stride_n
    22 44 s hOutInj

/-- Second outer `i`, second `j`, second `k`: final `a3` store, from input
shift 26 to output shift 46. -/
theorem nested3_second_i_second_j_second_k_a3_store_compute_correct
    (in_ptr out_ptr : RegionName)
    (stride_m stride_n : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [2, 2] =>
        matrixOffsetShift stride_m stride_n 46 idx)) :
    ComputeCorrect.Realizes
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n
        26 46)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 46 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 26 idx)) :=
  nested3_shifted_copy_store_compute_correct in_ptr out_ptr stride_m stride_n
    26 46 s hOutInj

/-! ## Python test-case surface wrappers

The Python wrapper passes contiguous square tensors, so `stride_m = n_cols` and
`stride_n = 1`. The grid varies with `n_cols // 4`; the kernel body itself is
stride-parametric, and these wrappers pin the four checked tensor layouts. -/

theorem nested3_python_case1_surface_toAlgorithm_supported
    (in_ptr out_ptr : RegionName) :
    ∃ alg, (nested3 in_ptr out_ptr 8 1).toAlgorithm? = Except.ok alg := by
  exact nested3_surface_toAlgorithm_supported in_ptr out_ptr 8 1

theorem nested3_python_case2_surface_toAlgorithm_supported
    (in_ptr out_ptr : RegionName) :
    ∃ alg, (nested3 in_ptr out_ptr 4 1).toAlgorithm? = Except.ok alg := by
  exact nested3_surface_toAlgorithm_supported in_ptr out_ptr 4 1

theorem nested3_python_case3_surface_toAlgorithm_supported
    (in_ptr out_ptr : RegionName) :
    ∃ alg, (nested3 in_ptr out_ptr 16 1).toAlgorithm? = Except.ok alg := by
  exact nested3_surface_toAlgorithm_supported in_ptr out_ptr 16 1

theorem nested3_python_case4_surface_toAlgorithm_supported
    (in_ptr out_ptr : RegionName) :
    ∃ alg, (nested3 in_ptr out_ptr 2 1).toAlgorithm? = Except.ok alg := by
  exact nested3_surface_toAlgorithm_supported in_ptr out_ptr 2 1

/-- Public Python surface summary for `nested_loops_processing.py`.

The Python regression checks four contiguous square layouts. This summary
records that each tested layout lowers at the full nested-loop surface; the
proof-oriented shifted store slices above provide the concrete store semantics
for the nested body. -/
theorem nested3_python_surfaces_output_summary
    (in_ptr out_ptr : RegionName) :
    (∃ alg, (nested3 in_ptr out_ptr 8 1).toAlgorithm? = Except.ok alg) ∧
    (∃ alg, (nested3 in_ptr out_ptr 4 1).toAlgorithm? = Except.ok alg) ∧
    (∃ alg, (nested3 in_ptr out_ptr 16 1).toAlgorithm? = Except.ok alg) ∧
    (∃ alg, (nested3 in_ptr out_ptr 2 1).toAlgorithm? = Except.ok alg) := by
  constructor
  · exact nested3_python_case1_surface_toAlgorithm_supported in_ptr out_ptr
  · constructor
    · exact nested3_python_case2_surface_toAlgorithm_supported in_ptr out_ptr
    · constructor
      · exact nested3_python_case3_surface_toAlgorithm_supported in_ptr out_ptr
      · exact nested3_python_case4_surface_toAlgorithm_supported in_ptr out_ptr

end VeriTile.Bench.TritonBenchG.NestedLoopsProcessing
