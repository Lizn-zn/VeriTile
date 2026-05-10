import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Proof-oriented `Y00` slice of `fifth_order_sph_harmonics.py`'s
`fifth_order_fwd`.

The full source kernel writes eleven fifth-order spherical harmonics channels.
This slice covers the common coordinate/block pointer arithmetic, masked
coordinate loads, the `Y00` fifth-order polynomial, and the first strided output
store. Remaining channels and backward gradients are future work. -/
def fifth_order_fwd_y00
    (coord_ptr output_ptr : RegionName)
    (block_size coord_numel output_numel col_offset output_stride : Nat) :
    ComputeKernel := triton {
  block_id = tl.program_id(0)
  coord_striding = tl.arange(0, $(block_size)) * $(3)
  coord_row_offset = coord_striding + $(block_size) * $(3) * block_id
  x = tl.load(coord_ptr + coord_row_offset, mask=coord_row_offset < $(coord_numel))
  z = tl.load(coord_ptr + coord_row_offset + $(2), mask=coord_row_offset + $(2) < $(coord_numel))
  x2 = x * x
  x3 = x2 * x
  x5 = x3 * x2
  z2 = z * z
  z4 = z2 * z2
  y00 = 2.32681380862329 * x5 + 11.6340690431164 * z4 * x - 23.2681380862329 * x3 * z2
  output_striding = tl.arange(0, $(block_size)) * $(output_stride)
  output_row_offset = output_striding + $(block_size) * $(output_stride) * block_id +
    $(col_offset)
  tl.store(output_ptr + output_row_offset, y00, mask=output_row_offset < $(output_numel))
}

def coordOffset (s : BlockState) (block_size : Nat) (i : Fin block_size) : Nat :=
  i.val * 3 + block_size * 3 * s.pid

def outOffset
    (s : BlockState) (block_size col_offset output_stride : Nat)
    (i : Fin block_size) : Nat :=
  i.val * output_stride + block_size * output_stride * s.pid + col_offset

noncomputable def coordX
    (s : BlockState) (coord_ptr : RegionName)
    (block_size coord_numel : Nat) (i : Fin block_size) : WithBot ℝ :=
  if coordOffset s block_size i < coord_numel then
    some (s.readMem coord_ptr (coordOffset s block_size i))
  else some (s.undef coord_ptr (coordOffset s block_size i))

noncomputable def coordZ
    (s : BlockState) (coord_ptr : RegionName)
    (block_size coord_numel : Nat) (i : Fin block_size) : WithBot ℝ :=
  if coordOffset s block_size i + 2 < coord_numel then
    some (s.readMem coord_ptr (coordOffset s block_size i + 2))
  else some (s.undef coord_ptr (coordOffset s block_size i + 2))

noncomputable def y00Spec
    (s : BlockState) (coord_ptr : RegionName)
    (block_size coord_numel : Nat) (i : Fin block_size) : ℝ :=
  let x := coordX s coord_ptr block_size coord_numel i
  let z := coordZ s coord_ptr block_size coord_numel i
  let x2 := Option.map₂ (fun x1 x2 => x1 * x2) x x
  let x3 := Option.map₂ (fun x1 x2 => x1 * x2) x2 x
  let x5 := Option.map₂ (fun x1 x2 => x1 * x2) x3 x2
  let z2 := Option.map₂ (fun x1 x2 => x1 * x2) z z
  let z4 := Option.map₂ (fun x1 x2 => x1 * x2) z2 z2
  WithBot.unbotD 0
    (Option.map₂ (fun x1 x2 => x1 - x2)
      (Option.map₂ (fun x1 x2 => x1 + x2)
        (Option.map (fun b => 2.32681380862329 * b) x5)
        (Option.map₂ (fun x1 x2 => x1 * x2)
          (Option.map (fun b => 11.6340690431164 * b) z4)
          x))
      (Option.map₂ (fun x1 x2 => x1 * x2)
        (Option.map (fun b => 23.2681380862329 * b) x3)
        z2))

/-- Algorithm-layer correctness for the `Y00` forward slice. -/
theorem fifth_order_fwd_y00_correct
    (coord_ptr output_ptr : RegionName)
    (block_size coord_numel output_numel col_offset output_stride : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin block_size => outOffset s block_size col_offset output_stride i))
    (hExec : exec (fifth_order_fwd_y00 coord_ptr output_ptr block_size coord_numel
        output_numel col_offset output_stride) s = some s') :
    ∀ i : Fin block_size,
      s'.readMem output_ptr (outOffset s block_size col_offset output_stride i) =
        if outOffset s block_size col_offset output_stride i < output_numel then
          y00Spec s coord_ptr block_size coord_numel i
        else s.readMem output_ptr (outOffset s block_size col_offset output_stride i) := by
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [block_size] =>
        idx.1.val * output_stride + block_size * output_stride * s.pid + col_offset) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [outOffset] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hB : 0 < block_size
  · simp [exec, fifth_order_fwd_y00, stepStmts, stepStmt, evalOp,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          NumericDType.add, NumericDType.mul, ComparableDType.lt, hB] at hExec
    subst s'
    simp only [outOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
    by_cases hActive : i.val * output_stride + block_size * output_stride * s.pid +
        col_offset < output_numel
    · simp [hActive, y00Spec, coordX, coordZ, coordOffset, NumericDType.mul]
      rfl
    · simp [hActive]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the `Y00` forward slice. -/
theorem fifth_order_fwd_y00_compute_correct
    (coord_ptr output_ptr : RegionName)
    (block_size coord_numel output_numel col_offset output_stride : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin block_size => outOffset s block_size col_offset output_stride i)) :
    ComputeCorrect.Realizes
      (kernel := fifth_order_fwd_y00 coord_ptr output_ptr block_size coord_numel
        output_numel col_offset output_stride)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin block_size =>
          outOffset s block_size col_offset output_stride i < output_numel)
        (fun i => (output_ptr, outOffset s block_size col_offset output_stride i)))
      (expected := fun i => y00Spec s coord_ptr block_size coord_numel i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fifth_order_fwd_y00]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := fifth_order_fwd_y00_correct coord_ptr output_ptr block_size coord_numel
    output_numel col_offset output_stride s s' hOutInj hExec i
  simpa [hActive] using h

end VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics
