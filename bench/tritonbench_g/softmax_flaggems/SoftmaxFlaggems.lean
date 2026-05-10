import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems

open VeriTile.Triton

/-- Proof-oriented `ONE_TILE_PER_CTA=true` slice of
`softmax_flaggems.py`'s `softmax_kernel_inner`.

This covers the inner-dimension fast path where one CTA covers a full row. It
preserves the source kernel's row program id, masked load with `-inf`, stable
softmax max/sum normalization, and masked store. The non-inner 2D path and
multi-tile online fallback remain future work. -/
def softmax_kernel_inner_one_tile
    (output_ptr input_ptr : RegionName)
    (N TILE_N : Nat) :
    ComputeKernel := triton {
  pid_m = tl.program_id(0)
  n_offsets = tl.arange(0, $(TILE_N))
  offset = pid_m * $(N) + n_offsets
  mask = n_offsets < $(N)
  input_ptrs = input_ptr + offset
  inp = tl.load(input_ptrs, mask=mask, other=-inf)
  m = tl.max(inp, axis=0)
  e = tl.exp(inp - m)
  z = tl.sum(e, axis=0)
  out = e / z
  output_ptrs = output_ptr + offset
  tl.store(output_ptrs, out, mask=mask)
}

def outOffset (s : BlockState) (N : Nat) (i : Fin TILE_N) : Nat :=
  s.pid * N + i.val

noncomputable def softmaxFlaggemsInputTile
    (s : BlockState) (input_ptr : RegionName) (N TILE_N : Nat) :
    Tile .real [TILE_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        some (s.readMem input_ptr (outOffset s N idx.1))
      else none }

noncomputable def softmaxFlaggemsSpec
    (s : BlockState) (input_ptr : RegionName)
    (N TILE_N : Nat) (idx : Fin TILE_N) : ℝ :=
  let row := softmaxFlaggemsInputTile s input_ptr N TILE_N
  match Tile.reduceMax (shape := [TILE_N]) ⟨0, by simp⟩ Bool.false row with
  | some rowMax =>
      let shifted := Tile.bop (NumericDType.sub .real) Broadcast.scalarR row rowMax
      let e := Tile.uop WithBot.realExp shifted
      let z := Tile.reduceSum (shape := [TILE_N]) ⟨0, by simp⟩ Bool.false e
      WithBot.unbotD 0
        ((Tile.bop (NumericDType.div .real) Broadcast.scalarR e z).data
          (idx, PUnit.unit))
  | none => 0

/-- Algorithm-layer correctness for the inner one-tile FlagGems softmax. -/
theorem softmax_kernel_inner_one_tile_correct
    (output_ptr input_ptr : RegionName)
    (N TILE_N : Nat)
    (s s' : BlockState)
    (hExec : exec (softmax_kernel_inner_one_tile output_ptr input_ptr N TILE_N) s =
        some s') :
    ∀ i : Fin TILE_N,
      s'.readMem output_ptr (outOffset s N i) =
        if i.val < N then
          softmaxFlaggemsSpec s input_ptr N TILE_N i
        else s.readMem output_ptr (outOffset s N i) := by
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [TILE_N] => s.pids 0 * N + idx.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  by_cases hT : 0 < TILE_N
  · simp [exec, softmax_kernel_inner_one_tile, stepStmts, stepStmt, evalOp,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
          TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
          NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
          ComparableDType.lt, hT] at hExec
    subst s'
    simp [BlockState.pid_eq, outOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
    by_cases hi : i.val < N
    · simp [hi, softmaxFlaggemsSpec, softmaxFlaggemsInputTile, outOffset,
            Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum,
            Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
            TileShape.insertAxisIndex, hT]
      congr
    · simp [hi]
  · exact False.elim (hT (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the inner one-tile FlagGems softmax. -/
theorem softmax_kernel_inner_one_tile_compute_correct
    (output_ptr input_ptr : RegionName)
    (N TILE_N : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := softmax_kernel_inner_one_tile output_ptr input_ptr N TILE_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin TILE_N => i.val < N)
        (fun i => (output_ptr, outOffset s N i)))
      (expected := fun i => softmaxFlaggemsSpec s input_ptr N TILE_N i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [softmax_kernel_inner_one_tile]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := softmax_kernel_inner_one_tile_correct output_ptr input_ptr N TILE_N
    s s' hExec i
  simpa [hActive] using h

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems
