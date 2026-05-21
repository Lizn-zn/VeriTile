import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.SoftmaxOptimize

open VeriTile.Triton

set_option linter.unusedVariables false

/-- Faithful transcription of `softmax_optimize.py`'s
`softmax_kernel_online_v2`.

This keeps the online max/sum recurrence over full tiles, the masked tail pass,
the final scalar normalization, and the two writeback loops. -/
def softmax_kernel_online_v2_surface
    (output_ptr input_ptr : RegionName)
    (M N TILE_N : Nat) :
    ComputeKernel := triton {
  pid_m = tl.program_id(0)
  m = tl.full([$(TILE_N)], value=-float("inf"), dtype=output_ptr.dtype.element_ty)
  z = tl.full([$(TILE_N)], value=0, dtype=output_ptr.dtype.element_ty)
  prev_multiple = $((N + TILE_N - 1)) // $(TILE_N) * $(TILE_N) - $(TILE_N)
  for start_n in range($(0), prev_multiple, $(TILE_N)) {
    n_offsets = start_n + tl.arange(0, $(TILE_N))
    offset = pid_m * $(N) + n_offsets
    input_ptrs = input_ptr + offset
    inp = (tl.load(input_ptrs)).to(output_ptr.dtype.element_ty)
    new_m = tl.maximum(m, inp)
    new_z = tl.exp(m - new_m) * z + tl.exp(inp - new_m)
    m = new_m
    z = new_z
  }
  for start_n in range(prev_multiple, $(N), $(TILE_N)) {
    n_offsets = start_n + tl.arange(0, $(TILE_N))
    offset = pid_m * $(N) + n_offsets
    input_ptrs = input_ptr + offset
    mask = n_offsets < $(N)
    inp = (tl.load(input_ptrs, mask=mask, other=-float("inf"))).to(output_ptr.dtype.element_ty)
    new_m = tl.maximum(m, inp)
    new_z = tl.exp(m - new_m) * z + tl.exp(inp - new_m)
    m = new_m
    z = new_z
  }
  final_m = tl.max(m, 0)
  z = tl.sum(tl.exp(m - final_m) * z)
  m = final_m

  prev_multiple = $((N + TILE_N - 1)) // $(TILE_N) * $(TILE_N) - $(TILE_N)
  for start_n in range($(0), prev_multiple, $(TILE_N)) {
    n_offsets = start_n + tl.arange(0, $(TILE_N))
    offset = pid_m * $(N) + n_offsets
    input_ptrs = input_ptr + offset
    inp = (tl.load(input_ptrs)).to(output_ptr.dtype.element_ty)
    e = tl.exp(inp - m)
    out = e / z
    output_ptrs = output_ptr + offset
    tl.store(output_ptrs, out)
  }
  for start_n in range(prev_multiple, $(N), $(TILE_N)) {
    n_offsets = start_n + tl.arange(0, $(TILE_N))
    offset = pid_m * $(N) + n_offsets
    input_ptrs = input_ptr + offset
    mask = n_offsets < $(N)
    inp = (tl.load(input_ptrs, mask=mask, other=-float("inf"))).to(output_ptr.dtype.element_ty)
    e = tl.exp(inp - m)
    out = e / z
    output_ptrs = output_ptr + offset
    tl.store(output_ptrs, out, mask=mask)
  }
}

/-- The full online-softmax surface lowers to the algorithm layer, including
the online recurrence loops, masked tail pass, final normalization, and both
writeback loops. -/
theorem softmax_kernel_online_v2_surface_toAlgorithm_supported
    (output_ptr input_ptr : RegionName)
    (M N TILE_N : Nat) :
    ∃ alg,
      (softmax_kernel_online_v2_surface output_ptr input_ptr M N TILE_N).toAlgorithm? =
        Except.ok alg := by
  simp [softmax_kernel_online_v2_surface, ComputeExpr.toAlgorithm?]

/-- Proof-oriented one-tile specialization of `softmax_kernel_online_v2`.

When `N <= TILE_N`, the online loops collapse to one masked row tile. This kernel
is kept as the small executable target for the existing algorithm proof. -/
def softmax_kernel_online_v2_one_tile
    (output_ptr input_ptr : RegionName)
    (N TILE_N : Nat) :
    ComputeKernel := triton {
  pid_m = tl.program_id(0)
  n_offsets = tl.arange(0, $(TILE_N))
  offset = pid_m * $(N) + n_offsets
  mask = n_offsets < $(N)
  input_ptrs = input_ptr + offset
  inp = (tl.load(input_ptrs, mask=mask, other=-float("inf"))).to(output_ptr.dtype.element_ty)
  m = tl.max(inp, 0)
  e = tl.exp(inp - m)
  z = tl.sum(e, 0)
  out = e / z
  output_ptrs = output_ptr + offset
  tl.store(output_ptrs, out, mask=mask)
}

def outOffset (s : BlockState) (N : Nat) (i : Fin TILE_N) : Nat :=
  s.pid * N + i.val

noncomputable def softmaxOptimizeInputTile
    (s : BlockState) (input_ptr : RegionName) (N TILE_N : Nat) :
    Tile .real [TILE_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        some (s.readMem input_ptr (outOffset s N idx.1))
      else none }

noncomputable def softmaxOptimizeSpec
    (s : BlockState) (input_ptr : RegionName)
    (N TILE_N : Nat) (idx : Fin TILE_N) : ℝ :=
  let row := softmaxOptimizeInputTile s input_ptr N TILE_N
  match Tile.reduceMax (shape := [TILE_N]) ⟨0, by simp⟩ Bool.false row with
  | some rowMax =>
      let shifted := Tile.bop (NumericDType.sub .real) Broadcast.scalarR row rowMax
      let e := Tile.uop WithBot.realExp shifted
      let z := Tile.reduceSum (shape := [TILE_N]) ⟨0, by simp⟩ Bool.false e
      WithBot.unbotD 0
        ((Tile.bop (NumericDType.div .real) Broadcast.scalarR e z).data
          (idx, PUnit.unit))
  | none => 0

/-- Algorithm-layer correctness for the one-tile optimized softmax slice. -/
theorem softmax_kernel_online_v2_one_tile_correct
    (output_ptr input_ptr : RegionName)
    (N TILE_N : Nat)
    (s s' : BlockState)
    (hExec : exec (softmax_kernel_online_v2_one_tile output_ptr input_ptr N TILE_N)
        s = some s') :
    ∀ i : Fin TILE_N,
      s'.readMem output_ptr (outOffset s N i) =
        if i.val < N then
          softmaxOptimizeSpec s input_ptr N TILE_N i
        else s.readMem output_ptr (outOffset s N i) := by
  intro i
  by_cases hT : 0 < TILE_N
  · simp [exec, softmax_kernel_online_v2_one_tile, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
          TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
          NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
          ComparableDType.lt, hT] at hExec
    subst s'
    simp [BlockState.pid_eq, outOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : i.val < N
    · simp [hi, softmaxOptimizeSpec, softmaxOptimizeInputTile, outOffset,
            Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum,
            Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
            TileShape.insertAxisIndex, hT]
      congr
    · simp [hi]
  · exact False.elim (hT (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the one-tile optimized softmax slice. -/
theorem softmax_kernel_online_v2_one_tile_compute_correct
    (output_ptr input_ptr : RegionName)
    (N TILE_N : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := softmax_kernel_online_v2_one_tile output_ptr input_ptr N TILE_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin TILE_N => i.val < N)
        (fun i => (output_ptr, outOffset s N i)))
      (expected := fun i => softmaxOptimizeSpec s input_ptr N TILE_N i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [softmax_kernel_online_v2_one_tile]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := softmax_kernel_online_v2_one_tile_correct output_ptr input_ptr N TILE_N
    s s' hExec i
  simpa [hActive] using h

end VeriTile.Bench.TritonBenchG.SoftmaxOptimize
