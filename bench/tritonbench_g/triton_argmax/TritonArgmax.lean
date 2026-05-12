import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.TritonArgmax

open VeriTile.Triton

set_option maxHeartbeats 5000000

/-- Faithful transcription of `triton_argmax.py`'s first-stage
`argmax_kernel_1`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` / `INT64_INDEX: tl.constexpr` -> Lean
  parameters.
- Python's optional `pid.to(tl.int64)` is not emitted as a separate node:
  VeriTile's address/index carrier is `Nat`, and the existing cast erasure for
  index arithmetic keeps this path identical at the algorithm layer.
- Python's `tl.max(..., return_indices=True)` is spelled as paired `tl.max`
  and `tl.argmax` expressions, which produce the same value/index channels in
  the current DSL. -/
def argmax_kernel_1
    (inp mid_value mid_index : RegionName)
    (M BLOCK_SIZE : Nat) (_INT64_INDEX : Bool := Bool.false) :
    ComputeKernel := triton {
  tl.region mid_index = tl.uint64
  pid = tl.program_id(0)
  offset = pid * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  inp_ptrs = inp + offset
  mask = offset < $(M)
  inp_val = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
  max_val = tl.max(inp_val, axis=0)
  local_index = tl.argmax(inp_val, axis=0)
  max_index = local_index + pid * $(BLOCK_SIZE)
  mid_value_ptr = mid_value + pid
  max_index_ptr = mid_index + pid
  tl.store(mid_value_ptr, max_val)
  tl.store(max_index_ptr, max_index)
}

/-- Faithful transcription of `triton_argmax.py`'s `argmax_kernel_2`.

`mid_index` and `out` carry int64 element data in the launch site
(`torch.int64` buffers); the in-body `tl.region` directive declares
the element dtype so `tl.load(mid_index_ptrs)` recovers `.nat`
without an explicit `dtype=` kwarg. The directive itself is metadata
— it is stripped from the emitted kernel body. -/
def argmax_kernel_2
    (mid_value mid_index out : RegionName)
    (mid_size BLOCK_MID : Nat) :
    ComputeKernel := triton {
  tl.region mid_index = tl.uint64, out = tl.uint64
  offset = tl.arange(0, $(BLOCK_MID))
  mid_ptrs = mid_value + offset
  mask = offset < $(mid_size)
  mid_val = tl.load(mid_ptrs, mask=mask, other=-float("inf"))
  index_val = tl.argmax(mid_val, axis=0)
  mid_index_ptrs = mid_index + index_val
  out_val = tl.load(mid_index_ptrs)
  tl.store(out, out_val)
}

/-- Faithful transcription of `triton_argmax.py`'s dim-specific
`argmax_kernel`.

The Python kernel accumulates only argmax indices for each `(m, k)` lane.
The `INT64_INDEX` program-id casts are erased for the same reason as in
`argmax_kernel_1`: VeriTile's index carrier is already `Nat`.

Known remaining dtype gap: Python initializes `max_values` with
`dtype=tl.float32`; the current DSL cannot compose a compute-only fp32 `tl.full`
through the later comparison/update, so this surface keeps the algorithm-layer
Real initializer while preserving the control/dataflow shape. Python also
initializes `argmax_values` with `dtype=tl.int64`; this is represented by the
Nat/index carrier expression `tl.arange(...) * 0` plus the `out_index = tl.uint64`
region declaration on the store target. -/
def argmax_kernel
    (inp out_index : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat) (_INT64_INDEX : Bool := Bool.false) :
    ComputeKernel := triton {
  tl.region out_index = tl.uint64
  pid_m = tl.program_id(0)
  pid_k = tl.program_id(1)
  m_offset = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  max_values = tl.full([$(BLOCK_M)], -inf)
  argmax_values = tl.arange(0, $(BLOCK_M)) * $(0)
  for start_n in range($(0), $(N), $(BLOCK_N)) {
    n_offset = start_n + tl.arange(0, $(BLOCK_N))
    offset = m_offset[:, None] * $(N) * $(K) + n_offset[None, :] * $(K) + pid_k
    mask = m_offset[:, None] < $(M) and n_offset[None, :] < $(N)
    inp_ptrs = inp + offset
    inp_vals = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
    local_max, local_argmax := tl.max(inp_vals, axis=1, return_indices=True)
    update = local_max > max_values
    max_values = tl.where(update, local_max, max_values)
    argmax_values = tl.where(update, start_n + local_argmax, argmax_values)
  }
  offset_index = m_offset * $(K) + pid_k
  out_index_ptrs = out_index + offset_index
  mask1 = m_offset < $(M)
  tl.store(out_index_ptrs, argmax_values, mask=mask1)
}

noncomputable def argmaxKernel2InputTile
    (s : BlockState) (mid_value : RegionName) (mid_size BLOCK_MID : Nat) :
    Tile .real [BLOCK_MID] :=
  { data := fun idx =>
      if idx.1.val < mid_size then some (s.readMem mid_value idx.1.val) else none }

noncomputable def argmaxKernel2IndexOffset
    (s : BlockState) (mid_value : RegionName) (mid_size BLOCK_MID : Nat) : Nat :=
  (Tile.argMaxDrop (shape := [BLOCK_MID]) ⟨0, by simp⟩
    (argmaxKernel2InputTile s mid_value mid_size BLOCK_MID)).data PUnit.unit

noncomputable def argmaxKernel2Spec
    (s : BlockState) (mid_value mid_index : RegionName) (mid_size BLOCK_MID : Nat) : Nat :=
  s.readMemValue .nat mid_index
    (argmaxKernel2IndexOffset s mid_value mid_size BLOCK_MID)

/-- Compute-facing correctness for `argmax_kernel_2`. -/
theorem argmax_kernel_2_compute_correct
    (mid_value mid_index out : RegionName)
    (mid_size BLOCK_MID : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := argmax_kernel_2 mid_value mid_index out mid_size BLOCK_MID)
      (initialState := s)
      (write := fun _ : PUnit => some (out, 0))
      (expected := fun _ => argmaxKernel2Spec s mid_value mid_index mid_size BLOCK_MID) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [argmax_kernel_2]
  intro s0 s' hExec hs0
  subst s0
  intro _
  by_cases hB : 0 < BLOCK_MID
  · simp [exec, argmax_kernel_2, stepStmts, stepStmt, evalOp, Option.bind,
          Tile.cop, Tile.ptrAdd, Tile.argMaxDrop, Tile.argBestDrop,
          TileShape.axisDim, TileShape.eraseAxis, ComparableDType.lt,
          argmaxKernel2Spec, argmaxKernel2IndexOffset,
          argmaxKernel2InputTile, hB] at hExec ⊢
    cases hExec
    simp [BlockState.writeMemTyped_nat_readMemValue_nat]
  · simp [exec, argmax_kernel_2, stepStmts, stepStmt, evalOp, Option.bind,
          Tile.cop, Tile.ptrAdd, Tile.argMaxDrop, Tile.argBestDrop,
          TileShape.axisDim, TileShape.eraseAxis, ComparableDType.lt,
          argmaxKernel2Spec, argmaxKernel2IndexOffset,
          hB] at hExec ⊢
    cases hExec
    simp [BlockState.writeMemTyped_nat_readMemValue_nat]

end VeriTile.Bench.TritonBenchG.TritonArgmax
