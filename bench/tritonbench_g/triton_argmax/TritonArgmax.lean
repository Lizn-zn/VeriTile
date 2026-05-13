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
  parameters. -/
def argmax_kernel_1
    (inp mid_value : RegionName) (mid_index : Region .int)
    (M BLOCK_SIZE : Nat) (INT64_INDEX : Bool := Bool.false) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  if INT64_INDEX {
    pid = (pid).to(tl.int64)
  }
  offset = pid * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  inp_ptrs = inp + offset
  mask = offset < $(M)
  inp_val = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
  max_val, local_index := tl.max(inp_val, axis=0, return_indices=True)
  local_index0 = local_index
  max_index = local_index0 + pid * $(BLOCK_SIZE)
  mid_value_ptr = mid_value + pid
  max_index_ptr = $((mid_index : Region .int)) + pid
  tl.store(mid_value_ptr, max_val)
  tl.store(max_index_ptr, max_index)
}

/-- Faithful transcription of `triton_argmax.py`'s `argmax_kernel_2`.

`mid_index` and `out` are typed Int regions matching the launch site's
`torch.int64` buffers, so their `tl.load` / `tl.store` calls do not need
extra `dtype=` kwargs. -/
def argmax_kernel_2
    (mid_value : RegionName) (mid_index out : Region .int)
    (mid_size BLOCK_MID : Nat) :
    ComputeKernel := triton {
  offset = tl.arange(0, $(BLOCK_MID))
  mid_ptrs = mid_value + offset
  mask = offset < $(mid_size)
  mid_val = tl.load(mid_ptrs, mask=mask, other=-float("inf"))
  index_val = tl.argmax(mid_val, axis=0)
  mid_index_ptrs = $((mid_index : Region .int)) + index_val
  out_val = tl.load(mid_index_ptrs)
  tl.store($((out : Region .int)), out_val)
}

/-- Faithful transcription of `triton_argmax.py`'s dim-specific
`argmax_kernel`.

The `out_index` region is typed Int, matching the launch site's `torch.int64`
buffer. -/
def argmax_kernel
    (inp : RegionName) (out_index : Region .int)
    (M N K BLOCK_M BLOCK_N : Nat) (INT64_INDEX : Bool := Bool.false) :
    ComputeKernel := triton {
  pid_m = tl.program_id(0)
  pid_k = tl.program_id(1)
  if INT64_INDEX {
    pid_m = (pid_m).to(tl.int64)
    pid_k = (pid_k).to(tl.int64)
  }
  m_offset = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  max_values = tl.full([$(BLOCK_M)], dtype=tl.float32, value=-inf)
  argmax_values = tl.full([$(BLOCK_M)], dtype=tl.int64, value=$(0))
  for start_n in range($(0), $(N), $(BLOCK_N)) {
    n_offset = start_n + tl.arange(0, $(BLOCK_N))
    offset = m_offset[:, None] * $(N) * $(K) + n_offset[None, :] * $(K) + pid_k
    mask = m_offset[:, None] < $(M) and n_offset[None, :] < $(N)
    inp_ptrs = inp + offset
    inp_vals = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
    local_max, local_argmax := tl.max(inp_vals, 1,
      return_indices=True, return_indices_tie_break_left=True)
    update = local_max > max_values
    max_values = tl.where(update, local_max, max_values)
    argmax_values = tl.where(update, start_n + local_argmax, argmax_values)
  }
  offset_index = m_offset * $(K) + pid_k
  out_index_ptrs = $((out_index : Region .int)) + offset_index
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
    (s : BlockState) (mid_value mid_index : RegionName) (mid_size BLOCK_MID : Nat) : Int :=
  s.readMemValue .int mid_index
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
    simp [BlockState.writeMemTyped_int_readMemValue_int]
  · simp [exec, argmax_kernel_2, stepStmts, stepStmt, evalOp, Option.bind,
          Tile.cop, Tile.ptrAdd, Tile.argMaxDrop, Tile.argBestDrop,
          TileShape.axisDim, TileShape.eraseAxis, ComparableDType.lt,
          argmaxKernel2Spec, argmaxKernel2IndexOffset,
          hB] at hExec ⊢
    cases hExec
    simp

end VeriTile.Bench.TritonBenchG.TritonArgmax
