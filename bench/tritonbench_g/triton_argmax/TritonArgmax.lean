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
  max_val, max_index = tl.max(inp_val, axis=0, return_indices=True)
  max_index = max_index + pid * $(BLOCK_SIZE)
  mid_value_ptr = mid_value + pid
  max_index_ptr = mid_index + pid
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
  mid_index_ptrs = mid_index + index_val
  out_val = tl.load(mid_index_ptrs)
  tl.store(out, out_val)
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
  max_values = tl.full([$(BLOCK_M)], dtype=tl.float32, value=float("-inf"))
  argmax_values = tl.full([$(BLOCK_M)], dtype=tl.int64, value=0)
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
  out_index_ptrs = out_index + offset_index
  mask1 = m_offset < $(M)
  tl.store(out_index_ptrs, argmax_values, mask=mask1)
}

/-- Masked input tile for `argmax_kernel_1`. Masked lanes evaluate to `none`
matching `other=-float("inf")`. -/
noncomputable def argmaxKernel1InputTile
    (s : BlockState) (inp : RegionName) (M BLOCK_SIZE : Nat) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      let off := s.pid * BLOCK_SIZE + idx.1.val
      if off < M then some (s.readMem inp off) else none }

/-- Exact `max_val` written by `argmax_kernel_1` at lane `pid`. -/
noncomputable def argmaxKernel1ValueSpec
    (s : BlockState) (inp : RegionName) (M BLOCK_SIZE : Nat) : ℝ :=
  match Tile.reduceMax (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false
      (argmaxKernel1InputTile s inp M BLOCK_SIZE) with
  | some out => WithBot.unbotD 0 (out.data PUnit.unit)
  | none => 0

/-- Exact `max_index` (after the `+ pid * BLOCK_SIZE` shift) written by
`argmax_kernel_1` at lane `pid`. -/
noncomputable def argmaxKernel1IndexSpec
    (s : BlockState) (inp : RegionName) (M BLOCK_SIZE : Nat) : Nat :=
  (Tile.argMaxDrop (shape := [BLOCK_SIZE]) ⟨0, by simp⟩
    (argmaxKernel1InputTile s inp M BLOCK_SIZE)).data PUnit.unit
    + s.pid * BLOCK_SIZE

/-- Compute-facing correctness for the value channel of `argmax_kernel_1`
(`INT64_INDEX=false` branch). The hypothesis `hRegions` rules out aliasing
between the real `mid_value` buffer and the int `mid_index` buffer at the same
offset; without it the `.nat` index store could overwrite the `.real` value
store. -/
theorem argmax_kernel_1_value_compute_correct
    (inp mid_value : RegionName) (mid_index : Region .int)
    (M BLOCK_SIZE : Nat)
    (s : BlockState)
    (hRegions : mid_value ≠ (Region.cast mid_index : RegionName)) :
    ComputeCorrect.Realizes
      (kernel := argmax_kernel_1 inp mid_value mid_index M BLOCK_SIZE Bool.false)
      (initialState := s)
      (write := fun _ : PUnit => some (mid_value, s.pid))
      (expected := fun _ => argmaxKernel1ValueSpec s inp M BLOCK_SIZE) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [argmax_kernel_1]
  intro s0 s' hExec hs0
  subst s0
  intro _
  by_cases hB : 0 < BLOCK_SIZE
  · simp [exec, argmax_kernel_1, stepStmts, stepStmt, evalOp, Option.bind,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.reduceMax, Tile.reduceMaxDrop,
          TileShape.axisDim, TileShape.eraseAxis,
          NumericDType.add, NumericDType.mul, ComparableDType.lt,
          argmaxKernel1ValueSpec, argmaxKernel1InputTile, hB] at hExec ⊢
    cases hExec
    rw [BlockState.writeMemTyped_nat_readMem_of_ne _ _ _ _ _ _ (by
      intro ⟨h1, _⟩
      exact hRegions h1)]
    simp
    congr
  · simp [exec, argmax_kernel_1, stepStmts, stepStmt, evalOp, Option.bind,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.reduceMax, Tile.reduceMaxDrop,
          TileShape.axisDim, TileShape.eraseAxis,
          NumericDType.add, NumericDType.mul, ComparableDType.lt, hB] at hExec

/-- Compute-facing correctness for the index channel of `argmax_kernel_1`
(`INT64_INDEX=false` branch). -/
theorem argmax_kernel_1_index_compute_correct
    (inp mid_value : RegionName) (mid_index : Region .int)
    (M BLOCK_SIZE : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := argmax_kernel_1 inp mid_value mid_index M BLOCK_SIZE Bool.false)
      (initialState := s)
      (write := fun _ : PUnit =>
        some ((Region.cast mid_index : RegionName), s.pid))
      (expected :=
        fun _ : PUnit =>
          (argmaxKernel1IndexSpec s inp M BLOCK_SIZE : Nat)) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [argmax_kernel_1]
  intro s0 s' hExec hs0
  subst s0
  intro _
  by_cases hB : 0 < BLOCK_SIZE
  · simp [exec, argmax_kernel_1, stepStmts, stepStmt, evalOp, Option.bind,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.reduceMax, Tile.reduceMaxDrop,
          Tile.argMaxDrop, Tile.argBestDrop,
          TileShape.axisDim, TileShape.eraseAxis,
          NumericDType.add, NumericDType.mul, ComparableDType.lt,
          argmaxKernel1IndexSpec, argmaxKernel1InputTile, hB] at hExec ⊢
    cases hExec
    simp [BlockState.writeMemTyped_nat_readMemValue_nat]
  · simp [exec, argmax_kernel_1, stepStmts, stepStmt, evalOp, Option.bind,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.reduceMax, Tile.reduceMaxDrop,
          TileShape.axisDim, TileShape.eraseAxis,
          NumericDType.add, NumericDType.mul, ComparableDType.lt, hB] at hExec

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

/-! ## Dim-specific `argmax_kernel` — single-block (N ≤ BLOCK_N) regime

The Python test only drives `argmax_kernel` with `N ≤ BLOCK_N`. Under that
regime the `for start_n in range(0, N, BLOCK_N)` loop runs exactly once
at `start_n = 0`. We collapse the loop with `forRangeDyn_single_step` and
apply a local int-channel scatter-readback to the final argmax store. -/

/-- Local int-channel analogue of `scatter_readback_nat_prop_masked_nd`.
The dim-specific `argmax_kernel` stores into an `.int`-typed `out_index`
region; the upstream `Semantics/State.lean` only provides nat/real
scatter-readback combinators. This helper folds along
`TileShape.allIndices shape` and bridges `writeMemTyped .int` writes to
the `readMemValue .int` readback at any element. -/
private theorem scatter_readback_int_prop_masked_nd
    {region : RegionName} {shape : TileShape}
    (s : BlockState) (offsetFn : TileIndex shape → Nat)
    (valueFn : TileIndex shape → Int) (P : TileIndex shape → Prop) [DecidablePred P]
    (h_inj : Function.Injective offsetFn) (i : TileIndex shape) :
    BlockState.readMemValue ((TileShape.allIndices shape).foldl
       (fun acc k =>
         if P k then acc.writeMemTyped .int region (offsetFn k) (valueFn k) else acc) s)
      .int region (offsetFn i)
    = if P i then valueFn i else s.readMemValue .int region (offsetFn i) := by
  -- Mirrors `scatter_readback_nat_prop_masked_nd` for the `.int` channel.
  have h_preserve :
      ∀ (l : List (TileIndex shape)) (s' : BlockState),
        (∀ k ∈ l, P k → offsetFn k ≠ offsetFn i) →
        BlockState.readMemValue ((l.foldl
            (fun acc k =>
              if P k then acc.writeMemTyped .int region (offsetFn k) (valueFn k) else acc) s'))
            .int region (offsetFn i)
          = s'.readMemValue .int region (offsetFn i) := by
    intro l
    induction l with
    | nil => intros; rfl
    | cons hd tl ih =>
        intro s' h
        rw [List.foldl_cons]
        have htl : ∀ k ∈ tl, P k → offsetFn k ≠ offsetFn i :=
          fun k hk hPk => h k (List.mem_cons_of_mem hd hk) hPk
        by_cases hPhd : P hd
        · have hhd : offsetFn hd ≠ offsetFn i :=
            h hd List.mem_cons_self hPhd
          simp only [hPhd, if_true]
          rw [ih _ htl]
          rw [BlockState.writeMemTyped_int_readMemValue_int]
          show (if region = region ∧ offsetFn i = offsetFn hd then valueFn hd
              else s'.readMemValue TileDType.int region (offsetFn i)) =
            s'.readMemValue TileDType.int region (offsetFn i)
          rw [if_neg]
          rintro ⟨_, h_eq⟩
          exact hhd h_eq.symm
        · simp only [hPhd, if_false]
          exact ih _ htl
  let l := TileShape.allIndices shape
  obtain ⟨l₁, l₂, hl⟩ := List.append_of_mem (TileShape.mem_allIndices shape i)
  have h_nodup := TileShape.allIndices_nodup shape
  change BlockState.readMemValue ((l.foldl
       (fun acc k =>
         if P k then acc.writeMemTyped .int region (offsetFn k) (valueFn k) else acc) s))
      .int region (offsetFn i)
    = if P i then valueFn i else s.readMemValue .int region (offsetFn i)
  rw [hl] at h_nodup
  rw [List.nodup_append, List.nodup_cons] at h_nodup
  obtain ⟨_, ⟨hi_notin_l2, _⟩, hl1_disj⟩ := h_nodup
  have hl' : l = l₁ ++ i :: l₂ := by simpa [l] using hl
  rw [hl', List.foldl_append, List.foldl_cons]
  have h_l1_not_in : ∀ k ∈ l₁, P k → offsetFn k ≠ offsetFn i := by
    intro k hk _ heq
    have hki : k = i := h_inj heq
    rw [hki] at hk
    exact (hl1_disj i hk i List.mem_cons_self) rfl
  have h_l2_not_in : ∀ k ∈ l₂, P k → offsetFn k ≠ offsetFn i := by
    intro k hk _ heq
    have hki : k = i := h_inj heq
    subst hki
    exact hi_notin_l2 hk
  rw [h_preserve l₂ _ h_l2_not_in]
  by_cases hPi : P i
  · simp only [hPi, if_true]
    rw [BlockState.writeMemTyped_int_readMemValue_int]
    show (if region = region ∧ offsetFn i = offsetFn i then valueFn i else _) = valueFn i
    exact if_pos ⟨rfl, rfl⟩
  · simp only [hPi, if_false]
    rw [h_preserve l₁ _ h_l1_not_in]

/-- The 2D `[BLOCK_M, BLOCK_N]` input tile loaded inside the `argmax_kernel`
loop body at `start_n = 0`. Masked-out lanes evaluate to `none` matching
`tl.load(..., mask=..., other=-float("inf"))`. -/
noncomputable def argmaxKernelInputTile
    (s : BlockState) (inp : RegionName) (M N K BLOCK_M BLOCK_N : Nat) :
    Tile .real [BLOCK_M, BLOCK_N] :=
  { data := fun idx =>
      let m := idx.1.val
      let n := idx.2.1.val
      let mOff := s.pids 0 * BLOCK_M + m
      let off := mOff * N * K + n * K + s.pids 1
      if mOff < M ∧ n < N then some (s.readMem inp off) else none }

/-- Per-row argmax over the 2D input tile, dropping axis 1 (matching
`tl.max(inp_vals, 1, return_indices=True, return_indices_tie_break_left=True)`). -/
noncomputable def argmaxKernelArgmaxSpec
    (s : BlockState) (inp : RegionName) (M N K BLOCK_M BLOCK_N : Nat)
    (i : Fin BLOCK_M) : Int :=
  Int.ofNat
    ((Tile.argMaxDrop (shape := [BLOCK_M, BLOCK_N]) ⟨1, by simp⟩
      (argmaxKernelInputTile s inp M N K BLOCK_M BLOCK_N)).data (i, PUnit.unit))

/-- Stub: full argmax_kernel single-block proof requires bridging the inner
loop's stepForRangeAux residual to the int-channel scatter-readback. Spec
defs (`argmaxKernelInputTile`, `argmaxKernelArgmaxSpec`) are in place above;
final scatter-readback bridge is the remaining gap. See memory v6. -/

end VeriTile.Bench.TritonBenchG.TritonArgmax
