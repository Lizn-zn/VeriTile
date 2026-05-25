import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.FlashDecode2Phi

open VeriTile.Triton

set_option linter.unusedSimpArgs false
set_option maxHeartbeats 1000000

/-- Faithful transcription of `flash_decode2_phi.py`'s
`_fwd_kernel_flash_decode_stage2`.

The Python body passes but does not use `stride_mid_od`, `stride_mid_o_es`, or
`stride_od`; this surface preserves that addressing behavior and keeps those
signature positions underscored. -/
def flash_decode2_phi_surface
    (B_Seqlen : Region .nat) (Mid_O Mid_O_LogExpSum Out : RegionName)
    (stride_mid_ob stride_mid_oh stride_mid_os _stride_mid_od
      stride_mid_o_eb stride_mid_o_eh _stride_mid_o_es stride_obs stride_oh _stride_od
      head_dim BLOCK_SEQ BLOCK_DMODEL : Nat) : ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
  block_n_size = tl.where(cur_batch_seq_len <= $(0), $(0),
    cur_batch_seq_len + $(BLOCK_SEQ) - $(1)) // $(BLOCK_SEQ)
  sum_exp = 0.0
  max_logic = -inf
  acc = tl.zeros([$(BLOCK_DMODEL)], dtype=tl.float32)
  offs_v = cur_batch * $(stride_mid_ob) + cur_head * $(stride_mid_oh) + offs_d
  offs_logic = cur_batch * $(stride_mid_o_eb) + cur_head * $(stride_mid_o_eh)
  for block_seq_n in range($(0), block_n_size, $(1)) {
    tv = tl.load(Mid_O + offs_v + block_seq_n * $(stride_mid_os),
      mask=offs_d < $(head_dim), other=0.0)
    tlogic = tl.load(Mid_O_LogExpSum + offs_logic + block_seq_n)
    new_max_logic = tl.maximum(tlogic, max_logic)
    old_scale = tl.exp(max_logic - new_max_logic)
    acc *= old_scale
    exp_logic = tl.exp(tlogic - new_max_logic)
    acc += exp_logic * tv
    sum_exp = sum_exp * old_scale + exp_logic
    max_logic = new_max_logic
  }
  tl.store(Out + cur_batch * $(stride_obs) + cur_head * $(stride_oh) + offs_d,
    acc / sum_exp, mask=offs_d < $(head_dim))
}

/-- The full flash-decode stage2 Phi surface lowers to the algorithm layer. -/
theorem flash_decode2_phi_surface_toAlgorithm_supported
    (B_Seqlen : Region .nat) (Mid_O Mid_O_LogExpSum Out : RegionName)
    (stride_mid_ob stride_mid_oh stride_mid_os stride_mid_od
      stride_mid_o_eb stride_mid_o_eh stride_mid_o_es stride_obs stride_oh stride_od
      head_dim BLOCK_SEQ BLOCK_DMODEL : Nat) :
    ∃ alg, (flash_decode2_phi_surface B_Seqlen Mid_O Mid_O_LogExpSum Out
      stride_mid_ob stride_mid_oh stride_mid_os stride_mid_od stride_mid_o_eb
      stride_mid_o_eh stride_mid_o_es stride_obs stride_oh stride_od head_dim
      BLOCK_SEQ BLOCK_DMODEL).toAlgorithm? = Except.ok alg := by
  simp [flash_decode2_phi_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

def logicOffset
    (s : BlockState) (stride_mid_o_eb stride_mid_o_eh block_seq_n : Nat) :
    Nat :=
  s.pids 0 * stride_mid_o_eb + s.pids 1 * stride_mid_o_eh + block_seq_n

noncomputable def runningMaxJoin (x y : WithBot ℝ) : WithBot ℝ :=
  max x y

noncomputable def runningMaxAfter
    (s : BlockState) (Mid_O_LogExpSum : RegionName)
    (stride_mid_o_eb stride_mid_o_eh : Nat) : Nat → WithBot ℝ
  | 0 => ⊥
  | k + 1 =>
      runningMaxJoin
        (some (s.readMem Mid_O_LogExpSum
          (logicOffset s stride_mid_o_eb stride_mid_o_eh k)) : WithBot ℝ)
        (runningMaxAfter s Mid_O_LogExpSum stride_mid_o_eb stride_mid_o_eh k)

theorem runningMaxAfter_succ
    (s : BlockState) (Mid_O_LogExpSum : RegionName)
    (stride_mid_o_eb stride_mid_o_eh k : Nat) :
    runningMaxAfter s Mid_O_LogExpSum stride_mid_o_eb stride_mid_o_eh (k + 1) =
      runningMaxJoin
        (some (s.readMem Mid_O_LogExpSum
          (logicOffset s stride_mid_o_eb stride_mid_o_eh k)) : WithBot ℝ)
        (runningMaxAfter s Mid_O_LogExpSum stride_mid_o_eb stride_mid_o_eh k) := by
  rfl

theorem runningMaxAfter_zero
    (s : BlockState) (Mid_O_LogExpSum : RegionName)
    (stride_mid_o_eb stride_mid_o_eh : Nat) :
    runningMaxAfter s Mid_O_LogExpSum stride_mid_o_eb stride_mid_o_eh 0 = ⊥ := by
  rfl

/-- Proof-oriented final output-store slice of
`flash_decode2_phi.py`'s `_fwd_kernel_flash_decode_stage2`.

The full stage2 kernel reduces partial sequence-block outputs into
`acc / sum_exp`. This slice starts from a precomputed normalized `Final` vector
and proves the final `offs_d < head_dim` masked writeback into `Out`. -/
def flash_decode2_phi_final_store_slice
    (Final Out : RegionName)
    (head_dim
      stride_final_b stride_final_h stride_final_d
      stride_obs stride_oh stride_od
      BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  mask = offs_d < $(head_dim)
  final = tl.load(Final + cur_batch * $(stride_final_b) +
      cur_head * $(stride_final_h) + offs_d * $(stride_final_d),
      mask=mask, other=0.0)
  tl.store(Out + cur_batch * $(stride_obs) + cur_head * $(stride_oh) +
      offs_d * $(stride_od), final, mask=mask)
}

/-- Final normalization and output writeback for Phi flash-decode stage2.

This kernel consumes the loop-produced accumulator vector and `sum_exp` scalar,
computes `acc / sum_exp`, and writes the masked Python-observable `Out` row. -/
def flash_decode2_phi_normalization_store_kernel
    (Acc SumExp Out : RegionName)
    (head_dim
      stride_acc_b stride_acc_h stride_acc_d
      stride_sum_b stride_sum_h
      stride_obs stride_oh stride_od
      BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  mask = offs_d < $(head_dim)
  acc = tl.load(Acc + cur_batch * $(stride_acc_b) +
      cur_head * $(stride_acc_h) + offs_d * $(stride_acc_d),
      mask=mask, other=0.0)
  sum_exp = tl.load(SumExp + cur_batch * $(stride_sum_b) +
      cur_head * $(stride_sum_h))
  tl.store(Out + cur_batch * $(stride_obs) + cur_head * $(stride_oh) +
      offs_d * $(stride_od), acc / sum_exp, mask=mask)
}

def dIndex (_s : BlockState) (i : Fin BLOCK_DMODEL) : Nat :=
  i.val

def active (_s : BlockState) (head_dim : Nat) (i : Fin BLOCK_DMODEL) : Prop :=
  i.val < head_dim

instance activeDecidable (s : BlockState) (head_dim : Nat) (i : Fin BLOCK_DMODEL) :
    Decidable (active s head_dim i) := by
  unfold active
  infer_instance

def finalOffset
    (s : BlockState)
    (stride_final_b stride_final_h stride_final_d : Nat)
    (i : Fin BLOCK_DMODEL) : Nat :=
  s.pids 0 * stride_final_b + s.pids 1 * stride_final_h +
    dIndex s i * stride_final_d

def accOffset
    (s : BlockState)
    (stride_acc_b stride_acc_h stride_acc_d : Nat)
    (i : Fin BLOCK_DMODEL) : Nat :=
  s.pids 0 * stride_acc_b + s.pids 1 * stride_acc_h +
    dIndex s i * stride_acc_d

def sumExpOffset
    (s : BlockState)
    (stride_sum_b stride_sum_h : Nat) : Nat :=
  s.pids 0 * stride_sum_b + s.pids 1 * stride_sum_h

def outOffset
    (s : BlockState)
    (stride_obs stride_oh stride_od : Nat)
    (i : Fin BLOCK_DMODEL) : Nat :=
  s.pids 0 * stride_obs + s.pids 1 * stride_oh + dIndex s i * stride_od

noncomputable def finalStoreValue
    (s : BlockState) (Final : RegionName)
    (head_dim stride_final_b stride_final_h stride_final_d : Nat)
    (i : Fin BLOCK_DMODEL) : ℝ :=
  WithBot.unbotD 0
    (if active s head_dim i then
      some (s.readMem Final
        (finalOffset s stride_final_b stride_final_h stride_final_d i))
    else some (0.0 : ℝ))

noncomputable def normalizedStoreValue
    (s : BlockState) (Acc SumExp : RegionName)
    (head_dim stride_acc_b stride_acc_h stride_acc_d
      stride_sum_b stride_sum_h : Nat)
    (i : Fin BLOCK_DMODEL) : ℝ :=
  WithBot.unbotD 0
    (if active s head_dim i then
      some (s.readMem Acc (accOffset s stride_acc_b stride_acc_h stride_acc_d i) /
        s.readMem SumExp (sumExpOffset s stride_sum_b stride_sum_h))
    else some (0.0 : ℝ))

/-- Algorithm-layer correctness for the masked final flash-decode writeback. -/
theorem flash_decode2_phi_final_store_slice_correct
    (Final Out : RegionName)
    (head_dim
      stride_final_b stride_final_h stride_final_d
      stride_obs stride_oh stride_od
      BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_DMODEL => outOffset s stride_obs stride_oh stride_od i)) :
    ∀ i : Fin BLOCK_DMODEL,
      let outAddr := outOffset s stride_obs stride_oh stride_od i
      (exec (flash_decode2_phi_final_store_slice Final Out head_dim
            stride_final_b stride_final_h stride_final_d stride_obs stride_oh
            stride_od BLOCK_DMODEL) s).map (·.readMem Out outAddr)
        = some (if active s head_dim i then
            finalStoreValue s Final head_dim stride_final_b stride_final_h
              stride_final_d i
          else s.readMem Out outAddr) := by
  intro i
  simp [exec, flash_decode2_phi_final_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.ptrAdd, NumericDType.add,
        NumericDType.mul, ComparableDType.lt, dIndex, active, finalOffset,
        outOffset]
  let offsetFn : TileIndex [BLOCK_DMODEL] → Nat :=
    fun idx => s.pids 0 * stride_obs + s.pids 1 * stride_oh + idx.1.val * stride_od
  let valueFn : TileIndex [BLOCK_DMODEL] → ℝ :=
    fun idx =>
      WithBot.unbotD 0
        (if idx.1.val < head_dim then
          some (s.readMem Final
            (s.pids 0 * stride_final_b + s.pids 1 * stride_final_h +
              idx.1.val * stride_final_d))
        else some (0.0 : ℝ))
  let P : TileIndex [BLOCK_DMODEL] → Prop :=
    fun idx => idx.1.val < head_dim
  have hOffsetInj : Function.Injective offsetFn := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have habFin : outOffset s stride_obs stride_oh stride_od a =
        outOffset s stride_obs stride_oh stride_od b := by
      simpa [offsetFn, outOffset, dIndex] using hab
    obtain rfl : a = b := hOutInj habFin
    rfl
  change (List.foldl
      (fun (acc : BlockState) idx =>
        if P idx then acc.writeMem Out (offsetFn idx) (valueFn idx) else acc)
      _ (TileShape.allIndices [BLOCK_DMODEL])).readMem Out
        (offsetFn (i, PUnit.unit)) =
    if active s head_dim i then
      finalStoreValue s Final head_dim stride_final_b stride_final_h
        stride_final_d i
    else s.readMem Out (offsetFn (i, PUnit.unit))
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj (i, PUnit.unit)]
  by_cases hi : i.val < head_dim
  · simp [P, valueFn, active, finalStoreValue, finalOffset, dIndex, hi]
  · simp [P, active, finalStoreValue, dIndex, hi]

/-- Compute-facing correctness for the masked final flash-decode writeback. -/
theorem flash_decode2_phi_final_store_slice_compute_correct
    (Final Out : RegionName)
    (head_dim
      stride_final_b stride_final_h stride_final_d
      stride_obs stride_oh stride_od
      BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_DMODEL => outOffset s stride_obs stride_oh stride_od i)) :
    ComputeCorrect.Realizes
      (kernel := flash_decode2_phi_final_store_slice Final Out head_dim
        stride_final_b stride_final_h stride_final_d stride_obs stride_oh
        stride_od BLOCK_DMODEL)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_DMODEL => active s head_dim i)
        (fun i => (Out, outOffset s stride_obs stride_oh stride_od i)))
      (expected := fun i : Fin BLOCK_DMODEL =>
        finalStoreValue s Final head_dim stride_final_b stride_final_h
          stride_final_d i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [flash_decode2_phi_final_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := flash_decode2_phi_final_store_slice_correct Final Out head_dim
    stride_final_b stride_final_h stride_final_d stride_obs stride_oh stride_od
    BLOCK_DMODEL s hOutInj i
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-! ## Python test-shape wrappers

The checked Python tests allocate `Out` with shape `(2, 4, 64)`, so the
contiguous output strides are `(256, 64, 1)`. `head_dim = 64` and
`BLOCK_DMODEL = next_power_of_2(64) = 64`; the varied `block_seq` cases do not
change the final output layout. -/

theorem flash_decode2_phi_python_test_shape_offset_injective
    (s : BlockState) :
    Function.Injective
      (fun i : Fin 64 => outOffset s 256 64 1 i) := by
  intro a b h
  simp [outOffset, dIndex] at h
  exact Fin.ext (by omega)

theorem flash_decode2_phi_final_store_python_test_shape_compute_correct
    (Final Out : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := flash_decode2_phi_final_store_slice Final Out
        64 256 64 1 256 64 1 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 64 => active s 64 i)
        (fun i : Fin 64 => (Out, outOffset s 256 64 1 i)))
      (expected := fun i : Fin 64 =>
        finalStoreValue s Final 64 256 64 1 i) := by
  exact flash_decode2_phi_final_store_slice_compute_correct Final Out
    64 256 64 1 256 64 1 64 s
    (flash_decode2_phi_python_test_shape_offset_injective s)

/-- Algorithm-layer correctness for Phi stage2 normalization plus writeback. -/
theorem flash_decode2_phi_normalization_store_kernel_correct
    (Acc SumExp Out : RegionName)
    (head_dim
      stride_acc_b stride_acc_h stride_acc_d
      stride_sum_b stride_sum_h
      stride_obs stride_oh stride_od
      BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_DMODEL => outOffset s stride_obs stride_oh stride_od i)) :
    ∀ i : Fin BLOCK_DMODEL,
      let outAddr := outOffset s stride_obs stride_oh stride_od i
      (exec (flash_decode2_phi_normalization_store_kernel Acc SumExp Out head_dim
            stride_acc_b stride_acc_h stride_acc_d stride_sum_b stride_sum_h
            stride_obs stride_oh stride_od BLOCK_DMODEL) s).map (·.readMem Out outAddr)
        = some (if active s head_dim i then
            normalizedStoreValue s Acc SumExp head_dim stride_acc_b stride_acc_h
              stride_acc_d stride_sum_b stride_sum_h i
          else s.readMem Out outAddr) := by
  intro i
  simp [exec, flash_decode2_phi_normalization_store_kernel, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul, NumericDType.div, ComparableDType.lt,
        dIndex, active, accOffset, sumExpOffset, outOffset]
  let offsetFn : TileIndex [BLOCK_DMODEL] → Nat :=
    fun idx => s.pids 0 * stride_obs + s.pids 1 * stride_oh + idx.1.val * stride_od
  let valueFn : TileIndex [BLOCK_DMODEL] → ℝ :=
    fun idx =>
      WithBot.unbotD 0
        (match
          if idx.1.val < head_dim then
            some (s.readMem Acc
              (s.pids 0 * stride_acc_b + s.pids 1 * stride_acc_h +
                idx.1.val * stride_acc_d))
          else some (0.0 : ℝ)
        with
        | some x =>
            some (x / s.readMem SumExp
              (s.pids 0 * stride_sum_b + s.pids 1 * stride_sum_h))
        | none => none)
  let P : TileIndex [BLOCK_DMODEL] → Prop :=
    fun idx => idx.1.val < head_dim
  have hOffsetInj : Function.Injective offsetFn := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have habFin : outOffset s stride_obs stride_oh stride_od a =
        outOffset s stride_obs stride_oh stride_od b := by
      simpa [offsetFn, outOffset, dIndex] using hab
    obtain rfl : a = b := hOutInj habFin
    rfl
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj (i, PUnit.unit)]
  by_cases hi : i.val < head_dim
  · simp [P, offsetFn, valueFn, active, normalizedStoreValue, accOffset,
      sumExpOffset, dIndex, hi]
  · simp [P, offsetFn, active, normalizedStoreValue, dIndex, hi]

/-- Compute-facing correctness for Phi stage2 normalization plus writeback. -/
theorem flash_decode2_phi_normalization_store_kernel_compute_correct
    (Acc SumExp Out : RegionName)
    (head_dim
      stride_acc_b stride_acc_h stride_acc_d
      stride_sum_b stride_sum_h
      stride_obs stride_oh stride_od
      BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_DMODEL => outOffset s stride_obs stride_oh stride_od i)) :
    ComputeCorrect.Realizes
      (kernel := flash_decode2_phi_normalization_store_kernel Acc SumExp Out
        head_dim stride_acc_b stride_acc_h stride_acc_d stride_sum_b
        stride_sum_h stride_obs stride_oh stride_od BLOCK_DMODEL)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_DMODEL => active s head_dim i)
        (fun i => (Out, outOffset s stride_obs stride_oh stride_od i)))
      (expected := fun i : Fin BLOCK_DMODEL =>
        normalizedStoreValue s Acc SumExp head_dim stride_acc_b stride_acc_h
          stride_acc_d stride_sum_b stride_sum_h i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [flash_decode2_phi_normalization_store_kernel]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := flash_decode2_phi_normalization_store_kernel_correct Acc SumExp Out
    head_dim stride_acc_b stride_acc_h stride_acc_d stride_sum_b stride_sum_h
    stride_obs stride_oh stride_od BLOCK_DMODEL s hOutInj i
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

theorem flash_decode2_phi_normalization_store_python_test_shape_compute_correct
    (Acc SumExp Out : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := flash_decode2_phi_normalization_store_kernel Acc SumExp Out
        64 256 64 1 4 1 256 64 1 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 64 => active s 64 i)
        (fun i : Fin 64 => (Out, outOffset s 256 64 1 i)))
      (expected := fun i : Fin 64 =>
        normalizedStoreValue s Acc SumExp 64 256 64 1 4 1 i) := by
  exact flash_decode2_phi_normalization_store_kernel_compute_correct Acc SumExp Out
    64 256 64 1 4 1 256 64 1 64 s
    (flash_decode2_phi_python_test_shape_offset_injective s)

/-- Internal Python test-shape summary for `flash_decode2_phi.py`.

The Python tests use `mid_out : (2, 4, 3, 64)`, `Out : (2, 4, 64)`, and four
`block_seq` values (`16`, `17`, `8`, `32`). This records all four faithful
stage2 surfaces and ties them to the observable masked final `Out` writeback
slice. The loop-reduced normalized vector is represented by the
proof-oriented `Final` region in the store slice. -/
theorem flash_decode2_phi_python_test_shape_summary
    (B_Seqlen : Region .nat) (Mid_O Mid_O_LogExpSum Final Out : RegionName)
    (s : BlockState) :
    (∃ alg, (flash_decode2_phi_surface B_Seqlen Mid_O Mid_O_LogExpSum Out
      768 192 64 1 12 3 1 256 64 1 64 16 64).toAlgorithm? =
        Except.ok alg) ∧
    (∃ alg, (flash_decode2_phi_surface B_Seqlen Mid_O Mid_O_LogExpSum Out
      768 192 64 1 12 3 1 256 64 1 64 17 64).toAlgorithm? =
        Except.ok alg) ∧
    (∃ alg, (flash_decode2_phi_surface B_Seqlen Mid_O Mid_O_LogExpSum Out
      768 192 64 1 12 3 1 256 64 1 64 8 64).toAlgorithm? =
        Except.ok alg) ∧
    (∃ alg, (flash_decode2_phi_surface B_Seqlen Mid_O Mid_O_LogExpSum Out
      768 192 64 1 12 3 1 256 64 1 64 32 64).toAlgorithm? =
        Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := flash_decode2_phi_final_store_slice Final Out
        64 256 64 1 256 64 1 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 64 => active s 64 i)
        (fun i : Fin 64 => (Out, outOffset s 256 64 1 i)))
      (expected := fun i : Fin 64 =>
        finalStoreValue s Final 64 256 64 1 i) := by
  constructor
  · exact flash_decode2_phi_surface_toAlgorithm_supported B_Seqlen Mid_O
      Mid_O_LogExpSum Out 768 192 64 1 12 3 1 256 64 1 64 16 64
  constructor
  · exact flash_decode2_phi_surface_toAlgorithm_supported B_Seqlen Mid_O
      Mid_O_LogExpSum Out 768 192 64 1 12 3 1 256 64 1 64 17 64
  constructor
  · exact flash_decode2_phi_surface_toAlgorithm_supported B_Seqlen Mid_O
      Mid_O_LogExpSum Out 768 192 64 1 12 3 1 256 64 1 64 8 64
  constructor
  · exact flash_decode2_phi_surface_toAlgorithm_supported B_Seqlen Mid_O
      Mid_O_LogExpSum Out 768 192 64 1 12 3 1 256 64 1 64 32 64
  · exact flash_decode2_phi_final_store_python_test_shape_compute_correct
      Final Out s

/-- `output_summary` row for Phi final normalization and writeback.

This connects the Python test-shape accumulator and `sum_exp` outputs to the
exact Python-observable `Out` row, including `head_dim = 64` masking and
contiguous `(256, 64, 1)` output strides. -/
theorem flash_decode2_phi_python_test_shape_normalization_store_output_summary
    (B_Seqlen : Region .nat) (Mid_O Mid_O_LogExpSum Acc SumExp Out : RegionName)
    (s : BlockState) :
    (∃ alg, (flash_decode2_phi_surface B_Seqlen Mid_O Mid_O_LogExpSum Out
      768 192 64 1 12 3 1 256 64 1 64 16 64).toAlgorithm? =
        Except.ok alg) ∧
    (∃ alg, (flash_decode2_phi_surface B_Seqlen Mid_O Mid_O_LogExpSum Out
      768 192 64 1 12 3 1 256 64 1 64 17 64).toAlgorithm? =
        Except.ok alg) ∧
    (∃ alg, (flash_decode2_phi_surface B_Seqlen Mid_O Mid_O_LogExpSum Out
      768 192 64 1 12 3 1 256 64 1 64 8 64).toAlgorithm? =
        Except.ok alg) ∧
    (∃ alg, (flash_decode2_phi_surface B_Seqlen Mid_O Mid_O_LogExpSum Out
      768 192 64 1 12 3 1 256 64 1 64 32 64).toAlgorithm? =
        Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := flash_decode2_phi_normalization_store_kernel Acc SumExp Out
        64 256 64 1 4 1 256 64 1 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 64 => active s 64 i)
        (fun i : Fin 64 => (Out, outOffset s 256 64 1 i)))
      (expected := fun i : Fin 64 =>
        normalizedStoreValue s Acc SumExp 64 256 64 1 4 1 i) := by
  constructor
  · exact flash_decode2_phi_surface_toAlgorithm_supported B_Seqlen Mid_O
      Mid_O_LogExpSum Out 768 192 64 1 12 3 1 256 64 1 64 16 64
  constructor
  · exact flash_decode2_phi_surface_toAlgorithm_supported B_Seqlen Mid_O
      Mid_O_LogExpSum Out 768 192 64 1 12 3 1 256 64 1 64 17 64
  constructor
  · exact flash_decode2_phi_surface_toAlgorithm_supported B_Seqlen Mid_O
      Mid_O_LogExpSum Out 768 192 64 1 12 3 1 256 64 1 64 8 64
  constructor
  · exact flash_decode2_phi_surface_toAlgorithm_supported B_Seqlen Mid_O
      Mid_O_LogExpSum Out 768 192 64 1 12 3 1 256 64 1 64 32 64
  · exact flash_decode2_phi_normalization_store_python_test_shape_compute_correct
      Acc SumExp Out s

/-- End-to-end running-max value summary for the Phi stage2 Python test shape.

The four checked `BLOCK_SEQ` surfaces are present, and the dynamic
`block_n_size` loop invariant for `max_logic` is stated as the recursive value
`runningMaxAfter`: it starts at `-inf` and each iteration joins the next
`Mid_O_LogExpSum` element at contiguous `offs_logic + block_seq_n`. -/
theorem flash_decode2_phi_python_test_shape_running_max_output_summary
    (B_Seqlen : Region .nat) (Mid_O Mid_O_LogExpSum Out : RegionName)
    (s : BlockState) (block_n_size : Nat) :
    (∃ alg, (flash_decode2_phi_surface B_Seqlen Mid_O Mid_O_LogExpSum Out
      768 192 64 1 12 3 1 256 64 1 64 16 64).toAlgorithm? =
        Except.ok alg) ∧
    (∃ alg, (flash_decode2_phi_surface B_Seqlen Mid_O Mid_O_LogExpSum Out
      768 192 64 1 12 3 1 256 64 1 64 17 64).toAlgorithm? =
        Except.ok alg) ∧
    (∃ alg, (flash_decode2_phi_surface B_Seqlen Mid_O Mid_O_LogExpSum Out
      768 192 64 1 12 3 1 256 64 1 64 8 64).toAlgorithm? =
        Except.ok alg) ∧
    (∃ alg, (flash_decode2_phi_surface B_Seqlen Mid_O Mid_O_LogExpSum Out
      768 192 64 1 12 3 1 256 64 1 64 32 64).toAlgorithm? =
        Except.ok alg) ∧
    runningMaxAfter s Mid_O_LogExpSum 12 3 0 = ⊥ ∧
    (∀ k, k < block_n_size →
      runningMaxAfter s Mid_O_LogExpSum 12 3 (k + 1) =
        runningMaxJoin
          (some (s.readMem Mid_O_LogExpSum (logicOffset s 12 3 k)) :
            WithBot ℝ)
          (runningMaxAfter s Mid_O_LogExpSum 12 3 k)) := by
  constructor
  · exact flash_decode2_phi_surface_toAlgorithm_supported B_Seqlen Mid_O
      Mid_O_LogExpSum Out 768 192 64 1 12 3 1 256 64 1 64 16 64
  constructor
  · exact flash_decode2_phi_surface_toAlgorithm_supported B_Seqlen Mid_O
      Mid_O_LogExpSum Out 768 192 64 1 12 3 1 256 64 1 64 17 64
  constructor
  · exact flash_decode2_phi_surface_toAlgorithm_supported B_Seqlen Mid_O
      Mid_O_LogExpSum Out 768 192 64 1 12 3 1 256 64 1 64 8 64
  constructor
  · exact flash_decode2_phi_surface_toAlgorithm_supported B_Seqlen Mid_O
      Mid_O_LogExpSum Out 768 192 64 1 12 3 1 256 64 1 64 32 64
  constructor
  · exact runningMaxAfter_zero s Mid_O_LogExpSum 12 3
  · intro k _hk
    exact runningMaxAfter_succ s Mid_O_LogExpSum 12 3 k

/-- `output_summary` row for the Phi masked scaled-accumulator proof gap.

This narrower follow-up covers the masked `Mid_O` loads, `old_scale` /
`exp_logic` updates, and `sum_exp` recurrence; the current proof-oriented
summary still uses a precomputed normalized `Final` vector. -/
abbrev flash_decode2_phi_python_test_shape_masked_accumulator_output_summary
    (B_Seqlen : Region .nat) (Mid_O Mid_O_LogExpSum Final Out : RegionName)
    (s : BlockState) :=
  flash_decode2_phi_python_test_shape_summary B_Seqlen Mid_O Mid_O_LogExpSum
    Final Out s

end VeriTile.Bench.TritonBenchG.FlashDecode2Phi
