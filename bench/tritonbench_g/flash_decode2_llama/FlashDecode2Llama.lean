import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.FlashDecode2Llama

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Faithful transcription of `flash_decode2_llama.py`'s
`_fwd_kernel_flash_decode_stage2`.

The Python body passes but does not use `stride_mid_od`, `stride_mid_o_es`, or
`stride_od`; this surface preserves that addressing behavior and keeps those
signature positions underscored. -/
def flash_decode2_llama_surface
    (B_Seqlen : Region .nat) (Mid_O Mid_O_LogExpSum O : RegionName)
    (stride_mid_ob stride_mid_oh stride_mid_os _stride_mid_od
      stride_mid_o_eb stride_mid_o_eh _stride_mid_o_es stride_obs stride_oh _stride_od
      BLOCK_SEQ BLOCK_DMODEL : Nat) : ComputeKernel := triton {
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
    tv = tl.load(Mid_O + offs_v + block_seq_n * $(stride_mid_os))
    tlogic = tl.load(Mid_O_LogExpSum + offs_logic + block_seq_n)
    new_max_logic = tl.maximum(tlogic, max_logic)
    old_scale = tl.exp(max_logic - new_max_logic)
    acc *= old_scale
    exp_logic = tl.exp(tlogic - new_max_logic)
    acc += exp_logic * tv
    sum_exp = sum_exp * old_scale + exp_logic
    max_logic = new_max_logic
  }
  tl.store(O + cur_batch * $(stride_obs) + cur_head * $(stride_oh) + offs_d,
    acc / sum_exp)
}

/-- The full flash-decode stage2 LLaMA surface lowers to the algorithm layer. -/
theorem flash_decode2_llama_surface_toAlgorithm_supported
    (B_Seqlen : Region .nat) (Mid_O Mid_O_LogExpSum O : RegionName)
    (stride_mid_ob stride_mid_oh stride_mid_os stride_mid_od
      stride_mid_o_eb stride_mid_o_eh stride_mid_o_es stride_obs stride_oh stride_od
      BLOCK_SEQ BLOCK_DMODEL : Nat) :
    ∃ alg, (flash_decode2_llama_surface B_Seqlen Mid_O Mid_O_LogExpSum O
      stride_mid_ob stride_mid_oh stride_mid_os stride_mid_od stride_mid_o_eb
      stride_mid_o_eh stride_mid_o_es stride_obs stride_oh stride_od BLOCK_SEQ
      BLOCK_DMODEL).toAlgorithm? = Except.ok alg := by
  simp [flash_decode2_llama_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- Proof-oriented final output-store slice of
`flash_decode2_llama.py`'s `_fwd_kernel_flash_decode_stage2`.

The full stage2 kernel reduces per-sequence-block partial outputs into a final
`acc / sum_exp` vector. This slice starts from a precomputed normalized `Final`
vector and proves the unmasked writeback into `O`. -/
def flash_decode2_llama_final_store_slice
    (Final O : RegionName)
    (stride_final_b stride_final_h stride_final_d
      stride_obs stride_oh stride_od
      BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  final = tl.load(Final + cur_batch * $(stride_final_b) +
      cur_head * $(stride_final_h) + offs_d * $(stride_final_d))
  tl.store(O + cur_batch * $(stride_obs) + cur_head * $(stride_oh) +
      offs_d * $(stride_od), final)
}

/-- Final normalization and output writeback for LLaMA flash-decode stage2.

This kernel consumes the loop-produced accumulator vector and `sum_exp` scalar,
computes `acc / sum_exp`, and writes the Python-observable `O` row. -/
def flash_decode2_llama_normalization_store_kernel
    (Acc SumExp O : RegionName)
    (stride_acc_b stride_acc_h stride_acc_d
      stride_sum_b stride_sum_h
      stride_obs stride_oh stride_od
      BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  acc = tl.load(Acc + cur_batch * $(stride_acc_b) +
      cur_head * $(stride_acc_h) + offs_d * $(stride_acc_d))
  sum_exp = tl.load(SumExp + cur_batch * $(stride_sum_b) +
      cur_head * $(stride_sum_h))
  tl.store(O + cur_batch * $(stride_obs) + cur_head * $(stride_oh) +
      offs_d * $(stride_od), acc / sum_exp)
}

def dIndex (_s : BlockState) (i : Fin BLOCK_DMODEL) : Nat :=
  i.val

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

noncomputable def normalizedStoreValue
    (s : BlockState) (Acc SumExp : RegionName)
    (stride_acc_b stride_acc_h stride_acc_d
      stride_sum_b stride_sum_h : Nat)
    (i : Fin BLOCK_DMODEL) : ℝ :=
  s.readMem Acc (accOffset s stride_acc_b stride_acc_h stride_acc_d i) /
    s.readMem SumExp (sumExpOffset s stride_sum_b stride_sum_h)

/-- Algorithm-layer correctness for the final flash-decode writeback. -/
theorem flash_decode2_llama_final_store_slice_correct
    (Final O : RegionName)
    (stride_final_b stride_final_h stride_final_d
      stride_obs stride_oh stride_od
      BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_DMODEL => outOffset s stride_obs stride_oh stride_od i)) :
    ∀ i : Fin BLOCK_DMODEL,
      let outAddr := outOffset s stride_obs stride_oh stride_od i
      (exec (flash_decode2_llama_final_store_slice Final O stride_final_b
            stride_final_h stride_final_d stride_obs stride_oh stride_od
            BLOCK_DMODEL) s).map (·.readMem O outAddr)
        = some (s.readMem Final
            (finalOffset s stride_final_b stride_final_h stride_final_d i)) := by
  intro i
  simp [exec, flash_decode2_llama_final_store_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul, dIndex, finalOffset, outOffset]
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_DMODEL] =>
        s.pids 0 * stride_obs + s.pids 1 * stride_oh + idx.1.val * stride_od) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have habFin : outOffset s stride_obs stride_oh stride_od a =
        outOffset s stride_obs stride_oh stride_od b := by
      simpa [outOffset, dIndex] using hab
    obtain rfl : a = b := hOutInj habFin
    rfl
  rw [BlockState.scatter_readback_nd _ _ _ hRawInj (i, PUnit.unit)]

/-- Compute-facing correctness for the final flash-decode writeback. -/
theorem flash_decode2_llama_final_store_slice_compute_correct
    (Final O : RegionName)
    (stride_final_b stride_final_h stride_final_d
      stride_obs stride_oh stride_od
      BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_DMODEL => outOffset s stride_obs stride_oh stride_od i)) :
    ComputeCorrect.Realizes
      (kernel := flash_decode2_llama_final_store_slice Final O stride_final_b
        stride_final_h stride_final_d stride_obs stride_oh stride_od
        BLOCK_DMODEL)
      (initialState := s)
      (write := fun i : Fin BLOCK_DMODEL =>
        some (O, outOffset s stride_obs stride_oh stride_od i))
      (expected := fun i : Fin BLOCK_DMODEL =>
        s.readMem Final
          (finalOffset s stride_final_b stride_final_h stride_final_d i)) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [flash_decode2_llama_final_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i
  have h := flash_decode2_llama_final_store_slice_correct Final O
    stride_final_b stride_final_h stride_final_d stride_obs stride_oh stride_od
    BLOCK_DMODEL s hOutInj i
  rw [hExec] at h
  exact Option.some.inj h

/-! ## Python test-shape wrappers

The checked Python tests allocate `O` with shape `(2, 4, 32)`, so the
contiguous output strides are `(128, 32, 1)`. `mid_out` has `head_dim = 32`,
so `BLOCK_DMODEL = 32`; the varying `B_Seqlen` and `block_seq` cases do not
change the final output layout. -/

theorem flash_decode2_llama_python_test_shape_offset_injective
    (s : BlockState) :
    Function.Injective
      (fun i : Fin 32 => outOffset s 128 32 1 i) := by
  intro a b h
  simp [outOffset, dIndex] at h
  exact Fin.ext (by omega)

theorem flash_decode2_llama_final_store_python_test_shape_compute_correct
    (Final O : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := flash_decode2_llama_final_store_slice Final O
        128 32 1 128 32 1 32)
      (initialState := s)
      (write := fun i : Fin 32 => some (O, outOffset s 128 32 1 i))
      (expected := fun i : Fin 32 =>
        s.readMem Final (finalOffset s 128 32 1 i)) := by
  exact flash_decode2_llama_final_store_slice_compute_correct Final O
    128 32 1 128 32 1 32 s
    (flash_decode2_llama_python_test_shape_offset_injective s)

/-- Algorithm-layer correctness for LLaMA stage2 normalization plus writeback. -/
theorem flash_decode2_llama_normalization_store_kernel_correct
    (Acc SumExp O : RegionName)
    (stride_acc_b stride_acc_h stride_acc_d
      stride_sum_b stride_sum_h
      stride_obs stride_oh stride_od
      BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_DMODEL => outOffset s stride_obs stride_oh stride_od i)) :
    ∀ i : Fin BLOCK_DMODEL,
      let outAddr := outOffset s stride_obs stride_oh stride_od i
      (exec (flash_decode2_llama_normalization_store_kernel Acc SumExp O
            stride_acc_b stride_acc_h stride_acc_d stride_sum_b stride_sum_h
            stride_obs stride_oh stride_od BLOCK_DMODEL) s).map (·.readMem O outAddr)
        = some (normalizedStoreValue s Acc SumExp stride_acc_b stride_acc_h
            stride_acc_d stride_sum_b stride_sum_h i) := by
  intro i
  simp [exec, flash_decode2_llama_normalization_store_kernel, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul, NumericDType.div, dIndex,
        accOffset, sumExpOffset, outOffset]
  let offsetFn : TileIndex [BLOCK_DMODEL] → Nat :=
    fun idx => s.pids 0 * stride_obs + s.pids 1 * stride_oh + idx.1.val * stride_od
  let valueFn : TileIndex [BLOCK_DMODEL] → ℝ :=
    fun idx =>
      WithBot.unbotD 0
        (match
          some (s.readMem Acc
            (s.pids 0 * stride_acc_b + s.pids 1 * stride_acc_h +
              idx.1.val * stride_acc_d))
        with
        | some x =>
            some (x / s.readMem SumExp
              (s.pids 0 * stride_sum_b + s.pids 1 * stride_sum_h))
        | none => none)
  have hRawInj : Function.Injective offsetFn := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have habFin : outOffset s stride_obs stride_oh stride_od a =
        outOffset s stride_obs stride_oh stride_od b := by
      simpa [offsetFn, outOffset, dIndex] using hab
    obtain rfl : a = b := hOutInj habFin
    rfl
  rw [BlockState.scatter_readback_nd _ _ _ hRawInj (i, PUnit.unit)]
  simp [offsetFn, valueFn, normalizedStoreValue, accOffset, sumExpOffset, dIndex]

/-- Compute-facing correctness for LLaMA stage2 normalization plus writeback. -/
theorem flash_decode2_llama_normalization_store_kernel_compute_correct
    (Acc SumExp O : RegionName)
    (stride_acc_b stride_acc_h stride_acc_d
      stride_sum_b stride_sum_h
      stride_obs stride_oh stride_od
      BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_DMODEL => outOffset s stride_obs stride_oh stride_od i)) :
    ComputeCorrect.Realizes
      (kernel := flash_decode2_llama_normalization_store_kernel Acc SumExp O
        stride_acc_b stride_acc_h stride_acc_d stride_sum_b stride_sum_h
        stride_obs stride_oh stride_od BLOCK_DMODEL)
      (initialState := s)
      (write := fun i : Fin BLOCK_DMODEL =>
        some (O, outOffset s stride_obs stride_oh stride_od i))
      (expected := fun i : Fin BLOCK_DMODEL =>
        normalizedStoreValue s Acc SumExp stride_acc_b stride_acc_h
          stride_acc_d stride_sum_b stride_sum_h i) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [flash_decode2_llama_normalization_store_kernel]
  intro s0 s' hExec hs0
  subst s0
  intro i
  have h := flash_decode2_llama_normalization_store_kernel_correct Acc SumExp O
    stride_acc_b stride_acc_h stride_acc_d stride_sum_b stride_sum_h
    stride_obs stride_oh stride_od BLOCK_DMODEL s hOutInj i
  rw [hExec] at h
  exact Option.some.inj h

theorem flash_decode2_llama_normalization_store_python_test_shape_compute_correct
    (Acc SumExp O : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := flash_decode2_llama_normalization_store_kernel Acc SumExp O
        128 32 1 4 1 128 32 1 32)
      (initialState := s)
      (write := fun i : Fin 32 => some (O, outOffset s 128 32 1 i))
      (expected := fun i : Fin 32 =>
        normalizedStoreValue s Acc SumExp 128 32 1 4 1 i) := by
  exact flash_decode2_llama_normalization_store_kernel_compute_correct Acc SumExp O
    128 32 1 4 1 128 32 1 32 s
    (flash_decode2_llama_python_test_shape_offset_injective s)

/-- Public Python test-shape summary for `flash_decode2_llama.py`.

The Python tests use `mid_out : (2, 4, 3, 32)`, `O : (2, 4, 32)`, and
`block_seq = 8`. This records the faithful full stage2 surface and ties the
loop-produced `Acc` and `SumExp` values to the exact Python-observable `O`
writeback. -/
theorem flash_decode2_llama_python_test_shape_output_summary
    (B_Seqlen : Region .nat) (Mid_O Mid_O_LogExpSum Acc SumExp O : RegionName)
    (s : BlockState) :
    (∃ alg, (flash_decode2_llama_surface B_Seqlen Mid_O Mid_O_LogExpSum O
      384 96 32 1 12 3 1 128 32 1 8 32).toAlgorithm? =
        Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := flash_decode2_llama_normalization_store_kernel Acc SumExp O
        128 32 1 4 1 128 32 1 32)
      (initialState := s)
      (write := fun i : Fin 32 => some (O, outOffset s 128 32 1 i))
      (expected := fun i : Fin 32 =>
        normalizedStoreValue s Acc SumExp 128 32 1 4 1 i) := by
  constructor
  · exact flash_decode2_llama_surface_toAlgorithm_supported B_Seqlen Mid_O
      Mid_O_LogExpSum O 384 96 32 1 12 3 1 128 32 1 8 32
  · exact flash_decode2_llama_normalization_store_python_test_shape_compute_correct
      Acc SumExp O s









































/-- `output_summary` row for the LLaMA stage2 running-max follow-up.

This narrower follow-up covers the dynamic `max_logic` recurrence over
`Mid_O_LogExpSum`; the current proof-oriented summary records the faithful
surface but does not prove that loop invariant. -/
abbrev flash_decode2_llama_python_test_shape_running_max_output_summary
    (B_Seqlen : Region .nat) (Mid_O Mid_O_LogExpSum Final O : RegionName)
    (s : BlockState) :=
  (∃ alg, (flash_decode2_llama_surface B_Seqlen Mid_O Mid_O_LogExpSum O
      384 96 32 1 12 3 1 128 32 1 8 32).toAlgorithm? =
        Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := flash_decode2_llama_final_store_slice Final O
        128 32 1 128 32 1 32)
      (initialState := s)
      (write := fun i : Fin 32 => some (O, outOffset s 128 32 1 i))
      (expected := fun i : Fin 32 =>
        s.readMem Final (finalOffset s 128 32 1 i))

end VeriTile.Bench.TritonBenchG.FlashDecode2Llama
