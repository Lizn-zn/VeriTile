import VeriTile.Triton

/-!
# `flash_decode2_llama` — strict per-kernel correctness

`_fwd_kernel_flash_decode_stage2` is the LLaMA flash-decoding **stage-2**
reduction: program `(cur_batch, cur_head)` iterates over the `block_n_size`
partial blocks produced by stage-1, combining their per-block partial outputs
`Mid_O` and log-sum-exp scalars `Mid_O_LogExpSum` via the online-softmax
recurrence (running max `max_logic`, rescaled `acc` and `sum_exp`), then stores
the normalized `acc / sum_exp` to `O[cur_batch, cur_head, :]`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`_fwd_kernel_flash_decode_stage2[(batch, head_num)](...)`,
the grid over `(batch, head)`, the host `BLOCK_DMODEL = Lk` / `BLOCK_SEQ` choices,
`num_warps` / `num_stages`, the stage-1 kernel that produced `Mid_O`, and how the
runtime composes per-program writes into `O`) is the *trusted boundary*, not a
proof obligation here. Because the program ids `cur_batch`/`cur_head` are
universally quantified (via `BlockState`), the per-program statements cover every
program of the grid.

## Proof architecture

```
flash_decode2_llama_normalization_output_summary_general      ← TOP THEOREM (final store)
  ├─ flash_decode2_llama_surface_toAlgorithm_supported          surface lowers
  └─ flash_decode2_llama_normalization_store_kernel_compute_correct
       └─ flash_decode2_llama_normalization_store_kernel_correct  (acc / sum_exp readback)

flash_decode2_llama_running_max_output_summary_general        ← TOP THEOREM (recurrence)
  ├─ flash_decode2_llama_surface_toAlgorithm_supported
  └─ flash_decode2_llama_running_max_step_kernel_compute_correct
```

There is also a `flash_decode2_llama_final_store_slice_*` chain (raw `O` store
without normalization).

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float, so the `-inf` init, `exp`,
running-max rescaling, and the final division are real-valued);
`@triton.autotune` / `num_warps` / `num_stages` are not modeled. The verification
is split across two separately-proven faces rather than one whole-loop theorem:
(1) the **final normalization store** to `O` — expected `normalizedStoreValue`
(= `acc / sum_exp`) at each `outOffset`, over a one-block output footprint; and
(2) one **running-max recurrence step** — `new_max_logic = max(tlogic, max_logic)`
(`runningMaxStepValue`) written as a scalar. Both hold for the universally
quantified program ids and are fully dimension-general — parameterized over a
symbolic `BLOCK_DMODEL` and every stride (no hardcoded shape literals). The
checked Python test shape (`BLOCK_DMODEL = 32`, `BLOCK_SEQ = 8`, `batch = 2`,
`head = 4`, `seq_block = 3`) is only background, an instance of the general
statements. The intermediate
`Acc`/`SumExp` register state is taken as given (it is what the running recurrence
maintains); the full loop is not unrolled into a single closed-form spec, so the
end-to-end "`O = softmax-weighted combine of all blocks`" statement is the
composition of these faces with the (trusted) loop scheduling.
-/

namespace VeriTile.Bench.TritonBenchG.FlashDecode2Llama

open VeriTile.Triton
open scoped VeriTile.Triton.Masked3DTileKernelIO₁

set_option linter.unusedSimpArgs false

/-! **★ Main theorems:** `flash_decode2_llama_normalization_output_summary_general`, `flash_decode2_llama_running_max_output_summary_general` -/

/-! # ══════════ CORRECT — genuine / dimension-general (review this) ══════════ -/

section Correct_without_Rounding

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

/-- One LLaMA stage2 running-maximum recurrence step. -/
def flash_decode2_llama_running_max_step_kernel
    (Mid_O_LogExpSum MaxLogic NewMaxLogic : RegionName)
    (block_seq_n
      stride_mid_o_eb stride_mid_o_eh
      stride_logic_b stride_logic_h : Nat) :
    ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  tlogic = tl.load(Mid_O_LogExpSum + cur_batch * $(stride_mid_o_eb) +
      cur_head * $(stride_mid_o_eh) + $(block_seq_n))
  max_logic = tl.load(MaxLogic + cur_batch * $(stride_logic_b) +
      cur_head * $(stride_logic_h))
  new_max_logic = tl.maximum(tlogic, max_logic)
  tl.store(NewMaxLogic + cur_batch * $(stride_logic_b) +
      cur_head * $(stride_logic_h), new_max_logic)
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

def logicOffset
    (s : BlockState)
    (block_seq_n stride_mid_o_eb stride_mid_o_eh : Nat) : Nat :=
  s.pids 0 * stride_mid_o_eb + s.pids 1 * stride_mid_o_eh + block_seq_n

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

noncomputable def runningMaxStepValue
    (s : BlockState) (Mid_O_LogExpSum MaxLogic : RegionName)
    (block_seq_n stride_mid_o_eb stride_mid_o_eh
      stride_logic_b stride_logic_h : Nat) : ℝ :=
  let tlogic := s.readMem Mid_O_LogExpSum
    (logicOffset s block_seq_n stride_mid_o_eb stride_mid_o_eh)
  let maxLogic := s.readMem MaxLogic (sumExpOffset s stride_logic_b stride_logic_h)
  if tlogic > maxLogic then tlogic else maxLogic

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
    ComputeCorrect.Realizes_without_Rounding
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

/-! ## Normalization-store correctness (dimension-general)

The theorems below are symbolic in `BLOCK_DMODEL` and every stride; no shape
literals are hardcoded. For background, the checked Python tests allocate `O`
with shape `(2, 4, 32)` (contiguous output strides `(128, 32, 1)`) and `mid_out`
has `head_dim = 32` (so `BLOCK_DMODEL = 32`); the varying `B_Seqlen` and
`block_seq` cases do not change the final output layout. That concrete shape is
just one instance of the general statements. -/

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
    ComputeCorrect.Realizes_without_Rounding
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

/-- Compute-facing correctness for one LLaMA running-maximum recurrence step. -/
theorem flash_decode2_llama_running_max_step_kernel_compute_correct
    (Mid_O_LogExpSum MaxLogic NewMaxLogic : RegionName)
    (block_seq_n
      stride_mid_o_eb stride_mid_o_eh
      stride_logic_b stride_logic_h : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := flash_decode2_llama_running_max_step_kernel Mid_O_LogExpSum
        MaxLogic NewMaxLogic block_seq_n stride_mid_o_eb stride_mid_o_eh
        stride_logic_b stride_logic_h)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.scalar NewMaxLogic
        (sumExpOffset s stride_logic_b stride_logic_h))
      (expected := fun _ : PUnit =>
        runningMaxStepValue s Mid_O_LogExpSum MaxLogic block_seq_n
          stride_mid_o_eb stride_mid_o_eh stride_logic_b stride_logic_h) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [flash_decode2_llama_running_max_step_kernel]
  intro s0 s' hExec hs0
  subst s0
  intro i
  simp [exec, flash_decode2_llama_running_max_step_kernel, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul, ComparableDType.gt, logicOffset,
        sumExpOffset, runningMaxStepValue, ComputeCorrect.WriteMap.scalar] at hExec ⊢
  rw [← hExec]
  simp [BlockState.writeMem_readMem]
  by_cases hlt :
      s.readMem MaxLogic (s.pids 0 * stride_logic_b + s.pids 1 * stride_logic_h) <
        s.readMem Mid_O_LogExpSum
          (s.pids 0 * stride_mid_o_eb + s.pids 1 * stride_mid_o_eh + block_seq_n)
  · have hbot :
        (s.readMem MaxLogic
          (s.pids 0 * stride_logic_b + s.pids 1 * stride_logic_h) : WithBot ℝ) <
          (s.readMem Mid_O_LogExpSum
            (s.pids 0 * stride_mid_o_eb + s.pids 1 * stride_mid_o_eh + block_seq_n) :
            WithBot ℝ) := by
      simpa using hlt
    split
    · exact WithBot.unbotD_coe 0 _
    · rename_i hnot
      exact False.elim (hnot hbot)
  · have hbot :
        ¬ (s.readMem MaxLogic
          (s.pids 0 * stride_logic_b + s.pids 1 * stride_logic_h) : WithBot ℝ) <
          (s.readMem Mid_O_LogExpSum
            (s.pids 0 * stride_mid_o_eb + s.pids 1 * stride_mid_o_eh + block_seq_n) :
            WithBot ℝ) := by
      simpa using hlt
    split
    · rename_i hpos
      exact False.elim (hbot hpos)
    · exact WithBot.unbotD_coe 0 _


/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/
/-- **General genuine final-normalization output summary.** Fully
dimension-parameterized over every stride and `BLOCK_DMODEL`: the full LLaMA
flash-decode stage2 surface lowers, and the normalization writeback *realizes*
the genuine closed form `O[d] = acc[d] / sum_exp` (`normalizedStoreValue`,
reading the loop-produced `Acc`/`SumExp` input memory) at each output lane. No
hardcoded shape literals. Side condition: output-footprint injectivity. -/
specification flash_decode2_llama_normalization_output_summary_general
    (B_Seqlen : Region .nat) (Mid_O Mid_O_LogExpSum Acc SumExp O : RegionName)
    (stride_mid_ob stride_mid_oh stride_mid_os stride_mid_od
      stride_mid_o_eb stride_mid_o_eh stride_mid_o_es stride_obs stride_oh stride_od
      BLOCK_SEQ BLOCK_DMODEL
      stride_acc_b stride_acc_h stride_acc_d stride_sum_b stride_sum_h : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_DMODEL => outOffset s stride_obs stride_oh stride_od i)) :
    (∃ alg, (flash_decode2_llama_surface B_Seqlen Mid_O Mid_O_LogExpSum O
      stride_mid_ob stride_mid_oh stride_mid_os stride_mid_od stride_mid_o_eb
      stride_mid_o_eh stride_mid_o_es stride_obs stride_oh stride_od BLOCK_SEQ
      BLOCK_DMODEL).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := flash_decode2_llama_normalization_store_kernel Acc SumExp O
        stride_acc_b stride_acc_h stride_acc_d stride_sum_b stride_sum_h
        stride_obs stride_oh stride_od BLOCK_DMODEL)
      (initialState := s)
      (write := fun i : Fin BLOCK_DMODEL =>
        some (O, outOffset s stride_obs stride_oh stride_od i))
      (expected := fun i : Fin BLOCK_DMODEL =>
        normalizedStoreValue s Acc SumExp stride_acc_b stride_acc_h
          stride_acc_d stride_sum_b stride_sum_h i) := by
  refine ⟨?_, ?_⟩
  · exact flash_decode2_llama_surface_toAlgorithm_supported B_Seqlen Mid_O
      Mid_O_LogExpSum O stride_mid_ob stride_mid_oh stride_mid_os stride_mid_od
      stride_mid_o_eb stride_mid_o_eh stride_mid_o_es stride_obs stride_oh stride_od
      BLOCK_SEQ BLOCK_DMODEL
  · exact flash_decode2_llama_normalization_store_kernel_compute_correct Acc SumExp O
      stride_acc_b stride_acc_h stride_acc_d stride_sum_b stride_sum_h
      stride_obs stride_oh stride_od BLOCK_DMODEL s hOutInj


/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/
/-- **General genuine running-maximum recurrence summary.** Fully
dimension-parameterized over the block index and every stride: the full LLaMA
flash-decode stage2 surface lowers, and one recurrence step *realizes* the
genuine closed form `new_max_logic = max(tlogic, max_logic)`
(`runningMaxStepValue`, reading `Mid_O_LogExpSum`/`MaxLogic` input memory). No
hardcoded shape literals. -/
specification flash_decode2_llama_running_max_output_summary_general
    (B_Seqlen : Region .nat)
    (Mid_O Mid_O_LogExpSum MaxLogic NewMaxLogic O : RegionName)
    (stride_mid_ob stride_mid_oh stride_mid_os stride_mid_od
      stride_mid_o_eb' stride_mid_o_eh' stride_mid_o_es stride_obs stride_oh stride_od
      BLOCK_SEQ BLOCK_DMODEL
      block_seq_n stride_mid_o_eb stride_mid_o_eh stride_logic_b stride_logic_h : Nat)
    (s : BlockState) :
    (∃ alg, (flash_decode2_llama_surface B_Seqlen Mid_O Mid_O_LogExpSum O
      stride_mid_ob stride_mid_oh stride_mid_os stride_mid_od stride_mid_o_eb'
      stride_mid_o_eh' stride_mid_o_es stride_obs stride_oh stride_od BLOCK_SEQ
      BLOCK_DMODEL).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := flash_decode2_llama_running_max_step_kernel Mid_O_LogExpSum
        MaxLogic NewMaxLogic block_seq_n stride_mid_o_eb stride_mid_o_eh
        stride_logic_b stride_logic_h)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.scalar NewMaxLogic
        (sumExpOffset s stride_logic_b stride_logic_h))
      (expected := fun _ : PUnit =>
        runningMaxStepValue s Mid_O_LogExpSum MaxLogic block_seq_n
          stride_mid_o_eb stride_mid_o_eh stride_logic_b stride_logic_h) := by
  refine ⟨?_, ?_⟩
  · exact flash_decode2_llama_surface_toAlgorithm_supported B_Seqlen Mid_O
      Mid_O_LogExpSum O stride_mid_ob stride_mid_oh stride_mid_os stride_mid_od
      stride_mid_o_eb' stride_mid_o_eh' stride_mid_o_es stride_obs stride_oh stride_od
      BLOCK_SEQ BLOCK_DMODEL
  · exact flash_decode2_llama_running_max_step_kernel_compute_correct
      Mid_O_LogExpSum MaxLogic NewMaxLogic block_seq_n stride_mid_o_eb stride_mid_o_eh
      stride_logic_b stride_logic_h s

end Correct_without_Rounding


/-! ## ════════ `⊨` IO face for the final writeback ════════

The summary above is stated per *declared write map*. This section restates the
final `Final → O` writeback on the audit-once IO surface
`Masked3DTileKernelIO₁.Implements` (`⊨`), which additionally pins the **flat
memory** placement.

The slice is a plain tile copy whose two windows are built from *both* program axes
(`cur_batch`, `cur_head`) with different strides on the two buffers — the shape the
three-axis tile skin exists for. Every lane is active (neither the load nor the
store carries a mask). -/

section IOFace

/-- Cell-level frame of an unmasked scatter (private copy — `bench` files are
standalone). -/
private theorem foldl_writeMem_frame_unmasked {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → ℝ) (R : RegionName) (off : Nat) :
    ∀ l : List α, (R ≠ region ∨ ∀ k ∈ l, offsetFn k ≠ off) →
      ∀ s : BlockState,
        ((l.foldl (fun acc k => acc.writeMem region (offsetFn k) (valueFn k))
            s).mem R off) = s.mem R off := by
  intro l
  induction l with
  | nil => intro _ s; rfl
  | cons hd tl ih =>
      intro hc s
      have htl : R ≠ region ∨ ∀ k ∈ tl, offsetFn k ≠ off := by
        rcases hc with h | h
        · exact Or.inl h
        · exact Or.inr fun k hk => h k (List.mem_cons_of_mem hd hk)
      rw [List.foldl_cons, ih htl, BlockState.writeMem_mem, if_neg ?_]
      rintro ⟨h1, h2⟩
      rcases hc with h | h
      · exact h h1
      · exact h hd List.mem_cons_self h2.symm

theorem final_store_flattenOk (Final O : RegionName)
    (stride_final_b stride_final_h stride_final_d stride_obs stride_oh
      stride_od BLOCK_DMODEL : Nat) :
    ((flash_decode2_llama_final_store_slice Final O stride_final_b
      stride_final_h stride_final_d stride_obs stride_oh stride_od
      BLOCK_DMODEL).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [flash_decode2_llama_final_store_slice, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, StmtList.FlattenOk,
    Stmt.FlattenOk, Op.FlattenOk]

theorem final_store_terminates (Final O : RegionName)
    (stride_final_b stride_final_h stride_final_d stride_obs stride_oh
      stride_od BLOCK_DMODEL : Nat) (s : BlockState) :
    ∃ s1, exec (flash_decode2_llama_final_store_slice Final O stride_final_b
      stride_final_h stride_final_d stride_obs stride_oh stride_od
      BLOCK_DMODEL) s = some s1 := by
  simp [exec, flash_decode2_llama_final_store_slice, stepStmts, stepStmt,
    evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.ptrAdd,
    NumericDType.add, NumericDType.mul]

theorem final_store_frame (Final O : RegionName)
    (stride_final_b stride_final_h stride_final_d stride_obs stride_oh
      stride_od BLOCK_DMODEL : Nat) (s s' : BlockState)
    (hExec : exec (flash_decode2_llama_final_store_slice Final O stride_final_b
      stride_final_h stride_final_d stride_obs stride_oh stride_od
      BLOCK_DMODEL) s = some s') :
    ∀ (r : RegionName) (o : Nat),
      (r ≠ O ∨ ∀ i : Fin BLOCK_DMODEL,
        o ≠ outOffset s stride_obs stride_oh stride_od i) →
      s'.mem r o = s.mem r o := by
  intro r o hcond
  simp [exec, flash_decode2_llama_final_store_slice, stepStmts, stepStmt,
    evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.ptrAdd,
    NumericDType.add, NumericDType.mul] at hExec
  subst hExec
  rw [foldl_writeMem_frame_unmasked (region := O)
    (fun i : TileIndex [BLOCK_DMODEL] =>
      s.pids 0 * stride_obs + s.pids 1 * stride_oh + i.1.val * stride_od)
    _ r o (TileShape.allIndices [BLOCK_DMODEL]) ?_]
  · simp
  · rcases hcond with h | h
    · exact Or.inl h
    · exact Or.inr fun i _ => Ne.symm (h i.1)

theorem final_store_traceSafe (Final O : RegionName)
    (stride_final_b stride_final_h stride_final_d stride_obs stride_oh
      stride_od BLOCK_DMODEL : Nat) (bounds : RegionBounds) (s : BlockState)
    (hin : ∀ i : Fin BLOCK_DMODEL,
      finalOffset s stride_final_b stride_final_h stride_final_d i
        < bounds Final)
    (hout : ∀ i : Fin BLOCK_DMODEL,
      outOffset s stride_obs stride_oh stride_od i < bounds O) :
    ((flash_decode2_llama_final_store_slice Final O stride_final_b
      stride_final_h stride_final_d stride_obs stride_oh stride_od
      BLOCK_DMODEL).toAlgKernel).TraceSafe bounds s := by
  simp [Kernel.TraceSafe, flash_decode2_llama_final_store_slice,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt, MaskOpt.SafeAt,
    MaskOpt.Active, MaskOpt.MemorySafe, MemAccess.SafeAt, MemAccess.MemorySafe,
    memAccessMemorySafe, MemAccess.ActiveAddressSafe,
    memAccessActiveAddressSafe, Op.PointerAddressesSafeOn, Op.MemorySafe,
    stepStmts, stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map,
    Tile.bop, Tile.ptrAdd, NumericDType.add, NumericDType.mul]
  exact ⟨fun a => hin a, fun a => hout a⟩

/-- Region-model run of the final writeback. -/
theorem final_store_region_run (Final O : RegionName)
    (stride_final_b stride_final_h stride_final_d stride_obs stride_oh
      stride_od BLOCK_DMODEL : Nat) (s₀ : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_DMODEL => outOffset s₀ stride_obs stride_oh stride_od i))
    (xs : TileIndex [BLOCK_DMODEL] → ℝ)
    (hx : ∀ i : TileIndex [BLOCK_DMODEL],
      s₀.readMem Final
          (finalOffset s₀ stride_final_b stride_final_h stride_final_d i.1)
        = xs i) :
    ∃ s1, exec (flash_decode2_llama_final_store_slice Final O stride_final_b
        stride_final_h stride_final_d stride_obs stride_oh stride_od
        BLOCK_DMODEL) s₀ = some s1
      ∧ (∀ i : TileIndex [BLOCK_DMODEL],
          s1.readMem O (outOffset s₀ stride_obs stride_oh stride_od i.1) = xs i)
      ∧ (∀ (r : RegionName) (o : Nat),
          (r ≠ O ∨ ∀ i : Fin BLOCK_DMODEL,
            o ≠ outOffset s₀ stride_obs stride_oh stride_od i) →
          s1.mem r o = s₀.mem r o) := by
  obtain ⟨s1, hexec⟩ := final_store_terminates Final O stride_final_b
    stride_final_h stride_final_d stride_obs stride_oh stride_od BLOCK_DMODEL s₀
  refine ⟨s1, hexec, ?_, final_store_frame Final O stride_final_b stride_final_h
    stride_final_d stride_obs stride_oh stride_od BLOCK_DMODEL s₀ s1 hexec⟩
  intro i
  have h := flash_decode2_llama_final_store_slice_correct Final O stride_final_b
    stride_final_h stride_final_d stride_obs stride_oh stride_od BLOCK_DMODEL s₀
    hOutInj i.1
  have h' : s1.readMem O (outOffset s₀ stride_obs stride_oh stride_od i.1)
      = s₀.readMem Final
        (finalOffset s₀ stride_final_b stride_final_h stride_final_d i.1) := by
    simpa [hexec] using h
  rw [h', hx i]

/-- IO signature of the final writeback on the three-axis tile surface. -/
def finalStoreIO (Final O : RegionName)
    (stride_final_b stride_final_h stride_final_d stride_obs stride_oh
      stride_od BLOCK_DMODEL : Nat) : Masked3DTileKernelIO₁ where
  kernel := flash_decode2_llama_final_store_slice Final O stride_final_b
    stride_final_h stride_final_d stride_obs stride_oh stride_od BLOCK_DMODEL
  inp := Final
  out := O
  shape := [BLOCK_DMODEL]
  read := fun p₀ p₁ _p₂ i =>
    p₀ * stride_final_b + p₁ * stride_final_h + i.1.val * stride_final_d
  write := fun p₀ p₁ _p₂ i =>
    p₀ * stride_obs + p₁ * stride_oh + i.1.val * stride_od
  mask := fun _p₀ _p₁ _p₂ _ => True

/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/

/-- **The headline on the IO surface** for `flash_decode2_llama.py`'s final
writeback: for every disjoint flat placement of `Final` / `O`, every program
coordinate whose lanes are in bounds, and every launch state whose `Final` row holds
`xs`, the translated pointer kernel terminates, every lane of the `O` row holds
`xs i`, and every other memory cell is unchanged.

Both windows are built from *two* program axes with different strides on the two
buffers — the shape the three-axis tile skin exists for. Dimension-general in all six
strides and `BLOCK_DMODEL`. Honest side-condition: output-address injectivity at every
program coordinate, the same hypothesis the per-write-map summary takes. -/
specification flash_decode2_llama_final_store_io_correctness (Final O : RegionName)
    (stride_final_b stride_final_h stride_final_d stride_obs stride_oh
      stride_od BLOCK_DMODEL : Nat)
    (hOutInj : ∀ p₀ p₁ : Nat, Function.Injective
      (fun i : Fin BLOCK_DMODEL =>
        p₀ * stride_obs + p₁ * stride_oh + i.val * stride_od)) :
    finalStoreIO Final O stride_final_b stride_final_h stride_final_d stride_obs
        stride_oh stride_od BLOCK_DMODEL
      ⊨ fun _p₀ _p₁ xs i => xs i := by
  refine Masked3DTileKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact final_store_flattenOk Final O stride_final_b stride_final_h
      stride_final_d stride_obs stride_oh stride_od BLOCK_DMODEL
  · intro bounds s h1 h2
    exact final_store_traceSafe Final O stride_final_b stride_final_h
      stride_final_d stride_obs stride_oh stride_od BLOCK_DMODEL bounds s
      (fun i => h1 (i, PUnit.unit) trivial) (fun i => h2 (i, PUnit.unit) trivial)
  · intro s₀ xs hin
    obtain ⟨s1, hexec, hval, hframe⟩ :=
      final_store_region_run Final O stride_final_b stride_final_h stride_final_d
        stride_obs stride_oh stride_od BLOCK_DMODEL s₀
        (hOutInj (s₀.pids 0) (s₀.pids 1)) xs (fun i => hin i trivial)
    exact ⟨s1, hexec, fun i _ => hval i,
      fun r o hcond => hframe r o (by
        rcases hcond with h | h
        · exact Or.inl h
        · exact Or.inr fun i => h (i, PUnit.unit) trivial)⟩

end IOFace

end VeriTile.Bench.TritonBenchG.FlashDecode2Llama
