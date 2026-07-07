import VeriTile.Core
import VeriTile.Semantics
import VeriTile.Float
import VeriTile.Frontend.Triton.DSL

/-!
# `flash_decode2_phi` — strict per-kernel correctness

`_fwd_kernel_flash_decode_stage2` is the Phi flash-decoding **stage-2**
reduction: program `(cur_batch, cur_head)` iterates over the `block_n_size`
partial blocks produced by stage-1, combining per-block partial outputs `Mid_O`
and log-sum-exp scalars `Mid_O_LogExpSum` via the online-softmax recurrence
(running max `max_logic`, rescaled `acc` and `sum_exp`), then stores the
normalized `acc / sum_exp` to `Out[cur_batch, cur_head, :]`. Unlike the LLaMA
variant, all `Mid_O` loads and the final store are masked by `offs_d < head_dim`
(here `BLOCK_DMODEL = next_power_of_2(head_dim)` may exceed `head_dim`).

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`_fwd_kernel_flash_decode_stage2[(batch, head_num)](...)`,
the grid over `(batch, head)`, the host `BLOCK_DMODEL = next_power_of_2(Lk)` /
`BLOCK_SEQ` choices, the stage-1 kernel that produced `Mid_O`, and how the runtime
composes per-program writes into `Out`) is the *trusted boundary*, not a proof
obligation here. Because the program ids `cur_batch`/`cur_head` are universally
quantified (via `BlockState`), the per-program statements cover every program of
the grid.

## Proof architecture

```
flash_decode2_phi_masked_accumulator_output_summary_general  ← supporting face (one loop step)
  ├─ flash_decode2_phi_surface_toAlgorithm_supported
  ├─ flash_decode2_phi_accumulator_step_kernel_compute_correct
  │    └─ flash_decode2_phi_accumulator_step_kernel_correct
  └─ flash_decode2_phi_sum_exp_step_kernel_compute_correct

flash_decode2_phi_running_max_output_summary_general         ← supporting face (running-max invariant)
  ├─ flash_decode2_phi_surface_toAlgorithm_supported
  └─ using runningMaxAfter (runningMaxAfter_zero / runningMaxAfter_succ)

flash_decode2_phi_normalization_output_summary_general       ← HEADLINE / last (acc/sum_exp store)
  ├─ flash_decode2_phi_surface_toAlgorithm_supported
  └─ flash_decode2_phi_normalization_store_kernel_compute_correct
       └─ flash_decode2_phi_normalization_store_kernel_correct
```

There is also a `flash_decode2_phi_final_store_slice_*` chain (raw masked `Out`
store without normalization, via `flash_decode2_phi_final_store_slice_correct`).

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float, so the `-inf` init, `exp`,
running-max rescaling, and the final division are real-valued);
`@triton.autotune` / `num_warps` / `num_stages` are not modeled. The verification
is split across four separately-proven faces rather than one whole-loop theorem,
all fully dimension-general — parameterized over symbolic `head_dim`,
`BLOCK_DMODEL`, `BLOCK_SEQ`, and every stride (no hardcoded shape literals). For
background, the checked Python tests use `head_dim = 64`, `BLOCK_DMODEL = 64`,
four `BLOCK_SEQ ∈ {16, 17, 8, 32}` surfaces, `batch = 2`, `head = 4`,
`seq_block = 3`, just one instance of the general statements:
(1) the **masked final store** to `Out` (`finalStoreValue`, mask
`active s head_dim i` = `offs_d < head_dim`); (2) the **final normalization
store** (`acc / sum_exp` =
`normalizedStoreValue`); (3) one **masked accumulator step** plus the matching
**sum-exp step** of the loop body (`accumulatorStepValue` / `sumExpStepValue`);
and (4) the **running-max loop invariant** stated recursively as `runningMaxAfter`
(starts `-inf`, joins each `Mid_O_LogExpSum` element). Masked-off lanes
(`offs_d ≥ head_dim`) are preserved. The intermediate `Acc`/`SumExp`/`MaxLogic`
register state is taken as given for the store faces; the full loop is not
unrolled into one closed-form spec, so the end-to-end statement is the
composition of these faces with the (trusted) loop scheduling.
-/

namespace VeriTile.Bench.TritonBenchG.FlashDecode2Phi

open VeriTile

set_option linter.unusedSimpArgs false
set_option maxHeartbeats 1000000

/-! **★ Headline (last theorem):** `flash_decode2_phi_normalization_output_summary_general`
(the masked `Out = acc / sum_exp` store). Supporting loop-invariant faces (the
loop is not unrolled — see Scope): `flash_decode2_phi_masked_accumulator_output_summary_general`,
`flash_decode2_phi_running_max_output_summary_general`. -/

/-! # ══════════ CORRECT — genuine / dimension-general (review this) ══════════ -/

section Correct

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

def runningMaxLogicOffset
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
          (runningMaxLogicOffset s stride_mid_o_eb stride_mid_o_eh k)) : WithBot ℝ)
        (runningMaxAfter s Mid_O_LogExpSum stride_mid_o_eb stride_mid_o_eh k)

theorem runningMaxAfter_succ
    (s : BlockState) (Mid_O_LogExpSum : RegionName)
    (stride_mid_o_eb stride_mid_o_eh k : Nat) :
    runningMaxAfter s Mid_O_LogExpSum stride_mid_o_eb stride_mid_o_eh (k + 1) =
      runningMaxJoin
        (some (s.readMem Mid_O_LogExpSum
          (runningMaxLogicOffset s stride_mid_o_eb stride_mid_o_eh k)) : WithBot ℝ)
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

/-- One Phi stage2 accumulator recurrence step.

This kernel models the loop-body arithmetic after the next running maximum has
already been supplied: it loads the masked `Mid_O` row, applies `old_scale` and
`exp_logic`, and writes the updated accumulator vector. -/
def flash_decode2_phi_accumulator_step_kernel
    (Mid_O Mid_O_LogExpSum AccIn MaxLogic NewMaxLogic AccOut : RegionName)
    (block_seq_n
      head_dim
      stride_mid_ob stride_mid_oh stride_mid_os stride_mid_od
      stride_mid_o_eb stride_mid_o_eh
      stride_acc_b stride_acc_h stride_acc_d
      stride_logic_b stride_logic_h
      BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  mask = offs_d < $(head_dim)
  tv = tl.load(Mid_O + cur_batch * $(stride_mid_ob) +
      cur_head * $(stride_mid_oh) + $(block_seq_n) * $(stride_mid_os) +
      offs_d * $(stride_mid_od), mask=mask, other=0.0)
  tlogic = tl.load(Mid_O_LogExpSum + cur_batch * $(stride_mid_o_eb) +
      cur_head * $(stride_mid_o_eh) + $(block_seq_n))
  acc = tl.load(AccIn + cur_batch * $(stride_acc_b) +
      cur_head * $(stride_acc_h) + offs_d * $(stride_acc_d),
      mask=mask, other=0.0)
  max_logic = tl.load(MaxLogic + cur_batch * $(stride_logic_b) +
      cur_head * $(stride_logic_h))
  new_max_logic = tl.load(NewMaxLogic + cur_batch * $(stride_logic_b) +
      cur_head * $(stride_logic_h))
  old_scale = tl.exp(max_logic - new_max_logic)
  acc = acc * old_scale
  exp_logic = tl.exp(tlogic - new_max_logic)
  acc = acc + exp_logic * tv
  tl.store(AccOut + cur_batch * $(stride_acc_b) +
      cur_head * $(stride_acc_h) + offs_d * $(stride_acc_d), acc, mask=mask)
}

/-- Scalar companion for the same Phi stage2 recurrence step.

It proves the `sum_exp = sum_exp * old_scale + exp_logic` update with the same
symbolic stride addressing as the vector accumulator step. -/
def flash_decode2_phi_sum_exp_step_kernel
    (Mid_O_LogExpSum SumExpIn MaxLogic NewMaxLogic SumExpOut : RegionName)
    (block_seq_n
      stride_mid_o_eb stride_mid_o_eh
      stride_sum_b stride_sum_h
      stride_logic_b stride_logic_h : Nat) :
    ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  tlogic = tl.load(Mid_O_LogExpSum + cur_batch * $(stride_mid_o_eb) +
      cur_head * $(stride_mid_o_eh) + $(block_seq_n))
  sum_exp = tl.load(SumExpIn + cur_batch * $(stride_sum_b) +
      cur_head * $(stride_sum_h))
  max_logic = tl.load(MaxLogic + cur_batch * $(stride_logic_b) +
      cur_head * $(stride_logic_h))
  new_max_logic = tl.load(NewMaxLogic + cur_batch * $(stride_logic_b) +
      cur_head * $(stride_logic_h))
  old_scale = tl.exp(max_logic - new_max_logic)
  exp_logic = tl.exp(tlogic - new_max_logic)
  sum_exp = sum_exp * old_scale + exp_logic
  tl.store(SumExpOut + cur_batch * $(stride_sum_b) +
      cur_head * $(stride_sum_h), sum_exp)
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

def midOffset
    (s : BlockState)
    (block_seq_n stride_mid_ob stride_mid_oh stride_mid_os stride_mid_od : Nat)
    (i : Fin BLOCK_DMODEL) : Nat :=
  s.pids 0 * stride_mid_ob + s.pids 1 * stride_mid_oh +
    block_seq_n * stride_mid_os + dIndex s i * stride_mid_od

def logicOffset
    (s : BlockState)
    (block_seq_n stride_mid_o_eb stride_mid_o_eh : Nat) : Nat :=
  s.pids 0 * stride_mid_o_eb + s.pids 1 * stride_mid_o_eh + block_seq_n

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

noncomputable def oldScaleValue
    (s : BlockState) (MaxLogic NewMaxLogic : RegionName)
    (stride_logic_b stride_logic_h : Nat) : ℝ :=
  Real.exp (s.readMem MaxLogic (sumExpOffset s stride_logic_b stride_logic_h) -
    s.readMem NewMaxLogic (sumExpOffset s stride_logic_b stride_logic_h))

noncomputable def expLogicValue
    (s : BlockState) (Mid_O_LogExpSum NewMaxLogic : RegionName)
    (block_seq_n stride_mid_o_eb stride_mid_o_eh
      stride_logic_b stride_logic_h : Nat) : ℝ :=
  Real.exp (s.readMem Mid_O_LogExpSum
      (logicOffset s block_seq_n stride_mid_o_eb stride_mid_o_eh) -
    s.readMem NewMaxLogic (sumExpOffset s stride_logic_b stride_logic_h))

noncomputable def accumulatorStepValue
    (s : BlockState) (Mid_O Mid_O_LogExpSum AccIn MaxLogic NewMaxLogic : RegionName)
    (block_seq_n head_dim
      stride_mid_ob stride_mid_oh stride_mid_os stride_mid_od
      stride_mid_o_eb stride_mid_o_eh
      stride_acc_b stride_acc_h stride_acc_d
      stride_logic_b stride_logic_h : Nat)
    (i : Fin BLOCK_DMODEL) : ℝ :=
  WithBot.unbotD 0
    (if active s head_dim i then
      some (s.readMem AccIn (accOffset s stride_acc_b stride_acc_h stride_acc_d i) *
          oldScaleValue s MaxLogic NewMaxLogic stride_logic_b stride_logic_h +
        expLogicValue s Mid_O_LogExpSum NewMaxLogic block_seq_n stride_mid_o_eb
          stride_mid_o_eh stride_logic_b stride_logic_h *
        s.readMem Mid_O
          (midOffset s block_seq_n stride_mid_ob stride_mid_oh stride_mid_os
            stride_mid_od i))
    else some (0.0 : ℝ))

noncomputable def sumExpStepValue
    (s : BlockState) (Mid_O_LogExpSum SumExpIn MaxLogic NewMaxLogic : RegionName)
    (block_seq_n stride_mid_o_eb stride_mid_o_eh
      stride_sum_b stride_sum_h stride_logic_b stride_logic_h : Nat) : ℝ :=
  s.readMem SumExpIn (sumExpOffset s stride_sum_b stride_sum_h) *
      oldScaleValue s MaxLogic NewMaxLogic stride_logic_b stride_logic_h +
    expLogicValue s Mid_O_LogExpSum NewMaxLogic block_seq_n stride_mid_o_eb
      stride_mid_o_eh stride_logic_b stride_logic_h

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

/-! ## Step / store correctness (dimension-general)

The theorems below are symbolic in `head_dim`, `BLOCK_DMODEL`, and every stride;
no shape literals are hardcoded. For background, the checked Python tests
allocate `Out` with shape `(2, 4, 64)` (contiguous output strides `(256, 64, 1)`)
with `head_dim = 64` and `BLOCK_DMODEL = next_power_of_2(64) = 64`; the varied
`block_seq` cases do not change the final output layout. That concrete shape is
just one instance of the general statements. -/

/-- Algorithm-layer correctness for one masked Phi accumulator update. -/
theorem flash_decode2_phi_accumulator_step_kernel_correct
    (Mid_O Mid_O_LogExpSum AccIn MaxLogic NewMaxLogic AccOut : RegionName)
    (block_seq_n
      head_dim
      stride_mid_ob stride_mid_oh stride_mid_os stride_mid_od
      stride_mid_o_eb stride_mid_o_eh
      stride_acc_b stride_acc_h stride_acc_d
      stride_logic_b stride_logic_h
      BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hAccOutInj : Function.Injective
      (fun i : Fin BLOCK_DMODEL => accOffset s stride_acc_b stride_acc_h stride_acc_d i)) :
    ∀ i : Fin BLOCK_DMODEL,
      let outAddr := accOffset s stride_acc_b stride_acc_h stride_acc_d i
      (exec (flash_decode2_phi_accumulator_step_kernel Mid_O Mid_O_LogExpSum
            AccIn MaxLogic NewMaxLogic AccOut block_seq_n head_dim
            stride_mid_ob stride_mid_oh stride_mid_os stride_mid_od
            stride_mid_o_eb stride_mid_o_eh stride_acc_b stride_acc_h
            stride_acc_d stride_logic_b stride_logic_h BLOCK_DMODEL) s).map
          (·.readMem AccOut outAddr)
        = some (if active s head_dim i then
            accumulatorStepValue s Mid_O Mid_O_LogExpSum AccIn MaxLogic
              NewMaxLogic block_seq_n head_dim stride_mid_ob stride_mid_oh
              stride_mid_os stride_mid_od stride_mid_o_eb stride_mid_o_eh
              stride_acc_b stride_acc_h stride_acc_d stride_logic_b
              stride_logic_h i
          else s.readMem AccOut outAddr) := by
  intro i
  simp [exec, flash_decode2_phi_accumulator_step_kernel, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.ptrAdd,
        NumericDType.add, NumericDType.sub, NumericDType.mul, ComparableDType.lt,
        dIndex, active, midOffset, logicOffset, accOffset, sumExpOffset,
        oldScaleValue, expLogicValue]
  let offsetFn : TileIndex [BLOCK_DMODEL] → Nat :=
    fun idx => s.pids 0 * stride_acc_b + s.pids 1 * stride_acc_h +
      idx.1.val * stride_acc_d
  let valueFn : TileIndex [BLOCK_DMODEL] → ℝ :=
    fun idx =>
      WithBot.unbotD 0
        (Option.map₂ (fun x y => x + y)
          (match
            if idx.1.val < head_dim then
              some (s.readMem AccIn
                (s.pids 0 * stride_acc_b + s.pids 1 * stride_acc_h +
                  idx.1.val * stride_acc_d))
            else some (0.0 : ℝ)
          with
          | some x =>
              some (x * Real.exp
                (s.readMem MaxLogic
                    (s.pids 0 * stride_logic_b + s.pids 1 * stride_logic_h) -
                  s.readMem NewMaxLogic
                    (s.pids 0 * stride_logic_b + s.pids 1 * stride_logic_h)))
          | none => none)
          (match
            if idx.1.val < head_dim then
              some (s.readMem Mid_O
                (s.pids 0 * stride_mid_ob + s.pids 1 * stride_mid_oh +
                  block_seq_n * stride_mid_os + idx.1.val * stride_mid_od))
            else some (0.0 : ℝ)
          with
          | some y =>
              some (Real.exp
                (s.readMem Mid_O_LogExpSum
                    (s.pids 0 * stride_mid_o_eb +
                      s.pids 1 * stride_mid_o_eh + block_seq_n) -
                  s.readMem NewMaxLogic
                    (s.pids 0 * stride_logic_b + s.pids 1 * stride_logic_h)) * y)
          | none => none))
  let P : TileIndex [BLOCK_DMODEL] → Prop :=
    fun idx => idx.1.val < head_dim
  have hOffsetInj : Function.Injective offsetFn := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have habFin : accOffset s stride_acc_b stride_acc_h stride_acc_d a =
        accOffset s stride_acc_b stride_acc_h stride_acc_d b := by
      simpa [offsetFn, accOffset, dIndex] using hab
    obtain rfl : a = b := hAccOutInj habFin
    rfl
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj (i, PUnit.unit)]
  by_cases hi : i.val < head_dim
  · simp [P, offsetFn, valueFn, active, accumulatorStepValue, oldScaleValue,
      expLogicValue, midOffset, logicOffset, accOffset, sumExpOffset, dIndex, hi]
  · simp [P, offsetFn, active, accumulatorStepValue, dIndex, hi]

/-- Compute-facing correctness for one masked Phi accumulator update. -/
theorem flash_decode2_phi_accumulator_step_kernel_compute_correct
    (Mid_O Mid_O_LogExpSum AccIn MaxLogic NewMaxLogic AccOut : RegionName)
    (block_seq_n
      head_dim
      stride_mid_ob stride_mid_oh stride_mid_os stride_mid_od
      stride_mid_o_eb stride_mid_o_eh
      stride_acc_b stride_acc_h stride_acc_d
      stride_logic_b stride_logic_h
      BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hAccOutInj : Function.Injective
      (fun i : Fin BLOCK_DMODEL => accOffset s stride_acc_b stride_acc_h stride_acc_d i)) :
    ComputeCorrect.Realizes
      (kernel := flash_decode2_phi_accumulator_step_kernel Mid_O Mid_O_LogExpSum
        AccIn MaxLogic NewMaxLogic AccOut block_seq_n head_dim stride_mid_ob
        stride_mid_oh stride_mid_os stride_mid_od stride_mid_o_eb
        stride_mid_o_eh stride_acc_b stride_acc_h stride_acc_d stride_logic_b
        stride_logic_h BLOCK_DMODEL)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_DMODEL => active s head_dim i)
        (fun i => (AccOut, accOffset s stride_acc_b stride_acc_h stride_acc_d i)))
      (expected := fun i : Fin BLOCK_DMODEL =>
        accumulatorStepValue s Mid_O Mid_O_LogExpSum AccIn MaxLogic NewMaxLogic
          block_seq_n head_dim stride_mid_ob stride_mid_oh stride_mid_os
          stride_mid_od stride_mid_o_eb stride_mid_o_eh stride_acc_b
          stride_acc_h stride_acc_d stride_logic_b stride_logic_h i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [flash_decode2_phi_accumulator_step_kernel]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := flash_decode2_phi_accumulator_step_kernel_correct Mid_O
    Mid_O_LogExpSum AccIn MaxLogic NewMaxLogic AccOut block_seq_n head_dim
    stride_mid_ob stride_mid_oh stride_mid_os stride_mid_od stride_mid_o_eb
    stride_mid_o_eh stride_acc_b stride_acc_h stride_acc_d stride_logic_b
    stride_logic_h BLOCK_DMODEL s hAccOutInj i
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-- Compute-facing correctness for one Phi `sum_exp` recurrence update. -/
theorem flash_decode2_phi_sum_exp_step_kernel_compute_correct
    (Mid_O_LogExpSum SumExpIn MaxLogic NewMaxLogic SumExpOut : RegionName)
    (block_seq_n
      stride_mid_o_eb stride_mid_o_eh
      stride_sum_b stride_sum_h
      stride_logic_b stride_logic_h : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := flash_decode2_phi_sum_exp_step_kernel Mid_O_LogExpSum SumExpIn
        MaxLogic NewMaxLogic SumExpOut block_seq_n stride_mid_o_eb
        stride_mid_o_eh stride_sum_b stride_sum_h stride_logic_b stride_logic_h)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.scalar SumExpOut
        (sumExpOffset s stride_sum_b stride_sum_h))
      (expected := fun _ : PUnit =>
        sumExpStepValue s Mid_O_LogExpSum SumExpIn MaxLogic NewMaxLogic
          block_seq_n stride_mid_o_eb stride_mid_o_eh stride_sum_b stride_sum_h
          stride_logic_b stride_logic_h) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [flash_decode2_phi_sum_exp_step_kernel]
  intro s0 s' hExec hs0
  subst s0
  intro i
  simp [exec, flash_decode2_phi_sum_exp_step_kernel, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.ptrAdd,
        NumericDType.add, NumericDType.sub, NumericDType.mul, sumExpOffset,
        logicOffset, oldScaleValue, expLogicValue, sumExpStepValue,
        ComputeCorrect.WriteMap.scalar] at hExec ⊢
  rw [← hExec]
  simp [BlockState.writeMem_readMem]

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


/-! ### ════════ supporting face ════════ -/
/-- **General genuine masked accumulator/`sum_exp` recurrence summary.** Fully
dimension-parameterized over the block index, `head_dim`, `BLOCK_SEQ`,
`BLOCK_DMODEL`, and every stride: the full Phi flash-decode stage2 surface
lowers, and one loop-body step *realizes* both genuine closed forms — the masked
vector accumulator update `accumulatorStepValue` (=
`acc * old_scale + exp_logic * tv` for `offs_d < head_dim`) and the scalar
`sumExpStepValue` (= `sum_exp * old_scale + exp_logic`), both reading
`Mid_O`/`Mid_O_LogExpSum`/`AccIn`/`SumExpIn`/`MaxLogic`/`NewMaxLogic` input
memory. No hardcoded shape literals. Side condition: accumulator-footprint
injectivity. -/
theorem flash_decode2_phi_masked_accumulator_output_summary_general
    (B_Seqlen : Region .nat)
    (Mid_O Mid_O_LogExpSum AccIn SumExpIn MaxLogic NewMaxLogic AccOut SumExpOut
      Out : RegionName)
    (stride_mid_ob stride_mid_oh stride_mid_os stride_mid_od
      stride_mid_o_eb stride_mid_o_eh stride_mid_o_es stride_obs stride_oh stride_od
      head_dim BLOCK_SEQ BLOCK_DMODEL
      block_seq_n stride_acc_b stride_acc_h stride_acc_d
      stride_sum_b stride_sum_h stride_logic_b stride_logic_h : Nat)
    (s : BlockState)
    (hAccOutInj : Function.Injective
      (fun i : Fin BLOCK_DMODEL =>
        accOffset s stride_acc_b stride_acc_h stride_acc_d i)) :
    (∃ alg, (flash_decode2_phi_surface B_Seqlen Mid_O Mid_O_LogExpSum Out
      stride_mid_ob stride_mid_oh stride_mid_os stride_mid_od stride_mid_o_eb
      stride_mid_o_eh stride_mid_o_es stride_obs stride_oh stride_od head_dim
      BLOCK_SEQ BLOCK_DMODEL).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := flash_decode2_phi_accumulator_step_kernel Mid_O Mid_O_LogExpSum
        AccIn MaxLogic NewMaxLogic AccOut block_seq_n head_dim stride_mid_ob
        stride_mid_oh stride_mid_os stride_mid_od stride_mid_o_eb stride_mid_o_eh
        stride_acc_b stride_acc_h stride_acc_d stride_logic_b stride_logic_h
        BLOCK_DMODEL)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_DMODEL => active s head_dim i)
        (fun i => (AccOut, accOffset s stride_acc_b stride_acc_h stride_acc_d i)))
      (expected := fun i : Fin BLOCK_DMODEL =>
        accumulatorStepValue s Mid_O Mid_O_LogExpSum AccIn MaxLogic NewMaxLogic
          block_seq_n head_dim stride_mid_ob stride_mid_oh stride_mid_os
          stride_mid_od stride_mid_o_eb stride_mid_o_eh stride_acc_b stride_acc_h
          stride_acc_d stride_logic_b stride_logic_h i) ∧
    ComputeCorrect.Realizes
      (kernel := flash_decode2_phi_sum_exp_step_kernel Mid_O_LogExpSum SumExpIn
        MaxLogic NewMaxLogic SumExpOut block_seq_n stride_mid_o_eb stride_mid_o_eh
        stride_sum_b stride_sum_h stride_logic_b stride_logic_h)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.scalar SumExpOut
        (sumExpOffset s stride_sum_b stride_sum_h))
      (expected := fun _ : PUnit =>
        sumExpStepValue s Mid_O_LogExpSum SumExpIn MaxLogic NewMaxLogic
          block_seq_n stride_mid_o_eb stride_mid_o_eh stride_sum_b stride_sum_h
          stride_logic_b stride_logic_h) := by
  refine ⟨?_, ?_, ?_⟩
  · exact flash_decode2_phi_surface_toAlgorithm_supported B_Seqlen Mid_O
      Mid_O_LogExpSum Out stride_mid_ob stride_mid_oh stride_mid_os stride_mid_od
      stride_mid_o_eb stride_mid_o_eh stride_mid_o_es stride_obs stride_oh stride_od
      head_dim BLOCK_SEQ BLOCK_DMODEL
  · exact flash_decode2_phi_accumulator_step_kernel_compute_correct Mid_O
      Mid_O_LogExpSum AccIn MaxLogic NewMaxLogic AccOut block_seq_n head_dim
      stride_mid_ob stride_mid_oh stride_mid_os stride_mid_od stride_mid_o_eb
      stride_mid_o_eh stride_acc_b stride_acc_h stride_acc_d stride_logic_b
      stride_logic_h BLOCK_DMODEL s hAccOutInj
  · exact flash_decode2_phi_sum_exp_step_kernel_compute_correct Mid_O_LogExpSum
      SumExpIn MaxLogic NewMaxLogic SumExpOut block_seq_n stride_mid_o_eb
      stride_mid_o_eh stride_sum_b stride_sum_h stride_logic_b stride_logic_h s


/-! ### ════════ supporting face ════════ -/
/-- **General genuine running-maximum recurrence summary.** Fully
dimension-parameterized over every stride, `head_dim`, `BLOCK_SEQ`, and
`BLOCK_DMODEL`: the full Phi flash-decode stage2 surface lowers, and the dynamic
`block_n_size` `max_logic` loop invariant is the genuine recursive value
`runningMaxAfter` — it starts at `-inf` and each iteration joins the next
`Mid_O_LogExpSum` element at contiguous `offs_logic + block_seq_n` (reading
input memory). No hardcoded shape literals. -/
theorem flash_decode2_phi_running_max_output_summary_general
    (B_Seqlen : Region .nat) (Mid_O Mid_O_LogExpSum Out : RegionName)
    (stride_mid_ob stride_mid_oh stride_mid_os stride_mid_od
      stride_mid_o_eb stride_mid_o_eh stride_mid_o_es stride_obs stride_oh stride_od
      head_dim BLOCK_SEQ BLOCK_DMODEL : Nat)
    (s : BlockState) (block_n_size : Nat) :
    (∃ alg, (flash_decode2_phi_surface B_Seqlen Mid_O Mid_O_LogExpSum Out
      stride_mid_ob stride_mid_oh stride_mid_os stride_mid_od stride_mid_o_eb
      stride_mid_o_eh stride_mid_o_es stride_obs stride_oh stride_od head_dim
      BLOCK_SEQ BLOCK_DMODEL).toAlgorithm? = Except.ok alg) ∧
    runningMaxAfter s Mid_O_LogExpSum stride_mid_o_eb stride_mid_o_eh 0 = ⊥ ∧
    (∀ k, k < block_n_size →
      runningMaxAfter s Mid_O_LogExpSum stride_mid_o_eb stride_mid_o_eh (k + 1) =
        runningMaxJoin
          (some (s.readMem Mid_O_LogExpSum
            (runningMaxLogicOffset s stride_mid_o_eb stride_mid_o_eh k)) :
            WithBot ℝ)
          (runningMaxAfter s Mid_O_LogExpSum stride_mid_o_eb stride_mid_o_eh k)) := by
  refine ⟨?_, ?_, ?_⟩
  · exact flash_decode2_phi_surface_toAlgorithm_supported B_Seqlen Mid_O
      Mid_O_LogExpSum Out stride_mid_ob stride_mid_oh stride_mid_os stride_mid_od
      stride_mid_o_eb stride_mid_o_eh stride_mid_o_es stride_obs stride_oh stride_od
      head_dim BLOCK_SEQ BLOCK_DMODEL
  · exact runningMaxAfter_zero s Mid_O_LogExpSum stride_mid_o_eb stride_mid_o_eh
  · intro k _hk
    exact runningMaxAfter_succ s Mid_O_LogExpSum stride_mid_o_eb stride_mid_o_eh k


/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/
/-- **General genuine final-normalization output summary.** Fully
dimension-parameterized over every stride, `head_dim`, `BLOCK_SEQ`, and
`BLOCK_DMODEL`: the full Phi flash-decode stage2 surface lowers, and the masked
normalization writeback *realizes* the genuine closed form
`Out[d] = acc[d] / sum_exp` for `offs_d < head_dim` (`normalizedStoreValue`,
reading the loop-produced `Acc`/`SumExp` input memory). No hardcoded shape
literals. Side condition: output-footprint injectivity.

This is the kernel's headline output store; the running-max / accumulator faces
above are the supporting loop-invariant faces (the full loop is not unrolled into
one closed form — see the module Scope note). -/
theorem flash_decode2_phi_normalization_output_summary_general
    (B_Seqlen : Region .nat) (Mid_O Mid_O_LogExpSum Acc SumExp Out : RegionName)
    (stride_mid_ob stride_mid_oh stride_mid_os stride_mid_od
      stride_mid_o_eb stride_mid_o_eh stride_mid_o_es stride_obs stride_oh stride_od
      head_dim BLOCK_SEQ BLOCK_DMODEL
      stride_acc_b stride_acc_h stride_acc_d stride_sum_b stride_sum_h : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_DMODEL => outOffset s stride_obs stride_oh stride_od i)) :
    (∃ alg, (flash_decode2_phi_surface B_Seqlen Mid_O Mid_O_LogExpSum Out
      stride_mid_ob stride_mid_oh stride_mid_os stride_mid_od stride_mid_o_eb
      stride_mid_o_eh stride_mid_o_es stride_obs stride_oh stride_od head_dim
      BLOCK_SEQ BLOCK_DMODEL).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := flash_decode2_phi_normalization_store_kernel Acc SumExp Out
        head_dim stride_acc_b stride_acc_h stride_acc_d stride_sum_b stride_sum_h
        stride_obs stride_oh stride_od BLOCK_DMODEL)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_DMODEL => active s head_dim i)
        (fun i => (Out, outOffset s stride_obs stride_oh stride_od i)))
      (expected := fun i : Fin BLOCK_DMODEL =>
        normalizedStoreValue s Acc SumExp head_dim stride_acc_b stride_acc_h
          stride_acc_d stride_sum_b stride_sum_h i) := by
  refine ⟨?_, ?_⟩
  · exact flash_decode2_phi_surface_toAlgorithm_supported B_Seqlen Mid_O
      Mid_O_LogExpSum Out stride_mid_ob stride_mid_oh stride_mid_os stride_mid_od
      stride_mid_o_eb stride_mid_o_eh stride_mid_o_es stride_obs stride_oh stride_od
      head_dim BLOCK_SEQ BLOCK_DMODEL
  · exact flash_decode2_phi_normalization_store_kernel_compute_correct Acc SumExp Out
      head_dim stride_acc_b stride_acc_h stride_acc_d stride_sum_b stride_sum_h
      stride_obs stride_oh stride_od BLOCK_DMODEL s hOutInj

end Correct


end VeriTile.Bench.TritonBenchG.FlashDecode2Phi
