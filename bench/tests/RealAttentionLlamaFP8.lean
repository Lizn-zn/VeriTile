import bench.tritonbench_g.attention_llama.Proofs.FP8RealLoop
import bench.tritonbench_g.attention_llama.Proofs.FP8RealMemory
import bench.tritonbench_g.attention_llama.Proofs.FP8RealFlat

/-! Real FP8 input decoding keeps byte interpretation, masking and memory
effects explicit. These are regression fixtures, not replacements for the
parameterized attention correctness obligation. -/
namespace RealAttentionLlamaFP8Tests
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AttentionLlama
open RealKernel
set_option maxRecDepth 16000
set_option maxHeartbeats 200000
set_option linter.unusedSimpArgs false

private def laneState : BlockState :=
  { (default : BlockState) with
    mem := fun _ o => .of .int (if o = 0 then 60 else if o = 1 then 127 else 128) }
private def lanePtrs : Tile .ptr [3] := ⟨fun i => ("K", i.1.val)⟩
private def laneMasks : Tile .bool [3] := ⟨fun i => decide (i.1.val = 0)⟩
private def maskedState : BlockState :=
  (laneState.setReg "ptr" .ptr [3] lanePtrs).setReg "mask" .bool [3] laneMasks
private def ptrExpr : Op .ptr [3] := .ref .ptr [3] "ptr"
private def maskExpr : Op .bool [3] := .ref .bool [3] "mask"
private def maskedBytes : Op .int [3] :=
  .load .int (.ptr ptrExpr) (.maskOther maskExpr (.broadcast (.constInt 0) [3]))

/-- A masked NaN byte and a masked out-of-range integer do not invalidate
the active finite lane. The public bridge, not a changed input buffer,
justifies replacing this particular load by the decoded input view. -/
theorem masked_invalid_lanes_skipped :
    @evalOp .real [3] (.decodeFloat8E5 .real maskedBytes) maskedState =
      some ⟨fun i => if i.1.val = 0 then some 1 else some 0⟩ := by
  have bridge := FP8RealMemory.masked_decode "K" "V" maskedState ptrExpr maskExpr
    lanePtrs laneMasks
    (by simp +decide [ptrExpr, evalOp_ref, maskedState])
    (by simp [maskExpr, evalOp_ref, maskedState])
    (by simp [ptrExpr, Op.SafeAt.eq_def])
    (by simp [maskExpr, Op.SafeAt.eq_def])
    (by intro i hi; exact Or.inl rfl)
    (by
      intro i hi
      have hz : i.1.val = 0 := by simpa [laneMasks] using hi
      simp [maskedState, laneState, lanePtrs, BlockState.readMemValue,
        BlockState.readMemTyped, MemCell.readAs, MemCell.of, hz,
        FP8RealMemory.FiniteByte, Float8E5Bits.isInt8,
        Float8E5Bits.ofInt, Float8E5Bits.exponent])
  refine bridge.trans ?_
  rw [RealKernel.al_load_ptr_maskOther_real ptrExpr maskExpr
    (.broadcast (.const 0) [3]) (FP8RealMemory.decoded "K" "V" maskedState)
    lanePtrs laneMasks (⟨fun _ => some 0⟩ : Tile .real [3])
    (by simp +decide [evalOp_ref, ptrExpr, FP8RealMemory.decoded, maskedState])
    (by simp [evalOp_ref, maskExpr, FP8RealMemory.decoded, maskedState])
    (by simp only [evalOp, bind, Option.bind_some, Tile.scalar_data])]
  apply congrArg some
  apply Tile.ext
  intro i
  by_cases hi : i.1.val = 0
  · simp only [lanePtrs, laneMasks, hi, decide_true, ↓reduceIte]
    rw [FP8RealMemory.read_decoded "K" "V" maskedState "K" 0 (Or.inl rfl)]
    unfold FP8RealMemory.value
    rw [show maskedState.readMemValue .int "K" 0 = 60 from rfl]
    norm_num [Int.toNat, Float8E5Bits.ofInt, Float8E5Bits.decodeRat,
      Float8E5Bits.exponent, Float8E5Bits.fraction, Float8E5Bits.sign]
  · have hfin : i.1 ≠ 0 := by intro h; exact hi (congrArg Fin.val h)
    simp [laneMasks, hi, hfin]

private def rawRows (validRows : Nat) : BlockState :=
  { (default : BlockState) with
    mem := fun _ o => .of .int (if o < validRows then 60 else 127) }

private theorem finite_row (validRows offset : Nat) (h : offset < validRows) :
    FP8RealMemory.FiniteByte ((rawRows validRows).readMemValue .int "K" offset) := by
  change FP8RealMemory.FiniteByte (if offset < validRows then 60 else 127)
  rw [if_pos h]
  unfold FP8RealMemory.FiniteByte
  decide

/-- With block size two and three keys, the last block's fourth lane is
masked out, so a NaN there does not invalidate the range's byte contract. -/
theorem partial_tail_ignores_nan :
    FP8RealLoop.BytesFrom "K" (rawRows 3) 0 1 0 0 3 3 2 1 := by
  intro k hk j d hj
  apply finite_row
  simpa using hj

/-- Keys beyond the dynamic stop are not read even when they lie within N.
Their NaNs must not become extra preconditions for the visited blocks. -/
theorem unvisited_blocks_ignore_nan :
    FP8RealLoop.BytesFrom "K" (rawRows 2) 0 1 0 0 2 4 2 1 := by
  intro k hk j d _
  apply finite_row
  have hj := j.isLt
  simp only [Nat.zero_add, Nat.mul_one, Nat.mul_zero, Nat.add_zero]
  omega

/-- An active NaN byte, in contrast, has no real projection. -/
theorem active_nan_rejects_range :
    ¬ FP8RealLoop.BytesFrom "K" (rawRows 1) 0 1 0 0 2 2 2 1 := by
  intro h
  have hf := h 0 (by decide) ⟨1, by decide⟩ ⟨0, by decide⟩ (by decide)
  change FP8RealMemory.FiniteByte 127 at hf
  exact (by unfold FP8RealMemory.FiniteByte; decide : ¬ FP8RealMemory.FiniteByte 127) hf

/- Whole-source regression: two noncausal streaming iterations, one causal
iteration, a masked output row, distinct decoded values and negative-zero raw
keys. The byte encoding must survive even though real decoding erases its sign. -/
private def initial : BlockState :=
  { (default : BlockState) with
    mem := fun r o => if r = "K" then .of .int (-128)
      else if r = "V" then .of .int (if o = 0 then 60 else 64)
      else if r = "Out" then .real 99 else .real 0
    pids := fun _ => 0 }

private def kernel (candidate causal : Bool) :=
  FP8RealSource.source "Q" "K" "V" "Out" 1
    2 2 1 1 2 2 1 1 2 2 1 1 2 2 1 1 1 1 2 0 2 1 1 causal candidate

private theorem output_injective : Function.Injective (fun i : TileIndex [2, 1] =>
    (initial.pids 0 * 2 + i.1.val) * 1 + i.2.1.val * 1) := by
  simpa only [initial, Nat.zero_mul, Nat.zero_add, Nat.mul_one] using
    (rowMajor2D_inj (I := 2) (J := 1) 0 1 (by decide))

private theorem valid_keys (stop : Nat) :
    FP8RealLoop.BytesFrom "K" initial (RealKernel.alBase initial 1 2 2) 1 1 0 stop 2 1 1 := by
  intro k hk j d hj
  change FP8RealMemory.FiniteByte (-128)
  unfold FP8RealMemory.FiniteByte
  decide

private theorem valid_values (stop : Nat) :
    FP8RealLoop.BytesFrom "V" initial (RealKernel.alBase initial 1 2 2) 1 1 0 stop 2 1 1 := by
  intro k hk j d hj
  change FP8RealMemory.FiniteByte (if _ = 0 then 60 else 64)
  split <;> unfold FP8RealMemory.FiniteByte <;> decide

private theorem query_zero : RealKernel.alQTileG initial "Q" 1 2 2 1 1 1 2 1 = fun _ => 0 := by
  funext i
  unfold RealKernel.alQTileG
  split <;> rfl

private theorem keys_zero (SEQ : Nat) :
    FP8RealOutput.byteTile initial "K" 1 2 2 1 1 SEQ 1 = fun _ => 0 := by
  funext i
  unfold FP8RealOutput.byteTile FP8RealMemory.value
  rw [show initial.readMemValue .int "K" _ = -128 from rfl]
  norm_num [Float8E5Bits.ofInt, Float8E5Bits.decodeRat, Float8E5Bits.exponent,
    Float8E5Bits.fraction, Float8E5Bits.sign, Int.toNat]

private theorem distinct_values (SEQ : Nat) :
    FP8RealOutput.byteTile initial "V" 1 2 2 1 1 SEQ 1 =
      fun i => if i.1.val = 0 then 1 else 2 := by
  funext i
  have hd : i.2.1.val = 0 := by omega
  unfold FP8RealOutput.byteTile FP8RealMemory.value
  simp only [RealKernel.alBase, initial, Nat.zero_div, Nat.zero_mod, Nat.zero_mul,
    Nat.zero_add, Nat.mul_one, hd, Nat.add_zero]
  change (((Float8E5Bits.ofInt (if i.1.val = 0 then 60 else 64)).decodeRat.getD 0 : ℚ) : ℝ) = _
  split <;> norm_num [Float8E5Bits.ofInt, Float8E5Bits.decodeRat, Float8E5Bits.exponent,
    Float8E5Bits.fraction, Float8E5Bits.sign, Int.toNat]

/-- The actual byte source returns 3/2 noncausally and one causally. The
masked output and original negative-zero K byte are preserved exactly. -/
theorem source_output_and_raw_frame (candidate causal : Bool) :
    ∃ t, exec (kernel candidate causal).eraseDType.toAlgKernel initial = some t ∧
      t.readMem "Out" 0 = (if causal then 1 else (3 / 2 : ℝ)) ∧
      t.mem "Out" 1 = .real 99 ∧ t.mem "K" 0 = .of .int (-128) ∧
      t.mem "V" 1 = .of .int 64 := by
  cases causal
  case' «false» =>
    obtain ⟨t, hrun, hread, hframe⟩ := FP8RealOutput.noncausal_correct
      "Q" "K" "V" "Out" 1
      2 2 1 1 2 2 1 1 2 2 1 1 2 2 1 1 1 1 2 0 2 1 1 candidate initial 2
      (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) rfl output_injective (by intros; rfl)
      (valid_keys 2) (valid_values 2)
  case' «true» =>
    obtain ⟨t, hrun, hread, hframe⟩ := FP8RealOutput.causal_correct
      "Q" "K" "V" "Out" 1
      2 2 1 1 2 2 1 1 2 2 1 1 2 2 1 1 1 1 2 0 2 1 1 candidate initial 1
      (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) rfl (by decide) (by decide)
      output_injective (by intros; rfl) (valid_keys 1) (valid_values 1)
  all_goals
    refine ⟨t, hrun, ?_, ?_, ?_, ?_⟩
    · have ho := hread (0, 0, PUnit.unit) (by decide)
      change t.readMem "Out" 0 = _ at ho
      rw [query_zero, keys_zero, distinct_values] at ho
      norm_num [attentionReal, attentionRealCausalBlock, scaledScore, initial,
        Fin.sum_univ_two, Finset.sum_ite_eq, Finset.sum_ite_eq'] at ho
      exact ho
    · have hf := hframe "Out" 1 (Or.inr (by
        intro i hi heq
        simp only [RealKernel.alActive, RealKernel.alRow, initial] at hi
        simp only [RealKernel.alOutOffset, RealKernel.alBase, RealKernel.alRow, initial] at heq
        have hd := i.2.1.isLt
        omega))
      simpa [initial] using hf
    · simpa [initial] using hframe "K" 0 (Or.inl (by decide))
    · simpa [initial] using hframe "V" 1 (Or.inl (by decide))

private def allocation : FlatAlloc where
  flat := "flat"
  regions := ["Q", "K", "V", "Out"]
  base := fun r => if r = "Q" then 0 else if r = "K" then 4 else if r = "V" then 8 else 12
  extent := fun r => if r = "Q" then 1 else if r ∈ (["K", "V", "Out"] : List RegionName) then 2 else 0

private theorem allocation_disjoint : allocation.Disjoint := by
  simp +decide [FlatAlloc.Disjoint, allocation, Region.mk.injEq]

private theorem allocation_covered :
    ∀ r, r ∉ allocation.regions → allocation.extent r = 0 := by
  intro r hr
  simp_all [allocation]

private theorem query_bounds (i : TileIndex [2, 1]) (hi : alActive initial 1 2 i) :
    alBase initial 1 2 2 + alRow initial 2 i.1 * 1 + i.2.1.val * 1 <
      allocation.extent "Q" := by
  simp only [alActive, alRow, initial] at hi
  simp +decide [alBase, alRow, initial, allocation]
  have hd := i.2.1.isLt
  omega

private theorem key_bounds (r : Nat) (hr : r < 2 * 1) (d : Fin 1) :
    alBase initial 1 2 2 + r * 1 + d.val * 1 < allocation.extent "K" := by
  simp +decide [alBase, initial, allocation, Region.mk.injEq]
  have hd := d.isLt
  omega

private theorem value_bounds (r : Nat) (hr : r < 2 * 1) (d : Fin 1) :
    alBase initial 1 2 2 + r * 1 + d.val * 1 < allocation.extent "V" := by
  simp +decide [alBase, initial, allocation, Region.mk.injEq]
  have hd := d.isLt
  omega

private theorem output_bounds (i : TileIndex [2, 1]) (hi : alActive initial 1 2 i) :
    alOutOffset initial 1 2 2 1 1 2 i < allocation.extent "Out" := by
  simp only [alActive, alRow, initial] at hi
  simp +decide [alOutOffset, alBase, alRow, initial, allocation, Region.mk.injEq]
  have hd := i.2.1.isLt
  omega

/-- The byte source executes in one allocation with space for only the active
Q row. Both branch outputs, the masked output and raw K/V bytes are preserved. -/
theorem flat_output_and_raw_frame (candidate causal : Bool) :
    ∃ t, exec (allocation.flattenKernel (kernel candidate causal).eraseDType.toAlgKernel)
        (allocation.flattenState initial) = some t ∧
      t.readMem "flat" 12 = (if causal then 1 else (3 / 2 : ℝ)) ∧
      t.mem "flat" 13 = .real 99 ∧ t.mem "flat" 4 = .of .int (-128) ∧
      t.mem "flat" 9 = .of .int 64 := by
  cases causal
  case' «false» =>
    obtain ⟨t, hrun, hread, hframe⟩ := FP8RealFlat.noncausal_correct
      "Q" "K" "V" "Out" 1
      2 2 1 1 2 2 1 1 2 2 1 1 2 2 1 1 1 1 2 0 2 1 1
      allocation allocation_disjoint rfl allocation_covered candidate initial 2
      (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) rfl output_injective (by intros; rfl)
      (valid_keys 2) (valid_values 2) query_bounds key_bounds value_bounds output_bounds
  case' «true» =>
    obtain ⟨t, hrun, hread, hframe⟩ := FP8RealFlat.causal_correct
      "Q" "K" "V" "Out" 1
      2 2 1 1 2 2 1 1 2 2 1 1 2 2 1 1 1 1 2 0 2 1 1
      allocation allocation_disjoint rfl allocation_covered candidate initial 1
      (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) rfl (by decide) (by decide)
      output_injective (by intros; rfl) (valid_keys 1) (valid_values 1) query_bounds
      (fun r hr => key_bounds r (by omega)) (fun r hr => value_bounds r (by omega)) output_bounds
  all_goals
    refine ⟨t, hrun, ?_, ?_, ?_, ?_⟩
    · have ho := hread (0, 0, PUnit.unit) (by decide)
      change t.readMem "flat" 12 = _ at ho
      rw [query_zero, keys_zero, distinct_values] at ho
      norm_num [attentionReal, attentionRealCausalBlock, scaledScore, initial,
        Fin.sum_univ_two, Finset.sum_ite_eq, Finset.sum_ite_eq'] at ho
      exact ho
    · have hf := hframe "flat" 13 (Or.inr (by
        intro i hi heq
        simp only [alActive, alRow, initial] at hi
        simp +decide [alOutOffset, alBase, alRow, initial, FlatAlloc.addr,
          allocation, Region.mk.injEq] at heq
        have hd := i.2.1.isLt
        omega))
      exact hf.trans (by rfl)
    · have hf := hframe "flat" 4 (Or.inr (by
        intro i hi heq
        simp +decide [alOutOffset, alBase, alRow, initial, FlatAlloc.addr,
          allocation, Region.mk.injEq] at heq
        omega))
      exact hf.trans (by rfl)
    · have hf := hframe "flat" 9 (Or.inr (by
        intro i hi heq
        simp +decide [alOutOffset, alBase, alRow, initial, FlatAlloc.addr,
          allocation, Region.mk.injEq] at heq
        omega))
      exact hf.trans (by rfl)

private def tailBounds : RegionBounds :=
  fun r => if r = "Q" then 1 else if r = "K" ∨ r = "V" then 3 else if r = "Out" then 2 else 0

/-- A partial key block needs no allocation for its masked fourth lane.
This checks source access safety, without claiming conventional attention for
the source's non-divisible key-tail behavior. -/
theorem partial_tail_source_safe (candidate causal : Bool) :
    (FP8RealSource.source "Q" "K" "V" "Out" 1
      2 2 1 1 2 2 1 1 2 2 1 1 2 2 1 1 1 1 3 0 2 2 1 causal candidate).eraseDType.toAlgKernel.TraceSafe
      tailBounds initial := by
  apply FP8RealSafety.source_safe "Q" "K" "V" "Out" 1
    2 2 1 1 2 2 1 1 2 2 1 1 2 2 1 1 1 1 3 0 2 2 1 causal candidate
    tailBounds initial (by intros; rfl)
  · intro i hi
    have hd := i.2.1.isLt
    simp only [alActive, alRow, initial] at hi
    simp +decide [tailBounds, alBase, alRow, initial]
    omega
  · intro k hk j d hj
    have hd : d.val = 0 := by omega
    simpa +decide [tailBounds, alBase, initial, hd] using hj
  · intro k hk j d hj
    have hd : d.val = 0 := by omega
    simpa +decide [tailBounds, alBase, initial, hd] using hj
  · intro i hi
    have hd := i.2.1.isLt
    simp only [alActive, alRow, initial] at hi
    simp +decide [tailBounds, alOutOffset, alBase, alRow, initial]
    omega

#guard_msgs (drop info) in
#auditModuleAxioms
end RealAttentionLlamaFP8Tests
