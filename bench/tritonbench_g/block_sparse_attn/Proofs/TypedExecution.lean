import bench.tritonbench_g.block_sparse_attn.Proofs.ExecutionLoop
import VeriTile.Triton.Float.ProfiledSoundness

/-! Complete nonempty executions in the same consistent abstract model.
The staged checks cover all mask/channel branches, nonzero batch/head/layout,
two signed CSR visits, independent metadata strides and masked stores.
These are execution/domain witnesses, not GPU results or real correctness. -/
namespace VeriTile.Bench.TritonBenchG.BlockSparseAttn.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open AttentionFwdTriton3.FPModel (model domain model_satisfies_rules)
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

theorem source_parts (em en two : Bool) :
    code em en two Bool.false =
      (slices em en two Bool.false).before ++
        .forRangeInt "col_idx_idx" (.ref .int [] "start_l") (.ref .int [] "end_l")
          (.constNat 1) (slices em en two Bool.false).body ::
          (slices em en two Bool.false).after := by
  cases em <;> cases en <;> cases two <;> rfl

theorem before_eq (em en two : Bool) :
    (slices em en two Bool.false).before = (slices em Bool.false two Bool.false).before := by
  cases em <;> cases en <;> cases two <;> rfl

theorem body_parts (em en two : Bool) :
    (slices em en two Bool.false).body = score em en two ++ rest em en two := by
  cases em <;> cases en <;> cases two <;> rfl

theorem prefix_bounds (em two : Bool) :
    (preloop em two).val.regs .int [] "start_l" = some (fun _ => 1) ∧
      (preloop em two).val.regs .int [] "end_l" = some (fun _ => 3) := by
  have names : "start_l" ≠ "end_l" := by decide
  cases em <;> cases two
  all_goals norm_num [preloop, State.setReg, initial, Cell.read, names]

theorem loop_executes (em en two : Bool) :
    step (engine model)
      (.forRangeInt "col_idx_idx" (.ref .int [] "start_l") (.ref .int [] "end_l")
        (.constNat 1) (slices em en two Bool.false).body)
      (preloop em two).val = some (secondBody em en two).val := by
  have first : run (engine model) (slices em en two Bool.false).body
      ((preloop em two).val.setReg "col_idx_idx" .int [] (fun _ => 1)) =
        some (firstBody em en two).val := by
    simp only [body_parts, run_append, (firstScore em en two).property,
      Option.bind_some, (firstBody em en two).property]
  have second : run (engine model) (slices em en two Bool.false).body
      ((firstBody em en two).val.setReg "col_idx_idx" .int [] (fun _ => 2)) =
        some (secondBody em en two).val := by
    simp only [body_parts, run_append, (secondScore em en two).property,
      Option.bind_some, (secondBody em en two).property]
  simp only [step, evalOp_unfold, (prefix_bounds em two).1,
    (prefix_bounds em two).2, bind, Option.bind_some]
  rw [rangeInt]
  simp only [show (1 : Nat) ≠ 0 from by decide, show (1 : Int) < 3 from by decide,
    ↓reduceIte, first, bind, Option.bind_some]
  change rangeInt (engine model) "col_idx_idx" 2 3 1
    (slices em en two Bool.false).body (firstBody em en two).val = _
  rw [rangeInt]
  simp only [show (1 : Nat) ≠ 0 from by decide, show (2 : Int) < 3 from by decide,
    ↓reduceIte, second, bind, Option.bind_some]
  change rangeInt (engine model) "col_idx_idx" 3 3 1
    (slices em en two Bool.false).body (secondBody em en two).val = _
  rw [rangeInt]
  simp only [show (1 : Nat) ≠ 0 from by decide, show ¬ (3 : Int) < 3 from by decide,
    ↓reduceIte]

theorem original_execution (em en two : Bool) :
    FP.Structural.exec (engine model) (original em en two).kernel initial =
      some (finish em en two).val := by
  change run (engine model) (code em en two Bool.false) initial = _
  simp only [source_parts, run_append, before_eq, (preloop em two).property,
    Option.bind_some]
  rw [run]
  simp only [loop_executes, bind, Option.bind_some, (finish em en two).property]

theorem domain_inhabited (em en two : Bool) :
    ∀ site ∈ (original em en two).domain, site.Holds model domain initial := by
  simp [original, FPEquivProofs.original, FPEquivProofs.requirementsFor,
    TypedSource.original, TypedSource.specialize, ComputeKernel.surfaceBody,
    Source.intRange?, Source.prefixBefore, Site.Holds, domain]

theorem candidate_execution (R : FP.ScalarArithmetic.Rules) (em en two : Bool) :
    FP.Structural.exec (engine model) (candidate em en two).kernel initial =
      some (finish em en two).val := by
  have h := FP.ProfiledRewrite.sound
    (FPEquivProofs.scaling_equiv .bf16 "Out" "Q" "K" "V" ⟨"Rows"⟩ ⟨"Cols"⟩ 5 13 2 3
      100 20 4 80 16 4 80 16 4 200 40 6 4 2 3 2 2 2 (if two then 2 else 1) em en R)
  exact (h ℝ model domain (model_satisfies_rules R) initial
    (domain_inhabited em en two)).symm.trans (original_execution em en two)

theorem complete_execution (R : FP.ScalarArithmetic.Rules) (em en two swapped : Bool) :
    FP.Structural.exec (engine model) (program em en two swapped).kernel initial =
      some (finish em en two).val := by
  cases swapped
  · exact original_execution em en two
  · exact candidate_execution R em en two

theorem source_executes (R : FP.ScalarArithmetic.Rules) (em en two swapped : Bool) :
    ∃ t, FP.Structural.exec (engine model) (program em en two swapped).kernel initial = some t :=
  ⟨_, complete_execution R em en two swapped⟩

/-- The loop visited both CSR entries; it was not empty or clipped to zero. -/
theorem two_csr_visits (R : FP.ScalarArithmetic.Rules) (em en two swapped : Bool) :
    (FP.Structural.exec (engine model) (program em en two swapped).kernel initial).map
      (fun t => (t.regs .int [] "col_idx_idx", t.regs .int [] "col_idx",
        t.regs .real [2] "l_i")) =
      some (some (fun _ => 2), some (fun _ => 1), some (fun _ => 2)) := by
  rw [complete_execution R]
  -- Check only the selected projections against the executed state.
  -- The change step is definitional; the remaining equality is real arithmetic
  -- in this particular consistency model.
  cases em <;> cases en <;> cases two
  all_goals
    change some (some (fun _ : TileIndex [] => (2 : Int)),
      some (fun _ : TileIndex [] => (1 : Int)),
      some (fun _ : TileIndex [2] => (1 : ℝ) * (1 * 0 + 1 * 1) + 1 * 1)) = _
    norm_num

/-- Both channel branches store at the actual nonzero output offset and
preserve the masked second row. -/
theorem output_and_mask (R : FP.ScalarArithmetic.Rules) (em en two swapped : Bool) :
    (FP.Structural.exec (engine model) (program em en two swapped).kernel initial).map
      (fun t => ((t.mem "Out" 252).read .bf16, (t.mem "Out" 253).read .bf16,
        (t.mem "Out" 254).read .bf16, (t.mem "Out" 255).read .bf16,
        (t.mem "Out" 258).read .bf16, (t.mem "Out" 260).read .bf16)) =
      some (if two then (0,0,0,0,9,9) else (0,0,9,9,9,9)) := by
  rw [complete_execution R]
  cases em <;> cases en <;> cases two
  all_goals
    simp only [Option.map_some, Bool.false_eq_true, ↓reduceIte,
      Option.some.injEq, Prod.mk.injEq]
    repeat' apply And.intro
    all_goals
      -- Reduce the selected cell without expanding unrelated registers.
      conv_lhs =>
        arg 2
        whnf
      simp [Cell.read, bop, Function.comp_def, numeric, ofFloat, toFloat,
        Broadcast.leftIndex, Broadcast.rightIndex, TileShape.dropInsertedIndex]

theorem consistent_execution (R : FP.ScalarArithmetic.Rules) (em en two : Bool) :
    FP.Guarded.Models R.assumptions model domain ∧
    (∃ t, FP.Structural.exec (engine model) (original em en two).kernel initial = some t) ∧
    (∃ t, FP.Structural.exec (engine model) (candidate em en two).kernel initial = some t) ∧
    (∀ site ∈ (original em en two).domain, site.Holds model domain initial) := by
  exact ⟨model_satisfies_rules R, source_executes R em en two Bool.false,
    source_executes R em en two Bool.true, domain_inhabited em en two⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BlockSparseAttn.TypedExecution
