import bench.tritonbench_g.var_len_copy.Proofs.FP
import VeriTile.Triton.Float.ProfiledSoundness
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.VarLenCopy.NumericExecution
open VeriTile Triton TypedSource FP.Structural FP.ProfiledRewrite
noncomputable section
open scoped Classical
set_option maxHeartbeats 300000
set_option linter.unusedSimpArgs false
set_option maxRecDepth 16000

/-- Nonempty exact-real backend for copy/conversion execution. This witness
checks typed cells and actual loops, not target rounding behavior. -/
def model : Algebra (WithBot ℝ) where
  literal := fun _ _ r ↦ some r
  negInf := none
  binary := fun _ _ op a b ↦ match op with
    | .add => WithBot.realAdd a b
    | .sub => WithBot.realSub a b
    | .mul => WithBot.realMul a b
    | .div => WithBot.realDiv a b
    | .max => max a b
    | .pow => some 0
  unary := fun _ _ x ↦ x
  cast := fun _ _ _ x ↦ x
  fromNat := fun _ n ↦ some n
  fromInt := fun _ n ↦ some n
  fp32Bits := fun _ ↦ some 0
  fp32Load := id
  reduceMax := fun _ {_} ax keep v ↦
    ((Tile.reduceMax ax keep ⟨v⟩).getD ⟨fun _ ↦ none⟩).data
  reduceSum := fun _ {_} ax keep v ↦ (Tile.reduceSum ax keep ⟨v⟩).data

def initial (length : Int) : State (WithBot ℝ) where
  mem := fun r o ↦
    if r = "length" then .mk .int length
    else if r = "old" then .mk .int 1
    else if r = "new" then .mk .int 10
    else if r = "src" then .mk .real (some (10 + o))
    else .mk .bf16 (ofFloat .bf16 (some 7))
  regs := fun _ _ _ ↦ none
  pids := fun ax ↦ if ax = 0 then 1 else 0
  numPids := fun _ ↦ 2
  undef := fun d _ _ ↦ defaultValue d


-- Explicit small states prevent full loop reduction from constructing an enormous
-- anonymous final-state term. Every transition still uses the actual source body.
def prefixState (length : Int) : State (WithBot ℝ) :=
  let s := (initial length).setReg "a_id" .nat [] (fun _ ↦ 1)
  let s := s.setReg "length" .int [] (fun _ ↦ length)
  let s := s.setReg "old_start" .int [] (fun _ ↦ 1)
  let s := s.setReg "new_start" .int [] (fun _ ↦ 10)
  let s := s.setReg "old_offset" .nat [2] (fun i ↦ i.1.val)
  s.setReg "new_offset" .nat [2] (fun i ↦ i.1.val)

def values (start : Nat) : Values (WithBot ℝ) .real [2] :=
  fun i ↦ some ((start : ℝ) + i.1.val)

def firstState : State (WithBot ℝ) :=
  let s := (prefixState 3).setReg "i" .int [] (fun _ ↦ 0)
  let s := s.setReg "v" .real [2] (values 11)
  let s := s.write "dst" 10 (.mk .bf16 (some 11))
  s.write "dst" 11 (.mk .bf16 (some 12))

def finalState : State (WithBot ℝ) :=
  let s := firstState.setReg "i" .int [] (fun _ ↦ 2)
  let s := s.setReg "v" .real [2] (values 13)
  let s := s.write "dst" 12 (.mk .bf16 (some 13))
  s.write "dst" 13 (.mk .bf16 (some 14))

theorem prefix_run (length : Int) :
    run (engine model) (FPProof.prefixBody ⟨"old"⟩ ⟨"length"⟩ ⟨"new"⟩ "src" "dst" 2)
      (initial length) = some (prefixState length) := by
  simp [FPProof.prefixBody, kernel, ComputeKernel.surfaceBody, run, step,
    FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, evalDeltas,
    Option.bind, engine, Algebra.withDefaultPrecision, resolvePrecision, model,
    ComputeDType.eraseDType, numeric, bop, natLt, initial, Cell.read, ofFloat, toFloat,
    TileShape.axisDim, Region.cast, State.setReg, prefixState]

abbrev copyBody :=
  FPProof.body .fp64 .bf16 Bool.false ⟨"old"⟩ ⟨"length"⟩ ⟨"new"⟩ "src" "dst" 2

theorem first_body :
    run (engine model) copyBody ((prefixState 3).setReg "i" .int [] (fun _ ↦ 0)) =
      some firstState := by
  simp [copyBody, FPProof.body, kernel, ComputeKernel.surfaceBody, run, step,
    FP.Structural.store, State.write, TileShape.allIndices, List.finRange_succ,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, numericLt,
    FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, evalDeltas,
    Option.bind, engine, Algebra.withDefaultPrecision, resolvePrecision, model,
    ComputeDType.eraseDType, numeric, bop, natLt, initial, Cell.read, ofFloat, toFloat,
    TileShape.axisDim, Region.cast, State.setReg, prefixState, firstState, values]
  -- If the value-register functions differ only by real coercion arithmetic:
  all_goals norm_num
  funext d sh n
  by_cases hn : n = "v"
  · subst n
    by_cases hd : d = .real
    · subst d
      by_cases hs : sh = [2]
      · subst sh
        simp only [dite_true, ite_true]
        congr 1
        funext ⟨i,u⟩
        fin_cases i <;> norm_num [values, Int.toNat]
      · simp [hs, Ne.symm hs]
    · simp [hd, Ne.symm hd]
  · simp [hn]

theorem second_body :
    run (engine model) copyBody (firstState.setReg "i" .int [] (fun _ ↦ 2)) =
      some finalState := by
  simp [copyBody, FPProof.body, kernel, ComputeKernel.surfaceBody, run, step,
    FP.Structural.store, State.write, TileShape.allIndices, List.finRange_succ,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, numericLt,
    FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, evalDeltas,
    Option.bind, engine, Algebra.withDefaultPrecision, resolvePrecision, model,
    ComputeDType.eraseDType, numeric, bop, natLt, initial, Cell.read, ofFloat, toFloat,
    TileShape.axisDim, Region.cast, State.setReg, prefixState, firstState, finalState, values]
  all_goals norm_num
  funext d sh n
  by_cases hn : n = "v"
  · subst n
    by_cases hd : d = .real
    · subst d
      by_cases hs : sh = [2]
      · subst sh
        simp only [dite_true, ite_true]
        congr 1
        funext ⟨i,u⟩
        fin_cases i <;> norm_num [values, Int.toNat]
      · simp [hs, Ne.symm hs]
    · simp [hd, Ne.symm hd]
  · simp [hn]

theorem two_iterations :
    rangeInt (engine model) "i" 0 3 2 copyBody (prefixState 3) = some finalState := by
  rw [rangeInt]
  rw [first_body]
  change rangeInt (engine model) "i" 2 3 2 copyBody firstState = some finalState
  rw [rangeInt]
  rw [second_body]
  change rangeInt (engine model) "i" 4 3 2 copyBody finalState = some finalState
  rw [rangeInt]
  norm_num

theorem complete_original :
    FP.Structural.exec (engine model)
      (kernel .fp64 .bf16 Bool.false ⟨"old"⟩ ⟨"length"⟩ ⟨"new"⟩ "src" "dst" 2)
      (initial 3) = some finalState := by
  rw [FPProof.source_decomposition]
  simp only [FP.Structural.exec, run_append, prefix_run, Option.bind_some]
  simp only [run, step, evalExpr, evalOp_unfold]
  change (rangeInt (engine model) "i" 0 3 2 copyBody (prefixState 3) >>=
    fun s ↦ some s) = some finalState
  rw [two_iterations]
  rfl

theorem complete_candidate :
    FP.Structural.exec (engine model)
      (kernel .fp64 .bf16 Bool.true ⟨"old"⟩ ⟨"length"⟩ ⟨"new"⟩ "src" "dst" 2)
      (initial 3) = some finalState := by
  rw [← FPProof.source_equivalence]
  exact complete_original

theorem final_cells :
    finalState.mem "dst" 10 = Cell.mk .bf16 (some 11) ∧
    finalState.mem "dst" 11 = Cell.mk .bf16 (some 12) ∧
    finalState.mem "dst" 12 = Cell.mk .bf16 (some 13) ∧
    finalState.mem "dst" 13 = Cell.mk .bf16 (some 14) ∧
    finalState.mem "dst" 14 = Cell.mk .bf16 (some 7) := by
  simp [finalState, firstState, prefixState, initial, State.write, ofFloat]
  exact ⟨rfl,rfl,rfl,rfl,rfl⟩


/-- Length three and block two execute two complete masked body iterations. -/
def overcopy_run (v : Bool) :
    {t : State (WithBot ℝ) // run (engine model)
      (kernel .fp64 .bf16 v ⟨"old"⟩ ⟨"length"⟩ ⟨"new"⟩ "src" "dst" 2).surfaceBody
        (initial 3) = some t} := by
  refine ⟨finalState, ?_⟩
  cases v
  · exact complete_original
  · exact complete_candidate

/-- The source copies one value beyond the requested length and preserves the
following cell. This checks the actual final store, not only the count formula. -/
theorem overcopy_cells (v : Bool) :
    (overcopy_run v).val.mem "dst" 10 = Cell.mk .bf16 (some 11) ∧
    (overcopy_run v).val.mem "dst" 11 = Cell.mk .bf16 (some 12) ∧
    (overcopy_run v).val.mem "dst" 12 = Cell.mk .bf16 (some 13) ∧
    (overcopy_run v).val.mem "dst" 13 = Cell.mk .bf16 (some 14) ∧
    (overcopy_run v).val.mem "dst" 14 = Cell.mk .bf16 (some 7) := final_cells

theorem complete_nonpositive (length : Int) (hle : length ≤ 0) (v : Bool) :
    FP.Structural.exec (engine model)
      (kernel .fp64 .bf16 v ⟨"old"⟩ ⟨"length"⟩ ⟨"new"⟩ "src" "dst" 2)
      (initial length) = some (prefixState length) := by
  rw [FPProof.source_decomposition]
  simp only [FP.Structural.exec, run_append, prefix_run, Option.bind_some]
  simp only [run, step, evalExpr, evalOp_unfold]
  change (rangeInt (engine model) "i" 0 length 2
    (FPProof.body .fp64 .bf16 v ⟨"old"⟩ ⟨"length"⟩ ⟨"new"⟩ "src" "dst" 2)
      (prefixState length) >>= fun s ↦ some s) = some (prefixState length)
  rw [rangeInt]
  simp [not_lt.mpr hle]

/-- Every nonpositive signed length leaves all memory cells unchanged. -/
def noCopy_run (length : Int) (hle : length ≤ 0) (v : Bool) :
    {t : State (WithBot ℝ) // run (engine model)
      (kernel .fp64 .bf16 v ⟨"old"⟩ ⟨"length"⟩ ⟨"new"⟩ "src" "dst" 2).surfaceBody
        (initial length) = some t} := by
  refine ⟨prefixState length, ?_⟩
  have h := complete_nonpositive length hle v
  cases v <;> exact h

theorem noCopy_cells (length : Int) (hle : length ≤ 0) (v : Bool) :
    (noCopy_run length hle v).val.mem = (initial length).mem := rfl

end
end VeriTile.Bench.TritonBenchG.VarLenCopy.NumericExecution

#guard_msgs (drop info) in
#auditModuleAxioms
