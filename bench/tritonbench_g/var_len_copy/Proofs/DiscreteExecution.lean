import bench.tritonbench_g.var_len_copy.Proofs.DiscreteFP
import bench.tritonbench_g.var_len_copy.Proofs.DiscreteRealTyped
import bench.tritonbench_g.var_len_copy.Proofs.Execution

namespace VeriTile.Bench.TritonBenchG.VarLenCopy.DiscreteExecution
open VeriTile Triton DiscreteSource FP.Structural FP.ProfiledRewrite
noncomputable section
open scoped Classical
set_option maxHeartbeats 300000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

abbrev model := NumericExecution.model

def payload : (k : Kind) → Nat → Value (WithBot ℝ) k.dtype
  | .signed, n => -10 - (n : Int)
  | .unsigned, n => 10 + n
  | .boolean, n => decide (n % 2 = 0)

def initial (k : Kind) : State (WithBot ℝ) where
  mem := fun r o ↦
    if r = "length" then .mk .int 3
    else if r = "old" then .mk .int 1
    else if r = "new" then .mk .int 10
    else if r = "src" then .mk k.dtype (payload k o)
    else .mk k.dtype (defaultValue k.dtype)
  regs := fun _ _ _ ↦ none
  pids := fun ax ↦ if ax = 0 then 1 else 0
  numPids := fun _ ↦ 2
  undef := fun d _ _ ↦ defaultValue d

def prefixState (k : Kind) : State (WithBot ℝ) :=
  let s := (initial k).setReg "a_id" .nat [] (fun _ ↦ 1)
  let s := s.setReg "length" .int [] (fun _ ↦ 3)
  let s := s.setReg "old_start" .int [] (fun _ ↦ 1)
  let s := s.setReg "new_start" .int [] (fun _ ↦ 10)
  let s := s.setReg "old_offset" .nat [2] (fun i ↦ i.1.val)
  s.setReg "new_offset" .nat [2] (fun i ↦ i.1.val)

def values (k : Kind) (start : Nat) : Values (WithBot ℝ) k.dtype [2] :=
  fun i ↦ payload k (start + i.1.val)

def firstState (k : Kind) : State (WithBot ℝ) :=
  let s := (prefixState k).setReg "i" .int [] (fun _ ↦ 0)
  let s := s.setReg "v" k.dtype [2] (values k 1)
  let s := s.write "dst" 10 (.mk k.dtype (payload k 1))
  s.write "dst" 11 (.mk k.dtype (payload k 2))

def finalState (k : Kind) : State (WithBot ℝ) :=
  let s := (firstState k).setReg "i" .int [] (fun _ ↦ 2)
  let s := s.setReg "v" k.dtype [2] (values k 3)
  let s := s.write "dst" 12 (.mk k.dtype (payload k 3))
  s.write "dst" 13 (.mk k.dtype (payload k 4))

theorem prefix_run (k : Kind) :
    run (engine model) (FPProof.prefixBody ⟨"old"⟩ ⟨"length"⟩ ⟨"new"⟩ "src" "dst" 2)
      (initial k) = some (prefixState k) := by
  simp [FPProof.prefixBody, TypedSource.kernel, ComputeKernel.surfaceBody, run, step,
    FP.Structural.evalExpr, evalOp_unfold, Option.bind, engine,
    ComputeDType.eraseDType, numeric, bop, initial, Cell.read,
    Region.cast, State.setReg, prefixState]

abbrev copyBody (k : Kind) :=
  DiscreteFP.body (DiscreteReal.representative k) Bool.false
    ⟨"old"⟩ ⟨"length"⟩ ⟨"new"⟩ "src" "dst" 2

theorem first_signed :
    run (engine model) (copyBody .signed)
      ((prefixState .signed).setReg "i" .int [] (fun _ ↦ 0)) =
      some (firstState .signed) := by
  simp [copyBody, DiscreteFP.body, kernel, DiscreteReal.representative, Kind.dtype, payload, ComputeKernel.surfaceBody, run, step,
    FP.Structural.store, State.write, TileShape.allIndices, List.finRange_succ,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, numericLt,
    FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, evalDeltas,
    Option.bind, engine, Algebra.withDefaultPrecision, resolvePrecision, model, NumericExecution.model,
    ComputeDType.eraseDType, numeric, bop, natLt, initial, Cell.read, ofFloat, toFloat,
    TileShape.axisDim, Region.cast, State.setReg, prefixState, firstState, finalState, values]
  all_goals norm_num
  funext d sh n
  by_cases hn : n = "v"
  · subst n
    by_cases hd : d = .int
    · subst d
      by_cases hs : sh = [2]
      · subst sh
        simp only [dite_true, ite_true]
        congr 1
        funext ⟨i,u⟩
        fin_cases i <;> norm_num [values, payload, Int.toNat]
      · simp [hs, Ne.symm hs]
    · simp [hd, Ne.symm hd]
  · simp [hn]

theorem second_signed :
    run (engine model) (copyBody .signed)
      ((firstState .signed).setReg "i" .int [] (fun _ ↦ 2)) =
      some (finalState .signed) := by
  simp [copyBody, DiscreteFP.body, kernel, DiscreteReal.representative, Kind.dtype, payload, ComputeKernel.surfaceBody, run, step,
    FP.Structural.store, State.write, TileShape.allIndices, List.finRange_succ,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, numericLt,
    FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, evalDeltas,
    Option.bind, engine, Algebra.withDefaultPrecision, resolvePrecision, model, NumericExecution.model,
    ComputeDType.eraseDType, numeric, bop, natLt, initial, Cell.read, ofFloat, toFloat,
    TileShape.axisDim, Region.cast, State.setReg, prefixState, firstState, finalState, values]
  all_goals norm_num
  funext d sh n
  by_cases hn : n = "v"
  · subst n
    by_cases hd : d = .int
    · subst d
      by_cases hs : sh = [2]
      · subst sh
        simp only [dite_true, ite_true]
        congr 1
        funext ⟨i,u⟩
        fin_cases i <;> norm_num [values, payload, Int.toNat]
      · simp [hs, Ne.symm hs]
    · simp [hd, Ne.symm hd]
  · simp [hn]

theorem first_unsigned :
    run (engine model) (copyBody .unsigned)
      ((prefixState .unsigned).setReg "i" .int [] (fun _ ↦ 0)) =
      some (firstState .unsigned) := by
  simp [copyBody, DiscreteFP.body, kernel, DiscreteReal.representative, Kind.dtype, payload, ComputeKernel.surfaceBody, run, step,
    FP.Structural.store, State.write, TileShape.allIndices, List.finRange_succ,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, numericLt,
    FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, evalDeltas,
    Option.bind, engine, Algebra.withDefaultPrecision, resolvePrecision, model, NumericExecution.model,
    ComputeDType.eraseDType, numeric, bop, natLt, initial, Cell.read, ofFloat, toFloat,
    TileShape.axisDim, Region.cast, State.setReg, prefixState, firstState, finalState, values]
  all_goals norm_num
  funext d sh n
  by_cases hn : n = "v"
  · subst n
    by_cases hd : d = .nat
    · subst d
      by_cases hs : sh = [2]
      · subst sh
        simp only [dite_true, ite_true]
        congr 1
        funext ⟨i,u⟩
        fin_cases i <;> norm_num [values, payload, Int.toNat]
      · simp [hs, Ne.symm hs]
    · simp [hd, Ne.symm hd]
  · simp [hn]

theorem second_unsigned :
    run (engine model) (copyBody .unsigned)
      ((firstState .unsigned).setReg "i" .int [] (fun _ ↦ 2)) =
      some (finalState .unsigned) := by
  simp [copyBody, DiscreteFP.body, kernel, DiscreteReal.representative, Kind.dtype, payload, ComputeKernel.surfaceBody, run, step,
    FP.Structural.store, State.write, TileShape.allIndices, List.finRange_succ,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, numericLt,
    FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, evalDeltas,
    Option.bind, engine, Algebra.withDefaultPrecision, resolvePrecision, model, NumericExecution.model,
    ComputeDType.eraseDType, numeric, bop, natLt, initial, Cell.read, ofFloat, toFloat,
    TileShape.axisDim, Region.cast, State.setReg, prefixState, firstState, finalState, values]
  all_goals norm_num
  funext d sh n
  by_cases hn : n = "v"
  · subst n
    by_cases hd : d = .nat
    · subst d
      by_cases hs : sh = [2]
      · subst sh
        simp only [dite_true, ite_true]
        congr 1
        funext ⟨i,u⟩
        fin_cases i <;> norm_num [values, payload, Int.toNat]
      · simp [hs, Ne.symm hs]
    · simp [hd, Ne.symm hd]
  · simp [hn]

theorem first_boolean :
    run (engine model) (copyBody .boolean)
      ((prefixState .boolean).setReg "i" .int [] (fun _ ↦ 0)) =
      some (firstState .boolean) := by
  simp [copyBody, DiscreteFP.body, kernel, DiscreteReal.representative, Kind.dtype, payload, ComputeKernel.surfaceBody, run, step,
    FP.Structural.store, State.write, TileShape.allIndices, List.finRange_succ,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, numericLt,
    FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, evalDeltas,
    Option.bind, engine, Algebra.withDefaultPrecision, resolvePrecision, model, NumericExecution.model,
    ComputeDType.eraseDType, numeric, bop, natLt, initial, Cell.read, ofFloat, toFloat,
    TileShape.axisDim, Region.cast, State.setReg, prefixState, firstState, finalState, values]
  all_goals norm_num
  funext d sh n
  by_cases hn : n = "v"
  · subst n
    by_cases hd : d = .bool
    · subst d
      by_cases hs : sh = [2]
      · subst sh
        simp only [dite_true, ite_true]
        congr 1
        funext ⟨i,u⟩
        fin_cases i <;> norm_num [values, payload, Int.toNat]
      · simp [hs, Ne.symm hs]
    · simp [hd, Ne.symm hd]
  · simp [hn]

theorem second_boolean :
    run (engine model) (copyBody .boolean)
      ((firstState .boolean).setReg "i" .int [] (fun _ ↦ 2)) =
      some (finalState .boolean) := by
  simp [copyBody, DiscreteFP.body, kernel, DiscreteReal.representative, Kind.dtype, payload, ComputeKernel.surfaceBody, run, step,
    FP.Structural.store, State.write, TileShape.allIndices, List.finRange_succ,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, numericLt,
    FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, evalDeltas,
    Option.bind, engine, Algebra.withDefaultPrecision, resolvePrecision, model, NumericExecution.model,
    ComputeDType.eraseDType, numeric, bop, natLt, initial, Cell.read, ofFloat, toFloat,
    TileShape.axisDim, Region.cast, State.setReg, prefixState, firstState, finalState, values]
  all_goals norm_num
  funext d sh n
  by_cases hn : n = "v"
  · subst n
    by_cases hd : d = .bool
    · subst d
      by_cases hs : sh = [2]
      · subst sh
        simp only [dite_true, ite_true]
        congr 1
        funext ⟨i,u⟩
        fin_cases i <;> norm_num [values, payload, Int.toNat]
      · simp [hs, Ne.symm hs]
    · simp [hd, Ne.symm hd]
  · simp [hn]

theorem first_body (k : Kind) :
    run (engine model) (copyBody k)
      ((prefixState k).setReg "i" .int [] (fun _ ↦ 0)) = some (firstState k) := by
  cases k
  · exact first_signed
  · exact first_unsigned
  · exact first_boolean

theorem second_body (k : Kind) :
    run (engine model) (copyBody k)
      ((firstState k).setReg "i" .int [] (fun _ ↦ 2)) = some (finalState k) := by
  cases k
  · exact second_signed
  · exact second_unsigned
  · exact second_boolean

theorem two_iterations (k : Kind) :
    rangeInt (engine model) "i" 0 3 2 (copyBody k) (prefixState k) = some (finalState k) := by
  rw [rangeInt, first_body]
  change rangeInt (engine model) "i" 2 3 2 (copyBody k) (firstState k) = some (finalState k)
  rw [rangeInt, second_body]
  change rangeInt (engine model) "i" 4 3 2 (copyBody k) (finalState k) = some (finalState k)
  rw [rangeInt]
  norm_num

theorem complete_original (k : Kind) :
    FP.Structural.exec (engine model)
      (kernel (DiscreteReal.representative k) Bool.false
        ⟨"old"⟩ ⟨"length"⟩ ⟨"new"⟩ "src" "dst" 2)
      (initial k) = some (finalState k) := by
  rw [DiscreteFP.source_decomposition]
  simp only [FP.Structural.exec, run_append, prefix_run, Option.bind_some]
  simp only [run, step, evalExpr, evalOp_unfold]
  change (rangeInt (engine model) "i" 0 3 2 (copyBody k) (prefixState k) >>=
    fun s ↦ some s) = some (finalState k)
  rw [two_iterations]
  rfl

theorem complete_profile (storage : Storage) (v : Bool) :
    FP.Structural.exec (engine model)
      (kernel storage v ⟨"old"⟩ ⟨"length"⟩ ⟨"new"⟩ "src" "dst" 2)
      (initial storage.kind) = some (finalState storage.kind) := by
  have h := complete_original storage.kind
  cases v
  · cases storage <;> exact h
  · rw [← DiscreteFP.source_equivalence]
    cases storage <;> exact h

/-- The fourth copied cell lies beyond the requested length; the fifth is untouched. -/
theorem final_cells (k : Kind) :
    (finalState k).mem "dst" 10 = Cell.mk k.dtype (payload k 1) ∧
    (finalState k).mem "dst" 11 = Cell.mk k.dtype (payload k 2) ∧
    (finalState k).mem "dst" 12 = Cell.mk k.dtype (payload k 3) ∧
    (finalState k).mem "dst" 13 = Cell.mk k.dtype (payload k 4) ∧
    (finalState k).mem "dst" 14 = Cell.mk k.dtype (defaultValue k.dtype) := by
  simp [finalState, firstState, prefixState, initial, State.write]

end
end VeriTile.Bench.TritonBenchG.VarLenCopy.DiscreteExecution

#guard_msgs (drop info) in
#auditModuleAxioms
