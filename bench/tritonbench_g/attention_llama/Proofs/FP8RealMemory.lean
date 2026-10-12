import bench.tritonbench_g.attention_llama.Proofs.RealKernel
import VeriTile.Triton.Memory.Locality

/-! A decoded view of the raw K/V input memory and its masked-load bridge.
Only active byte lanes need a finite E5M2 interpretation. Other memory cells
and all non-memory state are retained. This is an input representation view,
not a replacement source program or an FP equivalence assumption. -/
namespace VeriTile.Bench.TritonBenchG.AttentionLlama.FP8RealMemory
open VeriTile.Triton
set_option maxRecDepth 4000
set_option maxHeartbeats 500000

def FiniteByte (byte : Int) : Prop :=
  Float8E5Bits.isInt8 byte = Bool.true ∧ (Float8E5Bits.ofInt byte).decodeRat.isSome = Bool.true

/-- Exact decoded input when finite. The default is used only to make the
view total; every active load below requires a valid finite byte. -/
noncomputable def value (s : BlockState) (r : RegionName) (o : Nat) : ℝ :=
  (((Float8E5Bits.ofInt (s.readMemValue .int r o)).decodeRat.getD 0 : ℚ) : ℝ)

noncomputable def decoded (K V : RegionName) (s : BlockState) : BlockState :=
  { s with mem := fun r o =>
      if r = K ∨ r = V then MemCell.real (value s r o) else s.mem r o }

theorem outside_memory (K V : RegionName) (s : BlockState) (r : RegionName) (o : Nat)
    (hK : r ≠ K) (hV : r ≠ V) :
    (decoded K V s).mem r o = s.mem r o := by
  simp [decoded, hK, hV]

theorem read_decoded (K V : RegionName) (s : BlockState) (r : RegionName) (o : Nat)
    (h : r = K ∨ r = V) :
    (decoded K V s).readMem r o = value s r o := by
  have hm : (decoded K V s).mem r o = MemCell.real (value s r o) := by
    simp [decoded, h]
  unfold BlockState.readMem
  rw [hm]
  rfl

theorem decoded_setReg (K V : RegionName) (s : BlockState) (name : RegName)
    (tile : Tile d sh) :
    decoded K V (s.setReg name d sh tile) = (decoded K V s).setReg name d sh tile := rfl

/-- Q-only or empty access windows cannot observe the K/V representation
change. The register, launch and undefined-value channels are unchanged. -/
theorem agree_outside (K V : RegionName) (s : BlockState) (bounds : RegionBounds)
    (hK : bounds K = 0) (hV : bounds V = 0) :
    s.AgreeOn bounds.Window (decoded K V s) := by
  refine ⟨?_, rfl, rfl, rfl, rfl⟩
  intro r o ho
  have nrK : r ≠ K := by
    intro h
    subst r
    simp [RegionBounds.Window, hK] at ho
  have nrV : r ≠ V := by
    intro h
    subst r
    simp [RegionBounds.Window, hV] at ho
  exact (outside_memory K V s r o nrK nrV).symm

theorem eval_outside (K V : RegionName) (s : BlockState) (bounds : RegionBounds)
    (hK : bounds K = 0) (hV : bounds V = 0) (e : Op d sh)
    (hsafe : e.SafeAt bounds s) :
    evalOp e (decoded K V s) = evalOp e s :=
  (evalOp_agreeOn bounds e s (decoded K V s) (agree_outside K V s bounds hK hV) hsafe).symm

private theorem decode_zero : (Float8E5Bits.ofInt 0).decodeRat = some 0 := by
  norm_num [Float8E5Bits.decodeRat, Float8E5Bits.ofInt, Float8E5Bits.exponent,
    Float8E5Bits.fraction, Float8E5Bits.sign]

private theorem eval_decode (bytes : Op .int shape) (s : BlockState) :
    @evalOp .real shape (.decodeFloat8E5 .real bytes) s =
      (evalOp bytes s).bind (Float8E5Bits.decodeTile .real) := by
  rw [evalOp.eq_def]
  rfl

/-- The source's masked byte load, decoded after the load, gives the same
real tile as a masked real load from the decoded input view. Finite-byte and
region requirements apply only to active lanes; inactive lanes use zero. -/
theorem masked_decode (K V : RegionName) (s : BlockState)
    (ptr : Op .ptr shape) (mask : Op .bool shape)
    (ptrs : Tile .ptr shape) (masks : Tile .bool shape)
    (hp : evalOp ptr s = some ptrs) (hm : evalOp mask s = some masks)
    (hpure : ptr.SafeAt (fun _ => 0) s) (mpure : mask.SafeAt (fun _ => 0) s)
    (hregions : ∀ i, masks.data i = Bool.true → (ptrs.data i).1 = K ∨ (ptrs.data i).1 = V)
    (hfinite : ∀ i, masks.data i = Bool.true →
      FiniteByte (s.readMemValue .int (ptrs.data i).1 (ptrs.data i).2)) :
    @evalOp .real shape (.decodeFloat8E5 .real
      (.load .int (.ptr ptr) (.maskOther mask (.broadcast (.constInt 0) shape)))) s =
      evalOp (.load .real (.ptr ptr) (.maskOther mask (.broadcast (.const 0) shape)))
        (decoded K V s) := by
  let raw : Tile .int shape := ⟨fun i =>
    if masks.data i then s.readMemValue .int (ptrs.data i).1 (ptrs.data i).2 else 0⟩
  let values : TileIndex shape → ℚ := fun i =>
    if masks.data i then
      (Float8E5Bits.ofInt (s.readMemValue .int (ptrs.data i).1 (ptrs.data i).2)).decodeRat.getD 0
    else 0
  have hload : evalOp (.load .int (.ptr ptr)
      (.maskOther mask (.broadcast (.constInt 0) shape))) s = some raw := by
    simp only [evalOp, hp, hm, bind, Option.bind_some, raw, Tile.scalar_data, ↓reduceIte]
  have hdecode : Float8E5Bits.decodeTile .real raw =
      some (⟨fun i => some (values i : ℝ)⟩ : Tile .real shape) := by
    apply Float8E5Bits.decodeTile_of_finite .real raw values
    · intro i
      by_cases hi : masks.data i = Bool.true
      · simpa [raw, hi] using (hfinite i hi).1
      · simp [raw, hi, Float8E5Bits.isInt8]
    · intro i
      by_cases hi : masks.data i = Bool.true
      · have hf := (hfinite i hi).2
        cases hval : (Float8E5Bits.ofInt
          (s.readMemValue .int (ptrs.data i).1 (ptrs.data i).2)).decodeRat with
        | none => simp [hval] at hf
        | some q => simp [raw, values, hi, hval]
      · simp [raw, values, hi, decode_zero]
  rw [eval_decode, hload]
  simp only [Option.bind_some, hdecode]
  have hpview := (eval_outside K V s (fun _ => 0) rfl rfl ptr hpure).trans hp
  have hmview := (eval_outside K V s (fun _ => 0) rfl rfl mask mpure).trans hm
  rw [RealKernel.al_load_ptr_maskOther_real ptr mask (.broadcast (.const 0) shape)
    (decoded K V s) ptrs masks (⟨fun _ => some 0⟩ : Tile .real shape) hpview hmview (by
      simp only [evalOp, bind, Option.bind_some, Tile.scalar_data])]
  congr 1
  ext i
  by_cases hi : masks.data i = Bool.true
  · simp [values, hi, read_decoded K V s _ _ (hregions i hi), value]
  · simp [values, hi]

/-- The decoded input view commutes with an output-cell write when the
output does not alias either byte buffer. The complete state is preserved. -/
theorem decoded_writeCell (K V : RegionName) (s : BlockState)
    (r : RegionName) (o : Nat) (cell : MemCell) (hK : r ≠ K) (hV : r ≠ V) :
    decoded K V (s.writeCell r o cell) = (decoded K V s).writeCell r o cell := by
  apply BlockState.ext
  · intro region offset
    by_cases hin : region = K ∨ region = V
    · have hne : region ≠ r := by
        rcases hin with h | h <;> subst region <;> exact Ne.symm ‹_›
      simp [decoded, BlockState.writeCell, hin, hne, value,
        BlockState.readMemValue, BlockState.readMemTyped]
    · simp only [decoded, BlockState.writeCell, hin, ↓reduceIte]
  · intros; rfl
  · intros; rfl
  · intros; rfl
  · intros; rfl

theorem decoded_writeMem (K V : RegionName) (s : BlockState)
    (r : RegionName) (o : Nat) (v : ℝ) (hK : r ≠ K) (hV : r ≠ V) :
    decoded K V (s.writeMem r o v) = (decoded K V s).writeMem r o v :=
  decoded_writeCell K V s r o (.real v) hK hV

/-- Masked output writes commute one by one with the decoded input view.
No separation requirement is imposed on an inactive pointer. -/
theorem decoded_store_fold (K V : RegionName) (indices : List α)
    (ptr : α → RegionName × Nat) (values : α → ℝ) (active : α → Bool)
    (hregions : ∀ i, active i = Bool.true → (ptr i).1 ≠ K ∧ (ptr i).1 ≠ V)
    (s : BlockState) :
    decoded K V (indices.foldl (fun t i =>
      if active i then t.writeMem (ptr i).1 (ptr i).2 (values i) else t) s) =
      indices.foldl (fun t i =>
        if active i then t.writeMem (ptr i).1 (ptr i).2 (values i) else t) (decoded K V s) := by
  induction indices generalizing s with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons, ih]
    by_cases ha : active i = Bool.true
    · simp only [ha, ↓reduceIte, decoded_writeMem K V s _ _ _
        (hregions i ha).1 (hregions i ha).2]
    · simp [ha]

/-- A real masked pointer store has the same effect in the decoded view.
Pointer, mask and value expressions are register computations; active output
pointers must avoid both byte buffers. -/
theorem masked_store (K V : RegionName) (s : BlockState)
    (ptr : Op .ptr shape) (mask : Op .bool shape) (valueExpr : Op .real shape)
    (ptrs : Tile .ptr shape) (masks : Tile .bool shape) (values : Tile .real shape)
    (hp : evalOp ptr s = some ptrs) (hm : evalOp mask s = some masks)
    (hv : evalOp valueExpr s = some values)
    (hpure : ptr.SafeAt (fun _ => 0) s) (mpure : mask.SafeAt (fun _ => 0) s)
    (vpure : valueExpr.SafeAt (fun _ => 0) s)
    (hregions : ∀ i, masks.data i = Bool.true → (ptrs.data i).1 ≠ K ∧ (ptrs.data i).1 ≠ V) :
    (stepStmt (.store .real shape (.ptr ptr) valueExpr (.mask mask)) s).map (decoded K V) =
      stepStmt (.store .real shape (.ptr ptr) valueExpr (.mask mask)) (decoded K V s) := by
  have hpview := (eval_outside K V s (fun _ => 0) rfl rfl ptr hpure).trans hp
  have hmview := (eval_outside K V s (fun _ => 0) rfl rfl mask mpure).trans hm
  have hvview := (eval_outside K V s (fun _ => 0) rfl rfl valueExpr vpure).trans hv
  simp only [stepStmt, hp, hm, hv, hpview, hmview, hvview, bind,
    Option.bind_some, Option.map_some, BlockState.writeMemTyped_real]
  exact congrArg some (decoded_store_fold K V (TileShape.allIndices shape) ptrs.data
    (fun i => FloatDType.real.storeValue (values.data i)) masks.data hregions s)

/-- Assignments that cannot read K/V commute with changing their input
representation. This includes all purely arithmetic recurrence statements. -/
theorem assign_outside (K V : RegionName) (s : BlockState) (bounds : RegionBounds)
    (hK : bounds K = 0) (hV : bounds V = 0) (name : RegName) (e : Op d sh)
    (hsafe : e.SafeAt bounds s) :
    (stepStmt (.assign d sh name e) s).map (decoded K V) =
      stepStmt (.assign d sh name e) (decoded K V s) := by
  simp only [stepStmt, eval_outside K V s bounds hK hV e hsafe]
  cases evalOp e s <;> simp [decoded_setReg]

/-- A straight-line register fragment may read other input buffers, but
cannot observe the byte buffers under this access-safety condition. -/
theorem assignments_outside (K V : RegionName) (bounds : RegionBounds)
    (hK : bounds K = 0) (hV : bounds V = 0) (code : List Stmt)
    (hassign : ∀ st ∈ code, ∃ d sh name, ∃ e : Op d sh, st = .assign d sh name e)
    (s : BlockState) (hsafe : Stmt.TraceSafeList bounds code s) :
    (stepStmts code s).map (decoded K V) = stepStmts code (decoded K V s) := by
  induction code generalizing s with
  | nil => simp only [stepStmts.nil, Option.map_some]
  | cons st rest ih =>
    obtain ⟨d, sh, name, e, rfl⟩ := hassign st (by simp)
    have hrest : ∀ st ∈ rest, ∃ d sh name, ∃ e : Op d sh, st = .assign d sh name e :=
      fun st h => hassign st (by simp [h])
    simp only [Stmt.TraceSafeList, Stmt.TraceSafe] at hsafe
    have hs : e.SafeAt bounds s := hsafe.1
    have hstep := assign_outside K V s bounds hK hV name e hs
    cases hr : stepStmt (.assign d sh name e) s with
    | none =>
      have hd : stepStmt (.assign d sh name e) (decoded K V s) = none := by
        simpa only [hr, Option.map_none] using hstep.symm
      simp only [stepStmts, hr, hd]
      rfl
    | some t =>
      have hd : stepStmt (.assign d sh name e) (decoded K V s) = some (decoded K V t) := by
        simpa only [hr, Option.map_some] using hstep.symm
      have ht : Stmt.TraceSafeList bounds rest t := by
        simpa only [hr] using hsafe.2
      simpa only [stepStmts, hr, hd] using ih hrest t ht

/-- Compose an unchanged pure register assignment with a decoded-view
correspondence for the remaining program. -/
theorem pure_cons (K V : RegionName) (s : BlockState) (name : RegName)
    (e : Op d sh) (lhs rhs : List Stmt) (hpure : e.SafeAt (fun _ => 0) s)
    (htail : ∀ tile : Tile d sh,
      (stepStmts lhs (s.setReg name d sh tile)).map (decoded K V) =
        stepStmts rhs (decoded K V (s.setReg name d sh tile))) :
    (stepStmts (.assign d sh name e :: lhs) s).map (decoded K V) =
      stepStmts (.assign d sh name e :: rhs) (decoded K V s) := by
  simp only [stepStmts, stepStmt, eval_outside K V s (fun _ => 0) rfl rfl e hpure]
  cases he : evalOp e s with
  | none => simp
  | some tile => simpa [he, decoded_setReg] using htail tile

theorem evaluated_cons (K V : RegionName) (s : BlockState) (name : RegName)
    (e : Op d sh) (tile : Tile d sh) (lhs rhs : List Stmt)
    (hpure : e.SafeAt (fun _ => 0) s) (he : evalOp e s = some tile)
    (htail : (stepStmts lhs (s.setReg name d sh tile)).map (decoded K V) =
      stepStmts rhs (decoded K V (s.setReg name d sh tile))) :
    (stepStmts (.assign d sh name e :: lhs) s).map (decoded K V) =
      stepStmts (.assign d sh name e :: rhs) (decoded K V s) := by
  simpa only [stepStmts, stepStmt,
    eval_outside K V s (fun _ => 0) rfl rfl e hpure, he, bind, Option.bind_some,
    decoded_setReg] using htail

/-- One actual byte-load assignment simulates the matching real-load
assignment. The relation includes failure and every state channel. -/
theorem assign_masked_decode (K V : RegionName) (s : BlockState) (name : RegName)
    (ptr : Op .ptr shape) (mask : Op .bool shape)
    (ptrs : Tile .ptr shape) (masks : Tile .bool shape)
    (hp : evalOp ptr s = some ptrs) (hm : evalOp mask s = some masks)
    (hpure : ptr.SafeAt (fun _ => 0) s) (mpure : mask.SafeAt (fun _ => 0) s)
    (hregions : ∀ i, masks.data i = Bool.true → (ptrs.data i).1 = K ∨ (ptrs.data i).1 = V)
    (hfinite : ∀ i, masks.data i = Bool.true →
      FiniteByte (s.readMemValue .int (ptrs.data i).1 (ptrs.data i).2)) :
    (stepStmt (.assign .real shape name (.decodeFloat8E5 .real
      (.load .int (.ptr ptr) (.maskOther mask (.broadcast (.constInt 0) shape))))) s).map
        (decoded K V) =
      stepStmt (.assign .real shape name
        (.load .real (.ptr ptr) (.maskOther mask (.broadcast (.const 0) shape))))
        (decoded K V s) := by
  simp only [stepStmt]
  rw [masked_decode K V s ptr mask ptrs masks hp hm hpure mpure hregions hfinite]
  cases evalOp (.load .real (.ptr ptr) (.maskOther mask (.broadcast (.const 0) shape)))
      (decoded K V s) <;> simp [decoded_setReg]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionLlama.FP8RealMemory
