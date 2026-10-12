import bench.tritonbench_g.attention_llama.Proofs.FP8RealMemory

/-! The real output-address computation and masked store commute with the
K/V decoded input view. A separate raw-cell frame is needed because byte
decoding is not injective. The actual source postamble is used throughout. -/
namespace VeriTile.Bench.TritonBenchG.AttentionLlama.FP8RealPostamble
open VeriTile.Triton RealKernel FP8RealMemory
set_option maxRecDepth 16000
set_option maxHeartbeats 500000
set_option linter.unusedSimpArgs false

private theorem singleton (st : Stmt) (s : BlockState) :
    stepStmts [st] s = stepStmt st s := by
  simp only [stepStmts, stepStmts.nil]
  cases stepStmt st s <;> rfl

private theorem store_map (K V : RegionName) (s : BlockState)
    (ptr : Op .ptr sh) (mask : Op .bool sh) (valueExpr : Op .real sh)
    (ptrs : Tile .ptr sh) (hp : evalOp ptr s = some ptrs)
    (hpure : ptr.SafeAt (fun _ => 0) s) (mpure : mask.SafeAt (fun _ => 0) s)
    (vpure : valueExpr.SafeAt (fun _ => 0) s)
    (hregions : ∀ i, (ptrs.data i).1 ≠ K ∧ (ptrs.data i).1 ≠ V) :
    (stepStmt (.store .real sh (.ptr ptr) valueExpr (.mask mask)) s).map (decoded K V) =
      stepStmt (.store .real sh (.ptr ptr) valueExpr (.mask mask)) (decoded K V s) := by
  have hvview := eval_outside K V s (fun _ => 0) rfl rfl valueExpr vpure
  have hmview := eval_outside K V s (fun _ => 0) rfl rfl mask mpure
  cases hv : evalOp valueExpr s with
  | none => simp [stepStmt, hvview, hv]
  | some values =>
    cases hm : evalOp mask s with
    | none => simp [stepStmt, hvview, hmview, hv, hm]
    | some masks =>
      exact masked_store K V s ptr mask valueExpr ptrs masks values hp hm hv hpure mpure vpure
        (fun i _ => hregions i)

private theorem ptr_eval (Out : RegionName) (s : BlockState) (offsets : Tile .nat [BM, BD])
    (h : s.regs .nat [BM, BD] "off_o" = some offsets) :
    evalOp (.ptrAdd Broadcast.scalarL (.ptrBase Out) (.ref .nat [BM, BD] "off_o")) s =
      some (⟨fun i => (Out, offsets.data i)⟩ : Tile .ptr [BM, BD]) := by
  rw [evalOp_ptrAdd, evalOp_ptrBase]
  simp only [evalOp_ref, h, bind, Option.bind_some]
  apply congrArg some
  apply Tile.ext
  intro i
  exact congrArg (fun o => (Out, o)) (Nat.zero_add _)

/-- No accumulator equation is assumed: this relates the complete postamble,
including its missing-register failures and actual masked output effects. -/
theorem postamble (K V Out : RegionName) (s : BlockState)
    (soz soh som son H BM BD : Nat) (hK : Out ≠ K) (hV : Out ≠ V) :
    (stepStmts (alPostLoopG Out soz soh som son H BM BD) s).map (decoded K V) =
      stepStmts (alPostLoopG Out soz soh som son H BM BD) (decoded K V s) := by
  unfold alPostLoopG
  apply pure_cons (hpure := by simp [Op.SafeAt.eq_def]); intro start
  apply pure_cons (hpure := by simp [Op.SafeAt.eq_def]); intro rows
  apply pure_cons (hpure := by simp [Op.SafeAt.eq_def]); intro dims
  apply pure_cons (hpure := by simp [Op.SafeAt.eq_def]); intro offsets
  apply evaluated_cons (tile := (⟨fun i => (Out, offsets.data i)⟩ : Tile .ptr [BM, BD]))
    (hpure := by simp [Op.SafeAt.eq_def])
  · exact ptr_eval Out _ offsets (by simp)
  · rw [singleton, singleton]
    apply store_map (ptrs := (⟨fun i => (Out, offsets.data i)⟩ : Tile .ptr [BM, BD]))
    · simp [evalOp_ref]
    · simp [Op.SafeAt.eq_def]
    · simp [Op.SafeAt.eq_def]
    · simp [Op.SafeAt.eq_def]
    · intro i; exact ⟨hK, hV⟩

private theorem masked_fold_frame (Out r : RegionName) (o : Nat) (hr : r ≠ Out)
    (indices : List α) (offsets : α → Nat) (values : α → ℝ) (mask : α → Bool)
    (s : BlockState) :
    (indices.foldl (fun t i => if mask i then t.writeMem Out (offsets i) (values i) else t) s).mem r o =
      s.mem r o := by
  induction indices generalizing s with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons, ih]
    split <;> simp [BlockState.writeMem_mem, hr]

private theorem store_frame (Out r : RegionName) (o : Nat) (hr : r ≠ Out)
    (s t : BlockState) (ptr : Op .ptr sh) (mask : Op .bool sh) (val : Op .real sh)
    (offsets : Tile .nat sh)
    (hp : evalOp ptr s = some (⟨fun i => (Out, offsets.data i)⟩ : Tile .ptr sh))
    (hrun : stepStmt (.store .real sh (.ptr ptr) val (.mask mask)) s = some t) :
    t.mem r o = s.mem r o := by
  cases hv : evalOp val s with
  | none => simp [stepStmt, hv] at hrun
  | some values =>
    cases hm : evalOp mask s with
    | none => simp [stepStmt, hv, hm] at hrun
    | some masks =>
      simp only [stepStmt, hv, hm, hp, bind, Option.bind_some, Option.map_some,
        BlockState.writeMemTyped_real, Option.some.injEq] at hrun
      subst t
      exact masked_fold_frame Out r o hr _ offsets.data
        (fun i => FloatDType.real.storeValue (values.data i)) masks.data s

private theorem frame_cons (s t : BlockState) (r : RegionName) (o : Nat)
    (name : RegName) (e : Op d sh) (rest : List Stmt)
    (hrun : stepStmts (.assign d sh name e :: rest) s = some t)
    (htail : ∀ tile : Tile d sh, stepStmts rest (s.setReg name d sh tile) = some t →
      t.mem r o = (s.setReg name d sh tile).mem r o) :
    t.mem r o = s.mem r o := by
  cases he : evalOp e s with
  | none => simp [stepStmts, stepStmt, he] at hrun
  | some tile =>
    apply htail tile
    simpa only [stepStmts, stepStmt, he, bind, Option.bind_some] using hrun

/-- Every raw cell outside Out is preserved by the actual postamble. This
protects the K/V byte buffers without relying on decoding to be injective. -/
theorem memory_frame (Out r : RegionName) (o : Nat) (s t : BlockState)
    (soz soh som son H BM BD : Nat) (hr : r ≠ Out)
    (hrun : stepStmts (alPostLoopG Out soz soh som son H BM BD) s = some t) :
    t.mem r o = s.mem r o := by
  unfold alPostLoopG at hrun
  apply frame_cons (hrun := hrun); intro start hrun
  apply frame_cons (hrun := hrun); intro rows hrun
  apply frame_cons (hrun := hrun); intro dims hrun
  apply frame_cons (hrun := hrun); intro offsets hrun
  let ps := (((s.setReg "start_m" .nat [] start).setReg "offs_m" .nat [BM] rows).setReg
    "offs_d" .nat [BD] dims).setReg "off_o" .nat [BM, BD] offsets
  have hp := ptr_eval Out ps offsets (by simp [ps])
  rw [stepStmts.cons_some (stepStmt_assign_eq_some hp), singleton] at hrun
  apply store_frame Out r o hr _ t _ _ _ offsets (hrun := hrun)
  simp [evalOp_ref]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionLlama.FP8RealPostamble
