import bench.tritonbench_g.attn_fwd_causal.Proofs.RealSource

/-! Real normalization and the actual masked output store. Address
injectivity is needed only among active output lanes. Other cells are
preserved even when inactive tile lanes share physical addresses. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealEpilogue
open VeriTile.Triton RealLoads RealPointers
open AttentionFwdTriton3.RealTiles (finite)
open AttentionFwdTriton3.RealScores (Frame)
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

def active (c : Config) (BM BD : Nat) (i : TileIndex [BM,BD]) : Bool :=
  decide (c.startM*BM+i.1.val < c.context ∧ i.2.1.val < c.activeHead)
def address (c : Config) (BM BD : Nat) (i : TileIndex [BM,BD]) : Nat :=
  c.base+(c.startM*BM+i.1.val)*c.strideM+i.2.1.val*c.strideD

def writeOutput (s : BlockState) (c : Config) (Out : RegionName) (BM BD : Nat)
    (values : TileIndex [BM,BD] → ℝ) : BlockState :=
  (TileShape.allIndices [BM,BD]).foldl (fun st i =>
    if active c BM BD i then st.writeMem Out (address c BM BD i) (values i) else st) s

theorem normalize_executes (s : BlockState) (BM BD : Nat) (candidate : Bool)
    (a : TileIndex [BM,BD] → ℝ) (l : Fin BM → ℝ)
    (ha : s.regs .real [BM,BD] "acc" = some (finite a))
    (hl : s.regs .real [BM] "l_i" = some (finite (fun i : TileIndex [BM] => l i.1))) :
    stepStmt (RealSource.normalization BM BD candidate) s =
      some (s.setReg "acc" .real [BM,BD] (finite (fun i : TileIndex [BM,BD] => a i/l i.1))) := by
  apply stepStmt_assign_eq_some
  cases candidate
  all_goals
    simp only [RealSource.normalization, Bool.false_eq_true, Bool.true_eq, ↓reduceIte,
      evalOp.eq_def, ha, hl, bind, Option.bind]
  all_goals congr 1
  all_goals
    apply Tile.ext
    rintro ⟨r,d,⟨⟩⟩
    norm_num [finite, Tile.bop_data, Tile.expandDim_data, Tile.scalar,
      TileShape.dropInsertedIndex, NumericDType.mul, NumericDType.div,
      WithBot.realMul, WithBot.realDiv, div_eq_mul_inv, -Tile.scalar_eta]

theorem store_executes (s : BlockState) (c : Config) (Out : RegionName) (BM BD : Nat)
    (values : TileIndex [BM,BD] → ℝ)
    (hp : s.regs .ptr [BM,BD] "O_block_ptr" = some (outputPointer c Out BM BD))
    (hr : s.regs .nat [BM] "offs_m" = some (rows BM (c.startM*BM)))
    (hv : s.regs .real [BM,BD] "acc" = some (finite values)) :
    stepStmt (RealSource.store BM BD c.context c.activeHead) s = some (writeOutput s c Out BM BD values) := by
  have hm := RealInit.eval_mask s BM BD c.startM c.context c.activeHead hr
  simp only [RealSource.store]
  conv_lhs => unfold stepStmt
  simp only [evalOp_ref, hp, hv, hm, bind, Option.bind, Option.map_some]
  simp [writeOutput, active, address, outputPointer, rowPointer, RealInit.mask, finite,
    BlockState.writeMemTyped_real, FloatDType.storeValue, FloatDType.real_toWithBot, WithBot.unbotD_some, Region.cast]

theorem frame (s : BlockState) (c : Config) (Out : RegionName) (BM BD : Nat)
    (values : TileIndex [BM,BD] → ℝ) (r : RegionName) (a : Nat)
    (h : r ≠ Out ∨ ∀ i, active c BM BD i = Bool.true → a ≠ address c BM BD i) :
    (writeOutput s c Out BM BD values).mem r a = s.mem r a := by
  unfold writeOutput
  generalize TileShape.allIndices [BM,BD] = indices
  induction indices generalizing s with
  | nil => rfl
  | cons i rest ih =>
    rw [List.foldl_cons]
    split_ifs with hi
    · rw [ih]
      have hn : ¬ (r = Out ∧ a = address c BM BD i) := by
        rintro ⟨hr,ha⟩
        rcases h with hr' | hi'
        · exact hr' hr
        · exact hi' i hi ha
      simp [BlockState.writeMem, hn]
    · exact ih s

/-- Only other active writes can overwrite this lane's result. -/
theorem readback_list {α : Type} (indices : List α) (s : BlockState) (Out : RegionName)
    (offset : α → Nat) (value : α → ℝ) (mask : α → Bool) (i : α)
    (hnd : indices.Nodup) (hi : i ∈ indices) (hactive : mask i = Bool.true)
    (hinj : ∀ a b, mask a = Bool.true → mask b = Bool.true → offset a = offset b → a = b) :
    (indices.foldl (fun st j => if mask j then st.writeMem Out (offset j) (value j) else st) s).readMem Out (offset i) = value i := by
  obtain ⟨before,after,hlist⟩ := List.append_of_mem hi
  rw [hlist, List.nodup_append, List.nodup_cons] at hnd
  have hnot : i ∉ after := hnd.2.1.1
  rw [hlist, List.foldl_append, List.foldl_cons, hactive, if_pos rfl]
  rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem Out offset value
    (fun j => mask j = Bool.true) after _ (offset i) (by
    intro j hj hm he
    have hji := hinj j i hm hactive he.symm
    exact hnot (hji ▸ hj))]
  simp [BlockState.writeMem_readMem]

theorem readback (s : BlockState) (c : Config) (Out : RegionName) (BM BD : Nat)
    (values : TileIndex [BM,BD] → ℝ) (i : TileIndex [BM,BD])
    (hi : active c BM BD i = Bool.true)
    (hinj : ∀ a b, active c BM BD a = Bool.true → active c BM BD b = Bool.true →
      address c BM BD a = address c BM BD b → a = b) :
    (writeOutput s c Out BM BD values).readMem Out (address c BM BD i) = values i :=
  readback_list _ s Out _ _ _ i (TileShape.allIndices_nodup _) (TileShape.mem_allIndices _ i) hi hinj

theorem executes (s : BlockState) (c : Config) (Out : RegionName) (BM BD : Nat) (candidate : Bool)
    (a : TileIndex [BM,BD] → ℝ) (l : Fin BM → ℝ)
    (ha : s.regs .real [BM,BD] "acc" = some (finite a))
    (hl : s.regs .real [BM] "l_i" = some (finite (fun i : TileIndex [BM] => l i.1)))
    (hp : s.regs .ptr [BM,BD] "O_block_ptr" = some (outputPointer c Out BM BD))
    (hr : s.regs .nat [BM] "offs_m" = some (rows BM (c.startM*BM))) :
    ∃ t, stepStmts (RealSource.epilogue BM BD c.context c.activeHead candidate) s = some t ∧
      (∀ i, active c BM BD i = Bool.true →
        (∀ a b, active c BM BD a = Bool.true → active c BM BD b = Bool.true →
          address c BM BD a = address c BM BD b → a = b) →
        t.readMem Out (address c BM BD i) = a i/l i.1) ∧
      (∀ r offset, (r ≠ Out ∨ ∀ i, active c BM BD i = Bool.true → offset ≠ address c BM BD i) →
        t.mem r offset = s.mem r offset) := by
  let normalized := fun i : TileIndex [BM,BD] => a i/l i.1
  let sn := s.setReg "acc" .real [BM,BD] (finite normalized)
  have hn := normalize_executes s BM BD candidate a l ha hl
  have hstore := store_executes sn c Out BM BD normalized (by simp [sn, hp])
    (by simp [sn, hr]) (by simp [sn])
  refine ⟨writeOutput sn c Out BM BD normalized, ?_, ?_, ?_⟩
  · rw [RealSource.epilogue, stepStmts.cons_some hn, stepStmts.cons_some hstore, stepStmts.nil]
  · intro i hi hinj
    exact readback sn c Out BM BD normalized i hi hinj
  · intro r offset h
    exact frame sn c Out BM BD normalized r offset h

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealEpilogue
