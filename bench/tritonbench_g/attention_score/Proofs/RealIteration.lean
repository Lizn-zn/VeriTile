import VeriTile.Triton.Semantics.UncheckedBlockLoad
import bench.tritonbench_g.attention_score.Proofs.RealBody
import bench.tritonbench_g.attention_score.Proofs.RealArithmeticOutput

/-! Complete iteration execution from the original input memory: checked or
unchecked Q loads, unmasked M loads, independent arithmetic and both pointer
advances. The frame preserves the static K tile, scale and loop metadata. -/
namespace VeriTile.Bench.TritonBenchG.AttentionScore.RealIteration
open VeriTile.Triton RealFrame
set_option maxHeartbeats 1000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false
noncomputable section

theorem load_query (s : BlockState) (Q K M : RegionName)
    (B D N baseQ baseK baseM sqm sqk skn skk row : Nat) (evenM : Bool)
    (hp : s.regs .blockPtr [B,D] "Q_block_ptr" = some (RealLoads.queryPointer Q B D N baseQ sqm sqk row)) :
    stepStmt (RealBody.loadQuery B D evenM) s = some (s.setReg "q" .real [B,D]
      (RealTiles.query (RealLoads.inputs s Q K M D baseQ baseK baseM sqm sqk skn skk) B N row evenM)) := by
  have hl := RealLoads.query_load s Q K M B D N baseQ baseK baseM sqm sqk skn skk row evenM hp
  have hc : evalOp (.constBool evenM) s = some (Tile.scalar evenM) := by rw [evalOp]
  cases evenM
  all_goals simp only [RealLoads.checks, Bool.false_eq_true, ↓reduceIte] at hl
  all_goals simp [RealBody.loadQuery, stepStmt, hc, stepStmts, evalOp_unchecked_block_load, hl]

theorem load_key (s : BlockState) (Q K M : RegionName)
    (B D NK baseQ baseK baseM sqm sqk skn skk startN : Nat) (evenN : Bool)
    (hp : s.regs .blockPtr [D,B] "K_block_ptr" = some (RealLoads.keyPointer K B D NK baseK skn skk startN)) :
    stepStmt (RealBody.loadKey B D evenN) s = some (s.setReg "k" .real [D,B]
      (RealTiles.key (RealLoads.inputs s Q K M D baseQ baseK baseM sqm sqk skn skk) B NK startN evenN)) := by
  have hl := RealLoads.key_load s Q K M B D NK baseQ baseK baseM sqm sqk skn skk startN evenN hp
  have hc : evalOp (.constBool evenN) s = some (Tile.scalar evenN) := by rw [evalOp]
  cases evenN
  all_goals simp only [RealLoads.checks, Bool.false_eq_true, ↓reduceIte] at hl
  all_goals simp [RealBody.loadKey, stepStmt, hc, stepStmts, evalOp_unchecked_block_load, hl]

theorem load_prefix (initial s : BlockState) (Q K M : RegionName)
    (B D N baseQ baseK baseM sqm sqk skn skk row : Nat) (evenM : Bool)
    (hmem : s.mem = initial.mem)
    (hr : s.regs .nat [] "start_m" = some (Tile.scalar row))
    (hp : s.regs .blockPtr [B,D] "Q_block_ptr" = some (RealLoads.queryPointer Q B D N baseQ sqm sqk row))
    (hmp : s.regs .ptr [B] "m_ptrs" = some (RealLoads.normalizerPointers M B baseM row)) :
    let xs := RealLoads.inputs initial Q K M D baseQ baseK baseM sqm sqk skn skk
    ∃ t, stepStmts (RealBody.loadPrefix B D evenM) s = some t ∧
      t.regs .nat [] "start_m" = some (Tile.scalar row) ∧
      t.regs .real [B,D] "q" = some (RealTiles.query xs B N row evenM) ∧
      t.regs .real [B] "m" = some (RealTiles.normalizer xs B row) ∧
      Frame RealBody.prefixWrites s t := by
  dsimp only
  let xs := RealLoads.inputs initial Q K M D baseQ baseK baseM sqm sqk skn skk
  let sm := s.setReg "start_m" .nat [] (Tile.scalar row)
  let sq := sm.setReg "q" .real [B,D] (RealTiles.query xs B N row evenM)
  let st := sq.setReg "m" .real [B] (RealTiles.normalizer xs B row)
  have hs : stepStmt (.assign .nat [] "start_m" (.ref .nat [] "start_m")) s = some sm := by
    simp [stepStmt, evalOp_ref, hr, sm]
  have hq : stepStmt (RealBody.loadQuery B D evenM) sm = some sq := by
    have h := load_query sm Q K M B D N baseQ baseK baseM sqm sqk skn skk row evenM (by simp [sm,hp])
    rw [RealLoads.inputs_same_memory sm initial Q K M D baseQ baseK baseM sqm sqk skn skk hmem] at h
    exact h
  have hm : stepStmt (.assign .real [B] "m" (.load .real (.ptr (.ref .ptr [B] "m_ptrs")) .none)) sq = some st := by
    have h := RealLoads.normalizer_load sq Q K M B D baseQ baseK baseM sqm sqk skn skk row (by simp [sq,sm,hmp])
    rw [RealLoads.inputs_same_memory sq initial Q K M D baseQ baseK baseM sqm sqk skn skk hmem] at h
    simp only [stepStmt, h, bind, Option.bind, st, xs]
  have he : stepStmts (RealBody.loadPrefix B D evenM) s = some st := by
    rw [RealBody.loadPrefix, stepStmts.cons_some hs, stepStmts.cons_some hq,
      stepStmts.cons_some hm, stepStmts.nil]
  exact ⟨st,he,by simp [st,sq,sm],by simp [st,sq,xs],by simp [st,xs],(RealBody.prefix_code B D evenM).frame he⟩

theorem advance (s : BlockState) (Q M : RegionName)
    (B D N baseQ baseM sqm sqk row : Nat)
    (hp : s.regs .blockPtr [B,D] "Q_block_ptr" = some (RealLoads.queryPointer Q B D N baseQ sqm sqk row))
    (hmp : s.regs .ptr [B] "m_ptrs" = some (RealLoads.normalizerPointers M B baseM row)) :
    ∃ t, stepStmts (RealBody.suffix B D) s = some t ∧
      t.regs .blockPtr [B,D] "Q_block_ptr" = some (RealLoads.queryPointer Q B D N baseQ sqm sqk (row+B)) ∧
      t.regs .ptr [B] "m_ptrs" = some (RealLoads.normalizerPointers M B baseM (row+B)) ∧
      Frame RealBody.suffixWrites s t := by
  let sq := s.setReg "Q_block_ptr" .blockPtr [B,D] (RealLoads.queryPointer Q B D N baseQ sqm sqk (row+B))
  let st := sq.setReg "m_ptrs" .ptr [B] (RealLoads.normalizerPointers M B baseM (row+B))
  have hq : stepStmt (.assign .blockPtr [B,D] "Q_block_ptr"
      (.advanceBlockPtr (.ref .blockPtr [B,D] "Q_block_ptr") [B,0])) s = some sq := by
    simp only [stepStmt, RealLoads.query_advance s Q B D N baseQ sqm sqk row hp, bind, Option.bind_some, sq]
  have hm : stepStmt (.assign .ptr [B] "m_ptrs"
      (.ptrAdd .scalarR (.ref .ptr [B] "m_ptrs") (.constNat B))) sq = some st := by
    simp only [stepStmt, RealLoads.normalizer_advance sq M B baseM row (by simp [sq,hmp]), bind, Option.bind_some, st]
  have he : stepStmts (RealBody.suffix B D) s = some st := by
    rw [RealBody.suffix, stepStmts.cons_some hq, stepStmts.cons_some hm, stepStmts.nil]
  exact ⟨st,he,by simp [st,sq],by simp [st],(RealBody.suffix_code B D).frame he⟩

/-- Convert finite lane observations into the actual carried vector register.
A Triton block is nonempty; this also excludes vacuous observations of none. -/
theorem vector_of_cells {B : Nat} (hB : 0 < B) (reg : Option (Tile .real [B]))
    (values : Fin B → ℝ)
    (hv : ∀ i, reg.map (fun v => v.data (i,PUnit.unit)) = some (some (values i))) :
    reg = some (Tile.vec (fun i => some (values i))) := by
  cases reg with
  | none => simpa using hv ⟨0,hB⟩
  | some value =>
      congr 1
      apply Tile.ext
      rintro ⟨i,⟨⟩⟩
      exact Option.some.inj (hv i)

/-- Exactly the registers modified by a complete source iteration. -/
def writes : List RegName :=
  ["start_m","q","m","qk","dist","mask","p","o","Q_block_ptr","m_ptrs"]

/-- A complete iteration reads original Q/M cells, adds the independent query
block contribution, advances both pointers and preserves every other register.
All window/load flags and signed parameters remain unrestricted. -/
theorem correct (initial s : BlockState) (Q K M : RegionName)
    (B D N NK baseQ baseK baseM sqm sqk skn skk row startN : Nat) (hB : 0 < B)
    (offset size : Int) (scale : ℝ) (window complement evenM evenN candidate : Bool)
    (old : Fin B → ℝ)
    (hmem : s.mem = initial.mem)
    (hr : s.regs .nat [] "start_m" = some (Tile.scalar row))
    (hn : s.regs .nat [] "start_n" = some (Tile.scalar startN))
    (hp : s.regs .blockPtr [B,D] "Q_block_ptr" = some (RealLoads.queryPointer Q B D N baseQ sqm sqk row))
    (hmp : s.regs .ptr [B] "m_ptrs" = some (RealLoads.normalizerPointers M B baseM row))
    (hk : s.regs .real [D,B] "k" = some (RealTiles.key
      (RealLoads.inputs initial Q K M D baseQ baseK baseM sqm sqk skn skk) B NK startN evenN))
    (hs : s.regs .real [] "qk_scale" = some (Tile.scalar (some (scale * 1.4426950408889634))))
    (ho : s.regs .real [B] "o" = some (Tile.vec (fun k => some (old k)))) :
    let xs := RealLoads.inputs initial Q K M D baseQ baseK baseM sqm sqk skn skk
    ∃ t, stepStmts (RealBody.body B D N offset size window complement evenM evenN candidate) s = some t ∧
      t.regs .real [B] "o" = some (Tile.vec (fun k => some (old k + ∑ r : Fin B,
        RealFormula.weight xs scale N NK window complement evenM evenN offset size
          (row+r.val) (startN*B+k.val)))) ∧
      t.regs .blockPtr [B,D] "Q_block_ptr" = some (RealLoads.queryPointer Q B D N baseQ sqm sqk (row+B)) ∧
      t.regs .ptr [B] "m_ptrs" = some (RealLoads.normalizerPointers M B baseM (row+B)) ∧
      Frame writes s t := by
  dsimp only
  let xs := RealLoads.inputs initial Q K M D baseQ baseK baseM sqm sqk skn skk
  obtain ⟨loaded, hl, hr', hq', hm', fl⟩ := load_prefix initial s Q K M
    B D N baseQ baseK baseM sqm sqk skn skk row evenM hmem hr hp hmp
  have hn' := (fl.regs .nat [] "start_n" (by simp [RealBody.prefixWrites])).trans hn
  have hk' := (fl.regs .real [D,B] "k" (by simp [RealBody.prefixWrites])).trans hk
  have hs' := (fl.regs .real [] "qk_scale" (by simp [RealBody.prefixWrites])).trans hs
  have ho' := (fl.regs .real [B] "o" (by simp [RealBody.prefixWrites])).trans ho
  obtain ⟨computed, ha, _, _, _, hcells⟩ := RealArithmeticOutput.correct loaded xs
    B N NK row startN offset size scale window complement evenM evenN candidate old
    hq' hk' hm' ho' hs' hr' hn'
  have fa := (RealBody.arithmetic_code B D N offset size window complement evenN candidate).frame ha
  have hp' := (fa.regs .blockPtr [B,D] "Q_block_ptr" (by simp [RealBody.arithmeticWrites])).trans
    ((fl.regs .blockPtr [B,D] "Q_block_ptr" (by simp [RealBody.prefixWrites])).trans hp)
  have hmp' := (fa.regs .ptr [B] "m_ptrs" (by simp [RealBody.arithmeticWrites])).trans
    ((fl.regs .ptr [B] "m_ptrs" (by simp [RealBody.prefixWrites])).trans hmp)
  obtain ⟨t, ht, hpNext, hmNext, ft⟩ := advance computed Q M B D N baseQ baseM sqm sqk row hp' hmp'
  refine ⟨t, ?_, ?_, hpNext, hmNext, ?_⟩
  · simp only [RealBody.body, ProgramRewrite.append, hl, Option.bind_some, ha, ht]
  · rw [ft.regs .real [B] "o" (by simp [RealBody.suffixWrites])]
    exact vector_of_cells hB _ _ hcells
  · exact ((fl.mono (by simp [RealBody.prefixWrites,writes])).trans
      (fa.mono (by simp [RealBody.arithmeticWrites,writes]))).trans
        (ft.mono (by simp [RealBody.suffixWrites,writes]))

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttentionScore.RealIteration
