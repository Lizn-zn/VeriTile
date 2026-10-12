import bench.tritonbench_g.block_sparse_attn.Proofs.RealStores

/-! Access safety for the actual final stores. Only active output rows need
allocated cells; the second channel bound is required only on its branch. -/
namespace VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealStores
open VeriTile.Triton RealNormalizationValues
noncomputable section
set_option linter.unusedSimpArgs false

def Bounds (bounds : RegionBounds) (s : BlockState) (BM BD : Nat)
    (p : RealProgram.Parameters) (second : Bool) : Prop :=
  ∀ r d, RealSetup.rows s BM r < p.context → offset s BM BD p second r d < bounds p.Out

theorem store_safe (bounds : RegionBounds) (s t : BlockState) (BM BD : Nat)
    (p : RealProgram.Parameters) (second : Bool)
    (hp : t.regs .ptr [BM, BD] "out_ptrs" = some (pointers s BM BD p))
    (hr : t.regs .nat [BM] "offs_m" = some (Tile.vec (RealSetup.rows s BM)))
    (hctx : t.regs .nat [] "q_seq_len" = some (Tile.scalar p.context))
    (hb : Bounds bounds s BM BD p second) :
    Stmt.TraceSafe bounds (store BM BD second) t := by
  have hm := RealQuery.mask_eval (BD := BD) t (RealSetup.rows s BM) p.context hr hctx
  have ha : (MemAccess.ptr (d := .real) (pointer BM BD second)).ActiveAddressSafe bounds t
      ((MaskOpt.mask (dtype := .real) (RealQuery.mask BM BD)).Active t) := by
    refine ⟨?_, ?_⟩
    · cases second <;> simp [pointer, Op.MemorySafe]
    · intro tile ht
      rw [pointer_eval s t BM BD p second hp] at ht
      cases ht
      rintro ⟨r, d, ⟨⟩⟩ hactive
      have hrow : RealSetup.rows s BM r < p.context := by
        simpa [MaskOpt.Active, hm] using hactive
      exact hb r d hrow
  rw [store, Stmt.TraceSafe]
  refine ⟨?_, ?_, ?_, ha⟩
  · cases second <;> simp [pointer, MemAccess.SafeAt, Op.SafeAt.eq_def]
  · simp [Op.SafeAt.eq_def]
  · simp [MaskOpt.SafeAt, RealQuery.mask, Op.SafeAt.eq_def]

theorem code_safe (bounds : RegionBounds) (s t : BlockState) (BM BN BD : Nat)
    (p : RealProgram.Parameters) (x : Inputs BM BN BD)
    (hf : RealLoop.Fixed s BM BN BD p t) (hc : RealLoop.CarryAt x p.blocks t)
    (hb : Bounds bounds s BM BD p Bool.false)
    (hb2 : 2 ≤ p.blocks → Bounds bounds s BM BD p Bool.true) :
    Stmt.TraceSafeList bounds (code BM BD p) t := by
  have ho := stepStmt_assign_eq_some (name := "off_o") (offset_eval s t BM BN BD p hf)
  have hp : stepStmt (.assign .ptr [BM, BD] "out_ptrs"
      (.ptrAdd .scalarL (.ptrBase p.Out) (.ref .nat [BM, BD] "off_o")))
      (t.setReg "off_o" .nat [BM, BD] (Tile.mat (offset s BM BD p Bool.false))) =
      some (prepared s t BM BD p) := by
    apply stepStmt_assign_eq_some
    simp [evalOp.eq_def, bind, Option.bind, pointers]
    apply Tile.ext
    rintro ⟨r, d, ⟨⟩⟩
    simp only [Tile.ptrAdd, Tile.mat, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex, Nat.zero_add]
  unfold code
  refine Stmt.TraceSafeList.cons_intro ?_ ?_
  · simp [Stmt.TraceSafe, offsetOp, Op.SafeAt.eq_def]
  intro u hu
  rw [ho] at hu
  cases hu
  refine Stmt.TraceSafeList.cons_intro ?_ ?_
  · simp [Stmt.TraceSafe, Op.SafeAt.eq_def]
  intro u hu
  rw [hp] at hu
  cases hu
  have hptr : (prepared s t BM BD p).regs .ptr [BM, BD] "out_ptrs" = some (pointers s BM BD p) := by
    simp [prepared]
  have hrows : (prepared s t BM BD p).regs .nat [BM] "offs_m" = some (Tile.vec (RealSetup.rows s BM)) := by
    simpa [prepared] using hf.rows
  have hcontext : (prepared s t BM BD p).regs .nat [] "q_seq_len" = some (Tile.scalar p.context) := by
    simpa [prepared] using hf.context
  refine Stmt.TraceSafeList.cons_intro
    (store_safe bounds s _ BM BD p Bool.false hptr hrows hcontext hb) ?_
  intro u hu
  rw [store_executes s _ BM BD p Bool.false x.accumulator hptr hrows hcontext
    (by simpa [prepared] using hc.accumulator)] at hu
  cases hu
  refine Stmt.TraceSafeList.cons_intro ?_ (fun _ _ => Stmt.TraceSafeList.nil_intro)
  rw [Stmt.TraceSafe]
  simp only [evalOp.eq_def, Op.SafeAt.eq_def, Tile.scalar_data, true_and]
  by_cases htwo : 2 ≤ p.blocks
  · simp only [htwo, decide_true, ↓reduceIte]
    refine Stmt.TraceSafeList.cons_intro ?_ (fun _ _ => Stmt.TraceSafeList.nil_intro)
    exact store_safe bounds s _ BM BD p Bool.true
      (by simpa using hptr) (by simpa using hrows) (by simpa using hcontext) (hb2 htwo)
  · simp [htwo]

theorem source_safe (bounds : RegionBounds) (swapped : Bool) (s t : BlockState)
    (BM BN BD : Nat) (p : RealProgram.Parameters) (x : Inputs BM BN BD)
    (hf : RealLoop.Fixed s BM BN BD p t) (hc : RealLoop.CarryAt x p.blocks t)
    (hb : Bounds bounds s BM BD p Bool.false)
    (hb2 : 2 ≤ p.blocks → Bounds bounds s BM BD p Bool.true) :
    Stmt.TraceSafeList bounds (RealProgram.realPart (RealProgram.parts swapped BM BN BD p).after) t := by
  rw [matches_source]
  exact code_safe bounds s t BM BN BD p x hf hc hb hb2

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealStores
