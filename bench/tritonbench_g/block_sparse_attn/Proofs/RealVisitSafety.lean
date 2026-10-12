import bench.tritonbench_g.block_sparse_attn.Proofs.RealLoop

/-! Safety of one complete source visit, following the actual score,
normalization and accumulation executions. No body-safety premise is assumed. -/
namespace VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealVisitSafety
open VeriTile.Triton RealNormalizationValues
noncomputable section
set_option linter.unusedSimpArgs false

theorem normalization_safe (bounds : RegionBounds) (s : BlockState) (BM BN BD blocks : Nat) :
    Stmt.TraceSafeList bounds (RealNormalizationSource.code BM BN BD blocks) s := by
  apply TraceProofs.all_safe
  intro st hst t
  simp only [RealNormalizationSource.code, List.mem_cons, List.not_mem_nil, or_false] at hst
  rcases hst with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  all_goals simp [Stmt.TraceSafe, Stmt.TraceSafeList, Op.SafeAt.eq_def, evalOp.eq_def]
  intro _
  split <;> trivial

theorem load_dot_safe {BM BN BD : Nat} (bounds : RegionBounds) (s : BlockState)
    (ptrs : Tile .ptr [BN, BD]) (start : Int) (stride context : Nat) (even second : Bool)
    (hp : s.regs .ptr [BN, BD] "v_ptrs" = some ptrs)
    (hs : s.regs .int [] "start_n" = some (Tile.scalar start))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec Fin.val))
    (hb : RealValueSafety.Bounds bounds ptrs start stride context even second) :
    Stmt.TraceSafeList bounds (RealAccumulation.loadDot BM BN BD stride context even second) s := by
  apply TraceProofs.append_safe _ s (RealValueSafety.code_safe bounds s ptrs start stride context even second hp hs hn hb)
  intro t _
  refine Stmt.TraceSafeList.cons_intro ?_ (fun _ _ => Stmt.TraceSafeList.nil_intro)
  simp [Stmt.TraceSafe, RealAccumulationSource.valueDot, Op.SafeAt.eq_def]

theorem accumulation_safe {BM BN BD : Nat} (bounds : RegionBounds) (s : BlockState) (x : Inputs BM BN BD)
    (ptrs : Tile .ptr [BN, BD]) (start : Int) (stride context blocks : Nat) (even : Bool)
    (hp : s.regs .ptr [BN, BD] "v_ptrs" = some ptrs)
    (hs : s.regs .int [] "start_n" = some (Tile.scalar start))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec Fin.val))
    (hx : RealNormalization.Outputs x blocks s)
    (hb : RealValueSafety.Bounds bounds ptrs start stride context even Bool.false)
    (hb2 : 2 ≤ blocks → RealValueSafety.Bounds bounds ptrs start stride context even Bool.true) :
    Stmt.TraceSafeList bounds (RealAccumulation.code BM BN BD stride context blocks even) s := by
  unfold RealAccumulation.code
  apply TraceProofs.append_safe _ s
  · apply TraceProofs.append_safe _ s (load_dot_safe bounds s ptrs start stride context even Bool.false hp hs hn hb)
    intro t ht
    rw [RealAccumulation.load_dot_executes s ptrs start stride context even Bool.false
      x.scaledProbability x.scaledAccumulator hp hs hn hx.probability hx.accumulator] at ht
    cases ht
    refine Stmt.TraceSafeList.cons_intro ?_ (fun _ _ => Stmt.TraceSafeList.nil_intro)
    rw [Stmt.TraceSafe]
    simp only [evalOp.eq_def, Op.SafeAt.eq_def, Tile.scalar_data, true_and]
    by_cases htwo : 2 ≤ blocks
    · simp only [htwo, decide_true, ↓reduceIte]
      exact load_dot_safe bounds _ ptrs start stride context even Bool.true
        (by simpa [RealAccumulation.dotResult] using hp)
        (by simpa [RealAccumulation.dotResult] using hs)
        (by simpa [RealAccumulation.dotResult] using hn) (hb2 htwo)
    · simp [htwo]
  · intro t _
    apply TraceProofs.all_safe
    intro st hst u
    simp only [RealAccumulationSource.commit, List.mem_cons, List.not_mem_nil, or_false] at hst
    rcases hst with rfl | rfl <;> simp [Stmt.TraceSafe, Op.SafeAt.eq_def]

theorem tail_safe {BM BN BD : Nat} (bounds : RegionBounds) (hN : 0 < BN)
    (s : BlockState) (x : Inputs BM BN BD) (ptrs : Tile .ptr [BN, BD])
    (start : Int) (stride context blocks : Nat) (even : Bool)
    (hp : s.regs .ptr [BN, BD] "v_ptrs" = some ptrs)
    (hs : s.regs .int [] "start_n" = some (Tile.scalar start))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec Fin.val))
    (hq : s.regs .real [BM, BN] "qk" = some (extendedMatrix x.score))
    (hm : s.regs .real [BM] "m_i" = some (extendedRow x.maximum))
    (hl : s.regs .real [BM] "l_i" = some (row x.normalizer))
    (ha : s.regs .real [BM, BD] "acc" = some (matrix x.accumulator))
    (ha2 : 2 ≤ blocks → s.regs .real [BM, BD] "acc2" = some (matrix x.secondAccumulator))
    (hb : RealValueSafety.Bounds bounds ptrs start stride context even Bool.false)
    (hb2 : 2 ≤ blocks → RealValueSafety.Bounds bounds ptrs start stride context even Bool.true) :
    Stmt.TraceSafeList bounds (RealNormalizationSource.code BM BN BD blocks ++
      RealAccumulation.code BM BN BD stride context blocks even) s := by
  apply TraceProofs.append_safe _ s (normalization_safe bounds s BM BN BD blocks)
  intro t ht
  rw [RealNormalization.executes hN s x blocks hq hm hl ha ha2] at ht
  cases ht
  apply accumulation_safe bounds _ x ptrs start stride context blocks even
  · rw [RealNormalization.register_frame _ _ _ _ _ _ (by simp [RealNormalization.writes])]
    exact hp
  · rw [RealNormalization.register_frame _ _ _ _ _ _ (by simp [RealNormalization.writes])]
    exact hs
  · rw [RealNormalization.register_frame _ _ _ _ _ _ (by simp [RealNormalization.writes])]
    exact hn
  · exact RealNormalization.outputs s x blocks
  · exact hb
  · exact hb2

theorem source_safe {BM BN BD : Nat} (bounds : RegionBounds) (swapped : Bool) (hN : 0 < BN)
    (s : BlockState) (o : RealVisit.Operands BM BN BD) (p : RealProgram.Parameters) (x : Inputs BM BN BD)
    (hlayout : s.regs .nat [] "layout_h" = some (Tile.scalar o.layout))
    (hvisit : s.regs .int [] "col_idx_idx" = some (Tile.scalar o.visit))
    (hk : s.regs .ptr [BD, BN] "k_ptrs" = some o.keyPointers)
    (hv : s.regs .ptr [BN, BD] "v_ptrs" = some o.valuePointers)
    (hm : s.regs .nat [BM] "offs_m" = some (Tile.vec o.rows))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec Fin.val))
    (hq : s.regs .real [BM, BD] "q" = some (matrix o.query))
    (hq2 : 2 ≤ p.blocks → s.regs .real [BM, BD] "q2" = some (matrix o.secondQuery))
    (hmax : s.regs .real [BM] "m_i" = some (extendedRow x.maximum))
    (hnorm : s.regs .real [BM] "l_i" = some (row x.normalizer))
    (hacc : s.regs .real [BM, BD] "acc" = some (matrix x.accumulator))
    (hacc2 : 2 ≤ p.blocks → s.regs .real [BM, BD] "acc2" = some (matrix x.secondAccumulator))
    (hcol : RealColumn.address o.layout p.columnStride o.visit < bounds (Region.cast p.cols))
    (hkey : RealKey.Bounds bounds o.keyPointers (RealVisit.start s o p) p.kRow p.context p.evenN Bool.false)
    (hkey2 : 2 ≤ p.blocks → RealKey.Bounds bounds o.keyPointers (RealVisit.start s o p) p.kRow p.context p.evenN Bool.true)
    (hval : RealValueSafety.Bounds bounds o.valuePointers (RealVisit.start s o p) p.vRow p.context p.evenN Bool.false)
    (hval2 : 2 ≤ p.blocks → RealValueSafety.Bounds bounds o.valuePointers (RealVisit.start s o p) p.vRow p.context p.evenN Bool.true) :
    Stmt.TraceSafeList bounds (RealProgram.realPart (RealProgram.parts swapped BM BN BD p).body) s := by
  rw [RealProgram.loop_split, List.append_assoc]
  apply TraceProofs.append_safe _ s
    (RealScore.source_safe bounds swapped BM BN BD p s o.layout o.visit o.keyPointers o.query
      hlayout hvisit hk hn hq hcol hkey hkey2)
  intro t ht
  rw [RealScore.source_executes swapped BM BN BD p s o.layout o.visit o.keyPointers
    o.rows o.query o.secondQuery hlayout hvisit hk hm hn hq hq2] at ht
  cases ht
  rw [RealNormalizationSource.matches_source, RealAccumulation.matches_source]
  change Stmt.TraceSafeList bounds (RealNormalizationSource.code BM BN BD p.blocks ++
    RealAccumulation.code BM BN BD p.vRow p.context p.blocks p.evenN) (RealVisit.scored s o p)
  apply tail_safe bounds hN (RealVisit.scored s o p) (RealVisit.inputs s o p x)
    o.valuePointers (RealVisit.start s o p) p.vRow p.context p.blocks p.evenN
  · rw [RealVisit.scored_register_frame _ _ _ _ _ _ (by simp)]
    exact hv
  · exact RealVisit.start_readback s o p
  · rw [RealVisit.scored_register_frame _ _ _ _ _ _ (by simp)]
    exact hn
  · exact RealVisit.score_readback s o p x
  · rw [RealVisit.scored_register_frame _ _ _ _ _ _ (by simp)]
    exact hmax
  · rw [RealVisit.scored_register_frame _ _ _ _ _ _ (by simp)]
    exact hnorm
  · rw [RealVisit.scored_register_frame _ _ _ _ _ _ (by simp)]
    exact hacc
  · intro hb
    rw [RealVisit.scored_register_frame _ _ _ _ _ _ (by simp)]
    exact hacc2 hb
  · exact hval
  · exact hval2

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealVisitSafety
