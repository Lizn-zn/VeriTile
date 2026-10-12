import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealHeaderSource

/-! Execute the complete source prologue from the initial state. Program IDs,
grouped-head offsets, pointers, carry, scale, Q and signed window bounds are
derived by execution. No loop-entry register or output equation is a premise.
Memory validity remains part of the later whole-kernel IO contract. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealPrologue
open VeriTile.Triton RealScores
noncomputable section

structure Parameters where
  heads : Nat
  kvHeads : Nat
  strides : RealHeader.Strides
  layout : RealPointers.Layout
  scale : ℝ
  init : Nat
  evenM : Nat
  evenN : Nat
  window : Nat
  complement : Nat
  offset : Int
  size : Int

def code (BM BN BD : Nat) (p : Parameters) : List Stmt :=
  RealHeader.code p.heads p.kvHeads p.strides ++ RealPointers.code BM BN BD p.layout ++
    [RealInit.carryCode BM BD p.init] ++ RealInit.scaleCode p.scale ++
    [RealInit.queryCode BM BD p.evenM,
      RealWindow.code BM BN BD p.layout.kvContext p.window p.complement p.offset p.size]

def loopParameters (BM BN BD : Nat) (p : Parameters) (s : BlockState) : RealLoop.Parameters BM BN BD :=
  let c := RealHeader.coordinates s p.heads p.kvHeads
  let b := RealHeader.bases c p.strides
  { q := RealAccumulation.values s (RealPointers.queryPointer BM BD p.layout c b) (RealAccumulation.checks p.evenM),
    scale := p.scale * 1.4426950408889634, startM := c.startM, context := p.layout.kvContext,
    window := p.window, complement := p.complement, even := p.evenN, offset := p.offset, size := p.size,
    lo := RealKeys.lower c.startM BM BN p.window p.complement p.offset p.size,
    hi := RealWindow.upper c.startM BM BN p.layout.kvContext p.window p.complement p.offset,
    K := p.layout.K, V := p.layout.V, kBase := b.k, vBase := b.v,
    strideKD := p.layout.kDim, strideKN := p.layout.kRow, strideVN := p.layout.vRow, strideVD := p.layout.vDim }

def seed (BM BD : Nat) (p : Parameters) (s : BlockState) : RealLoopFormula.Carry BM BD :=
  let c := RealHeader.coordinates s p.heads p.kvHeads
  let b := RealHeader.bases c p.strides
  RealInit.seed s p.init
    (RealPointers.rowPointer BM p.layout.roundedContext p.layout.M c)
    (RealPointers.rowPointer BM p.layout.roundedContext p.layout.L c)
    (RealPointers.outputPointer BM BD p.layout c b)

def writes : List RegName :=
  RealHeader.writes ++ RealPointers.writes ++ ["m_i", "l_i", "acc", "qk_scale", "q"] ++ RealWindow.writes

/-- Output addresses are computed by the prologue and survive the loop. The
initial state supplies program IDs, not assumed pointer-register contents. -/
structure OutputRegisters (BM BD : Nat) (p : Parameters) (initial current : BlockState) : Prop where
  center : current.regs .ptr [BM] "m_ptrs" = some
    (RealPointers.rowPointer BM p.layout.roundedContext p.layout.M
      (RealHeader.coordinates initial p.heads p.kvHeads))
  denominator : current.regs .ptr [BM] "l_ptrs" = some
    (RealPointers.rowPointer BM p.layout.roundedContext p.layout.L
      (RealHeader.coordinates initial p.heads p.kvHeads))
  output : current.regs .blockPtr [BM, BD] "O_block_ptr" = some
    (RealPointers.outputPointer BM BD p.layout (RealHeader.coordinates initial p.heads p.kvHeads)
      (RealHeader.bases (RealHeader.coordinates initial p.heads p.kvHeads) p.strides))

theorem seed_same_memory {BM BD : Nat} (s t : BlockState) (init : Nat)
    (mp lp : Tile .ptr [BM]) (op : Tile .blockPtr [BM, BD]) (hmem : s.mem = t.mem) :
    RealInit.seed s init mp lp op = RealInit.seed t init mp lp op := by
  simp only [RealInit.seed, RealInit.pointerValues, RealAccumulation.values, BlockState.readMem, hmem]

theorem executes (s : BlockState) (BM BN BD : Nat) (p : Parameters) :
    ∃ s', stepStmts (code BM BN BD p) s = some s' ∧
      RealLoop.Registers (loopParameters BM BN BD p s) (seed BM BD p s) 0 s' ∧
      s'.regs .int [] "lo" = some (Tile.scalar ((loopParameters BM BN BD p s).lo : Int)) ∧
      s'.regs .int [] "hi" = some (Tile.scalar (loopParameters BM BN BD p s).hi) ∧
      Frame writes s s' ∧ OutputRegisters BM BD p s s' := by
  let c := RealHeader.coordinates s p.heads p.kvHeads
  let b := RealHeader.bases c p.strides
  let qp := RealPointers.queryPointer BM BD p.layout c b
  let mp := RealPointers.rowPointer BM p.layout.roundedContext p.layout.M c
  let lp := RealPointers.rowPointer BM p.layout.roundedContext p.layout.L c
  let op := RealPointers.outputPointer BM BD p.layout c b
  obtain ⟨sh, eh, hc, hb, fh⟩ := RealHeader.executes s p.heads p.kvHeads p.strides
  obtain ⟨sp, ep, hp, fp⟩ := RealPointers.executes sh BM BN BD p.layout c b hc hb
  obtain ⟨sc, ec, hmi, hli, hac, fc⟩ := RealInit.carry_executes sp BM BD p.init mp lp op hp.center hp.denominator hp.output
  have hseed := seed_same_memory sp s p.init mp lp op (fp.mem.trans fh.mem)
  rw [hseed] at hmi hli hac
  obtain ⟨ss, es, hscale, fs⟩ := RealInit.scale_executes sc p.scale
  obtain ⟨sq, eq, hq, fq⟩ := RealInit.query_executes ss BM BD p.evenM qp
    ((fs.regs .blockPtr [BM, BD] "Q_block_ptr" (by simp)).trans
      ((fc.regs .blockPtr [BM, BD] "Q_block_ptr" (by simp)).trans hp.query))
  have hmem := fs.mem.trans (fc.mem.trans (fp.mem.trans fh.mem))
  rw [RealIteration.values_of_same_memory ss s qp (RealAccumulation.checks p.evenM) hmem] at hq
  have hm : sq.regs .nat [] "start_m" = some (Tile.scalar c.startM) :=
    (fq.regs _ _ _ (by simp)).trans ((fs.regs _ _ _ (by simp)).trans
      ((fc.regs _ _ _ (by simp)).trans ((fp.regs _ _ _ (by simp [RealPointers.writes])).trans hc.startM)))
  have hk : sq.regs .blockPtr [BD, BN] "K_block_ptr" = some (RealPointers.keyPointer BN BD p.layout b) :=
    (fq.regs _ _ _ (by simp)).trans ((fs.regs _ _ _ (by simp)).trans ((fc.regs _ _ _ (by simp)).trans hp.key))
  have hv : sq.regs .blockPtr [BN, BD] "V_block_ptr" = some (RealPointers.valuePointer BN BD p.layout b) :=
    (fq.regs _ _ _ (by simp)).trans ((fs.regs _ _ _ (by simp)).trans ((fc.regs _ _ _ (by simp)).trans hp.value))
  obtain ⟨sf, ew, hlo, hhi, hkf, hvf, fw⟩ := RealWindow.enters_loop sq (loopParameters BM BN BD p s) rfl rfl hm
    (by simpa [RealLoop.keyPointer, RealLoop.cursor, loopParameters, RealPointers.keyPointer, b, c] using hk)
    (by simpa [RealLoop.valuePointer, RealLoop.cursor, loopParameters, RealPointers.valuePointer, b, c] using hv)
  refine ⟨sf, ?_, ?_, hlo, hhi, ?_, ?_⟩
  · simp only [code, List.append_assoc, List.singleton_append]
    erw [stepStmts.append_some eh, stepStmts.append_some ep, stepStmts.cons_some ec,
      stepStmts.append_some es, stepStmts.cons_some eq, stepStmts.cons_some ew, stepStmts.nil]
  · refine ⟨?_, ?_, (fw.regs _ _ _ (by simp [RealWindow.writes])).trans hm, ?_, ?_, ?_, hkf, hvf⟩
    · exact (fw.regs _ _ _ (by simp [RealWindow.writes])).trans hq
    · exact (fw.regs _ _ _ (by simp [RealWindow.writes])).trans ((fq.regs _ _ _ (by simp)).trans hscale)
    · exact (fw.regs _ _ _ (by simp [RealWindow.writes])).trans
        ((fq.regs _ _ _ (by simp)).trans ((fs.regs _ _ _ (by simp)).trans hmi))
    · exact (fw.regs _ _ _ (by simp [RealWindow.writes])).trans
        ((fq.regs _ _ _ (by simp)).trans ((fs.regs _ _ _ (by simp)).trans hli))
    · exact (fw.regs _ _ _ (by simp [RealWindow.writes])).trans
        ((fq.regs _ _ _ (by simp)).trans ((fs.regs _ _ _ (by simp)).trans hac))
  · exact (((((fh.mono (by simp [writes])).trans (fp.mono (by simp [writes]))).trans
      (fc.mono (by simp [writes]))).trans (fs.mono (by simp [writes]))).trans
      (fq.mono (by simp [writes]))).trans (fw.mono (by simp [writes, RealWindow.writes]))
  · constructor
    · exact (fw.regs _ _ _ (by simp [RealWindow.writes])).trans
        ((fq.regs _ _ _ (by simp)).trans ((fs.regs _ _ _ (by simp)).trans
          ((fc.regs _ _ _ (by simp)).trans hp.center)))
    · exact (fw.regs _ _ _ (by simp [RealWindow.writes])).trans
        ((fq.regs _ _ _ (by simp)).trans ((fs.regs _ _ _ (by simp)).trans
          ((fc.regs _ _ _ (by simp)).trans hp.denominator)))
    · exact (fw.regs _ _ _ (by simp [RealWindow.writes])).trans
        ((fq.regs _ _ _ (by simp)).trans ((fs.regs _ _ _ (by simp)).trans
          ((fc.regs _ _ _ (by simp)).trans hp.output)))

theorem result_same_memory {BM BN BD : Nat} (s t : BlockState) (p : RealLoop.Parameters BM BN BD)
    (initial : RealLoopFormula.Carry BM BD) (hmem : s.mem = t.mem) :
    RealLoopContext.fullResult s p initial = RealLoopContext.fullResult t p initial := by
  simp only [RealLoopContext.fullResult, RealLoopContext.fullRow, RealLoopContext.fullScore,
    RealLoopContext.keyInput, RealLoopContext.valueInput, BlockState.readMem, hmem]

/-- The actual prologue followed by the complete loop establishes independent
full-context row formulas from the initial memory, without entry-register
premises. END normalization and stores are subsequent source statements. -/
theorem through_loop_executes (s : BlockState) (BM BN BD : Nat) (p : Parameters)
    (hN : 0 < BN) (hflag : p.evenN ≠ 0 → p.layout.kvContext % BN = 0) :
    let lp := loopParameters BM BN BD p s
    ∃ s', stepStmts (code BM BN BD p ++
        [RealLoopSource.statement BM BN BD p.layout.kvContext p.window p.complement p.evenN p.offset p.size]) s = some s' ∧
      RealLoop.Registers lp (RealLoopContext.fullResult s lp (seed BM BD p s)) (RealLoop.count lp) s' ∧
      Frame (writes ++ RealLoop.writes) s s' ∧ OutputRegisters BM BD p s s' := by
  dsimp only
  obtain ⟨sp, ep, hp, hl, hh, fp, ho⟩ := executes s BM BN BD p
  obtain ⟨sf, ef, hf, ff⟩ := RealLoopSource.statement_executes sp (loopParameters BM BN BD p s)
    (seed BM BD p s) hN hflag rfl
    (RealWindow.upper_matches _ _ _ _ _ _ _) hl hh hp
  rw [result_same_memory sp s _ _ fp.mem] at hf
  refine ⟨sf, ?_, hf, (fp.mono (by simp)).trans (ff.mono (by simp)), ?_⟩
  · erw [stepStmts.append_some ep, stepStmts.cons_some ef, stepStmts.nil]
  · exact ⟨(ff.regs _ _ _ (by simp [RealLoop.writes])).trans ho.center,
      (ff.regs _ _ _ (by simp [RealLoop.writes])).trans ho.denominator,
      (ff.regs _ _ _ (by simp [RealLoop.writes])).trans ho.output⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealPrologue
