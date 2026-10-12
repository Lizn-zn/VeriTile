import bench.tritonbench_g.block_sparse_attn.Proofs.RealSetup
import bench.tritonbench_g.block_sparse_attn.Proofs.RealRows

/-! Execute the entire actual pre-loop source from the initial block state.
All Q/pointer/index registers, signed CSR bounds and zero carries are produced
by source statements. Array pinning and the signed-loop induction are separate
obligations; no prepared register or prefix-execution premise is used here. -/
namespace VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealPrologue
open VeriTile.Triton RealNormalizationValues
noncomputable section
set_option linter.unusedSimpArgs false

set_option maxRecDepth 22000 in
set_option maxHeartbeats 4000000 in
theorem source_split (swapped : Bool) (BM BN BD : Nat) (p : RealProgram.Parameters) :
    RealProgram.realPart (RealProgram.parts swapped BM BN BD p).before =
      RealSetup.code BM BN BD p ++ RealQuery.code BM BD p.blocks p.evenM ++
        RealRows.code p.rows p.rowStride p.layouts := by
  cases swapped
  all_goals
    simp [RealSetup.code, RealQuery.code, RealQuery.branch, RealQuery.load, RealQuery.pointer, RealQuery.mask,
      RealRows.code, RealRows.load, RealRows.pointer,
      RealProgram.realPart, RealProgram.parts, RealProgram.source, TypedSource.specialize,
      Source.intRange?, Source.prefixBefore, ComputeKernel.surfaceBody,
      ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?,
      ComputeStmt.listToAlgorithm?_map_alg, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Stmt.eraseDTypeList, Stmt.eraseDType,
      Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
      IntegralDType.eraseDType, Except.bind, bind, eraseDType_float]
    exact ⟨⟨rfl, rfl⟩, ⟨rfl, rfl⟩, ⟨rfl, rfl⟩, rfl⟩

def queried (s : BlockState) (BM BN BD : Nat) (p : RealProgram.Parameters) : BlockState :=
  RealQuery.result (RealSetup.result s BM BN BD p) (RealSetup.queryPointers s BM BD p)
    (RealSetup.rows s BM) p.context p.blocks p.evenM

def result (s : BlockState) (BM BN BD : Nat) (p : RealProgram.Parameters) : BlockState :=
  RealRows.result (queried s BM BN BD p) p.rows (RealSetup.head s p) p.layouts p.rowStride (s.pids 0)

theorem query_executes (s : BlockState) (BM BN BD : Nat) (p : RealProgram.Parameters) :
    stepStmts (RealQuery.code BM BD p.blocks p.evenM) (RealSetup.result s BM BN BD p) =
      some (queried s BM BN BD p) := by
  have h := RealSetup.outputs s BM BN BD p
  exact RealQuery.executes _ _ _ _ _ _ h.queryPointers h.queryRows h.context

theorem rows_executes (s : BlockState) (BM BN BD : Nat) (p : RealProgram.Parameters) :
    stepStmts (RealRows.code p.rows p.rowStride p.layouts) (queried s BM BN BD p) =
      some (result s BM BN BD p) := by
  have h := RealSetup.outputs s BM BN BD p
  apply RealRows.executes
  · rw [queried, RealQuery.register_frame _ _ _ _ _ _ _ _ _ (by decide) (by decide)]
    exact h.queryHead
  · rw [queried, RealQuery.register_frame _ _ _ _ _ _ _ _ _ (by decide) (by decide)]
    exact h.queryBlock

theorem source_executes (swapped : Bool) (s : BlockState) (BM BN BD : Nat) (p : RealProgram.Parameters) :
    stepStmts (RealProgram.realPart (RealProgram.parts swapped BM BN BD p).before) s =
      some (result s BM BN BD p) := by
  rw [source_split, List.append_assoc, stepStmts.append_some (RealSetup.executes s BM BN BD p),
    stepStmts.append_some (query_executes s BM BN BD p)]
  exact rows_executes s BM BN BD p

theorem queried_frame (s : BlockState) (BM BN BD : Nat) (p : RealProgram.Parameters) :
    {queried s BM BN BD p with regs := s.regs} = s := by
  have h := RealQuery.state_frame (RealSetup.result s BM BN BD p) (RealSetup.queryPointers s BM BD p)
    (RealSetup.rows s BM) p.context p.blocks p.evenM
  exact (congrArg (fun t => {t with regs := s.regs}) h).trans (RealSetup.state_frame s BM BN BD p)

theorem state_frame (s : BlockState) (BM BN BD : Nat) (p : RealProgram.Parameters) :
    {result s BM BN BD p with regs := s.regs} = s := by
  have h := RealRows.state_frame (queried s BM BN BD p) p.rows
    (RealSetup.head s p) p.layouts p.rowStride (s.pids 0)
  exact (congrArg (fun t => {t with regs := s.regs}) h).trans (queried_frame s BM BN BD p)

/-- Fixed setup registers are preserved by Q and CSR row-bound loads. -/
theorem setup_register (s : BlockState) (BM BN BD : Nat) (p : RealProgram.Parameters)
    (dtype : TileDType) (shape : TileShape) (name : RegName)
    (hn : name ∉ ["q", "q2"] ++ RealRows.writes) :
    (result s BM BN BD p).regs dtype shape name = (RealSetup.result s BM BN BD p).regs dtype shape name := by
  simp only [List.mem_append, not_or] at hn
  have hnames := hn.1
  simp only [List.mem_cons, List.not_mem_nil, or_false, not_or] at hnames
  rw [result, RealRows.register_frame _ _ _ _ _ _ _ _ _ hn.2,
    queried, RealQuery.register_frame _ _ _ _ _ _ _ _ _ hnames.1 hnames.2]

def writes : List RegName := RealSetup.writes ++ ["q", "q2"] ++ RealRows.writes

theorem register_frame (s : BlockState) (BM BN BD : Nat) (p : RealProgram.Parameters)
    (dtype : TileDType) (shape : TileShape) (name : RegName) (hn : name ∉ writes) :
    (result s BM BN BD p).regs dtype shape name = s.regs dtype shape name := by
  simp only [writes, List.mem_append, not_or] at hn
  rw [setup_register s BM BN BD p dtype shape name (by simpa only [List.mem_append, not_or] using And.intro hn.1.2 hn.2),
    RealSetup.register_frame _ _ _ _ _ _ _ _ hn.1.1]

structure Outputs (s : BlockState) (BM BN BD : Nat) (p : RealProgram.Parameters) (t : BlockState)
    : Prop extends RealSetup.Outputs s BM BN BD p t where
  query : t.regs .real [BM, BD] "q" = some (matrix
    (RealQuery.loaded s (RealSetup.queryPointers s BM BD p) (RealSetup.rows s BM) p.context p.evenM Bool.false))
  secondQuery : 2 ≤ p.blocks → t.regs .real [BM, BD] "q2" = some (matrix
    (RealQuery.loaded s (RealSetup.queryPointers s BM BD p) (RealSetup.rows s BM) p.context p.evenM Bool.true))
  layout : t.regs .nat [] "layout_h" = some (Tile.scalar (RealSetup.head s p % p.layouts))
  first : t.regs .int [] "start_l" = some (Tile.scalar
    (RealRows.bound s p.rows (RealSetup.head s p) p.layouts p.rowStride (s.pids 0) Bool.false))
  last : t.regs .int [] "end_l" = some (Tile.scalar
    (RealRows.bound s p.rows (RealSetup.head s p) p.layouts p.rowStride (s.pids 0) Bool.true))

theorem outputs (s : BlockState) (BM BN BD : Nat) (p : RealProgram.Parameters) :
    Outputs s BM BN BD p (result s BM BN BD p) := by
  have hsetup := RealSetup.outputs s BM BN BD p
  have hm := congrArg BlockState.mem (RealSetup.state_frame s BM BN BD p)
  have hu := congrArg BlockState.undef (RealSetup.state_frame s BM BN BD p)
  change (RealSetup.result s BM BN BD p).mem = s.mem at hm
  change (RealSetup.result s BM BN BD p).undef = s.undef at hu
  have hloaded (second : Bool) :
      RealQuery.loaded (RealSetup.result s BM BN BD p) (RealSetup.queryPointers s BM BD p)
        (RealSetup.rows s BM) p.context p.evenM second =
      RealQuery.loaded s (RealSetup.queryPointers s BM BD p) (RealSetup.rows s BM) p.context p.evenM second := by
    funext r d
    simp only [RealQuery.loaded, BlockState.readMem, hm, hu]
  have hquery := RealQuery.readbacks (RealSetup.result s BM BN BD p) (RealSetup.queryPointers s BM BD p)
    (RealSetup.rows s BM) p.context p.blocks p.evenM
  have hrows := RealRows.readbacks (queried s BM BN BD p) p.rows (RealSetup.head s p) p.layouts p.rowStride (s.pids 0)
  have hmemory := congrArg BlockState.mem (queried_frame s BM BN BD p)
  change (queried s BM BN BD p).mem = s.mem at hmemory
  have hbound (next : Bool) : RealRows.bound (queried s BM BN BD p) p.rows (RealSetup.head s p)
      p.layouts p.rowStride (s.pids 0) next =
      RealRows.bound s p.rows (RealSetup.head s p) p.layouts p.rowStride (s.pids 0) next := by
    simp only [RealRows.bound, BlockState.readMemValue, BlockState.readMemTyped, hmemory]
  constructor
  · rcases hsetup with ⟨hctx, hblock, hhead, hqp, hkp, hvp, hm, hn, hmax, hl, ha, ha2⟩
    constructor
    all_goals
      rw [setup_register s BM BN BD p _ _ _ (by simp [RealRows.writes])]
      assumption
  · rw [result, RealRows.register_frame _ _ _ _ _ _ _ _ _ (by simp [RealRows.writes])]
    simpa only [hloaded] using hquery.1
  · intro hb
    rw [result, RealRows.register_frame _ _ _ _ _ _ _ _ _ (by simp [RealRows.writes])]
    simpa only [hloaded] using hquery.2 hb
  · exact hrows.1
  · simpa only [hbound] using hrows.2.1
  · simpa only [hbound] using hrows.2.2

/-- Only actually active Q lanes and the two unmasked CSR row cells need
allocation bounds. The optional Q2 bound is required only for that branch. -/
theorem source_safe (bounds : RegionBounds) (swapped : Bool) (s : BlockState)
    (BM BN BD : Nat) (p : RealProgram.Parameters)
    (hq : RealQuery.Bounds bounds (RealSetup.queryPointers s BM BD p) (RealSetup.rows s BM)
      p.context p.evenM Bool.false)
    (hq2 : 2 ≤ p.blocks → RealQuery.Bounds bounds (RealSetup.queryPointers s BM BD p)
      (RealSetup.rows s BM) p.context p.evenM Bool.true)
    (hr : RealRows.address (RealSetup.head s p) p.layouts p.rowStride (s.pids 0) + 1 < bounds (Region.cast p.rows)) :
    Stmt.TraceSafeList bounds (RealProgram.realPart (RealProgram.parts swapped BM BN BD p).before) s := by
  rw [source_split, List.append_assoc]
  apply TraceProofs.append_safe _ s (RealSetup.code_safe bounds s BM BN BD p)
  intro t ht
  rw [RealSetup.executes] at ht
  cases ht
  have h := RealSetup.outputs s BM BN BD p
  apply TraceProofs.append_safe _ _ (RealQuery.code_safe bounds _ _ _ _ _ _ h.queryPointers h.queryRows h.context hq hq2)
  intro u hu
  rw [query_executes] at hu
  cases hu
  apply RealRows.code_safe (head := RealSetup.head s p) (queryBlock := s.pids 0)
  · rw [queried, RealQuery.register_frame _ _ _ _ _ _ _ _ _ (by decide) (by decide)]
    exact h.queryHead
  · rw [queried, RealQuery.register_frame _ _ _ _ _ _ _ _ _ (by decide) (by decide)]
    exact h.queryBlock
  · exact hr

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealPrologue
