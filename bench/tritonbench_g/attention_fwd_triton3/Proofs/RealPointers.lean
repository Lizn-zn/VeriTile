import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealHeader

/-! Construct all source pointers from the executed coordinate/base registers.
The metadata retains separate Q/K/V/Out strides, Q/KV/rounded context bounds,
and the flattened head/batch indexing used for M/L. Address validity and
disjoint output writes are later IO-contract obligations. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealPointers
open VeriTile.Triton RealScores RealHeader

structure Layout where
  Q : RegionName
  K : RegionName
  V : RegionName
  M : RegionName
  Out : RegionName
  L : RegionName
  queryContext : Nat
  roundedContext : Nat
  kvContext : Nat
  qRow : Nat
  qDim : Nat
  kDim : Nat
  kRow : Nat
  vRow : Nat
  vDim : Nat
  oRow : Nat
  oDim : Nat

def rowOffsetExpr (BM : Nat) : Op .nat [] := .mul .nat .nil (.ref .nat [] "start_m") (.constNat BM)

def queryExpr (BM BD : Nat) (p : Layout) : Op .blockPtr [BM, BD] :=
  .makeBlockPtrDynOffsets p.Q (.ref .nat [] "q_offset") [p.queryContext, BD] [BM, BD]
    [p.qRow, p.qDim] [rowOffsetExpr BM, .constNat 0]

def valueExpr (BN BD : Nat) (p : Layout) : Op .blockPtr [BN, BD] :=
  .makeBlockPtrDyn p.V (.ref .nat [] "v_offset") [p.kvContext, BD] [BN, BD] [p.vRow, p.vDim] [0, 0]

def keyExpr (BN BD : Nat) (p : Layout) : Op .blockPtr [BD, BN] :=
  .makeBlockPtrDyn p.K (.ref .nat [] "k_offset") [BD, p.kvContext] [BD, BN] [p.kDim, p.kRow] [0, 0]

def outputExpr (BM BD : Nat) (p : Layout) : Op .blockPtr [BM, BD] :=
  .makeBlockPtrDynOffsets p.Out (.ref .nat [] "o_offset") [p.roundedContext, BD] [BM, BD]
    [p.oRow, p.oDim] [rowOffsetExpr BM, .constNat 0]

def rowPointerExpr (BM roundedContext : Nat) (region : RegionName) : Op .ptr [BM] :=
  .ptrAdd .scalarL (.ptrBase (d := .real) region)
    (.add .nat .scalarL (.mul .nat .nil (.ref .nat [] "off_hz") (.constNat roundedContext))
      (.ref .nat [BM] "offs_m"))

/-- Pointer construction uses the source AST's real phantom tag. The fp32
M/L load/store precision is carried separately by the memory operations. -/
def code (BM BN BD : Nat) (p : Layout) : List Stmt :=
  [.assign .blockPtr [BM, BD] "Q_block_ptr" (queryExpr BM BD p),
   .assign .blockPtr [BN, BD] "V_block_ptr" (valueExpr BN BD p),
   .assign .blockPtr [BD, BN] "K_block_ptr" (keyExpr BN BD p),
   .assign .blockPtr [BM, BD] "O_block_ptr" (outputExpr BM BD p),
   .assign .nat [BM] "offs_m" (.add .nat .scalarL (rowOffsetExpr BM) (.arange BM)),
   .assign .ptr [BM] "m_ptrs" (rowPointerExpr BM p.roundedContext p.M),
   .assign .ptr [BM] "l_ptrs" (rowPointerExpr BM p.roundedContext p.L)]

def queryPointer (BM BD : Nat) (p : Layout) (c : Coordinates) (b : Bases) : Tile .blockPtr [BM, BD] :=
  ⟨fun _ =>
    { region := p.Q, baseOffset := b.q, parentShape := [p.queryContext, BD],
      blockShape := [BM, BD], strides := [p.qRow, p.qDim], offsets := [c.startM * BM, 0] }⟩

def valuePointer (BN BD : Nat) (p : Layout) (b : Bases) : Tile .blockPtr [BN, BD] :=
  ⟨fun _ =>
    { region := p.V, baseOffset := b.v, parentShape := [p.kvContext, BD],
      blockShape := [BN, BD], strides := [p.vRow, p.vDim], offsets := [0, 0] }⟩

def keyPointer (BN BD : Nat) (p : Layout) (b : Bases) : Tile .blockPtr [BD, BN] :=
  ⟨fun _ =>
    { region := p.K, baseOffset := b.k, parentShape := [BD, p.kvContext],
      blockShape := [BD, BN], strides := [p.kDim, p.kRow], offsets := [0, 0] }⟩

def outputPointer (BM BD : Nat) (p : Layout) (c : Coordinates) (b : Bases) : Tile .blockPtr [BM, BD] :=
  ⟨fun _ =>
    { region := p.Out, baseOffset := b.out, parentShape := [p.roundedContext, BD],
      blockShape := [BM, BD], strides := [p.oRow, p.oDim], offsets := [c.startM * BM, 0] }⟩

def rowOffsets (BM : Nat) (c : Coordinates) : Tile .nat [BM] :=
  ⟨fun i => c.startM * BM + i.1.val⟩

def rowPointer (BM roundedContext : Nat) (region : RegionName) (c : Coordinates) : Tile .ptr [BM] :=
  ⟨fun i => (region, c.headBatch * roundedContext + c.startM * BM + i.1.val)⟩

structure Registers (BM BN BD : Nat) (p : Layout) (c : Coordinates) (b : Bases) (s : BlockState) : Prop where
  query : s.regs .blockPtr [BM, BD] "Q_block_ptr" = some (queryPointer BM BD p c b)
  value : s.regs .blockPtr [BN, BD] "V_block_ptr" = some (valuePointer BN BD p b)
  key : s.regs .blockPtr [BD, BN] "K_block_ptr" = some (keyPointer BN BD p b)
  output : s.regs .blockPtr [BM, BD] "O_block_ptr" = some (outputPointer BM BD p c b)
  rows : s.regs .nat [BM] "offs_m" = some (rowOffsets BM c)
  center : s.regs .ptr [BM] "m_ptrs" = some (rowPointer BM p.roundedContext p.M c)
  denominator : s.regs .ptr [BM] "l_ptrs" = some (rowPointer BM p.roundedContext p.L c)

def writes : List RegName := ["Q_block_ptr", "V_block_ptr", "K_block_ptr", "O_block_ptr", "offs_m", "m_ptrs", "l_ptrs"]

theorem dynamic_rows_eval (s : BlockState) (region : RegionName) (name : RegName)
    (BM BD context strideRow strideDim base startM : Nat)
    (hb : s.regs .nat [] name = some (Tile.scalar base))
    (hm : s.regs .nat [] "start_m" = some (Tile.scalar startM)) :
    evalOp (.makeBlockPtrDynOffsets region (.ref .nat [] name) [context, BD] [BM, BD]
      [strideRow, strideDim] [rowOffsetExpr BM, .constNat 0]) s =
      some (⟨fun _ =>
        { region := region, baseOffset := base, parentShape := [context, BD],
          blockShape := [BM, BD], strides := [strideRow, strideDim], offsets := [startM * BM, 0] }⟩ : Tile .blockPtr [BM, BD]) := by
  rw [makeBlockPtr2_eval]
  have he : evalOp (rowOffsetExpr BM) s = some (Tile.scalar (startM * BM)) := by
    simp [rowOffsetExpr, evalOp.eq_def, hm]; rfl
  have hz : evalOp (.constNat 0) s = some (Tile.scalar 0) := evalOp_constNat 0 s
  simp only [evalOp_ref, hb, he, hz, List.mapM_cons, List.mapM_nil, pure, Tile.scalar]
  rfl

theorem static_pointer_eval (s : BlockState) (region : RegionName) (name : RegName)
    (shape parent strides : List Nat) (base : Nat)
    (hb : s.regs .nat [] name = some (Tile.scalar base)) :
    evalOp (.makeBlockPtrDyn region (.ref .nat [] name) parent shape strides [0, 0]) s =
      some (⟨fun _ =>
        { region := region, baseOffset := base, parentShape := parent,
          blockShape := shape, strides := strides, offsets := [0, 0] }⟩ : Tile .blockPtr shape) := by
  simp only [evalOp.eq_def, hb, bind, Option.bind_some]
  rfl

theorem row_offsets_eval (s : BlockState) (BM : Nat) (c : Coordinates)
    (hm : s.regs .nat [] "start_m" = some (Tile.scalar c.startM)) :
    evalOp (.add .nat .scalarL (rowOffsetExpr BM) (.arange BM)) s = some (rowOffsets BM c) := by
  simp only [rowOffsetExpr, evalOp.eq_def, hm, bind, Option.bind_some]
  rfl

theorem row_pointer_eval (s : BlockState) (BM roundedContext : Nat) (region : RegionName)
    (c : Coordinates) (hh : s.regs .nat [] "off_hz" = some (Tile.scalar c.headBatch))
    (ho : s.regs .nat [BM] "offs_m" = some (rowOffsets BM c)) :
    evalOp (rowPointerExpr BM roundedContext region) s = some (rowPointer BM roundedContext region c) := by
  simp only [rowPointerExpr, evalOp.eq_def, hh, ho, bind, Option.bind_some]
  congr 1
  apply Tile.ext
  intro i
  simp [Tile.bop_data, Tile.scalar, rowOffsets, rowPointer, NumericDType.mul, NumericDType.add,
    Nat.add_assoc, -Tile.scalar_eta]

theorem executes (s : BlockState) (BM BN BD : Nat) (p : Layout) (c : Coordinates) (b : Bases)
    (hc : CoordinateRegisters c s) (hb : OffsetRegisters b s) :
    ∃ s', stepStmts (code BM BN BD p) s = some s' ∧ Registers BM BN BD p c b s' ∧ Frame writes s s' := by
  let sq := s.setReg "Q_block_ptr" .blockPtr [BM, BD] (queryPointer BM BD p c b)
  let sv := sq.setReg "V_block_ptr" .blockPtr [BN, BD] (valuePointer BN BD p b)
  let sk := sv.setReg "K_block_ptr" .blockPtr [BD, BN] (keyPointer BN BD p b)
  let so := sk.setReg "O_block_ptr" .blockPtr [BM, BD] (outputPointer BM BD p c b)
  let sr := so.setReg "offs_m" .nat [BM] (rowOffsets BM c)
  let sm := sr.setReg "m_ptrs" .ptr [BM] (rowPointer BM p.roundedContext p.M c)
  have eq := dynamic_rows_eval s p.Q "q_offset" BM BD p.queryContext p.qRow p.qDim b.q c.startM hb.q hc.startM
  have ev := static_pointer_eval sq p.V "v_offset" [BN, BD] [p.kvContext, BD] [p.vRow, p.vDim] b.v
    (by simp [sq, hb.v])
  have ek := static_pointer_eval sv p.K "k_offset" [BD, BN] [BD, p.kvContext] [p.kDim, p.kRow] b.k
    (by simp [sv, sq, hb.k])
  have eo := dynamic_rows_eval sk p.Out "o_offset" BM BD p.roundedContext p.oRow p.oDim b.out c.startM
    (by simp [sk, sv, sq, hb.out]) (by simp [sk, sv, sq, hc.startM])
  have er := row_offsets_eval so BM c (by simp [so, sk, sv, sq, hc.startM])
  have em := row_pointer_eval sr BM p.roundedContext p.M c
    (by simp [sr, so, sk, sv, sq, hc.headBatch]) (by simp [sr])
  have el := row_pointer_eval sm BM p.roundedContext p.L c
    (by simp [sm, sr, so, sk, sv, sq, hc.headBatch]) (by simp [sm, sr])
  erw [code, stepStmts.cons_some (stepStmt_assign_eq_some eq), stepStmts.cons_some (stepStmt_assign_eq_some ev),
    stepStmts.cons_some (stepStmt_assign_eq_some ek), stepStmts.cons_some (stepStmt_assign_eq_some eo),
    stepStmts.cons_some (stepStmt_assign_eq_some er), stepStmts.cons_some (stepStmt_assign_eq_some em),
    stepStmts.cons_some (stepStmt_assign_eq_some el), stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_⟩
  · constructor <;> simp [sm, sr, so, sk, sv, sq]
  · exact ((((((Frame.setReg writes s "Q_block_ptr" _ _ _ (by simp [writes])).trans
      (Frame.setReg writes sq "V_block_ptr" _ _ _ (by simp [writes]))).trans
      (Frame.setReg writes sv "K_block_ptr" _ _ _ (by simp [writes]))).trans
      (Frame.setReg writes sk "O_block_ptr" _ _ _ (by simp [writes]))).trans
      (Frame.setReg writes so "offs_m" _ _ _ (by simp [writes]))).trans
      (Frame.setReg writes sr "m_ptrs" _ _ _ (by simp [writes]))).trans
      (Frame.setReg writes sm "l_ptrs" _ _ _ (by simp [writes]))

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealPointers
