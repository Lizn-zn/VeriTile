import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealSplitAlgebra

/-! Initialization and the complete pre-loop source prefix. Only query and
gate matrix tiles are cached; key/gate flat pointers retain their K stride. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealSplitSetup
open VeriTile.Triton RealSplitMath RealSplitProgram RealSplitMemory
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def inputs (r : Regions) (s : BlockState) : Inputs := ⟨s.readMem r.query,s.readMem r.key,s.readMem r.gate⟩
def initial (p : Layout) (s : BlockState) : BlockState :=
  ((((((s.setReg "i_k" .nat [] (Tile.scalar (s.pids 0))).setReg "i_tc" .nat [] (Tile.scalar (s.pids 1))).setReg
    "i_bh" .nat [] (Tile.scalar (s.pids 2))).setReg "i_t" .nat [] (Tile.scalar (s.pids 1 / p.NC))).setReg
    "i_i" .nat [] (Tile.scalar (s.pids 1 % p.NC))).setReg "i_j" .nat [] (Tile.scalar (s.pids 1 % p.NC))).setReg
    "n_bh" .nat [] (Tile.scalar (s.numPids 2))
def indexed (p : Layout) (l : Launch) (s : BlockState) : BlockState :=
  (((((s.setReg "o_i" .nat [p.BC] (indices p.BC)).setReg "o_k" .nat [p.BK] (keyIndices p l)).setReg
    "o_A" .nat [p.BC] (offsets p l)).setReg "m_k" .bool [p.BK] (keyMask p l)).setReg
    "m_A" .bool [p.BC] (rowMask p l))
def pointed (p : Layout) (l : Launch) (r : Regions) (s : BlockState) : BlockState :=
  ((((indexed p l s).setReg "p_q" .blockPtr [p.BC,p.BK] (queryPointer p l r.query)).setReg
    "p_g" .blockPtr [p.BC,p.BK] (queryPointer p l r.gate)).setReg "p_k" .ptr [p.BK] (keyPointer p l r.key 0)).setReg
    "p_gk" .ptr [p.BK] (keyPointer p l r.gate 0)
def result (p : Layout) (l : Launch) (r : Regions) (s : BlockState) : BlockState :=
  ((pointed p l r s).setReg "b_q" .real [p.BC,p.BK] (matrixTile p l (s.readMem r.query) (s.undef r.query))).setReg
    "b_g" .real [p.BC,p.BK] (matrixTile p l (s.readMem r.gate) (s.undef r.gate))

theorem initialization_executes (p : Layout) (s : BlockState) : stepStmts (initialization p) s = some (initial p s) := by
  simp only [initialization,stepStmts,stepStmt,evalOp,Option.bind_eq_bind,Option.bind_some]
  simp only [BlockState.setReg_same,BlockState.setReg_ne_name,ne_eq,String.reduceEq,not_false_eq_true,
    Option.bind_some,Option.bind_eq_bind]
  rfl

theorem initial_grid (p : Layout) (s : BlockState) : Grid (launch p s) (initial p s) := by
  constructor <;> simp [initial,launch]

theorem executes (p : Layout) (l : Launch) (r : Regions) (s : BlockState) (hg : Grid l s) :
    stepStmts (setup p r) s = some (result p l r s) := by
  unfold setup
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.arange p.BC) s = some (indices p.BC) from by rw [evalOp_arange]; rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (indices_eval p l _ (by simpa using hg.feature)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (offsets_eval p l _
    (by constructor <;> simp [hg.chunk,hg.head,hg.row,hg.column,hg.feature,hg.heads])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (key_mask_eval p l _ (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (row_mask_eval p l _
    (by constructor <;> simp [hg.chunk,hg.head,hg.row,hg.column,hg.feature,hg.heads])))]
  change stepStmts (.assign _ _ "p_q" _ :: _) (indexed p l s) = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (query_eval p l r.query _
    (by constructor <;> simp [indexed,hg.chunk,hg.head,hg.row,hg.column,hg.feature,hg.heads])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (query_eval p l r.gate _
    (by constructor <;> simp [indexed,hg.chunk,hg.head,hg.row,hg.column,hg.feature,hg.heads])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (key_eval p l r.key _
    (by constructor <;> simp [indexed,hg.chunk,hg.head,hg.row,hg.column,hg.feature,hg.heads]) (by simp [indexed])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (key_eval p l r.gate _
    (by constructor <;> simp [indexed,hg.chunk,hg.head,hg.row,hg.column,hg.feature,hg.heads]) (by simp [indexed])))]
  change stepStmts (.assign _ _ "b_q" _ :: _) (pointed p l r s) = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (matrix_load p l r.query (s.readMem r.query) (s.undef r.query) "p_q" _
    (by intro off; rfl) (by intro off; rfl) (by simp [pointed])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (matrix_load p l r.gate (s.readMem r.gate) (s.undef r.gate) "p_g" _
    (by intro off; rfl) (by intro off; rfl) (by simp [pointed]))),stepStmts.nil]
  rfl

structure Cached (p : Layout) (l : Launch) (r : Regions) (s0 s : BlockState) : Prop where
  grid : Grid l s
  rows : s.regs .nat [p.BC] "o_i" = some (indices p.BC)
  offsets : s.regs .nat [p.BC] "o_A" = some (RealSplitMemory.offsets p l)
  keyMask : s.regs .bool [p.BK] "m_k" = some (RealSplitMemory.keyMask p l)
  rowMask : s.regs .bool [p.BC] "m_A" = some (RealSplitMemory.rowMask p l)
  query : s.regs .real [p.BC,p.BK] "b_q" = some (matrixTile p l (s0.readMem r.query) (s0.undef r.query))
  gate : s.regs .real [p.BC,p.BK] "b_g" = some (matrixTile p l (s0.readMem r.gate) (s0.undef r.gate))

theorem result_cached (p : Layout) (r : Regions) (s : BlockState) :
    Cached p (launch p s) r s (result p (launch p s) r (initial p s)) := by
  constructor
  · constructor <;> simp [result,pointed,indexed,initial,launch]
  all_goals simp [result,pointed,indexed,initial,BlockState.readMem]
  all_goals rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealSplitSetup
