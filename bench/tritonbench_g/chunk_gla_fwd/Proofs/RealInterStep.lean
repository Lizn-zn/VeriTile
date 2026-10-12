import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealInterAlgebra

/-! One complete K iteration, including all pointers, masks, loads and gates.
The result preserves the complete tagged memory, launch and padding oracle.
No input/output separation is needed before the single final store. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealInterStep
open VeriTile.Triton RealInterMath RealInterProgram RealInterPointers RealInterLoads RealInterAlgebra
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def indexed (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (scale : ℝ) (kb : Nat) (s : BlockState) : BlockState :=
  ((s).setReg "o_k" .nat [p.BK] (keyIndices p kb)).setReg "m_k" .bool [p.BK] (keyMask p kb)

def pointed (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (scale : ℝ) (kb : Nat) (s : BlockState) : BlockState :=
  (((((indexed p l r a scale kb s).setReg "p_q" .blockPtr [p.BC,p.BK] (queryPointer p l r.query kb)).setReg "p_g" .blockPtr [p.BC,p.BK] (queryPointer p l r.gate kb)).setReg "p_k" .blockPtr [p.BK,p.BC] (keyPointer p l r.key kb)).setReg "p_gk" .blockPtr [p.BK,p.BC] (keyPointer p l r.gate kb)).setReg "p_gn" .ptr [p.BK] (referencePointer p l r.gate kb)

def queryLoaded (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (scale : ℝ) (kb : Nat) (s : BlockState) : BlockState :=
  ((((pointed p l r a scale kb s).setReg "b_gn" .real [p.BK] (referenceTile p l a kb)).setReg "b_q" .real [p.BC,p.BK] (queryTile p l a.query kb)).setReg "b_g" .real [p.BC,p.BK] (queryTile p l a.gate kb)).setReg "b_qg" .real [p.BC,p.BK] (gatedQuery p l a scale kb)

def keyLoaded (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (scale : ℝ) (kb : Nat) (s : BlockState) : BlockState :=
  (((queryLoaded p l r a scale kb s).setReg "b_k" .real [p.BK,p.BC] (keyTile p l a.key kb)).setReg "b_gk" .real [p.BK,p.BC] (keyTile p l a.gate kb)).setReg "b_kg" .real [p.BK,p.BC] (gatedKey p l a kb)

def result (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (scale : ℝ) (kb : Nat) (s : BlockState) : BlockState :=
  (keyLoaded p l r a scale kb s).setReg "b_A" .real [p.BC,p.BC] (accumulator p l a scale (kb+1))

theorem executes (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (scale : ℝ)
    (kb : Nat) (swapped : Bool) (s : BlockState)
    (hg : Grid p l s) (hk : s.regs .nat [] "i_k" = some (Tile.scalar kb))
    (hq : ∀ off, s.readMem r.query off = a.query off)
    (hkey : ∀ off, s.readMem r.key off = a.key off)
    (hgate : ∀ off, s.readMem r.gate off = a.gate off)
    (hu : ∀ region off, s.undef region off = 0)
    (hc : s.regs .real [p.BC,p.BC] "b_A" = some (accumulator p l a scale kb)) :
    stepStmts (iteration p r scale swapped) s = some (result p l r a scale kb s) := by
  unfold iteration
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (indices_eval p kb s hk))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (mask_eval p kb _ (by simp)))]
  change stepStmts (.assign _ _ "p_q" _ :: _) (indexed p l r a scale kb s) = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (query_eval p l r.query kb _
    (by constructor <;> simp [indexed,hg.chunk,hg.head,hg.row,hg.column]) (by simp [indexed,hk])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (query_eval p l r.gate kb _
    (by constructor <;> simp [indexed,hg.chunk,hg.head,hg.row,hg.column]) (by simp [indexed,hk])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (key_eval p l r.key kb _
    (by constructor <;> simp [indexed,hg.chunk,hg.head,hg.row,hg.column]) (by simp [indexed,hk])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (key_eval p l r.gate kb _
    (by constructor <;> simp [indexed,hg.chunk,hg.head,hg.row,hg.column]) (by simp [indexed,hk])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (reference_eval p l r.gate kb _
    (by constructor <;> simp [indexed,hg.chunk,hg.head,hg.row,hg.column]) (by simp [indexed])))]
  change stepStmts (.assign _ _ "b_gn" _ :: _) (pointed p l r a scale kb s) = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (reference_load p l r.gate a kb _
    (by simpa [pointed,indexed] using hgate) (by simp [pointed]) (by simp [pointed,indexed])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (query_load p l r.query a.query kb "p_q" _
    (by simpa [pointed,indexed] using hq) (by simpa [pointed,indexed] using hu r.query) (by simp [pointed])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (query_load p l r.gate a.gate kb "p_g" _
    (by simpa [pointed,indexed] using hgate) (by simpa [pointed,indexed] using hu r.gate) (by simp [pointed])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (query_gate p l a scale kb _
    (by simp) (by simp) (by simp)))]
  change stepStmts (.assign _ _ "b_k" _ :: _) (queryLoaded p l r a scale kb s) = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (key_load p l r.key a.key kb "p_k" _
    (by simpa [queryLoaded,pointed,indexed] using hkey)
    (by simpa [queryLoaded,pointed,indexed] using hu r.key) (by simp [queryLoaded,pointed])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (key_load p l r.gate a.gate kb "p_gk" _
    (by simpa [queryLoaded,pointed,indexed] using hgate)
    (by simpa [queryLoaded,pointed,indexed] using hu r.gate) (by simp [queryLoaded,pointed])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (key_gate p l a kb _
    (by simp) (by simp) (by simp [queryLoaded])))]
  change stepStmts [.assign _ _ "b_A" (updateOp p swapped)] (keyLoaded p l r a scale kb s) = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (update_eval p l a scale kb swapped _
    (by simp [keyLoaded,queryLoaded,pointed,indexed,hc])
    (by simp [keyLoaded,queryLoaded]) (by simp [keyLoaded]))),stepStmts.nil]
  rfl

theorem result_grid (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (scale : ℝ)
    (kb : Nat) (s : BlockState) (hg : Grid p l s) : Grid p l (result p l r a scale kb s) := by
  constructor <;> simp [result,keyLoaded,queryLoaded,pointed,indexed,hg.chunk,hg.head,hg.row,hg.column]

theorem result_memory (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (scale : ℝ)
    (kb : Nat) (s : BlockState) : (result p l r a scale kb s).mem = s.mem := rfl

theorem result_padding (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (scale : ℝ)
    (kb : Nat) (s : BlockState) : (result p l r a scale kb s).undef = s.undef := rfl

theorem result_carry (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (scale : ℝ)
    (kb : Nat) (s : BlockState) :
    (result p l r a scale kb s).regs .real [p.BC,p.BC] "b_A" = some (accumulator p l a scale (kb+1)) := by
  simp [result]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealInterStep
