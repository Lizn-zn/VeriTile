import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealIntraSetup

/-! One complete streamed column: key/gate loads, gated reduction, causal
scaling, masked store, and both pointer advances. Other tagged memory cells
and all cached registers survive the store. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealIntraStep
open VeriTile.Triton RealIntraMath RealIntraProgram RealIntraMemory RealIntraAlgebra
open RealIntraSetup (Cached)
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def loaded (p : Layout) (l : Launch) (a : Inputs) (j : Nat) (s : BlockState) : BlockState :=
  (s.setReg "b_k" .real [p.BK] (keyTile p l a.key j)).setReg "b_gk" .real [p.BK] (keyTile p l a.gate j)
def calculated (p : Layout) (l : Launch) (a : Inputs) (qp gp : Nat → ℝ)
    (scale : ℝ) (j : Nat) (s : BlockState) : BlockState :=
  ((loaded p l a j s).setReg "b_A" .real [p.BC] (reducedTile p l a qp gp j)).setReg
    "b_A" .real [p.BC] (scaledTile p l a qp gp scale j)
def scatter (p : Layout) (l : Launch) (out : RegionName) (a : Inputs) (qp gp : Nat → ℝ)
    (scale : ℝ) (j : Nat) (s : BlockState) : BlockState :=
  (TileShape.allIndices [p.BC]).foldl (fun u i =>
    if rowActive p l i.1.val then u.writeMem out (outputAddress p l i.1.val j)
      (RealIntraMath.result p l a qp gp scale i.1.val j) else u) s
def result (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (qp gp : Nat → ℝ)
    (scale : ℝ) (j : Nat) (s : BlockState) : BlockState :=
  ((scatter p l r.output a qp gp scale j (calculated p l a qp gp scale j s)).setReg
    "p_k" .ptr [p.BK] (keyPointer p l r.key (j+1))).setReg
    "p_gk" .ptr [p.BK] (keyPointer p l r.gate (j+1))

theorem store_eval (p : Layout) (l : Launch) (out : RegionName) (a : Inputs) (qp gp : Nat → ℝ)
    (scale : ℝ) (j : Nat) (s : BlockState)
    (ho : s.regs .nat [p.BC] "o_A" = some (offsets p l))
    (hj : s.regs .nat [] "j" = some (Tile.scalar j))
    (hm : s.regs .bool [p.BC] "m_A" = some (rowMask p l))
    (ha : s.regs .real [p.BC] "b_A" = some (scaledTile p l a qp gp scale j)) :
    stepStmt (.store .real [p.BC] (.region out (outputOffsets p)) (.ref .real [p.BC] "b_A")
      (.mask (.ref .bool [p.BC] "m_A"))) s = some (scatter p l out a qp gp scale j s) := by
  simp only [stepStmt,outputOffsets,evalOp_add,evalOp_ref,ho,hj,hm,ha,
    Option.bind_some,Option.bind_eq_bind]
  apply congrArg some
  simp only [scatter,offsets,rowMask,scaledTile,Tile.vec,Tile.scalar,Tile.bop,
    Broadcast.leftIndex,Broadcast.rightIndex,NumericDType.add,decide_eq_true_eq,outputAddress,Nat.add_zero]
  rfl

theorem executes (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (qp gp : Nat → ℝ)
    (scale : ℝ) (j : Nat) (swapped : Bool) (s : BlockState)
    (hi : s.regs .nat [p.BC] "o_i" = some (indices p.BC))
    (ho : s.regs .nat [p.BC] "o_A" = some (offsets p l))
    (hm : s.regs .bool [p.BC] "m_A" = some (rowMask p l))
    (hmask : s.regs .bool [p.BK] "m_k" = some (keyMask p))
    (hq : s.regs .real [p.BC,p.BK] "b_q" = some (matrixTile p l a.query qp))
    (hg : s.regs .real [p.BC,p.BK] "b_g" = some (matrixTile p l a.gate gp))
    (hj : s.regs .nat [] "j" = some (Tile.scalar j))
    (hp : s.regs .ptr [p.BK] "p_k" = some (keyPointer p l r.key j))
    (hpg : s.regs .ptr [p.BK] "p_gk" = some (keyPointer p l r.gate j))
    (hk : ∀ off, s.readMem r.key off = a.key off)
    (hgk : ∀ off, s.readMem r.gate off = a.gate off) :
    stepStmts (iteration p r scale swapped) s = some (result p l r a qp gp scale j s) := by
  unfold iteration
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (key_load p l r.key a.key j "p_k" s hk hp hmask))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (key_load p l r.gate a.gate j "p_gk" _
    (by simpa using hgk) (by simpa using hpg) (by simpa using hmask)))]
  change stepStmts (.assign _ _ "b_A" _ :: _) (loaded p l a j s) = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (reduced_eval p l a qp gp j _
    (by simp [loaded,hq]) (by simp [loaded,hg]) (by simp [loaded]) (by simp [loaded])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (scaled_eval p l a qp gp scale j swapped _
    (by simp [loaded,hi]) (by simp [loaded,hj]) (by simp)))]
  change stepStmts (.store _ _ _ _ _ :: _) (calculated p l a qp gp scale j s) = _
  rw [stepStmts.cons_some (store_eval p l r.output a qp gp scale j _
    (by simp [calculated,loaded,ho]) (by simp [calculated,loaded,hj])
    (by simp [calculated,loaded,hm]) (by simp [calculated]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (advance_eval p l r.key j "p_k" _
    (by simp [scatter,calculated,loaded,hp])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (advance_eval p l r.gate j "p_gk" _
    (by simp [scatter,calculated,loaded,hpg]))),stepStmts.nil]
  rfl

theorem cached (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (qp gp : Nat → ℝ)
    (scale : ℝ) (j : Nat) (s0 s : BlockState) (hc : Cached p l r s0 s) :
    Cached p l r s0 (result p l r a qp gp scale j s) := by
  constructor
  · constructor <;> simp [result,scatter,calculated,loaded,hc.grid.chunk,hc.grid.head,hc.grid.row,hc.grid.column]
  all_goals simp [result,scatter,calculated,loaded,hc.rows,hc.offsets,hc.keyMask,hc.rowMask,hc.query,hc.gate]

def columnFootprint (p : Layout) (l : Launch) (out : RegionName) (j : Nat) : WriteFootprint :=
  fun cell => cell.1 = out ∧ ∃ i : TileIndex [p.BC], rowActive p l i.1.val ∧ cell.2 = outputAddress p l i.1.val j

theorem frame (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (qp gp : Nat → ℝ)
    (scale : ℝ) (j : Nat) (s : BlockState) :
    BlockState.WriteWithin (columnFootprint p l r.output j) s (result p l r a qp gp scale j s) := by
  have hf : BlockState.WriteWithin (columnFootprint p l r.output j)
      (calculated p l a qp gp scale j s)
      (scatter p l r.output a qp gp scale j (calculated p l a qp gp scale j s)) := by
    apply BlockState.foldl_writeMem_prop_writeWithin
    intro i _ hi
    exact ⟨rfl,i,hi,rfl⟩
  exact hf

theorem readback (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (qp gp : Nat → ℝ)
    (scale : ℝ) (j : Nat) (s : BlockState) (hcols : columns p l ≤ p.BT) (hj : j < columns p l)
    (i : Fin p.BC) (hi : rowActive p l i.val) :
    (result p l r a qp gp scale j s).readMem r.output (outputAddress p l i.val j) =
      RealIntraMath.result p l a qp gp scale i.val j := by
  change (scatter p l r.output a qp gp scale j (calculated p l a qp gp scale j s)).readMem
    r.output (outputAddress p l i.val j) = _
  apply BlockState.scatter_readback_prop_masked_nd_of_true (shape := [p.BC])
    (calculated p l a qp gp scale j s)
    (fun ix => outputAddress p l ix.1.val j)
    (fun ix => RealIntraMath.result p l a qp gp scale ix.1.val j)
    (fun ix => rowActive p l ix.1.val) (i,PUnit.unit) hi
  rintro ⟨e,⟨⟩⟩ _ he
  have hx := (output_injective p l hcols e.val i.val j j hj hj he).1
  cases Fin.ext hx
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealIntraStep
