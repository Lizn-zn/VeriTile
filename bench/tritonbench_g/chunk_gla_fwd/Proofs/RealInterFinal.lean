import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealInterLoop

/-! The single final boundary-checked store. Readback uses the independent
unshifted contraction; all other tagged cells are preserved. Aliasing with any
input is allowed because every load precedes this store. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealInterFinal
open VeriTile.Triton RealInterMath RealInterProgram RealInterPointers RealInterAlgebra
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 14000
set_option linter.unusedSimpArgs false

instance active_decidable (p : Layout) (l : Launch) (i j : Nat) : Decidable (active p l i j) :=
  inferInstanceAs (Decidable (rowStart p l+i < p.T ∧ columnBlock p l*p.BC+j < p.BT))

def NoCollision (p : Layout) (l : Launch) : Prop :=
  ∀ i j : TileIndex [p.BC,p.BC], active p l i.1.val i.2.1.val → active p l j.1.val j.2.1.val →
    outputAddress p l i.1.val i.2.1.val = outputAddress p l j.1.val j.2.1.val → i = j

/-- The checked row-major output addresses are injective; no extra shape
restriction is needed. If BT=0, there are no active store lanes. -/
theorem no_collision (p : Layout) (l : Launch) : NoCollision p l := by
  intro i j hi hj he
  rcases i with ⟨i1,i2,⟨⟩⟩
  rcases j with ⟨j1,j2,⟨⟩⟩
  have hp : 0 < p.BT := Nat.zero_lt_of_lt hi.2
  have he' : (rowStart p l+i1.val)*p.BT+(columnBlock p l*p.BC+i2.val) =
      (rowStart p l+j1.val)*p.BT+(columnBlock p l*p.BC+j2.val) := by
    simpa only [outputAddress,Nat.add_assoc,Nat.add_left_cancel_iff] using he
  have hdiv := congrArg (fun n => n/p.BT) he'
  have hmod := congrArg (fun n => n%p.BT) he'
  simp only [Nat.mul_comm (rowStart p l+i1.val) p.BT,Nat.mul_comm (rowStart p l+j1.val) p.BT,Nat.mul_add_div hp,Nat.div_eq_of_lt hi.2,Nat.div_eq_of_lt hj.2,Nat.add_zero] at hdiv
  simp only [Nat.mul_add_mod_self_right,Nat.mod_eq_of_lt hi.2,Nat.mod_eq_of_lt hj.2] at hmod
  have h1 : i1=j1 := Fin.ext (Nat.add_left_cancel hdiv)
  have h2 : i2=j2 := Fin.ext (Nat.add_left_cancel hmod)
  cases h1; cases h2; rfl

def footprint (p : Layout) (l : Launch) (r : RegionName) : WriteFootprint :=
  fun cell => cell.1 = r ∧ enabled p l ∧ ∃ i : TileIndex [p.BC,p.BC],
    active p l i.1.val i.2.1.val ∧ cell.2 = outputAddress p l i.1.val i.2.1.val

def prepared (p : Layout) (l : Launch) (r : RegionName) (s : BlockState) : BlockState :=
  s.setReg "p_A" .blockPtr [p.BC,p.BC] (outputPointer p l r)
def scatter (p : Layout) (l : Launch) (r : RegionName) (a : Inputs) (scale : ℝ) (s : BlockState) : BlockState :=
  (TileShape.allIndices [p.BC,p.BC]).foldl (fun u i =>
    if active p l i.1.val i.2.1.val then
      u.writeMem r (outputAddress p l i.1.val i.2.1.val) (expected p l a scale i.1.val i.2.1.val) else u) s

theorem store_eval (p : Layout) (l : Launch) (r : RegionName) (a : Inputs) (scale : ℝ) (s : BlockState)
    (hp : s.regs .blockPtr [p.BC,p.BC] "p_A" = some (outputPointer p l r))
    (hc : s.regs .real [p.BC,p.BC] "b_A" = some (accumulator p l a scale (numBlocks p))) :
    stepStmt (.store .real [p.BC,p.BC] (.blockPtr (.ref .blockPtr [p.BC,p.BC] "p_A") [0,1]) (carry p) .none) s =
      some (scatter p l r a scale s) := by
  simp only [stepStmt,carry,evalOp_ref,hp,hc,Option.bind_some,Option.bind_eq_bind,
    Bool.true_and,outputPointer,ChunkGatedAttention.RealHPointers.matrixPointer,
    TileShape.indexToList,BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,
    decide_eq_true_eq,Nat.mul_one]
  rfl

theorem executes (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (scale : ℝ) (s : BlockState)
    (hg : Grid p l s) (hc : s.regs .real [p.BC,p.BC] "b_A" = some (accumulator p l a scale (numBlocks p))) :
    stepStmts (finalization p r) s = some (scatter p l r.output a scale (prepared p l r.output s)) := by
  unfold finalization
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (output_eval p l r.output s hg))]
  rw [stepStmts.cons_some (store_eval p l r.output a scale _ (by simp) (by simpa using hc)),stepStmts.nil]
  rfl

theorem readback (p : Layout) (l : Launch) (r : RegionName) (a : Inputs) (scale : ℝ) (s : BlockState)
    (hinj : NoCollision p l) (i : TileIndex [p.BC,p.BC]) (hi : active p l i.1.val i.2.1.val) :
    (scatter p l r a scale s).readMem r (outputAddress p l i.1.val i.2.1.val) =
      expected p l a scale i.1.val i.2.1.val := by
  exact BlockState.scatter_readback_prop_masked_nd_of_true s _ _ _ i hi (fun j hj he => hinj j i hj hi he)

theorem frame (p : Layout) (l : Launch) (r : RegionName) (a : Inputs) (scale : ℝ) (s : BlockState)
    (hen : enabled p l) : BlockState.WriteWithin (footprint p l r) s (scatter p l r a scale s) := by
  apply BlockState.foldl_writeMem_prop_writeWithin
  intro i _ hi
  exact ⟨rfl,hen,i,hi,rfl⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealInterFinal
