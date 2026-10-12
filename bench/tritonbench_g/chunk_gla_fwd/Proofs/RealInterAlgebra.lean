import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealInterLoads

/-! Evaluation of the original real gate, dot and carry expressions. Common
reference cancellation belongs only to this real proof, after the actual
loads. Both addition orders implement the same independent partial sum. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealInterAlgebra
open VeriTile.Triton RealInterMath RealInterProgram RealInterLoads
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def gatedQuery (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (kb : Nat) : Tile .real [p.BC,p.BK] :=
  Tile.mat (fun i e => some (queryFactor p l a scale i.val (kb*p.BK+e.val)))
def gatedKey (p : Layout) (l : Launch) (a : Inputs) (kb : Nat) : Tile .real [p.BK,p.BC] :=
  Tile.mat (fun e j => some (keyFactor p l a j.val (kb*p.BK+e.val)))
def product (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (kb : Nat) : Tile .real [p.BC,p.BC] :=
  Tile.mat (fun i j => some (∑ e : Fin p.BK,
    queryFactor p l a scale i.val (kb*p.BK+e.val)*keyFactor p l a j.val (kb*p.BK+e.val)))
def accumulator (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (n : Nat) : Tile .real [p.BC,p.BC] :=
  Tile.mat (fun i j => some (partialSum p l a scale n i.val j.val))
def updateOp (p : Layout) (swapped : Bool) : Op .real [p.BC,p.BC] :=
  if swapped then .add .real (.consSame (.consSame .nil)) (dot p) (carry p)
  else .add .real (.consSame (.consSame .nil)) (carry p) (dot p)

theorem query_gate (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (kb : Nat) (u : BlockState)
    (hq : u.regs .real [p.BC,p.BK] "b_q" = some (queryTile p l a.query kb))
    (hg : u.regs .real [p.BC,p.BK] "b_g" = some (queryTile p l a.gate kb))
    (hr : u.regs .real [p.BK] "b_gn" = some (referenceTile p l a kb)) :
    evalOp (queryGate p scale) u = some (gatedQuery p l a scale kb) := by
  simp only [queryGate,evalOp_mul,evalOp_exp,evalOp_sub,evalOp_expandDim,evalOp_ref,
    evalOp_const,hq,hg,hr,Option.bind_eq_bind,Option.bind_some]
  erw [evalOp_expandDim]
  simp only [evalOp_ref,hr,Option.bind_eq_bind,Option.bind_some]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,e,⟨⟩⟩
  simp [gatedQuery,queryFactor,queryTile,referenceTile,Tile.mat,Tile.vec,-Tile.scalar_eta,Tile.scalar,
    Tile.bop,Tile.uop,Tile.expandDim,Broadcast.leftIndex,Broadcast.rightIndex,
    TileShape.dropInsertedIndex,NumericDType.mul,NumericDType.sub,WithBot.realExp]

theorem key_gate (p : Layout) (l : Launch) (a : Inputs) (kb : Nat) (u : BlockState)
    (hk : u.regs .real [p.BK,p.BC] "b_k" = some (keyTile p l a.key kb))
    (hg : u.regs .real [p.BK,p.BC] "b_gk" = some (keyTile p l a.gate kb))
    (hr : u.regs .real [p.BK] "b_gn" = some (referenceTile p l a kb)) :
    evalOp (keyGate p) u = some (gatedKey p l a kb) := by
  simp only [keyGate,evalOp_mul,evalOp_exp,evalOp_sub,evalOp_expandDim,evalOp_ref,
    hk,hg,hr,Option.bind_eq_bind,Option.bind_some]
  erw [evalOp_expandDim]
  simp only [evalOp_ref,hr,Option.bind_eq_bind,Option.bind_some]
  apply congrArg some
  apply Tile.ext
  rintro ⟨e,j,⟨⟩⟩
  simp [gatedKey,keyFactor,keyTile,referenceTile,Tile.mat,Tile.vec,Tile.bop,Tile.uop,
    Tile.expandDim,Broadcast.leftIndex,Broadcast.rightIndex,TileShape.dropInsertedIndex,
    NumericDType.mul,NumericDType.sub,WithBot.realExp]

theorem product_eval (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (kb : Nat) (u : BlockState)
    (hq : u.regs .real [p.BC,p.BK] "b_qg" = some (gatedQuery p l a scale kb))
    (hk : u.regs .real [p.BK,p.BC] "b_kg" = some (gatedKey p l a kb)) :
    evalOp (dot p) u = some (product p l a scale kb) := by
  unfold dot
  erw [evalOp_dot]
  erw [evalOp_ref,hq,evalOp_ref,hk]
  simp only [Option.bind_eq_bind,Option.bind_some]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  rw [Tile.dot_nil_data]
  simp only [gatedQuery,gatedKey,product,Tile.mat,Option.map₂_some_some]
  exact (WithBot.coe_sum Finset.univ (fun e : Fin p.BK =>
    queryFactor p l a scale i.val (kb*p.BK+e.val)*keyFactor p l a j.val (kb*p.BK+e.val))).symm

theorem update_eval (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (kb : Nat)
    (swapped : Bool) (u : BlockState)
    (ha : u.regs .real [p.BC,p.BC] "b_A" = some (accumulator p l a scale kb))
    (hq : u.regs .real [p.BC,p.BK] "b_qg" = some (gatedQuery p l a scale kb))
    (hk : u.regs .real [p.BK,p.BC] "b_kg" = some (gatedKey p l a kb)) :
    evalOp (updateOp p swapped) u = some (accumulator p l a scale (kb+1)) := by
  cases swapped
  all_goals
    simp only [updateOp,Bool.false_eq_true,↓reduceIte,evalOp_add,carry,evalOp_ref,ha,
      product_eval p l a scale kb u hq hk,Option.bind_eq_bind,Option.bind_some]
    apply congrArg some
    apply Tile.ext
    rintro ⟨i,j,⟨⟩⟩
    simp [accumulator,product,Tile.mat,Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex,
      NumericDType.add,partial_step,add_comm]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealInterAlgebra
