import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealIntraMemory

/-! The gated product, K reduction and causal scaling in real semantics.
Arbitrary padding remains in cached tiles until result_valid applies. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealIntraAlgebra
open VeriTile.Triton RealIntraMath RealIntraProgram RealIntraMemory
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

def productTile (p : Layout) (l : Launch) (a : Inputs) (qp gp : Nat → ℝ) (j : Nat) : Tile .real [p.BC,p.BK] :=
  Tile.mat (fun i e => some (lane p l a qp gp i.val j e.val))
def reducedTile (p : Layout) (l : Launch) (a : Inputs) (qp gp : Nat → ℝ) (j : Nat) : Tile .real [p.BC] :=
  Tile.vec (fun i => some (reduced p l a qp gp i.val j))
def scaledTile (p : Layout) (l : Launch) (a : Inputs) (qp gp : Nat → ℝ) (scale : ℝ) (j : Nat) : Tile .real [p.BC] :=
  Tile.vec (fun i => some (result p l a qp gp scale i.val j))

theorem product_eval (p : Layout) (l : Launch) (a : Inputs) (qp gp : Nat → ℝ) (j : Nat) (s : BlockState)
    (hq : s.regs .real [p.BC,p.BK] "b_q" = some (matrixTile p l a.query qp))
    (hg : s.regs .real [p.BC,p.BK] "b_g" = some (matrixTile p l a.gate gp))
    (hk : s.regs .real [p.BK] "b_k" = some (keyTile p l a.key j))
    (hgk : s.regs .real [p.BK] "b_gk" = some (keyTile p l a.gate j)) :
    evalOp (product p) s = some (productTile p l a qp gp j) := by
  simp only [product,evalOp_mul,evalOp_exp,evalOp_sub,evalOp_expandDim,evalOp_ref,
    hq,hg,hk,hgk,Option.bind_eq_bind,Option.bind_some]
  erw [evalOp_expandDim,evalOp_expandDim]
  simp only [evalOp_ref,hk,hgk,Option.bind_eq_bind,Option.bind_some]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,e,⟨⟩⟩
  simp [productTile,lane,matrixTile,keyTile,Tile.mat,Tile.vec,Tile.bop,Tile.uop,
    Tile.expandDim,Broadcast.leftIndex,Broadcast.rightIndex,TileShape.dropInsertedIndex,
    NumericDType.mul,NumericDType.sub,WithBot.realExp]

theorem reduced_eval (p : Layout) (l : Launch) (a : Inputs) (qp gp : Nat → ℝ) (j : Nat) (s : BlockState)
    (hq : s.regs .real [p.BC,p.BK] "b_q" = some (matrixTile p l a.query qp))
    (hg : s.regs .real [p.BC,p.BK] "b_g" = some (matrixTile p l a.gate gp))
    (hk : s.regs .real [p.BK] "b_k" = some (keyTile p l a.key j))
    (hgk : s.regs .real [p.BK] "b_gk" = some (keyTile p l a.gate j)) :
    evalOp (reducedOp p) s = some (reducedTile p l a qp gp j) := by
  unfold reducedOp
  erw [evalOp_reduceSum]
  rw [product_eval p l a qp gp j s hq hg hk hgk]
  simp only [Option.bind_eq_bind,Option.bind_some]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,⟨⟩⟩
  simp only [Tile.reduceSum_false,Tile.reduceSumDrop_data,TileShape.axisDim,TileShape.insertAxisIndex,
    productTile,Tile.mat,reducedTile,Tile.vec,reduced]
  exact (WithBot.coe_sum Finset.univ (fun e : Fin p.BK => lane p l a qp gp i.val j e.val)).symm

theorem scaled_eval (p : Layout) (l : Launch) (a : Inputs) (qp gp : Nat → ℝ)
    (scale : ℝ) (j : Nat) (swapped : Bool) (s : BlockState)
    (hi : s.regs .nat [p.BC] "o_i" = some (indices p.BC))
    (hj : s.regs .nat [] "j" = some (Tile.scalar j))
    (ha : s.regs .real [p.BC] "b_A" = some (reducedTile p l a qp gp j)) :
    evalOp (scaledOp p scale swapped) s = some (scaledTile p l a qp gp scale j) := by
  cases swapped
  all_goals
    simp only [scaledOp,Bool.false_eq_true,↓reduceIte,evalOp,evalOp_ref,hi,hj,ha,Option.bind_eq_bind,Option.bind_some]
    apply congrArg some
    apply Tile.ext
    rintro ⟨i,⟨⟩⟩
    simp [scaledTile,result,reducedTile,indices,Tile.vec,-Tile.scalar_eta,Tile.scalar,
      Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex,ComparableDType.ge,
      NumericDType.mul,mul_comm]
    split <;> norm_num

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealIntraAlgebra
