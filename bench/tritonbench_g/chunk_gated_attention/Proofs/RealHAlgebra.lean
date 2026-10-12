import bench.tritonbench_g.chunk_gated_attention.Proofs.RealHLoads

/-! Evaluation of the actual real gate and carry expressions. These arithmetic
lemmas are instantiated after the source loads; they do not assume a supplied
kernel output, a prepared carry buffer or an FP exponential identity. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealHAlgebra
open VeriTile.Triton RealHMath RealHProgram RealHTiles
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def scaleKeyOp (p : Layout) : Op .real [p.BK,p.BV] :=
  .mul .real (.consSame (.consR .nil)) (carry p)
    (.expandDim ⟨1,by simp⟩ (.exp (.ref .real [p.BK] "b_gn")))
def scaleValueOp (p : Layout) : Op .real [p.BK,p.BV] :=
  .mul .real (.consR (.consSame .nil)) (carry p)
    (.expandDim ⟨0,by simp⟩ (.exp (.ref .real [p.BV] "b_gn")))
def gateKeyOp (p : Layout) : Op .real [p.BK,p.BT] :=
  .mul .real (.consSame (.consSame .nil)) (.ref .real [p.BK,p.BT] "b_k")
    (.exp (.sub .real (.consSame (.consL .nil))
      (.expandDim ⟨1,by simp⟩ (.ref .real [p.BK] "b_gn")) (.ref .real [p.BK,p.BT] "b_g")))
def gateValueOp (p : Layout) : Op .real [p.BT,p.BV] :=
  .mul .real (.consSame (.consSame .nil)) (.ref .real [p.BT,p.BV] "b_v")
    (.exp (.sub .real (.consL (.consSame .nil))
      (.expandDim ⟨0,by simp⟩ (.ref .real [p.BV] "b_gn")) (.ref .real [p.BT,p.BV] "b_g")))
def updateOp (p : Layout) (swapped : Bool) : Op .real [p.BK,p.BV] :=
  if swapped then .add .real (.consSame (.consSame .nil)) (dot p) (carry p)
  else .add .real (.consSame (.consSame .nil)) (carry p) (dot p)

theorem scale_key (p : Layout) (x : Loaded p) (f : StateValue p) (u : BlockState)
    (hh : u.regs .real [p.BK,p.BV] "b_h" = some (stateTile p f))
    (he : u.regs .real [p.BK] "b_gn" = some (Tile.vec (fun i => some (x.keyEnd i)))) :
    evalOp (scaleKeyOp p) u = some (stateTile p (scaled x Bool.true f)) := by
  simp only [scaleKeyOp,carry,evalOp_mul,evalOp_expandDim,evalOp_exp,evalOp_ref,hh,he,
    Option.bind_eq_bind,Option.bind_some]
  erw [evalOp_expandDim]
  simp only [evalOp_exp, evalOp_ref, he, Option.bind_eq_bind, Option.bind_some]
  apply congrArg some
  apply Tile.ext
  rintro ⟨e,j,⟨⟩⟩
  simp [stateTile,scaled,Tile.mat,Tile.vec,Tile.bop,Tile.uop,Tile.expandDim,
    Broadcast.leftIndex,Broadcast.rightIndex,TileShape.dropInsertedIndex,NumericDType.mul,WithBot.realExp]

theorem scale_value (p : Layout) (x : Loaded p) (f : StateValue p) (u : BlockState)
    (hh : u.regs .real [p.BK,p.BV] "b_h" = some (stateTile p f))
    (he : u.regs .real [p.BV] "b_gn" = some (Tile.vec (fun i => some (x.valueEnd i)))) :
    evalOp (scaleValueOp p) u = some (stateTile p (scaled x Bool.false f)) := by
  simp only [scaleValueOp,carry,evalOp_mul,evalOp_expandDim,evalOp_exp,evalOp_ref,hh,he,
    Option.bind_eq_bind,Option.bind_some]
  erw [evalOp_expandDim]
  simp only [evalOp_exp, evalOp_ref, he, Option.bind_eq_bind, Option.bind_some]
  apply congrArg some
  apply Tile.ext
  rintro ⟨e,j,⟨⟩⟩
  simp [stateTile,scaled,Tile.mat,Tile.vec,Tile.bop,Tile.uop,Tile.expandDim,
    Broadcast.leftIndex,Broadcast.rightIndex,TileShape.dropInsertedIndex,NumericDType.mul,WithBot.realExp]

theorem gate_key (p : Layout) (x : Loaded p) (u : BlockState)
    (hk : u.regs .real [p.BK,p.BT] "b_k" = some (Tile.mat (fun e t => some (x.key e t))))
    (hg : u.regs .real [p.BK,p.BT] "b_g" = some (Tile.mat (fun e t => some (x.keyGate e t))))
    (he : u.regs .real [p.BK] "b_gn" = some (Tile.vec (fun e => some (x.keyEnd e)))) :
    evalOp (gateKeyOp p) u = some (Tile.mat (fun e t => some (gatedKey x Bool.true e t))) := by
  simp only [gateKeyOp,evalOp_mul,evalOp_exp,evalOp_sub,evalOp_expandDim,evalOp_ref,hk,hg,he,
    Option.bind_eq_bind,Option.bind_some]
  erw [evalOp_expandDim]
  simp only [evalOp_exp, evalOp_ref, he, Option.bind_eq_bind, Option.bind_some]
  apply congrArg some
  apply Tile.ext
  rintro ⟨e,t,⟨⟩⟩
  simp [gatedKey,Tile.mat,Tile.vec,Tile.bop,Tile.uop,Tile.expandDim,
    Broadcast.leftIndex,Broadcast.rightIndex,TileShape.dropInsertedIndex,
    NumericDType.mul,NumericDType.sub,WithBot.realExp]

theorem gate_value (p : Layout) (x : Loaded p) (u : BlockState)
    (hv : u.regs .real [p.BT,p.BV] "b_v" = some (Tile.mat (fun t j => some (x.value t j))))
    (hg : u.regs .real [p.BT,p.BV] "b_g" = some (Tile.mat (fun t j => some (x.valueGate t j))))
    (he : u.regs .real [p.BV] "b_gn" = some (Tile.vec (fun j => some (x.valueEnd j)))) :
    evalOp (gateValueOp p) u = some (Tile.mat (fun t j => some (gatedValue x Bool.false t j))) := by
  simp only [gateValueOp,evalOp_mul,evalOp_exp,evalOp_sub,evalOp_expandDim,evalOp_ref,hv,hg,he,
    Option.bind_eq_bind,Option.bind_some]
  erw [evalOp_expandDim]
  simp only [evalOp_exp, evalOp_ref, he, Option.bind_eq_bind, Option.bind_some]
  apply congrArg some
  apply Tile.ext
  rintro ⟨t,j,⟨⟩⟩
  simp [gatedValue,Tile.mat,Tile.vec,Tile.bop,Tile.uop,Tile.expandDim,
    Broadcast.leftIndex,Broadcast.rightIndex,TileShape.dropInsertedIndex,
    NumericDType.mul,NumericDType.sub,WithBot.realExp]

theorem product_eval (p : Layout) (x : Loaded p) (gateK : Bool) (u : BlockState)
    (hk : u.regs .real [p.BK,p.BT] "b_k" = some (Tile.mat (fun e t => some (gatedKey x gateK e t))))
    (hv : u.regs .real [p.BT,p.BV] "b_v" = some (Tile.mat (fun t j => some (gatedValue x gateK t j)))) :
    evalOp (dot p) u = some (stateTile p (product x gateK)) := by
  unfold dot
  erw [evalOp_dot]
  erw [evalOp_ref, hk, evalOp_ref, hv]
  simp only [Option.bind_eq_bind,Option.bind_some]
  apply congrArg some
  apply Tile.ext
  rintro ⟨e,j,⟨⟩⟩
  rw [Tile.dot_nil_data]
  simp only [Tile.mat,Option.map₂_some_some,stateTile,product]
  exact (WithBot.coe_sum Finset.univ (fun t : Fin p.BT => gatedKey x gateK e t * gatedValue x gateK t j)).symm

theorem update_eval (p : Layout) (x : Loaded p) (gateK swapped : Bool) (f : StateValue p) (u : BlockState)
    (hh : u.regs .real [p.BK,p.BV] "b_h" = some (stateTile p (scaled x gateK f)))
    (hk : u.regs .real [p.BK,p.BT] "b_k" = some (Tile.mat (fun e t => some (gatedKey x gateK e t))))
    (hv : u.regs .real [p.BT,p.BV] "b_v" = some (Tile.mat (fun t j => some (gatedValue x gateK t j)))) :
    evalOp (updateOp p swapped) u = some (stateTile p (advance x gateK f)) := by
  cases swapped
  all_goals
    simp only [updateOp,Bool.false_eq_true,↓reduceIte,evalOp_add,carry,evalOp_ref,hh,
      product_eval p x gateK u hk hv,Option.bind_eq_bind,Option.bind_some]
    apply congrArg some
    apply Tile.ext
    rintro ⟨e,j,⟨⟩⟩
    simp [advance,stateTile,Tile.mat,Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex,
      NumericDType.add,add_comm]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealHAlgebra
