import bench.tritonbench_g.context_attn_bloom.Proofs.ScoresStaged
import bench.tritonbench_g.context_attn_bloom.Proofs.Invariant
namespace VeriTile.Bench.TritonBenchG.ContextAttnBloom.ScoresMeaning
open VeriTile.Triton Math Program RealSetup NumericOps ScoreTiles
noncomputable section
set_option maxHeartbeats 700000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

theorem cell (p : Layout) (swapped : Bool) (scale : ℝ) (rows : Fin p.BM → Nat) (prompt sn : Int)
    (q : Tile .real [p.BM,p.D]) (k : Tile .real [p.D,p.BN])
    (qf : Fin p.BM → Fin p.D → ℝ) (kf : Fin p.BN → Fin p.D → ℝ)
    (hq : ∀ i d, q.data (i,d,PUnit.unit) = some (qf i d))
    (hk : ∀ j d, k.data (d,j,PUnit.unit) = some (kf j d)) (i : Fin p.BM) (j : Fin p.BN) :
    (result p swapped scale rows prompt sn q k).data (i,j,PUnit.unit) =
      some (if sn+j.val ≤ rows i+prompt then scale*∑ d : Fin p.D, qf i d*kf j d else -100000000) := by
  have hdot := ctx_dot_score_cell q k i j qf kf (hq i) (hk j)
  have hadd : (dotAdded p q k).data (i,j,PUnit.unit) =
      some (∑ d : Fin p.D, qf i d*kf j d) := by
    simp only [dotAdded,Tile.bop_data,Broadcast.leftIndex,Broadcast.rightIndex,
      NumericDType.add,zero,Tile.mat_data,hdot,WithBot.realAdd,Option.map₂,Option.bind,Option.map,zero_add]
  rw [result,Tile.select_data]
  by_cases h : sn+j.val ≤ rows i+prompt
  · simp only [mask,Tile.mat_data,h,decide_true,↓reduceIte]
    cases swapped <;>
      simp [scaled,Tile.bop_data,Broadcast.leftIndex,Broadcast.rightIndex,NumericDType.mul,
        Tile.scalar_data,hadd,Option.map₂,mul_comm]
  · simp [mask,Tile.mat_data,h]

theorem register_matches (p : Layout) (r : Regions) (swapped : Bool) (scale : ℝ) (initial u : BlockState)
    (c : Nat) (q : Tile .real [p.BM,p.D]) (k : Tile .real [p.D,p.BN])
    (hq : ∀ i d, q.data (i,d,PUnit.unit) = some (Math.query p (launch initial) (inputs initial r) i.val d.val))
    (hk : ∀ j d, k.data (d,j,PUnit.unit) = some (Math.key p (launch initial) (inputs initial r) (c*p.BN+j.val) d.val)) :
    let l := launch initial
    let a := inputs initial r
    (prepared p swapped scale (fun i => queryRow p l i.val) (a.prompt l.batch) (c*p.BN) q k u).regs
      .real [p.BM,p.BN] "qk" = some (Tile.mat (dtype := .real)
        (fun i j => some (Math.score p l a scale i.val (c*p.BN+j.val)))) := by
  dsimp only
  simp only [prepared,BlockState.setReg_same]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  rw [cell p swapped scale _ _ _ q k _ _ hq hk i j]
  simp only [Math.score,Nat.cast_add,Nat.cast_mul,Tile.mat_data]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnBloom.ScoresMeaning
