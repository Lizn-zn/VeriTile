import bench.tritonbench_g.context_attn_bloom.Proofs.ScoreTiles
namespace VeriTile.Bench.TritonBenchG.ContextAttnBloom.ScoresStaged
open VeriTile.Triton Math Program RealSetup PreludeProgram ScoresProgram NumericOps ScoreTiles
noncomputable section
set_option maxHeartbeats 700000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

theorem scale_eval (p : Layout) (swapped : Bool) (scale : ℝ) (v : Tile .real [p.BM,p.BN])
    (u : BlockState) (hv : u.regs .real [p.BM,p.BN] "qk" = some v) :
    evalOp (scaling p swapped scale) u = some (scaled p swapped scale v) := by
  cases swapped <;> simp [scaling,scaled,evalOp_mul,hv]

theorem sentinel_eval (p : Layout) (u : BlockState) :
    evalOp (Op.broadcast (.sub .real .nil (.const (0.0 : ℝ)) (.const (100000000.0 : ℝ))) [p.BM,p.BN]) u =
      some (Tile.mat (dtype := .real) (fun _ _ => some (-100000000 : ℝ))) := by
  norm_num [evalOp.eq_def,Tile.bop,Tile.scalar,Tile.mat,NumericDType.sub,WithBot.realSub,
    Broadcast.leftIndex,Broadcast.rightIndex,Option.bind_eq_bind,Option.bind,-Tile.scalar_eta]

theorem executes (p : Layout) (r : Regions) (swapped : Bool) (scale : ℝ)
    (rows : Fin p.BM → Nat) (prompt sn : Int)
    (q : Tile .real [p.BM,p.D]) (k : Tile .real [p.D,p.BN]) (u : BlockState)
    (hr : u.regs .nat [p.BM] "offs_m" = some (Tile.vec rows))
    (hn : u.regs .nat [p.BN] "offs_n" = some (Tile.vec (fun j => j.val)))
    (hp : u.regs .int [] "prompt_cache_len" = some (Tile.scalar prompt))
    (hs : u.regs .int [] "start_n" = some (Tile.scalar sn))
    (hq : u.regs .real [p.BM,p.D] "q" = some q)
    (hk : u.regs .real [p.D,p.BN] "k" = some k) :
    stepStmts (scores p r swapped scale) u = some (prepared p swapped scale rows prompt sn q k u) := by
  let s1 := u.setReg "qk" .real [p.BM,p.BN] (zero p)
  let s2 := s1.setReg "qk" .real [p.BM,p.BN] (dotAdded p q k)
  let s3 := s2.setReg "qk" .real [p.BM,p.BN] (scaled p swapped scale (dotAdded p q k))
  have h1 : evalOp (Op.full [p.BM,p.BN] (.const 0)) u = some (zero p) := by
    simpa only [zero,Tile.mat] using (bloomQkFull_eval (BM := p.BM) (BN := p.BN) u)
  have h2 := bloomQkAddDot_eval s1 (zero p) q k
    (by simp [s1]) (by simp [s1,hq]) (by simp [s1,hk])
  change evalOp _ s1 = some (dotAdded p q k) at h2
  have h3 := scale_eval p swapped scale (dotAdded p q k) s2 (by simp [s2])
  have hc := mask_eval p rows prompt sn s3
    (by simp [s3,s2,s1,hr]) (by simp [s3,s2,s1,hn])
    (by simp [s3,s2,s1,hp]) (by simp [s3,s2,s1,hs])
  have hqk3 : s3.regs .real [p.BM,p.BN] "qk" = some (scaled p swapped scale (dotAdded p q k)) := by simp [s3]
  have h4 : evalOp (.where (causalMask p) (.ref .real [p.BM,p.BN] "qk")
      (.broadcast (.sub .real .nil (.const (0.0 : ℝ)) (.const (100000000.0 : ℝ))) [p.BM,p.BN])) s3 =
      some (result p swapped scale rows prompt sn q k) := by
    rw [evalOp_where,hc,evalOp_ref,hqk3,sentinel_eval]
    rfl
  rw [ScoresProgram.source]
  unfold code
  rw [stepStmts.cons_some (stepStmt_assign_eq_some h1)]
  change stepStmts _ s1 = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some h2)]
  change stepStmts _ s2 = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some h3)]
  change stepStmts _ s3 = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some h4)]
  simp only [stepStmts,prepared,s1,s2,s3]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnBloom.ScoresStaged
