import bench.tritonbench_g.context_attn_fwd.Proofs.LoopProgram
namespace VeriTile.Bench.TritonBenchG.ContextAttnFwd.ScoresExecution
open VeriTile.Triton Formula Program PreludeProgram LoopProgram NumericOps
noncomputable section
set_option maxHeartbeats 800000
set_option maxRecDepth 5000
set_option linter.unusedSimpArgs false
def mask (p : Layout) (rows : Fin p.BM → Nat) (prompt sn : Int) : Tile .bool [p.BM,p.BN] :=
  Tile.mat (fun i j => decide (sn+j.val ≤ rows i+prompt))

theorem mask_eval (p : Layout) (rows : Fin p.BM → Nat) (prompt sn : Int) (u : BlockState)
    (hr : u.regs .nat [p.BM] "offs_m" = some (Tile.vec rows))
    (hn : u.regs .nat [p.BN] "offs_n" = some (Tile.vec (fun j => j.val)))
    (hp : u.regs .int [] "prompt_cache_len" = some (Tile.scalar prompt))
    (hs : u.regs .int [] "start_n" = some (Tile.scalar sn)) :
    evalOp (causalMask p) u = some (mask p rows prompt sn) := by
  simp [causalMask,mask,n,z,rowView,PreludeProgram.rows,keyView,lanes,evalOp.eq_def,hr,hn,hp,hs,
    tile_elementwise,Tile.scalar,Tile.vec,Tile.mat,Broadcast.leftIndex,Broadcast.rightIndex,
    Option.bind_eq_bind,Option.bind,-Tile.scalar_eta]

def scaled (p : Layout) (swapped : Bool) (scale : ℝ) (q : Tile .real [p.BM,p.BN]) :=
  if swapped then Tile.bop NumericDType.real.mul Broadcast.scalarL (Tile.scalar (some scale)) q
  else Tile.bop NumericDType.real.mul Broadcast.scalarR q (Tile.scalar (some scale))
def result (p : Layout) (swapped : Bool) (scale : ℝ) (rows : Fin p.BM → Nat) (prompt sn : Int)
    (q : Tile .real [p.BM,p.D]) (k : Tile .real [p.D,p.BN]) :=
  Tile.select (mask p rows prompt sn) (scaled p swapped scale (Tile.dot [] q k))
    (Tile.mat (fun _ _ => some (-100000000 : ℝ)))
def prepared (p : Layout) (swapped : Bool) (scale : ℝ) (rows : Fin p.BM → Nat) (prompt sn : Int)
    (q : Tile .real [p.BM,p.D]) (k : Tile .real [p.D,p.BN]) (u : BlockState) :=
  let u := u.setReg "qk" .real [p.BM,p.BN] (Tile.dot [] q k)
  let u := u.setReg "mask" .bool [p.BM,p.BN] (mask p rows prompt sn)
  u.setReg "qk" .real [p.BM,p.BN] (result p swapped scale rows prompt sn q k)

theorem scale_eval (p : Layout) (swapped : Bool) (scale : ℝ) (v : Tile .real [p.BM,p.BN])
    (u : BlockState) (hv : u.regs .real [p.BM,p.BN] "qk" = some v) :
    evalOp (scaling p swapped scale) u = some (scaled p swapped scale v) := by
  cases swapped <;> simp [scaling,scaled,evalOp_mul,hv]

theorem sentinel_eval (p : Layout) (u : BlockState) :
    evalOp (Op.broadcast (.sub .real .nil (.const (0.0 : ℝ)) (.const (10e7 : ℝ))) [p.BM,p.BN]) u =
      some (Tile.mat (dtype := .real) (fun _ _ => some (-100000000 : ℝ))) := by
  norm_num [evalOp.eq_def,Tile.bop,Tile.scalar,Tile.mat,NumericDType.sub,WithBot.realSub,
    Broadcast.leftIndex,Broadcast.rightIndex,Option.bind_eq_bind,Option.bind,-Tile.scalar_eta]

theorem executes (p : Layout) (swapped : Bool) (scale : ℝ)
    (rows : Fin p.BM → Nat) (prompt sn : Int)
    (q : Tile .real [p.BM,p.D]) (k : Tile .real [p.D,p.BN]) (u : BlockState)
    (hr : u.regs .nat [p.BM] "offs_m" = some (Tile.vec rows))
    (hn : u.regs .nat [p.BN] "offs_n" = some (Tile.vec (fun j => j.val)))
    (hp : u.regs .int [] "prompt_cache_len" = some (Tile.scalar prompt))
    (hs : u.regs .int [] "start_n" = some (Tile.scalar sn))
    (hq : u.regs .real [p.BM,p.D] "q" = some q)
    (hk : u.regs .real [p.D,p.BN] "k" = some k) :
    stepStmts (scores p swapped scale) u = some (prepared p swapped scale rows prompt sn q k u) := by
  let s1 := u.setReg "qk" .real [p.BM,p.BN] (Tile.dot [] q k)
  let s2 := s1.setReg "mask" .bool [p.BM,p.BN] (mask p rows prompt sn)
  have h1 := ctxQk_op_eval u q k hq hk
  have h2 := mask_eval p rows prompt sn s1
    (by simp [s1,hr]) (by simp [s1,hn]) (by simp [s1,hp]) (by simp [s1,hs])
  have hscale := scale_eval p swapped scale (Tile.dot [] q k) s2 (by simp [s2,s1])
  have h3 : evalOp (.where (.ref .bool [p.BM,p.BN] "mask") (scaling p swapped scale)
      (.broadcast (.sub .real .nil (.const (0.0 : ℝ)) (.const (10e7 : ℝ))) [p.BM,p.BN])) s2 =
      some (result p swapped scale rows prompt sn q k) := by
    rw [evalOp_where,evalOp_ref,show s2.regs .bool [p.BM,p.BN] "mask" = some (mask p rows prompt sn) by simp [s2]]
    simp only [Option.bind_some,hscale,sentinel_eval]
    rfl
  unfold scores
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (dtype := .real) (shape := [p.BM,p.BN]) (name := "qk") h1)]
  change stepStmts _ s1 = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some h2)]
  change stepStmts _ s2 = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some h3)]
  simp only [stepStmts.nil,prepared,s2,s1]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnFwd.ScoresExecution
