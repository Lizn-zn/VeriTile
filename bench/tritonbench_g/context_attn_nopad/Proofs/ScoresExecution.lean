import bench.tritonbench_g.context_attn_nopad.Proofs.LoopProgram
import bench.tritonbench_g.context_attn_bloom.Proofs.NumericOps
namespace VeriTile.Bench.TritonBenchG.ContextAttnNopad.ScoresExecution
open VeriTile.Triton Formula Program PreludeProgram LoopProgram
open VeriTile.Bench.TritonBenchG.ContextAttnBloom.NumericOps
noncomputable section
set_option maxHeartbeats 800000
set_option maxRecDepth 5000
set_option linter.unusedSimpArgs false
def mask (p : Layout) (rows : Fin p.BM → Nat) (sn : Int) : Tile .bool [p.BM,p.BN] :=
  Tile.mat (fun i j => decide (sn+j.val ≤ (rows i : Int)))

theorem mask_eval (p : Layout) (rows : Fin p.BM → Nat) (sn : Int) (u : BlockState)
    (hr : u.regs .nat [p.BM] "offs_m" = some (Tile.vec rows))
    (hn : u.regs .nat [p.BN] "offs_n" = some (Tile.vec (fun j => j.val)))
    (hs : u.regs .int [] "start_n" = some (Tile.scalar sn)) :
    evalOp (causalMask p) u = some (mask p rows sn) := by
  simp [causalMask,mask,n,z,rowView,PreludeProgram.rows,keyView,lanes,evalOp.eq_def,hr,hn,hs,
    tile_elementwise,Tile.scalar,Tile.vec,Tile.mat,Broadcast.leftIndex,Broadcast.rightIndex,
    Option.bind_eq_bind,Option.bind,-Tile.scalar_eta]

def scaled (p : Layout) (swapped : Bool) (scale : ℝ) (q : Tile .real [p.BM,p.BN]) :=
  if swapped then Tile.bop NumericDType.real.mul Broadcast.scalarL (Tile.scalar (some scale)) q
  else Tile.bop NumericDType.real.mul Broadcast.scalarR q (Tile.scalar (some scale))
def zero (p : Layout) : Tile .real [p.BM,p.BN] := Tile.mat (fun _ _ => some 0)
def added (p : Layout) (q : Tile .real [p.BM,p.D]) (k : Tile .real [p.D,p.BN]) :=
  Tile.bop NumericDType.real.add Broadcast.nil.consSame.consSame (zero p) (Tile.dot [] q k)
def masked (p : Layout) (swapped : Bool) (scale : ℝ) (rows : Fin p.BM → Nat) (sn : Int)
    (q : Tile .real [p.BM,p.D]) (k : Tile .real [p.D,p.BN]) :=
  Tile.select (mask p rows sn) (scaled p swapped scale (added p q k))
    (Tile.mat (fun _ _ => ⊥))
def prepared (p : Layout) (swapped : Bool) (scale : ℝ) (rows : Fin p.BM → Nat) (sn : Int)
    (q : Tile .real [p.BM,p.D]) (k : Tile .real [p.D,p.BN]) (u : BlockState) :=
  let u := u.setReg "qk" .real [p.BM,p.BN] (zero p)
  let u := u.setReg "qk" .real [p.BM,p.BN] (added p q k)
  let u := u.setReg "qk" .real [p.BM,p.BN] (scaled p swapped scale (added p q k))
  u.setReg "qk" .real [p.BM,p.BN] (masked p swapped scale rows sn q k)

theorem scale_eval (p : Layout) (swapped : Bool) (scale : ℝ) (v : Tile .real [p.BM,p.BN])
    (u : BlockState) (hv : u.regs .real [p.BM,p.BN] "qk" = some v) :
    evalOp (scaling p swapped scale) u = some (scaled p swapped scale v) := by
  cases swapped <;> simp [scaling,scaled,evalOp_mul,hv]

theorem sentinel_eval (p : Layout) (u : BlockState) :
    evalOp (Op.broadcast .negInf [p.BM,p.BN]) u =
      some (Tile.mat (dtype := .real) (fun _ _ => ⊥)) := by
  simp only [evalOp.eq_def,Option.bind_some]
  rfl

theorem executes (p : Layout) (swapped : Bool) (scale : ℝ)
    (rows : Fin p.BM → Nat) (sn : Int)
    (q : Tile .real [p.BM,p.D]) (k : Tile .real [p.D,p.BN]) (u : BlockState)
    (hr : u.regs .nat [p.BM] "offs_m" = some (Tile.vec rows))
    (hn : u.regs .nat [p.BN] "offs_n" = some (Tile.vec (fun j => j.val)))
    (hs : u.regs .int [] "start_n" = some (Tile.scalar sn))
    (hq : u.regs .real [p.BM,p.D] "q" = some q)
    (hk : u.regs .real [p.D,p.BN] "k" = some k) :
    stepStmts (scores p swapped scale) u = some (prepared p swapped scale rows sn q k u) := by
  let s0 := u.setReg "qk" .real [p.BM,p.BN] (zero p)
  let s1 := s0.setReg "qk" .real [p.BM,p.BN] (added p q k)
  let s2 := s1.setReg "qk" .real [p.BM,p.BN] (scaled p swapped scale (added p q k))
  let s3 := s2.setReg "qk" .real [p.BM,p.BN] (masked p swapped scale rows sn q k)
  have h0 := bloomQkFull_eval (BM := p.BM) (BN := p.BN) u
  have h1 := bloomQkAddDot_eval s0 (zero p) q k
    (by simp [s0]) (by simp [s0,hq]) (by simp [s0,hk])
  change evalOp _ s0 = some (added p q k) at h1
  have h2 := scale_eval p swapped scale (added p q k) s1 (by simp [s1])
  have hcausal := mask_eval p rows sn s2
    (by simp [s2,s1,s0,hr]) (by simp [s2,s1,s0,hn]) (by simp [s2,s1,s0,hs])
  have h3 : evalOp (.where (causalMask p) (.ref .real [p.BM,p.BN] "qk")
      (.broadcast sentinel [p.BM,p.BN])) s2 = some (masked p swapped scale rows sn q k) := by
    rw [evalOp_where,hcausal,evalOp_ref,show s2.regs .real [p.BM,p.BN] "qk" = some (scaled p swapped scale (added p q k)) from by simp [s2]]
    simp only [sentinel,sentinel_eval,Option.bind_some]
    rfl
  unfold scores
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (shape := [p.BM,p.BN]) (name := "qk") h0)]
  change stepStmts _ s0 = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some h1)]
  change stepStmts _ s1 = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some h2)]
  change stepStmts _ s2 = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some h3)]
  simp only [stepStmts.nil,prepared,s3,s2,s1,s0]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnNopad.ScoresExecution
