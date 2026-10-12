import bench.tritonbench_g.context_attn_bloom.Proofs.ScoresProgram
import bench.tritonbench_g.context_attn_bloom.Proofs.NumericOps
namespace VeriTile.Bench.TritonBenchG.ContextAttnBloom.ScoreTiles
open VeriTile.Triton Math Program RealSetup PreludeProgram ScoresProgram NumericOps
noncomputable section
set_option maxHeartbeats 900000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

def mask (p : Layout) (rows : Fin p.BM → Nat) (prompt sn : Int) : Tile .bool [p.BM,p.BN] :=
  Tile.mat (fun i j => decide (sn+j.val ≤ rows i+prompt))

theorem mask_eval (p : Layout) (rows : Fin p.BM → Nat) (prompt sn : Int) (u : BlockState)
    (hr : u.regs .nat [p.BM] "offs_m" = some (Tile.vec rows))
    (hn : u.regs .nat [p.BN] "offs_n" = some (Tile.vec (fun j => j.val)))
    (hp : u.regs .int [] "prompt_cache_len" = some (Tile.scalar prompt))
    (hs : u.regs .int [] "start_n" = some (Tile.scalar sn)) :
    evalOp (causalMask p) u = some (mask p rows prompt sn) := by
  simp [causalMask,mask,n,z,evalOp.eq_def,hr,hn,hp,hs,
    tile_elementwise,Tile.scalar,Tile.vec,Tile.mat,Broadcast.leftIndex,Broadcast.rightIndex,
    Option.bind_eq_bind,Option.bind,-Tile.scalar_eta]

def zero (p : Layout) : Tile .real [p.BM,p.BN] := Tile.mat (fun _ _ => some 0)
def dotAdded (p : Layout) (q : Tile .real [p.BM,p.D]) (k : Tile .real [p.D,p.BN]) :=
  Tile.bop NumericDType.real.add Broadcast.nil.consSame.consSame (zero p) (Tile.dot [] q k)
def scaled (p : Layout) (swapped : Bool) (scale : ℝ) (q : Tile .real [p.BM,p.BN]) :=
  if swapped then Tile.bop NumericDType.real.mul Broadcast.scalarL (Tile.scalar (some scale)) q
  else Tile.bop NumericDType.real.mul Broadcast.scalarR q (Tile.scalar (some scale))
def result (p : Layout) (swapped : Bool) (scale : ℝ) (rows : Fin p.BM → Nat) (prompt sn : Int)
    (q : Tile .real [p.BM,p.D]) (k : Tile .real [p.D,p.BN]) :=
  Tile.select (mask p rows prompt sn) (scaled p swapped scale (dotAdded p q k))
    (Tile.mat (fun _ _ => some (-100000000 : ℝ)))
def prepared (p : Layout) (swapped : Bool) (scale : ℝ) (rows : Fin p.BM → Nat) (prompt sn : Int)
    (q : Tile .real [p.BM,p.D]) (k : Tile .real [p.D,p.BN]) (u : BlockState) :=
  let u := u.setReg "qk" .real [p.BM,p.BN] (zero p)
  let u := u.setReg "qk" .real [p.BM,p.BN] (dotAdded p q k)
  let u := u.setReg "qk" .real [p.BM,p.BN] (scaled p swapped scale (dotAdded p q k))
  u.setReg "qk" .real [p.BM,p.BN] (result p swapped scale rows prompt sn q k)

end
end VeriTile.Bench.TritonBenchG.ContextAttnBloom.ScoreTiles
