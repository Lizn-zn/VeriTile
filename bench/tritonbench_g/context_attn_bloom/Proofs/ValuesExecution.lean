import bench.tritonbench_g.context_attn_bloom.Proofs.ValuesProgram
namespace VeriTile.Bench.TritonBenchG.ContextAttnBloom.ValuesExecution
open VeriTile.Triton Math Program RealSetup PreludeProgram ValuesProgram
noncomputable section
set_option maxHeartbeats 600000
set_option maxRecDepth 3000
set_option linter.unusedSimpArgs false

def address (p : Layout) (kvHead : Nat) (page : Int) (d : Nat) : Int :=
  page*p.vRow+kvHead*p.vHead+d*p.vDim

def prepared (p : Layout) (r : Regions) (sn last : Int) (kvHead : Nat)
    (pages : Fin p.BN → Int) (u : BlockState) : BlockState :=
  let u := u.setReg "off_v" .int [p.BN,p.D]
    (Tile.mat (fun j d => address p kvHead (pages j) d.val))
  u.setReg "v" .real [p.BN,p.D]
    (Tile.mat (fun j d => some (if sn+j.val < last ∧ d.val < p.headDim then
      u.readMem r.value (address p kvHead (pages j) d.val).toNat else 0)))

theorem executes (p : Layout) (r : Regions) (sn last : Int) (kvHead : Nat)
    (pages : Fin p.BN → Int) (u : BlockState)
    (hsn : u.regs .int [] "start_n" = some (Tile.scalar sn))
    (hl : u.regs .int [] "block_end_loc" = some (Tile.scalar last))
    (hk : u.regs .nat [] "cur_kv_head" = some (Tile.scalar kvHead))
    (hn : u.regs .nat [p.BN] "offs_n" = some (Tile.vec (fun j => j.val)))
    (hd : u.regs .nat [p.D] "offs_d" = some (Tile.vec (fun d => d.val)))
    (hp : u.regs .int [p.BN] "kv_loc" = some (Tile.vec pages)) :
    stepStmts (values p r) u = some (prepared p r sn last kvHead pages u) := by
  rw [ValuesProgram.source]
  have hz : (0.0 : ℝ) = 0 := by norm_num
  simp [code,prepared,address,tokenView,channelView,offsets,mask,n,z,
    stepStmts,stepStmt,evalOp.eq_def,hsn,hl,hk,hn,hd,hp,hz,
    tile_elementwise,Tile.scalar,Tile.vec,Tile.mat,Broadcast.leftIndex,Broadcast.rightIndex,
    Option.bind_eq_bind,Option.bind,apply_ite,Nat.cast_add,Nat.cast_mul,-Tile.scalar_eta]
  rfl

theorem memory (p : Layout) (r : Regions) (sn last : Int) (kvHead : Nat)
    (pages : Fin p.BN → Int) (u : BlockState) :
    (prepared p r sn last kvHead pages u).mem = u.mem := by
  funext region off
  simp only [prepared,BlockState.setReg_mem]
end
end VeriTile.Bench.TritonBenchG.ContextAttnBloom.ValuesExecution
