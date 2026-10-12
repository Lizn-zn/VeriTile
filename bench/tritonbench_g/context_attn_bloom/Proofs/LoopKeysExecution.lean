import bench.tritonbench_g.context_attn_bloom.Proofs.LoopKeysProgram
namespace VeriTile.Bench.TritonBenchG.ContextAttnBloom.LoopKeysExecution
open VeriTile.Triton Math Program RealSetup PreludeProgram LoopKeysProgram
noncomputable section
set_option maxHeartbeats 700000
set_option maxRecDepth 3000
set_option linter.unusedSimpArgs false

def page (p : Layout) (r : Regions) (sn req last : Int) (u : BlockState) (j : Nat) : Int :=
  if sn+j < last then
    u.readMemValue .int r.tokens (p.requestRow*req+p.requestToken*(sn+j)).toNat
  else 0

def keyAddress (p : Layout) (kvHead : Nat) (page : Int) (d : Nat) : Int :=
  page*p.kRow+kvHead*p.kHead+d*p.kDim

def prepared (p : Layout) (r : Regions) (sn req last : Int) (kvHead : Nat) (u : BlockState) : BlockState :=
  let pages := page p r sn req last u
  let u := u.setReg "start_n" .int [] (Tile.scalar sn)
  let u := u.setReg "kv_loc" .int [p.BN] (Tile.vec (fun j => pages j.val))
  let u := u.setReg "off_k" .int [p.D,p.BN]
    (Tile.mat (fun d j => keyAddress p kvHead (pages j.val) d.val))
  let kt := Tile.mat (dtype := .real) (fun (d : Fin p.D) (j : Fin p.BN) =>
    some (if sn+j.val < last ∧ d.val < p.headDim then
      u.readMem r.key (keyAddress p kvHead (pages j.val) d.val).toNat else 0))
  u.setReg "k" .real [p.D,p.BN] kt

theorem executes (p : Layout) (r : Regions) (sn req last : Int) (kvHead : Nat) (u : BlockState)
    (hsn : u.regs .int [] "start_n" = some (Tile.scalar sn))
    (hr : u.regs .int [] "cur_batch_req_idx" = some (Tile.scalar req))
    (hl : u.regs .int [] "block_end_loc" = some (Tile.scalar last))
    (hk : u.regs .nat [] "cur_kv_head" = some (Tile.scalar kvHead))
    (hn : u.regs .nat [p.BN] "offs_n" = some (Tile.vec (fun j => j.val)))
    (hd : u.regs .nat [p.D] "offs_d" = some (Tile.vec (fun d => d.val))) :
    stepStmts (keys p r) u = some (prepared p r sn req last kvHead u) := by
  rw [LoopKeysProgram.source]
  have hz : (0.0 : ℝ) = 0 := by norm_num
  simp [code,prepared,page,keyAddress,gather,tokenIndices,lanes,tokenView,channelView,
    keyOffsets,keyMask,n,z,stepStmts,stepStmt,evalOp.eq_def,hsn,hr,hl,hk,hn,hd,hz,
    tile_elementwise,Tile.scalar,Tile.vec,Tile.mat,Broadcast.leftIndex,Broadcast.rightIndex,
    Option.bind_eq_bind,Option.bind,apply_ite,Nat.cast_add,Nat.cast_mul,-Tile.scalar_eta]
  rfl

theorem memory (p : Layout) (r : Regions) (sn req last : Int) (kvHead : Nat) (u : BlockState) :
    (prepared p r sn req last kvHead u).mem = u.mem := by
  funext region off
  simp only [prepared,BlockState.setReg_mem]
end
end VeriTile.Bench.TritonBenchG.ContextAttnBloom.LoopKeysExecution
