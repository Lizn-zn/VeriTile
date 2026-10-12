import bench.tritonbench_g.context_attn_fwd.Proofs.LoopProgram
namespace VeriTile.Bench.TritonBenchG.ContextAttnFwd.KeysExecution
open VeriTile.Triton Formula Program PreludeProgram LoopProgram
noncomputable section
set_option maxHeartbeats 1500000
set_option maxRecDepth 5000
set_option linter.unusedSimpArgs false

def address (p : Layout) (batch head : Nat) (sn : Int) (d : Fin p.D) (j : Fin p.BN) : Int :=
  batch*p.kBatch+(sn+j.val)*p.kRow+head*p.kHead+d.val*p.kDim

def loaded (p : Layout) (r : Regions) (batch head : Nat) (sn limit : Int) (u : BlockState) : Tile .real [p.D,p.BN] :=
  Tile.mat (fun d j => some (if sn+j.val < limit then
    u.readMem r.key (address p batch head sn d j).toNat else 0))

def prepared (p : Layout) (r : Regions) (batch head : Nat) (sn limit : Int) (u : BlockState) :=
  let u' := u.setReg "start_n" .int [] (Tile.scalar sn)
  let u' := u'.setReg "off_k" .int [p.D,p.BN] (Tile.mat (address p batch head sn))
  u'.setReg "k" .real [p.D,p.BN] (loaded p r batch head sn limit u)

theorem executes (p : Layout) (r : Regions) (batch head : Nat) (sn limit : Int) (u : BlockState)
    (hb : u.regs .nat [] "cur_batch" = some (Tile.scalar batch))
    (hh : u.regs .nat [] "cur_kv_head" = some (Tile.scalar head))
    (hs : u.regs .int [] "start_n" = some (Tile.scalar sn))
    (hl : u.regs .int [] "block_end_loc" = some (Tile.scalar limit))
    (hn : u.regs .nat [p.BN] "offs_n" = some (Tile.vec (fun j => j.val)))
    (hd : u.regs .nat [p.D] "offs_d" = some (Tile.vec (fun d => d.val))) :
    stepStmts (keys p r) u = some (prepared p r batch head sn limit u) := by
  have hz : (0.0 : ℝ) = 0 := by norm_num
  simp [keys,keyOffsets,keyMask,keyView,keyChannel,lanes,dimensions,n,z,prepared,loaded,address,
    stepStmts,stepStmt,evalOp.eq_def,hb,hh,hs,hl,hn,hd,hz,
    tile_elementwise,Tile.remap,Tile.scalar,Tile.vec,Tile.mat,Broadcast.leftIndex,Broadcast.rightIndex,
    Option.bind_eq_bind,Option.bind,apply_ite,Nat.cast_add,Nat.cast_mul,-Tile.scalar_eta]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnFwd.KeysExecution
