import bench.tritonbench_g.context_attn_fwd.Proofs.LoopProgram
namespace VeriTile.Bench.TritonBenchG.ContextAttnFwd.ValuesExecution
open VeriTile.Triton Formula Program PreludeProgram LoopProgram NumericOps
noncomputable section
set_option maxHeartbeats 1800000
set_option maxRecDepth 5000
set_option linter.unusedSimpArgs false

def address (p : Layout) (batch head : Nat) (sn : Int) (j : Fin p.BN) (d : Fin p.D) : Int :=
  batch*p.vBatch+(sn+j.val)*p.vRow+head*p.vHead+d.val*p.vDim

def loaded (p : Layout) (r : Regions) (batch head : Nat) (sn limit : Int) (u : BlockState) : Tile .real [p.BN,p.D] :=
  Tile.mat (fun j d => some (if sn+j.val < limit then
    u.readMem r.value (address p batch head sn j d).toNat else 0))

def updated (p : Layout) (prob : Tile .real [p.BM,p.BN]) (value : Tile .real [p.BN,p.D])
    (acc : Tile .real [p.BM,p.D]) :=
  Tile.bop NumericDType.real.add Broadcast.nil.consSame.consSame (Tile.dot [] prob value) acc

def loadedState (p : Layout) (r : Regions) (batch head : Nat) (sn limit : Int) (u : BlockState) :=
  let u' := u.setReg "off_v" .int [p.BN,p.D] (Tile.mat (address p batch head sn))
  u'.setReg "v" .real [p.BN,p.D] (loaded p r batch head sn limit u)

def prepared (p : Layout) (r : Regions) (batch head : Nat) (sn limit : Int)
    (prob : Tile .real [p.BM,p.BN]) (mx : Tile .real [p.BM]) (acc : Tile .real [p.BM,p.D]) (u : BlockState) :=
  let u' := loadedState p r batch head sn limit u
  let u' := u'.setReg "p" .real [p.BM,p.BN] prob
  let u' := u'.setReg "acc" .real [p.BM,p.D] (updated p prob (loaded p r batch head sn limit u) acc)
  u'.setReg "m_i" .real [p.BM] mx

theorem load_executes (p : Layout) (r : Regions) (batch head : Nat) (sn limit : Int) (u : BlockState)
    (hb : u.regs .nat [] "cur_batch" = some (Tile.scalar batch))
    (hh : u.regs .nat [] "cur_kv_head" = some (Tile.scalar head))
    (hs : u.regs .int [] "start_n" = some (Tile.scalar sn))
    (hl : u.regs .int [] "block_end_loc" = some (Tile.scalar limit))
    (hn : u.regs .nat [p.BN] "offs_n" = some (Tile.vec (fun j => j.val)))
    (hd : u.regs .nat [p.D] "offs_d" = some (Tile.vec (fun d => d.val))) :
    stepStmts ((values p r).take 2) u = some (loadedState p r batch head sn limit u) := by
  have hz : (0.0 : ℝ) = 0 := by norm_num
  simp [values,valueOffsets,valueMask,valueView,valueChannel,dimView,lanes,dimensions,n,z,loadedState,loaded,address,
    stepStmts,stepStmt,evalOp.eq_def,hb,hh,hs,hl,hn,hd,hz,
    tile_elementwise,Tile.remap,Tile.scalar,Tile.vec,Tile.mat,Broadcast.leftIndex,Broadcast.rightIndex,
    Option.bind_eq_bind,Option.bind,apply_ite,Nat.cast_add,Nat.cast_mul,-Tile.scalar_eta]
  rfl

theorem executes (p : Layout) (r : Regions) (batch head : Nat) (sn limit : Int)
    (prob : Tile .real [p.BM,p.BN]) (mx : Tile .real [p.BM]) (acc : Tile .real [p.BM,p.D]) (u : BlockState)
    (hb : u.regs .nat [] "cur_batch" = some (Tile.scalar batch))
    (hh : u.regs .nat [] "cur_kv_head" = some (Tile.scalar head))
    (hs : u.regs .int [] "start_n" = some (Tile.scalar sn))
    (hl : u.regs .int [] "block_end_loc" = some (Tile.scalar limit))
    (hn : u.regs .nat [p.BN] "offs_n" = some (Tile.vec (fun j => j.val)))
    (hd : u.regs .nat [p.D] "offs_d" = some (Tile.vec (fun d => d.val)))
    (hp : u.regs .real [p.BM,p.BN] "p" = some prob)
    (hm : u.regs .real [p.BM] "m_ij" = some mx)
    (ha : u.regs .real [p.BM,p.D] "acc" = some acc) :
    stepStmts (values p r) u = some (prepared p r batch head sn limit prob mx acc u) := by
  let s1 := loadedState p r batch head sn limit u
  let s2 := s1.setReg "p" .real [p.BM,p.BN] prob
  let s3 := s2.setReg "acc" .real [p.BM,p.D] (updated p prob (loaded p r batch head sn limit u) acc)
  have hp1 : evalOp (Op.ref .real [p.BM,p.BN] "p") s1 = some prob := by
    simp [s1,loadedState,hp]
  have ha2 := ctxAcc2_op_eval s2 acc prob (loaded p r batch head sn limit u)
    (by simp [s2]) (by simp [s2,s1,loadedState]) (by simp [s2,s1,loadedState,ha])
  have hm3 : evalOp (Op.ref .real [p.BM] "m_ij") s3 = some mx := by
    simp [s3,s2,s1,loadedState,hm]
  rw [← List.take_append_drop 2 (values p r),stepStmts.append_some (load_executes p r batch head sn limit u hb hh hs hl hn hd)]
  change stepStmts _ s1 = _
  simp only [values,List.drop_succ_cons,List.drop_zero]
  unfold mat vec
  rw [stepStmts.cons_some (stepStmt_assign_eq_some hp1)]
  change stepStmts _ s2 = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (dtype := .real) (shape := [p.BM,p.D]) (name := "acc") ha2)]
  change stepStmts _ s3 = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some hm3)]
  simp only [stepStmts.nil,prepared,s3,s2,s1]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnFwd.ValuesExecution
