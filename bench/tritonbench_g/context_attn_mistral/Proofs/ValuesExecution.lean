import bench.tritonbench_g.context_attn_mistral.Proofs.LoopProgram
namespace VeriTile.Bench.TritonBenchG.ContextAttnMistral.ValuesExecution
open VeriTile.Triton Formula Program PreludeProgram LoopProgram
noncomputable section
set_option maxHeartbeats 1500000
set_option maxRecDepth 5000
set_option linter.unusedSimpArgs false

/-- Actual address of the stored pointer plus the signed row advance. The later
source contract also checks that this advance is nonnegative before conversion. -/
def address (p : Layout) (head : Nat) (start sn : Int) (d : Fin p.D) (j : Fin p.BN) : Nat :=
  j.val*p.vRow+head*p.vHead+d.val*p.vDim+((start+sn)*p.vRow).toNat

def loaded (p : Layout) (r : Regions) (head : Nat) (start sn limit : Int) (u : BlockState) : Tile .real [p.BN,p.D] :=
  Tile.mat (fun j d => some (if sn+j.val < limit then
    u.readMem r.value (address p head start sn d j) else 0))

def prepared (p : Layout) (r : Regions) (head : Nat) (start sn limit : Int) (u : BlockState) :=
  u.setReg "v" .real [p.BN,p.D] (loaded p r head start sn limit u)

theorem executes (p : Layout) (r : Regions) (head : Nat) (start sn limit : Int) (u : BlockState)
    (hp : u.regs .ptr [p.BN,p.D] "v_ptrs" = some (Tile.mat (fun j d =>
      (Region.cast r.value,j.val*p.vRow+head*p.vHead+d.val*p.vDim))))
    (hs : u.regs .int [] "start_n" = some (Tile.scalar sn))
    (hb : u.regs .int [] "cur_batch_in_all_start_index" = some (Tile.scalar start))
    (hl : u.regs .int [] "cur_batch_seq_len" = some (Tile.scalar limit))
    (hn : u.regs .nat [p.BN] "offs_n" = some (Tile.vec (fun j => j.val))) :
    stepStmts (values p r) u = some (prepared p r head start sn limit u) := by
  have hz : (0.0 : ℝ) = 0 := by norm_num
  simp [values,valueAdvance,valueMask,valueView,lanes,n,z,prepared,loaded,address,
    stepStmts,stepStmt,evalOp.eq_def,hp,hs,hb,hl,hn,hz,
    tile_elementwise,Tile.remap,Tile.scalar,Tile.vec,Tile.mat,Broadcast.leftIndex,Broadcast.rightIndex,
    Option.bind_eq_bind,Option.bind,apply_ite,Nat.cast_add,Nat.cast_mul,-Tile.scalar_eta]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnMistral.ValuesExecution
