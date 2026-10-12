import bench.tritonbench_g.context_attn_bloom.Proofs.PreludeMetadata
namespace VeriTile.Bench.TritonBenchG.ContextAttnBloom.PreludeOffsets
open VeriTile.Triton Math Program RealSetup PreludeProgram
noncomputable section
set_option maxHeartbeats 600000
set_option maxRecDepth 2000
set_option linter.unusedSimpArgs false

def prepared (p : Layout) (head block : Nat) (start : Int) (u : BlockState) : BlockState :=
  let u := u.setReg "block_start_loc" .nat [] (Tile.scalar (p.BM*block))
  let u := u.setReg "offs_n" .nat [p.BN] (Tile.vec (fun j => j.val))
  let u := u.setReg "offs_d" .nat [p.D] (Tile.vec (fun j => j.val))
  let u := u.setReg "offs_m" .nat [p.BM] (Tile.vec (fun i => block*p.BM+i.val))
  u.setReg "off_q" .int [p.BM,p.D] (Tile.mat (fun i d =>
    (start+(block*p.BM+i.val : Nat))*p.qRow+head*p.qHead+d.val*p.qDim))

theorem executes (p : Layout) (head block : Nat) (start : Int) (u : BlockState)
    (hh : u.regs .nat [] "cur_head" = some (Tile.scalar head))
    (hb : u.regs .nat [] "start_m" = some (Tile.scalar block))
    (hs : u.regs .int [] "cur_batch_in_all_start_index" = some (Tile.scalar start)) :
    stepStmts (offsetCode p) u = some (prepared p head block start u) := by
  simp [offsetCode,prepared,offset,n,z,rowView,dimView,rows,dimensions,stepStmts,stepStmt,
    evalOp.eq_def,hh,hb,hs,tile_elementwise,Tile.scalar,Tile.vec,Tile.mat,
    Broadcast.leftIndex,Broadcast.rightIndex,Option.bind_eq_bind,Option.bind,
    Nat.cast_add,Nat.cast_mul,-Tile.scalar_eta]

theorem memory (p : Layout) (head block : Nat) (start : Int) (u : BlockState) :
    (prepared p head block start u).mem = u.mem := by
  funext region off
  simp only [prepared,BlockState.setReg_mem]

theorem off_q (p : Layout) (head block : Nat) (start : Int) (u : BlockState) :
    (prepared p head block start u).regs .int [p.BM,p.D] "off_q" =
      some (Tile.mat (dtype := .int) (fun i d => (start+((block*p.BM+i.val : Nat) : Int))*p.qRow+head*p.qHead+d.val*p.qDim)) := by
  simp [prepared]

theorem rows (p : Layout) (head block : Nat) (start : Int) (u : BlockState) :
    (prepared p head block start u).regs .nat [p.BM] "offs_m" =
      some (Tile.vec (fun i => block*p.BM+i.val)) := by simp [prepared]
theorem dims (p : Layout) (head block : Nat) (start : Int) (u : BlockState) :
    (prepared p head block start u).regs .nat [p.D] "offs_d" =
      some (Tile.vec (fun d => d.val)) := by simp [prepared]
theorem block_start (p : Layout) (head block : Nat) (start : Int) (u : BlockState) :
    (prepared p head block start u).regs .nat [] "block_start_loc" =
      some (Tile.scalar (p.BM*block)) := by simp [prepared]
theorem unchanged (p : Layout) (head block : Nat) (start : Int) (u : BlockState)
    (dtype : TileDType) (shape : TileShape) (name : RegName)
    (h1 : name ≠ "off_q") (h2 : name ≠ "offs_m") (h3 : name ≠ "offs_d")
    (h4 : name ≠ "offs_n") (h5 : name ≠ "block_start_loc") :
    (prepared p head block start u).regs dtype shape name = u.regs dtype shape name := by
  simp [prepared,h1,h2,h3,h4,h5]
end
end VeriTile.Bench.TritonBenchG.ContextAttnBloom.PreludeOffsets
