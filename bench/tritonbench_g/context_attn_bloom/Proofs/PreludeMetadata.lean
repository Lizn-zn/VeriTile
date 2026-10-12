import bench.tritonbench_g.context_attn_bloom.Proofs.PreludeLaunch
namespace VeriTile.Bench.TritonBenchG.ContextAttnBloom.PreludeMetadata
open VeriTile.Triton Math Program RealSetup PreludeProgram
noncomputable section
set_option maxHeartbeats 500000
set_option maxRecDepth 2000
set_option linter.unusedSimpArgs false

def prepared (r : Regions) (batch : Nat) (u : BlockState) : BlockState :=
  let a := inputs u r
  let u := u.setReg "cur_batch_in_all_start_index" .int [] (Tile.scalar (a.start batch))
  let u := u.setReg "prompt_cache_len" .int [] (Tile.scalar (a.prompt batch))
  let u := u.setReg "cur_batch_seq_len" .int [] (Tile.scalar (a.sequence batch-a.prompt batch))
  u.setReg "cur_batch_req_idx" .int [] (Tile.scalar (a.request batch))

theorem executes (r : Regions) (batch : Nat) (u : BlockState)
    (hb : u.regs .nat [] "cur_batch" = some (Tile.scalar batch)) :
    stepStmts (metadataCode r) u = some (prepared r batch u) := by
  simp [metadataCode,prepared,inputs,loadMeta,n,z,stepStmts,stepStmt,evalOp.eq_def,hb,
    tile_elementwise,Tile.scalar,Broadcast.leftIndex,Broadcast.rightIndex,
    Option.bind_eq_bind,Option.bind,-Tile.scalar_eta]

theorem memory (r : Regions) (batch : Nat) (u : BlockState) :
    (prepared r batch u).mem = u.mem := by
  funext region off
  simp only [prepared,BlockState.setReg_mem]
theorem pids (r : Regions) (batch : Nat) (u : BlockState) :
    (prepared r batch u).pids = u.pids := by
  simp only [prepared,BlockState.setReg_pids]
theorem start (r : Regions) (batch : Nat) (u : BlockState) :
    (prepared r batch u).regs .int [] "cur_batch_in_all_start_index" =
      some (Tile.scalar ((inputs u r).start batch)) := by simp [prepared]
theorem prompt (r : Regions) (batch : Nat) (u : BlockState) :
    (prepared r batch u).regs .int [] "prompt_cache_len" =
      some (Tile.scalar ((inputs u r).prompt batch)) := by simp [prepared]
theorem sequence (r : Regions) (batch : Nat) (u : BlockState) :
    (prepared r batch u).regs .int [] "cur_batch_seq_len" =
      some (Tile.scalar ((inputs u r).sequence batch-(inputs u r).prompt batch)) := by simp [prepared]
theorem unchanged (r : Regions) (batch : Nat) (u : BlockState) (dtype : TileDType)
    (shape : TileShape) (name : RegName)
    (h1 : name ≠ "cur_batch_req_idx") (h2 : name ≠ "cur_batch_seq_len")
    (h3 : name ≠ "prompt_cache_len") (h4 : name ≠ "cur_batch_in_all_start_index") :
    (prepared r batch u).regs dtype shape name = u.regs dtype shape name := by
  simp [prepared,h1,h2,h3,h4]
end
end VeriTile.Bench.TritonBenchG.ContextAttnBloom.PreludeMetadata
