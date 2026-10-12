import bench.tritonbench_g.context_attn_fwd.Proofs.PreludeProgram

namespace VeriTile.Bench.TritonBenchG.ContextAttnFwd.PreludeLaunch
open VeriTile.Triton Formula Program PreludeProgram
noncomputable section
set_option maxHeartbeats 500000
set_option maxRecDepth 2000
set_option linter.unusedSimpArgs false

def launched (p : Layout) (u : BlockState) : BlockState :=
  let l := launch u
  let u := u.setReg "start_m" .nat [] (Tile.scalar l.block)
  let u := u.setReg "cur_bh" .nat [] (Tile.scalar l.batchHead)
  let u := u.setReg "cur_batch" .nat [] (Tile.scalar (Formula.batch p l))
  let u := u.setReg "cur_head" .nat [] (Tile.scalar (Formula.head p l))
  u.setReg "cur_kv_head" .nat [] (Tile.scalar (Formula.head p l/p.kvGroup))

theorem executes (p : Layout) (u : BlockState) :
    stepStmts (launchCode p) u = some (launched p u) := by
  simp [launchCode,launched,launch,Formula.batch,Formula.head,n,stepStmts,stepStmt,evalOp.eq_def,
    tile_elementwise,Tile.scalar,Broadcast.leftIndex,Broadcast.rightIndex,
    Option.bind_eq_bind,Option.bind,-Tile.scalar_eta]

theorem memory (p : Layout) (u : BlockState) : (launched p u).mem = u.mem := by
  funext region off
  simp only [launched,BlockState.setReg_mem]
theorem pids (p : Layout) (u : BlockState) : (launched p u).pids = u.pids := by
  simp only [launched,BlockState.setReg_pids]
theorem batch (p : Layout) (u : BlockState) :
    (launched p u).regs .nat [] "cur_batch" = some (Tile.scalar (u.pids 1/p.heads)) := by
  simp [launched,launch,Formula.batch,Formula.head]
theorem head (p : Layout) (u : BlockState) :
    (launched p u).regs .nat [] "cur_head" = some (Tile.scalar (u.pids 1%p.heads)) := by
  simp [launched,launch,Formula.batch,Formula.head]
theorem block (p : Layout) (u : BlockState) :
    (launched p u).regs .nat [] "start_m" = some (Tile.scalar (u.pids 0)) := by
  simp [launched,launch,Formula.batch,Formula.head]
end
end VeriTile.Bench.TritonBenchG.ContextAttnFwd.PreludeLaunch

namespace VeriTile.Bench.TritonBenchG.ContextAttnFwd.PreludeMetadata
open VeriTile.Triton Formula Program PreludeProgram
noncomputable section
set_option maxHeartbeats 500000
set_option maxRecDepth 2000
set_option linter.unusedSimpArgs false

def prepared (r : Regions) (batch : Nat) (u : BlockState) : BlockState :=
  let a := inputs u r
  let u := u.setReg "prompt_cache_len" .int [] (Tile.scalar (a.prompt batch))
  let u := u.setReg "cur_batch_in_all_start_index" .int [] (Tile.scalar (a.start batch))
  u.setReg "cur_batch_seq_len" .int [] (Tile.scalar (a.sequence batch-a.prompt batch))

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
    (h2 : name ≠ "cur_batch_seq_len")
    (h3 : name ≠ "prompt_cache_len") (h4 : name ≠ "cur_batch_in_all_start_index") :
    (prepared r batch u).regs dtype shape name = u.regs dtype shape name := by
  simp [prepared,h2,h3,h4]
end
end VeriTile.Bench.TritonBenchG.ContextAttnFwd.PreludeMetadata

namespace VeriTile.Bench.TritonBenchG.ContextAttnFwd.PreludeOffsets
open VeriTile.Triton Formula Program PreludeProgram
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
    Nat.cast_add,Nat.cast_mul,Nat.mul_comm,-Tile.scalar_eta]

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
end VeriTile.Bench.TritonBenchG.ContextAttnFwd.PreludeOffsets

namespace VeriTile.Bench.TritonBenchG.ContextAttnFwd.PreludeQuery
open VeriTile.Triton Formula Program PreludeProgram
noncomputable section
set_option maxHeartbeats 600000
set_option maxRecDepth 2000
set_option linter.unusedSimpArgs false

def queried (p : Layout) (r : Regions) (head block : Nat) (start seq prompt : Int)
    (u : BlockState) : BlockState :=
  let q := Tile.mat (fun (i : Fin p.BM) (d : Fin p.D) => some
    (if ((block*p.BM+i.val : Nat) : Int) < seq then
      u.readMem r.query ((start+((block*p.BM+i.val : Nat) : Int))*p.qRow+head*p.qHead+d.val*p.qDim).toNat
     else 0))
  let u := u.setReg "q" .real [p.BM,p.D] q
  let u := u.setReg "m_i" .real [p.BM] (Tile.vec (fun _ => ⊥))
  let u := u.setReg "l_i" .real [p.BM] (Tile.vec (fun _ => some 0))
  let u := u.setReg "acc" .real [p.BM,p.D] (Tile.mat (fun _ _ => some 0))
  let u := u.setReg "block_mask" .nat [] (Tile.scalar (if ((p.BM*block : Nat) : Int) < seq then 1 else 0))
  u.setReg "block_end_loc" .int [] (Tile.scalar (min ((block+1)*p.BM+prompt) (seq+prompt)))

theorem executes (p : Layout) (r : Regions) (head block : Nat) (start seq prompt : Int)
    (u : BlockState)
    (ho : u.regs .int [p.BM,p.D] "off_q" = some (Tile.mat (dtype := .int) (fun i d =>
      (start+((block*p.BM+i.val : Nat) : Int))*p.qRow+head*p.qHead+d.val*p.qDim)))
    (hr : u.regs .nat [p.BM] "offs_m" = some (Tile.vec (fun i => block*p.BM+i.val)))
    (hb : u.regs .nat [] "block_start_loc" = some (Tile.scalar (p.BM*block)))
    (hs : u.regs .int [] "cur_batch_seq_len" = some (Tile.scalar seq))
    (hp : u.regs .int [] "prompt_cache_len" = some (Tile.scalar prompt)) :
    stepStmts (queryCode p r) u = some (queried p r head block start seq prompt u) := by
  have hz : (0.0 : ℝ) = 0 := by norm_num
  simp [queryCode,queried,loadQuery,queryMask,ending,endCandidate,fullLength,n,z,
    rowView,dimView,rows,dimensions,stepStmts,stepStmt,evalOp.eq_def,ho,hr,hb,hs,hp,hz,
    tile_elementwise,Tile.remap,Tile.scalar,Tile.vec,Tile.mat,Broadcast.leftIndex,Broadcast.rightIndex,
    Option.bind_eq_bind,Option.bind,apply_ite,min_def_lt,Nat.cast_add,Nat.cast_mul,Nat.mul_comm,add_mul,one_mul,-Tile.scalar_eta]
  all_goals rfl
end
end VeriTile.Bench.TritonBenchG.ContextAttnFwd.PreludeQuery
