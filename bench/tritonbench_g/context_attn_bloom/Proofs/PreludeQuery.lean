import bench.tritonbench_g.context_attn_bloom.Proofs.PreludeOffsets
namespace VeriTile.Bench.TritonBenchG.ContextAttnBloom.PreludeQuery
open VeriTile.Triton Math Program RealSetup PreludeProgram
noncomputable section
set_option maxHeartbeats 600000
set_option maxRecDepth 2000
set_option linter.unusedSimpArgs false

def queried (p : Layout) (r : Regions) (head block : Nat) (start seq prompt : Int)
    (u : BlockState) : BlockState :=
  let q := Tile.mat (fun (i : Fin p.BM) (d : Fin p.D) => some
    (if ((block*p.BM+i.val : Nat) : Int) < seq ∧ d.val < p.headDim then
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
    (hd : u.regs .nat [p.D] "offs_d" = some (Tile.vec (fun d => d.val)))
    (hb : u.regs .nat [] "block_start_loc" = some (Tile.scalar (p.BM*block)))
    (hm : u.regs .nat [] "start_m" = some (Tile.scalar block))
    (hs : u.regs .int [] "cur_batch_seq_len" = some (Tile.scalar seq))
    (hp : u.regs .int [] "prompt_cache_len" = some (Tile.scalar prompt)) :
    stepStmts (queryCode p r) u = some (queried p r head block start seq prompt u) := by
  have hz : (0.0 : ℝ) = 0 := by norm_num
  simp [queryCode,queried,loadQuery,queryMask,ending,endCandidate,fullLength,n,z,
    rowView,dimView,rows,dimensions,stepStmts,stepStmt,evalOp.eq_def,ho,hr,hd,hb,hm,hs,hp,hz,
    tile_elementwise,Tile.scalar,Tile.vec,Tile.mat,Broadcast.leftIndex,Broadcast.rightIndex,
    Option.bind_eq_bind,Option.bind,apply_ite,min_def_lt,Nat.cast_add,Nat.cast_mul,-Tile.scalar_eta]
  rfl
end
end VeriTile.Bench.TritonBenchG.ContextAttnBloom.PreludeQuery
