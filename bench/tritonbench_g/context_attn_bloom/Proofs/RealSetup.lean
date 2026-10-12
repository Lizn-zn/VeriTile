import bench.tritonbench_g.context_attn_bloom.Proofs.PrologueAlignment
namespace VeriTile.Bench.TritonBenchG.ContextAttnBloom.RealSetup
open VeriTile.Triton Math Program
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def setupScalar (p : Layout) (r : Regions) (u : BlockState) : BlockState :=
  let a := inputs u r
  let l := launch u
  let u := u.setReg "cur_batch" .nat [] (Tile.scalar l.batch)
  let u := u.setReg "cur_head" .nat [] (Tile.scalar l.head)
  let u := u.setReg "start_m" .nat [] (Tile.scalar l.block)
  let u := u.setReg "cur_kv_head" .nat [] (Tile.scalar (l.head/p.kvGroup))
  let u := u.setReg "cur_batch_in_all_start_index" .int [] (Tile.scalar (a.start l.batch))
  let u := u.setReg "prompt_cache_len" .int [] (Tile.scalar (a.prompt l.batch))
  let u := u.setReg "cur_batch_seq_len" .int [] (Tile.scalar (sequence a l))
  let u := u.setReg "cur_batch_req_idx" .int [] (Tile.scalar (a.request l.batch))
  let u := u.setReg "block_start_loc" .nat [] (Tile.scalar (p.BM*l.block))
  let u := u.setReg "offs_n" .nat [p.BN] (Tile.vec (fun j => j.val))
  let u := u.setReg "offs_d" .nat [p.D] (Tile.vec (fun j => j.val))
  let u := u.setReg "offs_m" .nat [p.BM] (Tile.vec (fun i => queryRow p l i.val))
  let u := u.setReg "off_q" .int [p.BM,p.D] (Tile.mat (fun i d =>
    (a.start l.batch+queryRow p l i.val)*p.qRow+l.head*p.qHead+d.val*p.qDim))
  u

def initialized (p : Layout) (r : Regions) (u : BlockState) : BlockState :=
  let a := inputs u r
  let l := launch u
  let u := setupScalar p r u
  let u := u.setReg "q" .real [p.BM,p.D] (Tile.mat (fun i d => some (query p l a i.val d.val)))
  let u := u.setReg "m_i" .real [p.BM] (Tile.vec (fun _ => ⊥))
  let u := u.setReg "l_i" .real [p.BM] (Tile.vec (fun _ => some 0))
  let u := u.setReg "acc" .real [p.BM,p.D] (Tile.mat (fun _ _ => some 0))
  let u := u.setReg "block_mask" .nat [] (Tile.scalar (if (p.BM*l.block : Int)<sequence a l then 1 else 0))
  u.setReg "block_end_loc" .int [] (Tile.scalar (limit p l a))

theorem initialized_memory (p : Layout) (r : Regions) (u : BlockState) :
    (initialized p r u).mem = u.mem := by
  funext region off
  simp only [initialized,setupScalar,BlockState.setReg_mem]

theorem initialized_pids (p : Layout) (r : Regions) (u : BlockState) :
    (initialized p r u).pids = u.pids := by
  simp only [initialized,setupScalar,BlockState.setReg_pids]

end
end VeriTile.Bench.TritonBenchG.ContextAttnBloom.RealSetup
