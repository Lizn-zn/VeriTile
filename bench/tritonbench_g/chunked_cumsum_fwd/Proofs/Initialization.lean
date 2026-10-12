import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.Program
import VeriTile.Triton.Semantics.BroadcastReshape

namespace VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.Initialization
open VeriTile.Triton Math Program
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def launch (u : BlockState) : Launch := ⟨u.pids 0,u.pids 1,u.pids 2⟩
def heads (p : Layout) (l : Launch) : Tile .nat [p.BH] := Tile.vec (fun i => head p l i.val)
def columns (p : Layout) : Tile .nat [p.BC] := Tile.vec (fun j => j.val)
def dtPointers (p : Layout) (l : Launch) (r : Regions) : Tile .ptr [p.BH,p.BC] :=
  Tile.mat (fun i j => (r.dt,dtAddress p l i.val j.val))
def outPointers (p : Layout) (l : Launch) (r : Regions) : Tile .ptr [p.BH,p.BC] :=
  Tile.mat (fun i j => (r.out,outAddress p l i.val j.val))
def sumPointers (p : Layout) (l : Launch) (r : Regions) : Tile .ptr [p.BH,p.BC] :=
  Tile.mat (fun i j => (r.sum,sumAddress p l i.val j.val))
def ratePointers (p : Layout) (l : Launch) (r : Regions) : Tile .ptr [p.BH] :=
  Tile.vec (fun i => (r.rate,head p l i.val*p.rateHead))
def initial (p : Layout) (r : Regions) (u : BlockState) : BlockState :=
  let l := launch u
  let u := u.setReg "pid_b" .nat [] (Tile.scalar l.batch)
  let u := u.setReg "pid_c" .nat [] (Tile.scalar l.chunk)
  let u := u.setReg "pid_h" .nat [] (Tile.scalar l.head)
  let u := u.setReg "dt_ptr" .ptr [] (Tile.scalar (r.dt,l.batch*p.dtBatch+l.chunk*p.chunk*p.dtTime))
  let u := u.setReg "dt_out_ptr" .ptr [] (Tile.scalar (r.out,l.batch*p.outBatch+l.chunk*p.outChunk))
  let u := u.setReg "dA_cumsum_ptr" .ptr [] (Tile.scalar (r.sum,l.batch*p.sumBatch+l.chunk*p.sumChunk))
  let u := u.setReg "offs_h" .nat [p.BH] (heads p l)
  let u := u.setReg "offs_c" .nat [p.BC] (columns p)
  let u := u.setReg "dt_ptrs" .ptr [p.BH,p.BC] (dtPointers p l r)
  let u := u.setReg "A_ptrs" .ptr [p.BH] (ratePointers p l r)
  let u := u.setReg "dt_out_ptrs" .ptr [p.BH,p.BC] (outPointers p l r)
  let u := u.setReg "dA_cs_ptrs" .ptr [p.BH,p.BC] (sumPointers p l r)
  u.setReg "chunk_size_limit" .nat [] (Tile.scalar (min p.chunk (p.sequence-l.chunk*p.chunk)))

theorem executes (p : Layout) (r : Regions) (u : BlockState) :
    stepStmts (initialization p r) u = some (initial p r u) := by
  simp [initialization,initial,stepStmts,stepStmt,evalOp.eq_def,
    offset,dtOffset,pointer,limit,remaining,headOp,columnOp,heads,columns,launch,
    dtPointers,outPointers,sumPointers,ratePointers,dtAddress,outAddress,sumAddress,head,
    tile_elementwise,Tile.scalar,Tile.vec,Tile.mat,Broadcast.leftIndex,Broadcast.rightIndex,
    min_def_lt,Option.bind_eq_bind,Option.bind,Option.bind_some,-Tile.scalar_eta]

end
end VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.Initialization
