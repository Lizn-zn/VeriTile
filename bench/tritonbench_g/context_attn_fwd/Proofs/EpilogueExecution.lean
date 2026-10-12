import bench.tritonbench_g.context_attn_fwd.Proofs.EpilogueProgram
namespace VeriTile.Bench.TritonBenchG.ContextAttnFwd.EpilogueExecution
open VeriTile.Triton Formula Program PreludeProgram
noncomputable section
set_option maxHeartbeats 600000
set_option maxRecDepth 3000
set_option linter.unusedSimpArgs false

def address (p : Layout) (head : Nat) (start : Int) (rows : Fin p.BM → Nat)
    (i : TileIndex [p.BM,p.D]) : Nat :=
  ((start+(rows i.1 : Int))*p.outRow+head*p.outHead+i.2.1.val*p.outDim).toNat

def active (p : Layout) (seq : Int) (rows : Fin p.BM → Nat)
    (i : TileIndex [p.BM,p.D]) : Bool :=
  decide ((rows i.1 : Int) < seq)

def addresses (p : Layout) (r : Regions) (head : Nat) (start : Int)
    (rows : Fin p.BM → Nat) (u : BlockState) : BlockState :=
  let u := u.setReg "off_o" .int [p.BM,p.D] (Tile.mat (fun i d =>
    (start+(rows i : Int))*p.outRow+head*p.outHead+d.val*p.outDim))
  u.setReg "out_ptrs" .ptr [p.BM,p.D]
    { data := fun i => (r.out,address p head start rows i) }

def stored (p : Layout) (r : Regions) (head : Nat) (start seq : Int)
    (rows : Fin p.BM → Nat) (v : Fin p.BM → Fin p.D → ℝ) (u : BlockState) : BlockState :=
  (TileShape.allIndices [p.BM,p.D]).foldl
    (fun s i => if active p seq rows i then
      s.writeMem r.out (address p head start rows i) (v i.1 i.2.1) else s) u

def normalized (p : Layout) (v : Fin p.BM → Fin p.D → ℝ) (den : Fin p.BM → ℝ) (u : BlockState) :=
  u.setReg "acc" .real [p.BM,p.D] (Tile.mat (fun i d => some (v i d / den i)))

def prepared (p : Layout) (r : Regions) (head : Nat) (start seq : Int)
    (rows : Fin p.BM → Nat) (v : Fin p.BM → Fin p.D → ℝ) (den : Fin p.BM → ℝ) (u : BlockState) : BlockState :=
  stored p r head start seq rows (fun i d => v i d / den i)
    (addresses p r head start rows (normalized p v den u))

theorem executes (p : Layout) (r : Regions) (head : Nat) (start seq : Int)
    (rows : Fin p.BM → Nat) (v : Fin p.BM → Fin p.D → ℝ) (den : Fin p.BM → ℝ) (u : BlockState)
    (hh : u.regs .nat [] "cur_head" = some (Tile.scalar head))
    (hs : u.regs .int [] "cur_batch_in_all_start_index" = some (Tile.scalar start))
    (hl : u.regs .int [] "cur_batch_seq_len" = some (Tile.scalar seq))
    (hr : u.regs .nat [p.BM] "offs_m" = some (Tile.vec rows))
    (hd : u.regs .nat [p.D] "offs_d" = some (Tile.vec (fun d => d.val)))
    (hden : u.regs .real [p.BM] "l_i" = some (Tile.vec (fun i => some (den i))))
    (hv : u.regs .real [p.BM,p.D] "acc" = some (Tile.mat (fun i d => some (v i d)))) :
    stepStmts (EpilogueProgram.post p r) u = some (prepared p r head start seq rows v den u) := by
  rw [EpilogueProgram.source]
  simp [EpilogueProgram.code,EpilogueProgram.offsets,prepared,normalized,addresses,stored,address,active,
    queryMask,rowView,dimView,PreludeProgram.rows,dimensions,n,z,
    stepStmts,stepStmt,evalOp.eq_def,hh,hs,hl,hr,hd,hv,hden,
    tile_elementwise,Tile.remap,Tile.scalar,Tile.vec,Tile.mat,Broadcast.leftIndex,Broadcast.rightIndex,
    Option.bind_eq_bind,Option.bind,Nat.cast_mul,-Tile.scalar_eta]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnFwd.EpilogueExecution
