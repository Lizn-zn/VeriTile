import bench.tritonbench_g.flash_attn.Proofs.RealLoop
namespace VeriTile.Bench.TritonBenchG.FlashAttn.RealExecution
open VeriTile.Triton Formula NormalizedSource
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def post (Q K V L O : RegionName) (p : Layout) (s : BlockState)
    (startm bshead base : Nat) (offm : Tile .nat [p.BLOCK_M])
    (m den : Tile .real [p.BLOCK_M]) (acc : Tile .real [p.BLOCK_M,p.DIM])
    (hsm : s.regs .nat [] "start_m"=some (Tile.scalar startm))
    (hb : s.regs .nat [] "off_bs_head"=some (Tile.scalar bshead))
    (hbase : s.regs .nat [] "qkv_base_offset"=some (Tile.scalar base))
    (hom : s.regs .nat [p.BLOCK_M] "off_m"=some offm)
    (hm : s.regs .real [p.BLOCK_M] "max"=some m)
    (hd : s.regs .real [p.BLOCK_M] "denom"=some den)
    (ha : s.regs .real [p.BLOCK_M,p.DIM] "out_buffer"=some acc) :
    {t : BlockState // stepStmts (after Q K V L O p) s=some t} := by
  refine ⟨?_,?_⟩
  swap
  simp only [after,algorithm,reduced,List.drop_succ_cons,List.drop_zero]
  simp (maxSteps := 300000) [stepStmts,stepStmt,evalOp,Option.bind,Region.cast,
    hsm,hb,hbase,hom,hm,hd,ha]
  simp (maxSteps := 300000) [stepStmts,stepStmt,↓evalOp.eq_def,Option.bind,Region.cast,
    hsm,hb,hbase,hom,hm,hd,ha]
  try rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FlashAttn.RealExecution
