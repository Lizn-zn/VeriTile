import bench.tritonbench_g.flash_attn.Proofs.RealEntry
namespace VeriTile.Bench.TritonBenchG.FlashAttn.RealExecution
open VeriTile.Triton Formula NormalizedSource
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def body (p : Layout) : List Stmt :=
  match ((algorithm "Q" "K" "V" "L" "O" p).body[16]? : Option Stmt) with
  | some (Stmt.forRangeDyn _ _ _ _ b) => b
  | _ => []

def after (Q K V L O : RegionName) (p : Layout) : List Stmt :=
  (algorithm Q K V L O p).body.drop 17

theorem decomposition (Q K V L O : RegionName) (p : Layout) :
    (algorithm Q K V L O p).body=before Q K V L O p ++
      .forRangeDyn "start_n" (.ref .nat [] "lo") (.ref .nat [] "hi") (.constNat p.BLOCK_N)
        (body p) :: after Q K V L O p := by rfl

/-- Evaluate one complete source iteration on arbitrary input memory and live
registers. The premises name the registers actually read by the loop; no
output equality or whole-program execution is assumed. -/
def iteration (p : Layout) (s : BlockState) (sn : Nat)
    (offm : Tile .nat [p.BLOCK_M]) (offn : Tile .nat [p.BLOCK_N])
    (q : Tile .real [p.BLOCK_M,p.DIM])
    (m den : Tile .real [p.BLOCK_M]) (acc : Tile .real [p.BLOCK_M,p.DIM])
    (kp : Tile .blockPtr [p.DIM,p.BLOCK_N]) (vp : Tile .blockPtr [p.BLOCK_N,p.DIM])
    (hBN : 0<p.BLOCK_N)
    (hsn : s.regs .nat [] "start_n"=some (Tile.scalar sn))
    (hom : s.regs .nat [p.BLOCK_M] "off_m"=some offm)
    (hon : s.regs .nat [p.BLOCK_N] "off_n"=some offn)
    (hq : s.regs .real [p.BLOCK_M,p.DIM] "q"=some q)
    (hm : s.regs .real [p.BLOCK_M] "max"=some m)
    (hd : s.regs .real [p.BLOCK_M] "denom"=some den)
    (ha : s.regs .real [p.BLOCK_M,p.DIM] "out_buffer"=some acc)
    (hkp : s.regs .blockPtr [p.DIM,p.BLOCK_N] "K_block_ptr"=some kp)
    (hvp : s.regs .blockPtr [p.BLOCK_N,p.DIM] "V_block_ptr"=some vp) :
    {t : BlockState // stepStmts (body p) s=some t} := by
  cases hc : p.causal
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp only [body,algorithm,reduced,List.getElem?_cons_zero,List.getElem?_cons_succ]
    simp (maxSteps := 300000) [stepStmts,stepStmt,↓evalOp.eq_def,Option.bind,Region.cast,
      hc,hsn,hom,hon,hq,hm,hd,ha,hkp,hvp,Tile.reduceMaxDrop,Tile.reduceMax,
      TileShape.axisDim,hBN]
    try rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FlashAttn.RealExecution
