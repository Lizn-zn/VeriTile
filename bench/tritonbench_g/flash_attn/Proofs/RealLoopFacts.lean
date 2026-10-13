import bench.tritonbench_g.flash_attn.Proofs.RealLoop
namespace VeriTile.Bench.TritonBenchG.FlashAttn.RealExecution
open VeriTile.Triton Formula NormalizedSource
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

section
variable (p : Layout) (s : BlockState) (sn : Nat)
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
    (hvp : s.regs .blockPtr [p.BLOCK_N,p.DIM] "V_block_ptr"=some vp)

theorem iteration_frame :
    (iteration p s sn offm offn q m den acc kp vp hBN hsn hom hon hq hm hd ha hkp hvp).val.mem=s.mem ∧
    (iteration p s sn offm offn q m den acc kp vp hBN hsn hom hon hq hm hd ha hkp hvp).val.pids=s.pids ∧
    (iteration p s sn offm offn q m den acc kp vp hBN hsn hom hon hq hm hd ha hkp hvp).val.undef=s.undef := by
  rcases p with ⟨sqh,sqs,sqd,sks,skd,svd,sos,sod,seq,bm,dim,bn,c,scale⟩
  cases c <;> simp [iteration]
  all_goals
    constructor
    · funext r o
      simp only [↓BlockState.setReg_mem]
    · funext r o
      simp only [↓BlockState.setReg_undef]

theorem iteration_preserves :
    (iteration p s sn offm offn q m den acc kp vp hBN hsn hom hon hq hm hd ha hkp hvp).val.regs .real [p.BLOCK_M,p.DIM] "q"=some q ∧
    (iteration p s sn offm offn q m den acc kp vp hBN hsn hom hon hq hm hd ha hkp hvp).val.regs .nat [p.BLOCK_M] "off_m"=some offm ∧
    (iteration p s sn offm offn q m den acc kp vp hBN hsn hom hon hq hm hd ha hkp hvp).val.regs .nat [p.BLOCK_N] "off_n"=some offn := by
  rcases p with ⟨sqh,sqs,sqd,sks,skd,svd,sos,sod,seq,bm,dim,bn,c,scale⟩
  cases c <;> simp [iteration,hq,hom,hon]

theorem iteration_pointers :
    (iteration p s sn offm offn q m den acc kp vp hBN hsn hom hon hq hm hd ha hkp hvp).val.regs .blockPtr [p.DIM,p.BLOCK_N] "K_block_ptr"=
      some ⟨fun i => (kp.data i).advance [0,Int.ofNat p.BLOCK_N]⟩ ∧
    (iteration p s sn offm offn q m den acc kp vp hBN hsn hom hon hq hm hd ha hkp hvp).val.regs .blockPtr [p.BLOCK_N,p.DIM] "V_block_ptr"=
      some ⟨fun i => (vp.data i).advance [Int.ofNat p.BLOCK_N,0]⟩ := by
  rcases p with ⟨sqh,sqs,sqd,sks,skd,svd,sos,sod,seq,bm,dim,bn,c,scale⟩
  cases c <;> simp [iteration]

/-- The body does not overwrite the scalar launch indices or loop bounds. -/
theorem iteration_nat_scalars :
    let t := (iteration p s sn offm offn q m den acc kp vp
      hBN hsn hom hon hq hm hd ha hkp hvp).val
    t.regs .nat [] "start_m" = s.regs .nat [] "start_m" ∧
    t.regs .nat [] "off_bs_head" = s.regs .nat [] "off_bs_head" ∧
    t.regs .nat [] "qkv_base_offset" = s.regs .nat [] "qkv_base_offset" ∧
    t.regs .nat [] "lo" = s.regs .nat [] "lo" ∧
    t.regs .nat [] "hi" = s.regs .nat [] "hi" := by
  rcases p with ⟨sqh,sqs,sqd,sks,skd,svd,sos,sod,seq,bm,dim,bn,c,scale⟩
  cases c <;> simp [iteration]

end
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FlashAttn.RealExecution
