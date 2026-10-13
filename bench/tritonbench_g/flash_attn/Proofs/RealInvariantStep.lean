import bench.tritonbench_g.flash_attn.Proofs.RealInvariant

/-! Preservation of the initial-memory numeric invariant by one actual source
body. No output equality or successful whole-kernel run is a premise. -/
namespace VeriTile.Bench.TritonBenchG.FlashAttn.RealExecution
open VeriTile.Triton Formula
noncomputable section
set_option maxHeartbeats 3000000

theorem PrefixInvariant.set_start_n (p : Layout) (s₀ : BlockState) (Q K V : RegionName)
    (c : Nat) (t : BlockState) (h : PrefixInvariant p s₀ Q K V c t) (n : Nat) :
    PrefixInvariant p s₀ Q K V c (t.setReg "start_n" .nat [] (Tile.scalar n)) := by
  constructor
  · exact h.mem
  · exact h.pids
  · exact h.undef
  · simpa using h.startm
  · simpa using h.bshead
  · simpa using h.base
  · simpa using h.offm
  · simpa using h.offn
  · simpa using h.query
  · simpa using h.maximum
  · simpa using h.denominator
  · simpa using h.accumulator
  · simpa using h.key
  · simpa using h.value
  · simpa using h.lo
  · simpa using h.hi

theorem scores_mem_eq (p : Layout) (s t : BlockState) (hm : t.mem = s.mem) (sn : Nat)
    (offm : Tile .nat [p.BLOCK_M]) (offn : Tile .nat [p.BLOCK_N])
    (q : Tile .real [p.BLOCK_M,p.DIM]) (kp : Tile .blockPtr [p.DIM,p.BLOCK_N]) :
    scores p t sn offm offn q kp = scores p s sn offm offn q kp := by
  unfold scores
  congr 1
  funext i
  erw [loaded_mem_eq s t hm kp]
  rfl

theorem updatedAcc_mem_eq (p : Layout) (s t : BlockState) (hm : t.mem = s.mem)
    (den alpha : Tile .real [p.BLOCK_M]) (acc : Tile .real [p.BLOCK_M,p.DIM])
    (prob : Tile .real [p.BLOCK_M,p.BLOCK_N]) (vp : Tile .blockPtr [p.BLOCK_N,p.DIM]) :
    updatedAcc p t den alpha acc prob vp = updatedAcc p s den alpha acc prob vp := by
  unfold updatedAcc
  dsimp only
  erw [loaded_mem_eq s t hm vp]
  rfl

theorem invariant_body_step (p : Layout) (s₀ : BlockState) (Q K V : RegionName)
    (c : Nat) (t : BlockState) (hBN : 0 < p.BLOCK_N)
    (h : PrefixInvariant p s₀ Q K V c t)
    (hsn : t.regs .nat [] "start_n" = some (Tile.scalar (c*p.BLOCK_N))) :
    ∃ u, stepStmts (body p) t = some u ∧ PrefixInvariant p s₀ Q K V (c+1) u := by
  let offm : Tile .nat [p.BLOCK_M] := Tile.vec (fun i => row p (s₀.pids 0) i.val)
  let offn : Tile .nat [p.BLOCK_N] := Tile.vec (fun j => j.val)
  let q := query p s₀ Q
  let m := prefixMax p s₀ Q K V c
  let den := prefixDen p s₀ Q K V c
  let acc := prefixAcc p s₀ Q K V c
  let kp := keyPointer p K (s₀.pids 1) (c*p.BLOCK_N)
  let vp := valuePointer p V (s₀.pids 1) (c*p.BLOCK_N)
  let result := iteration p t (c*p.BLOCK_N) offm offn q m den acc kp vp hBN
    hsn h.offm h.offn h.query h.maximum h.denominator h.accumulator h.key h.value
  refine ⟨result.val, result.property, ?_⟩
  obtain ⟨hmem,hpids,hundef⟩ := iteration_frame p t (c*p.BLOCK_N) offm offn q m den acc kp vp hBN
    hsn h.offm h.offn h.query h.maximum h.denominator h.accumulator h.key h.value
  obtain ⟨hq,hom,hon⟩ := iteration_preserves p t (c*p.BLOCK_N) offm offn q m den acc kp vp hBN
    hsn h.offm h.offn h.query h.maximum h.denominator h.accumulator h.key h.value
  obtain ⟨hk,hv⟩ := iteration_pointers p t (c*p.BLOCK_N) offm offn q m den acc kp vp hBN
    hsn h.offm h.offn h.query h.maximum h.denominator h.accumulator h.key h.value
  obtain ⟨hsm,hbs,hbase,hlo,hhi⟩ := iteration_nat_scalars p t (c*p.BLOCK_N) offm offn q m den acc kp vp hBN
    hsn h.offm h.offn h.query h.maximum h.denominator h.accumulator h.key h.value
  have hn := iteration_numeric p t (c*p.BLOCK_N) offm offn q m den acc kp vp hBN
    hsn h.offm h.offn h.query h.maximum h.denominator h.accumulator h.key h.value
  dsimp only at hn
  rw [scores_mem_eq p s₀ t h.mem] at hn
  rw [updatedAcc_mem_eq p s₀ t h.mem] at hn
  obtain ⟨hnm,hnd,hna⟩ := hn
  obtain ⟨hmstep,hdstep,hastep⟩ := prefix_update p s₀ Q K V c
  refine { mem := hmem.trans h.mem, pids := hpids.trans h.pids, undef := hundef.trans h.undef,
           startm := hsm.trans h.startm, bshead := hbs.trans h.bshead,
           base := hbase.trans h.base, offm := hom, offn := hon, query := hq,
           maximum := hnm.trans (congrArg some hmstep),
           denominator := hnd.trans (congrArg some hdstep),
           accumulator := hna.trans (congrArg some hastep), key := ?_, value := ?_,
           lo := hlo.trans h.lo, hi := hhi.trans h.hi }
  · simpa only [kp, keyPointer_advance, Nat.add_mul, one_mul] using hk
  · simpa only [vp, valuePointer_advance, Nat.add_mul, one_mul] using hv

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.FlashAttn.RealExecution
