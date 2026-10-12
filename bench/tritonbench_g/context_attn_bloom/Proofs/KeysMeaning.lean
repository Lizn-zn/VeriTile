import bench.tritonbench_g.context_attn_bloom.Proofs.LoopKeysExecution
import bench.tritonbench_g.context_attn_bloom.Proofs.Invariant
namespace VeriTile.Bench.TritonBenchG.ContextAttnBloom.KeysMeaning
open VeriTile.Triton Math Program RealSetup LoopKeysExecution
noncomputable section
set_option maxHeartbeats 600000
set_option maxRecDepth 3000
set_option linter.unusedSimpArgs false

theorem read_real_of_memory (u v : BlockState) (h : u.mem = v.mem) (r : RegionName) (off : Nat) :
    u.readMem r off = v.readMem r off := by unfold BlockState.readMem; rw [h]
theorem read_int_of_memory (u v : BlockState) (h : u.mem = v.mem) (r : RegionName) (off : Nat) :
    u.readMemValue .int r off = v.readMemValue .int r off := by
  simp only [BlockState.readMemValue,BlockState.readMemTyped,h]

theorem value_matches (p : Layout) (r : Regions) (u initial : BlockState) (c : Nat)
    (hm : u.mem = initial.mem) (d : Fin p.D) (j : Fin p.BN) :
    let l := launch initial
    let a := inputs initial r
    let sn : Int := c*p.BN
    let last := Math.limit p l a
    let req := a.request l.batch
    let g := l.head/p.kvGroup
    (if sn+j.val < last ∧ d.val < p.headDim then
      u.readMem r.key (keyAddress p g (page p r sn req last u j.val) d.val).toNat else 0) =
      Math.key p l a (c*p.BN+j.val) d.val := by
  dsimp only
  simp only [Math.key,Math.token,page,keyAddress,Nat.cast_add,Nat.cast_mul]
  by_cases hj : (c : Int)*p.BN+j.val < Math.limit p (launch initial) (inputs initial r)
  · simp only [hj,↓reduceIte,true_and]
    by_cases hd : d.val < p.headDim
    · simp only [hd,↓reduceIte,inputs,launch]
      rw [read_real_of_memory u initial hm,read_int_of_memory u initial hm]
      congr 2
      congr 2
      ring_nf
    · simp [hd]
  · simp [hj]

theorem register_matches (p : Layout) (r : Regions) (u initial : BlockState) (c : Nat)
    (hm : u.mem = initial.mem) :
    let l := launch initial
    let a := inputs initial r
    (prepared p r (c*p.BN) (a.request l.batch) (Math.limit p l a) (l.head/p.kvGroup) u).regs
      .real [p.D,p.BN] "k" = some (Tile.mat (dtype := .real)
        (fun d j => some (Math.key p l a (c*p.BN+j.val) d.val))) := by
  dsimp only
  simp only [prepared,BlockState.setReg_same,BlockState.setReg_readMem]
  apply congrArg some
  apply Tile.ext
  rintro ⟨d,j,⟨⟩⟩
  exact congrArg some (value_matches p r u initial c hm d j)

theorem static_preserved (p : Layout) (r : Regions) (u initial : BlockState)
    (sn req last : Int) (kvHead : Nat) (h : Invariant.Static p r initial u) :
    Invariant.Static p r initial (prepared p r sn req last kvHead u) := by
  have hm := (LoopKeysExecution.memory p r sn req last kvHead u).trans h.memory
  cases h
  constructor
  · exact hm
  all_goals simp_all [prepared]

theorem dynamic_preserved (p : Layout) (r : Regions) (u initial : BlockState)
    (sn req last : Int) (kvHead : Nat) (scale : ℝ) (c : Nat)
    (h : Invariant.Dynamic p r initial scale c u) :
    Invariant.Dynamic p r initial scale c (prepared p r sn req last kvHead u) := by
  cases h
  constructor <;> simp_all [prepared]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnBloom.KeysMeaning
