import bench.tritonbench_g.context_attn_bloom.Proofs.ValuesExecution
import bench.tritonbench_g.context_attn_bloom.Proofs.KeysMeaning
namespace VeriTile.Bench.TritonBenchG.ContextAttnBloom.ValuesMeaning
open VeriTile.Triton Math Program RealSetup ValuesExecution KeysMeaning
noncomputable section
set_option maxHeartbeats 600000
set_option maxRecDepth 3000
set_option linter.unusedSimpArgs false

theorem value_matches (p : Layout) (r : Regions) (u pagesState initial : BlockState) (c : Nat)
    (hm : u.mem = initial.mem) (hg : pagesState.mem = initial.mem) (j : Fin p.BN) (d : Fin p.D) :
    let l := launch initial
    let a := inputs initial r
    let sn : Int := c*p.BN
    let last := Math.limit p l a
    let req := a.request l.batch
    let g := l.head/p.kvGroup
    (if sn+j.val < last ∧ d.val < p.headDim then
      u.readMem r.value (address p g (LoopKeysExecution.page p r sn req last pagesState j.val) d.val).toNat else 0) =
      Math.value p l a (c*p.BN+j.val) d.val := by
  dsimp only
  simp only [Math.value,Math.token,LoopKeysExecution.page,address,Nat.cast_add,Nat.cast_mul]
  by_cases hj : (c : Int)*p.BN+j.val < Math.limit p (launch initial) (inputs initial r)
  · simp only [hj,↓reduceIte,true_and]
    by_cases hd : d.val < p.headDim
    · simp only [hd,↓reduceIte,inputs,launch]
      rw [read_real_of_memory u initial hm,read_int_of_memory pagesState initial hg]
      congr 2
      congr 2
      ring_nf
    · simp [hd]
  · simp [hj]

theorem register_matches (p : Layout) (r : Regions) (u pagesState initial : BlockState) (c : Nat)
    (hm : u.mem = initial.mem) (hg : pagesState.mem = initial.mem) :
    let l := launch initial
    let a := inputs initial r
    let pages := fun j : Fin p.BN =>
      LoopKeysExecution.page p r (c*p.BN) (a.request l.batch) (Math.limit p l a) pagesState j.val
    (prepared p r (c*p.BN) (Math.limit p l a) (l.head/p.kvGroup) pages u).regs
      .real [p.BN,p.D] "v" = some (Tile.mat (dtype := .real)
        (fun j d => some (Math.value p l a (c*p.BN+j.val) d.val))) := by
  dsimp only
  simp only [prepared,BlockState.setReg_same,BlockState.setReg_readMem]
  apply congrArg some
  apply Tile.ext
  rintro ⟨j,d,⟨⟩⟩
  exact congrArg some (value_matches p r u pagesState initial c hm hg j d)

theorem static_preserved (p : Layout) (r : Regions) (u initial : BlockState)
    (sn last : Int) (kvHead : Nat) (pages : Fin p.BN → Int)
    (h : Invariant.Static p r initial u) :
    Invariant.Static p r initial (prepared p r sn last kvHead pages u) := by
  have hm := (ValuesExecution.memory p r sn last kvHead pages u).trans h.memory
  cases h
  constructor
  · exact hm
  all_goals simp_all [prepared]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnBloom.ValuesMeaning
