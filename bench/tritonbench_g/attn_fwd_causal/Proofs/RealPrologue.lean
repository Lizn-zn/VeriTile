import bench.tritonbench_g.attn_fwd_causal.Proofs.RealInit

/-! The complete real source prologue establishes the helper entry contract
from program IDs and actual input memory. No loaded query or carry is assumed.
The only value convention here is zero for masked undefined reads; physical
input windows and output address bounds belong to the final IO contract. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealPrologue
open VeriTile.Triton RealLoads
open AttentionFwdTriton3.RealScores (Frame)
noncomputable section

def configuration (Q K V QS KS : RegionName) (strideM strideD strideKN N HD HA : Nat)
    (v : RealCoordinates.Values) : Config where
  Q := Q
  K := K
  V := V
  QS := QS
  KS := KS
  base := v.base
  queryScaleBase := v.queryScaleBase
  keyScaleBase := v.keyScaleBase
  strideM := strideM
  strideD := strideD
  strideKN := strideKN
  context := N
  headDim := HD
  activeHead := HA
  startM := v.row

def code (Q K V QS KS Out : RegionName) (strideZ strideH strideM strideD strideKN H N HD BM BN BD HA : Nat) : List Stmt :=
  RealCoordinates.code H N BM BN strideZ strideH strideM ++
    RealPointers.code (configuration Q K V QS KS strideM strideD strideKN N HD HA ⟨0,0,0,0,0,0,0,0⟩) Out BM BN BD ++
    RealInit.code BM BD N HA

def writes : List RegName := RealCoordinates.writes ++ RealPointers.writes ++ RealInit.writes

theorem executes (s : BlockState) (Q K V QS KS Out : RegionName)
    (strideZ strideH strideM strideD strideKN H N HD BM BN BD HA : Nat)
    (hu : ∀ r a, s.undef r a = 0) :
    let c := configuration Q K V QS KS strideM strideD strideKN N HD HA
      (RealCoordinates.values s H N BM BN strideZ strideH strideM)
    ∃ t, stepStmts (code Q K V QS KS Out strideZ strideH strideM strideD strideKN H N HD BM BN BD HA) s = some t ∧
      RealMiddle.Entry (inputs s c BM BN BD) c t ∧
      t.regs .ptr [BM,BD] "O_block_ptr" = some (RealPointers.outputPointer c Out BM BD) ∧ Frame writes s t := by
  dsimp only
  let c := configuration Q K V QS KS strideM strideD strideKN N HD HA
    (RealCoordinates.values s H N BM BN strideZ strideH strideM)
  obtain ⟨sc, hc, hcoords, hcf⟩ := RealCoordinates.executes s H N BM BN strideZ strideH strideM
  obtain ⟨sp, hp, hptrs, hpf⟩ := RealPointers.executes sc c Out BM BN BD
    hcoords.row hcoords.base hcoords.queryScaleBase hcoords.keyScaleBase
  obtain ⟨t, ht, hentry, hout, htf⟩ := RealInit.executes sp c Out BM BN BD hptrs
    (by rw [hpf.undef, hcf.undef]; exact hu)
  rw [inputs_same_memory sp s c BM BN BD (hpf.mem.trans hcf.mem)] at hentry
  refine ⟨t, ?_, hentry, hout, ?_⟩
  · have hptr := (stepStmts.append_some hc).trans hp
    exact (stepStmts.append_some hptr).trans ht
  · exact ((hcf.mono (by simp [writes])).trans (hpf.mono (by simp [writes]))).trans
      (htf.mono (by simp [writes]))

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealPrologue
