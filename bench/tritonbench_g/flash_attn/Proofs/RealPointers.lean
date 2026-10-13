import bench.tritonbench_g.flash_attn.Proofs.RealEntryFacts
import bench.tritonbench_g.flash_attn.Proofs.RealLoopFacts
namespace VeriTile.Bench.TritonBenchG.FlashAttn.RealExecution
open VeriTile.Triton Formula NormalizedSource
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def keyPointer (p : Layout) (K : RegionName) (bshead start : Nat) : Tile .blockPtr [p.DIM,p.BLOCK_N] :=
  ⟨fun _ => {region:=K,baseOffset:=base p bshead,parentShape:=[p.DIM,p.SEQLEN],
             blockShape:=[p.DIM,p.BLOCK_N],strides:=[p.stride_k_dim,p.stride_k_seqlen],offsets:=[0,start]}⟩
def valuePointer (p : Layout) (V : RegionName) (bshead start : Nat) : Tile .blockPtr [p.BLOCK_N,p.DIM] :=
  ⟨fun _ => {region:=V,baseOffset:=base p bshead,parentShape:=[p.SEQLEN,p.DIM],
             blockShape:=[p.BLOCK_N,p.DIM],strides:=[p.stride_k_seqlen,p.stride_v_dim],offsets:=[start,0]}⟩

theorem entry_pointers (Q K V L O : RegionName) (p : Layout) (s : BlockState) :
    (entry Q K V L O p s).val.regs .blockPtr [p.DIM,p.BLOCK_N] "K_block_ptr"=
      some (keyPointer p K (s.pids 1) 0) ∧
    (entry Q K V L O p s).val.regs .blockPtr [p.BLOCK_N,p.DIM] "V_block_ptr"=
      some (valuePointer p V (s.pids 1) 0) := by
  rcases p with ⟨sqh,sqs,sqd,sks,skd,svd,sos,sod,seq,bm,dim,bn,c,scale⟩
  cases c <;> simp [entry,keyPointer,valuePointer,base,NumericDType.mul]

theorem keyPointer_advance (p : Layout) (K : RegionName) (bshead start n : Nat) :
    (⟨fun i => ((keyPointer p K bshead start).data i).advance [0,Int.ofNat n]⟩ : Tile .blockPtr [p.DIM,p.BLOCK_N])=
      keyPointer p K bshead (start+n) := by
  ext i
  simp [keyPointer,BlockPtr.advance_2d_offsets]

theorem valuePointer_advance (p : Layout) (V : RegionName) (bshead start n : Nat) :
    (⟨fun i => ((valuePointer p V bshead start).data i).advance [Int.ofNat n,0]⟩ : Tile .blockPtr [p.BLOCK_N,p.DIM])=
      valuePointer p V bshead (start+n) := by
  ext i
  simp [valuePointer,BlockPtr.advance_2d_offsets]

theorem keyPointer_address (p : Layout) (K : RegionName) (bshead start : Nat)
    (i : TileIndex [p.DIM,p.BLOCK_N]) :
    ((keyPointer p K bshead start).data i).address (TileShape.indexToList _ i)=
      kOffset p bshead (start+i.2.1.val) i.1.val := by
  simp [keyPointer,TileShape.indexToList,kOffset]

theorem valuePointer_address (p : Layout) (V : RegionName) (bshead start : Nat)
    (i : TileIndex [p.BLOCK_N,p.DIM]) :
    ((valuePointer p V bshead start).data i).address (TileShape.indexToList _ i)=
      vOffset p bshead (start+i.1.val) i.2.1.val := by
  simp [valuePointer,TileShape.indexToList,vOffset]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FlashAttn.RealExecution
