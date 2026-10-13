import bench.tritonbench_g.flash_attn.Proofs.RealPost
import bench.tritonbench_g.flash_attn.Proofs.Writeback

/-! Exact memory result of both final stores, retaining arbitrary address
aliasing and the interpreter's source lane order. -/
namespace VeriTile.Bench.TritonBenchG.FlashAttn.RealExecution
open VeriTile.Triton Formula NormalizedSource
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def normalizedTile (p : Layout) (den : Tile .real [p.BLOCK_M])
    (acc : Tile .real [p.BLOCK_M,p.DIM]) : Tile .real [p.BLOCK_M,p.DIM] :=
  Tile.bop NumericDType.real.div (.consSame (.consR .nil)) acc (Tile.expandDim ⟨1, by simp⟩ den)

def logsumTile (p : Layout) (m den : Tile .real [p.BLOCK_M]) : Tile .real [p.BLOCK_M] :=
  Tile.bop NumericDType.real.add (.consSame .nil) m (Tile.uop WithBot.realLog2 den)

theorem post_mem (Q K V L O : RegionName) (p : Layout) (s₀ t : BlockState)
    (m den : Tile .real [p.BLOCK_M]) (acc : Tile .real [p.BLOCK_M,p.DIM])
    (hsm : t.regs .nat [] "start_m" = some (Tile.scalar (s₀.pids 0)))
    (hb : t.regs .nat [] "off_bs_head" = some (Tile.scalar (s₀.pids 1)))
    (hbase : t.regs .nat [] "qkv_base_offset" = some (Tile.scalar (base p (s₀.pids 1))))
    (hom : t.regs .nat [p.BLOCK_M] "off_m" = some (Tile.vec (fun i => row p (s₀.pids 0) i.val)))
    (hm : t.regs .real [p.BLOCK_M] "max" = some m)
    (hd : t.regs .real [p.BLOCK_M] "denom" = some den)
    (ha : t.regs .real [p.BLOCK_M,p.DIM] "out_buffer" = some acc) :
    (post Q K V L O p t (s₀.pids 0) (s₀.pids 1) (base p (s₀.pids 1))
      (Tile.vec (fun i => row p (s₀.pids 0) i.val)) m den acc hsm hb hbase hom hm hd ha).val.mem =
    (writeState p (s₀.pids 0) (s₀.pids 1) L O
      (fun i => FloatDType.real.storeValue ((logsumTile p m den).data (i,PUnit.unit)))
      (fun i => FloatDType.real.storeValue ((normalizedTile p den acc).data i)) t).mem := by
  rcases p with ⟨sqh,sqs,sqd,sks,skd,svd,sos,sod,seq,bm,dim,bn,c,scale⟩
  simp only [post]
  simp [writeState, normalizedTile, logsumTile, oOffset, lOffset, base, row,
    TileShape.indexToList, BlockPtr.address_2d_row_offset,
    NumericDType.mul, NumericDType.add, Tile.bop, Tile.scalar, Tile.vec,
    Broadcast.leftIndex, Broadcast.rightIndex, -Tile.scalar_eta]
  funext r o
  apply BlockState.foldl_writeMem_mem_congr _ _ _ _ (fun _ _ => rfl)
  simp only [BlockState.setReg_mem]
  apply BlockState.foldl_writeMem_mem_congr _ _ _ _ (fun _ _ => rfl)
  rfl

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.FlashAttn.RealExecution
