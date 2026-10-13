import bench.tritonbench_g.flash_attn.Proofs.RealEntry
namespace VeriTile.Bench.TritonBenchG.FlashAttn.RealExecution
open VeriTile.Triton Formula NormalizedSource
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

theorem entry_frame (Q K V L O : RegionName) (p : Layout) (s : BlockState) :
    (entry Q K V L O p s).val.mem=s.mem ∧
    (entry Q K V L O p s).val.pids=s.pids ∧
    (entry Q K V L O p s).val.undef=s.undef := by
  rcases p with ⟨sqh,sqs,sqd,sks,skd,svd,sos,sod,seq,bm,dim,bn,c,scale⟩
  cases c <;> simp [entry]

theorem entry_indices (Q K V L O : RegionName) (p : Layout) (s : BlockState) :
    (entry Q K V L O p s).val.regs .nat [] "start_m"=some (Tile.scalar (s.pids 0)) ∧
    (entry Q K V L O p s).val.regs .nat [] "off_bs_head"=some (Tile.scalar (s.pids 1)) ∧
    (entry Q K V L O p s).val.regs .nat [] "qkv_base_offset"=some (Tile.scalar (base p (s.pids 1))) ∧
    (entry Q K V L O p s).val.regs .nat [p.BLOCK_M] "off_m"=some (Tile.vec (fun i => row p (s.pids 0) i.val)) ∧
    (entry Q K V L O p s).val.regs .nat [p.BLOCK_N] "off_n"=some (Tile.vec (fun j => j.val)) ∧
    (entry Q K V L O p s).val.regs .nat [] "lo"=some (Tile.scalar 0) ∧
    (entry Q K V L O p s).val.regs .nat [] "hi"=some (Tile.scalar (hi p (s.pids 0))) := by
  rcases p with ⟨sqh,sqs,sqd,sks,skd,svd,sos,sod,seq,bm,dim,bn,c,scale⟩
  cases c <;> simp [entry,base,row,hi,Geometry.stop]
  all_goals simp [-Tile.scalar_eta,Tile.bop,Tile.scalar,Tile.vec,NumericDType.add,NumericDType.mul,
    Broadcast.leftIndex,Broadcast.rightIndex,WithBot.realAdd]

theorem entry_seed (Q K V L O : RegionName) (p : Layout) (s : BlockState) :
    (entry Q K V L O p s).val.regs .real [p.BLOCK_M] "max"=some ⟨fun _ => (⊥ : WithBot ℝ)⟩ ∧
    (entry Q K V L O p s).val.regs .real [p.BLOCK_M] "denom"=some ⟨fun _ => (0 : WithBot ℝ)⟩ ∧
    (entry Q K V L O p s).val.regs .real [p.BLOCK_M,p.DIM] "out_buffer"=some ⟨fun _ => (0 : WithBot ℝ)⟩ := by
  rcases p with ⟨sqh,sqs,sqd,sks,skd,svd,sos,sod,seq,bm,dim,bn,c,scale⟩
  cases c <;> simp [entry]
  all_goals simp [-Tile.scalar_eta,Tile.bop,Tile.scalar,Tile.vec,NumericDType.add,NumericDType.mul,
    Broadcast.leftIndex,Broadcast.rightIndex,WithBot.realAdd]
  all_goals exact ⟨rfl,rfl,rfl⟩

/-- The scaled query uses the exact decimal multiplier from the source. -/
def query (p : Layout) (s : BlockState) (Q : RegionName) : Tile .real [p.BLOCK_M,p.DIM] :=
  ⟨fun i => some (s.readMem Q (qOffset p (s.pids 0) (s.pids 1) i.1.val i.2.1.val) *
    (p.sm_scale*1.44269504))⟩

theorem entry_query (Q K V L O : RegionName) (p : Layout) (s : BlockState) :
    (entry Q K V L O p s).val.regs .real [p.BLOCK_M,p.DIM] "q"=some (query p s Q) := by
  rcases p with ⟨sqh,sqs,sqd,sks,skd,svd,sos,sod,seq,bm,dim,bn,c,scale⟩
  cases c <;> simp [entry,query]
  all_goals
    ext i
    simp [Tile.bop_data,Tile.scalar_data,Broadcast.leftIndex,Broadcast.rightIndex,
      NumericDType.mul,WithBot.realMul,BlockState.readMemValue_real,
      TileShape.indexToList,BlockPtr.address_2d_row_offset,qOffset,base,row]
    rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FlashAttn.RealExecution
