import bench.tritonbench_g.destindex_copy.Proofs.SourceIO
namespace VeriTile.Bench.TritonBenchG.DestindexCopy.RealWitness
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

def layout : Layout where
  KV_nope := "nope"
  KV_rope := "rope"
  Dest_loc := ⟨"dest"⟩
  O_nope := "outNope"
  O_rope := "outRope"
  stride_kv_nope_bs := 4
  stride_kv_nope_h := 7
  stride_kv_nope_d := 1
  stride_kv_rope_bs := 2
  stride_kv_rope_h := 9
  stride_kv_rope_d := 1
  stride_o_nope_bs := 4
  stride_o_nope_h := 7
  stride_o_nope_d := 1
  stride_o_rope_bs := 2
  stride_o_rope_h := 9
  stride_o_rope_d := 1
  kv_nope_head_num := 2
  kv_rope_head_num := 3
  BLOCK_DMODEL_NOPE := 4
  BLOCK_DMODEL_ROPE := 2

def allocation : FlatAlloc where
  flat := "flat"
  regions := ["nope","rope","dest","outNope","outRope"]
  base := fun r => if r="nope" then 0 else if r="rope" then 16 else if r="dest" then 32
    else if r="outNope" then 48 else 64
  extent := fun r => if r="nope" then 8 else if r="rope" then 4 else if r="dest" then 2
    else if r="outNope" then 12 else if r="outRope" then 6 else 0

def initial : BlockState where
  mem := fun r _ => if r="dest" then .mk .int 2
    else .mk .real (some (if r="nope" then -3 else 5))
  regs := fun _ _ _ => none
  pids := fun _ => 1
  numPids := fun _ => 2
  undef := fun _ _ => 0

def inputs : RealIO.Inputs 4 2 where
  row := 2
  nope := fun _ => -3
  rope := fun _ => 5

theorem disjoint : allocation.Disjoint := by
  intro r hr s hs hn
  simp only [allocation,List.mem_cons,List.not_mem_nil,or_false] at hr hs
  rcases hr with rfl | rfl | rfl | rfl | rfl
  all_goals rcases hs with rfl | rfl | rfl | rfl | rfl
  all_goals first | exact False.elim (hn rfl) | decide

theorem coverage : ∀ r, r ∉ allocation.regions → allocation.extent r=0 := by
  intro r hr
  simp only [allocation,List.mem_cons,List.not_mem_nil,or_false,not_or] at hr
  simp [allocation,hr.1,hr.2.1,hr.2.2.1,hr.2.2.2.1,hr.2.2.2.2]

/-- A nonzero source row is scattered into a different signed destination row
using two unequal tile widths. Every storage pair and both versions execute. -/
theorem executes (nopeType ropeType : TypedSource.Storage) (swapped : Bool) :
    ∃ final, exec (allocation.flattenKernel (layout.source nopeType ropeType swapped).eraseDType.toAlgKernel)
      (allocation.flattenState initial) = some final := by
  have h := SourceIO.correct layout nopeType ropeType swapped (by decide)
    (by intro i j h; apply Fin.ext; simpa [layout] using h)
    (by intro i j h; apply Fin.ext; simpa [layout] using h)
  have hb : (SourceIO.sourceIO layout nopeType ropeType swapped).Bounds allocation.extent 1 inputs := by
    refine ⟨by change 1 < 2; decide,?_,?_,?_,?_⟩
    all_goals intro j; simp [SourceIO.sourceIO,RealIO.referenceIO,layout,allocation,inputs]; omega
  have hp : (SourceIO.sourceIO layout nopeType ropeType swapped).Pins initial inputs := by
    exact ⟨rfl,fun _ => rfl,fun _ => rfl⟩
  obtain ⟨t,he,_,_,_⟩ := h allocation disjoint rfl coverage initial inputs (by decide) rfl hb hp
  exact ⟨t,he⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DestindexCopy.RealWitness
