import bench.tritonbench_g.destindex_copy_kv1.Proofs.SourceIO
namespace VeriTile.Bench.TritonBenchG.DestindexCopyKv1.RealWitness
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

def layout : Layout where
  K := "input"
  Dest_loc := ⟨"dest"⟩
  Out := "output"
  stride_k_bs := 16
  stride_k_h := 4
  stride_k_d := 1
  stride_o_bs := 16
  stride_o_h := 4
  stride_o_d := 1
  head_num := 3
  head_dim := 3
  BLOCK_DMODEL := 4
  BLOCK_HEAD := 4

def allocation : FlatAlloc where
  flat := "flat"
  regions := ["input","dest","output"]
  base := fun r => if r="input" then 0 else if r="dest" then 64 else 128
  extent := fun r => if r="input" then 32 else if r="dest" then 2 else if r="output" then 48 else 0

def initial : BlockState where
  mem := fun r _ => if r="dest" then .mk .int 2 else .mk .real (some (-3))
  regs := fun _ _ _ => none
  pids := fun _ => 1
  numPids := fun _ => 2
  undef := fun _ _ => 0

def inputs : RealIO.Inputs 4 4 where
  row := 2
  values := fun _ => -3

theorem disjoint : allocation.Disjoint := by
  intro r hr s hs hn
  simp only [allocation,List.mem_cons,List.not_mem_nil,or_false] at hr hs
  rcases hr with rfl | rfl | rfl
  all_goals rcases hs with rfl | rfl | rfl
  all_goals first | exact False.elim (hn rfl) | decide

theorem coverage : ∀ r, r ∉ allocation.regions → allocation.extent r=0 := by
  intro r hr
  simp only [allocation,List.mem_cons,List.not_mem_nil,or_false,not_or] at hr
  simp [allocation,hr.1,hr.2.1,hr.2.2]

theorem output_injective : Function.Injective
    (fun j : TileIndex [layout.BLOCK_HEAD,layout.BLOCK_DMODEL] =>
      layout.stride_o_h*j.1.val + layout.stride_o_d*j.2.1.val) := by
  intro ⟨a,b,c⟩ ⟨x,y,z⟩ h
  change 4*a.val+1*b.val = 4*x.val+1*y.val at h
  have ha : a=x := Fin.ext (by have hb := b.isLt; have hy := y.isLt; change b.val<4 at hb; change y.val<4 at hy; omega)
  subst ha
  have hb : b=y := Fin.ext (by omega)
  subst hb
  rw [Subsingleton.elim c z]

/-- Finite nonzero source/destination rows, a 4×4 tile with both head and
column tails masked, all nine input/output storage pairs and both variants. -/
theorem executes (input output : TypedSource.Storage) (swapped : Bool) :
    ∃ final, exec (allocation.flattenKernel (layout.source input output swapped).eraseDType.toAlgKernel)
      (allocation.flattenState initial) = some final := by
  have h := SourceIO.correct layout input output swapped output_injective
  have hb : (SourceIO.sourceIO layout input output swapped).Bounds allocation.extent 1 inputs := by
    refine ⟨by change 1<2; decide,?_,?_⟩
    all_goals intro j hj; change j.1.val<3 ∧ j.2.1.val<3 at hj
    all_goals simp [SourceIO.sourceIO,RealIO.referenceIO,layout,allocation,inputs]; omega
  have hp : (SourceIO.sourceIO layout input output swapped).Pins initial inputs := by
    exact ⟨rfl,fun _ _ => rfl⟩
  obtain ⟨t,he,_,_⟩ := h allocation disjoint rfl coverage initial inputs (by decide) rfl hb hp
  exact ⟨t,he⟩


def inactiveLayout : Layout := { layout with head_num := 0 }

def inactiveAllocation : FlatAlloc :=
  { allocation with extent := fun r => if r="dest" then 2 else 0 }

theorem inactive_disjoint : inactiveAllocation.Disjoint := by
  intro r hr s hs hn
  simp only [inactiveAllocation,allocation,List.mem_cons,List.not_mem_nil,or_false] at hr hs
  rcases hr with rfl | rfl | rfl
  all_goals rcases hs with rfl | rfl | rfl
  all_goals first | exact False.elim (hn rfl) | decide

theorem inactive_coverage : ∀ r, r ∉ inactiveAllocation.regions → inactiveAllocation.extent r=0 := by
  intro r hr
  simp only [inactiveAllocation,allocation,List.mem_cons,List.not_mem_nil,or_false,not_or] at hr
  simp [inactiveAllocation,hr.2.1]

/-- Empty payload allocations remain executable when every head is masked;
the unmasked metadata still has its own nonempty allocation. -/
theorem inactive_executes (input output : TypedSource.Storage) (swapped : Bool) :
    ∃ final, exec (inactiveAllocation.flattenKernel
      (inactiveLayout.source input output swapped).eraseDType.toAlgKernel)
      (inactiveAllocation.flattenState initial) = some final := by
  have h := SourceIO.correct inactiveLayout input output swapped output_injective
  have hb : (SourceIO.sourceIO inactiveLayout input output swapped).Bounds
      inactiveAllocation.extent 1 inputs := by
    refine ⟨by change 1<2; decide,?_,?_⟩
    all_goals intro j hj; exact False.elim (Nat.not_lt_zero _ hj.1)
  have hp : (SourceIO.sourceIO inactiveLayout input output swapped).Pins initial inputs := by
    refine ⟨rfl,?_⟩
    intro j hj
    exact False.elim (Nat.not_lt_zero _ hj.1)
  obtain ⟨t,he,_,_⟩ := h inactiveAllocation inactive_disjoint rfl inactive_coverage
    initial inputs (by decide) rfl hb hp
  exact ⟨t,he⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DestindexCopyKv1.RealWitness
