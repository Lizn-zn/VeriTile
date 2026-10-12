import bench.tritonbench_g.fast_ce_loss.Proofs.ChunkedIO
namespace VeriTile.Bench.TritonBenchG.FastCeLoss.ChunkedWitness
open VeriTile.Triton TypedSource
noncomputable section
set_option maxHeartbeats 1500000
set_option maxRecDepth 20000

def allocation : FlatAlloc where
  flat := "flat"
  regions := ["x","y","z","l"]
  base := fun r => if r="x" then 1 else if r="y" then 10 else if r="z" then 20 else 30
  extent := fun r => if r="x" then 4 else if r="y" ∨ r="z" ∨ r="l" then 1 else 0

def initial (later : Bool) : BlockState where
  mem := fun r o => if r="l" then .of .int 0
    else if r="x" then if o=3 then .of .int 99 else .real (if o=0 then 3 else 2)
    else .of .int 17
  regs := fun _ _ _ => none
  pids := fun ax => if ax=1 then if later then 1 else 0 else 0
  numPids := fun _ => 2
  undef := fun _ _ => 0

theorem disjoint : allocation.Disjoint := by
  intro r hr s hs hn
  simp only [allocation,List.mem_cons,List.not_mem_nil,or_false] at hr hs
  rcases hr with rfl|rfl|rfl|rfl
  all_goals rcases hs with rfl|rfl|rfl|rfl
  all_goals first | exact False.elim (hn rfl) | decide

theorem coverage : ∀ r, r ∉ allocation.regions → allocation.extent r=0 := by
  intro r hr
  simp only [allocation,List.mem_cons,List.not_mem_nil,or_false,not_or] at hr
  simp [allocation,hr.1,hr.2.1,hr.2.2.1,hr.2.2.2]

theorem bounds (later : Bool) : ChunkedIO.Bounds "x" "y" "z" ⟨"l"⟩ 3 2 3 2 (initial later) allocation.extent := by
  refine ⟨?_,⟨0,rfl⟩,Or.inr ⟨by change (0 : Int)≤0; decide,by change (0 : Int)<3; decide⟩,?_,?_,?_,?_⟩
  · change 0<1; decide
  · intro _; change 0<1; decide
  · intro _; change 0<1; decide
  · intro _ _; change 0<4; decide
  · intro j hj
    have hj4 := Nat.lt_trans hj (by decide : 3<4)
    simpa [initial,allocation] using hj4

def ZeroResult (kernel : Kernel) : Prop :=
  ∃ v, exec (allocation.flattenKernel kernel) (allocation.flattenState (initial Bool.false))=some v ∧
    v.readMem "flat" 20=Real.log (Real.exp 3+Real.exp 2) ∧
    v.readMem "flat" 10= -3 ∧ v.mem "flat" 4=MemCell.of .int 99

def LaterResult (kernel : Kernel) : Prop :=
  ∃ v, exec (allocation.flattenKernel kernel) (allocation.flattenState (initial Bool.true))=some v ∧
    v.mem=(allocation.flattenState (initial Bool.true)).mem

theorem zero_contract (kernel : Kernel)
    (h : ChunkedIO.Implements kernel Bool.false Bool.false "x" "y" "z" ⟨"l"⟩ 3 2 3 2 1 2) :
    ZeroResult kernel := by
  obtain ⟨v,he,ho⟩ := h (by intro h; cases h) (by intro h; cases h)
    allocation disjoint coverage (initial Bool.false) rfl (bounds Bool.false)
    (by decide) (by decide) (by decide)
  simp only [ChunkedIO.Outputs,show (initial Bool.false).pids 1=0 from rfl,↓reduceIte] at ho
  obtain ⟨hz,hy,hf⟩ := ho
  have hp : (initial Bool.false).pids=(fun _ => 0) := by
    funext ax
    simp [initial]
  have h0 : (initial Bool.false).readMem "x" 0=3 := rfl
  have h1 : (initial Bool.false).readMem "x" 1=2 := rfl
  have hl : (initial Bool.false).readMemValue .int (Region.cast (⟨"l"⟩ : Region .int)) 0=0 := rfl
  refine ⟨v,he,?_,?_,?_⟩
  · change v.readMem "flat" 20 = _ at hz
    rw [hz]
    norm_num [RealMath.lse,RealMath.transform,Fin.sum_univ_succ,hp,h0,h1]
  · change v.readMem "flat" 10 = _ at hy
    rw [hy]
    norm_num [ChunkedReal.partialLoss,RealMath.transform,hp,hl,h0]
  · rw [hf "flat" 4 (Or.inr (by change 4≠10 ∧ 4≠20; constructor <;> decide))]
    rfl

/-- The last real chunk has an active input, but its out-of-allocation LSE
address is never written: all finite memory, including output tags, is retained. -/
theorem later_contract (kernel : Kernel) (cap scale : Bool)
    (h : ChunkedIO.Implements kernel cap scale "x" "y" "z" ⟨"l"⟩ 3 2 3 2 1 2) :
    LaterResult kernel := by
  obtain ⟨v,he,ho⟩ := h (by intro _; norm_num) (by intro _; norm_num)
    allocation disjoint coverage (initial Bool.true) rfl (bounds Bool.true)
    (by decide) (by decide) (by decide)
  exact ⟨v,he,by simpa [ChunkedIO.Outputs,initial] using ho⟩

theorem zero_executes (storage : Storage) (swapped : Bool) :
    ZeroResult (chunked storage Bool.false Bool.false swapped "x" "y" "z" ⟨"l"⟩ 3 2 3 2 1 2).eraseDType.toAlgKernel := by
  cases swapped
  · exact zero_contract _ (ChunkedIO.typed_correct storage Bool.false Bool.false "x" "y" "z" ⟨"l"⟩ 3 2 3 2 1 2)
  · exact zero_contract _ (ChunkedIO.candidate_correct storage Bool.false Bool.false "x" "y" "z" ⟨"l"⟩ 3 2 3 2 1 2)

theorem later_executes (storage : Storage) (cap scale swapped : Bool) :
    LaterResult (chunked storage cap scale swapped "x" "y" "z" ⟨"l"⟩ 3 2 3 2 1 2).eraseDType.toAlgKernel := by
  cases swapped
  · exact later_contract _ cap scale (ChunkedIO.typed_correct storage cap scale "x" "y" "z" ⟨"l"⟩ 3 2 3 2 1 2)
  · exact later_contract _ cap scale (ChunkedIO.candidate_correct storage cap scale "x" "y" "z" ⟨"l"⟩ 3 2 3 2 1 2)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FastCeLoss.ChunkedWitness
