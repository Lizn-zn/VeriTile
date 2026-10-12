import bench.tritonbench_g.fast_ce_loss.Proofs.ForwardIO
import bench.tritonbench_g.fast_ce_loss.Proofs.BackwardIO
namespace VeriTile.Bench.TritonBenchG.FastCeLoss.RealWitness
open VeriTile.Triton TypedSource
noncomputable section
set_option maxHeartbeats 1500000
set_option maxRecDepth 20000

def allocation : FlatAlloc where
  flat := "flat"
  regions := ["x","y","z","l","d"]
  base := fun r => if r="x" then 1 else if r="y" then 10 else if r="z" then 20 else if r="l" then 30 else 40
  extent := fun r => if r="x" then 2 else if r="y" ∨ r="z" ∨ r="l" ∨ r="d" then 1 else 0

def initial (ignore : Bool) : BlockState where
  mem := fun r o => if r="l" then .of .int (if ignore then -100 else 0)
    else if r="x" then if o=0 then .real 0 else .of .int 99
    else if r="d" then .real 2 else if r="z" then .real (Real.log (1+Real.exp (-1))) else .of .int 17
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun _ _ => 0

theorem disjoint : allocation.Disjoint := by
  intro r hr s hs hn
  simp only [allocation,List.mem_cons,List.not_mem_nil,or_false] at hr hs
  rcases hr with rfl|rfl|rfl|rfl|rfl
  all_goals rcases hs with rfl|rfl|rfl|rfl|rfl
  all_goals first | exact False.elim (hn rfl) | decide

theorem coverage : ∀ r, r ∉ allocation.regions → allocation.extent r=0 := by
  intro r hr
  simp only [allocation,List.mem_cons,List.not_mem_nil,or_false,not_or] at hr
  simp [allocation,hr.1,hr.2.1,hr.2.2.1,hr.2.2.2.1,hr.2.2.2.2]

def ForwardResult (kernel : Kernel) (ignore : Bool) : Prop :=
  ∃ v, exec (allocation.flattenKernel kernel) (allocation.flattenState (initial ignore))=some v ∧
    v.readMem "flat" 20=Real.log (1+Real.exp (-1)) ∧
    v.readMem "flat" 10=(if ignore then 0 else Real.log (1+Real.exp (-1))) ∧
    v.mem "flat" 2=MemCell.of .int 99

theorem forward_contract (kernel : Kernel) (ignore : Bool)
    (h : ForwardIO.Implements kernel Bool.true Bool.false "x" "y" "z" ⟨"l"⟩ 1 1 2 1 1) :
    ForwardResult kernel ignore := by
  have hl : (initial ignore).readMemValue .int ⟨"l"⟩ 0=(if ignore then -100 else 0) := by
    cases ignore <;> rfl
  have hpid : (initial ignore).pids=(fun _ => 0) := rfl
  have hraw : (initial ignore).readMem "x" 0=0 := rfl
  have hb : ForwardIO.Bounds "x" "y" "z" ⟨"l"⟩ 1 1 2 (initial ignore) allocation.extent := by
    refine ⟨?_,⟨_,rfl⟩,?_,?_,?_,?_,?_⟩
    · change 0<1; decide
    · change (if ignore then (-100 : Int) else 0) = -100 ∨
        (0≤(if ignore then (-100 : Int) else 0) ∧ (if ignore then (-100 : Int) else 0)<1)
      cases ignore <;> norm_num
    · change 0<1; decide
    · change 0<1; decide
    · intro hi
      change (if ignore then (-100 : Int) else 0) ≠ -100 at hi
      simp only [hpid,Region.cast,hl,Nat.zero_mul,Nat.zero_add]
      cases ignore <;> norm_num [allocation] at *
    · intro j hj
      change j.val<1 at hj
      have hlt : j.val<2 := by omega
      simpa [hpid,allocation] using hlt
  obtain ⟨v,he,hz,hy,hf⟩ := h (by intro _; norm_num) (by intro h; cases h)
    allocation disjoint coverage (initial ignore) rfl hb (by decide) (by decide) (by decide)
  have hm : RealMath.lse Bool.true Bool.false 1 1 1 0
      (fun j : Fin 2 => (initial ignore).readMem "x" ((initial ignore).pids 0*1+j.val)) =
      Real.log (1+Real.exp (-1)) := by
    norm_num [RealMath.lse,RealMath.transform,Fin.sum_univ_succ,hpid,hraw]
  refine ⟨v,he,?_,?_,?_⟩
  · exact hz.trans hm
  · change v.readMem allocation.flat (allocation.addr "y" ((initial ignore).pids 0))=_
    rw [hy]
    simp only [hpid,Nat.zero_mul,Nat.zero_add] at hm
    cases ignore <;> norm_num [RealMath.loss,Region.cast,hpid,hl,hm,RealMath.transform,hraw]
  · rw [hf "flat" 2 (Or.inr (by change 2≠10 ∧ 2≠20; constructor <;> decide))]
    rfl

theorem forward_executes (storage : Storage) (swapped ignore : Bool) :
    ForwardResult (forward storage Bool.true Bool.false swapped "x" "y" "z" ⟨"l"⟩ 1 1 2 1 1).eraseDType.toAlgKernel ignore := by
  cases swapped
  · exact forward_contract _ ignore (ForwardIO.typed_correct storage Bool.true Bool.false "x" "y" "z" ⟨"l"⟩ 1 1 2 1 1)
  · exact forward_contract _ ignore (ForwardIO.candidate_correct storage Bool.true Bool.false "x" "y" "z" ⟨"l"⟩ 1 1 2 1 1)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FastCeLoss.RealWitness
