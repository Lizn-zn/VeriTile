import bench.tritonbench_g.fast_rms_layernorm.Correct
namespace VeriTile.Bench.TritonBenchG.FastRmsLayernorm.Execution
open VeriTile.Triton TypedSource
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000

def allocation (isGemma : Bool) : FlatAlloc where
  flat := "flat"
  regions := ["x","y","w","r"]
  base := fun r => if r="x" then 1 else if r="y" then 11 else if r="w" then 21 else 31
  extent := fun r => if r="x" then 7 else if r="y" then 8 else if r="w" then
    (if isGemma then 3 else 5) else if r="r" then 3 else 0

def initial : BlockState where
  mem := fun r o => if r="y" ∧ o=7 then .of .int 99
    else .real (if r="w" then (o : ℝ)+1 else if r="r" then 2 else 1)
  regs := fun _ _ _ => none
  pids := fun a => if a=0 then 1 else 0
  numPids := fun _ => 2
  undef := fun _ _ => 0

theorem disjoint (isGemma : Bool) : (allocation isGemma).Disjoint := by
  intro r hr s hs hn
  simp only [allocation,List.mem_cons,List.not_mem_nil,or_false] at hr hs
  rcases hr with rfl|rfl|rfl|rfl
  all_goals rcases hs with rfl|rfl|rfl|rfl
  all_goals first | exact False.elim (hn rfl) | cases isGemma <;> decide

theorem coverage (isGemma : Bool) : ∀ r,r∉(allocation isGemma).regions → (allocation isGemma).extent r=0 := by
  intro r hr
  simp only [allocation,List.mem_cons,List.not_mem_nil,or_false,not_or] at hr
  simp [allocation,hr.1,hr.2.1,hr.2.2.1,hr.2.2.2]

theorem ones_mean_square : RealMath.meanSquare 3 (fun _ : Fin 4 => (1 : ℝ))=1 := by
  unfold RealMath.meanSquare RealMath.squareSum
  simp only [Fin.sum_univ_succ]
  norm_num [RealMath.padded]

theorem ones_inv : RealMath.invStd 3 0 (fun _ : Fin 4 => (1 : ℝ))=1 := by
  norm_num [RealMath.invStd,ones_mean_square]

def ForwardResult (isGemma : Bool) (kernel : Kernel) : Prop :=
  ∃ v,exec ((allocation isGemma).flattenKernel kernel) ((allocation isGemma).flattenState initial)=some v ∧
    (∀ j : Fin 3,v.readMem "flat" (15+j.val)=if isGemma then (j.val : ℝ)+2 else 2*(j.val : ℝ)+1) ∧
    v.readMem "flat" 33=1 ∧ v.mem "flat" 18=MemCell.of .int 99

/-- With W_row_stride=2, regular forward loads weights 1,3,5. Gemma uses
contiguous weights 1,2,3 and adds one; its smaller weight allocation suffices. -/
theorem forward_witness (isGemma : Bool) (kernel : Kernel)
    (h : Correctness.forward_correct_target kernel isGemma "y" "x" "w" "r" 4 4 2 2 3 4 0) : ForwardResult isGemma kernel := by
  have hb : Correctness.ForwardBounds isGemma "y" "x" "w" "r" 4 4 2 2 3 4 initial (allocation isGemma).extent := by
    refine ⟨?_,?_,?_,by change 2<3;decide⟩
    · intro j hj;change 1*4+j.val<7;omega
    · intro j hj
      cases isGemma
      · change j.val*2<5;omega
      · exact hj
    · intro j hj;change 1*4+j.val<8;omega
  obtain ⟨v,he,hy,hr,hf⟩ := h (allocation isGemma) (disjoint isGemma) (coverage isGemma) initial rfl hb
    (by decide) (by decide) (by decide) (by
      change 0<RealMath.meanSquare 3 (fun _ : Fin 4 => (1 : ℝ))+0
      norm_num [ones_mean_square])
  refine ⟨v,he,?_,?_,?_⟩
  · intro j
    have hv := hy ⟨j.val,by omega⟩ j.isLt
    change v.readMem "flat" (11+(1*4+j.val))=_ at hv
    rw [show 11+(1*4+j.val)=15+j.val by omega] at hv
    rw [hv]
    change RealMath.output isGemma 3 0 (fun _ : Fin 4 => 1)
      (fun k : Fin 4 => (ForwardRun.weightOffset isGemma 2 k.val : ℝ)+1)
      ⟨j.val,by omega⟩=if isGemma then (j.val : ℝ)+2 else 2*(j.val : ℝ)+1
    cases isGemma <;> norm_num [RealMath.output,ones_inv,ForwardRun.weightOffset]
    all_goals ring
  · change v.readMem "flat" 33=_ at hr
    rw [hr]
    change RealMath.invStd 3 0 (fun _ : Fin 4 => 1)=1
    exact ones_inv
  · rw [hf "flat" 18 (Or.inr ⟨?_,by change 18≠33;decide⟩)]
    · rfl
    · intro j hj;change 18≠11+(1*4+j.val);omega

theorem forward_executes (input weight : Storage) (isGemma swapped : Bool) :
    ForwardResult isGemma (ForwardIO.source input weight isGemma swapped "y" "x" "w" "r" 4 4 2 2 3 4 0).eraseDType.toAlgKernel := by
  cases isGemma <;> cases swapped
  · exact forward_witness Bool.false _ (Correctness.forward_correctness input weight "y" "x" "w" "r" 4 4 2 2 3 4 0)
  · exact forward_witness Bool.false _ (Correctness.forward_index_commuted_correctness input weight "y" "x" "w" "r" 4 4 2 2 3 4 0)
  · exact forward_witness Bool.true _ (Correctness.gemma_correctness input weight "y" "x" "w" "r" 4 4 2 2 3 4 0)
  · exact forward_witness Bool.true _ (Correctness.gemma_index_commuted_correctness input weight "y" "x" "w" "r" 4 4 2 2 3 4 0)

def BackwardResult (isGemma : Bool) (kernel : Kernel) : Prop :=
  ∃ v,exec ((allocation Bool.true).flattenKernel kernel) ((allocation Bool.true).flattenState initial)=some v ∧
    (∀ j : Fin 3,v.readMem "flat" (15+j.val)=2*(j.val : ℝ)-(if isGemma then 20 else 14)) ∧
    v.mem "flat" 18=MemCell.of .int 99

/-- The caller supplies inv=2. Both GEMMA branches execute with only three
contiguous weights even though WR=99; dW is absent from the allocation. -/
theorem backward_witness (isGemma : Bool) (kernel : Kernel)
    (h : Correctness.backward_correct_target kernel isGemma "y" "x" "w" "r" 4 4 2 3 4) : BackwardResult isGemma kernel := by
  have hb : Correctness.BackwardBounds "y" "x" "w" "r" 4 4 2 3 4 initial (allocation Bool.true).extent := by
    refine ⟨?_,?_,?_,by change 2<3;decide⟩
    · intro j hj;change 1*4+j.val<8;omega
    · intro j hj;change 1*4+j.val<7;omega
    · intro j hj;exact hj
  obtain ⟨v,he,hy,hf⟩ := h (allocation Bool.true) (disjoint Bool.true) (coverage Bool.true) initial rfl hb (by decide) (by decide)
  refine ⟨v,he,?_,?_⟩
  · intro j
    have hv := hy ⟨j.val,by omega⟩ j.isLt
    change v.readMem "flat" (11+(1*4+j.val))=_ at hv
    rw [show 11+(1*4+j.val)=15+j.val by omega] at hv
    rw [hv]
    have hdy (k : Fin 4) : initial.readMem "y" (4+k.val)=(if k.val<3 then (1 : ℝ) else 0) := by
      fin_cases k <;> rfl
    change RealMath.gradient isGemma 3 2 (fun _ : Fin 4 => 1)
      (fun k : Fin 4 => initial.readMem "y" (4+k.val))
      (fun k : Fin 4 => (k.val : ℝ)+1) ⟨j.val,by omega⟩=2*(j.val : ℝ)-(if isGemma then 20 else 14)
    simp_rw [hdy]
    cases isGemma <;> fin_cases j
    all_goals norm_num [RealMath.gradient,RealMath.padded,Fin.sum_univ_succ]
  · rw [hf "flat" 18 (Or.inr (by intro j hj;change 18≠11+(1*4+j.val);omega))]
    rfl

theorem backward_executes (input weight : Storage) (isGemma swapped : Bool) :
    BackwardResult isGemma (backward input weight swapped "y" "x" "w" "r" "dw" 4 4 99 2 101 3 4 0 isGemma).eraseDType.toAlgKernel := by
  cases swapped
  · exact backward_witness isGemma _ (Correctness.backward_correctness input weight isGemma "y" "x" "w" "r" "dw" 4 4 99 2 101 3 4 0)
  · exact backward_witness isGemma _ (Correctness.backward_index_commuted_correctness input weight isGemma "y" "x" "w" "r" "dw" 4 4 99 2 101 3 4 0)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FastRmsLayernorm.Execution
