import bench.tritonbench_g.chunk_gated_attention.Proofs.RealHExecution
import VeriTile.Meta.Disprove

/-! A complete-source real-semantics counterexample to dropping the zero-tail
premise. T=1 with 16-wide dot tiles gives fifteen inactive time lanes. Their
unspecified K/V values are 1; valid inputs and gate values are zero. The actual
source stores 15 while the zero-extended formula gives 0. This is a model
counterexample, not an observation of a GPU choosing those unspecified values. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGatedAttention.UndefinedTail
open VeriTile.Triton RealHMath RealHProgram RealHTiles
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 6000
set_option linter.unusedSimpArgs false

private abbrev p : Layout := ⟨16,16,1,16,16,1,256,16,1,1,16,16,16,16,16,1⟩
private def r : Regions := ⟨"k","v","g","h","h0","ht"⟩
private def a : Inputs := ⟨fun _ => 0,fun _ => 0,fun _ => 0,fun _ => 0⟩
private def padding : RegionName → Nat → ℝ := fun region _ => if region.name = "k" ∨ region.name = "v" then 1 else 0
private def u : BlockState where
  mem := fun _ _ => MemCell.real 0
  regs := fun _ _ _ => none
  pids := fun _ => 0
  undef := padding
private def l : Launch := ⟨0,0,0⟩

def kernel (gateK swapped : Bool) : Kernel :=
  (source p r .fp16 .bf16 .bf16 .bf16 .fp32 gateK Bool.false Bool.true swapped).eraseDType.toAlgKernel

private theorem step_value (gateK : Bool) :
    advance (loaded p l r a padding 0) gateK (fun _ _ => 0) = fun _ _ => 15 := by
  funext e j
  have he : e.val < 16 := e.isLt
  have hj : j.val < 16 := j.isLt
  have heEnd : ¬ (240+e.val < 16) := by omega
  have hjEnd : ¬ (240+j.val < 16) := by omega
  cases gateK <;>
    norm_num [advance,scaled,product,gatedKey,gatedValue,loaded,keyData,valueData,endpointData,
      keyIndex,valueIndex,timeIndex,p,l,r,a,padding,he,hj,heEnd,hjEnd,Fin.sum_univ_succ]
  all_goals exact sub_self _

private theorem history_injective : RealHStore.NoCollision p l (RealHStep.historyBase p l 0) p.historyKey p.historyValue := by
  intro e f i j ha hb he
  dsimp [RealHStore.active,keyIndex,valueIndex,RealHStore.address,RealHStep.historyBase,p,l] at ha hb he
  have h : e.val = f.val ∧ i.val = j.val := by omega
  exact ⟨Fin.ext h.1,Fin.ext h.2⟩
private theorem final_injective : RealHStore.NoCollision p l (RealHFinal.base p l) p.V 1 := by
  intro e f i j ha hb he
  dsimp [RealHStore.active,keyIndex,valueIndex,RealHStore.address,RealHFinal.base,p,l] at ha hb he
  have h : e.val = f.val ∧ i.val = j.val := by omega
  exact ⟨Fin.ext h.1,Fin.ext h.2⟩

theorem source_stores_fifteen (gateK swapped : Bool) :
    ∃ v, exec (kernel gateK swapped) u = some v ∧ v.readMem r.final 0 = 15 := by
  let start := RealHInit.initial p r a.seed (fun _ => 0) Bool.false u
  let entry := start.setReg "i_t" .nat [] (Tile.scalar 0)
  have hei := RealHInit.executes p r a.seed (fun _ => 0) Bool.false u (by funext off; simp [u,padding,r])
    (by intro off; simp [u,r,a,BlockState.readMem])
  obtain ⟨mid,he,hm⟩ := RealHStep.executes p l r a padding 0 gateK swapped (fun _ _ => 0) entry
    (by constructor <;> decide) rfl history_injective
    (by simp [entry,start,RealHInit.initial,RealHInit.registers,u,l])
    (by simp [entry,start,RealHInit.initial,RealHInit.registers,u,l])
    (by simp [entry,start,RealHInit.initial,RealHInit.registers,u,l])
    (by simp [entry])
    (by intro off; simp [entry,start,RealHInit.initial,RealHInit.registers,u,a,BlockState.readMem])
    (by intro off; simp [entry,start,RealHInit.initial,RealHInit.registers,u,a,BlockState.readMem])
    (by intro off; simp [entry,start,RealHInit.initial,RealHInit.registers,u,a,BlockState.readMem])
    (by simp [entry,start,RealHInit.initial,RealHInit.registers])
  have hel : stepStmt (RealHLoop.statement p r gateK swapped) start = some mid := by
    simp only [RealHLoop.statement,stepStmt]
    change stepStmts (iteration p r gateK swapped) (start.setReg "i_t" .nat [] (Tile.scalar 0)) = _ at he
    rw [stepForRangeAux]
    simp only [Nat.reduceLT,↓reduceIte,he,bind,Option.bind_some]
    rw [stepForRangeAux]
    rfl
  have hcarry : mid.regs .real [p.BK,p.BV] "b_h" = some (stateTile p (fun _ _ => 15)) := by
    simpa only [step_value] using hm.carry
  have hef := RealHFinal.executes p l r Bool.true (fun _ _ => 15) mid hm.keyIndex hm.valueIndex hm.headIndex hcarry
  refine ⟨RealHFinal.output p l r Bool.true (fun _ _ => 15) mid,?_,?_⟩
  · rw [kernel,exec,source_body,body,List.append_assoc,stepStmts.append_some hei]
    change stepStmts (RealHLoop.statement p r gateK swapped :: finalization p r Bool.true) start = _
    rw [stepStmts.cons_some hel]
    exact hef
  · exact RealHFinal.readback p l r (fun _ _ => 15) mid final_injective 0 0 (by decide)

refutation final_state_without_zero_padding (gateK swapped : Bool) : ∀ v,
    exec (kernel gateK swapped) u = some v →
      v.readMem r.final 0 = closed p l a gateK Bool.false p.NT 0 0 := by
  obtain ⟨v,he,hv⟩ := source_stores_fifteen gateK swapped
  have hz : closed p l a gateK Bool.false p.NT 0 0 = 0 := by
    cases gateK <;> norm_num [closed,seed,contribution,keyValue,valueValue,a,p]
  intro h
  have hbad := h v he
  rw [hv,hz] at hbad
  norm_num at hbad

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGatedAttention.UndefinedTail
