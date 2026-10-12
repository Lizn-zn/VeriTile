import bench.tritonbench_g.context_attn_fwd.Proofs.StatsExecution
import bench.tritonbench_g.context_attn_fwd.Proofs.ValuesExecution
import bench.tritonbench_g.context_attn_fwd.Proofs.CellMath
namespace VeriTile.Bench.TritonBenchG.ContextAttnFwd.NumericUpdate
open VeriTile.Triton Formula Program CellMath
open VeriTile.Bench.TritonBenchG.ContextAttnBloom.Math
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 8000
set_option linter.unusedSimpArgs false

def maximum (p : Formula.Layout) (S c : Nat) (g : Fin p.BM → Nat → Fin S → ℝ × ℝ) : Tile .real [p.BM] :=
  ⟨fun r => (gStateBot S (c*p.BN) (g r.1 0)).1⟩
def denominator (p : Formula.Layout) (S c : Nat) (g : Fin p.BM → Nat → Fin S → ℝ × ℝ) : Tile .real [p.BM] :=
  ⟨fun r => some (gStateBot S (c*p.BN) (g r.1 0)).2.1⟩
def accumulator (p : Formula.Layout) (S c : Nat) (g : Fin p.BM → Nat → Fin S → ℝ × ℝ) : Tile .real [p.BM,p.D] :=
  ⟨fun idx => some (gStateBot S (c*p.BN) (g idx.1 idx.2.1.val)).2.2⟩

def update (p : Formula.Layout) (S c : Nat) (g : Fin p.BM → Nat → Fin S → ℝ × ℝ)
    (qk : Tile .real [p.BM,p.BN]) (value : Tile .real [p.BN,p.D]) (blockMax : Tile .real [p.BM]) :=
  let mi := maximum p S c g
  let li := denominator p S c g
  let acc := accumulator p S c g
  let mx := StatsExecution.maximum p mi blockMax
  let al := StatsExecution.alpha p mi mx
  let prob := StatsExecution.probabilities p qk mx
  let lij : Tile .real [p.BM] := StatsExecution.denominator p qk mx
  (mx,StatsExecution.updatedDenominator p li al lij,
    ValuesExecution.updated p prob value (StatsExecution.scaledAcc p acc al))

theorem correct (p : Formula.Layout) (S c : Nat) (hBN : 0 < p.BN) (hwin : (c+1)*p.BN ≤ S)
    (g : Fin p.BM → Nat → Fin S → ℝ × ℝ)
    (hscore : ∀ a dd j, (g a 0 j).1 = (g a dd j).1)
    (qkT : Tile .real [p.BM,p.BN]) (vloadT : Tile .real [p.BN,p.D]) (rmaxT : Tile .real [p.BM])
    (hrm : Tile.reduceMaxDrop (⟨1,by simp⟩ : Fin [p.BM,p.BN].length) qkT = some rmaxT)
    (hqk : ∀ a j dd, qkT.data (a,j,PUnit.unit) =
      some (g a dd ⟨c*p.BN+j.val,gBlock_idx_lt S p.BN c hwin j⟩).1)
    (hvf : ∀ a j d, vloadT.data (j,d,PUnit.unit) =
      some (g a d.val ⟨c*p.BN+j.val,gBlock_idx_lt S p.BN c hwin j⟩).2) :
    update p S c g qkT vloadT rmaxT =
      (maximum p S (c+1) g,denominator p S (c+1) g,accumulator p S (c+1) g) := by
  let d0 : Nat := 0
  let mtile := maximum p S c g
  let ltile := denominator p S c g
  let acctile := accumulator p S c g
  let mx := StatsExecution.maximum p mtile rmaxT
  let prob := StatsExecution.probabilities p qkT mx
  let lij : Tile .real [p.BM] := StatsExecution.denominator p qkT mx
  let al := StatsExecution.alpha p mtile mx
  let ln := StatsExecution.updatedDenominator p ltile al lij
  let ac := StatsExecution.scaledAcc p acctile al
  set st := gStateBot S (c * p.BN) with hstdef
  set M' := fun (a : Fin p.BM) (dd : Nat) => (gStateBot S ((c + 1) * p.BN) (g a dd)).1 with hM'def
  have hsucc : ∀ (a : Fin p.BM) (dd : Nat),
      gStateBot S ((c + 1) * p.BN) (g a dd)
        = (M' a dd,
           (st (g a dd)).2.1 * (WithBot.realExp2 (WithBot.realSub (st (g a dd)).1 (M' a dd))).unbotD 0
             + Finset.univ.sum (fun jL : Fin p.BN =>
                 pow2 ((g a dd ⟨c * p.BN + jL.val, gBlock_idx_lt S p.BN c hwin jL⟩).1 - (M' a dd).unbotD 0)),
           (st (g a dd)).2.2 * (WithBot.realExp2 (WithBot.realSub (st (g a dd)).1 (M' a dd))).unbotD 0
             + Finset.univ.sum (fun jL : Fin p.BN =>
                 pow2 ((g a dd ⟨c * p.BN + jL.val, gBlock_idx_lt S p.BN c hwin jL⟩).1 - (M' a dd).unbotD 0)
                   * (g a dd ⟨c * p.BN + jL.val, gBlock_idx_lt S p.BN c hwin jL⟩).2)) := by
    intro a dd
    have he := gStateBot_succ_explicit S p.BN c (g a dd) hwin
    simp only [] at he
    have hM'eq : M' a dd = (gStateBot S (c * p.BN) (g a dd)).1 ⊔ Finset.univ.sup (fun jL : Fin p.BN =>
        ((g a dd ⟨c * p.BN + jL.val, gBlock_idx_lt S p.BN c hwin jL⟩).1 : WithBot ℝ)) := by
      rw [hM'def]; simp only []; rw [he]
    rw [hM'eq]; exact he
  have hM'ne : ∀ (a : Fin p.BM) (dd : Nat), M' a dd ≠ ⊥ := by
    intro a dd
    rw [hM'def]; simp only []
    rw [gStateBot_fst_eq_runningMax]
    exact gRunningMax_succ_ne_bot S p.BN c (g a dd) hBN hwin
  have hMdd : ∀ (a : Fin p.BM) (dd : Nat), M' a d0 = M' a dd := by
    intro a dd
    rw [hM'def]; simp only []
    exact (gStateBot_score_congr S ((c + 1) * p.BN) _ _
      (fun j => hscore a dd j)).1
  have hst1 : ∀ (a : Fin p.BM) (dd : Nat), (st (g a d0)).1 = (st (g a dd)).1 := by
    intro a dd
    rw [hstdef]
    exact (gStateBot_score_congr S (c * p.BN) _ _
      (fun j => hscore a dd j)).1
  have hmc : ∀ a : Fin p.BM, rmaxT.data (a,PUnit.unit) =
      Finset.univ.sup (fun j : Fin p.BN => ((g a d0 ⟨c*p.BN+j.val,gBlock_idx_lt S p.BN c hwin j⟩).1 : WithBot ℝ)) := by
    intro a
    exact ctxg_reduceMaxDrop_data_row hBN qkT rmaxT hrm a _ (fun j => hqk a j d0)
  have hmxc : ∀ a : Fin p.BM, mx.data (a,PUnit.unit) = M' a d0 := by
    intro a
    rw [show mx = Tile.select (Tile.cop ComparableDType.real.gt Broadcast.nil.consSame mtile rmaxT) mtile rmaxT from rfl]
    rw [ctxg_mij_max mtile rmaxT a ((st (g a d0)).1) _ rfl (hmc a)]
    rw [hM'def]
    simp only []
    rw [congrArg Prod.fst (gStateBot_succ_explicit S p.BN c (g a d0) hwin)]
  have halc : ∀ (a : Fin p.BM) (dd : Nat), al.data (a,PUnit.unit) =
      some ((WithBot.realExp2 (WithBot.realSub (st (g a dd)).1 (M' a dd))).unbotD 0) := by
    intro a dd
    change WithBot.realExp2 ((Tile.bop NumericDType.real.sub Broadcast.nil.consSame mtile mx).data (a,PUnit.unit)) = _
    simp only [Tile.bop_data,Broadcast.leftIndex,Broadcast.rightIndex,NumericDType.sub]
    rw [show mtile.data (a,PUnit.unit) = (st (g a d0)).1 from rfl,
      hmxc a,hMdd a dd,hst1 a dd]
    exact ctxg_realExp2_eq_some_unbotD _
  have hmxf : ∀ (a : Fin p.BM) (dd : Nat), mx.data (a,PUnit.unit) = some ((M' a dd).unbotD 0) := by
    intro a dd
    rw [hmxc a,hMdd a dd]
    cases hM : M' a dd with
    | bot => exact absurd hM (hM'ne a dd)
    | coe v => rfl
  have hpc : ∀ (a : Fin p.BM) (j : Fin p.BN) (dd : Nat), prob.data (a,j,PUnit.unit) =
      some (pow2 ((g a dd ⟨c*p.BN+j.val,gBlock_idx_lt S p.BN c hwin j⟩).1-(M' a dd).unbotD 0)) := by
    intro a j dd
    change WithBot.realExp2 ((Tile.bop NumericDType.real.sub Broadcast.nil.consR.consSame qkT
      (Tile.expandDim ⟨1,by simp⟩ mx)).data (a,j,PUnit.unit)) = _
    rw [ctx_qk_sub_mij_cell qkT mx a j _ _ (hqk a j dd) (hmxf a dd)]
    simp [WithBot.realExp2,pow2,mul_comm]
  have hlij : ∀ (a : Fin p.BM) (dd : Nat), lij.data (a,PUnit.unit) =
      some (Finset.univ.sum (fun j : Fin p.BN =>
        pow2 ((g a dd ⟨c*p.BN+j.val,gBlock_idx_lt S p.BN c hwin j⟩).1-(M' a dd).unbotD 0))) := by
    intro a dd
    change (Tile.reduceSumDrop (⟨1,by simp⟩ : Fin [p.BM,p.BN].length) prob).data (a,PUnit.unit) = _
    rw [Tile.reduceSumDrop_data]
    simp only [TileShape.insertAxisIndex]
    refine Eq.trans (Finset.sum_congr rfl (fun (j : Fin p.BN) _ => hpc a j dd)) ?_
    exact ctxg_withBot_sum_some _
  have hln : ∀ (a : Fin p.BM) (dd : Nat), ln.data (a,PUnit.unit) =
      some (gStateBot S ((c+1)*p.BN) (g a dd)).2.1 := by
    intro a dd
    change (Tile.bop NumericDType.real.add Broadcast.nil.consSame
      (Tile.bop NumericDType.real.mul Broadcast.nil.consSame ltile al) lij).data (a,PUnit.unit) = _
    simp only [Tile.bop_data,Broadcast.leftIndex,Broadcast.rightIndex]
    rw [halc a dd,hlij a dd,
      show ltile.data (a,PUnit.unit) = some (st (g a dd)).2.1 from
        congrArg some (gStateBot_score_congr S (c*p.BN) (g a d0) (g a dd) (fun j => hscore a dd j)).2]
    rw [hsucc a dd]
    rfl
  have hac : ∀ idx : TileIndex [p.BM,p.D], ac.data idx =
      some ((st (g idx.1 idx.2.1.val)).2.2 *
        (WithBot.realExp2 (WithBot.realSub (st (g idx.1 idx.2.1.val)).1 (M' idx.1 idx.2.1.val))).unbotD 0) := by
    intro idx
    change (Tile.bop NumericDType.real.mul Broadcast.nil.consR.consSame acctile (Tile.expandDim ⟨1,by simp⟩ al)).data idx = _
    simp only [Tile.bop_data,Broadcast.leftIndex,Broadcast.rightIndex,Tile.expandDim_data,TileShape.dropInsertedIndex]
    rw [halc idx.1 idx.2.1.val]
    rfl
  change (mx,ln,ValuesExecution.updated p prob vloadT ac) = _
  apply Prod.ext
  · apply Tile.ext
    intro idx
    exact hmxc idx.1
  · apply Prod.ext
    · apply Tile.ext
      intro idx
      exact hln idx.1 d0
    · apply Tile.ext
      rintro ⟨i,d,⟨⟩⟩
      change (Tile.bop NumericDType.real.add Broadcast.nil.consSame.consSame (Tile.dot [] prob vloadT) ac).data (i,d,PUnit.unit) =
        some (gStateBot S ((c+1)*p.BN) (g i d.val)).2.2
      simp only [Tile.bop_data,Broadcast.leftIndex,Broadcast.rightIndex]
      rw [ctxg_dot_row prob vloadT i d _ _ (fun j => hpc i j d.val) (fun j => hvf i j d),hac]
      rw [hsucc i d.val]
      simp only [NumericDType.add,WithBot.realAdd,Option.map₂,Option.bind,Option.map]
      exact congrArg some (add_comm _ _)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnFwd.NumericUpdate
