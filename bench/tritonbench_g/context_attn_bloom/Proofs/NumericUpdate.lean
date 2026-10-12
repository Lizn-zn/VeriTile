import bench.tritonbench_g.context_attn_bloom.Proofs.ScalesExecution
import bench.tritonbench_g.context_attn_bloom.Proofs.NormalizerExecution
import bench.tritonbench_g.context_attn_bloom.Proofs.CommitExecution
namespace VeriTile.Bench.TritonBenchG.ContextAttnBloom.NumericUpdate
open VeriTile.Triton Math Program RealSetup NumericOps
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 8000
set_option linter.unusedSimpArgs false

def maximum (p : Layout) (S c : Nat) (g : Fin p.BM → Nat → Fin S → ℝ × ℝ) : Tile .real [p.BM] :=
  ⟨fun r => (gStateBot S (c*p.BN) (g r.1 0)).1.map (·*Real.log 2)⟩
def denominator (p : Layout) (S c : Nat) (g : Fin p.BM → Nat → Fin S → ℝ × ℝ) : Tile .real [p.BM] :=
  ⟨fun r => some (gStateBot S (c*p.BN) (g r.1 0)).2.1⟩
def accumulator (p : Layout) (S c : Nat) (qpos : Fin p.BM → Nat)
    (g : Fin p.BM → Nat → Fin S → ℝ × ℝ) : Tile .real [p.BM,p.D] :=
  ⟨fun idx => some (gAccN S p.BN (qpos idx.1) (g idx.1 idx.2.1.val) c)⟩

def update (p : Layout) (S c : Nat) (g : Fin p.BM → Nat → Fin S → ℝ × ℝ)
    (qpos rows : Fin p.BM → Nat) (prompt : Int)
    (qk : Tile .real [p.BM,p.BN]) (value : Tile .real [p.BN,p.D]) (blockMax : Tile .real [p.BM]) :=
  let mi := maximum p S c g
  let li := denominator p S c g
  let acc := accumulator p S c qpos g
  let prob := StatsExecution.exponentials p qk blockMax
  let lij := StatsExecution.denominator p qk blockMax
  let mn := NormalizerExecution.newMaximum p mi blockMax
  let al := NormalizerExecution.alpha p mi blockMax
  let be := NormalizerExecution.beta p mi blockMax
  let ln := NormalizerExecution.newDenominator p mi blockMax li lij
  let prob := ScalesExecution.probabilities p prob be ln
  let acc := ScalesExecution.accumulator p rows prompt (c*p.BN) li ln al acc
  (mn,ln,CommitExecution.accumulator p prob value acc)

theorem correct (p : Layout) (S c : Nat) (hBN : 0 < p.BN) (hwin : (c+1)*p.BN ≤ S)
    (g : Fin p.BM → Nat → Fin S → ℝ × ℝ)
    (hscore : ∀ a dd j, (g a 0 j).1 = (g a dd j).1)
    (qpos rows : Fin p.BM → Nat) (prompt : Int)
    (hguard : ∀ a, ((c*p.BN : Nat) : Int) ≤ rows a+prompt ↔ c*p.BN ≤ qpos a)
    (qkT : Tile .real [p.BM,p.BN]) (vloadT : Tile .real [p.BN,p.D]) (rmaxT : Tile .real [p.BM])
    (hrm : Tile.reduceMaxDrop (⟨1,by simp⟩ : Fin [p.BM,p.BN].length) qkT = some rmaxT)
    (hqk : ∀ a j dd, qkT.data (a,j,PUnit.unit) =
      some ((g a dd ⟨c*p.BN+j.val,gBlock_idx_lt S p.BN c hwin j⟩).1*Real.log 2))
    (hvf : ∀ a j d, vloadT.data (j,d,PUnit.unit) =
      some (g a d.val ⟨c*p.BN+j.val,gBlock_idx_lt S p.BN c hwin j⟩).2) :
    update p S c g qpos rows prompt qkT vloadT rmaxT =
      (maximum p S (c+1) g,denominator p S (c+1) g,accumulator p S (c+1) qpos g) := by
  simp only [Nat.cast_mul] at hguard
  let d0 : Nat := 0
  set mtile := maximum p S c g with hmtile
  set ltile := denominator p S c g with hltile
  set acctile := accumulator p S c qpos g with hacctile
  set pexpT := StatsExecution.exponentials p qkT rmaxT with hpexpeq
  set lijT : Tile .real [p.BM] := StatsExecution.denominator p qkT rmaxT with hlijeq
  set miNewT := NormalizerExecution.newMaximum p mtile rmaxT with hminew
  set alphaT := NormalizerExecution.alpha p mtile rmaxT with haleq
  set betaT := NormalizerExecution.beta p mtile rmaxT with hbetaeq
  set liNewT := NormalizerExecution.newDenominator p mtile rmaxT ltile lijT with hlineweq
  set pscaleT := ScalesExecution.probabilityScale p betaT liNewT with hpscaleeq
  set p2T := ScalesExecution.probabilities p pexpT betaT liNewT with hp2eq
  set accscale2T := ScalesExecution.accumulatorScale p rows prompt (c*p.BN) ltile liNewT alphaT
  have hascale2eq : accscale2T = Tile.select
      (⟨fun idx : TileIndex [p.BM] => decide (c*p.BN ≤ qpos idx.1)⟩ : Tile .bool [p.BM])
      (Tile.bop NumericDType.real.mul Broadcast.nil.consSame
        (Tile.bop NumericDType.real.div Broadcast.nil.consSame ltile liNewT) alphaT)
      (⟨fun _ => some (1.0 : ℝ)⟩ : Tile .real [p.BM]) := by
    norm_num [accscale2T,ScalesExecution.accumulatorScale,ScalesExecution.rawScale,ScalesExecution.guard,
      Tile.vec,hguard]
  set acc1T := ScalesExecution.accumulator p rows prompt (c*p.BN) ltile liNewT alphaT acctile with hacc1eq
  change mtile = ⟨fun r => (gStateBot S (c*p.BN) (g r.1 d0)).1.map (·*Real.log 2)⟩ at hmtile
  change ltile = ⟨fun r => some (gStateBot S (c*p.BN) (g r.1 d0)).2.1⟩ at hltile
  change acctile = ⟨fun idx => some (gAccN S p.BN (qpos idx.1) (g idx.1 idx.2.1.val) c)⟩ at hacctile
  change pexpT = Tile.uop WithBot.realExp (Tile.bop NumericDType.real.sub Broadcast.nil.consR.consSame
    qkT (Tile.expandDim ⟨1,by simp⟩ rmaxT)) at hpexpeq
  change lijT = Tile.reduceSumDrop (⟨1,by simp⟩ : Fin [p.BM,p.BN].length) pexpT at hlijeq
  change miNewT = Tile.select (Tile.cop ComparableDType.real.gt Broadcast.nil.consSame mtile rmaxT) mtile rmaxT at hminew
  change alphaT = Tile.uop WithBot.realExp (Tile.bop NumericDType.real.sub Broadcast.nil.consSame mtile miNewT) at haleq
  change betaT = Tile.uop WithBot.realExp (Tile.bop NumericDType.real.sub Broadcast.nil.consSame rmaxT miNewT) at hbetaeq
  change liNewT = Tile.bop NumericDType.real.add Broadcast.nil.consSame
    (Tile.bop NumericDType.real.mul Broadcast.nil.consSame alphaT ltile)
    (Tile.bop NumericDType.real.mul Broadcast.nil.consSame betaT lijT) at hlineweq
  change pscaleT = Tile.bop NumericDType.real.div Broadcast.nil.consSame betaT liNewT at hpscaleeq
  change p2T = Tile.bop NumericDType.real.mul Broadcast.nil.consR.consSame pexpT
    (Tile.expandDim ⟨1,by simp⟩ pscaleT) at hp2eq
  change acc1T = Tile.bop NumericDType.real.mul Broadcast.nil.consR.consSame acctile
    (Tile.expandDim ⟨1,by simp⟩ accscale2T) at hacc1eq
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
  have hlog2pos : (0 : ℝ) < Real.log 2 := Real.log_pos (by norm_num)
  set L : ℝ := Real.log 2 with hLdef
  have hmapsup2 : ∀ x y : WithBot ℝ, (x.map (· * L)) ⊔ (y.map (· * L)) = (x ⊔ y).map (· * L) := by
    intro x y
    cases x with
    | bot => simp
    | coe xr => cases y with
      | bot => simp
      | coe yr =>
        simp only [WithBot.map_coe, ← WithBot.coe_sup, WithBot.coe_le_coe]
        rcases le_total xr yr with h | h
        · rw [sup_eq_right.mpr h, sup_eq_right.mpr (mul_le_mul_of_nonneg_right h hlog2pos.le)]
        · rw [sup_eq_left.mpr h, sup_eq_left.mpr (mul_le_mul_of_nonneg_right h hlog2pos.le)]
  have hmapFsup : ∀ (f : Fin p.BN → WithBot ℝ),
      Finset.univ.sup (fun jL : Fin p.BN => (f jL).map (· * L)) = (Finset.univ.sup f).map (· * L) := by
    intro f
    refine Finset.cons_induction ?_ ?_ (Finset.univ : Finset (Fin p.BN))
    · simp
    · intro a s ha ih
      rw [Finset.sup_cons, Finset.sup_cons, ih, hmapsup2]
  have hexpmap : ∀ x y : WithBot ℝ,
      WithBot.realExp (WithBot.realSub (x.map (· * L)) (y.map (· * L)))
        = WithBot.realExp2 (WithBot.realSub x y) := by
    intro x y
    cases x with
    | bot => cases y <;> rfl
    | coe xr => cases y with
      | bot => rfl
      | coe yr =>
        show WithBot.realExp (some (xr * L - yr * L)) = WithBot.realExp2 (some (xr - yr))
        rw [WithBot.realExp2_some]
        show some (Real.exp (xr * L - yr * L)) = some (Real.exp ((xr - yr) * Real.log 2))
        rw [hLdef]; congr 1; congr 1; ring
  -- m_ij cell = (block base-2 sup) · L
  have hmijc : ∀ a : Fin p.BM, rmaxT.data (a, PUnit.unit)
      = (Finset.univ.sup (fun jL : Fin p.BN =>
          ((g a d0 ⟨c * p.BN + jL.val, gBlock_idx_lt S p.BN c hwin jL⟩).1 : WithBot ℝ))).map (· * L) := by
    intro a
    rw [ctxg_reduceMaxDrop_data_row hBN qkT rmaxT hrm a
      (fun jL => ((g a d0 ⟨c * p.BN + jL.val, gBlock_idx_lt S p.BN c hwin jL⟩).1 : WithBot ℝ).map (· * L))
      (fun jL => by rw [hqk a jL d0]; rfl)]
    exact hmapFsup _
  have hminewc : ∀ a : Fin p.BM, miNewT.data (a, PUnit.unit) = (M' a d0).map (· * L) := by
    intro a
    rw [hminew]
    rw [ctxg_mij_max mtile rmaxT a ((st (g a d0)).1.map (· * L)) _
      (by rw [hmtile]) (hmijc a)]
    rw [hM'def]; simp only []
    rw [congrArg Prod.fst (gStateBot_succ_explicit S p.BN c (g a d0) hwin)]
    simp only []
    rw [hmapsup2]
  have halc : ∀ (a : Fin p.BM) (dd : Nat), alphaT.data (a, PUnit.unit)
      = some ((WithBot.realExp2 (WithBot.realSub (st (g a dd)).1 (M' a dd))).unbotD 0) := by
    intro a dd
    rw [haleq]
    show WithBot.realExp ((Tile.bop NumericDType.real.sub Broadcast.nil.consSame mtile miNewT).data (a, PUnit.unit)) = _
    have hinner : (Tile.bop NumericDType.real.sub Broadcast.nil.consSame mtile miNewT).data (a, PUnit.unit)
        = WithBot.realSub ((st (g a dd)).1.map (· * L)) ((M' a dd).map (· * L)) := by
      simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.sub]
      rw [hmtile]; simp only []
      rw [hminewc a, hMdd a dd, hst1 a dd]
    rw [hinner, hexpmap]
    exact (ctxg_realExp2_eq_some_unbotD _)
  set Bsup := fun (a : Fin p.BM) (dd : Nat) => Finset.univ.sup (fun jL : Fin p.BN =>
      ((g a dd ⟨c * p.BN + jL.val, gBlock_idx_lt S p.BN c hwin jL⟩).1 : WithBot ℝ)) with hBsupdef
  have hBsupne : ∀ (a : Fin p.BM) (dd : Nat), Bsup a dd ≠ ⊥ := by
    intro a dd
    rw [hBsupdef]; simp only []
    intro hcon
    have h0 := Finset.le_sup (f := fun jL : Fin p.BN =>
        ((g a dd ⟨c * p.BN + jL.val, gBlock_idx_lt S p.BN c hwin jL⟩).1 : WithBot ℝ))
      (Finset.mem_univ (⟨0, hBN⟩ : Fin p.BN))
    rw [hcon] at h0; exact absurd (le_bot_iff.mp h0) (by simp)
  have hBsupdd : ∀ (a : Fin p.BM) (dd : Nat), Bsup a d0 = Bsup a dd := by
    intro a dd
    rw [hBsupdef]; simp only []
    refine Finset.sup_congr rfl (fun jL _ => ?_)
    exact congrArg (fun x : ℝ => (x : WithBot ℝ)) (hscore a dd _)
  have hBsup0 : ∀ a : Fin p.BM, Bsup a d0
      = Finset.univ.sup (fun jL : Fin p.BN =>
          ((g a d0 ⟨c * p.BN + jL.val, gBlock_idx_lt S p.BN c hwin jL⟩).1 : WithBot ℝ)) := by
    intro a; rw [hBsupdef]
  have hrmaxc : ∀ (a : Fin p.BM) (dd : Nat), rmaxT.data (a, PUnit.unit) = some ((Bsup a dd).unbotD 0 * L) := by
    intro a dd
    rw [hmijc a, ← hBsup0 a, hBsupdd a dd]
    rcases hB : Bsup a dd with _ | br
    · exact absurd hB (hBsupne a dd)
    · simp [WithBot.map]
  have hpc : ∀ (a : Fin p.BM) (jL : Fin p.BN) (dd : Nat), pexpT.data (a, jL, PUnit.unit)
      = some (pow2 ((g a dd ⟨c * p.BN + jL.val, gBlock_idx_lt S p.BN c hwin jL⟩).1 - (Bsup a dd).unbotD 0)) := by
    intro a jL dd
    rw [hpexpeq]
    show WithBot.realExp ((Tile.bop NumericDType.real.sub Broadcast.nil.consR.consSame qkT
        (Tile.expandDim ⟨1, by simp⟩ rmaxT)).data (a, jL, PUnit.unit)) = _
    rw [ctx_qk_sub_mij_cell qkT rmaxT a jL _ _ (hqk a jL dd) (hrmaxc a dd)]
    show WithBot.realExp (some _) = _
    simp only [WithBot.realExp, pow2]
    rw [show (g a dd ⟨c * p.BN + jL.val, gBlock_idx_lt S p.BN c hwin jL⟩).1 * L - (Bsup a dd).unbotD 0 * L
          = Real.log 2 * ((g a dd ⟨c * p.BN + jL.val, gBlock_idx_lt S p.BN c hwin jL⟩).1 - (Bsup a dd).unbotD 0)
        from by rw [hLdef]; ring]
  have hlijc : ∀ (a : Fin p.BM) (dd : Nat), lijT.data (a, PUnit.unit)
      = some (Finset.univ.sum (fun jL : Fin p.BN =>
          pow2 ((g a dd ⟨c * p.BN + jL.val, gBlock_idx_lt S p.BN c hwin jL⟩).1 - (Bsup a dd).unbotD 0))) := by
    intro a dd
    rw [hlijeq, Tile.reduceSumDrop_data]
    simp only [TileShape.insertAxisIndex]
    refine Eq.trans (Finset.sum_congr rfl (fun (jL : Fin p.BN) _ => hpc a jL dd)) ?_
    exact ctxg_withBot_sum_some
      (fun jL : Fin p.BN => pow2 ((g a dd ⟨c * p.BN + jL.val, gBlock_idx_lt S p.BN c hwin jL⟩).1 - (Bsup a dd).unbotD 0))
  have hbetav : ∀ (a : Fin p.BM) (dd : Nat), betaT.data (a, PUnit.unit)
      = some ((WithBot.realExp2 (WithBot.realSub (Bsup a dd) (M' a dd))).unbotD 0) := by
    intro a dd
    rw [hbetaeq]
    show WithBot.realExp ((Tile.bop NumericDType.real.sub Broadcast.nil.consSame rmaxT miNewT).data (a, PUnit.unit)) = _
    have hinner : (Tile.bop NumericDType.real.sub Broadcast.nil.consSame rmaxT miNewT).data (a, PUnit.unit)
        = WithBot.realSub ((Bsup a dd).map (· * L)) ((M' a dd).map (· * L)) := by
      simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.sub]
      rw [hmijc a, hminewc a, ← hBsup0 a, hBsupdd a dd, hMdd a dd]
    rw [hinner, hexpmap]
    exact (ctxg_realExp2_eq_some_unbotD _)
  have hbetapow : ∀ (a : Fin p.BM) (jL : Fin p.BN) (dd : Nat),
      (WithBot.realExp2 (WithBot.realSub (Bsup a dd) (M' a dd))).unbotD 0
        * pow2 ((g a dd ⟨c * p.BN + jL.val, gBlock_idx_lt S p.BN c hwin jL⟩).1 - (Bsup a dd).unbotD 0)
      = pow2 ((g a dd ⟨c * p.BN + jL.val, gBlock_idx_lt S p.BN c hwin jL⟩).1 - (M' a dd).unbotD 0) := by
    intro a jL dd
    rcases hB : Bsup a dd with _ | br
    · exact absurd hB (hBsupne a dd)
    · rcases hM : M' a dd with _ | Mr
      · exact absurd hM (hM'ne a dd)
      · have hfac : (WithBot.realExp2 (WithBot.realSub (some br) (some Mr))).unbotD 0
            = Real.exp ((br - Mr) * Real.log 2) := rfl
        have hb' : (WithBot.unbotD 0 (some br) : ℝ) = br := rfl
        have hm' : (WithBot.unbotD 0 (some Mr) : ℝ) = Mr := rfl
        rw [hfac, hb', hm', pow2, pow2, ← Real.exp_add]
        congr 1; ring
  have hliNewv : ∀ (a : Fin p.BM) (dd : Nat), liNewT.data (a, PUnit.unit)
      = some ((gStateBot S ((c + 1) * p.BN) (g a dd)).2.1) := by
    intro a dd
    rw [hlineweq]
    simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex]
    rw [halc a dd, show ltile.data (a, PUnit.unit) = some (st (g a dd)).2.1 from by
        rw [hltile]; simp only []; rw [hstdef]
        exact congrArg some (gStateBot_score_congr S (c * p.BN) (g a d0) (g a dd)
          (fun j => hscore a dd j)).2,
      hbetav a dd, hlijc a dd]
    rw [hsucc a dd]; simp only []
    show WithBot.realAdd (WithBot.realMul (some _) (some _)) (WithBot.realMul (some _) (some _)) = _
    simp only [WithBot.realAdd, WithBot.realMul, Option.map₂, Option.bind, Option.map]
    refine congrArg some ?_
    rw [Finset.mul_sum]
    rw [Finset.sum_congr rfl (fun (jL : Fin p.BN) _ => hbetapow a jL dd)]
    ring
  have hpscalev : ∀ (a : Fin p.BM) (dd : Nat), pscaleT.data (a, PUnit.unit)
      = some ((WithBot.realExp2 (WithBot.realSub (Bsup a dd) (M' a dd))).unbotD 0
          / (gStateBot S ((c + 1) * p.BN) (g a dd)).2.1) := by
    intro a dd
    rw [hpscaleeq]
    simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.div]
    rw [hbetav a dd, hliNewv a dd]; rfl
  have hp2c : ∀ (a : Fin p.BM) (jL : Fin p.BN) (dd : Nat), p2T.data (a, jL, PUnit.unit)
      = some (pow2 ((g a dd ⟨c * p.BN + jL.val, gBlock_idx_lt S p.BN c hwin jL⟩).1 - (M' a dd).unbotD 0)
          / (gStateBot S ((c + 1) * p.BN) (g a dd)).2.1) := by
    intro a jL dd
    rw [hp2eq]
    simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, Tile.expandDim_data,
      TileShape.dropInsertedIndex, TileShape.insertAxisIndex, NumericDType.mul]
    rw [hpc a jL dd, hpscalev a dd]
    show WithBot.realMul (some _) (some _) = _
    simp only [WithBot.realMul, Option.map₂, Option.bind, Option.map]
    refine congrArg some ?_
    rw [← hbetapow a jL dd]; ring
  have hascale2c : ∀ (a : Fin p.BM) (dd : Nat), accscale2T.data (a, PUnit.unit)
      = some (if c * p.BN ≤ qpos a then
          ((st (g a dd)).2.1 / (gStateBot S ((c + 1) * p.BN) (g a dd)).2.1)
            * (WithBot.realExp2 (WithBot.realSub (st (g a dd)).1 (M' a dd))).unbotD 0
          else 1) := by
    intro a dd
    rw [hascale2eq]
    rw [Tile.select_data]
    have hcond : (⟨fun idx : TileIndex [p.BM] => decide (c * p.BN ≤ qpos idx.1)⟩ : Tile .bool [p.BM]).data (a, PUnit.unit)
        = decide (c * p.BN ≤ qpos a) := rfl
    rw [hcond]
    by_cases hg : c * p.BN ≤ qpos a
    · rw [if_pos (by simp only [decide_eq_true_eq]; exact hg), if_pos hg]
      simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.mul, NumericDType.div]
      rw [show ltile.data (a, PUnit.unit) = some (st (g a dd)).2.1 from by
          rw [hltile]; simp only []; rw [hstdef]
          exact congrArg some (gStateBot_score_congr S (c * p.BN) (g a d0) (g a dd)
            (fun j => hscore a dd j)).2,
        hliNewv a dd, halc a dd]
      show WithBot.realMul (WithBot.realDiv (some _) (some _)) (some _) = _
      simp only [WithBot.realMul, WithBot.realDiv, Option.map₂, Option.bind, Option.map]
    · rw [if_neg (by simp only [decide_eq_true_eq]; exact hg), if_neg hg]; norm_num
  have hacc1c : ∀ idx : TileIndex [p.BM, p.D], acc1T.data idx
      = some (gAccN S p.BN (qpos idx.1) (g idx.1 idx.2.1.val) c
          * (if c * p.BN ≤ qpos idx.1 then
              ((st (g idx.1 idx.2.1.val)).2.1 / (gStateBot S ((c + 1) * p.BN) (g idx.1 idx.2.1.val)).2.1)
                * (WithBot.realExp2 (WithBot.realSub (st (g idx.1 idx.2.1.val)).1 (M' idx.1 idx.2.1.val))).unbotD 0
              else 1)) := by
    intro idx
    rw [hacc1eq]
    simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, Tile.expandDim_data,
      TileShape.dropInsertedIndex, TileShape.insertAxisIndex, NumericDType.mul]
    rw [show acctile.data idx = some (gAccN S p.BN (qpos idx.1) (g idx.1 idx.2.1.val) c) from by
        rw [hacctile], hascale2c idx.1 idx.2.1.val]
    show WithBot.realMul (some _) (some _) = _
    simp only [WithBot.realMul, Option.map₂, Option.bind, Option.map]
  change (miNewT,liNewT,CommitExecution.accumulator p p2T vloadT acc1T) = _
  apply Prod.ext
  · apply Tile.ext
    intro a
    exact hminewc a.1
  · apply Prod.ext
    · apply Tile.ext
      intro a
      exact hliNewv a.1 d0
    · apply Tile.ext
      intro idx
      change (Tile.bop NumericDType.real.add Broadcast.nil.consSame.consSame acc1T (Tile.dot [] p2T vloadT)).data idx =
        some (gAccN S p.BN (qpos idx.1) (g idx.1 idx.2.1.val) (c+1))
      simp only [Tile.bop_data,Broadcast.leftIndex,Broadcast.rightIndex]
      have hdotc : (Tile.dot [] p2T vloadT).data (idx.1, idx.2.1, PUnit.unit)
          = some (Finset.univ.sum (fun jL : Fin p.BN =>
              pow2 ((g idx.1 idx.2.1.val ⟨c * p.BN + jL.val, gBlock_idx_lt S p.BN c hwin jL⟩).1 - (M' idx.1 idx.2.1.val).unbotD 0)
                * (g idx.1 idx.2.1.val ⟨c * p.BN + jL.val, gBlock_idx_lt S p.BN c hwin jL⟩).2)
              / (gStateBot S ((c + 1) * p.BN) (g idx.1 idx.2.1.val)).2.1) := by
        rw [ctxg_dot_row p2T vloadT idx.1 idx.2.1 _ _ (fun jL => hp2c idx.1 jL idx.2.1.val)
          (fun jL => by rw [hvf idx.1 jL idx.2.1])]
        refine congrArg some ?_
        rw [Finset.sum_div]
        refine Finset.sum_congr rfl (fun jL _ => ?_)
        ring
      rw [hdotc, hacc1c idx]
      show WithBot.realAdd (some _) (some _) = _
      simp only [WithBot.realAdd, Option.map₂, Option.bind, Option.map]
      refine congrArg some ?_
      rw [gAccN_succ]
      simp only [hstdef, hM'def]
      have hincr : (gStateBot S ((c + 1) * p.BN) (g idx.1 idx.2.1.val)).2.2
            - (gStateBot S (c * p.BN) (g idx.1 idx.2.1.val)).2.2
                * (WithBot.realExp2 (WithBot.realSub (gStateBot S (c * p.BN) (g idx.1 idx.2.1.val)).1
                    (gStateBot S ((c + 1) * p.BN) (g idx.1 idx.2.1.val)).1)).unbotD 0
          = Finset.univ.sum (fun jL : Fin p.BN =>
              pow2 ((g idx.1 idx.2.1.val ⟨c * p.BN + jL.val, gBlock_idx_lt S p.BN c hwin jL⟩).1
                  - ((gStateBot S ((c + 1) * p.BN) (g idx.1 idx.2.1.val)).1).unbotD 0)
                * (g idx.1 idx.2.1.val ⟨c * p.BN + jL.val, gBlock_idx_lt S p.BN c hwin jL⟩).2) := by
        conv_lhs => rw [hsucc idx.1 idx.2.1.val]
        simp only [hstdef, hM'def]; ring
      rw [hincr]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnBloom.NumericUpdate
