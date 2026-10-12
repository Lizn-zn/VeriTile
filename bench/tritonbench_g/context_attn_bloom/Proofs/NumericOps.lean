import bench.tritonbench_g.context_attn_bloom.Proofs.Math
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.ContextAttnBloom.NumericOps
open VeriTile.Triton Math
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 5000

theorem ctxg_reduceMaxDrop_data_row {M N : Nat} (hN : 0 < N) (qk : Tile .real [M, N])
    (rmaxT : Tile .real [M]) (hrm : Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [M,N].length) qk = some rmaxT)
    (r : Fin M) (g : Fin N → WithBot ℝ) (hqk : ∀ jL : Fin N, qk.data (r, jL, PUnit.unit) = g jL) :
    rmaxT.data (r, PUnit.unit) = Finset.univ.sup g := by
  unfold Tile.reduceMaxDrop at hrm
  rw [dif_pos (show 0 < TileShape.axisDim [M,N] (⟨1, by simp⟩ : Fin [M,N].length) from hN)] at hrm
  rw [← Option.some.inj hrm]
  simp only [Finset.sup'_eq_sup]
  exact Finset.sup_congr rfl (fun jL _ => hqk jL)

/-- A `WithBot ℝ` sum of `some`-valued cells is `some` of the real sum. -/
theorem ctxg_withBot_sum_some {N : Nat} (g : Fin N → ℝ) :
    @Finset.sum (Fin N) (WithBot ℝ) _ Finset.univ (fun k => (some (g k) : WithBot ℝ))
      = some (Finset.univ.sum g) :=
  (WithBot.coe_sum Finset.univ g).symm

/-- `realExp2` is total (never `⊥`). -/
theorem ctxg_realExp2_eq_some_unbotD (z : WithBot ℝ) :
    WithBot.realExp2 z = some ((WithBot.realExp2 z).unbotD 0) := by
  cases z <;> rfl

/-- `m_ij = select(m_i > rmax) m_i rmax` collapses to `max` in `WithBot ℝ`. -/
theorem ctxg_mij_max {M : Nat} (m_i rmaxT : Tile .real [M]) (r : Fin M)
    (a b : WithBot ℝ) (hmi : m_i.data (r, PUnit.unit) = a) (hrm : rmaxT.data (r, PUnit.unit) = b) :
    (Tile.select (Tile.cop ComparableDType.real.gt Broadcast.nil.consSame m_i rmaxT) m_i rmaxT).data
        (r, PUnit.unit) = max a b := by
  rw [Tile.select_data, Tile.cop_data]
  simp only [Broadcast.leftIndex, Broadcast.rightIndex, ComparableDType.gt, hmi, hrm]
  by_cases h : a ≤ b
  · rw [if_neg (by simp [not_lt.mpr h]), max_eq_right h]
  · rw [if_pos (by simpa using not_le.mp h), max_eq_left (le_of_lt (not_le.mp h))]

/-- A `dot` row over all `K` keys when both factors are all-`some`. -/
theorem ctxg_dot_row {M K N : Nat} (p : Tile .real [M, K]) (v : Tile .real [K, N])
    (r : Fin M) (d : Fin N) (fp fv : Fin K → ℝ)
    (hp : ∀ jL : Fin K, p.data (r, jL, PUnit.unit) = some (fp jL))
    (hv : ∀ jL : Fin K, v.data (jL, d, PUnit.unit) = some (fv jL)) :
    (Tile.dot [] p v).data (r, d, PUnit.unit) = some (Finset.univ.sum fun jL : Fin K => fp jL * fv jL) := by
  rw [Tile.dot_nil_data]
  rw [show (@Finset.sum (Fin K) (WithBot ℝ) _ Finset.univ
        (fun k => Option.map₂ (· * ·) (p.data (r, k, PUnit.unit)) (v.data (k, d, PUnit.unit))))
      = @Finset.sum (Fin K) (WithBot ℝ) _ Finset.univ (fun k => (some (fp k * fv k) : WithBot ℝ))
      from Finset.sum_congr rfl (fun k _ => by rw [hp k, hv k]; rfl)]
  exact ctxg_withBot_sum_some _

/-- The `qk -= m_ij[:, None]` cell readback (avoids deep recursion inline). -/
theorem ctx_qk_sub_mij_cell {BM BN : Nat} (qkT : Tile .real [BM, BN]) (mijT : Tile .real [BM])
    (i : Fin BM) (jL : Fin BN) (sc mij : ℝ)
    (hqk : qkT.data (i, jL, PUnit.unit) = some sc)
    (hmij : mijT.data (i, PUnit.unit) = some mij) :
    (Tile.bop NumericDType.real.sub Broadcast.nil.consR.consSame qkT
        (Tile.expandDim ⟨1, by simp⟩ mijT)).data (i, jL, PUnit.unit)
      = some (sc - mij) := by
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, Tile.expandDim_data,
    TileShape.dropInsertedIndex, TileShape.insertAxisIndex, hqk, hmij,
    NumericDType.sub, WithBot.realSub, Option.map₂, Option.bind, Option.map]

/-- The `q·k` dot cell is the (unscaled) score (shape-generic `[BM,D]·[D,BN]`). -/
theorem ctx_dot_score_cell {BM BN D : Nat}
    (qtile : Tile .real [BM, D]) (ktile : Tile .real [D, BN]) (i : Fin BM) (j : Fin BN)
    (qf : Fin BM → Fin D → ℝ) (kf : Fin BN → Fin D → ℝ)
    (hq : ∀ e : Fin D, qtile.data (i, e, PUnit.unit) = some (qf i e))
    (hk : ∀ e : Fin D, ktile.data (e, j, PUnit.unit) = some (kf j e)) :
    (Tile.dot [] qtile ktile).data (i, j, PUnit.unit)
      = some (Finset.univ.sum (fun e : Fin D => qf i e * kf j e)) := by
  rw [Tile.dot_nil_data]
  rw [show (@Finset.sum (Fin D) (WithBot ℝ) _ Finset.univ
        (fun e => Option.map₂ (· * ·) (qtile.data (i, e, PUnit.unit)) (ktile.data (e, j, PUnit.unit))))
      = @Finset.sum (Fin D) (WithBot ℝ) _ Finset.univ
          (fun e => (some (qf i e * kf j e) : WithBot ℝ))
      from Finset.sum_congr rfl (fun e _ => by rw [hq e, hk e]; rfl)]
  rw [show (fun e : Fin D => (some (qf i e * kf j e) : WithBot ℝ))
        = (fun e : Fin D => ((qf i e * kf j e : ℝ) : WithBot ℝ)) from rfl,
    ← WithBot.coe_sum]; rfl


/-- Axis-1 `expandDim` over a `real` register (`m_ij[:, None]`/`alpha[:, None]`). -/
@[simp] theorem ctx_evalOp_expandDim_one_real {M : Nat} (name : RegName) (s : BlockState) :
    @evalOp .real [M, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [M] name)) s =
      (s.regs .real [M] name).bind (fun v =>
        some ({ data := fun i : TileIndex [M, 1] => v.data (i.1, PUnit.unit) } : Tile .real [M, 1])) := by
  unfold evalOp; simp [Tile.expandDim]; rfl


/-- `qk = tl.zeros([…])` full-zero eval. -/
theorem bloomQkFull_eval {BM BN : Nat} (s : BlockState) :
    evalOp (Op.full [BM, BN] (Op.const (0 : ℝ))) s
      = some (⟨fun _ : TileIndex [BM, BN] => some (0 : ℝ)⟩ : Tile .real [BM, BN]) := by
  simp [evalOp_full, evalOp_const]

/-- `qk += tl.dot(q, k)` eval (`qk = qk + dot q k`, `qk` is all-zero). -/
theorem bloomQkAddDot_eval {BM BN D : Nat} (s : BlockState)
    (qktile : Tile .real [BM, BN]) (qtile : Tile .real [BM, D]) (ktile : Tile .real [D, BN])
    (hqk : s.regs .real [BM, BN] "qk" = some qktile)
    (hq : s.regs .real [BM, D] "q" = some qtile) (hk : s.regs .real [D, BN] "k" = some ktile) :
    evalOp (Op.add .real Broadcast.nil.consSame.consSame (Op.ref .real [BM, BN] "qk")
        (Op.dot (batch := []) (Op.ref .real [BM, D] "q") (Op.ref .real [D, BN] "k"))) s
      = some (Tile.bop NumericDType.real.add Broadcast.nil.consSame.consSame qktile
          (Tile.dot [] qtile ktile)) := by
  have hdot : evalOp (Op.dot (batch := []) (Op.ref .real [BM, D] "q") (Op.ref .real [D, BN] "k")) s
      = some (Tile.dot [] qtile ktile) := by rw [evalOp_dot]; simp [hq, hk]
  have hdot2 : @evalOp TileDType.real [BM, BN]
      (Op.dot (batch := []) (Op.ref .real [BM, D] "q") (Op.ref .real [D, BN] "k")) s
      = some (Tile.dot [] qtile ktile) := hdot
  rw [evalOp_add]; simp only [evalOp_ref, hqk, hdot2, Option.bind_eq_bind, Option.bind_some]; rfl

/-- `qk *= sm_scale` eval. -/
theorem bloomQkScale_eval {BM BN : Nat} (s : BlockState) (sm : ℝ) (qktile : Tile .real [BM, BN])
    (hqk : s.regs .real [BM, BN] "qk" = some qktile) :
    evalOp (Op.mul .real Broadcast.scalarR (Op.ref .real [BM, BN] "qk") (Op.const sm)) s
      = some (Tile.bop NumericDType.real.mul Broadcast.scalarR qktile (Tile.scalar (some sm))) := by
  rw [evalOp_mul]; simp only [evalOp_ref, evalOp_const, hqk, Option.bind_eq_bind, Option.bind_some]


theorem bloomMij_eval {BM BN : Nat} (s : BlockState) (qkfull : Tile .real [BM, BN]) (rmaxT : Tile .real [BM])
    (hqk : s.regs .real [BM, BN] "qk" = some qkfull)
    (hrm : Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [BM, BN].length) qkfull = some rmaxT) :
    evalOp (Op.reduceMax ⟨1, by simp⟩ Bool.false (Op.ref .real [BM, BN] "qk")) s = some rmaxT := by
  rw [evalOp_reduceMax]; simp only [evalOp_ref, hqk, Option.bind_some]; exact hrm

/-- `l_ij = tl.sum(p, 1)` eval. -/
theorem bloomLij_eval {BM BN : Nat} (s : BlockState) (ptile : Tile .real [BM, BN])
    (hp : s.regs .real [BM, BN] "p" = some ptile) :
    evalOp (Op.reduceSum (⟨1, by simp⟩ : Fin [BM, BN].length) Bool.false (Op.ref .real [BM, BN] "p")) s
      = some (Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [BM, BN].length) ptile) := by
  rw [evalOp_reduceSum]; simp only [evalOp_ref, hp, Option.bind_some]; rfl

/-- `p = tl.exp(qk − m_ij[:, None])` natural-exp eval. -/
theorem bloomP_eval {BM BN : Nat} (s : BlockState) (qk2tile : Tile .real [BM, BN])
    (mij : Tile .real [BM])
    (hqk : s.regs .real [BM, BN] "qk" = some qk2tile) (hmij : s.regs .real [BM] "m_ij" = some mij) :
    evalOp (Op.sub .real Broadcast.nil.consR.consSame (Op.ref .real [BM, BN] "qk")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BM] "m_ij"))).exp s
      = some (Tile.uop WithBot.realExp (Tile.bop NumericDType.real.sub Broadcast.nil.consR.consSame
          qk2tile (Tile.expandDim ⟨1, by simp⟩ mij))) := by
  rw [evalOp_exp, evalOp_sub]
  have hexp : @evalOp TileDType.real [BM, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BM] "m_ij")) s
      = some (Tile.expandDim ⟨1, by simp⟩ mij) := by erw [ctx_evalOp_expandDim_one_real, hmij]; rfl
  simp only [evalOp_ref, hqk, hexp, Option.bind_eq_bind, Option.bind_some]; rfl

/-- `m_i_new = tl.maximum(m_i, m_ij)` = `where(m_i > m_ij, m_i, m_ij)` eval. -/
theorem bloomMiNew_eval {BM : Nat} (s : BlockState) (mi mij : Tile .real [BM])
    (hmi : s.regs .real [BM] "m_i" = some mi) (hmij : s.regs .real [BM] "m_ij" = some mij) :
    evalOp ((Op.gt .real Broadcast.nil.consSame (Op.ref .real [BM] "m_i") (Op.ref .real [BM] "m_ij")).where
        (Op.ref .real [BM] "m_i") (Op.ref .real [BM] "m_ij")) s
      = some (Tile.select (Tile.cop ComparableDType.real.gt Broadcast.nil.consSame mi mij) mi mij) := by
  rw [evalOp_where]
  simp only [evalOp_gt, evalOp_ref, hmi, hmij, Option.bind_eq_bind, Option.bind_some]

/-- `alpha = tl.exp(m_i − m_i_new)` / `beta = tl.exp(m_ij − m_i_new)` natural-exp eval. -/
theorem bloomExpSub_eval {BM : Nat} (s : BlockState) (nm1 nm2 : RegName) (t1 t2 : Tile .real [BM])
    (h1 : s.regs .real [BM] nm1 = some t1) (h2 : s.regs .real [BM] nm2 = some t2) :
    evalOp (Op.sub .real Broadcast.nil.consSame (Op.ref .real [BM] nm1) (Op.ref .real [BM] nm2)).exp s
      = some (Tile.uop WithBot.realExp (Tile.bop NumericDType.real.sub Broadcast.nil.consSame t1 t2)) := by
  rw [evalOp_exp, evalOp_sub]; simp [h1, h2]

/-- `l_i_new = alpha·l_i + beta·l_ij` eval. -/
theorem bloomLiNew_eval {BM : Nat} (s : BlockState) (alpha li beta lij : Tile .real [BM])
    (ha : s.regs .real [BM] "alpha" = some alpha) (hli : s.regs .real [BM] "l_i" = some li)
    (hb : s.regs .real [BM] "beta" = some beta) (hlij : s.regs .real [BM] "l_ij" = some lij) :
    evalOp (Op.add .real Broadcast.nil.consSame
        (Op.mul .real Broadcast.nil.consSame (Op.ref .real [BM] "alpha") (Op.ref .real [BM] "l_i"))
        (Op.mul .real Broadcast.nil.consSame (Op.ref .real [BM] "beta") (Op.ref .real [BM] "l_ij"))) s
      = some (Tile.bop NumericDType.real.add Broadcast.nil.consSame
          (Tile.bop NumericDType.real.mul Broadcast.nil.consSame alpha li)
          (Tile.bop NumericDType.real.mul Broadcast.nil.consSame beta lij)) := by
  rw [evalOp_add]; simp [evalOp_mul, ha, hli, hb, hlij]

/-- `p_scale = beta / l_i_new` eval. -/
theorem bloomPscale_eval {BM : Nat} (s : BlockState) (beta lin : Tile .real [BM])
    (hb : s.regs .real [BM] "beta" = some beta) (hlin : s.regs .real [BM] "l_i_new" = some lin) :
    evalOp (Op.div .real Broadcast.nil.consSame (Op.ref .real [BM] "beta") (Op.ref .real [BM] "l_i_new")) s
      = some (Tile.bop NumericDType.real.div Broadcast.nil.consSame beta lin) := by
  rw [evalOp_div]; simp [hb, hlin]

/-- `p = p · p_scale[:, None]` eval. -/
theorem bloomP2_eval {BM BN : Nat} (s : BlockState) (ptile : Tile .real [BM, BN]) (pscale : Tile .real [BM])
    (hp : s.regs .real [BM, BN] "p" = some ptile) (hps : s.regs .real [BM] "p_scale" = some pscale) :
    evalOp (Op.mul .real Broadcast.nil.consR.consSame (Op.ref .real [BM, BN] "p")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BM] "p_scale"))) s
      = some (Tile.bop NumericDType.real.mul Broadcast.nil.consR.consSame ptile
          (Tile.expandDim ⟨1, by simp⟩ pscale)) := by
  have hexp : @evalOp TileDType.real [BM, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BM] "p_scale")) s
      = some (Tile.expandDim ⟨1, by simp⟩ pscale) := by erw [ctx_evalOp_expandDim_one_real, hps]; rfl
  rw [evalOp_mul]; simp only [evalOp_ref, hp, hexp, Option.bind_eq_bind, Option.bind_some]; rfl

/-- `acc_scale = (l_i / l_i_new) · alpha` eval. -/
theorem bloomAccScale1_eval {BM : Nat} (s : BlockState) (li lin alpha : Tile .real [BM])
    (hli : s.regs .real [BM] "l_i" = some li) (hlin : s.regs .real [BM] "l_i_new" = some lin)
    (ha : s.regs .real [BM] "alpha" = some alpha) :
    evalOp (Op.mul .real Broadcast.nil.consSame
        (Op.div .real Broadcast.nil.consSame (Op.ref .real [BM] "l_i") (Op.ref .real [BM] "l_i_new"))
        (Op.ref .real [BM] "alpha")) s
      = some (Tile.bop NumericDType.real.mul Broadcast.nil.consSame
          (Tile.bop NumericDType.real.div Broadcast.nil.consSame li lin) alpha) := by
  rw [evalOp_mul]; simp [evalOp_div, hli, hlin, ha]


theorem bloomAcc1_eval {BM D : Nat} (s : BlockState) (acctile : Tile .real [BM, D]) (ascale : Tile .real [BM])
    (hacc : s.regs .real [BM, D] "acc" = some acctile) (has : s.regs .real [BM] "acc_scale" = some ascale) :
    evalOp (Op.mul .real Broadcast.nil.consR.consSame (Op.ref .real [BM, D] "acc")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BM] "acc_scale"))) s
      = some (Tile.bop NumericDType.real.mul Broadcast.nil.consR.consSame acctile
          (Tile.expandDim ⟨1, by simp⟩ ascale)) := by
  have hexp : @evalOp TileDType.real [BM, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BM] "acc_scale")) s
      = some (Tile.expandDim ⟨1, by simp⟩ ascale) := by erw [ctx_evalOp_expandDim_one_real, has]; rfl
  rw [evalOp_mul]; simp only [evalOp_ref, hacc, hexp, Option.bind_eq_bind, Option.bind_some]; rfl

/-- `acc += tl.dot(p, v)` eval (`acc = acc + dot p v`). -/
theorem bloomAcc2_eval {BM BN D : Nat} (s : BlockState) (acctile : Tile .real [BM, D])
    (ptile : Tile .real [BM, BN]) (vtile : Tile .real [BN, D])
    (hacc : s.regs .real [BM, D] "acc" = some acctile)
    (hp : s.regs .real [BM, BN] "p" = some ptile) (hv : s.regs .real [BN, D] "v" = some vtile) :
    evalOp (Op.add .real Broadcast.nil.consSame.consSame (Op.ref .real [BM, D] "acc")
        (Op.dot (batch := []) (Op.ref .real [BM, BN] "p") (Op.ref .real [BN, D] "v"))) s
      = some (Tile.bop NumericDType.real.add Broadcast.nil.consSame.consSame acctile
          (Tile.dot [] ptile vtile)) := by
  have hdot0 : evalOp (Op.dot (batch := []) (Op.ref .real [BM, BN] "p") (Op.ref .real [BN, D] "v")) s
      = some (Tile.dot [] ptile vtile) := by rw [evalOp_dot]; simp [hp, hv]
  have hdot : @evalOp TileDType.real [BM, D]
      (Op.dot (batch := []) (Op.ref .real [BM, BN] "p") (Op.ref .real [BN, D] "v")) s
      = some (Tile.dot [] ptile vtile) := hdot0
  rw [evalOp_add]; simp only [evalOp_ref, hacc, hdot, Option.bind_eq_bind, Option.bind_some]; rfl


#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnBloom.NumericOps
