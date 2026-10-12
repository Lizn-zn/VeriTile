import VeriTile.Triton.Semantics
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.ContextAttnLlama.NumericOps
open VeriTile.Triton
set_option linter.unusedSimpArgs false
noncomputable section
theorem ctxQk_op_eval {BM BN D : Nat} (s : BlockState) (qtile : Tile .real [BM, D]) (ktile : Tile .real [D, BN])
    (hq : s.regs .real [BM, D] "q" = some qtile) (hk : s.regs .real [D, BN] "k" = some ktile) :
    evalOp (Op.dot (batch := []) (Op.ref .real [BM, D] "q") (Op.ref .real [D, BN] "k")) s
      = some (Tile.dot [] qtile ktile) := by
  rw [evalOp_dot]; simp [hq, hk]

/-- Axis-1 `expandDim` over a `real` register (`m_ij[:, None]`/`alpha[:, None]`),
explicit `[M,1]` result shape so `erw` matches past the proof-irrelevant axis. -/
@[simp] theorem ctx_evalOp_expandDim_one_real {M : Nat} (name : RegName) (s : BlockState) :
    @evalOp .real [M, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [M] name)) s =
      (s.regs .real [M] name).bind (fun v =>
        some ({ data := fun i : TileIndex [M, 1] => v.data (i.1, PUnit.unit) } : Tile .real [M, 1])) := by
  unfold evalOp; simp [Tile.expandDim]; rfl

/-- `m_ij = tl.maximum(m_i, tl.max(qk, 1))` eval. -/
theorem ctxMij_op_eval {BM BN : Nat} (s : BlockState) (mtile rmaxT : Tile .real [BM])
    (qkfull : Tile .real [BM, BN])
    (hmi : s.regs .real [BM] "m_i" = some mtile)
    (hqk : s.regs .real [BM, BN] "qk" = some qkfull)
    (hrm : Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [BM, BN].length) qkfull = some rmaxT) :
    evalOp (Op.where
        (Op.gt .real Broadcast.nil.consSame (Op.ref .real [BM] "m_i")
          (Op.reduceMax ⟨1, by simp⟩ Bool.false (Op.ref .real [BM, BN] "qk")))
        (Op.ref .real [BM] "m_i")
        (Op.reduceMax ⟨1, by simp⟩ Bool.false (Op.ref .real [BM, BN] "qk"))) s
      = some (Tile.select (Tile.cop ComparableDType.real.gt Broadcast.nil.consSame mtile rmaxT) mtile rmaxT) := by
  have hrmaxN : evalOp (Op.reduceMax ⟨1, by simp⟩ Bool.false (Op.ref .real [BM, BN] "qk")) s = some rmaxT := by
    rw [evalOp_reduceMax]; simp only [evalOp_ref, hqk]; exact hrm
  have hrmax : @evalOp TileDType.real [BM]
      (Op.reduceMax ⟨1, by simp⟩ Bool.false (Op.ref .real [BM, BN] "qk")) s = some rmaxT := hrmaxN
  rw [evalOp_where]
  simp only [evalOp_gt, evalOp_ref, hmi, hrmax, Option.bind_eq_bind, Option.bind_some]

/-- `qk -= m_ij[:, None]` eval. -/
theorem ctxQk2_op_eval {BM BN : Nat} (s : BlockState) (qkfull : Tile .real [BM, BN]) (mij : Tile .real [BM])
    (hqk : s.regs .real [BM, BN] "qk" = some qkfull) (hmij : s.regs .real [BM] "m_ij" = some mij) :
    evalOp (Op.sub .real Broadcast.nil.consR.consSame (Op.ref .real [BM, BN] "qk")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BM] "m_ij"))) s
      = some (Tile.bop NumericDType.real.sub Broadcast.nil.consR.consSame qkfull
          (Tile.expandDim ⟨1, by simp⟩ mij)) := by
  have hexp : @evalOp TileDType.real [BM, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BM] "m_ij")) s
      = some (Tile.expandDim ⟨1, by simp⟩ mij) := by erw [ctx_evalOp_expandDim_one_real, hmij]; rfl
  rw [evalOp_sub]
  simp only [evalOp_ref, hqk, hexp, Option.bind_eq_bind, Option.bind_some]
  rfl

/-- `p = tl.math.exp2(qk)` eval. -/
theorem ctxP_op_eval {BM BN : Nat} (s : BlockState) (qk2tile : Tile .real [BM, BN])
    (hqk : s.regs .real [BM, BN] "qk" = some qk2tile) :
    evalOp (Op.exp2 (Op.ref .real [BM, BN] "qk")) s = some (Tile.uop WithBot.realExp2 qk2tile) := by
  rw [evalOp]; simp [hqk]

/-- `l_ij = tl.sum(p, 1)` eval. -/
theorem ctxLij_op_eval {BM BN : Nat} (s : BlockState) (ptile : Tile .real [BM, BN])
    (hp : s.regs .real [BM, BN] "p" = some ptile) :
    evalOp (Op.reduceSum (⟨1, by simp⟩ : Fin [BM, BN].length) Bool.false (Op.ref .real [BM, BN] "p")) s
      = some (Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [BM, BN].length) ptile) := by
  rw [evalOp_reduceSum]; simp only [evalOp_ref, hp, Option.bind_some]; rfl

/-- `alpha = tl.math.exp2(m_i − m_ij)` eval. -/
theorem ctxAlpha_op_eval {BM : Nat} (s : BlockState) (mi mij : Tile .real [BM])
    (hmi : s.regs .real [BM] "m_i" = some mi) (hmij : s.regs .real [BM] "m_ij" = some mij) :
    evalOp (Op.exp2 (Op.sub .real Broadcast.nil.consSame (Op.ref .real [BM] "m_i") (Op.ref .real [BM] "m_ij"))) s
      = some (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub Broadcast.nil.consSame mi mij)) := by
  rw [evalOp]; simp [evalOp_sub, hmi, hmij]

/-- `l_i = l_i · alpha + l_ij` eval. -/
theorem ctxLi_op_eval {BM : Nat} (s : BlockState) (li alpha lij : Tile .real [BM])
    (hli : s.regs .real [BM] "l_i" = some li) (ha : s.regs .real [BM] "alpha" = some alpha)
    (hlij : s.regs .real [BM] "l_ij" = some lij) :
    evalOp (Op.add .real Broadcast.nil.consSame
        (Op.mul .real Broadcast.nil.consSame (Op.ref .real [BM] "l_i") (Op.ref .real [BM] "alpha"))
        (Op.ref .real [BM] "l_ij")) s
      = some (Tile.bop NumericDType.real.add Broadcast.nil.consSame
          (Tile.bop NumericDType.real.mul Broadcast.nil.consSame li alpha) lij) := by
  rw [evalOp_add]; simp [evalOp_mul, hli, ha, hlij]

/-- `acc *= alpha[:, None]` eval. -/
theorem ctxAcc1_op_eval {BM D : Nat} (s : BlockState) (acctile : Tile .real [BM, D]) (alpha : Tile .real [BM])
    (hacc : s.regs .real [BM, D] "acc" = some acctile) (ha : s.regs .real [BM] "alpha" = some alpha) :
    evalOp (Op.mul .real Broadcast.nil.consR.consSame (Op.ref .real [BM, D] "acc")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BM] "alpha"))) s
      = some (Tile.bop NumericDType.real.mul Broadcast.nil.consR.consSame acctile
          (Tile.expandDim ⟨1, by simp⟩ alpha)) := by
  have hexp : @evalOp TileDType.real [BM, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BM] "alpha")) s
      = some (Tile.expandDim ⟨1, by simp⟩ alpha) := by erw [ctx_evalOp_expandDim_one_real, ha]; rfl
  rw [evalOp_mul]; simp only [evalOp_ref, hacc, hexp, Option.bind_eq_bind, Option.bind_some]; rfl

/-- Real evaluation of the source's single accumulator-dot operation. The FP
interpreter keeps that operation opaque; this lemma is only about real semantics. -/
theorem ctxAcc2_op_eval {BM BN D : Nat} (s : BlockState) (acc1tile : Tile .real [BM,D])
    (ptile : Tile .real [BM,BN]) (vtile : Tile .real [BN,D])
    (hp : s.regs .real [BM,BN] "p" = some ptile)
    (hv : s.regs .real [BN,D] "v" = some vtile)
    (hacc : s.regs .real [BM,D] "acc" = some acc1tile) :
    evalOp (Op.dotAcc .real .real .real (batch := [])
      (.ref .real [BM,BN] "p") (.ref .real [BN,D] "v") (.ref .real [BM,D] "acc")) s =
    some (Tile.bop NumericDType.real.add Broadcast.nil.consSame.consSame
      (Tile.dot [] ptile vtile) acc1tile) := by
  simp only [evalOp.eq_def, FloatDType.toTileDType, List.nil_append, hp, hv, hacc, Option.bind_eq_bind, Option.bind_some]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnLlama.NumericOps
