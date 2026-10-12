import bench.tritonbench_g.context_attn_llama.Proofs.Formula
namespace VeriTile.Bench.TritonBenchG.ContextAttnLlama.CellMath
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnBloom.Math
open scoped BigOperators
noncomputable section
theorem ctxg_reduceMaxDrop_data_row {M N : Nat} (hN : 0 < N) (qk : Tile .real [M, N])
    (rmaxT : Tile .real [M]) (hrm : Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [M,N].length) qk = some rmaxT)
    (r : Fin M) (g : Fin N → WithBot ℝ) (hqk : ∀ jL : Fin N, qk.data (r, jL, PUnit.unit) = g jL) :
    rmaxT.data (r, PUnit.unit) = Finset.univ.sup g := by
  unfold Tile.reduceMaxDrop at hrm
  rw [dif_pos (show 0 < TileShape.axisDim [M,N] (⟨1, by simp⟩ : Fin [M,N].length) from hN)] at hrm
  rw [← Option.some.inj hrm]
  simp only [Finset.sup'_eq_sup]
  exact Finset.sup_congr rfl (fun jL _ => hqk jL)

/-- `WithBot.realExp2` of a `some`-cell is `some (pow2 …)` (shape-generic). -/
theorem ctxg_exp2_some {M N : Nat} (h : Fin M → Fin N → ℝ) (x : Tile .real [M, N])
    (r : Fin M) (jL : Fin N) (hx : x.data (r, jL, PUnit.unit) = some (h r jL)) :
    (Tile.uop WithBot.realExp2 x).data (r, jL, PUnit.unit) = some (pow2 (h r jL)) := by
  show WithBot.realExp2 (x.data (r, jL, PUnit.unit)) = _
  rw [hx]; simp [WithBot.realExp2, pow2, mul_comm]

/-- A `WithBot ℝ` sum of `some`-valued cells is `some` of the real sum. -/
theorem ctxg_withBot_sum_some {N : Nat} (g : Fin N → ℝ) :
    @Finset.sum (Fin N) (WithBot ℝ) _ Finset.univ (fun k => (some (g k) : WithBot ℝ))
      = some (Finset.univ.sum g) :=
  (WithBot.coe_sum Finset.univ g).symm

/-- `realExp2` is total (never `⊥`): `realExp2 z = some ((realExp2 z).unbotD 0)`. -/
theorem ctxg_realExp2_eq_some_unbotD (z : WithBot ℝ) :
    WithBot.realExp2 z = some ((WithBot.realExp2 z).unbotD 0) := by
  cases z <;> rfl

/-- `m_ij = select(m_i > rmax) m_i rmax` collapses to `max` in `WithBot ℝ` (shape-generic). -/
theorem ctxg_mij_max {M : Nat} (m_i rmaxT : Tile .real [M]) (r : Fin M)
    (a b : WithBot ℝ) (hmi : m_i.data (r, PUnit.unit) = a) (hrm : rmaxT.data (r, PUnit.unit) = b) :
    (Tile.select (Tile.cop ComparableDType.real.gt Broadcast.nil.consSame m_i rmaxT) m_i rmaxT).data
        (r, PUnit.unit) = max a b := by
  rw [Tile.select_data, Tile.cop_data]
  simp only [Broadcast.leftIndex, Broadcast.rightIndex, ComparableDType.gt, hmi, hrm]
  by_cases h : a ≤ b
  · rw [if_neg (by simp [not_lt.mpr h]), max_eq_right h]
  · rw [if_pos (by simpa using not_le.mp h), max_eq_left (le_of_lt (not_le.mp h))]

/-- A `dot` row over all `K` keys when both factors are all-`some` (shape-generic
over `[M, K] · [K, N]`). -/
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


#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnLlama.CellMath
