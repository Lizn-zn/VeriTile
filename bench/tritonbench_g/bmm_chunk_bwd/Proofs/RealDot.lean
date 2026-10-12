import bench.tritonbench_g.bmm_chunk_bwd.Proofs.RealLoads

/-! The real value of a masked dot block, derived from the current pointer
and memory tiles. The partial contraction tail remains in the sum. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkBwd.RealDot
open VeriTile.Triton RealLoads
noncomputable section
set_option linter.unusedSimpArgs false

theorem finite_dot (M CS N : Nat) (x : Tile .real [M, CS]) (y : Tile .real [CS, N])
    (i : Fin M) (j : Fin N) (fx fy : Fin CS → ℝ)
    (hx : ∀ e, x.data (i,e,PUnit.unit) = some (fx e))
    (hy : ∀ e, y.data (e,j,PUnit.unit) = some (fy e)) :
    (Tile.dot [] x y).data (i,j,PUnit.unit) = some (∑ e, fx e * fy e) := by
  rw [Tile.dot_nil_data]
  rw [show (@Finset.sum (Fin CS) (WithBot ℝ) _ Finset.univ
      (fun e => Option.map₂ (· * ·) (x.data (i,e,PUnit.unit)) (y.data (e,j,PUnit.unit)))) =
      @Finset.sum (Fin CS) (WithBot ℝ) _ Finset.univ (fun e => some (fx e * fy e))
    from Finset.sum_congr rfl (fun e _ => by rw [hx e, hy e]; rfl)]
  show (∑ e, ((fx e * fy e : ℝ) : WithBot ℝ)) = _
  rw [← WithBot.coe_sum]
  rfl

def value {BM BN BCS : Nat} (s : BlockState)
    (dp : Tile .ptr [BM, BCS]) (ap : Tile .ptr [BCS, BN])
    (rows : Fin BM → Nat) (columns : Fin BN → Nat) (chunk K length c : Nat)
    (i : Fin BM) (j : Fin BN) : ℝ :=
  if rows i < chunk ∧ columns j < K then
    ∑ k : Fin BCS, if k.val < length - c * BCS then
      s.readMem (dp.data (i,k,PUnit.unit)).1 (dp.data (i,k,PUnit.unit)).2 *
      s.readMem (ap.data (k,j,PUnit.unit)).1 (ap.data (k,j,PUnit.unit)).2 else 0
  else 0

theorem masked_dot (BM BN BCS : Nat) (s : BlockState)
    (dp : Tile .ptr [BM, BCS]) (ap : Tile .ptr [BCS, BN])
    (rows : Fin BM → Nat) (columns : Fin BN → Nat) (chunk K length c : Nat) :
    Tile.dot [] (doutValues s dp rows chunk length c) (aValues s ap columns K length c) =
      Tile.mat (fun i j => some (value s dp ap rows columns chunk K length c i j)) := by
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  rw [finite_dot BM BCS BN _ _ i j _ _ (by intro e; rfl) (by intro e; rfl)]
  apply congrArg some
  by_cases hr : rows i < chunk <;> by_cases hc : columns j < K
  · simp only [value, hr, hc, true_and, ↓reduceIte]
    apply Finset.sum_congr rfl
    intro k _
    by_cases hk : k.val < length - c * BCS <;>
      simp [doutValues, aValues, Tile.mat, hr, hc, hk]
  all_goals simp [value, doutValues, aValues, Tile.mat, hr, hc]

/-- The source's accumulation op consumes the memory-derived tiles. -/
theorem accumulation_eval (BM BN BCS : Nat) (s : BlockState)
    (dp : Tile .ptr [BM, BCS]) (ap : Tile .ptr [BCS, BN])
    (rows : Fin BM → Nat) (columns : Fin BN → Nat) (chunk K length c : Nat)
    (carry : Fin BM → Fin BN → ℝ)
    (hz : s.regs .real [BM, BN] "acc" = some (Tile.mat fun i j => some (carry i j)))
    (hd : s.regs .real [BM, BCS] "dout" = some (doutValues s dp rows chunk length c))
    (ha : s.regs .real [BCS, BN] "a" = some (aValues s ap columns K length c)) :
    evalOp (.add .real (.consSame (.consSame .nil)) (.ref .real [BM, BN] "acc")
      (.dot (batch := []) (.ref .real [BM, BCS] "dout") (.ref .real [BCS, BN] "a"))) s =
      some (Tile.mat fun i j => some (carry i j + value s dp ap rows columns chunk K length c i j)) := by
  have hdot : evalOp (.dot (batch := []) (.ref .real [BM, BCS] "dout")
      (.ref .real [BCS, BN] "a")) s =
      some (Tile.mat fun i j => some (value s dp ap rows columns chunk K length c i j)) := by
    rw [evalOp_dot]
    simp [hd, ha, masked_dot]
  rw [evalOp_add]
  simp only [evalOp_ref, hz, bind, Option.bind_some]
  erw [hdot]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmChunkBwd.RealDot
