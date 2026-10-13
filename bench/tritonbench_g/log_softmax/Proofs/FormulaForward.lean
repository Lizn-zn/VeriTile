import bench.tritonbench_g.log_softmax.Proofs.Real
import bench.tritonbench_g.log_softmax.Proofs.Math

namespace VeriTile.Bench.TritonBenchG.LogSoftmax.Formula
open VeriTile.Triton
open scoped BigOperators
set_option maxHeartbeats 1600000
set_option linter.unusedSimpArgs false

private theorem sum_prefix (N C : Nat) (hNC : N ≤ C) (f : Nat → ℝ) :
    (∑ j : Fin C, if j.val<N then f j.val else 0) =
      ∑ j : Fin N, f j.val := by
  rw [Fin.sum_univ_eq_sum_range (fun j ↦ if j<N then f j else 0) C,
    Fin.sum_univ_eq_sum_range f N]
  rw [show C=N+(C-N) by omega, Finset.sum_range_add]
  have hhead : (∑ j ∈ Finset.range N, if j<N then f j else 0) =
      ∑ j ∈ Finset.range N, f j := by
    apply Finset.sum_congr rfl
    intro j hj
    rw [if_pos (Finset.mem_range.mp hj)]
  have htail : (∑ j ∈ Finset.range (C-N), if N+j<N then f (N+j) else 0) = 0 := by
    apply Finset.sum_eq_zero
    intro j _
    rw [if_neg (by omega)]
  rw [hhead, htail, add_zero]

private theorem padded_max_finite (N C : Nat) (hN : 0<N) (_hNC : N≤C)
    (f : Nat → ℝ) (hC : 0<C) :
    ∃ m : ℝ, @Finset.sup' (WithBot ℝ) (Fin C) _ Finset.univ
      ⟨⟨0,hC⟩,Finset.mem_univ _⟩
      (fun j ↦ if j.val<N then (some (f j.val) : WithBot ℝ) else (⊥ : WithBot ℝ)) = some m := by
  have hle := Finset.le_sup' (α := WithBot ℝ)
    (fun j : Fin C ↦ if j.val<N then (some (f j.val) : WithBot ℝ) else (⊥ : WithBot ℝ))
    (Finset.mem_univ (⟨0,hC⟩ : Fin C))
  simp only [hN, if_pos] at hle
  have hne : @Finset.sup' (WithBot ℝ) (Fin C) _ Finset.univ
      ⟨⟨0,hC⟩,Finset.mem_univ _⟩
      (fun j ↦ if j.val<N then (some (f j.val) : WithBot ℝ) else (⊥ : WithBot ℝ)) ≠ ⊥ := by
    intro heq
    rw [heq] at hle
    exact WithBot.coe_ne_bot (le_bot_iff.mp hle)
  obtain ⟨m,hm⟩ := WithBot.ne_bot_iff_exists.mp hne
  exact ⟨m,hm.symm⟩

/-- A complete active row has the independent logarithmic softmax formula;
masked columns contribute zero after exponentiation. -/
theorem forward_row (N R C : Nat) (hNC : N≤C) (inp : Tile .real [R,C])
    (r : Fin R) (i : Fin C) (hi : i.val<N) (x : Nat → ℝ)
    (hrow : ∀ j : Fin C, inp.data (r,j,PUnit.unit) =
      if j.val<N then some (x j.val) else (⊥ : WithBot ℝ)) :
    logSoftmaxOf R C inp (r,i,PUnit.unit) =
      Math.forward (fun j : Fin N ↦ x j.val) ⟨i.val,hi⟩ := by
  have hC : 0<C := Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt
  have hN : 0<N := Nat.lt_of_le_of_lt (Nat.zero_le _) hi
  obtain ⟨m,hm⟩ := padded_max_finite N C hN hNC x hC
  have hexp (j : Fin C) :
      WithBot.realExp (WithBot.realSub
        (if j.val<N then some (x j.val) else (⊥ : WithBot ℝ)) (some m)) =
      (some (if j.val<N then Real.exp (x j.val-m) else 0) : WithBot ℝ) := by
    split_ifs <;> rfl
  have hsum : (∑ j : Fin C, WithBot.realExp (WithBot.realSub
      (if j.val<N then some (x j.val) else (⊥ : WithBot ℝ)) (some m))) =
      (some (∑ j : Fin N, Real.exp (x j.val-m)) : WithBot ℝ) := by
    simp only [hexp, WithBot.sum_someTerm_eq_some]
    exact congrArg some (sum_prefix N C hNC (fun j ↦ Real.exp (x j-m)))
  simp only [logSoftmaxOf, Tile.reduceMax, Tile.reduceMaxKeep,
    TileShape.axisDim, List.getElem_cons_succ, List.getElem_cons_zero,
    hC, ↓reduceDIte]
  change (WithBot.realLog (WithBot.realDiv
    (WithBot.realExp (WithBot.realSub (inp.data (r,i,PUnit.unit))
      ((Finset.univ : Finset (Fin C)).sup' ⟨⟨0,hC⟩,Finset.mem_univ _⟩ (fun j ↦ inp.data (r,j,PUnit.unit)))))
    (∑ j : Fin C, WithBot.realExp (WithBot.realSub (inp.data (r,j,PUnit.unit))
      ((Finset.univ : Finset (Fin C)).sup' ⟨⟨0,hC⟩,Finset.mem_univ _⟩ (fun k ↦ inp.data (r,k,PUnit.unit))))))).unbotD 0 = _
  have hm' : (Finset.univ : Finset (Fin C)).sup' ⟨⟨0,hC⟩,Finset.mem_univ _⟩
      (fun j ↦ inp.data (r,j,PUnit.unit)) = some m := by
    calc
      _ = @Finset.sup' (WithBot ℝ) (Fin C) _ Finset.univ
          ⟨⟨0,hC⟩,Finset.mem_univ _⟩
          (fun j ↦ if j.val<N then (some (x j.val) : WithBot ℝ) else ⊥) := by
        congr 1
        funext j
        exact hrow j
      _ = some m := hm
  have hsum' : (∑ j : Fin C,
      WithBot.realExp (WithBot.realSub (inp.data (r,j,PUnit.unit)) (some m))) =
      (some (∑ j : Fin N, Real.exp (x j.val-m)) : WithBot ℝ) := by
    calc
      _ = (∑ j : Fin C, WithBot.realExp (WithBot.realSub
          (if j.val<N then some (x j.val) else (⊥ : WithBot ℝ)) (some m))) := by
        apply Finset.sum_congr rfl
        intro j _
        rw [hrow]
        rfl
      _ = _ := hsum
  rw [hm', hsum', hrow, if_pos hi]
  change Real.log (Real.exp (x i.val-m) / ∑ j : Fin N, Real.exp (x j.val-m)) = _
  exact Math.forward_shift (fun j : Fin N ↦ x j.val) m ⟨i.val,hi⟩

/-- Pure IO inputs retain their lane indexing while the result uses only the
active row's ordinary finite real values. -/
theorem forward_pure_tile (p M N R C : Nat) (hNC : N≤C)
    (xs : Fin (R*C) → ℝ) (r : Fin R) (i : Fin C)
    (hr : p*R+r.val<M) (hi : i.val<N) :
    logSoftmaxOf R C (logSoftmaxPureTile p M N R C xs) (r,i,PUnit.unit) =
      Math.forward (fun j : Fin N ↦
        xs (laneOf R C (r,⟨j.val,lt_of_lt_of_le j.isLt hNC⟩,PUnit.unit)))
        ⟨i.val,hi⟩ := by
  let x : Nat → ℝ := fun j ↦ if hj : j<C then
    xs (laneOf R C (r,⟨j,hj⟩,PUnit.unit)) else 0
  have h := forward_row N R C hNC (logSoftmaxPureTile p M N R C xs)
    r i hi x (by
      intro j
      simp [logSoftmaxPureTile, active, hr, x, j.isLt]; rfl)
  simpa [x, show ∀ j : Fin N, j.val<C from fun j ↦ lt_of_lt_of_le j.isLt hNC] using h

/-- The state-facing readback formula for one complete active row. -/
theorem forward_memory (s : BlockState) (X : RegionName)
    (p q M N K R C : Nat) (hNC : N≤C) (r : Fin R) (i : Fin C)
    (hr : p*R+r.val<M) (hi : i.val<N) :
    logSoftmaxSpec s X p q M N K R C (r,i,PUnit.unit) =
      Math.forward (fun j : Fin N ↦
        s.readMem X ((p*R+r.val)*N*K+j.val*K+q)) ⟨i.val,hi⟩ := by
  exact forward_row N R C hNC (logSoftmaxInputTile s X p q M N K R C)
    r i hi (fun j ↦ s.readMem X ((p*R+r.val)*N*K+j*K+q))
    (by intro j; simp [logSoftmaxInputTile, active, hr, outOffset]; rfl)

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.LogSoftmax.Formula
