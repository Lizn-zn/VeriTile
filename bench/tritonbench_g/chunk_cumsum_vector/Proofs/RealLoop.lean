import bench.tritonbench_g.chunk_cumsum_vector.Proofs.RealStep
import VeriTile.Triton.Memory.Frame.Basic

/-! The whole real time loop. Its invariant derives the column carry from
initial inputs and unspecified values, preserves earlier outputs and frames
complete memory cells outside the active output window. -/
namespace VeriTile.Bench.TritonBenchG.ChunkCumsumVector.RealLoop
open VeriTile.Triton RealProgram RealStep
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

def carry (p : Layout) (l : Launch) (a padding : Nat → ℝ) (c : Nat) (j : Fin p.BS) : ℝ :=
  RealMath.carry (fun t => value p l a padding t j) p.BT c

def expected (p : Layout) (l : Launch) (a : Nat → ℝ) (t : Nat) (j : Fin p.BS) : ℝ :=
  RealMath.prefixValue (column p l a j) p.T t

/-- Only active cells must have distinct addresses, across the whole time span. -/
def NoCollision (p : Layout) (l : Launch) : Prop :=
  ∀ (t u : Nat) (i j : Fin p.BS), RealStep.active p l t i.val → RealStep.active p l u j.val →
    address p l t i.val = address p l u j.val → t = u ∧ i = j

def footprint (p : Layout) (l : Launch) (o : RegionName) : WriteFootprint :=
  fun cell => cell.1 = o ∧ ∃ (t : Nat) (j : Fin p.BS), RealStep.active p l t j.val ∧
    cell.2 = address p l t j.val

theorem next_carry (p : Layout) (l : Launch) (a padding : Nat → ℝ) (c : Nat) :
    nextCarry p l a padding c (carry p l a padding c) = carry p l a padding (c + 1) := by
  funext j
  exact (RealMath.carry_succ (fun t => value p l a padding t j) p.BT c).symm

/-- Only the observed feature must be active. Other columns retain arbitrary
padding, but a triangular dot does not mix different feature columns. -/
theorem prefix_at_chunk (p : Layout) (l : Launch) (a padding : Nat → ℝ) (c : Nat)
    (i : Fin p.BT) (j : Fin p.BS) (hi : c * p.BT + i.val < p.T)
    (hj : l.feature * p.BS + j.val < p.S) :
    carry p l a padding c j + chunkPrefix p l a padding c i j =
      expected p l a (c * p.BT + i.val) j := by
  rw [add_comm]
  refine (RealMath.prefix_from_chunk (fun t => value p l a padding t j)
    p.T p.BT c i hi).trans ?_
  simp only [expected, RealMath.prefixValue]
  apply Finset.sum_congr rfl
  intro k hk
  have ht := (Finset.mem_filter.mp hk).1
  have ht' := Finset.mem_range.mp ht
  simp [RealStep.value, RealStep.active, ht', hj, RealStep.column]

structure Holds (initial : BlockState) (p : Layout) (l : Launch) (s o : RegionName)
    (a : Nat → ℝ) (c : Nat) (u : BlockState) : Prop where
  undef : u.undef = initial.undef
  count_le : c ≤ numChunks p
  feature : u.regs .nat [] "i_s" = some (Tile.scalar l.feature)
  head : u.regs .nat [] "i_bh" = some (Tile.scalar l.head)
  triangular : u.regs .real [p.BT, p.BT] "m_s" = some (lowerTriTile p.BT)
  accumulated : u.regs .real [p.BS] "b_z" =
    some (Tile.vec fun j => some (carry p l a (initial.undef s) c j))
  input : ∀ off, u.readMem s off = a off
  output : ∀ (t : Nat) (j : Fin p.BS), RealStep.active p l t j.val → t < c * p.BT →
    u.readMem o (address p l t j.val) = expected p l a t j
  frame : BlockState.WriteWithin (footprint p l o) initial u

/-- The scatter changes only active cells of the current chunk. -/
theorem scatter_frame (p : Layout) (l : Launch) (o : RegionName) (a padding : Nat → ℝ)
    (c : Nat) (acc : Fin p.BS → ℝ) (u : BlockState) :
    BlockState.WriteWithin (footprint p l o) u (scatter p l o a padding c acc u) := by
  apply BlockState.foldl_writeMem_prop_writeWithin
  intro idx _ ha
  exact ⟨rfl, c * p.BT + idx.1.val, idx.2.1, ha, rfl⟩

theorem iteration (initial : BlockState) (p : Layout) (l : Launch) (s o : RegionName)
    (a : Nat → ℝ) (hSO : s ≠ o) (_hBT : 0 < p.BT) (hinj : NoCollision p l)
    (c : Nat) (u : BlockState) (hc : c < numChunks p)
    (h : Holds initial p l s o a c u) :
    ∃ v, stepStmts (loopBody p s o Bool.false) (u.setReg "i_t" .nat [] (Tile.scalar c)) = some v ∧
      Holds initial p l s o a (c + 1) v := by
  let entry := u.setReg "i_t" .nat [] (Tile.scalar c)
  let pre := beforeStore p l s o a (initial.undef s) c (carry p l a (initial.undef s) c) entry
  let v := result p l s o a (initial.undef s) c (carry p l a (initial.undef s) c) entry
  have he := RealStep.executes p l s o a (initial.undef s) c
    (carry p l a (initial.undef s) c) entry (by intro off; simp [entry, h.undef])
    (by simp [entry, h.feature]) (by simp [entry, h.head]) (by simp [entry])
    (by intro off; simpa [entry] using h.input off)
    (by simp [entry, h.triangular]) (by simp [entry, h.accumulated])
  refine ⟨v, he, ?_, Nat.succ_le_of_lt hc, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · have hu := stepStmts_undef (loopBody p s o Bool.false) entry v (by
      simp [loopBody, pointerOp, carryOp, StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]
      all_goals simp [Op.FlattenOk.eq_def]) he
    exact hu.trans (by simpa [entry] using h.undef)
  · simp [v, result, scatter, beforeStore, entry, h.feature]
  · simp [v, result, scatter, beforeStore, entry, h.head]
  · simp [v, result, scatter, beforeStore, entry, h.triangular]
  · simp [v, result, next_carry p l a (initial.undef s) c]
  · intro off
    change (scatter p l o a (initial.undef s) c (carry p l a (initial.undef s) c) pre).readMem s off = a off
    rw [scatter, BlockState.scatter_prop_masked_preserves_other_region _ _ _ _ _ hSO]
    simpa [pre, beforeStore, entry] using h.input off
  · intro t j ha ht
    change (scatter p l o a (initial.undef s) c (carry p l a (initial.undef s) c) pre).readMem o (address p l t j.val) = _
    by_cases hprior : t < c * p.BT
    · rw [scatter, BlockState.scatter_prop_masked_preserves_other_offset]
      · simpa [pre, beforeStore, entry] using h.output t j ha hprior
      · intro idx hidx heq
        have := (hinj _ _ idx.2.1 j hidx ha heq).1
        omega
    · have hi : t - c * p.BT < p.BT := by
        rw [Nat.add_mul, Nat.one_mul] at ht
        omega
      let i : Fin p.BT := ⟨t - c * p.BT, hi⟩
      have hti : c * p.BT + i.val = t := by dsimp [i]; omega
      have hactive : RealStep.active p l (c * p.BT + i.val) j.val := by simpa [hti] using ha
      have hread := BlockState.scatter_readback_prop_masked_nd_of_true
        (region := o) pre
        (fun idx : TileIndex [p.BT, p.BS] => address p l (c * p.BT + idx.1.val) idx.2.1.val)
        (fun idx => carry p l a (initial.undef s) c idx.2.1 + chunkPrefix p l a (initial.undef s) c idx.1 idx.2.1)
        (fun idx => RealStep.active p l (c * p.BT + idx.1.val) idx.2.1.val)
        (i,j,PUnit.unit) hactive (by
          rintro ⟨k,e,⟨⟩⟩ hk heq
          obtain ⟨hrow, hcol⟩ := hinj _ _ e j hk hactive heq
          dsimp only at hrow hcol
          have hi' : k = i := Fin.ext (by omega)
          subst k; subst e; rfl)
      change (scatter p l o a (initial.undef s) c (carry p l a (initial.undef s) c) pre).readMem o
        (address p l (c * p.BT + i.val) j.val) = _ at hread
      dsimp only at hread
      rw [hti] at hread
      rw [hread, prefix_at_chunk p l a (initial.undef s) c i j hactive.1 hactive.2, hti]
  · have hs := scatter_frame p l o a (initial.undef s) c (carry p l a (initial.undef s) c) pre
    intro r off hn
    exact (h.frame r off hn).trans (hs r off hn)

def statement (p : Layout) (s o : RegionName) : Stmt :=
  .forRangeDyn "i_t" (.constNat 0) (stopOp p) (.constNat 1) (loopBody p s o Bool.false)

theorem stop_eval (p : Layout) (u : BlockState) :
    evalOp (stopOp p) u = some (Tile.scalar (numChunks p)) := by
  simp only [stopOp, evalOp_div, evalOp_sub, evalOp_add, evalOp_constNat,
    Option.bind_some, Option.bind_eq_bind]
  rfl

theorem executes (initial : BlockState) (p : Layout) (l : Launch) (s o : RegionName)
    (a : Nat → ℝ) (hSO : s ≠ o) (hBT : 0 < p.BT) (hinj : NoCollision p l)
    (u : BlockState) (h : Holds initial p l s o a 0 u) :
    ∃ v, stepStmt (statement p s o) u = some v ∧ Holds initial p l s o a (numChunks p) v := by
  obtain ⟨final, v, he, hf, hv⟩ := forRangeDyn_inv
    (P := Holds initial p l s o a)
    (evalOp_constNat 0 u) (stop_eval p u) (evalOp_constNat 1 u) (by decide) h
    (fun c u hc h => iteration initial p l s o a hSO hBT hinj c u hc h)
  have hfinal : final = numChunks p := Nat.le_antisymm hv.count_le hf
  subst final
  exact ⟨v, he, hv⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkCumsumVector.RealLoop
