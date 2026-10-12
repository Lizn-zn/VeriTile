import bench.tritonbench_g.chunk_delta_fwd.Proofs.RealStep
import bench.tritonbench_g.chunk_delta_fwd.Proofs.RealMath

/-! All inner blocks of one outer chunk. H stays frozen while the accumulator
collects each block's contribution. The invariant preserves corrected values
from earlier blocks and complete tagged cells outside the output footprint.
No divisibility condition is needed here: the source visits ceil(BT/BC) blocks. -/
namespace VeriTile.Bench.TritonBenchG.ChunkDeltaFwd.RealInnerLoop
open VeriTile.Triton RealProgram
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

def count (p : Layout) : Nat := (p.BT + p.BC - 1) / p.BC

def contribution (p : Layout) (l : Launch) (a : Inputs) (outer inner : Nat)
    (frozen : StateValue p) (e : Fin p.BK) (j : Fin p.BV) : ℝ :=
  ∑ c : Fin p.BC, keyValue p l a (timeBase p outer inner + c.val) e.val *
    correctedValue p l a frozen (timeBase p outer inner + c.val) j

def accumulated (p : Layout) (l : Launch) (a : Inputs) (outer inner : Nat)
    (frozen seed : StateValue p) : StateValue p := fun e j =>
  seed e j + ∑ b ∈ Finset.range inner, contribution p l a outer b frozen e j

/-- Only active output cells need distinct addresses, across the time span. -/
def NoCollision (p : Layout) (l : Launch) : Prop :=
  ∀ (t u : Nat) (i j : Fin p.BV), activeValue p l t i.val → activeValue p l u j.val →
    valueAddress p l t i.val = valueAddress p l u j.val → t = u ∧ i = j

def footprint (p : Layout) (l : Launch) (out : RegionName) (outer : Nat) : WriteFootprint :=
  fun cell => cell.1 = out ∧ ∃ (t : Nat) (j : Fin p.BV), activeValue p l t j.val ∧
    outer * p.BT ≤ t ∧ t < outer * p.BT + count p * p.BC ∧
    cell.2 = valueAddress p l t j.val

theorem next_accumulated (p : Layout) (l : Launch) (a : Inputs) (outer inner : Nat)
    (frozen seed : StateValue p) :
    nextAccum p l a outer inner frozen (accumulated p l a outer inner frozen seed) =
      accumulated p l a outer (inner + 1) frozen seed := by
  funext e j
  simp only [nextAccum, accumulated, Finset.sum_range_succ, contribution, add_assoc]

theorem scatter_frame (p : Layout) (l : Launch) (out : RegionName) (a : Inputs)
    (outer inner : Nat) (hc : inner < count p) (frozen : StateValue p) (u : BlockState) :
    BlockState.WriteWithin (footprint p l out outer) u (scatter p l out a outer inner frozen u) := by
  apply BlockState.foldl_writeMem_prop_writeWithin
  intro idx _ ha
  refine ⟨rfl, timeBase p outer inner + idx.1.val, idx.2.1, ha, ?_, ?_, rfl⟩
  · dsimp [timeBase]; omega
  · have hi := Nat.mul_le_mul_right p.BC (Nat.succ_le_of_lt hc)
    have hl := idx.1.isLt
    simp only [Nat.succ_eq_add_one, Nat.add_mul, Nat.one_mul] at hi
    dsimp [timeBase]; omega

structure Holds (initial : BlockState) (p : Layout) (l : Launch) (r : Regions)
    (a : Inputs) (outer : Nat) (frozen seed : StateValue p) (inner : Nat) (u : BlockState) : Prop where
  count_le : inner ≤ count p
  key : u.regs .nat [] "i_k" = some (Tile.scalar l.key)
  value : u.regs .nat [] "i_v" = some (Tile.scalar l.value)
  head : u.regs .nat [] "i_bh" = some (Tile.scalar l.head)
  time : u.regs .nat [] "i_t" = some (Tile.scalar outer)
  state : u.regs .real [p.BK,p.BV] "b_h" = some (stateTile p frozen)
  acc : u.regs .real [p.BK,p.BV] "b_h_cumsum" =
    some (stateTile p (accumulated p l a outer inner frozen seed))
  inputKey : ∀ off, u.readMem r.key off = a.key off
  inputDelta : ∀ off, u.readMem r.delta off = a.delta off
  inputValue : ∀ off, u.readMem r.value off = a.value off
  output : ∀ (t : Nat) (j : Fin p.BV), activeValue p l t j.val →
    outer * p.BT ≤ t → t < outer * p.BT + inner * p.BC →
      u.readMem r.corrected (valueAddress p l t j.val) = correctedValue p l a frozen t j
  frame : BlockState.WriteWithin (footprint p l r.corrected outer) initial u
  pids : u.pids = initial.pids
  undef : u.undef = initial.undef

theorem iteration (initial : BlockState) (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (outer : Nat) (frozen seed : StateValue p)
    (hu : initial.undef = fun _ _ => 0)
    (hk : r.key ≠ r.corrected) (hd : r.delta ≠ r.corrected) (hv : r.value ≠ r.corrected)
    (hinj : NoCollision p l) (inner : Nat) (u : BlockState) (hc : inner < count p)
    (h : Holds initial p l r a outer frozen seed inner u) :
    ∃ v, stepStmts (innerBody p r) (u.setReg "i_c" .nat [] (Tile.scalar inner)) = some v ∧
      Holds initial p l r a outer frozen seed (inner + 1) v := by
  let entry := u.setReg "i_c" .nat [] (Tile.scalar inner)
  let acc := accumulated p l a outer inner frozen seed
  let pre := beforeStore p l r a outer inner frozen entry
  let v := result p l r a outer inner frozen acc entry
  have he := RealStep.executes p l r a outer inner frozen acc entry (h.undef.trans hu)
    (by simp [entry, h.key]) (by simp [entry, h.value]) (by simp [entry, h.head])
    (by simp [entry, h.time]) (by simp [entry])
    (by intro off; simpa [entry] using h.inputKey off)
    (by intro off; simpa [entry] using h.inputDelta off)
    (by intro off; simpa [entry] using h.inputValue off)
    (by simp [entry, h.state]) (by simpa [entry, acc] using h.acc)
  refine ⟨v,he,Nat.succ_le_of_lt hc,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_⟩
  · simp [v, result, scatter, beforeStore, entry, h.key]
  · simp [v, result, scatter, beforeStore, entry, h.value]
  · simp [v, result, scatter, beforeStore, entry, h.head]
  · simp [v, result, scatter, beforeStore, entry, h.time]
  · simp [v, result, scatter, beforeStore, entry, h.state]
  · simp [v, result, acc, next_accumulated]
  · intro off
    change (scatter p l r.corrected a outer inner frozen pre).readMem r.key off = a.key off
    rw [scatter, BlockState.scatter_prop_masked_preserves_other_region _ _ _ _ _ hk]
    simpa [pre, beforeStore, entry] using h.inputKey off
  · intro off
    change (scatter p l r.corrected a outer inner frozen pre).readMem r.delta off = a.delta off
    rw [scatter, BlockState.scatter_prop_masked_preserves_other_region _ _ _ _ _ hd]
    simpa [pre, beforeStore, entry] using h.inputDelta off
  · intro off
    change (scatter p l r.corrected a outer inner frozen pre).readMem r.value off = a.value off
    rw [scatter, BlockState.scatter_prop_masked_preserves_other_region _ _ _ _ _ hv]
    simpa [pre, beforeStore, entry] using h.inputValue off
  · intro t j ha hbase ht
    change (scatter p l r.corrected a outer inner frozen pre).readMem r.corrected
      (valueAddress p l t j.val) = _
    by_cases hprior : t < outer * p.BT + inner * p.BC
    · rw [scatter, BlockState.scatter_prop_masked_preserves_other_offset]
      · simpa [pre, beforeStore, entry] using h.output t j ha hbase hprior
      · intro idx hi heq
        have := (hinj _ _ idx.2.1 j hi ha heq).1
        dsimp [timeBase] at this
        omega
    · have hi : t - timeBase p outer inner < p.BC := by
        simp only [Nat.add_mul, Nat.one_mul, timeBase] at ht ⊢
        omega
      let i : Fin p.BC := ⟨t - timeBase p outer inner,hi⟩
      have hti : timeBase p outer inner + i.val = t := by dsimp [i, timeBase]; omega
      have ha' : activeValue p l (timeBase p outer inner + i.val) j.val := by simpa [hti] using ha
      have hread := BlockState.scatter_readback_prop_masked_nd_of_true
        (region := r.corrected) pre
        (fun idx : TileIndex [p.BC,p.BV] => valueAddress p l (timeBase p outer inner + idx.1.val) idx.2.1.val)
        (fun idx => correctedValue p l a frozen (timeBase p outer inner + idx.1.val) idx.2.1)
        (fun idx => activeValue p l (timeBase p outer inner + idx.1.val) idx.2.1.val)
        (i,j,PUnit.unit) ha' (by
          rintro ⟨c,e,⟨⟩⟩ he heq
          obtain ⟨hrow,hcol⟩ := hinj _ _ e j he ha' heq
          dsimp only at hrow hcol
          have hi' : c = i := Fin.ext (by omega)
          subst c; subst e; rfl)
      change (scatter p l r.corrected a outer inner frozen pre).readMem r.corrected
        (valueAddress p l (timeBase p outer inner + i.val) j.val) =
          correctedValue p l a frozen (timeBase p outer inner + i.val) j at hread
      simpa only [hti] using hread
  · have hs := scatter_frame p l r.corrected a outer inner hc frozen pre
    intro rg off hn
    exact (h.frame rg off hn).trans (hs rg off hn)
  · simpa [v, result, scatter, BlockState.foldl_writeMem_prop_masked_pids, beforeStore, entry] using h.pids
  · simpa only [v, result, scatter, cdfWrites_undef, beforeStore, entry, BlockState.setReg] using h.undef

def statement (p : Layout) (r : Regions) : Stmt :=
  .forRangeDyn "i_c" (.constNat 0) (cdfStopOp p.BT p.BC) (.constNat 1) (innerBody p r)

theorem stop_eval (p : Layout) (u : BlockState) :
    evalOp (cdfStopOp p.BT p.BC) u = some (Tile.scalar (count p)) := by
  simp only [cdfStopOp, evalOp_div, evalOp_sub, evalOp_add, evalOp_constNat,
    Option.bind_some, Option.bind_eq_bind]
  rfl

theorem executes (initial : BlockState) (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (outer : Nat) (frozen seed : StateValue p)
    (hu : initial.undef = fun _ _ => 0)
    (hk : r.key ≠ r.corrected) (hd : r.delta ≠ r.corrected) (hv : r.value ≠ r.corrected)
    (hinj : NoCollision p l) (u : BlockState)
    (h : Holds initial p l r a outer frozen seed 0 u) :
    ∃ v, stepStmt (statement p r) u = some v ∧
      Holds initial p l r a outer frozen seed (count p) v := by
  obtain ⟨final,v,he,hf,hv⟩ := forRangeDyn_inv
    (P := Holds initial p l r a outer frozen seed)
    (evalOp_constNat 0 u) (stop_eval p u) (evalOp_constNat 1 u) (by decide) h
    (fun c u hc h => iteration initial p l r a outer frozen seed hu hk hd hv hinj c u hc h)
  have hfinal : final = count p := Nat.le_antisymm hv.count_le hf
  subst final
  exact ⟨v,he,hv⟩

/-- Build the loop invariant from ordinary input and register facts. -/
theorem entry (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (outer : Nat)
    (frozen seed : StateValue p) (u : BlockState)
    (hik : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hiv : u.regs .nat [] "i_v" = some (Tile.scalar l.value))
    (hih : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hit : u.regs .nat [] "i_t" = some (Tile.scalar outer))
    (hkey : ∀ off, u.readMem r.key off = a.key off)
    (hdelta : ∀ off, u.readMem r.delta off = a.delta off)
    (hvalue : ∀ off, u.readMem r.value off = a.value off)
    (hstate : u.regs .real [p.BK,p.BV] "b_h" = some (stateTile p frozen))
    (hacc : u.regs .real [p.BK,p.BV] "b_h_cumsum" = some (stateTile p seed)) :
    Holds u p l r a outer frozen seed 0 u := by
  refine ⟨Nat.zero_le _,hik,hiv,hih,hit,hstate,?_,hkey,hdelta,hvalue,?_,?_,rfl,rfl⟩
  · have hz : accumulated p l a outer 0 frozen seed = seed := by
      funext e j
      simp [accumulated]
    rw [hz]
    exact hacc
  · intro t j _ hb ht
    simp only [Nat.zero_mul, Nat.add_zero] at ht
    omega
  · intro _ _ _; rfl

/-- A real sum over all inner blocks equals the flat sum over the rows actually
visited. With nondivisible BC, that span may be wider than BT. -/
theorem accumulated_flat (p : Layout) (l : Launch) (a : Inputs) (outer inner : Nat)
    (frozen seed : StateValue p) (e : Fin p.BK) (j : Fin p.BV) :
    accumulated p l a outer inner frozen seed e j = seed e j +
      ∑ t : Fin (inner * p.BC), keyValue p l a (outer * p.BT + t.val) e.val *
        correctedValue p l a frozen (outer * p.BT + t.val) j := by
  unfold accumulated contribution
  rw [← Fin.sum_univ_eq_sum_range]
  congr 1
  simpa only [timeBase, Nat.add_assoc] using
    RealMath.block_sum (fun t => keyValue p l a (outer * p.BT + t) e.val *
      correctedValue p l a frozen (outer * p.BT + t) j) inner p.BC

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkDeltaFwd.RealInnerLoop
