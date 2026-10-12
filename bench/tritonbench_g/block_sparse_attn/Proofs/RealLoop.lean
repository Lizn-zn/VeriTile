import bench.tritonbench_g.block_sparse_attn.Proofs.RealPrologue
import bench.tritonbench_g.block_sparse_attn.Proofs.RealVisit

/-! Exact score/value history of the real source, starting from initial
memory and the Q tiles produced by its prologue. All physical query rows
are represented, including arbitrary values in masked rows. Independent
initial-array attention is compared only on stored rows in a later lemma.
The entire signed range executes by induction using the actual body step,
including empty and reversed intervals. Final output stores remain separate. -/
namespace VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealLoop
open VeriTile.Triton RealNormalizationValues
noncomputable section

def first (s : BlockState) (p : RealProgram.Parameters) : Int :=
  RealRows.bound s p.rows (RealSetup.head s p) p.layouts p.rowStride (s.pids 0) Bool.false

def last (s : BlockState) (p : RealProgram.Parameters) : Int :=
  RealRows.bound s p.rows (RealSetup.head s p) p.layouts p.rowStride (s.pids 0) Bool.true

def count (s : BlockState) (p : RealProgram.Parameters) : Nat := (last s p - first s p).toNat

def operands (s : BlockState) (BM BN BD : Nat) (p : RealProgram.Parameters) (visit : Nat) : RealVisit.Operands BM BN BD where
  layout := RealSetup.head s p % p.layouts
  visit := first s p + (visit : Int)
  keyPointers := RealSetup.keyPointers s BN BD p
  valuePointers := RealSetup.valuePointers s BN BD p
  rows := RealSetup.rows s BM
  query := RealQuery.loaded s (RealSetup.queryPointers s BM BD p) (RealSetup.rows s BM) p.context p.evenM Bool.false
  secondQuery := RealQuery.loaded s (RealSetup.queryPointers s BM BD p) (RealSetup.rows s BM) p.context p.evenM Bool.true

def score (s : BlockState) (BM BN BD : Nat) (p : RealProgram.Parameters) (visit : Nat) :
    Fin BM → Fin BN → WithBot ℝ :=
  let o := operands s BM BN BD p visit
  RealScore.masked (fun r j => RealScore.unscaled s o.keyPointers (RealVisit.start s o p)
    p.kRow p.context p.blocks p.evenN o.query o.secondQuery r j * p.scale) (RealVisit.start s o p) o.rows

def value (s : BlockState) (BM BN BD : Nat) (p : RealProgram.Parameters) (second : Bool) (visit : Nat) :
    Fin BN → Fin BD → ℝ :=
  let o := operands s BM BN BD p visit
  RealValueLoad.loaded s o.valuePointers (RealVisit.start s o p) p.vRow p.context p.evenN second

def carry (s : BlockState) (BM BN BD : Nat) (p : RealProgram.Parameters) (visit : Nat) : Inputs BM BN BD :=
  RealNormalizationMath.inputs (score s BM BN BD p) (value s BM BN BD p Bool.false)
    (value s BM BN BD p Bool.true) visit

structure CarryAt {BM BN BD : Nat} (x : Inputs BM BN BD) (blocks : Nat) (t : BlockState) : Prop where
  maximum : t.regs .real [BM] "m_i" = some (extendedRow x.maximum)
  normalizer : t.regs .real [BM] "l_i" = some (row x.normalizer)
  accumulator : t.regs .real [BM, BD] "acc" = some (matrix x.accumulator)
  secondAccumulator : 2 ≤ blocks → t.regs .real [BM, BD] "acc2" = some (matrix x.secondAccumulator)

theorem prologue_carry (s : BlockState) (BM BN BD : Nat) (p : RealProgram.Parameters) :
    CarryAt (carry s BM BN BD p 0) p.blocks (RealPrologue.result s BM BN BD p) := by
  have h := RealPrologue.outputs s BM BN BD p
  exact ⟨h.maximum, h.normalizer, h.accumulator, h.secondAccumulator⟩

theorem start_same_memory (s t : BlockState) (BM BN BD : Nat) (p : RealProgram.Parameters) (visit : Nat)
    (hm : t.mem = s.mem) :
    RealVisit.start t (operands s BM BN BD p visit) p = RealVisit.start s (operands s BM BN BD p visit) p := by
  simp only [RealVisit.start, RealColumn.column, BlockState.readMemValue, BlockState.readMemTyped, hm]

theorem inputs_eq_carry (s t : BlockState) (BM BN BD : Nat) (p : RealProgram.Parameters) (visit : Nat)
    (hm : t.mem = s.mem) (hu : t.undef = s.undef) :
    RealVisit.inputs t (operands s BM BN BD p visit) p (carry s BM BN BD p visit) = carry s BM BN BD p visit := by
  have hs := start_same_memory s t BM BN BD p visit hm
  simp only [RealVisit.inputs, hs, RealColumn.result, RealScore.unscaled, RealKey.loaded_setReg]
  have hl (second : Bool) : RealKey.loaded t (operands s BM BN BD p visit).keyPointers
      (RealVisit.start s (operands s BM BN BD p visit) p) p.kRow p.context p.evenN second =
      RealKey.loaded s (operands s BM BN BD p visit).keyPointers
      (RealVisit.start s (operands s BM BN BD p visit) p) p.kRow p.context p.evenN second := by
    funext d j
    simp only [RealKey.loaded, BlockState.readMem, hm, hu]
  simp only [hl]
  rfl

theorem value_same_memory (s t : BlockState) (BM BN BD : Nat) (p : RealProgram.Parameters)
    (visit : Nat) (second : Bool) (hm : t.mem = s.mem) (hu : t.undef = s.undef) :
    RealValueLoad.loaded t (operands s BM BN BD p visit).valuePointers
      (RealVisit.start t (operands s BM BN BD p visit) p) p.vRow p.context p.evenN second =
      value s BM BN BD p second visit := by
  rw [start_same_memory s t BM BN BD p visit hm]
  funext j d
  simp only [value, RealValueLoad.loaded, BlockState.readMem, hm, hu]

/-- The actual visit result contains the next exact normalization recurrence.
Only unchanged memory/undef is used to identify the source's load history. -/
theorem next_carry (s t : BlockState) (BM BN BD : Nat) (p : RealProgram.Parameters) (visit : Nat)
    (hm : t.mem = s.mem) (hu : t.undef = s.undef) :
    CarryAt (carry s BM BN BD p (visit+1)) p.blocks
      (RealVisit.result t (operands s BM BN BD p visit) p (carry s BM BN BD p visit)) := by
  have h := RealVisit.outputs t (operands s BM BN BD p visit) p (carry s BM BN BD p visit)
  rw [inputs_eq_carry s t BM BN BD p visit hm hu] at h
  have hv := value_same_memory s t BM BN BD p visit Bool.false hm hu
  have hv2 := value_same_memory s t BM BN BD p visit Bool.true hm hu
  constructor
  · exact h.maximum
  · have hn : (carry s BM BN BD p visit).nextNormalizer = (carry s BM BN BD p (visit+1)).normalizer :=
      funext (RealNormalizationMath.normalizer_next (score s BM BN BD p)
        (value s BM BN BD p Bool.false) (value s BM BN BD p Bool.true) visit)
    simpa only [hn] using h.normalizer
  · simpa only [hv] using h.accumulator
  · intro hb
    simpa only [hv2] using h.secondAccumulator hb

structure Fixed (s : BlockState) (BM BN BD : Nat) (p : RealProgram.Parameters) (t : BlockState) : Prop where
  layout : t.regs .nat [] "layout_h" = some (Tile.scalar (operands s BM BN BD p 0).layout)
  keys : t.regs .ptr [BD, BN] "k_ptrs" = some (operands s BM BN BD p 0).keyPointers
  values : t.regs .ptr [BN, BD] "v_ptrs" = some (operands s BM BN BD p 0).valuePointers
  rows : t.regs .nat [BM] "offs_m" = some (Tile.vec (operands s BM BN BD p 0).rows)
  lanes : t.regs .nat [BN] "offs_n" = some (Tile.vec Fin.val)
  query : t.regs .real [BM, BD] "q" = some (matrix (operands s BM BN BD p 0).query)
  secondQuery : 2 ≤ p.blocks → t.regs .real [BM, BD] "q2" = some (matrix (operands s BM BN BD p 0).secondQuery)
  context : t.regs .nat [] "q_seq_len" = some (Tile.scalar p.context)
  batch : t.regs .nat [] "off_b" = some (Tile.scalar (RealSetup.batch s p))
  head : t.regs .nat [] "off_h" = some (Tile.scalar (RealSetup.head s p))
  channels : t.regs .nat [BD] "offs_d" = some (Tile.vec Fin.val)
  frame : {t with regs := s.regs} = s

theorem prologue_fixed (s : BlockState) (BM BN BD : Nat) (p : RealProgram.Parameters) :
    Fixed s BM BN BD p (RealPrologue.result s BM BN BD p) := by
  have h := RealPrologue.outputs s BM BN BD p
  refine ⟨h.layout, h.keyPointers, h.valuePointers, h.queryRows, h.keyLanes,
    h.query, h.secondQuery, h.context, ?_, h.queryHead, ?_, RealPrologue.state_frame s BM BN BD p⟩
  all_goals
    rw [RealPrologue.setup_register _ _ _ _ _ _ _ _ (by simp [RealRows.writes])]
    by_cases hb : 2 ≤ p.blocks <;> simp [RealSetup.result, hb]

theorem fixed_set_index (s t : BlockState) (BM BN BD : Nat) (p : RealProgram.Parameters)
    (i : Int) (h : Fixed s BM BN BD p t) :
    Fixed s BM BN BD p (t.setReg "col_idx_idx" .int [] (Tile.scalar i)) := by
  rcases h with ⟨hl, hk, hv, hr, hn, hq, hq2, hctx, hbatch, hhead, hd, hf⟩
  constructor <;> simp_all [BlockState.setReg]

theorem carry_set_index {BM BN BD : Nat} (x : Inputs BM BN BD) (blocks : Nat) (t : BlockState)
    (i : Int) (h : CarryAt x blocks t) :
    CarryAt x blocks (t.setReg "col_idx_idx" .int [] (Tile.scalar i)) := by
  rcases h with ⟨hm, hl, ha, ha2⟩
  constructor <;> simp_all

theorem fixed_after_visit (s t : BlockState) (BM BN BD : Nat) (p : RealProgram.Parameters)
    (visit : Nat) (h : Fixed s BM BN BD p t) :
    Fixed s BM BN BD p
      (RealVisit.result t (operands s BM BN BD p visit) p (carry s BM BN BD p visit)) := by
  rcases h with ⟨hl, hk, hv, hr, hn, hq, hq2, hctx, hbatch, hhead, hd, hf⟩
  constructor
  all_goals first
    | (rw [RealVisit.register_frame _ _ _ _ _ _ _ (by simp [RealVisit.writes, RealNormalization.writes, RealAccumulation.writes])]; assumption)
    | (exact (congrArg (fun u => {u with regs := s.regs})
        (RealVisit.state_frame t (operands s BM BN BD p visit) p (carry s BM BN BD p visit))).trans hf)

/-- Derive the actual body step and its next invariant. The signed loop
counter is the source's first endpoint plus the natural visit count. -/
theorem body_step (swapped : Bool) (s t : BlockState) (BM BN BD : Nat) (p : RealProgram.Parameters)
    (visit : Nat) (hN : 0 < BN) (hf : Fixed s BM BN BD p t)
    (hc : CarryAt (carry s BM BN BD p visit) p.blocks t) :
    let u := t.setReg "col_idx_idx" .int [] (Tile.scalar (first s p + (visit : Int)))
    ∃ v, stepStmts (RealProgram.realPart (RealProgram.parts swapped BM BN BD p).body) u = some v ∧
      Fixed s BM BN BD p v ∧ CarryAt (carry s BM BN BD p (visit+1)) p.blocks v := by
  dsimp only
  let u := t.setReg "col_idx_idx" .int [] (Tile.scalar (first s p + (visit : Int)))
  have hfu := fixed_set_index s t BM BN BD p (first s p + (visit : Int)) hf
  have hcu := carry_set_index (carry s BM BN BD p visit) p.blocks t (first s p + (visit : Int)) hc
  refine ⟨RealVisit.result u (operands s BM BN BD p visit) p (carry s BM BN BD p visit), ?_,
    fixed_after_visit s u BM BN BD p visit hfu, ?_⟩
  · exact RealVisit.source_executes swapped hN u (operands s BM BN BD p visit) p (carry s BM BN BD p visit)
      hfu.layout (by simp [u, operands]) hfu.keys hfu.values hfu.rows hfu.lanes
      hfu.query hfu.secondQuery hcu.maximum hcu.normalizer hcu.accumulator hcu.secondAccumulator
  · have hm := congrArg BlockState.mem hfu.frame
    have hu := congrArg BlockState.undef hfu.frame
    exact next_carry s u BM BN BD p visit hm hu

theorem cursor_lt (s : BlockState) (p : RealProgram.Parameters) (visit : Nat) :
    first s p + (visit : Int) < last s p ↔ visit < count s p := by
  unfold count
  omega

/-- Execute every remaining signed visit, deriving each body's invariant
from its checked source execution. Empty and reversed intervals terminate
with the initial zero carry. -/
theorem loop_from (swapped : Bool) (s t : BlockState) (BM BN BD : Nat) (p : RealProgram.Parameters)
    (visit : Nat) (hi : visit ≤ count s p) (hN : 0 < BN)
    (hf : Fixed s BM BN BD p t) (hc : CarryAt (carry s BM BN BD p visit) p.blocks t) :
    ∃ v, stepForRangeIntAux "col_idx_idx" (first s p + (visit : Int)) (last s p) 1
        (RealProgram.realPart (RealProgram.parts swapped BM BN BD p).body) t = some v ∧
      Fixed s BM BN BD p v ∧ CarryAt (carry s BM BN BD p (count s p)) p.blocks v := by
  by_cases hv : visit < count s p
  · obtain ⟨u, hu, hfu, hcu⟩ := body_step swapped s t BM BN BD p visit hN hf hc
    obtain ⟨v, he, hfv, hcv⟩ := loop_from swapped s u BM BN BD p (visit+1) (by omega) hN hfu hcu
    refine ⟨v, ?_, hfv, hcv⟩
    rw [stepForRangeIntAux.step_lt (by decide) ((cursor_lt s p visit).mpr hv), hu, Option.bind_some]
    simpa only [Nat.cast_add, Nat.cast_one, Int.add_assoc] using he
  · have hi' : visit = count s p := by omega
    subst visit
    refine ⟨t, stepForRangeIntAux.step_ge (by decide) ?_, hf, hc⟩
    have hn := cursor_lt s p (count s p)
    omega
termination_by count s p - visit

theorem signed_loop_executes (swapped : Bool) (s : BlockState) (BM BN BD : Nat)
    (p : RealProgram.Parameters) (hN : 0 < BN) :
    ∃ v, stepStmt (.forRangeInt "col_idx_idx" (.ref .int [] "start_l") (.ref .int [] "end_l")
        (.constNat 1) (RealProgram.realPart (RealProgram.parts swapped BM BN BD p).body))
        (RealPrologue.result s BM BN BD p) = some v ∧
      Fixed s BM BN BD p v ∧ CarryAt (carry s BM BN BD p (count s p)) p.blocks v := by
  have hpre := RealPrologue.outputs s BM BN BD p
  obtain ⟨v, hv, hf, hc⟩ := loop_from swapped s (RealPrologue.result s BM BN BD p) BM BN BD p
    0 (Nat.zero_le _) hN (prologue_fixed s BM BN BD p) (prologue_carry s BM BN BD p)
  refine ⟨v, ?_, hf, hc⟩
  rw [stepForRangeIntAux.forRangeInt_unfold]
  simp only [evalOp_ref, hpre.first, hpre.last, evalOp_constNat, Tile.scalar_data,
    Option.bind_some]
  simpa only [Int.ofNat_zero, Int.add_zero, first, last] using hv

/-- The actual prefix followed by the whole signed range executes from the
initial state. Only the final output stores remain outside this composition. -/
theorem prologue_loop_executes (swapped : Bool) (s : BlockState) (BM BN BD : Nat)
    (p : RealProgram.Parameters) (hN : 0 < BN) :
    ∃ v, stepStmts (RealProgram.realPart (RealProgram.parts swapped BM BN BD p).before ++
        [.forRangeInt "col_idx_idx" (.ref .int [] "start_l") (.ref .int [] "end_l")
          (.constNat 1) (RealProgram.realPart (RealProgram.parts swapped BM BN BD p).body)]) s = some v ∧
      Fixed s BM BN BD p v ∧ CarryAt (carry s BM BN BD p (count s p)) p.blocks v := by
  obtain ⟨v, hv, hf, hc⟩ := signed_loop_executes swapped s BM BN BD p hN
  refine ⟨v, ?_, hf, hc⟩
  rw [stepStmts.append_some (RealPrologue.source_executes swapped s BM BN BD p),
    stepStmts.cons_some hv, stepStmts.nil]


#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealLoop
