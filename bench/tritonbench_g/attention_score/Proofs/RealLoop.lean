import bench.tritonbench_g.attention_score.Proofs.RealIteration
import Mathlib.Algebra.Order.Floor.Div

/-! Dynamic query range execution. The loop invariant connects accumulated
columns to initial Q/K/M memory, while preserving the static K tile, scale,
launch coordinates and memory. The stop may lie inside the last query block;
the unmasked M load therefore needs padded storage in the eventual IO bounds. -/
namespace VeriTile.Bench.TritonBenchG.AttentionScore.RealLoop
open VeriTile.Triton RealFrame
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false
noncomputable section

/-- Internal parameters of the query loop, after grouped-head addresses have
been evaluated by the prelude. All flags and signed window parameters remain. -/
structure Config where
  (Q K M : RegionName)
  (B D N NK : Nat)
  (baseQ baseK baseM sqm sqk skn skk startN : Nat)
  (offset size : Int)
  scale : ℝ
  (window complement evenM evenN candidate : Bool)

def Config.inputs (c : Config) (initial : BlockState) : RealFormula.Inputs c.D :=
  RealLoads.inputs initial c.Q c.K c.M c.D c.baseQ c.baseK c.baseM c.sqm c.sqk c.skn c.skk

def Config.body (c : Config) : List Stmt :=
  RealBody.body c.B c.D c.N c.offset c.size c.window c.complement c.evenM c.evenN c.candidate

def Config.column (c : Config) (initial : BlockState) (blocks : Nat) (k : Fin c.B) : ℝ :=
  RealFormula.column (c.inputs initial) c.scale c.N c.NK c.B blocks
    c.window c.complement c.evenM c.evenN c.offset c.size (c.startN*c.B+k.val)

/-- Only loop-carried registers and static loaded operands appear here.
This is an internal invariant, not a condition imposed on public inputs. -/
structure Invariant (c : Config) (initial : BlockState) (blocks : Nat) (s : BlockState) : Prop where
  mem : s.mem = initial.mem
  startN : s.regs .nat [] "start_n" = some (Tile.scalar c.startN)
  query : s.regs .blockPtr [c.B,c.D] "Q_block_ptr" = some
    (RealLoads.queryPointer c.Q c.B c.D c.N c.baseQ c.sqm c.sqk (blocks*c.B))
  normalizer : s.regs .ptr [c.B] "m_ptrs" = some
    (RealLoads.normalizerPointers c.M c.B c.baseM (blocks*c.B))
  key : s.regs .real [c.D,c.B] "k" = some
    (RealTiles.key (c.inputs initial) c.B c.NK c.startN c.evenN)
  scale : s.regs .real [] "qk_scale" = some (Tile.scalar (some (c.scale*1.4426950408889634)))
  output : s.regs .real [c.B] "o" = some (Tile.vec (fun k => some (c.column initial blocks k)))

/-- One complete body iteration, including the range's induction-variable
assignment, maintains the closed-form partial-column invariant. -/
theorem iteration (c : Config) (initial s : BlockState) (blocks : Nat) (hB : 0 < c.B)
    (hi : Invariant c initial blocks s) :
    ∃ t, stepStmts c.body (s.setReg "start_m" .nat [] (Tile.scalar (blocks*c.B))) = some t ∧
      Invariant c initial (blocks+1) t ∧ Frame RealIteration.writes s t := by
  let entered := s.setReg "start_m" .nat [] (Tile.scalar (blocks*c.B))
  have fe : Frame RealIteration.writes s entered :=
    Frame.setReg _ _ _ _ _ _ (by simp [RealIteration.writes])
  obtain ⟨t, ht, ho, hq, hm, ft⟩ := RealIteration.correct initial entered c.Q c.K c.M
    c.B c.D c.N c.NK c.baseQ c.baseK c.baseM c.sqm c.sqk c.skn c.skk (blocks*c.B) c.startN hB
    c.offset c.size c.scale c.window c.complement c.evenM c.evenN c.candidate
    (c.column initial blocks) hi.mem (by simp [entered])
    (by simpa [entered] using hi.startN) (by simpa [entered] using hi.query)
    (by simpa [entered] using hi.normalizer) (by simpa [entered] using hi.key)
    (by simpa [entered] using hi.scale) (by simpa [entered] using hi.output)
  have frame := fe.trans ft
  refine ⟨t, ht, ?_, frame⟩
  refine ⟨frame.mem.trans hi.mem, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact (frame.regs .nat [] "start_n" (by simp [RealIteration.writes])).trans hi.startN
  · simpa [Nat.add_mul] using hq
  · simpa [Nat.add_mul] using hm
  · exact (frame.regs .real [c.D,c.B] "k" (by simp [RealIteration.writes])).trans hi.key
  · exact (frame.regs .real [] "qk_scale" (by simp [RealIteration.writes])).trans hi.scale
  · simpa only [Config.column, Config.inputs, RealFormula.column_succ] using ho

/-- Execute the remaining range. The two endpoint inequalities characterize
the actual trip count; stop need not be a multiple of the block size. -/
theorem range (c : Config) (initial : BlockState) (hB : 0 < c.B)
    (remaining done stop : Nat) (s : BlockState)
    (hi : Invariant c initial done s)
    (hupper : stop ≤ (done+remaining)*c.B)
    (hlower : done+remaining = 0 ∨ (done+remaining-1)*c.B < stop) :
    ∃ t, stepForRangeAux "start_m" (done*c.B) stop c.B c.body s = some t ∧
      Invariant c initial (done+remaining) t ∧ Frame RealIteration.writes s t := by
  induction remaining generalizing done s with
  | zero =>
      refine ⟨s, stepForRangeAux.step_ge (Nat.ne_of_gt hB) (by simpa using hupper), ?_, Frame.refl _ _⟩
      simpa using hi
  | succ remaining ih =>
      have hlast : (done+remaining)*c.B < stop := by
        rcases hlower with hz | hl
        · omega
        · simpa only [Nat.add_succ, Nat.add_sub_cancel] using hl
      have hlt : done*c.B < stop :=
        lt_of_le_of_lt (Nat.mul_le_mul_right c.B (Nat.le_add_right done remaining)) hlast
      obtain ⟨u, hu, iu, fu⟩ := iteration c initial s done hB hi
      obtain ⟨t, ht, it, ft⟩ := ih (done+1) u iu
        (by simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using hupper)
        (by simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using hlower)
      refine ⟨t, ?_, ?_, fu.trans ft⟩
      · rw [stepForRangeAux.step_lt (Nat.ne_of_gt hB) hlt, hu, Option.bind_some]
        simpa only [Nat.add_mul, Nat.one_mul] using ht
      · simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using it

/-- The dynamic range visits one full block for each nonempty interval before
stop. This ceiling formula includes stop=0 and a partially filled final block. -/
def tripCount (stop B : Nat) : Nat := (stop+B-1)/B

theorem tripCount_bounds (stop B : Nat) (hB : 0 < B) :
    stop ≤ tripCount stop B * B ∧
      (tripCount stop B = 0 ∨ (tripCount stop B-1)*B < stop) := by
  have hc : tripCount stop B = stop ⌈/⌉ B := rfl
  constructor
  · simpa only [hc, Nat.mul_comm] using (le_smul_ceilDiv hB : stop ≤ B * (stop ⌈/⌉ B))
  · by_cases hz : tripCount stop B = 0
    · exact Or.inl hz
    · right
      by_contra hn
      have he := (ceilDiv_le_iff_le_mul hB).2 (show stop ≤ B * (tripCount stop B-1) by
        simpa only [Nat.mul_comm] using Nat.le_of_not_gt hn)
      rw [← hc] at he
      omega

/-- The complete query range computes the independent column sum from its
initialized invariant, for either source version and every stop value. -/
theorem correct (c : Config) (initial s : BlockState) (stop : Nat) (hB : 0 < c.B)
    (hi : Invariant c initial 0 s) :
    ∃ t, stepForRangeAux "start_m" 0 stop c.B c.body s = some t ∧
      Invariant c initial (tripCount stop c.B) t ∧ Frame RealIteration.writes s t := by
  obtain ⟨hu,hl⟩ := tripCount_bounds stop c.B hB
  simpa using range c initial hB (tripCount stop c.B) 0 stop s hi (by simpa using hu) (by simpa using hl)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttentionScore.RealLoop
