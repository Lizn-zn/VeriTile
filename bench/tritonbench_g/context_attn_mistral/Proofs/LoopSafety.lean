import bench.tritonbench_g.context_attn_mistral.Proofs.LoadSafety
import bench.tritonbench_g.context_attn_mistral.Proofs.PureSegments
namespace VeriTile.Bench.TritonBenchG.ContextAttnMistral.LoopSafety
open VeriTile.Triton Formula Program Frames
open VeriTile.Bench.TritonBenchG.ContextAttnFwd.PureSegments (assignments_frame)
noncomputable section
set_option maxHeartbeats 1200000
set_option maxRecDepth 5000
set_option linter.unusedSimpArgs false

theorem keys_safe (p : Layout) (r : Regions) (initial u : BlockState) (c : Nat)
    (bounds : RegionBounds) (hb : AddressBounds.Valid p r (launch initial) (inputs initial r) bounds)
    (hc : c < count p (launch initial) (inputs initial r))
    (hs : Invariant.Static p r initial u)
    (hn : u.regs .int [] "start_n" = some (Tile.scalar ((c*p.BN : Nat) : Int))) :
    Stmt.TraceSafeList bounds (LoopProgram.keys p r) u := by
  apply LoadSafety.key_safe p r (initial.pids 1/p.kvGroup)
    ((inputs initial r).start (initial.pids 0)) (c*p.BN)
    (Formula.sequence p (launch initial) (inputs initial r)) u bounds
    hs.key_ptrs hn hs.start hs.sequence hs.lanes
  intro j d hj
  have h := AddressBounds.inside_toNat (hb.key c hc j d (by simpa only [Nat.cast_add] using hj))
  have hshift : 0 ≤ (inputs initial r).start (initial.pids 0)+(c*p.BN : Nat) := by
    have := hb.advance
    exact add_nonneg this (Int.natCast_nonneg _)
  unfold KeysExecution.address
  rw [Meaning.pointer_address p.kRow p.kHead p.kDim (initial.pids 1/p.kvGroup)
    d.val j.val ((inputs initial r).start (initial.pids 0)) (c*p.BN) hshift]
  simpa only [AddressBounds.keyAddress,launch,Nat.cast_add,Nat.cast_mul,Int.natCast_div,add_assoc] using h

theorem values_safe (p : Layout) (r : Regions) (initial u : BlockState) (c : Nat)
    (bounds : RegionBounds) (hb : AddressBounds.Valid p r (launch initial) (inputs initial r) bounds)
    (hc : c < count p (launch initial) (inputs initial r))
    (hs : Invariant.Static p r initial u)
    (hn : u.regs .int [] "start_n" = some (Tile.scalar ((c*p.BN : Nat) : Int))) :
    Stmt.TraceSafeList bounds (LoopProgram.values p r) u := by
  apply LoadSafety.value_safe p r (initial.pids 1/p.kvGroup)
    ((inputs initial r).start (initial.pids 0)) (c*p.BN)
    (Formula.sequence p (launch initial) (inputs initial r)) u bounds
    hs.value_ptrs hn hs.start hs.sequence hs.lanes
  intro j d hj
  have h := AddressBounds.inside_toNat (hb.value c hc j d (by simpa only [Nat.cast_add] using hj))
  have hshift : 0 ≤ (inputs initial r).start (initial.pids 0)+(c*p.BN : Nat) := by
    have := hb.advance
    exact add_nonneg this (Int.natCast_nonneg _)
  unfold ValuesExecution.address
  rw [Meaning.pointer_address p.vRow p.vHead p.vDim (initial.pids 1/p.kvGroup)
    d.val j.val ((inputs initial r).start (initial.pids 0)) (c*p.BN) hshift]
  simpa only [AddressBounds.valueAddress,launch,Nat.cast_add,Nat.cast_mul,Int.natCast_div,add_assoc] using h

theorem body_safe (p : Layout) (r : Regions) (swapped : Bool) (scale : ℝ)
    (initial u : BlockState) (c : Nat) (bounds : RegionBounds)
    (hb : AddressBounds.Valid p r (launch initial) (inputs initial r) bounds)
    (hc : c < count p (launch initial) (inputs initial r))
    (hs : Invariant.Static p r initial u)
    (hn : u.regs .int [] "start_n" = some (Tile.scalar ((c*p.BN : Nat) : Int))) :
    Stmt.TraceSafeList bounds (LoopProgram.body p r swapped scale) u := by
  let sn : Int := c*p.BN
  let last := Formula.sequence p (launch initial) (inputs initial r)
  let start := (inputs initial r).start (initial.pids 0)
  let head := initial.pids 1/p.kvGroup
  let s1 := KeysExecution.prepared p r head start sn last u
  have he := KeysExecution.executes p r head start sn last u hs.key_ptrs hn hs.start hs.sequence hs.lanes
  have f1 := Frames.keys p r head start sn last u
  have hs1 : Invariant.Static p r initial s1 := by
    apply Frames.preserved p r initial u s1 _ f1 hs
    intro name hname
    simp [readNames] at hname
    rcases hname with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> decide
  have hn1 : s1.regs .int [] "start_n" = some (Tile.scalar sn) := by simp [s1,KeysExecution.prepared]
  rw [PureSegments.split]
  apply TraceProofs.append_safe
  · exact keys_safe p r initial u c bounds hb hc hs hn
  intro t ht
  rw [he] at ht
  cases ht
  change Stmt.TraceSafeList bounds (PureSegments.middle p swapped scale ++ _) s1
  apply TraceProofs.append_safe
  · exact PureSegments.middle_safe p swapped scale s1 bounds
  intro t ht
  have hf := assignments_frame PureSegments.names _ s1 t (PureSegments.middle_assigns p swapped scale) ht
  have hst : Invariant.Static p r initial t := by
    apply Frames.preserved p r initial s1 t _ hf hs1
    intro name hname
    simp [readNames] at hname
    rcases hname with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> decide
  have hnt : t.regs .int [] "start_n" = some (Tile.scalar sn) :=
    (hf.registers .int [] "start_n" (by decide)).trans hn1
  apply TraceProofs.append_safe
  · exact values_safe p r initial t c bounds hb hc hst hnt
  intro v _
  exact PureSegments.commit_safe p v bounds

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnMistral.LoopSafety
