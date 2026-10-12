import bench.tritonbench_g.context_attn_llama.Proofs.LoadSafety
import bench.tritonbench_g.context_attn_llama.Proofs.PureSegments
namespace VeriTile.Bench.TritonBenchG.ContextAttnLlama.LoopSafety
open VeriTile.Triton Formula Program Frames
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
  apply LoadSafety.key_safe p r (c*p.BN) ((inputs initial r).request (Formula.batch p (launch initial)))
    (Formula.limit p (launch initial) (inputs initial r)) ((initial.pids 1%p.heads)/p.kvGroup) u bounds
    hn hs.request hs.ending hs.kv_head hs.lanes hs.channels
  · intro j hj
    exact AddressBounds.inside_toNat (hb.pages c hc j (by simpa only [Nat.cast_add] using hj))
  · intro d j hj
    have h := AddressBounds.inside_toNat (hb.key c hc j d (by simpa only [Nat.cast_add] using hj))
    have hp := Meaning.pages_match p r u initial c hs.memory j
    simp only [Nat.cast_mul] at hp
    simp only [KeysExecution.address]
    rw [hp]
    simpa only [LoopState.pages,AddressBounds.keyAddress,Formula.batch,Formula.head,launch,
      Int.natCast_div] using h

theorem values_safe (p : Layout) (r : Regions) (initial u : BlockState) (c : Nat)
    (bounds : RegionBounds) (hb : AddressBounds.Valid p r (launch initial) (inputs initial r) bounds)
    (hc : c < count p (launch initial) (inputs initial r))
    (hs : Invariant.Static p r initial u)
    (hn : u.regs .int [] "start_n" = some (Tile.scalar ((c*p.BN : Nat) : Int)))
    (hp : u.regs .int [p.BN] "kv_loc" = some (Tile.vec (LoopState.pages p r initial c))) :
    Stmt.TraceSafeList bounds (LoopProgram.values p r) u := by
  apply LoadSafety.value_safe p r ((initial.pids 1%p.heads)/p.kvGroup) (LoopState.pages p r initial c) (c*p.BN)
    (Formula.limit p (launch initial) (inputs initial r)) u bounds
    hp hs.kv_head hn hs.ending hs.lanes hs.channels
  intro j d hj
  have h := AddressBounds.inside_toNat (hb.value c hc j d (by simpa only [Nat.cast_add] using hj))
  simpa only [ValuesExecution.address,LoopState.pages,AddressBounds.valueAddress,Formula.batch,Formula.head,launch,Nat.cast_add,Nat.cast_mul,Int.natCast_div] using h

theorem body_safe (p : Layout) (r : Regions) (swapped : Bool) (scale : ℝ)
    (initial u : BlockState) (c : Nat) (bounds : RegionBounds)
    (hb : AddressBounds.Valid p r (launch initial) (inputs initial r) bounds)
    (hc : c < count p (launch initial) (inputs initial r))
    (hs : Invariant.Static p r initial u)
    (hn : u.regs .int [] "start_n" = some (Tile.scalar ((c*p.BN : Nat) : Int))) :
    Stmt.TraceSafeList bounds (LoopProgram.body p r swapped scale) u := by
  let sn : Int := c*p.BN
  let last := Formula.limit p (launch initial) (inputs initial r)
  let request := (inputs initial r).request (initial.pids 1/p.heads)
  let head := (initial.pids 1%p.heads)/p.kvGroup
  let s1 := KeysExecution.prepared p r head request sn last u
  have he := KeysExecution.executes p r head request sn last u hs.request hs.kv_head hn hs.ending hs.lanes hs.channels
  have f1 := Frames.keys p r head request sn last u
  have hs1 : Invariant.Static p r initial s1 := by
    apply Frames.preserved p r initial u s1 _ f1 hs
    intro name hname
    simp [readNames] at hname
    rcases hname with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> decide
  have hp1 : s1.regs .int [p.BN] "kv_loc" = some (Tile.vec (LoopState.pages p r initial c)) := by
    simp only [s1,KeysExecution.prepared,BlockState.setReg_ne_name,ne_eq,String.reduceEq,not_false_eq_true,BlockState.setReg_same]
    congr 1
    apply Tile.ext
    rintro ⟨j,⟨⟩⟩
    exact Meaning.pages_match p r u initial c hs.memory j
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
  have hf := PureSegments.assignments_frame PureSegments.names _ s1 t (PureSegments.middle_assigns p swapped scale) ht
  have hst : Invariant.Static p r initial t := by
    apply Frames.preserved p r initial s1 t _ hf hs1
    intro name hname
    simp [readNames] at hname
    rcases hname with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> decide
  have hnt : t.regs .int [] "start_n" = some (Tile.scalar sn) :=
    (hf.registers .int [] "start_n" (by decide)).trans hn1
  exact values_safe p r initial t c bounds hb hc hst hnt
    ((hf.registers .int [p.BN] "kv_loc" (by decide)).trans hp1)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnLlama.LoopSafety
