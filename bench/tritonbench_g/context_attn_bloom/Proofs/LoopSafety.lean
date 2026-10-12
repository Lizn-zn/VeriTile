import bench.tritonbench_g.context_attn_bloom.Proofs.KeysSafety
import bench.tritonbench_g.context_attn_bloom.Proofs.ValuesSafety
import bench.tritonbench_g.context_attn_bloom.Proofs.PureSegments
namespace VeriTile.Bench.TritonBenchG.ContextAttnBloom.LoopSafety
open VeriTile.Triton Math Program RealSetup RegisterFrame
noncomputable section
set_option maxHeartbeats 800000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

theorem page_matches (p : Layout) (r : Regions) (initial u : BlockState) (c : Nat) (j : Fin p.BN)
    (hm : u.mem = initial.mem)
    (hj : ((c*p.BN : Nat) : Int)+j.val < Math.limit p (launch initial) (inputs initial r)) :
    LoopKeysExecution.page p r (c*p.BN) ((inputs initial r).request (initial.pids 0))
      (Math.limit p (launch initial) (inputs initial r)) u j.val =
      Math.token p (launch initial) (inputs initial r) (c*p.BN+j.val) := by
  simp only [Nat.cast_mul] at hj
  unfold LoopKeysExecution.page
  rw [if_pos hj,KeysMeaning.read_int_of_memory u initial hm]
  unfold Math.token inputs launch
  congr 1
  congr 1
  push_cast
  ring

theorem keys_safe (p : Layout) (r : Regions) (initial u : BlockState) (c : Nat)
    (bounds : RegionBounds) (hb : AddressBounds.Valid p r (launch initial) (inputs initial r) bounds)
    (hc : c < count p (launch initial) (inputs initial r))
    (hs : Invariant.Static p r initial u)
    (hn : u.regs .int [] "start_n" = some (Tile.scalar ((c*p.BN : Nat) : Int))) :
    Stmt.TraceSafeList bounds (LoopKeysProgram.keys p r) u := by
  apply KeysSafety.keys_safe p r (c*p.BN) ((inputs initial r).request (initial.pids 0))
    (Math.limit p (launch initial) (inputs initial r)) (initial.pids 1/p.kvGroup) u bounds
    hn hs.request hs.ending hs.kv_head hs.lanes hs.channels
  · intro j hj
    have h := AddressBounds.inside_toNat (hb.tokens c hc j (by simpa only [Nat.cast_add] using hj))
    simpa only [AddressBounds.tokenAddress,Nat.cast_add,Nat.cast_mul] using h
  · intro d j hj hd
    rw [page_matches p r initial u c j hs.memory hj]
    exact AddressBounds.inside_toNat (hb.key c hc j d (by simpa only [Nat.cast_add] using hj) hd)

theorem values_safe (p : Layout) (r : Regions) (initial source u : BlockState) (c : Nat)
    (bounds : RegionBounds) (hb : AddressBounds.Valid p r (launch initial) (inputs initial r) bounds)
    (hc : c < count p (launch initial) (inputs initial r)) (hm : source.mem = initial.mem)
    (hs : Invariant.Static p r initial u)
    (hn : u.regs .int [] "start_n" = some (Tile.scalar ((c*p.BN : Nat) : Int)))
    (hp : u.regs .int [p.BN] "kv_loc" = some (Tile.vec (fun j =>
      LoopKeysExecution.page p r (c*p.BN) ((inputs initial r).request (initial.pids 0))
        (Math.limit p (launch initial) (inputs initial r)) source j.val))) :
    Stmt.TraceSafeList bounds (ValuesProgram.values p r) u := by
  apply ValuesSafety.safe p r (c*p.BN) (Math.limit p (launch initial) (inputs initial r))
    (initial.pids 1/p.kvGroup) _ u bounds hn hs.ending hs.kv_head hs.lanes hs.channels hp
  intro j d hj hd
  rw [page_matches p r initial source c j hm hj]
  exact AddressBounds.inside_toNat (hb.value c hc j d (by simpa only [Nat.cast_add] using hj) hd)

theorem body_safe (p : Layout) (r : Regions) (swapped : Bool) (scale : ℝ)
    (initial u : BlockState) (c : Nat) (bounds : RegionBounds)
    (hb : AddressBounds.Valid p r (launch initial) (inputs initial r) bounds)
    (hc : c < count p (launch initial) (inputs initial r))
    (hs : Invariant.Static p r initial u)
    (hn : u.regs .int [] "start_n" = some (Tile.scalar ((c*p.BN : Nat) : Int))) :
    Stmt.TraceSafeList bounds (LoopProgram.body p r swapped scale) u := by
  let sn : Int := c*p.BN
  let req := (inputs initial r).request (initial.pids 0)
  let last := Math.limit p (launch initial) (inputs initial r)
  let head := initial.pids 1/p.kvGroup
  let s1 := LoopKeysExecution.prepared p r sn req last head u
  have he := LoopKeysExecution.executes p r sn req last head u hn hs.request hs.ending hs.kv_head hs.lanes hs.channels
  have hs1 : Invariant.Static p r initial s1 := KeysMeaning.static_preserved p r u initial sn req last head hs
  have hn1 : s1.regs .int [] "start_n" = some (Tile.scalar sn) := by simp [s1,LoopKeysExecution.prepared]
  have hp1 : s1.regs .int [p.BN] "kv_loc" = some (Tile.vec (fun j => LoopKeysExecution.page p r sn req last u j.val)) := by
    simp [s1,LoopKeysExecution.prepared]
  rw [PureSegments.split]
  apply TraceProofs.append_safe
  · exact keys_safe p r initial u c bounds hb hc hs hn
  intro t ht
  rw [he] at ht
  cases ht
  change Stmt.TraceSafeList bounds (PureSegments.middle p r swapped scale ++ _) s1
  apply TraceProofs.append_safe
  · exact PureSegments.middle_safe p r swapped scale s1 bounds
  intro t ht
  have hf := PureSegments.assignments_frame PureSegments.names _ s1 t (PureSegments.middle_assigns p r swapped scale) ht
  have hst : Invariant.Static p r initial t := by
    apply StaticFrame.preserved p r initial s1 t _ hf hs1
    intro name hname
    simp [StaticFrame.readNames] at hname
    rcases hname with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
    all_goals decide
  have hnt : t.regs .int [] "start_n" = some (Tile.scalar sn) :=
    (hf.registers .int [] "start_n" (by decide)).trans hn1
  have hpt : t.regs .int [p.BN] "kv_loc" = some (Tile.vec (fun j => LoopKeysExecution.page p r sn req last u j.val)) :=
    (hf.registers .int [p.BN] "kv_loc" (by decide)).trans hp1
  apply TraceProofs.append_safe
  · exact values_safe p r initial u t c bounds hb hc hs.memory hst hnt hpt
  intro v _
  exact PureSegments.commit_safe p r swapped scale v bounds

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnBloom.LoopSafety
