import bench.tritonbench_g.context_attn_nopad.Proofs.PreludeSafety
import bench.tritonbench_g.context_attn_nopad.Proofs.LoopSafety
import bench.tritonbench_g.context_attn_nopad.Proofs.EpilogueSafety
namespace VeriTile.Bench.TritonBenchG.ContextAttnNopad.WholeSafety
open VeriTile.Triton Formula Program PreludeProgram PreludeComplete
noncomputable section
set_option maxHeartbeats 700000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

theorem remaining (p : Layout) (r : Regions) (swapped : Bool) (scale : ℝ)
    (initial : BlockState) (hBN : 0 < p.BN)
    (bounds : RegionBounds) (hb : AddressBounds.Valid p r (launch initial) (inputs initial r) bounds)
    (n c : Nat) (u : BlockState)
    (hrem : c+n = count p (launch initial) (inputs initial r))
    (hs : Invariant.Static p r initial u) (hd : Invariant.Dynamic p r initial scale c u) :
    Stmt.forRangeIntTraceSafe bounds "start_n" ((c*p.BN : Nat) : Int)
      (Bounds.stop p (launch initial) (inputs initial r)) p.BN (LoopProgram.body p r swapped scale) u := by
  induction n generalizing c u with
  | zero =>
    have heq : c = count p (launch initial) (inputs initial r) := by omega
    subst c
    rw [Stmt.forRangeIntTraceSafe,if_neg (by omega),if_neg (by
      have := Bounds.loop_finished p (launch initial) (inputs initial r) hBN
      omega)]
    trivial
  | succ n ih =>
    have hc : c < count p (launch initial) (inputs initial r) := by omega
    obtain ⟨mid,he,hs',hd'⟩ := InvariantStep.indexed_step p r swapped scale initial u c hBN
      (add_nonneg hb.advance (Int.natCast_nonneg (c*p.BN))) hs hd
    rw [Stmt.forRangeIntTraceSafe,if_neg (by omega),
      if_pos ((Bounds.loop_test p (launch initial) (inputs initial r) c hBN).mp hc),he]
    refine ⟨LoopSafety.body_safe p r swapped scale initial _ c bounds hb hc
      (InvariantStep.static_index p r initial u _ hs) (by simp),?_⟩
    have hindex : ((c*p.BN : Nat) : Int)+(p.BN : Int) = (((c+1)*p.BN : Nat) : Int) := by push_cast; ring
    rw [hindex]
    exact ih (c+1) mid (by omega) hs' hd'

theorem loop_safe (p : Layout) (r : Regions) (swapped : Bool) (scale : ℝ)
    (initial : BlockState) (hBN : 0 < p.BN)
    (bounds : RegionBounds) (hb : AddressBounds.Valid p r (launch initial) (inputs initial r) bounds) :
    Stmt.TraceSafe bounds (WholeProgram.loop p r swapped scale) (initialized p r initial) := by
  have h := remaining p r swapped scale initial hBN bounds hb
    (count p (launch initial) (inputs initial r)) 0 (initialized p r initial) (by omega)
    (Invariant.static_initialized p r initial) (Invariant.dynamic_initialized p r initial scale)
  simp only [Nat.zero_mul,Nat.cast_zero] at h
  simp only [WholeProgram.loop,Stmt.TraceSafe,n,z,Op.SafeAt.eq_def,true_and,evalOp.eq_def]
  simpa [initialized,PreludeQuery.queried,offsetState,metadataState,launchState,PreludeOffsets.prepared,PreludeMetadata.prepared,PreludeLaunch.launched,Formula.limit,Formula.sequence,Nat.cast_mul,Nat.cast_add,Tile.scalar,Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex,Option.bind,
    Bounds.stop,launch,NumericDType.mul,NumericDType.add,mul_comm,ite_mul,-Tile.scalar_eta] using h

theorem complete (p : Layout) (r : Regions) (storage : Storage) (swapped : Bool) (scale : ℝ)
    (initial : BlockState) (hBN : 0 < p.BN)
    (bounds : RegionBounds) (hb : AddressBounds.Valid p r (launch initial) (inputs initial r) bounds) :
    (kernel p r storage swapped scale).eraseDType.toAlgKernel.TraceSafe bounds initial := by
  rw [Kernel.TraceSafe,Program.storage_erasure,WholeProgram.source,List.append_assoc]
  apply TraceProofs.append_safe
  · exact PreludeSafety.complete p r initial bounds hb
  intro s he
  rw [PreludeComplete.executes] at he
  cases he
  simp only [List.cons_append,List.nil_append]
  obtain ⟨mid,hl,hs,hd⟩ := LoopExecution.complete p r swapped scale initial hBN hb.advance
  have he : stepStmt (WholeProgram.loop p r swapped scale) (initialized p r initial) = some mid := by
    rw [WholeExecution.loop_executes]
    exact hl
  rw [TraceProofs.cons_of_run_iff he]
  refine ⟨loop_safe p r swapped scale initial hBN bounds hb,?_⟩
  apply EpilogueSafety.safe p r (initial.pids 1) ((inputs initial r).start (initial.pids 0))
    (Formula.sequence p (launch initial) (inputs initial r)) (fun i => queryRow p (launch initial) i.val)
    mid bounds hs.head hs.start hs.sequence hs.rows hs.channels
  intro i hi
  exact AddressBounds.inside_toNat (hb.output i.1 i.2.1 hi)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnNopad.WholeSafety
