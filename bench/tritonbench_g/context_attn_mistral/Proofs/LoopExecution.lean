import bench.tritonbench_g.context_attn_mistral.Proofs.InvariantStep
import bench.tritonbench_g.context_attn_mistral.Proofs.Bounds
namespace VeriTile.Bench.TritonBenchG.ContextAttnMistral.LoopExecution
open VeriTile.Triton Formula Program PreludeComplete
noncomputable section
set_option maxHeartbeats 600000
set_option maxRecDepth 3000

theorem remaining (p : Layout) (r : Regions) (swapped : Bool) (scale : ℝ)
    (initial : BlockState) (hBN : 0 < p.BN)
    (hstart : 0 ≤ (inputs initial r).start (launch initial).batch)
    (n c : Nat) (u : BlockState)
    (hrem : c+n = count p (launch initial) (inputs initial r))
    (hs : Invariant.Static p r initial u) (hd : Invariant.Dynamic p r initial scale c u) :
    ∃ next,
      stepForRangeIntAux "start_n" ((c*p.BN : Nat) : Int)
        (Bounds.stop p (launch initial) (inputs initial r)) p.BN (LoopProgram.body p r swapped scale) u = some next ∧
      Invariant.Static p r initial next ∧
      Invariant.Dynamic p r initial scale (count p (launch initial) (inputs initial r)) next := by
  induction n generalizing c u with
  | zero =>
    have heq : c = count p (launch initial) (inputs initial r) := by omega
    subst c
    refine ⟨u,?_,hs,hd⟩
    exact stepForRangeIntAux.step_ge (by omega) (Bounds.loop_finished p (launch initial) (inputs initial r) hBN)
  | succ n ih =>
    have hc : c < count p (launch initial) (inputs initial r) := by omega
    have hshift : 0 ≤ (inputs initial r).start (launch initial).batch+(c*p.BN : Nat) := by positivity
    obtain ⟨mid,he,hs',hd'⟩ := InvariantStep.indexed_step p r swapped scale initial u c hBN hshift hs hd
    obtain ⟨next,hr,hns,hnd⟩ := ih (c+1) mid (by omega) hs' hd'
    refine ⟨next,?_,hns,hnd⟩
    rw [stepForRangeIntAux.step_lt (by omega) ((Bounds.loop_test p (launch initial) (inputs initial r) c hBN).mp hc),
      he,Option.bind_some]
    have hindex : ((c*p.BN : Nat) : Int)+(p.BN : Int) = (((c+1)*p.BN : Nat) : Int) := by push_cast; ring
    rw [hindex]
    exact hr

theorem complete (p : Layout) (r : Regions) (swapped : Bool) (scale : ℝ)
    (initial : BlockState) (hBN : 0 < p.BN)
    (hstart : 0 ≤ (inputs initial r).start (launch initial).batch) :
    ∃ next,
      stepForRangeIntAux "start_n" 0 (Bounds.stop p (launch initial) (inputs initial r))
        p.BN (LoopProgram.body p r swapped scale) (initialized p r initial) = some next ∧
      Invariant.Static p r initial next ∧
      Invariant.Dynamic p r initial scale (count p (launch initial) (inputs initial r)) next := by
  simpa only [zero_mul,Nat.cast_zero] using remaining p r swapped scale initial hBN hstart
    (count p (launch initial) (inputs initial r)) 0 (initialized p r initial) (by omega)
    (Invariant.static_initialized p r initial) (Invariant.dynamic_initialized p r initial scale)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnMistral.LoopExecution
