import bench.tritonbench_g.fast_rms_layernorm.Proofs.MathBridge
namespace VeriTile.Bench.TritonBenchG.FastRmsLayernorm.ForwardRun
open VeriTile.Triton TypedSource RealExecution
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000

def original (isGemma : Bool) (Y X W r : RegionName) (YR XR WR RR N B : Nat) (eps : ℝ) : Kernel :=
  (if isGemma then gemma .fp32 .fp32 Bool.false Y X W r YR XR WR RR N B eps
   else forward .fp32 .fp32 Bool.false Y X W r YR XR WR RR N B eps).eraseDType.toAlgKernel

def weightOffset (isGemma : Bool) (WR j : Nat) := if isGemma then j else j*WR

theorem flatten_ok (isGemma : Bool) (Y X W r : RegionName) (YR XR WR RR N B : Nat) (eps : ℝ) :
    (original isGemma Y X W r YR XR WR RR N B eps).FlattenOk := by
  cases isGemma
  · exact rms_layernorm_forward_flattenOk Y X W r YR XR WR RR N B eps
  · exact gemma_rms_layernorm_forward_flattenOk Y X W r YR XR RR N B eps

theorem trace_safe (isGemma : Bool) (Y X W r : RegionName) (YR XR WR RR N B : Nat) (eps : ℝ)
    (bounds : RegionBounds) (u : BlockState)
    (hx : ∀ j : Fin B,j.val<N → u.pids 0*XR+j.val<bounds X)
    (hw : ∀ j : Fin B,j.val<N → weightOffset isGemma WR j.val<bounds W)
    (hy : ∀ j : Fin B,j.val<N → u.pids 0*YR+j.val<bounds Y)
    (hr : u.pids 0*RR<bounds r) :
    Kernel.TraceSafe bounds (original isGemma Y X W r YR XR WR RR N B eps) u := by
  cases isGemma
  · exact rms_layernorm_forward_traceSafe Y X W r YR XR WR RR N B eps bounds u hx hw hy hr
  · exact gemma_rms_layernorm_forward_traceSafe Y X W r YR XR RR N B eps bounds u hx hw hy hr

theorem run (isGemma : Bool) (Y X W r : RegionName) (YR XR WR RR N B : Nat) (eps : ℝ)
    (u : BlockState) (hne : Y≠r) :
    ∃ v,exec (original isGemma Y X W r YR XR WR RR N B eps) u=some v ∧
      (∀ j : Fin B,j.val<N → v.readMem Y (u.pids 0*YR+j.val)=
        RealMath.output isGemma N eps (fun k : Fin B => u.readMem X (u.pids 0*XR+k.val))
          (fun k : Fin B => u.readMem W (weightOffset isGemma WR k.val)) j) ∧
      v.readMem r (u.pids 0*RR)=RealMath.invStd N eps (fun k : Fin B => u.readMem X (u.pids 0*XR+k.val)) := by
  cases isGemma
  · obtain ⟨v,he,hy,hr,_⟩ := rms_layernorm_forward_region_run Y X W r YR XR WR RR N B eps hne u
      (fun k : Fin B => u.readMem X (u.pids 0*XR+k.val))
      (fun k : Fin B => u.readMem W (k.val*WR)) (fun _ _ => rfl) (fun _ _ => rfl)
    exact ⟨v,he,fun j hj => (hy j hj).trans (MathBridge.output_eq _ _ _ _ _ _),hr.trans (MathBridge.inv_eq _ _ _ _)⟩
  · obtain ⟨v,he,hy,hr,_⟩ := gemma_rms_layernorm_forward_region_run Y X W r YR XR RR N B eps hne u
      (fun k : Fin B => u.readMem X (u.pids 0*XR+k.val))
      (fun k : Fin B => u.readMem W k.val) (fun _ _ => rfl) (fun _ _ => rfl)
    exact ⟨v,he,fun j hj => (hy j hj).trans (MathBridge.gemma_output_eq _ _ _ _ _ _),hr.trans (MathBridge.inv_eq _ _ _ _)⟩

theorem frame (isGemma : Bool) (Y X W r : RegionName) (YR XR WR RR N B : Nat) (eps : ℝ)
    (u v : BlockState) (he : exec (original isGemma Y X W r YR XR WR RR N B eps) u=some v)
    (q : RegionName) (o : Nat)
    (hY : ∀ j : Fin B,j.val<N → ¬(Y=q ∧ u.pids 0*YR+j.val=o))
    (hR : ¬(r=q ∧ u.pids 0*RR=o)) : v.mem q o=u.mem q o := by
  cases isGemma
  · exact rms_layernorm_forward_frame Y X W r YR XR WR RR N B eps u v he q o hY hR
  · exact gemma_rms_layernorm_forward_frame Y X W r YR XR RR N B eps u v he q o hY hR

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FastRmsLayernorm.ForwardRun
