import bench.tritonbench_g.context_attn_nopad.Proofs.SafeOps
import bench.tritonbench_g.context_attn_fwd.Proofs.PureSegments
namespace VeriTile.Bench.TritonBenchG.ContextAttnNopad.PureSegments
open VeriTile.Triton Formula Program LoopProgram PreludeProgram
open VeriTile.Bench.TritonBenchG.ContextAttnFwd.PureSegments (assignsOnly assignments_frame)
noncomputable section
set_option maxHeartbeats 600000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

def middle (p : Layout) (swapped : Bool) (scale : ℝ) : List Stmt :=
  scores p swapped scale ++ statsCode p ++ normalizerCode p ++ scalesCode p

def names : List RegName :=
  ["qk","m_ij","p","l_ij","m_i_new","alpha","beta","l_i_new","p_scale","acc_scale","acc"]

theorem middle_assigns (p : Layout) (swapped : Bool) (scale : ℝ) :
    assignsOnly (middle p swapped scale) names := by
  simp [middle,assignsOnly,scores,statsCode,normalizerCode,scalesCode,names]

theorem middle_safe (p : Layout) (swapped : Bool) (scale : ℝ)
    (u : BlockState) (bounds : RegionBounds) :
    Stmt.TraceSafeList bounds (middle p swapped scale) u := by
  apply TraceProofs.all_safe
  cases swapped <;> simp [middle,scores,statsCode,normalizerCode,scalesCode,
    scaling,causalMask,sentinel,vec,mat,column,rowView,keyView,rows,lanes,
    n,z,Stmt.TraceSafe,Op.SafeAt.eq_def]

theorem commit_safe (p : Layout) (u : BlockState) (bounds : RegionBounds) :
    Stmt.TraceSafeList bounds (commitCode p) u := by
  apply TraceProofs.all_safe
  simp [commitCode,vec,mat,Stmt.TraceSafe,Op.SafeAt.eq_def]

theorem split (p : Layout) (r : Regions) (swapped : Bool) (scale : ℝ) :
    body p r swapped scale = keys p r ++
      (middle p swapped scale ++ (values p r ++ commitCode p)) := by
  rw [LoopProgram.source]
  simp only [middle,List.append_assoc]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnNopad.PureSegments
