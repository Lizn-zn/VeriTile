import bench.tritonbench_g.context_attn_fwd.Proofs.SafeOps
namespace VeriTile.Bench.TritonBenchG.ContextAttnFwd.PureSegments
open VeriTile.Triton Formula Program Frames
noncomputable section
set_option maxHeartbeats 600000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

def assignsOnly (code : List Stmt) (names : List RegName) : Prop :=
  ∀ st ∈ code, match st with
    | .assign _ _ name _ => name ∈ names
    | _ => False

theorem assignments_frame (names : List RegName) (code : List Stmt) (u v : BlockState)
    (ha : assignsOnly code names) (he : stepStmts code u = some v) : Frame names u v := by
  induction code generalizing u with
  | nil =>
    have hv : u = v := by simpa only [stepStmts,Option.some.injEq] using he
    subst v
    exact ⟨rfl,rfl,rfl,rfl,fun _ _ _ _ => rfl⟩
  | cons st tail ih =>
    have hn := ha st List.mem_cons_self
    obtain ⟨mid,hstep,htail⟩ := TraceProofs.cons_inv he
    have ft := ih mid (fun st hs => ha st (List.mem_cons_of_mem _ hs)) htail
    cases st <;> try contradiction
    case assign dtype shape name e =>
      obtain ⟨x,_,rfl⟩ := TraceProofs.assign_inv hstep
      constructor
      · funext R off
        simpa only [BlockState.setReg_mem] using congrFun (congrFun ft.memory R) off
      · simpa only [BlockState.setReg_pids] using ft.pids
      · simpa only [BlockState.setReg_numPids] using ft.numPids
      · funext R off
        simpa only [BlockState.setReg_undef] using congrFun (congrFun ft.undef R) off
      · intro dt sh nm hnm
        rw [ft.registers dt sh nm hnm]
        apply BlockState.setReg_ne_name
        intro heq
        exact hnm (heq ▸ hn)

def middle (p : Layout) (swapped : Bool) (scale : ℝ) :=
  LoopProgram.scores p swapped scale ++ LoopProgram.stats p

def names : List RegName := ["qk","mask","m_ij","p","l_ij","alpha","l_i","acc"]

theorem middle_assigns (p : Layout) (swapped : Bool) (scale : ℝ) :
    assignsOnly (middle p swapped scale) names := by
  simp [middle,assignsOnly,LoopProgram.scores,LoopProgram.stats,names]

theorem middle_safe (p : Layout) (swapped : Bool) (scale : ℝ) (u : BlockState) (bounds : RegionBounds) :
    Stmt.TraceSafeList bounds (middle p swapped scale) u := by
  apply TraceProofs.all_safe
  cases swapped <;> simp [middle,LoopProgram.scores,LoopProgram.stats,
    LoopProgram.scaling,LoopProgram.causalMask,LoopProgram.vec,LoopProgram.mat,LoopProgram.column,
    LoopProgram.keyView,LoopProgram.lanes,PreludeProgram.n,PreludeProgram.z,PreludeProgram.rowView,PreludeProgram.rows,
    Stmt.TraceSafe,Op.SafeAt.eq_def]

theorem split (p : Layout) (r : Regions) (swapped : Bool) (scale : ℝ) :
    LoopProgram.body p r swapped scale =
      LoopProgram.keys p r ++ (middle p swapped scale ++ LoopProgram.values p r) := by
  rw [LoopProgram.source]
  simp only [middle,List.append_assoc]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnFwd.PureSegments
