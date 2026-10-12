import bench.tritonbench_g.context_attn_bloom.Proofs.SafeOps
namespace VeriTile.Bench.TritonBenchG.ContextAttnBloom.PureSegments
open VeriTile.Triton Math Program RealSetup RegisterFrame
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

def middle (p : Layout) (r : Regions) (swapped : Bool) (scale : ℝ) : List Stmt :=
  ScoresProgram.scores p r swapped scale ++ UpdateProgram.stats p r swapped scale ++
    UpdateProgram.normalizer p r swapped scale ++ UpdateProgram.scales p r swapped scale

def names : List RegName :=
  ["qk","m_ij","p","l_ij","m_i_new","alpha","beta","l_i_new","p_scale","acc_scale","acc"]

theorem middle_assigns (p : Layout) (r : Regions) (swapped : Bool) (scale : ℝ) :
    assignsOnly (middle p r swapped scale) names := by
  unfold middle
  rw [ScoresProgram.source,(UpdateProgram.sources p r swapped scale).1,
    (UpdateProgram.sources p r swapped scale).2.1,(UpdateProgram.sources p r swapped scale).2.2.1]
  simp [assignsOnly,ScoresProgram.code,UpdateProgram.statsCode,UpdateProgram.normalizerCode,
    UpdateProgram.scalesCode,names]

theorem middle_safe (p : Layout) (r : Regions) (swapped : Bool) (scale : ℝ)
    (u : BlockState) (bounds : RegionBounds) :
    Stmt.TraceSafeList bounds (middle p r swapped scale) u := by
  unfold middle
  rw [ScoresProgram.source,(UpdateProgram.sources p r swapped scale).1,
    (UpdateProgram.sources p r swapped scale).2.1,(UpdateProgram.sources p r swapped scale).2.2.1]
  apply TraceProofs.all_safe
  cases swapped <;> simp [ScoresProgram.code,ScoresProgram.scaling,ScoresProgram.causalMask,
    UpdateProgram.statsCode,UpdateProgram.normalizerCode,UpdateProgram.scalesCode,
    UpdateProgram.vec,UpdateProgram.mat,UpdateProgram.column,UpdateProgram.rescaleGuard,
    PreludeProgram.n,PreludeProgram.z,Stmt.TraceSafe,Op.SafeAt.eq_def]

theorem commit_safe (p : Layout) (r : Regions) (swapped : Bool) (scale : ℝ)
    (u : BlockState) (bounds : RegionBounds) :
    Stmt.TraceSafeList bounds (UpdateProgram.commit p r swapped scale) u := by
  rw [(UpdateProgram.sources p r swapped scale).2.2.2]
  apply TraceProofs.all_safe
  simp [UpdateProgram.commitCode,UpdateProgram.vec,UpdateProgram.mat,Stmt.TraceSafe,Op.SafeAt.eq_def]

theorem split (p : Layout) (r : Regions) (swapped : Bool) (scale : ℝ) :
    LoopProgram.body p r swapped scale = LoopKeysProgram.keys p r ++
      (middle p r swapped scale ++ (ValuesProgram.values p r ++ UpdateProgram.commit p r swapped scale)) := by
  rw [LoopProgram.split]
  simp only [middle,List.append_assoc]

end
end VeriTile.Bench.TritonBenchG.ContextAttnBloom.PureSegments
