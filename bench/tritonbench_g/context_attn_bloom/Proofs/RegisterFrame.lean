import bench.tritonbench_g.context_attn_bloom.Proofs.Invariant
namespace VeriTile.Bench.TritonBenchG.ContextAttnBloom.RegisterFrame
open VeriTile.Triton Math Program RealSetup
noncomputable section

/-- A local execution fragment changes only the named registers. All tagged
memory cells, launch ids and undefined-load values are preserved exactly. -/
structure Frame (writes : List RegName) (before after : BlockState) : Prop where
  memory : after.mem = before.mem
  pids : after.pids = before.pids
  numPids : after.numPids = before.numPids
  undef : after.undef = before.undef
  registers : ∀ dtype shape name, name ∉ writes → after.regs dtype shape name = before.regs dtype shape name

theorem same (u : BlockState) : Frame [] u u :=
  ⟨rfl,rfl,rfl,rfl,fun _ _ _ _ => rfl⟩

theorem write {writes : List RegName} {before after : BlockState} (h : Frame writes before after)
    (name : RegName) (dtype : TileDType) (shape : TileShape) (value : Tile dtype shape) :
    Frame (name::writes) before (after.setReg name dtype shape value) := by
  constructor
  · funext r o
    simpa only [BlockState.setReg_mem] using congrFun (congrFun h.memory r) o
  · simpa only [BlockState.setReg_pids] using h.pids
  · simpa only [BlockState.setReg_numPids] using h.numPids
  · funext r o
    simpa only [BlockState.setReg_undef] using congrFun (congrFun h.undef r) o
  · intro dt sh nm hn
    have hname : nm ≠ name := by
      intro heq
      exact hn (List.mem_cons.mpr (Or.inl heq))
    have hrest : nm ∉ writes := by
      intro hm
      exact hn (List.mem_cons_of_mem name hm)
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hname,h.registers dt sh nm hrest]

theorem then_frame {first second : List RegName} {u v w : BlockState}
    (h1 : Frame first u v) (h2 : Frame second v w) : Frame (second++first) u w := by
  refine ⟨h2.memory.trans h1.memory,h2.pids.trans h1.pids,h2.numPids.trans h1.numPids,
    h2.undef.trans h1.undef,?_⟩
  intro dt sh name hn
  have hs : name ∉ second := fun h => hn (List.mem_append_left first h)
  have hf : name ∉ first := fun h => hn (List.mem_append_right second h)
  rw [h2.registers dt sh name hs,h1.registers dt sh name hf]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnBloom.RegisterFrame
