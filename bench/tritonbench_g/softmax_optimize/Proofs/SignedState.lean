import bench.tritonbench_g.softmax_optimize.Proofs.SignedTransportLoop

namespace VeriTile.Bench.TritonBenchG.SoftmaxOptimize.SignedTransport
open VeriTile.Triton

/-- These three scratch names are overwritten before use in each loop body. -/
def Scratch (name : RegName) : Prop :=
  name = "start_n" ∨ name = "n_offsets" ∨ name = "offset"

def Shared (name : RegName) : Prop := ¬ Scratch name ∧ name ≠ "prev_multiple"

/-- Signed index registers may differ from the historical natural registers.
Every other register and every memory/launch component agrees exactly. -/
structure LiveRelated (signed natural : BlockState) : Prop where
  pids_eq : signed.pids = natural.pids
  numPids_eq : signed.numPids = natural.numPids
  mem_eq : signed.mem = natural.mem
  undef_eq : signed.undef = natural.undef
  regs_eq : ∀ dtype shape name, Shared name →
    signed.regs dtype shape name = natural.regs dtype shape name
  prev_eq : signed.regs .int [] "prev_multiple" =
    (natural.regs .nat [] "prev_multiple").map
      (fun t ↦ Tile.scalar (t.data PUnit.unit : Int))

namespace LiveRelated

theorem pid {s t : BlockState} (h : LiveRelated s t) : s.pid = t.pid := by
  unfold BlockState.pid
  rw [h.pids_eq]

theorem readMem {s t : BlockState} (h : LiveRelated s t) (r : RegionName) (o : Nat) :
    s.readMem r o = t.readMem r o := by
  unfold BlockState.readMem
  rw [h.mem_eq]

theorem setShared {s t : BlockState} (h : LiveRelated s t)
    (name : RegName) (dtype : TileDType) (shape : TileShape) (value : Tile dtype shape)
    (hname : Shared name) :
    LiveRelated (s.setReg name dtype shape value) (t.setReg name dtype shape value) := by
  constructor
  · exact h.pids_eq
  · exact h.numPids_eq
  · exact h.mem_eq
  · exact h.undef_eq
  · intro dt sh nm hn
    by_cases he : nm = name
    · subst nm
      simp [BlockState.setReg]
    · simpa only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ he] using h.regs_eq dt sh nm hn
  · simpa only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hname.2.symm] using h.prev_eq

theorem setScratch {s t : BlockState} (h : LiveRelated s t)
    (name : RegName) (dt ds : TileDType) (sh ss : TileShape)
    (v : Tile dt sh) (w : Tile ds ss) (hscratch : Scratch name) :
    LiveRelated (s.setReg name dt sh v) (t.setReg name ds ss w) := by
  have hp : "prev_multiple" ≠ name := by
    rcases hscratch with h | h | h <;> subst name <;> decide
  constructor
  · exact h.pids_eq
  · exact h.numPids_eq
  · exact h.mem_eq
  · exact h.undef_eq
  · intro dtype shape nm hn
    have he : nm ≠ name := by intro he; subst nm; exact hn.1 hscratch
    simpa only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ he]
      using h.regs_eq dtype shape nm hn
  · simpa only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hp] using h.prev_eq

theorem setPrevious {s t : BlockState} (h : LiveRelated s t) (n : Nat) :
    LiveRelated (s.setReg "prev_multiple" .int [] (Tile.scalar (n : Int)))
      (t.setReg "prev_multiple" .nat [] (Tile.scalar n)) := by
  constructor
  · exact h.pids_eq
  · exact h.numPids_eq
  · exact h.mem_eq
  · exact h.undef_eq
  · intro dt sh nm hn
    simpa only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hn.2]
      using h.regs_eq dt sh nm hn
  · simp only [BlockState.setReg_same, Option.map_some, Tile.scalar_data]

end LiveRelated

/-- Composition preserves simultaneous failure and the supplied successful-state relation. -/
theorem results_bind {R S : BlockState → BlockState → Prop}
    {a b : Option BlockState} (h : ResultsRelated R a b)
    (f g : BlockState → Option BlockState)
    (hfg : ∀ s t, R s t → ResultsRelated S (f s) (g t)) :
    ResultsRelated S (a.bind f) (b.bind g) := by
  cases a <;> cases b <;> simp_all [ResultsRelated]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxOptimize.SignedTransport
