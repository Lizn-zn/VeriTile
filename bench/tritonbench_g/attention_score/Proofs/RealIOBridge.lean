import bench.tritonbench_g.attention_score.Proofs.RealIO

/-! Translate named Q/K/M windows to the existing shared flat-memory theorem.
Concrete kernels must supply checked projection, trace safety and an independent
region-memory output formula. No second interpreter or output assumption is added. -/
namespace VeriTile.Bench.TritonBenchG.AttentionScore.RealIO
open VeriTile.Triton
noncomputable section

theorem pins_of_core {rows B D : Nat} (io : IO rows B D) (s : BlockState) (vals : io.toU.Ctx)
    (hp : ∀ ch j, io.toU.imask ch vals (s.pids 0) (s.pids 1) (s.pids 2) j →
      (io.toU.ity ch).read s (io.toU.ibuf ch) (io.toU.iwin ch vals (s.pids 0) (s.pids 1) (s.pids 2) j) = vals ch j) :
    Pins io s (io.inputs vals) := by
  have h (ch : Fin 3) (i : TileIndex (inputShape rows B D ch)) :
      io.inputMask ch (s.pids 0) (s.pids 1) (s.pids 2) i →
      s.readMem (io.inputRegion ch) (io.inputWindow ch (s.pids 0) (s.pids 1) (s.pids 2) i) =
        vals ch (tilePos _ i) := by
    simpa only [IO.toU, ChanTy.read, get_tilePos] using hp ch (tilePos _ i)
  exact ⟨fun r d ha => h 0 (r,d,PUnit.unit) ha,
    fun k d ha => h 1 (d,k,PUnit.unit) ha, fun r => h 2 (r,PUnit.unit) trivial⟩

theorem bounds_of_core {rows B D : Nat} (io : IO rows B D) (s : BlockState)
    (bounds : RegionBounds) (vals : io.toU.Ctx)
    (hi : ∀ ch j, io.toU.imask ch vals (s.pids 0) (s.pids 1) (s.pids 2) j →
      io.toU.iwin ch vals (s.pids 0) (s.pids 1) (s.pids 2) j < bounds (io.toU.ibuf ch))
    (ho : ∀ ch j, io.toU.omask ch vals (s.pids 0) (s.pids 1) (s.pids 2) j →
      io.toU.owin ch vals (s.pids 0) (s.pids 1) (s.pids 2) j < bounds (io.toU.obuf ch)) :
    Bounds io s bounds := by
  have h (ch : Fin 3) (i : TileIndex (inputShape rows B D ch)) :
      io.inputMask ch (s.pids 0) (s.pids 1) (s.pids 2) i →
      io.inputWindow ch (s.pids 0) (s.pids 1) (s.pids 2) i < bounds (io.inputRegion ch) := by
    simpa only [IO.toU, get_tilePos] using hi ch (tilePos _ i)
  exact ⟨h 0, h 1, fun i => h 2 i trivial, ho (0 : Fin 1)⟩

theorem IO.Implements.intro {rows B D : Nat} (io : IO rows B D)
    (f : Nat → Nat → RealInputs.Inputs rows B D → Fin B → ℝ)
    (hok : io.kernel.toAlgKernel.FlattenOk)
    (hts : ∀ bounds s, Bounds io s bounds → io.kernel.toAlgKernel.TraceSafe bounds s)
    (hrun : ∀ s xs, Pins io s xs →
      ∃ t, exec io.kernel.toAlgKernel s = some t ∧ Results io s t (f (s.pids 0) (s.pids 1) xs)) :
    io.Implements f := by
  apply UKernelIO.Implements.intro io.toU hok
  · intro bounds s vals _ hi ho _
    exact hts bounds s (bounds_of_core io s bounds vals hi ho)
  · intro s vals _ hp
    obtain ⟨t,he,hy⟩ := hrun s (io.inputs vals) (pins_of_core io s vals hp)
    refine ⟨t,he,?_,?_⟩
    · intro _ j ha
      exact hy.output j ha
    · intro r a ho _
      apply hy.frame r a
      by_cases hr : r = io.Out
      · right
        intro i ha
        exact (ho (0 : Fin 1) i ha).resolve_left (fun hne => hne hr)
      · exact Or.inl hr

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttentionScore.RealIO
