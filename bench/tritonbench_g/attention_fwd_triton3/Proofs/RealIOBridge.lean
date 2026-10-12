import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealIO

/-! Translate the six named input views and three named outputs to the shared
UKernelIO contract. All proof obligations concern region execution; the core
supplies flat-memory transport, readback and preservation outside output cells. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealIO
open VeriTile.Triton
noncomputable section

theorem pins_of_core {BM BD NK : Nat} (io : IO BM BD NK) (s : BlockState) (vals : io.toU.Ctx)
    (hp : ∀ ch j, io.toU.imask ch vals (s.pids 0) (s.pids 1) (s.pids 2) j →
      (io.toU.ity ch).read s (io.toU.ibuf ch) (io.toU.iwin ch vals (s.pids 0) (s.pids 1) (s.pids 2) j) = vals ch j) :
    Pins io s (io.inputs vals) := by
  have h (ch : Fin 6) (i : TileIndex (inputShape BM BD NK ch)) :
      io.inputMask ch (s.pids 0) (s.pids 1) (s.pids 2) i →
      s.readMem (io.inputRegion ch) (io.inputWindow ch (s.pids 0) (s.pids 1) (s.pids 2) i) =
        vals ch (tilePos _ i) := by
    simpa only [IO.toU, ChanTy.read, get_tilePos] using hp ch (tilePos _ i)
  exact ⟨fun r d ha => h 0 (r, d, PUnit.unit) ha,
    fun k d => h 1 (k, d, PUnit.unit) trivial,
    fun k d => h 2 (k, d, PUnit.unit) trivial,
    fun ha r => h 3 (r, PUnit.unit) ha,
    fun ha r => h 4 (r, PUnit.unit) ha,
    fun ha r d => h 5 (r, d, PUnit.unit) ha⟩

theorem bounds_of_core {BM BD NK : Nat} (io : IO BM BD NK) (s : BlockState)
    (bounds : RegionBounds) (vals : io.toU.Ctx)
    (hi : ∀ ch j, io.toU.imask ch vals (s.pids 0) (s.pids 1) (s.pids 2) j →
      io.toU.iwin ch vals (s.pids 0) (s.pids 1) (s.pids 2) j < bounds (io.toU.ibuf ch))
    (ho : ∀ ch j, io.toU.omask ch vals (s.pids 0) (s.pids 1) (s.pids 2) j →
      io.toU.owin ch vals (s.pids 0) (s.pids 1) (s.pids 2) j < bounds (io.toU.obuf ch)) :
    Bounds io s bounds := by
  have ih (ch : Fin 6) (i : TileIndex (inputShape BM BD NK ch)) :
      io.inputMask ch (s.pids 0) (s.pids 1) (s.pids 2) i →
      io.inputWindow ch (s.pids 0) (s.pids 1) (s.pids 2) i < bounds (io.inputRegion ch) := by
    simpa only [IO.toU, get_tilePos] using hi ch (tilePos _ i)
  have oh (ch : Fin 3) (i : TileIndex (outputShape BM BD ch)) :
      io.outputMask ch → io.outputWindowFor ch (s.pids 0) (s.pids 1) (s.pids 2) i < bounds (io.outputRegion ch) := by
    simpa only [IO.toU, get_tilePos] using ho ch (tilePos _ i)
  refine ⟨ih 0, fun i => ih 1 i trivial, fun i => ih 2 i trivial,
    fun i => oh 0 i trivial, ?_, fun i => oh 2 i trivial⟩
  rintro (hr | hw) i
  · exact ih 4 i hr
  · exact oh 1 i hw

/-- The named contract inherits the shared flat-memory correctness theorem.
The caller proves concrete source safety and region execution against its
independent formula; no final-output or whole-equivalence premise is assumed. -/
theorem IO.Implements.intro {BM BD NK : Nat} (io : IO BM BD NK)
    (f : Nat → Nat → RealInputs.Inputs BM BD NK → Outputs BM BD)
    (hok : io.kernel.toAlgKernel.FlattenOk)
    (hts : ∀ bounds s xs, Pins io s xs → Bounds io s bounds → io.kernel.toAlgKernel.TraceSafe bounds s)
    (hrun : ∀ s xs, s.undef = (fun _ _ => 0) → Pins io s xs →
      ∃ t, exec io.kernel.toAlgKernel s = some t ∧ Results io s t (f (s.pids 0) (s.pids 1) xs)) :
    io.Implements f := by
  apply UKernelIO.Implements.intro io.toU hok
  · intro bounds s vals hp hi ho _
    exact hts bounds s (io.inputs vals) (pins_of_core io s vals hp) (bounds_of_core io s bounds vals hi ho)
  · intro s vals hu hp
    obtain ⟨t, he, hy⟩ := hrun s (io.inputs vals) hu (pins_of_core io s vals hp)
    refine ⟨t, he, ?_, ?_⟩
    · intro ch j ha
      fin_cases ch
      · exact hy.center _
      · exact hy.denominator ha _
      · exact hy.output _
    · intro r a ho _
      have h (ch : Fin 3) (i : TileIndex (outputShape BM BD ch)) :
          io.outputMask ch → r ≠ io.outputRegion ch ∨ a ≠ io.outputWindowFor ch (s.pids 0) (s.pids 1) (s.pids 2) i := by
        simpa only [IO.toU, get_tilePos] using ho ch (tilePos _ i)
      apply hy.frame r a
      · by_cases hr : r = io.M
        · right
          intro i
          exact (h 0 i trivial).resolve_left (by simpa [IO.outputRegion] using hr)
        · exact Or.inl hr
      · by_cases hw : io.writeDenominator
        · right
          by_cases hr : r = io.L
          · right
            intro i
            exact (h 1 i hw).resolve_left (by simpa [IO.outputRegion] using hr)
          · exact Or.inl hr
        · exact Or.inl hw
      · by_cases hr : r = io.Out
        · right
          intro i
          exact (h 2 i trivial).resolve_left (by simpa [IO.outputRegion] using hr)
        · exact Or.inl hr

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealIO
