import bench.tritonbench_g.bmm_chunk_fwd.Proofs.RealIO

/-! Named float/signed-metadata three-coordinate IO inherits the shared flat-memory bridge.
The caller proves source safety and initial-array values; no output equation
or complete execution is a public precondition. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealIO
open VeriTile.Triton
noncomputable section

theorem pins_of_core {BM BN K : Nat} (io : IO BM BN K) (s : BlockState) (vals : io.toU.Ctx)
    (hp : ∀ ch j, io.toU.imask ch vals (s.pids 0) (s.pids 1) (s.pids 2) j →
      (io.toU.ity ch).read s (io.toU.ibuf ch) (io.toU.iwin ch vals (s.pids 0) (s.pids 1) (s.pids 2) j) = vals ch j) :
    Pins io s (io.inputs vals) := by
  have h (ch : Fin 4) (i : TileIndex (inputShape BM BN K ch)) :
      io.inputMask ch (s.pids 0) (s.pids 1) (s.pids 2) i →
      (inputType ch).read s (io.inputRegion ch) (io.inputWindow ch (s.pids 0) (s.pids 1) (s.pids 2) i) =
        vals ch (tilePos _ i) := by
    simpa only [IO.toU, ChanTy.read, get_tilePos] using hp ch (tilePos _ i)
  exact ⟨fun i k ha => h 0 (i,k,PUnit.unit) ha,
    fun k j ha => h 1 (k,j,PUnit.unit) ha,
    fun hs i ha => h 2 (i,PUnit.unit) ⟨hs, ha⟩,
    fun hs j ha => h 3 (j,PUnit.unit) ⟨hs, ha⟩⟩

theorem bounds_of_core {BM BN K : Nat} (io : IO BM BN K) (s : BlockState)
    (bounds : RegionBounds) (vals : io.toU.Ctx)
    (hi : ∀ ch j, io.toU.imask ch vals (s.pids 0) (s.pids 1) (s.pids 2) j →
      io.toU.iwin ch vals (s.pids 0) (s.pids 1) (s.pids 2) j < bounds (io.toU.ibuf ch))
    (ho : ∀ ch j, io.toU.omask ch vals (s.pids 0) (s.pids 1) (s.pids 2) j →
      io.toU.owin ch vals (s.pids 0) (s.pids 1) (s.pids 2) j < bounds (io.toU.obuf ch)) :
    Bounds io s bounds := by
  have ih (ch : Fin 4) (i : TileIndex (inputShape BM BN K ch)) :
      io.inputMask ch (s.pids 0) (s.pids 1) (s.pids 2) i →
      io.inputWindow ch (s.pids 0) (s.pids 1) (s.pids 2) i < bounds (io.inputRegion ch) := by
    simpa only [IO.toU, get_tilePos] using hi ch (tilePos _ i)
  have oh (i : TileIndex [BM, BN]) :
      io.outputMask (s.pids 0) (s.pids 1) (s.pids 2) i →
      io.outputWindow (s.pids 0) (s.pids 1) (s.pids 2) i < bounds io.Out := by
    have h := ho (0 : Fin 1) (tilePos [BM, BN] i)
    change io.outputMask (s.pids 0) (s.pids 1) (s.pids 2)
        ((TileShape.allIndices [BM, BN]).get (tilePos [BM, BN] i)) →
      io.outputWindow (s.pids 0) (s.pids 1) (s.pids 2)
        ((TileShape.allIndices [BM, BN]).get (tilePos [BM, BN] i)) < bounds io.Out at h
    rw [get_tilePos] at h
    exact h
  exact ⟨ih 0, ih 1, fun hs i ha => ih 2 i ⟨hs, ha⟩, fun hs i ha => ih 3 i ⟨hs, ha⟩, oh⟩

theorem IO.Implements.intro {BM BN K : Nat} (io : IO BM BN K)
    (f : Nat → Nat → Nat → RealInputs.Inputs BM BN K → TileIndex [BM, BN] → ℝ)
    (hok : io.kernel.toAlgKernel.FlattenOk)
    (hts : ∀ bounds s xs, Pins io s xs → Bounds io s bounds → io.kernel.toAlgKernel.TraceSafe bounds s)
    (hrun : ∀ s xs, s.undef = (fun _ _ => 0) → Pins io s xs →
      ∃ t, exec io.kernel.toAlgKernel s = some t ∧ Results io s t (f (s.pids 0) (s.pids 1) (s.pids 2) xs)) :
    io.Implements f := by
  apply UKernelIO.Implements3D.intro io.toU hok
  · intro bounds s vals hp hi ho _
    exact hts bounds s (io.inputs vals) (pins_of_core io s vals hp) (bounds_of_core io s bounds vals hi ho)
  · intro s vals hu hp
    obtain ⟨t, he, hy⟩ := hrun s (io.inputs vals) hu (pins_of_core io s vals hp)
    refine ⟨t, he, fun _ j ha => hy.output _ ha, ?_⟩
    intro r a ho _
    apply hy.frame r a
    intro i ha
    have h := ho (0 : Fin 1) (tilePos [BM, BN] i)
    change io.outputMask (s.pids 0) (s.pids 1) (s.pids 2)
        ((TileShape.allIndices [BM, BN]).get (tilePos [BM, BN] i)) →
      r ≠ io.Out ∨ a ≠ io.outputWindow (s.pids 0) (s.pids 1) (s.pids 2)
        ((TileShape.allIndices [BM, BN]).get (tilePos [BM, BN] i)) at h
    rw [get_tilePos] at h
    exact h ha

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealIO
