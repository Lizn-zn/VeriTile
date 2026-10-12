import bench.tritonbench_g.chunk_gated_attention.Proofs.RealHFlat

/-! Explicit IO for pre-update history and optional final state. Input arrays
come only from initial memory. Successful full execution, independent outputs
and preservation of all other tagged cells are conclusions of Implements. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealHIO
open VeriTile.Triton RealHMath RealHProgram
noncomputable section
set_option maxHeartbeats 500000
set_option maxRecDepth 5000
set_option linter.unusedSimpArgs false

structure Outputs (p : Layout) where
  history : Nat → Fin p.BK → Fin p.BV → ℝ
  final : Fin p.BK → Fin p.BV → ℝ

structure IO (p : Layout) where
  kernel : ComputeKernel
  projection : kernel.toAlgorithm? = .ok kernel.toAlgKernel
  read : BlockState → Inputs
  valid : BlockState → Prop
  bounds : BlockState → RegionBounds → Prop
  history : RegionName
  final : RegionName
  storeFinal : Bool
  historyWindow : Launch → Nat → Fin p.BK → Fin p.BV → Nat
  finalWindow : Launch → Fin p.BK → Fin p.BV → Nat
  stateMask : Launch → Fin p.BK → Fin p.BV → Prop

def IO.regionWrites {p : Layout} (io : IO p) (l : Launch) : WriteFootprint := fun cell =>
  (cell.1 = io.history ∧ ∃ c e j, c < p.NT ∧ io.stateMask l e j ∧ cell.2 = io.historyWindow l c e j) ∨
  (io.storeFinal = Bool.true ∧ cell.1 = io.final ∧ ∃ e j, io.stateMask l e j ∧ cell.2 = io.finalWindow l e j)

def IO.writes {p : Layout} (io : IO p) (A : FlatAlloc) (l : Launch) : WriteFootprint := fun cell =>
  cell.1 = A.flat ∧ ∃ rg off, io.regionWrites l (rg,off) ∧ A.addr rg off = cell.2

/-- Zero unspecified lanes are explicit: the source does not request zero
padding. Disjoint allocation intervals and actual active-address bounds are
also required. No prepared carry, history or final output is supplied. -/
def IO.Implements {p : Layout} (io : IO p) (f : Launch → Inputs → Outputs p) : Prop :=
  ∀ A : FlatAlloc, A.Disjoint → (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ s : BlockState, s.undef = (fun _ _ => 0) → io.valid s → io.bounds s A.extent →
    ∃ v, exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s) = some v ∧
      (∀ c e j, c < p.NT → io.stateMask (launch s) e j →
        v.readMem A.flat (A.addr io.history (io.historyWindow (launch s) c e j)) =
          (f (launch s) (io.read s)).history c e j) ∧
      (io.storeFinal = Bool.true → ∀ e j, io.stateMask (launch s) e j →
        v.readMem A.flat (A.addr io.final (io.finalWindow (launch s) e j)) =
          (f (launch s) (io.read s)).final e j) ∧
      BlockState.WriteWithin (io.writes A (launch s)) (A.flattenState s) v

scoped notation:25 io " ⊨ " f => IO.Implements io f

theorem projection (p : Layout) (r : Regions) (dot gate history : TypedSource.Storage)
    (seed final : FloatDType) (gateK hasSeed storeFinal swapped : Bool) :
    (source p r dot gate history seed final gateK hasSeed storeFinal swapped).eraseDType.toAlgorithm? =
      .ok (source p r dot gate history seed final gateK hasSeed storeFinal swapped).eraseDType.toAlgKernel := by
  simp only [source,ComputeKernel.eraseDType,RealHSource.projects,ComputeKernel.toAlgorithm?_fromKernelBody,
    ComputeKernel.toAlgKernel_fromKernelBody]

def sourceIO (p : Layout) (r : Regions) (dot gate history : TypedSource.Storage)
    (seed final : FloatDType) (gateK hasSeed storeFinal swapped : Bool) : IO p where
  kernel := (source p r dot gate history seed final gateK hasSeed storeFinal swapped).eraseDType
  projection := projection p r dot gate history seed final gateK hasSeed storeFinal swapped
  read := fun s => ⟨s.readMem r.key,s.readMem r.value,s.readMem r.gate,s.readMem r.seed⟩
  valid := fun s => RealHLoop.HistoryNoCollision p (launch s) ∧
    (storeFinal = Bool.true → RealHStore.NoCollision p (launch s) (RealHFinal.base p (launch s)) p.V 1)
  bounds := fun s => RealHSafety.Bounds p (launch s) r gateK hasSeed storeFinal
  history := r.history
  final := r.final
  storeFinal := storeFinal
  historyWindow := fun l c e j => historyAddress p l c (keyIndex p l e.val) (valueIndex p l j.val)
  finalWindow := fun l e j => stateAddress p l (keyIndex p l e.val) (valueIndex p l j.val)
  stateMask := fun l e j => RealHStore.active p l e.val j.val

def expected (p : Layout) (gateK hasSeed : Bool) (l : Launch) (a : Inputs) : Outputs p :=
  { history := fun c e j => closed p l a gateK hasSeed c (keyIndex p l e.val) (valueIndex p l j.val)
    final := fun e j => closed p l a gateK hasSeed p.NT (keyIndex p l e.val) (valueIndex p l j.val) }

theorem implements (p : Layout) (r : Regions) (dot gate history : TypedSource.Storage)
    (seed final : FloatDType) (gateK hasSeed storeFinal swapped : Bool)
    (hs : RealHStep.Separate r) (hfsep : storeFinal = Bool.true → r.history ≠ r.final) :
    sourceIO p r dot gate history seed final gateK hasSeed storeFinal swapped ⊨ expected p gateK hasSeed := by
  intro A hd hcov s hu hv hb
  obtain ⟨v,he,ho,hf⟩ := RealHFlat.source_executes A hd hcov p r dot gate history seed final
    gateK hasSeed storeFinal swapped ⟨s.readMem r.key,s.readMem r.value,s.readMem r.gate,s.readMem r.seed⟩ s
    hu hs hfsep hv.1 hv.2 (fun _ => rfl) (fun _ => rfl) (fun _ => rfl) (fun _ => rfl) hb
  refine ⟨v,he,ho.history,ho.final,?_⟩
  intro region off hn
  apply hf region off
  simpa only [IO.writes,IO.regionWrites,sourceIO,RealHFlat.footprint,RealHExecution.footprint,
    WriteFootprint.union,RealHLoop.footprint,RealHLoop.address,RealHStep.historyBase,
    RealHFinal.footprint,RealHStore.footprint,RealHStore.address,RealHFinal.base,
    historyAddress,stateAddress,Nat.mul_one] using hn

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealHIO
