import bench.tritonbench_g.chunk_delta_fwd.Proofs.RealFlat

/-! IO contract for the complete delta-rule program. Its three outputs are
observed through explicit windows and masks. Initial memory feeds independent
real arrays; successful execution and the full-cell frame are conclusions. -/
namespace VeriTile.Bench.TritonBenchG.ChunkDeltaFwd.RealIO
open VeriTile.Triton RealProgram
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 18000

structure Outputs (p : Layout) where
  history : Nat → Fin p.BK → Fin p.BV → ℝ
  corrected : Nat → Fin p.BV → ℝ
  final : Fin p.BK → Fin p.BV → ℝ

structure IO (p : Layout) where
  kernel : ComputeKernel
  projection : kernel.toAlgorithm? = .ok kernel.toAlgKernel
  read : BlockState → Inputs
  seed : BlockState → Nat → ℝ
  valid : BlockState → Prop
  bounds : BlockState → RegionBounds → Prop
  history : RegionName
  corrected : RegionName
  final : RegionName
  storeFinal : Bool
  historyWindow : Launch → Nat → Fin p.BK → Fin p.BV → Nat
  valueWindow : Launch → Nat → Fin p.BV → Nat
  finalWindow : Launch → Fin p.BK → Fin p.BV → Nat
  stateMask : Launch → Fin p.BK → Fin p.BV → Prop
  valueMask : Launch → Nat → Fin p.BV → Prop

def IO.regionWrites {p : Layout} (io : IO p) (l : Launch) : WriteFootprint := fun cell =>
  ((cell.1 = io.history ∧ ∃ c e j, c < p.NT ∧ io.stateMask l e j ∧ cell.2 = io.historyWindow l c e j) ∨
   (cell.1 = io.corrected ∧ ∃ t j, io.valueMask l t j ∧ cell.2 = io.valueWindow l t j)) ∨
  (io.storeFinal = Bool.true ∧ cell.1 = io.final ∧ ∃ e j, io.stateMask l e j ∧ cell.2 = io.finalWindow l e j)

def IO.writes {p : Layout} (io : IO p) (A : FlatAlloc) (l : Launch) : WriteFootprint := fun cell =>
  cell.1 = A.flat ∧ ∃ rg off, io.regionWrites l (rg,off) ∧ A.addr rg off = cell.2

/-- The unspecified-value premise applies to omitted source padding. It does
not assert that Triton supplies zero. Allocation intervals are disjoint, and
all active read/write addresses must fit their declared extents. -/
def IO.Implements {p : Layout} (io : IO p)
    (f : Launch → Inputs → (Nat → ℝ) → Outputs p) : Prop :=
  ∀ A : FlatAlloc, A.Disjoint → (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ s : BlockState, s.undef = (fun _ _ => 0) → io.valid s → io.bounds s A.extent →
    ∃ v, exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s) = some v ∧
      (∀ c e j, c < p.NT → io.stateMask (RealInit.launch s) e j →
        v.readMem A.flat (A.addr io.history (io.historyWindow (RealInit.launch s) c e j)) =
          (f (RealInit.launch s) (io.read s) (io.seed s)).history c e j) ∧
      (∀ t j, io.valueMask (RealInit.launch s) t j →
        v.readMem A.flat (A.addr io.corrected (io.valueWindow (RealInit.launch s) t j)) =
          (f (RealInit.launch s) (io.read s) (io.seed s)).corrected t j) ∧
      (io.storeFinal = Bool.true → ∀ e j, io.stateMask (RealInit.launch s) e j →
        v.readMem A.flat (A.addr io.final (io.finalWindow (RealInit.launch s) e j)) =
          (f (RealInit.launch s) (io.read s) (io.seed s)).final e j) ∧
      BlockState.WriteWithin (io.writes A (RealInit.launch s)) (A.flattenState s) v

scoped notation:25 io " ⊨ " f => IO.Implements io f

theorem projection (p : Layout) (r : Regions) (kd value initial final : TypedSource.Storage)
    (swapped useSeed storeFinal : Bool) :
    (RealCandidate.source p r kd value initial final swapped useSeed storeFinal).eraseDType.toAlgorithm? =
      .ok (RealCandidate.source p r kd value initial final swapped useSeed storeFinal).eraseDType.toAlgKernel := by
  have hp : (RealCandidate.source p r kd value initial final swapped useSeed storeFinal).toAlgorithm? =
      .ok (RealCandidate.source p r kd value initial final swapped useSeed storeFinal).toAlgKernel := by
    cases value <;> cases final <;> cases swapped <;> rfl
  simp only [ComputeKernel.eraseDType, hp, ComputeKernel.toAlgorithm?_fromKernelBody,
    ComputeKernel.toAlgKernel_fromKernelBody]

def sourceIO (p : Layout) (r : Regions) (kd value initial final : TypedSource.Storage)
    (swapped useSeed storeFinal : Bool) : IO p where
  kernel := (RealCandidate.source p r kd value initial final swapped useSeed storeFinal).eraseDType
  projection := projection p r kd value initial final swapped useSeed storeFinal
  read := fun s => ⟨s.readMem r.key,s.readMem r.delta,s.readMem r.value⟩
  seed := fun s => s.readMem r.initial
  valid := fun s => s.pids 0 = 0 ∧ RealInnerLoop.NoCollision p (RealInit.launch s) ∧
    RealOuterLoop.HistoryNoCollision p (RealInit.launch s) ∧
    (storeFinal = Bool.true → RealStateStore.NoCollision p (RealInit.launch s) (RealInit.base p (RealInit.launch s)) p.V)
  bounds := fun s => RealSafety.Bounds p (RealInit.launch s) r useSeed storeFinal
  history := r.history
  corrected := r.corrected
  final := r.final
  storeFinal := storeFinal
  historyWindow := fun l c e j => RealOuterStep.historyAddress p l c e.val j.val
  valueWindow := fun l t j => valueAddress p l t j.val
  finalWindow := fun l e j => RealStateStore.address p l (RealInit.base p l) p.V e.val j.val
  stateMask := fun l e j => RealStateStore.active p l e.val j.val
  valueMask := fun l t j => activeValue p l t j.val

def expected (p : Layout) (useSeed : Bool) (l : Launch) (a : Inputs) (a₀ : Nat → ℝ) : Outputs p :=
  let state := RealWindow.expected p l a (RealInit.seed p l a₀) useSeed
  { history := fun c e j => state c e.val (valueIndex p l j.val)
    corrected := fun t j => RealMath.corrected (RealValues.arrays p l a (RealInit.seed p l a₀)) p.K
      (state (t / p.BT)) t (valueIndex p l j.val)
    final := fun e j => state p.NT e.val (valueIndex p l j.val) }

theorem implements (p : Layout) (r : Regions) (kd value initial final : TypedSource.Storage)
    (swapped useSeed storeFinal : Bool) (hK : p.K ≤ p.BK) (hBT : 0 < p.BT) (hBC : 0 < p.BC)
    (hcover : p.T ≤ p.NT * p.BT) (hs : RealOuterStep.Separate r)
    (hfsep : storeFinal = Bool.true → r.history ≠ r.final ∧ r.corrected ≠ r.final) :
    sourceIO p r kd value initial final swapped useSeed storeFinal ⊨ expected p useSeed := by
  intro A hd hcov s hu hvalid hb
  obtain ⟨v,he,ho,hf⟩ := RealFlat.source_executes A hd hcov p r kd value initial final swapped useSeed storeFinal
    ⟨s.readMem r.key,s.readMem r.delta,s.readMem r.value⟩ (s.readMem r.initial) s hu
    hvalid.1 hK hBT hBC hcover hs hfsep hvalid.2.1 hvalid.2.2.1 hvalid.2.2.2
    (fun _ => rfl) (fun _ => rfl) (fun _ => rfl) (fun _ => rfl) hb
  exact ⟨v,he,ho.history,ho.corrected,ho.final,hf⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkDeltaFwd.RealIO
