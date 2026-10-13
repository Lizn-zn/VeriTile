import bench.tritonbench_g.kldiv_triton.Proofs.FlatIO
import bench.tritonbench_g.kldiv_triton.Proofs.CandidateSafety
import bench.tritonbench_g.kldiv_triton.Proofs.SourceDomain

namespace VeriTile.Bench.TritonBenchG.KldivTriton.Contracts
open VeriTile.Triton TypedSource

/-- Successful full flat execution and independent loss values, with the exact chunk footprint. -/
def ForwardIO (kernel : ComputeKernel) (Y T O : RegionName)
    (sy st so N B : Nat) (flag : Bool) (reduction : Nat) : Prop :=
  ∀ (A : FlatAlloc) (s : BlockState), A.Disjoint → A.regions = [Y, T, O] →
    (∀ r, r ∉ A.regions → A.extent r = 0) → s.undef = (fun _ _ ↦ 0) →
    (∀ k : Fin N, s.pids 0 * sy + k.val < A.extent Y) →
    (∀ k : Fin N, s.pids 0 * st + k.val < A.extent T) →
    (∀ k : Fin (FullMath.outputCount N B reduction),
      s.pids 0 * so + k.val < A.extent O) →
    ∃ t, exec (A.flattenKernel kernel.eraseDType.toAlgKernel) (A.flattenState s) = some t ∧
      (∀ k : Fin (FullMath.outputCount N B reduction),
        t.readMem A.flat (A.addr O (s.pids 0 * so + k.val)) =
          FullMath.forwardOutput flag (fun i ↦ s.readMem Y (s.pids 0 * sy + i))
            (fun i ↦ s.readMem T (s.pids 0 * st + i)) N B reduction k.val) ∧
      (∀ r off, (r ≠ A.flat ∨ ∀ k : Fin (FullMath.outputCount N B reduction),
        off ≠ A.addr O (s.pids 0 * so + k.val)) →
          t.mem r off = (A.flattenState s).mem r off)

/-- Successful full flat backward execution, independent scalar gradients and every untouched cell. -/
def BackwardIO (kernel : ComputeKernel) (T O : RegionName)
    (st so N : Nat) (flag : Bool) : Prop :=
  ∀ (A : FlatAlloc) (s : BlockState), A.Disjoint → A.regions = [T, O] →
    (∀ r, r ∉ A.regions → A.extent r = 0) → s.undef = (fun _ _ ↦ 0) →
    (∀ k : Fin N, s.pids 0 * st + k.val < A.extent T) →
    (∀ k : Fin N, s.pids 0 * so + k.val < A.extent O) →
    ∃ t, exec (A.flattenKernel kernel.eraseDType.toAlgKernel) (A.flattenState s) = some t ∧
      (∀ k : Fin N, t.readMem A.flat (A.addr O (s.pids 0 * so + k.val)) =
        FullMath.gradient flag (s.readMem T (s.pids 0 * st + k.val))) ∧
      (∀ r off, (r ≠ A.flat ∨ ∀ k : Fin N, off ≠ A.addr O (s.pids 0 * so + k.val)) →
        t.mem r off = (A.flattenState s).mem r off)

theorem forward_io (pred : Storage) (target : MathStorage) (output : Storage) (v : Bool)
    (Y T O : RegionName) (sy st so N B : Nat) (flag : Bool) (reduction : Nat)
    (hB : 0 < B) (hYO : Y ≠ O) (hTO : T ≠ O) :
    ForwardIO (forward pred target output v Y T O sy st so N B flag reduction)
      Y T O sy st so N B flag reduction := by
  intro A s hd hr hc hu hby hbt hbo
  obtain ⟨t, he, hv, hf⟩ := FlatIO.forward_run A s Y T O sy st so N B flag reduction
    hd hr hc hu hB hYO hTO hby hbt hbo
  refine ⟨t, ?_, hv, hf⟩
  rw [Source.forward_projection]
  exact (Candidate.forward_flat A v Y T O sy st so N B flag reduction
    (A.flattenState s)).trans he

theorem backward_default_io (target output : Storage) (v : Bool)
    (T O : RegionName) (st so N B : Nat) (hB : 0 < B) (hTO : T ≠ O) :
    BackwardIO (backwardDefault target output v T O st so N B) T O st so N Bool.false := by
  intro A s hd hr hc hu hbt hbo
  obtain ⟨t, he, hv, hf⟩ := FlatIO.backward_run A s T O st so N B Bool.false
    hd hr hc hu hB hTO hbt hbo
  refine ⟨t, ?_, hv, hf⟩
  rw [Source.backward_default_projection]
  exact (Candidate.backward_flat A v T O st so N B Bool.false (A.flattenState s)).trans he

theorem backward_log_io (target : MathStorage) (output : Storage) (v : Bool)
    (T O : RegionName) (st so N B : Nat) (hB : 0 < B) (hTO : T ≠ O) :
    BackwardIO (backwardLog target output v T O st so N B) T O st so N Bool.true := by
  intro A s hd hr hc hu hbt hbo
  obtain ⟨t, he, hv, hf⟩ := FlatIO.backward_run A s T O st so N B Bool.true
    hd hr hc hu hB hTO hbt hbo
  refine ⟨t, ?_, hv, hf⟩
  rw [Source.backward_log_projection]
  exact (Candidate.backward_flat A v T O st so N B Bool.true (A.flattenState s)).trans he

theorem forward_safe (pred : Storage) (target : MathStorage) (output : Storage) (v : Bool)
    (bounds : RegionBounds) (s : BlockState) (Y T O : RegionName)
    (sy st so N B : Nat) (flag : Bool) (reduction : Nat)
    (hB : 0 < B) (hYO : Y ≠ O) (hTO : T ≠ O)
    (hby : ∀ k : Fin N, s.pids 0 * sy + k.val < bounds Y)
    (hbt : ∀ k : Fin N, s.pids 0 * st + k.val < bounds T)
    (hbo : ∀ k : Fin (FullMath.outputCount N B reduction),
      s.pids 0 * so + k.val < bounds O) :
    (forward pred target output v Y T O sy st so N B flag reduction).eraseDType.toAlgKernel.TraceSafeR
      .triv bounds s := by
  rw [Source.forward_projection]
  exact (Candidate.forward_safe_iff bounds v Y T O sy st so N B flag reduction s).mpr
    (FullForward.source_safe bounds s Y T O sy st so N B flag reduction hB hYO hTO hby hbt hbo)

theorem backward_default_safe (target output : Storage) (v : Bool)
    (bounds : RegionBounds) (s : BlockState) (T O : RegionName) (st so N B : Nat)
    (hTO : T ≠ O) (hbt : ∀ k : Fin N, s.pids 0 * st + k.val < bounds T)
    (hbo : ∀ k : Fin N, s.pids 0 * so + k.val < bounds O) :
    (backwardDefault target output v T O st so N B).eraseDType.toAlgKernel.TraceSafeR
      .triv bounds s := by
  rw [Source.backward_default_projection]
  exact (Candidate.backward_safe_iff bounds v T O st so N B Bool.false s).mpr
    (FullBackward.source_safe bounds s T O st so N B Bool.false hTO hbt hbo)

theorem backward_log_safe (target : MathStorage) (output : Storage) (v : Bool)
    (bounds : RegionBounds) (s : BlockState) (T O : RegionName) (st so N B : Nat)
    (hTO : T ≠ O) (hbt : ∀ k : Fin N, s.pids 0 * st + k.val < bounds T)
    (hbo : ∀ k : Fin N, s.pids 0 * so + k.val < bounds O) :
    (backwardLog target output v T O st so N B).eraseDType.toAlgKernel.TraceSafeR
      .triv bounds s := by
  rw [Source.backward_log_projection]
  exact (Candidate.backward_safe_iff bounds v T O st so N B Bool.true s).mpr
    (FullBackward.source_safe bounds s T O st so N B Bool.true hTO hbt hbo)

end VeriTile.Bench.TritonBenchG.KldivTriton.Contracts
#guard_msgs (drop info) in
#auditModuleAxioms
