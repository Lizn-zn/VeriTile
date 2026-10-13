import bench.tritonbench_g.nested_loops_processing.Proofs.RealIO
import bench.tritonbench_g.nested_loops_processing.Proofs.RealSafetySource
import VeriTile.Triton.Memory.KernelCore

namespace VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.FullReal
open VeriTile.Triton TypedSource
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

def flatFootprint (A : FlatAlloc) (P : WriteFootprint) : WriteFootprint :=
  fun a ↦ a.1 = A.flat ∧ ∃ r o, P (r, o) ∧ a.2 = A.addr r o

theorem transport_frame (A : FlatAlloc) (P : WriteFootprint) (s t : BlockState)
    (h : BlockState.WriteWithin P s t) :
    BlockState.WriteWithin (flatFootprint A P) (A.flattenState s) (A.flattenState t) := by
  intro r off hnot
  by_cases hr : r = A.flat
  · subst r
    simp only [FlatAlloc.flattenState]
    unfold FlatAlloc.readFlat
    cases hdec : A.decode off with
    | none => rfl
    | some cell =>
      obtain ⟨r, o⟩ := cell
      obtain ⟨_, haddr, _⟩ := A.decode_sound hdec
      change A.trCell (s.mem r o) = A.trCell (t.mem r o)
      apply congrArg A.trCell
      apply h r o
      intro hp
      exact hnot ⟨rfl, r, o, hp, haddr⟩
  · simp only [FlatAlloc.flattenState, if_neg hr]


theorem source_flattenOk (src dst : RegionName) (sm sn : Int) :
    (source src dst sm sn).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [source, kernel, ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-- Named flat-memory IO with an independent ordered-write oracle and the full
raw-cell frame. Untouched destination cells retain their original stored type. -/
def FlatCopyContract (A : FlatAlloc) (program : Kernel) (s : BlockState)
    (src dst : RegionName) (sm sn : Nat) : Prop :=
  ∃ final, exec (A.flattenKernel program) (A.flattenState s) = some final ∧
    BlockState.WriteWithin (flatFootprint A (outputFootprint dst sm sn))
      (A.flattenState s) final ∧
    ∀ address, address < A.extent dst →
      final.mem A.flat (A.addr dst address) = A.trCell
        (CopyMath.expected (fun n ↦ MemCell.of .int (s.readMemValue .int src n))
          (s.mem dst) sm sn address)

/-- The library flattening bridge's zero-undef assumption is explicit; the
region-level source theorem has no such restriction. -/
theorem source_flat_io (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0) (s : BlockState)
    (src dst : RegionName) (sm sn : Nat) (hsep : src ≠ dst)
    (hsrc : sm + 27 * sn < A.extent src)
    (hdst : sm + 47 * sn < A.extent dst)
    (hu : s.undef = fun _ _ ↦ 0) :
    FlatCopyContract A (source src dst sm sn) s src dst sm sn := by
  obtain ⟨final, he, hf, hv⟩ := source_io s src dst sm sn hsep
  refine ⟨A.flattenState final, ?_, transport_frame A _ s final hf, ?_⟩
  · rw [A.exec_flatten hd hcov _ s (source_safe A.extent src dst sm sn s hsrc hdst)
      (source_flattenOk src dst sm sn) hu, he, Option.map_some]
  · intro address haddress
    have hr : dst ∈ A.regions := by
      by_contra hnot
      have := hcov dst hnot
      omega
    rw [A.flattenState_mem_addr hd final hr haddress, hv]

end VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.FullReal

#guard_msgs (drop info) in
#auditModuleAxioms
