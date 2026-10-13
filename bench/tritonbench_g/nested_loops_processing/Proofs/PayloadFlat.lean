import bench.tritonbench_g.nested_loops_processing.Proofs.PayloadIO
import bench.tritonbench_g.nested_loops_processing.Proofs.PayloadSafetySource
import bench.tritonbench_g.nested_loops_processing.Proofs.RealFlat
import VeriTile.Triton.Memory.KernelCore

namespace VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.PayloadReal
open VeriTile.Triton TypedSource FullReal
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

theorem source_flattenOk (payload : TileDType) (src dst : RegionName) (sm sn : Int)
    (hptr : payload ≠ .ptr) (hblock : payload ≠ .blockPtr) :
    (PayloadAlgorithm.program payload src dst sm sn).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [PayloadAlgorithm.program, PayloadAlgorithm.outer, PayloadAlgorithm.middle,
    PayloadAlgorithm.inner, PayloadAlgorithm.load, PayloadAlgorithm.store,
    PayloadAlgorithm.inputAdvance, PayloadAlgorithm.outputAdvance,
    prefixBody, FullReal.inner, FullReal.middle, FullReal.outer, FullReal.source, kernel, ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def, hptr, hblock]

/-- Named flat-memory IO with an independent ordered-write oracle and the full
raw-cell frame. Untouched destination cells retain their original stored type. -/
def FlatCopyContract (payload : TileDType) (A : FlatAlloc) (program : Kernel) (s : BlockState)
    (src dst : RegionName) (sm sn : Nat) : Prop :=
  ∃ final, exec (A.flattenKernel program) (A.flattenState s) = some final ∧
    BlockState.WriteWithin (flatFootprint A (outputFootprint dst sm sn))
      (A.flattenState s) final ∧
    ∀ address, address < A.extent dst →
      final.mem A.flat (A.addr dst address) = A.trCell
        (CopyMath.expected (fun n ↦ storedCell payload (s.readMemValue payload src n))
          (s.mem dst) sm sn address)

/-- The library flattening bridge's zero-undef assumption is explicit; the
region-level source theorem has no such restriction. -/
theorem source_flat_io (payload : TileDType)
    (hptr : payload ≠ .ptr) (hblock : payload ≠ .blockPtr) (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0) (s : BlockState)
    (src dst : RegionName) (sm sn : Nat) (hsep : src ≠ dst)
    (hsrc : sm + 27 * sn < A.extent src)
    (hdst : sm + 47 * sn < A.extent dst)
    (hu : s.undef = fun _ _ ↦ 0) :
    FlatCopyContract payload A (PayloadAlgorithm.program payload src dst sm sn) s src dst sm sn := by
  obtain ⟨final, he, hf, hv⟩ := source_io payload s src dst sm sn hsep
  refine ⟨A.flattenState final, ?_, transport_frame A _ s final hf, ?_⟩
  · rw [A.exec_flatten hd hcov _ s (source_safe payload A.extent src dst sm sn s hsrc hdst)
      (source_flattenOk payload src dst sm sn hptr hblock) hu, he, Option.map_some]
  · intro address haddress
    have hr : dst ∈ A.regions := by
      by_contra hnot
      have := hcov dst hnot
      omega
    rw [A.flattenState_mem_addr hd final hr haddress, hv]

end VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.PayloadReal

#guard_msgs (drop info) in
#auditModuleAxioms
