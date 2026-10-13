import bench.tritonbench_g.var_len_copy.Proofs.DiscreteRealIO
import bench.tritonbench_g.var_len_copy.Proofs.DiscreteRealSafetySource
import bench.tritonbench_g.var_len_copy.Proofs.RealFlat
import VeriTile.Triton.Memory.KernelCore

namespace VeriTile.Bench.TritonBenchG.VarLenCopy.DiscreteReal
open VeriTile Triton DiscreteSource
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

abbrev flatFootprint := FullReal.flatFootprint

theorem source_flattenOk (k : Kind) (os ol ns : Region .int) (src dst : RegionName) (B : Nat) :
    (source k os ol ns src dst B).FlattenOk := by
  unfold Kernel.FlattenOk
  cases k <;> simp [source,representative,kernel,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?,ComputeDType.eraseDType,Kernel.eraseDType,
    Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType.eq_def,
    NumericDType.eraseDType,ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType,Region.cast,Except.bind,bind, StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-- Flat IO uses the bridge's explicit zero-undef domain. -/
theorem source_flat_io (k : Kind) (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0) (s : BlockState)
    (os ol ns : Region .int) (src dst : RegionName)
    (B : Nat) (length : Int) (oldStart newStart : Nat)
    (hB : 0 < B) (hsd : src ≠ dst)
    (hL : s.readMemValue .int ol.cast (s.pids 0) = length)
    (hO : s.readMemValue .int os.cast (s.pids 0) = (oldStart : Int))
    (hN : s.readMemValue .int ns.cast (s.pids 0) = (newStart : Int))
    (hLb : s.pids 0 < A.extent ol.cast) (hOb : s.pids 0 < A.extent os.cast)
    (hNb : s.pids 0 < A.extent ns.cast)
    (hsrc : oldStart + Math.copiedLength length B ≤ A.extent src)
    (hdst : newStart + Math.copiedLength length B ≤ A.extent dst)
    (hdr : dst ∈ A.regions) (hu : s.undef = fun _ _ ↦ 0) :
    ∃ final, exec (A.flattenKernel (source k os ol ns src dst B)) (A.flattenState s) = some final ∧
      BlockState.WriteWithin (flatFootprint A (footprint dst newStart length B))
        (A.flattenState s) final ∧
      (∀ n, n < Math.copiedLength length B →
        final.readMemValue k.dtype A.flat (A.addr dst (newStart + n)) = s.readMemValue k.dtype src (oldStart + n)) := by
  obtain ⟨final,he,hf,hv,_,_,_⟩ := source_io k s os ol ns src dst B length oldStart newStart hB hsd hL hO hN
  refine ⟨A.flattenState final, ?_, FullReal.transport_frame A _ s final hf, ?_⟩
  · rw [A.exec_flatten hd hcov _ s
      (source_safe k A.extent s os ol ns src dst B length oldStart newStart hB hsd
        hL hO hN hLb hOb hNb hsrc hdst) (source_flattenOk k os ol ns src dst B) hu,
      he, Option.map_some]
  · intro n hn
    rw [A.flattenState_readMemValue hd final hdr (by omega) k.dtype (by cases k <;> decide)]
    exact hv n hn

end VeriTile.Bench.TritonBenchG.VarLenCopy.DiscreteReal

#guard_msgs (drop info) in
#auditModuleAxioms
