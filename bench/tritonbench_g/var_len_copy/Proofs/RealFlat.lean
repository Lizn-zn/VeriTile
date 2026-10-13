import bench.tritonbench_g.var_len_copy.Proofs.RealIO
import bench.tritonbench_g.var_len_copy.Proofs.RealSafetySource
import VeriTile.Triton.Memory.KernelCore

namespace VeriTile.Bench.TritonBenchG.VarLenCopy.FullReal
open VeriTile Triton TypedSource
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
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


theorem source_flattenOk (os ol ns : Region .int) (src dst : RegionName) (B : Nat) :
    (source os ol ns src dst B).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [source,kernel,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?,ComputeDType.eraseDType,Kernel.eraseDType,
    Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType.eq_def,
    NumericDType.eraseDType,ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType,Region.cast,Except.bind,bind, StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-- Flat IO uses the bridge's explicit zero-undef domain. -/
theorem source_flat_io (A : FlatAlloc) (hd : A.Disjoint)
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
    ∃ final, exec (A.flattenKernel (source os ol ns src dst B)) (A.flattenState s) = some final ∧
      BlockState.WriteWithin (flatFootprint A (footprint dst newStart length B))
        (A.flattenState s) final ∧
      (∀ n, n < Math.copiedLength length B →
        final.readMem A.flat (A.addr dst (newStart + n)) = s.readMem src (oldStart + n)) := by
  obtain ⟨final,he,hf,hv,_,_,_⟩ := source_io s os ol ns src dst B length oldStart newStart hB hsd hL hO hN
  refine ⟨A.flattenState final, ?_, transport_frame A _ s final hf, ?_⟩
  · rw [A.exec_flatten hd hcov _ s
      (source_safe A.extent s os ol ns src dst B length oldStart newStart hB hsd
        hL hO hN hLb hOb hNb hsrc hdst) (source_flattenOk os ol ns src dst B) hu,
      he, Option.map_some]
  · intro n hn
    rw [A.flattenState_readMem hd final hdr (by omega)]
    exact hv n hn

end VeriTile.Bench.TritonBenchG.VarLenCopy.FullReal

#guard_msgs (drop info) in
#auditModuleAxioms
