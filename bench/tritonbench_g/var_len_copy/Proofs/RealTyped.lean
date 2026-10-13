import bench.tritonbench_g.var_len_copy.Proofs.RealCandidate

namespace VeriTile.Bench.TritonBenchG.VarLenCopy.FullReal
open VeriTile Triton TypedSource

theorem typed_flat_io (input output : Storage) (v : Bool) (A : FlatAlloc) (hd : A.Disjoint)
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
    ∃ final, exec (A.flattenKernel (kernel input output v os ol ns src dst B).eraseDType.toAlgKernel) (A.flattenState s) = some final ∧
      BlockState.WriteWithin (flatFootprint A (footprint dst newStart length B))
        (A.flattenState s) final ∧
      (∀ n, n < Math.copiedLength length B →
        final.readMem A.flat (A.addr dst (newStart + n)) = s.readMem src (oldStart + n)) := by
  obtain ⟨final,he,hf,hv⟩ := source_flat_io A hd hcov s os ol ns src dst B length oldStart newStart
    hB hsd hL hO hN hLb hOb hNb hsrc hdst hdr hu
  refine ⟨final,?_,hf,hv⟩
  rw [TypedSource.projection]
  cases v
  · exact he
  · exact (flat_exec_commutes A os ol ns src dst B (A.flattenState s)).symm.trans he

end VeriTile.Bench.TritonBenchG.VarLenCopy.FullReal

#guard_msgs (drop info) in
#auditModuleAxioms
