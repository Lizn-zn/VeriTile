import bench.tritonbench_g.var_len_copy.Proofs.RealInitialize

namespace VeriTile.Bench.TritonBenchG.VarLenCopy.FullReal
open VeriTile Triton

/-- Successful complete signed-loop execution copies precisely the actual
source span from the initial source buffer and preserves every other cell. -/
theorem source_io (s : BlockState) (os ol ns : Region .int) (src dst : RegionName)
    (B : Nat) (length : Int) (oldStart newStart : Nat)
    (hB : 0 < B) (hsd : src ≠ dst)
    (hL : s.readMemValue .int ol.cast (s.pids 0) = length)
    (hO : s.readMemValue .int os.cast (s.pids 0) = (oldStart : Int))
    (hN : s.readMemValue .int ns.cast (s.pids 0) = (newStart : Int)) :
    ∃ final, exec (source os ol ns src dst B) s = some final ∧
      BlockState.WriteWithin (footprint dst newStart length B) s final ∧
      (∀ n, n < Math.copiedLength length B →
        final.readMem dst (newStart + n) = s.readMem src (oldStart + n)) ∧
      final.pids = s.pids ∧ final.numPids = s.numPids ∧ final.undef = s.undef := by
  obtain ⟨final,he,hf⟩ := source_run s os ol ns src dst B length oldStart newStart hB hsd hL hO hN
  refine ⟨final,he,hf.frame,?_,hf.pids,hf.numPids,hf.undef⟩
  intro n hn
  obtain ⟨c,lane,hc,hl,hL,rfl⟩ := (Math.touched_iff length hB).mpr hn
  exact hf.values c lane ((Math.loop_test length hB).mp hc) hl hL

/-- Empty and negative signed lengths execute no stores at any address. -/
theorem source_nonpositive (s : BlockState) (os ol ns : Region .int) (src dst : RegionName)
    (B : Nat) (length : Int) (oldStart newStart : Nat)
    (hB : 0 < B) (hsd : src ≠ dst) (hle : length ≤ 0)
    (hL : s.readMemValue .int ol.cast (s.pids 0) = length)
    (hO : s.readMemValue .int os.cast (s.pids 0) = (oldStart : Int))
    (hN : s.readMemValue .int ns.cast (s.pids 0) = (newStart : Int)) :
    ∃ final, exec (source os ol ns src dst B) s = some final ∧ final.mem = s.mem := by
  obtain ⟨final,he,hf,_,_,_,_⟩ := source_io s os ol ns src dst B length oldStart newStart hB hsd hL hO hN
  refine ⟨final,he,?_⟩
  funext r o
  symm
  apply hf r o
  simp [footprint, Math.copiedLength_nonpositive hle B]

end VeriTile.Bench.TritonBenchG.VarLenCopy.FullReal

#guard_msgs (drop info) in
#auditModuleAxioms
