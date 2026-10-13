import bench.tritonbench_g.var_len_copy.Proofs.DiscreteRealInitialize
import bench.tritonbench_g.var_len_copy.Proofs.RealSafetySource
import bench.tritonbench_g.var_len_copy.Proofs.DiscreteRealSafetyLoop

namespace VeriTile.Bench.TritonBenchG.VarLenCopy.DiscreteReal
open VeriTile Triton DiscreteSource
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

theorem source_safe (k : Kind) (bounds : RegionBounds) (s : BlockState)
    (os ol ns : Region .int) (src dst : RegionName)
    (B : Nat) (length : Int) (oldStart newStart : Nat)
    (hB : 0 < B) (hsd : src ≠ dst)
    (hL : s.readMemValue .int ol.cast (s.pids 0) = length)
    (hO : s.readMemValue .int os.cast (s.pids 0) = (oldStart : Int))
    (hN : s.readMemValue .int ns.cast (s.pids 0) = (newStart : Int))
    (hLb : s.pids 0 < bounds ol.cast) (hOb : s.pids 0 < bounds os.cast)
    (hNb : s.pids 0 < bounds ns.cast)
    (hsrc : oldStart + Math.copiedLength length B ≤ bounds src)
    (hdst : newStart + Math.copiedLength length B ≤ bounds dst) :
    (source k os ol ns src dst B).TraceSafe bounds s := by
  obtain ⟨mid,hp,hm⟩ := prefix_run k s os ol ns src dst B length oldStart newStart hL hO hN
  unfold Kernel.TraceSafe
  rw [source_decomposition, FullReal.trace_append_iff]
  refine ⟨FullReal.prefix_safe bounds s os ol ns src dst B hLb hOb hNb, ?_⟩
  intro t ht
  have : t = mid := Option.some.inj (ht.symm.trans hp)
  subst t
  rw [Stmt.TraceSafeList]
  constructor
  · simp only [Stmt.TraceSafe, Op.SafeAt.eq_def, evalOp.eq_def, hm.lengthReg,
      Tile.scalar_data, Option.bind_eq_bind, Option.bind, true_and]
    simpa only [Nat.zero_mul, Nat.cast_zero] using
      loop_safe k bounds s os ol ns src dst B length oldStart newStart hB hsd hsrc hdst
        0 mid (Nat.zero_le _) hm
  · cases stepStmt _ mid <;> simp [Stmt.TraceSafeList]

end VeriTile.Bench.TritonBenchG.VarLenCopy.DiscreteReal

#guard_msgs (drop info) in
#auditModuleAxioms
