import bench.tritonbench_g.softmax_optimize.Proofs.SignedLoad
import bench.tritonbench_g.softmax_optimize.Proofs.SignedBodies

namespace VeriTile.Bench.TritonBenchG.SoftmaxOptimize.SignedTransport
open VeriTile.Triton

/-- The unchanged online recurrence observes only shared floating registers. -/
theorem online_update_related (X : RegionName) (N B : Nat) :
    StatementsRelated B ((loopAbody X N B).drop 4) ((loopAbody X N B).drop 4) := by
  have ref (name : RegName) (h : Shared name) := ref_related B .real [B] name h
  have hm := ref "m" (by simp [Shared, Scratch])
  have hi := ref "inp" (by simp [Shared, Scratch])
  have hz := ref "z" (by simp [Shared, Scratch])
  have hn := ref "new_m" (by simp [Shared, Scratch])
  have hmax := where_related (gt_related .real (Broadcast.nil.consSame) hm hi) hm hi
  have hsum := add_related .real (Broadcast.nil.consSame)
    (mul_related .real (Broadcast.nil.consSame)
      (exp_related (sub_related .real (Broadcast.nil.consSame) hm hn)) hz)
    (exp_related (sub_related .real (Broadcast.nil.consSame) hi hn))
  exact cons_related (assign_related "new_m" (by simp [Shared, Scratch]) hmax)
    (cons_related (assign_related "new_z" (by simp [Shared, Scratch]) hsum)
      (cons_related (assign_related "m" (by simp [Shared, Scratch]) hn)
        (cons_related (assign_related "z" (by simp [Shared, Scratch]) (ref "new_z" (by simp [Shared, Scratch])))
          (nil_related B))))

theorem input_full_suffix_related (X : RegionName) (N B : Nat) :
    StatementsRelated B ((loopAbody X N B).drop 3) ((loopAbody X N B).drop 3) := by
  exact cons_related (assign_related "inp" (by simp [Shared, Scratch]) (load_none_related B))
    (online_update_related X N B)

theorem input_tail_suffix_related (X : RegionName) (N B : Nat) :
    StatementsRelated B ((loopBbody X N B).drop 4) ((loopBbody X N B).drop 4) := by
  exact cons_related (assign_related "inp" (by simp [Shared, Scratch]) (load_tail_related B))
    (online_update_related X N B)

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxOptimize.SignedTransport
