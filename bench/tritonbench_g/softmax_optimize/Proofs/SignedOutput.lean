import bench.tritonbench_g.softmax_optimize.Proofs.SignedLoad
import bench.tritonbench_g.softmax_optimize.Proofs.SignedBodies

namespace VeriTile.Bench.TritonBenchG.SoftmaxOptimize.SignedTransport
open VeriTile.Triton

/-- The two writeback arithmetic assignments read only related floating values. -/
theorem output_arithmetic_related (Y X : RegionName) (N B : Nat) :
    StatementsRelated B (((loopApass2body Y X N B).drop 4).take 2)
      (((loopApass2body Y X N B).drop 4).take 2) := by
  have hi := ref_related B .real [B] "inp" (by simp [Shared, Scratch])
  have hm := ref_related B .real [] "m" (by simp [Shared, Scratch])
  have he := ref_related B .real [B] "e" (by simp [Shared, Scratch])
  have hz := ref_related B .real [] "z" (by simp [Shared, Scratch])
  exact cons_related (assign_related "e" (by simp [Shared, Scratch])
    (exp_related (sub_related .real .scalarR hi hm)))
    (cons_related (assign_related "out" (by simp [Shared, Scratch])
      (div_related .real .scalarR he hz)) (nil_related B))

theorem output_full_arithmetic_related (Y X : RegionName) (N B : Nat) :
    StatementsRelated B (((loopApass2body Y X N B).drop 3).take 3)
      (((loopApass2body Y X N B).drop 3).take 3) :=
  cons_related (assign_related "inp" (by simp [Shared, Scratch]) (load_none_related B))
    (output_arithmetic_related Y X N B)

theorem output_tail_arithmetic_related (Y X : RegionName) (N B : Nat) :
    StatementsRelated B (((loopBpass2body Y X N B).drop 4).take 3)
      (((loopBpass2body Y X N B).drop 4).take 3) :=
  cons_related (assign_related "inp" (by simp [Shared, Scratch]) (load_tail_related B))
    (output_arithmetic_related Y X N B)

theorem output_pointer_expression_related (Y : RegionName) (B : Nat) : ExprRelated B
    (.ptrAdd .scalarL (.ptrBase Y) (.castIntToNat (.ref .int [B] "offset")))
    (.ptrAdd .scalarL (.ptrBase Y) (.ref .nat [B] "offset")) := by
  intro s t h
  simp only [evalOp_ptrAdd, evalOp_ptrBase, h.offset_eval]

theorem output_pointer_related (Y : RegionName) (B : Nat) : ∀ s t,
    IndexedRelated B s t → ResultsRelated (IndexedRelated B)
      (stepStmt (outputPointer Y B) s)
      (stepStmt (.assign .ptr [B] "output_ptrs"
        (.ptrAdd .scalarL (.ptrBase Y) (.ref .nat [B] "offset"))) t) :=
  assign_related "output_ptrs" (by simp [Shared, Scratch]) (output_pointer_expression_related Y B)

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxOptimize.SignedTransport
