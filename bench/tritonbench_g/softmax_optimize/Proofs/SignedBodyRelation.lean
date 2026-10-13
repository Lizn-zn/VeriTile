import bench.tritonbench_g.softmax_optimize.Proofs.SignedPrefixRelation
import bench.tritonbench_g.softmax_optimize.Proofs.SignedInput
import bench.tritonbench_g.softmax_optimize.Proofs.SignedMask
import bench.tritonbench_g.softmax_optimize.Proofs.SignedOutput
import bench.tritonbench_g.softmax_optimize.Proofs.SignedStore
import VeriTile.Triton.Semantics.ProgramRewrite

namespace VeriTile.Bench.TritonBenchG.SoftmaxOptimize.SignedTransport
open VeriTile.Triton

theorem append_related {B : Nat} {a b c d : List Stmt}
    (h₁ : StatementsRelated B a b) (h₂ : StatementsRelated B c d) :
    StatementsRelated B (a ++ c) (b ++ d) := by
  intro s t h
  rw [ProgramRewrite.append, ProgramRewrite.append]
  exact results_bind (h₁ s t h) _ _ h₂

theorem results_live {B : Nat} {s t : Option BlockState}
    (h : ResultsRelated (IndexedRelated B) s t) : ResultsRelated LiveRelated s t := by
  cases s <;> cases t <;> simp only [ResultsRelated] at h ⊢
  exact h.toLiveRelated

theorem prefix_suffix_related (X : RegionName) (N B i : Nat)
    {left right : List Stmt} (hsuffix : StatementsRelated B left right)
    (s t : BlockState) (h : LiveRelated s t) : ResultsRelated LiveRelated
      (stepStmts (indexPrefix X N B ++ left) (s.setReg "start_n" .int [] (Tile.scalar (i:Int))))
      (stepStmts (naturalPrefix X N B ++ right) (t.setReg "start_n" .nat [] (Tile.scalar i))) := by
  rw [ProgramRewrite.append, ProgramRewrite.append]
  exact results_live (results_bind (prefix_related X N B i s t h) _ _ hsuffix)

theorem input_full_related (X : RegionName) (N B i : Nat)
    (s t : BlockState) (h : LiveRelated s t) : ResultsRelated LiveRelated
      (stepStmts (inputFull X N B) (s.setReg "start_n" .int [] (Tile.scalar (i:Int))))
      (stepStmts (loopAbody X N B) (t.setReg "start_n" .nat [] (Tile.scalar i))) := by
  have hh := prefix_suffix_related X N B i (input_full_suffix_related X N B) s t h
  simpa only [inputFull, naturalPrefix, List.take_append_drop] using hh

theorem input_tail_related (X : RegionName) (N B i : Nat)
    (s t : BlockState) (h : LiveRelated s t) : ResultsRelated LiveRelated
      (stepStmts (inputTail X N B) (s.setReg "start_n" .int [] (Tile.scalar (i:Int))))
      (stepStmts (loopBbody X N B) (t.setReg "start_n" .nat [] (Tile.scalar i))) := by
  have hh := prefix_suffix_related X N B i
    (cons_related (mask_related N B) (input_tail_suffix_related X N B)) s t h
  simpa only [inputTail, List.append_assoc, List.singleton_append, naturalPrefix,
    naturalMask, loopAbody, loopBbody, List.take_succ_cons, List.take_zero,
    List.drop_succ_cons, List.drop_zero, List.cons_append, List.nil_append] using hh


theorem output_full_related (Y X : RegionName) (N B i : Nat)
    (s t : BlockState) (h : LiveRelated s t) : ResultsRelated LiveRelated
      (stepStmts (outputFull Y X N B) (s.setReg "start_n" .int [] (Tile.scalar (i:Int))))
      (stepStmts (loopApass2body Y X N B) (t.setReg "start_n" .nat [] (Tile.scalar i))) := by
  have hsuffix := append_related (output_full_arithmetic_related Y X N B)
    (cons_related (output_pointer_related Y B)
      (cons_related (store_full_related B) (nil_related B)))
  have hh := prefix_suffix_related X N B i hsuffix s t h
  simpa only [outputFull, List.append_assoc, List.singleton_append, naturalPrefix,
    loopAbody, loopApass2body, List.take_succ_cons, List.take_zero,
    List.drop_succ_cons, List.drop_zero, List.cons_append, List.nil_append] using hh

theorem output_tail_related (Y X : RegionName) (N B i : Nat)
    (s t : BlockState) (h : LiveRelated s t) : ResultsRelated LiveRelated
      (stepStmts (outputTail Y X N B) (s.setReg "start_n" .int [] (Tile.scalar (i:Int))))
      (stepStmts (loopBpass2body Y X N B) (t.setReg "start_n" .nat [] (Tile.scalar i))) := by
  have hsuffix := cons_related (mask_related N B)
    (append_related (output_tail_arithmetic_related Y X N B)
      (cons_related (output_pointer_related Y B)
        (cons_related (store_tail_related B) (nil_related B))))
  have hh := prefix_suffix_related X N B i hsuffix s t h
  simpa only [outputTail, List.append_assoc, List.singleton_append, naturalPrefix,
    naturalMask, loopAbody, loopBpass2body, List.take_succ_cons, List.take_zero,
    List.drop_succ_cons, List.drop_zero, List.cons_append, List.nil_append] using hh

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxOptimize.SignedTransport
