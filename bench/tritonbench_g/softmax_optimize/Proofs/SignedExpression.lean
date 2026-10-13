import bench.tritonbench_g.softmax_optimize.Proofs.SignedIndexed

namespace VeriTile.Bench.TritonBenchG.SoftmaxOptimize.SignedTransport
open VeriTile.Triton

def ExprRelated (B : Nat) {dtype : TileDType} {shape : TileShape}
    (a b : Op dtype shape) : Prop :=
  ∀ s t, IndexedRelated B s t → evalOp a s = evalOp b t

theorem ref_related (B : Nat) (dtype : TileDType) (shape : TileShape)
    (name : RegName) (hn : Shared name) :
    ExprRelated B (.ref dtype shape name) (.ref dtype shape name) := by
  intro s t h
  simpa only [evalOp_ref] using h.regs_eq dtype shape name hn

theorem add_related {B : Nat} {dtype : TileDType} {sa sb out : TileShape}
    (num : NumericDType dtype) (bc : Broadcast sa sb out)
    {a a' : Op dtype sa} {b b' : Op dtype sb}
    (ha : ExprRelated B a a') (hb : ExprRelated B b b') :
    ExprRelated B (.add num bc a b) (.add num bc a' b') := by
  intro s t h
  simp only [evalOp_add, ha s t h, hb s t h]

theorem sub_related {B : Nat} {dtype : TileDType} {sa sb out : TileShape}
    (num : NumericDType dtype) (bc : Broadcast sa sb out)
    {a a' : Op dtype sa} {b b' : Op dtype sb}
    (ha : ExprRelated B a a') (hb : ExprRelated B b b') :
    ExprRelated B (.sub num bc a b) (.sub num bc a' b') := by
  intro s t h
  simp only [evalOp_sub, ha s t h, hb s t h]

theorem mul_related {B : Nat} {dtype : TileDType} {sa sb out : TileShape}
    (num : NumericDType dtype) (bc : Broadcast sa sb out)
    {a a' : Op dtype sa} {b b' : Op dtype sb}
    (ha : ExprRelated B a a') (hb : ExprRelated B b b') :
    ExprRelated B (.mul num bc a b) (.mul num bc a' b') := by
  intro s t h
  simp only [evalOp_mul, ha s t h, hb s t h]

theorem div_related {B : Nat} {dtype : TileDType} {sa sb out : TileShape}
    (num : NumericDType dtype) (bc : Broadcast sa sb out)
    {a a' : Op dtype sa} {b b' : Op dtype sb}
    (ha : ExprRelated B a a') (hb : ExprRelated B b b') :
    ExprRelated B (.div num bc a b) (.div num bc a' b') := by
  intro s t h
  simp only [evalOp_div, ha s t h, hb s t h]

theorem gt_related {B : Nat} {dtype : TileDType} {sa sb out : TileShape}
    (cmp : ComparableDType dtype) (bc : Broadcast sa sb out)
    {a a' : Op dtype sa} {b b' : Op dtype sb}
    (ha : ExprRelated B a a') (hb : ExprRelated B b b') :
    ExprRelated B (.gt cmp bc a b) (.gt cmp bc a' b') := by
  intro s t h
  simp only [evalOp_gt, ha s t h, hb s t h]

theorem exp_related {B : Nat} {shape : TileShape} {a b : Op .real shape}
    (h : ExprRelated B a b) : ExprRelated B a.exp b.exp := by
  intro s t hr
  simp only [evalOp_exp, h s t hr]

theorem where_related {B : Nat} {dtype : TileDType} {shape : TileShape}
    {c c' : Op .bool shape} {a a' b b' : Op dtype shape}
    (hc : ExprRelated B c c') (ha : ExprRelated B a a') (hb : ExprRelated B b b') :
    ExprRelated B (c.where a b) (c'.where a' b') := by
  intro s t h
  simp only [evalOp_where, hc s t h, ha s t h, hb s t h]

def StatementsRelated (B : Nat) (left right : List Stmt) : Prop :=
  ∀ s t, IndexedRelated B s t → ResultsRelated (IndexedRelated B)
    (stepStmts left s) (stepStmts right t)

theorem nil_related (B : Nat) : StatementsRelated B [] [] := by
  intro s t h
  simpa only [stepStmts.nil, ResultsRelated] using h

theorem assign_related {B : Nat} {dtype : TileDType} {shape : TileShape}
    (name : RegName) (hn : Shared name) {a b : Op dtype shape}
    (hab : ExprRelated B a b) : ∀ s t, IndexedRelated B s t →
    ResultsRelated (IndexedRelated B)
      (stepStmt (.assign dtype shape name a) s) (stepStmt (.assign dtype shape name b) t) := by
  intro s t h
  simp only [stepStmt, hab s t h]
  cases hv : evalOp b t with
  | none => exact True.intro
  | some v => exact h.setShared name dtype shape v hn

theorem cons_related {B : Nat} {a b : Stmt} {left right : List Stmt}
    (hab : ∀ s t, IndexedRelated B s t → ResultsRelated (IndexedRelated B)
      (stepStmt a s) (stepStmt b t))
    (hrest : StatementsRelated B left right) :
    StatementsRelated B (a :: left) (b :: right) := by
  intro s t h
  have hh := results_bind (hab s t h) (stepStmts left) (stepStmts right) hrest
  cases hs : stepStmt a s <;> cases ht : stepStmt b t <;>
    simpa only [stepStmts, hs, ht, Option.bind] using hh

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxOptimize.SignedTransport
