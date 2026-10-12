import bench.tritonbench_g.attention_score.Proofs.RealLoop

/-! Finite mathematical input arrays for one key block and all visited query
rows. The independent formula keeps the source's query/key load masks, signed
window, key-even-controlled query selection and supplied M normalization.
Input pinning is conditional only for the checked Q/K accesses. -/
namespace VeriTile.Bench.TritonBenchG.AttentionScore.RealInputs
open VeriTile.Triton
noncomputable section

structure Inputs (rows B D : Nat) where
  query : Fin rows → Fin D → ℝ
  key : Fin B → Fin D → ℝ
  normalizer : Fin rows → ℝ

def rowIndex {blocks B : Nat} (block : Fin blocks) (row : Fin B) : Fin (blocks*B) :=
  ⟨block.val*B+row.val, by
    have h := Nat.mul_le_mul_right B (Nat.succ_le_of_lt block.isLt)
    have hr := row.isLt
    simp only [Nat.succ_mul] at h
    omega⟩

def score {rows B D : Nat} (xs : Inputs rows B D) (N NK startN : Nat) (scale : ℝ)
    (evenM evenN : Bool) (row : Fin rows) (key : Fin B) : ℝ :=
  (∑ d : Fin D,
    (if evenM = Bool.true ∨ row.val < N then xs.query row d else 0) *
    (if evenN = Bool.true ∨ startN*B+key.val < NK then xs.key key d else 0)) *
      (scale * 1.4426950408889634)

def weight {rows B D : Nat} (xs : Inputs rows B D) (N NK startN : Nat) (scale : ℝ)
    (window complement evenM evenN : Bool) (offset size : Int) (row : Fin rows) (key : Fin B) : ℝ :=
  if RealFormula.windowAllows window complement offset size row.val (startN*B+key.val) ∧
      (evenN = Bool.true ∨ row.val < N) then
    pow2 (score xs N NK startN scale evenM evenN row key - xs.normalizer row)
  else 0

def column {blocks B D : Nat} (xs : Inputs (blocks*B) B D) (N NK startN : Nat) (scale : ℝ)
    (window complement evenM evenN : Bool) (offset size : Int) (key : Fin B) : ℝ :=
  ∑ block : Fin blocks, ∑ row : Fin B,
    weight xs N NK startN scale window complement evenM evenN offset size (rowIndex block row) key

structure Pins (c : RealLoop.Config) (s : BlockState) (rows : Nat) (xs : Inputs rows c.B c.D) : Prop where
  query : ∀ row d, (c.evenM = Bool.true ∨ row.val < c.N) →
    s.readMem c.Q (c.baseQ+row.val*c.sqm+d.val*c.sqk) = xs.query row d
  key : ∀ key d, (c.evenN = Bool.true ∨ c.startN*c.B+key.val < c.NK) →
    s.readMem c.K (c.baseK+d.val*c.skk+(c.startN*c.B+key.val)*c.skn) = xs.key key d
  normalizer : ∀ row, s.readMem c.M (c.baseM+row.val) = xs.normalizer row

theorem score_pinned (c : RealLoop.Config) (s : BlockState) (rows : Nat)
    (xs : Inputs rows c.B c.D) (hp : Pins c s rows xs) (row : Fin rows) (key : Fin c.B) :
    RealFormula.score (c.inputs s) c.scale c.N c.NK c.evenM c.evenN row.val (c.startN*c.B+key.val) =
      score xs c.N c.NK c.startN c.scale c.evenM c.evenN row key := by
  unfold RealFormula.score score
  congr 1
  apply Finset.sum_congr rfl
  intro d _
  simp only [Bool.or_eq_true, decide_eq_true_eq]
  by_cases hq : c.evenM = Bool.true ∨ row.val < c.N
  · rw [if_pos hq, if_pos hq]
    by_cases hk : c.evenN = Bool.true ∨ c.startN*c.B+key.val < c.NK
    · rw [if_pos hk, if_pos hk]
      exact congrArg₂ (· * ·) (hp.query row d hq) (hp.key key d hk)
    · rw [if_neg hk, if_neg hk]; simp
  · rw [if_neg hq, if_neg hq]; simp

theorem weight_pinned (c : RealLoop.Config) (s : BlockState) (rows : Nat)
    (xs : Inputs rows c.B c.D) (hp : Pins c s rows xs) (row : Fin rows) (key : Fin c.B) :
    RealFormula.weight (c.inputs s) c.scale c.N c.NK c.window c.complement c.evenM c.evenN
      c.offset c.size row.val (c.startN*c.B+key.val) =
    weight xs c.N c.NK c.startN c.scale c.window c.complement c.evenM c.evenN c.offset c.size row key := by
  unfold RealFormula.weight weight
  rw [score_pinned c s rows xs hp row key]
  have hm : (c.inputs s).normalizer row.val = xs.normalizer row := hp.normalizer row
  rw [hm]

theorem column_pinned (c : RealLoop.Config) (s : BlockState) (blocks : Nat)
    (xs : Inputs (blocks*c.B) c.B c.D) (hp : Pins c s (blocks*c.B) xs) (key : Fin c.B) :
    c.column s blocks key = column xs c.N c.NK c.startN c.scale
      c.window c.complement c.evenM c.evenN c.offset c.size key := by
  unfold RealLoop.Config.column RealFormula.column column
  apply Finset.sum_congr rfl
  intro block _
  apply Finset.sum_congr rfl
  intro row _
  exact weight_pinned c s (blocks*c.B) xs hp (rowIndex block row) key

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttentionScore.RealInputs
