import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealTiles

/-! Independent formulas for all rows and output lanes after a prefix of
attention blocks. The formulas contain only input score/value pairs, maxima
and weighted sums. They do not evaluate statements or fold machine states.
The zero-block case preserves the seed, including an initial denominator 1. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealLoopFormula
open VeriTile.Triton RealTiles
noncomputable section

@[ext] structure Carry (BM BD : Nat) where
  center : Fin BM → WithBot ℝ
  denominator : Fin BM → ℝ
  numerator : TileIndex [BM, BD] → ℝ

def rowSeed {BM BD : Nat} (seed : Carry BM BD) (row : Fin BM) (a : ℝ) : RealFormula.RowState :=
  ⟨seed.center row, seed.denominator row, a⟩

def blocks {BN : Nat} (score : Nat → Fin BN → ℝ) (active : Nat → Fin BN → Bool)
    (value : Nat → Fin BN → ℝ) (n : Nat) : List (List (ℝ × ℝ)) :=
  (List.range n).map (fun i => pairs (score i) (value i) (active i))

theorem result_blocks_succ {BN : Nat} (seed : RealFormula.RowState)
    (score : Nat → Fin BN → ℝ) (active : Nat → Fin BN → Bool)
    (value : Nat → Fin BN → ℝ) (n : Nat) :
    RealFormula.result seed (blocks score active value (n + 1)) =
      RealFormula.summarize (RealFormula.result seed (blocks score active value n))
        (pairs (score n) (value n) (active n)) := by
  simp only [← RealFormula.block_updates_eq_closed, blocks, List.range_succ, List.map_append,
    List.map_cons, List.map_nil, List.foldl_append, List.foldl_cons, List.foldl_nil]

/-- Row centers and denominators depend on the scores, independently of the
value vector and initial numerator. -/
theorem summary_same_mass {BN : Nat} (a b : RealFormula.RowState)
    (score va vb : Fin BN → ℝ) (active : Fin BN → Bool)
    (hc : a.center = b.center) (hd : a.denominator = b.denominator) :
    (RealFormula.summarize a (pairs score va active)).center =
        (RealFormula.summarize b (pairs score vb active)).center ∧
      (RealFormula.summarize a (pairs score va active)).denominator =
        (RealFormula.summarize b (pairs score vb active)).denominator := by
  simp [RealFormula.summarize, RealFormula.sumAt, maxScore_pairs, sum_pairs, hc, hd]

theorem result_same_mass {BN : Nat} (a b : RealFormula.RowState)
    (score : Nat → Fin BN → ℝ) (active : Nat → Fin BN → Bool)
    (va vb : Nat → Fin BN → ℝ) (n : Nat)
    (hc : a.center = b.center) (hd : a.denominator = b.denominator) :
    (RealFormula.result a (blocks score active va n)).center =
        (RealFormula.result b (blocks score active vb n)).center ∧
      (RealFormula.result a (blocks score active va n)).denominator =
        (RealFormula.result b (blocks score active vb n)).denominator := by
  induction n with
  | zero => simpa [blocks, RealFormula.result] using And.intro hc hd
  | succ n ih =>
    rw [result_blocks_succ, result_blocks_succ]
    exact summary_same_mass _ _ (score n) (va n) (vb n) (active n) ih.1 ih.2

/-- Center and denominator use a dummy zero value, so their definitions also
work when BD is zero. Each numerator uses its actual output-lane values. -/
def closed {BM BN BD : Nat} (seed : Carry BM BD)
    (score : Nat → Fin BM → Fin BN → ℝ) (active : Nat → Fin BM → Fin BN → Bool)
    (value : Nat → TileIndex [BN, BD] → ℝ) (n : Nat) : Carry BM BD where
  center := fun row => (RealFormula.result (rowSeed seed row 0)
    (blocks (fun i => score i row) (fun i => active i row) (fun _ _ => 0) n)).center
  denominator := fun row => (RealFormula.result (rowSeed seed row 0)
    (blocks (fun i => score i row) (fun i => active i row) (fun _ _ => 0) n)).denominator
  numerator := fun i => (RealFormula.result (rowSeed seed i.1 (seed.numerator i))
    (blocks (fun j => score j i.1) (fun j => active j i.1)
      (fun j key => value j (key, i.2.1, PUnit.unit)) n)).numerator

def update {BM BN BD : Nat} (carry : Carry BM BD)
    (score : Fin BM → Fin BN → ℝ) (active : Fin BM → Fin BN → Bool)
    (value : TileIndex [BN, BD] → ℝ) : Carry BM BD where
  center := fun row => RealTiles.center score active carry.center row
  denominator := fun row => (RealFormula.summarize (rowSeed carry row 0)
    (pairs (score row) (fun _ => 0) (active row))).denominator
  numerator := fun i => (RealFormula.summarize (rowSeed carry i.1 (carry.numerator i))
    (pairs (score i.1) (fun key => value (key, i.2.1, PUnit.unit)) (active i.1))).numerator

theorem closed_zero {BM BN BD : Nat} (seed : Carry BM BD)
    (score : Nat → Fin BM → Fin BN → ℝ) (active : Nat → Fin BM → Fin BN → Bool)
    (value : Nat → TileIndex [BN, BD] → ℝ) : closed seed score active value 0 = seed := by
  simp [closed, blocks, RealFormula.result, rowSeed]

/-- Every output lane has the same center and denominator as the shared
row registers. This identifies the vector carry with independent scalar rows. -/
theorem closed_row {BM BN BD : Nat} (seed : Carry BM BD)
    (score : Nat → Fin BM → Fin BN → ℝ) (active : Nat → Fin BM → Fin BN → Bool)
    (value : Nat → TileIndex [BN, BD] → ℝ) (n : Nat) (row : Fin BM) (lane : Fin BD) :
    rowSeed (closed seed score active value n) row
        ((closed seed score active value n).numerator (row, lane, PUnit.unit)) =
      RealFormula.result (rowSeed seed row (seed.numerator (row, lane, PUnit.unit)))
        (blocks (fun j => score j row) (fun j => active j row)
          (fun j key => value j (key, lane, PUnit.unit)) n) := by
  have h := result_same_mass (rowSeed seed row 0)
    (rowSeed seed row (seed.numerator (row, lane, PUnit.unit)))
    (fun j => score j row) (fun j => active j row) (fun _ _ => 0)
    (fun j key => value j (key, lane, PUnit.unit)) n rfl rfl
  exact RealFormula.RowState.ext h.1 h.2 rfl

/-- One numerical block update advances the closed formula. In particular,
an empty block is retained as an executed iteration rather than discarded. -/
theorem closed_succ {BM BN BD : Nat} (seed : Carry BM BD)
    (score : Nat → Fin BM → Fin BN → ℝ) (active : Nat → Fin BM → Fin BN → Bool)
    (value : Nat → TileIndex [BN, BD] → ℝ) (n : Nat) :
    closed seed score active value (n + 1) =
      update (closed seed score active value n) (score n) (active n) (value n) := by
  apply Carry.ext
  · funext row
    simp only [closed, update, result_blocks_succ, RealTiles.center, RealFormula.summarize, rowSeed]
  · funext row
    simp only [closed, update, result_blocks_succ, RealFormula.summarize, rowSeed]
  · funext i
    rcases i with ⟨row, lane, ⟨⟩⟩
    change (RealFormula.result _ _).numerator = (RealFormula.summarize _ _).numerator
    rw [result_blocks_succ, ← closed_row seed score active value n row lane]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealLoopFormula
