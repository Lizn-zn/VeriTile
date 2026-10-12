import bench.tritonbench_g.attn_fwd_causal.Proofs.RealFormula
import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealTiles

/-! Real operands and row updates for the finite-penalty helper. Every lane
contributes to the normalization; K/V load masks zero inputs, not weights.
Only mathematical tile algebra is reused from the general online formulas. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealTiles
open VeriTile.Triton
open AttentionFwdTriton3.RealTiles (finite dot_finite pairs)
open AttentionFwdTriton3.RealFormula (summarize)
noncomputable section

variable {BM BN BD : Nat}

def query (xs : RealFormula.Inputs BM BN BD) (startM N HA : Nat) : Tile .real [BM,BD] :=
  finite (fun i => if startM*BM+i.1.val < N ∧ i.2.1.val < HA then xs.query i.1 i.2.1 else 0)

def key (xs : RealFormula.Inputs BM BN BD) (startN N HA : Nat) : Tile .real [BD,BN] :=
  finite (fun i => if startN+i.2.1.val < N ∧ i.1.val < HA then xs.key startN i.2.1 i.1 else 0)

def value (xs : RealFormula.Inputs BM BN BD) (v : RealFormula.Visit) (N HA : Nat) : Tile .real [BN,BD] :=
  finite (fun i => RealFormula.loadedValue xs N HA v i.1 i.2.1)

def rawScores (xs : RealFormula.Inputs BM BN BD) (startM N HA : Nat) (v : RealFormula.Visit) :
    Tile .real [BM,BN] :=
  Tile.bop NumericDType.real.mul .scalarR
    (Tile.bop NumericDType.real.mul .scalarR
      (Tile.dot [] (query xs startM N HA) (key xs v.start N HA))
      (Tile.scalar (some xs.queryScale)))
    (Tile.scalar (some (xs.keyScale (v.start / BN))))

theorem raw_scores_formula (xs : RealFormula.Inputs BM BN BD) (startM N HA : Nat)
    (v : RealFormula.Visit) : rawScores xs startM N HA v =
      finite (fun i : TileIndex [BM,BN] => RealFormula.score xs startM N HA v i.1 i.2.1) := by
  simp only [rawScores, query, key]
  erw [dot_finite BM BN BD]
  rfl

def logits (xs : RealFormula.Inputs BM BN BD) (startM N HA : Nat) (v : RealFormula.Visit) :
    Tile .real [BM,BN] := finite (fun i => RealFormula.logit xs startM N HA v i.1 i.2.1)

/-- Unlike exact causal masking, all visited scores are kept, including the
finite-penalty and zero-padded lanes. -/
def keep (_ : Fin BM) (_ : Fin BN) : Bool := Bool.true

theorem keep_eq : @keep BM BN = (fun _ _ => Bool.true) := rfl

def center (score : Fin BM → Fin BN → ℝ) (old : Fin BM → WithBot ℝ) : Fin BM → WithBot ℝ :=
  AttentionFwdTriton3.RealTiles.center score keep old

def probability (score : Fin BM → Fin BN → ℝ) (old : Fin BM → WithBot ℝ) :
    Fin BM → Fin BN → ℝ := AttentionFwdTriton3.RealTiles.probability score keep old

theorem all_pairs (score values : Fin BN → ℝ) :
    pairs score values (fun _ => Bool.true) = (List.finRange BN).map (fun j => (score j,values j)) := by
  simp [pairs]

theorem input_pairs (xs : RealFormula.Inputs BM BN BD) (startM N HA : Nat)
    (v : RealFormula.Visit) (row : Fin BM) (d : Fin BD) :
    pairs (RealFormula.logit xs startM N HA v row) (fun j => RealFormula.loadedValue xs N HA v j d)
      (keep row) = RealFormula.blockPairs xs startM N HA v row d := by
  simp [pairs, keep_eq, RealFormula.blockPairs]

theorem center_matches (score : Fin BM → Fin BN → ℝ) (old : Fin BM → WithBot ℝ)
    (row : Fin BM) (l a : ℝ) (values : Fin BN → ℝ) :
    center score old row =
      (summarize ⟨old row,l,a⟩ ((List.finRange BN).map (fun j => (score row j,values j)))).center := by
  simpa only [center, keep_eq, all_pairs] using
    AttentionFwdTriton3.RealTiles.center_matches_summary score keep old row l a values

theorem reduce_max (hN : 0 < BN) (score : Fin BM → Fin BN → ℝ) :
    Tile.reduceMaxDrop ⟨1, by simp⟩ (finite (fun i : TileIndex [BM,BN] => score i.1 i.2.1)) =
    some (⟨fun i => AttentionFwdTriton3.RealFormula.maxScore
      ((List.finRange BN).map (fun j => (score i.1 j, (0 : ℝ))))⟩ : Tile .real [BM]) := by
  simpa only [keep_eq, Bool.true_eq, ↓reduceIte, all_pairs] using
    AttentionFwdTriton3.RealTiles.reduce_max_masked BM BN hN score (@keep BM BN)

theorem probabilities_match (score : Fin BM → Fin BN → ℝ) (old : Fin BM → WithBot ℝ) :
    Tile.uop WithBot.realExp2
      (Tile.bop NumericDType.real.sub (.consSame (.consR .nil))
        (finite (fun i : TileIndex [BM,BN] => score i.1 i.2.1))
        (Tile.expandDim ⟨1, by simp⟩ (⟨fun i => center score old i.1⟩ : Tile .real [BM]))) =
      finite (fun i : TileIndex [BM,BN] => probability score old i.1 i.2.1) := by
  simpa only [center, probability, keep, Bool.true_eq, ↓reduceIte] using
    AttentionFwdTriton3.RealTiles.probabilities_match BM BN score keep old

theorem alpha_matches (score : Fin BM → Fin BN → ℝ) (old : Fin BM → WithBot ℝ) :
    Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (.consSame .nil)
      (⟨fun i => old i.1⟩ : Tile .real [BM]) (⟨fun i => center score old i.1⟩ : Tile .real [BM])) =
      finite (fun i : TileIndex [BM] => AttentionFwdTriton3.RealFormula.weight (old i.1) (center score old i.1)) :=
  AttentionFwdTriton3.RealTiles.alpha_matches BM BN score keep old

theorem denominator_matches (xs : RealFormula.Inputs BM BN BD) (startM N HA : Nat)
    (v : RealFormula.Visit) (old : Fin BM → WithBot ℝ) (l : Fin BM → ℝ)
    (row : Fin BM) (d : Fin BD) (a : ℝ) :
    let score := RealFormula.logit xs startM N HA v
    (Tile.bop NumericDType.real.add (.consSame .nil)
      (Tile.bop NumericDType.real.mul (.consSame .nil)
        (finite (fun i : TileIndex [BM] => l i.1))
        (finite (fun i : TileIndex [BM] => AttentionFwdTriton3.RealFormula.weight (old i.1) (center score old i.1))))
      (Tile.reduceSumDrop ⟨1, by simp⟩
        (finite (fun i : TileIndex [BM,BN] => probability score old i.1 i.2.1)))).data (row,PUnit.unit) =
    some (summarize ⟨old row,l row,a⟩ (RealFormula.blockPairs xs startM N HA v row d)).denominator := by
  simpa only [center, probability, input_pairs] using
    AttentionFwdTriton3.RealTiles.denominator_matches BM BN (RealFormula.logit xs startM N HA v) keep old l row
      (fun j => RealFormula.loadedValue xs N HA v j d) a

theorem numerator_matches (xs : RealFormula.Inputs BM BN BD) (startM N HA : Nat)
    (v : RealFormula.Visit) (old : Fin BM → WithBot ℝ) (acc : TileIndex [BM,BD] → ℝ)
    (row : Fin BM) (d : Fin BD) (l : ℝ) :
    let score := RealFormula.logit xs startM N HA v
    (Tile.bop NumericDType.real.add (.consSame (.consSame .nil))
      (Tile.bop NumericDType.real.mul (.consSame (.consR .nil)) (finite acc)
        (Tile.expandDim ⟨1, by simp⟩
          (finite (fun i : TileIndex [BM] => AttentionFwdTriton3.RealFormula.weight (old i.1) (center score old i.1)))))
      (Tile.dot [] (finite (fun i : TileIndex [BM,BN] => probability score old i.1 i.2.1))
        (value xs v N HA))).data (row,d,PUnit.unit) =
      some (summarize ⟨old row,l,acc (row,d,PUnit.unit)⟩ (RealFormula.blockPairs xs startM N HA v row d)).numerator := by
  simpa only [center, probability, input_pairs, value] using
    AttentionFwdTriton3.RealTiles.numerator_matches BM BN BD (RealFormula.logit xs startM N HA v) keep old acc
      (fun i : TileIndex [BN,BD] => RealFormula.loadedValue xs N HA v i.1 i.2.1) row d l

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealTiles
