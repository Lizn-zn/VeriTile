import bench.tritonbench_g.attn_fwd_causal.Proofs.RealLoop

/-! Match the executed helper's closed carry with the independent list of
visited input pairs. Two ranges compose in source order, retaining repeated
diagonal visits. This is the arithmetic bridge; prologue, pointer reset and
stores still have to connect the two complete source loops. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealLoopFormula
open VeriTile.Triton RealLoads RealLoop
open AttentionFwdTriton3.RealLoopFormula (Carry rowSeed closed_row)
open AttentionFwdTriton3.RealFormula (RowState result block_updates_eq_closed)
noncomputable section

theorem visited_blocks {BM BN BD : Nat} (xs : RealFormula.Inputs BM BN BD) (c : Config)
    (lower upper : Nat) (second firstDiagonal : Bool) (row : Fin BM) (lane : Fin BD) :
    AttentionFwdTriton3.RealLoopFormula.blocks
      (fun i => score xs c lower second firstDiagonal i row) (fun _ => RealTiles.keep row)
      (fun i j => value xs c lower second firstDiagonal i (j,lane,PUnit.unit))
      (RealFormula.tripCount lower upper BN) =
    (RealFormula.visits lower upper BN (second || firstDiagonal)).map
      (fun v => RealFormula.blockPairs xs c.startM c.context c.activeHead v row lane) := by
  simp [AttentionFwdTriton3.RealLoopFormula.blocks, AttentionFwdTriton3.RealTiles.pairs,
    RealTiles.keep_eq, RealFormula.visits, RealFormula.blockPairs, score, value, cursor,
    List.map_map, Function.comp_def]

theorem after_row {BM BN BD : Nat} (xs : RealFormula.Inputs BM BN BD) (c : Config)
    (lower upper : Nat) (second firstDiagonal : Bool) (seed : Carry BM BD) (row : Fin BM) (lane : Fin BD) :
    let final := after xs c lower second firstDiagonal seed (RealFormula.tripCount lower upper BN)
    rowSeed final row (final.numerator (row,lane,PUnit.unit)) =
      result (rowSeed seed row (seed.numerator (row,lane,PUnit.unit)))
        ((RealFormula.visits lower upper BN (second || firstDiagonal)).map
          (fun v => RealFormula.blockPairs xs c.startM c.context c.activeHead v row lane)) := by
  dsimp only
  rw [after, closed_row, visited_blocks]

theorem result_append (seed : RowState) (a b : List (List (ℝ × ℝ))) :
    result (result seed a) b = result seed (a++b) := by
  simp only [← block_updates_eq_closed, List.foldl_append]

def initial (BM BD : Nat) : Carry BM BD := ⟨fun _ => ⊥, fun _ => 1, fun _ => 0⟩
def firstLower (startM BM : Nat) (diagonal : Bool) : Nat := if diagonal then startM*BM else 0
def firstUpper (startM BM : Nat) (diagonal : Bool) : Nat := if diagonal then (startM+1)*BM else startM*BM

def firstResult {BM BN BD : Nat} (xs : RealFormula.Inputs BM BN BD) (c : Config)
    (diagonal : Bool) : Carry BM BD :=
  after xs c (firstLower c.startM BM diagonal) Bool.false diagonal (initial BM BD)
    (RealFormula.tripCount (firstLower c.startM BM diagonal) (firstUpper c.startM BM diagonal) BN)

def bothResult {BM BN BD : Nat} (xs : RealFormula.Inputs BM BN BD) (c : Config)
    (diagonal : Bool) : Carry BM BD :=
  after xs c (c.startM*BM) Bool.true diagonal (firstResult xs c diagonal)
    (RealFormula.tripCount (c.startM*BM) ((c.startM+1)*BM) BN)

theorem both_row {BM BN BD : Nat} (xs : RealFormula.Inputs BM BN BD) (c : Config)
    (diagonal : Bool) (row : Fin BM) (lane : Fin BD) :
    rowSeed (bothResult xs c diagonal) row ((bothResult xs c diagonal).numerator (row,lane,PUnit.unit)) =
      result AttentionFwdTriton3.RealFormula.initial
        (RealFormula.blocks xs c.startM c.context c.activeHead (if diagonal then 2 else 3) row lane) := by
  rw [bothResult, after_row, firstResult, after_row, result_append]
  cases diagonal <;> simp [rowSeed, initial, firstLower, firstUpper, RealFormula.blocks,
    RealFormula.allVisits, RealFormula.firstVisits, RealFormula.secondVisits, List.map_append,
    AttentionFwdTriton3.RealFormula.initial]

/-- Both helper formulas normalize to the independent source formula for
each legal stage, with the stage-2 repetitions kept in the pair list. -/
theorem both_normalized {BM BN BD : Nat} (xs : RealFormula.Inputs BM BN BD) (c : Config)
    (diagonal : Bool) (row : Fin BM) (lane : Fin BD) :
    (bothResult xs c diagonal).numerator (row,lane,PUnit.unit) /
        (bothResult xs c diagonal).denominator row =
      RealFormula.expected xs c.startM c.context c.activeHead (if diagonal then 2 else 3) row lane := by
  have h := both_row xs c diagonal row lane
  have hn := congrArg RowState.numerator h
  have hd := congrArg RowState.denominator h
  change (bothResult xs c diagonal).numerator (row,lane,PUnit.unit) = _ at hn
  change (bothResult xs c diagonal).denominator row = _ at hd
  rw [hn, hd]
  simpa only [← block_updates_eq_closed] using
    RealFormula.online_updates_normalized xs c.startM c.context c.activeHead
      (if diagonal then 2 else 3) row lane

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealLoopFormula
