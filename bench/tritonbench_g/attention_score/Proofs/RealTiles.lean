import bench.tritonbench_g.attention_score.Proofs.RealExpressions

/-! Relate the real tile arithmetic to the independent input formula.
These lemmas do not assume an output equation; the Q/K tile values incorporate
the source's two independent block-load checks. The memory-load and loop
invariants must still establish these tiles at each iteration. -/
namespace VeriTile.Bench.TritonBenchG.AttentionScore.RealTiles
open VeriTile.Triton RealFormula
noncomputable section
set_option maxHeartbeats 500000

def query {D : Nat} (xs : Inputs D) (B N startM : Nat) (evenM : Bool) : Tile .real [B,D] :=
  ⟨fun i => some (if evenM || decide (startM + i.1.val < N)
    then xs.query (startM + i.1.val) i.2.1 else 0)⟩
def key {D : Nat} (xs : Inputs D) (B NK startN : Nat) (evenN : Bool) : Tile .real [D,B] :=
  ⟨fun i => some (if evenN || decide (startN * B + i.2.1.val < NK)
    then xs.key (startN * B + i.2.1.val) i.1 else 0)⟩
def normalizer {D : Nat} (xs : Inputs D) (B startM : Nat) : Tile .real [B] :=
  ⟨fun i => some (xs.normalizer (startM + i.1.val))⟩
def zero (sh : TileShape) : Tile .real sh := ⟨fun _ => some 0⟩

def rawProbability {D : Nat} (xs : Inputs D) (scale : ℝ)
    (B N NK startM startN : Nat) (evenM evenN : Bool) : Tile .real [B,B] :=
  Tile.uop WithBot.realExp2
    (Tile.bop NumericDType.real.sub Broadcast.nil.consR.consSame
      (Tile.bop NumericDType.real.mul Broadcast.scalarR
        (Tile.bop NumericDType.real.add Broadcast.nil.consSame.consSame
          (zero [B,B]) (Tile.dot [] (query xs B N startM evenM) (key xs B NK startN evenN)))
        (Tile.scalar (some (scale * 1.4426950408889634))))
      (Tile.expandDim ⟨1, by simp⟩ (normalizer xs B startM)))

theorem raw_probability_cell {D : Nat} (xs : Inputs D) (scale : ℝ)
    (B N NK startM startN : Nat) (evenM evenN : Bool) (r k : Fin B) :
    (rawProbability xs scale B N NK startM startN evenM evenN).data (r,k,PUnit.unit) =
      some (pow2 (score xs scale N NK evenM evenN (startM+r.val) (startN*B+k.val)
        - xs.normalizer (startM+r.val))) := by
  have hdot : (Tile.dot [] (query xs B N startM evenM)
      (key xs B NK startN evenN)).data (r,k,PUnit.unit) =
      some (∑ d : Fin D,
        (if evenM || decide (startM+r.val < N) then xs.query (startM+r.val) d else 0) *
        (if evenN || decide (startN*B+k.val < NK) then xs.key (startN*B+k.val) d else 0)) := by
    rw [Tile.dot_nil_data]
    simp only [query, key, Option.map₂, Option.bind_some, Option.map_some]
    exact (WithBot.coe_sum _ _).symm
  simp only [rawProbability, Tile.uop_data, Tile.bop_data, Tile.expandDim_data,
    Broadcast.leftIndex, Broadcast.rightIndex, TileShape.dropInsertedIndex,
    hdot, normalizer, zero, Tile.scalar, NumericDType.add, NumericDType.mul,
    NumericDType.sub, WithBot.realAdd, WithBot.realMul, WithBot.realSub,
    Option.map₂, Option.bind_some, Option.map_some, WithBot.realExp2_some,
    zero_add, score]
  simp [pow2, mul_comm]

def windowMask (B startM startN : Nat) (offset size : Int) (complement : Bool) : Tile .bool [B,B] :=
  ⟨fun i => decide (if complement then size ≤ distance (startM+i.1.val) (startN*B+i.2.1.val) offset
    else 0 ≤ distance (startM+i.1.val) (startN*B+i.2.1.val) offset ∧
      distance (startM+i.1.val) (startN*B+i.2.1.val) offset < size)⟩

def queryMask (B N startM : Nat) : Tile .bool [B,B] :=
  ⟨fun i => decide (startM + i.1.val < N)⟩

def windowed {D : Nat} (xs : Inputs D) (scale : ℝ)
    (B N NK startM startN : Nat) (window complement evenM evenN : Bool)
    (offset size : Int) : Tile .real [B,B] :=
  let raw := rawProbability xs scale B N NK startM startN evenM evenN
  if window then Tile.select (windowMask B startM startN offset size complement) raw (zero [B,B])
  else raw

def contributions {D : Nat} (xs : Inputs D) (scale : ℝ)
    (B N NK startM startN : Nat) (window complement evenM evenN : Bool)
    (offset size : Int) : Tile .real [B,B] :=
  let p := windowed xs scale B N NK startM startN window complement evenM evenN offset size
  if evenN then p else Tile.select (queryMask B N startM) p (zero [B,B])

theorem contribution_cell {D : Nat} (xs : Inputs D) (scale : ℝ)
    (B N NK startM startN : Nat) (window complement evenM evenN : Bool)
    (offset size : Int) (r k : Fin B) :
    (contributions xs scale B N NK startM startN window complement evenM evenN offset size).data
      (r,k,PUnit.unit) = some (weight xs scale N NK window complement evenM evenN offset size
        (startM+r.val) (startN*B+k.val)) := by
  have hp := raw_probability_cell xs scale B N NK startM startN evenM evenN r k
  cases window <;> cases complement <;> cases evenN
  all_goals simp [contributions, windowed, queryMask, windowMask, Tile.select,
    zero, weight, windowAllows, hp]
  all_goals split_ifs <;> simp_all

theorem column_sum {D : Nat} (xs : Inputs D) (scale : ℝ)
    (B N NK startM startN : Nat) (window complement evenM evenN : Bool)
    (offset size : Int) (k : Fin B) :
    (Tile.reduceSumDrop ⟨0, by simp⟩
      (contributions xs scale B N NK startM startN window complement evenM evenN offset size)).data
        (k,PUnit.unit) = some (∑ r : Fin B,
          weight xs scale N NK window complement evenM evenN offset size
            (startM+r.val) (startN*B+k.val)) := by
  change (@Finset.sum (Fin B) (WithBot ℝ) _ Finset.univ (fun r => (contributions xs scale B N NK startM startN window complement evenM evenN offset size).data
    (r,k,PUnit.unit))) = _
  calc
    _ = @Finset.sum (Fin B) (WithBot ℝ) _ Finset.univ (fun r => (some (weight xs scale N NK window complement evenM evenN offset size
        (startM+r.val) (startN*B+k.val)) : WithBot ℝ)) := by
      apply Finset.sum_congr rfl
      intro r _
      exact contribution_cell xs scale B N NK startM startN window complement evenM evenN offset size r k
    _ = _ := (WithBot.coe_sum _ _).symm

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttentionScore.RealTiles
