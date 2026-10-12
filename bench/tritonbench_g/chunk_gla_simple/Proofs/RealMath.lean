import VeriTile.Triton
import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.chunk_gla_simple.Proofs.RealStep

/-! Independent zero-extended contraction over initial arrays. The kernel's
unspecified boundary values require an explicit zero-oracle premise in the
execution theorem; they are not silently changed in the source. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaSimple.Math
open scoped BigOperators
noncomputable section
structure Layout where
  keyHead : Nat
  keyTime : Nat
  valueHead : Nat
  valueTime : Nat
  historyHead : Nat
  historyKey : Nat
  T : Nat
  K : Nat
  V : Nat
  BT : Nat
  BK : Nat
  BV : Nat
structure Launch where
  value : Nat
  chunk : Nat
  head : Nat
structure Inputs where
  query : Nat → ℝ
  key : Nat → ℝ
  value : Nat → ℝ
  history : Nat → ℝ
  gate : Nat → ℝ

def numBlocks (p : Layout) := (p.K+p.BK-1)/p.BK
def row (p : Layout) (l : Launch) (i : Nat) := l.chunk*p.BT+i
def column (p : Layout) (l : Launch) (j : Nat) := l.value*p.BV+j
def keyAddress (p : Layout) (l : Launch) (t k : Nat) := l.head*p.keyHead+t*p.keyTime+k
def historyAddress (p : Layout) (l : Launch) (k j : Nat) :=
  l.head*p.historyHead+l.chunk*p.K*p.V+k*p.historyKey+column p l j

def outputAddress (p : Layout) (l : Launch) (i j : Nat) :=
  l.head*p.valueHead+row p l i*p.valueTime+column p l j

def gateAddress (p : Layout) (l : Launch) (i : Nat) := l.head*p.T+row p l i

def active (p : Layout) (l : Launch) (i j : Nat) : Prop := row p l i < p.T ∧ column p l j < p.V

instance (p : Layout) (l : Launch) (i j : Nat) : Decidable (active p l i j) :=
  inferInstanceAs (Decidable (_ ∧ _))

def query (p : Layout) (l : Launch) (a : Inputs) (i k : Nat) : ℝ :=
  if row p l i < p.T ∧ k < p.K then a.query (keyAddress p l (row p l i) k) else 0

def key (p : Layout) (l : Launch) (a : Inputs) (k j : Nat) : ℝ :=
  if k < p.K ∧ row p l j < p.T then a.key (l.head*p.keyHead+k+row p l j*p.keyTime) else 0

def history (p : Layout) (l : Launch) (a : Inputs) (k j : Nat) : ℝ :=
  if k < p.K ∧ column p l j < p.V then a.history (historyAddress p l k j) else 0

def value (p : Layout) (l : Launch) (a : Inputs) (i j : Nat) : ℝ :=
  if active p l i j then a.value (outputAddress p l i j) else 0

def gate (p : Layout) (l : Launch) (a : Inputs) (i : Nat) : ℝ :=
  if row p l i < p.T then a.gate (gateAddress p l i) else 0

/-- Contributions from all completed key blocks, including the checked tail. -/
def qhSum (p : Layout) (l : Launch) (a : Inputs) (n i j : Nat) : ℝ :=
  ∑ kb ∈ Finset.range n, ∑ e : Fin p.BK,
    query p l a i (kb*p.BK+e.val)*history p l a (kb*p.BK+e.val) j

def qkSum (p : Layout) (l : Launch) (a : Inputs) (n i j : Nat) : ℝ :=
  ∑ kb ∈ Finset.range n, ∑ e : Fin p.BK,
    query p l a i (kb*p.BK+e.val)*key p l a (kb*p.BK+e.val) j

def score (p : Layout) (l : Launch) (a : Inputs) (i j : Nat) : ℝ :=
  if j ≤ i then qkSum p l a (numBlocks p) i j*Real.exp (gate p l a i-gate p l a j) else 0

/-- Gated history plus the causal query/key/value contraction, then scale. -/
def expected (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i j : Nat) : ℝ :=
  (qhSum p l a (numBlocks p) i j*Real.exp (gate p l a i)+
    ∑ t : Fin p.BT, score p l a i t.val*value p l a t.val j)*scale

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaSimple.Math

namespace VeriTile.Bench.TritonBenchG.ChunkGlaSimple.Real
open VeriTile.Triton Math
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

structure Regions where
  query : RegionName
  key : RegionName
  value : RegionName
  history : RegionName
  gate : RegionName
  output : RegionName

def launch (s : BlockState) : Launch := ⟨s.pids 0,s.pids 1,s.pids 2⟩
def inputs (r : Regions) (s : BlockState) : Inputs :=
  ⟨s.readMem r.query,s.readMem r.key,s.readMem r.value,s.readMem r.history,s.readMem r.gate⟩

def queryTile (p : Layout) (l : Launch) (a : Inputs) (kb : Nat) : Tile .real [p.BT,p.BK] :=
  Tile.mat fun i e => some (query p l a i.val (kb*p.BK+e.val))
def keyTile (p : Layout) (l : Launch) (a : Inputs) (kb : Nat) : Tile .real [p.BK,p.BT] :=
  Tile.mat fun e j => some (key p l a (kb*p.BK+e.val) j.val)
def historyTile (p : Layout) (l : Launch) (a : Inputs) (kb : Nat) : Tile .real [p.BK,p.BV] :=
  Tile.mat fun e j => some (history p l a (kb*p.BK+e.val) j.val)
def carryO (p : Layout) (l : Launch) (a : Inputs) (n : Nat) : Tile .real [p.BT,p.BV] :=
  Tile.mat fun i j => some (qhSum p l a n i.val j.val)
def carryS (p : Layout) (l : Launch) (a : Inputs) (n : Nat) : Tile .real [p.BT,p.BT] :=
  Tile.mat fun i j => some (qkSum p l a n i.val j.val)
def maskTile (p : Layout) : Tile .bool [p.BT,p.BT] :=
  Tile.mat fun i j => decide (j.val ≤ i.val)

theorem query_tile (p : Layout) (r : Regions) (s : BlockState) (kb : Nat) :
    bqTile s r.query p.keyHead p.keyTime p.T p.K p.BT p.BK kb = queryTile p (launch s) (inputs r s) kb := by
  apply Tile.ext
  rintro ⟨i,e,⟨⟩⟩
  simp only [bqTile,queryTile,Tile.mat,query,keyAddress,row,launch,inputs,mul_one]
  split_ifs <;> rfl

theorem key_tile (p : Layout) (r : Regions) (s : BlockState) (kb : Nat) :
    bkTile s r.key p.keyHead p.keyTime p.T p.K p.BT p.BK kb = keyTile p (launch s) (inputs r s) kb := by
  apply Tile.ext
  rintro ⟨e,j,⟨⟩⟩
  simp only [bkTile,keyTile,Tile.mat,key,row,launch,inputs,mul_one]
  split_ifs <;> rfl

theorem history_tile (p : Layout) (r : Regions) (s : BlockState) (kb : Nat) :
    bhTile s r.history p.historyHead p.historyKey p.K p.V p.BT p.BK p.BV kb = historyTile p (launch s) (inputs r s) kb := by
  apply Tile.ext
  rintro ⟨e,j,⟨⟩⟩
  simp only [bhTile,historyTile,Tile.mat,history,historyAddress,column,launch,inputs,mul_one]
  split_ifs <;> rfl

theorem output_update (p : Layout) (l : Launch) (a : Inputs) (n : Nat) :
    Tile.bop NumericDType.real.add (.consSame (.consSame .nil))
      (carryO p l a n) (Tile.dot [] (queryTile p l a n) (historyTile p l a n)) = carryO p l a (n+1) := by
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  rw [Tile.bop_data]
  have hd := dot2d_elem (queryTile p l a n) (historyTile p l a n) i j
    (fun e => query p l a i.val (n*p.BK+e.val))
    (fun e => history p l a (n*p.BK+e.val) j.val) (fun _ => rfl) (fun _ => rfl)
  simp only [Broadcast.leftIndex,Broadcast.rightIndex,hd,carryO,Tile.mat,
    NumericDType.add,WithBot.realAdd,Option.map₂_some_some,qhSum,Finset.sum_range_succ]

theorem score_update (p : Layout) (l : Launch) (a : Inputs) (n : Nat) :
    Tile.bop NumericDType.real.add (.consSame (.consSame .nil))
      (carryS p l a n) (Tile.dot [] (queryTile p l a n) (keyTile p l a n)) = carryS p l a (n+1) := by
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  rw [Tile.bop_data]
  have hd := dot2d_elem (queryTile p l a n) (keyTile p l a n) i j
    (fun e => query p l a i.val (n*p.BK+e.val))
    (fun e => key p l a (n*p.BK+e.val) j.val) (fun _ => rfl) (fun _ => rfl)
  simp only [Broadcast.leftIndex,Broadcast.rightIndex,hd,carryS,Tile.mat,
    NumericDType.add,WithBot.realAdd,Option.map₂_some_some,qkSum,Finset.sum_range_succ]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaSimple.Real

namespace VeriTile.Bench.TritonBenchG.ChunkGlaSimple.Real
open VeriTile.Triton Math
noncomputable section
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

def gateTile (p : Layout) (l : Launch) (a : Inputs) : Tile .real [p.BT] :=
  Tile.vec fun i => some (gate p l a i.val)
def valueTile (p : Layout) (l : Launch) (a : Inputs) : Tile .real [p.BT,p.BV] :=
  Tile.mat fun i j => some (value p l a i.val j.val)
def decayedO (p : Layout) (l : Launch) (a : Inputs) : Tile .real [p.BT,p.BV] :=
  Tile.bop NumericDType.real.mul (.consSame (.consR .nil)) (carryO p l a (numBlocks p))
    (Tile.expandDim ⟨1,by simp⟩ (Tile.uop WithBot.realExp (gateTile p l a)))
def decayedS (p : Layout) (l : Launch) (a : Inputs) : Tile .real [p.BT,p.BT] :=
  Tile.bop NumericDType.real.mul (.consSame (.consSame .nil)) (carryS p l a (numBlocks p))
    (Tile.uop WithBot.realExp (Tile.bop NumericDType.real.sub (.consR (.consL .nil))
      (Tile.expandDim ⟨1,by simp⟩ (gateTile p l a)) (Tile.expandDim ⟨0,by simp⟩ (gateTile p l a))))
def maskedS (p : Layout) (l : Launch) (a : Inputs) : Tile .real [p.BT,p.BT] :=
  Tile.select (maskTile p) (decayedS p l a) (⟨fun _ => some 0⟩ : Tile .real [p.BT,p.BT])
def resultTile (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) : Tile .real [p.BT,p.BV] :=
  Tile.bop NumericDType.real.mul .scalarR
    (Tile.bop NumericDType.real.add (.consSame (.consSame .nil))
      (decayedO p l a) (Tile.dot [] (maskedS p l a) (valueTile p l a))) (Tile.scalar (some scale))

theorem masked_value (p : Layout) (l : Launch) (a : Inputs) (i j : Fin p.BT) :
    (maskedS p l a).data (i,j,⟨⟩) = some (score p l a i.val j.val) := by
  by_cases h : j.val ≤ i.val
  all_goals simp [maskedS,maskTile,decayedS,carryS,gateTile,Tile.select,Tile.bop,Tile.uop,Tile.expandDim,
    Tile.mat,Tile.vec,TileShape.dropInsertedIndex,Broadcast.leftIndex,Broadcast.rightIndex,
    NumericDType.mul,NumericDType.sub,WithBot.realExp,score,h,-Fin.val_fin_le]

theorem result_value (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i : Fin p.BT) (j : Fin p.BV) :
    (resultTile p l a scale).data (i,j,⟨⟩) = some (expected p l a scale i.val j.val) := by
  have hd := dot2d_elem (maskedS p l a) (valueTile p l a) i j
    (fun t => score p l a i.val t.val) (fun t => value p l a t.val j.val)
    (masked_value p l a i) (fun _ => rfl)
  simp only [resultTile,decayedO,carryO,gateTile,Tile.bop_data,Tile.uop_data,
    Tile.expandDim_data,TileShape.dropInsertedIndex,Broadcast.leftIndex,Broadcast.rightIndex,
    Tile.mat,Tile.vec,hd,Tile.scalar,NumericDType.mul,NumericDType.add,WithBot.realMul,
    WithBot.realAdd,WithBot.realExp,Option.map₂_some_some,Option.map_some,expected]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaSimple.Real
