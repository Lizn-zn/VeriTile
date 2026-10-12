import bench.tritonbench_g.context_attn_bloom.Proofs.NumericUpdate
import bench.tritonbench_g.context_attn_bloom.Proofs.Invariant
namespace VeriTile.Bench.TritonBenchG.ContextAttnBloom.LoopState
open VeriTile.Triton Math Program RealSetup
noncomputable section
set_option maxHeartbeats 700000
set_option maxRecDepth 3000
set_option linter.unusedSimpArgs false

def size (p : Layout) (r : Regions) (initial : BlockState) := count p (launch initial) (inputs initial r)*p.BN
def rows (p : Layout) (initial : BlockState) : Fin p.BM → Nat := fun i => queryRow p (launch initial) i.val
def position (p : Layout) (r : Regions) (initial : BlockState) : Fin p.BM → Nat :=
  fun i => (rows p initial i+(inputs initial r).prompt (initial.pids 0)).toNat

def query (p : Layout) (r : Regions) (initial : BlockState) : Tile .real [p.BM,p.D] :=
  Tile.mat (fun i d => some (Math.query p (launch initial) (inputs initial r) i.val d.val))
def key (p : Layout) (r : Regions) (initial : BlockState) (c : Nat) : Tile .real [p.D,p.BN] :=
  Tile.mat (fun d j => some (Math.key p (launch initial) (inputs initial r) (c*p.BN+j.val) d.val))
def value (p : Layout) (r : Regions) (initial : BlockState) (c : Nat) : Tile .real [p.BN,p.D] :=
  Tile.mat (fun j d => some (Math.value p (launch initial) (inputs initial r) (c*p.BN+j.val) d.val))
def score (p : Layout) (r : Regions) (initial : BlockState) (scale : ℝ) (c : Nat) : Tile .real [p.BM,p.BN] :=
  Tile.mat (fun i j => some (Math.score p (launch initial) (inputs initial r) scale i.val (c*p.BN+j.val)))

def pairs (p : Layout) (r : Regions) (initial : BlockState) (scale : ℝ) (i : Fin p.BM) :=
  Invariant.rowPairs p r initial scale i.val

theorem initial_maximum (p : Layout) (r : Regions) (initial : BlockState) (scale : ℝ) (c : Nat) :
    NumericUpdate.maximum p (size p r initial) c (pairs p r initial scale) =
      Invariant.maxTile p r initial scale c := rfl

theorem initial_denominator (p : Layout) (r : Regions) (initial : BlockState) (scale : ℝ) (c : Nat) :
    NumericUpdate.denominator p (size p r initial) c (pairs p r initial scale) =
      Invariant.denominatorTile p r initial scale c := rfl

theorem initial_accumulator (p : Layout) (r : Regions) (initial : BlockState) (scale : ℝ) (c : Nat) :
    NumericUpdate.accumulator p (size p r initial) c (position p r initial) (pairs p r initial scale) =
      Invariant.accumulatorTile p r initial scale c := rfl

theorem numeric_correct (p : Layout) (r : Regions) (initial : BlockState) (scale : ℝ) (c : Nat)
    (hBN : 0 < p.BN) (hc : (c+1)*p.BN ≤ size p r initial)
    (hp : 0 ≤ (inputs initial r).prompt (initial.pids 0))
    (maximum : Tile .real [p.BM])
    (hrm : Tile.reduceMaxDrop (⟨1,by simp⟩ : Fin [p.BM,p.BN].length) (score p r initial scale c) = some maximum) :
    NumericUpdate.update p (size p r initial) c (pairs p r initial scale)
      (position p r initial) (rows p initial) ((inputs initial r).prompt (initial.pids 0))
      (score p r initial scale c) (value p r initial c) maximum =
      (Invariant.maxTile p r initial scale (c+1),Invariant.denominatorTile p r initial scale (c+1),
        Invariant.accumulatorTile p r initial scale (c+1)) := by
  rw [← initial_maximum,← initial_denominator,← initial_accumulator]
  refine NumericUpdate.correct p (size p r initial) c hBN hc _ ?_ _ _ _ ?_ _ _ maximum hrm ?_ ?_
  · intro i d j
    rfl
  · intro i
    simp only [position]
    omega
  · intro i j d
    simp only [score,Tile.mat_data,pairs,Invariant.rowPairs,Formula.pairs]
    rw [div_mul_cancel₀ _ (by positivity : Real.log 2 ≠ 0)]
  · intro i j d
    rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnBloom.LoopState
