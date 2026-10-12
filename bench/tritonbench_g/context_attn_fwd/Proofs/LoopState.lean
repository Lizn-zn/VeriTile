import bench.tritonbench_g.context_attn_fwd.Proofs.NumericUpdate
import bench.tritonbench_g.context_attn_fwd.Proofs.Invariant
namespace VeriTile.Bench.TritonBenchG.ContextAttnFwd.LoopState
open VeriTile.Triton Formula Program
noncomputable section
set_option maxHeartbeats 700000
set_option maxRecDepth 3000

def size (p : Layout) (r : Regions) (initial : BlockState) := count p (launch initial) (inputs initial r)*p.BN
def rows (p : Layout) (initial : BlockState) : Fin p.BM → Nat := fun i => queryRow p (launch initial) i.val

def query (p : Layout) (r : Regions) (initial : BlockState) : Tile .real [p.BM,p.D] :=
  Tile.mat (fun i d => some (Formula.query p (launch initial) (inputs initial r) i.val d.val))
def key (p : Layout) (r : Regions) (initial : BlockState) (c : Nat) : Tile .real [p.D,p.BN] :=
  Tile.mat (fun d j => some (Formula.key p (launch initial) (inputs initial r) (c*p.BN+j.val) d.val))
def value (p : Layout) (r : Regions) (initial : BlockState) (c : Nat) : Tile .real [p.BN,p.D] :=
  Tile.mat (fun j d => some (Formula.value p (launch initial) (inputs initial r) (c*p.BN+j.val) d.val))
def score (p : Layout) (r : Regions) (initial : BlockState) (scale : ℝ) (c : Nat) : Tile .real [p.BM,p.BN] :=
  Tile.mat (fun i j => some (Formula.score p (launch initial) (inputs initial r) scale i.val (c*p.BN+j.val)))

def pairs (p : Layout) (r : Regions) (initial : BlockState) (scale : ℝ) (i : Fin p.BM) :=
  Invariant.rowPairs p r initial scale i.val

theorem numeric_correct (p : Layout) (r : Regions) (initial : BlockState) (scale : ℝ) (c : Nat)
    (hBN : 0 < p.BN) (hc : (c+1)*p.BN ≤ size p r initial)
    (maximum : Tile .real [p.BM])
    (hrm : Tile.reduceMaxDrop (⟨1,by simp⟩ : Fin [p.BM,p.BN].length) (score p r initial scale c) = some maximum) :
    NumericUpdate.update p (size p r initial) c (pairs p r initial scale)
      (score p r initial scale c) (value p r initial c) maximum =
      (Invariant.maxTile p r initial scale (c+1),Invariant.denominatorTile p r initial scale (c+1),
        Invariant.accumulatorTile p r initial scale (c+1)) := by
  exact NumericUpdate.correct p (size p r initial) c hBN hc _ (by intros; rfl)
    _ _ maximum hrm (by intros; rfl) (by intros; rfl)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnFwd.LoopState
