import bench.tritonbench_g.context_attn_mistral.Proofs.Invariant
namespace VeriTile.Bench.TritonBenchG.ContextAttnMistral.LoopState
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

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnMistral.LoopState
