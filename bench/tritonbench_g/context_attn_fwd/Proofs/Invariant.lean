import bench.tritonbench_g.context_attn_fwd.Proofs.PreludeComplete
namespace VeriTile.Bench.TritonBenchG.ContextAttnFwd.Invariant
open VeriTile.Triton Formula Program PreludeComplete
open VeriTile.Bench.TritonBenchG.ContextAttnBloom.Math (gStateBot gKeysUpto)
noncomputable section
set_option maxHeartbeats 1600000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

/-- The loop reads only source-initialized registers and initial memory. -/
structure Static (p : Layout) (r : Regions) (initial state : BlockState) : Prop where
  memory : state.mem = initial.mem
  pids : state.pids = initial.pids
  batch : state.regs .nat [] "cur_batch" = some (Tile.scalar (initial.pids 1/p.heads))
  head : state.regs .nat [] "cur_head" = some (Tile.scalar (initial.pids 1%p.heads))
  kv_head : state.regs .nat [] "cur_kv_head" = some (Tile.scalar ((initial.pids 1%p.heads)/p.kvGroup))
  start : state.regs .int [] "cur_batch_in_all_start_index" =
    some (Tile.scalar ((inputs initial r).start (initial.pids 1/p.heads)))
  prompt : state.regs .int [] "prompt_cache_len" =
    some (Tile.scalar ((inputs initial r).prompt (initial.pids 1/p.heads)))
  sequence : state.regs .int [] "cur_batch_seq_len" =
    some (Tile.scalar (Formula.sequence p (launch initial) (inputs initial r)))
  ending : state.regs .int [] "block_end_loc" =
    some (Tile.scalar (Formula.limit p (launch initial) (inputs initial r)))
  rows : state.regs .nat [p.BM] "offs_m" =
    some (Tile.vec (fun i => queryRow p (launch initial) i.val))
  lanes : state.regs .nat [p.BN] "offs_n" = some (Tile.vec (fun j => j.val))
  channels : state.regs .nat [p.D] "offs_d" = some (Tile.vec (fun d => d.val))
  query : state.regs .real [p.BM,p.D] "q" =
    some (Tile.mat (dtype := .real) (fun i d => some (Formula.query p (launch initial) (inputs initial r) i.val d.val)))

def rowPairs (p : Layout) (r : Regions) (u : BlockState) (scale : ℝ) (i d : Nat) :=
  Formula.pairs p (launch u) (inputs u r) scale i d

def maxTile (p : Layout) (r : Regions) (u : BlockState) (scale : ℝ) (c : Nat) : Tile .real [p.BM] :=
  Tile.vec (fun i => (gStateBot (count p (launch u) (inputs u r)*p.BN) (c*p.BN)
    (rowPairs p r u scale i.val 0)).1)
def denominatorTile (p : Layout) (r : Regions) (u : BlockState) (scale : ℝ) (c : Nat) : Tile .real [p.BM] :=
  Tile.vec (fun i => some (gStateBot (count p (launch u) (inputs u r)*p.BN) (c*p.BN)
    (rowPairs p r u scale i.val 0)).2.1)
def accumulatorTile (p : Layout) (r : Regions) (u : BlockState) (scale : ℝ) (c : Nat) : Tile .real [p.BM,p.D] :=
  Tile.mat (fun i d => some (gStateBot (count p (launch u) (inputs u r)*p.BN) (c*p.BN)
    (rowPairs p r u scale i.val d.val)).2.2)

structure Dynamic (p : Layout) (r : Regions) (initial : BlockState) (scale : ℝ) (c : Nat)
    (state : BlockState) : Prop where
  maximum : state.regs .real [p.BM] "m_i" = some (maxTile p r initial scale c)
  denominator : state.regs .real [p.BM] "l_i" = some (denominatorTile p r initial scale c)
  accumulator : state.regs .real [p.BM,p.D] "acc" = some (accumulatorTile p r initial scale c)

theorem static_initialized (p : Layout) (r : Regions) (u : BlockState) :
    Static p r u (initialized p r u) := by
  constructor
  · exact PreludeComplete.memory p r u
  · exact PreludeComplete.pids p r u
  all_goals simp [initialized,PreludeQuery.queried,offsetState,metadataState,launchState,
    PreludeOffsets.prepared,PreludeMetadata.prepared,PreludeLaunch.launched,
    launch,Formula.batch,Formula.head,Formula.sequence,Formula.limit,Formula.query,queryRow,
    inputs,Nat.cast_add,Nat.cast_mul,Int.sub_add_cancel]

theorem empty_fold (S : Nat) (g : Fin S → ℝ × ℝ) : gStateBot S 0 g = (⊥,0,0) := by
  rw [gStateBot, show gKeysUpto S 0 g = [] from by
    rw [gKeysUpto]; apply List.filterMap_eq_nil_iff.mpr; intro j _; simp]
  rfl

theorem dynamic_initialized (p : Layout) (r : Regions) (u : BlockState) (scale : ℝ) :
    Dynamic p r u scale 0 (initialized p r u) := by
  constructor
  all_goals simp [initialized,PreludeQuery.queried,maxTile,denominatorTile,accumulatorTile,empty_fold]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnFwd.Invariant
