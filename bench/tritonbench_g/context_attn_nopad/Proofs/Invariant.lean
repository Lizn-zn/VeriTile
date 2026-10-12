import bench.tritonbench_g.context_attn_nopad.Proofs.PreludeComplete
import bench.tritonbench_g.context_attn_nopad.Proofs.NormalizedStep
namespace VeriTile.Bench.TritonBenchG.ContextAttnNopad.Invariant
open VeriTile.Triton Formula Program PreludeComplete
noncomputable section
set_option maxHeartbeats 1600000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

/-- The loop reads only source-initialized registers and initial memory. -/
structure Static (p : Layout) (r : Regions) (initial state : BlockState) : Prop where
  memory : state.mem = initial.mem
  pids : state.pids = initial.pids
  batch : state.regs .nat [] "cur_batch" = some (Tile.scalar (initial.pids 0))
  head : state.regs .nat [] "cur_head" = some (Tile.scalar (initial.pids 1))
  start : state.regs .int [] "cur_batch_in_all_start_index" =
    some (Tile.scalar ((inputs initial r).start (initial.pids 0)))
  sequence : state.regs .int [] "cur_batch_seq_len" =
    some (Tile.scalar (Formula.sequence p (launch initial) (inputs initial r)))
  rows : state.regs .nat [p.BM] "offs_m" =
    some (Tile.vec (fun i => queryRow p (launch initial) i.val))
  lanes : state.regs .nat [p.BN] "offs_n" = some (Tile.vec (fun j => j.val))
  channels : state.regs .nat [p.D] "offs_d" = some (Tile.vec (fun d => d.val))
  query : state.regs .real [p.BM,p.D] "q" =
    some (Tile.mat (dtype := .real) (fun i d => some (Formula.query p (launch initial) (inputs initial r) i.val d.val)))
  key_ptrs : state.regs .ptr [p.D,p.BN] "k_ptrs" =
    some (Tile.mat (fun d j => (Region.cast r.key,
      j.val*p.kRow+(initial.pids 1)*p.kHead+d.val*p.kDim)))
  value_ptrs : state.regs .ptr [p.BN,p.D] "v_ptrs" =
    some (Tile.mat (fun j d => (Region.cast r.value,
      j.val*p.vRow+(initial.pids 1)*p.vHead+d.val*p.vDim)))

def denominatorTile (p : Layout) (r : Regions) (u : BlockState) (scale : ℝ) (c : Nat)
    (m : Fin p.BM → WithBot ℝ) : Tile .real [p.BM] :=
  Tile.vec (fun i => some (NormalizedStep.factor (m i)*
    Formula.denominator p (launch u) (inputs u r) scale i.val c))
def accumulatorTile (p : Layout) (r : Regions) (u : BlockState) (scale : ℝ) (c : Nat) : Tile .real [p.BM,p.D] :=
  Tile.mat (fun i d => some (Formula.numerator p (launch u) (inputs u r) scale i.val d.val c /
    Formula.denominator p (launch u) (inputs u r) scale i.val c))

/-- A bottom running maximum occurs only at the empty prefix. Accumulators
are the independent causal weighted ratio after every completed key block. -/
structure Dynamic (p : Layout) (r : Regions) (initial : BlockState) (scale : ℝ) (c : Nat)
    (state : BlockState) : Prop where
  centers : ∃ m : Fin p.BM → WithBot ℝ,
    state.regs .real [p.BM] "m_i" = some (Tile.vec m) ∧
    (∀ i, m i = ⊥ → c = 0) ∧
    state.regs .real [p.BM] "l_i" = some (denominatorTile p r initial scale c m)
  accumulator : state.regs .real [p.BM,p.D] "acc" = some (accumulatorTile p r initial scale c)

theorem static_initialized (p : Layout) (r : Regions) (u : BlockState) :
    Static p r u (initialized p r u) := by
  constructor
  · exact PreludeComplete.memory p r u
  · exact PreludeComplete.pids p r u
  all_goals simp [initialized,PreludeQuery.queried,offsetState,metadataState,launchState,
    PreludeOffsets.prepared,PreludeMetadata.prepared,PreludeLaunch.launched,
    launch,Formula.sequence,Formula.query,queryRow,
    inputs,Nat.cast_add,Nat.cast_mul,Int.sub_add_cancel]

theorem dynamic_initialized (p : Layout) (r : Regions) (u : BlockState) (scale : ℝ) :
    Dynamic p r u scale 0 (initialized p r u) := by
  constructor
  · refine ⟨fun _ => ⊥,?_,?_,?_⟩
    · simp [initialized,PreludeQuery.queried]
    · intros; rfl
    · simp [initialized,PreludeQuery.queried,denominatorTile,Formula.denominator_zero]
  · simp [initialized,PreludeQuery.queried,accumulatorTile,Formula.denominator_zero,Formula.numerator_zero]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnNopad.Invariant
