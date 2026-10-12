import bench.tritonbench_g.context_attn_bloom.Kernels
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.ContextAttnBloom.SourceTypes
open VeriTile.Triton
set_option maxRecDepth 20000
set_option maxHeartbeats 1000000

def selected (storage : Storage) (candidate : Bool) (BM BN D : Nat) :=
  (if candidate then scaleFirst storage else original storage)
    "q" "k" "v" 2 "start" "sequence" "out" "tokens" "request" "prompt"
    8 2 1 8 2 1 8 2 1 8 2 1 8 1 2 D BM D BN

def dtype : Storage → TileDType
  | .fp16 => .fp16
  | .bf16 => .bf16
  | .fp32 => .real

def assignType? (code : List ComputeStmt) (name : RegName) (occurrence : Nat := 0) : Option TileDType :=
  match code with
  | [] => none
  | .assign d _ out _ :: rest =>
    if out = name then match occurrence with
      | 0 => some d
      | k+1 => assignType? rest name k
    else assignType? rest name occurrence
  | _ :: rest => assignType? rest name occurrence

def loopType? (code : List ComputeStmt) (name : RegName) (occurrence : Nat := 0) :=
  (Source.intRange? code "start_n").bind (fun lp => assignType? lp.body name occurrence)
def storeType? (code : List ComputeStmt) := code.getLast?.bind fun st => match st with
  | .store d _ _ _ _ => some d
  | _ => none

/-- Both sources preserve all lower-precision loads, the probability conversion
before the second dot, and the accumulator conversion at the output store. -/
theorem floating_storage (storage : Storage) (candidate : Bool) (BM BN D : Nat) :
    let code := (selected storage candidate BM BN D).surfaceBody
    assignType? code "q" = some (dtype storage) ∧
    loopType? code "k" = some (dtype storage) ∧
    loopType? code "v" = some (dtype storage) ∧
    loopType? code "p" 2 = some (dtype storage) ∧
    storeType? code = some (dtype storage) := by
  cases storage <;> cases candidate <;> exact ⟨rfl,rfl,rfl,rfl,rfl⟩

def assignedLoad? (code : List ComputeStmt) (name : RegName) : Option ComputeDType :=
  (code.drop (Source.prefixBefore code name).length).head?.bind fun st => match st with
  | .assign _ _ _ (.compute (.load precision _ _ _)) => some precision
  | _ => none

/-- The real carrier on the fp32 branch still carries fp32 load annotations. -/
theorem fp32_load_precision (candidate : Bool) (BM BN D : Nat) :
    let code := (selected .fp32 candidate BM BN D).surfaceBody
    assignedLoad? code "q" = some .fp32 ∧
    ((Source.intRange? code "start_n").bind fun lp => assignedLoad? lp.body "k") = some .fp32 ∧
    ((Source.intRange? code "start_n").bind fun lp => assignedLoad? lp.body "v") = some .fp32 := by
  cases candidate <;> exact ⟨rfl,rfl,rfl⟩

/-- Signed metadata, subtraction, min-bound and gathered indices survive all
storage branches. No natural-number clipping changes the source comparisons. -/
theorem signed_metadata (storage : Storage) (candidate : Bool) (BM BN D : Nat) :
    let code := (selected storage candidate BM BN D).surfaceBody
    assignType? code "cur_batch_in_all_start_index" = some .int ∧
    assignType? code "prompt_cache_len" = some .int ∧
    assignType? code "cur_batch_seq_len" = some .int ∧
    assignType? code "cur_batch_req_idx" = some .int ∧
    assignType? code "block_end_loc" = some .int ∧
    loopType? code "kv_loc" = some .int ∧
    loopType? code "start_n" = some .int := by
  cases storage <;> cases candidate <;> exact ⟨rfl,rfl,rfl,rfl,rfl,rfl,rfl⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ContextAttnBloom.SourceTypes
