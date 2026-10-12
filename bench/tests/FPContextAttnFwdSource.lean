import bench.tritonbench_g.context_attn_fwd.Kernels
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.ContextAttnFwd.SourceTypes
open VeriTile.Triton
set_option maxRecDepth 20000
set_option maxHeartbeats 1000000

def selected (storage : Storage) (candidate : Bool) (BM BN D : Nat) :=
  (if candidate then scaleFirst storage else original storage)
    "q" "k" "v" 2 "out" "start" "sequence" "prompt"
    8 2 1 64 8 2 1 64 8 2 1 8 2 1 2 4 D BM BN

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
    loopType? code "p" 1 = some (dtype storage) ∧
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

/-- Signed metadata, subtraction and min-bound survive all
storage branches. No natural-number clipping changes the source comparisons. -/
theorem signed_metadata (storage : Storage) (candidate : Bool) (BM BN D : Nat) :
    let code := (selected storage candidate BM BN D).surfaceBody
    assignType? code "cur_batch_in_all_start_index" = some .int ∧
    assignType? code "prompt_cache_len" = some .int ∧
    assignType? code "cur_batch_seq_len" = some .int ∧
    assignType? code "block_end_loc" = some .int ∧
    loopType? code "start_n" = some .int := by
  cases storage <;> cases candidate <;> exact ⟨rfl,rfl,rfl,rfl,rfl⟩

def fusedStatement (storage : Storage) (BM BN D : Nat) : ComputeStmt :=
  match storage with
  | .fp16 => .assign .real [BM,D] "acc" (.compute (.alg .fp32
      (.dotAcc .fp16 .fp16 .real (batch := [])
        (.ref .fp16 [BM,BN] "p") (.ref .fp16 [BN,D] "v") (.ref .real [BM,D] "acc"))))
  | .bf16 => .assign .real [BM,D] "acc" (.compute (.alg .fp32
      (.dotAcc .bf16 .bf16 .real (batch := [])
        (.ref .bf16 [BM,BN] "p") (.ref .bf16 [BN,D] "v") (.ref .real [BM,D] "acc"))))
  | .fp32 => .assign .real [BM,D] "acc" (.compute (.alg .fp32
      (.dotAcc .real .real .real (batch := [])
        (.ref .real [BM,BN] "p") (.ref .real [BN,D] "v") (.ref .real [BM,D] "acc"))))

/-- The source's explicit fp32 accumulator remains inside the same dot node. -/
theorem accumulator_dot (storage : Storage) (candidate : Bool) (BM BN D : Nat) :
    ((Source.intRange? (selected storage candidate BM BN D).surfaceBody "start_n").bind
      fun lp => (lp.body.drop (Source.prefixBefore lp.body "acc" 1).length).head?) =
    some (fusedStatement storage BM BN D) := by
  cases storage <;> cases candidate <;> rfl

/-- The first dot computes fp32 scores even when Q/K storage has lower precision. -/
theorem score_precision (storage : Storage) (candidate : Bool) (BM BN D : Nat) :
    loopType? (selected storage candidate BM BN D).surfaceBody "qk" =
      some (match storage with | .fp16 | .bf16 => .fp32 | .fp32 => .real) := by
  cases storage <;> cases candidate <;> rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ContextAttnFwd.SourceTypes
