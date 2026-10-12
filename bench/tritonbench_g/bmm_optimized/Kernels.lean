import VeriTile.Triton.DSL
import VeriTile.Triton.Source
import VeriTile.Meta.StatementAudit

/-! Complete source of `bmm_optimized.py`'s `bmm_kernel`, with all eight divisible-dimension mask choices and
both CTA-ordering branches. The candidate reverses only each fp32 addition.
The source omits `other` on masked loads; no zero fill is inserted here.
Typed storage, default fp32 dot output and allow_tf32=false remain explicit.
Triton's implicit store conversion to the pointer element type is written
explicitly as `.to(dtype)` before both masked and unmasked stores:
https://triton-lang.org/main/python-api/generated/triton.language.store.html

Translation-surface blocker: host constexpr mask choices are specialized by
the macro because `None` is syntax-level, not a mask register. The tuple
assignment is split into two statements. Pointer parameter updates, grid
swizzle including the final short group, all load/store masks and the actual
K loop remain. A/B/Out share the selected floating storage type. -/
open Lean VeriTile.Triton
namespace VeriTile.Bench.TritonBenchG.BmmOptimized.TypedSource
set_option maxHeartbeats 4000000
set_option maxRecDepth 12000
syntax "bmmOptimizedSource%" "[" tritonDType "," ident "," ident "," ident "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(bmmOptimizedSource%[$dtype:tritonDType, $dm:ident, $dn:ident, $dk:ident, $version:ident]) => do
    for flag in [dm, dn, dk] do
      unless flag.getId == `yes || flag.getId == `no do
        Macro.throwError "divisibility flags must be yes or no"
    unless version.getId == `original || version.getId == `dotFirst do
      Macro.throwError "expected original or dotFirst"
    let m := dm.getId == `yes
    let n := dn.getId == `yes
    let k := dk.getId == `yes
    let mut before : Array (TSyntax `tritonStmt) := #[]
    if !m then before := before.push (← `(tritonStmt| mask_m = offs_m < $(M)))
    if !n then before := before.push (← `(tritonStmt| mask_n = offs_n < $(N)))
    let mut masks : Array (TSyntax `tritonStmt) := #[]
    if !k then masks := masks.push (← `(tritonStmt| mask_k = offs_k < $(K)))
    if !m && !k then masks := masks.push (← `(tritonStmt| mask_a = mask_m[:, None] & mask_k[None, :]))
    else if !m then masks := masks.push (← `(tritonStmt| mask_a = mask_m[:, None]))
    else if !k then masks := masks.push (← `(tritonStmt| mask_a = mask_k[None, :]))
    if !k && !n then masks := masks.push (← `(tritonStmt| mask_b = mask_k[:, None] & mask_n[None, :]))
    else if !k then masks := masks.push (← `(tritonStmt| mask_b = mask_k[:, None]))
    else if !n then masks := masks.push (← `(tritonStmt| mask_b = mask_n[None, :]))
    let loadA : TSyntax `tritonStmt ← if m && k then `(tritonStmt| a = tl.load(a_ptrs))
      else `(tritonStmt| a = tl.load(a_ptrs, mask=mask_a))
    let loadB : TSyntax `tritonStmt ← if k && n then `(tritonStmt| b = tl.load(b_ptrs))
      else `(tritonStmt| b = tl.load(b_ptrs, mask=mask_b))
    let add : TSyntax `tritonStmt ← if version.getId == `original
      then `(tritonStmt| o += tl.dot(a, b, allow_tf32=false))
      else `(tritonStmt| o = tl.dot(a, b, allow_tf32=false) + o)
    let mut after : Array (TSyntax `tritonStmt) := #[]
    if !m && !n then after := after.push (← `(tritonStmt| mask_c = mask_m[:, None] & mask_n[None, :]))
    else if !m then after := after.push (← `(tritonStmt| mask_c = mask_m[:, None]))
    else if !n then after := after.push (← `(tritonStmt| mask_c = mask_n[None, :]))
    let store : TSyntax `tritonStmt ← if m && n then `(tritonStmt| tl.store(o_ptrs, (o).to($dtype:tritonDType)))
      else `(tritonStmt| tl.store(o_ptrs, (o).to($dtype:tritonDType), mask=mask_c))
    `(fun (A B O : RegionName) (M N K TILE_M TILE_N TILE_K GROUP_M : Nat) =>
      ComputeKernel.mk [A,B] [O] (triton {
        tl.region A = $dtype, B = $dtype, O = $dtype
        pid_b = tl.program_id(2)
        A += pid_b * $(M) * $(K)
        B += pid_b * $(K) * $(N)
        O += pid_b * $(M) * $(N)
        pidx = tl.program_id(0)
        pidy = tl.program_id(1)
        if $(GROUP_M) == $(1) {
          pid_m = pidx
          pid_n = pidy
        } else {
          gridx = tl.num_programs(0)
          gridy = tl.num_programs(1)
          pid = pidx + pidy * gridx
          num_CTA_per_group = gridy * $(GROUP_M)
          group_id = pid // num_CTA_per_group
          inner_group_id = pid % num_CTA_per_group
          if (group_id * $(GROUP_M) + $(GROUP_M)) > gridx {
            GROUP_SIZE = gridx % $(GROUP_M)
          } else {
            GROUP_SIZE = $(GROUP_M)
          }
          pid_m = group_id * $(GROUP_M) + inner_group_id % GROUP_SIZE
          pid_n = inner_group_id // GROUP_SIZE
        }
        offs_m = pid_m * $(TILE_M) + tl.arange(0, $(TILE_M))
        offs_n = pid_n * $(TILE_N) + tl.arange(0, $(TILE_N))
        offs_k = tl.arange(0, $(TILE_K))
        $before*
        a_ptrs = A + offs_m[:, None] * $(K) + offs_k[None, :]
        b_ptrs = B + offs_k[:, None] * $(N) + offs_n[None, :]
        o_ptrs = O + offs_m[:, None] * $(N) + offs_n[None, :]
        num_iters = tl.cdiv($(K), $(TILE_K))
        o = tl.zeros([$(TILE_M), $(TILE_N)], dtype=tl.float32)
        for _i in range($(0), num_iters, $(1)) {
          $masks*
          $loadA
          $loadB
          offs_k += $(TILE_K)
          a_ptrs += $(TILE_K)
          b_ptrs += $(TILE_K) * $(N)
          $add
        }
        $after*
        $store
      }).surfaceBody)

inductive Storage where
  | fp32 | fp16 | bf16
  deriving DecidableEq

def specialize (dtype : Storage) (divM divN divK swapped : Bool) :=
  match dtype, divM, divN, divK, swapped with
  | .fp32, Bool.false, Bool.false, Bool.false, Bool.false => bmmOptimizedSource%[tl.float32, no, no, no, original]
  | .fp32, Bool.false, Bool.false, Bool.false, Bool.true => bmmOptimizedSource%[tl.float32, no, no, no, dotFirst]
  | .fp32, Bool.false, Bool.false, Bool.true, Bool.false => bmmOptimizedSource%[tl.float32, no, no, yes, original]
  | .fp32, Bool.false, Bool.false, Bool.true, Bool.true => bmmOptimizedSource%[tl.float32, no, no, yes, dotFirst]
  | .fp32, Bool.false, Bool.true, Bool.false, Bool.false => bmmOptimizedSource%[tl.float32, no, yes, no, original]
  | .fp32, Bool.false, Bool.true, Bool.false, Bool.true => bmmOptimizedSource%[tl.float32, no, yes, no, dotFirst]
  | .fp32, Bool.false, Bool.true, Bool.true, Bool.false => bmmOptimizedSource%[tl.float32, no, yes, yes, original]
  | .fp32, Bool.false, Bool.true, Bool.true, Bool.true => bmmOptimizedSource%[tl.float32, no, yes, yes, dotFirst]
  | .fp32, Bool.true, Bool.false, Bool.false, Bool.false => bmmOptimizedSource%[tl.float32, yes, no, no, original]
  | .fp32, Bool.true, Bool.false, Bool.false, Bool.true => bmmOptimizedSource%[tl.float32, yes, no, no, dotFirst]
  | .fp32, Bool.true, Bool.false, Bool.true, Bool.false => bmmOptimizedSource%[tl.float32, yes, no, yes, original]
  | .fp32, Bool.true, Bool.false, Bool.true, Bool.true => bmmOptimizedSource%[tl.float32, yes, no, yes, dotFirst]
  | .fp32, Bool.true, Bool.true, Bool.false, Bool.false => bmmOptimizedSource%[tl.float32, yes, yes, no, original]
  | .fp32, Bool.true, Bool.true, Bool.false, Bool.true => bmmOptimizedSource%[tl.float32, yes, yes, no, dotFirst]
  | .fp32, Bool.true, Bool.true, Bool.true, Bool.false => bmmOptimizedSource%[tl.float32, yes, yes, yes, original]
  | .fp32, Bool.true, Bool.true, Bool.true, Bool.true => bmmOptimizedSource%[tl.float32, yes, yes, yes, dotFirst]
  | .fp16, Bool.false, Bool.false, Bool.false, Bool.false => bmmOptimizedSource%[tl.float16, no, no, no, original]
  | .fp16, Bool.false, Bool.false, Bool.false, Bool.true => bmmOptimizedSource%[tl.float16, no, no, no, dotFirst]
  | .fp16, Bool.false, Bool.false, Bool.true, Bool.false => bmmOptimizedSource%[tl.float16, no, no, yes, original]
  | .fp16, Bool.false, Bool.false, Bool.true, Bool.true => bmmOptimizedSource%[tl.float16, no, no, yes, dotFirst]
  | .fp16, Bool.false, Bool.true, Bool.false, Bool.false => bmmOptimizedSource%[tl.float16, no, yes, no, original]
  | .fp16, Bool.false, Bool.true, Bool.false, Bool.true => bmmOptimizedSource%[tl.float16, no, yes, no, dotFirst]
  | .fp16, Bool.false, Bool.true, Bool.true, Bool.false => bmmOptimizedSource%[tl.float16, no, yes, yes, original]
  | .fp16, Bool.false, Bool.true, Bool.true, Bool.true => bmmOptimizedSource%[tl.float16, no, yes, yes, dotFirst]
  | .fp16, Bool.true, Bool.false, Bool.false, Bool.false => bmmOptimizedSource%[tl.float16, yes, no, no, original]
  | .fp16, Bool.true, Bool.false, Bool.false, Bool.true => bmmOptimizedSource%[tl.float16, yes, no, no, dotFirst]
  | .fp16, Bool.true, Bool.false, Bool.true, Bool.false => bmmOptimizedSource%[tl.float16, yes, no, yes, original]
  | .fp16, Bool.true, Bool.false, Bool.true, Bool.true => bmmOptimizedSource%[tl.float16, yes, no, yes, dotFirst]
  | .fp16, Bool.true, Bool.true, Bool.false, Bool.false => bmmOptimizedSource%[tl.float16, yes, yes, no, original]
  | .fp16, Bool.true, Bool.true, Bool.false, Bool.true => bmmOptimizedSource%[tl.float16, yes, yes, no, dotFirst]
  | .fp16, Bool.true, Bool.true, Bool.true, Bool.false => bmmOptimizedSource%[tl.float16, yes, yes, yes, original]
  | .fp16, Bool.true, Bool.true, Bool.true, Bool.true => bmmOptimizedSource%[tl.float16, yes, yes, yes, dotFirst]
  | .bf16, Bool.false, Bool.false, Bool.false, Bool.false => bmmOptimizedSource%[tl.bfloat16, no, no, no, original]
  | .bf16, Bool.false, Bool.false, Bool.false, Bool.true => bmmOptimizedSource%[tl.bfloat16, no, no, no, dotFirst]
  | .bf16, Bool.false, Bool.false, Bool.true, Bool.false => bmmOptimizedSource%[tl.bfloat16, no, no, yes, original]
  | .bf16, Bool.false, Bool.false, Bool.true, Bool.true => bmmOptimizedSource%[tl.bfloat16, no, no, yes, dotFirst]
  | .bf16, Bool.false, Bool.true, Bool.false, Bool.false => bmmOptimizedSource%[tl.bfloat16, no, yes, no, original]
  | .bf16, Bool.false, Bool.true, Bool.false, Bool.true => bmmOptimizedSource%[tl.bfloat16, no, yes, no, dotFirst]
  | .bf16, Bool.false, Bool.true, Bool.true, Bool.false => bmmOptimizedSource%[tl.bfloat16, no, yes, yes, original]
  | .bf16, Bool.false, Bool.true, Bool.true, Bool.true => bmmOptimizedSource%[tl.bfloat16, no, yes, yes, dotFirst]
  | .bf16, Bool.true, Bool.false, Bool.false, Bool.false => bmmOptimizedSource%[tl.bfloat16, yes, no, no, original]
  | .bf16, Bool.true, Bool.false, Bool.false, Bool.true => bmmOptimizedSource%[tl.bfloat16, yes, no, no, dotFirst]
  | .bf16, Bool.true, Bool.false, Bool.true, Bool.false => bmmOptimizedSource%[tl.bfloat16, yes, no, yes, original]
  | .bf16, Bool.true, Bool.false, Bool.true, Bool.true => bmmOptimizedSource%[tl.bfloat16, yes, no, yes, dotFirst]
  | .bf16, Bool.true, Bool.true, Bool.false, Bool.false => bmmOptimizedSource%[tl.bfloat16, yes, yes, no, original]
  | .bf16, Bool.true, Bool.true, Bool.false, Bool.true => bmmOptimizedSource%[tl.bfloat16, yes, yes, no, dotFirst]
  | .bf16, Bool.true, Bool.true, Bool.true, Bool.false => bmmOptimizedSource%[tl.bfloat16, yes, yes, yes, original]
  | .bf16, Bool.true, Bool.true, Bool.true, Bool.true => bmmOptimizedSource%[tl.bfloat16, yes, yes, yes, dotFirst]

def original (dtype : Storage) (divM divN divK : Bool) := specialize dtype divM divN divK Bool.false
def dotFirst (dtype : Storage) (divM divN divK : Bool) := specialize dtype divM divN divK Bool.true

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.BmmOptimized.TypedSource
