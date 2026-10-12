import VeriTile.Triton.DSL
import VeriTile.Triton.Source
import VeriTile.Meta.StatementAudit

/-! Complete source of `bgmv_shrink_kernel.py`'s `_bgmv_shrink_kernel`.
Both versions retain the signed sentinel, every split-K iteration and mask,
the exact SPLIT_K == 1 choice between store and atomic addition, and explicit
output conversion. The candidate commutes only the final fp32 scaling.

Input and weights share fp16 or bf16 storage, as required by the host. The
unused xk_stride parameter remains in the interface: the source reads its
input contiguously. Program IDs, dimensions, other strides and scaling stay
symbolic. Atomic execution is per program; no cross-program interleaving or
launch-wide reduction is claimed here.

Translation-surface blocker: this macro specializes storage dtypes and makes
Triton's implicit store/atomic operand conversion explicit. The early return
becomes a non-sentinel active guard. `tl.max_contiguous` is a value-preserving
layout hint. The original Python source remains in this directory.

Correct.lean proves both complete real flat-IO contracts; FPEquiv.lean proves
their equivalence from the admitted scaling atom. Proofs/RealFoundation keeps
older active surfaces solely as a source of reusable mathematical lemmas.
-/
open Lean
open VeriTile.Triton
namespace VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.TypedSource
syntax "bgmvShrinkSource%" "[" tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(bgmvShrinkSource%[$storage:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `scalarLeft do
      Macro.throwError "expected original or scalarLeft"
    let scale : TSyntax `tritonExpr ←
      if version.getId == `scalarLeft then `(tritonExpr| $(scaling) * accumulator)
      else `(tritonExpr| accumulator * $(scaling))
    `(fun (out_dtype : FloatDType) (input_ptr lora_ptr out_ptr : RegionName)
        (N K : Nat) (lora_indices : Region .int) (scaling : ℝ)
        (xm_stride _xk_stride l0_stride lora_k_stride lora_n_stride cm_stride
          cn_stride BLOCK_N BLOCK_K SPLIT_K : Nat) =>
    ComputeKernel.mk [input_ptr, lora_ptr, out_ptr, Region.cast lora_indices] [out_ptr]
    (triton {
      tl.region input_ptr = $storage, lora_ptr = $storage, out_ptr = OUT_DTYPE
      pid_sk = tl.program_id(axis=0)
      cur_batch = tl.program_id(axis=1)
      lora_index = tl.load(lora_indices + cur_batch)
      if lora_index != $((-1 : Int)) {
        offset_n = tl.arange(0, $(BLOCK_N))
        offset_k = tl.arange(0, $(BLOCK_K)) + pid_sk * $(BLOCK_K)
        a_ptr = input_ptr + cur_batch * $(xm_stride)
        b_ptr = lora_ptr + $(l0_stride) * lora_index
        accumulator = tl.zeros([$(BLOCK_N)], dtype=tl.float32)
        for k in range($(0), $(K), $(BLOCK_K * SPLIT_K)) {
          current_k = k + offset_k
          current_k_c = tl.max_contiguous(current_k, $(BLOCK_K))
          tiled_a = tl.load(a_ptr + current_k_c, mask=current_k < $(K), other=0.0)
          b_ptr_mask = (offset_n[:, None] < $(N)) & (current_k[None, :] < $(K))
          tiled_b = tl.load(
            b_ptr + offset_n[:, None] * $(lora_k_stride) +
              current_k[None, :] * $(lora_n_stride),
            mask=b_ptr_mask, other=0.0)
          accumulator += tl.sum(tiled_a * tiled_b, 1)
        }
        accumulator = $scale
        offset_cn = tl.arange(0, $(BLOCK_N))
        c_ptr = out_ptr + cur_batch * $(cm_stride) + offset_cn * $(cn_stride)
        c_mask = offset_cn < $(N)
        if $(SPLIT_K) == $(1) {
          tl.store(c_ptr, (accumulator).to(OUT_DTYPE), mask=c_mask)
        } else {
          tl.atomic_add(c_ptr, (accumulator).to(OUT_DTYPE), mask=c_mask)
        }
      }
    }).surfaceBody)

inductive Storage where
  | fp16 | bf16
  deriving DecidableEq

abbrev KernelSource := FloatDType → RegionName → RegionName → RegionName → Nat → Nat →
  Region .int → ℝ → Nat → Nat → Nat → Nat → Nat → Nat → Nat → Nat → Nat → Nat → ComputeKernel

def specialize (storage : Storage) (swapped : Bool) : KernelSource :=
  match storage, swapped with
  | .fp16, Bool.false => bgmvShrinkSource%[tl.float16, original]
  | .fp16, Bool.true => bgmvShrinkSource%[tl.float16, scalarLeft]
  | .bf16, Bool.false => bgmvShrinkSource%[tl.bfloat16, original]
  | .bf16, Bool.true => bgmvShrinkSource%[tl.bfloat16, scalarLeft]

def original (storage : Storage) : KernelSource := specialize storage Bool.false

def scalarLeft (storage : Storage) : KernelSource := specialize storage Bool.true

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.TypedSource
