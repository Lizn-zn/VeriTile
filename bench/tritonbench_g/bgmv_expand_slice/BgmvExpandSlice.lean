import VeriTile.Triton

/-!
# `bgmv_expand_slice` — strict per-kernel correctness

`_bgmv_expand_slice_kernel` is a batched LoRA expand GEMV with a slice offset:
program `(pid_sn, cur_batch)` loads the batch's input vector, selects the LoRA-B
weight via `lora_indices[cur_batch]` (with the `-1` sentinel skipping the batch),
and for each output-`n` block reduces `sum(tiled_a * tiled_b, axis=1)` over the
rank dimension `K`, storing into the output slice shifted by `slice_offset`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`_bgmv_expand_slice_kernel[grid](...)`, the 2D grid
`(SPLIT_N, batches)`, the host setup, and how the runtime composes per-program
writes) is the *trusted boundary*, not a proof obligation here. Because the
program ids are universally quantified, the per-program statement covers every
program of the grid.

## Proof architecture

```
bgmv_expand_slice_surface_toAlgorithm_supported   full general surface (all flags) lowers to the algorithm layer
Full.Final.bgmv_full_output_summary               ← TOP THEOREM: full `for n` loop correctness
  └─ Full.Final.bgmv_full_compute_correct         ← ComputeCorrect over the masked GEMV store, every active lane
       └─ Full.Final.bgmv_full_correct            ← per active lane `g < split_n_length`: readback = bgmvFullSpec
```

The full general kernel surface (all `EVEN_K` / `ADD_INPUTS` / `CAST_TYPE`
branches, the split loop, the slice offset) is shown to lower to the algorithm
layer. Value-level correctness is proved for the full `for n` loop over an
arbitrary `split_n_length` via a writeback loop invariant composed through
`forRangeDyn_inv`: for every active output lane `g < split_n_length`,
`out[g] = Σ_{k < BLOCK_K} prodGK` (`bgmvFullSpec`) — the masked rank-`K`
reduction `sum(tiled_a · tiled_b)` of the loaded input/LoRA-B tiles, stored at
the slice-offset output address — including the `lora_index = -1` sentinel guard.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float). The numeric theorem covers
the main `ADD_INPUTS = false`, no-`CAST_TYPE`
config (`bgmv_full`); the `ADD_INPUTS` accumulation and `CAST_TYPE` variants are
exercised only by the lowering theorem on the general surface, not folded into
the numeric statement. The GEMV `tl.sum` accumulator is modeled exactly as
`Tile.reduceSum` over `ℝ` (no float reassociation claimed). The LoRA-B base is
data-dependent (read from `lora_indices`); output-offset injectivity is
discharged from the one-glance bound `0 < cn_stride` (nonzero output column
stride) via `affine1D_inj`. The statement is scoped to *active* lanes
(`g < split_n_length`).
-/

namespace VeriTile.Bench.TritonBenchG.BgmvExpandSlice

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Faithful transcription of
`bgmv_expand_slice.py`'s `_bgmv_expand_slice_kernel`.

This covers the `EVEN_K` load path, `tl.cdiv` split length, output-block
loop, optional `CAST_TYPE` conversion, and `ADD_INPUTS` accumulation path.
Python's signed `lora_index == -1` early return is represented as a guard
around the active body. -/
def bgmv_expand_slice_surface
    (input_ptr lora_ptr out_ptr : RegionName)
    (N K : Nat) (lora_indices : Region .int)
    (xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride
      cn_stride slice_offset BLOCK_N BLOCK_K SPLIT_N : Nat)
    (EVEN_K ADD_INPUTS CAST_TYPE : Bool) :
    ComputeKernel := triton {
  pid_sn = tl.program_id(axis=0)
  cur_batch = tl.program_id(axis=1)
  lora_index = tl.load(lora_indices + cur_batch)
  if lora_index != $((-1 : Int)) {
    offset_k = tl.arange(0, $(BLOCK_K))
    offset_n = tl.arange(0, $(BLOCK_N))
    if EVEN_K {
      tiled_a = tl.load(input_ptr + cur_batch * $(xm_stride) + offset_k * $(xk_stride))
    } else {
      tiled_a = tl.load(input_ptr + cur_batch * $(xm_stride) + offset_k * $(xk_stride),
        mask=offset_k < $(K), other=0)
    }
    split_n_length = tl.cdiv($(N), $(SPLIT_N))
    if CAST_TYPE {
      tiled_a = (tiled_a).to(lora_ptr.dtype.element_ty)
    }
    b_ptr = lora_ptr + $(l0_stride) * lora_index +
      pid_sn * split_n_length * $(lora_k_stride)
    c_ptr = out_ptr + cur_batch * $(cm_stride) + pid_sn * split_n_length +
      $(slice_offset) * $(cn_stride)
    for n in range($(0), split_n_length, $(BLOCK_N)) {
      current_n = n + offset_n
      b_ptr_mask = (current_n[:, None] < split_n_length) & (offset_k[None, :] < $(K))
      c_mask = current_n < split_n_length
      tiled_b = tl.load(
        b_ptr + current_n[:, None] * $(lora_k_stride) +
          offset_k[None, :] * $(lora_n_stride),
        mask=b_ptr_mask, other=0.0)
      if ADD_INPUTS {
        tiled_out = tl.load(c_ptr + current_n * $(cn_stride), mask=c_mask)
        accumulator = tl.sum(tiled_a * tiled_b, 1) + tiled_out
      } else {
        accumulator = tl.sum(tiled_a * tiled_b, 1)
      }
      tl.store(c_ptr + current_n * $(cn_stride), accumulator, mask=c_mask)
    }
  }
}

/-- The full BGMV expand-slice surface lowers to the algorithm layer, including
the signed sentinel guard, split loop, optional cast, add-inputs branch, and
slice offset in the output pointer. -/
theorem bgmv_expand_slice_surface_toAlgorithm_supported
    (input_ptr lora_ptr out_ptr : RegionName)
    (N K : Nat) (lora_indices : Region .int)
    (xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride
      cn_stride slice_offset BLOCK_N BLOCK_K SPLIT_N : Nat)
    (EVEN_K ADD_INPUTS CAST_TYPE : Bool) :
    ∃ alg,
      (bgmv_expand_slice_surface input_ptr lora_ptr out_ptr N K lora_indices
        xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride
        cn_stride slice_offset BLOCK_N BLOCK_K SPLIT_N EVEN_K ADD_INPUTS
        CAST_TYPE).toAlgorithm? = Except.ok alg := by
  simp [bgmv_expand_slice_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-! ## Full general-loop correctness (arbitrary `split_n_length`, `for n` loop, sentinel guard) -/
namespace Full

namespace ScratchBgmv
open VeriTile.Triton
set_option maxHeartbeats 2000000

def bgmv_full
    (input_ptr lora_ptr out_ptr : RegionName) (lora_indices : Region .int)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride
      lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K : Nat) :
    ComputeKernel := triton {
  pid_sn = tl.program_id(axis=0)
  cur_batch = tl.program_id(axis=1)
  lora_index = tl.load(lora_indices + cur_batch)
  if lora_index != $((-1 : Int)) {
    offset_k = tl.arange(0, $(BLOCK_K))
    offset_n = tl.arange(0, $(BLOCK_N))
    tiled_a = tl.load(input_ptr + cur_batch * $(xm_stride) + offset_k * $(xk_stride),
      mask=offset_k < $(K), other=0)
    b_ptr = lora_ptr + $(l0_stride) * lora_index +
      pid_sn * $(split_n_length) * $(lora_k_stride)
    c_ptr = out_ptr + cur_batch * $(cm_stride) + pid_sn * $(split_n_length) +
      $(slice_offset) * $(cn_stride)
    for n in range($(0), $(split_n_length), $(BLOCK_N)) {
      current_n = n + offset_n
      b_ptr_mask = (current_n[:, None] < $(split_n_length)) & (offset_k[None, :] < $(K))
      c_mask = current_n < $(split_n_length)
      tiled_b = tl.load(
        b_ptr + current_n[:, None] * $(lora_k_stride) +
          offset_k[None, :] * $(lora_n_stride),
        mask=b_ptr_mask, other=0.0)
      accumulator = tl.sum(tiled_a * tiled_b, 1)
      tl.store(c_ptr + current_n * $(cn_stride), accumulator, mask=c_mask)
    }
  }
}

def wbBody (lora_k_stride lora_n_stride cn_stride K split_n_length BLOCK_N BLOCK_K : Nat) : List Stmt :=
  [Stmt.assign TileDType.nat [BLOCK_N] "current_n"
      (Op.add NumericDType.nat Broadcast.scalarL (Op.ref TileDType.nat [] "n")
        (Op.ref TileDType.nat [BLOCK_N] "offset_n")),
    Stmt.assign TileDType.bool [BLOCK_N, BLOCK_K] "b_ptr_mask"
      (Op.boolAnd Broadcast.nil.consL.consR
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.nat [BLOCK_N] "current_n"))
          (Op.constNat split_n_length))
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref TileDType.nat [BLOCK_K] "offset_k"))
          (Op.constNat K))),
    Stmt.assign TileDType.bool [BLOCK_N] "c_mask"
      (Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref TileDType.nat [BLOCK_N] "current_n") (Op.constNat split_n_length)),
    Stmt.assign TileDType.real [BLOCK_N, BLOCK_K] "tiled_b"
      (Op.load TileDType.real
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.nil.consL.consR
            (Op.ptrAdd Broadcast.scalarL (Op.ref TileDType.ptr [] "b_ptr")
              (Op.mul NumericDType.nat Broadcast.scalarR
                (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.nat [BLOCK_N] "current_n")) (Op.constNat lora_k_stride)))
            (Op.mul NumericDType.nat Broadcast.scalarR
              (Op.expandDim ⟨0, by simp⟩ (Op.ref TileDType.nat [BLOCK_K] "offset_k")) (Op.constNat lora_n_stride))))
        (MaskOpt.maskOther (Op.ref TileDType.bool [BLOCK_N, BLOCK_K] "b_ptr_mask")
          ((Op.const (0.0:ℝ)).broadcast [BLOCK_N, BLOCK_K]))),
    Stmt.assign TileDType.real [BLOCK_N] "accumulator"
      (Op.reduceSum ⟨1, by simp⟩ Bool.false
        (Op.mul NumericDType.real Broadcast.nil.consSame.leadL (Op.ref TileDType.real [BLOCK_K] "tiled_a")
          (Op.ref TileDType.real [BLOCK_N, BLOCK_K] "tiled_b"))),
    Stmt.store TileDType.real [BLOCK_N]
      (MemAccess.ptr
        (Op.ptrAdd Broadcast.scalarL (Op.ref TileDType.ptr [] "c_ptr")
          (Op.mul NumericDType.nat Broadcast.scalarR (Op.ref TileDType.nat [BLOCK_N] "current_n") (Op.constNat cn_stride))))
      (Op.ref TileDType.real [BLOCK_N] "accumulator") (MaskOpt.mask (Op.ref TileDType.bool [BLOCK_N] "c_mask"))]

def prefixGuardBody
    (input_ptr lora_ptr out_ptr : RegionName)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride
      lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K : Nat) : List Stmt :=
  [Stmt.assign TileDType.nat [BLOCK_K] "offset_k" (Op.arange BLOCK_K),
   Stmt.assign TileDType.nat [BLOCK_N] "offset_n" (Op.arange BLOCK_N),
   Stmt.assign TileDType.real [BLOCK_K] "tiled_a"
      (Op.load TileDType.real
        (MemAccess.region input_ptr
          (Op.add NumericDType.nat Broadcast.scalarL
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "cur_batch") (Op.constNat xm_stride))
            (Op.mul NumericDType.nat Broadcast.scalarR (Op.ref TileDType.nat [BLOCK_K] "offset_k") (Op.constNat xk_stride))))
        (MaskOpt.maskOther
          (Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref TileDType.nat [BLOCK_K] "offset_k") (Op.constNat K))
          ((Op.const (0:ℝ)).broadcast [BLOCK_K]))),
   Stmt.assign TileDType.ptr [] "b_ptr"
      (Op.ptrAdd Broadcast.nil (Op.ptrBase lora_ptr)
        (Op.add NumericDType.nat Broadcast.nil
          (Op.mul NumericDType.int Broadcast.nil (Op.constNat l0_stride).castNatToInt
              (Op.ref TileDType.int [] "lora_index")).castIntToNat
          (Op.mul NumericDType.nat Broadcast.nil
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "pid_sn") (Op.constNat split_n_length))
            (Op.constNat lora_k_stride)))),
   Stmt.assign TileDType.ptr [] "c_ptr"
      (Op.ptrAdd Broadcast.nil (Op.ptrBase out_ptr)
        (Op.add NumericDType.nat Broadcast.nil
          (Op.add NumericDType.nat Broadcast.nil
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "cur_batch") (Op.constNat cm_stride))
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "pid_sn") (Op.constNat split_n_length)))
          (Op.mul NumericDType.nat Broadcast.nil (Op.constNat slice_offset) (Op.constNat cn_stride)))),
   Stmt.forRange "n" 0 split_n_length BLOCK_N (wbBody lora_k_stride lora_n_stride cn_stride K split_n_length BLOCK_N BLOCK_K)]

theorem body_decomp
    (input_ptr lora_ptr out_ptr : RegionName) (lora_indices : Region .int)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride
      lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K : Nat) :
    (bgmv_full input_ptr lora_ptr out_ptr lora_indices K split_n_length xm_stride xk_stride
      l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K).toAlgKernel.body
      = [Stmt.assign TileDType.nat [] "pid_sn" (Op.programId 0),
         Stmt.assign TileDType.nat [] "cur_batch" (Op.programId 1),
         Stmt.assign TileDType.int [] "lora_index"
            (Op.load TileDType.int (MemAccess.region lora_indices (Op.ref TileDType.nat [] "cur_batch")) MaskOpt.none),
         Stmt.ifThen
            (Op.ne ComparableDType.int Broadcast.nil (Op.ref TileDType.int [] "lora_index") (Op.constInt (-1)))
            (prefixGuardBody input_ptr lora_ptr out_ptr K split_n_length xm_stride xk_stride
              l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K)] := by
  rfl


set_option linter.unusedVariables false
open Finset

/-- Input-vector element `A[cur_batch, k]`: this program's batch row
(`cur_batch = pid1`, row stride `xm_stride`) at rank lane `k` (lane stride
`xk_stride`). -/
noncomputable def aElem (s0 : BlockState) (input_ptr : RegionName) (xm_stride xk_stride : Nat) (k : Nat) : ℝ :=
  s0.readMem input_ptr (s0.pids 1 * xm_stride + k * xk_stride)

/-- LoRA-B weight element `B[li][pid_sn·split_n_length + g, k]`: LoRA index
`li` selects the `l0_stride` slab, the split (`pid_sn = pid0`) plus in-split
output lane `g` select the row (stride `lora_k_stride`), rank lane `k` the
column (stride `lora_n_stride`). -/
noncomputable def bElem (s0 : BlockState) (lora_ptr : RegionName) (li : Nat)
    (split_n_length l0_stride lora_k_stride lora_n_stride : Nat) (g k : Nat) : ℝ :=
  s0.readMem lora_ptr (l0_stride * li + s0.pids 0 * split_n_length * lora_k_stride + g * lora_k_stride + k * lora_n_stride)

-- output offset for global lane g (relative to out_ptr region)
def cOff (s0 : BlockState) (split_n_length cm_stride cn_stride slice_offset : Nat) (g : Nat) : Nat :=
  s0.pids 1 * cm_stride + s0.pids 0 * split_n_length + slice_offset * cn_stride + g * cn_stride

-- masked product for global lane g, key k (over Fin BLOCK_K)
noncomputable def prodGK (s0 : BlockState) (input_ptr lora_ptr : RegionName) (li : Nat)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride : Nat) (g k : Nat) : ℝ :=
  (if k < K then aElem s0 input_ptr xm_stride xk_stride k else 0) *
  (if g < split_n_length ∧ k < K then bElem s0 lora_ptr li split_n_length l0_stride lora_k_stride lora_n_stride g k else 0)

-- full spec for global lane g : sum over k : Fin BLOCK_K of prodGK
noncomputable def bgmvFullSpec (s0 : BlockState) (input_ptr lora_ptr : RegionName) (li : Nat)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride BLOCK_K : Nat) (g : Nat) : ℝ :=
  ∑ k : Fin BLOCK_K, prodGK s0 input_ptr lora_ptr li K split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride g k.val

set_option maxHeartbeats 4000000
set_option maxRecDepth 8000
set_option linter.unusedSimpArgs false

-- matmul-style expandDim helpers
@[simp] theorem evalOp_expandDim_zero_nat {D : Nat} (name : RegName) (s : BlockState) :
    @evalOp .nat [1, D] (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] name)) s =
      (s.regs .nat [D] name).bind (fun v =>
        some ({ data := fun i : TileIndex [1, D] => v.data (i.2.1, PUnit.unit) } : Tile .nat [1, D])) := by
  unfold evalOp; simp [Tile.expandDim]; rfl

@[simp] theorem evalOp_expandDim_one_nat {M : Nat} (name : RegName) (s : BlockState) :
    @evalOp .nat [M, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] name)) s =
      (s.regs .nat [M] name).bind (fun v =>
        some ({ data := fun i : TileIndex [M, 1] => v.data (i.1, PUnit.unit) } : Tile .nat [M, 1])) := by
  unfold evalOp; simp [Tile.expandDim]; rfl

-- current_n eval
theorem currentn_eval (BLOCK_N c : Nat) (s : BlockState)
    (hon : s.regs .nat [BLOCK_N] "offset_n" = some (Tile.vec (fun i : Fin BLOCK_N => i.val))) :
    evalOp (Op.add NumericDType.nat Broadcast.scalarL (Op.ref TileDType.nat [] "n")
        (Op.ref TileDType.nat [BLOCK_N] "offset_n"))
      (s.setReg "n" .nat [] (Tile.scalar (c*BLOCK_N)))
      = some (Tile.vec (fun i : Fin BLOCK_N => c*BLOCK_N + i.val)) := by
  rw [evalOp_add, evalOp_ref_setReg_same,
    evalOp_ref_setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_n":RegName) ≠ "n" by decide),
    evalOp_ref, hon]
  apply congrArg some
  ext i
  simp only [Tile.bop_data, Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL,
    Tile.scalar, Tile.vec, NumericDType.add]

-- c_mask eval (1D)
theorem cmask_eval (split_n_length BLOCK_N c : Nat) (s : BlockState)
    (hcn : s.regs .nat [BLOCK_N] "current_n" = some (Tile.vec (fun i : Fin BLOCK_N => c*BLOCK_N + i.val))) :
    evalOp (Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref TileDType.nat [BLOCK_N] "current_n") (Op.constNat split_n_length)) s
      = some (Tile.vec (fun i : Fin BLOCK_N => decide (c*BLOCK_N + i.val < split_n_length))) := by
  rw [evalOp_lt, evalOp_ref, hcn, evalOp_constNat]
  apply congrArg some
  ext i
  simp only [Tile.cop_data, Tile.vec_data, Tile.scalar_data, Broadcast.leftIndex_scalarR,
    Broadcast.rightIndex_scalarR, ComparableDType.lt, decide_eq_decide]


-- b_ptr_mask eval (2D)
theorem bmask_eval (K split_n_length BLOCK_N BLOCK_K c : Nat) (s : BlockState)
    (hcn : s.regs .nat [BLOCK_N] "current_n" = some (Tile.vec (fun i : Fin BLOCK_N => c*BLOCK_N + i.val)))
    (hok : s.regs .nat [BLOCK_K] "offset_k" = some (Tile.vec (fun j : Fin BLOCK_K => j.val))) :
    evalOp (Op.boolAnd Broadcast.nil.consL.consR
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.nat [BLOCK_N] "current_n"))
          (Op.constNat split_n_length))
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref TileDType.nat [BLOCK_K] "offset_k"))
          (Op.constNat K))) s
      = some ⟨fun idx : TileIndex [BLOCK_N, BLOCK_K] =>
          (decide (c*BLOCK_N + idx.1.val < split_n_length) && decide (idx.2.1.val < K))⟩ := by
  unfold evalOp
  simp only [evalOp_lt, evalOp_constNat, evalOp_expandDim_zero_nat, evalOp_expandDim_one_nat,
    hcn, hok, Option.bind_eq_bind, Option.bind_some, Option.bind]
  apply congrArg some
  ext idx
  simp only [Tile.bop_data, Tile.cop_data, Tile.scalar_data, Tile.vec,
    Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR,
    Broadcast.leftIndex_consL, Broadcast.leftIndex_consR,
    Broadcast.rightIndex_consL, Broadcast.rightIndex_consR,
    Broadcast.leftIndex_nil, Broadcast.rightIndex_nil, ComparableDType.lt]
  rfl


-- tiled_b address tile eval (2D ptr)
theorem bptrs_eval (lora_ptr : RegionName) (lbase lora_k_stride lora_n_stride BLOCK_N BLOCK_K c : Nat) (s : BlockState)
    (hbp : s.regs .ptr [] "b_ptr" = some (Tile.scalar (lora_ptr, lbase)))
    (hcn : s.regs .nat [BLOCK_N] "current_n" = some (Tile.vec (fun i : Fin BLOCK_N => c*BLOCK_N + i.val)))
    (hok : s.regs .nat [BLOCK_K] "offset_k" = some (Tile.vec (fun j : Fin BLOCK_K => j.val))) :
    evalOp (Op.ptrAdd Broadcast.nil.consL.consR
        (Op.ptrAdd Broadcast.scalarL (Op.ref TileDType.ptr [] "b_ptr")
          (Op.mul NumericDType.nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.nat [BLOCK_N] "current_n")) (Op.constNat lora_k_stride)))
        (Op.mul NumericDType.nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref TileDType.nat [BLOCK_K] "offset_k")) (Op.constNat lora_n_stride))) s
      = some (⟨fun idx : TileIndex [BLOCK_N, BLOCK_K] =>
          (lora_ptr, lbase + (c*BLOCK_N + idx.1.val)*lora_k_stride + idx.2.1.val*lora_n_stride)⟩ : Tile .ptr [BLOCK_N, BLOCK_K]) := by
  rw [evalOp_ptrAdd, evalOp_ptrAdd, evalOp_ref, hbp]
  simp only [evalOp_mul, evalOp_constNat, evalOp_expandDim_one_nat, evalOp_expandDim_zero_nat,
    hcn, hok, Option.bind_eq_bind, Option.bind_some, Option.bind]
  apply congrArg some
  ext idx
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec, Tile.scalar_data,
      Broadcast.leftIndex_consL, Broadcast.leftIndex_consR, Broadcast.rightIndex_consL,
      Broadcast.rightIndex_consR, Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL,
      Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR]
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec, Tile.scalar_data, NumericDType.add, NumericDType.mul,
      Broadcast.leftIndex_consL, Broadcast.leftIndex_consR, Broadcast.rightIndex_consL,
      Broadcast.rightIndex_consR, Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL,
      Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR]


-- tiled_b load eval (2D masked ptr)
theorem tiledb_eval (lora_ptr : RegionName) (lbase K split_n_length lora_k_stride lora_n_stride BLOCK_N BLOCK_K c : Nat) (s : BlockState)
    (hbp : s.regs .ptr [] "b_ptr" = some (Tile.scalar (lora_ptr, lbase)))
    (hcn : s.regs .nat [BLOCK_N] "current_n" = some (Tile.vec (fun i : Fin BLOCK_N => c*BLOCK_N + i.val)))
    (hok : s.regs .nat [BLOCK_K] "offset_k" = some (Tile.vec (fun j : Fin BLOCK_K => j.val)))
    (hbm : s.regs .bool [BLOCK_N, BLOCK_K] "b_ptr_mask"
      = some ⟨fun idx : TileIndex [BLOCK_N, BLOCK_K] => (decide (c*BLOCK_N + idx.1.val < split_n_length) && decide (idx.2.1.val < K))⟩) :
    evalOp (Op.load TileDType.real
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.nil.consL.consR
            (Op.ptrAdd Broadcast.scalarL (Op.ref TileDType.ptr [] "b_ptr")
              (Op.mul NumericDType.nat Broadcast.scalarR
                (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.nat [BLOCK_N] "current_n")) (Op.constNat lora_k_stride)))
            (Op.mul NumericDType.nat Broadcast.scalarR
              (Op.expandDim ⟨0, by simp⟩ (Op.ref TileDType.nat [BLOCK_K] "offset_k")) (Op.constNat lora_n_stride))))
        (MaskOpt.maskOther (Op.ref TileDType.bool [BLOCK_N, BLOCK_K] "b_ptr_mask")
          ((Op.const (0.0:ℝ)).broadcast [BLOCK_N, BLOCK_K]))) s
      = some ⟨fun idx : TileIndex [BLOCK_N, BLOCK_K] =>
          some (if c*BLOCK_N + idx.1.val < split_n_length ∧ idx.2.1.val < K then
            s.readMem lora_ptr (lbase + (c*BLOCK_N + idx.1.val)*lora_k_stride + idx.2.1.val*lora_n_stride) else 0.0)⟩ := by
  simp only [evalOp, bptrs_eval lora_ptr lbase lora_k_stride lora_n_stride BLOCK_N BLOCK_K c s hbp hcn hok, hbm]
  apply congrArg some
  ext idx
  simp only [Tile.scalar_data, BlockState.readMemValue_real]
  by_cases h1 : c*BLOCK_N + idx.1.val < split_n_length <;> by_cases h2 : idx.2.1.val < K <;>
    simp [h1, h2]

-- accumulator eval: reduceSum over K of product = bgmvFullSpec lane i
theorem acc_eval (input_ptr lora_ptr : RegionName)
    (li K split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride BLOCK_N BLOCK_K c : Nat)
    (s s0 : BlockState)
    (hrl : s.readMem lora_ptr = s0.readMem lora_ptr)
    (hta : s.regs .real [BLOCK_K] "tiled_a"
      = some ⟨fun j : TileIndex [BLOCK_K] => some (if j.1.val < K then aElem s0 input_ptr xm_stride xk_stride j.1.val else 0)⟩)
    (htb : s.regs .real [BLOCK_N, BLOCK_K] "tiled_b"
      = some ⟨fun idx : TileIndex [BLOCK_N, BLOCK_K] =>
          some (if c*BLOCK_N + idx.1.val < split_n_length ∧ idx.2.1.val < K then
            s.readMem lora_ptr ((l0_stride * li + s0.pids 0 * split_n_length * lora_k_stride) + (c*BLOCK_N + idx.1.val)*lora_k_stride + idx.2.1.val*lora_n_stride) else 0.0)⟩) :
    evalOp (Op.reduceSum ⟨1, by simp⟩ Bool.false
        (Op.mul NumericDType.real Broadcast.nil.consSame.leadL (Op.ref TileDType.real [BLOCK_K] "tiled_a")
          (Op.ref TileDType.real [BLOCK_N, BLOCK_K] "tiled_b"))) s
      = some ⟨fun i : TileIndex [BLOCK_N] =>
          some (bgmvFullSpec s0 input_ptr lora_ptr li K split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride BLOCK_K (c*BLOCK_N + i.1.val))⟩ := by
  rw [evalOp_reduceSum, evalOp_mul, evalOp_ref, hta, evalOp_ref, htb]
  apply congrArg some
  ext i
  simp only [Tile.reduceSum_false, Tile.reduceSumDrop_data, TileShape.axisDim, TileShape.eraseAxis,
    TileShape.insertAxisIndex, Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex]
  rw [bgmvFullSpec]
  rw [← withBot_sum_some]
  apply Finset.sum_congr rfl
  intro k _
  rw [prodGK]
  simp only [Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.mul, WithBot.realMul,
    Option.map₂, Option.bind, Option.map]
  rw [hrl, bElem]
  by_cases h1 : c*BLOCK_N + i.1.val < split_n_length <;> by_cases h2 : k.val < K <;>
    simp only [h1, h2, and_true, and_false, if_true, if_false, true_and, false_and] <;> norm_num


-- c_ptrs address tile eval (1D ptr)
theorem cptrs_eval (out_ptr : RegionName) (cbase cn_stride BLOCK_N c : Nat) (s : BlockState)
    (hcp : s.regs .ptr [] "c_ptr" = some (Tile.scalar (out_ptr, cbase)))
    (hcn : s.regs .nat [BLOCK_N] "current_n" = some (Tile.vec (fun i : Fin BLOCK_N => c*BLOCK_N + i.val))) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ref TileDType.ptr [] "c_ptr")
        (Op.mul NumericDType.nat Broadcast.scalarR (Op.ref TileDType.nat [BLOCK_N] "current_n") (Op.constNat cn_stride))) s
      = some (Tile.vec (fun i : Fin BLOCK_N => (out_ptr, cbase + (c*BLOCK_N + i.val)*cn_stride))) := by
  rw [evalOp_ptrAdd, evalOp_ref, hcp, evalOp_mul, evalOp_ref, hcn, evalOp_constNat]
  apply congrArg some
  ext i
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec, Tile.scalar_data,
      Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL,
      Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR]
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec, Tile.scalar_data, NumericDType.add, NumericDType.mul,
      Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL,
      Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR]


-- store step: the masked ptr store folds to writeMem over lanes
theorem store_step (out_ptr : RegionName) (cbase cn_stride BLOCK_N c : Nat) (s : BlockState)
    (vals : Fin BLOCK_N → ℝ) (mask : Fin BLOCK_N → Bool)
    (hcp : s.regs .ptr [] "c_ptr" = some (Tile.scalar (out_ptr, cbase)))
    (hcn : s.regs .nat [BLOCK_N] "current_n" = some (Tile.vec (fun i : Fin BLOCK_N => c*BLOCK_N + i.val)))
    (hacc : s.regs .real [BLOCK_N] "accumulator" = some ⟨fun i : TileIndex [BLOCK_N] => some (vals i.1)⟩)
    (hcm : s.regs .bool [BLOCK_N] "c_mask" = some (Tile.vec mask)) :
    stepStmt (Stmt.store TileDType.real [BLOCK_N]
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.scalarL (Op.ref TileDType.ptr [] "c_ptr")
            (Op.mul NumericDType.nat Broadcast.scalarR (Op.ref TileDType.nat [BLOCK_N] "current_n") (Op.constNat cn_stride))))
        (Op.ref TileDType.real [BLOCK_N] "accumulator") (MaskOpt.mask (Op.ref TileDType.bool [BLOCK_N] "c_mask"))) s
      = some ((TileShape.allIndices [BLOCK_N]).foldl
          (fun acc (i : TileIndex [BLOCK_N]) =>
            if mask i.1 then acc.writeMem out_ptr (cbase + (c*BLOCK_N + i.1.val)*cn_stride) (vals i.1) else acc) s) := by
  simp only [stepStmt, evalOp_ref, hacc, hcm,
    cptrs_eval out_ptr cbase cn_stride BLOCK_N c s hcp hcn,
    Option.bind_eq_bind, Option.bind_some, Option.map,
    Tile.vec, BlockState.writeMemTyped_real, FloatDType.real_storeValue, WithBot.unbotD_some]

end ScratchBgmv

namespace Step2
open VeriTile.Triton ScratchBgmv

-- abbreviations for the two pointer bases
def lbaseOf (s0 : BlockState) (li split_n_length l0_stride lora_k_stride : Nat) : Nat :=
  l0_stride * li + s0.pids 0 * split_n_length * lora_k_stride

def cbaseOf (s0 : BlockState) (split_n_length cm_stride cn_stride slice_offset : Nat) : Nat :=
  s0.pids 1 * cm_stride + s0.pids 0 * split_n_length + slice_offset * cn_stride

-- cOff matches cbase + g*cn_stride
theorem cOff_eq (s0 : BlockState) (split_n_length cm_stride cn_stride slice_offset g : Nat) :
    cOff s0 split_n_length cm_stride cn_stride slice_offset g
      = cbaseOf s0 split_n_length cm_stride cn_stride slice_offset + g * cn_stride := by
  unfold cOff cbaseOf; ring


theorem wbStep
    (input_ptr lora_ptr out_ptr : RegionName)
    (li K split_n_length xm_stride xk_stride l0_stride lora_k_stride
      lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K : Nat)
    (s0 s : BlockState) (c : Nat)
    (hinj : Function.Injective (fun g : Fin split_n_length =>
      cOff s0 split_n_length cm_stride cn_stride slice_offset g.val))
    (hoi : out_ptr ≠ input_ptr) (hol : out_ptr ≠ lora_ptr)
    (hpids : s.pids = s0.pids)
    (hri : s.readMem input_ptr = s0.readMem input_ptr) (hrl : s.readMem lora_ptr = s0.readMem lora_ptr)
    (hok : s.regs .nat [BLOCK_K] "offset_k" = some (Tile.vec (fun j : Fin BLOCK_K => j.val)))
    (hon : s.regs .nat [BLOCK_N] "offset_n" = some (Tile.vec (fun i : Fin BLOCK_N => i.val)))
    (hta : s.regs .real [BLOCK_K] "tiled_a"
      = some ⟨fun j : TileIndex [BLOCK_K] => some (if j.1.val < K then aElem s0 input_ptr xm_stride xk_stride j.1.val else 0)⟩)
    (hbp : s.regs .ptr [] "b_ptr" = some (Tile.scalar (lora_ptr, lbaseOf s0 li split_n_length l0_stride lora_k_stride)))
    (hcp : s.regs .ptr [] "c_ptr" = some (Tile.scalar (out_ptr, cbaseOf s0 split_n_length cm_stride cn_stride slice_offset))) :
    ∃ s', stepStmts (wbBody lora_k_stride lora_n_stride cn_stride K split_n_length BLOCK_N BLOCK_K)
        (s.setReg "n" .nat [] (Tile.scalar (c*BLOCK_N))) = some s'
      ∧ s'.pids = s0.pids
      ∧ s'.regs .nat [BLOCK_K] "offset_k" = some (Tile.vec (fun j : Fin BLOCK_K => j.val))
      ∧ s'.regs .nat [BLOCK_N] "offset_n" = some (Tile.vec (fun i : Fin BLOCK_N => i.val))
      ∧ s'.regs .real [BLOCK_K] "tiled_a"
          = some ⟨fun j : TileIndex [BLOCK_K] => some (if j.1.val < K then aElem s0 input_ptr xm_stride xk_stride j.1.val else 0)⟩
      ∧ s'.regs .ptr [] "b_ptr" = some (Tile.scalar (lora_ptr, lbaseOf s0 li split_n_length l0_stride lora_k_stride))
      ∧ s'.regs .ptr [] "c_ptr" = some (Tile.scalar (out_ptr, cbaseOf s0 split_n_length cm_stride cn_stride slice_offset))
      ∧ s'.readMem input_ptr = s0.readMem input_ptr ∧ s'.readMem lora_ptr = s0.readMem lora_ptr
      ∧ ∀ g : Fin split_n_length, s'.readMem out_ptr (cOff s0 split_n_length cm_stride cn_stride slice_offset g.val)
          = if (c*BLOCK_N ≤ g.val ∧ g.val < c*BLOCK_N+BLOCK_N) then
              bgmvFullSpec s0 input_ptr lora_ptr li K split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride BLOCK_K g.val
            else s.readMem out_ptr (cOff s0 split_n_length cm_stride cn_stride slice_offset g.val) := by
  unfold wbBody
  -- current_n
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (currentn_eval BLOCK_N c s hon))]
  set s1 := (s.setReg "n" .nat [] (Tile.scalar (c*BLOCK_N))).setReg "current_n" .nat [BLOCK_N]
      (Tile.vec (fun i : Fin BLOCK_N => c*BLOCK_N + i.val)) with hs1
  have hcn1 : s1.regs .nat [BLOCK_N] "current_n" = some (Tile.vec (fun i : Fin BLOCK_N => c*BLOCK_N + i.val)) := by simp [hs1]
  have hok1 : s1.regs .nat [BLOCK_K] "offset_k" = some (Tile.vec (fun j : Fin BLOCK_K => j.val)) := by
    rw [hs1]; simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_k":RegName) ≠ "current_n" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_k":RegName) ≠ "n" by decide)]; exact hok
  -- b_ptr_mask
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (bmask_eval K split_n_length BLOCK_N BLOCK_K c s1 hcn1 hok1))]
  set s2 := s1.setReg "b_ptr_mask" .bool [BLOCK_N, BLOCK_K]
      ⟨fun idx : TileIndex [BLOCK_N, BLOCK_K] => (decide (c*BLOCK_N + idx.1.val < split_n_length) && decide (idx.2.1.val < K))⟩ with hs2
  have hcn2 : s2.regs .nat [BLOCK_N] "current_n" = some (Tile.vec (fun i : Fin BLOCK_N => c*BLOCK_N + i.val)) := by
    rw [hs2]; simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("current_n":RegName) ≠ "b_ptr_mask" by decide)]; exact hcn1
  -- c_mask
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (cmask_eval split_n_length BLOCK_N c s2 hcn2))]
  set s3 := s2.setReg "c_mask" .bool [BLOCK_N] (Tile.vec (fun i : Fin BLOCK_N => decide (c*BLOCK_N + i.val < split_n_length))) with hs3
  have hbp3 : s3.regs .ptr [] "b_ptr" = some (Tile.scalar (lora_ptr, lbaseOf s0 li split_n_length l0_stride lora_k_stride)) := by
    rw [hs3, hs2, hs1]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("b_ptr":RegName) ≠ "c_mask" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("b_ptr":RegName) ≠ "b_ptr_mask" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("b_ptr":RegName) ≠ "current_n" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("b_ptr":RegName) ≠ "n" by decide)]
    exact hbp
  have hcn3 : s3.regs .nat [BLOCK_N] "current_n" = some (Tile.vec (fun i : Fin BLOCK_N => c*BLOCK_N + i.val)) := by
    rw [hs3]; simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("current_n":RegName) ≠ "c_mask" by decide)]; exact hcn2
  have hok3 : s3.regs .nat [BLOCK_K] "offset_k" = some (Tile.vec (fun j : Fin BLOCK_K => j.val)) := by
    rw [hs3, hs2]; simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_k":RegName) ≠ "c_mask" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_k":RegName) ≠ "b_ptr_mask" by decide)]; exact hok1
  have hbm3 : s3.regs .bool [BLOCK_N, BLOCK_K] "b_ptr_mask"
      = some ⟨fun idx : TileIndex [BLOCK_N, BLOCK_K] => (decide (c*BLOCK_N + idx.1.val < split_n_length) && decide (idx.2.1.val < K))⟩ := by
    rw [hs3]; simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("b_ptr_mask":RegName) ≠ "c_mask" by decide)]
    rw [hs2]; exact BlockState.setReg_same _ _ _ _ _
  -- tiled_b
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (tiledb_eval lora_ptr (lbaseOf s0 li split_n_length l0_stride lora_k_stride) K split_n_length lora_k_stride lora_n_stride BLOCK_N BLOCK_K c s3 hbp3 hcn3 hok3 hbm3))]
  set s4 := s3.setReg "tiled_b" .real [BLOCK_N, BLOCK_K]
      ⟨fun idx : TileIndex [BLOCK_N, BLOCK_K] =>
          some (if c*BLOCK_N + idx.1.val < split_n_length ∧ idx.2.1.val < K then
            s3.readMem lora_ptr ((lbaseOf s0 li split_n_length l0_stride lora_k_stride) + (c*BLOCK_N + idx.1.val)*lora_k_stride + idx.2.1.val*lora_n_stride) else 0.0)⟩ with hs4
  have hs3rm : s3.readMem = s.readMem := by funext rg ofs; rw [hs3, hs2, hs1]; simp only [BlockState.setReg_readMem]
  have hs4ta : s4.regs .real [BLOCK_K] "tiled_a"
      = some ⟨fun j : TileIndex [BLOCK_K] => some (if j.1.val < K then aElem s0 input_ptr xm_stride xk_stride j.1.val else 0)⟩ := by
    rw [hs4, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("tiled_a":RegName) ≠ "tiled_b" by decide), hs3, hs2, hs1]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("tiled_a":RegName) ≠ "c_mask" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("tiled_a":RegName) ≠ "b_ptr_mask" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("tiled_a":RegName) ≠ "current_n" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("tiled_a":RegName) ≠ "n" by decide)]
    exact hta
  have hs4tb : s4.regs .real [BLOCK_N, BLOCK_K] "tiled_b"
      = some ⟨fun idx : TileIndex [BLOCK_N, BLOCK_K] =>
          some (if c*BLOCK_N + idx.1.val < split_n_length ∧ idx.2.1.val < K then
            s4.readMem lora_ptr ((l0_stride * li + s0.pids 0 * split_n_length * lora_k_stride) + (c*BLOCK_N + idx.1.val)*lora_k_stride + idx.2.1.val*lora_n_stride) else 0.0)⟩ := by
    rw [hs4]; rw [BlockState.setReg_same]
    apply congrArg some; ext idx
    have : s4.readMem lora_ptr = s3.readMem lora_ptr := by rw [hs4]; funext ofs; simp [BlockState.setReg_readMem]
    rw [this]; rfl
  have hs4pids : s4.pids = s0.pids := by rw [hs4, hs3, hs2, hs1]; simp [hpids]
  have hs4rm : s4.readMem = s.readMem := by funext rg ofs; rw [hs4]; simp only [BlockState.setReg_readMem]; exact congrFun (congrFun hs3rm rg) ofs
  -- accumulator
  erw [stepStmts.cons_some (stepStmt_assign_eq_some
    (acc_eval input_ptr lora_ptr li K split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride BLOCK_N BLOCK_K c s4 s0
      (by rw [hs4rm]; exact hrl) hs4ta hs4tb))]
  set s5 := s4.setReg "accumulator" .real (TileShape.reduceShape [BLOCK_N, BLOCK_K] ⟨1, by simp⟩ Bool.false)
      ⟨fun i : TileIndex [BLOCK_N] =>
          some (bgmvFullSpec s0 input_ptr lora_ptr li K split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride BLOCK_K (c*BLOCK_N + i.1.val))⟩ with hs5
  have hcn5 : s5.regs .nat [BLOCK_N] "current_n" = some (Tile.vec (fun i : Fin BLOCK_N => c*BLOCK_N + i.val)) := by
    rw [hs5, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("current_n":RegName) ≠ "accumulator" by decide), hs4,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("current_n":RegName) ≠ "tiled_b" by decide)]; exact hcn3
  have hcm5 : s5.regs .bool [BLOCK_N] "c_mask" = some (Tile.vec (fun i : Fin BLOCK_N => decide (c*BLOCK_N + i.val < split_n_length))) := by
    rw [hs5, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("c_mask":RegName) ≠ "accumulator" by decide), hs4,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("c_mask":RegName) ≠ "tiled_b" by decide), hs3]
    exact BlockState.setReg_same _ _ _ _ _
  have hcp5 : s5.regs .ptr [] "c_ptr" = some (Tile.scalar (out_ptr, cbaseOf s0 split_n_length cm_stride cn_stride slice_offset)) := by
    rw [hs5, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("c_ptr":RegName) ≠ "accumulator" by decide), hs4,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("c_ptr":RegName) ≠ "tiled_b" by decide), hs3, hs2, hs1]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("c_ptr":RegName) ≠ "c_mask" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("c_ptr":RegName) ≠ "b_ptr_mask" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("c_ptr":RegName) ≠ "current_n" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("c_ptr":RegName) ≠ "n" by decide)]
    exact hcp
  have hacc5 : s5.regs .real [BLOCK_N] "accumulator"
      = some ⟨fun i : TileIndex [BLOCK_N] => some (bgmvFullSpec s0 input_ptr lora_ptr li K split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride BLOCK_K (c*BLOCK_N + i.1.val))⟩ := by
    rw [hs5]; exact BlockState.setReg_same _ _ _ _ _
  -- store
  rw [stepStmts.cons_some (store_step out_ptr (cbaseOf s0 split_n_length cm_stride cn_stride slice_offset) cn_stride BLOCK_N c s5
    (fun i : Fin BLOCK_N => bgmvFullSpec s0 input_ptr lora_ptr li K split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride BLOCK_K (c*BLOCK_N + i.val))
    (fun i : Fin BLOCK_N => decide (c*BLOCK_N + i.val < split_n_length)) hcp5 hcn5 hacc5 hcm5)]
  rw [stepStmts.nil]
  -- now assemble s'
  set sF := (TileShape.allIndices [BLOCK_N]).foldl
      (fun acc (i : TileIndex [BLOCK_N]) =>
        if decide (c*BLOCK_N + i.1.val < split_n_length) then
          acc.writeMem out_ptr (cbaseOf s0 split_n_length cm_stride cn_stride slice_offset + (c*BLOCK_N + i.1.val)*cn_stride)
            (bgmvFullSpec s0 input_ptr lora_ptr li K split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride BLOCK_K (c*BLOCK_N + i.1.val))
        else acc) s5 with hsF
  have hs5pids : s5.pids = s0.pids := by rw [hs5, BlockState.setReg_pids]; exact hs4pids
  have hs5rmAll : ∀ (rg : RegionName) (ofs : Nat), s5.readMem rg ofs = s.readMem rg ofs := by
    intro rg ofs; rw [hs5]; simp only [BlockState.setReg_readMem]; exact congrFun (congrFun hs4rm rg) ofs
  -- register facts at s5 for offset_k/offset_n/tiled_a/b_ptr
  have hok5 : s5.regs .nat [BLOCK_K] "offset_k" = some (Tile.vec (fun j : Fin BLOCK_K => j.val)) := by
    rw [hs5, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_k":RegName) ≠ "accumulator" by decide), hs4,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_k":RegName) ≠ "tiled_b" by decide)]; exact hok3
  have hon5 : s5.regs .nat [BLOCK_N] "offset_n" = some (Tile.vec (fun i : Fin BLOCK_N => i.val)) := by
    rw [hs5, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_n":RegName) ≠ "accumulator" by decide), hs4,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_n":RegName) ≠ "tiled_b" by decide), hs3, hs2, hs1]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_n":RegName) ≠ "c_mask" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_n":RegName) ≠ "b_ptr_mask" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_n":RegName) ≠ "current_n" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_n":RegName) ≠ "n" by decide)]
    exact hon
  have hta5 : s5.regs .real [BLOCK_K] "tiled_a"
      = some ⟨fun j : TileIndex [BLOCK_K] => some (if j.1.val < K then aElem s0 input_ptr xm_stride xk_stride j.1.val else 0)⟩ := by
    rw [hs5, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("tiled_a":RegName) ≠ "accumulator" by decide)]; exact hs4ta
  have hbp5 : s5.regs .ptr [] "b_ptr" = some (Tile.scalar (lora_ptr, lbaseOf s0 li split_n_length l0_stride lora_k_stride)) := by
    rw [hs5, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("b_ptr":RegName) ≠ "accumulator" by decide), hs4,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("b_ptr":RegName) ≠ "tiled_b" by decide)]; exact hbp3
  refine ⟨sF, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hsF, foldl_store_pids]; exact hs5pids
  · rw [hsF, foldl_store_regs]; exact hok5
  · rw [hsF, foldl_store_regs]; exact hon5
  · rw [hsF, foldl_store_regs]; exact hta5
  · rw [hsF, foldl_store_regs]; exact hbp5
  · rw [hsF, foldl_store_regs]; exact hcp5
  · funext ofs; rw [hsF, foldl_store_other_region _ _ _ _ _ _ _ (Ne.symm hoi), hs5rmAll input_ptr ofs, hri]
  · funext ofs; rw [hsF, foldl_store_other_region _ _ _ _ _ _ _ (Ne.symm hol), hs5rmAll lora_ptr ofs, hrl]
  · -- the readback
    intro g
    rw [hsF, cOff_eq]
    by_cases hin : (c*BLOCK_N ≤ g.val ∧ g.val < c*BLOCK_N+BLOCK_N)
    · rw [if_pos hin]
      set jk : Fin BLOCK_N := ⟨g.val - c*BLOCK_N, by omega⟩ with hjk_def
      have hjk : c*BLOCK_N + jk.val = g.val := by simp only [hjk_def]; omega
      rw [foldl_store_at
            (fun i : TileIndex [BLOCK_N] => cbaseOf s0 split_n_length cm_stride cn_stride slice_offset + (c*BLOCK_N + i.1.val)*cn_stride)
            (fun i : TileIndex [BLOCK_N] => bgmvFullSpec s0 input_ptr lora_ptr li K split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride BLOCK_K (c*BLOCK_N + i.1.val))
            (fun i : TileIndex [BLOCK_N] => decide (c*BLOCK_N + i.1.val < split_n_length))
            (cbaseOf s0 split_n_length cm_stride cn_stride slice_offset + g.val*cn_stride) (TileShape.allIndices [BLOCK_N]) s5
            (jk, PUnit.unit) (TileShape.mem_allIndices [BLOCK_N] _)
            (by simp only [hjk_def]; rw [hjk]; exact decide_eq_true g.isLt)
            (by simp only [hjk_def]; rw [hjk])
            (by intro b hb hmb hofb
                have hbsnl : c*BLOCK_N + b.1.val < split_n_length := by simpa using hmb
                have heq : (⟨c*BLOCK_N + b.1.val, hbsnl⟩ : Fin split_n_length) = ⟨g.val, g.isLt⟩ := by
                  apply hinj; simp only [cOff_eq]; omega
                have : c*BLOCK_N + b.1.val = g.val := by have := congrArg Fin.val heq; simpa using this
                apply Prod.ext
                · apply Fin.ext; simp only [hjk_def]; omega
                · rfl)
            (TileShape.allIndices_nodup [BLOCK_N])]
      simp only [hjk_def]; rw [hjk]
    · rw [if_neg hin]
      rw [foldl_store_preserve
            (fun i : TileIndex [BLOCK_N] => cbaseOf s0 split_n_length cm_stride cn_stride slice_offset + (c*BLOCK_N + i.1.val)*cn_stride)
            (fun i : TileIndex [BLOCK_N] => bgmvFullSpec s0 input_ptr lora_ptr li K split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride BLOCK_K (c*BLOCK_N + i.1.val))
            (fun i : TileIndex [BLOCK_N] => decide (c*BLOCK_N + i.1.val < split_n_length))
            (cbaseOf s0 split_n_length cm_stride cn_stride slice_offset + g.val*cn_stride) (TileShape.allIndices [BLOCK_N]) s5
            (by intro b hb hmb hofb
                have hbsnl : c*BLOCK_N + b.1.val < split_n_length := by simpa using hmb
                have heq : (⟨c*BLOCK_N + b.1.val, hbsnl⟩ : Fin split_n_length) = ⟨g.val, g.isLt⟩ := by
                  apply hinj; simp only [cOff_eq]; omega
                have : c*BLOCK_N + b.1.val = g.val := by have := congrArg Fin.val heq; simpa using this
                omega)]
      rw [hs5rmAll out_ptr _]


end Step2

namespace Loop
open VeriTile.Triton ScratchBgmv Step2

noncomputable def wbInv
    (input_ptr lora_ptr out_ptr : RegionName)
    (li K split_n_length xm_stride xk_stride l0_stride lora_k_stride
      lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K : Nat)
    (s0 sorig : BlockState) (i : Nat) (s : BlockState) : Prop :=
  s.pids = s0.pids ∧ (BLOCK_N ∣ i) ∧
  s.regs .nat [BLOCK_K] "offset_k" = some (Tile.vec (fun j : Fin BLOCK_K => j.val)) ∧
  s.regs .nat [BLOCK_N] "offset_n" = some (Tile.vec (fun i : Fin BLOCK_N => i.val)) ∧
  s.regs .real [BLOCK_K] "tiled_a"
      = some ⟨fun j : TileIndex [BLOCK_K] => some (if j.1.val < K then aElem s0 input_ptr xm_stride xk_stride j.1.val else 0)⟩ ∧
  s.regs .ptr [] "b_ptr" = some (Tile.scalar (lora_ptr, lbaseOf s0 li split_n_length l0_stride lora_k_stride)) ∧
  s.regs .ptr [] "c_ptr" = some (Tile.scalar (out_ptr, cbaseOf s0 split_n_length cm_stride cn_stride slice_offset)) ∧
  s.readMem input_ptr = s0.readMem input_ptr ∧ s.readMem lora_ptr = s0.readMem lora_ptr ∧
  (∀ g : Fin split_n_length, s.readMem out_ptr (cOff s0 split_n_length cm_stride cn_stride slice_offset g.val)
    = if g.val < i then
        bgmvFullSpec s0 input_ptr lora_ptr li K split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride BLOCK_K g.val
      else sorig.readMem out_ptr (cOff s0 split_n_length cm_stride cn_stride slice_offset g.val))

theorem wbInv_step
    (input_ptr lora_ptr out_ptr : RegionName)
    (li K split_n_length xm_stride xk_stride l0_stride lora_k_stride
      lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K : Nat)
    (s0 sorig : BlockState) (hBN : 0 < BLOCK_N)
    (hinj : Function.Injective (fun g : Fin split_n_length =>
      cOff s0 split_n_length cm_stride cn_stride slice_offset g.val))
    (hoi : out_ptr ≠ input_ptr) (hol : out_ptr ≠ lora_ptr)
    (i : Nat) (s : BlockState) (hlt : i < split_n_length)
    (hinv : wbInv input_ptr lora_ptr out_ptr li K split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K s0 sorig i s) :
    ∃ s', stepStmts (wbBody lora_k_stride lora_n_stride cn_stride K split_n_length BLOCK_N BLOCK_K)
        (s.setReg "n" .nat [] (Tile.scalar i)) = some s'
      ∧ wbInv input_ptr lora_ptr out_ptr li K split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K s0 sorig (i+BLOCK_N) s' := by
  obtain ⟨hpids, ⟨c, hc⟩, hok, hon, hta, hbp, hcp, hri, hrl, hmem⟩ := hinv
  subst hc
  rw [show BLOCK_N*c = c*BLOCK_N from Nat.mul_comm BLOCK_N c]
  obtain ⟨s', hstep, hp', hok', hon', hta', hbp', hcp', hri', hrl', hread'⟩ :=
    wbStep input_ptr lora_ptr out_ptr li K split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K
      s0 s c hinj hoi hol hpids hri hrl hok hon hta hbp hcp
  refine ⟨s', hstep, hp', ⟨c+1, by ring⟩, hok', hon', hta', hbp', hcp', ?_, ?_, ?_⟩
  · exact hri'
  · exact hrl'
  · intro g
    rw [hread' g]
    by_cases hin : (c*BLOCK_N ≤ g.val ∧ g.val < c*BLOCK_N+BLOCK_N)
    · rw [if_pos hin, if_pos (show g.val < c*BLOCK_N + BLOCK_N by omega)]
    · rw [if_neg hin, hmem g]
      have hcomm : BLOCK_N*c = c*BLOCK_N := Nat.mul_comm BLOCK_N c
      by_cases hlt2 : g.val < BLOCK_N*c
      · rw [if_pos hlt2, if_pos (show g.val < c*BLOCK_N+BLOCK_N by omega)]
      · rw [if_neg hlt2, if_neg (show ¬ g.val < c*BLOCK_N+BLOCK_N by omega)]


theorem wb_forRange
    (input_ptr lora_ptr out_ptr : RegionName)
    (li K split_n_length xm_stride xk_stride l0_stride lora_k_stride
      lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K : Nat)
    (s0 sorig : BlockState) (hBN : 0 < BLOCK_N)
    (hinj : Function.Injective (fun g : Fin split_n_length =>
      cOff s0 split_n_length cm_stride cn_stride slice_offset g.val))
    (hoi : out_ptr ≠ input_ptr) (hol : out_ptr ≠ lora_ptr)
    (s : BlockState)
    (hinit : wbInv input_ptr lora_ptr out_ptr li K split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K s0 sorig 0 s) :
    ∃ final s', stepStmt (.forRange "n" 0 split_n_length BLOCK_N (wbBody lora_k_stride lora_n_stride cn_stride K split_n_length BLOCK_N BLOCK_K)) s = some s'
      ∧ split_n_length ≤ final
      ∧ wbInv input_ptr lora_ptr out_ptr li K split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K s0 sorig final s' := by
  exact forRange_inv (idx := "n") (start := 0) (stop := split_n_length) (step := BLOCK_N)
    (Nat.pos_iff_ne_zero.mp hBN) hinit
    (fun i st hlt hP => wbInv_step input_ptr lora_ptr out_ptr li K split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K s0 sorig hBN hinj hoi hol i st hlt hP)

end Loop

namespace Prefix2
open VeriTile.Triton ScratchBgmv Step2 Loop
set_option linter.unusedVariables false

-- tiled_a load eval (region masked)
theorem tileda_eval (input_ptr : RegionName) (K xm_stride xk_stride BLOCK_K : Nat) (s s0 : BlockState)
    (hpids : s.pids = s0.pids) (hrm : s.readMem input_ptr = s0.readMem input_ptr)
    (hcb : s.regs .nat [] "cur_batch" = some (Tile.scalar (s0.pids 1)))
    (hok : s.regs .nat [BLOCK_K] "offset_k" = some (Tile.vec (fun j : Fin BLOCK_K => j.val))) :
    evalOp (Op.load TileDType.real
        (MemAccess.region input_ptr
          (Op.add NumericDType.nat Broadcast.scalarL
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "cur_batch") (Op.constNat xm_stride))
            (Op.mul NumericDType.nat Broadcast.scalarR (Op.ref TileDType.nat [BLOCK_K] "offset_k") (Op.constNat xk_stride))))
        (MaskOpt.maskOther
          (Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref TileDType.nat [BLOCK_K] "offset_k") (Op.constNat K))
          ((Op.const (0:ℝ)).broadcast [BLOCK_K]))) s
      = some ⟨fun j : TileIndex [BLOCK_K] => some (if j.1.val < K then aElem s0 input_ptr xm_stride xk_stride j.1.val else 0)⟩ := by
  simp only [evalOp, hcb, hok, Tile.bop, Tile.cop, Tile.scalar_data, Tile.vec,
    Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL, Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR,
    NumericDType.add, NumericDType.mul, ComparableDType.lt, BlockState.readMemValue_real]
  apply congrArg some
  ext j
  simp only [aElem, hpids]
  by_cases h : j.1.val < K <;> simp [h, hrm]


-- b_ptr eval (with int casts); requires lora_index = Int.ofNat li
theorem bptr_eval (lora_ptr : RegionName) (li l0_stride split_n_length lora_k_stride : Nat) (s s0 : BlockState)
    (hpids : s.pids = s0.pids)
    (hps : s.regs .nat [] "pid_sn" = some (Tile.scalar (s0.pids 0)))
    (hlx : s.regs .int [] "lora_index" = some (Tile.scalar (Int.ofNat li))) :
    evalOp (Op.ptrAdd Broadcast.nil (Op.ptrBase lora_ptr)
        (Op.add NumericDType.nat Broadcast.nil
          (Op.mul NumericDType.int Broadcast.nil (Op.constNat l0_stride).castNatToInt
              (Op.ref TileDType.int [] "lora_index")).castIntToNat
          (Op.mul NumericDType.nat Broadcast.nil
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "pid_sn") (Op.constNat split_n_length))
            (Op.constNat lora_k_stride)))) s
      = some (Tile.scalar (lora_ptr, lbaseOf s0 li split_n_length l0_stride lora_k_stride)) := by
  rw [evalOp_ptrAdd, evalOp_ptrBase]
  simp only [evalOp, hps, hlx, Tile.bop, Tile.uop, Tile.ptrAdd, Tile.scalar,
    Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul]
  apply congrArg some
  apply Tile.ext; intro idx
  simp only [Tile.ptrAdd, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex, lbaseOf]
  congr 1
  rw [Nat.zero_add]
  simp only [Int.ofNat_eq_natCast, ← Nat.cast_mul, Int.toNat_natCast]

-- c_ptr eval
theorem cptr_eval (out_ptr : RegionName) (split_n_length cm_stride cn_stride slice_offset : Nat) (s s0 : BlockState)
    (hpids : s.pids = s0.pids)
    (hcb : s.regs .nat [] "cur_batch" = some (Tile.scalar (s0.pids 1)))
    (hps : s.regs .nat [] "pid_sn" = some (Tile.scalar (s0.pids 0))) :
    evalOp (Op.ptrAdd Broadcast.nil (Op.ptrBase out_ptr)
        (Op.add NumericDType.nat Broadcast.nil
          (Op.add NumericDType.nat Broadcast.nil
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "cur_batch") (Op.constNat cm_stride))
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "pid_sn") (Op.constNat split_n_length)))
          (Op.mul NumericDType.nat Broadcast.nil (Op.constNat slice_offset) (Op.constNat cn_stride)))) s
      = some (Tile.scalar (out_ptr, cbaseOf s0 split_n_length cm_stride cn_stride slice_offset)) := by
  rw [evalOp_ptrAdd, evalOp_ptrBase]
  simp only [evalOp, hcb, hps, Tile.bop, Tile.ptrAdd, Tile.scalar,
    Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul]
  apply congrArg some
  apply Tile.ext; intro idx
  simp only [Tile.ptrAdd, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex, cbaseOf]
  congr 1
  rw [Nat.zero_add]


-- the 5 setup statements (offset_k, offset_n, tiled_a, b_ptr, c_ptr) of prefixGuardBody
def setupStmts (input_ptr lora_ptr out_ptr : RegionName)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride
      lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K : Nat) : List Stmt :=
  (prefixGuardBody input_ptr lora_ptr out_ptr K split_n_length xm_stride xk_stride
    l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K).dropLast

theorem prefixGuardBody_eq (input_ptr lora_ptr out_ptr : RegionName)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride
      lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K : Nat) :
    prefixGuardBody input_ptr lora_ptr out_ptr K split_n_length xm_stride xk_stride
      l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K
    = setupStmts input_ptr lora_ptr out_ptr K split_n_length xm_stride xk_stride
        l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K
      ++ [Stmt.forRange "n" 0 split_n_length BLOCK_N (wbBody lora_k_stride lora_n_stride cn_stride K split_n_length BLOCK_N BLOCK_K)] := by
  rfl

theorem preLoop (input_ptr lora_ptr out_ptr : RegionName)
    (li K split_n_length xm_stride xk_stride l0_stride lora_k_stride
      lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K : Nat)
    (s0 s : BlockState)
    (hpids : s.pids = s0.pids) (hrm : s.readMem = s0.readMem)
    (hps : s.regs .nat [] "pid_sn" = some (Tile.scalar (s0.pids 0)))
    (hcb : s.regs .nat [] "cur_batch" = some (Tile.scalar (s0.pids 1)))
    (hlx : s.regs .int [] "lora_index" = some (Tile.scalar (Int.ofNat li))) :
    ∃ s', stepStmts (setupStmts input_ptr lora_ptr out_ptr K split_n_length xm_stride xk_stride
        l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K) s = some s'
      ∧ wbInv input_ptr lora_ptr out_ptr li K split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K s0 s 0 s' := by
  unfold setupStmts prefixGuardBody
  simp only [List.dropLast]
  -- offset_k
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (by simp [evalOp_arange] : evalOp (Op.arange BLOCK_K) s = some (Tile.vec (fun j : Fin BLOCK_K => j.val))))]
  set s1 := s.setReg "offset_k" .nat [BLOCK_K] (Tile.vec (fun j : Fin BLOCK_K => j.val)) with hs1
  -- offset_n
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (by simp [evalOp_arange] : evalOp (Op.arange BLOCK_N) s1 = some (Tile.vec (fun i : Fin BLOCK_N => i.val))))]
  set s2 := s1.setReg "offset_n" .nat [BLOCK_N] (Tile.vec (fun i : Fin BLOCK_N => i.val)) with hs2
  have hcb2 : s2.regs .nat [] "cur_batch" = some (Tile.scalar (s0.pids 1)) := by
    rw [hs2, hs1]; simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("cur_batch":RegName) ≠ "offset_n" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("cur_batch":RegName) ≠ "offset_k" by decide)]; exact hcb
  have hok2 : s2.regs .nat [BLOCK_K] "offset_k" = some (Tile.vec (fun j : Fin BLOCK_K => j.val)) := by
    rw [hs2]; simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_k":RegName) ≠ "offset_n" by decide)]
    rw [hs1]; exact BlockState.setReg_same _ _ _ _ _
  have hpids2 : s2.pids = s0.pids := by rw [hs2, hs1]; simp [hpids]
  have hrm2i : s2.readMem input_ptr = s0.readMem input_ptr := by rw [hs2, hs1]; funext ofs; simp only [BlockState.setReg_readMem]; exact congrFun (congrFun hrm input_ptr) ofs
  -- tiled_a
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (tileda_eval input_ptr K xm_stride xk_stride BLOCK_K s2 s0 hpids2 hrm2i hcb2 hok2))]
  set s3 := s2.setReg "tiled_a" .real [BLOCK_K] ⟨fun j : TileIndex [BLOCK_K] => some (if j.1.val < K then aElem s0 input_ptr xm_stride xk_stride j.1.val else 0)⟩ with hs3
  have hps3 : s3.regs .nat [] "pid_sn" = some (Tile.scalar (s0.pids 0)) := by
    rw [hs3, hs2, hs1]; simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("pid_sn":RegName) ≠ "tiled_a" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("pid_sn":RegName) ≠ "offset_n" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("pid_sn":RegName) ≠ "offset_k" by decide)]; exact hps
  have hlx3 : s3.regs .int [] "lora_index" = some (Tile.scalar (Int.ofNat li)) := by
    rw [hs3, hs2, hs1]; simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("lora_index":RegName) ≠ "tiled_a" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("lora_index":RegName) ≠ "offset_n" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("lora_index":RegName) ≠ "offset_k" by decide)]; exact hlx
  have hpids3 : s3.pids = s0.pids := by rw [hs3]; simp [hpids2]
  -- b_ptr
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (bptr_eval lora_ptr li l0_stride split_n_length lora_k_stride s3 s0 hpids3 hps3 hlx3))]
  set s4 := s3.setReg "b_ptr" .ptr [] (Tile.scalar (lora_ptr, lbaseOf s0 li split_n_length l0_stride lora_k_stride)) with hs4
  have hps4 : s4.regs .nat [] "pid_sn" = some (Tile.scalar (s0.pids 0)) := by
    rw [hs4, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("pid_sn":RegName) ≠ "b_ptr" by decide)]; exact hps3
  have hcb4 : s4.regs .nat [] "cur_batch" = some (Tile.scalar (s0.pids 1)) := by
    rw [hs4, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("cur_batch":RegName) ≠ "b_ptr" by decide), hs3]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("cur_batch":RegName) ≠ "tiled_a" by decide)]; exact hcb2
  have hpids4 : s4.pids = s0.pids := by rw [hs4]; simp [hpids3]
  -- c_ptr
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (cptr_eval out_ptr split_n_length cm_stride cn_stride slice_offset s4 s0 hpids4 hcb4 hps4))]
  rw [stepStmts.nil]
  set s5 := s4.setReg "c_ptr" .ptr [] (Tile.scalar (out_ptr, cbaseOf s0 split_n_length cm_stride cn_stride slice_offset)) with hs5
  refine ⟨s5, rfl, ?_, ⟨0, rfl⟩, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hs5]; simp [hpids4]
  · rw [hs5, hs4, hs3]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_k":RegName) ≠ "c_ptr" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_k":RegName) ≠ "b_ptr" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_k":RegName) ≠ "tiled_a" by decide)]; exact hok2
  · rw [hs5, hs4, hs3, hs2]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_n":RegName) ≠ "c_ptr" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_n":RegName) ≠ "b_ptr" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_n":RegName) ≠ "tiled_a" by decide)]
    exact BlockState.setReg_same _ _ _ _ _
  · rw [hs5, hs4]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("tiled_a":RegName) ≠ "c_ptr" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("tiled_a":RegName) ≠ "b_ptr" by decide)]
    rw [hs3]; exact BlockState.setReg_same _ _ _ _ _
  · rw [hs5, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("b_ptr":RegName) ≠ "c_ptr" by decide)]
    rw [hs4]; exact BlockState.setReg_same _ _ _ _ _
  · rw [hs5]; exact BlockState.setReg_same _ _ _ _ _
  · rw [hs5, hs4, hs3, hs2, hs1]; funext ofs; simp only [BlockState.setReg_readMem]; exact congrFun (congrFun hrm input_ptr) ofs
  · rw [hs5, hs4, hs3, hs2, hs1]; funext ofs; simp only [BlockState.setReg_readMem]; exact congrFun (congrFun hrm lora_ptr) ofs
  · intro g; rw [if_neg (by omega)]
    rw [hs5, hs4, hs3, hs2, hs1]; simp only [BlockState.setReg_readMem]

end Prefix2

namespace Final
open VeriTile.Triton ScratchBgmv Step2 Loop Prefix2

theorem bgmv_full_correct
    (input_ptr lora_ptr out_ptr : RegionName) (lora_indices : Region .int)
    (li K split_n_length xm_stride xk_stride l0_stride lora_k_stride
      lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K : Nat)
    (s s' : BlockState) (hBN : 0 < BLOCK_N)
    (hoi : out_ptr ≠ input_ptr) (hol : out_ptr ≠ lora_ptr)
    (hcn : 0 < cn_stride)
    (hlx : s.readMemValue .int (Region.cast lora_indices) (s.pids 1) = Int.ofNat li)
    (hExec : exec (bgmv_full input_ptr lora_ptr out_ptr lora_indices K split_n_length xm_stride xk_stride
        l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K) s = some s') :
    ∀ g : Fin split_n_length, s'.readMem out_ptr (cOff s split_n_length cm_stride cn_stride slice_offset g.val)
      = bgmvFullSpec s input_ptr lora_ptr li K split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride BLOCK_K g.val := by
  rw [exec, body_decomp] at hExec
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (by simp [evalOp_programId] : evalOp (Op.programId 0) s = some (Tile.scalar (s.pids 0))))] at hExec
  set s1 := s.setReg "pid_sn" .nat [] (Tile.scalar (s.pids 0)) with hs1
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (by simp [evalOp_programId, hs1] : evalOp (Op.programId 1) s1 = some (Tile.scalar (s.pids 1))))] at hExec
  set s2 := s1.setReg "cur_batch" .nat [] (Tile.scalar (s.pids 1)) with hs2
  have hcb2 : s2.regs .nat [] "cur_batch" = some (Tile.scalar (s.pids 1)) := by rw [hs2]; exact BlockState.setReg_same _ _ _ _ _
  have hloadeval : evalOp (Op.load TileDType.int (MemAccess.region lora_indices (Op.ref TileDType.nat [] "cur_batch")) MaskOpt.none) s2
      = some (Tile.scalar (Int.ofNat li)) := by
    simp only [evalOp_load_region_none, evalOp_ref, hcb2, Option.bind_some]
    refine congrArg some (Tile.ext ?_); intro idx
    show s2.readMemValue .int (Region.cast lora_indices) (s.pids 1) = _
    rw [hs2, hs1]; simp only [BlockState.setReg_readMemValue]; rw [hlx]; rfl
  rw [stepStmts.cons_some (stepStmt_assign_eq_some hloadeval)] at hExec
  set s3 := s2.setReg "lora_index" .int [] (Tile.scalar (Int.ofNat li)) with hs3
  have hlx3 : s3.regs .int [] "lora_index" = some (Tile.scalar (Int.ofNat li)) := by rw [hs3]; exact BlockState.setReg_same _ _ _ _ _
  have hli : (Int.ofNat li : Int) ≠ -1 := by
    have : (0 : Int) ≤ Int.ofNat li := Int.natCast_nonneg li
    omega
  have hguard : evalOp (Op.ne ComparableDType.int Broadcast.nil (Op.ref TileDType.int [] "lora_index") (Op.constInt (-1))) s3
      = some (Tile.scalar Bool.true) := by
    simp only [evalOp, hlx3, Tile.cop, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex]
    refine congrArg some (Tile.ext ?_); intro idx
    show ComparableDType.int.ne (Int.ofNat li) (-1) = _
    simp only [ComparableDType.ne, Tile.scalar]
    exact decide_eq_true hli
  have hifeval : stepStmt (Stmt.ifThen (Op.ne ComparableDType.int Broadcast.nil (Op.ref TileDType.int [] "lora_index") (Op.constInt (-1)))
      (prefixGuardBody input_ptr lora_ptr out_ptr K split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K)) s3
      = stepStmts (prefixGuardBody input_ptr lora_ptr out_ptr K split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K) s3 := by
    rw [stepStmt]; simp only [hguard, Tile.scalar, Option.bind_some]; rfl
  have hcollapse : stepStmts [Stmt.ifThen (Op.ne ComparableDType.int Broadcast.nil (Op.ref TileDType.int [] "lora_index") (Op.constInt (-1)))
      (prefixGuardBody input_ptr lora_ptr out_ptr K split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K)] s3
      = stepStmts (prefixGuardBody input_ptr lora_ptr out_ptr K split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K) s3 := by
    rw [stepStmts, hifeval]
    cases h : stepStmts (prefixGuardBody input_ptr lora_ptr out_ptr K split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K) s3 <;>
      simp [stepStmts]
  rw [hcollapse] at hExec
  -- s3 has pid_sn, cur_batch, lora_index; pids = s.pids, readMem = s.readMem
  have hpids3 : s3.pids = s.pids := by rw [hs3, hs2, hs1]; simp
  have hrm3 : s3.readMem = s.readMem := by funext rg ofs; rw [hs3, hs2, hs1]; simp only [BlockState.setReg_readMem]
  have hps3 : s3.regs .nat [] "pid_sn" = some (Tile.scalar (s.pids 0)) := by
    rw [hs3, hs2]; simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("pid_sn":RegName) ≠ "lora_index" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("pid_sn":RegName) ≠ "cur_batch" by decide)]
    rw [hs1]; exact BlockState.setReg_same _ _ _ _ _
  have hcb3 : s3.regs .nat [] "cur_batch" = some (Tile.scalar (s.pids 1)) := by
    rw [hs3]; simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("cur_batch":RegName) ≠ "lora_index" by decide)]; exact hcb2
  -- now hExec : stepStmts (prefixGuardBody) s3 = some s'
  rw [prefixGuardBody_eq, stepStmts.append_some_iff] at hExec
  obtain ⟨sp, hsp, hloop⟩ := hExec
  obtain ⟨s'', hsetup, hinvP⟩ := preLoop input_ptr lora_ptr out_ptr li K split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K
    s s3 hpids3 hrm3 hps3 hcb3 hlx3
  rw [hsetup] at hsp
  injection hsp with hsp; subst hsp
  -- now run wb_forRange from hinvP (wbInv ... s s'' 0 s'')
  have hinjs : Function.Injective (fun g : Fin split_n_length =>
      cOff s split_n_length cm_stride cn_stride slice_offset g.val) := by
    have heq : (fun g : Fin split_n_length => cOff s split_n_length cm_stride cn_stride slice_offset g.val)
        = (fun g : Fin split_n_length =>
            (s.pids 1 * cm_stride + s.pids 0 * split_n_length + slice_offset * cn_stride)
              + g.val * cn_stride) := by
      funext g; simp only [cOff]
    rw [heq]; exact affine1D_inj _ cn_stride hcn
  obtain ⟨final, sw, hsw, hle, hinvW⟩ := wb_forRange input_ptr lora_ptr out_ptr li K split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K
    s s3 hBN hinjs hoi hol s'' hinvP
  rw [stepStmts.cons_some hsw, stepStmts.nil] at hloop
  injection hloop with hloop; subst hloop
  obtain ⟨_, _, _, _, _, _, _, _, _, hread⟩ := hinvW
  intro g
  rw [hread g, if_pos (lt_of_lt_of_le g.isLt hle)]

end Final


namespace Final
open VeriTile.Triton ScratchBgmv Step2 Loop Prefix2

theorem bgmv_full_compute_correct
    (input_ptr lora_ptr out_ptr : RegionName) (lora_indices : Region .int)
    (li K split_n_length xm_stride xk_stride l0_stride lora_k_stride
      lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K : Nat)
    (s : BlockState) (hBN : 0 < BLOCK_N)
    (hoi : out_ptr ≠ input_ptr) (hol : out_ptr ≠ lora_ptr)
    (hcn : 0 < cn_stride)
    (hlx : s.readMemValue .int (Region.cast lora_indices) (s.pids 1) = Int.ofNat li) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := bgmv_full input_ptr lora_ptr out_ptr lora_indices K split_n_length xm_stride xk_stride
        l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin split_n_length => True)
        (fun g => (out_ptr, cOff s split_n_length cm_stride cn_stride slice_offset g.val)))
      (expected := fun g : Fin split_n_length =>
        bgmvFullSpec s input_ptr lora_ptr li K split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride BLOCK_K g.val) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [bgmv_full, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro g _
  exact bgmv_full_correct input_ptr lora_ptr out_ptr lora_indices li K split_n_length xm_stride xk_stride
    l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K s s' hBN hoi hol hcn hlx hExec g

/-- **Full output summary** for the general `bgmv_expand_slice` kernel (arbitrary
`split_n_length`, multi-block `for n` loop, signed `-1` sentinel guard): the DSL
surface lowers to the algorithm layer, and the masked GEMV store to the
slice-offset output realizes the genuine rank-`K` reduction
`bgmvFullSpec g = Σ_k (k<K ? A[k] : 0)·(g<split_n_length ∧ k<K ? B[g,k] : 0)` at
every output lane `g < split_n_length`. Requires the active LoRA index
(`lora_index = Int.ofNat li ≥ 0`, so the `-1` early return is not taken),
out-of-place output (`out ≠ input`, `out ≠ lora`), and per-lane output-offset
injectivity. -/
theorem bgmv_full_output_summary
    (input_ptr lora_ptr out_ptr : RegionName) (lora_indices : Region .int)
    (li K split_n_length xm_stride xk_stride l0_stride lora_k_stride
      lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K : Nat)
    (s : BlockState) (hBN : 0 < BLOCK_N)
    (hoi : out_ptr ≠ input_ptr) (hol : out_ptr ≠ lora_ptr)
    (hcn : 0 < cn_stride)
    (hlx : s.readMemValue .int (Region.cast lora_indices) (s.pids 1) = Int.ofNat li) :
    (∃ alg, (bgmv_full input_ptr lora_ptr out_ptr lora_indices K split_n_length xm_stride xk_stride
        l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := bgmv_full input_ptr lora_ptr out_ptr lora_indices K split_n_length xm_stride xk_stride
        l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin split_n_length => True)
        (fun g => (out_ptr, cOff s split_n_length cm_stride cn_stride slice_offset g.val)))
      (expected := fun g : Fin split_n_length =>
        bgmvFullSpec s input_ptr lora_ptr li K split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride BLOCK_K g.val) := by
  refine ⟨?_, bgmv_full_compute_correct input_ptr lora_ptr out_ptr lora_indices li K split_n_length xm_stride xk_stride
    l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K s hBN hoi hol hcn hlx⟩
  simp [bgmv_full, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

end Final

end Full

end VeriTile.Bench.TritonBenchG.BgmvExpandSlice
