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
specification bgmv_full_output_summary
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

/-! ## The `⊨[R]` streaming metadata emit headline (wave-5 S3 MetaEmit genre)

Everything below is purely additive; the exact surface above is untouched.
This is the first consumer of the metadata emit skin
`StreamMetaEmitMasked3DKernelIO₂` (`Stream` + `Meta` + `Emit` capability
stack): the pre-loop `.int` slot load (`lora_index = lora_indices[cur_batch]`,
with the `-1` skip sentinel **kept signed** and gating `writeMask`) composed
with the S3 per-step store genre (the masked GEMV store sits inside the
`for n` loop, one disjoint `BLOCK_N`-lane window per step).

Structure of the `execR R` story: this kernel has **zero rounding events** —
every load and the masked in-loop store are at `.real` (or the typed `.int`
slot read, not a rounding site), and the only casts are the `b_ptr` int
casts `castNatToInt`/`castIntToNat`, which `evalOpR` evaluates exactly. The
whole guarded body therefore collapses verbatim onto the exact stepper, and
the proven `preLoop` / `wbInv` / `wbStep` / `forRange_inv` stack above rides
unchanged; the `⊨[R]` face adds the `TraceSafeR` walk, the per-cell memory
frame, the stream-lane spec bridge — and, new in this genre, a **genuine
sentinel branch**: at `lora_index = -1` the guard skips the body, the io's
write-active window family is empty, and the frame keeps the whole output
untouched (the exact headline above pins `lora_index = Int.ofNat li ≥ 0` and
says nothing about the sentinel program). -/

section IOFace

open Full Full.ScratchBgmv Full.Step2 Full.Loop Full.Prefix2 Full.Final Finset
open scoped VeriTile.Triton.StreamMetaEmitMasked3DKernelIO₂

set_option maxHeartbeats 4000000
set_option maxRecDepth 8000
set_option linter.unusedVariables false

/-! ### Stream geometry: trip count and lane bridge -/

/-- Trip count of `for n in range(0, split_n_length, BLOCK_N)`:
`⌈split_n_length / BLOCK_N⌉`. -/
def bgmvNumSteps (snl BN : Nat) : Nat := (snl + BN - 1) / BN

private theorem bgmvNumSteps_mul_ge (snl BN : Nat) (hB : 0 < BN) :
    snl ≤ bgmvNumSteps snl BN * BN := by
  rcases Nat.eq_zero_or_pos snl with rfl | hN
  · exact Nat.zero_le _
  · unfold bgmvNumSteps
    have heq : snl + BN - 1 = (snl - 1) + BN := by omega
    rw [heq, Nat.add_div_right _ hB]
    have h2 : (snl - 1) % BN + 1 ≤ BN := Nat.mod_lt _ hB
    calc snl = (snl - 1) + 1 := by omega
      _ = (snl - 1) / BN * BN + ((snl - 1) % BN + 1) := by
          rw [← Nat.add_assoc, Nat.div_add_mod']
      _ ≤ (snl - 1) / BN * BN + BN := Nat.add_le_add_left h2 _
      _ = ((snl - 1) / BN + 1) * BN := (Nat.succ_mul _ _).symm

private theorem bgmvStep_lt_numSteps (snl BN i : Nat) (hB : 0 < BN)
    (hi : i < snl) : i / BN < bgmvNumSteps snl BN := by
  have h2 : i / BN * BN < bgmvNumSteps snl BN * BN :=
    Nat.lt_of_le_of_lt (Nat.div_mul_le_self i BN)
      (Nat.lt_of_lt_of_le hi (bgmvNumSteps_mul_ge snl BN hB))
  exact Nat.lt_of_mul_lt_mul_right h2

/-- The `tiled_b`-stream lane feeding output lane `j` at rank key `k`:
`(j, k)` row-major over the `[BLOCK_N, BLOCK_K]` per-step weight tile, via
the shared `Lane2D` bridge. -/
def bTileLane (BLOCK_N BLOCK_K : Nat) (j : Fin BLOCK_N) (k : Fin BLOCK_K) :
    Fin (BLOCK_N * BLOCK_K) :=
  Lane2D.encode (j, k, PUnit.unit)

/-! ### IO signature -/

/-- **Streaming metadata emit IO signature** of the verified `bgmv_full`
config (`EVEN_K = false`, `ADD_INPUTS = false`, no `CAST_TYPE`) on the
metadata-parametrized two-stream per-step emit skin
(`StreamMetaEmitMasked3DKernelIO₂`, style S3: the store sits inside the
`for n` loop). One **`.int`** metadata slot: `lora_index =
lora_indices[cur_batch]`, read at the pid-only cell `cur_batch = pid₁`
(`mwin`) at the kernel's own signed dtype — the `-1` skip sentinel stays
visible in the slot value and gates `writeMask` (the honest sentinel gate:
at `m 0 = -1` no window is write-active and the program stores nothing).
The kernel launches on a 2-D grid `(pid_sn, cur_batch)`; the skin's `pid₂`
slot is unused — every window is constant in it, and the headline still
quantifies over all three pids (the `bgmv_shrink` precedent).

Step `t` of the loop (at `n = t·BLOCK_N`) reads the `[BLOCK_N, BLOCK_K]`
LoRA-B tile and emits the `BLOCK_N`-lane output window; the `BLOCK_K`-lane
input row is the genre's degenerate **static stream** (`read1` ignores `t`:
the kernel loads `tiled_a` once before the loop and register-caches it
across steps — the uniform per-step pin is harmless and keeps the contract
one-shaped). The windows transcribe the kernel's pointer arithmetic
verbatim, with the loaded slot value `m 0` in place of the in-state
`lora_index` read:

* `read1` lane `k` (any step): `cur_batch·xm_stride + k·xk_stride`;
  `mask1`: `k < K` — the masked `EVEN_K = false` load path.
* `read2` step `t`, lane `l = (r, k)` (row-major over
  `[BLOCK_N, BLOCK_K]`, `r = l / BLOCK_K`, `k = l % BLOCK_K`):
  `l0_stride·(m 0).toNat + pid_sn·split_n_length·lora_k_stride
  + (t·BLOCK_N + r)·lora_k_stride + k·lora_n_stride` — the `b_ptr` cell.
  **Negative non-sentinel indices are modeled honestly**: the kernel's
  `b_ptr` arithmetic clamps `l0_stride · lora_index` through `castIntToNat`
  (`Int.toNat`, so any negative product clamps to `0`), and `(m 0).toNat`
  reproduces exactly that clamp — the io does not pretend `m 0 ≥ 0`.
* `write` step `t`, lane `j`: `cur_batch·cm_stride + pid_sn·split_n_length
  + slice_offset·cn_stride + (t·BLOCK_N + j)·cn_stride` — the kernel's
  `c_ptr + current_n·cn_stride` cell (slot-independent).
* `mask2` transcribes `b_ptr_mask` (`current_n < split_n_length &
  offset_k < K`); `writeMask` is `m 0 ≠ -1 ∧ t·BLOCK_N + j <
  split_n_length` — the store's `c_mask` under the sentinel guard. -/
def bgmvExpandSliceIO (input_ptr lora_ptr out_ptr : RegionName)
    (lora_indices : Region .int)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride
      lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K : Nat) :
    StreamMetaEmitMasked3DKernelIO₂ where
  kernel := bgmv_full input_ptr lora_ptr out_ptr lora_indices K
    split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
    cm_stride cn_stride slice_offset BLOCK_N BLOCK_K
  inp1 := input_ptr
  inp2 := lora_ptr
  out := out_ptr
  nMeta := 1
  sty := fun _ => .int
  mbuf := fun _ => Region.cast lora_indices
  mwin := fun _ _ pid₁ _ => pid₁
  T := bgmvNumSteps split_n_length BLOCK_N
  B1 := BLOCK_K
  B2 := BLOCK_N * BLOCK_K
  C := BLOCK_N
  read1 := fun _ pid₁ _ _ _ k => pid₁ * xm_stride + k.val * xk_stride
  read2 := fun pid₀ _ _ m t l =>
    l0_stride * (m ⟨0, Nat.one_pos⟩).toNat
      + pid₀ * split_n_length * lora_k_stride
      + (t.val * BLOCK_N + l.val / BLOCK_K) * lora_k_stride
      + l.val % BLOCK_K * lora_n_stride
  write := fun pid₀ pid₁ _ _ t j =>
    pid₁ * cm_stride + pid₀ * split_n_length + slice_offset * cn_stride
      + (t.val * BLOCK_N + j.val) * cn_stride
  mask1 := fun _ _ _ _ _ k => k.val < K
  mask2 := fun _ _ _ _ t l =>
    t.val * BLOCK_N + l.val / BLOCK_K < split_n_length ∧ l.val % BLOCK_K < K
  writeMask := fun _ _ _ m t j =>
    m ⟨0, Nat.one_pos⟩ ≠ -1 ∧ t.val * BLOCK_N + j.val < split_n_length

/-! ### The stream-lane spec bridge -/

/-- Under the stream pins, the exact stack's `bgmvFullSpec` at the active
global lane `g = t·BLOCK_N + j` **is** the skin-level masked rank-`K` dot of
the streamed tiles (with `li` the clamped slot value feeding the `b_ptr`
base). -/
private theorem bgmvSpec_eq_streamSum
    (input_ptr lora_ptr : RegionName) (s₀ : BlockState) (li : Nat)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride
      lora_n_stride BLOCK_N BLOCK_K : Nat)
    (xs : Fin (bgmvNumSteps split_n_length BLOCK_N) → Fin BLOCK_K → ℝ)
    (ys : Fin (bgmvNumSteps split_n_length BLOCK_N) →
      Fin (BLOCK_N * BLOCK_K) → ℝ)
    (hx : ∀ (t : Fin (bgmvNumSteps split_n_length BLOCK_N)) (k : Fin BLOCK_K),
      k.val < K →
      s₀.readMem input_ptr (s₀.pids 1 * xm_stride + k.val * xk_stride)
        = xs t k)
    (hy : ∀ (t : Fin (bgmvNumSteps split_n_length BLOCK_N))
        (l : Fin (BLOCK_N * BLOCK_K)),
      t.val * BLOCK_N + l.val / BLOCK_K < split_n_length ∧
        l.val % BLOCK_K < K →
      s₀.readMem lora_ptr (l0_stride * li
          + s₀.pids 0 * split_n_length * lora_k_stride
          + (t.val * BLOCK_N + l.val / BLOCK_K) * lora_k_stride
          + l.val % BLOCK_K * lora_n_stride) = ys t l)
    (t : Fin (bgmvNumSteps split_n_length BLOCK_N)) (j : Fin BLOCK_N)
    (hj : t.val * BLOCK_N + j.val < split_n_length) :
    bgmvFullSpec s₀ input_ptr lora_ptr li K split_n_length xm_stride
        xk_stride l0_stride lora_k_stride lora_n_stride BLOCK_K
        (t.val * BLOCK_N + j.val)
      = ∑ k : Fin BLOCK_K,
          (if k.val < K then xs t k else 0)
            * (if t.val * BLOCK_N + j.val < split_n_length ∧ k.val < K
               then ys t (bTileLane BLOCK_N BLOCK_K j k) else 0) := by
  unfold bgmvFullSpec
  refine Finset.sum_congr rfl fun k _ => ?_
  unfold prodGK
  by_cases hk : k.val < K
  · rw [if_pos hk, if_pos hk, if_pos ⟨hj, hk⟩, if_pos ⟨hj, hk⟩]
    congr 1
    · unfold aElem
      exact hx t k hk
    · have hcnd : t.val * BLOCK_N
          + (bTileLane BLOCK_N BLOCK_K j k).val / BLOCK_K < split_n_length ∧
          (bTileLane BLOCK_N BLOCK_K j k).val % BLOCK_K < K := by
        unfold bTileLane
        rw [Lane2D.encode_div, Lane2D.encode_mod]
        exact ⟨hj, hk⟩
      have h := hy t (bTileLane BLOCK_N BLOCK_K j k) hcnd
      unfold bTileLane at h
      rw [Lane2D.encode_div, Lane2D.encode_mod] at h
      unfold bElem
      exact h
  · rw [if_neg hk, if_neg hk, if_neg (fun h => hk h.2),
      if_neg (fun h => hk h.2)]

/-! ### Covered fragment -/

/-- The full guarded surface (sentinel `ifThen`, int-cast `b_ptr`
arithmetic, masked loads, in-loop masked store) sits inside the flat-memory
bridge's covered fragment. -/
private theorem bgmv_flattenOk
    (input_ptr lora_ptr out_ptr : RegionName) (lora_indices : Region .int)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride
      lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K : Nat) :
    ((bgmv_full input_ptr lora_ptr out_ptr lora_indices K split_n_length
        xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride
        cn_stride slice_offset BLOCK_N BLOCK_K).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  rw [body_decomp]
  simp [prefixGuardBody, wbBody, StmtList.FlattenOk, Stmt.FlattenOk,
    Op.FlattenOk.eq_def]

/-! ### Body split, cast-free collapses, prelude run -/

/-- The three pre-guard statements (`pid_sn`, `cur_batch`, the `.int` slot
load). -/
private def preludeStmts (lora_indices : Region .int) : List Stmt :=
  [Stmt.assign TileDType.nat [] "pid_sn" (Op.programId 0),
   Stmt.assign TileDType.nat [] "cur_batch" (Op.programId 1),
   Stmt.assign TileDType.int [] "lora_index"
      (Op.load TileDType.int
        (MemAccess.region lora_indices (Op.ref TileDType.nat [] "cur_batch"))
        MaskOpt.none)]

/-- The algorithm body is the prelude followed by the sentinel-guarded
body. -/
private theorem bgmv_body_split
    (input_ptr lora_ptr out_ptr : RegionName) (lora_indices : Region .int)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride
      lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K : Nat) :
    (bgmv_full input_ptr lora_ptr out_ptr lora_indices K split_n_length
        xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride
        cn_stride slice_offset BLOCK_N BLOCK_K).toAlgKernel.body
      = preludeStmts lora_indices
        ++ [Stmt.ifThen
            (Op.ne ComparableDType.int Broadcast.nil
              (Op.ref TileDType.int [] "lora_index") (Op.constInt (-1)))
            (prefixGuardBody input_ptr lora_ptr out_ptr K split_n_length
              xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
              cm_stride cn_stride slice_offset BLOCK_N BLOCK_K)] := rfl

/-- The prelude is cast-free (`programId` assigns and a typed `.int` load —
loads are not rounding events): it steps identically under `stepStmtsR R`. -/
private theorem bgmvPrelude_castFree (R : RoundingModel)
    (lora_indices : Region .int) (t : BlockState) :
    stepStmtsR R (preludeStmts lora_indices) t
      = stepStmts (preludeStmts lora_indices) t := by
  simp only [preludeStmts, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def]
  rfl

/-- The 5 setup statements are cast-free (`arange`s, the masked `.real`
`tiled_a` load, the int-cast `b_ptr` arithmetic — `castNatToInt` /
`castIntToNat` are exact under every `R` — and the `c_ptr` assign). -/
private theorem bgmvSetup_castFree (R : RoundingModel)
    (input_ptr lora_ptr out_ptr : RegionName)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride
      lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K : Nat)
    (t : BlockState) :
    stepStmtsR R (setupStmts input_ptr lora_ptr out_ptr K split_n_length
        xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride
        cn_stride slice_offset BLOCK_N BLOCK_K) t
      = stepStmts (setupStmts input_ptr lora_ptr out_ptr K split_n_length
          xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride
          cn_stride slice_offset BLOCK_N BLOCK_K) t := by
  unfold setupStmts prefixGuardBody
  simp only [List.dropLast, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def]
  rfl

/-- The loop body is cast-free **including its in-loop masked `.real`
store** (`stepStmtR` delegates a `.real`-typed store to the exact
`writeMemTyped`), so the exact `wbInv` stack transports to `execR`. -/
private theorem bgmvWbBody_castFree (R : RoundingModel)
    (lora_k_stride lora_n_stride cn_stride K split_n_length BLOCK_N
      BLOCK_K : Nat) (t : BlockState) :
    stepStmtsR R (wbBody lora_k_stride lora_n_stride cn_stride K
        split_n_length BLOCK_N BLOCK_K) t
      = stepStmts (wbBody lora_k_stride lora_n_stride cn_stride K
          split_n_length BLOCK_N BLOCK_K) t := by
  simp only [wbBody, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def, BlockState.writeMemTypedR]
  rfl

/-- Exact run of the prelude from an arbitrary launch state, under the slot
pin (`m0` an **arbitrary** `Int` — sentinel and negative values included). -/
private theorem bgmvPrelude_run (lora_indices : Region .int)
    (s₀ : BlockState) (m0 : Int)
    (hm : s₀.readMemValue .int (Region.cast lora_indices) (s₀.pids 1) = m0) :
    stepStmts (preludeStmts lora_indices) s₀
      = some (((s₀.setReg "pid_sn" .nat [] (Tile.scalar (s₀.pids 0))).setReg
          "cur_batch" .nat [] (Tile.scalar (s₀.pids 1))).setReg
          "lora_index" .int [] (Tile.scalar m0)) := by
  unfold preludeStmts
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (by simp [evalOp_programId] :
      evalOp (Op.programId 0) s₀ = some (Tile.scalar (s₀.pids 0))))]
  set s1 := s₀.setReg "pid_sn" .nat [] (Tile.scalar (s₀.pids 0)) with hs1
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (by simp [evalOp_programId, hs1] :
      evalOp (Op.programId 1) s1 = some (Tile.scalar (s₀.pids 1))))]
  set s2 := s1.setReg "cur_batch" .nat [] (Tile.scalar (s₀.pids 1)) with hs2
  have hcb2 : s2.regs .nat [] "cur_batch" = some (Tile.scalar (s₀.pids 1)) := by
    rw [hs2]; exact BlockState.setReg_same _ _ _ _ _
  have hload : evalOp (Op.load TileDType.int
      (MemAccess.region lora_indices (Op.ref TileDType.nat [] "cur_batch"))
      MaskOpt.none) s2 = some (Tile.scalar m0) := by
    simp only [evalOp_load_region_none, evalOp_ref, hcb2, Option.bind_some]
    refine congrArg some (Tile.ext ?_)
    intro idx
    show s2.readMemValue .int (Region.cast lora_indices) (s₀.pids 1) = _
    rw [hs2, hs1]
    simp only [BlockState.setReg_readMemValue]
    rw [hm]
    rfl
  rw [stepStmts.cons_some (stepStmt_assign_eq_some hload), stepStmts.nil]

/-- The sentinel guard is `R`-free. -/
private theorem bgmvGuardR_eq (R : RoundingModel) (s : BlockState) :
    evalOpR R (Op.ne ComparableDType.int Broadcast.nil
        (Op.ref TileDType.int [] "lora_index") (Op.constInt (-1))) s
      = evalOp (Op.ne ComparableDType.int Broadcast.nil
          (Op.ref TileDType.int [] "lora_index") (Op.constInt (-1))) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

/-- Guard evaluation, non-sentinel slot value. -/
private theorem bgmvGuardT_eval (m0 : Int) (hm0 : m0 ≠ -1) (s : BlockState)
    (hlx : s.regs .int [] "lora_index" = some (Tile.scalar m0)) :
    evalOp (Op.ne ComparableDType.int Broadcast.nil
        (Op.ref TileDType.int [] "lora_index") (Op.constInt (-1))) s
      = some (Tile.scalar Bool.true) := by
  simp only [evalOp, hlx, Tile.cop, Tile.scalar_data, Broadcast.leftIndex,
    Broadcast.rightIndex]
  refine congrArg some (Tile.ext ?_)
  intro idx
  show ComparableDType.int.ne m0 (-1) = _
  simp only [ComparableDType.ne, Tile.scalar]
  exact decide_eq_true hm0

/-- Guard evaluation at the `-1` sentinel. -/
private theorem bgmvGuardF_eval (m0 : Int) (hm0 : m0 = -1) (s : BlockState)
    (hlx : s.regs .int [] "lora_index" = some (Tile.scalar m0)) :
    evalOp (Op.ne ComparableDType.int Broadcast.nil
        (Op.ref TileDType.int [] "lora_index") (Op.constInt (-1))) s
      = some (Tile.scalar Bool.false) := by
  simp only [evalOp, hlx, Tile.cop, Tile.scalar_data, Broadcast.leftIndex,
    Broadcast.rightIndex]
  refine congrArg some (Tile.ext ?_)
  intro idx
  show ComparableDType.int.ne m0 (-1) = _
  simp only [ComparableDType.ne, Tile.scalar]
  exact decide_eq_false (not_not_intro hm0)

/-! ### The generalized `b_ptr` evaluation

The exact stack's `bptr_eval` is pinned to `lora_index = Int.ofNat li`.
The io headline quantifies the slot value over **all** of `Int`, so the
copy below evaluates the kernel's `castIntToNat` clamp at an arbitrary
`mv : Int` directly: `(l0_stride · mv).toNat = l0_stride · mv.toNat`
(negative products clamp to `0` on both sides). -/

private theorem bptrGen_eval (lora_ptr : RegionName) (mv : Int)
    (l0_stride split_n_length lora_k_stride : Nat) (s s0 : BlockState)
    (hpids : s.pids = s0.pids)
    (hps : s.regs .nat [] "pid_sn" = some (Tile.scalar (s0.pids 0)))
    (hlx : s.regs .int [] "lora_index" = some (Tile.scalar mv)) :
    evalOp (Op.ptrAdd Broadcast.nil (Op.ptrBase lora_ptr)
        (Op.add NumericDType.nat Broadcast.nil
          (Op.mul NumericDType.int Broadcast.nil
              (Op.constNat l0_stride).castNatToInt
              (Op.ref TileDType.int [] "lora_index")).castIntToNat
          (Op.mul NumericDType.nat Broadcast.nil
            (Op.mul NumericDType.nat Broadcast.nil
              (Op.ref TileDType.nat [] "pid_sn")
              (Op.constNat split_n_length))
            (Op.constNat lora_k_stride)))) s
      = some (Tile.scalar
          (lora_ptr, lbaseOf s0 mv.toNat split_n_length l0_stride
            lora_k_stride)) := by
  rw [evalOp_ptrAdd, evalOp_ptrBase]
  simp only [evalOp, hps, hlx, Tile.bop, Tile.uop, Tile.ptrAdd, Tile.scalar,
    Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add,
    NumericDType.mul]
  apply congrArg some
  apply Tile.ext; intro idx
  simp only [Tile.ptrAdd, Tile.scalar, Broadcast.leftIndex,
    Broadcast.rightIndex, lbaseOf]
  congr 1
  rw [Nat.zero_add]
  have hmul : (Int.ofNat l0_stride * mv).toNat = l0_stride * mv.toNat := by
    cases mv with
    | ofNat n =>
        simp only [Int.ofNat_eq_natCast, ← Nat.cast_mul, Int.toNat_natCast]
    | negSucc n =>
        have h1 : Int.ofNat l0_stride * Int.negSucc n ≤ 0 :=
          mul_nonpos_of_nonneg_of_nonpos (Int.natCast_nonneg l0_stride)
            (le_of_lt (Int.negSucc_lt_zero n))
        rw [Int.toNat_eq_zero.mpr h1]
        rfl
  rw [hmul]

/-- Generalized `preLoop`: the 5 setup statements from a state whose
`lora_index` register holds an **arbitrary** `mv : Int`, seeding `wbInv` at
the clamped base `li := mv.toNat` (the copy of `Prefix2.preLoop` with
`bptrGen_eval` in place of `bptr_eval`; everything else identical). -/
private theorem preLoopGen (input_ptr lora_ptr out_ptr : RegionName)
    (mv : Int)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride
      lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K : Nat)
    (s0 s : BlockState)
    (hpids : s.pids = s0.pids) (hrm : s.readMem = s0.readMem)
    (hps : s.regs .nat [] "pid_sn" = some (Tile.scalar (s0.pids 0)))
    (hcb : s.regs .nat [] "cur_batch" = some (Tile.scalar (s0.pids 1)))
    (hlx : s.regs .int [] "lora_index" = some (Tile.scalar mv)) :
    ∃ s', stepStmts (setupStmts input_ptr lora_ptr out_ptr K split_n_length
        xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride
        cn_stride slice_offset BLOCK_N BLOCK_K) s = some s'
      ∧ wbInv input_ptr lora_ptr out_ptr mv.toNat K split_n_length xm_stride
          xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
          slice_offset BLOCK_N BLOCK_K s0 s 0 s' := by
  unfold setupStmts prefixGuardBody
  simp only [List.dropLast]
  -- offset_k
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (by simp [evalOp_arange] :
    evalOp (Op.arange BLOCK_K) s
      = some (Tile.vec (fun j : Fin BLOCK_K => j.val))))]
  set s1 := s.setReg "offset_k" .nat [BLOCK_K]
    (Tile.vec (fun j : Fin BLOCK_K => j.val)) with hs1
  -- offset_n
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (by simp [evalOp_arange] :
    evalOp (Op.arange BLOCK_N) s1
      = some (Tile.vec (fun i : Fin BLOCK_N => i.val))))]
  set s2 := s1.setReg "offset_n" .nat [BLOCK_N]
    (Tile.vec (fun i : Fin BLOCK_N => i.val)) with hs2
  have hcb2 : s2.regs .nat [] "cur_batch" = some (Tile.scalar (s0.pids 1)) := by
    rw [hs2, hs1]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("cur_batch":RegName) ≠ "offset_n" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("cur_batch":RegName) ≠ "offset_k" by decide)]
    exact hcb
  have hok2 : s2.regs .nat [BLOCK_K] "offset_k"
      = some (Tile.vec (fun j : Fin BLOCK_K => j.val)) := by
    rw [hs2]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
      (show ("offset_k":RegName) ≠ "offset_n" by decide)]
    rw [hs1]; exact BlockState.setReg_same _ _ _ _ _
  have hpids2 : s2.pids = s0.pids := by rw [hs2, hs1]; simp [hpids]
  have hrm2i : s2.readMem input_ptr = s0.readMem input_ptr := by
    rw [hs2, hs1]; funext ofs
    simp only [BlockState.setReg_readMem]
    exact congrFun (congrFun hrm input_ptr) ofs
  -- tiled_a
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (tileda_eval input_ptr K xm_stride xk_stride BLOCK_K s2 s0 hpids2 hrm2i
      hcb2 hok2))]
  set s3 := s2.setReg "tiled_a" .real [BLOCK_K]
    ⟨fun j : TileIndex [BLOCK_K] =>
      some (if j.1.val < K then aElem s0 input_ptr xm_stride xk_stride j.1.val
        else 0)⟩ with hs3
  have hps3 : s3.regs .nat [] "pid_sn" = some (Tile.scalar (s0.pids 0)) := by
    rw [hs3, hs2, hs1]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("pid_sn":RegName) ≠ "tiled_a" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("pid_sn":RegName) ≠ "offset_n" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("pid_sn":RegName) ≠ "offset_k" by decide)]
    exact hps
  have hlx3 : s3.regs .int [] "lora_index" = some (Tile.scalar mv) := by
    rw [hs3, hs2, hs1]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("lora_index":RegName) ≠ "tiled_a" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("lora_index":RegName) ≠ "offset_n" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("lora_index":RegName) ≠ "offset_k" by decide)]
    exact hlx
  have hpids3 : s3.pids = s0.pids := by rw [hs3]; simp [hpids2]
  -- b_ptr (generalized: castIntToNat clamps at an arbitrary mv)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bptrGen_eval lora_ptr mv l0_stride split_n_length lora_k_stride s3 s0
      hpids3 hps3 hlx3))]
  set s4 := s3.setReg "b_ptr" .ptr []
    (Tile.scalar (lora_ptr,
      lbaseOf s0 mv.toNat split_n_length l0_stride lora_k_stride)) with hs4
  have hps4 : s4.regs .nat [] "pid_sn" = some (Tile.scalar (s0.pids 0)) := by
    rw [hs4, BlockState.setReg_ne_name _ _ _ _ _ _ _ _
      (show ("pid_sn":RegName) ≠ "b_ptr" by decide)]
    exact hps3
  have hcb4 : s4.regs .nat [] "cur_batch" = some (Tile.scalar (s0.pids 1)) := by
    rw [hs4, BlockState.setReg_ne_name _ _ _ _ _ _ _ _
      (show ("cur_batch":RegName) ≠ "b_ptr" by decide), hs3]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
      (show ("cur_batch":RegName) ≠ "tiled_a" by decide)]
    exact hcb2
  have hpids4 : s4.pids = s0.pids := by rw [hs4]; simp [hpids3]
  -- c_ptr
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cptr_eval out_ptr split_n_length cm_stride cn_stride slice_offset s4 s0
      hpids4 hcb4 hps4))]
  rw [stepStmts.nil]
  set s5 := s4.setReg "c_ptr" .ptr []
    (Tile.scalar (out_ptr,
      cbaseOf s0 split_n_length cm_stride cn_stride slice_offset)) with hs5
  refine ⟨s5, rfl, ?_, ⟨0, rfl⟩, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hs5]; simp [hpids4]
  · rw [hs5, hs4, hs3]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("offset_k":RegName) ≠ "c_ptr" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("offset_k":RegName) ≠ "b_ptr" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("offset_k":RegName) ≠ "tiled_a" by decide)]
    exact hok2
  · rw [hs5, hs4, hs3, hs2]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("offset_n":RegName) ≠ "c_ptr" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("offset_n":RegName) ≠ "b_ptr" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("offset_n":RegName) ≠ "tiled_a" by decide)]
    exact BlockState.setReg_same _ _ _ _ _
  · rw [hs5, hs4]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("tiled_a":RegName) ≠ "c_ptr" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("tiled_a":RegName) ≠ "b_ptr" by decide)]
    rw [hs3]; exact BlockState.setReg_same _ _ _ _ _
  · rw [hs5, BlockState.setReg_ne_name _ _ _ _ _ _ _ _
      (show ("b_ptr":RegName) ≠ "c_ptr" by decide)]
    rw [hs4]; exact BlockState.setReg_same _ _ _ _ _
  · rw [hs5]; exact BlockState.setReg_same _ _ _ _ _
  · rw [hs5, hs4, hs3, hs2, hs1]; funext ofs
    simp only [BlockState.setReg_readMem]
    exact congrFun (congrFun hrm input_ptr) ofs
  · rw [hs5, hs4, hs3, hs2, hs1]; funext ofs
    simp only [BlockState.setReg_readMem]
    exact congrFun (congrFun hrm lora_ptr) ofs
  · intro g; rw [if_neg (by omega)]
    rw [hs5, hs4, hs3, hs2, hs1]
    simp only [BlockState.setReg_readMem]

/-! ### `evalOpR` collapses for the safety walk

The safety walk (`hts`) quantifies over arbitrary launch states, so it
re-evaluates the address/mask ops under `evalOpR R`. Every such op is
`R`-free (nat/bool/pointer arithmetic — no `castFloat`), so each collapses
onto `evalOp` and reuses the exact eval lemmas above. -/

private theorem bgmv_evalOpR_programId (R : RoundingModel) (ax : Nat)
    (s : BlockState) :
    evalOpR R (Op.programId ax) s = some (Tile.scalar (s.pids ax)) := by
  simp only [evalOpR]

private theorem bgmv_evalOpR_arange (R : RoundingModel) (n : Nat)
    (s : BlockState) :
    evalOpR R (Op.arange n) s = some (Tile.vec (fun j : Fin n => j.val)) := by
  simp only [evalOpR, Tile.vec]

/-- `current_n = n + offset_n` is `R`-free. -/
private theorem bgmv_currentnR_eq (R : RoundingModel) (BLOCK_N : Nat)
    (s : BlockState) :
    evalOpR R (Op.add NumericDType.nat Broadcast.scalarL
        (Op.ref TileDType.nat [] "n") (Op.ref TileDType.nat [BLOCK_N] "offset_n")) s
      = evalOp (Op.add NumericDType.nat Broadcast.scalarL
          (Op.ref TileDType.nat [] "n") (Op.ref TileDType.nat [BLOCK_N] "offset_n")) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

/-- The `b_ptr_mask` op is `R`-free. -/
private theorem bgmv_bmaskR_eq (R : RoundingModel)
    (K split_n_length BLOCK_N BLOCK_K : Nat) (s : BlockState) :
    evalOpR R (Op.boolAnd Broadcast.nil.consL.consR
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.nat [BLOCK_N] "current_n"))
          (Op.constNat split_n_length))
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref TileDType.nat [BLOCK_K] "offset_k"))
          (Op.constNat K))) s
      = evalOp (Op.boolAnd Broadcast.nil.consL.consR
          (Op.lt ComparableDType.nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.nat [BLOCK_N] "current_n"))
            (Op.constNat split_n_length))
          (Op.lt ComparableDType.nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref TileDType.nat [BLOCK_K] "offset_k"))
            (Op.constNat K))) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

/-- The `c_mask` op is `R`-free. -/
private theorem bgmv_cmaskR_eq (R : RoundingModel)
    (split_n_length BLOCK_N : Nat) (s : BlockState) :
    evalOpR R (Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.ref TileDType.nat [BLOCK_N] "current_n") (Op.constNat split_n_length)) s
      = evalOp (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.ref TileDType.nat [BLOCK_N] "current_n") (Op.constNat split_n_length)) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

/-- The `tiled_b` address op is `R`-free. -/
private theorem bgmv_bptrsR_eq (R : RoundingModel)
    (lora_k_stride lora_n_stride BLOCK_N BLOCK_K : Nat) (s : BlockState) :
    evalOpR R (Op.ptrAdd Broadcast.nil.consL.consR
        (Op.ptrAdd Broadcast.scalarL (Op.ref TileDType.ptr [] "b_ptr")
          (Op.mul NumericDType.nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.nat [BLOCK_N] "current_n"))
            (Op.constNat lora_k_stride)))
        (Op.mul NumericDType.nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref TileDType.nat [BLOCK_K] "offset_k"))
          (Op.constNat lora_n_stride))) s
      = evalOp (Op.ptrAdd Broadcast.nil.consL.consR
          (Op.ptrAdd Broadcast.scalarL (Op.ref TileDType.ptr [] "b_ptr")
            (Op.mul NumericDType.nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.nat [BLOCK_N] "current_n"))
              (Op.constNat lora_k_stride)))
          (Op.mul NumericDType.nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref TileDType.nat [BLOCK_K] "offset_k"))
            (Op.constNat lora_n_stride))) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

/-- The store address op `c_ptr + current_n·cn_stride` is `R`-free. -/
private theorem bgmv_cptrsR_eq (R : RoundingModel)
    (cn_stride BLOCK_N : Nat) (s : BlockState) :
    evalOpR R (Op.ptrAdd Broadcast.scalarL (Op.ref TileDType.ptr [] "c_ptr")
        (Op.mul NumericDType.nat Broadcast.scalarR
          (Op.ref TileDType.nat [BLOCK_N] "current_n") (Op.constNat cn_stride))) s
      = evalOp (Op.ptrAdd Broadcast.scalarL (Op.ref TileDType.ptr [] "c_ptr")
          (Op.mul NumericDType.nat Broadcast.scalarR
            (Op.ref TileDType.nat [BLOCK_N] "current_n") (Op.constNat cn_stride))) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

/-- The `tiled_a` address op is `R`-free. -/
private theorem bgmv_aAddrR_eq (R : RoundingModel)
    (xm_stride xk_stride BLOCK_K : Nat) (s : BlockState) :
    evalOpR R (Op.add NumericDType.nat Broadcast.scalarL
        (Op.mul NumericDType.nat Broadcast.nil
          (Op.ref TileDType.nat [] "cur_batch") (Op.constNat xm_stride))
        (Op.mul NumericDType.nat Broadcast.scalarR
          (Op.ref TileDType.nat [BLOCK_K] "offset_k") (Op.constNat xk_stride))) s
      = evalOp (Op.add NumericDType.nat Broadcast.scalarL
          (Op.mul NumericDType.nat Broadcast.nil
            (Op.ref TileDType.nat [] "cur_batch") (Op.constNat xm_stride))
          (Op.mul NumericDType.nat Broadcast.scalarR
            (Op.ref TileDType.nat [BLOCK_K] "offset_k") (Op.constNat xk_stride))) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

/-- The `tiled_a` mask op `offset_k < K` is `R`-free. -/
private theorem bgmv_aMaskR_eq (R : RoundingModel) (K BLOCK_K : Nat)
    (s : BlockState) :
    evalOpR R (Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.ref TileDType.nat [BLOCK_K] "offset_k") (Op.constNat K)) s
      = evalOp (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.ref TileDType.nat [BLOCK_K] "offset_k") (Op.constNat K)) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

/-- Per-lane value of the `tiled_a` address op. -/
private theorem bgmv_aAddr_eval (xm_stride xk_stride BLOCK_K : Nat)
    (s : BlockState) (cb : Nat)
    (hcb : s.regs .nat [] "cur_batch" = some (Tile.scalar cb))
    (hok : s.regs .nat [BLOCK_K] "offset_k"
      = some (Tile.vec (fun k : Fin BLOCK_K => k.val))) :
    evalOp (Op.add NumericDType.nat Broadcast.scalarL
        (Op.mul NumericDType.nat Broadcast.nil
          (Op.ref TileDType.nat [] "cur_batch") (Op.constNat xm_stride))
        (Op.mul NumericDType.nat Broadcast.scalarR
          (Op.ref TileDType.nat [BLOCK_K] "offset_k") (Op.constNat xk_stride))) s
      = some (Tile.vec (fun k : Fin BLOCK_K =>
          cb * xm_stride + k.val * xk_stride)) := by
  simp only [evalOp_add, evalOp_mul, evalOp_ref, evalOp_constNat, hcb, hok,
    Option.bind_eq_bind, Option.bind_some]
  apply congrArg some
  ext k
  simp only [Tile.bop_data, Tile.vec, Tile.scalar, Broadcast.leftIndex_scalarL,
    Broadcast.rightIndex_scalarL, Broadcast.leftIndex_scalarR,
    Broadcast.rightIndex_scalarR, Broadcast.leftIndex_nil,
    Broadcast.rightIndex_nil, NumericDType.add, NumericDType.mul]

/-- Per-lane value of the `tiled_a` mask op. -/
private theorem bgmv_aMask_eval (K BLOCK_K : Nat) (s : BlockState)
    (hok : s.regs .nat [BLOCK_K] "offset_k"
      = some (Tile.vec (fun k : Fin BLOCK_K => k.val))) :
    evalOp (Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.ref TileDType.nat [BLOCK_K] "offset_k") (Op.constNat K)) s
      = some (Tile.vec (fun k : Fin BLOCK_K => decide (k.val < K))) := by
  rw [evalOp_lt, evalOp_ref, hok, evalOp_constNat]
  apply congrArg some
  ext k
  simp only [Tile.cop_data, Tile.vec_data, Tile.scalar_data,
    Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR,
    ComparableDType.lt, decide_eq_decide]

/-! ### Cell-level memory frames -/

/-- A run of `assign` statements never touches memory. -/
private theorem bgmv_stepStmts_assigns_mem :
    ∀ (l : List Stmt),
      (∀ stmt ∈ l, ∃ dt sh nm, ∃ e : Op dt sh, stmt = Stmt.assign dt sh nm e) →
      ∀ {s s' : BlockState}, stepStmts l s = some s' → s'.mem = s.mem
  | [], _, s, s', h => by
      rw [stepStmts.nil] at h
      obtain rfl := Option.some.inj h
      rfl
  | stmt :: rest, hall, s, s', h => by
      obtain ⟨dt, sh, nm, e, rfl⟩ := hall _ List.mem_cons_self
      cases hv : evalOp e s with
      | none => simp [stepStmts, stepStmt, hv] at h
      | some v =>
          rw [stepStmts.cons_some (stepStmt_assign_eq_some hv)] at h
          rw [bgmv_stepStmts_assigns_mem rest
            (fun st' hst' => hall st' (List.mem_cons_of_mem _ hst')) h]
          rfl

/-- Cell-level frame of a `Bool`-masked exact `writeMem` scatter `foldl`:
every cell not hit by an active lane is untouched. -/
private theorem bgmv_foldl_writeMem_preserve_cell {α : Type}
    {region : RegionName} (ofn : α → Nat) (vfn : α → ℝ) (mask : α → Bool)
    (r : RegionName) (oo : Nat) (l : List α) (s : BlockState)
    (hnot : ∀ k ∈ l, mask k → ¬(r = region ∧ oo = ofn k)) :
    (l.foldl (fun acc k =>
        if mask k then acc.writeMem region (ofn k) (vfn k) else acc) s).mem r oo
      = s.mem r oo := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons]
      cases hm : mask hd
      · simp only [hm, Bool.false_eq_true, if_false]
        exact ih _ fun k hk hmk => hnot k (List.mem_cons_of_mem hd hk) hmk
      · simp only [hm, if_true]
        rw [ih _ fun k hk hmk => hnot k (List.mem_cons_of_mem hd hk) hmk,
          BlockState.writeMem_mem]
        exact if_neg (hnot hd List.mem_cons_self (by rw [hm]))

set_option maxHeartbeats 8000000 in
/-- **Cell-level frame of one loop iteration** (the `mem` twin of `wbStep`,
same walk): from the `wbInv` register pins, one storing body iteration
leaves every cell off the `{(out, cOff g) : g < split_n_length}` write
window untouched — the masked scatter store only hits active lanes
`c·BLOCK_N + i < split_n_length`, whose offsets are `cOff (c·BLOCK_N + i)`. -/
private theorem bgmv_wbBody_step_frame
    (input_ptr lora_ptr out_ptr : RegionName)
    (li K split_n_length xm_stride xk_stride l0_stride lora_k_stride
      lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K : Nat)
    (s0 s st' : BlockState) (c : Nat)
    (hpids : s.pids = s0.pids)
    (hri : s.readMem input_ptr = s0.readMem input_ptr)
    (hrl : s.readMem lora_ptr = s0.readMem lora_ptr)
    (hok : s.regs .nat [BLOCK_K] "offset_k"
      = some (Tile.vec (fun j : Fin BLOCK_K => j.val)))
    (hon : s.regs .nat [BLOCK_N] "offset_n"
      = some (Tile.vec (fun i : Fin BLOCK_N => i.val)))
    (hta : s.regs .real [BLOCK_K] "tiled_a"
      = some ⟨fun j : TileIndex [BLOCK_K] =>
          some (if j.1.val < K then aElem s0 input_ptr xm_stride xk_stride j.1.val else 0)⟩)
    (hbp : s.regs .ptr [] "b_ptr"
      = some (Tile.scalar (lora_ptr, lbaseOf s0 li split_n_length l0_stride lora_k_stride)))
    (hcp : s.regs .ptr [] "c_ptr"
      = some (Tile.scalar (out_ptr, cbaseOf s0 split_n_length cm_stride cn_stride slice_offset)))
    (hstep : stepStmts (wbBody lora_k_stride lora_n_stride cn_stride K
        split_n_length BLOCK_N BLOCK_K)
        (s.setReg "n" .nat [] (Tile.scalar (c*BLOCK_N))) = some st')
    (r : RegionName) (oo : Nat)
    (hcond : r ≠ out_ptr ∨ ∀ g : Fin split_n_length,
      oo ≠ cOff s0 split_n_length cm_stride cn_stride slice_offset g.val) :
    st'.mem r oo = s.mem r oo := by
  unfold wbBody at hstep
  -- current_n
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (currentn_eval BLOCK_N c s hon))] at hstep
  set s1 := (s.setReg "n" .nat [] (Tile.scalar (c*BLOCK_N))).setReg
    "current_n" .nat [BLOCK_N]
    (Tile.vec (fun i : Fin BLOCK_N => c*BLOCK_N + i.val)) with hs1
  have hcn1 : s1.regs .nat [BLOCK_N] "current_n"
      = some (Tile.vec (fun i : Fin BLOCK_N => c*BLOCK_N + i.val)) := by
    simp [hs1]
  have hok1 : s1.regs .nat [BLOCK_K] "offset_k"
      = some (Tile.vec (fun j : Fin BLOCK_K => j.val)) := by
    rw [hs1]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("offset_k":RegName) ≠ "current_n" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("offset_k":RegName) ≠ "n" by decide)]
    exact hok
  -- b_ptr_mask
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bmask_eval K split_n_length BLOCK_N BLOCK_K c s1 hcn1 hok1))] at hstep
  set s2 := s1.setReg "b_ptr_mask" .bool [BLOCK_N, BLOCK_K]
    ⟨fun idx : TileIndex [BLOCK_N, BLOCK_K] =>
      (decide (c*BLOCK_N + idx.1.val < split_n_length)
        && decide (idx.2.1.val < K))⟩ with hs2
  have hcn2 : s2.regs .nat [BLOCK_N] "current_n"
      = some (Tile.vec (fun i : Fin BLOCK_N => c*BLOCK_N + i.val)) := by
    rw [hs2]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
      (show ("current_n":RegName) ≠ "b_ptr_mask" by decide)]
    exact hcn1
  -- c_mask
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cmask_eval split_n_length BLOCK_N c s2 hcn2))] at hstep
  set s3 := s2.setReg "c_mask" .bool [BLOCK_N]
    (Tile.vec (fun i : Fin BLOCK_N =>
      decide (c*BLOCK_N + i.val < split_n_length))) with hs3
  have hbp3 : s3.regs .ptr [] "b_ptr"
      = some (Tile.scalar (lora_ptr,
          lbaseOf s0 li split_n_length l0_stride lora_k_stride)) := by
    rw [hs3, hs2, hs1]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("b_ptr":RegName) ≠ "c_mask" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("b_ptr":RegName) ≠ "b_ptr_mask" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("b_ptr":RegName) ≠ "current_n" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("b_ptr":RegName) ≠ "n" by decide)]
    exact hbp
  have hcn3 : s3.regs .nat [BLOCK_N] "current_n"
      = some (Tile.vec (fun i : Fin BLOCK_N => c*BLOCK_N + i.val)) := by
    rw [hs3]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
      (show ("current_n":RegName) ≠ "c_mask" by decide)]
    exact hcn2
  have hok3 : s3.regs .nat [BLOCK_K] "offset_k"
      = some (Tile.vec (fun j : Fin BLOCK_K => j.val)) := by
    rw [hs3, hs2]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("offset_k":RegName) ≠ "c_mask" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("offset_k":RegName) ≠ "b_ptr_mask" by decide)]
    exact hok1
  have hbm3 : s3.regs .bool [BLOCK_N, BLOCK_K] "b_ptr_mask"
      = some ⟨fun idx : TileIndex [BLOCK_N, BLOCK_K] =>
          (decide (c*BLOCK_N + idx.1.val < split_n_length)
            && decide (idx.2.1.val < K))⟩ := by
    rw [hs3]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
      (show ("b_ptr_mask":RegName) ≠ "c_mask" by decide)]
    rw [hs2]; exact BlockState.setReg_same _ _ _ _ _
  -- tiled_b
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (tiledb_eval lora_ptr (lbaseOf s0 li split_n_length l0_stride lora_k_stride)
      K split_n_length lora_k_stride lora_n_stride BLOCK_N BLOCK_K c s3
      hbp3 hcn3 hok3 hbm3))] at hstep
  set s4 := s3.setReg "tiled_b" .real [BLOCK_N, BLOCK_K]
    ⟨fun idx : TileIndex [BLOCK_N, BLOCK_K] =>
        some (if c*BLOCK_N + idx.1.val < split_n_length ∧ idx.2.1.val < K then
          s3.readMem lora_ptr ((lbaseOf s0 li split_n_length l0_stride lora_k_stride)
            + (c*BLOCK_N + idx.1.val)*lora_k_stride
            + idx.2.1.val*lora_n_stride) else 0.0)⟩ with hs4
  have hs3rm : s3.readMem = s.readMem := by
    funext rg ofs; rw [hs3, hs2, hs1]
    simp only [BlockState.setReg_readMem]
  have hs4ta : s4.regs .real [BLOCK_K] "tiled_a"
      = some ⟨fun j : TileIndex [BLOCK_K] =>
          some (if j.1.val < K then aElem s0 input_ptr xm_stride xk_stride j.1.val else 0)⟩ := by
    rw [hs4, BlockState.setReg_ne_name _ _ _ _ _ _ _ _
      (show ("tiled_a":RegName) ≠ "tiled_b" by decide), hs3, hs2, hs1]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("tiled_a":RegName) ≠ "c_mask" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("tiled_a":RegName) ≠ "b_ptr_mask" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("tiled_a":RegName) ≠ "current_n" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("tiled_a":RegName) ≠ "n" by decide)]
    exact hta
  have hs4tb : s4.regs .real [BLOCK_N, BLOCK_K] "tiled_b"
      = some ⟨fun idx : TileIndex [BLOCK_N, BLOCK_K] =>
          some (if c*BLOCK_N + idx.1.val < split_n_length ∧ idx.2.1.val < K then
            s4.readMem lora_ptr ((l0_stride * li + s0.pids 0 * split_n_length * lora_k_stride)
              + (c*BLOCK_N + idx.1.val)*lora_k_stride
              + idx.2.1.val*lora_n_stride) else 0.0)⟩ := by
    rw [hs4]; rw [BlockState.setReg_same]
    apply congrArg some; ext idx
    have : s4.readMem lora_ptr = s3.readMem lora_ptr := by
      rw [hs4]; funext ofs; simp [BlockState.setReg_readMem]
    rw [this]; rfl
  have hs4rm : s4.readMem = s.readMem := by
    funext rg ofs; rw [hs4]
    simp only [BlockState.setReg_readMem]
    exact congrFun (congrFun hs3rm rg) ofs
  -- accumulator
  erw [stepStmts.cons_some (stepStmt_assign_eq_some
    (acc_eval input_ptr lora_ptr li K split_n_length xm_stride xk_stride
      l0_stride lora_k_stride lora_n_stride BLOCK_N BLOCK_K c s4 s0
      (by rw [hs4rm]; exact hrl) hs4ta hs4tb))] at hstep
  set s5 := s4.setReg "accumulator" .real
    (TileShape.reduceShape [BLOCK_N, BLOCK_K] ⟨1, by simp⟩ Bool.false)
    ⟨fun i : TileIndex [BLOCK_N] =>
        some (bgmvFullSpec s0 input_ptr lora_ptr li K split_n_length
          xm_stride xk_stride l0_stride lora_k_stride lora_n_stride BLOCK_K
          (c*BLOCK_N + i.1.val))⟩ with hs5
  have hcn5 : s5.regs .nat [BLOCK_N] "current_n"
      = some (Tile.vec (fun i : Fin BLOCK_N => c*BLOCK_N + i.val)) := by
    rw [hs5, BlockState.setReg_ne_name _ _ _ _ _ _ _ _
      (show ("current_n":RegName) ≠ "accumulator" by decide), hs4,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
      (show ("current_n":RegName) ≠ "tiled_b" by decide)]
    exact hcn3
  have hcm5 : s5.regs .bool [BLOCK_N] "c_mask"
      = some (Tile.vec (fun i : Fin BLOCK_N =>
          decide (c*BLOCK_N + i.val < split_n_length))) := by
    rw [hs5, BlockState.setReg_ne_name _ _ _ _ _ _ _ _
      (show ("c_mask":RegName) ≠ "accumulator" by decide), hs4,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
      (show ("c_mask":RegName) ≠ "tiled_b" by decide), hs3]
    exact BlockState.setReg_same _ _ _ _ _
  have hcp5 : s5.regs .ptr [] "c_ptr"
      = some (Tile.scalar (out_ptr,
          cbaseOf s0 split_n_length cm_stride cn_stride slice_offset)) := by
    rw [hs5, BlockState.setReg_ne_name _ _ _ _ _ _ _ _
      (show ("c_ptr":RegName) ≠ "accumulator" by decide), hs4,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
      (show ("c_ptr":RegName) ≠ "tiled_b" by decide), hs3, hs2, hs1]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("c_ptr":RegName) ≠ "c_mask" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("c_ptr":RegName) ≠ "b_ptr_mask" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("c_ptr":RegName) ≠ "current_n" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("c_ptr":RegName) ≠ "n" by decide)]
    exact hcp
  have hacc5 : s5.regs .real [BLOCK_N] "accumulator"
      = some ⟨fun i : TileIndex [BLOCK_N] =>
          some (bgmvFullSpec s0 input_ptr lora_ptr li K split_n_length
            xm_stride xk_stride l0_stride lora_k_stride lora_n_stride BLOCK_K
            (c*BLOCK_N + i.1.val))⟩ := by
    rw [hs5]; exact BlockState.setReg_same _ _ _ _ _
  -- store
  rw [stepStmts.cons_some (store_step out_ptr
    (cbaseOf s0 split_n_length cm_stride cn_stride slice_offset) cn_stride
    BLOCK_N c s5
    (fun i : Fin BLOCK_N => bgmvFullSpec s0 input_ptr lora_ptr li K
      split_n_length xm_stride xk_stride l0_stride lora_k_stride
      lora_n_stride BLOCK_K (c*BLOCK_N + i.val))
    (fun i : Fin BLOCK_N => decide (c*BLOCK_N + i.val < split_n_length))
    hcp5 hcn5 hacc5 hcm5)] at hstep
  rw [stepStmts.nil] at hstep
  obtain rfl := Option.some.inj hstep
  refine Eq.trans (bgmv_foldl_writeMem_preserve_cell _ _ _ r oo _ _ ?_) ?_
  · intro b _ hmb hbad
    rcases hcond with hne | hno
    · exact hne hbad.1
    · have hbsnl : c*BLOCK_N + b.1.val < split_n_length := by simpa using hmb
      exact hno ⟨c*BLOCK_N + b.1.val, hbsnl⟩
        (hbad.2.trans (cOff_eq s0 split_n_length cm_stride cn_stride
          slice_offset (c*BLOCK_N + b.1.val)).symm)
  · rw [hs5, hs4, hs3, hs2, hs1]
    rfl

/-! ### The `TraceSafeR` walk -/

set_option maxHeartbeats 8000000 in
/-- Per-iteration `TraceSafeListR` for the loop body: the index/mask assigns
and the `tl.sum` reduce are register-only; the masked `tiled_b` load's and
the masked store's **active** lanes are the skin's `mask2` / `writeMask`
windows at step `i / BLOCK_N`, in bounds by the corresponding window bounds
(instantiated at raw counter `i`; the caller has already discharged the
`writeMask`'s sentinel conjunct — this walk only runs inside the taken
guard). -/
private theorem bgmv_wbBodySafeR (R : RoundingModel) (bounds : RegionBounds)
    (input_ptr lora_ptr out_ptr : RegionName)
    (li K split_n_length xm_stride xk_stride l0_stride lora_k_stride
      lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K : Nat)
    (s0 st : BlockState) (i : Nat) (hiB : BLOCK_N ∣ i)
    (hok : st.regs .nat [BLOCK_K] "offset_k"
      = some (Tile.vec (fun k : Fin BLOCK_K => k.val)))
    (hon : st.regs .nat [BLOCK_N] "offset_n"
      = some (Tile.vec (fun r : Fin BLOCK_N => r.val)))
    (hbp : st.regs .ptr [] "b_ptr"
      = some (Tile.scalar (lora_ptr,
          lbaseOf s0 li split_n_length l0_stride lora_k_stride)))
    (hcp : st.regs .ptr [] "c_ptr"
      = some (Tile.scalar (out_ptr,
          cbaseOf s0 split_n_length cm_stride cn_stride slice_offset)))
    (hbB : ∀ (rr : Fin BLOCK_N) (k : Fin BLOCK_K),
      i + rr.val < split_n_length → k.val < K →
      lbaseOf s0 li split_n_length l0_stride lora_k_stride
        + (i + rr.val) * lora_k_stride + k.val * lora_n_stride
        < bounds lora_ptr)
    (hbC : ∀ jj : Fin BLOCK_N, i + jj.val < split_n_length →
      cbaseOf s0 split_n_length cm_stride cn_stride slice_offset
        + (i + jj.val) * cn_stride < bounds out_ptr) :
    Stmt.TraceSafeListR R bounds
      (wbBody lora_k_stride lora_n_stride cn_stride K split_n_length BLOCK_N BLOCK_K)
      (st.setReg "n" .nat [] (Tile.scalar i)) := by
  obtain ⟨c, rfl⟩ := hiB
  rw [show BLOCK_N * c = c * BLOCK_N from Nat.mul_comm BLOCK_N c] at hbB hbC ⊢
  unfold wbBody
  -- current_n
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun t1 ht1 => ?_)
  rw [stepStmtR_assign_eq_some
    ((bgmv_currentnR_eq R BLOCK_N _).trans (currentn_eval BLOCK_N c st hon))] at ht1
  obtain rfl := Option.some.inj ht1
  set q1 := (st.setReg "n" .nat [] (Tile.scalar (c*BLOCK_N))).setReg
    "current_n" .nat [BLOCK_N]
    (Tile.vec (fun r : Fin BLOCK_N => c*BLOCK_N + r.val)) with hq1
  have hcn1 : q1.regs .nat [BLOCK_N] "current_n"
      = some (Tile.vec (fun r : Fin BLOCK_N => c*BLOCK_N + r.val)) := by
    simp [hq1]
  have hok1 : q1.regs .nat [BLOCK_K] "offset_k"
      = some (Tile.vec (fun k : Fin BLOCK_K => k.val)) := by
    rw [hq1]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("offset_k":RegName) ≠ "current_n" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("offset_k":RegName) ≠ "n" by decide)]
    exact hok
  -- b_ptr_mask
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun t2 ht2 => ?_)
  rw [stepStmtR_assign_eq_some
    ((bgmv_bmaskR_eq R K split_n_length BLOCK_N BLOCK_K _).trans
      (bmask_eval K split_n_length BLOCK_N BLOCK_K c q1 hcn1 hok1))] at ht2
  obtain rfl := Option.some.inj ht2
  set q2 := q1.setReg "b_ptr_mask" .bool [BLOCK_N, BLOCK_K]
    ⟨fun idx : TileIndex [BLOCK_N, BLOCK_K] =>
      (decide (c*BLOCK_N + idx.1.val < split_n_length)
        && decide (idx.2.1.val < K))⟩ with hq2
  have hcn2 : q2.regs .nat [BLOCK_N] "current_n"
      = some (Tile.vec (fun r : Fin BLOCK_N => c*BLOCK_N + r.val)) := by
    rw [hq2]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
      (show ("current_n":RegName) ≠ "b_ptr_mask" by decide)]
    exact hcn1
  -- c_mask
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun t3 ht3 => ?_)
  rw [stepStmtR_assign_eq_some
    ((bgmv_cmaskR_eq R split_n_length BLOCK_N _).trans
      (cmask_eval split_n_length BLOCK_N c q2 hcn2))] at ht3
  obtain rfl := Option.some.inj ht3
  set q3 := q2.setReg "c_mask" .bool [BLOCK_N]
    (Tile.vec (fun r : Fin BLOCK_N =>
      decide (c*BLOCK_N + r.val < split_n_length))) with hq3
  have hbp3 : q3.regs .ptr [] "b_ptr"
      = some (Tile.scalar (lora_ptr,
          lbaseOf s0 li split_n_length l0_stride lora_k_stride)) := by
    rw [hq3, hq2, hq1]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("b_ptr":RegName) ≠ "c_mask" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("b_ptr":RegName) ≠ "b_ptr_mask" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("b_ptr":RegName) ≠ "current_n" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("b_ptr":RegName) ≠ "n" by decide)]
    exact hbp
  have hcn3 : q3.regs .nat [BLOCK_N] "current_n"
      = some (Tile.vec (fun r : Fin BLOCK_N => c*BLOCK_N + r.val)) := by
    rw [hq3]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
      (show ("current_n":RegName) ≠ "c_mask" by decide)]
    exact hcn2
  have hok3 : q3.regs .nat [BLOCK_K] "offset_k"
      = some (Tile.vec (fun k : Fin BLOCK_K => k.val)) := by
    rw [hq3, hq2]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("offset_k":RegName) ≠ "c_mask" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("offset_k":RegName) ≠ "b_ptr_mask" by decide)]
    exact hok1
  have hbm3 : q3.regs .bool [BLOCK_N, BLOCK_K] "b_ptr_mask"
      = some ⟨fun idx : TileIndex [BLOCK_N, BLOCK_K] =>
          (decide (c*BLOCK_N + idx.1.val < split_n_length)
            && decide (idx.2.1.val < K))⟩ := by
    rw [hq3]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
      (show ("b_ptr_mask":RegName) ≠ "c_mask" by decide)]
    rw [hq2]; exact BlockState.setReg_same _ _ _ _ _
  -- the masked tiled_b load
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun t4 ht4 => ?_)
  · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR,
      and_true, true_and, and_self]
    intro ptrs hptrs idx hactive
    rw [bgmv_bptrsR_eq R lora_k_stride lora_n_stride BLOCK_N BLOCK_K q3,
      bptrs_eval lora_ptr (lbaseOf s0 li split_n_length l0_stride lora_k_stride)
        lora_k_stride lora_n_stride BLOCK_N BLOCK_K c q3 hbp3 hcn3 hok3] at hptrs
    obtain rfl := Option.some.inj hptrs
    obtain ⟨masks, hmv, hmi⟩ := hactive
    rw [evalOpR_ref, hbm3] at hmv
    obtain rfl := Option.some.inj hmv
    have hact : c*BLOCK_N + idx.1.val < split_n_length ∧ idx.2.1.val < K := by
      simpa using hmi
    exact hbB idx.1 idx.2.1 hact.1 hact.2
  · obtain ⟨v4, -, rfl⟩ := stepStmtR_assign_inv ht4
    set q4 := q3.setReg "tiled_b" .real [BLOCK_N, BLOCK_K] v4 with hq4
    -- accumulator (register-only)
    refine Stmt.TraceSafeListR.cons_intro
      (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun t5 ht5 => ?_)
    obtain ⟨v5, -, rfl⟩ := stepStmtR_assign_inv ht5
    set q5 := q4.setReg "accumulator" .real [BLOCK_N] v5 with hq5
    have hcn5 : q5.regs .nat [BLOCK_N] "current_n"
        = some (Tile.vec (fun r : Fin BLOCK_N => c*BLOCK_N + r.val)) := by
      rw [hq5, BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("current_n":RegName) ≠ "accumulator" by decide), hq4,
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("current_n":RegName) ≠ "tiled_b" by decide)]
      exact hcn3
    have hcm5 : q5.regs .bool [BLOCK_N] "c_mask"
        = some (Tile.vec (fun r : Fin BLOCK_N =>
            decide (c*BLOCK_N + r.val < split_n_length))) := by
      rw [hq5, BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("c_mask":RegName) ≠ "accumulator" by decide), hq4,
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("c_mask":RegName) ≠ "tiled_b" by decide), hq3]
      exact BlockState.setReg_same _ _ _ _ _
    have hcp5 : q5.regs .ptr [] "c_ptr"
        = some (Tile.scalar (out_ptr,
            cbaseOf s0 split_n_length cm_stride cn_stride slice_offset)) := by
      rw [hq5, BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("c_ptr":RegName) ≠ "accumulator" by decide), hq4,
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("c_ptr":RegName) ≠ "tiled_b" by decide), hq3, hq2, hq1]
      simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
          (show ("c_ptr":RegName) ≠ "c_mask" by decide),
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _
          (show ("c_ptr":RegName) ≠ "b_ptr_mask" by decide),
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _
          (show ("c_ptr":RegName) ≠ "current_n" by decide),
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _
          (show ("c_ptr":RegName) ≠ "n" by decide)]
      exact hcp
    -- the masked store
    refine Stmt.TraceSafeListR.cons_intro ?_
      (fun _ _ => Stmt.TraceSafeListR.nil_intro)
    simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MemAccess.SafeAtR,
      MaskOpt.SafeAtR, MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR,
      memAccessActiveAddressSafeR, and_true, true_and, and_self]
    intro ptrs hptrs idx hactive
    rw [bgmv_cptrsR_eq R cn_stride BLOCK_N q5,
      cptrs_eval out_ptr (cbaseOf s0 split_n_length cm_stride cn_stride slice_offset)
        cn_stride BLOCK_N c q5 hcp5 hcn5] at hptrs
    obtain rfl := Option.some.inj hptrs
    obtain ⟨masks, hmv, hmi⟩ := hactive
    rw [evalOpR_ref, hcm5] at hmv
    obtain rfl := Option.some.inj hmv
    have hact : c*BLOCK_N + idx.1.val < split_n_length := by
      simpa [Tile.vec] using hmi
    simpa [Tile.vec] using hbC idx.1 hact

set_option maxHeartbeats 8000000 in
/-- **The `TraceSafeR` walk for the whole guarded kernel** — the prelude
slot load is bounded by the skin's `mwin` bound; at the `-1` sentinel the
guard evaluates false and the skipped body needs no bounds at all; in the
taken branch the setup's masked `tiled_a` load is bounded by the (static
stream) `read1` window and the loop rides `Stmt.forRangeTraceSafeR_inv`
over the exact `wbInv`. -/
private theorem bgmv_traceSafeR (R : RoundingModel) (bounds : RegionBounds)
    (input_ptr lora_ptr out_ptr : RegionName) (lora_indices : Region .int)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride
      lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K : Nat)
    (hBN : 0 < BLOCK_N) (hoi : out_ptr ≠ input_ptr) (hol : out_ptr ≠ lora_ptr)
    (hcn : 0 < cn_stride) (s : BlockState) (m0 : Int)
    (hm : s.readMemValue .int (Region.cast lora_indices) (s.pids 1) = m0)
    (hbm : s.pids 1 < bounds (Region.cast lora_indices))
    (hbA : ∀ k : Fin BLOCK_K, k.val < K →
      s.pids 1 * xm_stride + k.val * xk_stride < bounds input_ptr)
    (hbB : ∀ (t : Fin (bgmvNumSteps split_n_length BLOCK_N))
        (l : Fin (BLOCK_N * BLOCK_K)),
      t.val * BLOCK_N + l.val / BLOCK_K < split_n_length ∧
        l.val % BLOCK_K < K →
      l0_stride * m0.toNat + s.pids 0 * split_n_length * lora_k_stride
        + (t.val * BLOCK_N + l.val / BLOCK_K) * lora_k_stride
        + l.val % BLOCK_K * lora_n_stride < bounds lora_ptr)
    (hbC : ∀ (t : Fin (bgmvNumSteps split_n_length BLOCK_N)) (j : Fin BLOCK_N),
      m0 ≠ -1 ∧ t.val * BLOCK_N + j.val < split_n_length →
      s.pids 1 * cm_stride + s.pids 0 * split_n_length
        + slice_offset * cn_stride + (t.val * BLOCK_N + j.val) * cn_stride
        < bounds out_ptr) :
    ((bgmv_full input_ptr lora_ptr out_ptr lora_indices K split_n_length
        xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride
        cn_stride slice_offset BLOCK_N BLOCK_K).toAlgKernel).TraceSafeR
      R bounds s := by
  unfold Kernel.TraceSafeR
  rw [bgmv_body_split]
  refine Stmt.TraceSafeListR.append_intro _ _ ?_ ?_
  · -- the prelude: two programId assigns around the `.int` slot load
    unfold preludeStmts
    refine Stmt.TraceSafeListR.cons_intro
      (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun q1 h1 => ?_)
    obtain ⟨v1, -, rfl⟩ := stepStmtR_assign_inv h1
    refine Stmt.TraceSafeListR.cons_intro
      (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun q2 h2 => ?_)
    obtain ⟨v2, hv2, rfl⟩ := stepStmtR_assign_inv h2
    rw [bgmv_evalOpR_programId] at hv2
    obtain rfl := Option.some.inj hv2
    refine Stmt.TraceSafeListR.cons_intro ?_
      (fun _ _ => Stmt.TraceSafeListR.nil_intro)
    simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
    refine ⟨trivial, trivial, ?_⟩
    intro offsets hoffs i _
    rw [evalOpR_ref, BlockState.setReg_same] at hoffs
    obtain rfl := Option.some.inj hoffs
    show ((s.setReg "pid_sn" .nat [] v1).pids 1)
      < bounds (Region.cast lora_indices)
    exact hbm
  · intro s3' hs3'
    rw [bgmvPrelude_castFree, bgmvPrelude_run lora_indices s m0 hm] at hs3'
    obtain rfl := Option.some.inj hs3'
    set s3 := ((s.setReg "pid_sn" .nat [] (Tile.scalar (s.pids 0))).setReg
      "cur_batch" .nat [] (Tile.scalar (s.pids 1))).setReg
      "lora_index" .int [] (Tile.scalar m0) with hs3
    have hlx3 : s3.regs .int [] "lora_index" = some (Tile.scalar m0) := by
      rw [hs3]; exact BlockState.setReg_same _ _ _ _ _
    have hps3 : s3.regs .nat [] "pid_sn" = some (Tile.scalar (s.pids 0)) := by
      rw [hs3]
      simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
          (show ("pid_sn":RegName) ≠ "lora_index" by decide),
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _
          (show ("pid_sn":RegName) ≠ "cur_batch" by decide)]
      exact BlockState.setReg_same _ _ _ _ _
    have hcb3 : s3.regs .nat [] "cur_batch" = some (Tile.scalar (s.pids 1)) := by
      rw [hs3]
      simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("cur_batch":RegName) ≠ "lora_index" by decide)]
      exact BlockState.setReg_same _ _ _ _ _
    have hpids3 : s3.pids = s.pids := by rw [hs3]; simp
    have hrm3 : s3.readMem = s.readMem := by
      funext rg ofs; rw [hs3]
      simp only [BlockState.setReg_readMem]
    refine Stmt.TraceSafeListR.cons_intro ?_
      (fun _ _ => Stmt.TraceSafeListR.nil_intro)
    simp only [Stmt.TraceSafeR]
    refine ⟨by simp [Op.SafeAtR.eq_def], ?_⟩
    by_cases hm0 : m0 = -1
    · rw [(bgmvGuardR_eq R s3).trans (bgmvGuardF_eval m0 hm0 s3 hlx3)]
      exact trivial
    · rw [(bgmvGuardR_eq R s3).trans (bgmvGuardT_eval m0 hm0 s3 hlx3)]
      show Stmt.TraceSafeListR R bounds
        (prefixGuardBody input_ptr lora_ptr out_ptr K split_n_length
          xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride
          cn_stride slice_offset BLOCK_N BLOCK_K) s3
      -- injectivity (needed by `wbInv_step`)
      have hinjs : Function.Injective (fun g : Fin split_n_length =>
          cOff s split_n_length cm_stride cn_stride slice_offset g.val) := by
        have heq : (fun g : Fin split_n_length =>
            cOff s split_n_length cm_stride cn_stride slice_offset g.val)
            = (fun g : Fin split_n_length =>
                (s.pids 1 * cm_stride + s.pids 0 * split_n_length
                  + slice_offset * cn_stride) + g.val * cn_stride) := by
          funext g; simp only [cOff]
        rw [heq]; exact affine1D_inj _ cn_stride hcn
      -- raw-counter bound converters
      have hbB' : ∀ i, BLOCK_N ∣ i → ∀ (rr : Fin BLOCK_N) (k : Fin BLOCK_K),
          i + rr.val < split_n_length → k.val < K →
          lbaseOf s m0.toNat split_n_length l0_stride lora_k_stride
            + (i + rr.val) * lora_k_stride + k.val * lora_n_stride
            < bounds lora_ptr := by
        intro i hiB rr k hg hk
        have hiN : i < split_n_length :=
          lt_of_le_of_lt (Nat.le_add_right _ _) hg
        have hcnd : i / BLOCK_N * BLOCK_N
            + (Lane2D.encode (rr, k, PUnit.unit) :
                Fin (BLOCK_N * BLOCK_K)).val / BLOCK_K < split_n_length ∧
            (Lane2D.encode (rr, k, PUnit.unit) :
                Fin (BLOCK_N * BLOCK_K)).val % BLOCK_K < K := by
          simp only [Lane2D.encode_div, Lane2D.encode_mod]
          rw [Nat.div_mul_cancel hiB]
          exact ⟨hg, hk⟩
        have h : l0_stride * m0.toNat
            + s.pids 0 * split_n_length * lora_k_stride
            + (i / BLOCK_N * BLOCK_N
                + (Lane2D.encode (rr, k, PUnit.unit) :
                    Fin (BLOCK_N * BLOCK_K)).val / BLOCK_K) * lora_k_stride
            + (Lane2D.encode (rr, k, PUnit.unit) :
                Fin (BLOCK_N * BLOCK_K)).val % BLOCK_K * lora_n_stride
            < bounds lora_ptr :=
          hbB ⟨i / BLOCK_N,
            bgmvStep_lt_numSteps split_n_length BLOCK_N i hBN hiN⟩
            (Lane2D.encode (rr, k, PUnit.unit)) hcnd
        simp only [Lane2D.encode_div, Lane2D.encode_mod] at h
        rw [Nat.div_mul_cancel hiB] at h
        exact h
      have hbC' : ∀ i, BLOCK_N ∣ i → ∀ jj : Fin BLOCK_N,
          i + jj.val < split_n_length →
          cbaseOf s split_n_length cm_stride cn_stride slice_offset
            + (i + jj.val) * cn_stride < bounds out_ptr := by
        intro i hiB jj hg
        have hiN : i < split_n_length :=
          lt_of_le_of_lt (Nat.le_add_right _ _) hg
        have h : s.pids 1 * cm_stride + s.pids 0 * split_n_length
            + slice_offset * cn_stride
            + (i / BLOCK_N * BLOCK_N + jj.val) * cn_stride < bounds out_ptr :=
          hbC ⟨i / BLOCK_N,
            bgmvStep_lt_numSteps split_n_length BLOCK_N i hBN hiN⟩ jj
            ⟨hm0, by rw [Nat.div_mul_cancel hiB]; exact hg⟩
        rw [Nat.div_mul_cancel hiB] at h
        exact h
      rw [prefixGuardBody_eq]
      refine Stmt.TraceSafeListR.append_intro _ _ ?_ ?_
      · -- the 5 setup statements
        unfold setupStmts prefixGuardBody
        simp only [List.dropLast]
        refine Stmt.TraceSafeListR.cons_intro
          (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun q1 h1 => ?_)
        rw [stepStmtR_assign_eq_some (bgmv_evalOpR_arange R BLOCK_K _)] at h1
        obtain rfl := Option.some.inj h1
        refine Stmt.TraceSafeListR.cons_intro
          (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun q2 h2 => ?_)
        obtain ⟨v2, -, rfl⟩ := stepStmtR_assign_inv h2
        have hcbq : ((s3.setReg "offset_k" .nat [BLOCK_K]
            (Tile.vec (fun k : Fin BLOCK_K => k.val))).setReg
            "offset_n" .nat [BLOCK_N] v2).regs .nat [] "cur_batch"
            = some (Tile.scalar (s.pids 1)) := by
          simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
              (show ("cur_batch":RegName) ≠ "offset_n" by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _
              (show ("cur_batch":RegName) ≠ "offset_k" by decide)]
          exact hcb3
        have hokq : ((s3.setReg "offset_k" .nat [BLOCK_K]
            (Tile.vec (fun k : Fin BLOCK_K => k.val))).setReg
            "offset_n" .nat [BLOCK_N] v2).regs .nat [BLOCK_K] "offset_k"
            = some (Tile.vec (fun k : Fin BLOCK_K => k.val)) := by
          simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("offset_k":RegName) ≠ "offset_n" by decide)]
          exact BlockState.setReg_same _ _ _ _ _
        -- the masked tiled_a load
        refine Stmt.TraceSafeListR.cons_intro ?_ (fun q3 h3 => ?_)
        · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
            MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR,
            and_true, true_and, and_self]
          intro offsets hoffs idx hactive
          rw [bgmv_aAddrR_eq R xm_stride xk_stride BLOCK_K _,
            bgmv_aAddr_eval xm_stride xk_stride BLOCK_K _ (s.pids 1)
              hcbq hokq] at hoffs
          obtain rfl := Option.some.inj hoffs
          obtain ⟨masks, hmv, hmi⟩ := hactive
          rw [bgmv_aMaskR_eq R K BLOCK_K _,
            bgmv_aMask_eval K BLOCK_K _ hokq] at hmv
          obtain rfl := Option.some.inj hmv
          have hk : idx.1.val < K := by simpa [Tile.vec] using hmi
          simpa [Region.cast_id, Tile.vec] using hbA idx.1 hk
        · obtain ⟨v3, -, rfl⟩ := stepStmtR_assign_inv h3
          refine Stmt.TraceSafeListR.of_forall _ _ ?_
          intro stmt hst s'
          simp only [List.mem_cons, List.not_mem_nil, or_false] at hst
          rcases hst with rfl | rfl <;>
            simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]
      · intro s5' hs5'
        obtain ⟨s5, hsetup, hinvP⟩ := preLoopGen input_ptr lora_ptr out_ptr
          m0 K split_n_length xm_stride xk_stride l0_stride lora_k_stride
          lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K
          s s3 hpids3 hrm3 hps3 hcb3 hlx3
        rw [bgmvSetup_castFree, hsetup] at hs5'
        obtain rfl := Option.some.inj hs5'
        refine Stmt.TraceSafeListR.cons_intro ?_
          (fun _ _ => Stmt.TraceSafeListR.nil_intro)
        simp only [Stmt.TraceSafeR]
        refine Stmt.forRangeTraceSafeR_inv R bounds "n" split_n_length BLOCK_N
          (wbBody lora_k_stride lora_n_stride cn_stride K split_n_length
            BLOCK_N BLOCK_K)
          (wbInv input_ptr lora_ptr out_ptr m0.toNat K split_n_length
            xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
            cm_stride cn_stride slice_offset BLOCK_N BLOCK_K s s3)
          ?_ 0 s5 hinvP
        intro i stt hi hP
        obtain ⟨hpidsQ, hdvdQ, hokQ, honQ, htaQ, hbpQ, hcpQ, hriQ, hrlQ,
          hmemQ⟩ := hP
        refine ⟨bgmv_wbBodySafeR R bounds input_ptr lora_ptr out_ptr m0.toNat
          K split_n_length xm_stride xk_stride l0_stride lora_k_stride
          lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K
          s stt i hdvdQ hokQ honQ hbpQ hcpQ
          (fun rr k => hbB' i hdvdQ rr k) (fun jj => hbC' i hdvdQ jj), ?_⟩
        obtain ⟨st', hstep, hP'⟩ := wbInv_step input_ptr lora_ptr out_ptr
          m0.toNat K split_n_length xm_stride xk_stride l0_stride
          lora_k_stride lora_n_stride cm_stride cn_stride slice_offset
          BLOCK_N BLOCK_K s s3 hBN hinjs hoi hol i stt hi
          ⟨hpidsQ, hdvdQ, hokQ, honQ, htaQ, hbpQ, hcpQ, hriQ, hrlQ, hmemQ⟩
        exact ⟨st', by rw [bgmvWbBody_castFree]; exact hstep, hP'⟩

/-! ### The rounded Hoare triple (`hrun`) -/

set_option maxHeartbeats 8000000 in
/-- Termination, per-lane values and the per-cell frame of the whole guarded
kernel under `execR R`, from an **arbitrary** launch state and an
**arbitrary** slot value `m0 : Int`: at the `-1` sentinel the guard skips
the body (nothing stored, memory untouched); otherwise the exact
`preLoopGen` / `wbInv` / `wbStep` / `forRange_inv` stack runs verbatim
(the guarded body is cast-free, so `execR R` collapses onto the exact
stepper), extended with the per-cell memory frame. -/
private theorem bgmv_runR (R : RoundingModel)
    (input_ptr lora_ptr out_ptr : RegionName) (lora_indices : Region .int)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride
      lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K : Nat)
    (hBN : 0 < BLOCK_N) (hoi : out_ptr ≠ input_ptr) (hol : out_ptr ≠ lora_ptr)
    (hcn : 0 < cn_stride) (s₀ : BlockState) (m0 : Int)
    (hm : s₀.readMemValue .int (Region.cast lora_indices) (s₀.pids 1) = m0) :
    ∃ sfin,
      execR R (bgmv_full input_ptr lora_ptr out_ptr lora_indices K
          split_n_length xm_stride xk_stride l0_stride lora_k_stride
          lora_n_stride cm_stride cn_stride slice_offset BLOCK_N
          BLOCK_K).toAlgKernel s₀ = some sfin
      ∧ (m0 ≠ -1 → ∀ g : Fin split_n_length,
          sfin.readMem out_ptr
              (cOff s₀ split_n_length cm_stride cn_stride slice_offset g.val)
            = bgmvFullSpec s₀ input_ptr lora_ptr m0.toNat K split_n_length
                xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
                BLOCK_K g.val)
      ∧ (∀ r oo, (r ≠ out_ptr ∨ ∀ g : Fin split_n_length, m0 ≠ -1 →
            oo ≠ cOff s₀ split_n_length cm_stride cn_stride slice_offset g.val) →
          sfin.mem r oo = s₀.mem r oo) := by
  have hpre := bgmvPrelude_run lora_indices s₀ m0 hm
  set s3 := ((s₀.setReg "pid_sn" .nat [] (Tile.scalar (s₀.pids 0))).setReg
    "cur_batch" .nat [] (Tile.scalar (s₀.pids 1))).setReg
    "lora_index" .int [] (Tile.scalar m0) with hs3
  have hlx3 : s3.regs .int [] "lora_index" = some (Tile.scalar m0) := by
    rw [hs3]; exact BlockState.setReg_same _ _ _ _ _
  have hps3 : s3.regs .nat [] "pid_sn" = some (Tile.scalar (s₀.pids 0)) := by
    rw [hs3]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("pid_sn":RegName) ≠ "lora_index" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("pid_sn":RegName) ≠ "cur_batch" by decide)]
    exact BlockState.setReg_same _ _ _ _ _
  have hcb3 : s3.regs .nat [] "cur_batch" = some (Tile.scalar (s₀.pids 1)) := by
    rw [hs3]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
      (show ("cur_batch":RegName) ≠ "lora_index" by decide)]
    exact BlockState.setReg_same _ _ _ _ _
  have hpids3 : s3.pids = s₀.pids := by rw [hs3]; simp
  have hrm3 : s3.readMem = s₀.readMem := by
    funext rg ofs; rw [hs3]
    simp only [BlockState.setReg_readMem]
  have hmem3 : ∀ r oo, s3.mem r oo = s₀.mem r oo := by
    intro r oo; rw [hs3]
    simp only [BlockState.setReg_mem]
  by_cases hm0 : m0 = -1
  · -- sentinel: the guard skips the body, nothing is stored
    have hifR : stepStmtR R (Stmt.ifThen
        (Op.ne ComparableDType.int Broadcast.nil
          (Op.ref TileDType.int [] "lora_index") (Op.constInt (-1)))
        (prefixGuardBody input_ptr lora_ptr out_ptr K split_n_length
          xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride
          cn_stride slice_offset BLOCK_N BLOCK_K)) s3 = some s3 := by
      rw [stepStmtR]
      simp only [(bgmvGuardR_eq R s3).trans (bgmvGuardF_eval m0 hm0 s3 hlx3),
        Tile.scalar, Option.bind_some]
      rfl
    refine ⟨s3, ?_, fun hne => absurd hm0 hne, fun r oo _ => hmem3 r oo⟩
    show execR R (bgmv_full input_ptr lora_ptr out_ptr lora_indices K
        split_n_length xm_stride xk_stride l0_stride lora_k_stride
        lora_n_stride cm_stride cn_stride slice_offset BLOCK_N
        BLOCK_K).toAlgKernel s₀ = some s3
    unfold execR
    rw [bgmv_body_split,
      stepStmtsR_append R (preludeStmts lora_indices) _ s₀,
      bgmvPrelude_castFree R lora_indices s₀, hpre, Option.bind_some,
      stepStmtsR_cons_some hifR, stepStmtsR_nil]
  · -- active: ride the exact stack with li := m0.toNat
    have hinjs : Function.Injective (fun g : Fin split_n_length =>
        cOff s₀ split_n_length cm_stride cn_stride slice_offset g.val) := by
      have heq : (fun g : Fin split_n_length =>
          cOff s₀ split_n_length cm_stride cn_stride slice_offset g.val)
          = (fun g : Fin split_n_length =>
              (s₀.pids 1 * cm_stride + s₀.pids 0 * split_n_length
                + slice_offset * cn_stride) + g.val * cn_stride) := by
        funext g; simp only [cOff]
      rw [heq]; exact affine1D_inj _ cn_stride hcn
    obtain ⟨s5, hsetup, hinvP⟩ := preLoopGen input_ptr lora_ptr out_ptr m0 K
      split_n_length xm_stride xk_stride l0_stride lora_k_stride
      lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K
      s₀ s3 hpids3 hrm3 hps3 hcb3 hlx3
    have hs5mem : ∀ r oo, s5.mem r oo = s₀.mem r oo := by
      intro r oo
      have h1 : s5.mem = s3.mem := by
        refine bgmv_stepStmts_assigns_mem _ ?_ hsetup
        intro stmt hst
        simp only [setupStmts, prefixGuardBody, List.dropLast, List.mem_cons,
          List.not_mem_nil, or_false] at hst
        rcases hst with rfl | rfl | rfl | rfl | rfl <;> exact ⟨_, _, _, _, rfl⟩
      rw [h1]; exact hmem3 r oo
    obtain ⟨final, sw, hsw, hle, hPW⟩ :=
      forRange_inv (idx := "n") (start := 0) (stop := split_n_length)
        (step := BLOCK_N)
        (P := fun i st =>
          wbInv input_ptr lora_ptr out_ptr m0.toNat K split_n_length
            xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
            cm_stride cn_stride slice_offset BLOCK_N BLOCK_K s₀ s3 i st
          ∧ ∀ r oo, (r ≠ out_ptr ∨ ∀ g : Fin split_n_length,
              oo ≠ cOff s₀ split_n_length cm_stride cn_stride slice_offset g.val) →
            st.mem r oo = s₀.mem r oo)
        (Nat.pos_iff_ne_zero.mp hBN)
        ⟨hinvP, fun r oo _ => hs5mem r oo⟩
        (fun i st hlt hP => by
          obtain ⟨st', hstep, hinv'⟩ := wbInv_step input_ptr lora_ptr out_ptr
            m0.toNat K split_n_length xm_stride xk_stride l0_stride
            lora_k_stride lora_n_stride cm_stride cn_stride slice_offset
            BLOCK_N BLOCK_K s₀ s3 hBN hinjs hoi hol i st hlt hP.1
          refine ⟨st', hstep, hinv', ?_⟩
          intro r oo hcond
          obtain ⟨hpidsQ, hdvdQ, hokQ, honQ, htaQ, hbpQ, hcpQ, hriQ, hrlQ,
            hmemQ⟩ := hP.1
          obtain ⟨cc, rfl⟩ := hdvdQ
          rw [show BLOCK_N * cc = cc * BLOCK_N from
            Nat.mul_comm BLOCK_N cc] at hstep
          rw [bgmv_wbBody_step_frame input_ptr lora_ptr out_ptr m0.toNat K
            split_n_length xm_stride xk_stride l0_stride lora_k_stride
            lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K
            s₀ st st' cc hpidsQ hriQ hrlQ hokQ honQ htaQ hbpQ hcpQ hstep
            r oo hcond]
          exact hP.2 r oo hcond)
    have hLoopR : stepStmtR R (Stmt.forRange "n" 0 split_n_length BLOCK_N
        (wbBody lora_k_stride lora_n_stride cn_stride K split_n_length
          BLOCK_N BLOCK_K)) s5 = some sw := by
      rw [stepStmtR_forRange,
        stepForRangeAuxR_castFree R _
          (bgmvWbBody_castFree R lora_k_stride lora_n_stride cn_stride K
            split_n_length BLOCK_N BLOCK_K) "n",
        ← stepForRangeAux.forRange_unfold]
      exact hsw
    have hpgbR : stepStmtsR R (prefixGuardBody input_ptr lora_ptr out_ptr K
        split_n_length xm_stride xk_stride l0_stride lora_k_stride
        lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K) s3
        = some sw := by
      rw [prefixGuardBody_eq,
        stepStmtsR_append R (setupStmts input_ptr lora_ptr out_ptr K
          split_n_length xm_stride xk_stride l0_stride lora_k_stride
          lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K) _ s3,
        bgmvSetup_castFree R input_ptr lora_ptr out_ptr K split_n_length
          xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride
          cn_stride slice_offset BLOCK_N BLOCK_K s3,
        hsetup, Option.bind_some, stepStmtsR_cons_some hLoopR, stepStmtsR_nil]
    have hifR : stepStmtR R (Stmt.ifThen
        (Op.ne ComparableDType.int Broadcast.nil
          (Op.ref TileDType.int [] "lora_index") (Op.constInt (-1)))
        (prefixGuardBody input_ptr lora_ptr out_ptr K split_n_length
          xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride
          cn_stride slice_offset BLOCK_N BLOCK_K)) s3 = some sw := by
      rw [stepStmtR]
      simp only [(bgmvGuardR_eq R s3).trans (bgmvGuardT_eval m0 hm0 s3 hlx3),
        Tile.scalar, Option.bind_some]
      exact hpgbR
    refine ⟨sw, ?_, ?_, ?_⟩
    · show execR R (bgmv_full input_ptr lora_ptr out_ptr lora_indices K
          split_n_length xm_stride xk_stride l0_stride lora_k_stride
          lora_n_stride cm_stride cn_stride slice_offset BLOCK_N
          BLOCK_K).toAlgKernel s₀ = some sw
      unfold execR
      rw [bgmv_body_split,
        stepStmtsR_append R (preludeStmts lora_indices) _ s₀,
        bgmvPrelude_castFree R lora_indices s₀, hpre, Option.bind_some,
        stepStmtsR_cons_some hifR, stepStmtsR_nil]
    · intro _ g
      obtain ⟨_, _, _, _, _, _, _, _, _, hread⟩ := hPW.1
      rw [hread g, if_pos (lt_of_lt_of_le g.isLt hle)]
    · intro r oo hcond
      refine hPW.2 r oo ?_
      rcases hcond with hne | hno
      · exact Or.inl hne
      · exact Or.inr fun g => hno g hm0

/-! ### ════════ ★ STREAMING METADATA EMIT HEADLINE ★ ════════ -/

set_option maxHeartbeats 8000000 in
/-- **The `⊨[R]` streaming metadata emit headline (wave-5, MetaEmit
genre).** For every rounding model `R`, the verified `bgmv_full` config of
`_bgmv_expand_slice_kernel` (`EVEN_K = false`, `ADD_INPUTS = false`, no
`CAST_TYPE` — exactly the config the exact numeric stack above verifies)
implements, on its `StreamMetaEmitMasked3DKernelIO₂` signature, the
**ideal ℝ masked rank-`K` GEMV dot** at every write-active emitted lane:
step `t`, lane `j` of the output stream holds

```
Σ_{k < BLOCK_K} ([k < K]·a-tile[k]) · ([t·BLOCK_N + j < split_n_length ∧ k < K]·b-tile[t](j, k))
```

— exact real arithmetic over the pinned streams. The `.int` metadata slot
`m 0 = lora_indices[cur_batch]` enters the value contract **only** through
the sentinel gate and the `read2` base geometry:

* **Sentinel branch is genuine.** `writeMask` carries `m 0 ≠ -1`, so at the
  `-1` sentinel the write-active window family is empty, the readback is
  vacuous, and the frame clause proves the program leaves **all** of memory
  untouched (the guard skips the body). This is the io headline's added
  value over the exact `bgmv_full_output_summary`, whose `hlx` hypothesis
  pins `lora_index = Int.ofNat li ≥ 0` and says nothing about sentinel
  programs.
* **Negative non-sentinel values are honest, not hypothesized away.** The
  slot is quantified over all of `Int`; for `m 0 ∉ {-1} ∪ ℕ` the kernel
  *runs* the body with the `castIntToNat` clamp `(l0_stride·m 0).toNat = 0`
  on the `b_ptr` base, and `read2`'s `(m 0).toNat` reproduces exactly that
  clamp (the generalized `bptrGen_eval`), so the statement stays true
  without any sign hypothesis.

The kernel has **zero rounding events** (typed `.int` slot load, masked
`.real` loads, `.real` `tl.sum`, `.real` masked in-loop store; the only
casts are the exact int casts in `b_ptr`), so with the default
`outDType := .real` the skin's boundary quantization degenerates: the
readback's `R.round .real` is the identity and the per-step stores are
exact under `execR R` — the ∀-`R` face holds via the `RoundingModel`
`.real` identity fields, not as a `.triv` special case. Layer map: the
guarded body is cast-free, so under `execR R` it collapses verbatim onto
the exact stepper and the proven `preLoop` / `wbInv` / `wbStep` /
`forRange_inv` stack above is reused unchanged (with `li := (m 0).toNat`
threaded through `preLoopGen`); the `⊨[R]` face adds the `TraceSafeR` walk,
the per-cell memory frame (`bgmv_wbBody_step_frame`, the `mem` twin of
`wbStep`) and the stream-lane spec bridge (`bgmvSpec_eq_streamSum`).

All five hypotheses are truth-forced; provenance:

* `hBN : 0 < BLOCK_N` — the loop steps by `BLOCK_N`
  (`range(0, split_n_length, BLOCK_N)`); at `BLOCK_N = 0` the loop never
  advances and the step index `n / BLOCK_N` is meaningless. A launched
  constexpr tile is nonempty. (Same hypothesis as the exact headline.)
* `hsn : 0 < split_n_length` — `split_n_length = ⌈N / SPLIT_N⌉ ≥ 1` for
  every nonempty output (`N ≥ 1`). Needed because the trip count
  `T = ⌈split_n_length / BLOCK_N⌉` vanishes at `split_n_length = 0`, which
  erases the static `read1` window while the kernel's pre-loop `tiled_a`
  load still executes — the safety walk would have no bound to cite. The
  degenerate `N = 0` launch stores nothing and is out of scope.
* `hoi : out_ptr ≠ input_ptr`, `hol : out_ptr ≠ lora_ptr` — the loop
  stores into `out` **between** re-reads of the LoRA-B tiles (and the
  register-cached `tiled_a` was read from `input`); aliasing would let a
  later block read already-overwritten values. (Same as the exact
  headline.)
* `hcn : 0 < cn_stride` — output-lane footprint injectivity
  (`affine1D_inj`): with `cn_stride = 0` all output lanes collide on one
  cell and the per-lane readback would be last-writer-wins. Torch strides
  of a non-degenerate output are ≥ 1. (Same as the exact headline.)

The exact headline's `hlx` slot pin is **not** carried: the skin pins the
slot internally and the proof case-splits on the sentinel.

Inherited modeling boundary (unchanged from the file docstring): the
numeric face covers the main `ADD_INPUTS = false`, no-`CAST_TYPE` config
(`bgmv_full`); the `ADD_INPUTS` accumulation and `CAST_TYPE` value faces
remain future work, exercised only by the lowering theorem
`bgmv_expand_slice_surface_toAlgorithm_supported` on the general surface. -/
specification bgmv_expand_slice_io_correctness (R : RoundingModel)
    (input_ptr lora_ptr out_ptr : RegionName) (lora_indices : Region .int)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride
      lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K : Nat)
    (hBN : 0 < BLOCK_N) (hsn : 0 < split_n_length)
    (hoi : out_ptr ≠ input_ptr) (hol : out_ptr ≠ lora_ptr)
    (hcn : 0 < cn_stride) :
    bgmvExpandSliceIO input_ptr lora_ptr out_ptr lora_indices K
        split_n_length xm_stride xk_stride l0_stride lora_k_stride
        lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K
      ⊨[R] fun _ _ _ m xs ys t j =>
        ∑ k : Fin BLOCK_K,
          (if k.val < K then xs t k else 0)
            * (if t.val * BLOCK_N + j.val < split_n_length ∧ k.val < K
               then ys t (bTileLane BLOCK_N BLOCK_K j k) else 0) := by
  refine StreamMetaEmitMasked3DKernelIO₂.ImplementsR.intro _ ?_ ?_ ?_
  · exact bgmv_flattenOk input_ptr lora_ptr out_ptr lora_indices K
      split_n_length xm_stride xk_stride l0_stride lora_k_stride
      lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K
  · -- safety walk
    intro bounds s m xs ys hm _hx _hy hbm hbr1 hbr2 hbw
    simp only [bgmvExpandSliceIO] at hm hbm hbr1 hbr2 hbw
    have hT : 0 < bgmvNumSteps split_n_length BLOCK_N := by
      have h0 := bgmvStep_lt_numSteps split_n_length BLOCK_N 0 hBN hsn
      simpa using h0
    exact bgmv_traceSafeR R bounds input_ptr lora_ptr out_ptr lora_indices K
      split_n_length xm_stride xk_stride l0_stride lora_k_stride
      lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K
      hBN hoi hol hcn s (m ⟨0, Nat.one_pos⟩) (hm ⟨0, Nat.one_pos⟩)
      (hbm ⟨0, Nat.one_pos⟩) (fun k hk => hbr1 ⟨0, hT⟩ k hk) hbr2 hbw
  · -- the rounded Hoare triple
    intro s₀ m xs ys _hundef hm hx hy
    simp only [bgmvExpandSliceIO] at hm hx hy ⊢
    obtain ⟨sfin, hexec, hval, hframe⟩ := bgmv_runR R input_ptr lora_ptr
      out_ptr lora_indices K split_n_length xm_stride xk_stride l0_stride
      lora_k_stride lora_n_stride cm_stride cn_stride slice_offset BLOCK_N
      BLOCK_K hBN hoi hol hcn s₀ (m ⟨0, Nat.one_pos⟩) (hm ⟨0, Nat.one_pos⟩)
    refine ⟨sfin, hexec, ?_, ?_⟩
    · intro t j hj
      have hg : t.val * BLOCK_N + j.val < split_n_length := hj.2
      have hcell : sfin.readMem out_ptr
          (cOff s₀ split_n_length cm_stride cn_stride slice_offset
            (t.val * BLOCK_N + j.val))
          = bgmvFullSpec s₀ input_ptr lora_ptr (m ⟨0, Nat.one_pos⟩).toNat K
              split_n_length xm_stride xk_stride l0_stride lora_k_stride
              lora_n_stride BLOCK_K (t.val * BLOCK_N + j.val) :=
        hval hj.1 ⟨t.val * BLOCK_N + j.val, hg⟩
      rw [show s₀.pids 1 * cm_stride + s₀.pids 0 * split_n_length
            + slice_offset * cn_stride + (t.val * BLOCK_N + j.val) * cn_stride
          = cOff s₀ split_n_length cm_stride cn_stride slice_offset
              (t.val * BLOCK_N + j.val) from rfl,
        BlockState.readMemAs_real, hcell,
        bgmvSpec_eq_streamSum input_ptr lora_ptr s₀
          (m ⟨0, Nat.one_pos⟩).toNat K split_n_length xm_stride xk_stride
          l0_stride lora_k_stride lora_n_stride BLOCK_N BLOCK_K xs ys hx hy
          t j hg]
      simp [FloatDType.ofReal]
    · intro r oo hcond
      refine hframe r oo ?_
      rcases hcond with hne | hno
      · exact Or.inl hne
      · refine Or.inr fun g hgne => ?_
        have hdm : g.val / BLOCK_N * BLOCK_N + g.val % BLOCK_N = g.val := by
          rw [Nat.mul_comm]
          exact Nat.div_add_mod g.val BLOCK_N
        have h : oo ≠ s₀.pids 1 * cm_stride + s₀.pids 0 * split_n_length
            + slice_offset * cn_stride
            + (g.val / BLOCK_N * BLOCK_N + g.val % BLOCK_N) * cn_stride := hno
          ⟨g.val / BLOCK_N,
            bgmvStep_lt_numSteps split_n_length BLOCK_N g.val hBN g.isLt⟩
          ⟨g.val % BLOCK_N, Nat.mod_lt _ hBN⟩
          ⟨hgne, by
            show g.val / BLOCK_N * BLOCK_N + g.val % BLOCK_N < split_n_length
            rw [hdm]
            exact g.isLt⟩
        rw [hdm] at h
        show oo ≠ cOff s₀ split_n_length cm_stride cn_stride slice_offset g.val
        exact h

end IOFace

end VeriTile.Bench.TritonBenchG.BgmvExpandSlice
