import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.Kernel

/-!
# `lora_expand_gemv` — strict per-kernel correctness

`_bgmv_expand_kernel` is a grouped LoRA expand GEMV: program `(pid_sn, cur_batch)`
loads the batch's input vector, selects the LoRA-B weight matrix via
`lora_indices[cur_batch]` (with the signed `-1` sentinel skipping the batch), and
for each output-`n` block reduces `sum(tiled_a * tiled_b, axis=1)` over the rank
dimension `K`, optionally adding the existing output (`ADD_INPUTS`).

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`_bgmv_expand_kernel[grid](...)`, the 2D grid
`(SPLIT_N, batches)`, the host stride/shape setup, and how the runtime composes
per-program writes) is the *trusted boundary*, not a proof obligation here.
Because the program ids are universally quantified, the per-program statement
covers every program of the grid.

## Proof architecture

The **full multi-block** development lives in the `Full` namespace and proves the
complete `for n` output-block loop correct against a genuine matrix-vector
product:

```
Full.gemv_full_output_summary                ← FULL multi-block TOP THEOREM
  ├─ (toAlgorithm? = Except.ok _)            surface lowers to the algorithm layer
  └─ Full.gemv_compute_correct               ← ComputeCorrect over the masked store
       └─ Full.gemv_exec_correct             ← per global lane m < split_n_length readback
            ├─ Full.prefix_inv               (P 0: loop registers seeded, output untouched)
            ├─ Full.gemvWbInv_step           (one n-block: store gemvSpec, preserve earlier)
            └─ forRange_inv                  (loop-invariant principle drives ⌈snl/BLOCK_N⌉ blocks)
```

For every *global* active output lane `m < split_n_length`, the stored value is
`gemvSpec(m) = Σ_{k < K} x[k]·W[m,k]` over `ℝ` — the genuine LoRA-B GEMV of the
loaded input vector `x` and the selected LoRA-B matrix `W`, NOT the kernel's own
emitted value. The writeback invariant `gemvWbInv` carries: after the first `i`
lanes, every global lane `m < i` holds `gemvSpec(m)` and later blocks do not
clobber earlier ones (`hinj`, global output-offset injectivity). The reduction
bridge `reduceSum_active_eq_spec` collapses the `tl.sum` over `BLOCK_K` keys to
`Σ_{k<K}` (using `K ≤ BLOCK_K`, the kernel's `next_power_of_2(K)` choice).

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` is not
modeled. The full theorem holds for arbitrary `split_n_length` and `BLOCK_N`
(`0 < BLOCK_N`), with the `for n` loop running `⌈split_n_length/BLOCK_N⌉` blocks;
masked lanes load `0` (matching `other=0.0`). The verified path is
`ADD_INPUTS = false`, no `CAST_TYPE`, and the masked input load (general `EVEN_K`
variant), with `K ≤ BLOCK_K`. The GEMV `tl.sum` accumulator is modeled exactly as
`Tile.reduceSum` over `ℝ` (no floating-point accumulation-order reassociation is
claimed). The signed-sentinel `lora_index == -1` early return is the host's
trusted boundary (the `Full` surface verifies the active body, which the host
only launches when `lora_index ≠ -1`). The LoRA-B base is data-dependent (read
from `lora_indices`); the per-lane global output-offset injectivity `hinj` and
`out_ptr ≠ lora_ptr` are required as side conditions.
-/

namespace VeriTile.Bench.TritonBenchG.LoraExpandGemv

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Faithful transcription of
`lora_expand_gemv.py`'s `_bgmv_expand_kernel`.

This covers the `EVEN_K` load path, `tl.cdiv` split length, output-block
loop, optional `CAST_TYPE` conversion, and `ADD_INPUTS` accumulation path.
Python's signed `lora_index == -1` early return is represented as a guard
around the active body. -/
def bgmv_expand_surface
    (input_ptr lora_ptr out_ptr : RegionName)
    (N K : Nat) (lora_indices : Region .int)
    (xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride
      cn_stride BLOCK_N BLOCK_K SPLIT_N : Nat)
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
    c_ptr = out_ptr + cur_batch * $(cm_stride) + pid_sn * split_n_length
    for n in range($(0), split_n_length, $(BLOCK_N)) {
      current_n = n + offset_n
      current_n_c = tl.max_contiguous(current_n, $(BLOCK_N))
      b_ptr_mask = (current_n[:, None] < split_n_length) & (offset_k[None, :] < $(K))
      c_mask = current_n < split_n_length
      tiled_b = tl.load(
        b_ptr + current_n_c[:, None] * $(lora_k_stride) +
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

/-- The full LoRA expand GEMV surface lowers to the algorithm layer, including
the signed sentinel guard, split loop, optional cast, and add-inputs branch. -/
theorem bgmv_expand_surface_toAlgorithm_supported
    (input_ptr lora_ptr out_ptr : RegionName)
    (N K : Nat) (lora_indices : Region .int)
    (xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride
      cn_stride BLOCK_N BLOCK_K SPLIT_N : Nat)
    (EVEN_K ADD_INPUTS CAST_TYPE : Bool) :
    ∃ alg,
      (bgmv_expand_surface input_ptr lora_ptr out_ptr N K lora_indices
        xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride
        cn_stride BLOCK_N BLOCK_K SPLIT_N EVEN_K ADD_INPUTS CAST_TYPE).toAlgorithm? =
        Except.ok alg := by
  simp [bgmv_expand_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-! ## Full multi-block GEMV correctness

The development below verifies the **complete `for n` output-block loop** of the
LoRA expand GEMV against a genuine mathematical matrix-vector product. For every
*global* active output lane `m < split_n_length`, the stored value equals
`Σ_{k < K} x[k] · W[m,k]` over `ℝ`, where `x` is the loaded input vector and `W`
is the selected LoRA-B matrix. The proof drives the masked-store loop through
`forRange_inv` with a writeback invariant `gemvWbInv` (after processing the first
`n` lanes, every global lane `m < n` holds the GEMV value and later blocks do not
clobber earlier ones — needs global output-offset injectivity), exactly mirroring
the rmsnorm writeback-loop architecture. The no-cast (`CAST_TYPE = false`) path
is verified; `BLOCK_K ≥ K` (the kernel's `next_power_of_2(K)` choice) lets the
`tl.sum` over `BLOCK_K` keys collapse to `Σ_{k < K}`. -/

namespace Full

open VeriTile.Triton

/-- The full loop surface: signed-sentinel guard elided (the host only launches
the active body when `lora_index ≠ -1`), `RegionName`-typed `lora_indices` so the
selected base is read back via `readMemValue .nat`, `ADD_INPUTS = false`,
`CAST_TYPE = false`, `EVEN_K = false` (masked input load). The `for n` loop runs
`⌈split_n_length / BLOCK_N⌉` blocks. -/
def bgmv_loop_surface
    (input_ptr lora_ptr out_ptr : RegionName) (lora_indices : Region .nat)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride
      lora_n_stride cm_stride cn_stride BLOCK_N BLOCK_K : Nat) :
    ComputeKernel := triton {
  pid_sn = tl.program_id(axis=0)
  cur_batch = tl.program_id(axis=1)
  lora_index = tl.load(lora_indices + cur_batch)
  offset_k = tl.arange(0, $(BLOCK_K))
  offset_n = tl.arange(0, $(BLOCK_N))
  tiled_a = tl.load(input_ptr + cur_batch * $(xm_stride) + offset_k * $(xk_stride),
    mask=offset_k < $(K), other=0.0)
  b_ptr = lora_ptr + $(l0_stride) * lora_index +
    pid_sn * $(split_n_length) * $(lora_k_stride)
  c_ptr = out_ptr + cur_batch * $(cm_stride) + pid_sn * $(split_n_length)
  for n in range($(0), $(split_n_length), $(BLOCK_N)) {
    current_n = n + offset_n
    tiled_b = tl.load(
      b_ptr + current_n[:, None] * $(lora_k_stride) +
        offset_k[None, :] * $(lora_n_stride),
      mask=(current_n[:, None] < $(split_n_length)) and (offset_k[None, :] < $(K)),
      other=0.0)
    accumulator = tl.sum(tiled_a * tiled_b, 1)
    tl.store(c_ptr + current_n * $(cn_stride), accumulator,
      mask=current_n < $(split_n_length))
  }
}

theorem bgmv_loop_surface_toAlgorithm_supported
    (input_ptr lora_ptr out_ptr : RegionName) (lora_indices : Region .nat)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride
      lora_n_stride cm_stride cn_stride BLOCK_N BLOCK_K : Nat) :
    ∃ alg, (bgmv_loop_surface input_ptr lora_ptr out_ptr lora_indices K
      split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride BLOCK_N BLOCK_K).toAlgorithm? = Except.ok alg := by
  simp [bgmv_loop_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-! ### Genuine GEMV spec (global output lane `m`) -/

/-- The selected LoRA index (`lora_indices[cur_batch]`). -/
def loraIdx (s : BlockState) (lora_indices : Region .nat) : Nat :=
  s.readMemValue .nat lora_indices (s.pids 1)

/-- The input vector element `x[k] = input[cur_batch·xm + k·xk]`. -/
noncomputable def aElem (s : BlockState) (input_ptr : RegionName)
    (xm_stride xk_stride : Nat) (k : Nat) : ℝ :=
  s.readMem input_ptr (s.pids 1 * xm_stride + k * xk_stride)

/-- The LoRA-B element `W[m,k] = lora[l0·idx + pid_sn·snl·lk + m·lk + k·ln]`. -/
noncomputable def bElem (s : BlockState) (lora_ptr : RegionName)
    (lora_indices : Region .nat)
    (split_n_length l0_stride lora_k_stride lora_n_stride : Nat)
    (m k : Nat) : ℝ :=
  s.readMem lora_ptr
    (l0_stride * loraIdx s lora_indices +
      s.pids 0 * split_n_length * lora_k_stride +
      m * lora_k_stride + k * lora_n_stride)

/-- **Genuine GEMV spec**: output lane `m` equals `Σ_{k<K} x[k]·W[m,k]`. -/
noncomputable def gemvSpec (s : BlockState) (input_ptr lora_ptr : RegionName)
    (lora_indices : Region .nat)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride
      lora_n_stride : Nat) (m : Nat) : ℝ :=
  gemmSum (fun k => aElem s input_ptr xm_stride xk_stride k)
    (fun k => bElem s lora_ptr lora_indices split_n_length l0_stride lora_k_stride
      lora_n_stride m k) K

/-- Global output offset for lane `m`: `cur_batch·cm + pid_sn·snl + m·cn`. -/
def outOffG (s : BlockState) (split_n_length cm_stride cn_stride : Nat)
    (m : Nat) : Nat :=
  s.pids 1 * cm_stride + s.pids 0 * split_n_length + m * cn_stride

/-! ### Body decomposition -/

/-- The 4-statement `for n` loop body, transcribed. -/
def loopBody
    (K split_n_length lora_k_stride lora_n_stride cn_stride BLOCK_N BLOCK_K : Nat) :
    List Stmt :=
  [ Stmt.assign .nat [BLOCK_N] "current_n"
      (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "n")
        (Op.ref .nat [BLOCK_N] "offset_n")),
    Stmt.assign .real [BLOCK_N, BLOCK_K] "tiled_b"
      (Op.load .real
        (MemAccess.ptr
          (Op.ptrAdd (Broadcast.consR (Broadcast.consL Broadcast.nil))
            (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "b_ptr")
              (Op.mul .nat Broadcast.scalarR
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "current_n"))
                (Op.constNat lora_k_stride)))
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_K] "offset_k"))
              (Op.constNat lora_n_stride))))
        (MaskOpt.maskOther
          (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
            (Op.lt .nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "current_n"))
              (Op.constNat split_n_length))
            (Op.lt .nat Broadcast.scalarR
              (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_K] "offset_k"))
              (Op.constNat K)))
          ((Op.const 0.0).broadcast [BLOCK_N, BLOCK_K]))),
    Stmt.assign .real [BLOCK_N] "accumulator"
      (Op.reduceSum ⟨1, by simp⟩ Bool.false
        (Op.mul .real (Broadcast.leadL (Broadcast.consSame Broadcast.nil))
          (Op.ref .real [BLOCK_K] "tiled_a")
          (Op.ref .real [BLOCK_N, BLOCK_K] "tiled_b"))),
    Stmt.store .real [BLOCK_N]
      (MemAccess.ptr
        (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "c_ptr")
          (Op.mul .nat Broadcast.scalarR (Op.ref .nat [BLOCK_N] "current_n")
            (Op.constNat cn_stride))))
      (Op.ref .real [BLOCK_N] "accumulator")
      (MaskOpt.mask
        (Op.lt .nat Broadcast.scalarR (Op.ref .nat [BLOCK_N] "current_n")
          (Op.constNat split_n_length))) ]

/-- Body decomposition: prefix (8 statements) ++ [for-loop]. By `rfl`. -/
theorem body_split
    (input_ptr lora_ptr out_ptr : RegionName) (lora_indices : Region .nat)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride
      lora_n_stride cm_stride cn_stride BLOCK_N BLOCK_K : Nat) :
    (bgmv_loop_surface input_ptr lora_ptr out_ptr lora_indices K
      split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride BLOCK_N BLOCK_K).toAlgKernel.body
      = (bgmv_loop_surface input_ptr lora_ptr out_ptr lora_indices K
          split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
          cm_stride cn_stride BLOCK_N BLOCK_K).toAlgKernel.body.take 8
        ++ [Stmt.forRange "n" 0 split_n_length BLOCK_N
              (loopBody K split_n_length lora_k_stride lora_n_stride cn_stride
                BLOCK_N BLOCK_K)] := by
  rfl

/-! ### Writeback invariant -/

/-- Captured input tile `tiled_a[k] = if k<K then x[k] else 0`. -/
noncomputable def tiledA (s : BlockState) (input_ptr : RegionName)
    (K xm_stride xk_stride BLOCK_K : Nat) : Tile .real [BLOCK_K] :=
  ⟨fun idx : TileIndex [BLOCK_K] =>
    some (if idx.1.val < K then aElem s input_ptr xm_stride xk_stride idx.1.val else 0)⟩

/-- The `b_ptr` scalar pointer value. -/
def bPtrVal (s : BlockState) (lora_ptr : RegionName) (lora_indices : Region .nat)
    (split_n_length l0_stride lora_k_stride : Nat) : RegionName × Nat :=
  (lora_ptr,
    l0_stride * loraIdx s lora_indices +
      s.pids 0 * split_n_length * lora_k_stride)

/-- The `c_ptr` scalar pointer value. -/
def cPtrVal (s : BlockState) (out_ptr : RegionName)
    (split_n_length cm_stride : Nat) : RegionName × Nat :=
  (out_ptr, s.pids 1 * cm_stride + s.pids 0 * split_n_length)

/-- **Writeback invariant** (counter `i = c·BLOCK_N`). After processing the
first `i` output lanes: pids fixed, the prefix registers seeded, `lora_ptr`
memory unchanged, and every global lane `m < split_n_length` holds the GEMV
value if `m < i`, else the original output. -/
noncomputable def gemvWbInv
    (input_ptr lora_ptr out_ptr : RegionName) (lora_indices : Region .nat)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride
      lora_n_stride cm_stride cn_stride BLOCK_N BLOCK_K : Nat)
    (s0 sorig : BlockState) (i : Nat) (s : BlockState) : Prop :=
  s.pids = s0.pids ∧ (BLOCK_N ∣ i) ∧
  s.regs .nat [BLOCK_K] "offset_k" = some (Tile.vec (fun k : Fin BLOCK_K => k.val)) ∧
  s.regs .nat [BLOCK_N] "offset_n" = some (Tile.vec (fun j : Fin BLOCK_N => j.val)) ∧
  s.regs .real [BLOCK_K] "tiled_a" =
    some (tiledA s0 input_ptr K xm_stride xk_stride BLOCK_K) ∧
  s.regs .ptr [] "b_ptr" =
    some (Tile.scalar (bPtrVal s0 lora_ptr lora_indices split_n_length l0_stride
      lora_k_stride)) ∧
  s.regs .ptr [] "c_ptr" =
    some (Tile.scalar (cPtrVal s0 out_ptr split_n_length cm_stride)) ∧
  s.readMem lora_ptr = s0.readMem lora_ptr ∧
  (∀ m : Fin split_n_length,
    s.readMem out_ptr (outOffG s0 split_n_length cm_stride cn_stride m.val)
      = if m.val < i then
          gemvSpec s0 input_ptr lora_ptr lora_indices K split_n_length
            xm_stride xk_stride l0_stride lora_k_stride lora_n_stride m.val
        else
          sorig.readMem out_ptr (outOffG s0 split_n_length cm_stride cn_stride m.val))

/-! ### Prefix eval helpers -/

theorem tiledA_eval (input_ptr : RegionName) (K xm_stride xk_stride BLOCK_K : Nat)
    (s0 s : BlockState)
    (hck : s.regs .nat [] "cur_batch" = some (Tile.scalar (s0.pids 1)))
    (hok : s.regs .nat [BLOCK_K] "offset_k" = some (Tile.vec (fun k : Fin BLOCK_K => k.val)))
    (_hpids : s.pids = s0.pids) (hrm : s.readMem = s0.readMem) :
    evalOp (Op.load .real
        (MemAccess.region input_ptr
          (Op.add .nat Broadcast.scalarL
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_batch") (Op.constNat xm_stride))
            (Op.mul .nat Broadcast.scalarR (Op.ref .nat [BLOCK_K] "offset_k") (Op.constNat xk_stride))))
        (MaskOpt.maskOther
          (Op.lt .nat Broadcast.scalarR (Op.ref .nat [BLOCK_K] "offset_k") (Op.constNat K))
          ((Op.const 0.0).broadcast [BLOCK_K]))) s
      = some (tiledA s0 input_ptr K xm_stride xk_stride BLOCK_K) := by
  simp only [evalOp, evalOp_ref, hck, hok, Option.bind, Option.map,
    Tile.bop, Tile.cop, Tile.scalar, Tile.vec, NumericDType.add, NumericDType.mul, ComparableDType.lt]
  apply congrArg some
  apply Tile.ext
  intro idx
  simp only [tiledA, aElem, hrm, Tile.scalar, Tile.cop, Tile.vec, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add, NumericDType.mul, ComparableDType.lt]
  by_cases hk : idx.1.val < K
  · simp [hk, BlockState.readMemValue_real, hrm]
  · simp only [hk, decide_false, Bool.false_eq_true, if_false, if_neg]
    norm_num

theorem bptr_eval (lora_ptr : RegionName) (lora_indices : Region .nat)
    (split_n_length l0_stride lora_k_stride : Nat) (s0 s : BlockState)
    (hli : s.regs .nat [] "lora_index" = some (Tile.scalar (loraIdx s0 lora_indices)))
    (hsn : s.regs .nat [] "pid_sn" = some (Tile.scalar (s0.pids 0)))
    (_hpids : s.pids = s0.pids) :
    evalOp (Op.ptrAdd Broadcast.nil (Op.ptrBase lora_ptr)
        (Op.add .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.constNat l0_stride) (Op.ref .nat [] "lora_index"))
          (Op.mul .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_sn") (Op.constNat split_n_length))
            (Op.constNat lora_k_stride)))) s
      = some (Tile.scalar (bPtrVal s0 lora_ptr lora_indices split_n_length l0_stride lora_k_stride)) := by
  simp only [evalOp, evalOp_ref, hli, hsn, Option.bind,
    Tile.bop, Tile.ptrAdd, Tile.scalar, NumericDType.add, NumericDType.mul]
  apply congrArg some
  apply Tile.ext
  intro _
  simp only [bPtrVal, Region.cast, Broadcast.leftIndex, Broadcast.rightIndex]
  refine Prod.ext rfl ?_
  ring

theorem cptr_eval (out_ptr : RegionName) (split_n_length cm_stride : Nat) (s0 s : BlockState)
    (hck : s.regs .nat [] "cur_batch" = some (Tile.scalar (s0.pids 1)))
    (hsn : s.regs .nat [] "pid_sn" = some (Tile.scalar (s0.pids 0)))
    (_hpids : s.pids = s0.pids) :
    evalOp (Op.ptrAdd Broadcast.nil (Op.ptrBase out_ptr)
        (Op.add .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_batch") (Op.constNat cm_stride))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_sn") (Op.constNat split_n_length)))) s
      = some (Tile.scalar (cPtrVal s0 out_ptr split_n_length cm_stride)) := by
  simp only [evalOp, evalOp_ref, hck, hsn, Option.bind,
    Tile.bop, Tile.ptrAdd, Tile.scalar, NumericDType.add, NumericDType.mul]
  apply congrArg some
  apply Tile.ext
  intro _
  simp only [cPtrVal, Region.cast, Broadcast.leftIndex, Broadcast.rightIndex]
  refine Prod.ext rfl ?_
  ring

/-! ### Prefix -/

/-- The 8 prologue statements, transcribed (= `body.take 8`, by `rfl`). -/
def prefixStmts
    (input_ptr lora_ptr out_ptr : RegionName) (lora_indices : Region .nat)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride
      cm_stride BLOCK_N BLOCK_K : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "pid_sn" (Op.programId 0),
    Stmt.assign .nat [] "cur_batch" (Op.programId 1),
    Stmt.assign .nat [] "lora_index"
      (Op.load .nat (MemAccess.region lora_indices (Op.ref .nat [] "cur_batch"))
        MaskOpt.none),
    Stmt.assign .nat [BLOCK_K] "offset_k" (Op.arange BLOCK_K),
    Stmt.assign .nat [BLOCK_N] "offset_n" (Op.arange BLOCK_N),
    Stmt.assign .real [BLOCK_K] "tiled_a"
      (Op.load .real
        (MemAccess.region input_ptr
          (Op.add .nat Broadcast.scalarL
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_batch") (Op.constNat xm_stride))
            (Op.mul .nat Broadcast.scalarR (Op.ref .nat [BLOCK_K] "offset_k") (Op.constNat xk_stride))))
        (MaskOpt.maskOther
          (Op.lt .nat Broadcast.scalarR (Op.ref .nat [BLOCK_K] "offset_k") (Op.constNat K))
          ((Op.const 0.0).broadcast [BLOCK_K]))),
    Stmt.assign .ptr [] "b_ptr"
      (Op.ptrAdd Broadcast.nil (Op.ptrBase lora_ptr)
        (Op.add .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.constNat l0_stride) (Op.ref .nat [] "lora_index"))
          (Op.mul .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_sn") (Op.constNat split_n_length))
            (Op.constNat lora_k_stride)))),
    Stmt.assign .ptr [] "c_ptr"
      (Op.ptrAdd Broadcast.nil (Op.ptrBase out_ptr)
        (Op.add .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_batch") (Op.constNat cm_stride))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_sn") (Op.constNat split_n_length)))) ]

theorem prefix_split
    (input_ptr lora_ptr out_ptr : RegionName) (lora_indices : Region .nat)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride
      lora_n_stride cm_stride cn_stride BLOCK_N BLOCK_K : Nat) :
    (bgmv_loop_surface input_ptr lora_ptr out_ptr lora_indices K
        split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
        cm_stride cn_stride BLOCK_N BLOCK_K).toAlgKernel.body.take 8
      = prefixStmts input_ptr lora_ptr out_ptr lora_indices K split_n_length
          xm_stride xk_stride l0_stride lora_k_stride cm_stride BLOCK_N BLOCK_K := by
  rfl

set_option maxHeartbeats 4000000 in
/-- **Prefix**: the 8 prologue statements seed the loop registers and establish
the writeback invariant at `i = 0` (no output lane written yet). -/
theorem prefix_inv
    (input_ptr lora_ptr out_ptr : RegionName) (lora_indices : Region .nat)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride
      lora_n_stride cm_stride cn_stride BLOCK_N BLOCK_K : Nat)
    (s : BlockState) (_hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ s', stepStmts ((bgmv_loop_surface input_ptr lora_ptr out_ptr lora_indices K
        split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
        cm_stride cn_stride BLOCK_N BLOCK_K).toAlgKernel.body.take 8) s = some s'
      ∧ gemvWbInv input_ptr lora_ptr out_ptr lora_indices K split_n_length
          xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride
          cn_stride BLOCK_N BLOCK_K s s 0 s' := by
  rw [prefix_split]
  unfold prefixStmts
  -- pid_sn, cur_batch
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (show evalOp (Op.programId 0) s = some (Tile.scalar (s.pids 0)) by simp))]
  set s1 := s.setReg "pid_sn" .nat [] (Tile.scalar (s.pids 0)) with hs1
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (show evalOp (Op.programId 1) s1 = some (Tile.scalar (s.pids 1)) by simp [hs1]))]
  set s2 := s1.setReg "cur_batch" .nat [] (Tile.scalar (s.pids 1)) with hs2
  -- lora_index load
  have hli2 : evalOp (Op.load .nat (MemAccess.region lora_indices (Op.ref .nat [] "cur_batch")) MaskOpt.none) s2
      = some (Tile.scalar (loraIdx s lora_indices)) := by
    simp only [evalOp, evalOp_ref, hs2, BlockState.setReg_same, Option.bind, loraIdx,
      BlockState.setReg_pids, BlockState.setReg_readMemValue, Tile.scalar]
    rfl
  rw [stepStmts.cons_some (stepStmt_assign_eq_some hli2)]
  set s3 := s2.setReg "lora_index" .nat [] (Tile.scalar (loraIdx s lora_indices)) with hs3
  -- offset_k, offset_n
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (show evalOp (Op.arange BLOCK_K) s3 = some (Tile.vec (fun k : Fin BLOCK_K => k.val)) by simp [Tile.vec]))]
  set s4 := s3.setReg "offset_k" .nat [BLOCK_K] (Tile.vec (fun k : Fin BLOCK_K => k.val)) with hs4
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (show evalOp (Op.arange BLOCK_N) s4 = some (Tile.vec (fun j : Fin BLOCK_N => j.val)) by simp [Tile.vec]))]
  set s5 := s4.setReg "offset_n" .nat [BLOCK_N] (Tile.vec (fun j : Fin BLOCK_N => j.val)) with hs5
  -- common register facts for s5
  have hck5 : s5.regs .nat [] "cur_batch" = some (Tile.scalar (s.pids 1)) := by
    rw [hs5, hs4, hs3]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("cur_batch":RegName) ≠ "offset_n" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("cur_batch":RegName) ≠ "offset_k" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("cur_batch":RegName) ≠ "lora_index" by decide)]
    rw [hs2, BlockState.setReg_same]
  have hsn5 : s5.regs .nat [] "pid_sn" = some (Tile.scalar (s.pids 0)) := by
    rw [hs5, hs4, hs3, hs2]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("pid_sn":RegName) ≠ "offset_n" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("pid_sn":RegName) ≠ "offset_k" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("pid_sn":RegName) ≠ "lora_index" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("pid_sn":RegName) ≠ "cur_batch" by decide)]
    rw [hs1, BlockState.setReg_same]
  have hli5 : s5.regs .nat [] "lora_index" = some (Tile.scalar (loraIdx s lora_indices)) := by
    rw [hs5, hs4]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("lora_index":RegName) ≠ "offset_n" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("lora_index":RegName) ≠ "offset_k" by decide)]
    rw [hs3, BlockState.setReg_same]
  have hok5 : s5.regs .nat [BLOCK_K] "offset_k" = some (Tile.vec (fun k : Fin BLOCK_K => k.val)) := by
    rw [hs5]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_k":RegName) ≠ "offset_n" by decide)]
    rw [hs4, BlockState.setReg_same]
  have hpids5 : s5.pids = s.pids := by rw [hs5, hs4, hs3, hs2, hs1]; simp
  have hrm5 : s5.readMem = s.readMem := by funext rg ofs; rw [hs5, hs4, hs3, hs2, hs1]; simp [BlockState.setReg_readMem]
  -- tiled_a load
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (tiledA_eval input_ptr K xm_stride xk_stride BLOCK_K s s5 hck5 hok5 hpids5 hrm5))]
  set s6 := s5.setReg "tiled_a" .real [BLOCK_K] (tiledA s input_ptr K xm_stride xk_stride BLOCK_K) with hs6
  have hli6 : s6.regs .nat [] "lora_index" = some (Tile.scalar (loraIdx s lora_indices)) := by
    rw [hs6, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("lora_index":RegName) ≠ "tiled_a" by decide)]; exact hli5
  have hsn6 : s6.regs .nat [] "pid_sn" = some (Tile.scalar (s.pids 0)) := by
    rw [hs6, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("pid_sn":RegName) ≠ "tiled_a" by decide)]; exact hsn5
  have hck6 : s6.regs .nat [] "cur_batch" = some (Tile.scalar (s.pids 1)) := by
    rw [hs6, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("cur_batch":RegName) ≠ "tiled_a" by decide)]; exact hck5
  have hpids6 : s6.pids = s.pids := by rw [hs6, BlockState.setReg_pids]; exact hpids5
  -- b_ptr
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (bptr_eval lora_ptr lora_indices split_n_length l0_stride lora_k_stride s s6 hli6 hsn6 hpids6))]
  set s7 := s6.setReg "b_ptr" .ptr [] (Tile.scalar (bPtrVal s lora_ptr lora_indices split_n_length l0_stride lora_k_stride)) with hs7
  have hsn7 : s7.regs .nat [] "pid_sn" = some (Tile.scalar (s.pids 0)) := by
    rw [hs7, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("pid_sn":RegName) ≠ "b_ptr" by decide)]; exact hsn6
  have hck7 : s7.regs .nat [] "cur_batch" = some (Tile.scalar (s.pids 1)) := by
    rw [hs7, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("cur_batch":RegName) ≠ "b_ptr" by decide)]; exact hck6
  have hpids7 : s7.pids = s.pids := by rw [hs7, BlockState.setReg_pids]; exact hpids6
  -- c_ptr
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (cptr_eval out_ptr split_n_length cm_stride s s7 hck7 hsn7 hpids7))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  refine ⟨by simp only [BlockState.setReg_pids]; exact hpids7, dvd_zero _, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · -- offset_k
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_k":RegName) ≠ "c_ptr" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_k":RegName) ≠ "b_ptr" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_k":RegName) ≠ "tiled_a" by decide)]
    exact hok5
  · -- offset_n
    rw [hs7, hs6, hs5]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_n":RegName) ≠ "c_ptr" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_n":RegName) ≠ "b_ptr" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_n":RegName) ≠ "tiled_a" by decide),
      BlockState.setReg_same]
  · -- tiled_a
    rw [hs7, hs6]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("tiled_a":RegName) ≠ "c_ptr" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("tiled_a":RegName) ≠ "b_ptr" by decide),
      BlockState.setReg_same]
  · -- b_ptr
    rw [hs7]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("b_ptr":RegName) ≠ "c_ptr" by decide),
      BlockState.setReg_same]
  · -- c_ptr
    rw [BlockState.setReg_same]
  · -- readMem lora_ptr
    funext ofs
    rw [BlockState.setReg_readMem, hs7, BlockState.setReg_readMem, hs6, BlockState.setReg_readMem]
    exact congrFun (congrFun hrm5 lora_ptr) ofs
  · -- output readback at i = 0: nothing written, every lane preserved
    intro m
    rw [hs7, hs6, hs5, hs4, hs3, hs2, hs1]
    simp only [Nat.not_lt_zero, if_false, BlockState.setReg_readMem]

/-! ### Loop body eval helpers -/

/-- The loaded LoRA-B tile for block `c`, lane `(j,k)`:
`tiledB[j,k] = if (c·BLOCK_N+j < snl ∧ k<K) then W[c·BLOCK_N+j,k] else 0`. -/
noncomputable def tiledB (s0 : BlockState) (lora_ptr : RegionName)
    (lora_indices : Region .nat)
    (K split_n_length l0_stride lora_k_stride lora_n_stride BLOCK_N BLOCK_K c : Nat) :
    Tile .real [BLOCK_N, BLOCK_K] :=
  ⟨fun idx : TileIndex [BLOCK_N, BLOCK_K] =>
    some (if (c*BLOCK_N + idx.1.val < split_n_length ∧ idx.2.1.val < K)
      then bElem s0 lora_ptr lora_indices split_n_length l0_stride lora_k_stride
        lora_n_stride (c*BLOCK_N + idx.1.val) idx.2.1.val else 0)⟩

theorem currentN_eval (BLOCK_N c : Nat) (s : BlockState)
    (hn : s.regs .nat [] "n" = some (Tile.scalar (c * BLOCK_N)))
    (hon : s.regs .nat [BLOCK_N] "offset_n" = some (Tile.vec (fun j : Fin BLOCK_N => j.val))) :
    evalOp (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "n") (Op.ref .nat [BLOCK_N] "offset_n")) s
      = some (Tile.vec (fun j : Fin BLOCK_N => c * BLOCK_N + j.val)) := by
  simp only [evalOp_add, evalOp_ref, hn, hon, Option.bind]
  apply congrArg some
  apply Tile.ext
  intro j
  simp only [Tile.bop, Tile.scalar, Tile.vec, Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add]

theorem tiledB_eval (lora_ptr : RegionName) (lora_indices : Region .nat)
    (K split_n_length l0_stride lora_k_stride lora_n_stride BLOCK_N BLOCK_K c : Nat)
    (s0 s : BlockState)
    (hbp : s.regs .ptr [] "b_ptr" = some (Tile.scalar
      (bPtrVal s0 lora_ptr lora_indices split_n_length l0_stride lora_k_stride)))
    (hcn : s.regs .nat [BLOCK_N] "current_n" = some (Tile.vec (fun j : Fin BLOCK_N => c*BLOCK_N + j.val)))
    (hok : s.regs .nat [BLOCK_K] "offset_k" = some (Tile.vec (fun k : Fin BLOCK_K => k.val)))
    (_hpids : s.pids = s0.pids) (hrm : s.readMem lora_ptr = s0.readMem lora_ptr) :
    evalOp (Op.load .real
      (MemAccess.ptr
        (Op.ptrAdd (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "b_ptr")
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "current_n")) (Op.constNat lora_k_stride)))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_K] "offset_k")) (Op.constNat lora_n_stride))))
      (MaskOpt.maskOther
        (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.lt .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "current_n")) (Op.constNat split_n_length))
          (Op.lt .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_K] "offset_k")) (Op.constNat K)))
        ((Op.const 0.0).broadcast [BLOCK_N, BLOCK_K]))) s
      = some (tiledB s0 lora_ptr lora_indices K split_n_length l0_stride lora_k_stride
          lora_n_stride BLOCK_N BLOCK_K c) := by
  simp only [evalOp, evalOp.eq_def, evalOp_ref, hbp, hcn, hok, Option.bind, Option.map,
    Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim, Tile.scalar, Tile.vec,
    Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul, ComparableDType.lt]
  apply congrArg some
  apply Tile.ext
  intro idx
  simp only [Tile.cop, Tile.expandDim, Tile.ptrAdd, Tile.bop, Tile.scalar]
  simp only [TileShape.dropInsertedIndex]
  simp only [tiledB, bElem, bPtrVal, loraIdx]
  by_cases hm : c*BLOCK_N + idx.1.val < split_n_length
  · by_cases hk : idx.2.1.val < K
    · simp only [hm, hk, decide_true, Bool.and_self, if_true, and_true, BlockState.readMemValue_real]
      rw [hrm]
    · simp only [hk, decide_false, Bool.and_false, Bool.false_eq_true, if_false, and_false]; norm_num
  · simp only [hm, decide_false, Bool.false_and, Bool.false_eq_true, if_false, false_and]; norm_num

theorem acc_eval (BLOCK_N BLOCK_K : Nat) (ta : Tile .real [BLOCK_K])
    (tb : Tile .real [BLOCK_N, BLOCK_K]) (s : BlockState)
    (hta : s.regs .real [BLOCK_K] "tiled_a" = some ta)
    (htb : s.regs .real [BLOCK_N, BLOCK_K] "tiled_b" = some tb) :
    evalOp (Op.reduceSum ⟨1, by simp⟩ Bool.false
      (Op.mul .real (Broadcast.leadL (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BLOCK_K] "tiled_a") (Op.ref .real [BLOCK_N, BLOCK_K] "tiled_b"))) s
      = some (Tile.reduceSum (shape := [BLOCK_N, BLOCK_K]) ⟨1, by simp⟩ Bool.false
          (Tile.bop NumericDType.real.mul (Broadcast.leadL (Broadcast.consSame Broadcast.nil)) ta tb)) := by
  simp only [evalOp, evalOp_ref, hta, htb, Option.bind, Option.map]
  rfl

/-- Range-to-`Fin` conversion: `Σ_{k:Fin BK} (if k<K then f k else 0) = Σ_{k<K} f k`. -/
theorem sum_fin_ite_range (BLOCK_K K : Nat) (hKB : K ≤ BLOCK_K) (f : Nat → ℝ) :
    (Finset.univ : Finset (Fin BLOCK_K)).sum (fun k : Fin BLOCK_K => if k.val < K then f k.val else 0)
      = (Finset.range K).sum f := by
  rw [Fin.sum_univ_eq_sum_range (fun k => if k < K then f k else 0) BLOCK_K]
  rw [← Finset.sum_subset (s₁ := Finset.range K) (s₂ := Finset.range BLOCK_K)
      (by intro x hx; rw [Finset.mem_range] at hx ⊢; omega)
      (fun x hx hxK => by rw [Finset.mem_range, not_lt] at hxK; rw [if_neg (by omega)])]
  apply Finset.sum_congr rfl
  intro x hx
  rw [if_pos (Finset.mem_range.mp hx)]

/-- **Reduction bridge**: at an *active* lane `j` (`c·BLOCK_N+j < snl`), the
`tl.sum(tiled_a·tiled_b,1)` reduction equals the genuine GEMV value
`gemvSpec(c·BLOCK_N+j)` (using `K ≤ BLOCK_K`). -/
theorem reduceSum_active_eq_spec
    (input_ptr lora_ptr : RegionName) (lora_indices : Region .nat)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      BLOCK_N BLOCK_K c : Nat) (s : BlockState) (j : Fin BLOCK_N)
    (hKB : K ≤ BLOCK_K) (hact : c*BLOCK_N + j.val < split_n_length) :
    (Tile.reduceSum (shape := [BLOCK_N, BLOCK_K]) ⟨1, by simp⟩ Bool.false
      (Tile.bop NumericDType.real.mul (Broadcast.leadL (Broadcast.consSame Broadcast.nil))
        (tiledA s input_ptr K xm_stride xk_stride BLOCK_K)
        (tiledB s lora_ptr lora_indices K split_n_length l0_stride lora_k_stride
          lora_n_stride BLOCK_N BLOCK_K c))).data (j, PUnit.unit)
      = some (gemvSpec s input_ptr lora_ptr lora_indices K split_n_length
          xm_stride xk_stride l0_stride lora_k_stride lora_n_stride (c*BLOCK_N + j.val)) := by
  simp only [Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
    TileShape.insertAxisIndex, Tile.bop, tiledA, tiledB, Broadcast.leftIndex, Broadcast.rightIndex]
  have hterm : ∀ k : Fin BLOCK_K,
      NumericDType.real.mul (some (if k.val < K then aElem s input_ptr xm_stride xk_stride k.val else 0))
        (some (if c*BLOCK_N + j.val < split_n_length ∧ k.val < K then
            bElem s lora_ptr lora_indices split_n_length l0_stride lora_k_stride lora_n_stride
              (c*BLOCK_N+j.val) k.val else 0))
      = ((if k.val < K then aElem s input_ptr xm_stride xk_stride k.val *
            bElem s lora_ptr lora_indices split_n_length l0_stride lora_k_stride lora_n_stride
              (c*BLOCK_N+j.val) k.val else 0 : ℝ) : WithBot ℝ) := by
    intro k
    by_cases hk : k.val < K
    · simp only [hk, hact, true_and, if_true, NumericDType.mul, WithBot.realMul, Option.map₂,
        Option.bind, Option.map]
      rfl
    · simp only [hk, if_false, and_false, NumericDType.mul, WithBot.realMul, Option.map₂,
        Option.bind, Option.map]
      norm_num
      rfl
  rw [Finset.sum_congr rfl (fun k _ => hterm k), ← WithBot.coe_sum, gemvSpec, gemmSum,
      ← sum_fin_ite_range BLOCK_K K hKB
        (fun k => aElem s input_ptr xm_stride xk_stride k *
          bElem s lora_ptr lora_indices split_n_length l0_stride lora_k_stride lora_n_stride
            (c*BLOCK_N+j.val) k)]
  rfl

/-- The accumulator tile (reduceSum of the elementwise product of two all-`some`
tiles) has `some` data at every lane. -/
theorem accT_data_some (BN BK : Nat) (ga : Fin BK → ℝ) (gb : Fin BN → Fin BK → ℝ)
    (j : Fin BN) :
    ((Tile.reduceSum (shape := [BN, BK]) ⟨1, by simp⟩ Bool.false
      (Tile.bop NumericDType.real.mul (Broadcast.leadL (Broadcast.consSame Broadcast.nil))
        (⟨fun idx => some (ga idx.1)⟩ : Tile .real [BK])
        (⟨fun idx => some (gb idx.1 idx.2.1)⟩ : Tile .real [BN, BK]))).data (j, PUnit.unit))
      = some (WithBot.unbotD 0
          ((Tile.reduceSum (shape := [BN, BK]) ⟨1, by simp⟩ Bool.false
            (Tile.bop NumericDType.real.mul (Broadcast.leadL (Broadcast.consSame Broadcast.nil))
              (⟨fun idx => some (ga idx.1)⟩ : Tile .real [BK])
              (⟨fun idx => some (gb idx.1 idx.2.1)⟩ : Tile .real [BN, BK]))).data (j, PUnit.unit))) := by
  have h : (Tile.reduceSum (shape := [BN, BK]) ⟨1, by simp⟩ Bool.false
      (Tile.bop NumericDType.real.mul (Broadcast.leadL (Broadcast.consSame Broadcast.nil))
        (⟨fun idx => some (ga idx.1)⟩ : Tile .real [BK])
        (⟨fun idx => some (gb idx.1 idx.2.1)⟩ : Tile .real [BN, BK]))).data (j, PUnit.unit)
      = some ((Finset.univ : Finset (Fin BK)).sum (fun k => ga k * gb j k)) := by
    simp only [Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
      TileShape.insertAxisIndex, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex]
    rw [Finset.sum_congr rfl (fun k _ =>
          show NumericDType.real.mul (some (ga k)) (some (gb j k)) = ((ga k * gb j k : ℝ) : WithBot ℝ) from rfl),
        ← WithBot.coe_sum]
    rfl
  rw [h]; rfl

/-- The masked GEMV store reduces to a `writeMem` foldl over local lanes. -/
theorem store_step (out_ptr : RegionName) (split_n_length cn_stride cpvo BLOCK_N : Nat)
    (cN : Fin BLOCK_N → Nat) (vals : Fin BLOCK_N → ℝ) (s : BlockState)
    (hcp : s.regs .ptr [] "c_ptr" = some (Tile.scalar ((out_ptr, cpvo) : RegionName × Nat)))
    (hcn : s.regs .nat [BLOCK_N] "current_n" = some (Tile.vec cN))
    (hacc : s.regs .real [BLOCK_N] "accumulator" = some ⟨fun idx : TileIndex [BLOCK_N] => some (vals idx.1)⟩) :
    stepStmt (Stmt.store .real [BLOCK_N]
      (MemAccess.ptr (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "c_ptr")
          (Op.mul .nat Broadcast.scalarR (Op.ref .nat [BLOCK_N] "current_n") (Op.constNat cn_stride))))
      (Op.ref .real [BLOCK_N] "accumulator")
      (MaskOpt.mask (Op.lt .nat Broadcast.scalarR (Op.ref .nat [BLOCK_N] "current_n") (Op.constNat split_n_length)))) s
      = some ((TileShape.allIndices [BLOCK_N]).foldl
          (fun acc (idx : TileIndex [BLOCK_N]) =>
            if decide (cN idx.1 < split_n_length) then acc.writeMem out_ptr (cpvo + cN idx.1 * cn_stride) (vals idx.1) else acc) s) := by
  simp only [stepStmt, evalOp, evalOp.eq_def, evalOp_ref, hcp, hcn, hacc, Option.bind, Option.map,
    Tile.bop, Tile.cop, Tile.ptrAdd, Tile.scalar, Tile.vec, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.mul, ComparableDType.lt]
  apply congrArg some
  apply List.foldl_ext
  intro acc idx _
  simp only [Region.cast, BlockState.writeMemTyped_real, FloatDType.real_storeValue, WithBot.unbotD_some]
  by_cases h : cN idx.1 < split_n_length
  · simp [h]
  · simp [h]

set_option maxHeartbeats 4000000 in
/-- **Step**: one `for n` body iteration advances the writeback invariant by one
`BLOCK_N`-block — it stores `gemvSpec` at every active global lane in the block,
preserving earlier blocks (needs global output-offset injectivity and
`out_ptr ≠ lora_ptr`). -/
theorem gemvWbInv_step
    (input_ptr lora_ptr out_ptr : RegionName) (lora_indices : Region .nat)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride BLOCK_N BLOCK_K : Nat)
    (s0 sorig : BlockState) (hBN : 0 < BLOCK_N) (hKB : K ≤ BLOCK_K)
    (hol : out_ptr ≠ lora_ptr)
    (hinj : Function.Injective
      (fun m : Fin split_n_length => outOffG s0 split_n_length cm_stride cn_stride m.val))
    (i : Nat) (s : BlockState) (hilt : i < split_n_length)
    (hinv : gemvWbInv input_ptr lora_ptr out_ptr lora_indices K split_n_length
        xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
        BLOCK_N BLOCK_K s0 sorig i s) :
    ∃ s', stepStmts (loopBody K split_n_length lora_k_stride lora_n_stride cn_stride BLOCK_N BLOCK_K)
        (s.setReg "n" .nat [] (Tile.scalar i)) = some s'
      ∧ gemvWbInv input_ptr lora_ptr out_ptr lora_indices K split_n_length
          xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
          BLOCK_N BLOCK_K s0 sorig (i + BLOCK_N) s' := by
  obtain ⟨hpids, ⟨c, hc⟩, hok, hon, hta, hbp, hcp, hrml, hread⟩ := hinv
  subst hc
  rw [show BLOCK_N*c = c*BLOCK_N from Nat.mul_comm _ _] at *
  -- the kk-set state
  set sk := s.setReg "n" .nat [] (Tile.scalar (c*BLOCK_N)) with hsk
  have hskpids : sk.pids = s0.pids := by rw [hsk, BlockState.setReg_pids, hpids]
  have hskrm : sk.readMem = s.readMem := by funext rg ofs; rw [hsk]; simp [BlockState.setReg_readMem]
  have hn_sk : sk.regs .nat [] "n" = some (Tile.scalar (c*BLOCK_N)) := by rw [hsk, BlockState.setReg_same]
  have hon_sk : sk.regs .nat [BLOCK_N] "offset_n" = some (Tile.vec (fun j : Fin BLOCK_N => j.val)) := by
    rw [hsk, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_n":RegName) ≠ "n" by decide)]; exact hon
  have hok_sk : sk.regs .nat [BLOCK_K] "offset_k" = some (Tile.vec (fun k : Fin BLOCK_K => k.val)) := by
    rw [hsk, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_k":RegName) ≠ "n" by decide)]; exact hok
  have hta_sk : sk.regs .real [BLOCK_K] "tiled_a" = some (tiledA s0 input_ptr K xm_stride xk_stride BLOCK_K) := by
    rw [hsk, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("tiled_a":RegName) ≠ "n" by decide)]; exact hta
  have hbp_sk : sk.regs .ptr [] "b_ptr" = some (Tile.scalar (bPtrVal s0 lora_ptr lora_indices split_n_length l0_stride lora_k_stride)) := by
    rw [hsk, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("b_ptr":RegName) ≠ "n" by decide)]; exact hbp
  have hcp_sk : sk.regs .ptr [] "c_ptr" = some (Tile.scalar (cPtrVal s0 out_ptr split_n_length cm_stride)) := by
    rw [hsk, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("c_ptr":RegName) ≠ "n" by decide)]; exact hcp
  -- step current_n
  unfold loopBody
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (currentN_eval BLOCK_N c sk hn_sk hon_sk))]
  set s1 := sk.setReg "current_n" .nat [BLOCK_N] (Tile.vec (fun j : Fin BLOCK_N => c*BLOCK_N + j.val)) with hs1
  have hcn1 : s1.regs .nat [BLOCK_N] "current_n" = some (Tile.vec (fun j : Fin BLOCK_N => c*BLOCK_N + j.val)) := by
    rw [hs1, BlockState.setReg_same]
  have hbp1 : s1.regs .ptr [] "b_ptr" = some (Tile.scalar (bPtrVal s0 lora_ptr lora_indices split_n_length l0_stride lora_k_stride)) := by
    rw [hs1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("b_ptr":RegName) ≠ "current_n" by decide)]; exact hbp_sk
  have hok1 : s1.regs .nat [BLOCK_K] "offset_k" = some (Tile.vec (fun k : Fin BLOCK_K => k.val)) := by
    rw [hs1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_k":RegName) ≠ "current_n" by decide)]; exact hok_sk
  have hs1pids : s1.pids = s0.pids := by rw [hs1, BlockState.setReg_pids, hskpids]
  have hs1rm : s1.readMem = s.readMem := by funext rg ofs; rw [hs1]; simp only [BlockState.setReg_readMem]; exact congrFun (congrFun hskrm rg) ofs
  have hs1rml : s1.readMem lora_ptr = s0.readMem lora_ptr := by rw [hs1rm]; exact hrml
  -- step tiled_b
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (tiledB_eval lora_ptr lora_indices K split_n_length l0_stride lora_k_stride lora_n_stride
          BLOCK_N BLOCK_K c s0 s1 hbp1 hcn1 hok1 hs1pids hs1rml))]
  set s2 := s1.setReg "tiled_b" .real [BLOCK_N, BLOCK_K]
    (tiledB s0 lora_ptr lora_indices K split_n_length l0_stride lora_k_stride lora_n_stride BLOCK_N BLOCK_K c) with hs2
  have hta2 : s2.regs .real [BLOCK_K] "tiled_a" = some (tiledA s0 input_ptr K xm_stride xk_stride BLOCK_K) := by
    rw [hs2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("tiled_a":RegName) ≠ "tiled_b" by decide), hs1,
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("tiled_a":RegName) ≠ "current_n" by decide)]; exact hta_sk
  have htb2 : s2.regs .real [BLOCK_N, BLOCK_K] "tiled_b" = some (tiledB s0 lora_ptr lora_indices K split_n_length l0_stride lora_k_stride lora_n_stride BLOCK_N BLOCK_K c) := by
    rw [hs2, BlockState.setReg_same]
  -- step accumulator
  erw [stepStmts.cons_some (stepStmt_assign_eq_some
        (acc_eval BLOCK_N BLOCK_K (tiledA s0 input_ptr K xm_stride xk_stride BLOCK_K)
          (tiledB s0 lora_ptr lora_indices K split_n_length l0_stride lora_k_stride lora_n_stride BLOCK_N BLOCK_K c)
          s2 hta2 htb2))]
  set accT : Tile .real [BLOCK_N] :=
    Tile.reduceSum (shape := [BLOCK_N, BLOCK_K]) ⟨1, by simp⟩ Bool.false
      (Tile.bop NumericDType.real.mul (Broadcast.leadL (Broadcast.consSame Broadcast.nil))
        (tiledA s0 input_ptr K xm_stride xk_stride BLOCK_K)
        (tiledB s0 lora_ptr lora_indices K split_n_length l0_stride lora_k_stride lora_n_stride BLOCK_N BLOCK_K c)) with haccT
  set s3 := s2.setReg "accumulator" .real [BLOCK_N] accT with hs3
  have hcp3 : s3.regs .ptr [] "c_ptr" = some (Tile.scalar (cPtrVal s0 out_ptr split_n_length cm_stride)) := by
    rw [hs3, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("c_ptr":RegName) ≠ "accumulator" by decide), hs2,
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("c_ptr":RegName) ≠ "tiled_b" by decide), hs1,
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("c_ptr":RegName) ≠ "current_n" by decide)]; exact hcp_sk
  have hcn3 : s3.regs .nat [BLOCK_N] "current_n" = some (Tile.vec (fun j : Fin BLOCK_N => c*BLOCK_N + j.val)) := by
    rw [hs3, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("current_n":RegName) ≠ "accumulator" by decide), hs2,
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("current_n":RegName) ≠ "tiled_b" by decide)]; exact hcn1
  have hacc3 : s3.regs .real [BLOCK_N] "accumulator" = some ⟨fun idx : TileIndex [BLOCK_N] => accT.data (idx.1, PUnit.unit)⟩ := by
    rw [hs3, BlockState.setReg_same]; apply congrArg some; apply Tile.ext; intro idx; rfl
  -- step store
  erw [stepStmts.cons_some (store_step out_ptr split_n_length cn_stride
        (s0.pids 1 * cm_stride + s0.pids 0 * split_n_length) BLOCK_N
        (fun j : Fin BLOCK_N => c*BLOCK_N + j.val)
        (fun j : Fin BLOCK_N => WithBot.unbotD 0 (accT.data (j, PUnit.unit))) s3
        (by rw [hcp3]; rfl) hcn3 ?_)]
  · rw [stepStmts.nil]
    -- the store foldl, with offset/value/mask as functions of local lanes
    set offF : TileIndex [BLOCK_N] → Nat :=
      fun idx => (s0.pids 1 * cm_stride + s0.pids 0 * split_n_length) + (c*BLOCK_N + idx.1.val) * cn_stride with hoffF
    set valF : TileIndex [BLOCK_N] → ℝ :=
      fun idx => WithBot.unbotD 0 (accT.data (idx.1, PUnit.unit)) with hvalF
    set mskF : TileIndex [BLOCK_N] → Bool :=
      fun idx => decide (c*BLOCK_N + idx.1.val < split_n_length) with hmskF
    set foldS := (TileShape.allIndices [BLOCK_N]).foldl
        (fun acc (idx : TileIndex [BLOCK_N]) =>
          if mskF idx then acc.writeMem out_ptr (offF idx) (valF idx) else acc) s3 with hfoldS
    refine ⟨foldS, rfl, ?_⟩
    have hfpids : foldS.pids = s0.pids := by
      rw [hfoldS, foldl_store_pids, hs3, BlockState.setReg_pids, hs2,
        BlockState.setReg_pids, hs1, BlockState.setReg_pids, hskpids]
    -- value at active lane jm equals gemvSpec(c*BLOCK_N + jm)
    have hvalSpec : ∀ jm : Fin BLOCK_N, c*BLOCK_N + jm.val < split_n_length →
        valF (jm, PUnit.unit) = gemvSpec s0 input_ptr lora_ptr lora_indices K split_n_length
          xm_stride xk_stride l0_stride lora_k_stride lora_n_stride (c*BLOCK_N + jm.val) := by
      intro jm hjmlt
      have hb := reduceSum_active_eq_spec input_ptr lora_ptr lora_indices K split_n_length
        xm_stride xk_stride l0_stride lora_k_stride lora_n_stride BLOCK_N BLOCK_K c s0 jm hKB hjmlt
      show WithBot.unbotD 0 (accT.data (jm, PUnit.unit)) = _
      rw [show accT.data (jm, PUnit.unit) = some (gemvSpec s0 input_ptr lora_ptr lora_indices K
          split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride (c*BLOCK_N + jm.val))
        from by rw [haccT]; exact hb]
      rfl
    refine ⟨hfpids, ⟨c+1, by ring⟩, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · rw [hfoldS, foldl_store_regs, hs3,
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_k":RegName) ≠ "accumulator" by decide), hs2,
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_k":RegName) ≠ "tiled_b" by decide)]; exact hok1
    · rw [hfoldS, foldl_store_regs, hs3,
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_n":RegName) ≠ "accumulator" by decide), hs2,
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_n":RegName) ≠ "tiled_b" by decide), hs1,
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_n":RegName) ≠ "current_n" by decide)]; exact hon_sk
    · rw [hfoldS, foldl_store_regs, hs3,
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("tiled_a":RegName) ≠ "accumulator" by decide)]
      exact hta2
    · rw [hfoldS, foldl_store_regs, hs3,
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("b_ptr":RegName) ≠ "accumulator" by decide), hs2,
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("b_ptr":RegName) ≠ "tiled_b" by decide)]; exact hbp1
    · rw [hfoldS, foldl_store_regs]; exact hcp3
    · funext ofs
      rw [hfoldS, foldl_store_other_region offF valF mskF _ _ lora_ptr ofs hol.symm, hs3,
        BlockState.setReg_readMem, hs2, BlockState.setReg_readMem]
      exact congrFun hs1rml ofs
    · -- readback per global lane
      intro m
      have hmN : m.val < split_n_length := m.isLt
      have hs3rm : ∀ R o, s3.readMem R o = s.readMem R o := by
        intro R o; rw [hs3, BlockState.setReg_readMem, hs2, BlockState.setReg_readMem]
        exact congrFun (congrFun hs1rm R) o
      by_cases hin : (c*BLOCK_N ≤ m.val ∧ m.val < c*BLOCK_N + BLOCK_N)
      · -- m is in this block (lane jm); the store writes gemvSpec(m)
        set jm : Fin BLOCK_N := ⟨m.val - c*BLOCK_N, by omega⟩ with hjm
        have hjmeq : c*BLOCK_N + jm.val = m.val := by simp only [hjm]; omega
        have hofeq : offF (jm, PUnit.unit) = outOffG s0 split_n_length cm_stride cn_stride m.val := by
          simp only [hoffF, outOffG, hjmeq]
        rw [hfoldS, ← hofeq]
        rw [foldl_store_at offF valF mskF (offF (jm, PUnit.unit)) (TileShape.allIndices [BLOCK_N]) s3
          (jm, PUnit.unit) (TileShape.mem_allIndices [BLOCK_N] _)
          (by simp only [hmskF]; exact decide_eq_true (by rw [hjmeq]; exact hmN))
          rfl
          (by -- uniqueness among active members at this offset
            intro b hb hmb hofb
            have hbN : c*BLOCK_N + b.1.val < split_n_length := by simpa [hmskF] using hmb
            have hofb' : (⟨c*BLOCK_N + b.1.val, hbN⟩ : Fin split_n_length) =
                (⟨c*BLOCK_N + jm.val, by rw [hjmeq]; exact hmN⟩ : Fin split_n_length) := by
              apply hinj; simp only [outOffG]
              have : offF b = offF (jm, PUnit.unit) := hofb
              simp only [hoffF] at this; omega
            have : b.1.val = jm.val := by
              have := Fin.mk.injEq .. ▸ hofb'; omega
            exact Prod.ext (Fin.ext this) rfl)
          (TileShape.allIndices_nodup [BLOCK_N])]
        rw [hvalSpec jm (by rw [hjmeq]; exact hmN)]
        rw [if_pos (by omega : m.val < c*BLOCK_N + BLOCK_N), hjmeq]
      · -- m not in this block: preserved from s, dispatch on m < c*BLOCK_N
        rw [hfoldS]
        rw [foldl_store_preserve offF valF mskF (outOffG s0 split_n_length cm_stride cn_stride m.val)
          (TileShape.allIndices [BLOCK_N]) s3
          (by intro b hb hmb hofb
              have hbN : c*BLOCK_N + b.1.val < split_n_length := by simpa [hmskF] using hmb
              -- offsets equal + both active ⇒ global lanes equal (hinj)
              have heqlane : (⟨c*BLOCK_N + b.1.val, hbN⟩ : Fin split_n_length) = m := by
                apply hinj
                show outOffG s0 split_n_length cm_stride cn_stride (c*BLOCK_N + b.1.val)
                  = outOffG s0 split_n_length cm_stride cn_stride m.val
                have : offF b = outOffG s0 split_n_length cm_stride cn_stride m.val := hofb
                simp only [hoffF, outOffG] at this ⊢
                omega
              have hbm : c*BLOCK_N + b.1.val = m.val := by rw [← heqlane]
              have : b.1.val < BLOCK_N := b.1.isLt
              omega)]
        rw [hs3rm, hread m]
        by_cases hlt : m.val < c*BLOCK_N
        · rw [if_pos hlt, if_pos (by omega : m.val < c*BLOCK_N + BLOCK_N)]
        · rw [if_neg hlt, if_neg (by omega : ¬ m.val < c*BLOCK_N + BLOCK_N)]
  · -- accumulator readback: every lane of the reduceSum is `some`
    rw [hacc3]
    apply congrArg some
    apply Tile.ext
    intro idx
    exact accT_data_some BLOCK_N BLOCK_K
      (fun k => if k.val < K then aElem s0 input_ptr xm_stride xk_stride k.val else 0)
      (fun j k => if c*BLOCK_N + j.val < split_n_length ∧ k.val < K then
          bElem s0 lora_ptr lora_indices split_n_length l0_stride lora_k_stride lora_n_stride
            (c*BLOCK_N + j.val) k.val else 0)
      idx.1

/-! ### Composition: full exec correctness -/

set_option maxHeartbeats 2000000 in
/-- **Exec-level correctness**: running the full loop kernel from a clean state,
every active output lane `m < split_n_length` ends up holding the genuine GEMV
value `gemvSpec(m)`. -/
theorem gemv_exec_correct
    (input_ptr lora_ptr out_ptr : RegionName) (lora_indices : Region .nat)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride BLOCK_N BLOCK_K : Nat)
    (s s' : BlockState) (hBN : 0 < BLOCK_N) (hKB : K ≤ BLOCK_K) (hol : out_ptr ≠ lora_ptr)
    (hundef : ∀ rg o, s.undef rg o = 0)
    (hcn : 0 < cn_stride)
    (hExec : exec (bgmv_loop_surface input_ptr lora_ptr out_ptr lora_indices K
        split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
        cm_stride cn_stride BLOCK_N BLOCK_K) s = some s') :
    ∀ m : Fin split_n_length,
      s'.readMem out_ptr (outOffG s split_n_length cm_stride cn_stride m.val)
        = gemvSpec s input_ptr lora_ptr lora_indices K split_n_length
            xm_stride xk_stride l0_stride lora_k_stride lora_n_stride m.val := by
  have hinj : Function.Injective
      (fun m : Fin split_n_length => outOffG s split_n_length cm_stride cn_stride m.val) := by
    have heq : (fun m : Fin split_n_length => outOffG s split_n_length cm_stride cn_stride m.val)
        = (fun m : Fin split_n_length =>
            (s.pids 1 * cm_stride + s.pids 0 * split_n_length) + m.val * cn_stride) := by
      funext m; simp only [outOffG]
    rw [heq]; exact affine1D_inj _ cn_stride hcn
  -- prefix establishes P 0
  obtain ⟨sp, hsp, hP0⟩ := prefix_inv input_ptr lora_ptr out_ptr lora_indices K split_n_length
    xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
    BLOCK_N BLOCK_K s hundef
  -- drive the loop
  obtain ⟨final, sLoop, hLoopStmt, hfinal, hPLoop⟩ :=
    forRange_inv (idx := "n") (start := 0) (stop := split_n_length) (step := BLOCK_N)
      (Nat.pos_iff_ne_zero.mp hBN) hP0
      (fun i st hlt hinv => gemvWbInv_step input_ptr lora_ptr out_ptr lora_indices K
        split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
        BLOCK_N BLOCK_K s s hBN hKB hol hinj i st hlt hinv)
  -- assemble exec
  have hbody : exec (bgmv_loop_surface input_ptr lora_ptr out_ptr lora_indices K
      split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride BLOCK_N BLOCK_K) s = some sLoop := by
    rw [exec, body_split input_ptr lora_ptr out_ptr lora_indices K split_n_length
      xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride BLOCK_N BLOCK_K,
      stepStmts.append_some hsp, stepStmts.cons_some hLoopStmt, stepStmts.nil]
  rw [hbody] at hExec
  injection hExec with hExec
  subst hExec
  obtain ⟨_, _, _, _, _, _, _, _, hread⟩ := hPLoop
  intro m
  rw [hread m, if_pos (lt_of_lt_of_le m.isLt hfinal)]

set_option maxHeartbeats 2000000 in
/-- **Compute-facing correctness** for the full multi-block LoRA expand GEMV: the
masked store to `out_ptr` is compute-correct over every global active lane `m <
split_n_length`; each holds the genuine GEMV `Σ_{k<K} x[k]·W[m,k]`. -/
theorem gemv_compute_correct
    (input_ptr lora_ptr out_ptr : RegionName) (lora_indices : Region .nat)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride BLOCK_N BLOCK_K : Nat)
    (s : BlockState) (hBN : 0 < BLOCK_N) (hKB : K ≤ BLOCK_K) (hol : out_ptr ≠ lora_ptr)
    (hundef : ∀ rg o, s.undef rg o = 0)
    (hcn : 0 < cn_stride) :
    ComputeCorrect.Realizes
      (kernel := bgmv_loop_surface input_ptr lora_ptr out_ptr lora_indices K
        split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
        cm_stride cn_stride BLOCK_N BLOCK_K)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin split_n_length => True)
        (fun m => (out_ptr, outOffG s split_n_length cm_stride cn_stride m.val)))
      (expected := fun m : Fin split_n_length =>
        gemvSpec s input_ptr lora_ptr lora_indices K split_n_length
          xm_stride xk_stride l0_stride lora_k_stride lora_n_stride m.val) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [bgmv_loop_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro m _
  exact gemv_exec_correct input_ptr lora_ptr out_ptr lora_indices K split_n_length
    xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
    BLOCK_N BLOCK_K s s' hBN hKB hol hundef hcn hExec m

/-- **Full output summary**: the full LoRA expand GEMV surface lowers to the
algorithm layer, and the masked store realizes the genuine matrix-vector product
`out[m] = Σ_{k<K} x[k]·W[m,k]` at every active global lane `m < split_n_length`
(general `⌈split_n_length/BLOCK_N⌉`-block loop). -/
theorem gemv_full_output_summary
    (input_ptr lora_ptr out_ptr : RegionName) (lora_indices : Region .nat)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride BLOCK_N BLOCK_K : Nat)
    (s : BlockState) (hBN : 0 < BLOCK_N) (hKB : K ≤ BLOCK_K) (hol : out_ptr ≠ lora_ptr)
    (hundef : ∀ rg o, s.undef rg o = 0)
    (hcn : 0 < cn_stride) :
    (∃ alg, (bgmv_loop_surface input_ptr lora_ptr out_ptr lora_indices K
        split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
        cm_stride cn_stride BLOCK_N BLOCK_K).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := bgmv_loop_surface input_ptr lora_ptr out_ptr lora_indices K
        split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
        cm_stride cn_stride BLOCK_N BLOCK_K)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin split_n_length => True)
        (fun m => (out_ptr, outOffG s split_n_length cm_stride cn_stride m.val)))
      (expected := fun m : Fin split_n_length =>
        gemvSpec s input_ptr lora_ptr lora_indices K split_n_length
          xm_stride xk_stride l0_stride lora_k_stride lora_n_stride m.val) :=
  ⟨bgmv_loop_surface_toAlgorithm_supported input_ptr lora_ptr out_ptr lora_indices K
      split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride BLOCK_N BLOCK_K,
    gemv_compute_correct input_ptr lora_ptr out_ptr lora_indices K split_n_length
      xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
      BLOCK_N BLOCK_K s hBN hKB hol hundef hcn⟩

end Full

end VeriTile.Bench.TritonBenchG.LoraExpandGemv
