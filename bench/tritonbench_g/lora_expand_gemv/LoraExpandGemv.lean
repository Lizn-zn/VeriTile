import VeriTile.Triton

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

Arithmetic is over `ℝ` (not bit-accurate IEEE float). The full theorem holds for
arbitrary `split_n_length` and `BLOCK_N`
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
    ComputeCorrect.Realizes_without_Rounding
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
specification gemv_full_output_summary
    (input_ptr lora_ptr out_ptr : RegionName) (lora_indices : Region .nat)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride BLOCK_N BLOCK_K : Nat)
    (s : BlockState) (hBN : 0 < BLOCK_N) (hKB : K ≤ BLOCK_K) (hol : out_ptr ≠ lora_ptr)
    (hundef : ∀ rg o, s.undef rg o = 0)
    (hcn : 0 < cn_stride) :
    (∃ alg, (bgmv_loop_surface input_ptr lora_ptr out_ptr lora_indices K
        split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
        cm_stride cn_stride BLOCK_N BLOCK_K).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
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

/-! ## The `⊨[R]` streaming metadata emit headline (wave-5 S3 genre)

Everything below is purely additive; the exact surface above is untouched.
This is the first consumer of the metadata-parametrized per-step emit skin
`StreamMetaEmitMasked3DKernelIO₂` (streaming genre, style S3 on the 3-D pid
grid): the store sits **inside** the `for n` loop, so the output is a
per-step `BLOCK_N`-lane window family, and the per-batch adapter choice
`lora_indices[cur_batch]` is a pre-loop `.nat` metadata slot whose loaded
value parametrizes the `loraB` read window.

Structure of the `execR R` story: this kernel has **zero rounding events**
(the `.nat` slot load, the masked `.real` loads, the `.real` reduce and the
masked `.real` in-loop stores — the verified `CAST_TYPE = false` path has no
`castFloat` at all). The whole body therefore collapses verbatim onto the
exact stepper, and the proven `prefix_inv` / `gemvWbInv` / `gemvWbInv_step` /
`forRange_inv` stack above is reused unchanged; the `⊨[R]` face adds only the
`TraceSafeR` walk, the per-cell memory frame, and the stream-lane spec
bridge. -/

section IOFace

open Full
open scoped VeriTile.Triton.StreamMetaEmitMasked3DKernelIO₂

set_option linter.unusedVariables false

/-! ### Stream geometry: trip count and lane bridge -/

/-- Trip count of the kernel's `for n in range(0, split_n_length, BLOCK_N)`
loop: `⌈split_n_length / BLOCK_N⌉`. -/
def bgmvNumSteps (snl B : Nat) : Nat := (snl + B - 1) / B

private theorem bgmvNumSteps_mul_ge (snl B : Nat) (hB : 0 < B) :
    snl ≤ bgmvNumSteps snl B * B := by
  rcases Nat.eq_zero_or_pos snl with rfl | hN
  · exact Nat.zero_le _
  · unfold bgmvNumSteps
    have heq : snl + B - 1 = (snl - 1) + B := by omega
    rw [heq, Nat.add_div_right _ hB]
    have h2 : (snl - 1) % B + 1 ≤ B := Nat.mod_lt _ hB
    calc snl = (snl - 1) + 1 := by omega
      _ = (snl - 1) / B * B + ((snl - 1) % B + 1) := by
          rw [← Nat.add_assoc, Nat.div_add_mod']
      _ ≤ (snl - 1) / B * B + B := Nat.add_le_add_left h2 _
      _ = ((snl - 1) / B + 1) * B := (Nat.succ_mul _ _).symm

private theorem bgmvStep_lt_numSteps (snl B i : Nat) (hB : 0 < B) (hi : i < snl) :
    i / B < bgmvNumSteps snl B := by
  have h2 : i / B * B < bgmvNumSteps snl B * B :=
    Nat.lt_of_le_of_lt (Nat.div_mul_le_self i B)
      (Nat.lt_of_lt_of_le hi (bgmvNumSteps_mul_ge snl B hB))
  exact Nat.lt_of_mul_lt_mul_right h2

/-- The `loraB`-stream lane feeding output lane `j` at rank key `k`: row `j`
paired with `k`, row-major over the `[BLOCK_N, BLOCK_K]` per-step `b`-tile,
via the shared `Lane2D` bridge. -/
def bLane (BLOCK_N BLOCK_K : Nat) (j : Fin BLOCK_N) (k : Fin BLOCK_K) :
    Fin (BLOCK_N * BLOCK_K) :=
  Lane2D.encode (j, k, PUnit.unit)

private theorem bLane_div (BLOCK_N BLOCK_K : Nat) (j : Fin BLOCK_N) (k : Fin BLOCK_K) :
    (bLane BLOCK_N BLOCK_K j k).val / BLOCK_K = j.val :=
  Lane2D.encode_div _

private theorem bLane_mod (BLOCK_N BLOCK_K : Nat) (j : Fin BLOCK_N) (k : Fin BLOCK_K) :
    (bLane BLOCK_N BLOCK_K j k).val % BLOCK_K = k.val :=
  Lane2D.encode_mod _

/-! ### IO signature -/

/-- **Streaming metadata IO signature** of the verified `bgmv_loop_surface`
on the metadata-parametrized per-step emit skin (S3: in-loop store, 3-D pid
grid). The kernel is 2-pid (`pid_sn = pids 0`, `cur_batch = pids 1`); the
skin's third pid is quantified and every window ignores it. The single
`.nat` metadata slot is the kernel's own per-batch adapter index, loaded at
cell `cur_batch = pid₁` of `lora_indices`; the loaded slot value `m 0` banks
the `loraB` read window (`l0_stride · m 0`, in place of the in-state
`loraIdx` read). Step `t` of the loop (at `n = t·BLOCK_N`) reads the static
`BLOCK_K`-lane `input` tile (`read1` ignores `t` — the genre's degenerate
static stream: the kernel loads it once, pre-loop, and register-caches it)
and the `[BLOCK_N, BLOCK_K]` `loraB` tile (`read2`, row-major lane
`r·BLOCK_K + k`), and masked-stores the `BLOCK_N`-lane output window
(`write`) at the **`.real`** grid (`outDType` default — the store is an
untyped `tl.store`, no quantization event). The windows transcribe the
kernel's pointer arithmetic verbatim:

* `read1` lane `k`: `pid₁·xm_stride + k·xk_stride`; `mask1`: `k < K`
  (the kernel's `offset_k < K` load mask).
* `read2` step `t`, lane `l = (r, k)`:
  `l0_stride·m 0 + pid₀·split_n_length·lora_k_stride +
  (t·BLOCK_N + r)·lora_k_stride + k·lora_n_stride`; `mask2`:
  `t·BLOCK_N + r < split_n_length ∧ k < K` (the kernel's `b_ptr_mask`).
* `write` step `t`, lane `j`:
  `pid₁·cm_stride + pid₀·split_n_length + (t·BLOCK_N + j)·cn_stride`;
  `writeMask`: `t·BLOCK_N + j < split_n_length` (the kernel's `c_mask`).

**Sentinel disclosure** (mirrors the surface's docstring): Python's signed
`lora_index == -1` early return lives at the **trusted host boundary** — the
host only launches the active body when `lora_index ≠ -1` — so this
signature's `.nat` slot only covers active launches and `writeMask` carries
**no** sentinel gate. The skin's empty-write-window sentinel idiom is *not*
exercised here; expressing the `-1` skip as an `.int` slot gating `writeMask`
would need the guarded surface (`bgmv_expand_surface`), not the verified
active-body surface. -/
def loraExpandGemvIO (input_ptr lora_ptr out_ptr : RegionName)
    (lora_indices : Region .nat)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride
      lora_n_stride cm_stride cn_stride BLOCK_N BLOCK_K : Nat) :
    StreamMetaEmitMasked3DKernelIO₂ where
  kernel := bgmv_loop_surface input_ptr lora_ptr out_ptr lora_indices K
    split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
    cm_stride cn_stride BLOCK_N BLOCK_K
  inp1 := input_ptr
  inp2 := lora_ptr
  out := out_ptr
  nMeta := 1
  sty := fun _ => ChanTy.nat
  mbuf := fun _ => lora_indices.cast
  mwin := fun _ _ pid₁ _ => pid₁
  T := bgmvNumSteps split_n_length BLOCK_N
  B1 := BLOCK_K
  B2 := BLOCK_N * BLOCK_K
  C := BLOCK_N
  read1 := fun _ pid₁ _ _ _ k => pid₁ * xm_stride + k.val * xk_stride
  read2 := fun pid₀ _ _ m t l =>
    l0_stride * m (⟨0, by omega⟩ : Fin 1)
      + pid₀ * split_n_length * lora_k_stride
      + (t.val * BLOCK_N + l.val / BLOCK_K) * lora_k_stride
      + (l.val % BLOCK_K) * lora_n_stride
  write := fun pid₀ pid₁ _ _ t j =>
    pid₁ * cm_stride + pid₀ * split_n_length + (t.val * BLOCK_N + j.val) * cn_stride
  mask1 := fun _ _ _ _ _ k => k.val < K
  mask2 := fun _ _ _ _ t l =>
    t.val * BLOCK_N + l.val / BLOCK_K < split_n_length ∧ l.val % BLOCK_K < K
  writeMask := fun _ _ _ _ t j => t.val * BLOCK_N + j.val < split_n_length

/-! ### The stream-lane spec bridge -/

/-- Under the stream pins, the exact stack's per-lane GEMV value
`gemvSpec(t·BLOCK_N + j) = Σ_{k<K} x[k]·W[t·BLOCK_N+j, k]` **is** the
stream-level guarded sum `Σ_{k<BLOCK_K} [k<K] xs[t,k]·ys[t, j·BLOCK_K+k]`
(via `sum_fin_ite_range`, using `K ≤ BLOCK_K`). The guard keeps the spec
reading only `mask1`/`mask2`-pinned lanes. -/
private theorem gemvSpec_eq_streamSum
    (input_ptr lora_ptr : RegionName) (lora_indices : Region .nat)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      BLOCK_N BLOCK_K : Nat) (hKB : K ≤ BLOCK_K) (s₀ : BlockState)
    (xs : Fin (bgmvNumSteps split_n_length BLOCK_N) → Fin BLOCK_K → ℝ)
    (ys : Fin (bgmvNumSteps split_n_length BLOCK_N) → Fin (BLOCK_N * BLOCK_K) → ℝ)
    (hx : ∀ (t : Fin (bgmvNumSteps split_n_length BLOCK_N)) (k : Fin BLOCK_K),
      k.val < K →
      s₀.readMem input_ptr (s₀.pids 1 * xm_stride + k.val * xk_stride) = xs t k)
    (hy : ∀ (t : Fin (bgmvNumSteps split_n_length BLOCK_N)) (l : Fin (BLOCK_N * BLOCK_K)),
      t.val * BLOCK_N + l.val / BLOCK_K < split_n_length → l.val % BLOCK_K < K →
      s₀.readMem lora_ptr (l0_stride * loraIdx s₀ lora_indices
          + s₀.pids 0 * split_n_length * lora_k_stride
          + (t.val * BLOCK_N + l.val / BLOCK_K) * lora_k_stride
          + (l.val % BLOCK_K) * lora_n_stride) = ys t l)
    (t : Fin (bgmvNumSteps split_n_length BLOCK_N)) (j : Fin BLOCK_N)
    (hj : t.val * BLOCK_N + j.val < split_n_length) :
    gemvSpec s₀ input_ptr lora_ptr lora_indices K split_n_length xm_stride
        xk_stride l0_stride lora_k_stride lora_n_stride (t.val * BLOCK_N + j.val)
      = ∑ k : Fin BLOCK_K, if k.val < K then
          xs t k * ys t (bLane BLOCK_N BLOCK_K j k) else 0 := by
  have hspec : gemvSpec s₀ input_ptr lora_ptr lora_indices K split_n_length xm_stride
        xk_stride l0_stride lora_k_stride lora_n_stride (t.val * BLOCK_N + j.val)
      = (Finset.range K).sum (fun kk =>
          aElem s₀ input_ptr xm_stride xk_stride kk
            * bElem s₀ lora_ptr lora_indices split_n_length l0_stride lora_k_stride
                lora_n_stride (t.val * BLOCK_N + j.val) kk) := rfl
  rw [hspec, ← sum_fin_ite_range BLOCK_K K hKB (fun kk =>
    aElem s₀ input_ptr xm_stride xk_stride kk
      * bElem s₀ lora_ptr lora_indices split_n_length l0_stride lora_k_stride
          lora_n_stride (t.val * BLOCK_N + j.val) kk)]
  refine Finset.sum_congr rfl fun k _ => ?_
  by_cases hk : k.val < K
  · rw [if_pos hk, if_pos hk]
    have hyl := hy t (bLane BLOCK_N BLOCK_K j k)
      (by rw [bLane_div]; exact hj) (by rw [bLane_mod]; exact hk)
    rw [bLane_div, bLane_mod] at hyl
    rw [← hx t k hk, ← hyl]
    rfl
  · rw [if_neg hk, if_neg hk]

/-! ### Cast-free collapses and the covered fragment -/

/-- `R.cast .real .real` is the exact cast — `R.roundW .real` is the
identity by the model's defining `round_real` (private copy of the shared
wave-5 helper; bench files never import each other). Not actually hit by
this kernel (the verified path has no `castFloat`), but kept in the collapse
simp sets for uniformity with the genre recipe. -/
private theorem Rcast_real_real (R : RoundingModel) :
    R.cast .real .real = FloatDType.cast .real .real := by
  funext v
  simp [RoundingModel.cast, FloatDType.cast]

/-- The 8-statement prologue is cast-free (`.nat` slot load, masked `.real`
input load with `other=0`, nat/ptr arithmetic): it steps identically under
`stepStmtsR R`, so the exact `prefix_inv` transports to `execR`. -/
private theorem bgmvPrefix_castFree (R : RoundingModel)
    (input_ptr lora_ptr out_ptr : RegionName) (lora_indices : Region .nat)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride
      cm_stride BLOCK_N BLOCK_K : Nat) (t : BlockState) :
    stepStmtsR R (prefixStmts input_ptr lora_ptr out_ptr lora_indices K split_n_length
        xm_stride xk_stride l0_stride lora_k_stride cm_stride BLOCK_N BLOCK_K) t
      = stepStmts (prefixStmts input_ptr lora_ptr out_ptr lora_indices K split_n_length
          xm_stride xk_stride l0_stride lora_k_stride cm_stride BLOCK_N BLOCK_K) t := by
  simp only [prefixStmts, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def]
  rfl

/-- The `for n` body is cast-free **including its in-loop masked `.real`
store**: `stepStmtR` delegates a `.real`-typed store to the exact
`writeMemTyped`, so the storing loop steps identically under `stepStmtsR R`
and the exact `gemvWbInv` stack transports to `execR`. -/
private theorem bgmvLoopBody_castFree (R : RoundingModel)
    (K split_n_length lora_k_stride lora_n_stride cn_stride BLOCK_N BLOCK_K : Nat)
    (t : BlockState) :
    stepStmtsR R (loopBody K split_n_length lora_k_stride lora_n_stride cn_stride
        BLOCK_N BLOCK_K) t
      = stepStmts (loopBody K split_n_length lora_k_stride lora_n_stride cn_stride
          BLOCK_N BLOCK_K) t := by
  simp only [loopBody, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def, Rcast_real_real, BlockState.writeMemTypedR]
  rfl

/-- The full loop surface sits inside the flat-memory bridge's covered
fragment (`FlattenOk`; the `forRange` clause recurses into the cast-free
body). -/
private theorem bgmv_flattenOk (input_ptr lora_ptr out_ptr : RegionName)
    (lora_indices : Region .nat)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride
      lora_n_stride cm_stride cn_stride BLOCK_N BLOCK_K : Nat) :
    ((bgmv_loop_surface input_ptr lora_ptr out_ptr lora_indices K split_n_length
        xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride
        cn_stride BLOCK_N BLOCK_K).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  rw [body_split input_ptr lora_ptr out_ptr lora_indices K split_n_length xm_stride
      xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride BLOCK_N BLOCK_K,
    prefix_split input_ptr lora_ptr out_ptr lora_indices K split_n_length xm_stride
      xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride BLOCK_N BLOCK_K]
  simp [prefixStmts, loopBody, StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-! ### Cell-level memory frames

The exact stack proves per-lane `readMem` values; the `⊨[R]` Hoare triple
additionally needs a per-**cell** frame, so the register-only prologue gets
a generic assigns-don't-touch-memory lemma and the storing loop body gets a
cell-level frame twin of `gemvWbInv_step`. -/

/-- A run of `assign` statements never touches memory (private copy of the
shared wave-5 helper). -/
private theorem stepStmts_assigns_mem :
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
          rw [stepStmts_assigns_mem rest
            (fun st' hst' => hall st' (List.mem_cons_of_mem _ hst')) h]
          rfl

/-- Cell-level frame of a `Bool`-masked exact `writeMem` scatter `foldl`:
every cell not hit by an active lane is untouched (private copy of the
shared wave-5 helper). -/
private theorem foldl_writeMem_bool_preserve_cell {α : Type}
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
/-- **Cell-level frame of one `for n` iteration** (the `mem` twin of
`gemvWbInv_step`, same walk): from the `gemvWbInv` register pins, one
storing body iteration leaves every cell off the
`{(out_ptr, outOffG m) : m < split_n_length}` write window untouched — the
masked scatter store only hits active lanes `i + j < split_n_length`, whose
offsets are `outOffG (i + j)`. -/
private theorem bgmv_loopBody_step_frame
    (input_ptr lora_ptr out_ptr : RegionName) (lora_indices : Region .nat)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride BLOCK_N BLOCK_K : Nat)
    (s0 st st' : BlockState) (i : Nat) (hiB : BLOCK_N ∣ i)
    (hok : st.regs .nat [BLOCK_K] "offset_k" = some (Tile.vec (fun k : Fin BLOCK_K => k.val)))
    (hon : st.regs .nat [BLOCK_N] "offset_n" = some (Tile.vec (fun j : Fin BLOCK_N => j.val)))
    (hta : st.regs .real [BLOCK_K] "tiled_a"
      = some (tiledA s0 input_ptr K xm_stride xk_stride BLOCK_K))
    (hbp : st.regs .ptr [] "b_ptr" = some (Tile.scalar
      (bPtrVal s0 lora_ptr lora_indices split_n_length l0_stride lora_k_stride)))
    (hcp : st.regs .ptr [] "c_ptr" = some (Tile.scalar
      (cPtrVal s0 out_ptr split_n_length cm_stride)))
    (hpids : st.pids = s0.pids)
    (hrml : st.readMem lora_ptr = s0.readMem lora_ptr)
    (hstep : stepStmts (loopBody K split_n_length lora_k_stride lora_n_stride cn_stride
        BLOCK_N BLOCK_K) (st.setReg "n" .nat [] (Tile.scalar i)) = some st')
    (r : RegionName) (oo : Nat)
    (hcond : r ≠ out_ptr ∨
      ∀ mm : Fin split_n_length, oo ≠ outOffG s0 split_n_length cm_stride cn_stride mm.val) :
    st'.mem r oo = st.mem r oo := by
  obtain ⟨c, hc⟩ := hiB
  subst hc
  rw [show BLOCK_N*c = c*BLOCK_N from Nat.mul_comm _ _] at hstep
  set sk := st.setReg "n" .nat [] (Tile.scalar (c*BLOCK_N)) with hsk
  have hskpids : sk.pids = s0.pids := by rw [hsk, BlockState.setReg_pids, hpids]
  have hskrm : sk.readMem = st.readMem := by
    funext rg ofs; rw [hsk]; simp [BlockState.setReg_readMem]
  have hn_sk : sk.regs .nat [] "n" = some (Tile.scalar (c*BLOCK_N)) := by
    rw [hsk, BlockState.setReg_same]
  have hon_sk : sk.regs .nat [BLOCK_N] "offset_n" = some (Tile.vec (fun j : Fin BLOCK_N => j.val)) := by
    rw [hsk, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_n":RegName) ≠ "n" by decide)]
    exact hon
  have hok_sk : sk.regs .nat [BLOCK_K] "offset_k" = some (Tile.vec (fun k : Fin BLOCK_K => k.val)) := by
    rw [hsk, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_k":RegName) ≠ "n" by decide)]
    exact hok
  have hta_sk : sk.regs .real [BLOCK_K] "tiled_a" = some (tiledA s0 input_ptr K xm_stride xk_stride BLOCK_K) := by
    rw [hsk, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("tiled_a":RegName) ≠ "n" by decide)]
    exact hta
  have hbp_sk : sk.regs .ptr [] "b_ptr" = some (Tile.scalar (bPtrVal s0 lora_ptr lora_indices split_n_length l0_stride lora_k_stride)) := by
    rw [hsk, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("b_ptr":RegName) ≠ "n" by decide)]
    exact hbp
  have hcp_sk : sk.regs .ptr [] "c_ptr" = some (Tile.scalar (cPtrVal s0 out_ptr split_n_length cm_stride)) := by
    rw [hsk, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("c_ptr":RegName) ≠ "n" by decide)]
    exact hcp
  unfold loopBody at hstep
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (currentN_eval BLOCK_N c sk hn_sk hon_sk))] at hstep
  set s1 := sk.setReg "current_n" .nat [BLOCK_N] (Tile.vec (fun j : Fin BLOCK_N => c*BLOCK_N + j.val)) with hs1
  have hcn1 : s1.regs .nat [BLOCK_N] "current_n" = some (Tile.vec (fun j : Fin BLOCK_N => c*BLOCK_N + j.val)) := by
    rw [hs1, BlockState.setReg_same]
  have hbp1 : s1.regs .ptr [] "b_ptr" = some (Tile.scalar (bPtrVal s0 lora_ptr lora_indices split_n_length l0_stride lora_k_stride)) := by
    rw [hs1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("b_ptr":RegName) ≠ "current_n" by decide)]
    exact hbp_sk
  have hok1 : s1.regs .nat [BLOCK_K] "offset_k" = some (Tile.vec (fun k : Fin BLOCK_K => k.val)) := by
    rw [hs1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_k":RegName) ≠ "current_n" by decide)]
    exact hok_sk
  have hs1pids : s1.pids = s0.pids := by rw [hs1, BlockState.setReg_pids, hskpids]
  have hs1rm : s1.readMem = st.readMem := by
    funext rg ofs; rw [hs1]; simp only [BlockState.setReg_readMem]
    exact congrFun (congrFun hskrm rg) ofs
  have hs1rml : s1.readMem lora_ptr = s0.readMem lora_ptr := by rw [hs1rm]; exact hrml
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (tiledB_eval lora_ptr lora_indices K split_n_length l0_stride lora_k_stride lora_n_stride
          BLOCK_N BLOCK_K c s0 s1 hbp1 hcn1 hok1 hs1pids hs1rml))] at hstep
  set s2 := s1.setReg "tiled_b" .real [BLOCK_N, BLOCK_K]
    (tiledB s0 lora_ptr lora_indices K split_n_length l0_stride lora_k_stride lora_n_stride BLOCK_N BLOCK_K c) with hs2
  have hta2 : s2.regs .real [BLOCK_K] "tiled_a" = some (tiledA s0 input_ptr K xm_stride xk_stride BLOCK_K) := by
    rw [hs2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("tiled_a":RegName) ≠ "tiled_b" by decide), hs1,
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("tiled_a":RegName) ≠ "current_n" by decide)]
    exact hta_sk
  have htb2 : s2.regs .real [BLOCK_N, BLOCK_K] "tiled_b" = some (tiledB s0 lora_ptr lora_indices K split_n_length l0_stride lora_k_stride lora_n_stride BLOCK_N BLOCK_K c) := by
    rw [hs2, BlockState.setReg_same]
  erw [stepStmts.cons_some (stepStmt_assign_eq_some
        (acc_eval BLOCK_N BLOCK_K (tiledA s0 input_ptr K xm_stride xk_stride BLOCK_K)
          (tiledB s0 lora_ptr lora_indices K split_n_length l0_stride lora_k_stride lora_n_stride BLOCK_N BLOCK_K c)
          s2 hta2 htb2))] at hstep
  set accT : Tile .real [BLOCK_N] :=
    Tile.reduceSum (shape := [BLOCK_N, BLOCK_K]) ⟨1, by simp⟩ Bool.false
      (Tile.bop NumericDType.real.mul (Broadcast.leadL (Broadcast.consSame Broadcast.nil))
        (tiledA s0 input_ptr K xm_stride xk_stride BLOCK_K)
        (tiledB s0 lora_ptr lora_indices K split_n_length l0_stride lora_k_stride lora_n_stride BLOCK_N BLOCK_K c)) with haccT
  set s3 := s2.setReg "accumulator" .real [BLOCK_N] accT with hs3
  have hcp3 : s3.regs .ptr [] "c_ptr" = some (Tile.scalar (cPtrVal s0 out_ptr split_n_length cm_stride)) := by
    rw [hs3, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("c_ptr":RegName) ≠ "accumulator" by decide), hs2,
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("c_ptr":RegName) ≠ "tiled_b" by decide), hs1,
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("c_ptr":RegName) ≠ "current_n" by decide)]
    exact hcp_sk
  have hcn3 : s3.regs .nat [BLOCK_N] "current_n" = some (Tile.vec (fun j : Fin BLOCK_N => c*BLOCK_N + j.val)) := by
    rw [hs3, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("current_n":RegName) ≠ "accumulator" by decide), hs2,
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("current_n":RegName) ≠ "tiled_b" by decide)]
    exact hcn1
  have hacc3 : s3.regs .real [BLOCK_N] "accumulator" = some ⟨fun idx : TileIndex [BLOCK_N] => accT.data (idx.1, PUnit.unit)⟩ := by
    rw [hs3, BlockState.setReg_same]
    apply congrArg some; apply Tile.ext; intro idx; rfl
  have haccSome : s3.regs .real [BLOCK_N] "accumulator"
      = some ⟨fun idx : TileIndex [BLOCK_N] =>
          some (WithBot.unbotD 0 (accT.data (idx.1, PUnit.unit)))⟩ := by
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
  erw [stepStmts.cons_some (store_step out_ptr split_n_length cn_stride
        (s0.pids 1 * cm_stride + s0.pids 0 * split_n_length) BLOCK_N
        (fun j : Fin BLOCK_N => c*BLOCK_N + j.val)
        (fun j : Fin BLOCK_N => WithBot.unbotD 0 (accT.data (j, PUnit.unit))) s3
        (by rw [hcp3]; rfl) hcn3 haccSome)] at hstep
  rw [stepStmts.nil] at hstep
  obtain rfl := Option.some.inj hstep
  refine Eq.trans (foldl_writeMem_bool_preserve_cell _ _ _ r oo _ _ ?_) ?_
  · intro b _ hmb hbad
    rcases hcond with hne | hno
    · exact hne hbad.1
    · have hbN2 : c*BLOCK_N + b.1.val < split_n_length := by simpa using hmb
      exact hno ⟨c*BLOCK_N + b.1.val, hbN2⟩ hbad.2
  · rw [hs3, hs2, hs1, hsk]
    rfl

/-! ### The `TraceSafeR` walk -/

/-- `evalOpR` = `evalOp` on the cast-free ops of the surface (register refs,
nat/bool/ptr index arithmetic — `R` never enters). One tiny lemma per op
shape, per the wave-5 recipe. -/
private theorem bgmvR_eq_pid (R : RoundingModel) (i : Nat) (s : BlockState) :
    evalOpR R (Op.programId i) s = evalOp (Op.programId i) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

private theorem bgmvR_eq_arange (R : RoundingModel) (n : Nat) (s : BlockState) :
    evalOpR R (Op.arange n) s = evalOp (Op.arange n) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

private theorem bgmvR_eq_currentN (R : RoundingModel) (BLOCK_N : Nat) (s : BlockState) :
    evalOpR R (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "n")
        (Op.ref .nat [BLOCK_N] "offset_n")) s
      = evalOp (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "n")
        (Op.ref .nat [BLOCK_N] "offset_n")) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

private theorem bgmvR_eq_aAddr (R : RoundingModel) (xm_stride xk_stride BLOCK_K : Nat)
    (s : BlockState) :
    evalOpR R (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_batch") (Op.constNat xm_stride))
        (Op.mul .nat Broadcast.scalarR (Op.ref .nat [BLOCK_K] "offset_k") (Op.constNat xk_stride))) s
      = evalOp (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_batch") (Op.constNat xm_stride))
        (Op.mul .nat Broadcast.scalarR (Op.ref .nat [BLOCK_K] "offset_k") (Op.constNat xk_stride))) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

private theorem bgmvR_eq_aMask (R : RoundingModel) (K BLOCK_K : Nat) (s : BlockState) :
    evalOpR R (Op.lt .nat Broadcast.scalarR (Op.ref .nat [BLOCK_K] "offset_k") (Op.constNat K)) s
      = evalOp (Op.lt .nat Broadcast.scalarR (Op.ref .nat [BLOCK_K] "offset_k") (Op.constNat K)) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

private theorem bgmvR_eq_bAddr (R : RoundingModel)
    (lora_k_stride lora_n_stride BLOCK_N BLOCK_K : Nat) (s : BlockState) :
    evalOpR R (Op.ptrAdd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "b_ptr")
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "current_n"))
            (Op.constNat lora_k_stride)))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_K] "offset_k"))
          (Op.constNat lora_n_stride))) s
      = evalOp (Op.ptrAdd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "b_ptr")
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "current_n"))
            (Op.constNat lora_k_stride)))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_K] "offset_k"))
          (Op.constNat lora_n_stride))) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

private theorem bgmvR_eq_bMask (R : RoundingModel)
    (K split_n_length BLOCK_N BLOCK_K : Nat) (s : BlockState) :
    evalOpR R (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.lt .nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "current_n"))
          (Op.constNat split_n_length))
        (Op.lt .nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_K] "offset_k"))
          (Op.constNat K))) s
      = evalOp (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.lt .nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "current_n"))
          (Op.constNat split_n_length))
        (Op.lt .nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_K] "offset_k"))
          (Op.constNat K))) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

private theorem bgmvR_eq_cAddr (R : RoundingModel) (cn_stride BLOCK_N : Nat) (s : BlockState) :
    evalOpR R (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "c_ptr")
        (Op.mul .nat Broadcast.scalarR (Op.ref .nat [BLOCK_N] "current_n")
          (Op.constNat cn_stride))) s
      = evalOp (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "c_ptr")
        (Op.mul .nat Broadcast.scalarR (Op.ref .nat [BLOCK_N] "current_n")
          (Op.constNat cn_stride))) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

private theorem bgmvR_eq_cMask (R : RoundingModel) (split_n_length BLOCK_N : Nat)
    (s : BlockState) :
    evalOpR R (Op.lt .nat Broadcast.scalarR (Op.ref .nat [BLOCK_N] "current_n")
        (Op.constNat split_n_length)) s
      = evalOp (Op.lt .nat Broadcast.scalarR (Op.ref .nat [BLOCK_N] "current_n")
        (Op.constNat split_n_length)) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

/-- The per-lane value of the input load's address op
(`cur_batch·xm + offset_k·xk`). -/
private theorem aAddr_eval (xm_stride xk_stride BLOCK_K : Nat) (s0 s : BlockState)
    (hck : s.regs .nat [] "cur_batch" = some (Tile.scalar (s0.pids 1)))
    (hok : s.regs .nat [BLOCK_K] "offset_k" = some (Tile.vec (fun k : Fin BLOCK_K => k.val))) :
    evalOp (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_batch") (Op.constNat xm_stride))
        (Op.mul .nat Broadcast.scalarR (Op.ref .nat [BLOCK_K] "offset_k") (Op.constNat xk_stride))) s
      = some (Tile.vec (fun k : Fin BLOCK_K => s0.pids 1 * xm_stride + k.val * xk_stride)) := by
  simp only [evalOp, evalOp.eq_def, evalOp_ref, hck, hok, Option.bind,
    Tile.bop, Tile.scalar, Tile.vec, NumericDType.add, NumericDType.mul]
  apply congrArg some
  apply Tile.ext
  intro k
  simp only [Tile.bop, Tile.scalar, Tile.vec, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.add, NumericDType.mul]

/-- The input load's mask (`offset_k < K`). -/
private theorem aMask_eval (K BLOCK_K : Nat) (s : BlockState)
    (hok : s.regs .nat [BLOCK_K] "offset_k" = some (Tile.vec (fun k : Fin BLOCK_K => k.val))) :
    evalOp (Op.lt .nat Broadcast.scalarR (Op.ref .nat [BLOCK_K] "offset_k") (Op.constNat K)) s
      = some (Tile.vec (fun k : Fin BLOCK_K => decide (k.val < K))) := by
  simp only [evalOp, evalOp.eq_def, evalOp_ref, hok, Option.bind,
    Tile.cop, Tile.scalar, Tile.vec, ComparableDType.lt]
  apply congrArg some
  apply Tile.ext
  intro k
  simp only [Tile.cop, Tile.scalar, Tile.vec, Broadcast.leftIndex, Broadcast.rightIndex,
    ComparableDType.lt]
  rfl

/-- The per-lane value of the `loraB` load's address op at block `c`:
`(lora_ptr, bPtr + (c·BLOCK_N + r)·lk + k·ln)`. -/
private theorem bAddr_eval (lora_ptr : RegionName) (lora_indices : Region .nat)
    (split_n_length l0_stride lora_k_stride lora_n_stride BLOCK_N BLOCK_K c : Nat)
    (s0 s : BlockState)
    (hbp : s.regs .ptr [] "b_ptr" = some (Tile.scalar
      (bPtrVal s0 lora_ptr lora_indices split_n_length l0_stride lora_k_stride)))
    (hcn : s.regs .nat [BLOCK_N] "current_n" = some (Tile.vec (fun j : Fin BLOCK_N => c*BLOCK_N + j.val)))
    (hok : s.regs .nat [BLOCK_K] "offset_k" = some (Tile.vec (fun k : Fin BLOCK_K => k.val))) :
    evalOp (Op.ptrAdd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "b_ptr")
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "current_n"))
            (Op.constNat lora_k_stride)))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_K] "offset_k"))
          (Op.constNat lora_n_stride))) s
      = some ⟨fun idx : TileIndex [BLOCK_N, BLOCK_K] =>
          (lora_ptr, l0_stride * loraIdx s0 lora_indices
            + s0.pids 0 * split_n_length * lora_k_stride
            + (c*BLOCK_N + idx.1.val) * lora_k_stride
            + idx.2.1.val * lora_n_stride)⟩ := by
  simp only [evalOp, evalOp.eq_def, evalOp_ref, hbp, hcn, hok, Option.bind, Option.map,
    Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim, Tile.scalar, Tile.vec,
    Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul]
  apply congrArg some
  apply Tile.ext
  intro idx
  simp only [Tile.ptrAdd, Tile.bop, Tile.expandDim, Tile.scalar, Tile.vec,
    TileShape.dropInsertedIndex, bPtrVal, Region.cast]

/-- The `loraB` load's mask (`b_ptr_mask`) at block `c`. -/
private theorem bMask_eval (K split_n_length BLOCK_N BLOCK_K c : Nat) (s : BlockState)
    (hcn : s.regs .nat [BLOCK_N] "current_n" = some (Tile.vec (fun j : Fin BLOCK_N => c*BLOCK_N + j.val)))
    (hok : s.regs .nat [BLOCK_K] "offset_k" = some (Tile.vec (fun k : Fin BLOCK_K => k.val))) :
    evalOp (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.lt .nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "current_n"))
          (Op.constNat split_n_length))
        (Op.lt .nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_K] "offset_k"))
          (Op.constNat K))) s
      = some ⟨fun idx : TileIndex [BLOCK_N, BLOCK_K] =>
          decide (c*BLOCK_N + idx.1.val < split_n_length) && decide (idx.2.1.val < K)⟩ := by
  simp only [evalOp, evalOp.eq_def, evalOp_ref, hcn, hok, Option.bind, Option.map,
    Tile.bop, Tile.cop, Tile.expandDim, Tile.scalar, Tile.vec,
    Broadcast.leftIndex, Broadcast.rightIndex, ComparableDType.lt]
  apply congrArg some
  apply Tile.ext
  intro idx
  simp only [Tile.bop, Tile.cop, Tile.expandDim, Tile.scalar, Tile.vec,
    TileShape.dropInsertedIndex]
  rfl

/-- The per-lane value of the store's address op at block `c`:
`(out_ptr, cPtr + (c·BLOCK_N + j)·cn)`. -/
private theorem cAddr_eval (out_ptr : RegionName)
    (split_n_length cm_stride cn_stride BLOCK_N c : Nat) (s0 s : BlockState)
    (hcp : s.regs .ptr [] "c_ptr" = some (Tile.scalar
      (cPtrVal s0 out_ptr split_n_length cm_stride)))
    (hcn : s.regs .nat [BLOCK_N] "current_n" = some (Tile.vec (fun j : Fin BLOCK_N => c*BLOCK_N + j.val))) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "c_ptr")
        (Op.mul .nat Broadcast.scalarR (Op.ref .nat [BLOCK_N] "current_n")
          (Op.constNat cn_stride))) s
      = some ⟨fun idx : TileIndex [BLOCK_N] =>
          (out_ptr, s0.pids 1 * cm_stride + s0.pids 0 * split_n_length
            + (c*BLOCK_N + idx.1.val) * cn_stride)⟩ := by
  simp only [evalOp, evalOp.eq_def, evalOp_ref, hcp, hcn, Option.bind, Option.map,
    Tile.bop, Tile.ptrAdd, Tile.scalar, Tile.vec,
    Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.mul]
  apply congrArg some
  apply Tile.ext
  intro idx
  simp only [Tile.ptrAdd, Tile.bop, Tile.scalar, Tile.vec, cPtrVal, Region.cast]

/-- The store's mask (`c_mask`) at block `c`. -/
private theorem cMask_eval (split_n_length BLOCK_N c : Nat) (s : BlockState)
    (hcn : s.regs .nat [BLOCK_N] "current_n" = some (Tile.vec (fun j : Fin BLOCK_N => c*BLOCK_N + j.val))) :
    evalOp (Op.lt .nat Broadcast.scalarR (Op.ref .nat [BLOCK_N] "current_n")
        (Op.constNat split_n_length)) s
      = some (Tile.vec (fun j : Fin BLOCK_N => decide (c*BLOCK_N + j.val < split_n_length))) := by
  simp only [evalOp, evalOp.eq_def, evalOp_ref, hcn, Option.bind,
    Tile.cop, Tile.scalar, Tile.vec, ComparableDType.lt]
  apply congrArg some
  apply Tile.ext
  intro j
  simp only [Tile.cop, Tile.scalar, Tile.vec, Broadcast.leftIndex, Broadcast.rightIndex,
    ComparableDType.lt]
  rfl

/-- Safety of the scalar metadata load (`tl.load(lora_indices + cur_batch)`):
the only active address is the pinned `cur_batch` register value, in bounds
by the skin's slot window (private copy of the sgmv wave-5 helper). -/
private theorem bgmv_metaLoad_safeR (R : RoundingModel) (bounds : RegionBounds)
    (buf : Region .nat) (name : RegName) (t : BlockState) (a : Nat)
    (hcur : t.regs .nat [] "cur_batch" = some (Tile.scalar a))
    (hb : a < bounds buf.cast) :
    Stmt.TraceSafeR R bounds
      (Stmt.assign .nat [] name (Op.load .nat (.region buf (Op.ref .nat [] "cur_batch")) .none)) t := by
  simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
    MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
  refine ⟨trivial, trivial, ?_⟩
  intro offs hoffs i _
  rw [evalOpR_ref, hcur] at hoffs
  obtain rfl := Option.some.inj hoffs
  exact hb

set_option maxHeartbeats 8000000 in
/-- Per-iteration `TraceSafeListR` for the `for n` body: the index assign
and the reduce are register-only; the masked `loraB` load's and the masked
store's **active** lanes are the skin's `mask2` / `writeMask` windows at
step `i / BLOCK_N`, in bounds by the corresponding window bounds
(instantiated at raw counter `i`). -/
private theorem bgmv_loopBodySafeR (R : RoundingModel) (bounds : RegionBounds)
    (input_ptr lora_ptr out_ptr : RegionName) (lora_indices : Region .nat)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride BLOCK_N BLOCK_K : Nat)
    (s0 st : BlockState) (i : Nat) (hiB : BLOCK_N ∣ i)
    (hon : st.regs .nat [BLOCK_N] "offset_n" = some (Tile.vec (fun j : Fin BLOCK_N => j.val)))
    (hok : st.regs .nat [BLOCK_K] "offset_k" = some (Tile.vec (fun k : Fin BLOCK_K => k.val)))
    (hbp : st.regs .ptr [] "b_ptr" = some (Tile.scalar
      (bPtrVal s0 lora_ptr lora_indices split_n_length l0_stride lora_k_stride)))
    (hcp : st.regs .ptr [] "c_ptr" = some (Tile.scalar
      (cPtrVal s0 out_ptr split_n_length cm_stride)))
    (hbB : ∀ (j : Fin BLOCK_N) (k : Fin BLOCK_K), i + j.val < split_n_length → k.val < K →
      l0_stride * loraIdx s0 lora_indices + s0.pids 0 * split_n_length * lora_k_stride
        + (i + j.val) * lora_k_stride + k.val * lora_n_stride < bounds lora_ptr)
    (hbC : ∀ j : Fin BLOCK_N, i + j.val < split_n_length →
      s0.pids 1 * cm_stride + s0.pids 0 * split_n_length
        + (i + j.val) * cn_stride < bounds out_ptr) :
    Stmt.TraceSafeListR R bounds
      (loopBody K split_n_length lora_k_stride lora_n_stride cn_stride BLOCK_N BLOCK_K)
      (st.setReg "n" .nat [] (Tile.scalar i)) := by
  obtain ⟨c, rfl⟩ := hiB
  rw [show BLOCK_N*c = c*BLOCK_N from Nat.mul_comm _ _] at hbB hbC ⊢
  unfold loopBody
  have hn_sk : (st.setReg "n" .nat [] (Tile.scalar (c*BLOCK_N))).regs .nat [] "n"
      = some (Tile.scalar (c*BLOCK_N)) := BlockState.setReg_same _ _ _ _ _
  have hon_sk : (st.setReg "n" .nat [] (Tile.scalar (c*BLOCK_N))).regs .nat [BLOCK_N] "offset_n"
      = some (Tile.vec (fun j : Fin BLOCK_N => j.val)) := by
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_n":RegName) ≠ "n" by decide)]
    exact hon
  -- current_n (register-only)
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun t1 ht1 => ?_)
  rw [stepStmtR_assign_eq_some ((bgmvR_eq_currentN R BLOCK_N _).trans
        (currentN_eval BLOCK_N c _ hn_sk hon_sk))] at ht1
  obtain rfl := Option.some.inj ht1
  set q1 := (st.setReg "n" .nat [] (Tile.scalar (c*BLOCK_N))).setReg "current_n" .nat [BLOCK_N]
    (Tile.vec (fun j : Fin BLOCK_N => c*BLOCK_N + j.val)) with hq1
  have hcn1 : q1.regs .nat [BLOCK_N] "current_n"
      = some (Tile.vec (fun j : Fin BLOCK_N => c*BLOCK_N + j.val)) := by
    rw [hq1]; exact BlockState.setReg_same _ _ _ _ _
  have hbp1 : q1.regs .ptr [] "b_ptr" = some (Tile.scalar
      (bPtrVal s0 lora_ptr lora_indices split_n_length l0_stride lora_k_stride)) := by
    rw [hq1,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("b_ptr":RegName) ≠ "current_n" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("b_ptr":RegName) ≠ "n" by decide)]
    exact hbp
  have hok1 : q1.regs .nat [BLOCK_K] "offset_k"
      = some (Tile.vec (fun k : Fin BLOCK_K => k.val)) := by
    rw [hq1,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_k":RegName) ≠ "current_n" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_k":RegName) ≠ "n" by decide)]
    exact hok
  -- the masked loraB load
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun t2 ht2 => ?_)
  · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR,
      and_true, true_and, and_self]
    intro ptrs hptrs idx hactive
    rw [bgmvR_eq_bAddr,
      bAddr_eval lora_ptr lora_indices split_n_length l0_stride lora_k_stride lora_n_stride
        BLOCK_N BLOCK_K c s0 q1 hbp1 hcn1 hok1] at hptrs
    obtain rfl := Option.some.inj hptrs
    obtain ⟨masks, hm, hmi⟩ := hactive
    rw [bgmvR_eq_bMask,
      bMask_eval K split_n_length BLOCK_N BLOCK_K c q1 hcn1 hok1] at hm
    obtain rfl := Option.some.inj hm
    have hact : c*BLOCK_N + idx.1.val < split_n_length ∧ idx.2.1.val < K := by
      simpa using hmi
    simpa using hbB idx.1 idx.2.1 hact.1 hact.2
  · obtain ⟨v2, -, rfl⟩ := stepStmtR_assign_inv ht2
    set q2 := q1.setReg "tiled_b" .real [BLOCK_N, BLOCK_K] v2 with hq2
    -- accumulator (register-only)
    refine Stmt.TraceSafeListR.cons_intro
      (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun t3 ht3 => ?_)
    obtain ⟨v3, -, rfl⟩ := stepStmtR_assign_inv ht3
    set q3 := q2.setReg "accumulator" .real [BLOCK_N] v3 with hq3
    have hcp3 : q3.regs .ptr [] "c_ptr" = some (Tile.scalar
        (cPtrVal s0 out_ptr split_n_length cm_stride)) := by
      rw [hq3,
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("c_ptr":RegName) ≠ "accumulator" by decide),
        hq2,
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("c_ptr":RegName) ≠ "tiled_b" by decide),
        hq1,
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("c_ptr":RegName) ≠ "current_n" by decide),
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("c_ptr":RegName) ≠ "n" by decide)]
      exact hcp
    have hcn3 : q3.regs .nat [BLOCK_N] "current_n"
        = some (Tile.vec (fun j : Fin BLOCK_N => c*BLOCK_N + j.val)) := by
      rw [hq3,
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("current_n":RegName) ≠ "accumulator" by decide),
        hq2,
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("current_n":RegName) ≠ "tiled_b" by decide)]
      exact hcn1
    -- the masked store
    refine Stmt.TraceSafeListR.cons_intro ?_ (fun _ _ => Stmt.TraceSafeListR.nil_intro)
    simp only [Stmt.TraceSafeR, MemAccess.SafeAtR, MaskOpt.SafeAtR, Op.SafeAtR.eq_def,
      MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR,
      and_true, true_and, and_self]
    intro ptrs hptrs idx hactive
    rw [bgmvR_eq_cAddr,
      cAddr_eval out_ptr split_n_length cm_stride cn_stride BLOCK_N c s0 q3 hcp3 hcn3] at hptrs
    obtain rfl := Option.some.inj hptrs
    obtain ⟨masks, hm, hmi⟩ := hactive
    rw [bgmvR_eq_cMask, cMask_eval split_n_length BLOCK_N c q3 hcn3] at hm
    obtain rfl := Option.some.inj hm
    have hlt : c*BLOCK_N + idx.1.val < split_n_length := by simpa [Tile.vec] using hmi
    simpa using hbC idx.1 hlt

set_option maxHeartbeats 4000000 in
/-- Walk-side clone of `prefix_inv` **without** the (unused) `undef`
hypothesis: the skin's `TraceSafeR` obligation carries no `undef` pin, and
`prefix_inv`'s proof never reads it (the masked input load carries
`other=0`). Statement and proof otherwise verbatim. -/
private theorem prefix_inv_walk
    (input_ptr lora_ptr out_ptr : RegionName) (lora_indices : Region .nat)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride
      lora_n_stride cm_stride cn_stride BLOCK_N BLOCK_K : Nat)
    (s : BlockState) :
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

set_option maxHeartbeats 8000000 in
/-- **The `TraceSafeR` walk for the whole loop surface** — the 8-statement
prologue by hand (the `.nat` slot load and the masked static input load are
the two memory touches; everything else is register-only), then the `for n`
loop driven by `Stmt.forRangeTraceSafeR_inv` over the (launch-state-robust)
`gemvWbInv`. The bound groups are the skin's slot / `read1` / `read2` /
`write` windows, with the raw loop counter converted to the step index
`i / BLOCK_N < ⌈split_n_length/BLOCK_N⌉` the windows are phrased over. -/
private theorem bgmv_traceSafeR (R : RoundingModel) (bounds : RegionBounds)
    (input_ptr lora_ptr out_ptr : RegionName) (lora_indices : Region .nat)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride BLOCK_N BLOCK_K : Nat)
    (hBN : 0 < BLOCK_N) (hKB : K ≤ BLOCK_K) (hol : out_ptr ≠ lora_ptr) (hcn : 0 < cn_stride)
    (s : BlockState)
    (hbm : s.pids 1 < bounds lora_indices.cast)
    (hbx : ∀ k : Fin BLOCK_K, k.val < K →
      s.pids 1 * xm_stride + k.val * xk_stride < bounds input_ptr)
    (hbB : ∀ (t : Fin (bgmvNumSteps split_n_length BLOCK_N)) (j : Fin BLOCK_N) (k : Fin BLOCK_K),
      t.val * BLOCK_N + j.val < split_n_length → k.val < K →
      l0_stride * loraIdx s lora_indices + s.pids 0 * split_n_length * lora_k_stride
        + (t.val * BLOCK_N + j.val) * lora_k_stride + k.val * lora_n_stride < bounds lora_ptr)
    (hbC : ∀ (t : Fin (bgmvNumSteps split_n_length BLOCK_N)) (j : Fin BLOCK_N),
      t.val * BLOCK_N + j.val < split_n_length →
      s.pids 1 * cm_stride + s.pids 0 * split_n_length
        + (t.val * BLOCK_N + j.val) * cn_stride < bounds out_ptr) :
    ((bgmv_loop_surface input_ptr lora_ptr out_ptr lora_indices K split_n_length
        xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
        BLOCK_N BLOCK_K).toAlgKernel).TraceSafeR R bounds s := by
  -- raw-counter instantiators for the loop windows
  have hstepT : ∀ i, BLOCK_N ∣ i → ∀ j : Fin BLOCK_N, i + j.val < split_n_length →
      ∃ t : Fin (bgmvNumSteps split_n_length BLOCK_N), t.val * BLOCK_N + j.val = i + j.val := by
    intro i hiB j hij
    have hiN : i < split_n_length := Nat.lt_of_le_of_lt (Nat.le_add_right _ _) hij
    refine ⟨⟨i / BLOCK_N, bgmvStep_lt_numSteps split_n_length BLOCK_N i hBN hiN⟩, ?_⟩
    simp [Nat.div_mul_cancel hiB]
  have hbB' : ∀ i, BLOCK_N ∣ i → ∀ (j : Fin BLOCK_N) (k : Fin BLOCK_K),
      i + j.val < split_n_length → k.val < K →
      l0_stride * loraIdx s lora_indices + s.pids 0 * split_n_length * lora_k_stride
        + (i + j.val) * lora_k_stride + k.val * lora_n_stride < bounds lora_ptr := by
    intro i hiB j k hij hk
    obtain ⟨t, ht⟩ := hstepT i hiB j hij
    have h := hbB t j k (by rw [ht]; exact hij) hk
    rwa [ht] at h
  have hbC' : ∀ i, BLOCK_N ∣ i → ∀ j : Fin BLOCK_N, i + j.val < split_n_length →
      s.pids 1 * cm_stride + s.pids 0 * split_n_length
        + (i + j.val) * cn_stride < bounds out_ptr := by
    intro i hiB j hij
    obtain ⟨t, ht⟩ := hstepT i hiB j hij
    have h := hbC t j (by rw [ht]; exact hij)
    rwa [ht] at h
  have hinj : Function.Injective (fun mm : Fin split_n_length =>
      outOffG s split_n_length cm_stride cn_stride mm.val) := by
    have heq : (fun mm : Fin split_n_length => outOffG s split_n_length cm_stride cn_stride mm.val)
        = (fun mm : Fin split_n_length =>
            (s.pids 1 * cm_stride + s.pids 0 * split_n_length) + mm.val * cn_stride) := by
      funext mm; simp only [outOffG]
    rw [heq]; exact affine1D_inj _ cn_stride hcn
  unfold Kernel.TraceSafeR
  rw [body_split input_ptr lora_ptr out_ptr lora_indices K split_n_length xm_stride xk_stride
      l0_stride lora_k_stride lora_n_stride cm_stride cn_stride BLOCK_N BLOCK_K,
    prefix_split input_ptr lora_ptr out_ptr lora_indices K split_n_length xm_stride xk_stride
      l0_stride lora_k_stride lora_n_stride cm_stride cn_stride BLOCK_N BLOCK_K]
  refine Stmt.TraceSafeListR.append_intro _ _ ?_ ?_
  · -- the 8 prologue statements
    unfold prefixStmts
    -- pid_sn
    refine Stmt.TraceSafeListR.cons_intro
      (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun t1 ht1 => ?_)
    obtain ⟨v1, -, rfl⟩ := stepStmtR_assign_inv ht1
    -- cur_batch
    refine Stmt.TraceSafeListR.cons_intro
      (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun t2 ht2 => ?_)
    rw [stepStmtR_assign_eq_some ((bgmvR_eq_pid R 1 _).trans
      (show evalOp (Op.programId 1) (s.setReg "pid_sn" .nat [] v1)
        = some (Tile.scalar (s.pids 1)) by simp))] at ht2
    obtain rfl := Option.some.inj ht2
    set q2 := (s.setReg "pid_sn" .nat [] v1).setReg "cur_batch" .nat [] (Tile.scalar (s.pids 1)) with hq2
    have hck2 : q2.regs .nat [] "cur_batch" = some (Tile.scalar (s.pids 1)) := by
      rw [hq2]; exact BlockState.setReg_same _ _ _ _ _
    -- lora_index load
    refine Stmt.TraceSafeListR.cons_intro
      (bgmv_metaLoad_safeR R bounds lora_indices "lora_index" q2 (s.pids 1) hck2 hbm)
      (fun t3 ht3 => ?_)
    obtain ⟨v3, -, rfl⟩ := stepStmtR_assign_inv ht3
    -- offset_k
    refine Stmt.TraceSafeListR.cons_intro
      (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun t4 ht4 => ?_)
    rw [stepStmtR_assign_eq_some ((bgmvR_eq_arange R BLOCK_K _).trans
      (show evalOp (Op.arange BLOCK_K) (q2.setReg "lora_index" .nat [] v3)
        = some (Tile.vec (fun k : Fin BLOCK_K => k.val)) by simp [Tile.vec]))] at ht4
    obtain rfl := Option.some.inj ht4
    set q4 := (q2.setReg "lora_index" .nat [] v3).setReg "offset_k" .nat [BLOCK_K]
      (Tile.vec (fun k : Fin BLOCK_K => k.val)) with hq4
    -- offset_n
    refine Stmt.TraceSafeListR.cons_intro
      (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun t5 ht5 => ?_)
    obtain ⟨v5, -, rfl⟩ := stepStmtR_assign_inv ht5
    set q5 := q4.setReg "offset_n" .nat [BLOCK_N] v5 with hq5
    have hck5 : q5.regs .nat [] "cur_batch" = some (Tile.scalar (s.pids 1)) := by
      rw [hq5,
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("cur_batch":RegName) ≠ "offset_n" by decide),
        hq4,
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("cur_batch":RegName) ≠ "offset_k" by decide),
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("cur_batch":RegName) ≠ "lora_index" by decide)]
      exact hck2
    have hok5 : q5.regs .nat [BLOCK_K] "offset_k"
        = some (Tile.vec (fun k : Fin BLOCK_K => k.val)) := by
      rw [hq5,
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_k":RegName) ≠ "offset_n" by decide),
        hq4]
      exact BlockState.setReg_same _ _ _ _ _
    -- tiled_a load
    refine Stmt.TraceSafeListR.cons_intro ?_ (fun t6 ht6 => ?_)
    · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
        MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR,
        and_true, true_and, and_self]
      intro offsets hoffsets idx hactive
      rw [bgmvR_eq_aAddr, aAddr_eval xm_stride xk_stride BLOCK_K s q5 hck5 hok5] at hoffsets
      obtain rfl := Option.some.inj hoffsets
      obtain ⟨masks, hm, hmi⟩ := hactive
      rw [bgmvR_eq_aMask, aMask_eval K BLOCK_K q5 hok5] at hm
      obtain rfl := Option.some.inj hm
      have hk : idx.1.val < K := by simpa [Tile.vec] using hmi
      simpa [Region.cast_id, Tile.vec] using hbx idx.1 hk
    · obtain ⟨v6, -, rfl⟩ := stepStmtR_assign_inv ht6
      -- b_ptr, c_ptr: register-only
      refine Stmt.TraceSafeListR.cons_intro
        (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun t7 _ => ?_)
      exact Stmt.TraceSafeListR.cons_intro
        (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
        (fun _ _ => Stmt.TraceSafeListR.nil_intro)
  · -- after the prologue: the for-n loop
    intro s1 hs1
    obtain ⟨sp, hsp, hP0⟩ := prefix_inv_walk input_ptr lora_ptr out_ptr lora_indices K
      split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride BLOCK_N BLOCK_K s
    rw [prefix_split input_ptr lora_ptr out_ptr lora_indices K split_n_length xm_stride
      xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride BLOCK_N BLOCK_K] at hsp
    rw [bgmvPrefix_castFree R input_ptr lora_ptr out_ptr lora_indices K split_n_length
      xm_stride xk_stride l0_stride lora_k_stride cm_stride BLOCK_N BLOCK_K s, hsp] at hs1
    obtain rfl := Option.some.inj hs1
    refine Stmt.TraceSafeListR.cons_intro ?_ (fun _ _ => Stmt.TraceSafeListR.nil_intro)
    simp only [Stmt.TraceSafeR]
    refine Stmt.forRangeTraceSafeR_inv R bounds "n" split_n_length BLOCK_N
      (loopBody K split_n_length lora_k_stride lora_n_stride cn_stride BLOCK_N BLOCK_K)
      (gemvWbInv input_ptr lora_ptr out_ptr lora_indices K split_n_length xm_stride
        xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride BLOCK_N BLOCK_K s s)
      ?_ 0 sp hP0
    intro i stt hi hP
    obtain ⟨hpidsP, hiBP, hokP, honP, htaP, hbpP, hcpP, hrmlP, hreadP⟩ := hP
    refine ⟨bgmv_loopBodySafeR R bounds input_ptr lora_ptr out_ptr lora_indices K split_n_length
        xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride BLOCK_N BLOCK_K
        s stt i hiBP honP hokP hbpP hcpP
        (fun j k hj hk => hbB' i hiBP j k hj hk) (fun j hj => hbC' i hiBP j hj), ?_⟩
    obtain ⟨st', hstep, hP'⟩ := gemvWbInv_step input_ptr lora_ptr out_ptr lora_indices K
      split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride
      cn_stride BLOCK_N BLOCK_K s s hBN hKB hol hinj i stt hi
      ⟨hpidsP, hiBP, hokP, honP, htaP, hbpP, hcpP, hrmlP, hreadP⟩
    exact ⟨st', by rw [bgmvLoopBody_castFree]; exact hstep, hP'⟩

/-! ### The rounded Hoare triple (`hrun`) -/

set_option maxHeartbeats 8000000 in
/-- Termination, per-lane values and the per-cell frame of the whole loop
surface under `execR R`, from an arbitrary launch state: the exact
`prefix_inv` / `gemvWbInv_step` / `forRange_inv` stack runs verbatim (the
body is cast-free, so `execR R` collapses onto the exact stepper), extended
with the per-segment memory frames. -/
private theorem bgmv_runR (R : RoundingModel)
    (input_ptr lora_ptr out_ptr : RegionName) (lora_indices : Region .nat)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride BLOCK_N BLOCK_K : Nat)
    (hBN : 0 < BLOCK_N) (hKB : K ≤ BLOCK_K) (hol : out_ptr ≠ lora_ptr) (hcn : 0 < cn_stride)
    (s₀ : BlockState) (hundef : ∀ rg o, s₀.undef rg o = 0) :
    ∃ sfin,
      execR R (bgmv_loop_surface input_ptr lora_ptr out_ptr lora_indices K
          split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
          cm_stride cn_stride BLOCK_N BLOCK_K).toAlgKernel s₀ = some sfin
      ∧ (∀ mm : Fin split_n_length,
          sfin.readMem out_ptr (outOffG s₀ split_n_length cm_stride cn_stride mm.val)
            = gemvSpec s₀ input_ptr lora_ptr lora_indices K split_n_length
                xm_stride xk_stride l0_stride lora_k_stride lora_n_stride mm.val)
      ∧ (∀ r oo, (r ≠ out_ptr ∨ ∀ mm : Fin split_n_length,
            oo ≠ outOffG s₀ split_n_length cm_stride cn_stride mm.val) →
          sfin.mem r oo = s₀.mem r oo) := by
  have hinj : Function.Injective (fun mm : Fin split_n_length =>
      outOffG s₀ split_n_length cm_stride cn_stride mm.val) := by
    have heq : (fun mm : Fin split_n_length => outOffG s₀ split_n_length cm_stride cn_stride mm.val)
        = (fun mm : Fin split_n_length =>
            (s₀.pids 1 * cm_stride + s₀.pids 0 * split_n_length) + mm.val * cn_stride) := by
      funext mm; simp only [outOffG]
    rw [heq]; exact affine1D_inj _ cn_stride hcn
  -- prologue (the exact stack's prefix, `hundef` discharged by the skin's pin)
  obtain ⟨sp, hsp, hP0⟩ := prefix_inv input_ptr lora_ptr out_ptr lora_indices K split_n_length
    xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
    BLOCK_N BLOCK_K s₀ hundef
  have hsp' : stepStmts (prefixStmts input_ptr lora_ptr out_ptr lora_indices K split_n_length
      xm_stride xk_stride l0_stride lora_k_stride cm_stride BLOCK_N BLOCK_K) s₀ = some sp := by
    rw [← prefix_split input_ptr lora_ptr out_ptr lora_indices K split_n_length xm_stride
      xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride BLOCK_N BLOCK_K]
    exact hsp
  have hspMem : sp.mem = s₀.mem :=
    stepStmts_assigns_mem
      (prefixStmts input_ptr lora_ptr out_ptr lora_indices K split_n_length
        xm_stride xk_stride l0_stride lora_k_stride cm_stride BLOCK_N BLOCK_K)
      (by
        intro stmt hst
        simp only [prefixStmts, List.mem_cons, List.not_mem_nil, or_false] at hst
        rcases hst with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> exact ⟨_, _, _, _, rfl⟩)
      hsp'
  -- the loop with the conditional cell frame carried alongside `gemvWbInv`
  obtain ⟨final, sLoop, hLoopStmt, hfinal, hPLoop⟩ :=
    forRange_inv (idx := "n") (start := 0) (stop := split_n_length) (step := BLOCK_N)
      (body := loopBody K split_n_length lora_k_stride lora_n_stride cn_stride BLOCK_N BLOCK_K)
      (P := fun i stt => gemvWbInv input_ptr lora_ptr out_ptr lora_indices K split_n_length
          xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
          BLOCK_N BLOCK_K s₀ s₀ i stt
        ∧ ∀ r oo, (r ≠ out_ptr ∨ ∀ mm : Fin split_n_length,
              oo ≠ outOffG s₀ split_n_length cm_stride cn_stride mm.val) →
            stt.mem r oo = s₀.mem r oo)
      (Nat.pos_iff_ne_zero.mp hBN)
      ⟨hP0, fun r oo _ => by rw [hspMem]⟩
      (fun i stt hlt hQ => by
        obtain ⟨st', hstep, hinv'⟩ := gemvWbInv_step input_ptr lora_ptr out_ptr lora_indices K
          split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride
          cn_stride BLOCK_N BLOCK_K s₀ s₀ hBN hKB hol hinj i stt hlt hQ.1
        refine ⟨st', hstep, hinv', ?_⟩
        intro r oo hcond
        obtain ⟨hpidsQ, hiBQ, hokQ, honQ, htaQ, hbpQ, hcpQ, hrmlQ, hreadQ⟩ := hQ.1
        rw [bgmv_loopBody_step_frame input_ptr lora_ptr out_ptr lora_indices K split_n_length
          xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
          BLOCK_N BLOCK_K s₀ stt st' i hiBQ hokQ honQ htaQ hbpQ hcpQ hpidsQ hrmlQ hstep r oo hcond]
        exact hQ.2 r oo hcond)
  obtain ⟨⟨_, _, _, _, _, _, _, _, hread⟩, hframe⟩ := hPLoop
  -- assemble the `execR` run through the cast-free collapses
  have hLoopR : stepStmtR R (Stmt.forRange "n" 0 split_n_length BLOCK_N
      (loopBody K split_n_length lora_k_stride lora_n_stride cn_stride BLOCK_N BLOCK_K)) sp
      = some sLoop := by
    rw [stepStmtR_forRange,
      stepForRangeAuxR_castFree R _
        (bgmvLoopBody_castFree R K split_n_length lora_k_stride lora_n_stride cn_stride
          BLOCK_N BLOCK_K) "n",
      ← stepForRangeAux.forRange_unfold]
    exact hLoopStmt
  refine ⟨sLoop, ?_, ?_, ?_⟩
  · show execR R (bgmv_loop_surface input_ptr lora_ptr out_ptr lora_indices K
        split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
        cm_stride cn_stride BLOCK_N BLOCK_K).toAlgKernel s₀ = some sLoop
    unfold execR
    rw [body_split input_ptr lora_ptr out_ptr lora_indices K split_n_length xm_stride xk_stride
        l0_stride lora_k_stride lora_n_stride cm_stride cn_stride BLOCK_N BLOCK_K,
      prefix_split input_ptr lora_ptr out_ptr lora_indices K split_n_length xm_stride xk_stride
        l0_stride lora_k_stride lora_n_stride cm_stride cn_stride BLOCK_N BLOCK_K,
      stepStmtsR_append R (prefixStmts input_ptr lora_ptr out_ptr lora_indices K split_n_length
        xm_stride xk_stride l0_stride lora_k_stride cm_stride BLOCK_N BLOCK_K) _ s₀,
      bgmvPrefix_castFree R input_ptr lora_ptr out_ptr lora_indices K split_n_length
        xm_stride xk_stride l0_stride lora_k_stride cm_stride BLOCK_N BLOCK_K s₀,
      hsp', Option.bind_some, stepStmtsR_cons_some hLoopR, stepStmtsR_nil]
  · intro mm
    rw [hread mm, if_pos (lt_of_lt_of_le mm.isLt hfinal)]
  · exact hframe

/-! ### The headline -/

set_option maxHeartbeats 4000000 in
/-- **The `⊨[R]` streaming metadata emit headline (wave-5 S3 genre).** For
every rounding model `R`, the verified `bgmv_loop_surface` implements, on
its `StreamMetaEmitMasked3DKernelIO₂` signature, the **ideal ℝ LoRA expand
GEMV** over the streamed tiles: every step-`t` write-active output lane `j`
(global lane `t·BLOCK_N + j < split_n_length`) holds
`Σ_{k<BLOCK_K} [k<K] xs[t,k] · ys[t, j·BLOCK_K+k]` — exact real arithmetic;
the slot vector `m` (the adapter index `lora_indices[cur_batch]`) enters the
spec only through the bank select of the `read2` window. The kernel has
**zero rounding events** (`.nat` slot load, `.real` masked loads/reduce,
`.real` masked in-loop stores; the verified `CAST_TYPE = false` path has no
`castFloat`), so the skin's boundary quantization degenerates: the readback
contract's `R.round .real` is the identity by the model's defining
`round_real`, and the `.real` in-loop stores are exact under `execR R` — the
∀-`R` face holds via the `RoundingModel` `.real` identity fields, not as a
`.triv` special case.

Layer map: the whole body is cast-free, so under `execR R` it collapses
verbatim onto the exact stepper and the proven
`prefix_inv` / `gemvWbInv` / `gemvWbInv_step` / `forRange_inv` stack above
is reused unchanged; the `⊨[R]` face adds the `TraceSafeR` walk, the
per-cell memory frame (`bgmv_loopBody_step_frame`, the `mem` twin of
`gemvWbInv_step`), and the stream-lane spec bridge
(`sum_fin_ite_range` re-blocking the `Σ_{k<K}` contraction, `Lane2D`
re-blocking the `[BLOCK_N, BLOCK_K]` tile).

All five hypotheses are truth-forced:

* `hBN : 0 < BLOCK_N` — the loop steps by `BLOCK_N`
  (`range(0, split_n_length, BLOCK_N)`); at `BLOCK_N = 0` the loop never
  advances and the step index `i / BLOCK_N` is meaningless. The same
  hypothesis the exact headline `gemv_full_output_summary` carries.
* `hKB : K ≤ BLOCK_K` — the kernel's own `BLOCK_K = next_power_of_2(K)`
  choice; it is what collapses the `tl.sum` over `BLOCK_K` keys to
  `Σ_{k<K}` (`reduceSum_active_eq_spec`). Same as the exact headline.
* `hsnl : 0 < split_n_length` — `split_n_length = tl.cdiv(N, SPLIT_N) ≥ 1`
  for every real launch (`N ≥ 1`). This face needs it **only for the safety
  walk**: at `split_n_length = 0` the stream is empty (`T = 0`) while the
  kernel still performs its pre-loop static `input` load, so the skin's
  step-indexed `read1` window bounds could not cover it. (The exact headline
  does not need it because `Realizes` carries no bounds obligation.)
* `hol : out_ptr ≠ lora_ptr` — the loop stores into `out` **between** its
  re-reads of the `loraB` tile; if they aliased, later blocks would re-read
  already-overwritten weights and the closed form would be false. Same as
  the exact headline.
* `hcn : 0 < cn_stride` — the per-lane write window
  `pid₁·cm + pid₀·snl + m·cn` is injective over global lanes only when the
  output column stride is nonzero (`affine1D_inj`); with `cn_stride = 0` all
  lanes collide and the per-lane readback would be last-writer-wins. Same as
  the exact headline.

The exact headline's remaining hypothesis `hundef` (`∀ rg o, s.undef rg o =
0`) is **not** carried here: `s₀.undef = fun _ _ => 0` is a precondition of
the skin's Hoare triple itself, so it comes free from `ImplementsR.intro`.

Modeling boundary (mirrors the surface's docstring): the statement covers
the verified `bgmv_loop_surface` configuration — `ADD_INPUTS = false`,
`CAST_TYPE = false`, masked input load (general `EVEN_K` path), and the
signed `lora_index == -1` sentinel elided at the trusted host boundary (the
`.nat` slot only covers active launches; see the `loraExpandGemvIO`
docstring). An `ADD_INPUTS = true` value face (accumulating onto the
existing output, `tiled_out +` in the spec) is future work — the verified
exact stack does not cover that branch either.

Relation to the exact surface: the exact headline
`gemv_full_output_summary` (`Realizes_without_Rounding`) above is retained
unchanged; this `⊨[R]` face restates the same GEMV content on the streaming
metadata emit skin, for every `R` at once (at the `.real` grid the two faces
carry the same exact cell). Both faces are kept per the rounding-as-default
doctrine. -/
specification gemv_full_io_correctness (R : RoundingModel)
    (input_ptr lora_ptr out_ptr : RegionName) (lora_indices : Region .nat)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride BLOCK_N BLOCK_K : Nat)
    (hBN : 0 < BLOCK_N) (hKB : K ≤ BLOCK_K) (hsnl : 0 < split_n_length)
    (hol : out_ptr ≠ lora_ptr) (hcn : 0 < cn_stride) :
    loraExpandGemvIO input_ptr lora_ptr out_ptr lora_indices K split_n_length
        xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride
        cn_stride BLOCK_N BLOCK_K ⊨[R]
      fun _ _ _ _ xs ys t j =>
        ∑ k : Fin BLOCK_K, if k.val < K then
          xs t k * ys t (bLane BLOCK_N BLOCK_K j k) else 0 := by
  refine StreamMetaEmitMasked3DKernelIO₂.ImplementsR.intro _ ?_ ?_ ?_
  · exact bgmv_flattenOk input_ptr lora_ptr out_ptr lora_indices K split_n_length
      xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
      BLOCK_N BLOCK_K
  · -- safety walk
    intro bounds s m xs ys hm _hx _hy hbm hbr1 hbr2 hbw
    simp only [loraExpandGemvIO] at hm hbm hbr1 hbr2 hbw
    have hm0 : loraIdx s lora_indices = m (⟨0, by omega⟩ : Fin 1) :=
      hm (⟨0, by omega⟩ : Fin 1)
    have hT0 : 0 < bgmvNumSteps split_n_length BLOCK_N := by
      have h := bgmvStep_lt_numSteps split_n_length BLOCK_N 0 hBN hsnl
      simpa using h
    refine bgmv_traceSafeR R bounds input_ptr lora_ptr out_ptr lora_indices K split_n_length
      xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
      BLOCK_N BLOCK_K hBN hKB hol hcn s (hbm (⟨0, by omega⟩ : Fin 1)) ?_ ?_ ?_
    · intro k hk
      exact hbr1 (⟨0, hT0⟩ : Fin (bgmvNumSteps split_n_length BLOCK_N)) k hk
    · intro t j k hj hk
      have h := hbr2 t (bLane BLOCK_N BLOCK_K j k)
        ⟨by rw [bLane_div]; exact hj, by rw [bLane_mod]; exact hk⟩
      rw [bLane_div, bLane_mod, ← hm0] at h
      exact h
    · intro t j hj
      exact hbw t j hj
  · -- the rounded Hoare triple
    intro s₀ m xs ys hundef hm hx hy
    simp only [loraExpandGemvIO] at hm hx hy ⊢
    have hundef' : ∀ rg o, s₀.undef rg o = 0 := fun rg o => by rw [hundef]
    have hm0 : loraIdx s₀ lora_indices = m (⟨0, by omega⟩ : Fin 1) :=
      hm (⟨0, by omega⟩ : Fin 1)
    have hx' : ∀ (t : Fin (bgmvNumSteps split_n_length BLOCK_N)) (k : Fin BLOCK_K),
        k.val < K →
        s₀.readMem input_ptr (s₀.pids 1 * xm_stride + k.val * xk_stride) = xs t k :=
      fun t k hk => hx t k hk
    have hy' : ∀ (t : Fin (bgmvNumSteps split_n_length BLOCK_N))
        (l : Fin (BLOCK_N * BLOCK_K)),
        t.val * BLOCK_N + l.val / BLOCK_K < split_n_length → l.val % BLOCK_K < K →
        s₀.readMem lora_ptr (l0_stride * loraIdx s₀ lora_indices
            + s₀.pids 0 * split_n_length * lora_k_stride
            + (t.val * BLOCK_N + l.val / BLOCK_K) * lora_k_stride
            + (l.val % BLOCK_K) * lora_n_stride) = ys t l := by
      intro t l h1 h2
      rw [hm0]
      exact hy t l ⟨h1, h2⟩
    obtain ⟨sfin, hexec, hval, hframe⟩ := bgmv_runR R input_ptr lora_ptr out_ptr lora_indices
      K split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride BLOCK_N BLOCK_K hBN hKB hol hcn s₀ hundef'
    refine ⟨sfin, hexec, ?_, ?_⟩
    · intro t j hj
      have hjlt : t.val * BLOCK_N + j.val < split_n_length := hj
      have hval' : sfin.readMem out_ptr
          (outOffG s₀ split_n_length cm_stride cn_stride (t.val * BLOCK_N + j.val))
          = gemvSpec s₀ input_ptr lora_ptr lora_indices K split_n_length
              xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
              (t.val * BLOCK_N + j.val) :=
        hval ⟨t.val * BLOCK_N + j.val, hjlt⟩
      have haddr : s₀.pids 1 * cm_stride + s₀.pids 0 * split_n_length
            + (t.val * BLOCK_N + j.val) * cn_stride
          = outOffG s₀ split_n_length cm_stride cn_stride (t.val * BLOCK_N + j.val) := rfl
      rw [haddr, BlockState.readMemAs_real, hval',
        gemvSpec_eq_streamSum input_ptr lora_ptr lora_indices K split_n_length xm_stride
          xk_stride l0_stride lora_k_stride lora_n_stride BLOCK_N BLOCK_K hKB s₀ xs ys
          hx' hy' t j hjlt]
      simp [FloatDType.ofReal]
    · intro r o hcond
      refine hframe r o ?_
      rcases hcond with hne | hno
      · exact Or.inl hne
      · refine Or.inr fun mm => ?_
        have hdm : mm.val / BLOCK_N * BLOCK_N + mm.val % BLOCK_N = mm.val := by
          rw [Nat.mul_comm]
          exact Nat.div_add_mod mm.val BLOCK_N
        have h := hno ⟨mm.val / BLOCK_N,
            bgmvStep_lt_numSteps split_n_length BLOCK_N mm.val hBN mm.isLt⟩
          ⟨mm.val % BLOCK_N, Nat.mod_lt _ hBN⟩ (by simp only [hdm]; exact mm.isLt)
        simpa [hdm, outOffG] using h

end IOFace

end VeriTile.Bench.TritonBenchG.LoraExpandGemv
