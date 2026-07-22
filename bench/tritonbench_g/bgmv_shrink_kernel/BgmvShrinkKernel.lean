import VeriTile.Triton

/-!
# `bgmv_shrink_kernel` — strict per-kernel correctness

`_bgmv_shrink_kernel` is the batched LoRA *shrink* GEMV: program
`(pid_sk, cur_batch)` selects the LoRA-A weight matrix via
`lora_indices[cur_batch]` (with the signed `-1` sentinel skipping the batch),
walks its `pid_sk`-th split of the rank dimension `K` in strides of
`BLOCK_K · SPLIT_K` accumulating `accumulator += tl.sum(tiled_a * tiled_b, 1)`,
scales by `scaling`, and either **masked-stores** (`SPLIT_K == 1`) or
**atomically adds** (`SPLIT_K > 1`, `tl.atomic_add`) the `BLOCK_N`-vector into
`out[cur_batch, 0:N]`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`_bgmv_shrink_kernel[grid](...)`, the 2D grid
`(SPLIT_K, batches)`, the host stride/shape setup, and how the runtime composes
per-program writes) is the *trusted boundary*, not a proof obligation here.
Because the program ids are universally quantified, the per-program statement
covers every program of the grid. For the `SPLIT_K > 1` atomic branch, the
per-program obligation is `out-after = out-before + contribution`; the
cross-program sum over `pid_sk` is composed by the host (trusted boundary),
consistent with the FA-1/FA-2 backward atomic kernels.

## Proof architecture

```
bgmv_shrink_kernel_output_summary_general        ← TOP THEOREM (bundled)
  ├─ bgmv_shrink_surface_toAlgorithm_supported     faithful guarded surface lowers
  ├─ bgmv_shrink_{store,atomic}_surface_toAlgorithm_supported
  ├─ bgmv_shrink_sentinel_skip_no_write            lora_index = -1 ⇒ memory untouched
  ├─ store_compute_correct  (SPLIT_K = 1 branch)   ComputeCorrect.Realizes_without_Rounding, masked store
  │    └─ store_exec_correct
  └─ atomic_compute_correct (SPLIT_K > 1 branch)   ComputeCorrect.Realizes_without_Rounding, atomic add:
       └─ atomic_exec_correct                        out-after = out-before + contribution
            ├─ shrink_preLoop      (P 0: registers seeded, accumulator = 0)
            ├─ shrink_step         (one K-split block advances the partial sum)
            ├─ forRange_inv        (loop-invariant principle, drives the K-loop)
            ├─ tail_common_steps   (accumulator *= scaling; c_ptr/c_mask seeded)
            └─ foldl_atomicAdd_at / foldl_atomicAdd_preserve
                                   (in-file masked atomic-RMW scatter readback;
                                    store branch uses ScatterStore.foldl_store_at)
```

For every active output lane `n < N`, the per-program contribution is the
genuine rank-slice contraction

```
scaling · Σ_{c < ⌈K/(BLOCK_K·SPLIT_K)⌉} Σ_{e < BLOCK_K}
  [k(c,e) < K] · input[cur_batch·xm_stride + k(c,e)]
              · loraA[l0_stride·lora_index + n·lora_k_stride + k(c,e)·lora_n_stride]
```

over `ℝ`, where `k(c,e) = c·BLOCK_K·SPLIT_K + pid_sk·BLOCK_K + e` is the
kernel's own `current_k` index — this program's `pid_sk`-slice of `[0,K)` —
NOT the kernel's own emitted value.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float; the fp16/bf16 carriers
erase to the real channel at the algorithm layer). The masked loads pad with
`other=0.0` exactly as upstream; the `for k in range(0, K, BLOCK_K*SPLIT_K)`
loop is transcribed with its masked tail, so `K` is **not** required to be a
multiple of the split stride. `tl.max_contiguous` is a layout hint erased to
its value argument. The upstream `xk_stride` argument is dropped (it is unused
in the upstream kernel body — the host asserts `inputs.is_contiguous()`).
Output-offset injectivity is discharged from `0 < cn_stride`. The atomic
branch's statement is per-program (single interleaving of one program's
lane-fold); true inter-program interleaving is the host boundary above.

Translation-surface blocker: the Python early `return` on the signed sentinel
(`lora_index == -1`) is represented as an `if lora_index != -1 { ... }` guard
around the active body in the faithful surface (the DSL has no early exit) —
the guard-false path is proven write-free (`bgmv_shrink_sentinel_skip_no_write`)
and the two proof surfaces verify the active body with a `Region .nat`-typed
`lora_indices` (the sentinel skip is the trusted host boundary, as in
`lora_expand_gemv`/`sgmv_expand_slice`). The constexpr tail branch
`if SPLIT_K == 1: tl.store(...) else: tl.atomic_add(...)` is split into two
proof surfaces (`bgmv_shrink_store_surface` / `bgmv_shrink_atomic_surface`),
one per branch (`matmul_tma` precedent), gated by a `SPLIT_K_ONE : Bool` in
the faithful surface. `tl.max_contiguous` is erased to its value argument, and
the upstream `xk_stride` parameter is dropped as unused. The textual py↔lean
scans in `bench/audit_tritonbench_g.sh` exempt this port on this marker
(registered in `proof_blockers.md`).
-/

namespace VeriTile.Bench.TritonBenchG.BgmvShrinkKernel

open VeriTile.Triton

set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

/-- Faithful transcription of `bgmv_shrink_kernel.py`'s `_bgmv_shrink_kernel`.

Python's signed `lora_index == -1` early return is represented as a guard
around the active body; the constexpr `SPLIT_K == 1` store-vs-atomic tail is
gated by `SPLIT_K_ONE` (`= decide (SPLIT_K = 1)` at the trusted host boundary).
`tl.max_contiguous` is a layout hint (erased to its value argument by the
DSL). -/
def bgmv_shrink_surface
    (input_ptr lora_ptr out_ptr : RegionName)
    (N K : Nat) (lora_indices : Region .int) (scaling : ℝ)
    (xm_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
      BLOCK_N BLOCK_K SPLIT_K : Nat) (SPLIT_K_ONE : Bool) :
    ComputeKernel := triton {
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
    accumulator *= $(scaling)
    offset_cn = tl.arange(0, $(BLOCK_N))
    c_ptr = out_ptr + cur_batch * $(cm_stride) + offset_cn * $(cn_stride)
    c_mask = offset_cn < $(N)
    if SPLIT_K_ONE {
      tl.store(c_ptr, accumulator, mask=c_mask)
    } else {
      tl.atomic_add(c_ptr, accumulator, mask=c_mask)
    }
  }
}

/-- The faithful guarded surface (sentinel guard, both `SPLIT_K` tail branches)
lowers to the algorithm layer. -/
theorem bgmv_shrink_surface_toAlgorithm_supported
    (input_ptr lora_ptr out_ptr : RegionName)
    (N K : Nat) (lora_indices : Region .int) (scaling : ℝ)
    (xm_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
      BLOCK_N BLOCK_K SPLIT_K : Nat) (SPLIT_K_ONE : Bool) :
    ∃ alg, (bgmv_shrink_surface input_ptr lora_ptr out_ptr N K lora_indices
      scaling xm_stride l0_stride lora_k_stride lora_n_stride cm_stride
      cn_stride BLOCK_N BLOCK_K SPLIT_K SPLIT_K_ONE).toAlgorithm? =
        Except.ok alg := by
  simp [bgmv_shrink_surface, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Except.bind, bind]

/-- The `SPLIT_K = 1` proof surface: sentinel guard elided (trusted host
boundary; the host only launches the active body when `lora_index ≠ -1`),
`Region .nat`-typed `lora_indices`, and the `tl.store` tail branch. The loop
stride is kept symbolic in `SPLIT_K` (the branch is taken at `SPLIT_K = 1`). -/
def bgmv_shrink_store_surface
    (input_ptr lora_ptr out_ptr : RegionName)
    (N K : Nat) (lora_indices : Region .nat) (scaling : ℝ)
    (xm_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
      BLOCK_N BLOCK_K SPLIT_K : Nat) :
    ComputeKernel := triton {
  pid_sk = tl.program_id(axis=0)
  cur_batch = tl.program_id(axis=1)
  lora_index = tl.load(lora_indices + cur_batch)
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
  accumulator *= $(scaling)
  offset_cn = tl.arange(0, $(BLOCK_N))
  c_ptr = out_ptr + cur_batch * $(cm_stride) + offset_cn * $(cn_stride)
  c_mask = offset_cn < $(N)
  tl.store(c_ptr, accumulator, mask=c_mask)
}

/-- The `SPLIT_K > 1` proof surface: identical to the store surface except the
tail is the upstream `tl.atomic_add(c_ptr, accumulator, mask=c_mask)`. -/
def bgmv_shrink_atomic_surface
    (input_ptr lora_ptr out_ptr : RegionName)
    (N K : Nat) (lora_indices : Region .nat) (scaling : ℝ)
    (xm_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
      BLOCK_N BLOCK_K SPLIT_K : Nat) :
    ComputeKernel := triton {
  pid_sk = tl.program_id(axis=0)
  cur_batch = tl.program_id(axis=1)
  lora_index = tl.load(lora_indices + cur_batch)
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
  accumulator *= $(scaling)
  offset_cn = tl.arange(0, $(BLOCK_N))
  c_ptr = out_ptr + cur_batch * $(cm_stride) + offset_cn * $(cn_stride)
  c_mask = offset_cn < $(N)
  tl.atomic_add(c_ptr, accumulator, mask=c_mask)
}

/-- The store proof surface lowers to the algorithm layer. -/
theorem bgmv_shrink_store_surface_toAlgorithm_supported
    (input_ptr lora_ptr out_ptr : RegionName)
    (N K : Nat) (lora_indices : Region .nat) (scaling : ℝ)
    (xm_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
      BLOCK_N BLOCK_K SPLIT_K : Nat) :
    ∃ alg, (bgmv_shrink_store_surface input_ptr lora_ptr out_ptr N K
      lora_indices scaling xm_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride BLOCK_N BLOCK_K SPLIT_K).toAlgorithm? =
        Except.ok alg := by
  simp [bgmv_shrink_store_surface, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Except.bind, bind]

/-- The atomic proof surface lowers to the algorithm layer. -/
theorem bgmv_shrink_atomic_surface_toAlgorithm_supported
    (input_ptr lora_ptr out_ptr : RegionName)
    (N K : Nat) (lora_indices : Region .nat) (scaling : ℝ)
    (xm_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
      BLOCK_N BLOCK_K SPLIT_K : Nat) :
    ∃ alg, (bgmv_shrink_atomic_surface input_ptr lora_ptr out_ptr N K
      lora_indices scaling xm_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride BLOCK_N BLOCK_K SPLIT_K).toAlgorithm? =
        Except.ok alg := by
  simp [bgmv_shrink_atomic_surface, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Except.bind, bind]

/-! ## Body decomposition (shared by both proof surfaces) -/

/-- The 6-statement K-split loop body, transcribed (`tl.max_contiguous` erased
to its value argument by the DSL, hence `current_k_c := current_k`). -/
def shrinkLoopBody
    (N K lora_k_stride lora_n_stride BLOCK_N BLOCK_K : Nat) : List Stmt :=
  [ Stmt.assign .nat [BLOCK_K] "current_k"
      (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "k")
        (Op.ref .nat [BLOCK_K] "offset_k")),
    Stmt.assign .nat [BLOCK_K] "current_k_c" (Op.ref .nat [BLOCK_K] "current_k"),
    Stmt.assign .real [BLOCK_K] "tiled_a"
      (Op.load .real
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "a_ptr")
            (Op.ref .nat [BLOCK_K] "current_k_c")))
        (MaskOpt.maskOther
          (Op.lt .nat Broadcast.scalarR (Op.ref .nat [BLOCK_K] "current_k")
            (Op.constNat K))
          ((Op.const 0.0).broadcast [BLOCK_K]))),
    Stmt.assign .bool [BLOCK_N, BLOCK_K] "b_ptr_mask"
      (Op.boolAnd Broadcast.nil.consL.consR
        (Op.lt .nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "offset_n"))
          (Op.constNat N))
        (Op.lt .nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_K] "current_k"))
          (Op.constNat K))),
    Stmt.assign .real [BLOCK_N, BLOCK_K] "tiled_b"
      (Op.load .real
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.nil.consL.consR
            (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "b_ptr")
              (Op.mul .nat Broadcast.scalarR
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "offset_n"))
                (Op.constNat lora_k_stride)))
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_K] "current_k"))
              (Op.constNat lora_n_stride))))
        (MaskOpt.maskOther (Op.ref .bool [BLOCK_N, BLOCK_K] "b_ptr_mask")
          ((Op.const 0.0).broadcast [BLOCK_N, BLOCK_K]))),
    Stmt.assign .real [BLOCK_N] "accumulator"
      (Op.add .real Broadcast.nil.consSame
        (Op.ref .real [BLOCK_N] "accumulator")
        (Op.reduceSum ⟨1, by simp⟩ Bool.false
          (Op.mul .real Broadcast.nil.consSame.leadL
            (Op.ref .real [BLOCK_K] "tiled_a")
            (Op.ref .real [BLOCK_N, BLOCK_K] "tiled_b")))) ]

/-- The 4 post-loop statements shared by both tail branches
(`accumulator *= scaling`, `offset_cn`, `c_ptr`, `c_mask`). -/
def shrinkTailCommon
    (out_ptr : RegionName) (N cm_stride cn_stride BLOCK_N : Nat) (scaling : ℝ) :
    List Stmt :=
  [ Stmt.assign .real [BLOCK_N] "accumulator"
      (Op.mul .real Broadcast.scalarR (Op.ref .real [BLOCK_N] "accumulator")
        (Op.const scaling)),
    Stmt.assign .nat [BLOCK_N] "offset_cn" (Op.arange BLOCK_N),
    Stmt.assign .ptr [BLOCK_N] "c_ptr"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase out_ptr)
        (Op.add .nat Broadcast.scalarL
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_batch")
            (Op.constNat cm_stride))
          (Op.mul .nat Broadcast.scalarR (Op.ref .nat [BLOCK_N] "offset_cn")
            (Op.constNat cn_stride)))),
    Stmt.assign .bool [BLOCK_N] "c_mask"
      (Op.lt .nat Broadcast.scalarR (Op.ref .nat [BLOCK_N] "offset_cn")
        (Op.constNat N)) ]

/-- The masked store tail statement (`SPLIT_K = 1` branch). -/
def shrinkStoreStmt (BLOCK_N : Nat) : Stmt :=
  Stmt.store .real [BLOCK_N] (MemAccess.ptr (Op.ref .ptr [BLOCK_N] "c_ptr"))
    (Op.ref .real [BLOCK_N] "accumulator")
    (MaskOpt.mask (Op.ref .bool [BLOCK_N] "c_mask"))

/-- The masked atomic-add tail statement (`SPLIT_K > 1` branch). -/
def shrinkAtomicStmt (BLOCK_N : Nat) : Stmt :=
  Stmt.atomicAdd NumericDType.real [BLOCK_N]
    (MemAccess.ptr (Op.ref .ptr [BLOCK_N] "c_ptr"))
    (Op.ref .real [BLOCK_N] "accumulator")
    (MaskOpt.mask (Op.ref .bool [BLOCK_N] "c_mask"))

/-- Store-surface body decomposition: prologue (8) ++ [K-loop] ++ common tail
(4) ++ [store]. By `rfl`. -/
theorem store_body_split
    (input_ptr lora_ptr out_ptr : RegionName)
    (N K : Nat) (lora_indices : Region .nat) (scaling : ℝ)
    (xm_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
      BLOCK_N BLOCK_K SPLIT_K : Nat) :
    (bgmv_shrink_store_surface input_ptr lora_ptr out_ptr N K lora_indices
        scaling xm_stride l0_stride lora_k_stride lora_n_stride cm_stride
        cn_stride BLOCK_N BLOCK_K SPLIT_K).toAlgKernel.body
      = (bgmv_shrink_store_surface input_ptr lora_ptr out_ptr N K lora_indices
          scaling xm_stride l0_stride lora_k_stride lora_n_stride cm_stride
          cn_stride BLOCK_N BLOCK_K SPLIT_K).toAlgKernel.body.take 8
        ++ Stmt.forRange "k" 0 K (BLOCK_K * SPLIT_K)
            (shrinkLoopBody N K lora_k_stride lora_n_stride BLOCK_N BLOCK_K)
          :: (shrinkTailCommon out_ptr N cm_stride cn_stride BLOCK_N scaling
              ++ [shrinkStoreStmt BLOCK_N]) := by
  rfl

/-- Atomic-surface body decomposition: identical shape, atomic tail. By `rfl`. -/
theorem atomic_body_split
    (input_ptr lora_ptr out_ptr : RegionName)
    (N K : Nat) (lora_indices : Region .nat) (scaling : ℝ)
    (xm_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
      BLOCK_N BLOCK_K SPLIT_K : Nat) :
    (bgmv_shrink_atomic_surface input_ptr lora_ptr out_ptr N K lora_indices
        scaling xm_stride l0_stride lora_k_stride lora_n_stride cm_stride
        cn_stride BLOCK_N BLOCK_K SPLIT_K).toAlgKernel.body
      = (bgmv_shrink_atomic_surface input_ptr lora_ptr out_ptr N K lora_indices
          scaling xm_stride l0_stride lora_k_stride lora_n_stride cm_stride
          cn_stride BLOCK_N BLOCK_K SPLIT_K).toAlgKernel.body.take 8
        ++ Stmt.forRange "k" 0 K (BLOCK_K * SPLIT_K)
            (shrinkLoopBody N K lora_k_stride lora_n_stride BLOCK_N BLOCK_K)
          :: (shrinkTailCommon out_ptr N cm_stride cn_stride BLOCK_N scaling
              ++ [shrinkAtomicStmt BLOCK_N]) := by
  rfl

/-! ## Genuine closed-form spec -/

/-- The selected LoRA index (`lora_indices[cur_batch]`). -/
def loraIdx (s : BlockState) (lora_indices : Region .nat) : Nat :=
  s.readMemValue .nat lora_indices (s.pids 1)

/-- Input element `x[k] = input[cur_batch·xm_stride + k]` (upstream addresses
the input row directly by `current_k`; its `xk_stride` argument is unused). -/
noncomputable def aElem (s : BlockState) (input_ptr : RegionName)
    (xm_stride k : Nat) : ℝ :=
  s.readMem input_ptr (s.pids 1 * xm_stride + k)

/-- LoRA-A element
`W[n,k] = lora[l0_stride·lora_index + n·lora_k_stride + k·lora_n_stride]`. -/
noncomputable def bElem (s : BlockState) (lora_ptr : RegionName)
    (lora_indices : Region .nat)
    (l0_stride lora_k_stride lora_n_stride n k : Nat) : ℝ :=
  s.readMem lora_ptr
    (l0_stride * loraIdx s lora_indices + n * lora_k_stride
      + k * lora_n_stride)

/-- The kernel's `current_k` rank index at loop iteration `c`, lane `e`:
`k(c,e) = c·(BLOCK_K·SPLIT_K) + (e + pid_sk·BLOCK_K)` — this program's
(`pid_sk`) slice of the rank dimension. -/
def kIdx (s : BlockState) (BLOCK_K SPLIT_K c e : Nat) : Nat :=
  c * (BLOCK_K * SPLIT_K) + (e + s.pids 0 * BLOCK_K)

/-- One K-split block's masked contribution at output lane `n`:
`Σ_{e<BLOCK_K} [k(c,e) < K] · x[k(c,e)] · W[n,k(c,e)]` (the `tl.sum` of the
masked `tiled_a · tiled_b` products; out-of-range rank lanes load `0`). -/
noncomputable def blockTerm (s : BlockState) (input_ptr lora_ptr : RegionName)
    (lora_indices : Region .nat)
    (K xm_stride l0_stride lora_k_stride lora_n_stride
      BLOCK_K SPLIT_K c n : Nat) : ℝ :=
  (Finset.univ : Finset (Fin BLOCK_K)).sum fun e =>
    if kIdx s BLOCK_K SPLIT_K c e.val < K then
      aElem s input_ptr xm_stride (kIdx s BLOCK_K SPLIT_K c e.val)
        * bElem s lora_ptr lora_indices l0_stride lora_k_stride lora_n_stride
            n (kIdx s BLOCK_K SPLIT_K c e.val)
    else 0

/-- Partial accumulator after `c` K-split blocks at lane `n`. -/
noncomputable def accPartial (s : BlockState) (input_ptr lora_ptr : RegionName)
    (lora_indices : Region .nat)
    (K xm_stride l0_stride lora_k_stride lora_n_stride
      BLOCK_K SPLIT_K c n : Nat) : ℝ :=
  (Finset.range c).sum fun c' =>
    blockTerm s input_ptr lora_ptr lora_indices K xm_stride l0_stride
      lora_k_stride lora_n_stride BLOCK_K SPLIT_K c' n

/-- One-block recurrence: `accPartial (c+1) = accPartial c + blockTerm c`. -/
theorem accPartial_succ (s : BlockState) (input_ptr lora_ptr : RegionName)
    (lora_indices : Region .nat)
    (K xm_stride l0_stride lora_k_stride lora_n_stride
      BLOCK_K SPLIT_K c n : Nat) :
    accPartial s input_ptr lora_ptr lora_indices K xm_stride l0_stride
        lora_k_stride lora_n_stride BLOCK_K SPLIT_K (c + 1) n
      = accPartial s input_ptr lora_ptr lora_indices K xm_stride l0_stride
          lora_k_stride lora_n_stride BLOCK_K SPLIT_K c n
        + blockTerm s input_ptr lora_ptr lora_indices K xm_stride l0_stride
            lora_k_stride lora_n_stride BLOCK_K SPLIT_K c n := by
  unfold accPartial
  rw [Finset.sum_range_succ]

/-- Trip count of `for k in range(0, K, S)`: `⌈K/S⌉`. -/
def numKIters (K S : Nat) : Nat := (K + S - 1) / S

/-- **Genuine spec**: this program's scaled rank-slice contribution at output
lane `n` — `scaling · Σ_{c<⌈K/S⌉} Σ_{e<BLOCK_K} [k(c,e)<K] · x[k(c,e)] ·
W[n,k(c,e)]`, a closed form over INPUT memory. -/
noncomputable def shrinkSpec (s : BlockState) (input_ptr lora_ptr : RegionName)
    (lora_indices : Region .nat) (scaling : ℝ)
    (K xm_stride l0_stride lora_k_stride lora_n_stride
      BLOCK_K SPLIT_K n : Nat) : ℝ :=
  scaling * accPartial s input_ptr lora_ptr lora_indices K xm_stride l0_stride
    lora_k_stride lora_n_stride BLOCK_K SPLIT_K
    (numKIters K (BLOCK_K * SPLIT_K)) n

/-- Output offset of lane `n`: `cur_batch·cm_stride + n·cn_stride`. -/
def outOff (s : BlockState) (cm_stride cn_stride n : Nat) : Nat :=
  s.pids 1 * cm_stride + n * cn_stride

/-- The ceil trip count sandwiches `K`: `K ≤ ⌈K/S⌉·S < K + S`. -/
theorem numKIters_bounds (K S : Nat) (hS : 0 < S) :
    K ≤ numKIters K S * S ∧ numKIters K S * S < K + S := by
  unfold numKIters
  have h := Nat.div_add_mod (K + S - 1) S
  have hm : (K + S - 1) % S < S := Nat.mod_lt _ hS
  rw [show (K + S - 1) / S * S = S * ((K + S - 1) / S) from Nat.mul_comm _ _]
  generalize S * ((K + S - 1) / S) = x at h ⊢
  omega

/-- Any counter with `K ≤ c·S < K + S` is the ceil trip count — pins the
`forRange_inv` exit counter to `numKIters`. -/
theorem kloop_final_iters (K S c : Nat) (hS : 0 < S)
    (h1 : K ≤ c * S) (h2 : c * S < K + S) : c = numKIters K S := by
  obtain ⟨hl, hr⟩ := numKIters_bounds K S hS
  rcases Nat.lt_trichotomy c (numKIters K S) with h | h | h
  · exfalso
    have hcs : c * S + S ≤ numKIters K S * S := by
      rw [show c * S + S = (c + 1) * S from (Nat.succ_mul c S).symm]
      exact Nat.mul_le_mul_right S h
    exact Nat.lt_irrefl _
      (lt_of_le_of_lt (le_trans (Nat.add_le_add_right h1 S) hcs) hr)
  · exact h
  · exfalso
    have hcs : numKIters K S * S + S ≤ c * S := by
      rw [show numKIters K S * S + S = (numKIters K S + 1) * S from
        (Nat.succ_mul _ S).symm]
      exact Nat.mul_le_mul_right S h
    exact Nat.lt_irrefl _
      (lt_of_le_of_lt (le_trans (Nat.add_le_add_right hl S) hcs) h2)

/-! ## Masked atomic-add scatter readback (generic `foldl` lemmas)

`stepStmt` reduces a masked `tl.atomic_add` to a lane `foldl` of per-cell
read-modify-writes. The shared library provides register/pid preservation for
this fold (`foldl_atomicAddAt_regs` / `foldl_atomicAddAt_pid`); the *memory
readback* lemmas below — the atomic analogue of `ScatterStore.foldl_store_at`
/ `foldl_store_preserve` — did not exist yet (this is the first bench consumer
of the atomic-add exec path), so they are proved here. -/

/-- A masked atomic-add `foldl` leaves an offset `o` untouched if no active
lane adds to it. -/
theorem foldl_atomicAdd_preserve {α : Type} {region : RegionName}
    (ofn : α → Nat) (vfn : α → ℝ) (mask : α → Bool) (o : Nat) (l : List α)
    (s : BlockState) (hnot : ∀ k ∈ l, mask k → ofn k ≠ o) :
    (l.foldl (fun acc k =>
        if mask k then
          acc.writeMem region (ofn k) (acc.readMem region (ofn k) + vfn k)
        else acc) s).readMem region o = s.readMem region o := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
    rw [List.foldl_cons]
    cases hm : mask hd
    · simp only [hm, Bool.false_eq_true, if_false]
      exact ih _ (fun k hk hmk => hnot k (List.mem_cons_of_mem hd hk) hmk)
    · simp only [hm, if_true]
      rw [ih _ (fun k hk hmk => hnot k (List.mem_cons_of_mem hd hk) hmk)]
      exact BlockState.writeMem_readMem_of_ne_offset s region (ofn hd) _ region o
        (hnot hd List.mem_cons_self (by rw [hm])).symm

/-- Readback at offset `O` of a masked atomic-add `foldl`: if a unique active
lane `a` adds to `O`, the readback is the **initial value plus its
contribution** — the per-program atomic-add obligation. -/
theorem foldl_atomicAdd_at {α : Type} {region : RegionName}
    (ofn : α → Nat) (vfn : α → ℝ) (mask : α → Bool) (O : Nat) (l : List α)
    (s : BlockState) (a : α) (ha : a ∈ l) (hma : mask a) (hoa : ofn a = O)
    (huniq : ∀ b ∈ l, mask b → ofn b = O → b = a) (hnodup : l.Nodup) :
    (l.foldl (fun acc k =>
        if mask k then
          acc.writeMem region (ofn k) (acc.readMem region (ofn k) + vfn k)
        else acc) s).readMem region O
      = s.readMem region O + vfn a := by
  obtain ⟨l₁, l₂, hl⟩ := List.append_of_mem ha
  subst hl
  rw [List.foldl_append, List.foldl_cons]
  rw [List.nodup_append, List.nodup_cons] at hnodup
  obtain ⟨hnd1, ⟨ha_notin2, hnd2⟩, hdisj⟩ := hnodup
  have h2 : ∀ b ∈ l₂, mask b → ofn b ≠ O := by
    intro b hb hmb heq
    exact ha_notin2 ((huniq b (by simp [List.mem_append, hb]) hmb heq) ▸ hb)
  have h1 : ∀ b ∈ l₁, mask b → ofn b ≠ O := by
    intro b hb hmb heq
    exact (hdisj b hb a List.mem_cons_self)
      (huniq b (by simp [List.mem_append, hb]) hmb heq)
  rw [foldl_atomicAdd_preserve ofn vfn mask O l₂ _ h2]
  simp only [hma, if_true, hoa]
  rw [BlockState.writeMem_readMem]
  rw [if_pos ⟨rfl, rfl⟩]
  rw [foldl_atomicAdd_preserve ofn vfn mask O l₁ _ h1]

/-! ## Prologue -/

/-- The 8 prologue statements (= `body.take 8` of both proof surfaces). -/
def shrinkPrologue
    (input_ptr lora_ptr : RegionName) (lora_indices : Region .nat)
    (xm_stride l0_stride BLOCK_N BLOCK_K : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "pid_sk" (Op.programId 0),
    Stmt.assign .nat [] "cur_batch" (Op.programId 1),
    Stmt.assign .nat [] "lora_index"
      (Op.load .nat (MemAccess.region lora_indices (Op.ref .nat [] "cur_batch"))
        MaskOpt.none),
    Stmt.assign .nat [BLOCK_N] "offset_n" (Op.arange BLOCK_N),
    Stmt.assign .nat [BLOCK_K] "offset_k"
      (Op.add .nat Broadcast.scalarR (Op.arange BLOCK_K)
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_sk")
          (Op.constNat BLOCK_K))),
    Stmt.assign .ptr [] "a_ptr"
      (Op.ptrAdd Broadcast.nil (Op.ptrBase input_ptr)
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_batch")
          (Op.constNat xm_stride))),
    Stmt.assign .ptr [] "b_ptr"
      (Op.ptrAdd Broadcast.nil (Op.ptrBase lora_ptr)
        (Op.mul .nat Broadcast.nil (Op.constNat l0_stride)
          (Op.ref .nat [] "lora_index"))),
    Stmt.assign .real [BLOCK_N] "accumulator" (Op.full [BLOCK_N] (Op.const 0)) ]

/-- The store surface's first 8 statements are the shared prologue. By `rfl`. -/
theorem store_prologue_split
    (input_ptr lora_ptr out_ptr : RegionName)
    (N K : Nat) (lora_indices : Region .nat) (scaling : ℝ)
    (xm_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
      BLOCK_N BLOCK_K SPLIT_K : Nat) :
    (bgmv_shrink_store_surface input_ptr lora_ptr out_ptr N K lora_indices
        scaling xm_stride l0_stride lora_k_stride lora_n_stride cm_stride
        cn_stride BLOCK_N BLOCK_K SPLIT_K).toAlgKernel.body.take 8
      = shrinkPrologue input_ptr lora_ptr lora_indices xm_stride l0_stride
          BLOCK_N BLOCK_K := by
  rfl

/-- The atomic surface's first 8 statements are the shared prologue. By `rfl`. -/
theorem atomic_prologue_split
    (input_ptr lora_ptr out_ptr : RegionName)
    (N K : Nat) (lora_indices : Region .nat) (scaling : ℝ)
    (xm_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
      BLOCK_N BLOCK_K SPLIT_K : Nat) :
    (bgmv_shrink_atomic_surface input_ptr lora_ptr out_ptr N K lora_indices
        scaling xm_stride l0_stride lora_k_stride lora_n_stride cm_stride
        cn_stride BLOCK_N BLOCK_K SPLIT_K).toAlgKernel.body.take 8
      = shrinkPrologue input_ptr lora_ptr lora_indices xm_stride l0_stride
          BLOCK_N BLOCK_K := by
  rfl

/-! ## Loop invariant -/

/-- **K-loop invariant** (counter `i = c · BLOCK_K·SPLIT_K`). After `c`
K-split blocks: pids fixed, the prologue registers seeded, memory untouched,
and `accumulator` holds the partial rank-slice sum `accPartial c` at active
lanes (`n < N`; inactive lanes hold `0` — their `tiled_b` loads are fully
masked). -/
noncomputable def shrinkInv
    (input_ptr lora_ptr : RegionName) (lora_indices : Region .nat)
    (N K xm_stride l0_stride lora_k_stride lora_n_stride
      BLOCK_N BLOCK_K SPLIT_K : Nat) (s0 : BlockState)
    (i : Nat) (s : BlockState) : Prop :=
  let c := i / (BLOCK_K * SPLIT_K)
  s.pids = s0.pids ∧ i = c * (BLOCK_K * SPLIT_K) ∧
  i < K + BLOCK_K * SPLIT_K ∧
  s.regs .nat [] "cur_batch" = some (Tile.scalar (s0.pids 1)) ∧
  s.regs .nat [BLOCK_N] "offset_n"
    = some (Tile.vec (fun j : Fin BLOCK_N => j.val)) ∧
  s.regs .nat [BLOCK_K] "offset_k"
    = some (Tile.vec (fun e : Fin BLOCK_K => e.val + s0.pids 0 * BLOCK_K)) ∧
  s.regs .ptr [] "a_ptr"
    = some (Tile.scalar ((input_ptr, s0.pids 1 * xm_stride) : RegionName × Nat)) ∧
  s.regs .ptr [] "b_ptr"
    = some (Tile.scalar
        ((lora_ptr, l0_stride * loraIdx s0 lora_indices) : RegionName × Nat)) ∧
  s.regs .real [BLOCK_N] "accumulator"
    = some ⟨fun idx : TileIndex [BLOCK_N] =>
        some (if idx.1.val < N then
          accPartial s0 input_ptr lora_ptr lora_indices K xm_stride l0_stride
            lora_k_stride lora_n_stride BLOCK_K SPLIT_K c idx.1.val
        else 0)⟩ ∧
  s.mem = s0.mem

/-! ## Prologue eval helpers -/

theorem evalOp_ptrAdd' {a b shape} (bc : Broadcast a b shape)
    (ptr : Op .ptr a) (off : Op .nat b) (s : BlockState) :
    evalOp (.ptrAdd bc ptr off) s = (do
      let ptrs ← evalOp ptr s; let offs ← evalOp off s;
      some (Tile.ptrAdd bc ptrs offs)) := by simp [evalOp]

theorem evalOp_ptrBase' (region : RegionName) (s : BlockState) :
    evalOp (.ptrBase region) s = some (Tile.scalar (region.cast, 0)) := by
  simp [evalOp]

/-- `offset_k = tl.arange(0,BLOCK_K) + pid_sk·BLOCK_K` eval. -/
theorem offk_eval (BLOCK_K psk : Nat) (s : BlockState)
    (hsk : s.regs .nat [] "pid_sk" = some (Tile.scalar psk)) :
    evalOp (Op.add .nat Broadcast.scalarR (Op.arange BLOCK_K)
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_sk")
          (Op.constNat BLOCK_K))) s
      = some (Tile.vec (fun e : Fin BLOCK_K => e.val + psk * BLOCK_K)) := by
  have ha : evalOp (Op.arange BLOCK_K) s
      = some (Tile.vec (fun e : Fin BLOCK_K => e.val)) := by simp [Tile.vec]
  simp only [evalOp_add, evalOp_mul, evalOp_ref, evalOp_constNat, ha, hsk,
    Option.bind_some, bind, Option.bind_eq_bind]
  apply congrArg some
  apply Tile.ext
  intro e
  simp only [Tile.bop, Tile.vec, Tile.scalar, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add, NumericDType.mul]

/-- `a_ptr = input_ptr + cur_batch·xm_stride` eval. -/
theorem aptr_eval (input_ptr : RegionName) (xm_stride cb : Nat) (s : BlockState)
    (hcb : s.regs .nat [] "cur_batch" = some (Tile.scalar cb)) :
    evalOp (Op.ptrAdd Broadcast.nil (Op.ptrBase input_ptr)
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_batch")
          (Op.constNat xm_stride))) s
      = some (Tile.scalar ((input_ptr, cb * xm_stride) : RegionName × Nat)) := by
  rw [evalOp_ptrAdd', evalOp_ptrBase']
  simp only [evalOp_mul, evalOp_ref, evalOp_constNat, hcb, Option.bind_some,
    bind, Option.bind_eq_bind]
  apply congrArg some
  apply Tile.ext
  intro _
  simp only [Tile.ptrAdd, Tile.bop, Tile.scalar, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.mul, Region.cast]
  exact Prod.ext rfl (Nat.zero_add _)

/-- `b_ptr = lora_ptr + l0_stride·lora_index` eval. -/
theorem bptr_eval (lora_ptr : RegionName) (l0_stride li : Nat) (s : BlockState)
    (hli : s.regs .nat [] "lora_index" = some (Tile.scalar li)) :
    evalOp (Op.ptrAdd Broadcast.nil (Op.ptrBase lora_ptr)
        (Op.mul .nat Broadcast.nil (Op.constNat l0_stride)
          (Op.ref .nat [] "lora_index"))) s
      = some (Tile.scalar ((lora_ptr, l0_stride * li) : RegionName × Nat)) := by
  rw [evalOp_ptrAdd', evalOp_ptrBase']
  simp only [evalOp_mul, evalOp_ref, evalOp_constNat, hli, Option.bind_some,
    bind, Option.bind_eq_bind]
  apply congrArg some
  apply Tile.ext
  intro _
  simp only [Tile.ptrAdd, Tile.bop, Tile.scalar, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.mul, Region.cast]
  exact Prod.ext rfl (Nat.zero_add _)

/-- `accumulator = tl.zeros([BLOCK_N])` eval. -/
theorem acc_init_eval (BLOCK_N : Nat) (s : BlockState) :
    evalOp (Op.full [BLOCK_N] (Op.const 0)) s
      = some (⟨fun _ : TileIndex [BLOCK_N] => some (0 : ℝ)⟩ : Tile .real [BLOCK_N]) := by
  simp [evalOp_full, evalOp_const, Option.bind]

set_option maxHeartbeats 2000000 in
/-- **preLoop**: the 8 prologue statements step to a state satisfying the
invariant at `i = 0` (registers seeded, `accumulator = 0`, memory
untouched). -/
theorem shrink_preLoop
    (input_ptr lora_ptr : RegionName) (lora_indices : Region .nat)
    (N K xm_stride l0_stride lora_k_stride lora_n_stride
      BLOCK_N BLOCK_K SPLIT_K : Nat) (s : BlockState)
    (hS : 0 < BLOCK_K * SPLIT_K) :
    ∃ s', stepStmts (shrinkPrologue input_ptr lora_ptr lora_indices xm_stride
        l0_stride BLOCK_N BLOCK_K) s = some s'
      ∧ shrinkInv input_ptr lora_ptr lora_indices N K xm_stride l0_stride
          lora_k_stride lora_n_stride BLOCK_N BLOCK_K SPLIT_K s 0 s' := by
  unfold shrinkPrologue
  -- pid_sk, cur_batch
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (show evalOp (Op.programId 0) s = some (Tile.scalar (s.pids 0)) by simp))]
  set s1 := s.setReg "pid_sk" .nat [] (Tile.scalar (s.pids 0)) with hs1
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (show evalOp (Op.programId 1) s1 = some (Tile.scalar (s.pids 1)) by
          simp [hs1]))]
  set s2 := s1.setReg "cur_batch" .nat [] (Tile.scalar (s.pids 1)) with hs2
  -- lora_index load
  have hli2 : evalOp (Op.load .nat
      (MemAccess.region lora_indices (Op.ref .nat [] "cur_batch")) MaskOpt.none) s2
      = some (Tile.scalar (loraIdx s lora_indices)) := by
    simp only [evalOp, evalOp_ref, hs2, BlockState.setReg_same, Option.bind, loraIdx,
      BlockState.setReg_pids, BlockState.setReg_readMemValue, Tile.scalar]
    rfl
  rw [stepStmts.cons_some (stepStmt_assign_eq_some hli2)]
  set s3 := s2.setReg "lora_index" .nat [] (Tile.scalar (loraIdx s lora_indices)) with hs3
  -- offset_n
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (show evalOp (Op.arange BLOCK_N) s3
            = some (Tile.vec (fun j : Fin BLOCK_N => j.val)) by simp [Tile.vec]))]
  set s4 := s3.setReg "offset_n" .nat [BLOCK_N] (Tile.vec (fun j : Fin BLOCK_N => j.val)) with hs4
  -- offset_k
  have hsk4 : s4.regs .nat [] "pid_sk" = some (Tile.scalar (s.pids 0)) := by
    rw [hs4, hs3, hs2]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("pid_sk":RegName) ≠ "offset_n" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("pid_sk":RegName) ≠ "lora_index" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("pid_sk":RegName) ≠ "cur_batch" by decide)]
    rw [hs1, BlockState.setReg_same]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (offk_eval BLOCK_K (s.pids 0) s4 hsk4))]
  set s5 := s4.setReg "offset_k" .nat [BLOCK_K]
    (Tile.vec (fun e : Fin BLOCK_K => e.val + s.pids 0 * BLOCK_K)) with hs5
  -- a_ptr
  have hcb5 : s5.regs .nat [] "cur_batch" = some (Tile.scalar (s.pids 1)) := by
    rw [hs5, hs4, hs3]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("cur_batch":RegName) ≠ "offset_k" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("cur_batch":RegName) ≠ "offset_n" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("cur_batch":RegName) ≠ "lora_index" by decide)]
    rw [hs2, BlockState.setReg_same]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (aptr_eval input_ptr xm_stride (s.pids 1) s5 hcb5))]
  set s6 := s5.setReg "a_ptr" .ptr []
    (Tile.scalar ((input_ptr, s.pids 1 * xm_stride) : RegionName × Nat)) with hs6
  -- b_ptr
  have hli6 : s6.regs .nat [] "lora_index" = some (Tile.scalar (loraIdx s lora_indices)) := by
    rw [hs6, hs5, hs4]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("lora_index":RegName) ≠ "a_ptr" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("lora_index":RegName) ≠ "offset_k" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("lora_index":RegName) ≠ "offset_n" by decide)]
    rw [hs3, BlockState.setReg_same]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (bptr_eval lora_ptr l0_stride (loraIdx s lora_indices) s6 hli6))]
  set s7 := s6.setReg "b_ptr" .ptr []
    (Tile.scalar ((lora_ptr, l0_stride * loraIdx s lora_indices) : RegionName × Nat)) with hs7
  -- accumulator
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (acc_init_eval BLOCK_N s7))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  simp only [shrinkInv, Nat.zero_div]
  refine ⟨?_, by simp, by simpa using Nat.lt_of_lt_of_le hS (Nat.le_add_left _ K),
    ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · -- pids
    simp only [BlockState.setReg_pids, hs7, hs6, hs5, hs4, hs3, hs2, hs1]
  · -- cur_batch
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("cur_batch":RegName) ≠ "accumulator" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("cur_batch":RegName) ≠ "b_ptr" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("cur_batch":RegName) ≠ "a_ptr" by decide)]
    exact hcb5
  · -- offset_n
    rw [hs7, hs6, hs5, hs4]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_n":RegName) ≠ "accumulator" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_n":RegName) ≠ "b_ptr" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_n":RegName) ≠ "a_ptr" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_n":RegName) ≠ "offset_k" by decide),
      BlockState.setReg_same]
  · -- offset_k
    rw [hs7, hs6, hs5]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_k":RegName) ≠ "accumulator" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_k":RegName) ≠ "b_ptr" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_k":RegName) ≠ "a_ptr" by decide),
      BlockState.setReg_same]
  · -- a_ptr
    rw [hs7, hs6]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("a_ptr":RegName) ≠ "accumulator" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("a_ptr":RegName) ≠ "b_ptr" by decide),
      BlockState.setReg_same]
  · -- b_ptr
    rw [hs7]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("b_ptr":RegName) ≠ "accumulator" by decide),
      BlockState.setReg_same]
  · -- accumulator = accPartial 0 = 0
    rw [BlockState.setReg_same]
    apply congrArg some
    apply Tile.ext
    intro idx
    simp only [accPartial, Nat.zero_div, Finset.range_zero, Finset.sum_empty, ite_self]
  · -- mem
    rw [hs7, hs6, hs5, hs4, hs3, hs2, hs1]
    rfl

/-! ## Loop-body eval helpers -/

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

/-- `current_k = k + offset_k` eval. -/
theorem currentK_eval (BLOCK_K i : Nat) (g : Fin BLOCK_K → Nat) (s : BlockState)
    (hk : s.regs .nat [] "k" = some (Tile.scalar i))
    (hok : s.regs .nat [BLOCK_K] "offset_k" = some (Tile.vec g)) :
    evalOp (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "k")
        (Op.ref .nat [BLOCK_K] "offset_k")) s
      = some (Tile.vec (fun e : Fin BLOCK_K => i + g e)) := by
  simp only [evalOp_add, evalOp_ref, hk, hok, Option.bind]
  apply congrArg some
  apply Tile.ext
  intro e
  simp only [Tile.bop, Tile.scalar, Tile.vec, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add]

/-- `tiled_a = tl.load(a_ptr + current_k_c, mask=current_k < K, other=0.0)`
eval: lane `e` reads `input[apo + ck e]` when `ck e < K`, else `0`. -/
theorem tiledA_eval (input_ptr : RegionName) (K BLOCK_K apo : Nat)
    (ck : Fin BLOCK_K → Nat) (s0 s : BlockState)
    (hap : s.regs .ptr [] "a_ptr"
      = some (Tile.scalar ((input_ptr, apo) : RegionName × Nat)))
    (hckc : s.regs .nat [BLOCK_K] "current_k_c" = some (Tile.vec ck))
    (hck : s.regs .nat [BLOCK_K] "current_k" = some (Tile.vec ck))
    (hrm : ∀ o, s.readMem input_ptr o = s0.readMem input_ptr o) :
    evalOp (Op.load .real
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "a_ptr")
            (Op.ref .nat [BLOCK_K] "current_k_c")))
        (MaskOpt.maskOther
          (Op.lt .nat Broadcast.scalarR (Op.ref .nat [BLOCK_K] "current_k")
            (Op.constNat K))
          ((Op.const 0.0).broadcast [BLOCK_K]))) s
      = some (⟨fun e : TileIndex [BLOCK_K] =>
          some (if ck e.1 < K then s0.readMem input_ptr (apo + ck e.1) else 0)⟩
            : Tile .real [BLOCK_K]) := by
  simp only [evalOp, evalOp.eq_def, evalOp_ref, hap, hckc, hck, Option.bind, Option.map,
    Tile.bop, Tile.cop, Tile.ptrAdd, Tile.scalar, Tile.vec,
    Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul,
    ComparableDType.lt]
  apply congrArg some
  apply Tile.ext
  intro e
  simp only [Tile.cop, Tile.ptrAdd, Tile.bop, Tile.scalar, Tile.vec]
  by_cases hkK : ck e.1 < K
  · simp only [hkK, decide_true, if_true, BlockState.readMemValue_real, hrm]
  · simp only [hkK, decide_false, Bool.false_eq_true, if_false]
    norm_num

/-- `b_ptr_mask = (offset_n[:,None] < N) & (current_k[None,:] < K)` eval. -/
theorem bmask_eval (N K BLOCK_N BLOCK_K : Nat) (ck : Fin BLOCK_K → Nat)
    (s : BlockState)
    (hon : s.regs .nat [BLOCK_N] "offset_n"
      = some (Tile.vec (fun j : Fin BLOCK_N => j.val)))
    (hck : s.regs .nat [BLOCK_K] "current_k" = some (Tile.vec ck)) :
    evalOp (Op.boolAnd Broadcast.nil.consL.consR
        (Op.lt .nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "offset_n"))
          (Op.constNat N))
        (Op.lt .nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_K] "current_k"))
          (Op.constNat K))) s
      = some (⟨fun idx : TileIndex [BLOCK_N, BLOCK_K] =>
          decide (idx.1.val < N) && decide (ck idx.2.1 < K)⟩
            : Tile .bool [BLOCK_N, BLOCK_K]) := by
  simp only [evalOp, evalOp.eq_def, evalOp_expandDim_one_nat, evalOp_expandDim_zero_nat,
    evalOp_ref, hon, hck, Option.bind_some, bind, Option.bind_eq_bind, Option.bind,
    evalOp_constNat]
  apply congrArg some
  apply Tile.ext
  intro idx
  simp only [Tile.bop, Tile.cop, Tile.vec, Broadcast.leftIndex, Broadcast.rightIndex,
    ComparableDType.lt]
  rfl

/-- `tiled_b` masked-load eval: lane `(n,e)` reads
`lora[bpo + n·lora_k_stride + ck e·lora_n_stride]` when both mask legs hold,
else `0`. -/
theorem tiledB_eval (lora_ptr : RegionName)
    (N K lora_k_stride lora_n_stride BLOCK_N BLOCK_K bpo : Nat)
    (ck : Fin BLOCK_K → Nat) (s0 s : BlockState)
    (hbp : s.regs .ptr [] "b_ptr"
      = some (Tile.scalar ((lora_ptr, bpo) : RegionName × Nat)))
    (hon : s.regs .nat [BLOCK_N] "offset_n"
      = some (Tile.vec (fun j : Fin BLOCK_N => j.val)))
    (hck : s.regs .nat [BLOCK_K] "current_k" = some (Tile.vec ck))
    (hmask : s.regs .bool [BLOCK_N, BLOCK_K] "b_ptr_mask"
      = some (⟨fun idx : TileIndex [BLOCK_N, BLOCK_K] =>
          decide (idx.1.val < N) && decide (ck idx.2.1 < K)⟩
            : Tile .bool [BLOCK_N, BLOCK_K]))
    (hrm : ∀ o, s.readMem lora_ptr o = s0.readMem lora_ptr o) :
    evalOp (Op.load .real
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.nil.consL.consR
            (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "b_ptr")
              (Op.mul .nat Broadcast.scalarR
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "offset_n"))
                (Op.constNat lora_k_stride)))
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_K] "current_k"))
              (Op.constNat lora_n_stride))))
        (MaskOpt.maskOther (Op.ref .bool [BLOCK_N, BLOCK_K] "b_ptr_mask")
          ((Op.const 0.0).broadcast [BLOCK_N, BLOCK_K]))) s
      = some (⟨fun idx : TileIndex [BLOCK_N, BLOCK_K] =>
          some (if idx.1.val < N ∧ ck idx.2.1 < K then
            s0.readMem lora_ptr
              (bpo + idx.1.val * lora_k_stride + ck idx.2.1 * lora_n_stride)
          else 0)⟩ : Tile .real [BLOCK_N, BLOCK_K]) := by
  simp only [evalOp, evalOp.eq_def, evalOp_expandDim_one_nat, evalOp_expandDim_zero_nat,
    evalOp_ref, hbp, hon, hck, hmask, Option.bind, Option.map,
    Tile.bop, Tile.cop, Tile.ptrAdd, Tile.scalar, Tile.vec,
    Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul,
    ComparableDType.lt, evalOp_constNat]
  apply congrArg some
  apply Tile.ext
  intro idx
  simp only [Tile.cop, Tile.ptrAdd, Tile.bop, Tile.scalar, Tile.vec,
    Tile.expandDim, TileShape.dropInsertedIndex]
  by_cases hn : idx.1.val < N
  · by_cases hkK : ck idx.2.1 < K
    · simp only [hn, hkK, decide_true, Bool.and_self, if_true, and_true, true_and,
        BlockState.readMemValue_real, hrm]
    · simp only [hn, hkK, decide_true, decide_false, Bool.and_false,
        Bool.false_eq_true, if_false, and_false]
      norm_num
  · simp only [hn, decide_false, Bool.false_and, Bool.false_eq_true, if_false,
      false_and]
    norm_num

/-- `accumulator = accumulator + tl.sum(tiled_a * tiled_b, 1)` eval. -/
theorem accadd_eval (BLOCK_N BLOCK_K : Nat) (za : Tile .real [BLOCK_N])
    (ta : Tile .real [BLOCK_K]) (tb : Tile .real [BLOCK_N, BLOCK_K])
    (s : BlockState)
    (hz : s.regs .real [BLOCK_N] "accumulator" = some za)
    (hta : s.regs .real [BLOCK_K] "tiled_a" = some ta)
    (htb : s.regs .real [BLOCK_N, BLOCK_K] "tiled_b" = some tb) :
    evalOp (Op.add .real Broadcast.nil.consSame
        (Op.ref .real [BLOCK_N] "accumulator")
        (Op.reduceSum ⟨1, by simp⟩ Bool.false
          (Op.mul .real Broadcast.nil.consSame.leadL
            (Op.ref .real [BLOCK_K] "tiled_a")
            (Op.ref .real [BLOCK_N, BLOCK_K] "tiled_b")))) s
      = some (Tile.bop NumericDType.real.add Broadcast.nil.consSame za
          (Tile.reduceSum (shape := [BLOCK_N, BLOCK_K]) ⟨1, by simp⟩ Bool.false
            (Tile.bop NumericDType.real.mul Broadcast.nil.consSame.leadL ta tb))) := by
  have hrs : evalOp (Op.reduceSum (shape := [BLOCK_N, BLOCK_K]) ⟨1, by simp⟩ Bool.false
      (Op.mul .real Broadcast.nil.consSame.leadL
        (Op.ref .real [BLOCK_K] "tiled_a")
        (Op.ref .real [BLOCK_N, BLOCK_K] "tiled_b"))) s
      = some (Tile.reduceSum (shape := [BLOCK_N, BLOCK_K]) ⟨1, by simp⟩ Bool.false
          (Tile.bop NumericDType.real.mul Broadcast.nil.consSame.leadL ta tb)) := by
    simp only [evalOp, evalOp_ref, hta, htb, Option.bind, Option.map]
    rfl
  rw [evalOp_add]
  simp only [evalOp_ref, hz, Option.bind_some, bind, Option.bind_eq_bind]
  erw [hrs]
  rfl

/-- **Per-lane reduction bridge**: the `tl.sum(tiled_a·tiled_b, 1)` of the two
masked loaded tiles at output lane `j` is the masked block contribution — `0`
whenever `j ≥ N` (the fully-masked `tiled_b` row). -/
theorem reduceSum_lane (N K BLOCK_N BLOCK_K : Nat) (ck : Fin BLOCK_K → Nat)
    (av : Nat → ℝ) (bv : Nat → Nat → ℝ) (j : Fin BLOCK_N) :
    (Tile.reduceSum (shape := [BLOCK_N, BLOCK_K]) ⟨1, by simp⟩ Bool.false
      (Tile.bop NumericDType.real.mul Broadcast.nil.consSame.leadL
        (⟨fun e : TileIndex [BLOCK_K] =>
          some (if ck e.1 < K then av (ck e.1) else 0)⟩ : Tile .real [BLOCK_K])
        (⟨fun idx : TileIndex [BLOCK_N, BLOCK_K] =>
          some (if idx.1.val < N ∧ ck idx.2.1 < K then bv idx.1.val (ck idx.2.1)
            else 0)⟩ : Tile .real [BLOCK_N, BLOCK_K]))).data (j, PUnit.unit)
      = some (if j.val < N then
          (Finset.univ : Finset (Fin BLOCK_K)).sum fun e =>
            if ck e < K then av (ck e) * bv j.val (ck e) else 0
        else 0) := by
  simp only [Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
    TileShape.insertAxisIndex, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex]
  have hterm : ∀ e : Fin BLOCK_K,
      NumericDType.real.mul (some (if ck e < K then av (ck e) else 0))
        (some (if j.val < N ∧ ck e < K then bv j.val (ck e) else 0))
      = ((if j.val < N ∧ ck e < K then av (ck e) * bv j.val (ck e) else 0 : ℝ)
          : WithBot ℝ) := by
    intro e
    by_cases hkK : ck e < K
    · by_cases hn : j.val < N
      · simp only [hkK, hn, true_and, and_self, if_true, NumericDType.mul,
          WithBot.realMul, Option.map₂, Option.bind, Option.map]
        rfl
      · simp only [hkK, hn, if_true, false_and, if_false, NumericDType.mul,
          WithBot.realMul, Option.map₂, Option.bind, Option.map]
        norm_num
        rfl
    · simp only [hkK, if_false, and_false, NumericDType.mul,
        WithBot.realMul, Option.map₂, Option.bind, Option.map]
      norm_num
      rfl
  rw [Finset.sum_congr rfl (fun e _ => hterm e), ← WithBot.coe_sum]
  by_cases hn : j.val < N
  · simp only [hn, true_and, if_true]
    rfl
  · simp only [hn, false_and, if_false, Finset.sum_const_zero]
    rfl

set_option maxHeartbeats 4000000 in
/-- **Step**: one K-split loop iteration advances the invariant by one block —
the masked `tl.sum` adds `blockTerm c` to the partial accumulator at every
active lane (and `0` at masked lanes). -/
theorem shrink_step
    (input_ptr lora_ptr : RegionName) (lora_indices : Region .nat)
    (N K xm_stride l0_stride lora_k_stride lora_n_stride
      BLOCK_N BLOCK_K SPLIT_K : Nat) (s0 : BlockState)
    (hS : 0 < BLOCK_K * SPLIT_K)
    (i : Nat) (s : BlockState) (hilt : i < K)
    (hinv : shrinkInv input_ptr lora_ptr lora_indices N K xm_stride l0_stride
        lora_k_stride lora_n_stride BLOCK_N BLOCK_K SPLIT_K s0 i s) :
    ∃ s', stepStmts (shrinkLoopBody N K lora_k_stride lora_n_stride BLOCK_N BLOCK_K)
        (s.setReg "k" .nat [] (Tile.scalar i)) = some s'
      ∧ shrinkInv input_ptr lora_ptr lora_indices N K xm_stride l0_stride
          lora_k_stride lora_n_stride BLOCK_N BLOCK_K SPLIT_K s0
          (i + BLOCK_K * SPLIT_K) s' := by
  obtain ⟨hpids, hi, hbound, hcb, hon, hok, hap, hbp, hacc, hmem⟩ := hinv
  have hc1 : (i + BLOCK_K * SPLIT_K) / (BLOCK_K * SPLIT_K)
      = i / (BLOCK_K * SPLIT_K) + 1 := Nat.add_div_right i hS
  set c := i / (BLOCK_K * SPLIT_K) with hcdef
  set ckF : Fin BLOCK_K → Nat :=
    fun e => i + (e.val + s0.pids 0 * BLOCK_K) with hckF
  set sk := s.setReg "k" .nat [] (Tile.scalar i) with hsk
  have hskpids : sk.pids = s0.pids := by rw [hsk, BlockState.setReg_pids, hpids]
  have hskmem : sk.mem = s0.mem := by rw [hsk]; exact hmem
  have hskrm : ∀ R o, sk.readMem R o = s0.readMem R o := by
    intro R o; unfold BlockState.readMem; rw [hskmem]
  have hk_sk : sk.regs .nat [] "k" = some (Tile.scalar i) := by
    rw [hsk, BlockState.setReg_same]
  have hcb_sk : sk.regs .nat [] "cur_batch" = some (Tile.scalar (s0.pids 1)) := by
    rw [hsk, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("cur_batch":RegName) ≠ "k" by decide)]
    exact hcb
  have hon_sk : sk.regs .nat [BLOCK_N] "offset_n"
      = some (Tile.vec (fun j : Fin BLOCK_N => j.val)) := by
    rw [hsk, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_n":RegName) ≠ "k" by decide)]
    exact hon
  have hok_sk : sk.regs .nat [BLOCK_K] "offset_k"
      = some (Tile.vec (fun e : Fin BLOCK_K => e.val + s0.pids 0 * BLOCK_K)) := by
    rw [hsk, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_k":RegName) ≠ "k" by decide)]
    exact hok
  have hap_sk : sk.regs .ptr [] "a_ptr"
      = some (Tile.scalar ((input_ptr, s0.pids 1 * xm_stride) : RegionName × Nat)) := by
    rw [hsk, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("a_ptr":RegName) ≠ "k" by decide)]
    exact hap
  have hbp_sk : sk.regs .ptr [] "b_ptr"
      = some (Tile.scalar
          ((lora_ptr, l0_stride * loraIdx s0 lora_indices) : RegionName × Nat)) := by
    rw [hsk, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("b_ptr":RegName) ≠ "k" by decide)]
    exact hbp
  have hacc_sk : sk.regs .real [BLOCK_N] "accumulator"
      = some ⟨fun idx : TileIndex [BLOCK_N] =>
          some (if idx.1.val < N then
            accPartial s0 input_ptr lora_ptr lora_indices K xm_stride l0_stride
              lora_k_stride lora_n_stride BLOCK_K SPLIT_K c idx.1.val
          else 0)⟩ := by
    rw [hsk, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("accumulator":RegName) ≠ "k" by decide)]
    exact hacc
  -- step current_k
  unfold shrinkLoopBody
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (currentK_eval BLOCK_K i (fun e : Fin BLOCK_K => e.val + s0.pids 0 * BLOCK_K)
          sk hk_sk hok_sk))]
  set s1 := sk.setReg "current_k" .nat [BLOCK_K] (Tile.vec ckF) with hs1
  have hck1 : s1.regs .nat [BLOCK_K] "current_k" = some (Tile.vec ckF) := by
    rw [hs1, BlockState.setReg_same]
  -- step current_k_c (max_contiguous hint erased: plain copy)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (show evalOp (Op.ref .nat [BLOCK_K] "current_k") s1 = some (Tile.vec ckF) by
          rw [evalOp_ref]; simp [hck1]))]
  set s2 := s1.setReg "current_k_c" .nat [BLOCK_K] (Tile.vec ckF) with hs2
  have hckc2 : s2.regs .nat [BLOCK_K] "current_k_c" = some (Tile.vec ckF) := by
    rw [hs2, BlockState.setReg_same]
  have hck2 : s2.regs .nat [BLOCK_K] "current_k" = some (Tile.vec ckF) := by
    rw [hs2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("current_k":RegName) ≠ "current_k_c" by decide)]
    exact hck1
  have hap2 : s2.regs .ptr [] "a_ptr"
      = some (Tile.scalar ((input_ptr, s0.pids 1 * xm_stride) : RegionName × Nat)) := by
    rw [hs2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("a_ptr":RegName) ≠ "current_k_c" by decide),
      hs1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("a_ptr":RegName) ≠ "current_k" by decide)]
    exact hap_sk
  have hrm2 : ∀ R o, s2.readMem R o = s0.readMem R o := by
    intro R o
    rw [hs2, BlockState.setReg_readMem, hs1, BlockState.setReg_readMem]
    exact hskrm R o
  -- step tiled_a
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (tiledA_eval input_ptr K BLOCK_K (s0.pids 1 * xm_stride) ckF s0 s2
          hap2 hckc2 hck2 (fun o => hrm2 input_ptr o)))]
  set taT : Tile .real [BLOCK_K] :=
    ⟨fun e : TileIndex [BLOCK_K] =>
      some (if ckF e.1 < K then s0.readMem input_ptr (s0.pids 1 * xm_stride + ckF e.1)
        else 0)⟩ with htaT
  set s3 := s2.setReg "tiled_a" .real [BLOCK_K] taT with hs3
  have hon3 : s3.regs .nat [BLOCK_N] "offset_n"
      = some (Tile.vec (fun j : Fin BLOCK_N => j.val)) := by
    rw [hs3, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_n":RegName) ≠ "tiled_a" by decide),
      hs2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_n":RegName) ≠ "current_k_c" by decide),
      hs1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_n":RegName) ≠ "current_k" by decide)]
    exact hon_sk
  have hck3 : s3.regs .nat [BLOCK_K] "current_k" = some (Tile.vec ckF) := by
    rw [hs3, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("current_k":RegName) ≠ "tiled_a" by decide)]
    exact hck2
  -- step b_ptr_mask
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (bmask_eval N K BLOCK_N BLOCK_K ckF s3 hon3 hck3))]
  set bmT : Tile .bool [BLOCK_N, BLOCK_K] :=
    ⟨fun idx : TileIndex [BLOCK_N, BLOCK_K] =>
      decide (idx.1.val < N) && decide (ckF idx.2.1 < K)⟩ with hbmT
  set s4 := s3.setReg "b_ptr_mask" .bool [BLOCK_N, BLOCK_K] bmT with hs4
  have hbp4 : s4.regs .ptr [] "b_ptr"
      = some (Tile.scalar
          ((lora_ptr, l0_stride * loraIdx s0 lora_indices) : RegionName × Nat)) := by
    rw [hs4, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("b_ptr":RegName) ≠ "b_ptr_mask" by decide),
      hs3, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("b_ptr":RegName) ≠ "tiled_a" by decide),
      hs2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("b_ptr":RegName) ≠ "current_k_c" by decide),
      hs1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("b_ptr":RegName) ≠ "current_k" by decide)]
    exact hbp_sk
  have hon4 : s4.regs .nat [BLOCK_N] "offset_n"
      = some (Tile.vec (fun j : Fin BLOCK_N => j.val)) := by
    rw [hs4, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_n":RegName) ≠ "b_ptr_mask" by decide)]
    exact hon3
  have hck4 : s4.regs .nat [BLOCK_K] "current_k" = some (Tile.vec ckF) := by
    rw [hs4, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("current_k":RegName) ≠ "b_ptr_mask" by decide)]
    exact hck3
  have hbm4 : s4.regs .bool [BLOCK_N, BLOCK_K] "b_ptr_mask" = some bmT := by
    rw [hs4, BlockState.setReg_same]
  have hrm4 : ∀ R o, s4.readMem R o = s0.readMem R o := by
    intro R o
    rw [hs4, BlockState.setReg_readMem, hs3, BlockState.setReg_readMem]
    exact hrm2 R o
  -- step tiled_b
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (tiledB_eval lora_ptr N K lora_k_stride lora_n_stride BLOCK_N BLOCK_K
          (l0_stride * loraIdx s0 lora_indices) ckF s0 s4 hbp4 hon4 hck4
          (by rw [hbm4, hbmT]) (fun o => hrm4 lora_ptr o)))]
  set tbT : Tile .real [BLOCK_N, BLOCK_K] :=
    ⟨fun idx : TileIndex [BLOCK_N, BLOCK_K] =>
      some (if idx.1.val < N ∧ ckF idx.2.1 < K then
        s0.readMem lora_ptr
          (l0_stride * loraIdx s0 lora_indices + idx.1.val * lora_k_stride
            + ckF idx.2.1 * lora_n_stride)
      else 0)⟩ with htbT
  set s5 := s4.setReg "tiled_b" .real [BLOCK_N, BLOCK_K] tbT with hs5
  have hta5 : s5.regs .real [BLOCK_K] "tiled_a" = some taT := by
    rw [hs5, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("tiled_a":RegName) ≠ "tiled_b" by decide),
      hs4, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("tiled_a":RegName) ≠ "b_ptr_mask" by decide),
      hs3, BlockState.setReg_same]
  have htb5 : s5.regs .real [BLOCK_N, BLOCK_K] "tiled_b" = some tbT := by
    rw [hs5, BlockState.setReg_same]
  have hacc5 : s5.regs .real [BLOCK_N] "accumulator"
      = some ⟨fun idx : TileIndex [BLOCK_N] =>
          some (if idx.1.val < N then
            accPartial s0 input_ptr lora_ptr lora_indices K xm_stride l0_stride
              lora_k_stride lora_n_stride BLOCK_K SPLIT_K c idx.1.val
          else 0)⟩ := by
    rw [hs5, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("accumulator":RegName) ≠ "tiled_b" by decide),
      hs4, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("accumulator":RegName) ≠ "b_ptr_mask" by decide),
      hs3, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("accumulator":RegName) ≠ "tiled_a" by decide),
      hs2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("accumulator":RegName) ≠ "current_k_c" by decide),
      hs1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("accumulator":RegName) ≠ "current_k" by decide)]
    exact hacc_sk
  -- step accumulator
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (accadd_eval BLOCK_N BLOCK_K _ taT tbT s5 hacc5 hta5 htb5))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  simp only [shrinkInv, hc1]
  refine ⟨?_, by rw [hi]; ring, by omega, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · -- pids
    simp only [hs5, hs4, hs3, hs2, hs1, BlockState.setReg_pids]
    exact hskpids
  · -- cur_batch
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("cur_batch":RegName) ≠ "accumulator" by decide),
      hs5, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("cur_batch":RegName) ≠ "tiled_b" by decide),
      hs4, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("cur_batch":RegName) ≠ "b_ptr_mask" by decide),
      hs3, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("cur_batch":RegName) ≠ "tiled_a" by decide),
      hs2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("cur_batch":RegName) ≠ "current_k_c" by decide),
      hs1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("cur_batch":RegName) ≠ "current_k" by decide)]
    exact hcb_sk
  · -- offset_n
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_n":RegName) ≠ "accumulator" by decide),
      hs5, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_n":RegName) ≠ "tiled_b" by decide)]
    exact hon4
  · -- offset_k
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_k":RegName) ≠ "accumulator" by decide),
      hs5, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_k":RegName) ≠ "tiled_b" by decide),
      hs4, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_k":RegName) ≠ "b_ptr_mask" by decide),
      hs3, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_k":RegName) ≠ "tiled_a" by decide),
      hs2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_k":RegName) ≠ "current_k_c" by decide),
      hs1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_k":RegName) ≠ "current_k" by decide)]
    exact hok_sk
  · -- a_ptr
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("a_ptr":RegName) ≠ "accumulator" by decide),
      hs5, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("a_ptr":RegName) ≠ "tiled_b" by decide),
      hs4, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("a_ptr":RegName) ≠ "b_ptr_mask" by decide),
      hs3, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("a_ptr":RegName) ≠ "tiled_a" by decide)]
    exact hap2
  · -- b_ptr
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("b_ptr":RegName) ≠ "accumulator" by decide),
      hs5, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("b_ptr":RegName) ≠ "tiled_b" by decide)]
    exact hbp4
  · -- accumulator advanced by one block
    rw [BlockState.setReg_same]
    apply congrArg some
    apply Tile.ext
    intro idx
    have hlane := reduceSum_lane N K BLOCK_N BLOCK_K ckF
      (fun k => s0.readMem input_ptr (s0.pids 1 * xm_stride + k))
      (fun n k => s0.readMem lora_ptr
        (l0_stride * loraIdx s0 lora_indices + n * lora_k_stride
          + k * lora_n_stride)) idx.1
    simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex]
    rw [htaT, htbT]
    erw [hlane]
    have hkIdx : ∀ e : Fin BLOCK_K, ckF e = kIdx s0 BLOCK_K SPLIT_K c e.val := by
      intro e
      simp only [hckF, kIdx]
      omega
    by_cases hn : idx.1.val < N
    · simp only [hn, if_true, NumericDType.add, WithBot.realAdd, Option.map₂,
        Option.bind, Option.map]
      apply congrArg some
      rw [accPartial_succ]
      congr 1
      unfold blockTerm
      refine Finset.sum_congr rfl (fun e _ => ?_)
      rw [hkIdx e]
      rfl
    · simp only [hn, if_false, NumericDType.add, WithBot.realAdd, Option.map₂,
        Option.bind, Option.map]
      apply congrArg some
      norm_num
  · -- mem
    show _ = s0.mem
    simp only [hs5, hs4, hs3, hs2, hs1, BlockState.setReg_mem]
    exact hskmem

/-! ## Tail eval helpers -/

/-- `accumulator *= scaling` eval. -/
theorem scale_eval (BLOCK_N : Nat) (scaling : ℝ) (g : Fin BLOCK_N → ℝ)
    (s : BlockState)
    (hacc : s.regs .real [BLOCK_N] "accumulator"
      = some ⟨fun idx : TileIndex [BLOCK_N] => some (g idx.1)⟩) :
    evalOp (Op.mul .real Broadcast.scalarR (Op.ref .real [BLOCK_N] "accumulator")
        (Op.const scaling)) s
      = some (⟨fun idx : TileIndex [BLOCK_N] => some (g idx.1 * scaling)⟩
          : Tile .real [BLOCK_N]) := by
  simp only [evalOp_mul, evalOp_ref, evalOp_const, hacc, Option.bind_some, bind,
    Option.bind_eq_bind]
  apply congrArg some
  apply Tile.ext
  intro idx
  simp only [Tile.bop, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.mul, WithBot.realMul, Option.map₂, Option.bind, Option.map]

/-- `c_ptr = out_ptr + cur_batch·cm_stride + offset_cn·cn_stride` eval. -/
theorem cptr_tail_eval (out_ptr : RegionName) (cm_stride cn_stride BLOCK_N cb : Nat)
    (s : BlockState)
    (hcb : s.regs .nat [] "cur_batch" = some (Tile.scalar cb))
    (hocn : s.regs .nat [BLOCK_N] "offset_cn"
      = some (Tile.vec (fun j : Fin BLOCK_N => j.val))) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase out_ptr)
        (Op.add .nat Broadcast.scalarL
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_batch")
            (Op.constNat cm_stride))
          (Op.mul .nat Broadcast.scalarR (Op.ref .nat [BLOCK_N] "offset_cn")
            (Op.constNat cn_stride)))) s
      = some (⟨fun idx : TileIndex [BLOCK_N] =>
          ((out_ptr, cb * cm_stride + idx.1.val * cn_stride) : RegionName × Nat)⟩
            : Tile .ptr [BLOCK_N]) := by
  rw [evalOp_ptrAdd', evalOp_ptrBase']
  simp only [evalOp_add, evalOp_mul, evalOp_ref, evalOp_constNat, hcb, hocn,
    Option.bind_some, bind, Option.bind_eq_bind]
  apply congrArg some
  apply Tile.ext
  intro idx
  simp only [Tile.ptrAdd, Tile.bop, Tile.scalar, Tile.vec, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add, NumericDType.mul, Region.cast]
  exact Prod.ext rfl (Nat.zero_add _)

/-- `c_mask = offset_cn < N` eval. -/
theorem cmask_eval (N BLOCK_N : Nat) (s : BlockState)
    (hocn : s.regs .nat [BLOCK_N] "offset_cn"
      = some (Tile.vec (fun j : Fin BLOCK_N => j.val))) :
    evalOp (Op.lt .nat Broadcast.scalarR (Op.ref .nat [BLOCK_N] "offset_cn")
        (Op.constNat N)) s
      = some (⟨fun idx : TileIndex [BLOCK_N] => decide (idx.1.val < N)⟩
          : Tile .bool [BLOCK_N]) := by
  simp only [evalOp, evalOp_ref, evalOp_constNat, hocn, Option.bind_some, bind,
    Option.bind_eq_bind, Option.bind]
  apply congrArg some
  apply Tile.ext
  intro idx
  simp only [Tile.cop, Tile.vec, Tile.scalar, Broadcast.leftIndex,
    Broadcast.rightIndex, ComparableDType.lt]
  rfl

set_option maxHeartbeats 2000000 in
/-- **Common tail** (`accumulator *= scaling`, `offset_cn`, `c_ptr`, `c_mask`):
from the invariant at loop exit, the four shared tail statements seed the
store/atomic registers with the scaled partial sums and the masked output
addresses; memory is still untouched. -/
theorem tail_common_steps
    (input_ptr lora_ptr out_ptr : RegionName) (lora_indices : Region .nat)
    (scaling : ℝ)
    (N K xm_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
      BLOCK_N BLOCK_K SPLIT_K : Nat) (s0 st : BlockState) (i : Nat)
    (hinv : shrinkInv input_ptr lora_ptr lora_indices N K xm_stride l0_stride
        lora_k_stride lora_n_stride BLOCK_N BLOCK_K SPLIT_K s0 i st) :
    ∃ st', stepStmts (shrinkTailCommon out_ptr N cm_stride cn_stride BLOCK_N
        scaling) st = some st'
      ∧ st'.regs .ptr [BLOCK_N] "c_ptr"
          = some (⟨fun idx : TileIndex [BLOCK_N] =>
              ((out_ptr, outOff s0 cm_stride cn_stride idx.1.val)
                : RegionName × Nat)⟩ : Tile .ptr [BLOCK_N])
      ∧ st'.regs .bool [BLOCK_N] "c_mask"
          = some (⟨fun idx : TileIndex [BLOCK_N] => decide (idx.1.val < N)⟩
              : Tile .bool [BLOCK_N])
      ∧ st'.regs .real [BLOCK_N] "accumulator"
          = some ⟨fun idx : TileIndex [BLOCK_N] =>
              some ((if idx.1.val < N then
                accPartial s0 input_ptr lora_ptr lora_indices K xm_stride
                  l0_stride lora_k_stride lora_n_stride BLOCK_K SPLIT_K
                  (i / (BLOCK_K * SPLIT_K)) idx.1.val
              else 0) * scaling)⟩
      ∧ st'.mem = s0.mem := by
  obtain ⟨hpids, hi, hbound, hcb, hon, hok, hap, hbp, hacc, hmem⟩ := hinv
  unfold shrinkTailCommon
  -- accumulator *= scaling
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (scale_eval BLOCK_N scaling
          (fun n : Fin BLOCK_N => if n.val < N then
            accPartial s0 input_ptr lora_ptr lora_indices K xm_stride l0_stride
              lora_k_stride lora_n_stride BLOCK_K SPLIT_K
              (i / (BLOCK_K * SPLIT_K)) n.val
          else 0) st (by rw [hacc])))]
  set accST : Tile .real [BLOCK_N] :=
    ⟨fun idx : TileIndex [BLOCK_N] =>
      some ((if idx.1.val < N then
        accPartial s0 input_ptr lora_ptr lora_indices K xm_stride l0_stride
          lora_k_stride lora_n_stride BLOCK_K SPLIT_K
          (i / (BLOCK_K * SPLIT_K)) idx.1.val
      else 0) * scaling)⟩ with haccST
  set t1 := st.setReg "accumulator" .real [BLOCK_N] accST with ht1
  -- offset_cn
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (show evalOp (Op.arange BLOCK_N) t1
            = some (Tile.vec (fun j : Fin BLOCK_N => j.val)) by simp [Tile.vec]))]
  set t2 := t1.setReg "offset_cn" .nat [BLOCK_N]
    (Tile.vec (fun j : Fin BLOCK_N => j.val)) with ht2
  -- c_ptr
  have hcb2 : t2.regs .nat [] "cur_batch" = some (Tile.scalar (s0.pids 1)) := by
    rw [ht2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("cur_batch":RegName) ≠ "offset_cn" by decide),
      ht1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("cur_batch":RegName) ≠ "accumulator" by decide)]
    exact hcb
  have hocn2 : t2.regs .nat [BLOCK_N] "offset_cn"
      = some (Tile.vec (fun j : Fin BLOCK_N => j.val)) := by
    rw [ht2, BlockState.setReg_same]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (cptr_tail_eval out_ptr cm_stride cn_stride BLOCK_N (s0.pids 1) t2
          hcb2 hocn2))]
  set t3 := t2.setReg "c_ptr" .ptr [BLOCK_N]
    (⟨fun idx : TileIndex [BLOCK_N] =>
      ((out_ptr, s0.pids 1 * cm_stride + idx.1.val * cn_stride)
        : RegionName × Nat)⟩ : Tile .ptr [BLOCK_N]) with ht3
  -- c_mask
  have hocn3 : t3.regs .nat [BLOCK_N] "offset_cn"
      = some (Tile.vec (fun j : Fin BLOCK_N => j.val)) := by
    rw [ht3, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_cn":RegName) ≠ "c_ptr" by decide)]
    exact hocn2
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (cmask_eval N BLOCK_N t3 hocn3))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_⟩
  · -- c_ptr
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("c_ptr":RegName) ≠ "c_mask" by decide),
      ht3, BlockState.setReg_same]
    apply congrArg some
    apply Tile.ext
    intro idx
    simp only [outOff]
  · -- c_mask
    rw [BlockState.setReg_same]
  · -- accumulator (scaled)
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("accumulator":RegName) ≠ "c_mask" by decide),
      ht3, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("accumulator":RegName) ≠ "c_ptr" by decide),
      ht2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("accumulator":RegName) ≠ "offset_cn" by decide),
      ht1, BlockState.setReg_same]
  · -- mem
    simp only [BlockState.setReg_mem, ht3, ht2, ht1, BlockState.setReg_mem]
    exact hmem

/-! ## Terminal statements: masked store / masked atomic add -/

/-- The masked store tail reduces to a plain `writeMem` scatter `foldl`. -/
theorem store_tail_step (out_ptr : RegionName) (N BLOCK_N : Nat)
    (offF : Fin BLOCK_N → Nat) (vals : Fin BLOCK_N → ℝ) (s : BlockState)
    (hcp : s.regs .ptr [BLOCK_N] "c_ptr"
      = some (⟨fun idx : TileIndex [BLOCK_N] =>
          ((out_ptr, offF idx.1) : RegionName × Nat)⟩ : Tile .ptr [BLOCK_N]))
    (hcm : s.regs .bool [BLOCK_N] "c_mask"
      = some (⟨fun idx : TileIndex [BLOCK_N] => decide (idx.1.val < N)⟩
          : Tile .bool [BLOCK_N]))
    (hacc : s.regs .real [BLOCK_N] "accumulator"
      = some ⟨fun idx : TileIndex [BLOCK_N] => some (vals idx.1)⟩) :
    stepStmt (shrinkStoreStmt BLOCK_N) s
      = some ((TileShape.allIndices [BLOCK_N]).foldl
          (fun acc (idx : TileIndex [BLOCK_N]) =>
            if decide (idx.1.val < N) then
              acc.writeMem out_ptr (offF idx.1) (vals idx.1)
            else acc) s) := by
  unfold shrinkStoreStmt
  simp only [stepStmt, evalOp_ref, hcp, hcm, hacc, Option.bind, Option.map, bind,
    Option.bind_eq_bind, Option.bind_some]
  apply congrArg some
  apply List.foldl_ext
  intro acc idx _
  simp only [BlockState.writeMemTyped_real, FloatDType.real_storeValue,
    WithBot.unbotD_some]

/-- The masked atomic-add tail reduces to a read-modify-write scatter `foldl`
(the per-cell value is `old + contribution`). -/
theorem atomic_tail_step (out_ptr : RegionName) (N BLOCK_N : Nat)
    (offF : Fin BLOCK_N → Nat) (vals : Fin BLOCK_N → ℝ) (s : BlockState)
    (hcp : s.regs .ptr [BLOCK_N] "c_ptr"
      = some (⟨fun idx : TileIndex [BLOCK_N] =>
          ((out_ptr, offF idx.1) : RegionName × Nat)⟩ : Tile .ptr [BLOCK_N]))
    (hcm : s.regs .bool [BLOCK_N] "c_mask"
      = some (⟨fun idx : TileIndex [BLOCK_N] => decide (idx.1.val < N)⟩
          : Tile .bool [BLOCK_N]))
    (hacc : s.regs .real [BLOCK_N] "accumulator"
      = some ⟨fun idx : TileIndex [BLOCK_N] => some (vals idx.1)⟩) :
    stepStmt (shrinkAtomicStmt BLOCK_N) s
      = some ((TileShape.allIndices [BLOCK_N]).foldl
          (fun acc (idx : TileIndex [BLOCK_N]) =>
            if decide (idx.1.val < N) then
              acc.writeMem out_ptr (offF idx.1)
                (acc.readMem out_ptr (offF idx.1) + vals idx.1)
            else acc) s) := by
  unfold shrinkAtomicStmt
  simp only [stepStmt, evalOp_ref, hcp, hcm, hacc, Option.bind, Option.map, bind,
    Option.bind_eq_bind, Option.bind_some]
  apply congrArg some
  apply List.foldl_ext
  intro acc idx _
  simp only [BlockState.writeMemTyped_real, BlockState.readMemValue_real,
    NumericDType.add, WithBot.realAdd, Option.map₂, Option.bind, Option.map,
    FloatDType.real_storeValue, WithBot.unbotD_some]

/-! ## Exec-level correctness (both tail branches) -/

set_option maxHeartbeats 2000000 in
/-- **`SPLIT_K = 1` branch, exec level**: after the store surface runs, every
active output lane `n < N` holds the genuine scaled rank-slice contraction
`shrinkSpec`. -/
theorem store_exec_correct
    (input_ptr lora_ptr out_ptr : RegionName)
    (N K : Nat) (lora_indices : Region .nat) (scaling : ℝ)
    (xm_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
      BLOCK_N BLOCK_K SPLIT_K : Nat)
    (s s' : BlockState) (hS : 0 < BLOCK_K * SPLIT_K) (hcn : 0 < cn_stride)
    (hExec : exec (bgmv_shrink_store_surface input_ptr lora_ptr out_ptr N K
        lora_indices scaling xm_stride l0_stride lora_k_stride lora_n_stride
        cm_stride cn_stride BLOCK_N BLOCK_K SPLIT_K) s = some s') :
    ∀ n : Fin BLOCK_N, n.val < N →
      s'.readMem out_ptr (outOff s cm_stride cn_stride n.val)
        = shrinkSpec s input_ptr lora_ptr lora_indices scaling K xm_stride
            l0_stride lora_k_stride lora_n_stride BLOCK_K SPLIT_K n.val := by
  have hinj : Function.Injective
      (fun m : Fin BLOCK_N => outOff s cm_stride cn_stride m.val) := by
    have heq : (fun m : Fin BLOCK_N => outOff s cm_stride cn_stride m.val)
        = (fun m : Fin BLOCK_N => s.pids 1 * cm_stride + m.val * cn_stride) := rfl
    rw [heq]
    exact affine1D_inj _ cn_stride hcn
  obtain ⟨sp, hsp, hP0⟩ := shrink_preLoop input_ptr lora_ptr lora_indices N K
    xm_stride l0_stride lora_k_stride lora_n_stride BLOCK_N BLOCK_K SPLIT_K s hS
  obtain ⟨final, sLoop, hLoopStmt, hfinal, hPLoop⟩ :=
    forRange_inv (idx := "k") (start := 0) (stop := K) (step := BLOCK_K * SPLIT_K)
      (Nat.pos_iff_ne_zero.mp hS) hP0
      (fun i st hlt hinv => shrink_step input_ptr lora_ptr lora_indices N K
        xm_stride l0_stride lora_k_stride lora_n_stride BLOCK_N BLOCK_K SPLIT_K
        s hS i st hlt hinv)
  have hiF := hPLoop.2.1
  have hboundF := hPLoop.2.2.1
  have hcF : final / (BLOCK_K * SPLIT_K) = numKIters K (BLOCK_K * SPLIT_K) := by
    apply kloop_final_iters K (BLOCK_K * SPLIT_K) _ hS
    · rw [← hiF]; exact hfinal
    · rw [← hiF]; exact hboundF
  obtain ⟨st', hTail, hcp', hcm', hacc', hmem'⟩ := tail_common_steps input_ptr
    lora_ptr out_ptr lora_indices scaling N K xm_stride l0_stride lora_k_stride
    lora_n_stride cm_stride cn_stride BLOCK_N BLOCK_K SPLIT_K s sLoop final hPLoop
  have hStore := store_tail_step out_ptr N BLOCK_N
    (fun n : Fin BLOCK_N => outOff s cm_stride cn_stride n.val)
    (fun n : Fin BLOCK_N =>
      (if n.val < N then
        accPartial s input_ptr lora_ptr lora_indices K xm_stride l0_stride
          lora_k_stride lora_n_stride BLOCK_K SPLIT_K
          (final / (BLOCK_K * SPLIT_K)) n.val
      else 0) * scaling)
    st' hcp' hcm' hacc'
  have hbody : exec (bgmv_shrink_store_surface input_ptr lora_ptr out_ptr N K
      lora_indices scaling xm_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride BLOCK_N BLOCK_K SPLIT_K) s
      = some ((TileShape.allIndices [BLOCK_N]).foldl
          (fun acc (idx : TileIndex [BLOCK_N]) =>
            if decide (idx.1.val < N) then
              acc.writeMem out_ptr (outOff s cm_stride cn_stride idx.1.val)
                ((if idx.1.val < N then
                  accPartial s input_ptr lora_ptr lora_indices K xm_stride
                    l0_stride lora_k_stride lora_n_stride BLOCK_K SPLIT_K
                    (final / (BLOCK_K * SPLIT_K)) idx.1.val
                else 0) * scaling)
            else acc) st') := by
    rw [exec, store_body_split input_ptr lora_ptr out_ptr N K lora_indices
        scaling xm_stride l0_stride lora_k_stride lora_n_stride cm_stride
        cn_stride BLOCK_N BLOCK_K SPLIT_K,
      store_prologue_split input_ptr lora_ptr out_ptr N K lora_indices scaling
        xm_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
        BLOCK_N BLOCK_K SPLIT_K,
      stepStmts.append_some hsp, stepStmts.cons_some hLoopStmt,
      stepStmts.append_some hTail, stepStmts.cons_some hStore, stepStmts.nil]
  rw [hbody] at hExec
  injection hExec with hExec
  subst hExec
  intro n hn
  rw [foldl_store_at
      (fun idx : TileIndex [BLOCK_N] => outOff s cm_stride cn_stride idx.1.val)
      (fun idx : TileIndex [BLOCK_N] =>
        (if idx.1.val < N then
          accPartial s input_ptr lora_ptr lora_indices K xm_stride l0_stride
            lora_k_stride lora_n_stride BLOCK_K SPLIT_K
            (final / (BLOCK_K * SPLIT_K)) idx.1.val
        else 0) * scaling)
      (fun idx : TileIndex [BLOCK_N] => decide (idx.1.val < N))
      (outOff s cm_stride cn_stride n.val)
      (TileShape.allIndices [BLOCK_N]) st' (n, PUnit.unit)
      (TileShape.mem_allIndices [BLOCK_N] _)
      (decide_eq_true hn) rfl
      (by
        intro b _ hmb hofb
        have hb1 : b.1 = n := hinj hofb
        exact Prod.ext hb1 rfl)
      (TileShape.allIndices_nodup [BLOCK_N])]
  rw [if_pos hn, hcF, shrinkSpec]
  ring

set_option maxHeartbeats 2000000 in
/-- **`SPLIT_K > 1` branch, exec level**: after the atomic surface runs, every
active output lane `n < N` holds **its initial value plus** the genuine scaled
rank-slice contribution — the per-program `tl.atomic_add` obligation. -/
theorem atomic_exec_correct
    (input_ptr lora_ptr out_ptr : RegionName)
    (N K : Nat) (lora_indices : Region .nat) (scaling : ℝ)
    (xm_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
      BLOCK_N BLOCK_K SPLIT_K : Nat)
    (s s' : BlockState) (hS : 0 < BLOCK_K * SPLIT_K) (hcn : 0 < cn_stride)
    (hExec : exec (bgmv_shrink_atomic_surface input_ptr lora_ptr out_ptr N K
        lora_indices scaling xm_stride l0_stride lora_k_stride lora_n_stride
        cm_stride cn_stride BLOCK_N BLOCK_K SPLIT_K) s = some s') :
    ∀ n : Fin BLOCK_N, n.val < N →
      s'.readMem out_ptr (outOff s cm_stride cn_stride n.val)
        = s.readMem out_ptr (outOff s cm_stride cn_stride n.val)
          + shrinkSpec s input_ptr lora_ptr lora_indices scaling K xm_stride
              l0_stride lora_k_stride lora_n_stride BLOCK_K SPLIT_K n.val := by
  have hinj : Function.Injective
      (fun m : Fin BLOCK_N => outOff s cm_stride cn_stride m.val) := by
    have heq : (fun m : Fin BLOCK_N => outOff s cm_stride cn_stride m.val)
        = (fun m : Fin BLOCK_N => s.pids 1 * cm_stride + m.val * cn_stride) := rfl
    rw [heq]
    exact affine1D_inj _ cn_stride hcn
  obtain ⟨sp, hsp, hP0⟩ := shrink_preLoop input_ptr lora_ptr lora_indices N K
    xm_stride l0_stride lora_k_stride lora_n_stride BLOCK_N BLOCK_K SPLIT_K s hS
  obtain ⟨final, sLoop, hLoopStmt, hfinal, hPLoop⟩ :=
    forRange_inv (idx := "k") (start := 0) (stop := K) (step := BLOCK_K * SPLIT_K)
      (Nat.pos_iff_ne_zero.mp hS) hP0
      (fun i st hlt hinv => shrink_step input_ptr lora_ptr lora_indices N K
        xm_stride l0_stride lora_k_stride lora_n_stride BLOCK_N BLOCK_K SPLIT_K
        s hS i st hlt hinv)
  have hiF := hPLoop.2.1
  have hboundF := hPLoop.2.2.1
  have hcF : final / (BLOCK_K * SPLIT_K) = numKIters K (BLOCK_K * SPLIT_K) := by
    apply kloop_final_iters K (BLOCK_K * SPLIT_K) _ hS
    · rw [← hiF]; exact hfinal
    · rw [← hiF]; exact hboundF
  obtain ⟨st', hTail, hcp', hcm', hacc', hmem'⟩ := tail_common_steps input_ptr
    lora_ptr out_ptr lora_indices scaling N K xm_stride l0_stride lora_k_stride
    lora_n_stride cm_stride cn_stride BLOCK_N BLOCK_K SPLIT_K s sLoop final hPLoop
  have hAtomic := atomic_tail_step out_ptr N BLOCK_N
    (fun n : Fin BLOCK_N => outOff s cm_stride cn_stride n.val)
    (fun n : Fin BLOCK_N =>
      (if n.val < N then
        accPartial s input_ptr lora_ptr lora_indices K xm_stride l0_stride
          lora_k_stride lora_n_stride BLOCK_K SPLIT_K
          (final / (BLOCK_K * SPLIT_K)) n.val
      else 0) * scaling)
    st' hcp' hcm' hacc'
  have hbody : exec (bgmv_shrink_atomic_surface input_ptr lora_ptr out_ptr N K
      lora_indices scaling xm_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride BLOCK_N BLOCK_K SPLIT_K) s
      = some ((TileShape.allIndices [BLOCK_N]).foldl
          (fun acc (idx : TileIndex [BLOCK_N]) =>
            if decide (idx.1.val < N) then
              acc.writeMem out_ptr (outOff s cm_stride cn_stride idx.1.val)
                (acc.readMem out_ptr (outOff s cm_stride cn_stride idx.1.val)
                  + (if idx.1.val < N then
                      accPartial s input_ptr lora_ptr lora_indices K xm_stride
                        l0_stride lora_k_stride lora_n_stride BLOCK_K SPLIT_K
                        (final / (BLOCK_K * SPLIT_K)) idx.1.val
                    else 0) * scaling)
            else acc) st') := by
    rw [exec, atomic_body_split input_ptr lora_ptr out_ptr N K lora_indices
        scaling xm_stride l0_stride lora_k_stride lora_n_stride cm_stride
        cn_stride BLOCK_N BLOCK_K SPLIT_K,
      atomic_prologue_split input_ptr lora_ptr out_ptr N K lora_indices scaling
        xm_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
        BLOCK_N BLOCK_K SPLIT_K,
      stepStmts.append_some hsp, stepStmts.cons_some hLoopStmt,
      stepStmts.append_some hTail, stepStmts.cons_some hAtomic, stepStmts.nil]
  rw [hbody] at hExec
  injection hExec with hExec
  subst hExec
  intro n hn
  rw [foldl_atomicAdd_at
      (fun idx : TileIndex [BLOCK_N] => outOff s cm_stride cn_stride idx.1.val)
      (fun idx : TileIndex [BLOCK_N] =>
        (if idx.1.val < N then
          accPartial s input_ptr lora_ptr lora_indices K xm_stride l0_stride
            lora_k_stride lora_n_stride BLOCK_K SPLIT_K
            (final / (BLOCK_K * SPLIT_K)) idx.1.val
        else 0) * scaling)
      (fun idx : TileIndex [BLOCK_N] => decide (idx.1.val < N))
      (outOff s cm_stride cn_stride n.val)
      (TileShape.allIndices [BLOCK_N]) st' (n, PUnit.unit)
      (TileShape.mem_allIndices [BLOCK_N] _)
      (decide_eq_true hn) rfl
      (by
        intro b _ hmb hofb
        have hb1 : b.1 = n := hinj hofb
        exact Prod.ext hb1 rfl)
      (TileShape.allIndices_nodup [BLOCK_N])]
  have hstrm : st'.readMem out_ptr (outOff s cm_stride cn_stride n.val)
      = s.readMem out_ptr (outOff s cm_stride cn_stride n.val) := by
    unfold BlockState.readMem
    rw [hmem']
  rw [hstrm, if_pos hn, hcF, shrinkSpec]
  ring_nf

/-! ## Realizes_without_Rounding summaries -/

/-- **`SPLIT_K = 1` branch (masked store)**: the store surface realizes the
genuine scaled rank-slice contraction `shrinkSpec` at every active output lane
`n < N`. -/
theorem store_compute_correct
    (input_ptr lora_ptr out_ptr : RegionName)
    (N K : Nat) (lora_indices : Region .nat) (scaling : ℝ)
    (xm_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
      BLOCK_N BLOCK_K SPLIT_K : Nat)
    (s : BlockState) (hS : 0 < BLOCK_K * SPLIT_K) (hcn : 0 < cn_stride) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := bgmv_shrink_store_surface input_ptr lora_ptr out_ptr N K
        lora_indices scaling xm_stride l0_stride lora_k_stride lora_n_stride
        cm_stride cn_stride BLOCK_N BLOCK_K SPLIT_K)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun n : Fin BLOCK_N => n.val < N)
        (fun n => (out_ptr, outOff s cm_stride cn_stride n.val)))
      (expected := fun n : Fin BLOCK_N =>
        shrinkSpec s input_ptr lora_ptr lora_indices scaling K xm_stride
          l0_stride lora_k_stride lora_n_stride BLOCK_K SPLIT_K n.val) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [bgmv_shrink_store_surface, ComputeStmt.listToAlgorithm?,
      ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
      Except.bind, bind]
  intro s0 s' hExec hs0
  subst hs0
  intro n hn
  exact store_exec_correct input_ptr lora_ptr out_ptr N K lora_indices scaling
    xm_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
    BLOCK_N BLOCK_K SPLIT_K s0 s' hS hcn hExec n hn

/-- **`SPLIT_K > 1` branch (masked `tl.atomic_add`)**: the atomic surface
realizes `out-before + shrinkSpec` at every active output lane `n < N` — the
per-program atomic accumulation obligation (the cross-program sum over
`pid_sk` is the host's trusted composition). -/
theorem atomic_compute_correct
    (input_ptr lora_ptr out_ptr : RegionName)
    (N K : Nat) (lora_indices : Region .nat) (scaling : ℝ)
    (xm_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
      BLOCK_N BLOCK_K SPLIT_K : Nat)
    (s : BlockState) (hS : 0 < BLOCK_K * SPLIT_K) (hcn : 0 < cn_stride) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := bgmv_shrink_atomic_surface input_ptr lora_ptr out_ptr N K
        lora_indices scaling xm_stride l0_stride lora_k_stride lora_n_stride
        cm_stride cn_stride BLOCK_N BLOCK_K SPLIT_K)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun n : Fin BLOCK_N => n.val < N)
        (fun n => (out_ptr, outOff s cm_stride cn_stride n.val)))
      (expected := fun n : Fin BLOCK_N =>
        s.readMem out_ptr (outOff s cm_stride cn_stride n.val)
          + shrinkSpec s input_ptr lora_ptr lora_indices scaling K xm_stride
              l0_stride lora_k_stride lora_n_stride BLOCK_K SPLIT_K n.val) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [bgmv_shrink_atomic_surface, ComputeStmt.listToAlgorithm?,
      ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
      Except.bind, bind]
  intro s0 s' hExec hs0
  subst hs0
  intro n hn
  exact atomic_exec_correct input_ptr lora_ptr out_ptr N K lora_indices scaling
    xm_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
    BLOCK_N BLOCK_K SPLIT_K s0 s' hS hcn hExec n hn

/-! ## Sentinel early-return path -/

/-- The faithful surface's body: three prologue assigns followed by the
sentinel guard around the whole active body. -/
theorem surface_body_shape
    (input_ptr lora_ptr out_ptr : RegionName)
    (N K : Nat) (lora_indices : Region .int) (scaling : ℝ)
    (xm_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
      BLOCK_N BLOCK_K SPLIT_K : Nat) (SPLIT_K_ONE : Bool) :
    ∃ guardBody, (bgmv_shrink_surface input_ptr lora_ptr out_ptr N K
        lora_indices scaling xm_stride l0_stride lora_k_stride lora_n_stride
        cm_stride cn_stride BLOCK_N BLOCK_K SPLIT_K SPLIT_K_ONE).toAlgKernel.body
      = [ Stmt.assign .nat [] "pid_sk" (Op.programId 0),
          Stmt.assign .nat [] "cur_batch" (Op.programId 1),
          Stmt.assign .int [] "lora_index"
            (Op.load .int (MemAccess.region lora_indices
              (Op.ref .nat [] "cur_batch")) MaskOpt.none),
          Stmt.ifThen
            (Op.ne .int Broadcast.nil (Op.ref .int [] "lora_index")
              (Op.constInt (-1)))
            guardBody ] :=
  ⟨_, rfl⟩

set_option maxHeartbeats 2000000 in
/-- **Sentinel skip is write-free**: when `lora_indices[cur_batch] = -1`
(Python's early `return`), the faithful guarded surface leaves memory
untouched. -/
theorem bgmv_shrink_sentinel_skip_no_write
    (input_ptr lora_ptr out_ptr : RegionName)
    (N K : Nat) (lora_indices : Region .int) (scaling : ℝ)
    (xm_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
      BLOCK_N BLOCK_K SPLIT_K : Nat) (SPLIT_K_ONE : Bool)
    (s s' : BlockState)
    (hSent : s.readMemValue .int lora_indices (s.pids 1) = (-1 : Int))
    (hExec : exec (bgmv_shrink_surface input_ptr lora_ptr out_ptr N K
        lora_indices scaling xm_stride l0_stride lora_k_stride lora_n_stride
        cm_stride cn_stride BLOCK_N BLOCK_K SPLIT_K SPLIT_K_ONE) s = some s') :
    s'.mem = s.mem := by
  obtain ⟨guardBody, hbody⟩ := surface_body_shape input_ptr lora_ptr out_ptr
    N K lora_indices scaling xm_stride l0_stride lora_k_stride lora_n_stride
    cm_stride cn_stride BLOCK_N BLOCK_K SPLIT_K SPLIT_K_ONE
  rw [exec, hbody] at hExec
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (show evalOp (Op.programId 0) s = some (Tile.scalar (s.pids 0)) by
          simp))] at hExec
  set s1 := s.setReg "pid_sk" .nat [] (Tile.scalar (s.pids 0)) with hs1
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (show evalOp (Op.programId 1) s1 = some (Tile.scalar (s.pids 1)) by
          simp [hs1]))] at hExec
  set s2 := s1.setReg "cur_batch" .nat [] (Tile.scalar (s.pids 1)) with hs2
  have hli2 : evalOp (Op.load .int
      (MemAccess.region lora_indices (Op.ref .nat [] "cur_batch")) MaskOpt.none) s2
      = some (Tile.scalar (-1 : Int)) := by
    rw [← hSent]
    simp only [evalOp, evalOp_ref, hs2, BlockState.setReg_same, Option.bind,
      BlockState.setReg_pids, BlockState.setReg_readMemValue, Tile.scalar]
    rfl
  rw [stepStmts.cons_some (stepStmt_assign_eq_some hli2)] at hExec
  set s3 := s2.setReg "lora_index" .int [] (Tile.scalar (-1 : Int)) with hs3
  have hli3 : s3.regs .int [] "lora_index" = some (Tile.scalar (-1 : Int)) := by
    rw [hs3, BlockState.setReg_same]
  have hcond : evalOp (Op.ne .int Broadcast.nil (Op.ref .int [] "lora_index")
      (Op.constInt (-1))) s3
      = some (⟨fun _ => Bool.false⟩ : Tile .bool []) := by
    simp only [evalOp, evalOp_ref, hli3, Option.bind, Option.bind_some, bind,
      Option.bind_eq_bind]
    apply congrArg some
    apply Tile.ext
    intro _
    simp only [Tile.cop, Tile.bop, Tile.scalar, Broadcast.leftIndex,
      Broadcast.rightIndex, ComparableDType.ne]
    simp
  have hguard : stepStmt (Stmt.ifThen
      (Op.ne .int Broadcast.nil (Op.ref .int [] "lora_index") (Op.constInt (-1)))
      guardBody) s3 = some s3 := by
    unfold stepStmt
    rw [hcond]
    simp
  rw [stepStmts.cons_some hguard, stepStmts.nil] at hExec
  injection hExec with hExec
  subst hExec
  rw [hs3, hs2, hs1]
  rfl

/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/

/-- **Dimension-general** correctness summary for `bgmv_shrink_kernel.py`'s
`_bgmv_shrink_kernel`, against the **genuine closed form**

```
shrinkSpec n = scaling · Σ_{c<⌈K/(BLOCK_K·SPLIT_K)⌉} Σ_{e<BLOCK_K}
  [k(c,e) < K] · input[cur_batch·xm_stride + k(c,e)]
             · loraA[l0_stride·lora_index + n·lora_k_stride + k(c,e)·lora_n_stride]
```

(`k(c,e) = c·BLOCK_K·SPLIT_K + pid_sk·BLOCK_K + e`, this program's rank
slice) — a pure function of INPUT memory, never a read-back of the kernel's
own output — for arbitrary `N`, `K`, `BLOCK_N`, `BLOCK_K`, `SPLIT_K`, strides,
`scaling`, program ids, and data-dependent `lora_index`. It packages:

* all three surfaces lower to the algorithm layer (the faithful guarded
  surface with both constexpr tail branches, and the two per-branch proof
  surfaces);
* the sentinel path: `lora_indices[cur_batch] = -1` (Python's early `return`)
  leaves memory untouched;
* the `SPLIT_K = 1` branch (`tl.store`): every active lane `n < N` of
  `out[cur_batch]` holds `shrinkSpec n`;
* the `SPLIT_K > 1` branch (`tl.atomic_add`): every active lane holds
  `out-before + shrinkSpec n` — the per-program atomic accumulation
  obligation; the cross-program sum over `pid_sk` (and the sentinel skip
  itself for the proof surfaces) is the host launch's trusted composition.

Honest side-conditions: `0 < BLOCK_K`, `0 < SPLIT_K` (a launched constexpr
tile is nonempty — also the K-loop stride), and `0 < cn_stride` (output-lane
footprint injectivity; torch strides of a non-degenerate output are ≥ 1). -/
specification bgmv_shrink_kernel_output_summary_general
    (input_ptr lora_ptr out_ptr : RegionName)
    (N K : Nat) (lora_indices_int : Region .int) (lora_indices : Region .nat)
    (scaling : ℝ)
    (xm_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
      BLOCK_N BLOCK_K SPLIT_K : Nat) (SPLIT_K_ONE : Bool)
    (s : BlockState)
    (hBK : 0 < BLOCK_K) (hSK : 0 < SPLIT_K) (hcn : 0 < cn_stride) :
    -- (1) the faithful guarded surface (sentinel guard + both constexpr tail
    --     branches) lowers to the algorithm layer
    (∃ alg, (bgmv_shrink_surface input_ptr lora_ptr out_ptr N K
      lora_indices_int scaling xm_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride BLOCK_N BLOCK_K SPLIT_K SPLIT_K_ONE).toAlgorithm?
        = Except.ok alg) ∧
    -- (2) both per-branch proof surfaces lower to the algorithm layer
    (∃ alg, (bgmv_shrink_store_surface input_ptr lora_ptr out_ptr N K
      lora_indices scaling xm_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride BLOCK_N BLOCK_K SPLIT_K).toAlgorithm?
        = Except.ok alg) ∧
    (∃ alg, (bgmv_shrink_atomic_surface input_ptr lora_ptr out_ptr N K
      lora_indices scaling xm_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride BLOCK_N BLOCK_K SPLIT_K).toAlgorithm?
        = Except.ok alg) ∧
    -- (3) the sentinel early-return path writes nothing
    (∀ s', s.readMemValue .int lora_indices_int (s.pids 1) = (-1 : Int) →
      exec (bgmv_shrink_surface input_ptr lora_ptr out_ptr N K lora_indices_int
        scaling xm_stride l0_stride lora_k_stride lora_n_stride cm_stride
        cn_stride BLOCK_N BLOCK_K SPLIT_K SPLIT_K_ONE) s = some s' →
      s'.mem = s.mem) ∧
    -- (4) SPLIT_K = 1: masked store of the genuine contraction
    ComputeCorrect.Realizes_without_Rounding
      (kernel := bgmv_shrink_store_surface input_ptr lora_ptr out_ptr N K
        lora_indices scaling xm_stride l0_stride lora_k_stride lora_n_stride
        cm_stride cn_stride BLOCK_N BLOCK_K SPLIT_K)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun n : Fin BLOCK_N => n.val < N)
        (fun n => (out_ptr, outOff s cm_stride cn_stride n.val)))
      (expected := fun n : Fin BLOCK_N =>
        shrinkSpec s input_ptr lora_ptr lora_indices scaling K xm_stride
          l0_stride lora_k_stride lora_n_stride BLOCK_K SPLIT_K n.val) ∧
    -- (5) SPLIT_K > 1: masked atomic add of the genuine contraction
    ComputeCorrect.Realizes_without_Rounding
      (kernel := bgmv_shrink_atomic_surface input_ptr lora_ptr out_ptr N K
        lora_indices scaling xm_stride l0_stride lora_k_stride lora_n_stride
        cm_stride cn_stride BLOCK_N BLOCK_K SPLIT_K)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun n : Fin BLOCK_N => n.val < N)
        (fun n => (out_ptr, outOff s cm_stride cn_stride n.val)))
      (expected := fun n : Fin BLOCK_N =>
        s.readMem out_ptr (outOff s cm_stride cn_stride n.val)
          + shrinkSpec s input_ptr lora_ptr lora_indices scaling K xm_stride
              l0_stride lora_k_stride lora_n_stride BLOCK_K SPLIT_K n.val) :=
  ⟨bgmv_shrink_surface_toAlgorithm_supported input_ptr lora_ptr out_ptr N K
      lora_indices_int scaling xm_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride BLOCK_N BLOCK_K SPLIT_K SPLIT_K_ONE,
    bgmv_shrink_store_surface_toAlgorithm_supported input_ptr lora_ptr out_ptr
      N K lora_indices scaling xm_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride BLOCK_N BLOCK_K SPLIT_K,
    bgmv_shrink_atomic_surface_toAlgorithm_supported input_ptr lora_ptr out_ptr
      N K lora_indices scaling xm_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride BLOCK_N BLOCK_K SPLIT_K,
    fun s' hSent hExec => bgmv_shrink_sentinel_skip_no_write input_ptr lora_ptr
      out_ptr N K lora_indices_int scaling xm_stride l0_stride lora_k_stride
      lora_n_stride cm_stride cn_stride BLOCK_N BLOCK_K SPLIT_K SPLIT_K_ONE
      s s' hSent hExec,
    store_compute_correct input_ptr lora_ptr out_ptr N K lora_indices scaling
      xm_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
      BLOCK_N BLOCK_K SPLIT_K s (Nat.mul_pos hBK hSK) hcn,
    atomic_compute_correct input_ptr lora_ptr out_ptr N K lora_indices scaling
      xm_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
      BLOCK_N BLOCK_K SPLIT_K s (Nat.mul_pos hBK hSK) hcn⟩

/-! ## The `⊨[R]` streaming metadata headline (store face)

Everything below is purely additive; the exact surfaces and the main theorem
above are untouched. The `SPLIT_K = 1` store face is restated on the
metadata-slot streaming skin `StreamMetaMasked3DKernelIO₂`: one `.nat` slot
(`lora_indices[cur_batch]`, read at the pid-only cell `cur_batch`), two
masked streamed inputs with `other=0` (the `tiled_a` row tile and the
`tiled_b` weight tile, both gated by the genuine `current_k < K` tail mask),
and one masked terminal `.real` store. This kernel has **zero rounding
events** — every load, the whole K-loop arithmetic and the terminal store
are at `.real` — so under `execR R` the prologue, the loop and the common
tail collapse verbatim onto the exact stepper and the proven
`shrink_preLoop` / `shrink_step` / `forRange_inv` / `tail_common_steps`
stack is reused unchanged; only the single masked store statement is
re-proved on the `R` side, where the `.real` write coincides with the
`writeMemAsR R .real` rounded write (`RoundingModel.storeValue_real`) and
the readback contract's `R.round .real` is the identity (`round_real`) —
the `⊨[R]` face at the `.real` grid is the exact streaming contract, stated
once for every `R`.

The **atomic face** (`SPLIT_K > 1`, `tl.atomic_add`) is a read-modify-write,
not a terminal single-store fold, so it is *not* expressible in this skin's
S1 genre; its per-program `out-before + contribution` obligation stays with
the existing five-conjunct exact summary
`bgmv_shrink_kernel_output_summary_general` above (conjunct (5)). Likewise
the `-1` skip sentinel lives on the faithful `.int` surface
(`bgmv_shrink_sentinel_skip_no_write`, conjunct (3)); the store face below
follows the store proof surface's `Region .nat` metadata convention, the
sentinel skip being the trusted host boundary. -/

open scoped VeriTile.Triton.StreamMetaMasked3DKernelIO₂

/-! ### Cast-free collapses -/

/-- The 8-statement prologue is cast-free (register assigns and one `.nat`
metadata load — loads are not rounding events): it steps identically under
`stepStmtsR R`. -/
private theorem shrinkPrologue_castFree (R : RoundingModel)
    (input_ptr lora_ptr : RegionName) (lora_indices : Region .nat)
    (xm_stride l0_stride BLOCK_N BLOCK_K : Nat) (t : BlockState) :
    stepStmtsR R (shrinkPrologue input_ptr lora_ptr lora_indices xm_stride
        l0_stride BLOCK_N BLOCK_K) t
      = stepStmts (shrinkPrologue input_ptr lora_ptr lora_indices xm_stride
          l0_stride BLOCK_N BLOCK_K) t := by
  simp only [shrinkPrologue, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def]
  rfl

/-- The K-loop body is cast-free (two masked `.real` loads with `other=0`,
`tl.sum` + add, nat index arithmetic — no `castFloat`, no store): it steps
identically under `stepStmtsR R`, so the exact invariant stack transports to
`execR`. -/
private theorem shrinkBody_castFree (R : RoundingModel)
    (N K lora_k_stride lora_n_stride BLOCK_N BLOCK_K : Nat) (t : BlockState) :
    stepStmtsR R (shrinkLoopBody N K lora_k_stride lora_n_stride BLOCK_N
        BLOCK_K) t
      = stepStmts (shrinkLoopBody N K lora_k_stride lora_n_stride BLOCK_N
          BLOCK_K) t := by
  simp only [shrinkLoopBody, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def]
  rfl

/-- The 4-statement common tail (`accumulator *= scaling`, `offset_cn`,
`c_ptr`, `c_mask`) is cast-free. -/
private theorem shrinkTailCommon_castFree (R : RoundingModel)
    (out_ptr : RegionName) (N cm_stride cn_stride BLOCK_N : Nat) (scaling : ℝ)
    (t : BlockState) :
    stepStmtsR R (shrinkTailCommon out_ptr N cm_stride cn_stride BLOCK_N
        scaling) t
      = stepStmts (shrinkTailCommon out_ptr N cm_stride cn_stride BLOCK_N
          scaling) t := by
  simp only [shrinkTailCommon, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def]
  rfl

/-! ### `evalOpR` collapses and eval helpers for the safety walk

The safety walk (`hts`) quantifies over arbitrary launch states, so it
re-evaluates the address/mask registers under `evalOpR R`. Every such op is
`R`-free (pure nat/bool/pointer arithmetic), so each collapses onto `evalOp`
and reuses the exact eval lemmas above. -/

private theorem evalOpR_programId (R : RoundingModel) (ax : Nat)
    (s : BlockState) :
    evalOpR R (Op.programId ax) s = some (Tile.scalar (s.pids ax)) := by
  simp only [evalOpR]

private theorem evalOpR_arange (R : RoundingModel) (n : Nat) (s : BlockState) :
    evalOpR R (Op.arange n) s
      = some (Tile.vec (fun j : Fin n => j.val)) := by
  simp only [evalOpR, Tile.vec]

/-- `current_k = k + offset_k` is `R`-free. -/
private theorem evalR_currentK (R : RoundingModel) (BLOCK_K : Nat)
    (s : BlockState) :
    evalOpR R (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "k")
        (Op.ref .nat [BLOCK_K] "offset_k")) s
      = evalOp (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "k")
          (Op.ref .nat [BLOCK_K] "offset_k")) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

/-- The `tiled_a` load mask `current_k < K` is `R`-free. -/
private theorem evalR_maskA (R : RoundingModel) (K BLOCK_K : Nat)
    (s : BlockState) :
    evalOpR R (Op.lt .nat Broadcast.scalarR (Op.ref .nat [BLOCK_K] "current_k")
        (Op.constNat K)) s
      = evalOp (Op.lt .nat Broadcast.scalarR (Op.ref .nat [BLOCK_K] "current_k")
          (Op.constNat K)) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

/-- The `tiled_a` pointer `a_ptr + current_k_c` is `R`-free. -/
private theorem evalR_aTilePtr (R : RoundingModel) (BLOCK_K : Nat)
    (s : BlockState) :
    evalOpR R (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "a_ptr")
        (Op.ref .nat [BLOCK_K] "current_k_c")) s
      = evalOp (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "a_ptr")
          (Op.ref .nat [BLOCK_K] "current_k_c")) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

/-- The `b_ptr_mask` op is `R`-free. -/
private theorem evalR_bmask (R : RoundingModel) (N K BLOCK_N BLOCK_K : Nat)
    (s : BlockState) :
    evalOpR R (Op.boolAnd Broadcast.nil.consL.consR
        (Op.lt .nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "offset_n"))
          (Op.constNat N))
        (Op.lt .nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_K] "current_k"))
          (Op.constNat K))) s
      = evalOp (Op.boolAnd Broadcast.nil.consL.consR
          (Op.lt .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "offset_n"))
            (Op.constNat N))
          (Op.lt .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_K] "current_k"))
            (Op.constNat K))) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

/-- The `tiled_b` pointer op is `R`-free. -/
private theorem evalR_bTilePtr (R : RoundingModel)
    (lora_k_stride lora_n_stride BLOCK_N BLOCK_K : Nat) (s : BlockState) :
    evalOpR R (Op.ptrAdd Broadcast.nil.consL.consR
        (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "b_ptr")
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "offset_n"))
            (Op.constNat lora_k_stride)))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_K] "current_k"))
          (Op.constNat lora_n_stride))) s
      = evalOp (Op.ptrAdd Broadcast.nil.consL.consR
          (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "b_ptr")
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "offset_n"))
              (Op.constNat lora_k_stride)))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_K] "current_k"))
            (Op.constNat lora_n_stride))) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

/-- The `c_ptr` tail op is `R`-free. -/
private theorem evalR_cptr (R : RoundingModel) (out_ptr : RegionName)
    (cm_stride cn_stride BLOCK_N : Nat) (s : BlockState) :
    evalOpR R (Op.ptrAdd Broadcast.scalarL (Op.ptrBase out_ptr)
        (Op.add .nat Broadcast.scalarL
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_batch")
            (Op.constNat cm_stride))
          (Op.mul .nat Broadcast.scalarR (Op.ref .nat [BLOCK_N] "offset_cn")
            (Op.constNat cn_stride)))) s
      = evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase out_ptr)
          (Op.add .nat Broadcast.scalarL
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_batch")
              (Op.constNat cm_stride))
            (Op.mul .nat Broadcast.scalarR (Op.ref .nat [BLOCK_N] "offset_cn")
              (Op.constNat cn_stride)))) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

/-- The `c_mask` tail op is `R`-free. -/
private theorem evalR_cmask (R : RoundingModel) (N BLOCK_N : Nat)
    (s : BlockState) :
    evalOpR R (Op.lt .nat Broadcast.scalarR (Op.ref .nat [BLOCK_N] "offset_cn")
        (Op.constNat N)) s
      = evalOp (Op.lt .nat Broadcast.scalarR (Op.ref .nat [BLOCK_N] "offset_cn")
          (Op.constNat N)) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

/-- `current_k < K` mask eval (the 1-D sibling of `cmask_eval`). -/
private theorem maskA_eval (K BLOCK_K : Nat) (ck : Fin BLOCK_K → Nat)
    (s : BlockState)
    (hck : s.regs .nat [BLOCK_K] "current_k" = some (Tile.vec ck)) :
    evalOp (Op.lt .nat Broadcast.scalarR (Op.ref .nat [BLOCK_K] "current_k")
        (Op.constNat K)) s
      = some (⟨fun e : TileIndex [BLOCK_K] => decide (ck e.1 < K)⟩
          : Tile .bool [BLOCK_K]) := by
  simp only [evalOp, evalOp_ref, evalOp_constNat, hck, Option.bind_some, bind,
    Option.bind_eq_bind, Option.bind]
  apply congrArg some
  apply Tile.ext
  intro e
  simp only [Tile.cop, Tile.vec, Tile.scalar, Broadcast.leftIndex,
    Broadcast.rightIndex, ComparableDType.lt]
  rfl

/-- `a_ptr + current_k_c` pointer-tile eval. -/
private theorem aTilePtr_eval (input_ptr : RegionName) (BLOCK_K apo : Nat)
    (ck : Fin BLOCK_K → Nat) (s : BlockState)
    (hap : s.regs .ptr [] "a_ptr"
      = some (Tile.scalar ((input_ptr, apo) : RegionName × Nat)))
    (hckc : s.regs .nat [BLOCK_K] "current_k_c" = some (Tile.vec ck)) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "a_ptr")
        (Op.ref .nat [BLOCK_K] "current_k_c")) s
      = some (⟨fun e : TileIndex [BLOCK_K] =>
          ((input_ptr, apo + ck e.1) : RegionName × Nat)⟩
            : Tile .ptr [BLOCK_K]) := by
  rw [evalOp_ptrAdd']
  simp only [evalOp_ref, hap, hckc, Option.bind_some, bind, Option.bind_eq_bind]
  apply congrArg some
  apply Tile.ext
  intro e
  simp only [Tile.ptrAdd, Tile.bop, Tile.scalar, Tile.vec, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add]

/-- `tiled_b` pointer-tile eval: cell `(n, e)` addresses
`bpo + n·lora_k_stride + ck e·lora_n_stride` in the LoRA buffer. -/
private theorem bTilePtr_eval (lora_ptr : RegionName)
    (lora_k_stride lora_n_stride BLOCK_N BLOCK_K bpo : Nat)
    (ck : Fin BLOCK_K → Nat) (s : BlockState)
    (hbp : s.regs .ptr [] "b_ptr"
      = some (Tile.scalar ((lora_ptr, bpo) : RegionName × Nat)))
    (hon : s.regs .nat [BLOCK_N] "offset_n"
      = some (Tile.vec (fun j : Fin BLOCK_N => j.val)))
    (hck : s.regs .nat [BLOCK_K] "current_k" = some (Tile.vec ck)) :
    evalOp (Op.ptrAdd Broadcast.nil.consL.consR
        (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "b_ptr")
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "offset_n"))
            (Op.constNat lora_k_stride)))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_K] "current_k"))
          (Op.constNat lora_n_stride))) s
      = some (⟨fun idx : TileIndex [BLOCK_N, BLOCK_K] =>
          ((lora_ptr, bpo + idx.1.val * lora_k_stride
              + ck idx.2.1 * lora_n_stride) : RegionName × Nat)⟩
            : Tile .ptr [BLOCK_N, BLOCK_K]) := by
  simp only [evalOp, evalOp.eq_def, evalOp_expandDim_one_nat,
    evalOp_expandDim_zero_nat, evalOp_ref, evalOp_constNat, hbp, hon, hck,
    Option.bind, Option.map, Option.bind_some, bind, Option.bind_eq_bind]
  apply congrArg some
  apply Tile.ext
  intro idx
  simp only [Tile.ptrAdd, Tile.bop, Tile.scalar, Tile.vec, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.mul, NumericDType.add]
  rfl

/-! ### The masked `.real` store under `execR R` -/

/-- A `.real`-typed rounded write **is** the `writeMemAsR R .real` write
(`writeMemTypedR` delegates `.real` to the exact `writeMemTyped`, and
`RoundingModel.storeValue_real` says `R.storeValue .real` is exactly the
finite-fallback demotion). Lets the `.real` masked store reuse the
`writeMemAsR` scatter readback/frame family. -/
private theorem writeMemTypedR_real_eq (R : RoundingModel) (s : BlockState)
    (region : RegionName) (offset : Nat) (v : TileCarrier TileDType.real) :
    s.writeMemTypedR R .real region offset v
      = s.writeMemAsR R .real region offset v := by
  show s.writeMemTyped .real region offset v = _
  simp only [BlockState.writeMemTyped, BlockState.writeMemAs,
    BlockState.writeMemAsR, RoundingModel.storeValue_real]

/-- Tag-exact readback of a stored `.real` cell through `readMemAs .real`:
the `storeValue ∘ ofReal` round trip is the identity on a defined real. -/
private theorem readMemAs_real_of_cell {s : BlockState} {region : RegionName}
    {offset : Nat} {x : ℝ}
    (h : s.mem region offset
      = MemCell.of FloatDType.real.toTileDType (FloatDType.real.ofReal x)) :
    s.readMemAs .real region offset = FloatDType.real.ofReal x := by
  simp [BlockState.readMemAs, h, FloatDType.storeValue, FloatDType.ofReal]

/-- The masked store tail under `stepStmtR R` reduces to a masked
`writeMemAsR R .real` scatter `foldl` (the `R`-mirror of
`store_tail_step`; the `.real` write carries no rounding event). -/
private theorem store_tail_stepR (R : RoundingModel) (out_ptr : RegionName)
    (N BLOCK_N : Nat) (offF : Fin BLOCK_N → Nat) (vals : Fin BLOCK_N → ℝ)
    (s : BlockState)
    (hcp : s.regs .ptr [BLOCK_N] "c_ptr"
      = some (⟨fun idx : TileIndex [BLOCK_N] =>
          ((out_ptr, offF idx.1) : RegionName × Nat)⟩ : Tile .ptr [BLOCK_N]))
    (hcm : s.regs .bool [BLOCK_N] "c_mask"
      = some (⟨fun idx : TileIndex [BLOCK_N] => decide (idx.1.val < N)⟩
          : Tile .bool [BLOCK_N]))
    (hacc : s.regs .real [BLOCK_N] "accumulator"
      = some ⟨fun idx : TileIndex [BLOCK_N] => some (vals idx.1)⟩) :
    stepStmtR R (shrinkStoreStmt BLOCK_N) s
      = some ((TileShape.allIndices [BLOCK_N]).foldl
          (fun acc (idx : TileIndex [BLOCK_N]) =>
            if decide (idx.1.val < N) then
              acc.writeMemAsR R .real out_ptr (offF idx.1)
                (some (vals idx.1))
            else acc) s) := by
  unfold shrinkStoreStmt
  simp only [stepStmtR]
  rw [show evalOpR R (Op.ref .real [BLOCK_N] "accumulator") s
      = some (⟨fun idx : TileIndex [BLOCK_N] => some (vals idx.1)⟩
          : Tile .real [BLOCK_N]) from by rw [evalOpR_ref, hacc]]
  rw [show evalOpR R (Op.ref .bool [BLOCK_N] "c_mask") s
      = some (⟨fun idx : TileIndex [BLOCK_N] => decide (idx.1.val < N)⟩
          : Tile .bool [BLOCK_N]) from by rw [evalOpR_ref, hcm]]
  rw [show evalOpR R (Op.ref .ptr [BLOCK_N] "c_ptr") s
      = some (⟨fun idx : TileIndex [BLOCK_N] =>
          ((out_ptr, offF idx.1) : RegionName × Nat)⟩ : Tile .ptr [BLOCK_N])
        from by rw [evalOpR_ref, hcp]]
  simp only [bind, Option.map_some, Option.bind_some]
  refine congrArg some (List.foldl_ext _ _ _ (fun acc i _ => ?_))
  by_cases hmask : (decide (i.1.val < N) : Bool)
  · simp only [hmask, if_true, writeMemTypedR_real_eq]
  · simp only [hmask, Bool.false_eq_true, if_false]

/-- The output-window scatter offsets are injective over the whole lane tile
(from `0 < cn_stride`, via `affine1D_inj`). -/
private theorem outOff_tileInj (s : BlockState) (cm_stride cn_stride : Nat)
    (hcn : 0 < cn_stride) {BLOCK_N : Nat} :
    Function.Injective (fun idx : TileIndex [BLOCK_N] =>
      outOff s cm_stride cn_stride idx.1.val) := by
  intro a b hab
  have h1 : a.1 = b.1 :=
    affine1D_inj (s.pids 1 * cm_stride) cn_stride hcn
      (show (fun k : Fin BLOCK_N => s.pids 1 * cm_stride + k.val * cn_stride) a.1
          = (fun k : Fin BLOCK_N => s.pids 1 * cm_stride + k.val * cn_stride) b.1
        from hab)
  exact Prod.ext h1 rfl

/-- **R-postLoop (store face)**: from the exact invariant at loop exit, the
`execR R` common tail + masked store terminate and leave, at every active
output lane `n < N`, the exact-ℝ cell
`MemCell.of .real (real.ofReal (accPartial (i/S) n · scaling))` — the `.real`
store has no rounding event — with a per-cell frame outside the write-active
window (stated against the pristine pre-launch memory `s0.mem`; the loop and
tail never write). -/
private theorem shrink_postLoopR (R : RoundingModel)
    (input_ptr lora_ptr out_ptr : RegionName) (lora_indices : Region .nat)
    (scaling : ℝ)
    (N K xm_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
      BLOCK_N BLOCK_K SPLIT_K : Nat) (hcn : 0 < cn_stride)
    (s0 st : BlockState) (i : Nat)
    (hinv : shrinkInv input_ptr lora_ptr lora_indices N K xm_stride l0_stride
        lora_k_stride lora_n_stride BLOCK_N BLOCK_K SPLIT_K s0 i st) :
    ∃ sfin, stepStmtsR R (shrinkTailCommon out_ptr N cm_stride cn_stride
          BLOCK_N scaling ++ [shrinkStoreStmt BLOCK_N]) st = some sfin
      ∧ (∀ n : Fin BLOCK_N, n.val < N →
          sfin.mem out_ptr (outOff s0 cm_stride cn_stride n.val)
            = MemCell.of FloatDType.real.toTileDType
                (FloatDType.real.ofReal
                  (accPartial s0 input_ptr lora_ptr lora_indices K xm_stride
                      l0_stride lora_k_stride lora_n_stride BLOCK_K SPLIT_K
                      (i / (BLOCK_K * SPLIT_K)) n.val * scaling)))
      ∧ (∀ r o, (r ≠ out_ptr ∨ ∀ n : Fin BLOCK_N, n.val < N →
            o ≠ outOff s0 cm_stride cn_stride n.val) →
          sfin.mem r o = s0.mem r o) := by
  obtain ⟨st', hTail, hcp', hcm', hacc', hmem'⟩ := tail_common_steps input_ptr
    lora_ptr out_ptr lora_indices scaling N K xm_stride l0_stride lora_k_stride
    lora_n_stride cm_stride cn_stride BLOCK_N BLOCK_K SPLIT_K s0 st i hinv
  set vals : Fin BLOCK_N → ℝ := fun n =>
    (if n.val < N then
      accPartial s0 input_ptr lora_ptr lora_indices K xm_stride l0_stride
        lora_k_stride lora_n_stride BLOCK_K SPLIT_K
        (i / (BLOCK_K * SPLIT_K)) n.val
    else 0) * scaling with hvals
  have hStoreR := store_tail_stepR R out_ptr N BLOCK_N
    (fun n : Fin BLOCK_N => outOff s0 cm_stride cn_stride n.val) vals st'
    hcp' hcm' (by rw [hacc'])
  refine ⟨(TileShape.allIndices [BLOCK_N]).foldl
      (fun acc (idx : TileIndex [BLOCK_N]) =>
        if decide (idx.1.val < N) then
          acc.writeMemAsR R .real out_ptr (outOff s0 cm_stride cn_stride idx.1.val)
            (some (vals idx.1))
        else acc) st', ?_, ?_, ?_⟩
  · rw [stepStmtsR_append, shrinkTailCommon_castFree, hTail, Option.bind_some,
      stepStmtsR_cons_some hStoreR, stepStmtsR_nil]
  · intro n hn
    rw [BlockState.scatter_memcell_R_prop_masked_nd R .real st'
        (fun idx : TileIndex [BLOCK_N] => outOff s0 cm_stride cn_stride idx.1.val)
        (fun idx : TileIndex [BLOCK_N] => some (vals idx.1))
        (fun idx : TileIndex [BLOCK_N] => decide (idx.1.val < N) = Bool.true)
        (outOff_tileInj s0 cm_stride cn_stride hcn) (n, PUnit.unit)]
    rw [if_pos (decide_eq_true hn)]
    simp only [hvals, RoundingModel.storeValue_real, FloatDType.real_storeValue,
      WithBot.unbotD_some, if_pos hn]
  · intro r o hcond
    by_cases hr : r = out_ptr
    · subst hr
      have hno : ∀ n : Fin BLOCK_N, n.val < N →
          o ≠ outOff s0 cm_stride cn_stride n.val := by
        rcases hcond with h | h
        · exact absurd rfl h
        · exact h
      rw [BlockState.foldl_writeMemAsR_preserve_masked_prop R .real
            (fun idx : TileIndex [BLOCK_N] => outOff s0 cm_stride cn_stride idx.1.val)
            (fun idx : TileIndex [BLOCK_N] => some (vals idx.1))
            (fun idx : TileIndex [BLOCK_N] => decide (idx.1.val < N) = Bool.true)
            o (TileShape.allIndices [BLOCK_N])
            (fun k _ hk => fun heq =>
              hno k.1 (of_decide_eq_true hk) heq.symm) st', hmem']
    · rw [BlockState.foldl_writeMemAsR_preserve_other_region R .real
            (fun idx : TileIndex [BLOCK_N] => outOff s0 cm_stride cn_stride idx.1.val)
            (fun idx : TileIndex [BLOCK_N] => some (vals idx.1))
            (fun idx : TileIndex [BLOCK_N] => decide (idx.1.val < N) = Bool.true)
            r hr o (TileShape.allIndices [BLOCK_N]) st', hmem']

/-! ### The `TraceSafeR` safety walk -/

/-- The loop counter of an in-range iteration sits below the ceil trip count:
`i < K` with `i = (i/S)·S` forces `i/S < ⌈K/S⌉`. -/
private theorem kloop_counter_lt (K S i : Nat) (hS : 0 < S) (hilt : i < K)
    (hi : i = i / S * S) : i / S < numKIters K S := by
  by_contra h
  have h1 : numKIters K S * S ≤ i / S * S :=
    Nat.mul_le_mul_right S (Nat.le_of_not_lt h)
  have h2 : K ≤ i / S * S := le_trans (numKIters_bounds K S hS).1 h1
  rw [← hi] at h2
  exact absurd hilt (Nat.not_lt.mpr h2)

set_option maxHeartbeats 2000000 in
/-- Per-iteration `TraceSafeListR` for the K-loop body: the two **masked**
loads' active lanes are exactly the kernel's `current_k < K` (and, for
`tiled_b`, `offset_n < N`) windows, so their addresses are covered by the
mask-gated `read1`/`read2` bound groups; the four remaining assigns are
register-only. -/
private theorem shrink_bodySafeR (R : RoundingModel) (bounds : RegionBounds)
    (input_ptr lora_ptr : RegionName) (lora_indices : Region .nat)
    (N K xm_stride l0_stride lora_k_stride lora_n_stride
      BLOCK_N BLOCK_K SPLIT_K : Nat) (s0 : BlockState)
    (hS : 0 < BLOCK_K * SPLIT_K) (i : Nat) (hilt : i < K)
    (hi : i = i / (BLOCK_K * SPLIT_K) * (BLOCK_K * SPLIT_K))
    (sk : BlockState)
    (hk : sk.regs .nat [] "k" = some (Tile.scalar i))
    (hon : sk.regs .nat [BLOCK_N] "offset_n"
      = some (Tile.vec (fun j : Fin BLOCK_N => j.val)))
    (hok : sk.regs .nat [BLOCK_K] "offset_k"
      = some (Tile.vec (fun e : Fin BLOCK_K => e.val + s0.pids 0 * BLOCK_K)))
    (hap : sk.regs .ptr [] "a_ptr"
      = some (Tile.scalar ((input_ptr, s0.pids 1 * xm_stride) : RegionName × Nat)))
    (hbp : sk.regs .ptr [] "b_ptr"
      = some (Tile.scalar
          ((lora_ptr, l0_stride * loraIdx s0 lora_indices) : RegionName × Nat)))
    (hbA : ∀ (t : Fin (numKIters K (BLOCK_K * SPLIT_K))) (e : Fin BLOCK_K),
      t.val * (BLOCK_K * SPLIT_K) + (e.val + s0.pids 0 * BLOCK_K) < K →
      s0.pids 1 * xm_stride
          + (t.val * (BLOCK_K * SPLIT_K) + (e.val + s0.pids 0 * BLOCK_K))
        < bounds input_ptr)
    (hbB : ∀ (t : Fin (numKIters K (BLOCK_K * SPLIT_K)))
        (l : Fin (BLOCK_N * BLOCK_K)),
      l.val / BLOCK_K < N ∧
        t.val * (BLOCK_K * SPLIT_K) + (l.val % BLOCK_K + s0.pids 0 * BLOCK_K) < K →
      l0_stride * loraIdx s0 lora_indices + l.val / BLOCK_K * lora_k_stride
          + (t.val * (BLOCK_K * SPLIT_K)
              + (l.val % BLOCK_K + s0.pids 0 * BLOCK_K)) * lora_n_stride
        < bounds lora_ptr) :
    Stmt.TraceSafeListR R bounds
      (shrinkLoopBody N K lora_k_stride lora_n_stride BLOCK_N BLOCK_K) sk := by
  have hcT : i / (BLOCK_K * SPLIT_K) < numKIters K (BLOCK_K * SPLIT_K) :=
    kloop_counter_lt K (BLOCK_K * SPLIT_K) i hS hilt hi
  set ckF : Fin BLOCK_K → Nat :=
    fun e => i + (e.val + s0.pids 0 * BLOCK_K) with hckF
  have hck_eq : ∀ e : Fin BLOCK_K,
      ckF e = i / (BLOCK_K * SPLIT_K) * (BLOCK_K * SPLIT_K)
        + (e.val + s0.pids 0 * BLOCK_K) := by
    intro e
    rw [hckF, ← hi]
  unfold shrinkLoopBody
  -- 1. current_k (register-only)
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) ?_
  intro s1 h1
  obtain ⟨v1, hv1, rfl⟩ := stepStmtR_assign_inv h1
  rw [evalR_currentK,
    currentK_eval BLOCK_K i (fun e : Fin BLOCK_K => e.val + s0.pids 0 * BLOCK_K)
      sk hk hok] at hv1
  obtain rfl := Option.some.inj hv1
  -- 2. current_k_c (register-only)
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) ?_
  intro s2 h2
  obtain ⟨v2, hv2, rfl⟩ := stepStmtR_assign_inv h2
  rw [evalOpR_ref, BlockState.setReg_same] at hv2
  obtain rfl := Option.some.inj hv2
  have hck2 : ((sk.setReg "current_k" .nat [BLOCK_K] (Tile.vec ckF)).setReg
      "current_k_c" .nat [BLOCK_K] (Tile.vec ckF)).regs .nat [BLOCK_K]
      "current_k" = some (Tile.vec ckF) := by
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
      (show ("current_k" : RegName) ≠ "current_k_c" by decide),
      BlockState.setReg_same]
  have hckc2 : ((sk.setReg "current_k" .nat [BLOCK_K] (Tile.vec ckF)).setReg
      "current_k_c" .nat [BLOCK_K] (Tile.vec ckF)).regs .nat [BLOCK_K]
      "current_k_c" = some (Tile.vec ckF) := by
    rw [BlockState.setReg_same]
  have hap2 : ((sk.setReg "current_k" .nat [BLOCK_K] (Tile.vec ckF)).setReg
      "current_k_c" .nat [BLOCK_K] (Tile.vec ckF)).regs .ptr [] "a_ptr"
      = some (Tile.scalar ((input_ptr, s0.pids 1 * xm_stride) : RegionName × Nat)) := by
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("a_ptr" : RegName) ≠ "current_k_c" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("a_ptr" : RegName) ≠ "current_k" by decide)]
    exact hap
  -- 3. tiled_a (masked load, mask-gated bound group)
  refine Stmt.TraceSafeListR.cons_intro ?_ ?_
  · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
    refine ⟨by simp, by simp, ?_⟩
    intro ptrs hptrs e hactive
    rw [evalR_aTilePtr,
      aTilePtr_eval input_ptr BLOCK_K (s0.pids 1 * xm_stride) ckF _
        hap2 hckc2] at hptrs
    obtain rfl := Option.some.inj hptrs
    obtain ⟨masks, hmasks, hme⟩ := hactive
    rw [evalR_maskA, maskA_eval K BLOCK_K ckF _ hck2] at hmasks
    obtain rfl := Option.some.inj hmasks
    have hkK : ckF e.1 < K := of_decide_eq_true hme
    show s0.pids 1 * xm_stride + ckF e.1 < bounds input_ptr
    have h' := hbA ⟨i / (BLOCK_K * SPLIT_K), hcT⟩ e.1
      (by rw [← hck_eq e.1]; exact hkK)
    rw [← hck_eq e.1] at h'
    exact h'
  · intro s3 h3
    obtain ⟨v3, -, rfl⟩ := stepStmtR_assign_inv h3
    -- 4. b_ptr_mask (register-only)
    refine Stmt.TraceSafeListR.cons_intro
      (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) ?_
    intro s4 h4
    obtain ⟨v4, hv4, rfl⟩ := stepStmtR_assign_inv h4
    have hon3 : (((sk.setReg "current_k" .nat [BLOCK_K] (Tile.vec ckF)).setReg
        "current_k_c" .nat [BLOCK_K] (Tile.vec ckF)).setReg "tiled_a" .real
        [BLOCK_K] v3).regs .nat [BLOCK_N] "offset_n"
        = some (Tile.vec (fun j : Fin BLOCK_N => j.val)) := by
      rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
          (show ("offset_n" : RegName) ≠ "tiled_a" by decide),
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _
          (show ("offset_n" : RegName) ≠ "current_k_c" by decide),
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _
          (show ("offset_n" : RegName) ≠ "current_k" by decide)]
      exact hon
    have hck3 : (((sk.setReg "current_k" .nat [BLOCK_K] (Tile.vec ckF)).setReg
        "current_k_c" .nat [BLOCK_K] (Tile.vec ckF)).setReg "tiled_a" .real
        [BLOCK_K] v3).regs .nat [BLOCK_K] "current_k"
        = some (Tile.vec ckF) := by
      rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("current_k" : RegName) ≠ "tiled_a" by decide)]
      exact hck2
    rw [evalR_bmask, bmask_eval N K BLOCK_N BLOCK_K ckF _ hon3 hck3] at hv4
    obtain rfl := Option.some.inj hv4
    -- 5. tiled_b (masked load, mask-gated bound group)
    refine Stmt.TraceSafeListR.cons_intro ?_ ?_
    · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
        MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
      refine ⟨by simp, by simp, ?_⟩
      intro ptrs hptrs idx hactive
      have hbp4 : ((((sk.setReg "current_k" .nat [BLOCK_K] (Tile.vec ckF)).setReg
          "current_k_c" .nat [BLOCK_K] (Tile.vec ckF)).setReg "tiled_a" .real
          [BLOCK_K] v3).setReg "b_ptr_mask" .bool [BLOCK_N, BLOCK_K]
          (⟨fun idx : TileIndex [BLOCK_N, BLOCK_K] =>
            decide (idx.1.val < N) && decide (ckF idx.2.1 < K)⟩
              : Tile .bool [BLOCK_N, BLOCK_K])).regs .ptr [] "b_ptr"
          = some (Tile.scalar
              ((lora_ptr, l0_stride * loraIdx s0 lora_indices) : RegionName × Nat)) := by
        rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("b_ptr" : RegName) ≠ "b_ptr_mask" by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("b_ptr" : RegName) ≠ "tiled_a" by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("b_ptr" : RegName) ≠ "current_k_c" by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("b_ptr" : RegName) ≠ "current_k" by decide)]
        exact hbp
      have hon4 : ((((sk.setReg "current_k" .nat [BLOCK_K]
          (Tile.vec ckF)).setReg "current_k_c" .nat [BLOCK_K]
          (Tile.vec ckF)).setReg "tiled_a" .real [BLOCK_K] v3).setReg
          "b_ptr_mask" .bool [BLOCK_N, BLOCK_K]
          (⟨fun idx : TileIndex [BLOCK_N, BLOCK_K] =>
            decide (idx.1.val < N) && decide (ckF idx.2.1 < K)⟩
              : Tile .bool [BLOCK_N, BLOCK_K])).regs .nat [BLOCK_N] "offset_n"
          = some (Tile.vec (fun j : Fin BLOCK_N => j.val)) := by
        rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
          (show ("offset_n" : RegName) ≠ "b_ptr_mask" by decide)]
        exact hon3
      have hck4 : ((((sk.setReg "current_k" .nat [BLOCK_K] (Tile.vec ckF)).setReg
          "current_k_c" .nat [BLOCK_K] (Tile.vec ckF)).setReg "tiled_a" .real
          [BLOCK_K] v3).setReg "b_ptr_mask" .bool [BLOCK_N, BLOCK_K]
          (⟨fun idx : TileIndex [BLOCK_N, BLOCK_K] =>
            decide (idx.1.val < N) && decide (ckF idx.2.1 < K)⟩
              : Tile .bool [BLOCK_N, BLOCK_K])).regs .nat [BLOCK_K] "current_k"
          = some (Tile.vec ckF) := by
        rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
          (show ("current_k" : RegName) ≠ "b_ptr_mask" by decide)]
        exact hck3
      rw [evalR_bTilePtr,
        bTilePtr_eval lora_ptr lora_k_stride lora_n_stride BLOCK_N BLOCK_K
          (l0_stride * loraIdx s0 lora_indices) ckF _ hbp4 hon4 hck4] at hptrs
      obtain rfl := Option.some.inj hptrs
      obtain ⟨masks, hmasks, hmi⟩ := hactive
      rw [evalOpR_ref, BlockState.setReg_same] at hmasks
      obtain rfl := Option.some.inj hmasks
      have hmi' : idx.1.val < N ∧ ckF idx.2.1 < K := by
        have := hmi
        simp only [Bool.and_eq_true, decide_eq_true_eq] at this
        exact this
      show l0_stride * loraIdx s0 lora_indices + idx.1.val * lora_k_stride
          + ckF idx.2.1 * lora_n_stride < bounds lora_ptr
      have h' := hbB ⟨i / (BLOCK_K * SPLIT_K), hcT⟩
        (Lane2D.encode (idx.1, idx.2.1, PUnit.unit))
        (by
          rw [Lane2D.encode_div, Lane2D.encode_mod, ← hck_eq idx.2.1]
          exact hmi')
      rw [Lane2D.encode_div, Lane2D.encode_mod, ← hck_eq idx.2.1] at h'
      exact h'
    · intro s5 h5
      obtain ⟨v5, -, rfl⟩ := stepStmtR_assign_inv h5
      -- 6. accumulator (register-only)
      refine Stmt.TraceSafeListR.cons_intro
        (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
        (fun s6 _ => Stmt.TraceSafeListR.nil_intro)

set_option maxHeartbeats 2000000 in
/-- `TraceSafeListR` for the common tail + masked store: the four assigns are
register-only, and the masked `.real` store's active lanes are exactly the
`offset_cn < N` window, so their addresses are the `writeMask`-gated `write`
bounds. -/
private theorem shrink_tailSafeR (R : RoundingModel) (bounds : RegionBounds)
    (out_ptr : RegionName) (N cm_stride cn_stride BLOCK_N : Nat) (scaling : ℝ)
    (cb : Nat) (st : BlockState)
    (hcb : st.regs .nat [] "cur_batch" = some (Tile.scalar cb))
    (hbC : ∀ j : Fin BLOCK_N, j.val < N →
      cb * cm_stride + j.val * cn_stride < bounds out_ptr) :
    Stmt.TraceSafeListR R bounds
      (shrinkTailCommon out_ptr N cm_stride cn_stride BLOCK_N scaling
        ++ [shrinkStoreStmt BLOCK_N]) st := by
  simp only [shrinkTailCommon, shrinkStoreStmt, List.cons_append,
    List.nil_append]
  -- 1. accumulator *= scaling
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) ?_
  intro s1 h1
  obtain ⟨v1, -, rfl⟩ := stepStmtR_assign_inv h1
  -- 2. offset_cn
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) ?_
  intro s2 h2
  obtain ⟨v2, hv2, rfl⟩ := stepStmtR_assign_inv h2
  rw [evalOpR_arange] at hv2
  obtain rfl := Option.some.inj hv2
  -- 3. c_ptr
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) ?_
  intro s3 h3
  obtain ⟨v3, hv3, rfl⟩ := stepStmtR_assign_inv h3
  have hcb2 : ((st.setReg "accumulator" .real [BLOCK_N] v1).setReg "offset_cn"
      .nat [BLOCK_N] (Tile.vec (fun j : Fin BLOCK_N => j.val))).regs .nat []
      "cur_batch" = some (Tile.scalar cb) := by
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("cur_batch" : RegName) ≠ "offset_cn" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("cur_batch" : RegName) ≠ "accumulator" by decide)]
    exact hcb
  have hocn2 : ((st.setReg "accumulator" .real [BLOCK_N] v1).setReg "offset_cn"
      .nat [BLOCK_N] (Tile.vec (fun j : Fin BLOCK_N => j.val))).regs .nat
      [BLOCK_N] "offset_cn"
      = some (Tile.vec (fun j : Fin BLOCK_N => j.val)) := by
    rw [BlockState.setReg_same]
  rw [evalR_cptr,
    cptr_tail_eval out_ptr cm_stride cn_stride BLOCK_N cb _ hcb2 hocn2] at hv3
  obtain rfl := Option.some.inj hv3
  -- 4. c_mask
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) ?_
  intro s4 h4
  obtain ⟨v4, hv4, rfl⟩ := stepStmtR_assign_inv h4
  have hocn3 : (((st.setReg "accumulator" .real [BLOCK_N] v1).setReg "offset_cn"
      .nat [BLOCK_N] (Tile.vec (fun j : Fin BLOCK_N => j.val))).setReg "c_ptr"
      .ptr [BLOCK_N]
      (⟨fun idx : TileIndex [BLOCK_N] =>
        ((out_ptr, cb * cm_stride + idx.1.val * cn_stride) : RegionName × Nat)⟩
          : Tile .ptr [BLOCK_N])).regs .nat [BLOCK_N] "offset_cn"
      = some (Tile.vec (fun j : Fin BLOCK_N => j.val)) := by
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
      (show ("offset_cn" : RegName) ≠ "c_ptr" by decide)]
    exact hocn2
  rw [evalR_cmask, cmask_eval N BLOCK_N _ hocn3] at hv4
  obtain rfl := Option.some.inj hv4
  -- 5. the masked store
  refine Stmt.TraceSafeListR.cons_intro ?_
    (fun s5 _ => Stmt.TraceSafeListR.nil_intro)
  simp only [Stmt.TraceSafeR, MemAccess.SafeAtR, MaskOpt.SafeAtR,
    Op.SafeAtR.eq_def, MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR,
    memAccessActiveAddressSafeR]
  refine ⟨trivial, trivial, trivial, ?_⟩
  intro ptrs hptrs idx hactive
  rw [evalOpR_ref] at hptrs
  rw [show ((((st.setReg "accumulator" .real [BLOCK_N] v1).setReg "offset_cn"
      .nat [BLOCK_N] (Tile.vec (fun j : Fin BLOCK_N => j.val))).setReg "c_ptr"
      .ptr [BLOCK_N]
      (⟨fun idx : TileIndex [BLOCK_N] =>
        ((out_ptr, cb * cm_stride + idx.1.val * cn_stride) : RegionName × Nat)⟩
          : Tile .ptr [BLOCK_N])).setReg "c_mask" .bool [BLOCK_N]
      (⟨fun idx : TileIndex [BLOCK_N] => decide (idx.1.val < N)⟩
        : Tile .bool [BLOCK_N])).regs .ptr [BLOCK_N] "c_ptr"
      = some (⟨fun idx : TileIndex [BLOCK_N] =>
          ((out_ptr, cb * cm_stride + idx.1.val * cn_stride) : RegionName × Nat)⟩
            : Tile .ptr [BLOCK_N]) from by
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("c_ptr" : RegName) ≠ "c_mask" by decide),
      BlockState.setReg_same]] at hptrs
  obtain rfl := Option.some.inj hptrs
  obtain ⟨masks, hmasks, hmi⟩ := hactive
  rw [evalOpR_ref, BlockState.setReg_same] at hmasks
  obtain rfl := Option.some.inj hmasks
  have hn : idx.1.val < N := of_decide_eq_true hmi
  show cb * cm_stride + idx.1.val * cn_stride < bounds out_ptr
  exact hbC idx.1 hn

set_option maxHeartbeats 2000000 in
/-- **The `TraceSafeR` walk for the whole store surface**, driven by
`Stmt.forRangeTraceSafeR_inv` over the full exact invariant `shrinkInv`
(usable from arbitrary launch states — bgmv's stack needs no clean-`undef`
precondition, both loop loads carry `other=0`). The four bound groups are the
skin's slot window (`hbM`), the two mask-gated streamed read windows
(`hbA`/`hbB`, stated on the *loaded* `loraIdx`) and the `writeMask`-gated
write window (`hbC`). -/
private theorem shrink_store_traceSafeR (R : RoundingModel)
    (bounds : RegionBounds) (input_ptr lora_ptr out_ptr : RegionName)
    (N K : Nat) (lora_indices : Region .nat) (scaling : ℝ)
    (xm_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
      BLOCK_N BLOCK_K SPLIT_K : Nat) (s : BlockState)
    (hS : 0 < BLOCK_K * SPLIT_K)
    (hbM : s.pids 1 < bounds (Region.cast lora_indices))
    (hbA : ∀ (t : Fin (numKIters K (BLOCK_K * SPLIT_K))) (e : Fin BLOCK_K),
      t.val * (BLOCK_K * SPLIT_K) + (e.val + s.pids 0 * BLOCK_K) < K →
      s.pids 1 * xm_stride
          + (t.val * (BLOCK_K * SPLIT_K) + (e.val + s.pids 0 * BLOCK_K))
        < bounds input_ptr)
    (hbB : ∀ (t : Fin (numKIters K (BLOCK_K * SPLIT_K)))
        (l : Fin (BLOCK_N * BLOCK_K)),
      l.val / BLOCK_K < N ∧
        t.val * (BLOCK_K * SPLIT_K) + (l.val % BLOCK_K + s.pids 0 * BLOCK_K) < K →
      l0_stride * loraIdx s lora_indices + l.val / BLOCK_K * lora_k_stride
          + (t.val * (BLOCK_K * SPLIT_K)
              + (l.val % BLOCK_K + s.pids 0 * BLOCK_K)) * lora_n_stride
        < bounds lora_ptr)
    (hbC : ∀ j : Fin BLOCK_N, j.val < N →
      s.pids 1 * cm_stride + j.val * cn_stride < bounds out_ptr) :
    ((bgmv_shrink_store_surface input_ptr lora_ptr out_ptr N K lora_indices
        scaling xm_stride l0_stride lora_k_stride lora_n_stride cm_stride
        cn_stride BLOCK_N BLOCK_K SPLIT_K).toAlgKernel).TraceSafeR R bounds s := by
  unfold Kernel.TraceSafeR
  rw [store_body_split input_ptr lora_ptr out_ptr N K lora_indices scaling
      xm_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
      BLOCK_N BLOCK_K SPLIT_K,
    store_prologue_split input_ptr lora_ptr out_ptr N K lora_indices scaling
      xm_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
      BLOCK_N BLOCK_K SPLIT_K]
  have hstep : ∀ c s', c < K →
      shrinkInv input_ptr lora_ptr lora_indices N K xm_stride l0_stride
        lora_k_stride lora_n_stride BLOCK_N BLOCK_K SPLIT_K s c s' →
      Stmt.TraceSafeListR R bounds
        (shrinkLoopBody N K lora_k_stride lora_n_stride BLOCK_N BLOCK_K)
        (s'.setReg "k" .nat [] (Tile.scalar c)) ∧
      ∃ s'', stepStmtsR R
          (shrinkLoopBody N K lora_k_stride lora_n_stride BLOCK_N BLOCK_K)
          (s'.setReg "k" .nat [] (Tile.scalar c)) = some s'' ∧
        shrinkInv input_ptr lora_ptr lora_indices N K xm_stride l0_stride
          lora_k_stride lora_n_stride BLOCK_N BLOCK_K SPLIT_K s
          (c + BLOCK_K * SPLIT_K) s'' := by
    intro c s' hc hP
    obtain ⟨hpids, hi, hbound, hcb, hon, hok, hap, hbp, hacc, hmem⟩ := hP
    constructor
    · refine shrink_bodySafeR R bounds input_ptr lora_ptr lora_indices N K
        xm_stride l0_stride lora_k_stride lora_n_stride BLOCK_N BLOCK_K SPLIT_K
        s hS c hc hi _ (by rw [BlockState.setReg_same]) ?_ ?_ ?_ ?_ hbA hbB
      · rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
          (show ("offset_n" : RegName) ≠ "k" by decide)]
        exact hon
      · rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
          (show ("offset_k" : RegName) ≠ "k" by decide)]
        exact hok
      · rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
          (show ("a_ptr" : RegName) ≠ "k" by decide)]
        exact hap
      · rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
          (show ("b_ptr" : RegName) ≠ "k" by decide)]
        exact hbp
    · obtain ⟨s'', hs'', hP''⟩ := shrink_step input_ptr lora_ptr lora_indices
        N K xm_stride l0_stride lora_k_stride lora_n_stride BLOCK_N BLOCK_K
        SPLIT_K s hS c s' hc
        ⟨hpids, hi, hbound, hcb, hon, hok, hap, hbp, hacc, hmem⟩
      exact ⟨s'', by
        rw [shrinkBody_castFree]
        exact hs'', hP''⟩
  refine Stmt.TraceSafeListR.append_intro _ _ ?_ ?_
  · -- the prologue: seven register-only assigns around one `.nat` slot load
    unfold shrinkPrologue
    refine Stmt.TraceSafeListR.cons_intro
      (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) ?_
    intro s1 h1
    obtain ⟨v1, -, rfl⟩ := stepStmtR_assign_inv h1
    refine Stmt.TraceSafeListR.cons_intro
      (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) ?_
    intro s2 h2
    obtain ⟨v2, hv2, rfl⟩ := stepStmtR_assign_inv h2
    rw [evalOpR_programId] at hv2
    obtain rfl := Option.some.inj hv2
    refine Stmt.TraceSafeListR.cons_intro ?_ ?_
    · -- the lora_indices slot load: address = cur_batch = pids 1
      simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
        MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
      refine ⟨trivial, trivial, ?_⟩
      intro offsets hoffs i _
      rw [evalOpR_ref, BlockState.setReg_same] at hoffs
      obtain rfl := Option.some.inj hoffs
      show ((s.setReg "pid_sk" .nat [] v1).pids 1) < bounds (Region.cast lora_indices)
      exact hbM
    · intro s3 h3
      obtain ⟨v3, -, rfl⟩ := stepStmtR_assign_inv h3
      refine Stmt.TraceSafeListR.of_forall _ _ ?_
      intro st hst s'
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hst
      rcases hst with rfl | rfl | rfl | rfl | rfl <;>
        simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]
  · intro s1 hs1
    obtain ⟨sp, hsp, hP0⟩ := shrink_preLoop input_ptr lora_ptr lora_indices
      N K xm_stride l0_stride lora_k_stride lora_n_stride BLOCK_N BLOCK_K
      SPLIT_K s hS
    rw [shrinkPrologue_castFree, hsp] at hs1
    obtain rfl := Option.some.inj hs1
    refine Stmt.TraceSafeListR.cons_intro ?_ ?_
    · -- the K-loop is trace-safe (invariant principle over `shrinkInv`)
      simp only [Stmt.TraceSafeR]
      exact Stmt.forRangeTraceSafeR_inv R bounds "k" K (BLOCK_K * SPLIT_K)
        (shrinkLoopBody N K lora_k_stride lora_n_stride BLOCK_N BLOCK_K)
        (shrinkInv input_ptr lora_ptr lora_indices N K xm_stride l0_stride
          lora_k_stride lora_n_stride BLOCK_N BLOCK_K SPLIT_K s)
        hstep 0 sp hP0
    · intro s2 hs2
      obtain ⟨final, sLoop, hLoopStmt, hfinal, hPLoop⟩ :=
        forRange_inv (idx := "k") (start := 0) (stop := K)
          (step := BLOCK_K * SPLIT_K)
          (Nat.pos_iff_ne_zero.mp hS) hP0
          (fun i st hlt hinv => shrink_step input_ptr lora_ptr lora_indices
            N K xm_stride l0_stride lora_k_stride lora_n_stride BLOCK_N
            BLOCK_K SPLIT_K s hS i st hlt hinv)
      rw [stepStmtR_forRange,
        stepForRangeAuxR_castFree R _
          (shrinkBody_castFree R N K lora_k_stride lora_n_stride BLOCK_N
            BLOCK_K) "k",
        ← stepForRangeAux.forRange_unfold, hLoopStmt] at hs2
      obtain rfl := Option.some.inj hs2
      obtain ⟨-, -, -, hcbL, -, -, -, -, -, -⟩ := hPLoop
      exact shrink_tailSafeR R bounds out_ptr N cm_stride cn_stride BLOCK_N
        scaling (s.pids 1) sLoop hcbL hbC

/-- The store surface sits inside the flat-memory bridge's covered fragment. -/
private theorem shrink_store_flattenOk
    (input_ptr lora_ptr out_ptr : RegionName)
    (N K : Nat) (lora_indices : Region .nat) (scaling : ℝ)
    (xm_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
      BLOCK_N BLOCK_K SPLIT_K : Nat) :
    ((bgmv_shrink_store_surface input_ptr lora_ptr out_ptr N K lora_indices
        scaling xm_stride l0_stride lora_k_stride lora_n_stride cm_stride
        cn_stride BLOCK_N BLOCK_K SPLIT_K).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  rw [store_body_split input_ptr lora_ptr out_ptr N K lora_indices scaling
      xm_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
      BLOCK_N BLOCK_K SPLIT_K,
    store_prologue_split input_ptr lora_ptr out_ptr N K lora_indices scaling
      xm_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
      BLOCK_N BLOCK_K SPLIT_K]
  simp [shrinkPrologue, shrinkLoopBody, shrinkTailCommon, shrinkStoreStmt,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-! ### IO signature, lane bridge, spec bridge -/

/-- The `tiled_b`-stream lane feeding output lane `j` at rank key `e`: `(j, e)`
row-major over the `[BLOCK_N, BLOCK_K]` per-step weight tile, via the shared
`Lane2D` bridge. -/
def bStreamLane (BLOCK_N BLOCK_K : Nat) (j : Fin BLOCK_N) (e : Fin BLOCK_K) :
    Fin (BLOCK_N * BLOCK_K) :=
  Lane2D.encode (j, e, PUnit.unit)

/-- **Streaming metadata IO signature** of the `SPLIT_K = 1` store face on the
metadata-slot two-stream fold skin (S1: fold + terminal store). One `.nat`
slot: `lora_indices[cur_batch]`, read at the pid-only cell `pid₁`
(`mwin`; the store surface's `Region .nat` metadata convention — the `-1`
skip sentinel is the faithful `.int` surface's separate obligation). Step `t`
of the K-loop reads the `[BLOCK_K]` input row tile and the
`[BLOCK_N, BLOCK_K]` LoRA-A weight tile; after the loop one `[BLOCK_N]`
output tile is masked-stored at the exact `.real` grid (`outDType := .real` —
the kernel's store is untyped/fp32-register, no rounding event). The windows
transcribe the kernel's pointer arithmetic verbatim:

* `read1` step `t`, lane `e`: `cur_batch·xm_stride + current_k` with
  `current_k = t·(BLOCK_K·SPLIT_K) + (e + pid_sk·BLOCK_K)` — the running
  `a_ptr + current_k_c` cell (`tl.max_contiguous` erased).
* `read2` step `t`, lane `l = (n, e)` (row-major over `[BLOCK_N, BLOCK_K]`):
  `l0_stride·slot + n·lora_k_stride + current_k·lora_n_stride` — the
  `b_ptr` cell; **the slot value enters this base address** (the loaded
  adapter index selects the LoRA-A matrix).
* `write` lane `j`: `cur_batch·cm_stride + j·cn_stride` — the `c_ptr` cell
  (pure pid geometry; the slot never reaches the write address).
* `mask1`/`mask2` transcribe the loads' genuine `current_k < K` K-tail
  windows (`mask2` also the `offset_n < N` leg); `writeMask` transcribes
  the store's `offset_cn < N` boundary mask verbatim.

The kernel launches on a 2-D grid `(pid_sk, cur_batch)`; the skin's `pid₂`
slot is unused — every window is constant in it, and the headline still
quantifies over all three pids. -/
def bgmvShrinkStoreIO (input_ptr lora_ptr out_ptr : RegionName)
    (N K : Nat) (lora_indices : Region .nat) (scaling : ℝ)
    (xm_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
      BLOCK_N BLOCK_K SPLIT_K : Nat) : StreamMetaMasked3DKernelIO₂ where
  kernel := bgmv_shrink_store_surface input_ptr lora_ptr out_ptr N K
    lora_indices scaling xm_stride l0_stride lora_k_stride lora_n_stride
    cm_stride cn_stride BLOCK_N BLOCK_K SPLIT_K
  inp1 := input_ptr
  inp2 := lora_ptr
  out := out_ptr
  nMeta := 1
  sty := fun _ => .nat
  mbuf := fun _ => Region.cast lora_indices
  mwin := fun _ _ pid₁ _ => pid₁
  T := numKIters K (BLOCK_K * SPLIT_K)
  B1 := BLOCK_K
  B2 := BLOCK_N * BLOCK_K
  C := BLOCK_N
  outDType := .real
  read1 := fun pid₀ pid₁ _ _ t e =>
    pid₁ * xm_stride + (t.val * (BLOCK_K * SPLIT_K) + (e.val + pid₀ * BLOCK_K))
  read2 := fun pid₀ _ _ m t l =>
    l0_stride * m 0 + l.val / BLOCK_K * lora_k_stride
      + (t.val * (BLOCK_K * SPLIT_K) + (l.val % BLOCK_K + pid₀ * BLOCK_K))
        * lora_n_stride
  write := fun _ pid₁ _ _ j => pid₁ * cm_stride + j.val * cn_stride
  mask1 := fun pid₀ _ _ _ t e =>
    t.val * (BLOCK_K * SPLIT_K) + (e.val + pid₀ * BLOCK_K) < K
  mask2 := fun pid₀ _ _ _ t l =>
    l.val / BLOCK_K < N ∧
      t.val * (BLOCK_K * SPLIT_K) + (l.val % BLOCK_K + pid₀ * BLOCK_K) < K
  writeMask := fun _ _ _ _ j => j.val < N

/-- Under the slot pin and the two stream pins, the exact stack's
`accPartial` at the full trip count **is** the skin-level masked double fold
`∑ t, ∑ e, [current_k < K] · xs·ys` (per write-active lane `j < N`; the
`tiled_b` pins are addressed through the loaded slot value `m0`). -/
private theorem shrinkSpec_eq_streamSum
    (input_ptr lora_ptr : RegionName) (lora_indices : Region .nat)
    (N K xm_stride l0_stride lora_k_stride lora_n_stride
      BLOCK_N BLOCK_K SPLIT_K : Nat) (s₀ : BlockState) (m0 : Nat)
    (hli : loraIdx s₀ lora_indices = m0)
    (xs : Fin (numKIters K (BLOCK_K * SPLIT_K)) → Fin BLOCK_K → ℝ)
    (ys : Fin (numKIters K (BLOCK_K * SPLIT_K)) → Fin (BLOCK_N * BLOCK_K) → ℝ)
    (hx : ∀ (t : Fin (numKIters K (BLOCK_K * SPLIT_K))) (e : Fin BLOCK_K),
      t.val * (BLOCK_K * SPLIT_K) + (e.val + s₀.pids 0 * BLOCK_K) < K →
      s₀.readMem input_ptr (s₀.pids 1 * xm_stride
          + (t.val * (BLOCK_K * SPLIT_K) + (e.val + s₀.pids 0 * BLOCK_K)))
        = xs t e)
    (hy : ∀ (t : Fin (numKIters K (BLOCK_K * SPLIT_K)))
        (l : Fin (BLOCK_N * BLOCK_K)),
      l.val / BLOCK_K < N ∧
        t.val * (BLOCK_K * SPLIT_K) + (l.val % BLOCK_K + s₀.pids 0 * BLOCK_K) < K →
      s₀.readMem lora_ptr (l0_stride * m0 + l.val / BLOCK_K * lora_k_stride
          + (t.val * (BLOCK_K * SPLIT_K)
              + (l.val % BLOCK_K + s₀.pids 0 * BLOCK_K)) * lora_n_stride)
        = ys t l)
    (j : Fin BLOCK_N) (hj : j.val < N) :
    accPartial s₀ input_ptr lora_ptr lora_indices K xm_stride l0_stride
        lora_k_stride lora_n_stride BLOCK_K SPLIT_K
        (numKIters K (BLOCK_K * SPLIT_K)) j.val
      = ∑ t : Fin (numKIters K (BLOCK_K * SPLIT_K)), ∑ e : Fin BLOCK_K,
          if t.val * (BLOCK_K * SPLIT_K) + (e.val + s₀.pids 0 * BLOCK_K) < K
          then xs t e * ys t (bStreamLane BLOCK_N BLOCK_K j e)
          else 0 := by
  unfold accPartial
  rw [← Fin.sum_univ_eq_sum_range]
  refine Finset.sum_congr rfl fun t _ => ?_
  unfold blockTerm
  refine Finset.sum_congr rfl fun e _ => ?_
  have hkidx : kIdx s₀ BLOCK_K SPLIT_K t.val e.val
      = t.val * (BLOCK_K * SPLIT_K) + (e.val + s₀.pids 0 * BLOCK_K) := rfl
  rw [hkidx]
  by_cases hkK : t.val * (BLOCK_K * SPLIT_K) + (e.val + s₀.pids 0 * BLOCK_K) < K
  · rw [if_pos hkK, if_pos hkK]
    have hxa : aElem s₀ input_ptr xm_stride
        (t.val * (BLOCK_K * SPLIT_K) + (e.val + s₀.pids 0 * BLOCK_K))
        = xs t e := hx t e hkK
    have hyb : bElem s₀ lora_ptr lora_indices l0_stride lora_k_stride
        lora_n_stride j.val
        (t.val * (BLOCK_K * SPLIT_K) + (e.val + s₀.pids 0 * BLOCK_K))
        = ys t (bStreamLane BLOCK_N BLOCK_K j e) := by
      rw [← hy t (bStreamLane BLOCK_N BLOCK_K j e)
          (by
            unfold bStreamLane
            rw [Lane2D.encode_div, Lane2D.encode_mod]
            exact ⟨hj, hkK⟩)]
      unfold bElem bStreamLane
      rw [Lane2D.encode_div, Lane2D.encode_mod, hli]
    rw [hxa, hyb]
  · rw [if_neg hkK, if_neg hkK]

/-! ### ════════ ★ STREAMING METADATA HEADLINE (store face) ★ ════════ -/

set_option maxHeartbeats 4000000 in
/-- **The `⊨[R]` streaming metadata headline (wave-5, store face;
`SPLIT_K = 1` branch).** For every rounding model `R`, the store proof
surface implements, on its `StreamMetaMasked3DKernelIO₂` signature, the
**ideal ℝ scaled rank-slice fold** over the streamed tiles: for every
write-active output lane `j < N`,

```
out[cur_batch·cm_stride + j·cn_stride] =
  scaling · Σ_{t<⌈K/(BLOCK_K·SPLIT_K)⌉} Σ_{e<BLOCK_K}
    [current_k(t,e) < K] · x-tile[t](e) · loraA-tile[t](j,e)
```

with `current_k(t,e) = t·BLOCK_K·SPLIT_K + (e + pid_sk·BLOCK_K)` — the spec
`f` is exact real arithmetic over the pinned streams; the loaded metadata
slot `lora_indices[cur_batch]` enters only the `read2` base-address geometry
(the adapter-matrix select), never the value contract. The K-tail is genuine:
the streamed masks reproduce the kernel's `current_k < K` windows verbatim,
so `K` need not be a multiple of the split stride, and the masked spec sum
runs exactly over the kernel's live lanes.

This kernel has **zero rounding events** (every load, the K-loop and the
terminal masked store are `.real`), so with `outDType := .real` the readback
contract's `R.round .real` is the identity — the `⊨[R]` face at the `.real`
grid is the exact streaming contract stated once for every `R` (`R := .triv`
is the literal exact degeneration). Layer map: the prologue, K-loop and
common tail are cast-free and collapse onto the proven exact
`shrink_preLoop` / `shrink_step` / `forRange_inv` / `tail_common_steps`
stack; only the masked store statement is re-proved on the `R` side
(`shrink_postLoopR`).

Both hypotheses are truth-forced, inherited verbatim from
`store_exec_correct`:

* `hS : 0 < BLOCK_K * SPLIT_K` — the K-loop stride is positive (a launched
  constexpr tile is nonempty; at stride `0` the trip-count closed form
  `⌈K/(BLOCK_K·SPLIT_K)⌉` is meaningless and the loop would not advance).
* `hcn : 0 < cn_stride` — output-lane footprint injectivity
  (`affine1D_inj`): with `cn_stride = 0` all output lanes collide on one
  cell, the per-lane readback would be last-writer-wins and the statement
  false. Torch strides of a non-degenerate output are ≥ 1.

Division of labor: the `SPLIT_K > 1` **atomic face** is a read-modify-write
(`tl.atomic_add`), not a terminal single-store fold, hence outside this
skin's S1 genre — its per-program `out-before + contribution` obligation
remains conjunct (5) of the exact five-way summary
`bgmv_shrink_kernel_output_summary_general`, which is retained unchanged
(as are the faithful `.int` sentinel surface and its write-free skip path,
conjuncts (1)–(4)). -/
specification bgmv_shrink_store_io_correctness (R : RoundingModel)
    (input_ptr lora_ptr out_ptr : RegionName)
    (N K : Nat) (lora_indices : Region .nat) (scaling : ℝ)
    (xm_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
      BLOCK_N BLOCK_K SPLIT_K : Nat)
    (hS : 0 < BLOCK_K * SPLIT_K) (hcn : 0 < cn_stride) :
    bgmvShrinkStoreIO input_ptr lora_ptr out_ptr N K lora_indices scaling
        xm_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
        BLOCK_N BLOCK_K SPLIT_K
      ⊨[R] fun pid₀ _ _ _ xs ys j =>
        scaling * ∑ t : Fin (numKIters K (BLOCK_K * SPLIT_K)),
          ∑ e : Fin BLOCK_K,
            if t.val * (BLOCK_K * SPLIT_K) + (e.val + pid₀ * BLOCK_K) < K
            then xs t e * ys t (bStreamLane BLOCK_N BLOCK_K j e)
            else 0 := by
  refine StreamMetaMasked3DKernelIO₂.ImplementsR.intro _ ?_ ?_ ?_
  · exact shrink_store_flattenOk input_ptr lora_ptr out_ptr N K lora_indices
      scaling xm_stride l0_stride lora_k_stride lora_n_stride cm_stride
      cn_stride BLOCK_N BLOCK_K SPLIT_K
  · -- safety walk
    intro bounds s m xs ys hm _hx _hy hbm hbr1 hbr2 hbw
    simp only [bgmvShrinkStoreIO] at hm hbm hbr1 hbr2 hbw ⊢
    have hli : loraIdx s lora_indices = m ⟨0, Nat.one_pos⟩ :=
      hm ⟨0, Nat.one_pos⟩
    refine shrink_store_traceSafeR R bounds input_ptr lora_ptr out_ptr N K
      lora_indices scaling xm_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride BLOCK_N BLOCK_K SPLIT_K s hS (hbm ⟨0, Nat.one_pos⟩)
      hbr1 ?_ hbw
    intro t l hmask
    rw [hli]
    exact hbr2 t l hmask
  · -- the rounded Hoare triple
    intro s₀ m xs ys _hundef hm hx hy
    simp only [bgmvShrinkStoreIO] at hm hx hy ⊢
    have hli : loraIdx s₀ lora_indices = m ⟨0, Nat.one_pos⟩ :=
      hm ⟨0, Nat.one_pos⟩
    obtain ⟨sp, hsp, hP0⟩ := shrink_preLoop input_ptr lora_ptr lora_indices
      N K xm_stride l0_stride lora_k_stride lora_n_stride BLOCK_N BLOCK_K
      SPLIT_K s₀ hS
    obtain ⟨final, sLoop, hLoopStmt, hfinal, hPLoop⟩ :=
      forRange_inv (idx := "k") (start := 0) (stop := K)
        (step := BLOCK_K * SPLIT_K)
        (Nat.pos_iff_ne_zero.mp hS) hP0
        (fun i st hlt hinv => shrink_step input_ptr lora_ptr lora_indices
          N K xm_stride l0_stride lora_k_stride lora_n_stride BLOCK_N BLOCK_K
          SPLIT_K s₀ hS i st hlt hinv)
    have hiF := hPLoop.2.1
    have hboundF := hPLoop.2.2.1
    have hcF : final / (BLOCK_K * SPLIT_K) = numKIters K (BLOCK_K * SPLIT_K) := by
      apply kloop_final_iters K (BLOCK_K * SPLIT_K) _ hS
      · rw [← hiF]; exact hfinal
      · rw [← hiF]; exact hboundF
    obtain ⟨sfin, hTailR, hval, hframe⟩ := shrink_postLoopR R input_ptr
      lora_ptr out_ptr lora_indices scaling N K xm_stride l0_stride
      lora_k_stride lora_n_stride cm_stride cn_stride BLOCK_N BLOCK_K SPLIT_K
      hcn s₀ sLoop final hPLoop
    have hLoopR : stepStmtR R (Stmt.forRange "k" 0 K (BLOCK_K * SPLIT_K)
          (shrinkLoopBody N K lora_k_stride lora_n_stride BLOCK_N BLOCK_K)) sp
        = some sLoop := by
      rw [stepStmtR_forRange,
        stepForRangeAuxR_castFree R _
          (shrinkBody_castFree R N K lora_k_stride lora_n_stride BLOCK_N
            BLOCK_K) "k",
        ← stepForRangeAux.forRange_unfold]
      exact hLoopStmt
    refine ⟨sfin, ?_, ?_, ?_⟩
    · show execR R (bgmv_shrink_store_surface input_ptr lora_ptr out_ptr N K
          lora_indices scaling xm_stride l0_stride lora_k_stride lora_n_stride
          cm_stride cn_stride BLOCK_N BLOCK_K SPLIT_K).toAlgKernel s₀
        = some sfin
      unfold execR
      rw [store_body_split input_ptr lora_ptr out_ptr N K lora_indices scaling
          xm_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
          BLOCK_N BLOCK_K SPLIT_K,
        store_prologue_split input_ptr lora_ptr out_ptr N K lora_indices
          scaling xm_stride l0_stride lora_k_stride lora_n_stride cm_stride
          cn_stride BLOCK_N BLOCK_K SPLIT_K,
        stepStmtsR_append, shrinkPrologue_castFree, hsp, Option.bind_some,
        stepStmtsR_cons_some hLoopR]
      exact hTailR
    · intro j hj
      have hcell := hval j hj
      rw [hcF] at hcell
      show sfin.readMemAs .real out_ptr
          (s₀.pids 1 * cm_stride + j.val * cn_stride) = _
      rw [show s₀.pids 1 * cm_stride + j.val * cn_stride
          = outOff s₀ cm_stride cn_stride j.val from rfl,
        readMemAs_real_of_cell hcell, R.round_real_apply,
        shrinkSpec_eq_streamSum input_ptr lora_ptr lora_indices N K xm_stride
          l0_stride lora_k_stride lora_n_stride BLOCK_N BLOCK_K SPLIT_K s₀
          (m ⟨0, Nat.one_pos⟩) hli xs ys hx hy j hj]
      exact congrArg _ (mul_comm _ _)
    · intro r o hcond
      refine hframe r o ?_
      rcases hcond with hne | hno
      · exact Or.inl hne
      · exact Or.inr fun n hn => hno n hn

end VeriTile.Bench.TritonBenchG.BgmvShrinkKernel
