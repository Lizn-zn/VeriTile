import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.Kernel

/-!
# `bmm_chunk_fwd` — closed-form batched-matmul correctness

`_bmm_chunk_fwd_kernel` is the chunked batched matmul forward of Mamba-style
SSMs: for each `(chunk-row, chunk-col)` program in a `(batch, chunk·group)` grid
it accumulates `acc += tl.dot(a, b)` over `cdiv(K, BLOCK_SIZE_K)` K-blocks (with
per-block K-tail masking and a `chunk_size_limit` row/col mask on the loads),
optionally zeroes cross-sequence pairs (`HAS_SEQ_IDX`) and the causal upper
triangle (`IS_CAUSAL`), and stores the `BLOCK_SIZE_M×BLOCK_SIZE_N` chunk tile to
`out` under a `(m<chunk_size)&(n<chunk_size)` mask.

This file proves the **full kernel** for the genuine batched-matmul
configuration (`IS_CAUSAL = false`, `HAS_SEQ_IDX = false` — Python test cases 1
and 4) correct against a genuine mathematical reference: every active output
cell `Out[batch, chunk, head, i, j]` of the computed tile equals
`Σ_{k < K} A[batch,chunk,head, i, k] · B[batch,chunk,head, k, j]` over `ℝ`,
where `K = BLOCK_SIZE_K · numKBlocks` is the contracted dimension. This is NOT
the kernel's own emitted value — it is the independent closed-form per-program
`Σ_k A·B` batched-matmul reference, derived from the loaded `A`/`B` tiles under
the kernel's own strided batch/chunk/head/row/col addressing.

## Proof architecture

```
bmm_chunk_fwd_closed_form_correct                 ← TOP THEOREM (ComputeCorrect.Realizes)
  └─ bmm_exec_closed_form                         ← exec-side closed form (every active cell = ∑_k A·B)
       ├─ bmm_preLoop      (P 0: acc = 0, a/b pointers seeded with batch offset)
       ├─ bmm_step         (one K-block: acc += tl.dot advances the partial sum)
       ├─ bmm_loop         (forRangeDyn drives the K-loop via forRangeAux_inv)
       └─ bmm_postLoop     (final masked store = the closed form)
```

`bmm_chunk_fwd_surface_toAlgorithm_supported` (and the four per-case lowerings)
additionally witness that the **fully general** surface — with causal gating,
optional `seq_idx` filtering, and the destination dtype cast — lowers to the
algorithm layer for every constexpr combination.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float; the runtime `dot_dtype` and
`out_ptr.dtype.element_ty` casts collapse to the real carrier).
`@triton.autotune` / `num_warps` / `num_stages` are not modeled. The host launch
(the 3D grid over chunk tiles / batch / chunk·groups, and how per-program writes
compose into one buffer) is the trusted boundary; the per-program statement is
universally quantified over `s`, so it covers every program of the grid. The
layout contract is the kernel's own strided pointer arithmetic:
`a[i,k]` at `A + batchOff_a + (PM·BM+i)·stride_a_seqlen + k·stride_ak`,
`b[k,j]` at `B + batchOff_b + k·stride_bk + (PN·BN+j)·stride_b_seqlen`,
`out[i,j]` at `Out + batchOff_out + (PM·BM+i)·stride_outm + (PN·BN+j)·stride_outn`,
where `batchOff_x = pid_b·stride_x_batch + pid_c·chunk_size·stride_x_seqlen +
pid_h·stride_x_head` (`stride_out_chunk`/`stride_out_head` for `out`).

Preconditions for the general theorem: `0 < BLOCK_SIZE_K` and `K` divisible into
`numKBlocks` full K-blocks (so each per-block K-tail load mask is all-true); all
tile rows/cols in-bounds (`PM·BM+i < chunk_size`, `PN·BN+j < chunk_size`, making
both the `chunk_size_limit` load masks and the store mask all-true);
output-address injectivity; clean initial `undef`.
-/

namespace VeriTile.Bench.TritonBenchG.BmmChunkFwd

open VeriTile.Triton

set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

/-! ## Fully general surface (all constexpr configurations) -/

/-- Faithful transcription of `bmm_chunk_fwd.py`'s `_bmm_chunk_fwd_kernel`.

This covers both `HAS_SEQ_IDX=false` and `HAS_SEQ_IDX=true`; the sequence-index
loads use an `Int` region so the Python `other=-1` / `other=-2` sentinels are
represented directly. The causal early-return is represented as a guard around
the active body. The source kernel's runtime `dot_dtype` cast is preserved as a
surface dtype annotation on the `a`/`b` loads. -/
def bmm_chunk_fwd_surface
    (a_ptr b_ptr out_ptr : RegionName) (seq_idx_ptr : Region .int)
    (seqlen chunk_size K ngroups
      stride_a_batch stride_a_seqlen stride_a_head stride_ak
      stride_b_batch stride_b_seqlen stride_b_head stride_bk
      stride_out_batch stride_out_chunk stride_out_head stride_outm stride_outn
      stride_seq_idx_batch stride_seq_idx_seqlen
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (dot_dtype : TileDType)
    (IS_CAUSAL HAS_SEQ_IDX : Bool) :
    ComputeKernel := triton {
  pid_b = tl.program_id(axis=1)
  pid_ch = tl.program_id(axis=2)
  pid_c = pid_ch // $(ngroups)
  pid_h = pid_ch - pid_c * $(ngroups)
  num_pid_n = tl.cdiv($(chunk_size), $(BLOCK_SIZE_N))
  pid_m = tl.program_id(axis=0) // num_pid_n
  pid_n = tl.program_id(axis=0) % num_pid_n
  active_block = true
  if IS_CAUSAL {
    active_block = pid_n * $(BLOCK_SIZE_N) < (pid_m + $(1)) * $(BLOCK_SIZE_M)
  }
  if active_block {

  a_ptr += pid_b * $(stride_a_batch) +
    pid_c * $(chunk_size) * $(stride_a_seqlen) + pid_h * $(stride_a_head)
  b_ptr += pid_b * $(stride_b_batch) +
    pid_c * $(chunk_size) * $(stride_b_seqlen) + pid_h * $(stride_b_head)
  if HAS_SEQ_IDX {
    seq_idx_ptr += pid_b * $(stride_seq_idx_batch) +
      pid_c * $(chunk_size) * $(stride_seq_idx_seqlen)
  }

  offs_m = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_n = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  offs_k = tl.arange(0, $(BLOCK_SIZE_K))
  a_ptrs = a_ptr + offs_m[:, None] * $(stride_a_seqlen) +
    offs_k[None, :] * $(stride_ak)
  b_ptrs = b_ptr + offs_k[:, None] * $(stride_bk) +
    offs_n[None, :] * $(stride_b_seqlen)
  chunk_size_limit = min($(chunk_size), $(seqlen) - pid_c * $(chunk_size))

  acc = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
  for k in range($(0), tl.cdiv($(K), $(BLOCK_SIZE_K)), $(1)) {
    a = tl.load(a_ptrs, mask=(offs_m[:, None] < chunk_size_limit) &
      (offs_k[None, :] < $(K) - k * $(BLOCK_SIZE_K)), other=0.0).to(dot_dtype)
    b = tl.load(b_ptrs, mask=(offs_k[:, None] < $(K) - k * $(BLOCK_SIZE_K)) &
      (offs_n[None, :] < chunk_size_limit), other=0.0).to(dot_dtype)
    acc += tl.dot(a, b)
    a_ptrs += $(BLOCK_SIZE_K) * $(stride_ak)
    b_ptrs += $(BLOCK_SIZE_K) * $(stride_bk)
  }

  offs_m = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_n = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  if HAS_SEQ_IDX {
    chunk_size_limit = min($(chunk_size), $(seqlen) - pid_c * $(chunk_size))
    seq_idx_m = tl.load(seq_idx_ptr + offs_m * $(stride_seq_idx_seqlen),
      mask=offs_m < chunk_size_limit, other=-1)
    seq_idx_n = tl.load(seq_idx_ptr + offs_n * $(stride_seq_idx_seqlen),
      mask=offs_n < chunk_size_limit, other=-2)
    acc = tl.where(seq_idx_m[:, None] == seq_idx_n[None, :], acc, 0.0)
  }
  out = (acc).to(out_ptr.dtype.element_ty)
  out_ptr += pid_b * $(stride_out_batch) +
    pid_c * $(stride_out_chunk) + pid_h * $(stride_out_head)
  out_ptrs = out_ptr + $(stride_outm) * offs_m[:, None] + offs_n[None, :] * $(stride_outn)
  tl.store(out_ptrs, out, mask=(offs_m[:, None] < $(chunk_size)) &
    (offs_n[None, :] < $(chunk_size)))
  }
}

/-- The full BMM chunk forward surface lowers to the algorithm layer, including
causal gating, optional sequence-index filtering, K-loop dot accumulation, and
the destination dtype cast. -/
theorem bmm_chunk_fwd_surface_toAlgorithm_supported
    (a_ptr b_ptr out_ptr : RegionName) (seq_idx_ptr : Region .int)
    (seqlen chunk_size K ngroups
      stride_a_batch stride_a_seqlen stride_a_head stride_ak
      stride_b_batch stride_b_seqlen stride_b_head stride_bk
      stride_out_batch stride_out_chunk stride_out_head stride_outm stride_outn
      stride_seq_idx_batch stride_seq_idx_seqlen
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (dot_dtype : TileDType)
    (IS_CAUSAL HAS_SEQ_IDX : Bool) :
    ∃ alg,
      (bmm_chunk_fwd_surface a_ptr b_ptr out_ptr seq_idx_ptr seqlen chunk_size
        K ngroups stride_a_batch stride_a_seqlen stride_a_head stride_ak
        stride_b_batch stride_b_seqlen stride_b_head stride_bk stride_out_batch
        stride_out_chunk stride_out_head stride_outm stride_outn
        stride_seq_idx_batch stride_seq_idx_seqlen BLOCK_SIZE_M BLOCK_SIZE_N
        BLOCK_SIZE_K dot_dtype IS_CAUSAL HAS_SEQ_IDX).toAlgorithm? =
        Except.ok alg := by
  simp [bmm_chunk_fwd_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-! ## Genuine batched-matmul surface (`IS_CAUSAL = false`, `HAS_SEQ_IDX = false`)

This is the faithful transcription of `_bmm_chunk_fwd_kernel` for the pure
batched-matmul configuration (Python test cases 1 and 4): no causal early
return, no sequence-index filtering. Since `out_ptr.dtype.element_ty` collapses
to the real carrier the `out` store is the real-valued accumulator. -/

/-- The genuine batched-matmul surface: chunked `acc += tl.dot(a, b)` with the
kernel's batch/chunk/head pointer offsets, K-block dot loop, per-block K-tail and
`chunk_size` row/col load masks, and the masked output store. -/
def bmm_matmul_surface
    (a_ptr b_ptr out_ptr : RegionName)
    (chunk_size K ngroups
      stride_a_batch stride_a_seqlen stride_a_head stride_ak
      stride_b_batch stride_b_seqlen stride_b_head stride_bk
      stride_out_batch stride_out_chunk stride_out_head stride_outm stride_outn
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat) :
    ComputeKernel := triton {
  pid_b = tl.program_id(axis=1)
  pid_ch = tl.program_id(axis=2)
  pid_c = pid_ch // $(ngroups)
  pid_h = pid_ch - pid_c * $(ngroups)
  num_pid_n = tl.cdiv($(chunk_size), $(BLOCK_SIZE_N))
  pid_m = tl.program_id(axis=0) // num_pid_n
  pid_n = tl.program_id(axis=0) % num_pid_n
  a_ptr += pid_b * $(stride_a_batch) +
    pid_c * $(chunk_size) * $(stride_a_seqlen) + pid_h * $(stride_a_head)
  b_ptr += pid_b * $(stride_b_batch) +
    pid_c * $(chunk_size) * $(stride_b_seqlen) + pid_h * $(stride_b_head)
  offs_m = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_n = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  offs_k = tl.arange(0, $(BLOCK_SIZE_K))
  a_ptrs = a_ptr + offs_m[:, None] * $(stride_a_seqlen) + offs_k[None, :] * $(stride_ak)
  b_ptrs = b_ptr + offs_k[:, None] * $(stride_bk) + offs_n[None, :] * $(stride_b_seqlen)
  acc = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
  for k in range($(0), tl.cdiv($(K), $(BLOCK_SIZE_K)), $(1)) {
    a = tl.load(a_ptrs, mask=(offs_m[:, None] < $(chunk_size)) &
      (offs_k[None, :] < $(K) - k * $(BLOCK_SIZE_K)), other=0.0)
    b = tl.load(b_ptrs, mask=(offs_k[:, None] < $(K) - k * $(BLOCK_SIZE_K)) &
      (offs_n[None, :] < $(chunk_size)), other=0.0)
    acc += tl.dot(a, b)
    a_ptrs += $(BLOCK_SIZE_K) * $(stride_ak)
    b_ptrs += $(BLOCK_SIZE_K) * $(stride_bk)
  }
  offs_m = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_n = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  out = (acc).to(out_ptr.dtype.element_ty)
  out_ptr += pid_b * $(stride_out_batch) +
    pid_c * $(stride_out_chunk) + pid_h * $(stride_out_head)
  out_ptrs = out_ptr + $(stride_outm) * offs_m[:, None] + offs_n[None, :] * $(stride_outn)
  tl.store(out_ptrs, out, mask=(offs_m[:, None] < $(chunk_size)) &
    (offs_n[None, :] < $(chunk_size)))
}

/-- The genuine batched-matmul surface lowers to the algorithm layer. -/
theorem bmm_matmul_surface_toAlgorithm_supported
    (A B Out : RegionName)
    (chunk_size K ngroups
      stride_a_batch stride_a_seqlen stride_a_head stride_ak
      stride_b_batch stride_b_seqlen stride_b_head stride_bk
      stride_out_batch stride_out_chunk stride_out_head stride_outm stride_outn
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat) :
    ∃ alg, (bmm_matmul_surface A B Out chunk_size K ngroups
      stride_a_batch stride_a_seqlen stride_a_head stride_ak
      stride_b_batch stride_b_seqlen stride_b_head stride_bk
      stride_out_batch stride_out_chunk stride_out_head stride_outm stride_outn
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K).toAlgorithm? = Except.ok alg := by
  simp [bmm_matmul_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-! ## exec-stepping helpers -/

theorem evalOp_floorDiv {dtype a b shape} (h : IntegralDType dtype)
    (bc : Broadcast a b shape) (x : Op dtype a) (y : Op dtype b) (s : BlockState) :
    evalOp (.floorDiv h bc x y) s = (do
      let vx ← evalOp x s; let vy ← evalOp y s; some (Tile.bop h.floorDiv bc vx vy)) := by
  simp [evalOp]

theorem evalOp_mod {dtype a b shape} (h : IntegralDType dtype)
    (bc : Broadcast a b shape) (x : Op dtype a) (y : Op dtype b) (s : BlockState) :
    evalOp (.mod h bc x y) s = (do
      let vx ← evalOp x s; let vy ← evalOp y s; some (Tile.bop h.mod bc vx vy)) := by
  simp [evalOp]

theorem evalOp_boolAnd {a b shape} (bc : Broadcast a b shape)
    (x : Op .bool a) (y : Op .bool b) (s : BlockState) :
    evalOp (.boolAnd bc x y) s = (do
      let vx ← evalOp x s; let vy ← evalOp y s;
      some (Tile.bop (fun p q : Bool => p && q) bc vx vy)) := by
  simp [evalOp]

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

theorem evalOp_ptrAdd {a b shape} (bc : Broadcast a b shape)
    (ptr : Op .ptr a) (off : Op .nat b) (s : BlockState) :
    evalOp (.ptrAdd bc ptr off) s = (do
      let ptrs ← evalOp ptr s; let offs ← evalOp off s;
      some (Tile.ptrAdd bc ptrs offs)) := by simp [evalOp]

theorem evalOp_ptrBase (region : RegionName) (s : BlockState) :
    evalOp (.ptrBase region) s = some (Tile.scalar (region.cast, 0)) := by simp [evalOp]

/-! ## Index/value abbreviations for the batched-matmul spec -/

/-- Ceiling division `⌈a / b⌉`, matching Triton's `tl.cdiv`. -/
def cdiv (a b : Nat) : Nat := (a + b - 1) / b

/-- The kernel's `num_pid_n = cdiv(chunk_size, BLOCK_SIZE_N)`. -/
def numPidN (chunk_size BN : Nat) : Nat := cdiv chunk_size BN

/-- `pid_c = pid_ch // ngroups` (program axis 2 split by `ngroups`). -/
def pidC (PCH ngroups : Nat) : Nat := PCH / ngroups

/-- `pid_h = pid_ch - pid_c · ngroups`. -/
def pidH (PCH ngroups : Nat) : Nat := PCH - PCH / ngroups * ngroups

/-- `pid_m = program_id(0) // num_pid_n`. -/
def pidM (P0 chunk_size BN : Nat) : Nat := P0 / numPidN chunk_size BN

/-- `pid_n = program_id(0) % num_pid_n`. -/
def pidN (P0 chunk_size BN : Nat) : Nat := P0 % numPidN chunk_size BN

/-- Global chunk row of tile lane `i`: `PM · BM + i`. -/
def rowIndex (PM BM : Nat) (i : Fin BM) : Nat := PM * BM + i.val

/-- Global chunk col of tile lane `j`: `PN · BN + j`. -/
def colIndex (PN BN : Nat) (j : Fin BN) : Nat := PN * BN + j.val

/-- The kernel's `a`/`b` batch+chunk+head base offset:
`pid_b·stride_batch + pid_c·chunk_size·stride_seqlen + pid_h·stride_head`. -/
def batchOff (PB PC PH chunk_size stride_batch stride_seqlen stride_head : Nat) : Nat :=
  PB * stride_batch + PC * chunk_size * stride_seqlen + PH * stride_head

/-- `A[i, k] = readMem A (batchOff_a + (PM·BM+i)·stride_a_seqlen + k·stride_ak)`. -/
noncomputable def aElem (s : BlockState) (A : RegionName)
    (PB PC PH PM BM chunk_size SAB SAS SAH SAK : Nat) (i : Fin BM) (k : Nat) : ℝ :=
  s.readMem A (batchOff PB PC PH chunk_size SAB SAS SAH + rowIndex PM BM i * SAS + k * SAK)

/-- `B[k, j] = readMem B (batchOff_b + k·stride_bk + (PN·BN+j)·stride_b_seqlen)`. -/
noncomputable def bElem (s : BlockState) (B : RegionName)
    (PB PC PH PN BN chunk_size SBB SBS SBH SBK : Nat) (j : Fin BN) (k : Nat) : ℝ :=
  s.readMem B (batchOff PB PC PH chunk_size SBB SBS SBH + k * SBK + colIndex PN BN j * SBS)

/-- **Genuine batched-matmul spec**:
`Out[i,j] = Σ_{k < BLOCK_K·numKBlocks} A[i,k] · B[k,j]`. -/
noncomputable def bmmSpec (s : BlockState) (A B : RegionName)
    (PB PC PH PM PN BM BN chunk_size SAB SAS SAH SAK SBB SBS SBH SBK
      BLOCK_K numKBlocks : Nat) (i : Fin BM) (j : Fin BN) : ℝ :=
  (Finset.range (BLOCK_K * numKBlocks)).sum
    (fun k => aElem s A PB PC PH PM BM chunk_size SAB SAS SAH SAK i k
      * bElem s B PB PC PH PN BN chunk_size SBB SBS SBH SBK j k)

/-- Partial accumulator after `c` K-blocks: `Σ_{k < c·BLOCK_K} A·B`. -/
noncomputable def accPartial (s : BlockState) (A B : RegionName)
    (PB PC PH PM PN BM BN chunk_size SAB SAS SAH SAK SBB SBS SBH SBK
      BLOCK_K : Nat) (i : Fin BM) (j : Fin BN) (c : Nat) : ℝ :=
  (Finset.range (c * BLOCK_K)).sum
    (fun k => aElem s A PB PC PH PM BM chunk_size SAB SAS SAH SAK i k
      * bElem s B PB PC PH PN BN chunk_size SBB SBS SBH SBK j k)

/-- One-block step of the partial accumulator. -/
theorem accPartial_succ (s : BlockState) (A B : RegionName)
    (PB PC PH PM PN BM BN chunk_size SAB SAS SAH SAK SBB SBS SBH SBK
      BLOCK_K : Nat) (i : Fin BM) (j : Fin BN) (c : Nat) :
    accPartial s A B PB PC PH PM PN BM BN chunk_size SAB SAS SAH SAK SBB SBS SBH SBK BLOCK_K i j (c + 1)
      = accPartial s A B PB PC PH PM PN BM BN chunk_size SAB SAS SAH SAK SBB SBS SBH SBK BLOCK_K i j c
        + (Finset.univ.sum fun e : Fin BLOCK_K =>
            aElem s A PB PC PH PM BM chunk_size SAB SAS SAH SAK i (c * BLOCK_K + e.val)
              * bElem s B PB PC PH PN BN chunk_size SBB SBS SBH SBK j (c * BLOCK_K + e.val)) := by
  unfold accPartial
  have h : (c + 1) * BLOCK_K = c * BLOCK_K + BLOCK_K := by ring
  rw [h, Finset.sum_range_add]
  congr 1
  rw [Finset.sum_range fun e => aElem s A PB PC PH PM BM chunk_size SAB SAS SAH SAK i (c * BLOCK_K + e)
        * bElem s B PB PC PH PN BN chunk_size SBB SBS SBH SBK j (c * BLOCK_K + e)]

/-! ## Pointer / load / dot eval lemmas -/

/-- The scalar base-pointer add `A + batchOff` evaluates to the scalar pointer
tile `(A, batchOff)`. -/
theorem aptr_base_eval (s : BlockState) (A : RegionName) (PB PC PH chunk_size SAB SAS SAH : Nat)
    (hpb : s.regs .nat [] "pid_b" = some (Tile.scalar PB))
    (hpc : s.regs .nat [] "pid_c" = some (Tile.scalar PC))
    (hph : s.regs .nat [] "pid_h" = some (Tile.scalar PH)) :
    evalOp (Op.ptrAdd Broadcast.nil (Op.ptrBase A)
      (Op.add .nat Broadcast.nil
        (Op.add .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_b") (Op.constNat SAB))
          (Op.mul .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_c") (Op.constNat chunk_size)) (Op.constNat SAS)))
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_h") (Op.constNat SAH)))) s
      = some (Tile.scalar (A.cast, batchOff PB PC PH chunk_size SAB SAS SAH)) := by
  rw [evalOp_ptrAdd, evalOp_ptrBase]
  simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref, hpb, hpc, hph,
    Option.bind, Option.bind_some, Option.bind_eq_bind]
  refine congrArg some ?_
  simp only [Tile.ptrAdd, Tile.bop, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.add, NumericDType.mul, batchOff]
  ring_nf

/-- `a_ptrs` eval from a scalar `a_ptr` register: cell `(i,e) = (A, base + offs_am i · SAS + e · SAK)`. -/
theorem aptrs_eval (s : BlockState) (A : RegionName) (M K SAS SAK base : Nat) (gm : Fin M → Nat)
    (hbase : s.regs .ptr [] "a_ptr" = some (Tile.scalar (A.cast, base)))
    (hm : s.regs .nat [M] "offs_m" = some (Tile.vec gm))
    (hk : s.regs .nat [K] "offs_k" = some (Tile.vec (fun e : Fin K => e.val))) :
    evalOp (Op.ptrAdd Broadcast.nil.consL.consR
      (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "a_ptr")
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_m")) (Op.constNat SAS)))
      (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [K] "offs_k")) (Op.constNat SAK))) s
      = some (⟨fun idx : TileIndex [M, K] => (A.cast, base + gm idx.1 * SAS + idx.2.1.val * SAK)⟩ : Tile .ptr [M, K]) := by
  rw [evalOp_ptrAdd, evalOp_ptrAdd]
  simp only [evalOp_mul, evalOp_constNat, evalOp_ref, evalOp_expandDim_one_nat,
    evalOp_expandDim_zero_nat, hbase, hm, hk, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec]
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec, NumericDType.add, NumericDType.mul]

/-- `b_ptrs` eval from a scalar `b_ptr` register: cell `(e,j) = (B, base + e · SBK + offs_bn j · SBS)`. -/
theorem bptrs_eval (s : BlockState) (B : RegionName) (K N SBK SBS base : Nat) (gn : Fin N → Nat)
    (hbase : s.regs .ptr [] "b_ptr" = some (Tile.scalar (B.cast, base)))
    (hk : s.regs .nat [K] "offs_k" = some (Tile.vec (fun e : Fin K => e.val)))
    (hn : s.regs .nat [N] "offs_n" = some (Tile.vec gn)) :
    evalOp (Op.ptrAdd Broadcast.nil.consL.consR
      (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "b_ptr")
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [K] "offs_k")) (Op.constNat SBK)))
      (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [N] "offs_n")) (Op.constNat SBS))) s
      = some (⟨fun idx : TileIndex [K, N] => (B.cast, base + idx.1.val * SBK + gn idx.2.1 * SBS)⟩ : Tile .ptr [K, N]) := by
  rw [evalOp_ptrAdd, evalOp_ptrAdd]
  simp only [evalOp_mul, evalOp_constNat, evalOp_ref, evalOp_expandDim_one_nat,
    evalOp_expandDim_zero_nat, hbase, hk, hn, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec]
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec, NumericDType.add, NumericDType.mul]

/-- `acc` init eval: `tl.zeros` → the all-`0` tile. -/
theorem acc_init_eval (s : BlockState) (M N : Nat) :
    evalOp (Op.full [M, N] (Op.const 0)) s
      = some (⟨fun _ : TileIndex [M, N] => some (0 : ℝ)⟩ : Tile .real [M, N]) := by
  simp [evalOp_full, evalOp_const, Option.bind]

/-- `a_ptrs += BLOCK_SIZE_K · stride_ak` eval (scalar add via `scalarR`). -/
theorem aptr_adv_eval (s : BlockState) (M K BK SAK : Nat) (ap : Tile .ptr [M, K])
    (hx : s.regs .ptr [M, K] "a_ptrs" = some ap) :
    evalOp (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [M, K] "a_ptrs")
      (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat SAK))) s
      = some (Tile.ptrAdd Broadcast.scalarR ap (Tile.scalar (BK * SAK))) := by
  rw [evalOp_ptrAdd]
  simp [evalOp_ref, hx, evalOp_mul, evalOp_constNat, NumericDType.mul, Tile.bop]

/-- `b_ptrs += BLOCK_SIZE_K · stride_bk` eval. -/
theorem bptr_adv_eval (s : BlockState) (K N BK SBK : Nat) (bp : Tile .ptr [K, N])
    (hy : s.regs .ptr [K, N] "b_ptrs" = some bp) :
    evalOp (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [K, N] "b_ptrs")
      (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat SBK))) s
      = some (Tile.ptrAdd Broadcast.scalarR bp (Tile.scalar (BK * SBK))) := by
  rw [evalOp_ptrAdd]
  simp [evalOp_ref, hy, evalOp_mul, evalOp_constNat, NumericDType.mul, Tile.bop]

/-- The dot of two all-`some` loaded tiles, lane `(i,j)`, equals
`some (Σ_e fx e · fy e)`. -/
theorem dot_ab (M K N : Nat) (x : Tile .real [M, K]) (y : Tile .real [K, N])
    (i : Fin M) (j : Fin N) (fx : Fin K → ℝ) (fy : Fin K → ℝ)
    (hx : ∀ e : Fin K, x.data (i, e, PUnit.unit) = some (fx e))
    (hy : ∀ e : Fin K, y.data (e, j, PUnit.unit) = some (fy e)) :
    (Tile.dot [] x y).data (i, j, PUnit.unit)
      = some (Finset.univ.sum fun e : Fin K => fx e * fy e) := by
  rw [Tile.dot_nil_data]
  rw [show (@Finset.sum (Fin K) (WithBot ℝ) _ Finset.univ
        (fun e => Option.map₂ (· * ·) (x.data (i, e, PUnit.unit)) (y.data (e, j, PUnit.unit))))
      = @Finset.sum (Fin K) (WithBot ℝ) _ Finset.univ (fun e => (some (fx e * fy e) : WithBot ℝ))
      from Finset.sum_congr rfl (fun e _ => by rw [hx e, hy e]; rfl)]
  show (Finset.univ.sum fun e => ((fx e * fy e : ℝ) : WithBot ℝ)) = _
  rw [← WithBot.coe_sum]; rfl

/-- **`acc = acc + tl.dot(a, b)` statement eval.** -/
theorem accdot_op_eval (M K N : Nat) (st : BlockState)
    (zt : Tile .real [M, N]) (xt : Tile .real [M, K]) (yt : Tile .real [K, N])
    (hz : st.regs .real [M, N] "acc" = some zt)
    (hx : st.regs .real [M, K] "a" = some xt)
    (hy : st.regs .real [K, N] "b" = some yt) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [M, N] "acc")
        (Op.dot (batch := []) (Op.ref .real [M, K] "a") (Op.ref .real [K, N] "b"))) st
      = some (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          zt (Tile.dot [] xt yt)) := by
  have hd : evalOp (Op.dot (batch := []) (Op.ref .real [M, K] "a")
        (Op.ref .real [K, N] "b")) st = some (Tile.dot [] xt yt) := by
    rw [evalOp_dot]; simp [hx, hy]
  rw [evalOp_add]
  simp only [evalOp_ref, hz, bind, Option.bind_some]
  erw [hd]
  rfl

/-- `acc + dot` lane `(i,j)`: `some (zv + dv)`. -/
theorem accadd_eval (M N : Nat) (zt dt : Tile .real [M, N]) (i : Fin M) (j : Fin N) (zv dv : ℝ)
    (hz : zt.data (i, j, PUnit.unit) = some zv) (hd : dt.data (i, j, PUnit.unit) = some dv) :
    (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) zt dt).data
        (i, j, PUnit.unit) = some (zv + dv) := by
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, hz, hd, NumericDType.add,
    WithBot.realAdd, Option.map₂, Option.bind, Option.map]

/-- A `.ptr` masked load with `.maskOther`, when every mask lane is `true`,
reads `readMem` at each pointer (clean `undef`). -/
theorem load_ptr_maskOther_alltrue {shape : TileShape}
    (ptrOp : Op .ptr shape) (maskOp : Op .bool shape) (otherOp : Op .real shape)
    (s : BlockState) (ptrs : Tile .ptr shape) (mtile : Tile .bool shape) (otile : Tile .real shape)
    (hp : evalOp ptrOp s = some ptrs)
    (hm : evalOp maskOp s = some mtile)
    (ho : evalOp otherOp s = some otile)
    (hall : ∀ i, mtile.data i = Bool.true) :
    evalOp (.load .real (.ptr ptrOp) (.maskOther maskOp otherOp)) s
      = some ⟨fun i => some (s.readMem (ptrs.data i).1 (ptrs.data i).2)⟩ := by
  unfold evalOp
  simp only [hp, hm, ho, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext i
  simp only [hall i, if_true, BlockState.readMemValue_real]

/-- The constant `other = 0.0` broadcast tile. -/
theorem other_broadcast_eval (s : BlockState) (shape : TileShape) :
    evalOp ((Op.const 0.0).broadcast shape) s = some (⟨fun _ : TileIndex shape => some (0.0 : ℝ)⟩ : Tile .real shape) := by
  simp [evalOp, evalOp_const, Option.bind]

/-- The a-load mask `(offs_m < chunk_size) & (offs_k < K - k·BK)`: every lane is
`true` when each tile row is `< chunk_size` and `c·BK + e < K`. -/
theorem amask_alltrue (s : BlockState) (BM BK chunk_size K c : Nat) (gm : Fin BM → Nat)
    (hm : s.regs .nat [BM] "offs_m" = some (Tile.vec gm))
    (hk : s.regs .nat [BK] "offs_k" = some (Tile.vec (fun e : Fin BK => e.val)))
    (hkk : s.regs .nat [] "k" = some (Tile.scalar c))
    (hmlt : ∀ i : Fin BM, gm i < chunk_size)
    (hlt : ∀ e : Fin BK, e.val < K - c * BK) :
    ∃ mtile : Tile .bool [BM, BK],
      evalOp (Op.boolAnd Broadcast.nil.consL.consR
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m"))
          (Op.constNat chunk_size))
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BK] "offs_k"))
          (Op.sub .nat Broadcast.nil (Op.constNat K)
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "k") (Op.constNat BK))))) s = some mtile
      ∧ ∀ i, mtile.data i = Bool.true := by
  refine ⟨_, by
    simp only [evalOp, evalOp.eq_def, hm, hk, hkk, Option.bind, Option.bind_some, Option.bind_eq_bind,
      Tile.expandDim, pure, Option.some.injEq]; rfl, ?_⟩
  intro i
  simp only [Tile.bop, Tile.cop, Tile.expandDim, Tile.vec, Tile.scalar, ComparableDType.lt,
    NumericDType.sub, NumericDType.mul, Broadcast.leftIndex, Broadcast.rightIndex,
    TileShape.dropInsertedIndex, Bool.and_eq_true, decide_eq_true_eq]
  exact ⟨by simpa using hmlt _, by simpa using hlt _⟩

/-- The b-load mask `(offs_k < K - k·BK) & (offs_n < chunk_size)`: every lane is
`true` under the same conditions. -/
theorem bmask_alltrue (s : BlockState) (BN BK chunk_size K c : Nat) (gn : Fin BN → Nat)
    (hk : s.regs .nat [BK] "offs_k" = some (Tile.vec (fun e : Fin BK => e.val)))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec gn))
    (hkk : s.regs .nat [] "k" = some (Tile.scalar c))
    (hnlt : ∀ j : Fin BN, gn j < chunk_size)
    (hlt : ∀ e : Fin BK, e.val < K - c * BK) :
    ∃ mtile : Tile .bool [BK, BN],
      evalOp (Op.boolAnd Broadcast.nil.consL.consR
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "offs_k"))
          (Op.sub .nat Broadcast.nil (Op.constNat K)
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "k") (Op.constNat BK))))
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n"))
          (Op.constNat chunk_size))) s = some mtile
      ∧ ∀ i, mtile.data i = Bool.true := by
  refine ⟨_, by
    simp only [evalOp, evalOp.eq_def, hk, hn, hkk, Option.bind, Option.bind_some, Option.bind_eq_bind,
      Tile.expandDim, pure, Option.some.injEq]; rfl, ?_⟩
  intro i
  simp only [Tile.bop, Tile.cop, Tile.expandDim, Tile.vec, Tile.scalar, ComparableDType.lt,
    NumericDType.sub, NumericDType.mul, Broadcast.leftIndex, Broadcast.rightIndex,
    TileShape.dropInsertedIndex, Bool.and_eq_true, decide_eq_true_eq]
  exact ⟨by simpa using hlt _, by simpa using hnlt _⟩

/-! ## Prologue: the scalar pids and index vectors -/

/-- **preLoop scalars** (statements 0–6): the pid derivation plus the three
index vectors. Steps to a state where `pid_b/pid_c/pid_h/pid_m/pid_n` hold their
derived values and `offs_m/offs_n/offs_k` hold their readbacks. -/
theorem preLoop_scalars (s : BlockState) (chunk_size ngroups BM BN BK : Nat) :
    ∃ s7,
      stepStmts
        [ Stmt.assign .nat [] "pid_b" (Op.programId 1),
          Stmt.assign .nat [] "pid_ch" (Op.programId 2),
          Stmt.assign .nat [] "pid_c"
            (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "pid_ch") (Op.constNat ngroups)),
          Stmt.assign .nat [] "pid_h"
            (Op.sub .nat Broadcast.nil (Op.ref .nat [] "pid_ch")
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_c") (Op.constNat ngroups))),
          Stmt.assign .nat [] "num_pid_n"
            (Op.div .nat Broadcast.nil (Op.sub .nat Broadcast.nil
              (Op.add .nat Broadcast.nil (Op.constNat chunk_size) (Op.constNat BN)) (Op.constNat 1)) (Op.constNat BN)),
          Stmt.assign .nat [] "pid_m"
            (Op.floorDiv .nat Broadcast.nil (Op.programId 0) (Op.ref .nat [] "num_pid_n")),
          Stmt.assign .nat [] "pid_n"
            (Op.mod .nat Broadcast.nil (Op.programId 0) (Op.ref .nat [] "num_pid_n")) ] s = some s7
      ∧ s7.pids = s.pids
      ∧ s7.regs .nat [] "pid_b" = some (Tile.scalar (s.pids 1))
      ∧ s7.regs .nat [] "pid_c" = some (Tile.scalar (pidC (s.pids 2) ngroups))
      ∧ s7.regs .nat [] "pid_h" = some (Tile.scalar (pidH (s.pids 2) ngroups))
      ∧ s7.regs .nat [] "pid_m" = some (Tile.scalar (pidM (s.pids 0) chunk_size BN))
      ∧ s7.regs .nat [] "pid_n" = some (Tile.scalar (pidN (s.pids 0) chunk_size BN))
      ∧ s7.undef = s.undef
      ∧ s7.mem = s.mem := by
  simp only [pidC, pidH, pidM, pidN, numPidN, cdiv]
  simp [stepStmts, stepStmt, evalOp_floorDiv, evalOp_mod, Option.bind, BlockState.setReg,
    Tile.bop, Tile.cop, Tile.vec, NumericDType.add, NumericDType.mul, NumericDType.div,
    NumericDType.sub, IntegralDType.floorDiv, IntegralDType.mod]

/-! ## Loop body and invariant -/

/-- The 5-statement K-loop body, transcribed. -/
def bmmLoopBody (BM BN BK chunk_size K SAK SBK : Nat) : List Stmt :=
  [ Stmt.assign .real [BM, BK] "a"
      (Op.load .real (.ptr (Op.ref .ptr [BM, BK] "a_ptrs"))
        (.maskOther
          (Op.boolAnd Broadcast.nil.consL.consR
            (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m"))
              (Op.constNat chunk_size))
            (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BK] "offs_k"))
              (Op.sub .nat Broadcast.nil (Op.constNat K)
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "k") (Op.constNat BK)))))
          ((Op.const 0.0).broadcast [BM, BK]))),
    Stmt.assign .real [BK, BN] "b"
      (Op.load .real (.ptr (Op.ref .ptr [BK, BN] "b_ptrs"))
        (.maskOther
          (Op.boolAnd Broadcast.nil.consL.consR
            (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "offs_k"))
              (Op.sub .nat Broadcast.nil (Op.constNat K)
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "k") (Op.constNat BK))))
            (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n"))
              (Op.constNat chunk_size)))
          ((Op.const 0.0).broadcast [BK, BN]))),
    Stmt.assign .real [BM, BN] "acc"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BM, BN] "acc")
        (Op.dot (batch := []) (Op.ref .real [BM, BK] "a") (Op.ref .real [BK, BN] "b"))),
    Stmt.assign .ptr [BM, BK] "a_ptrs"
      (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BM, BK] "a_ptrs")
        (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat SAK))),
    Stmt.assign .ptr [BK, BN] "b_ptrs"
      (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BK, BN] "b_ptrs")
        (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat SBK))) ]

/-- **Loop invariant** (counter `i = c`, the K-block index).

After `c` K-blocks: program ids and `mem`/`undef` fixed; `pid_*` and
`offs_m/offs_n/offs_k` seeded; `a_ptr`/`b_ptr` base pointers seeded; `acc` equals
the partial accumulator `accPartial … c`; and `a_ptrs`/`b_ptrs` advanced by `c`
blocks. -/
noncomputable def bmmInvariant
    (A B : RegionName) (s0 : BlockState)
    (PB PC PH PM PN chunk_size BM BN SAB SAS SAH SAK SBB SBS SBH SBK BLOCK_K numKBlocks : Nat)
    (c : Nat) (s : BlockState) : Prop :=
  s.pids = s0.pids ∧ c ≤ numKBlocks ∧
  (s.regs .real [BM, BN] "acc" = some ⟨fun idx : TileIndex [BM, BN] =>
      some (accPartial s0 A B PB PC PH PM PN BM BN chunk_size SAB SAS SAH SAK SBB SBS SBH SBK BLOCK_K idx.1 idx.2.1 c)⟩) ∧
  (s.regs .nat [] "pid_b" = some (Tile.scalar PB)) ∧
  (s.regs .nat [] "pid_c" = some (Tile.scalar PC)) ∧
  (s.regs .nat [] "pid_h" = some (Tile.scalar PH)) ∧
  (s.regs .nat [] "pid_m" = some (Tile.scalar PM)) ∧
  (s.regs .nat [] "pid_n" = some (Tile.scalar PN)) ∧
  (s.regs .nat [BM] "offs_m" = some (Tile.vec (fun i : Fin BM => PM * BM + i.val))) ∧
  (s.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => PN * BN + j.val))) ∧
  (s.regs .nat [BLOCK_K] "offs_k" = some (Tile.vec (fun e : Fin BLOCK_K => e.val))) ∧
  (s.regs .ptr [BM, BLOCK_K] "a_ptrs" = some ⟨fun idx : TileIndex [BM, BLOCK_K] =>
      (A.cast, batchOff PB PC PH chunk_size SAB SAS SAH + (PM * BM + idx.1.val) * SAS + idx.2.1.val * SAK + c * BLOCK_K * SAK)⟩) ∧
  (s.regs .ptr [BLOCK_K, BN] "b_ptrs" = some ⟨fun idx : TileIndex [BLOCK_K, BN] =>
      (B.cast, batchOff PB PC PH chunk_size SBB SBS SBH + idx.1.val * SBK + (PN * BN + idx.2.1.val) * SBS + c * BLOCK_K * SBK)⟩) ∧
  (∀ rg o, s.undef rg o = 0) ∧ (s.mem = s0.mem)

set_option maxHeartbeats 1000000 in
/-- **preLoop** (statements 0–14): from a clean input state (`undef = 0`), the
prologue steps to a state satisfying `bmmInvariant … 0` — the base case
(`acc = 0`, pointers seeded with the batch offset). -/
theorem bmm_preLoop (A B Out : RegionName) (s : BlockState)
    (chunk_size _K ngroups SAB SAS SAH SAK SBB SBS SBH SBK SOB SOC SOH SOM SON BM BN BK numKBlocks : Nat)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ s', stepStmts ((bmm_matmul_surface A B Out chunk_size (BK * numKBlocks) ngroups
        SAB SAS SAH SAK SBB SBS SBH SBK SOB SOC SOH SOM SON BM BN BK).toAlgKernel.body.take 15) s = some s'
      ∧ bmmInvariant A B s (s.pids 1) (pidC (s.pids 2) ngroups) (pidH (s.pids 2) ngroups)
          (pidM (s.pids 0) chunk_size BN) (pidN (s.pids 0) chunk_size BN)
          chunk_size BM BN SAB SAS SAH SAK SBB SBS SBH SBK BK numKBlocks 0 s' := by
  obtain ⟨s7, h7, hpids, hpb, hpc, hph, hpm, hpn, huf, hmem⟩ :=
    preLoop_scalars s chunk_size ngroups BM BN BK
  set PB := s.pids 1 with hPB
  set PC := pidC (s.pids 2) ngroups with hPC
  set PH := pidH (s.pids 2) ngroups with hPH
  set PM := pidM (s.pids 0) chunk_size BN with hPM
  set PN := pidN (s.pids 0) chunk_size BN with hPN
  rw [show ((bmm_matmul_surface A B Out chunk_size (BK * numKBlocks) ngroups
        SAB SAS SAH SAK SBB SBS SBH SBK SOB SOC SOH SOM SON BM BN BK).toAlgKernel.body.take 15)
      = [ Stmt.assign .nat [] "pid_b" (Op.programId 1),
          Stmt.assign .nat [] "pid_ch" (Op.programId 2),
          Stmt.assign .nat [] "pid_c"
            (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "pid_ch") (Op.constNat ngroups)),
          Stmt.assign .nat [] "pid_h"
            (Op.sub .nat Broadcast.nil (Op.ref .nat [] "pid_ch")
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_c") (Op.constNat ngroups))),
          Stmt.assign .nat [] "num_pid_n"
            (Op.div .nat Broadcast.nil (Op.sub .nat Broadcast.nil
              (Op.add .nat Broadcast.nil (Op.constNat chunk_size) (Op.constNat BN)) (Op.constNat 1)) (Op.constNat BN)),
          Stmt.assign .nat [] "pid_m"
            (Op.floorDiv .nat Broadcast.nil (Op.programId 0) (Op.ref .nat [] "num_pid_n")),
          Stmt.assign .nat [] "pid_n"
            (Op.mod .nat Broadcast.nil (Op.programId 0) (Op.ref .nat [] "num_pid_n")) ]
      ++ [ Stmt.assign .ptr [] "a_ptr"
            (Op.ptrAdd Broadcast.nil (Op.ptrBase A)
              (Op.add .nat Broadcast.nil
                (Op.add .nat Broadcast.nil
                  (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_b") (Op.constNat SAB))
                  (Op.mul .nat Broadcast.nil
                    (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_c") (Op.constNat chunk_size)) (Op.constNat SAS)))
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_h") (Op.constNat SAH)))),
          Stmt.assign .ptr [] "b_ptr"
            (Op.ptrAdd Broadcast.nil (Op.ptrBase B)
              (Op.add .nat Broadcast.nil
                (Op.add .nat Broadcast.nil
                  (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_b") (Op.constNat SBB))
                  (Op.mul .nat Broadcast.nil
                    (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_c") (Op.constNat chunk_size)) (Op.constNat SBS)))
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_h") (Op.constNat SBH)))),
          Stmt.assign .nat [BM] "offs_m"
            (Op.add .nat Broadcast.scalarL
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat BM)) (Op.arange BM)),
          Stmt.assign .nat [BN] "offs_n"
            (Op.add .nat Broadcast.scalarL
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_n") (Op.constNat BN)) (Op.arange BN)),
          Stmt.assign .nat [BK] "offs_k" (Op.arange BK),
          Stmt.assign .ptr [BM, BK] "a_ptrs"
            (Op.ptrAdd Broadcast.nil.consL.consR
              (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "a_ptr")
                (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m")) (Op.constNat SAS)))
              (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BK] "offs_k")) (Op.constNat SAK))),
          Stmt.assign .ptr [BK, BN] "b_ptrs"
            (Op.ptrAdd Broadcast.nil.consL.consR
              (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "b_ptr")
                (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "offs_k")) (Op.constNat SBK)))
              (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n")) (Op.constNat SBS))),
          Stmt.assign .real [BM, BN] "acc" (Op.full [BM, BN] (Op.const 0)) ] from rfl]
  -- step the pid prologue
  rw [stepStmts.append_some h7]
  -- a_ptr
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (aptr_base_eval s7 A PB PC PH chunk_size SAB SAS SAH hpb hpc hph))]
  set s8 := s7.setReg "a_ptr" .ptr [] (Tile.scalar (A.cast, batchOff PB PC PH chunk_size SAB SAS SAH)) with hs8
  have hpb8 : s8.regs .nat [] "pid_b" = some (Tile.scalar PB) := by simp [hs8, hpb]
  have hpc8 : s8.regs .nat [] "pid_c" = some (Tile.scalar PC) := by simp [hs8, hpc]
  have hph8 : s8.regs .nat [] "pid_h" = some (Tile.scalar PH) := by simp [hs8, hph]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (aptr_base_eval s8 B PB PC PH chunk_size SBB SBS SBH hpb8 hpc8 hph8))]
  set s9 := s8.setReg "b_ptr" .ptr [] (Tile.scalar (B.cast, batchOff PB PC PH chunk_size SBB SBS SBH)) with hs9
  -- offs_m
  have hpm9 : s9.regs .nat [] "pid_m" = some (Tile.scalar PM) := by simp [hs9, hs8, hpm]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.scalarL
      (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat BM)) (Op.arange BM)) s9
      = some (Tile.vec (fun i : Fin BM => PM * BM + i.val)) by
      simp only [evalOp_add, evalOp_mul, evalOp_ref, hpm9, evalOp_constNat, evalOp_arange, Option.bind, Option.bind_some]
      refine congrArg some ?_; ext i
      simp only [Tile.bop, Tile.vec, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul]))]
  set s10 := s9.setReg "offs_m" .nat [BM] (Tile.vec (fun i : Fin BM => PM * BM + i.val)) with hs10
  -- offs_n
  have hpn10 : s10.regs .nat [] "pid_n" = some (Tile.scalar PN) := by simp [hs10, hs9, hs8, hpn]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.scalarL
      (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_n") (Op.constNat BN)) (Op.arange BN)) s10
      = some (Tile.vec (fun j : Fin BN => PN * BN + j.val)) by
      simp only [evalOp_add, evalOp_mul, evalOp_ref, hpn10, evalOp_constNat, evalOp_arange, Option.bind, Option.bind_some]
      refine congrArg some ?_; ext j
      simp only [Tile.bop, Tile.vec, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul]))]
  set s11 := s10.setReg "offs_n" .nat [BN] (Tile.vec (fun j : Fin BN => PN * BN + j.val)) with hs11
  -- offs_k
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.arange BK) s11 = some (Tile.vec (fun e : Fin BK => e.val)) by simp [evalOp_arange]))]
  set s12 := s11.setReg "offs_k" .nat [BK] (Tile.vec (fun e : Fin BK => e.val)) with hs12
  -- a_ptrs
  have habase12 : s12.regs .ptr [] "a_ptr" = some (Tile.scalar (A.cast, batchOff PB PC PH chunk_size SAB SAS SAH)) := by
    simp [hs12, hs11, hs10, hs9, hs8]
  have hm12 : s12.regs .nat [BM] "offs_m" = some (Tile.vec (fun i : Fin BM => PM * BM + i.val)) := by simp [hs12, hs11, hs10]
  have hk12 : s12.regs .nat [BK] "offs_k" = some (Tile.vec (fun e : Fin BK => e.val)) := by simp [hs12]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aptrs_eval s12 A BM BK SAS SAK (batchOff PB PC PH chunk_size SAB SAS SAH) (fun i => PM * BM + i.val) habase12 hm12 hk12))]
  set s13 := s12.setReg "a_ptrs" .ptr [BM, BK]
    (⟨fun idx : TileIndex [BM, BK] => (A.cast, batchOff PB PC PH chunk_size SAB SAS SAH + (PM * BM + idx.1.val) * SAS + idx.2.1.val * SAK)⟩) with hs13
  -- b_ptrs
  have hbbase13 : s13.regs .ptr [] "b_ptr" = some (Tile.scalar (B.cast, batchOff PB PC PH chunk_size SBB SBS SBH)) := by
    simp [hs13, hs12, hs11, hs10, hs9]
  have hk13 : s13.regs .nat [BK] "offs_k" = some (Tile.vec (fun e : Fin BK => e.val)) := by simp [hs13, hs12]
  have hn13 : s13.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => PN * BN + j.val)) := by simp [hs13, hs12, hs11]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bptrs_eval s13 B BK BN SBK SBS (batchOff PB PC PH chunk_size SBB SBS SBH) (fun j => PN * BN + j.val) hbbase13 hk13 hn13))]
  set s14 := s13.setReg "b_ptrs" .ptr [BK, BN]
    (⟨fun idx : TileIndex [BK, BN] => (B.cast, batchOff PB PC PH chunk_size SBB SBS SBH + idx.1.val * SBK + (PN * BN + idx.2.1.val) * SBS)⟩) with hs14
  -- acc init
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (acc_init_eval s14 BM BN)), stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  refine ⟨by simp [hs14, hs13, hs12, hs11, hs10, hs9, hs8, hpids], by simp, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · -- acc = accPartial … 0 = 0
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx
    simp only [accPartial, Nat.zero_mul, Finset.range_zero, Finset.sum_empty]
  · simp only [hs14, hs13, hs12, hs11, hs10, hs9, hs8, BlockState.setReg_ne_name,
      BlockState.setReg_ne_shape, BlockState.setReg_ne_dtype, ne_eq, reduceCtorEq,
      not_false_eq_true, String.reduceEq]; exact hpb
  · simp only [hs14, hs13, hs12, hs11, hs10, hs9, hs8, BlockState.setReg_ne_name,
      BlockState.setReg_ne_shape, BlockState.setReg_ne_dtype, ne_eq, reduceCtorEq,
      not_false_eq_true, String.reduceEq]; exact hpc
  · simp only [hs14, hs13, hs12, hs11, hs10, hs9, hs8, BlockState.setReg_ne_name,
      BlockState.setReg_ne_shape, BlockState.setReg_ne_dtype, ne_eq, reduceCtorEq,
      not_false_eq_true, String.reduceEq]; exact hph
  · simp only [hs14, hs13, hs12, hs11, hs10, hs9, hs8, BlockState.setReg_ne_name,
      BlockState.setReg_ne_shape, BlockState.setReg_ne_dtype, ne_eq, reduceCtorEq,
      not_false_eq_true, String.reduceEq]; exact hpm
  · simp only [hs14, hs13, hs12, hs11, hs10, hs9, hs8, BlockState.setReg_ne_name,
      BlockState.setReg_ne_shape, BlockState.setReg_ne_dtype, ne_eq, reduceCtorEq,
      not_false_eq_true, String.reduceEq]; exact hpn
  · simp only [hs14, hs13, hs12, BlockState.setReg_ne_name,
      BlockState.setReg_ne_shape, BlockState.setReg_ne_dtype, ne_eq, reduceCtorEq,
      not_false_eq_true, String.reduceEq]; exact hm12
  · simp only [hs14, hs13, BlockState.setReg_ne_name,
      BlockState.setReg_ne_shape, BlockState.setReg_ne_dtype, ne_eq, reduceCtorEq,
      not_false_eq_true, String.reduceEq]; exact hn13
  · simp only [hs14, BlockState.setReg_ne_name,
      BlockState.setReg_ne_shape, BlockState.setReg_ne_dtype, ne_eq, reduceCtorEq,
      not_false_eq_true, String.reduceEq]; exact hk13
  · -- a_ptrs (c = 0)
    simp only [hs14, hs13, BlockState.setReg_same, BlockState.setReg_ne_name,
      BlockState.setReg_ne_shape, BlockState.setReg_ne_dtype, ne_eq, reduceCtorEq,
      not_false_eq_true, String.reduceEq, Option.some.injEq]
    ext idx <;> simp [Nat.zero_mul]
  · -- b_ptrs (c = 0)
    simp only [hs14, BlockState.setReg_same, BlockState.setReg_ne_name, BlockState.setReg_ne_shape,
      BlockState.setReg_ne_dtype, ne_eq, reduceCtorEq, not_false_eq_true, String.reduceEq,
      Option.some.injEq]
    ext idx <;> simp [Nat.zero_mul]
  · intro rg o; simp only [hs14, hs13, hs12, hs11, hs10, hs9, hs8, BlockState.setReg_undef]; exact huf ▸ hundef rg o
  · show _ = s.mem
    simp only [hs14, hs13, hs12, hs11, hs10, hs9, hs8, BlockState.setReg_mem]; exact hmem

set_option maxHeartbeats 4000000 in
/-- **Step lemma**: one K-loop body iteration advances the invariant by one
block (`acc += tl.dot(a, b)` adds the `c`-th block's dot to the partial
accumulator; the `a`/`b` pointers advance one step). Under
`K = BK · numKBlocks` and `PM·BM+i < chunk_size`, `PN·BN+j < chunk_size` the
per-block load masks are all satisfied. -/
theorem bmm_step (A B : RegionName) (s0 : BlockState)
    (PB PC PH PM PN chunk_size BM BN SAB SAS SAH SAK SBB SBS SBH SBK BK numKBlocks : Nat) (hBK : 0 < BK)
    (hmlt : ∀ i : Fin BM, PM * BM + i.val < chunk_size)
    (hnlt : ∀ j : Fin BN, PN * BN + j.val < chunk_size)
    (c : Nat) (s : BlockState) (hclt : c < numKBlocks)
    (hinv : bmmInvariant A B s0 PB PC PH PM PN chunk_size BM BN SAB SAS SAH SAK SBB SBS SBH SBK BK numKBlocks c s) :
    ∃ s', stepStmts (bmmLoopBody BM BN BK chunk_size (BK * numKBlocks) SAK SBK)
        (s.setReg "k" .nat [] (Tile.scalar c)) = some s'
      ∧ bmmInvariant A B s0 PB PC PH PM PN chunk_size BM BN SAB SAS SAH SAK SBB SBS SBH SBK BK numKBlocks (c + 1) s' := by
  set K := BK * numKBlocks with hKdef
  simp only [bmmInvariant] at hinv
  obtain ⟨hpids, hcle, hz, hpb, hpc, hph, hpm, hpn, hmm, hnn, hk, hap, hbp, hundef, hmem⟩ := hinv
  have hlt : ∀ e : Fin BK, e.val < K - c * BK := by
    intro e
    have hcK : c * BK + BK ≤ K := by
      rw [hKdef]; calc c * BK + BK = (c + 1) * BK := by ring
        _ ≤ numKBlocks * BK := Nat.mul_le_mul_right _ hclt
        _ = BK * numKBlocks := Nat.mul_comm _ _
    omega
  set apT : Tile .ptr [BM, BK] :=
    ⟨fun idx : TileIndex [BM, BK] => (A.cast, batchOff PB PC PH chunk_size SAB SAS SAH + (PM * BM + idx.1.val) * SAS + idx.2.1.val * SAK + c * BK * SAK)⟩ with hapT
  set bpT : Tile .ptr [BK, BN] :=
    ⟨fun idx : TileIndex [BK, BN] => (B.cast, batchOff PB PC PH chunk_size SBB SBS SBH + idx.1.val * SBK + (PN * BN + idx.2.1.val) * SBS + c * BK * SBK)⟩ with hbpT
  set zT : Tile .real [BM, BN] :=
    ⟨fun idx : TileIndex [BM, BN] => some (accPartial s0 A B PB PC PH PM PN BM BN chunk_size SAB SAS SAH SAK SBB SBS SBH SBK BK idx.1 idx.2.1 c)⟩ with hzT
  set sk := s.setReg "k" .nat [] (Tile.scalar c) with hsk
  have hrmem : ∀ (R : RegionName) (o : Nat), sk.readMem R o = s0.readMem R o := by
    intro R o; simp only [hsk, BlockState.setReg_readMem]; unfold BlockState.readMem; rw [hmem]
  have hapk : sk.regs .ptr [BM, BK] "a_ptrs" = some apT := by simp [hsk, hap, hapT]
  have hbpk : sk.regs .ptr [BK, BN] "b_ptrs" = some bpT := by simp [hsk, hbp, hbpT]
  have hzk : sk.regs .real [BM, BN] "acc" = some zT := by simp [hsk, hz, hzT]
  have hmk : sk.regs .nat [BM] "offs_m" = some (Tile.vec (fun i : Fin BM => PM * BM + i.val)) := by simp [hsk, hmm]
  have hnk : sk.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => PN * BN + j.val)) := by simp [hsk, hnn]
  have hkk : sk.regs .nat [BK] "offs_k" = some (Tile.vec (fun e : Fin BK => e.val)) := by simp [hsk, hk]
  have hkkv : sk.regs .nat [] "k" = some (Tile.scalar c) := by simp [hsk]
  set asub : Tile .real [BM, BK] :=
    ⟨fun idx => some (sk.readMem (apT.data idx).1 (apT.data idx).2)⟩ with hasub
  set sk1 := sk.setReg "a" .real [BM, BK] asub with hsk1
  set bsub : Tile .real [BK, BN] :=
    ⟨fun idx => some (sk1.readMem (bpT.data idx).1 (bpT.data idx).2)⟩ with hbsub
  set sk2 := sk1.setReg "b" .real [BK, BN] bsub with hsk2
  have hmk1 : sk1.regs .nat [BK] "offs_k" = some (Tile.vec (fun e : Fin BK => e.val)) := by simp [hsk1, hkk]
  have hnk1 : sk1.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => PN * BN + j.val)) := by simp [hsk1, hnk]
  have hkkv1 : sk1.regs .nat [] "k" = some (Tile.scalar c) := by simp [hsk1, hkkv]
  obtain ⟨amt, ham_eval, ham_true⟩ := amask_alltrue sk BM BK chunk_size K c _ hmk hkk hkkv hmlt hlt
  obtain ⟨bmt, hbm_eval, hbm_true⟩ := bmask_alltrue sk1 BN BK chunk_size K c _ hmk1 hnk1 hkkv1 hnlt hlt
  unfold bmmLoopBody
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (load_ptr_maskOther_alltrue (Op.ref .ptr [BM, BK] "a_ptrs") _ _ sk apT amt _
          (by rw [evalOp_ref]; simp [hapk]) ham_eval (other_broadcast_eval sk [BM, BK]) ham_true))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (load_ptr_maskOther_alltrue (Op.ref .ptr [BK, BN] "b_ptrs") _ _ sk1 bpT bmt _
          (by rw [evalOp_ref]; simp [hsk1, hbpk]) hbm_eval (other_broadcast_eval sk1 [BK, BN]) hbm_true))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (accdot_op_eval BM BK BN sk2 zT asub bsub
          (by simp [hsk2, hsk1, hzk])
          (by simp [hsk2, hsk1])
          (by simp [hsk2])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (aptr_adv_eval _ BM BK BK SAK apT (by simp [hsk2, hsk1, hapk])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (bptr_adv_eval _ BK BN BK SBK bpT (by simp [hsk2, hsk1, hbpk])))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  simp only [bmmInvariant]
  refine ⟨by simp [hsk2, hsk1, hsk, hpids], by omega, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · -- acc = accPartial (c+1)
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx
    have has : ∀ e : Fin BK, asub.data (idx.1, e, PUnit.unit)
        = some (aElem s0 A PB PC PH PM BM chunk_size SAB SAS SAH SAK idx.1 (c * BK + e.val)) := by
      intro e
      simp only [hasub, hapT, hrmem, aElem, rowIndex, batchOff]
      congr 2
      ring
    have hrmem1 : ∀ (R : RegionName) (o : Nat), sk1.readMem R o = s0.readMem R o := by
      intro R o; rw [hsk1, BlockState.setReg_readMem]; exact hrmem R o
    have hbs : ∀ e : Fin BK, bsub.data (e, idx.2.1, PUnit.unit)
        = some (bElem s0 B PB PC PH PN BN chunk_size SBB SBS SBH SBK idx.2.1 (c * BK + e.val)) := by
      intro e
      simp only [hbsub, hbpT, hrmem1, bElem, colIndex, batchOff]
      congr 2
      ring
    rw [accadd_eval BM BN zT (Tile.dot [] asub bsub) idx.1 idx.2.1
        (accPartial s0 A B PB PC PH PM PN BM BN chunk_size SAB SAS SAH SAK SBB SBS SBH SBK BK idx.1 idx.2.1 c)
        (Finset.univ.sum fun e : Fin BK =>
          aElem s0 A PB PC PH PM BM chunk_size SAB SAS SAH SAK idx.1 (c * BK + e.val)
            * bElem s0 B PB PC PH PN BN chunk_size SBB SBS SBH SBK idx.2.1 (c * BK + e.val))
        (by rw [hzT])
        (dot_ab BM BK BN asub bsub idx.1 idx.2.1 _ _ has hbs)]
    show some _ = some (accPartial s0 A B PB PC PH PM PN BM BN chunk_size SAB SAS SAH SAK SBB SBS SBH SBK BK idx.1 idx.2.1 (c + 1))
    rw [accPartial_succ]
  · simp [hsk2, hsk1, hsk, hpb]
  · simp [hsk2, hsk1, hsk, hpc]
  · simp [hsk2, hsk1, hsk, hph]
  · simp [hsk2, hsk1, hsk, hpm]
  · simp [hsk2, hsk1, hsk, hpn]
  · simp [hsk2, hsk1, hsk, hmm]
  · simp [hsk2, hsk1, hsk, hnn]
  · simp [hsk2, hsk1, hsk, hk]
  · -- a_ptrs advanced
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx <;> simp only [Tile.ptrAdd, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
      Tile.scalar, hapT, NumericDType.add]
    ring
  · -- b_ptrs advanced
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx <;> simp only [Tile.ptrAdd, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
      Tile.scalar, hbpT, NumericDType.add]
    ring
  · intro rg o; simp [hsk2, hsk1, hsk, hundef]
  · show _ = s0.mem; rw [← hmem]; rfl

/-- The dynamic K-loop bound resolves: `cdiv K BK = numKBlocks` and the loop runs
`numKBlocks` iterations, advancing `bmmInvariant` from `0` to `numKBlocks`. -/
theorem bmm_loop (A B : RegionName) (s0 : BlockState)
    (PB PC PH PM PN chunk_size BM BN SAB SAS SAH SAK SBB SBS SBH SBK BK numKBlocks : Nat) (hBK : 0 < BK)
    (hmlt : ∀ i : Fin BM, PM * BM + i.val < chunk_size)
    (hnlt : ∀ j : Fin BN, PN * BN + j.val < chunk_size)
    (s : BlockState)
    (hP0 : bmmInvariant A B s0 PB PC PH PM PN chunk_size BM BN SAB SAS SAH SAK SBB SBS SBH SBK BK numKBlocks 0 s) :
    ∃ sLoop, stepStmt (Stmt.forRangeDyn "k" (Op.constNat 0)
        (Op.div .nat Broadcast.nil (Op.sub .nat Broadcast.nil
          (Op.add .nat Broadcast.nil (Op.constNat (BK * numKBlocks)) (Op.constNat BK)) (Op.constNat 1)) (Op.constNat BK))
        (Op.constNat 1) (bmmLoopBody BM BN BK chunk_size (BK * numKBlocks) SAK SBK)) s = some sLoop
      ∧ bmmInvariant A B s0 PB PC PH PM PN chunk_size BM BN SAB SAS SAH SAK SBB SBS SBH SBK BK numKBlocks numKBlocks sLoop := by
  have hcdiv : (BK * numKBlocks + BK - 1) / BK = numKBlocks := by
    have he : BK * numKBlocks + BK - 1 = (BK - 1) + BK * numKBlocks := by omega
    rw [he, Nat.add_mul_div_left _ _ hBK, Nat.div_eq_of_lt (by omega), Nat.zero_add]
  have hresolve : stepStmt (Stmt.forRangeDyn "k" (Op.constNat 0)
      (Op.div .nat Broadcast.nil (Op.sub .nat Broadcast.nil
        (Op.add .nat Broadcast.nil (Op.constNat (BK * numKBlocks)) (Op.constNat BK)) (Op.constNat 1)) (Op.constNat BK))
      (Op.constNat 1) (bmmLoopBody BM BN BK chunk_size (BK * numKBlocks) SAK SBK)) s
      = stepForRangeAux "k" 0 numKBlocks 1 (bmmLoopBody BM BN BK chunk_size (BK * numKBlocks) SAK SBK) s := by
    rw [stepForRangeAux.forRangeDyn_unfold]
    simp only [evalOp_constNat, evalOp_div, evalOp_sub, evalOp_add, Tile.scalar, Tile.bop,
      Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.div, NumericDType.sub,
      NumericDType.add, Tile.data, Option.bind_some, bind]
    rw [hcdiv]
  rw [hresolve]
  obtain ⟨final, sLoop, haux, hfinal, hPfinal⟩ :=
    forRangeAux_inv (idx := "k") (stop := numKBlocks) (step := 1)
      (P := bmmInvariant A B s0 PB PC PH PM PN chunk_size BM BN SAB SAS SAH SAK SBB SBS SBH SBK BK numKBlocks)
      (by norm_num)
      (fun i st hlt hinv => bmm_step A B s0 PB PC PH PM PN chunk_size BM BN SAB SAS SAH SAK SBB SBS SBH SBK BK numKBlocks hBK hmlt hnlt i st hlt hinv)
      0 s hP0
  have hfinalEq : final = numKBlocks := by
    simp only [bmmInvariant] at hPfinal
    omega
  subst hfinalEq
  exact ⟨sLoop, haux, hPfinal⟩

/-! ## Post-loop: the masked output store -/

/-- The 6 post-loop statements: refresh `offs_m`/`offs_n`, the `out` real copy,
the `out_ptr` base-pointer advance, the output pointers, and the masked store. -/
def bmmPostBody (Out : RegionName) (chunk_size SOB SOC SOH SOM SON BM BN : Nat) : List Stmt :=
  [ Stmt.assign .nat [BM] "offs_m"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat BM)) (Op.arange BM)),
    Stmt.assign .nat [BN] "offs_n"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_n") (Op.constNat BN)) (Op.arange BN)),
    Stmt.assign .real [BM, BN] "out" (Op.ref .real [BM, BN] "acc"),
    Stmt.assign .ptr [] "out_ptr"
      (Op.ptrAdd Broadcast.nil (Op.ptrBase Out)
        (Op.add .nat Broadcast.nil
          (Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_b") (Op.constNat SOB))
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_c") (Op.constNat SOC)))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_h") (Op.constNat SOH)))),
    Stmt.assign .ptr [BM, BN] "out_ptrs"
      (Op.ptrAdd Broadcast.nil.consL.consR
        (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "out_ptr")
          (Op.mul .nat Broadcast.scalarL (Op.constNat SOM) (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m"))))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n")) (Op.constNat SON))),
    Stmt.store .real [BM, BN] (.ptr (Op.ref .ptr [BM, BN] "out_ptrs")) (Op.ref .real [BM, BN] "out")
      (.mask (Op.boolAnd Broadcast.nil.consL.consR
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m"))
          (Op.constNat chunk_size))
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n"))
          (Op.constNat chunk_size)))) ]

/-- The kernel body decomposes as prefix (15) ++ [K-loop, post-statements]. By `rfl`. -/
theorem bmm_body_split (A B Out : RegionName)
    (chunk_size ngroups SAB SAS SAH SAK SBB SBS SBH SBK SOB SOC SOH SOM SON BM BN BK numKBlocks : Nat) :
    (bmm_matmul_surface A B Out chunk_size (BK * numKBlocks) ngroups
        SAB SAS SAH SAK SBB SBS SBH SBK SOB SOC SOH SOM SON BM BN BK).toAlgKernel.body
      = (bmm_matmul_surface A B Out chunk_size (BK * numKBlocks) ngroups
          SAB SAS SAH SAK SBB SBS SBH SBK SOB SOC SOH SOM SON BM BN BK).toAlgKernel.body.take 15
        ++ (Stmt.forRangeDyn "k" (Op.constNat 0)
              (Op.div .nat Broadcast.nil (Op.sub .nat Broadcast.nil
                (Op.add .nat Broadcast.nil (Op.constNat (BK * numKBlocks)) (Op.constNat BK)) (Op.constNat 1)) (Op.constNat BK))
              (Op.constNat 1) (bmmLoopBody BM BN BK chunk_size (BK * numKBlocks) SAK SBK)
            :: bmmPostBody Out chunk_size SOB SOC SOH SOM SON BM BN) := by
  rfl

/-- The kernel's output offset for tile lane `idx`. -/
def outOffset (PB PC PH chunk_size SOB SOC SOH SOM SON PM PN BM BN : Nat)
    (idx : TileIndex [BM, BN]) : Nat :=
  PB * SOB + PC * SOC + PH * SOH + SOM * rowIndex PM BM idx.1 + SON * colIndex PN BN idx.2.1

/-- The genuine output cell `Σ_k A·B` as a real `MemCell`. -/
noncomputable def outputCell (s0 : BlockState) (A B : RegionName)
    (PB PC PH PM PN chunk_size BM BN SAB SAS SAH SAK SBB SBS SBH SBK BK numKBlocks : Nat)
    (idx : TileIndex [BM, BN]) : MemCell :=
  MemCell.of .real (FloatDType.real.ofReal (FloatDType.real.storeValue
    (some (bmmSpec s0 A B PB PC PH PM PN BM BN chunk_size SAB SAS SAH SAK SBB SBS SBH SBK BK numKBlocks idx.1 idx.2.1))))

private theorem foldl_writeMemTyped_real_preserves {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → TileCarrier TileDType.real)
    (mask : α → Bool) (o : Nat) (l : List α) :
    ∀ s : BlockState,
      (∀ k ∈ l, mask k = Bool.true → offsetFn k ≠ o) →
        ((l.foldl
          (fun acc k =>
            if mask k then acc.writeMemTyped .real region (offsetFn k) (valueFn k) else acc)
          s).mem region o) = s.mem region o := by
  induction l with
  | nil => intro s _h; rfl
  | cons hd tl ih =>
      intro s h
      rw [List.foldl_cons]
      have htl : ∀ k ∈ tl, mask k = Bool.true → offsetFn k ≠ o :=
        fun k hk hmk => h k (List.mem_cons_of_mem hd hk) hmk
      by_cases hmaskhd : mask hd = Bool.true
      · have hhd : offsetFn hd ≠ o := h hd (List.mem_cons_self) hmaskhd
        simp only [hmaskhd, if_true]
        rw [ih _ htl]
        unfold BlockState.writeMemTyped BlockState.writeMemAs
        change
          (if region = region ∧ o = offsetFn hd then
            MemCell.of .real (FloatDType.real.ofReal (FloatDType.real.storeValue (valueFn hd)))
          else
            s.mem region o) = s.mem region o
        rw [if_neg (by intro hsame; exact hhd hsame.2.symm)]
      · have hmaskhd' : mask hd = Bool.false := by
          cases hm : mask hd
          · rfl
          · exact False.elim (hmaskhd hm)
        simp only [hmaskhd', if_false, Bool.false_eq_true]
        exact ih _ htl

private theorem scatter_memcell_real_prop_masked_nd {region : RegionName} {shape : TileShape}
    (s : BlockState) (offsetFn : TileIndex shape → Nat)
    (valueFn : TileIndex shape → TileCarrier TileDType.real)
    (P : TileIndex shape → Prop) [DecidablePred P]
    (h_inj : Function.Injective offsetFn) (i : TileIndex shape) :
    ((TileShape.allIndices shape).foldl
       (fun acc k =>
         if P k then acc.writeMemTyped .real region (offsetFn k) (valueFn k) else acc)
       s).mem region (offsetFn i)
    = if P i then
        MemCell.of .real (FloatDType.real.ofReal (FloatDType.real.storeValue (valueFn i)))
      else
        s.mem region (offsetFn i) := by
  let l := TileShape.allIndices shape
  obtain ⟨l₁, l₂, hl⟩ := List.append_of_mem (TileShape.mem_allIndices shape i)
  have h_nodup := TileShape.allIndices_nodup shape
  change ((l.foldl
       (fun acc k =>
         if P k then acc.writeMemTyped .real region (offsetFn k) (valueFn k) else acc)
       s).mem region (offsetFn i))
    = if P i then
        MemCell.of .real (FloatDType.real.ofReal (FloatDType.real.storeValue (valueFn i)))
      else
        s.mem region (offsetFn i)
  rw [hl] at h_nodup
  rw [List.nodup_append, List.nodup_cons] at h_nodup
  obtain ⟨_, ⟨hi_notin_l2, _⟩, hl1_disj⟩ := h_nodup
  have hl' : l = l₁ ++ i :: l₂ := by simpa [l] using hl
  rw [hl', List.foldl_append, List.foldl_cons]
  have h_l1_not_in : ∀ k ∈ l₁, decide (P k) = Bool.true → offsetFn k ≠ offsetFn i := by
    intro k hk _hmk heq
    have hki : k = i := h_inj heq
    rw [hki] at hk
    exact (hl1_disj i hk i (List.mem_cons_self)) rfl
  have h_l2_not_in : ∀ k ∈ l₂, decide (P k) = Bool.true → offsetFn k ≠ offsetFn i := by
    intro k hk _hmk heq
    have hki : k = i := h_inj heq
    subst hki
    exact hi_notin_l2 hk
  have hstep :
      (fun (acc : BlockState) k =>
        if P k then acc.writeMemTyped .real region (offsetFn k) (valueFn k) else acc)
        =
      (fun (acc : BlockState) k =>
        if decide (P k) then acc.writeMemTyped .real region (offsetFn k) (valueFn k) else acc) := by
    funext acc k
    by_cases hk : P k <;> simp [hk]
  rw [hstep]
  rw [foldl_writeMemTyped_real_preserves offsetFn valueFn (fun k => decide (P k))
    (offsetFn i) l₂ _ h_l2_not_in]
  by_cases hPi : P i
  · simp only [hPi, if_true]
    unfold BlockState.writeMemTyped BlockState.writeMemAs
    simp
  · simp only [hPi, if_false]
    rw [foldl_writeMemTyped_real_preserves offsetFn valueFn (fun k => decide (P k))
      (offsetFn i) l₁]
    exact h_l1_not_in

/-- `offs_m` eval refresh. -/
theorem offs_m_eval (s : BlockState) (PM BM : Nat)
    (hpm : s.regs .nat [] "pid_m" = some (Tile.scalar PM)) :
    evalOp (Op.add .nat Broadcast.scalarL
      (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat BM)) (Op.arange BM)) s
      = some (Tile.vec (fun i : Fin BM => rowIndex PM BM i)) := by
  simp only [evalOp_add, evalOp_mul, evalOp_ref, hpm, evalOp_constNat, evalOp_arange, Option.bind,
    Option.bind_some]
  refine congrArg some ?_; ext i
  simp only [Tile.bop, Tile.vec, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.add, NumericDType.mul, rowIndex]

/-- `offs_n` eval refresh. -/
theorem offs_n_eval (s : BlockState) (PN BN : Nat)
    (hpn : s.regs .nat [] "pid_n" = some (Tile.scalar PN)) :
    evalOp (Op.add .nat Broadcast.scalarL
      (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_n") (Op.constNat BN)) (Op.arange BN)) s
      = some (Tile.vec (fun j : Fin BN => colIndex PN BN j)) := by
  simp only [evalOp_add, evalOp_mul, evalOp_ref, hpn, evalOp_constNat, evalOp_arange, Option.bind,
    Option.bind_some]
  refine congrArg some ?_; ext j
  simp only [Tile.bop, Tile.vec, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.add, NumericDType.mul, colIndex]

/-- `out_ptr` base eval: `Out + pid_b·SOB + pid_c·SOC + pid_h·SOH`. -/
theorem outptr_base_eval (s : BlockState) (Out : RegionName) (PB PC PH SOB SOC SOH : Nat)
    (hpb : s.regs .nat [] "pid_b" = some (Tile.scalar PB))
    (hpc : s.regs .nat [] "pid_c" = some (Tile.scalar PC))
    (hph : s.regs .nat [] "pid_h" = some (Tile.scalar PH)) :
    evalOp (Op.ptrAdd Broadcast.nil (Op.ptrBase Out)
      (Op.add .nat Broadcast.nil
        (Op.add .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_b") (Op.constNat SOB))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_c") (Op.constNat SOC)))
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_h") (Op.constNat SOH)))) s
      = some (Tile.scalar (Out.cast, PB * SOB + PC * SOC + PH * SOH)) := by
  rw [evalOp_ptrAdd, evalOp_ptrBase]
  simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref, hpb, hpc, hph,
    Option.bind, Option.bind_some, Option.bind_eq_bind]
  refine congrArg some ?_
  simp only [Tile.ptrAdd, Tile.bop, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.add, NumericDType.mul]
  ring_nf

/-- `out_ptrs` eval: cell `(i,j) = (Out, base + SOM·offs_m i + offs_n j·SON)`. -/
theorem outptrs_eval (s : BlockState) (Out : RegionName) (BM BN SOM SON base : Nat) (gm : Fin BM → Nat) (gn : Fin BN → Nat)
    (hbase : s.regs .ptr [] "out_ptr" = some (Tile.scalar (Out.cast, base)))
    (hm : s.regs .nat [BM] "offs_m" = some (Tile.vec gm))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec gn)) :
    evalOp (Op.ptrAdd Broadcast.nil.consL.consR
      (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "out_ptr")
        (Op.mul .nat Broadcast.scalarL (Op.constNat SOM) (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m"))))
      (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n")) (Op.constNat SON))) s
      = some (⟨fun idx : TileIndex [BM, BN] => (Out.cast, base + SOM * gm idx.1 + gn idx.2.1 * SON)⟩ : Tile .ptr [BM, BN]) := by
  rw [evalOp_ptrAdd, evalOp_ptrAdd]
  simp only [evalOp_mul, evalOp_constNat, evalOp_ref, evalOp_expandDim_one_nat,
    evalOp_expandDim_zero_nat, hbase, hm, hn, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec]
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec, NumericDType.add, NumericDType.mul]

/-- The output store mask `(offs_m < chunk_size) & (offs_n < chunk_size)`: all-`true` when in-bounds. -/
theorem outmask_alltrue (s : BlockState) (chunk_size BM BN : Nat) (gm : Fin BM → Nat) (gn : Fin BN → Nat)
    (hm : s.regs .nat [BM] "offs_m" = some (Tile.vec gm))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec gn))
    (hmlt : ∀ i, gm i < chunk_size) (hnlt : ∀ j, gn j < chunk_size) :
    ∃ mtile : Tile .bool [BM, BN],
      evalOp (Op.boolAnd Broadcast.nil.consL.consR
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m")) (Op.constNat chunk_size))
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n")) (Op.constNat chunk_size))) s
        = some mtile
      ∧ ∀ idx : TileIndex [BM, BN], mtile.data idx = Bool.true := by
  refine ⟨_, by
    simp only [evalOp, evalOp.eq_def, hm, hn, Option.bind, Option.bind_some, Option.bind_eq_bind,
      Tile.expandDim, pure, Option.some.injEq]; rfl, ?_⟩
  intro idx
  simp only [Tile.bop, Tile.cop, Tile.expandDim, Tile.vec, Tile.scalar, Broadcast.leftIndex,
    Broadcast.rightIndex, ComparableDType.lt, TileShape.dropInsertedIndex, Bool.and_eq_true,
    decide_eq_true_eq]
  exact ⟨by simpa using hmlt _, by simpa using hnlt _⟩

/-- The masked real store reduces to the masked scatter foldl. -/
theorem store_eval (BM BN : Nat) (st : BlockState)
    (oT : Tile .real [BM, BN]) (opT : Tile .ptr [BM, BN]) (mT : Tile .bool [BM, BN])
    (chunk_size : Nat)
    (ho : st.regs .real [BM, BN] "out" = some oT)
    (hop : st.regs .ptr [BM, BN] "out_ptrs" = some opT)
    (hmeval : evalOp (Op.boolAnd Broadcast.nil.consL.consR
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m")) (Op.constNat chunk_size))
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n")) (Op.constNat chunk_size))) st = some mT) :
    stepStmt (Stmt.store .real [BM, BN] (.ptr (Op.ref .ptr [BM, BN] "out_ptrs")) (Op.ref .real [BM, BN] "out")
        (.mask (Op.boolAnd Broadcast.nil.consL.consR
          (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m")) (Op.constNat chunk_size))
          (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n")) (Op.constNat chunk_size))))) st
      = some ((TileShape.allIndices [BM, BN]).foldl
          (fun acc idx =>
            if mT.data idx then acc.writeMemTyped .real (opT.data idx).1 (opT.data idx).2 (oT.data idx) else acc) st) := by
  simp only [stepStmt, evalOp_ref, ho, hop, hmeval, bind, Option.bind_some]
  refine congrArg some (List.foldl_ext _ _ st (fun acc idx _ => ?_))
  by_cases hb : mT.data idx
  · simp only [hb, if_true]
  · simp only [hb, if_false, Bool.false_eq_true]

set_option maxHeartbeats 2000000 in
/-- **postLoop**: from the invariant at `numKBlocks` blocks, the masked store
writes the genuine value `Σ_k A·B` at every in-bounds output lane. -/
theorem bmm_postLoop (A B Out : RegionName) (s0 : BlockState)
    (PB PC PH PM PN chunk_size BM BN SAB SAS SAH SAK SBB SBS SBH SBK SOB SOC SOH SOM SON BK numKBlocks : Nat)
    (hInj : Function.Injective (outOffset PB PC PH chunk_size SOB SOC SOH SOM SON PM PN BM BN))
    (hmlt : ∀ i : Fin BM, rowIndex PM BM i < chunk_size)
    (hnlt : ∀ j : Fin BN, colIndex PN BN j < chunk_size)
    (st : BlockState)
    (hinv : bmmInvariant A B s0 PB PC PH PM PN chunk_size BM BN SAB SAS SAH SAK SBB SBS SBH SBK BK numKBlocks numKBlocks st) :
    ∃ sfin, stepStmts (bmmPostBody Out chunk_size SOB SOC SOH SOM SON BM BN) st = some sfin
      ∧ ∀ idx : TileIndex [BM, BN],
          sfin.mem Out (outOffset PB PC PH chunk_size SOB SOC SOH SOM SON PM PN BM BN idx)
            = outputCell s0 A B PB PC PH PM PN chunk_size BM BN SAB SAS SAH SAK SBB SBS SBH SBK BK numKBlocks idx := by
  simp only [bmmInvariant] at hinv
  obtain ⟨hpids, hcle, hz, hpb, hpc, hph, hpm, hpn, hmm, hnn, hk, hap, hbp, hundef, hmem⟩ := hinv
  set g : TileIndex [BM, BN] → ℝ :=
    fun idx => bmmSpec s0 A B PB PC PH PM PN BM BN chunk_size SAB SAS SAH SAK SBB SBS SBH SBK BK numKBlocks idx.1 idx.2.1 with hg
  have hzspec : st.regs .real [BM, BN] "acc" = some ⟨fun idx => some (g idx)⟩ := by
    rw [hz]; refine congrArg some ?_; ext idx; simp only [hg, bmmSpec, accPartial, Nat.mul_comm numKBlocks BK]
  unfold bmmPostBody
  -- step 1: offs_m
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (offs_m_eval st PM BM hpm))]
  set s1 := st.setReg "offs_m" .nat [BM] (Tile.vec (fun i : Fin BM => rowIndex PM BM i)) with hs1
  -- step 2: offs_n
  have hpn1 : s1.regs .nat [] "pid_n" = some (Tile.scalar PN) := by simp [hs1, hpn]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (offs_n_eval s1 PN BN hpn1))]
  set s2 := s1.setReg "offs_n" .nat [BN] (Tile.vec (fun j : Fin BN => colIndex PN BN j)) with hs2
  -- step 3: out = acc
  have hacc2 : s2.regs .real [BM, BN] "acc" = some ⟨fun idx => some (g idx)⟩ := by simp [hs2, hs1, hzspec]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .real [BM, BN] "acc") s2 = some ⟨fun idx => some (g idx)⟩ by rw [evalOp_ref, hacc2]))]
  set s3 := s2.setReg "out" .real [BM, BN] (⟨fun idx => some (g idx)⟩ : Tile .real [BM, BN]) with hs3
  -- step 4: out_ptr base
  have hpb3 : s3.regs .nat [] "pid_b" = some (Tile.scalar PB) := by simp [hs3, hs2, hs1, hpb]
  have hpc3 : s3.regs .nat [] "pid_c" = some (Tile.scalar PC) := by simp [hs3, hs2, hs1, hpc]
  have hph3 : s3.regs .nat [] "pid_h" = some (Tile.scalar PH) := by simp [hs3, hs2, hs1, hph]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (outptr_base_eval s3 Out PB PC PH SOB SOC SOH hpb3 hpc3 hph3))]
  set s4 := s3.setReg "out_ptr" .ptr [] (Tile.scalar (Out.cast, PB * SOB + PC * SOC + PH * SOH)) with hs4
  -- step 5: out_ptrs
  have hobase4 : s4.regs .ptr [] "out_ptr" = some (Tile.scalar (Out.cast, PB * SOB + PC * SOC + PH * SOH)) := by simp [hs4]
  have hm4 : s4.regs .nat [BM] "offs_m" = some (Tile.vec (fun i : Fin BM => rowIndex PM BM i)) := by simp [hs4, hs3, hs2, hs1]
  have hn4 : s4.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => colIndex PN BN j)) := by simp [hs4, hs3, hs2]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (outptrs_eval s4 Out BM BN SOM SON (PB * SOB + PC * SOC + PH * SOH) (fun i => rowIndex PM BM i) (fun j => colIndex PN BN j) hobase4 hm4 hn4))]
  set opT : Tile .ptr [BM, BN] :=
    ⟨fun idx : TileIndex [BM, BN] => (Out.cast, (PB * SOB + PC * SOC + PH * SOH) + SOM * rowIndex PM BM idx.1 + colIndex PN BN idx.2.1 * SON)⟩ with hopT
  set s5 := s4.setReg "out_ptrs" .ptr [BM, BN] opT with hs5
  -- step 6: masked store
  have hm5 : s5.regs .nat [BM] "offs_m" = some (Tile.vec (fun i : Fin BM => rowIndex PM BM i)) := by simp [hs5, hm4]
  have hn5 : s5.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => colIndex PN BN j)) := by simp [hs5, hn4]
  obtain ⟨mT, hmask_eval, hmask_true⟩ :=
    outmask_alltrue s5 chunk_size BM BN (fun i => rowIndex PM BM i) (fun j => colIndex PN BN j) hm5 hn5 hmlt hnlt
  set oT : Tile .real [BM, BN] := (⟨fun idx => some (g idx)⟩ : Tile .real [BM, BN]) with hoT
  have hout5 : s5.regs .real [BM, BN] "out" = some oT := by simp [hs5, hs4, hs3, hs2, hoT]
  have hop5 : s5.regs .ptr [BM, BN] "out_ptrs" = some opT := by simp [hs5]
  rw [stepStmts.cons_some (store_eval BM BN s5 oT opT mT chunk_size hout5 hop5 hmask_eval), stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  intro idx
  have hstep_eq :
      (fun (acc : BlockState) k =>
        if mT.data k then acc.writeMemTyped .real (opT.data k).1 (opT.data k).2 (oT.data k) else acc)
        =
      (fun (acc : BlockState) k =>
        if (fun _ : TileIndex [BM, BN] => True) k then acc.writeMemTyped .real (opT.data k).1 (opT.data k).2 (oT.data k) else acc) := by
    funext acc k; rw [hmask_true k]; simp
  rw [hstep_eq]
  have hOutRegion : ∀ k : TileIndex [BM, BN], (opT.data k).1 = Out := by intro k; simp [hopT]
  rw [show
      ((TileShape.allIndices [BM, BN]).foldl
        (fun acc k => if (fun _ : TileIndex [BM, BN] => True) k then acc.writeMemTyped .real (opT.data k).1 (opT.data k).2 (oT.data k) else acc) s5)
      = ((TileShape.allIndices [BM, BN]).foldl
        (fun acc k => if (fun _ : TileIndex [BM, BN] => True) k then acc.writeMemTyped .real Out ((opT.data k).2) (oT.data k) else acc) s5)
      from List.foldl_ext _ _ s5 (fun acc k _ => by rw [hOutRegion k])]
  have hooff : ∀ k : TileIndex [BM, BN], (opT.data k).2 = outOffset PB PC PH chunk_size SOB SOC SOH SOM SON PM PN BM BN k := by
    intro k; simp only [hopT, outOffset]; ring
  have hoffInj : Function.Injective (fun idx : TileIndex [BM, BN] => (opT.data idx).2) := by
    intro a b hab; apply hInj
    rw [← hooff a, ← hooff b]; exact hab
  rw [← hooff idx]
  rw [scatter_memcell_real_prop_masked_nd (region := Out) (s := s5)
    (offsetFn := fun idx : TileIndex [BM, BN] => (opT.data idx).2)
    (valueFn := fun idx => oT.data idx)
    (P := fun _ => True) hoffInj idx]
  simp only [if_pos trivial, outputCell, hoT, hg]

/-! ## Composition: full exec closed form -/

set_option maxHeartbeats 2000000 in
/-- **Top exec reduction**: `preLoop` + K-loop (`bmm_loop`) + `postLoop` compose
into the full `exec`. Every in-bounds output lane equals `Σ_k A·B`. -/
theorem bmm_exec_closed_form (A B Out : RegionName) (s : BlockState)
    (chunk_size ngroups SAB SAS SAH SAK SBB SBS SBH SBK SOB SOC SOH SOM SON BM BN BK numKBlocks : Nat) (hBK : 0 < BK)
    (hInj : Function.Injective (outOffset (s.pids 1) (pidC (s.pids 2) ngroups) (pidH (s.pids 2) ngroups)
      chunk_size SOB SOC SOH SOM SON (pidM (s.pids 0) chunk_size BN) (pidN (s.pids 0) chunk_size BN) BM BN))
    (hmlt : ∀ i : Fin BM, rowIndex (pidM (s.pids 0) chunk_size BN) BM i < chunk_size)
    (hnlt : ∀ j : Fin BN, colIndex (pidN (s.pids 0) chunk_size BN) BN j < chunk_size)
    (hundef : ∀ rg o, s.undef rg o = 0)
    (idx : TileIndex [BM, BN]) :
    (match exec (bmm_matmul_surface A B Out chunk_size (BK * numKBlocks) ngroups
        SAB SAS SAH SAK SBB SBS SBH SBK SOB SOC SOH SOM SON BM BN BK) s with
      | some s' => s'.mem Out (outOffset (s.pids 1) (pidC (s.pids 2) ngroups) (pidH (s.pids 2) ngroups)
          chunk_size SOB SOC SOH SOM SON (pidM (s.pids 0) chunk_size BN) (pidN (s.pids 0) chunk_size BN) BM BN idx)
      | none => (0 : MemCell)) =
      outputCell s A B (s.pids 1) (pidC (s.pids 2) ngroups) (pidH (s.pids 2) ngroups)
        (pidM (s.pids 0) chunk_size BN) (pidN (s.pids 0) chunk_size BN)
        chunk_size BM BN SAB SAS SAH SAK SBB SBS SBH SBK BK numKBlocks idx := by
  set PB := s.pids 1 with hPB
  set PC := pidC (s.pids 2) ngroups with hPC
  set PH := pidH (s.pids 2) ngroups with hPH
  set PM := pidM (s.pids 0) chunk_size BN with hPM
  set PN := pidN (s.pids 0) chunk_size BN with hPN
  obtain ⟨s0, hpre_eq, hP0⟩ := bmm_preLoop A B Out s chunk_size (BK * numKBlocks) ngroups
    SAB SAS SAH SAK SBB SBS SBH SBK SOB SOC SOH SOM SON BM BN BK numKBlocks hundef
  obtain ⟨sLoop, hLoopStmt, hPLoop⟩ :=
    bmm_loop A B s PB PC PH PM PN chunk_size BM BN SAB SAS SAH SAK SBB SBS SBH SBK BK numKBlocks hBK hmlt hnlt s0 hP0
  obtain ⟨sfin, hTail, hpost⟩ :=
    bmm_postLoop A B Out s PB PC PH PM PN chunk_size BM BN SAB SAS SAH SAK SBB SBS SBH SBK SOB SOC SOH SOM SON BK numKBlocks hInj hmlt hnlt sLoop hPLoop
  have hexec : exec (bmm_matmul_surface A B Out chunk_size (BK * numKBlocks) ngroups
      SAB SAS SAH SAK SBB SBS SBH SBK SOB SOC SOH SOM SON BM BN BK) s = some sfin := by
    rw [exec, bmm_body_split A B Out chunk_size ngroups SAB SAS SAH SAK SBB SBS SBH SBK SOB SOC SOH SOM SON BM BN BK numKBlocks,
      stepStmts.append_some hpre_eq, stepStmts.cons_some hLoopStmt, hTail]
  rw [hexec]
  exact hpost idx

/-- **Closed-form batched-matmul correctness for `bmm_chunk_fwd` (general statement).**

For arbitrary `chunk_size`, `ngroups`, tile dims `BM`/`BN`, strides, K-block size
`BK`, and K-block count `numKBlocks` (so `K = BK · numKBlocks`), every in-bounds
output cell of the computed `BM × BN` chunk tile equals the genuine batched
matrix product `Σ_{k < BK·numKBlocks} A[i,k] · B[k,j]` over ℝ — NOT the kernel's
own executed value — under the kernel's batch/chunk/head/row/col addressing.

`PB/PC/PH/PM/PN` are the kernel's own derived program coordinates. Preconditions:
`0 < BK`; all tile rows/cols in-bounds (`PM·BM+i < chunk_size`,
`PN·BN+j < chunk_size`), making the load and store masks all-true;
output-address injectivity; clean initial `undef`. -/
theorem bmm_chunk_fwd_closed_form_correct
    (A B Out : RegionName) (s : BlockState)
    (chunk_size ngroups SAB SAS SAH SAK SBB SBS SBH SBK SOB SOC SOH SOM SON BM BN BK numKBlocks : Nat) (hBK : 0 < BK)
    (hInj : Function.Injective (outOffset (s.pids 1) (pidC (s.pids 2) ngroups) (pidH (s.pids 2) ngroups)
      chunk_size SOB SOC SOH SOM SON (pidM (s.pids 0) chunk_size BN) (pidN (s.pids 0) chunk_size BN) BM BN))
    (hmlt : ∀ i : Fin BM, rowIndex (pidM (s.pids 0) chunk_size BN) BM i < chunk_size)
    (hnlt : ∀ j : Fin BN, colIndex (pidN (s.pids 0) chunk_size BN) BN j < chunk_size)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes
      (kernel := bmm_matmul_surface A B Out chunk_size (BK * numKBlocks) ngroups
        SAB SAS SAH SAK SBB SBS SBH SBK SOB SOC SOH SOM SON BM BN BK)
      (initialState := s)
      (write := fun idx : TileIndex [BM, BN] =>
        some (Out, outOffset (s.pids 1) (pidC (s.pids 2) ngroups) (pidH (s.pids 2) ngroups)
          chunk_size SOB SOC SOH SOM SON (pidM (s.pids 0) chunk_size BN) (pidN (s.pids 0) chunk_size BN) BM BN idx))
      (expected := fun idx : TileIndex [BM, BN] =>
        outputCell s A B (s.pids 1) (pidC (s.pids 2) ngroups) (pidH (s.pids 2) ngroups)
          (pidM (s.pids 0) chunk_size BN) (pidN (s.pids 0) chunk_size BN)
          chunk_size BM BN SAB SAS SAH SAK SBB SBS SBH SBK BK numKBlocks idx) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [bmm_matmul_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst hs0
  intro idx
  have hmain := bmm_exec_closed_form A B Out s0 chunk_size ngroups SAB SAS SAH SAK SBB SBS SBH SBK
    SOB SOC SOH SOM SON BM BN BK numKBlocks hBK hInj hmlt hnlt hundef idx
  rw [hExec] at hmain
  exact hmain

/-! ## Public Python test-shape summaries (genuine spec)

Python test case 1 (ungrouped) and case 4 (grouped) are the pure batched-matmul
configurations (`causal = false`, `seq_idx = None`). Both use `chunk_size = 32`,
`K = 64`, fp16 inputs, with the `(2,128,64)` / `(2,128,4,64)` layouts. -/

/-- **Python case 1 summary** (ungrouped: `batch=2`, `seqlen=128`, `K=64`,
`chunk_size=32`; strides `a=b=(8192,64,0,1)`, `out=(4096,1024,0,32,1)`,
`ngroups=1`; `BLOCK_SIZE_K=32`, `numKBlocks=2`): the genuine batched matmul
surface lowers to the algorithm layer and realizes the per-program matrix
product `Σ_{k<64} A[i,k]·B[k,j]` on every in-bounds output lane. -/
theorem bmm_chunk_fwd_python_case1_summary
    (A B Out : RegionName) (s : BlockState) (BM BN : Nat)
    (hInj : Function.Injective (outOffset (s.pids 1) (pidC (s.pids 2) 1) (pidH (s.pids 2) 1)
      32 4096 1024 0 32 1 (pidM (s.pids 0) 32 BN) (pidN (s.pids 0) 32 BN) BM BN))
    (hmlt : ∀ i : Fin BM, rowIndex (pidM (s.pids 0) 32 BN) BM i < 32)
    (hnlt : ∀ j : Fin BN, colIndex (pidN (s.pids 0) 32 BN) BN j < 32)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    (∃ alg, (bmm_matmul_surface A B Out 32 64 1 8192 64 0 1 8192 64 0 1
        4096 1024 0 32 1 BM BN 32).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := bmm_matmul_surface A B Out 32 64 1 8192 64 0 1 8192 64 0 1
        4096 1024 0 32 1 BM BN 32)
      (initialState := s)
      (write := fun idx : TileIndex [BM, BN] =>
        some (Out, outOffset (s.pids 1) (pidC (s.pids 2) 1) (pidH (s.pids 2) 1)
          32 4096 1024 0 32 1 (pidM (s.pids 0) 32 BN) (pidN (s.pids 0) 32 BN) BM BN idx))
      (expected := fun idx : TileIndex [BM, BN] =>
        outputCell s A B (s.pids 1) (pidC (s.pids 2) 1) (pidH (s.pids 2) 1)
          (pidM (s.pids 0) 32 BN) (pidN (s.pids 0) 32 BN)
          32 BM BN 8192 64 0 1 8192 64 0 1 32 2 idx) := by
  refine ⟨bmm_matmul_surface_toAlgorithm_supported A B Out 32 64 1 8192 64 0 1 8192 64 0 1 4096 1024 0 32 1 BM BN 32, ?_⟩
  exact bmm_chunk_fwd_closed_form_correct A B Out s 32 1 8192 64 0 1 8192 64 0 1 4096 1024 0 32 1 BM BN 32 2
    (by norm_num) hInj hmlt hnlt hundef

/-- **Python case 4 summary** (grouped: `batch=2`, `seqlen=128`, `ngroups=4`,
`K=64`, `chunk_size=32`; strides `a=b=(32768,256,64,1)`,
`out=(16384,4096,1024,32,1)`; `BLOCK_SIZE_K=32`, `numKBlocks=2`): the genuine
batched matmul surface lowers and realizes `Σ_{k<64} A[i,k]·B[k,j]` per program. -/
theorem bmm_chunk_fwd_python_case4_summary
    (A B Out : RegionName) (s : BlockState) (BM BN : Nat)
    (hInj : Function.Injective (outOffset (s.pids 1) (pidC (s.pids 2) 4) (pidH (s.pids 2) 4)
      32 16384 4096 1024 32 1 (pidM (s.pids 0) 32 BN) (pidN (s.pids 0) 32 BN) BM BN))
    (hmlt : ∀ i : Fin BM, rowIndex (pidM (s.pids 0) 32 BN) BM i < 32)
    (hnlt : ∀ j : Fin BN, colIndex (pidN (s.pids 0) 32 BN) BN j < 32)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    (∃ alg, (bmm_matmul_surface A B Out 32 64 4 32768 256 64 1 32768 256 64 1
        16384 4096 1024 32 1 BM BN 32).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := bmm_matmul_surface A B Out 32 64 4 32768 256 64 1 32768 256 64 1
        16384 4096 1024 32 1 BM BN 32)
      (initialState := s)
      (write := fun idx : TileIndex [BM, BN] =>
        some (Out, outOffset (s.pids 1) (pidC (s.pids 2) 4) (pidH (s.pids 2) 4)
          32 16384 4096 1024 32 1 (pidM (s.pids 0) 32 BN) (pidN (s.pids 0) 32 BN) BM BN idx))
      (expected := fun idx : TileIndex [BM, BN] =>
        outputCell s A B (s.pids 1) (pidC (s.pids 2) 4) (pidH (s.pids 2) 4)
          (pidM (s.pids 0) 32 BN) (pidN (s.pids 0) 32 BN)
          32 BM BN 32768 256 64 1 32768 256 64 1 32 2 idx) := by
  refine ⟨bmm_matmul_surface_toAlgorithm_supported A B Out 32 64 4 32768 256 64 1 32768 256 64 1 16384 4096 1024 32 1 BM BN 32, ?_⟩
  exact bmm_chunk_fwd_closed_form_correct A B Out s 32 4 32768 256 64 1 32768 256 64 1 16384 4096 1024 32 1 BM BN 32 2
    (by norm_num) hInj hmlt hnlt hundef

end VeriTile.Bench.TritonBenchG.BmmChunkFwd
