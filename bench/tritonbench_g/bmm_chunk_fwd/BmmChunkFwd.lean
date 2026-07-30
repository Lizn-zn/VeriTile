import VeriTile.Triton

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
bmm_chunk_fwd_closed_form_correct                 ← TOP THEOREM (ComputeCorrect.Realizes_without_Rounding)
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

/-- **preLoop scalars** (statements 0–6): the pid derivation. Steps to a state
where `pid_b/pid_c/pid_h/pid_m/pid_n` hold their derived values. -/
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
    ComputeCorrect.Realizes_without_Rounding
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

/-! ## Public output summary (genuine spec, dimension-general)

The pure batched-matmul configuration (`causal = false`, `seq_idx = None`,
covering Python test cases 1 and 4) over **fully symbolic** dimensions: arbitrary
`chunk_size`, `ngroups`, tile dims `BM`/`BN`, all batch/chunk/head/row/col
strides, K-block size `BK` (with `0 < BK`), and K-block count `numKBlocks` (so
`K = BK · numKBlocks`). -/

/-- **`bmm_chunk_fwd` output summary (dimension-general).**

The genuine batched-matmul surface (1) lowers to the algorithm layer and (2)
realizes the per-program matrix product `Σ_{k < BK·numKBlocks} A[i,k]·B[k,j]`
over ℝ on every in-bounds output lane — the closed-form spec read from INPUT
memory (`outputCell`/`bmmSpec` via `bmm_chunk_fwd_closed_form_correct`), NOT the
kernel's own executed value.

Preconditions: `0 < BK`; all tile rows/cols in-bounds
(`PM·BM+i < chunk_size`, `PN·BN+j < chunk_size`), making the load/store masks
all-true; output-address injectivity; clean initial `undef`. `PB/PC/PH/PM/PN`
are the kernel's own derived program coordinates. -/
specification bmm_chunk_fwd_output_summary_general
    (A B Out : RegionName) (s : BlockState)
    (chunk_size ngroups SAB SAS SAH SAK SBB SBS SBH SBK SOB SOC SOH SOM SON BM BN BK numKBlocks : Nat)
    (hBK : 0 < BK)
    (hInj : Function.Injective (outOffset (s.pids 1) (pidC (s.pids 2) ngroups) (pidH (s.pids 2) ngroups)
      chunk_size SOB SOC SOH SOM SON (pidM (s.pids 0) chunk_size BN) (pidN (s.pids 0) chunk_size BN) BM BN))
    (hmlt : ∀ i : Fin BM, rowIndex (pidM (s.pids 0) chunk_size BN) BM i < chunk_size)
    (hnlt : ∀ j : Fin BN, colIndex (pidN (s.pids 0) chunk_size BN) BN j < chunk_size)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    (∃ alg, (bmm_matmul_surface A B Out chunk_size (BK * numKBlocks) ngroups
        SAB SAS SAH SAK SBB SBS SBH SBK SOB SOC SOH SOM SON BM BN BK).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
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
  refine ⟨bmm_matmul_surface_toAlgorithm_supported A B Out chunk_size (BK * numKBlocks) ngroups
    SAB SAS SAH SAK SBB SBS SBH SBK SOB SOC SOH SOM SON BM BN BK, ?_⟩
  exact bmm_chunk_fwd_closed_form_correct A B Out s chunk_size ngroups
    SAB SAS SAH SAK SBB SBS SBH SBK SOB SOC SOH SOM SON BM BN BK numKBlocks
    hBK hInj hmlt hnlt hundef


/-! ## The `⊨[R]` streaming headline (wave-5 S1 fold genre)

Everything below is purely additive; the kernel surface and the exact stack
above are untouched.

**The genre.** `_bmm_chunk_fwd_kernel` in its genuine batched-matmul
configuration is the textbook S1 streaming shape: a `forRangeDyn` fold over
`numKBlocks` K-blocks that accumulates `acc += tl.dot(a, b)` from two masked
per-step loads, followed by a single terminal masked store. That is exactly
`StreamMetaMasked3DKernelIO₂` at `nMeta := 0` — the kernel loads **no** scalar
metadata, so the slot vector is empty (`sty` / `mbuf` / `mwin` are `Fin.elim0`)
and the region list `List.ofFn io.mbuf ++ [inp1, inp2, out]` collapses to
`[A, B, Out]`. Step `t` reads a `[BM, BK]` tile of `A` and a `[BK, BN]` tile of
`B`; the terminal store writes one `[BM, BN]` tile of `Out`.

**Rounding events: none.** The surface has no `Op.castFloat` at all — the Python
`.to(dot_dtype)` is absent from the genuine configuration's loads and
`acc.to(out_ptr.dtype.element_ty)` transcribes to the plain register copy
`out = acc` at the real carrier (see `bmmPostBody`'s third statement), so the
terminal `tl.store` is a `Stmt.store .real` and `outDType := .real`. Under
`execR R` every statement therefore steps identically to the exact stepper for
**every** `R` (`bmmIO_*_castFree` below), the proven `bmm_preLoop` / `bmm_step` /
`bmm_loop` invariant tower is reused verbatim, and the headline needs no
rounding-boundary hypothesis: only the terminal store is re-proved on the `R`
side (`bmm_postLoopR`), where the `.real` grid degenerates the skin's boundary
quantization (`R.round .real = id`).

**Masks.** Under the in-bounds pins `hmlt` / `hnlt` (every tile row/col of the
program is `< chunk_size`) and `K = BK · numKBlocks` the kernel's own
`(offs_m < chunk_size) & (offs_k < K - k·BK)` load masks and its
`(offs_m < chunk_size) & (offs_n < chunk_size)` store mask are **all-true**, so
every lane of every streamed tile is genuinely read and every output lane is
genuinely written: the skin's `mask1` / `mask2` / `writeMask` are the full
windows. -/

section IOFace

open scoped VeriTile.Triton.StreamMetaMasked3DKernelIO₂

/-! ### Cast-free collapse

The surface contains no `Op.castFloat`, so `evalOpR R` **is** `evalOp` on every
one of its expressions and `writeMemTypedR R .real` is the exact `.real` write.
Consequently every segment steps identically under `stepStmtsR R` and
`stepStmts`, for every rounding model. -/

/-- `.real` stores never round: `writeMemTypedR` delegates to the exact write. -/
private theorem bmmIO_wmtR_real (R : RoundingModel) (s : BlockState)
    (region : RegionName) (o : Nat) (v : TileCarrier .real) :
    s.writeMemTypedR R .real region o v = s.writeMemTyped .real region o v := rfl

/-- A `.real`-typed rounded write **is** the `writeMemAsR R .real` write
(`RoundingModel.storeValue_real`): lets the masked `.real` terminal store reuse
the `writeMemAsR` scatter readback / frame lemma family. -/
private theorem bmmIO_writeMemTypedR_real_eq (R : RoundingModel) (s : BlockState)
    (region : RegionName) (offset : Nat) (v : TileCarrier TileDType.real) :
    s.writeMemTypedR R .real region offset v
      = s.writeMemAsR R .real region offset v := by
  show s.writeMemTyped .real region offset v = _
  simp only [BlockState.writeMemTyped, BlockState.writeMemAs, BlockState.writeMemAsR,
    RoundingModel.storeValue_real]

/-- Tag-exact readback of a stored `.real` cell through `readMemAs .real`. -/
private theorem bmmIO_readMemAs_real_of_cell {s : BlockState} {region : RegionName}
    {offset : Nat} {x : ℝ}
    (h : s.mem region offset
      = MemCell.of FloatDType.real.toTileDType (FloatDType.real.ofReal x)) :
    s.readMemAs .real region offset = FloatDType.real.ofReal x := by
  simp [BlockState.readMemAs, h, FloatDType.storeValue, FloatDType.ofReal]

/-- Per-statement cast-free collapse lifts to statement lists (walks the actual
successor chain; a failing step collapses on both sides). -/
private theorem bmmIO_stepStmtsR_castFree_of_stmts (R : RoundingModel) :
    ∀ (l : List Stmt), (∀ st ∈ l, ∀ u, stepStmtR R st u = stepStmt st u) →
      ∀ s, stepStmtsR R l s = stepStmts l s
  | [], _, s => by simp only [stepStmtsR, stepStmts]
  | st :: rest, h, s => by
      simp only [stepStmtsR, stepStmts, h st List.mem_cons_self s]
      cases stepStmt st s with
      | none => rfl
      | some s' =>
          exact bmmIO_stepStmtsR_castFree_of_stmts R rest
            (fun st' h' u => h st' (List.mem_cons_of_mem _ h') u) s'

/-! ### Named body decomposition

`bmm_body_split` above splits the kernel as `body.take 15 ++ (K-loop ::
bmmPostBody …)`. The `⊨[R]` walk needs the prefix as an explicit statement list
(to run `TraceSafeListR` and the cast-free collapse over it), so the fifteen
statements are named here in two halves — the pid derivation (`bmmPidBody`,
exactly `preLoop_scalars`' list) and the index/pointer seeding (`bmmSeedBody`) —
and `bmm_take15_eq` locks the concatenation to the surface. -/

/-- The kernel's 7-statement pid prologue (the list `preLoop_scalars` steps). -/
private def bmmPidBody (chunk_size ngroups BN : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "pid_b" (Op.programId 1),
    Stmt.assign .nat [] "pid_ch" (Op.programId 2),
    Stmt.assign .nat [] "pid_c"
      (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "pid_ch") (Op.constNat ngroups)),
    Stmt.assign .nat [] "pid_h"
      (Op.sub .nat Broadcast.nil (Op.ref .nat [] "pid_ch")
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_c") (Op.constNat ngroups))),
    Stmt.assign .nat [] "num_pid_n"
      (Op.div .nat Broadcast.nil (Op.sub .nat Broadcast.nil
        (Op.add .nat Broadcast.nil (Op.constNat chunk_size) (Op.constNat BN)) (Op.constNat 1))
        (Op.constNat BN)),
    Stmt.assign .nat [] "pid_m"
      (Op.floorDiv .nat Broadcast.nil (Op.programId 0) (Op.ref .nat [] "num_pid_n")),
    Stmt.assign .nat [] "pid_n"
      (Op.mod .nat Broadcast.nil (Op.programId 0) (Op.ref .nat [] "num_pid_n")) ]

/-- The kernel's 8-statement seeding prologue: the `a`/`b` batch·chunk·head base
pointers, the three index vectors, the two 2-D pointer tiles, `acc = 0`. -/
private def bmmSeedBody (A B : RegionName)
    (chunk_size SAB SAS SAH SAK SBB SBS SBH SBK BM BN BK : Nat) : List Stmt :=
  [ Stmt.assign .ptr [] "a_ptr"
      (Op.ptrAdd Broadcast.nil (Op.ptrBase A)
        (Op.add .nat Broadcast.nil
          (Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_b") (Op.constNat SAB))
            (Op.mul .nat Broadcast.nil
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_c") (Op.constNat chunk_size))
              (Op.constNat SAS)))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_h") (Op.constNat SAH)))),
    Stmt.assign .ptr [] "b_ptr"
      (Op.ptrAdd Broadcast.nil (Op.ptrBase B)
        (Op.add .nat Broadcast.nil
          (Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_b") (Op.constNat SBB))
            (Op.mul .nat Broadcast.nil
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_c") (Op.constNat chunk_size))
              (Op.constNat SBS)))
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
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m"))
            (Op.constNat SAS)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BK] "offs_k"))
          (Op.constNat SAK))),
    Stmt.assign .ptr [BK, BN] "b_ptrs"
      (Op.ptrAdd Broadcast.nil.consL.consR
        (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "b_ptr")
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "offs_k"))
            (Op.constNat SBK)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n"))
          (Op.constNat SBS))),
    Stmt.assign .real [BM, BN] "acc" (Op.full [BM, BN] (Op.const 0)) ]

/-- The kernel's whole 15-statement prologue — `body.take 15`, named. -/
private def bmmPreBody (A B : RegionName)
    (chunk_size ngroups SAB SAS SAH SAK SBB SBS SBH SBK BM BN BK : Nat) : List Stmt :=
  bmmPidBody chunk_size ngroups BN
    ++ bmmSeedBody A B chunk_size SAB SAS SAH SAK SBB SBS SBH SBK BM BN BK

/-- `body.take 15` **is** the named prologue. -/
private theorem bmm_take15_eq (A B Out : RegionName)
    (chunk_size ngroups SAB SAS SAH SAK SBB SBS SBH SBK SOB SOC SOH SOM SON
      BM BN BK numKBlocks : Nat) :
    (bmm_matmul_surface A B Out chunk_size (BK * numKBlocks) ngroups
        SAB SAS SAH SAK SBB SBS SBH SBK SOB SOC SOH SOM SON BM BN BK).toAlgKernel.body.take 15
      = bmmPreBody A B chunk_size ngroups SAB SAS SAH SAK SBB SBS SBH SBK BM BN BK := rfl

/-- The K-loop statement, named. -/
private def bmmLoopStmt (chunk_size SAK SBK BM BN BK numKBlocks : Nat) : Stmt :=
  Stmt.forRangeDyn "k" (Op.constNat 0)
    (Op.div .nat Broadcast.nil (Op.sub .nat Broadcast.nil
      (Op.add .nat Broadcast.nil (Op.constNat (BK * numKBlocks)) (Op.constNat BK))
      (Op.constNat 1)) (Op.constNat BK))
    (Op.constNat 1) (bmmLoopBody BM BN BK chunk_size (BK * numKBlocks) SAK SBK)

/-- `bmm_body_split` with the prologue and the K-loop statement named. -/
private theorem bmm_body_split' (A B Out : RegionName)
    (chunk_size ngroups SAB SAS SAH SAK SBB SBS SBH SBK SOB SOC SOH SOM SON
      BM BN BK numKBlocks : Nat) :
    (bmm_matmul_surface A B Out chunk_size (BK * numKBlocks) ngroups
        SAB SAS SAH SAK SBB SBS SBH SBK SOB SOC SOH SOM SON BM BN BK).toAlgKernel.body
      = bmmPreBody A B chunk_size ngroups SAB SAS SAH SAK SBB SBS SBH SBK BM BN BK
        ++ (bmmLoopStmt chunk_size SAK SBK BM BN BK numKBlocks
            :: bmmPostBody Out chunk_size SOB SOC SOH SOM SON BM BN) := rfl

set_option maxHeartbeats 1000000 in
/-- Every prologue statement is cast-free (`.nat`/`.ptr` index arithmetic and the
`acc = 0` fill — no rounding site, no memory access). -/
private theorem bmmIO_preBody_stmt_castFree (R : RoundingModel) (A B : RegionName)
    (chunk_size ngroups SAB SAS SAH SAK SBB SBS SBH SBK BM BN BK : Nat) :
    ∀ st ∈ bmmPreBody A B chunk_size ngroups SAB SAS SAH SAK SBB SBS SBH SBK BM BN BK,
      ∀ u, stepStmtR R st u = stepStmt st u := by
  intro st hst u
  simp only [bmmPreBody, bmmPidBody, bmmSeedBody, List.cons_append, List.nil_append,
    List.mem_cons, List.not_mem_nil, or_false] at hst
  rcases hst with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
    rfl | rfl | rfl <;>
    simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def]

set_option maxHeartbeats 1000000 in
/-- Every K-loop body statement is cast-free: the two masked loads are `.real`,
`acc += tl.dot(a, b)` is real arithmetic, and the two pointer advances are
`.ptr`/`.nat`. -/
private theorem bmmIO_loopBody_stmt_castFree (R : RoundingModel)
    (chunk_size K SAK SBK BM BN BK : Nat) :
    ∀ st ∈ bmmLoopBody BM BN BK chunk_size K SAK SBK,
      ∀ u, stepStmtR R st u = stepStmt st u := by
  intro st hst u
  simp only [bmmLoopBody, List.mem_cons, List.not_mem_nil, or_false] at hst
  rcases hst with rfl | rfl | rfl | rfl | rfl <;>
    simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def]

set_option maxHeartbeats 1000000 in
/-- Every post-loop statement is cast-free, the terminal masked `.real` store
included (`bmmIO_wmtR_real`). -/
private theorem bmmIO_postBody_stmt_castFree (R : RoundingModel) (Out : RegionName)
    (chunk_size SOB SOC SOH SOM SON BM BN : Nat) :
    ∀ st ∈ bmmPostBody Out chunk_size SOB SOC SOH SOM SON BM BN,
      ∀ u, stepStmtR R st u = stepStmt st u := by
  intro st hst u
  simp only [bmmPostBody, List.mem_cons, List.not_mem_nil, or_false] at hst
  rcases hst with rfl | rfl | rfl | rfl | rfl | rfl <;>
    simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def, bmmIO_wmtR_real R]

/-- The prologue collapses onto the exact stepper. -/
private theorem bmmIO_preBody_castFree (R : RoundingModel) (A B : RegionName)
    (chunk_size ngroups SAB SAS SAH SAK SBB SBS SBH SBK BM BN BK : Nat) (t : BlockState) :
    stepStmtsR R (bmmPreBody A B chunk_size ngroups SAB SAS SAH SAK SBB SBS SBH SBK BM BN BK) t
      = stepStmts (bmmPreBody A B chunk_size ngroups SAB SAS SAH SAK SBB SBS SBH SBK BM BN BK) t :=
  bmmIO_stepStmtsR_castFree_of_stmts R _
    (bmmIO_preBody_stmt_castFree R A B chunk_size ngroups SAB SAS SAH SAK SBB SBS SBH SBK
      BM BN BK) t

/-- The K-loop body collapses onto the exact stepper. -/
private theorem bmmIO_loopBody_castFree (R : RoundingModel)
    (chunk_size K SAK SBK BM BN BK : Nat) (t : BlockState) :
    stepStmtsR R (bmmLoopBody BM BN BK chunk_size K SAK SBK) t
      = stepStmts (bmmLoopBody BM BN BK chunk_size K SAK SBK) t :=
  bmmIO_stepStmtsR_castFree_of_stmts R _
    (bmmIO_loopBody_stmt_castFree R chunk_size K SAK SBK BM BN BK) t

/-- The post-loop tail collapses onto the exact stepper. -/
private theorem bmmIO_postBody_castFree (R : RoundingModel) (Out : RegionName)
    (chunk_size SOB SOC SOH SOM SON BM BN : Nat) (t : BlockState) :
    stepStmtsR R (bmmPostBody Out chunk_size SOB SOC SOH SOM SON BM BN) t
      = stepStmts (bmmPostBody Out chunk_size SOB SOC SOH SOM SON BM BN) t :=
  bmmIO_stepStmtsR_castFree_of_stmts R _
    (bmmIO_postBody_stmt_castFree R Out chunk_size SOB SOC SOH SOM SON BM BN) t

/-- **The `forRangeDyn` cast-free lift.** `StepR` ships `stepStmtR_forRange` for
the *static* loop but no `forRangeDyn` mirror, so the K-loop's collapse is proved
here straight from `stepStmtR`'s definition: the three bound expressions (`0`,
`cdiv K BK`, `1`) are cast-free `.nat` arithmetic, evaluating identically under
`evalOpR R` and `evalOp`, after which the strided fold is
`stepForRangeAuxR_castFree`. -/
private theorem bmmIO_loopStmt_castFree (R : RoundingModel)
    (chunk_size SAK SBK BM BN BK numKBlocks : Nat) (u : BlockState) :
    stepStmtR R (bmmLoopStmt chunk_size SAK SBK BM BN BK numKBlocks) u
      = stepStmt (bmmLoopStmt chunk_size SAK SBK BM BN BK numKBlocks) u := by
  simp only [bmmLoopStmt, stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def, bind,
    Option.bind_eq_bind, Option.bind_some]
  exact stepForRangeAuxR_castFree R _
    (bmmIO_loopBody_castFree R chunk_size (BK * numKBlocks) SAK SBK BM BN BK) "k" _ _ _ u

/-! ### Flat-memory coverage

The two shape-indexed `Op` arms (`expandDim`'s `insertAxis` and `dot`'s
`batch ++ [M, N]`) are the canonical simp blind spot: their `Op.FlattenOk`
equations match only up to unfolding the *type index*, so they are supplied as
recipe lemmas **applied**, never put in a simp set. -/

private theorem bmmIO_flattenOk_expandDim {dtype : TileDType} {shape : TileShape}
    (ax : Fin (shape.length + 1)) (a : Op dtype shape) :
    (Op.expandDim ax a).FlattenOk ↔ a.FlattenOk := by simp [Op.FlattenOk]

private theorem bmmIO_flattenOk_dot {batch : TileShape} {M K N : Nat}
    (a : Op .real (batch ++ [M, K])) (b : Op .real (batch ++ [K, N])) :
    (Op.dot a b).FlattenOk ↔ a.FlattenOk ∧ b.FlattenOk := by simp [Op.FlattenOk]

private theorem bmmIO_flattenOk_cons (st : Stmt) (l : List Stmt)
    (h1 : Stmt.FlattenOk st) (h2 : StmtList.FlattenOk l) :
    StmtList.FlattenOk (st :: l) := ⟨h1, h2⟩

private theorem bmmIO_flattenOk_append : ∀ (l1 l2 : List Stmt),
    StmtList.FlattenOk l1 → StmtList.FlattenOk l2 → StmtList.FlattenOk (l1 ++ l2)
  | [], _, _, h2 => h2
  | _ :: rest, l2, h1, h2 => ⟨h1.1, bmmIO_flattenOk_append rest l2 h1.2 h2⟩

set_option maxHeartbeats 2000000 in
/-- The genuine batched-matmul surface sits inside the flat-memory bridge's
covered fragment (plain `ptrAdd` walks only — no `ptrSub`, no atomics, no block
pointers; the `forRangeDyn` clause recurses into the K-loop body). -/
private theorem bmm_flattenOk (A B Out : RegionName)
    (chunk_size ngroups SAB SAS SAH SAK SBB SBS SBH SBK SOB SOC SOH SOM SON
      BM BN BK numKBlocks : Nat) :
    ((bmm_matmul_surface A B Out chunk_size (BK * numKBlocks) ngroups
        SAB SAS SAH SAK SBB SBS SBH SBK SOB SOC SOH SOM SON BM BN BK).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  rw [bmm_body_split' A B Out chunk_size ngroups SAB SAS SAH SAK SBB SBS SBH SBK
    SOB SOC SOH SOM SON BM BN BK numKBlocks]
  refine bmmIO_flattenOk_append _ _ ?_ (bmmIO_flattenOk_cons _ _ ?_ ?_)
  · simp [bmmPreBody, bmmPidBody, bmmSeedBody, StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]
    repeat' first
      | exact trivial
      | apply And.intro
      | exact (bmmIO_flattenOk_expandDim _ _).mpr (by simp [Op.FlattenOk])
  · simp [bmmLoopStmt, bmmLoopBody, Stmt.FlattenOk, StmtList.FlattenOk, Op.FlattenOk]
    repeat' first
      | exact trivial
      | apply And.intro
      | exact (bmmIO_flattenOk_expandDim _ _).mpr (by simp [Op.FlattenOk])
      | apply (bmmIO_flattenOk_dot _ _).mpr
      | simp [Op.FlattenOk]
  · simp [bmmPostBody, StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]
    repeat' first
      | exact trivial
      | apply And.intro
      | exact (bmmIO_flattenOk_expandDim _ _).mpr (by simp [Op.FlattenOk])

/-! ### pid-form address vocabulary

The exact stack's `aElem` / `bElem` / `outOffset` already take the derived
program coordinates `PB/PC/PH/PM/PN` as plain `Nat`s, so no `s0`-form ↔ pid-form
restating is needed: the skin hands its window fields the three raw pids and the
kernel's own derivations `pidC` / `pidH` / `pidM` / `pidN` recover the rest. Each
address below is **definitionally** the exact stack's address at `PB := pid₁`,
`PC := pidC pid₂ ngroups`, … (the bridges are `rfl`). -/

/-- Step `t`, lane `(i, e)`'s `A` read address: the kernel's `a_ptrs` cell at
K-block `t`, i.e. `aElem`'s address at contracted index `k = t·BK + e`. -/
private def bmmAAddr (pid₀ pid₁ pid₂ ngroups chunk_size BM BN BK
    SAB SAS SAH SAK t i e : Nat) : Nat :=
  batchOff pid₁ (pidC pid₂ ngroups) (pidH pid₂ ngroups) chunk_size SAB SAS SAH
    + (pidM pid₀ chunk_size BN * BM + i) * SAS + (t * BK + e) * SAK

/-- Step `t`, lane `(e, j)`'s `B` read address: the kernel's `b_ptrs` cell at
K-block `t`, i.e. `bElem`'s address at contracted index `k = t·BK + e`. -/
private def bmmBAddr (pid₀ pid₁ pid₂ ngroups chunk_size BN BK
    SBB SBS SBH SBK t e j : Nat) : Nat :=
  batchOff pid₁ (pidC pid₂ ngroups) (pidH pid₂ ngroups) chunk_size SBB SBS SBH
    + (t * BK + e) * SBK + (pidN pid₀ chunk_size BN * BN + j) * SBS

/-- Lane `(i, j)`'s terminal `Out` store address — the pid form of `outOffset`. -/
private def bmmOutAddr (pid₀ pid₁ pid₂ ngroups chunk_size BM BN
    SOB SOC SOH SOM SON i j : Nat) : Nat :=
  pid₁ * SOB + pidC pid₂ ngroups * SOC + pidH pid₂ ngroups * SOH
    + SOM * (pidM pid₀ chunk_size BN * BM + i) + SON * (pidN pid₀ chunk_size BN * BN + j)

/-- `aElem`'s address **is** the pid-form `A` read address (bridge, `rfl`). -/
private theorem bmmAAddr_eq (s : BlockState) (A : RegionName)
    (ngroups chunk_size BM BN BK SAB SAS SAH SAK t : Nat) (i : Fin BM) (e : Fin BK) :
    aElem s A (s.pids 1) (pidC (s.pids 2) ngroups) (pidH (s.pids 2) ngroups)
        (pidM (s.pids 0) chunk_size BN) BM chunk_size SAB SAS SAH SAK i (t * BK + e.val)
      = s.readMem A (bmmAAddr (s.pids 0) (s.pids 1) (s.pids 2) ngroups chunk_size BM BN BK
          SAB SAS SAH SAK t i.val e.val) := rfl

/-- `bElem`'s address **is** the pid-form `B` read address (bridge, `rfl`). -/
private theorem bmmBAddr_eq (s : BlockState) (B : RegionName)
    (ngroups chunk_size BN BK SBB SBS SBH SBK t : Nat) (e : Fin BK) (j : Fin BN) :
    bElem s B (s.pids 1) (pidC (s.pids 2) ngroups) (pidH (s.pids 2) ngroups)
        (pidN (s.pids 0) chunk_size BN) BN chunk_size SBB SBS SBH SBK j (t * BK + e.val)
      = s.readMem B (bmmBAddr (s.pids 0) (s.pids 1) (s.pids 2) ngroups chunk_size BN BK
          SBB SBS SBH SBK t e.val j.val) := rfl

/-- `outOffset` **is** the pid-form terminal store address (bridge, `rfl`). -/
private theorem bmmOutAddr_eq (s : BlockState)
    (ngroups chunk_size BM BN SOB SOC SOH SOM SON : Nat) (idx : TileIndex [BM, BN]) :
    outOffset (s.pids 1) (pidC (s.pids 2) ngroups) (pidH (s.pids 2) ngroups) chunk_size
        SOB SOC SOH SOM SON (pidM (s.pids 0) chunk_size BN) (pidN (s.pids 0) chunk_size BN)
        BM BN idx
      = bmmOutAddr (s.pids 0) (s.pids 1) (s.pids 2) ngroups chunk_size BM BN
          SOB SOC SOH SOM SON idx.1.val idx.2.1.val := rfl

/-! ### IO signature and the streamed spec -/

/-- **The streamed batched-matmul spec**: the ideal ℝ value of output lane
`l = (i, j)` as a single `numKBlocks`-step fold over the streamed tiles,

`∑_{t < numKBlocks} ∑_{e < BK} aTile(t)[i, e] · bTile(t)[e, j]`.

This is the `⊨[R]` face of `bmmSpec`: the same `∑_{k < K} A[i,k]·B[k,j]`
batched-matmul reference, re-expressed on the skin's per-step tiles instead of on
`s0`-indexed `readMem`s (`bmmSpec_eq_streamSum`). -/
noncomputable def bmmStreamSum (BM BN BK numKBlocks : Nat)
    (xs : Fin numKBlocks → Fin (BM * BK) → ℝ)
    (ys : Fin numKBlocks → Fin (BK * BN) → ℝ)
    (l : Fin (BM * BN)) : ℝ :=
  ∑ t : Fin numKBlocks, ∑ e : Fin BK,
    xs t (Lane2D.encode ((Lane2D.decode l).1, e, PUnit.unit))
      * ys t (Lane2D.encode (e, (Lane2D.decode l).2.1, PUnit.unit))

/-- **Streaming IO signature** of `bmm_chunk_fwd`'s genuine batched-matmul
surface on the metadata-parametrized two-stream fold skin (S1: fold + terminal
masked store, 3-D pid grid), at `nMeta := 0` — this kernel loads **no** scalar
metadata, so the slot vector is empty (`sty` / `mbuf` / `mwin` are `Fin.elim0`,
and `List.ofFn io.mbuf ++ [A, B, Out]` collapses to `[A, B, Out]`) and every
window is a function of the three pids alone. The 3-D grid is the kernel's own:
`pid₀` = flattened `(chunk-row, chunk-col)` tile, `pid₁` = batch, `pid₂` =
chunk·group.

Per step the kernel reads a `[BM, BK]` tile of `A` and a `[BK, BN]` tile of `B`
(both masked, with `other = 0.0`); after the `numKBlocks`-step fold one
`[BM, BN]` output tile is masked-stored at the **`.real`** grid (`outDType`
default — the Python `acc.to(out_ptr.dtype.element_ty)` transcribes to the plain
`out = acc` register copy, so the `tl.store` is a `Stmt.store .real` and carries
no quantization event).

* `read1` lane `j = (i, e)` (row-major over `[BM, BK]`): `bmmAAddr`, the `a_ptrs`
  cell at K-block `t`.
* `read2` lane `j = (e, n)` (row-major over `[BK, BN]`): `bmmBAddr`.
* `write` lane `j = (i, n)` (row-major over `[BM, BN]`): `bmmOutAddr`.
* `mask1` / `mask2` / `writeMask`: the full windows — under the headline's
  in-bounds pins the kernel's own row/col and K-tail masks are all-true, so every
  lane really is read / written (see the section docstring). -/
def bmm_chunk_fwd_IO (A B Out : RegionName)
    (chunk_size ngroups SAB SAS SAH SAK SBB SBS SBH SBK SOB SOC SOH SOM SON
      BM BN BK numKBlocks : Nat) :
    StreamMetaMasked3DKernelIO₂ where
  kernel := bmm_matmul_surface A B Out chunk_size (BK * numKBlocks) ngroups
    SAB SAS SAH SAK SBB SBS SBH SBK SOB SOC SOH SOM SON BM BN BK
  inp1 := A
  inp2 := B
  out := Out
  nMeta := 0
  sty := Fin.elim0
  mbuf := Fin.elim0
  mwin := Fin.elim0
  T := numKBlocks
  B1 := BM * BK
  B2 := BK * BN
  C := BM * BN
  outDType := .real
  read1 := fun pid₀ pid₁ pid₂ _ t j =>
    bmmAAddr pid₀ pid₁ pid₂ ngroups chunk_size BM BN BK SAB SAS SAH SAK
      t.val (j.val / BK) (j.val % BK)
  read2 := fun pid₀ pid₁ pid₂ _ t j =>
    bmmBAddr pid₀ pid₁ pid₂ ngroups chunk_size BN BK SBB SBS SBH SBK
      t.val (j.val / BN) (j.val % BN)
  write := fun pid₀ pid₁ pid₂ _ j =>
    bmmOutAddr pid₀ pid₁ pid₂ ngroups chunk_size BM BN SOB SOC SOH SOM SON
      (j.val / BN) (j.val % BN)
  mask1 := fun _ _ _ _ _ _ => True
  mask2 := fun _ _ _ _ _ _ => True
  writeMask := fun _ _ _ _ _ => True

/-! ### The spec bridge: `bmmSpec` **is** the streamed fold -/

/-- **`bmmSpec` = `bmmStreamSum`.** Under the two stream pins the genuine
batched-matmul reference at output lane `l = (i, j)` is the skin-level fold: the
`K = BK · numKBlocks` contracted range splits row-major into `numKBlocks` steps
of `BK` lanes (`StreamLane.sum_range_mul`), and the per-`k` factors are exactly
the streamed tiles at the decoded lanes. -/
private theorem bmmSpec_eq_streamSum (s₀ : BlockState) (A B : RegionName)
    (PB PC PH PM PN chunk_size BM BN SAB SAS SAH SAK SBB SBS SBH SBK BK numKBlocks : Nat)
    (xs : Fin numKBlocks → Fin (BM * BK) → ℝ)
    (ys : Fin numKBlocks → Fin (BK * BN) → ℝ)
    (hx : ∀ (t : Fin numKBlocks) (i : Fin BM) (e : Fin BK),
      aElem s₀ A PB PC PH PM BM chunk_size SAB SAS SAH SAK i (t.val * BK + e.val)
        = xs t (Lane2D.encode (i, e, PUnit.unit)))
    (hy : ∀ (t : Fin numKBlocks) (e : Fin BK) (j : Fin BN),
      bElem s₀ B PB PC PH PN BN chunk_size SBB SBS SBH SBK j (t.val * BK + e.val)
        = ys t (Lane2D.encode (e, j, PUnit.unit)))
    (l : Fin (BM * BN)) :
    bmmSpec s₀ A B PB PC PH PM PN BM BN chunk_size SAB SAS SAH SAK SBB SBS SBH SBK
        BK numKBlocks (Lane2D.decode l).1 (Lane2D.decode l).2.1
      = bmmStreamSum BM BN BK numKBlocks xs ys l := by
  unfold bmmSpec bmmStreamSum
  rw [show BK * numKBlocks = numKBlocks * BK from Nat.mul_comm _ _,
    StreamLane.sum_range_mul numKBlocks BK
      (fun k => aElem s₀ A PB PC PH PM BM chunk_size SAB SAS SAH SAK (Lane2D.decode l).1 k
        * bElem s₀ B PB PC PH PN BN chunk_size SBB SBS SBH SBK (Lane2D.decode l).2.1 k)]
  exact Finset.sum_congr rfl fun t _ => Finset.sum_congr rfl fun e _ => by
    rw [hx t (Lane2D.decode l).1 e, hy t e (Lane2D.decode l).2.1]

/-! ### The safety-walk context (weak, `undef`-free)

The skin's `hts` obligation quantifies over an **arbitrary** launch state, so the
trace-safety walk cannot assume the clean-`undef` precondition that
`bmmInvariant` (and hence `bmm_preLoop` / `bmm_step`) carries. `BmmSafeCtx` is
the shape half of `bmmInvariant`: the prologue-derived registers and the two
loop-carried pointer tiles at K-block `c`, plus "there is *some* accumulator
tile" (needed only for the body's `acc += tl.dot` step to succeed), with no
`undef` / `mem` / accumulator-value pins. Trace safety never needs a loaded
value — only the address and mask tiles, which are `undef`-independent. -/

private structure BmmSafeCtx (A B : RegionName)
    (PB PC PH PM PN chunk_size BM BN SAB SAS SAH SAK SBB SBS SBH SBK BK : Nat)
    (c : Nat) (s : BlockState) : Prop where
  regPidB : s.regs .nat [] "pid_b" = some (Tile.scalar PB)
  regPidC : s.regs .nat [] "pid_c" = some (Tile.scalar PC)
  regPidH : s.regs .nat [] "pid_h" = some (Tile.scalar PH)
  regPidM : s.regs .nat [] "pid_m" = some (Tile.scalar PM)
  regPidN : s.regs .nat [] "pid_n" = some (Tile.scalar PN)
  regOffsM : s.regs .nat [BM] "offs_m" = some (Tile.vec (fun i : Fin BM => PM * BM + i.val))
  regOffsN : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => PN * BN + j.val))
  regOffsK : s.regs .nat [BK] "offs_k" = some (Tile.vec (fun e : Fin BK => e.val))
  regAPtrs : s.regs .ptr [BM, BK] "a_ptrs" = some ⟨fun idx : TileIndex [BM, BK] =>
    (A.cast, batchOff PB PC PH chunk_size SAB SAS SAH + (PM * BM + idx.1.val) * SAS
      + idx.2.1.val * SAK + c * BK * SAK)⟩
  regBPtrs : s.regs .ptr [BK, BN] "b_ptrs" = some ⟨fun idx : TileIndex [BK, BN] =>
    (B.cast, batchOff PB PC PH chunk_size SBB SBS SBH + idx.1.val * SBK
      + (PN * BN + idx.2.1.val) * SBS + c * BK * SBK)⟩
  regAcc : ∃ t : Tile .real [BM, BN], s.regs .real [BM, BN] "acc" = some t

set_option maxHeartbeats 1000000 in
/-- **Weak preLoop**: from an *arbitrary* state the 15 prologue statements step
successfully to a `BmmSafeCtx` state at K-block `0`. The value half of
`bmm_preLoop` (clean `undef`, `mem = s.mem`, `acc = 0`) is dropped. -/
private theorem bmm_preLoopW (A B : RegionName) (s : BlockState)
    (chunk_size ngroups SAB SAS SAH SAK SBB SBS SBH SBK BM BN BK : Nat) :
    ∃ s', stepStmts (bmmPreBody A B chunk_size ngroups SAB SAS SAH SAK SBB SBS SBH SBK
        BM BN BK) s = some s'
      ∧ BmmSafeCtx A B (s.pids 1) (pidC (s.pids 2) ngroups) (pidH (s.pids 2) ngroups)
          (pidM (s.pids 0) chunk_size BN) (pidN (s.pids 0) chunk_size BN)
          chunk_size BM BN SAB SAS SAH SAK SBB SBS SBH SBK BK 0 s' := by
  obtain ⟨s7, h7, hpids, hpb, hpc, hph, hpm, hpn, huf, hmem⟩ :=
    preLoop_scalars s chunk_size ngroups BM BN BK
  set PB := s.pids 1 with hPB
  set PC := pidC (s.pids 2) ngroups with hPC
  set PH := pidH (s.pids 2) ngroups with hPH
  set PM := pidM (s.pids 0) chunk_size BN with hPM
  set PN := pidN (s.pids 0) chunk_size BN with hPN
  unfold bmmPreBody bmmPidBody bmmSeedBody
  rw [stepStmts.append_some h7]
  -- a_ptr
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aptr_base_eval s7 A PB PC PH chunk_size SAB SAS SAH hpb hpc hph))]
  set s8 := s7.setReg "a_ptr" .ptr [] (Tile.scalar (A.cast,
    batchOff PB PC PH chunk_size SAB SAS SAH)) with hs8
  have hpb8 : s8.regs .nat [] "pid_b" = some (Tile.scalar PB) := by simp [hs8, hpb]
  have hpc8 : s8.regs .nat [] "pid_c" = some (Tile.scalar PC) := by simp [hs8, hpc]
  have hph8 : s8.regs .nat [] "pid_h" = some (Tile.scalar PH) := by simp [hs8, hph]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aptr_base_eval s8 B PB PC PH chunk_size SBB SBS SBH hpb8 hpc8 hph8))]
  set s9 := s8.setReg "b_ptr" .ptr [] (Tile.scalar (B.cast,
    batchOff PB PC PH chunk_size SBB SBS SBH)) with hs9
  -- offs_m
  have hpm9 : s9.regs .nat [] "pid_m" = some (Tile.scalar PM) := by simp [hs9, hs8, hpm]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.scalarL
      (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat BM)) (Op.arange BM)) s9
      = some (Tile.vec (fun i : Fin BM => PM * BM + i.val)) by
      simp only [evalOp_add, evalOp_mul, evalOp_ref, hpm9, evalOp_constNat, evalOp_arange,
        Option.bind, Option.bind_some]
      refine congrArg some ?_; ext i
      simp only [Tile.bop, Tile.vec, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex,
        NumericDType.add, NumericDType.mul]))]
  set s10 := s9.setReg "offs_m" .nat [BM] (Tile.vec (fun i : Fin BM => PM * BM + i.val)) with hs10
  -- offs_n
  have hpn10 : s10.regs .nat [] "pid_n" = some (Tile.scalar PN) := by simp [hs10, hs9, hs8, hpn]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.scalarL
      (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_n") (Op.constNat BN)) (Op.arange BN)) s10
      = some (Tile.vec (fun j : Fin BN => PN * BN + j.val)) by
      simp only [evalOp_add, evalOp_mul, evalOp_ref, hpn10, evalOp_constNat, evalOp_arange,
        Option.bind, Option.bind_some]
      refine congrArg some ?_; ext j
      simp only [Tile.bop, Tile.vec, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex,
        NumericDType.add, NumericDType.mul]))]
  set s11 := s10.setReg "offs_n" .nat [BN] (Tile.vec (fun j : Fin BN => PN * BN + j.val)) with hs11
  -- offs_k
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.arange BK) s11 = some (Tile.vec (fun e : Fin BK => e.val)) by
      simp [evalOp_arange]))]
  set s12 := s11.setReg "offs_k" .nat [BK] (Tile.vec (fun e : Fin BK => e.val)) with hs12
  -- a_ptrs
  have habase12 : s12.regs .ptr [] "a_ptr"
      = some (Tile.scalar (A.cast, batchOff PB PC PH chunk_size SAB SAS SAH)) := by
    simp [hs12, hs11, hs10, hs9, hs8]
  have hm12 : s12.regs .nat [BM] "offs_m"
      = some (Tile.vec (fun i : Fin BM => PM * BM + i.val)) := by simp [hs12, hs11, hs10]
  have hk12 : s12.regs .nat [BK] "offs_k"
      = some (Tile.vec (fun e : Fin BK => e.val)) := by simp [hs12]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aptrs_eval s12 A BM BK SAS SAK (batchOff PB PC PH chunk_size SAB SAS SAH)
      (fun i => PM * BM + i.val) habase12 hm12 hk12))]
  set s13 := s12.setReg "a_ptrs" .ptr [BM, BK]
    (⟨fun idx : TileIndex [BM, BK] => (A.cast, batchOff PB PC PH chunk_size SAB SAS SAH
      + (PM * BM + idx.1.val) * SAS + idx.2.1.val * SAK)⟩) with hs13
  -- b_ptrs
  have hbbase13 : s13.regs .ptr [] "b_ptr"
      = some (Tile.scalar (B.cast, batchOff PB PC PH chunk_size SBB SBS SBH)) := by
    simp [hs13, hs12, hs11, hs10, hs9]
  have hk13 : s13.regs .nat [BK] "offs_k"
      = some (Tile.vec (fun e : Fin BK => e.val)) := by simp [hs13, hs12]
  have hn13 : s13.regs .nat [BN] "offs_n"
      = some (Tile.vec (fun j : Fin BN => PN * BN + j.val)) := by simp [hs13, hs12, hs11]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bptrs_eval s13 B BK BN SBK SBS (batchOff PB PC PH chunk_size SBB SBS SBH)
      (fun j => PN * BN + j.val) hbbase13 hk13 hn13))]
  set s14 := s13.setReg "b_ptrs" .ptr [BK, BN]
    (⟨fun idx : TileIndex [BK, BN] => (B.cast, batchOff PB PC PH chunk_size SBB SBS SBH
      + idx.1.val * SBK + (PN * BN + idx.2.1.val) * SBS)⟩) with hs14
  -- acc init
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (acc_init_eval s14 BM BN)), stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
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
  · simp only [hs14, hs13, BlockState.setReg_same, BlockState.setReg_ne_name,
      BlockState.setReg_ne_shape, BlockState.setReg_ne_dtype, ne_eq, reduceCtorEq,
      not_false_eq_true, String.reduceEq, Option.some.injEq]
    ext idx <;> simp [Nat.zero_mul]
  · simp only [hs14, BlockState.setReg_same, BlockState.setReg_ne_name,
      BlockState.setReg_ne_shape, BlockState.setReg_ne_dtype, ne_eq, reduceCtorEq,
      not_false_eq_true, String.reduceEq, Option.some.injEq]
    ext idx <;> simp [Nat.zero_mul]
  · exact ⟨⟨fun _ : TileIndex [BM, BN] => some (0 : ℝ)⟩, by simp only [BlockState.setReg_same]⟩

/-! ### Trace-safety helpers

Same story as `Op.FlattenOk`: the two shape-indexed `Op` arms need `SafeAtR`
recipes **applied**, never simp'd. `bmmIO_tsl_cons` is the cons form of
`Stmt.TraceSafeListR` specialised to a statement whose successor is already
known from the exact walk (every statement of this kernel is cast-free, so the
`R` successor *is* the exact one). -/

private theorem bmmIO_safeAtR_expandDim (R : RoundingModel) (bounds : RegionBounds)
    (s : BlockState) {dtype : TileDType} {shape : TileShape}
    (ax : Fin (shape.length + 1)) (a : Op dtype shape) :
    (Op.expandDim ax a).SafeAtR R bounds s ↔ a.SafeAtR R bounds s := by simp [Op.SafeAtR]

private theorem bmmIO_safeAtR_dot (R : RoundingModel) (bounds : RegionBounds)
    (s : BlockState) {batch : TileShape} {M K N : Nat}
    (a : Op .real (batch ++ [M, K])) (b : Op .real (batch ++ [K, N])) :
    (Op.dot a b).SafeAtR R bounds s ↔ a.SafeAtR R bounds s ∧ b.SafeAtR R bounds s := by
  simp [Op.SafeAtR]

/-- `TraceSafeListR` cons with a **known** successor: the head statement is
cast-free, so its `R`-successor is the exact one, which the walk already
computed. -/
private theorem bmmIO_tsl_cons {R : RoundingModel} {bounds : RegionBounds} {st : Stmt}
    {rest : List Stmt} {s s' : BlockState}
    (hcf : ∀ u, stepStmtR R st u = stepStmt st u)
    (hstep : stepStmt st s = some s')
    (h1 : Stmt.TraceSafeR R bounds st s)
    (h2 : Stmt.TraceSafeListR R bounds rest s') :
    Stmt.TraceSafeListR R bounds (st :: rest) s :=
  Stmt.TraceSafeListR.cons_intro h1 (fun s'' hs'' => by
    rw [hcf s, hstep] at hs''
    obtain rfl := Option.some.inj hs''
    exact h2)

/-- Safety of a `.real` masked-`other` pointer load: children safe, and every
*lane*'s pointer in bounds (a fortiori every active lane's). -/
private theorem bmmIO_loadSafeAtR {shape : TileShape} (R : RoundingModel)
    (bounds : RegionBounds) (s : BlockState)
    (ptrOp : Op .ptr shape) (maskOp : Op .bool shape) (otherOp : Op .real shape)
    (hptr : ptrOp.SafeAtR R bounds s) (hmask : maskOp.SafeAtR R bounds s)
    (hother : otherOp.SafeAtR R bounds s)
    (hb : ∀ ptrs, evalOpR R ptrOp s = some ptrs →
      ∀ i : TileIndex shape, (ptrs.data i).2 < bounds (ptrs.data i).1) :
    (Op.load .real (MemAccess.ptr ptrOp) (MaskOpt.maskOther maskOp otherOp)).SafeAtR
      R bounds s := by
  simp only [Op.SafeAtR]
  exact ⟨hptr, ⟨hmask, hother⟩, fun ptrs hptrs i _ => hb ptrs hptrs i⟩

/-- Safety of a `.real` masked pointer store: children safe, and every lane's
pointer in bounds. -/
private theorem bmmIO_storeSafeR {shape : TileShape} (R : RoundingModel)
    (bounds : RegionBounds) (s : BlockState)
    (ptrOp : Op .ptr shape) (valOp : Op .real shape) (maskOp : Op .bool shape)
    (hptr : ptrOp.SafeAtR R bounds s) (hval : valOp.SafeAtR R bounds s)
    (hmask : maskOp.SafeAtR R bounds s)
    (hb : ∀ ptrs, evalOpR R ptrOp s = some ptrs →
      ∀ i : TileIndex shape, (ptrs.data i).2 < bounds (ptrs.data i).1) :
    Stmt.TraceSafeR R bounds (Stmt.store .real shape (MemAccess.ptr ptrOp) valOp
      (MaskOpt.mask maskOp)) s := by
  simp only [Stmt.TraceSafeR, MemAccess.SafeAtR, MaskOpt.SafeAtR]
  exact ⟨hptr, hval, hmask, fun ptrs hptrs i _ => hb ptrs hptrs i⟩

set_option maxHeartbeats 4000000 in
/-- **Weak K-block body + trace safety.** One iteration of the K-loop from an
*arbitrary* `BmmSafeCtx` state at block `c`: the five statements are trace-safe
(the only two memory accesses are the masked `a`/`b` loads, whose lanes are
exactly the skin's `read1` / `read2` windows at step `c`) and step successfully,
re-establishing the weak context at block `c + 1`. This is the shape half of
`bmm_step`, valid without the clean-`undef` precondition. -/
private theorem bmm_bodyW (R : RoundingModel) (bounds : RegionBounds) (A B : RegionName)
    (PB PC PH PM PN chunk_size BM BN SAB SAS SAH SAK SBB SBS SBH SBK BK numKBlocks : Nat)
    (hmlt : ∀ i : Fin BM, PM * BM + i.val < chunk_size)
    (hnlt : ∀ j : Fin BN, PN * BN + j.val < chunk_size)
    (hbr1 : ∀ (t : Fin numKBlocks) (i : Fin BM) (e : Fin BK),
      batchOff PB PC PH chunk_size SAB SAS SAH + (PM * BM + i.val) * SAS
        + (t.val * BK + e.val) * SAK < bounds A)
    (hbr2 : ∀ (t : Fin numKBlocks) (e : Fin BK) (j : Fin BN),
      batchOff PB PC PH chunk_size SBB SBS SBH + (t.val * BK + e.val) * SBK
        + (PN * BN + j.val) * SBS < bounds B)
    (c : Nat) (hclt : c < numKBlocks) (s : BlockState)
    (hctx : BmmSafeCtx A B PB PC PH PM PN chunk_size BM BN SAB SAS SAH SAK SBB SBS SBH SBK
      BK c s) :
    Stmt.TraceSafeListR R bounds (bmmLoopBody BM BN BK chunk_size (BK * numKBlocks) SAK SBK)
        (s.setReg "k" .nat [] (Tile.scalar c))
      ∧ ∃ s', stepStmts (bmmLoopBody BM BN BK chunk_size (BK * numKBlocks) SAK SBK)
            (s.setReg "k" .nat [] (Tile.scalar c)) = some s'
          ∧ BmmSafeCtx A B PB PC PH PM PN chunk_size BM BN SAB SAS SAH SAK SBB SBS SBH SBK
              BK (c + 1) s' := by
  obtain ⟨hpb, hpc, hph, hpm, hpn, hmm, hnn, hkk0, hap, hbp, ⟨accT, hacc⟩⟩ := hctx
  set K := BK * numKBlocks with hKdef
  have hlt : ∀ e : Fin BK, e.val < K - c * BK := by
    intro e
    have hcK : c * BK + BK ≤ K := by
      rw [hKdef]; calc c * BK + BK = (c + 1) * BK := by ring
        _ ≤ numKBlocks * BK := Nat.mul_le_mul_right _ hclt
        _ = BK * numKBlocks := Nat.mul_comm _ _
    omega
  set apT : Tile .ptr [BM, BK] :=
    ⟨fun idx : TileIndex [BM, BK] => (A.cast, batchOff PB PC PH chunk_size SAB SAS SAH
      + (PM * BM + idx.1.val) * SAS + idx.2.1.val * SAK + c * BK * SAK)⟩ with hapT
  set bpT : Tile .ptr [BK, BN] :=
    ⟨fun idx : TileIndex [BK, BN] => (B.cast, batchOff PB PC PH chunk_size SBB SBS SBH
      + idx.1.val * SBK + (PN * BN + idx.2.1.val) * SBS + c * BK * SBK)⟩ with hbpT
  -- the two window-bound facts, at this block, in the pointer tiles' own spelling
  have hbA : ∀ idx : TileIndex [BM, BK], (apT.data idx).2 < bounds (apT.data idx).1 := by
    intro idx
    have h1 : (apT.data idx).1 = A := by simp [hapT]
    have h2 : (apT.data idx).2 = batchOff PB PC PH chunk_size SAB SAS SAH
        + (PM * BM + idx.1.val) * SAS + (c * BK + idx.2.1.val) * SAK := by
      simp only [hapT]; ring
    rw [h1, h2]
    exact hbr1 ⟨c, hclt⟩ idx.1 idx.2.1
  have hbB : ∀ idx : TileIndex [BK, BN], (bpT.data idx).2 < bounds (bpT.data idx).1 := by
    intro idx
    have h1 : (bpT.data idx).1 = B := by simp [hbpT]
    have h2 : (bpT.data idx).2 = batchOff PB PC PH chunk_size SBB SBS SBH
        + (c * BK + idx.1.val) * SBK + (PN * BN + idx.2.1.val) * SBS := by
      simp only [hbpT]; ring
    rw [h1, h2]
    exact hbr2 ⟨c, hclt⟩ idx.1 idx.2.1
  -- the loop-set state
  set sk := s.setReg "k" .nat [] (Tile.scalar c) with hsk
  have hapk : sk.regs .ptr [BM, BK] "a_ptrs" = some apT := by simp [hsk, hap, hapT]
  have hbpk : sk.regs .ptr [BK, BN] "b_ptrs" = some bpT := by simp [hsk, hbp, hbpT]
  have hacck : sk.regs .real [BM, BN] "acc" = some accT := by simp [hsk, hacc]
  have hmk : sk.regs .nat [BM] "offs_m"
      = some (Tile.vec (fun i : Fin BM => PM * BM + i.val)) := by simp [hsk, hmm]
  have hnk : sk.regs .nat [BN] "offs_n"
      = some (Tile.vec (fun j : Fin BN => PN * BN + j.val)) := by simp [hsk, hnn]
  have hkk : sk.regs .nat [BK] "offs_k"
      = some (Tile.vec (fun e : Fin BK => e.val)) := by simp [hsk, hkk0]
  have hkkv : sk.regs .nat [] "k" = some (Tile.scalar c) := by simp [hsk]
  obtain ⟨amt, ham_eval, ham_true⟩ := amask_alltrue sk BM BK chunk_size K c _ hmk hkk hkkv hmlt hlt
  -- statement 0: the masked `a` load
  set asub : Tile .real [BM, BK] :=
    ⟨fun idx => some (sk.readMem (apT.data idx).1 (apT.data idx).2)⟩ with hasub
  have E0 := stepStmt_assign_eq_some (name := "a")
    (load_ptr_maskOther_alltrue (Op.ref .ptr [BM, BK] "a_ptrs") _ _ sk apT amt _
      (by rw [evalOp_ref]; exact hapk) ham_eval (other_broadcast_eval sk [BM, BK]) ham_true)
  set sk1 := sk.setReg "a" .real [BM, BK] asub with hsk1
  have hbpk1 : sk1.regs .ptr [BK, BN] "b_ptrs" = some bpT := by simp [hsk1, hbpk]
  have hkk1 : sk1.regs .nat [BK] "offs_k"
      = some (Tile.vec (fun e : Fin BK => e.val)) := by simp [hsk1, hkk]
  have hnk1 : sk1.regs .nat [BN] "offs_n"
      = some (Tile.vec (fun j : Fin BN => PN * BN + j.val)) := by simp [hsk1, hnk]
  have hkkv1 : sk1.regs .nat [] "k" = some (Tile.scalar c) := by simp [hsk1, hkkv]
  obtain ⟨bmt, hbm_eval, hbm_true⟩ :=
    bmask_alltrue sk1 BN BK chunk_size K c _ hkk1 hnk1 hkkv1 hnlt hlt
  -- statement 1: the masked `b` load
  set bsub : Tile .real [BK, BN] :=
    ⟨fun idx => some (sk1.readMem (bpT.data idx).1 (bpT.data idx).2)⟩ with hbsub
  have E1 := stepStmt_assign_eq_some (name := "b")
    (load_ptr_maskOther_alltrue (Op.ref .ptr [BK, BN] "b_ptrs") _ _ sk1 bpT bmt _
      (by rw [evalOp_ref]; exact hbpk1) hbm_eval (other_broadcast_eval sk1 [BK, BN]) hbm_true)
  set sk2 := sk1.setReg "b" .real [BK, BN] bsub with hsk2
  -- statement 2: acc += tl.dot(a, b)
  have E2 := stepStmt_assign_eq_some (name := "acc")
    (accdot_op_eval BM BK BN sk2 accT asub bsub
      (by simp [hsk2, hsk1, hacck]) (by simp [hsk2, hsk1]) (by simp [hsk2]))
  set sk3 := sk2.setReg "acc" .real [BM, BN]
    (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
      accT (Tile.dot [] asub bsub)) with hsk3
  -- statement 3: a_ptrs advance
  have E3 := stepStmt_assign_eq_some (name := "a_ptrs")
    (aptr_adv_eval sk3 BM BK BK SAK apT (by simp [hsk3, hsk2, hsk1, hapk]))
  set sk4 := sk3.setReg "a_ptrs" .ptr [BM, BK]
    (Tile.ptrAdd Broadcast.scalarR apT (Tile.scalar (BK * SAK))) with hsk4
  -- statement 4: b_ptrs advance
  have E4 := stepStmt_assign_eq_some (name := "b_ptrs")
    (bptr_adv_eval sk4 BK BN BK SBK bpT (by simp [hsk4, hsk3, hsk2, hsk1, hbpk]))
  set sk5 := sk4.setReg "b_ptrs" .ptr [BK, BN]
    (Tile.ptrAdd Broadcast.scalarR bpT (Tile.scalar (BK * SBK))) with hsk5
  have castFreeAll := bmmIO_loopBody_stmt_castFree R chunk_size K SAK SBK BM BN BK
  constructor
  · -- trace safety of the five statements
    unfold bmmLoopBody
    refine bmmIO_tsl_cons (castFreeAll _ (by unfold bmmLoopBody; simp)) E0 ?_ ?_
    · simp only [Stmt.TraceSafeR]
      refine bmmIO_loadSafeAtR R bounds sk _ _ _ (by simp [Op.SafeAtR]) ?_
        (by simp [Op.SafeAtR]) ?_
      · repeat' first
          | exact trivial
          | apply And.intro
          | exact (bmmIO_safeAtR_expandDim R bounds _ _ _).mpr (by simp [Op.SafeAtR])
          | simp [Op.SafeAtR]
      · intro ptrs hptrs idx
        rw [evalOpR_ref, hapk] at hptrs
        obtain rfl := Option.some.inj hptrs
        exact hbA idx
    refine bmmIO_tsl_cons (castFreeAll _ (by unfold bmmLoopBody; simp)) E1 ?_ ?_
    · simp only [Stmt.TraceSafeR]
      refine bmmIO_loadSafeAtR R bounds sk1 _ _ _ (by simp [Op.SafeAtR]) ?_
        (by simp [Op.SafeAtR]) ?_
      · repeat' first
          | exact trivial
          | apply And.intro
          | exact (bmmIO_safeAtR_expandDim R bounds _ _ _).mpr (by simp [Op.SafeAtR])
          | simp [Op.SafeAtR]
      · intro ptrs hptrs idx
        rw [evalOpR_ref, hbpk1] at hptrs
        obtain rfl := Option.some.inj hptrs
        exact hbB idx
    refine bmmIO_tsl_cons (castFreeAll _ (by unfold bmmLoopBody; simp)) E2 ?_ ?_
    · simp only [Stmt.TraceSafeR]
      repeat' first
        | exact trivial
        | apply And.intro
        | apply (bmmIO_safeAtR_dot R bounds _ _ _).mpr
        | simp [Op.SafeAtR]
    refine bmmIO_tsl_cons (castFreeAll _ (by unfold bmmLoopBody; simp)) E3 ?_ ?_
    · simp only [Stmt.TraceSafeR]; simp [Op.SafeAtR]
    refine bmmIO_tsl_cons (castFreeAll _ (by unfold bmmLoopBody; simp)) E4 ?_ ?_
    · simp only [Stmt.TraceSafeR]; simp [Op.SafeAtR]
    exact Stmt.TraceSafeListR.nil_intro
  · -- the step itself
    refine ⟨sk5, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · unfold bmmLoopBody
      rw [stepStmts.cons_some E0, stepStmts.cons_some E1, stepStmts.cons_some E2,
        stepStmts.cons_some E3, stepStmts.cons_some E4, stepStmts.nil]
    · simp [hsk5, hsk4, hsk3, hsk2, hsk1, hsk, hpb]
    · simp [hsk5, hsk4, hsk3, hsk2, hsk1, hsk, hpc]
    · simp [hsk5, hsk4, hsk3, hsk2, hsk1, hsk, hph]
    · simp [hsk5, hsk4, hsk3, hsk2, hsk1, hsk, hpm]
    · simp [hsk5, hsk4, hsk3, hsk2, hsk1, hsk, hpn]
    · simp [hsk5, hsk4, hsk3, hsk2, hsk1, hsk, hmm]
    · simp [hsk5, hsk4, hsk3, hsk2, hsk1, hsk, hnn]
    · simp [hsk5, hsk4, hsk3, hsk2, hsk1, hsk, hkk0]
    · simp only [hsk5, hsk4, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
        String.reduceEq, not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff, Option.some.injEq]
      ext idx <;> simp only [Tile.ptrAdd, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
        Tile.scalar, hapT, NumericDType.add]
      ring
    · simp only [hsk5, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
        String.reduceEq, not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff, Option.some.injEq]
      ext idx <;> simp only [Tile.ptrAdd, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
        Tile.scalar, hbpT, NumericDType.add]
      ring
    · exact ⟨Tile.bop NumericDType.real.add
        (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) accT (Tile.dot [] asub bsub),
        by simp [hsk5, hsk4, hsk3]⟩

set_option maxHeartbeats 2000000 in
/-- **Weak K-loop driver + safety**: the `forRangeDyn "k" 0 (cdiv K BK) 1`
statement runs to completion from any `BmmSafeCtx` state at block `0`, is
trace-safe throughout, and lands in the weak context at block `numKBlocks`. -/
private theorem bmm_loopW (R : RoundingModel) (bounds : RegionBounds) (A B : RegionName)
    (PB PC PH PM PN chunk_size BM BN SAB SAS SAH SAK SBB SBS SBH SBK BK numKBlocks : Nat)
    (hBK : 0 < BK)
    (hmlt : ∀ i : Fin BM, PM * BM + i.val < chunk_size)
    (hnlt : ∀ j : Fin BN, PN * BN + j.val < chunk_size)
    (hbr1 : ∀ (t : Fin numKBlocks) (i : Fin BM) (e : Fin BK),
      batchOff PB PC PH chunk_size SAB SAS SAH + (PM * BM + i.val) * SAS
        + (t.val * BK + e.val) * SAK < bounds A)
    (hbr2 : ∀ (t : Fin numKBlocks) (e : Fin BK) (j : Fin BN),
      batchOff PB PC PH chunk_size SBB SBS SBH + (t.val * BK + e.val) * SBK
        + (PN * BN + j.val) * SBS < bounds B)
    (s : BlockState)
    (hctx : BmmSafeCtx A B PB PC PH PM PN chunk_size BM BN SAB SAS SAH SAK SBB SBS SBH SBK
      BK 0 s) :
    Stmt.TraceSafeR R bounds (bmmLoopStmt chunk_size SAK SBK BM BN BK numKBlocks) s
      ∧ ∃ s', stepStmt (bmmLoopStmt chunk_size SAK SBK BM BN BK numKBlocks) s = some s'
        ∧ BmmSafeCtx A B PB PC PH PM PN chunk_size BM BN SAB SAS SAH SAK SBB SBS SBH SBK
            BK numKBlocks s' := by
  have hcdiv : (BK * numKBlocks + BK - 1) / BK = numKBlocks := by
    have he : BK * numKBlocks + BK - 1 = (BK - 1) + BK * numKBlocks := by omega
    rw [he, Nat.add_mul_div_left _ _ hBK, Nat.div_eq_of_lt (by omega), Nat.zero_add]
  set P : Nat → BlockState → Prop := fun c st =>
    BmmSafeCtx A B PB PC PH PM PN chunk_size BM BN SAB SAS SAH SAK SBB SBS SBH SBK BK c st ∧
    c ≤ numKBlocks with hP
  have hP0 : P 0 s := ⟨hctx, Nat.zero_le _⟩
  have hbody : ∀ c st, c < numKBlocks → P c st →
      Stmt.TraceSafeListR R bounds (bmmLoopBody BM BN BK chunk_size (BK * numKBlocks) SAK SBK)
        (st.setReg "k" .nat [] (Tile.scalar c)) ∧
      ∃ st', stepStmts (bmmLoopBody BM BN BK chunk_size (BK * numKBlocks) SAK SBK)
          (st.setReg "k" .nat [] (Tile.scalar c)) = some st' ∧ P (c + 1) st' := by
    intro c st hclt hPc
    obtain ⟨hctxc, _⟩ := hPc
    obtain ⟨hsafe, st', hrun, hctx'⟩ :=
      bmm_bodyW R bounds A B PB PC PH PM PN chunk_size BM BN SAB SAS SAH SAK SBB SBS SBH SBK
        BK numKBlocks hmlt hnlt hbr1 hbr2 c hclt st hctxc
    exact ⟨hsafe, st', hrun, hctx', by omega⟩
  constructor
  · simp only [bmmLoopStmt, Stmt.TraceSafeR]
    refine ⟨by simp [Op.SafeAtR], by simp [Op.SafeAtR], by simp [Op.SafeAtR], ?_⟩
    rw [show evalOpR R (Op.constNat 0) s = some (Tile.scalar 0) from by simp only [evalOpR],
      show evalOpR R (Op.div .nat Broadcast.nil (Op.sub .nat Broadcast.nil
          (Op.add .nat Broadcast.nil (Op.constNat (BK * numKBlocks)) (Op.constNat BK))
          (Op.constNat 1)) (Op.constNat BK)) s = some (Tile.scalar numKBlocks) from by
        simp only [evalOpR.eq_def, evalOp.eq_def]
        simp only [Option.bind_some, bind, Tile.bop, Tile.scalar, Broadcast.leftIndex,
          Broadcast.rightIndex, NumericDType.div, NumericDType.sub, NumericDType.add]
        rw [hcdiv],
      show evalOpR R (Op.constNat 1) s = some (Tile.scalar 1) from by simp only [evalOpR]]
    exact Stmt.forRangeTraceSafeR_inv R bounds "k" numKBlocks 1 _ P
      (fun c st hlt hPc => by
        obtain ⟨hsafe, st', hrun, hP'⟩ := hbody c st hlt hPc
        exact ⟨hsafe, st', by
          rw [bmmIO_loopBody_castFree R chunk_size (BK * numKBlocks) SAK SBK BM BN BK]
          exact hrun, hP'⟩)
      0 s hP0
  · have hresolve : stepStmt (bmmLoopStmt chunk_size SAK SBK BM BN BK numKBlocks) s
        = stepForRangeAux "k" 0 numKBlocks 1
            (bmmLoopBody BM BN BK chunk_size (BK * numKBlocks) SAK SBK) s := by
      rw [bmmLoopStmt, stepForRangeAux.forRangeDyn_unfold]
      simp only [evalOp_constNat, evalOp_div, evalOp_sub, evalOp_add, Tile.scalar, Tile.bop,
        Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.div, NumericDType.sub,
        NumericDType.add, Tile.data, Option.bind_some, bind]
      rw [hcdiv]
    rw [hresolve]
    obtain ⟨final, sLoop, haux, hfinal, hPfinal⟩ :=
      forRangeAux_inv (idx := "k") (stop := numKBlocks) (step := 1) (P := P)
        (by norm_num) (fun c st hlt hPc => (hbody c st hlt hPc).2) 0 s hP0
    obtain ⟨hctxF, hleF⟩ := hPfinal
    have hfinalEq : final = numKBlocks := by omega
    subst hfinalEq
    exact ⟨sLoop, haux, hctxF⟩

set_option maxHeartbeats 2000000 in
/-- **Post-loop trace safety**: the two index refreshes, the `out = acc` copy, the
two output-pointer assigns and the terminal masked `.real` store, whose lanes hit
exactly the skin's `write` window (in bounds by `hbw`). -/
private theorem bmm_postSafeR (R : RoundingModel) (bounds : RegionBounds)
    (A B Out : RegionName)
    (PB PC PH PM PN chunk_size BM BN SAB SAS SAH SAK SBB SBS SBH SBK
      SOB SOC SOH SOM SON BK c : Nat)
    (st : BlockState)
    (hctx : BmmSafeCtx A B PB PC PH PM PN chunk_size BM BN SAB SAS SAH SAK SBB SBS SBH SBK
      BK c st)
    (hbw : ∀ idx : TileIndex [BM, BN],
      outOffset PB PC PH chunk_size SOB SOC SOH SOM SON PM PN BM BN idx < bounds Out) :
    Stmt.TraceSafeListR R bounds (bmmPostBody Out chunk_size SOB SOC SOH SOM SON BM BN) st := by
  obtain ⟨hpb, hpc, hph, hpm, hpn, hmm, hnn, hkk, hap, hbp, ⟨accT, hacc⟩⟩ := hctx
  have castAll := bmmIO_postBody_stmt_castFree R Out chunk_size SOB SOC SOH SOM SON BM BN
  -- statement 0: offs_m refresh
  have E0 := stepStmt_assign_eq_some (name := "offs_m") (offs_m_eval st PM BM hpm)
  set v1 := st.setReg "offs_m" .nat [BM] (Tile.vec (fun i : Fin BM => rowIndex PM BM i)) with hv1
  -- statement 1: offs_n refresh
  have E1 := stepStmt_assign_eq_some (name := "offs_n")
    (offs_n_eval v1 PN BN (by simp [hv1, hpn]))
  set v2 := v1.setReg "offs_n" .nat [BN] (Tile.vec (fun j : Fin BN => colIndex PN BN j)) with hv2
  -- statement 2: out = acc
  have E2 := stepStmt_assign_eq_some (name := "out")
    (show evalOp (Op.ref .real [BM, BN] "acc") v2 = some accT by
      rw [evalOp_ref]; simp [hv2, hv1, hacc])
  set v3 := v2.setReg "out" .real [BM, BN] accT with hv3
  -- statement 3: out_ptr base
  have E3 := stepStmt_assign_eq_some (name := "out_ptr")
    (outptr_base_eval v3 Out PB PC PH SOB SOC SOH
      (by simp [hv3, hv2, hv1, hpb]) (by simp [hv3, hv2, hv1, hpc]) (by simp [hv3, hv2, hv1, hph]))
  set v4 := v3.setReg "out_ptr" .ptr [] (Tile.scalar (Out.cast,
    PB * SOB + PC * SOC + PH * SOH)) with hv4
  -- statement 4: out_ptrs
  have E4 := stepStmt_assign_eq_some (name := "out_ptrs")
    (outptrs_eval v4 Out BM BN SOM SON (PB * SOB + PC * SOC + PH * SOH)
      (fun i => rowIndex PM BM i) (fun j => colIndex PN BN j)
      (by simp [hv4]) (by simp [hv4, hv3, hv2, hv1]) (by simp [hv4, hv3, hv2]))
  set opT : Tile .ptr [BM, BN] :=
    ⟨fun idx : TileIndex [BM, BN] => (Out.cast,
      (PB * SOB + PC * SOC + PH * SOH) + SOM * rowIndex PM BM idx.1
        + colIndex PN BN idx.2.1 * SON)⟩ with hopT
  set v5 := v4.setReg "out_ptrs" .ptr [BM, BN] opT with hv5
  have hbOut : ∀ idx : TileIndex [BM, BN], (opT.data idx).2 < bounds (opT.data idx).1 := by
    intro idx
    have h1 : (opT.data idx).1 = Out := by simp [hopT]
    have h2 : (opT.data idx).2
        = outOffset PB PC PH chunk_size SOB SOC SOH SOM SON PM PN BM BN idx := by
      simp only [hopT, outOffset]; ring
    rw [h1, h2]; exact hbw idx
  unfold bmmPostBody
  refine bmmIO_tsl_cons (castAll _ (by unfold bmmPostBody; simp)) E0 ?_ ?_
  · simp only [Stmt.TraceSafeR]; simp [Op.SafeAtR]
  refine bmmIO_tsl_cons (castAll _ (by unfold bmmPostBody; simp)) E1 ?_ ?_
  · simp only [Stmt.TraceSafeR]; simp [Op.SafeAtR]
  refine bmmIO_tsl_cons (castAll _ (by unfold bmmPostBody; simp)) E2 ?_ ?_
  · simp only [Stmt.TraceSafeR]; simp [Op.SafeAtR]
  refine bmmIO_tsl_cons (castAll _ (by unfold bmmPostBody; simp)) E3 ?_ ?_
  · simp only [Stmt.TraceSafeR]; simp [Op.SafeAtR]
  refine bmmIO_tsl_cons (castAll _ (by unfold bmmPostBody; simp)) E4 ?_ ?_
  · simp only [Stmt.TraceSafeR]
    repeat' first
      | exact trivial
      | apply And.intro
      | exact (bmmIO_safeAtR_expandDim R bounds _ _ _).mpr (by simp [Op.SafeAtR])
      | simp [Op.SafeAtR]
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun _ _ => Stmt.TraceSafeListR.nil_intro)
  refine bmmIO_storeSafeR R bounds v5 _ _ _ (by simp [Op.SafeAtR]) (by simp [Op.SafeAtR]) ?_ ?_
  · repeat' first
      | exact trivial
      | apply And.intro
      | exact (bmmIO_safeAtR_expandDim R bounds _ _ _).mpr (by simp [Op.SafeAtR])
      | simp [Op.SafeAtR]
  · intro ptrs hptrs idx
    rw [evalOpR_ref, show v5.regs .ptr [BM, BN] "out_ptrs" = some opT from by
      simp [hv5]] at hptrs
    obtain rfl := Option.some.inj hptrs
    exact hbOut idx

set_option maxHeartbeats 4000000 in
/-- **The whole-kernel `TraceSafeR` walk.** Prologue (register-only), the K-loop
(`bmm_loopW`'s `forRangeTraceSafeR_inv` over the shape-only invariant), then the
terminal masked store. The three bound groups are the skin's `read1` / `read2` /
`write` windows. -/
private theorem bmm_traceSafeR (R : RoundingModel) (bounds : RegionBounds)
    (A B Out : RegionName)
    (chunk_size ngroups SAB SAS SAH SAK SBB SBS SBH SBK SOB SOC SOH SOM SON
      BM BN BK numKBlocks : Nat) (hBK : 0 < BK) (s : BlockState)
    (hmlt : ∀ i : Fin BM, pidM (s.pids 0) chunk_size BN * BM + i.val < chunk_size)
    (hnlt : ∀ j : Fin BN, pidN (s.pids 0) chunk_size BN * BN + j.val < chunk_size)
    (hbr1 : ∀ (t : Fin numKBlocks) (i : Fin BM) (e : Fin BK),
      batchOff (s.pids 1) (pidC (s.pids 2) ngroups) (pidH (s.pids 2) ngroups) chunk_size
          SAB SAS SAH + (pidM (s.pids 0) chunk_size BN * BM + i.val) * SAS
        + (t.val * BK + e.val) * SAK < bounds A)
    (hbr2 : ∀ (t : Fin numKBlocks) (e : Fin BK) (j : Fin BN),
      batchOff (s.pids 1) (pidC (s.pids 2) ngroups) (pidH (s.pids 2) ngroups) chunk_size
          SBB SBS SBH + (t.val * BK + e.val) * SBK
        + (pidN (s.pids 0) chunk_size BN * BN + j.val) * SBS < bounds B)
    (hbw : ∀ idx : TileIndex [BM, BN],
      outOffset (s.pids 1) (pidC (s.pids 2) ngroups) (pidH (s.pids 2) ngroups) chunk_size
        SOB SOC SOH SOM SON (pidM (s.pids 0) chunk_size BN) (pidN (s.pids 0) chunk_size BN)
        BM BN idx < bounds Out) :
    ((bmm_matmul_surface A B Out chunk_size (BK * numKBlocks) ngroups
        SAB SAS SAH SAK SBB SBS SBH SBK SOB SOC SOH SOM SON BM BN BK).toAlgKernel).TraceSafeR
      R bounds s := by
  unfold Kernel.TraceSafeR
  rw [bmm_body_split' A B Out chunk_size ngroups SAB SAS SAH SAK SBB SBS SBH SBK
    SOB SOC SOH SOM SON BM BN BK numKBlocks]
  refine Stmt.TraceSafeListR.append_intro _ _ ?_ ?_
  · -- the prologue: register-only at every state
    refine Stmt.TraceSafeListR.of_forall _ _ ?_
    intro stmt hstmt u
    simp only [bmmPreBody, bmmPidBody, bmmSeedBody, List.cons_append, List.nil_append,
      List.mem_cons, List.not_mem_nil, or_false] at hstmt
    rcases hstmt with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
      rfl | rfl | rfl <;>
      · simp only [Stmt.TraceSafeR]
        repeat' first
          | exact trivial
          | apply And.intro
          | exact (bmmIO_safeAtR_expandDim R bounds _ _ _).mpr (by simp [Op.SafeAtR])
          | simp [Op.SafeAtR]
  · -- after the prologue: the K-loop, then the store tail
    intro s1 hs1
    obtain ⟨s1x, hpre, hctx0⟩ :=
      bmm_preLoopW A B s chunk_size ngroups SAB SAS SAH SAK SBB SBS SBH SBK BM BN BK
    rw [bmmIO_preBody_castFree R A B chunk_size ngroups SAB SAS SAH SAK SBB SBS SBH SBK
      BM BN BK s, hpre] at hs1
    obtain rfl := Option.some.inj hs1
    obtain ⟨hsafeLoop, sLoop, hrunLoop, hctxL⟩ :=
      bmm_loopW R bounds A B (s.pids 1) (pidC (s.pids 2) ngroups) (pidH (s.pids 2) ngroups)
        (pidM (s.pids 0) chunk_size BN) (pidN (s.pids 0) chunk_size BN)
        chunk_size BM BN SAB SAS SAH SAK SBB SBS SBH SBK BK numKBlocks hBK hmlt hnlt
        hbr1 hbr2 s1x hctx0
    refine bmmIO_tsl_cons
      (bmmIO_loopStmt_castFree R chunk_size SAK SBK BM BN BK numKBlocks) hrunLoop hsafeLoop ?_
    exact bmm_postSafeR R bounds A B Out (s.pids 1) (pidC (s.pids 2) ngroups)
      (pidH (s.pids 2) ngroups) (pidM (s.pids 0) chunk_size BN) (pidN (s.pids 0) chunk_size BN)
      chunk_size BM BN SAB SAS SAH SAK SBB SBS SBH SBK SOB SOC SOH SOM SON BK numKBlocks
      sLoop hctxL hbw

/-! ### The terminal masked store on the `R` side

`bmm_postLoop` already runs the six post-loop statements under the exact stepper,
but the skin needs (a) the readback at the `MemCell` layer for `readMemAs .real`
and (b) the full outside-the-window frame, neither of which the exact lemma
exposes. Since the store is `.real`-typed the `R` run is the same run
(`bmmIO_writeMemTypedR_real_eq`); the scatter is re-derived here through the
`writeMemAsR` readback / frame family. -/

/-- The masked `.real` store under `stepStmtR R` reduces to the `writeMemAsR`
masked scatter foldl. -/
private theorem bmmIO_storeR_eval (R : RoundingModel) (BM BN chunk_size : Nat) (st : BlockState)
    (oT : Tile .real [BM, BN]) (opT : Tile .ptr [BM, BN]) (mT : Tile .bool [BM, BN])
    (ho : st.regs .real [BM, BN] "out" = some oT)
    (hop : st.regs .ptr [BM, BN] "out_ptrs" = some opT)
    (hmeval : evalOpR R (Op.boolAnd Broadcast.nil.consL.consR
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m")) (Op.constNat chunk_size))
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n"))
          (Op.constNat chunk_size))) st = some mT) :
    stepStmtR R (Stmt.store .real [BM, BN] (.ptr (Op.ref .ptr [BM, BN] "out_ptrs"))
        (Op.ref .real [BM, BN] "out")
        (.mask (Op.boolAnd Broadcast.nil.consL.consR
          (Op.lt ComparableDType.nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m")) (Op.constNat chunk_size))
          (Op.lt ComparableDType.nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n"))
            (Op.constNat chunk_size))))) st
      = some ((TileShape.allIndices [BM, BN]).foldl
          (fun acc idx =>
            if mT.data idx then
              acc.writeMemAsR R .real (opT.data idx).1 (opT.data idx).2 (oT.data idx)
            else acc) st) := by
  simp only [stepStmtR, evalOpR_ref, ho, hop, hmeval, bind, Option.bind_some, Option.map_some]
  refine congrArg some (List.foldl_ext _ _ st (fun acc idx _ => ?_))
  by_cases hb : mT.data idx
  · simp only [hb, if_true, bmmIO_writeMemTypedR_real_eq]
  · simp only [hb, if_false, Bool.false_eq_true]

set_option maxHeartbeats 4000000 in
/-- **R-postLoop**: from `bmmInvariant` at `numKBlocks` (so `acc = bmmSpec`), the
six post statements terminate under `stepStmtsR R`; every output lane holds the
exact-ℝ cell `MemCell.of .real (bmmSpec …)` — a `.real` store has no rounding
event — with a per-cell frame outside the output window. -/
private theorem bmm_postLoopR (R : RoundingModel) (A B Out : RegionName) (s0 : BlockState)
    (PB PC PH PM PN chunk_size BM BN SAB SAS SAH SAK SBB SBS SBH SBK
      SOB SOC SOH SOM SON BK numKBlocks : Nat)
    (hInj : Function.Injective
      (outOffset PB PC PH chunk_size SOB SOC SOH SOM SON PM PN BM BN))
    (hmlt : ∀ i : Fin BM, rowIndex PM BM i < chunk_size)
    (hnlt : ∀ j : Fin BN, colIndex PN BN j < chunk_size)
    (st : BlockState)
    (hinv : bmmInvariant A B s0 PB PC PH PM PN chunk_size BM BN SAB SAS SAH SAK SBB SBS SBH SBK
      BK numKBlocks numKBlocks st) :
    ∃ sfin, stepStmtsR R (bmmPostBody Out chunk_size SOB SOC SOH SOM SON BM BN) st = some sfin
      ∧ (∀ idx : TileIndex [BM, BN],
          sfin.mem Out (outOffset PB PC PH chunk_size SOB SOC SOH SOM SON PM PN BM BN idx)
            = MemCell.of FloatDType.real.toTileDType (FloatDType.real.ofReal
                (bmmSpec s0 A B PB PC PH PM PN BM BN chunk_size SAB SAS SAH SAK SBB SBS SBH SBK
                  BK numKBlocks idx.1 idx.2.1)))
      ∧ (∀ r o, (r ≠ Out ∨ ∀ idx : TileIndex [BM, BN],
            o ≠ outOffset PB PC PH chunk_size SOB SOC SOH SOM SON PM PN BM BN idx) →
          sfin.mem r o = st.mem r o) := by
  simp only [bmmInvariant] at hinv
  obtain ⟨hpids, hcle, hz, hpb, hpc, hph, hpm, hpn, hmm, hnn, hkk, hap, hbp, hundef, hmem⟩ := hinv
  set g : TileIndex [BM, BN] → ℝ :=
    fun idx => bmmSpec s0 A B PB PC PH PM PN BM BN chunk_size SAB SAS SAH SAK SBB SBS SBH SBK
      BK numKBlocks idx.1 idx.2.1 with hg
  have hzspec : st.regs .real [BM, BN] "acc" = some ⟨fun idx => some (g idx)⟩ := by
    rw [hz]; refine congrArg some ?_; ext idx
    simp only [hg, bmmSpec, accPartial, Nat.mul_comm numKBlocks BK]
  set oT : Tile .real [BM, BN] := (⟨fun idx => some (g idx)⟩ : Tile .real [BM, BN]) with hoT
  unfold bmmPostBody
  -- statement 0: offs_m refresh
  rw [stepStmtsR_cons_some (stepStmtR_assign_eq_some
    (show evalOpR R (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat BM)) (Op.arange BM)) st
      = some (Tile.vec (fun i : Fin BM => rowIndex PM BM i)) from by
      rw [show evalOpR R (Op.add .nat Broadcast.scalarL
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat BM)) (Op.arange BM)) st
        = evalOp (Op.add .nat Broadcast.scalarL
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat BM)) (Op.arange BM)) st
        from by simp only [evalOpR.eq_def, evalOp.eq_def]]
      exact offs_m_eval st PM BM hpm))]
  set w1 := st.setReg "offs_m" .nat [BM] (Tile.vec (fun i : Fin BM => rowIndex PM BM i)) with hw1
  -- statement 1: offs_n refresh
  rw [stepStmtsR_cons_some (stepStmtR_assign_eq_some
    (show evalOpR R (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_n") (Op.constNat BN)) (Op.arange BN)) w1
      = some (Tile.vec (fun j : Fin BN => colIndex PN BN j)) from by
      rw [show evalOpR R (Op.add .nat Broadcast.scalarL
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_n") (Op.constNat BN)) (Op.arange BN)) w1
        = evalOp (Op.add .nat Broadcast.scalarL
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_n") (Op.constNat BN)) (Op.arange BN)) w1
        from by simp only [evalOpR.eq_def, evalOp.eq_def]]
      exact offs_n_eval w1 PN BN (by simp [hw1, hpn])))]
  set w2 := w1.setReg "offs_n" .nat [BN] (Tile.vec (fun j : Fin BN => colIndex PN BN j)) with hw2
  -- statement 2: out = acc
  rw [stepStmtsR_cons_some (stepStmtR_assign_eq_some
    (show evalOpR R (Op.ref .real [BM, BN] "acc") w2 = some oT from by
      rw [evalOpR_ref]; simp [hw2, hw1, hzspec, hoT]))]
  set w3 := w2.setReg "out" .real [BM, BN] oT with hw3
  -- statement 3: out_ptr base
  rw [stepStmtsR_cons_some (stepStmtR_assign_eq_some
    (show evalOpR R (Op.ptrAdd Broadcast.nil (Op.ptrBase Out)
        (Op.add .nat Broadcast.nil
          (Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_b") (Op.constNat SOB))
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_c") (Op.constNat SOC)))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_h") (Op.constNat SOH)))) w3
      = some (Tile.scalar (Out.cast, PB * SOB + PC * SOC + PH * SOH)) from by
      rw [show evalOpR R (Op.ptrAdd Broadcast.nil (Op.ptrBase Out)
          (Op.add .nat Broadcast.nil
            (Op.add .nat Broadcast.nil
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_b") (Op.constNat SOB))
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_c") (Op.constNat SOC)))
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_h") (Op.constNat SOH)))) w3
        = evalOp (Op.ptrAdd Broadcast.nil (Op.ptrBase Out)
          (Op.add .nat Broadcast.nil
            (Op.add .nat Broadcast.nil
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_b") (Op.constNat SOB))
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_c") (Op.constNat SOC)))
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_h") (Op.constNat SOH)))) w3
        from by simp only [evalOpR.eq_def, evalOp.eq_def]]
      exact outptr_base_eval w3 Out PB PC PH SOB SOC SOH
        (by simp [hw3, hw2, hw1, hpb]) (by simp [hw3, hw2, hw1, hpc])
        (by simp [hw3, hw2, hw1, hph])))]
  set w4 := w3.setReg "out_ptr" .ptr []
    (Tile.scalar (Out.cast, PB * SOB + PC * SOC + PH * SOH)) with hw4
  -- statement 4: out_ptrs
  set opT : Tile .ptr [BM, BN] :=
    ⟨fun idx : TileIndex [BM, BN] => (Out.cast,
      (PB * SOB + PC * SOC + PH * SOH) + SOM * rowIndex PM BM idx.1
        + colIndex PN BN idx.2.1 * SON)⟩ with hopT
  rw [stepStmtsR_cons_some (stepStmtR_assign_eq_some
    (show evalOpR R (Op.ptrAdd Broadcast.nil.consL.consR
        (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "out_ptr")
          (Op.mul .nat Broadcast.scalarL (Op.constNat SOM)
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m"))))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n"))
          (Op.constNat SON))) w4 = some opT from by
      rw [show evalOpR R (Op.ptrAdd Broadcast.nil.consL.consR
          (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "out_ptr")
            (Op.mul .nat Broadcast.scalarL (Op.constNat SOM)
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m"))))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n"))
            (Op.constNat SON))) w4
        = evalOp (Op.ptrAdd Broadcast.nil.consL.consR
          (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "out_ptr")
            (Op.mul .nat Broadcast.scalarL (Op.constNat SOM)
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m"))))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n"))
            (Op.constNat SON))) w4
        from by simp only [evalOpR.eq_def, evalOp.eq_def]]
      exact outptrs_eval w4 Out BM BN SOM SON (PB * SOB + PC * SOC + PH * SOH)
        (fun i => rowIndex PM BM i) (fun j => colIndex PN BN j)
        (by simp [hw4]) (by simp [hw4, hw3, hw2, hw1]) (by simp [hw4, hw3, hw2])))]
  set w5 := w4.setReg "out_ptrs" .ptr [BM, BN] opT with hw5
  -- statement 5: the terminal masked `.real` store
  have hm5 : w5.regs .nat [BM] "offs_m"
      = some (Tile.vec (fun i : Fin BM => rowIndex PM BM i)) := by
    simp [hw5, hw4, hw3, hw2, hw1]
  have hn5 : w5.regs .nat [BN] "offs_n"
      = some (Tile.vec (fun j : Fin BN => colIndex PN BN j)) := by simp [hw5, hw4, hw3, hw2]
  obtain ⟨mT, hmask_eval, hmask_true⟩ :=
    outmask_alltrue w5 chunk_size BM BN (fun i => rowIndex PM BM i) (fun j => colIndex PN BN j)
      hm5 hn5 hmlt hnlt
  have hmask_evalR : evalOpR R (Op.boolAnd Broadcast.nil.consL.consR
      (Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m")) (Op.constNat chunk_size))
      (Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n"))
        (Op.constNat chunk_size))) w5 = some mT := by
    rw [show evalOpR R (Op.boolAnd Broadcast.nil.consL.consR
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m")) (Op.constNat chunk_size))
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n"))
          (Op.constNat chunk_size))) w5
      = evalOp (Op.boolAnd Broadcast.nil.consL.consR
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m")) (Op.constNat chunk_size))
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n"))
          (Op.constNat chunk_size))) w5
      from by simp only [evalOpR.eq_def, evalOp.eq_def]]
    exact hmask_eval
  have hout5 : w5.regs .real [BM, BN] "out" = some oT := by simp [hw5, hw4, hw3]
  have hop5 : w5.regs .ptr [BM, BN] "out_ptrs" = some opT := by simp [hw5]
  rw [stepStmtsR_cons_some
    (bmmIO_storeR_eval R BM BN chunk_size w5 oT opT mT hout5 hop5 hmask_evalR), stepStmtsR_nil]
  -- the scatter is unmasked (every lane active) and lands in `Out`
  have hstep_eq :
      (fun (acc : BlockState) k =>
        if mT.data k then acc.writeMemAsR R .real (opT.data k).1 (opT.data k).2 (oT.data k)
        else acc)
        =
      (fun (acc : BlockState) k =>
        if (fun _ : TileIndex [BM, BN] => True) k then
          acc.writeMemAsR R .real Out
            (outOffset PB PC PH chunk_size SOB SOC SOH SOM SON PM PN BM BN k) (oT.data k)
        else acc) := by
    funext acc k
    rw [hmask_true k]
    have h1 : (opT.data k).1 = Out := by simp [hopT]
    have h2 : (opT.data k).2
        = outOffset PB PC PH chunk_size SOB SOC SOH SOM SON PM PN BM BN k := by
      simp only [hopT, outOffset]; ring
    rw [h1, h2]; simp
  rw [hstep_eq]
  refine ⟨_, rfl, ?_, ?_⟩
  · intro idx
    rw [BlockState.scatter_memcell_R_prop_masked_nd R .real (region := Out) w5
        (outOffset PB PC PH chunk_size SOB SOC SOH SOM SON PM PN BM BN)
        (fun k => oT.data k) (fun _ => True) hInj idx,
      if_pos trivial]
    simp only [hoT, RoundingModel.storeValue_real, FloatDType.real_storeValue,
      WithBot.unbotD_some, hg]
  · intro r o hcond
    by_cases hr : r = Out
    · rcases hcond with hne | hno
      · exact absurd hr hne
      · rw [hr, BlockState.foldl_writeMemAsR_preserve_masked_prop R .real
          (outOffset PB PC PH chunk_size SOB SOC SOH SOM SON PM PN BM BN)
          (fun k => oT.data k) (fun _ : TileIndex [BM, BN] => True) o
          (TileShape.allIndices [BM, BN]) (fun k _ _ => (hno k).symm) w5]
        rfl
    · rw [BlockState.foldl_writeMemAsR_preserve_other_region R .real
        (outOffset PB PC PH chunk_size SOB SOC SOH SOM SON PM PN BM BN)
        (fun k => oT.data k) (fun _ : TileIndex [BM, BN] => True) r hr o
        (TileShape.allIndices [BM, BN]) w5]
      rfl

/-! ### The headline -/

set_option maxHeartbeats 4000000 in
/-- **The `⊨[R]` streaming headline (wave-5 S1 fold genre).**

For **every** rounding model `R` — no rounding-boundary hypothesis is needed, the
surface is cast-free — the faithful genuine batched-matmul surface of
`bmm_chunk_fwd` implements, on its `StreamMetaMasked3DKernelIO₂` signature at
`nMeta := 0`, the **ideal ℝ batched matrix product** over the streamed tiles:
every output lane `l = (i, n)` holds

`∑_{t < numKBlocks} ∑_{e < BK} aTile(t)[i, e] · bTile(t)[e, n]`

— exact real arithmetic, the `K = BK · numKBlocks`-long contraction split
row-major into the `numKBlocks` streamed K-blocks. This is `bmmSpec` re-expressed
on the skin's per-step tiles (`bmmSpec_eq_streamSum`).

Layer map: the prologue, the K-loop and the tail are all cast-free (the surface
has no `Op.castFloat`), so under `execR R` they collapse verbatim onto the exact
stepper and the proven `bmm_preLoop` / `bmm_step` / `bmm_loop` invariant tower
above is reused unchanged; only the masked terminal store is re-proved on the `R`
side (`bmm_postLoopR`). The store is `.real`-typed, so the skin's boundary
quantization degenerates: the readback's `R.round .real` is the identity
(`round_real_apply`).

Hypotheses, each inherited from the exact headline
`bmm_chunk_fwd_output_summary_general` (see its docstring), restated in pid form
because the skin quantifies the launch state internally:

* `hBK : 0 < BK` — the K-loop's block size; at `0` the `cdiv K BK` trip count and
  the block index are meaningless. Exact headline's `hBK`.
* `hmlt` / `hnlt` — every tile row / column of every program is `< chunk_size`.
  This is what makes the kernel's own per-block K-tail and `chunk_size` row/col
  load masks and its store mask all-true, i.e. what makes the skin's full-window
  `mask1` / `mask2` / `writeMask` the honest read/write sets. Exact headline's
  `hmlt` / `hnlt`, with `s.pids 0` generalized to `∀ pid₀`.
* `hOutInj` — output-address injectivity of the write window. With colliding
  output lanes the per-lane readback would be last-writer-wins and the statement
  false. As in the exact headline it is carried as an open side condition, not
  discharged.

The exact headline's `hundef` is **not** a hypothesis here: the skin's triple
already pins `s₀.undef = fun _ _ => 0`.

Relation to the exact surface: `bmm_chunk_fwd_output_summary_general`
(`Realizes_without_Rounding`) above is retained unchanged; this `⊨[R]` face
restates the same batched matmul on the streaming skin, for every `R` at once (at
the `.real` grid the two faces carry the same exact cell). Both are kept per the
rounding-as-default doctrine. -/
specification bmm_chunk_fwd_io_correctness (R : RoundingModel) (A B Out : RegionName)
    (chunk_size ngroups SAB SAS SAH SAK SBB SBS SBH SBK SOB SOC SOH SOM SON
      BM BN BK numKBlocks : Nat)
    (hBK : 0 < BK)
    (hmlt : ∀ (pid₀ : Nat) (i : Fin BM), pidM pid₀ chunk_size BN * BM + i.val < chunk_size)
    (hnlt : ∀ (pid₀ : Nat) (j : Fin BN), pidN pid₀ chunk_size BN * BN + j.val < chunk_size)
    (hOutInj : ∀ pid₀ pid₁ pid₂ : Nat,
      Function.Injective (fun idx : TileIndex [BM, BN] =>
        bmmOutAddr pid₀ pid₁ pid₂ ngroups chunk_size BM BN SOB SOC SOH SOM SON
          idx.1.val idx.2.1.val)) :
    bmm_chunk_fwd_IO A B Out chunk_size ngroups SAB SAS SAH SAK SBB SBS SBH SBK
        SOB SOC SOH SOM SON BM BN BK numKBlocks ⊨[R]
      fun _ _ _ _ xs ys l => bmmStreamSum BM BN BK numKBlocks xs ys l := by
  refine StreamMetaMasked3DKernelIO₂.ImplementsR.intro _ ?_ ?_ ?_
  · exact bmm_flattenOk A B Out chunk_size ngroups SAB SAS SAH SAK SBB SBS SBH SBK
      SOB SOC SOH SOM SON BM BN BK numKBlocks
  · -- the safety walk
    intro bounds s m xs ys _hm _hx _hy _hbm hbr1 hbr2 hbw
    have hbr1' : ∀ (t : Fin numKBlocks) (i : Fin BM) (e : Fin BK),
        batchOff (s.pids 1) (pidC (s.pids 2) ngroups) (pidH (s.pids 2) ngroups) chunk_size
            SAB SAS SAH + (pidM (s.pids 0) chunk_size BN * BM + i.val) * SAS
          + (t.val * BK + e.val) * SAK < bounds A := by
      intro t i e
      have h := hbr1 t (Lane2D.encode (i, e, PUnit.unit)) (by trivial)
      simp only [bmm_chunk_fwd_IO, bmmAAddr, Lane2D.encode_div, Lane2D.encode_mod] at h
      exact h
    have hbr2' : ∀ (t : Fin numKBlocks) (e : Fin BK) (j : Fin BN),
        batchOff (s.pids 1) (pidC (s.pids 2) ngroups) (pidH (s.pids 2) ngroups) chunk_size
            SBB SBS SBH + (t.val * BK + e.val) * SBK
          + (pidN (s.pids 0) chunk_size BN * BN + j.val) * SBS < bounds B := by
      intro t e j
      have h := hbr2 t (Lane2D.encode (e, j, PUnit.unit)) (by trivial)
      simp only [bmm_chunk_fwd_IO, bmmBAddr, Lane2D.encode_div, Lane2D.encode_mod] at h
      exact h
    have hbw' : ∀ idx : TileIndex [BM, BN],
        outOffset (s.pids 1) (pidC (s.pids 2) ngroups) (pidH (s.pids 2) ngroups) chunk_size
          SOB SOC SOH SOM SON (pidM (s.pids 0) chunk_size BN) (pidN (s.pids 0) chunk_size BN)
          BM BN idx < bounds Out := by
      intro idx
      have h := hbw (Lane2D.encode idx) (by trivial)
      simp only [bmm_chunk_fwd_IO, bmmOutAddr, Lane2D.encode_div, Lane2D.encode_mod] at h
      exact h
    exact bmm_traceSafeR R bounds A B Out chunk_size ngroups SAB SAS SAH SAK SBB SBS SBH SBK
      SOB SOC SOH SOM SON BM BN BK numKBlocks hBK s (hmlt (s.pids 0)) (hnlt (s.pids 0))
      hbr1' hbr2' hbw'
  · -- the rounded Hoare triple
    intro s₀ m xs ys hundef _hm hx hy
    have hundef' : ∀ rg o, s₀.undef rg o = 0 := fun rg o => by rw [hundef]
    have hInj' : Function.Injective
        (outOffset (s₀.pids 1) (pidC (s₀.pids 2) ngroups) (pidH (s₀.pids 2) ngroups) chunk_size
          SOB SOC SOH SOM SON (pidM (s₀.pids 0) chunk_size BN)
          (pidN (s₀.pids 0) chunk_size BN) BM BN) :=
      hOutInj (s₀.pids 0) (s₀.pids 1) (s₀.pids 2)
    -- the exact prologue and the exact K-loop (both cast-free, so they *are* the `R` run)
    obtain ⟨s1, hpre, hP0⟩ := bmm_preLoop A B Out s₀ chunk_size (BK * numKBlocks) ngroups
      SAB SAS SAH SAK SBB SBS SBH SBK SOB SOC SOH SOM SON BM BN BK numKBlocks hundef'
    rw [bmm_take15_eq A B Out chunk_size ngroups SAB SAS SAH SAK SBB SBS SBH SBK
      SOB SOC SOH SOM SON BM BN BK numKBlocks] at hpre
    obtain ⟨sLoop, hloop, hPloop⟩ := bmm_loop A B s₀ (s₀.pids 1) (pidC (s₀.pids 2) ngroups)
      (pidH (s₀.pids 2) ngroups) (pidM (s₀.pids 0) chunk_size BN)
      (pidN (s₀.pids 0) chunk_size BN) chunk_size BM BN SAB SAS SAH SAK SBB SBS SBH SBK
      BK numKBlocks hBK (hmlt (s₀.pids 0)) (hnlt (s₀.pids 0)) s1 hP0
    obtain ⟨sfin, hTailR, hval, hframe⟩ := bmm_postLoopR R A B Out s₀
      (s₀.pids 1) (pidC (s₀.pids 2) ngroups) (pidH (s₀.pids 2) ngroups)
      (pidM (s₀.pids 0) chunk_size BN) (pidN (s₀.pids 0) chunk_size BN)
      chunk_size BM BN SAB SAS SAH SAK SBB SBS SBH SBK SOB SOC SOH SOM SON BK numKBlocks
      hInj' (hmlt (s₀.pids 0)) (hnlt (s₀.pids 0)) sLoop hPloop
    have hLoopR : stepStmtR R (bmmLoopStmt chunk_size SAK SBK BM BN BK numKBlocks) s1
        = some sLoop := by
      rw [bmmIO_loopStmt_castFree R chunk_size SAK SBK BM BN BK numKBlocks]
      exact hloop
    have hmemL : sLoop.mem = s₀.mem := by
      simp only [bmmInvariant] at hPloop; exact hPloop.2.2.2.2.2.2.2.2.2.2.2.2.2.2
    refine ⟨sfin, ?_, ?_, ?_⟩
    · show execR R (bmm_matmul_surface A B Out chunk_size (BK * numKBlocks) ngroups
          SAB SAS SAH SAK SBB SBS SBH SBK SOB SOC SOH SOM SON BM BN BK).toAlgKernel s₀
        = some sfin
      unfold execR
      rw [bmm_body_split' A B Out chunk_size ngroups SAB SAS SAH SAK SBB SBS SBH SBK
          SOB SOC SOH SOM SON BM BN BK numKBlocks,
        stepStmtsR_append,
        bmmIO_preBody_castFree R A B chunk_size ngroups SAB SAS SAH SAK SBB SBS SBH SBK
          BM BN BK s₀,
        hpre, Option.bind_some, stepStmtsR_cons_some hLoopR]
      exact hTailR
    · -- the readback: `bmmSpec` at the decoded lane is the streamed fold
      intro j _hj
      simp only [bmm_chunk_fwd_IO]
      rw [show bmmOutAddr (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) ngroups chunk_size BM BN
            SOB SOC SOH SOM SON (j.val / BN) (j.val % BN)
          = outOffset (s₀.pids 1) (pidC (s₀.pids 2) ngroups) (pidH (s₀.pids 2) ngroups)
              chunk_size SOB SOC SOH SOM SON (pidM (s₀.pids 0) chunk_size BN)
              (pidN (s₀.pids 0) chunk_size BN) BM BN (Lane2D.decode j) from rfl,
        bmmIO_readMemAs_real_of_cell (hval (Lane2D.decode j)), R.round_real_apply]
      refine congrArg _ ?_
      have hx' : ∀ (t : Fin numKBlocks) (i : Fin BM) (e : Fin BK),
          aElem s₀ A (s₀.pids 1) (pidC (s₀.pids 2) ngroups) (pidH (s₀.pids 2) ngroups)
              (pidM (s₀.pids 0) chunk_size BN) BM chunk_size SAB SAS SAH SAK i
              (t.val * BK + e.val)
            = xs t (Lane2D.encode (i, e, PUnit.unit)) := by
        intro t i e
        have h := hx t (Lane2D.encode (i, e, PUnit.unit)) (by trivial)
        simp only [bmm_chunk_fwd_IO, Lane2D.encode_div, Lane2D.encode_mod] at h
        rw [bmmAAddr_eq s₀ A ngroups chunk_size BM BN BK SAB SAS SAH SAK t.val i e]
        exact h
      have hy' : ∀ (t : Fin numKBlocks) (e : Fin BK) (n : Fin BN),
          bElem s₀ B (s₀.pids 1) (pidC (s₀.pids 2) ngroups) (pidH (s₀.pids 2) ngroups)
              (pidN (s₀.pids 0) chunk_size BN) BN chunk_size SBB SBS SBH SBK n
              (t.val * BK + e.val)
            = ys t (Lane2D.encode (e, n, PUnit.unit)) := by
        intro t e n
        have h := hy t (Lane2D.encode (e, n, PUnit.unit)) (by trivial)
        simp only [bmm_chunk_fwd_IO, Lane2D.encode_div, Lane2D.encode_mod] at h
        rw [bmmBAddr_eq s₀ B ngroups chunk_size BN BK SBB SBS SBH SBK t.val e n]
        exact h
      exact bmmSpec_eq_streamSum s₀ A B (s₀.pids 1) (pidC (s₀.pids 2) ngroups)
        (pidH (s₀.pids 2) ngroups) (pidM (s₀.pids 0) chunk_size BN)
        (pidN (s₀.pids 0) chunk_size BN) chunk_size BM BN SAB SAS SAH SAK SBB SBS SBH SBK
        BK numKBlocks xs ys hx' hy' j
    · -- the frame
      intro r o hcond
      have hcond' : r ≠ Out ∨ ∀ idx : TileIndex [BM, BN],
          o ≠ outOffset (s₀.pids 1) (pidC (s₀.pids 2) ngroups) (pidH (s₀.pids 2) ngroups)
            chunk_size SOB SOC SOH SOM SON (pidM (s₀.pids 0) chunk_size BN)
            (pidN (s₀.pids 0) chunk_size BN) BM BN idx := by
        rcases hcond with hne | hno
        · exact Or.inl hne
        · refine Or.inr fun idx => ?_
          have h := hno (Lane2D.encode idx) (by trivial)
          simp only [bmm_chunk_fwd_IO, bmmOutAddr, Lane2D.encode_div, Lane2D.encode_mod] at h
          exact h
      rw [hframe r o hcond', hmemL]

end IOFace

end VeriTile.Bench.TritonBenchG.BmmChunkFwd
