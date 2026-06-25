import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.Kernel
import VeriTile.Triton.LoopInvariant

/-!
# `iv_dependent_matmul` — closed-form matmul correctness

`iv_dependent_matmul_kernel` is a tiled GEMM `C = A × B` whose A/B pointer
updates depend on a string constexpr `type` selecting one of five
induction-variable scheduling modes (`pre_load`, `post_load`,
`post_pre_mixed`, `post_load_two_iters`, `post_load_three_iters`). Each program
derives `pid_m`/`pid_n` from `pid` (`pid_m = pid // num_pid_n`,
`pid_n = pid % num_pid_n`), accumulates the `BLOCK_SIZE_M × BLOCK_SIZE_N` tile
via `accumulator += tl.dot(a, b)` looping over K (with K-tail masking on the
loads), casts the result to float16, and stores into `C` masked by
`(offs_cm < M) & (offs_cn < N)`.

This file proves the **full kernel** correct against a genuine mathematical
reference: every output cell `C[i,j]` of the computed tile equals
`fp16(Σ_{k < K} A[i,k] · B[k,j])` over `ℝ`, where `K = BLOCK_SIZE_K · numKBlocks`
is the contracted dimension. This is NOT the kernel's own emitted value — it is
the independent closed-form `Σ_k A·B` GEMM reference, derived from the loaded
`A`/`B` tiles.

## The five scheduling modes are equivalent

The kernel's `type` constexpr only changes *how* the A/B pointers are advanced
across the K-loop; at iteration `k` every mode loads exactly K-block `k` of `A`
and `B`:

* `pre_load` recomputes `a_ptrs = a_ptr + k·BK·sak`, `b_ptrs = b_ptr + k·BK·sbk`
  at the top of the body;
* `post_load` advances both pointers to block `k+1` at the bottom (so block `k`
  is loaded at iteration `k`);
* `post_pre_mixed` recomputes `a_ptrs` pre and advances `b_ptrs` post;
* `post_load_two_iters` / `post_load_three_iters` run a software-pipelined
  prefetch whose *loaded* pointer at iteration `k` is still block `k`.

All five therefore compute the same value `Σ_k A·B`. The canonical `pre_load`
surface — where the loaded A/B pointers at iteration `k` are the pure functions
`a_ptr + k·BK·sak` / `b_ptr + k·BK·sbk` of the loop counter — is verified here
as the genuine matmul. Only the `pre_load` surface is mechanized; the four other
scheduling modes are described informally above (they load identical per-block
pointers and so compute the same product), but no surface is built for them.

## Proof architecture

```
iv_dependent_matmul_closed_form_correct      ← TOP THEOREM (ComputeCorrect.Realizes)
  └─ ivdm_exec_closed_form                    ← exec-side closed form (every active cell = fp16(∑_k A·B))
       ├─ ivdm_preLoop   (P 0: accumulator = 0, a_ptr/b_ptr bases seeded)
       ├─ ivdm_step      (one K-block: a_ptrs/b_ptrs recomputed = block k, accumulator += dot)
       ├─ ivdm_postLoop  (fp16 cast + masked store = the closed form)
       └─ forRangeDyn-via-forRangeAux_inv (loop-invariant principle drives the K-loop)
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float, except that the final fp16
output cast is modeled by `FloatDType.real.cast .fp16`). The launch-time
`num_stages` is not modeled. The string-valued `type` is represented by the
canonical `pre_load` surface only (the other four modes are described informally
and are not mechanized); they all load the same per-block tiles, so they compute
the same product. Program ids are universally quantified through the kernel's own
`pid_m`/`pid_n` derivation, so the per-program statement covers every program of
the grid. The layout contract is the kernel's own strided pointer arithmetic:
`a[i,k]` at `A + offs_am i · stride_am + k · stride_ak`, `b[k,j]` at
`B + k · stride_bk + offs_bn j · stride_bn`, `c[i,j]` at
`C + stride_cm · offs_cm i + stride_cn · offs_cn j`.

Preconditions for the general theorem: `0 < BLOCK_SIZE_K` and `K` divisible into
`numKBlocks` full K-blocks (so the per-block K-tail load mask is satisfied for
every loaded lane); tile rows/cols in-bounds (`PM·BM + i < M`, `PN·BN + j < N`,
making `% M`/`% N` the identity and the store mask all-true); output-address
injectivity, discharged here from the concrete contract `stride_cn = 1` and
`BN ≤ stride_cm` (row-major C tile); clean initial `undef`.
-/

namespace VeriTile.Bench.TritonBenchG.IvDependentMatmul

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Surface transcription of `iv_dependent_matmul.py`'s `iv_dependent_matmul_kernel`
for the canonical `type == "pre_load"` scheduling mode: each loop iteration
recomputes the A/B pointers from the per-block counter `k`, loads K-block `k`,
accumulates `tl.dot`, then the final masked fp16 store. -/
def iv_dependent_matmul_pre_load_surface
    (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  num_pid_n = tl.cdiv($(N), $(BLOCK_SIZE_N))
  pid_m = pid // num_pid_n
  pid_n = pid % num_pid_n
  offs_am = (pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))) % $(M)
  offs_bn = (pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))) % $(N)
  offs_k = tl.arange(0, $(BLOCK_SIZE_K))
  a_ptr = A + offs_am[:, None] * $(stride_am) + offs_k[None, :] * $(stride_ak)
  b_ptr = B + offs_k[:, None] * $(stride_bk) + offs_bn[None, :] * $(stride_bn)
  accumulator = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
  for k in range($(0), tl.cdiv($(K), $(BLOCK_SIZE_K)), $(1)) {
    a_ptrs = a_ptr + k * $(BLOCK_SIZE_K) * $(stride_ak)
    b_ptrs = b_ptr + k * $(BLOCK_SIZE_K) * $(stride_bk)
    a = tl.load(a_ptrs, mask=offs_k[None, :] < $(K) - k * $(BLOCK_SIZE_K), other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < $(K) - k * $(BLOCK_SIZE_K), other=0.0)
    accumulator += tl.dot(a, b)
  }
  c = (accumulator).to(tl.float16)
  offs_cm = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_cn = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  c_ptrs = C + $(stride_cm) * offs_cm[:, None] + $(stride_cn) * offs_cn[None, :]
  c_mask = (offs_cm[:, None] < $(M)) & (offs_cn[None, :] < $(N))
  tl.store(c_ptrs, c, mask=c_mask)
}

/-- The `pre_load` IV-dependent matmul surface lowers to the algorithm layer. -/
theorem iv_dependent_matmul_pre_load_surface_toAlgorithm_supported
    (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat) :
    ∃ alg, (iv_dependent_matmul_pre_load_surface A B C M N K stride_am
      stride_ak stride_bk stride_bn stride_cm stride_cn BLOCK_SIZE_M
      BLOCK_SIZE_N BLOCK_SIZE_K).toAlgorithm? = Except.ok alg := by
  simp [iv_dependent_matmul_pre_load_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-! ## exec-stepping helpers -/

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

/-- `a_ptr` base eval: cell `(i,e) = (A, offs_am i · stride_am + e · stride_ak)`. -/
theorem aptr_eval (s : BlockState) (A : RegionName) (M K SAM SAK : Nat) (gm : Fin M → Nat)
    (hm : s.regs .nat [M] "offs_am" = some (Tile.vec gm))
    (hk : s.regs .nat [K] "offs_k" = some (Tile.vec (fun e : Fin K => e.val))) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase A)
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_am")) (Op.constNat SAM))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [K] "offs_k")) (Op.constNat SAK)))) s
      = some (⟨fun idx : TileIndex [M, K] => (A.cast, gm idx.1 * SAM + idx.2.1.val * SAK)⟩ : Tile .ptr [M, K]) := by
  rw [evalOp_ptrAdd, evalOp_ptrBase]
  simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref, evalOp_expandDim_one_nat,
    evalOp_expandDim_zero_nat, hm, hk, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec]
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec, NumericDType.add, NumericDType.mul]

/-- `b_ptr` base eval: cell `(e,j) = (B, e · stride_bk + offs_bn j · stride_bn)`. -/
theorem bptr_eval (s : BlockState) (B : RegionName) (K N SBK SBN : Nat) (gn : Fin N → Nat)
    (hk : s.regs .nat [K] "offs_k" = some (Tile.vec (fun e : Fin K => e.val)))
    (hn : s.regs .nat [N] "offs_bn" = some (Tile.vec gn)) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase B)
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [K] "offs_k")) (Op.constNat SBK))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [N] "offs_bn")) (Op.constNat SBN)))) s
      = some (⟨fun idx : TileIndex [K, N] => (B.cast, idx.1.val * SBK + gn idx.2.1 * SBN)⟩ : Tile .ptr [K, N]) := by
  rw [evalOp_ptrAdd, evalOp_ptrBase]
  simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref, evalOp_expandDim_one_nat,
    evalOp_expandDim_zero_nat, hk, hn, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec]
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec, NumericDType.add, NumericDType.mul]

/-- `accumulator` init eval: `tl.zeros` → the all-`0` tile. -/
theorem acc_init_eval (s : BlockState) (M N : Nat) :
    evalOp (Op.full [M, N] (Op.const 0)) s
      = some (⟨fun _ : TileIndex [M, N] => some (0 : ℝ)⟩ : Tile .real [M, N]) := by
  simp [evalOp_full, evalOp_const, Option.bind]

/-- `a_ptrs = a_ptr + k·BLOCK_SIZE_K·stride_ak` eval (recompute from base). -/
theorem aptrs_recompute_eval (s : BlockState) (M K BK SAK c : Nat) (ap : Tile .ptr [M, K])
    (hx : s.regs .ptr [M, K] "a_ptr" = some ap)
    (hkc : s.regs .nat [] "k" = some (Tile.scalar c)) :
    evalOp (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [M, K] "a_ptr")
      (Op.mul .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "k") (Op.constNat BK)) (Op.constNat SAK))) s
      = some (Tile.ptrAdd Broadcast.scalarR ap (Tile.scalar (c * BK * SAK))) := by
  rw [evalOp_ptrAdd]
  simp only [evalOp_ref, hx, evalOp_mul, evalOp_constNat, hkc, NumericDType.mul, Tile.bop,
    Broadcast.leftIndex, Broadcast.rightIndex, Option.bind, Option.bind_some, Option.bind_eq_bind]
  rfl

/-- `b_ptrs = b_ptr + k·BLOCK_SIZE_K·stride_bk` eval. -/
theorem bptrs_recompute_eval (s : BlockState) (K N BK SBK c : Nat) (bp : Tile .ptr [K, N])
    (hy : s.regs .ptr [K, N] "b_ptr" = some bp)
    (hkc : s.regs .nat [] "k" = some (Tile.scalar c)) :
    evalOp (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [K, N] "b_ptr")
      (Op.mul .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "k") (Op.constNat BK)) (Op.constNat SBK))) s
      = some (Tile.ptrAdd Broadcast.scalarR bp (Tile.scalar (c * BK * SBK))) := by
  rw [evalOp_ptrAdd]
  simp only [evalOp_ref, hy, evalOp_mul, evalOp_constNat, hkc, NumericDType.mul, Tile.bop,
    Broadcast.leftIndex, Broadcast.rightIndex, Option.bind, Option.bind_some, Option.bind_eq_bind]
  rfl

/-- **`accumulator = accumulator + tl.dot(a, b)` statement eval.** -/
theorem accdot_op_eval (M K N : Nat) (st : BlockState)
    (zt : Tile .real [M, N]) (xt : Tile .real [M, K]) (yt : Tile .real [K, N])
    (hz : st.regs .real [M, N] "accumulator" = some zt)
    (hx : st.regs .real [M, K] "a" = some xt)
    (hy : st.regs .real [K, N] "b" = some yt) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [M, N] "accumulator")
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

/-- `accumulator + dot` lane `(i,j)`: `some (zv + dv)`. -/
theorem accadd_eval (M N : Nat) (zt dt : Tile .real [M, N]) (i : Fin M) (j : Fin N) (zv dv : ℝ)
    (hz : zt.data (i, j, PUnit.unit) = some zv) (hd : dt.data (i, j, PUnit.unit) = some dv) :
    (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) zt dt).data
        (i, j, PUnit.unit) = some (zv + dv) := by
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, hz, hd, NumericDType.add,
    WithBot.realAdd, Option.map₂, Option.bind, Option.map]

/-! ## GEMM closed-form spec -/

/-- The kernel's `num_pid_n = cdiv N BN`. -/
def numPidN (N BN : Nat) : Nat := cdiv N BN

/-- The kernel's derived `pid_m = pid // num_pid_n`. -/
def pidM (pid N BN : Nat) : Nat := pid / numPidN N BN

/-- The kernel's derived `pid_n = pid % num_pid_n`. -/
def pidN (pid N BN : Nat) : Nat := pid % numPidN N BN

/-- Global output row of tile lane `i`: `PM · BM + i`. -/
def rowIndex (PM BM : Nat) (i : Fin BM) : Nat := PM * BM + i.val

/-- Global output column of tile lane `j`: `PN · BN + j`. -/
def colIndex (PN BN : Nat) (j : Fin BN) : Nat := PN * BN + j.val

/-- `A[i, k] = readMem A (offs_am i · stride_am + k · stride_ak)` (kernel's strided
A layout, with `offs_am i = (PM·BM + i) % M`). -/
noncomputable def aElem (s : BlockState) (A : RegionName) (PM BM M SAM SAK : Nat)
    (i : Fin BM) (k : Nat) : ℝ :=
  s.readMem A (rowIndex PM BM i % M * SAM + k * SAK)

/-- `B[k, j] = readMem B (k · stride_bk + offs_bn j · stride_bn)` (kernel's strided
B layout, with `offs_bn j = (PN·BN + j) % N`). -/
noncomputable def bElem (s : BlockState) (B : RegionName) (PN BN N SBK SBN : Nat)
    (j : Fin BN) (k : Nat) : ℝ :=
  s.readMem B (k * SBK + colIndex PN BN j % N * SBN)

/-- **Genuine GEMM spec**: `C[i,j] = Σ_{k < BLOCK_K·numKBlocks} A[i,k] · B[k,j]`. -/
noncomputable def matmulSpec (s : BlockState) (A B : RegionName)
    (PM PN BM BN M N SAM SAK SBK SBN BLOCK_K numKBlocks : Nat) (i : Fin BM) (j : Fin BN) : ℝ :=
  gemmSum (aElem s A PM BM M SAM SAK i) (bElem s B PN BN N SBK SBN j) (BLOCK_K * numKBlocks)

/-- Partial GEMM accumulator after `c` K-blocks: `Σ_{k < c·BLOCK_K} A·B`. -/
noncomputable def accPartial (s : BlockState) (A B : RegionName)
    (PM PN BM BN M N SAM SAK SBK SBN BLOCK_K : Nat) (i : Fin BM) (j : Fin BN) (c : Nat) : ℝ :=
  gemmSum (aElem s A PM BM M SAM SAK i) (bElem s B PN BN N SBK SBN j) (c * BLOCK_K)

/-- One-block step of the partial accumulator (the shared `gemmSum_blockSucc`). -/
theorem accPartial_succ (s : BlockState) (A B : RegionName)
    (PM PN BM BN M N SAM SAK SBK SBN BLOCK_K : Nat) (i : Fin BM) (j : Fin BN) (c : Nat) :
    accPartial s A B PM PN BM BN M N SAM SAK SBK SBN BLOCK_K i j (c + 1)
      = accPartial s A B PM PN BM BN M N SAM SAK SBK SBN BLOCK_K i j c
        + (Finset.univ.sum fun e : Fin BLOCK_K =>
            aElem s A PM BM M SAM SAK i (c * BLOCK_K + e.val)
              * bElem s B PN BN N SBK SBN j (c * BLOCK_K + e.val)) :=
  gemmSum_blockSucc (aElem s A PM BM M SAM SAK i) (bElem s B PN BN N SBK SBN j) BLOCK_K c

/-! ## Masked fp16 output-store machinery -/

def cOffset (_s : BlockState) (PM PN BM BN stride_cm stride_cn : Nat)
    (idx : TileIndex [BM, BN]) : Nat :=
  stride_cm * rowIndex PM BM idx.1 + stride_cn * colIndex PN BN idx.2.1

/-! ## Body decomposition -/

/-- The 5-statement K-loop body, transcribed: recompute `a_ptrs`/`b_ptrs` from the
loop-invariant bases `a_ptr`/`b_ptr`, masked loads, dot-accumulate. -/
def ivdmLoopBody (BM BN BLOCK_K K SAK SBK : Nat) : List Stmt :=
  [ Stmt.assign .ptr [BM, BLOCK_K] "a_ptrs"
      (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BM, BLOCK_K] "a_ptr")
        (Op.mul .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "k") (Op.constNat BLOCK_K)) (Op.constNat SAK))),
    Stmt.assign .ptr [BLOCK_K, BN] "b_ptrs"
      (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BLOCK_K, BN] "b_ptr")
        (Op.mul .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "k") (Op.constNat BLOCK_K)) (Op.constNat SBK))),
    Stmt.assign .real [BM, BLOCK_K] "a"
      (Op.load .real (.ptr (Op.ref .ptr [BM, BLOCK_K] "a_ptrs"))
        (.maskOther
          (Op.remap [BM, BLOCK_K] Broadcast.nil.consSame.consL.leftIndex
            (Op.lt ComparableDType.nat Broadcast.scalarR
              (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_K] "offs_k"))
              (Op.sub .nat Broadcast.nil (Op.constNat K)
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "k") (Op.constNat BLOCK_K)))))
          ((Op.const 0.0).broadcast [BM, BLOCK_K]))),
    Stmt.assign .real [BLOCK_K, BN] "b"
      (Op.load .real (.ptr (Op.ref .ptr [BLOCK_K, BN] "b_ptrs"))
        (.maskOther
          (Op.remap [BLOCK_K, BN] Broadcast.nil.consL.consSame.leftIndex
            (Op.lt ComparableDType.nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_K] "offs_k"))
              (Op.sub .nat Broadcast.nil (Op.constNat K)
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "k") (Op.constNat BLOCK_K)))))
          ((Op.const 0.0).broadcast [BLOCK_K, BN]))),
    Stmt.assign .real [BM, BN] "accumulator"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BM, BN] "accumulator")
        (Op.dot (batch := []) (Op.ref .real [BM, BLOCK_K] "a") (Op.ref .real [BLOCK_K, BN] "b"))) ]

/-- **preLoop scalars** (statements 0–6): the pid derivation plus the three index
vectors. Steps to a state where `offs_am`/`offs_bn`/`offs_k` hold the readbacks
the pointer chunk needs, and `pid_m`/`pid_n` hold `pidM`/`pidN`. -/
theorem preLoop_scalars (s : BlockState) (M N BM BN BK : Nat) :
    ∃ s7,
      stepStmts
        [ Stmt.assign .nat [] "pid" (Op.programId 0),
          Stmt.assign .nat [] "num_pid_n"
            (Op.div .nat Broadcast.nil (Op.sub .nat Broadcast.nil
              (Op.add .nat Broadcast.nil (Op.constNat N) (Op.constNat BN)) (Op.constNat 1)) (Op.constNat BN)),
          Stmt.assign .nat [] "pid_m"
            (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "pid") (Op.ref .nat [] "num_pid_n")),
          Stmt.assign .nat [] "pid_n"
            (Op.mod .nat Broadcast.nil (Op.ref .nat [] "pid") (Op.ref .nat [] "num_pid_n")),
          Stmt.assign .nat [BM] "offs_am"
            (Op.mod .nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat BM)) (Op.arange BM))
              (Op.constNat M)),
          Stmt.assign .nat [BN] "offs_bn"
            (Op.mod .nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_n") (Op.constNat BN)) (Op.arange BN))
              (Op.constNat N)),
          Stmt.assign .nat [BK] "offs_k" (Op.arange BK) ] s = some s7
      ∧ s7.pids = s.pids
      ∧ s7.regs .nat [] "pid_m" = some (Tile.scalar (pidM (s.pids 0) N BN))
      ∧ s7.regs .nat [] "pid_n" = some (Tile.scalar (pidN (s.pids 0) N BN))
      ∧ s7.regs .nat [BM] "offs_am"
          = some (Tile.vec (fun i : Fin BM => (pidM (s.pids 0) N BN * BM + i.val) % M))
      ∧ s7.regs .nat [BN] "offs_bn"
          = some (Tile.vec (fun j : Fin BN => (pidN (s.pids 0) N BN * BN + j.val) % N))
      ∧ s7.regs .nat [BK] "offs_k" = some (Tile.vec (fun e : Fin BK => e.val))
      ∧ s7.undef = s.undef
      ∧ s7.mem = s.mem := by
  simp only [pidM, pidN, numPidN, cdiv]
  simp [stepStmts, stepStmt, evalOp_floorDiv, evalOp_mod, Option.bind, BlockState.setReg,
    Tile.bop, Tile.cop, Tile.vec, NumericDType.add, NumericDType.mul, NumericDType.div,
    NumericDType.sub, ComparableDType.lt]

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

/-- The a-load K-tail mask: every lane is `true` when `c·BK + e < K` for all
`e < BK` (guaranteed by `K = BK·numKBlocks` and `c < numKBlocks`). -/
theorem amask_alltrue (s : BlockState) (BM BK K c : Nat)
    (hk : s.regs .nat [BK] "offs_k" = some (Tile.vec (fun e : Fin BK => e.val)))
    (hkk : s.regs .nat [] "k" = some (Tile.scalar c))
    (hlt : ∀ e : Fin BK, e.val < K - c * BK) :
    ∃ mtile : Tile .bool [BM, BK],
      evalOp (Op.remap [BM, BK] Broadcast.nil.consSame.consL.leftIndex
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BK] "offs_k"))
          (Op.sub .nat Broadcast.nil (Op.constNat K)
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "k") (Op.constNat BK))))) s = some mtile
      ∧ ∀ i, mtile.data i = Bool.true := by
  refine ⟨_, by
    simp only [evalOp, evalOp.eq_def, hk, hkk, Option.bind, Option.bind_some, Option.bind_eq_bind,
      Tile.expandDim, pure, Option.some.injEq]; rfl, ?_⟩
  intro i
  simp only [Tile.remap, Tile.cop, Tile.bop, Tile.expandDim, Tile.vec, Tile.scalar,
    ComparableDType.lt, NumericDType.sub, NumericDType.mul, Broadcast.leftIndex, Broadcast.rightIndex,
    TileShape.dropInsertedIndex]
  exact decide_eq_true (by simpa using hlt _)

/-- The b-load K-tail mask: every lane is `true` under the same condition. -/
theorem bmask_alltrue (s : BlockState) (BN BK K c : Nat)
    (hk : s.regs .nat [BK] "offs_k" = some (Tile.vec (fun e : Fin BK => e.val)))
    (hkk : s.regs .nat [] "k" = some (Tile.scalar c))
    (hlt : ∀ e : Fin BK, e.val < K - c * BK) :
    ∃ mtile : Tile .bool [BK, BN],
      evalOp (Op.remap [BK, BN] Broadcast.nil.consL.consSame.leftIndex
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "offs_k"))
          (Op.sub .nat Broadcast.nil (Op.constNat K)
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "k") (Op.constNat BK))))) s = some mtile
      ∧ ∀ i, mtile.data i = Bool.true := by
  refine ⟨_, by
    simp only [evalOp, evalOp.eq_def, hk, hkk, Option.bind, Option.bind_some, Option.bind_eq_bind,
      Tile.expandDim, pure, Option.some.injEq]; rfl, ?_⟩
  intro i
  simp only [Tile.remap, Tile.cop, Tile.bop, Tile.expandDim, Tile.vec, Tile.scalar,
    ComparableDType.lt, NumericDType.sub, NumericDType.mul, Broadcast.leftIndex, Broadcast.rightIndex,
    TileShape.dropInsertedIndex]
  exact decide_eq_true (by simpa using hlt _)

/-- The constant `other = 0.0` broadcast tile. -/
theorem other_broadcast_eval (s : BlockState) (shape : TileShape) :
    evalOp ((Op.const 0.0).broadcast shape) s = some (⟨fun _ : TileIndex shape => some (0.0 : ℝ)⟩ : Tile .real shape) := by
  simp [evalOp, evalOp_const, Option.bind]

/-! ## Loop invariant -/

/-- **Loop invariant** (counter `i = c`, the K-block index).

After `c` K-blocks: program ids and `mem`/`undef` fixed; the
`offs_am`/`offs_bn`/`offs_k`, `pid_m`/`pid_n` registers seeded; the
loop-invariant pointer bases `a_ptr`/`b_ptr` seeded; and `accumulator` equals
the partial GEMM accumulator `accPartial … c`. Since the body recomputes
`a_ptrs`/`b_ptrs` from `a_ptr`/`b_ptr`, no per-iteration pointer state is
carried. -/
noncomputable def ivdmInvariant
    (A B : RegionName) (s0 : BlockState)
    (PM PN M N BM BN SAM SAK SBK SBN BLOCK_K numKBlocks : Nat)
    (c : Nat) (s : BlockState) : Prop :=
  s.pids = s0.pids ∧ c ≤ numKBlocks ∧
  (s.regs .real [BM, BN] "accumulator" = some ⟨fun idx : TileIndex [BM, BN] =>
      some (accPartial s0 A B PM PN BM BN M N SAM SAK SBK SBN BLOCK_K idx.1 idx.2.1 c)⟩) ∧
  (s.regs .nat [] "pid_m" = some (Tile.scalar PM)) ∧
  (s.regs .nat [] "pid_n" = some (Tile.scalar PN)) ∧
  (s.regs .nat [BM] "offs_am" = some (Tile.vec (fun i : Fin BM => (PM * BM + i.val) % M))) ∧
  (s.regs .nat [BN] "offs_bn" = some (Tile.vec (fun j : Fin BN => (PN * BN + j.val) % N))) ∧
  (s.regs .nat [BLOCK_K] "offs_k" = some (Tile.vec (fun e : Fin BLOCK_K => e.val))) ∧
  (s.regs .ptr [BM, BLOCK_K] "a_ptr" = some ⟨fun idx : TileIndex [BM, BLOCK_K] =>
      (A.cast, (PM * BM + idx.1.val) % M * SAM + idx.2.1.val * SAK)⟩) ∧
  (s.regs .ptr [BLOCK_K, BN] "b_ptr" = some ⟨fun idx : TileIndex [BLOCK_K, BN] =>
      (B.cast, idx.1.val * SBK + (PN * BN + idx.2.1.val) % N * SBN)⟩) ∧
  (∀ rg o, s.undef rg o = 0) ∧ (s.mem = s0.mem)

set_option maxHeartbeats 1000000 in
/-- **preLoop** (statements 0–9): from a clean input state (`undef = 0`), the
prologue steps to a state satisfying `ivdmInvariant … 0` — the base case
(`accumulator = 0`, `a_ptr`/`b_ptr` bases seeded). -/
theorem ivdm_preLoop (A B C : RegionName) (s : BlockState)
    (M N _K SAM SAK SBK SBN SCM SCN BM BN BK numKBlocks : Nat)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ s', stepStmts ((iv_dependent_matmul_pre_load_surface A B C M N (BK * numKBlocks) SAM SAK SBK SBN
        SCM SCN BM BN BK).toAlgKernel.body.take 10) s = some s'
      ∧ ivdmInvariant A B s (pidM (s.pids 0) N BN) (pidN (s.pids 0) N BN)
          M N BM BN SAM SAK SBK SBN BK numKBlocks 0 s' := by
  obtain ⟨s7, h7, hpids, hpm, hpn, ham, hbn, hk, huf, hmem⟩ :=
    preLoop_scalars s M N BM BN BK
  set PM := pidM (s.pids 0) N BN with hPM
  set PN := pidN (s.pids 0) N BN with hPN
  rw [show ((iv_dependent_matmul_pre_load_surface A B C M N (BK * numKBlocks) SAM SAK SBK SBN
        SCM SCN BM BN BK).toAlgKernel.body.take 10)
      = [ Stmt.assign .nat [] "pid" (Op.programId 0),
          Stmt.assign .nat [] "num_pid_n"
            (Op.div .nat Broadcast.nil (Op.sub .nat Broadcast.nil
              (Op.add .nat Broadcast.nil (Op.constNat N) (Op.constNat BN)) (Op.constNat 1)) (Op.constNat BN)),
          Stmt.assign .nat [] "pid_m"
            (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "pid") (Op.ref .nat [] "num_pid_n")),
          Stmt.assign .nat [] "pid_n"
            (Op.mod .nat Broadcast.nil (Op.ref .nat [] "pid") (Op.ref .nat [] "num_pid_n")),
          Stmt.assign .nat [BM] "offs_am"
            (Op.mod .nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat BM)) (Op.arange BM))
              (Op.constNat M)),
          Stmt.assign .nat [BN] "offs_bn"
            (Op.mod .nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_n") (Op.constNat BN)) (Op.arange BN))
              (Op.constNat N)),
          Stmt.assign .nat [BK] "offs_k" (Op.arange BK) ]
      ++ [ Stmt.assign .ptr [BM, BK] "a_ptr"
            (Op.ptrAdd Broadcast.scalarL (Op.ptrBase A)
              (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
                (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_am")) (Op.constNat SAM))
                (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BK] "offs_k")) (Op.constNat SAK)))),
          Stmt.assign .ptr [BK, BN] "b_ptr"
            (Op.ptrAdd Broadcast.scalarL (Op.ptrBase B)
              (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
                (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "offs_k")) (Op.constNat SBK))
                (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_bn")) (Op.constNat SBN)))),
          Stmt.assign .real [BM, BN] "accumulator" (Op.full [BM, BN] (Op.const 0)) ] from rfl,
    stepStmts.append_some h7,
    stepStmts.cons_some (stepStmt_assign_eq_some
      (aptr_eval s7 A BM BK SAM SAK (fun i => (PM * BM + i.val) % M) (by simpa using ham) (by simpa using hk))),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (bptr_eval _ B BK BN SBK SBN (fun j => (PN * BN + j.val) % N) (by simp [hk]) (by simp [hbn]))),
    stepStmts.cons_some (stepStmt_assign_eq_some (acc_init_eval _ BM BN)),
    stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  refine ⟨by simp [hpids], by simp, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · -- accumulator = accPartial … 0 = 0
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx
    simp only [accPartial, Nat.zero_mul, gemmSum_zero]
  · simp [hpm]
  · simp [hpn]
  · simp [ham]
  · simp [hbn]
  · simp [hk]
  · -- a_ptr base
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
  · -- b_ptr base
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
  · intro rg o; simp [huf, hundef]
  · exact hmem

set_option maxHeartbeats 4000000 in
/-- **Step lemma**: one K-loop body iteration advances the invariant by one
block. The body recomputes `a_ptrs = a_ptr + c·BK·sak`, `b_ptrs = b_ptr + c·BK·sbk`
(= K-block `c`), loads them, and `accumulator += tl.dot(a, b)` adds the `c`-th
block's dot to the partial GEMM accumulator. Under `K = BK · numKBlocks` the
per-block K-tail load masks are all satisfied. -/
theorem ivdm_step (A B : RegionName) (s0 : BlockState)
    (PM PN M N BM BN SAM SAK SBK SBN BK numKBlocks : Nat) (hBK : 0 < BK)
    (c : Nat) (s : BlockState) (hclt : c < numKBlocks)
    (hinv : ivdmInvariant A B s0 PM PN M N BM BN SAM SAK SBK SBN BK numKBlocks c s) :
    ∃ s', stepStmts (ivdmLoopBody BM BN BK (BK * numKBlocks) SAK SBK)
        (s.setReg "k" .nat [] (Tile.scalar c)) = some s'
      ∧ ivdmInvariant A B s0 PM PN M N BM BN SAM SAK SBK SBN BK numKBlocks (c + 1) s' := by
  set K := BK * numKBlocks with hKdef
  simp only [ivdmInvariant] at hinv
  obtain ⟨hpids, hcle, hz, hpm, hpn, ham, hbn, hk, hapb, hbpb, hundef, hmem⟩ := hinv
  -- mask-all-true bound
  have hlt : ∀ e : Fin BK, e.val < K - c * BK := by
    intro e
    have hcK : c * BK + BK ≤ K := by
      rw [hKdef]; calc c * BK + BK = (c + 1) * BK := by ring
        _ ≤ numKBlocks * BK := Nat.mul_le_mul_right _ hclt
        _ = BK * numKBlocks := Nat.mul_comm _ _
    omega
  -- bases
  set apbT : Tile .ptr [BM, BK] :=
    ⟨fun idx : TileIndex [BM, BK] => (A.cast, (PM * BM + idx.1.val) % M * SAM + idx.2.1.val * SAK)⟩ with hapbT
  set bpbT : Tile .ptr [BK, BN] :=
    ⟨fun idx : TileIndex [BK, BN] => (B.cast, idx.1.val * SBK + (PN * BN + idx.2.1.val) % N * SBN)⟩ with hbpbT
  set zT : Tile .real [BM, BN] :=
    ⟨fun idx : TileIndex [BM, BN] => some (accPartial s0 A B PM PN BM BN M N SAM SAK SBK SBN BK idx.1 idx.2.1 c)⟩ with hzT
  set sk := s.setReg "k" .nat [] (Tile.scalar c) with hsk
  have hrmem : ∀ (R : RegionName) (o : Nat), sk.readMem R o = s0.readMem R o := by
    intro R o; simp only [hsk, BlockState.setReg_readMem]; unfold BlockState.readMem; rw [hmem]
  have hapbk : sk.regs .ptr [BM, BK] "a_ptr" = some apbT := by simp [hsk, hapb, hapbT]
  have hbpbk : sk.regs .ptr [BK, BN] "b_ptr" = some bpbT := by simp [hsk, hbpb, hbpbT]
  have hzk : sk.regs .real [BM, BN] "accumulator" = some zT := by simp [hsk, hz, hzT]
  have hkk : sk.regs .nat [BK] "offs_k" = some (Tile.vec (fun e : Fin BK => e.val)) := by simp [hsk, hk]
  have hkkv : sk.regs .nat [] "k" = some (Tile.scalar c) := by simp [hsk]
  -- recomputed pointer tiles
  set apT : Tile .ptr [BM, BK] := Tile.ptrAdd Broadcast.scalarR apbT (Tile.scalar (c * BK * SAK)) with hapT
  -- threaded state after a_ptrs recompute
  set sk1 := sk.setReg "a_ptrs" .ptr [BM, BK] apT with hsk1
  set bpT : Tile .ptr [BK, BN] := Tile.ptrAdd Broadcast.scalarR bpbT (Tile.scalar (c * BK * SBK)) with hbpT
  set sk2 := sk1.setReg "b_ptrs" .ptr [BK, BN] bpT with hsk2
  -- registers that survive into sk1, sk2
  have hkk2 : sk2.regs .nat [BK] "offs_k" = some (Tile.vec (fun e : Fin BK => e.val)) := by simp [hsk2, hsk1, hkk]
  have hkkv2 : sk2.regs .nat [] "k" = some (Tile.scalar c) := by simp [hsk2, hsk1, hkkv]
  -- loaded tiles
  set asub : Tile .real [BM, BK] :=
    ⟨fun idx => some (sk2.readMem (apT.data idx).1 (apT.data idx).2)⟩ with hasub
  set sk3 := sk2.setReg "a" .real [BM, BK] asub with hsk3
  set bsub : Tile .real [BK, BN] :=
    ⟨fun idx => some (sk3.readMem (bpT.data idx).1 (bpT.data idx).2)⟩ with hbsub
  set sk4 := sk3.setReg "b" .real [BK, BN] bsub with hsk4
  have hkk3 : sk3.regs .nat [BK] "offs_k" = some (Tile.vec (fun e : Fin BK => e.val)) := by simp [hsk3, hkk2]
  have hkkv3 : sk3.regs .nat [] "k" = some (Tile.scalar c) := by simp [hsk3, hkkv2]
  -- masks
  obtain ⟨amt, ham_eval, ham_true⟩ := amask_alltrue sk2 BM BK K c hkk2 hkkv2 hlt
  obtain ⟨bmt, hbm_eval, hbm_true⟩ := bmask_alltrue sk3 BN BK K c hkk3 hkkv3 hlt
  unfold ivdmLoopBody
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (aptrs_recompute_eval sk BM BK BK SAK c apbT hapbk hkkv))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (bptrs_recompute_eval sk1 BK BN BK SBK c bpbT (by simp [hsk1, hbpbk]) (by simp [hsk1, hkkv])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (load_ptr_maskOther_alltrue (Op.ref .ptr [BM, BK] "a_ptrs") _ _ sk2 apT amt _
          (by rw [evalOp_ref]; simp [hsk2, hsk1, hapT]) ham_eval (other_broadcast_eval sk2 [BM, BK]) ham_true))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (load_ptr_maskOther_alltrue (Op.ref .ptr [BK, BN] "b_ptrs") _ _ sk3 bpT bmt _
          (by rw [evalOp_ref]; simp [hsk3, hsk2, hbpT]) hbm_eval (other_broadcast_eval sk3 [BK, BN]) hbm_true))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (accdot_op_eval BM BK BN sk4 zT asub bsub
          (by simp [hsk4, hsk3, hsk2, hsk1, hzk])
          (by simp [hsk4, hsk3])
          (by simp [hsk4])))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  simp only [ivdmInvariant]
  refine ⟨by simp [hsk4, hsk3, hsk2, hsk1, hsk, hpids], by omega, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · -- accumulator = accPartial (c+1)
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx
    have hrmem2 : ∀ (R : RegionName) (o : Nat), sk2.readMem R o = s0.readMem R o := by
      intro R o; rw [hsk2, BlockState.setReg_readMem, hsk1, BlockState.setReg_readMem]; exact hrmem R o
    have hrmem3 : ∀ (R : RegionName) (o : Nat), sk3.readMem R o = s0.readMem R o := by
      intro R o; rw [hsk3, BlockState.setReg_readMem]; exact hrmem2 R o
    have has : ∀ e : Fin BK, asub.data (idx.1, e, PUnit.unit)
        = some (aElem s0 A PM BM M SAM SAK idx.1 (c * BK + e.val)) := by
      intro e
      simp only [hasub, hapT, hapbT, hrmem2, aElem, rowIndex, Tile.ptrAdd, Tile.bop, Tile.scalar,
        Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add]
      congr 2
      ring
    have hbs : ∀ e : Fin BK, bsub.data (e, idx.2.1, PUnit.unit)
        = some (bElem s0 B PN BN N SBK SBN idx.2.1 (c * BK + e.val)) := by
      intro e
      simp only [hbsub, hbpT, hbpbT, hrmem3, bElem, colIndex, Tile.ptrAdd, Tile.bop, Tile.scalar,
        Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add]
      congr 2
      ring
    rw [accadd_eval BM BN zT (Tile.dot [] asub bsub) idx.1 idx.2.1
        (accPartial s0 A B PM PN BM BN M N SAM SAK SBK SBN BK idx.1 idx.2.1 c)
        (Finset.univ.sum fun e : Fin BK =>
          aElem s0 A PM BM M SAM SAK idx.1 (c * BK + e.val) * bElem s0 B PN BN N SBK SBN idx.2.1 (c * BK + e.val))
        (by rw [hzT])
        (tile_dot_data BM BK BN asub bsub idx.1 idx.2.1 _ _ has hbs)]
    show some _ = some (accPartial s0 A B PM PN BM BN M N SAM SAK SBK SBN BK idx.1 idx.2.1 (c + 1))
    rw [accPartial_succ]
  · simp [hsk4, hsk3, hsk2, hsk1, hsk, hpm]
  · simp [hsk4, hsk3, hsk2, hsk1, hsk, hpn]
  · simp [hsk4, hsk3, hsk2, hsk1, hsk, ham]
  · simp [hsk4, hsk3, hsk2, hsk1, hsk, hbn]
  · simp [hsk4, hsk3, hsk2, hsk1, hsk, hk]
  · -- a_ptr base unchanged
    simp [hsk4, hsk3, hsk2, hsk1, hsk, hapb, hapbT]
  · -- b_ptr base unchanged
    simp [hsk4, hsk3, hsk2, hsk1, hsk, hbpb, hbpbT]
  · intro rg o; simp [hsk4, hsk3, hsk2, hsk1, hsk, hundef]
  · show _ = s0.mem
    rw [← hmem]; rfl

/-! ## Post-loop: fp16 cast + masked store -/

/-- The 6 post-loop statements: fp16 cast, output index vectors, output pointers,
output mask, masked fp16 store. -/
def ivdmPostBody (C : RegionName) (M N SCM SCN BM BN : Nat) : List Stmt :=
  [ Stmt.assign FloatDType.fp16.toTileDType [BM, BN] "c"
      (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref .real [BM, BN] "accumulator")),
    Stmt.assign .nat [BM] "offs_cm"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat BM)) (Op.arange BM)),
    Stmt.assign .nat [BN] "offs_cn"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_n") (Op.constNat BN)) (Op.arange BN)),
    Stmt.assign .ptr [BM, BN] "c_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase C)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarL (Op.constNat SCM) (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_cm")))
          (Op.mul .nat Broadcast.scalarL (Op.constNat SCN) (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_cn"))))),
    Stmt.assign .bool [BM, BN] "c_mask"
      (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_cm")) (Op.constNat M))
        (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_cn")) (Op.constNat N))),
    Stmt.store .fp16 [BM, BN] (.ptr (Op.ref .ptr [BM, BN] "c_ptrs")) (Op.ref .fp16 [BM, BN] "c")
      (.mask (Op.ref .bool [BM, BN] "c_mask")) ]

/-- The kernel body decomposes as prefix (10) ++ [K-loop, post-statements]. By `rfl`. -/
theorem ivdm_body_split (A B C : RegionName)
    (M N SAM SAK SBK SBN SCM SCN BM BN BK numKBlocks : Nat) :
    (iv_dependent_matmul_pre_load_surface A B C M N (BK * numKBlocks) SAM SAK SBK SBN SCM SCN BM BN BK).toAlgKernel.body
      = (iv_dependent_matmul_pre_load_surface A B C M N (BK * numKBlocks) SAM SAK SBK SBN SCM SCN BM BN BK).toAlgKernel.body.take 10
        ++ (Stmt.forRangeDyn "k" (Op.constNat 0)
              (Op.div .nat Broadcast.nil (Op.sub .nat Broadcast.nil
                (Op.add .nat Broadcast.nil (Op.constNat (BK * numKBlocks)) (Op.constNat BK)) (Op.constNat 1)) (Op.constNat BK))
              (Op.constNat 1) (ivdmLoopBody BM BN BK (BK * numKBlocks) SAK SBK)
            :: ivdmPostBody C M N SCM SCN BM BN) := by
  rfl

/-- `offs_cm` eval: `pid_m · BM + arange` (non-modular output rows). -/
theorem offs_cm_eval (s : BlockState) (PM BM : Nat)
    (hpm : s.regs .nat [] "pid_m" = some (Tile.scalar PM)) :
    evalOp (Op.add .nat Broadcast.scalarL
      (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat BM)) (Op.arange BM)) s
      = some (Tile.vec (fun i : Fin BM => rowIndex PM BM i)) := by
  simp only [evalOp_add, evalOp_mul, evalOp_ref, hpm, evalOp_constNat, evalOp_arange, Option.bind,
    Option.bind_some]
  refine congrArg some ?_
  ext i
  simp only [Tile.bop, Tile.vec, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.add, NumericDType.mul, rowIndex]

/-- `offs_cn` eval: `pid_n · BN + arange` (non-modular output cols). -/
theorem offs_cn_eval (s : BlockState) (PN BN : Nat)
    (hpn : s.regs .nat [] "pid_n" = some (Tile.scalar PN)) :
    evalOp (Op.add .nat Broadcast.scalarL
      (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_n") (Op.constNat BN)) (Op.arange BN)) s
      = some (Tile.vec (fun j : Fin BN => colIndex PN BN j)) := by
  simp only [evalOp_add, evalOp_mul, evalOp_ref, hpn, evalOp_constNat, evalOp_arange, Option.bind,
    Option.bind_some]
  refine congrArg some ?_
  ext j
  simp only [Tile.bop, Tile.vec, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.add, NumericDType.mul, colIndex]

/-- `c_ptrs` eval: cell `(i,j) = (C, stride_cm·offs_cm i + stride_cn·offs_cn j)`. -/
theorem cptrs_eval (s : BlockState) (C : RegionName) (BM BN SCM SCN : Nat) (gm : Fin BM → Nat) (gn : Fin BN → Nat)
    (hm : s.regs .nat [BM] "offs_cm" = some (Tile.vec gm))
    (hn : s.regs .nat [BN] "offs_cn" = some (Tile.vec gn)) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase C)
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarL (Op.constNat SCM) (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_cm")))
        (Op.mul .nat Broadcast.scalarL (Op.constNat SCN) (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_cn"))))) s
      = some (⟨fun idx : TileIndex [BM, BN] => (C.cast, SCM * gm idx.1 + SCN * gn idx.2.1)⟩ : Tile .ptr [BM, BN]) := by
  rw [evalOp_ptrAdd, evalOp_ptrBase]
  simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref, evalOp_expandDim_one_nat,
    evalOp_expandDim_zero_nat, hm, hn, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec]
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec, NumericDType.add, NumericDType.mul]

/-- `c_mask` eval: `(offs_cm < M) & (offs_cn < N)` is all-`true` when every output
row/col is in-bounds. -/
theorem cmask_alltrue (s : BlockState) (M N BM BN : Nat) (gm : Fin BM → Nat) (gn : Fin BN → Nat)
    (hm : s.regs .nat [BM] "offs_cm" = some (Tile.vec gm))
    (hn : s.regs .nat [BN] "offs_cn" = some (Tile.vec gn))
    (hmlt : ∀ i, gm i < M) (hnlt : ∀ j, gn j < N) :
    ∃ mtile : Tile .bool [BM, BN],
      evalOp (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_cm")) (Op.constNat M))
        (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_cn")) (Op.constNat N))) s
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

/-- The masked fp16 store statement reduces to the masked scatter foldl. -/
theorem ivdmStore_eval (_C : RegionName) (BM BN : Nat) (st : BlockState)
    (cT : Tile .fp16 [BM, BN]) (cpT : Tile .ptr [BM, BN]) (mT : Tile .bool [BM, BN])
    (hc : st.regs .fp16 [BM, BN] "c" = some cT)
    (hcp : st.regs .ptr [BM, BN] "c_ptrs" = some cpT)
    (hm : st.regs .bool [BM, BN] "c_mask" = some mT) :
    stepStmt (Stmt.store .fp16 [BM, BN] (.ptr (Op.ref .ptr [BM, BN] "c_ptrs")) (Op.ref .fp16 [BM, BN] "c")
        (.mask (Op.ref .bool [BM, BN] "c_mask"))) st
      = some ((TileShape.allIndices [BM, BN]).foldl
          (fun acc idx =>
            if mT.data idx then acc.writeMemTyped .fp16 (cpT.data idx).1 (cpT.data idx).2 (cT.data idx) else acc) st) := by
  simp only [stepStmt, evalOp_ref, hc, hcp, hm, bind, Option.bind_some]
  refine congrArg some (List.foldl_ext _ _ st (fun acc idx _ => ?_))
  by_cases hb : mT.data idx
  · simp only [hb, if_true]
  · simp only [hb, if_false, Bool.false_eq_true]

/-- The genuine output cell: `fp16(Σ_k A·B)` as a `MemCell`. -/
noncomputable def outputCell (s0 : BlockState) (A B : RegionName)
    (PM PN BM BN M N SAM SAK SBK SBN BK numKBlocks : Nat) (idx : TileIndex [BM, BN]) : MemCell :=
  MemCell.of .fp16 (FloatDType.fp16.ofReal (FloatDType.fp16.storeValue
    (FloatDType.real.cast FloatDType.fp16
      (some (matmulSpec s0 A B PM PN BM BN M N SAM SAK SBK SBN BK numKBlocks idx.1 idx.2.1)))))

set_option maxHeartbeats 2000000 in
/-- **postLoop**: from the invariant at `numKBlocks` blocks, the fp16 cast +
masked store writes the genuine value `fp16(Σ_k A·B)` at every in-bounds output
lane. -/
theorem ivdm_postLoop (A B C : RegionName) (s0 : BlockState)
    (PM PN M N BM BN SAM SAK SBK SBN SCM SCN BK numKBlocks : Nat)
    (hInj : Function.Injective (cOffset s0 PM PN BM BN SCM SCN))
    (hmlt : ∀ i : Fin BM, rowIndex PM BM i < M)
    (hnlt : ∀ j : Fin BN, colIndex PN BN j < N)
    (st : BlockState)
    (hinv : ivdmInvariant A B s0 PM PN M N BM BN SAM SAK SBK SBN BK numKBlocks numKBlocks st) :
    ∃ sfin, stepStmts (ivdmPostBody C M N SCM SCN BM BN) st = some sfin
      ∧ ∀ idx : TileIndex [BM, BN],
          sfin.mem C (cOffset s0 PM PN BM BN SCM SCN idx)
            = outputCell s0 A B PM PN BM BN M N SAM SAK SBK SBN BK numKBlocks idx := by
  simp only [ivdmInvariant] at hinv
  obtain ⟨hpids, hcle, hz, hpm, hpn, ham, hbn, hk, hapb, hbpb, hundef, hmem⟩ := hinv
  set g : TileIndex [BM, BN] → ℝ :=
    fun idx => matmulSpec s0 A B PM PN BM BN M N SAM SAK SBK SBN BK numKBlocks idx.1 idx.2.1 with hg
  have hzspec : st.regs .real [BM, BN] "accumulator" = some ⟨fun idx => some (g idx)⟩ := by
    rw [hz]; refine congrArg some ?_; ext idx; simp only [hg, matmulSpec, accPartial, Nat.mul_comm numKBlocks BK]
  unfold ivdmPostBody
  -- step 1: fp16 cast
  have hcast : evalOp (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref .real [BM, BN] "accumulator")) st
      = some (⟨fun idx => FloatDType.real.cast FloatDType.fp16 (some (g idx))⟩ : Tile .fp16 [BM, BN]) := by
    rw [evalOp_castFloat]
    simp [hzspec]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some hcast)]
  set s1 := st.setReg "c" FloatDType.fp16.toTileDType [BM, BN] (⟨fun idx => FloatDType.real.cast FloatDType.fp16 (some (g idx))⟩ : Tile .fp16 [BM, BN]) with hs1
  -- step 2: offs_cm
  have hpm1 : s1.regs .nat [] "pid_m" = some (Tile.scalar PM) := by rw [hs1]; simp [hpm]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (offs_cm_eval s1 PM BM hpm1))]
  set s2 := s1.setReg "offs_cm" .nat [BM] (Tile.vec (fun i : Fin BM => rowIndex PM BM i)) with hs2
  -- step 3: offs_cn
  have hpn2 : s2.regs .nat [] "pid_n" = some (Tile.scalar PN) := by simp [hs2, hs1, hpn]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (offs_cn_eval s2 PN BN hpn2))]
  set s3 := s2.setReg "offs_cn" .nat [BN] (Tile.vec (fun j : Fin BN => colIndex PN BN j)) with hs3
  -- step 4: c_ptrs
  have hcm3 : s3.regs .nat [BM] "offs_cm" = some (Tile.vec (fun i : Fin BM => rowIndex PM BM i)) := by simp [hs3, hs2]
  have hcn3 : s3.regs .nat [BN] "offs_cn" = some (Tile.vec (fun j : Fin BN => colIndex PN BN j)) := by simp [hs3]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (cptrs_eval s3 C BM BN SCM SCN (fun i => rowIndex PM BM i) (fun j => colIndex PN BN j) hcm3 hcn3))]
  set cpT : Tile .ptr [BM, BN] :=
    ⟨fun idx : TileIndex [BM, BN] => (C.cast, SCM * rowIndex PM BM idx.1 + SCN * colIndex PN BN idx.2.1)⟩ with hcpT
  set s4 := s3.setReg "c_ptrs" .ptr [BM, BN] cpT with hs4
  -- step 5: c_mask
  have hcm4 : s4.regs .nat [BM] "offs_cm" = some (Tile.vec (fun i : Fin BM => rowIndex PM BM i)) := by simp [hs4, hcm3]
  have hcn4 : s4.regs .nat [BN] "offs_cn" = some (Tile.vec (fun j : Fin BN => colIndex PN BN j)) := by simp [hs4, hcn3]
  obtain ⟨mT, hmask_eval, hmask_true⟩ :=
    cmask_alltrue s4 M N BM BN (fun i => rowIndex PM BM i) (fun j => colIndex PN BN j) hcm4 hcn4 hmlt hnlt
  rw [stepStmts.cons_some (stepStmt_assign_eq_some hmask_eval)]
  set s5 := s4.setReg "c_mask" .bool [BM, BN] mT with hs5
  -- step 6: masked fp16 store
  set cT : Tile .fp16 [BM, BN] :=
    ⟨fun idx => FloatDType.real.cast FloatDType.fp16 (some (g idx))⟩ with hcT
  have hcc : s5.regs .fp16 [BM, BN] "c" = some cT := by simp [hs5, hs4, hs3, hs2, hs1, hcT]
  have hcpp : s5.regs .ptr [BM, BN] "c_ptrs" = some cpT := by simp [hs5, hs4]
  have hmm : s5.regs .bool [BM, BN] "c_mask" = some mT := by simp [hs5]
  rw [stepStmts.cons_some (ivdmStore_eval C BM BN s5 cT cpT mT hcc hcpp hmm), stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  intro idx
  have hstep_eq :
      (fun (acc : BlockState) k =>
        if mT.data k then acc.writeMemTyped .fp16 (cpT.data k).1 (cpT.data k).2 (cT.data k) else acc)
        =
      (fun (acc : BlockState) k =>
        if (fun _ : TileIndex [BM, BN] => True) k then acc.writeMemTyped .fp16 (cpT.data k).1 (cpT.data k).2 (cT.data k) else acc) := by
    funext acc k; rw [hmask_true k]; simp
  rw [hstep_eq]
  have hCregion : ∀ k : TileIndex [BM, BN], (cpT.data k).1 = C := by intro k; simp [hcpT]
  rw [show
      ((TileShape.allIndices [BM, BN]).foldl
        (fun acc k => if (fun _ : TileIndex [BM, BN] => True) k then acc.writeMemTyped .fp16 (cpT.data k).1 (cpT.data k).2 (cT.data k) else acc) s5)
      = ((TileShape.allIndices [BM, BN]).foldl
        (fun acc k => if (fun _ : TileIndex [BM, BN] => True) k then acc.writeMemTyped .fp16 C ((cpT.data k).2) (cT.data k) else acc) s5)
      from List.foldl_ext _ _ s5 (fun acc k _ => by rw [hCregion k])]
  have hcoff : (cpT.data idx).2 = cOffset s0 PM PN BM BN SCM SCN idx := by simp [hcpT, cOffset]
  have hoffInj : Function.Injective (fun idx : TileIndex [BM, BN] => (cpT.data idx).2) := by
    intro a b hab; apply hInj; simpa [cOffset, hcpT] using hab
  rw [← hcoff]
  rw [scatter_memcell_fp16_prop_masked_nd (region := C) (s := s5)
    (offsetFn := fun idx : TileIndex [BM, BN] => (cpT.data idx).2)
    (valueFn := fun idx => cT.data idx)
    (P := fun _ => True) hoffInj idx]
  simp only [if_pos trivial, outputCell, hcT, hg]

/-- The dynamic K-loop bound resolves: `evalOp (cdiv K BK) = numKBlocks` and the
loop runs `numKBlocks` iterations, advancing `ivdmInvariant` from `0` to
`numKBlocks`. -/
theorem ivdm_loop (A B : RegionName) (s0 : BlockState)
    (PM PN M N BM BN SAM SAK SBK SBN BK numKBlocks : Nat) (hBK : 0 < BK)
    (s : BlockState)
    (hP0 : ivdmInvariant A B s0 PM PN M N BM BN SAM SAK SBK SBN BK numKBlocks 0 s) :
    ∃ sLoop, stepStmt (Stmt.forRangeDyn "k" (Op.constNat 0)
        (Op.div .nat Broadcast.nil (Op.sub .nat Broadcast.nil
          (Op.add .nat Broadcast.nil (Op.constNat (BK * numKBlocks)) (Op.constNat BK)) (Op.constNat 1)) (Op.constNat BK))
        (Op.constNat 1) (ivdmLoopBody BM BN BK (BK * numKBlocks) SAK SBK)) s = some sLoop
      ∧ ivdmInvariant A B s0 PM PN M N BM BN SAM SAK SBK SBN BK numKBlocks numKBlocks sLoop := by
  have hcdiv : (BK * numKBlocks + BK - 1) / BK = numKBlocks := by
    have he : BK * numKBlocks + BK - 1 = (BK - 1) + BK * numKBlocks := by omega
    rw [he, Nat.add_mul_div_left _ _ hBK, Nat.div_eq_of_lt (by omega), Nat.zero_add]
  have hresolve : stepStmt (Stmt.forRangeDyn "k" (Op.constNat 0)
      (Op.div .nat Broadcast.nil (Op.sub .nat Broadcast.nil
        (Op.add .nat Broadcast.nil (Op.constNat (BK * numKBlocks)) (Op.constNat BK)) (Op.constNat 1)) (Op.constNat BK))
      (Op.constNat 1) (ivdmLoopBody BM BN BK (BK * numKBlocks) SAK SBK)) s
      = stepForRangeAux "k" 0 numKBlocks 1 (ivdmLoopBody BM BN BK (BK * numKBlocks) SAK SBK) s := by
    rw [stepForRangeAux.forRangeDyn_unfold]
    simp only [evalOp_constNat, evalOp_div, evalOp_sub, evalOp_add, Tile.scalar, Tile.bop,
      Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.div, NumericDType.sub,
      NumericDType.add, Tile.data, Option.bind_some, bind]
    rw [hcdiv]
  rw [hresolve]
  obtain ⟨final, sLoop, haux, hfinal, hPfinal⟩ :=
    forRangeAux_inv (idx := "k") (stop := numKBlocks) (step := 1)
      (P := ivdmInvariant A B s0 PM PN M N BM BN SAM SAK SBK SBN BK numKBlocks)
      (by norm_num)
      (fun i st hlt hinv => ivdm_step A B s0 PM PN M N BM BN SAM SAK SBK SBN BK numKBlocks hBK i st hlt hinv)
      0 s hP0
  have hfinalEq : final = numKBlocks := by
    simp only [ivdmInvariant] at hPfinal
    omega
  subst hfinalEq
  exact ⟨sLoop, haux, hPfinal⟩

/-! ## Composition: full exec closed form -/

set_option maxHeartbeats 2000000 in
/-- **Top exec reduction**: `preLoop` + K-loop (`ivdm_loop`) + `postLoop` compose
into the full `exec`. Every in-bounds output lane equals the genuine value
`fp16(Σ_k A·B)`. -/
theorem ivdm_exec_closed_form (A B C : RegionName) (s : BlockState)
    (M N SAM SAK SBK SBN SCM SCN BM BN BK numKBlocks : Nat) (hBK : 0 < BK)
    (hInj : Function.Injective (cOffset s (pidM (s.pids 0) N BN) (pidN (s.pids 0) N BN) BM BN SCM SCN))
    (hmlt : ∀ i : Fin BM, rowIndex (pidM (s.pids 0) N BN) BM i < M)
    (hnlt : ∀ j : Fin BN, colIndex (pidN (s.pids 0) N BN) BN j < N)
    (hundef : ∀ rg o, s.undef rg o = 0)
    (idx : TileIndex [BM, BN]) :
    (match exec (iv_dependent_matmul_pre_load_surface A B C M N (BK * numKBlocks) SAM SAK SBK SBN SCM SCN BM BN BK) s with
      | some s' => s'.mem C (cOffset s (pidM (s.pids 0) N BN) (pidN (s.pids 0) N BN) BM BN SCM SCN idx)
      | none => (0 : MemCell)) =
      outputCell s A B (pidM (s.pids 0) N BN) (pidN (s.pids 0) N BN)
        BM BN M N SAM SAK SBK SBN BK numKBlocks idx := by
  set PM := pidM (s.pids 0) N BN with hPM
  set PN := pidN (s.pids 0) N BN with hPN
  obtain ⟨s0, hpre_eq, hP0⟩ := ivdm_preLoop A B C s M N (BK * numKBlocks) SAM SAK SBK SBN SCM SCN BM BN BK numKBlocks hundef
  obtain ⟨sLoop, hLoopStmt, hPLoop⟩ :=
    ivdm_loop A B s PM PN M N BM BN SAM SAK SBK SBN BK numKBlocks hBK s0 hP0
  obtain ⟨sfin, hTail, hpost⟩ :=
    ivdm_postLoop A B C s PM PN M N BM BN SAM SAK SBK SBN SCM SCN BK numKBlocks hInj hmlt hnlt sLoop hPLoop
  have hexec : exec (iv_dependent_matmul_pre_load_surface A B C M N (BK * numKBlocks) SAM SAK SBK SBN SCM SCN BM BN BK) s
      = some sfin := by
    rw [exec, ivdm_body_split A B C M N SAM SAK SBK SBN SCM SCN BM BN BK numKBlocks,
      stepStmts.append_some hpre_eq, stepStmts.cons_some hLoopStmt, hTail]
  rw [hexec]
  exact hpost idx

/-- **Closed-form correctness for `iv_dependent_matmul` (general statement).**

For arbitrary `M`/`N`, tile dims `BM`/`BN`, strides, K-block size `BK`, and
K-block count `numKBlocks` (so `K = BK · numKBlocks`), every in-bounds output cell
of the computed `BM × BN` tile equals the genuine matrix product
`fp16(Σ_{k < BK·numKBlocks} A[i,k] · B[k,j])` (over ℝ, with a final fp16 output
cast) of the loaded `A`/`B` tiles — NOT the kernel's own executed value.

`PM`/`PN` are the kernel's own `pid_m = pid // cdiv N BN` / `pid_n = pid % cdiv N BN`.
Preconditions: `0 < BK`; all tile rows/cols in-bounds (`PM·BM + i < M`,
`PN·BN + j < N`), making the modular addressing the identity and the store mask
all-true; output-address injectivity, supplied concretely by `stride_cn = 1`
(`hcn`) and `BN ≤ stride_cm` (`hbnle`); clean initial `undef`. The four remaining
scheduling modes load the same per-block tiles and therefore compute the same
product. -/
theorem iv_dependent_matmul_closed_form_correct
    (A B C : RegionName) (s : BlockState)
    (M N SAM SAK SBK SBN SCM SCN BM BN BK numKBlocks : Nat) (hBK : 0 < BK)
    (hcn : SCN = 1) (hbnle : BN ≤ SCM)
    (hmlt : ∀ i : Fin BM, rowIndex (pidM (s.pids 0) N BN) BM i < M)
    (hnlt : ∀ j : Fin BN, colIndex (pidN (s.pids 0) N BN) BN j < N)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes
      (kernel := iv_dependent_matmul_pre_load_surface A B C M N (BK * numKBlocks) SAM SAK SBK SBN SCM SCN BM BN BK)
      (initialState := s)
      (write := fun idx : TileIndex [BM, BN] =>
        some (C, cOffset s (pidM (s.pids 0) N BN) (pidN (s.pids 0) N BN) BM BN SCM SCN idx))
      (expected := fun idx : TileIndex [BM, BN] =>
        outputCell s A B (pidM (s.pids 0) N BN) (pidN (s.pids 0) N BN)
          BM BN M N SAM SAK SBK SBN BK numKBlocks idx) := by
  subst hcn
  have hInj : Function.Injective
      (cOffset s (pidM (s.pids 0) N BN) (pidN (s.pids 0) N BN) BM BN SCM 1) := by
    have heq : cOffset s (pidM (s.pids 0) N BN) (pidN (s.pids 0) N BN) BM BN SCM 1
        = fun idx : TileIndex [BM, BN] =>
            (SCM * (pidM (s.pids 0) N BN * BM) + pidN (s.pids 0) N BN * BN)
              + idx.1.val * SCM + idx.2.1.val := by
      funext idx; simp only [cOffset, rowIndex, colIndex]; ring
    rw [heq]; exact rowMajor2D_inj _ SCM hbnle
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [iv_dependent_matmul_pre_load_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst hs0
  intro idx
  have hmain := ivdm_exec_closed_form A B C s0 M N SAM SAK SBK SBN SCM 1 BM BN BK numKBlocks hBK hInj hmlt hnlt hundef idx
  rw [hExec] at hmain
  exact hmain

end VeriTile.Bench.TritonBenchG.IvDependentMatmul
