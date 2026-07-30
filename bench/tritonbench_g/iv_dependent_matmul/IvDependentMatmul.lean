import VeriTile.Triton

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
iv_dependent_matmul_closed_form_correct      ← TOP THEOREM (ComputeCorrect.Realizes_without_Rounding)
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

## Translation-surface blocker

Translation-surface blocker: only the canonical `type == "pre_load"`
scheduling mode is mechanized as a surface; the string constexpr `type` and
the four other mode branches (`post_load`, `post_pre_mixed`,
`post_load_two_iters`, `post_load_three_iters`) are described informally
above and are not transcribed, so the `type` parameter and those `if`
branches do not appear in the Lean surface. The textual py↔lean scans in
`bench/audit_tritonbench_g.sh` exempt this port on this marker (registered in
`proof_blockers.md`).
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
specification iv_dependent_matmul_closed_form_correct
    (A B C : RegionName) (s : BlockState)
    (M N SAM SAK SBK SBN SCM SCN BM BN BK numKBlocks : Nat) (hBK : 0 < BK)
    (hcn : SCN = 1) (hbnle : BN ≤ SCM)
    (hmlt : ∀ i : Fin BM, rowIndex (pidM (s.pids 0) N BN) BM i < M)
    (hnlt : ∀ j : Fin BN, colIndex (pidN (s.pids 0) N BN) BN j < N)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes_without_Rounding
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

/-! ## The `⊨[R]` streaming headline (wave-5 S1 fold genre)

Everything below is purely additive; the exact surface above is untouched.
Structure of the `execR R` story: the kernel's only two rounding events live
in `ivdmPostBody` — the `(accumulator).to(tl.float16)` (`evalOpR` site 1) and
the `.fp16`-typed masked `tl.store` (`writeMemAsR` site 2). **Verified
against the lowered body**: the `.to(tl.float16)` genuinely survives
lowering as `Stmt.assign FloatDType.fp16.toTileDType … (Op.castFloat …)`
feeding a `Stmt.store .fp16 …`, so `outDType := .fp16` is the honest grid
(a `.real` declaration would be false here).

The 10-statement prologue and the whole dynamic K-loop (the two IV-dependent
pointer recomputes plus masked loads with `other=0.0` carry no `castFloat`)
are cast-free, so under `execR R` they collapse verbatim onto the exact
stepper and the proven `ivdm_preLoop` / `ivdm_step` / `forRangeAux_inv`
stack above is reused unchanged; only the 6-statement store tail is
re-proved on the `R` side (`ivdm_postLoopR`). `round_idem` (via
`RoundingModel.storeValue_cast`) collapses the tail's cast+store double
round into the single boundary `R.round .fp16` the skin's readback contract
states. -/

open scoped VeriTile.Triton.StreamMasked2DKernelIO₂

/-! ### Body decomposition names and cast-free collapses -/

/-- The 10-statement prologue (statements 0–9) as an explicit list: the four
pid scalars, the three index vectors, the two loop-invariant pointer bases,
and the zero accumulator. -/
private def ivdmPrologue (A B : RegionName) (M N SAM SAK SBK SBN BM BN BK : Nat) : List Stmt :=
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
    Stmt.assign .nat [BK] "offs_k" (Op.arange BK),
    Stmt.assign .ptr [BM, BK] "a_ptr"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase A)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_am")) (Op.constNat SAM))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BK] "offs_k")) (Op.constNat SAK)))),
    Stmt.assign .ptr [BK, BN] "b_ptr"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase B)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "offs_k")) (Op.constNat SBK))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_bn")) (Op.constNat SBN)))),
    Stmt.assign .real [BM, BN] "accumulator" (Op.full [BM, BN] (Op.const 0)) ]

/-- The prologue is exactly `body.take 10`. By `rfl`. -/
private theorem ivdm_take10_eq (A B C : RegionName)
    (M N SAM SAK SBK SBN SCM SCN BM BN BK numKBlocks : Nat) :
    (iv_dependent_matmul_pre_load_surface A B C M N (BK * numKBlocks) SAM SAK SBK SBN
        SCM SCN BM BN BK).toAlgKernel.body.take 10
      = ivdmPrologue A B M N SAM SAK SBK SBN BM BN BK := rfl

/-- The dynamic K-loop's `stop` op (`tl.cdiv(K, BLOCK_K)`). -/
private def ivdmKStop (K BK : Nat) : Op .nat [] :=
  Op.div .nat Broadcast.nil (Op.sub .nat Broadcast.nil
    (Op.add .nat Broadcast.nil (Op.constNat K) (Op.constNat BK)) (Op.constNat 1)) (Op.constNat BK)

/-- `ivdm_body_split` with the prologue and the loop's `stop` op named. -/
private theorem ivdm_body_split' (A B C : RegionName)
    (M N SAM SAK SBK SBN SCM SCN BM BN BK numKBlocks : Nat) :
    (iv_dependent_matmul_pre_load_surface A B C M N (BK * numKBlocks) SAM SAK SBK SBN
        SCM SCN BM BN BK).toAlgKernel.body
      = ivdmPrologue A B M N SAM SAK SBK SBN BM BN BK
        ++ (Stmt.forRangeDyn "k" (Op.constNat 0) (ivdmKStop (BK * numKBlocks) BK) (Op.constNat 1)
              (ivdmLoopBody BM BN BK (BK * numKBlocks) SAK SBK)
            :: ivdmPostBody C M N SCM SCN BM BN) := by
  rw [ivdm_body_split A B C M N SAM SAK SBK SBN SCM SCN BM BN BK numKBlocks,
    ivdm_take10_eq A B C M N SAM SAK SBK SBN SCM SCN BM BN BK numKBlocks]
  rfl

set_option maxHeartbeats 2000000 in
/-- The prologue is cast-free: it steps identically under `stepStmtsR R`. -/
private theorem ivdmPrologue_castFree (R : RoundingModel) (A B : RegionName)
    (M N SAM SAK SBK SBN BM BN BK : Nat) (t : BlockState) :
    stepStmtsR R (ivdmPrologue A B M N SAM SAK SBK SBN BM BN BK) t
      = stepStmts (ivdmPrologue A B M N SAM SAK SBK SBN BM BN BK) t := by
  simp only [ivdmPrologue, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def]
  rfl

set_option maxHeartbeats 1000000 in
/-- The K-loop body is cast-free: the two IV-dependent pointer recomputes are
`.ptr`/`.nat` arithmetic, the masked loads' `mask=`/`other=` operands are nat
comparisons and a real constant (no `castFloat`), the fused `tl.dot`+add is an
exact op, and the body has no store. -/
private theorem ivdmBody_castFree (R : RoundingModel) (BM BN BK K SAK SBK : Nat)
    (t : BlockState) :
    stepStmtsR R (ivdmLoopBody BM BN BK K SAK SBK) t
      = stepStmts (ivdmLoopBody BM BN BK K SAK SBK) t := by
  simp only [ivdmLoopBody, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def]
  rfl

/-- `tl.cdiv(K, BLOCK_K) = numKBlocks` at the exact-multiple contraction. -/
private theorem ivdmKStop_eval (s : BlockState) (BK numKBlocks : Nat) (hBK : 0 < BK) :
    evalOp (ivdmKStop (BK * numKBlocks) BK) s = some (Tile.scalar numKBlocks) := by
  have hcdiv : (BK * numKBlocks + BK - 1) / BK = numKBlocks := by
    have he : BK * numKBlocks + BK - 1 = (BK - 1) + BK * numKBlocks := by omega
    rw [he, Nat.add_mul_div_left _ _ hBK, Nat.div_eq_of_lt (by omega), Nat.zero_add]
  unfold ivdmKStop
  simp only [evalOp_div, evalOp_sub, evalOp_add, evalOp_constNat, Option.bind_some,
    Tile.bop, NumericDType.div, NumericDType.sub, NumericDType.add]
  refine congrArg some ?_
  ext
  show (BK * numKBlocks + BK - 1) / BK = numKBlocks
  exact hcdiv

/-- The loop's `stop` op is cast-free, so it resolves the same under `R`. -/
private theorem ivdmKStop_evalR (R : RoundingModel) (s : BlockState)
    (BK numKBlocks : Nat) (hBK : 0 < BK) :
    evalOpR R (ivdmKStop (BK * numKBlocks) BK) s = some (Tile.scalar numKBlocks) := by
  rw [show evalOpR R (ivdmKStop (BK * numKBlocks) BK) s
      = evalOp (ivdmKStop (BK * numKBlocks) BK) s from by
    simp only [ivdmKStop, evalOpR.eq_def, evalOp.eq_def]]
  exact ivdmKStop_eval s BK numKBlocks hBK

/-- `stepStmtR R` on the dynamic K-loop resolves its three cast-free bound
ops and becomes the `R` strided fold at `0 .. numKBlocks` step `1`. -/
private theorem stepStmtR_ivdmKLoop (R : RoundingModel) (BM BN BK SAK SBK numKBlocks : Nat)
    (hBK : 0 < BK) (s : BlockState) :
    stepStmtR R (Stmt.forRangeDyn "k" (Op.constNat 0) (ivdmKStop (BK * numKBlocks) BK)
        (Op.constNat 1) (ivdmLoopBody BM BN BK (BK * numKBlocks) SAK SBK)) s
      = stepForRangeAuxR R "k" 0 numKBlocks 1
          (ivdmLoopBody BM BN BK (BK * numKBlocks) SAK SBK) s := by
  simp only [stepStmtR, ivdmKStop_evalR R s BK numKBlocks hBK,
    show evalOpR R (Op.constNat (0 : Nat)) s = some (Tile.scalar 0) from by
      simp only [evalOpR.eq_def, evalOp_constNat],
    show evalOpR R (Op.constNat (1 : Nat)) s = some (Tile.scalar 1) from by
      simp only [evalOpR.eq_def, evalOp_constNat],
    Option.bind_eq_bind, Option.bind_some]
  rfl

/-- The whole dynamic K-loop statement is cast-free (bound ops + body). -/
private theorem ivdmKLoop_castFree (R : RoundingModel) (BM BN BK SAK SBK numKBlocks : Nat)
    (hBK : 0 < BK) (s : BlockState) :
    stepStmtR R (Stmt.forRangeDyn "k" (Op.constNat 0) (ivdmKStop (BK * numKBlocks) BK)
        (Op.constNat 1) (ivdmLoopBody BM BN BK (BK * numKBlocks) SAK SBK)) s
      = stepStmt (Stmt.forRangeDyn "k" (Op.constNat 0) (ivdmKStop (BK * numKBlocks) BK)
        (Op.constNat 1) (ivdmLoopBody BM BN BK (BK * numKBlocks) SAK SBK)) s := by
  rw [stepStmtR_ivdmKLoop R BM BN BK SAK SBK numKBlocks hBK s,
    stepForRangeAuxR_castFree R _
      (ivdmBody_castFree R BM BN BK (BK * numKBlocks) SAK SBK) "k",
    stepForRangeAux.forRangeDyn_unfold, evalOp_constNat,
    ivdmKStop_eval s BK numKBlocks hBK, evalOp_constNat]
  rfl

/-! ### Cast-free op collapses and the two rounding-event sites -/

/-- The IV-dependent `a_ptrs` / `b_ptrs` recompute op is cast-free. -/
private theorem evalR_ptrsRecompute (R : RoundingModel) (P Q : Nat)
    (baseReg : RegName) (BK Sk : Nat) (s : BlockState) :
    evalOpR R (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [P, Q] baseReg)
        (Op.mul .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "k") (Op.constNat BK)) (Op.constNat Sk))) s
      = evalOp (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [P, Q] baseReg)
        (Op.mul .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "k") (Op.constNat BK)) (Op.constNat Sk))) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

/-- A scalar `ptrAdd` on an explicit region-tagged address tile is the tile
with the constant folded into every address. -/
private theorem ptrAdd_scalarR_data (P Q : Nat) (Rg : RegionName)
    (f : TileIndex [P, Q] → Nat) (d : Nat) :
    (Tile.ptrAdd Broadcast.scalarR (⟨fun idx => (Rg.cast, f idx)⟩ : Tile .ptr [P, Q])
        (Tile.scalar d))
      = (⟨fun idx => (Rg.cast, f idx + d)⟩ : Tile .ptr [P, Q]) := by
  ext idx <;> simp only [Tile.ptrAdd, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    Tile.scalar, NumericDType.add]

/-- The `offs_c*` index-vector op is cast-free. -/
private theorem evalR_offsc (R : RoundingModel) (Bv : Nat) (pidReg : RegName) (s : BlockState) :
    evalOpR R (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] pidReg) (Op.constNat Bv))
        (Op.arange Bv)) s
      = evalOp (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] pidReg) (Op.constNat Bv))
        (Op.arange Bv)) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

/-- The `c_ptrs` pointer op is cast-free. -/
private theorem evalR_cptrs (R : RoundingModel) (C : RegionName) (BM BN SCM SCN : Nat)
    (s : BlockState) :
    evalOpR R (Op.ptrAdd Broadcast.scalarL (Op.ptrBase C)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarL (Op.constNat SCM) (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_cm")))
          (Op.mul .nat Broadcast.scalarL (Op.constNat SCN) (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_cn"))))) s
      = evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase C)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarL (Op.constNat SCM) (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_cm")))
          (Op.mul .nat Broadcast.scalarL (Op.constNat SCN) (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_cn"))))) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

/-- The `c_mask` boolean op is cast-free. -/
private theorem evalR_cmask (R : RoundingModel) (M N BM BN : Nat) (s : BlockState) :
    evalOpR R (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_cm")) (Op.constNat M))
        (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_cn")) (Op.constNat N))) s
      = evalOp (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_cm")) (Op.constNat M))
        (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_cn")) (Op.constNat N))) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

/-- **`c_mask` eval** (the genuine boundary mask, not the all-true
degeneration): lane `(i,j)` is `offs_cm i < M ∧ offs_cn j < N`. -/
private theorem cmask_eval (s : BlockState) (M N BM BN : Nat)
    (gm : Fin BM → Nat) (gn : Fin BN → Nat)
    (hm : s.regs .nat [BM] "offs_cm" = some (Tile.vec gm))
    (hn : s.regs .nat [BN] "offs_cn" = some (Tile.vec gn)) :
    evalOp (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_cm")) (Op.constNat M))
        (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_cn")) (Op.constNat N))) s
      = some ⟨fun idx : TileIndex [BM, BN] =>
          (decide (gm idx.1 < M) && decide (gn idx.2.1 < N))⟩ := by
  simp only [evalOp, evalOp.eq_def, hm, hn, Option.bind, Option.bind_some, Option.bind_eq_bind,
    Tile.expandDim, pure, Option.some.injEq]
  ext idx
  simp [Tile.bop, Tile.cop, Tile.expandDim, Tile.vec, ComparableDType.lt]

/-- `c = tl.cast(accumulator, tl.float16)` eval under `R`: rounding-event
site 1, `R.cast` applied lane-wise to the accumulator tile. -/
private theorem castAcc_evalR (R : RoundingModel) (BM BN : Nat) (s : BlockState)
    (zT : Tile .real [BM, BN]) (hz : s.regs .real [BM, BN] "accumulator" = some zT) :
    evalOpR R (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref .real [BM, BN] "accumulator")) s
      = some ⟨fun idx => RoundingModel.cast R FloatDType.real FloatDType.fp16 (zT.data idx)⟩ := by
  have hz' : s.regs FloatDType.real.toTileDType [BM, BN] "accumulator" = some zT := hz
  simp only [evalOpR.eq_def, hz']
  rfl

/-! ### The tail under `execR R` -/

set_option maxHeartbeats 4000000 in
/-- **R-postLoop**: from the exact invariant at `numKBlocks` blocks, the
`execR R` fp16 cast + masked store tail terminates and writes, at every
**active** output lane, the cell `fp16(R.round .fp16 (matmulSpec …))` — the
ideal GEMM value rounded **once** at the fp16 grid (`R.cast` site +
`R.storeValue` site collapsed by `round_idem`); inactive lanes' cells and
every cell not hit by an active lane are untouched (the masked-store
frame). -/
private theorem ivdm_postLoopR (R : RoundingModel) (A B C : RegionName) (s0 : BlockState)
    (PM PN M N BM BN SAM SAK SBK SBN SCM SCN BK numKBlocks : Nat)
    (hInj : Function.Injective (cOffset s0 PM PN BM BN SCM SCN))
    (st : BlockState)
    (hinv : ivdmInvariant A B s0 PM PN M N BM BN SAM SAK SBK SBN BK numKBlocks numKBlocks st) :
    ∃ sfin, stepStmtsR R (ivdmPostBody C M N SCM SCN BM BN) st = some sfin
      ∧ (∀ idx : TileIndex [BM, BN],
          sfin.mem C (cOffset s0 PM PN BM BN SCM SCN idx)
            = if rowIndex PM BM idx.1 < M ∧ colIndex PN BN idx.2.1 < N then
                MemCell.of FloatDType.fp16.toTileDType
                  (FloatDType.fp16.ofReal
                    (R.round .fp16
                      (matmulSpec s0 A B PM PN BM BN M N SAM SAK SBK SBN BK numKBlocks
                        idx.1 idx.2.1)))
              else st.mem C (cOffset s0 PM PN BM BN SCM SCN idx))
      ∧ (∀ r o, (r ≠ C ∨ ∀ idx : TileIndex [BM, BN],
            (rowIndex PM BM idx.1 < M ∧ colIndex PN BN idx.2.1 < N) →
              o ≠ cOffset s0 PM PN BM BN SCM SCN idx) →
          sfin.mem r o = st.mem r o) := by
  simp only [ivdmInvariant] at hinv
  obtain ⟨hpids, hcle, hz, hpm, hpn, ham, hbn, hk, hapb, hbpb, hundef, hmem⟩ := hinv
  set g : TileIndex [BM, BN] → ℝ :=
    fun idx => matmulSpec s0 A B PM PN BM BN M N SAM SAK SBK SBN BK numKBlocks idx.1 idx.2.1 with hg
  have hzspec : st.regs .real [BM, BN] "accumulator" = some ⟨fun idx => some (g idx)⟩ := by
    rw [hz]; refine congrArg some ?_; ext idx
    simp only [hg, matmulSpec, accPartial, Nat.mul_comm numKBlocks BK]
  unfold ivdmPostBody
  -- 1. c = cast(accumulator, fp16): rounding-event site 1
  set zT : Tile .real [BM, BN] := ⟨fun idx => some (g idx)⟩ with hzT
  set cT : Tile .fp16 [BM, BN] :=
    ⟨fun idx => RoundingModel.cast R FloatDType.real FloatDType.fp16 (zT.data idx)⟩ with hcT
  erw [stepStmtsR_cons_some (stepStmtR_assign_eq_some
        (show evalOpR R (Op.castFloat FloatDType.real FloatDType.fp16
              (Op.ref .real [BM, BN] "accumulator")) st = some cT
          from castAcc_evalR R BM BN st zT hzspec))]
  set s1 := st.setReg "c" FloatDType.fp16.toTileDType [BM, BN] cT with hs1
  -- 2/3. offs_cm / offs_cn (cast-free)
  have hpm1 : s1.regs .nat [] "pid_m" = some (Tile.scalar PM) := by simp [hs1, hpm]
  rw [stepStmtsR_cons_some (stepStmtR_assign_eq_some
        (show evalOpR R (Op.add .nat Broadcast.scalarL
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat BM)) (Op.arange BM)) s1
          = some (Tile.vec (fun i : Fin BM => rowIndex PM BM i))
          from by rw [evalR_offsc]; exact offs_cm_eval s1 PM BM hpm1))]
  set s2 := s1.setReg "offs_cm" .nat [BM] (Tile.vec (fun i : Fin BM => rowIndex PM BM i)) with hs2
  have hpn2 : s2.regs .nat [] "pid_n" = some (Tile.scalar PN) := by simp [hs2, hs1, hpn]
  rw [stepStmtsR_cons_some (stepStmtR_assign_eq_some
        (show evalOpR R (Op.add .nat Broadcast.scalarL
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_n") (Op.constNat BN)) (Op.arange BN)) s2
          = some (Tile.vec (fun j : Fin BN => colIndex PN BN j))
          from by rw [evalR_offsc]; exact offs_cn_eval s2 PN BN hpn2))]
  set s3 := s2.setReg "offs_cn" .nat [BN] (Tile.vec (fun j : Fin BN => colIndex PN BN j)) with hs3
  -- 4. c_ptrs (cast-free)
  set cpT : Tile .ptr [BM, BN] :=
    ⟨fun idx : TileIndex [BM, BN] => (C.cast, SCM * rowIndex PM BM idx.1 + SCN * colIndex PN BN idx.2.1)⟩
    with hcpT
  have hcm3 : s3.regs .nat [BM] "offs_cm" = some (Tile.vec (fun i : Fin BM => rowIndex PM BM i)) := by
    simp [hs3, hs2]
  have hcn3 : s3.regs .nat [BN] "offs_cn" = some (Tile.vec (fun j : Fin BN => colIndex PN BN j)) := by
    simp [hs3]
  rw [stepStmtsR_cons_some (stepStmtR_assign_eq_some
        (show evalOpR R (Op.ptrAdd Broadcast.scalarL (Op.ptrBase C)
            (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
              (Op.mul .nat Broadcast.scalarL (Op.constNat SCM) (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_cm")))
              (Op.mul .nat Broadcast.scalarL (Op.constNat SCN) (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_cn"))))) s3
          = some cpT from by
          rw [evalR_cptrs, cptrs_eval s3 C BM BN SCM SCN
            (fun i => rowIndex PM BM i) (fun j => colIndex PN BN j) hcm3 hcn3]))]
  set s4 := s3.setReg "c_ptrs" .ptr [BM, BN] cpT with hs4
  -- 5. c_mask (cast-free, genuine boundary mask)
  set cmT : Tile .bool [BM, BN] :=
    ⟨fun idx : TileIndex [BM, BN] =>
      (decide (rowIndex PM BM idx.1 < M) && decide (colIndex PN BN idx.2.1 < N))⟩ with hcmT
  have hcm4 : s4.regs .nat [BM] "offs_cm" = some (Tile.vec (fun i : Fin BM => rowIndex PM BM i)) := by
    simp [hs4, hcm3]
  have hcn4 : s4.regs .nat [BN] "offs_cn" = some (Tile.vec (fun j : Fin BN => colIndex PN BN j)) := by
    simp [hs4, hcn3]
  rw [stepStmtsR_cons_some (stepStmtR_assign_eq_some
        (show evalOpR R (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
            (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_cm")) (Op.constNat M))
            (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_cn")) (Op.constNat N))) s4
          = some cmT from by
          rw [evalR_cmask, cmask_eval s4 M N BM BN
            (fun i => rowIndex PM BM i) (fun j => colIndex PN BN j) hcm4 hcn4]))]
  set s5 := s4.setReg "c_mask" .bool [BM, BN] cmT with hs5
  have hc5 : s5.regs .fp16 [BM, BN] "c" = some cT := by simp [hs5, hs4, hs3, hs2, hs1]
  have hcp5 : s5.regs .ptr [BM, BN] "c_ptrs" = some cpT := by simp [hs5, hs4]
  have hcm5 : s5.regs .bool [BM, BN] "c_mask" = some cmT := by simp [hs5]
  have hmem5 : s5.mem = st.mem := by
    funext region offset
    simp only [hs5, hs4, hs3, hs2, hs1, BlockState.setReg_mem]
  -- 6. the masked fp16 store: rounding-event site 2
  have hstore : stepStmtR R (Stmt.store .fp16 [BM, BN] (.ptr (Op.ref .ptr [BM, BN] "c_ptrs"))
        (Op.ref .fp16 [BM, BN] "c") (.mask (Op.ref .bool [BM, BN] "c_mask"))) s5
      = some ((TileShape.allIndices [BM, BN]).foldl
          (fun acc i =>
            if cmT.data i then
              acc.writeMemAsR R .fp16 C (cOffset s0 PM PN BM BN SCM SCN i) (cT.data i)
            else acc) s5) := by
    simp only [stepStmtR]
    rw [show evalOpR R (Op.ref .fp16 [BM, BN] "c") s5 = some cT from by rw [evalOpR_ref, hc5]]
    rw [show evalOpR R (Op.ref .bool [BM, BN] "c_mask") s5 = some cmT from by rw [evalOpR_ref, hcm5]]
    rw [show evalOpR R (Op.ref .ptr [BM, BN] "c_ptrs") s5 = some cpT from by rw [evalOpR_ref, hcp5]]
    simp only [bind, Option.map_some, Option.bind_some]
    refine congrArg some (List.foldl_ext _ _ _ (fun acc i _ => ?_))
    by_cases hmask : cmT.data i
    · simp only [hmask, if_true, hcpT, cOffset, Region.cast_id, BlockState.writeMemTypedR_fp16]
    · simp only [hmask, Bool.false_eq_true, if_false]
  rw [stepStmtsR_cons_some hstore, stepStmtsR_nil]
  refine ⟨_, rfl, ?_, ?_⟩
  · intro idx
    rw [BlockState.scatter_memcell_R_prop_masked_nd R .fp16 (region := C) s5
        (cOffset s0 PM PN BM BN SCM SCN) (fun i => cT.data i)
        (fun i => cmT.data i = Bool.true) hInj idx]
    by_cases hact : rowIndex PM BM idx.1 < M ∧ colIndex PN BN idx.2.1 < N
    · have hmasktrue : cmT.data idx = Bool.true := by
        simp only [hcmT]
        obtain ⟨hr, hcc⟩ := hact
        simp [hr, hcc]
      rw [if_pos hmasktrue, if_pos hact]
      have hdata : cT.data idx
          = RoundingModel.cast R FloatDType.real FloatDType.fp16 (some (g idx)) := rfl
      rw [hdata, RoundingModel.storeValue_cast]
    · have hmaskfalse : ¬ (cmT.data idx = Bool.true) := by
        simp only [hcmT, Bool.and_eq_true, decide_eq_true_eq, not_and]
        intro hr hcc
        exact hact ⟨hr, hcc⟩
      rw [if_neg hmaskfalse, if_neg hact, hmem5]
  · intro r o hcond
    by_cases hr : r = C
    · subst hr
      have hno : ∀ idx : TileIndex [BM, BN],
          (rowIndex PM BM idx.1 < M ∧ colIndex PN BN idx.2.1 < N) →
            o ≠ cOffset s0 PM PN BM BN SCM SCN idx := by
        rcases hcond with h | h
        · exact absurd rfl h
        · exact h
      rw [BlockState.foldl_writeMemAsR_preserve_masked_prop R .fp16
            (cOffset s0 PM PN BM BN SCM SCN) (fun i => cT.data i)
            (fun i => cmT.data i = Bool.true) o (TileShape.allIndices [BM, BN])
            (fun k _ hk2 => by
              have hkact : rowIndex PM BM k.1 < M ∧ colIndex PN BN k.2.1 < N := by
                simp only [hcmT, Bool.and_eq_true, decide_eq_true_eq] at hk2
                exact ⟨hk2.1, hk2.2⟩
              exact fun heq => hno k hkact heq.symm) s5, hmem5]
    · rw [BlockState.foldl_writeMemAsR_preserve_other_region R .fp16
            (cOffset s0 PM PN BM BN SCM SCN) (fun i => cT.data i)
            (fun i => cmT.data i = Bool.true) r hr o (TileShape.allIndices [BM, BN]) s5,
          hmem5]

/-! ### Safety-walk invariant (weak shape half of `ivdmInvariant`) -/

/-- Safety-walk loop invariant: the *shape* half of `ivdmInvariant`
(`pid_m`/`pid_n` seeded, `offs_k`, *some* accumulator tile, and the two
loop-invariant pointer **bases**) with no `undef`/`mem`/value pins. Because
the body recomputes `a_ptrs`/`b_ptrs` from those bases every iteration, the
invariant carries **no** loop-counter index at all. Needed because the
`⊨[R]` skin's `hts` obligation quantifies over arbitrary launch states. -/
private def ivdmSafeInv (A B : RegionName) (PM PN M N BM BN SAM SAK SBK SBN BK : Nat)
    (s : BlockState) : Prop :=
  (s.regs .nat [] "pid_m" = some (Tile.scalar PM)) ∧
  (s.regs .nat [] "pid_n" = some (Tile.scalar PN)) ∧
  (s.regs .nat [BK] "offs_k" = some (Tile.vec (fun e : Fin BK => e.val))) ∧
  (∃ zT : Tile .real [BM, BN], s.regs .real [BM, BN] "accumulator" = some zT) ∧
  (s.regs .ptr [BM, BK] "a_ptr" = some ⟨fun idx : TileIndex [BM, BK] =>
      (A.cast, (PM * BM + idx.1.val) % M * SAM + idx.2.1.val * SAK)⟩) ∧
  (s.regs .ptr [BK, BN] "b_ptr" = some ⟨fun idx : TileIndex [BK, BN] =>
      (B.cast, idx.1.val * SBK + (PN * BN + idx.2.1.val) % N * SBN)⟩)

set_option maxHeartbeats 2000000 in
/-- Weak `preLoop`: from an **arbitrary** state the prologue steps to a state
satisfying `ivdmSafeInv` (no clean-`undef` hypothesis; the value half of
`ivdm_preLoop` is dropped). -/
private theorem ivdm_preLoopW (A B : RegionName) (s : BlockState)
    (M N SAM SAK SBK SBN BM BN BK : Nat) :
    ∃ s', stepStmts (ivdmPrologue A B M N SAM SAK SBK SBN BM BN BK) s = some s'
      ∧ ivdmSafeInv A B (pidM (s.pids 0) N BN) (pidN (s.pids 0) N BN)
          M N BM BN SAM SAK SBK SBN BK s' := by
  obtain ⟨s7, h7, hpids, hpm, hpn, ham, hbn, hk, huf, hmem⟩ :=
    preLoop_scalars s M N BM BN BK
  set PM := pidM (s.pids 0) N BN with hPM
  set PN := pidN (s.pids 0) N BN with hPN
  rw [show ivdmPrologue A B M N SAM SAK SBK SBN BM BN BK
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
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp [hpm]
  · simp [hpn]
  · simp [hk]
  · refine ⟨⟨fun _ => some (0 : ℝ)⟩, ?_⟩
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
      String.reduceEq, not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]

set_option maxHeartbeats 4000000 in
/-- Weak step lemma: one body iteration from `ivdmSafeInv` steps successfully
(exact stepper; the body is cast-free, and at exact-multiple `K` the load
masks evaluate all-true) and re-establishes the invariant. -/
private theorem ivdm_stepW (A B : RegionName)
    (PM PN M N BM BN SAM SAK SBK SBN BK T : Nat)
    (c : Nat) (s : BlockState) (hclt : c < T)
    (hinv : ivdmSafeInv A B PM PN M N BM BN SAM SAK SBK SBN BK s) :
    ∃ s', stepStmts (ivdmLoopBody BM BN BK (BK * T) SAK SBK)
        (s.setReg "k" .nat [] (Tile.scalar c)) = some s'
      ∧ ivdmSafeInv A B PM PN M N BM BN SAM SAK SBK SBN BK s' := by
  obtain ⟨hpm, hpn, hk, ⟨zT, hz⟩, hapb, hbpb⟩ := hinv
  have hlt : ∀ e : Fin BK, e.val < BK * T - c * BK := by
    intro e
    have hcK : c * BK + BK ≤ BK * T := by
      calc c * BK + BK = (c + 1) * BK := by ring
        _ ≤ T * BK := Nat.mul_le_mul_right _ hclt
        _ = BK * T := Nat.mul_comm _ _
    omega
  set apbT : Tile .ptr [BM, BK] :=
    ⟨fun idx : TileIndex [BM, BK] => (A.cast, (PM * BM + idx.1.val) % M * SAM + idx.2.1.val * SAK)⟩
    with hapbT
  set bpbT : Tile .ptr [BK, BN] :=
    ⟨fun idx : TileIndex [BK, BN] => (B.cast, idx.1.val * SBK + (PN * BN + idx.2.1.val) % N * SBN)⟩
    with hbpbT
  set sk := s.setReg "k" .nat [] (Tile.scalar c) with hsk
  have hapbk : sk.regs .ptr [BM, BK] "a_ptr" = some apbT := by simp [hsk, hapb, hapbT]
  have hbpbk : sk.regs .ptr [BK, BN] "b_ptr" = some bpbT := by simp [hsk, hbpb, hbpbT]
  have hzk : sk.regs .real [BM, BN] "accumulator" = some zT := by simp [hsk, hz]
  have hkk : sk.regs .nat [BK] "offs_k" = some (Tile.vec (fun e : Fin BK => e.val)) := by simp [hsk, hk]
  have hkkv : sk.regs .nat [] "k" = some (Tile.scalar c) := by simp [hsk]
  set apT : Tile .ptr [BM, BK] := Tile.ptrAdd Broadcast.scalarR apbT (Tile.scalar (c * BK * SAK)) with hapT
  set sk1 := sk.setReg "a_ptrs" .ptr [BM, BK] apT with hsk1
  set bpT : Tile .ptr [BK, BN] := Tile.ptrAdd Broadcast.scalarR bpbT (Tile.scalar (c * BK * SBK)) with hbpT
  set sk2 := sk1.setReg "b_ptrs" .ptr [BK, BN] bpT with hsk2
  have hkk2 : sk2.regs .nat [BK] "offs_k" = some (Tile.vec (fun e : Fin BK => e.val)) := by
    simp [hsk2, hsk1, hkk]
  have hkkv2 : sk2.regs .nat [] "k" = some (Tile.scalar c) := by simp [hsk2, hsk1, hkkv]
  set asub : Tile .real [BM, BK] :=
    ⟨fun idx => some (sk2.readMem (apT.data idx).1 (apT.data idx).2)⟩ with hasub
  set sk3 := sk2.setReg "a" .real [BM, BK] asub with hsk3
  set bsub : Tile .real [BK, BN] :=
    ⟨fun idx => some (sk3.readMem (bpT.data idx).1 (bpT.data idx).2)⟩ with hbsub
  set sk4 := sk3.setReg "b" .real [BK, BN] bsub with hsk4
  have hkk3 : sk3.regs .nat [BK] "offs_k" = some (Tile.vec (fun e : Fin BK => e.val)) := by
    simp [hsk3, hkk2]
  have hkkv3 : sk3.regs .nat [] "k" = some (Tile.scalar c) := by simp [hsk3, hkkv2]
  obtain ⟨amt, ham_eval, ham_true⟩ := amask_alltrue sk2 BM BK (BK * T) c hkk2 hkkv2 hlt
  obtain ⟨bmt, hbm_eval, hbm_true⟩ := bmask_alltrue sk3 BN BK (BK * T) c hkk3 hkkv3 hlt
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
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp [hsk4, hsk3, hsk2, hsk1, hsk, hpm]
  · simp [hsk4, hsk3, hsk2, hsk1, hsk, hpn]
  · simp [hsk4, hsk3, hsk2, hsk1, hsk, hk]
  · refine ⟨Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
      zT (Tile.dot [] asub bsub), ?_⟩
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
      String.reduceEq, not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
  · simp [hsk4, hsk3, hsk2, hsk1, hsk, hapb, hapbT]
  · simp [hsk4, hsk3, hsk2, hsk1, hsk, hbpb, hbpbT]

set_option maxHeartbeats 2000000 in
/-- Per-iteration `TraceSafeListR` for the K-loop body: the two IV-dependent
pointer recomputes land on the base tiles shifted by `c·BK·stride`, and the
two **masked** loads' addresses are exactly those, in bounds at *every* lane
by the all-lane bound groups (active lanes are a subset); the accumulator
assign is unconditionally safe. -/
private theorem ivdm_bodySafeR (R : RoundingModel) (bounds : RegionBounds)
    (A B : RegionName)
    (PM PN M N BM BN SAM SAK SBK SBN BK T K : Nat) (c : Nat) (hc : c < T)
    (sk : BlockState)
    (hapb : sk.regs .ptr [BM, BK] "a_ptr" = some ⟨fun idx : TileIndex [BM, BK] =>
        (A.cast, (PM * BM + idx.1.val) % M * SAM + idx.2.1.val * SAK)⟩)
    (hbpb : sk.regs .ptr [BK, BN] "b_ptr" = some ⟨fun idx : TileIndex [BK, BN] =>
        (B.cast, idx.1.val * SBK + (PN * BN + idx.2.1.val) % N * SBN)⟩)
    (hkkv : sk.regs .nat [] "k" = some (Tile.scalar c))
    (hbA : ∀ (t : Fin T) (l : Fin (BM * BK)),
      (PM * BM + l.val / BK) % M * SAM + (t.val * BK + l.val % BK) * SAK < bounds A)
    (hbB : ∀ (t : Fin T) (l : Fin (BK * BN)),
      (t.val * BK + l.val / BN) * SBK + (PN * BN + l.val % BN) % N * SBN < bounds B) :
    Stmt.TraceSafeListR R bounds (ivdmLoopBody BM BN BK K SAK SBK) sk := by
  unfold ivdmLoopBody
  -- 1. a_ptrs recompute
  refine Stmt.TraceSafeListR.cons_intro (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) ?_
  intro s1 h1
  obtain ⟨v1, hv1, rfl⟩ := stepStmtR_assign_inv h1
  rw [evalR_ptrsRecompute, aptrs_recompute_eval sk BM BK BK SAK c _ hapb hkkv,
    ptrAdd_scalarR_data] at hv1
  obtain rfl := Option.some.inj hv1
  -- 2. b_ptrs recompute
  refine Stmt.TraceSafeListR.cons_intro (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) ?_
  intro s2 h2
  obtain ⟨v2, hv2, rfl⟩ := stepStmtR_assign_inv h2
  rw [evalR_ptrsRecompute, bptrs_recompute_eval _ BK BN BK SBK c
      (⟨fun idx : TileIndex [BK, BN] =>
        (B.cast, idx.1.val * SBK + (PN * BN + idx.2.1.val) % N * SBN)⟩ : Tile .ptr [BK, BN])
      (by simp [hbpb]) (by simp [hkkv]),
    ptrAdd_scalarR_data] at hv2
  obtain rfl := Option.some.inj hv2
  -- 3. a = load A[a_ptrs] (masked)
  refine Stmt.TraceSafeListR.cons_intro ?_ ?_
  · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
    refine ⟨trivial, by simp, ?_⟩
    intro ptrs hptrs i _
    rw [evalOpR_ref] at hptrs
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff] at hptrs
    obtain rfl := Option.some.inj hptrs
    show (PM * BM + i.1.val) % M * SAM + i.2.1.val * SAK + c * BK * SAK < bounds (Region.cast A)
    have h' := hbA ⟨c, hc⟩ (Lane2D.encode (i.1, i.2.1, PUnit.unit))
    rw [Lane2D.encode_div, Lane2D.encode_mod] at h'
    simp only [Region.cast_id]
    calc (PM * BM + i.1.val) % M * SAM + i.2.1.val * SAK + c * BK * SAK
        = (PM * BM + i.1.val) % M * SAM + (c * BK + i.2.1.val) * SAK := by ring
      _ < bounds A := h'
  intro s3 h3
  obtain ⟨v3, -, rfl⟩ := stepStmtR_assign_inv h3
  -- 4. b = load B[b_ptrs] (masked)
  refine Stmt.TraceSafeListR.cons_intro ?_ ?_
  · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
    refine ⟨trivial, by simp, ?_⟩
    intro ptrs hptrs i _
    rw [evalOpR_ref] at hptrs
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff] at hptrs
    obtain rfl := Option.some.inj hptrs
    show i.1.val * SBK + (PN * BN + i.2.1.val) % N * SBN + c * BK * SBK < bounds (Region.cast B)
    have h' := hbB ⟨c, hc⟩ (Lane2D.encode (i.1, i.2.1, PUnit.unit))
    rw [Lane2D.encode_div, Lane2D.encode_mod] at h'
    simp only [Region.cast_id]
    calc i.1.val * SBK + (PN * BN + i.2.1.val) % N * SBN + c * BK * SBK
        = (c * BK + i.1.val) * SBK + (PN * BN + i.2.1.val) % N * SBN := by ring
      _ < bounds B := h'
  intro s4 h4
  obtain ⟨v4, -, rfl⟩ := stepStmtR_assign_inv h4
  -- 5. accumulator += dot
  refine Stmt.TraceSafeListR.of_forall _ _ ?_
  intro stx hstx s'
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hstx
  rcases hstx with rfl <;> simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]

set_option maxHeartbeats 2000000 in
/-- `TraceSafeListR` for the store tail: the five assigns are register-only
(the fp16 cast is not a memory event) and the single **masked** `.fp16`
store's active lanes are exactly the `(row<M)&(col<N)` window, so their
addresses are the skin's `writeMask`-gated `write` bounds. -/
private theorem ivdm_tailSafeR (R : RoundingModel) (bounds : RegionBounds)
    (C : RegionName) (PM PN M N BM BN SCM SCN : Nat) (st : BlockState)
    (hpm : st.regs .nat [] "pid_m" = some (Tile.scalar PM))
    (hpn : st.regs .nat [] "pid_n" = some (Tile.scalar PN))
    (hbC : ∀ l : Fin (BM * BN),
      (PM * BM + l.val / BN < M ∧ PN * BN + l.val % BN < N) →
      SCM * (PM * BM + l.val / BN) + SCN * (PN * BN + l.val % BN) < bounds C) :
    Stmt.TraceSafeListR R bounds (ivdmPostBody C M N SCM SCN BM BN) st := by
  unfold ivdmPostBody
  -- 1. fp16 cast
  refine Stmt.TraceSafeListR.cons_intro (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) ?_
  intro s1 h1
  obtain ⟨v1, -, rfl⟩ := stepStmtR_assign_inv h1
  -- 2. offs_cm
  refine Stmt.TraceSafeListR.cons_intro (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) ?_
  intro s2 h2
  obtain ⟨v2, hv2, rfl⟩ := stepStmtR_assign_inv h2
  rw [evalR_offsc, offs_cm_eval _ PM BM (by simp [hpm])] at hv2
  obtain rfl := Option.some.inj hv2
  -- 3. offs_cn
  refine Stmt.TraceSafeListR.cons_intro (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) ?_
  intro s3 h3
  obtain ⟨v3, hv3, rfl⟩ := stepStmtR_assign_inv h3
  rw [evalR_offsc, offs_cn_eval _ PN BN (by simp [hpn])] at hv3
  obtain rfl := Option.some.inj hv3
  -- 4. c_ptrs
  refine Stmt.TraceSafeListR.cons_intro (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) ?_
  intro s4 h4
  obtain ⟨v4, hv4, rfl⟩ := stepStmtR_assign_inv h4
  rw [evalR_cptrs, cptrs_eval _ C BM BN SCM SCN
    (fun i => rowIndex PM BM i) (fun j => colIndex PN BN j) (by simp) (by simp)] at hv4
  obtain rfl := Option.some.inj hv4
  -- 5. c_mask
  refine Stmt.TraceSafeListR.cons_intro (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) ?_
  intro s5 h5
  obtain ⟨v5, hv5, rfl⟩ := stepStmtR_assign_inv h5
  rw [evalR_cmask, cmask_eval _ M N BM BN
    (fun i => rowIndex PM BM i) (fun j => colIndex PN BN j) (by simp) (by simp)] at hv5
  obtain rfl := Option.some.inj hv5
  -- 6. the masked store
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun s' _ => Stmt.TraceSafeListR.nil_intro)
  simp only [Stmt.TraceSafeR, MemAccess.SafeAtR, MaskOpt.SafeAtR, Op.SafeAtR.eq_def,
    MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
  refine ⟨trivial, trivial, trivial, ?_⟩
  intro ptrs hptrs i hactive
  rw [evalOpR_ref] at hptrs
  simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
    not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff] at hptrs
  obtain rfl := Option.some.inj hptrs
  obtain ⟨masks, hmasks, hmi⟩ := hactive
  rw [evalOpR_ref] at hmasks
  simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
    not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff] at hmasks
  obtain rfl := Option.some.inj hmasks
  simp only [Bool.and_eq_true, decide_eq_true_eq, rowIndex, colIndex] at hmi
  show SCM * rowIndex PM BM i.1 + SCN * colIndex PN BN i.2.1 < bounds (Region.cast C)
  have h' := hbC (Lane2D.encode (i.1, i.2.1, PUnit.unit))
    (by rw [Lane2D.encode_div, Lane2D.encode_mod]; exact hmi)
  rw [Lane2D.encode_div, Lane2D.encode_mod] at h'
  simpa only [Region.cast_id, rowIndex, colIndex] using h'

set_option maxHeartbeats 4000000 in
/-- **The `TraceSafeR` walk for the whole kernel**, driven by
`Stmt.forRangeTraceSafeR_inv` over the weak `ivdmSafeInv` (reached through
the `forRangeDyn` arm of `TraceSafeR`, whose bound ops resolve to
`0 / numKBlocks / 1`). The three bound groups are the skin's `read1`/`read2`
windows (widened to all lanes, since the load masks are all-true at
exact-multiple `K`) and the `writeMask`-gated `write` window. -/
private theorem ivdm_traceSafeR (R : RoundingModel) (bounds : RegionBounds)
    (A B C : RegionName)
    (M N SAM SAK SBK SBN SCM SCN BM BN BK numKBlocks : Nat) (hBK : 0 < BK)
    (s : BlockState)
    (hbA : ∀ (t : Fin numKBlocks) (l : Fin (BM * BK)),
      (pidM (s.pids 0) N BN * BM + l.val / BK) % M * SAM
        + (t.val * BK + l.val % BK) * SAK < bounds A)
    (hbB : ∀ (t : Fin numKBlocks) (l : Fin (BK * BN)),
      (t.val * BK + l.val / BN) * SBK
        + (pidN (s.pids 0) N BN * BN + l.val % BN) % N * SBN < bounds B)
    (hbC : ∀ l : Fin (BM * BN),
      (pidM (s.pids 0) N BN * BM + l.val / BN < M ∧
        pidN (s.pids 0) N BN * BN + l.val % BN < N) →
      SCM * (pidM (s.pids 0) N BN * BM + l.val / BN)
        + SCN * (pidN (s.pids 0) N BN * BN + l.val % BN) < bounds C) :
    ((iv_dependent_matmul_pre_load_surface A B C M N (BK * numKBlocks) SAM SAK SBK SBN
        SCM SCN BM BN BK).toAlgKernel).TraceSafeR R bounds s := by
  unfold Kernel.TraceSafeR
  rw [ivdm_body_split' A B C M N SAM SAK SBK SBN SCM SCN BM BN BK numKBlocks]
  set PM := pidM (s.pids 0) N BN with hPM
  set PN := pidN (s.pids 0) N BN with hPN
  have hstep : ∀ c s', c < numKBlocks →
      ivdmSafeInv A B PM PN M N BM BN SAM SAK SBK SBN BK s' →
      Stmt.TraceSafeListR R bounds (ivdmLoopBody BM BN BK (BK * numKBlocks) SAK SBK)
        (s'.setReg "k" .nat [] (Tile.scalar c)) ∧
      ∃ s'', stepStmtsR R (ivdmLoopBody BM BN BK (BK * numKBlocks) SAK SBK)
          (s'.setReg "k" .nat [] (Tile.scalar c)) = some s'' ∧
        ivdmSafeInv A B PM PN M N BM BN SAM SAK SBK SBN BK s'' := by
    intro c s' hcx hP
    obtain ⟨hpmx, hpnx, hkx, hzE, hapbx, hbpbx⟩ := hP
    refine ⟨ivdm_bodySafeR R bounds A B PM PN M N BM BN SAM SAK SBK SBN BK numKBlocks
        (BK * numKBlocks) c hcx _ (by simp [hapbx]) (by simp [hbpbx]) (by simp) hbA hbB, ?_⟩
    obtain ⟨s'', hs'', hP''⟩ := ivdm_stepW A B PM PN M N BM BN SAM SAK SBK SBN BK numKBlocks
      c s' hcx ⟨hpmx, hpnx, hkx, hzE, hapbx, hbpbx⟩
    exact ⟨s'', by rw [ivdmBody_castFree]; exact hs'', hP''⟩
  refine Stmt.TraceSafeListR.append_intro _ _ ?_ ?_
  · refine Stmt.TraceSafeListR.of_forall _ _ ?_
    intro st hst s'
    simp only [ivdmPrologue, List.mem_cons, List.not_mem_nil, or_false] at hst
    rcases hst with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
      simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]
  · intro s1 hs1
    obtain ⟨s1x, hpre, hP0⟩ := ivdm_preLoopW A B s M N SAM SAK SBK SBN BM BN BK
    rw [ivdmPrologue_castFree R A B M N SAM SAK SBK SBN BM BN BK s, hpre] at hs1
    obtain rfl := Option.some.inj hs1
    refine Stmt.TraceSafeListR.cons_intro ?_ ?_
    · simp only [Stmt.TraceSafeR]
      refine ⟨by simp [Op.SafeAtR.eq_def], by simp [ivdmKStop, Op.SafeAtR.eq_def],
        by simp [Op.SafeAtR.eq_def], ?_⟩
      rw [ivdmKStop_evalR R s1x BK numKBlocks hBK,
        show evalOpR R (Op.constNat (0 : Nat)) s1x = some (Tile.scalar 0) from by
          simp only [evalOpR.eq_def, evalOp_constNat],
        show evalOpR R (Op.constNat (1 : Nat)) s1x = some (Tile.scalar 1) from by
          simp only [evalOpR.eq_def, evalOp_constNat]]
      show Stmt.forRangeTraceSafeR R bounds "k" 0 numKBlocks 1
        (ivdmLoopBody BM BN BK (BK * numKBlocks) SAK SBK) s1x
      exact Stmt.forRangeTraceSafeR_inv R bounds "k" numKBlocks 1
        (ivdmLoopBody BM BN BK (BK * numKBlocks) SAK SBK)
        (fun _ st => ivdmSafeInv A B PM PN M N BM BN SAM SAK SBK SBN BK st) hstep 0 s1x hP0
    · intro s2 hs2
      obtain ⟨final, sLoop, hLoopStmt, hfinal, hPL⟩ :=
        forRangeAux_inv (idx := "k") (stop := numKBlocks) (step := 1)
          (body := ivdmLoopBody BM BN BK (BK * numKBlocks) SAK SBK)
          (P := fun _ st => ivdmSafeInv A B PM PN M N BM BN SAM SAK SBK SBN BK st)
          (by norm_num)
          (fun c stx hlt hinv => ivdm_stepW A B PM PN M N BM BN SAM SAK SBK SBN BK numKBlocks
            c stx hlt hinv)
          0 s1x hP0
      rw [stepStmtR_ivdmKLoop R BM BN BK SAK SBK numKBlocks hBK,
        stepForRangeAuxR_castFree R _ (ivdmBody_castFree R BM BN BK (BK * numKBlocks) SAK SBK) "k",
        hLoopStmt] at hs2
      obtain rfl := Option.some.inj hs2
      obtain ⟨hpmL, hpnL, -, -, -, -⟩ := hPL
      exact ivdm_tailSafeR R bounds C PM PN M N BM BN SCM SCN sLoop hpmL hpnL hbC

/-- The full IV-dependent (`pre_load`) matmul surface (masked loads, fp16
cast, masked store) sits inside the flat-memory bridge's covered fragment
(`FlattenOk`). -/
theorem iv_dependent_matmul_flattenOk (A B C : RegionName)
    (M N SAM SAK SBK SBN SCM SCN BM BN BK numKBlocks : Nat) :
    ((iv_dependent_matmul_pre_load_surface A B C M N (BK * numKBlocks) SAM SAK SBK SBN
        SCM SCN BM BN BK).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  rw [ivdm_body_split' A B C M N SAM SAK SBK SBN SCM SCN BM BN BK numKBlocks]
  simp [ivdmPrologue, ivdmLoopBody, ivdmPostBody, ivdmKStop, StmtList.FlattenOk,
    Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-! ### IO signature, lane bridges, spec bridge -/

/-- The `A`-stream lane feeding output lane `l` at inner key `e`: the row of
`l` (row-major over the `[BM, BN]` output tile) paired with `e` over the
`[BM, BLOCK_K]` per-step `A`-tile. -/
def aLane (BM BN BK : Nat) (l : Fin (BM * BN)) (e : Fin BK) : Fin (BM * BK) :=
  Lane2D.encode ((Lane2D.decode l).1, e, PUnit.unit)

/-- The `B`-stream lane feeding output lane `l` at inner key `e`: `e` paired
with the column of `l` over the `[BLOCK_K, BN]` per-step `B`-tile. -/
def bLane (BM BN BK : Nat) (l : Fin (BM * BN)) (e : Fin BK) : Fin (BK * BN) :=
  Lane2D.encode (e, (Lane2D.decode l).2.1, PUnit.unit)

/-- **Streaming IO signature** of `iv_dependent_matmul` (the `pre_load`
scheduling mode) on the two-stream fold skin (S1: fold + terminal store).
Step `t` of the K-loop recomputes both pointers from the loop-invariant bases
and reads the `[BM, BLOCK_K]` `A`-tile and the `[BLOCK_K, BN]` `B`-tile;
after the loop one `[BM, BN]` output tile is stored at the **fp16** grid
(`outDType := .fp16` — the kernel's `(accumulator).to(tl.float16)` survives
lowering as an `Op.castFloat` feeding a `Stmt.store .fp16`). The kernel
schedules on a **single** linear `pid` (`program_id(0)`; the skin's `pid₁`
slot is unused), with `pid_m = pid // cdiv N BN` and `pid_n = pid % cdiv N BN`:

* `read1` lane `l = (i, e)` (row-major over `[BM, BLOCK_K]`), step `t`:
  `((pid_m·BM + i) % M)·stride_am + (t·BK + e)·stride_ak` — the base `a_ptr`
  cell shifted by the IV-dependent `t·BK·stride_ak`.
* `read2` lane `l = (e, j)` (row-major over `[BLOCK_K, BN]`), step `t`:
  `(t·BK + e)·stride_bk + ((pid_n·BN + j) % N)·stride_bn` — likewise for
  `b_ptr`.
* `write` lane `l = (i, j)`:
  `stride_cm·(pid_m·BM + i) + stride_cn·(pid_n·BN + j)` — the kernel's
  un-wrapped `c_ptrs` (= `cOffset` in pid form).
* `mask1`/`mask2` transcribe the loads' `offs_k < K − k·BK` windows in the
  per-lane spelling `t·BK + e < K`; `writeMask` transcribes the store's
  `(offs_cm < M) & (offs_cn < N)` boundary mask verbatim. -/
def ivDependentMatmulIO (A B C : RegionName)
    (M N SAM SAK SBK SBN SCM SCN BM BN BK numKBlocks : Nat) :
    StreamMasked2DKernelIO₂ where
  kernel := iv_dependent_matmul_pre_load_surface A B C M N (BK * numKBlocks) SAM SAK SBK SBN
    SCM SCN BM BN BK
  inp1 := A
  inp2 := B
  out := C
  T := numKBlocks
  B1 := BM * BK
  B2 := BK * BN
  C := BM * BN
  outDType := .fp16
  read1 := fun p₀ _ t l =>
    (pidM p₀ N BN * BM + l.val / BK) % M * SAM + (t.val * BK + l.val % BK) * SAK
  read2 := fun p₀ _ t l =>
    (t.val * BK + l.val / BN) * SBK + (pidN p₀ N BN * BN + l.val % BN) % N * SBN
  write := fun p₀ _ l =>
    SCM * (pidM p₀ N BN * BM + l.val / BN) + SCN * (pidN p₀ N BN * BN + l.val % BN)
  mask1 := fun _ _ t l => t.val * BK + l.val % BK < BK * numKBlocks
  mask2 := fun _ _ t l => t.val * BK + l.val / BN < BK * numKBlocks
  writeMask := fun p₀ _ l =>
    pidM p₀ N BN * BM + l.val / BN < M ∧ pidN p₀ N BN * BN + l.val % BN < N

/-- At the exact-multiple contraction `K = BK · numKBlocks`, every in-tile
inner key is inside the kernel's `offs_k < K − k·BK` load window. -/
private theorem stream_mask_lt (BK numKBlocks : Nat)
    (t : Fin numKBlocks) {e : Nat} (he : e < BK) : t.val * BK + e < BK * numKBlocks := by
  have h1 : (t.val + 1) * BK ≤ numKBlocks * BK :=
    Nat.mul_le_mul_right BK (Nat.succ_le_of_lt t.isLt)
  have h2 : t.val * BK + e < (t.val + 1) * BK := by
    rw [Nat.succ_mul]; omega
  calc t.val * BK + e < (t.val + 1) * BK := h2
    _ ≤ numKBlocks * BK := h1
    _ = BK * numKBlocks := Nat.mul_comm _ _

/-- Under the two stream pins, `matmulSpec` at the decoded output lane **is**
the skin-level double fold `∑ t, ∑ e, xs · ys` (`gemmSum_blocks` +
address-identity of the windows with the recomputed pointer shapes; the pins'
`mask1`/`mask2` guards are discharged by `stream_mask_lt`). -/
private theorem matmulSpec_eq_streamSum (A B : RegionName) (s₀ : BlockState)
    (M N BM BN SAM SAK SBK SBN BK numKBlocks : Nat)
    (xs : Fin numKBlocks → Fin (BM * BK) → ℝ) (ys : Fin numKBlocks → Fin (BK * BN) → ℝ)
    (hx : ∀ (t : Fin numKBlocks) (l : Fin (BM * BK)),
      t.val * BK + l.val % BK < BK * numKBlocks →
      s₀.readMem A ((pidM (s₀.pids 0) N BN * BM + l.val / BK) % M * SAM
          + (t.val * BK + l.val % BK) * SAK)
        = xs t l)
    (hy : ∀ (t : Fin numKBlocks) (l : Fin (BK * BN)),
      t.val * BK + l.val / BN < BK * numKBlocks →
      s₀.readMem B ((t.val * BK + l.val / BN) * SBK
          + (pidN (s₀.pids 0) N BN * BN + l.val % BN) % N * SBN)
        = ys t l)
    (l : Fin (BM * BN)) :
    matmulSpec s₀ A B (pidM (s₀.pids 0) N BN) (pidN (s₀.pids 0) N BN)
        BM BN M N SAM SAK SBK SBN BK numKBlocks (Lane2D.decode l).1 (Lane2D.decode l).2.1
      = ∑ t : Fin numKBlocks, ∑ e : Fin BK,
          xs t (aLane BM BN BK l e) * ys t (bLane BM BN BK l e) := by
  unfold matmulSpec
  rw [Nat.mul_comm BK numKBlocks, gemmSum_blocks]
  refine Finset.sum_congr rfl fun t _ => Finset.sum_congr rfl fun e _ => ?_
  have hxa : aElem s₀ A (pidM (s₀.pids 0) N BN) BM M SAM SAK (Lane2D.decode l).1
      (t.val * BK + e.val) = xs t (aLane BM BN BK l e) := by
    rw [← hx t (aLane BM BN BK l e)
        (by simpa [aLane, Lane2D.encode_mod] using stream_mask_lt BK numKBlocks t e.isLt)]
    simp only [aLane, Lane2D.encode_div, Lane2D.encode_mod]
    unfold aElem rowIndex
    rfl
  have hyb : bElem s₀ B (pidN (s₀.pids 0) N BN) BN N SBK SBN (Lane2D.decode l).2.1
      (t.val * BK + e.val) = ys t (bLane BM BN BK l e) := by
    rw [← hy t (bLane BM BN BK l e)
        (by simpa [bLane, Lane2D.encode_div] using stream_mask_lt BK numKBlocks t e.isLt)]
    simp only [bLane, Lane2D.encode_div, Lane2D.encode_mod]
    unfold bElem colIndex
    rfl
  rw [hxa, hyb]

/-! ### The headline -/

set_option maxHeartbeats 4000000 in
/-- **The `⊨[R]` streaming headline (wave-5 S1 fold genre).** For every
rounding model `R`, the faithful `iv_dependent_matmul` (`pre_load` mode)
surface implements, on its `StreamMasked2DKernelIO₂` signature, the **ideal ℝ
GEMM fold** over the streamed tiles: output lane `l = (i, j)` holds
`∑ t, ∑ e, A-tile[t](i,e) · B-tile[t](e,j)` — the spec `f` is exact real
arithmetic, and the single boundary quantization is carried by the skin's
readback contract (`readMemAs .fp16` holds
`fp16.ofReal (R.round .fp16 (f …))`), where the kernel's two rounding events
(the `.to(tl.float16)` cast and the fp16-typed masked store) collapse to one
`R.round .fp16` by the defining `round_idem`.

Layer map: the 10-statement prologue and the whole dynamic K-loop (the
IV-dependent pointer recomputes plus the masked loads) are cast-free, so
under `execR R` they collapse verbatim onto the exact stepper and the proven
`ivdm_preLoop` / `ivdm_step` / `forRangeAux_inv` stack above is reused
unchanged; only the 6-statement store tail is re-proved on the `R` side
(`ivdm_postLoopR`).

Every hypothesis is truth-forced:

* `hBK : 0 < BLOCK_K` — the loop's trip count is `tl.cdiv(K, BLOCK_K)`,
  which equals `numKBlocks` at `K = BLOCK_K · numKBlocks` only for a nonzero
  block size; at `BLOCK_K = 0` the block-index arithmetic `l / BLOCK_K` is
  meaningless. It holds for every real launch (`tl.arange(0, 0)` is not a
  tile).
* `hcn : stride_cn = 1` and `hbnle : BLOCK_N ≤ stride_cm` —
  output-offset injectivity (`rowMajor2D_inj`, exactly the exact headline's
  precondition): the column stride is the unit stride and the column-block
  width fits the row stride, so distinct output lanes hit distinct
  addresses; with colliding lanes the per-lane readback would be
  last-writer-wins and the statement false. Both hold for every valid
  row-major tiling.

Note this `⊨[R]` face is *stronger* than the exact headline
`iv_dependent_matmul_closed_form_correct` in one respect: that one assumes
the store mask is all-true (`hmlt`/`hnlt`), while the skin carries the
genuine `(offs_cm < M) & (offs_cn < N)` boundary mask in `writeMask`, so no
in-bounds-tile hypothesis is needed here. Both faces are kept per the
rounding-as-default doctrine. -/
specification iv_dependent_matmul_io_correctness (R : RoundingModel)
    (A B C : RegionName)
    (M N SAM SAK SBK SBN SCM SCN BM BN BK numKBlocks : Nat)
    (hBK : 0 < BK) (hcn : SCN = 1) (hbnle : BN ≤ SCM) :
    ivDependentMatmulIO A B C M N SAM SAK SBK SBN SCM SCN BM BN BK numKBlocks
      ⊨[R] fun _ _ xs ys l =>
        ∑ t : Fin numKBlocks, ∑ e : Fin BK,
          xs t (aLane BM BN BK l e) * ys t (bLane BM BN BK l e) := by
  subst hcn
  refine StreamMasked2DKernelIO₂.ImplementsR.intro _ ?_ ?_ ?_
  · exact iv_dependent_matmul_flattenOk A B C M N SAM SAK SBK SBN SCM 1 BM BN BK numKBlocks
  · -- safety walk
    intro bounds s xs ys _hx _hy hbr1 hbr2 hbw
    simp only [ivDependentMatmulIO] at hbr1 hbr2 hbw ⊢
    refine ivdm_traceSafeR R bounds A B C M N SAM SAK SBK SBN SCM 1 BM BN BK numKBlocks
      hBK s ?_ ?_ ?_
    · intro t l
      exact hbr1 t l (stream_mask_lt BK numKBlocks t (Nat.mod_lt _ hBK))
    · intro t l
      have hdiv : l.val / BN < BK :=
        Nat.div_lt_of_lt_mul (lt_of_lt_of_eq l.isLt (Nat.mul_comm BK BN))
      exact hbr2 t l (stream_mask_lt BK numKBlocks t hdiv)
    · intro l hl
      exact hbw l hl
  · -- the rounded Hoare triple
    intro s₀ xs ys hundef hx hy
    simp only [ivDependentMatmulIO] at hx hy ⊢
    have hundef' : ∀ rg o, s₀.undef rg o = 0 := fun rg o => by rw [hundef]
    set PM := pidM (s₀.pids 0) N BN with hPM
    set PN := pidN (s₀.pids 0) N BN with hPN
    have hInj : Function.Injective (cOffset s₀ PM PN BM BN SCM 1) := by
      have heq : cOffset s₀ PM PN BM BN SCM 1
          = fun idx : TileIndex [BM, BN] =>
              (SCM * (PM * BM) + PN * BN) + idx.1.val * SCM + idx.2.1.val := by
        funext idx; simp only [cOffset, rowIndex, colIndex]; ring
      rw [heq]; exact rowMajor2D_inj _ SCM hbnle
    obtain ⟨s1, hpre, hP0⟩ := ivdm_preLoop A B C s₀ M N (BK * numKBlocks) SAM SAK SBK SBN
      SCM 1 BM BN BK numKBlocks hundef'
    obtain ⟨sLoop, hLoopStmt, hPLoop⟩ :=
      ivdm_loop A B s₀ PM PN M N BM BN SAM SAK SBK SBN BK numKBlocks hBK s1 hP0
    have hmem0 : sLoop.mem = s₀.mem := by
      have h := hPLoop
      simp only [ivdmInvariant] at h
      exact h.2.2.2.2.2.2.2.2.2.2.2
    obtain ⟨sfin, hTailR, hval, hframe⟩ :=
      ivdm_postLoopR R A B C s₀ PM PN M N BM BN SAM SAK SBK SBN SCM 1 BK numKBlocks
        hInj sLoop hPLoop
    have hLoopR : stepStmtR R (Stmt.forRangeDyn "k" (Op.constNat 0)
        (ivdmKStop (BK * numKBlocks) BK) (Op.constNat 1)
        (ivdmLoopBody BM BN BK (BK * numKBlocks) SAK SBK)) s1 = some sLoop := by
      rw [ivdmKLoop_castFree R BM BN BK SAK SBK numKBlocks hBK]
      exact hLoopStmt
    have hpre' : stepStmts (ivdmPrologue A B M N SAM SAK SBK SBN BM BN BK) s₀ = some s1 := by
      rw [← ivdm_take10_eq A B C M N SAM SAK SBK SBN SCM 1 BM BN BK numKBlocks]
      exact hpre
    refine ⟨sfin, ?_, ?_, ?_⟩
    · show execR R (iv_dependent_matmul_pre_load_surface A B C M N (BK * numKBlocks)
          SAM SAK SBK SBN SCM 1 BM BN BK).toAlgKernel s₀ = some sfin
      unfold execR
      rw [ivdm_body_split' A B C M N SAM SAK SBK SBN SCM 1 BM BN BK numKBlocks,
        stepStmtsR_append,
        ivdmPrologue_castFree R A B M N SAM SAK SBK SBN BM BN BK s₀, hpre',
        Option.bind_some, stepStmtsR_cons_some hLoopR]
      exact hTailR
    · intro l hl
      have hact : rowIndex PM BM (Lane2D.decode l).1 < M
          ∧ colIndex PN BN (Lane2D.decode l).2.1 < N := by
        obtain ⟨h1, h2⟩ := hl
        exact ⟨by simpa [rowIndex] using h1, by simpa [colIndex] using h2⟩
      have hcell := hval (Lane2D.decode l)
      rw [if_pos hact] at hcell
      have haddr : SCM * (PM * BM + l.val / BN) + 1 * (PN * BN + l.val % BN)
          = cOffset s₀ PM PN BM BN SCM 1 (Lane2D.decode l) := rfl
      rw [haddr, BlockState.readMemAs_fp16_of_cell hcell,
        matmulSpec_eq_streamSum A B s₀ M N BM BN SAM SAK SBK SBN BK numKBlocks xs ys hx hy l]
    · intro r o hcond
      have hcond' : r ≠ C ∨ ∀ idx : TileIndex [BM, BN],
          (rowIndex PM BM idx.1 < M ∧ colIndex PN BN idx.2.1 < N) →
            o ≠ cOffset s₀ PM PN BM BN SCM 1 idx := by
        rcases hcond with hne | hno
        · exact Or.inl hne
        · refine Or.inr fun idx hidx => ?_
          have h := hno (Lane2D.encode idx) (by
            rw [Lane2D.encode_div, Lane2D.encode_mod]
            simpa [rowIndex, colIndex] using hidx)
          rw [Lane2D.encode_div, Lane2D.encode_mod] at h
          exact h
      rw [hframe r o hcond', hmem0]

end VeriTile.Bench.TritonBenchG.IvDependentMatmul
