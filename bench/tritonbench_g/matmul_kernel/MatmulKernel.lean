import VeriTile.Triton

/-!
# `matmul_kernel` — closed-form GEMM correctness

`matmul_kernel.py`'s `matmul_kernel` is a tiled GEMM with a 2-D program id:
program `(pid_m, pid_n)` accumulates a `BLOCK_SIZE_M × BLOCK_SIZE_N` output tile
by looping over K with the fused `accumulator = tl.dot(a, b, accumulator)`
reduction, casts the accumulator to `float16`, and stores the tile into `c_ptr`.

This file proves the **full K-loop** correct against a genuine mathematical
matrix product: every output cell `C[i,j]` of the computed tile equals
`fp16( Σ_{k < K} A[i,k] · B[k,j] )` over `ℝ`, where `K = BLOCK_SIZE_K · numKBlocks`
is the contracted dimension. This is NOT the kernel's own emitted value — the
real-valued `Σ_k A·B` GEMM reference is derived independently from the loaded
`A`/`B` tiles, and the final `tl.cast(..., tl.float16)` is applied to that
reference value.

## Proof architecture

```
matmul_kernel_closed_form_correct                 ← TOP THEOREM (ComputeCorrect.Realizes_without_Rounding)
  └─ matmul_kernel_exec_closed_form               ← exec-side closed form (every cell = fp16(∑_k A·B))
       ├─ preLoop      (P 0: accumulator = 0, pointers seeded)
       ├─ matmul_step         (one K-block: accumulator += dot advances the partial sum)
       ├─ matmul_postLoop     (cast to fp16 + final store = the closed form)
       └─ forRange_inv        (loop-invariant principle, drives the K-loop)
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); the dot reduction is
accumulated over `ℝ` and the modeled `tl.cast(..., fp16)` is the placeholder
`FloatDType.real.cast .fp16`.
The host launch (grid, 2-D program scheduling) is the trusted boundary; the
per-program statement is universally quantified over `s`, covering every program
of the grid. The layout contract is the kernel's hard-coded `4096×4096`
row-major strides: `a[i,k]` at `A + offs_am(i)·4096 + k`, `b[k,j]` at
`B + k·4096 + offs_bn(j)`, `c[i,j]` at `C + 4096·offs_cm(i) + offs_cn(j)`, with
`offs_am(i) = (pid_m·BM + i) % 4096`, `offs_bn(j) = (pid_n·BN + j) % 4096`,
exactly as the kernel's pointer arithmetic constructs them.

## Translation-surface blocker

Translation-surface blocker: the Python kernel's in-body constants
(`M, N, K = 4096, 4096, 4096` and the six stride literals) and the loop trip
count `tl.cdiv(K, BLOCK_SIZE_K)` are supplied as antiquoted binders
(`K = BLOCK_SIZE_K · numKBlocks`, trip count `numKBlocks`), so the in-body
constant assignments and the `tl.cdiv` call do not appear as surface
statements. The textual py↔lean scans in `bench/audit_tritonbench_g.sh`
exempt this port on this marker (registered in `proof_blockers.md`).
-/

namespace VeriTile.Bench.TritonBenchG.MatmulKernel

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Faithful transcription of `matmul_kernel.py`'s `matmul_kernel`.

The Python kernel hard-codes `M = N = K = 4096` and contiguous strides; here `K`
is presented as `BLOCK_SIZE_K · numKBlocks` so the loop trip count
`cdiv(K, BLOCK_SIZE_K) = numKBlocks` is exact, and the loop bound is supplied as
the antiquoted `numKBlocks` (`= tl.cdiv(4096, BLOCK_SIZE_K)` for the modeled
block shapes). All other surface structure — the `% 4096` index wrap, the fused
`tl.dot(a, b, accumulator)`, the `float16` cast, and the row-major store — is
transcribed verbatim. -/
def matmul_kernel_surface
    (C A B : RegionName) (BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K numKBlocks : Nat) :
    ComputeKernel := triton {
  pid_m = tl.program_id(axis=0)
  pid_n = tl.program_id(axis=1)
  offs_am = (pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))) % $(4096)
  offs_bn = (pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))) % $(4096)
  offs_k = tl.arange(0, $(BLOCK_SIZE_K))
  a_ptrs = A + (offs_am[:, None] * $(4096) + offs_k[None, :] * $(1))
  b_ptrs = B + (offs_k[:, None] * $(4096) + offs_bn[None, :] * $(1))
  accumulator = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
  for k in range($(0), $(numKBlocks), $(1)) {
    a = tl.load(a_ptrs)
    b = tl.load(b_ptrs)
    accumulator = tl.dot(a, b, accumulator)
    a_ptrs += $(BLOCK_SIZE_K) * $(1)
    b_ptrs += $(BLOCK_SIZE_K) * $(4096)
  }
  c = tl.cast(accumulator, tl.float16)
  offs_cm = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_cn = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  c_ptrs = C + $(4096) * offs_cm[:, None] + $(1) * offs_cn[None, :]
  tl.store(c_ptrs, c)
}

/-- The full `matmul_kernel` surface lowers to the algorithm layer. -/
theorem matmul_kernel_surface_toAlgorithm_supported
    (C A B : RegionName) (BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K numKBlocks : Nat) :
    ∃ alg, (matmul_kernel_surface C A B BLOCK_SIZE_M BLOCK_SIZE_N
      BLOCK_SIZE_K numKBlocks).toAlgorithm? = Except.ok alg := by
  simp [matmul_kernel_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

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

/-- No-mask `.ptr` load: reads `readMem` at each pointer (clean `undef`). -/
theorem load_ptr_none_real {shape : TileShape}
    (ptrOp : Op .ptr shape) (s : BlockState) (ptrs : Tile .ptr shape)
    (hp : evalOp ptrOp s = some ptrs) :
    evalOp (.load .real (.ptr ptrOp) .none) s
      = some ⟨fun i => some (s.readMem (ptrs.data i).1 (ptrs.data i).2)⟩ := by
  simp only [evalOp, hp]
  refine congrArg some ?_
  ext i
  simp [BlockState.readMemValue_real]

/-- `a_ptrs` eval: cell `(i,e) = (A, offs_am i · 4096 + offs_k e · 1)`. -/
theorem aptrs_eval (s : BlockState) (A : RegionName) (M K : Nat) (gm : Fin M → Nat)
    (hm : s.regs .nat [M] "offs_am" = some (Tile.vec gm))
    (hk : s.regs .nat [K] "offs_k" = some (Tile.vec (fun e : Fin K => e.val))) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase A)
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_am")) (Op.constNat 4096))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [K] "offs_k")) (Op.constNat 1)))) s
      = some (⟨fun idx : TileIndex [M, K] => (A.cast, gm idx.1 * 4096 + idx.2.1.val * 1)⟩ : Tile .ptr [M, K]) := by
  rw [evalOp_ptrAdd, evalOp_ptrBase]
  simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref, evalOp_expandDim_one_nat,
    evalOp_expandDim_zero_nat, hm, hk, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec]
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec, NumericDType.add, NumericDType.mul]

/-- `b_ptrs` eval: cell `(e,j) = (B, offs_k e · 4096 + offs_bn j · 1)`. -/
theorem bptrs_eval (s : BlockState) (B : RegionName) (K N : Nat) (gn : Fin N → Nat)
    (hk : s.regs .nat [K] "offs_k" = some (Tile.vec (fun e : Fin K => e.val)))
    (hn : s.regs .nat [N] "offs_bn" = some (Tile.vec gn)) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase B)
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [K] "offs_k")) (Op.constNat 4096))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [N] "offs_bn")) (Op.constNat 1)))) s
      = some (⟨fun idx : TileIndex [K, N] => (B.cast, idx.1.val * 4096 + gn idx.2.1 * 1)⟩ : Tile .ptr [K, N]) := by
  rw [evalOp_ptrAdd, evalOp_ptrBase]
  simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref, evalOp_expandDim_one_nat,
    evalOp_expandDim_zero_nat, hk, hn, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec]
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec, NumericDType.add, NumericDType.mul]

/-- `c_ptrs` eval: cell `(i,j) = (C, 4096 · offs_cm i + 1 · offs_cn j)`
(strides on the **left** of the products). -/
theorem cptrs_eval (s : BlockState) (C : RegionName) (M N : Nat) (gm : Fin M → Nat) (gn : Fin N → Nat)
    (hm : s.regs .nat [M] "offs_cm" = some (Tile.vec gm))
    (hn : s.regs .nat [N] "offs_cn" = some (Tile.vec gn)) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase C)
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarL (Op.constNat 4096) (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_cm")))
        (Op.mul .nat Broadcast.scalarL (Op.constNat 1) (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [N] "offs_cn"))))) s
      = some (⟨fun idx : TileIndex [M, N] => (C.cast, 4096 * gm idx.1 + 1 * gn idx.2.1)⟩ : Tile .ptr [M, N]) := by
  rw [evalOp_ptrAdd, evalOp_ptrBase]
  simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref, evalOp_expandDim_one_nat,
    evalOp_expandDim_zero_nat, hm, hn, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec]
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec, NumericDType.add, NumericDType.mul]

/-- `accumulator` init eval: `tl.zeros` → the all-`0` tile. -/
theorem acc_init_eval (s : BlockState) (M N : Nat) :
    evalOp (Op.full [M, N] (Op.const 0)) s
      = some (⟨fun _ : TileIndex [M, N] => some (0 : ℝ)⟩ : Tile .real [M, N]) := by
  simp [evalOp_full, evalOp_const, Option.bind]

/-- `a_ptrs += BLOCK_K · 1` eval. -/
theorem aptr_adv_eval (s : BlockState) (M K BK : Nat) (ap : Tile .ptr [M, K])
    (ha : s.regs .ptr [M, K] "a_ptrs" = some ap) :
    evalOp (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [M, K] "a_ptrs")
      (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat 1))) s
      = some (Tile.ptrAdd Broadcast.scalarR ap (Tile.scalar (BK * 1))) := by
  rw [evalOp_ptrAdd]
  simp [evalOp_ref, ha, evalOp_mul, evalOp_constNat, NumericDType.mul, Tile.bop]

/-- `b_ptrs += BLOCK_K · 4096` eval. -/
theorem bptr_adv_eval (s : BlockState) (K N BK : Nat) (bp : Tile .ptr [K, N])
    (hb : s.regs .ptr [K, N] "b_ptrs" = some bp) :
    evalOp (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [K, N] "b_ptrs")
      (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat 4096))) s
      = some (Tile.ptrAdd Broadcast.scalarR bp (Tile.scalar (BK * 4096))) := by
  rw [evalOp_ptrAdd]
  simp [evalOp_ref, hb, evalOp_mul, evalOp_constNat, NumericDType.mul, Tile.bop]

/-- **`accumulator = tl.dot(a, b, accumulator)` statement eval** (fused form
`dot a b + accumulator`). -/
theorem accdot_op_eval (M K N : Nat) (st : BlockState)
    (zt : Tile .real [M, N]) (xt : Tile .real [M, K]) (yt : Tile .real [K, N])
    (hz : st.regs .real [M, N] "accumulator" = some zt)
    (hx : st.regs .real [M, K] "a" = some xt)
    (hy : st.regs .real [K, N] "b" = some yt) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.dot (batch := []) (Op.ref .real [M, K] "a") (Op.ref .real [K, N] "b"))
        (Op.ref .real [M, N] "accumulator")) st
      = some (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          (Tile.dot [] xt yt) zt) := by
  have hd : evalOp (Op.dot (batch := []) (Op.ref .real [M, K] "a")
        (Op.ref .real [K, N] "b")) st = some (Tile.dot [] xt yt) := by
    rw [evalOp_dot]; simp [hx, hy]
  rw [evalOp_add]
  simp only [evalOp_ref, hz, bind, Option.bind_some]
  erw [hd]
  rfl

/-- `dot + accumulator` lane `(i,j)`: `some (dv + zv)`. -/
theorem dotadd_eval (M N : Nat) (dt zt : Tile .real [M, N]) (i : Fin M) (j : Fin N) (dv zv : ℝ)
    (hd : dt.data (i, j, PUnit.unit) = some dv) (hz : zt.data (i, j, PUnit.unit) = some zv) :
    (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) dt zt).data
        (i, j, PUnit.unit) = some (dv + zv) := by
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, hd, hz, NumericDType.add,
    WithBot.realAdd, Option.map₂, Option.bind, Option.map]

/-! ## GEMM closed-form spec -/

/-- Global output row of tile lane `i` for program `(pid_m, pid_n)`:
`pid_m · BLOCK_M + i`, **before** the `% 4096` wrap. -/
def rowGlobal (s : BlockState) (BM : Nat) (i : Fin BM) : Nat :=
  s.pids 0 * BM + i.val

/-- Global output column of tile lane `j`: `pid_n · BLOCK_N + j`, before wrap. -/
def colGlobal (s : BlockState) (BN : Nat) (j : Fin BN) : Nat :=
  s.pids 1 * BN + j.val

/-- The `% 4096`-wrapped A-row index of tile lane `i` (the kernel's `offs_am`). -/
def rowIndex (s : BlockState) (BM : Nat) (i : Fin BM) : Nat :=
  rowGlobal s BM i % 4096

/-- The `% 4096`-wrapped B-column index of tile lane `j` (the kernel's `offs_bn`). -/
def colIndex (s : BlockState) (BN : Nat) (j : Fin BN) : Nat :=
  colGlobal s BN j % 4096

/-- `A[i, k] = readMem A (offs_am i · 4096 + k)` (kernel's row-major A layout). -/
noncomputable def aElem (s : BlockState) (A : RegionName) (BM : Nat)
    (i : Fin BM) (k : Nat) : ℝ :=
  s.readMem A (rowIndex s BM i * 4096 + k)

/-- `B[k, j] = readMem B (k · 4096 + offs_bn j)` (kernel's row-major B layout). -/
noncomputable def bElem (s : BlockState) (B : RegionName) (BN : Nat)
    (j : Fin BN) (k : Nat) : ℝ :=
  s.readMem B (k * 4096 + colIndex s BN j)

/-- **Genuine GEMM spec** (over ℝ): `C[i,j] = Σ_{k < BLOCK_K·numKBlocks} A[i,k] · B[k,j]`,
an instance of the shared `gemmSum` (`Math.Matmul`) with this kernel's `A`/`B`
layout accessors. -/
noncomputable def matmulSpec (s : BlockState) (A B : RegionName)
    (BM BN BLOCK_K numKBlocks : Nat) (i : Fin BM) (j : Fin BN) : ℝ :=
  gemmSum (aElem s A BM i) (bElem s B BN j) (BLOCK_K * numKBlocks)

/-- Partial GEMM accumulator after `c` K-blocks: `Σ_{k < c·BLOCK_K} A·B`. -/
noncomputable def accPartial (s : BlockState) (A B : RegionName)
    (BM BN BLOCK_K : Nat) (i : Fin BM) (j : Fin BN) (c : Nat) : ℝ :=
  gemmSum (aElem s A BM i) (bElem s B BN j) (c * BLOCK_K)

/-- One-block step of the partial accumulator: the new block's dot is over the
`BLOCK_K` keys `c·BLOCK_K + e` (the shared `gemmSum_blockSucc`). -/
theorem accPartial_succ (s : BlockState) (A B : RegionName)
    (BM BN BLOCK_K : Nat) (i : Fin BM) (j : Fin BN) (c : Nat) :
    accPartial s A B BM BN BLOCK_K i j (c + 1)
      = accPartial s A B BM BN BLOCK_K i j c
        + (Finset.univ.sum fun e : Fin BLOCK_K =>
            aElem s A BM i (c * BLOCK_K + e.val)
              * bElem s B BN j (c * BLOCK_K + e.val)) :=
  gemmSum_blockSucc (aElem s A BM i) (bElem s B BN j) BLOCK_K c

/-! ## Body decomposition (prefix ++ for-loop ++ cast/offs/store) -/

/-- The 5-statement K-loop body, transcribed. -/
def matmulLoopBody (M N BK : Nat) : List Stmt :=
  [ Stmt.assign .real [M, BK] "a"
      (Op.load .real (.ptr (Op.ref .ptr [M, BK] "a_ptrs")) .none),
    Stmt.assign .real [BK, N] "b"
      (Op.load .real (.ptr (Op.ref .ptr [BK, N] "b_ptrs")) .none),
    Stmt.assign .real [M, N] "accumulator"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.dot (batch := []) (Op.ref .real [M, BK] "a") (Op.ref .real [BK, N] "b"))
        (Op.ref .real [M, N] "accumulator")),
    Stmt.assign .ptr [M, BK] "a_ptrs"
      (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [M, BK] "a_ptrs")
        (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat 1))),
    Stmt.assign .ptr [BK, N] "b_ptrs"
      (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BK, N] "b_ptrs")
        (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat 4096))) ]

/-- The 5-statement post-loop tail: cast to fp16, the two `offs_c*` index
vectors, the `c_ptrs` pointer chunk, and the unmasked store. -/
def matmulStoreTail (C : RegionName) (M N BM BN : Nat) : List Stmt :=
  [ Stmt.assign .fp16 [M, N] "c"
      (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref .real [M, N] "accumulator")),
    Stmt.assign .nat [M] "offs_cm"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat BM))
        (Op.arange M)),
    Stmt.assign .nat [N] "offs_cn"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_n") (Op.constNat BN))
        (Op.arange N)),
    Stmt.assign .ptr [M, N] "c_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase C)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarL (Op.constNat 4096) (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_cm")))
          (Op.mul .nat Broadcast.scalarL (Op.constNat 1) (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [N] "offs_cn"))))),
    Stmt.store .fp16 [M, N] (.ptr (Op.ref .ptr [M, N] "c_ptrs"))
      (Op.ref .fp16 [M, N] "c") .none ]

/-- Body decomposition: prefix (8) ++ [for-loop] ++ store-tail (5). By `rfl`. -/
theorem matmul_body_split (C A B : RegionName) (BM BN BLOCK_K numKBlocks : Nat) :
    (matmul_kernel_surface C A B BM BN BLOCK_K numKBlocks).toAlgKernel.body
      = (matmul_kernel_surface C A B BM BN BLOCK_K numKBlocks).toAlgKernel.body.take 8
        ++ (Stmt.forRange "k" 0 numKBlocks 1 (matmulLoopBody BM BN BLOCK_K)
            :: matmulStoreTail C BM BN BM BN) := by
  rfl

/-! ## Loop invariant -/

/-- **Loop invariant** (counter `c = block index`, step `1`).

After `c` K-blocks: program ids and `mem`/`undef` fixed; the `offs_am` /
`offs_bn` / `offs_k` / `pid_m` / `pid_n` registers seeded; `accumulator` equals
the partial GEMM accumulator `accPartial … c`; and `a_ptrs` / `b_ptrs` advanced
by `c` blocks. -/
noncomputable def matmulInvariant
    (A B : RegionName) (s0 : BlockState) (BM BN BLOCK_K numKBlocks : Nat)
    (c : Nat) (s : BlockState) : Prop :=
  s.pids = s0.pids ∧ c ≤ numKBlocks ∧
  (s.regs .nat [] "pid_m" = some (Tile.scalar (s0.pids 0))) ∧
  (s.regs .nat [] "pid_n" = some (Tile.scalar (s0.pids 1))) ∧
  (s.regs .real [BM, BN] "accumulator" = some ⟨fun idx : TileIndex [BM, BN] =>
      some (accPartial s0 A B BM BN BLOCK_K idx.1 idx.2.1 c)⟩) ∧
  (s.regs .nat [BM] "offs_am" = some (Tile.vec (fun r : Fin BM => rowIndex s0 BM r))) ∧
  (s.regs .nat [BN] "offs_bn" = some (Tile.vec (fun j : Fin BN => colIndex s0 BN j))) ∧
  (s.regs .nat [BLOCK_K] "offs_k" = some (Tile.vec (fun e : Fin BLOCK_K => e.val))) ∧
  (s.regs .ptr [BM, BLOCK_K] "a_ptrs" = some ⟨fun idx : TileIndex [BM, BLOCK_K] =>
      (A.cast, rowIndex s0 BM idx.1 * 4096 + idx.2.1.val * 1 + c * BLOCK_K)⟩) ∧
  (s.regs .ptr [BLOCK_K, BN] "b_ptrs" = some ⟨fun idx : TileIndex [BLOCK_K, BN] =>
      (B.cast, idx.1.val * 4096 + colIndex s0 BN idx.2.1 * 1 + c * BLOCK_K * 4096)⟩) ∧
  (∀ rg o, s.undef rg o = 0) ∧ (s.mem = s0.mem)

/-- **preLoop scalars** (statements 0–4): `pid_m`, `pid_n`, and the 3 index
vectors `offs_am` / `offs_bn` / `offs_k`. -/
theorem preLoop_scalars (s : BlockState) (M N K : Nat) :
    ∃ s5, stepStmts [ Stmt.assign .nat [] "pid_m" (Op.programId 0),
          Stmt.assign .nat [] "pid_n" (Op.programId 1),
          Stmt.assign .nat [M] "offs_am"
            (Op.mod .nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat M))
                (Op.arange M))
              (Op.constNat 4096)),
          Stmt.assign .nat [N] "offs_bn"
            (Op.mod .nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_n") (Op.constNat N))
                (Op.arange N))
              (Op.constNat 4096)),
          Stmt.assign .nat [K] "offs_k" (Op.arange K) ] s = some s5
      ∧ s5.pids = s.pids
      ∧ s5.regs .nat [] "pid_m" = some (Tile.scalar (s.pids 0))
      ∧ s5.regs .nat [] "pid_n" = some (Tile.scalar (s.pids 1))
      ∧ s5.regs .nat [M] "offs_am" = some (Tile.vec (fun i : Fin M => rowIndex s M i))
      ∧ s5.regs .nat [N] "offs_bn" = some (Tile.vec (fun j : Fin N => colIndex s N j))
      ∧ s5.regs .nat [K] "offs_k" = some (Tile.vec (fun e : Fin K => e.val))
      ∧ s5.undef = s.undef
      ∧ s5.mem = s.mem := by
  simp only [rowIndex, colIndex, rowGlobal, colGlobal]
  simp [stepStmts, stepStmt, evalOp_mod, Option.bind, BlockState.setReg,
    Tile.bop, Tile.vec, NumericDType.add, NumericDType.mul, IntegralDType.mod]

set_option maxHeartbeats 1000000 in
/-- **preLoop** (statements 0–7): from a clean input state (`undef = 0`), the
prologue steps to a state satisfying `matmulInvariant … 0` — the base case
(`accumulator = 0`, pointers seeded). -/
theorem preLoop (C A B : RegionName) (s : BlockState)
    (BM BN BLOCK_K numKBlocks : Nat) (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ s', stepStmts ((matmul_kernel_surface C A B BM BN BLOCK_K numKBlocks).toAlgKernel.body.take 8) s = some s'
      ∧ matmulInvariant A B s BM BN BLOCK_K numKBlocks 0 s' := by
  obtain ⟨s5, h5, hpids, hpm, hpn, hm, hn, hk, huf, hmem⟩ := preLoop_scalars s BM BN BLOCK_K
  rw [show ((matmul_kernel_surface C A B BM BN BLOCK_K numKBlocks).toAlgKernel.body.take 8)
      = [ Stmt.assign .nat [] "pid_m" (Op.programId 0),
          Stmt.assign .nat [] "pid_n" (Op.programId 1),
          Stmt.assign .nat [BM] "offs_am"
            (Op.mod .nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat BM))
                (Op.arange BM))
              (Op.constNat 4096)),
          Stmt.assign .nat [BN] "offs_bn"
            (Op.mod .nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_n") (Op.constNat BN))
                (Op.arange BN))
              (Op.constNat 4096)),
          Stmt.assign .nat [BLOCK_K] "offs_k" (Op.arange BLOCK_K) ]
      ++ [ Stmt.assign .ptr [BM, BLOCK_K] "a_ptrs"
            (Op.ptrAdd Broadcast.scalarL (Op.ptrBase A)
              (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
                (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_am")) (Op.constNat 4096))
                (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_K] "offs_k")) (Op.constNat 1)))),
          Stmt.assign .ptr [BLOCK_K, BN] "b_ptrs"
            (Op.ptrAdd Broadcast.scalarL (Op.ptrBase B)
              (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
                (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_K] "offs_k")) (Op.constNat 4096))
                (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_bn")) (Op.constNat 1)))),
          Stmt.assign .real [BM, BN] "accumulator" (Op.full [BM, BN] (Op.const 0)) ] from rfl,
    stepStmts.append_some h5,
    stepStmts.cons_some (stepStmt_assign_eq_some
      (aptrs_eval s5 A BM BLOCK_K (fun i => rowIndex s BM i) (by simpa using hm) (by simpa using hk))),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (bptrs_eval _ B BLOCK_K BN (fun j => colIndex s BN j) (by simp [hk]) (by simp [hn]))),
    stepStmts.cons_some (stepStmt_assign_eq_some (acc_init_eval _ BM BN)),
    stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  refine ⟨by simp [hpids], by simp, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp [hpm]
  · simp [hpn]
  · -- accumulator = accPartial … 0 = 0
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx
    simp only [accPartial, Nat.zero_mul, gemmSum_zero]
  · simp [hm]
  · simp [hn]
  · simp [hk]
  · -- a_ptrs (c = 0)
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx <;> simp [Nat.zero_mul]
  · -- b_ptrs (c = 0)
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx <;> simp [Nat.zero_mul]
  · intro rg o; simp [huf, hundef]
  · exact hmem

set_option maxHeartbeats 2000000 in
/-- **Step lemma**: one K-loop body iteration advances the invariant by one
block (`accumulator = tl.dot(a, b, accumulator)` adds the `c`-th block's dot to
the partial GEMM accumulator; the `a`/`b` pointers advance one step). -/
theorem matmul_step (A B : RegionName) (s0 : BlockState)
    (BM BN BLOCK_K numKBlocks : Nat)
    (c : Nat) (s : BlockState) (hclt : c < numKBlocks)
    (hinv : matmulInvariant A B s0 BM BN BLOCK_K numKBlocks c s) :
    ∃ s', stepStmts (matmulLoopBody BM BN BLOCK_K)
        (s.setReg "k" .nat [] (Tile.scalar c)) = some s'
      ∧ matmulInvariant A B s0 BM BN BLOCK_K numKBlocks (c + 1) s' := by
  simp only [matmulInvariant] at hinv
  obtain ⟨hpids, hcle, hpm, hpn, hz, hm, hn, hk, hap, hbp, hundef, hmem⟩ := hinv
  -- abbreviations for the seeded a/b pointer tiles
  set apT : Tile .ptr [BM, BLOCK_K] :=
    ⟨fun idx : TileIndex [BM, BLOCK_K] => (A.cast, rowIndex s0 BM idx.1 * 4096 + idx.2.1.val * 1 + c * BLOCK_K)⟩ with hapT
  set bpT : Tile .ptr [BLOCK_K, BN] :=
    ⟨fun idx : TileIndex [BLOCK_K, BN] => (B.cast, idx.1.val * 4096 + colIndex s0 BN idx.2.1 * 1 + c * BLOCK_K * 4096)⟩ with hbpT
  set zT : Tile .real [BM, BN] :=
    ⟨fun idx : TileIndex [BM, BN] => some (accPartial s0 A B BM BN BLOCK_K idx.1 idx.2.1 c)⟩ with hzT
  -- the k-set state
  set sk := s.setReg "k" .nat [] (Tile.scalar c) with hsk
  have hrmem : ∀ (R : RegionName) (o : Nat), sk.readMem R o = s0.readMem R o := by
    intro R o; simp only [hsk, BlockState.setReg_readMem]; unfold BlockState.readMem; rw [hmem]
  have hapk : sk.regs .ptr [BM, BLOCK_K] "a_ptrs" = some apT := by simp [hsk, hap, hapT]
  have hbpk : sk.regs .ptr [BLOCK_K, BN] "b_ptrs" = some bpT := by simp [hsk, hbp, hbpT]
  have hzk : sk.regs .real [BM, BN] "accumulator" = some zT := by simp [hsk, hz, hzT]
  -- a / b loaded tiles
  set asub : Tile .real [BM, BLOCK_K] :=
    ⟨fun idx => some (sk.readMem (apT.data idx).1 (apT.data idx).2)⟩ with hasub
  set bsub : Tile .real [BLOCK_K, BN] :=
    ⟨fun idx => some (sk.readMem (bpT.data idx).1 (bpT.data idx).2)⟩ with hbsub
  unfold matmulLoopBody
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (load_ptr_none_real (Op.ref .ptr [BM, BLOCK_K] "a_ptrs") _ apT (by rw [evalOp_ref]; simp [hapk])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (load_ptr_none_real (Op.ref .ptr [BLOCK_K, BN] "b_ptrs") _ bpT (by rw [evalOp_ref]; simp [hbpk])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (accdot_op_eval BM BLOCK_K BN _ zT asub bsub
          (by simp [hzk, hasub, hbsub, BlockState.setReg_readMem])
          (by simp [hasub, BlockState.setReg_readMem])
          (by simp [hbsub, BlockState.setReg_readMem])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (aptr_adv_eval _ BM BLOCK_K BLOCK_K apT (by simp [hapk])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (bptr_adv_eval _ BLOCK_K BN BLOCK_K bpT (by simp [hbpk])))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  simp only [matmulInvariant]
  refine ⟨by simp [hsk, hpids], by omega, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp [hsk, hpm]
  · simp [hsk, hpn]
  · -- accumulator = accPartial (c+1)
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx
    have has : ∀ e : Fin BLOCK_K, asub.data (idx.1, e, PUnit.unit)
        = some (aElem s0 A BM idx.1 (c * BLOCK_K + e.val)) := by
      intro e
      simp only [hasub, hapT, hrmem, aElem, Region.cast_id]
      rw [show rowIndex s0 BM idx.1 * 4096 + e.val * 1 + c * BLOCK_K
            = rowIndex s0 BM idx.1 * 4096 + (c * BLOCK_K + e.val) from by ring]
    have hbs : ∀ e : Fin BLOCK_K, bsub.data (e, idx.2.1, PUnit.unit)
        = some (bElem s0 B BN idx.2.1 (c * BLOCK_K + e.val)) := by
      intro e
      simp only [hbsub, hbpT, hrmem, bElem, Region.cast_id]
      rw [show e.val * 4096 + colIndex s0 BN idx.2.1 * 1 + c * BLOCK_K * 4096
            = (c * BLOCK_K + e.val) * 4096 + colIndex s0 BN idx.2.1 from by ring]
    rw [dotadd_eval BM BN (Tile.dot [] asub bsub) zT idx.1 idx.2.1
        (Finset.univ.sum fun e : Fin BLOCK_K =>
          aElem s0 A BM idx.1 (c * BLOCK_K + e.val) * bElem s0 B BN idx.2.1 (c * BLOCK_K + e.val))
        (accPartial s0 A B BM BN BLOCK_K idx.1 idx.2.1 c)
        (tile_dot_data BM BLOCK_K BN asub bsub idx.1 idx.2.1 _ _ has hbs)
        (by rw [hzT])]
    show some _ = some (accPartial s0 A B BM BN BLOCK_K idx.1 idx.2.1 (c + 1))
    rw [accPartial_succ]
    congr 1
    ring
  · simp [hsk, hm]
  · simp [hsk, hn]
  · simp [hsk, hk]
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
  · intro rg o; simp [hsk, hundef]
  · show _ = s0.mem
    rw [← hmem, hsk]; rfl

/-! ## Post-loop: fp16 cast + store -/

/-- The output store address for tile lane `(i,j)`: `4096 · rowGlobal i + colGlobal j`
(the kernel's `c_ptrs`, which uses the **un-wrapped** `offs_cm` / `offs_cn`). -/
def cOffset (s0 : BlockState) (BM BN : Nat) (idx : TileIndex [BM, BN]) : Nat :=
  4096 * rowGlobal s0 BM idx.1 + 1 * colGlobal s0 BN idx.2.1

/-- `offs_cm` / `offs_cn` eval (the **un-wrapped** global index, no `% 4096`). -/
theorem offscm_eval (s : BlockState) (M BM : Nat) (pm : Nat) (pidReg : RegName)
    (hpm : s.regs .nat [] pidReg = some (Tile.scalar pm)) :
    evalOp (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] pidReg) (Op.constNat BM)) (Op.arange M)) s
      = some (Tile.vec (fun i : Fin M => pm * BM + i.val)) := by
  simp only [evalOp_add, evalOp_mul, evalOp_ref, evalOp_constNat, evalOp_arange, hpm,
    Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext i
  simp only [Tile.bop_data, Tile.scalar_data, Tile.vec_data, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add, NumericDType.mul]

set_option maxHeartbeats 2000000 in
/-- **postLoop**: from the invariant at `numKBlocks` blocks, the cast-to-fp16 and
store of the accumulator to `c_ptrs` writes the genuine GEMM value
`fp16(matmulSpec)` at every output lane (given the output-offset map is
injective). -/
theorem matmul_postLoop (C A B : RegionName) (s0 : BlockState)
    (BM BN BLOCK_K numKBlocks : Nat)
    (hInj : Function.Injective (cOffset s0 BM BN))
    (st : BlockState)
    (hinv : matmulInvariant A B s0 BM BN BLOCK_K numKBlocks numKBlocks st) :
    ∃ sfin, stepStmts (matmulStoreTail C BM BN BM BN) st = some sfin
      ∧ ∀ idx : TileIndex [BM, BN],
          sfin.mem C (cOffset s0 BM BN idx)
            = MemCell.of .fp16
                (FloatDType.real.cast FloatDType.fp16
                  (some (matmulSpec s0 A B BM BN BLOCK_K numKBlocks idx.1 idx.2.1))) := by
  simp only [matmulInvariant] at hinv
  obtain ⟨hpids, hcle, hpm, hpn, hz, hm, hn, hk, hap, hbp, hundef, hmem⟩ := hinv
  set zT : Tile .real [BM, BN] :=
    ⟨fun idx : TileIndex [BM, BN] => some (accPartial s0 A B BM BN BLOCK_K idx.1 idx.2.1 numKBlocks)⟩ with hzT
  -- fp16-cast `c` tile
  set cT : Tile .fp16 [BM, BN] :=
    ⟨fun idx => FloatDType.real.cast FloatDType.fp16 (zT.data idx)⟩ with hcT
  -- output pointer chunk
  set cpT : Tile .ptr [BM, BN] :=
    ⟨fun idx : TileIndex [BM, BN] => (C.cast, cOffset s0 BM BN idx)⟩ with hcpT
  unfold matmulStoreTail
  -- c = cast(accumulator, fp16)
  erw [stepStmts.cons_some (stepStmt_assign_eq_some
        (show evalOp (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref .real [BM, BN] "accumulator")) st
          = some cT from by rw [evalOp_castFloat]; simp [evalOp_ref, hz, hcT, hzT]))]
  -- offs_cm
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (offscm_eval _ BM BM (s0.pids 0) "pid_m" (by simp [hpm])))]
  -- offs_cn
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (offscm_eval _ BN BN (s0.pids 1) "pid_n" (by simp [hpn])))]
  -- c_ptrs (strides on the left)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase C)
            (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
              (Op.mul .nat Broadcast.scalarL (Op.constNat 4096) (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_cm")))
              (Op.mul .nat Broadcast.scalarL (Op.constNat 1) (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_cn"))))) _
          = some cpT from by
          rw [cptrs_eval _ C BM BN (fun i => rowGlobal s0 BM i) (fun j => colGlobal s0 BN j)
                (by simp [rowGlobal]) (by simp [colGlobal])]
          rfl))]
  -- abstract the post-assign state; only the `c` and `c_ptrs` readbacks matter
  generalize hst4 : ((((st.setReg "c" FloatDType.fp16.toTileDType [BM, BN] cT).setReg "offs_cm" .nat [BM]
        (Tile.vec fun i : Fin BM => s0.pids 0 * BM + i.val)).setReg "offs_cn" .nat [BN]
        (Tile.vec fun j : Fin BN => s0.pids 1 * BN + j.val)).setReg "c_ptrs" .ptr [BM, BN] cpT) = st4
  have hc4 : st4.regs .fp16 [BM, BN] "c" = some cT := by
    rw [← hst4]; simp
  have hcp4 : st4.regs .ptr [BM, BN] "c_ptrs" = some cpT := by
    rw [← hst4]; simp
  have hstore : stepStmt (Stmt.store .fp16 [BM, BN] (.ptr (Op.ref .ptr [BM, BN] "c_ptrs"))
        (Op.ref .fp16 [BM, BN] "c") .none) st4
      = some ((TileShape.allIndices [BM, BN]).foldl
          (fun acc i => acc.writeMemTyped .fp16 C (cOffset s0 BM BN i) (cT.data i)) st4) := by
    simp only [stepStmt]
    rw [show evalOp (Op.ref .fp16 [BM, BN] "c") st4 = some cT from by rw [evalOp_ref, hc4]]
    rw [show evalOp (Op.ref .ptr [BM, BN] "c_ptrs") st4 = some cpT from by rw [evalOp_ref, hcp4]]
    simp only [bind, Option.bind_some]
    refine congrArg some (List.foldl_ext _ _ _ (fun acc i _ => ?_))
    simp only [if_true, cpT, hcpT, Region.cast_id]
  rw [stepStmts.cons_some hstore, stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  intro idx
  rw [scatter_memcell_fp16_nd (region := C) (s := st4)
        (offsetFn := cOffset s0 BM BN) (valueFn := fun i => cT.data i) hInj idx]
  -- collapse the fp16 round-trip and rewrite acc → matmulSpec
  simp only [hcT, hzT, matmulSpec, accPartial, Nat.mul_comm numKBlocks BLOCK_K,
    FloatDType.cast, FloatDType.ofReal, FloatDType.storeValue, FloatDType.real_toWithBot,
    FloatDType.fp16_ofWithBot, FloatDType.fp16_toWithBot, WithBot.unbotD_some]

/-! ## Composition: full exec closed form -/

set_option maxHeartbeats 2000000 in
/-- **Top exec reduction**: composes `preLoop` + `matmul_step` (driven by
`forRange_inv`) + `matmul_postLoop` into the full `exec` result. Every output
lane's memory cell equals the genuine `fp16(∑_k A·B)` GEMM value. -/
theorem matmul_kernel_exec_closed_form (C A B : RegionName) (s : BlockState)
    (BM BN BLOCK_K numKBlocks : Nat)
    (hInj : Function.Injective (cOffset s BM BN))
    (hundef : ∀ rg o, s.undef rg o = 0)
    (idx : TileIndex [BM, BN]) :
    (match exec (matmul_kernel_surface C A B BM BN BLOCK_K numKBlocks) s with
      | some s' => s'.mem C (cOffset s BM BN idx)
      | none => (0 : MemCell)) =
      MemCell.of .fp16
        (FloatDType.real.cast FloatDType.fp16
          (some (matmulSpec s A B BM BN BLOCK_K numKBlocks idx.1 idx.2.1))) := by
  -- preLoop establishes P 0
  obtain ⟨s0, hpre_eq, hP0⟩ := preLoop C A B s BM BN BLOCK_K numKBlocks hundef
  -- drive the K-loop
  obtain ⟨final, sLoop, hLoopStmt, hfinal, hPLoop⟩ :=
    forRange_inv (idx := "k") (start := 0) (stop := numKBlocks) (step := 1)
      (by omega) hP0
      (fun c st hlt hinv => by
        have := matmul_step A B s BM BN BLOCK_K numKBlocks c st hlt hinv
        simpa using this)
  -- loop exit: final = numKBlocks
  have hfinalEq : final = numKBlocks := by
    have hle : final ≤ numKBlocks := by
      simp only [matmulInvariant] at hPLoop
      exact hPLoop.2.1
    exact le_antisymm hle hfinal
  rw [hfinalEq] at hPLoop
  -- postLoop reads off the closed form
  obtain ⟨sfin, hTail, hpost⟩ := matmul_postLoop C A B s BM BN BLOCK_K numKBlocks hInj sLoop hPLoop
  have hexec : exec (matmul_kernel_surface C A B BM BN BLOCK_K numKBlocks) s = some sfin := by
    rw [exec, matmul_body_split C A B BM BN BLOCK_K numKBlocks,
      stepStmts.append_some hpre_eq, stepStmts.cons_some hLoopStmt, hTail]
  rw [hexec]
  exact hpost idx

/-- The contiguous `4096 × 4096` output tile addresses are injective whenever the
column-block width fits a row (`BN ≤ 4096`), which holds for every valid tiling.
(`cOffset = 4096·(pid_m·BM + i) + (pid_n·BN + j)` is then a proper base-`4096`
encoding in `(i,j)`.) -/
theorem matmul_kernel_output_offset_injective
    (s : BlockState) {BM BN : Nat} (hBN : BN ≤ 4096) :
    Function.Injective (cOffset s BM BN) := by
  -- `cOffset = 4096·(P·BM + i) + (Q·BN + j) = (4096·P·BM + Q·BN) + i·4096 + j`
  have heq : cOffset s BM BN
      = fun idx : TileIndex [BM, BN] =>
          (4096 * (s.pids 0 * BM) + s.pids 1 * BN) + idx.1.val * 4096 + idx.2.1.val := by
    funext idx; simp only [cOffset, rowGlobal, colGlobal]; ring
  rw [heq]
  exact rowMajor2D_inj _ 4096 hBN

/-- **Closed-form correctness for `matmul_kernel` (general statement).**

For arbitrary 2-D program coordinates `(pid_m, pid_n)`, tile dims `BM`/`BN`,
K-block size `BLOCK_K`, and K-block count `numKBlocks` (so the contracted
dimension is `K = BLOCK_K · numKBlocks`), every output cell of the computed
`BM × BN` tile equals `fp16( Σ_{k < BLOCK_K·numKBlocks} A[i,k] · B[k,j] )` — the
genuine matrix product (over ℝ) of the loaded `A`/`B` tiles, cast to float16 —
**not** the kernel's own executed value.

Layout: `A[i,k]` at `A + offs_am(i) · 4096 + k`, `B[k,j]` at
`B + k · 4096 + offs_bn(j)`, `C[i,j]` at `C + 4096 · offs_cm(i) + offs_cn(j)`,
with `offs_am(i) = (pid_m·BM + i) % 4096`, `offs_bn(j) = (pid_n·BN + j) % 4096`
(the kernel's row-major pointer arithmetic with `% 4096` wrap). Preconditions:
`BN ≤ 4096` (column-block width ≤ the row stride 4096, always true for a valid
tiling — this discharges output-offset injectivity via
`matmul_kernel_output_offset_injective`) and clean initial `undef`. -/
theorem matmul_kernel_closed_form_correct
    (C A B : RegionName) (s : BlockState)
    (BM BN BLOCK_K numKBlocks : Nat)
    (hBN : BN ≤ 4096)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := matmul_kernel_surface C A B BM BN BLOCK_K numKBlocks)
      (initialState := s)
      (write := fun idx : TileIndex [BM, BN] => some (C, cOffset s BM BN idx))
      (expected := fun idx : TileIndex [BM, BN] =>
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (matmulSpec s A B BM BN BLOCK_K numKBlocks idx.1 idx.2.1)))) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [matmul_kernel_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst hs0
  intro idx
  have hInj : Function.Injective (cOffset s0 BM BN) := matmul_kernel_output_offset_injective s0 hBN
  have hmain := matmul_kernel_exec_closed_form C A B s0 BM BN BLOCK_K numKBlocks hInj hundef idx
  have hExec2 : exec (matmul_kernel_surface C A B BM BN BLOCK_K numKBlocks) s0 = some s' := hExec
  rw [hExec2] at hmain
  simpa only [ComputeCorrect.OutputReadable.read_memcell] using hmain

end VeriTile.Bench.TritonBenchG.MatmulKernel
