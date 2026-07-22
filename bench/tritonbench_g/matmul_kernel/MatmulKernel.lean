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
specification matmul_kernel_closed_form_correct
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

/-! ## The `⊨[R]` streaming headline (wave-5 S1 fold genre)

Everything below is purely additive; the exact surface above is untouched.
Structure of the `execR R` story: the kernel's only two rounding events live in
`matmulStoreTail` — the `tl.cast(accumulator, tl.float16)` (`evalOpR` site 1)
and the `.fp16`-typed `tl.store` (`writeMemAsR` site 2). The prologue and the
whole K-loop are cast-free, so under `execR R` they collapse verbatim onto the
exact stepper (`stepForRangeAuxR_castFree`) and the exact
`preLoop`/`matmul_step`/`forRange_inv` invariant stack above is reused
unchanged. `round_idem` collapses the tail's double round into the single
boundary `R.round .fp16` the skin's readback contract states. -/

open scoped VeriTile.Triton.StreamMasked2DKernelIO₂

/-! ### R-stepper glue

Small general helpers, local for now (library candidates: the `stepStmtsR`
list lemmas and `stepStmtR_assign_inv` belong in `Float.StepR`; the
`TraceSafeListR` append/of-forall principles and the unmasked scatter/readback
lemmas in `Memory.FlattenR` / `Float.StepR`). -/

/-- `stepStmtsR` on `[]` (R-mirror of `stepStmts.nil`). -/
private theorem stepStmtsR_nil' (R : RoundingModel) (s : BlockState) :
    stepStmtsR R [] s = some s := by
  simp only [stepStmtsR]

/-- `stepStmtsR` cons chaining (R-mirror of `stepStmts.cons_some`). -/
private theorem stepStmtsR_cons_some {R : RoundingModel} {st : Stmt} {rest : List Stmt}
    {s s' : BlockState} (h : stepStmtR R st s = some s') :
    stepStmtsR R (st :: rest) s = stepStmtsR R rest s' := by
  simp only [stepStmtsR, h]

/-- R-mirror of `stepStmt_assign_eq_some`. -/
private theorem stepStmtR_assign_someR {R : RoundingModel} {dtype : TileDType}
    {shape : TileShape} {name : RegName} {e : Op dtype shape} {s : BlockState}
    {v : Tile dtype shape} (h : evalOpR R e s = some v) :
    stepStmtR R (.assign dtype shape name e) s = some (s.setReg name dtype shape v) := by
  simp [stepStmtR, h]

/-- Inversion of a successful R-step of an assign: the successor is a
`setReg` of the evaluated value. Lets a `TraceSafeListR` walk thread successor
states of register-only statements without computing their values. -/
private theorem stepStmtR_assign_inv {R : RoundingModel} {dtype : TileDType}
    {shape : TileShape} {name : RegName} {e : Op dtype shape} {s s' : BlockState}
    (h : stepStmtR R (.assign dtype shape name e) s = some s') :
    ∃ v, evalOpR R e s = some v ∧ s' = s.setReg name dtype shape v := by
  cases hv : evalOpR R e s with
  | none => simp [stepStmtR, hv] at h
  | some v =>
      simp [stepStmtR, hv] at h
      exact ⟨v, rfl, h.symm⟩

/-- `evalOpR` on a register reference is the register file (R-independent). -/
private theorem evalOpR_ref' (R : RoundingModel) {dtype : TileDType} {shape : TileShape}
    (name : RegName) (s : BlockState) :
    evalOpR R (Op.ref dtype shape name) s = s.regs dtype shape name := by
  simp only [evalOpR]

/-- `stepStmtR` on a static `forRange` is the R loop auxiliary. -/
private theorem stepStmtR_forRange' (R : RoundingModel) (idx : RegName)
    (start stop step : Nat) (body : List Stmt) (s : BlockState) :
    stepStmtR R (.forRange idx start stop step body) s
      = stepForRangeAuxR R idx start stop step body s := by
  simp only [stepStmtR]

/-- `TraceSafeListR` append principle: a concatenation is trace-safe when the
first part is and every successor it actually reaches makes the second part
trace-safe. -/
private theorem traceSafeListR_append {R : RoundingModel} {bounds : RegionBounds} :
    ∀ (l1 : List Stmt) {l2 : List Stmt} (s : BlockState),
      Stmt.TraceSafeListR R bounds l1 s →
      (∀ s', stepStmtsR R l1 s = some s' → Stmt.TraceSafeListR R bounds l2 s') →
      Stmt.TraceSafeListR R bounds (l1 ++ l2) s
  | [], _, s, _, h2 => h2 s (by simp only [stepStmtsR])
  | st :: rest, l2, s, h1, h2 => by
      rw [Stmt.TraceSafeListR] at h1
      refine Stmt.TraceSafeListR.cons_intro h1.1 (fun s' hs' => ?_)
      have htl := h1.2
      rw [hs'] at htl
      exact traceSafeListR_append rest s' htl
        (fun s'' hs'' => h2 s'' ((stepStmtsR_cons_some hs').trans hs''))

/-- Statements safe at *every* state are trace-safe as a list from any state
(covers register-only assign runs, where no successor computation is needed). -/
private theorem traceSafeListR_of_forall {R : RoundingModel} {bounds : RegionBounds} :
    ∀ (l : List Stmt) (s : BlockState),
      (∀ st ∈ l, ∀ s', Stmt.TraceSafeR R bounds st s') →
      Stmt.TraceSafeListR R bounds l s
  | [], _, _ => Stmt.TraceSafeListR.nil_intro
  | st :: rest, s, h => by
      refine Stmt.TraceSafeListR.cons_intro (h st List.mem_cons_self s) (fun s' _ => ?_)
      exact traceSafeListR_of_forall rest s'
        (fun st' hst' => h st' (List.mem_cons_of_mem st hst'))

/-- Unmasked `writeMemAsR` scatter readback (the mask-free specialization of
`scatter_memcell_R_prop_masked_nd`). -/
private theorem scatter_memcell_R_nd (R : RoundingModel) (dtype : FloatDType)
    {region : RegionName} {shape : TileShape} (s : BlockState)
    (offsetFn : TileIndex shape → Nat)
    (valueFn : TileIndex shape → TileCarrier dtype.toTileDType)
    (h_inj : Function.Injective offsetFn) (i : TileIndex shape) :
    ((TileShape.allIndices shape).foldl
       (fun acc k => acc.writeMemAsR R dtype region (offsetFn k) (valueFn k)) s).mem
      region (offsetFn i)
    = MemCell.of dtype.toTileDType (dtype.ofReal (R.storeValue dtype (valueFn i))) := by
  have h := BlockState.scatter_memcell_R_prop_masked_nd R dtype (region := region)
    s offsetFn valueFn (fun _ => True) h_inj i
  simpa using h

/-- Tag-exact readback of a stored fp16 cell through `readMemAs .fp16` (the
`readMemAs` sibling of `readMemValue_bf16_of_cell`): the `storeValue ∘ ofReal`
round trip is the identity on a defined real. -/
private theorem readMemAs_fp16_of_cell {s : BlockState} {region : RegionName}
    {offset : Nat} {x : ℝ}
    (h : s.mem region offset
      = MemCell.of FloatDType.fp16.toTileDType (FloatDType.fp16.ofReal x)) :
    s.readMemAs .fp16 region offset = FloatDType.fp16.ofReal x := by
  simp [BlockState.readMemAs, h, FloatDType.storeValue, FloatDType.ofReal]

/-- The tail's two rounding events collapse to one boundary round:
`R.storeValue .fp16` after `R.cast .real .fp16` is a single `R.round .fp16`
(by the defining `round_idem`, same recipe as the quantize family). -/
private theorem R_storeValue_cast_fp16 (R : RoundingModel) (v : ℝ) :
    R.storeValue .fp16 (RoundingModel.cast R .real .fp16 (some v)) = R.round .fp16 v := by
  simp [RoundingModel.storeValue, RoundingModel.cast, FloatDType.storeValue, R.round_idem]

/-! ### Body decomposition names and cast-free collapses -/

/-- The 8-statement prologue (statements 0–7) as an explicit list. -/
private def matmulPrologue (A B : RegionName) (BM BN BK : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "pid_m" (Op.programId 0),
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
    Stmt.assign .nat [BK] "offs_k" (Op.arange BK),
    Stmt.assign .ptr [BM, BK] "a_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase A)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_am")) (Op.constNat 4096))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BK] "offs_k")) (Op.constNat 1)))),
    Stmt.assign .ptr [BK, BN] "b_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase B)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "offs_k")) (Op.constNat 4096))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_bn")) (Op.constNat 1)))),
    Stmt.assign .real [BM, BN] "accumulator" (Op.full [BM, BN] (Op.const 0)) ]

private theorem matmul_take8_eq (C A B : RegionName) (BM BN BK T : Nat) :
    (matmul_kernel_surface C A B BM BN BK T).toAlgKernel.body.take 8
      = matmulPrologue A B BM BN BK := rfl

/-- `matmul_body_split` with the prologue named. By `rfl`. -/
private theorem matmul_body_split' (C A B : RegionName) (BM BN BK T : Nat) :
    (matmul_kernel_surface C A B BM BN BK T).toAlgKernel.body
      = matmulPrologue A B BM BN BK
        ++ (Stmt.forRange "k" 0 T 1 (matmulLoopBody BM BN BK)
            :: matmulStoreTail C BM BN BM BN) := rfl

set_option maxHeartbeats 1000000 in
/-- The prologue is cast-free: it steps identically under `stepStmtsR R`. -/
private theorem matmulPrologue_castFree (R : RoundingModel) (A B : RegionName)
    (BM BN BK : Nat) (t : BlockState) :
    stepStmtsR R (matmulPrologue A B BM BN BK) t
      = stepStmts (matmulPrologue A B BM BN BK) t := by
  simp only [matmulPrologue, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def]
  rfl

set_option maxHeartbeats 1000000 in
/-- The K-loop body is cast-free (loads at `.real`, `tl.dot`+add, nat pointer
advances — no `castFloat`, no narrow-float store): it steps identically under
`stepStmtsR R`, so the exact invariant stack transports to `execR`. -/
private theorem matmulBody_castFree (R : RoundingModel) (BM BN BK : Nat) (t : BlockState) :
    stepStmtsR R (matmulLoopBody BM BN BK) t = stepStmts (matmulLoopBody BM BN BK) t := by
  simp only [matmulLoopBody, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def]
  rfl

/-- The `offs_c*` index-vector op is cast-free. -/
private theorem evalR_offsc (R : RoundingModel) (M BMc : Nat) (pidReg : RegName)
    (s : BlockState) :
    evalOpR R (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] pidReg) (Op.constNat BMc))
        (Op.arange M)) s
      = evalOp (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] pidReg) (Op.constNat BMc))
        (Op.arange M)) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

/-- The `c_ptrs` pointer op is cast-free. -/
private theorem evalR_cptrs (R : RoundingModel) (Creg : RegionName) (M N : Nat)
    (s : BlockState) :
    evalOpR R (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Creg)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarL (Op.constNat 4096) (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_cm")))
          (Op.mul .nat Broadcast.scalarL (Op.constNat 1) (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [N] "offs_cn"))))) s
      = evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Creg)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarL (Op.constNat 4096) (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_cm")))
          (Op.mul .nat Broadcast.scalarL (Op.constNat 1) (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [N] "offs_cn"))))) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

/-- `offs_cm` / `offs_cn` eval under `R` (via the cast-free collapse). -/
private theorem offscm_evalR (R : RoundingModel) (s : BlockState) (M BMc : Nat)
    (pm : Nat) (pidReg : RegName)
    (hpm : s.regs .nat [] pidReg = some (Tile.scalar pm)) :
    evalOpR R (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] pidReg) (Op.constNat BMc))
        (Op.arange M)) s
      = some (Tile.vec (fun i : Fin M => pm * BMc + i.val)) := by
  rw [evalR_offsc]
  exact offscm_eval s M BMc pm pidReg hpm

/-- `c = tl.cast(accumulator, tl.float16)` eval under `R`: rounding-event
site 1, `R.cast` applied lane-wise to the accumulator tile. Proven by direct
`eq_def` unfolding (a `rw` through `evalOpR_castFloat` trips over the
`FloatDType.real.toTileDType` vs `TileDType.real` spelling). -/
private theorem castAcc_evalR (R : RoundingModel) (M N : Nat) (s : BlockState)
    (zT : Tile .real [M, N]) (hz : s.regs .real [M, N] "accumulator" = some zT) :
    evalOpR R (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref .real [M, N] "accumulator")) s
      = some ⟨fun idx => RoundingModel.cast R FloatDType.real FloatDType.fp16 (zT.data idx)⟩ := by
  have hz' : s.regs FloatDType.real.toTileDType [M, N] "accumulator" = some zT := hz
  simp only [evalOpR.eq_def, hz']
  rfl

/-! ### The tail under `execR R` -/

set_option maxHeartbeats 2000000 in
/-- **R-postLoop**: from the exact invariant at `numKBlocks` blocks, the
`execR R` store tail terminates and writes, at every output lane, the cell
`fp16(R.round .fp16 (matmulSpec …))` — the ideal GEMM value rounded **once**
at the fp16 grid (`R.cast` site + `R.storeValue` site collapsed by
`round_idem`) — with a per-cell frame outside the output window. -/
private theorem matmul_postLoopR (R : RoundingModel) (C A B : RegionName) (s0 : BlockState)
    (BM BN BLOCK_K numKBlocks : Nat)
    (hInj : Function.Injective (cOffset s0 BM BN))
    (st : BlockState)
    (hinv : matmulInvariant A B s0 BM BN BLOCK_K numKBlocks numKBlocks st) :
    ∃ sfin, stepStmtsR R (matmulStoreTail C BM BN BM BN) st = some sfin
      ∧ (∀ idx : TileIndex [BM, BN],
          sfin.mem C (cOffset s0 BM BN idx)
            = MemCell.of FloatDType.fp16.toTileDType
                (FloatDType.fp16.ofReal
                  (R.round .fp16 (matmulSpec s0 A B BM BN BLOCK_K numKBlocks idx.1 idx.2.1))))
      ∧ (∀ r o, (r ≠ C ∨ ∀ idx : TileIndex [BM, BN], o ≠ cOffset s0 BM BN idx) →
          sfin.mem r o = st.mem r o) := by
  simp only [matmulInvariant] at hinv
  obtain ⟨hpids, hcle, hpm, hpn, hz, hm, hn, hk, hap, hbp, hundef, hmem⟩ := hinv
  set zT : Tile .real [BM, BN] :=
    ⟨fun idx : TileIndex [BM, BN] => some (accPartial s0 A B BM BN BLOCK_K idx.1 idx.2.1 numKBlocks)⟩ with hzT
  set cT : Tile .fp16 [BM, BN] :=
    ⟨fun idx => RoundingModel.cast R FloatDType.real FloatDType.fp16 (zT.data idx)⟩ with hcT
  set cpT : Tile .ptr [BM, BN] :=
    ⟨fun idx : TileIndex [BM, BN] => (C.cast, cOffset s0 BM BN idx)⟩ with hcpT
  unfold matmulStoreTail
  -- c = cast(accumulator, fp16): rounding-event site 1 (`R.cast`)
  erw [stepStmtsR_cons_some (stepStmtR_assign_someR
        (show evalOpR R (Op.castFloat FloatDType.real FloatDType.fp16
              (Op.ref .real [BM, BN] "accumulator")) st = some cT
          from castAcc_evalR R BM BN st zT hz))]
  -- offs_cm / offs_cn (cast-free)
  rw [stepStmtsR_cons_some (stepStmtR_assign_someR
        (offscm_evalR R _ BM BM (s0.pids 0) "pid_m" (by simp [hpm])))]
  rw [stepStmtsR_cons_some (stepStmtR_assign_someR
        (offscm_evalR R _ BN BN (s0.pids 1) "pid_n" (by simp [hpn])))]
  -- c_ptrs (cast-free)
  rw [stepStmtsR_cons_some (stepStmtR_assign_someR
        (show evalOpR R (Op.ptrAdd Broadcast.scalarL (Op.ptrBase C)
            (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
              (Op.mul .nat Broadcast.scalarL (Op.constNat 4096) (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_cm")))
              (Op.mul .nat Broadcast.scalarL (Op.constNat 1) (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_cn"))))) _
          = some cpT from by
          rw [evalR_cptrs,
            cptrs_eval _ C BM BN (fun i => rowGlobal s0 BM i) (fun j => colGlobal s0 BN j)
              (by simp [rowGlobal]) (by simp [colGlobal])]
          rfl))]
  -- abstract the post-assign state; only the `c` / `c_ptrs` readbacks matter
  generalize hst4 : ((((st.setReg "c" FloatDType.fp16.toTileDType [BM, BN] cT).setReg "offs_cm" .nat [BM]
        (Tile.vec fun i : Fin BM => s0.pids 0 * BM + i.val)).setReg "offs_cn" .nat [BN]
        (Tile.vec fun j : Fin BN => s0.pids 1 * BN + j.val)).setReg "c_ptrs" .ptr [BM, BN] cpT) = st4
  have hc4 : st4.regs .fp16 [BM, BN] "c" = some cT := by
    rw [← hst4]; simp
  have hcp4 : st4.regs .ptr [BM, BN] "c_ptrs" = some cpT := by
    rw [← hst4]; simp
  have hmem4 : st4.mem = st.mem := by
    rw [← hst4]; rfl
  -- the unmasked fp16 store: rounding-event site 2 (`writeMemAsR`)
  have hstore : stepStmtR R (Stmt.store .fp16 [BM, BN] (.ptr (Op.ref .ptr [BM, BN] "c_ptrs"))
        (Op.ref .fp16 [BM, BN] "c") .none) st4
      = some ((TileShape.allIndices [BM, BN]).foldl
          (fun acc i => acc.writeMemAsR R .fp16 C (cOffset s0 BM BN i) (cT.data i)) st4) := by
    simp only [stepStmtR]
    rw [show evalOpR R (Op.ref .fp16 [BM, BN] "c") st4 = some cT from by rw [evalOpR_ref', hc4]]
    rw [show evalOpR R (Op.ref .ptr [BM, BN] "c_ptrs") st4 = some cpT from by rw [evalOpR_ref', hcp4]]
    simp only [bind, Option.bind_some]
    refine congrArg some (List.foldl_ext _ _ _ (fun acc i _ => ?_))
    simp only [if_true, BlockState.writeMemTypedR_fp16, hcpT, Region.cast_id]
  rw [stepStmtsR_cons_some hstore, stepStmtsR_nil']
  refine ⟨_, rfl, ?_, ?_⟩
  · intro idx
    rw [scatter_memcell_R_nd R .fp16 st4 (cOffset s0 BM BN) (fun i => cT.data i) hInj idx]
    have hdata : cT.data idx = RoundingModel.cast R FloatDType.real FloatDType.fp16
        (some (accPartial s0 A B BM BN BLOCK_K idx.1 idx.2.1 numKBlocks)) := rfl
    rw [hdata, R_storeValue_cast_fp16]
    have hspec : matmulSpec s0 A B BM BN BLOCK_K numKBlocks idx.1 idx.2.1
        = accPartial s0 A B BM BN BLOCK_K idx.1 idx.2.1 numKBlocks := by
      unfold matmulSpec accPartial
      rw [Nat.mul_comm]
    rw [hspec]
  · intro r o hcond
    have hnot : ∀ k ∈ TileShape.allIndices [BM, BN], ¬(C = r ∧ cOffset s0 BM BN k = o) := by
      intro k _ hk
      obtain ⟨hCr, hko⟩ := hk
      rcases hcond with hne | hall
      · exact hne hCr.symm
      · exact hall k hko.symm
    rw [BlockState.foldl_writeMemAsR_preserve_cell R .fp16 (cOffset s0 BM BN)
        (fun i => cT.data i) r o (TileShape.allIndices [BM, BN]) hnot st4, hmem4]

/-! ### Safety-walk invariant (weak shape half of `matmulInvariant`) -/

/-- Safety-walk loop invariant: the *shape* half of `matmulInvariant` (`pid_m`
/ `pid_n` seeded, *some* accumulator tile, and the exact `a_ptrs` / `b_ptrs`
address shapes) with no `undef` / `mem` / value pins. Needed because the
`⊨[R]` skin's `hts` obligation quantifies over arbitrary launch states, so the
safety walk cannot assume the clean-`undef` precondition that `preLoop`'s full
invariant needs. -/
private def matmulSafeInv (A B : RegionName) (s0 : BlockState) (BM BN BLOCK_K T : Nat)
    (c : Nat) (s : BlockState) : Prop :=
  c ≤ T ∧
  (s.regs .nat [] "pid_m" = some (Tile.scalar (s0.pids 0))) ∧
  (s.regs .nat [] "pid_n" = some (Tile.scalar (s0.pids 1))) ∧
  (∃ zT : Tile .real [BM, BN], s.regs .real [BM, BN] "accumulator" = some zT) ∧
  (s.regs .ptr [BM, BLOCK_K] "a_ptrs" = some ⟨fun idx : TileIndex [BM, BLOCK_K] =>
      (A.cast, rowIndex s0 BM idx.1 * 4096 + idx.2.1.val * 1 + c * BLOCK_K)⟩) ∧
  (s.regs .ptr [BLOCK_K, BN] "b_ptrs" = some ⟨fun idx : TileIndex [BLOCK_K, BN] =>
      (B.cast, idx.1.val * 4096 + colIndex s0 BN idx.2.1 * 1 + c * BLOCK_K * 4096)⟩)

set_option maxHeartbeats 1000000 in
/-- Weak `preLoop`: from an **arbitrary** state the prologue steps to a state
satisfying `matmulSafeInv … 0` (no clean-`undef` hypothesis; the value half of
`preLoop` is dropped). -/
private theorem preLoopW (A B : RegionName) (s : BlockState) (BM BN BLOCK_K T : Nat) :
    ∃ s', stepStmts (matmulPrologue A B BM BN BLOCK_K) s = some s'
      ∧ matmulSafeInv A B s BM BN BLOCK_K T 0 s' := by
  obtain ⟨s5, h5, hpids, hpm, hpn, hm, hn, hk, huf, hmem⟩ := preLoop_scalars s BM BN BLOCK_K
  rw [show matmulPrologue A B BM BN BLOCK_K
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
  refine ⟨Nat.zero_le T, ?_, ?_, ?_, ?_, ?_⟩
  · simp [hpm]
  · simp [hpn]
  · refine ⟨⟨fun _ => some (0 : ℝ)⟩, ?_⟩
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
      String.reduceEq, not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx <;> simp [Nat.zero_mul]
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx <;> simp [Nat.zero_mul]

set_option maxHeartbeats 2000000 in
/-- Weak step lemma: one body iteration from `matmulSafeInv c` steps
successfully (exact stepper; the body is cast-free) and re-establishes
`matmulSafeInv (c+1)` — the shape half of `matmul_step`, valid from arbitrary
launch states. -/
private theorem matmul_stepW (A B : RegionName) (s0 : BlockState)
    (BM BN BLOCK_K T : Nat) (c : Nat) (s : BlockState) (hclt : c < T)
    (hinv : matmulSafeInv A B s0 BM BN BLOCK_K T c s) :
    ∃ s', stepStmts (matmulLoopBody BM BN BLOCK_K)
        (s.setReg "k" .nat [] (Tile.scalar c)) = some s'
      ∧ matmulSafeInv A B s0 BM BN BLOCK_K T (c + 1) s' := by
  obtain ⟨hcle, hpm, hpn, ⟨zT, hz⟩, hap, hbp⟩ := hinv
  set apT : Tile .ptr [BM, BLOCK_K] :=
    ⟨fun idx : TileIndex [BM, BLOCK_K] => (A.cast, rowIndex s0 BM idx.1 * 4096 + idx.2.1.val * 1 + c * BLOCK_K)⟩ with hapT
  set bpT : Tile .ptr [BLOCK_K, BN] :=
    ⟨fun idx : TileIndex [BLOCK_K, BN] => (B.cast, idx.1.val * 4096 + colIndex s0 BN idx.2.1 * 1 + c * BLOCK_K * 4096)⟩ with hbpT
  set sk := s.setReg "k" .nat [] (Tile.scalar c) with hsk
  have hapk : sk.regs .ptr [BM, BLOCK_K] "a_ptrs" = some apT := by simp [hsk, hap, hapT]
  have hbpk : sk.regs .ptr [BLOCK_K, BN] "b_ptrs" = some bpT := by simp [hsk, hbp, hbpT]
  have hzk : sk.regs .real [BM, BN] "accumulator" = some zT := by simp [hsk, hz]
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
  refine ⟨by omega, ?_, ?_, ?_, ?_, ?_⟩
  · simp [hsk, hpm]
  · simp [hsk, hpn]
  · refine ⟨Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
      (Tile.dot [] asub bsub) zT, ?_⟩
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
      String.reduceEq, not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx <;> simp only [Tile.ptrAdd, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
      Tile.scalar, hapT, NumericDType.add]
    ring
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx <;> simp only [Tile.ptrAdd, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
      Tile.scalar, hbpT, NumericDType.add]
    ring

set_option maxHeartbeats 1000000 in
/-- Per-iteration `TraceSafeListR` for the K-loop body: the two unmasked
loads' addresses are the invariant's pointer shapes, in bounds by the skin's
`read1`/`read2` windows (instantiated at step `c`); the three register-only
assigns are unconditionally safe. -/
private theorem matmul_bodySafeR (R : RoundingModel) (bounds : RegionBounds)
    (A B : RegionName) (s0 : BlockState) (BM BN BLOCK_K T : Nat) (c : Nat) (hc : c < T)
    (sk : BlockState)
    (hap : sk.regs .ptr [BM, BLOCK_K] "a_ptrs" = some ⟨fun idx : TileIndex [BM, BLOCK_K] =>
        (A.cast, rowIndex s0 BM idx.1 * 4096 + idx.2.1.val * 1 + c * BLOCK_K)⟩)
    (hbp : sk.regs .ptr [BLOCK_K, BN] "b_ptrs" = some ⟨fun idx : TileIndex [BLOCK_K, BN] =>
        (B.cast, idx.1.val * 4096 + colIndex s0 BN idx.2.1 * 1 + c * BLOCK_K * 4096)⟩)
    (hbA : ∀ (t : Fin T) (j : Fin (BM * BLOCK_K)),
      ((s0.pids 0 * BM + j.val / BLOCK_K) % 4096) * 4096 + (t.val * BLOCK_K + j.val % BLOCK_K)
        < bounds A)
    (hbB : ∀ (t : Fin T) (j : Fin (BLOCK_K * BN)),
      (t.val * BLOCK_K + j.val / BN) * 4096 + (s0.pids 1 * BN + j.val % BN) % 4096
        < bounds B) :
    Stmt.TraceSafeListR R bounds (matmulLoopBody BM BN BLOCK_K) sk := by
  unfold matmulLoopBody
  refine Stmt.TraceSafeListR.cons_intro ?_ ?_
  · -- load a
    simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR,
      memAccessActiveAddressSafeR]
    refine ⟨trivial, trivial, ?_⟩
    intro ptrs hptrs i _
    rw [evalOpR_ref', hap] at hptrs
    obtain rfl := Option.some.inj hptrs
    show rowIndex s0 BM i.1 * 4096 + i.2.1.val * 1 + c * BLOCK_K < bounds (Region.cast A)
    have h' : ((s0.pids 0 * BM + i.1.val) % 4096) * 4096 + (c * BLOCK_K + i.2.1.val)
        < bounds A := by
      simpa using hbA ⟨c, hc⟩ (Lane2D.encode (i.1, i.2.1, PUnit.unit))
    simp only [Region.cast_id]
    calc rowIndex s0 BM i.1 * 4096 + i.2.1.val * 1 + c * BLOCK_K
        = ((s0.pids 0 * BM + i.1.val) % 4096) * 4096 + (c * BLOCK_K + i.2.1.val) := by
          unfold rowIndex rowGlobal; ring
      _ < bounds A := h'
  · intro s1 h1
    obtain ⟨v1, -, rfl⟩ := stepStmtR_assign_inv h1
    refine Stmt.TraceSafeListR.cons_intro ?_ ?_
    · -- load b
      simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR,
        memAccessActiveAddressSafeR]
      refine ⟨trivial, trivial, ?_⟩
      intro ptrs hptrs i _
      rw [evalOpR_ref'] at hptrs
      rw [show (sk.setReg "a" .real [BM, BLOCK_K] v1).regs .ptr [BLOCK_K, BN] "b_ptrs"
          = some (⟨fun idx : TileIndex [BLOCK_K, BN] =>
            (B.cast, idx.1.val * 4096 + colIndex s0 BN idx.2.1 * 1 + c * BLOCK_K * 4096)⟩ :
              Tile .ptr [BLOCK_K, BN]) from by simp [hbp]] at hptrs
      obtain rfl := Option.some.inj hptrs
      show i.1.val * 4096 + colIndex s0 BN i.2.1 * 1 + c * BLOCK_K * 4096 < bounds (Region.cast B)
      have h' : (c * BLOCK_K + i.1.val) * 4096 + (s0.pids 1 * BN + i.2.1.val) % 4096
          < bounds B := by
        simpa using hbB ⟨c, hc⟩ (Lane2D.encode (i.1, i.2.1, PUnit.unit))
      simp only [Region.cast_id]
      calc i.1.val * 4096 + colIndex s0 BN i.2.1 * 1 + c * BLOCK_K * 4096
          = (c * BLOCK_K + i.1.val) * 4096 + (s0.pids 1 * BN + i.2.1.val) % 4096 := by
            unfold colIndex colGlobal; ring
        _ < bounds B := h'
    · intro s2 h2
      obtain ⟨v2, -, rfl⟩ := stepStmtR_assign_inv h2
      refine traceSafeListR_of_forall _ _ ?_
      intro st hst s'
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hst
      rcases hst with rfl | rfl | rfl <;>
        simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]

set_option maxHeartbeats 1000000 in
/-- `TraceSafeListR` for the store tail: the four assigns are register-only
(the cast is not a memory event) and the single unmasked `.fp16` store's
addresses are the skin's `write` window. -/
private theorem matmul_tailSafeR (R : RoundingModel) (bounds : RegionBounds)
    (C : RegionName) (s0 : BlockState) (BM BN : Nat) (st : BlockState)
    (hpm : st.regs .nat [] "pid_m" = some (Tile.scalar (s0.pids 0)))
    (hpn : st.regs .nat [] "pid_n" = some (Tile.scalar (s0.pids 1)))
    (hzE : ∃ zT : Tile .real [BM, BN], st.regs .real [BM, BN] "accumulator" = some zT)
    (hbC : ∀ j : Fin (BM * BN),
      4096 * (s0.pids 0 * BM + j.val / BN) + 1 * (s0.pids 1 * BN + j.val % BN) < bounds C) :
    Stmt.TraceSafeListR R bounds (matmulStoreTail C BM BN BM BN) st := by
  obtain ⟨zT, hz⟩ := hzE
  unfold matmulStoreTail
  -- 1. c = cast(accumulator, fp16)
  refine Stmt.TraceSafeListR.cons_intro (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) ?_
  intro s1 h1
  obtain ⟨v1, -, rfl⟩ := stepStmtR_assign_inv h1
  -- 2. offs_cm
  refine Stmt.TraceSafeListR.cons_intro (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) ?_
  intro s2 h2
  obtain ⟨v2, hv2, rfl⟩ := stepStmtR_assign_inv h2
  rw [offscm_evalR R _ BM BM (s0.pids 0) "pid_m" (by simp [hpm])] at hv2
  obtain rfl := Option.some.inj hv2
  -- 3. offs_cn
  refine Stmt.TraceSafeListR.cons_intro (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) ?_
  intro s3 h3
  obtain ⟨v3, hv3, rfl⟩ := stepStmtR_assign_inv h3
  rw [offscm_evalR R _ BN BN (s0.pids 1) "pid_n" (by simp [hpn])] at hv3
  obtain rfl := Option.some.inj hv3
  -- 4. c_ptrs
  refine Stmt.TraceSafeListR.cons_intro (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) ?_
  intro s4 h4
  obtain ⟨v4, hv4, rfl⟩ := stepStmtR_assign_inv h4
  rw [evalR_cptrs,
    cptrs_eval _ C BM BN (fun i => s0.pids 0 * BM + i.val) (fun j => s0.pids 1 * BN + j.val)
      (by simp) (by simp)] at hv4
  obtain rfl := Option.some.inj hv4
  -- 5. the store
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun s' _ => Stmt.TraceSafeListR.nil_intro)
  simp only [Stmt.TraceSafeR, MemAccess.SafeAtR, MaskOpt.SafeAtR, Op.SafeAtR.eq_def,
    MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
  refine ⟨trivial, trivial, trivial, ?_⟩
  intro ptrs hptrs i _
  rw [evalOpR_ref'] at hptrs
  simp only [BlockState.setReg_same] at hptrs
  obtain rfl := Option.some.inj hptrs
  show 4096 * (s0.pids 0 * BM + i.1.val) + 1 * (s0.pids 1 * BN + i.2.1.val)
      < bounds (Region.cast C)
  have h' := hbC (Lane2D.encode (i.1, i.2.1, PUnit.unit))
  rw [Lane2D.encode_div, Lane2D.encode_mod] at h'
  simpa only [Region.cast_id] using h'

set_option maxHeartbeats 1000000 in
/-- **The `TraceSafeR` walk for the whole kernel** — the library's first
looping safety walk, driven by `Stmt.forRangeTraceSafeR_inv` over the weak
`matmulSafeInv`. The three bound groups are the skin's `read1`/`read2`/`write`
windows. -/
private theorem matmul_kernel_traceSafeR (R : RoundingModel) (bounds : RegionBounds)
    (C A B : RegionName) (BM BN BLOCK_K T : Nat) (s : BlockState)
    (hbA : ∀ (t : Fin T) (j : Fin (BM * BLOCK_K)),
      ((s.pids 0 * BM + j.val / BLOCK_K) % 4096) * 4096 + (t.val * BLOCK_K + j.val % BLOCK_K)
        < bounds A)
    (hbB : ∀ (t : Fin T) (j : Fin (BLOCK_K * BN)),
      (t.val * BLOCK_K + j.val / BN) * 4096 + (s.pids 1 * BN + j.val % BN) % 4096
        < bounds B)
    (hbC : ∀ j : Fin (BM * BN),
      4096 * (s.pids 0 * BM + j.val / BN) + 1 * (s.pids 1 * BN + j.val % BN) < bounds C) :
    ((matmul_kernel_surface C A B BM BN BLOCK_K T).toAlgKernel).TraceSafeR R bounds s := by
  unfold Kernel.TraceSafeR
  rw [matmul_body_split' C A B BM BN BLOCK_K T]
  have hstep : ∀ c s', c < T → matmulSafeInv A B s BM BN BLOCK_K T c s' →
      Stmt.TraceSafeListR R bounds (matmulLoopBody BM BN BLOCK_K)
        (s'.setReg "k" .nat [] (Tile.scalar c)) ∧
      ∃ s'', stepStmtsR R (matmulLoopBody BM BN BLOCK_K)
          (s'.setReg "k" .nat [] (Tile.scalar c)) = some s'' ∧
        matmulSafeInv A B s BM BN BLOCK_K T (c + 1) s'' := by
    intro c s' hc hP
    obtain ⟨hcle, hpm, hpn, hzE, hap, hbp⟩ := hP
    refine ⟨matmul_bodySafeR R bounds A B s BM BN BLOCK_K T c hc _
        (by simp [hap]) (by simp [hbp]) hbA hbB, ?_⟩
    obtain ⟨s'', hs'', hP''⟩ :=
      matmul_stepW A B s BM BN BLOCK_K T c s' hc ⟨hcle, hpm, hpn, hzE, hap, hbp⟩
    exact ⟨s'', by rw [matmulBody_castFree]; exact hs'', hP''⟩
  refine traceSafeListR_append _ _ ?_ ?_
  · -- prologue: register-only assigns, safe at every state
    refine traceSafeListR_of_forall _ _ ?_
    intro st hst s'
    simp only [matmulPrologue, List.mem_cons, List.not_mem_nil, or_false] at hst
    rcases hst with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
      simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]
  · intro s1 hs1
    obtain ⟨s1x, hpre, hP0⟩ := preLoopW A B s BM BN BLOCK_K T
    rw [matmulPrologue_castFree R A B BM BN BLOCK_K s, hpre] at hs1
    obtain rfl := Option.some.inj hs1
    refine Stmt.TraceSafeListR.cons_intro ?_ ?_
    · -- the K-loop is trace-safe (invariant principle over the weak invariant)
      simp only [Stmt.TraceSafeR]
      exact Stmt.forRangeTraceSafeR_inv R bounds "k" T 1 (matmulLoopBody BM BN BLOCK_K)
        (matmulSafeInv A B s BM BN BLOCK_K T) hstep 0 s1x hP0
    · intro s2 hs2
      obtain ⟨final, sLoop, hLoopStmt, hfinal, hPL⟩ :=
        forRange_inv (idx := "k") (start := 0) (stop := T) (step := 1)
          (body := matmulLoopBody BM BN BLOCK_K)
          (by omega) hP0
          (fun c st hlt hinv => matmul_stepW A B s BM BN BLOCK_K T c st hlt hinv)
      rw [stepStmtR_forRange',
        stepForRangeAuxR_castFree R _ (matmulBody_castFree R BM BN BLOCK_K) "k",
        ← stepForRangeAux.forRange_unfold, hLoopStmt] at hs2
      obtain rfl := Option.some.inj hs2
      obtain ⟨-, hpmL, hpnL, hzL, -, -⟩ := hPL
      exact matmul_tailSafeR R bounds C s BM BN sLoop hpmL hpnL hzL hbC

/-- The full matmul surface sits inside the flat-memory bridge's covered
fragment (`FlattenOk`; the `forRange` clause recurses into the cast-free
body). -/
theorem matmul_kernel_flattenOk (C A B : RegionName) (BM BN BLOCK_K T : Nat) :
    ((matmul_kernel_surface C A B BM BN BLOCK_K T).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  rw [matmul_body_split' C A B BM BN BLOCK_K T]
  simp [matmulPrologue, matmulLoopBody, matmulStoreTail, StmtList.FlattenOk,
    Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-! ### IO signature, lane bridges, spec bridge -/

/-- The `A`-stream lane feeding output lane `l` at inner key `e`: the row of
`l` (row-major over the `[BM, BN]` output tile) paired with `e` over the
`[BM, BLOCK_K]` per-step `A`-tile, both via the shared `Lane2D` bridge. -/
def aLane (BM BN BK : Nat) (l : Fin (BM * BN)) (e : Fin BK) : Fin (BM * BK) :=
  Lane2D.encode ((Lane2D.decode l).1, e, PUnit.unit)

/-- The `B`-stream lane feeding output lane `l` at inner key `e`: `e` paired
with the column of `l` over the `[BLOCK_K, BN]` per-step `B`-tile. -/
def bLane (BM BN BK : Nat) (l : Fin (BM * BN)) (e : Fin BK) : Fin (BK * BN) :=
  Lane2D.encode (e, (Lane2D.decode l).2.1, PUnit.unit)

/-- **Streaming IO signature** of `matmul_kernel` on the two-stream fold skin
(S1: fold + terminal store). Step `t` of the K-loop reads the `[BM, BLOCK_K]`
`A`-tile and the `[BLOCK_K, BN]` `B`-tile; after the loop one `[BM, BN]`
output tile is stored at the **fp16** grid (`outDType := .fp16` — the
kernel's `tl.cast(…, tl.float16)` + fp16 store). The windows transcribe the
kernel's pointer arithmetic exactly:

* `read1` lane `l = (i, e)` (row-major over `[BM, BLOCK_K]`), step `t`:
  `((pid₀·BM + i) % 4096) · 4096 + (t·BLOCK_K + e)` — the invariant's
  `a_ptrs` cell `offs_am(i)·4096 + e·1 + t·BLOCK_K` after `t` advances.
* `read2` lane `l = (e, j)` (row-major over `[BLOCK_K, BN]`), step `t`:
  `(t·BLOCK_K + e) · 4096 + (pid₁·BN + j) % 4096` — the `b_ptrs` cell.
* `write` lane `l = (i, j)`: `4096·(pid₀·BM + i) + 1·(pid₁·BN + j)` — the
  kernel's un-wrapped `c_ptrs` (= `cOffset` in pid form).

The kernel has no masks: all windows are `True`. -/
def matmulKernelIO (C A B : RegionName) (BM BN BK T : Nat) : StreamMasked2DKernelIO₂ where
  kernel := matmul_kernel_surface C A B BM BN BK T
  inp1 := A
  inp2 := B
  out := C
  T := T
  B1 := BM * BK
  B2 := BK * BN
  C := BM * BN
  outDType := .fp16
  read1 := fun p₀ _ t l => ((p₀ * BM + l.val / BK) % 4096) * 4096 + (t.val * BK + l.val % BK)
  read2 := fun _ p₁ t l => (t.val * BK + l.val / BN) * 4096 + (p₁ * BN + l.val % BN) % 4096
  write := fun p₀ p₁ l => 4096 * (p₀ * BM + l.val / BN) + 1 * (p₁ * BN + l.val % BN)
  mask1 := fun _ _ _ _ => True
  mask2 := fun _ _ _ _ => True
  writeMask := fun _ _ _ => True

/-- Under the two stream pins, `matmulSpec` at the decoded output lane **is**
the skin-level double fold `∑ t, ∑ e, xs · ys` (`gemmSum_blocks` +
address-identity of the windows with the invariant's pointer shapes). -/
private theorem matmulSpec_eq_streamSum (A B : RegionName) (s₀ : BlockState)
    (BM BN BK T : Nat)
    (xs : Fin T → Fin (BM * BK) → ℝ) (ys : Fin T → Fin (BK * BN) → ℝ)
    (hx : ∀ (t : Fin T) (j : Fin (BM * BK)),
      s₀.readMem A (((s₀.pids 0 * BM + j.val / BK) % 4096) * 4096 + (t.val * BK + j.val % BK))
        = xs t j)
    (hy : ∀ (t : Fin T) (j : Fin (BK * BN)),
      s₀.readMem B ((t.val * BK + j.val / BN) * 4096 + (s₀.pids 1 * BN + j.val % BN) % 4096)
        = ys t j)
    (l : Fin (BM * BN)) :
    matmulSpec s₀ A B BM BN BK T (Lane2D.decode l).1 (Lane2D.decode l).2.1
      = ∑ t : Fin T, ∑ e : Fin BK, xs t (aLane BM BN BK l e) * ys t (bLane BM BN BK l e) := by
  unfold matmulSpec
  rw [Nat.mul_comm BK T, gemmSum_blocks]
  refine Finset.sum_congr rfl fun t _ => Finset.sum_congr rfl fun e _ => ?_
  have hxa : aElem s₀ A BM (Lane2D.decode l).1 (t.val * BK + e.val)
      = xs t (aLane BM BN BK l e) := by
    rw [← hx t (aLane BM BN BK l e)]
    simp only [aLane, Lane2D.encode_div, Lane2D.encode_mod]
    rfl
  have hyb : bElem s₀ B BN (Lane2D.decode l).2.1 (t.val * BK + e.val)
      = ys t (bLane BM BN BK l e) := by
    rw [← hy t (bLane BM BN BK l e)]
    simp only [bLane, Lane2D.encode_div, Lane2D.encode_mod]
    rfl
  rw [hxa, hyb]

/-! ### The headline -/

set_option maxHeartbeats 2000000 in
/-- **The `⊨[R]` streaming headline (wave-5 S1 fold genre; first streaming
`⊨[R]` in the bench).** For every rounding model `R`, the faithful
`matmul_kernel` surface implements, on its `StreamMasked2DKernelIO₂`
signature, the **ideal ℝ GEMM fold** over the streamed tiles: output lane
`l = (i, j)` holds `∑ t, ∑ e, A-tile[t](i,e) · B-tile[t](e,j)` — the spec `f`
is exact real arithmetic; the single boundary quantization is carried by the
skin's readback contract (`readMemAs .fp16` holds
`fp16.ofReal (R.round .fp16 (f …))`), where the kernel's two rounding events
(the `tl.cast(…, tl.float16)` and the fp16-typed store) collapse to one
`R.round .fp16` by the defining `round_idem`.

Layer map: the prologue and the whole K-loop are cast-free, so under
`execR R` they collapse verbatim onto the exact stepper and the proven
`preLoop` / `matmul_step` / `forRange_inv` stack above is reused unchanged;
only the 5-statement store tail is re-proved on the `R` side
(`matmul_postLoopR`).

The single hypothesis `hBN : BN ≤ 4096` is truth-forced: the output window
`4096·row + col` is injective only when the column-block width fits the
hard-coded row stride `4096` (`matmul_kernel_output_offset_injective` /
`rowMajor2D_inj`); with colliding output lanes the per-lane readback would be
last-writer-wins and the statement false. It holds for every valid tiling.

Relation to the exact surface: the exact headline
`matmul_kernel_closed_form_correct` (`Realizes_without_Rounding`) above is
retained unchanged; this `⊨[R]` face strictly generalizes its content — at
`R := .triv` the readback contract degenerates to the exact fp16-cast cell of
the same GEMM value. Both faces are kept per the rounding-as-default
doctrine. -/
specification matmul_kernel_io_correctness (R : RoundingModel)
    (C A B : RegionName) (BM BN BK T : Nat) (hBN : BN ≤ 4096) :
    matmulKernelIO C A B BM BN BK T ⊨[R] fun _ _ xs ys l =>
      ∑ t : Fin T, ∑ e : Fin BK, xs t (aLane BM BN BK l e) * ys t (bLane BM BN BK l e) := by
  refine StreamMasked2DKernelIO₂.ImplementsR.intro _ ?_ ?_ ?_
  · exact matmul_kernel_flattenOk C A B BM BN BK T
  · -- safety walk
    intro bounds s xs ys _hx _hy hbr1 hbr2 hbw
    simp only [matmulKernelIO] at hbr1 hbr2 hbw ⊢
    exact matmul_kernel_traceSafeR R bounds C A B BM BN BK T s
      (fun t j => hbr1 t j trivial) (fun t j => hbr2 t j trivial)
      (fun j => hbw j trivial)
  · -- the rounded Hoare triple
    intro s₀ xs ys hundef hx hy
    simp only [matmulKernelIO] at hx hy ⊢
    have hundef' : ∀ rg o, s₀.undef rg o = 0 := fun rg o => by rw [hundef]
    have hInj : Function.Injective (cOffset s₀ BM BN) :=
      matmul_kernel_output_offset_injective s₀ hBN
    -- exact preLoop + K-loop (cast-free, so they are the `execR` run too)
    obtain ⟨s1, hpre, hP0⟩ := preLoop C A B s₀ BM BN BK T hundef'
    obtain ⟨final, sLoop, hLoopStmt, hfinal, hPLoop⟩ :=
      forRange_inv (idx := "k") (start := 0) (stop := T) (step := 1)
        (by omega) hP0
        (fun c st hlt hinv => by
          simpa using matmul_step A B s₀ BM BN BK T c st hlt hinv)
    have hfinalEq : final = T := by
      have hle : final ≤ T := by
        simp only [matmulInvariant] at hPLoop
        exact hPLoop.2.1
      exact le_antisymm hle hfinal
    rw [hfinalEq] at hPLoop
    have hmem0 : sLoop.mem = s₀.mem := by
      have h := hPLoop
      simp only [matmulInvariant] at h
      exact h.2.2.2.2.2.2.2.2.2.2.2
    -- R-side tail
    obtain ⟨sfin, hTailR, hval, hframe⟩ :=
      matmul_postLoopR R C A B s₀ BM BN BK T hInj sLoop hPLoop
    have hLoopR : stepStmtR R (Stmt.forRange "k" 0 T 1 (matmulLoopBody BM BN BK)) s1
        = some sLoop := by
      rw [stepStmtR_forRange',
        stepForRangeAuxR_castFree R _ (matmulBody_castFree R BM BN BK) "k",
        ← stepForRangeAux.forRange_unfold]
      exact hLoopStmt
    have hpre' : stepStmts (matmulPrologue A B BM BN BK) s₀ = some s1 := by
      rw [← matmul_take8_eq C A B BM BN BK T]
      exact hpre
    refine ⟨sfin, ?_, ?_, ?_⟩
    · show execR R (matmul_kernel_surface C A B BM BN BK T).toAlgKernel s₀ = some sfin
      unfold execR
      rw [matmul_body_split' C A B BM BN BK T, stepStmtsR_append,
        matmulPrologue_castFree R A B BM BN BK s₀, hpre', Option.bind_some,
        stepStmtsR_cons_some hLoopR]
      exact hTailR
    · intro l _
      have hcell := hval (Lane2D.decode l)
      have haddr : 4096 * (s₀.pids 0 * BM + l.val / BN) + 1 * (s₀.pids 1 * BN + l.val % BN)
          = cOffset s₀ BM BN (Lane2D.decode l) := rfl
      rw [haddr, readMemAs_fp16_of_cell hcell,
        matmulSpec_eq_streamSum A B s₀ BM BN BK T xs ys
          (fun t j => hx t j trivial) (fun t j => hy t j trivial) l]
    · intro r o hcond
      have hcond' : r ≠ C ∨ ∀ idx : TileIndex [BM, BN], o ≠ cOffset s₀ BM BN idx := by
        rcases hcond with hne | hno
        · exact Or.inl hne
        · refine Or.inr fun idx => ?_
          have h := hno (Lane2D.encode idx) trivial
          simpa [Lane2D.encode_div, Lane2D.encode_mod, cOffset, rowGlobal, colGlobal] using h
      rw [hframe r o hcond', hmem0]

end VeriTile.Bench.TritonBenchG.MatmulKernel
