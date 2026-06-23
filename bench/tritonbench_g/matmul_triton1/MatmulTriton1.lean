import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.Kernel

/-!
# `matmul_triton1` — closed-form GEMM correctness

`matmul_kernel` is a tiled GEMM with a linearized program id: program `pid` is
split into an `(m_block, n_block)` tile coordinate, accumulates a
`m_block_size × n_block_size` output tile by looping over K with
`z += tl.dot(x_sub, y_sub, allow_tf32=False)`, and stores the tile into `z_ptr`.

This file proves the **full K-loop** correct against a genuine mathematical
matrix product: every output cell `C[i,j]` of the computed tile equals
`Σ_{k < K} X[i,k] · Y[k,j]` over `ℝ`, where `K = BLOCK_K · numKBlocks` is the
contracted dimension. This is NOT the kernel's own emitted value — it is the
independent closed-form `∑_k X·Y` GEMM reference, derived from the loaded
`X`/`Y` tiles.

## Proof architecture

```
matmul_triton1_closed_form_correct                ← TOP THEOREM (ComputeCorrect.Realizes)
  └─ matmul_triton1_exec_closed_form              ← exec-side closed form (every cell = ∑_k X·Y)
       ├─ preLoop      (P 0: z = 0, pointers seeded)
       ├─ matmul_step         (one K-block: z += dot advances the partial sum)
       ├─ matmul_postLoop     (final unmasked store = the closed form)
       └─ forRange_inv        (loop-invariant principle, drives the K-loop)
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float). The host launch (grid,
linear-pid scheduling) is
the trusted boundary; the per-program statement is universally quantified over
`s`, so it covers every program of the grid. The layout contract is the kernel's
own contiguity: `x[i,k]` at `X + i·k_size + k`, `y[k,j]` at `Y + k·n_size + j`,
`z[i,j]` at `Z + i·n_size + j` (row-major), exactly as the kernel's pointer
arithmetic constructs them.
-/

namespace VeriTile.Bench.TritonBenchG.MatmulTriton1

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Faithful transcription of `matmul_triton1.py`'s `matmul_kernel`.

Python passes `m_size` but the kernel body does not use it; this surface keeps
the signature position as `_m_size`. -/
def matmul_triton1_surface
    (X Y Z : RegionName)
    (_m_size k_size n_size m_block_size k_block_size n_block_size : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  num_n_blocks = tl.cdiv($(n_size), $(n_block_size))
  m_block = pid // num_n_blocks
  n_block = pid % num_n_blocks
  m_offsets = tl.arange(0, $(m_block_size)) + m_block * $(m_block_size)
  n_offsets = tl.arange(0, $(n_block_size)) + n_block * $(n_block_size)
  k_offsets = tl.arange(0, $(k_block_size))
  x_ptrs = X + m_offsets[:, None] * $(k_size) + k_offsets[None, :]
  y_ptrs = Y + k_offsets[:, None] * $(n_size) + n_offsets[None, :]
  z_ptrs = Z + m_offsets[:, None] * $(n_size) + n_offsets[None, :]
  z = tl.zeros([$(m_block_size), $(n_block_size)], dtype=tl.float32)
  for kk in range($(0), $(k_size), $(k_block_size)) {
    x_sub = tl.load(x_ptrs)
    y_sub = tl.load(y_ptrs)
    z += tl.dot(x_sub, y_sub, allow_tf32=false)
    x_ptrs += $(k_block_size)
    y_ptrs += $(k_block_size) * $(n_size)
  }
  tl.store(z_ptrs, z)
}

/-- The full `matmul_triton1` surface lowers to the algorithm layer. -/
theorem matmul_triton1_surface_toAlgorithm_supported
    (X Y Z : RegionName)
    (m_size k_size n_size m_block_size k_block_size n_block_size : Nat) :
    ∃ alg, (matmul_triton1_surface X Y Z m_size k_size n_size m_block_size
      k_block_size n_block_size).toAlgorithm? = Except.ok alg := by
  simp [matmul_triton1_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-! ## exec-stepping helpers (ported from `AttentionForwardClosedForm`) -/

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

/-- `x_ptrs` eval: cell `(i,e) = (X, m_offsets i · KS + e)`. -/
theorem xptrs_eval (s : BlockState) (X : RegionName) (M K KS : Nat) (gm : Fin M → Nat)
    (hm : s.regs .nat [M] "m_offsets" = some (Tile.vec gm))
    (hk : s.regs .nat [K] "k_offsets" = some (Tile.vec (fun e : Fin K => e.val))) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase X)
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "m_offsets")) (Op.constNat KS))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [K] "k_offsets")))) s
      = some (⟨fun idx : TileIndex [M, K] => (X.cast, gm idx.1 * KS + idx.2.1.val)⟩ : Tile .ptr [M, K]) := by
  rw [evalOp_ptrAdd, evalOp_ptrBase]
  simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref, evalOp_expandDim_one_nat,
    evalOp_expandDim_zero_nat, hm, hk, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec]
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec, NumericDType.add, NumericDType.mul]

/-- `y_ptrs` eval: cell `(e,j) = (Y, k_offsets e · NS + n_offsets j)`. -/
theorem yptrs_eval (s : BlockState) (Y : RegionName) (K N NS : Nat) (gn : Fin N → Nat)
    (hk : s.regs .nat [K] "k_offsets" = some (Tile.vec (fun e : Fin K => e.val)))
    (hn : s.regs .nat [N] "n_offsets" = some (Tile.vec gn)) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Y)
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [K] "k_offsets")) (Op.constNat NS))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [N] "n_offsets")))) s
      = some (⟨fun idx : TileIndex [K, N] => (Y.cast, idx.1.val * NS + gn idx.2.1)⟩ : Tile .ptr [K, N]) := by
  rw [evalOp_ptrAdd, evalOp_ptrBase]
  simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref, evalOp_expandDim_one_nat,
    evalOp_expandDim_zero_nat, hk, hn, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec]
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec, NumericDType.add, NumericDType.mul]

/-- `z_ptrs` eval: cell `(i,j) = (Z, m_offsets i · NS + n_offsets j)`. -/
theorem zptrs_eval (s : BlockState) (Z : RegionName) (M N NS : Nat) (gm : Fin M → Nat) (gn : Fin N → Nat)
    (hm : s.regs .nat [M] "m_offsets" = some (Tile.vec gm))
    (hn : s.regs .nat [N] "n_offsets" = some (Tile.vec gn)) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Z)
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "m_offsets")) (Op.constNat NS))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [N] "n_offsets")))) s
      = some (⟨fun idx : TileIndex [M, N] => (Z.cast, gm idx.1 * NS + gn idx.2.1)⟩ : Tile .ptr [M, N]) := by
  rw [evalOp_ptrAdd, evalOp_ptrBase]
  simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref, evalOp_expandDim_one_nat,
    evalOp_expandDim_zero_nat, hm, hn, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec]
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec, NumericDType.add, NumericDType.mul]

/-- `z` init eval: `tl.zeros` → the all-`0` tile. -/
theorem z_init_eval (s : BlockState) (M N : Nat) :
    evalOp (Op.full [M, N] (Op.const 0)) s
      = some (⟨fun _ : TileIndex [M, N] => some (0 : ℝ)⟩ : Tile .real [M, N]) := by
  simp [evalOp_full, evalOp_const, Option.bind]

/-- `x_ptrs += k_block_size` eval (scalar `BLOCK_K` add via `scalarR`). -/
theorem xptr_adv_eval (s : BlockState) (M K BK : Nat) (xp : Tile .ptr [M, K])
    (hx : s.regs .ptr [M, K] "x_ptrs" = some xp) :
    evalOp (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [M, K] "x_ptrs") (Op.constNat BK)) s
      = some (Tile.ptrAdd Broadcast.scalarR xp (Tile.scalar BK)) := by
  rw [evalOp_ptrAdd]; simp [evalOp_ref, hx, evalOp_constNat]

/-- `y_ptrs += k_block_size · n_size` eval. -/
theorem yptr_adv_eval (s : BlockState) (K N BK NS : Nat) (yp : Tile .ptr [K, N])
    (hy : s.regs .ptr [K, N] "y_ptrs" = some yp) :
    evalOp (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [K, N] "y_ptrs")
      (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat NS))) s
      = some (Tile.ptrAdd Broadcast.scalarR yp (Tile.scalar (BK * NS))) := by
  rw [evalOp_ptrAdd]
  simp [evalOp_ref, hy, evalOp_mul, evalOp_constNat, NumericDType.mul, Tile.bop]

/-- **`z = z + tl.dot(x_sub, y_sub)` statement eval.** -/
theorem zdot_op_eval (M K N : Nat) (st : BlockState)
    (zt : Tile .real [M, N]) (xt : Tile .real [M, K]) (yt : Tile .real [K, N])
    (hz : st.regs .real [M, N] "z" = some zt)
    (hx : st.regs .real [M, K] "x_sub" = some xt)
    (hy : st.regs .real [K, N] "y_sub" = some yt) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [M, N] "z")
        (Op.dot (batch := []) (Op.ref .real [M, K] "x_sub") (Op.ref .real [K, N] "y_sub"))) st
      = some (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          zt (Tile.dot [] xt yt)) := by
  have hd : evalOp (Op.dot (batch := []) (Op.ref .real [M, K] "x_sub")
        (Op.ref .real [K, N] "y_sub")) st = some (Tile.dot [] xt yt) := by
    rw [evalOp_dot]; simp [hx, hy]
  rw [evalOp_add]
  simp only [evalOp_ref, hz, bind, Option.bind_some]
  erw [hd]
  rfl

/-- `z + dot` lane `(i,j)`: `some (zv + dv)`. -/
theorem zadd_eval (M N : Nat) (zt dt : Tile .real [M, N]) (i : Fin M) (j : Fin N) (zv dv : ℝ)
    (hz : zt.data (i, j, PUnit.unit) = some zv) (hd : dt.data (i, j, PUnit.unit) = some dv) :
    (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) zt dt).data
        (i, j, PUnit.unit) = some (zv + dv) := by
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, hz, hd, NumericDType.add,
    WithBot.realAdd, Option.map₂, Option.bind, Option.map]

/-! ## GEMM closed-form spec -/

/-- `num_n_blocks = cdiv n_size n_block_size` (the kernel's tile column count). -/
def numNBlocks (n_size n_block_size : Nat) : Nat := cdiv n_size n_block_size

/-- Global output row of tile lane `i`: `(pid / num_n_blocks) · M + i`. -/
def rowIndex (s : BlockState) (NS NB M : Nat) (i : Fin M) : Nat :=
  s.pids 0 / numNBlocks NS NB * M + i.val

/-- Global output column of tile lane `j`: `(pid % num_n_blocks) · N + j`. -/
def colIndex (s : BlockState) (NS NB N : Nat) (j : Fin N) : Nat :=
  s.pids 0 % numNBlocks NS NB * N + j.val

/-- `X[i, k] = readMem X (rowIndex i · KS + k)` (kernel's row-major X layout). -/
noncomputable def xElem (s : BlockState) (X : RegionName) (KS NS NB M : Nat)
    (i : Fin M) (k : Nat) : ℝ :=
  s.readMem X (rowIndex s NS NB M i * KS + k)

/-- `Y[k, j] = readMem Y (k · NS + colIndex j)` (kernel's row-major Y layout). -/
noncomputable def yElem (s : BlockState) (Y : RegionName) (NS NB N : Nat)
    (j : Fin N) (k : Nat) : ℝ :=
  s.readMem Y (k * NS + colIndex s NS NB N j)

/-- **Genuine GEMM spec**: `C[i,j] = Σ_{k < BLOCK_K·numKBlocks} X[i,k] · Y[k,j]`,
an instance of the shared `gemmSum` (`Kernel.Matmul`) with this kernel's `X`/`Y`
layout accessors. -/
noncomputable def matmulSpec (s : BlockState) (X Y : RegionName)
    (KS NS NB M N BLOCK_K numKBlocks : Nat) (i : Fin M) (j : Fin N) : ℝ :=
  gemmSum (xElem s X KS NS NB M i) (yElem s Y NS NB N j) (BLOCK_K * numKBlocks)

/-- Partial GEMM accumulator after `c` K-blocks: `Σ_{k < c·BLOCK_K} X·Y`. -/
noncomputable def accPartial (s : BlockState) (X Y : RegionName)
    (KS NS NB M N BLOCK_K : Nat) (i : Fin M) (j : Fin N) (c : Nat) : ℝ :=
  gemmSum (xElem s X KS NS NB M i) (yElem s Y NS NB N j) (c * BLOCK_K)

/-- One-block step of the partial accumulator: the new block's dot is over the
`BLOCK_K` keys `c·BLOCK_K + e` (the shared `gemmSum_blockSucc`). -/
theorem accPartial_succ (s : BlockState) (X Y : RegionName)
    (KS NS NB M N BLOCK_K : Nat) (i : Fin M) (j : Fin N) (c : Nat) :
    accPartial s X Y KS NS NB M N BLOCK_K i j (c + 1)
      = accPartial s X Y KS NS NB M N BLOCK_K i j c
        + (Finset.univ.sum fun e : Fin BLOCK_K =>
            xElem s X KS NS NB M i (c * BLOCK_K + e.val)
              * yElem s Y NS NB N j (c * BLOCK_K + e.val)) :=
  gemmSum_blockSucc (xElem s X KS NS NB M i) (yElem s Y NS NB N j) BLOCK_K c

/-! ## Body decomposition (prefix ++ for-loop ++ store) -/

/-- The 5-statement K-loop body, transcribed. -/
def matmulLoopBody (M _K N _KS NS BLOCK_K : Nat) : List Stmt :=
  [ Stmt.assign .real [M, BLOCK_K] "x_sub"
      (Op.load .real (.ptr (Op.ref .ptr [M, BLOCK_K] "x_ptrs")) .none),
    Stmt.assign .real [BLOCK_K, N] "y_sub"
      (Op.load .real (.ptr (Op.ref .ptr [BLOCK_K, N] "y_ptrs")) .none),
    Stmt.assign .real [M, N] "z"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [M, N] "z")
        (Op.dot (batch := []) (Op.ref .real [M, BLOCK_K] "x_sub") (Op.ref .real [BLOCK_K, N] "y_sub"))),
    Stmt.assign .ptr [M, BLOCK_K] "x_ptrs"
      (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [M, BLOCK_K] "x_ptrs") (Op.constNat BLOCK_K)),
    Stmt.assign .ptr [BLOCK_K, N] "y_ptrs"
      (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BLOCK_K, N] "y_ptrs")
        (Op.mul .nat Broadcast.nil (Op.constNat BLOCK_K) (Op.constNat NS))) ]

/-- Post-loop unmasked store of `z` to `z_ptrs`. -/
def matmulStoreStmt (M N : Nat) : Stmt :=
  Stmt.store .real [M, N] (.ptr (Op.ref .ptr [M, N] "z_ptrs"))
    (Op.ref .real [M, N] "z") .none

/-- Body decomposition: prefix (11) ++ [for-loop, store]. By `rfl`. -/
theorem matmul_body_split (X Y Z : RegionName) (KS NS M BLOCK_K N numKBlocks : Nat) :
    (matmul_triton1_surface X Y Z 0 (BLOCK_K * numKBlocks) NS M BLOCK_K N).toAlgKernel.body
      = (matmul_triton1_surface X Y Z 0 (BLOCK_K * numKBlocks) NS M BLOCK_K N).toAlgKernel.body.take 11
        ++ (Stmt.forRange "kk" 0 (BLOCK_K * numKBlocks) BLOCK_K (matmulLoopBody M BLOCK_K N KS NS BLOCK_K)
            :: matmulStoreStmt M N :: []) := by
  rfl

/-! ## Loop invariant -/

/-- **Loop invariant** (counter `i = c · BLOCK_K`).

After `c` K-blocks: program ids and `mem`/`undef` fixed; the `m_offsets` /
`n_offsets` / `k_offsets` / `z_ptrs` registers seeded; `z` equals the partial
GEMM accumulator `accPartial … c`; and `x_ptrs` / `y_ptrs` advanced by `c`
blocks. -/
noncomputable def matmulInvariant
    (X Y Z : RegionName) (s0 : BlockState) (KS NS NB M N BLOCK_K numKBlocks : Nat)
    (i : Nat) (s : BlockState) : Prop :=
  let c := i / BLOCK_K
  s.pids = s0.pids ∧ i = c * BLOCK_K ∧ c ≤ numKBlocks ∧
  (s.regs .real [M, N] "z" = some ⟨fun idx : TileIndex [M, N] =>
      some (accPartial s0 X Y KS NS NB M N BLOCK_K idx.1 idx.2.1 c)⟩) ∧
  (s.regs .nat [M] "m_offsets" = some (Tile.vec (fun r : Fin M => rowIndex s0 NS NB M r))) ∧
  (s.regs .nat [N] "n_offsets" = some (Tile.vec (fun j : Fin N => colIndex s0 NS NB N j))) ∧
  (s.regs .nat [BLOCK_K] "k_offsets" = some (Tile.vec (fun e : Fin BLOCK_K => e.val))) ∧
  (s.regs .ptr [M, N] "z_ptrs" = some ⟨fun idx : TileIndex [M, N] =>
      (Z.cast, rowIndex s0 NS NB M idx.1 * NS + colIndex s0 NS NB N idx.2.1)⟩) ∧
  (s.regs .ptr [M, BLOCK_K] "x_ptrs" = some ⟨fun idx : TileIndex [M, BLOCK_K] =>
      (X.cast, rowIndex s0 NS NB M idx.1 * KS + idx.2.1.val + c * BLOCK_K)⟩) ∧
  (s.regs .ptr [BLOCK_K, N] "y_ptrs" = some ⟨fun idx : TileIndex [BLOCK_K, N] =>
      (Y.cast, idx.1.val * NS + colIndex s0 NS NB N idx.2.1 + c * BLOCK_K * NS)⟩) ∧
  (∀ rg o, s.undef rg o = 0) ∧ (s.mem = s0.mem)

/-- **preLoop scalars** (statements 0–6): the 4 scalar offsets + the 3 index
vectors. Steps to a state with the `m_offsets`/`n_offsets`/`k_offsets`
readbacks the pointer chunk needs. -/
theorem preLoop_scalars (s : BlockState) (_KS NS M K N NB : Nat) :
    ∃ s7, stepStmts [ Stmt.assign .nat [] "pid" (Op.programId 0),
          Stmt.assign .nat [] "num_n_blocks"
            (Op.div .nat Broadcast.nil (Op.sub .nat Broadcast.nil
              (Op.add .nat Broadcast.nil (Op.constNat NS) (Op.constNat NB)) (Op.constNat 1)) (Op.constNat NB)),
          Stmt.assign .nat [] "m_block" (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "pid") (Op.ref .nat [] "num_n_blocks")),
          Stmt.assign .nat [] "n_block" (Op.mod .nat Broadcast.nil (Op.ref .nat [] "pid") (Op.ref .nat [] "num_n_blocks")),
          Stmt.assign .nat [M] "m_offsets"
            (Op.add .nat Broadcast.scalarR (Op.arange M)
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "m_block") (Op.constNat M))),
          Stmt.assign .nat [N] "n_offsets"
            (Op.add .nat Broadcast.scalarR (Op.arange N)
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "n_block") (Op.constNat N))),
          Stmt.assign .nat [K] "k_offsets" (Op.arange K) ] s = some s7
      ∧ s7.pids = s.pids
      ∧ s7.regs .nat [M] "m_offsets" = some (Tile.vec (fun i : Fin M => rowIndex s NS NB M i))
      ∧ s7.regs .nat [N] "n_offsets" = some (Tile.vec (fun j : Fin N => colIndex s NS NB N j))
      ∧ s7.regs .nat [K] "k_offsets" = some (Tile.vec (fun e : Fin K => e.val))
      ∧ s7.undef = s.undef
      ∧ s7.mem = s.mem := by
  simp only [rowIndex, colIndex, numNBlocks, cdiv]
  simp [stepStmts, stepStmt, evalOp_floorDiv, evalOp_mod, Option.bind, BlockState.setReg,
    Tile.bop, Tile.vec, NumericDType.add, NumericDType.mul, NumericDType.div, NumericDType.sub]
  constructor <;> (funext i; ring)

set_option maxHeartbeats 1000000 in
/-- **preLoop** (statements 0–10): from a clean input state (`undef = 0`), the
prologue steps to a state satisfying `matmulInvariant … 0` — the base case
(`z = 0`, pointers seeded). -/
theorem preLoop (X Y Z : RegionName) (s : BlockState)
    (NS M BLOCK_K N numKBlocks : Nat) (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ s', stepStmts ((matmul_triton1_surface X Y Z 0 (BLOCK_K * numKBlocks) NS M BLOCK_K N).toAlgKernel.body.take 11) s = some s'
      ∧ matmulInvariant X Y Z s (BLOCK_K * numKBlocks) NS N M N BLOCK_K numKBlocks 0 s' := by
  obtain ⟨s7, h7, hpids, hm, hn, hk, huf, hmem⟩ := preLoop_scalars s (BLOCK_K * numKBlocks) NS M BLOCK_K N N
  rw [show ((matmul_triton1_surface X Y Z 0 (BLOCK_K * numKBlocks) NS M BLOCK_K N).toAlgKernel.body.take 11)
      = [ Stmt.assign .nat [] "pid" (Op.programId 0),
          Stmt.assign .nat [] "num_n_blocks"
            (Op.div .nat Broadcast.nil (Op.sub .nat Broadcast.nil
              (Op.add .nat Broadcast.nil (Op.constNat NS) (Op.constNat N)) (Op.constNat 1)) (Op.constNat N)),
          Stmt.assign .nat [] "m_block" (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "pid") (Op.ref .nat [] "num_n_blocks")),
          Stmt.assign .nat [] "n_block" (Op.mod .nat Broadcast.nil (Op.ref .nat [] "pid") (Op.ref .nat [] "num_n_blocks")),
          Stmt.assign .nat [M] "m_offsets"
            (Op.add .nat Broadcast.scalarR (Op.arange M)
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "m_block") (Op.constNat M))),
          Stmt.assign .nat [N] "n_offsets"
            (Op.add .nat Broadcast.scalarR (Op.arange N)
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "n_block") (Op.constNat N))),
          Stmt.assign .nat [BLOCK_K] "k_offsets" (Op.arange BLOCK_K) ]
      ++ [ Stmt.assign .ptr [M, BLOCK_K] "x_ptrs"
            (Op.ptrAdd Broadcast.scalarL (Op.ptrBase X)
              (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
                (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "m_offsets")) (Op.constNat (BLOCK_K * numKBlocks)))
                (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_K] "k_offsets")))),
          Stmt.assign .ptr [BLOCK_K, N] "y_ptrs"
            (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Y)
              (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
                (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_K] "k_offsets")) (Op.constNat NS))
                (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [N] "n_offsets")))),
          Stmt.assign .ptr [M, N] "z_ptrs"
            (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Z)
              (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
                (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "m_offsets")) (Op.constNat NS))
                (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [N] "n_offsets")))),
          Stmt.assign .real [M, N] "z" (Op.full [M, N] (Op.const 0)) ] from rfl,
    stepStmts.append_some h7,
    stepStmts.cons_some (stepStmt_assign_eq_some
      (xptrs_eval s7 X M BLOCK_K (BLOCK_K * numKBlocks) (fun i => rowIndex s NS N M i) (by simpa using hm) (by simpa using hk))),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (yptrs_eval _ Y BLOCK_K N NS (fun j => colIndex s NS N N j) (by simp [hk]) (by simp [hn]))),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (zptrs_eval _ Z M N NS (fun i => rowIndex s NS N M i) (fun j => colIndex s NS N N j) (by simp [hm]) (by simp [hn]))),
    stepStmts.cons_some (stepStmt_assign_eq_some (z_init_eval _ M N)),
    stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  refine ⟨by simp [hpids], by simp, by simp, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · -- z = accPartial … 0 = 0
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx
    simp only [accPartial, Nat.zero_div, Nat.zero_mul, gemmSum_zero]
  · -- m_offsets
    simp [hm]
  · -- n_offsets
    simp [hn]
  · -- k_offsets
    simp [hk]
  · -- z_ptrs
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
  · -- x_ptrs (c = 0)
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx <;> simp [Nat.zero_mul]
  · -- y_ptrs (c = 0)
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx <;> simp [Nat.zero_mul]
  · -- undef = 0
    intro rg o; simp [huf, hundef]
  · -- mem
    exact hmem

set_option maxHeartbeats 2000000 in
/-- **Step lemma**: one K-loop body iteration advances the invariant by one
block (`acc += tl.dot(x_sub, y_sub)` adds the `c`-th block's dot to the partial
GEMM accumulator; the `x`/`y` pointers advance one step). -/
theorem matmul_step (X Y Z : RegionName) (s0 : BlockState)
    (NS M BLOCK_K N numKBlocks : Nat) (hBK : 0 < BLOCK_K)
    (i : Nat) (s : BlockState) (hilt : i < BLOCK_K * numKBlocks)
    (hinv : matmulInvariant X Y Z s0 (BLOCK_K * numKBlocks) NS N M N BLOCK_K numKBlocks i s) :
    ∃ s', stepStmts (matmulLoopBody M BLOCK_K N (BLOCK_K * numKBlocks) NS BLOCK_K)
        (s.setReg "kk" .nat [] (Tile.scalar i)) = some s'
      ∧ matmulInvariant X Y Z s0 (BLOCK_K * numKBlocks) NS N M N BLOCK_K numKBlocks (i + BLOCK_K) s' := by
  set KS := BLOCK_K * numKBlocks with hKS
  have hc : i / BLOCK_K < numKBlocks := (Nat.div_lt_iff_lt_mul hBK).mpr (by rw [Nat.mul_comm]; exact hilt)
  have hc1 : (i + BLOCK_K) / BLOCK_K = i / BLOCK_K + 1 := Nat.add_div_right i hBK
  simp only [matmulInvariant] at hinv
  obtain ⟨hpids, hi, hcle, hz, hm, hn, hk, hzp, hxp, hyp, hundef, hmem⟩ := hinv
  set c := i / BLOCK_K with hcdef
  -- abbreviations for the seeded x/y pointer tiles
  set xpT : Tile .ptr [M, BLOCK_K] :=
    ⟨fun idx : TileIndex [M, BLOCK_K] => (X.cast, rowIndex s0 NS N M idx.1 * KS + idx.2.1.val + c * BLOCK_K)⟩ with hxpT
  set ypT : Tile .ptr [BLOCK_K, N] :=
    ⟨fun idx : TileIndex [BLOCK_K, N] => (Y.cast, idx.1.val * NS + colIndex s0 NS N N idx.2.1 + c * BLOCK_K * NS)⟩ with hypT
  set zT : Tile .real [M, N] :=
    ⟨fun idx : TileIndex [M, N] => some (accPartial s0 X Y KS NS N M N BLOCK_K idx.1 idx.2.1 c)⟩ with hzT
  -- the kk-set state
  set sk := s.setReg "kk" .nat [] (Tile.scalar i) with hsk
  have hrmem : ∀ (R : RegionName) (o : Nat), sk.readMem R o = s0.readMem R o := by
    intro R o; simp only [hsk, BlockState.setReg_readMem]; unfold BlockState.readMem; rw [hmem]
  -- register readbacks on sk
  have hxpk : sk.regs .ptr [M, BLOCK_K] "x_ptrs" = some xpT := by simp [hsk, hxp, hxpT]
  have hypk : sk.regs .ptr [BLOCK_K, N] "y_ptrs" = some ypT := by simp [hsk, hyp, hypT]
  have hzk : sk.regs .real [M, N] "z" = some zT := by simp [hsk, hz, hzT]
  have hundefk : ∀ rg o, sk.undef rg o = 0 := by intro rg o; simp [hsk, hundef]
  -- x_sub / y_sub loaded tiles
  set xsub : Tile .real [M, BLOCK_K] :=
    ⟨fun idx => some (sk.readMem (xpT.data idx).1 (xpT.data idx).2)⟩ with hxsub
  set ysub : Tile .real [BLOCK_K, N] :=
    ⟨fun idx => some (sk.readMem (ypT.data idx).1 (ypT.data idx).2)⟩ with hysub
  unfold matmulLoopBody
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (load_ptr_none_real (Op.ref .ptr [M, BLOCK_K] "x_ptrs") _ xpT (by rw [evalOp_ref]; simp [hxpk])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (load_ptr_none_real (Op.ref .ptr [BLOCK_K, N] "y_ptrs") _ ypT (by rw [evalOp_ref]; simp [hypk])))]
  -- z = z + dot(x_sub, y_sub)
  set zaddT : Tile .real [M, N] :=
    Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) zT
      (Tile.dot [] xsub ysub) with hzaddT
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (zdot_op_eval M BLOCK_K N _ zT xsub ysub
          (by simp [hzk, hxsub, hysub, BlockState.setReg_readMem])
          (by simp [hxsub, BlockState.setReg_readMem])
          (by simp [hysub, BlockState.setReg_readMem])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (xptr_adv_eval _ M BLOCK_K BLOCK_K xpT (by simp [hxpk])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (yptr_adv_eval _ BLOCK_K N BLOCK_K NS ypT (by simp [hypk])))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  simp only [matmulInvariant, hc1]
  refine ⟨by simp [hsk, hpids], by rw [hi]; ring, by omega, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · -- z = accPartial (c+1)
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx
    have hxs : ∀ e : Fin BLOCK_K, xsub.data (idx.1, e, PUnit.unit)
        = some (xElem s0 X KS NS N M idx.1 (c * BLOCK_K + e.val)) := by
      intro e
      simp only [hxsub, hxpT, hrmem, xElem]
      congr 2
      ring
    have hys : ∀ e : Fin BLOCK_K, ysub.data (e, idx.2.1, PUnit.unit)
        = some (yElem s0 Y NS N N idx.2.1 (c * BLOCK_K + e.val)) := by
      intro e
      simp only [hysub, hypT, hrmem, yElem]
      congr 2
      ring
    rw [zadd_eval M N zT (Tile.dot [] xsub ysub) idx.1 idx.2.1
        (accPartial s0 X Y KS NS N M N BLOCK_K idx.1 idx.2.1 c)
        (Finset.univ.sum fun e : Fin BLOCK_K =>
          xElem s0 X KS NS N M idx.1 (c * BLOCK_K + e.val) * yElem s0 Y NS N N idx.2.1 (c * BLOCK_K + e.val))
        (by rw [hzT])
        (tile_dot_data M BLOCK_K N xsub ysub idx.1 idx.2.1 _ _ hxs hys)]
    show some _ = some (accPartial s0 X Y KS NS N M N BLOCK_K idx.1 idx.2.1 (c + 1))
    rw [accPartial_succ]
  · simp [hsk, hm]
  · simp [hsk, hn]
  · simp [hsk, hk]
  · -- z_ptrs unchanged
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    simp [hsk, hzp]
  · -- x_ptrs advanced
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx <;> simp only [Tile.ptrAdd, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
      Tile.scalar, hxpT, NumericDType.add]
    ring
  · -- y_ptrs advanced
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx <;> simp only [Tile.ptrAdd, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
      Tile.scalar, hypT, NumericDType.add]
    ring
  · intro rg o; simp [hsk, hundef]
  · show _ = s0.mem
    rw [← hmem, hsk]; rfl

/-- The output store address for tile lane `(i,j)`: `rowIndex i · NS + colIndex j`. -/
def zOffset (s0 : BlockState) (NS N M : Nat) (idx : TileIndex [M, N]) : Nat :=
  rowIndex s0 NS N M idx.1 * NS + colIndex s0 NS N N idx.2.1

/-- **postLoop**: from the invariant at `numKBlocks` blocks, the unmasked store
of `z` to `z_ptrs` writes the genuine GEMM value `matmulSpec` at every output
lane (given the output-offset map is injective). -/
theorem matmul_postLoop (X Y Z : RegionName) (s0 : BlockState)
    (NS M BLOCK_K N numKBlocks : Nat) (hBK : 0 < BLOCK_K)
    (hInj : Function.Injective (zOffset s0 NS N M))
    (st : BlockState)
    (hinv : matmulInvariant X Y Z s0 (BLOCK_K * numKBlocks) NS N M N BLOCK_K numKBlocks
        (BLOCK_K * numKBlocks) st) :
    ∃ sfin, stepStmts (matmulStoreStmt M N :: []) st = some sfin
      ∧ ∀ idx : TileIndex [M, N],
          sfin.readMem Z (zOffset s0 NS N M idx)
            = matmulSpec s0 X Y (BLOCK_K * numKBlocks) NS N M N BLOCK_K numKBlocks idx.1 idx.2.1 := by
  have hc : (BLOCK_K * numKBlocks) / BLOCK_K = numKBlocks := by
    rw [Nat.mul_comm, Nat.mul_div_cancel _ hBK]
  simp only [matmulInvariant, hc] at hinv
  obtain ⟨hpids, hi, hcle, hz, hm, hn, hk, hzp, hxp, hyp, hundef, hmem⟩ := hinv
  set KS := BLOCK_K * numKBlocks with hKS
  set zT : Tile .real [M, N] :=
    ⟨fun idx : TileIndex [M, N] => some (accPartial s0 X Y KS NS N M N BLOCK_K idx.1 idx.2.1 numKBlocks)⟩ with hzT
  set zpT : Tile .ptr [M, N] :=
    ⟨fun idx : TileIndex [M, N] => (Z.cast, rowIndex s0 NS N M idx.1 * NS + colIndex s0 NS N N idx.2.1)⟩ with hzpT
  -- the store statement evaluates to the scatter foldl
  have hstore : stepStmt (matmulStoreStmt M N) st
      = some ((TileShape.allIndices [M, N]).foldl
          (fun acc i =>
            acc.writeMem Z (zOffset s0 NS N M i)
              (accPartial s0 X Y KS NS N M N BLOCK_K i.1 i.2.1 numKBlocks)) st) := by
    simp only [matmulStoreStmt, stepStmt]
    rw [show evalOp (Op.ref .real [M, N] "z") st = some zT from by rw [evalOp_ref, hz]]
    rw [show evalOp (Op.ref .ptr [M, N] "z_ptrs") st = some zpT from by rw [evalOp_ref, hzp]]
    simp only [bind, Option.bind_some]
    refine congrArg some (List.foldl_ext _ _ st (fun acc i _ => ?_))
    simp only [if_true, zpT, hzpT, zOffset, zT, hzT, BlockState.writeMemTyped_real, Region.cast_id,
      FloatDType.real_storeValue, WithBot.unbotD_some]
  rw [stepStmts.cons_some hstore, stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  intro idx
  rw [BlockState.scatter_readback_nd (region := Z) (s := st)
      (offsetFn := zOffset s0 NS N M)
      (valueFn := fun i => accPartial s0 X Y KS NS N M N BLOCK_K i.1 i.2.1 numKBlocks)
      hInj idx]
  simp only [matmulSpec, accPartial, Nat.mul_comm numKBlocks BLOCK_K]

/-! ## Composition: full exec closed form -/

set_option maxHeartbeats 2000000 in
/-- **Top exec reduction**: composes `preLoop` + `matmul_step` (driven by
`forRange_inv`) + `matmul_postLoop` into the full `exec` result. Every output
lane equals the genuine GEMM value `matmulSpec`. -/
theorem matmul_triton1_exec_closed_form (X Y Z : RegionName) (s : BlockState)
    (NS M BLOCK_K N numKBlocks : Nat) (hBK : 0 < BLOCK_K)
    (hInj : Function.Injective (zOffset s NS N M))
    (hundef : ∀ rg o, s.undef rg o = 0)
    (idx : TileIndex [M, N]) :
    (match exec (matmul_triton1_surface X Y Z 0 (BLOCK_K * numKBlocks) NS M BLOCK_K N) s with
      | some s' => s'.readMem Z (zOffset s NS N M idx)
      | none => (0.0 : ℝ)) =
      matmulSpec s X Y (BLOCK_K * numKBlocks) NS N M N BLOCK_K numKBlocks idx.1 idx.2.1 := by
  -- preLoop establishes P 0
  obtain ⟨s0, hpre_eq, hP0⟩ := preLoop X Y Z s NS M BLOCK_K N numKBlocks hundef
  -- drive the K-loop
  obtain ⟨final, sLoop, hLoopStmt, hfinal, hPLoop⟩ :=
    forRange_inv (idx := "kk") (start := 0) (stop := BLOCK_K * numKBlocks) (step := BLOCK_K)
      (by omega) hP0
      (fun i st hlt hinv => matmul_step X Y Z s NS M BLOCK_K N numKBlocks hBK i st hlt hinv)
  -- loop exit: final = BLOCK_K * numKBlocks
  have hfinalEq : final = BLOCK_K * numKBlocks := by
    have hle : final ≤ BLOCK_K * numKBlocks := by
      simp only [matmulInvariant] at hPLoop
      obtain ⟨_, hieq, hcle, _⟩ := hPLoop
      calc final = final / BLOCK_K * BLOCK_K := hieq
        _ ≤ numKBlocks * BLOCK_K := Nat.mul_le_mul_right _ hcle
        _ = BLOCK_K * numKBlocks := Nat.mul_comm _ _
    exact le_antisymm hle hfinal
  subst hfinalEq
  -- postLoop reads off the closed form
  obtain ⟨sfin, hTail, hpost⟩ := matmul_postLoop X Y Z s NS M BLOCK_K N numKBlocks hBK hInj sLoop hPLoop
  have hexec : exec (matmul_triton1_surface X Y Z 0 (BLOCK_K * numKBlocks) NS M BLOCK_K N) s
      = some sfin := by
    rw [exec, matmul_body_split X Y Z (BLOCK_K * numKBlocks) NS M BLOCK_K N numKBlocks,
      stepStmts.append_some hpre_eq, stepStmts.cons_some hLoopStmt, hTail]
  rw [hexec]
  exact hpost idx

/-- **Output-offset injectivity from a geometric well-formedness bound.**

`zOffset` is the row-major map `lane (i,j) ↦ rowIndex i · n_size + colIndex j`.
It is injective as soon as the tile width does not exceed the row stride
(`N ≤ NS`, i.e. `BLOCK_N ≤ n_size`) — which always holds for a valid tiling, a
tile cannot be wider than the matrix. Geometrically: the in-row column offset
never spills into the next row, so `(row, col)` ↔ address is a bijection. This
discharges the abstract `Function.Injective` hypothesis from a one-glance bound. -/
theorem zOffset_injective_of_le (s : BlockState) (NS N M : Nat) (hN : N ≤ NS) :
    Function.Injective (zOffset s NS N M) := by
  -- `zOffset = (R + i)·NS + (Q + j) = (R·NS + Q) + i·NS + j`, the row-major form
  have heq : zOffset s NS N M
      = fun idx : TileIndex [M, N] =>
          (s.pids 0 / numNBlocks NS N * M * NS + s.pids 0 % numNBlocks NS N * N)
            + idx.1.val * NS + idx.2.1.val := by
    funext idx; simp only [zOffset, rowIndex, colIndex]; ring
  rw [heq]
  exact rowMajor2D_inj _ NS hN

/-- **Closed-form correctness for `matmul_triton1` (general statement).**

For arbitrary `n_size`, tile dims `M`/`N`, K-block size `BLOCK_K`, and K-block
count `numKBlocks` (so `k_size = BLOCK_K · numKBlocks`), every output cell of the
computed `M × N` tile equals the genuine matrix product
`Σ_{k < BLOCK_K·numKBlocks} X[i,k] · Y[k,j]` (over ℝ) of the loaded `X`/`Y`
tiles — NOT the kernel's own executed value.

Layout: `X[i,k]` at `X + rowIndex i · k_size + k`, `Y[k,j]` at
`Y + k · n_size + colIndex j`, `Z[i,j]` at `Z + rowIndex i · n_size + colIndex j`
(the kernel's row-major pointer arithmetic). Preconditions: `0 < BLOCK_K` and
`N ≤ NS` (tile width `BLOCK_N` ≤ row stride `n_size`, always true for a valid
tiling — this discharges output-offset injectivity via `zOffset_injective_of_le`),
plus a clean initial `undef`. -/
theorem matmul_triton1_closed_form_correct
    (X Y Z : RegionName) (s : BlockState)
    (NS M BLOCK_K N numKBlocks : Nat) (hBK : 0 < BLOCK_K)
    (hN : N ≤ NS)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes
      (kernel := matmul_triton1_surface X Y Z 0 (BLOCK_K * numKBlocks) NS M BLOCK_K N)
      (initialState := s)
      (write := fun idx : TileIndex [M, N] => some (Z, zOffset s NS N M idx))
      (expected := fun idx : TileIndex [M, N] =>
        matmulSpec s X Y (BLOCK_K * numKBlocks) NS N M N BLOCK_K numKBlocks idx.1 idx.2.1) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [matmul_triton1_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst hs0
  intro idx
  have hmain := matmul_triton1_exec_closed_form X Y Z s0 NS M BLOCK_K N numKBlocks hBK
    (zOffset_injective_of_le s0 NS N M hN) hundef idx
  have hExec2 : exec (matmul_triton1_surface X Y Z 0 (BLOCK_K * numKBlocks) NS M BLOCK_K N) s0
      = some s' := hExec
  rw [hExec2] at hmain
  simpa only [ComputeCorrect.OutputReadable.read_real] using hmain

end VeriTile.Bench.TritonBenchG.MatmulTriton1
