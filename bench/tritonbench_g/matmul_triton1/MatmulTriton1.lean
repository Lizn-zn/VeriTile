import VeriTile.Triton

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
matmul_triton1_closed_form_correct                ← TOP THEOREM (ComputeCorrect.Realizes_without_Rounding)
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
specification matmul_triton1_closed_form_correct
    (X Y Z : RegionName) (s : BlockState)
    (NS M BLOCK_K N numKBlocks : Nat) (hBK : 0 < BLOCK_K)
    (hN : N ≤ NS)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes_without_Rounding
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

/-! ## The `⊨[R]` streaming headline (wave-5 S1 fold genre)

Everything below is purely additive; the exact surface above is untouched.
Structure of the `execR R` story: this kernel has **zero rounding events** —
the loads and the K-loop arithmetic are all at `.real`, and the terminal
store is an unmasked `.real` store, which `stepStmtR` delegates to the exact
`writeMemTyped` (a `.real` write is not a quantization). The prologue and
the whole K-loop are therefore cast-free and collapse verbatim onto the
exact stepper (`stepForRangeAuxR_castFree`), reusing the proven
`preLoop` / `matmul_step` / `forRange_inv` invariant stack unchanged; only
the single store statement is re-proved on the `R` side
(`matmul_postLoopR`), where the `.real` write coincides with the
`writeMemAsR R .real` rounded write because `R.storeValue .real` **is** the
exact store demotion (`RoundingModel.storeValue_real`). The skin's readback
contract then carries `R.round .real`, which is the identity by the model's
defining `round_real` — the `⊨[R]` face at the `.real` grid is the exact
streaming contract, stated once for every `R`. -/

open scoped VeriTile.Triton.StreamMasked2DKernelIO₂

/-! ### Body decomposition names and cast-free collapses -/

/-- The 11-statement prologue (statements 0–10) as an explicit list. -/
private def matmulPrologue (X Y Z : RegionName)
    (NS M BLOCK_K N numKBlocks : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "pid" (Op.programId 0),
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
    Stmt.assign .nat [BLOCK_K] "k_offsets" (Op.arange BLOCK_K),
    Stmt.assign .ptr [M, BLOCK_K] "x_ptrs"
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
    Stmt.assign .real [M, N] "z" (Op.full [M, N] (Op.const 0)) ]

private theorem matmul_take11_eq (X Y Z : RegionName) (NS M BLOCK_K N numKBlocks : Nat) :
    (matmul_triton1_surface X Y Z 0 (BLOCK_K * numKBlocks) NS M BLOCK_K N).toAlgKernel.body.take 11
      = matmulPrologue X Y Z NS M BLOCK_K N numKBlocks := rfl

/-- `matmul_body_split` with the prologue named. By `rfl`. -/
private theorem matmul_body_split' (X Y Z : RegionName) (NS M BLOCK_K N numKBlocks : Nat) :
    (matmul_triton1_surface X Y Z 0 (BLOCK_K * numKBlocks) NS M BLOCK_K N).toAlgKernel.body
      = matmulPrologue X Y Z NS M BLOCK_K N numKBlocks
        ++ (Stmt.forRange "kk" 0 (BLOCK_K * numKBlocks) BLOCK_K
              (matmulLoopBody M BLOCK_K N (BLOCK_K * numKBlocks) NS BLOCK_K)
            :: matmulStoreStmt M N :: []) := rfl

set_option maxHeartbeats 2000000 in
/-- The prologue is cast-free: it steps identically under `stepStmtsR R`. -/
private theorem matmulPrologue_castFree (R : RoundingModel) (X Y Z : RegionName)
    (NS M BLOCK_K N numKBlocks : Nat) (t : BlockState) :
    stepStmtsR R (matmulPrologue X Y Z NS M BLOCK_K N numKBlocks) t
      = stepStmts (matmulPrologue X Y Z NS M BLOCK_K N numKBlocks) t := by
  simp only [matmulPrologue, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def]
  rfl

set_option maxHeartbeats 1000000 in
/-- The K-loop body is cast-free (loads at `.real`, `tl.dot`+add, nat pointer
advances — no `castFloat`, no narrow-float store): it steps identically under
`stepStmtsR R`, so the exact invariant stack transports to `execR`. -/
private theorem matmulBody_castFree (R : RoundingModel) (M K N KS NS BK : Nat)
    (t : BlockState) :
    stepStmtsR R (matmulLoopBody M K N KS NS BK) t
      = stepStmts (matmulLoopBody M K N KS NS BK) t := by
  simp only [matmulLoopBody, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def]
  rfl

/-! ### The tail under `execR R` -/

/-- A `.real`-typed rounded write **is** the `writeMemAsR R .real` write:
`writeMemTypedR` delegates `.real` to the exact `writeMemTyped`, whose stored
cell is definitionally `MemCell.of .real (real.ofReal (real.storeValue v))`,
and `RoundingModel.storeValue_real` says `R.storeValue .real` is exactly that
demotion. Lets the `.real` terminal store reuse the `writeMemAsR` scatter
readback/frame lemma family. -/
private theorem writeMemTypedR_real_eq (R : RoundingModel) (s : BlockState)
    (region : RegionName) (offset : Nat) (v : TileCarrier TileDType.real) :
    s.writeMemTypedR R .real region offset v
      = s.writeMemAsR R .real region offset v := by
  show s.writeMemTyped .real region offset v = _
  simp only [BlockState.writeMemTyped, BlockState.writeMemAs, BlockState.writeMemAsR,
    RoundingModel.storeValue_real]

/-- Tag-exact readback of a stored `.real` cell through `readMemAs .real`
(the `.real` sibling of `BlockState.readMemAs_fp16_of_cell`): the
`storeValue ∘ ofReal` round trip is the identity on a defined real. -/
private theorem readMemAs_real_of_cell {s : BlockState} {region : RegionName}
    {offset : Nat} {x : ℝ}
    (h : s.mem region offset
      = MemCell.of FloatDType.real.toTileDType (FloatDType.real.ofReal x)) :
    s.readMemAs .real region offset = FloatDType.real.ofReal x := by
  simp [BlockState.readMemAs, h, FloatDType.storeValue, FloatDType.ofReal]

set_option maxHeartbeats 2000000 in
/-- **R-postLoop**: from the exact invariant at `numKBlocks` blocks, the
`execR R` terminal store terminates and writes, at every output lane, the
exact-ℝ cell `MemCell.of .real (some (matmulSpec …))` — the `.real` store has
no rounding event (`R.storeValue .real` is the exact demotion) — with a
per-cell frame outside the output window. -/
private theorem matmul_postLoopR (R : RoundingModel) (X Y Z : RegionName) (s0 : BlockState)
    (NS M BLOCK_K N numKBlocks : Nat) (hBK : 0 < BLOCK_K)
    (hInj : Function.Injective (zOffset s0 NS N M))
    (st : BlockState)
    (hinv : matmulInvariant X Y Z s0 (BLOCK_K * numKBlocks) NS N M N BLOCK_K numKBlocks
        (BLOCK_K * numKBlocks) st) :
    ∃ sfin, stepStmtsR R (matmulStoreStmt M N :: []) st = some sfin
      ∧ (∀ idx : TileIndex [M, N],
          sfin.mem Z (zOffset s0 NS N M idx)
            = MemCell.of FloatDType.real.toTileDType
                (FloatDType.real.ofReal
                  (matmulSpec s0 X Y (BLOCK_K * numKBlocks) NS N M N BLOCK_K numKBlocks
                    idx.1 idx.2.1)))
      ∧ (∀ r o, (r ≠ Z ∨ ∀ idx : TileIndex [M, N], o ≠ zOffset s0 NS N M idx) →
          sfin.mem r o = st.mem r o) := by
  have hc : (BLOCK_K * numKBlocks) / BLOCK_K = numKBlocks := by
    rw [Nat.mul_comm, Nat.mul_div_cancel _ hBK]
  simp only [matmulInvariant, hc] at hinv
  obtain ⟨hpids, hi, hcle, hz, hm, hn, hk, hzp, hxp, hyp, hundef, hmem⟩ := hinv
  set KS := BLOCK_K * numKBlocks with hKS
  set zT : Tile .real [M, N] :=
    ⟨fun idx : TileIndex [M, N] => some (accPartial s0 X Y KS NS N M N BLOCK_K idx.1 idx.2.1 numKBlocks)⟩ with hzT
  set zpT : Tile .ptr [M, N] :=
    ⟨fun idx : TileIndex [M, N] => (Z.cast, rowIndex s0 NS N M idx.1 * NS + colIndex s0 NS N N idx.2.1)⟩ with hzpT
  -- the unmasked `.real` store: no rounding event, `writeMemAsR R .real` form
  have hstore : stepStmtR R (matmulStoreStmt M N) st
      = some ((TileShape.allIndices [M, N]).foldl
          (fun acc i =>
            acc.writeMemAsR R .real Z (zOffset s0 NS N M i) (zT.data i)) st) := by
    simp only [matmulStoreStmt, stepStmtR]
    rw [show evalOpR R (Op.ref .real [M, N] "z") st = some zT from by rw [evalOpR_ref, hz]]
    rw [show evalOpR R (Op.ref .ptr [M, N] "z_ptrs") st = some zpT from by rw [evalOpR_ref, hzp]]
    simp only [bind, Option.bind_some]
    refine congrArg some (List.foldl_ext _ _ st (fun acc i _ => ?_))
    simp only [if_true, zpT, hzpT, zOffset, Region.cast_id, writeMemTypedR_real_eq]
  rw [stepStmtsR_cons_some hstore, stepStmtsR_nil]
  refine ⟨_, rfl, ?_, ?_⟩
  · intro idx
    rw [BlockState.scatter_memcell_R_nd R .real st (zOffset s0 NS N M)
        (fun i => zT.data i) hInj idx]
    simp only [hzT, RoundingModel.storeValue_real, FloatDType.real_storeValue,
      WithBot.unbotD_some, matmulSpec, accPartial, Nat.mul_comm numKBlocks BLOCK_K]
  · intro r o hcond
    have hnot : ∀ k ∈ TileShape.allIndices [M, N], ¬(Z = r ∧ zOffset s0 NS N M k = o) := by
      intro k _ hk
      obtain ⟨hZr, hko⟩ := hk
      rcases hcond with hne | hall
      · exact hne hZr.symm
      · exact hall k hko.symm
    rw [BlockState.foldl_writeMemAsR_preserve_cell R .real (zOffset s0 NS N M)
        (fun i => zT.data i) r o (TileShape.allIndices [M, N]) hnot st]

/-! ### Safety-walk invariant (weak shape half of `matmulInvariant`) -/

/-- Safety-walk loop invariant: the *shape* half of `matmulInvariant`
(counter `i = c · BLOCK_K`; *some* accumulator tile, and the exact `z_ptrs` /
`x_ptrs` / `y_ptrs` address shapes) with no `undef` / `mem` / value pins.
Needed because the `⊨[R]` skin's `hts` obligation quantifies over arbitrary
launch states, so the safety walk cannot assume the clean-`undef`
precondition that `preLoop`'s full invariant needs. -/
private def matmulSafeInv (X Y Z : RegionName) (s0 : BlockState)
    (KS NS M N BLOCK_K T : Nat) (i : Nat) (s : BlockState) : Prop :=
  let c := i / BLOCK_K
  i = c * BLOCK_K ∧ c ≤ T ∧
  (∃ zT : Tile .real [M, N], s.regs .real [M, N] "z" = some zT) ∧
  (s.regs .ptr [M, N] "z_ptrs" = some ⟨fun idx : TileIndex [M, N] =>
      (Z.cast, rowIndex s0 NS N M idx.1 * NS + colIndex s0 NS N N idx.2.1)⟩) ∧
  (s.regs .ptr [M, BLOCK_K] "x_ptrs" = some ⟨fun idx : TileIndex [M, BLOCK_K] =>
      (X.cast, rowIndex s0 NS N M idx.1 * KS + idx.2.1.val + c * BLOCK_K)⟩) ∧
  (s.regs .ptr [BLOCK_K, N] "y_ptrs" = some ⟨fun idx : TileIndex [BLOCK_K, N] =>
      (Y.cast, idx.1.val * NS + colIndex s0 NS N N idx.2.1 + c * BLOCK_K * NS)⟩)

set_option maxHeartbeats 1000000 in
/-- Weak `preLoop`: from an **arbitrary** state the prologue steps to a state
satisfying `matmulSafeInv … 0` (no clean-`undef` hypothesis; the value half of
`preLoop` is dropped). -/
private theorem preLoopW (X Y Z : RegionName) (s : BlockState)
    (NS M BLOCK_K N numKBlocks : Nat) :
    ∃ s', stepStmts (matmulPrologue X Y Z NS M BLOCK_K N numKBlocks) s = some s'
      ∧ matmulSafeInv X Y Z s (BLOCK_K * numKBlocks) NS M N BLOCK_K numKBlocks 0 s' := by
  obtain ⟨s7, h7, hpids, hm, hn, hk, huf, hmem⟩ :=
    preLoop_scalars s (BLOCK_K * numKBlocks) NS M BLOCK_K N N
  rw [show matmulPrologue X Y Z NS M BLOCK_K N numKBlocks
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
  refine ⟨by simp, by simp, ?_, ?_, ?_, ?_⟩
  · -- some accumulator tile
    refine ⟨⟨fun _ => some (0 : ℝ)⟩, ?_⟩
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
      String.reduceEq, not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
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

set_option maxHeartbeats 2000000 in
/-- Weak step lemma: one body iteration from `matmulSafeInv i` steps
successfully (exact stepper; the body is cast-free) and re-establishes
`matmulSafeInv (i + BLOCK_K)` — the shape half of `matmul_step`, valid from
arbitrary launch states. -/
private theorem matmul_stepW (X Y Z : RegionName) (s0 : BlockState)
    (NS M BLOCK_K N numKBlocks : Nat) (hBK : 0 < BLOCK_K)
    (i : Nat) (s : BlockState) (hilt : i < BLOCK_K * numKBlocks)
    (hinv : matmulSafeInv X Y Z s0 (BLOCK_K * numKBlocks) NS M N BLOCK_K numKBlocks i s) :
    ∃ s', stepStmts (matmulLoopBody M BLOCK_K N (BLOCK_K * numKBlocks) NS BLOCK_K)
        (s.setReg "kk" .nat [] (Tile.scalar i)) = some s'
      ∧ matmulSafeInv X Y Z s0 (BLOCK_K * numKBlocks) NS M N BLOCK_K numKBlocks (i + BLOCK_K) s' := by
  set KS := BLOCK_K * numKBlocks with hKS
  have hc : i / BLOCK_K < numKBlocks := (Nat.div_lt_iff_lt_mul hBK).mpr (by rw [Nat.mul_comm]; exact hilt)
  have hc1 : (i + BLOCK_K) / BLOCK_K = i / BLOCK_K + 1 := Nat.add_div_right i hBK
  simp only [matmulSafeInv] at hinv
  obtain ⟨hi, hcle, ⟨zT, hz⟩, hzp, hxp, hyp⟩ := hinv
  set c := i / BLOCK_K with hcdef
  set xpT : Tile .ptr [M, BLOCK_K] :=
    ⟨fun idx : TileIndex [M, BLOCK_K] => (X.cast, rowIndex s0 NS N M idx.1 * KS + idx.2.1.val + c * BLOCK_K)⟩ with hxpT
  set ypT : Tile .ptr [BLOCK_K, N] :=
    ⟨fun idx : TileIndex [BLOCK_K, N] => (Y.cast, idx.1.val * NS + colIndex s0 NS N N idx.2.1 + c * BLOCK_K * NS)⟩ with hypT
  set sk := s.setReg "kk" .nat [] (Tile.scalar i) with hsk
  have hxpk : sk.regs .ptr [M, BLOCK_K] "x_ptrs" = some xpT := by simp [hsk, hxp, hxpT]
  have hypk : sk.regs .ptr [BLOCK_K, N] "y_ptrs" = some ypT := by simp [hsk, hyp, hypT]
  have hzk : sk.regs .real [M, N] "z" = some zT := by simp [hsk, hz]
  set xsub : Tile .real [M, BLOCK_K] :=
    ⟨fun idx => some (sk.readMem (xpT.data idx).1 (xpT.data idx).2)⟩ with hxsub
  set ysub : Tile .real [BLOCK_K, N] :=
    ⟨fun idx => some (sk.readMem (ypT.data idx).1 (ypT.data idx).2)⟩ with hysub
  unfold matmulLoopBody
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (load_ptr_none_real (Op.ref .ptr [M, BLOCK_K] "x_ptrs") _ xpT (by rw [evalOp_ref]; simp [hxpk])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (load_ptr_none_real (Op.ref .ptr [BLOCK_K, N] "y_ptrs") _ ypT (by rw [evalOp_ref]; simp [hypk])))]
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
  simp only [matmulSafeInv, hc1]
  refine ⟨by rw [hi]; ring, by omega, ?_, ?_, ?_, ?_⟩
  · -- some accumulator tile
    refine ⟨Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
      zT (Tile.dot [] xsub ysub), ?_⟩
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
      String.reduceEq, not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
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

set_option maxHeartbeats 1000000 in
/-- Per-iteration `TraceSafeListR` for the K-loop body: the two unmasked
loads' addresses are the invariant's pointer shapes, in bounds by the skin's
`read1`/`read2` windows (instantiated at block `c`); the three register-only
assigns are unconditionally safe. -/
private theorem matmul_bodySafeR (R : RoundingModel) (bounds : RegionBounds)
    (X Y : RegionName) (s0 : BlockState) (NS M BLOCK_K N numKBlocks : Nat)
    (c : Nat) (hc : c < numKBlocks) (sk : BlockState)
    (hxp : sk.regs .ptr [M, BLOCK_K] "x_ptrs" = some ⟨fun idx : TileIndex [M, BLOCK_K] =>
        (X.cast, rowIndex s0 NS N M idx.1 * (BLOCK_K * numKBlocks) + idx.2.1.val + c * BLOCK_K)⟩)
    (hyp : sk.regs .ptr [BLOCK_K, N] "y_ptrs" = some ⟨fun idx : TileIndex [BLOCK_K, N] =>
        (Y.cast, idx.1.val * NS + colIndex s0 NS N N idx.2.1 + c * BLOCK_K * NS)⟩)
    (hbX : ∀ (t : Fin numKBlocks) (j : Fin (M * BLOCK_K)),
      (s0.pids 0 / numNBlocks NS N * M + j.val / BLOCK_K) * (BLOCK_K * numKBlocks)
        + (t.val * BLOCK_K + j.val % BLOCK_K) < bounds X)
    (hbY : ∀ (t : Fin numKBlocks) (j : Fin (BLOCK_K * N)),
      (t.val * BLOCK_K + j.val / N) * NS
        + (s0.pids 0 % numNBlocks NS N * N + j.val % N) < bounds Y) :
    Stmt.TraceSafeListR R bounds
      (matmulLoopBody M BLOCK_K N (BLOCK_K * numKBlocks) NS BLOCK_K) sk := by
  unfold matmulLoopBody
  refine Stmt.TraceSafeListR.cons_intro ?_ ?_
  · -- load x_sub
    simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR,
      memAccessActiveAddressSafeR]
    refine ⟨trivial, trivial, ?_⟩
    intro ptrs hptrs i _
    rw [evalOpR_ref, hxp] at hptrs
    obtain rfl := Option.some.inj hptrs
    show rowIndex s0 NS N M i.1 * (BLOCK_K * numKBlocks) + i.2.1.val + c * BLOCK_K
        < bounds (Region.cast X)
    have h' : (s0.pids 0 / numNBlocks NS N * M + i.1.val) * (BLOCK_K * numKBlocks)
        + (c * BLOCK_K + i.2.1.val) < bounds X := by
      simpa using hbX ⟨c, hc⟩ (Lane2D.encode (i.1, i.2.1, PUnit.unit))
    simp only [Region.cast_id]
    calc rowIndex s0 NS N M i.1 * (BLOCK_K * numKBlocks) + i.2.1.val + c * BLOCK_K
        = (s0.pids 0 / numNBlocks NS N * M + i.1.val) * (BLOCK_K * numKBlocks)
            + (c * BLOCK_K + i.2.1.val) := by
          unfold rowIndex; ring
      _ < bounds X := h'
  · intro s1 h1
    obtain ⟨v1, -, rfl⟩ := stepStmtR_assign_inv h1
    refine Stmt.TraceSafeListR.cons_intro ?_ ?_
    · -- load y_sub
      simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR,
        memAccessActiveAddressSafeR]
      refine ⟨trivial, trivial, ?_⟩
      intro ptrs hptrs i _
      rw [evalOpR_ref] at hptrs
      rw [show (sk.setReg "x_sub" .real [M, BLOCK_K] v1).regs .ptr [BLOCK_K, N] "y_ptrs"
          = some (⟨fun idx : TileIndex [BLOCK_K, N] =>
            (Y.cast, idx.1.val * NS + colIndex s0 NS N N idx.2.1 + c * BLOCK_K * NS)⟩ :
              Tile .ptr [BLOCK_K, N]) from by simp [hyp]] at hptrs
      obtain rfl := Option.some.inj hptrs
      show i.1.val * NS + colIndex s0 NS N N i.2.1 + c * BLOCK_K * NS < bounds (Region.cast Y)
      have h' : (c * BLOCK_K + i.1.val) * NS
          + (s0.pids 0 % numNBlocks NS N * N + i.2.1.val) < bounds Y := by
        simpa using hbY ⟨c, hc⟩ (Lane2D.encode (i.1, i.2.1, PUnit.unit))
      simp only [Region.cast_id]
      calc i.1.val * NS + colIndex s0 NS N N i.2.1 + c * BLOCK_K * NS
          = (c * BLOCK_K + i.1.val) * NS + (s0.pids 0 % numNBlocks NS N * N + i.2.1.val) := by
            unfold colIndex; ring
        _ < bounds Y := h'
    · intro s2 h2
      obtain ⟨v2, -, rfl⟩ := stepStmtR_assign_inv h2
      refine Stmt.TraceSafeListR.of_forall _ _ ?_
      intro st hst s'
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hst
      rcases hst with rfl | rfl | rfl <;>
        simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]

set_option maxHeartbeats 1000000 in
/-- `TraceSafeListR` for the terminal store: the single unmasked `.real`
store's addresses are the invariant's `z_ptrs` shape — the skin's `write`
window. -/
private theorem matmul_tailSafeR (R : RoundingModel) (bounds : RegionBounds)
    (Z : RegionName) (s0 : BlockState) (NS M N : Nat) (st : BlockState)
    (hzp : st.regs .ptr [M, N] "z_ptrs" = some ⟨fun idx : TileIndex [M, N] =>
        (Z.cast, rowIndex s0 NS N M idx.1 * NS + colIndex s0 NS N N idx.2.1)⟩)
    (hbZ : ∀ j : Fin (M * N),
      (s0.pids 0 / numNBlocks NS N * M + j.val / N) * NS
        + (s0.pids 0 % numNBlocks NS N * N + j.val % N) < bounds Z) :
    Stmt.TraceSafeListR R bounds (matmulStoreStmt M N :: []) st := by
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun s' _ => Stmt.TraceSafeListR.nil_intro)
  simp only [matmulStoreStmt, Stmt.TraceSafeR, MemAccess.SafeAtR, MaskOpt.SafeAtR,
    Op.SafeAtR.eq_def, MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR,
    memAccessActiveAddressSafeR]
  refine ⟨trivial, trivial, trivial, ?_⟩
  intro ptrs hptrs i _
  rw [evalOpR_ref, hzp] at hptrs
  obtain rfl := Option.some.inj hptrs
  show rowIndex s0 NS N M i.1 * NS + colIndex s0 NS N N i.2.1 < bounds (Region.cast Z)
  have h' := hbZ (Lane2D.encode (i.1, i.2.1, PUnit.unit))
  rw [Lane2D.encode_div, Lane2D.encode_mod] at h'
  simp only [Region.cast_id]
  calc rowIndex s0 NS N M i.1 * NS + colIndex s0 NS N N i.2.1
      = (s0.pids 0 / numNBlocks NS N * M + i.1.val) * NS
          + (s0.pids 0 % numNBlocks NS N * N + i.2.1.val) := by
        unfold rowIndex colIndex; ring
    _ < bounds Z := h'

set_option maxHeartbeats 1000000 in
/-- **The `TraceSafeR` walk for the whole kernel** — driven by
`Stmt.forRangeTraceSafeR_inv` over the weak `matmulSafeInv`, with the
counter advancing by the loop's stride `BLOCK_K` (not 1). The three bound
groups are the skin's `read1`/`read2`/`write` windows. `hBK` is needed to
convert the raw counter `i < BLOCK_K · numKBlocks` into the block index
`i / BLOCK_K < numKBlocks` the per-step windows are phrased over. -/
private theorem matmul_triton1_traceSafeR (R : RoundingModel) (bounds : RegionBounds)
    (X Y Z : RegionName) (NS M BLOCK_K N numKBlocks : Nat) (hBK : 0 < BLOCK_K)
    (s : BlockState)
    (hbX : ∀ (t : Fin numKBlocks) (j : Fin (M * BLOCK_K)),
      (s.pids 0 / numNBlocks NS N * M + j.val / BLOCK_K) * (BLOCK_K * numKBlocks)
        + (t.val * BLOCK_K + j.val % BLOCK_K) < bounds X)
    (hbY : ∀ (t : Fin numKBlocks) (j : Fin (BLOCK_K * N)),
      (t.val * BLOCK_K + j.val / N) * NS
        + (s.pids 0 % numNBlocks NS N * N + j.val % N) < bounds Y)
    (hbZ : ∀ j : Fin (M * N),
      (s.pids 0 / numNBlocks NS N * M + j.val / N) * NS
        + (s.pids 0 % numNBlocks NS N * N + j.val % N) < bounds Z) :
    ((matmul_triton1_surface X Y Z 0 (BLOCK_K * numKBlocks) NS M BLOCK_K N).toAlgKernel).TraceSafeR
      R bounds s := by
  unfold Kernel.TraceSafeR
  rw [matmul_body_split' X Y Z NS M BLOCK_K N numKBlocks]
  have hstep : ∀ i s', i < BLOCK_K * numKBlocks →
      matmulSafeInv X Y Z s (BLOCK_K * numKBlocks) NS M N BLOCK_K numKBlocks i s' →
      Stmt.TraceSafeListR R bounds (matmulLoopBody M BLOCK_K N (BLOCK_K * numKBlocks) NS BLOCK_K)
        (s'.setReg "kk" .nat [] (Tile.scalar i)) ∧
      ∃ s'', stepStmtsR R (matmulLoopBody M BLOCK_K N (BLOCK_K * numKBlocks) NS BLOCK_K)
          (s'.setReg "kk" .nat [] (Tile.scalar i)) = some s'' ∧
        matmulSafeInv X Y Z s (BLOCK_K * numKBlocks) NS M N BLOCK_K numKBlocks (i + BLOCK_K) s'' := by
    intro i s' hi hP
    have hPd := hP
    simp only [matmulSafeInv] at hPd
    obtain ⟨hieq, hcle, hzE, hzp, hxp, hyp⟩ := hPd
    have hcT : i / BLOCK_K < numKBlocks :=
      (Nat.div_lt_iff_lt_mul hBK).mpr (by rw [Nat.mul_comm]; exact hi)
    refine ⟨matmul_bodySafeR R bounds X Y s NS M BLOCK_K N numKBlocks (i / BLOCK_K) hcT _
        (by simp [hxp]) (by simp [hyp]) hbX hbY, ?_⟩
    obtain ⟨s'', hs'', hP''⟩ :=
      matmul_stepW X Y Z s NS M BLOCK_K N numKBlocks hBK i s' hi hP
    exact ⟨s'', by rw [matmulBody_castFree]; exact hs'', hP''⟩
  refine Stmt.TraceSafeListR.append_intro _ _ ?_ ?_
  · -- prologue: register-only assigns, safe at every state
    refine Stmt.TraceSafeListR.of_forall _ _ ?_
    intro st hst s'
    simp only [matmulPrologue, List.mem_cons, List.not_mem_nil, or_false] at hst
    rcases hst with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
      simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]
  · intro s1 hs1
    obtain ⟨s1x, hpre, hP0⟩ := preLoopW X Y Z s NS M BLOCK_K N numKBlocks
    rw [matmulPrologue_castFree R X Y Z NS M BLOCK_K N numKBlocks s, hpre] at hs1
    obtain rfl := Option.some.inj hs1
    refine Stmt.TraceSafeListR.cons_intro ?_ ?_
    · -- the K-loop is trace-safe (invariant principle over the weak invariant)
      simp only [Stmt.TraceSafeR]
      exact Stmt.forRangeTraceSafeR_inv R bounds "kk" (BLOCK_K * numKBlocks) BLOCK_K
        (matmulLoopBody M BLOCK_K N (BLOCK_K * numKBlocks) NS BLOCK_K)
        (matmulSafeInv X Y Z s (BLOCK_K * numKBlocks) NS M N BLOCK_K numKBlocks)
        hstep 0 s1x hP0
    · intro s2 hs2
      obtain ⟨final, sLoop, hLoopStmt, hfinal, hPL⟩ :=
        forRange_inv (idx := "kk") (start := 0) (stop := BLOCK_K * numKBlocks) (step := BLOCK_K)
          (body := matmulLoopBody M BLOCK_K N (BLOCK_K * numKBlocks) NS BLOCK_K)
          (by omega) hP0
          (fun i st hlt hinv => matmul_stepW X Y Z s NS M BLOCK_K N numKBlocks hBK i st hlt hinv)
      rw [stepStmtR_forRange,
        stepForRangeAuxR_castFree R _ (matmulBody_castFree R M BLOCK_K N (BLOCK_K * numKBlocks) NS BLOCK_K) "kk",
        ← stepForRangeAux.forRange_unfold, hLoopStmt] at hs2
      obtain rfl := Option.some.inj hs2
      simp only [matmulSafeInv] at hPL
      obtain ⟨-, -, -, hzpL, -, -⟩ := hPL
      exact matmul_tailSafeR R bounds Z s NS M N sLoop hzpL hbZ

/-- The full matmul surface sits inside the flat-memory bridge's covered
fragment (`FlattenOk`; the `forRange` clause recurses into the cast-free
body). -/
theorem matmul_triton1_flattenOk (X Y Z : RegionName) (NS M BLOCK_K N numKBlocks : Nat) :
    ((matmul_triton1_surface X Y Z 0 (BLOCK_K * numKBlocks) NS M BLOCK_K N).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  rw [matmul_body_split' X Y Z NS M BLOCK_K N numKBlocks]
  simp [matmulPrologue, matmulLoopBody, matmulStoreStmt, StmtList.FlattenOk,
    Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-! ### IO signature, lane bridges, spec bridge -/

/-- The `X`-stream lane feeding output lane `l` at inner key `e`: the row of
`l` (row-major over the `[M, N]` output tile) paired with `e` over the
`[M, BLOCK_K]` per-step `X`-tile, both via the shared `Lane2D` bridge. -/
def aLane (M N BK : Nat) (l : Fin (M * N)) (e : Fin BK) : Fin (M * BK) :=
  Lane2D.encode ((Lane2D.decode l).1, e, PUnit.unit)

/-- The `Y`-stream lane feeding output lane `l` at inner key `e`: `e` paired
with the column of `l` over the `[BLOCK_K, N]` per-step `Y`-tile. -/
def bLane (M N BK : Nat) (l : Fin (M * N)) (e : Fin BK) : Fin (BK * N) :=
  Lane2D.encode (e, (Lane2D.decode l).2.1, PUnit.unit)

/-- **Streaming IO signature** of `matmul_triton1` on the two-stream fold
skin (S1: fold + terminal store). Step `c` of the K-loop reads the
`[M, BLOCK_K]` `X`-tile and the `[BLOCK_K, N]` `Y`-tile; after the loop one
`[M, N]` output tile is stored at the **`.real`** grid (`outDType` default —
the kernel's store is untyped `tl.store(z_ptrs, z)` at `.real`, so the
terminal store has no quantization event). The kernel uses only
`program_id(0)`: the `(m, n)` tile coordinate is the linear split
`pid₀ / num_n_blocks` × `pid₀ % num_n_blocks` (with
`num_n_blocks = cdiv n_size n_block_size = numNBlocks NS N`), so all three
windows read `pid₀` only. They transcribe the kernel's pointer arithmetic
exactly:

* `read1` lane `l = (i, e)` (row-major over `[M, BLOCK_K]`), step `t`:
  `(pid₀/num_n_blocks·M + i) · k_size + (t·BLOCK_K + e)` with
  `k_size = BLOCK_K · numKBlocks` — the invariant's `x_ptrs` cell
  `rowIndex(i)·k_size + e + t·BLOCK_K` after `t` advances.
* `read2` lane `l = (e, j)` (row-major over `[BLOCK_K, N]`), step `t`:
  `(t·BLOCK_K + e) · n_size + (pid₀%num_n_blocks·N + j)` — the `y_ptrs`
  cell after `t` advances of `BLOCK_K·n_size`.
* `write` lane `l = (i, j)`:
  `(pid₀/num_n_blocks·M + i) · n_size + (pid₀%num_n_blocks·N + j)` — the
  kernel's `z_ptrs` (= `zOffset` in pid form).

The kernel has no masks: all windows are `True`. -/
def matmulTriton1IO (X Y Z : RegionName) (NS M BLOCK_K N numKBlocks : Nat) :
    StreamMasked2DKernelIO₂ where
  kernel := matmul_triton1_surface X Y Z 0 (BLOCK_K * numKBlocks) NS M BLOCK_K N
  inp1 := X
  inp2 := Y
  out := Z
  T := numKBlocks
  B1 := M * BLOCK_K
  B2 := BLOCK_K * N
  C := M * N
  read1 := fun p₀ _ t l =>
    (p₀ / numNBlocks NS N * M + l.val / BLOCK_K) * (BLOCK_K * numKBlocks)
      + (t.val * BLOCK_K + l.val % BLOCK_K)
  read2 := fun p₀ _ t l =>
    (t.val * BLOCK_K + l.val / N) * NS + (p₀ % numNBlocks NS N * N + l.val % N)
  write := fun p₀ _ l =>
    (p₀ / numNBlocks NS N * M + l.val / N) * NS + (p₀ % numNBlocks NS N * N + l.val % N)
  mask1 := fun _ _ _ _ => True
  mask2 := fun _ _ _ _ => True
  writeMask := fun _ _ _ => True

/-- Under the two stream pins, `matmulSpec` at the decoded output lane **is**
the skin-level double fold `∑ t, ∑ e, xs · ys` (`gemmSum_blocks` +
address-identity of the windows with the invariant's pointer shapes). -/
private theorem matmulSpec_eq_streamSum (X Y : RegionName) (s₀ : BlockState)
    (NS M BLOCK_K N numKBlocks : Nat)
    (xs : Fin numKBlocks → Fin (M * BLOCK_K) → ℝ)
    (ys : Fin numKBlocks → Fin (BLOCK_K * N) → ℝ)
    (hx : ∀ (t : Fin numKBlocks) (j : Fin (M * BLOCK_K)),
      s₀.readMem X ((s₀.pids 0 / numNBlocks NS N * M + j.val / BLOCK_K) * (BLOCK_K * numKBlocks)
          + (t.val * BLOCK_K + j.val % BLOCK_K))
        = xs t j)
    (hy : ∀ (t : Fin numKBlocks) (j : Fin (BLOCK_K * N)),
      s₀.readMem Y ((t.val * BLOCK_K + j.val / N) * NS
          + (s₀.pids 0 % numNBlocks NS N * N + j.val % N))
        = ys t j)
    (l : Fin (M * N)) :
    matmulSpec s₀ X Y (BLOCK_K * numKBlocks) NS N M N BLOCK_K numKBlocks
        (Lane2D.decode l).1 (Lane2D.decode l).2.1
      = ∑ t : Fin numKBlocks, ∑ e : Fin BLOCK_K,
          xs t (aLane M N BLOCK_K l e) * ys t (bLane M N BLOCK_K l e) := by
  unfold matmulSpec
  rw [Nat.mul_comm BLOCK_K numKBlocks, gemmSum_blocks]
  refine Finset.sum_congr rfl fun t _ => Finset.sum_congr rfl fun e _ => ?_
  have hxa : xElem s₀ X (numKBlocks * BLOCK_K) NS N M (Lane2D.decode l).1 (t.val * BLOCK_K + e.val)
      = xs t (aLane M N BLOCK_K l e) := by
    rw [← hx t (aLane M N BLOCK_K l e)]
    simp only [aLane, Lane2D.encode_div, Lane2D.encode_mod]
    unfold xElem rowIndex
    congr 1
    ring
  have hyb : yElem s₀ Y NS N N (Lane2D.decode l).2.1 (t.val * BLOCK_K + e.val)
      = ys t (bLane M N BLOCK_K l e) := by
    rw [← hy t (bLane M N BLOCK_K l e)]
    simp only [bLane, Lane2D.encode_div, Lane2D.encode_mod]
    rfl
  rw [hxa, hyb]

/-! ### The headline -/

set_option maxHeartbeats 2000000 in
/-- **The `⊨[R]` streaming headline (wave-5 S1 fold genre).** For every
rounding model `R`, the faithful `matmul_triton1` surface implements, on its
`StreamMasked2DKernelIO₂` signature, the **ideal ℝ GEMM fold** over the
streamed tiles: output lane `l = (i, j)` holds
`∑ t, ∑ e, X-tile[t](i,e) · Y-tile[t](e,j)` — the spec `f` is exact real
arithmetic. The kernel has **no rounding events** (loads, dot and store are
all at `.real`), so the skin's boundary quantization degenerates: the
readback contract's `R.round .real` is the identity by the model's defining
`round_real`, and the `.real` terminal store is exact under `execR R`
(`stepStmtR` delegates `.real` writes to the exact `writeMemTyped`;
`RoundingModel.storeValue_real`).

Layer map: the prologue and the whole K-loop are cast-free, so under
`execR R` they collapse verbatim onto the exact stepper and the proven
`preLoop` / `matmul_step` / `forRange_inv` stack above is reused unchanged;
only the terminal store is re-proved on the `R` side (`matmul_postLoopR`).

Both hypotheses are truth-forced:

* `hBK : 0 < BLOCK_K` — the surface's K-loop steps by `BLOCK_K`
  (`range(0, k_size, k_block_size)`); at `BLOCK_K = 0` the loop never
  advances, `execR` diverges from the invariant stack (`forRange_inv`
  requires a nonzero step), and the block-index arithmetic `i / BLOCK_K` is
  meaningless. It holds for every real launch (`tl.arange(0, 0)` is not a
  tile).
* `hN : N ≤ NS` — the output window `row·n_size + col` is injective only
  when the tile width `n_block_size` fits the row stride `n_size`
  (`zOffset_injective_of_le` / `rowMajor2D_inj`, exactly the exact
  headline's precondition); with colliding output lanes the per-lane
  readback would be last-writer-wins and the statement false. It holds for
  every valid tiling.

Relation to the exact surface: the exact headline
`matmul_triton1_closed_form_correct` (`Realizes_without_Rounding`) above is
retained unchanged; this `⊨[R]` face restates the same GEMM content on the
streaming skin, for every `R` at once (at the `.real` grid the two faces
carry the same exact cell). Both faces are kept per the rounding-as-default
doctrine. -/
specification matmul_triton1_io_correctness (R : RoundingModel)
    (X Y Z : RegionName) (NS M BLOCK_K N numKBlocks : Nat)
    (hBK : 0 < BLOCK_K) (hN : N ≤ NS) :
    matmulTriton1IO X Y Z NS M BLOCK_K N numKBlocks ⊨[R] fun _ _ xs ys l =>
      ∑ t : Fin numKBlocks, ∑ e : Fin BLOCK_K,
        xs t (aLane M N BLOCK_K l e) * ys t (bLane M N BLOCK_K l e) := by
  refine StreamMasked2DKernelIO₂.ImplementsR.intro _ ?_ ?_ ?_
  · exact matmul_triton1_flattenOk X Y Z NS M BLOCK_K N numKBlocks
  · -- safety walk
    intro bounds s xs ys _hx _hy hbr1 hbr2 hbw
    simp only [matmulTriton1IO] at hbr1 hbr2 hbw ⊢
    exact matmul_triton1_traceSafeR R bounds X Y Z NS M BLOCK_K N numKBlocks hBK s
      (fun t j => hbr1 t j trivial) (fun t j => hbr2 t j trivial)
      (fun j => hbw j trivial)
  · -- the rounded Hoare triple
    intro s₀ xs ys hundef hx hy
    simp only [matmulTriton1IO] at hx hy ⊢
    have hundef' : ∀ rg o, s₀.undef rg o = 0 := fun rg o => by rw [hundef]
    have hInj : Function.Injective (zOffset s₀ NS N M) :=
      zOffset_injective_of_le s₀ NS N M hN
    -- exact preLoop + K-loop (cast-free, so they are the `execR` run too)
    obtain ⟨s1, hpre, hP0⟩ := preLoop X Y Z s₀ NS M BLOCK_K N numKBlocks hundef'
    obtain ⟨final, sLoop, hLoopStmt, hfinal, hPLoop⟩ :=
      forRange_inv (idx := "kk") (start := 0) (stop := BLOCK_K * numKBlocks) (step := BLOCK_K)
        (by omega) hP0
        (fun i st hlt hinv => matmul_step X Y Z s₀ NS M BLOCK_K N numKBlocks hBK i st hlt hinv)
    have hfinalEq : final = BLOCK_K * numKBlocks := by
      have hle : final ≤ BLOCK_K * numKBlocks := by
        simp only [matmulInvariant] at hPLoop
        obtain ⟨_, hieq, hcle, _⟩ := hPLoop
        calc final = final / BLOCK_K * BLOCK_K := hieq
          _ ≤ numKBlocks * BLOCK_K := Nat.mul_le_mul_right _ hcle
          _ = BLOCK_K * numKBlocks := Nat.mul_comm _ _
      exact le_antisymm hle hfinal
    subst hfinalEq
    have hmem0 : sLoop.mem = s₀.mem := by
      have h := hPLoop
      simp only [matmulInvariant] at h
      exact h.2.2.2.2.2.2.2.2.2.2.2
    -- R-side tail
    obtain ⟨sfin, hTailR, hval, hframe⟩ :=
      matmul_postLoopR R X Y Z s₀ NS M BLOCK_K N numKBlocks hBK hInj sLoop hPLoop
    have hLoopR : stepStmtR R (Stmt.forRange "kk" 0 (BLOCK_K * numKBlocks) BLOCK_K
        (matmulLoopBody M BLOCK_K N (BLOCK_K * numKBlocks) NS BLOCK_K)) s1 = some sLoop := by
      rw [stepStmtR_forRange,
        stepForRangeAuxR_castFree R _ (matmulBody_castFree R M BLOCK_K N (BLOCK_K * numKBlocks) NS BLOCK_K) "kk",
        ← stepForRangeAux.forRange_unfold]
      exact hLoopStmt
    have hpre' : stepStmts (matmulPrologue X Y Z NS M BLOCK_K N numKBlocks) s₀ = some s1 := by
      rw [← matmul_take11_eq X Y Z NS M BLOCK_K N numKBlocks]
      exact hpre
    refine ⟨sfin, ?_, ?_, ?_⟩
    · show execR R (matmul_triton1_surface X Y Z 0 (BLOCK_K * numKBlocks) NS M BLOCK_K N).toAlgKernel s₀
          = some sfin
      unfold execR
      rw [matmul_body_split' X Y Z NS M BLOCK_K N numKBlocks, stepStmtsR_append,
        matmulPrologue_castFree R X Y Z NS M BLOCK_K N numKBlocks s₀, hpre', Option.bind_some,
        stepStmtsR_cons_some hLoopR]
      exact hTailR
    · intro l _
      have hcell := hval (Lane2D.decode l)
      have haddr : (s₀.pids 0 / numNBlocks NS N * M + l.val / N) * NS
            + (s₀.pids 0 % numNBlocks NS N * N + l.val % N)
          = zOffset s₀ NS N M (Lane2D.decode l) := rfl
      rw [haddr, readMemAs_real_of_cell hcell, R.round_real_apply,
        matmulSpec_eq_streamSum X Y s₀ NS M BLOCK_K N numKBlocks xs ys
          (fun t j => hx t j trivial) (fun t j => hy t j trivial) l]
    · intro r o hcond
      have hcond' : r ≠ Z ∨ ∀ idx : TileIndex [M, N], o ≠ zOffset s₀ NS N M idx := by
        rcases hcond with hne | hno
        · exact Or.inl hne
        · refine Or.inr fun idx => ?_
          have h := hno (Lane2D.encode idx) trivial
          simpa [Lane2D.encode_div, Lane2D.encode_mod, zOffset, rowIndex, colIndex] using h
      rw [hframe r o hcond', hmem0]

end VeriTile.Bench.TritonBenchG.MatmulTriton1
