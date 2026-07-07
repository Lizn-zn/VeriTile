import VeriTile.Core
import VeriTile.Semantics
import VeriTile.Float
import VeriTile.Frontend.Triton.DSL
import VeriTile.Kernel

/-!
# `matmul_triton2` — closed-form GEMM correctness

`matmul_kernel` is an autotuned, group-scheduled tiled GEMM: program `pid` is
mapped through an L2-grouping schedule (`GROUP_SIZE_M`) to a tile coordinate
`(pid_m, pid_n)`, accumulates a `BLOCK_SIZE_M × BLOCK_SIZE_N` output tile via
`accumulator += tl.dot(a, b)` over the K dimension (with `offs_k < K - k·BLOCK_K`
masking on the `tl.load(..., other=0.0)` loads), and stores the tile into `C`
masked by `(offs_am < M) & (offs_bn < N)`.

This file proves the **full K-loop** correct against a genuine mathematical
matrix product: every active output cell `C[i,j]` of the computed tile equals
`Σ_{k < K} A[i,k] · B[k,j]` over `ℝ`, where `K = BLOCK_K · numKBlocks` is the
contracted dimension. This is NOT the kernel's own emitted value — it is the
independent closed-form `∑_k A·B` GEMM reference, derived from the loaded
`A`/`B` tiles. The masked loads zero out the contraction tail
(`e ≥ K − k·BLOCK_K`), so the masked-dot collapses to exactly the in-range keys.

## Proof architecture

```
matmul_triton2_closed_form_correct                ← TOP THEOREM (ComputeCorrect.Realizes)
  └─ matmul_triton2_exec_closed_form              ← exec-side closed form (every active cell = ∑_k A·B)
       ├─ preLoop      (P 0: accumulator = 0, pointers seeded)
       ├─ matmul_step         (one K-block: acc += dot(a,b) advances the partial sum)
       ├─ matmul_postLoop     (final masked store = the closed form on active lanes)
       └─ forRangeDyn_inv     (dynamic loop-invariant principle, drives the K-loop)
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` /
`num_warps` / `num_stages` are not modeled (the autotune config is fixed per
case). The host launch (grid, grouped-pid scheduling) is the trusted boundary;
the per-program statement is universally quantified over `s`, so it covers every
program of the grid. The layout contract is the kernel's own contiguity, with
the strides taken from the Python launch (`a.stride = (K,1)`, `b.stride =
(N,1)`, `c.stride = (N,1)`): `A[i,k]` at `A + rowIndex i · K + k`, `B[k,j]` at
`B + k · N + colIndex j`, `C[i,j]` at `C + rowIndex i · N + colIndex j`
(row-major), exactly as the kernel's pointer arithmetic constructs them. The
contraction `K = BLOCK_K · numKBlocks` is exact (no ragged tail); the load masks
are then `e < K − k·BLOCK_K`.
-/

namespace VeriTile.Bench.TritonBenchG.MatmulTriton2

open VeriTile

set_option linter.unusedSimpArgs false

/-- Faithful transcription of `matmul_triton2.py`'s `matmul_kernel`. -/
def matmul_triton2_surface
    (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  num_pid_m = tl.cdiv($(M), $(BLOCK_SIZE_M))
  num_pid_n = tl.cdiv($(N), $(BLOCK_SIZE_N))
  num_pid_in_group = $(GROUP_SIZE_M) * num_pid_n
  group_id = pid // num_pid_in_group
  first_pid_m = group_id * $(GROUP_SIZE_M)
  group_size_m = min(num_pid_m - first_pid_m, $(GROUP_SIZE_M))
  pid_m = first_pid_m + (pid % group_size_m)
  pid_n = (pid % num_pid_in_group) // group_size_m
  offs_am = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_bn = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  offs_k = tl.arange(0, $(BLOCK_SIZE_K))
  a_ptrs = A + offs_am[:, None] * $(stride_am) + offs_k[None, :] * $(stride_ak)
  b_ptrs = B + offs_k[:, None] * $(stride_bk) + offs_bn[None, :] * $(stride_bn)
  accumulator = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
  for k in range($(0), tl.cdiv($(K), $(BLOCK_SIZE_K)), $(1)) {
    a = tl.load(a_ptrs, mask=offs_k[None, :] < $(K) - k * $(BLOCK_SIZE_K), other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < $(K) - k * $(BLOCK_SIZE_K), other=0.0)
    accumulator += tl.dot(a, b)
    a_ptrs += $(BLOCK_SIZE_K) * $(stride_ak)
    b_ptrs += $(BLOCK_SIZE_K) * $(stride_bk)
  }
  c_ptrs = C + offs_am[:, None] * $(stride_cm) + offs_bn[None, :] * $(stride_cn)
  c_mask = (offs_am[:, None] < $(M)) & (offs_bn[None, :] < $(N))
  tl.store(c_ptrs, accumulator, mask=c_mask)
}

/-- The full `matmul_triton2` surface lowers to the algorithm layer. -/
theorem matmul_triton2_surface_toAlgorithm_supported
    (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M : Nat) :
    ∃ alg, (matmul_triton2_surface A B C M N K stride_am stride_ak stride_bk
      stride_bn stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K
      GROUP_SIZE_M).toAlgorithm? = Except.ok alg := by
  simp [matmul_triton2_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-! ## exec-stepping helpers (ported from `MatmulTriton1`/`AttentionForwardClosedForm`) -/

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

theorem evalOp_boolAnd {a b shape} (bc : Broadcast a b shape)
    (x : Op .bool a) (y : Op .bool b) (s : BlockState) :
    evalOp (.boolAnd bc x y) s = (do
      let vx ← evalOp x s; let vy ← evalOp y s;
      some (Tile.bop (fun p q => p && q) bc vx vy)) := by simp [evalOp]

/-- `.ptr` masked load with `other = 0.0` (`maskOther`): reads `readMem` where
the mask holds, `0` where it doesn't — i.e. `some (if mask then readMem else 0)`.
Unlike `MaskOpt.mask`, the `other` operand is explicit so this does not depend on
`undef`. -/
theorem load_ptr_maskOther0_real {shape : TileShape}
    (ptrOp : Op .ptr shape) (maskOp : Op .bool shape) (s : BlockState)
    (ptrs : Tile .ptr shape) (masks : Tile .bool shape)
    (hp : evalOp ptrOp s = some ptrs) (hm : evalOp maskOp s = some masks) :
    evalOp (.load .real (.ptr ptrOp) (.maskOther maskOp (Op.broadcast (Op.const 0.0) shape))) s
      = some ⟨fun i => some (if masks.data i then
          s.readMem (ptrs.data i).1 (ptrs.data i).2 else 0)⟩ := by
  have hother : evalOp (Op.broadcast (Op.const 0.0) shape) s
      = some (⟨fun _ : TileIndex shape => (some (0.0 : ℝ) : WithBot ℝ)⟩ : Tile .real shape) := by
    simp only [evalOp, evalOp_const, Option.bind]
    rfl
  simp only [evalOp, hp, hm, hother, Option.bind]
  refine congrArg some ?_
  ext i
  cases hmi : masks.data i <;>
    simp only [hmi, Bool.false_eq_true, if_true, if_false, BlockState.readMemValue_real]
  norm_num

/-- `a_ptrs` eval: cell `(i,e) = (A, offs_am i · SAM + e · SAK)`. -/
theorem aptrs_eval (s : BlockState) (A : RegionName) (M BK SAM SAK : Nat) (gm : Fin M → Nat)
    (hm : s.regs .nat [M] "offs_am" = some (Tile.vec gm))
    (hk : s.regs .nat [BK] "offs_k" = some (Tile.vec (fun e : Fin BK => e.val))) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase A)
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_am")) (Op.constNat SAM))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BK] "offs_k")) (Op.constNat SAK)))) s
      = some (⟨fun idx : TileIndex [M, BK] => (A.cast, gm idx.1 * SAM + idx.2.1.val * SAK)⟩ : Tile .ptr [M, BK]) := by
  rw [evalOp_ptrAdd, evalOp_ptrBase]
  simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref, evalOp_expandDim_one_nat,
    evalOp_expandDim_zero_nat, hm, hk, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec]
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec, NumericDType.add, NumericDType.mul]

/-- `b_ptrs` eval: cell `(e,j) = (B, offs_k e · SBK + offs_bn j · SBN)`. -/
theorem bptrs_eval (s : BlockState) (B : RegionName) (BK N SBK SBN : Nat) (gn : Fin N → Nat)
    (hk : s.regs .nat [BK] "offs_k" = some (Tile.vec (fun e : Fin BK => e.val)))
    (hn : s.regs .nat [N] "offs_bn" = some (Tile.vec gn)) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase B)
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "offs_k")) (Op.constNat SBK))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [N] "offs_bn")) (Op.constNat SBN)))) s
      = some (⟨fun idx : TileIndex [BK, N] => (B.cast, idx.1.val * SBK + gn idx.2.1 * SBN)⟩ : Tile .ptr [BK, N]) := by
  rw [evalOp_ptrAdd, evalOp_ptrBase]
  simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref, evalOp_expandDim_one_nat,
    evalOp_expandDim_zero_nat, hk, hn, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec]
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec, NumericDType.add, NumericDType.mul]

/-- `c_ptrs` eval: cell `(i,j) = (C, offs_am i · SCM + offs_bn j · SCN)`. -/
theorem cptrs_eval (s : BlockState) (C : RegionName) (M N SCM SCN : Nat) (gm : Fin M → Nat) (gn : Fin N → Nat)
    (hm : s.regs .nat [M] "offs_am" = some (Tile.vec gm))
    (hn : s.regs .nat [N] "offs_bn" = some (Tile.vec gn)) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase C)
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_am")) (Op.constNat SCM))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [N] "offs_bn")) (Op.constNat SCN)))) s
      = some (⟨fun idx : TileIndex [M, N] => (C.cast, gm idx.1 * SCM + gn idx.2.1 * SCN)⟩ : Tile .ptr [M, N]) := by
  rw [evalOp_ptrAdd, evalOp_ptrBase]
  simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref, evalOp_expandDim_one_nat,
    evalOp_expandDim_zero_nat, hm, hn, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec]
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec, NumericDType.add, NumericDType.mul]

/-- **`a`-load mask eval**: `remap [M,BK] (lt (offs_k[None,:]) (K − k·BK))`.
Lane `(i,e)` is active iff `e < K − kc·BK`. -/
theorem amask_eval (s : BlockState) (M BK K kc : Nat)
    (hk : s.regs .nat [BK] "offs_k" = some (Tile.vec (fun e : Fin BK => e.val)))
    (hkc : s.regs .nat [] "k" = some (Tile.scalar kc)) :
    evalOp (Op.remap [M, BK] (Broadcast.leftIndex (Broadcast.consL (Broadcast.consSame Broadcast.nil)))
              (Op.lt .nat Broadcast.scalarR
                (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BK] "offs_k"))
                (Op.sub .nat Broadcast.nil (Op.constNat K)
                  (Op.mul .nat Broadcast.nil (Op.ref .nat [] "k") (Op.constNat BK))))) s
      = some (⟨fun idx : TileIndex [M, BK] => decide (idx.2.1.val < K - kc * BK)⟩ : Tile .bool [M, BK]) := by
  conv_lhs => unfold evalOp
  conv_lhs => arg 1; rw [evalOp_lt]; arg 1; rw [evalOp_expandDim_zero_nat, hk]
  simp only [Option.bind_eq_bind, Option.bind_some, evalOp_sub, evalOp_constNat, evalOp_mul,
    evalOp_ref, hkc, NumericDType.sub, NumericDType.mul]
  refine congrArg some ?_
  ext idx
  simp [Tile.remap, Tile.bop, Tile.cop, Tile.expandDim, Tile.vec, ComparableDType.lt,
    Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.sub, NumericDType.mul]

/-- **`b`-load mask eval**: `remap [BK,N] (lt (offs_k[:,None]) (K − k·BK))`.
Lane `(e,j)` is active iff `e < K − kc·BK`. -/
theorem bmask_eval (s : BlockState) (BK N K kc : Nat)
    (hk : s.regs .nat [BK] "offs_k" = some (Tile.vec (fun e : Fin BK => e.val)))
    (hkc : s.regs .nat [] "k" = some (Tile.scalar kc)) :
    evalOp (Op.remap [BK, N] (Broadcast.leftIndex (Broadcast.consSame (Broadcast.consL Broadcast.nil)))
              (Op.lt .nat Broadcast.scalarR
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "offs_k"))
                (Op.sub .nat Broadcast.nil (Op.constNat K)
                  (Op.mul .nat Broadcast.nil (Op.ref .nat [] "k") (Op.constNat BK))))) s
      = some (⟨fun idx : TileIndex [BK, N] => decide (idx.1.val < K - kc * BK)⟩ : Tile .bool [BK, N]) := by
  conv_lhs => unfold evalOp
  conv_lhs => arg 1; rw [evalOp_lt]; arg 1; rw [evalOp_expandDim_one_nat, hk]
  simp only [Option.bind_eq_bind, Option.bind_some, evalOp_sub, evalOp_constNat, evalOp_mul,
    evalOp_ref, hkc, NumericDType.sub, NumericDType.mul]
  refine congrArg some ?_
  ext idx
  simp [Tile.remap, Tile.bop, Tile.cop, Tile.expandDim, Tile.vec, ComparableDType.lt,
    Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.sub, NumericDType.mul]

/-- **Masked dot collapse** (the crux tile-op of the step lemma). When the
contraction lanes `e ≥ Kc` of both operands are masked to `some 0` (which is what
the kernel's `a`/`b` `maskOther(other=0)` loads produce), `tl.dot a b` at `(i,j)`
equals `Σ_{e < BK} [e < Kc] · A·B` — the masked-off tail contributes nothing. -/
theorem dot_masked {M BK N Kc : Nat} (a : Tile .real [M, BK]) (b : Tile .real [BK, N])
    (i : Fin M) (j : Fin N) (fa : Fin BK → ℝ) (fb : Fin BK → ℝ)
    (ha : ∀ e : Fin BK, a.data (i, e, PUnit.unit) = some (if e.val < Kc then fa e else 0))
    (hb : ∀ e : Fin BK, b.data (e, j, PUnit.unit) = some (if e.val < Kc then fb e else 0)) :
    (Tile.dot [] a b).data (i, j, PUnit.unit)
      = some (Finset.univ.sum fun e : Fin BK =>
          (if e.val < Kc then fa e * fb e else 0)) := by
  rw [Tile.dot_nil_data]
  rw [show (@Finset.sum (Fin BK) (WithBot ℝ) _ Finset.univ
        (fun e => Option.map₂ (· * ·) (a.data (i, e, PUnit.unit)) (b.data (e, j, PUnit.unit))))
      = @Finset.sum (Fin BK) (WithBot ℝ) _ Finset.univ
          (fun e => (some (if e.val < Kc then fa e * fb e else 0) : WithBot ℝ))
      from Finset.sum_congr rfl (fun e _ => by
        rw [ha e, hb e]
        by_cases he : e.val < Kc
        · simp only [if_pos he]; rfl
        · simp only [if_neg he, Option.map₂, Option.bind, Option.map, mul_zero])]
  show (Finset.univ.sum fun e : Fin BK => (((if e.val < Kc then fa e * fb e else 0 : ℝ)) : WithBot ℝ)) = _
  rw [← WithBot.coe_sum]; rfl

/-! ## GEMM closed-form spec -/

/-- Ceiling division `⌈a / b⌉`, matching Triton's `tl.cdiv`. -/
def cdiv (a b : Nat) : Nat := (a + b - 1) / b

/-- `num_pid_m = cdiv M BLOCK_M`. -/
def numPidM (M BM : Nat) : Nat := cdiv M BM

/-- `num_pid_n = cdiv N BLOCK_N`. -/
def numPidN (N BN : Nat) : Nat := cdiv N BN

/-- Group-scheduled tile-row index of program `pid`. Mirrors the kernel's
`pid_m = first_pid_m + (pid % group_size_m)` with `group_size_m =
min(num_pid_m − first_pid_m, GROUP_SIZE_M)` and `first_pid_m = (pid //
num_pid_in_group)·GROUP_SIZE_M`. -/
def pidM (s : BlockState) (M N BM BN GM : Nat) : Nat :=
  let nm := numPidM M BM
  let nn := numPidN N BN
  let nig := GM * nn
  let gid := s.pids 0 / nig
  let fpm := gid * GM
  let gsm := if nm - fpm < GM then nm - fpm else GM
  fpm + s.pids 0 % gsm

/-- Group-scheduled tile-col index of program `pid`. Mirrors
`pid_n = (pid % num_pid_in_group) // group_size_m`. -/
def pidN (s : BlockState) (M N BM BN GM : Nat) : Nat :=
  let nm := numPidM M BM
  let nn := numPidN N BN
  let nig := GM * nn
  let gid := s.pids 0 / nig
  let fpm := gid * GM
  let gsm := if nm - fpm < GM then nm - fpm else GM
  s.pids 0 % nig / gsm

/-- Global output row of tile lane `i`: `pid_m · BLOCK_M + i`. -/
def rowIndex (s : BlockState) (M N BM BN GM : Nat) (i : Fin BM) : Nat :=
  pidM s M N BM BN GM * BM + i.val

/-- Global output column of tile lane `j`: `pid_n · BLOCK_N + j`. -/
def colIndex (s : BlockState) (M N BM BN GM : Nat) (j : Fin BN) : Nat :=
  pidN s M N BM BN GM * BN + j.val

/-- `A[i, k] = readMem A (rowIndex i · SAM + k · SAK)` (kernel's A layout). -/
noncomputable def aElem (s : BlockState) (A : RegionName) (M N BM BN GM SAM SAK : Nat)
    (i : Fin BM) (k : Nat) : ℝ :=
  s.readMem A (rowIndex s M N BM BN GM i * SAM + k * SAK)

/-- `B[k, j] = readMem B (k · SBK + colIndex j · SBN)` (kernel's B layout). -/
noncomputable def bElem (s : BlockState) (B : RegionName) (M N BM BN GM SBK SBN : Nat)
    (j : Fin BN) (k : Nat) : ℝ :=
  s.readMem B (k * SBK + colIndex s M N BM BN GM j * SBN)

/-- **Genuine GEMM spec**: `C[i,j] = Σ_{k < BLOCK_K·numKBlocks} A[i,k] · B[k,j]`. -/
noncomputable def matmulSpec (s : BlockState) (A B : RegionName)
    (M N BM BN GM SAM SAK SBK SBN BLOCK_K numKBlocks : Nat) (i : Fin BM) (j : Fin BN) : ℝ :=
  (Finset.range (BLOCK_K * numKBlocks)).sum
    (fun k => aElem s A M N BM BN GM SAM SAK i k * bElem s B M N BM BN GM SBK SBN j k)

/-- Partial GEMM accumulator after `c` K-blocks: `Σ_{k < c·BLOCK_K} A·B`. -/
noncomputable def accPartial (s : BlockState) (A B : RegionName)
    (M N BM BN GM SAM SAK SBK SBN BLOCK_K : Nat) (i : Fin BM) (j : Fin BN) (c : Nat) : ℝ :=
  (Finset.range (c * BLOCK_K)).sum
    (fun k => aElem s A M N BM BN GM SAM SAK i k * bElem s B M N BM BN GM SBK SBN j k)

/-- One-block step of the partial accumulator: the new block's dot is over the
`BLOCK_K` keys `c·BLOCK_K + e`. -/
theorem accPartial_succ (s : BlockState) (A B : RegionName)
    (M N BM BN GM SAM SAK SBK SBN BLOCK_K : Nat) (i : Fin BM) (j : Fin BN) (c : Nat) :
    accPartial s A B M N BM BN GM SAM SAK SBK SBN BLOCK_K i j (c + 1)
      = accPartial s A B M N BM BN GM SAM SAK SBK SBN BLOCK_K i j c
        + (Finset.univ.sum fun e : Fin BLOCK_K =>
            aElem s A M N BM BN GM SAM SAK i (c * BLOCK_K + e.val)
              * bElem s B M N BM BN GM SBK SBN j (c * BLOCK_K + e.val)) := by
  unfold accPartial
  have h : (c + 1) * BLOCK_K = c * BLOCK_K + BLOCK_K := by ring
  rw [h, Finset.sum_range_add]
  congr 1
  rw [Finset.sum_range fun e => aElem s A M N BM BN GM SAM SAK i (c * BLOCK_K + e)
        * bElem s B M N BM BN GM SBK SBN j (c * BLOCK_K + e)]

/-! ## Body decomposition (prefix ++ for-loop ++ stores) -/

/-- The 5-statement K-loop body, transcribed. -/
def matmulLoopBody (M BK N K SAK SBK : Nat) : List Stmt :=
  [ Stmt.assign .real [M, BK] "a"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [M, BK] "a_ptrs"))
        (MaskOpt.maskOther
          (Op.remap [M, BK] (Broadcast.leftIndex (Broadcast.consL (Broadcast.consSame Broadcast.nil)))
            (Op.lt .nat Broadcast.scalarR
              (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BK] "offs_k"))
              (Op.sub .nat Broadcast.nil (Op.constNat K)
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "k") (Op.constNat BK)))))
          (Op.broadcast (Op.const 0.0) [M, BK]))),
    Stmt.assign .real [BK, N] "b"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [BK, N] "b_ptrs"))
        (MaskOpt.maskOther
          (Op.remap [BK, N] (Broadcast.leftIndex (Broadcast.consSame (Broadcast.consL Broadcast.nil)))
            (Op.lt .nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "offs_k"))
              (Op.sub .nat Broadcast.nil (Op.constNat K)
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "k") (Op.constNat BK)))))
          (Op.broadcast (Op.const 0.0) [BK, N]))),
    Stmt.assign .real [M, N] "accumulator"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [M, N] "accumulator")
        (Op.dot (batch := []) (Op.ref .real [M, BK] "a") (Op.ref .real [BK, N] "b"))),
    Stmt.assign .ptr [M, BK] "a_ptrs"
      (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [M, BK] "a_ptrs")
        (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat SAK))),
    Stmt.assign .ptr [BK, N] "b_ptrs"
      (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BK, N] "b_ptrs")
        (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat SBK))) ]

/-- Post-loop `c_ptrs` / `c_mask` / masked store statements. `Mb`/`Nb` are the
matrix bounds (the `c_mask` cutoffs); `BM`/`BN` are the tile shape. -/
def matmulTailStmts (C : RegionName) (BM BN Mb Nb SCM SCN : Nat) : List Stmt :=
  [ Stmt.assign .ptr [BM, BN] "c_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase C)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_am")) (Op.constNat SCM))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_bn")) (Op.constNat SCN)))),
    Stmt.assign .bool [BM, BN] "c_mask"
      (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_am")) (Op.constNat Mb))
        (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_bn")) (Op.constNat Nb))),
    Stmt.store .real [BM, BN] (MemAccess.ptr (Op.ref .ptr [BM, BN] "c_ptrs"))
      (Op.ref .real [BM, BN] "accumulator")
      (MaskOpt.mask (Op.ref .bool [BM, BN] "c_mask")) ]

/-- The dynamic K-loop's `stop` op (`tl.cdiv(K, BLOCK_K)`). -/
def kStopOp (K BK : Nat) : Op .nat [] :=
  Op.div .nat Broadcast.nil
    (Op.sub .nat Broadcast.nil
      (Op.add .nat Broadcast.nil (Op.constNat K) (Op.constNat BK)) (Op.constNat 1))
    (Op.constNat BK)

set_option maxRecDepth 8000 in
/-- The K-loop + tail (`drop 15` of the body): the dynamic for-loop followed by
the three masked-store statements. By `rfl`. -/
theorem matmul_body_drop15 (A B C : RegionName)
    (M N K SAM SAK SBK SBN SCM SCN BM BN BK GM : Nat) :
    (matmul_triton2_surface A B C M N K SAM SAK SBK SBN SCM SCN BM BN BK GM).toAlgKernel.body.drop 15
      = (Stmt.forRangeDyn "k" (Op.constNat 0) (kStopOp K BK) (Op.constNat 1)
              (matmulLoopBody BM BK BN K SAK SBK)
            :: matmulTailStmts C BM BN M N SCM SCN) := by
  rfl

/-- Body decomposition: prefix (15) ++ [forRangeDyn-loop] ++ tail (3). -/
theorem matmul_body_split (A B C : RegionName)
    (M N K SAM SAK SBK SBN SCM SCN BM BN BK GM : Nat) :
    (matmul_triton2_surface A B C M N K SAM SAK SBK SBN SCM SCN BM BN BK GM).toAlgKernel.body
      = (matmul_triton2_surface A B C M N K SAM SAK SBK SBN SCM SCN BM BN BK GM).toAlgKernel.body.take 15
        ++ (Stmt.forRangeDyn "k" (Op.constNat 0) (kStopOp K BK) (Op.constNat 1)
              (matmulLoopBody BM BK BN K SAK SBK)
            :: matmulTailStmts C BM BN M N SCM SCN) := by
  conv_lhs => rw [← List.take_append_drop 15
    (matmul_triton2_surface A B C M N K SAM SAK SBK SBN SCM SCN BM BN BK GM).toAlgKernel.body]
  rw [matmul_body_drop15]

/-! ## Loop invariant -/

/-- **Loop invariant** (counter `i = c`).

After `c` K-blocks: program ids and `mem`/`undef` fixed; the `offs_am` / `offs_bn`
/ `offs_k` registers seeded; `accumulator` equals the partial GEMM accumulator
`accPartial … c`; and `a_ptrs` / `b_ptrs` advanced by `c` blocks. -/
noncomputable def matmulInvariant
    (A B : RegionName) (s0 : BlockState) (M N BM BN GM SAM SAK SBK SBN BLOCK_K numKBlocks : Nat)
    (i : Nat) (s : BlockState) : Prop :=
  let c := i
  s.pids = s0.pids ∧ c ≤ numKBlocks ∧
  (s.regs .real [BM, BN] "accumulator" = some ⟨fun idx : TileIndex [BM, BN] =>
      some (accPartial s0 A B M N BM BN GM SAM SAK SBK SBN BLOCK_K idx.1 idx.2.1 c)⟩) ∧
  (s.regs .nat [BM] "offs_am" = some (Tile.vec (fun r : Fin BM => rowIndex s0 M N BM BN GM r))) ∧
  (s.regs .nat [BN] "offs_bn" = some (Tile.vec (fun j : Fin BN => colIndex s0 M N BM BN GM j))) ∧
  (s.regs .nat [BLOCK_K] "offs_k" = some (Tile.vec (fun e : Fin BLOCK_K => e.val))) ∧
  (s.regs .ptr [BM, BLOCK_K] "a_ptrs" = some ⟨fun idx : TileIndex [BM, BLOCK_K] =>
      (A.cast, rowIndex s0 M N BM BN GM idx.1 * SAM + idx.2.1.val * SAK + c * BLOCK_K * SAK)⟩) ∧
  (s.regs .ptr [BLOCK_K, BN] "b_ptrs" = some ⟨fun idx : TileIndex [BLOCK_K, BN] =>
      (B.cast, idx.1.val * SBK + colIndex s0 M N BM BN GM idx.2.1 * SBN + c * BLOCK_K * SBK)⟩) ∧
  (∀ rg o, s.undef rg o = 0) ∧ (s.mem = s0.mem)

/-- The 15-statement prologue (grouped-pid scheduling + pointer seeding + zero
init), transcribed at the algorithm layer. -/
def matmulPreBody (A B : RegionName) (M N _K SAM SAK SBK SBN BM BN BK GM : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "pid" (Op.programId 0),
    Stmt.assign .nat [] "num_pid_m"
      (Op.div .nat Broadcast.nil
        (Op.sub .nat Broadcast.nil (Op.add .nat Broadcast.nil (Op.constNat M) (Op.constNat BM)) (Op.constNat 1))
        (Op.constNat BM)),
    Stmt.assign .nat [] "num_pid_n"
      (Op.div .nat Broadcast.nil
        (Op.sub .nat Broadcast.nil (Op.add .nat Broadcast.nil (Op.constNat N) (Op.constNat BN)) (Op.constNat 1))
        (Op.constNat BN)),
    Stmt.assign .nat [] "num_pid_in_group"
      (Op.mul .nat Broadcast.nil (Op.constNat GM) (Op.ref .nat [] "num_pid_n")),
    Stmt.assign .nat [] "group_id"
      (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "pid") (Op.ref .nat [] "num_pid_in_group")),
    Stmt.assign .nat [] "first_pid_m"
      (Op.mul .nat Broadcast.nil (Op.ref .nat [] "group_id") (Op.constNat GM)),
    Stmt.assign .nat [] "group_size_m"
      (Op.where
        (Op.lt .nat Broadcast.nil
          (Op.sub .nat Broadcast.nil (Op.ref .nat [] "num_pid_m") (Op.ref .nat [] "first_pid_m"))
          (Op.constNat GM))
        (Op.sub .nat Broadcast.nil (Op.ref .nat [] "num_pid_m") (Op.ref .nat [] "first_pid_m"))
        (Op.constNat GM)),
    Stmt.assign .nat [] "pid_m"
      (Op.add .nat Broadcast.nil (Op.ref .nat [] "first_pid_m")
        (Op.mod .nat Broadcast.nil (Op.ref .nat [] "pid") (Op.ref .nat [] "group_size_m"))),
    Stmt.assign .nat [] "pid_n"
      (Op.floorDiv .nat Broadcast.nil
        (Op.mod .nat Broadcast.nil (Op.ref .nat [] "pid") (Op.ref .nat [] "num_pid_in_group"))
        (Op.ref .nat [] "group_size_m")),
    Stmt.assign .nat [BM] "offs_am"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat BM)) (Op.arange BM)),
    Stmt.assign .nat [BN] "offs_bn"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_n") (Op.constNat BN)) (Op.arange BN)),
    Stmt.assign .nat [BK] "offs_k" (Op.arange BK),
    Stmt.assign .ptr [BM, BK] "a_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase A)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_am")) (Op.constNat SAM))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BK] "offs_k")) (Op.constNat SAK)))),
    Stmt.assign .ptr [BK, BN] "b_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase B)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "offs_k")) (Op.constNat SBK))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_bn")) (Op.constNat SBN)))),
    Stmt.assign .real [BM, BN] "accumulator" (Op.full [BM, BN] (Op.const 0)) ]

/-- The prologue is exactly `body.take 15`. By `rfl`. -/
theorem matmul_preBody_eq (A B C : RegionName)
    (M N K SAM SAK SBK SBN SCM SCN BM BN BK GM : Nat) :
    (matmul_triton2_surface A B C M N K SAM SAK SBK SBN SCM SCN BM BN BK GM).toAlgKernel.body.take 15
      = matmulPreBody A B M N K SAM SAK SBK SBN BM BN BK GM := by
  rfl

set_option maxHeartbeats 1000000 in
/-- **preLoop scalars** (statements 0–11): the grouped-pid schedule + index
vectors. Steps to a state with `offs_am = vec(rowIndex)`, `offs_bn =
vec(colIndex)`, `offs_k = vec id`. -/
theorem preLoop_scalars (A B : RegionName) (s : BlockState)
    (M N K SAM SAK SBK SBN BM BN BK GM : Nat) :
    ∃ s12, stepStmts ((matmulPreBody A B M N K SAM SAK SBK SBN BM BN BK GM).take 12) s = some s12
      ∧ s12.pids = s.pids
      ∧ s12.regs .nat [BM] "offs_am" = some (Tile.vec (fun i : Fin BM => rowIndex s M N BM BN GM i))
      ∧ s12.regs .nat [BN] "offs_bn" = some (Tile.vec (fun j : Fin BN => colIndex s M N BM BN GM j))
      ∧ s12.regs .nat [BK] "offs_k" = some (Tile.vec (fun e : Fin BK => e.val))
      ∧ s12.undef = s.undef
      ∧ s12.mem = s.mem := by
  simp only [matmulPreBody, rowIndex, colIndex, pidM, pidN, numPidM, numPidN, cdiv]
  simp [stepStmts, stepStmt, evalOp_floorDiv, evalOp_mod, evalOp_where, Option.bind,
    BlockState.setReg, Tile.bop, Tile.cop, Tile.vec, Tile.select, NumericDType.add,
    NumericDType.mul, NumericDType.div, NumericDType.sub, IntegralDType.floorDiv,
    IntegralDType.mod, ComparableDType.lt]

theorem z_init_eval (s : BlockState) (M N : Nat) :
    evalOp (Op.full [M, N] (Op.const 0)) s
      = some (⟨fun _ : TileIndex [M, N] => some (0 : ℝ)⟩ : Tile .real [M, N]) := by
  simp [evalOp_full, evalOp_const, Option.bind]

set_option maxHeartbeats 1000000 in
/-- **preLoop** (statements 0–14): from a clean input state (`undef = 0`), the
prologue steps to a state satisfying `matmulInvariant … 0` (accumulator = 0,
pointers seeded). -/
theorem preLoop (A B : RegionName) (s : BlockState)
    (M N K SAM SAK SBK SBN BM BN BK GM numKBlocks : Nat)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ s', stepStmts (matmulPreBody A B M N K SAM SAK SBK SBN BM BN BK GM) s = some s'
      ∧ matmulInvariant A B s M N BM BN GM SAM SAK SBK SBN BK numKBlocks 0 s' := by
  obtain ⟨s12, h12, hpids, ham, hbn, hk, huf, hmem⟩ :=
    preLoop_scalars A B s M N K SAM SAK SBK SBN BM BN BK GM
  rw [show matmulPreBody A B M N K SAM SAK SBK SBN BM BN BK GM
      = (matmulPreBody A B M N K SAM SAK SBK SBN BM BN BK GM).take 12
        ++ [ Stmt.assign .ptr [BM, BK] "a_ptrs"
              (Op.ptrAdd Broadcast.scalarL (Op.ptrBase A)
                (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
                  (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_am")) (Op.constNat SAM))
                  (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BK] "offs_k")) (Op.constNat SAK)))),
            Stmt.assign .ptr [BK, BN] "b_ptrs"
              (Op.ptrAdd Broadcast.scalarL (Op.ptrBase B)
                (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
                  (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "offs_k")) (Op.constNat SBK))
                  (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_bn")) (Op.constNat SBN)))),
            Stmt.assign .real [BM, BN] "accumulator" (Op.full [BM, BN] (Op.const 0)) ] from rfl,
    stepStmts.append_some h12,
    stepStmts.cons_some (stepStmt_assign_eq_some
      (aptrs_eval s12 A BM BK SAM SAK (fun i => rowIndex s M N BM BN GM i) (by simpa using ham) (by simpa using hk))),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (bptrs_eval _ B BK BN SBK SBN (fun j => colIndex s M N BM BN GM j) (by simp [hk]) (by simp [hbn]))),
    stepStmts.cons_some (stepStmt_assign_eq_some (z_init_eval _ BM BN)),
    stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  refine ⟨by simp [hpids], by simp, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · -- accumulator = accPartial 0 = 0
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx
    simp only [accPartial, Nat.zero_mul, Finset.range_zero, Finset.sum_empty]
  · simp [ham]
  · simp [hbn]
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

/-- **`acc += dot(a,b)` statement eval.** -/
theorem accdot_op_eval (M K N : Nat) (st : BlockState)
    (zt : Tile .real [M, N]) (at_ : Tile .real [M, K]) (bt : Tile .real [K, N])
    (hz : st.regs .real [M, N] "accumulator" = some zt)
    (ha : st.regs .real [M, K] "a" = some at_)
    (hb : st.regs .real [K, N] "b" = some bt) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [M, N] "accumulator")
        (Op.dot (batch := []) (Op.ref .real [M, K] "a") (Op.ref .real [K, N] "b"))) st
      = some (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          zt (Tile.dot [] at_ bt)) := by
  have hd : evalOp (Op.dot (batch := []) (Op.ref .real [M, K] "a")
        (Op.ref .real [K, N] "b")) st = some (Tile.dot [] at_ bt) := by
    rw [evalOp_dot]; simp [ha, hb]
  rw [evalOp_add]
  simp only [evalOp_ref, hz, bind, Option.bind_some]
  erw [hd]
  rfl

/-- `acc + dot` lane `(i,j)`: `some (zv + dv)`. -/
theorem zadd_eval (M N : Nat) (zt dt : Tile .real [M, N]) (i : Fin M) (j : Fin N) (zv dv : ℝ)
    (hz : zt.data (i, j, PUnit.unit) = some zv) (hd : dt.data (i, j, PUnit.unit) = some dv) :
    (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) zt dt).data
        (i, j, PUnit.unit) = some (zv + dv) := by
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, hz, hd, NumericDType.add,
    WithBot.realAdd, Option.map₂, Option.bind, Option.map]

/-- `a_ptrs += BLOCK_K · stride_ak` eval. -/
theorem aptr_adv_eval (s : BlockState) (M BK BKv SAK : Nat) (ap : Tile .ptr [M, BK])
    (hx : s.regs .ptr [M, BK] "a_ptrs" = some ap) :
    evalOp (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [M, BK] "a_ptrs")
      (Op.mul .nat Broadcast.nil (Op.constNat BKv) (Op.constNat SAK))) s
      = some (Tile.ptrAdd Broadcast.scalarR ap (Tile.scalar (BKv * SAK))) := by
  rw [evalOp_ptrAdd]
  simp [evalOp_ref, hx, evalOp_mul, evalOp_constNat, NumericDType.mul, Tile.bop]

/-- `b_ptrs += BLOCK_K · stride_bk` eval. -/
theorem bptr_adv_eval (s : BlockState) (BK N BKv SBK : Nat) (bp : Tile .ptr [BK, N])
    (hy : s.regs .ptr [BK, N] "b_ptrs" = some bp) :
    evalOp (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BK, N] "b_ptrs")
      (Op.mul .nat Broadcast.nil (Op.constNat BKv) (Op.constNat SBK))) s
      = some (Tile.ptrAdd Broadcast.scalarR bp (Tile.scalar (BKv * SBK))) := by
  rw [evalOp_ptrAdd]
  simp [evalOp_ref, hy, evalOp_mul, evalOp_constNat, NumericDType.mul, Tile.bop]

set_option maxHeartbeats 4000000 in
/-- **Step lemma**: one K-loop body iteration advances the invariant by one
block. `acc += tl.dot(a,b)` adds the `i`-th block's masked dot to the partial
GEMM accumulator; the `a`/`b` pointers advance one step. Requires the exact
contraction `K = BLOCK_K · numKBlocks` (so the load masks are all-true within
the loop range). -/
theorem matmul_step (A B : RegionName) (s0 : BlockState)
    (M N K BM BN GM SAM SAK SBK SBN BK numKBlocks : Nat) (hBK : 0 < BK)
    (hK : K = BK * numKBlocks)
    (i : Nat) (s : BlockState) (hilt : i < numKBlocks)
    (hinv : matmulInvariant A B s0 M N BM BN GM SAM SAK SBK SBN BK numKBlocks i s) :
    ∃ s', stepStmts (matmulLoopBody BM BK BN K SAK SBK)
        (s.setReg "k" .nat [] (Tile.scalar i)) = some s'
      ∧ matmulInvariant A B s0 M N BM BN GM SAM SAK SBK SBN BK numKBlocks (i + 1) s' := by
  subst hK
  simp only [matmulInvariant] at hinv
  obtain ⟨hpids, hcle, hz, ham, hbn, hk, hap, hbp, hundef, hmem⟩ := hinv
  -- mask threshold: K - i·BK = (numKBlocks - i)·BK ≥ BK, so every e < BK is active
  have hactive : ∀ e : Fin BK, e.val < BK * numKBlocks - i * BK := by
    intro e
    have : BK * numKBlocks - i * BK = (numKBlocks - i) * BK := by
      rw [Nat.mul_comm BK numKBlocks, ← Nat.sub_mul]
    rw [this]
    have h1 : 1 ≤ numKBlocks - i := by omega
    calc e.val < BK := e.isLt
      _ = 1 * BK := (Nat.one_mul BK).symm
      _ ≤ (numKBlocks - i) * BK := Nat.mul_le_mul_right _ h1
  -- abbreviations
  set apT : Tile .ptr [BM, BK] :=
    ⟨fun idx : TileIndex [BM, BK] => (A.cast, rowIndex s0 M N BM BN GM idx.1 * SAM + idx.2.1.val * SAK + i * BK * SAK)⟩ with hapT
  set bpT : Tile .ptr [BK, BN] :=
    ⟨fun idx : TileIndex [BK, BN] => (B.cast, idx.1.val * SBK + colIndex s0 M N BM BN GM idx.2.1 * SBN + i * BK * SBK)⟩ with hbpT
  set zT : Tile .real [BM, BN] :=
    ⟨fun idx : TileIndex [BM, BN] => some (accPartial s0 A B M N BM BN GM SAM SAK SBK SBN BK idx.1 idx.2.1 i)⟩ with hzT
  set sk := s.setReg "k" .nat [] (Tile.scalar i) with hsk
  have hrmem : ∀ (R : RegionName) (o : Nat), sk.readMem R o = s0.readMem R o := by
    intro R o; simp only [hsk, BlockState.setReg_readMem]; unfold BlockState.readMem; rw [hmem]
  have hapk : sk.regs .ptr [BM, BK] "a_ptrs" = some apT := by simp [hsk, hap, hapT]
  have hbpk : sk.regs .ptr [BK, BN] "b_ptrs" = some bpT := by simp [hsk, hbp, hbpT]
  have hzk : sk.regs .real [BM, BN] "accumulator" = some zT := by simp [hsk, hz, hzT]
  have hkk : sk.regs .nat [] "k" = some (Tile.scalar i) := by simp [hsk]
  have hoffk : sk.regs .nat [BK] "offs_k" = some (Tile.vec (fun e : Fin BK => e.val)) := by simp [hsk, hk]
  -- masked load tiles (masks all-true ⇒ pure readMem)
  set amaskT : Tile .bool [BM, BK] := ⟨fun idx => decide (idx.2.1.val < BK * numKBlocks - i * BK)⟩ with hamask
  set bmaskT : Tile .bool [BK, BN] := ⟨fun idx => decide (idx.1.val < BK * numKBlocks - i * BK)⟩ with hbmask
  set aload : Tile .real [BM, BK] :=
    ⟨fun idx => some (if amaskT.data idx then sk.readMem (apT.data idx).1 (apT.data idx).2 else 0)⟩ with haload
  set bload : Tile .real [BK, BN] :=
    ⟨fun idx => some (if bmaskT.data idx then sk.readMem (bpT.data idx).1 (bpT.data idx).2 else 0)⟩ with hbload
  set s1 := sk.setReg "a" .real [BM, BK] aload with hs1
  set s2 := s1.setReg "b" .real [BK, BN] bload with hs2
  set s3 := s2.setReg "accumulator" .real [BM, BN]
    (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
      zT (Tile.dot [] aload bload)) with hs3
  set s4 := s3.setReg "a_ptrs" .ptr [BM, BK] (Tile.ptrAdd Broadcast.scalarR apT (Tile.scalar (BK * SAK))) with hs4
  unfold matmulLoopBody
  rw [stepStmts.cons_some (s' := s1) (stepStmt_assign_eq_some
        (load_ptr_maskOther0_real (Op.ref .ptr [BM, BK] "a_ptrs") _ sk apT amaskT
          (by rw [evalOp_ref]; simp [hapk])
          (amask_eval sk BM BK (BK * numKBlocks) i hoffk hkk))),
    stepStmts.cons_some (s' := s2) (stepStmt_assign_eq_some
        (load_ptr_maskOther0_real (Op.ref .ptr [BK, BN] "b_ptrs") _ s1 bpT bmaskT
          (by rw [evalOp_ref]; simp [hs1, hbpk])
          (bmask_eval s1 BK BN (BK * numKBlocks) i (by simp [hs1, hoffk]) (by simp [hs1, hkk])))),
    stepStmts.cons_some (s' := s3) (stepStmt_assign_eq_some
        (accdot_op_eval BM BK BN s2 zT aload bload
          (by simp [hs2, hs1, hzk])
          (by simp [hs2, hs1, haload, BlockState.setReg_readMem])
          (by simp [hs2, hbload, BlockState.setReg_readMem]))),
    stepStmts.cons_some (s' := s4) (stepStmt_assign_eq_some
        (aptr_adv_eval s3 BM BK BK SAK apT (by simp [hs3, hs2, hs1, hapk]))),
    stepStmts.cons_some (stepStmt_assign_eq_some
        (bptr_adv_eval s4 BK BN BK SBK bpT (by simp [hs4, hs3, hs2, hs1, hbpk]))),
    stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  simp only [matmulInvariant, hs4, hs3, hs2, hs1, hsk]
  refine ⟨by simp [hpids], by omega, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · -- accumulator = accPartial (i+1)
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx
    have has : ∀ e : Fin BK, aload.data (idx.1, e, PUnit.unit)
        = some (if e.val < BK * numKBlocks - i * BK then
            aElem s0 A M N BM BN GM SAM SAK idx.1 (i * BK + e.val) else 0) := by
      intro e
      simp only [haload, hamask, hapT, hrmem]
      by_cases he : e.val < BK * numKBlocks - i * BK
      · rw [if_pos he, if_pos (by exact decide_eq_true he)]
        congr 1
        simp only [aElem]
        congr 1
        ring
      · rw [if_neg he, if_neg (by simp [he])]
    have hbs : ∀ e : Fin BK, bload.data (e, idx.2.1, PUnit.unit)
        = some (if e.val < BK * numKBlocks - i * BK then
            bElem s0 B M N BM BN GM SBK SBN idx.2.1 (i * BK + e.val) else 0) := by
      intro e
      simp only [hbload, hbmask, hbpT, hrmem]
      by_cases he : e.val < BK * numKBlocks - i * BK
      · rw [if_pos he, if_pos (by exact decide_eq_true he)]
        congr 1
        simp only [bElem]
        congr 1
        ring
      · rw [if_neg he, if_neg (by simp [he])]
    rw [zadd_eval BM BN zT (Tile.dot [] aload bload) idx.1 idx.2.1
        (accPartial s0 A B M N BM BN GM SAM SAK SBK SBN BK idx.1 idx.2.1 i)
        (Finset.univ.sum fun e : Fin BK =>
          if e.val < BK * numKBlocks - i * BK then
            aElem s0 A M N BM BN GM SAM SAK idx.1 (i * BK + e.val)
              * bElem s0 B M N BM BN GM SBK SBN idx.2.1 (i * BK + e.val) else 0)
        (by rw [hzT])
        (dot_masked aload bload idx.1 idx.2.1 _ _ has hbs)]
    show some _ = some (accPartial s0 A B M N BM BN GM SAM SAK SBK SBN BK idx.1 idx.2.1 (i + 1))
    rw [accPartial_succ]
    refine congrArg some (congrArg _ ?_)
    apply Finset.sum_congr rfl
    intro e _
    rw [if_pos (hactive e)]
  · simp [ham]
  · simp [hbn]
  · simp [hk]
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
  · intro rg o; simp [hundef]
  · exact hmem

/-! ## Masked store (postLoop) -/

/-- Active output lane: `rowIndex i < M ∧ colIndex j < N`. -/
def active (s0 : BlockState) (M N BM BN GM : Nat) (idx : TileIndex [BM, BN]) : Prop :=
  rowIndex s0 M N BM BN GM idx.1 < M ∧ colIndex s0 M N BM BN GM idx.2.1 < N

instance (s0 : BlockState) (M N BM BN GM : Nat) (idx : TileIndex [BM, BN]) :
    Decidable (active s0 M N BM BN GM idx) := by unfold active; infer_instance

/-- The output store address for tile lane `(i,j)`: `rowIndex i · SCM + colIndex j · SCN`. -/
def cOffset (s0 : BlockState) (M N BM BN GM SCM SCN : Nat) (idx : TileIndex [BM, BN]) : Nat :=
  rowIndex s0 M N BM BN GM idx.1 * SCM + colIndex s0 M N BM BN GM idx.2.1 * SCN

/-- **`c_mask` eval**: `(offs_am[:,None] < M) & (offs_bn[None,:] < N)`, lane `(i,j)`
is `rowIndex i < M ∧ colIndex j < N`. -/
theorem cmask_eval (s : BlockState) (M N BM BN _GM : Nat) (gm : Fin BM → Nat) (gn : Fin BN → Nat)
    (ham : s.regs .nat [BM] "offs_am" = some (Tile.vec gm))
    (hbn : s.regs .nat [BN] "offs_bn" = some (Tile.vec gn)) :
    evalOp (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_am")) (Op.constNat M))
        (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_bn")) (Op.constNat N))) s
      = some ⟨fun idx : TileIndex [BM, BN] => (decide (gm idx.1 < M) && decide (gn idx.2.1 < N))⟩ := by
  rw [evalOp_boolAnd]
  conv_lhs => arg 1; rw [evalOp_lt]; arg 1; rw [evalOp_expandDim_one_nat, ham]
  simp only [Option.bind_eq_bind, Option.bind_some, evalOp_constNat]
  conv_lhs => arg 1; rw [evalOp_lt]; arg 1; rw [evalOp_expandDim_zero_nat, hbn]
  simp only [Option.bind_eq_bind, Option.bind_some, evalOp_constNat]
  refine congrArg some ?_
  ext idx
  simp [Tile.bop, Tile.cop, Tile.expandDim, Tile.vec, ComparableDType.lt]

set_option maxHeartbeats 2000000 in
/-- **postLoop**: from the invariant at `numKBlocks` blocks, the masked store of
`accumulator` to `c_ptrs` writes the genuine GEMM value `matmulSpec` at every
active output lane (given the output-offset map is injective). -/
theorem matmul_postLoop (A B C : RegionName) (s0 : BlockState)
    (M N K BM BN GM SAM SAK SBK SBN SCM SCN BK numKBlocks : Nat) (hBK : 0 < BK)
    (hK : K = BK * numKBlocks)
    (hInj : Function.Injective (cOffset s0 M N BM BN GM SCM SCN))
    (st : BlockState)
    (hinv : matmulInvariant A B s0 M N BM BN GM SAM SAK SBK SBN BK numKBlocks numKBlocks st) :
    ∃ sfin, stepStmts (matmulTailStmts C BM BN M N SCM SCN) st = some sfin
      ∧ ∀ idx : TileIndex [BM, BN],
          sfin.readMem C (cOffset s0 M N BM BN GM SCM SCN idx)
            = if active s0 M N BM BN GM idx then
                matmulSpec s0 A B M N BM BN GM SAM SAK SBK SBN BK numKBlocks idx.1 idx.2.1
              else st.readMem C (cOffset s0 M N BM BN GM SCM SCN idx) := by
  simp only [matmulInvariant] at hinv
  obtain ⟨hpids, hcle, hz, ham, hbn, hk, hap, hbp, hundef, hmem⟩ := hinv
  set zT : Tile .real [BM, BN] :=
    ⟨fun idx : TileIndex [BM, BN] =>
      some (accPartial s0 A B M N BM BN GM SAM SAK SBK SBN BK idx.1 idx.2.1 numKBlocks)⟩ with hzT
  set cpT : Tile .ptr [BM, BN] :=
    ⟨fun idx : TileIndex [BM, BN] =>
      (C.cast, rowIndex s0 M N BM BN GM idx.1 * SCM + colIndex s0 M N BM BN GM idx.2.1 * SCN)⟩ with hcpT
  set cmT : Tile .bool [BM, BN] :=
    ⟨fun idx => (decide (rowIndex s0 M N BM BN GM idx.1 < M) && decide (colIndex s0 M N BM BN GM idx.2.1 < N))⟩ with hcmT
  set s1 := st.setReg "c_ptrs" .ptr [BM, BN] cpT with hs1
  set s2 := s1.setReg "c_mask" .bool [BM, BN] cmT with hs2
  unfold matmulTailStmts
  rw [stepStmts.cons_some (s' := s1) (stepStmt_assign_eq_some
        (cptrs_eval st C BM BN SCM SCN (fun i => rowIndex s0 M N BM BN GM i)
          (fun j => colIndex s0 M N BM BN GM j) ham hbn)),
    stepStmts.cons_some (s' := s2) (stepStmt_assign_eq_some
        (cmask_eval s1 M N BM BN GM (fun i => rowIndex s0 M N BM BN GM i)
          (fun j => colIndex s0 M N BM BN GM j) (by simp [hs1, ham]) (by simp [hs1, hbn])))]
  -- the masked store
  have hstore : stepStmt
      (Stmt.store .real [BM, BN] (MemAccess.ptr (Op.ref .ptr [BM, BN] "c_ptrs"))
        (Op.ref .real [BM, BN] "accumulator") (MaskOpt.mask (Op.ref .bool [BM, BN] "c_mask"))) s2
      = some ((TileShape.allIndices [BM, BN]).foldl
          (fun acc k =>
            if active s0 M N BM BN GM k then
              acc.writeMem C (cOffset s0 M N BM BN GM SCM SCN k)
                (accPartial s0 A B M N BM BN GM SAM SAK SBK SBN BK k.1 k.2.1 numKBlocks)
            else acc) s2) := by
    have hacc : s2.regs .real [BM, BN] "accumulator" = some zT := by simp [hs2, hs1, hz, hzT]
    have hcp : s2.regs .ptr [BM, BN] "c_ptrs" = some cpT := by simp [hs2, hs1, hcpT]
    have hcm : s2.regs .bool [BM, BN] "c_mask" = some cmT := by simp [hs2, hcmT]
    simp only [stepStmt, evalOp, evalOp.eq_def, hacc, hcp, hcm]
    simp only [bind, Option.bind_some, Option.map]
    refine congrArg some (List.foldl_ext _ _ s2 (fun acc k _ => ?_))
    have hcmval : (cmT.data k = Bool.true) ↔ active s0 M N BM BN GM k := by
      simp only [cmT, hcmT, active, rowIndex, colIndex, Bool.and_eq_true, decide_eq_true_eq]
    by_cases hact : active s0 M N BM BN GM k
    · rw [if_pos (hcmval.mpr hact), if_pos hact]
      simp only [cpT, hcpT, cOffset, zT, hzT, rowIndex, colIndex,
        BlockState.writeMemTyped_real, Region.cast_id, FloatDType.real_storeValue, WithBot.unbotD_some]
    · rw [if_neg (fun h => hact (hcmval.mp h)), if_neg hact]
  rw [stepStmts.cons_some hstore, stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  intro idx
  rw [BlockState.scatter_readback_prop_masked_nd (region := C) (s := s2)
      (offsetFn := cOffset s0 M N BM BN GM SCM SCN)
      (valueFn := fun k => accPartial s0 A B M N BM BN GM SAM SAK SBK SBN BK k.1 k.2.1 numKBlocks)
      (P := active s0 M N BM BN GM) hInj idx]
  by_cases hact : active s0 M N BM BN GM idx
  · rw [if_pos hact, if_pos hact]
    simp only [matmulSpec, accPartial, Nat.mul_comm numKBlocks BK]
  · rw [if_neg hact, if_neg hact]
    show s2.readMem C _ = st.readMem C _
    simp [hs2, hs1, BlockState.setReg_readMem]

/-! ## Composition: full exec closed form -/

/-- `kStopOp` evaluates to `cdiv K BK = numKBlocks` when `K = BK · numKBlocks`. -/
theorem kStopOp_eval (s : BlockState) (BK numKBlocks : Nat) (hBK : 0 < BK) :
    evalOp (kStopOp (BK * numKBlocks) BK) s = some (Tile.scalar numKBlocks) := by
  unfold kStopOp
  simp only [evalOp_div, evalOp_sub, evalOp_add, evalOp_constNat, Option.bind_some,
    Tile.bop, NumericDType.div, NumericDType.sub, NumericDType.add]
  refine congrArg some ?_
  ext
  show (BK * numKBlocks + BK - 1) / BK = numKBlocks
  rw [show BK * numKBlocks + BK - 1 = (BK - 1) + numKBlocks * BK by ring_nf; omega,
    Nat.add_mul_div_right _ _ hBK, Nat.div_eq_of_lt (by omega), Nat.zero_add]

set_option maxHeartbeats 2000000 in
/-- **Top exec reduction**: composes `preLoop` + `matmul_step` (driven by
`forRangeDyn_inv`) + `matmul_postLoop` into the full `exec` result. Every active
output lane equals the genuine GEMM value `matmulSpec`. -/
theorem matmul_triton2_exec_closed_form (A B C : RegionName) (s : BlockState)
    (M N BM BN GM SAM SAK SBK SBN SCM SCN BK numKBlocks : Nat) (hBK : 0 < BK)
    (hInj : Function.Injective (cOffset s M N BM BN GM SCM SCN))
    (hundef : ∀ rg o, s.undef rg o = 0)
    (idx : TileIndex [BM, BN]) :
    (match exec (matmul_triton2_surface A B C M N (BK * numKBlocks) SAM SAK SBK SBN SCM SCN
        BM BN BK GM) s with
      | some s' => s'.readMem C (cOffset s M N BM BN GM SCM SCN idx)
      | none => (0.0 : ℝ)) =
      (if active s M N BM BN GM idx then
        matmulSpec s A B M N BM BN GM SAM SAK SBK SBN BK numKBlocks idx.1 idx.2.1
      else s.readMem C (cOffset s M N BM BN GM SCM SCN idx)) := by
  -- preLoop establishes P 0
  obtain ⟨s0, hpre_eq, hP0⟩ :=
    preLoop A B s M N (BK * numKBlocks) SAM SAK SBK SBN BM BN BK GM numKBlocks hundef
  -- drive the K-loop via forRangeDyn_inv
  obtain ⟨final, sLoop, hLoopStmt, hfinal, hPLoop⟩ :=
    forRangeDyn_inv (idx := "k") (start := 0) (stop := numKBlocks) (step := 1)
      (startOp := Op.constNat 0) (stopOp := kStopOp (BK * numKBlocks) BK) (stepOp := Op.constNat 1)
      (by simp [evalOp_constNat]) (kStopOp_eval s0 BK numKBlocks hBK) (by simp [evalOp_constNat])
      (by omega) hP0
      (fun i st hlt hinv => matmul_step A B s M N (BK * numKBlocks) BM BN GM SAM SAK SBK SBN BK
        numKBlocks hBK rfl i st hlt hinv)
  -- loop exit: final = numKBlocks
  have hfinalEq : final = numKBlocks := by
    have hle : final ≤ numKBlocks := by
      simp only [matmulInvariant] at hPLoop
      exact hPLoop.2.1
    omega
  rw [hfinalEq] at hPLoop
  -- postLoop reads off the closed form
  obtain ⟨sfin, hTail, hpost⟩ :=
    matmul_postLoop A B C s M N (BK * numKBlocks) BM BN GM SAM SAK SBK SBN SCM SCN BK numKBlocks
      hBK rfl hInj sLoop hPLoop
  have hexec : exec (matmul_triton2_surface A B C M N (BK * numKBlocks) SAM SAK SBK SBN SCM SCN
      BM BN BK GM) s = some sfin := by
    rw [exec, matmul_body_split A B C M N (BK * numKBlocks) SAM SAK SBK SBN SCM SCN BM BN BK GM,
      matmul_preBody_eq, stepStmts.append_some hpre_eq, stepStmts.cons_some hLoopStmt, hTail]
  -- the loop preserves C's memory (invariant: sLoop.mem = s.mem)
  have hLoopMem : sLoop.readMem C (cOffset s M N BM BN GM SCM SCN idx)
      = s.readMem C (cOffset s M N BM BN GM SCM SCN idx) := by
    simp only [matmulInvariant] at hPLoop
    show BlockState.readMem _ _ _ = BlockState.readMem _ _ _
    unfold BlockState.readMem
    rw [hPLoop.2.2.2.2.2.2.2.2.2]
  rw [hexec]
  show sfin.readMem C _ = _
  rw [hpost idx, hLoopMem]

/-- **Closed-form correctness for `matmul_triton2` (general statement).**

For arbitrary matrix dims `M`/`N`, tile dims `BM`/`BN`, group size `GM`, strides,
K-block size `BK`, and K-block count `numKBlocks` (so `K = BK · numKBlocks`),
every **active** output lane of the computed `BM × BN` tile equals the genuine
matrix product `Σ_{k < BK·numKBlocks} A[i,k] · B[k,j]` (over ℝ) of the loaded
`A`/`B` tiles — NOT the kernel's own executed value. Inactive lanes (out of the
`M × N` bounds) are preserved. -/
theorem matmul_triton2_closed_form_correct
    (A B C : RegionName) (s : BlockState)
    (M N BM BN GM SAM SAK SBK SBN SCM SCN BK numKBlocks : Nat) (hBK : 0 < BK)
    (hInj : Function.Injective (cOffset s M N BM BN GM SCM SCN))
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes
      (kernel := matmul_triton2_surface A B C M N (BK * numKBlocks) SAM SAK SBK SBN SCM SCN
        BM BN BK GM)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s M N BM BN GM)
        (fun idx => (C, cOffset s M N BM BN GM SCM SCN idx)))
      (expected := fun idx : TileIndex [BM, BN] =>
        matmulSpec s A B M N BM BN GM SAM SAK SBK SBN BK numKBlocks idx.1 idx.2.1) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [matmul_triton2_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst hs0
  intro idx hActive
  have hmain := matmul_triton2_exec_closed_form A B C s0 M N BM BN GM SAM SAK SBK SBN SCM SCN BK
    numKBlocks hBK hInj hundef idx
  have hExec2 : exec (matmul_triton2_surface A B C M N (BK * numKBlocks) SAM SAK SBK SBN SCM SCN
      BM BN BK GM) s0 = some s' := hExec
  rw [hExec2, if_pos hActive] at hmain
  simpa only [ComputeCorrect.OutputReadable.read_real] using hmain

/-! ## Public dimension-general output summary -/

/-- **Public dimension-general summary**: for arbitrary matrix dims `M`/`N`, tile
dims `BM`/`BN`, group size `GM`, strides, K-block size `BK` (`0 < BK`), and
K-block count `numKBlocks` (so `K = BK · numKBlocks`), the full `matmul_triton2`
surface (1) lowers to the algorithm layer and (2) realizes the genuine matrix
product `Σ_{k < BK·numKBlocks} A[i,k]·B[k,j]` (over ℝ, reading INPUT memory) on
every active output lane. The `expected` is the closed-form `matmulSpec`, NOT the
kernel's own executed value. -/
theorem matmul_triton2_output_summary_general
    (A B C : RegionName) (s : BlockState)
    (M N BM BN GM SAM SAK SBK SBN SCM SCN BK numKBlocks : Nat) (hBK : 0 < BK)
    (hInj : Function.Injective (cOffset s M N BM BN GM SCM SCN))
    (hundef : ∀ rg o, s.undef rg o = 0) :
    (∃ alg, (matmul_triton2_surface A B C M N (BK * numKBlocks) SAM SAK SBK SBN SCM SCN
        BM BN BK GM).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := matmul_triton2_surface A B C M N (BK * numKBlocks) SAM SAK SBK SBN SCM SCN
        BM BN BK GM)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s M N BM BN GM)
        (fun idx => (C, cOffset s M N BM BN GM SCM SCN idx)))
      (expected := fun idx : TileIndex [BM, BN] =>
        matmulSpec s A B M N BM BN GM SAM SAK SBK SBN BK numKBlocks idx.1 idx.2.1) :=
  ⟨matmul_triton2_surface_toAlgorithm_supported A B C M N (BK * numKBlocks) SAM SAK SBK SBN
      SCM SCN BM BN BK GM,
   matmul_triton2_closed_form_correct A B C s M N BM BN GM SAM SAK SBK SBN SCM SCN BK numKBlocks
      hBK hInj hundef⟩

end VeriTile.Bench.TritonBenchG.MatmulTriton2
