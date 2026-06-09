import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.ScatterStore
import VeriTile.Triton.LoopInvariant
import VeriTile.Triton.Math.Matmul
import VeriTile.Triton.Math.OffsetInjective

/-!
# `matmul_triton_autotune` — closed-form matmul+activation correctness

`matmul_triton_autotune.py`'s `matmul_kernel` is an L2-grouped, autotuned tiled
GEMM `C = act(A × B)`: a single linear `pid` is split into an `(pid_m, pid_n)`
tile coordinate via the L2-grouping schedule, a `BLOCK_SIZE_M × BLOCK_SIZE_N`
output tile is accumulated by the fused `accumulator = tl.dot(a, b, accumulator)`
reduction over the K dimension (with the per-block `offs_k < K - k·BLOCK_K` load
masks), an optional `leaky_relu` activation tail is applied, the accumulator is
downcast to `float16`, and the tile is stored to `C` under the
`(row<M) & (col<N)` boundary mask.

This file proves the **full K-loop + activation tail** correct against a genuine
mathematical matrix product: every active output cell `C[i,j]` of the computed
tile equals `fp16( act( Σ_{k < K} A[i,k] · B[k,j] ) )` over `ℝ`, where
`K = BLOCK_SIZE_K · numKBlocks` and `act` is `leakyRelu` when `ACTIVATION = true`
and the identity otherwise. This is NOT the kernel's own emitted value — the
real-valued `Σ_k A·B` GEMM reference and the activation are derived
independently of the kernel from the loaded `A`/`B` tiles.

## Proof architecture

```
matmul_autotune_closed_form_correct               ← TOP THEOREM (ComputeCorrect.Realizes)
  └─ matmul_autotune_exec_closed_form             ← exec-side closed form (every active cell)
       ├─ matmul_preLoop      (P 0: accumulator = 0, pointers seeded, schedule derived)
       ├─ matmul_step         (one K-block: masked dot advances the partial sum)
       ├─ matmul_postLoop     (activation tail + fp16 cast + masked store = closed form)
       └─ forRange_inv        (loop-invariant principle, drives the K-loop)
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); the dot reduction is
accumulated over `ℝ` and the modeled `tl.cast(..., fp16)` is the placeholder
`FloatDType.real.cast .fp16`. The contracted dimension is presented as
`K = BLOCK_SIZE_K · numKBlocks` so the loop trip count `cdiv(K, BLOCK_SIZE_K) =
numKBlocks` is exact and the per-block load masks `offs_k < K - k·BLOCK_K` are
uniformly satisfied (the kernel's tail-masking is genuinely modeled but vacuous
at exact-multiple K). `@triton.autotune`, `num_warps`/`num_stages`, and the
autotune-config choice are NOT modeled: the surface is a single concrete config,
the trusted boundary. The host launch (grid, linear-pid scheduling) is trusted;
the per-program statement is universally quantified over `s`, covering every
program of the grid. The L2-grouping index math (`pid → (pid_m, pid_n)`) is
transcribed exactly as the kernel computes it and the spec's layout references
the same derived `offs_am`/`offs_bn`/`offs_cm`/`offs_cn`, so it is not a separate
proof obligation. Output-offset injectivity (distinct lanes hit distinct
addresses) is the only assumed disjointness.
-/

namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune

open VeriTile.Triton
open VeriTile.Triton.Math

set_option linter.unusedSimpArgs false

/-- Faithful transcription of `matmul_triton_autotune.py`'s `matmul_kernel`.

The contracted dimension is presented as `K = BLOCK_SIZE_K · numKBlocks` so the
loop bound `tl.cdiv(K, BLOCK_SIZE_K) = numKBlocks` is exact; it is supplied as the
antiquoted `numKBlocks`. All other surface structure — the L2-grouping schedule,
the per-block `offs_k < K - k·BLOCK_K` load masks, the fused
`tl.dot(a, b, accumulator)`, the optional `leaky_relu`, the `float16` cast, and
the `(row<M)&(col<N)`-masked store — is transcribed verbatim. The Python string
constexpr `ACTIVATION == "leaky_relu"` is the Lean `Bool` parameter `ACTIVATION`. -/
def matmul_autotune_surface
    (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M numKBlocks : Nat)
    (ACTIVATION : Bool) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  num_pid_m = tl.cdiv($(M), $(BLOCK_SIZE_M))
  num_pid_n = tl.cdiv($(N), $(BLOCK_SIZE_N))
  num_pid_in_group = $(GROUP_SIZE_M) * num_pid_n
  group_id = pid // num_pid_in_group
  first_pid_m = group_id * $(GROUP_SIZE_M)
  group_size_m = min(num_pid_m - first_pid_m, $(GROUP_SIZE_M))
  pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
  pid_n = (pid % num_pid_in_group) // group_size_m
  offs_am = (pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))) % $(M)
  offs_bn = (pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))) % $(N)
  offs_k = tl.arange(0, $(BLOCK_SIZE_K))
  a_ptrs = A + offs_am[:, None] * $(stride_am) + offs_k[None, :] * $(stride_ak)
  b_ptrs = B + offs_k[:, None] * $(stride_bk) + offs_bn[None, :] * $(stride_bn)
  accumulator = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
  for kk in range($(0), $(numKBlocks), $(1)) {
    a = tl.load(a_ptrs, mask=offs_k[None, :] < $(K) - kk * $(BLOCK_SIZE_K), other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < $(K) - kk * $(BLOCK_SIZE_K), other=0.0)
    accumulator = tl.dot(a, b, accumulator)
    a_ptrs += $(BLOCK_SIZE_K) * $(stride_ak)
    b_ptrs += $(BLOCK_SIZE_K) * $(stride_bk)
  }
  if ACTIVATION {
    accumulator = leaky_relu(accumulator)
  }
  c = (accumulator).to(tl.float16)
  offs_cm = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_cn = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  c_ptrs = C + $(stride_cm) * offs_cm[:, None] + $(stride_cn) * offs_cn[None, :]
  c_mask = (offs_cm[:, None] < $(M)) & (offs_cn[None, :] < $(N))
  tl.store(c_ptrs, c, mask=c_mask)
}

/-- The full autotuned matmul surface lowers to the algorithm layer. -/
theorem matmul_autotune_surface_toAlgorithm_supported
    (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M numKBlocks : Nat)
    (ACTIVATION : Bool) :
    ∃ alg, (matmul_autotune_surface A B C M N K stride_am stride_ak
      stride_bk stride_bn stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N
      BLOCK_SIZE_K GROUP_SIZE_M numKBlocks ACTIVATION).toAlgorithm? = Except.ok alg := by
  simp [matmul_autotune_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

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

theorem evalOp_remap {dtype shape} (outShape : TileShape)
    (map : TileIndex outShape → TileIndex shape) (a : Op dtype shape) (s : BlockState) :
    evalOp (.remap outShape map a) s = (do
      let v ← evalOp a s; some (Tile.remap map v)) := by simp [evalOp]

/-- A `.ptr` load with `mask=…, other=…` whose mask is **uniformly true** reduces
to the clean `readMem` load. (The kernel's `offs_k < K - k·BLOCK_K` mask is
uniformly true at exact-multiple `K`.) -/
theorem load_ptr_maskOther_true_real {shape : TileShape}
    (ptrOp : Op .ptr shape) (maskOp : Op .bool shape) (otherOp : Op .real shape)
    (s : BlockState) (ptrs : Tile .ptr shape) (masks : Tile .bool shape)
    (hp : evalOp ptrOp s = some ptrs) (hm : evalOp maskOp s = some masks)
    (others : Tile .real shape) (ho : evalOp otherOp s = some others)
    (hmask : ∀ i, masks.data i = Bool.true) :
    evalOp (.load .real (.ptr ptrOp) (.maskOther maskOp otherOp)) s
      = some ⟨fun i => some (s.readMem (ptrs.data i).1 (ptrs.data i).2)⟩ := by
  simp only [evalOp, hp, hm, ho, Option.bind]
  refine congrArg some ?_
  ext i
  simp only [hmask i, if_true, BlockState.readMemValue_real]

/-- `a_ptrs` eval: cell `(i,e) = (A, offs_am i · sam + offs_k e · sak)`. -/
theorem aptrs_eval (s : BlockState) (A : RegionName) (M K sam sak : Nat) (gm : Fin M → Nat)
    (hm : s.regs .nat [M] "offs_am" = some (Tile.vec gm))
    (hk : s.regs .nat [K] "offs_k" = some (Tile.vec (fun e : Fin K => e.val))) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase A)
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_am")) (Op.constNat sam))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [K] "offs_k")) (Op.constNat sak)))) s
      = some (⟨fun idx : TileIndex [M, K] => (A.cast, gm idx.1 * sam + idx.2.1.val * sak)⟩ : Tile .ptr [M, K]) := by
  rw [evalOp_ptrAdd, evalOp_ptrBase]
  simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref, evalOp_expandDim_one_nat,
    evalOp_expandDim_zero_nat, hm, hk, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec]
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec, NumericDType.add, NumericDType.mul]

/-- `b_ptrs` eval: cell `(e,j) = (B, offs_k e · sbk + offs_bn j · sbn)`. -/
theorem bptrs_eval (s : BlockState) (B : RegionName) (K N sbk sbn : Nat) (gn : Fin N → Nat)
    (hk : s.regs .nat [K] "offs_k" = some (Tile.vec (fun e : Fin K => e.val)))
    (hn : s.regs .nat [N] "offs_bn" = some (Tile.vec gn)) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase B)
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [K] "offs_k")) (Op.constNat sbk))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [N] "offs_bn")) (Op.constNat sbn)))) s
      = some (⟨fun idx : TileIndex [K, N] => (B.cast, idx.1.val * sbk + gn idx.2.1 * sbn)⟩ : Tile .ptr [K, N]) := by
  rw [evalOp_ptrAdd, evalOp_ptrBase]
  simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref, evalOp_expandDim_one_nat,
    evalOp_expandDim_zero_nat, hk, hn, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec]
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec, NumericDType.add, NumericDType.mul]

/-- `c_ptrs` eval: cell `(i,j) = (C, scm · offs_cm i + scn · offs_cn j)`
(strides on the **left** of the products). -/
theorem cptrs_eval (s : BlockState) (C : RegionName) (M N scm scn : Nat) (gm : Fin M → Nat) (gn : Fin N → Nat)
    (hm : s.regs .nat [M] "offs_cm" = some (Tile.vec gm))
    (hn : s.regs .nat [N] "offs_cn" = some (Tile.vec gn)) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase C)
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarL (Op.constNat scm) (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_cm")))
        (Op.mul .nat Broadcast.scalarL (Op.constNat scn) (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [N] "offs_cn"))))) s
      = some (⟨fun idx : TileIndex [M, N] => (C.cast, scm * gm idx.1 + scn * gn idx.2.1)⟩ : Tile .ptr [M, N]) := by
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

/-- `a_ptrs += BLOCK_K · sak` eval. -/
theorem aptr_adv_eval (s : BlockState) (M K BK sak : Nat) (ap : Tile .ptr [M, K])
    (ha : s.regs .ptr [M, K] "a_ptrs" = some ap) :
    evalOp (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [M, K] "a_ptrs")
      (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat sak))) s
      = some (Tile.ptrAdd Broadcast.scalarR ap (Tile.scalar (BK * sak))) := by
  rw [evalOp_ptrAdd]
  simp [evalOp_ref, ha, evalOp_mul, evalOp_constNat, NumericDType.mul, Tile.bop]

/-- `b_ptrs += BLOCK_K · sbk` eval. -/
theorem bptr_adv_eval (s : BlockState) (K N BK sbk : Nat) (bp : Tile .ptr [K, N])
    (hb : s.regs .ptr [K, N] "b_ptrs" = some bp) :
    evalOp (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [K, N] "b_ptrs")
      (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat sbk))) s
      = some (Tile.ptrAdd Broadcast.scalarR bp (Tile.scalar (BK * sbk))) := by
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

/-! ## Scheduling + GEMM closed-form spec -/

/-- Ceiling division `⌈a / b⌉`, matching Triton's `tl.cdiv`. -/
def cdiv (a b : Nat) : Nat := (a + b - 1) / b

/-- `min` as the kernel's `tl.where(a < b, a, b)` spells it. -/
def kernelMin (a b : Nat) : Nat := if a < b then a else b

/-- The kernel's L2-grouping derivation of `pid_m` from the linear `pid`. -/
def pidM (pid M N BM BN GM : Nat) : Nat :=
  let num_pid_m := cdiv M BM
  let num_pid_n := cdiv N BN
  let num_pid_in_group := GM * num_pid_n
  let group_id := pid / num_pid_in_group
  let first_pid_m := group_id * GM
  let group_size_m := kernelMin (num_pid_m - first_pid_m) GM
  first_pid_m + ((pid % num_pid_in_group) % group_size_m)

/-- The kernel's L2-grouping derivation of `pid_n` from the linear `pid`. -/
def pidN (pid M N BM BN GM : Nat) : Nat :=
  let num_pid_m := cdiv M BM
  let num_pid_n := cdiv N BN
  let num_pid_in_group := GM * num_pid_n
  let group_id := pid / num_pid_in_group
  let first_pid_m := group_id * GM
  let group_size_m := kernelMin (num_pid_m - first_pid_m) GM
  (pid % num_pid_in_group) / group_size_m

/-- Global output row of tile lane `i`: `pid_m · BLOCK_M + i`, **before** the
`% M` wrap (the kernel's `offs_cm`). -/
def rowGlobal (s : BlockState) (M N BM BN GM : Nat) (i : Fin BM) : Nat :=
  pidM (s.pids 0) M N BM BN GM * BM + i.val

/-- Global output column of tile lane `j`: `pid_n · BLOCK_N + j`, before wrap. -/
def colGlobal (s : BlockState) (M N BM BN GM : Nat) (j : Fin BN) : Nat :=
  pidN (s.pids 0) M N BM BN GM * BN + j.val

/-- The `% M`-wrapped A-row index of tile lane `i` (the kernel's `offs_am`). -/
def rowIndex (s : BlockState) (M N BM BN GM : Nat) (i : Fin BM) : Nat :=
  rowGlobal s M N BM BN GM i % M

/-- The `% N`-wrapped B-column index of tile lane `j` (the kernel's `offs_bn`). -/
def colIndex (s : BlockState) (M N BM BN GM : Nat) (j : Fin BN) : Nat :=
  colGlobal s M N BM BN GM j % N

/-- `A[i, k] = readMem A (offs_am i · stride_am + k · stride_ak)`. -/
noncomputable def aElem (s : BlockState) (A : RegionName) (M N BM BN GM sam sak : Nat)
    (i : Fin BM) (k : Nat) : ℝ :=
  s.readMem A (rowIndex s M N BM BN GM i * sam + k * sak)

/-- `B[k, j] = readMem B (k · stride_bk + offs_bn j · stride_bn)`. -/
noncomputable def bElem (s : BlockState) (B : RegionName) (M N BM BN GM sbk sbn : Nat)
    (j : Fin BN) (k : Nat) : ℝ :=
  s.readMem B (k * sbk + colIndex s M N BM BN GM j * sbn)

/-- Real-valued leaky-ReLU activation (slope `0.01` below zero), matching the
kernel's `leaky_relu`. -/
noncomputable def leakyReLU (x : ℝ) : ℝ := if x ≥ 0 then x else 0.01 * x

/-- The applied activation: `leakyReLU` when `ACTIVATION`, else the identity. -/
noncomputable def act (ACTIVATION : Bool) (x : ℝ) : ℝ :=
  if ACTIVATION then leakyReLU x else x

/-- **Genuine matmul+activation spec** (over ℝ):
`C[i,j] = act( Σ_{k < BLOCK_K·numKBlocks} A[i,k] · B[k,j] )`. -/
noncomputable def matmulSpec (s : BlockState) (A B : RegionName)
    (M N BM BN GM sam sak sbk sbn BLOCK_K numKBlocks : Nat) (ACTIVATION : Bool)
    (i : Fin BM) (j : Fin BN) : ℝ :=
  act ACTIVATION
    (gemmSum (aElem s A M N BM BN GM sam sak i) (bElem s B M N BM BN GM sbk sbn j)
      (BLOCK_K * numKBlocks))

/-- Partial GEMM accumulator after `c` K-blocks: `Σ_{k < c·BLOCK_K} A·B`. -/
noncomputable def accPartial (s : BlockState) (A B : RegionName)
    (M N BM BN GM sam sak sbk sbn BLOCK_K : Nat) (i : Fin BM) (j : Fin BN) (c : Nat) : ℝ :=
  gemmSum (aElem s A M N BM BN GM sam sak i) (bElem s B M N BM BN GM sbk sbn j) (c * BLOCK_K)

/-- One-block step of the partial accumulator (the shared `gemmSum_blockSucc`). -/
theorem accPartial_succ (s : BlockState) (A B : RegionName)
    (M N BM BN GM sam sak sbk sbn BLOCK_K : Nat) (i : Fin BM) (j : Fin BN) (c : Nat) :
    accPartial s A B M N BM BN GM sam sak sbk sbn BLOCK_K i j (c + 1)
      = accPartial s A B M N BM BN GM sam sak sbk sbn BLOCK_K i j c
        + (Finset.univ.sum fun e : Fin BLOCK_K =>
            aElem s A M N BM BN GM sam sak i (c * BLOCK_K + e.val)
              * bElem s B M N BM BN GM sbk sbn j (c * BLOCK_K + e.val)) :=
  gemmSum_blockSucc (aElem s A M N BM BN GM sam sak i) (bElem s B M N BM BN GM sbk sbn j) BLOCK_K c

/-! ## Body decomposition -/

/-- The masked-load `mask=` operand for the `a` load (`offs_k[None,:] < K - kk·BK`,
remapped over the row axis). -/
def aMaskOp (K BM BK : Nat) : Op .bool [BM, BK] :=
  Op.remap [BM, BK] Broadcast.nil.consSame.consL.leftIndex
    (Op.lt ComparableDType.nat Broadcast.scalarR
      (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BK] "offs_k"))
      (Op.sub .nat Broadcast.nil (Op.constNat K)
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "kk") (Op.constNat BK))))

/-- The masked-load `mask=` operand for the `b` load (`offs_k[:,None] < K - kk·BK`,
remapped over the column axis). -/
def bMaskOp (K BK BN : Nat) : Op .bool [BK, BN] :=
  Op.remap [BK, BN] Broadcast.nil.consL.consSame.leftIndex
    (Op.lt ComparableDType.nat Broadcast.scalarR
      (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "offs_k"))
      (Op.sub .nat Broadcast.nil (Op.constNat K)
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "kk") (Op.constNat BK))))

/-- The 5-statement K-loop body, transcribed (masked loads + fused dot + advances). -/
def matmulLoopBody (K BM BN BK sak sbk : Nat) : List Stmt :=
  [ Stmt.assign .real [BM, BK] "a"
      (Op.load .real (.ptr (Op.ref .ptr [BM, BK] "a_ptrs"))
        (.maskOther (aMaskOp K BM BK) ((Op.const 0.0).broadcast [BM, BK]))),
    Stmt.assign .real [BK, BN] "b"
      (Op.load .real (.ptr (Op.ref .ptr [BK, BN] "b_ptrs"))
        (.maskOther (bMaskOp K BK BN) ((Op.const 0.0).broadcast [BK, BN]))),
    Stmt.assign .real [BM, BN] "accumulator"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.dot (batch := []) (Op.ref .real [BM, BK] "a") (Op.ref .real [BK, BN] "b"))
        (Op.ref .real [BM, BN] "accumulator")),
    Stmt.assign .ptr [BM, BK] "a_ptrs"
      (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BM, BK] "a_ptrs")
        (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat sak))),
    Stmt.assign .ptr [BK, BN] "b_ptrs"
      (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BK, BN] "b_ptrs")
        (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat sbk))) ]

/-- The activation tail (`if ACTIVATION { accumulator = leaky_relu(accumulator) }`). -/
def matmulActStmt (BM BN : Nat) (ACTIVATION : Bool) : Stmt :=
  Stmt.ifThen (Op.constBool ACTIVATION)
    [ Stmt.assign .real [BM, BN] "accumulator"
        (Op.where
          (Op.ge ComparableDType.real Broadcast.scalarR (Op.ref .real [BM, BN] "accumulator")
            (Op.const 0))
          (Op.ref .real [BM, BN] "accumulator")
          (Op.mul .real Broadcast.scalarL (Op.const (1e-2 : ℝ))
            (Op.ref .real [BM, BN] "accumulator"))) ]

/-- The 6-statement post-loop tail: fp16 cast, the two `offs_c*` vectors, the
`c_ptrs` chunk, the `c_mask`, and the masked store. -/
def matmulStoreTail (C : RegionName) (M N scm scn BM BN : Nat) : List Stmt :=
  [ Stmt.assign .fp16 [BM, BN] "c"
      (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref .real [BM, BN] "accumulator")),
    Stmt.assign .nat [BM] "offs_cm"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat BM))
        (Op.arange BM)),
    Stmt.assign .nat [BN] "offs_cn"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_n") (Op.constNat BN))
        (Op.arange BN)),
    Stmt.assign .ptr [BM, BN] "c_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase C)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarL (Op.constNat scm) (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_cm")))
          (Op.mul .nat Broadcast.scalarL (Op.constNat scn) (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_cn"))))),
    Stmt.assign .bool [BM, BN] "c_mask"
      (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_cm")) (Op.constNat M))
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_cn")) (Op.constNat N))),
    Stmt.store .fp16 [BM, BN] (.ptr (Op.ref .ptr [BM, BN] "c_ptrs"))
      (Op.ref .fp16 [BM, BN] "c") (.mask (Op.ref .bool [BM, BN] "c_mask")) ]

/-- Body decomposition: prefix (15) ++ [for-loop, ifThen-activation] ++ store-tail (6).
By `rfl`. -/
theorem matmul_body_split
    (A B C : RegionName)
    (M N K sam sak sbk sbn scm scn BM BN BK GM numKBlocks : Nat) (ACTIVATION : Bool) :
    (matmul_autotune_surface A B C M N K sam sak sbk sbn scm scn BM BN BK GM
        numKBlocks ACTIVATION).toAlgKernel.body
      = (matmul_autotune_surface A B C M N K sam sak sbk sbn scm scn BM BN BK GM
          numKBlocks ACTIVATION).toAlgKernel.body.take 15
        ++ (Stmt.forRange "kk" 0 numKBlocks 1 (matmulLoopBody K BM BN BK sak sbk)
            :: matmulActStmt BM BN ACTIVATION
            :: matmulStoreTail C M N scm scn BM BN) := by
  rfl

/-! ## Loop invariant -/

/-- **Loop invariant** (counter `c = block index`, step `1`).

After `c` K-blocks: program ids and `mem`/`undef` fixed; the `pid_m`/`pid_n`/
`offs_am`/`offs_bn`/`offs_k` registers seeded (the latter from the L2 schedule);
`accumulator` equals the partial GEMM accumulator `accPartial … c`; and the
`a_ptrs`/`b_ptrs` advanced by `c` blocks. -/
noncomputable def matmulInvariant
    (A B : RegionName) (s0 : BlockState) (M N BM BN GM sam sak sbk sbn BLOCK_K numKBlocks : Nat)
    (c : Nat) (s : BlockState) : Prop :=
  s.pids = s0.pids ∧ c ≤ numKBlocks ∧
  (s.regs .nat [] "pid_m" = some (Tile.scalar (pidM (s0.pids 0) M N BM BN GM))) ∧
  (s.regs .nat [] "pid_n" = some (Tile.scalar (pidN (s0.pids 0) M N BM BN GM))) ∧
  (s.regs .real [BM, BN] "accumulator" = some ⟨fun idx : TileIndex [BM, BN] =>
      some (accPartial s0 A B M N BM BN GM sam sak sbk sbn BLOCK_K idx.1 idx.2.1 c)⟩) ∧
  (s.regs .nat [BM] "offs_am" = some (Tile.vec (fun r : Fin BM => rowIndex s0 M N BM BN GM r))) ∧
  (s.regs .nat [BN] "offs_bn" = some (Tile.vec (fun j : Fin BN => colIndex s0 M N BM BN GM j))) ∧
  (s.regs .nat [BLOCK_K] "offs_k" = some (Tile.vec (fun e : Fin BLOCK_K => e.val))) ∧
  (s.regs .ptr [BM, BLOCK_K] "a_ptrs" = some ⟨fun idx : TileIndex [BM, BLOCK_K] =>
      (A.cast, rowIndex s0 M N BM BN GM idx.1 * sam + idx.2.1.val * sak + c * BLOCK_K * sak)⟩) ∧
  (s.regs .ptr [BLOCK_K, BN] "b_ptrs" = some ⟨fun idx : TileIndex [BLOCK_K, BN] =>
      (B.cast, idx.1.val * sbk + colIndex s0 M N BM BN GM idx.2.1 * sbn + c * BLOCK_K * sbk)⟩) ∧
  (∀ rg o, s.undef rg o = 0) ∧ (s.mem = s0.mem)

/-- **preLoop scalars** (statements 0–11): the 9 L2-schedule scalars and the 3
index vectors `offs_am`/`offs_bn`/`offs_k`. -/
theorem preLoop_scalars (s : BlockState) (M N BM BN BK GM : Nat) :
    ∃ s12, stepStmts
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
          ((Op.lt ComparableDType.nat Broadcast.nil
              (Op.sub .nat Broadcast.nil (Op.ref .nat [] "num_pid_m") (Op.ref .nat [] "first_pid_m"))
              (Op.constNat GM)).where
            (Op.sub .nat Broadcast.nil (Op.ref .nat [] "num_pid_m") (Op.ref .nat [] "first_pid_m"))
            (Op.constNat GM)),
        Stmt.assign .nat [] "pid_m"
          (Op.add .nat Broadcast.nil (Op.ref .nat [] "first_pid_m")
            (Op.mod .nat Broadcast.nil
              (Op.mod .nat Broadcast.nil (Op.ref .nat [] "pid") (Op.ref .nat [] "num_pid_in_group"))
              (Op.ref .nat [] "group_size_m"))),
        Stmt.assign .nat [] "pid_n"
          (Op.floorDiv .nat Broadcast.nil
            (Op.mod .nat Broadcast.nil (Op.ref .nat [] "pid") (Op.ref .nat [] "num_pid_in_group"))
            (Op.ref .nat [] "group_size_m")),
        Stmt.assign .nat [BM] "offs_am"
          (Op.mod .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat BM))
              (Op.arange BM))
            (Op.constNat M)),
        Stmt.assign .nat [BN] "offs_bn"
          (Op.mod .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_n") (Op.constNat BN))
              (Op.arange BN))
            (Op.constNat N)),
        Stmt.assign .nat [BK] "offs_k" (Op.arange BK) ] s = some s12
      ∧ s12.pids = s.pids
      ∧ s12.regs .nat [] "pid_m" = some (Tile.scalar (pidM (s.pids 0) M N BM BN GM))
      ∧ s12.regs .nat [] "pid_n" = some (Tile.scalar (pidN (s.pids 0) M N BM BN GM))
      ∧ s12.regs .nat [BM] "offs_am" = some (Tile.vec (fun i : Fin BM => rowIndex s M N BM BN GM i))
      ∧ s12.regs .nat [BN] "offs_bn" = some (Tile.vec (fun j : Fin BN => colIndex s M N BM BN GM j))
      ∧ s12.regs .nat [BK] "offs_k" = some (Tile.vec (fun e : Fin BK => e.val))
      ∧ s12.undef = s.undef
      ∧ s12.mem = s.mem := by
  simp only [pidM, pidN, rowIndex, colIndex, rowGlobal, colGlobal, cdiv, kernelMin]
  simp [stepStmts, stepStmt, evalOp_floorDiv, evalOp_mod, Option.bind, BlockState.setReg,
    Tile.bop, Tile.vec, NumericDType.add, NumericDType.mul, NumericDType.div, NumericDType.sub,
    IntegralDType.floorDiv, IntegralDType.mod, Tile.select, Tile.cop, ComparableDType.lt]

set_option maxHeartbeats 1000000 in
/-- **preLoop** (statements 0–14): from a clean input state (`undef = 0`), the
prologue steps to a state satisfying `matmulInvariant … 0` — the base case
(`accumulator = 0`, pointers seeded, schedule derived). -/
theorem preLoop (A B C : RegionName) (s : BlockState)
    (M N BM BN BK sam sak sbk sbn scm scn GM numKBlocks : Nat) (K : Nat) (ACTIVATION : Bool)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ s', stepStmts ((matmul_autotune_surface A B C M N K sam sak sbk sbn
        scm scn BM BN BK GM numKBlocks ACTIVATION).toAlgKernel.body.take 15) s = some s'
      ∧ matmulInvariant A B s M N BM BN GM sam sak sbk sbn BK numKBlocks 0 s' := by
  obtain ⟨s12, h12, hpids, hpm, hpn, hm, hn, hk, huf, hmem⟩ := preLoop_scalars s M N BM BN BK GM
  rw [show ((matmul_autotune_surface A B C M N K sam sak sbk sbn
        scm scn BM BN BK GM numKBlocks ACTIVATION).toAlgKernel.body.take 15)
      = [ Stmt.assign .nat [] "pid" (Op.programId 0),
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
            ((Op.lt ComparableDType.nat Broadcast.nil
                (Op.sub .nat Broadcast.nil (Op.ref .nat [] "num_pid_m") (Op.ref .nat [] "first_pid_m"))
                (Op.constNat GM)).where
              (Op.sub .nat Broadcast.nil (Op.ref .nat [] "num_pid_m") (Op.ref .nat [] "first_pid_m"))
              (Op.constNat GM)),
          Stmt.assign .nat [] "pid_m"
            (Op.add .nat Broadcast.nil (Op.ref .nat [] "first_pid_m")
              (Op.mod .nat Broadcast.nil
                (Op.mod .nat Broadcast.nil (Op.ref .nat [] "pid") (Op.ref .nat [] "num_pid_in_group"))
                (Op.ref .nat [] "group_size_m"))),
          Stmt.assign .nat [] "pid_n"
            (Op.floorDiv .nat Broadcast.nil
              (Op.mod .nat Broadcast.nil (Op.ref .nat [] "pid") (Op.ref .nat [] "num_pid_in_group"))
              (Op.ref .nat [] "group_size_m")),
          Stmt.assign .nat [BM] "offs_am"
            (Op.mod .nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat BM))
                (Op.arange BM))
              (Op.constNat M)),
          Stmt.assign .nat [BN] "offs_bn"
            (Op.mod .nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_n") (Op.constNat BN))
                (Op.arange BN))
              (Op.constNat N)),
          Stmt.assign .nat [BK] "offs_k" (Op.arange BK) ]
      ++ [ Stmt.assign .ptr [BM, BK] "a_ptrs"
            (Op.ptrAdd Broadcast.scalarL (Op.ptrBase A)
              (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
                (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_am")) (Op.constNat sam))
                (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BK] "offs_k")) (Op.constNat sak)))),
          Stmt.assign .ptr [BK, BN] "b_ptrs"
            (Op.ptrAdd Broadcast.scalarL (Op.ptrBase B)
              (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
                (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "offs_k")) (Op.constNat sbk))
                (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_bn")) (Op.constNat sbn)))),
          Stmt.assign .real [BM, BN] "accumulator" (Op.full [BM, BN] (Op.const 0)) ] from rfl,
    stepStmts.append_some h12,
    stepStmts.cons_some (stepStmt_assign_eq_some
      (aptrs_eval s12 A BM BK sam sak (fun i => rowIndex s M N BM BN GM i) (by simpa using hm) (by simpa using hk))),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (bptrs_eval _ B BK BN sbk sbn (fun j => colIndex s M N BM BN GM j) (by simp [hk]) (by simp [hn]))),
    stepStmts.cons_some (stepStmt_assign_eq_some (acc_init_eval _ BM BN)),
    stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  refine ⟨by simp [hpids], by simp, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp [hpm]
  · simp [hpn]
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx
    simp only [accPartial, Nat.zero_mul, gemmSum_zero]
  · simp [hm]
  · simp [hn]
  · simp [hk]
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx <;> simp [Nat.zero_mul]
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx <;> simp [Nat.zero_mul]
  · intro rg o; simp [huf, hundef]
  · exact hmem

/-- The `a` load mask tile evaluates to **all-true** when the K-block index `c`
satisfies `c < numKBlocks` and `K = BK · numKBlocks` (so `offs_k[e] < K - c·BK`
holds for every K-lane). -/
theorem aMaskOp_eval (st : BlockState) (K BM BK numKBlocks c : Nat)
    (hK : K = BK * numKBlocks) (hc : c < numKBlocks)
    (hk : st.regs .nat [BK] "offs_k" = some (Tile.vec (fun e : Fin BK => e.val)))
    (hkk : st.regs .nat [] "kk" = some (Tile.scalar c)) :
    ∃ masks : Tile .bool [BM, BK], evalOp (aMaskOp K BM BK) st = some masks
      ∧ ∀ i, masks.data i = Bool.true := by
  have heval : evalOp (aMaskOp K BM BK) st = some
      (Tile.remap (dtype := .bool) Broadcast.nil.consSame.consL.leftIndex
        (Tile.cop ComparableDType.nat.lt Broadcast.scalarR
          (⟨fun i : TileIndex [1, BK] => (Tile.vec (fun e : Fin BK => (e.val : Nat))).data (i.2.1, PUnit.unit)⟩
            : Tile .nat [1, BK])
          (Tile.bop NumericDType.nat.sub Broadcast.nil (Tile.scalar K)
            (Tile.bop NumericDType.nat.mul Broadcast.nil (Tile.scalar c) (Tile.scalar BK))))) := by
    unfold aMaskOp
    rw [evalOp_remap]
    conv_lhs => arg 1; rw [evalOp_lt]; arg 1; rw [evalOp_expandDim_zero_nat, hk]
    simp only [Option.bind_eq_bind, Option.bind_some, evalOp_sub, evalOp_mul, evalOp_constNat,
      evalOp_ref, hkk]
  refine ⟨_, heval, ?_⟩
  intro i
  simp only [Tile.remap, Tile.cop_data, Tile.vec, Tile.scalar, Tile.bop_data,
    Broadcast.leftIndex, Broadcast.rightIndex, ComparableDType.lt, NumericDType.sub, NumericDType.mul,
    decide_eq_true_eq]
  have : (i.2.1.val : Nat) < K - c * BK := by
    have hlt : (i.2.1.val : Nat) < BK := i.2.1.isLt
    have : c * BK + BK ≤ K := by
      rw [hK]; calc c * BK + BK = (c + 1) * BK := by ring
        _ ≤ numKBlocks * BK := Nat.mul_le_mul_right _ hc
        _ = BK * numKBlocks := Nat.mul_comm _ _
    omega
  simpa using this

/-- The `b` load mask tile evaluates to all-true under the same condition. -/
theorem bMaskOp_eval (st : BlockState) (K BK BN numKBlocks c : Nat)
    (hK : K = BK * numKBlocks) (hc : c < numKBlocks)
    (hk : st.regs .nat [BK] "offs_k" = some (Tile.vec (fun e : Fin BK => e.val)))
    (hkk : st.regs .nat [] "kk" = some (Tile.scalar c)) :
    ∃ masks : Tile .bool [BK, BN], evalOp (bMaskOp K BK BN) st = some masks
      ∧ ∀ i, masks.data i = Bool.true := by
  have heval : evalOp (bMaskOp K BK BN) st = some
      (Tile.remap (dtype := .bool) Broadcast.nil.consL.consSame.leftIndex
        (Tile.cop ComparableDType.nat.lt Broadcast.scalarR
          (⟨fun i : TileIndex [BK, 1] => (Tile.vec (fun e : Fin BK => (e.val : Nat))).data (i.1, PUnit.unit)⟩
            : Tile .nat [BK, 1])
          (Tile.bop NumericDType.nat.sub Broadcast.nil (Tile.scalar K)
            (Tile.bop NumericDType.nat.mul Broadcast.nil (Tile.scalar c) (Tile.scalar BK))))) := by
    unfold bMaskOp
    rw [evalOp_remap]
    conv_lhs => arg 1; rw [evalOp_lt]; arg 1; rw [evalOp_expandDim_one_nat, hk]
    simp only [Option.bind_eq_bind, Option.bind_some, evalOp_sub, evalOp_mul, evalOp_constNat,
      evalOp_ref, hkk]
  refine ⟨_, heval, ?_⟩
  intro i
  simp only [Tile.remap, Tile.cop_data, Tile.vec, Tile.scalar, Tile.bop_data,
    Broadcast.leftIndex, Broadcast.rightIndex, ComparableDType.lt, NumericDType.sub, NumericDType.mul,
    decide_eq_true_eq]
  have : (i.1.val : Nat) < K - c * BK := by
    have hlt : (i.1.val : Nat) < BK := i.1.isLt
    have : c * BK + BK ≤ K := by
      rw [hK]; calc c * BK + BK = (c + 1) * BK := by ring
        _ ≤ numKBlocks * BK := Nat.mul_le_mul_right _ hc
        _ = BK * numKBlocks := Nat.mul_comm _ _
    omega
  simpa using this

/-- `(Op.const 0.0).broadcast [M,N]` eval succeeds (the `other` operand; its value
is irrelevant since the load mask is uniformly true). -/
theorem const_broadcast_eval (st : BlockState) (M N : Nat) :
    ∃ t : Tile .real [M, N], evalOp ((Op.const (0.0 : ℝ)).broadcast [M, N]) st = some t := by
  simp only [evalOp, evalOp_const, Option.bind_eq_bind, Option.bind_some]
  exact ⟨_, rfl⟩

/-- The masked `a` load, keyed on register readbacks: at exact-multiple `K` and
block index `c < numKBlocks`, it reads the genuine `readMem` cells. -/
theorem load_a_eval (st : BlockState) (K BM BK numKBlocks c : Nat) (ap : Tile .ptr [BM, BK])
    (hK : K = BK * numKBlocks) (hc : c < numKBlocks)
    (hap : st.regs .ptr [BM, BK] "a_ptrs" = some ap)
    (hkof : st.regs .nat [BK] "offs_k" = some (Tile.vec (fun e : Fin BK => e.val)))
    (hkk : st.regs .nat [] "kk" = some (Tile.scalar c)) :
    evalOp (Op.load .real (.ptr (Op.ref .ptr [BM, BK] "a_ptrs"))
        (.maskOther (aMaskOp K BM BK) ((Op.const 0.0).broadcast [BM, BK]))) st
      = some ⟨fun i => some (st.readMem (ap.data i).1 (ap.data i).2)⟩ := by
  obtain ⟨amasks, hamasks, htrue⟩ := aMaskOp_eval st K BM BK numKBlocks c hK hc hkof hkk
  obtain ⟨aother, haother⟩ := const_broadcast_eval st BM BK
  exact load_ptr_maskOther_true_real (Op.ref .ptr [BM, BK] "a_ptrs") (aMaskOp K BM BK)
    ((Op.const 0.0).broadcast [BM, BK]) st ap amasks (by rw [evalOp_ref, hap]) hamasks
    aother haother htrue

/-- The masked `b` load, keyed on register readbacks. -/
theorem load_b_eval (st : BlockState) (K BK BN numKBlocks c : Nat) (bp : Tile .ptr [BK, BN])
    (hK : K = BK * numKBlocks) (hc : c < numKBlocks)
    (hbp : st.regs .ptr [BK, BN] "b_ptrs" = some bp)
    (hkof : st.regs .nat [BK] "offs_k" = some (Tile.vec (fun e : Fin BK => e.val)))
    (hkk : st.regs .nat [] "kk" = some (Tile.scalar c)) :
    evalOp (Op.load .real (.ptr (Op.ref .ptr [BK, BN] "b_ptrs"))
        (.maskOther (bMaskOp K BK BN) ((Op.const 0.0).broadcast [BK, BN]))) st
      = some ⟨fun i => some (st.readMem (bp.data i).1 (bp.data i).2)⟩ := by
  obtain ⟨bmasks, hbmasks, htrue⟩ := bMaskOp_eval st K BK BN numKBlocks c hK hc hkof hkk
  obtain ⟨bother, hbother⟩ := const_broadcast_eval st BK BN
  exact load_ptr_maskOther_true_real (Op.ref .ptr [BK, BN] "b_ptrs") (bMaskOp K BK BN)
    ((Op.const 0.0).broadcast [BK, BN]) st bp bmasks (by rw [evalOp_ref, hbp]) hbmasks
    bother hbother htrue

set_option maxHeartbeats 2000000 in
/-- **Step lemma**: one K-loop body iteration advances the invariant by one block.
The masked loads (uniformly-true masks at exact-multiple `K`) read the genuine
`A`/`B` cells, the fused `tl.dot(a, b, accumulator)` adds the `c`-th block's dot to
the partial GEMM accumulator, and the `a`/`b` pointers advance one step. -/
theorem matmul_step (A B : RegionName) (s0 : BlockState)
    (M N BM BN GM sam sak sbk sbn BLOCK_K numKBlocks : Nat) (K : Nat) (hK : K = BLOCK_K * numKBlocks)
    (c : Nat) (s : BlockState) (hclt : c < numKBlocks)
    (hinv : matmulInvariant A B s0 M N BM BN GM sam sak sbk sbn BLOCK_K numKBlocks c s) :
    ∃ s', stepStmts (matmulLoopBody K BM BN BLOCK_K sak sbk)
        (s.setReg "kk" .nat [] (Tile.scalar c)) = some s'
      ∧ matmulInvariant A B s0 M N BM BN GM sam sak sbk sbn BLOCK_K numKBlocks (c + 1) s' := by
  simp only [matmulInvariant] at hinv
  obtain ⟨hpids, hcle, hpm, hpn, hz, hm, hn, hk, hap, hbp, hundef, hmem⟩ := hinv
  set apT : Tile .ptr [BM, BLOCK_K] :=
    ⟨fun idx : TileIndex [BM, BLOCK_K] =>
      (A.cast, rowIndex s0 M N BM BN GM idx.1 * sam + idx.2.1.val * sak + c * BLOCK_K * sak)⟩ with hapT
  set bpT : Tile .ptr [BLOCK_K, BN] :=
    ⟨fun idx : TileIndex [BLOCK_K, BN] =>
      (B.cast, idx.1.val * sbk + colIndex s0 M N BM BN GM idx.2.1 * sbn + c * BLOCK_K * sbk)⟩ with hbpT
  set zT : Tile .real [BM, BN] :=
    ⟨fun idx : TileIndex [BM, BN] =>
      some (accPartial s0 A B M N BM BN GM sam sak sbk sbn BLOCK_K idx.1 idx.2.1 c)⟩ with hzT
  set sk := s.setReg "kk" .nat [] (Tile.scalar c) with hsk
  have hrmem : ∀ (R : RegionName) (o : Nat), sk.readMem R o = s0.readMem R o := by
    intro R o; simp only [hsk, BlockState.setReg_readMem]; unfold BlockState.readMem; rw [hmem]
  have hapk : sk.regs .ptr [BM, BLOCK_K] "a_ptrs" = some apT := by simp [hsk, hap, hapT]
  have hbpk : sk.regs .ptr [BLOCK_K, BN] "b_ptrs" = some bpT := by simp [hsk, hbp, hbpT]
  have hzk : sk.regs .real [BM, BN] "accumulator" = some zT := by simp [hsk, hz, hzT]
  have hkk : sk.regs .nat [] "kk" = some (Tile.scalar c) := by simp [hsk]
  have hkofk : sk.regs .nat [BLOCK_K] "offs_k" = some (Tile.vec (fun e : Fin BLOCK_K => e.val)) := by
    simp [hsk, hk]
  set asub : Tile .real [BM, BLOCK_K] :=
    ⟨fun idx => some (sk.readMem (apT.data idx).1 (apT.data idx).2)⟩ with hasub
  set bsub : Tile .real [BLOCK_K, BN] :=
    ⟨fun idx => some (sk.readMem (bpT.data idx).1 (bpT.data idx).2)⟩ with hbsub
  unfold matmulLoopBody
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (load_a_eval sk K BM BLOCK_K numKBlocks c apT hK hclt hapk hkofk hkk))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (load_b_eval _ K BLOCK_K BN numKBlocks c bpT hK hclt
          (by simp [hbpk]) (by simp [hkofk]) (by simp [hkk])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (accdot_op_eval BM BLOCK_K BN _ zT asub bsub
          (by simp [hzk, hasub, hbsub, BlockState.setReg_readMem])
          (by simp [hasub, BlockState.setReg_readMem])
          (by simp [hbsub, BlockState.setReg_readMem])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (aptr_adv_eval _ BM BLOCK_K BLOCK_K sak apT (by simp [hapk])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (bptr_adv_eval _ BLOCK_K BN BLOCK_K sbk bpT (by simp [hbpk])))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  simp only [matmulInvariant]
  refine ⟨by simp [hsk, hpids], by omega, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp [hsk, hpm]
  · simp [hsk, hpn]
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx
    have has : ∀ e : Fin BLOCK_K, asub.data (idx.1, e, PUnit.unit)
        = some (aElem s0 A M N BM BN GM sam sak idx.1 (c * BLOCK_K + e.val)) := by
      intro e
      simp only [hasub, hapT, hrmem, aElem, Region.cast_id]
      rw [show rowIndex s0 M N BM BN GM idx.1 * sam + e.val * sak + c * BLOCK_K * sak
            = rowIndex s0 M N BM BN GM idx.1 * sam + (c * BLOCK_K + e.val) * sak from by ring]
    have hbs : ∀ e : Fin BLOCK_K, bsub.data (e, idx.2.1, PUnit.unit)
        = some (bElem s0 B M N BM BN GM sbk sbn idx.2.1 (c * BLOCK_K + e.val)) := by
      intro e
      simp only [hbsub, hbpT, hrmem, bElem, Region.cast_id]
      rw [show e.val * sbk + colIndex s0 M N BM BN GM idx.2.1 * sbn + c * BLOCK_K * sbk
            = (c * BLOCK_K + e.val) * sbk + colIndex s0 M N BM BN GM idx.2.1 * sbn from by ring]
    rw [dotadd_eval BM BN (Tile.dot [] asub bsub) zT idx.1 idx.2.1
        (Finset.univ.sum fun e : Fin BLOCK_K =>
          aElem s0 A M N BM BN GM sam sak idx.1 (c * BLOCK_K + e.val)
            * bElem s0 B M N BM BN GM sbk sbn idx.2.1 (c * BLOCK_K + e.val))
        (accPartial s0 A B M N BM BN GM sam sak sbk sbn BLOCK_K idx.1 idx.2.1 c)
        (tile_dot_data BM BLOCK_K BN asub bsub idx.1 idx.2.1 _ _ has hbs)
        (by rw [hzT])]
    show some _ = some (accPartial s0 A B M N BM BN GM sam sak sbk sbn BLOCK_K idx.1 idx.2.1 (c + 1))
    rw [accPartial_succ, add_comm]
  · simp [hsk, hm]
  · simp [hsk, hn]
  · simp [hsk, hk]
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
  · intro rg o; simp [hsk, hundef]
  · show _ = s0.mem
    rw [← hmem, hsk]; rfl

/-! ## Post-loop: activation tail + fp16 cast + masked store -/

/-- The output store address for tile lane `(i,j)`: `scm · offs_cm i + scn · offs_cn j`
(the kernel's `c_ptrs`, using the **un-wrapped** global `offs_cm`/`offs_cn`). -/
def cOffset (s0 : BlockState) (M N BM BN GM scm scn : Nat) (idx : TileIndex [BM, BN]) : Nat :=
  scm * rowGlobal s0 M N BM BN GM idx.1 + scn * colGlobal s0 M N BM BN GM idx.2.1

/-- The boundary predicate `(row < M) & (col < N)` for tile lane `(i,j)`. -/
def active (s0 : BlockState) (M N BM BN GM : Nat) (idx : TileIndex [BM, BN]) : Prop :=
  rowGlobal s0 M N BM BN GM idx.1 < M ∧ colGlobal s0 M N BM BN GM idx.2.1 < N

instance activeDecidable (s0 : BlockState) (M N BM BN GM : Nat)
    (idx : TileIndex [BM, BN]) : Decidable (active s0 M N BM BN GM idx) := by
  unfold active; infer_instance

/-- `offs_cm` / `offs_cn` eval (the **un-wrapped** global index, no `% M`). -/
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

theorem evalOp_boolAnd {a b shape} (bc : Broadcast a b shape)
    (x : Op .bool a) (y : Op .bool b) (s : BlockState) :
    evalOp (.boolAnd bc x y) s = (do
      let vx ← evalOp x s; let vy ← evalOp y s; some (Tile.bop (· && ·) bc vx vy)) := by
  simp [evalOp]

theorem evalOp_ge {dtype a b shape} (h : ComparableDType dtype) (bc : Broadcast a b shape)
    (x : Op dtype a) (y : Op dtype b) (s : BlockState) :
    evalOp (.ge h bc x y) s = (do
      let vx ← evalOp x s; let vy ← evalOp y s; some (Tile.cop h.ge bc vx vy)) := by
  simp [evalOp]

@[simp] theorem evalOp_constBool (b : Bool) (s : BlockState) :
    evalOp (.constBool b) s = some (Tile.scalar b) := by simp [evalOp]

/-- `c_mask` eval: the `(offs_cm < M) & (offs_cn < N)` boolean tile. -/
theorem cmask_eval (s : BlockState) (M N BM BN : Nat) (gm : Fin BM → Nat) (gn : Fin BN → Nat)
    (hm : s.regs .nat [BM] "offs_cm" = some (Tile.vec gm))
    (hn : s.regs .nat [BN] "offs_cn" = some (Tile.vec gn)) :
    evalOp (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_cm")) (Op.constNat M))
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_cn")) (Op.constNat N))) s
      = some ⟨fun idx : TileIndex [BM, BN] => (decide (gm idx.1 < M) && decide (gn idx.2.1 < N))⟩ := by
  rw [evalOp_boolAnd, evalOp_lt, evalOp_lt, evalOp_expandDim_one_nat, evalOp_expandDim_zero_nat]
  simp only [hm, hn, evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp [Tile.bop_data, Tile.cop_data, Tile.expandDim, Tile.vec, Tile.scalar_data, ComparableDType.lt,
    Broadcast.leftIndex, Broadcast.rightIndex]

/-- The activation tail `if ACTIVATION { accumulator = leaky_relu(accumulator) }`
applied to a state whose `accumulator` register holds the tile of `accSum` values
produces a state whose `accumulator` register holds the `act ACTIVATION`-mapped
values (and leaves the rest of the relevant registers/`mem` untouched). -/
theorem matmulActStmt_eval (st : BlockState) (BM BN : Nat) (ACTIVATION : Bool)
    (accSum : TileIndex [BM, BN] → ℝ)
    (hacc : st.regs .real [BM, BN] "accumulator"
      = some ⟨fun idx => some (accSum idx)⟩) :
    ∃ st', stepStmt (matmulActStmt BM BN ACTIVATION) st = some st'
      ∧ st'.regs .real [BM, BN] "accumulator"
          = some ⟨fun idx => some (act ACTIVATION (accSum idx))⟩
      ∧ (∀ {dt} {sh} (nm : RegName), nm ≠ "accumulator" → st'.regs dt sh nm = st.regs dt sh nm)
      ∧ st'.mem = st.mem ∧ st'.pids = st.pids
      ∧ (∀ rg o, st'.undef rg o = st.undef rg o) := by
  unfold matmulActStmt
  have hactEval : evalOp (Op.where
        (Op.ge ComparableDType.real Broadcast.scalarR (Op.ref .real [BM, BN] "accumulator") (Op.const 0))
        (Op.ref .real [BM, BN] "accumulator")
        (Op.mul .real Broadcast.scalarL (Op.const (1e-2 : ℝ)) (Op.ref .real [BM, BN] "accumulator"))) st
      = some ⟨fun idx : TileIndex [BM, BN] => some (leakyReLU (accSum idx))⟩ := by
    rw [evalOp_where, evalOp_ge, evalOp_mul]
    simp only [evalOp_ref, evalOp_const, hacc, Option.bind_eq_bind, Option.bind_some]
    refine congrArg some ?_
    ext idx
    simp only [Tile.select_data, Tile.cop_data, Tile.bop_data, Tile.scalar_data,
      Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.mul, leakyReLU]
    by_cases h : ComparableDType.real.ge (some (accSum idx)) (some 0) = Bool.true
    · have hge : accSum idx ≥ 0 := by
        have hh := (ComparableDType.real_ge_eq_true (some (accSum idx)) (some 0)).mp h
        simp only [ge_iff_le] at hh ⊢
        exact WithBot.coe_le_coe.mp hh
      simp only [h, if_true, if_pos hge]
    · have hge : ¬ accSum idx ≥ 0 := by
        intro hc
        apply h
        apply (ComparableDType.real_ge_eq_true (some (accSum idx)) (some 0)).mpr
        simp only [ge_iff_le]
        exact WithBot.coe_le_coe.mpr hc
      have hf : ComparableDType.real.ge (some (accSum idx)) (some 0) = Bool.false := by
        cases hb : ComparableDType.real.ge (some (accSum idx)) (some 0)
        · rfl
        · exact absurd hb h
      simp only [hf, Bool.false_eq_true, if_false, if_neg hge]
      simp only [WithBot.realMul, Option.map₂, Option.bind, Option.map]
  rcases ACTIVATION with _ | _
  · refine ⟨st, ?_, ?_, fun nm _ => rfl, rfl, rfl, fun _ _ => rfl⟩
    · show stepStmt (Stmt.ifThen (Op.constBool Bool.false) _) st = some st
      simp only [stepStmt, evalOp_constBool]
      rfl
    · simpa [act] using hacc
  · set st' := st.setReg "accumulator" .real [BM, BN]
      (⟨fun idx : TileIndex [BM, BN] => some (leakyReLU (accSum idx))⟩ : Tile .real [BM, BN]) with hst'
    have hstep : stepStmt (Stmt.ifThen (Op.constBool Bool.true)
        [Stmt.assign .real [BM, BN] "accumulator"
          (Op.where
            (Op.ge ComparableDType.real Broadcast.scalarR (Op.ref .real [BM, BN] "accumulator") (Op.const 0))
            (Op.ref .real [BM, BN] "accumulator")
            (Op.mul .real Broadcast.scalarL (Op.const (1e-2 : ℝ)) (Op.ref .real [BM, BN] "accumulator")))]) st
        = some st' := by
      simp only [stepStmt, evalOp_constBool, Option.bind_eq_bind, Option.bind_some,
        Tile.scalar_data, if_true, stepStmts, hactEval, hst']
    refine ⟨st', hstep, ?_, ?_, ?_, ?_, ?_⟩
    · rw [hst']
      simp [BlockState.setReg_same, act]
    · intro dt sh nm hnm
      rw [hst']; simp [BlockState.setReg_ne_name, hnm]
    · rw [hst']; rfl
    · rw [hst']; rfl
    · intro rg o; rw [hst']; rfl

set_option maxHeartbeats 4000000 in
/-- **postLoop**: from the invariant at `numKBlocks` blocks, the activation tail +
cast-to-fp16 + masked store writes the genuine closed form
`fp16( act( Σ_k A·B ) )` at every active output lane (given the output-offset map
is injective). -/
theorem matmul_postLoop (A B C : RegionName) (s0 : BlockState)
    (M N BM BN GM sam sak sbk sbn scm scn BLOCK_K numKBlocks : Nat) (ACTIVATION : Bool)
    (hInj : Function.Injective (cOffset s0 M N BM BN GM scm scn))
    (st : BlockState)
    (hinv : matmulInvariant A B s0 M N BM BN GM sam sak sbk sbn BLOCK_K numKBlocks numKBlocks st) :
    ∃ sfin, stepStmts (matmulActStmt BM BN ACTIVATION :: matmulStoreTail C M N scm scn BM BN) st
        = some sfin
      ∧ ∀ idx : TileIndex [BM, BN],
          sfin.mem C (cOffset s0 M N BM BN GM scm scn idx)
            = if active s0 M N BM BN GM idx then
                MemCell.of .fp16
                  (FloatDType.real.cast FloatDType.fp16
                    (some (matmulSpec s0 A B M N BM BN GM sam sak sbk sbn BLOCK_K numKBlocks ACTIVATION idx.1 idx.2.1)))
              else
                st.mem C (cOffset s0 M N BM BN GM scm scn idx) := by
  simp only [matmulInvariant] at hinv
  obtain ⟨hpids, hcle, hpm, hpn, hz, hm, hn, hk, hap, hbp, hundef, hmem⟩ := hinv
  -- run the activation tail
  obtain ⟨sa, hsa, hsa_acc, hsa_other, hsa_mem, hsa_pids, hsa_undef⟩ :=
    matmulActStmt_eval st BM BN ACTIVATION
      (fun idx => accPartial s0 A B M N BM BN GM sam sak sbk sbn BLOCK_K idx.1 idx.2.1 numKBlocks)
      hz
  rw [stepStmts.cons_some hsa]
  -- registers preserved by the activation tail
  have hpm' : sa.regs .nat [] "pid_m" = some (Tile.scalar (pidM (s0.pids 0) M N BM BN GM)) := by
    rw [hsa_other "pid_m" (by decide)]; exact hpm
  have hpn' : sa.regs .nat [] "pid_n" = some (Tile.scalar (pidN (s0.pids 0) M N BM BN GM)) := by
    rw [hsa_other "pid_n" (by decide)]; exact hpn
  -- the activated accumulator tile
  set accT : Tile .real [BM, BN] :=
    ⟨fun idx : TileIndex [BM, BN] =>
      some (act ACTIVATION (accPartial s0 A B M N BM BN GM sam sak sbk sbn BLOCK_K idx.1 idx.2.1 numKBlocks))⟩
      with haccT
  set cT : Tile .fp16 [BM, BN] :=
    ⟨fun idx => FloatDType.real.cast FloatDType.fp16 (accT.data idx)⟩ with hcT
  set cpT : Tile .ptr [BM, BN] :=
    ⟨fun idx : TileIndex [BM, BN] => (C.cast, cOffset s0 M N BM BN GM scm scn idx)⟩ with hcpT
  set cmaskT : Tile .bool [BM, BN] :=
    ⟨fun idx : TileIndex [BM, BN] =>
      (decide (rowGlobal s0 M N BM BN GM idx.1 < M) && decide (colGlobal s0 M N BM BN GM idx.2.1 < N))⟩
      with hcmaskT
  unfold matmulStoreTail
  -- c = cast(accumulator, fp16)
  erw [stepStmts.cons_some (stepStmt_assign_eq_some
        (show evalOp (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref .real [BM, BN] "accumulator")) sa
          = some cT from by rw [evalOp_castFloat]; simp [evalOp_ref, hsa_acc, hcT, haccT]))]
  -- offs_cm
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (offscm_eval _ BM BM (pidM (s0.pids 0) M N BM BN GM) "pid_m" (by simp [hpm'])))]
  -- offs_cn
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (offscm_eval _ BN BN (pidN (s0.pids 0) M N BM BN GM) "pid_n" (by simp [hpn'])))]
  -- c_ptrs
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase C)
            (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
              (Op.mul .nat Broadcast.scalarL (Op.constNat scm) (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_cm")))
              (Op.mul .nat Broadcast.scalarL (Op.constNat scn) (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_cn"))))) _
          = some cpT from by
          rw [cptrs_eval _ C BM BN scm scn (fun i => rowGlobal s0 M N BM BN GM i) (fun j => colGlobal s0 M N BM BN GM j)
                (by simp [rowGlobal]) (by simp [colGlobal])]
          rfl))]
  -- c_mask
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (show evalOp (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
            (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_cm")) (Op.constNat M))
            (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_cn")) (Op.constNat N))) _
          = some cmaskT from by
          rw [cmask_eval _ M N BM BN (fun i => rowGlobal s0 M N BM BN GM i) (fun j => colGlobal s0 M N BM BN GM j)
                (by simp [rowGlobal]) (by simp [colGlobal])]))]
  -- abstract the post-assign state
  generalize hst4 : (((((sa.setReg "c" FloatDType.fp16.toTileDType [BM, BN] cT).setReg "offs_cm" .nat [BM]
        (Tile.vec fun i : Fin BM => pidM (s0.pids 0) M N BM BN GM * BM + i.val)).setReg "offs_cn" .nat [BN]
        (Tile.vec fun j : Fin BN => pidN (s0.pids 0) M N BM BN GM * BN + j.val)).setReg "c_ptrs" .ptr [BM, BN] cpT).setReg
        "c_mask" .bool [BM, BN] cmaskT) = st4
  have hc4 : st4.regs .fp16 [BM, BN] "c" = some cT := by rw [← hst4]; simp
  have hcp4 : st4.regs .ptr [BM, BN] "c_ptrs" = some cpT := by rw [← hst4]; simp
  have hcm4 : st4.regs .bool [BM, BN] "c_mask" = some cmaskT := by rw [← hst4]; simp
  have hmem4 : st4.mem = st.mem := by
    rw [← hst4]; funext region offset; simp only [BlockState.setReg_mem]; rw [hsa_mem]
  -- the masked store
  have hstore : stepStmt (Stmt.store .fp16 [BM, BN] (.ptr (Op.ref .ptr [BM, BN] "c_ptrs"))
        (Op.ref .fp16 [BM, BN] "c") (.mask (Op.ref .bool [BM, BN] "c_mask"))) st4
      = some ((TileShape.allIndices [BM, BN]).foldl
          (fun acc i =>
            if cmaskT.data i then
              acc.writeMemTyped .fp16 C (cOffset s0 M N BM BN GM scm scn i) (cT.data i)
            else acc) st4) := by
    simp only [stepStmt]
    rw [show evalOp (Op.ref .fp16 [BM, BN] "c") st4 = some cT from by rw [evalOp_ref, hc4]]
    rw [show evalOp (Op.ref .ptr [BM, BN] "c_ptrs") st4 = some cpT from by rw [evalOp_ref, hcp4]]
    rw [show evalOp (Op.ref .bool [BM, BN] "c_mask") st4 = some cmaskT from by rw [evalOp_ref, hcm4]]
    simp only [bind, Option.bind_some]
    refine congrArg some (List.foldl_ext _ _ _ (fun acc i _ => ?_))
    by_cases hmask : cmaskT.data i
    · simp only [hmask, if_true, cpT, hcpT, Region.cast_id]
    · simp only [hmask, Bool.false_eq_true, if_false]
  rw [stepStmts.cons_some hstore, stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  intro idx
  rw [scatter_memcell_fp16_prop_masked_nd (region := C) (s := st4)
        (offsetFn := cOffset s0 M N BM BN GM scm scn)
        (valueFn := fun i => cT.data i)
        (P := fun i => cmaskT.data i = Bool.true) hInj idx]
  by_cases hact : active s0 M N BM BN GM idx
  · have hmasktrue : cmaskT.data idx = Bool.true := by
      simp only [hcmaskT]
      obtain ⟨hr, hcc⟩ := hact
      simp [hr, hcc]
    simp only [hmasktrue, if_true, if_pos hact]
    -- collapse the fp16 round-trip and rewrite acc → matmulSpec
    simp only [hcT, haccT, matmulSpec, accPartial, Nat.mul_comm numKBlocks BLOCK_K,
      FloatDType.cast, FloatDType.ofReal, FloatDType.storeValue, FloatDType.real_toWithBot,
      FloatDType.fp16_ofWithBot, FloatDType.fp16_toWithBot, WithBot.unbotD_some]
  · have hmaskfalse : ¬ (cmaskT.data idx = Bool.true) := by
      simp only [hcmaskT, Bool.and_eq_true, decide_eq_true_eq, not_and]
      intro hr hcc
      exact hact ⟨hr, hcc⟩
    rw [if_neg hmaskfalse, if_neg hact, hmem4]

/-! ## Composition: full exec closed form -/

set_option maxHeartbeats 4000000 in
/-- **Top exec reduction**: composes `preLoop` + `matmul_step` (driven by
`forRange_inv`) + the activation tail + `matmul_postLoop` into the full `exec`
result. Every active output lane's memory cell equals the genuine closed form
`fp16( act( Σ_k A·B ) )`. -/
theorem matmul_autotune_exec_closed_form (A B C : RegionName) (s : BlockState)
    (M N BM BN GM sam sak sbk sbn scm scn BLOCK_K numKBlocks : Nat) (K : Nat)
    (hK : K = BLOCK_K * numKBlocks) (ACTIVATION : Bool)
    (hInj : Function.Injective (cOffset s M N BM BN GM scm scn))
    (hundef : ∀ rg o, s.undef rg o = 0)
    (idx : TileIndex [BM, BN]) :
    (match exec (matmul_autotune_surface A B C M N K sam sak sbk sbn scm scn
        BM BN BLOCK_K GM numKBlocks ACTIVATION) s with
      | some s' => s'.mem C (cOffset s M N BM BN GM scm scn idx)
      | none => (0 : MemCell)) =
      (if active s M N BM BN GM idx then
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (matmulSpec s A B M N BM BN GM sam sak sbk sbn BLOCK_K numKBlocks ACTIVATION idx.1 idx.2.1)))
      else
        s.mem C (cOffset s M N BM BN GM scm scn idx)) := by
  -- preLoop establishes P 0
  obtain ⟨s0, hpre_eq, hP0⟩ := preLoop A B C s M N BM BN BLOCK_K sam sak sbk sbn scm scn GM
    numKBlocks K ACTIVATION hundef
  -- drive the K-loop
  obtain ⟨final, sLoop, hLoopStmt, hfinal, hPLoop⟩ :=
    forRange_inv (idx := "kk") (start := 0) (stop := numKBlocks) (step := 1)
      (by omega) hP0
      (fun c st hlt hinv => by
        have := matmul_step A B s M N BM BN GM sam sak sbk sbn BLOCK_K numKBlocks K hK c st hlt hinv
        simpa using this)
  -- loop exit: final = numKBlocks
  have hfinalEq : final = numKBlocks := by
    have hle : final ≤ numKBlocks := by
      simp only [matmulInvariant] at hPLoop
      exact hPLoop.2.1
    exact le_antisymm hle hfinal
  rw [hfinalEq] at hPLoop
  -- activation tail + postLoop read off the closed form
  obtain ⟨sfin, hTail, hpost⟩ :=
    matmul_postLoop A B C s M N BM BN GM sam sak sbk sbn scm scn BLOCK_K numKBlocks ACTIVATION
      hInj sLoop hPLoop
  have hexec : exec (matmul_autotune_surface A B C M N K sam sak sbk sbn scm scn
      BM BN BLOCK_K GM numKBlocks ACTIVATION) s = some sfin := by
    rw [exec, matmul_body_split A B C M N K sam sak sbk sbn scm scn BM BN BLOCK_K GM numKBlocks ACTIVATION,
      stepStmts.append_some hpre_eq, stepStmts.cons_some hLoopStmt, hTail]
  rw [hexec]
  -- postLoop's `st.mem` is `sLoop.mem`, which equals `s.mem` (invariant)
  have hsloopmem : sLoop.mem = s.mem := by
    simp only [matmulInvariant] at hPLoop
    exact hPLoop.2.2.2.2.2.2.2.2.2.2.2
  have := hpost idx
  rw [hsloopmem] at this
  exact this

/-- **Closed-form correctness for `matmul_triton_autotune` (general statement).**

For arbitrary linear program id `pid`, tile dims `BM`/`BN`, K-block size `BLOCK_K`,
and K-block count `numKBlocks` (so the contracted dimension is
`K = BLOCK_K · numKBlocks`), every **active** output cell of the computed
`BM × BN` tile equals `fp16( act( Σ_{k < BLOCK_K·numKBlocks} A[i,k] · B[k,j] ) )` —
the genuine matrix product (over ℝ) of the loaded `A`/`B` tiles, optionally passed
through `leaky_relu`, cast to float16 — **not** the kernel's own executed value;
inactive lanes are left untouched.

Layout: `A[i,k]` at `A + offs_am(i)·stride_am + k·stride_ak`, `B[k,j]` at
`B + k·stride_bk + offs_bn(j)·stride_bn`, `C[i,j]` at
`C + stride_cm·offs_cm(i) + stride_cn·offs_cn(j)`, with `pid_m`/`pid_n` derived by
the kernel's L2-grouping schedule, `offs_am(i) = (pid_m·BM + i) % M`,
`offs_bn(j) = (pid_n·BN + j) % N` (the row-major pointer arithmetic with index
wrap). Preconditions: output-offset injectivity and clean initial `undef`. -/
theorem matmul_autotune_closed_form_correct
    (A B C : RegionName) (s : BlockState)
    (M N BM BN GM sam sak sbk sbn scm scn BLOCK_K numKBlocks : Nat) (K : Nat)
    (hK : K = BLOCK_K * numKBlocks) (ACTIVATION : Bool)
    (hcn : scn = 1) (hbnle : BN ≤ scm)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes
      (kernel := matmul_autotune_surface A B C M N K sam sak sbk sbn scm scn
        BM BN BLOCK_K GM numKBlocks ACTIVATION)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s M N BM BN GM)
        (fun idx => (C, cOffset s M N BM BN GM scm scn idx)))
      (expected := fun idx : TileIndex [BM, BN] =>
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (matmulSpec s A B M N BM BN GM sam sak sbk sbn BLOCK_K numKBlocks ACTIVATION idx.1 idx.2.1)))) := by
  subst hcn
  -- output-offset injectivity from the row-major bound `BN ≤ scm` (col stride 1)
  have hInj : Function.Injective (cOffset s M N BM BN GM scm 1) := by
    have heq : cOffset s M N BM BN GM scm 1
        = fun idx : TileIndex [BM, BN] =>
            (scm * (pidM (s.pids 0) M N BM BN GM * BM) + pidN (s.pids 0) M N BM BN GM * BN)
              + idx.1.val * scm + idx.2.1.val := by
      funext idx; simp only [cOffset, rowGlobal, colGlobal]; ring
    rw [heq]; exact rowMajor2D_inj _ scm hbnle
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [matmul_autotune_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst hs0
  intro idx hActive
  have hmain := matmul_autotune_exec_closed_form A B C s0 M N BM BN GM sam sak sbk sbn
    scm 1 BLOCK_K numKBlocks K hK ACTIVATION hInj hundef idx
  have hExec2 : exec (matmul_autotune_surface A B C M N K sam sak sbk sbn scm 1
      BM BN BLOCK_K GM numKBlocks ACTIVATION) s0 = some s' := hExec
  rw [hExec2] at hmain
  rw [if_pos hActive] at hmain
  simpa only [ComputeCorrect.OutputReadable.read_memcell] using hmain

end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune
