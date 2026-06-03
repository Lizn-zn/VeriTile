import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.LoopInvariant

/-!
# `matmul_leakyrelu` — closed-form matmul + leaky-ReLU correctness

`matmul_kernel` is a group-scheduled tiled GEMM with a fused activation: program
`pid` is mapped through an L2-grouping schedule to a tile coordinate
`(pid_m, pid_n)`, accumulates a `BLOCK_SIZE_M × BLOCK_SIZE_N` output tile by
looping over K with `accumulator += tl.dot(a, b)` (with K-tail masking on the
loads), applies `leaky_relu(x) = where(x ≥ 0, x, 0.01·x)`, casts the result to
float16, and stores into `C` masked by `(offs_cm < M) & (offs_cn < N)`.

This file proves the **full kernel** correct against a genuine mathematical
reference: every output cell `C[i,j]` of the computed tile equals
`fp16(leakyrelu(Σ_{k < K} A[i,k] · B[k,j]))` over `ℝ`, where
`K = BLOCK_SIZE_K · numKBlocks` is the contracted dimension and
`leakyrelu(v) = if v ≥ 0 then v else 0.01·v`. This is NOT the kernel's own
emitted value — it is the independent closed-form `Σ_k A·B` GEMM reference, with
the leaky-ReLU activation, derived from the loaded `A`/`B` tiles.

## Proof architecture

```
matmul_leakyrelu_closed_form_correct                 ← TOP THEOREM (ComputeCorrect.Realizes)
  └─ matmul_leakyrelu_exec_closed_form               ← exec-side closed form (every active cell = fp16(leakyrelu(∑_k A·B)))
       ├─ mlr_preLoop      (P 0: accumulator = 0, a/b pointers seeded)
       ├─ mlr_step         (one K-block: accumulator += tl.dot advances the partial sum)
       ├─ mlr_postLoop     (activation + fp16 cast + masked store = the closed form)
       └─ forRange-via-forRangeDyn (loop-invariant principle drives the K-loop)
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float, except that the final fp16
output cast is modeled by `FloatDType.real.cast .fp16`); `@triton.autotune` /
`num_warps` are not modeled. The grouped program-id schedule (`pid_m`/`pid_n`
derivation) is the kernel's own deterministic computation; the spec is stated in
terms of the resulting `pid_m`/`pid_n` (carried as `PM`/`PN`), so it covers every
program of the grid. The layout contract is the kernel's own strided pointer
arithmetic: `a[i,k]` at `A + offs_am i · stride_am + k · stride_ak`, `b[k,j]` at
`B + k · stride_bk + offs_bn j · stride_bn`, `c[i,j]` at
`C + stride_cm · offs_cm i + stride_cn · offs_cn j`.

Preconditions for the general theorem: `0 < BLOCK_SIZE_K` and `K` divisible into
`numKBlocks` full K-blocks (so the per-block K-tail load mask is satisfied for
every loaded lane); tile rows/cols in-bounds (`PM·BM + i < M`, `PN·BN + j < N`,
making `% M`/`% N` the identity and the store mask all-true); output-address
injectivity; clean initial `undef`.
-/

namespace VeriTile.Bench.TritonBenchG.MatmulLeakyrelu

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Faithful transcription of `matmul_leakyrelu.py`'s `matmul_kernel`.

This preserves the grouped program-id mapping, K-block dot loop, pointer
advances, optional leaky-ReLU helper call, output dtype cast, and masked store.
Python's string constexpr `ACTIVATION == "leaky_relu"` is represented by the
Lean Bool parameter `ACTIVATION`. -/
def matmul_kernel_surface
    (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M : Nat)
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
  for k in range($(0), tl.cdiv($(K), $(BLOCK_SIZE_K)), $(1)) {
    a = tl.load(a_ptrs, mask=offs_k[None, :] < $(K) - k * $(BLOCK_SIZE_K), other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < $(K) - k * $(BLOCK_SIZE_K), other=0.0)
    accumulator += tl.dot(a, b)
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

/-- The full matmul/leaky-ReLU surface lowers to the algorithm layer. -/
theorem matmul_kernel_surface_toAlgorithm_supported
    (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M : Nat)
    (ACTIVATION : Bool) :
    ∃ alg, (matmul_kernel_surface A B C M N K stride_am stride_ak stride_bk
      stride_bn stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K
      GROUP_SIZE_M ACTIVATION).toAlgorithm? = Except.ok alg := by
  simp [matmul_kernel_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- Surface transcription of `matmul_leakyrelu.py`'s `matmul_kernel` for
`ACTIVATION == "leaky_relu"`. -/
def matmul_leaky_relu_surface
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
  pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
  pid_n = (pid % num_pid_in_group) // group_size_m
  offs_am = (pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))) % $(M)
  offs_bn = (pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))) % $(N)
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
  accumulator = tl.where(accumulator >= 0.0, accumulator, 0.01 * accumulator)
  c = (accumulator).to(tl.float16)
  offs_cm = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_cn = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  c_ptrs = C + $(stride_cm) * offs_cm[:, None] + $(stride_cn) * offs_cn[None, :]
  c_mask = (offs_cm[:, None] < $(M)) & (offs_cn[None, :] < $(N))
  tl.store(c_ptrs, c, mask=c_mask)
}

/-- The leaky-ReLU specialized matmul surface lowers to the algorithm layer. -/
theorem matmul_leaky_relu_surface_toAlgorithm_supported
    (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M : Nat) :
    ∃ alg, (matmul_leaky_relu_surface A B C M N K stride_am stride_ak
      stride_bk stride_bn stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N
      BLOCK_SIZE_K GROUP_SIZE_M).toAlgorithm? = Except.ok alg := by
  simp [matmul_leaky_relu_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-! ## exec-stepping helpers -/

@[simp] theorem evalOp_ge {dtype a b shape} (h : ComparableDType dtype)
    (bc : Broadcast a b shape) (x : Op dtype a) (y : Op dtype b) (s : BlockState) :
    evalOp (.ge h bc x y) s = (do
      let vx ← evalOp x s; let vy ← evalOp y s; some (Tile.cop h.ge bc vx vy)) := by
  simp [evalOp]

theorem evalOp_boolAnd {a b shape} (bc : Broadcast a b shape)
    (x : Op .bool a) (y : Op .bool b) (s : BlockState) :
    evalOp (.boolAnd bc x y) s = (do
      let vx ← evalOp x s; let vy ← evalOp y s;
      some (Tile.bop (fun p q : Bool => p && q) bc vx vy)) := by
  simp [evalOp]

@[simp] theorem evalOp_remap {dtype inShape outShape}
    (map : TileIndex outShape → TileIndex inShape) (a : Op dtype inShape) (s : BlockState) :
    evalOp (.remap outShape map a) s = (do
      let v ← evalOp a s; some (Tile.remap map v)) := by
  simp [evalOp]

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

/-- `a_ptrs` eval: cell `(i,e) = (A, offs_am i · stride_am + e · stride_ak)`. -/
theorem aptrs_eval (s : BlockState) (A : RegionName) (M K SAM SAK : Nat) (gm : Fin M → Nat)
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

/-- `b_ptrs` eval: cell `(e,j) = (B, e · stride_bk + offs_bn j · stride_bn)`. -/
theorem bptrs_eval (s : BlockState) (B : RegionName) (K N SBK SBN : Nat) (gn : Fin N → Nat)
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

/-- The masked dot of two all-`some` loaded tiles, lane `(i,j)`, equals
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

/-! ## GEMM + leaky-ReLU closed-form spec -/

/-- Ceiling division `⌈a / b⌉`, matching Triton's `tl.cdiv`. -/
def cdiv (a b : Nat) : Nat := (a + b - 1) / b

/-- Leaky-ReLU activation `if v ≥ 0 then v else 0.01·v`. -/
noncomputable def leakyrelu (v : ℝ) : ℝ := if v ≥ 0 then v else (1e-2 : ℝ) * v

/-- The kernel's grouped `num_pid_in_group = GROUP · cdiv N BN`. -/
def numPidInGroup (N BN GROUP : Nat) : Nat := GROUP * cdiv N BN

/-- The kernel's `group_size_m` for program `pid`
(`min(num_pid_m − first_pid_m, GROUP)`, written with `<` as the kernel does). -/
def groupSizeM (pid M N BM BN GROUP : Nat) : Nat :=
  let num_pid_m := cdiv M BM
  let first_pid_m := pid / numPidInGroup N BN GROUP * GROUP
  if num_pid_m - first_pid_m < GROUP then num_pid_m - first_pid_m else GROUP

/-- The kernel's derived `pid_m` for program `pid`. -/
def pidM (pid M N BM BN GROUP : Nat) : Nat :=
  pid / numPidInGroup N BN GROUP * GROUP
    + pid % numPidInGroup N BN GROUP % groupSizeM pid M N BM BN GROUP

/-- The kernel's derived `pid_n` for program `pid`. -/
def pidN (pid M N BM BN GROUP : Nat) : Nat :=
  pid % numPidInGroup N BN GROUP / groupSizeM pid M N BM BN GROUP

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
  (Finset.range (BLOCK_K * numKBlocks)).sum
    (fun k => aElem s A PM BM M SAM SAK i k * bElem s B PN BN N SBK SBN j k)

/-- Partial GEMM accumulator after `c` K-blocks: `Σ_{k < c·BLOCK_K} A·B`. -/
noncomputable def accPartial (s : BlockState) (A B : RegionName)
    (PM PN BM BN M N SAM SAK SBK SBN BLOCK_K : Nat) (i : Fin BM) (j : Fin BN) (c : Nat) : ℝ :=
  (Finset.range (c * BLOCK_K)).sum
    (fun k => aElem s A PM BM M SAM SAK i k * bElem s B PN BN N SBK SBN j k)

/-- One-block step of the partial accumulator. -/
theorem accPartial_succ (s : BlockState) (A B : RegionName)
    (PM PN BM BN M N SAM SAK SBK SBN BLOCK_K : Nat) (i : Fin BM) (j : Fin BN) (c : Nat) :
    accPartial s A B PM PN BM BN M N SAM SAK SBK SBN BLOCK_K i j (c + 1)
      = accPartial s A B PM PN BM BN M N SAM SAK SBK SBN BLOCK_K i j c
        + (Finset.univ.sum fun e : Fin BLOCK_K =>
            aElem s A PM BM M SAM SAK i (c * BLOCK_K + e.val)
              * bElem s B PN BN N SBK SBN j (c * BLOCK_K + e.val)) := by
  unfold accPartial
  have h : (c + 1) * BLOCK_K = c * BLOCK_K + BLOCK_K := by ring
  rw [h, Finset.sum_range_add]
  congr 1
  rw [Finset.sum_range fun e => aElem s A PM BM M SAM SAK i (c * BLOCK_K + e)
        * bElem s B PN BN N SBK SBN j (c * BLOCK_K + e)]

/-! ## Masked fp16 output-store machinery (reused for the activated store) -/

def cOffset (_s : BlockState) (PM PN BM BN stride_cm stride_cn : Nat)
    (idx : TileIndex [BM, BN]) : Nat :=
  stride_cm * rowIndex PM BM idx.1 + stride_cn * colIndex PN BN idx.2.1

private theorem foldl_writeMemTyped_fp16_preserves {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → TileCarrier TileDType.fp16)
    (mask : α → Bool) (o : Nat) (l : List α) :
    ∀ s : BlockState,
      (∀ k ∈ l, mask k = Bool.true → offsetFn k ≠ o) →
        ((l.foldl
          (fun acc k =>
            if mask k then acc.writeMemTyped .fp16 region (offsetFn k) (valueFn k) else acc)
          s).mem region o) = s.mem region o := by
  induction l with
  | nil =>
      intro s _h
      rfl
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
            MemCell.of .fp16 (FloatDType.fp16.ofReal (FloatDType.fp16.storeValue (valueFn hd)))
          else
            s.mem region o) = s.mem region o
        rw [if_neg (by
          intro hsame
          exact hhd hsame.2.symm)]
      · have hmaskhd' : mask hd = Bool.false := by
          cases hm : mask hd
          · rfl
          · exact False.elim (hmaskhd hm)
        simp only [hmaskhd', if_false, Bool.false_eq_true]
        exact ih _ htl

private theorem scatter_memcell_fp16_prop_masked_nd {region : RegionName} {shape : TileShape}
    (s : BlockState) (offsetFn : TileIndex shape → Nat)
    (valueFn : TileIndex shape → TileCarrier TileDType.fp16)
    (P : TileIndex shape → Prop) [DecidablePred P]
    (h_inj : Function.Injective offsetFn) (i : TileIndex shape) :
    ((TileShape.allIndices shape).foldl
       (fun acc k =>
         if P k then acc.writeMemTyped .fp16 region (offsetFn k) (valueFn k) else acc)
       s).mem region (offsetFn i)
    = if P i then
        MemCell.of .fp16 (FloatDType.fp16.ofReal (FloatDType.fp16.storeValue (valueFn i)))
      else
        s.mem region (offsetFn i) := by
  let l := TileShape.allIndices shape
  obtain ⟨l₁, l₂, hl⟩ := List.append_of_mem (TileShape.mem_allIndices shape i)
  have h_nodup := TileShape.allIndices_nodup shape
  change ((l.foldl
       (fun acc k =>
         if P k then acc.writeMemTyped .fp16 region (offsetFn k) (valueFn k) else acc)
       s).mem region (offsetFn i))
    = if P i then
        MemCell.of .fp16 (FloatDType.fp16.ofReal (FloatDType.fp16.storeValue (valueFn i)))
      else
        s.mem region (offsetFn i)
  rw [hl] at h_nodup
  rw [List.nodup_append, List.nodup_cons] at h_nodup
  obtain ⟨_, ⟨hi_notin_l2, _⟩, hl1_disj⟩ := h_nodup
  have hl' : l = l₁ ++ i :: l₂ := by
    simpa [l] using hl
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
        if P k then acc.writeMemTyped .fp16 region (offsetFn k) (valueFn k) else acc)
        =
      (fun (acc : BlockState) k =>
        if decide (P k) then acc.writeMemTyped .fp16 region (offsetFn k) (valueFn k) else acc) := by
    funext acc k
    by_cases hk : P k <;> simp [hk]
  rw [hstep]
  rw [foldl_writeMemTyped_fp16_preserves offsetFn valueFn (fun k => decide (P k))
    (offsetFn i) l₂ _ h_l2_not_in]
  by_cases hPi : P i
  · simp only [hPi, if_true]
    unfold BlockState.writeMemTyped BlockState.writeMemAs
    simp
  · simp only [hPi, if_false]
    rw [foldl_writeMemTyped_fp16_preserves offsetFn valueFn (fun k => decide (P k))
      (offsetFn i) l₁]
    exact h_l1_not_in

/-! ## Body decomposition -/

/-- The 5-statement K-loop body, transcribed (under K = BLOCK_K · numKBlocks the
load masks are all-true, but the body keeps them faithfully). -/
def mlrLoopBody (BM BN BLOCK_K K SAK SBK : Nat) : List Stmt :=
  [ Stmt.assign .real [BM, BLOCK_K] "a"
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
        (Op.dot (batch := []) (Op.ref .real [BM, BLOCK_K] "a") (Op.ref .real [BLOCK_K, BN] "b"))),
    Stmt.assign .ptr [BM, BLOCK_K] "a_ptrs"
      (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BM, BLOCK_K] "a_ptrs")
        (Op.mul .nat Broadcast.nil (Op.constNat BLOCK_K) (Op.constNat SAK))),
    Stmt.assign .ptr [BLOCK_K, BN] "b_ptrs"
      (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BLOCK_K, BN] "b_ptrs")
        (Op.mul .nat Broadcast.nil (Op.constNat BLOCK_K) (Op.constNat SBK))) ]

/-- **preLoop scalars** (statements 0–11): the grouped pid derivation plus the
three index vectors. Steps to a state where `offs_am`/`offs_bn`/`offs_k` hold
the readbacks the pointer chunk needs, and `pid_m`/`pid_n` hold `pidM`/`pidN`. -/
theorem preLoop_scalars (s : BlockState) (M N BM BN BK GROUP : Nat) :
    ∃ s12,
      stepStmts
        [ Stmt.assign .nat [] "pid" (Op.programId 0),
          Stmt.assign .nat [] "num_pid_m"
            (Op.div .nat Broadcast.nil (Op.sub .nat Broadcast.nil
              (Op.add .nat Broadcast.nil (Op.constNat M) (Op.constNat BM)) (Op.constNat 1)) (Op.constNat BM)),
          Stmt.assign .nat [] "num_pid_n"
            (Op.div .nat Broadcast.nil (Op.sub .nat Broadcast.nil
              (Op.add .nat Broadcast.nil (Op.constNat N) (Op.constNat BN)) (Op.constNat 1)) (Op.constNat BN)),
          Stmt.assign .nat [] "num_pid_in_group"
            (Op.mul .nat Broadcast.nil (Op.constNat GROUP) (Op.ref .nat [] "num_pid_n")),
          Stmt.assign .nat [] "group_id"
            (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "pid") (Op.ref .nat [] "num_pid_in_group")),
          Stmt.assign .nat [] "first_pid_m"
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "group_id") (Op.constNat GROUP)),
          Stmt.assign .nat [] "group_size_m"
            ((Op.lt .nat Broadcast.nil
                (Op.sub .nat Broadcast.nil (Op.ref .nat [] "num_pid_m") (Op.ref .nat [] "first_pid_m"))
                (Op.constNat GROUP)).where
              (Op.sub .nat Broadcast.nil (Op.ref .nat [] "num_pid_m") (Op.ref .nat [] "first_pid_m"))
              (Op.constNat GROUP)),
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
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat BM)) (Op.arange BM))
              (Op.constNat M)),
          Stmt.assign .nat [BN] "offs_bn"
            (Op.mod .nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_n") (Op.constNat BN)) (Op.arange BN))
              (Op.constNat N)),
          Stmt.assign .nat [BK] "offs_k" (Op.arange BK) ] s = some s12
      ∧ s12.pids = s.pids
      ∧ s12.regs .nat [] "pid_m" = some (Tile.scalar (pidM (s.pids 0) M N BM BN GROUP))
      ∧ s12.regs .nat [] "pid_n" = some (Tile.scalar (pidN (s.pids 0) M N BM BN GROUP))
      ∧ s12.regs .nat [BM] "offs_am"
          = some (Tile.vec (fun i : Fin BM => (pidM (s.pids 0) M N BM BN GROUP * BM + i.val) % M))
      ∧ s12.regs .nat [BN] "offs_bn"
          = some (Tile.vec (fun j : Fin BN => (pidN (s.pids 0) M N BM BN GROUP * BN + j.val) % N))
      ∧ s12.regs .nat [BK] "offs_k" = some (Tile.vec (fun e : Fin BK => e.val))
      ∧ s12.undef = s.undef
      ∧ s12.mem = s.mem := by
  simp only [pidM, pidN, groupSizeM, numPidInGroup, cdiv]
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
`offs_am`/`offs_bn`/`offs_k` and `pid_m`/`pid_n` registers seeded; `accumulator`
equals the partial GEMM accumulator `accPartial … c`; and `a_ptrs`/`b_ptrs`
advanced by `c` blocks. -/
noncomputable def mlrInvariant
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
  (s.regs .ptr [BM, BLOCK_K] "a_ptrs" = some ⟨fun idx : TileIndex [BM, BLOCK_K] =>
      (A.cast, (PM * BM + idx.1.val) % M * SAM + idx.2.1.val * SAK + c * BLOCK_K * SAK)⟩) ∧
  (s.regs .ptr [BLOCK_K, BN] "b_ptrs" = some ⟨fun idx : TileIndex [BLOCK_K, BN] =>
      (B.cast, idx.1.val * SBK + (PN * BN + idx.2.1.val) % N * SBN + c * BLOCK_K * SBK)⟩) ∧
  (∀ rg o, s.undef rg o = 0) ∧ (s.mem = s0.mem)

set_option maxHeartbeats 1000000 in
/-- **preLoop** (statements 0–14): from a clean input state (`undef = 0`), the
prologue steps to a state satisfying `mlrInvariant … 0` — the base case
(`accumulator = 0`, pointers seeded). -/
theorem mlr_preLoop (A B C : RegionName) (s : BlockState)
    (M N _K SAM SAK SBK SBN SCM SCN BM BN BK GROUP numKBlocks : Nat)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ s', stepStmts ((matmul_leaky_relu_surface A B C M N (BK * numKBlocks) SAM SAK SBK SBN
        SCM SCN BM BN BK GROUP).toAlgKernel.body.take 15) s = some s'
      ∧ mlrInvariant A B s (pidM (s.pids 0) M N BM BN GROUP) (pidN (s.pids 0) M N BM BN GROUP)
          M N BM BN SAM SAK SBK SBN BK numKBlocks 0 s' := by
  obtain ⟨s12, h12, hpids, hpm, hpn, ham, hbn, hk, huf, hmem⟩ :=
    preLoop_scalars s M N BM BN BK GROUP
  set PM := pidM (s.pids 0) M N BM BN GROUP with hPM
  set PN := pidN (s.pids 0) M N BM BN GROUP with hPN
  rw [show ((matmul_leaky_relu_surface A B C M N (BK * numKBlocks) SAM SAK SBK SBN
        SCM SCN BM BN BK GROUP).toAlgKernel.body.take 15)
      = [ Stmt.assign .nat [] "pid" (Op.programId 0),
          Stmt.assign .nat [] "num_pid_m"
            (Op.div .nat Broadcast.nil (Op.sub .nat Broadcast.nil
              (Op.add .nat Broadcast.nil (Op.constNat M) (Op.constNat BM)) (Op.constNat 1)) (Op.constNat BM)),
          Stmt.assign .nat [] "num_pid_n"
            (Op.div .nat Broadcast.nil (Op.sub .nat Broadcast.nil
              (Op.add .nat Broadcast.nil (Op.constNat N) (Op.constNat BN)) (Op.constNat 1)) (Op.constNat BN)),
          Stmt.assign .nat [] "num_pid_in_group"
            (Op.mul .nat Broadcast.nil (Op.constNat GROUP) (Op.ref .nat [] "num_pid_n")),
          Stmt.assign .nat [] "group_id"
            (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "pid") (Op.ref .nat [] "num_pid_in_group")),
          Stmt.assign .nat [] "first_pid_m"
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "group_id") (Op.constNat GROUP)),
          Stmt.assign .nat [] "group_size_m"
            ((Op.lt .nat Broadcast.nil
                (Op.sub .nat Broadcast.nil (Op.ref .nat [] "num_pid_m") (Op.ref .nat [] "first_pid_m"))
                (Op.constNat GROUP)).where
              (Op.sub .nat Broadcast.nil (Op.ref .nat [] "num_pid_m") (Op.ref .nat [] "first_pid_m"))
              (Op.constNat GROUP)),
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
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat BM)) (Op.arange BM))
              (Op.constNat M)),
          Stmt.assign .nat [BN] "offs_bn"
            (Op.mod .nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_n") (Op.constNat BN)) (Op.arange BN))
              (Op.constNat N)),
          Stmt.assign .nat [BK] "offs_k" (Op.arange BK) ]
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
      (aptrs_eval s12 A BM BK SAM SAK (fun i => (PM * BM + i.val) % M) (by simpa using ham) (by simpa using hk))),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (bptrs_eval _ B BK BN SBK SBN (fun j => (PN * BN + j.val) % N) (by simp [hk]) (by simp [hbn]))),
    stepStmts.cons_some (stepStmt_assign_eq_some (acc_init_eval _ BM BN)),
    stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  refine ⟨by simp [hpids], by simp, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · -- accumulator = accPartial … 0 = 0
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx
    simp only [accPartial, Nat.zero_mul, Finset.range_zero, Finset.sum_empty]
  · simp [hpm]
  · simp [hpn]
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

set_option maxHeartbeats 4000000 in
/-- **Step lemma**: one K-loop body iteration advances the invariant by one
block (`accumulator += tl.dot(a, b)` adds the `c`-th block's dot to the partial
GEMM accumulator; the `a`/`b` pointers advance one step). Under
`K = BK · numKBlocks` the per-block K-tail load masks are all satisfied. -/
theorem mlr_step (A B : RegionName) (s0 : BlockState)
    (PM PN M N BM BN SAM SAK SBK SBN BK numKBlocks : Nat) (hBK : 0 < BK)
    (c : Nat) (s : BlockState) (hclt : c < numKBlocks)
    (hinv : mlrInvariant A B s0 PM PN M N BM BN SAM SAK SBK SBN BK numKBlocks c s) :
    ∃ s', stepStmts (mlrLoopBody BM BN BK (BK * numKBlocks) SAK SBK)
        (s.setReg "k" .nat [] (Tile.scalar c)) = some s'
      ∧ mlrInvariant A B s0 PM PN M N BM BN SAM SAK SBK SBN BK numKBlocks (c + 1) s' := by
  set K := BK * numKBlocks with hKdef
  simp only [mlrInvariant] at hinv
  obtain ⟨hpids, hcle, hz, hpm, hpn, ham, hbn, hk, hap, hbp, hundef, hmem⟩ := hinv
  -- mask-all-true bound: every lane `e < BK` satisfies `e < K - c·BK`
  have hlt : ∀ e : Fin BK, e.val < K - c * BK := by
    intro e
    have hcK : c * BK + BK ≤ K := by
      rw [hKdef]; calc c * BK + BK = (c + 1) * BK := by ring
        _ ≤ numKBlocks * BK := Nat.mul_le_mul_right _ hclt
        _ = BK * numKBlocks := Nat.mul_comm _ _
    omega
  -- abbreviations for the seeded pointer/accumulator tiles
  set apT : Tile .ptr [BM, BK] :=
    ⟨fun idx : TileIndex [BM, BK] => (A.cast, (PM * BM + idx.1.val) % M * SAM + idx.2.1.val * SAK + c * BK * SAK)⟩ with hapT
  set bpT : Tile .ptr [BK, BN] :=
    ⟨fun idx : TileIndex [BK, BN] => (B.cast, idx.1.val * SBK + (PN * BN + idx.2.1.val) % N * SBN + c * BK * SBK)⟩ with hbpT
  set zT : Tile .real [BM, BN] :=
    ⟨fun idx : TileIndex [BM, BN] => some (accPartial s0 A B PM PN BM BN M N SAM SAK SBK SBN BK idx.1 idx.2.1 c)⟩ with hzT
  set sk := s.setReg "k" .nat [] (Tile.scalar c) with hsk
  have hrmem : ∀ (R : RegionName) (o : Nat), sk.readMem R o = s0.readMem R o := by
    intro R o; simp only [hsk, BlockState.setReg_readMem]; unfold BlockState.readMem; rw [hmem]
  have hapk : sk.regs .ptr [BM, BK] "a_ptrs" = some apT := by simp [hsk, hap, hapT]
  have hbpk : sk.regs .ptr [BK, BN] "b_ptrs" = some bpT := by simp [hsk, hbp, hbpT]
  have hzk : sk.regs .real [BM, BN] "accumulator" = some zT := by simp [hsk, hz, hzT]
  have hkk : sk.regs .nat [BK] "offs_k" = some (Tile.vec (fun e : Fin BK => e.val)) := by simp [hsk, hk]
  have hkkv : sk.regs .nat [] "k" = some (Tile.scalar c) := by simp [hsk]
  -- loaded tiles
  set asub : Tile .real [BM, BK] :=
    ⟨fun idx => some (sk.readMem (apT.data idx).1 (apT.data idx).2)⟩ with hasub
  -- threaded state after the a-load
  set sk1 := sk.setReg "a" .real [BM, BK] asub with hsk1
  set bsub : Tile .real [BK, BN] :=
    ⟨fun idx => some (sk1.readMem (bpT.data idx).1 (bpT.data idx).2)⟩ with hbsub
  set sk2 := sk1.setReg "b" .real [BK, BN] bsub with hsk2
  have hkk1 : sk1.regs .nat [BK] "offs_k" = some (Tile.vec (fun e : Fin BK => e.val)) := by simp [hsk1, hkk]
  have hkkv1 : sk1.regs .nat [] "k" = some (Tile.scalar c) := by simp [hsk1, hkkv]
  -- masks (each against the threaded state at the load point)
  obtain ⟨amt, ham_eval, ham_true⟩ := amask_alltrue sk BM BK K c hkk hkkv hlt
  obtain ⟨bmt, hbm_eval, hbm_true⟩ := bmask_alltrue sk1 BN BK K c hkk1 hkkv1 hlt
  unfold mlrLoopBody
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
  simp only [mlrInvariant]
  refine ⟨by simp [hsk2, hsk1, hsk, hpids], by omega, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · -- accumulator = accPartial (c+1)
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx
    have has : ∀ e : Fin BK, asub.data (idx.1, e, PUnit.unit)
        = some (aElem s0 A PM BM M SAM SAK idx.1 (c * BK + e.val)) := by
      intro e
      simp only [hasub, hapT, hrmem, aElem, rowIndex]
      congr 2
      ring
    have hrmem1 : ∀ (R : RegionName) (o : Nat), sk1.readMem R o = s0.readMem R o := by
      intro R o; rw [hsk1, BlockState.setReg_readMem]; exact hrmem R o
    have hbs : ∀ e : Fin BK, bsub.data (e, idx.2.1, PUnit.unit)
        = some (bElem s0 B PN BN N SBK SBN idx.2.1 (c * BK + e.val)) := by
      intro e
      simp only [hbsub, hbpT, hrmem1, bElem, colIndex]
      congr 2
      ring
    rw [accadd_eval BM BN zT (Tile.dot [] asub bsub) idx.1 idx.2.1
        (accPartial s0 A B PM PN BM BN M N SAM SAK SBK SBN BK idx.1 idx.2.1 c)
        (Finset.univ.sum fun e : Fin BK =>
          aElem s0 A PM BM M SAM SAK idx.1 (c * BK + e.val) * bElem s0 B PN BN N SBK SBN idx.2.1 (c * BK + e.val))
        (by rw [hzT])
        (dot_ab BM BK BN asub bsub idx.1 idx.2.1 _ _ has hbs)]
    show some _ = some (accPartial s0 A B PM PN BM BN M N SAM SAK SBK SBN BK idx.1 idx.2.1 (c + 1))
    rw [accPartial_succ]
  · simp [hsk2, hsk1, hsk, hpm]
  · simp [hsk2, hsk1, hsk, hpn]
  · simp [hsk2, hsk1, hsk, ham]
  · simp [hsk2, hsk1, hsk, hbn]
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
  · show _ = s0.mem
    rw [← hmem]; rfl

/-! ## Post-loop: activation + fp16 cast + masked store -/

/-- The 7 post-loop statements: leaky-ReLU activation, fp16 cast, output index
vectors, output pointers, output mask, masked fp16 store. -/
def mlrPostBody (C : RegionName) (M N SCM SCN BM BN : Nat) : List Stmt :=
  [ Stmt.assign .real [BM, BN] "accumulator"
      ((Op.ge ComparableDType.real Broadcast.scalarR (Op.ref .real [BM, BN] "accumulator")
            (Op.const 0.0)).where
        (Op.ref .real [BM, BN] "accumulator")
        (Op.mul .real Broadcast.scalarL (Op.const 1e-2) (Op.ref .real [BM, BN] "accumulator"))),
    Stmt.assign FloatDType.fp16.toTileDType [BM, BN] "c"
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

/-- The kernel body decomposes as prefix (15) ++ [K-loop, post-statements]. The
K-loop is a `forRangeDyn` whose stop op is `cdiv K BK`. By `rfl`. -/
theorem mlr_body_split (A B C : RegionName)
    (M N SAM SAK SBK SBN SCM SCN BM BN BK GROUP numKBlocks : Nat) :
    (matmul_leaky_relu_surface A B C M N (BK * numKBlocks) SAM SAK SBK SBN SCM SCN BM BN BK GROUP).toAlgKernel.body
      = (matmul_leaky_relu_surface A B C M N (BK * numKBlocks) SAM SAK SBK SBN SCM SCN BM BN BK GROUP).toAlgKernel.body.take 15
        ++ (Stmt.forRangeDyn "k" (Op.constNat 0)
              (Op.div .nat Broadcast.nil (Op.sub .nat Broadcast.nil
                (Op.add .nat Broadcast.nil (Op.constNat (BK * numKBlocks)) (Op.constNat BK)) (Op.constNat 1)) (Op.constNat BK))
              (Op.constNat 1) (mlrLoopBody BM BN BK (BK * numKBlocks) SAK SBK)
            :: mlrPostBody C M N SCM SCN BM BN) := by
  rfl

/-- **Activation eval**: `where(acc ≥ 0, acc, 0.01·acc)` applied to an
all-`some` real tile is the cell-wise leaky-ReLU. -/
theorem activation_eval (BM BN : Nat) (s : BlockState) (g : TileIndex [BM, BN] → ℝ)
    (hacc : s.regs .real [BM, BN] "accumulator" = some ⟨fun idx => some (g idx)⟩) :
    evalOp ((Op.ge ComparableDType.real Broadcast.scalarR (Op.ref .real [BM, BN] "accumulator")
            (Op.const 0.0)).where
        (Op.ref .real [BM, BN] "accumulator")
        (Op.mul .real Broadcast.scalarL (Op.const 1e-2) (Op.ref .real [BM, BN] "accumulator"))) s
      = some (⟨fun idx : TileIndex [BM, BN] => some (leakyrelu (g idx))⟩ : Tile .real [BM, BN]) := by
  simp only [evalOp_where, evalOp_ge, evalOp_mul, evalOp_const, evalOp_ref, hacc, Option.bind,
    Option.bind_some, Option.bind_eq_bind]
  refine congrArg some ?_
  ext idx
  simp only [Tile.select_data, Tile.cop_data, Tile.scalar, Tile.data, Broadcast.leftIndex,
    Broadcast.rightIndex]
  have hge := ComparableDType.real_ge_eq_true (some (g idx)) (some (0.0:ℝ))
  have hbridge : ((((g idx:ℝ)) : WithBot ℝ) ≥ (((0.0:ℝ)) : WithBot ℝ)) ↔ (0:ℝ) ≤ g idx := by
    rw [ge_iff_le, WithBot.coe_le_coe]; norm_num
  by_cases h : (0:ℝ) ≤ g idx
  · have hb : ComparableDType.real.ge (some (g idx)) (some 0.0) = Bool.true := hge.mpr (hbridge.mpr h)
    rw [if_pos hb]
    simp only [leakyrelu, ge_iff_le, if_pos h]
  · have hb : ComparableDType.real.ge (some (g idx)) (some 0.0) = Bool.false := by
      cases hbb : ComparableDType.real.ge (some (g idx)) (some 0.0)
      · rfl
      · exact absurd (h (hbridge.mp (hge.mp hbb))) (by simp)
    rw [if_neg (by rw [hb]; simp)]
    simp only [leakyrelu, ge_iff_le, if_neg h, Tile.bop_data, Broadcast.leftIndex,
      Broadcast.rightIndex, Tile.scalar, Tile.data, NumericDType.mul, WithBot.realMul, Option.map₂,
      Option.bind, Option.map]

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
theorem mlrStore_eval (_C : RegionName) (BM BN : Nat) (st : BlockState)
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

/-- The genuine output cell: `fp16(leakyrelu(Σ_k A·B))` as a `MemCell`. -/
noncomputable def outputCell (s0 : BlockState) (A B : RegionName)
    (PM PN BM BN M N SAM SAK SBK SBN BK numKBlocks : Nat) (idx : TileIndex [BM, BN]) : MemCell :=
  MemCell.of .fp16 (FloatDType.fp16.ofReal (FloatDType.fp16.storeValue
    (FloatDType.real.cast FloatDType.fp16
      (some (leakyrelu (matmulSpec s0 A B PM PN BM BN M N SAM SAK SBK SBN BK numKBlocks idx.1 idx.2.1))))))

set_option maxHeartbeats 2000000 in
/-- **postLoop**: from the invariant at `numKBlocks` blocks, the activation +
fp16 cast + masked store writes the genuine value
`fp16(leakyrelu(Σ_k A·B))` at every in-bounds output lane. -/
theorem mlr_postLoop (A B C : RegionName) (s0 : BlockState)
    (PM PN M N BM BN SAM SAK SBK SBN SCM SCN BK numKBlocks : Nat)
    (hInj : Function.Injective (cOffset s0 PM PN BM BN SCM SCN))
    (hmlt : ∀ i : Fin BM, rowIndex PM BM i < M)
    (hnlt : ∀ j : Fin BN, colIndex PN BN j < N)
    (st : BlockState)
    (hinv : mlrInvariant A B s0 PM PN M N BM BN SAM SAK SBK SBN BK numKBlocks numKBlocks st) :
    ∃ sfin, stepStmts (mlrPostBody C M N SCM SCN BM BN) st = some sfin
      ∧ ∀ idx : TileIndex [BM, BN],
          sfin.mem C (cOffset s0 PM PN BM BN SCM SCN idx)
            = outputCell s0 A B PM PN BM BN M N SAM SAK SBK SBN BK numKBlocks idx := by
  simp only [mlrInvariant] at hinv
  obtain ⟨hpids, hcle, hz, hpm, hpn, ham, hbn, hk, hap, hbp, hundef, hmem⟩ := hinv
  -- abbreviation for the full GEMM accumulator (matmulSpec)
  set g : TileIndex [BM, BN] → ℝ :=
    fun idx => matmulSpec s0 A B PM PN BM BN M N SAM SAK SBK SBN BK numKBlocks idx.1 idx.2.1 with hg
  have hzspec : st.regs .real [BM, BN] "accumulator" = some ⟨fun idx => some (g idx)⟩ := by
    rw [hz]; refine congrArg some ?_; ext idx; simp only [hg, matmulSpec, accPartial, Nat.mul_comm numKBlocks BK]
  unfold mlrPostBody
  -- step 1: activation
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (activation_eval BM BN st g hzspec))]
  set s1 := st.setReg "accumulator" .real [BM, BN] ⟨fun idx => some (leakyrelu (g idx))⟩ with hs1
  -- step 2: fp16 cast
  have hcast : evalOp (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref .real [BM, BN] "accumulator")) s1
      = some (⟨fun idx => FloatDType.real.cast FloatDType.fp16 (some (leakyrelu (g idx)))⟩ : Tile .fp16 [BM, BN]) := by
    rw [hs1, evalOp_castFloat]
    simp [BlockState.setReg_same]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some hcast)]
  set s2 := s1.setReg "c" FloatDType.fp16.toTileDType [BM, BN] (⟨fun idx => FloatDType.real.cast FloatDType.fp16 (some (leakyrelu (g idx)))⟩ : Tile .fp16 [BM, BN]) with hs2
  -- step 3: offs_cm
  have hpm2 : s2.regs .nat [] "pid_m" = some (Tile.scalar PM) := by rw [hs2, hs1]; simp [hpm]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (offs_cm_eval s2 PM BM hpm2))]
  set s3 := s2.setReg "offs_cm" .nat [BM] (Tile.vec (fun i : Fin BM => rowIndex PM BM i)) with hs3
  -- step 4: offs_cn
  have hpn3 : s3.regs .nat [] "pid_n" = some (Tile.scalar PN) := by simp [hs3, hs2, hs1, hpn]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (offs_cn_eval s3 PN BN hpn3))]
  set s4 := s3.setReg "offs_cn" .nat [BN] (Tile.vec (fun j : Fin BN => colIndex PN BN j)) with hs4
  -- step 5: c_ptrs
  have hcm4 : s4.regs .nat [BM] "offs_cm" = some (Tile.vec (fun i : Fin BM => rowIndex PM BM i)) := by simp [hs4, hs3]
  have hcn4 : s4.regs .nat [BN] "offs_cn" = some (Tile.vec (fun j : Fin BN => colIndex PN BN j)) := by simp [hs4]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (cptrs_eval s4 C BM BN SCM SCN (fun i => rowIndex PM BM i) (fun j => colIndex PN BN j) hcm4 hcn4))]
  set cpT : Tile .ptr [BM, BN] :=
    ⟨fun idx : TileIndex [BM, BN] => (C.cast, SCM * rowIndex PM BM idx.1 + SCN * colIndex PN BN idx.2.1)⟩ with hcpT
  set s5 := s4.setReg "c_ptrs" .ptr [BM, BN] cpT with hs5
  -- step 6: c_mask
  have hcm5 : s5.regs .nat [BM] "offs_cm" = some (Tile.vec (fun i : Fin BM => rowIndex PM BM i)) := by simp [hs5, hcm4]
  have hcn5 : s5.regs .nat [BN] "offs_cn" = some (Tile.vec (fun j : Fin BN => colIndex PN BN j)) := by simp [hs5, hcn4]
  obtain ⟨mT, hmask_eval, hmask_true⟩ :=
    cmask_alltrue s5 M N BM BN (fun i => rowIndex PM BM i) (fun j => colIndex PN BN j) hcm5 hcn5 hmlt hnlt
  rw [stepStmts.cons_some (stepStmt_assign_eq_some hmask_eval)]
  set s6 := s5.setReg "c_mask" .bool [BM, BN] mT with hs6
  -- step 7: masked fp16 store
  set cT : Tile .fp16 [BM, BN] :=
    ⟨fun idx => FloatDType.real.cast FloatDType.fp16 (some (leakyrelu (g idx)))⟩ with hcT
  have hcc : s6.regs .fp16 [BM, BN] "c" = some cT := by simp [hs6, hs5, hs4, hs3, hs2, hcT]
  have hcpp : s6.regs .ptr [BM, BN] "c_ptrs" = some cpT := by simp [hs6, hs5]
  have hmm : s6.regs .bool [BM, BN] "c_mask" = some mT := by simp [hs6]
  rw [stepStmts.cons_some (mlrStore_eval C BM BN s6 cT cpT mT hcc hcpp hmm), stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  intro idx
  -- mask is all-true, so the store is unconditional; readback via the scatter lemma
  have hstep_eq :
      (fun (acc : BlockState) k =>
        if mT.data k then acc.writeMemTyped .fp16 (cpT.data k).1 (cpT.data k).2 (cT.data k) else acc)
        =
      (fun (acc : BlockState) k =>
        if (fun _ : TileIndex [BM, BN] => True) k then acc.writeMemTyped .fp16 (cpT.data k).1 (cpT.data k).2 (cT.data k) else acc) := by
    funext acc k; rw [hmask_true k]; simp
  rw [hstep_eq]
  have hCregion : ∀ k : TileIndex [BM, BN], (cpT.data k).1 = C := by intro k; simp [hcpT]
  -- rewrite the foldl region from `(cpT.data k).1` to the literal `C`
  rw [show
      ((TileShape.allIndices [BM, BN]).foldl
        (fun acc k => if (fun _ : TileIndex [BM, BN] => True) k then acc.writeMemTyped .fp16 (cpT.data k).1 (cpT.data k).2 (cT.data k) else acc) s6)
      = ((TileShape.allIndices [BM, BN]).foldl
        (fun acc k => if (fun _ : TileIndex [BM, BN] => True) k then acc.writeMemTyped .fp16 C ((cpT.data k).2) (cT.data k) else acc) s6)
      from List.foldl_ext _ _ s6 (fun acc k _ => by rw [hCregion k])]
  have hcoff : (cpT.data idx).2 = cOffset s0 PM PN BM BN SCM SCN idx := by simp [hcpT, cOffset]
  have hoffInj : Function.Injective (fun idx : TileIndex [BM, BN] => (cpT.data idx).2) := by
    intro a b hab; apply hInj; simpa [cOffset, hcpT] using hab
  rw [← hcoff]
  rw [scatter_memcell_fp16_prop_masked_nd (region := C) (s := s6)
    (offsetFn := fun idx : TileIndex [BM, BN] => (cpT.data idx).2)
    (valueFn := fun idx => cT.data idx)
    (P := fun _ => True) hoffInj idx]
  simp only [if_pos trivial, outputCell, hcT, hg]

/-- The dynamic K-loop bound resolves: `evalOp (cdiv K BK) = numKBlocks` and the
loop runs `numKBlocks` iterations, advancing `mlrInvariant` from `0` to
`numKBlocks`. -/
theorem mlr_loop (A B : RegionName) (s0 : BlockState)
    (PM PN M N BM BN SAM SAK SBK SBN BK numKBlocks : Nat) (hBK : 0 < BK)
    (s : BlockState)
    (hP0 : mlrInvariant A B s0 PM PN M N BM BN SAM SAK SBK SBN BK numKBlocks 0 s) :
    ∃ sLoop, stepStmt (Stmt.forRangeDyn "k" (Op.constNat 0)
        (Op.div .nat Broadcast.nil (Op.sub .nat Broadcast.nil
          (Op.add .nat Broadcast.nil (Op.constNat (BK * numKBlocks)) (Op.constNat BK)) (Op.constNat 1)) (Op.constNat BK))
        (Op.constNat 1) (mlrLoopBody BM BN BK (BK * numKBlocks) SAK SBK)) s = some sLoop
      ∧ mlrInvariant A B s0 PM PN M N BM BN SAM SAK SBK SBN BK numKBlocks numKBlocks sLoop := by
  have hcdiv : (BK * numKBlocks + BK - 1) / BK = numKBlocks := by
    have he : BK * numKBlocks + BK - 1 = (BK - 1) + BK * numKBlocks := by omega
    rw [he, Nat.add_mul_div_left _ _ hBK, Nat.div_eq_of_lt (by omega), Nat.zero_add]
  -- resolve the dynamic bounds: `stepStmt (forRangeDyn ...) s = stepForRangeAux "k" 0 numKBlocks 1 body s`
  have hresolve : stepStmt (Stmt.forRangeDyn "k" (Op.constNat 0)
      (Op.div .nat Broadcast.nil (Op.sub .nat Broadcast.nil
        (Op.add .nat Broadcast.nil (Op.constNat (BK * numKBlocks)) (Op.constNat BK)) (Op.constNat 1)) (Op.constNat BK))
      (Op.constNat 1) (mlrLoopBody BM BN BK (BK * numKBlocks) SAK SBK)) s
      = stepForRangeAux "k" 0 numKBlocks 1 (mlrLoopBody BM BN BK (BK * numKBlocks) SAK SBK) s := by
    rw [stepForRangeAux.forRangeDyn_unfold]
    simp only [evalOp_constNat, evalOp_div, evalOp_sub, evalOp_add, Tile.scalar, Tile.bop,
      Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.div, NumericDType.sub,
      NumericDType.add, Tile.data, Option.bind_some, bind]
    rw [hcdiv]
  rw [hresolve]
  obtain ⟨final, sLoop, haux, hfinal, hPfinal⟩ :=
    forRangeAux_inv (idx := "k") (stop := numKBlocks) (step := 1)
      (P := mlrInvariant A B s0 PM PN M N BM BN SAM SAK SBK SBN BK numKBlocks)
      (by norm_num)
      (fun i st hlt hinv => mlr_step A B s0 PM PN M N BM BN SAM SAK SBK SBN BK numKBlocks hBK i st hlt hinv)
      0 s hP0
  have hfinalEq : final = numKBlocks := by
    simp only [mlrInvariant] at hPfinal
    omega
  subst hfinalEq
  exact ⟨sLoop, haux, hPfinal⟩

/-! ## Composition: full exec closed form -/

set_option maxHeartbeats 2000000 in
/-- **Top exec reduction**: `preLoop` + K-loop (`mlr_loop`) + `postLoop`
compose into the full `exec`. Every in-bounds output lane equals the genuine
value `fp16(leakyrelu(Σ_k A·B))`. -/
theorem mlr_exec_closed_form (A B C : RegionName) (s : BlockState)
    (M N SAM SAK SBK SBN SCM SCN BM BN BK GROUP numKBlocks : Nat) (hBK : 0 < BK)
    (hInj : Function.Injective (cOffset s (pidM (s.pids 0) M N BM BN GROUP) (pidN (s.pids 0) M N BM BN GROUP) BM BN SCM SCN))
    (hmlt : ∀ i : Fin BM, rowIndex (pidM (s.pids 0) M N BM BN GROUP) BM i < M)
    (hnlt : ∀ j : Fin BN, colIndex (pidN (s.pids 0) M N BM BN GROUP) BN j < N)
    (hundef : ∀ rg o, s.undef rg o = 0)
    (idx : TileIndex [BM, BN]) :
    (match exec (matmul_leaky_relu_surface A B C M N (BK * numKBlocks) SAM SAK SBK SBN SCM SCN BM BN BK GROUP) s with
      | some s' => s'.mem C (cOffset s (pidM (s.pids 0) M N BM BN GROUP) (pidN (s.pids 0) M N BM BN GROUP) BM BN SCM SCN idx)
      | none => (0 : MemCell)) =
      outputCell s A B (pidM (s.pids 0) M N BM BN GROUP) (pidN (s.pids 0) M N BM BN GROUP)
        BM BN M N SAM SAK SBK SBN BK numKBlocks idx := by
  set PM := pidM (s.pids 0) M N BM BN GROUP with hPM
  set PN := pidN (s.pids 0) M N BM BN GROUP with hPN
  obtain ⟨s0, hpre_eq, hP0⟩ := mlr_preLoop A B C s M N (BK * numKBlocks) SAM SAK SBK SBN SCM SCN BM BN BK GROUP numKBlocks hundef
  obtain ⟨sLoop, hLoopStmt, hPLoop⟩ :=
    mlr_loop A B s PM PN M N BM BN SAM SAK SBK SBN BK numKBlocks hBK s0 hP0
  obtain ⟨sfin, hTail, hpost⟩ :=
    mlr_postLoop A B C s PM PN M N BM BN SAM SAK SBK SBN SCM SCN BK numKBlocks hInj hmlt hnlt sLoop hPLoop
  have hexec : exec (matmul_leaky_relu_surface A B C M N (BK * numKBlocks) SAM SAK SBK SBN SCM SCN BM BN BK GROUP) s
      = some sfin := by
    rw [exec, mlr_body_split A B C M N SAM SAK SBK SBN SCM SCN BM BN BK GROUP numKBlocks,
      stepStmts.append_some hpre_eq, stepStmts.cons_some hLoopStmt, hTail]
  rw [hexec]
  exact hpost idx

/-- **Closed-form correctness for `matmul_leakyrelu` (general statement).**

For arbitrary `M`/`N`, tile dims `BM`/`BN`, strides, group size, K-block size
`BK`, and K-block count `numKBlocks` (so `K = BK · numKBlocks`), every in-bounds
output cell of the computed `BM × BN` tile equals the genuine fused value
`fp16(leakyrelu(Σ_{k < BK·numKBlocks} A[i,k] · B[k,j]))` (over ℝ, with a final
fp16 output cast) of the loaded `A`/`B` tiles — NOT the kernel's own executed
value.

`PM`/`PN` are the kernel's own grouped `pid_m`/`pid_n`. Preconditions: `0 < BK`;
all tile rows/cols in-bounds (`PM·BM + i < M`, `PN·BN + j < N`), making the
modular addressing the identity and the store mask all-true; output-address
injectivity; clean initial `undef`. -/
theorem matmul_leakyrelu_closed_form_correct
    (A B C : RegionName) (s : BlockState)
    (M N SAM SAK SBK SBN SCM SCN BM BN BK GROUP numKBlocks : Nat) (hBK : 0 < BK)
    (hInj : Function.Injective (cOffset s (pidM (s.pids 0) M N BM BN GROUP) (pidN (s.pids 0) M N BM BN GROUP) BM BN SCM SCN))
    (hmlt : ∀ i : Fin BM, rowIndex (pidM (s.pids 0) M N BM BN GROUP) BM i < M)
    (hnlt : ∀ j : Fin BN, colIndex (pidN (s.pids 0) M N BM BN GROUP) BN j < N)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes
      (kernel := matmul_leaky_relu_surface A B C M N (BK * numKBlocks) SAM SAK SBK SBN SCM SCN BM BN BK GROUP)
      (initialState := s)
      (write := fun idx : TileIndex [BM, BN] =>
        some (C, cOffset s (pidM (s.pids 0) M N BM BN GROUP) (pidN (s.pids 0) M N BM BN GROUP) BM BN SCM SCN idx))
      (expected := fun idx : TileIndex [BM, BN] =>
        outputCell s A B (pidM (s.pids 0) M N BM BN GROUP) (pidN (s.pids 0) M N BM BN GROUP)
          BM BN M N SAM SAK SBK SBN BK numKBlocks idx) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [matmul_leaky_relu_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst hs0
  intro idx
  have hmain := mlr_exec_closed_form A B C s0 M N SAM SAK SBK SBN SCM SCN BM BN BK GROUP numKBlocks hBK hInj hmlt hnlt hundef idx
  rw [hExec] at hmain
  exact hmain

/-- **Public Python test-shape summary** (`64×128 @ 128×64`, leaky-ReLU enabled,
`BLOCK_M = BLOCK_N = BLOCK_K = 32`, `GROUP_SIZE_M = 4`, so `numKBlocks = 4`,
strides `a=(128,1)`, `b=(64,1)`, `c=(64,1)`): the full `matmul_leakyrelu` surface
lowers to the algorithm layer and realizes the genuine fused matrix product
`fp16(leakyrelu(Σ_{k<128} A[i,k]·B[k,j]))` on every in-bounds output lane. The
in-bounds / output-injectivity hypotheses are kept because they depend on the
runtime grouped `pid`. -/
theorem matmul_leakyrelu_python_test_shape_summary
    (A B C : RegionName) (s : BlockState)
    (hInj : Function.Injective (cOffset s (pidM (s.pids 0) 64 64 32 32 4) (pidN (s.pids 0) 64 64 32 32 4) 32 32 64 1))
    (hmlt : ∀ i : Fin 32, rowIndex (pidM (s.pids 0) 64 64 32 32 4) 32 i < 64)
    (hnlt : ∀ j : Fin 32, colIndex (pidN (s.pids 0) 64 64 32 32 4) 32 j < 64)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    (∃ alg, (matmul_leaky_relu_surface A B C 64 64 128 128 1 64 1 64 1 32 32 32 4).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := matmul_leaky_relu_surface A B C 64 64 128 128 1 64 1 64 1 32 32 32 4)
      (initialState := s)
      (write := fun idx : TileIndex [32, 32] =>
        some (C, cOffset s (pidM (s.pids 0) 64 64 32 32 4) (pidN (s.pids 0) 64 64 32 32 4) 32 32 64 1 idx))
      (expected := fun idx : TileIndex [32, 32] =>
        outputCell s A B (pidM (s.pids 0) 64 64 32 32 4) (pidN (s.pids 0) 64 64 32 32 4)
          32 32 64 64 128 1 64 1 32 4 idx) := by
  refine ⟨matmul_leaky_relu_surface_toAlgorithm_supported A B C 64 64 128 128 1 64 1 64 1 32 32 32 4, ?_⟩
  exact matmul_leakyrelu_closed_form_correct A B C s 64 64 128 1 64 1 64 1 32 32 32 4 4
    (by norm_num) hInj hmlt hnlt hundef

end VeriTile.Bench.TritonBenchG.MatmulLeakyrelu
