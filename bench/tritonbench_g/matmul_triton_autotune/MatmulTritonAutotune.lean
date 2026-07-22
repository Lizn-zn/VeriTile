import VeriTile.Triton

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
matmul_autotune_closed_form_correct               ← TOP THEOREM (ComputeCorrect.Realizes_without_Rounding)
  └─ matmul_autotune_exec_closed_form             ← exec-side closed form (every active cell)
       ├─ preLoop      (P 0: accumulator = 0, pointers seeded, schedule derived)
       ├─ matmul_step         (one K-block: masked dot advances the partial sum)
       ├─ matmul_postLoop     (activation tail + fp16 cast + masked store = closed form)
       └─ forRange_inv        (loop-invariant principle, drives the K-loop)

matmul_autotune_io_correctness                    ← STREAMING `⊨[R]` HEADLINE (StreamMasked2DKernelIO₂)
  ├─ matmul_autotune_flattenOk / matmul_autotune_traceSafeR    (flat bridge + looping safety walk)
  └─ preLoop + matmul_step (reused verbatim; prologue/loop/activation are cast-free)
       └─ matmul_autotune_postLoopR   (R store tail: R.cast + masked writeMemAsR, one boundary round)
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

## Translation-surface blocker

Translation-surface blocker: the loop trip count `tl.cdiv(K, BLOCK_SIZE_K)`
is supplied as the antiquoted `numKBlocks` binder (so that `tl.cdiv` call does
not appear as a surface statement), and the K-loop counter is spelled `kk`
where Python spells it `k` (avoiding a clash with the antiquoted dimension
binder `K`). The textual py↔lean scans in `bench/audit_tritonbench_g.sh`
exempt this port on this marker (registered in `proof_blockers.md`).
-/

namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune

open VeriTile.Triton

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
specification matmul_autotune_closed_form_correct
    (A B C : RegionName) (s : BlockState)
    (M N BM BN GM sam sak sbk sbn scm scn BLOCK_K numKBlocks : Nat) (K : Nat)
    (hK : K = BLOCK_K * numKBlocks) (ACTIVATION : Bool)
    (hcn : scn = 1) (hbnle : BN ≤ scm)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes_without_Rounding
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

/-! ## The `⊨[R]` streaming headline (wave-5 S1 fold genre)

Everything below is purely additive; the exact surface above is untouched.
Structure of the `execR R` story: the kernel's only two rounding events live
in `matmulStoreTail` — the `tl.cast(accumulator, tl.float16)` (`evalOpR`
site 1) and the `.fp16`-typed masked `tl.store` (`writeMemAsR` site 2). The
prologue, the whole K-loop (masked loads with `other=0.0` carry no
`castFloat`), and the `leaky_relu` activation tail are cast-free, so under
`execR R` they collapse verbatim onto the exact stepper and the proven
`preLoop` / `matmul_step` / `forRange_inv` / `matmulActStmt_eval` stack above
is reused unchanged; only the 6-statement store tail is re-proved on the `R`
side (`matmul_autotune_postLoopR`). `round_idem` (via
`RoundingModel.storeValue_cast`) collapses the tail's double round into the
single boundary `R.round .fp16` the skin's readback contract states. The
constexpr `ACTIVATION` gate stays the `Bool` parameter throughout: the io,
the safety walk, and the headline all quantify over it, and the spec routes
it through the named `act` brancher (both branches genuine). -/

open scoped VeriTile.Triton.StreamMasked2DKernelIO₂

/-! ### Body decomposition names and cast-free collapses -/

/-- The 15-statement prologue (statements 0–14) as an explicit list: the 9
L2-grouping schedule scalars, the three index vectors
`offs_am`/`offs_bn`/`offs_k`, the two pointer seeds, and the zero
accumulator. -/
private def matmulAutotunePrologue (A B : RegionName)
    (M N BM BN BK GM sam sak sbk sbn : Nat) : List Stmt :=
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
    Stmt.assign .nat [BK] "offs_k" (Op.arange BK),
    Stmt.assign .ptr [BM, BK] "a_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase A)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_am")) (Op.constNat sam))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BK] "offs_k")) (Op.constNat sak)))),
    Stmt.assign .ptr [BK, BN] "b_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase B)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "offs_k")) (Op.constNat sbk))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_bn")) (Op.constNat sbn)))),
    Stmt.assign .real [BM, BN] "accumulator" (Op.full [BM, BN] (Op.const 0)) ]

private theorem matmul_autotune_take15_eq (A B C : RegionName)
    (M N K sam sak sbk sbn scm scn BM BN BK GM numKBlocks : Nat) (ACTIVATION : Bool) :
    ((matmul_autotune_surface A B C M N K sam sak sbk sbn scm scn BM BN BK GM
        numKBlocks ACTIVATION).toAlgKernel.body.take 15)
      = matmulAutotunePrologue A B M N BM BN BK GM sam sak sbk sbn := rfl

/-- `matmul_body_split` with the prologue named. By `rfl`. -/
private theorem matmul_autotune_body_split' (A B C : RegionName)
    (M N K sam sak sbk sbn scm scn BM BN BK GM numKBlocks : Nat) (ACTIVATION : Bool) :
    (matmul_autotune_surface A B C M N K sam sak sbk sbn scm scn BM BN BK GM
        numKBlocks ACTIVATION).toAlgKernel.body
      = matmulAutotunePrologue A B M N BM BN BK GM sam sak sbk sbn
        ++ (Stmt.forRange "kk" 0 numKBlocks 1 (matmulLoopBody K BM BN BK sak sbk)
            :: matmulActStmt BM BN ACTIVATION
            :: matmulStoreTail C M N scm scn BM BN) := rfl

set_option maxHeartbeats 1000000 in
/-- The prologue is cast-free: it steps identically under `stepStmtsR R`. -/
private theorem matmulAutotunePrologue_castFree (R : RoundingModel) (A B : RegionName)
    (M N BM BN BK GM sam sak sbk sbn : Nat) (t : BlockState) :
    stepStmtsR R (matmulAutotunePrologue A B M N BM BN BK GM sam sak sbk sbn) t
      = stepStmts (matmulAutotunePrologue A B M N BM BN BK GM sam sak sbk sbn) t := by
  simp only [matmulAutotunePrologue, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def]
  rfl

set_option maxHeartbeats 1000000 in
/-- The K-loop body is cast-free: the masked loads' `mask=`/`other=` operands
are nat comparisons and a real constant (no `castFloat`), the fused
`tl.dot`+add and the pointer advances are exact ops, and the body has no
store — so it steps identically under `stepStmtsR R` and the exact invariant
stack transports to `execR`. -/
private theorem matmulAutotuneBody_castFree (R : RoundingModel)
    (K BM BN BK sak sbk : Nat) (t : BlockState) :
    stepStmtsR R (matmulLoopBody K BM BN BK sak sbk) t
      = stepStmts (matmulLoopBody K BM BN BK sak sbk) t := by
  simp only [matmulLoopBody, aMaskOp, bMaskOp, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def]
  rfl

set_option maxHeartbeats 1000000 in
/-- The activation tail is cast-free (`tl.where`/`>=`/`*` over `.real`; the
`ACTIVATION` gate is a `constBool`): it steps identically under `stepStmtR R`
for **both** values of the gate, so the exact `matmulActStmt_eval` above is
also its `execR` story. -/
private theorem matmulActStmt_castFree (R : RoundingModel) (BM BN : Nat)
    (ACTIVATION : Bool) (t : BlockState) :
    stepStmtR R (matmulActStmt BM BN ACTIVATION) t
      = stepStmt (matmulActStmt BM BN ACTIVATION) t := by
  simp only [matmulActStmt, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def]
  rfl

/-! ### Cast-free op collapses and the two rounding-event sites -/

/-- The `offs_c*` index-vector op is cast-free. -/
private theorem evalR_offsc (R : RoundingModel) (Mv BMc : Nat) (pidReg : RegName)
    (s : BlockState) :
    evalOpR R (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] pidReg) (Op.constNat BMc))
        (Op.arange Mv)) s
      = evalOp (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] pidReg) (Op.constNat BMc))
        (Op.arange Mv)) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

/-- `offs_cm` / `offs_cn` eval under `R` (via the cast-free collapse). -/
private theorem offscm_evalR (R : RoundingModel) (s : BlockState) (Mv BMc : Nat)
    (pm : Nat) (pidReg : RegName)
    (hpm : s.regs .nat [] pidReg = some (Tile.scalar pm)) :
    evalOpR R (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] pidReg) (Op.constNat BMc))
        (Op.arange Mv)) s
      = some (Tile.vec (fun i : Fin Mv => pm * BMc + i.val)) := by
  rw [evalR_offsc]
  exact offscm_eval s Mv BMc pm pidReg hpm

/-- The `c_ptrs` pointer op (general `stride_cm`/`stride_cn`) is cast-free. -/
private theorem evalR_cptrsAT (R : RoundingModel) (Creg : RegionName)
    (BMv BNv scm scn : Nat) (s : BlockState) :
    evalOpR R (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Creg)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarL (Op.constNat scm) (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BMv] "offs_cm")))
          (Op.mul .nat Broadcast.scalarL (Op.constNat scn) (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BNv] "offs_cn"))))) s
      = evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Creg)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarL (Op.constNat scm) (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BMv] "offs_cm")))
          (Op.mul .nat Broadcast.scalarL (Op.constNat scn) (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BNv] "offs_cn"))))) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

/-- The `c_mask` boolean op is cast-free. -/
private theorem evalR_cmaskAT (R : RoundingModel) (Mv Nv BMv BNv : Nat) (s : BlockState) :
    evalOpR R (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BMv] "offs_cm")) (Op.constNat Mv))
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BNv] "offs_cn")) (Op.constNat Nv))) s
      = evalOp (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BMv] "offs_cm")) (Op.constNat Mv))
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BNv] "offs_cn")) (Op.constNat Nv))) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

/-- `c = tl.cast(accumulator, tl.float16)` eval under `R`: rounding-event
site 1, `R.cast` applied lane-wise to the accumulator tile. Proven by direct
`eq_def` unfolding (a `rw` through `evalOpR_castFloat` trips over the
`FloatDType.real.toTileDType` vs `TileDType.real` spelling). -/
private theorem castAcc_evalR (R : RoundingModel) (Mv Nv : Nat) (s : BlockState)
    (zT : Tile .real [Mv, Nv]) (hz : s.regs .real [Mv, Nv] "accumulator" = some zT) :
    evalOpR R (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref .real [Mv, Nv] "accumulator")) s
      = some ⟨fun idx => RoundingModel.cast R FloatDType.real FloatDType.fp16 (zT.data idx)⟩ := by
  have hz' : s.regs FloatDType.real.toTileDType [Mv, Nv] "accumulator" = some zT := hz
  simp only [evalOpR.eq_def, hz']
  rfl

/-! ### The tail under `execR R` -/

set_option maxHeartbeats 4000000 in
/-- **R-postLoop**: from the exact invariant at `numKBlocks` blocks, the
`execR R` activation tail + store tail terminates and writes, at every
**active** output lane, the cell `fp16(R.round .fp16 (matmulSpec …))` — the
ideal activated GEMM value rounded **once** at the fp16 grid (`R.cast` site +
`R.storeValue` site collapsed by `round_idem`); inactive lanes' cells and
every cell not hit by an active lane are untouched (the masked-store frame). -/
private theorem matmul_autotune_postLoopR (R : RoundingModel) (A B C : RegionName)
    (s0 : BlockState)
    (M N BM BN GM sam sak sbk sbn scm scn BLOCK_K numKBlocks : Nat) (ACTIVATION : Bool)
    (hInj : Function.Injective (cOffset s0 M N BM BN GM scm scn))
    (st : BlockState)
    (hinv : matmulInvariant A B s0 M N BM BN GM sam sak sbk sbn BLOCK_K numKBlocks numKBlocks st) :
    ∃ sfin, stepStmtsR R
        (matmulActStmt BM BN ACTIVATION :: matmulStoreTail C M N scm scn BM BN) st
        = some sfin
      ∧ (∀ idx : TileIndex [BM, BN],
          sfin.mem C (cOffset s0 M N BM BN GM scm scn idx)
            = if active s0 M N BM BN GM idx then
                MemCell.of FloatDType.fp16.toTileDType
                  (FloatDType.fp16.ofReal
                    (R.round .fp16
                      (matmulSpec s0 A B M N BM BN GM sam sak sbk sbn BLOCK_K numKBlocks ACTIVATION idx.1 idx.2.1)))
              else st.mem C (cOffset s0 M N BM BN GM scm scn idx))
      ∧ (∀ r o, (r ≠ C ∨ ∀ idx : TileIndex [BM, BN],
            active s0 M N BM BN GM idx → o ≠ cOffset s0 M N BM BN GM scm scn idx) →
          sfin.mem r o = st.mem r o) := by
  simp only [matmulInvariant] at hinv
  obtain ⟨hpids, hcle, hpm, hpn, hz, hm, hn, hk, hap, hbp, hundef, hmem⟩ := hinv
  -- run the activation tail (cast-free) via the exact evaluator
  obtain ⟨sa, hsa, hsa_acc, hsa_other, hsa_mem, hsa_pids, hsa_undef⟩ :=
    matmulActStmt_eval st BM BN ACTIVATION
      (fun idx => accPartial s0 A B M N BM BN GM sam sak sbk sbn BLOCK_K idx.1 idx.2.1 numKBlocks)
      hz
  rw [stepStmtsR_cons_some ((matmulActStmt_castFree R BM BN ACTIVATION st).trans hsa)]
  -- registers preserved by the activation tail
  have hpm' : sa.regs .nat [] "pid_m" = some (Tile.scalar (pidM (s0.pids 0) M N BM BN GM)) := by
    rw [hsa_other "pid_m" (by decide)]; exact hpm
  have hpn' : sa.regs .nat [] "pid_n" = some (Tile.scalar (pidN (s0.pids 0) M N BM BN GM)) := by
    rw [hsa_other "pid_n" (by decide)]; exact hpn
  -- the activated accumulator tile and its R-cast
  set zT : Tile .real [BM, BN] :=
    ⟨fun idx : TileIndex [BM, BN] =>
      some (act ACTIVATION (accPartial s0 A B M N BM BN GM sam sak sbk sbn BLOCK_K idx.1 idx.2.1 numKBlocks))⟩
      with hzT
  set cT : Tile .fp16 [BM, BN] :=
    ⟨fun idx => RoundingModel.cast R FloatDType.real FloatDType.fp16 (zT.data idx)⟩ with hcT
  set cpT : Tile .ptr [BM, BN] :=
    ⟨fun idx : TileIndex [BM, BN] => (C.cast, cOffset s0 M N BM BN GM scm scn idx)⟩ with hcpT
  set cmaskT : Tile .bool [BM, BN] :=
    ⟨fun idx : TileIndex [BM, BN] =>
      (decide (rowGlobal s0 M N BM BN GM idx.1 < M) && decide (colGlobal s0 M N BM BN GM idx.2.1 < N))⟩
      with hcmaskT
  unfold matmulStoreTail
  -- c = cast(accumulator, fp16): rounding-event site 1 (`R.cast`)
  erw [stepStmtsR_cons_some (stepStmtR_assign_eq_some
        (show evalOpR R (Op.castFloat FloatDType.real FloatDType.fp16
              (Op.ref .real [BM, BN] "accumulator")) sa = some cT
          from castAcc_evalR R BM BN sa zT hsa_acc))]
  -- offs_cm / offs_cn (cast-free)
  rw [stepStmtsR_cons_some (stepStmtR_assign_eq_some
        (offscm_evalR R _ BM BM (pidM (s0.pids 0) M N BM BN GM) "pid_m" (by simp [hpm'])))]
  rw [stepStmtsR_cons_some (stepStmtR_assign_eq_some
        (offscm_evalR R _ BN BN (pidN (s0.pids 0) M N BM BN GM) "pid_n" (by simp [hpn'])))]
  -- c_ptrs (cast-free)
  rw [stepStmtsR_cons_some (stepStmtR_assign_eq_some
        (show evalOpR R (Op.ptrAdd Broadcast.scalarL (Op.ptrBase C)
            (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
              (Op.mul .nat Broadcast.scalarL (Op.constNat scm) (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_cm")))
              (Op.mul .nat Broadcast.scalarL (Op.constNat scn) (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_cn"))))) _
          = some cpT from by
          rw [evalR_cptrsAT,
            cptrs_eval _ C BM BN scm scn (fun i => rowGlobal s0 M N BM BN GM i) (fun j => colGlobal s0 M N BM BN GM j)
              (by simp [rowGlobal]) (by simp [colGlobal])]
          rfl))]
  -- c_mask (cast-free)
  rw [stepStmtsR_cons_some (stepStmtR_assign_eq_some
        (show evalOpR R (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
            (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_cm")) (Op.constNat M))
            (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_cn")) (Op.constNat N))) _
          = some cmaskT from by
          rw [evalR_cmaskAT,
            cmask_eval _ M N BM BN (fun i => rowGlobal s0 M N BM BN GM i) (fun j => colGlobal s0 M N BM BN GM j)
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
  -- the masked fp16 store: rounding-event site 2 (`writeMemAsR`)
  have hstore : stepStmtR R (Stmt.store .fp16 [BM, BN] (.ptr (Op.ref .ptr [BM, BN] "c_ptrs"))
        (Op.ref .fp16 [BM, BN] "c") (.mask (Op.ref .bool [BM, BN] "c_mask"))) st4
      = some ((TileShape.allIndices [BM, BN]).foldl
          (fun acc i =>
            if cmaskT.data i then
              acc.writeMemAsR R .fp16 C (cOffset s0 M N BM BN GM scm scn i) (cT.data i)
            else acc) st4) := by
    simp only [stepStmtR]
    rw [show evalOpR R (Op.ref .fp16 [BM, BN] "c") st4 = some cT from by rw [evalOpR_ref, hc4]]
    rw [show evalOpR R (Op.ref .bool [BM, BN] "c_mask") st4 = some cmaskT from by rw [evalOpR_ref, hcm4]]
    rw [show evalOpR R (Op.ref .ptr [BM, BN] "c_ptrs") st4 = some cpT from by rw [evalOpR_ref, hcp4]]
    simp only [bind, Option.map_some, Option.bind_some]
    refine congrArg some (List.foldl_ext _ _ _ (fun acc i _ => ?_))
    by_cases hmask : cmaskT.data i
    · simp only [hmask, if_true, hcpT, Region.cast_id, BlockState.writeMemTypedR_fp16]
    · simp only [hmask, Bool.false_eq_true, if_false]
  rw [stepStmtsR_cons_some hstore, stepStmtsR_nil]
  refine ⟨_, rfl, ?_, ?_⟩
  · intro idx
    rw [BlockState.scatter_memcell_R_prop_masked_nd R .fp16 (region := C) st4
          (cOffset s0 M N BM BN GM scm scn) (fun i => cT.data i)
          (fun i => cmaskT.data i = Bool.true) hInj idx]
    by_cases hact : active s0 M N BM BN GM idx
    · have hmasktrue : cmaskT.data idx = Bool.true := by
        simp only [hcmaskT]
        obtain ⟨hr, hcc⟩ := hact
        simp [hr, hcc]
      rw [if_pos hmasktrue, if_pos hact]
      have hdata : cT.data idx = RoundingModel.cast R FloatDType.real FloatDType.fp16
          (some (act ACTIVATION (accPartial s0 A B M N BM BN GM sam sak sbk sbn BLOCK_K idx.1 idx.2.1 numKBlocks))) := rfl
      rw [hdata, RoundingModel.storeValue_cast]
      have hspec : matmulSpec s0 A B M N BM BN GM sam sak sbk sbn BLOCK_K numKBlocks ACTIVATION idx.1 idx.2.1
          = act ACTIVATION (accPartial s0 A B M N BM BN GM sam sak sbk sbn BLOCK_K idx.1 idx.2.1 numKBlocks) := by
        unfold matmulSpec accPartial
        rw [Nat.mul_comm]
      rw [hspec]
    · have hmaskfalse : ¬ (cmaskT.data idx = Bool.true) := by
        simp only [hcmaskT, Bool.and_eq_true, decide_eq_true_eq, not_and]
        intro hr hcc
        exact hact ⟨hr, hcc⟩
      rw [if_neg hmaskfalse, if_neg hact, hmem4]
  · intro r o hcond
    by_cases hr : r = C
    · subst hr
      have hno : ∀ idx : TileIndex [BM, BN],
          active s0 M N BM BN GM idx → o ≠ cOffset s0 M N BM BN GM scm scn idx := by
        rcases hcond with h | h
        · exact absurd rfl h
        · exact h
      rw [BlockState.foldl_writeMemAsR_preserve_masked_prop R .fp16
            (cOffset s0 M N BM BN GM scm scn) (fun i => cT.data i)
            (fun i => cmaskT.data i = Bool.true) o (TileShape.allIndices [BM, BN])
            (fun k _ hk => by
              have hkact : active s0 M N BM BN GM k := by
                simp only [hcmaskT, Bool.and_eq_true, decide_eq_true_eq] at hk
                exact ⟨hk.1, hk.2⟩
              exact fun heq => hno k hkact heq.symm) st4, hmem4]
    · rw [BlockState.foldl_writeMemAsR_preserve_other_region R .fp16
            (cOffset s0 M N BM BN GM scm scn) (fun i => cT.data i)
            (fun i => cmaskT.data i = Bool.true) r hr o (TileShape.allIndices [BM, BN]) st4,
          hmem4]

/-! ### Safety-walk invariant (weak shape half of `matmulInvariant`) -/

/-- Safety-walk loop invariant: the *shape* half of `matmulInvariant`
(`pid_m`/`pid_n` seeded from the L2 schedule, *some* accumulator tile,
`offs_k`, and the exact `a_ptrs`/`b_ptrs` address shapes) with no
`undef`/`mem`/value pins. Needed because the `⊨[R]` skin's `hts` obligation
quantifies over arbitrary launch states, so the safety walk cannot assume
the clean-`undef` precondition that `preLoop`'s full invariant needs.
`offs_k` is carried because the loop body's **masked** loads must evaluate
their `offs_k`-based masks to step at all. -/
private def matmulAutotuneSafeInv (A B : RegionName) (s0 : BlockState)
    (M N BM BN GM sam sak sbk sbn BK T : Nat) (c : Nat) (s : BlockState) : Prop :=
  c ≤ T ∧
  (s.regs .nat [] "pid_m" = some (Tile.scalar (pidM (s0.pids 0) M N BM BN GM))) ∧
  (s.regs .nat [] "pid_n" = some (Tile.scalar (pidN (s0.pids 0) M N BM BN GM))) ∧
  (∃ zT : Tile .real [BM, BN], s.regs .real [BM, BN] "accumulator" = some zT) ∧
  (s.regs .nat [BK] "offs_k" = some (Tile.vec (fun e : Fin BK => e.val))) ∧
  (s.regs .ptr [BM, BK] "a_ptrs" = some ⟨fun idx : TileIndex [BM, BK] =>
      (A.cast, rowIndex s0 M N BM BN GM idx.1 * sam + idx.2.1.val * sak + c * BK * sak)⟩) ∧
  (s.regs .ptr [BK, BN] "b_ptrs" = some ⟨fun idx : TileIndex [BK, BN] =>
      (B.cast, idx.1.val * sbk + colIndex s0 M N BM BN GM idx.2.1 * sbn + c * BK * sbk)⟩)

set_option maxHeartbeats 1000000 in
/-- Weak `preLoop`: from an **arbitrary** state the prologue steps to a state
satisfying `matmulAutotuneSafeInv … 0` (no clean-`undef` hypothesis; the
value half of `preLoop` is dropped). -/
private theorem matmul_autotune_preLoopW (A B : RegionName) (s : BlockState)
    (M N BM BN BK GM sam sak sbk sbn T : Nat) :
    ∃ s', stepStmts (matmulAutotunePrologue A B M N BM BN BK GM sam sak sbk sbn) s = some s'
      ∧ matmulAutotuneSafeInv A B s M N BM BN GM sam sak sbk sbn BK T 0 s' := by
  obtain ⟨s12, h12, hpids, hpm, hpn, hm, hn, hk, huf, hmem⟩ := preLoop_scalars s M N BM BN BK GM
  rw [show matmulAutotunePrologue A B M N BM BN BK GM sam sak sbk sbn
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
  refine ⟨Nat.zero_le T, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp [hpm]
  · simp [hpn]
  · refine ⟨⟨fun _ => some (0 : ℝ)⟩, ?_⟩
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
      String.reduceEq, not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
  · simp [hk]
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx <;> simp [Nat.zero_mul]
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx <;> simp [Nat.zero_mul]

set_option maxHeartbeats 2000000 in
/-- Weak step lemma: one body iteration from `matmulAutotuneSafeInv c` steps
successfully (exact stepper; the body is cast-free, and at exact-multiple
`K` the load masks evaluate all-true) and re-establishes the invariant at
`c + 1` — the shape half of `matmul_step`, valid from arbitrary launch
states. -/
private theorem matmul_autotune_stepW (A B : RegionName) (s0 : BlockState)
    (M N BM BN GM sam sak sbk sbn BK T K : Nat) (hK : K = BK * T)
    (c : Nat) (s : BlockState) (hclt : c < T)
    (hinv : matmulAutotuneSafeInv A B s0 M N BM BN GM sam sak sbk sbn BK T c s) :
    ∃ s', stepStmts (matmulLoopBody K BM BN BK sak sbk)
        (s.setReg "kk" .nat [] (Tile.scalar c)) = some s'
      ∧ matmulAutotuneSafeInv A B s0 M N BM BN GM sam sak sbk sbn BK T (c + 1) s' := by
  obtain ⟨hcle, hpm, hpn, ⟨zT, hz⟩, hk, hap, hbp⟩ := hinv
  set apT : Tile .ptr [BM, BK] :=
    ⟨fun idx : TileIndex [BM, BK] =>
      (A.cast, rowIndex s0 M N BM BN GM idx.1 * sam + idx.2.1.val * sak + c * BK * sak)⟩ with hapT
  set bpT : Tile .ptr [BK, BN] :=
    ⟨fun idx : TileIndex [BK, BN] =>
      (B.cast, idx.1.val * sbk + colIndex s0 M N BM BN GM idx.2.1 * sbn + c * BK * sbk)⟩ with hbpT
  set sk := s.setReg "kk" .nat [] (Tile.scalar c) with hsk
  have hapk : sk.regs .ptr [BM, BK] "a_ptrs" = some apT := by simp [hsk, hap, hapT]
  have hbpk : sk.regs .ptr [BK, BN] "b_ptrs" = some bpT := by simp [hsk, hbp, hbpT]
  have hzk : sk.regs .real [BM, BN] "accumulator" = some zT := by simp [hsk, hz]
  have hkk : sk.regs .nat [] "kk" = some (Tile.scalar c) := by simp [hsk]
  have hkofk : sk.regs .nat [BK] "offs_k" = some (Tile.vec (fun e : Fin BK => e.val)) := by
    simp [hsk, hk]
  set asub : Tile .real [BM, BK] :=
    ⟨fun idx => some (sk.readMem (apT.data idx).1 (apT.data idx).2)⟩ with hasub
  set bsub : Tile .real [BK, BN] :=
    ⟨fun idx => some (sk.readMem (bpT.data idx).1 (bpT.data idx).2)⟩ with hbsub
  unfold matmulLoopBody
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (load_a_eval sk K BM BK T c apT hK hclt hapk hkofk hkk))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (load_b_eval _ K BK BN T c bpT hK hclt
          (by simp [hbpk]) (by simp [hkofk]) (by simp [hkk])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (accdot_op_eval BM BK BN _ zT asub bsub
          (by simp [hzk, hasub, hbsub, BlockState.setReg_readMem])
          (by simp [hasub, BlockState.setReg_readMem])
          (by simp [hbsub, BlockState.setReg_readMem])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (aptr_adv_eval _ BM BK BK sak apT (by simp [hapk])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (bptr_adv_eval _ BK BN BK sbk bpT (by simp [hbpk])))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  refine ⟨by omega, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp [hsk, hpm]
  · simp [hsk, hpn]
  · refine ⟨Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
      (Tile.dot [] asub bsub) zT, ?_⟩
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
      String.reduceEq, not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
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

/-! ### Activation-tail safety helpers -/

/-- The activation tail is trace-safe at **every** state and **both** gate
values: the `constBool` condition and the single register-only `where`
assign touch no memory. -/
private theorem matmulActStmt_safeR (R : RoundingModel) (bounds : RegionBounds)
    (BM BN : Nat) (ACTIVATION : Bool) (s : BlockState) :
    Stmt.TraceSafeR R bounds (matmulActStmt BM BN ACTIVATION) s := by
  unfold matmulActStmt
  simp only [Stmt.TraceSafeR]
  refine ⟨by simp [Op.SafeAtR.eq_def], ?_⟩
  split
  · split
    · exact Stmt.TraceSafeListR.of_forall _ _ (fun st hst s' => by
        simp only [List.mem_cons, List.not_mem_nil, or_false] at hst
        subst hst
        simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    · trivial
  · trivial

/-- Weak activation-tail step (both gate values, arbitrary accumulator
tile): the `stepStmtR R` step succeeds, non-`accumulator` registers pass
through, and *some* accumulator tile survives — the successor pins the tail
safety walk needs. -/
private theorem matmulActStmt_stepW (R : RoundingModel) (BM BN : Nat) (ACTIVATION : Bool)
    (st : BlockState) (zT : Tile .real [BM, BN])
    (hz : st.regs .real [BM, BN] "accumulator" = some zT) :
    ∃ st', stepStmtR R (matmulActStmt BM BN ACTIVATION) st = some st'
      ∧ (∀ {dt : TileDType} {sh : TileShape} (nm : RegName), nm ≠ "accumulator" →
          st'.regs dt sh nm = st.regs dt sh nm)
      ∧ ∃ zT' : Tile .real [BM, BN], st'.regs .real [BM, BN] "accumulator" = some zT' := by
  rw [matmulActStmt_castFree R BM BN ACTIVATION st]
  unfold matmulActStmt
  rcases ACTIVATION with _ | _
  · refine ⟨st, ?_, fun nm _ => rfl, ⟨zT, hz⟩⟩
    show stepStmt (Stmt.ifThen (Op.constBool Bool.false) _) st = some st
    simp only [stepStmt, evalOp_constBool]
    rfl
  · have heval : ∃ vT : Tile .real [BM, BN], evalOp (Op.where
        (Op.ge ComparableDType.real Broadcast.scalarR (Op.ref .real [BM, BN] "accumulator") (Op.const 0))
        (Op.ref .real [BM, BN] "accumulator")
        (Op.mul .real Broadcast.scalarL (Op.const (1e-2 : ℝ)) (Op.ref .real [BM, BN] "accumulator"))) st
        = some vT := by
      rw [evalOp_where, evalOp_ge, evalOp_mul]
      simp only [evalOp_ref, evalOp_const, hz, Option.bind_eq_bind, Option.bind_some]
      exact ⟨_, rfl⟩
    obtain ⟨vT, hv⟩ := heval
    refine ⟨st.setReg "accumulator" .real [BM, BN] vT, ?_, ?_, ⟨vT, by simp⟩⟩
    · show stepStmt (Stmt.ifThen (Op.constBool Bool.true) _) st
          = some (st.setReg "accumulator" .real [BM, BN] vT)
      simp only [stepStmt, evalOp_constBool, Option.bind_eq_bind, Option.bind_some,
        Tile.scalar_data, if_true, stepStmts, hv]
    · intro dt sh nm hnm
      simp [BlockState.setReg_ne_name, hnm]

set_option maxHeartbeats 1000000 in
/-- Per-iteration `TraceSafeListR` for the K-loop body: the two **masked**
loads' addresses are the invariant's pointer shapes, in bounds at *every*
lane by the all-lane bound groups (active lanes are a subset, so the masks
never need evaluating for safety); the `mask=`/`other=` operands are
register-only ops; the three remaining assigns are unconditionally safe. -/
private theorem matmul_autotune_bodySafeR (R : RoundingModel) (bounds : RegionBounds)
    (A B : RegionName) (s0 : BlockState)
    (M N BM BN GM sam sak sbk sbn BK T K : Nat) (c : Nat) (hc : c < T)
    (sk : BlockState)
    (hap : sk.regs .ptr [BM, BK] "a_ptrs" = some ⟨fun idx : TileIndex [BM, BK] =>
        (A.cast, rowIndex s0 M N BM BN GM idx.1 * sam + idx.2.1.val * sak + c * BK * sak)⟩)
    (hbp : sk.regs .ptr [BK, BN] "b_ptrs" = some ⟨fun idx : TileIndex [BK, BN] =>
        (B.cast, idx.1.val * sbk + colIndex s0 M N BM BN GM idx.2.1 * sbn + c * BK * sbk)⟩)
    (hbA : ∀ (t : Fin T) (j : Fin (BM * BK)),
      ((pidM (s0.pids 0) M N BM BN GM * BM + j.val / BK) % M) * sam
        + (t.val * BK + j.val % BK) * sak < bounds A)
    (hbB : ∀ (t : Fin T) (j : Fin (BK * BN)),
      (t.val * BK + j.val / BN) * sbk
        + ((pidN (s0.pids 0) M N BM BN GM * BN + j.val % BN) % N) * sbn < bounds B) :
    Stmt.TraceSafeListR R bounds (matmulLoopBody K BM BN BK sak sbk) sk := by
  unfold matmulLoopBody
  refine Stmt.TraceSafeListR.cons_intro ?_ ?_
  · -- load a (masked)
    simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, aMaskOp, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
    refine ⟨trivial, by simp, ?_⟩
    intro ptrs hptrs i _
    rw [evalOpR_ref, hap] at hptrs
    obtain rfl := Option.some.inj hptrs
    show rowIndex s0 M N BM BN GM i.1 * sam + i.2.1.val * sak + c * BK * sak
        < bounds (Region.cast A)
    have h' := hbA ⟨c, hc⟩ (Lane2D.encode (i.1, i.2.1, PUnit.unit))
    rw [Lane2D.encode_div, Lane2D.encode_mod] at h'
    simp only [Region.cast_id]
    calc rowIndex s0 M N BM BN GM i.1 * sam + i.2.1.val * sak + c * BK * sak
        = ((pidM (s0.pids 0) M N BM BN GM * BM + i.1.val) % M) * sam
            + (c * BK + i.2.1.val) * sak := by
          unfold rowIndex rowGlobal; ring
      _ < bounds A := h'
  · intro s1 h1
    obtain ⟨v1, -, rfl⟩ := stepStmtR_assign_inv h1
    refine Stmt.TraceSafeListR.cons_intro ?_ ?_
    · -- load b (masked)
      simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, bMaskOp, MaskOpt.ActiveR,
        MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
      refine ⟨trivial, by simp, ?_⟩
      intro ptrs hptrs i _
      rw [evalOpR_ref] at hptrs
      rw [show (sk.setReg "a" .real [BM, BK] v1).regs .ptr [BK, BN] "b_ptrs"
          = some (⟨fun idx : TileIndex [BK, BN] =>
            (B.cast, idx.1.val * sbk + colIndex s0 M N BM BN GM idx.2.1 * sbn + c * BK * sbk)⟩ :
              Tile .ptr [BK, BN]) from by simp [hbp]] at hptrs
      obtain rfl := Option.some.inj hptrs
      show i.1.val * sbk + colIndex s0 M N BM BN GM i.2.1 * sbn + c * BK * sbk
          < bounds (Region.cast B)
      have h' := hbB ⟨c, hc⟩ (Lane2D.encode (i.1, i.2.1, PUnit.unit))
      rw [Lane2D.encode_div, Lane2D.encode_mod] at h'
      simp only [Region.cast_id]
      calc i.1.val * sbk + colIndex s0 M N BM BN GM i.2.1 * sbn + c * BK * sbk
          = (c * BK + i.1.val) * sbk
              + ((pidN (s0.pids 0) M N BM BN GM * BN + i.2.1.val) % N) * sbn := by
            unfold colIndex colGlobal; ring
        _ < bounds B := h'
    · intro s2 h2
      obtain ⟨v2, -, rfl⟩ := stepStmtR_assign_inv h2
      refine Stmt.TraceSafeListR.of_forall _ _ ?_
      intro stx hstx s'
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hstx
      rcases hstx with rfl | rfl | rfl <;>
        simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]

set_option maxHeartbeats 1000000 in
/-- `TraceSafeListR` for the store tail: the five assigns are register-only
(the cast is not a memory event) and the single **masked** `.fp16` store's
active lanes are exactly the `(row<M)&(col<N)` window, so their addresses
are the skin's `writeMask`-gated `write` bounds. -/
private theorem matmul_autotune_tailSafeR (R : RoundingModel) (bounds : RegionBounds)
    (C : RegionName) (s0 : BlockState) (M N scm scn BM BN GM : Nat) (st : BlockState)
    (hpm : st.regs .nat [] "pid_m" = some (Tile.scalar (pidM (s0.pids 0) M N BM BN GM)))
    (hpn : st.regs .nat [] "pid_n" = some (Tile.scalar (pidN (s0.pids 0) M N BM BN GM)))
    (hzE : ∃ zT : Tile .real [BM, BN], st.regs .real [BM, BN] "accumulator" = some zT)
    (hbC : ∀ j : Fin (BM * BN),
      (pidM (s0.pids 0) M N BM BN GM * BM + j.val / BN < M ∧
        pidN (s0.pids 0) M N BM BN GM * BN + j.val % BN < N) →
      scm * (pidM (s0.pids 0) M N BM BN GM * BM + j.val / BN)
        + scn * (pidN (s0.pids 0) M N BM BN GM * BN + j.val % BN) < bounds C) :
    Stmt.TraceSafeListR R bounds (matmulStoreTail C M N scm scn BM BN) st := by
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
  rw [offscm_evalR R _ BM BM (pidM (s0.pids 0) M N BM BN GM) "pid_m" (by simp [hpm])] at hv2
  obtain rfl := Option.some.inj hv2
  -- 3. offs_cn
  refine Stmt.TraceSafeListR.cons_intro (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) ?_
  intro s3 h3
  obtain ⟨v3, hv3, rfl⟩ := stepStmtR_assign_inv h3
  rw [offscm_evalR R _ BN BN (pidN (s0.pids 0) M N BM BN GM) "pid_n" (by simp [hpn])] at hv3
  obtain rfl := Option.some.inj hv3
  -- 4. c_ptrs
  refine Stmt.TraceSafeListR.cons_intro (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) ?_
  intro s4 h4
  obtain ⟨v4, hv4, rfl⟩ := stepStmtR_assign_inv h4
  rw [evalR_cptrsAT,
    cptrs_eval _ C BM BN scm scn
      (fun i => pidM (s0.pids 0) M N BM BN GM * BM + i.val)
      (fun j => pidN (s0.pids 0) M N BM BN GM * BN + j.val) (by simp) (by simp)] at hv4
  obtain rfl := Option.some.inj hv4
  -- 5. c_mask
  refine Stmt.TraceSafeListR.cons_intro (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) ?_
  intro s5 h5
  obtain ⟨v5, hv5, rfl⟩ := stepStmtR_assign_inv h5
  rw [evalR_cmaskAT,
    cmask_eval _ M N BM BN
      (fun i => pidM (s0.pids 0) M N BM BN GM * BM + i.val)
      (fun j => pidN (s0.pids 0) M N BM BN GM * BN + j.val) (by simp) (by simp)] at hv5
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
  simp only [Bool.and_eq_true, decide_eq_true_eq] at hmi
  show scm * (pidM (s0.pids 0) M N BM BN GM * BM + i.1.val)
      + scn * (pidN (s0.pids 0) M N BM BN GM * BN + i.2.1.val) < bounds (Region.cast C)
  have h' := hbC (Lane2D.encode (i.1, i.2.1, PUnit.unit))
    (by rw [Lane2D.encode_div, Lane2D.encode_mod]; exact hmi)
  rw [Lane2D.encode_div, Lane2D.encode_mod] at h'
  simpa only [Region.cast_id] using h'

set_option maxHeartbeats 1000000 in
/-- **The `TraceSafeR` walk for the whole kernel**, driven by
`Stmt.forRangeTraceSafeR_inv` over the weak `matmulAutotuneSafeInv`. The
three bound groups are the skin's `read1`/`read2` windows (widened to all
lanes via `hK`, since the load masks are all-true at exact-multiple `K`) and
the `writeMask`-gated `write` window. -/
private theorem matmul_autotune_traceSafeR (R : RoundingModel) (bounds : RegionBounds)
    (A B C : RegionName)
    (M N K sam sak sbk sbn scm scn BM BN BK GM numKBlocks : Nat) (ACTIVATION : Bool)
    (hK : K = BK * numKBlocks) (s : BlockState)
    (hbA : ∀ (t : Fin numKBlocks) (j : Fin (BM * BK)),
      ((pidM (s.pids 0) M N BM BN GM * BM + j.val / BK) % M) * sam
        + (t.val * BK + j.val % BK) * sak < bounds A)
    (hbB : ∀ (t : Fin numKBlocks) (j : Fin (BK * BN)),
      (t.val * BK + j.val / BN) * sbk
        + ((pidN (s.pids 0) M N BM BN GM * BN + j.val % BN) % N) * sbn < bounds B)
    (hbC : ∀ j : Fin (BM * BN),
      (pidM (s.pids 0) M N BM BN GM * BM + j.val / BN < M ∧
        pidN (s.pids 0) M N BM BN GM * BN + j.val % BN < N) →
      scm * (pidM (s.pids 0) M N BM BN GM * BM + j.val / BN)
        + scn * (pidN (s.pids 0) M N BM BN GM * BN + j.val % BN) < bounds C) :
    ((matmul_autotune_surface A B C M N K sam sak sbk sbn scm scn BM BN BK GM
        numKBlocks ACTIVATION).toAlgKernel).TraceSafeR R bounds s := by
  unfold Kernel.TraceSafeR
  rw [matmul_autotune_body_split' A B C M N K sam sak sbk sbn scm scn BM BN BK GM
    numKBlocks ACTIVATION]
  have hstep : ∀ c s', c < numKBlocks →
      matmulAutotuneSafeInv A B s M N BM BN GM sam sak sbk sbn BK numKBlocks c s' →
      Stmt.TraceSafeListR R bounds (matmulLoopBody K BM BN BK sak sbk)
        (s'.setReg "kk" .nat [] (Tile.scalar c)) ∧
      ∃ s'', stepStmtsR R (matmulLoopBody K BM BN BK sak sbk)
          (s'.setReg "kk" .nat [] (Tile.scalar c)) = some s'' ∧
        matmulAutotuneSafeInv A B s M N BM BN GM sam sak sbk sbn BK numKBlocks (c + 1) s'' := by
    intro c s' hcx hP
    obtain ⟨hcle, hpm, hpn, hzE, hkx, hapx, hbpx⟩ := hP
    refine ⟨matmul_autotune_bodySafeR R bounds A B s M N BM BN GM sam sak sbk sbn BK
        numKBlocks K c hcx _ (by simp [hapx]) (by simp [hbpx]) hbA hbB, ?_⟩
    obtain ⟨s'', hs'', hP''⟩ := matmul_autotune_stepW A B s M N BM BN GM sam sak sbk sbn BK
      numKBlocks K hK c s' hcx ⟨hcle, hpm, hpn, hzE, hkx, hapx, hbpx⟩
    exact ⟨s'', by rw [matmulAutotuneBody_castFree]; exact hs'', hP''⟩
  refine Stmt.TraceSafeListR.append_intro _ _ ?_ ?_
  · -- prologue: register-only assigns, safe at every state
    refine Stmt.TraceSafeListR.of_forall _ _ ?_
    intro st hst s'
    simp only [matmulAutotunePrologue, List.mem_cons, List.not_mem_nil, or_false] at hst
    rcases hst with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
      simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]
  · intro s1 hs1
    obtain ⟨s1x, hpre, hP0⟩ :=
      matmul_autotune_preLoopW A B s M N BM BN BK GM sam sak sbk sbn numKBlocks
    rw [matmulAutotunePrologue_castFree R A B M N BM BN BK GM sam sak sbk sbn s, hpre] at hs1
    obtain rfl := Option.some.inj hs1
    refine Stmt.TraceSafeListR.cons_intro ?_ ?_
    · -- the K-loop is trace-safe (invariant principle over the weak invariant)
      simp only [Stmt.TraceSafeR]
      exact Stmt.forRangeTraceSafeR_inv R bounds "kk" numKBlocks 1
        (matmulLoopBody K BM BN BK sak sbk)
        (matmulAutotuneSafeInv A B s M N BM BN GM sam sak sbk sbn BK numKBlocks)
        hstep 0 s1x hP0
    · intro s2 hs2
      obtain ⟨final, sLoop, hLoopStmt, hfinal, hPL⟩ :=
        forRange_inv (idx := "kk") (start := 0) (stop := numKBlocks) (step := 1)
          (body := matmulLoopBody K BM BN BK sak sbk)
          (by omega) hP0
          (fun c st hlt hinv => matmul_autotune_stepW A B s M N BM BN GM sam sak sbk sbn BK
            numKBlocks K hK c st hlt hinv)
      rw [stepStmtR_forRange,
        stepForRangeAuxR_castFree R _ (matmulAutotuneBody_castFree R K BM BN BK sak sbk) "kk",
        ← stepForRangeAux.forRange_unfold, hLoopStmt] at hs2
      obtain rfl := Option.some.inj hs2
      obtain ⟨-, hpmL, hpnL, hzL, -, -, -⟩ := hPL
      refine Stmt.TraceSafeListR.cons_intro
        (matmulActStmt_safeR R bounds BM BN ACTIVATION sLoop) ?_
      intro s3 hs3
      obtain ⟨zTL, hzTL⟩ := hzL
      obtain ⟨s3', hstep3, hother3, hz3⟩ := matmulActStmt_stepW R BM BN ACTIVATION sLoop zTL hzTL
      rw [hstep3] at hs3
      obtain rfl := Option.some.inj hs3
      exact matmul_autotune_tailSafeR R bounds C s M N scm scn BM BN GM s3'
        (by rw [hother3 "pid_m" (by decide)]; exact hpmL)
        (by rw [hother3 "pid_n" (by decide)]; exact hpnL)
        hz3 hbC

/-- The full autotuned surface (masked loads, `if` activation tail, masked
store) sits inside the flat-memory bridge's covered fragment (`FlattenOk`). -/
theorem matmul_autotune_flattenOk (A B C : RegionName)
    (M N K sam sak sbk sbn scm scn BM BN BK GM numKBlocks : Nat) (ACTIVATION : Bool) :
    ((matmul_autotune_surface A B C M N K sam sak sbk sbn scm scn BM BN BK GM
        numKBlocks ACTIVATION).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  rw [matmul_autotune_body_split' A B C M N K sam sak sbk sbn scm scn BM BN BK GM
    numKBlocks ACTIVATION]
  simp [matmulAutotunePrologue, matmulLoopBody, matmulActStmt, matmulStoreTail,
    aMaskOp, bMaskOp, StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

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

/-- Under `K = BK · numKBlocks`, every in-tile inner key is inside the
kernel's `offs_k < K - kk·BK` load window: `t·BK + e < K` for `e < BK`. -/
private theorem stream_mask_lt (K BK numKBlocks : Nat) (hK : K = BK * numKBlocks)
    (t : Fin numKBlocks) {e : Nat} (he : e < BK) : t.val * BK + e < K := by
  have h1 : (t.val + 1) * BK ≤ numKBlocks * BK :=
    Nat.mul_le_mul_right BK (Nat.succ_le_of_lt t.isLt)
  have h2 : t.val * BK + e < (t.val + 1) * BK := by
    rw [Nat.succ_mul]; omega
  calc t.val * BK + e < (t.val + 1) * BK := h2
    _ ≤ numKBlocks * BK := h1
    _ = BK * numKBlocks := Nat.mul_comm _ _
    _ = K := hK.symm

/-- **Streaming IO signature** of `matmul_triton_autotune` on the two-stream
fold skin (S1: fold + terminal store). Step `t` of the K-loop reads the
`[BM, BLOCK_K]` `A`-tile and the `[BLOCK_K, BN]` `B`-tile; after the loop
one `[BM, BN]` output tile is stored at the **fp16** grid
(`outDType := .fp16` — the kernel's `.to(tl.float16)` + fp16 store). The
kernel schedules on a **single** linear `pid` (`program_id(0)`; the skin's
`pid₁` slot is unused), so every window derives `(pid_m, pid_n)` through the
transcribed L2-grouping arithmetic `pidM`/`pidN`:

* `read1` lane `l = (i, e)` (row-major over `[BM, BLOCK_K]`), step `t`:
  `((pid_m·BM + i) % M)·sam + (t·BK + e)·sak` — the invariant's `a_ptrs`
  cell after `t` advances.
* `read2` lane `l = (e, j)` (row-major over `[BLOCK_K, BN]`), step `t`:
  `(t·BK + e)·sbk + ((pid_n·BN + j) % N)·sbn` — the `b_ptrs` cell.
* `write` lane `l = (i, j)`: `scm·(pid_m·BM + i) + scn·(pid_n·BN + j)` — the
  kernel's un-wrapped `c_ptrs` (= `cOffset` in pid form).
* `mask1`/`mask2` transcribe the loads' `offs_k < K - kk·BK` windows in the
  per-lane spelling `t·BK + e < K`; `writeMask` transcribes the store's
  `(row<M) & (col<N)` boundary mask verbatim.

The constexpr `ACTIVATION` gate is the `Bool` parameter, kept symbolic. -/
def matmulAutotuneIO (A B C : RegionName)
    (M N K sam sak sbk sbn scm scn BM BN BK GM numKBlocks : Nat) (ACTIVATION : Bool) :
    StreamMasked2DKernelIO₂ where
  kernel := matmul_autotune_surface A B C M N K sam sak sbk sbn scm scn BM BN BK GM
    numKBlocks ACTIVATION
  inp1 := A
  inp2 := B
  out := C
  T := numKBlocks
  B1 := BM * BK
  B2 := BK * BN
  C := BM * BN
  outDType := .fp16
  read1 := fun p₀ _ t l =>
    ((pidM p₀ M N BM BN GM * BM + l.val / BK) % M) * sam + (t.val * BK + l.val % BK) * sak
  read2 := fun p₀ _ t l =>
    (t.val * BK + l.val / BN) * sbk + ((pidN p₀ M N BM BN GM * BN + l.val % BN) % N) * sbn
  write := fun p₀ _ l =>
    scm * (pidM p₀ M N BM BN GM * BM + l.val / BN) + scn * (pidN p₀ M N BM BN GM * BN + l.val % BN)
  mask1 := fun _ _ t l => t.val * BK + l.val % BK < K
  mask2 := fun _ _ t l => t.val * BK + l.val / BN < K
  writeMask := fun p₀ _ l =>
    pidM p₀ M N BM BN GM * BM + l.val / BN < M ∧ pidN p₀ M N BM BN GM * BN + l.val % BN < N

/-- Under the two stream pins, `matmulSpec` at the decoded output lane **is**
`act ACTIVATION` of the skin-level double fold `∑ t, ∑ e, xs · ys`
(`gemmSum_blocks` + address-identity of the windows with the invariant's
pointer shapes; the pins' `mask1`/`mask2` guards are discharged by
`stream_mask_lt` from `hK`). -/
private theorem matmulSpec_eq_streamSum (A B : RegionName) (s₀ : BlockState)
    (M N BM BN GM sam sak sbk sbn BK numKBlocks K : Nat) (ACTIVATION : Bool)
    (hK : K = BK * numKBlocks)
    (xs : Fin numKBlocks → Fin (BM * BK) → ℝ) (ys : Fin numKBlocks → Fin (BK * BN) → ℝ)
    (hx : ∀ (t : Fin numKBlocks) (j : Fin (BM * BK)), t.val * BK + j.val % BK < K →
      s₀.readMem A (((pidM (s₀.pids 0) M N BM BN GM * BM + j.val / BK) % M) * sam
          + (t.val * BK + j.val % BK) * sak)
        = xs t j)
    (hy : ∀ (t : Fin numKBlocks) (j : Fin (BK * BN)), t.val * BK + j.val / BN < K →
      s₀.readMem B ((t.val * BK + j.val / BN) * sbk
          + ((pidN (s₀.pids 0) M N BM BN GM * BN + j.val % BN) % N) * sbn)
        = ys t j)
    (l : Fin (BM * BN)) :
    matmulSpec s₀ A B M N BM BN GM sam sak sbk sbn BK numKBlocks ACTIVATION
        (Lane2D.decode l).1 (Lane2D.decode l).2.1
      = act ACTIVATION (∑ t : Fin numKBlocks, ∑ e : Fin BK,
          xs t (aLane BM BN BK l e) * ys t (bLane BM BN BK l e)) := by
  unfold matmulSpec
  refine congrArg (act ACTIVATION) ?_
  rw [Nat.mul_comm BK numKBlocks, gemmSum_blocks]
  refine Finset.sum_congr rfl fun t _ => Finset.sum_congr rfl fun e _ => ?_
  have hxa : aElem s₀ A M N BM BN GM sam sak (Lane2D.decode l).1 (t.val * BK + e.val)
      = xs t (aLane BM BN BK l e) := by
    rw [← hx t (aLane BM BN BK l e)
        (by simpa [aLane, Lane2D.encode_mod] using stream_mask_lt K BK numKBlocks hK t e.isLt)]
    simp only [aLane, Lane2D.encode_div, Lane2D.encode_mod]
    rfl
  have hyb : bElem s₀ B M N BM BN GM sbk sbn (Lane2D.decode l).2.1 (t.val * BK + e.val)
      = ys t (bLane BM BN BK l e) := by
    rw [← hy t (bLane BM BN BK l e)
        (by simpa [bLane, Lane2D.encode_div] using stream_mask_lt K BK numKBlocks hK t e.isLt)]
    simp only [bLane, Lane2D.encode_div, Lane2D.encode_mod]
    rfl
  rw [hxa, hyb]

/-! ### The headline -/

set_option maxHeartbeats 4000000 in
/-- **The `⊨[R]` streaming headline (wave-5 S1 fold genre).** For every
rounding model `R` and **both** values of the constexpr `ACTIVATION` gate,
the faithful `matmul_triton_autotune` surface implements, on its
`StreamMasked2DKernelIO₂` signature, the **ideal ℝ activated GEMM fold**
over the streamed tiles: output lane `l = (i, j)` holds
`act ACTIVATION (∑ t, ∑ e, A-tile[t](i,e) · B-tile[t](e,j))` — the spec `f`
is exact real arithmetic with the leaky-ReLU branch expressed through the
named `act` brancher (identity when the gate is off), and the single
boundary quantization is carried by the skin's readback contract
(`readMemAs .fp16` holds `fp16.ofReal (R.round .fp16 (f …))`), where the
kernel's two rounding events (the `.to(tl.float16)` cast and the fp16-typed
masked store) collapse to one `R.round .fp16` by the defining `round_idem`.

Layer map: the prologue, the whole K-loop (masked loads, `other=0.0`), and
the activation tail are cast-free, so under `execR R` they collapse verbatim
onto the exact stepper and the proven `preLoop` / `matmul_step` /
`forRange_inv` / `matmulActStmt_eval` stack above is reused unchanged; only
the 6-statement store tail is re-proved on the `R` side
(`matmul_autotune_postLoopR`).

Every hypothesis is truth-forced:

* `hK : K = BK · numKBlocks` — the exact-multiple contraction presentation
  shared with the exact surface: it makes the trip count
  `cdiv(K, BLOCK_K) = numKBlocks` exact and the per-step
  `offs_k < K - kk·BK` load masks all-true, which is what lets the streamed
  pins cover every lane the `tl.dot` consumes (at non-multiple `K` the spec
  sum would over-count the tail block and the statement would be false).
* `hcn : scn = 1` and `hBN : BN ≤ scm` — output-offset injectivity
  (`rowMajor2D_inj`): the column stride is the unit stride and the
  column-block width fits the row stride, so distinct output lanes hit
  distinct addresses; with colliding lanes the per-lane readback would be
  last-writer-wins and the statement false. Both hold for every valid
  row-major tiling.

Relation to the exact surface: the exact headline
`matmul_autotune_closed_form_correct` (`Realizes_without_Rounding`) above is
retained unchanged; this `⊨[R]` face strictly generalizes its content — at
`R := .triv` the readback contract degenerates to the exact fp16-cast cell
of the same activated GEMM value. Both faces are kept per the
rounding-as-default doctrine. -/
specification matmul_autotune_io_correctness (R : RoundingModel)
    (A B C : RegionName)
    (M N K sam sak sbk sbn scm scn BM BN BK GM numKBlocks : Nat) (ACTIVATION : Bool)
    (hK : K = BK * numKBlocks) (hcn : scn = 1) (hBN : BN ≤ scm) :
    matmulAutotuneIO A B C M N K sam sak sbk sbn scm scn BM BN BK GM numKBlocks ACTIVATION
      ⊨[R] fun _ _ xs ys l =>
        act ACTIVATION (∑ t : Fin numKBlocks, ∑ e : Fin BK,
          xs t (aLane BM BN BK l e) * ys t (bLane BM BN BK l e)) := by
  subst hcn
  refine StreamMasked2DKernelIO₂.ImplementsR.intro _ ?_ ?_ ?_
  · exact matmul_autotune_flattenOk A B C M N K sam sak sbk sbn scm 1 BM BN BK GM
      numKBlocks ACTIVATION
  · -- safety walk
    intro bounds s xs ys _hx _hy hbr1 hbr2 hbw
    simp only [matmulAutotuneIO] at hbr1 hbr2 hbw ⊢
    refine matmul_autotune_traceSafeR R bounds A B C M N K sam sak sbk sbn scm 1 BM BN BK GM
      numKBlocks ACTIVATION hK s ?_ ?_ ?_
    · intro t j
      have hBK : 0 < BK := by
        rcases Nat.eq_zero_or_pos BK with h | h
        · exact absurd j.pos (by simp [h])
        · exact h
      exact hbr1 t j (stream_mask_lt K BK numKBlocks hK t (Nat.mod_lt _ hBK))
    · intro t j
      have hdiv : j.val / BN < BK :=
        Nat.div_lt_of_lt_mul (lt_of_lt_of_eq j.isLt (Nat.mul_comm BK BN))
      exact hbr2 t j (stream_mask_lt K BK numKBlocks hK t hdiv)
    · intro j hj
      exact hbw j hj
  · -- the rounded Hoare triple
    intro s₀ xs ys hundef hx hy
    simp only [matmulAutotuneIO] at hx hy ⊢
    have hundef' : ∀ rg o, s₀.undef rg o = 0 := fun rg o => by rw [hundef]
    have hInj : Function.Injective (cOffset s₀ M N BM BN GM scm 1) := by
      have heq : cOffset s₀ M N BM BN GM scm 1
          = fun idx : TileIndex [BM, BN] =>
              (scm * (pidM (s₀.pids 0) M N BM BN GM * BM) + pidN (s₀.pids 0) M N BM BN GM * BN)
                + idx.1.val * scm + idx.2.1.val := by
        funext idx; simp only [cOffset, rowGlobal, colGlobal]; ring
      rw [heq]; exact rowMajor2D_inj _ scm hBN
    -- exact preLoop + K-loop (cast-free, so they are the `execR` run too)
    obtain ⟨s1, hpre, hP0⟩ := preLoop A B C s₀ M N BM BN BK sam sak sbk sbn scm 1 GM
      numKBlocks K ACTIVATION hundef'
    obtain ⟨final, sLoop, hLoopStmt, hfinal, hPLoop⟩ :=
      forRange_inv (idx := "kk") (start := 0) (stop := numKBlocks) (step := 1)
        (by omega) hP0
        (fun c stx hlt hinv => by
          simpa using matmul_step A B s₀ M N BM BN GM sam sak sbk sbn BK numKBlocks K hK
            c stx hlt hinv)
    have hfinalEq : final = numKBlocks := by
      have hle : final ≤ numKBlocks := by
        simp only [matmulInvariant] at hPLoop
        exact hPLoop.2.1
      exact le_antisymm hle hfinal
    rw [hfinalEq] at hPLoop
    have hmem0 : sLoop.mem = s₀.mem := by
      have h := hPLoop
      simp only [matmulInvariant] at h
      exact h.2.2.2.2.2.2.2.2.2.2.2
    -- R-side activation + store tail
    obtain ⟨sfin, hTailR, hval, hframe⟩ :=
      matmul_autotune_postLoopR R A B C s₀ M N BM BN GM sam sak sbk sbn scm 1 BK
        numKBlocks ACTIVATION hInj sLoop hPLoop
    have hLoopR : stepStmtR R (Stmt.forRange "kk" 0 numKBlocks 1
          (matmulLoopBody K BM BN BK sak sbk)) s1
        = some sLoop := by
      rw [stepStmtR_forRange,
        stepForRangeAuxR_castFree R _ (matmulAutotuneBody_castFree R K BM BN BK sak sbk) "kk",
        ← stepForRangeAux.forRange_unfold]
      exact hLoopStmt
    have hpre' : stepStmts (matmulAutotunePrologue A B M N BM BN BK GM sam sak sbk sbn) s₀
        = some s1 := by
      rw [← matmul_autotune_take15_eq A B C M N K sam sak sbk sbn scm 1 BM BN BK GM
        numKBlocks ACTIVATION]
      exact hpre
    refine ⟨sfin, ?_, ?_, ?_⟩
    · show execR R (matmul_autotune_surface A B C M N K sam sak sbk sbn scm 1 BM BN BK GM
          numKBlocks ACTIVATION).toAlgKernel s₀ = some sfin
      unfold execR
      rw [matmul_autotune_body_split' A B C M N K sam sak sbk sbn scm 1 BM BN BK GM
          numKBlocks ACTIVATION,
        stepStmtsR_append,
        matmulAutotunePrologue_castFree R A B M N BM BN BK GM sam sak sbk sbn s₀, hpre',
        Option.bind_some, stepStmtsR_cons_some hLoopR]
      exact hTailR
    · intro l hj
      have hact : active s₀ M N BM BN GM (Lane2D.decode l) := hj
      have hcell := hval (Lane2D.decode l)
      rw [if_pos hact] at hcell
      have haddr : scm * (pidM (s₀.pids 0) M N BM BN GM * BM + l.val / BN)
            + 1 * (pidN (s₀.pids 0) M N BM BN GM * BN + l.val % BN)
          = cOffset s₀ M N BM BN GM scm 1 (Lane2D.decode l) := rfl
      rw [haddr, BlockState.readMemAs_fp16_of_cell hcell,
        matmulSpec_eq_streamSum A B s₀ M N BM BN GM sam sak sbk sbn BK numKBlocks K
          ACTIVATION hK xs ys hx hy l]
    · intro r o hcond
      have hcond' : r ≠ C ∨ ∀ idx : TileIndex [BM, BN],
          active s₀ M N BM BN GM idx → o ≠ cOffset s₀ M N BM BN GM scm 1 idx := by
        rcases hcond with hne | hno
        · exact Or.inl hne
        · refine Or.inr fun idx hidx => ?_
          have h := hno (Lane2D.encode idx)
            (by
              rw [Lane2D.encode_div, Lane2D.encode_mod]
              exact hidx)
          rw [Lane2D.encode_div, Lane2D.encode_mod] at h
          exact h
      rw [hframe r o hcond', hmem0]

end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune
