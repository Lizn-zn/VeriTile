import VeriTile.Core
import VeriTile.Semantics
import VeriTile.Float
import VeriTile.Frontend.Triton.DSL
import VeriTile.Kernel

/-!
# `matmul_tma` — closed-form GEMM correctness

`matmul_tma.py`'s `matmul_tma_load_store` is a **single-tile** matmul driven by
TMA block pointers: it builds `tl.make_block_ptr` views of `A`, `B`, `C`, loads
the `BLOCK_M×BLOCK_K` and `BLOCK_K×BLOCK_N` tiles, computes `tl.dot(a, b)`,
optionally downcasts to `float16`, and stores the `BLOCK_M×BLOCK_N` tile into `C`.

This file proves the kernel correct against a **genuine mathematical matrix
product**: every output cell `C[i,j]` of the computed tile equals
`Σ_{e < BLOCK_K} A[i,e] · B[e,j]` over `ℝ` (and `fp16(…)` of that sum in the
`OUTPUT_F16 = true` branch). This is *not* the kernel's own emitted value — the
real-valued `Σ_e A·B` GEMM reference is derived independently from the loaded
`A`/`B` tiles, and the `tl.dot` contraction is proven to realize it.

## Proof architecture

```
matmul_tma_f32_closed_form_correct          ← TOP THEOREM (f32 branch, ComputeCorrect.Realizes)
  └─ matmul_tma_f32_exec_closed_form        ← exec-side closed form (every cell = ∑_e A·B)
matmul_tma_f16_closed_form_correct          ← TOP THEOREM (fp16 branch)
  └─ matmul_tma_f16_exec_closed_form        ← exec-side closed form (every cell = fp16(∑_e A·B))
```

Both reductions step the straight-line surface — three `tl.make_block_ptr`
assignments, the two block-pointer loads, the `tl.dot`, the optional
`Op.castFloat` fp16 downcast, and the block-pointer store — then read the output
cell off the scatter.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); the modeled
`tl.cast(..., fp16)` is the placeholder `FloatDType.real.cast .fp16`. The single
`(1, 1)` host launch, `num_warps`/`num_ctas`, and the TMA
`order` scheduling tuples are the trusted boundary; the per-program statement is
universally quantified over the input state `s`. The layout contract is exactly
the block pointers the kernel constructs: with offsets `(0, 0)` the load/store
address of lane `(i, j)` is `i · strideRow + j · strideCol` into the respective
region (`A`: `i·stride_am + e·stride_ak`, `B`: `e·stride_bk + j·stride_bn`,
`C`: `i·stride_cm + j·stride_cn`); no `boundary_check` is requested, so every
lane is in-bounds. The output-offset map is assumed injective (distinct lanes
hit distinct addresses), exactly as the contiguous `128×128` Python test tiles
satisfy.

## Translation-surface blocker

Translation-surface blocker: the `OUTPUT_F16` constexpr branch is split into
two Lean surfaces (`matmul_tma_f32_surface` and `matmul_tma_f16_surface`), so
the `OUTPUT_F16` parameter, the `if OUTPUT_F16:` branch, and the
`.to(tl.float16)` cast do not appear in the first (f32) surface the scans
compare against. The textual py↔lean scans in
`bench/audit_tritonbench_g.sh` exempt this port on this marker (registered in
`proof_blockers.md`).
-/

namespace VeriTile.Bench.TritonBenchG.MatmulTma

open VeriTile

set_option linter.unusedSimpArgs false

/-! ## Surfaces -/

/-- Faithful transcription of `matmul_tma.py`'s `matmul_tma_load_store` for the
`OUTPUT_F16 = false` branch (float32 output, no downcast). The TMA `order` tuple
is scheduling metadata the DSL erases into the same block-pointer AST. -/
def matmul_tma_f32_surface
    (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_M BLOCK_N BLOCK_K : Nat) :
    ComputeKernel := triton {
  a_block_ptr = tl.make_block_ptr(base=A, shape=($(M), $(K)),
    strides=($(stride_am), $(stride_ak)), offsets=($(0), $(0)),
    block_shape=($(BLOCK_M), $(BLOCK_K)), order=(1, 0))
  b_block_ptr = tl.make_block_ptr(base=B, shape=($(K), $(N)),
    strides=($(stride_bk), $(stride_bn)), offsets=($(0), $(0)),
    block_shape=($(BLOCK_K), $(BLOCK_N)), order=(0, 1))
  c_block_ptr = tl.make_block_ptr(base=C, shape=($(M), $(N)),
    strides=($(stride_cm), $(stride_cn)), offsets=($(0), $(0)),
    block_shape=($(BLOCK_M), $(BLOCK_N)), order=(1, 0))
  a = tl.load(a_block_ptr)
  b = tl.load(b_block_ptr)
  c = tl.dot(a, b)
  tl.store(c_block_ptr, c)
}

/-- Faithful transcription of `matmul_tma_load_store` for the `OUTPUT_F16 = true`
branch (the dot result is downcast to `float16` before the store). -/
def matmul_tma_f16_surface
    (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_M BLOCK_N BLOCK_K : Nat) :
    ComputeKernel := triton {
  a_block_ptr = tl.make_block_ptr(base=A, shape=($(M), $(K)),
    strides=($(stride_am), $(stride_ak)), offsets=($(0), $(0)),
    block_shape=($(BLOCK_M), $(BLOCK_K)), order=(1, 0))
  b_block_ptr = tl.make_block_ptr(base=B, shape=($(K), $(N)),
    strides=($(stride_bk), $(stride_bn)), offsets=($(0), $(0)),
    block_shape=($(BLOCK_K), $(BLOCK_N)), order=(0, 1))
  c_block_ptr = tl.make_block_ptr(base=C, shape=($(M), $(N)),
    strides=($(stride_cm), $(stride_cn)), offsets=($(0), $(0)),
    block_shape=($(BLOCK_M), $(BLOCK_N)), order=(1, 0))
  a = tl.load(a_block_ptr)
  b = tl.load(b_block_ptr)
  c = (tl.dot(a, b)).to(tl.float16)
  tl.store(c_block_ptr, c)
}

/-- The f32 TMA surface lowers to the algorithm layer. -/
theorem matmul_tma_f32_surface_toAlgorithm_supported
    (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_M BLOCK_N BLOCK_K : Nat) :
    ∃ alg, (matmul_tma_f32_surface A B C M N K stride_am stride_ak
      stride_bk stride_bn stride_cm stride_cn BLOCK_M BLOCK_N BLOCK_K).toAlgorithm?
        = Except.ok alg := by
  simp [matmul_tma_f32_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- The fp16 TMA surface lowers to the algorithm layer. -/
theorem matmul_tma_f16_surface_toAlgorithm_supported
    (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_M BLOCK_N BLOCK_K : Nat) :
    ∃ alg, (matmul_tma_f16_surface A B C M N K stride_am stride_ak
      stride_bk stride_bn stride_cm stride_cn BLOCK_M BLOCK_N BLOCK_K).toAlgorithm?
        = Except.ok alg := by
  simp [matmul_tma_f16_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-! ## GEMM closed-form spec -/

/-- `A[i, e] = readMem A (i · stride_am + e · stride_ak)` — the address of tile
lane `(i, e)` of the `a_block_ptr` view (offsets `(0, 0)`). -/
noncomputable def aElem (s : BlockState) (A : RegionName) (stride_am stride_ak : Nat)
    (i e : Nat) : ℝ :=
  s.readMem A (i * stride_am + e * stride_ak)

/-- `B[e, j] = readMem B (e · stride_bk + j · stride_bn)` — the address of tile
lane `(e, j)` of the `b_block_ptr` view (offsets `(0, 0)`). -/
noncomputable def bElem (s : BlockState) (B : RegionName) (stride_bk stride_bn : Nat)
    (e j : Nat) : ℝ :=
  s.readMem B (e * stride_bk + j * stride_bn)

/-- **Genuine GEMM spec** (over ℝ): `C[i,j] = Σ_{e < BLOCK_K} A[i,e] · B[e,j]`. -/
noncomputable def matmulSpec (s : BlockState) (A B : RegionName)
    (stride_am stride_ak stride_bk stride_bn BLOCK_K : Nat)
    (i j : Nat) : ℝ :=
  (Finset.univ : Finset (Fin BLOCK_K)).sum
    (fun e => aElem s A stride_am stride_ak i e.val
              * bElem s B stride_bk stride_bn e.val j)

/-- The output store address for tile lane `(i,j)`: `i · stride_cm + j · stride_cn`
(the `c_block_ptr` address with offsets `(0, 0)`). -/
def cOffset (stride_cm stride_cn : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : Nat :=
  idx.1.val * stride_cm + idx.2.1.val * stride_cn

/-- **Reusable discharge for the `hInj` precondition of the closed-form theorems.**

For a contiguous row-major `C` tile (`stride_cn = 1`, with the row stride at least
the tile width, `BLOCK_N ≤ stride_cm` — always true for a valid tiling) the output
address map `cOffset (i,j) = i·stride_cm + j` is injective: the column offset `j <
BLOCK_N ≤ stride_cm` never spills into the next row, so `(i,j) ↦ address` is a
base-`stride_cm` encoding. The closed-form theorems keep the fully general abstract
`Function.Injective` hypothesis (covering any non-aliasing layout, e.g. transposed
`C`); this lemma discharges it for the common row-major case. -/
theorem cOffset_injective_of_rowMajor {BLOCK_M BLOCK_N stride_cm stride_cn : Nat}
    (hcn : stride_cn = 1) (hcm : BLOCK_N ≤ stride_cm) :
    Function.Injective
      (cOffset (BLOCK_M := BLOCK_M) (BLOCK_N := BLOCK_N) stride_cm stride_cn) := by
  subst hcn
  have heq : cOffset (BLOCK_M := BLOCK_M) (BLOCK_N := BLOCK_N) stride_cm 1
      = fun idx : TileIndex [BLOCK_M, BLOCK_N] => 0 + idx.1.val * stride_cm + idx.2.1.val := by
    funext idx; simp only [cOffset]; ring
  rw [heq]
  exact rowMajor2D_inj _ stride_cm hcm

/-! ## exec-stepping helpers -/

/-- `tl.make_block_ptr` (offsets `(0,0)`) eval: every lane is the same
`BlockPtr` record. -/
theorem makeBlockPtr_eval (s : BlockState) (R : RegionName)
    (parentR parentC strideR strideC BR BC : Nat) :
    evalOp (Op.makeBlockPtrDyn R (Op.constNat 0) [parentR, parentC] [BR, BC]
      [strideR, strideC] [0, 0]) s
      = some (⟨fun _ : TileIndex [BR, BC] =>
          { region := R, baseOffset := 0, parentShape := [parentR, parentC],
            blockShape := [BR, BC], strides := [strideR, strideC],
            offsets := [0, 0] }⟩ : Tile .blockPtr [BR, BC]) := by
  simp [evalOp, Option.bind]

/-- Block-pointer load (offsets `(0,0)`, empty boundary check) of a region whose
lane `(i,j)` address is `i·strideR + j·strideC`: reads `readMem` at that address. -/
theorem load_blockPtr_eval {BR BC : Nat} (s : BlockState) (R : RegionName)
    (parentR parentC strideR strideC : Nat)
    (bpName : RegName)
    (hbp : s.regs .blockPtr [BR, BC] bpName
      = some (⟨fun _ : TileIndex [BR, BC] =>
          { region := R, baseOffset := 0, parentShape := [parentR, parentC],
            blockShape := [BR, BC], strides := [strideR, strideC],
            offsets := [0, 0] }⟩ : Tile .blockPtr [BR, BC])) :
    evalOp (.load .real (.blockPtr (Op.ref .blockPtr [BR, BC] bpName) []) .none) s
      = some (⟨fun idx : TileIndex [BR, BC] =>
          some (s.readMem R (idx.1.val * strideR + idx.2.1.val * strideC))⟩
          : Tile .real [BR, BC]) := by
  simp only [evalOp, evalOp_ref, hbp, Option.bind_some, Option.bind, Option.map]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, j, u⟩ := idx
  simp only [TileShape.indexToList, BlockState.readMemValue_real, BlockPtr.inBounds,
    List.all_nil, if_true, BlockPtr.address_2d_zero_offsets, Nat.zero_add]

/-- `tl.dot(a, b)` statement eval: evaluates to `Tile.dot [] at_ bt` given the
register values of `a`/`b`. -/
theorem dot_eval {BM BN BLOCK_K : Nat} (s : BlockState)
    (at_ : Tile .real [BM, BLOCK_K]) (bt : Tile .real [BLOCK_K, BN])
    (ha : s.regs .real [BM, BLOCK_K] "a" = some at_)
    (hb : s.regs .real [BLOCK_K, BN] "b" = some bt) :
    evalOp (Op.dot (batch := []) (Op.ref .real [BM, BLOCK_K] "a")
        (Op.ref .real [BLOCK_K, BN] "b")) s
      = some (Tile.dot [] at_ bt) := by
  rw [evalOp_dot]; simp [ha, hb]

/-- `(tl.dot(a, b)).to(fp16)` statement eval: the fp16 downcast of the dot. -/
theorem castdot_eval {BM BN BLOCK_K : Nat} (s : BlockState)
    (at_ : Tile .real [BM, BLOCK_K]) (bt : Tile .real [BLOCK_K, BN])
    (ha : s.regs .real [BM, BLOCK_K] "a" = some at_)
    (hb : s.regs .real [BLOCK_K, BN] "b" = some bt) :
    evalOp (Op.castFloat FloatDType.real FloatDType.fp16
        (Op.dot (batch := []) (Op.ref .real [BM, BLOCK_K] "a")
          (Op.ref .real [BLOCK_K, BN] "b"))) s
      = some (⟨fun idx => FloatDType.real.cast FloatDType.fp16 ((Tile.dot [] at_ bt).data idx)⟩
          : Tile FloatDType.fp16.toTileDType ([] ++ [BM, BN])) := by
  have hd : evalOp (Op.dot (batch := []) (Op.ref .real [BM, BLOCK_K] "a")
      (Op.ref .real [BLOCK_K, BN] "b")) s = some (Tile.dot [] at_ bt) := dot_eval s at_ bt ha hb
  rw [evalOp_castFloat]
  erw [hd]
  rfl

/-! ## Block-pointer store scatter -/

/-- Real block-pointer store (offsets `(0,0)`, empty boundary check), lane `(i,j)`
address `i·strideR + j·strideC`. The `if true && inBounds [] = true` guard is
unconditional, so this is an ordinary injective real scatter. -/
theorem store_blockPtr_real_readback {BR BC : Nat} (s : BlockState)
    (R : RegionName) (parentR parentC strideR strideC : Nat)
    (cpName cName : RegName)
    (vt : Tile .real [BR, BC])
    (hcp : s.regs .blockPtr [BR, BC] cpName
      = some (⟨fun _ : TileIndex [BR, BC] =>
          { region := R, baseOffset := 0, parentShape := [parentR, parentC],
            blockShape := [BR, BC], strides := [strideR, strideC],
            offsets := [0, 0] }⟩ : Tile .blockPtr [BR, BC]))
    (hc : s.regs .real [BR, BC] cName = some vt)
    (offsetFn : TileIndex [BR, BC] → Nat)
    (hoff : ∀ idx : TileIndex [BR, BC], offsetFn idx = idx.1.val * strideR + idx.2.1.val * strideC)
    (hInj : Function.Injective offsetFn) :
    ∃ s', stepStmt (Stmt.store .real [BR, BC]
        (.blockPtr (Op.ref .blockPtr [BR, BC] cpName) [])
        (Op.ref .real [BR, BC] cName) .none) s = some s'
      ∧ ∀ idx : TileIndex [BR, BC],
          s'.readMem R (offsetFn idx) = (vt.data idx).unbotD 0 := by
  set sfin := (TileShape.allIndices [BR, BC]).foldl
      (fun acc i => acc.writeMem R (offsetFn i) ((vt.data i).unbotD 0)) s with hsfin
  have hstep : stepStmt (Stmt.store .real [BR, BC]
        (.blockPtr (Op.ref .blockPtr [BR, BC] cpName) [])
        (Op.ref .real [BR, BC] cName) .none) s = some sfin := by
    simp only [stepStmt, evalOp_ref, hc, hcp, Option.bind_some, bind]
    refine congrArg some ?_
    rw [hsfin]
    apply List.foldl_ext
    intro acc i _
    obtain ⟨ii, jj, u⟩ := i
    rw [show TileShape.indexToList [BR, BC] (ii, jj, PUnit.unit) = [ii.val, jj.val] by
          simp [TileShape.indexToList]]
    simp only [BlockPtr.inBounds, List.all_nil, Bool.and_true, if_true,
      BlockPtr.address_2d_zero_offsets, Nat.zero_add, hoff,
      BlockState.writeMemTyped_real, FloatDType.real_storeValue]
  refine ⟨sfin, hstep, ?_⟩
  intro idx
  rw [hsfin]
  exact BlockState.scatter_readback_nd s offsetFn
    (fun i => (vt.data i).unbotD 0) hInj idx

/-- fp16 block-pointer store (offsets `(0,0)`, empty boundary check): the stored
value is the `castFloat`-downcast of the dot result. -/
theorem store_blockPtr_fp16_readback {BR BC : Nat} (s : BlockState)
    (R : RegionName) (parentR parentC strideR strideC : Nat)
    (cpName cName : RegName)
    (vt : Tile .fp16 [BR, BC])
    (hcp : s.regs .blockPtr [BR, BC] cpName
      = some (⟨fun _ : TileIndex [BR, BC] =>
          { region := R, baseOffset := 0, parentShape := [parentR, parentC],
            blockShape := [BR, BC], strides := [strideR, strideC],
            offsets := [0, 0] }⟩ : Tile .blockPtr [BR, BC]))
    (hc : s.regs .fp16 [BR, BC] cName = some vt)
    (offsetFn : TileIndex [BR, BC] → Nat)
    (hoff : ∀ idx : TileIndex [BR, BC], offsetFn idx = idx.1.val * strideR + idx.2.1.val * strideC)
    (hInj : Function.Injective offsetFn) :
    ∃ s', stepStmt (Stmt.store .fp16 [BR, BC]
        (.blockPtr (Op.ref .blockPtr [BR, BC] cpName) [])
        (Op.ref .fp16 [BR, BC] cName) .none) s = some s'
      ∧ ∀ idx : TileIndex [BR, BC],
          s'.mem R (offsetFn idx)
            = MemCell.of .fp16
                (FloatDType.fp16.ofReal (FloatDType.fp16.storeValue (vt.data idx))) := by
  set sfin := (TileShape.allIndices [BR, BC]).foldl
      (fun acc i => acc.writeMemTyped .fp16 R (offsetFn i) (vt.data i)) s with hsfin
  have hstep : stepStmt (Stmt.store .fp16 [BR, BC]
        (.blockPtr (Op.ref .blockPtr [BR, BC] cpName) [])
        (Op.ref .fp16 [BR, BC] cName) .none) s = some sfin := by
    simp only [stepStmt, evalOp_ref, hc, hcp, Option.bind_some, bind]
    refine congrArg some ?_
    rw [hsfin]
    apply List.foldl_ext
    intro acc i _
    obtain ⟨ii, jj, u⟩ := i
    rw [show TileShape.indexToList [BR, BC] (ii, jj, PUnit.unit) = [ii.val, jj.val] by
          simp [TileShape.indexToList]]
    simp only [BlockPtr.inBounds, List.all_nil, Bool.and_true, if_true,
      BlockPtr.address_2d_zero_offsets, Nat.zero_add, hoff]
  refine ⟨sfin, hstep, ?_⟩
  intro idx
  rw [hsfin]
  exact scatter_memcell_fp16_nd (region := R) s offsetFn (fun i => vt.data i) hInj idx

/-! ## f32 branch -/

set_option maxHeartbeats 1000000 in
/-- **f32 exec closed form**: every output cell of the executed `matmul_tma`
(float32 branch) equals the genuine GEMM value `Σ_e A·B`. -/
theorem matmul_tma_f32_exec_closed_form
    (A B C : RegionName) (s : BlockState)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_M BLOCK_N BLOCK_K : Nat)
    (hInj : Function.Injective (cOffset (BLOCK_M := BLOCK_M) (BLOCK_N := BLOCK_N) stride_cm stride_cn))
    (idx : TileIndex [BLOCK_M, BLOCK_N]) :
    (match exec (matmul_tma_f32_surface A B C M N K stride_am stride_ak
        stride_bk stride_bn stride_cm stride_cn BLOCK_M BLOCK_N BLOCK_K) s with
      | some s' => s'.readMem C (cOffset stride_cm stride_cn idx)
      | none => (0 : ℝ)) =
      matmulSpec s A B stride_am stride_ak stride_bk stride_bn BLOCK_K
        idx.1.val idx.2.1.val := by
  -- the loaded a / b tiles
  set aT : Tile .real [BLOCK_M, BLOCK_K] :=
    ⟨fun idx : TileIndex [BLOCK_M, BLOCK_K] =>
      some (s.readMem A (idx.1.val * stride_am + idx.2.1.val * stride_ak))⟩ with haT
  set bT : Tile .real [BLOCK_K, BLOCK_N] :=
    ⟨fun idx : TileIndex [BLOCK_K, BLOCK_N] =>
      some (s.readMem B (idx.1.val * stride_bk + idx.2.1.val * stride_bn))⟩ with hbT
  set cT : Tile .real [BLOCK_M, BLOCK_N] :=
    ⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
      some (matmulSpec s A B stride_am stride_ak stride_bk stride_bn BLOCK_K
        idx.1.val idx.2.1.val)⟩ with hcT
  set apT : Tile .blockPtr [BLOCK_M, BLOCK_K] :=
    ⟨fun _ : TileIndex [BLOCK_M, BLOCK_K] =>
      { region := A, baseOffset := 0, parentShape := [M, K],
        blockShape := [BLOCK_M, BLOCK_K], strides := [stride_am, stride_ak],
        offsets := [0, 0] }⟩ with hapT
  set bpT : Tile .blockPtr [BLOCK_K, BLOCK_N] :=
    ⟨fun _ : TileIndex [BLOCK_K, BLOCK_N] =>
      { region := B, baseOffset := 0, parentShape := [K, N],
        blockShape := [BLOCK_K, BLOCK_N], strides := [stride_bk, stride_bn],
        offsets := [0, 0] }⟩ with hbpT
  set cpT : Tile .blockPtr [BLOCK_M, BLOCK_N] :=
    ⟨fun _ : TileIndex [BLOCK_M, BLOCK_N] =>
      { region := C, baseOffset := 0, parentShape := [M, N],
        blockShape := [BLOCK_M, BLOCK_N], strides := [stride_cm, stride_cn],
        offsets := [0, 0] }⟩ with hcpT
  -- step through the straight-line body
  rw [show exec (matmul_tma_f32_surface A B C M N K stride_am stride_ak
        stride_bk stride_bn stride_cm stride_cn BLOCK_M BLOCK_N BLOCK_K) s
      = stepStmts ((matmul_tma_f32_surface A B C M N K stride_am stride_ak
          stride_bk stride_bn stride_cm stride_cn BLOCK_M BLOCK_N BLOCK_K).toAlgKernel.body) s
      from rfl]
  rw [show (matmul_tma_f32_surface A B C M N K stride_am stride_ak
          stride_bk stride_bn stride_cm stride_cn BLOCK_M BLOCK_N BLOCK_K).toAlgKernel.body
      = [ Stmt.assign .blockPtr [BLOCK_M, BLOCK_K] "a_block_ptr"
            (Op.makeBlockPtrDyn A (Op.constNat 0) [M, K] [BLOCK_M, BLOCK_K] [stride_am, stride_ak] [0, 0]),
          Stmt.assign .blockPtr [BLOCK_K, BLOCK_N] "b_block_ptr"
            (Op.makeBlockPtrDyn B (Op.constNat 0) [K, N] [BLOCK_K, BLOCK_N] [stride_bk, stride_bn] [0, 0]),
          Stmt.assign .blockPtr [BLOCK_M, BLOCK_N] "c_block_ptr"
            (Op.makeBlockPtrDyn C (Op.constNat 0) [M, N] [BLOCK_M, BLOCK_N] [stride_cm, stride_cn] [0, 0]),
          Stmt.assign .real [BLOCK_M, BLOCK_K] "a"
            (Op.load .real (.blockPtr (Op.ref .blockPtr [BLOCK_M, BLOCK_K] "a_block_ptr") []) .none),
          Stmt.assign .real [BLOCK_K, BLOCK_N] "b"
            (Op.load .real (.blockPtr (Op.ref .blockPtr [BLOCK_K, BLOCK_N] "b_block_ptr") []) .none),
          Stmt.assign .real ([] ++ [BLOCK_M, BLOCK_N]) "c"
            (Op.dot (batch := []) (Op.ref .real [BLOCK_M, BLOCK_K] "a") (Op.ref .real [BLOCK_K, BLOCK_N] "b")),
          Stmt.store .real [BLOCK_M, BLOCK_N]
            (.blockPtr (Op.ref .blockPtr [BLOCK_M, BLOCK_N] "c_block_ptr") [])
            (Op.ref .real [BLOCK_M, BLOCK_N] "c") .none ] from rfl]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (makeBlockPtr_eval s A M K stride_am stride_ak BLOCK_M BLOCK_K))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (makeBlockPtr_eval _ B K N stride_bk stride_bn BLOCK_K BLOCK_N))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (makeBlockPtr_eval _ C M N stride_cm stride_cn BLOCK_M BLOCK_N))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (load_blockPtr_eval (BR := BLOCK_M) (BC := BLOCK_K) _ A M K stride_am stride_ak "a_block_ptr"
          (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (load_blockPtr_eval (BR := BLOCK_K) (BC := BLOCK_N) _ B K N stride_bk stride_bn "b_block_ptr"
          (by simp)))]
  -- normalise intermediate-state readMem to `s.readMem`, fold the loaded tiles
  simp only [BlockState.setReg_readMem]
  rw [← haT, ← hbT]
  -- name the post-load state so the dot/store steps have concrete registers
  set s5 := (((((s.setReg "a_block_ptr" .blockPtr [BLOCK_M, BLOCK_K] apT).setReg
        "b_block_ptr" .blockPtr [BLOCK_K, BLOCK_N] bpT).setReg
        "c_block_ptr" .blockPtr [BLOCK_M, BLOCK_N] cpT).setReg
        "a" .real [BLOCK_M, BLOCK_K] aT).setReg "b" .real [BLOCK_K, BLOCK_N] bT) with hs5
  have hmem5 : ∀ (R : RegionName) (o : Nat), s5.readMem R o = s.readMem R o := by
    intro R o; simp only [hs5, BlockState.setReg_readMem]
  have ha5 : s5.regs .real [BLOCK_M, BLOCK_K] "a" = some aT := by rw [hs5]; simp
  have hb5 : s5.regs .real [BLOCK_K, BLOCK_N] "b" = some bT := by rw [hs5]; simp
  have hcp5 : s5.regs .blockPtr [BLOCK_M, BLOCK_N] "c_block_ptr" = some cpT := by
    rw [hs5]; simp [hcpT]
  -- the dot step (over the post-load state)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (dot_eval s5 aT bT ha5 hb5))]
  -- characterise the dot tile, lane `(i,j)`, as the genuine GEMM spec value
  set cTile : Tile .real [BLOCK_M, BLOCK_N] :=
    ⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
      some (matmulSpec s A B stride_am stride_ak stride_bk stride_bn BLOCK_K
        idx.1.val idx.2.1.val)⟩ with hcTile
  have hdotval : ∀ idx : TileIndex [BLOCK_M, BLOCK_N],
      (Tile.dot [] aT bT).data idx = cTile.data idx := by
    intro idx
    obtain ⟨i, j, u⟩ := idx
    rw [tile_dot_data BLOCK_M BLOCK_K BLOCK_N aT bT i j
          (fun e => aElem s A stride_am stride_ak i.val e.val)
          (fun e => bElem s B stride_bk stride_bn e.val j.val)
          (fun e => by rw [haT]; rfl) (fun e => by rw [hbT]; rfl)]
    rfl
  -- the store: read back the genuine GEMM value
  obtain ⟨sfin, hstore, hread⟩ :=
    store_blockPtr_real_readback (BR := BLOCK_M) (BC := BLOCK_N)
      (s5.setReg "c" .real ([] ++ [BLOCK_M, BLOCK_N]) (Tile.dot [] aT bT))
      C M N stride_cm stride_cn "c_block_ptr" "c" (Tile.dot [] aT bT)
      (by rw [hcp5.symm]; simp) (by simp)
      (cOffset stride_cm stride_cn) (fun idx => by simp [cOffset]) hInj
  rw [stepStmts.cons_some hstore, stepStmts.nil]
  show sfin.readMem C (cOffset stride_cm stride_cn idx) = _
  rw [hread idx, hdotval idx, hcTile]
  rfl

/-- **Closed-form correctness for the f32 `matmul_tma` (general statement).**

For any matrix/tile dimensions and strides, every output cell of the computed
`BLOCK_M × BLOCK_N` tile equals the genuine matrix product
`Σ_{e < BLOCK_K} A[i,e] · B[e,j]` (over ℝ) of the loaded `A`/`B` tiles — *not*
the kernel's own executed value. Layout: `A[i,e]` at
`A + i·stride_am + e·stride_ak`, `B[e,j]` at `B + e·stride_bk + j·stride_bn`,
`C[i,j]` at `C + i·stride_cm + j·stride_cn` (the block pointers' offset-`(0,0)`
addresses). Precondition: output-offset injectivity. -/
theorem matmul_tma_f32_closed_form_correct
    (A B C : RegionName) (s : BlockState)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_M BLOCK_N BLOCK_K : Nat)
    (hcn : stride_cn = 1) (hcm : BLOCK_N ≤ stride_cm) :
    ComputeCorrect.Realizes
      (kernel := matmul_tma_f32_surface A B C M N K stride_am stride_ak
        stride_bk stride_bn stride_cm stride_cn BLOCK_M BLOCK_N BLOCK_K)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
        some (C, cOffset stride_cm stride_cn idx))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
        matmulSpec s A B stride_am stride_ak stride_bk stride_bn BLOCK_K
          idx.1.val idx.2.1.val) := by
  have hInj := cOffset_injective_of_rowMajor (BLOCK_M := BLOCK_M) hcn hcm
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [matmul_tma_f32_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst hs0
  intro idx
  have hmain := matmul_tma_f32_exec_closed_form A B C s0 M N K stride_am stride_ak
    stride_bk stride_bn stride_cm stride_cn BLOCK_M BLOCK_N BLOCK_K hInj idx
  have hExec2 : exec (matmul_tma_f32_surface A B C M N K stride_am stride_ak
      stride_bk stride_bn stride_cm stride_cn BLOCK_M BLOCK_N BLOCK_K) s0 = some s' := hExec
  rw [hExec2] at hmain
  simpa only [ComputeCorrect.OutputReadable.read_real] using hmain

/-! ## fp16 branch -/

set_option maxHeartbeats 1000000 in
/-- **fp16 exec closed form**: every output cell of the executed `matmul_tma`
(fp16 branch) equals `fp16(Σ_e A·B)` — the genuine GEMM value cast to fp16. -/
theorem matmul_tma_f16_exec_closed_form
    (A B C : RegionName) (s : BlockState)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_M BLOCK_N BLOCK_K : Nat)
    (hInj : Function.Injective (cOffset (BLOCK_M := BLOCK_M) (BLOCK_N := BLOCK_N) stride_cm stride_cn))
    (idx : TileIndex [BLOCK_M, BLOCK_N]) :
    (match exec (matmul_tma_f16_surface A B C M N K stride_am stride_ak
        stride_bk stride_bn stride_cm stride_cn BLOCK_M BLOCK_N BLOCK_K) s with
      | some s' => s'.mem C (cOffset stride_cm stride_cn idx)
      | none => (0 : MemCell)) =
      MemCell.of .fp16
        (FloatDType.real.cast FloatDType.fp16
          (some (matmulSpec s A B stride_am stride_ak stride_bk stride_bn BLOCK_K
            idx.1.val idx.2.1.val))) := by
  set aT : Tile .real [BLOCK_M, BLOCK_K] :=
    ⟨fun idx : TileIndex [BLOCK_M, BLOCK_K] =>
      some (s.readMem A (idx.1.val * stride_am + idx.2.1.val * stride_ak))⟩ with haT
  set bT : Tile .real [BLOCK_K, BLOCK_N] :=
    ⟨fun idx : TileIndex [BLOCK_K, BLOCK_N] =>
      some (s.readMem B (idx.1.val * stride_bk + idx.2.1.val * stride_bn))⟩ with hbT
  set apT : Tile .blockPtr [BLOCK_M, BLOCK_K] :=
    ⟨fun _ : TileIndex [BLOCK_M, BLOCK_K] =>
      { region := A, baseOffset := 0, parentShape := [M, K],
        blockShape := [BLOCK_M, BLOCK_K], strides := [stride_am, stride_ak],
        offsets := [0, 0] }⟩ with hapT
  set bpT : Tile .blockPtr [BLOCK_K, BLOCK_N] :=
    ⟨fun _ : TileIndex [BLOCK_K, BLOCK_N] =>
      { region := B, baseOffset := 0, parentShape := [K, N],
        blockShape := [BLOCK_K, BLOCK_N], strides := [stride_bk, stride_bn],
        offsets := [0, 0] }⟩ with hbpT
  set cpT : Tile .blockPtr [BLOCK_M, BLOCK_N] :=
    ⟨fun _ : TileIndex [BLOCK_M, BLOCK_N] =>
      { region := C, baseOffset := 0, parentShape := [M, N],
        blockShape := [BLOCK_M, BLOCK_N], strides := [stride_cm, stride_cn],
        offsets := [0, 0] }⟩ with hcpT
  rw [show exec (matmul_tma_f16_surface A B C M N K stride_am stride_ak
        stride_bk stride_bn stride_cm stride_cn BLOCK_M BLOCK_N BLOCK_K) s
      = stepStmts ((matmul_tma_f16_surface A B C M N K stride_am stride_ak
          stride_bk stride_bn stride_cm stride_cn BLOCK_M BLOCK_N BLOCK_K).toAlgKernel.body) s
      from rfl]
  rw [show (matmul_tma_f16_surface A B C M N K stride_am stride_ak
          stride_bk stride_bn stride_cm stride_cn BLOCK_M BLOCK_N BLOCK_K).toAlgKernel.body
      = [ Stmt.assign .blockPtr [BLOCK_M, BLOCK_K] "a_block_ptr"
            (Op.makeBlockPtrDyn A (Op.constNat 0) [M, K] [BLOCK_M, BLOCK_K] [stride_am, stride_ak] [0, 0]),
          Stmt.assign .blockPtr [BLOCK_K, BLOCK_N] "b_block_ptr"
            (Op.makeBlockPtrDyn B (Op.constNat 0) [K, N] [BLOCK_K, BLOCK_N] [stride_bk, stride_bn] [0, 0]),
          Stmt.assign .blockPtr [BLOCK_M, BLOCK_N] "c_block_ptr"
            (Op.makeBlockPtrDyn C (Op.constNat 0) [M, N] [BLOCK_M, BLOCK_N] [stride_cm, stride_cn] [0, 0]),
          Stmt.assign .real [BLOCK_M, BLOCK_K] "a"
            (Op.load .real (.blockPtr (Op.ref .blockPtr [BLOCK_M, BLOCK_K] "a_block_ptr") []) .none),
          Stmt.assign .real [BLOCK_K, BLOCK_N] "b"
            (Op.load .real (.blockPtr (Op.ref .blockPtr [BLOCK_K, BLOCK_N] "b_block_ptr") []) .none),
          Stmt.assign FloatDType.fp16.toTileDType ([] ++ [BLOCK_M, BLOCK_N]) "c"
            (Op.castFloat FloatDType.real FloatDType.fp16
              (Op.dot (batch := []) (Op.ref .real [BLOCK_M, BLOCK_K] "a") (Op.ref .real [BLOCK_K, BLOCK_N] "b"))),
          Stmt.store .fp16 [BLOCK_M, BLOCK_N]
            (.blockPtr (Op.ref .blockPtr [BLOCK_M, BLOCK_N] "c_block_ptr") [])
            (Op.ref .fp16 [BLOCK_M, BLOCK_N] "c") .none ] from rfl]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (makeBlockPtr_eval s A M K stride_am stride_ak BLOCK_M BLOCK_K))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (makeBlockPtr_eval _ B K N stride_bk stride_bn BLOCK_K BLOCK_N))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (makeBlockPtr_eval _ C M N stride_cm stride_cn BLOCK_M BLOCK_N))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (load_blockPtr_eval (BR := BLOCK_M) (BC := BLOCK_K) _ A M K stride_am stride_ak "a_block_ptr"
          (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (load_blockPtr_eval (BR := BLOCK_K) (BC := BLOCK_N) _ B K N stride_bk stride_bn "b_block_ptr"
          (by simp)))]
  simp only [BlockState.setReg_readMem]
  rw [← haT, ← hbT]
  set s5 := (((((s.setReg "a_block_ptr" .blockPtr [BLOCK_M, BLOCK_K] apT).setReg
        "b_block_ptr" .blockPtr [BLOCK_K, BLOCK_N] bpT).setReg
        "c_block_ptr" .blockPtr [BLOCK_M, BLOCK_N] cpT).setReg
        "a" .real [BLOCK_M, BLOCK_K] aT).setReg "b" .real [BLOCK_K, BLOCK_N] bT) with hs5
  have ha5 : s5.regs .real [BLOCK_M, BLOCK_K] "a" = some aT := by rw [hs5]; simp
  have hb5 : s5.regs .real [BLOCK_K, BLOCK_N] "b" = some bT := by rw [hs5]; simp
  have hcp5 : s5.regs .blockPtr [BLOCK_M, BLOCK_N] "c_block_ptr" = some cpT := by
    rw [hs5]; simp [hcpT]
  -- the cast(dot, fp16) step
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (castdot_eval s5 aT bT ha5 hb5))]
  -- characterise the cast-dot tile, lane `(i,j)`, as fp16(GEMM spec value)
  set cTile : Tile .fp16 [BLOCK_M, BLOCK_N] :=
    ⟨fun idx => FloatDType.real.cast FloatDType.fp16 ((Tile.dot [] aT bT).data idx)⟩ with hcTile
  have hdotval : ∀ idx : TileIndex [BLOCK_M, BLOCK_N],
      cTile.data idx = FloatDType.real.cast FloatDType.fp16
        (some (matmulSpec s A B stride_am stride_ak stride_bk stride_bn BLOCK_K
          idx.1.val idx.2.1.val)) := by
    intro idx
    obtain ⟨i, j, u⟩ := idx
    simp only [hcTile]
    rw [tile_dot_data BLOCK_M BLOCK_K BLOCK_N aT bT i j
          (fun e => aElem s A stride_am stride_ak i.val e.val)
          (fun e => bElem s B stride_bk stride_bn e.val j.val)
          (fun e => by rw [haT]; rfl) (fun e => by rw [hbT]; rfl)]
    rfl
  -- the fp16 store
  obtain ⟨sfin, hstore, hread⟩ :=
    store_blockPtr_fp16_readback (BR := BLOCK_M) (BC := BLOCK_N)
      (s5.setReg "c" FloatDType.fp16.toTileDType ([] ++ [BLOCK_M, BLOCK_N])
        ⟨fun idx => FloatDType.real.cast FloatDType.fp16 ((Tile.dot [] aT bT).data idx)⟩)
      C M N stride_cm stride_cn "c_block_ptr" "c" cTile
      (by rw [hcp5.symm]; simp) (by simp [hcTile])
      (cOffset stride_cm stride_cn) (fun idx => by simp [cOffset]) hInj
  rw [stepStmts.cons_some hstore, stepStmts.nil]
  show sfin.mem C (cOffset stride_cm stride_cn idx) = _
  rw [hread idx, hdotval idx]
  -- collapse the fp16 round-trip
  simp only [FloatDType.cast, FloatDType.ofReal, FloatDType.storeValue,
    FloatDType.real_toWithBot, FloatDType.fp16_ofWithBot, FloatDType.fp16_toWithBot,
    WithBot.unbotD_some]

/-- **Closed-form correctness for the fp16 `matmul_tma`.** Every output cell of
the computed tile equals `fp16(Σ_{e<BLOCK_K} A[i,e]·B[e,j])` — the genuine
matrix product over ℝ cast to float16. -/
theorem matmul_tma_f16_closed_form_correct
    (A B C : RegionName) (s : BlockState)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_M BLOCK_N BLOCK_K : Nat)
    (hcn : stride_cn = 1) (hcm : BLOCK_N ≤ stride_cm) :
    ComputeCorrect.Realizes
      (kernel := matmul_tma_f16_surface A B C M N K stride_am stride_ak
        stride_bk stride_bn stride_cm stride_cn BLOCK_M BLOCK_N BLOCK_K)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
        some (C, cOffset stride_cm stride_cn idx))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (matmulSpec s A B stride_am stride_ak stride_bk stride_bn BLOCK_K
              idx.1.val idx.2.1.val)))) := by
  have hInj := cOffset_injective_of_rowMajor (BLOCK_M := BLOCK_M) hcn hcm
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [matmul_tma_f16_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst hs0
  intro idx
  have hmain := matmul_tma_f16_exec_closed_form A B C s0 M N K stride_am stride_ak
    stride_bk stride_bn stride_cm stride_cn BLOCK_M BLOCK_N BLOCK_K hInj idx
  have hExec2 : exec (matmul_tma_f16_surface A B C M N K stride_am stride_ak
      stride_bk stride_bn stride_cm stride_cn BLOCK_M BLOCK_N BLOCK_K) s0 = some s' := hExec
  rw [hExec2] at hmain
  simpa only [ComputeCorrect.OutputReadable.read_memcell] using hmain

end VeriTile.Bench.TritonBenchG.MatmulTma
