import VeriTile.Triton

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
matmul_tma_f32_io_correctness               ← TOP THEOREM (`⊨`, f32 branch)
  ├─ matmul_tma_f32_flattenOk               inside the flat-memory bridge
  ├─ matmul_tma_f32_traceSafe               per-execution address safety
  └─ matmul_tma_f32_region_run              region-model run
       ├─ matmul_tma_f32_terminates
       ├─ matmul_tma_f32_exec_closed_form   exec-side closed form (shared, below)
       ├─ matmulSpec_eq_of                  memory spec = value spec under the pins
       └─ matmul_tma_f32_frame              cell-level frame off the write window

matmul_tma_f32_closed_form_correct          per-write-map summary (f32 branch)
  └─ matmul_tma_f32_exec_closed_form        ← exec-side closed form (every cell = ∑_e A·B)
matmul_tma_f16_closed_form_correct          per-write-map summary (fp16 branch)
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

open VeriTile.Triton
open scoped VeriTile.Triton.MaskedTileShapedKernelIO₂

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
specification matmul_tma_f32_closed_form_correct
    (A B C : RegionName) (s : BlockState)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_M BLOCK_N BLOCK_K : Nat)
    (hcn : stride_cn = 1) (hcm : BLOCK_N ≤ stride_cm) :
    ComputeCorrect.Realizes_without_Rounding
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
specification matmul_tma_f16_closed_form_correct
    (A B C : RegionName) (s : BlockState)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_M BLOCK_N BLOCK_K : Nat)
    (hcn : stride_cn = 1) (hcm : BLOCK_N ≤ stride_cm) :
    ComputeCorrect.Realizes_without_Rounding
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

/-! ## ════════ `⊨` IO face for the f32 branch ════════

The closed forms above are stated per *declared write map*. This section restates
the f32 branch on the audit-once IO surface
`MaskedTileShapedKernelIO₂.Implements` (`⊨`), which additionally pins the **flat
memory** placement.

A contraction is exactly the case the per-channel-shape skin exists for: the two
inputs live on `[BLOCK_M, BLOCK_N]`-incompatible lane sets — `A` on
`[BLOCK_M, BLOCK_K]`, `B` on `[BLOCK_K, BLOCK_N]` — and the output on
`[BLOCK_M, BLOCK_N]`. The three are related only through `f`, which reads both
inputs at lanes the output index does not name. Every lane of all three channels
is active: the block pointers carry no `boundary_check`. -/

section IOFace

/-- Cell-level frame of an **unmasked** scatter (private copy — `bench` files are
standalone). -/
private theorem foldl_writeMem_frame_unmasked {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → ℝ) (R : RegionName) (off : Nat) :
    ∀ l : List α, (R ≠ region ∨ ∀ k ∈ l, offsetFn k ≠ off) →
      ∀ s : BlockState,
        ((l.foldl (fun acc k => acc.writeMem region (offsetFn k) (valueFn k))
            s).mem R off) = s.mem R off := by
  intro l
  induction l with
  | nil => intro _ s; rfl
  | cons hd tl ih =>
      intro hc s
      have htl : R ≠ region ∨ ∀ k ∈ tl, offsetFn k ≠ off := by
        rcases hc with h | h
        · exact Or.inl h
        · exact Or.inr fun k hk => h k (List.mem_cons_of_mem hd hk)
      rw [List.foldl_cons, ih htl, BlockState.writeMem_mem, if_neg ?_]
      rintro ⟨h1, h2⟩
      rcases hc with h | h
      · exact h h1
      · exact h hd List.mem_cons_self h2.symm

/-- Value-level GEMM spec: `Σ_{e < BLOCK_K} xs[i, e] · ys[e, j]`, over the
*loaded values* rather than over memory — which is what the IO surface
quantifies. Reading both inputs at lanes the output index does not name is the
whole reason the channels need separate shapes. -/
noncomputable def matmulSpecOf (BLOCK_M BLOCK_N BLOCK_K : Nat)
    (xs : TileIndex [BLOCK_M, BLOCK_K] → ℝ)
    (ys : TileIndex [BLOCK_K, BLOCK_N] → ℝ)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : ℝ :=
  (Finset.univ : Finset (Fin BLOCK_K)).sum
    (fun e => xs (idx.1, e, PUnit.unit) * ys (e, idx.2.1, PUnit.unit))

/-- The memory-level and value-level GEMM specs agree once both input tiles are
pinned. -/
theorem matmulSpec_eq_of (A B : RegionName)
    (stride_am stride_ak stride_bk stride_bn BLOCK_M BLOCK_N BLOCK_K : Nat)
    (s : BlockState)
    (xs : TileIndex [BLOCK_M, BLOCK_K] → ℝ)
    (ys : TileIndex [BLOCK_K, BLOCK_N] → ℝ)
    (hx : ∀ k : TileIndex [BLOCK_M, BLOCK_K],
      s.readMem A (k.1.val * stride_am + k.2.1.val * stride_ak) = xs k)
    (hy : ∀ k : TileIndex [BLOCK_K, BLOCK_N],
      s.readMem B (k.1.val * stride_bk + k.2.1.val * stride_bn) = ys k)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) :
    matmulSpec s A B stride_am stride_ak stride_bk stride_bn BLOCK_K
        idx.1.val idx.2.1.val
      = matmulSpecOf BLOCK_M BLOCK_N BLOCK_K xs ys idx := by
  simp only [matmulSpec, matmulSpecOf, aElem, bElem]
  refine Finset.sum_congr rfl fun e _ => ?_
  rw [hx (idx.1, e, PUnit.unit), hy (e, idx.2.1, PUnit.unit)]

/-- The f32 branch sits inside the flat-memory bridge's covered fragment. -/
theorem matmul_tma_f32_flattenOk (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_M BLOCK_N BLOCK_K : Nat) :
    ((matmul_tma_f32_surface A B C M N K stride_am stride_ak stride_bk stride_bn
      stride_cm stride_cn BLOCK_M BLOCK_N BLOCK_K).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [matmul_tma_f32_surface, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, StmtList.FlattenOk,
    Stmt.FlattenOk, Op.FlattenOk]
  -- the `tl.dot` node: `Op.FlattenOk`'s per-case equation does not fire under
  -- `simp` on it, but `.eq_def` is fine on this small op-level residual
  simp [Op.FlattenOk.eq_def]

/-- Termination of the f32 branch. -/
theorem matmul_tma_f32_terminates (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_M BLOCK_N BLOCK_K : Nat) (s : BlockState) :
    ∃ s1, exec (matmul_tma_f32_surface A B C M N K stride_am stride_ak stride_bk
      stride_bn stride_cm stride_cn BLOCK_M BLOCK_N BLOCK_K) s = some s1 := by
  simp [exec, matmul_tma_f32_surface, stepStmts, stepStmt, evalOp.eq_def,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Option.bind, Option.map,
    Tile.bop, Tile.dot, TileShape.indexToList, BlockPtr.inBounds]

/-- Cell-level frame of the f32 branch. -/
theorem matmul_tma_f32_frame (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_M BLOCK_N BLOCK_K : Nat) (s s' : BlockState)
    (hExec : exec (matmul_tma_f32_surface A B C M N K stride_am stride_ak
      stride_bk stride_bn stride_cm stride_cn BLOCK_M BLOCK_N BLOCK_K) s
      = some s') :
    ∀ (r : RegionName) (o : Nat),
      (r ≠ C ∨ ∀ idx : TileIndex [BLOCK_M, BLOCK_N],
        o ≠ cOffset stride_cm stride_cn idx) →
      s'.mem r o = s.mem r o := by
  intro r o hcond
  simp [exec, matmul_tma_f32_surface, stepStmts, stepStmt, evalOp.eq_def,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Option.bind, Option.map,
    Tile.bop, Tile.dot, TileShape.indexToList, BlockPtr.inBounds] at hExec
  subst hExec
  rw [foldl_writeMem_frame_unmasked (region := C)
    (fun i : TileIndex [BLOCK_M, BLOCK_N] =>
      i.1.val * stride_cm + i.2.1.val * stride_cn)
    _ r o (TileShape.allIndices [BLOCK_M, BLOCK_N]) ?_]
  · simp
  · rcases hcond with h | h
    · exact Or.inl h
    · exact Or.inr fun i _ => Ne.symm (h i)

/-- Per-execution safety walk for the f32 branch: the three block pointers carry
no `boundary_check`, so every lane of both loads and of the store must be in
bounds. -/
theorem matmul_tma_f32_traceSafe (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_M BLOCK_N BLOCK_K : Nat) (bounds : RegionBounds) (s : BlockState)
    (hA : ∀ idx : TileIndex [BLOCK_M, BLOCK_K],
      idx.1.val * stride_am + idx.2.1.val * stride_ak < bounds A)
    (hB : ∀ idx : TileIndex [BLOCK_K, BLOCK_N],
      idx.1.val * stride_bk + idx.2.1.val * stride_bn < bounds B)
    (hC : ∀ idx : TileIndex [BLOCK_M, BLOCK_N],
      cOffset stride_cm stride_cn idx < bounds C) :
    ((matmul_tma_f32_surface A B C M N K stride_am stride_ak stride_bk stride_bn
      stride_cm stride_cn BLOCK_M BLOCK_N BLOCK_K).toAlgKernel).TraceSafe
      bounds s := by
  simp [Kernel.TraceSafe, matmul_tma_f32_surface, Stmt.TraceSafeList,
    Stmt.TraceSafe, Op.SafeAt, MaskOpt.SafeAt, MaskOpt.Active,
    MaskOpt.MemorySafe, MemAccess.SafeAt, MemAccess.MemorySafe,
    memAccessMemorySafe, MemAccess.ActiveAddressSafe,
    memAccessActiveAddressSafe, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, stepStmt, evalOp, evalOp.eq_def, Option.bind,
    Option.map, Tile.bop, Tile.dot, TileShape.indexToList, BlockPtr.inBounds]
  -- what survives: the three block pointers' `Op.MemorySafe` (vacuous for a
  -- register reference, but `simp` will not peel it), the two load windows'
  -- lane bounds, the `dot` node, and the store window's lane bounds
  refine ⟨⟨?_, fun a b => hA (a, b, PUnit.unit)⟩,
    ⟨?_, fun a b => hB (a, b, PUnit.unit)⟩, ?_, ?_,
    fun a b => hC (a, b, PUnit.unit)⟩ <;>
    first
      | simp [Op.MemorySafe]
      | simp [Op.SafeAt.eq_def]

/-- Region-model run of the f32 branch, in the shape
`MaskedTileShapedKernelIO₂.Implements.intro` consumes. -/
theorem matmul_tma_f32_region_run (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_M BLOCK_N BLOCK_K : Nat)
    (hInj : Function.Injective
      (cOffset (BLOCK_M := BLOCK_M) (BLOCK_N := BLOCK_N) stride_cm stride_cn))
    (s₀ : BlockState)
    (xs : TileIndex [BLOCK_M, BLOCK_K] → ℝ)
    (ys : TileIndex [BLOCK_K, BLOCK_N] → ℝ)
    (hx : ∀ k : TileIndex [BLOCK_M, BLOCK_K],
      s₀.readMem A (k.1.val * stride_am + k.2.1.val * stride_ak) = xs k)
    (hy : ∀ k : TileIndex [BLOCK_K, BLOCK_N],
      s₀.readMem B (k.1.val * stride_bk + k.2.1.val * stride_bn) = ys k) :
    ∃ s1, exec (matmul_tma_f32_surface A B C M N K stride_am stride_ak stride_bk
        stride_bn stride_cm stride_cn BLOCK_M BLOCK_N BLOCK_K) s₀ = some s1
      ∧ (∀ idx : TileIndex [BLOCK_M, BLOCK_N],
          s1.readMem C (cOffset stride_cm stride_cn idx)
            = matmulSpecOf BLOCK_M BLOCK_N BLOCK_K xs ys idx)
      ∧ (∀ (r : RegionName) (o : Nat),
          (r ≠ C ∨ ∀ idx : TileIndex [BLOCK_M, BLOCK_N],
            o ≠ cOffset stride_cm stride_cn idx) →
          s1.mem r o = s₀.mem r o) := by
  obtain ⟨s1, hexec⟩ := matmul_tma_f32_terminates A B C M N K stride_am
    stride_ak stride_bk stride_bn stride_cm stride_cn BLOCK_M BLOCK_N BLOCK_K s₀
  refine ⟨s1, hexec, ?_, matmul_tma_f32_frame A B C M N K stride_am stride_ak
    stride_bk stride_bn stride_cm stride_cn BLOCK_M BLOCK_N BLOCK_K s₀ s1 hexec⟩
  intro idx
  have h := matmul_tma_f32_exec_closed_form A B C s₀ M N K stride_am stride_ak
    stride_bk stride_bn stride_cm stride_cn BLOCK_M BLOCK_N BLOCK_K hInj idx
  have h' : s1.readMem C (cOffset stride_cm stride_cn idx)
      = matmulSpec s₀ A B stride_am stride_ak stride_bk stride_bn BLOCK_K
          idx.1.val idx.2.1.val := by
    simpa [hexec] using h
  rw [h', matmulSpec_eq_of A B stride_am stride_ak stride_bk stride_bn BLOCK_M
    BLOCK_N BLOCK_K s₀ xs ys hx hy idx]

/-- IO signature of the f32 branch on the **per-channel-shape** surface: `A` on
`[BLOCK_M, BLOCK_K]`, `B` on `[BLOCK_K, BLOCK_N]`, `C` on `[BLOCK_M, BLOCK_N]`,
every lane of all three active (the block pointers carry no `boundary_check`),
and the addresses are the block pointers' offset-`(0,0)` maps. -/
def matmulTmaF32IO (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_M BLOCK_N BLOCK_K : Nat) : MaskedTileShapedKernelIO₂ where
  kernel := matmul_tma_f32_surface A B C M N K stride_am stride_ak stride_bk
    stride_bn stride_cm stride_cn BLOCK_M BLOCK_N BLOCK_K
  in1 := A
  in2 := B
  out := C
  shape1 := [BLOCK_M, BLOCK_K]
  shape2 := [BLOCK_K, BLOCK_N]
  shapeOut := [BLOCK_M, BLOCK_N]
  read1 := fun _p₀ _p₁ k => k.1.val * stride_am + k.2.1.val * stride_ak
  read2 := fun _p₀ _p₁ k => k.1.val * stride_bk + k.2.1.val * stride_bn
  write := fun _p₀ _p₁ o => cOffset stride_cm stride_cn o
  mask1 := fun _p₀ _p₁ _ => True
  mask2 := fun _p₀ _p₁ _ => True
  writeMask := fun _p₀ _p₁ _ => True

/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/

/-- **The headline on the IO surface** for `matmul_tma.py`'s
`matmul_tma_load_store`, `OUTPUT_F16 = false` branch: for every disjoint flat
placement of the three buffers, and every launch state whose `A` and `B` tiles are
pinned, the translated pointer kernel terminates, every cell of the output tile
holds the genuine GEMM value `Σ_{e < BLOCK_K} xs[i, e] · ys[e, j]`, and every
other memory cell is unchanged.

This is the first **contraction** on an `io ⊨ f` face, and it is what the
per-channel-shape skin exists for: the three channels live on three different lane
sets (`[BLOCK_M, BLOCK_K]`, `[BLOCK_K, BLOCK_N]`, `[BLOCK_M, BLOCK_N]`), related
only through `f`, which reads both inputs at lanes the output index does not name.

Dimension-general in `M`, `N`, `K`, all six strides and all three block sizes.
Honest side-condition: output-address injectivity (`hInj`) — the same hypothesis
the closed-form theorems take, dischargeable for row-major `C` by
`cOffset_injective_of_rowMajor`. -/
specification matmul_tma_f32_io_correctness (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_M BLOCK_N BLOCK_K : Nat)
    (hInj : Function.Injective
      (cOffset (BLOCK_M := BLOCK_M) (BLOCK_N := BLOCK_N) stride_cm stride_cn)) :
    matmulTmaF32IO A B C M N K stride_am stride_ak stride_bk stride_bn stride_cm
        stride_cn BLOCK_M BLOCK_N BLOCK_K
      ⊨ fun _p₀ _p₁ xs ys idx =>
          matmulSpecOf BLOCK_M BLOCK_N BLOCK_K xs ys idx := by
  refine MaskedTileShapedKernelIO₂.Implements.intro _ ?_ ?_ ?_
  · exact matmul_tma_f32_flattenOk A B C M N K stride_am stride_ak stride_bk
      stride_bn stride_cm stride_cn BLOCK_M BLOCK_N BLOCK_K
  · intro bounds s h1 h2 h3
    exact matmul_tma_f32_traceSafe A B C M N K stride_am stride_ak stride_bk
      stride_bn stride_cm stride_cn BLOCK_M BLOCK_N BLOCK_K bounds s
      (fun idx => h1 idx trivial) (fun idx => h2 idx trivial)
      (fun idx => h3 idx trivial)
  · intro s₀ xs ys hx hy
    obtain ⟨s1, hexec, hval, hframe⟩ :=
      matmul_tma_f32_region_run A B C M N K stride_am stride_ak stride_bk
        stride_bn stride_cm stride_cn BLOCK_M BLOCK_N BLOCK_K hInj s₀ xs ys
        (fun k => hx k trivial) (fun k => hy k trivial)
    exact ⟨s1, hexec, fun o _ => hval o,
      fun r o hcond => hframe r o (by
        rcases hcond with h | h
        · exact Or.inl h
        · exact Or.inr fun idx => h idx trivial)⟩

/-! ### ════════ `⊨[R]` rounding face for the **fp16** branch ════════

Every `⊨[R]` face on a **port** in `bench/tritonbench_g` so far says *the kernel
introduces no rounding event of its own*: the terminal cast erases to `.real`
(`.to(X.dtype.element_ty)`, and also `.to(tl.float32)`, which the DSL
special-cases to `Op.castFloat _ FloatDType.real` — the bit-accurate fp32
channel is the separate `round_to` surface), so `execR R = exec` and the exact
run transports verbatim.

**This branch is different.** `.to(tl.float16)` falls through to the generic arm
and becomes `Op.castFloat _ FloatDType.fp16`, which *is* a rounding site. So the
face below is not a transport: it states

  `C[i, j] = R.round .fp16 (Σ_{e < BLOCK_K} xs[i, e] · ys[e, j])`

— the exact ℝ contraction, quantized once onto the fp16 grid.

Two precedents, so this is not overclaimed:

* the showcase `bench/examples/FloatDTypeCorrect.lean` already carries a
  rounding-real `⊨[R]`, at `.fp32` via the **explicit** quantization spelling
  `.round_to(...)`;
* ten ports (the attention family, `triton_conv2d_fwd`, `triton_attention`) do
  have `⊨[R]` faces over fp16 kernels — but they carry the hypothesis
  `hfp16 : R.round .fp16 = id`, i.e. they **assume the fp16 rounding away** and
  declare it their modeling boundary.

What is new here is that this face *states* the fp16 quantization instead of
assuming it away or erasing it: no `hfp16`, `outDType := .fp16`, and the
rounded quantity is a **contraction** rather than an elementwise add. The same
technique is what would let those ten ports drop `hfp16`.

The value is in fact rounded **twice** — once by the `.to(tl.float16)` cast
(site 1) and again by the fp16 typed store (site 2) — and
`RoundingModel.round_idem`, a *defining* field of the model rather than an opt-in
mixin, is exactly what collapses the pair to the single boundary round the
headline states. -/

/-- `evalOpR` on the block-pointer constructor: R-independent (no float
arithmetic), the R-mirror of `makeBlockPtr_eval`. -/
private theorem makeBlockPtr_evalR (Rm : RoundingModel) (s : BlockState)
    (Reg : RegionName) (parentR parentC strideR strideC BR BC : Nat) :
    evalOpR Rm (Op.makeBlockPtrDyn Reg (Op.constNat 0) [parentR, parentC] [BR, BC]
      [strideR, strideC] [0, 0]) s
      = some (⟨fun _ : TileIndex [BR, BC] =>
          { region := Reg, baseOffset := 0, parentShape := [parentR, parentC],
            blockShape := [BR, BC], strides := [strideR, strideC],
            offsets := [0, 0] }⟩ : Tile .blockPtr [BR, BC]) := by
  simp [evalOpR, Option.bind]

/-- `evalOpR` on an unguarded block-pointer load: R-independent, the R-mirror of
`load_blockPtr_eval`. -/
private theorem load_blockPtr_evalR {BR BC : Nat} (Rm : RoundingModel)
    (s : BlockState) (Reg : RegionName)
    (parentR parentC strideR strideC : Nat) (bpName : RegName)
    (hbp : s.regs .blockPtr [BR, BC] bpName
      = some (⟨fun _ : TileIndex [BR, BC] =>
          { region := Reg, baseOffset := 0, parentShape := [parentR, parentC],
            blockShape := [BR, BC], strides := [strideR, strideC],
            offsets := [0, 0] }⟩ : Tile .blockPtr [BR, BC])) :
    evalOpR Rm (.load .real (.blockPtr (Op.ref .blockPtr [BR, BC] bpName) []) .none) s
      = some (⟨fun idx : TileIndex [BR, BC] =>
          some (s.readMem Reg (idx.1.val * strideR + idx.2.1.val * strideC))⟩
          : Tile .real [BR, BC]) := by
  simp only [evalOpR, evalOpR_ref, hbp, Option.bind_some, Option.bind, Option.map]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, j, u⟩ := idx
  simp only [TileShape.indexToList, BlockState.readMemValue_real, BlockPtr.inBounds,
    List.all_nil, if_true, BlockPtr.address_2d_zero_offsets, Nat.zero_add]

/-- **The rounding site.** `evalOpR` on `(tl.dot a b).to(tl.float16)`: the `dot`
itself is R-independent, and the cast applies the model's fp16 rounding via
`R.cast`. Compare `castdot_eval`, whose cast is the exact `FloatDType.cast`
round-trip. -/
private theorem castdot_evalR {BM BN BLOCK_K : Nat} (Rm : RoundingModel)
    (s : BlockState)
    (at_ : Tile .real [BM, BLOCK_K]) (bt : Tile .real [BLOCK_K, BN])
    (ha : s.regs .real [BM, BLOCK_K] "a" = some at_)
    (hb : s.regs .real [BLOCK_K, BN] "b" = some bt) :
    evalOpR Rm (Op.castFloat FloatDType.real FloatDType.fp16
        (Op.dot (batch := []) (Op.ref .real [BM, BLOCK_K] "a")
          (Op.ref .real [BLOCK_K, BN] "b"))) s
      = some (⟨fun idx => Rm.cast FloatDType.real FloatDType.fp16
            ((Tile.dot [] at_ bt).data idx)⟩
          : Tile FloatDType.fp16.toTileDType ([] ++ [BM, BN])) := by
  have hd : evalOpR Rm (Op.dot (batch := []) (Op.ref .real [BM, BLOCK_K] "a")
      (Op.ref .real [BLOCK_K, BN] "b")) s = some (Tile.dot [] at_ bt) := by
    simp only [evalOpR]
    simp [evalOpR_ref, ha, hb]
  rw [evalOpR_castFloat]
  erw [hd]
  rfl

/-- The fp16 block-pointer store **under the rounding model**: `writeMemTypedR`
at `.fp16` is `writeMemAsR`, so every written cell carries the model's stored
value, and untouched cells are framed by `foldl_writeMemAsR_preserve_cell`. -/
private theorem store_blockPtr_fp16_readbackR {BR BC : Nat} (Rm : RoundingModel)
    (s : BlockState) (Reg : RegionName) (parentR parentC strideR strideC : Nat)
    (cpName cName : RegName) (vt : Tile .fp16 [BR, BC])
    (hcp : s.regs .blockPtr [BR, BC] cpName
      = some (⟨fun _ : TileIndex [BR, BC] =>
          { region := Reg, baseOffset := 0, parentShape := [parentR, parentC],
            blockShape := [BR, BC], strides := [strideR, strideC],
            offsets := [0, 0] }⟩ : Tile .blockPtr [BR, BC]))
    (hc : s.regs .fp16 [BR, BC] cName = some vt)
    (offsetFn : TileIndex [BR, BC] → Nat)
    (hoff : ∀ idx : TileIndex [BR, BC],
      offsetFn idx = idx.1.val * strideR + idx.2.1.val * strideC)
    (hInj : Function.Injective offsetFn) :
    ∃ s', stepStmtR Rm (Stmt.store .fp16 [BR, BC]
        (.blockPtr (Op.ref .blockPtr [BR, BC] cpName) [])
        (Op.ref .fp16 [BR, BC] cName) .none) s = some s'
      ∧ (∀ idx : TileIndex [BR, BC],
          s'.mem Reg (offsetFn idx)
            = MemCell.of FloatDType.fp16.toTileDType
                (FloatDType.fp16.ofReal (Rm.storeValue FloatDType.fp16 (vt.data idx))))
      ∧ (∀ (r : RegionName) (o : Nat),
          (r ≠ Reg ∨ ∀ idx : TileIndex [BR, BC], o ≠ offsetFn idx) →
          s'.mem r o = s.mem r o) := by
  set sfin := (TileShape.allIndices [BR, BC]).foldl
      (fun acc i => acc.writeMemAsR Rm FloatDType.fp16 Reg (offsetFn i) (vt.data i))
      s with hsfin
  have hstep : stepStmtR Rm (Stmt.store .fp16 [BR, BC]
        (.blockPtr (Op.ref .blockPtr [BR, BC] cpName) [])
        (Op.ref .fp16 [BR, BC] cName) .none) s = some sfin := by
    simp only [stepStmtR, evalOpR_ref, hc, hcp, Option.bind_some, bind,
      BlockState.writeMemTypedR_fp16]
    refine congrArg some ?_
    rw [hsfin]
    apply List.foldl_ext
    intro acc i _
    obtain ⟨ii, jj, u⟩ := i
    rw [show TileShape.indexToList [BR, BC] (ii, jj, PUnit.unit) = [ii.val, jj.val] by
          simp [TileShape.indexToList]]
    simp only [BlockPtr.inBounds, List.all_nil, Bool.and_true, if_true,
      BlockPtr.address_2d_zero_offsets, Nat.zero_add, hoff]
  refine ⟨sfin, hstep, ?_, ?_⟩
  · intro idx
    rw [hsfin]
    exact BlockState.scatter_memcell_R_nd Rm FloatDType.fp16 (region := Reg) s
      offsetFn (fun i => vt.data i) hInj idx
  · intro r o hcond
    rw [hsfin]
    refine BlockState.foldl_writeMemAsR_preserve_cell Rm FloatDType.fp16
      offsetFn (fun i => vt.data i) r o (TileShape.allIndices [BR, BC]) ?_ s
    intro k _ hk
    rcases hcond with h | h
    · exact h hk.1.symm
    · exact h k hk.2.symm

set_option maxHeartbeats 1000000 in
/-- Region-model run of the fp16 branch **under `execR R`**: the exec-side
analogue of `matmul_tma_f16_exec_closed_form`, walking the same body with the
library's R stepping lemmas. Every output cell carries the model's fp16 stored
value of `R.cast .real .fp16 (Σ_e A·B)`. -/
theorem matmul_tma_f16_region_runR (Rm : RoundingModel) (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_M BLOCK_N BLOCK_K : Nat)
    (hInj : Function.Injective
      (cOffset (BLOCK_M := BLOCK_M) (BLOCK_N := BLOCK_N) stride_cm stride_cn))
    (s : BlockState) :
    ∃ s1, execR Rm ((matmul_tma_f16_surface A B C M N K stride_am stride_ak
        stride_bk stride_bn stride_cm stride_cn BLOCK_M BLOCK_N
        BLOCK_K).toAlgKernel) s = some s1
      ∧ (∀ idx : TileIndex [BLOCK_M, BLOCK_N],
          s1.mem C (cOffset stride_cm stride_cn idx)
            = MemCell.of FloatDType.fp16.toTileDType
                (FloatDType.fp16.ofReal (Rm.storeValue FloatDType.fp16
                  (Rm.cast FloatDType.real FloatDType.fp16
                    (some (matmulSpec s A B stride_am stride_ak stride_bk
                      stride_bn BLOCK_K idx.1.val idx.2.1.val))))))
      ∧ (∀ (r : RegionName) (o : Nat),
          (r ≠ C ∨ ∀ idx : TileIndex [BLOCK_M, BLOCK_N],
            o ≠ cOffset stride_cm stride_cn idx) →
          s1.mem r o = s.mem r o) := by
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
  rw [show execR Rm ((matmul_tma_f16_surface A B C M N K stride_am stride_ak
        stride_bk stride_bn stride_cm stride_cn BLOCK_M BLOCK_N
        BLOCK_K).toAlgKernel) s
      = stepStmtsR Rm ((matmul_tma_f16_surface A B C M N K stride_am stride_ak
          stride_bk stride_bn stride_cm stride_cn BLOCK_M BLOCK_N
          BLOCK_K).toAlgKernel.body) s from rfl]
  rw [show (matmul_tma_f16_surface A B C M N K stride_am stride_ak
          stride_bk stride_bn stride_cm stride_cn BLOCK_M BLOCK_N
          BLOCK_K).toAlgKernel.body
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
  rw [stepStmtsR_cons_some (stepStmtR_assign_eq_some
        (makeBlockPtr_evalR Rm s A M K stride_am stride_ak BLOCK_M BLOCK_K))]
  rw [stepStmtsR_cons_some (stepStmtR_assign_eq_some
        (makeBlockPtr_evalR Rm _ B K N stride_bk stride_bn BLOCK_K BLOCK_N))]
  rw [stepStmtsR_cons_some (stepStmtR_assign_eq_some
        (makeBlockPtr_evalR Rm _ C M N stride_cm stride_cn BLOCK_M BLOCK_N))]
  rw [stepStmtsR_cons_some (stepStmtR_assign_eq_some
        (load_blockPtr_evalR (BR := BLOCK_M) (BC := BLOCK_K) Rm _ A M K stride_am stride_ak
          "a_block_ptr" (by simp)))]
  rw [stepStmtsR_cons_some (stepStmtR_assign_eq_some
        (load_blockPtr_evalR (BR := BLOCK_K) (BC := BLOCK_N) Rm _ B K N stride_bk stride_bn
          "b_block_ptr" (by simp)))]
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
  rw [stepStmtsR_cons_some (stepStmtR_assign_eq_some (castdot_evalR Rm s5 aT bT ha5 hb5))]
  set cTile : Tile .fp16 [BLOCK_M, BLOCK_N] :=
    ⟨fun idx => Rm.cast FloatDType.real FloatDType.fp16 ((Tile.dot [] aT bT).data idx)⟩
    with hcTile
  have hdotval : ∀ idx : TileIndex [BLOCK_M, BLOCK_N],
      cTile.data idx = Rm.cast FloatDType.real FloatDType.fp16
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
  obtain ⟨sfin, hstore, hread, hframe⟩ :=
    store_blockPtr_fp16_readbackR (BR := BLOCK_M) (BC := BLOCK_N) Rm
      (s5.setReg "c" FloatDType.fp16.toTileDType ([] ++ [BLOCK_M, BLOCK_N])
        ⟨fun idx => Rm.cast FloatDType.real FloatDType.fp16 ((Tile.dot [] aT bT).data idx)⟩)
      C M N stride_cm stride_cn "c_block_ptr" "c" cTile
      (by rw [hcp5.symm]; simp) (by simp [hcTile])
      (cOffset stride_cm stride_cn) (fun idx => by simp [cOffset]) hInj
  refine ⟨sfin, ?_, ?_, ?_⟩
  · rw [stepStmtsR_cons_some hstore, stepStmtsR_nil]
  · intro idx
    rw [hread idx, hdotval idx]
  · intro r o hcond
    rw [hframe r o hcond, hs5]
    simp

/-- The fp16 branch sits inside the flat-memory bridge's covered fragment. -/
theorem matmul_tma_f16_flattenOk (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_M BLOCK_N BLOCK_K : Nat) :
    ((matmul_tma_f16_surface A B C M N K stride_am stride_ak stride_bk stride_bn
      stride_cm stride_cn BLOCK_M BLOCK_N BLOCK_K).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [matmul_tma_f16_surface, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, StmtList.FlattenOk,
    Stmt.FlattenOk, Op.FlattenOk]
  simp [Op.FlattenOk.eq_def]

/-- Per-execution safety walk of the fp16 branch **under the rounding model**:
the three block pointers carry no `boundary_check`, so every lane of both loads
and of the store must be in bounds. -/
theorem matmul_tma_f16_traceSafeR (Rm : RoundingModel) (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_M BLOCK_N BLOCK_K : Nat) (bounds : RegionBounds) (s : BlockState)
    (hA : ∀ idx : TileIndex [BLOCK_M, BLOCK_K],
      idx.1.val * stride_am + idx.2.1.val * stride_ak < bounds A)
    (hB : ∀ idx : TileIndex [BLOCK_K, BLOCK_N],
      idx.1.val * stride_bk + idx.2.1.val * stride_bn < bounds B)
    (hC : ∀ idx : TileIndex [BLOCK_M, BLOCK_N],
      cOffset stride_cm stride_cn idx < bounds C) :
    Kernel.TraceSafeR Rm bounds
      ((matmul_tma_f16_surface A B C M N K stride_am stride_ak stride_bk
        stride_bn stride_cm stride_cn BLOCK_M BLOCK_N BLOCK_K).toAlgKernel) s := by
  simp [Kernel.TraceSafeR, matmul_tma_f16_surface, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Stmt.TraceSafeListR,
    Stmt.TraceSafeR, Op.SafeAtR, MaskOpt.SafeAtR, MaskOpt.ActiveR,
    MemAccess.SafeAtR, MemAccess.ActiveAddressSafeR,
    memAccessActiveAddressSafeR, stepStmtsR, stepStmtR, evalOpR,
    evalOpR.eq_def, Option.bind, Option.map, Tile.bop, Tile.dot,
    TileShape.indexToList, BlockPtr.inBounds]
  and_intros
  all_goals try exact fun a b => hA (a, b, PUnit.unit)
  all_goals try exact fun a b => hB (a, b, PUnit.unit)
  all_goals try exact fun a b => hC (a, b, PUnit.unit)
  all_goals try simp [Op.SafeAtR.eq_def]

/-- IO signature of the fp16 branch on the per-channel-shape surface — the same
windows as `matmulTmaF32IO`, on the fp16 kernel. -/
def matmulTmaF16IO (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_M BLOCK_N BLOCK_K : Nat) : MaskedTileShapedKernelIO₂ where
  kernel := matmul_tma_f16_surface A B C M N K stride_am stride_ak stride_bk
    stride_bn stride_cm stride_cn BLOCK_M BLOCK_N BLOCK_K
  in1 := A
  in2 := B
  out := C
  shape1 := [BLOCK_M, BLOCK_K]
  shape2 := [BLOCK_K, BLOCK_N]
  shapeOut := [BLOCK_M, BLOCK_N]
  read1 := fun _p₀ _p₁ k => k.1.val * stride_am + k.2.1.val * stride_ak
  read2 := fun _p₀ _p₁ k => k.1.val * stride_bk + k.2.1.val * stride_bn
  write := fun _p₀ _p₁ o => cOffset stride_cm stride_cn o
  mask1 := fun _p₀ _p₁ _ => True
  mask2 := fun _p₀ _p₁ _ => True
  writeMask := fun _p₀ _p₁ _ => True

/-! ### ════════ ★ MAIN THEOREM (rounding face) ★ ════════ -/

/-- **The `⊨[R]` headline** for `matmul_tma.py`'s `matmul_tma_load_store`,
`OUTPUT_F16 = true` branch: for **every** rounding model `R`, every disjoint flat
placement of the three buffers, and every launch state whose `A` and `B` tiles are
pinned, the kernel run under `execR R` terminates, every cell of the output tile
reads back at `.fp16` holding

  `R.round .fp16 (Σ_{e < BLOCK_K} xs[i, e] · ys[e, j])`

and every other memory cell is unchanged.

Unlike every other `⊨[R]` face on a port in `bench/tritonbench_g`, this one is
**not** a transport of an exact result: `.to(tl.float16)` is a genuine rounding
site, so the `R.round .fp16` on the right is the quantization the kernel actually
performs (the showcase `bench/examples/FloatDTypeCorrect.lean` is the existing
precedent for a rounding-real `⊨[R]`, there via explicit `.round_to`). `f` remains the exact ℝ contraction — the face's content is precisely
"the fp16 output is the real GEMM value, rounded once".

The kernel rounds the value **twice** (the cast, then the fp16 typed store);
`RoundingModel.round_idem` collapses that to the single round stated here.

Dimension-general in `M`, `N`, `K`, all six strides and all three block sizes.
Honest side-condition: output-address injectivity (`hInj`) — the same hypothesis
the exact closed forms take, dischargeable for row-major `C` by
`cOffset_injective_of_rowMajor`. -/
specification matmul_tma_f16_io_correctnessR (Rm : RoundingModel)
    (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_M BLOCK_N BLOCK_K : Nat)
    (hInj : Function.Injective
      (cOffset (BLOCK_M := BLOCK_M) (BLOCK_N := BLOCK_N) stride_cm stride_cn)) :
    matmulTmaF16IO A B C M N K stride_am stride_ak stride_bk stride_bn stride_cm
        stride_cn BLOCK_M BLOCK_N BLOCK_K
      ⊨[Rm, FloatDType.fp16] fun _p₀ _p₁ xs ys idx =>
          matmulSpecOf BLOCK_M BLOCK_N BLOCK_K xs ys idx := by
  refine MaskedTileShapedKernelIO₂.ImplementsR.intro _ ?_ ?_ ?_
  · exact matmul_tma_f16_flattenOk A B C M N K stride_am stride_ak stride_bk
      stride_bn stride_cm stride_cn BLOCK_M BLOCK_N BLOCK_K
  · intro bounds s h1 h2 h3
    exact matmul_tma_f16_traceSafeR Rm A B C M N K stride_am stride_ak stride_bk
      stride_bn stride_cm stride_cn BLOCK_M BLOCK_N BLOCK_K bounds s
      (fun idx => h1 idx trivial) (fun idx => h2 idx trivial)
      (fun idx => h3 idx trivial)
  · intro s₀ xs ys hx hy
    obtain ⟨s1, hexec, hread, hframe⟩ :=
      matmul_tma_f16_region_runR Rm A B C M N K stride_am stride_ak stride_bk
        stride_bn stride_cm stride_cn BLOCK_M BLOCK_N BLOCK_K hInj s₀
    refine ⟨s1, hexec, ?_, fun r o hcond => hframe r o (by
      rcases hcond with h | h
      · exact Or.inl h
      · exact Or.inr fun idx => h idx trivial)⟩
    intro idx _
    simp only [matmulTmaF16IO]
    -- the read-back: the stored cell is the fp16 store value of the fp16 cast of
    -- the ℝ contraction; `round_idem` collapses cast-then-store to one round
    rw [show matmulSpecOf BLOCK_M BLOCK_N BLOCK_K xs ys idx
        = matmulSpec s₀ A B stride_am stride_ak stride_bk stride_bn BLOCK_K
            idx.1.val idx.2.1.val from
      (matmulSpec_eq_of A B stride_am stride_ak stride_bk stride_bn BLOCK_M
        BLOCK_N BLOCK_K s₀ xs ys (fun k => hx k trivial) (fun k => hy k trivial)
        idx).symm]
    unfold BlockState.readMemAs
    rw [hread idx]
    simp [MemCell.readAs_of_same, FloatDType.ofReal, FloatDType.storeValue,
      RoundingModel.storeValue, RoundingModel.cast, Rm.round_idem]

end IOFace

end VeriTile.Bench.TritonBenchG.MatmulTma
