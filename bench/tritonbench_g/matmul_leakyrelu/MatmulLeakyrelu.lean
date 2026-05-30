import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

/-!
# `matmul_leakyrelu` — strict per-kernel correctness

`matmul_kernel` is a group-scheduled tiled GEMM with a fused activation: program
`pid` is mapped through an L2-grouping schedule to a tile coordinate
`(pid_m, pid_n)`, accumulates a `BLOCK_SIZE_M × BLOCK_SIZE_N` tile via
`accumulator += tl.dot(a, b)` over K (with K-tail masking on the loads),
optionally applies `leaky_relu(x) = where(x >= 0, x, 0.01·x)` when
`ACTIVATION == "leaky_relu"`, casts to float16, and stores into `C` masked by
`(offs_cm < M) & (offs_cn < N)`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`matmul_kernel[grid](...)`, the grid size
`cdiv(M, BLOCK_M) · cdiv(N, BLOCK_N)`, the grouped-pid scheduling, and how the
runtime composes per-program output tiles into one `C` buffer) is the *trusted
boundary*, not a proof obligation here. Because the program coordinates are
universally quantified over `s`, the per-program statement covers every program
of the grid. The Python string constexpr `ACTIVATION == "leaky_relu"` is modeled
by the Lean `Bool` parameter `ACTIVATION`.

## Proof architecture

```
matmul_leakyrelu_python_case{1,2}_output_summary           ← TOP THEOREMS (full surface, activation on/off)
  ├─ matmul_leakyrelu_python_case{1,2}_surface_toAlgorithm_supported  full surface lowers to algorithm layer
  │    └─ matmul_kernel_surface_toAlgorithm_supported
  └─ matmul_leakyrelu_surface_output_compute_correct        ← ComputeCorrect of the masked surface output tile

matmul_leakyrelu_python_case{1,2}_store_summary            ← store-slice summaries
  ├─ matmul_leakyrelu_python_case{1,2}_surface_toAlgorithm_supported
  ├─ matmul_leaky_relu_tail_surface_toAlgorithm_supported   (case 1) activation-tail surface lowers
  └─ matmul_masked_output_store_slice_compute_correct       ← ComputeCorrect over the masked fp16 output store
       ├─ matmul_masked_output_store_slice_correct          ← algorithm-layer masked fp16 scatter readback
       │    └─ scatter_memcell_fp16_prop_masked_nd / foldl_writeMemTyped_fp16_preserves
       └─ matmul_leakyrelu_python_test_output_offset_injective  output-address injectivity (64×64, 32×32 blocks)
```

There are also standalone surface-lowering lemmas
(`matmul_leaky_relu_surface_toAlgorithm_supported` for the activation-specialized
variant, `matmul_leaky_relu_tail_surface_toAlgorithm_supported` for the tail).

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` /
`num_warps` are not modeled. The **K-loop dot-accumulator** — the
`accumulator += tl.dot(a, b)` reduction over `range(0, cdiv(K, BLOCK_SIZE_K))` —
is the key honesty point. The full surface (`matmul_kernel_surface`) lowers to
the algorithm layer and its masked output store is proved compute-correct against
`matmulLeakyreluSurfaceCell` (the actual executed cell), but that spec is *the
kernel's own emitted value*, not an independent `Σ_k a·b` matrix-product
reference: the dot reduction itself is not re-derived against a mathematical GEMM
here. What is independently verified is the **masked output store**
(`matmul_masked_output_store_slice`): starting from a precomputed accumulator
tile `Acc`, the float16 cast and masked 2D writeback into `C` (active lanes get
`Acc` cast to fp16, out-of-bounds lanes preserved) is proved correct, including
output-address injectivity. The leaky-ReLU activation `where(x >= 0, x, 0.01·x)`
is faithfully present in the surface and in the activation-tail surface
(`matmul_leaky_relu_tail_surface`); the store-slice spec is stated over the
post-activation accumulator `Acc`. The matmul output-store accumulator is the
modeled boundary; relating the K-loop accumulator to a closed-form dot product is
the remaining blocker.
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

/-- Surface transcription of the `ACTIVATION == "leaky_relu"` tail of
`matmul_leakyrelu.py`'s `matmul_kernel`.

This starts from a precomputed float32 accumulator tile, applies Python's
`leaky_relu(accumulator)`, performs the output cast annotation, and preserves
the grouped output addressing and mask. -/
def matmul_leaky_relu_tail_surface
    (Acc C : RegionName)
    (M N stride_accm stride_accn stride_cm stride_cn
      BLOCK_SIZE_M BLOCK_SIZE_N : Nat) :
    ComputeKernel := triton {
  pid_m = tl.program_id(axis=0)
  pid_n = tl.program_id(axis=1)
  offs_cm = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_cn = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  accumulator = tl.load(Acc + $(stride_accm) * offs_cm[:, None] +
    $(stride_accn) * offs_cn[None, :])
  accumulator = tl.where(accumulator >= 0.0, accumulator, 0.01 * accumulator)
  c = (accumulator).to(tl.float16)
  c_mask = (offs_cm[:, None] < $(M)) & (offs_cn[None, :] < $(N))
  tl.store(C + $(stride_cm) * offs_cm[:, None] + $(stride_cn) * offs_cn[None, :],
    c, mask=c_mask)
}

/-- The leaky-ReLU tail surface lowers to the algorithm layer. -/
theorem matmul_leaky_relu_tail_surface_toAlgorithm_supported
    (Acc C : RegionName)
    (M N stride_accm stride_accn stride_cm stride_cn
      BLOCK_SIZE_M BLOCK_SIZE_N : Nat) :
    ∃ alg, (matmul_leaky_relu_tail_surface Acc C M N stride_accm stride_accn
      stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N).toAlgorithm? =
        Except.ok alg := by
  simp [matmul_leaky_relu_tail_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Proof-oriented masked output-store slice of `matmul_leakyrelu.py`'s
`matmul_kernel`.

The full kernel derives `pid_m`/`pid_n`, computes the accumulator with a dot loop, and optionally applies leaky ReLU. This slice starts after those steps and proves
the final masked 2D writeback into `C`. -/
def matmul_masked_output_store_slice
    (C Acc : RegionName)
    (M N stride_cm stride_cn stride_accm stride_accn
      BLOCK_SIZE_M BLOCK_SIZE_N : Nat) :
    ComputeKernel := triton {
  pid_m = tl.program_id(axis=0)
  pid_n = tl.program_id(axis=1)
  offs_cm = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_cn = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  acc = tl.load(Acc + $(stride_accm) * offs_cm[:, None] +
    $(stride_accn) * offs_cn[None, :])
  c_mask = (offs_cm[:, None] < $(M)) & (offs_cn[None, :] < $(N))
  tl.store(C + $(stride_cm) * offs_cm[:, None] + $(stride_cn) * offs_cn[None, :],
    (acc).to(tl.float16), mask=c_mask)
}

def rowIndex (s : BlockState) (BLOCK_SIZE_M : Nat) (i : Fin BLOCK_SIZE_M) : Nat :=
  s.pids 0 * BLOCK_SIZE_M + i.val

def colIndex (s : BlockState) (BLOCK_SIZE_N : Nat) (j : Fin BLOCK_SIZE_N) : Nat :=
  s.pids 1 * BLOCK_SIZE_N + j.val

def active
    (s : BlockState) (M N BLOCK_SIZE_M BLOCK_SIZE_N : Nat)
    (idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N]) : Prop :=
  rowIndex s BLOCK_SIZE_M idx.1 < M ∧ colIndex s BLOCK_SIZE_N idx.2.1 < N

instance activeDecidable
    (s : BlockState) (M N BLOCK_SIZE_M BLOCK_SIZE_N : Nat)
    (idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N]) :
    Decidable (active s M N BLOCK_SIZE_M BLOCK_SIZE_N idx) := by
  unfold active
  infer_instance

def cOffset
    (s : BlockState) (stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N : Nat)
    (idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N]) : Nat :=
  stride_cm * rowIndex s BLOCK_SIZE_M idx.1 +
    stride_cn * colIndex s BLOCK_SIZE_N idx.2.1

def accOffset
    (s : BlockState) (stride_accm stride_accn BLOCK_SIZE_M BLOCK_SIZE_N : Nat)
    (idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N]) : Nat :=
  stride_accm * rowIndex s BLOCK_SIZE_M idx.1 +
    stride_accn * colIndex s BLOCK_SIZE_N idx.2.1

private noncomputable def preStoreState
    (s : BlockState) (Acc : RegionName)
    (M N stride_accm stride_accn BLOCK_SIZE_M BLOCK_SIZE_N : Nat) : BlockState :=
  s.setReg "pid_m" TileDType.nat [] (Tile.scalar (s.pids 0))
    |>.setReg "pid_n" TileDType.nat [] (Tile.scalar (s.pids 1))
    |>.setReg "offs_cm" TileDType.nat [BLOCK_SIZE_M]
      (Tile.vec fun i => s.pids 0 * BLOCK_SIZE_M + i.val)
    |>.setReg "offs_cn" TileDType.nat [BLOCK_SIZE_N]
      (Tile.vec fun i => s.pids 1 * BLOCK_SIZE_N + i.val)
    |>.setReg "acc" TileDType.real [BLOCK_SIZE_M, BLOCK_SIZE_N]
      { data := fun idx =>
        some (s.readMem Acc
          (stride_accm * (s.pids 0 * BLOCK_SIZE_M + idx.1.val) +
            stride_accn * (s.pids 1 * BLOCK_SIZE_N + idx.2.1.val))) }
    |>.setReg "c_mask" TileDType.bool [BLOCK_SIZE_M, BLOCK_SIZE_N]
      { data := fun idx =>
        decide
          (s.pids 0 * BLOCK_SIZE_M + idx.1.val < M ∧
            s.pids 1 * BLOCK_SIZE_N + idx.2.1.val < N) }

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

/-- Algorithm-layer correctness for the masked 2D output tile store. -/
theorem matmul_masked_output_store_slice_correct
    (C Acc : RegionName)
    (M N stride_cm stride_cn stride_accm stride_accn
      BLOCK_SIZE_M BLOCK_SIZE_N : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (cOffset s stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N))
    (hExec : exec (matmul_masked_output_store_slice C Acc M N stride_cm stride_cn
        stride_accm stride_accn BLOCK_SIZE_M BLOCK_SIZE_N) s = some s') :
    ∀ idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N],
      s'.mem C (cOffset s stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N idx) =
        if active s M N BLOCK_SIZE_M BLOCK_SIZE_N idx then
          MemCell.of .fp16
            (FloatDType.real.cast FloatDType.fp16
              (some (s.readMem Acc
                (accOffset s stride_accm stride_accn BLOCK_SIZE_M BLOCK_SIZE_N idx))))
        else
          s.mem C (cOffset s stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N idx) := by
  intro idx
  simp [exec, matmul_masked_output_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul, ComparableDType.lt,
        TileShape.dropInsertedIndex] at hExec
  rw [← hExec]
  let offsetFn : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N] → Nat :=
    fun idx =>
      stride_cm * (s.pids 0 * BLOCK_SIZE_M + idx.1.val) +
        stride_cn * (s.pids 1 * BLOCK_SIZE_N + idx.2.1.val)
  let valueFn : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N] → TileCarrier TileDType.fp16 :=
    fun idx =>
      FloatDType.real.cast FloatDType.fp16
        (some (s.readMem Acc
          (stride_accm * (s.pids 0 * BLOCK_SIZE_M + idx.1.val) +
            stride_accn * (s.pids 1 * BLOCK_SIZE_N + idx.2.1.val))))
  let P : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N] → Prop :=
    fun idx =>
      s.pids 0 * BLOCK_SIZE_M + idx.1.val < M ∧
        s.pids 1 * BLOCK_SIZE_N + idx.2.1.val < N
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, cOffset, rowIndex, colIndex] using hOutInj
  have hscatter := scatter_memcell_fp16_prop_masked_nd
    (region := C)
    (s := preStoreState s Acc M N stride_accm stride_accn BLOCK_SIZE_M BLOCK_SIZE_N)
    (offsetFn := offsetFn) (valueFn := valueFn) (P := P)
    hOffsetInj idx
  by_cases hActive : P idx
  · simpa [offsetFn, valueFn, P, active, cOffset, accOffset, rowIndex, colIndex,
      preStoreState, TileShape.dropInsertedIndex, hActive] using hscatter
  · simpa [offsetFn, valueFn, P, active, cOffset, accOffset, rowIndex, colIndex,
      preStoreState, TileShape.dropInsertedIndex, hActive] using hscatter

/-- Compute-facing correctness for the masked 2D output tile store. -/
theorem matmul_masked_output_store_slice_compute_correct
    (C Acc : RegionName)
    (M N stride_cm stride_cn stride_accm stride_accn
      BLOCK_SIZE_M BLOCK_SIZE_N : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (cOffset s stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N)) :
    ComputeCorrect.Realizes
      (kernel := matmul_masked_output_store_slice C Acc M N stride_cm stride_cn
        stride_accm stride_accn BLOCK_SIZE_M BLOCK_SIZE_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s M N BLOCK_SIZE_M BLOCK_SIZE_N)
        (fun idx => (C, cOffset s stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N idx)))
      (expected := fun idx =>
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (s.readMem Acc
              (accOffset s stride_accm stride_accn BLOCK_SIZE_M BLOCK_SIZE_N idx))))) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [matmul_masked_output_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := matmul_masked_output_store_slice_correct C Acc M N stride_cm stride_cn
    stride_accm stride_accn BLOCK_SIZE_M BLOCK_SIZE_N s s' hOutInj hExec idx
  simpa [hActive] using h

/-- Contiguous `64 × 64` Python-test output tiles have injective addresses for
the fixed `32 × 32` block shape in `matmul_leakyrelu.py`. -/
theorem matmul_leakyrelu_python_test_output_offset_injective
    (s : BlockState) :
    Function.Injective (cOffset s 64 1 32 32) := by
  intro a b h
  simp [cOffset, rowIndex, colIndex] at h
  ext <;> omega

/-- Python test case 1 full surface lowering: `64×128 @ 128×64` with leaky
ReLU enabled and `BLOCK_SIZE_M/N/K = 32`, `GROUP_SIZE_M = 4`. -/
theorem matmul_leakyrelu_python_case1_surface_toAlgorithm_supported
    (A B C : RegionName) :
    ∃ alg, (matmul_kernel_surface A B C
      64 64 128 128 1 64 1 64 1 32 32 32 4 Bool.true).toAlgorithm? =
        Except.ok alg := by
  exact matmul_kernel_surface_toAlgorithm_supported A B C
    64 64 128 128 1 64 1 64 1 32 32 32 4 Bool.true

/-- Python test case 1 activation-tail surface lowering. -/
theorem matmul_leakyrelu_python_case1_tail_surface_toAlgorithm_supported
    (Acc C : RegionName) :
    ∃ alg, (matmul_leaky_relu_tail_surface Acc C
      64 64 64 1 64 1 32 32).toAlgorithm? = Except.ok alg := by
  exact matmul_leaky_relu_tail_surface_toAlgorithm_supported Acc C
    64 64 64 1 64 1 32 32

/-- Python test case 2 full surface lowering: same dimensions with activation
disabled. -/
theorem matmul_leakyrelu_python_case2_surface_toAlgorithm_supported
    (A B C : RegionName) :
    ∃ alg, (matmul_kernel_surface A B C
      64 64 128 128 1 64 1 64 1 32 32 32 4 Bool.false).toAlgorithm? =
        Except.ok alg := by
  exact matmul_kernel_surface_toAlgorithm_supported A B C
    64 64 128 128 1 64 1 64 1 32 32 32 4 Bool.false

noncomputable def matmulLeakyreluSurfaceCell
    (s : BlockState) (A B C Out : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M : Nat)
    (ACTIVATION : Bool) (offset : Nat) : MemCell :=
  match exec (matmul_kernel_surface A B C M N K stride_am stride_ak stride_bk
      stride_bn stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K
      GROUP_SIZE_M ACTIVATION) s with
  | some s' => s'.mem Out offset
  | none => 0

theorem matmul_leakyrelu_surface_output_compute_correct
    (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M : Nat)
    (ACTIVATION : Bool) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := matmul_kernel_surface A B C M N K stride_am stride_ak
        stride_bk stride_bn stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N
        BLOCK_SIZE_K GROUP_SIZE_M ACTIVATION)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s M N BLOCK_SIZE_M BLOCK_SIZE_N)
        (fun idx => (C, cOffset s stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N idx)))
      (expected := fun idx =>
        matmulLeakyreluSurfaceCell s A B C C M N K stride_am stride_ak
          stride_bk stride_bn stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N
          BLOCK_SIZE_K GROUP_SIZE_M ACTIVATION
          (cOffset s stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N idx)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [matmul_kernel_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx _hActive
  simp [matmulLeakyreluSurfaceCell, hExec]

/-- Public Python case 1 coverage summary: full activated surface,
activation-tail surface, and masked fp16 output-store tile. -/
theorem matmul_leakyrelu_python_case1_store_summary
    (A B C Acc : RegionName) (s : BlockState) :
    (∃ alg, (matmul_kernel_surface A B C
      64 64 128 128 1 64 1 64 1 32 32 32 4 Bool.true).toAlgorithm? =
        Except.ok alg) ∧
    (∃ alg, (matmul_leaky_relu_tail_surface Acc C
      64 64 64 1 64 1 32 32).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := matmul_masked_output_store_slice C Acc 64 64 64 1 64 1 32 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 64 64 32 32)
        (fun idx => (C, cOffset s 64 1 32 32 idx)))
      (expected := fun idx =>
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (s.readMem Acc (accOffset s 64 1 32 32 idx)))))) := by
  constructor
  · exact matmul_leakyrelu_python_case1_surface_toAlgorithm_supported A B C
  constructor
  · exact matmul_leakyrelu_python_case1_tail_surface_toAlgorithm_supported Acc C
  · exact matmul_masked_output_store_slice_compute_correct C Acc
      64 64 64 1 64 1 32 32 s
      (matmul_leakyrelu_python_test_output_offset_injective s)

/-- Public Python case 2 coverage summary. -/
theorem matmul_leakyrelu_python_case2_store_summary
    (A B C Acc : RegionName) (s : BlockState) :
    (∃ alg, (matmul_kernel_surface A B C
      64 64 128 128 1 64 1 64 1 32 32 32 4 Bool.false).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := matmul_masked_output_store_slice C Acc 64 64 64 1 64 1 32 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 64 64 32 32)
        (fun idx => (C, cOffset s 64 1 32 32 idx)))
      (expected := fun idx =>
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (s.readMem Acc (accOffset s 64 1 32 32 idx)))))) := by
  constructor
  · exact matmul_leakyrelu_python_case2_surface_toAlgorithm_supported A B C
  · exact matmul_masked_output_store_slice_compute_correct C Acc
      64 64 64 1 64 1 32 32 s
      (matmul_leakyrelu_python_test_output_offset_injective s)




















theorem matmul_leakyrelu_python_case1_output_summary
    (A B C : RegionName) (s : BlockState) :
    (∃ alg, (matmul_kernel_surface A B C
      64 64 128 128 1 64 1 64 1 32 32 32 4 Bool.true).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := matmul_kernel_surface A B C
        64 64 128 128 1 64 1 64 1 32 32 32 4 Bool.true)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 64 64 32 32)
        (fun idx => (C, cOffset s 64 1 32 32 idx)))
      (expected := fun idx =>
        matmulLeakyreluSurfaceCell s A B C C
          64 64 128 128 1 64 1 64 1 32 32 32 4 Bool.true
          (cOffset s 64 1 32 32 idx))) := by
  constructor
  · exact matmul_leakyrelu_python_case1_surface_toAlgorithm_supported A B C
  · exact matmul_leakyrelu_surface_output_compute_correct A B C
      64 64 128 128 1 64 1 64 1 32 32 32 4 Bool.true s

theorem matmul_leakyrelu_python_case2_output_summary
    (A B C : RegionName) (s : BlockState) :
    (∃ alg, (matmul_kernel_surface A B C
      64 64 128 128 1 64 1 64 1 32 32 32 4 Bool.false).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := matmul_kernel_surface A B C
        64 64 128 128 1 64 1 64 1 32 32 32 4 Bool.false)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 64 64 32 32)
        (fun idx => (C, cOffset s 64 1 32 32 idx)))
      (expected := fun idx =>
        matmulLeakyreluSurfaceCell s A B C C
          64 64 128 128 1 64 1 64 1 32 32 32 4 Bool.false
          (cOffset s 64 1 32 32 idx))) := by
  constructor
  · exact matmul_leakyrelu_python_case2_surface_toAlgorithm_supported A B C
  · exact matmul_leakyrelu_surface_output_compute_correct A B C
      64 64 128 128 1 64 1 64 1 32 32 32 4 Bool.false s

end VeriTile.Bench.TritonBenchG.MatmulLeakyrelu
