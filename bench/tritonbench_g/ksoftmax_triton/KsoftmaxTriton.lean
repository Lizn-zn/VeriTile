import VeriTile.Triton

/-!
# `ksoftmax_triton` — strict per-kernel correctness

`_softmax` is a fused 3D softmax over the last dimension with optional `qk`/`bk`
additive mask, optional causal masking, optional `LOG`-softmax output, and an
optional fp16-accumulator cast: programs `(m, n)` load a `DEPTH`-lane row (masked
by `k < K`, masked lanes read as `-inf`), subtract the row max, exponentiate,
sum, normalize (or take `z - log(denom)` for `LOG`), and store back masked by
`k < K`. The companion `_softmax_backward` computes the corresponding gradient.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`_softmax[grid](...)` with grid `(X.shape[0],
X.shape[1])`, the `DEPTH = next_power_of_2(K)` heuristic, the `IS_FP16` dtype
heuristic, `@triton.autotune` over `num_warps`, and how the runtime composes
per-program writes) is the *trusted boundary*, not a proof obligation here.
Because `(m, n)` are universally quantified, the per-program statement covers
every program of the grid.

## Proof architecture

```
ksoftmax_forward_plain_correctness            ← TOP THEOREM (ksoftmaxIO ⊨ ksoftmaxSpec)
  ├─ ksoftmax_forward_plain_flattenOk         bridge fragment membership
  ├─ ksoftmax_forward_plain_traceSafe         per-execution lane-wise safety walk
  └─ ksoftmax_forward_plain_region_run        region-model masked Hoare triple
       ├─ ksoftmax_forward_plain_exec_isSome  termination
       ├─ ksoftmax_forward_plain_correct      ← algorithm-layer readback per lane
       │    └─ ksoftmaxSpec_congr             only active lanes feed the spec
       └─ ksoftmax_forward_plain_frame        masked scatter frame
```

The full Python surfaces (`ksoftmax_forward_surface`, the `qk`/`bk` mask
variants, and `ksoftmax_backward_surface`) each have a `*_toAlgorithm_supported`
lemma showing they lower to the algorithm layer. The arithmetic correctness is
proved for the plain forward slice `ksoftmax_forward_plain` (the
non-mask, non-causal, non-fp16, non-log path) as the masked 2D-grid
Hoare-triple combinator `ksoftmaxIO … ⊨ ksoftmaxSpec`
(`Masked2DKernelIO₁.Implements`): for every disjoint flat placement of the two
buffers, every program `(m, n)` all of whose *active* lanes (`k < K`) are in
bounds, and every launch state whose active input-row lanes hold `xs`, the
translated pointer kernel terminates, every active output-row lane holds
`ksoftmaxSpec K DEPTH xs k`, and every other memory cell is unchanged.
`ksoftmaxSpec` is the exact stable softmax over the masked row (`reduceMax`,
lane-wise `exp`, `reduceSum`, quotient).

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); the approximate `tl.exp`
is modeled as exact `WithBot.realExp`; `@triton.autotune` / `num_warps` and the
`IS_FP16` heuristic are not modeled (the fp16 `.to(tl.float32)` cast reduces to
the identity at the algorithm layer). The reduction runs over the full `DEPTH`
block, but masked lanes load `⊥` (matching `other=float("-inf")`), so the
reduction-over-padded-block matches the upstream `-inf` semantics. The broader
mask/causal/log/backward branches are covered only at the lowering
(`toAlgorithm?`) level, not the arithmetic level.
-/

namespace VeriTile.Bench.TritonBenchG.KsoftmaxTriton

open VeriTile.Triton
open scoped VeriTile.Triton.Masked2DKernelIO₁

/-- Lean transcription of `ksoftmax_triton.py`'s `_softmax`.

`MASK_TYPE` records whether the optional mask is present; `MASK_QK` selects
Python's `qk` layout when true and `bk` layout when false. -/
def ksoftmax_forward_surface
    (Y X M : RegionName)
    (stride_ym stride_yn stride_xm stride_xn stride_m K DEPTH : Nat)
    (LOG MASK_TYPE CAUSAL IS_FP16 MASK_QK : Bool) :
    ComputeKernel := triton {
  m = tl.program_id(0)
  n = tl.program_id(1)
  k = tl.arange(0, $(DEPTH))
  x_ptrs = X + m * $(stride_xm) + n * $(stride_xn) + k
  io_mask = k < $(K)
  if CAUSAL {
    io_mask = io_mask & (k <= n)
  }
  x = tl.load(x_ptrs, mask=io_mask, other=float("-inf"))
  if CAUSAL {
    off = float("-inf")
    off = (off).to(x.dtype)
    x = tl.where(k > n, off, x)
  }
  if MASK_TYPE {
    if MASK_QK {
      mask_ptrs = M + n * $(stride_m) + k
    } else {
      mask_ptrs = M + m * $(stride_m) + k
    }
    add_mask = tl.load(mask_ptrs, io_mask, other=float("-inf"))
    x += add_mask
  }
  z = x - tl.max(x, axis=0)
  if IS_FP16 {
    z = (z).to(tl.float32)
  }
  num = tl.exp(z)
  denom = tl.sum(num, axis=0)
  if LOG {
    y = z - tl.log(denom)
  } else {
    y = num / denom
  }
  y_ptrs = Y + m * $(stride_ym) + n * $(stride_yn) + k
  tl.store(y_ptrs, y, mask=k < $(K))
}

/-- The full k-softmax forward surface lowers to the algorithm layer. -/
theorem ksoftmax_forward_surface_toAlgorithm_supported
    (Y X M : RegionName)
    (stride_ym stride_yn stride_xm stride_xn stride_m K DEPTH : Nat)
    (LOG MASK_TYPE CAUSAL IS_FP16 MASK_QK : Bool) :
    ∃ alg, (ksoftmax_forward_surface Y X M stride_ym stride_yn stride_xm
      stride_xn stride_m K DEPTH LOG MASK_TYPE CAUSAL IS_FP16 MASK_QK).toAlgorithm?
        = Except.ok alg := by
  simp [ksoftmax_forward_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- Surface transcription of `ksoftmax_triton.py`'s `_softmax` for the
`MASK_TYPE='qk'` branch.

This covers the tested qk-mask forward paths, including optional causal masking,
optional fp16 accumulator cast, and the `LOG`/softmax output branch. -/
def ksoftmax_forward_qk_surface
    (Y X Mask : RegionName)
    (stride_ym stride_yn stride_xm stride_xn stride_m K DEPTH : Nat)
    (LOG CAUSAL IS_FP16 : Bool) :
    ComputeKernel := triton {
  m = tl.program_id(0)
  n = tl.program_id(1)
  k = tl.arange(0, $(DEPTH))
  x_ptrs = X + m * $(stride_xm) + n * $(stride_xn) + k
  io_mask = k < $(K)
  if CAUSAL {
    io_mask = io_mask & (k <= n)
  }
  x = tl.load(x_ptrs, mask=io_mask, other=-inf)
  if CAUSAL {
    off = -inf
    off = (off).to(x.dtype)
    x = tl.where(k > n, off, x)
  }
  mask_ptrs = Mask + n * $(stride_m) + k
  add_mask = tl.load(mask_ptrs, mask=io_mask, other=-inf)
  x += add_mask
  z = x - tl.max(x, axis=0)
  if IS_FP16 {
    z = (z).to(tl.float32)
  }
  num = tl.exp(z)
  denom = tl.sum(num, axis=0)
  if LOG {
    y = z - tl.log(denom)
  } else {
    y = num / denom
  }
  y_ptrs = Y + m * $(stride_ym) + n * $(stride_yn) + k
  tl.store(y_ptrs, y, mask=k < $(K))
}

/-- The qk-mask k-softmax forward surface lowers to the algorithm layer. -/
theorem ksoftmax_forward_qk_surface_toAlgorithm_supported
    (Y X Mask : RegionName)
    (stride_ym stride_yn stride_xm stride_xn stride_m K DEPTH : Nat)
    (LOG CAUSAL IS_FP16 : Bool) :
    ∃ alg, (ksoftmax_forward_qk_surface Y X Mask stride_ym stride_yn
      stride_xm stride_xn stride_m K DEPTH LOG CAUSAL IS_FP16).toAlgorithm?
        = Except.ok alg := by
  simp [ksoftmax_forward_qk_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- Surface transcription of `ksoftmax_triton.py`'s `_softmax` for the
`MASK_TYPE='bk'` branch.

This covers the tested bk-mask forward paths, including the `LOG` branch. The
same definition keeps the optional causal branch because the Python kernel also
allows that combination. -/
def ksoftmax_forward_bk_surface
    (Y X Mask : RegionName)
    (stride_ym stride_yn stride_xm stride_xn stride_m K DEPTH : Nat)
    (LOG CAUSAL IS_FP16 : Bool) :
    ComputeKernel := triton {
  m = tl.program_id(0)
  n = tl.program_id(1)
  k = tl.arange(0, $(DEPTH))
  x_ptrs = X + m * $(stride_xm) + n * $(stride_xn) + k
  io_mask = k < $(K)
  if CAUSAL {
    io_mask = io_mask & (k <= n)
  }
  x = tl.load(x_ptrs, mask=io_mask, other=-inf)
  if CAUSAL {
    off = -inf
    off = (off).to(x.dtype)
    x = tl.where(k > n, off, x)
  }
  mask_ptrs = Mask + m * $(stride_m) + k
  add_mask = tl.load(mask_ptrs, mask=io_mask, other=-inf)
  x += add_mask
  z = x - tl.max(x, axis=0)
  if IS_FP16 {
    z = (z).to(tl.float32)
  }
  num = tl.exp(z)
  denom = tl.sum(num, axis=0)
  if LOG {
    y = z - tl.log(denom)
  } else {
    y = num / denom
  }
  y_ptrs = Y + m * $(stride_ym) + n * $(stride_yn) + k
  tl.store(y_ptrs, y, mask=k < $(K))
}

/-- The bk-mask k-softmax forward surface lowers to the algorithm layer. -/
theorem ksoftmax_forward_bk_surface_toAlgorithm_supported
    (Y X Mask : RegionName)
    (stride_ym stride_yn stride_xm stride_xn stride_m K DEPTH : Nat)
    (LOG CAUSAL IS_FP16 : Bool) :
    ∃ alg, (ksoftmax_forward_bk_surface Y X Mask stride_ym stride_yn
      stride_xm stride_xn stride_m K DEPTH LOG CAUSAL IS_FP16).toAlgorithm?
        = Except.ok alg := by
  simp [ksoftmax_forward_bk_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- Surface transcription of `ksoftmax_triton.py`'s `_softmax_backward`.

This covers the tested backward paths for `LOG=true/false` and
`CAUSAL=true/false`, preserving the masked gradient/out loads and the two
gradient formulas. -/
def ksoftmax_backward_surface
    (GradIn GradOut Out : RegionName)
    (stride_bm stride_bn stride_gm stride_gn stride_om stride_on K DEPTH : Nat)
    (LOG CAUSAL IS_FP16 : Bool) :
    ComputeKernel := triton {
  m = tl.program_id(0)
  n = tl.program_id(1)
  k = tl.arange(0, $(DEPTH))
  grad_out_ptrs = GradOut + m * $(stride_gm) + n * $(stride_gn) + k
  out_ptrs = Out + m * $(stride_om) + n * $(stride_on) + k
  io_mask = k < $(K)
  if CAUSAL {
    io_mask = io_mask & (k <= n)
  }
  g = tl.load(grad_out_ptrs, mask=io_mask, other=0.0)
  o = tl.load(out_ptrs, mask=io_mask, other=0.0)
  if CAUSAL {
    zero = 0.0
    zero = (zero).to(g.dtype)
    g = tl.where(k > n, zero, g)
    o = tl.where(k > n, zero, o)
  }
  if LOG {
    s = tl.sum(g, 0)
    if IS_FP16 {
      o = (o).to(tl.float32)
    }
    grad_in = g - tl.exp(o) * s
  } else {
    s = tl.sum(g * o, 0)
    grad_in = o * (g - s)
  }
  grad_in_ptrs = GradIn + m * $(stride_bm) + n * $(stride_bn) + k
  tl.store(grad_in_ptrs, grad_in, mask=k < $(K))
}

/-- The k-softmax backward surface lowers to the algorithm layer. -/
theorem ksoftmax_backward_surface_toAlgorithm_supported
    (GradIn GradOut Out : RegionName)
    (stride_bm stride_bn stride_gm stride_gn stride_om stride_on K DEPTH : Nat)
    (LOG CAUSAL IS_FP16 : Bool) :
    ∃ alg, (ksoftmax_backward_surface GradIn GradOut Out stride_bm stride_bn
      stride_gm stride_gn stride_om stride_on K DEPTH LOG CAUSAL
      IS_FP16).toAlgorithm? = Except.ok alg := by
  simp [ksoftmax_backward_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- Proof-oriented forward softmax slice of `ksoftmax_triton.py`'s `_softmax`.

This specializes the constexpr branches to:
- `LOG = false`
- `MASK_TYPE = None`
- `CAUSAL = false`
- `IS_FP16 = false`

It preserves the 2D `(m, n)` program ids, strided 3D row addressing, masked
load over the last dimension, stable softmax normalization, and masked store. -/
def ksoftmax_forward_plain
    (Y X : RegionName)
    (stride_ym stride_yn stride_xm stride_xn K DEPTH : Nat) :
    ComputeKernel := triton {
  m = tl.program_id(0)
  n = tl.program_id(1)
  k = tl.arange(0, $(DEPTH))
  x_ptrs = X + m * $(stride_xm) + n * $(stride_xn) + k
  io_mask = k < $(K)
  x = tl.load(x_ptrs, mask=io_mask, other=-inf)
  z = x - tl.max(x, axis=0)
  num = tl.exp(z)
  denom = tl.sum(num, axis=0)
  y = num / denom
  y_ptrs = Y + m * $(stride_ym) + n * $(stride_yn) + k
  tl.store(y_ptrs, y, mask=k < $(K))
}

/-- Masked input row tile used by `ksoftmax_forward_plain`: lane `k < K` holds
`xs k`, masked lanes are `⊥`, matching `other=-inf`. -/
noncomputable def ksoftmaxInputTile (K DEPTH : Nat)
    (xs : Fin DEPTH → ℝ) : Tile .real [DEPTH] :=
  { data := fun idx => if idx.1.val < K then some (xs idx.1) else none }

/-- Exact stable-softmax value computed by the kernel at lane `idx`, as a pure
function of the active row prefix `xs k`, `k < K` (masked lanes enter the
reductions as `⊥`, neutral for both `max` and the `exp`-sum). -/
noncomputable def ksoftmaxSpec (K DEPTH : Nat)
    (xs : Fin DEPTH → ℝ) (idx : Fin DEPTH) : ℝ :=
  let row := ksoftmaxInputTile K DEPTH xs
  match Tile.reduceMax (shape := [DEPTH]) ⟨0, by simp⟩ Bool.false row with
  | some rowMax =>
      let shifted := Tile.bop (NumericDType.sub .real) Broadcast.scalarR row rowMax
      let num := Tile.uop WithBot.realExp shifted
      let denom := Tile.reduceSum (shape := [DEPTH]) ⟨0, by simp⟩ Bool.false num
      WithBot.unbotD 0
        ((Tile.bop (NumericDType.div .real) Broadcast.scalarR num denom).data
          (idx, PUnit.unit))
  | none => 0

/-- `ksoftmaxSpec` only reads the active lanes of its input: two rows agreeing
below `K` yield the same softmax value (masked lanes are `⊥` in the tile
either way). -/
theorem ksoftmaxSpec_congr (K DEPTH : Nat) (xs ys : Fin DEPTH → ℝ)
    (h : ∀ j : Fin DEPTH, j.val < K → xs j = ys j) (i : Fin DEPTH) :
    ksoftmaxSpec K DEPTH xs i = ksoftmaxSpec K DEPTH ys i := by
  have htile : ksoftmaxInputTile K DEPTH xs = ksoftmaxInputTile K DEPTH ys := by
    unfold ksoftmaxInputTile
    congr 1
    funext idx
    by_cases hj : idx.1.val < K
    · simp only [if_pos hj, h idx.1 hj]
    · simp only [if_neg hj]
  unfold ksoftmaxSpec
  rw [htile]

/-- Algorithm-layer cellwise correctness for the plain forward softmax slice:
in-bounds lanes hold `ksoftmaxSpec` of the loaded row, out-of-bounds lanes
are preserved. -/
theorem ksoftmax_forward_plain_correct
    (Y X : RegionName)
    (stride_ym stride_yn stride_xm stride_xn K DEPTH : Nat)
    (s s' : BlockState)
    (hExec : exec (ksoftmax_forward_plain Y X
        stride_ym stride_yn stride_xm stride_xn K DEPTH) s = some s') :
    ∀ i : Fin DEPTH,
      s'.readMem Y (s.pids 0 * stride_ym + s.pids 1 * stride_yn + i.val) =
        if i.val < K then
          ksoftmaxSpec K DEPTH
            (fun j => s.readMem X
              (s.pids 0 * stride_xm + s.pids 1 * stride_xn + j.val)) i
        else s.readMem Y (s.pids 0 * stride_ym + s.pids 1 * stride_yn + i.val) := by
  intro i
  by_cases hD : 0 < DEPTH
  · simp [exec, ksoftmax_forward_plain, stepStmts, stepStmt, evalOp.eq_def,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
          TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
          NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
          ComparableDType.lt, hD] at hExec
    subst s'
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : i.val < K
    · simp [hi, ksoftmaxSpec, ksoftmaxInputTile,
            Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum,
            Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
            TileShape.insertAxisIndex, hD]
      congr
    · simp [hi]
  · exact False.elim (hD (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- A masked scatter-store `foldl` leaves every memory cell it does not
actively hit unchanged (cell-level frame for the masked store). -/
private theorem foldl_store_preserve_cell {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → ℝ) (P : α → Prop) [DecidablePred P]
    (r : RegionName) (o : Nat) (l : List α) (s : BlockState)
    (hnot : ∀ k ∈ l, P k → ¬(region = r ∧ offsetFn k = o)) :
    (l.foldl (fun acc k =>
        if P k then acc.writeMem region (offsetFn k) (valueFn k) else acc)
      s).mem r o = s.mem r o := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons]
      by_cases hP : P hd
      · rw [if_pos hP,
          ih _ (fun k hk => hnot k (List.mem_cons_of_mem hd hk)),
          BlockState.writeMem_mem]
        exact if_neg (fun hc =>
          hnot hd List.mem_cons_self hP ⟨hc.1.symm, hc.2.symm⟩)
      · rw [if_neg hP]
        exact ih _ (fun k hk => hnot k (List.mem_cons_of_mem hd hk))

set_option maxHeartbeats 1600000 in
/-- Frame half: every memory cell not actively written by the masked output
store — every cell of every region other than `Y`, and the *inactive* lanes
of the output row itself — is preserved by the run. -/
private theorem ksoftmax_forward_plain_frame
    (Y X : RegionName)
    (stride_ym stride_yn stride_xm stride_xn K DEPTH : Nat)
    (hD : 0 < DEPTH) (s s1 : BlockState)
    (hExec : exec ((ksoftmax_forward_plain Y X stride_ym stride_yn stride_xm
        stride_xn K DEPTH).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ i : Fin DEPTH, i.val < K →
      ¬(Y = r ∧ s.pids 0 * stride_ym + s.pids 1 * stride_yn + i.val = o)) :
    s1.mem r o = s.mem r o := by
  simp [exec, ksoftmax_forward_plain, ComputeKernel.toAlgKernel, stepStmts,
        stepStmt, evalOp.eq_def,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
        Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
        ComparableDType.lt, hD] at hExec
  subst s1
  refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
  intro k _ hmk hc
  exact hmiss k.1 (by simpa using hmk) hc

set_option maxHeartbeats 1600000 in
/-- Termination: the kernel executes to completion from any state. `0 < DEPTH`
is required because the `max` reduce (like `Finset.sup'`) is only defined on
non-empty axes. -/
private theorem ksoftmax_forward_plain_exec_isSome
    (Y X : RegionName)
    (stride_ym stride_yn stride_xm stride_xn K DEPTH : Nat)
    (hD : 0 < DEPTH) (s : BlockState) :
    ∃ s1, exec ((ksoftmax_forward_plain Y X stride_ym stride_yn stride_xm
        stride_xn K DEPTH).toAlgKernel) s = some s1 := by
  simp [exec, ksoftmax_forward_plain, ComputeKernel.toAlgKernel, stepStmts,
        stepStmt, evalOp.eq_def,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
        Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
        ComparableDType.lt, hD]

/-- **The region-model masked Hoare triple** — termination, active-lane output
values, and frame off the active output lanes, from any launch state whose
input row is loaded at the **active lanes only** (`j < K`). This is the
`hrun` obligation of the `⊨` headline. -/
theorem ksoftmax_forward_plain_region_run
    (Y X : RegionName)
    (stride_ym stride_yn stride_xm stride_xn K DEPTH : Nat)
    (hD : 0 < DEPTH) (s₀ : BlockState) (xs : Fin DEPTH → ℝ)
    (hx : ∀ j : Fin DEPTH, j.val < K →
      s₀.readMem X (s₀.pids 0 * stride_xm + s₀.pids 1 * stride_xn + j.val)
        = xs j) :
    ∃ s1, exec ((ksoftmax_forward_plain Y X stride_ym stride_yn stride_xm
          stride_xn K DEPTH).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin DEPTH, j.val < K →
          s1.readMem Y (s₀.pids 0 * stride_ym + s₀.pids 1 * stride_yn + j.val)
            = ksoftmaxSpec K DEPTH xs j)
      ∧ (∀ r o,
          (r ≠ Y ∨ ∀ j : Fin DEPTH, j.val < K →
            o ≠ s₀.pids 0 * stride_ym + s₀.pids 1 * stride_yn + j.val) →
          s1.mem r o = s₀.mem r o) := by
  obtain ⟨s1, hs1⟩ := ksoftmax_forward_plain_exec_isSome Y X
    stride_ym stride_yn stride_xm stride_xn K DEPTH hD s₀
  refine ⟨s1, hs1, fun j hj => ?_, fun r o hcond => ?_⟩
  · have h := ksoftmax_forward_plain_correct Y X
      stride_ym stride_yn stride_xm stride_xn K DEPTH s₀ s1 hs1 j
    simp only [hj, if_pos] at h
    rw [h]
    exact ksoftmaxSpec_congr K DEPTH _ xs (fun k hk => hx k hk) j
  · refine ksoftmax_forward_plain_frame Y X
      stride_ym stride_yn stride_xm stride_xn K DEPTH hD s₀ s1 hs1 r o
      (fun i hi ⟨hr, ho⟩ => ?_)
    rcases hcond with hne | hno
    · exact hne hr.symm
    · exact hno i hi ho.symm

/-- The kernel sits inside the flat-memory bridge's covered fragment (pointer
arithmetic, masked load with `other`, reductions, masked store). -/
theorem ksoftmax_forward_plain_flattenOk
    (Y X : RegionName)
    (stride_ym stride_yn stride_xm stride_xn K DEPTH : Nat) :
    ((ksoftmax_forward_plain Y X stride_ym stride_yn stride_xm stride_xn
        K DEPTH).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [ksoftmax_forward_plain, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, StmtList.FlattenOk, Stmt.FlattenOk,
    Op.FlattenOk.eq_def]

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk: one computational unfold walks all the
statements — the pointer/index staging, the reductions, and the register
arithmetic are memory-silent — and reduces the two masked accesses (row load,
row store) to the **lane-wise** bounds hypotheses: every *active* lane's
address is below the region bound. -/
theorem ksoftmax_forward_plain_traceSafe
    (Y X : RegionName)
    (stride_ym stride_yn stride_xm stride_xn K DEPTH : Nat)
    (hD : 0 < DEPTH) (bounds : RegionBounds) (s : BlockState)
    (hin : ∀ j : Fin DEPTH, j.val < K →
      s.pids 0 * stride_xm + s.pids 1 * stride_xn + j.val < bounds X)
    (hout : ∀ j : Fin DEPTH, j.val < K →
      s.pids 0 * stride_ym + s.pids 1 * stride_yn + j.val < bounds Y) :
    Kernel.TraceSafe bounds
      ((ksoftmax_forward_plain Y X stride_ym stride_yn stride_xm stride_xn
        K DEPTH).toAlgKernel) s := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hD.ne'
  unfold Kernel.TraceSafe
  simp [ksoftmax_forward_plain, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, Op.PointerAddressesSafeOn, Op.MemorySafe,
    BlockState.setReg,
    Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd,
    NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
    ComparableDType.lt,
    Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]
  exact ⟨fun a ha => hin a ha, fun a ha => hout a ha⟩

/-- `ksoftmax_forward_plain`'s masked **2D IO signature** — the whole
kernel-specific audit surface of the `⊨` headline:

* `inp`/`out` — which buffer is which argument (the wiring);
* `B = DEPTH` — the last-dimension row window each program owns;
* `read`/`write` — program `(m, n)` reads lane `k` of its row at
  `m * stride_xm + n * stride_xn + k` and writes it at
  `m * stride_ym + n * stride_yn + k` (the host-side one-program-per-`(m, n)`
  launch convention over the leading two dimensions);
* `mask` — the active lanes `k < K`, **the same for every program**: the row
  prefix that actually exists in the tensor. Inactive lanes (the padding of
  `DEPTH = next_power_of_2(K)`) carry no obligations on either side.

The windows and mask are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing and masking match them. Buffer sizes
are not signature content: the headline quantifies over every allocation whose
extents cover the active lanes. -/
def ksoftmaxIO (Y X : RegionName)
    (stride_ym stride_yn stride_xm stride_xn K DEPTH : Nat) :
    Masked2DKernelIO₁ where
  kernel := ksoftmax_forward_plain Y X stride_ym stride_yn stride_xm
    stride_xn K DEPTH
  inp := X
  out := Y
  B := DEPTH
  read := fun m n j => m * stride_xm + n * stride_xn + j.val
  write := fun m n j => m * stride_ym + n * stride_yn + j.val
  mask := fun _ _ j => j.val < K

/-- **The headline**: the plain forward slice of `_softmax` (`LOG=false`, no
mask, non-causal, no fp16 cast) implements the exact stable softmax over the
active row prefix on its masked 2D IO signature — for every disjoint flat
placement of the two buffers, every program `(m, n)` whose active lanes are in
bounds, and every launch state whose active input-row lanes hold `xs`, the
translated pointer kernel terminates, every active output-row lane `k` holds
`ksoftmaxSpec K DEPTH xs k`, and every other memory cell is unchanged.
`0 < DEPTH` is required: the kernel's `max` reduce (like `Finset.sup'`) is
only defined on non-empty tiles. Proof: `Implements.intro` assembles the
region-model masked triple with the bridge side conditions. -/
specification ksoftmax_forward_plain_correctness
    (Y X : RegionName)
    (stride_ym stride_yn stride_xm stride_xn K DEPTH : Nat)
    (hD : 0 < DEPTH) :
    ksoftmaxIO Y X stride_ym stride_yn stride_xm stride_xn K DEPTH ⊨
      fun _ _ xs k => ksoftmaxSpec K DEPTH xs k := by
  refine Masked2DKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact ksoftmax_forward_plain_flattenOk Y X
      stride_ym stride_yn stride_xm stride_xn K DEPTH
  · intro bounds s h1 h2 _
    exact ksoftmax_forward_plain_traceSafe Y X
      stride_ym stride_yn stride_xm stride_xn K DEPTH hD bounds s h1 h2
  · intro s₀ xs hx
    obtain ⟨s1, hexec, hval, hframe⟩ := ksoftmax_forward_plain_region_run Y X
      stride_ym stride_yn stride_xm stride_xn K DEPTH hD s₀ xs hx
    -- scratch is empty, so its frame side condition is vacuous
    exact ⟨s1, hexec, hval, fun r o hout _ => hframe r o hout⟩

end VeriTile.Bench.TritonBenchG.KsoftmaxTriton
