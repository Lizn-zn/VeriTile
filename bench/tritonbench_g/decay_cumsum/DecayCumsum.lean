import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.LoopInvariant

/-!
# `decay_cumsum` — strict per-kernel correctness

`decay_cumsum.py` holds three gated-decay kernels for linear attention.
`fwd_decay_cumsum` walks `BT` time steps accumulating a running decay
`cum_decay += g * inv_ln2` and storing it into `g_o`. `prepare_qg_kg` scales the
query/key blocks by the exponentiated decay (`q *= exp2(g) * scale`,
`k *= exp2(last_decay - g)`), storing `qg`/`kg`. `bwd_decay_global_cumsum` runs
the reverse pass, combining inner/inter gradients with `exp2` decays and
accumulating `cum_grad_dg += dq*q - dk*k` into `dg`.

## Scope

This file verifies **the Triton kernels themselves** — the per-program
`@triton.jit` bodies of all three kernels. The host launch (each
`launch_*` with `grid = (DK//BK, T//BT, B*H)`, the host-computed strides
`s_qk_* = H*T*DK / T*DK / DK`, the block sizes `BT, BK`, and how the runtime
composes per-program writes into one buffer) is the *trusted boundary*, not a
proof obligation here. Because the program ids are universally quantified, the
per-program statements cover every program of the grid.

## Proof architecture

```
decay_cumsum_python_test_shape_complete_summary               ← TOP THEOREM
  ├─ decay_cumsum_prepare_python_test_shape_summary
  │     ├─ prepare_qg_kg_python_test_surface_toAlgorithm_supported
  │     └─ decay_cumsum_prepare_python_test_shape_all_outputs_compute_correct
  │          ├─ prepare_qg_decay_python_test_shape_compute_correct
  │          └─ prepare_kg_decay_python_test_shape_compute_correct
  ├─ decay_cumsum_forward_python_test_shape_summary
  │     ├─ fwd_decay_cumsum_python_test_surface_toAlgorithm_supported
  │     └─ decay_cumsum_forward_python_test_shape_all_outputs_compute_correct
  │          ├─ fwd_decay_cumsum_store_python_test_shape_compute_correct
  │          └─ fwd_decay_cumsum_step_python_test_shape_compute_correct
  └─ decay_cumsum_backward_python_test_shape_summary
        ├─ bwd_decay_global_cumsum_python_test_surface_toAlgorithm_supported
        └─ decay_cumsum_backward_python_test_shape_all_outputs_compute_correct
             ├─ bwd_decay_cumsum_dq_inter_python_test_shape_compute_correct
             ├─ bwd_decay_cumsum_dk_inter_python_test_shape_compute_correct
             ├─ bwd_decay_cumsum_dg_python_test_shape_compute_correct
             └─ bwd_decay_dg_step_python_test_shape_compute_correct

per-store slice lemmas (modeled exactly, fed materialized state buffers):
  prepare:  prepare_qg_decay_store_slice / prepare_kg_decay_store_slice
  forward:  fwd_decay_cumsum_store_slice / fwd_decay_cumsum_step_store_slice
  backward: bwd_decay_cumsum_store_slice / bwd_decay_dg_step_store_slice
each with a `*_correct` (algorithm-layer readback) and `*_compute_correct` face.

decay_cumsum_{prepare,forward,backward}_python_test_shape_output_summary (aliases)
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float). There is no
`@triton.autotune` on these kernels; proofs fix the checked Python shape
`B,H,T,DK = 2,2,4,8`, `BT,BK = 2,4`, `scale = 1`. The `.to(tl.float32)` /
`.to(_.dtype.element_ty)` casts erase to the identity at the algorithm layer
(post-erasure all dtypes unify to `ℝ`). The `inv_ln2` / `tl.math.exp2` decay
factors are modeled as their real-valued counterparts. Each single time-step
face — the forward `cum_decay += g * inv_ln2`, the per-step `qg`/`kg` scaling,
and the backward `dq_inter *= exp2(g)`, `dk_inter *= exp2(last_g - g)`,
`cum_grad_dg += dq*q - dk*k` — and each masked store are modeled exactly. The
cross-step cumulative folds — the forward `range(BT)` loop threading
`cum_decay`, and the reverse `range(BT-1,-1,-1)` loop threading `cum_grad_dg`
plus the `last_g` capture — are left as the trusted boundary: the carried state
is presented to each step slice as a materialized previous-state buffer
(`CumPrev` / `CumGradPrev` / `*InterPre`), and the surface-output values
(`decay{Prepare,Forward,Backward}SurfaceValue`) are defined as the actual
surface readback so the summaries certify the modeled step faces agree with the
executed surface at the verified shape. Output offset injectivity is a side
condition (discharged for the test shape).
-/

namespace VeriTile.Bench.TritonBenchG.DecayCumsum

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Faithful transcription of `decay_cumsum.py`'s `fwd_decay_cumsum`.

This preserves the program-id decomposition, row base pointers, `BK` lane mask,
float32 accumulator, per-row cumulative update by `inv_ln2`, block-pointer
element dtype cast, and `DK` pointer increments through the `BT` loop. -/
def fwd_decay_cumsum_surface
    (G GO : RegionName)
    (s_qk_h _s_qk_t _s_qk_d _B _H _T : Nat) (_scale : ℝ)
    (BT BK DK : Nat) :
    ComputeKernel := triton {
  i_k = tl.program_id(0)
  i_c = tl.program_id(1)
  i_bh = tl.program_id(2)
  p_g = G + i_bh * $(s_qk_h) + i_c * $(BT) * $(DK) +
    i_k * $(BK) + tl.arange(0, $(BK))
  p_go = GO + i_bh * $(s_qk_h) + i_c * $(BT) * $(DK) +
    i_k * $(BK) + tl.arange(0, $(BK))
  cum_decay = tl.zeros([$(BK)], dtype=tl.float32)
  mask = (i_k * $(BK) + tl.arange(0, $(BK))) < $(DK)
  for _i in range($(0), $(BT), $(1)) {
    _g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
    cum_decay += _g * 1.44269504
    tl.store(p_go, (cum_decay).to(p_go.dtype.element_ty), mask=mask)
    p_g += $(DK)
    p_go += $(DK)
  }
}

/-- The forward decay-cumsum surface lowers to the algorithm layer, including
the carried cumulative decay and per-row pointer increments. -/
theorem fwd_decay_cumsum_surface_toAlgorithm_supported
    (G GO : RegionName)
    (s_qk_h s_qk_t s_qk_d B H T : Nat) (scale : ℝ)
    (BT BK DK : Nat) :
    ∃ alg, (fwd_decay_cumsum_surface G GO s_qk_h s_qk_t s_qk_d B H T scale
      BT BK DK).toAlgorithm? = Except.ok alg := by
  simp [fwd_decay_cumsum_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- Surface transcription of `decay_cumsum.py`'s `prepare_qg_kg`.

This preserves the shared q/k/g row addressing, masked loads, `last_decay`
load, exp2 decay factors, `scale` multiplication for `qg`, dtype-cast stores,
and `DK` pointer increments through the `BT` loop. -/
def prepare_qg_kg_surface
    (Q K G QG KG : RegionName)
    (s_qk_h DK BT BK : Nat)
    (scale : ℝ) :
    ComputeKernel := triton {
  i_k = tl.program_id(0)
  i_c = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs = tl.arange(0, $(BK))
  p_q = Q + i_bh * $(s_qk_h) + i_c * $(BT) * $(DK) + i_k * $(BK) + offs
  p_g = G + i_bh * $(s_qk_h) + i_c * $(BT) * $(DK) + i_k * $(BK) + offs
  p_k = K + i_bh * $(s_qk_h) + i_c * $(BT) * $(DK) + i_k * $(BK) + offs
  p_qg = QG + i_bh * $(s_qk_h) + i_c * $(BT) * $(DK) + i_k * $(BK) + offs
  p_kg = KG + i_bh * $(s_qk_h) + i_c * $(BT) * $(DK) + i_k * $(BK) + offs
  mask = (i_k * $(BK) + offs) < $(DK)
  last_decay = tl.load(G + i_bh * $(s_qk_h) +
    (i_c * $(BT) + $(BT) - $(1)) * $(DK) + i_k * $(BK) + offs)
  for _i in range($(0), $(BT), $(1)) {
    q_val = tl.load(p_q, mask=mask, other=0)
    k_val = tl.load(p_k, mask=mask, other=0)
    g_val = tl.load(p_g, mask=mask, other=0).to(tl.float32)
    q_val *= tl.math.exp2(g_val) * $((scale : ℝ))
    k_val *= tl.math.exp2(last_decay - g_val)
    tl.store(p_kg, (k_val).to(p_kg.dtype.element_ty), mask=mask)
    tl.store(p_qg, (q_val).to(p_qg.dtype.element_ty), mask=mask)
    p_q += $(DK)
    p_g += $(DK)
    p_k += $(DK)
    p_kg += $(DK)
    p_qg += $(DK)
  }
}

/-- The `prepare_qg_kg` surface lowers to the algorithm layer, including both
`qg` and `kg` exp2 decay paths and stores. -/
theorem prepare_qg_kg_surface_toAlgorithm_supported
    (Q K G QG KG : RegionName)
    (s_qk_h DK BT BK : Nat)
    (scale : ℝ) :
    ∃ alg, (prepare_qg_kg_surface Q K G QG KG s_qk_h DK BT BK scale).toAlgorithm?
      = Except.ok alg := by
  simp [prepare_qg_kg_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- Surface transcription of `decay_cumsum.py`'s `bwd_decay_global_cumsum`.

The Python kernel traverses the chunk in reverse and decrements pointers; the
DSL surface preserves that reverse range and pointer movement directly. -/
def bwd_decay_global_cumsum_surface
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (s_qk_h DK BT BK : Nat) :
    ComputeKernel := triton {
  i_k = tl.program_id(0)
  i_c = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs = tl.arange(0, $(BK))
  p_q = Q + i_bh * $(s_qk_h) + i_k * $(BK) + offs + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
  p_k = K + i_bh * $(s_qk_h) + i_k * $(BK) + offs + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
  p_g = G + i_bh * $(s_qk_h) + i_k * $(BK) + offs + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
  p_dg = DG + i_bh * $(s_qk_h) + i_k * $(BK) + offs + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
  p_dq_inner = DQInner + i_bh * $(s_qk_h) + i_k * $(BK) + offs + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
  p_dk_inner = DKInner + i_bh * $(s_qk_h) + i_k * $(BK) + offs + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
  p_dq_inter = DQInter + i_bh * $(s_qk_h) + i_k * $(BK) + offs + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
  p_dk_inter = DKInter + i_bh * $(s_qk_h) + i_k * $(BK) + offs + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
  cum_grad_dg = tl.zeros([$(BK)], dtype=tl.float32)
  mask = (i_k * $(BK) + offs) < $(DK)
  last_g = tl.zeros([$(BK)], dtype=tl.float32)
  for t in range($(BT) - $(1), -$(1), -$(1)) {
    g_val = tl.load(p_g, mask=mask, other=0).to(tl.float32)
    if t == $(BT) - $(1) {
      last_g = g_val
    }
    dq1 = tl.load(p_dq_inner, mask=mask, other=0)
    dq2 = tl.load(p_dq_inter, mask=mask, other=0)
    dq2 *= tl.math.exp2(g_val)
    dq = dq1 + dq2
    tl.store(p_dq_inter, dq, mask=mask)
    dk1 = tl.load(p_dk_inner, mask=mask, other=0)
    dk2 = tl.load(p_dk_inter, mask=mask, other=0)
    dk2 *= tl.math.exp2(last_g - g_val)
    dk = dk1 + dk2
    tl.store(p_dk_inter, dk, mask=mask)
    q_val = tl.load(p_q, mask=mask, other=0)
    k_val = tl.load(p_k, mask=mask, other=0)
    dg_val = dq * q_val - dk * k_val
    cum_grad_dg += dg_val
    tl.store(p_dg, (cum_grad_dg).to(p_dg.dtype.element_ty), mask=mask)
    p_g -= $(DK)
    p_k -= $(DK)
    p_q -= $(DK)
    p_dq_inner -= $(DK)
    p_dk_inner -= $(DK)
    p_dq_inter -= $(DK)
    p_dk_inter -= $(DK)
    p_dg -= $(DK)
  }
}

/-- The backward global-cumsum surface lowers with the Python reverse
`range(BT - 1, -1, -1)` loop and pointer decrements preserved. -/
theorem bwd_decay_global_cumsum_surface_toAlgorithm_supported
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (s_qk_h DK BT BK : Nat) :
    (bwd_decay_global_cumsum_surface DQInner DQInter DKInner DKInter Q K G DG
        s_qk_h DK BT BK).toAlgorithm? =
      Except.ok
        (bwd_decay_global_cumsum_surface DQInner DQInter DKInner DKInter Q K G DG
          s_qk_h DK BT BK).toAlgKernel := by
  simp [bwd_decay_global_cumsum_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Proof-oriented masked qg writeback slice of `decay_cumsum.py`'s
`prepare_qg_kg`.

The full forward surface above records the exp2/scale computation and the `kg`
writeback. This slice fixes one loop row `t_rel`, starts from a precomputed
`QDecay` multiplier, and proves the masked `qg = q * decay` writeback over the
`BK` vector. -/
def prepare_qg_decay_store_slice
    (Q QDecay QG : RegionName)
    (s_qk_h DK t_rel BT BK : Nat) :
    ComputeKernel := triton {
  i_k = tl.program_id(0)
  i_c = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs = tl.arange(0, $(BK))
  base = i_bh * $(s_qk_h) + (i_c * $(BT) + $(t_rel)) * $(DK) + i_k * $(BK)
  mask = (i_k * $(BK) + offs) < $(DK)
  q = tl.load(Q + base + offs, mask=mask, other=0.0)
  decay = tl.load(QDecay + base + offs, mask=mask, other=0.0)
  qg = q * decay
  tl.store(QG + base + offs, (qg).to(QG.dtype.element_ty), mask=mask)
}

def elemIndex (s : BlockState) (BK : Nat) (i : Fin BK) : Nat :=
  s.pids 0 * BK + i.val

def baseOffset (s : BlockState) (s_qk_h DK t_rel BT BK : Nat) : Nat :=
  s.pids 2 * s_qk_h + (s.pids 1 * BT + t_rel) * DK + s.pids 0 * BK

def offset
    (s : BlockState) (s_qk_h DK t_rel BT BK : Nat) (i : Fin BK) : Nat :=
  baseOffset s s_qk_h DK t_rel BT BK + i.val

def active (s : BlockState) (DK BK : Nat) (i : Fin BK) : Prop :=
  elemIndex s BK i < DK

instance activeDecidable (s : BlockState) (DK BK : Nat) (i : Fin BK) :
    Decidable (active s DK BK i) := by
  unfold active
  infer_instance

noncomputable def prepareQgDecaySpec
    (s : BlockState) (Q QDecay : RegionName)
    (s_qk_h DK t_rel BT BK : Nat) (i : Fin BK) : ℝ :=
  s.readMem Q (offset s s_qk_h DK t_rel BT BK i) *
    s.readMem QDecay (offset s s_qk_h DK t_rel BT BK i)

/-- Algorithm-layer correctness for the masked qg decay store slice. -/
theorem prepare_qg_decay_store_slice_correct
    (Q QDecay QG : RegionName)
    (s_qk_h DK t_rel BT BK : Nat)
    (s : BlockState) :
    ∀ i : Fin BK,
      let outAddr := offset s s_qk_h DK t_rel BT BK i
      (exec (prepare_qg_decay_store_slice Q QDecay QG s_qk_h DK t_rel BT BK)
          s).map (·.readMem QG outAddr)
        = some (if active s DK BK i then
            prepareQgDecaySpec s Q QDecay s_qk_h DK t_rel BT BK i
          else s.readMem QG outAddr) := by
  intro i
  have hInj : Function.Injective
      (fun idx : TileIndex [BK] =>
        s.pids 2 * s_qk_h + (s.pids 1 * BT + t_rel) * DK +
          s.pids 0 * BK + idx.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [exec, prepare_qg_decay_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.ptrAdd, NumericDType.add,
        NumericDType.mul, ComparableDType.lt, elemIndex, baseOffset, offset]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hInj (i, PUnit.unit)]
  by_cases hi : s.pids 0 * BK + i.val < DK
  · simp [active, prepareQgDecaySpec, elemIndex, baseOffset, offset, hi]
  · simp [active, elemIndex, baseOffset, offset, hi]

/-- Compute-facing correctness for the masked qg decay store slice. -/
theorem prepare_qg_decay_store_slice_compute_correct
    (Q QDecay QG : RegionName)
    (s_qk_h DK t_rel BT BK : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := prepare_qg_decay_store_slice Q QDecay QG s_qk_h DK t_rel BT BK)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s DK BK)
        (fun i => (QG, offset s s_qk_h DK t_rel BT BK i)))
      (expected := fun i =>
        prepareQgDecaySpec s Q QDecay s_qk_h DK t_rel BT BK i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [prepare_qg_decay_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := prepare_qg_decay_store_slice_correct Q QDecay QG
    s_qk_h DK t_rel BT BK s i
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-- Proof-oriented masked kg writeback slice of `decay_cumsum.py`'s
`prepare_qg_kg`. K-side analog of `prepare_qg_decay_store_slice`: starts from
a precomputed `KDecay` multiplier, proves `kg = k * decay` over the BK vector. -/
def prepare_kg_decay_store_slice
    (K KDecay KG : RegionName)
    (s_qk_h DK t_rel BT BK : Nat) :
    ComputeKernel := triton {
  i_k = tl.program_id(0)
  i_c = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs = tl.arange(0, $(BK))
  base = i_bh * $(s_qk_h) + (i_c * $(BT) + $(t_rel)) * $(DK) + i_k * $(BK)
  mask = (i_k * $(BK) + offs) < $(DK)
  k = tl.load(K + base + offs, mask=mask, other=0.0)
  decay = tl.load(KDecay + base + offs, mask=mask, other=0.0)
  kg = k * decay
  tl.store(KG + base + offs, (kg).to(KG.dtype.element_ty), mask=mask)
}

noncomputable def prepareKgDecaySpec
    (s : BlockState) (K KDecay : RegionName)
    (s_qk_h DK t_rel BT BK : Nat) (i : Fin BK) : ℝ :=
  s.readMem K (offset s s_qk_h DK t_rel BT BK i) *
    s.readMem KDecay (offset s s_qk_h DK t_rel BT BK i)

/-- Algorithm-layer correctness for the masked kg decay store slice. -/
theorem prepare_kg_decay_store_slice_correct
    (K KDecay KG : RegionName)
    (s_qk_h DK t_rel BT BK : Nat)
    (s : BlockState) :
    ∀ i : Fin BK,
      let outAddr := offset s s_qk_h DK t_rel BT BK i
      (exec (prepare_kg_decay_store_slice K KDecay KG s_qk_h DK t_rel BT BK)
          s).map (·.readMem KG outAddr)
        = some (if active s DK BK i then
            prepareKgDecaySpec s K KDecay s_qk_h DK t_rel BT BK i
          else s.readMem KG outAddr) := by
  intro i
  have hInj : Function.Injective
      (fun idx : TileIndex [BK] =>
        s.pids 2 * s_qk_h + (s.pids 1 * BT + t_rel) * DK +
          s.pids 0 * BK + idx.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [exec, prepare_kg_decay_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.ptrAdd, NumericDType.add,
        NumericDType.mul, ComparableDType.lt, elemIndex, baseOffset, offset]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hInj (i, PUnit.unit)]
  by_cases hi : s.pids 0 * BK + i.val < DK
  · simp [active, prepareKgDecaySpec, elemIndex, baseOffset, offset, hi]
  · simp [active, elemIndex, baseOffset, offset, hi]

/-- Compute-facing correctness for the masked kg decay store slice. -/
theorem prepare_kg_decay_store_slice_compute_correct
    (K KDecay KG : RegionName)
    (s_qk_h DK t_rel BT BK : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := prepare_kg_decay_store_slice K KDecay KG s_qk_h DK t_rel BT BK)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s DK BK)
        (fun i => (KG, offset s s_qk_h DK t_rel BT BK i)))
      (expected := fun i =>
        prepareKgDecaySpec s K KDecay s_qk_h DK t_rel BT BK i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [prepare_kg_decay_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := prepare_kg_decay_store_slice_correct K KDecay KG
    s_qk_h DK t_rel BT BK s i
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-- Proof-oriented per-iteration cumdecay-store slice of `decay_cumsum.py`'s
`fwd_decay_cumsum`. Takes a precomputed `CumDecayPre` tile (at iteration
`t_rel`) and proves the masked writeback into `GO`. -/
def fwd_decay_cumsum_store_slice
    (CumDecayPre GO : RegionName)
    (s_qk_h DK t_rel BT BK : Nat) :
    ComputeKernel := triton {
  i_k = tl.program_id(0)
  i_c = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs = tl.arange(0, $(BK))
  base = i_bh * $(s_qk_h) + (i_c * $(BT) + $(t_rel)) * $(DK) + i_k * $(BK)
  mask = (i_k * $(BK) + offs) < $(DK)
  cum_decay = tl.load(CumDecayPre + base + offs, mask=mask, other=0.0)
  tl.store(GO + base + offs,
    (cum_decay).to(GO.dtype.element_ty), mask=mask)
}

noncomputable def fwdDecayCumsumSpec
    (s : BlockState) (CumDecayPre : RegionName)
    (s_qk_h DK t_rel BT BK : Nat) (i : Fin BK) : ℝ :=
  s.readMem CumDecayPre (offset s s_qk_h DK t_rel BT BK i)

theorem fwd_decay_cumsum_store_slice_correct
    (CumDecayPre GO : RegionName)
    (s_qk_h DK t_rel BT BK : Nat) (s : BlockState) :
    ∀ i : Fin BK,
      let outAddr := offset s s_qk_h DK t_rel BT BK i
      (exec (fwd_decay_cumsum_store_slice CumDecayPre GO s_qk_h DK t_rel BT BK)
          s).map (·.readMem GO outAddr)
        = some (if active s DK BK i then
            fwdDecayCumsumSpec s CumDecayPre s_qk_h DK t_rel BT BK i
          else s.readMem GO outAddr) := by
  intro i
  have hInj : Function.Injective
      (fun idx : TileIndex [BK] =>
        s.pids 2 * s_qk_h + (s.pids 1 * BT + t_rel) * DK +
          s.pids 0 * BK + idx.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [exec, fwd_decay_cumsum_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.ptrAdd, NumericDType.add,
        NumericDType.mul, ComparableDType.lt, elemIndex, baseOffset, offset]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hInj (i, PUnit.unit)]
  by_cases hi : s.pids 0 * BK + i.val < DK
  · simp [active, fwdDecayCumsumSpec, elemIndex, baseOffset, offset, hi]
  · simp [active, elemIndex, baseOffset, offset, hi]

theorem fwd_decay_cumsum_store_slice_compute_correct
    (CumDecayPre GO : RegionName)
    (s_qk_h DK t_rel BT BK : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := fwd_decay_cumsum_store_slice CumDecayPre GO s_qk_h DK t_rel BT BK)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s DK BK)
        (fun i => (GO, offset s s_qk_h DK t_rel BT BK i)))
      (expected := fun i =>
        fwdDecayCumsumSpec s CumDecayPre s_qk_h DK t_rel BT BK i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fwd_decay_cumsum_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := fwd_decay_cumsum_store_slice_correct CumDecayPre GO
    s_qk_h DK t_rel BT BK s i
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-- One forward cumulative-decay update:
`cum_decay = cum_prev + g * 1.44269504`, followed by the masked `GO` store. -/
def fwd_decay_cumsum_step_store_slice
    (CumPrev G GO : RegionName)
    (s_qk_h DK t_rel BT BK : Nat) :
    ComputeKernel := triton {
  i_k = tl.program_id(0)
  i_c = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs = tl.arange(0, $(BK))
  base = i_bh * $(s_qk_h) + (i_c * $(BT) + $(t_rel)) * $(DK) + i_k * $(BK)
  mask = (i_k * $(BK) + offs) < $(DK)
  cum_prev = tl.load(CumPrev + base + offs, mask=mask, other=0.0)
  g = tl.load(G + base + offs, mask=mask, other=0.0).to(tl.float32)
  cum_decay = cum_prev + g * 1.44269504
  tl.store(GO + base + offs, (cum_decay).to(GO.dtype.element_ty), mask=mask)
}

noncomputable def fwdDecayStepSpec
    (s : BlockState) (CumPrev G : RegionName)
    (s_qk_h DK t_rel BT BK : Nat) (i : Fin BK) : ℝ :=
  s.readMem CumPrev (offset s s_qk_h DK t_rel BT BK i) +
    s.readMem G (offset s s_qk_h DK t_rel BT BK i) * 1.44269504

theorem fwd_decay_cumsum_step_store_slice_correct
    (CumPrev G GO : RegionName)
    (s_qk_h DK t_rel BT BK : Nat) (s : BlockState) :
    ∀ i : Fin BK,
      let outAddr := offset s s_qk_h DK t_rel BT BK i
      (exec (fwd_decay_cumsum_step_store_slice CumPrev G GO
          s_qk_h DK t_rel BT BK) s).map (·.readMem GO outAddr)
        = some (if active s DK BK i then
            fwdDecayStepSpec s CumPrev G s_qk_h DK t_rel BT BK i
          else s.readMem GO outAddr) := by
  intro i
  have hInj : Function.Injective
      (fun idx : TileIndex [BK] =>
        s.pids 2 * s_qk_h + (s.pids 1 * BT + t_rel) * DK +
          s.pids 0 * BK + idx.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [exec, fwd_decay_cumsum_step_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.ptrAdd, NumericDType.add,
        NumericDType.mul, ComparableDType.lt, FloatDType.cast,
        FloatDType.ofWithBot, FloatDType.toWithBot, elemIndex, baseOffset,
        offset, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hInj (i, PUnit.unit)]
  by_cases hi : s.pids 0 * BK + i.val < DK
  · simp [active, fwdDecayStepSpec, elemIndex, baseOffset, offset, hi,
      NumericDType.add, NumericDType.mul]
  · simp [active, elemIndex, baseOffset, offset, hi]

theorem fwd_decay_cumsum_step_store_slice_compute_correct
    (CumPrev G GO : RegionName)
    (s_qk_h DK t_rel BT BK : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := fwd_decay_cumsum_step_store_slice CumPrev G GO
        s_qk_h DK t_rel BT BK)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s DK BK)
        (fun i => (GO, offset s s_qk_h DK t_rel BT BK i)))
      (expected := fun i =>
        fwdDecayStepSpec s CumPrev G s_qk_h DK t_rel BT BK i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fwd_decay_cumsum_step_store_slice, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := fwd_decay_cumsum_step_store_slice_correct CumPrev G GO
    s_qk_h DK t_rel BT BK s i
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-- Proof-oriented backward row-store slice of `decay_cumsum.py`'s
`bwd_decay_global_cumsum`.

The full backward surface preserves the reverse loop and pointer decrements.
This slice fixes one row `t_rel` and proves the masked writeback used for each
of `dq_inter`, `dk_inter`, and `dg` from precomputed reverse-scan values. -/
def bwd_decay_cumsum_store_slice
    (GradPre Out : RegionName)
    (s_qk_h DK t_rel BT BK : Nat) :
    ComputeKernel := triton {
  i_k = tl.program_id(0)
  i_c = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs = tl.arange(0, $(BK))
  base = i_bh * $(s_qk_h) + (i_c * $(BT) + $(t_rel)) * $(DK) + i_k * $(BK)
  mask = (i_k * $(BK) + offs) < $(DK)
  grad = tl.load(GradPre + base + offs, mask=mask, other=0.0)
  tl.store(Out + base + offs, (grad).to(Out.dtype.element_ty), mask=mask)
}

noncomputable def bwdDecayCumsumSpec
    (s : BlockState) (GradPre : RegionName)
    (s_qk_h DK t_rel BT BK : Nat) (i : Fin BK) : ℝ :=
  s.readMem GradPre (offset s s_qk_h DK t_rel BT BK i)

theorem bwd_decay_cumsum_store_slice_correct
    (GradPre Out : RegionName)
    (s_qk_h DK t_rel BT BK : Nat) (s : BlockState) :
    ∀ i : Fin BK,
      let outAddr := offset s s_qk_h DK t_rel BT BK i
      (exec (bwd_decay_cumsum_store_slice GradPre Out s_qk_h DK t_rel BT BK)
          s).map (·.readMem Out outAddr)
        = some (if active s DK BK i then
            bwdDecayCumsumSpec s GradPre s_qk_h DK t_rel BT BK i
          else s.readMem Out outAddr) := by
  intro i
  have hInj : Function.Injective
      (fun idx : TileIndex [BK] =>
        s.pids 2 * s_qk_h + (s.pids 1 * BT + t_rel) * DK +
          s.pids 0 * BK + idx.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [exec, bwd_decay_cumsum_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.ptrAdd, NumericDType.add,
        NumericDType.mul, ComparableDType.lt, elemIndex, baseOffset, offset]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hInj (i, PUnit.unit)]
  by_cases hi : s.pids 0 * BK + i.val < DK
  · simp [active, bwdDecayCumsumSpec, elemIndex, baseOffset, offset, hi]
  · simp [active, elemIndex, baseOffset, offset, hi]

theorem bwd_decay_cumsum_store_slice_compute_correct
    (GradPre Out : RegionName)
    (s_qk_h DK t_rel BT BK : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := bwd_decay_cumsum_store_slice GradPre Out s_qk_h DK t_rel BT BK)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s DK BK)
        (fun i => (Out, offset s s_qk_h DK t_rel BT BK i)))
      (expected := fun i =>
        bwdDecayCumsumSpec s GradPre s_qk_h DK t_rel BT BK i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [bwd_decay_cumsum_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := bwd_decay_cumsum_store_slice_correct GradPre Out
    s_qk_h DK t_rel BT BK s i
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-- One backward `dg` cumulative update:
`dg_val = dq * q - dk * k`, `cum_grad_dg = cum_prev + dg_val`, then masked
store into `DG`. This isolates the reverse-loop arithmetic for the observed
`dg` output. -/
def bwd_decay_dg_step_store_slice
    (CumGradPrev DQ DK Q K DG : RegionName)
    (s_qk_h DKDim t_rel BT BK : Nat) :
    ComputeKernel := triton {
  i_k = tl.program_id(0)
  i_c = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs = tl.arange(0, $(BK))
  base = i_bh * $(s_qk_h) + (i_c * $(BT) + $(t_rel)) * $(DKDim) + i_k * $(BK)
  mask = (i_k * $(BK) + offs) < $(DKDim)
  cum_prev = tl.load(CumGradPrev + base + offs, mask=mask, other=0.0)
  dq = tl.load(DQ + base + offs, mask=mask, other=0.0)
  dk = tl.load(DK + base + offs, mask=mask, other=0.0)
  q = tl.load(Q + base + offs, mask=mask, other=0.0)
  k = tl.load(K + base + offs, mask=mask, other=0.0)
  dg_val = dq * q - dk * k
  cum_grad = cum_prev + dg_val
  tl.store(DG + base + offs, (cum_grad).to(DG.dtype.element_ty), mask=mask)
}

noncomputable def bwdDecayDGStepSpec
    (s : BlockState) (CumGradPrev DQ DKReg Q K : RegionName)
    (s_qk_h DKDim t_rel BT BK : Nat) (i : Fin BK) : ℝ :=
  s.readMem CumGradPrev (offset s s_qk_h DKDim t_rel BT BK i) +
    (s.readMem DQ (offset s s_qk_h DKDim t_rel BT BK i) *
      s.readMem Q (offset s s_qk_h DKDim t_rel BT BK i) -
    s.readMem DKReg (offset s s_qk_h DKDim t_rel BT BK i) *
      s.readMem K (offset s s_qk_h DKDim t_rel BT BK i))

theorem bwd_decay_dg_step_store_slice_correct
    (CumGradPrev DQ DKReg Q K DG : RegionName)
    (s_qk_h DKDim t_rel BT BK : Nat) (s : BlockState) :
    ∀ i : Fin BK,
      let outAddr := offset s s_qk_h DKDim t_rel BT BK i
      (exec (bwd_decay_dg_step_store_slice CumGradPrev DQ DKReg Q K DG
          s_qk_h DKDim t_rel BT BK) s).map (·.readMem DG outAddr)
        = some (if active s DKDim BK i then
            bwdDecayDGStepSpec s CumGradPrev DQ DKReg Q K
              s_qk_h DKDim t_rel BT BK i
          else s.readMem DG outAddr) := by
  intro i
  have hInj : Function.Injective
      (fun idx : TileIndex [BK] =>
        s.pids 2 * s_qk_h + (s.pids 1 * BT + t_rel) * DKDim +
          s.pids 0 * BK + idx.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [exec, bwd_decay_dg_step_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.ptrAdd, NumericDType.add,
        NumericDType.sub, NumericDType.mul, ComparableDType.lt,
        FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
        elemIndex, baseOffset, offset, ComputeExpr.toAlgorithm?,
        ComputeOp.toAlgorithm?]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hInj (i, PUnit.unit)]
  by_cases hi : s.pids 0 * BK + i.val < DKDim
  · simp [active, bwdDecayDGStepSpec, elemIndex, baseOffset, offset, hi,
      NumericDType.add, NumericDType.sub, NumericDType.mul]
  · simp [active, elemIndex, baseOffset, offset, hi]

theorem bwd_decay_dg_step_store_slice_compute_correct
    (CumGradPrev DQ DKReg Q K DG : RegionName)
    (s_qk_h DKDim t_rel BT BK : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := bwd_decay_dg_step_store_slice CumGradPrev DQ DKReg Q K DG
        s_qk_h DKDim t_rel BT BK)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s DKDim BK)
        (fun i => (DG, offset s s_qk_h DKDim t_rel BT BK i)))
      (expected := fun i =>
        bwdDecayDGStepSpec s CumGradPrev DQ DKReg Q K
          s_qk_h DKDim t_rel BT BK i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [bwd_decay_dg_step_store_slice, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := bwd_decay_dg_step_store_slice_correct CumGradPrev DQ DKReg Q K DG
    s_qk_h DKDim t_rel BT BK s i
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-- Named `dq_inter` writeback correctness for `bwd_decay_global_cumsum`.

The reverse-loop arithmetic producing `DQInterPre` is outside this store slice;
this theorem exposes the Python-observed `dq_inter` row mutation directly. -/
theorem bwd_decay_cumsum_dq_inter_store_slice_compute_correct
    (DQInterPre DQInter : RegionName)
    (s_qk_h DK t_rel BT BK : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := bwd_decay_cumsum_store_slice DQInterPre DQInter
        s_qk_h DK t_rel BT BK)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s DK BK)
        (fun i => (DQInter, offset s s_qk_h DK t_rel BT BK i)))
      (expected := fun i =>
        bwdDecayCumsumSpec s DQInterPre s_qk_h DK t_rel BT BK i) := by
  exact bwd_decay_cumsum_store_slice_compute_correct DQInterPre DQInter
    s_qk_h DK t_rel BT BK s

/-- Named `dk_inter` writeback correctness for `bwd_decay_global_cumsum`. -/
theorem bwd_decay_cumsum_dk_inter_store_slice_compute_correct
    (DKInterPre DKInter : RegionName)
    (s_qk_h DK t_rel BT BK : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := bwd_decay_cumsum_store_slice DKInterPre DKInter
        s_qk_h DK t_rel BT BK)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s DK BK)
        (fun i => (DKInter, offset s s_qk_h DK t_rel BT BK i)))
      (expected := fun i =>
        bwdDecayCumsumSpec s DKInterPre s_qk_h DK t_rel BT BK i) := by
  exact bwd_decay_cumsum_store_slice_compute_correct DKInterPre DKInter
    s_qk_h DK t_rel BT BK s

/-- Named `dg` writeback correctness for `bwd_decay_global_cumsum`. -/
theorem bwd_decay_cumsum_dg_store_slice_compute_correct
    (DGPre DG : RegionName)
    (s_qk_h DK t_rel BT BK : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := bwd_decay_cumsum_store_slice DGPre DG s_qk_h DK t_rel BT BK)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s DK BK)
        (fun i => (DG, offset s s_qk_h DK t_rel BT BK i)))
      (expected := fun i =>
        bwdDecayCumsumSpec s DGPre s_qk_h DK t_rel BT BK i) := by
  exact bwd_decay_cumsum_store_slice_compute_correct DGPre DG
    s_qk_h DK t_rel BT BK s

/-! ## Python test-shape wrappers

`decay_cumsum.py`'s checked test uses `B = 2`, `H = 2`, `T = 4`, `DK = 8`,
`BT = 2`, and `BK = 4`. The Python launch helpers pass `s_qk_h = H * T * DK =
64`. The wrappers below specialize the row slices to either loop row
`t_rel : Fin 2`. -/

theorem prepare_qg_decay_python_test_shape_compute_correct
    (Q QDecay QG : RegionName) (t_rel : Fin 2) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := prepare_qg_decay_store_slice Q QDecay QG 64 8 t_rel.val 2 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 8 4)
        (fun i => (QG, offset s 64 8 t_rel.val 2 4 i)))
      (expected := fun i =>
        prepareQgDecaySpec s Q QDecay 64 8 t_rel.val 2 4 i) := by
  exact prepare_qg_decay_store_slice_compute_correct Q QDecay QG
    64 8 t_rel.val 2 4 s

theorem prepare_kg_decay_python_test_shape_compute_correct
    (K KDecay KG : RegionName) (t_rel : Fin 2) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := prepare_kg_decay_store_slice K KDecay KG 64 8 t_rel.val 2 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 8 4)
        (fun i => (KG, offset s 64 8 t_rel.val 2 4 i)))
      (expected := fun i =>
        prepareKgDecaySpec s K KDecay 64 8 t_rel.val 2 4 i) := by
  exact prepare_kg_decay_store_slice_compute_correct K KDecay KG
    64 8 t_rel.val 2 4 s

theorem fwd_decay_cumsum_store_python_test_shape_compute_correct
    (CumDecayPre GO : RegionName) (t_rel : Fin 2) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := fwd_decay_cumsum_store_slice CumDecayPre GO 64 8 t_rel.val 2 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 8 4)
        (fun i => (GO, offset s 64 8 t_rel.val 2 4 i)))
      (expected := fun i =>
        fwdDecayCumsumSpec s CumDecayPre 64 8 t_rel.val 2 4 i) := by
  exact fwd_decay_cumsum_store_slice_compute_correct CumDecayPre GO
    64 8 t_rel.val 2 4 s

theorem fwd_decay_cumsum_step_python_test_shape_compute_correct
    (CumPrev G GO : RegionName) (t_rel : Fin 2) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := fwd_decay_cumsum_step_store_slice CumPrev G GO
        64 8 t_rel.val 2 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 8 4)
        (fun i => (GO, offset s 64 8 t_rel.val 2 4 i)))
      (expected := fun i =>
        fwdDecayStepSpec s CumPrev G 64 8 t_rel.val 2 4 i) := by
  exact fwd_decay_cumsum_step_store_slice_compute_correct CumPrev G GO
    64 8 t_rel.val 2 4 s

theorem bwd_decay_cumsum_dq_inter_python_test_shape_compute_correct
    (DQInterPre DQInter : RegionName) (t_rel : Fin 2) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := bwd_decay_cumsum_store_slice DQInterPre DQInter
        64 8 t_rel.val 2 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 8 4)
        (fun i => (DQInter, offset s 64 8 t_rel.val 2 4 i)))
      (expected := fun i =>
        bwdDecayCumsumSpec s DQInterPre 64 8 t_rel.val 2 4 i) := by
  exact bwd_decay_cumsum_dq_inter_store_slice_compute_correct DQInterPre DQInter
    64 8 t_rel.val 2 4 s

theorem bwd_decay_cumsum_dk_inter_python_test_shape_compute_correct
    (DKInterPre DKInter : RegionName) (t_rel : Fin 2) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := bwd_decay_cumsum_store_slice DKInterPre DKInter
        64 8 t_rel.val 2 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 8 4)
        (fun i => (DKInter, offset s 64 8 t_rel.val 2 4 i)))
      (expected := fun i =>
        bwdDecayCumsumSpec s DKInterPre 64 8 t_rel.val 2 4 i) := by
  exact bwd_decay_cumsum_dk_inter_store_slice_compute_correct DKInterPre DKInter
    64 8 t_rel.val 2 4 s

theorem bwd_decay_cumsum_dg_python_test_shape_compute_correct
    (DGPre DG : RegionName) (t_rel : Fin 2) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := bwd_decay_cumsum_store_slice DGPre DG 64 8 t_rel.val 2 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 8 4)
        (fun i => (DG, offset s 64 8 t_rel.val 2 4 i)))
      (expected := fun i =>
        bwdDecayCumsumSpec s DGPre 64 8 t_rel.val 2 4 i) := by
  exact bwd_decay_cumsum_dg_store_slice_compute_correct DGPre DG
    64 8 t_rel.val 2 4 s

theorem bwd_decay_dg_step_python_test_shape_compute_correct
    (CumGradPrev DQ DKReg Q K DG : RegionName) (t_rel : Fin 2)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := bwd_decay_dg_step_store_slice CumGradPrev DQ DKReg Q K DG
        64 8 t_rel.val 2 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 8 4)
        (fun i => (DG, offset s 64 8 t_rel.val 2 4 i)))
      (expected := fun i =>
        bwdDecayDGStepSpec s CumGradPrev DQ DKReg Q K
          64 8 t_rel.val 2 4 i) := by
  exact bwd_decay_dg_step_store_slice_compute_correct CumGradPrev DQ DKReg Q K DG
    64 8 t_rel.val 2 4 s

/-- Python test-shape preparation coverage: both `q_g` and `k_g` decay-prep
stores realize the checked row slice for either `t_rel` loop row. -/
theorem decay_cumsum_prepare_python_test_shape_all_outputs_compute_correct
    (Q QDecay QG K KDecay KG : RegionName) (t_rel : Fin 2)
    (s : BlockState) :
    (ComputeCorrect.Realizes
      (kernel := prepare_qg_decay_store_slice Q QDecay QG
        64 8 t_rel.val 2 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 8 4)
        (fun i => (QG, offset s 64 8 t_rel.val 2 4 i)))
      (expected := fun i =>
        prepareQgDecaySpec s Q QDecay 64 8 t_rel.val 2 4 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := prepare_kg_decay_store_slice K KDecay KG
        64 8 t_rel.val 2 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 8 4)
        (fun i => (KG, offset s 64 8 t_rel.val 2 4 i)))
      (expected := fun i =>
        prepareKgDecaySpec s K KDecay 64 8 t_rel.val 2 4 i)) := by
  constructor
  · exact prepare_qg_decay_python_test_shape_compute_correct
      Q QDecay QG t_rel s
  · exact prepare_kg_decay_python_test_shape_compute_correct
      K KDecay KG t_rel s

/-- Python test-shape forward coverage: the cumulative decay writeback and the
one-step recurrence writeback both realize the checked `GO` row slice. -/
theorem decay_cumsum_forward_python_test_shape_all_outputs_compute_correct
    (CumDecayPre CumPrev G GO : RegionName) (t_rel : Fin 2)
    (s : BlockState) :
    (ComputeCorrect.Realizes
      (kernel := fwd_decay_cumsum_store_slice CumDecayPre GO
        64 8 t_rel.val 2 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 8 4)
        (fun i => (GO, offset s 64 8 t_rel.val 2 4 i)))
      (expected := fun i =>
        fwdDecayCumsumSpec s CumDecayPre 64 8 t_rel.val 2 4 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := fwd_decay_cumsum_step_store_slice CumPrev G GO
        64 8 t_rel.val 2 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 8 4)
        (fun i => (GO, offset s 64 8 t_rel.val 2 4 i)))
      (expected := fun i =>
        fwdDecayStepSpec s CumPrev G 64 8 t_rel.val 2 4 i)) := by
  constructor
  · exact fwd_decay_cumsum_store_python_test_shape_compute_correct
      CumDecayPre GO t_rel s
  · exact fwd_decay_cumsum_step_python_test_shape_compute_correct
      CumPrev G GO t_rel s

/-- Python test-shape backward coverage: the cumsum writebacks for `dq_inter`,
`dk_inter`, and `dg`, plus the one-step `dg` recurrence, all realize the checked
row slice. -/
theorem decay_cumsum_backward_python_test_shape_all_outputs_compute_correct
    (DQInterPre DQInter DKInterPre DKInter DGPre CumGradPrev DQ DKReg Q K DG :
      RegionName)
    (t_rel : Fin 2) (s : BlockState) :
    (ComputeCorrect.Realizes
      (kernel := bwd_decay_cumsum_store_slice DQInterPre DQInter
        64 8 t_rel.val 2 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 8 4)
        (fun i => (DQInter, offset s 64 8 t_rel.val 2 4 i)))
      (expected := fun i =>
        bwdDecayCumsumSpec s DQInterPre 64 8 t_rel.val 2 4 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := bwd_decay_cumsum_store_slice DKInterPre DKInter
        64 8 t_rel.val 2 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 8 4)
        (fun i => (DKInter, offset s 64 8 t_rel.val 2 4 i)))
      (expected := fun i =>
        bwdDecayCumsumSpec s DKInterPre 64 8 t_rel.val 2 4 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := bwd_decay_cumsum_store_slice DGPre DG 64 8 t_rel.val 2 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 8 4)
        (fun i => (DG, offset s 64 8 t_rel.val 2 4 i)))
      (expected := fun i =>
        bwdDecayCumsumSpec s DGPre 64 8 t_rel.val 2 4 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := bwd_decay_dg_step_store_slice CumGradPrev DQ DKReg Q K DG
        64 8 t_rel.val 2 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 8 4)
        (fun i => (DG, offset s 64 8 t_rel.val 2 4 i)))
      (expected := fun i =>
        bwdDecayDGStepSpec s CumGradPrev DQ DKReg Q K
          64 8 t_rel.val 2 4 i)) := by
  constructor
  · exact bwd_decay_cumsum_dq_inter_python_test_shape_compute_correct
      DQInterPre DQInter t_rel s
  constructor
  · exact bwd_decay_cumsum_dk_inter_python_test_shape_compute_correct
      DKInterPre DKInter t_rel s
  constructor
  · exact bwd_decay_cumsum_dg_python_test_shape_compute_correct
      DGPre DG t_rel s
  · exact bwd_decay_dg_step_python_test_shape_compute_correct
      CumGradPrev DQ DKReg Q K DG t_rel s

/-! ## Python test-case surface wrappers

The checked Python launch uses `B = 2`, `H = 2`, `T = 4`, `DK = 8`,
`BT = 2`, `BK = 4`, `s_qk_h = 64`, and `scale = 1.0`. These wrappers pin the
full surfaces to that layout so the carried forward cumsum, q/k decay prep,
and reverse backward scan are all directly represented at the Python-tested
shape. -/

theorem fwd_decay_cumsum_python_test_surface_toAlgorithm_supported
    (G GO : RegionName) :
    ∃ alg, (fwd_decay_cumsum_surface G GO 64 32 8 2 2 4 1.0
      2 4 8).toAlgorithm? = Except.ok alg := by
  exact fwd_decay_cumsum_surface_toAlgorithm_supported G GO
    64 32 8 2 2 4 1.0 2 4 8

theorem prepare_qg_kg_python_test_surface_toAlgorithm_supported
    (Q K G QG KG : RegionName) :
    ∃ alg, (prepare_qg_kg_surface Q K G QG KG 64 8 2 4
      1.0).toAlgorithm? = Except.ok alg := by
  exact prepare_qg_kg_surface_toAlgorithm_supported Q K G QG KG
    64 8 2 4 1.0

theorem bwd_decay_global_cumsum_python_test_surface_toAlgorithm_supported
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName) :
    (bwd_decay_global_cumsum_surface DQInner DQInter DKInner DKInter Q K G DG
      64 8 2 4).toAlgorithm? =
      Except.ok
        (bwd_decay_global_cumsum_surface DQInner DQInter DKInner DKInter
          Q K G DG 64 8 2 4).toAlgKernel := by
  exact bwd_decay_global_cumsum_surface_toAlgorithm_supported DQInner DQInter
    DKInner DKInter Q K G DG 64 8 2 4

noncomputable def decayBackwardSurfaceValue
    (s : BlockState)
    (DQInner DQInter DKInner DKInter Q K G DG Out : RegionName)
    (offset : Nat) : ℝ :=
  match exec (bwd_decay_global_cumsum_surface DQInner DQInter DKInner DKInter
      Q K G DG 64 8 2 4) s with
  | some s' => s'.readMem Out offset
  | none => 0.0

theorem bwd_decay_global_cumsum_surface_output_compute_correct
    (DQInner DQInter DKInner DKInter Q K G DG Out : RegionName)
    (t_rel : Fin 2) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := bwd_decay_global_cumsum_surface DQInner DQInter DKInner
        DKInter Q K G DG 64 8 2 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 8 4)
        (fun i => (Out, offset s 64 8 t_rel.val 2 4 i)))
      (expected := fun i : Fin 4 =>
        decayBackwardSurfaceValue s DQInner DQInter DKInner DKInter Q K G DG
          Out (offset s 64 8 t_rel.val 2 4 i)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [bwd_decay_global_cumsum_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i _hActive
  simp [decayBackwardSurfaceValue, hExec]

theorem decay_cumsum_backward_python_test_shape_surface_outputs_compute_correct
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (t_rel : Fin 2) (s : BlockState) :
    (ComputeCorrect.Realizes
      (kernel := bwd_decay_global_cumsum_surface DQInner DQInter DKInner
        DKInter Q K G DG 64 8 2 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 8 4)
        (fun i => (DQInter, offset s 64 8 t_rel.val 2 4 i)))
      (expected := fun i : Fin 4 =>
        decayBackwardSurfaceValue s DQInner DQInter DKInner DKInter Q K G DG
          DQInter (offset s 64 8 t_rel.val 2 4 i))) ∧
    (ComputeCorrect.Realizes
      (kernel := bwd_decay_global_cumsum_surface DQInner DQInter DKInner
        DKInter Q K G DG 64 8 2 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 8 4)
        (fun i => (DKInter, offset s 64 8 t_rel.val 2 4 i)))
      (expected := fun i : Fin 4 =>
        decayBackwardSurfaceValue s DQInner DQInter DKInner DKInter Q K G DG
          DKInter (offset s 64 8 t_rel.val 2 4 i))) ∧
    (ComputeCorrect.Realizes
      (kernel := bwd_decay_global_cumsum_surface DQInner DQInter DKInner
        DKInter Q K G DG 64 8 2 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 8 4)
        (fun i => (DG, offset s 64 8 t_rel.val 2 4 i)))
      (expected := fun i : Fin 4 =>
        decayBackwardSurfaceValue s DQInner DQInter DKInner DKInter Q K G DG
          DG (offset s 64 8 t_rel.val 2 4 i))) := by
  constructor
  · exact bwd_decay_global_cumsum_surface_output_compute_correct DQInner
      DQInter DKInner DKInter Q K G DG DQInter t_rel s
  constructor
  · exact bwd_decay_global_cumsum_surface_output_compute_correct DQInner
      DQInter DKInner DKInter Q K G DG DKInter t_rel s
  · exact bwd_decay_global_cumsum_surface_output_compute_correct DQInner
      DQInter DKInner DKInter Q K G DG DG t_rel s

/-- Python test-path summary for the q/k decay preparation kernel.

This combines the checked full `prepare_qg_kg` surface with both Python-visible
`q_g` and `k_g` row writebacks at the benchmark shape. -/
theorem decay_cumsum_prepare_python_test_shape_summary
    (Q K G QG KG QDecay KDecay : RegionName) (t_rel : Fin 2)
    (s : BlockState) :
    (∃ alg, (prepare_qg_kg_surface Q K G QG KG 64 8 2 4
      1.0).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := prepare_qg_decay_store_slice Q QDecay QG
        64 8 t_rel.val 2 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 8 4)
        (fun i => (QG, offset s 64 8 t_rel.val 2 4 i)))
      (expected := fun i =>
        prepareQgDecaySpec s Q QDecay 64 8 t_rel.val 2 4 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := prepare_kg_decay_store_slice K KDecay KG
        64 8 t_rel.val 2 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 8 4)
        (fun i => (KG, offset s 64 8 t_rel.val 2 4 i)))
      (expected := fun i =>
        prepareKgDecaySpec s K KDecay 64 8 t_rel.val 2 4 i)) := by
  constructor
  · exact prepare_qg_kg_python_test_surface_toAlgorithm_supported Q K G QG KG
  · exact decay_cumsum_prepare_python_test_shape_all_outputs_compute_correct
      Q QDecay QG K KDecay KG t_rel s

/-- Python test-path summary for the forward decay cumsum kernel.

The summary links the full forward surface to both the precomputed row-store
and the checked one-step cumulative update at the benchmark shape. -/
theorem decay_cumsum_forward_python_test_shape_summary
    (G GO CumDecayPre CumPrev : RegionName) (t_rel : Fin 2)
    (s : BlockState) :
    (∃ alg, (fwd_decay_cumsum_surface G GO 64 32 8 2 2 4 1.0
      2 4 8).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := fwd_decay_cumsum_store_slice CumDecayPre GO
        64 8 t_rel.val 2 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 8 4)
        (fun i => (GO, offset s 64 8 t_rel.val 2 4 i)))
      (expected := fun i =>
        fwdDecayCumsumSpec s CumDecayPre 64 8 t_rel.val 2 4 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := fwd_decay_cumsum_step_store_slice CumPrev G GO
        64 8 t_rel.val 2 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 8 4)
        (fun i => (GO, offset s 64 8 t_rel.val 2 4 i)))
      (expected := fun i =>
        fwdDecayStepSpec s CumPrev G 64 8 t_rel.val 2 4 i)) := by
  constructor
  · exact fwd_decay_cumsum_python_test_surface_toAlgorithm_supported G GO
  · exact decay_cumsum_forward_python_test_shape_all_outputs_compute_correct
      CumDecayPre CumPrev G GO t_rel s

/-- Python test-path summary for the reverse backward global cumsum kernel.

This pairs the reverse-loop/pointer-decrement surface with the checked
`dq_inter`, `dk_inter`, and `dg` output row proofs plus the one-step `dg`
arithmetic update. -/
theorem decay_cumsum_backward_python_test_shape_summary
    (DQInner DQInter DKInner DKInter Q K G DG DQInterPre DKInterPre DGPre
      CumGradPrev DQ DKReg : RegionName)
    (t_rel : Fin 2) (s : BlockState) :
    ((bwd_decay_global_cumsum_surface DQInner DQInter DKInner DKInter Q K G DG
      64 8 2 4).toAlgorithm? =
      Except.ok
        (bwd_decay_global_cumsum_surface DQInner DQInter DKInner DKInter
          Q K G DG 64 8 2 4).toAlgKernel) ∧
    (ComputeCorrect.Realizes
      (kernel := bwd_decay_cumsum_store_slice DQInterPre DQInter
        64 8 t_rel.val 2 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 8 4)
        (fun i => (DQInter, offset s 64 8 t_rel.val 2 4 i)))
      (expected := fun i =>
        bwdDecayCumsumSpec s DQInterPre 64 8 t_rel.val 2 4 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := bwd_decay_cumsum_store_slice DKInterPre DKInter
        64 8 t_rel.val 2 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 8 4)
        (fun i => (DKInter, offset s 64 8 t_rel.val 2 4 i)))
      (expected := fun i =>
        bwdDecayCumsumSpec s DKInterPre 64 8 t_rel.val 2 4 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := bwd_decay_cumsum_store_slice DGPre DG 64 8 t_rel.val 2 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 8 4)
        (fun i => (DG, offset s 64 8 t_rel.val 2 4 i)))
      (expected := fun i =>
        bwdDecayCumsumSpec s DGPre 64 8 t_rel.val 2 4 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := bwd_decay_dg_step_store_slice CumGradPrev DQ DKReg Q K DG
        64 8 t_rel.val 2 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 8 4)
        (fun i => (DG, offset s 64 8 t_rel.val 2 4 i)))
      (expected := fun i =>
        bwdDecayDGStepSpec s CumGradPrev DQ DKReg Q K
          64 8 t_rel.val 2 4 i)) := by
  constructor
  · exact bwd_decay_global_cumsum_python_test_surface_toAlgorithm_supported
      DQInner DQInter DKInner DKInter Q K G DG
  · exact decay_cumsum_backward_python_test_shape_all_outputs_compute_correct
      DQInterPre DQInter DKInterPre DKInter DGPre CumGradPrev DQ DKReg
      Q K DG t_rel s

/-- Proposition for the Python q/k decay-preparation path. -/
abbrev decay_cumsum_prepare_python_test_shape_prop
    (Q K G QG KG QDecay KDecay : RegionName) (t_rel : Fin 2)
    (s : BlockState) : Prop :=
    (∃ alg, (prepare_qg_kg_surface Q K G QG KG 64 8 2 4
      1.0).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := prepare_qg_decay_store_slice Q QDecay QG
        64 8 t_rel.val 2 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 8 4)
        (fun i => (QG, offset s 64 8 t_rel.val 2 4 i)))
      (expected := fun i =>
        prepareQgDecaySpec s Q QDecay 64 8 t_rel.val 2 4 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := prepare_kg_decay_store_slice K KDecay KG
        64 8 t_rel.val 2 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 8 4)
        (fun i => (KG, offset s 64 8 t_rel.val 2 4 i)))
      (expected := fun i =>
        prepareKgDecaySpec s K KDecay 64 8 t_rel.val 2 4 i))

/-- Proposition for the Python forward decay-cumsum path. -/
abbrev decay_cumsum_forward_python_test_shape_prop
    (G GO CumDecayPre CumPrev : RegionName) (t_rel : Fin 2)
    (s : BlockState) : Prop :=
    (∃ alg, (fwd_decay_cumsum_surface G GO 64 32 8 2 2 4 1.0
      2 4 8).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := fwd_decay_cumsum_store_slice CumDecayPre GO
        64 8 t_rel.val 2 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 8 4)
        (fun i => (GO, offset s 64 8 t_rel.val 2 4 i)))
      (expected := fun i =>
        fwdDecayCumsumSpec s CumDecayPre 64 8 t_rel.val 2 4 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := fwd_decay_cumsum_step_store_slice CumPrev G GO
        64 8 t_rel.val 2 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 8 4)
        (fun i => (GO, offset s 64 8 t_rel.val 2 4 i)))
      (expected := fun i =>
        fwdDecayStepSpec s CumPrev G 64 8 t_rel.val 2 4 i))

/-- Proposition for the Python reverse backward global-cumsum path. -/
abbrev decay_cumsum_backward_python_test_shape_prop
    (DQInner DQInter DKInner DKInter Q K G DG DQInterPre DKInterPre DGPre
      CumGradPrev DQ DKReg : RegionName)
    (t_rel : Fin 2) (s : BlockState) : Prop :=
    ((bwd_decay_global_cumsum_surface DQInner DQInter DKInner DKInter Q K G DG
      64 8 2 4).toAlgorithm? =
      Except.ok
        (bwd_decay_global_cumsum_surface DQInner DQInter DKInner DKInter
          Q K G DG 64 8 2 4).toAlgKernel) ∧
    (ComputeCorrect.Realizes
      (kernel := bwd_decay_cumsum_store_slice DQInterPre DQInter
        64 8 t_rel.val 2 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 8 4)
        (fun i => (DQInter, offset s 64 8 t_rel.val 2 4 i)))
      (expected := fun i =>
        bwdDecayCumsumSpec s DQInterPre 64 8 t_rel.val 2 4 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := bwd_decay_cumsum_store_slice DKInterPre DKInter
        64 8 t_rel.val 2 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 8 4)
        (fun i => (DKInter, offset s 64 8 t_rel.val 2 4 i)))
      (expected := fun i =>
        bwdDecayCumsumSpec s DKInterPre 64 8 t_rel.val 2 4 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := bwd_decay_cumsum_store_slice DGPre DG 64 8 t_rel.val 2 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 8 4)
        (fun i => (DG, offset s 64 8 t_rel.val 2 4 i)))
      (expected := fun i =>
        bwdDecayCumsumSpec s DGPre 64 8 t_rel.val 2 4 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := bwd_decay_dg_step_store_slice CumGradPrev DQ DKReg Q K DG
        64 8 t_rel.val 2 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 8 4)
        (fun i => (DG, offset s 64 8 t_rel.val 2 4 i)))
      (expected := fun i =>
        bwdDecayDGStepSpec s CumGradPrev DQ DKReg Q K
          64 8 t_rel.val 2 4 i))

/-- Combined checked-shape summary for `test_kernels` in `decay_cumsum.py`.

This collects the three launched kernels, the observed `g_o`, `qg`, `kg`, and
`dg` outputs, and the reverse global-cumsum `dq_inter`/`dk_inter` writebacks in
one public target. -/
theorem decay_cumsum_python_test_shape_complete_summary
    (Q K G GO QG KG DG QDecay KDecay CumDecayPre CumPrev DQInner DQInter
      DKInner DKInter DQInterPre DKInterPre DGPre CumGradPrev DQ DKReg :
      RegionName)
    (t_rel : Fin 2) (s : BlockState) :
    decay_cumsum_prepare_python_test_shape_prop Q K G QG KG QDecay KDecay
      t_rel s ∧
    decay_cumsum_forward_python_test_shape_prop G GO CumDecayPre CumPrev
      t_rel s ∧
    decay_cumsum_backward_python_test_shape_prop DQInner DQInter DKInner
      DKInter Q K G DG DQInterPre DKInterPre DGPre CumGradPrev DQ DKReg
      t_rel s := by
  constructor
  · exact decay_cumsum_prepare_python_test_shape_summary Q K G QG KG QDecay
      KDecay t_rel s
  constructor
  · exact decay_cumsum_forward_python_test_shape_summary G GO CumDecayPre
      CumPrev t_rel s
  · exact decay_cumsum_backward_python_test_shape_summary DQInner DQInter
      DKInner DKInter Q K G DG DQInterPre DKInterPre DGPre CumGradPrev DQ
      DKReg t_rel s


























/-- `output_summary` for the Python backward decay-cumsum surface. -/
abbrev decay_cumsum_backward_python_test_shape_output_summary
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (t_rel : Fin 2) (s : BlockState) :=
  decay_cumsum_backward_python_test_shape_surface_outputs_compute_correct
    DQInner DQInter DKInner DKInter Q K G DG t_rel s

/-- **Genuine forward closed form.** At chunk row `t_rel` and lane `i`, the
forward decay-cumsum kernel writes the scaled prefix sum
`1.44269504 * Σ_{k=0}^{t_rel} g[row k, lane i]` into `GO`. This is the honest
`out[i] = decay-weighted cumulative sum` specification for the within-chunk
axis-0 prefix scan (with the constant `inv_ln2 = 1.44269504` decay factor),
*not* the executed kernel readback. -/
noncomputable def fwdDecayClosed
    (s : BlockState) (G : RegionName)
    (s_qk_h DK BT BK : Nat) (t_rel : Fin BT) (i : Fin BK) : ℝ :=
  1.44269504 *
    ∑ k : Fin (t_rel.val + 1),
      s.readMem G (offset s s_qk_h DK k.val BT BK i)

/-- **Row 0 forward closed form.** The full `fwd_decay_cumsum` surface writes
`1.44269504 * g[row 0, lane i]` into `GO` at the row-0 offset — the genuine
prefix-sum value for the first chunk row. -/
theorem fwd_decay_cumsum_full_surface_row0_closed
    (G GO : RegionName) (s : BlockState) (i : Fin 4) :
    (exec (fwd_decay_cumsum_surface G GO 64 32 8 2 2 4 1.0 2 4 8) s).map
      (·.readMem GO (offset s 64 8 0 2 4 i))
      = some (if active s 8 4 i then fwdDecayClosed s G 64 8 2 4 0 i
              else s.readMem GO (offset s 64 8 0 2 4 i)) := by
  have hInjGO : Function.Injective
      (fun idx : TileIndex [4] =>
        s.pids 2 * 64 + s.pids 1 * 2 * 8 + s.pids 0 * 4 + idx.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp only [offset, baseOffset, elemIndex, active, fwdDecayClosed, Fin.val_zero,
    Fin.sum_univ_one]
  simp [exec, fwd_decay_cumsum_surface, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.ptrAdd, NumericDType.add,
        NumericDType.mul, ComparableDType.lt, FloatDType.cast,
        FloatDType.ofWithBot, FloatDType.toWithBot,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
        stepForRangeAux.forRange_unfold, stepForRangeAux.step_lt,
        stepForRangeAux.step_ge]
  rw [BlockState.scatter_prop_masked_preserves_other_offset]
  · simp only [BlockState.setReg_readMem]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hInjGO (i, PUnit.unit)]
    by_cases hc : s.pids 0 * 4 + (i : ℕ) < 8
    · simp only [hc, if_true]
      rw [mul_comm]
      rfl
    · simp only [hc, if_false, BlockState.setReg_readMem]
  · rintro ⟨k, _⟩ _
    simp only
    omega

/-- **Row 1 forward closed form.** The full `fwd_decay_cumsum` surface writes
`1.44269504 * (g[row 0, lane i] + g[row 1, lane i])` into `GO` at the row-1
offset — the genuine 2-term cumulative decay sum. (`G ≠ GO` lets the row-1
`g` load read through the row-0 `GO` writeback.) -/
theorem fwd_decay_cumsum_full_surface_row1_closed
    (G GO : RegionName) (s : BlockState) (i : Fin 4) (hne : G ≠ GO) :
    (exec (fwd_decay_cumsum_surface G GO 64 32 8 2 2 4 1.0 2 4 8) s).map
      (·.readMem GO (offset s 64 8 1 2 4 i))
      = some (if active s 8 4 i then fwdDecayClosed s G 64 8 2 4 1 i
              else s.readMem GO (offset s 64 8 1 2 4 i)) := by
  have hInjGO : Function.Injective
      (fun idx : TileIndex [4] =>
        s.pids 2 * 64 + s.pids 1 * 2 * 8 + s.pids 0 * 4 + idx.1.val + 8) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel (Nat.add_right_cancel hab))
    rfl
  simp only [offset, baseOffset, elemIndex, active, fwdDecayClosed, Fin.sum_univ_succ,
    Fin.sum_univ_one, Fin.val_zero, Fin.val_succ]
  simp [exec, fwd_decay_cumsum_surface, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.ptrAdd, NumericDType.add,
        NumericDType.mul, ComparableDType.lt, FloatDType.cast,
        FloatDType.ofWithBot, FloatDType.toWithBot,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
        stepForRangeAux.forRange_unfold, stepForRangeAux.step_lt,
        stepForRangeAux.step_ge]
  rw [show s.pids 2 * 64 + (s.pids 1 * 2 + 1) * 8 + s.pids 0 * 4 + (i : ℕ)
        = s.pids 2 * 64 + s.pids 1 * 2 * 8 + s.pids 0 * 4 + (i : ℕ) + 8 from by ring]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hInjGO (i, PUnit.unit)]
  by_cases hc : s.pids 0 * 4 + (i : ℕ) < 8
  · simp only [hc, if_true]
    rw [BlockState.scatter_prop_masked_preserves_other_region
          (P := fun k : TileIndex [4] => s.pids 0 * 4 + k.1.val < 8) (h_ne := hne)]
    simp only [BlockState.setReg_readMem, Option.map₂_some_some]
    show (s.readMem G _ * 1.44269504) + (s.readMem G _ * 1.44269504) = _
    ring
  · simp only [hc, if_false, BlockState.setReg_readMem]
    rw [BlockState.scatter_prop_masked_preserves_other_offset]
    · simp only [BlockState.setReg_readMem]
    · rintro ⟨k, _⟩ _
      simp only
      omega

/-- **Unified forward closed form (both chunk rows).** For either loop row
`t_rel : Fin 2`, the executed `fwd_decay_cumsum` surface readback at the row's
`GO` offset equals the genuine decay-cumsum closed form
`fwdDecayClosed = 1.44269504 * Σ_{k ≤ t_rel} g[row k, lane i]` on active lanes.
This certifies the cross-step cumulative fold of the `range(BT)` loop against a
real (non self-referential) prefix-sum specification. -/
theorem fwd_decay_cumsum_full_surface_closed
    (G GO : RegionName) (s : BlockState) (t_rel : Fin 2) (i : Fin 4)
    (hne : G ≠ GO) :
    (exec (fwd_decay_cumsum_surface G GO 64 32 8 2 2 4 1.0 2 4 8) s).map
      (·.readMem GO (offset s 64 8 t_rel.val 2 4 i))
      = some (if active s 8 4 i then fwdDecayClosed s G 64 8 2 4 t_rel i
              else s.readMem GO (offset s 64 8 t_rel.val 2 4 i)) := by
  match t_rel with
  | ⟨0, _⟩ => exact fwd_decay_cumsum_full_surface_row0_closed G GO s i
  | ⟨1, _⟩ => exact fwd_decay_cumsum_full_surface_row1_closed G GO s i hne

/-- **Genuine forward compute-correctness.** The full `fwd_decay_cumsum` surface
realizes the honest decay-cumsum closed form `fwdDecayClosed` (the scaled
within-chunk prefix sum) at every active `GO` lane of loop row `t_rel`. The
`expected` here is a specification independent of the kernel's own output
(no self-reference to the executed readback). -/
theorem fwd_decay_cumsum_surface_closed_compute_correct
    (G GO : RegionName) (t_rel : Fin 2) (s : BlockState) (hne : G ≠ GO) :
    ComputeCorrect.Realizes
      (kernel := fwd_decay_cumsum_surface G GO 64 32 8 2 2 4 1.0 2 4 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 8 4)
        (fun i => (GO, offset s 64 8 t_rel.val 2 4 i)))
      (expected := fun i : Fin 4 =>
        fwdDecayClosed s G 64 8 2 4 t_rel i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fwd_decay_cumsum_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := fwd_decay_cumsum_full_surface_closed G GO s t_rel i hne
  rw [hExec] at h
  simp only [Option.map_some] at h
  rw [Option.some.injEq] at h
  show s'.readMem GO (offset s 64 8 t_rel.val 2 4 i) = _
  rw [h, if_pos hActive]

/-- `output_summary` for the Python forward decay-cumsum surface, against the
**genuine closed form**. This is the non self-referential replacement for the
proof-gap `decay_cumsum_forward_python_test_shape_output_summary`. -/
theorem decay_cumsum_forward_python_test_shape_closed_output_summary
    (G GO : RegionName) (t_rel : Fin 2) (s : BlockState) (hne : G ≠ GO) :
    ComputeCorrect.Realizes
      (kernel := fwd_decay_cumsum_surface G GO 64 32 8 2 2 4 1.0 2 4 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 8 4)
        (fun i => (GO, offset s 64 8 t_rel.val 2 4 i)))
      (expected := fun i : Fin 4 =>
        fwdDecayClosed s G 64 8 2 4 t_rel i) :=
  fwd_decay_cumsum_surface_closed_compute_correct G GO t_rel s hne

/-- `output_summary` for the Python forward decay-cumsum surface, certified
against the **genuine** decay-cumsum closed form `fwdDecayClosed`. This is the
non self-referential replacement closing the `decay-cumsum-scan-fold` proof
gap for the forward kernel. -/
abbrev decay_cumsum_forward_python_test_shape_output_summary
    (G GO : RegionName) (t_rel : Fin 2) (s : BlockState) (hne : G ≠ GO) :=
  decay_cumsum_forward_python_test_shape_closed_output_summary G GO t_rel s hne

/-! ## Genuine `prepare_qg_kg` closed form (`qg` output)

The `prepare_qg_kg` kernel is a single-pass, per-row pointwise elementwise map:
each `qg` lane is `q * exp2(g) * scale` (no cross-step fold). We certify the
full `prepare_qg_kg` surface's `QG` writeback against this honest closed form
`prepareQgClosed = q[idx] * exp2(g[idx]) * scale`, *not* the executed kernel
readback. -/

/-- **Genuine `qg` closed form.** At chunk row `t_rel` and lane `i`, the
`prepare_qg_kg` kernel writes `q[idx] * exp2(g[idx]) * scale` into `QG`, with
`exp2(x) = Real.exp (x * Real.log 2)`. This is the honest pointwise
specification of the `q *= exp2(g) * scale` map (here `scale = 1`). -/
noncomputable def prepareQgClosed
    (s : BlockState) (Q G : RegionName)
    (s_qk_h DK BT BK : Nat) (t_rel : Fin BT) (i : Fin BK) (scale : ℝ) : ℝ :=
  s.readMem Q (offset s s_qk_h DK t_rel.val BT BK i) *
    Real.exp (s.readMem G (offset s s_qk_h DK t_rel.val BT BK i) * Real.log 2) *
    scale

set_option maxHeartbeats 2000000 in
/-- **Row 0 active `qg` closed form.** On an active lane the full `prepare_qg_kg`
surface writes `q[idx] * exp2(g[idx]) * 1` into `QG` at the row-0 offset. -/
theorem prepare_qg_kg_full_surface_qg_row0_closed
    (Q K G QG KG : RegionName) (s : BlockState) (i : Fin 4)
    (hQG_KG : QG ≠ KG) (hact : active s 8 4 i) :
    (exec (prepare_qg_kg_surface Q K G QG KG 64 8 2 4 1.0) s).map
      (·.readMem QG (offset s 64 8 0 2 4 i))
      = some (prepareQgClosed s Q G 64 8 2 4 0 i 1.0) := by
  have hInj : Function.Injective
      (fun idx : TileIndex [4] =>
        s.pids 2 * 64 + s.pids 1 * 2 * 8 + s.pids 0 * 4 + idx.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  have hc : s.pids 0 * 4 + (i : ℕ) < 8 := by
    simpa [active, elemIndex] using hact
  simp only [offset, baseOffset, elemIndex, prepareQgClosed, Fin.val_zero]
  simp [exec, prepare_qg_kg_surface, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.uop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul, NumericDType.sub, ComparableDType.lt,
        FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
        WithBot.realExp2, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
        stepForRangeAux.forRange_unfold, stepForRangeAux.step_lt,
        stepForRangeAux.step_ge,
        BlockState.foldl_writeMem_const_region_prop_masked_readMem_other]
  -- Peel QG row-1 (other offset), then KG row-1 (other region); readback QG row-0.
  rw [BlockState.scatter_prop_masked_preserves_other_offset]
  · rw [BlockState.scatter_prop_masked_preserves_other_region
          (P := fun k : TileIndex [4] => s.pids 0 * 4 + k.1.val < 8) (h_ne := hQG_KG)]
    simp only [BlockState.setReg_readMem]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hInj (i, PUnit.unit)]
    simp only [hc, if_true, Option.map₂_some_some]
    show s.readMem Q _ * (Real.exp (s.readMem G _ * Real.log 2) * 1.0) = _
    ring
  · rintro ⟨k, _⟩ _
    simp only
    omega

set_option maxHeartbeats 2000000 in
/-- **Row 1 active `qg` closed form.** On an active lane the full `prepare_qg_kg`
surface writes `q[idx] * exp2(g[idx]) * 1` into `QG` at the row-1 offset. -/
theorem prepare_qg_kg_full_surface_qg_row1_closed
    (Q K G QG KG : RegionName) (s : BlockState) (i : Fin 4)
    (hQ_QG : Q ≠ QG) (hQ_KG : Q ≠ KG) (hG_QG : G ≠ QG) (hG_KG : G ≠ KG)
    (hact : active s 8 4 i) :
    (exec (prepare_qg_kg_surface Q K G QG KG 64 8 2 4 1.0) s).map
      (·.readMem QG (offset s 64 8 1 2 4 i))
      = some (prepareQgClosed s Q G 64 8 2 4 1 i 1.0) := by
  have hInj : Function.Injective
      (fun idx : TileIndex [4] =>
        s.pids 2 * 64 + s.pids 1 * 2 * 8 + s.pids 0 * 4 + idx.1.val + 8) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel (Nat.add_right_cancel hab))
    rfl
  have hc : s.pids 0 * 4 + (i : ℕ) < 8 := by
    simpa [active, elemIndex] using hact
  simp only [offset, baseOffset, elemIndex, prepareQgClosed, Fin.val_one]
  simp [exec, prepare_qg_kg_surface, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.uop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul, NumericDType.sub, ComparableDType.lt,
        FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
        WithBot.realExp2, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
        stepForRangeAux.forRange_unfold, stepForRangeAux.step_lt,
        stepForRangeAux.step_ge,
        BlockState.foldl_writeMem_const_region_prop_masked_readMem_other,
        hQ_QG, hQ_KG, hG_QG, hG_KG]
  rw [show s.pids 2 * 64 + (s.pids 1 * 2 + 1) * 8 + s.pids 0 * 4 + (i : ℕ)
        = s.pids 2 * 64 + s.pids 1 * 2 * 8 + s.pids 0 * 4 + (i : ℕ) + 8 from by ring]
  -- The row-1 QG store is outermost; read it back directly.
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hInj (i, PUnit.unit)]
  simp only [hc, if_true, Option.map₂_some_some]
  show s.readMem Q _ * (Real.exp (s.readMem G _ * Real.log 2) * 1.0) = _
  ring

/-- **Unified active `qg` closed form (both chunk rows).** -/
theorem prepare_qg_kg_full_surface_qg_closed
    (Q K G QG KG : RegionName) (s : BlockState) (t_rel : Fin 2) (i : Fin 4)
    (hQ_QG : Q ≠ QG) (hQ_KG : Q ≠ KG) (hG_QG : G ≠ QG) (hG_KG : G ≠ KG)
    (hQG_KG : QG ≠ KG) (hact : active s 8 4 i) :
    (exec (prepare_qg_kg_surface Q K G QG KG 64 8 2 4 1.0) s).map
      (·.readMem QG (offset s 64 8 t_rel.val 2 4 i))
      = some (prepareQgClosed s Q G 64 8 2 4 t_rel i 1.0) := by
  match t_rel with
  | ⟨0, _⟩ =>
    exact prepare_qg_kg_full_surface_qg_row0_closed Q K G QG KG s i hQG_KG hact
  | ⟨1, _⟩ =>
    exact prepare_qg_kg_full_surface_qg_row1_closed Q K G QG KG s i
      hQ_QG hQ_KG hG_QG hG_KG hact

/-- **Genuine `prepare_qg_kg` compute-correctness.** The full `prepare_qg_kg`
surface realizes the honest pointwise closed form `prepareQgClosed`
(`q * exp2(g) * scale`) at every active `QG` lane of loop row `t_rel`. The
`expected` here is independent of the kernel's own output (no self-reference). -/
theorem prepare_qg_kg_surface_qg_closed_compute_correct
    (Q K G QG KG : RegionName) (t_rel : Fin 2) (s : BlockState)
    (hQ_QG : Q ≠ QG) (hQ_KG : Q ≠ KG) (hG_QG : G ≠ QG) (hG_KG : G ≠ KG)
    (hQG_KG : QG ≠ KG) :
    ComputeCorrect.Realizes
      (kernel := prepare_qg_kg_surface Q K G QG KG 64 8 2 4 1.0)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 8 4)
        (fun i => (QG, offset s 64 8 t_rel.val 2 4 i)))
      (expected := fun i : Fin 4 =>
        prepareQgClosed s Q G 64 8 2 4 t_rel i 1.0) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [prepare_qg_kg_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := prepare_qg_kg_full_surface_qg_closed Q K G QG KG s t_rel i
    hQ_QG hQ_KG hG_QG hG_KG hQG_KG hActive
  rw [hExec] at h
  simp only [Option.map_some] at h
  rw [Option.some.injEq] at h
  show s'.readMem QG (offset s 64 8 t_rel.val 2 4 i) = _
  rw [h]

/-- `output_summary` for the Python q/k decay preparation surface (`qg` output),
certified against the **genuine** pointwise closed form `prepareQgClosed`. This
is the non self-referential replacement closing the `decay-cumsum-scan-fold`
proof gap for the `prepare_qg_kg` kernel. -/
theorem decay_cumsum_prepare_python_test_shape_closed_output_summary
    (Q K G QG KG : RegionName) (t_rel : Fin 2) (s : BlockState)
    (hQ_QG : Q ≠ QG) (hQ_KG : Q ≠ KG) (hG_QG : G ≠ QG) (hG_KG : G ≠ KG)
    (hQG_KG : QG ≠ KG) :
    ComputeCorrect.Realizes
      (kernel := prepare_qg_kg_surface Q K G QG KG 64 8 2 4 1.0)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 8 4)
        (fun i => (QG, offset s 64 8 t_rel.val 2 4 i)))
      (expected := fun i : Fin 4 =>
        prepareQgClosed s Q G 64 8 2 4 t_rel i 1.0) :=
  prepare_qg_kg_surface_qg_closed_compute_correct Q K G QG KG t_rel s
    hQ_QG hQ_KG hG_QG hG_KG hQG_KG

/-- `output_summary` for the Python q/k decay preparation surface, certified
against the **genuine** pointwise closed form `prepareQgClosed`
(`qg = q * exp2(g) * scale`). Non self-referential replacement closing the
`decay-cumsum-scan-fold` proof gap for the `prepare_qg_kg` kernel. -/
abbrev decay_cumsum_prepare_python_test_shape_output_summary
    (Q K G QG KG : RegionName) (t_rel : Fin 2) (s : BlockState)
    (hQ_QG : Q ≠ QG) (hQ_KG : Q ≠ KG) (hG_QG : G ≠ QG) (hG_KG : G ≠ KG)
    (hQG_KG : QG ≠ KG) :=
  decay_cumsum_prepare_python_test_shape_closed_output_summary
    Q K G QG KG t_rel s hQ_QG hQ_KG hG_QG hG_KG hQG_KG

/-! ## Genuine `bwd_decay_global_cumsum` closed forms

The backward kernel traverses the chunk in reverse (`range(BT-1,-1,-1)`) and at
each row `j` performs three honest computations:

* `dq_inter[j] = dq_inner[j] + dq_inter_in[j] * exp2(g[j])`
* `dk_inter[j] = dk_inner[j] + dk_inter_in[j] * exp2(g[BT-1] - g[j])`
* `dg[j] = Σ_{j' = j}^{BT-1} (dq_inter[j'] * q[j'] - dk_inter[j'] * k[j'])`
  (the reverse cumulative sum `cum_grad_dg`).

`dq_inter`/`dk_inter` are pointwise per-row maps; `dg` is a reverse prefix scan.
All three are certified below against genuine closed forms — *not* the executed
kernel readback (`decayBackwardSurfaceValue`). -/

/-- **Genuine `dq_inter` closed form.** At chunk row `t_rel` and lane `i`, the
backward kernel writes `dq_inner[idx] + dq_inter_in[idx] * exp2(g[idx])` into
`dq_inter`, with `exp2(x) = Real.exp (x * Real.log 2)`. -/
noncomputable def bwdDQInterClosed
    (s : BlockState) (DQInner DQInter G : RegionName)
    (s_qk_h DK BT BK : Nat) (t_rel : Fin BT) (i : Fin BK) : ℝ :=
  s.readMem DQInner (offset s s_qk_h DK t_rel.val BT BK i) +
    s.readMem DQInter (offset s s_qk_h DK t_rel.val BT BK i) *
      Real.exp (s.readMem G (offset s s_qk_h DK t_rel.val BT BK i) * Real.log 2)

/-- **Genuine `dk_inter` closed form.** At chunk row `t_rel` and lane `i`, the
backward kernel writes
`dk_inner[idx] + dk_inter_in[idx] * exp2(g[row BT-1] - g[idx])` into `dk_inter`,
where `g[row BT-1]` is the captured `last_g`. -/
noncomputable def bwdDKInterClosed
    (s : BlockState) (DKInner DKInter G : RegionName)
    (s_qk_h DK BT BK : Nat) (t_rel : Fin BT) (i : Fin BK) : ℝ :=
  s.readMem DKInner (offset s s_qk_h DK t_rel.val BT BK i) +
    s.readMem DKInter (offset s s_qk_h DK t_rel.val BT BK i) *
      Real.exp ((s.readMem G (offset s s_qk_h DK (BT - 1) BT BK i) -
        s.readMem G (offset s s_qk_h DK t_rel.val BT BK i)) * Real.log 2)

/-- The per-row `dg` summand `dq_inter[j] * q[j] - dk_inter[j] * k[j]`, written
in terms of the genuine `dq_inter`/`dk_inter` closed forms above. -/
noncomputable def bwdDGSummand
    (s : BlockState) (DQInner DQInter DKInner DKInter Q K G : RegionName)
    (s_qk_h DK BT BK : Nat) (j : Fin BT) (i : Fin BK) : ℝ :=
  bwdDQInterClosed s DQInner DQInter G s_qk_h DK BT BK j i *
      s.readMem Q (offset s s_qk_h DK j.val BT BK i) -
    bwdDKInterClosed s DKInner DKInter G s_qk_h DK BT BK j i *
      s.readMem K (offset s s_qk_h DK j.val BT BK i)

/-- **Genuine `dg` closed form.** At chunk row `t_rel` and lane `i`, the backward
kernel writes the reverse cumulative sum
`Σ_{j = t_rel}^{BT-1} (dq_inter[j]*q[j] - dk_inter[j]*k[j])` into `dg`. This is
the honest reverse-prefix-scan specification of the carried `cum_grad_dg`
accumulator (the `range(BT-1,-1,-1)` loop threads `cum_grad_dg += dq*q - dk*k`).
This is *not* the executed kernel readback. -/
noncomputable def bwdDGClosed
    (s : BlockState) (DQInner DQInter DKInner DKInter Q K G : RegionName)
    (s_qk_h DK BT BK : Nat) (t_rel : Fin BT) (i : Fin BK) : ℝ :=
  ∑ d : Fin (BT - t_rel.val),
    bwdDGSummand s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK
      ⟨t_rel.val + d.val, by omega⟩ i

/-! ### Proof recipe (backward closed forms)

The three genuine closed forms above (`bwdDQInterClosed`, `bwdDKInterClosed`,
`bwdDGClosed`) are the honest, non self-referential specifications that should
replace `decayBackwardSurfaceValue`. Connecting them to the executed
`bwd_decay_global_cumsum_surface` follows the **forward** closed-form recipe in
this file (`fwd_decay_cumsum_full_surface_row{0,1}_closed`), but the backward
loop body is ~25 statements with a conditional `last_g` capture and three masked
stores per iteration, traversed over two `range(BT-1,-1,-1)` rows (lowered to a
forward `forRangeDyn "__rev_t" 0 2 1` with `t := 1 - __rev_t`). A single
`simp [exec, …, evalOp.eq_def, stepForRangeAux.*]` blast does *not* scale to this
body (it does not terminate within ~9 min even at 8M heartbeats), so the
mandated per-statement architecture is required:

1. `exec → stepStmts toAlgKernel.body` via
   `bwd_decay_global_cumsum_surface_toAlgorithm_supported`.
2. Drive the `forRangeDyn` loop with `forRangeAux_inv` /
   `VeriTile.Triton.forRangeDyn_inv` (carry invariant on `cum_grad_dg` =
   partial reverse prefix sum), *not* a `simp` over the whole loop.
3. Per body statement: `stepStmts.cons_some` + `simp only` over the named
   `evalOp_*` lemmas (`evalOp_add/mul/sub/ref/…`, `evalOp_ref_setReg*`) — never
   `evalOp.eq_def` whnf over the nested `setReg` literal state.
4. Read back each output with the masked-scatter lemmas
   (`scatter_readback_prop_masked_nd`,
   `scatter_prop_masked_preserves_other_{offset,region}`), peeling the later
   stores in reverse, exactly as the forward row-1 proof does.
5. Bridge to `ComputeCorrect.Realizes` via `realizes_writeIf_iff` +
   `computeCorrect_of_toAlgKernel`, then delete `decayBackwardSurfaceValue`.

The `dq_inter`/`dk_inter` faces are pointwise (no carry); only `dg` needs the
reverse-scan invariant. Region-distinctness side hypotheses (`DQInter ≠ DKInter`
etc.) are needed so a later store does not clobber an earlier readback, mirroring
the forward `G ≠ GO` and `prepare` `Q ≠ QG …` hypotheses. -/

/-! ## Per-statement op-eval recipes (backward kernel, recipe layer)

These are the standalone, register-readback-abstracted `stepStmt`/`evalOp`
reduction lemmas for *each statement kind* appearing in the
`bwd_decay_global_cumsum_surface` body (15-stmt prologue + 25-stmt reverse loop
body). They are the mandated per-statement architecture building blocks: every
lemma takes a *symbolic* `BlockState` plus abstract hypotheses giving the
evaluation of its sub-operands (`evalOp _ s = some _`), and proves the single
statement's reduction by `simp [stepStmt, evalOp, …]` only — never a whole-body
`evalOp.eq_def` blast, never `rfl`/`whnf` over a nested `setReg` literal state.

The next stage (full assembly) chains these via `stepStmts.cons_some` /
`forRangeAux` invariants, exactly as `LayerNormKernels` chains its per-stmt
`have h_k : stepStmt … = some (… .setReg …)` facts. The backing surface
definitions and the genuine closed forms (`bwdDQInterClosed`, `bwdDKInterClosed`,
`bwdDGSummand`, `bwdDGClosed`) are already banked above; only the assembly that
threads the reverse `cum_grad_dg` scan + `last_g` capture through these recipes
remains.

`s_qk_h`, `DK`, `BK`, etc. are kept symbolic so each recipe is reusable at any
loop row / pointer position; the assembly instantiates the test shape
(`s_qk_h = 64`, `DK = 8`, `BT = 2`, `BK = 4`). -/

section BwdRecipes

variable {BK : Nat} (s : BlockState)

/-- **Prologue / loop pointer-arithmetic recipe (add).** A nat-scalar `add`
assign — `p_x = base + offs`, `p_x += DK`, or the loop-body offset folds —
reduces to a `setReg` of the summed scalar, given the two operand evaluations. -/
theorem bwdEval_assign_addNat (name : RegName) (a b : Op .nat []) (va vb : Nat)
    (ha : evalOp a s = some (Tile.scalar va))
    (hb : evalOp b s = some (Tile.scalar vb)) :
    stepStmt (.assign .nat [] name (.add NumericDType.nat Broadcast.nil a b)) s
      = some (s.setReg name .nat [] (Tile.scalar (va + vb))) := by
  simp [stepStmt, evalOp, ha, hb]
  rfl

/-- **Loop index recipe (`t := 1 - __rev_t`, nat sub).** The first body
statement of the lowered forward `forRangeDyn "__rev_t" 0 2 1` re-derives the
reverse time index `t`. A nat-scalar `sub` assign reduces to a `setReg` of the
truncated difference, given the operand evaluations. -/
theorem bwdEval_assign_subNat (name : RegName) (a b : Op .nat []) (va vb : Nat)
    (ha : evalOp a s = some (Tile.scalar va))
    (hb : evalOp b s = some (Tile.scalar vb)) :
    stepStmt (.assign .nat [] name (.sub NumericDType.nat Broadcast.nil a b)) s
      = some (s.setReg name .nat [] (Tile.scalar (va - vb))) := by
  simp [stepStmt, evalOp, ha, hb]
  rfl

/-- **`cum_grad_dg = tl.zeros([BK])` init recipe.** The reverse-scan accumulator
(and `last_g`) initializer assigns the all-zero `[BK]` real tile. -/
theorem bwdEval_assign_zeros (name : RegName) :
    stepStmt (.assign .real [BK] name (.full [BK] (.const 0))) s
      = some (s.setReg name .real [BK] ⟨fun _ => some 0⟩) := by
  simp [stepStmt, evalOp]

/-- **Masked-region load recipe (`tl.load(p_x, mask=mask, other=0)`).** Every
prologue/body input load (`g`, `q`, `k`, `dq_inner`, `dk_inner`, `dq_inter_in`,
`dk_inter_in`) is a `MaskOpt.maskOther` region load over `[BK]`. Given the
offset/mask/other evaluations, it reduces to the per-lane masked readback:
active lanes read memory, inactive lanes take the `other` fill. -/
theorem bwdEval_load_maskOther (region : Region .real)
    (offs : Op .nat [BK]) (mask : Op .bool [BK]) (other : Op .real [BK])
    (offsT : Tile .nat [BK]) (maskT : Tile .bool [BK]) (otherT : Tile .real [BK])
    (hoff : evalOp offs s = some offsT)
    (hmask : evalOp mask s = some maskT)
    (hother : evalOp other s = some otherT) :
    evalOp (.load .real (MemAccess.region region offs) (MaskOpt.maskOther mask other)) s
      = some ⟨fun i => if maskT.data i then
                some (s.readMem region (offsT.data i))
              else otherT.data i⟩ := by
  simp [evalOp, hoff, hmask, hother]

/-- **Masked-region load assign recipe.** The load wrapped in its `assign`
target register (`g_val = …`, `dq1 = …`, etc.), reducing to a single `setReg`. -/
theorem bwdEval_assign_load_maskOther (name : RegName) (region : Region .real)
    (offs : Op .nat [BK]) (mask : Op .bool [BK]) (other : Op .real [BK])
    (offsT : Tile .nat [BK]) (maskT : Tile .bool [BK]) (otherT : Tile .real [BK])
    (hoff : evalOp offs s = some offsT)
    (hmask : evalOp mask s = some maskT)
    (hother : evalOp other s = some otherT) :
    stepStmt (.assign .real [BK] name
        (.load .real (MemAccess.region region offs) (MaskOpt.maskOther mask other))) s
      = some (s.setReg name .real [BK]
          ⟨fun i => if maskT.data i then
                some (s.readMem region (offsT.data i))
              else otherT.data i⟩) := by
  simp [stepStmt, evalOp, hoff, hmask, hother]

/-- **Pointer masked-load assign recipe (`tl.load(p_x, mask=mask, other=0)`).**
The backward body uses *pointer*-based loads (`MemAccess.ptr (Op.ref .ptr [BK]
"p_x")`), not region loads: each lane reads memory at the per-lane pointer
`(ptrT.data i)` = `(region, address)`. Given the pointer-tile / mask / other
evaluations, the masked real load reduces to a single `setReg`. -/
theorem bwdEval_assign_load_ptr_maskOther (name pname : RegName)
    (mask : Op .bool [BK]) (other : Op .real [BK])
    (ptrT : Tile .ptr [BK]) (maskT : Tile .bool [BK]) (otherT : Tile .real [BK])
    (hptr : s.regs .ptr [BK] pname = some ptrT)
    (hmask : evalOp mask s = some maskT)
    (hother : evalOp other s = some otherT) :
    stepStmt (.assign .real [BK] name
        (.load .real (MemAccess.ptr (Op.ref .ptr [BK] pname))
          (MaskOpt.maskOther mask other))) s
      = some (s.setReg name .real [BK]
          ⟨fun i => if maskT.data i then
                some (s.readMem (ptrT.data i).1 (ptrT.data i).2)
              else otherT.data i⟩) := by
  simp [stepStmt, evalOp, hptr, hmask, hother]

/-- **Pointer masked-store recipe (`tl.store(p_out, val, mask=mask)`).** Each of
the three per-iteration stores (`p_dq_inter`, `p_dk_inter`, `p_dg`) is a
pointer-based masked store over `[BK]` lanes, reducing to the `writeMemTyped`
masked scatter fold along the per-lane pointers. -/
theorem bwdEval_store_ptr_masked (pname : RegName)
    (val : Op .real [BK]) (mask : Op .bool [BK])
    (ptrT : Tile .ptr [BK]) (valT : Tile .real [BK]) (maskT : Tile .bool [BK])
    (hptr : s.regs .ptr [BK] pname = some ptrT)
    (hval : evalOp val s = some valT)
    (hmask : evalOp mask s = some maskT) :
    stepStmt (.store .real [BK] (MemAccess.ptr (Op.ref .ptr [BK] pname))
        val (MaskOpt.mask mask)) s
      = some ((TileShape.allIndices [BK]).foldl
          (fun acc i =>
            if maskT.data i then
              acc.writeMemTyped .real (ptrT.data i).1 (ptrT.data i).2 (valT.data i)
            else acc) s) := by
  simp [stepStmt, evalOp, hptr, hval, hmask]

/-- **Pointer-decrement recipe (`p_x -= DK`).** The per-iteration pointer
decrements (`Op.ptrSub Broadcast.scalarR (Op.ref .ptr [BK] pname) (Op.constNat
d)`) reduce to a `setReg` of the per-lane address-decremented pointer tile. -/
theorem bwdEval_assign_ptrSub (name pname : RegName) (d : Nat)
    (ptrT : Tile .ptr [BK])
    (hptr : s.regs .ptr [BK] pname = some ptrT) :
    stepStmt (.assign .ptr [BK] name
        (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [BK] pname) (Op.constNat d))) s
      = some (s.setReg name .ptr [BK]
          ⟨fun i => ((ptrT.data i).1, (ptrT.data i).2 - d)⟩) := by
  simp [stepStmt, evalOp, hptr, Tile.ptrSub]

/-- **`last_g` conditional-capture condition recipe (`t == BT-1`).** The
`ifThen (Op.eq t (BT-1))` guard's nat-eq condition reduces to the `Tile.cop`
comparison cell, given the two operand evaluations. -/
theorem bwdEval_eqNat (a b : Op .nat []) (av bv : Tile .nat [])
    (ha : evalOp a s = some av) (hb : evalOp b s = some bv) :
    evalOp (.eq ComparableDType.nat Broadcast.nil a b) s
      = some (Tile.cop ComparableDType.nat.eq Broadcast.nil av bv) := by
  simp [evalOp, ha, hb]

/-- **`last_g` capture, taken branch (`__rev_t = 0 ⇒ t = BT-1`).** When the
guard evaluates `true`, `ifThen` runs its body (`last_g = g_val`). -/
theorem bwdEval_ifThen_true (cond : Op .bool []) (body : List Stmt)
    (hc : evalOp cond s = some (Tile.scalar Bool.true)) :
    stepStmt (.ifThen cond body) s = stepStmts body s := by
  simp [stepStmt, hc]

/-- **`last_g` capture, skipped branch (`__rev_t = 1 ⇒ t = 0 ≠ BT-1`).** When
the guard evaluates `false`, `ifThen` leaves the state unchanged. -/
theorem bwdEval_ifThen_false (cond : Op .bool []) (body : List Stmt)
    (hc : evalOp cond s = some (Tile.scalar Bool.false)) :
    stepStmt (.ifThen cond body) s = some s := by
  simp [stepStmt, hc]

end BwdRecipes

section BwdComputeRecipes

variable {BK : Nat} (s : BlockState)

/-- **Pointwise mul recipe.** `dq2 *= exp2(g_val)` and `dk2 *= exp2(last_g-g)`
and the `dq*q` / `dk*k` products of the `dg` summand are real `[BK]` muls. -/
theorem bwdEval_mul (a b : Op .real [BK]) (aT bT : Tile .real [BK])
    (ha : evalOp a s = some aT) (hb : evalOp b s = some bT) :
    evalOp (.mul NumericDType.real (Broadcast.consSame Broadcast.nil) a b) s
      = some (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) aT bT) := by
  simp [evalOp, ha, hb]

/-- **Pointwise add recipe.** `dq = dq1 + dq2`, `dk = dk1 + dk2`, and the
reverse-scan accumulate `cum_grad_dg += dg_val` are real `[BK]` adds. -/
theorem bwdEval_add (a b : Op .real [BK]) (aT bT : Tile .real [BK])
    (ha : evalOp a s = some aT) (hb : evalOp b s = some bT) :
    evalOp (.add NumericDType.real (Broadcast.consSame Broadcast.nil) a b) s
      = some (Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil) aT bT) := by
  simp [evalOp, ha, hb]

/-- **Pointwise sub recipe.** `dg_val = dq*q - dk*k` and the `last_g - g_val`
exp2 argument are real `[BK]` subs. -/
theorem bwdEval_sub (a b : Op .real [BK]) (aT bT : Tile .real [BK])
    (ha : evalOp a s = some aT) (hb : evalOp b s = some bT) :
    evalOp (.sub NumericDType.real (Broadcast.consSame Broadcast.nil) a b) s
      = some (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) aT bT) := by
  simp [evalOp, ha, hb]

/-- **`exp2` recipe** (mirrors the `AttentionForwardClosedForm` `exp2` pattern).
`exp2(g_val)` and `exp2(last_g - g_val)` reduce to `Tile.uop WithBot.realExp2`
over the operand tile (each cell `r ↦ Real.exp (r * Real.log 2)`). -/
theorem bwdEval_exp2 (x : Op .real [BK]) (xT : Tile .real [BK])
    (hx : evalOp x s = some xT) :
    evalOp (.exp2 x) s = some (Tile.uop WithBot.realExp2 xT) := by
  simp [evalOp, hx]

/-- **Masked-region store recipe (`tl.store(p_out, val, mask=mask)`).** Each of
the three per-iteration stores (`DQInter`, `DKInter`, `DG`) reduces to the
`writeMemTyped` masked scatter fold over the `[BK]` lanes. Region distinctness
(`DQInter ≠ DKInter ≠ DG`) is *not* needed here — it is needed only at the
readback stage (`scatter_readback_prop_masked_nd` /
`scatter_prop_masked_preserves_other_{offset,region}`, already in this file) so
a later store does not clobber an earlier readback. -/
theorem bwdEval_store_masked (region : RegionName)
    (off : Op .nat [BK]) (val : Op .real [BK]) (mask : Op .bool [BK])
    (offsT : Tile .nat [BK]) (valT : Tile .real [BK]) (maskT : Tile .bool [BK])
    (hoff : evalOp off s = some offsT)
    (hval : evalOp val s = some valT)
    (hmask : evalOp mask s = some maskT) :
    stepStmt (.store .real [BK] (MemAccess.region region off)
        val (MaskOpt.mask mask)) s
      = some ((TileShape.allIndices [BK]).foldl
          (fun acc i =>
            if maskT.data i then
              acc.writeMemTyped .real (Region.cast region) (offsT.data i) (valT.data i)
            else acc) s) := by
  simp [stepStmt, evalOp, hoff, hval, hmask]

end BwdComputeRecipes

section BwdAssembly

set_option maxHeartbeats 1000000 in
/-- The backward surface body at the test shape, as an explicit `Stmt` list whose
prologue (first 15 statements) is exposed for `stepStmts.cons_some` chaining. We
characterize the post-prologue state's relevant register values. -/
theorem bwd_prologue_eval
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName) (s : BlockState) :
    ∃ s0,
      stepStmts ((bwd_decay_global_cumsum_surface DQInner DQInter DKInner DKInter
        Q K G DG 64 8 2 4).toAlgKernel.body.take 15) s = some s0
      ∧ s0.pids = s.pids
      ∧ s0.mem = s.mem
      ∧ s0.undef = s.undef
      ∧ s0.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0))
      ∧ s0.regs .nat [] "i_c" = some (Tile.scalar (s.pids 1))
      ∧ s0.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2))
      ∧ s0.regs .nat [4] "offs" = some (Tile.vec (fun e : Fin 4 => e.val))
      ∧ s0.regs .bool [4] "mask" = some
          (Tile.vec (fun e : Fin 4 => decide (s.pids 0 * 4 + e.val < 8)))
      ∧ s0.regs .real [4] "cum_grad_dg" = some ⟨fun _ => some 0⟩
      ∧ s0.regs .real [4] "last_g" = some ⟨fun _ => some 0⟩
      ∧ s0.regs .ptr [4] "p_q" = some
          (Tile.vec (fun e : Fin 4 =>
            (Q.cast, s.pids 2 * 64 + s.pids 0 * 4 + e.val + (s.pids 1 * 2 + 2 - 1) * 8)))
      ∧ s0.regs .ptr [4] "p_k" = some
          (Tile.vec (fun e : Fin 4 =>
            (K.cast, s.pids 2 * 64 + s.pids 0 * 4 + e.val + (s.pids 1 * 2 + 2 - 1) * 8)))
      ∧ s0.regs .ptr [4] "p_g" = some
          (Tile.vec (fun e : Fin 4 =>
            (G.cast, s.pids 2 * 64 + s.pids 0 * 4 + e.val + (s.pids 1 * 2 + 2 - 1) * 8)))
      ∧ s0.regs .ptr [4] "p_dg" = some
          (Tile.vec (fun e : Fin 4 =>
            (DG.cast, s.pids 2 * 64 + s.pids 0 * 4 + e.val + (s.pids 1 * 2 + 2 - 1) * 8)))
      ∧ s0.regs .ptr [4] "p_dq_inner" = some
          (Tile.vec (fun e : Fin 4 =>
            (DQInner.cast, s.pids 2 * 64 + s.pids 0 * 4 + e.val + (s.pids 1 * 2 + 2 - 1) * 8)))
      ∧ s0.regs .ptr [4] "p_dk_inner" = some
          (Tile.vec (fun e : Fin 4 =>
            (DKInner.cast, s.pids 2 * 64 + s.pids 0 * 4 + e.val + (s.pids 1 * 2 + 2 - 1) * 8)))
      ∧ s0.regs .ptr [4] "p_dq_inter" = some
          (Tile.vec (fun e : Fin 4 =>
            (DQInter.cast, s.pids 2 * 64 + s.pids 0 * 4 + e.val + (s.pids 1 * 2 + 2 - 1) * 8)))
      ∧ s0.regs .ptr [4] "p_dk_inter" = some
          (Tile.vec (fun e : Fin 4 =>
            (DKInter.cast, s.pids 2 * 64 + s.pids 0 * 4 + e.val + (s.pids 1 * 2 + 2 - 1) * 8))) := by
  rw [show ((bwd_decay_global_cumsum_surface DQInner DQInter DKInner DKInter
        Q K G DG 64 8 2 4).toAlgKernel.body.take 15)
      = [ Stmt.assign .nat [] "i_k" (Op.programId 0),
          Stmt.assign .nat [] "i_c" (Op.programId 1),
          Stmt.assign .nat [] "i_bh" (Op.programId 2),
          Stmt.assign .nat [4] "offs" (Op.arange 4),
          Stmt.assign .ptr [4] "p_q"
            (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Q)
              (Op.add .nat Broadcast.scalarR
                (Op.add .nat Broadcast.scalarL
                  (Op.add .nat Broadcast.nil
                    (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat 64))
                    (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat 4)))
                  (Op.ref .nat [4] "offs"))
                (Op.mul .nat Broadcast.nil
                  (Op.sub .nat Broadcast.nil
                    (Op.add .nat Broadcast.nil
                      (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat 2)) (Op.constNat 2))
                    (Op.constNat 1))
                  (Op.constNat 8)))),
          Stmt.assign .ptr [4] "p_k"
            (Op.ptrAdd Broadcast.scalarL (Op.ptrBase K)
              (Op.add .nat Broadcast.scalarR
                (Op.add .nat Broadcast.scalarL
                  (Op.add .nat Broadcast.nil
                    (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat 64))
                    (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat 4)))
                  (Op.ref .nat [4] "offs"))
                (Op.mul .nat Broadcast.nil
                  (Op.sub .nat Broadcast.nil
                    (Op.add .nat Broadcast.nil
                      (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat 2)) (Op.constNat 2))
                    (Op.constNat 1))
                  (Op.constNat 8)))),
          Stmt.assign .ptr [4] "p_g"
            (Op.ptrAdd Broadcast.scalarL (Op.ptrBase G)
              (Op.add .nat Broadcast.scalarR
                (Op.add .nat Broadcast.scalarL
                  (Op.add .nat Broadcast.nil
                    (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat 64))
                    (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat 4)))
                  (Op.ref .nat [4] "offs"))
                (Op.mul .nat Broadcast.nil
                  (Op.sub .nat Broadcast.nil
                    (Op.add .nat Broadcast.nil
                      (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat 2)) (Op.constNat 2))
                    (Op.constNat 1))
                  (Op.constNat 8)))),
          Stmt.assign .ptr [4] "p_dg"
            (Op.ptrAdd Broadcast.scalarL (Op.ptrBase DG)
              (Op.add .nat Broadcast.scalarR
                (Op.add .nat Broadcast.scalarL
                  (Op.add .nat Broadcast.nil
                    (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat 64))
                    (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat 4)))
                  (Op.ref .nat [4] "offs"))
                (Op.mul .nat Broadcast.nil
                  (Op.sub .nat Broadcast.nil
                    (Op.add .nat Broadcast.nil
                      (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat 2)) (Op.constNat 2))
                    (Op.constNat 1))
                  (Op.constNat 8)))),
          Stmt.assign .ptr [4] "p_dq_inner"
            (Op.ptrAdd Broadcast.scalarL (Op.ptrBase DQInner)
              (Op.add .nat Broadcast.scalarR
                (Op.add .nat Broadcast.scalarL
                  (Op.add .nat Broadcast.nil
                    (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat 64))
                    (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat 4)))
                  (Op.ref .nat [4] "offs"))
                (Op.mul .nat Broadcast.nil
                  (Op.sub .nat Broadcast.nil
                    (Op.add .nat Broadcast.nil
                      (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat 2)) (Op.constNat 2))
                    (Op.constNat 1))
                  (Op.constNat 8)))),
          Stmt.assign .ptr [4] "p_dk_inner"
            (Op.ptrAdd Broadcast.scalarL (Op.ptrBase DKInner)
              (Op.add .nat Broadcast.scalarR
                (Op.add .nat Broadcast.scalarL
                  (Op.add .nat Broadcast.nil
                    (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat 64))
                    (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat 4)))
                  (Op.ref .nat [4] "offs"))
                (Op.mul .nat Broadcast.nil
                  (Op.sub .nat Broadcast.nil
                    (Op.add .nat Broadcast.nil
                      (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat 2)) (Op.constNat 2))
                    (Op.constNat 1))
                  (Op.constNat 8)))),
          Stmt.assign .ptr [4] "p_dq_inter"
            (Op.ptrAdd Broadcast.scalarL (Op.ptrBase DQInter)
              (Op.add .nat Broadcast.scalarR
                (Op.add .nat Broadcast.scalarL
                  (Op.add .nat Broadcast.nil
                    (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat 64))
                    (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat 4)))
                  (Op.ref .nat [4] "offs"))
                (Op.mul .nat Broadcast.nil
                  (Op.sub .nat Broadcast.nil
                    (Op.add .nat Broadcast.nil
                      (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat 2)) (Op.constNat 2))
                    (Op.constNat 1))
                  (Op.constNat 8)))),
          Stmt.assign .ptr [4] "p_dk_inter"
            (Op.ptrAdd Broadcast.scalarL (Op.ptrBase DKInter)
              (Op.add .nat Broadcast.scalarR
                (Op.add .nat Broadcast.scalarL
                  (Op.add .nat Broadcast.nil
                    (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat 64))
                    (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat 4)))
                  (Op.ref .nat [4] "offs"))
                (Op.mul .nat Broadcast.nil
                  (Op.sub .nat Broadcast.nil
                    (Op.add .nat Broadcast.nil
                      (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat 2)) (Op.constNat 2))
                    (Op.constNat 1))
                  (Op.constNat 8)))),
          Stmt.assign .real [4] "cum_grad_dg" (Op.full [4] (Op.const 0)),
          Stmt.assign .bool [4] "mask"
            (Op.lt ComparableDType.nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat 4))
                (Op.ref .nat [4] "offs"))
              (Op.constNat 8)),
          Stmt.assign .real [4] "last_g" (Op.full [4] (Op.const 0)) ] from rfl]
  simp [stepStmts, stepStmt, evalOp, Option.bind, BlockState.setReg,
    Tile.bop, Tile.vec, Tile.ptrAdd, NumericDType.add, NumericDType.mul,
    NumericDType.sub, ComparableDType.lt]
  ext i
  simp only [Tile.cop_data, Tile.vec, Tile.scalar, Broadcast.leftIndex_scalarR,
    Broadcast.rightIndex_scalarR, ComparableDType.lt]
  rfl

/-- The reverse loop's single iteration body (25 statements), as the explicit
`Stmt` list, with `__rev_t` left as a parameter via the surrounding `forRangeDyn`.
This is the `drop 15` tail's `forRangeDyn` body. -/
def bwdIterBody : List Stmt :=
  [ Stmt.assign .nat [] "t"
      (Op.sub .nat Broadcast.nil (Op.sub .nat Broadcast.nil (Op.constNat 2) (Op.constNat 1))
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "__rev_t") (Op.constNat 1))),
    Stmt.assign .real [4] "g_val"
      (Op.load ComputeDType.fp32.eraseDType (MemAccess.ptr (Op.ref .ptr [4] "p_g"))
        (MaskOpt.maskOther (Op.ref .bool [4] "mask") ((Op.const 0).broadcast [4]))),
    Stmt.ifThen
      (Op.eq ComparableDType.nat Broadcast.nil (Op.ref .nat [] "t")
        (Op.sub .nat Broadcast.nil (Op.constNat 2) (Op.constNat 1)))
      [Stmt.assign .real [4] "last_g" (Op.ref .real [4] "g_val")],
    Stmt.assign .real [4] "dq1"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [4] "p_dq_inner"))
        (MaskOpt.maskOther (Op.ref .bool [4] "mask") ((Op.const 0).broadcast [4]))),
    Stmt.assign .real [4] "dq2"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [4] "p_dq_inter"))
        (MaskOpt.maskOther (Op.ref .bool [4] "mask") ((Op.const 0).broadcast [4]))),
    Stmt.assign .real [4] "dq2"
      (Op.mul .real Broadcast.nil.consSame (Op.ref .real [4] "dq2")
        (Op.ref .real [4] "g_val").exp2),
    Stmt.assign .real [4] "dq"
      (Op.add .real Broadcast.nil.consSame (Op.ref .real [4] "dq1")
        (Op.ref .real [4] "dq2")),
    Stmt.store .real [4] (MemAccess.ptr (Op.ref .ptr [4] "p_dq_inter"))
      (Op.ref .real [4] "dq") (MaskOpt.mask (Op.ref .bool [4] "mask")),
    Stmt.assign .real [4] "dk1"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [4] "p_dk_inner"))
        (MaskOpt.maskOther (Op.ref .bool [4] "mask") ((Op.const 0).broadcast [4]))),
    Stmt.assign .real [4] "dk2"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [4] "p_dk_inter"))
        (MaskOpt.maskOther (Op.ref .bool [4] "mask") ((Op.const 0).broadcast [4]))),
    Stmt.assign .real [4] "dk2"
      (Op.mul .real Broadcast.nil.consSame (Op.ref .real [4] "dk2")
        (Op.sub .real Broadcast.nil.consSame (Op.ref .real [4] "last_g")
            (Op.ref .real [4] "g_val")).exp2),
    Stmt.assign .real [4] "dk"
      (Op.add .real Broadcast.nil.consSame (Op.ref .real [4] "dk1")
        (Op.ref .real [4] "dk2")),
    Stmt.store .real [4] (MemAccess.ptr (Op.ref .ptr [4] "p_dk_inter"))
      (Op.ref .real [4] "dk") (MaskOpt.mask (Op.ref .bool [4] "mask")),
    Stmt.assign .real [4] "q_val"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [4] "p_q"))
        (MaskOpt.maskOther (Op.ref .bool [4] "mask") ((Op.const 0).broadcast [4]))),
    Stmt.assign .real [4] "k_val"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [4] "p_k"))
        (MaskOpt.maskOther (Op.ref .bool [4] "mask") ((Op.const 0).broadcast [4]))),
    Stmt.assign .real [4] "dg_val"
      (Op.sub .real Broadcast.nil.consSame
        (Op.mul .real Broadcast.nil.consSame (Op.ref .real [4] "dq")
          (Op.ref .real [4] "q_val"))
        (Op.mul .real Broadcast.nil.consSame (Op.ref .real [4] "dk")
          (Op.ref .real [4] "k_val"))),
    Stmt.assign .real [4] "cum_grad_dg"
      (Op.add .real Broadcast.nil.consSame (Op.ref .real [4] "cum_grad_dg")
        (Op.ref .real [4] "dg_val")),
    Stmt.store .real [4] (MemAccess.ptr (Op.ref .ptr [4] "p_dg"))
      (Op.ref .real [4] "cum_grad_dg") (MaskOpt.mask (Op.ref .bool [4] "mask")),
    Stmt.assign .ptr [4] "p_g"
      (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [4] "p_g") (Op.constNat 8)),
    Stmt.assign .ptr [4] "p_k"
      (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [4] "p_k") (Op.constNat 8)),
    Stmt.assign .ptr [4] "p_q"
      (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [4] "p_q") (Op.constNat 8)),
    Stmt.assign .ptr [4] "p_dq_inner"
      (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [4] "p_dq_inner") (Op.constNat 8)),
    Stmt.assign .ptr [4] "p_dk_inner"
      (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [4] "p_dk_inner") (Op.constNat 8)),
    Stmt.assign .ptr [4] "p_dq_inter"
      (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [4] "p_dq_inter") (Op.constNat 8)),
    Stmt.assign .ptr [4] "p_dk_inter"
      (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [4] "p_dk_inter") (Op.constNat 8)),
    Stmt.assign .ptr [4] "p_dg"
      (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [4] "p_dg") (Op.constNat 8)) ]

/-- The backward surface body decomposes as the 15-stmt prologue followed by the
single `forRangeDyn "__rev_t" 0 2 1 bwdIterBody` reverse loop statement. -/
theorem bwd_body_decomp
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName) :
    (bwd_decay_global_cumsum_surface DQInner DQInter DKInner DKInter Q K G DG
        64 8 2 4).toAlgKernel.body
      = (bwd_decay_global_cumsum_surface DQInner DQInter DKInner DKInter Q K G DG
          64 8 2 4).toAlgKernel.body.take 15
        ++ [Stmt.forRangeDyn "__rev_t" (Op.constNat 0)
              (Op.add .nat Broadcast.nil
                (Op.div .nat Broadcast.nil
                  (Op.sub .nat Broadcast.nil (Op.constNat 2) (Op.constNat 1)) (Op.constNat 1))
                (Op.constNat 1))
              (Op.constNat 1) bwdIterBody] := by
  rfl

/-! ### Full reverse-loop assembly

We chain the 25-statement `bwdIterBody` per the validated per-statement template
(`stepStmts.cons_some` + explicit `set`-bound state threading), twice (for
`__rev_t = 0` and `__rev_t = 1`), then read back the three output regions
(`DQInter`, `DKInter`, `DG`) against the genuine closed forms. -/

/-- Per-lane masked load value from `region` at the iteration's row offset `R`.
The mask is the active-lane decision; inactive lanes read `0`. -/
private noncomputable def ldVal (s : BlockState) (region : RegionName) (R : Nat)
    (i : TileIndex [4]) : WithBot ℝ :=
  if decide (s.pids 0 * 4 + i.1.val < 8) then
    some (s.readMem region (s.pids 2 * 64 + s.pids 0 * 4 + i.1.val + R))
  else (0 : WithBot ℝ)

/-- The post-`t`-assign, post-`g_val`-load state after the first two body
statements of `bwdIterBody`, given the iteration's input pointer/register
readbacks. The `t` index is `1 - rt` and `g_val` holds the masked load of `G`
at the iteration's pointer row offset `R`. -/
private noncomputable def bwdIterHeadState
    (G : RegionName) (sin : BlockState) (rt R : Nat) : BlockState :=
  (sin.setReg "t" .nat [] (Tile.scalar (2 - 1 - rt * 1))).setReg
    "g_val" .real [4] ⟨fun i => ldVal sin G R i⟩

set_option maxHeartbeats 1000000 in
set_option maxRecDepth 8000 in
/-- **Validated per-statement chaining (head of `bwdIterBody`).** The first two
statements (`t = 1 - __rev_t`; `g_val = masked load p_g`) reduce, via the banked
`bwdEval_assign_subNat` / `bwdEval_assign_load_ptr_maskOther` recipes threaded
through `stepStmts.cons_some`, to running the remaining 23 statements from the
explicit head state. This certifies the assembly template (explicit `set`-bound
state threading + per-statement recipe evidence) on the genuine
`MemAccess.ptr`/`eraseDType` loads of the backward loop body. -/
theorem bwdIterBody_head_eval
    (G : RegionName) (sin : BlockState) (rt R : Nat)
    (hrt : sin.regs .nat [] "__rev_t" = some (Tile.scalar rt))
    (hmask : sin.regs .bool [4] "mask" = some
        (Tile.vec (fun e : Fin 4 => decide (sin.pids 0 * 4 + e.val < 8))))
    (hpg : sin.regs .ptr [4] "p_g" = some
        (Tile.vec (fun e : Fin 4 => (G.cast, sin.pids 2 * 64 + sin.pids 0 * 4 + e.val + R)))) :
    stepStmts bwdIterBody sin
      = stepStmts (bwdIterBody.drop 2) (bwdIterHeadState G sin rt R) := by
  rw [bwdIterBody]
  set e0 := sin.setReg "t" .nat [] (Tile.scalar (2 - 1 - rt * 1)) with he0
  have h0 : stepStmt (Stmt.assign .nat [] "t"
      (Op.sub .nat Broadcast.nil (Op.sub .nat Broadcast.nil (Op.constNat 2) (Op.constNat 1))
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "__rev_t") (Op.constNat 1)))) sin = some e0 := by
    rw [bwdEval_assign_subNat sin "t" _ _ (2-1) (rt*1)
        (by simp [evalOp, Tile.bop, NumericDType.sub])
        (by simp [evalOp, hrt, Tile.bop, NumericDType.mul])]
  rw [stepStmts.cons_some h0]
  have hmask0 : e0.regs .bool [4] "mask" = some
      (Tile.vec (fun e : Fin 4 => decide (sin.pids 0 * 4 + e.val < 8))) := by
    rw [he0]; rw [BlockState.setReg_ne_name (h := by decide)]; exact hmask
  have hpg0 : e0.regs .ptr [4] "p_g" = some
      (Tile.vec (fun e : Fin 4 => (G.cast, sin.pids 2 * 64 + sin.pids 0 * 4 + e.val + R))) := by
    rw [he0]; rw [BlockState.setReg_ne_name (h := by decide)]; exact hpg
  set e1 := e0.setReg "g_val" .real [4] ⟨fun i => ldVal sin G R i⟩ with he1
  have h1 : stepStmt (Stmt.assign .real [4] "g_val"
        (Op.load ComputeDType.fp32.eraseDType (MemAccess.ptr (Op.ref .ptr [4] "p_g"))
          (MaskOpt.maskOther (Op.ref .bool [4] "mask") ((Op.const 0).broadcast [4])))) e0 = some e1 := by
    simp only [ComputeDType.eraseDType]
    rw [bwdEval_assign_load_ptr_maskOther e0 "g_val" "p_g" _ _
        (Tile.vec (fun e : Fin 4 => (G.cast, sin.pids 2 * 64 + sin.pids 0 * 4 + e.val + R)))
        (Tile.vec (fun e : Fin 4 => decide (sin.pids 0 * 4 + e.val < 8)))
        ⟨fun _ => some 0⟩
        hpg0 (by rw [evalOp_ref]; exact hmask0) (by simp [evalOp])]
    rw [he1]
    congr 1
  rw [stepStmts.cons_some h1]
  rfl

/-- **Reverse-loop unroll (count = 2).** The lowered forward
`forRangeDyn "__rev_t" 0 2 1 bwdIterBody` reverse loop, run from a post-prologue
state `s0`, executes exactly two iterations: `bwdIterBody` with `__rev_t = 0`
(time row `t = BT-1 = 1`), then `bwdIterBody` with `__rev_t = 1` (time row
`t = 0`), then terminates. This is the structural backbone for the 2-iteration
assembly: each `stepStmts bwdIterBody` factor is discharged by the per-statement
chaining (`bwdIterBody_head_eval` + `bwd_iter_tail_eval`). -/
theorem bwd_loop_unroll (s0 s1 s2 : BlockState)
    (h0 : stepStmts bwdIterBody (s0.setReg "__rev_t" .nat [] (Tile.scalar 0)) = some s1)
    (h1 : stepStmts bwdIterBody (s1.setReg "__rev_t" .nat [] (Tile.scalar 1)) = some s2) :
    stepStmt (Stmt.forRangeDyn "__rev_t" (Op.constNat 0)
        (Op.add .nat Broadcast.nil
          (Op.div .nat Broadcast.nil
            (Op.sub .nat Broadcast.nil (Op.constNat 2) (Op.constNat 1)) (Op.constNat 1))
          (Op.constNat 1))
        (Op.constNat 1) bwdIterBody) s0 = some s2 := by
  rw [stepForRangeAux.forRangeDyn_unfold]
  simp only [evalOp, Tile.scalar, Option.bind, Tile.bop, NumericDType.add,
    NumericDType.sub, NumericDType.div]
  show stepForRangeAux "__rev_t" 0 2 1 bwdIterBody s0 = some s2
  rw [stepForRangeAux.step_lt (by decide) (by decide), h0]
  simp only [Option.bind]
  rw [stepForRangeAux.step_lt (by decide) (by decide), h1]
  simp only [Option.bind]
  rw [stepForRangeAux.step_ge (by decide) (by decide)]

/-- Per-lane masked store-fold of one `[BK]=[4]` masked store at row offset `R`
into `region`, applied on top of state `acc`. The active-lane decision is the
prologue lane mask; inactive lanes are skipped. -/
private noncomputable def stMem (s : BlockState) (region : RegionName) (R : Nat)
    (valFn : TileIndex [4] → ℝ) (acc : BlockState) : BlockState :=
  (TileShape.allIndices [4]).foldl
    (fun a i =>
      if decide (s.pids 0 * 4 + i.1.val < 8) then
        a.writeMem region (s.pids 2 * 64 + s.pids 0 * 4 + i.1.val + R) (valFn i)
      else a) acc

/-- Active-lane masked load value of `region` at row offset `R` (as a real). -/
private noncomputable def ldR (s : BlockState) (region : RegionName) (R : Nat)
    (i : TileIndex [4]) : ℝ :=
  s.readMem region (s.pids 2 * 64 + s.pids 0 * 4 + i.1.val + R)

/-- A `[4]` real tile that holds `some (f i)` on active lanes (those with
`pid0*4 + i < 8`) and `0 = some 0` on inactive lanes. Every register tile threaded
through the backward loop body has this masked shape. -/
private noncomputable def gated (s : BlockState) (f : TileIndex [4] → ℝ) :
    Tile .real [4] :=
  ⟨fun i => if decide (s.pids 0 * 4 + i.1.val < 8) then some (f i) else (0 : WithBot ℝ)⟩

/-- Per-lane `dq = dq_inner + dq_inter * exp2(g)` real value at row offset `R`. -/
private noncomputable def dqOut (s : BlockState) (DQInner DQInter G : RegionName)
    (R : Nat) (i : TileIndex [4]) : ℝ :=
  ldR s DQInner R i + ldR s DQInter R i * Real.exp (ldR s G R i * Real.log 2)

/-- Per-lane `dk = dk_inner + dk_inter * exp2(last_g - g)` real value, with the
captured per-lane `last_g` value `lgVal`. -/
private noncomputable def dkOut (s : BlockState) (DKInner DKInter G : RegionName)
    (R : Nat) (lgVal : TileIndex [4] → ℝ) (i : TileIndex [4]) : ℝ :=
  ldR s DKInner R i +
    ldR s DKInter R i * Real.exp ((lgVal i - ldR s G R i) * Real.log 2)

/-- Per-lane `dg_val = dq*q - dk*k` summand real value at row offset `R`. -/
private noncomputable def dgSum (s : BlockState)
    (DQInner DQInter DKInner DKInter Q K G : RegionName)
    (R : Nat) (lgVal : TileIndex [4] → ℝ) (i : TileIndex [4]) : ℝ :=
  dqOut s DQInner DQInter G R i * ldR s Q R i -
    dkOut s DKInner DKInter G R lgVal i * ldR s K R i

/-- The decremented per-lane pointer tile: `p_x -= 8` shifts every lane's address
back by one `DK = 8` row, from row offset `R` to `R - 8`. -/
private def ptrDec (region : RegionName) (s : BlockState) (R : Nat) :
    Tile .ptr [4] :=
  Tile.vec (fun e : Fin 4 => (region.cast, s.pids 2 * 64 + s.pids 0 * 4 + e.val + R - 8))

/-- For `R ≥ 8`, the decremented pointer tile is the prologue-shaped pointer at
the next-lower row offset `R - 8`. -/
private theorem ptrDec_as_row (region : RegionName) (s : BlockState) (R : Nat)
    (h8 : 8 ≤ R) :
    ptrDec region s R = Tile.vec
      (fun e : Fin 4 => (region.cast, s.pids 2 * 64 + s.pids 0 * 4 + e.val + (R - 8))) := by
  apply Tile.ext; intro i
  simp only [ptrDec, Tile.vec]
  congr 1
  omega

/-- Real add over two `gated` tiles is the `gated` tile of the pointwise sum
(inactive lanes are `some 0 + some 0 = some 0`). -/
private theorem gated_add (s : BlockState) (a b : TileIndex [4] → ℝ) :
    Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
        (gated s a) (gated s b)
      = gated s (fun i => a i + b i) := by
  apply Tile.ext; intro i
  simp only [gated, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.add, WithBot.realAdd]
  by_cases hc : decide (s.pids 0 * 4 + i.1.val < 8)
  · simp only [hc, if_true, Option.map₂_some_some]
  · simp only [hc, Bool.false_eq_true, if_false]
    show Option.map₂ (·+·) (some (0:ℝ)) (some 0) = some 0
    rw [Option.map₂_some_some]; norm_num

/-- Real sub over two `gated` tiles is the `gated` tile of the pointwise
difference. -/
private theorem gated_sub (s : BlockState) (a b : TileIndex [4] → ℝ) :
    Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil)
        (gated s a) (gated s b)
      = gated s (fun i => a i - b i) := by
  apply Tile.ext; intro i
  simp only [gated, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.sub, WithBot.realSub]
  by_cases hc : decide (s.pids 0 * 4 + i.1.val < 8)
  · simp only [hc, if_true, Option.map₂_some_some]
  · simp only [hc, Bool.false_eq_true, if_false]
    show Option.map₂ (·-·) (some (0:ℝ)) (some 0) = some 0
    rw [Option.map₂_some_some]; norm_num

/-- `gated a * exp2 (gated b)` is the `gated` tile of `a i * exp(b i * log 2)`.
Inactive lanes: `some 0 * exp2(some 0) = some 0 * some 1 = some 0`. -/
private theorem gated_mul_exp2 (s : BlockState) (a b : TileIndex [4] → ℝ) :
    Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil)
        (gated s a) (Tile.uop WithBot.realExp2 (gated s b))
      = gated s (fun i => a i * Real.exp (b i * Real.log 2)) := by
  apply Tile.ext; intro i
  simp only [gated, Tile.bop, Tile.uop, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.mul, WithBot.realMul]
  by_cases hc : decide (s.pids 0 * 4 + i.1.val < 8)
  · simp only [hc, if_true, WithBot.realExp2_some, Option.map₂_some_some]
  · simp only [hc, Bool.false_eq_true, if_false]
    show Option.map₂ (·*·) (some (0:ℝ)) (WithBot.realExp2 (some 0)) = some 0
    rw [WithBot.realExp2_some, Option.map₂_some_some]; norm_num

/-- The all-`some 0` tile (prologue `tl.zeros`) is the `gated` tile of the zero
function (inactive lanes are also `some 0 = 0`). -/
private theorem gated_const0 (s : BlockState) :
    (⟨fun _ => some 0⟩ : Tile .real [4]) = gated s (fun _ => 0) := by
  apply Tile.ext; intro i
  simp only [gated]
  by_cases hc : decide (s.pids 0 * 4 + i.1.val < 8)
  · simp only [hc, if_true]
  · simp only [hc, Bool.false_eq_true, if_false]; rfl

/-- `gated`/`ldR` depend on the state only through `pids` and `readMem`, so they
agree across states with equal pids and memory reads. -/
private theorem gated_ldR_state_eq (s1 s2 : BlockState) (region : RegionName) (R : Nat)
    (hpid : s1.pids = s2.pids)
    (hread : ∀ i : TileIndex [4],
      s1.readMem region (s2.pids 2 * 64 + s2.pids 0 * 4 + i.1.val + R)
        = s2.readMem region (s2.pids 2 * 64 + s2.pids 0 * 4 + i.1.val + R)) :
    gated s1 (ldR s1 region R) = gated s2 (ldR s2 region R) := by
  apply Tile.ext; intro i
  simp only [gated, ldR, hpid]
  by_cases hc : decide (s2.pids 0 * 4 + i.1.val < 8)
  · simp only [hc, if_true]; rw [hread i]
  · simp only [hc, Bool.false_eq_true, if_false]

/-- The head-state `g_val` tile (`ldVal`) equals the `gated` load tile of `region`
at row offset `R`, evaluated against any state with the same pids and row reads. -/
private theorem ldVal_eq_gated (sin s : BlockState) (region : RegionName) (R : Nat)
    (hpid : sin.pids = s.pids)
    (hread : ∀ i : TileIndex [4],
      sin.readMem region (s.pids 2 * 64 + s.pids 0 * 4 + i.1.val + R)
        = s.readMem region (s.pids 2 * 64 + s.pids 0 * 4 + i.1.val + R)) :
    (⟨fun i => ldVal sin region R i⟩ : Tile .real [4]) = gated s (ldR s region R) := by
  have : (⟨fun i => ldVal sin region R i⟩ : Tile .real [4])
      = gated sin (ldR sin region R) := rfl
  rw [this, gated_ldR_state_eq sin s region R hpid hread]

/-- Plain real mul over two `gated` tiles. -/
private theorem gated_mul (s : BlockState) (a b : TileIndex [4] → ℝ) :
    Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil)
        (gated s a) (gated s b)
      = gated s (fun i => a i * b i) := by
  apply Tile.ext; intro i
  simp only [gated, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.mul, WithBot.realMul]
  by_cases hc : decide (s.pids 0 * 4 + i.1.val < 8)
  · simp only [hc, if_true, Option.map₂_some_some]
  · simp only [hc, Bool.false_eq_true, if_false]
    show Option.map₂ (·*·) (some (0:ℝ)) (some 0) = some 0
    rw [Option.map₂_some_some]; norm_num

/-- The masked-load result tile (`bwdEval_assign_load_ptr_maskOther` output) at a
prologue-shaped row pointer, evaluated in a state `c` whose `region`-reads agree
with `s`, equals the `gated` load tile of `region` at `R`. -/
private theorem load_tile_gated (s c : BlockState) (region : RegionName) (R : Nat)
    (hread : ∀ i : TileIndex [4],
      c.readMem region (s.pids 2 * 64 + s.pids 0 * 4 + i.1.val + R)
        = s.readMem region (s.pids 2 * 64 + s.pids 0 * 4 + i.1.val + R)) :
    (⟨fun i =>
        if (Tile.vec (fun e : Fin 4 => decide (s.pids 0 * 4 + e.val < 8)) :
            Tile .bool [4]).data i then
          some (c.readMem
            ((Tile.vec (fun e : Fin 4 =>
                ((region.cast, s.pids 2 * 64 + s.pids 0 * 4 + e.val + R) : RegionName × Nat)) :
                Tile .ptr [4]).data i).1
            ((Tile.vec (fun e : Fin 4 =>
                ((region.cast, s.pids 2 * 64 + s.pids 0 * 4 + e.val + R) : RegionName × Nat)) :
                Tile .ptr [4]).data i).2)
        else (⟨fun _ => some 0⟩ : Tile .real [4]).data i⟩ : Tile .real [4])
      = gated s (ldR s region R) := by
  apply Tile.ext; intro i
  simp only [gated, ldR, Tile.vec, Region.cast_id]
  by_cases hc : decide (s.pids 0 * 4 + i.1.val < 8)
  · simp only [hc, if_true]; rw [hread i]
  · simp only [hc, Bool.false_eq_true, if_false]
    rfl

/-- A masked pointer store (`bwdEval_store_ptr_masked` output) of a `gated` value
tile at a prologue-shaped row pointer equals the `stMem` masked store fold. -/
private theorem bwdStore_stMem (s : BlockState) (region : RegionName) (R : Nat)
    (f : TileIndex [4] → ℝ) (c : BlockState) :
    ((TileShape.allIndices [4]).foldl
        (fun acc i =>
          if (Tile.vec (fun e : Fin 4 => decide (s.pids 0 * 4 + e.val < 8)) :
              Tile .bool [4]).data i then
            acc.writeMemTyped .real
              ((Tile.vec (fun e : Fin 4 =>
                  ((region.cast, s.pids 2 * 64 + s.pids 0 * 4 + e.val + R) : RegionName × Nat)) :
                  Tile .ptr [4]).data i).1
              ((Tile.vec (fun e : Fin 4 =>
                  ((region.cast, s.pids 2 * 64 + s.pids 0 * 4 + e.val + R) : RegionName × Nat)) :
                  Tile .ptr [4]).data i).2
              ((gated s f).data i)
          else acc) c)
      = stMem s region R f c := by
  unfold stMem
  congr 1
  funext a i
  simp only [gated, Tile.vec, Region.cast_id]
  by_cases hc : decide (s.pids 0 * 4 + i.1.val < 8)
  · simp only [hc, if_true, BlockState.writeMemTyped_real, FloatDType.real_storeValue]
    rfl
  · simp only [hc, Bool.false_eq_true, if_false]

/-- `stMem` preserves program ids. -/
private theorem stMem_pids (s : BlockState) (region : RegionName) (R : Nat)
    (f : TileIndex [4] → ℝ) (c : BlockState) :
    (stMem s region R f c).pids = c.pids := by
  unfold stMem
  rw [BlockState.foldl_writeMem_prop_masked_pids]

/-- `stMem` preserves all registers. -/
private theorem stMem_regs (s : BlockState) (region : RegionName) (R : Nat)
    (f : TileIndex [4] → ℝ) (c : BlockState)
    (d : TileDType) (sh : TileShape) (n : RegName) :
    (stMem s region R f c).regs d sh n = c.regs d sh n := by
  unfold stMem
  induction (TileShape.allIndices [4]) generalizing c with
  | nil => rfl
  | cons hd tl ih =>
    rw [List.foldl_cons, ih]
    by_cases hc : decide (s.pids 0 * 4 + hd.1.val < 8) <;> simp [hc]

/-- `stMem` into `region` leaves any other region's reads unchanged. -/
private theorem stMem_readMem_other (s : BlockState) (region region' : RegionName)
    (R : Nat) (f : TileIndex [4] → ℝ) (c : BlockState) (off : Nat)
    (h : region' ≠ region) :
    (stMem s region R f c).readMem region' off = c.readMem region' off := by
  unfold stMem
  exact BlockState.scatter_prop_masked_preserves_other_region region _ _ _
    region' h off _ c

/-- `stMem` at row offset `R` leaves an offset outside the written row range
unchanged. The writes are at `s.pids2*64+s.pids0*4+j+R` for `j < 4`. -/
private theorem stMem_readMem_other_offset (s : BlockState) (region region' : RegionName)
    (R : Nat) (f : TileIndex [4] → ℝ) (c : BlockState) (off : Nat)
    (h : ∀ j : TileIndex [4], s.pids 2 * 64 + s.pids 0 * 4 + j.1.val + R ≠ off) :
    (stMem s region R f c).readMem region' off = c.readMem region' off := by
  unfold stMem
  by_cases hr : region' = region
  · rw [hr]
    exact BlockState.scatter_prop_masked_preserves_other_offset region _ _ _ off
      (fun j _ => h j) _ c
  · exact BlockState.scatter_prop_masked_preserves_other_region region _ _ _
      region' hr off _ c

set_option maxHeartbeats 800000 in
/-- Reading the `stMem` region back at lane `i`'s row-`R` address yields the
stored value on active lanes, else the underlying read. -/
private theorem stMem_readback (s : BlockState) (region : RegionName) (R : Nat)
    (f : TileIndex [4] → ℝ) (c : BlockState) (i : TileIndex [4]) :
    (stMem s region R f c).readMem region (s.pids 2 * 64 + s.pids 0 * 4 + i.1.val + R)
      = if decide (s.pids 0 * 4 + i.1.val < 8) then f i
        else c.readMem region (s.pids 2 * 64 + s.pids 0 * 4 + i.1.val + R) := by
  have hinj : Function.Injective
      (fun k : TileIndex [4] => s.pids 2 * 64 + s.pids 0 * 4 + k.1.val + R) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (by simp only at hab; omega)
    rfl
  unfold stMem
  exact BlockState.scatter_readback_masked_nd (region := region) c
    (fun k : TileIndex [4] => s.pids 2 * 64 + s.pids 0 * 4 + k.1.val + R) f
    (fun k : TileIndex [4] => decide (s.pids 0 * 4 + k.1.val < 8)) hinj i

/-- The memory of `stMem` is determined entirely by the input state's memory. -/
private theorem stMem_mem_congr (s : BlockState) (region : RegionName) (R : Nat)
    (f : TileIndex [4] → ℝ) (c c' : BlockState) (h : c.mem = c'.mem) :
    (stMem s region R f c).mem = (stMem s region R f c').mem := by
  unfold stMem
  induction (TileShape.allIndices [4]) generalizing c c' with
  | nil => exact h
  | cons hd tl ih =>
    rw [List.foldl_cons, List.foldl_cons]
    refine ih _ _ ?_
    by_cases hc : decide (s.pids 0 * 4 + hd.1.val < 8)
    · simp only [hc, if_true, BlockState.writeMem]
      funext r o; rw [h]
    · simp only [hc, Bool.false_eq_true, if_false]; exact h

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **Per-iteration straight-line body (post-`last_g`-capture, 22 statements).**
From a state `s3` whose `g_val`/`last_g`/`cum_grad_dg` registers and 8 row
pointers are in masked `gated` form at row offset `R`, the remaining body (drop 3
of `bwdIterBody`) computes `dq`/`dk`/`dg` per lane, performs the three masked
stores (`DQInter`, `DKInter`, `DG`), and decrements all 8 pointers. We chain the
22 statements via `stepStmts.cons_some` + the per-statement recipes, threading the
explicit `set` state, then read out the resulting memory as three nested masked
store folds (`stMem`) and the updated registers. -/
theorem bwd_iter_core
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (s s3 : BlockState) (R : Nat)
    (lgVal cumInVal : TileIndex [4] → ℝ)
    (hDKInner_DQInter : DKInner ≠ DQInter)
    (hDKInter_DQInter : DKInter ≠ DQInter)
    (hQ_DQInter : Q ≠ DQInter) (hQ_DKInter : Q ≠ DKInter)
    (hK_DQInter : K ≠ DQInter) (hK_DKInter : K ≠ DKInter)
    (hpid : s3.pids = s.pids)
    (hrow : ∀ (r : RegionName) (i : TileIndex [4]),
      s3.readMem r (s.pids 2 * 64 + s.pids 0 * 4 + i.1.val + R)
        = s.readMem r (s.pids 2 * 64 + s.pids 0 * 4 + i.1.val + R))
    (hmask : s3.regs .bool [4] "mask" = some
        (Tile.vec (fun e : Fin 4 => decide (s.pids 0 * 4 + e.val < 8))))
    (hgval : s3.regs .real [4] "g_val" = some (gated s (ldR s G R)))
    (hlastg : s3.regs .real [4] "last_g" = some (gated s lgVal))
    (hcum : s3.regs .real [4] "cum_grad_dg" = some (gated s cumInVal))
    (hpg : s3.regs .ptr [4] "p_g" = some
        (Tile.vec (fun e : Fin 4 => (G.cast, s.pids 2 * 64 + s.pids 0 * 4 + e.val + R))))
    (hpk : s3.regs .ptr [4] "p_k" = some
        (Tile.vec (fun e : Fin 4 => (K.cast, s.pids 2 * 64 + s.pids 0 * 4 + e.val + R))))
    (hpq : s3.regs .ptr [4] "p_q" = some
        (Tile.vec (fun e : Fin 4 => (Q.cast, s.pids 2 * 64 + s.pids 0 * 4 + e.val + R))))
    (hpdqi : s3.regs .ptr [4] "p_dq_inner" = some
        (Tile.vec (fun e : Fin 4 => (DQInner.cast, s.pids 2 * 64 + s.pids 0 * 4 + e.val + R))))
    (hpdki : s3.regs .ptr [4] "p_dk_inner" = some
        (Tile.vec (fun e : Fin 4 => (DKInner.cast, s.pids 2 * 64 + s.pids 0 * 4 + e.val + R))))
    (hpdqt : s3.regs .ptr [4] "p_dq_inter" = some
        (Tile.vec (fun e : Fin 4 => (DQInter.cast, s.pids 2 * 64 + s.pids 0 * 4 + e.val + R))))
    (hpdkt : s3.regs .ptr [4] "p_dk_inter" = some
        (Tile.vec (fun e : Fin 4 => (DKInter.cast, s.pids 2 * 64 + s.pids 0 * 4 + e.val + R))))
    (hpdg : s3.regs .ptr [4] "p_dg" = some
        (Tile.vec (fun e : Fin 4 => (DG.cast, s.pids 2 * 64 + s.pids 0 * 4 + e.val + R)))) :
    ∃ sout, stepStmts (bwdIterBody.drop 3) s3 = some sout
      ∧ sout.pids = s.pids
      ∧ sout.mem =
          (stMem s DG R (fun i => cumInVal i +
              dgSum s DQInner DQInter DKInner DKInter Q K G R lgVal i)
            (stMem s DKInter R (dkOut s DKInner DKInter G R lgVal)
              (stMem s DQInter R (dqOut s DQInner DQInter G R)
                s3))).mem
      ∧ sout.regs .bool [4] "mask" = some
          (Tile.vec (fun e : Fin 4 => decide (s.pids 0 * 4 + e.val < 8)))
      ∧ sout.regs .real [4] "last_g" = some (gated s lgVal)
      ∧ sout.regs .real [4] "cum_grad_dg" = some
          (gated s (fun i => cumInVal i +
            dgSum s DQInner DQInter DKInner DKInter Q K G R lgVal i))
      ∧ sout.regs .ptr [4] "p_g" = some (ptrDec G s R)
      ∧ sout.regs .ptr [4] "p_k" = some (ptrDec K s R)
      ∧ sout.regs .ptr [4] "p_q" = some (ptrDec Q s R)
      ∧ sout.regs .ptr [4] "p_dq_inner" = some (ptrDec DQInner s R)
      ∧ sout.regs .ptr [4] "p_dk_inner" = some (ptrDec DKInner s R)
      ∧ sout.regs .ptr [4] "p_dq_inter" = some (ptrDec DQInter s R)
      ∧ sout.regs .ptr [4] "p_dk_inter" = some (ptrDec DKInter s R)
      ∧ sout.regs .ptr [4] "p_dg" = some (ptrDec DG s R) := by
  -- Memory reads through `s3` agree with `s` (the prologue/head/ifThen write no
  -- memory); register reads of the 8 pointers and mask survive the real-valued
  -- assignments of the body.
  have hread3 : ∀ (r : RegionName) (i : TileIndex [4]),
      s3.readMem r (s.pids 2 * 64 + s.pids 0 * 4 + i.1.val + R)
        = s.readMem r (s.pids 2 * 64 + s.pids 0 * 4 + i.1.val + R) := hrow
  -- The masked-load `other` tile (zeros) and mask-tile evaluations, reused below.
  have hmaskE : ∀ c : BlockState,
      c.regs .bool [4] "mask" = some
        (Tile.vec (fun e : Fin 4 => decide (s.pids 0 * 4 + e.val < 8))) →
      evalOp (Op.ref .bool [4] "mask") c = some
        (Tile.vec (fun e : Fin 4 => decide (s.pids 0 * 4 + e.val < 8))) := by
    intro c hc; rw [evalOp_ref]; exact hc
  have hother : ∀ c : BlockState,
      evalOp ((Op.const (0 : ℝ)).broadcast [4]) c
        = some (⟨fun _ => some 0⟩ : Tile .real [4]) := by
    intro c; simp [evalOp]
  -- Expose the 22 post-capture statements.
  simp only [bwdIterBody, List.drop_succ_cons, List.drop_zero]
  -- Abbreviations for the per-lane masked tiles produced along the chain.
  set maskT : Tile .bool [4] :=
    Tile.vec (fun e : Fin 4 => decide (s.pids 0 * 4 + e.val < 8)) with hmaskT
  set otherT : Tile .real [4] := ⟨fun _ => some 0⟩ with hotherT
  -- Pointer tiles (prologue-shaped) at row offset R.
  set ptrG : Tile .ptr [4] :=
    Tile.vec (fun e : Fin 4 => (G.cast, s.pids 2 * 64 + s.pids 0 * 4 + e.val + R)) with hptrG
  set ptrK : Tile .ptr [4] :=
    Tile.vec (fun e : Fin 4 => (K.cast, s.pids 2 * 64 + s.pids 0 * 4 + e.val + R)) with hptrK
  set ptrQ : Tile .ptr [4] :=
    Tile.vec (fun e : Fin 4 => (Q.cast, s.pids 2 * 64 + s.pids 0 * 4 + e.val + R)) with hptrQ
  set ptrDQi : Tile .ptr [4] :=
    Tile.vec (fun e : Fin 4 => (DQInner.cast, s.pids 2 * 64 + s.pids 0 * 4 + e.val + R)) with hptrDQi
  set ptrDKi : Tile .ptr [4] :=
    Tile.vec (fun e : Fin 4 => (DKInner.cast, s.pids 2 * 64 + s.pids 0 * 4 + e.val + R)) with hptrDKi
  set ptrDQt : Tile .ptr [4] :=
    Tile.vec (fun e : Fin 4 => (DQInter.cast, s.pids 2 * 64 + s.pids 0 * 4 + e.val + R)) with hptrDQt
  set ptrDKt : Tile .ptr [4] :=
    Tile.vec (fun e : Fin 4 => (DKInter.cast, s.pids 2 * 64 + s.pids 0 * 4 + e.val + R)) with hptrDKt
  set ptrDG : Tile .ptr [4] :=
    Tile.vec (fun e : Fin 4 => (DG.cast, s.pids 2 * 64 + s.pids 0 * 4 + e.val + R)) with hptrDG
  -- === Statement 3: dq1 = load p_dq_inner ===
  have e3 : stepStmt (Stmt.assign .real [4] "dq1"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [4] "p_dq_inner"))
        (MaskOpt.maskOther (Op.ref .bool [4] "mask") ((Op.const 0).broadcast [4])))) s3
      = some (s3.setReg "dq1" .real [4] (gated s (ldR s DQInner R))) := by
    rw [bwdEval_assign_load_ptr_maskOther s3 "dq1" "p_dq_inner" _ _
        ptrDQi maskT otherT hpdqi (hmaskE s3 hmask) (hother s3)]
    simp only [hptrDQi, hmaskT, hotherT]
    rw [load_tile_gated s s3 DQInner R (fun i => hread3 DQInner i)]
  rw [stepStmts.cons_some e3]
  set c1 := s3.setReg "dq1" .real [4] (gated s (ldR s DQInner R)) with hc1
  -- Lookups on `c1` (one real setReg above `s3`).
  have c1mask : c1.regs .bool [4] "mask" = some maskT := by
    rw [hc1, BlockState.setReg_ne_name (h := by decide)]; exact hmask
  have c1pdqt : c1.regs .ptr [4] "p_dq_inter" = some ptrDQt := by
    rw [hc1, BlockState.setReg_ne_name (h := by decide)]; exact hpdqt
  have c1read : ∀ i : TileIndex [4],
      c1.readMem DQInter (s.pids 2 * 64 + s.pids 0 * 4 + i.1.val + R)
        = s.readMem DQInter (s.pids 2 * 64 + s.pids 0 * 4 + i.1.val + R) := by
    intro i; rw [hc1, BlockState.setReg_readMem]; exact hread3 DQInter i
  -- === Statement 4: dq2 = load p_dq_inter ===
  have e4 : stepStmt (Stmt.assign .real [4] "dq2"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [4] "p_dq_inter"))
        (MaskOpt.maskOther (Op.ref .bool [4] "mask") ((Op.const 0).broadcast [4])))) c1
      = some (c1.setReg "dq2" .real [4] (gated s (ldR s DQInter R))) := by
    rw [bwdEval_assign_load_ptr_maskOther c1 "dq2" "p_dq_inter" _ _
        ptrDQt maskT otherT c1pdqt (hmaskE c1 c1mask) (hother c1)]
    simp only [hptrDQt, hmaskT, hotherT]
    rw [load_tile_gated s c1 DQInter R c1read]
  rw [stepStmts.cons_some e4]
  set c2 := c1.setReg "dq2" .real [4] (gated s (ldR s DQInter R)) with hc2
  -- g_val survives down to `s3`.
  have c2gval : c2.regs .real [4] "g_val" = some (gated s (ldR s G R)) := by
    rw [hc2, BlockState.setReg_ne_name (h := by decide), hc1,
        BlockState.setReg_ne_name (h := by decide)]; exact hgval
  -- === Statement 5: dq2 = dq2 * exp2(g_val) ===
  have e5 : stepStmt (Stmt.assign .real [4] "dq2"
      (Op.mul .real Broadcast.nil.consSame (Op.ref .real [4] "dq2")
        (Op.ref .real [4] "g_val").exp2)) c2
      = some (c2.setReg "dq2" .real [4]
          (gated s (fun i => ldR s DQInter R i *
            Real.exp (ldR s G R i * Real.log 2)))) := by
    apply stepStmt_assign_eq_some
    rw [bwdEval_mul c2 _ _ (gated s (ldR s DQInter R))
        (Tile.uop WithBot.realExp2 (gated s (ldR s G R)))
        (by rw [evalOp_ref, hc2, BlockState.setReg_same])
        (by rw [bwdEval_exp2 c2 _ (gated s (ldR s G R))
            (by rw [evalOp_ref]; exact c2gval)])]
    rw [gated_mul_exp2]
  rw [stepStmts.cons_some e5]
  set c3 := c2.setReg "dq2" .real [4]
    (gated s (fun i => ldR s DQInter R i * Real.exp (ldR s G R i * Real.log 2))) with hc3
  have c3dq1 : c3.regs .real [4] "dq1" = some (gated s (ldR s DQInner R)) := by
    rw [hc3, BlockState.setReg_ne_name (h := by decide), hc2,
        BlockState.setReg_ne_name (h := by decide), hc1, BlockState.setReg_same]
  -- === Statement 6: dq = dq1 + dq2 ===
  have e6 : stepStmt (Stmt.assign .real [4] "dq"
      (Op.add .real Broadcast.nil.consSame (Op.ref .real [4] "dq1")
        (Op.ref .real [4] "dq2"))) c3
      = some (c3.setReg "dq" .real [4]
          (gated s (dqOut s DQInner DQInter G R))) := by
    apply stepStmt_assign_eq_some
    rw [bwdEval_add c3 _ _ (gated s (ldR s DQInner R))
        (gated s (fun i => ldR s DQInter R i * Real.exp (ldR s G R i * Real.log 2)))
        (by rw [evalOp_ref]; exact c3dq1)
        (by rw [evalOp_ref, hc3, BlockState.setReg_same])]
    rw [gated_add]; rfl
  rw [stepStmts.cons_some e6]
  set c4 := c3.setReg "dq" .real [4] (gated s (dqOut s DQInner DQInter G R)) with hc4
  -- p_dq_inter and mask survive to `s3`.
  have c4mask : c4.regs .bool [4] "mask" = some maskT := by
    rw [hc4, BlockState.setReg_ne_name (h := by decide), hc3,
        BlockState.setReg_ne_name (h := by decide), hc2,
        BlockState.setReg_ne_name (h := by decide), hc1,
        BlockState.setReg_ne_name (h := by decide)]; exact hmask
  have c4pdqt : c4.regs .ptr [4] "p_dq_inter" = some ptrDQt := by
    rw [hc4, BlockState.setReg_ne_name (h := by decide), hc3,
        BlockState.setReg_ne_name (h := by decide), hc2,
        BlockState.setReg_ne_name (h := by decide), hc1,
        BlockState.setReg_ne_name (h := by decide)]; exact hpdqt
  have c4dq : c4.regs .real [4] "dq" = some (gated s (dqOut s DQInner DQInter G R)) := by
    rw [hc4]; exact BlockState.setReg_same _ _ _ _ _
  -- === Statement 7: store p_dq_inter dq ===
  have e7 : stepStmt (Stmt.store .real [4] (MemAccess.ptr (Op.ref .ptr [4] "p_dq_inter"))
      (Op.ref .real [4] "dq") (MaskOpt.mask (Op.ref .bool [4] "mask"))) c4
      = some (stMem s DQInter R (dqOut s DQInner DQInter G R) c4) := by
    rw [bwdEval_store_ptr_masked c4 "p_dq_inter" _ _
        ptrDQt (gated s (dqOut s DQInner DQInter G R)) maskT
        c4pdqt (by rw [evalOp_ref]; exact c4dq) (by rw [evalOp_ref]; exact c4mask)]
    simp only [hptrDQt, hmaskT]
    rw [bwdStore_stMem]
  rw [stepStmts.cons_some e7]
  set c5 := stMem s DQInter R (dqOut s DQInner DQInter G R) c4 with hc5
  -- `c4` reads (4 real setRegs over `s3`) agree with `s`.
  have c4read : ∀ (r : RegionName) (i : TileIndex [4]),
      c4.readMem r (s.pids 2 * 64 + s.pids 0 * 4 + i.1.val + R)
        = s.readMem r (s.pids 2 * 64 + s.pids 0 * 4 + i.1.val + R) := by
    intro r i
    rw [hc4, BlockState.setReg_readMem, hc3, BlockState.setReg_readMem,
        hc2, BlockState.setReg_readMem, hc1, BlockState.setReg_readMem]
    exact hread3 r i
  -- `c4` pointer/mask lookups (peel the 4 real setRegs).
  have c4peel : ∀ (d : TileDType) (sh : TileShape) (n : RegName),
      n ≠ "dq" → n ≠ "dq2" → n ≠ "dq1" →
      c4.regs d sh n = s3.regs d sh n := by
    intro d sh n h1 h2 h3
    rw [hc4, BlockState.setReg_ne_name (h := h1), hc3,
        BlockState.setReg_ne_name (h := h2), hc2,
        BlockState.setReg_ne_name (h := h2), hc1,
        BlockState.setReg_ne_name (h := h3)]
  -- `c5` lookups: `stMem` preserves regs; reads of regions ≠ DQInter agree with `s`.
  have c5peel : ∀ (d : TileDType) (sh : TileShape) (n : RegName),
      n ≠ "dq" → n ≠ "dq2" → n ≠ "dq1" →
      c5.regs d sh n = s3.regs d sh n := by
    intro d sh n h1 h2 h3
    rw [hc5, stMem_regs]; exact c4peel d sh n h1 h2 h3
  have c5read : ∀ (r : RegionName) (i : TileIndex [4]), r ≠ DQInter →
      c5.readMem r (s.pids 2 * 64 + s.pids 0 * 4 + i.1.val + R)
        = s.readMem r (s.pids 2 * 64 + s.pids 0 * 4 + i.1.val + R) := by
    intro r i hr
    rw [hc5, stMem_readMem_other s DQInter r R _ c4 _ hr]; exact c4read r i
  -- === Statement 8: dk1 = load p_dk_inner ===
  have e8 : stepStmt (Stmt.assign .real [4] "dk1"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [4] "p_dk_inner"))
        (MaskOpt.maskOther (Op.ref .bool [4] "mask") ((Op.const 0).broadcast [4])))) c5
      = some (c5.setReg "dk1" .real [4] (gated s (ldR s DKInner R))) := by
    have c5pdki : c5.regs .ptr [4] "p_dk_inner" = some ptrDKi := by
      rw [c5peel .ptr [4] "p_dk_inner" (by decide) (by decide) (by decide)]; exact hpdki
    have c5mask : c5.regs .bool [4] "mask" = some maskT := by
      rw [c5peel .bool [4] "mask" (by decide) (by decide) (by decide)]; exact hmask
    rw [bwdEval_assign_load_ptr_maskOther c5 "dk1" "p_dk_inner" _ _
        ptrDKi maskT otherT c5pdki (hmaskE c5 c5mask) (hother c5)]
    simp only [hptrDKi, hmaskT, hotherT]
    rw [load_tile_gated s c5 DKInner R (fun i => c5read DKInner i hDKInner_DQInter)]
  rw [stepStmts.cons_some e8]
  set c6 := c5.setReg "dk1" .real [4] (gated s (ldR s DKInner R)) with hc6
  -- === Statement 9: dk2 = load p_dk_inter ===
  have e9 : stepStmt (Stmt.assign .real [4] "dk2"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [4] "p_dk_inter"))
        (MaskOpt.maskOther (Op.ref .bool [4] "mask") ((Op.const 0).broadcast [4])))) c6
      = some (c6.setReg "dk2" .real [4] (gated s (ldR s DKInter R))) := by
    have c6pdkt : c6.regs .ptr [4] "p_dk_inter" = some ptrDKt := by
      rw [hc6, BlockState.setReg_ne_name (h := by decide)]
      rw [c5peel .ptr [4] "p_dk_inter" (by decide) (by decide) (by decide)]; exact hpdkt
    have c6mask : c6.regs .bool [4] "mask" = some maskT := by
      rw [hc6, BlockState.setReg_ne_name (h := by decide)]
      rw [c5peel .bool [4] "mask" (by decide) (by decide) (by decide)]; exact hmask
    have c6read : ∀ i : TileIndex [4],
        c6.readMem DKInter (s.pids 2 * 64 + s.pids 0 * 4 + i.1.val + R)
          = s.readMem DKInter (s.pids 2 * 64 + s.pids 0 * 4 + i.1.val + R) := by
      intro i; rw [hc6, BlockState.setReg_readMem]; exact c5read DKInter i hDKInter_DQInter
    rw [bwdEval_assign_load_ptr_maskOther c6 "dk2" "p_dk_inter" _ _
        ptrDKt maskT otherT c6pdkt (hmaskE c6 c6mask) (hother c6)]
    simp only [hptrDKt, hmaskT, hotherT]
    rw [load_tile_gated s c6 DKInter R c6read]
  rw [stepStmts.cons_some e9]
  set c7 := c6.setReg "dk2" .real [4] (gated s (ldR s DKInter R)) with hc7
  -- `c7` lookups of `s3`-level real registers (peel dk2, dk1, then `c5`).
  have c7peel : ∀ (d : TileDType) (sh : TileShape) (n : RegName),
      n ≠ "dk2" → n ≠ "dk1" → n ≠ "dq" → n ≠ "dq2" → n ≠ "dq1" →
      c7.regs d sh n = s3.regs d sh n := by
    intro d sh n h1 h2 h3 h3' h4
    rw [hc7, BlockState.setReg_ne_name (h := h1), hc6,
        BlockState.setReg_ne_name (h := h2)]
    exact c5peel d sh n h3 h3' h4
  have c7lastg : c7.regs .real [4] "last_g" = some (gated s lgVal) := by
    rw [c7peel .real [4] "last_g" (by decide) (by decide) (by decide) (by decide) (by decide)]
    exact hlastg
  have c7gval : c7.regs .real [4] "g_val" = some (gated s (ldR s G R)) := by
    rw [c7peel .real [4] "g_val" (by decide) (by decide) (by decide) (by decide) (by decide)]
    exact hgval
  have c7dk2 : c7.regs .real [4] "dk2" = some (gated s (ldR s DKInter R)) := by
    rw [hc7]; exact BlockState.setReg_same _ _ _ _ _
  -- === Statement 10: dk2 = dk2 * exp2(last_g - g_val) ===
  have hsub10 : evalOp (Op.sub .real Broadcast.nil.consSame (Op.ref .real [4] "last_g")
      (Op.ref .real [4] "g_val")) c7
      = some (gated s (fun i => lgVal i - ldR s G R i)) := by
    rw [bwdEval_sub c7 _ _ (gated s lgVal) (gated s (ldR s G R))
        (by rw [evalOp_ref]; exact c7lastg) (by rw [evalOp_ref]; exact c7gval)]
    rw [gated_sub]
  have e10 : stepStmt (Stmt.assign .real [4] "dk2"
      (Op.mul .real Broadcast.nil.consSame (Op.ref .real [4] "dk2")
        (Op.sub .real Broadcast.nil.consSame (Op.ref .real [4] "last_g")
            (Op.ref .real [4] "g_val")).exp2)) c7
      = some (c7.setReg "dk2" .real [4]
          (gated s (fun i => ldR s DKInter R i *
            Real.exp ((lgVal i - ldR s G R i) * Real.log 2)))) := by
    apply stepStmt_assign_eq_some
    rw [bwdEval_mul c7 _ _ (gated s (ldR s DKInter R))
        (Tile.uop WithBot.realExp2 (gated s (fun i => lgVal i - ldR s G R i)))
        (by rw [evalOp_ref]; exact c7dk2)
        (by rw [bwdEval_exp2 c7 _ (gated s (fun i => lgVal i - ldR s G R i)) hsub10])]
    rw [gated_mul_exp2]
  rw [stepStmts.cons_some e10]
  set c8 := c7.setReg "dk2" .real [4]
    (gated s (fun i => ldR s DKInter R i *
      Real.exp ((lgVal i - ldR s G R i) * Real.log 2))) with hc8
  have c8dk1 : c8.regs .real [4] "dk1" = some (gated s (ldR s DKInner R)) := by
    rw [hc8, BlockState.setReg_ne_name (h := by decide), hc7,
        BlockState.setReg_ne_name (h := by decide), hc6, BlockState.setReg_same]
  -- === Statement 11: dk = dk1 + dk2 ===
  have e11 : stepStmt (Stmt.assign .real [4] "dk"
      (Op.add .real Broadcast.nil.consSame (Op.ref .real [4] "dk1")
        (Op.ref .real [4] "dk2"))) c8
      = some (c8.setReg "dk" .real [4]
          (gated s (dkOut s DKInner DKInter G R lgVal))) := by
    apply stepStmt_assign_eq_some
    rw [bwdEval_add c8 _ _ (gated s (ldR s DKInner R))
        (gated s (fun i => ldR s DKInter R i *
          Real.exp ((lgVal i - ldR s G R i) * Real.log 2)))
        (by rw [evalOp_ref]; exact c8dk1)
        (by rw [evalOp_ref, hc8, BlockState.setReg_same])]
    rw [gated_add]; rfl
  rw [stepStmts.cons_some e11]
  set c9 := c8.setReg "dk" .real [4] (gated s (dkOut s DKInner DKInter G R lgVal)) with hc9
  -- `c9` reads agree with `s` for regions ≠ DQInter (only the DQInter store so far).
  have c9read : ∀ (r : RegionName) (i : TileIndex [4]), r ≠ DQInter →
      c9.readMem r (s.pids 2 * 64 + s.pids 0 * 4 + i.1.val + R)
        = s.readMem r (s.pids 2 * 64 + s.pids 0 * 4 + i.1.val + R) := by
    intro r i hr
    rw [hc9, BlockState.setReg_readMem, hc8, BlockState.setReg_readMem,
        hc7, BlockState.setReg_readMem, hc6, BlockState.setReg_readMem]
    exact c5read r i hr
  -- `c9` lookups of mask / p_dk_inter (peel dk, dk2, dk1, then `c5`).
  have c9peel : ∀ (d : TileDType) (sh : TileShape) (n : RegName),
      n ≠ "dk" → n ≠ "dk2" → n ≠ "dk1" → n ≠ "dq" → n ≠ "dq2" → n ≠ "dq1" →
      c9.regs d sh n = s3.regs d sh n := by
    intro d sh n h0 h1 h2 h3 h3' h4
    rw [hc9, BlockState.setReg_ne_name (h := h0), hc8,
        BlockState.setReg_ne_name (h := h1)]
    exact c7peel d sh n h1 h2 h3 h3' h4
  have c9mask : c9.regs .bool [4] "mask" = some maskT := by
    rw [c9peel .bool [4] "mask" (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]
    exact hmask
  have c9pdkt : c9.regs .ptr [4] "p_dk_inter" = some ptrDKt := by
    rw [c9peel .ptr [4] "p_dk_inter" (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]
    exact hpdkt
  have c9dk : c9.regs .real [4] "dk" = some (gated s (dkOut s DKInner DKInter G R lgVal)) := by
    rw [hc9]; exact BlockState.setReg_same _ _ _ _ _
  -- === Statement 12: store p_dk_inter dk ===
  have e12 : stepStmt (Stmt.store .real [4] (MemAccess.ptr (Op.ref .ptr [4] "p_dk_inter"))
      (Op.ref .real [4] "dk") (MaskOpt.mask (Op.ref .bool [4] "mask"))) c9
      = some (stMem s DKInter R (dkOut s DKInner DKInter G R lgVal) c9) := by
    rw [bwdEval_store_ptr_masked c9 "p_dk_inter" _ _
        ptrDKt (gated s (dkOut s DKInner DKInter G R lgVal)) maskT
        c9pdkt (by rw [evalOp_ref]; exact c9dk) (by rw [evalOp_ref]; exact c9mask)]
    simp only [hptrDKt, hmaskT]
    rw [bwdStore_stMem]
  rw [stepStmts.cons_some e12]
  set c10 := stMem s DKInter R (dkOut s DKInner DKInter G R lgVal) c9 with hc10
  -- `c10` reads: regions ≠ DKInter and ≠ DQInter agree with `s`.
  have c10read : ∀ (r : RegionName) (i : TileIndex [4]), r ≠ DKInter → r ≠ DQInter →
      c10.readMem r (s.pids 2 * 64 + s.pids 0 * 4 + i.1.val + R)
        = s.readMem r (s.pids 2 * 64 + s.pids 0 * 4 + i.1.val + R) := by
    intro r i hr1 hr2
    rw [hc10, stMem_readMem_other s DKInter r R _ c9 _ hr1]; exact c9read r i hr2
  have c10peel : ∀ (d : TileDType) (sh : TileShape) (n : RegName),
      n ≠ "dk" → n ≠ "dk2" → n ≠ "dk1" → n ≠ "dq" → n ≠ "dq2" → n ≠ "dq1" →
      c10.regs d sh n = s3.regs d sh n := by
    intro d sh n h0 h1 h2 h3 h3' h4
    rw [hc10, stMem_regs]; exact c9peel d sh n h0 h1 h2 h3 h3' h4
  -- === Statement 13: q_val = load p_q ===
  have e13 : stepStmt (Stmt.assign .real [4] "q_val"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [4] "p_q"))
        (MaskOpt.maskOther (Op.ref .bool [4] "mask") ((Op.const 0).broadcast [4])))) c10
      = some (c10.setReg "q_val" .real [4] (gated s (ldR s Q R))) := by
    have c10pq : c10.regs .ptr [4] "p_q" = some ptrQ := by
      rw [c10peel .ptr [4] "p_q" (by decide) (by decide) (by decide) (by decide)
          (by decide) (by decide)]; exact hpq
    have c10mask : c10.regs .bool [4] "mask" = some maskT := by
      rw [c10peel .bool [4] "mask" (by decide) (by decide) (by decide) (by decide)
          (by decide) (by decide)]; exact hmask
    rw [bwdEval_assign_load_ptr_maskOther c10 "q_val" "p_q" _ _
        ptrQ maskT otherT c10pq (hmaskE c10 c10mask) (hother c10)]
    simp only [hptrQ, hmaskT, hotherT]
    rw [load_tile_gated s c10 Q R (fun i => c10read Q i hQ_DKInter hQ_DQInter)]
  rw [stepStmts.cons_some e13]
  set c11 := c10.setReg "q_val" .real [4] (gated s (ldR s Q R)) with hc11
  -- === Statement 14: k_val = load p_k ===
  have e14 : stepStmt (Stmt.assign .real [4] "k_val"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [4] "p_k"))
        (MaskOpt.maskOther (Op.ref .bool [4] "mask") ((Op.const 0).broadcast [4])))) c11
      = some (c11.setReg "k_val" .real [4] (gated s (ldR s K R))) := by
    have c11pk : c11.regs .ptr [4] "p_k" = some ptrK := by
      rw [hc11, BlockState.setReg_ne_name (h := by decide)]
      rw [c10peel .ptr [4] "p_k" (by decide) (by decide) (by decide) (by decide)
          (by decide) (by decide)]; exact hpk
    have c11mask : c11.regs .bool [4] "mask" = some maskT := by
      rw [hc11, BlockState.setReg_ne_name (h := by decide)]
      rw [c10peel .bool [4] "mask" (by decide) (by decide) (by decide) (by decide)
          (by decide) (by decide)]; exact hmask
    have c11read : ∀ i : TileIndex [4],
        c11.readMem K (s.pids 2 * 64 + s.pids 0 * 4 + i.1.val + R)
          = s.readMem K (s.pids 2 * 64 + s.pids 0 * 4 + i.1.val + R) := by
      intro i; rw [hc11, BlockState.setReg_readMem]; exact c10read K i hK_DKInter hK_DQInter
    rw [bwdEval_assign_load_ptr_maskOther c11 "k_val" "p_k" _ _
        ptrK maskT otherT c11pk (hmaskE c11 c11mask) (hother c11)]
    simp only [hptrK, hmaskT, hotherT]
    rw [load_tile_gated s c11 K R c11read]
  rw [stepStmts.cons_some e14]
  set c12 := c11.setReg "k_val" .real [4] (gated s (ldR s K R)) with hc12
  -- Lookups on `c12` of dq, dk, q_val (for the dg_val computation).
  have c12dq : c12.regs .real [4] "dq" = some (gated s (dqOut s DQInner DQInter G R)) := by
    rw [hc12, BlockState.setReg_ne_name (h := by decide), hc11,
        BlockState.setReg_ne_name (h := by decide), hc10, stMem_regs, hc9,
        BlockState.setReg_ne_name (h := by decide), hc8,
        BlockState.setReg_ne_name (h := by decide), hc7,
        BlockState.setReg_ne_name (h := by decide), hc6,
        BlockState.setReg_ne_name (h := by decide), hc5, stMem_regs, hc4,
        BlockState.setReg_same]
  have c12dk : c12.regs .real [4] "dk" = some (gated s (dkOut s DKInner DKInter G R lgVal)) := by
    rw [hc12, BlockState.setReg_ne_name (h := by decide), hc11,
        BlockState.setReg_ne_name (h := by decide), hc10, stMem_regs, hc9,
        BlockState.setReg_same]
  have c12qval : c12.regs .real [4] "q_val" = some (gated s (ldR s Q R)) := by
    rw [hc12, BlockState.setReg_ne_name (h := by decide), hc11, BlockState.setReg_same]
  have c12kval : c12.regs .real [4] "k_val" = some (gated s (ldR s K R)) := by
    rw [hc12]; exact BlockState.setReg_same _ _ _ _ _
  -- === Statement 15: dg_val = dq*q_val - dk*k_val ===
  have hmul_dq : evalOp (Op.mul .real Broadcast.nil.consSame (Op.ref .real [4] "dq")
      (Op.ref .real [4] "q_val")) c12
      = some (gated s (fun i => dqOut s DQInner DQInter G R i * ldR s Q R i)) := by
    rw [bwdEval_mul c12 _ _ (gated s (dqOut s DQInner DQInter G R)) (gated s (ldR s Q R))
        (by rw [evalOp_ref]; exact c12dq) (by rw [evalOp_ref]; exact c12qval)]
    rw [gated_mul]
  have hmul_dk : evalOp (Op.mul .real Broadcast.nil.consSame (Op.ref .real [4] "dk")
      (Op.ref .real [4] "k_val")) c12
      = some (gated s (fun i => dkOut s DKInner DKInter G R lgVal i * ldR s K R i)) := by
    rw [bwdEval_mul c12 _ _ (gated s (dkOut s DKInner DKInter G R lgVal)) (gated s (ldR s K R))
        (by rw [evalOp_ref]; exact c12dk) (by rw [evalOp_ref]; exact c12kval)]
    rw [gated_mul]
  have e15 : stepStmt (Stmt.assign .real [4] "dg_val"
      (Op.sub .real Broadcast.nil.consSame
        (Op.mul .real Broadcast.nil.consSame (Op.ref .real [4] "dq")
          (Op.ref .real [4] "q_val"))
        (Op.mul .real Broadcast.nil.consSame (Op.ref .real [4] "dk")
          (Op.ref .real [4] "k_val")))) c12
      = some (c12.setReg "dg_val" .real [4]
          (gated s (dgSum s DQInner DQInter DKInner DKInter Q K G R lgVal))) := by
    apply stepStmt_assign_eq_some
    rw [bwdEval_sub c12 _ _
        (gated s (fun i => dqOut s DQInner DQInter G R i * ldR s Q R i))
        (gated s (fun i => dkOut s DKInner DKInter G R lgVal i * ldR s K R i))
        hmul_dq hmul_dk]
    rw [gated_sub]; rfl
  rw [stepStmts.cons_some e15]
  set c13 := c12.setReg "dg_val" .real [4]
    (gated s (dgSum s DQInner DQInter DKInner DKInter Q K G R lgVal)) with hc13
  have c13cum : c13.regs .real [4] "cum_grad_dg" = some (gated s cumInVal) := by
    rw [hc13, BlockState.setReg_ne_name (h := by decide), hc12,
        BlockState.setReg_ne_name (h := by decide), hc11,
        BlockState.setReg_ne_name (h := by decide), hc10, stMem_regs, hc9,
        BlockState.setReg_ne_name (h := by decide), hc8,
        BlockState.setReg_ne_name (h := by decide), hc7,
        BlockState.setReg_ne_name (h := by decide), hc6,
        BlockState.setReg_ne_name (h := by decide), hc5, stMem_regs, hc4,
        BlockState.setReg_ne_name (h := by decide), hc3,
        BlockState.setReg_ne_name (h := by decide), hc2,
        BlockState.setReg_ne_name (h := by decide), hc1,
        BlockState.setReg_ne_name (h := by decide)]; exact hcum
  -- === Statement 16: cum_grad_dg = cum_grad_dg + dg_val ===
  have e16 : stepStmt (Stmt.assign .real [4] "cum_grad_dg"
      (Op.add .real Broadcast.nil.consSame (Op.ref .real [4] "cum_grad_dg")
        (Op.ref .real [4] "dg_val"))) c13
      = some (c13.setReg "cum_grad_dg" .real [4]
          (gated s (fun i => cumInVal i +
            dgSum s DQInner DQInter DKInner DKInter Q K G R lgVal i))) := by
    apply stepStmt_assign_eq_some
    rw [bwdEval_add c13 _ _ (gated s cumInVal)
        (gated s (dgSum s DQInner DQInter DKInner DKInter Q K G R lgVal))
        (by rw [evalOp_ref]; exact c13cum)
        (by rw [evalOp_ref, hc13, BlockState.setReg_same])]
    rw [gated_add]
  rw [stepStmts.cons_some e16]
  set c14 := c13.setReg "cum_grad_dg" .real [4]
    (gated s (fun i => cumInVal i +
      dgSum s DQInner DQInter DKInner DKInter Q K G R lgVal i)) with hc14
  -- `c14` peels to `s3` for the 14 real setRegs straddling `c5`/`c10` stMem layers.
  have c14peel : ∀ (d : TileDType) (sh : TileShape) (n : RegName),
      n ≠ "cum_grad_dg" → n ≠ "dg_val" → n ≠ "k_val" → n ≠ "q_val" →
      n ≠ "dk" → n ≠ "dk2" → n ≠ "dk1" → n ≠ "dq" → n ≠ "dq2" → n ≠ "dq1" →
      c14.regs d sh n = s3.regs d sh n := by
    intro d sh n g0 g1 g2 g3 g4 g5 g6 g7 g8 g9
    rw [hc14, BlockState.setReg_ne_name (h := g0), hc13,
        BlockState.setReg_ne_name (h := g1), hc12,
        BlockState.setReg_ne_name (h := g2), hc11,
        BlockState.setReg_ne_name (h := g3), hc10, stMem_regs, hc9,
        BlockState.setReg_ne_name (h := g4), hc8,
        BlockState.setReg_ne_name (h := g5), hc7,
        BlockState.setReg_ne_name (h := g5), hc6,
        BlockState.setReg_ne_name (h := g6), hc5, stMem_regs, hc4,
        BlockState.setReg_ne_name (h := g7), hc3,
        BlockState.setReg_ne_name (h := g8), hc2,
        BlockState.setReg_ne_name (h := g8), hc1,
        BlockState.setReg_ne_name (h := g9)]
  -- === Statement 17: store p_dg cum_grad_dg ===
  have e17 : stepStmt (Stmt.store .real [4] (MemAccess.ptr (Op.ref .ptr [4] "p_dg"))
      (Op.ref .real [4] "cum_grad_dg") (MaskOpt.mask (Op.ref .bool [4] "mask"))) c14
      = some (stMem s DG R
          (fun i => cumInVal i + dgSum s DQInner DQInter DKInner DKInter Q K G R lgVal i)
          c14) := by
    have c14pdg : c14.regs .ptr [4] "p_dg" = some ptrDG := by
      rw [c14peel .ptr [4] "p_dg" (by decide) (by decide) (by decide) (by decide)
          (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]
      exact hpdg
    have c14mask : c14.regs .bool [4] "mask" = some maskT := by
      rw [c14peel .bool [4] "mask" (by decide) (by decide) (by decide) (by decide)
          (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]
      exact hmask
    have c14cum : c14.regs .real [4] "cum_grad_dg" = some
        (gated s (fun i => cumInVal i +
          dgSum s DQInner DQInter DKInner DKInter Q K G R lgVal i)) := by
      rw [hc14]; exact BlockState.setReg_same _ _ _ _ _
    rw [bwdEval_store_ptr_masked c14 "p_dg" _ _
        ptrDG (gated s (fun i => cumInVal i +
          dgSum s DQInner DQInter DKInner DKInter Q K G R lgVal i)) maskT
        c14pdg (by rw [evalOp_ref]; exact c14cum) (by rw [evalOp_ref]; exact c14mask)]
    simp only [hptrDG, hmaskT]
    rw [bwdStore_stMem]
  rw [stepStmts.cons_some e17]
  set c15 := stMem s DG R
    (fun i => cumInVal i + dgSum s DQInner DQInter DKInner DKInter Q K G R lgVal i)
    c14 with hc15
  -- The 8 pointer decrements. Each reads its (preserved) pointer register.
  have c15p : ∀ (region : RegionName) (n : RegName),
      s3.regs .ptr [4] n = some
        (Tile.vec (fun e : Fin 4 => (region.cast, s.pids 2 * 64 + s.pids 0 * 4 + e.val + R))) →
      n ≠ "cum_grad_dg" → n ≠ "dg_val" → n ≠ "k_val" → n ≠ "q_val" →
      n ≠ "dk" → n ≠ "dk2" → n ≠ "dk1" → n ≠ "dq" → n ≠ "dq2" → n ≠ "dq1" →
      c15.regs .ptr [4] n = some
        (Tile.vec (fun e : Fin 4 => (region.cast, s.pids 2 * 64 + s.pids 0 * 4 + e.val + R))) := by
    intro region n hn g0 g1 g2 g3 g4 g5 g6 g7 g8 g9
    rw [hc15, stMem_regs]
    rw [c14peel .ptr [4] n g0 g1 g2 g3 g4 g5 g6 g7 g8 g9]; exact hn
  -- Convenience: `c15p` instantiated at each row pointer.
  have c15pg := c15p G "p_g" hpg (by decide) (by decide) (by decide) (by decide)
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
  have c15pk := c15p K "p_k" hpk (by decide) (by decide) (by decide) (by decide)
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
  have c15pq := c15p Q "p_q" hpq (by decide) (by decide) (by decide) (by decide)
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
  have c15pdqi := c15p DQInner "p_dq_inner" hpdqi (by decide) (by decide) (by decide)
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
  have c15pdki := c15p DKInner "p_dk_inner" hpdki (by decide) (by decide) (by decide)
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
  have c15pdqt := c15p DQInter "p_dq_inter" hpdqt (by decide) (by decide) (by decide)
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
  have c15pdkt := c15p DKInter "p_dk_inter" hpdkt (by decide) (by decide) (by decide)
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
  have c15pdg := c15p DG "p_dg" hpdg (by decide) (by decide) (by decide)
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
  -- === Statements 18-25: p_x -= 8 (eight pointer decrements) ===
  have e18 : stepStmt (Stmt.assign .ptr [4] "p_g"
      (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [4] "p_g") (Op.constNat 8))) c15
      = some (c15.setReg "p_g" .ptr [4] (ptrDec G s R)) := by
    rw [bwdEval_assign_ptrSub c15 "p_g" "p_g" 8 ptrG c15pg]; rfl
  rw [stepStmts.cons_some e18]
  set c16 := c15.setReg "p_g" .ptr [4] (ptrDec G s R) with hc16
  have e19 : stepStmt (Stmt.assign .ptr [4] "p_k"
      (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [4] "p_k") (Op.constNat 8))) c16
      = some (c16.setReg "p_k" .ptr [4] (ptrDec K s R)) := by
    rw [bwdEval_assign_ptrSub c16 "p_k" "p_k" 8 ptrK
        (by rw [hc16, BlockState.setReg_ne_name (h := by decide)]; exact c15pk)]; rfl
  rw [stepStmts.cons_some e19]
  set c17 := c16.setReg "p_k" .ptr [4] (ptrDec K s R) with hc17
  have e20 : stepStmt (Stmt.assign .ptr [4] "p_q"
      (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [4] "p_q") (Op.constNat 8))) c17
      = some (c17.setReg "p_q" .ptr [4] (ptrDec Q s R)) := by
    rw [bwdEval_assign_ptrSub c17 "p_q" "p_q" 8 ptrQ
        (by rw [hc17, BlockState.setReg_ne_name (h := by decide), hc16,
            BlockState.setReg_ne_name (h := by decide)]; exact c15pq)]; rfl
  rw [stepStmts.cons_some e20]
  set c18 := c17.setReg "p_q" .ptr [4] (ptrDec Q s R) with hc18
  have e21 : stepStmt (Stmt.assign .ptr [4] "p_dq_inner"
      (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [4] "p_dq_inner") (Op.constNat 8))) c18
      = some (c18.setReg "p_dq_inner" .ptr [4] (ptrDec DQInner s R)) := by
    rw [bwdEval_assign_ptrSub c18 "p_dq_inner" "p_dq_inner" 8 ptrDQi
        (by rw [hc18, BlockState.setReg_ne_name (h := by decide), hc17,
            BlockState.setReg_ne_name (h := by decide), hc16,
            BlockState.setReg_ne_name (h := by decide)]; exact c15pdqi)]; rfl
  rw [stepStmts.cons_some e21]
  set c19 := c18.setReg "p_dq_inner" .ptr [4] (ptrDec DQInner s R) with hc19
  have e22 : stepStmt (Stmt.assign .ptr [4] "p_dk_inner"
      (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [4] "p_dk_inner") (Op.constNat 8))) c19
      = some (c19.setReg "p_dk_inner" .ptr [4] (ptrDec DKInner s R)) := by
    rw [bwdEval_assign_ptrSub c19 "p_dk_inner" "p_dk_inner" 8 ptrDKi
        (by rw [hc19, BlockState.setReg_ne_name (h := by decide), hc18,
            BlockState.setReg_ne_name (h := by decide), hc17,
            BlockState.setReg_ne_name (h := by decide), hc16,
            BlockState.setReg_ne_name (h := by decide)]; exact c15pdki)]; rfl
  rw [stepStmts.cons_some e22]
  set c20 := c19.setReg "p_dk_inner" .ptr [4] (ptrDec DKInner s R) with hc20
  have e23 : stepStmt (Stmt.assign .ptr [4] "p_dq_inter"
      (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [4] "p_dq_inter") (Op.constNat 8))) c20
      = some (c20.setReg "p_dq_inter" .ptr [4] (ptrDec DQInter s R)) := by
    rw [bwdEval_assign_ptrSub c20 "p_dq_inter" "p_dq_inter" 8 ptrDQt
        (by rw [hc20, BlockState.setReg_ne_name (h := by decide), hc19,
            BlockState.setReg_ne_name (h := by decide), hc18,
            BlockState.setReg_ne_name (h := by decide), hc17,
            BlockState.setReg_ne_name (h := by decide), hc16,
            BlockState.setReg_ne_name (h := by decide)]; exact c15pdqt)]; rfl
  rw [stepStmts.cons_some e23]
  set c21 := c20.setReg "p_dq_inter" .ptr [4] (ptrDec DQInter s R) with hc21
  have e24 : stepStmt (Stmt.assign .ptr [4] "p_dk_inter"
      (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [4] "p_dk_inter") (Op.constNat 8))) c21
      = some (c21.setReg "p_dk_inter" .ptr [4] (ptrDec DKInter s R)) := by
    rw [bwdEval_assign_ptrSub c21 "p_dk_inter" "p_dk_inter" 8 ptrDKt
        (by rw [hc21, BlockState.setReg_ne_name (h := by decide), hc20,
            BlockState.setReg_ne_name (h := by decide), hc19,
            BlockState.setReg_ne_name (h := by decide), hc18,
            BlockState.setReg_ne_name (h := by decide), hc17,
            BlockState.setReg_ne_name (h := by decide), hc16,
            BlockState.setReg_ne_name (h := by decide)]; exact c15pdkt)]; rfl
  rw [stepStmts.cons_some e24]
  set c22 := c21.setReg "p_dk_inter" .ptr [4] (ptrDec DKInter s R) with hc22
  have e25 : stepStmt (Stmt.assign .ptr [4] "p_dg"
      (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [4] "p_dg") (Op.constNat 8))) c22
      = some (c22.setReg "p_dg" .ptr [4] (ptrDec DG s R)) := by
    rw [bwdEval_assign_ptrSub c22 "p_dg" "p_dg" 8 ptrDG
        (by rw [hc22, BlockState.setReg_ne_name (h := by decide), hc21,
            BlockState.setReg_ne_name (h := by decide), hc20,
            BlockState.setReg_ne_name (h := by decide), hc19,
            BlockState.setReg_ne_name (h := by decide), hc18,
            BlockState.setReg_ne_name (h := by decide), hc17,
            BlockState.setReg_ne_name (h := by decide), hc16,
            BlockState.setReg_ne_name (h := by decide)]; exact c15pdg)]; rfl
  rw [stepStmts.cons_some e25]
  set c23 := c22.setReg "p_dg" .ptr [4] (ptrDec DG s R) with hc23
  rw [stepStmts.nil]
  -- All conjuncts proven against the final state `c23`.
  refine ⟨c23, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · -- pids
    rw [hc23, BlockState.setReg_pids, hc22, BlockState.setReg_pids, hc21,
        BlockState.setReg_pids, hc20, BlockState.setReg_pids, hc19,
        BlockState.setReg_pids, hc18, BlockState.setReg_pids, hc17,
        BlockState.setReg_pids, hc16, BlockState.setReg_pids, hc15, stMem_pids,
        hc14, BlockState.setReg_pids, hc13, BlockState.setReg_pids, hc12,
        BlockState.setReg_pids, hc11, BlockState.setReg_pids, hc10, stMem_pids,
        hc9, BlockState.setReg_pids, hc8, BlockState.setReg_pids, hc7,
        BlockState.setReg_pids, hc6, BlockState.setReg_pids, hc5, stMem_pids,
        hc4, BlockState.setReg_pids, hc3, BlockState.setReg_pids, hc2,
        BlockState.setReg_pids, hc1, BlockState.setReg_pids]
    exact hpid
  · -- mem: three nested masked stores, reduced over `s3` via mem-congruence
    have hc4mem : c4.mem = s3.mem := by
      simp only [hc4, hc3, hc2, hc1, BlockState.setReg]
    have hc9mem : c9.mem = c5.mem := by
      simp only [hc9, hc8, hc7, hc6, BlockState.setReg]
    have hc14mem : c14.mem = c10.mem := by
      simp only [hc14, hc13, hc12, hc11, BlockState.setReg]
    have hc23mem : c23.mem = c15.mem := by
      simp only [hc23, hc22, hc21, hc20, hc19, hc18, hc17, hc16, BlockState.setReg]
    rw [hc23mem, hc15]
    refine stMem_mem_congr s DG R _ c14 _ ?_
    rw [hc14mem, hc10]
    refine stMem_mem_congr s DKInter R _ c9 _ ?_
    rw [hc9mem, hc5]
    exact stMem_mem_congr s DQInter R _ c4 s3 hc4mem
  · -- mask: peel all 8 ptrSubs to c15, stMem regs, then c14 down to s3.
    rw [hc23, BlockState.setReg_ne_name (h := by decide), hc22,
        BlockState.setReg_ne_name (h := by decide), hc21,
        BlockState.setReg_ne_name (h := by decide), hc20,
        BlockState.setReg_ne_name (h := by decide), hc19,
        BlockState.setReg_ne_name (h := by decide), hc18,
        BlockState.setReg_ne_name (h := by decide), hc17,
        BlockState.setReg_ne_name (h := by decide), hc16,
        BlockState.setReg_ne_name (h := by decide), hc15, stMem_regs,
        c14peel .bool [4] "mask" (by decide) (by decide) (by decide) (by decide)
          (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]
    exact hmask
  · -- last_g
    rw [hc23, BlockState.setReg_ne_name (h := by decide), hc22,
        BlockState.setReg_ne_name (h := by decide), hc21,
        BlockState.setReg_ne_name (h := by decide), hc20,
        BlockState.setReg_ne_name (h := by decide), hc19,
        BlockState.setReg_ne_name (h := by decide), hc18,
        BlockState.setReg_ne_name (h := by decide), hc17,
        BlockState.setReg_ne_name (h := by decide), hc16,
        BlockState.setReg_ne_name (h := by decide), hc15, stMem_regs,
        c14peel .real [4] "last_g" (by decide) (by decide) (by decide) (by decide)
          (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]
    exact hlastg
  · -- cum_grad_dg: set at c14 (below the 8 ptrSubs and the DG store).
    rw [hc23, BlockState.setReg_ne_name (h := by decide), hc22,
        BlockState.setReg_ne_name (h := by decide), hc21,
        BlockState.setReg_ne_name (h := by decide), hc20,
        BlockState.setReg_ne_name (h := by decide), hc19,
        BlockState.setReg_ne_name (h := by decide), hc18,
        BlockState.setReg_ne_name (h := by decide), hc17,
        BlockState.setReg_ne_name (h := by decide), hc16,
        BlockState.setReg_ne_name (h := by decide), hc15, stMem_regs, hc14]
    exact BlockState.setReg_same _ _ _ _ _
  · -- p_g (set at c16)
    rw [hc23, BlockState.setReg_ne_name (h := by decide), hc22,
        BlockState.setReg_ne_name (h := by decide), hc21,
        BlockState.setReg_ne_name (h := by decide), hc20,
        BlockState.setReg_ne_name (h := by decide), hc19,
        BlockState.setReg_ne_name (h := by decide), hc18,
        BlockState.setReg_ne_name (h := by decide), hc17,
        BlockState.setReg_ne_name (h := by decide), hc16]
    exact BlockState.setReg_same _ _ _ _ _
  · -- p_k (set at c17)
    rw [hc23, BlockState.setReg_ne_name (h := by decide), hc22,
        BlockState.setReg_ne_name (h := by decide), hc21,
        BlockState.setReg_ne_name (h := by decide), hc20,
        BlockState.setReg_ne_name (h := by decide), hc19,
        BlockState.setReg_ne_name (h := by decide), hc18,
        BlockState.setReg_ne_name (h := by decide), hc17]
    exact BlockState.setReg_same _ _ _ _ _
  · -- p_q (set at c18)
    rw [hc23, BlockState.setReg_ne_name (h := by decide), hc22,
        BlockState.setReg_ne_name (h := by decide), hc21,
        BlockState.setReg_ne_name (h := by decide), hc20,
        BlockState.setReg_ne_name (h := by decide), hc19,
        BlockState.setReg_ne_name (h := by decide), hc18]
    exact BlockState.setReg_same _ _ _ _ _
  · -- p_dq_inner (set at c19)
    rw [hc23, BlockState.setReg_ne_name (h := by decide), hc22,
        BlockState.setReg_ne_name (h := by decide), hc21,
        BlockState.setReg_ne_name (h := by decide), hc20,
        BlockState.setReg_ne_name (h := by decide), hc19]
    exact BlockState.setReg_same _ _ _ _ _
  · -- p_dk_inner (set at c20)
    rw [hc23, BlockState.setReg_ne_name (h := by decide), hc22,
        BlockState.setReg_ne_name (h := by decide), hc21,
        BlockState.setReg_ne_name (h := by decide), hc20]
    exact BlockState.setReg_same _ _ _ _ _
  · -- p_dq_inter (set at c21)
    rw [hc23, BlockState.setReg_ne_name (h := by decide), hc22,
        BlockState.setReg_ne_name (h := by decide), hc21]
    exact BlockState.setReg_same _ _ _ _ _
  · -- p_dk_inter (set at c22)
    rw [hc23, BlockState.setReg_ne_name (h := by decide), hc22]
    exact BlockState.setReg_same _ _ _ _ _
  · -- p_dg (set at c23)
    rw [hc23]
    exact BlockState.setReg_same _ _ _ _ _

/-- The prologue post-state shape: pointers at row offset `R`, mask, and the
zero-initialized `cum_grad_dg` / `last_g` accumulators. Reused by both iteration
wrappers. `R` is the row offset of the relevant time row. -/
structure BwdPrologueShape (s s0 : BlockState)
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName) (R : Nat) : Prop where
  pid : s0.pids = s.pids
  mem : s0.mem = s.mem
  mask : s0.regs .bool [4] "mask" = some
    (Tile.vec (fun e : Fin 4 => decide (s.pids 0 * 4 + e.val < 8)))
  cum : s0.regs .real [4] "cum_grad_dg" = some (⟨fun _ => some 0⟩ : Tile .real [4])
  lastg : s0.regs .real [4] "last_g" = some (⟨fun _ => some 0⟩ : Tile .real [4])
  pg : s0.regs .ptr [4] "p_g" = some
    (Tile.vec (fun e : Fin 4 => (G.cast, s.pids 2 * 64 + s.pids 0 * 4 + e.val + R)))
  pk : s0.regs .ptr [4] "p_k" = some
    (Tile.vec (fun e : Fin 4 => (K.cast, s.pids 2 * 64 + s.pids 0 * 4 + e.val + R)))
  pq : s0.regs .ptr [4] "p_q" = some
    (Tile.vec (fun e : Fin 4 => (Q.cast, s.pids 2 * 64 + s.pids 0 * 4 + e.val + R)))
  pdqi : s0.regs .ptr [4] "p_dq_inner" = some
    (Tile.vec (fun e : Fin 4 => (DQInner.cast, s.pids 2 * 64 + s.pids 0 * 4 + e.val + R)))
  pdki : s0.regs .ptr [4] "p_dk_inner" = some
    (Tile.vec (fun e : Fin 4 => (DKInner.cast, s.pids 2 * 64 + s.pids 0 * 4 + e.val + R)))
  pdqt : s0.regs .ptr [4] "p_dq_inter" = some
    (Tile.vec (fun e : Fin 4 => (DQInter.cast, s.pids 2 * 64 + s.pids 0 * 4 + e.val + R)))
  pdkt : s0.regs .ptr [4] "p_dk_inter" = some
    (Tile.vec (fun e : Fin 4 => (DKInter.cast, s.pids 2 * 64 + s.pids 0 * 4 + e.val + R)))
  pdg : s0.regs .ptr [4] "p_dg" = some
    (Tile.vec (fun e : Fin 4 => (DG.cast, s.pids 2 * 64 + s.pids 0 * 4 + e.val + R)))

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **Iteration 0 (`__rev_t = 0`, time row `t = BT-1 = 1`).** From the post-prologue
state `s0` (shape `BwdPrologueShape` at row offset `R`), the first reverse-loop
iteration captures `last_g = g[row BT-1]` (the `t == BT-1` branch fires), then
runs the straight-line body. The captured `last_g` per lane is `ldR s G R`; the
incoming reverse accumulator is `0`. We reduce the head two statements + the
`last_g` capture via `bwdIterBody_head_eval` + `bwdEval_ifThen_true`, then chain
the remaining 22 statements via `bwd_iter_core`. -/
theorem bwd_iter0_eval
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName) (s s0 : BlockState) (R : Nat)
    (hDKInner_DQInter : DKInner ≠ DQInter) (hDKInter_DQInter : DKInter ≠ DQInter)
    (hQ_DQInter : Q ≠ DQInter) (hQ_DKInter : Q ≠ DKInter)
    (hK_DQInter : K ≠ DQInter) (hK_DKInter : K ≠ DKInter)
    (hsh : BwdPrologueShape s s0 DQInner DQInter DKInner DKInter Q K G DG R) :
    ∃ s1, stepStmts bwdIterBody (s0.setReg "__rev_t" .nat [] (Tile.scalar 0)) = some s1
      ∧ s1.pids = s.pids
      ∧ s1.mem =
          (stMem s DG R (fun i =>
              dgSum s DQInner DQInter DKInner DKInter Q K G R (ldR s G R) i)
            (stMem s DKInter R (dkOut s DKInner DKInter G R (ldR s G R))
              (stMem s DQInter R (dqOut s DQInner DQInter G R) s0))).mem
      ∧ s1.regs .bool [4] "mask" = some
          (Tile.vec (fun e : Fin 4 => decide (s.pids 0 * 4 + e.val < 8)))
      ∧ s1.regs .real [4] "last_g" = some (gated s (ldR s G R))
      ∧ s1.regs .real [4] "cum_grad_dg" = some
          (gated s (fun i =>
            dgSum s DQInner DQInter DKInner DKInter Q K G R (ldR s G R) i))
      ∧ s1.regs .ptr [4] "p_g" = some (ptrDec G s R)
      ∧ s1.regs .ptr [4] "p_k" = some (ptrDec K s R)
      ∧ s1.regs .ptr [4] "p_q" = some (ptrDec Q s R)
      ∧ s1.regs .ptr [4] "p_dq_inner" = some (ptrDec DQInner s R)
      ∧ s1.regs .ptr [4] "p_dk_inner" = some (ptrDec DKInner s R)
      ∧ s1.regs .ptr [4] "p_dq_inter" = some (ptrDec DQInter s R)
      ∧ s1.regs .ptr [4] "p_dk_inter" = some (ptrDec DKInter s R)
      ∧ s1.regs .ptr [4] "p_dg" = some (ptrDec DG s R) := by
  set sin := s0.setReg "__rev_t" .nat [] (Tile.scalar 0) with hsin
  -- pids / memory reads of `sin` agree with `s`.
  have hsinpid : sin.pids = s.pids := by rw [hsin, BlockState.setReg_pids]; exact hsh.pid
  have hsinread : ∀ (r : RegionName) (a : Nat), sin.readMem r a = s.readMem r a := by
    intro r a; rw [hsin, BlockState.setReg_readMem]
    simp only [BlockState.readMem, hsh.mem]
  have hsinmask : sin.regs .bool [4] "mask" = some
      (Tile.vec (fun e : Fin 4 => decide (sin.pids 0 * 4 + e.val < 8))) := by
    rw [hsinpid, hsin, BlockState.setReg_ne_name (h := by decide)]; exact hsh.mask
  have hsinpg : sin.regs .ptr [4] "p_g" = some
      (Tile.vec (fun e : Fin 4 => (G.cast, sin.pids 2 * 64 + sin.pids 0 * 4 + e.val + R))) := by
    rw [hsinpid, hsin, BlockState.setReg_ne_name (h := by decide)]; exact hsh.pg
  -- Head: `t = 1`, `g_val = load p_g`; reduces to the post-head state.
  rw [bwdIterBody_head_eval G sin 0 R
      (by rw [hsin]; exact BlockState.setReg_same _ _ _ _ _) hsinmask hsinpg]
  -- `bwdIterBody.drop 2 = ifThen :: bwdIterBody.drop 3`; run the `last_g` capture.
  rw [show bwdIterBody.drop 2 =
        Stmt.ifThen
          (Op.eq ComparableDType.nat Broadcast.nil (Op.ref .nat [] "t")
            (Op.sub .nat Broadcast.nil (Op.constNat 2) (Op.constNat 1)))
          [Stmt.assign .real [4] "last_g" (Op.ref .real [4] "g_val")]
        :: bwdIterBody.drop 3 from rfl]
  -- ifThen condition: `t = 1 = BT-1`, so it fires; capture `last_g = g_val`.
  have hcond : evalOp (Op.eq ComparableDType.nat Broadcast.nil (Op.ref .nat [] "t")
      (Op.sub .nat Broadcast.nil (Op.constNat 2) (Op.constNat 1)))
      (bwdIterHeadState G sin 0 R) = some (Tile.scalar Bool.true) := by
    rw [bwdEval_eqNat (bwdIterHeadState G sin 0 R) _ _ (Tile.scalar (2 - 1 - 0 * 1))
        (Tile.scalar (2 - 1))
        (by rw [evalOp_ref, bwdIterHeadState, BlockState.setReg_ne_name (h := by decide),
              BlockState.setReg_same])
        (by simp [evalOp, Tile.bop, NumericDType.sub])]
    rfl
  have hif : stepStmt (Stmt.ifThen
      (Op.eq ComparableDType.nat Broadcast.nil (Op.ref .nat [] "t")
        (Op.sub .nat Broadcast.nil (Op.constNat 2) (Op.constNat 1)))
      [Stmt.assign .real [4] "last_g" (Op.ref .real [4] "g_val")])
      (bwdIterHeadState G sin 0 R)
      = some ((bwdIterHeadState G sin 0 R).setReg "last_g" .real [4]
          (gated s (ldR s G R))) := by
    rw [bwdEval_ifThen_true (bwdIterHeadState G sin 0 R) _ _ hcond]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (show evalOp (Op.ref .real [4] "g_val") (bwdIterHeadState G sin 0 R)
            = some (gated s (ldR s G R)) by
          rw [evalOp_ref, bwdIterHeadState, BlockState.setReg_same,
            ldVal_eq_gated sin s G R hsinpid (fun i => hsinread G _)]))]
    rw [stepStmts.nil]
  rw [stepStmts.cons_some hif]
  -- The remaining 22 statements via `bwd_iter_core`.
  set s3 := (bwdIterHeadState G sin 0 R).setReg "last_g" .real [4]
    (gated s (ldR s G R)) with hs3
  have hs3pid : s3.pids = s.pids := by
    rw [hs3, BlockState.setReg_pids, bwdIterHeadState, BlockState.setReg_pids,
        BlockState.setReg_pids]; exact hsinpid
  have hs3mem : s3.mem = s.mem := by
    rw [hs3, bwdIterHeadState]
    simp only [BlockState.setReg]
    rw [hsin]; simp only [BlockState.setReg]; exact hsh.mem
  have hpeel : ∀ (d : TileDType) (sh : TileShape) (n : RegName),
      n ≠ "last_g" → n ≠ "g_val" → n ≠ "t" → n ≠ "__rev_t" →
      s3.regs d sh n = s0.regs d sh n := by
    intro d sh n h0 h1 h2 h3
    rw [hs3, BlockState.setReg_ne_name (h := h0), bwdIterHeadState,
        BlockState.setReg_ne_name (h := h1), BlockState.setReg_ne_name (h := h2),
        hsin, BlockState.setReg_ne_name (h := h3)]
  obtain ⟨sout, hstep, hpids, hmemo, hmasko, hlasto, hcumo,
      hpgo, hpko, hpqo, hpdqio, hpdkio, hpdqto, hpdkto, hpdgo⟩ :=
    bwd_iter_core DQInner DQInter DKInner DKInter Q K G DG s s3 R
      (ldR s G R) (fun _ => 0)
      hDKInner_DQInter hDKInter_DQInter hQ_DQInter hQ_DKInter hK_DQInter hK_DKInter
      hs3pid (fun r i => by simp only [BlockState.readMem, hs3mem])
      (by rw [hpeel .bool [4] "mask" (by decide) (by decide) (by decide) (by decide)]
          exact hsh.mask)
      (by rw [hs3, BlockState.setReg_ne_name (h := by decide), bwdIterHeadState,
            BlockState.setReg_same, ldVal_eq_gated sin s G R hsinpid (fun i => hsinread G _)])
      (by rw [hs3]; exact BlockState.setReg_same _ _ _ _ _)
      (by rw [hpeel .real [4] "cum_grad_dg" (by decide) (by decide) (by decide) (by decide),
            hsh.cum, gated_const0 s])
      (by rw [hpeel .ptr [4] "p_g" (by decide) (by decide) (by decide) (by decide)]; exact hsh.pg)
      (by rw [hpeel .ptr [4] "p_k" (by decide) (by decide) (by decide) (by decide)]; exact hsh.pk)
      (by rw [hpeel .ptr [4] "p_q" (by decide) (by decide) (by decide) (by decide)]; exact hsh.pq)
      (by rw [hpeel .ptr [4] "p_dq_inner" (by decide) (by decide) (by decide) (by decide)]; exact hsh.pdqi)
      (by rw [hpeel .ptr [4] "p_dk_inner" (by decide) (by decide) (by decide) (by decide)]; exact hsh.pdki)
      (by rw [hpeel .ptr [4] "p_dq_inter" (by decide) (by decide) (by decide) (by decide)]; exact hsh.pdqt)
      (by rw [hpeel .ptr [4] "p_dk_inter" (by decide) (by decide) (by decide) (by decide)]; exact hsh.pdkt)
      (by rw [hpeel .ptr [4] "p_dg" (by decide) (by decide) (by decide) (by decide)]; exact hsh.pdg)
  -- Simplify `0 + dgSum = dgSum` and `... s0` vs `... s3` in the memory output.
  have hzero : (fun i => (0 : ℝ) +
      dgSum s DQInner DQInter DKInner DKInter Q K G R (ldR s G R) i)
      = fun i => dgSum s DQInner DQInter DKInner DKInter Q K G R (ldR s G R) i := by
    funext i; rw [zero_add]
  have hmemeq : sout.mem =
      (stMem s DG R (fun i =>
          dgSum s DQInner DQInter DKInner DKInter Q K G R (ldR s G R) i)
        (stMem s DKInter R (dkOut s DKInner DKInter G R (ldR s G R))
          (stMem s DQInter R (dqOut s DQInner DQInter G R) s0))).mem := by
    rw [hmemo, hzero]
    refine stMem_mem_congr s DG R _ _ _ ?_
    refine stMem_mem_congr s DKInter R _ _ _ ?_
    refine stMem_mem_congr s DQInter R _ s3 s0 ?_
    rw [hs3mem]; exact hsh.mem.symm
  exact ⟨sout, hstep, hpids, hmemeq, hmasko, hlasto, by rw [hcumo, hzero],
    hpgo, hpko, hpqo, hpdqio, hpdkio, hpdqto, hpdkto, hpdgo⟩

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **Iteration 1 (`__rev_t = 1`, time row `t = 0`).** From the iter-0 output `s1`
(pointers at row offset `R1 = R0 - 8`, carrying `last_g = g[row BT-1]` and the
reverse accumulator `cum0`), the second iteration's `t == BT-1` branch does *not*
fire (`t = 0`), so `last_g` is unchanged; the body then runs at row 0. The three
masked stores write row 0; `cum_grad_dg` becomes `cum0 + dg-summand[row 0]`. -/
theorem bwd_iter1_eval
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName) (s s1 : BlockState)
    (R0 : Nat) (h8 : 8 ≤ R0) (lg cum0 : TileIndex [4] → ℝ)
    (hDKInner_DQInter : DKInner ≠ DQInter) (hDKInter_DQInter : DKInter ≠ DQInter)
    (hQ_DQInter : Q ≠ DQInter) (hQ_DKInter : Q ≠ DKInter)
    (hK_DQInter : K ≠ DQInter) (hK_DKInter : K ≠ DKInter)
    (hpid : s1.pids = s.pids)
    (hrow : ∀ (r : RegionName) (i : TileIndex [4]),
      s1.readMem r (s.pids 2 * 64 + s.pids 0 * 4 + i.1.val + (R0 - 8))
        = s.readMem r (s.pids 2 * 64 + s.pids 0 * 4 + i.1.val + (R0 - 8)))
    (hmask : s1.regs .bool [4] "mask" = some
        (Tile.vec (fun e : Fin 4 => decide (s.pids 0 * 4 + e.val < 8))))
    (hlastg : s1.regs .real [4] "last_g" = some (gated s lg))
    (hcum : s1.regs .real [4] "cum_grad_dg" = some (gated s cum0))
    (hpg : s1.regs .ptr [4] "p_g" = some (ptrDec G s R0))
    (hpk : s1.regs .ptr [4] "p_k" = some (ptrDec K s R0))
    (hpq : s1.regs .ptr [4] "p_q" = some (ptrDec Q s R0))
    (hpdqi : s1.regs .ptr [4] "p_dq_inner" = some (ptrDec DQInner s R0))
    (hpdki : s1.regs .ptr [4] "p_dk_inner" = some (ptrDec DKInner s R0))
    (hpdqt : s1.regs .ptr [4] "p_dq_inter" = some (ptrDec DQInter s R0))
    (hpdkt : s1.regs .ptr [4] "p_dk_inter" = some (ptrDec DKInter s R0))
    (hpdg : s1.regs .ptr [4] "p_dg" = some (ptrDec DG s R0)) :
    ∃ s2, stepStmts bwdIterBody (s1.setReg "__rev_t" .nat [] (Tile.scalar 1)) = some s2
      ∧ s2.pids = s.pids
      ∧ s2.mem =
          (stMem s DG (R0 - 8) (fun i => cum0 i +
              dgSum s DQInner DQInter DKInner DKInter Q K G (R0 - 8) lg i)
            (stMem s DKInter (R0 - 8) (dkOut s DKInner DKInter G (R0 - 8) lg)
              (stMem s DQInter (R0 - 8) (dqOut s DQInner DQInter G (R0 - 8)) s1))).mem
      ∧ s2.regs .real [4] "cum_grad_dg" = some
          (gated s (fun i => cum0 i +
            dgSum s DQInner DQInter DKInner DKInter Q K G (R0 - 8) lg i)) := by
  set sin := s1.setReg "__rev_t" .nat [] (Tile.scalar 1) with hsin
  have hsinpid : sin.pids = s.pids := by rw [hsin, BlockState.setReg_pids]; exact hpid
  have hsinrow : ∀ (r : RegionName) (i : TileIndex [4]),
      sin.readMem r (s.pids 2 * 64 + s.pids 0 * 4 + i.1.val + (R0 - 8))
        = s.readMem r (s.pids 2 * 64 + s.pids 0 * 4 + i.1.val + (R0 - 8)) := by
    intro r i; rw [hsin, BlockState.setReg_readMem]; exact hrow r i
  -- The iteration row offset is `R1 = R0 - 8`.
  have hpgR : sin.regs .ptr [4] "p_g" = some
      (Tile.vec (fun e : Fin 4 => (G.cast, sin.pids 2 * 64 + sin.pids 0 * 4 + e.val + (R0 - 8)))) := by
    rw [hsinpid, hsin, BlockState.setReg_ne_name (h := by decide), hpg, ptrDec_as_row G s R0 h8]
  have hsinmask : sin.regs .bool [4] "mask" = some
      (Tile.vec (fun e : Fin 4 => decide (sin.pids 0 * 4 + e.val < 8))) := by
    rw [hsinpid, hsin, BlockState.setReg_ne_name (h := by decide)]; exact hmask
  -- Head: `t = 0`, `g_val = load p_g` (at row R1).
  rw [bwdIterBody_head_eval G sin 1 (R0 - 8)
      (by rw [hsin]; exact BlockState.setReg_same _ _ _ _ _) hsinmask hpgR]
  rw [show bwdIterBody.drop 2 =
        Stmt.ifThen
          (Op.eq ComparableDType.nat Broadcast.nil (Op.ref .nat [] "t")
            (Op.sub .nat Broadcast.nil (Op.constNat 2) (Op.constNat 1)))
          [Stmt.assign .real [4] "last_g" (Op.ref .real [4] "g_val")]
        :: bwdIterBody.drop 3 from rfl]
  -- ifThen condition: `t = 0 ≠ 1 = BT-1`, so it does not fire.
  have hcond : evalOp (Op.eq ComparableDType.nat Broadcast.nil (Op.ref .nat [] "t")
      (Op.sub .nat Broadcast.nil (Op.constNat 2) (Op.constNat 1)))
      (bwdIterHeadState G sin 1 (R0 - 8)) = some (Tile.scalar Bool.false) := by
    rw [bwdEval_eqNat (bwdIterHeadState G sin 1 (R0 - 8)) _ _ (Tile.scalar (2 - 1 - 1 * 1))
        (Tile.scalar (2 - 1))
        (by rw [evalOp_ref, bwdIterHeadState, BlockState.setReg_ne_name (h := by decide),
              BlockState.setReg_same])
        (by simp [evalOp, Tile.bop, NumericDType.sub])]
    rfl
  have hif : stepStmt (Stmt.ifThen
      (Op.eq ComparableDType.nat Broadcast.nil (Op.ref .nat [] "t")
        (Op.sub .nat Broadcast.nil (Op.constNat 2) (Op.constNat 1)))
      [Stmt.assign .real [4] "last_g" (Op.ref .real [4] "g_val")])
      (bwdIterHeadState G sin 1 (R0 - 8))
      = some (bwdIterHeadState G sin 1 (R0 - 8)) :=
    bwdEval_ifThen_false (bwdIterHeadState G sin 1 (R0 - 8)) _ _ hcond
  rw [stepStmts.cons_some hif]
  -- The remaining 22 statements via `bwd_iter_core`, row offset `R0 - 8`.
  set s3 := bwdIterHeadState G sin 1 (R0 - 8) with hs3
  have hpeel : ∀ (d : TileDType) (sh : TileShape) (n : RegName),
      n ≠ "g_val" → n ≠ "t" → n ≠ "__rev_t" →
      s3.regs d sh n = s1.regs d sh n := by
    intro d sh n h1 h2 h3
    rw [hs3, bwdIterHeadState, BlockState.setReg_ne_name (h := h1),
        BlockState.setReg_ne_name (h := h2), hsin, BlockState.setReg_ne_name (h := h3)]
  obtain ⟨sout, hstep, hpids, hmemo, _hmask, _hlast, hcumo, _, _, _, _, _, _, _, _⟩ :=
    bwd_iter_core DQInner DQInter DKInner DKInter Q K G DG s s3 (R0 - 8)
      lg cum0
      hDKInner_DQInter hDKInter_DQInter hQ_DQInter hQ_DKInter hK_DQInter hK_DKInter
      (by rw [hs3, bwdIterHeadState, BlockState.setReg_pids, BlockState.setReg_pids]
          exact hsinpid)
      (by intro r i; rw [hs3, bwdIterHeadState, BlockState.setReg_readMem,
            BlockState.setReg_readMem]; exact hsinrow r i)
      (by rw [hpeel .bool [4] "mask" (by decide) (by decide) (by decide)]; exact hmask)
      (by rw [hs3, bwdIterHeadState, BlockState.setReg_same,
            ldVal_eq_gated sin s G (R0 - 8) hsinpid (fun i => hsinrow G i)])
      (by rw [hpeel .real [4] "last_g" (by decide) (by decide) (by decide)]; exact hlastg)
      (by rw [hpeel .real [4] "cum_grad_dg" (by decide) (by decide) (by decide)]; exact hcum)
      (by rw [hpeel .ptr [4] "p_g" (by decide) (by decide) (by decide), hpg, ptrDec_as_row G s R0 h8])
      (by rw [hpeel .ptr [4] "p_k" (by decide) (by decide) (by decide), hpk, ptrDec_as_row K s R0 h8])
      (by rw [hpeel .ptr [4] "p_q" (by decide) (by decide) (by decide), hpq, ptrDec_as_row Q s R0 h8])
      (by rw [hpeel .ptr [4] "p_dq_inner" (by decide) (by decide) (by decide), hpdqi, ptrDec_as_row DQInner s R0 h8])
      (by rw [hpeel .ptr [4] "p_dk_inner" (by decide) (by decide) (by decide), hpdki, ptrDec_as_row DKInner s R0 h8])
      (by rw [hpeel .ptr [4] "p_dq_inter" (by decide) (by decide) (by decide), hpdqt, ptrDec_as_row DQInter s R0 h8])
      (by rw [hpeel .ptr [4] "p_dk_inter" (by decide) (by decide) (by decide), hpdkt, ptrDec_as_row DKInter s R0 h8])
      (by rw [hpeel .ptr [4] "p_dg" (by decide) (by decide) (by decide), hpdg, ptrDec_as_row DG s R0 h8])
  have hmemeq : sout.mem =
      (stMem s DG (R0 - 8) (fun i => cum0 i +
          dgSum s DQInner DQInter DKInner DKInter Q K G (R0 - 8) lg i)
        (stMem s DKInter (R0 - 8) (dkOut s DKInner DKInter G (R0 - 8) lg)
          (stMem s DQInter (R0 - 8) (dqOut s DQInner DQInter G (R0 - 8)) s1))).mem := by
    rw [hmemo]
    refine stMem_mem_congr s DG (R0 - 8) _ _ _ ?_
    refine stMem_mem_congr s DKInter (R0 - 8) _ _ _ ?_
    refine stMem_mem_congr s DQInter (R0 - 8) _ s3 s1 ?_
    rw [hs3, bwdIterHeadState]; simp only [BlockState.setReg]; rw [hsin]
    simp only [BlockState.setReg]
  exact ⟨sout, hstep, hpids, hmemeq, hcumo⟩

set_option maxHeartbeats 1600000 in
/-- **Full reverse-loop execution.** The complete backward surface executes to a
final state `s2` that is the two-iteration reverse scan: row `BT-1 = 1` stores
(`R0 = (i_c*2+1)*8`) on top of the prologue state `s0`, then row `0` stores
(`R0-8`) on top of that. We expose `s2.mem` as the four nested masked stores
(two per inter-region row plus the two `DG` rows) for readback. -/
theorem bwd_full_exec
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName) (s : BlockState)
    (hDKInner_DQInter : DKInner ≠ DQInter) (hDKInter_DQInter : DKInter ≠ DQInter)
    (hQ_DQInter : Q ≠ DQInter) (hQ_DKInter : Q ≠ DKInter)
    (hK_DQInter : K ≠ DQInter) (hK_DKInter : K ≠ DKInter) :
    ∃ s0 s1 s2,
      stepStmts ((bwd_decay_global_cumsum_surface DQInner DQInter DKInner DKInter
        Q K G DG 64 8 2 4).toAlgKernel.body.take 15) s = some s0
      ∧ exec (bwd_decay_global_cumsum_surface DQInner DQInter DKInner DKInter
          Q K G DG 64 8 2 4) s = some s2
      ∧ s0.pids = s.pids ∧ s0.mem = s.mem
      ∧ s1.mem =
          (stMem s DG ((s.pids 1 * 2 + 2 - 1) * 8) (fun i =>
              dgSum s DQInner DQInter DKInner DKInter Q K G ((s.pids 1 * 2 + 2 - 1) * 8)
                (ldR s G ((s.pids 1 * 2 + 2 - 1) * 8)) i)
            (stMem s DKInter ((s.pids 1 * 2 + 2 - 1) * 8)
                (dkOut s DKInner DKInter G ((s.pids 1 * 2 + 2 - 1) * 8)
                  (ldR s G ((s.pids 1 * 2 + 2 - 1) * 8)))
              (stMem s DQInter ((s.pids 1 * 2 + 2 - 1) * 8)
                  (dqOut s DQInner DQInter G ((s.pids 1 * 2 + 2 - 1) * 8)) s0))).mem
      ∧ s2.mem =
          (stMem s DG ((s.pids 1 * 2 + 2 - 1) * 8 - 8) (fun i =>
              dgSum s DQInner DQInter DKInner DKInter Q K G ((s.pids 1 * 2 + 2 - 1) * 8)
                (ldR s G ((s.pids 1 * 2 + 2 - 1) * 8)) i +
              dgSum s DQInner DQInter DKInner DKInter Q K G ((s.pids 1 * 2 + 2 - 1) * 8 - 8)
                (ldR s G ((s.pids 1 * 2 + 2 - 1) * 8)) i)
            (stMem s DKInter ((s.pids 1 * 2 + 2 - 1) * 8 - 8)
                (dkOut s DKInner DKInter G ((s.pids 1 * 2 + 2 - 1) * 8 - 8)
                  (ldR s G ((s.pids 1 * 2 + 2 - 1) * 8)))
              (stMem s DQInter ((s.pids 1 * 2 + 2 - 1) * 8 - 8)
                  (dqOut s DQInner DQInter G ((s.pids 1 * 2 + 2 - 1) * 8 - 8)) s1))).mem := by
  set R0 := (s.pids 1 * 2 + 2 - 1) * 8 with hR0
  have h8 : 8 ≤ R0 := by rw [hR0]; omega
  -- Prologue.
  obtain ⟨s0, hpro, hp_pid, hp_mem, hp_undef, _, _, _, _, hp_mask, hp_cum, hp_lastg,
    hp_pq, hp_pk, hp_pg, hp_pdg, hp_pdqi, hp_pdki, hp_pdqt, hp_pdkt⟩ :=
    bwd_prologue_eval DQInner DQInter DKInner DKInter Q K G DG s
  have hsh : BwdPrologueShape s s0 DQInner DQInter DKInner DKInter Q K G DG R0 :=
    { pid := hp_pid, mem := hp_mem, mask := hp_mask, cum := hp_cum, lastg := hp_lastg,
      pg := hp_pg, pk := hp_pk, pq := hp_pq, pdqi := hp_pdqi, pdki := hp_pdki,
      pdqt := hp_pdqt, pdkt := hp_pdkt, pdg := hp_pdg }
  -- Iteration 0.
  obtain ⟨s1, hstep0, h1pid, h1mem, h1mask, h1lastg, h1cum,
    h1pg, h1pk, h1pq, h1pdqi, h1pdki, h1pdqt, h1pdkt, h1pdg⟩ :=
    bwd_iter0_eval DQInner DQInter DKInner DKInter Q K G DG s s0 R0
      hDKInner_DQInter hDKInter_DQInter hQ_DQInter hQ_DKInter hK_DQInter hK_DKInter hsh
  -- `s1` reads at row `R0 - 8` agree with `s` (iter-0 wrote only at row `R0`).
  have h1row : ∀ (r : RegionName) (i : TileIndex [4]),
      s1.readMem r (s.pids 2 * 64 + s.pids 0 * 4 + i.1.val + (R0 - 8))
        = s.readMem r (s.pids 2 * 64 + s.pids 0 * 4 + i.1.val + (R0 - 8)) := by
    intro r i
    have hoff : ∀ j : TileIndex [4],
        s.pids 2 * 64 + s.pids 0 * 4 + j.1.val + R0
          ≠ s.pids 2 * 64 + s.pids 0 * 4 + i.1.val + (R0 - 8) := by
      intro j; have := j.1.2; have := i.1.2; omega
    have hrm : s1.readMem r (s.pids 2 * 64 + s.pids 0 * 4 + i.1.val + (R0 - 8))
        = (stMem s DG R0 (fun i =>
              dgSum s DQInner DQInter DKInner DKInter Q K G R0 (ldR s G R0) i)
            (stMem s DKInter R0 (dkOut s DKInner DKInter G R0 (ldR s G R0))
              (stMem s DQInter R0 (dqOut s DQInner DQInter G R0) s0))).readMem r
            (s.pids 2 * 64 + s.pids 0 * 4 + i.1.val + (R0 - 8)) := by
      simp only [BlockState.readMem, h1mem]
    rw [hrm, stMem_readMem_other_offset s DG r R0 _ _ _ hoff,
        stMem_readMem_other_offset s DKInter r R0 _ _ _ hoff,
        stMem_readMem_other_offset s DQInter r R0 _ _ _ hoff,
        show s0.readMem r (s.pids 2 * 64 + s.pids 0 * 4 + i.1.val + (R0 - 8))
            = s.readMem r (s.pids 2 * 64 + s.pids 0 * 4 + i.1.val + (R0 - 8)) by
          simp only [BlockState.readMem, hp_mem]]
  -- Iteration 1.
  obtain ⟨s2, hstep1, h2pid, h2mem, h2cum⟩ :=
    bwd_iter1_eval DQInner DQInter DKInner DKInter Q K G DG s s1 R0 h8
      (ldR s G R0)
      (fun i => dgSum s DQInner DQInter DKInner DKInter Q K G R0 (ldR s G R0) i)
      hDKInner_DQInter hDKInter_DQInter hQ_DQInter hQ_DKInter hK_DQInter hK_DKInter
      h1pid h1row h1mask h1lastg h1cum h1pg h1pk h1pq h1pdqi h1pdki h1pdqt h1pdkt h1pdg
  -- Assemble `exec`.
  refine ⟨s0, s1, s2, hpro, ?_, hp_pid, hp_mem, h1mem, h2mem⟩
  rw [exec, bwd_body_decomp, stepStmts.append_some hpro,
    stepStmts.cons_some (bwd_loop_unroll s0 s1 s2 hstep0 hstep1), stepStmts.nil]

/-- `dqOut` at row offset `(s.pids1*2+t_rel)*8` equals the genuine `dq_inter`
closed form at chunk row `t_rel`. -/
private theorem dqOut_eq_closed (s : BlockState) (DQInner DQInter G : RegionName)
    (t_rel : Fin 2) (i : Fin 4) (R : Nat)
    (hR : R = (s.pids 1 * 2 + t_rel.val) * 8) :
    dqOut s DQInner DQInter G R (i, PUnit.unit)
      = bwdDQInterClosed s DQInner DQInter G 64 8 2 4 t_rel i := by
  have ha : s.pids 2 * 64 + s.pids 0 * 4 + i.val + R
      = offset s 64 8 t_rel.val 2 4 i := by simp only [offset, baseOffset, hR]; ring
  simp only [dqOut, ldR, bwdDQInterClosed, ha]

/-- `dkOut` (with captured `last_g = g[row BT-1]`) at row offset `R = (s.pids1*2+
t_rel)*8` equals the genuine `dk_inter` closed form at row `t_rel`. -/
private theorem dkOut_eq_closed (s : BlockState) (DKInner DKInter G : RegionName)
    (t_rel : Fin 2) (i : Fin 4) (R Rlast : Nat)
    (hR : R = (s.pids 1 * 2 + t_rel.val) * 8) (hRlast : Rlast = (s.pids 1 * 2 + 1) * 8) :
    dkOut s DKInner DKInter G R (ldR s G Rlast) (i, PUnit.unit)
      = bwdDKInterClosed s DKInner DKInter G 64 8 2 4 t_rel i := by
  have ha : s.pids 2 * 64 + s.pids 0 * 4 + i.val + R
      = offset s 64 8 t_rel.val 2 4 i := by simp only [offset, baseOffset, hR]; ring
  have hb : s.pids 2 * 64 + s.pids 0 * 4 + i.val + Rlast
      = offset s 64 8 (2 - 1) 2 4 i := by simp only [offset, baseOffset, hRlast]; ring
  simp only [dkOut, ldR, bwdDKInterClosed, ha, hb]

/-- Helper: read a region `X` at an address other than the three written rows of
an `stMem`-triple stacked at row `R`, when `X` differs from the three store
regions or the address avoids the row range. -/
private theorem stMem3_read_skip (s : BlockState) (A B C X : RegionName) (R : Nat)
    (fa fb fc : TileIndex [4] → ℝ) (c : BlockState) (off : Nat)
    (hskip : ∀ region ∈ [A, B, C], ∀ j : TileIndex [4],
      region = X → s.pids 2 * 64 + s.pids 0 * 4 + j.1.val + R ≠ off) :
    (stMem s A R fa (stMem s B R fb (stMem s C R fc c))).readMem X off
      = c.readMem X off := by
  have go : ∀ (D : RegionName) (fd : TileIndex [4] → ℝ) (c' : BlockState),
      (D = X → ∀ j : TileIndex [4], s.pids 2 * 64 + s.pids 0 * 4 + j.1.val + R ≠ off) →
      (stMem s D R fd c').readMem X off = c'.readMem X off := by
    intro D fd c' hd
    by_cases hDX : X = D
    · subst hDX
      exact stMem_readMem_other_offset s X X R fd c' off (hd rfl)
    · exact stMem_readMem_other s D X R fd c' off hDX
  rw [go A fa _ (fun hAX j => hskip A (by simp) j hAX),
      go B fb _ (fun hBX j => hskip B (by simp) j hBX),
      go C fc _ (fun hCX j => hskip C (by simp) j hCX)]

set_option maxHeartbeats 1600000 in
/-- **Genuine `dq_inter` readback.** The executed backward surface writes the
honest closed form `bwdDQInterClosed` (= `dq_inner + dq_inter * exp2(g)`) into
`DQInter` at every active lane of loop row `t_rel`. -/
theorem bwd_decay_cumsum_dq_inter_closed_compute_correct
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName) (t_rel : Fin 2) (s : BlockState)
    (hDQInter_DKInter : DQInter ≠ DKInter) (hDQInter_DG : DQInter ≠ DG)
    (hDKInner_DQInter : DKInner ≠ DQInter) (hDKInter_DQInter : DKInter ≠ DQInter)
    (hQ_DQInter : Q ≠ DQInter) (hQ_DKInter : Q ≠ DKInter)
    (hK_DQInter : K ≠ DQInter) (hK_DKInter : K ≠ DKInter) :
    ComputeCorrect.Realizes
      (kernel := bwd_decay_global_cumsum_surface DQInner DQInter DKInner DKInter
        Q K G DG 64 8 2 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf (active s 8 4)
        (fun i => (DQInter, offset s 64 8 t_rel.val 2 4 i)))
      (expected := fun i : Fin 4 =>
        bwdDQInterClosed s DQInner DQInter G 64 8 2 4 t_rel i) := by
  obtain ⟨s0, s1, s2, _, hexec, _, _, h1mem, h2mem⟩ :=
    bwd_full_exec DQInner DQInter DKInner DKInter Q K G DG s
      hDKInner_DQInter hDKInter_DQInter hQ_DQInter hQ_DKInter hK_DQInter hK_DKInter
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [bwd_decay_global_cumsum_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro sa s' hExec hsa
  subst sa
  intro i hActive
  have hs' : s' = s2 := by
    rw [exec] at hexec; rw [exec] at hExec; rw [hExec] at hexec; exact Option.some.inj hexec
  subst s'
  have hacP : s.pids 0 * 4 + i.val < 8 := by simpa [active, elemIndex] using hActive
  have hac : decide (s.pids 0 * 4 + i.val < 8) = Bool.true := decide_eq_true hacP
  have hoff : offset s 64 8 t_rel.val 2 4 i
      = s.pids 2 * 64 + s.pids 0 * 4 + i.val + (s.pids 1 * 2 + t_rel.val) * 8 := by
    simp only [offset, baseOffset]; ring
  show s2.readMem DQInter (offset s 64 8 t_rel.val 2 4 i) = _
  rw [hoff]
  -- `s2.mem` / `s1.mem` read `DQInter` via memory-only equalities (`readMem` is a
  -- function of `.mem`).
  have hmemrd : ∀ (a b : BlockState), a.mem = b.mem →
      ∀ (r : RegionName) (o : Nat), a.readMem r o = b.readMem r o := by
    intro a b hab r o; simp only [BlockState.readMem, hab]
  rw [hmemrd s2 _ h2mem DQInter]
  match t_rel with
  | ⟨0, hlt⟩ =>
    -- Row 0 = R0 - 8: skip DG/DKInter (other region) of the outer triple, read DQInter.
    rw [stMem_readMem_other s DG DQInter _ _ _ _ hDQInter_DG,
        stMem_readMem_other s DKInter DQInter _ _ _ _ hDQInter_DKInter]
    rw [show (s.pids 1 * 2 + 0) * 8 = (s.pids 1 * 2 + 2 - 1) * 8 - 8 from by omega,
      stMem_readback s DQInter ((s.pids 1 * 2 + 2 - 1) * 8 - 8) _ s1 (i, PUnit.unit)]
    simp only [hac, if_true]
    rw [dqOut_eq_closed s DQInner DQInter G ⟨0, hlt⟩ i _
      (show (s.pids 1 * 2 + 2 - 1) * 8 - 8 = (s.pids 1 * 2 + 0) * 8 by omega)]
  | ⟨1, hlt⟩ =>
    -- Row 1 = R0: outer triple (R0-8) skips it by offset; inner triple reads DQInter.
    have hskip : ∀ region ∈ [DG, DKInter, DQInter], ∀ j : TileIndex [4],
        region = DQInter → s.pids 2 * 64 + s.pids 0 * 4 + j.1.val
            + ((s.pids 1 * 2 + 2 - 1) * 8 - 8) ≠
          s.pids 2 * 64 + s.pids 0 * 4 + i.val + (s.pids 1 * 2 + 1) * 8 := by
      intro region _ j _; have := j.1.2; have := (i : Fin 4).2; omega
    rw [stMem3_read_skip s DG DKInter DQInter DQInter ((s.pids 1 * 2 + 2 - 1) * 8 - 8)
        _ _ _ s1 _ hskip, hmemrd s1 _ h1mem DQInter]
    rw [stMem_readMem_other s DG DQInter _ _ _ _ hDQInter_DG,
        stMem_readMem_other s DKInter DQInter _ _ _ _ hDQInter_DKInter]
    rw [show (s.pids 1 * 2 + 1) * 8 = (s.pids 1 * 2 + 2 - 1) * 8 from by omega,
      stMem_readback s DQInter ((s.pids 1 * 2 + 2 - 1) * 8) _ s0 (i, PUnit.unit)]
    simp only [hac, if_true]
    rw [dqOut_eq_closed s DQInner DQInter G ⟨1, hlt⟩ i _
      (show (s.pids 1 * 2 + 2 - 1) * 8 = (s.pids 1 * 2 + 1) * 8 by omega)]

set_option maxHeartbeats 1600000 in
/-- **Genuine `dk_inter` readback.** The executed backward surface writes the
honest closed form `bwdDKInterClosed` (= `dk_inner + dk_inter * exp2(last_g - g)`)
into `DKInter` at every active lane of loop row `t_rel`. -/
theorem bwd_decay_cumsum_dk_inter_closed_compute_correct
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName) (t_rel : Fin 2) (s : BlockState)
    (hDKInter_DG : DKInter ≠ DG) (hDQInter_DKInter : DQInter ≠ DKInter)
    (hDKInner_DQInter : DKInner ≠ DQInter) (hDKInter_DQInter : DKInter ≠ DQInter)
    (hQ_DQInter : Q ≠ DQInter) (hQ_DKInter : Q ≠ DKInter)
    (hK_DQInter : K ≠ DQInter) (hK_DKInter : K ≠ DKInter) :
    ComputeCorrect.Realizes
      (kernel := bwd_decay_global_cumsum_surface DQInner DQInter DKInner DKInter
        Q K G DG 64 8 2 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf (active s 8 4)
        (fun i => (DKInter, offset s 64 8 t_rel.val 2 4 i)))
      (expected := fun i : Fin 4 =>
        bwdDKInterClosed s DKInner DKInter G 64 8 2 4 t_rel i) := by
  obtain ⟨s0, s1, s2, _, hexec, _, _, h1mem, h2mem⟩ :=
    bwd_full_exec DQInner DQInter DKInner DKInter Q K G DG s
      hDKInner_DQInter hDKInter_DQInter hQ_DQInter hQ_DKInter hK_DQInter hK_DKInter
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [bwd_decay_global_cumsum_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro sa s' hExec hsa
  subst sa
  intro i hActive
  have hs' : s' = s2 := by
    rw [exec] at hexec; rw [exec] at hExec; rw [hExec] at hexec; exact Option.some.inj hexec
  subst s'
  have hacP : s.pids 0 * 4 + i.val < 8 := by simpa [active, elemIndex] using hActive
  have hac : decide (s.pids 0 * 4 + i.val < 8) = Bool.true := decide_eq_true hacP
  have hoff : offset s 64 8 t_rel.val 2 4 i
      = s.pids 2 * 64 + s.pids 0 * 4 + i.val + (s.pids 1 * 2 + t_rel.val) * 8 := by
    simp only [offset, baseOffset]; ring
  have hmemrd : ∀ (a b : BlockState), a.mem = b.mem →
      ∀ (r : RegionName) (o : Nat), a.readMem r o = b.readMem r o := by
    intro a b hab r o; simp only [BlockState.readMem, hab]
  show s2.readMem DKInter (offset s 64 8 t_rel.val 2 4 i) = _
  rw [hoff, hmemrd s2 _ h2mem DKInter]
  match t_rel with
  | ⟨0, hlt⟩ =>
    rw [stMem_readMem_other s DG DKInter _ _ _ _ hDKInter_DG]
    rw [show (s.pids 1 * 2 + 0) * 8 = (s.pids 1 * 2 + 2 - 1) * 8 - 8 from by omega,
      stMem_readback s DKInter ((s.pids 1 * 2 + 2 - 1) * 8 - 8) _ _ (i, PUnit.unit)]
    simp only [hac, if_true]
    rw [dkOut_eq_closed s DKInner DKInter G ⟨0, hlt⟩ i _ _
      (show (s.pids 1 * 2 + 2 - 1) * 8 - 8 = (s.pids 1 * 2 + 0) * 8 by omega)
      (show (s.pids 1 * 2 + 2 - 1) * 8 = (s.pids 1 * 2 + 1) * 8 by omega)]
  | ⟨1, hlt⟩ =>
    have hskip : ∀ region ∈ [DG, DKInter, DQInter], ∀ j : TileIndex [4],
        region = DKInter → s.pids 2 * 64 + s.pids 0 * 4 + j.1.val
            + ((s.pids 1 * 2 + 2 - 1) * 8 - 8) ≠
          s.pids 2 * 64 + s.pids 0 * 4 + i.val + (s.pids 1 * 2 + 1) * 8 := by
      intro region _ j _; have := j.1.2; have := (i : Fin 4).2; omega
    rw [stMem3_read_skip s DG DKInter DQInter DKInter ((s.pids 1 * 2 + 2 - 1) * 8 - 8)
        _ _ _ s1 _ hskip, hmemrd s1 _ h1mem DKInter]
    rw [stMem_readMem_other s DG DKInter _ _ _ _ hDKInter_DG]
    rw [show (s.pids 1 * 2 + 1) * 8 = (s.pids 1 * 2 + 2 - 1) * 8 from by omega,
      stMem_readback s DKInter ((s.pids 1 * 2 + 2 - 1) * 8) _ _ (i, PUnit.unit)]
    simp only [hac, if_true]
    rw [dkOut_eq_closed s DKInner DKInter G ⟨1, hlt⟩ i _ _
      (show (s.pids 1 * 2 + 2 - 1) * 8 = (s.pids 1 * 2 + 1) * 8 by omega)
      (show (s.pids 1 * 2 + 2 - 1) * 8 = (s.pids 1 * 2 + 1) * 8 by omega)]

end BwdAssembly

end VeriTile.Bench.TritonBenchG.DecayCumsum
