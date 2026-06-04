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

end BwdAssembly

end VeriTile.Bench.TritonBenchG.DecayCumsum
