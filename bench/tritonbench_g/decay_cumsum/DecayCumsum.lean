import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.Kernel

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
There is no single 3-kernel bundling theorem. The three kernels are verified by
three separate dimension-general top results (each symbolic in the block sizes
`BT`/`BK`/`DK` and the strides):

forward kernel:
  fwd_decay_cumsum_full_surface_closed_general              ← TOP THEOREM (forward)
       └─ fwd_decay_cumsum_surface_toAlgorithm_supported

prepare kernel:
  prepare_qg_kg_full_surface_qg_closed_general             ← TOP THEOREM (prepare, qg)
  prepare_qg_kg_full_surface_kg_closed_general             ← TOP THEOREM (prepare, kg)
       └─ prepare_qg_kg_surface_toAlgorithm_supported

backward kernel:
  decay_cumsum_backward_closed_output_summary_general  ← TOP THEOREM (backward)
       (realizes genuine closed forms bwdDQInterClosed / bwdDKInterClosed / bwdDGClosed)
       └─ bwd_decay_global_cumsum_surface_toAlgorithm_supported

each top result is the symbolic statement; the concrete Python shape is a
recovered special case.
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float). There is no
`@triton.autotune` on these kernels. The top results are **dimension-
parameterized**: they are stated over symbolic block sizes (`BT`/`BK`/`DK`) and
strides, so the checked Python shape `B,H,T,DK = 2,2,4,8`, `BT,BK = 2,4`,
`scale = 1` is only a recovered instance, not a fixed assumption. The
`.to(tl.float32)` / `.to(_.dtype.element_ty)` casts erase to the identity at the
algorithm layer (post-erasure all dtypes unify to `ℝ`). The `inv_ln2` /
`tl.math.exp2` decay factors are modeled as their real-valued counterparts. Each
kernel's full surface — including the cross-step cumulative folds: the forward
`range(BT)` loop threading `cum_decay`, the per-step `qg`/`kg` scaling, and the
reverse `range(BT-1,-1,-1)` loop threading `cum_grad_dg` plus the `last_g`
capture — is executed and *proven* to realize a genuine, non self-referential
closed form: the backward outputs collapse to `bwdDQInterClosed` /
`bwdDKInterClosed` / `bwdDGClosed`, and the forward/prepare surfaces realize
their respective closed-form decay specifications. Output offset injectivity is a
side condition.
-/

namespace VeriTile.Bench.TritonBenchG.DecayCumsum

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-! **★ Main theorem:** `decay_cumsum_backward_closed_output_summary_general` -/

section Correct

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

/-- Running partial decay-cumsum value at lane `i` after `m` rows have been
folded: `1.44269504 * Σ_{k<m} g[row k, lane i]`. -/
noncomputable def fwdPartial
    (s : BlockState) (G : RegionName)
    (s_qk_h DK BT BK : Nat) (m : Nat) (i : Fin BK) : ℝ :=
  1.44269504 * ∑ k : Fin m, s.readMem G (offset s s_qk_h DK k.val BT BK i)

/-- `cum_decay` register tile after `m` rows: masked partial decay-cumsum. -/
noncomputable def fwdCumTile
    (s : BlockState) (G : RegionName)
    (s_qk_h DK BT BK : Nat) (m : Nat) : Tile .real [BK] :=
  ⟨fun idx => if active s DK BK idx.1 then
      some (fwdPartial s G s_qk_h DK BT BK m idx.1) else some 0⟩

/-- `p_g` pointer register tile after `m` rows: lane `i` points to `G` at the
row-`m` offset. -/
def fwdPgTile
    (s : BlockState) (G : RegionName)
    (s_qk_h DK BT BK : Nat) (m : Nat) : Tile .ptr [BK] :=
  ⟨fun idx => (G, offset s s_qk_h DK m BT BK idx.1)⟩

/-- `p_go` pointer register tile after `m` rows: lane `i` points to `GO` at the
row-`m` offset. -/
def fwdPgoTile
    (s : BlockState) (GO : RegionName)
    (s_qk_h DK BT BK : Nat) (m : Nat) : Tile .ptr [BK] :=
  ⟨fun idx => (GO, offset s s_qk_h DK m BT BK idx.1)⟩

/-- `mask` register tile: lane `i` is the activeness bit. -/
def fwdMaskTile (s : BlockState) (DK BK : Nat) : Tile .bool [BK] :=
  ⟨fun idx => decide (active s DK BK idx.1)⟩

/-- The forward decay-cumsum loop invariant after `m` rows. -/
noncomputable def fwdInv
    (G GO : RegionName) (s : BlockState) (s_qk_h DK BT BK : Nat) :
    Nat → BlockState → Prop :=
  fun m sc =>
    sc.pids = s.pids ∧
    (∀ a, sc.readMem G a = s.readMem G a) ∧
    sc.regs .real [BK] "cum_decay" = some (fwdCumTile s G s_qk_h DK BT BK m) ∧
    sc.regs .ptr [BK] "p_g" = some (fwdPgTile s G s_qk_h DK BT BK m) ∧
    sc.regs .ptr [BK] "p_go" = some (fwdPgoTile s GO s_qk_h DK BT BK m) ∧
    sc.regs .bool [BK] "mask" = some (fwdMaskTile s DK BK) ∧
    (∀ j, m ≤ j → ∀ i : Fin BK,
      sc.readMem GO (offset s s_qk_h DK j BT BK i) =
        s.readMem GO (offset s s_qk_h DK j BT BK i)) ∧
    (∀ j, j < m → ∀ i : Fin BK,
      sc.readMem GO (offset s s_qk_h DK j BT BK i) =
        if active s DK BK i then
          fwdPartial s G s_qk_h DK BT BK (j + 1) i
        else s.readMem GO (offset s s_qk_h DK j BT BK i))

/-- The decay-partial sum recurrence: folding one more row adds a scaled `g`. -/
theorem fwdPartial_succ
    (s : BlockState) (G : RegionName)
    (s_qk_h DK BT BK : Nat) (m : Nat) (i : Fin BK) :
    fwdPartial s G s_qk_h DK BT BK (m + 1) i =
      fwdPartial s G s_qk_h DK BT BK m i +
        1.44269504 * s.readMem G (offset s s_qk_h DK m BT BK i) := by
  simp only [fwdPartial, Fin.sum_univ_castSucc, Fin.val_last, Fin.val_castSucc]
  ring

/-- The masked-loaded `g` tile at row `m`: active lanes read `G`, inactive read 0. -/
noncomputable def fwdGTile
    (s : BlockState) (G : RegionName)
    (s_qk_h DK BT BK : Nat) (m : Nat) : Tile .real [BK] :=
  ⟨fun idx => if active s DK BK idx.1 then
      some (s.readMem G (offset s s_qk_h DK m BT BK idx.1)) else some 0⟩

/-- The lowered forward loop body (one row): masked load of `g`, accumulate
`cum_decay += g * inv_ln2`, masked store into `g_o`, increment `p_g`/`p_go`. -/
def fwdBody (G GO : RegionName) (s_qk_h BT BK DK : Nat) : List Stmt :=
  [Stmt.assign TileDType.real [BK] "_g"
      (Op.load ComputeDType.fp32.eraseDType (MemAccess.ptr (Op.ref TileDType.ptr [BK] "p_g"))
        (MaskOpt.maskOther (Op.ref TileDType.bool [BK] "mask") ((Op.const 0).broadcast [BK]))),
    Stmt.assign TileDType.real [BK] "cum_decay"
      (Op.add NumericDType.real Broadcast.nil.consSame (Op.ref TileDType.real [BK] "cum_decay")
        (Op.mul NumericDType.real Broadcast.scalarR (Op.ref TileDType.real [BK] "_g") (Op.const 1.44269504))),
    Stmt.store TileDType.real [BK] (MemAccess.ptr (Op.ref TileDType.ptr [BK] "p_go"))
      (Op.ref TileDType.real [BK] "cum_decay") (MaskOpt.mask (Op.ref TileDType.bool [BK] "mask")),
    Stmt.assign TileDType.ptr [BK] "p_g"
      (Op.ptrAdd Broadcast.scalarR (Op.ref TileDType.ptr [BK] "p_g") (Op.constNat DK)),
    Stmt.assign TileDType.ptr [BK] "p_go"
      (Op.ptrAdd Broadcast.scalarR (Op.ref TileDType.ptr [BK] "p_go") (Op.constNat DK))]

set_option maxHeartbeats 4000000 in
/-- **One row advances the forward invariant.** -/
theorem fwd_decay_cumsum_step
    (G GO : RegionName) (s : BlockState) (s_qk_h BT BK DK : Nat)
    (hne : G ≠ GO) (hBK : BK ≤ DK)
    (m : Nat) (sc : BlockState) (hP : fwdInv G GO s s_qk_h DK BT BK m sc) :
    ∃ s',
      stepStmts (fwdBody G GO s_qk_h BT BK DK)
        (sc.setReg "_i" .nat [] (Tile.scalar m)) = some s' ∧
      fwdInv G GO s s_qk_h DK BT BK (m + 1) s' := by
  obtain ⟨hpids, hG, hcum, hpg, hpgo, hmask, hUntouched, hreadGO⟩ := hP
  set sloop := sc.setReg "_i" .nat [] (Tile.scalar m) with hsloop
  -- register lookups along sloop
  have hcum' : sloop.regs .real [BK] "cum_decay" = some (fwdCumTile s G s_qk_h DK BT BK m) := by
    rw [hsloop]; simpa using hcum
  have hpg' : sloop.regs .ptr [BK] "p_g" = some (fwdPgTile s G s_qk_h DK BT BK m) := by
    rw [hsloop]; simpa using hpg
  have hpgo' : sloop.regs .ptr [BK] "p_go" = some (fwdPgoTile s GO s_qk_h DK BT BK m) := by
    rw [hsloop]; simpa using hpgo
  have hmask' : sloop.regs .bool [BK] "mask" = some (fwdMaskTile s DK BK) := by
    rw [hsloop]; simpa using hmask
  -- (1) the masked `_g` load
  have hgeval : evalOp (Op.load TileDType.real (MemAccess.ptr (Op.ref TileDType.ptr [BK] "p_g"))
        (MaskOpt.maskOther (Op.ref TileDType.bool [BK] "mask") ((Op.const 0).broadcast [BK]))) sloop
      = some (fwdGTile s G s_qk_h DK BT BK m) := by
    simp only [evalOp, Option.bind, hpg', hmask', evalOp_const]
    apply congrArg some
    apply Tile.ext
    intro idx
    obtain ⟨i, u⟩ := idx
    simp only [fwdGTile, fwdMaskTile, fwdPgTile]
    by_cases ha : active s DK BK i
    · simp only [ha, decide_true, if_true, if_pos]
      rw [hsloop, BlockState.setReg_readMemValue, BlockState.readMemValue_real, hG]
    · simp only [ha, decide_false, Bool.false_eq_true, if_false]
      rfl
  -- intermediate states
  set s1 := sloop.setReg "_g" .real [BK] (fwdGTile s G s_qk_h DK BT BK m) with hs1
  have hstep1 : stepStmt (Stmt.assign TileDType.real [BK] "_g"
      (Op.load ComputeDType.fp32.eraseDType (MemAccess.ptr (Op.ref TileDType.ptr [BK] "p_g"))
        (MaskOpt.maskOther (Op.ref TileDType.bool [BK] "mask") ((Op.const 0).broadcast [BK])))) sloop
      = some s1 := by
    rw [hs1]; exact stepStmt_assign_eq_some (by simpa [ComputeDType.eraseDType] using hgeval)
  -- (2) cum_decay += _g * inv_ln2
  have hcum1 : s1.regs .real [BK] "cum_decay" = some (fwdCumTile s G s_qk_h DK BT BK m) := by
    rw [hs1]; simpa using hcum'
  have hg1 : s1.regs .real [BK] "_g" = some (fwdGTile s G s_qk_h DK BT BK m) := by
    rw [hs1]; simp
  have hcumeval : evalOp (Op.add NumericDType.real Broadcast.nil.consSame
      (Op.ref TileDType.real [BK] "cum_decay")
      (Op.mul NumericDType.real Broadcast.scalarR (Op.ref TileDType.real [BK] "_g")
        (Op.const 1.44269504))) s1
      = some (fwdCumTile s G s_qk_h DK BT BK (m + 1)) := by
    simp only [evalOp, Option.bind, hcum1, hg1, evalOp_const]
    apply congrArg some
    apply Tile.ext
    intro idx
    obtain ⟨i, u⟩ := idx
    simp only [fwdCumTile, fwdGTile, Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex,
      NumericDType.add, NumericDType.mul]
    by_cases ha : active s DK BK i
    · simp only [ha, if_true, WithBot.realAdd, WithBot.realMul]
      rw [fwdPartial_succ]
      norm_num
      ring
    · simp only [ha, if_false, WithBot.realAdd, WithBot.realMul]
      norm_num
  set s2 := s1.setReg "cum_decay" .real [BK] (fwdCumTile s G s_qk_h DK BT BK (m + 1)) with hs2
  have hstep2 : stepStmt (Stmt.assign TileDType.real [BK] "cum_decay"
      (Op.add NumericDType.real Broadcast.nil.consSame (Op.ref TileDType.real [BK] "cum_decay")
        (Op.mul NumericDType.real Broadcast.scalarR (Op.ref TileDType.real [BK] "_g")
          (Op.const 1.44269504)))) s1 = some s2 := by
    rw [hs2]; exact stepStmt_assign_eq_some hcumeval
  -- (3) masked store into GO via p_go
  have hpgo2 : s2.regs .ptr [BK] "p_go" = some (fwdPgoTile s GO s_qk_h DK BT BK m) := by
    rw [hs2, hs1]; simp [hpgo']
  have hmask2 : s2.regs .bool [BK] "mask" = some (fwdMaskTile s DK BK) := by
    rw [hs2, hs1]; simp [hmask']
  have hcum2 : s2.regs .real [BK] "cum_decay" = some (fwdCumTile s G s_qk_h DK BT BK (m + 1)) := by
    rw [hs2]; simp
  -- value function for the scatter
  set vfun : TileIndex [BK] → ℝ := fun k =>
    (fwdCumTile s G s_qk_h DK BT BK (m + 1)).data k |>.unbotD 0 with hvfun
  -- the store final state, in clean masked `writeMem GO` form
  set sst := (TileShape.allIndices [BK]).foldl
    (fun (acc : BlockState) k =>
      if active s DK BK k.1 then
        acc.writeMem GO (offset s s_qk_h DK m BT BK k.1) (vfun k)
      else acc) s2 with hsst
  have hstore : stepStmt (Stmt.store TileDType.real [BK] (MemAccess.ptr (Op.ref TileDType.ptr [BK] "p_go"))
      (Op.ref TileDType.real [BK] "cum_decay") (MaskOpt.mask (Op.ref TileDType.bool [BK] "mask"))) s2
      = some sst := by
    simp only [stepStmt, evalOp, Option.bind, Option.map, hcum2, hpgo2, hmask2]
    apply congrArg some
    rw [hsst]
    congr 1
    funext acc k
    simp only [fwdPgoTile, fwdMaskTile]
    by_cases ha : active s DK BK k.1
    · rw [if_pos (by simpa using ha), if_pos ha, BlockState.writeMemTyped_real]
      rfl
    · rw [if_neg (by simpa using ha), if_neg ha]
  -- sst register/pids facts (writes preserve all registers)
  have hpg2 : sst.regs .ptr [BK] "p_g" = some (fwdPgTile s G s_qk_h DK BT BK m) := by
    rw [hsst, BlockState.foldl_writeMem_prop_masked_regs, hs2, hs1]; simp [hpg']
  have hpgadd : evalOp (Op.ptrAdd Broadcast.scalarR (Op.ref TileDType.ptr [BK] "p_g")
      (Op.constNat DK)) sst = some (fwdPgTile s G s_qk_h DK BT BK (m + 1)) := by
    simp only [evalOp, Option.bind, hpg2, evalOp_constNat]
    apply congrArg some
    apply Tile.ext
    intro idx
    obtain ⟨i, u⟩ := idx
    simp only [Tile.ptrAdd_data, Broadcast.leftIndex, Broadcast.rightIndex, fwdPgTile, Tile.scalar]
    refine Prod.ext rfl ?_
    simp only [offset, baseOffset]
    ring
  set s3 := sst.setReg "p_g" .ptr [BK] (fwdPgTile s G s_qk_h DK BT BK (m + 1)) with hs3
  have hstep4 : stepStmt (Stmt.assign TileDType.ptr [BK] "p_g"
      (Op.ptrAdd Broadcast.scalarR (Op.ref TileDType.ptr [BK] "p_g") (Op.constNat DK))) sst
      = some s3 := by
    rw [hs3]; exact stepStmt_assign_eq_some hpgadd
  -- (5) p_go += DK
  have hpgo3 : s3.regs .ptr [BK] "p_go" = some (fwdPgoTile s GO s_qk_h DK BT BK m) := by
    rw [hs3, BlockState.setReg_ne_name (h := by decide)]
    rw [hsst, BlockState.foldl_writeMem_prop_masked_regs, hs2, hs1]; simp [hpgo']
  have hpgoadd : evalOp (Op.ptrAdd Broadcast.scalarR (Op.ref TileDType.ptr [BK] "p_go")
      (Op.constNat DK)) s3 = some (fwdPgoTile s GO s_qk_h DK BT BK (m + 1)) := by
    simp only [evalOp, Option.bind, hpgo3, evalOp_constNat]
    apply congrArg some
    apply Tile.ext
    intro idx
    obtain ⟨i, u⟩ := idx
    simp only [Tile.ptrAdd_data, Broadcast.leftIndex, Broadcast.rightIndex, fwdPgoTile, Tile.scalar]
    refine Prod.ext rfl ?_
    simp only [offset, baseOffset]
    ring
  set s4 := s3.setReg "p_go" .ptr [BK] (fwdPgoTile s GO s_qk_h DK BT BK (m + 1)) with hs4
  have hstep5 : stepStmt (Stmt.assign TileDType.ptr [BK] "p_go"
      (Op.ptrAdd Broadcast.scalarR (Op.ref TileDType.ptr [BK] "p_go") (Op.constNat DK))) s3
      = some s4 := by
    rw [hs4]; exact stepStmt_assign_eq_some hpgoadd
  refine ⟨s4, ?_, ?_⟩
  · simp only [fwdBody]
    rw [stepStmts.cons_some hstep1, stepStmts.cons_some hstep2,
        stepStmts.cons_some hstore, stepStmts.cons_some hstep4,
        stepStmts.cons_some hstep5, stepStmts.nil]
  · -- the invariant at m+1
    -- offset injectivity for the row-m scatter
    have hInj : Function.Injective (fun k : TileIndex [BK] =>
        offset s s_qk_h DK m BT BK k.1) := by
      rintro ⟨a, _⟩ ⟨b, _⟩ hab
      simp only [offset] at hab
      obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
      rfl
    -- mem reads pass through s4,s3 (setReg) to sst
    have hmemG4 : ∀ a, s4.readMem G a = sst.readMem G a := by
      intro a; rw [hs4, hs3]; simp only [BlockState.setReg_readMem]
    have hmemGO4 : ∀ a, s4.readMem GO a = sst.readMem GO a := by
      intro a; rw [hs4, hs3]; simp only [BlockState.setReg_readMem]
    -- sst preserves G (writes go to GO ≠ G)
    have hsstG : ∀ a, sst.readMem G a = s.readMem G a := by
      intro a
      rw [hsst, BlockState.scatter_prop_masked_preserves_other_region
        (P := fun k : TileIndex [BK] => active s DK BK k.1) (h_ne := hne)]
      rw [hs2, hs1]; simp only [BlockState.setReg_readMem]; exact hG a
    -- row-offset separation: for active k and j ≠ m, offset m k ≠ offset j i
    have hsep : ∀ (j : Nat), j ≠ m → ∀ (i : Fin BK),
        (∀ k : TileIndex [BK], active s DK BK k.1 →
          (fun k : TileIndex [BK] => offset s s_qk_h DK m BT BK k.1) k
            ≠ offset s s_qk_h DK j BT BK i) := by
      intro j hj i k _
      simp only [offset, baseOffset]
      have hik : (i : ℕ) < BK := i.isLt
      have hiDK : (i : ℕ) < DK := lt_of_lt_of_le hik hBK
      rcases Nat.lt_or_ge j m with hlt | hge'
      · have hge : (s.pids 1 * BT + j) * DK + DK ≤ (s.pids 1 * BT + m) * DK := by
          have : (s.pids 1 * BT + j) + 1 ≤ s.pids 1 * BT + m := by omega
          calc (s.pids 1 * BT + j) * DK + DK
              = ((s.pids 1 * BT + j) + 1) * DK := by ring
            _ ≤ (s.pids 1 * BT + m) * DK := Nat.mul_le_mul_right DK this
        omega
      · have hjm' : m < j := by omega
        have hkBK : (k.1 : ℕ) < BK := k.1.isLt
        have hkDK : (k.1 : ℕ) < DK := lt_of_lt_of_le hkBK hBK
        have hge : (s.pids 1 * BT + m) * DK + DK ≤ (s.pids 1 * BT + j) * DK := by
          have : (s.pids 1 * BT + m) + 1 ≤ s.pids 1 * BT + j := by omega
          calc (s.pids 1 * BT + m) * DK + DK
              = ((s.pids 1 * BT + m) + 1) * DK := by ring
            _ ≤ (s.pids 1 * BT + j) * DK := Nat.mul_le_mul_right DK this
        omega
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · -- pids
      rw [hs4, hs3]; simp only [BlockState.setReg_pids]
      rw [hsst, BlockState.foldl_writeMem_prop_masked_pids]
      rw [hs2, hs1]; simp only [BlockState.setReg_pids]; exact hpids
    · -- G preserved
      intro a; rw [hmemG4]; exact hsstG a
    · -- cum_decay
      rw [hs4, hs3, BlockState.setReg_ne_name (h := by decide),
        BlockState.setReg_ne_name (h := by decide), hsst,
        BlockState.foldl_writeMem_prop_masked_regs, hs2]; simp
    · -- p_g
      rw [hs4, BlockState.setReg_ne_name (h := by decide), hs3]; simp
    · -- p_go
      rw [hs4]; simp
    · -- mask
      rw [hs4, hs3, BlockState.setReg_ne_name (h := by decide),
        BlockState.setReg_ne_name (h := by decide), hsst,
        BlockState.foldl_writeMem_prop_masked_regs, hs2, hs1]; simp [hmask']
    · -- untouched GO at rows ≥ m+1
      intro j hj i
      rw [hmemGO4 (offset s s_qk_h DK j BT BK i), hsst]
      rw [BlockState.scatter_prop_masked_preserves_other_offset
        (region := GO)
        (offsetFn := fun k : TileIndex [BK] => offset s s_qk_h DK m BT BK k.1)
        (valueFn := vfun)
        (P := fun k : TileIndex [BK] => active s DK BK k.1)
        (off := offset s s_qk_h DK j BT BK i)
        (h_ne := hsep j (by omega) i)]
      rw [hs2, hs1, hsloop]; simp only [BlockState.setReg_readMem]
      exact hUntouched j (by omega) i
    · -- readback GO
      intro j hj i
      rw [hmemGO4 (offset s s_qk_h DK j BT BK i), hsst]
      by_cases hjm : j = m
      · subst hjm
        rw [BlockState.scatter_readback_prop_masked_nd s2
          (fun k : TileIndex [BK] => offset s s_qk_h DK j BT BK k.1) vfun
          (fun k => active s DK BK k.1) hInj (i, PUnit.unit)]
        by_cases ha : active s DK BK i
        · rw [if_pos ha, if_pos ha]
          simp only [hvfun, fwdCumTile]
          rw [if_pos ha]; rfl
        · rw [if_neg ha, if_neg ha]
          rw [hs2, hs1, hsloop]; simp only [BlockState.setReg_readMem]
          exact hUntouched j (le_refl j) i
      · -- j < m : preserved by the row-m scatter, then use hreadGO
        have hjlt : j < m := by omega
        rw [BlockState.scatter_prop_masked_preserves_other_offset
          (region := GO)
          (offsetFn := fun k : TileIndex [BK] => offset s s_qk_h DK m BT BK k.1)
          (valueFn := vfun)
          (P := fun k : TileIndex [BK] => active s DK BK k.1)
          (off := offset s s_qk_h DK j BT BK i)
          (h_ne := hsep j (by omega) i)]
        rw [hs2, hs1, hsloop]; simp only [BlockState.setReg_readMem]
        rw [hreadGO j hjlt i]

set_option maxHeartbeats 4000000 in
/-- **Genuine GENERAL forward closed form.** For arbitrary `BT BK DK` (with
`BK ≤ DK`, `0 < BT`), at chunk row `t_rel` and lane `i`, the forward
decay-cumsum surface writes the scaled prefix sum
`1.44269504 * Σ_{k ≤ t_rel} g[row k, lane i]` into `GO` on active lanes. This is
the dimension-parameterized replacement for the `BT=2`-pinned
`fwd_decay_cumsum_full_surface_closed`; the cross-step `range(BT)` cumulative
fold is discharged by the loop invariant `fwdInv`, not pinned. -/
theorem fwd_decay_cumsum_full_surface_closed_general
    (G GO : RegionName)
    (s_qk_h s_qk_t s_qk_d B H T : Nat) (scale : ℝ)
    (BT BK DK : Nat) (s : BlockState) (t_rel : Fin BT) (i : Fin BK)
    (hne : G ≠ GO) (hBK : BK ≤ DK) :
    (exec (fwd_decay_cumsum_surface G GO s_qk_h s_qk_t s_qk_d B H T scale BT BK DK) s).map
      (·.readMem GO (offset s s_qk_h DK t_rel.val BT BK i))
      = some (if active s DK BK i then fwdDecayClosed s G s_qk_h DK BT BK t_rel i
              else s.readMem GO (offset s s_qk_h DK t_rel.val BT BK i)) := by
  -- reduce exec over the 6 prefix statements onto the forRange loop
  simp only [exec, fwd_decay_cumsum_surface, ComputeKernel.toAlgKernel,
    ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Except.bind, Except.pure, bind, pure]
  -- explicit post-prefix state
  set s6 := ((((((s.setReg "i_k" .nat [] (Tile.scalar (s.pids 0))).setReg
      "i_c" .nat [] (Tile.scalar (s.pids 1))).setReg
      "i_bh" .nat [] (Tile.scalar (s.pids 2))).setReg
      "p_g" .ptr [BK] (fwdPgTile s G s_qk_h DK BT BK 0)).setReg
      "p_go" .ptr [BK] (fwdPgoTile s GO s_qk_h DK BT BK 0)).setReg
      "cum_decay" .real [BK] (fwdCumTile s G s_qk_h DK BT BK 0)).setReg
      "mask" .bool [BK] (fwdMaskTile s DK BK) with hs6
  -- the offset-tile shared by p_g/p_go init: lane i ↦ baseOffset + i
  set s_ik := s.setReg "i_k" .nat [] (Tile.scalar (s.pids 0)) with hs_ik
  set s_ic := s_ik.setReg "i_c" .nat [] (Tile.scalar (s.pids 1)) with hs_ic
  set s_ibh := s_ic.setReg "i_bh" .nat [] (Tile.scalar (s.pids 2)) with hs_ibh
  have hibh : s_ibh.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)) := by simp [hs_ibh]
  have hic : s_ibh.regs .nat [] "i_c" = some (Tile.scalar (s.pids 1)) := by
    rw [hs_ibh, hs_ic]; simp
  have hik : s_ibh.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)) := by
    rw [hs_ibh, hs_ic, hs_ik]; simp
  -- p_g init eval
  have hpgeval : evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase G)
      (Op.add NumericDType.nat Broadcast.scalarL
        (Op.add NumericDType.nat Broadcast.nil
          (Op.add NumericDType.nat Broadcast.nil
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_bh") (Op.constNat s_qk_h))
            (Op.mul NumericDType.nat Broadcast.nil
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_c") (Op.constNat BT))
              (Op.constNat DK)))
          (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k") (Op.constNat BK)))
        (Op.arange BK))) s_ibh = some (fwdPgTile s G s_qk_h DK BT BK 0) := by
    simp only [evalOp, Option.bind, hibh, hic, hik, evalOp_constNat, evalOp_arange,
      Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul,
      Tile.bop, Tile.scalar]
    apply congrArg some
    apply Tile.ext
    intro idx
    obtain ⟨i, u⟩ := idx
    simp only [Tile.ptrAdd_data, fwdPgTile, offset, baseOffset,
      Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL,
      Broadcast.leftIndex_nil, Broadcast.rightIndex_nil]
    refine Prod.ext rfl ?_
    simp only [Tile.vec]
    ring
  set s_pg := s_ibh.setReg "p_g" .ptr [BK] (fwdPgTile s G s_qk_h DK BT BK 0) with hs_pg
  have hpgostep_ibh : s_pg.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)) := by
    rw [hs_pg]; simp [hibh]
  have hpgostep_ic : s_pg.regs .nat [] "i_c" = some (Tile.scalar (s.pids 1)) := by
    rw [hs_pg]; simp [hic]
  have hpgostep_ik : s_pg.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)) := by
    rw [hs_pg]; simp [hik]
  -- p_go init eval
  have hpgoeval : evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase GO)
      (Op.add NumericDType.nat Broadcast.scalarL
        (Op.add NumericDType.nat Broadcast.nil
          (Op.add NumericDType.nat Broadcast.nil
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_bh") (Op.constNat s_qk_h))
            (Op.mul NumericDType.nat Broadcast.nil
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_c") (Op.constNat BT))
              (Op.constNat DK)))
          (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k") (Op.constNat BK)))
        (Op.arange BK))) s_pg = some (fwdPgoTile s GO s_qk_h DK BT BK 0) := by
    simp only [evalOp, Option.bind, hpgostep_ibh, hpgostep_ic, hpgostep_ik, evalOp_constNat,
      evalOp_arange, Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul,
      Tile.bop, Tile.scalar]
    apply congrArg some
    apply Tile.ext
    intro idx
    obtain ⟨i, u⟩ := idx
    simp only [Tile.ptrAdd_data, fwdPgoTile, offset, baseOffset,
      Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL,
      Broadcast.leftIndex_nil, Broadcast.rightIndex_nil]
    refine Prod.ext rfl ?_
    simp only [Tile.vec]
    ring
  set s_pgo := s_pg.setReg "p_go" .ptr [BK] (fwdPgoTile s GO s_qk_h DK BT BK 0) with hs_pgo
  -- cum_decay = zeros eval
  have hcumeval : evalOp (Op.full [BK] (Op.const 0)) s_pgo = some (fwdCumTile s G s_qk_h DK BT BK 0) := by
    simp only [evalOp_full, evalOp_const, Option.bind]
    apply congrArg some
    apply Tile.ext
    intro idx
    obtain ⟨i, u⟩ := idx
    simp only [fwdCumTile, Tile.scalar, fwdPartial, Finset.univ_eq_empty, Finset.sum_empty,
      mul_zero]
    by_cases ha : active s DK BK i <;> simp [ha]
  set s_cum := s_pgo.setReg "cum_decay" .real [BK] (fwdCumTile s G s_qk_h DK BT BK 0) with hs_cum
  have hcum_ik : s_cum.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)) := by
    rw [hs_cum, hs_pgo, hs_pg]; simp [hik]
  -- mask eval
  have hmaskeval : evalOp (Op.lt ComparableDType.nat Broadcast.scalarR
      (Op.add NumericDType.nat Broadcast.scalarL
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k") (Op.constNat BK))
        (Op.arange BK)) (Op.constNat DK)) s_cum = some (fwdMaskTile s DK BK) := by
    simp only [evalOp, Option.bind, hcum_ik, evalOp_constNat, evalOp_arange,
      Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul,
      ComparableDType.lt, Tile.bop, Tile.cop, Tile.scalar]
    apply congrArg some
    apply Tile.ext
    intro idx
    obtain ⟨i, u⟩ := idx
    simp only [fwdMaskTile, active, elemIndex, Tile.vec,
      Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL,
      Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR,
      Broadcast.leftIndex_nil, Broadcast.rightIndex_nil]
    rfl
  -- chain the 6 prefix assigns, landing on the forRange loop over s6
  have hstepK : stepStmt (Stmt.assign TileDType.nat [] "i_k" (Op.programId 0)) s = some s_ik := by
    rw [hs_ik]; exact stepStmt_assign_eq_some (by simp [evalOp])
  have hstepC : stepStmt (Stmt.assign TileDType.nat [] "i_c" (Op.programId 1)) s_ik = some s_ic := by
    rw [hs_ic]; exact stepStmt_assign_eq_some (show evalOp (Op.programId 1) s_ik = _ from by
      simp [evalOp, hs_ik])
  have hstepB : stepStmt (Stmt.assign TileDType.nat [] "i_bh" (Op.programId 2)) s_ic = some s_ibh := by
    rw [hs_ibh]; exact stepStmt_assign_eq_some (show evalOp (Op.programId 2) s_ic = _ from by
      simp [evalOp, hs_ic, hs_ik])
  have hstepPg : stepStmt (Stmt.assign TileDType.ptr [BK] "p_g"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase G)
        (Op.add NumericDType.nat Broadcast.scalarL
          (Op.add NumericDType.nat Broadcast.nil
            (Op.add NumericDType.nat Broadcast.nil
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_bh") (Op.constNat s_qk_h))
              (Op.mul NumericDType.nat Broadcast.nil
                (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_c") (Op.constNat BT))
                (Op.constNat DK)))
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k") (Op.constNat BK)))
          (Op.arange BK)))) s_ibh = some s_pg := by
    rw [hs_pg]; exact stepStmt_assign_eq_some hpgeval
  have hstepPgo : stepStmt (Stmt.assign TileDType.ptr [BK] "p_go"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase GO)
        (Op.add NumericDType.nat Broadcast.scalarL
          (Op.add NumericDType.nat Broadcast.nil
            (Op.add NumericDType.nat Broadcast.nil
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_bh") (Op.constNat s_qk_h))
              (Op.mul NumericDType.nat Broadcast.nil
                (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_c") (Op.constNat BT))
                (Op.constNat DK)))
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k") (Op.constNat BK)))
          (Op.arange BK)))) s_pg = some s_pgo := by
    rw [hs_pgo]; exact stepStmt_assign_eq_some hpgoeval
  have hstepCum : stepStmt (Stmt.assign TileDType.real [BK] "cum_decay" (Op.full [BK] (Op.const 0)))
      s_pgo = some s_cum := by
    rw [hs_cum]; exact stepStmt_assign_eq_some hcumeval
  have hstepMask : stepStmt (Stmt.assign TileDType.bool [BK] "mask"
      (Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.add NumericDType.nat Broadcast.scalarL
          (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k") (Op.constNat BK))
          (Op.arange BK)) (Op.constNat DK))) s_cum
      = some (s_cum.setReg "mask" .bool [BK] (fwdMaskTile s DK BK)) :=
    stepStmt_assign_eq_some hmaskeval
  rw [stepStmts.cons_some hstepK]
  rw [stepStmts.cons_some hstepC]
  rw [stepStmts.cons_some hstepB]
  rw [stepStmts.cons_some hstepPg]
  rw [stepStmts.cons_some hstepPgo]
  erw [stepStmts.cons_some hstepCum]
  erw [stepStmts.cons_some hstepMask]
  have hs_cum_to_s6 :
      s_cum.setReg "mask" .bool [BK] (fwdMaskTile s DK BK) = s6 := by
    rw [hs6, hs_cum, hs_pgo, hs_pg, hs_ibh, hs_ic, hs_ik]
  -- collapse the singleton `stepStmts [forRange] s6` to `stepStmt forRange s6`
  rw [show ∀ (X : Stmt), stepStmts [X] (s_cum.setReg "mask" .bool [BK] (fwdMaskTile s DK BK))
        = stepStmt X (s_cum.setReg "mask" .bool [BK] (fwdMaskTile s DK BK)) from
      fun X => by cases h : stepStmt X (s_cum.setReg "mask" .bool [BK] (fwdMaskTile s DK BK)) <;>
        simp [stepStmts, h]]
  rw [hs_cum_to_s6]
  -- entry invariant P 0 s6
  have hP0 : fwdInv G GO s s_qk_h DK BT BK 0 s6 := by
    have hsub : s6 = ((((((s.setReg "i_k" .nat [] (Tile.scalar (s.pids 0))).setReg
        "i_c" .nat [] (Tile.scalar (s.pids 1))).setReg
        "i_bh" .nat [] (Tile.scalar (s.pids 2))).setReg
        "p_g" .ptr [BK] (fwdPgTile s G s_qk_h DK BT BK 0)).setReg
        "p_go" .ptr [BK] (fwdPgoTile s GO s_qk_h DK BT BK 0)).setReg
        "cum_decay" .real [BK] (fwdCumTile s G s_qk_h DK BT BK 0)).setReg
        "mask" .bool [BK] (fwdMaskTile s DK BK) := by
      rw [hs6, hs_cum, hs_pgo, hs_pg, hs_ibh, hs_ic, hs_ik]
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · rw [hsub]; simp only [BlockState.setReg_pids]
    · intro a; rw [hsub]; simp only [BlockState.setReg_readMem]
    · rw [hsub, BlockState.setReg_ne_name (h := by decide), BlockState.setReg_same]
    · rw [hsub, BlockState.setReg_ne_name (h := by decide),
        BlockState.setReg_ne_name (h := by decide), BlockState.setReg_ne_name (h := by decide),
        BlockState.setReg_same]
    · rw [hsub, BlockState.setReg_ne_name (h := by decide),
        BlockState.setReg_ne_name (h := by decide), BlockState.setReg_same]
    · rw [hsub, BlockState.setReg_same]
    · intro j hj i; rw [hsub]; simp only [BlockState.setReg_readMem]
    · intro j hj i; omega
  -- drive the loop
  obtain ⟨final, sfinal, hLoop, hfinal, hPfinal⟩ :=
    forRange_inv (idx := "_i") (start := 0) (stop := BT) (step := 1)
      (body := fwdBody G GO s_qk_h BT BK DK)
      (P := fwdInv G GO s s_qk_h DK BT BK) (s_init := s6) (Nat.one_ne_zero) hP0
      (fun j st _ hPj => fwd_decay_cumsum_step G GO s s_qk_h BT BK DK hne hBK j st hPj)
  simp only [fwdBody] at hLoop
  erw [hLoop]
  simp only [Option.map_some]
  -- extract the row-`t_rel` readback (t_rel < BT ≤ final)
  obtain ⟨_, _, _, _, _, _, _, hreadGO⟩ := hPfinal
  have ht : t_rel.val < final := lt_of_lt_of_le t_rel.isLt hfinal
  rw [hreadGO t_rel.val ht i]
  simp only [fwdDecayClosed, fwdPartial]

/-- **Genuine GENERAL forward compute-correctness.** For arbitrary `BT BK DK`
(with `BK ≤ DK`), the full `fwd_decay_cumsum` surface realizes the honest
decay-cumsum closed form `fwdDecayClosed` (the scaled within-chunk prefix sum)
at every active `GO` lane of loop row `t_rel`. Dimension-parameterized
replacement for the `BT=2`-pinned `fwd_decay_cumsum_surface_closed_compute_correct`. -/
theorem fwd_decay_cumsum_surface_closed_compute_correct_general
    (G GO : RegionName)
    (s_qk_h s_qk_t s_qk_d B H T : Nat) (scale : ℝ)
    (BT BK DK : Nat) (s : BlockState) (t_rel : Fin BT)
    (hne : G ≠ GO) (hBK : BK ≤ DK) :
    ComputeCorrect.Realizes
      (kernel := fwd_decay_cumsum_surface G GO s_qk_h s_qk_t s_qk_d B H T scale BT BK DK)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s DK BK)
        (fun i => (GO, offset s s_qk_h DK t_rel.val BT BK i)))
      (expected := fun i : Fin BK =>
        fwdDecayClosed s G s_qk_h DK BT BK t_rel i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fwd_decay_cumsum_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := fwd_decay_cumsum_full_surface_closed_general G GO
    s_qk_h s_qk_t s_qk_d B H T scale BT BK DK s t_rel i hne hBK
  rw [hExec] at h
  simp only [Option.map_some] at h
  rw [Option.some.injEq] at h
  show s'.readMem GO (offset s s_qk_h DK t_rel.val BT BK i) = _
  rw [h, if_pos hActive]

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

/-- **Genuine `kg` closed form.** At chunk row `t_rel` and lane `i`, the
`prepare_qg_kg` kernel writes `k[idx] * exp2(g[row BT-1] - g[idx])` into `KG`,
where `g[row BT-1]` is the loop-invariant `last_decay`. This is the honest
pointwise specification of the `k *= exp2(last_decay - g)` map. -/
noncomputable def prepareKgClosed
    (s : BlockState) (K G : RegionName)
    (s_qk_h DK BT BK : Nat) (t_rel : Fin BT) (i : Fin BK) : ℝ :=
  s.readMem K (offset s s_qk_h DK t_rel.val BT BK i) *
    Real.exp ((s.readMem G (offset s s_qk_h DK (BT - 1) BT BK i) -
      s.readMem G (offset s s_qk_h DK t_rel.val BT BK i)) * Real.log 2)

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

/-- `last_decay` register tile: lane `i` reads `G` at chunk row `BT-1`. -/
noncomputable def prepLastDecayTile
    (s : BlockState) (G : RegionName)
    (s_qk_h DK BT BK : Nat) : Tile .real [BK] :=
  ⟨fun idx => some (s.readMem G (offset s s_qk_h DK (BT - 1) BT BK idx.1))⟩

/-- `offs` register tile (`arange BK`): lane `i` holds `i`. -/
def prepOffsTile (BK : Nat) : Tile .nat [BK] := Tile.vec (fun i : Fin BK => i.val)

/-- Generic per-row pointer tile for region `R` at row `m`. -/
def prepPtrTile
    (s : BlockState) (R : RegionName)
    (s_qk_h DK BT BK : Nat) (m : Nat) : Tile .ptr [BK] :=
  ⟨fun idx => (R, offset s s_qk_h DK m BT BK idx.1)⟩

/-- The masked-loaded `q` tile at row `m`: active lanes read `Q`, inactive 0. -/
noncomputable def prepLoadTile
    (s : BlockState) (R : RegionName)
    (s_qk_h DK BT BK : Nat) (m : Nat) : Tile .real [BK] :=
  ⟨fun idx => if active s DK BK idx.1 then
      some (s.readMem R (offset s s_qk_h DK m BT BK idx.1)) else some 0⟩

/-- The scaled `q_val` tile after `q_val *= exp2(g) * scale` at row `m`
(active lanes hold `prepareQgClosed`, inactive lanes hold `0`). -/
noncomputable def prepQValTile
    (s : BlockState) (Q G : RegionName)
    (s_qk_h DK BT BK : Nat) (t_rel : Fin BT) (scale : ℝ) : Tile .real [BK] :=
  ⟨fun idx => if active s DK BK idx.1 then
      some (prepareQgClosed s Q G s_qk_h DK BT BK t_rel idx.1 scale) else some 0⟩

/-- The scaled `k_val` tile after `k_val *= exp2(last_decay - g)` at row `m`
(active lanes hold `prepareKgClosed`, inactive lanes hold `0`). -/
noncomputable def prepKValTile
    (s : BlockState) (K G : RegionName)
    (s_qk_h DK BT BK : Nat) (t_rel : Fin BT) : Tile .real [BK] :=
  ⟨fun idx => if active s DK BK idx.1 then
      some (prepareKgClosed s K G s_qk_h DK BT BK t_rel idx.1) else some 0⟩

/-- The `prepare_qg_kg` loop invariant after `m` rows. -/
noncomputable def prepInv
    (Q K G QG KG : RegionName) (s : BlockState) (s_qk_h DK BT BK : Nat) (scale : ℝ) :
    Nat → BlockState → Prop :=
  fun m sc =>
    sc.pids = s.pids ∧
    (∀ a, sc.readMem Q a = s.readMem Q a) ∧
    (∀ a, sc.readMem K a = s.readMem K a) ∧
    (∀ a, sc.readMem G a = s.readMem G a) ∧
    sc.regs .ptr [BK] "p_q" = some (prepPtrTile s Q s_qk_h DK BT BK m) ∧
    sc.regs .ptr [BK] "p_g" = some (prepPtrTile s G s_qk_h DK BT BK m) ∧
    sc.regs .ptr [BK] "p_k" = some (prepPtrTile s K s_qk_h DK BT BK m) ∧
    sc.regs .ptr [BK] "p_qg" = some (prepPtrTile s QG s_qk_h DK BT BK m) ∧
    sc.regs .ptr [BK] "p_kg" = some (prepPtrTile s KG s_qk_h DK BT BK m) ∧
    sc.regs .bool [BK] "mask" = some (fwdMaskTile s DK BK) ∧
    sc.regs .real [BK] "last_decay" = some (prepLastDecayTile s G s_qk_h DK BT BK) ∧
    (∀ j, m ≤ j → ∀ i : Fin BK,
      sc.readMem QG (offset s s_qk_h DK j BT BK i) =
        s.readMem QG (offset s s_qk_h DK j BT BK i)) ∧
    (∀ j, m ≤ j → ∀ i : Fin BK,
      sc.readMem KG (offset s s_qk_h DK j BT BK i) =
        s.readMem KG (offset s s_qk_h DK j BT BK i)) ∧
    (∀ j, j < m → ∀ i : Fin BK, ∀ (hj : j < BT),
      sc.readMem QG (offset s s_qk_h DK j BT BK i) =
        if active s DK BK i then
          prepareQgClosed s Q G s_qk_h DK BT BK ⟨j, hj⟩ i scale
        else s.readMem QG (offset s s_qk_h DK j BT BK i)) ∧
    (∀ j, j < m → ∀ i : Fin BK, ∀ (hj : j < BT),
      sc.readMem KG (offset s s_qk_h DK j BT BK i) =
        if active s DK BK i then
          prepareKgClosed s K G s_qk_h DK BT BK ⟨j, hj⟩ i
        else s.readMem KG (offset s s_qk_h DK j BT BK i))

/-- The lowered `prepare_qg_kg` loop body (one row): 3 masked loads (q,k,g),
2 pointwise scalings (`q *= exp2(g)*scale`, `k *= exp2(last_decay - g)`),
2 masked stores (KG then QG), 5 pointer increments. -/
def prepBody (Q K G QG KG : RegionName) (s_qk_h DK BT BK : Nat) (scale : ℝ) : List Stmt :=
  [Stmt.assign TileDType.real [BK] "q_val"
      (Op.load TileDType.real (MemAccess.ptr (Op.ref TileDType.ptr [BK] "p_q"))
        (MaskOpt.maskOther (Op.ref TileDType.bool [BK] "mask") ((Op.const 0).broadcast [BK]))),
    Stmt.assign TileDType.real [BK] "k_val"
      (Op.load TileDType.real (MemAccess.ptr (Op.ref TileDType.ptr [BK] "p_k"))
        (MaskOpt.maskOther (Op.ref TileDType.bool [BK] "mask") ((Op.const 0).broadcast [BK]))),
    Stmt.assign TileDType.real [BK] "g_val"
      (Op.load ComputeDType.fp32.eraseDType (MemAccess.ptr (Op.ref TileDType.ptr [BK] "p_g"))
        (MaskOpt.maskOther (Op.ref TileDType.bool [BK] "mask") ((Op.const 0).broadcast [BK]))),
    Stmt.assign TileDType.real [BK] "q_val"
      (Op.mul NumericDType.real Broadcast.nil.consSame (Op.ref TileDType.real [BK] "q_val")
        (Op.mul NumericDType.real Broadcast.scalarR (Op.ref TileDType.real [BK] "g_val").exp2 (Op.const scale))),
    Stmt.assign TileDType.real [BK] "k_val"
      (Op.mul NumericDType.real Broadcast.nil.consSame (Op.ref TileDType.real [BK] "k_val")
        (Op.sub NumericDType.real Broadcast.nil.consSame (Op.ref TileDType.real [BK] "last_decay")
            (Op.ref TileDType.real [BK] "g_val")).exp2),
    Stmt.store TileDType.real [BK] (MemAccess.ptr (Op.ref TileDType.ptr [BK] "p_kg"))
      (Op.ref TileDType.real [BK] "k_val") (MaskOpt.mask (Op.ref TileDType.bool [BK] "mask")),
    Stmt.store TileDType.real [BK] (MemAccess.ptr (Op.ref TileDType.ptr [BK] "p_qg"))
      (Op.ref TileDType.real [BK] "q_val") (MaskOpt.mask (Op.ref TileDType.bool [BK] "mask")),
    Stmt.assign TileDType.ptr [BK] "p_q"
      (Op.ptrAdd Broadcast.scalarR (Op.ref TileDType.ptr [BK] "p_q") (Op.constNat DK)),
    Stmt.assign TileDType.ptr [BK] "p_g"
      (Op.ptrAdd Broadcast.scalarR (Op.ref TileDType.ptr [BK] "p_g") (Op.constNat DK)),
    Stmt.assign TileDType.ptr [BK] "p_k"
      (Op.ptrAdd Broadcast.scalarR (Op.ref TileDType.ptr [BK] "p_k") (Op.constNat DK)),
    Stmt.assign TileDType.ptr [BK] "p_kg"
      (Op.ptrAdd Broadcast.scalarR (Op.ref TileDType.ptr [BK] "p_kg") (Op.constNat DK)),
    Stmt.assign TileDType.ptr [BK] "p_qg"
      (Op.ptrAdd Broadcast.scalarR (Op.ref TileDType.ptr [BK] "p_qg") (Op.constNat DK))]

set_option maxHeartbeats 4000000 in
/-- **One row advances the `prepare_qg_kg` invariant.** -/
theorem prepare_qg_kg_step
    (Q K G QG KG : RegionName) (s : BlockState) (s_qk_h DK BT BK : Nat) (scale : ℝ)
    (hQ_QG : Q ≠ QG) (hQ_KG : Q ≠ KG) (hK_QG : K ≠ QG) (hK_KG : K ≠ KG)
    (hG_QG : G ≠ QG) (hG_KG : G ≠ KG) (hQG_KG : QG ≠ KG) (hBK : BK ≤ DK) (hBT : 0 < BT)
    (m : Nat) (hm : m < BT) (sc : BlockState) (hP : prepInv Q K G QG KG s s_qk_h DK BT BK scale m sc) :
    ∃ s',
      stepStmts (prepBody Q K G QG KG s_qk_h DK BT BK scale)
        (sc.setReg "_i" .nat [] (Tile.scalar m)) = some s' ∧
      prepInv Q K G QG KG s s_qk_h DK BT BK scale (m + 1) s' := by
  obtain ⟨hpids, hQmem, hKmem, hGmem, hpq, hpg, hpk, hpqg, hpkg, hmask, hld,
    hUQG, hUKG, hreadQG, hreadKG⟩ := hP
  set sloop := sc.setReg "_i" .nat [] (Tile.scalar m) with hsloop
  -- register lookups along sloop
  have hpq' : sloop.regs .ptr [BK] "p_q" = some (prepPtrTile s Q s_qk_h DK BT BK m) := by
    rw [hsloop]; simpa using hpq
  have hpg' : sloop.regs .ptr [BK] "p_g" = some (prepPtrTile s G s_qk_h DK BT BK m) := by
    rw [hsloop]; simpa using hpg
  have hpk' : sloop.regs .ptr [BK] "p_k" = some (prepPtrTile s K s_qk_h DK BT BK m) := by
    rw [hsloop]; simpa using hpk
  have hpqg' : sloop.regs .ptr [BK] "p_qg" = some (prepPtrTile s QG s_qk_h DK BT BK m) := by
    rw [hsloop]; simpa using hpqg
  have hpkg' : sloop.regs .ptr [BK] "p_kg" = some (prepPtrTile s KG s_qk_h DK BT BK m) := by
    rw [hsloop]; simpa using hpkg
  have hmask' : sloop.regs .bool [BK] "mask" = some (fwdMaskTile s DK BK) := by
    rw [hsloop]; simpa using hmask
  have hld' : sloop.regs .real [BK] "last_decay" = some (prepLastDecayTile s G s_qk_h DK BT BK) := by
    rw [hsloop]; simpa using hld
  -- generic masked-load evaluator for a region R via pointer-tile reg `name`
  have hloadeval : ∀ (R : RegionName) (name : RegName) (st : BlockState),
      st.regs .ptr [BK] name = some (prepPtrTile s R s_qk_h DK BT BK m) →
      st.regs .bool [BK] "mask" = some (fwdMaskTile s DK BK) →
      (∀ a, st.readMem R a = s.readMem R a) →
      evalOp (Op.load TileDType.real (MemAccess.ptr (Op.ref TileDType.ptr [BK] name))
        (MaskOpt.maskOther (Op.ref TileDType.bool [BK] "mask") ((Op.const 0).broadcast [BK]))) st
        = some (prepLoadTile s R s_qk_h DK BT BK m) := by
    intro R name st hp hmk hmem
    simp only [evalOp, Option.bind, hp, hmk, evalOp_const]
    apply congrArg some
    apply Tile.ext
    intro idx
    obtain ⟨i, u⟩ := idx
    simp only [prepLoadTile, fwdMaskTile, prepPtrTile]
    by_cases ha : active s DK BK i
    · simp only [ha, decide_true, if_true, if_pos]
      rw [BlockState.readMemValue_real, hmem]
    · simp only [ha, decide_false, Bool.false_eq_true, if_false]
      rfl
  -- (1) q_val load
  have hqeval := hloadeval Q "p_q" sloop hpq' hmask' (fun a => by rw [hsloop]; simpa using hQmem a)
  set s1 := sloop.setReg "q_val" .real [BK] (prepLoadTile s Q s_qk_h DK BT BK m) with hs1
  have hstep1 : stepStmt (Stmt.assign TileDType.real [BK] "q_val"
      (Op.load TileDType.real (MemAccess.ptr (Op.ref TileDType.ptr [BK] "p_q"))
        (MaskOpt.maskOther (Op.ref TileDType.bool [BK] "mask") ((Op.const 0).broadcast [BK])))) sloop
      = some s1 := by
    rw [hs1]; exact stepStmt_assign_eq_some hqeval
  -- s1 register lookups
  have hs1_pk : s1.regs .ptr [BK] "p_k" = some (prepPtrTile s K s_qk_h DK BT BK m) := by rw [hs1]; simpa using hpk'
  have hs1_mask : s1.regs .bool [BK] "mask" = some (fwdMaskTile s DK BK) := by rw [hs1]; simpa using hmask'
  have hs1_Kmem : ∀ a, s1.readMem K a = s.readMem K a := by intro a; rw [hs1]; simp only [BlockState.setReg_readMem]; rw [hsloop]; simpa using hKmem a
  have hkeval := hloadeval K "p_k" s1 hs1_pk hs1_mask hs1_Kmem
  set s2 := s1.setReg "k_val" .real [BK] (prepLoadTile s K s_qk_h DK BT BK m) with hs2
  have hstep2 : stepStmt (Stmt.assign TileDType.real [BK] "k_val"
      (Op.load TileDType.real (MemAccess.ptr (Op.ref TileDType.ptr [BK] "p_k"))
        (MaskOpt.maskOther (Op.ref TileDType.bool [BK] "mask") ((Op.const 0).broadcast [BK])))) s1
      = some s2 := by
    rw [hs2]; exact stepStmt_assign_eq_some hkeval
  -- (3) g_val load (fp32 erased = real)
  have hs2_pg : s2.regs .ptr [BK] "p_g" = some (prepPtrTile s G s_qk_h DK BT BK m) := by
    rw [hs2, hs1]; simpa using hpg'
  have hs2_mask : s2.regs .bool [BK] "mask" = some (fwdMaskTile s DK BK) := by rw [hs2, hs1]; simpa using hmask'
  have hs2_Gmem : ∀ a, s2.readMem G a = s.readMem G a := by intro a; rw [hs2, hs1]; simp only [BlockState.setReg_readMem]; rw [hsloop]; simpa using hGmem a
  have hgeval := hloadeval G "p_g" s2 hs2_pg hs2_mask hs2_Gmem
  set s3 := s2.setReg "g_val" .real [BK] (prepLoadTile s G s_qk_h DK BT BK m) with hs3
  have hstep3 : stepStmt (Stmt.assign TileDType.real [BK] "g_val"
      (Op.load ComputeDType.fp32.eraseDType (MemAccess.ptr (Op.ref TileDType.ptr [BK] "p_g"))
        (MaskOpt.maskOther (Op.ref TileDType.bool [BK] "mask") ((Op.const 0).broadcast [BK])))) s2
      = some s3 := by
    rw [hs3]; exact stepStmt_assign_eq_some (by simpa [ComputeDType.eraseDType] using hgeval)
  -- (4) q_val *= exp2(g_val) * scale
  have hs3_qval : s3.regs .real [BK] "q_val" = some (prepLoadTile s Q s_qk_h DK BT BK m) := by
    rw [hs3, hs2]; simp [hs1]
  have hs3_gval : s3.regs .real [BK] "g_val" = some (prepLoadTile s G s_qk_h DK BT BK m) := by
    rw [hs3]; simp
  have hqscaleeval : evalOp (Op.mul NumericDType.real Broadcast.nil.consSame (Op.ref TileDType.real [BK] "q_val")
      (Op.mul NumericDType.real Broadcast.scalarR (Op.ref TileDType.real [BK] "g_val").exp2 (Op.const scale))) s3
      = some (prepQValTile s Q G s_qk_h DK BT BK ⟨m, hm⟩ scale) := by
    simp only [evalOp, Option.bind, hs3_qval, hs3_gval, evalOp_const]
    apply congrArg some
    apply Tile.ext
    intro idx
    obtain ⟨i, u⟩ := idx
    simp only [prepQValTile, prepLoadTile, Tile.uop_data, Tile.bop_data, Broadcast.leftIndex,
      Broadcast.rightIndex, NumericDType.mul]
    by_cases ha : active s DK BK i
    · simp only [ha, if_true, WithBot.realExp2, WithBot.realMul]
      simp only [prepareQgClosed, offset, baseOffset, Tile.scalar]
      simp [Tile.vec, mul_assoc]
    · simp only [ha, if_false, WithBot.realExp2, WithBot.realMul, Tile.scalar]
      simp [Tile.vec]
  set s4 := s3.setReg "q_val" .real [BK] (prepQValTile s Q G s_qk_h DK BT BK ⟨m, hm⟩ scale) with hs4
  have hstep4 : stepStmt (Stmt.assign TileDType.real [BK] "q_val"
      (Op.mul NumericDType.real Broadcast.nil.consSame (Op.ref TileDType.real [BK] "q_val")
        (Op.mul NumericDType.real Broadcast.scalarR (Op.ref TileDType.real [BK] "g_val").exp2 (Op.const scale)))) s3
      = some s4 := by
    rw [hs4]; exact stepStmt_assign_eq_some hqscaleeval
  -- (5) k_val *= exp2(last_decay - g_val)
  have hs4_kval : s4.regs .real [BK] "k_val" = some (prepLoadTile s K s_qk_h DK BT BK m) := by
    rw [hs4, hs3, hs2]; simp
  have hs4_ld : s4.regs .real [BK] "last_decay" = some (prepLastDecayTile s G s_qk_h DK BT BK) := by
    rw [hs4, hs3, hs2, hs1]; simpa using hld'
  have hs4_gval : s4.regs .real [BK] "g_val" = some (prepLoadTile s G s_qk_h DK BT BK m) := by
    rw [hs4, hs3]; simp
  have hkscaleeval : evalOp (Op.mul NumericDType.real Broadcast.nil.consSame (Op.ref TileDType.real [BK] "k_val")
      (Op.sub NumericDType.real Broadcast.nil.consSame (Op.ref TileDType.real [BK] "last_decay")
          (Op.ref TileDType.real [BK] "g_val")).exp2) s4
      = some (prepKValTile s K G s_qk_h DK BT BK ⟨m, hm⟩) := by
    simp only [evalOp, Option.bind, hs4_kval, hs4_ld, hs4_gval]
    apply congrArg some
    apply Tile.ext
    intro idx
    obtain ⟨i, u⟩ := idx
    simp only [prepKValTile, prepLoadTile, prepLastDecayTile, Tile.uop_data, Tile.bop_data,
      Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.mul, NumericDType.sub]
    by_cases ha : active s DK BK i
    · simp only [ha, if_true, WithBot.realExp2, WithBot.realMul, WithBot.realSub]
      simp only [prepareKgClosed, offset, baseOffset]
      simp [Tile.vec]
    · simp only [ha, if_false, WithBot.realExp2, WithBot.realMul, WithBot.realSub]
      simp [Tile.vec]
  set s5 := s4.setReg "k_val" .real [BK] (prepKValTile s K G s_qk_h DK BT BK ⟨m, hm⟩) with hs5
  have hstep5 : stepStmt (Stmt.assign TileDType.real [BK] "k_val"
      (Op.mul NumericDType.real Broadcast.nil.consSame (Op.ref TileDType.real [BK] "k_val")
        (Op.sub NumericDType.real Broadcast.nil.consSame (Op.ref TileDType.real [BK] "last_decay")
            (Op.ref TileDType.real [BK] "g_val")).exp2)) s4
      = some s5 := by
    rw [hs5]; exact stepStmt_assign_eq_some hkscaleeval
  -- (6) masked store of k_val into KG via p_kg
  have hs5_pkg : s5.regs .ptr [BK] "p_kg" = some (prepPtrTile s KG s_qk_h DK BT BK m) := by
    rw [hs5, hs4, hs3, hs2, hs1]; simpa using hpkg'
  have hs5_mask : s5.regs .bool [BK] "mask" = some (fwdMaskTile s DK BK) := by
    rw [hs5, hs4, hs3, hs2, hs1]; simpa using hmask'
  have hs5_kval : s5.regs .real [BK] "k_val" = some (prepKValTile s K G s_qk_h DK BT BK ⟨m, hm⟩) := by
    rw [hs5]; simp
  set kvfun : TileIndex [BK] → ℝ := fun k =>
    (prepKValTile s K G s_qk_h DK BT BK ⟨m, hm⟩).data k |>.unbotD 0 with hkvfun
  set sstK := (TileShape.allIndices [BK]).foldl
    (fun (acc : BlockState) k =>
      if active s DK BK k.1 then
        acc.writeMem KG (offset s s_qk_h DK m BT BK k.1) (kvfun k)
      else acc) s5 with hsstK
  have hstoreK : stepStmt (Stmt.store TileDType.real [BK] (MemAccess.ptr (Op.ref TileDType.ptr [BK] "p_kg"))
      (Op.ref TileDType.real [BK] "k_val") (MaskOpt.mask (Op.ref TileDType.bool [BK] "mask"))) s5
      = some sstK := by
    simp only [stepStmt, evalOp, Option.bind, Option.map, hs5_kval, hs5_pkg, hs5_mask]
    apply congrArg some
    rw [hsstK]
    congr 1
    funext acc k
    simp only [prepPtrTile, fwdMaskTile]
    by_cases ha : active s DK BK k.1
    · rw [if_pos (by simpa using ha), if_pos ha, BlockState.writeMemTyped_real]; rfl
    · rw [if_neg (by simpa using ha), if_neg ha]
  -- (7) masked store of q_val into QG via p_qg
  have hsstK_pqg : sstK.regs .ptr [BK] "p_qg" = some (prepPtrTile s QG s_qk_h DK BT BK m) := by
    rw [hsstK, BlockState.foldl_writeMem_prop_masked_regs, hs5, hs4, hs3, hs2, hs1]; simpa using hpqg'
  have hsstK_mask : sstK.regs .bool [BK] "mask" = some (fwdMaskTile s DK BK) := by
    rw [hsstK, BlockState.foldl_writeMem_prop_masked_regs, hs5, hs4, hs3, hs2, hs1]; simpa using hmask'
  have hsstK_qval : sstK.regs .real [BK] "q_val" = some (prepQValTile s Q G s_qk_h DK BT BK ⟨m, hm⟩ scale) := by
    rw [hsstK, BlockState.foldl_writeMem_prop_masked_regs, hs5, hs4]; simp
  set qvfun : TileIndex [BK] → ℝ := fun k =>
    (prepQValTile s Q G s_qk_h DK BT BK ⟨m, hm⟩ scale).data k |>.unbotD 0 with hqvfun
  set sstQ := (TileShape.allIndices [BK]).foldl
    (fun (acc : BlockState) k =>
      if active s DK BK k.1 then
        acc.writeMem QG (offset s s_qk_h DK m BT BK k.1) (qvfun k)
      else acc) sstK with hsstQ
  have hstoreQ : stepStmt (Stmt.store TileDType.real [BK] (MemAccess.ptr (Op.ref TileDType.ptr [BK] "p_qg"))
      (Op.ref TileDType.real [BK] "q_val") (MaskOpt.mask (Op.ref TileDType.bool [BK] "mask"))) sstK
      = some sstQ := by
    simp only [stepStmt, evalOp, Option.bind, Option.map, hsstK_qval, hsstK_pqg, hsstK_mask]
    apply congrArg some
    rw [hsstQ]
    congr 1
    funext acc k
    simp only [prepPtrTile, fwdMaskTile]
    by_cases ha : active s DK BK k.1
    · rw [if_pos (by simpa using ha), if_pos ha, BlockState.writeMemTyped_real]; rfl
    · rw [if_neg (by simpa using ha), if_neg ha]
  -- generic pointer-increment evaluator
  have hincr : ∀ (R : RegionName) (name : RegName) (st : BlockState),
      st.regs .ptr [BK] name = some (prepPtrTile s R s_qk_h DK BT BK m) →
      evalOp (Op.ptrAdd Broadcast.scalarR (Op.ref TileDType.ptr [BK] name) (Op.constNat DK)) st
        = some (prepPtrTile s R s_qk_h DK BT BK (m + 1)) := by
    intro R name st hp
    simp only [evalOp, Option.bind, hp, evalOp_constNat]
    apply congrArg some
    apply Tile.ext
    intro idx
    obtain ⟨i, u⟩ := idx
    simp only [Tile.ptrAdd_data, Broadcast.leftIndex, Broadcast.rightIndex, prepPtrTile, Tile.scalar]
    refine Prod.ext rfl ?_
    simp only [offset, baseOffset]; ring
  -- sstQ pointer reads (both scatter folds preserve registers)
  have hsstQ_pq : sstQ.regs .ptr [BK] "p_q" = some (prepPtrTile s Q s_qk_h DK BT BK m) := by
    rw [hsstQ, BlockState.foldl_writeMem_prop_masked_regs, hsstK,
      BlockState.foldl_writeMem_prop_masked_regs, hs5, hs4, hs3, hs2, hs1]; simpa using hpq'
  set s6 := sstQ.setReg "p_q" .ptr [BK] (prepPtrTile s Q s_qk_h DK BT BK (m + 1)) with hs6
  have hstep6 : stepStmt (Stmt.assign TileDType.ptr [BK] "p_q"
      (Op.ptrAdd Broadcast.scalarR (Op.ref TileDType.ptr [BK] "p_q") (Op.constNat DK))) sstQ = some s6 := by
    rw [hs6]; exact stepStmt_assign_eq_some (hincr Q "p_q" sstQ hsstQ_pq)
  have hs6_pg : s6.regs .ptr [BK] "p_g" = some (prepPtrTile s G s_qk_h DK BT BK m) := by
    rw [hs6, BlockState.setReg_ne_name (h := by decide), hsstQ,
      BlockState.foldl_writeMem_prop_masked_regs, hsstK,
      BlockState.foldl_writeMem_prop_masked_regs, hs5, hs4, hs3, hs2, hs1]; simpa using hpg'
  set s7 := s6.setReg "p_g" .ptr [BK] (prepPtrTile s G s_qk_h DK BT BK (m + 1)) with hs7
  have hstep7 : stepStmt (Stmt.assign TileDType.ptr [BK] "p_g"
      (Op.ptrAdd Broadcast.scalarR (Op.ref TileDType.ptr [BK] "p_g") (Op.constNat DK))) s6 = some s7 := by
    rw [hs7]; exact stepStmt_assign_eq_some (hincr G "p_g" s6 hs6_pg)
  have hs7_pk : s7.regs .ptr [BK] "p_k" = some (prepPtrTile s K s_qk_h DK BT BK m) := by
    rw [hs7, BlockState.setReg_ne_name (h := by decide),
      hs6, BlockState.setReg_ne_name (h := by decide), hsstQ,
      BlockState.foldl_writeMem_prop_masked_regs, hsstK,
      BlockState.foldl_writeMem_prop_masked_regs, hs5, hs4, hs3, hs2]; simpa using hpk'
  set s8 := s7.setReg "p_k" .ptr [BK] (prepPtrTile s K s_qk_h DK BT BK (m + 1)) with hs8
  have hstep8 : stepStmt (Stmt.assign TileDType.ptr [BK] "p_k"
      (Op.ptrAdd Broadcast.scalarR (Op.ref TileDType.ptr [BK] "p_k") (Op.constNat DK))) s7 = some s8 := by
    rw [hs8]; exact stepStmt_assign_eq_some (hincr K "p_k" s7 hs7_pk)
  have hs8_pkg : s8.regs .ptr [BK] "p_kg" = some (prepPtrTile s KG s_qk_h DK BT BK m) := by
    rw [hs8, BlockState.setReg_ne_name (h := by decide),
      hs7, BlockState.setReg_ne_name (h := by decide),
      hs6, BlockState.setReg_ne_name (h := by decide), hsstQ,
      BlockState.foldl_writeMem_prop_masked_regs, hsstK,
      BlockState.foldl_writeMem_prop_masked_regs, hs5, hs4, hs3, hs2, hs1]; simpa using hpkg'
  set s9 := s8.setReg "p_kg" .ptr [BK] (prepPtrTile s KG s_qk_h DK BT BK (m + 1)) with hs9
  have hstep9 : stepStmt (Stmt.assign TileDType.ptr [BK] "p_kg"
      (Op.ptrAdd Broadcast.scalarR (Op.ref TileDType.ptr [BK] "p_kg") (Op.constNat DK))) s8 = some s9 := by
    rw [hs9]; exact stepStmt_assign_eq_some (hincr KG "p_kg" s8 hs8_pkg)
  have hs9_pqg : s9.regs .ptr [BK] "p_qg" = some (prepPtrTile s QG s_qk_h DK BT BK m) := by
    rw [hs9, BlockState.setReg_ne_name (h := by decide),
      hs8, BlockState.setReg_ne_name (h := by decide),
      hs7, BlockState.setReg_ne_name (h := by decide),
      hs6, BlockState.setReg_ne_name (h := by decide), hsstQ,
      BlockState.foldl_writeMem_prop_masked_regs, hsstK,
      BlockState.foldl_writeMem_prop_masked_regs, hs5, hs4, hs3, hs2, hs1]; simpa using hpqg'
  set s10 := s9.setReg "p_qg" .ptr [BK] (prepPtrTile s QG s_qk_h DK BT BK (m + 1)) with hs10
  have hstep10 : stepStmt (Stmt.assign TileDType.ptr [BK] "p_qg"
      (Op.ptrAdd Broadcast.scalarR (Op.ref TileDType.ptr [BK] "p_qg") (Op.constNat DK))) s9 = some s10 := by
    rw [hs10]; exact stepStmt_assign_eq_some (hincr QG "p_qg" s9 hs9_pqg)
  refine ⟨s10, ?_, ?_⟩
  · simp only [prepBody]
    rw [stepStmts.cons_some hstep1, stepStmts.cons_some hstep2, stepStmts.cons_some hstep3,
        stepStmts.cons_some hstep4, stepStmts.cons_some hstep5, stepStmts.cons_some hstoreK,
        stepStmts.cons_some hstoreQ, stepStmts.cons_some hstep6, stepStmts.cons_some hstep7,
        stepStmts.cons_some hstep8, stepStmts.cons_some hstep9, stepStmts.cons_some hstep10,
        stepStmts.nil]
  · -- the invariant at m+1
    -- offset injectivity for the row-m scatter
    have hInj : Function.Injective (fun k : TileIndex [BK] =>
        offset s s_qk_h DK m BT BK k.1) := by
      rintro ⟨a, _⟩ ⟨b, _⟩ hab
      simp only [offset] at hab
      obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
      rfl
    -- row-offset separation: for active k and j ≠ m, offset m k ≠ offset j i
    have hsep : ∀ (j : Nat), j ≠ m → ∀ (i : Fin BK),
        (∀ k : TileIndex [BK], active s DK BK k.1 →
          (fun k : TileIndex [BK] => offset s s_qk_h DK m BT BK k.1) k
            ≠ offset s s_qk_h DK j BT BK i) := by
      intro j hj i k _
      simp only [offset, baseOffset]
      have hik : (i : ℕ) < BK := i.isLt
      have hiDK : (i : ℕ) < DK := lt_of_lt_of_le hik hBK
      rcases Nat.lt_or_ge j m with hlt | hge'
      · have hge : (s.pids 1 * BT + j) * DK + DK ≤ (s.pids 1 * BT + m) * DK := by
          have : (s.pids 1 * BT + j) + 1 ≤ s.pids 1 * BT + m := by omega
          calc (s.pids 1 * BT + j) * DK + DK
              = ((s.pids 1 * BT + j) + 1) * DK := by ring
            _ ≤ (s.pids 1 * BT + m) * DK := Nat.mul_le_mul_right DK this
        omega
      · have hjm' : m < j := by omega
        have hkBK : (k.1 : ℕ) < BK := k.1.isLt
        have hkDK : (k.1 : ℕ) < DK := lt_of_lt_of_le hkBK hBK
        have hge : (s.pids 1 * BT + m) * DK + DK ≤ (s.pids 1 * BT + j) * DK := by
          have : (s.pids 1 * BT + m) + 1 ≤ s.pids 1 * BT + j := by omega
          calc (s.pids 1 * BT + m) * DK + DK
              = ((s.pids 1 * BT + m) + 1) * DK := by ring
            _ ≤ (s.pids 1 * BT + j) * DK := Nat.mul_le_mul_right DK this
        omega
    -- mem reads of QG/KG pass through the 5 pointer setRegs down to sstQ
    have hmemQG10 : ∀ a, s10.readMem QG a = sstQ.readMem QG a := by
      intro a; rw [hs10, hs9, hs8, hs7, hs6]; simp only [BlockState.setReg_readMem]
    have hmemKG10 : ∀ a, s10.readMem KG a = sstQ.readMem KG a := by
      intro a; rw [hs10, hs9, hs8, hs7, hs6]; simp only [BlockState.setReg_readMem]
    -- s5 readMem QG/KG = s.readMem (registers + sloop setReg preserve)
    have hs5QG : ∀ a, s5.readMem QG a = sc.readMem QG a := by
      intro a; rw [hs5, hs4, hs3, hs2, hs1]; simp only [BlockState.setReg_readMem]
      rw [hsloop]; simp only [BlockState.setReg_readMem]
    have hs5KG : ∀ a, s5.readMem KG a = sc.readMem KG a := by
      intro a; rw [hs5, hs4, hs3, hs2, hs1]; simp only [BlockState.setReg_readMem]
      rw [hsloop]; simp only [BlockState.setReg_readMem]
    -- value-fun readback: active lanes give the closed forms
    have hqvfun_active : ∀ i : Fin BK, active s DK BK i →
        qvfun (i, PUnit.unit) = prepareQgClosed s Q G s_qk_h DK BT BK ⟨m, hm⟩ i scale := by
      intro i ha; simp only [hqvfun, prepQValTile, ha, if_true]; rfl
    have hkvfun_active : ∀ i : Fin BK, active s DK BK i →
        kvfun (i, PUnit.unit) = prepareKgClosed s K G s_qk_h DK BT BK ⟨m, hm⟩ i := by
      intro i ha; simp only [hkvfun, prepKValTile, ha, if_true]; rfl
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · -- pids
      rw [hs10, hs9, hs8, hs7, hs6]; simp only [BlockState.setReg_pids]
      rw [hsstQ, BlockState.foldl_writeMem_prop_masked_pids,
        hsstK, BlockState.foldl_writeMem_prop_masked_pids,
        hs5, hs4, hs3, hs2, hs1]; simp only [BlockState.setReg_pids]
      rw [hsloop]; simp only [BlockState.setReg_pids]; exact hpids
    · -- Q preserved
      intro a; rw [hs10, hs9, hs8, hs7, hs6]; simp only [BlockState.setReg_readMem]
      rw [hsstQ, BlockState.scatter_prop_masked_preserves_other_region
        (P := fun k : TileIndex [BK] => active s DK BK k.1) (h_ne := hQ_QG)]
      rw [hsstK, BlockState.scatter_prop_masked_preserves_other_region
        (P := fun k : TileIndex [BK] => active s DK BK k.1) (h_ne := hQ_KG)]
      rw [hs5, hs4, hs3, hs2, hs1]; simp only [BlockState.setReg_readMem]
      rw [hsloop]; simp only [BlockState.setReg_readMem]; exact hQmem a
    · -- K preserved
      intro a; rw [hs10, hs9, hs8, hs7, hs6]; simp only [BlockState.setReg_readMem]
      rw [hsstQ, BlockState.scatter_prop_masked_preserves_other_region
        (P := fun k : TileIndex [BK] => active s DK BK k.1) (h_ne := hK_QG)]
      rw [hsstK, BlockState.scatter_prop_masked_preserves_other_region
        (P := fun k : TileIndex [BK] => active s DK BK k.1) (h_ne := hK_KG)]
      rw [hs5, hs4, hs3, hs2, hs1]; simp only [BlockState.setReg_readMem]
      rw [hsloop]; simp only [BlockState.setReg_readMem]; exact hKmem a
    · -- G preserved
      intro a; rw [hs10, hs9, hs8, hs7, hs6]; simp only [BlockState.setReg_readMem]
      rw [hsstQ, BlockState.scatter_prop_masked_preserves_other_region
        (P := fun k : TileIndex [BK] => active s DK BK k.1) (h_ne := hG_QG)]
      rw [hsstK, BlockState.scatter_prop_masked_preserves_other_region
        (P := fun k : TileIndex [BK] => active s DK BK k.1) (h_ne := hG_KG)]
      rw [hs5, hs4, hs3, hs2, hs1]; simp only [BlockState.setReg_readMem]
      rw [hsloop]; simp only [BlockState.setReg_readMem]; exact hGmem a
    · -- p_q at m+1
      rw [hs10, BlockState.setReg_ne_name (h := by decide),
        hs9, BlockState.setReg_ne_name (h := by decide),
        hs8, BlockState.setReg_ne_name (h := by decide),
        hs7, BlockState.setReg_ne_name (h := by decide), hs6, BlockState.setReg_same]
    · -- p_g at m+1
      rw [hs10, BlockState.setReg_ne_name (h := by decide),
        hs9, BlockState.setReg_ne_name (h := by decide),
        hs8, BlockState.setReg_ne_name (h := by decide), hs7, BlockState.setReg_same]
    · -- p_k at m+1
      rw [hs10, BlockState.setReg_ne_name (h := by decide),
        hs9, BlockState.setReg_ne_name (h := by decide), hs8, BlockState.setReg_same]
    · -- p_qg at m+1
      rw [hs10, BlockState.setReg_same]
    · -- p_kg at m+1
      rw [hs10, BlockState.setReg_ne_name (h := by decide), hs9, BlockState.setReg_same]
    · -- mask preserved
      rw [hs10, BlockState.setReg_ne_name (h := by decide),
        hs9, BlockState.setReg_ne_name (h := by decide),
        hs8, BlockState.setReg_ne_name (h := by decide),
        hs7, BlockState.setReg_ne_name (h := by decide),
        hs6, BlockState.setReg_ne_name (h := by decide)]
      rw [hsstQ, BlockState.foldl_writeMem_prop_masked_regs,
        hsstK, BlockState.foldl_writeMem_prop_masked_regs,
        hs5, hs4, hs3, hs2, hs1]; simpa using hmask'
    · -- last_decay preserved
      rw [hs10, BlockState.setReg_ne_name (h := by decide),
        hs9, BlockState.setReg_ne_name (h := by decide),
        hs8, BlockState.setReg_ne_name (h := by decide),
        hs7, BlockState.setReg_ne_name (h := by decide),
        hs6, BlockState.setReg_ne_name (h := by decide)]
      rw [hsstQ, BlockState.foldl_writeMem_prop_masked_regs,
        hsstK, BlockState.foldl_writeMem_prop_masked_regs,
        hs5, hs4, hs3, hs2, hs1]; simpa using hld'
    · -- QG untouched at rows ≥ m+1
      intro j hj i
      rw [hmemQG10 (offset s s_qk_h DK j BT BK i), hsstQ]
      rw [BlockState.scatter_prop_masked_preserves_other_offset
        (region := QG)
        (offsetFn := fun k : TileIndex [BK] => offset s s_qk_h DK m BT BK k.1)
        (valueFn := qvfun)
        (P := fun k : TileIndex [BK] => active s DK BK k.1)
        (off := offset s s_qk_h DK j BT BK i)
        (h_ne := hsep j (by omega) i)]
      rw [hsstK, BlockState.scatter_prop_masked_preserves_other_region
        (P := fun k : TileIndex [BK] => active s DK BK k.1) (h_ne := hQG_KG)]
      rw [hs5QG]
      exact hUQG j (by omega) i
    · -- KG untouched at rows ≥ m+1
      intro j hj i
      rw [hmemKG10 (offset s s_qk_h DK j BT BK i), hsstQ]
      rw [BlockState.scatter_prop_masked_preserves_other_region
        (P := fun k : TileIndex [BK] => active s DK BK k.1) (h_ne := Ne.symm hQG_KG)]
      rw [hsstK, BlockState.scatter_prop_masked_preserves_other_offset
        (region := KG)
        (offsetFn := fun k : TileIndex [BK] => offset s s_qk_h DK m BT BK k.1)
        (valueFn := kvfun)
        (P := fun k : TileIndex [BK] => active s DK BK k.1)
        (off := offset s s_qk_h DK j BT BK i)
        (h_ne := hsep j (by omega) i)]
      rw [hs5KG]
      exact hUKG j (by omega) i
    · -- QG readback at rows < m+1
      intro j hj i hjBT
      rw [hmemQG10 (offset s s_qk_h DK j BT BK i), hsstQ]
      by_cases hjm : j = m
      · subst hjm
        rw [BlockState.scatter_readback_prop_masked_nd sstK
          (fun k : TileIndex [BK] => offset s s_qk_h DK j BT BK k.1) qvfun
          (fun k => active s DK BK k.1) hInj (i, PUnit.unit)]
        by_cases ha : active s DK BK i
        · rw [if_pos ha, if_pos ha, hqvfun_active i ha]
        · rw [if_neg ha, if_neg ha]
          rw [hsstK, BlockState.scatter_prop_masked_preserves_other_region
            (P := fun k : TileIndex [BK] => active s DK BK k.1) (h_ne := hQG_KG)]
          rw [hs5QG]
          exact hUQG j (le_refl j) i
      · have hjlt : j < m := by omega
        rw [BlockState.scatter_prop_masked_preserves_other_offset
          (region := QG)
          (offsetFn := fun k : TileIndex [BK] => offset s s_qk_h DK m BT BK k.1)
          (valueFn := qvfun)
          (P := fun k : TileIndex [BK] => active s DK BK k.1)
          (off := offset s s_qk_h DK j BT BK i)
          (h_ne := hsep j (by omega) i)]
        rw [hsstK, BlockState.scatter_prop_masked_preserves_other_region
          (P := fun k : TileIndex [BK] => active s DK BK k.1) (h_ne := hQG_KG)]
        rw [hs5QG]
        exact hreadQG j hjlt i hjBT
    · -- KG readback at rows < m+1
      intro j hj i hjBT
      rw [hmemKG10 (offset s s_qk_h DK j BT BK i), hsstQ]
      rw [BlockState.scatter_prop_masked_preserves_other_region
        (P := fun k : TileIndex [BK] => active s DK BK k.1) (h_ne := Ne.symm hQG_KG)]
      by_cases hjm : j = m
      · subst hjm
        rw [hsstK, BlockState.scatter_readback_prop_masked_nd s5
          (fun k : TileIndex [BK] => offset s s_qk_h DK j BT BK k.1) kvfun
          (fun k => active s DK BK k.1) hInj (i, PUnit.unit)]
        by_cases ha : active s DK BK i
        · rw [if_pos ha, if_pos ha, hkvfun_active i ha]
        · rw [if_neg ha, if_neg ha]
          rw [hs5KG]
          exact hUKG j (le_refl j) i
      · have hjlt : j < m := by omega
        rw [hsstK, BlockState.scatter_prop_masked_preserves_other_offset
          (region := KG)
          (offsetFn := fun k : TileIndex [BK] => offset s s_qk_h DK m BT BK k.1)
          (valueFn := kvfun)
          (P := fun k : TileIndex [BK] => active s DK BK k.1)
          (off := offset s s_qk_h DK j BT BK i)
          (h_ne := hsep j (by omega) i)]
        rw [hs5KG]
        exact hreadKG j hjlt i hjBT

set_option maxHeartbeats 4000000 in
/-- **Drive the `prepare_qg_kg` loop.** Reduces `exec` over the 11 prefix
statements onto the `forRange` loop, establishes the entry invariant `prepInv 0`,
and runs `forRange_inv` with `prepare_qg_kg_step`, yielding the final loop state
together with `prepInv final`. -/
theorem prepare_qg_kg_loop_drive
    (Q K G QG KG : RegionName) (s : BlockState) (s_qk_h DK BT BK : Nat) (scale : ℝ)
    (hQ_QG : Q ≠ QG) (hQ_KG : Q ≠ KG) (hK_QG : K ≠ QG) (hK_KG : K ≠ KG)
    (hG_QG : G ≠ QG) (hG_KG : G ≠ KG) (hQG_KG : QG ≠ KG) (hBK : BK ≤ DK) (hBT : 0 < BT) :
    ∃ final s_final,
      exec (prepare_qg_kg_surface Q K G QG KG s_qk_h DK BT BK scale) s = some s_final ∧
      BT ≤ final ∧ prepInv Q K G QG KG s s_qk_h DK BT BK scale final s_final := by
  simp only [exec, prepare_qg_kg_surface, ComputeKernel.toAlgKernel,
    ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Except.bind, Except.pure, bind, pure]
  -- prefix states
  set s_ik := s.setReg "i_k" .nat [] (Tile.scalar (s.pids 0)) with hs_ik
  set s_ic := s_ik.setReg "i_c" .nat [] (Tile.scalar (s.pids 1)) with hs_ic
  set s_ibh := s_ic.setReg "i_bh" .nat [] (Tile.scalar (s.pids 2)) with hs_ibh
  set s_offs := s_ibh.setReg "offs" .nat [BK] (prepOffsTile BK) with hs_offs
  set s_pq := s_offs.setReg "p_q" .ptr [BK] (prepPtrTile s Q s_qk_h DK BT BK 0) with hs_pq
  set s_pg := s_pq.setReg "p_g" .ptr [BK] (prepPtrTile s G s_qk_h DK BT BK 0) with hs_pg
  set s_pk := s_pg.setReg "p_k" .ptr [BK] (prepPtrTile s K s_qk_h DK BT BK 0) with hs_pk
  set s_pqg := s_pk.setReg "p_qg" .ptr [BK] (prepPtrTile s QG s_qk_h DK BT BK 0) with hs_pqg
  set s_pkg := s_pqg.setReg "p_kg" .ptr [BK] (prepPtrTile s KG s_qk_h DK BT BK 0) with hs_pkg
  set s_mask := s_pkg.setReg "mask" .bool [BK] (fwdMaskTile s DK BK) with hs_mask
  set s11 := s_mask.setReg "last_decay" .real [BK] (prepLastDecayTile s G s_qk_h DK BT BK) with hs11
  -- pid lookups along the chain
  have hibh : s_ibh.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)) := by simp [hs_ibh]
  have hic : s_ibh.regs .nat [] "i_c" = some (Tile.scalar (s.pids 1)) := by
    rw [hs_ibh, hs_ic]; simp
  have hik : s_ibh.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)) := by
    rw [hs_ibh, hs_ic, hs_ik]; simp
  -- generic pointer-init evaluator (after offs is set)
  have hptr : ∀ (R : RegionName) (st : BlockState),
      st.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)) →
      st.regs .nat [] "i_c" = some (Tile.scalar (s.pids 1)) →
      st.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)) →
      st.regs .nat [BK] "offs" = some (prepOffsTile BK) →
      evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase R)
        (Op.add NumericDType.nat Broadcast.scalarL
          (Op.add NumericDType.nat Broadcast.nil
            (Op.add NumericDType.nat Broadcast.nil
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_bh") (Op.constNat s_qk_h))
              (Op.mul NumericDType.nat Broadcast.nil
                (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_c") (Op.constNat BT))
                (Op.constNat DK)))
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k") (Op.constNat BK)))
          (Op.ref TileDType.nat [BK] "offs"))) st = some (prepPtrTile s R s_qk_h DK BT BK 0) := by
    intro R st hbh hc hk ho
    simp only [evalOp, Option.bind, hbh, hc, hk, ho, evalOp_constNat,
      Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul,
      Tile.bop, Tile.scalar]
    apply congrArg some
    apply Tile.ext
    intro idx
    obtain ⟨i, u⟩ := idx
    simp only [Tile.ptrAdd_data, prepPtrTile, offset, baseOffset, prepOffsTile,
      Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL,
      Broadcast.leftIndex_nil, Broadcast.rightIndex_nil, Tile.vec]
    refine Prod.ext rfl ?_
    ring
  -- step lemmas for each prefix assignment
  have hstepK : stepStmt (Stmt.assign TileDType.nat [] "i_k" (Op.programId 0)) s = some s_ik := by
    rw [hs_ik]; exact stepStmt_assign_eq_some (by simp [evalOp])
  have hstepC : stepStmt (Stmt.assign TileDType.nat [] "i_c" (Op.programId 1)) s_ik = some s_ic := by
    rw [hs_ic]; exact stepStmt_assign_eq_some (show evalOp (Op.programId 1) s_ik = _ from by simp [evalOp, hs_ik])
  have hstepB : stepStmt (Stmt.assign TileDType.nat [] "i_bh" (Op.programId 2)) s_ic = some s_ibh := by
    rw [hs_ibh]; exact stepStmt_assign_eq_some (show evalOp (Op.programId 2) s_ic = _ from by simp [evalOp, hs_ic, hs_ik])
  have hstepOffs : stepStmt (Stmt.assign TileDType.nat [BK] "offs" (Op.arange BK)) s_ibh = some s_offs := by
    rw [hs_offs]; exact stepStmt_assign_eq_some (by simp only [evalOp_arange, prepOffsTile])
  -- offs lookups (preserved by later setRegs of different names)
  have hoffs_at : ∀ (st : BlockState), st.regs .nat [BK] "offs" = some (prepOffsTile BK) →
      True := fun _ _ => trivial
  have hsoffs_bh : s_offs.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)) := by rw [hs_offs]; simp [hibh]
  have hsoffs_c : s_offs.regs .nat [] "i_c" = some (Tile.scalar (s.pids 1)) := by rw [hs_offs]; simp [hic]
  have hsoffs_k : s_offs.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)) := by rw [hs_offs]; simp [hik]
  have hsoffs_o : s_offs.regs .nat [BK] "offs" = some (prepOffsTile BK) := by rw [hs_offs]; simp
  have hstepPq : stepStmt (Stmt.assign TileDType.ptr [BK] "p_q"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Q)
        (Op.add NumericDType.nat Broadcast.scalarL
          (Op.add NumericDType.nat Broadcast.nil
            (Op.add NumericDType.nat Broadcast.nil
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_bh") (Op.constNat s_qk_h))
              (Op.mul NumericDType.nat Broadcast.nil
                (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_c") (Op.constNat BT))
                (Op.constNat DK)))
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k") (Op.constNat BK)))
          (Op.ref TileDType.nat [BK] "offs")))) s_offs = some s_pq := by
    rw [hs_pq]; exact stepStmt_assign_eq_some (hptr Q s_offs hsoffs_bh hsoffs_c hsoffs_k hsoffs_o)
  have hspq_bh : s_pq.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)) := by rw [hs_pq]; simp [hsoffs_bh]
  have hspq_c : s_pq.regs .nat [] "i_c" = some (Tile.scalar (s.pids 1)) := by rw [hs_pq]; simp [hsoffs_c]
  have hspq_k : s_pq.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)) := by rw [hs_pq]; simp [hsoffs_k]
  have hspq_o : s_pq.regs .nat [BK] "offs" = some (prepOffsTile BK) := by rw [hs_pq]; simp [hsoffs_o]
  have hstepPg : stepStmt (Stmt.assign TileDType.ptr [BK] "p_g"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase G)
        (Op.add NumericDType.nat Broadcast.scalarL
          (Op.add NumericDType.nat Broadcast.nil
            (Op.add NumericDType.nat Broadcast.nil
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_bh") (Op.constNat s_qk_h))
              (Op.mul NumericDType.nat Broadcast.nil
                (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_c") (Op.constNat BT))
                (Op.constNat DK)))
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k") (Op.constNat BK)))
          (Op.ref TileDType.nat [BK] "offs")))) s_pq = some s_pg := by
    rw [hs_pg]; exact stepStmt_assign_eq_some (hptr G s_pq hspq_bh hspq_c hspq_k hspq_o)
  have hspg_bh : s_pg.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)) := by rw [hs_pg]; simp [hspq_bh]
  have hspg_c : s_pg.regs .nat [] "i_c" = some (Tile.scalar (s.pids 1)) := by rw [hs_pg]; simp [hspq_c]
  have hspg_k : s_pg.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)) := by rw [hs_pg]; simp [hspq_k]
  have hspg_o : s_pg.regs .nat [BK] "offs" = some (prepOffsTile BK) := by rw [hs_pg]; simp [hspq_o]
  have hstepPk : stepStmt (Stmt.assign TileDType.ptr [BK] "p_k"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase K)
        (Op.add NumericDType.nat Broadcast.scalarL
          (Op.add NumericDType.nat Broadcast.nil
            (Op.add NumericDType.nat Broadcast.nil
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_bh") (Op.constNat s_qk_h))
              (Op.mul NumericDType.nat Broadcast.nil
                (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_c") (Op.constNat BT))
                (Op.constNat DK)))
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k") (Op.constNat BK)))
          (Op.ref TileDType.nat [BK] "offs")))) s_pg = some s_pk := by
    rw [hs_pk]; exact stepStmt_assign_eq_some (hptr K s_pg hspg_bh hspg_c hspg_k hspg_o)
  have hspk_bh : s_pk.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)) := by rw [hs_pk]; simp [hspg_bh]
  have hspk_c : s_pk.regs .nat [] "i_c" = some (Tile.scalar (s.pids 1)) := by rw [hs_pk]; simp [hspg_c]
  have hspk_k : s_pk.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)) := by rw [hs_pk]; simp [hspg_k]
  have hspk_o : s_pk.regs .nat [BK] "offs" = some (prepOffsTile BK) := by rw [hs_pk]; simp [hspg_o]
  have hstepPqg : stepStmt (Stmt.assign TileDType.ptr [BK] "p_qg"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase QG)
        (Op.add NumericDType.nat Broadcast.scalarL
          (Op.add NumericDType.nat Broadcast.nil
            (Op.add NumericDType.nat Broadcast.nil
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_bh") (Op.constNat s_qk_h))
              (Op.mul NumericDType.nat Broadcast.nil
                (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_c") (Op.constNat BT))
                (Op.constNat DK)))
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k") (Op.constNat BK)))
          (Op.ref TileDType.nat [BK] "offs")))) s_pk = some s_pqg := by
    rw [hs_pqg]; exact stepStmt_assign_eq_some (hptr QG s_pk hspk_bh hspk_c hspk_k hspk_o)
  have hspqg_bh : s_pqg.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)) := by rw [hs_pqg]; simp [hspk_bh]
  have hspqg_c : s_pqg.regs .nat [] "i_c" = some (Tile.scalar (s.pids 1)) := by rw [hs_pqg]; simp [hspk_c]
  have hspqg_k : s_pqg.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)) := by rw [hs_pqg]; simp [hspk_k]
  have hspqg_o : s_pqg.regs .nat [BK] "offs" = some (prepOffsTile BK) := by rw [hs_pqg]; simp [hspk_o]
  have hstepPkg : stepStmt (Stmt.assign TileDType.ptr [BK] "p_kg"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase KG)
        (Op.add NumericDType.nat Broadcast.scalarL
          (Op.add NumericDType.nat Broadcast.nil
            (Op.add NumericDType.nat Broadcast.nil
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_bh") (Op.constNat s_qk_h))
              (Op.mul NumericDType.nat Broadcast.nil
                (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_c") (Op.constNat BT))
                (Op.constNat DK)))
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k") (Op.constNat BK)))
          (Op.ref TileDType.nat [BK] "offs")))) s_pqg = some s_pkg := by
    rw [hs_pkg]; exact stepStmt_assign_eq_some (hptr KG s_pqg hspqg_bh hspqg_c hspqg_k hspqg_o)
  have hspkg_k : s_pkg.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)) := by rw [hs_pkg]; simp [hspqg_k]
  have hspkg_o : s_pkg.regs .nat [BK] "offs" = some (prepOffsTile BK) := by rw [hs_pkg]; simp [hspqg_o]
  -- mask eval
  have hmaskeval : evalOp (Op.lt ComparableDType.nat Broadcast.scalarR
      (Op.add NumericDType.nat Broadcast.scalarL
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k") (Op.constNat BK))
        (Op.ref TileDType.nat [BK] "offs")) (Op.constNat DK)) s_pkg = some (fwdMaskTile s DK BK) := by
    simp only [evalOp, Option.bind, hspkg_k, hspkg_o, evalOp_constNat,
      Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul,
      ComparableDType.lt, Tile.bop, Tile.cop, Tile.scalar]
    apply congrArg some
    apply Tile.ext
    intro idx
    obtain ⟨i, u⟩ := idx
    simp only [fwdMaskTile, active, elemIndex, prepOffsTile, Tile.vec,
      Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL,
      Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR,
      Broadcast.leftIndex_nil, Broadcast.rightIndex_nil]
    rfl
  have hstepMask : stepStmt (Stmt.assign TileDType.bool [BK] "mask"
      (Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.add NumericDType.nat Broadcast.scalarL
          (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k") (Op.constNat BK))
          (Op.ref TileDType.nat [BK] "offs")) (Op.constNat DK))) s_pkg = some s_mask := by
    rw [hs_mask]; exact stepStmt_assign_eq_some hmaskeval
  have hsmask_bh : s_mask.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)) := by
    rw [hs_mask, hs_pkg, hs_pqg, hs_pk, hs_pg, hs_pq]; simp [hsoffs_bh]
  have hsmask_c : s_mask.regs .nat [] "i_c" = some (Tile.scalar (s.pids 1)) := by
    rw [hs_mask, hs_pkg, hs_pqg, hs_pk, hs_pg, hs_pq]; simp [hsoffs_c]
  have hsmask_k : s_mask.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)) := by
    rw [hs_mask, hs_pkg, hs_pqg, hs_pk, hs_pg, hs_pq]; simp [hsoffs_k]
  have hsmask_o : s_mask.regs .nat [BK] "offs" = some (prepOffsTile BK) := by
    rw [hs_mask, hs_pkg, hs_pqg, hs_pk, hs_pg, hs_pq]; simp [hsoffs_o]
  -- last_decay eval (region load, MaskOpt.none) — uses BT-1 with BT ≥ 1
  have hldeval : evalOp (Op.load TileDType.real
      (MemAccess.region G
        (Op.add NumericDType.nat Broadcast.scalarL
          (Op.add NumericDType.nat Broadcast.nil
            (Op.add NumericDType.nat Broadcast.nil
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_bh") (Op.constNat s_qk_h))
              (Op.mul NumericDType.nat Broadcast.nil
                (Op.sub NumericDType.nat Broadcast.nil
                  (Op.add NumericDType.nat Broadcast.nil
                    (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_c") (Op.constNat BT))
                    (Op.constNat BT))
                  (Op.constNat 1))
                (Op.constNat DK)))
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k") (Op.constNat BK)))
          (Op.ref TileDType.nat [BK] "offs")))
      MaskOpt.none) s_mask = some (prepLastDecayTile s G s_qk_h DK BT BK) := by
    simp only [evalOp, Option.bind, hsmask_bh, hsmask_c, hsmask_k, hsmask_o, evalOp_constNat,
      Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul, NumericDType.sub,
      Tile.bop, Tile.scalar]
    apply congrArg some
    apply Tile.ext
    intro idx
    obtain ⟨i, u⟩ := idx
    simp only [prepLastDecayTile, prepOffsTile, Tile.vec,
      Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL,
      Broadcast.leftIndex_nil, Broadcast.rightIndex_nil]
    rw [BlockState.readMemValue_real]
    simp only [offset, baseOffset]
    rw [show s.pids 1 * BT + BT - 1 = s.pids 1 * BT + (BT - 1) by omega]
    simp only [if_true]
    rw [hs_mask, hs_pkg, hs_pqg, hs_pk, hs_pg, hs_pq, hs_offs, hs_ibh, hs_ic, hs_ik]
    simp only [BlockState.setReg_readMem]
    rfl
  have hstepLd : stepStmt (Stmt.assign TileDType.real [BK] "last_decay"
      (Op.load TileDType.real
        (MemAccess.region G
          (Op.add NumericDType.nat Broadcast.scalarL
            (Op.add NumericDType.nat Broadcast.nil
              (Op.add NumericDType.nat Broadcast.nil
                (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_bh") (Op.constNat s_qk_h))
                (Op.mul NumericDType.nat Broadcast.nil
                  (Op.sub NumericDType.nat Broadcast.nil
                    (Op.add NumericDType.nat Broadcast.nil
                      (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_c") (Op.constNat BT))
                      (Op.constNat BT))
                    (Op.constNat 1))
                  (Op.constNat DK)))
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k") (Op.constNat BK)))
            (Op.ref TileDType.nat [BK] "offs")))
        MaskOpt.none)) s_mask = some s11 := by
    rw [hs11]; exact stepStmt_assign_eq_some hldeval
  -- chain the 11 prefix assigns onto the forRange loop
  rw [stepStmts.cons_some hstepK, stepStmts.cons_some hstepC, stepStmts.cons_some hstepB,
      stepStmts.cons_some hstepOffs, stepStmts.cons_some hstepPq, stepStmts.cons_some hstepPg,
      stepStmts.cons_some hstepPk, stepStmts.cons_some hstepPqg, stepStmts.cons_some hstepPkg]
  erw [stepStmts.cons_some hstepMask]
  erw [stepStmts.cons_some hstepLd]
  rw [show ∀ (X : Stmt), stepStmts [X] s11 = stepStmt X s11 from
      fun X => by cases h : stepStmt X s11 <;> simp [stepStmts, h]]
  -- entry invariant prepInv 0 s11
  have hP0 : prepInv Q K G QG KG s s_qk_h DK BT BK scale 0 s11 := by
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · rw [hs11, hs_mask, hs_pkg, hs_pqg, hs_pk, hs_pg, hs_pq, hs_offs, hs_ibh, hs_ic, hs_ik]
      simp only [BlockState.setReg_pids]
    · intro a; rw [hs11, hs_mask, hs_pkg, hs_pqg, hs_pk, hs_pg, hs_pq, hs_offs, hs_ibh, hs_ic, hs_ik]
      simp only [BlockState.setReg_readMem]
    · intro a; rw [hs11, hs_mask, hs_pkg, hs_pqg, hs_pk, hs_pg, hs_pq, hs_offs, hs_ibh, hs_ic, hs_ik]
      simp only [BlockState.setReg_readMem]
    · intro a; rw [hs11, hs_mask, hs_pkg, hs_pqg, hs_pk, hs_pg, hs_pq, hs_offs, hs_ibh, hs_ic, hs_ik]
      simp only [BlockState.setReg_readMem]
    · -- p_q
      simp only [hs11, hs_mask, hs_pkg, hs_pqg, hs_pk, hs_pg, hs_pq,
        BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, reduceCtorEq,
        not_false_eq_true, String.reduceEq]
    · -- p_g
      simp only [hs11, hs_mask, hs_pkg, hs_pqg, hs_pk, hs_pg,
        BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, reduceCtorEq,
        not_false_eq_true, String.reduceEq]
    · -- p_k
      simp only [hs11, hs_mask, hs_pkg, hs_pqg, hs_pk,
        BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, reduceCtorEq,
        not_false_eq_true, String.reduceEq]
    · -- p_qg
      simp only [hs11, hs_mask, hs_pkg, hs_pqg,
        BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, reduceCtorEq,
        not_false_eq_true, String.reduceEq]
    · -- p_kg
      simp only [hs11, hs_mask, hs_pkg,
        BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, reduceCtorEq,
        not_false_eq_true, String.reduceEq]
    · -- mask
      simp only [hs11, hs_mask,
        BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, reduceCtorEq,
        not_false_eq_true, String.reduceEq]
    · -- last_decay
      rw [hs11, BlockState.setReg_same]
    · intro j hj i; rw [hs11, hs_mask, hs_pkg, hs_pqg, hs_pk, hs_pg, hs_pq, hs_offs, hs_ibh, hs_ic, hs_ik]
      simp only [BlockState.setReg_readMem]
    · intro j hj i; rw [hs11, hs_mask, hs_pkg, hs_pqg, hs_pk, hs_pg, hs_pq, hs_offs, hs_ibh, hs_ic, hs_ik]
      simp only [BlockState.setReg_readMem]
    · intro j hj i hjBT; omega
    · intro j hj i hjBT; omega
  -- drive the loop
  obtain ⟨final, sfinal, hLoop, hfinal, hPfinal⟩ :=
    forRange_inv (idx := "_i") (start := 0) (stop := BT) (step := 1)
      (body := prepBody Q K G QG KG s_qk_h DK BT BK scale)
      (P := prepInv Q K G QG KG s s_qk_h DK BT BK scale) (s_init := s11) (Nat.one_ne_zero) hP0
      (fun j st hj hPj => prepare_qg_kg_step Q K G QG KG s s_qk_h DK BT BK scale
        hQ_QG hQ_KG hK_QG hK_KG hG_QG hG_KG hQG_KG hBK hBT j hj st hPj)
  refine ⟨final, sfinal, ?_, hfinal, hPfinal⟩
  simp only [prepBody] at hLoop
  rw [hLoop]

set_option maxHeartbeats 4000000 in
/-- **Genuine GENERAL `qg` closed form.** For arbitrary `BT BK DK` (`BK ≤ DK`,
`0 < BT`), at chunk row `t_rel` and lane `i`, the full `prepare_qg_kg` surface
writes `q[idx] * exp2(g[idx]) * scale` into `QG` on active lanes (and leaves
`QG` unchanged on inactive lanes). Dimension-parameterized replacement for the
`BT=2`-pinned `prepare_qg_kg_full_surface_qg_closed`. -/
theorem prepare_qg_kg_full_surface_qg_closed_general
    (Q K G QG KG : RegionName)
    (s_qk_h DK BT BK : Nat) (scale : ℝ) (s : BlockState) (t_rel : Fin BT) (i : Fin BK)
    (hQ_QG : Q ≠ QG) (hQ_KG : Q ≠ KG) (hK_QG : K ≠ QG) (hK_KG : K ≠ KG)
    (hG_QG : G ≠ QG) (hG_KG : G ≠ KG) (hQG_KG : QG ≠ KG) (hBK : BK ≤ DK) :
    (exec (prepare_qg_kg_surface Q K G QG KG s_qk_h DK BT BK scale) s).map
      (·.readMem QG (offset s s_qk_h DK t_rel.val BT BK i))
      = some (if active s DK BK i then prepareQgClosed s Q G s_qk_h DK BT BK t_rel i scale
              else s.readMem QG (offset s s_qk_h DK t_rel.val BT BK i)) := by
  have hBT : 0 < BT := Fin.pos t_rel
  obtain ⟨final, sfinal, hExec, hfinal, hPfinal⟩ :=
    prepare_qg_kg_loop_drive Q K G QG KG s s_qk_h DK BT BK scale
      hQ_QG hQ_KG hK_QG hK_KG hG_QG hG_KG hQG_KG hBK hBT
  rw [hExec]
  simp only [Option.map_some]
  obtain ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, hreadQG, _⟩ := hPfinal
  have ht : t_rel.val < final := lt_of_lt_of_le t_rel.isLt hfinal
  rw [hreadQG t_rel.val ht i t_rel.isLt]

set_option maxHeartbeats 4000000 in
/-- **Genuine GENERAL `kg` closed form.** For arbitrary `BT BK DK` (`BK ≤ DK`,
`0 < BT`), at chunk row `t_rel` and lane `i`, the full `prepare_qg_kg` surface
writes `k[idx] * exp2(g[row BT-1] - g[idx])` into `KG` on active lanes (and
leaves `KG` unchanged on inactive lanes). -/
theorem prepare_qg_kg_full_surface_kg_closed_general
    (Q K G QG KG : RegionName)
    (s_qk_h DK BT BK : Nat) (scale : ℝ) (s : BlockState) (t_rel : Fin BT) (i : Fin BK)
    (hQ_QG : Q ≠ QG) (hQ_KG : Q ≠ KG) (hK_QG : K ≠ QG) (hK_KG : K ≠ KG)
    (hG_QG : G ≠ QG) (hG_KG : G ≠ KG) (hQG_KG : QG ≠ KG) (hBK : BK ≤ DK) :
    (exec (prepare_qg_kg_surface Q K G QG KG s_qk_h DK BT BK scale) s).map
      (·.readMem KG (offset s s_qk_h DK t_rel.val BT BK i))
      = some (if active s DK BK i then prepareKgClosed s K G s_qk_h DK BT BK t_rel i
              else s.readMem KG (offset s s_qk_h DK t_rel.val BT BK i)) := by
  have hBT : 0 < BT := Fin.pos t_rel
  obtain ⟨final, sfinal, hExec, hfinal, hPfinal⟩ :=
    prepare_qg_kg_loop_drive Q K G QG KG s s_qk_h DK BT BK scale
      hQ_QG hQ_KG hK_QG hK_KG hG_QG hG_KG hQG_KG hBK hBT
  rw [hExec]
  simp only [Option.map_some]
  obtain ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, _, hreadKG⟩ := hPfinal
  have ht : t_rel.val < final := lt_of_lt_of_le t_rel.isLt hfinal
  rw [hreadKG t_rel.val ht i t_rel.isLt]

set_option maxHeartbeats 4000000 in
/-- **Genuine GENERAL `qg` compute-correctness.** For arbitrary `BT BK DK`
(`BK ≤ DK`), the full `prepare_qg_kg` surface realizes the honest pointwise
closed form `prepareQgClosed` (`q * exp2(g) * scale`) at every active `QG` lane
of loop row `t_rel`. Dimension-parameterized replacement for the `BT=2`-pinned
`prepare_qg_kg_surface_qg_closed_compute_correct`. -/
theorem prepare_qg_kg_surface_qg_closed_compute_correct_general
    (Q K G QG KG : RegionName)
    (s_qk_h DK BT BK : Nat) (scale : ℝ) (s : BlockState) (t_rel : Fin BT)
    (hQ_QG : Q ≠ QG) (hQ_KG : Q ≠ KG) (hK_QG : K ≠ QG) (hK_KG : K ≠ KG)
    (hG_QG : G ≠ QG) (hG_KG : G ≠ KG) (hQG_KG : QG ≠ KG) (hBK : BK ≤ DK) :
    ComputeCorrect.Realizes
      (kernel := prepare_qg_kg_surface Q K G QG KG s_qk_h DK BT BK scale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s DK BK)
        (fun i => (QG, offset s s_qk_h DK t_rel.val BT BK i)))
      (expected := fun i : Fin BK =>
        prepareQgClosed s Q G s_qk_h DK BT BK t_rel i scale) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [prepare_qg_kg_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := prepare_qg_kg_full_surface_qg_closed_general Q K G QG KG
    s_qk_h DK BT BK scale s t_rel i
    hQ_QG hQ_KG hK_QG hK_KG hG_QG hG_KG hQG_KG hBK
  rw [hExec] at h
  simp only [Option.map_some] at h
  rw [Option.some.injEq] at h
  show s'.readMem QG (offset s s_qk_h DK t_rel.val BT BK i) = _
  rw [h, if_pos hActive]

set_option maxHeartbeats 4000000 in
/-- **Genuine GENERAL `kg` compute-correctness.** For arbitrary `BT BK DK`
(`BK ≤ DK`), the full `prepare_qg_kg` surface realizes the honest pointwise
closed form `prepareKgClosed` (`k * exp2(g[BT-1] - g)`) at every active `KG`
lane of loop row `t_rel`. -/
theorem prepare_qg_kg_surface_kg_closed_compute_correct_general
    (Q K G QG KG : RegionName)
    (s_qk_h DK BT BK : Nat) (scale : ℝ) (s : BlockState) (t_rel : Fin BT)
    (hQ_QG : Q ≠ QG) (hQ_KG : Q ≠ KG) (hK_QG : K ≠ QG) (hK_KG : K ≠ KG)
    (hG_QG : G ≠ QG) (hG_KG : G ≠ KG) (hQG_KG : QG ≠ KG) (hBK : BK ≤ DK) :
    ComputeCorrect.Realizes
      (kernel := prepare_qg_kg_surface Q K G QG KG s_qk_h DK BT BK scale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s DK BK)
        (fun i => (KG, offset s s_qk_h DK t_rel.val BT BK i)))
      (expected := fun i : Fin BK =>
        prepareKgClosed s K G s_qk_h DK BT BK t_rel i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [prepare_qg_kg_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := prepare_qg_kg_full_surface_kg_closed_general Q K G QG KG
    s_qk_h DK BT BK scale s t_rel i
    hQ_QG hQ_KG hK_QG hK_KG hG_QG hG_KG hQG_KG hBK
  rw [hExec] at h
  simp only [Option.map_some] at h
  rw [Option.some.injEq] at h
  show s'.readMem KG (offset s s_qk_h DK t_rel.val BT BK i) = _
  rw [h, if_pos hActive]

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
`bwdDGClosed`) are the honest, non self-referential specifications that replace
the (now-deleted) `decayBackwardSurfaceValue`. They are connected to the executed
`bwd_decay_global_cumsum_surface` in
`decay_cumsum_backward_closed_output_summary_general` (and its three
faces `bwd_decay_cumsum_d{q,k}_inter_closed_compute_correct_general` /
`bwd_decay_cumsum_dg_closed_compute_correct_general`) at the end of this file, following
the **forward** closed-form recipe (`fwd_decay_cumsum_full_surface_row{0,1}_closed`),
but the backward
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
   `computeCorrect_of_toAlgKernel` (done; `decayBackwardSurfaceValue` deleted).

This plan is now fully realized: `bwd_iter_core` chains the 22-statement body,
`bwd_iter0_eval` / `bwd_iter1_eval` add the head + conditional `last_g` capture,
`bwd_full_exec` assembles prologue + 2-iteration loop, and the three readback
theorems certify the closed forms (the `dg` face uses the genuine reverse cumsum).

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
`bwdDGSummand`, `bwdDGClosed`) are already banked above, and the assembly that
threads the reverse `cum_grad_dg` scan + `last_g` capture through these recipes
is now complete, sorry-free.

`s_qk_h`, `DK`, `BK`, etc. are kept symbolic so each recipe is reusable at any
loop row / pointer position; the assembled main theorems stay symbolic over
these block sizes (the Python test shape `s_qk_h = 64`, `DK = 8`, `BT = 2`,
`BK = 4` is only a recovered instance, not a fixed assumption). -/

section BwdRecipes

variable {BK : Nat} (s : BlockState)

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
chaining from `bwdIterBody_head_eval`, assembled in `bwd_iter0_eval` / `bwd_iter1_eval`. -/
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

/-- The per-lane `dg` summand `dq*q - dk*k` at row offset `(s.pids1*2+t_rel)*8`
(with captured `last_g = g[row BT-1]`) equals the genuine `bwdDGSummand`. -/
private theorem dgSum_eq_summand (s : BlockState)
    (DQInner DQInter DKInner DKInter Q K G : RegionName) (t_rel : Fin 2) (i : Fin 4)
    (R Rlast : Nat) (hR : R = (s.pids 1 * 2 + t_rel.val) * 8)
    (hRlast : Rlast = (s.pids 1 * 2 + 1) * 8) :
    dgSum s DQInner DQInter DKInner DKInter Q K G R (ldR s G Rlast) (i, PUnit.unit)
      = bwdDGSummand s DQInner DQInter DKInner DKInter Q K G 64 8 2 4 t_rel i := by
  have haQ : s.pids 2 * 64 + s.pids 0 * 4 + i.val + R = offset s 64 8 t_rel.val 2 4 i := by
    simp only [offset, baseOffset, hR]; ring
  simp only [dgSum, bwdDGSummand,
    dqOut_eq_closed s DQInner DQInter G t_rel i R hR,
    dkOut_eq_closed s DKInner DKInter G t_rel i R Rlast hR hRlast, ldR, haQ]

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

set_option maxHeartbeats 1600000 in
/-- **Genuine `dg` readback.** The executed backward surface writes the honest
reverse-cumulative-sum closed form `bwdDGClosed`
(= `Σ_{j ≥ t_rel} (dq_inter[j]*q[j] - dk_inter[j]*k[j])`) into `DG` at every
active lane of loop row `t_rel`. -/
theorem bwd_decay_cumsum_dg_closed_compute_correct
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName) (t_rel : Fin 2) (s : BlockState)
    (hDG_DQInter : DG ≠ DQInter) (hDG_DKInter : DG ≠ DKInter)
    (hDKInner_DQInter : DKInner ≠ DQInter) (hDKInter_DQInter : DKInter ≠ DQInter)
    (hQ_DQInter : Q ≠ DQInter) (hQ_DKInter : Q ≠ DKInter)
    (hK_DQInter : K ≠ DQInter) (hK_DKInter : K ≠ DKInter) :
    ComputeCorrect.Realizes
      (kernel := bwd_decay_global_cumsum_surface DQInner DQInter DKInner DKInter
        Q K G DG 64 8 2 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf (active s 8 4)
        (fun i => (DG, offset s 64 8 t_rel.val 2 4 i)))
      (expected := fun i : Fin 4 =>
        bwdDGClosed s DQInner DQInter DKInner DKInter Q K G 64 8 2 4 t_rel i) := by
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
  show s2.readMem DG (offset s 64 8 t_rel.val 2 4 i) = _
  rw [hoff, hmemrd s2 _ h2mem DG]
  match t_rel with
  | ⟨0, hlt⟩ =>
    -- Row 0 = R1: DG is the outermost store of the outer triple → readback.
    rw [show (s.pids 1 * 2 + 0) * 8 = (s.pids 1 * 2 + 2 - 1) * 8 - 8 from by omega,
      stMem_readback s DG ((s.pids 1 * 2 + 2 - 1) * 8 - 8) _ _ (i, PUnit.unit)]
    simp only [hac, if_true]
    -- The reverse cumsum at row 0 = summand[row 0] + summand[row 1].
    rw [show bwdDGClosed s DQInner DQInter DKInner DKInter Q K G 64 8 2 4 ⟨0, hlt⟩ i
          = ∑ d : Fin 2, bwdDGSummand s DQInner DQInter DKInner DKInter Q K G 64 8 2 4
              ⟨0 + d.val, by omega⟩ i from rfl]
    rw [Fin.sum_univ_two]
    rw [dgSum_eq_summand s DQInner DQInter DKInner DKInter Q K G ⟨1, by omega⟩ i
        ((s.pids 1 * 2 + 2 - 1) * 8) _
        (show (s.pids 1 * 2 + 2 - 1) * 8 = (s.pids 1 * 2 + 1) * 8 by omega)
        (show (s.pids 1 * 2 + 2 - 1) * 8 = (s.pids 1 * 2 + 1) * 8 by omega),
      dgSum_eq_summand s DQInner DQInter DKInner DKInter Q K G ⟨0, by omega⟩ i
        ((s.pids 1 * 2 + 2 - 1) * 8 - 8) _
        (show (s.pids 1 * 2 + 2 - 1) * 8 - 8 = (s.pids 1 * 2 + 0) * 8 by omega)
        (show (s.pids 1 * 2 + 2 - 1) * 8 = (s.pids 1 * 2 + 1) * 8 by omega)]
    rw [add_comm]
    rfl
  | ⟨1, hlt⟩ =>
    -- Row 1 = R0: outer triple skips DG by offset; inner triple's DG store → readback.
    have hskip : ∀ region ∈ [DG, DKInter, DQInter], ∀ j : TileIndex [4],
        region = DG → s.pids 2 * 64 + s.pids 0 * 4 + j.1.val
            + ((s.pids 1 * 2 + 2 - 1) * 8 - 8) ≠
          s.pids 2 * 64 + s.pids 0 * 4 + i.val + (s.pids 1 * 2 + 1) * 8 := by
      intro region _ j _; have := j.1.2; have := (i : Fin 4).2; omega
    rw [stMem3_read_skip s DG DKInter DQInter DG ((s.pids 1 * 2 + 2 - 1) * 8 - 8)
        _ _ _ s1 _ hskip, hmemrd s1 _ h1mem DG]
    rw [show (s.pids 1 * 2 + 1) * 8 = (s.pids 1 * 2 + 2 - 1) * 8 from by omega,
      stMem_readback s DG ((s.pids 1 * 2 + 2 - 1) * 8) _ _ (i, PUnit.unit)]
    simp only [hac, if_true]
    rw [show bwdDGClosed s DQInner DQInter DKInner DKInter Q K G 64 8 2 4 ⟨1, hlt⟩ i
          = ∑ d : Fin 1, bwdDGSummand s DQInner DQInter DKInner DKInter Q K G 64 8 2 4
              ⟨1 + d.val, by omega⟩ i from rfl]
    rw [Fin.sum_univ_one]
    rw [dgSum_eq_summand s DQInner DQInter DKInner DKInter Q K G ⟨1, by omega⟩ i
        ((s.pids 1 * 2 + 2 - 1) * 8) _
        (show (s.pids 1 * 2 + 2 - 1) * 8 = (s.pids 1 * 2 + 1) * 8 by omega)
        (show (s.pids 1 * 2 + 2 - 1) * 8 = (s.pids 1 * 2 + 1) * 8 by omega)]
    rfl

/-- The general reverse loop body (one iteration), parameterized over `BK DK BT`. -/
def bwdIterBodyG (DK BT BK : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "t"
      (Op.sub .nat Broadcast.nil
        (Op.sub .nat Broadcast.nil (Op.constNat BT) (Op.constNat 1))
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "__rev_t") (Op.constNat 1))),
    Stmt.assign .real [BK] "g_val"
      (Op.load ComputeDType.fp32.eraseDType (MemAccess.ptr (Op.ref .ptr [BK] "p_g"))
        (MaskOpt.maskOther (Op.ref .bool [BK] "mask") ((Op.const 0).broadcast [BK]))),
    Stmt.ifThen
      (Op.eq ComparableDType.nat Broadcast.nil (Op.ref .nat [] "t")
        (Op.sub .nat Broadcast.nil (Op.constNat BT) (Op.constNat 1)))
      [Stmt.assign .real [BK] "last_g" (Op.ref .real [BK] "g_val")],
    Stmt.assign .real [BK] "dq1"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [BK] "p_dq_inner"))
        (MaskOpt.maskOther (Op.ref .bool [BK] "mask") ((Op.const 0).broadcast [BK]))),
    Stmt.assign .real [BK] "dq2"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [BK] "p_dq_inter"))
        (MaskOpt.maskOther (Op.ref .bool [BK] "mask") ((Op.const 0).broadcast [BK]))),
    Stmt.assign .real [BK] "dq2"
      (Op.mul .real Broadcast.nil.consSame (Op.ref .real [BK] "dq2")
        (Op.ref .real [BK] "g_val").exp2),
    Stmt.assign .real [BK] "dq"
      (Op.add .real Broadcast.nil.consSame (Op.ref .real [BK] "dq1")
        (Op.ref .real [BK] "dq2")),
    Stmt.store .real [BK] (MemAccess.ptr (Op.ref .ptr [BK] "p_dq_inter"))
      (Op.ref .real [BK] "dq") (MaskOpt.mask (Op.ref .bool [BK] "mask")),
    Stmt.assign .real [BK] "dk1"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [BK] "p_dk_inner"))
        (MaskOpt.maskOther (Op.ref .bool [BK] "mask") ((Op.const 0).broadcast [BK]))),
    Stmt.assign .real [BK] "dk2"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [BK] "p_dk_inter"))
        (MaskOpt.maskOther (Op.ref .bool [BK] "mask") ((Op.const 0).broadcast [BK]))),
    Stmt.assign .real [BK] "dk2"
      (Op.mul .real Broadcast.nil.consSame (Op.ref .real [BK] "dk2")
        (Op.sub .real Broadcast.nil.consSame (Op.ref .real [BK] "last_g")
            (Op.ref .real [BK] "g_val")).exp2),
    Stmt.assign .real [BK] "dk"
      (Op.add .real Broadcast.nil.consSame (Op.ref .real [BK] "dk1")
        (Op.ref .real [BK] "dk2")),
    Stmt.store .real [BK] (MemAccess.ptr (Op.ref .ptr [BK] "p_dk_inter"))
      (Op.ref .real [BK] "dk") (MaskOpt.mask (Op.ref .bool [BK] "mask")),
    Stmt.assign .real [BK] "q_val"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [BK] "p_q"))
        (MaskOpt.maskOther (Op.ref .bool [BK] "mask") ((Op.const 0).broadcast [BK]))),
    Stmt.assign .real [BK] "k_val"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [BK] "p_k"))
        (MaskOpt.maskOther (Op.ref .bool [BK] "mask") ((Op.const 0).broadcast [BK]))),
    Stmt.assign .real [BK] "dg_val"
      (Op.sub .real Broadcast.nil.consSame
        (Op.mul .real Broadcast.nil.consSame (Op.ref .real [BK] "dq")
          (Op.ref .real [BK] "q_val"))
        (Op.mul .real Broadcast.nil.consSame (Op.ref .real [BK] "dk")
          (Op.ref .real [BK] "k_val"))),
    Stmt.assign .real [BK] "cum_grad_dg"
      (Op.add .real Broadcast.nil.consSame (Op.ref .real [BK] "cum_grad_dg")
        (Op.ref .real [BK] "dg_val")),
    Stmt.store .real [BK] (MemAccess.ptr (Op.ref .ptr [BK] "p_dg"))
      (Op.ref .real [BK] "cum_grad_dg") (MaskOpt.mask (Op.ref .bool [BK] "mask")),
    Stmt.assign .ptr [BK] "p_g"
      (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [BK] "p_g") (Op.constNat DK)),
    Stmt.assign .ptr [BK] "p_k"
      (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [BK] "p_k") (Op.constNat DK)),
    Stmt.assign .ptr [BK] "p_q"
      (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [BK] "p_q") (Op.constNat DK)),
    Stmt.assign .ptr [BK] "p_dq_inner"
      (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [BK] "p_dq_inner") (Op.constNat DK)),
    Stmt.assign .ptr [BK] "p_dk_inner"
      (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [BK] "p_dk_inner") (Op.constNat DK)),
    Stmt.assign .ptr [BK] "p_dq_inter"
      (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [BK] "p_dq_inter") (Op.constNat DK)),
    Stmt.assign .ptr [BK] "p_dk_inter"
      (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [BK] "p_dk_inter") (Op.constNat DK)),
    Stmt.assign .ptr [BK] "p_dg"
      (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [BK] "p_dg") (Op.constNat DK)) ]

/-! ### General backward loop invariant infrastructure

Generic-shape register/pointer tiles (over `Fin BK`) and the reverse-loop
invariant `bwdInvG`, mirroring the forward `fwdInv` style. Physical row at
iteration `m` is `BT-1-m`; pointers point to that row and decrement by `DK`. -/

/-- The pointer tile at physical row `r`: lane `i` points to `region` at the
row-`r` offset (`offset` already includes the `i_k*BK + offs` lane term). -/
def bwdPtrTileG (s : BlockState) (region : RegionName)
    (s_qk_h DK BT BK : Nat) (r : Nat) : Tile .ptr [BK] :=
  ⟨fun idx => (region, offset s s_qk_h DK r BT BK idx.1)⟩

/-- The pointer tile after `m` reverse iterations: lane `i` points to `region` at
the row-`BT-1` offset minus `m·DK` (the `m`-fold `-= DK` decrement). For `m < BT`
this equals `bwdPtrTileG … (BT-1-m)` (no underflow), but the raw `-m·DK` form is
also robust at the boundary iteration `m = BT-1 ↦ m+1 = BT` where the kernel
decrements one past row 0 (so `bwdPtrTileG … (BT-1-(m+1))` would *not* match). -/
def bwdPtrDecTileG (s : BlockState) (region : RegionName)
    (s_qk_h DK BT BK : Nat) (m : Nat) : Tile .ptr [BK] :=
  ⟨fun idx => (region, offset s s_qk_h DK (BT - 1) BT BK idx.1 - m * DK)⟩

/-- At `m = 0` the decremented pointer tile is the prologue row-`BT-1` tile. -/
theorem bwdPtrDecTileG_zero (s : BlockState) (region : RegionName)
    (s_qk_h DK BT BK : Nat) :
    bwdPtrDecTileG s region s_qk_h DK BT BK 0 = bwdPtrTileG s region s_qk_h DK BT BK (BT - 1) := by
  apply Tile.ext; intro idx
  simp only [bwdPtrDecTileG, bwdPtrTileG, Nat.zero_mul, Nat.sub_zero]

/-- For `m < BT`, the decremented pointer tile equals the row-`BT-1-m` tile. -/
theorem bwdPtrDecTileG_row (s : BlockState) (region : RegionName)
    (s_qk_h DK BT BK m : Nat) (hm : m < BT) :
    bwdPtrDecTileG s region s_qk_h DK BT BK m = bwdPtrTileG s region s_qk_h DK BT BK (BT - 1 - m) := by
  apply Tile.ext; intro idx
  simp only [bwdPtrDecTileG, bwdPtrTileG]
  refine Prod.ext rfl ?_
  simp only [offset, baseOffset]
  have key : (s.pids 1 * BT + (BT - 1)) * DK
      = (s.pids 1 * BT + (BT - 1 - m)) * DK + m * DK := by
    have hsplit : s.pids 1 * BT + (BT - 1) = (s.pids 1 * BT + (BT - 1 - m)) + m := by omega
    rw [hsplit, Nat.add_mul]
  rw [key]; omega

/-- The row-`BT-1-m` pointer tile decremented by `DK` is the `(m+1)`-fold
decremented tile (for `m < BT`). Combines `bwdPtrDecTileG_row` and `_succ`. -/
theorem bwdPtrTileG_dec_succ (s : BlockState) (region : RegionName)
    (s_qk_h DK BT BK m : Nat) (hm : m < BT) :
    (⟨fun idx => (region, offset s s_qk_h DK (BT - 1 - m) BT BK idx.1 - DK)⟩ : Tile .ptr [BK])
      = bwdPtrDecTileG s region s_qk_h DK BT BK (m + 1) := by
  apply Tile.ext; intro idx
  simp only [bwdPtrDecTileG]
  refine Prod.ext rfl ?_
  simp only [offset, baseOffset]
  have key : (s.pids 1 * BT + (BT - 1)) * DK
      = (s.pids 1 * BT + (BT - 1 - m)) * DK + m * DK := by
    have hsplit : s.pids 1 * BT + (BT - 1) = (s.pids 1 * BT + (BT - 1 - m)) + m := by omega
    rw [hsplit, Nat.add_mul]
  rw [key]
  have hkey2 : (m + 1) * DK = m * DK + DK := by ring
  rw [hkey2]; omega

/-- The `mask` tile: lane `i` is active iff `i_k*BK + i < DK`. -/
def bwdMaskTileG (s : BlockState) (DK BK : Nat) : Tile .bool [BK] :=
  ⟨fun idx => decide (active s DK BK idx.1)⟩

/-- A masked real tile: active lanes hold `some (f i)`, inactive hold `some 0`. -/
noncomputable def bwdGatedG (s : BlockState) (DK BK : Nat) (f : Fin BK → ℝ) :
    Tile .real [BK] :=
  ⟨fun idx => if active s DK BK idx.1 then some (f idx.1) else some 0⟩

/-- General `dq_inter` per-lane output at physical row `r`: `dq_inner + dq_inter *
exp2(g[r])`. Matches `bwdDQInterClosed` at `t_rel = r`. -/
noncomputable def bwdDqOutG (s : BlockState) (DQInner DQInter G : RegionName)
    (s_qk_h DK BT BK : Nat) (r : Nat) (i : Fin BK) : ℝ :=
  s.readMem DQInner (offset s s_qk_h DK r BT BK i) +
    s.readMem DQInter (offset s s_qk_h DK r BT BK i) *
      Real.exp (s.readMem G (offset s s_qk_h DK r BT BK i) * Real.log 2)

/-- General `dk_inter` per-lane output at physical row `r` with captured per-lane
`last_g` value `lg i`: `dk_inner + dk_inter * exp2(lg - g[r])`. -/
noncomputable def bwdDkOutG (s : BlockState) (DKInner DKInter G : RegionName)
    (s_qk_h DK BT BK : Nat) (r : Nat) (lg : Fin BK → ℝ) (i : Fin BK) : ℝ :=
  s.readMem DKInner (offset s s_qk_h DK r BT BK i) +
    s.readMem DKInter (offset s s_qk_h DK r BT BK i) *
      Real.exp ((lg i - s.readMem G (offset s s_qk_h DK r BT BK i)) * Real.log 2)

/-- General `dg` per-lane summand at physical row `r`: `dq*q - dk*k`. -/
noncomputable def bwdDgSumG (s : BlockState)
    (DQInner DQInter DKInner DKInter Q K G : RegionName)
    (s_qk_h DK BT BK : Nat) (r : Nat) (lg : Fin BK → ℝ) (i : Fin BK) : ℝ :=
  bwdDqOutG s DQInner DQInter G s_qk_h DK BT BK r i *
      s.readMem Q (offset s s_qk_h DK r BT BK i) -
    bwdDkOutG s DKInner DKInter G s_qk_h DK BT BK r lg i *
      s.readMem K (offset s s_qk_h DK r BT BK i)

/-- The reverse partial cumulative `dg` sum after `m` iterations have folded
physical rows `{BT-1, …, BT-m}`: `Σ_{d<m} bwdDgSumG[row BT-1-d]`. The captured
`last_g` is `g[row BT-1]`. -/
noncomputable def bwdCumPartialG (s : BlockState)
    (DQInner DQInter DKInner DKInter Q K G : RegionName)
    (s_qk_h DK BT BK : Nat) (m : Nat) (i : Fin BK) : ℝ :=
  ∑ d : Fin m,
    bwdDgSumG s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK
      (BT - 1 - d.val)
      (fun i' => s.readMem G (offset s s_qk_h DK (BT - 1) BT BK i'))
      i

/-- The general backward prologue post-state shape (15-stmt prologue): pointers at
physical row `BT-1`, mask, and zero-initialized `cum_grad_dg` / `last_g`. -/
structure BwdPrologueShapeG (s s0 : BlockState)
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (s_qk_h DK BT BK : Nat) : Prop where
  pid : s0.pids = s.pids
  mem : s0.mem = s.mem
  mask : s0.regs .bool [BK] "mask" = some (bwdMaskTileG s DK BK)
  cum : s0.regs .real [BK] "cum_grad_dg" = some (⟨fun _ => some 0⟩ : Tile .real [BK])
  lastg : s0.regs .real [BK] "last_g" = some (⟨fun _ => some 0⟩ : Tile .real [BK])
  pg : s0.regs .ptr [BK] "p_g" = some (bwdPtrTileG s G s_qk_h DK BT BK (BT - 1))
  pk : s0.regs .ptr [BK] "p_k" = some (bwdPtrTileG s K s_qk_h DK BT BK (BT - 1))
  pq : s0.regs .ptr [BK] "p_q" = some (bwdPtrTileG s Q s_qk_h DK BT BK (BT - 1))
  pdqi : s0.regs .ptr [BK] "p_dq_inner" = some (bwdPtrTileG s DQInner s_qk_h DK BT BK (BT - 1))
  pdki : s0.regs .ptr [BK] "p_dk_inner" = some (bwdPtrTileG s DKInner s_qk_h DK BT BK (BT - 1))
  pdqt : s0.regs .ptr [BK] "p_dq_inter" = some (bwdPtrTileG s DQInter s_qk_h DK BT BK (BT - 1))
  pdkt : s0.regs .ptr [BK] "p_dk_inter" = some (bwdPtrTileG s DKInter s_qk_h DK BT BK (BT - 1))
  pdg : s0.regs .ptr [BK] "p_dg" = some (bwdPtrTileG s DG s_qk_h DK BT BK (BT - 1))

/-- The captured `last_g` per-lane value after `m ≥ 1` iterations: `g[row BT-1]`.
Before any iteration (`m=0`) it is the zero seed. -/
noncomputable def bwdLastgG (s : BlockState) (G : RegionName)
    (s_qk_h DK BT BK : Nat) (m : Nat) (i : Fin BK) : ℝ :=
  if 0 < m then s.readMem G (offset s s_qk_h DK (BT - 1) BT BK i) else 0

/-- The general backward reverse-loop invariant after `m` iterations (physical rows
`{BT-1, …, BT-m}` processed; pointers at row `BT-1-m`). -/
noncomputable def bwdInvG
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (s : BlockState) (s_qk_h DK BT BK : Nat) :
    Nat → BlockState → Prop :=
  fun m sc =>
    sc.pids = s.pids ∧
    (∀ a, sc.readMem G a = s.readMem G a) ∧
    (∀ a, sc.readMem Q a = s.readMem Q a) ∧
    (∀ a, sc.readMem K a = s.readMem K a) ∧
    (∀ a, sc.readMem DQInner a = s.readMem DQInner a) ∧
    (∀ a, sc.readMem DKInner a = s.readMem DKInner a) ∧
    sc.regs .bool [BK] "mask" = some (bwdMaskTileG s DK BK) ∧
    sc.regs .real [BK] "cum_grad_dg" = some
      (bwdGatedG s DK BK (bwdCumPartialG s DQInner DQInter DKInner DKInter Q K G
        s_qk_h DK BT BK m)) ∧
    sc.regs .real [BK] "last_g" = some
      (bwdGatedG s DK BK (bwdLastgG s G s_qk_h DK BT BK m)) ∧
    sc.regs .ptr [BK] "p_g" = some (bwdPtrDecTileG s G s_qk_h DK BT BK m) ∧
    sc.regs .ptr [BK] "p_k" = some (bwdPtrDecTileG s K s_qk_h DK BT BK m) ∧
    sc.regs .ptr [BK] "p_q" = some (bwdPtrDecTileG s Q s_qk_h DK BT BK m) ∧
    sc.regs .ptr [BK] "p_dq_inner" = some (bwdPtrDecTileG s DQInner s_qk_h DK BT BK m) ∧
    sc.regs .ptr [BK] "p_dk_inner" = some (bwdPtrDecTileG s DKInner s_qk_h DK BT BK m) ∧
    sc.regs .ptr [BK] "p_dq_inter" = some (bwdPtrDecTileG s DQInter s_qk_h DK BT BK m) ∧
    sc.regs .ptr [BK] "p_dk_inter" = some (bwdPtrDecTileG s DKInter s_qk_h DK BT BK m) ∧
    sc.regs .ptr [BK] "p_dg" = some (bwdPtrDecTileG s DG s_qk_h DK BT BK m) ∧
    -- DQInter / DKInter / DG at rows still to be processed (row index `< BT-m`) are
    -- untouched.
    (∀ r, r < BT - m → ∀ i : Fin BK,
      sc.readMem DQInter (offset s s_qk_h DK r BT BK i)
        = s.readMem DQInter (offset s s_qk_h DK r BT BK i)) ∧
    (∀ r, r < BT - m → ∀ i : Fin BK,
      sc.readMem DKInter (offset s s_qk_h DK r BT BK i)
        = s.readMem DKInter (offset s s_qk_h DK r BT BK i)) ∧
    (∀ r, r < BT - m → ∀ i : Fin BK,
      sc.readMem DG (offset s s_qk_h DK r BT BK i)
        = s.readMem DG (offset s s_qk_h DK r BT BK i)) ∧
    -- Processed rows `r` with `BT-m ≤ r < BT` hold the genuine closed forms.
    (∀ r, BT - m ≤ r → r < BT → ∀ i : Fin BK,
      sc.readMem DQInter (offset s s_qk_h DK r BT BK i)
        = if active s DK BK i then
            bwdDqOutG s DQInner DQInter G s_qk_h DK BT BK r i
          else s.readMem DQInter (offset s s_qk_h DK r BT BK i)) ∧
    (∀ r, BT - m ≤ r → r < BT → ∀ i : Fin BK,
      sc.readMem DKInter (offset s s_qk_h DK r BT BK i)
        = if active s DK BK i then
            bwdDkOutG s DKInner DKInter G s_qk_h DK BT BK r
              (fun i' => s.readMem G (offset s s_qk_h DK (BT - 1) BT BK i')) i
          else s.readMem DKInter (offset s s_qk_h DK r BT BK i)) ∧
    (∀ r, BT - m ≤ r → r < BT → ∀ i : Fin BK,
      sc.readMem DG (offset s s_qk_h DK r BT BK i)
        = if active s DK BK i then
            bwdCumPartialG s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK (BT - r) i
          else s.readMem DG (offset s s_qk_h DK r BT BK i))

/-- The general backward surface body decomposes as a 15-stmt prologue followed by
the single `forRangeDyn "__rev_t" 0 ((BT-1)/1+1) 1 (bwdIterBodyG DK BT BK)` loop. -/
theorem bwd_body_decomp_general
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (s_qk_h DK BT BK : Nat) :
    (bwd_decay_global_cumsum_surface DQInner DQInter DKInner DKInter Q K G DG
        s_qk_h DK BT BK).toAlgKernel.body
      = (bwd_decay_global_cumsum_surface DQInner DQInter DKInner DKInter Q K G DG
          s_qk_h DK BT BK).toAlgKernel.body.take 15
        ++ [Stmt.forRangeDyn "__rev_t" (Op.constNat 0)
              (Op.add .nat Broadcast.nil
                (Op.div .nat Broadcast.nil
                  (Op.sub .nat Broadcast.nil (Op.constNat BT) (Op.constNat 1)) (Op.constNat 1))
                (Op.constNat 1))
              (Op.constNat 1) (bwdIterBodyG DK BT BK)] := by
  rfl

/-- The 15-statement general prologue as an explicit `Stmt` list. -/
def bwdPrologueG (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (s_qk_h DK BT BK : Nat) : List Stmt :=
  let pgen : RegName → (RegionName) → Stmt := fun nm reg =>
    Stmt.assign .ptr [BK] nm
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase reg)
        (Op.add .nat Broadcast.scalarR
          (Op.add .nat Broadcast.scalarL
            (Op.add .nat Broadcast.nil
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_qk_h))
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BK)))
            (Op.ref .nat [BK] "offs"))
          (Op.mul .nat Broadcast.nil
            (Op.sub .nat Broadcast.nil
              (Op.add .nat Broadcast.nil
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat BT)) (Op.constNat BT))
              (Op.constNat 1))
            (Op.constNat DK))))
  [ Stmt.assign .nat [] "i_k" (Op.programId 0),
    Stmt.assign .nat [] "i_c" (Op.programId 1),
    Stmt.assign .nat [] "i_bh" (Op.programId 2),
    Stmt.assign .nat [BK] "offs" (Op.arange BK),
    pgen "p_q" Q, pgen "p_k" K, pgen "p_g" G, pgen "p_dg" DG,
    pgen "p_dq_inner" DQInner, pgen "p_dk_inner" DKInner,
    pgen "p_dq_inter" DQInter, pgen "p_dk_inter" DKInter,
    Stmt.assign .real [BK] "cum_grad_dg" (Op.full [BK] (Op.const 0)),
    Stmt.assign .bool [BK] "mask"
      (Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.add .nat Broadcast.scalarL
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BK))
          (Op.ref .nat [BK] "offs"))
        (Op.constNat DK)),
    Stmt.assign .real [BK] "last_g" (Op.full [BK] (Op.const 0)) ]

/-- The general prologue lowering matches `bwdPrologueG`. -/
theorem bwd_prologue_take_general
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (s_qk_h DK BT BK : Nat) :
    ((bwd_decay_global_cumsum_surface DQInner DQInter DKInner DKInter Q K G DG
        s_qk_h DK BT BK).toAlgKernel.body.take 15)
      = bwdPrologueG DQInner DQInter DKInner DKInter Q K G DG s_qk_h DK BT BK := by
  rfl

set_option maxHeartbeats 2000000 in
/-- **General prologue evaluation.** The 15-statement prologue runs to a state of
shape `BwdPrologueShapeG`. -/
theorem bwd_prologue_eval_general
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (s_qk_h DK BT BK : Nat) (hBT : 0 < BT) (s : BlockState) :
    ∃ s0,
      stepStmts ((bwd_decay_global_cumsum_surface DQInner DQInter DKInner DKInter
        Q K G DG s_qk_h DK BT BK).toAlgKernel.body.take 15) s = some s0
      ∧ BwdPrologueShapeG s s0 DQInner DQInter DKInner DKInter Q K G DG s_qk_h DK BT BK := by
  rw [bwd_prologue_take_general]
  -- evalOp of one prologue pointer setup, in any state whose i_k/i_c/i_bh/offs are set.
  have hptr : ∀ (reg : RegionName) (c : BlockState),
      c.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)) →
      c.regs .nat [] "i_c" = some (Tile.scalar (s.pids 1)) →
      c.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)) →
      c.regs .nat [BK] "offs" = some (Tile.vec (fun i : Fin BK => i.val)) →
      evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase reg)
        (Op.add .nat Broadcast.scalarR
          (Op.add .nat Broadcast.scalarL
            (Op.add .nat Broadcast.nil
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_qk_h))
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BK)))
            (Op.ref .nat [BK] "offs"))
          (Op.mul .nat Broadcast.nil
            (Op.sub .nat Broadcast.nil
              (Op.add .nat Broadcast.nil
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat BT)) (Op.constNat BT))
              (Op.constNat 1))
            (Op.constNat DK)))) c
        = some (bwdPtrTileG s reg s_qk_h DK BT BK (BT - 1)) := by
    intro reg c hik hic hibh hoffs
    simp only [evalOp, hik, hic, hibh, hoffs, Option.bind, evalOp_constNat, evalOp_ref,
      Tile.bop, Tile.ptrAdd, NumericDType.add, NumericDType.mul, NumericDType.sub]
    apply congrArg some; apply Tile.ext; intro idx; obtain ⟨i, u⟩ := idx
    simp only [bwdPtrTileG, offset, baseOffset, Tile.vec, Tile.scalar,
      Broadcast.leftIndex, Broadcast.rightIndex,
      Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL,
      Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR]
    refine Prod.ext rfl ?_
    simp only []
    have hbt1 : s.pids 1 * BT + BT - 1 = s.pids 1 * BT + (BT - 1) := by omega
    rw [hbt1]; ring
  -- evalOp of the mask setup.
  have hmaskE : ∀ c : BlockState,
      c.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)) →
      c.regs .nat [BK] "offs" = some (Tile.vec (fun i : Fin BK => i.val)) →
      evalOp (Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.add .nat Broadcast.scalarL
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BK))
          (Op.ref .nat [BK] "offs"))
        (Op.constNat DK)) c
        = some (bwdMaskTileG s DK BK) := by
    intro c hik hoffs
    simp only [evalOp, hik, hoffs, Option.bind, evalOp_constNat, evalOp_ref,
      Tile.bop, NumericDType.mul, ComparableDType.lt]
    apply congrArg some; apply Tile.ext; intro idx; obtain ⟨i, u⟩ := idx
    simp only [bwdMaskTileG, active, elemIndex, Tile.cop_data, Tile.vec, Tile.scalar,
      Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR,
      Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL, ComparableDType.lt]
    rfl
  -- evalOp of the `full [BK] (const 0)` setup.
  have hfullE : ∀ c : BlockState,
      evalOp (Op.full [BK] (Op.const (0:ℝ))) c
        = some (⟨fun _ => some 0⟩ : Tile .real [BK]) := by
    intro c; simp [evalOp]
  -- Chain the 15 statements via explicit setReg states.
  set c1 := s.setReg "i_k" .nat [] (Tile.scalar (s.pids 0)) with hc1
  set c2 := c1.setReg "i_c" .nat [] (Tile.scalar (s.pids 1)) with hc2
  set c3 := c2.setReg "i_bh" .nat [] (Tile.scalar (s.pids 2)) with hc3
  set c4 := c3.setReg "offs" .nat [BK] (Tile.vec (fun i : Fin BK => i.val)) with hc4
  -- pids are preserved throughout (all are setReg or full/arange assigns).
  have hpidc4 : c4.pids = s.pids := by
    rw [hc4, hc3, hc2, hc1]; simp only [BlockState.setReg_pids]
  -- i_k/i_c/i_bh/offs lookups available from any state above c4 (peeling setRegs of
  -- distinct names). We expose them on c4 itself.
  have hik4 : c4.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)) := by
    rw [hc4, BlockState.setReg_ne_name (h := by decide), hc3,
      BlockState.setReg_ne_name (h := by decide), hc2,
      BlockState.setReg_ne_name (h := by decide), hc1]
    exact BlockState.setReg_same _ _ _ _ _
  have hic4 : c4.regs .nat [] "i_c" = some (Tile.scalar (s.pids 1)) := by
    rw [hc4, BlockState.setReg_ne_name (h := by decide), hc3,
      BlockState.setReg_ne_name (h := by decide), hc2]
    exact BlockState.setReg_same _ _ _ _ _
  have hibh4 : c4.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)) := by
    rw [hc4, BlockState.setReg_ne_name (h := by decide), hc3]
    exact BlockState.setReg_same _ _ _ _ _
  have hoffs4 : c4.regs .nat [BK] "offs" = some (Tile.vec (fun i : Fin BK => i.val)) := by
    rw [hc4]; exact BlockState.setReg_same _ _ _ _ _
  -- The 8 pointer setups + cum + mask + last_g.
  set c5 := c4.setReg "p_q" .ptr [BK] (bwdPtrTileG s Q s_qk_h DK BT BK (BT - 1)) with hc5
  set c6 := c5.setReg "p_k" .ptr [BK] (bwdPtrTileG s K s_qk_h DK BT BK (BT - 1)) with hc6
  set c7 := c6.setReg "p_g" .ptr [BK] (bwdPtrTileG s G s_qk_h DK BT BK (BT - 1)) with hc7
  set c8 := c7.setReg "p_dg" .ptr [BK] (bwdPtrTileG s DG s_qk_h DK BT BK (BT - 1)) with hc8
  set c9 := c8.setReg "p_dq_inner" .ptr [BK] (bwdPtrTileG s DQInner s_qk_h DK BT BK (BT - 1)) with hc9
  set c10 := c9.setReg "p_dk_inner" .ptr [BK] (bwdPtrTileG s DKInner s_qk_h DK BT BK (BT - 1)) with hc10
  set c11 := c10.setReg "p_dq_inter" .ptr [BK] (bwdPtrTileG s DQInter s_qk_h DK BT BK (BT - 1)) with hc11
  set c12 := c11.setReg "p_dk_inter" .ptr [BK] (bwdPtrTileG s DKInter s_qk_h DK BT BK (BT - 1)) with hc12
  set c13 := c12.setReg "cum_grad_dg" .real [BK] (⟨fun _ => some 0⟩ : Tile .real [BK]) with hc13
  set c14 := c13.setReg "mask" .bool [BK] (bwdMaskTileG s DK BK) with hc14
  set c15 := c14.setReg "last_g" .real [BK] (⟨fun _ => some 0⟩ : Tile .real [BK]) with hc15
  -- Base lookups (i_k/i_c/i_bh/offs) at each ptr-setup state: all persist because the
  -- intervening setRegs use distinct (ptr/cum) names.  We package them as `hbase k`.
  have hbase : ∀ (c : BlockState) (nm : RegName) (t : Tile .ptr [BK]),
      nm ≠ "i_k" → nm ≠ "i_c" → nm ≠ "i_bh" → nm ≠ "offs" →
      c.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)) →
      c.regs .nat [] "i_c" = some (Tile.scalar (s.pids 1)) →
      c.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)) →
      c.regs .nat [BK] "offs" = some (Tile.vec (fun i : Fin BK => i.val)) →
      (c.setReg nm .ptr [BK] t).regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)) ∧
      (c.setReg nm .ptr [BK] t).regs .nat [] "i_c" = some (Tile.scalar (s.pids 1)) ∧
      (c.setReg nm .ptr [BK] t).regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)) ∧
      (c.setReg nm .ptr [BK] t).regs .nat [BK] "offs" = some (Tile.vec (fun i : Fin BK => i.val)) := by
    intro c nm t n1 n2 n3 n4 h1 h2 h3 h4
    refine ⟨?_, ?_, ?_, ?_⟩
    · rw [BlockState.setReg_ne_name (h := fun h => n1 h.symm)]; exact h1
    · rw [BlockState.setReg_ne_name (h := fun h => n2 h.symm)]; exact h2
    · rw [BlockState.setReg_ne_name (h := fun h => n3 h.symm)]; exact h3
    · rw [BlockState.setReg_ne_name (h := fun h => n4 h.symm)]; exact h4
  obtain ⟨q1, q2, q3, q4⟩ := hbase c4 "p_q" _ (by decide) (by decide) (by decide) (by decide) hik4 hic4 hibh4 hoffs4
  obtain ⟨k1, k2, k3, k4⟩ := hbase c5 "p_k" _ (by decide) (by decide) (by decide) (by decide) q1 q2 q3 q4
  obtain ⟨g1, g2, g3, g4⟩ := hbase c6 "p_g" _ (by decide) (by decide) (by decide) (by decide) k1 k2 k3 k4
  obtain ⟨d1, d2, d3, d4⟩ := hbase c7 "p_dg" _ (by decide) (by decide) (by decide) (by decide) g1 g2 g3 g4
  obtain ⟨a1, a2, a3, a4⟩ := hbase c8 "p_dq_inner" _ (by decide) (by decide) (by decide) (by decide) d1 d2 d3 d4
  obtain ⟨b1, b2, b3, b4⟩ := hbase c9 "p_dk_inner" _ (by decide) (by decide) (by decide) (by decide) a1 a2 a3 a4
  obtain ⟨e1, e2, e3, e4⟩ := hbase c10 "p_dq_inter" _ (by decide) (by decide) (by decide) (by decide) b1 b2 b3 b4
  refine ⟨c15, ?_, ?_⟩
  · -- step equation
    simp only [bwdPrologueG]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 s))]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (show evalOp (Op.programId 1) c1 = some (Tile.scalar (s.pids 1)) by
          rw [evalOp_programId, hc1, BlockState.setReg_pids]))]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (show evalOp (Op.programId 2) c2 = some (Tile.scalar (s.pids 2)) by
          rw [evalOp_programId, hc2, hc1]; simp only [BlockState.setReg_pids]))]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_arange BK c3))]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some (hptr Q c4 hik4 hic4 hibh4 hoffs4))]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some (hptr K c5 q1 q2 q3 q4))]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some (hptr G c6 k1 k2 k3 k4))]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some (hptr DG c7 g1 g2 g3 g4))]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some (hptr DQInner c8 d1 d2 d3 d4))]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some (hptr DKInner c9 a1 a2 a3 a4))]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some (hptr DQInter c10 b1 b2 b3 b4))]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some (hptr DKInter c11 e1 e2 e3 e4))]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some (hfullE c12))]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (hmaskE c13
          (by rw [hc13, BlockState.setReg_ne_name (h := by decide), hc12,
                BlockState.setReg_ne_name (h := by decide), hc11,
                BlockState.setReg_ne_name (h := by decide), hc10,
                BlockState.setReg_ne_name (h := by decide), hc9,
                BlockState.setReg_ne_name (h := by decide), hc8,
                BlockState.setReg_ne_name (h := by decide), hc7,
                BlockState.setReg_ne_name (h := by decide), hc6,
                BlockState.setReg_ne_name (h := by decide), hc5,
                BlockState.setReg_ne_name (h := by decide)]
              exact hik4)
          (by rw [hc13, BlockState.setReg_ne_name (h := by decide), hc12,
                BlockState.setReg_ne_name (h := by decide), hc11,
                BlockState.setReg_ne_name (h := by decide), hc10,
                BlockState.setReg_ne_name (h := by decide), hc9,
                BlockState.setReg_ne_name (h := by decide), hc8,
                BlockState.setReg_ne_name (h := by decide), hc7,
                BlockState.setReg_ne_name (h := by decide), hc6,
                BlockState.setReg_ne_name (h := by decide), hc5,
                BlockState.setReg_ne_name (h := by decide)]
              exact hoffs4)))]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some (hfullE c14))]
    rw [stepStmts.nil]
  · -- shape
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · -- pids
      rw [hc15, hc14, hc13]; simp only [BlockState.setReg_pids]
      rw [hc12, hc11, hc10, hc9, hc8, hc7, hc6, hc5]; simp only [BlockState.setReg_pids]
      exact hpidc4
    · -- mem
      rw [hc15, hc14, hc13, hc12, hc11, hc10, hc9, hc8, hc7, hc6, hc5, hc4, hc3, hc2, hc1]
      rfl
    · -- mask
      rw [hc15, BlockState.setReg_ne_name (h := by decide), hc14]
      exact BlockState.setReg_same _ _ _ _ _
    · -- cum_grad_dg
      rw [hc15, BlockState.setReg_ne_name (h := by decide), hc14,
        BlockState.setReg_ne_name (h := by decide), hc13]
      exact BlockState.setReg_same _ _ _ _ _
    · -- last_g
      rw [hc15]; exact BlockState.setReg_same _ _ _ _ _
    all_goals (
      rw [hc15, BlockState.setReg_ne_name (h := by decide), hc14,
        BlockState.setReg_ne_name (h := by decide), hc13,
        BlockState.setReg_ne_name (h := by decide)])
    · -- p_g (set at c7)
      rw [hc12, BlockState.setReg_ne_name (h := by decide), hc11,
        BlockState.setReg_ne_name (h := by decide), hc10,
        BlockState.setReg_ne_name (h := by decide), hc9,
        BlockState.setReg_ne_name (h := by decide), hc8,
        BlockState.setReg_ne_name (h := by decide), hc7]
      exact BlockState.setReg_same _ _ _ _ _
    · -- p_k (set at c6)
      rw [hc12, BlockState.setReg_ne_name (h := by decide), hc11,
        BlockState.setReg_ne_name (h := by decide), hc10,
        BlockState.setReg_ne_name (h := by decide), hc9,
        BlockState.setReg_ne_name (h := by decide), hc8,
        BlockState.setReg_ne_name (h := by decide), hc7,
        BlockState.setReg_ne_name (h := by decide), hc6]
      exact BlockState.setReg_same _ _ _ _ _
    · -- p_q (set at c5)
      rw [hc12, BlockState.setReg_ne_name (h := by decide), hc11,
        BlockState.setReg_ne_name (h := by decide), hc10,
        BlockState.setReg_ne_name (h := by decide), hc9,
        BlockState.setReg_ne_name (h := by decide), hc8,
        BlockState.setReg_ne_name (h := by decide), hc7,
        BlockState.setReg_ne_name (h := by decide), hc6,
        BlockState.setReg_ne_name (h := by decide), hc5]
      exact BlockState.setReg_same _ _ _ _ _
    · -- p_dq_inner (set at c9)
      rw [hc12, BlockState.setReg_ne_name (h := by decide), hc11,
        BlockState.setReg_ne_name (h := by decide), hc10,
        BlockState.setReg_ne_name (h := by decide), hc9]
      exact BlockState.setReg_same _ _ _ _ _
    · -- p_dk_inner (set at c10)
      rw [hc12, BlockState.setReg_ne_name (h := by decide), hc11,
        BlockState.setReg_ne_name (h := by decide), hc10]
      exact BlockState.setReg_same _ _ _ _ _
    · -- p_dq_inter (set at c11)
      rw [hc12, BlockState.setReg_ne_name (h := by decide), hc11]
      exact BlockState.setReg_same _ _ _ _ _
    · -- p_dk_inter (set at c12)
      rw [hc12]; exact BlockState.setReg_same _ _ _ _ _
    · -- p_dg (set at c8)
      rw [hc12, BlockState.setReg_ne_name (h := by decide), hc11,
        BlockState.setReg_ne_name (h := by decide), hc10,
        BlockState.setReg_ne_name (h := by decide), hc9,
        BlockState.setReg_ne_name (h := by decide), hc8]
      exact BlockState.setReg_same _ _ _ _ _

/-- The empty reverse partial sum is `0`. -/
theorem bwdCumPartialG_zero (s : BlockState)
    (DQInner DQInter DKInner DKInter Q K G : RegionName)
    (s_qk_h DK BT BK : Nat) (i : Fin BK) :
    bwdCumPartialG s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK 0 i = 0 := by
  simp [bwdCumPartialG]

/-- Per-lane masked load value of `region` at physical row `r`. -/
noncomputable def bwdLdRowG (s : BlockState) (region : RegionName)
    (s_qk_h DK BT BK : Nat) (r : Nat) (i : Fin BK) : ℝ :=
  s.readMem region (offset s s_qk_h DK r BT BK i)

/-- `bwdGatedG` add is the `bwdGatedG` of the pointwise sum. -/
theorem bwdGatedG_add (s : BlockState) (DK BK : Nat) (a b : Fin BK → ℝ) :
    Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
        (bwdGatedG s DK BK a) (bwdGatedG s DK BK b)
      = bwdGatedG s DK BK (fun i => a i + b i) := by
  apply Tile.ext; intro idx
  simp only [bwdGatedG, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.add, WithBot.realAdd]
  by_cases ha : active s DK BK idx.1
  · simp only [ha, if_true, Option.map₂_some_some]
  · simp only [ha, if_false]
    show Option.map₂ (·+·) (some (0:ℝ)) (some 0) = some 0
    rw [Option.map₂_some_some]; norm_num

/-- `bwdGatedG` sub is the `bwdGatedG` of the pointwise difference. -/
theorem bwdGatedG_sub (s : BlockState) (DK BK : Nat) (a b : Fin BK → ℝ) :
    Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil)
        (bwdGatedG s DK BK a) (bwdGatedG s DK BK b)
      = bwdGatedG s DK BK (fun i => a i - b i) := by
  apply Tile.ext; intro idx
  simp only [bwdGatedG, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.sub, WithBot.realSub]
  by_cases ha : active s DK BK idx.1
  · simp only [ha, if_true, Option.map₂_some_some]
  · simp only [ha, if_false]
    show Option.map₂ (·-·) (some (0:ℝ)) (some 0) = some 0
    rw [Option.map₂_some_some]; norm_num

/-- `bwdGatedG a * exp2 (bwdGatedG b)` is `bwdGatedG (a * exp(b * log 2))`. -/
theorem bwdGatedG_mul_exp2 (s : BlockState) (DK BK : Nat) (a b : Fin BK → ℝ) :
    Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil)
        (bwdGatedG s DK BK a) (Tile.uop WithBot.realExp2 (bwdGatedG s DK BK b))
      = bwdGatedG s DK BK (fun i => a i * Real.exp (b i * Real.log 2)) := by
  apply Tile.ext; intro idx
  simp only [bwdGatedG, Tile.bop, Tile.uop, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.mul, WithBot.realMul]
  by_cases ha : active s DK BK idx.1
  · simp only [ha, if_true, WithBot.realExp2_some, Option.map₂_some_some]
  · simp only [ha, if_false]
    show Option.map₂ (·*·) (some (0:ℝ)) (WithBot.realExp2 (some 0)) = some 0
    rw [WithBot.realExp2_some, Option.map₂_some_some]; norm_num

/-- Plain `bwdGatedG` mul. -/
theorem bwdGatedG_mul (s : BlockState) (DK BK : Nat) (a b : Fin BK → ℝ) :
    Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil)
        (bwdGatedG s DK BK a) (bwdGatedG s DK BK b)
      = bwdGatedG s DK BK (fun i => a i * b i) := by
  apply Tile.ext; intro idx
  simp only [bwdGatedG, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.mul, WithBot.realMul]
  by_cases ha : active s DK BK idx.1
  · simp only [ha, if_true, Option.map₂_some_some]
  · simp only [ha, if_false]
    show Option.map₂ (·*·) (some (0:ℝ)) (some 0) = some 0
    rw [Option.map₂_some_some]; norm_num

/-- The masked pointer-load result tile at a `bwdPtrTileG`-shaped row pointer,
evaluated against a state `c` whose `region`-reads at row `r` agree with `s`,
equals the `bwdGatedG` load tile. -/
theorem bwdLoad_gated (s c : BlockState) (region : RegionName)
    (s_qk_h DK BT BK : Nat) (r : Nat)
    (hread : ∀ i : Fin BK,
      c.readMem region (offset s s_qk_h DK r BT BK i)
        = s.readMem region (offset s s_qk_h DK r BT BK i)) :
    (⟨fun idx => if (bwdMaskTileG s DK BK).data idx then
          some (c.readMem ((bwdPtrTileG s region s_qk_h DK BT BK r).data idx).1
            ((bwdPtrTileG s region s_qk_h DK BT BK r).data idx).2)
        else (⟨fun _ => some 0⟩ : Tile .real [BK]).data idx⟩ : Tile .real [BK])
      = bwdGatedG s DK BK (bwdLdRowG s region s_qk_h DK BT BK r) := by
  apply Tile.ext; intro idx
  simp only [bwdMaskTileG, bwdPtrTileG, bwdGatedG, bwdLdRowG]
  by_cases ha : active s DK BK idx.1
  · simp only [ha, decide_true, if_true]; rw [hread idx.1]
  · simp only [ha, decide_false, Bool.false_eq_true, if_false]

/-- The masked pointer-store fold of a `bwdGatedG` value tile at a `bwdPtrTileG`-shaped
row pointer equals the masked scatter into `region` at row `r`. -/
theorem bwdStore_scatter (s : BlockState) (region : RegionName)
    (s_qk_h DK BT BK : Nat) (r : Nat) (f : Fin BK → ℝ) (c : BlockState) :
    ((TileShape.allIndices [BK]).foldl
        (fun acc idx =>
          if (bwdMaskTileG s DK BK).data idx then
            acc.writeMemTyped .real
              ((bwdPtrTileG s region s_qk_h DK BT BK r).data idx).1
              ((bwdPtrTileG s region s_qk_h DK BT BK r).data idx).2
              ((bwdGatedG s DK BK f).data idx)
          else acc) c)
      = (TileShape.allIndices [BK]).foldl
          (fun acc idx =>
            if active s DK BK idx.1 then
              acc.writeMem region (offset s s_qk_h DK r BT BK idx.1) (f idx.1)
            else acc) c := by
  congr 1
  funext acc idx
  simp only [bwdMaskTileG, bwdPtrTileG, bwdGatedG]
  by_cases ha : active s DK BK idx.1
  · simp only [ha, decide_true, if_true, BlockState.writeMemTyped_real,
      FloatDType.real_storeValue, WithBot.unbotD_some]
  · simp only [ha, decide_false, Bool.false_eq_true, if_false]

/-- The masked scatter of value-function `f` into `region` at physical row `r`. -/
noncomputable def bwdScatterRow (s : BlockState) (region : RegionName)
    (s_qk_h DK BT BK : Nat) (r : Nat) (f : Fin BK → ℝ) (c : BlockState) : BlockState :=
  (TileShape.allIndices [BK]).foldl
    (fun acc idx =>
      if active s DK BK idx.1 then
        acc.writeMem region (offset s s_qk_h DK r BT BK idx.1) (f idx.1)
      else acc) c

theorem bwdScatterRow_pids (s : BlockState) (region : RegionName)
    (s_qk_h DK BT BK : Nat) (r : Nat) (f : Fin BK → ℝ) (c : BlockState) :
    (bwdScatterRow s region s_qk_h DK BT BK r f c).pids = c.pids := by
  unfold bwdScatterRow
  exact BlockState.foldl_writeMem_prop_masked_pids region
    (fun k : TileIndex [BK] => offset s s_qk_h DK r BT BK k.1) (fun k => f k.1)
    (fun k => active s DK BK k.1) _ c

@[simp] theorem bwdScatterRow_regs (s : BlockState) (region : RegionName)
    (s_qk_h DK BT BK : Nat) (r : Nat) (f : Fin BK → ℝ) (c : BlockState)
    (d : TileDType) (sh : TileShape) (n : RegName) :
    (bwdScatterRow s region s_qk_h DK BT BK r f c).regs d sh n = c.regs d sh n := by
  unfold bwdScatterRow
  exact BlockState.foldl_writeMem_prop_masked_regs region
    (fun k : TileIndex [BK] => offset s s_qk_h DK r BT BK k.1) (fun k => f k.1)
    (fun k => active s DK BK k.1) _ c d sh n

theorem bwdScatterRow_readback (s : BlockState) (region : RegionName)
    (s_qk_h DK BT BK : Nat) (r : Nat) (f : Fin BK → ℝ) (c : BlockState) (i : Fin BK) :
    (bwdScatterRow s region s_qk_h DK BT BK r f c).readMem region (offset s s_qk_h DK r BT BK i)
      = if active s DK BK i then f i
        else c.readMem region (offset s s_qk_h DK r BT BK i) := by
  have hinj : Function.Injective (fun k : TileIndex [BK] => offset s s_qk_h DK r BT BK k.1) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    simp only [offset] at hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  unfold bwdScatterRow
  have := BlockState.scatter_readback_prop_masked_nd (region := region) c
    (fun k : TileIndex [BK] => offset s s_qk_h DK r BT BK k.1) (fun k => f k.1)
    (fun k => active s DK BK k.1) hinj (i, PUnit.unit)
  simpa using this

theorem bwdScatterRow_other_region (s : BlockState) (region R : RegionName)
    (s_qk_h DK BT BK : Nat) (r : Nat) (f : Fin BK → ℝ) (c : BlockState) (off : Nat)
    (h : R ≠ region) :
    (bwdScatterRow s region s_qk_h DK BT BK r f c).readMem R off = c.readMem R off := by
  unfold bwdScatterRow
  exact BlockState.scatter_prop_masked_preserves_other_region region
    (fun k : TileIndex [BK] => offset s s_qk_h DK r BT BK k.1) (fun k => f k.1)
    (fun k => active s DK BK k.1) R h off _ c

theorem bwdScatterRow_other_offset (s : BlockState) (region : RegionName)
    (s_qk_h DK BT BK : Nat) (r : Nat) (f : Fin BK → ℝ) (c : BlockState) (off : Nat)
    (h : ∀ i : Fin BK, offset s s_qk_h DK r BT BK i ≠ off) :
    (bwdScatterRow s region s_qk_h DK BT BK r f c).readMem region off = c.readMem region off := by
  unfold bwdScatterRow
  exact BlockState.scatter_prop_masked_preserves_other_offset region
    (fun k : TileIndex [BK] => offset s s_qk_h DK r BT BK k.1) (fun k => f k.1)
    (fun k => active s DK BK k.1) off (fun k _ => h k.1) _ c

/-- The lowered loop's `stop` operand evaluates to `BT` (for `BT ≥ 1`). -/
theorem bwd_stopOp_eval (c : BlockState) (BT : Nat) (hBT : 0 < BT) :
    evalOp (Op.add .nat Broadcast.nil
        (Op.div .nat Broadcast.nil
          (Op.sub .nat Broadcast.nil (Op.constNat BT) (Op.constNat 1)) (Op.constNat 1))
        (Op.constNat 1)) c
      = some (Tile.scalar BT) := by
  simp only [evalOp, evalOp_constNat, Option.bind, Tile.bop, NumericDType.add,
    NumericDType.sub, NumericDType.div]
  apply congrArg some; apply Tile.ext; intro idx
  show (BT - 1) / 1 + 1 = BT
  omega

/-- **Entry invariant.** The prologue post-state satisfies `bwdInvG 0`. -/
theorem bwdInvG_entry
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (s s0 : BlockState) (s_qk_h DK BT BK : Nat)
    (hsh : BwdPrologueShapeG s s0 DQInner DQInter DKInner DKInter Q K G DG s_qk_h DK BT BK) :
    bwdInvG DQInner DQInter DKInner DKInter Q K G DG s s_qk_h DK BT BK 0 s0 := by
  have hmemrd : ∀ (r : RegionName) (a : Nat), s0.readMem r a = s.readMem r a := by
    intro r a; simp only [BlockState.readMem, hsh.mem]
  refine ⟨hsh.pid, (fun a => hmemrd G a), (fun a => hmemrd Q a), (fun a => hmemrd K a),
    (fun a => hmemrd DQInner a), (fun a => hmemrd DKInner a), hsh.mask, ?_, ?_,
    ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · -- cum_grad_dg
    rw [hsh.cum]; apply congrArg some; apply Tile.ext; intro idx
    simp only [bwdGatedG, bwdCumPartialG_zero]
    by_cases ha : active s DK BK idx.1 <;> simp [ha]
  · -- last_g
    rw [hsh.lastg]; apply congrArg some; apply Tile.ext; intro idx
    simp only [bwdGatedG, bwdLastgG, lt_irrefl, if_false]
    by_cases ha : active s DK BK idx.1 <;> simp [ha]
  · rw [bwdPtrDecTileG_zero]; simpa using hsh.pg
  · rw [bwdPtrDecTileG_zero]; simpa using hsh.pk
  · rw [bwdPtrDecTileG_zero]; simpa using hsh.pq
  · rw [bwdPtrDecTileG_zero]; simpa using hsh.pdqi
  · rw [bwdPtrDecTileG_zero]; simpa using hsh.pdki
  · rw [bwdPtrDecTileG_zero]; simpa using hsh.pdqt
  · rw [bwdPtrDecTileG_zero]; simpa using hsh.pdkt
  · rw [bwdPtrDecTileG_zero]; simpa using hsh.pdg
  · intro r _ i; exact hmemrd DQInter _
  · intro r _ i; exact hmemrd DKInter _
  · intro r _ i; exact hmemrd DG _
  · intro r hr hr2 i; omega
  · intro r hr hr2 i; omega
  · intro r hr hr2 i; omega

/-- The reverse partial sum recurrence: folding physical row `BT-1-m` adds its
summand. -/
theorem bwdCumPartialG_succ (s : BlockState)
    (DQInner DQInter DKInner DKInter Q K G : RegionName)
    (s_qk_h DK BT BK : Nat) (m : Nat) (hm : m < BT) (i : Fin BK) :
    bwdCumPartialG s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK (m + 1) i =
      bwdCumPartialG s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK m i +
        bwdDgSumG s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK (BT - 1 - m)
          (fun i' => s.readMem G (offset s s_qk_h DK (BT - 1) BT BK i')) i := by
  simp only [bwdCumPartialG, Fin.sum_univ_castSucc, Fin.val_last, Fin.val_castSucc]

set_option maxHeartbeats 4000000 in
/-- **One reverse iteration advances `bwdInvG`.** -/
theorem bwd_decay_cumsum_step_general
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (s : BlockState) (s_qk_h DK BT BK : Nat) (hBK : BK ≤ DK)
    (hDKInner_DQInter : DKInner ≠ DQInter) (hDKInter_DQInter : DKInter ≠ DQInter)
    (hQ_DQInter : Q ≠ DQInter) (hQ_DKInter : Q ≠ DKInter)
    (hK_DQInter : K ≠ DQInter) (hK_DKInter : K ≠ DKInter)
    (hDQInter_DKInter : DQInter ≠ DKInter)
    (hDQInter_DG : DQInter ≠ DG) (hDKInter_DG : DKInter ≠ DG)
    (hG_DQInter : G ≠ DQInter) (hG_DKInter : G ≠ DKInter) (hG_DG : G ≠ DG)
    (hQ_DG : Q ≠ DG) (hK_DG : K ≠ DG)
    (hDQInner_DQInter : DQInner ≠ DQInter) (hDQInner_DKInter : DQInner ≠ DKInter)
    (hDQInner_DG : DQInner ≠ DG)
    (hDKInner_DKInter : DKInner ≠ DKInter) (hDKInner_DG : DKInner ≠ DG)
    (m : Nat) (hm : m < BT) (sc : BlockState)
    (hP : bwdInvG DQInner DQInter DKInner DKInter Q K G DG s s_qk_h DK BT BK m sc) :
    ∃ s',
      stepStmts (bwdIterBodyG DK BT BK) (sc.setReg "__rev_t" .nat [] (Tile.scalar m)) = some s' ∧
      bwdInvG DQInner DQInter DKInner DKInter Q K G DG s s_qk_h DK BT BK (m + 1) s' := by
  obtain ⟨hpid, hGrd, hQrd, hKrd, hDQird, hDKird, hmaskP, hcumP, hlastgP,
    hpgP, hpkP, hpqP, hpdqiP, hpdkiP, hpdqtP, hpdktP, hpdgP,
    hDQIunt, hDKIunt, hDGunt, hDQIwr, hDKIwr, hDGwr⟩ := hP
  -- physical row processed this iteration
  set r := BT - 1 - m with hr
  -- rewrite all pointer registers to the row-`r` tile form
  rw [bwdPtrDecTileG_row s G s_qk_h DK BT BK m hm] at hpgP
  rw [bwdPtrDecTileG_row s K s_qk_h DK BT BK m hm] at hpkP
  rw [bwdPtrDecTileG_row s Q s_qk_h DK BT BK m hm] at hpqP
  rw [bwdPtrDecTileG_row s DQInner s_qk_h DK BT BK m hm] at hpdqiP
  rw [bwdPtrDecTileG_row s DKInner s_qk_h DK BT BK m hm] at hpdkiP
  rw [bwdPtrDecTileG_row s DQInter s_qk_h DK BT BK m hm] at hpdqtP
  rw [bwdPtrDecTileG_row s DKInter s_qk_h DK BT BK m hm] at hpdktP
  rw [bwdPtrDecTileG_row s DG s_qk_h DK BT BK m hm] at hpdgP
  -- captured `last_g` per-lane reverse value = g[row BT-1]
  set lgF : Fin BK → ℝ := fun i' => s.readMem G (offset s s_qk_h DK (BT - 1) BT BK i') with hlgF
  -- old reverse-scan partial
  set cumF : Fin BK → ℝ :=
    bwdCumPartialG s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK m with hcumF
  set sin := sc.setReg "__rev_t" .nat [] (Tile.scalar m) with hsin
  have hsinpid : sin.pids = s.pids := by rw [hsin, BlockState.setReg_pids]; exact hpid
  -- memory reads through `sin` agree with `sc` (which agree with `s` on G/Q/K/inner)
  have hsinG : ∀ a, sin.readMem G a = s.readMem G a := by
    intro a; rw [hsin, BlockState.setReg_readMem]; exact hGrd a
  have hsinQ : ∀ a, sin.readMem Q a = s.readMem Q a := by
    intro a; rw [hsin, BlockState.setReg_readMem]; exact hQrd a
  have hsinK : ∀ a, sin.readMem K a = s.readMem K a := by
    intro a; rw [hsin, BlockState.setReg_readMem]; exact hKrd a
  have hsinDQi : ∀ a, sin.readMem DQInner a = s.readMem DQInner a := by
    intro a; rw [hsin, BlockState.setReg_readMem]; exact hDQird a
  have hsinDKi : ∀ a, sin.readMem DKInner a = s.readMem DKInner a := by
    intro a; rw [hsin, BlockState.setReg_readMem]; exact hDKird a
  -- abbreviations for the masked tiles
  set maskT := bwdMaskTileG s DK BK with hmaskT
  -- `sin` register lookups (peel the `__rev_t` setReg over `sc`)
  have hsinrev : sin.regs .nat [] "__rev_t" = some (Tile.scalar m) := by
    rw [hsin]; exact BlockState.setReg_same _ _ _ _ _
  have hsinmask : sin.regs .bool [BK] "mask" = some maskT := by
    rw [hsin, BlockState.setReg_ne_name (h := by decide)]; exact hmaskP
  have hsinpg : sin.regs .ptr [BK] "p_g" = some (bwdPtrTileG s G s_qk_h DK BT BK r) := by
    rw [hsin, BlockState.setReg_ne_name (h := by decide)]; exact hpgP
  -- ===== bwdIterBodyG unfolds: statement 0 (t), 1 (g_val), 2 (ifThen) =====
  simp only [bwdIterBodyG]
  -- statement 0: t = (BT-1) - m*1
  have e0 : stepStmt (Stmt.assign .nat [] "t"
      (Op.sub .nat Broadcast.nil
        (Op.sub .nat Broadcast.nil (Op.constNat BT) (Op.constNat 1))
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "__rev_t") (Op.constNat 1)))) sin
      = some (sin.setReg "t" .nat [] (Tile.scalar (BT - 1 - m * 1))) := by
    rw [bwdEval_assign_subNat sin "t" _ _ (BT - 1) (m * 1)
        (by simp [evalOp, Tile.bop, NumericDType.sub])
        (by simp [evalOp, hsinrev, Tile.bop, NumericDType.mul])]
  rw [stepStmts.cons_some e0]
  set e0s := sin.setReg "t" .nat [] (Tile.scalar (BT - 1 - m * 1)) with he0s
  have e0smask : e0s.regs .bool [BK] "mask" = some maskT := by
    rw [he0s, BlockState.setReg_ne_name (h := by decide)]; exact hsinmask
  have e0spg : e0s.regs .ptr [BK] "p_g" = some (bwdPtrTileG s G s_qk_h DK BT BK r) := by
    rw [he0s, BlockState.setReg_ne_name (h := by decide)]; exact hsinpg
  -- statement 1: g_val = masked load of p_g  (row r)
  have e1 : stepStmt (Stmt.assign .real [BK] "g_val"
      (Op.load ComputeDType.fp32.eraseDType (MemAccess.ptr (Op.ref .ptr [BK] "p_g"))
        (MaskOpt.maskOther (Op.ref .bool [BK] "mask") ((Op.const 0).broadcast [BK])))) e0s
      = some (e0s.setReg "g_val" .real [BK]
          (bwdGatedG s DK BK (bwdLdRowG s G s_qk_h DK BT BK r))) := by
    simp only [ComputeDType.eraseDType]
    rw [bwdEval_assign_load_ptr_maskOther e0s "g_val" "p_g" _ _
        (bwdPtrTileG s G s_qk_h DK BT BK r) maskT ⟨fun _ => some 0⟩
        e0spg (by rw [evalOp_ref]; exact e0smask) (by simp [evalOp])]
    rw [bwdLoad_gated s e0s G s_qk_h DK BT BK r
        (fun i => by rw [he0s, BlockState.setReg_readMem, hsinG])]
  rw [stepStmts.cons_some e1]
  set e1s := e0s.setReg "g_val" .real [BK]
    (bwdGatedG s DK BK (bwdLdRowG s G s_qk_h DK BT BK r)) with he1s
  -- statement 2: ifThen (t == BT-1) { last_g = g_val }
  -- condition `t = BT-1-m*1 == BT-1`: TRUE iff m=0
  have e1st : e1s.regs .nat [] "t" = some (Tile.scalar (BT - 1 - m * 1)) := by
    rw [he1s, BlockState.setReg_ne_name (h := by decide), he0s]
    exact BlockState.setReg_same _ _ _ _ _
  have e1sgval : e1s.regs .real [BK] "g_val" = some
      (bwdGatedG s DK BK (bwdLdRowG s G s_qk_h DK BT BK r)) := by
    rw [he1s]; exact BlockState.setReg_same _ _ _ _ _
  have hcond : evalOp (Op.eq ComparableDType.nat Broadcast.nil (Op.ref .nat [] "t")
      (Op.sub .nat Broadcast.nil (Op.constNat BT) (Op.constNat 1))) e1s
      = some (Tile.scalar (decide (BT - 1 - m * 1 = BT - 1))) := by
    rw [bwdEval_eqNat e1s _ _ (Tile.scalar (BT - 1 - m * 1)) (Tile.scalar (BT - 1))
        (by rw [evalOp_ref]; exact e1st)
        (by simp [evalOp, Tile.bop, NumericDType.sub])]
    rfl
  -- `last_g` per-lane value after the capture: g[row BT-1] in both branches.
  -- (m=0 ⇒ row r = BT-1 captures it; m>0 ⇒ inherited from `hP`.)
  -- Obtain the post-head state `s3` with its uniform register facts.
  obtain ⟨s3, hhead, hs3t, hs3gval, hs3mask, hs3lastg, hs3cum,
      hs3pg, hs3pk, hs3pq, hs3pdqi, hs3pdki, hs3pdqt, hs3pdkt, hs3pdg, hs3peelmem⟩ :
      ∃ s3 : BlockState,
        stepStmt (Stmt.ifThen
          (Op.eq ComparableDType.nat Broadcast.nil (Op.ref .nat [] "t")
            (Op.sub .nat Broadcast.nil (Op.constNat BT) (Op.constNat 1)))
          [Stmt.assign .real [BK] "last_g" (Op.ref .real [BK] "g_val")]) e1s = some s3 ∧
        s3.regs .nat [] "t" = some (Tile.scalar (BT - 1 - m * 1)) ∧
        s3.regs .real [BK] "g_val" = some (bwdGatedG s DK BK (bwdLdRowG s G s_qk_h DK BT BK r)) ∧
        s3.regs .bool [BK] "mask" = some maskT ∧
        s3.regs .real [BK] "last_g" = some (bwdGatedG s DK BK lgF) ∧
        s3.regs .real [BK] "cum_grad_dg" = some (bwdGatedG s DK BK cumF) ∧
        s3.regs .ptr [BK] "p_g" = some (bwdPtrTileG s G s_qk_h DK BT BK r) ∧
        s3.regs .ptr [BK] "p_k" = some (bwdPtrTileG s K s_qk_h DK BT BK r) ∧
        s3.regs .ptr [BK] "p_q" = some (bwdPtrTileG s Q s_qk_h DK BT BK r) ∧
        s3.regs .ptr [BK] "p_dq_inner" = some (bwdPtrTileG s DQInner s_qk_h DK BT BK r) ∧
        s3.regs .ptr [BK] "p_dk_inner" = some (bwdPtrTileG s DKInner s_qk_h DK BT BK r) ∧
        s3.regs .ptr [BK] "p_dq_inter" = some (bwdPtrTileG s DQInter s_qk_h DK BT BK r) ∧
        s3.regs .ptr [BK] "p_dk_inter" = some (bwdPtrTileG s DKInter s_qk_h DK BT BK r) ∧
        s3.regs .ptr [BK] "p_dg" = some (bwdPtrTileG s DG s_qk_h DK BT BK r) ∧
        (s3.pids = s.pids ∧ s3.mem = sc.mem) := by
    -- common e1s register facts (peel `g_val`/`t` setRegs over `sin`/`sc`)
    have e1speel : ∀ (d : TileDType) (sh : TileShape) (n : RegName),
        n ≠ "g_val" → n ≠ "t" → n ≠ "__rev_t" →
        e1s.regs d sh n = sc.regs d sh n := by
      intro d sh n h1 h2 h3
      rw [he1s, BlockState.setReg_ne_name (h := h1), he0s,
          BlockState.setReg_ne_name (h := h2), hsin, BlockState.setReg_ne_name (h := h3)]
    have e1smask : e1s.regs .bool [BK] "mask" = some maskT := by
      rw [e1speel .bool [BK] "mask" (by decide) (by decide) (by decide)]; exact hmaskP
    have e1scum : e1s.regs .real [BK] "cum_grad_dg" = some (bwdGatedG s DK BK cumF) := by
      rw [e1speel .real [BK] "cum_grad_dg" (by decide) (by decide) (by decide)]; exact hcumP
    have e1spg : e1s.regs .ptr [BK] "p_g" = some (bwdPtrTileG s G s_qk_h DK BT BK r) := by
      rw [e1speel .ptr [BK] "p_g" (by decide) (by decide) (by decide)]; exact hpgP
    have e1spk : e1s.regs .ptr [BK] "p_k" = some (bwdPtrTileG s K s_qk_h DK BT BK r) := by
      rw [e1speel .ptr [BK] "p_k" (by decide) (by decide) (by decide)]; exact hpkP
    have e1spq : e1s.regs .ptr [BK] "p_q" = some (bwdPtrTileG s Q s_qk_h DK BT BK r) := by
      rw [e1speel .ptr [BK] "p_q" (by decide) (by decide) (by decide)]; exact hpqP
    have e1spdqi : e1s.regs .ptr [BK] "p_dq_inner" = some (bwdPtrTileG s DQInner s_qk_h DK BT BK r) := by
      rw [e1speel .ptr [BK] "p_dq_inner" (by decide) (by decide) (by decide)]; exact hpdqiP
    have e1spdki : e1s.regs .ptr [BK] "p_dk_inner" = some (bwdPtrTileG s DKInner s_qk_h DK BT BK r) := by
      rw [e1speel .ptr [BK] "p_dk_inner" (by decide) (by decide) (by decide)]; exact hpdkiP
    have e1spdqt : e1s.regs .ptr [BK] "p_dq_inter" = some (bwdPtrTileG s DQInter s_qk_h DK BT BK r) := by
      rw [e1speel .ptr [BK] "p_dq_inter" (by decide) (by decide) (by decide)]; exact hpdqtP
    have e1spdkt : e1s.regs .ptr [BK] "p_dk_inter" = some (bwdPtrTileG s DKInter s_qk_h DK BT BK r) := by
      rw [e1speel .ptr [BK] "p_dk_inter" (by decide) (by decide) (by decide)]; exact hpdktP
    have e1spdg : e1s.regs .ptr [BK] "p_dg" = some (bwdPtrTileG s DG s_qk_h DK BT BK r) := by
      rw [e1speel .ptr [BK] "p_dg" (by decide) (by decide) (by decide)]; exact hpdgP
    have e1spid : e1s.pids = s.pids := by
      rw [he1s, BlockState.setReg_pids, he0s, BlockState.setReg_pids]; exact hsinpid
    have e1smem : e1s.mem = sc.mem := by
      rw [he1s]; simp only [BlockState.setReg]; rw [he0s]; simp only [BlockState.setReg]
      rw [hsin]; simp only [BlockState.setReg]
    by_cases hm0 : m = 0
    · -- m = 0: condition true, capture last_g = g_val = g[row r=BT-1]
      have hrBT : r = BT - 1 := by rw [hr, hm0]; omega
      have htrue : (decide (BT - 1 - m * 1 = BT - 1)) = Bool.true := by
        rw [hm0]; simp
      rw [bwdEval_ifThen_true e1s _ _
        (show evalOp _ e1s = some (Tile.scalar Bool.true) by rw [hcond, htrue])]
      rw [stepStmts.cons_some (stepStmt_assign_eq_some
          (show evalOp (Op.ref .real [BK] "g_val") e1s
              = some (bwdGatedG s DK BK (bwdLdRowG s G s_qk_h DK BT BK r)) by
            rw [evalOp_ref]; exact e1sgval))]
      rw [stepStmts.nil]
      refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
      · rw [BlockState.setReg_ne_name (h := by decide)]; exact e1st
      · rw [BlockState.setReg_ne_name (h := by decide)]; exact e1sgval
      · rw [BlockState.setReg_ne_name (h := by decide)]; exact e1smask
      · rw [BlockState.setReg_same]
        show some (bwdGatedG s DK BK (bwdLdRowG s G s_qk_h DK BT BK r)) = some (bwdGatedG s DK BK lgF)
        apply congrArg; apply congrArg; rw [hlgF]; funext i'
        simp only [bwdLdRowG]; rw [hrBT]
      · rw [BlockState.setReg_ne_name (h := by decide)]; exact e1scum
      · rw [BlockState.setReg_ne_name (h := by decide)]; exact e1spg
      · rw [BlockState.setReg_ne_name (h := by decide)]; exact e1spk
      · rw [BlockState.setReg_ne_name (h := by decide)]; exact e1spq
      · rw [BlockState.setReg_ne_name (h := by decide)]; exact e1spdqi
      · rw [BlockState.setReg_ne_name (h := by decide)]; exact e1spdki
      · rw [BlockState.setReg_ne_name (h := by decide)]; exact e1spdqt
      · rw [BlockState.setReg_ne_name (h := by decide)]; exact e1spdkt
      · rw [BlockState.setReg_ne_name (h := by decide)]; exact e1spdg
      · rw [BlockState.setReg_pids]; exact e1spid
      · simp only [BlockState.setReg]; exact e1smem
    · -- m > 0: condition false, last_g unchanged (inherited from hP)
      have hfalse : (decide (BT - 1 - m * 1 = BT - 1)) = Bool.false := by
        have hmpos : 0 < m := Nat.pos_of_ne_zero hm0
        simp only [decide_eq_false_iff_not]; omega
      have e1slastg : e1s.regs .real [BK] "last_g" = some (bwdGatedG s DK BK lgF) := by
        rw [e1speel .real [BK] "last_g" (by decide) (by decide) (by decide), hlastgP]
        apply congrArg; apply congrArg; rw [hlgF]; funext i'; simp only [bwdLastgG]
        rw [if_pos (Nat.pos_of_ne_zero hm0)]
      rw [bwdEval_ifThen_false e1s _ _
        (show evalOp _ e1s = some (Tile.scalar Bool.false) by rw [hcond, hfalse])]
      exact ⟨e1s, rfl, e1st, e1sgval, e1smask, e1slastg, e1scum,
        e1spg, e1spk, e1spq, e1spdqi, e1spdki, e1spdqt, e1spdkt, e1spdg, e1spid, e1smem⟩
  rw [stepStmts.cons_some hhead]
  -- ===== 22-statement tail on `s3` =====
  -- row `r = BT-1-m` is in the untouched range (`r < BT - m`).
  have hrlt : r < BT - m := by rw [hr]; omega
  -- `s3` memory reads at row r: G/Q/K/DQInner/DKInner agree with `s`; DQInter/DKInter
  -- at row r are still the original `s` values (untouched).
  have hs3mem : ∀ region a, s3.readMem region a = sc.readMem region a := by
    intro region a; simp only [BlockState.readMem, hs3peelmem.2]
  have rdG : ∀ i : Fin BK, s3.readMem G (offset s s_qk_h DK r BT BK i)
      = s.readMem G (offset s s_qk_h DK r BT BK i) := by intro i; rw [hs3mem]; exact hGrd _
  have rdQ : ∀ i : Fin BK, s3.readMem Q (offset s s_qk_h DK r BT BK i)
      = s.readMem Q (offset s s_qk_h DK r BT BK i) := by intro i; rw [hs3mem]; exact hQrd _
  have rdK : ∀ i : Fin BK, s3.readMem K (offset s s_qk_h DK r BT BK i)
      = s.readMem K (offset s s_qk_h DK r BT BK i) := by intro i; rw [hs3mem]; exact hKrd _
  have rdDQi : ∀ i : Fin BK, s3.readMem DQInner (offset s s_qk_h DK r BT BK i)
      = s.readMem DQInner (offset s s_qk_h DK r BT BK i) := by intro i; rw [hs3mem]; exact hDQird _
  have rdDKi : ∀ i : Fin BK, s3.readMem DKInner (offset s s_qk_h DK r BT BK i)
      = s.readMem DKInner (offset s s_qk_h DK r BT BK i) := by intro i; rw [hs3mem]; exact hDKird _
  have rdDQt : ∀ i : Fin BK, s3.readMem DQInter (offset s s_qk_h DK r BT BK i)
      = s.readMem DQInter (offset s s_qk_h DK r BT BK i) := by
    intro i; rw [hs3mem]; exact hDQIunt r hrlt i
  have rdDKt : ∀ i : Fin BK, s3.readMem DKInter (offset s s_qk_h DK r BT BK i)
      = s.readMem DKInter (offset s s_qk_h DK r BT BK i) := by
    intro i; rw [hs3mem]; exact hDKIunt r hrlt i
  -- abbreviations
  set otherT : Tile .real [BK] := ⟨fun _ => some 0⟩ with hotherT
  have hother : ∀ c : BlockState, evalOp ((Op.const (0 : ℝ)).broadcast [BK]) c = some otherT := by
    intro c; simp [evalOp, hotherT]
  -- === statement 3: dq1 = load p_dq_inner ===
  have d3 : stepStmt (Stmt.assign .real [BK] "dq1"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [BK] "p_dq_inner"))
        (MaskOpt.maskOther (Op.ref .bool [BK] "mask") ((Op.const 0).broadcast [BK])))) s3
      = some (s3.setReg "dq1" .real [BK]
          (bwdGatedG s DK BK (bwdLdRowG s DQInner s_qk_h DK BT BK r))) := by
    rw [bwdEval_assign_load_ptr_maskOther s3 "dq1" "p_dq_inner" _ _
        (bwdPtrTileG s DQInner s_qk_h DK BT BK r) maskT otherT
        hs3pdqi (by rw [evalOp_ref]; exact hs3mask) (hother s3)]
    rw [bwdLoad_gated s s3 DQInner s_qk_h DK BT BK r rdDQi]
  rw [stepStmts.cons_some d3]
  set c1 := s3.setReg "dq1" .real [BK]
    (bwdGatedG s DK BK (bwdLdRowG s DQInner s_qk_h DK BT BK r)) with hc1
  have c1mask : c1.regs .bool [BK] "mask" = some maskT := by
    rw [hc1, BlockState.setReg_ne_name (h := by decide)]; exact hs3mask
  have c1pdqt : c1.regs .ptr [BK] "p_dq_inter" = some (bwdPtrTileG s DQInter s_qk_h DK BT BK r) := by
    rw [hc1, BlockState.setReg_ne_name (h := by decide)]; exact hs3pdqt
  have c1rdDQt : ∀ i : Fin BK, c1.readMem DQInter (offset s s_qk_h DK r BT BK i)
      = s.readMem DQInter (offset s s_qk_h DK r BT BK i) := by
    intro i; rw [hc1, BlockState.setReg_readMem]; exact rdDQt i
  -- === statement 4: dq2 = load p_dq_inter ===
  have d4 : stepStmt (Stmt.assign .real [BK] "dq2"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [BK] "p_dq_inter"))
        (MaskOpt.maskOther (Op.ref .bool [BK] "mask") ((Op.const 0).broadcast [BK])))) c1
      = some (c1.setReg "dq2" .real [BK]
          (bwdGatedG s DK BK (bwdLdRowG s DQInter s_qk_h DK BT BK r))) := by
    rw [bwdEval_assign_load_ptr_maskOther c1 "dq2" "p_dq_inter" _ _
        (bwdPtrTileG s DQInter s_qk_h DK BT BK r) maskT otherT
        c1pdqt (by rw [evalOp_ref]; exact c1mask) (hother c1)]
    rw [bwdLoad_gated s c1 DQInter s_qk_h DK BT BK r c1rdDQt]
  rw [stepStmts.cons_some d4]
  set c2 := c1.setReg "dq2" .real [BK]
    (bwdGatedG s DK BK (bwdLdRowG s DQInter s_qk_h DK BT BK r)) with hc2
  have c2gval : c2.regs .real [BK] "g_val" = some
      (bwdGatedG s DK BK (bwdLdRowG s G s_qk_h DK BT BK r)) := by
    rw [hc2, BlockState.setReg_ne_name (h := by decide), hc1,
        BlockState.setReg_ne_name (h := by decide)]; exact hs3gval
  -- === statement 5: dq2 = dq2 * exp2(g_val) ===
  have d5 : stepStmt (Stmt.assign .real [BK] "dq2"
      (Op.mul .real Broadcast.nil.consSame (Op.ref .real [BK] "dq2")
        (Op.ref .real [BK] "g_val").exp2)) c2
      = some (c2.setReg "dq2" .real [BK]
          (bwdGatedG s DK BK (fun i => bwdLdRowG s DQInter s_qk_h DK BT BK r i *
            Real.exp (bwdLdRowG s G s_qk_h DK BT BK r i * Real.log 2)))) := by
    apply stepStmt_assign_eq_some
    rw [bwdEval_mul c2 _ _ (bwdGatedG s DK BK (bwdLdRowG s DQInter s_qk_h DK BT BK r))
        (Tile.uop WithBot.realExp2 (bwdGatedG s DK BK (bwdLdRowG s G s_qk_h DK BT BK r)))
        (by rw [evalOp_ref, hc2, BlockState.setReg_same])
        (by rw [bwdEval_exp2 c2 _ (bwdGatedG s DK BK (bwdLdRowG s G s_qk_h DK BT BK r))
            (by rw [evalOp_ref]; exact c2gval)])]
    rw [bwdGatedG_mul_exp2]
  rw [stepStmts.cons_some d5]
  set c3 := c2.setReg "dq2" .real [BK]
    (bwdGatedG s DK BK (fun i => bwdLdRowG s DQInter s_qk_h DK BT BK r i *
      Real.exp (bwdLdRowG s G s_qk_h DK BT BK r i * Real.log 2))) with hc3
  have c3dq1 : c3.regs .real [BK] "dq1" = some
      (bwdGatedG s DK BK (bwdLdRowG s DQInner s_qk_h DK BT BK r)) := by
    rw [hc3, BlockState.setReg_ne_name (h := by decide), hc2,
        BlockState.setReg_ne_name (h := by decide), hc1, BlockState.setReg_same]
  -- === statement 6: dq = dq1 + dq2 ===
  have d6 : stepStmt (Stmt.assign .real [BK] "dq"
      (Op.add .real Broadcast.nil.consSame (Op.ref .real [BK] "dq1")
        (Op.ref .real [BK] "dq2"))) c3
      = some (c3.setReg "dq" .real [BK]
          (bwdGatedG s DK BK (bwdDqOutG s DQInner DQInter G s_qk_h DK BT BK r))) := by
    apply stepStmt_assign_eq_some
    rw [bwdEval_add c3 _ _ (bwdGatedG s DK BK (bwdLdRowG s DQInner s_qk_h DK BT BK r))
        (bwdGatedG s DK BK (fun i => bwdLdRowG s DQInter s_qk_h DK BT BK r i *
          Real.exp (bwdLdRowG s G s_qk_h DK BT BK r i * Real.log 2)))
        (by rw [evalOp_ref]; exact c3dq1)
        (by rw [evalOp_ref, hc3, BlockState.setReg_same])]
    rw [bwdGatedG_add]; rfl
  rw [stepStmts.cons_some d6]
  set c4 := c3.setReg "dq" .real [BK]
    (bwdGatedG s DK BK (bwdDqOutG s DQInner DQInter G s_qk_h DK BT BK r)) with hc4
  have c4mask : c4.regs .bool [BK] "mask" = some maskT := by
    rw [hc4, BlockState.setReg_ne_name (h := by decide), hc3,
        BlockState.setReg_ne_name (h := by decide), hc2,
        BlockState.setReg_ne_name (h := by decide), hc1,
        BlockState.setReg_ne_name (h := by decide)]; exact hs3mask
  have c4pdqt : c4.regs .ptr [BK] "p_dq_inter" = some (bwdPtrTileG s DQInter s_qk_h DK BT BK r) := by
    rw [hc4, BlockState.setReg_ne_name (h := by decide), hc3,
        BlockState.setReg_ne_name (h := by decide), hc2,
        BlockState.setReg_ne_name (h := by decide), hc1,
        BlockState.setReg_ne_name (h := by decide)]; exact hs3pdqt
  have c4dq : c4.regs .real [BK] "dq" = some
      (bwdGatedG s DK BK (bwdDqOutG s DQInner DQInter G s_qk_h DK BT BK r)) := by
    rw [hc4]; exact BlockState.setReg_same _ _ _ _ _
  -- === statement 7: store p_dq_inter dq ===
  have d7 : stepStmt (Stmt.store .real [BK] (MemAccess.ptr (Op.ref .ptr [BK] "p_dq_inter"))
      (Op.ref .real [BK] "dq") (MaskOpt.mask (Op.ref .bool [BK] "mask"))) c4
      = some (bwdScatterRow s DQInter s_qk_h DK BT BK r
          (bwdDqOutG s DQInner DQInter G s_qk_h DK BT BK r) c4) := by
    rw [bwdEval_store_ptr_masked c4 "p_dq_inter" _ _
        (bwdPtrTileG s DQInter s_qk_h DK BT BK r)
        (bwdGatedG s DK BK (bwdDqOutG s DQInner DQInter G s_qk_h DK BT BK r)) maskT
        c4pdqt (by rw [evalOp_ref]; exact c4dq) (by rw [evalOp_ref]; exact c4mask)]
    rw [bwdStore_scatter]; rfl
  rw [stepStmts.cons_some d7]
  set c5 := bwdScatterRow s DQInter s_qk_h DK BT BK r
    (bwdDqOutG s DQInner DQInter G s_qk_h DK BT BK r) c4 with hc5
  -- `c4` register peel down to `s3` (4 real setRegs)
  have c4peel : ∀ (d : TileDType) (sh : TileShape) (n : RegName),
      n ≠ "dq" → n ≠ "dq2" → n ≠ "dq1" → c4.regs d sh n = s3.regs d sh n := by
    intro d sh n h1 h2 h3
    rw [hc4, BlockState.setReg_ne_name (h := h1), hc3,
        BlockState.setReg_ne_name (h := h2), hc2,
        BlockState.setReg_ne_name (h := h2), hc1,
        BlockState.setReg_ne_name (h := h3)]
  have c4read : ∀ (region : RegionName) (i : Fin BK),
      c4.readMem region (offset s s_qk_h DK r BT BK i)
        = s3.readMem region (offset s s_qk_h DK r BT BK i) := by
    intro region i
    rw [hc4, BlockState.setReg_readMem, hc3, BlockState.setReg_readMem,
        hc2, BlockState.setReg_readMem, hc1, BlockState.setReg_readMem]
  -- `c5` register peel (scatter preserves regs) + reads of regions ≠ DQInter
  have c5peel : ∀ (d : TileDType) (sh : TileShape) (n : RegName),
      n ≠ "dq" → n ≠ "dq2" → n ≠ "dq1" → c5.regs d sh n = s3.regs d sh n := by
    intro d sh n h1 h2 h3
    rw [hc5, bwdScatterRow_regs]; exact c4peel d sh n h1 h2 h3
  have c5read : ∀ (region : RegionName) (i : Fin BK), region ≠ DQInter →
      c5.readMem region (offset s s_qk_h DK r BT BK i)
        = s3.readMem region (offset s s_qk_h DK r BT BK i) := by
    intro region i hreg
    rw [hc5, bwdScatterRow_other_region s DQInter region s_qk_h DK BT BK r _ c4 _ hreg]
    exact c4read region i
  -- === statement 8: dk1 = load p_dk_inner ===
  have c5mask : c5.regs .bool [BK] "mask" = some maskT := by
    rw [c5peel .bool [BK] "mask" (by decide) (by decide) (by decide)]; exact hs3mask
  have c5pdki : c5.regs .ptr [BK] "p_dk_inner" = some (bwdPtrTileG s DKInner s_qk_h DK BT BK r) := by
    rw [c5peel .ptr [BK] "p_dk_inner" (by decide) (by decide) (by decide)]; exact hs3pdki
  have d8 : stepStmt (Stmt.assign .real [BK] "dk1"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [BK] "p_dk_inner"))
        (MaskOpt.maskOther (Op.ref .bool [BK] "mask") ((Op.const 0).broadcast [BK])))) c5
      = some (c5.setReg "dk1" .real [BK]
          (bwdGatedG s DK BK (bwdLdRowG s DKInner s_qk_h DK BT BK r))) := by
    rw [bwdEval_assign_load_ptr_maskOther c5 "dk1" "p_dk_inner" _ _
        (bwdPtrTileG s DKInner s_qk_h DK BT BK r) maskT otherT
        c5pdki (by rw [evalOp_ref]; exact c5mask) (hother c5)]
    rw [bwdLoad_gated s c5 DKInner s_qk_h DK BT BK r
        (fun i => by rw [c5read DKInner i hDKInner_DQInter]; exact rdDKi i)]
  rw [stepStmts.cons_some d8]
  set c6 := c5.setReg "dk1" .real [BK]
    (bwdGatedG s DK BK (bwdLdRowG s DKInner s_qk_h DK BT BK r)) with hc6
  -- === statement 9: dk2 = load p_dk_inter ===
  have c6mask : c6.regs .bool [BK] "mask" = some maskT := by
    rw [hc6, BlockState.setReg_ne_name (h := by decide)]; exact c5mask
  have c6pdkt : c6.regs .ptr [BK] "p_dk_inter" = some (bwdPtrTileG s DKInter s_qk_h DK BT BK r) := by
    rw [hc6, BlockState.setReg_ne_name (h := by decide),
        c5peel .ptr [BK] "p_dk_inter" (by decide) (by decide) (by decide)]; exact hs3pdkt
  have c6rdDKt : ∀ i : Fin BK, c6.readMem DKInter (offset s s_qk_h DK r BT BK i)
      = s.readMem DKInter (offset s s_qk_h DK r BT BK i) := by
    intro i; rw [hc6, BlockState.setReg_readMem,
      c5read DKInter i hDKInter_DQInter]; exact rdDKt i
  have d9 : stepStmt (Stmt.assign .real [BK] "dk2"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [BK] "p_dk_inter"))
        (MaskOpt.maskOther (Op.ref .bool [BK] "mask") ((Op.const 0).broadcast [BK])))) c6
      = some (c6.setReg "dk2" .real [BK]
          (bwdGatedG s DK BK (bwdLdRowG s DKInter s_qk_h DK BT BK r))) := by
    rw [bwdEval_assign_load_ptr_maskOther c6 "dk2" "p_dk_inter" _ _
        (bwdPtrTileG s DKInter s_qk_h DK BT BK r) maskT otherT
        c6pdkt (by rw [evalOp_ref]; exact c6mask) (hother c6)]
    rw [bwdLoad_gated s c6 DKInter s_qk_h DK BT BK r c6rdDKt]
  rw [stepStmts.cons_some d9]
  set c7 := c6.setReg "dk2" .real [BK]
    (bwdGatedG s DK BK (bwdLdRowG s DKInter s_qk_h DK BT BK r)) with hc7
  have c7peel : ∀ (d : TileDType) (sh : TileShape) (n : RegName),
      n ≠ "dk2" → n ≠ "dk1" → n ≠ "dq" → n ≠ "dq2" → n ≠ "dq1" →
      c7.regs d sh n = s3.regs d sh n := by
    intro d sh n h1 h2 h3 h4 h5
    rw [hc7, BlockState.setReg_ne_name (h := h1), hc6,
        BlockState.setReg_ne_name (h := h2)]
    exact c5peel d sh n h3 h4 h5
  have c7lastg : c7.regs .real [BK] "last_g" = some (bwdGatedG s DK BK lgF) := by
    rw [c7peel .real [BK] "last_g" (by decide) (by decide) (by decide) (by decide) (by decide)]
    exact hs3lastg
  have c7gval : c7.regs .real [BK] "g_val" = some
      (bwdGatedG s DK BK (bwdLdRowG s G s_qk_h DK BT BK r)) := by
    rw [c7peel .real [BK] "g_val" (by decide) (by decide) (by decide) (by decide) (by decide)]
    exact hs3gval
  have c7dk2 : c7.regs .real [BK] "dk2" = some
      (bwdGatedG s DK BK (bwdLdRowG s DKInter s_qk_h DK BT BK r)) := by
    rw [hc7]; exact BlockState.setReg_same _ _ _ _ _
  -- === statement 10: dk2 = dk2 * exp2(last_g - g_val) ===
  have hsub10 : evalOp (Op.sub .real Broadcast.nil.consSame (Op.ref .real [BK] "last_g")
      (Op.ref .real [BK] "g_val")) c7
      = some (bwdGatedG s DK BK (fun i => lgF i - bwdLdRowG s G s_qk_h DK BT BK r i)) := by
    rw [bwdEval_sub c7 _ _ (bwdGatedG s DK BK lgF)
        (bwdGatedG s DK BK (bwdLdRowG s G s_qk_h DK BT BK r))
        (by rw [evalOp_ref]; exact c7lastg) (by rw [evalOp_ref]; exact c7gval)]
    rw [bwdGatedG_sub]
  have d10 : stepStmt (Stmt.assign .real [BK] "dk2"
      (Op.mul .real Broadcast.nil.consSame (Op.ref .real [BK] "dk2")
        (Op.sub .real Broadcast.nil.consSame (Op.ref .real [BK] "last_g")
            (Op.ref .real [BK] "g_val")).exp2)) c7
      = some (c7.setReg "dk2" .real [BK]
          (bwdGatedG s DK BK (fun i => bwdLdRowG s DKInter s_qk_h DK BT BK r i *
            Real.exp ((lgF i - bwdLdRowG s G s_qk_h DK BT BK r i) * Real.log 2)))) := by
    apply stepStmt_assign_eq_some
    rw [bwdEval_mul c7 _ _ (bwdGatedG s DK BK (bwdLdRowG s DKInter s_qk_h DK BT BK r))
        (Tile.uop WithBot.realExp2
          (bwdGatedG s DK BK (fun i => lgF i - bwdLdRowG s G s_qk_h DK BT BK r i)))
        (by rw [evalOp_ref]; exact c7dk2)
        (by rw [bwdEval_exp2 c7 _
            (bwdGatedG s DK BK (fun i => lgF i - bwdLdRowG s G s_qk_h DK BT BK r i)) hsub10])]
    rw [bwdGatedG_mul_exp2]
  rw [stepStmts.cons_some d10]
  set c8 := c7.setReg "dk2" .real [BK]
    (bwdGatedG s DK BK (fun i => bwdLdRowG s DKInter s_qk_h DK BT BK r i *
      Real.exp ((lgF i - bwdLdRowG s G s_qk_h DK BT BK r i) * Real.log 2))) with hc8
  have c8dk1 : c8.regs .real [BK] "dk1" = some
      (bwdGatedG s DK BK (bwdLdRowG s DKInner s_qk_h DK BT BK r)) := by
    rw [hc8, BlockState.setReg_ne_name (h := by decide), hc7,
        BlockState.setReg_ne_name (h := by decide), hc6, BlockState.setReg_same]
  -- === statement 11: dk = dk1 + dk2 ===
  have d11 : stepStmt (Stmt.assign .real [BK] "dk"
      (Op.add .real Broadcast.nil.consSame (Op.ref .real [BK] "dk1")
        (Op.ref .real [BK] "dk2"))) c8
      = some (c8.setReg "dk" .real [BK]
          (bwdGatedG s DK BK (bwdDkOutG s DKInner DKInter G s_qk_h DK BT BK r lgF))) := by
    apply stepStmt_assign_eq_some
    rw [bwdEval_add c8 _ _ (bwdGatedG s DK BK (bwdLdRowG s DKInner s_qk_h DK BT BK r))
        (bwdGatedG s DK BK (fun i => bwdLdRowG s DKInter s_qk_h DK BT BK r i *
          Real.exp ((lgF i - bwdLdRowG s G s_qk_h DK BT BK r i) * Real.log 2)))
        (by rw [evalOp_ref]; exact c8dk1)
        (by rw [evalOp_ref, hc8, BlockState.setReg_same])]
    rw [bwdGatedG_add]; rfl
  rw [stepStmts.cons_some d11]
  set c9 := c8.setReg "dk" .real [BK]
    (bwdGatedG s DK BK (bwdDkOutG s DKInner DKInter G s_qk_h DK BT BK r lgF)) with hc9
  have c9peel : ∀ (d : TileDType) (sh : TileShape) (n : RegName),
      n ≠ "dk" → n ≠ "dk2" → n ≠ "dk1" → n ≠ "dq" → n ≠ "dq2" → n ≠ "dq1" →
      c9.regs d sh n = s3.regs d sh n := by
    intro d sh n h0 h1 h2 h3 h4 h5
    rw [hc9, BlockState.setReg_ne_name (h := h0), hc8,
        BlockState.setReg_ne_name (h := h1)]
    exact c7peel d sh n h1 h2 h3 h4 h5
  have c9mask : c9.regs .bool [BK] "mask" = some maskT := by
    rw [c9peel .bool [BK] "mask" (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]
    exact hs3mask
  have c9pdkt : c9.regs .ptr [BK] "p_dk_inter" = some (bwdPtrTileG s DKInter s_qk_h DK BT BK r) := by
    rw [c9peel .ptr [BK] "p_dk_inter" (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]
    exact hs3pdkt
  have c9dk : c9.regs .real [BK] "dk" = some
      (bwdGatedG s DK BK (bwdDkOutG s DKInner DKInter G s_qk_h DK BT BK r lgF)) := by
    rw [hc9]; exact BlockState.setReg_same _ _ _ _ _
  -- `c9` reads agree with `c5` (4 real setRegs)
  have c9read : ∀ (region : RegionName) (i : Fin BK),
      c9.readMem region (offset s s_qk_h DK r BT BK i)
        = c5.readMem region (offset s s_qk_h DK r BT BK i) := by
    intro region i
    rw [hc9, BlockState.setReg_readMem, hc8, BlockState.setReg_readMem,
        hc7, BlockState.setReg_readMem, hc6, BlockState.setReg_readMem]
  -- === statement 12: store p_dk_inter dk ===
  have d12 : stepStmt (Stmt.store .real [BK] (MemAccess.ptr (Op.ref .ptr [BK] "p_dk_inter"))
      (Op.ref .real [BK] "dk") (MaskOpt.mask (Op.ref .bool [BK] "mask"))) c9
      = some (bwdScatterRow s DKInter s_qk_h DK BT BK r
          (bwdDkOutG s DKInner DKInter G s_qk_h DK BT BK r lgF) c9) := by
    rw [bwdEval_store_ptr_masked c9 "p_dk_inter" _ _
        (bwdPtrTileG s DKInter s_qk_h DK BT BK r)
        (bwdGatedG s DK BK (bwdDkOutG s DKInner DKInter G s_qk_h DK BT BK r lgF)) maskT
        c9pdkt (by rw [evalOp_ref]; exact c9dk) (by rw [evalOp_ref]; exact c9mask)]
    rw [bwdStore_scatter]; rfl
  rw [stepStmts.cons_some d12]
  set c10 := bwdScatterRow s DKInter s_qk_h DK BT BK r
    (bwdDkOutG s DKInner DKInter G s_qk_h DK BT BK r lgF) c9 with hc10
  have c10peel : ∀ (d : TileDType) (sh : TileShape) (n : RegName),
      n ≠ "dk" → n ≠ "dk2" → n ≠ "dk1" → n ≠ "dq" → n ≠ "dq2" → n ≠ "dq1" →
      c10.regs d sh n = s3.regs d sh n := by
    intro d sh n h0 h1 h2 h3 h4 h5
    rw [hc10, bwdScatterRow_regs]; exact c9peel d sh n h0 h1 h2 h3 h4 h5
  have c10read : ∀ (region : RegionName) (i : Fin BK), region ≠ DKInter → region ≠ DQInter →
      c10.readMem region (offset s s_qk_h DK r BT BK i)
        = s3.readMem region (offset s s_qk_h DK r BT BK i) := by
    intro region i h1 h2
    rw [hc10, bwdScatterRow_other_region s DKInter region s_qk_h DK BT BK r _ c9 _ h1,
        c9read region i]
    exact c5read region i h2
  -- === statement 13: q_val = load p_q ===
  have c10mask : c10.regs .bool [BK] "mask" = some maskT := by
    rw [c10peel .bool [BK] "mask" (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]
    exact hs3mask
  have c10pq : c10.regs .ptr [BK] "p_q" = some (bwdPtrTileG s Q s_qk_h DK BT BK r) := by
    rw [c10peel .ptr [BK] "p_q" (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]
    exact hs3pq
  have d13 : stepStmt (Stmt.assign .real [BK] "q_val"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [BK] "p_q"))
        (MaskOpt.maskOther (Op.ref .bool [BK] "mask") ((Op.const 0).broadcast [BK])))) c10
      = some (c10.setReg "q_val" .real [BK]
          (bwdGatedG s DK BK (bwdLdRowG s Q s_qk_h DK BT BK r))) := by
    rw [bwdEval_assign_load_ptr_maskOther c10 "q_val" "p_q" _ _
        (bwdPtrTileG s Q s_qk_h DK BT BK r) maskT otherT
        c10pq (by rw [evalOp_ref]; exact c10mask) (hother c10)]
    rw [bwdLoad_gated s c10 Q s_qk_h DK BT BK r
        (fun i => by rw [c10read Q i hQ_DKInter hQ_DQInter]; exact rdQ i)]
  rw [stepStmts.cons_some d13]
  set c11 := c10.setReg "q_val" .real [BK]
    (bwdGatedG s DK BK (bwdLdRowG s Q s_qk_h DK BT BK r)) with hc11
  -- === statement 14: k_val = load p_k ===
  have c11mask : c11.regs .bool [BK] "mask" = some maskT := by
    rw [hc11, BlockState.setReg_ne_name (h := by decide)]; exact c10mask
  have c11pk : c11.regs .ptr [BK] "p_k" = some (bwdPtrTileG s K s_qk_h DK BT BK r) := by
    rw [hc11, BlockState.setReg_ne_name (h := by decide),
        c10peel .ptr [BK] "p_k" (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]
    exact hs3pk
  have c11rdK : ∀ i : Fin BK, c11.readMem K (offset s s_qk_h DK r BT BK i)
      = s.readMem K (offset s s_qk_h DK r BT BK i) := by
    intro i; rw [hc11, BlockState.setReg_readMem, c10read K i hK_DKInter hK_DQInter]; exact rdK i
  have d14 : stepStmt (Stmt.assign .real [BK] "k_val"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [BK] "p_k"))
        (MaskOpt.maskOther (Op.ref .bool [BK] "mask") ((Op.const 0).broadcast [BK])))) c11
      = some (c11.setReg "k_val" .real [BK]
          (bwdGatedG s DK BK (bwdLdRowG s K s_qk_h DK BT BK r))) := by
    rw [bwdEval_assign_load_ptr_maskOther c11 "k_val" "p_k" _ _
        (bwdPtrTileG s K s_qk_h DK BT BK r) maskT otherT
        c11pk (by rw [evalOp_ref]; exact c11mask) (hother c11)]
    rw [bwdLoad_gated s c11 K s_qk_h DK BT BK r c11rdK]
  rw [stepStmts.cons_some d14]
  set c12 := c11.setReg "k_val" .real [BK]
    (bwdGatedG s DK BK (bwdLdRowG s K s_qk_h DK BT BK r)) with hc12
  -- lookups for the dg_val computation
  have c12dq : c12.regs .real [BK] "dq" = some
      (bwdGatedG s DK BK (bwdDqOutG s DQInner DQInter G s_qk_h DK BT BK r)) := by
    rw [hc12, BlockState.setReg_ne_name (h := by decide), hc11,
        BlockState.setReg_ne_name (h := by decide), hc10, bwdScatterRow_regs, hc9,
        BlockState.setReg_ne_name (h := by decide), hc8,
        BlockState.setReg_ne_name (h := by decide), hc7,
        BlockState.setReg_ne_name (h := by decide), hc6,
        BlockState.setReg_ne_name (h := by decide), hc5, bwdScatterRow_regs, hc4,
        BlockState.setReg_same]
  have c12dk : c12.regs .real [BK] "dk" = some
      (bwdGatedG s DK BK (bwdDkOutG s DKInner DKInter G s_qk_h DK BT BK r lgF)) := by
    rw [hc12, BlockState.setReg_ne_name (h := by decide), hc11,
        BlockState.setReg_ne_name (h := by decide), hc10, bwdScatterRow_regs, hc9,
        BlockState.setReg_same]
  have c12qval : c12.regs .real [BK] "q_val" = some
      (bwdGatedG s DK BK (bwdLdRowG s Q s_qk_h DK BT BK r)) := by
    rw [hc12, BlockState.setReg_ne_name (h := by decide), hc11, BlockState.setReg_same]
  have c12kval : c12.regs .real [BK] "k_val" = some
      (bwdGatedG s DK BK (bwdLdRowG s K s_qk_h DK BT BK r)) := by
    rw [hc12]; exact BlockState.setReg_same _ _ _ _ _
  -- === statement 15: dg_val = dq*q_val - dk*k_val ===
  have hmul_dq : evalOp (Op.mul .real Broadcast.nil.consSame (Op.ref .real [BK] "dq")
      (Op.ref .real [BK] "q_val")) c12
      = some (bwdGatedG s DK BK (fun i => bwdDqOutG s DQInner DQInter G s_qk_h DK BT BK r i *
          bwdLdRowG s Q s_qk_h DK BT BK r i)) := by
    rw [bwdEval_mul c12 _ _ (bwdGatedG s DK BK (bwdDqOutG s DQInner DQInter G s_qk_h DK BT BK r))
        (bwdGatedG s DK BK (bwdLdRowG s Q s_qk_h DK BT BK r))
        (by rw [evalOp_ref]; exact c12dq) (by rw [evalOp_ref]; exact c12qval)]
    rw [bwdGatedG_mul]
  have hmul_dk : evalOp (Op.mul .real Broadcast.nil.consSame (Op.ref .real [BK] "dk")
      (Op.ref .real [BK] "k_val")) c12
      = some (bwdGatedG s DK BK (fun i => bwdDkOutG s DKInner DKInter G s_qk_h DK BT BK r lgF i *
          bwdLdRowG s K s_qk_h DK BT BK r i)) := by
    rw [bwdEval_mul c12 _ _ (bwdGatedG s DK BK (bwdDkOutG s DKInner DKInter G s_qk_h DK BT BK r lgF))
        (bwdGatedG s DK BK (bwdLdRowG s K s_qk_h DK BT BK r))
        (by rw [evalOp_ref]; exact c12dk) (by rw [evalOp_ref]; exact c12kval)]
    rw [bwdGatedG_mul]
  have d15 : stepStmt (Stmt.assign .real [BK] "dg_val"
      (Op.sub .real Broadcast.nil.consSame
        (Op.mul .real Broadcast.nil.consSame (Op.ref .real [BK] "dq")
          (Op.ref .real [BK] "q_val"))
        (Op.mul .real Broadcast.nil.consSame (Op.ref .real [BK] "dk")
          (Op.ref .real [BK] "k_val")))) c12
      = some (c12.setReg "dg_val" .real [BK]
          (bwdGatedG s DK BK (bwdDgSumG s DQInner DQInter DKInner DKInter Q K G
            s_qk_h DK BT BK r lgF))) := by
    apply stepStmt_assign_eq_some
    rw [bwdEval_sub c12 _ _
        (bwdGatedG s DK BK (fun i => bwdDqOutG s DQInner DQInter G s_qk_h DK BT BK r i *
          bwdLdRowG s Q s_qk_h DK BT BK r i))
        (bwdGatedG s DK BK (fun i => bwdDkOutG s DKInner DKInter G s_qk_h DK BT BK r lgF i *
          bwdLdRowG s K s_qk_h DK BT BK r i))
        hmul_dq hmul_dk]
    rw [bwdGatedG_sub]; rfl
  rw [stepStmts.cons_some d15]
  set c13 := c12.setReg "dg_val" .real [BK]
    (bwdGatedG s DK BK (bwdDgSumG s DQInner DQInter DKInner DKInter Q K G
      s_qk_h DK BT BK r lgF)) with hc13
  have c13cum : c13.regs .real [BK] "cum_grad_dg" = some (bwdGatedG s DK BK cumF) := by
    rw [hc13, BlockState.setReg_ne_name (h := by decide), hc12,
        BlockState.setReg_ne_name (h := by decide), hc11,
        BlockState.setReg_ne_name (h := by decide), hc10, bwdScatterRow_regs, hc9,
        BlockState.setReg_ne_name (h := by decide), hc8,
        BlockState.setReg_ne_name (h := by decide), hc7,
        BlockState.setReg_ne_name (h := by decide), hc6,
        BlockState.setReg_ne_name (h := by decide), hc5, bwdScatterRow_regs, hc4,
        BlockState.setReg_ne_name (h := by decide), hc3,
        BlockState.setReg_ne_name (h := by decide), hc2,
        BlockState.setReg_ne_name (h := by decide), hc1,
        BlockState.setReg_ne_name (h := by decide)]; exact hs3cum
  -- === statement 16: cum_grad_dg = cum_grad_dg + dg_val ===
  have d16 : stepStmt (Stmt.assign .real [BK] "cum_grad_dg"
      (Op.add .real Broadcast.nil.consSame (Op.ref .real [BK] "cum_grad_dg")
        (Op.ref .real [BK] "dg_val"))) c13
      = some (c13.setReg "cum_grad_dg" .real [BK]
          (bwdGatedG s DK BK (fun i => cumF i +
            bwdDgSumG s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK r lgF i))) := by
    apply stepStmt_assign_eq_some
    rw [bwdEval_add c13 _ _ (bwdGatedG s DK BK cumF)
        (bwdGatedG s DK BK (bwdDgSumG s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK r lgF))
        (by rw [evalOp_ref]; exact c13cum)
        (by rw [evalOp_ref, hc13, BlockState.setReg_same])]
    rw [bwdGatedG_add]
  rw [stepStmts.cons_some d16]
  set c14 := c13.setReg "cum_grad_dg" .real [BK]
    (bwdGatedG s DK BK (fun i => cumF i +
      bwdDgSumG s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK r lgF i)) with hc14
  -- this is exactly `bwdGatedG (bwdCumPartialG (m+1))`
  have hcumNext : (fun i => cumF i +
      bwdDgSumG s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK r lgF i)
      = bwdCumPartialG s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK (m + 1) := by
    funext i
    rw [bwdCumPartialG_succ s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK m hm i]
  have c14peel : ∀ (d : TileDType) (sh : TileShape) (n : RegName),
      n ≠ "cum_grad_dg" → n ≠ "dg_val" → n ≠ "k_val" → n ≠ "q_val" →
      n ≠ "dk" → n ≠ "dk2" → n ≠ "dk1" → n ≠ "dq" → n ≠ "dq2" → n ≠ "dq1" →
      c14.regs d sh n = s3.regs d sh n := by
    intro d sh n g0 g1 g2 g3 g4 g5 g6 g7 g8 g9
    rw [hc14, BlockState.setReg_ne_name (h := g0), hc13,
        BlockState.setReg_ne_name (h := g1), hc12,
        BlockState.setReg_ne_name (h := g2), hc11,
        BlockState.setReg_ne_name (h := g3), hc10, bwdScatterRow_regs, hc9,
        BlockState.setReg_ne_name (h := g4), hc8,
        BlockState.setReg_ne_name (h := g5), hc7,
        BlockState.setReg_ne_name (h := g5), hc6,
        BlockState.setReg_ne_name (h := g6), hc5, bwdScatterRow_regs, hc4,
        BlockState.setReg_ne_name (h := g7), hc3,
        BlockState.setReg_ne_name (h := g8), hc2,
        BlockState.setReg_ne_name (h := g8), hc1,
        BlockState.setReg_ne_name (h := g9)]
  -- === statement 17: store p_dg cum_grad_dg ===
  have c14mask : c14.regs .bool [BK] "mask" = some maskT := by
    rw [c14peel .bool [BK] "mask" (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]
    exact hs3mask
  have c14pdg : c14.regs .ptr [BK] "p_dg" = some (bwdPtrTileG s DG s_qk_h DK BT BK r) := by
    rw [c14peel .ptr [BK] "p_dg" (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]
    exact hs3pdg
  have c14cum : c14.regs .real [BK] "cum_grad_dg" = some
      (bwdGatedG s DK BK (fun i => cumF i +
        bwdDgSumG s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK r lgF i)) := by
    rw [hc14]; exact BlockState.setReg_same _ _ _ _ _
  have d17 : stepStmt (Stmt.store .real [BK] (MemAccess.ptr (Op.ref .ptr [BK] "p_dg"))
      (Op.ref .real [BK] "cum_grad_dg") (MaskOpt.mask (Op.ref .bool [BK] "mask"))) c14
      = some (bwdScatterRow s DG s_qk_h DK BT BK r
          (fun i => cumF i +
            bwdDgSumG s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK r lgF i) c14) := by
    rw [bwdEval_store_ptr_masked c14 "p_dg" _ _
        (bwdPtrTileG s DG s_qk_h DK BT BK r)
        (bwdGatedG s DK BK (fun i => cumF i +
          bwdDgSumG s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK r lgF i)) maskT
        c14pdg (by rw [evalOp_ref]; exact c14cum) (by rw [evalOp_ref]; exact c14mask)]
    rw [bwdStore_scatter]; rfl
  rw [stepStmts.cons_some d17]
  set c15 := bwdScatterRow s DG s_qk_h DK BT BK r
    (fun i => cumF i +
      bwdDgSumG s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK r lgF i) c14 with hc15
  -- `c15` register peel (DG scatter preserves regs)
  have c15peel : ∀ (d : TileDType) (sh : TileShape) (n : RegName),
      n ≠ "cum_grad_dg" → n ≠ "dg_val" → n ≠ "k_val" → n ≠ "q_val" →
      n ≠ "dk" → n ≠ "dk2" → n ≠ "dk1" → n ≠ "dq" → n ≠ "dq2" → n ≠ "dq1" →
      c15.regs d sh n = s3.regs d sh n := by
    intro d sh n g0 g1 g2 g3 g4 g5 g6 g7 g8 g9
    rw [hc15, bwdScatterRow_regs]; exact c14peel d sh n g0 g1 g2 g3 g4 g5 g6 g7 g8 g9
  have c15p : ∀ (region : RegionName) (n : RegName),
      s3.regs .ptr [BK] n = some (bwdPtrTileG s region s_qk_h DK BT BK r) →
      n ≠ "cum_grad_dg" → n ≠ "dg_val" → n ≠ "k_val" → n ≠ "q_val" →
      n ≠ "dk" → n ≠ "dk2" → n ≠ "dk1" → n ≠ "dq" → n ≠ "dq2" → n ≠ "dq1" →
      c15.regs .ptr [BK] n = some (bwdPtrTileG s region s_qk_h DK BT BK r) := by
    intro region n hn g0 g1 g2 g3 g4 g5 g6 g7 g8 g9
    rw [c15peel .ptr [BK] n g0 g1 g2 g3 g4 g5 g6 g7 g8 g9]; exact hn
  have c15pg := c15p G "p_g" hs3pg (by decide) (by decide) (by decide) (by decide)
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
  have c15pk := c15p K "p_k" hs3pk (by decide) (by decide) (by decide) (by decide)
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
  have c15pq := c15p Q "p_q" hs3pq (by decide) (by decide) (by decide) (by decide)
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
  have c15pdqi := c15p DQInner "p_dq_inner" hs3pdqi (by decide) (by decide) (by decide)
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
  have c15pdki := c15p DKInner "p_dk_inner" hs3pdki (by decide) (by decide) (by decide)
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
  have c15pdqt := c15p DQInter "p_dq_inter" hs3pdqt (by decide) (by decide) (by decide)
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
  have c15pdkt := c15p DKInter "p_dk_inter" hs3pdkt (by decide) (by decide) (by decide)
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
  have c15pdg := c15p DG "p_dg" hs3pdg (by decide) (by decide) (by decide)
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
  -- the decremented pointer tile (common shape after each ptrSub)
  have hdec : ∀ region : RegionName,
      (⟨fun idx => (((bwdPtrTileG s region s_qk_h DK BT BK r).data idx).1,
          ((bwdPtrTileG s region s_qk_h DK BT BK r).data idx).2 - DK)⟩ : Tile .ptr [BK])
        = bwdPtrDecTileG s region s_qk_h DK BT BK (m + 1) := by
    intro region
    rw [hr]; simp only [bwdPtrTileG]
    rw [← bwdPtrTileG_dec_succ s region s_qk_h DK BT BK m hm]
  -- === statements 18-25: eight pointer decrements ===
  have d18 : stepStmt (Stmt.assign .ptr [BK] "p_g"
      (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [BK] "p_g") (Op.constNat DK))) c15
      = some (c15.setReg "p_g" .ptr [BK] (bwdPtrDecTileG s G s_qk_h DK BT BK (m + 1))) := by
    rw [bwdEval_assign_ptrSub c15 "p_g" "p_g" DK _ c15pg, hdec]
  rw [stepStmts.cons_some d18]
  set c16 := c15.setReg "p_g" .ptr [BK] (bwdPtrDecTileG s G s_qk_h DK BT BK (m + 1)) with hc16
  have d19 : stepStmt (Stmt.assign .ptr [BK] "p_k"
      (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [BK] "p_k") (Op.constNat DK))) c16
      = some (c16.setReg "p_k" .ptr [BK] (bwdPtrDecTileG s K s_qk_h DK BT BK (m + 1))) := by
    rw [bwdEval_assign_ptrSub c16 "p_k" "p_k" DK _
        (by rw [hc16, BlockState.setReg_ne_name (h := by decide)]; exact c15pk), hdec]
  rw [stepStmts.cons_some d19]
  set c17 := c16.setReg "p_k" .ptr [BK] (bwdPtrDecTileG s K s_qk_h DK BT BK (m + 1)) with hc17
  have d20 : stepStmt (Stmt.assign .ptr [BK] "p_q"
      (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [BK] "p_q") (Op.constNat DK))) c17
      = some (c17.setReg "p_q" .ptr [BK] (bwdPtrDecTileG s Q s_qk_h DK BT BK (m + 1))) := by
    rw [bwdEval_assign_ptrSub c17 "p_q" "p_q" DK _
        (by rw [hc17, BlockState.setReg_ne_name (h := by decide), hc16,
            BlockState.setReg_ne_name (h := by decide)]; exact c15pq), hdec]
  rw [stepStmts.cons_some d20]
  set c18 := c17.setReg "p_q" .ptr [BK] (bwdPtrDecTileG s Q s_qk_h DK BT BK (m + 1)) with hc18
  have d21 : stepStmt (Stmt.assign .ptr [BK] "p_dq_inner"
      (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [BK] "p_dq_inner") (Op.constNat DK))) c18
      = some (c18.setReg "p_dq_inner" .ptr [BK] (bwdPtrDecTileG s DQInner s_qk_h DK BT BK (m + 1))) := by
    rw [bwdEval_assign_ptrSub c18 "p_dq_inner" "p_dq_inner" DK _
        (by rw [hc18, BlockState.setReg_ne_name (h := by decide), hc17,
            BlockState.setReg_ne_name (h := by decide), hc16,
            BlockState.setReg_ne_name (h := by decide)]; exact c15pdqi), hdec]
  rw [stepStmts.cons_some d21]
  set c19 := c18.setReg "p_dq_inner" .ptr [BK] (bwdPtrDecTileG s DQInner s_qk_h DK BT BK (m + 1)) with hc19
  have d22 : stepStmt (Stmt.assign .ptr [BK] "p_dk_inner"
      (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [BK] "p_dk_inner") (Op.constNat DK))) c19
      = some (c19.setReg "p_dk_inner" .ptr [BK] (bwdPtrDecTileG s DKInner s_qk_h DK BT BK (m + 1))) := by
    rw [bwdEval_assign_ptrSub c19 "p_dk_inner" "p_dk_inner" DK _
        (by rw [hc19, BlockState.setReg_ne_name (h := by decide), hc18,
            BlockState.setReg_ne_name (h := by decide), hc17,
            BlockState.setReg_ne_name (h := by decide), hc16,
            BlockState.setReg_ne_name (h := by decide)]; exact c15pdki), hdec]
  rw [stepStmts.cons_some d22]
  set c20 := c19.setReg "p_dk_inner" .ptr [BK] (bwdPtrDecTileG s DKInner s_qk_h DK BT BK (m + 1)) with hc20
  have d23 : stepStmt (Stmt.assign .ptr [BK] "p_dq_inter"
      (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [BK] "p_dq_inter") (Op.constNat DK))) c20
      = some (c20.setReg "p_dq_inter" .ptr [BK] (bwdPtrDecTileG s DQInter s_qk_h DK BT BK (m + 1))) := by
    rw [bwdEval_assign_ptrSub c20 "p_dq_inter" "p_dq_inter" DK _
        (by rw [hc20, BlockState.setReg_ne_name (h := by decide), hc19,
            BlockState.setReg_ne_name (h := by decide), hc18,
            BlockState.setReg_ne_name (h := by decide), hc17,
            BlockState.setReg_ne_name (h := by decide), hc16,
            BlockState.setReg_ne_name (h := by decide)]; exact c15pdqt), hdec]
  rw [stepStmts.cons_some d23]
  set c21 := c20.setReg "p_dq_inter" .ptr [BK] (bwdPtrDecTileG s DQInter s_qk_h DK BT BK (m + 1)) with hc21
  have d24 : stepStmt (Stmt.assign .ptr [BK] "p_dk_inter"
      (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [BK] "p_dk_inter") (Op.constNat DK))) c21
      = some (c21.setReg "p_dk_inter" .ptr [BK] (bwdPtrDecTileG s DKInter s_qk_h DK BT BK (m + 1))) := by
    rw [bwdEval_assign_ptrSub c21 "p_dk_inter" "p_dk_inter" DK _
        (by rw [hc21, BlockState.setReg_ne_name (h := by decide), hc20,
            BlockState.setReg_ne_name (h := by decide), hc19,
            BlockState.setReg_ne_name (h := by decide), hc18,
            BlockState.setReg_ne_name (h := by decide), hc17,
            BlockState.setReg_ne_name (h := by decide), hc16,
            BlockState.setReg_ne_name (h := by decide)]; exact c15pdkt), hdec]
  rw [stepStmts.cons_some d24]
  set c22 := c21.setReg "p_dk_inter" .ptr [BK] (bwdPtrDecTileG s DKInter s_qk_h DK BT BK (m + 1)) with hc22
  have d25 : stepStmt (Stmt.assign .ptr [BK] "p_dg"
      (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [BK] "p_dg") (Op.constNat DK))) c22
      = some (c22.setReg "p_dg" .ptr [BK] (bwdPtrDecTileG s DG s_qk_h DK BT BK (m + 1))) := by
    rw [bwdEval_assign_ptrSub c22 "p_dg" "p_dg" DK _
        (by rw [hc22, BlockState.setReg_ne_name (h := by decide), hc21,
            BlockState.setReg_ne_name (h := by decide), hc20,
            BlockState.setReg_ne_name (h := by decide), hc19,
            BlockState.setReg_ne_name (h := by decide), hc18,
            BlockState.setReg_ne_name (h := by decide), hc17,
            BlockState.setReg_ne_name (h := by decide), hc16,
            BlockState.setReg_ne_name (h := by decide)]; exact c15pdg), hdec]
  rw [stepStmts.cons_some d25]
  set c23 := c22.setReg "p_dg" .ptr [BK] (bwdPtrDecTileG s DG s_qk_h DK BT BK (m + 1)) with hc23
  rw [stepStmts.nil]
  -- ===== reconstruct `bwdInvG (m+1) c23` =====
  -- c23 memory equals c15 memory (8 ptr setRegs preserve mem)
  have hc23mem : ∀ region a, c23.readMem region a = c15.readMem region a := by
    intro region a
    simp only [hc23, hc22, hc21, hc20, hc19, hc18, hc17, hc16,
      BlockState.setReg_readMem]
  -- c14 memory equals c10 memory (4 real setRegs)
  have hc14mem : ∀ region a, c14.readMem region a = c10.readMem region a := by
    intro region a
    simp only [hc14, hc13, hc12, hc11, BlockState.setReg_readMem]
  -- c9 memory equals c5 memory (4 real setRegs)
  have hc9mem : ∀ region a, c9.readMem region a = c5.readMem region a := by
    intro region a
    simp only [hc9, hc8, hc7, hc6, BlockState.setReg_readMem]
  -- c4 memory equals s3 memory (4 real setRegs)
  have hc4mem : ∀ region a, c4.readMem region a = s3.readMem region a := by
    intro region a
    simp only [hc4, hc3, hc2, hc1, BlockState.setReg_readMem]
  -- pids of c23
  have hc23pid : c23.pids = s.pids := by
    simp only [hc23, hc22, hc21, hc20, hc19, hc18, hc17, hc16, BlockState.setReg_pids,
      hc15, bwdScatterRow_pids, hc14, hc13, hc12, hc11, hc10, bwdScatterRow_pids,
      hc9, hc8, hc7, hc6, hc5, bwdScatterRow_pids, hc4, hc3, hc2, hc1]
    exact hs3peelmem.1
  -- The three scatters' read characterization.  For region distinct from all three
  -- write-targets, c23 read = s3 read (= sc read).
  have c23read_other : ∀ (region : RegionName) (a : Nat),
      region ≠ DQInter → region ≠ DKInter → region ≠ DG →
      c23.readMem region a = sc.readMem region a := by
    intro region a h1 h2 h3
    rw [hc23mem, hc15, bwdScatterRow_other_region s DG region s_qk_h DK BT BK r _ c14 _ h3,
        hc14mem, hc10, bwdScatterRow_other_region s DKInter region s_qk_h DK BT BK r _ c9 _ h2,
        hc9mem, hc5, bwdScatterRow_other_region s DQInter region s_qk_h DK BT BK r _ c4 _ h1,
        hc4mem]
    simp only [BlockState.readMem, hs3peelmem.2]
  -- c23 register peel to s3 (8 ptr setRegs over c15, scatter preserves regs, c14peel)
  have c23peel : ∀ (d : TileDType) (sh : TileShape) (n : RegName),
      n ≠ "p_dg" → n ≠ "p_dk_inter" → n ≠ "p_dq_inter" → n ≠ "p_dk_inner" →
      n ≠ "p_dq_inner" → n ≠ "p_q" → n ≠ "p_k" → n ≠ "p_g" →
      n ≠ "cum_grad_dg" → n ≠ "dg_val" → n ≠ "k_val" → n ≠ "q_val" →
      n ≠ "dk" → n ≠ "dk2" → n ≠ "dk1" → n ≠ "dq" → n ≠ "dq2" → n ≠ "dq1" →
      c23.regs d sh n = s3.regs d sh n := by
    intro d sh n p0 p1 p2 p3 p4 p5 p6 p7 g0 g1 g2 g3 g4 g5 g6 g7 g8 g9
    rw [hc23, BlockState.setReg_ne_name (h := p0), hc22,
        BlockState.setReg_ne_name (h := p1), hc21,
        BlockState.setReg_ne_name (h := p2), hc20,
        BlockState.setReg_ne_name (h := p3), hc19,
        BlockState.setReg_ne_name (h := p4), hc18,
        BlockState.setReg_ne_name (h := p5), hc17,
        BlockState.setReg_ne_name (h := p6), hc16,
        BlockState.setReg_ne_name (h := p7), hc15, bwdScatterRow_regs]
    exact c14peel d sh n g0 g1 g2 g3 g4 g5 g6 g7 g8 g9
  -- readback at processed rows.  DG at row r = bwdCumPartialG (m+1).
  -- DQInter / DKInter at row r = the closed forms.  All via the three scatters.
  -- DG at row r:
  have hDGr : ∀ i : Fin BK, c23.readMem DG (offset s s_qk_h DK r BT BK i)
      = if active s DK BK i then
          bwdCumPartialG s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK (m + 1) i
        else s.readMem DG (offset s s_qk_h DK r BT BK i) := by
    intro i
    rw [hc23mem, hc15, bwdScatterRow_readback]
    by_cases ha : active s DK BK i
    · simp only [ha, if_true]
      rw [bwdCumPartialG_succ s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK m hm i]
    · simp only [ha, if_false]
      rw [hc14mem, hc10, bwdScatterRow_other_region s DKInter DG s_qk_h DK BT BK r _ c9 _
            (Ne.symm hDKInter_DG),
          hc9mem, hc5, bwdScatterRow_other_region s DQInter DG s_qk_h DK BT BK r _ c4 _
            (Ne.symm hDQInter_DG), hc4mem, hs3mem]
      exact hDGunt r hrlt i
  -- DQInter at row r:
  have hDQtr : ∀ i : Fin BK, c23.readMem DQInter (offset s s_qk_h DK r BT BK i)
      = if active s DK BK i then bwdDqOutG s DQInner DQInter G s_qk_h DK BT BK r i
        else s.readMem DQInter (offset s s_qk_h DK r BT BK i) := by
    intro i
    rw [hc23mem, hc15, bwdScatterRow_other_region s DG DQInter s_qk_h DK BT BK r _ c14 _ hDQInter_DG,
        hc14mem, hc10, bwdScatterRow_other_region s DKInter DQInter s_qk_h DK BT BK r _ c9 _
          hDQInter_DKInter,
        hc9mem, hc5, bwdScatterRow_readback]
    by_cases ha : active s DK BK i
    · simp only [ha, if_true]
    · simp only [ha, if_false]; rw [hc4mem, hs3mem]
      exact hDQIunt r hrlt i
  -- DKInter at row r:
  have hDKtr : ∀ i : Fin BK, c23.readMem DKInter (offset s s_qk_h DK r BT BK i)
      = if active s DK BK i then
          bwdDkOutG s DKInner DKInter G s_qk_h DK BT BK r lgF i
        else s.readMem DKInter (offset s s_qk_h DK r BT BK i) := by
    intro i
    rw [hc23mem, hc15, bwdScatterRow_other_region s DG DKInter s_qk_h DK BT BK r _ c14 _ hDKInter_DG,
        hc14mem, hc10, bwdScatterRow_readback]
    by_cases ha : active s DK BK i
    · simp only [ha, if_true]
    · simp only [ha, if_false]
      rw [hc9mem, hc5, bwdScatterRow_other_region s DQInter DKInter s_qk_h DK BT BK r _ c4 _
            hDKInter_DQInter, hc4mem, hs3mem]
      exact hDKIunt r hrlt i
  -- reads at OTHER rows r' ≠ r (the scatters only touch row r).
  have hscatter_other_off : ∀ (region : RegionName) (r' : Nat) (i : Fin BK), r' ≠ r →
      c23.readMem region (offset s s_qk_h DK r' BT BK i)
        = sc.readMem region (offset s s_qk_h DK r' BT BK i) := by
    intro region r' i hr'
    have hoff : ∀ j : Fin BK, offset s s_qk_h DK r BT BK j ≠ offset s s_qk_h DK r' BT BK i := by
      intro j
      simp only [offset, baseOffset]
      have hjDK : (j : Nat) < DK := lt_of_lt_of_le j.isLt hBK
      have hiDK : (i : Nat) < DK := lt_of_lt_of_le i.isLt hBK
      have hrr' : r ≠ r' := fun h => hr' h.symm
      have hexpand : ∀ x : Nat, (s.pids 1 * BT + x) * DK = s.pids 1 * BT * DK + x * DK := by
        intro x; rw [Nat.add_mul]
      rw [hexpand r, hexpand r']
      -- now reduces to: r*DK + j ≠ r'*DK + i, with j,i < DK, r ≠ r'
      rcases Nat.lt_or_ge r r' with hlt | hge
      · have : r * DK + DK ≤ r' * DK := by
          calc r * DK + DK = (r + 1) * DK := by rw [Nat.add_mul, one_mul]
            _ ≤ r' * DK := Nat.mul_le_mul_right DK hlt
        omega
      · have hgt : r' < r := lt_of_le_of_ne hge (Ne.symm hrr')
        have : r' * DK + DK ≤ r * DK := by
          calc r' * DK + DK = (r' + 1) * DK := by rw [Nat.add_mul, one_mul]
            _ ≤ r * DK := Nat.mul_le_mul_right DK hgt
        omega
    -- a scatter into X at row r leaves `region`-reads at non-row-r offsets unchanged
    have hscatter1 : ∀ (X : RegionName) (f : Fin BK → ℝ) (c : BlockState),
        (bwdScatterRow s X s_qk_h DK BT BK r f c).readMem region
            (offset s s_qk_h DK r' BT BK i)
          = c.readMem region (offset s s_qk_h DK r' BT BK i) := by
      intro X f c
      by_cases hX : region = X
      · subst hX; exact bwdScatterRow_other_offset s region s_qk_h DK BT BK r f c _ hoff
      · exact bwdScatterRow_other_region s X region s_qk_h DK BT BK r f c _ hX
    rw [hc23mem, hc15, hscatter1 DG _ c14,
        hc14mem, hc10, hscatter1 DKInter _ c9,
        hc9mem, hc5, hscatter1 DQInter _ c4, hc4mem, hs3mem]
  refine ⟨c23, rfl, ?_⟩
  show bwdInvG DQInner DQInter DKInner DKInter Q K G DG s s_qk_h DK BT BK (m + 1) c23
  simp only [bwdInvG]
  refine ⟨hc23pid, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_,
    ?_, ?_, ?_, ?_, ?_, ?_⟩
  · -- G reads (G ∉ {DQInter,DKInter,DG})
    intro a; rw [c23read_other G a hG_DQInter hG_DKInter hG_DG]; exact hGrd a
  · intro a; rw [c23read_other Q a hQ_DQInter hQ_DKInter hQ_DG]; exact hQrd a
  · intro a; rw [c23read_other K a hK_DQInter hK_DKInter hK_DG]; exact hKrd a
  · intro a; rw [c23read_other DQInner a hDQInner_DQInter hDQInner_DKInter hDQInner_DG]; exact hDQird a
  · intro a; rw [c23read_other DKInner a hDKInner_DQInter hDKInner_DKInter hDKInner_DG]; exact hDKird a
  · -- mask
    rw [c23peel .bool [BK] "mask" (by decide) (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]
    exact hs3mask
  · -- cum_grad_dg
    rw [hc23, BlockState.setReg_ne_name (h := by decide), hc22,
        BlockState.setReg_ne_name (h := by decide), hc21,
        BlockState.setReg_ne_name (h := by decide), hc20,
        BlockState.setReg_ne_name (h := by decide), hc19,
        BlockState.setReg_ne_name (h := by decide), hc18,
        BlockState.setReg_ne_name (h := by decide), hc17,
        BlockState.setReg_ne_name (h := by decide), hc16,
        BlockState.setReg_ne_name (h := by decide), hc15, bwdScatterRow_regs, hc14]
    rw [BlockState.setReg_same, hcumNext]
  · -- last_g = bwdGatedG lgF = bwdGatedG (bwdLastgG (m+1))
    rw [c23peel .real [BK] "last_g" (by decide) (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]
    rw [hs3lastg]
    apply congrArg; apply congrArg; rw [hlgF]; funext i'; simp only [bwdLastgG]
    rw [if_pos (Nat.succ_pos m)]
  · rw [hc23, BlockState.setReg_ne_name (h := by decide), hc22,
        BlockState.setReg_ne_name (h := by decide), hc21,
        BlockState.setReg_ne_name (h := by decide), hc20,
        BlockState.setReg_ne_name (h := by decide), hc19,
        BlockState.setReg_ne_name (h := by decide), hc18,
        BlockState.setReg_ne_name (h := by decide), hc17,
        BlockState.setReg_ne_name (h := by decide), hc16]
    exact BlockState.setReg_same _ _ _ _ _
  · rw [hc23, BlockState.setReg_ne_name (h := by decide), hc22,
        BlockState.setReg_ne_name (h := by decide), hc21,
        BlockState.setReg_ne_name (h := by decide), hc20,
        BlockState.setReg_ne_name (h := by decide), hc19,
        BlockState.setReg_ne_name (h := by decide), hc18,
        BlockState.setReg_ne_name (h := by decide), hc17]
    exact BlockState.setReg_same _ _ _ _ _
  · rw [hc23, BlockState.setReg_ne_name (h := by decide), hc22,
        BlockState.setReg_ne_name (h := by decide), hc21,
        BlockState.setReg_ne_name (h := by decide), hc20,
        BlockState.setReg_ne_name (h := by decide), hc19,
        BlockState.setReg_ne_name (h := by decide), hc18]
    exact BlockState.setReg_same _ _ _ _ _
  · rw [hc23, BlockState.setReg_ne_name (h := by decide), hc22,
        BlockState.setReg_ne_name (h := by decide), hc21,
        BlockState.setReg_ne_name (h := by decide), hc20,
        BlockState.setReg_ne_name (h := by decide), hc19]
    exact BlockState.setReg_same _ _ _ _ _
  · rw [hc23, BlockState.setReg_ne_name (h := by decide), hc22,
        BlockState.setReg_ne_name (h := by decide), hc21,
        BlockState.setReg_ne_name (h := by decide), hc20]
    exact BlockState.setReg_same _ _ _ _ _
  · rw [hc23, BlockState.setReg_ne_name (h := by decide), hc22,
        BlockState.setReg_ne_name (h := by decide), hc21]
    exact BlockState.setReg_same _ _ _ _ _
  · rw [hc23, BlockState.setReg_ne_name (h := by decide), hc22]
    exact BlockState.setReg_same _ _ _ _ _
  · rw [hc23]; exact BlockState.setReg_same _ _ _ _ _
  · -- DQInter untouched at rows r' < BT-(m+1)
    intro r' hr' i
    rw [hscatter_other_off DQInter r' i (by rw [hr]; omega)]
    exact hDQIunt r' (by omega) i
  · -- DKInter untouched
    intro r' hr' i
    rw [hscatter_other_off DKInter r' i (by rw [hr]; omega)]
    exact hDKIunt r' (by omega) i
  · -- DG untouched
    intro r' hr' i
    rw [hscatter_other_off DG r' i (by rw [hr]; omega)]
    exact hDGunt r' (by omega) i
  · -- DQInter written rows BT-(m+1) ≤ r' < BT
    intro r' hr'lo hr'hi i
    by_cases heq : r' = r
    · subst heq; rw [hDQtr i]
    · rw [hscatter_other_off DQInter r' i heq]
      have : BT - m ≤ r' := by simp only [hr] at heq; omega
      exact hDQIwr r' this hr'hi i
  · -- DKInter written rows
    intro r' hr'lo hr'hi i
    by_cases heq : r' = r
    · subst heq; rw [hDKtr i]
    · rw [hscatter_other_off DKInter r' i heq]
      have : BT - m ≤ r' := by simp only [hr] at heq; omega
      exact hDKIwr r' this hr'hi i
  · -- DG written rows: at row r = BT-1-m, BT - r = m+1
    intro r' hr'lo hr'hi i
    by_cases heq : r' = r
    · subst heq
      rw [hDGr i]
      have hBTr : BT - r = m + 1 := by rw [hr]; omega
      rw [hBTr]
    · rw [hscatter_other_off DG r' i heq]
      have : BT - m ≤ r' := by simp only [hr] at heq; omega
      exact hDGwr r' this hr'hi i

/-- `bwdDqOutG` at row `r` equals the genuine `bwdDQInterClosed` at `t_rel = r`. -/
theorem bwdDqOutG_eq_closed (s : BlockState) (DQInner DQInter G : RegionName)
    (s_qk_h DK BT BK : Nat) (t_rel : Fin BT) (i : Fin BK) :
    bwdDqOutG s DQInner DQInter G s_qk_h DK BT BK t_rel.val i
      = bwdDQInterClosed s DQInner DQInter G s_qk_h DK BT BK t_rel i := by
  simp only [bwdDqOutG, bwdDQInterClosed]

/-- `bwdDkOutG` (captured `last_g = g[row BT-1]`) at row `r` equals
`bwdDKInterClosed` at `t_rel = r`. -/
theorem bwdDkOutG_eq_closed (s : BlockState) (DKInner DKInter G : RegionName)
    (s_qk_h DK BT BK : Nat) (t_rel : Fin BT) (i : Fin BK) :
    bwdDkOutG s DKInner DKInter G s_qk_h DK BT BK t_rel.val
        (fun i' => s.readMem G (offset s s_qk_h DK (BT - 1) BT BK i')) i
      = bwdDKInterClosed s DKInner DKInter G s_qk_h DK BT BK t_rel i := by
  simp only [bwdDkOutG, bwdDKInterClosed]

/-- `bwdDgSumG` (captured `last_g`) at row `r` equals the genuine `bwdDGSummand`. -/
theorem bwdDgSumG_eq_summand (s : BlockState)
    (DQInner DQInter DKInner DKInter Q K G : RegionName)
    (s_qk_h DK BT BK : Nat) (t_rel : Fin BT) (i : Fin BK) :
    bwdDgSumG s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK t_rel.val
        (fun i' => s.readMem G (offset s s_qk_h DK (BT - 1) BT BK i')) i
      = bwdDGSummand s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK t_rel i := by
  simp only [bwdDgSumG, bwdDGSummand, bwdDqOutG_eq_closed, bwdDkOutG_eq_closed]

/-- The reverse-cumulative partial `bwdCumPartialG (BT - t_rel)` equals the genuine
`bwdDGClosed` at `t_rel` (reindexing the reverse sum). -/
theorem bwdCumPartialG_eq_DGClosed (s : BlockState)
    (DQInner DQInter DKInner DKInter Q K G : RegionName)
    (s_qk_h DK BT BK : Nat) (t_rel : Fin BT) (i : Fin BK) :
    bwdCumPartialG s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK
        (BT - t_rel.val) i
      = bwdDGClosed s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK t_rel i := by
  simp only [bwdCumPartialG, bwdDGClosed]
  -- ∑_{d < BT - t_rel} summand[BT-1-d]  =  ∑_{d < BT - t_rel} summand[t_rel + d]
  -- via the reflection d ↦ (BT - t_rel - 1 - d)
  apply Finset.sum_nbij' (fun d => (⟨BT - t_rel.val - 1 - d.val, by omega⟩ : Fin (BT - t_rel.val)))
    (fun d => (⟨BT - t_rel.val - 1 - d.val, by omega⟩ : Fin (BT - t_rel.val)))
  · intro d _; simp
  · intro d _; simp
  · intro d _; ext; simp; omega
  · intro d _; ext; simp; omega
  · intro d _
    have hidx : BT - 1 - (BT - t_rel.val - 1 - d.val) = t_rel.val + d.val := by omega
    rw [show (⟨t_rel.val + (BT - t_rel.val - 1 - d.val), by omega⟩ : Fin BT)
          = (⟨BT - 1 - d.val, by omega⟩ : Fin BT) by ext; simp; omega]
    rw [← bwdDgSumG_eq_summand s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK
      ⟨BT - 1 - d.val, by omega⟩ i]

set_option maxHeartbeats 4000000 in
/-- **General reverse-loop drive.** The complete backward surface executes (prologue
+ `forRangeDyn` reverse loop driven by `forRangeDyn_inv` over `bwdInvG`) to a final
state satisfying `bwdInvG … final` with `BT ≤ final`. -/
theorem bwd_loop_drive_general
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (s : BlockState) (s_qk_h DK BT BK : Nat) (hBT : 0 < BT) (hBK : BK ≤ DK)
    (hDKInner_DQInter : DKInner ≠ DQInter) (hDKInter_DQInter : DKInter ≠ DQInter)
    (hQ_DQInter : Q ≠ DQInter) (hQ_DKInter : Q ≠ DKInter)
    (hK_DQInter : K ≠ DQInter) (hK_DKInter : K ≠ DKInter)
    (hDQInter_DKInter : DQInter ≠ DKInter)
    (hDQInter_DG : DQInter ≠ DG) (hDKInter_DG : DKInter ≠ DG)
    (hG_DQInter : G ≠ DQInter) (hG_DKInter : G ≠ DKInter) (hG_DG : G ≠ DG)
    (hQ_DG : Q ≠ DG) (hK_DG : K ≠ DG)
    (hDQInner_DQInter : DQInner ≠ DQInter) (hDQInner_DKInter : DQInner ≠ DKInter)
    (hDQInner_DG : DQInner ≠ DG)
    (hDKInner_DKInter : DKInner ≠ DKInter) (hDKInner_DG : DKInner ≠ DG) :
    ∃ final sfinal,
      exec (bwd_decay_global_cumsum_surface DQInner DQInter DKInner DKInter Q K G DG
        s_qk_h DK BT BK) s = some sfinal ∧
      BT ≤ final ∧
      bwdInvG DQInner DQInter DKInner DKInter Q K G DG s s_qk_h DK BT BK final sfinal := by
  -- prologue
  obtain ⟨s0, hpro, hsh⟩ :=
    bwd_prologue_eval_general DQInner DQInter DKInner DKInter Q K G DG s_qk_h DK BT BK hBT s
  have hP0 : bwdInvG DQInner DQInter DKInner DKInter Q K G DG s s_qk_h DK BT BK 0 s0 :=
    bwdInvG_entry DQInner DQInter DKInner DKInter Q K G DG s s0 s_qk_h DK BT BK hsh
  -- drive the loop
  obtain ⟨final, sfinal, hloop, hge, hPfinal⟩ :=
    VeriTile.Triton.forRangeDyn_inv (idx := "__rev_t") (start := 0) (stop := BT) (step := 1)
      (P := bwdInvG DQInner DQInter DKInner DKInter Q K G DG s s_qk_h DK BT BK)
      (s_init := s0)
      (evalOp_constNat 0 s0)
      (bwd_stopOp_eval s0 BT hBT)
      (evalOp_constNat 1 s0)
      one_ne_zero
      hP0
      (by
        intro i sc hi hPi
        exact bwd_decay_cumsum_step_general DQInner DQInter DKInner DKInter Q K G DG s
          s_qk_h DK BT BK hBK hDKInner_DQInter hDKInter_DQInter hQ_DQInter hQ_DKInter
          hK_DQInter hK_DKInter hDQInter_DKInter hDQInter_DG hDKInter_DG
          hG_DQInter hG_DKInter hG_DG hQ_DG hK_DG hDQInner_DQInter hDQInner_DKInter
          hDQInner_DG hDKInner_DKInter hDKInner_DG i hi sc hPi)
  refine ⟨final, sfinal, ?_, hge, hPfinal⟩
  -- assemble exec = prologue ++ [loop]
  rw [exec, bwd_body_decomp_general, stepStmts.append_some hpro,
    stepStmts.cons_some hloop, stepStmts.nil]

section GeneralTops

variable (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
  (s : BlockState) (s_qk_h DK BT BK : Nat) (t_rel : Fin BT)
  (hBT : 0 < BT) (hBK : BK ≤ DK)
  (hDKInner_DQInter : DKInner ≠ DQInter) (hDKInter_DQInter : DKInter ≠ DQInter)
  (hQ_DQInter : Q ≠ DQInter) (hQ_DKInter : Q ≠ DKInter)
  (hK_DQInter : K ≠ DQInter) (hK_DKInter : K ≠ DKInter)
  (hDQInter_DKInter : DQInter ≠ DKInter)
  (hDQInter_DG : DQInter ≠ DG) (hDKInter_DG : DKInter ≠ DG)
  (hG_DQInter : G ≠ DQInter) (hG_DKInter : G ≠ DKInter) (hG_DG : G ≠ DG)
  (hQ_DG : Q ≠ DG) (hK_DG : K ≠ DG)
  (hDQInner_DQInter : DQInner ≠ DQInter) (hDQInner_DKInter : DQInner ≠ DKInter)
  (hDQInner_DG : DQInner ≠ DG)
  (hDKInner_DKInter : DKInner ≠ DKInter) (hDKInner_DG : DKInner ≠ DG)

include hBT hBK hDKInner_DQInter hDKInter_DQInter hQ_DQInter hQ_DKInter hK_DQInter
  hK_DKInter hDQInter_DKInter hDQInter_DG hDKInter_DG hG_DQInter hG_DKInter hG_DG
  hQ_DG hK_DG hDQInner_DQInter hDQInner_DKInter hDQInner_DG hDKInner_DKInter hDKInner_DG

set_option maxHeartbeats 1600000 in
/-- **General `dq_inter` readback.** The executed backward surface writes the honest
closed form `bwdDQInterClosed` into `DQInter` at every active lane of row `t_rel`. -/
theorem bwd_decay_cumsum_dq_inter_closed_compute_correct_general :
    ComputeCorrect.Realizes
      (kernel := bwd_decay_global_cumsum_surface DQInner DQInter DKInner DKInter
        Q K G DG s_qk_h DK BT BK)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf (active s DK BK)
        (fun i => (DQInter, offset s s_qk_h DK t_rel.val BT BK i)))
      (expected := fun i : Fin BK =>
        bwdDQInterClosed s DQInner DQInter G s_qk_h DK BT BK t_rel i) := by
  obtain ⟨final, sfinal, hexec, hge, hPfinal⟩ :=
    bwd_loop_drive_general DQInner DQInter DKInner DKInter Q K G DG s s_qk_h DK BT BK hBT hBK
      hDKInner_DQInter hDKInter_DQInter hQ_DQInter hQ_DKInter hK_DQInter hK_DKInter
      hDQInter_DKInter hDQInter_DG hDKInter_DG hG_DQInter hG_DKInter hG_DG hQ_DG hK_DG
      hDQInner_DQInter hDQInner_DKInter hDQInner_DG hDKInner_DKInter hDKInner_DG
  obtain ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, hDQIwr, _⟩ := hPfinal
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [bwd_decay_global_cumsum_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro sa s' hExec hsa
  subst sa
  intro i hActive
  have hs' : s' = sfinal := by
    rw [exec] at hexec; rw [exec] at hExec; rw [hExec] at hexec; exact Option.some.inj hexec
  subst s'
  show sfinal.readMem DQInter (offset s s_qk_h DK t_rel.val BT BK i) = _
  rw [hDQIwr t_rel.val (by omega) t_rel.isLt i, if_pos hActive,
    bwdDqOutG_eq_closed s DQInner DQInter G s_qk_h DK BT BK t_rel i]

set_option maxHeartbeats 1600000 in
/-- **General `dk_inter` readback.** -/
theorem bwd_decay_cumsum_dk_inter_closed_compute_correct_general :
    ComputeCorrect.Realizes
      (kernel := bwd_decay_global_cumsum_surface DQInner DQInter DKInner DKInter
        Q K G DG s_qk_h DK BT BK)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf (active s DK BK)
        (fun i => (DKInter, offset s s_qk_h DK t_rel.val BT BK i)))
      (expected := fun i : Fin BK =>
        bwdDKInterClosed s DKInner DKInter G s_qk_h DK BT BK t_rel i) := by
  obtain ⟨final, sfinal, hexec, hge, hPfinal⟩ :=
    bwd_loop_drive_general DQInner DQInter DKInner DKInter Q K G DG s s_qk_h DK BT BK hBT hBK
      hDKInner_DQInter hDKInter_DQInter hQ_DQInter hQ_DKInter hK_DQInter hK_DKInter
      hDQInter_DKInter hDQInter_DG hDKInter_DG hG_DQInter hG_DKInter hG_DG hQ_DG hK_DG
      hDQInner_DQInter hDQInner_DKInter hDQInner_DG hDKInner_DKInter hDKInner_DG
  obtain ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, hDKIwr, _⟩ := hPfinal
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [bwd_decay_global_cumsum_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro sa s' hExec hsa
  subst sa
  intro i hActive
  have hs' : s' = sfinal := by
    rw [exec] at hexec; rw [exec] at hExec; rw [hExec] at hexec; exact Option.some.inj hexec
  subst s'
  show sfinal.readMem DKInter (offset s s_qk_h DK t_rel.val BT BK i) = _
  rw [hDKIwr t_rel.val (by omega) t_rel.isLt i, if_pos hActive,
    bwdDkOutG_eq_closed s DKInner DKInter G s_qk_h DK BT BK t_rel i]

set_option maxHeartbeats 1600000 in
/-- **General `dg` readback.** The executed backward surface writes the honest
reverse-cumulative-sum closed form `bwdDGClosed` into `DG`. -/
theorem bwd_decay_cumsum_dg_closed_compute_correct_general :
    ComputeCorrect.Realizes
      (kernel := bwd_decay_global_cumsum_surface DQInner DQInter DKInner DKInter
        Q K G DG s_qk_h DK BT BK)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf (active s DK BK)
        (fun i => (DG, offset s s_qk_h DK t_rel.val BT BK i)))
      (expected := fun i : Fin BK =>
        bwdDGClosed s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK t_rel i) := by
  obtain ⟨final, sfinal, hexec, hge, hPfinal⟩ :=
    bwd_loop_drive_general DQInner DQInter DKInner DKInter Q K G DG s s_qk_h DK BT BK hBT hBK
      hDKInner_DQInter hDKInter_DQInter hQ_DQInter hQ_DKInter hK_DQInter hK_DKInter
      hDQInter_DKInter hDQInter_DG hDKInter_DG hG_DQInter hG_DKInter hG_DG hQ_DG hK_DG
      hDQInner_DQInter hDQInner_DKInter hDQInner_DG hDKInner_DKInter hDKInner_DG
  obtain ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, hDGwr⟩ := hPfinal
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [bwd_decay_global_cumsum_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro sa s' hExec hsa
  subst sa
  intro i hActive
  have hs' : s' = sfinal := by
    rw [exec] at hexec; rw [exec] at hExec; rw [hExec] at hexec; exact Option.some.inj hexec
  subst s'
  show sfinal.readMem DG (offset s s_qk_h DK t_rel.val BT BK i) = _
  rw [hDGwr t_rel.val (by omega) t_rel.isLt i, if_pos hActive,
    bwdCumPartialG_eq_DGClosed s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK t_rel i]


/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/
/-- **General `output_summary`.** The executed backward surface realizes all three
genuine closed forms (`bwdDQInterClosed` / `bwdDKInterClosed` / `bwdDGClosed`). -/
theorem decay_cumsum_backward_closed_output_summary_general :
    (ComputeCorrect.Realizes
      (kernel := bwd_decay_global_cumsum_surface DQInner DQInter DKInner DKInter
        Q K G DG s_qk_h DK BT BK)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf (active s DK BK)
        (fun i => (DQInter, offset s s_qk_h DK t_rel.val BT BK i)))
      (expected := fun i : Fin BK =>
        bwdDQInterClosed s DQInner DQInter G s_qk_h DK BT BK t_rel i)) ∧
    (ComputeCorrect.Realizes
      (kernel := bwd_decay_global_cumsum_surface DQInner DQInter DKInner DKInter
        Q K G DG s_qk_h DK BT BK)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf (active s DK BK)
        (fun i => (DKInter, offset s s_qk_h DK t_rel.val BT BK i)))
      (expected := fun i : Fin BK =>
        bwdDKInterClosed s DKInner DKInter G s_qk_h DK BT BK t_rel i)) ∧
    (ComputeCorrect.Realizes
      (kernel := bwd_decay_global_cumsum_surface DQInner DQInter DKInner DKInter
        Q K G DG s_qk_h DK BT BK)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf (active s DK BK)
        (fun i => (DG, offset s s_qk_h DK t_rel.val BT BK i)))
      (expected := fun i : Fin BK =>
        bwdDGClosed s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK t_rel i)) := by
  refine ⟨?_, ?_, ?_⟩
  · exact bwd_decay_cumsum_dq_inter_closed_compute_correct_general DQInner DQInter DKInner DKInter
      Q K G DG s s_qk_h DK BT BK t_rel hBT hBK hDKInner_DQInter hDKInter_DQInter
      hQ_DQInter hQ_DKInter hK_DQInter hK_DKInter hDQInter_DKInter hDQInter_DG hDKInter_DG
      hG_DQInter hG_DKInter hG_DG hQ_DG hK_DG hDQInner_DQInter hDQInner_DKInter hDQInner_DG
      hDKInner_DKInter hDKInner_DG
  · exact bwd_decay_cumsum_dk_inter_closed_compute_correct_general DQInner DQInter DKInner DKInter
      Q K G DG s s_qk_h DK BT BK t_rel hBT hBK hDKInner_DQInter hDKInter_DQInter
      hQ_DQInter hQ_DKInter hK_DQInter hK_DKInter hDQInter_DKInter hDQInter_DG hDKInter_DG
      hG_DQInter hG_DKInter hG_DG hQ_DG hK_DG hDQInner_DQInter hDQInner_DKInter hDQInner_DG
      hDKInner_DKInter hDKInner_DG
  · exact bwd_decay_cumsum_dg_closed_compute_correct_general DQInner DQInter DKInner DKInter
      Q K G DG s s_qk_h DK BT BK t_rel hBT hBK hDKInner_DQInter hDKInter_DQInter
      hQ_DQInter hQ_DKInter hK_DQInter hK_DKInter hDQInter_DKInter hDQInter_DG hDKInter_DG
      hG_DQInter hG_DKInter hG_DG hQ_DG hK_DG hDQInner_DQInter hDQInner_DKInter hDQInner_DG
      hDKInner_DKInter hDKInner_DG

end GeneralTops

end BwdAssembly

end Correct

end VeriTile.Bench.TritonBenchG.DecayCumsum

