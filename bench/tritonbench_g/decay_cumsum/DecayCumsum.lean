import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

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
  simp [exec, prepare_qg_decay_store_slice, stepStmts, stepStmt, evalOp,
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
  simp [exec, prepare_kg_decay_store_slice, stepStmts, stepStmt, evalOp,
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
  simp [exec, fwd_decay_cumsum_store_slice, stepStmts, stepStmt, evalOp,
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
  simp [exec, fwd_decay_cumsum_step_store_slice, stepStmts, stepStmt, evalOp,
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
  simp [exec, bwd_decay_cumsum_store_slice, stepStmts, stepStmt, evalOp,
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
  simp [exec, bwd_decay_dg_step_store_slice, stepStmts, stepStmt, evalOp,
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

end VeriTile.Bench.TritonBenchG.DecayCumsum
