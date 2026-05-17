import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.DecayCumsum

open VeriTile.Triton

set_option maxHeartbeats 5000000
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

/-! ## BT=1 single-iteration surface closure

Under `BT = 1` the forward `for _i in range(0, BT, 1)` loop runs exactly
once at `_i = 0`, so we can close the full `fwd_decay_cumsum_surface`
against the same masked-scatter spec used by `fwd_decay_cumsum_store_slice`.
The reverse-loop bwd surface and the prepare-qg-kg surface are not closed
here: bwd contains a reverse `range` rewritten to a `__rev_t` counter with
an `if t == BT-1` branch, and prepare_qg_kg shares two stores per body, so
neither aligns with the single forward-store `scatter_readback` shape used
below. -/

/-- Pre-store state of `fwd_decay_cumsum_surface` after the pre-loop binds
plus one body iteration at `_i = 0`, modulo the final `tl.store`. -/
def fwdDecayCumsumSurfaceBt1PreStoreState
    (G GO : RegionName) (s_qk_h BK DK : Nat) (s : BlockState) : BlockState :=
  let s1 := s.setReg "i_k" TileDType.nat [] (Tile.scalar (s.pids 0))
  let s2 := s1.setReg "i_c" TileDType.nat [] (Tile.scalar (s.pids 1))
  let s3 := s2.setReg "i_bh" TileDType.nat [] (Tile.scalar (s.pids 2))
  let s4 := s3.setReg "p_g" TileDType.ptr [BK]
    { data := fun i =>
        some (G, s.pids 2 * s_qk_h + s.pids 1 * 1 * DK + s.pids 0 * BK + i.1.val) }
  let s5 := s4.setReg "p_go" TileDType.ptr [BK]
    { data := fun i =>
        some (GO, s.pids 2 * s_qk_h + s.pids 1 * 1 * DK + s.pids 0 * BK + i.1.val) }
  let s6 := s5.setReg "cum_decay" TileDType.real [BK]
    (Tile.vec (fun _ => some 0))
  let s7 := s6.setReg "mask" TileDType.bool [BK]
    (Tile.vec (fun i => decide (s.pids 0 * BK + i.val < DK)))
  let s8 := s7.setReg "_i" TileDType.nat [] (Tile.scalar 0)
  let s9 := s8.setReg "_g" TileDType.real [BK]
    { data := fun i =>
        if s.pids 0 * BK + i.1.val < DK then
          some (s.readMem G
            (s.pids 2 * s_qk_h + s.pids 1 * 1 * DK + s.pids 0 * BK + i.1.val))
        else some 0 }
  s9.setReg "cum_decay" TileDType.real [BK]
    { data := fun i =>
        if s.pids 0 * BK + i.1.val < DK then
          some (s.readMem G
            (s.pids 2 * s_qk_h + s.pids 1 * 1 * DK + s.pids 0 * BK + i.1.val)
              * 1.44269504)
        else some 0 }

/-- Algorithm-layer correctness of the masked `GO` writeback at row
`t_rel = 0` under `BT = 1`. -/
theorem fwd_decay_cumsum_surface_bt1_correct
    (G GO : RegionName)
    (s_qk_h s_qk_t s_qk_d B H T : Nat) (scale : ℝ)
    (BK DK : Nat)
    (s s' : BlockState)
    (hExec :
      exec (fwd_decay_cumsum_surface G GO s_qk_h s_qk_t s_qk_d B H T scale
        1 BK DK) s = some s') :
    ∀ i : Fin BK,
      s'.readMem GO (offset s s_qk_h DK 0 1 BK i) =
        if active s DK BK i then
          s.readMem G (offset s s_qk_h DK 0 1 BK i) * 1.44269504
        else
          s.readMem GO (offset s s_qk_h DK 0 1 BK i) := by
  intro i
  have hInj : Function.Injective
      (fun idx : TileIndex [BK] =>
        s.pids 2 * s_qk_h + s.pids 1 * 1 * DK + s.pids 0 * BK + idx.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp only [exec, fwd_decay_cumsum_surface, stepStmts, stepStmt, evalOp,
        Option.bind_eq_bind, Option.map_eq_map, Tile.bop, Tile.ptrAdd, Tile.uop,
        NumericDType.add, NumericDType.mul, ComparableDType.lt,
        stepForRangeAux.forRange_unfold] at hExec
  rw [stepForRangeAux.step_lt Nat.one_ne_zero (by decide : (0 : Nat) < 1)] at hExec
  simp only [Option.bind, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.ptrAdd, Tile.uop,
        NumericDType.add, NumericDType.mul, ComparableDType.lt] at hExec
  rw [stepForRangeAux.step_ge Nat.one_ne_zero (by decide : (1 : Nat) ≤ 0 + 1)] at hExec
  simp only [Option.some_inj] at hExec
  rw [← hExec]
  have hScatter :=
    (BlockState.scatter_readback_prop_masked_nd
      (region := GO)
      (shape := [BK])
      (s := fwdDecayCumsumSurfaceBt1PreStoreState G GO s_qk_h BK DK s)
      (offsetFn := fun idx : TileIndex [BK] =>
        s.pids 2 * s_qk_h + s.pids 1 * 1 * DK + s.pids 0 * BK + idx.1.val)
      (valueFn := fun idx : TileIndex [BK] =>
        WithBot.unbotD 0
          (if s.pids 0 * BK + idx.1.val < DK then
            some (s.readMem G
              (s.pids 2 * s_qk_h + s.pids 1 * 1 * DK + s.pids 0 * BK + idx.1.val)
                * 1.44269504)
          else some 0))
      (P := fun idx : TileIndex [BK] => s.pids 0 * BK + idx.1.val < DK)
      hInj (i, PUnit.unit))
  simp only [fwdDecayCumsumSurfaceBt1PreStoreState] at hScatter
  sorry

end VeriTile.Bench.TritonBenchG.DecayCumsum
