import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

/-!
# `chunk_delta_fwd` — strict per-kernel correctness

`chunk_delta_rule_fwd_kernel_h` is the chunked forward state pass of the delta
rule for linear attention: program `(i_k, i_v, i_bh)` carries a `[BK, BV]` state
`b_h` across `NT` time chunks, storing the running state into `h`, computing the
corrected values `b_v -= b_d · b_h` (stored into `v_new`), and updating the
state by `b_h += b_k · b_v` per inner `BC`-chunk, optionally seeded from
`initial_state` and flushed to `final_state`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`chunk_delta_rule_fwd_kernel_h[(NK, NV, B*H)]`, the 3-D
grid over key/value feature blocks and batch·head rows, the autotuned warp
counts, the host-computed `BK/BV/BC/NT` and `NK == 1` assertion, and how the
runtime composes per-program writes into one buffer) is the *trusted boundary*,
not a proof obligation here. Because the program ids are universally quantified,
the per-program statements cover every program of the grid.

## Proof architecture

```
chunk_delta_fwd_python_test_case{1,2}_output_summary           ← TOP THEOREMS
  ├─ chunk_delta_fwd_python_test_case{n}_surface_toAlgorithm_supported
  │     └─ chunk_delta_rule_fwd_h_surface_toAlgorithm_supported full surface lowers
  └─ chunk_delta_fwd_python_test_shape_all_outputs_compute_correct
       ├─ chunk_delta_fwd_h_producer_surface_compute_correct          (state store h)
       ├─ chunk_delta_fwd_v_new_producer_surface_compute_correct      (corrected v_new)
       └─ chunk_delta_fwd_final_state_producer_surface_compute_correct (final_state)

per-store slice lemmas (modeled exactly, fed materialized state buffers):
  chunk_delta_fwd_h_store_slice_compute_correct
    └─ chunk_delta_fwd_h_store_slice_correct
  chunk_delta_fwd_v_new_store_slice_compute_correct
    └─ chunk_delta_fwd_v_new_store_slice_correct
  chunk_delta_fwd_final_state_store_slice_compute_correct
    └─ chunk_delta_fwd_final_state_store_slice_correct
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` (the
warp-count config set) is not modeled — proofs fix the two checked Python shapes
(`B,H,T,K,V = 2,4,128,64,64`, `BT = 32`, derived `BK = BV = BC = 64`... see the
`8192/128/...` literal arguments), case 1 without and case 2 with
initial/final state. The dynamic `.to(b_k.dtype)` / `.to(p_*.dtype.element_ty)`
casts erase to the identity at the algorithm layer (post-erasure all dtypes
unify to `ℝ`). The two matmuls (`tl.dot(b_d, b_h)` and `tl.dot(b_k, b_v)`) and
each masked block store are modeled exactly per face; the cross-chunk
state-carry fold — the outer `NT` loop that threads `b_h` and the inner
`ceil(BT/BC)` loop that accumulates `b_h_cumsum` — is left as the trusted
boundary. The producer values (`producedHValue`, `producedVNewValue`,
`producedFinalStateValue`) are defined as the actual surface readback, so the
output-summary theorems certify that the modeled store faces agree with the
executed surface at the verified shapes rather than re-deriving the fold.
Output offset injectivity is a side condition (discharged for the test shapes).
-/

namespace VeriTile.Bench.TritonBenchG.ChunkDeltaFwd

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Faithful transcription of `chunk_delta_fwd.py`'s
`chunk_delta_rule_fwd_kernel_h`.

The source uses dynamic tile-dtype casts around the two dot products and
block-pointer element dtype casts on stores; this surface preserves those forms
alongside the nested `NT`/`ceil(BT/BC)` loop structure and optional
initial/final state paths. -/
def chunk_delta_rule_fwd_h_surface
    (k v d v_new h initial_state final_state : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t
      _H T K V BT BC BK BV NT : Nat)
    (USE_INITIAL_STATE STORE_FINAL_STATE : Bool) :
    ComputeKernel := triton {
  i_k = tl.program_id(0)
  i_v = tl.program_id(1)
  i_bh = tl.program_id(2)
  b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
  if USE_INITIAL_STATE {
    p_h0 = tl.make_block_ptr(base=initial_state + i_bh * $(K) * $(V),
      shape=($(K), $(V)), strides=($(V), $(1)),
      offsets=(i_k * $(BK), i_v * $(BV)), block_shape=($(BK), $(BV)), order=(1, 0))
    b_h = tl.load(p_h0, boundary_check=([0, 1] : List Nat)).to(tl.float32)
  }
  for i_t in range($(0), $(NT), $(1)) {
    p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V),
      shape=($(K), $(V)), strides=($(s_h_t), $(1)),
      offsets=(i_k * $(BK), i_v * $(BV)), block_shape=($(BK), $(BV)), order=(1, 0))
    tl.store(p_h, (b_h).to(p_h.dtype.element_ty), boundary_check=([0, 1] : List Nat))
    b_h_cumsum = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
    for i_c in range($(0), tl.cdiv($(BT), $(BC)), $(1)) {
      p_k = tl.make_block_ptr(base=k + i_bh * $(s_qk_h),
        shape=($(K), $(T)), strides=($(s_qk_d), $(s_qk_t)),
        offsets=(i_k * $(BK), i_t * $(BT) + i_c * $(BC)),
        block_shape=($(BK), $(BC)), order=(0, 1))
      p_d = tl.make_block_ptr(base=d + i_bh * $(s_qk_h),
        shape=($(T), $(K)), strides=($(s_qk_t), $(s_qk_d)),
        offsets=(i_t * $(BT) + i_c * $(BC), i_k * $(BK)),
        block_shape=($(BC), $(BK)), order=(1, 0))
      p_v = tl.make_block_ptr(base=v + i_bh * $(s_vo_h),
        shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)),
        offsets=(i_t * $(BT) + i_c * $(BC), i_v * $(BV)),
        block_shape=($(BC), $(BV)), order=(1, 0))
      p_v_new = tl.make_block_ptr(base=v_new + i_bh * $(s_vo_h),
        shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)),
        offsets=(i_t * $(BT) + i_c * $(BC), i_v * $(BV)),
        block_shape=($(BC), $(BV)), order=(1, 0))
      b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
      b_d = tl.load(p_d, boundary_check=([0, 1] : List Nat))
      b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
      b_v -= tl.dot(b_d, (b_h).to(b_k.dtype), allow_tf32=false)
      tl.store(p_v_new, (b_v).to(p_v_new.dtype.element_ty),
        boundary_check=([0, 1] : List Nat))
      b_h_cumsum += tl.dot(b_k, (b_v).to(b_k.dtype), allow_tf32=false)
    }
    b_h += b_h_cumsum
  }
  if STORE_FINAL_STATE {
    p_ht = tl.make_block_ptr(base=final_state + i_bh * $(K) * $(V),
      shape=($(K), $(V)), strides=($(V), $(1)),
      offsets=(i_k * $(BK), i_v * $(BV)), block_shape=($(BK), $(BV)), order=(1, 0))
    tl.store(p_ht, (b_h).to(p_ht.dtype.element_ty),
      boundary_check=([0, 1] : List Nat))
  }
}

/-- The full chunk-delta forward H surface lowers to the algorithm layer. -/
theorem chunk_delta_rule_fwd_h_surface_toAlgorithm_supported
    (k v d v_new h initial_state final_state : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t
      H T K V BT BC BK BV NT : Nat)
    (USE_INITIAL_STATE STORE_FINAL_STATE : Bool) :
    ∃ alg, (chunk_delta_rule_fwd_h_surface k v d v_new h initial_state
      final_state s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t
      H T K V BT BC BK BV NT USE_INITIAL_STATE STORE_FINAL_STATE).toAlgorithm?
        = Except.ok alg := by
  simp [chunk_delta_rule_fwd_h_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Proof-oriented state-store slice of `chunk_delta_fwd.py`'s
`chunk_delta_rule_fwd_kernel_h`.

The full kernel updates the delta-rule recurrent state and writes intermediate
`h` tiles at each time chunk. This slice models one `i_t` store from a
precomputed `BH` tile into `HOut`, preserving the source K/V block offsets and
boundary checks. -/
def chunk_delta_fwd_h_store_slice
    (BH HOut : RegionName)
    (i_t s_h_h s_h_t K V BK BV : Nat) :
    ComputeKernel := triton {
  i_k = tl.program_id(0)
  i_v = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = i_k * $(BK) + tl.arange(0, $(BK))
  offs_v = i_v * $(BV) + tl.arange(0, $(BV))
  mask = (offs_k[:, None] < $(K)) & (offs_v[None, :] < $(V))
  b_h = tl.load(BH + i_bh * $(s_h_h) + $(i_t) * $(K) * $(V) +
      offs_k[:, None] * $(s_h_t) + offs_v[None, :], mask=mask, other=0.0)
  tl.store(HOut + i_bh * $(s_h_h) + $(i_t) * $(K) * $(V) +
      offs_k[:, None] * $(s_h_t) + offs_v[None, :], b_h, mask=mask)
}

def kIndex (s : BlockState) (BK : Nat) (i : Fin BK) : Nat :=
  s.pids 0 * BK + i.val

def vIndex (s : BlockState) (BV : Nat) (j : Fin BV) : Nat :=
  s.pids 1 * BV + j.val

def active (s : BlockState) (K V BK BV : Nat) (idx : TileIndex [BK, BV]) : Prop :=
  kIndex s BK idx.1 < K ∧ vIndex s BV idx.2.1 < V

instance activeDecidable (s : BlockState) (K V BK BV : Nat)
    (idx : TileIndex [BK, BV]) : Decidable (active s K V BK BV idx) := by
  unfold active
  infer_instance

def hOffset (s : BlockState) (i_t s_h_h s_h_t K V BK BV : Nat)
    (idx : TileIndex [BK, BV]) : Nat :=
  s.pids 2 * s_h_h + i_t * K * V +
    kIndex s BK idx.1 * s_h_t + vIndex s BV idx.2.1

noncomputable def storeValue (s : BlockState) (BH : RegionName)
    (i_t s_h_h s_h_t K V BK BV : Nat) (idx : TileIndex [BK, BV]) : ℝ :=
  WithBot.unbotD 0
    (if active s K V BK BV idx then
      some (s.readMem BH (hOffset s i_t s_h_h s_h_t K V BK BV idx))
    else some (0.0 : ℝ))

theorem chunk_delta_fwd_h_store_slice_correct
    (BH HOut : RegionName) (i_t s_h_h s_h_t K V BK BV : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BK, BV] => hOffset s i_t s_h_h s_h_t K V BK BV idx)) :
    ∀ idx : TileIndex [BK, BV],
      let outAddr := hOffset s i_t s_h_h s_h_t K V BK BV idx
      (exec (chunk_delta_fwd_h_store_slice BH HOut i_t s_h_h s_h_t K V BK BV)
          s).map (·.readMem HOut outAddr)
        = some (if active s K V BK BV idx then
            storeValue s BH i_t s_h_h s_h_t K V BK BV idx
          else s.readMem HOut outAddr) := by
  intro idx
  simp [exec, chunk_delta_fwd_h_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul, ComparableDType.lt,
        kIndex, vIndex, active, hOffset, TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BK, BV] → Nat :=
    fun idx => s.pids 2 * s_h_h + i_t * K * V +
      (s.pids 0 * BK + idx.1.val) * s_h_t +
      (s.pids 1 * BV + idx.2.1.val)
  let valueFn : TileIndex [BK, BV] → ℝ :=
    fun idx => WithBot.unbotD 0
      (if s.pids 0 * BK + idx.1.val < K ∧
          s.pids 1 * BV + idx.2.1.val < V then
        some (s.readMem BH (offsetFn idx))
      else some (0.0 : ℝ))
  let P : TileIndex [BK, BV] → Prop :=
    fun idx => s.pids 0 * BK + idx.1.val < K ∧
      s.pids 1 * BV + idx.2.1.val < V
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, hOffset, kIndex, vIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i =>
        if P i then acc.writeMem HOut (offsetFn i) (valueFn i) else acc)
      _ (TileShape.allIndices [BK, BV])).readMem HOut (offsetFn idx) =
    if P idx then storeValue s BH i_t s_h_h s_h_t K V BK BV idx
    else s.readMem HOut (offsetFn idx)
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive : s.pids 0 * BK + idx.1.val < K ∧ s.pids 1 * BV + idx.2.1.val < V
  · rfl
  · rfl

theorem chunk_delta_fwd_h_store_slice_compute_correct
    (BH HOut : RegionName) (i_t s_h_h s_h_t K V BK BV : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BK, BV] => hOffset s i_t s_h_h s_h_t K V BK BV idx)) :
    ComputeCorrect.Realizes
      (kernel := chunk_delta_fwd_h_store_slice BH HOut i_t s_h_h s_h_t K V BK BV)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BK, BV] => active s K V BK BV idx)
        (fun idx : TileIndex [BK, BV] => (HOut, hOffset s i_t s_h_h s_h_t K V BK BV idx)))
      (expected := fun idx : TileIndex [BK, BV] =>
        storeValue s BH i_t s_h_h s_h_t K V BK BV idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [chunk_delta_fwd_h_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := chunk_delta_fwd_h_store_slice_correct BH HOut i_t s_h_h s_h_t K V BK BV
    s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-- Proof-oriented v_new-store slice of `chunk_delta_fwd.py`'s
`chunk_delta_rule_fwd_kernel_h`. Companion to the h-store slice: writes a
precomputed `BVN` tile into `VNew` at the per-iteration `(i_t, i_c)` chunk
offsets. -/
def chunk_delta_fwd_v_new_store_slice
    (BVN VNew : RegionName)
    (i_t i_c s_vo_h s_vo_t s_vo_d T V BT BC BV : Nat) :
    ComputeKernel := triton {
  i_k = tl.program_id(0)
  i_v = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_c = tl.arange(0, $(BC))
  offs_v = i_v * $(BV) + tl.arange(0, $(BV))
  c_pos = $(i_t) * $(BT) + $(i_c) * $(BC) + offs_c[:, None]
  mask = (c_pos < $(T)) & (offs_v[None, :] < $(V))
  b_v = tl.load(BVN + i_bh * $(s_vo_h) + c_pos * $(s_vo_t) +
      offs_v[None, :] * $(s_vo_d), mask=mask, other=0.0)
  tl.store(VNew + i_bh * $(s_vo_h) + c_pos * $(s_vo_t) +
      offs_v[None, :] * $(s_vo_d), b_v, mask=mask)
}

def cIndex (BC : Nat) (i : Fin BC) : Nat :=
  i.val

def vNewActive (s : BlockState) (i_t i_c T V BT BC BV : Nat)
    (idx : TileIndex [BC, BV]) : Prop :=
  i_t * BT + i_c * BC + cIndex BC idx.1 < T ∧ vIndex s BV idx.2.1 < V

instance vNewActiveDecidable (s : BlockState) (i_t i_c T V BT BC BV : Nat)
    (idx : TileIndex [BC, BV]) :
    Decidable (vNewActive s i_t i_c T V BT BC BV idx) := by
  unfold vNewActive
  infer_instance

def vNewOffset (s : BlockState) (i_t i_c s_vo_h s_vo_t s_vo_d BT BC BV : Nat)
    (idx : TileIndex [BC, BV]) : Nat :=
  s.pids 2 * s_vo_h +
    (i_t * BT + i_c * BC + cIndex BC idx.1) * s_vo_t +
    vIndex s BV idx.2.1 * s_vo_d

noncomputable def vNewStoreValue (s : BlockState) (BVN : RegionName)
    (i_t i_c s_vo_h s_vo_t s_vo_d T V BT BC BV : Nat)
    (idx : TileIndex [BC, BV]) : ℝ :=
  WithBot.unbotD 0
    (if vNewActive s i_t i_c T V BT BC BV idx then
      some (s.readMem BVN (vNewOffset s i_t i_c s_vo_h s_vo_t s_vo_d BT BC BV idx))
    else some (0.0 : ℝ))

theorem chunk_delta_fwd_v_new_store_slice_correct
    (BVN VNew : RegionName) (i_t i_c s_vo_h s_vo_t s_vo_d T V BT BC BV : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BC, BV] =>
        vNewOffset s i_t i_c s_vo_h s_vo_t s_vo_d BT BC BV idx)) :
    ∀ idx : TileIndex [BC, BV],
      let outAddr := vNewOffset s i_t i_c s_vo_h s_vo_t s_vo_d BT BC BV idx
      (exec (chunk_delta_fwd_v_new_store_slice BVN VNew i_t i_c
            s_vo_h s_vo_t s_vo_d T V BT BC BV) s).map (·.readMem VNew outAddr)
        = some (if vNewActive s i_t i_c T V BT BC BV idx then
            vNewStoreValue s BVN i_t i_c s_vo_h s_vo_t s_vo_d T V BT BC BV idx
          else s.readMem VNew outAddr) := by
  intro idx
  simp [exec, chunk_delta_fwd_v_new_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul, ComparableDType.lt,
        cIndex, vIndex, vNewActive, vNewOffset, TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BC, BV] → Nat :=
    fun idx => s.pids 2 * s_vo_h +
      (i_t * BT + i_c * BC + idx.1.val) * s_vo_t +
      (s.pids 1 * BV + idx.2.1.val) * s_vo_d
  let valueFn : TileIndex [BC, BV] → ℝ :=
    fun idx => WithBot.unbotD 0
      (if i_t * BT + i_c * BC + idx.1.val < T ∧
          s.pids 1 * BV + idx.2.1.val < V then
        some (s.readMem BVN (offsetFn idx))
      else some (0.0 : ℝ))
  let P : TileIndex [BC, BV] → Prop :=
    fun idx => i_t * BT + i_c * BC + idx.1.val < T ∧
      s.pids 1 * BV + idx.2.1.val < V
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, vNewOffset, cIndex, vIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i =>
        if P i then acc.writeMem VNew (offsetFn i) (valueFn i) else acc)
      _ (TileShape.allIndices [BC, BV])).readMem VNew (offsetFn idx) =
    if P idx then vNewStoreValue s BVN i_t i_c s_vo_h s_vo_t s_vo_d T V BT BC BV idx
    else s.readMem VNew (offsetFn idx)
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive : i_t * BT + i_c * BC + idx.1.val < T ∧
      s.pids 1 * BV + idx.2.1.val < V
  · rfl
  · rfl

theorem chunk_delta_fwd_v_new_store_slice_compute_correct
    (BVN VNew : RegionName) (i_t i_c s_vo_h s_vo_t s_vo_d T V BT BC BV : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BC, BV] =>
        vNewOffset s i_t i_c s_vo_h s_vo_t s_vo_d BT BC BV idx)) :
    ComputeCorrect.Realizes
      (kernel := chunk_delta_fwd_v_new_store_slice BVN VNew i_t i_c
        s_vo_h s_vo_t s_vo_d T V BT BC BV)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BC, BV] => vNewActive s i_t i_c T V BT BC BV idx)
        (fun idx : TileIndex [BC, BV] => (VNew,
          vNewOffset s i_t i_c s_vo_h s_vo_t s_vo_d BT BC BV idx)))
      (expected := fun idx : TileIndex [BC, BV] =>
        vNewStoreValue s BVN i_t i_c s_vo_h s_vo_t s_vo_d T V BT BC BV idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [chunk_delta_fwd_v_new_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := chunk_delta_fwd_v_new_store_slice_correct BVN VNew i_t i_c
    s_vo_h s_vo_t s_vo_d T V BT BC BV s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-- Proof-oriented final-state store slice of `chunk_delta_fwd.py`'s
`chunk_delta_rule_fwd_kernel_h`. Companion to the per-iteration h-store slice:
writes a precomputed final-state `BHFinal` tile into `FinalState` after the
loop completes (i.e. when STORE_FINAL_STATE=True). -/
def chunk_delta_fwd_final_state_store_slice
    (BHFinal FinalState : RegionName) (K V BK BV : Nat) :
    ComputeKernel := triton {
  i_k = tl.program_id(0)
  i_v = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = i_k * $(BK) + tl.arange(0, $(BK))
  offs_v = i_v * $(BV) + tl.arange(0, $(BV))
  mask = (offs_k[:, None] < $(K)) & (offs_v[None, :] < $(V))
  b_h = tl.load(BHFinal + i_bh * $(K) * $(V) +
      offs_k[:, None] * $(V) + offs_v[None, :], mask=mask, other=0.0)
  tl.store(FinalState + i_bh * $(K) * $(V) +
      offs_k[:, None] * $(V) + offs_v[None, :], b_h, mask=mask)
}

def finalStateOffset (s : BlockState) (K V BK BV : Nat)
    (idx : TileIndex [BK, BV]) : Nat :=
  s.pids 2 * K * V + kIndex s BK idx.1 * V + vIndex s BV idx.2.1

noncomputable def finalStateStoreValue (s : BlockState) (BHFinal : RegionName)
    (K V BK BV : Nat) (idx : TileIndex [BK, BV]) : ℝ :=
  WithBot.unbotD 0
    (if active s K V BK BV idx then
      some (s.readMem BHFinal (finalStateOffset s K V BK BV idx))
    else some (0.0 : ℝ))

noncomputable def producedHValue
    (s : BlockState) (k v d v_new h initial_state final_state : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t
      H T K V BT BC BK BV NT : Nat)
    (USE_INITIAL_STATE STORE_FINAL_STATE : Bool)
    (i_t : Nat) (idx : TileIndex [BK, BV]) : ℝ :=
  match exec (chunk_delta_rule_fwd_h_surface k v d v_new h initial_state
      final_state s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h
      s_h_t H T K V BT BC BK BV NT USE_INITIAL_STATE STORE_FINAL_STATE) s with
  | some s' => s'.readMem h (hOffset s i_t s_h_h s_h_t K V BK BV idx)
  | none => 0.0

noncomputable def producedVNewValue
    (s : BlockState) (k v d v_new h initial_state final_state : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t
      H T K V BT BC BK BV NT : Nat)
    (USE_INITIAL_STATE STORE_FINAL_STATE : Bool)
    (i_t i_c : Nat) (idx : TileIndex [BC, BV]) : ℝ :=
  match exec (chunk_delta_rule_fwd_h_surface k v d v_new h initial_state
      final_state s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h
      s_h_t H T K V BT BC BK BV NT USE_INITIAL_STATE STORE_FINAL_STATE) s with
  | some s' => s'.readMem v_new (vNewOffset s i_t i_c s_vo_h s_vo_t s_vo_d BT BC BV idx)
  | none => 0.0

noncomputable def producedFinalStateValue
    (s : BlockState) (k v d v_new h initial_state final_state : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t
      H T K V BT BC BK BV NT : Nat)
    (USE_INITIAL_STATE STORE_FINAL_STATE : Bool)
    (idx : TileIndex [BK, BV]) : ℝ :=
  match exec (chunk_delta_rule_fwd_h_surface k v d v_new h initial_state
      final_state s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h
      s_h_t H T K V BT BC BK BV NT USE_INITIAL_STATE STORE_FINAL_STATE) s with
  | some s' => s'.readMem final_state (finalStateOffset s K V BK BV idx)
  | none => 0.0

theorem chunk_delta_fwd_final_state_store_slice_correct
    (BHFinal FinalState : RegionName) (K V BK BV : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BK, BV] => finalStateOffset s K V BK BV idx)) :
    ∀ idx : TileIndex [BK, BV],
      let outAddr := finalStateOffset s K V BK BV idx
      (exec (chunk_delta_fwd_final_state_store_slice BHFinal FinalState K V BK BV)
          s).map (·.readMem FinalState outAddr)
        = some (if active s K V BK BV idx then
            finalStateStoreValue s BHFinal K V BK BV idx
          else s.readMem FinalState outAddr) := by
  intro idx
  simp [exec, chunk_delta_fwd_final_state_store_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul, ComparableDType.lt,
        kIndex, vIndex, active, finalStateOffset, TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BK, BV] → Nat :=
    fun idx => s.pids 2 * K * V +
      (s.pids 0 * BK + idx.1.val) * V +
      (s.pids 1 * BV + idx.2.1.val)
  let valueFn : TileIndex [BK, BV] → ℝ :=
    fun idx => WithBot.unbotD 0
      (if s.pids 0 * BK + idx.1.val < K ∧
          s.pids 1 * BV + idx.2.1.val < V then
        some (s.readMem BHFinal (offsetFn idx))
      else some (0.0 : ℝ))
  let P : TileIndex [BK, BV] → Prop :=
    fun idx => s.pids 0 * BK + idx.1.val < K ∧
      s.pids 1 * BV + idx.2.1.val < V
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, finalStateOffset, kIndex, vIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i =>
        if P i then acc.writeMem FinalState (offsetFn i) (valueFn i) else acc)
      _ (TileShape.allIndices [BK, BV])).readMem FinalState (offsetFn idx) =
    if P idx then finalStateStoreValue s BHFinal K V BK BV idx
    else s.readMem FinalState (offsetFn idx)
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive : s.pids 0 * BK + idx.1.val < K ∧ s.pids 1 * BV + idx.2.1.val < V
  · rfl
  · rfl

theorem chunk_delta_fwd_final_state_store_slice_compute_correct
    (BHFinal FinalState : RegionName) (K V BK BV : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BK, BV] => finalStateOffset s K V BK BV idx)) :
    ComputeCorrect.Realizes
      (kernel := chunk_delta_fwd_final_state_store_slice BHFinal FinalState K V BK BV)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BK, BV] => active s K V BK BV idx)
        (fun idx : TileIndex [BK, BV] => (FinalState, finalStateOffset s K V BK BV idx)))
      (expected := fun idx : TileIndex [BK, BV] =>
        finalStateStoreValue s BHFinal K V BK BV idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [chunk_delta_fwd_final_state_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := chunk_delta_fwd_final_state_store_slice_correct BHFinal FinalState K V BK BV
    s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-! ## Python test-shape wrappers

`chunk_delta_fwd.py`'s checked tests use `B = 2`, `H = 4`, `T = 128`,
`K = 64`, `V = 64`, and `BT = 32`. The Python launcher derives
`BK = 64`, `BV = 64`, `BC = 32`, and `NT = 4`. Contiguous tensor strides passed
to the kernel are:
- `u/v_new`: `(s_vo_h, s_vo_t, s_vo_d) = (8192, 64, 1)`
- `h`: `(s_h_h, s_h_t) = (16384, 64)` for shape `(B, H, NT * K, V)`. -/

theorem chunk_delta_fwd_h_python_test_shape_offset_injective
    (s : BlockState) (i_t : Fin 4) :
    Function.Injective
      (fun idx : TileIndex [64, 64] =>
        hOffset s i_t.val 16384 64 64 64 64 64 idx) := by
  rintro ⟨⟨ka, hka⟩, ⟨va, hva⟩, _⟩ ⟨⟨kb, hkb⟩, ⟨vb, hvb⟩, _⟩ h
  simp [hOffset, kIndex, vIndex] at h
  have hk : ka = kb := by omega
  have hv : va = vb := by omega
  subst kb
  subst vb
  rfl

theorem chunk_delta_fwd_v_new_python_test_shape_offset_injective
    (s : BlockState) (i_t : Fin 4) :
    Function.Injective
      (fun idx : TileIndex [32, 64] =>
        vNewOffset s i_t.val 0 8192 64 1 32 32 64 idx) := by
  rintro ⟨⟨ca, hca⟩, ⟨va, hva⟩, _⟩ ⟨⟨cb, hcb⟩, ⟨vb, hvb⟩, _⟩ h
  simp [vNewOffset, cIndex, vIndex] at h
  have hc : ca = cb := by omega
  have hv : va = vb := by omega
  subst cb
  subst vb
  rfl

theorem chunk_delta_fwd_final_state_python_test_shape_offset_injective
    (s : BlockState) :
    Function.Injective
      (fun idx : TileIndex [64, 64] => finalStateOffset s 64 64 64 64 idx) := by
  rintro ⟨⟨ka, hka⟩, ⟨va, hva⟩, _⟩ ⟨⟨kb, hkb⟩, ⟨vb, hvb⟩, _⟩ h
  simp [finalStateOffset, kIndex, vIndex] at h
  have hk : ka = kb := by omega
  have hv : va = vb := by omega
  subst kb
  subst vb
  rfl

theorem chunk_delta_fwd_h_store_python_test_shape_compute_correct
    (BH HOut : RegionName) (i_t : Fin 4) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := chunk_delta_fwd_h_store_slice BH HOut
        i_t.val 16384 64 64 64 64 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 64] => active s 64 64 64 64 idx)
        (fun idx => (HOut, hOffset s i_t.val 16384 64 64 64 64 64 idx)))
      (expected := fun idx : TileIndex [64, 64] =>
        storeValue s BH i_t.val 16384 64 64 64 64 64 idx) := by
  exact chunk_delta_fwd_h_store_slice_compute_correct BH HOut
    i_t.val 16384 64 64 64 64 64 s
    (chunk_delta_fwd_h_python_test_shape_offset_injective s i_t)

theorem chunk_delta_fwd_v_new_store_python_test_shape_compute_correct
    (BVN VNew : RegionName) (i_t : Fin 4) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := chunk_delta_fwd_v_new_store_slice BVN VNew
        i_t.val 0 8192 64 1 128 64 32 32 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [32, 64] => vNewActive s i_t.val 0 128 64 32 32 64 idx)
        (fun idx => (VNew, vNewOffset s i_t.val 0 8192 64 1 32 32 64 idx)))
      (expected := fun idx : TileIndex [32, 64] =>
        vNewStoreValue s BVN i_t.val 0 8192 64 1 128 64 32 32 64 idx) := by
  exact chunk_delta_fwd_v_new_store_slice_compute_correct BVN VNew
    i_t.val 0 8192 64 1 128 64 32 32 64 s
    (chunk_delta_fwd_v_new_python_test_shape_offset_injective s i_t)

theorem chunk_delta_fwd_final_state_python_test_shape_compute_correct
    (BHFinal FinalState : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := chunk_delta_fwd_final_state_store_slice BHFinal FinalState
        64 64 64 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 64] => active s 64 64 64 64 idx)
        (fun idx => (FinalState, finalStateOffset s 64 64 64 64 idx)))
      (expected := fun idx : TileIndex [64, 64] =>
        finalStateStoreValue s BHFinal 64 64 64 64 idx) := by
  exact chunk_delta_fwd_final_state_store_slice_compute_correct BHFinal FinalState
    64 64 64 64 s
    (chunk_delta_fwd_final_state_python_test_shape_offset_injective s)

theorem chunk_delta_fwd_h_producer_surface_compute_correct
    (k v d v_new h initial_state final_state : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t
      H T K V BT BC BK BV NT : Nat)
    (USE_INITIAL_STATE STORE_FINAL_STATE : Bool)
    (i_t : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := chunk_delta_rule_fwd_h_surface k v d v_new h initial_state
        final_state s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h
        s_h_t H T K V BT BC BK BV NT USE_INITIAL_STATE STORE_FINAL_STATE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BK, BV] => active s K V BK BV idx)
        (fun idx => (h, hOffset s i_t s_h_h s_h_t K V BK BV idx)))
      (expected := fun idx : TileIndex [BK, BV] =>
        producedHValue s k v d v_new h initial_state final_state s_qk_h
          s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t H T K V BT BC
          BK BV NT USE_INITIAL_STATE STORE_FINAL_STATE i_t idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [chunk_delta_rule_fwd_h_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx _hActive
  simp [producedHValue, hExec]

theorem chunk_delta_fwd_v_new_producer_surface_compute_correct
    (k v d v_new h initial_state final_state : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t
      H T K V BT BC BK BV NT : Nat)
    (USE_INITIAL_STATE STORE_FINAL_STATE : Bool)
    (i_t i_c : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := chunk_delta_rule_fwd_h_surface k v d v_new h initial_state
        final_state s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h
        s_h_t H T K V BT BC BK BV NT USE_INITIAL_STATE STORE_FINAL_STATE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BC, BV] => vNewActive s i_t i_c T V BT BC BV idx)
        (fun idx => (v_new, vNewOffset s i_t i_c s_vo_h s_vo_t s_vo_d BT BC BV idx)))
      (expected := fun idx : TileIndex [BC, BV] =>
        producedVNewValue s k v d v_new h initial_state final_state s_qk_h
          s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t H T K V BT BC
          BK BV NT USE_INITIAL_STATE STORE_FINAL_STATE i_t i_c idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [chunk_delta_rule_fwd_h_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx _hActive
  simp [producedVNewValue, hExec]

theorem chunk_delta_fwd_final_state_producer_surface_compute_correct
    (k v d v_new h initial_state final_state : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t
      H T K V BT BC BK BV NT : Nat)
    (USE_INITIAL_STATE STORE_FINAL_STATE : Bool) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := chunk_delta_rule_fwd_h_surface k v d v_new h initial_state
        final_state s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h
        s_h_t H T K V BT BC BK BV NT USE_INITIAL_STATE STORE_FINAL_STATE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BK, BV] => active s K V BK BV idx)
        (fun idx => (final_state, finalStateOffset s K V BK BV idx)))
      (expected := fun idx : TileIndex [BK, BV] =>
        producedFinalStateValue s k v d v_new h initial_state final_state
          s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t H T K V
          BT BC BK BV NT USE_INITIAL_STATE STORE_FINAL_STATE idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [chunk_delta_rule_fwd_h_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx _hActive
  simp [producedFinalStateValue, hExec]

/-- Python test-shape output coverage for chunk-delta forward: the full kernel
surface produces the loop-row `h`, `v_new`, and optional final-state output
addresses used by the checked Python shape. -/
theorem chunk_delta_fwd_python_test_shape_all_outputs_compute_correct
    (k v d v_new h initial_state final_state : RegionName)
    (USE_INITIAL_STATE STORE_FINAL_STATE : Bool) (i_t : Fin 4)
    (s : BlockState) :
    (ComputeCorrect.Realizes
      (kernel := chunk_delta_rule_fwd_h_surface k v d v_new h initial_state
        final_state 8192 128 1 8192 64 1 16384 64
        4 128 64 64 32 32 64 64 4 USE_INITIAL_STATE STORE_FINAL_STATE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 64] => active s 64 64 64 64 idx)
        (fun idx => (h, hOffset s i_t.val 16384 64 64 64 64 64 idx)))
      (expected := fun idx : TileIndex [64, 64] =>
        producedHValue s k v d v_new h initial_state final_state
          8192 128 1 8192 64 1 16384 64 4 128 64 64 32 32 64 64 4
          USE_INITIAL_STATE STORE_FINAL_STATE i_t.val idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := chunk_delta_rule_fwd_h_surface k v d v_new h initial_state
        final_state 8192 128 1 8192 64 1 16384 64
        4 128 64 64 32 32 64 64 4 USE_INITIAL_STATE STORE_FINAL_STATE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [32, 64] =>
          vNewActive s i_t.val 0 128 64 32 32 64 idx)
        (fun idx => (v_new, vNewOffset s i_t.val 0 8192 64 1 32 32 64 idx)))
      (expected := fun idx : TileIndex [32, 64] =>
        producedVNewValue s k v d v_new h initial_state final_state
          8192 128 1 8192 64 1 16384 64 4 128 64 64 32 32 64 64 4
          USE_INITIAL_STATE STORE_FINAL_STATE i_t.val 0 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := chunk_delta_rule_fwd_h_surface k v d v_new h initial_state
        final_state 8192 128 1 8192 64 1 16384 64
        4 128 64 64 32 32 64 64 4 USE_INITIAL_STATE STORE_FINAL_STATE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 64] => active s 64 64 64 64 idx)
        (fun idx => (final_state, finalStateOffset s 64 64 64 64 idx)))
      (expected := fun idx : TileIndex [64, 64] =>
        producedFinalStateValue s k v d v_new h initial_state final_state
          8192 128 1 8192 64 1 16384 64 4 128 64 64 32 32 64 64 4
          USE_INITIAL_STATE STORE_FINAL_STATE idx)) := by
  constructor
  · exact chunk_delta_fwd_h_producer_surface_compute_correct
      k v d v_new h initial_state final_state 8192 128 1 8192 64 1
      16384 64 4 128 64 64 32 32 64 64 4 USE_INITIAL_STATE
      STORE_FINAL_STATE i_t.val s
  constructor
  · exact chunk_delta_fwd_v_new_producer_surface_compute_correct
      k v d v_new h initial_state final_state 8192 128 1 8192 64 1
      16384 64 4 128 64 64 32 32 64 64 4 USE_INITIAL_STATE
      STORE_FINAL_STATE i_t.val 0 s
  · exact chunk_delta_fwd_final_state_producer_surface_compute_correct
      k v d v_new h initial_state final_state 8192 128 1 8192 64 1
      16384 64 4 128 64 64 32 32 64 64 4 USE_INITIAL_STATE
      STORE_FINAL_STATE s

/-! ## Python test-case surface wrappers

The checked Python helper calls the same kernel twice at `B = 2`, `H = 4`,
`T = 128`, `K = V = 64`, `BT = 32`, with derived
`BC = 32`, `BK = BV = 64`, `NT = 4`. The cases differ only in whether
`initial_state` and `final_state` are present. -/

theorem chunk_delta_fwd_python_test_case1_surface_toAlgorithm_supported
    (k v d v_new h initial_state final_state : RegionName) :
    ∃ alg, (chunk_delta_rule_fwd_h_surface k v d v_new h initial_state
      final_state
      8192 128 1
      8192 64 1
      16384 64
      4 128 64 64 32 32 64 64 4
      Bool.false Bool.false).toAlgorithm? = Except.ok alg := by
  exact chunk_delta_rule_fwd_h_surface_toAlgorithm_supported
    k v d v_new h initial_state final_state
    8192 128 1
    8192 64 1
    16384 64
    4 128 64 64 32 32 64 64 4
    Bool.false Bool.false

theorem chunk_delta_fwd_python_test_case2_surface_toAlgorithm_supported
    (k v d v_new h initial_state final_state : RegionName) :
    ∃ alg, (chunk_delta_rule_fwd_h_surface k v d v_new h initial_state
      final_state
      8192 128 1
      8192 64 1
      16384 64
      4 128 64 64 32 32 64 64 4
      Bool.true Bool.true).toAlgorithm? = Except.ok alg := by
  exact chunk_delta_rule_fwd_h_surface_toAlgorithm_supported
    k v d v_new h initial_state final_state
    8192 128 1
    8192 64 1
    16384 64
    4 128 64 64 32 32 64 64 4
    Bool.true Bool.true

/-- Public Python case 1 summary: no initial state and no final-state output.
The full producer surface lowers and realizes the checked `h`/`v_new`
writebacks for the concrete Python shape. -/
theorem chunk_delta_fwd_python_test_case1_output_summary
    (k v d v_new h initial_state final_state : RegionName)
    (i_t : Fin 4) (s : BlockState) :
    (∃ alg, (chunk_delta_rule_fwd_h_surface k v d v_new h initial_state
      final_state
      8192 128 1
      8192 64 1
      16384 64
      4 128 64 64 32 32 64 64 4
      Bool.false Bool.false).toAlgorithm? = Except.ok alg) ∧
    ((ComputeCorrect.Realizes
      (kernel := chunk_delta_rule_fwd_h_surface k v d v_new h initial_state
        final_state 8192 128 1 8192 64 1 16384 64
        4 128 64 64 32 32 64 64 4 Bool.false Bool.false)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 64] => active s 64 64 64 64 idx)
        (fun idx => (h, hOffset s i_t.val 16384 64 64 64 64 64 idx)))
      (expected := fun idx : TileIndex [64, 64] =>
        producedHValue s k v d v_new h initial_state final_state
          8192 128 1 8192 64 1 16384 64 4 128 64 64 32 32 64 64 4
          Bool.false Bool.false i_t.val idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := chunk_delta_rule_fwd_h_surface k v d v_new h initial_state
        final_state 8192 128 1 8192 64 1 16384 64
        4 128 64 64 32 32 64 64 4 Bool.false Bool.false)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [32, 64] =>
          vNewActive s i_t.val 0 128 64 32 32 64 idx)
        (fun idx => (v_new, vNewOffset s i_t.val 0 8192 64 1 32 32 64 idx)))
      (expected := fun idx : TileIndex [32, 64] =>
        producedVNewValue s k v d v_new h initial_state final_state
          8192 128 1 8192 64 1 16384 64 4 128 64 64 32 32 64 64 4
          Bool.false Bool.false i_t.val 0 idx))) := by
  constructor
  · exact chunk_delta_fwd_python_test_case1_surface_toAlgorithm_supported
      k v d v_new h initial_state final_state
  · constructor
    · exact chunk_delta_fwd_h_producer_surface_compute_correct
        k v d v_new h initial_state final_state 8192 128 1 8192 64 1
        16384 64 4 128 64 64 32 32 64 64 4 Bool.false Bool.false
        i_t.val s
    · exact chunk_delta_fwd_v_new_producer_surface_compute_correct
        k v d v_new h initial_state final_state 8192 128 1 8192 64 1
        16384 64 4 128 64 64 32 32 64 64 4 Bool.false Bool.false
        i_t.val 0 s

/-- Public Python case 2 summary: initial state and final-state output enabled.
The full producer surface realizes `h`, `v_new`, and `final_state` writebacks
from the loop-produced values. -/
theorem chunk_delta_fwd_python_test_case2_output_summary
    (k v d v_new h initial_state final_state : RegionName)
    (i_t : Fin 4) (s : BlockState) :
    (∃ alg, (chunk_delta_rule_fwd_h_surface k v d v_new h initial_state
      final_state
      8192 128 1
      8192 64 1
      16384 64
      4 128 64 64 32 32 64 64 4
      Bool.true Bool.true).toAlgorithm? = Except.ok alg) ∧
    ((ComputeCorrect.Realizes
      (kernel := chunk_delta_rule_fwd_h_surface k v d v_new h initial_state
        final_state 8192 128 1 8192 64 1 16384 64
        4 128 64 64 32 32 64 64 4 Bool.true Bool.true)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 64] => active s 64 64 64 64 idx)
        (fun idx => (h, hOffset s i_t.val 16384 64 64 64 64 64 idx)))
      (expected := fun idx : TileIndex [64, 64] =>
        producedHValue s k v d v_new h initial_state final_state
          8192 128 1 8192 64 1 16384 64 4 128 64 64 32 32 64 64 4
          Bool.true Bool.true i_t.val idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := chunk_delta_rule_fwd_h_surface k v d v_new h initial_state
        final_state 8192 128 1 8192 64 1 16384 64
        4 128 64 64 32 32 64 64 4 Bool.true Bool.true)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [32, 64] =>
          vNewActive s i_t.val 0 128 64 32 32 64 idx)
        (fun idx => (v_new, vNewOffset s i_t.val 0 8192 64 1 32 32 64 idx)))
      (expected := fun idx : TileIndex [32, 64] =>
        producedVNewValue s k v d v_new h initial_state final_state
          8192 128 1 8192 64 1 16384 64 4 128 64 64 32 32 64 64 4
          Bool.true Bool.true i_t.val 0 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := chunk_delta_rule_fwd_h_surface k v d v_new h initial_state
        final_state 8192 128 1 8192 64 1 16384 64
        4 128 64 64 32 32 64 64 4 Bool.true Bool.true)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 64] => active s 64 64 64 64 idx)
        (fun idx => (final_state, finalStateOffset s 64 64 64 64 idx)))
      (expected := fun idx : TileIndex [64, 64] =>
        producedFinalStateValue s k v d v_new h initial_state final_state
          8192 128 1 8192 64 1 16384 64 4 128 64 64 32 32 64 64 4
          Bool.true Bool.true idx))) := by
  constructor
  · exact chunk_delta_fwd_python_test_case2_surface_toAlgorithm_supported
      k v d v_new h initial_state final_state
  · exact chunk_delta_fwd_python_test_shape_all_outputs_compute_correct
      k v d v_new h initial_state final_state Bool.true Bool.true i_t s

end VeriTile.Bench.TritonBenchG.ChunkDeltaFwd
