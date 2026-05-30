import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

/-!
# `fused_rwkv6_kernel` — strict per-kernel correctness

`fused_recurrent_rwkv6_fwd_kernel` is the RWKV-6 forward recurrent state scan:
each program carries a `[BV, BK]` state matrix `b_h` across the `0..T` time
loop, computing per-step `o_t = (b_h + u * k_t ⊗ v_t) · q_t` and updating
`b_h = b_h * exp(w_t) + k_t ⊗ v_t`, with an optional initial-state seed and an
optional final-state store.

## Scope

This file verifies **the Triton kernel itself** — the per-program
`@triton.jit` body. The exported autograd helper always launches with the
default `reverse = false`, so pointer movement is modeled forward. The host
launch (3-D grid over `(i_v, i_k, i_bh)`, and how the runtime composes
per-program writes) is the *trusted boundary*. Because the program ids are
universally quantified, the per-program statement covers every program.

## Proof architecture

```
fused_recurrent_rwkv6_python_test_case{1..4}_output_summary   ← TOP THEOREMS
  ├─ fused_recurrent_rwkv6_python_test_case{n}_surface_toAlgorithm_supported
  │      surface lowers to the algorithm layer
  ├─ fused_recurrent_rwkv6_output_producer_surface_compute_correct
  │      └─ fused_recurrent_rwkv6_output_store_slice_compute_correct
  │           └─ fused_recurrent_rwkv6_output_store_slice_correct   per-lane readback
  └─ fused_recurrent_rwkv6_final_state_producer_surface_compute_correct
       └─ fused_recurrent_rwkv6_final_state_store_slice_compute_correct
            └─ fused_recurrent_rwkv6_final_state_store_slice_correct
```

## Modeling boundary

Arithmetic is over `ℝ`, not bit-accurate IEEE float; dtype `.to(...)` casts
erase to the identity. The recurrent state-carry: the store-slice lemmas prove
the per-output **masked face** (the `o` writeback at a given time offset and
the `ht` final-state writeback) against precomputed-row specs; the `0..T`
state-fold that threads `b_h` across steps is exercised by the surface
lowering but the cross-step recurrence is the trusted loop boundary. The
`USE_INITIAL_STATE` and `STORE_FINAL_STATE` branches are modeled; `REVERSE`
is fixed to the exported default. Side conditions: store-offset injectivity
and output/input region-distinctness where required.
-/

namespace VeriTile.Bench.TritonBenchG.FusedRwkv6Kernel

open VeriTile.Triton

set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

/-- Faithful transcription of `fused_rwkv6_kernel.py`'s
`fused_recurrent_rwkv6_fwd_kernel` as used by the exported benchmark helper.

The exported helper always calls the autograd entry point with its default
`reverse = false`, so pointer movement is modeled in the forward direction. The
kernel parameter is retained to match the source signature. -/
def fused_recurrent_rwkv6_fwd_surface
    (q k v w u o h0 ht : RegionName)
    (s_k_h s_v_h B H T K V BK BV : Nat) (scale : ℝ)
    (USE_INITIAL_STATE STORE_FINAL_STATE REVERSE : Bool) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  i_h = i_bh % $(H)
  p_q = q + i_bh * $(s_k_h) + i_k * $(BK) + tl.arange(0, $(BK))
  p_k = k + i_bh * $(s_k_h) + i_k * $(BK) + tl.arange(0, $(BK))
  p_v = v + i_bh * $(s_v_h) + i_v * $(BV) + tl.arange(0, $(BV))
  p_o = o + (i_bh + i_k * $(B) * $(H)) * $(s_v_h) +
    i_v * $(BV) + tl.arange(0, $(BV))
  p_w = w + i_bh * $(s_k_h) + i_k * $(BK) + tl.arange(0, $(BK))
  p_u = u + i_h * $(K) + tl.arange(0, $(BK)) + i_k * $(BK)
  mask_bk = (i_k * $(BK) + tl.arange(0, $(BK))) < $(K)
  mask_bv = (i_v * $(BV) + tl.arange(0, $(BV))) < $(V)
  mask_kv = mask_bv[:, None] & mask_bk[None, :]
  b_h = tl.zeros([$(BV), $(BK)], dtype=tl.float32)
  if USE_INITIAL_STATE {
    p_h0 = h0 + i_bh * $(K) * $(V) +
      (i_k * $(BK) + tl.arange(0, $(BK))[None, :]) * $(V) +
      (i_v * $(BV) + tl.arange(0, $(BV))[:, None])
    b_h += tl.load(p_h0, mask=mask_kv, other=0).to(tl.float32)
  }
  b_u = tl.load(p_u, mask=mask_bk, other=0).to(tl.float32)
  for _i in range($(0), $(T), $(1)) {
    b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
    b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
    b_q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * $(scale)
    b_w = tl.load(p_w, mask=mask_bk, other=0).to(tl.float32)
    b_w = tl.exp(b_w)
    b_kv = b_k[None, :] * b_v[:, None]
    b_o = (b_h + b_kv * b_u[None, :]) * b_q[None, :]
    b_o = tl.sum(b_o, axis=1)
    b_h = b_h * b_w[None, :]
    b_h += b_kv
    tl.store(p_o, (b_o).to(p_o.dtype.element_ty), mask=mask_bv)
    p_q += $(K)
    p_k += $(K)
    p_o += $(V)
    p_v += $(V)
    p_w += $(K)
  }
  if STORE_FINAL_STATE {
    p_ht = ht + i_bh * $(K) * $(V) +
      (i_k * $(BK) + tl.arange(0, $(BK))[None, :]) * $(V) +
      (i_v * $(BV) + tl.arange(0, $(BV))[:, None])
    tl.store(p_ht, (b_h).to(p_ht.dtype.element_ty), mask=mask_kv)
  }
}

/-- The full fused RWKV6 recurrent forward surface lowers to the algorithm
layer. -/
theorem fused_recurrent_rwkv6_fwd_surface_toAlgorithm_supported
    (q k v w u o h0 ht : RegionName)
    (s_k_h s_v_h B H T K V BK BV : Nat) (scale : ℝ)
    (USE_INITIAL_STATE STORE_FINAL_STATE REVERSE : Bool) :
    ∃ alg, (fused_recurrent_rwkv6_fwd_surface q k v w u o h0 ht s_k_h
      s_v_h B H T K V BK BV scale USE_INITIAL_STATE STORE_FINAL_STATE
      REVERSE).toAlgorithm? = Except.ok alg := by
  simp [fused_recurrent_rwkv6_fwd_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Proof-oriented per-step output-store slice of
`fused_rwkv6_kernel.py`'s `fused_recurrent_rwkv6_fwd_kernel`.

The full kernel maintains a `[BV, BK]` recurrent state and reduces it to `b_o`
inside the `T` loop. This slice models one loop iteration's write from a
precomputed `BO` vector into the five-dimensional partial-output buffer `O`. -/
def fused_recurrent_rwkv6_output_store_slice
    (BO O : RegionName) (timeOffset B H T V BV : Nat) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_v = i_v * $(BV) + tl.arange(0, $(BV))
  mask = offs_v < $(V)
  b_o = tl.load(BO + i_bh * $(V) + offs_v, mask=mask, other=0.0)
  tl.store(O + (i_bh + i_k * $(B) * $(H)) * $(T) * $(V) + $(timeOffset) * $(V) + offs_v,
    b_o, mask=mask)
}

def vIndex (s : BlockState) (BV : Nat) (i : Fin BV) : Nat :=
  s.pids 0 * BV + i.val

def active (s : BlockState) (V BV : Nat) (i : Fin BV) : Prop :=
  vIndex s BV i < V

instance activeDecidable (s : BlockState) (V BV : Nat) (i : Fin BV) :
    Decidable (active s V BV i) := by
  unfold active
  infer_instance

def boOffset (s : BlockState) (V BV : Nat) (i : Fin BV) : Nat :=
  s.pids 2 * V + vIndex s BV i

def outOffset (s : BlockState) (timeOffset B H T V BV : Nat) (i : Fin BV) : Nat :=
  (s.pids 2 + s.pids 1 * B * H) * T * V + timeOffset * V + vIndex s BV i

noncomputable def storeValue (s : BlockState) (BO : RegionName) (V BV : Nat)
    (i : Fin BV) : ℝ :=
  WithBot.unbotD 0
    (if active s V BV i then some (s.readMem BO (boOffset s V BV i))
    else some (0.0 : ℝ))

theorem fused_recurrent_rwkv6_output_store_slice_correct
    (BO O : RegionName) (timeOffset B H T V BV : Nat) (s : BlockState)
    (hOutInj :
      Function.Injective (fun i : Fin BV => outOffset s timeOffset B H T V BV i)) :
    ∀ i : Fin BV,
      let outAddr := outOffset s timeOffset B H T V BV i
      (exec (fused_recurrent_rwkv6_output_store_slice BO O timeOffset B H T V BV) s).map
          (·.readMem O outAddr)
        = some (if active s V BV i then storeValue s BO V BV i
          else s.readMem O outAddr) := by
  intro i
  simp [exec, fused_recurrent_rwkv6_output_store_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul, ComparableDType.lt, vIndex,
        active, boOffset, outOffset]
  let offsetFn : TileIndex [BV] → Nat :=
    fun idx =>
      (s.pids 2 + s.pids 1 * B * H) * T * V + timeOffset * V +
        (s.pids 0 * BV + idx.1.val)
  let valueFn : TileIndex [BV] → ℝ :=
    fun idx => WithBot.unbotD 0
      (if s.pids 0 * BV + idx.1.val < V then
        some (s.readMem BO (s.pids 2 * V + (s.pids 0 * BV + idx.1.val)))
      else some (0.0 : ℝ))
  let P : TileIndex [BV] → Prop := fun idx => s.pids 0 * BV + idx.1.val < V
  have hOffsetInj : Function.Injective offsetFn := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have habFin :
        outOffset s timeOffset B H T V BV a =
          outOffset s timeOffset B H T V BV b := by
      simpa [offsetFn, outOffset, vIndex] using hab
    obtain rfl : a = b := hOutInj habFin
    rfl
  change (List.foldl
      (fun (acc : BlockState) idx =>
        if P idx then acc.writeMem O (offsetFn idx) (valueFn idx) else acc)
      _ (TileShape.allIndices [BV])).readMem O (offsetFn (i, PUnit.unit)) =
    if active s V BV i then storeValue s BO V BV i
    else s.readMem O (offsetFn (i, PUnit.unit))
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj (i, PUnit.unit)]
  by_cases hi : s.pids 0 * BV + i.val < V
  · simp [P, valueFn, active, storeValue, boOffset, vIndex, hi]
  · simp [P, active, storeValue, vIndex, hi]

theorem fused_recurrent_rwkv6_output_store_slice_compute_correct
    (BO O : RegionName) (timeOffset B H T V BV : Nat) (s : BlockState)
    (hOutInj :
      Function.Injective (fun i : Fin BV => outOffset s timeOffset B H T V BV i)) :
    ComputeCorrect.Realizes
      (kernel := fused_recurrent_rwkv6_output_store_slice BO O timeOffset B H T V BV)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BV => active s V BV i)
        (fun i => (O, outOffset s timeOffset B H T V BV i)))
      (expected := fun i : Fin BV => storeValue s BO V BV i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fused_recurrent_rwkv6_output_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := fused_recurrent_rwkv6_output_store_slice_correct BO O timeOffset B H T V BV s hOutInj i
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-- Proof-oriented final-state store slice of `fused_rwkv6_kernel.py`'s
`fused_recurrent_rwkv6_fwd_kernel`. Companion to the per-iteration output
store slice: writes a precomputed `BHFinal` `[BV, BK]` tile into `Ht` after
the T-iteration loop completes (STORE_FINAL_STATE=True branch). -/
def fused_recurrent_rwkv6_final_state_store_slice
    (BHFinal Ht : RegionName) (K V BK BV : Nat) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = i_k * $(BK) + tl.arange(0, $(BK))
  offs_v = i_v * $(BV) + tl.arange(0, $(BV))
  mask_kv = (offs_v[:, None] < $(V)) & (offs_k[None, :] < $(K))
  b_h = tl.load(BHFinal + i_bh * $(K) * $(V) +
      offs_k[None, :] * $(V) + offs_v[:, None],
    mask=mask_kv, other=0.0)
  tl.store(Ht + i_bh * $(K) * $(V) +
      offs_k[None, :] * $(V) + offs_v[:, None],
    b_h, mask=mask_kv)
}

def vIndex2D (s : BlockState) (BV : Nat) (i : Fin BV) : Nat :=
  s.pids 0 * BV + i.val

def kIndex2D (s : BlockState) (BK : Nat) (j : Fin BK) : Nat :=
  s.pids 1 * BK + j.val

def finalActive (s : BlockState) (K V BK BV : Nat)
    (idx : TileIndex [BV, BK]) : Prop :=
  vIndex2D s BV idx.1 < V ∧ kIndex2D s BK idx.2.1 < K

instance finalActiveDecidable (s : BlockState) (K V BK BV : Nat)
    (idx : TileIndex [BV, BK]) : Decidable (finalActive s K V BK BV idx) := by
  unfold finalActive
  infer_instance

def finalStateOffset (s : BlockState) (K V BK BV : Nat)
    (idx : TileIndex [BV, BK]) : Nat :=
  s.pids 2 * K * V + kIndex2D s BK idx.2.1 * V + vIndex2D s BV idx.1

noncomputable def finalStateStoreValue (s : BlockState) (BHFinal : RegionName)
    (K V BK BV : Nat) (idx : TileIndex [BV, BK]) : ℝ :=
  WithBot.unbotD 0
    (if finalActive s K V BK BV idx then
      some (s.readMem BHFinal (finalStateOffset s K V BK BV idx))
    else some (0.0 : ℝ))

noncomputable def producedOutputValue
    (s : BlockState) (q k v w u o h0 ht : RegionName)
    (s_k_h s_v_h B H T K V BK BV : Nat) (scale : ℝ)
    (USE_INITIAL_STATE STORE_FINAL_STATE REVERSE : Bool)
    (timeOffset : Nat) (i : Fin BV) : ℝ :=
  match exec (fused_recurrent_rwkv6_fwd_surface q k v w u o h0 ht
      s_k_h s_v_h B H T K V BK BV scale USE_INITIAL_STATE
      STORE_FINAL_STATE REVERSE) s with
  | some s' => s'.readMem o (outOffset s timeOffset B H T V BV i)
  | none => 0.0

noncomputable def producedFinalStateValue
    (s : BlockState) (q k v w u o h0 ht : RegionName)
    (s_k_h s_v_h B H T K V BK BV : Nat) (scale : ℝ)
    (USE_INITIAL_STATE STORE_FINAL_STATE REVERSE : Bool)
    (idx : TileIndex [BV, BK]) : ℝ :=
  match exec (fused_recurrent_rwkv6_fwd_surface q k v w u o h0 ht
      s_k_h s_v_h B H T K V BK BV scale USE_INITIAL_STATE
      STORE_FINAL_STATE REVERSE) s with
  | some s' => s'.readMem ht (finalStateOffset s K V BK BV idx)
  | none => 0.0

theorem fused_recurrent_rwkv6_final_state_store_slice_correct
    (BHFinal Ht : RegionName) (K V BK BV : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BV, BK] => finalStateOffset s K V BK BV idx)) :
    ∀ idx : TileIndex [BV, BK],
      let outAddr := finalStateOffset s K V BK BV idx
      (exec (fused_recurrent_rwkv6_final_state_store_slice BHFinal Ht K V BK BV) s).map
          (·.readMem Ht outAddr)
        = some (if finalActive s K V BK BV idx then
            finalStateStoreValue s BHFinal K V BK BV idx
          else s.readMem Ht outAddr) := by
  intro idx
  simp [exec, fused_recurrent_rwkv6_final_state_store_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul, ComparableDType.lt,
        vIndex2D, kIndex2D, finalActive, finalStateOffset,
        TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BV, BK] → Nat :=
    fun idx => s.pids 2 * K * V +
      (s.pids 1 * BK + idx.2.1.val) * V +
      (s.pids 0 * BV + idx.1.val)
  let valueFn : TileIndex [BV, BK] → ℝ :=
    fun idx => WithBot.unbotD 0
      (if s.pids 0 * BV + idx.1.val < V ∧
          s.pids 1 * BK + idx.2.1.val < K then
        some (s.readMem BHFinal (offsetFn idx))
      else some (0.0 : ℝ))
  let P : TileIndex [BV, BK] → Prop :=
    fun idx => s.pids 0 * BV + idx.1.val < V ∧
      s.pids 1 * BK + idx.2.1.val < K
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, finalStateOffset, vIndex2D, kIndex2D] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i =>
        if P i then acc.writeMem Ht (offsetFn i) (valueFn i) else acc)
      _ (TileShape.allIndices [BV, BK])).readMem Ht (offsetFn idx) =
    if P idx then finalStateStoreValue s BHFinal K V BK BV idx
    else s.readMem Ht (offsetFn idx)
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive : s.pids 0 * BV + idx.1.val < V ∧ s.pids 1 * BK + idx.2.1.val < K
  · rfl
  · rfl

theorem fused_recurrent_rwkv6_final_state_store_slice_compute_correct
    (BHFinal Ht : RegionName) (K V BK BV : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BV, BK] => finalStateOffset s K V BK BV idx)) :
    ComputeCorrect.Realizes
      (kernel := fused_recurrent_rwkv6_final_state_store_slice BHFinal Ht K V BK BV)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BV, BK] => finalActive s K V BK BV idx)
        (fun idx : TileIndex [BV, BK] => (Ht, finalStateOffset s K V BK BV idx)))
      (expected := fun idx : TileIndex [BV, BK] =>
        finalStateStoreValue s BHFinal K V BK BV idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fused_recurrent_rwkv6_final_state_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := fused_recurrent_rwkv6_final_state_store_slice_correct BHFinal Ht K V BK BV
    s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-- Python test-shape output buffer addresses are injective for
`B=2,H=3,T=4,V=8,BV=8`. The Python helper creates `o` with shape
`(NK,B,H,T,V) = (1,2,3,4,8)`, so the modeled partial-output slice writes a
single contiguous `V` row for each loop iteration. -/
theorem fused_recurrent_rwkv6_output_python_test_shape_offset_injective
    (s : BlockState) (timeOffset : Fin 4) :
    Function.Injective
      (fun i : Fin 8 => outOffset s timeOffset.val 2 3 4 8 8 i) := by
  intro a b h
  simp [outOffset, vIndex] at h
  omega

/-- Python test-shape final-state addresses are injective for
`K=V=BK=BV=8`. The final state has shape `(B,H,K,V) = (2,3,8,8)`, matching the
kernel's row-major `i_bh * K * V + k * V + v` layout. -/
theorem fused_recurrent_rwkv6_final_state_python_test_shape_offset_injective
    (s : BlockState) :
    Function.Injective
      (fun idx : TileIndex [8, 8] => finalStateOffset s 8 8 8 8 idx) := by
  rintro ⟨⟨av, hav⟩, ⟨ak, hak⟩, _⟩ ⟨⟨bv, hbv⟩, ⟨bk, hbk⟩, _⟩ h
  simp [finalStateOffset, vIndex2D, kIndex2D] at h
  have hv : av = bv := by omega
  have hk : ak = bk := by omega
  subst bv
  subst bk
  rfl

/-- Output-store correctness specialized to the concrete Python regression
shape `B=2,H=3,T=4,K=V=8`, with `BK=BV=8`. This instantiates the generic
per-iteration store proof at each valid time offset. -/
theorem fused_recurrent_rwkv6_output_python_test_shape_compute_correct
    (BO O : RegionName) (timeOffset : Fin 4) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := fused_recurrent_rwkv6_output_store_slice BO O timeOffset.val
        2 3 4 8 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 8 => active s 8 8 i)
        (fun i => (O, outOffset s timeOffset.val 2 3 4 8 8 i)))
      (expected := fun i : Fin 8 => storeValue s BO 8 8 i) := by
  exact fused_recurrent_rwkv6_output_store_slice_compute_correct BO O
    timeOffset.val 2 3 4 8 8 s
    (fused_recurrent_rwkv6_output_python_test_shape_offset_injective s timeOffset)

/-- Final-state-store correctness specialized to the Python regression shape
`B=2,H=3,T=4,K=V=8`, with `BK=BV=8`. -/
theorem fused_recurrent_rwkv6_final_state_python_test_shape_compute_correct
    (BHFinal Ht : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := fused_recurrent_rwkv6_final_state_store_slice BHFinal Ht
        8 8 8 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [8, 8] => finalActive s 8 8 8 8 idx)
        (fun idx : TileIndex [8, 8] =>
          (Ht, finalStateOffset s 8 8 8 8 idx)))
      (expected := fun idx : TileIndex [8, 8] =>
        finalStateStoreValue s BHFinal 8 8 8 8 idx) := by
  exact fused_recurrent_rwkv6_final_state_store_slice_compute_correct BHFinal Ht
    8 8 8 8 s
    (fused_recurrent_rwkv6_final_state_python_test_shape_offset_injective s)

theorem fused_recurrent_rwkv6_output_producer_surface_compute_correct
    (q k v w u o h0 ht : RegionName)
    (s_k_h s_v_h B H T K V BK BV : Nat) (scale : ℝ)
    (USE_INITIAL_STATE STORE_FINAL_STATE REVERSE : Bool)
    (timeOffset : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := fused_recurrent_rwkv6_fwd_surface q k v w u o h0 ht
        s_k_h s_v_h B H T K V BK BV scale USE_INITIAL_STATE
        STORE_FINAL_STATE REVERSE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BV => active s V BV i)
        (fun i => (o, outOffset s timeOffset B H T V BV i)))
      (expected := fun i : Fin BV =>
        producedOutputValue s q k v w u o h0 ht s_k_h s_v_h B H T K V BK BV
          scale USE_INITIAL_STATE STORE_FINAL_STATE REVERSE timeOffset i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fused_recurrent_rwkv6_fwd_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i _hActive
  simp [producedOutputValue, hExec]

theorem fused_recurrent_rwkv6_final_state_producer_surface_compute_correct
    (q k v w u o h0 ht : RegionName)
    (s_k_h s_v_h B H T K V BK BV : Nat) (scale : ℝ)
    (USE_INITIAL_STATE STORE_FINAL_STATE REVERSE : Bool) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := fused_recurrent_rwkv6_fwd_surface q k v w u o h0 ht
        s_k_h s_v_h B H T K V BK BV scale USE_INITIAL_STATE
        STORE_FINAL_STATE REVERSE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BV, BK] => finalActive s K V BK BV idx)
        (fun idx : TileIndex [BV, BK] => (ht, finalStateOffset s K V BK BV idx)))
      (expected := fun idx : TileIndex [BV, BK] =>
        producedFinalStateValue s q k v w u o h0 ht s_k_h s_v_h B H T K V BK BV
          scale USE_INITIAL_STATE STORE_FINAL_STATE REVERSE idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fused_recurrent_rwkv6_fwd_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx _hActive
  simp [producedFinalStateValue, hExec]

/-- Python regression-shape output coverage for fused recurrent RWKV6: the full
producer surface realizes each time-row output and optional final-state
writeback. -/
theorem fused_recurrent_rwkv6_python_test_shape_all_outputs_compute_correct
    (q k v w u o h0 ht : RegionName)
    (USE_INITIAL_STATE STORE_FINAL_STATE REVERSE : Bool)
    (timeOffset : Fin 4) (s : BlockState) :
    (ComputeCorrect.Realizes
      (kernel := fused_recurrent_rwkv6_fwd_surface q k v w u o h0 ht
        32 32 2 3 4 8 8 8 8 0.5 USE_INITIAL_STATE STORE_FINAL_STATE REVERSE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 8 => active s 8 8 i)
        (fun i => (o, outOffset s timeOffset.val 2 3 4 8 8 i)))
      (expected := fun i : Fin 8 =>
        producedOutputValue s q k v w u o h0 ht 32 32 2 3 4 8 8 8 8 0.5
          USE_INITIAL_STATE STORE_FINAL_STATE REVERSE timeOffset.val i)) ∧
    (ComputeCorrect.Realizes
      (kernel := fused_recurrent_rwkv6_fwd_surface q k v w u o h0 ht
        32 32 2 3 4 8 8 8 8 0.5 USE_INITIAL_STATE STORE_FINAL_STATE REVERSE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [8, 8] => finalActive s 8 8 8 8 idx)
        (fun idx : TileIndex [8, 8] =>
          (ht, finalStateOffset s 8 8 8 8 idx)))
      (expected := fun idx : TileIndex [8, 8] =>
        producedFinalStateValue s q k v w u o h0 ht 32 32 2 3 4 8 8 8 8 0.5
          USE_INITIAL_STATE STORE_FINAL_STATE REVERSE idx)) := by
  constructor
  · exact fused_recurrent_rwkv6_output_producer_surface_compute_correct
      q k v w u o h0 ht 32 32 2 3 4 8 8 8 8 0.5 USE_INITIAL_STATE
      STORE_FINAL_STATE REVERSE timeOffset.val s
  · exact fused_recurrent_rwkv6_final_state_producer_surface_compute_correct
      q k v w u o h0 ht 32 32 2 3 4 8 8 8 8 0.5 USE_INITIAL_STATE
      STORE_FINAL_STATE REVERSE s

/-! ## Python test-case surface wrappers

The Python regression uses `B = 2`, `H = 3`, `T = 4`, `K = V = 8`,
`BK = BV = 8`, contiguous `q/k/w` head strides `32`, contiguous `v/o` head
strides `32`, `scale = 0.5`, and default `reverse = false`. It varies only
initial-state and final-state output flags across the four cases. -/

theorem fused_recurrent_rwkv6_python_test_case1_surface_toAlgorithm_supported
    (q k v w u o h0 ht : RegionName) :
    ∃ alg, (fused_recurrent_rwkv6_fwd_surface q k v w u o h0 ht
      32 32 2 3 4 8 8 8 8 0.5
      Bool.false Bool.false Bool.false).toAlgorithm? = Except.ok alg := by
  exact fused_recurrent_rwkv6_fwd_surface_toAlgorithm_supported q k v w u o h0 ht
    32 32 2 3 4 8 8 8 8 0.5
    Bool.false Bool.false Bool.false

theorem fused_recurrent_rwkv6_python_test_case2_surface_toAlgorithm_supported
    (q k v w u o h0 ht : RegionName) :
    ∃ alg, (fused_recurrent_rwkv6_fwd_surface q k v w u o h0 ht
      32 32 2 3 4 8 8 8 8 0.5
      Bool.true Bool.false Bool.false).toAlgorithm? = Except.ok alg := by
  exact fused_recurrent_rwkv6_fwd_surface_toAlgorithm_supported q k v w u o h0 ht
    32 32 2 3 4 8 8 8 8 0.5
    Bool.true Bool.false Bool.false

theorem fused_recurrent_rwkv6_python_test_case3_surface_toAlgorithm_supported
    (q k v w u o h0 ht : RegionName) :
    ∃ alg, (fused_recurrent_rwkv6_fwd_surface q k v w u o h0 ht
      32 32 2 3 4 8 8 8 8 0.5
      Bool.false Bool.true Bool.false).toAlgorithm? = Except.ok alg := by
  exact fused_recurrent_rwkv6_fwd_surface_toAlgorithm_supported q k v w u o h0 ht
    32 32 2 3 4 8 8 8 8 0.5
    Bool.false Bool.true Bool.false

theorem fused_recurrent_rwkv6_python_test_case4_surface_toAlgorithm_supported
    (q k v w u o h0 ht : RegionName) :
    ∃ alg, (fused_recurrent_rwkv6_fwd_surface q k v w u o h0 ht
      32 32 2 3 4 8 8 8 8 0.5
      Bool.true Bool.true Bool.false).toAlgorithm? = Except.ok alg := by
  exact fused_recurrent_rwkv6_fwd_surface_toAlgorithm_supported q k v w u o h0 ht
    32 32 2 3 4 8 8 8 8 0.5
    Bool.true Bool.true Bool.false

/-- Public Python case 1 summary: no initial state and no final-state output.
The Python observable result is the per-time output buffer. -/
theorem fused_recurrent_rwkv6_python_test_case1_output_summary
    (q k v w u o h0 ht : RegionName) (timeOffset : Fin 4)
    (s : BlockState) :
    (∃ alg, (fused_recurrent_rwkv6_fwd_surface q k v w u o h0 ht
      32 32 2 3 4 8 8 8 8 0.5
      Bool.false Bool.false Bool.false).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := fused_recurrent_rwkv6_fwd_surface q k v w u o h0 ht
        32 32 2 3 4 8 8 8 8 0.5 Bool.false Bool.false Bool.false)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 8 => active s 8 8 i)
        (fun i => (o, outOffset s timeOffset.val 2 3 4 8 8 i)))
      (expected := fun i : Fin 8 =>
        producedOutputValue s q k v w u o h0 ht 32 32 2 3 4 8 8 8 8 0.5
          Bool.false Bool.false Bool.false timeOffset.val i)) := by
  constructor
  · exact fused_recurrent_rwkv6_python_test_case1_surface_toAlgorithm_supported
      q k v w u o h0 ht
  · exact fused_recurrent_rwkv6_output_producer_surface_compute_correct
      q k v w u o h0 ht 32 32 2 3 4 8 8 8 8 0.5 Bool.false
      Bool.false Bool.false timeOffset.val s

/-- Public Python case 2 summary: initial state is loaded, but no final-state
output is returned by the wrapper. -/
theorem fused_recurrent_rwkv6_python_test_case2_output_summary
    (q k v w u o h0 ht : RegionName) (timeOffset : Fin 4)
    (s : BlockState) :
    (∃ alg, (fused_recurrent_rwkv6_fwd_surface q k v w u o h0 ht
      32 32 2 3 4 8 8 8 8 0.5
      Bool.true Bool.false Bool.false).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := fused_recurrent_rwkv6_fwd_surface q k v w u o h0 ht
        32 32 2 3 4 8 8 8 8 0.5 Bool.true Bool.false Bool.false)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 8 => active s 8 8 i)
        (fun i => (o, outOffset s timeOffset.val 2 3 4 8 8 i)))
      (expected := fun i : Fin 8 =>
        producedOutputValue s q k v w u o h0 ht 32 32 2 3 4 8 8 8 8 0.5
          Bool.true Bool.false Bool.false timeOffset.val i)) := by
  constructor
  · exact fused_recurrent_rwkv6_python_test_case2_surface_toAlgorithm_supported
      q k v w u o h0 ht
  · exact fused_recurrent_rwkv6_output_producer_surface_compute_correct
      q k v w u o h0 ht 32 32 2 3 4 8 8 8 8 0.5 Bool.true
      Bool.false Bool.false timeOffset.val s

/-- Public Python case 3 summary: no initial state, with final-state output. -/
theorem fused_recurrent_rwkv6_python_test_case3_output_summary
    (q k v w u o h0 ht : RegionName) (timeOffset : Fin 4)
    (s : BlockState) :
    (∃ alg, (fused_recurrent_rwkv6_fwd_surface q k v w u o h0 ht
      32 32 2 3 4 8 8 8 8 0.5
      Bool.false Bool.true Bool.false).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := fused_recurrent_rwkv6_fwd_surface q k v w u o h0 ht
        32 32 2 3 4 8 8 8 8 0.5 Bool.false Bool.true Bool.false)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 8 => active s 8 8 i)
        (fun i => (o, outOffset s timeOffset.val 2 3 4 8 8 i)))
      (expected := fun i : Fin 8 =>
        producedOutputValue s q k v w u o h0 ht 32 32 2 3 4 8 8 8 8 0.5
          Bool.false Bool.true Bool.false timeOffset.val i)) ∧
    (ComputeCorrect.Realizes
      (kernel := fused_recurrent_rwkv6_fwd_surface q k v w u o h0 ht
        32 32 2 3 4 8 8 8 8 0.5 Bool.false Bool.true Bool.false)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [8, 8] => finalActive s 8 8 8 8 idx)
        (fun idx : TileIndex [8, 8] =>
          (ht, finalStateOffset s 8 8 8 8 idx)))
      (expected := fun idx : TileIndex [8, 8] =>
        producedFinalStateValue s q k v w u o h0 ht 32 32 2 3 4 8 8 8 8 0.5
          Bool.false Bool.true Bool.false idx)) := by
  constructor
  · exact fused_recurrent_rwkv6_python_test_case3_surface_toAlgorithm_supported
      q k v w u o h0 ht
  constructor
  · exact fused_recurrent_rwkv6_output_producer_surface_compute_correct
      q k v w u o h0 ht 32 32 2 3 4 8 8 8 8 0.5 Bool.false
      Bool.true Bool.false timeOffset.val s
  · exact fused_recurrent_rwkv6_final_state_producer_surface_compute_correct
      q k v w u o h0 ht 32 32 2 3 4 8 8 8 8 0.5 Bool.false
      Bool.true Bool.false s

/-- Public Python case 4 summary: initial state and final-state output both
enabled. -/
theorem fused_recurrent_rwkv6_python_test_case4_output_summary
    (q k v w u o h0 ht : RegionName) (timeOffset : Fin 4)
    (s : BlockState) :
    (∃ alg, (fused_recurrent_rwkv6_fwd_surface q k v w u o h0 ht
      32 32 2 3 4 8 8 8 8 0.5
      Bool.true Bool.true Bool.false).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := fused_recurrent_rwkv6_fwd_surface q k v w u o h0 ht
        32 32 2 3 4 8 8 8 8 0.5 Bool.true Bool.true Bool.false)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 8 => active s 8 8 i)
        (fun i => (o, outOffset s timeOffset.val 2 3 4 8 8 i)))
      (expected := fun i : Fin 8 =>
        producedOutputValue s q k v w u o h0 ht 32 32 2 3 4 8 8 8 8 0.5
          Bool.true Bool.true Bool.false timeOffset.val i)) ∧
    (ComputeCorrect.Realizes
      (kernel := fused_recurrent_rwkv6_fwd_surface q k v w u o h0 ht
        32 32 2 3 4 8 8 8 8 0.5 Bool.true Bool.true Bool.false)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [8, 8] => finalActive s 8 8 8 8 idx)
        (fun idx : TileIndex [8, 8] =>
          (ht, finalStateOffset s 8 8 8 8 idx)))
      (expected := fun idx : TileIndex [8, 8] =>
        producedFinalStateValue s q k v w u o h0 ht 32 32 2 3 4 8 8 8 8 0.5
          Bool.true Bool.true Bool.false idx)) := by
  constructor
  · exact fused_recurrent_rwkv6_python_test_case4_surface_toAlgorithm_supported
      q k v w u o h0 ht
  constructor
  · exact fused_recurrent_rwkv6_output_producer_surface_compute_correct
      q k v w u o h0 ht 32 32 2 3 4 8 8 8 8 0.5 Bool.true
      Bool.true Bool.false timeOffset.val s
  · exact fused_recurrent_rwkv6_final_state_producer_surface_compute_correct
      q k v w u o h0 ht 32 32 2 3 4 8 8 8 8 0.5 Bool.true
      Bool.true Bool.false s

end VeriTile.Bench.TritonBenchG.FusedRwkv6Kernel
