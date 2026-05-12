import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.FusedRwkv6Kernel

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Surface transcription of the `REVERSE = false` path of
`fused_rwkv6_kernel.py`'s `fused_recurrent_rwkv6_fwd_kernel`.

The public benchmark calls the wrapper with `reverse = false`. The reverse path
updates pointers by `-K`/`-V`, which requires signed pointer movement and is not
encoded here. The source fp32 load casts and final destination dtype casts are
represented explicitly. -/
def fused_recurrent_rwkv6_fwd_surface
    (Q KReg VReg W U O H0 HT : RegionName)
    (s_k_h s_v_h B H T K V BK BV : Nat) (scale : ℝ)
    (USE_INITIAL_STATE STORE_FINAL_STATE : Bool) :
    ComputeKernel := triton {
  i_v = tl.program_id(axis=0)
  i_k = tl.program_id(axis=1)
  i_bh = tl.program_id(axis=2)
  i_h = i_bh % $(H)
  offs_k = i_k * $(BK) + tl.arange(0, $(BK))
  offs_v = i_v * $(BV) + tl.arange(0, $(BV))
  p_q = Q + i_bh * $(s_k_h) + i_k * $(BK) + tl.arange(0, $(BK))
  p_k = KReg + i_bh * $(s_k_h) + i_k * $(BK) + tl.arange(0, $(BK))
  p_v = VReg + i_bh * $(s_v_h) + i_v * $(BV) + tl.arange(0, $(BV))
  p_o = O + (i_bh + i_k * $(B) * $(H)) * $(s_v_h) +
    i_v * $(BV) + tl.arange(0, $(BV))
  p_w = W + i_bh * $(s_k_h) + i_k * $(BK) + tl.arange(0, $(BK))
  p_u = U + i_h * $(K) + tl.arange(0, $(BK)) + i_k * $(BK)
  mask_bk = offs_k < $(K)
  mask_bv = offs_v < $(V)
  mask_kv = mask_bv[:, None] & mask_bk[None, :]
  b_h = tl.zeros([$(BV), $(BK)], dtype=tl.float32)
  if USE_INITIAL_STATE {
    p_h0 = H0 + i_bh * $(K) * $(V) +
      offs_k[None, :] * $(V) + offs_v[:, None]
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
    tl.store(p_o, (b_o).to(O.dtype.element_ty), mask=mask_bv)
    p_q += $(K)
    p_k += $(K)
    p_o += $(V)
    p_v += $(V)
    p_w += $(K)
  }
  if STORE_FINAL_STATE {
    p_ht = HT + i_bh * $(K) * $(V) +
      offs_k[None, :] * $(V) + offs_v[:, None]
    tl.store(p_ht, (b_h).to(HT.dtype.element_ty), mask=mask_kv)
  }
}

/-- Proof-oriented per-step output-store slice of
`fused_rwkv6_kernel.py`'s `fused_recurrent_rwkv6_fwd_kernel`.

The full kernel maintains a `[BV, BK]` recurrent state and reduces it to `b_o`
inside the `T` loop. This slice models one loop iteration's write from a
precomputed `BO` vector into the five-dimensional partial-output buffer `O`. -/
def fused_recurrent_rwkv6_output_store_slice
    (BO O : RegionName) (timeOffset B H T V BV : Nat) :
    ComputeKernel := triton {
  i_v = tl.program_id(axis=0)
  i_k = tl.program_id(axis=1)
  i_bh = tl.program_id(axis=2)
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
        evalOp, Option.bind, Option.map, Tile.bop, Tile.ptrAdd,
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

end VeriTile.Bench.TritonBenchG.FusedRwkv6Kernel
