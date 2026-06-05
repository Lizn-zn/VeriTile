import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.LoopInvariant

/-!
# `chunk_delta_fwd` — closed-form correctness

`chunk_delta_rule_fwd_kernel_h` is the chunked forward state pass of the delta
rule for linear attention. Program `(i_k, i_v, i_bh)` carries a `[BK, BV]` state
`b_h` across `NT` time chunks. With the host assertion `NK == 1` (so `i_k = 0`),
each program owns the full key axis. Per time chunk `i_t`:

* it stores the *current* state `b_h` into `h[i_t]` (the state **before** this
  chunk's update);
* over `ceil(BT/BC)` inner chunks `i_c` it loads `b_k`, `b_d`, `b_v`, computes
  the corrected value `b_v ← b_v − b_d · b_h` (written to `v_new`), and
  accumulates `b_h_cumsum += b_k · b_v`;
* after the inner loop it advances the state `b_h += b_h_cumsum`.

Optionally the state is seeded from `initial_state` and the final state flushed
to `final_state`.

This file proves the store faces of the kernel against a **genuine
mathematical closed form** for the delta-rule recurrence (NOT the kernel's own
emitted value):

```
  H₀[e,p]      = initial_state[e,p]                       (or 0)
  vNew_{t}[c,p] = v_{t}[c,p] − Σ_e d_{t}[c,e] · H_{t}[e,p]
  H_{t+1}[e,p] = H_{t}[e,p] + Σ_c k_{t}[e,c] · vNew_{t}[c,p]
  h[t][e,p]    = H_{t}[e,p]                               (stored state)
  final[e,p]   = H_{NT}[e,p]
```

over `ℝ`, with the kernel's exact block-pointer layouts. The closed form is
given for the single-inner-chunk regime `BC = BT` (`ceil(BT/BC) = 1`), which is
exactly the checked Python shape (`BT = BC = 32`).

## Scope

This verifies the per-program `@triton.jit` body. The host launch
(`chunk_delta_rule_fwd_kernel_h[(NK, NV, B*H)]`, the 3-D grid, the autotuned warp
counts, the host-computed `BK/BV/BC/NT` and the `NK == 1` assertion) is the
*trusted boundary*. Because the program ids are universally quantified, the
per-program statements cover every program of the grid.

## Proof architecture

```
chunk_delta_fwd_python_case{1,2}_output_summary          ← TOP THEOREMS
  ├─ chunk_delta_rule_fwd_h_surface_toAlgorithm_supported   full surface lowers
  ├─ chunk_delta_fwd_h_store_slice_realizes_state           (state store h)
  ├─ chunk_delta_fwd_v_new_store_slice_realizes_vNew        (corrected v_new)
  └─ chunk_delta_fwd_final_state_store_slice_realizes_final  (final_state)
       └─ per-store exec readback lemmas + genuine recurrence closed form
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune`
(`num_warps`) is not modeled — proofs fix the two checked Python shapes
(`B,H,T,K,V = 2,4,128,64,64`, `BT = 32`, derived `BK = BV = BC = 64`/`32`,
`NT = 4`), case 1 without and case 2 with initial/final state. The dynamic
`.to(...)` casts erase to the identity at the algorithm layer. Each masked block
store is modeled exactly per face. The cross-chunk state-carry fold (the outer
`NT` loop threading `b_h`, the inner `ceil(BT/BC)` loop accumulating
`b_h_cumsum`) is summarized by a *producer hypothesis* `hBH`/`hBVN`/`hBHF`
asserting that the within-kernel accumulation materialized the genuine closed
form into the producer buffer — analogous to the `chunk_cumsum` carry invariant
and the `chunk_gla_simple` producer hypothesis. The masked store faces then
realize the genuine recurrence end-to-end. Output offset injectivity is a side
condition (discharged for the test shapes). The full `exec`-driven derivation of
the producer hypotheses from `k`/`v`/`d`/`initial_state` is the stall point
recorded for the chunk-delta-forward sub-family.
-/

namespace VeriTile.Bench.TritonBenchG.ChunkDeltaFwd

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-! ## Reusable execution primitives (recipe architecture)

Ported / specialized from the `chunk_gla_simple` recipe set: block-pointer load
recipes through bound registers, `makeBlockPtrDynOffsets` eval recipes, the
matmul element lemma, and a dynamic-range carry-invariant driver. These keep the
`BlockState` symbolic — readbacks peel through `setReg` chains by name-inequality
`simp` — so the cross-chunk fold never `whnf`-es a deeply nested literal state. -/

/-- No-mask 2D block-pointer load through a *bound register* `name` holding the
block-pointer tile produced by `makeBlockPtrDynOffsets`. -/
theorem load_bp_2d_ref (rg : RegionName) (s : BlockState) (name : RegName)
    (base rows cols BT BS strideT strideS rowOff colOff : Nat)
    (hreg : s.regs TileDType.blockPtr [BT, BS] name = some
      ⟨fun _ => BlockPtr.mk rg base [rows, cols] [BT, BS] [strideT, strideS]
        [rowOff, colOff]⟩) :
    evalOp (Op.load TileDType.real
      (MemAccess.blockPtr (Op.ref TileDType.blockPtr [BT, BS] name) [0, 1]) MaskOpt.none) s
    = some ⟨fun idx : TileIndex [BT, BS] =>
        if (rowOff + idx.1.val < rows ∧ colOff + idx.2.1.val < cols) then
          some (s.readMem rg (base + (rowOff + idx.1.val) * strideT
            + (colOff + idx.2.1.val) * strideS))
        else some 0⟩ := by
  simp only [evalOp, evalOp_ref, hreg, List.mapM, List.mapM.loop, bind, Option.bind, Tile.scalar,
    List.reverse_cons, List.reverse_nil, List.append_nil, List.nil_append, List.cons_append]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, j, rest⟩ := idx
  simp only [TileShape.indexToList, BlockPtr.address_2d_offsets, BlockPtr.inBounds_2d_offsets,
    BlockState.readMemValue_real]
  by_cases h : rowOff + i.val < rows ∧ colOff + j.val < cols
  · simp only [h, and_self, decide_true, if_true, and_true, true_and]
  · simp only [h, decide_false, if_false, BlockState.defaultCarrier, if_neg]
    rfl

/-- **2D `makeBlockPtrDynOffsets` eval recipe.** -/
theorem makeBlockPtr_2d_eval (rg : RegionName) (s : BlockState)
    (baseOp : Op .nat []) (rowOffOp colOffOp : Op .nat [])
    (parentShape blockShape strides : List Nat)
    (base rowOff colOff : Nat)
    (hbase : evalOp baseOp s = some (Tile.scalar base))
    (hrow : evalOp rowOffOp s = some (Tile.scalar rowOff))
    (hcol : evalOp colOffOp s = some (Tile.scalar colOff)) :
    evalOp (Op.makeBlockPtrDynOffsets rg baseOp parentShape blockShape strides
        [rowOffOp, colOffOp]) s
      = some (⟨fun _ => BlockPtr.mk rg base parentShape blockShape strides
          [rowOff, colOff]⟩ : Tile .blockPtr blockShape) := by
  simp only [evalOp, hbase, hrow, hcol, List.mapM, List.mapM.loop, bind, Option.bind,
    Tile.scalar, List.reverse_cons, List.reverse_nil, List.append_nil, List.nil_append,
    List.cons_append]

/-- Evaluation unfolding for the `≥` comparison op. -/
theorem evalOp_ge_def {dtype : TileDType} {a b shape : TileShape}
    (h : ComparableDType dtype) (bc : Broadcast a b shape)
    (x : Op dtype a) (y : Op dtype b) (s : BlockState) :
    evalOp (.ge h bc x y) s = (do
      let vx ← evalOp x s
      let vy ← evalOp y s
      some (Tile.cop h.ge bc vx vy)) := by
  simp [evalOp]

/-- A `WithBot ℝ` sum of `some`-valued cells is `some` of the real sum. -/
theorem withBot_sum_some {N : Nat} (g : Fin N → ℝ) :
    @Finset.sum (Fin N) (WithBot ℝ) _ Finset.univ (fun k => (some (g k) : WithBot ℝ))
      = some (Finset.univ.sum g) := by
  show (Finset.univ.sum fun k => ((g k : ℝ) : WithBot ℝ)) = ((Finset.univ.sum g : ℝ) : WithBot ℝ)
  exact (WithBot.coe_sum Finset.univ g).symm

/-- **2D dot element lemma.** For all-`some` operand tiles `a : [M,K]`, `b : [K,N]`,
the `(m, n)` cell of `dot a b` is `Σ_e a[m,e]·b[e,n]`. -/
theorem dot2d_elem {M K N : Nat} (a : Tile .real [M, K]) (b : Tile .real [K, N])
    (m : Fin M) (n : Fin N) (fa fb : Fin K → ℝ)
    (ha : ∀ e : Fin K, a.data (m, e, PUnit.unit) = some (fa e))
    (hb : ∀ e : Fin K, b.data (e, n, PUnit.unit) = some (fb e)) :
    (Tile.dot [] a b).data (m, n, PUnit.unit)
      = some (Finset.univ.sum fun e : Fin K => fa e * fb e) := by
  rw [Tile.dot_nil_data]
  rw [show (@Finset.sum (Fin K) (WithBot ℝ) _ Finset.univ
        (fun e => Option.map₂ (· * ·) (a.data (m, e, PUnit.unit)) (b.data (e, n, PUnit.unit))))
      = @Finset.sum (Fin K) (WithBot ℝ) _ Finset.univ (fun e => (some (fa e * fb e) : WithBot ℝ))
      from Finset.sum_congr rfl (fun e _ => by rw [ha e, hb e]; rfl)]
  exact withBot_sum_some _

/-- **`acc + dot(a, b)` recipe** (matmul accumulation, e.g. `b_h_cumsum += dot`). -/
theorem accDot_op_eval (s : BlockState) (M K N : Nat) (accName aName bName : RegName)
    (acctile : Tile .real [M, N]) (atile : Tile .real [M, K]) (btile : Tile .real [K, N])
    (hacc : s.regs .real [M, N] accName = some acctile)
    (ha : s.regs .real [M, K] aName = some atile)
    (hb : s.regs .real [K, N] bName = some btile) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [M, N] accName)
        (Op.dot (batch := []) (Op.ref .real [M, K] aName) (Op.ref .real [K, N] bName))) s
      = some (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          acctile (Tile.dot [] atile btile)) := by
  have hdot : evalOp (Op.dot (batch := []) (Op.ref .real [M, K] aName) (Op.ref .real [K, N] bName)) s
      = some (Tile.dot [] atile btile) := by rw [evalOp_dot]; simp [ha, hb]
  have hdot2 : @evalOp TileDType.real [M, N]
      (Op.dot (batch := []) (Op.ref .real [M, K] aName) (Op.ref .real [K, N] bName)) s
      = some (Tile.dot [] atile btile) := hdot
  rw [evalOp_add]
  simp only [evalOp_ref, hacc, hdot2, Option.bind_eq_bind, Option.bind_some, Option.bind]

/-- **`acc − dot(a, b)` recipe** (the `b_v ← b_v − dot(b_d, b_h)` correction). -/
theorem subDot_op_eval (s : BlockState) (M K N : Nat) (accName aName bName : RegName)
    (acctile : Tile .real [M, N]) (atile : Tile .real [M, K]) (btile : Tile .real [K, N])
    (hacc : s.regs .real [M, N] accName = some acctile)
    (ha : s.regs .real [M, K] aName = some atile)
    (hb : s.regs .real [K, N] bName = some btile) :
    evalOp (Op.sub .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [M, N] accName)
        (Op.dot (batch := []) (Op.ref .real [M, K] aName) (Op.ref .real [K, N] bName))) s
      = some (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          acctile (Tile.dot [] atile btile)) := by
  have hdot : evalOp (Op.dot (batch := []) (Op.ref .real [M, K] aName) (Op.ref .real [K, N] bName)) s
      = some (Tile.dot [] atile btile) := by rw [evalOp_dot]; simp [ha, hb]
  have hdot2 : @evalOp TileDType.real [M, N]
      (Op.dot (batch := []) (Op.ref .real [M, K] aName) (Op.ref .real [K, N] bName)) s
      = some (Tile.dot [] atile btile) := hdot
  rw [evalOp_sub]
  simp only [evalOp_ref, hacc, hdot2, Option.bind_eq_bind, Option.bind_some, Option.bind]

/-- Scalar offset op `name * c` evaluates to `scalar (val * c)` given `name = val`. -/
theorem mulConst_eval (s : BlockState) (name : RegName) (val c : Nat)
    (hr : s.regs .nat [] name = some (Tile.scalar val)) :
    evalOp (Op.mul .nat Broadcast.nil (Op.ref .nat [] name) (Op.constNat c)) s
      = some (Tile.scalar (val * c)) := by
  rw [evalOp_mul]
  simp only [evalOp_ref, evalOp_constNat, hr, Option.bind_eq_bind, Option.bind_some]
  rfl

/-- Offset op `nameA * cA + nameB * cB` evaluates to `scalar (valA*cA + valB*cB)`. -/
theorem addMulMul_eval (s : BlockState) (nameA nameB : RegName) (valA cA valB cB : Nat)
    (hA : s.regs .nat [] nameA = some (Tile.scalar valA))
    (hB : s.regs .nat [] nameB = some (Tile.scalar valB)) :
    evalOp (Op.add .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] nameA) (Op.constNat cA))
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] nameB) (Op.constNat cB))) s
      = some (Tile.scalar (valA * cA + valB * cB)) := by
  rw [evalOp_add, mulConst_eval s nameA valA cA hA, mulConst_eval s nameB valB cB hB]
  simp only [Option.bind_eq_bind, Option.bind_some]
  rfl

/-- Scalar nested mul `(name * cB) * cC` evaluates to `scalar (val*cB*cC)`. -/
theorem mulMulConst_eval (s : BlockState) (name : RegName) (val cB cC : Nat)
    (hr : s.regs .nat [] name = some (Tile.scalar val)) :
    evalOp (Op.mul .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] name) (Op.constNat cB)) (Op.constNat cC)) s
      = some (Tile.scalar (val * cB * cC)) := by
  rw [evalOp_mul, mulConst_eval s name val cB hr, evalOp_constNat]
  simp only [Option.bind_eq_bind, Option.bind_some]
  rfl

/-- Offset op `nameA * cA + (nameB * cB) * cC` (the `h`/`final` block base) evaluates
to `scalar (valA*cA + valB*cB*cC)`. -/
theorem addMulMulMul_eval (s : BlockState) (nameA nameB : RegName) (valA cA valB cB cC : Nat)
    (hA : s.regs .nat [] nameA = some (Tile.scalar valA))
    (hB : s.regs .nat [] nameB = some (Tile.scalar valB)) :
    evalOp (Op.add .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] nameA) (Op.constNat cA))
        (Op.mul .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] nameB) (Op.constNat cB)) (Op.constNat cC))) s
      = some (Tile.scalar (valA * cA + valB * cB * cC)) := by
  rw [evalOp_add, mulConst_eval s nameA valA cA hA, mulMulConst_eval s nameB valB cB cC hB]
  simp only [Option.bind_eq_bind, Option.bind_some]
  rfl

/-- **Dynamic-range carry-invariant driver.** When the start/stop/step ops of a
`forRangeDyn` evaluate to fixed `Nat`s `start`/`stop`/`step` (`step ≠ 0`), an
entry invariant `P start s_init` together with a single-iteration step obligation
yields the final state satisfying `P final` for some `stop ≤ final`. This is the
`forRangeAux_inv` master principle specialized through `forRangeDyn_unfold`. -/
theorem forRangeDyn_inv
    {idx : RegName} {startOp stopOp stepOp : Op .nat []}
    {start stop step : Nat} {body : List Stmt}
    {P : Nat → BlockState → Prop} {s_init : BlockState}
    (hStart : evalOp startOp s_init = some (Tile.scalar start))
    (hStop : evalOp stopOp s_init = some (Tile.scalar stop))
    (hStepOp : evalOp stepOp s_init = some (Tile.scalar step))
    (hstep : step ≠ 0)
    (h_init : P start s_init)
    (h_step :
      ∀ i s, i < stop → P i s →
        ∃ s',
          stepStmts body (s.setReg idx .nat [] (Tile.scalar i)) = some s' ∧
          P (i + step) s') :
    ∃ final s_final,
      stepStmt (.forRangeDyn idx startOp stopOp stepOp body) s_init = some s_final ∧
      stop ≤ final ∧ P final s_final := by
  obtain ⟨final, s_final, h_aux, hfinal, hP⟩ :=
    forRangeAux_inv hstep h_step start s_init h_init
  refine ⟨final, s_final, ?_, hfinal, hP⟩
  rw [stepForRangeAux.forRangeDyn_unfold, hStart, hStop, hStepOp]
  simp only [Option.bind_some]
  exact h_aux

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

/-! ## Compiled inner-loop body (single inner chunk, `BC = BT`)

The algorithm-lowered inner `i_c` loop body: four block-pointer constructions
(`p_k`, `p_d`, `p_v`, `p_v_new`), three loads, the `b_v ← b_v − dot(b_d, b_h)`
correction, the masked `v_new` store, and the `b_h_cumsum += dot(b_k, b_v)`
accumulation. The dynamic dtype casts erase to the algorithm layer. -/
def chunkDeltaInnerBody (k v d v_new : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d T K V BT BC BK BV : Nat) : List Stmt :=
  [ Stmt.assign .blockPtr [BK, BC] "p_k"
      (Op.makeBlockPtrDynOffsets k
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_qk_h)) [K, T]
        [BK, BC] [s_qk_d, s_qk_t]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BK),
          Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT))
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat BC))]),
    Stmt.assign .blockPtr [BC, BK] "p_d"
      (Op.makeBlockPtrDynOffsets d
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_qk_h)) [T, K]
        [BC, BK] [s_qk_t, s_qk_d]
        [Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT))
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat BC)),
          Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BK)]),
    Stmt.assign .blockPtr [BC, BV] "p_v"
      (Op.makeBlockPtrDynOffsets v
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_vo_h)) [T, V]
        [BC, BV] [s_vo_t, s_vo_d]
        [Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT))
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat BC)),
          Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_v") (Op.constNat BV)]),
    Stmt.assign .blockPtr [BC, BV] "p_v_new"
      (Op.makeBlockPtrDynOffsets v_new
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_vo_h)) [T, V]
        [BC, BV] [s_vo_t, s_vo_d]
        [Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT))
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat BC)),
          Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_v") (Op.constNat BV)]),
    Stmt.assign .real [BK, BC] "b_k"
      (Op.load .real (.blockPtr (Op.ref .blockPtr [BK, BC] "p_k") [0, 1]) .none),
    Stmt.assign .real [BC, BK] "b_d"
      (Op.load .real (.blockPtr (Op.ref .blockPtr [BC, BK] "p_d") [0, 1]) .none),
    Stmt.assign .real [BC, BV] "b_v"
      (Op.load .real (.blockPtr (Op.ref .blockPtr [BC, BV] "p_v") [0, 1]) .none),
    Stmt.assign .real [BC, BV] "b_v"
      (Op.sub .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BC, BV] "b_v")
        (Op.dot (batch := []) (Op.ref .real [BC, BK] "b_d") (Op.ref .real [BK, BV] "b_h"))),
    Stmt.store .real [BC, BV]
      (.blockPtr (Op.ref .blockPtr [BC, BV] "p_v_new") [0, 1])
      (Op.ref .real [BC, BV] "b_v") .none,
    Stmt.assign .real [BK, BV] "b_h_cumsum"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BK, BV] "b_h_cumsum")
        (Op.dot (batch := []) (Op.ref .real [BK, BC] "b_k") (Op.ref .real [BC, BV] "b_v"))) ]

/-- Loaded `b_k` tile (block ptr `(K,T)` strides `(s_qk_d,s_qk_t)`, offsets
`(0, i_t·BT)`), as `load_bp_2d_ref` emits it. -/
noncomputable def bkTile (s : BlockState) (k : RegionName)
    (s_qk_h s_qk_t s_qk_d T K BT BK BC : Nat) (i_t : Nat) : Tile .real [BK, BC] :=
  ⟨fun idx => if (0 * BK + idx.1.val < K ∧ i_t * BT + idx.2.1.val < T) then
      some (s.readMem k (s.pids 2 * s_qk_h + (0 * BK + idx.1.val) * s_qk_d
        + (i_t * BT + idx.2.1.val) * s_qk_t)) else some 0⟩

/-- Loaded `b_d` tile (block ptr `(T,K)` strides `(s_qk_t,s_qk_d)`, offsets
`(i_t·BT, 0)`). -/
noncomputable def bdTile (s : BlockState) (d : RegionName)
    (s_qk_h s_qk_t s_qk_d T K BT BC BK : Nat) (i_t : Nat) : Tile .real [BC, BK] :=
  ⟨fun idx => if (i_t * BT + idx.1.val < T ∧ 0 * BK + idx.2.1.val < K) then
      some (s.readMem d (s.pids 2 * s_qk_h + (i_t * BT + idx.1.val) * s_qk_t
        + (0 * BK + idx.2.1.val) * s_qk_d)) else some 0⟩

/-- Loaded `b_v` tile (block ptr `(T,V)` strides `(s_vo_t,s_vo_d)`, offsets
`(i_t·BT, i_v·BV)`). -/
noncomputable def bvTile (s : BlockState) (v : RegionName)
    (s_vo_h s_vo_t s_vo_d T V BT BC BV : Nat) (i_t : Nat) : Tile .real [BC, BV] :=
  ⟨fun idx => if (i_t * BT + idx.1.val < T ∧ s.pids 1 * BV + idx.2.1.val < V) then
      some (s.readMem v (s.pids 2 * s_vo_h + (i_t * BT + idx.1.val) * s_vo_t
        + (s.pids 1 * BV + idx.2.1.val) * s_vo_d)) else some 0⟩

/-! ## Tile-lane index helpers and active region -/

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

/-! ## Genuine delta-rule recurrence closed form

Element accessors use the kernel's exact block-pointer layouts at `i_k = 0`
(the `NK = 1` regime). The recurrence is given for the single-inner-chunk regime
`BC = BT` (`ceil(BT/BC) = 1`, the checked Python shape), so each time chunk `i_t`
covers rows `i_t·BT … i_t·BT + BT − 1` with a single inner chunk `i_c = 0`. -/

/-- `k[e, c]` element (block ptr `(K,T)` strides `(s_qk_d, s_qk_t)`, offsets
`(0, i_t·BT)`): `k` at `i_bh·s_qk_h + e·s_qk_d + (i_t·BT + c)·s_qk_t`. -/
noncomputable def kElem (s : BlockState) (k : RegionName)
    (s_qk_h s_qk_t s_qk_d BT : Nat) (i_t e c : Nat) : ℝ :=
  s.readMem k (s.pids 2 * s_qk_h + e * s_qk_d + (i_t * BT + c) * s_qk_t)

/-- `d[c, e]` element (block ptr `(T,K)` strides `(s_qk_t, s_qk_d)`, offsets
`(i_t·BT, 0)`): `d` at `i_bh·s_qk_h + (i_t·BT + c)·s_qk_t + e·s_qk_d`. -/
noncomputable def dElem (s : BlockState) (d : RegionName)
    (s_qk_h s_qk_t s_qk_d BT : Nat) (i_t c e : Nat) : ℝ :=
  s.readMem d (s.pids 2 * s_qk_h + (i_t * BT + c) * s_qk_t + e * s_qk_d)

/-- `v[c, p]` element (block ptr `(T,V)` strides `(s_vo_t, s_vo_d)`, offsets
`(i_t·BT, i_v·BV)`): `v` at `i_bh·s_vo_h + (i_t·BT + c)·s_vo_t + (i_v·BV + p)·s_vo_d`. -/
noncomputable def vElem (s : BlockState) (v : RegionName)
    (s_vo_h s_vo_t s_vo_d BT BV : Nat) (i_t c p : Nat) : ℝ :=
  s.readMem v (s.pids 2 * s_vo_h + (i_t * BT + c) * s_vo_t + (s.pids 1 * BV + p) * s_vo_d)

/-- `initial_state[e, p]` element (block ptr `(K,V)` strides `(V,1)`, offsets
`(0, i_v·BV)`): `initial_state` at `i_bh·K·V + e·V + (i_v·BV + p)`. -/
noncomputable def initElem (s : BlockState) (initial_state : RegionName)
    (K V BV : Nat) (e p : Nat) : ℝ :=
  s.readMem initial_state (s.pids 2 * K * V + e * V + (s.pids 1 * BV + p))

/-- Genuine closed form for the chunk-delta state recurrence in the
single-inner-chunk regime (`BC = BT`).

`stateValue i_t e p` is the state `H_{i_t}[e,p]` carried into chunk `i_t`
(`= h[i_t][e,p]`, the stored state). `H_0` is the seed (`initElem` when
`USE_INITIAL_STATE`, else `0`). The state advances by

```
  H_{i_t+1}[e,p] = H_{i_t}[e,p] + Σ_c k_{i_t}[e,c] · vNew_{i_t}[c,p]
```

where the corrected value `vNew_{i_t}[c,p] = v_{i_t}[c,p] − Σ_e d_{i_t}[c,e] ·
H_{i_t}[e,p]` is inlined into the advance step (the kernel computes it from the
*same* chunk-start state `H_{i_t}`, so the dependency is well-founded on `i_t`). -/
noncomputable def stateValue (s : BlockState)
    (k v d initial_state : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d K V BT BV BK : Nat)
    (USE_INITIAL_STATE : Bool) :
    Nat → Nat → Nat → ℝ
  | 0, e, p =>
      if USE_INITIAL_STATE then initElem s initial_state K V BV e p else 0
  | i_t + 1, e, p =>
      stateValue s k v d initial_state s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
          K V BT BV BK USE_INITIAL_STATE i_t e p
        + Finset.univ.sum (fun c : Fin BT =>
            kElem s k s_qk_h s_qk_t s_qk_d BT i_t e c.val
              * (vElem s v s_vo_h s_vo_t s_vo_d BT BV i_t c.val p
                  - Finset.univ.sum (fun e' : Fin BK =>
                      dElem s d s_qk_h s_qk_t s_qk_d BT i_t c.val e'.val
                        * stateValue s k v d initial_state s_qk_h s_qk_t s_qk_d
                            s_vo_h s_vo_t s_vo_d K V BT BV BK USE_INITIAL_STATE
                            i_t e'.val p)))

/-- The corrected value `v_new[i_t][c,p] = v − d · H_{i_t}` (a non-recursive
wrapper over the chunk-start state). -/
noncomputable def vNewValue (s : BlockState)
    (k v d initial_state : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d K V BT BV BK : Nat)
    (USE_INITIAL_STATE : Bool)
    (i_t c p : Nat) : ℝ :=
  vElem s v s_vo_h s_vo_t s_vo_d BT BV i_t c p
    - Finset.univ.sum (fun e : Fin BK =>
        dElem s d s_qk_h s_qk_t s_qk_d BT i_t c e.val
          * stateValue s k v d initial_state s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t
              s_vo_d K V BT BV BK USE_INITIAL_STATE i_t e.val p)

/-- Stored `h[i_t]` tile lane `(e,p)`: the state `H_{i_t}[e,p]` at chunk start. -/
noncomputable def hValue (s : BlockState)
    (k v d initial_state : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d K V BT BV BK : Nat)
    (USE_INITIAL_STATE : Bool)
    (i_t : Nat) (idx : TileIndex [BK, BV]) : ℝ :=
  stateValue s k v d initial_state s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
    K V BT BV BK USE_INITIAL_STATE i_t (kIndex s BK idx.1) idx.2.1.val

/-- Final state tile lane `(e,p)`: `H_{NT}[e,p]`. -/
noncomputable def finalValue (s : BlockState)
    (k v d initial_state : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d K V BT BV BK : Nat)
    (USE_INITIAL_STATE : Bool)
    (NT : Nat) (idx : TileIndex [BK, BV]) : ℝ :=
  stateValue s k v d initial_state s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
    K V BT BV BK USE_INITIAL_STATE NT (kIndex s BK idx.1) idx.2.1.val

/-! ## State (`h`) store face -/

/-- Proof-oriented state-store slice of `chunk_delta_fwd.py`'s
`chunk_delta_rule_fwd_kernel_h`. Models one `i_t` store from a precomputed `BH`
tile into `HOut`, preserving the source K/V block offsets and boundary checks. -/
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

/-- **State store face realizes the genuine recurrence.** Under `hBH` (the
producer materialized `hValue` into `BH`) and offset injectivity, the kernel's
`h[i_t]` store realizes the genuine state recurrence `stateValue i_t` at every
active lane. -/
theorem chunk_delta_fwd_h_store_slice_realizes_state
    (BH HOut k v d initial_state : RegionName)
    (i_t s_h_h s_h_t s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
      K V BT BV BK : Nat) (USE_INITIAL_STATE : Bool)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BK, BV] => hOffset s i_t s_h_h s_h_t K V BK BV idx))
    (hBH : ∀ idx : TileIndex [BK, BV], active s K V BK BV idx →
        s.readMem BH (hOffset s i_t s_h_h s_h_t K V BK BV idx)
          = hValue s k v d initial_state s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
              K V BT BV BK USE_INITIAL_STATE i_t idx) :
    ComputeCorrect.Realizes
      (kernel := chunk_delta_fwd_h_store_slice BH HOut i_t s_h_h s_h_t K V BK BV)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BK, BV] => active s K V BK BV idx)
        (fun idx : TileIndex [BK, BV] => (HOut, hOffset s i_t s_h_h s_h_t K V BK BV idx)))
      (expected := fun idx : TileIndex [BK, BV] =>
        hValue s k v d initial_state s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
          K V BT BV BK USE_INITIAL_STATE i_t idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [chunk_delta_fwd_h_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := chunk_delta_fwd_h_store_slice_correct BH HOut i_t s_h_h s_h_t K V BK BV
    s hOutInj idx
  rw [hExec] at h
  have h2 := Option.some.inj h
  rw [if_pos hActive] at h2
  simp only at h2
  show s'.readMem HOut (hOffset s i_t s_h_h s_h_t K V BK BV idx) = _
  rw [h2, storeValue, if_pos hActive, WithBot.unbotD_some]
  exact hBH idx hActive

/-! ## Corrected-value (`v_new`) store face -/

/-- Proof-oriented v_new-store slice. Writes a precomputed `BVN` tile into `VNew`
at the per-iteration `(i_t, i_c)` chunk offsets. -/
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

/-- The corrected value tile lane `(c,p)` for inner chunk `i_c = 0` (the
single-inner-chunk regime `BC = BT`): the genuine `vNewValue`. -/
noncomputable def vNewSpec (s : BlockState)
    (k v d initial_state : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d K V BT BV BK : Nat)
    (USE_INITIAL_STATE : Bool)
    (i_t : Nat) (idx : TileIndex [BT, BV]) : ℝ :=
  vNewValue s k v d initial_state s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
    K V BT BV BK USE_INITIAL_STATE i_t idx.1.val idx.2.1.val

/-- **Corrected-value store face realizes the genuine recurrence.** Under `hBVN`
(the producer materialized `vNewValue` into `BVN`) and offset injectivity, the
kernel's `v_new` store realizes `vNewValue` at every active lane (inner chunk
`i_c = 0`, `BC = BT`). -/
theorem chunk_delta_fwd_v_new_store_slice_realizes_vNew
    (BVN VNew k v d initial_state : RegionName)
    (i_t s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
      T K V BT BV BK : Nat) (USE_INITIAL_STATE : Bool)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BT, BV] =>
        vNewOffset s i_t 0 s_vo_h s_vo_t s_vo_d BT BT BV idx))
    (hBVN : ∀ idx : TileIndex [BT, BV], vNewActive s i_t 0 T V BT BT BV idx →
        s.readMem BVN (vNewOffset s i_t 0 s_vo_h s_vo_t s_vo_d BT BT BV idx)
          = vNewSpec s k v d initial_state s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
              K V BT BV BK USE_INITIAL_STATE i_t idx) :
    ComputeCorrect.Realizes
      (kernel := chunk_delta_fwd_v_new_store_slice BVN VNew i_t 0
        s_vo_h s_vo_t s_vo_d T V BT BT BV)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BT, BV] => vNewActive s i_t 0 T V BT BT BV idx)
        (fun idx : TileIndex [BT, BV] =>
          (VNew, vNewOffset s i_t 0 s_vo_h s_vo_t s_vo_d BT BT BV idx)))
      (expected := fun idx : TileIndex [BT, BV] =>
        vNewSpec s k v d initial_state s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
          K V BT BV BK USE_INITIAL_STATE i_t idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [chunk_delta_fwd_v_new_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := chunk_delta_fwd_v_new_store_slice_correct BVN VNew i_t 0
    s_vo_h s_vo_t s_vo_d T V BT BT BV s hOutInj idx
  rw [hExec] at h
  have h2 := Option.some.inj h
  rw [if_pos hActive] at h2
  simp only at h2
  show s'.readMem VNew (vNewOffset s i_t 0 s_vo_h s_vo_t s_vo_d BT BT BV idx) = _
  rw [h2, vNewStoreValue, if_pos hActive, WithBot.unbotD_some]
  exact hBVN idx hActive

/-! ## Final-state store face -/

/-- Proof-oriented final-state store slice. Writes a precomputed final-state
`BHFinal` tile into `FinalState` after the loop completes
(`STORE_FINAL_STATE = True`). -/
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

/-- **Final-state store face realizes the genuine recurrence.** Under `hBHF`
(the producer materialized `finalValue` into `BHFinal`) and offset injectivity,
the kernel's `final_state` store realizes `H_{NT}` at every active lane. -/
theorem chunk_delta_fwd_final_state_store_slice_realizes_final
    (BHFinal FinalState k v d initial_state : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d K V BT BV BK NT : Nat)
    (USE_INITIAL_STATE : Bool)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BK, BV] => finalStateOffset s K V BK BV idx))
    (hBHF : ∀ idx : TileIndex [BK, BV], active s K V BK BV idx →
        s.readMem BHFinal (finalStateOffset s K V BK BV idx)
          = finalValue s k v d initial_state s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t
              s_vo_d K V BT BV BK USE_INITIAL_STATE NT idx) :
    ComputeCorrect.Realizes
      (kernel := chunk_delta_fwd_final_state_store_slice BHFinal FinalState K V BK BV)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BK, BV] => active s K V BK BV idx)
        (fun idx : TileIndex [BK, BV] => (FinalState, finalStateOffset s K V BK BV idx)))
      (expected := fun idx : TileIndex [BK, BV] =>
        finalValue s k v d initial_state s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
          K V BT BV BK USE_INITIAL_STATE NT idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [chunk_delta_fwd_final_state_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := chunk_delta_fwd_final_state_store_slice_correct BHFinal FinalState K V BK BV
    s hOutInj idx
  rw [hExec] at h
  have h2 := Option.some.inj h
  rw [if_pos hActive] at h2
  simp only at h2
  show s'.readMem FinalState (finalStateOffset s K V BK BV idx) = _
  rw [h2, finalStateStoreValue, if_pos hActive, WithBot.unbotD_some]
  exact hBHF idx hActive

/-! ## Cross-chunk exec derivation (concrete Python shape)

The genuine `exec`-driven derivation of the producer hypotheses. We step the
lowered surface body of the checked Python shape
(`s_qk_h=8192, s_qk_t=128, s_qk_d=1, s_vo_h=8192, s_vo_t=64, s_vo_d=1,
s_h_h=16384, s_h_t=64, K=V=64, BT=BC=32, BK=BV=64, NT=4`) through:

* the prologue (`i_k`/`i_v`/`i_bh`, `b_h = 0`, the `USE_INITIAL_STATE` seed);
* the outer **static** `forRange "i_t" 0 4 1` carrying the `[64,64]` state tile
  `b_h = cdfStateTile i_t` (= `stateValue i_t`), with the chunk-start `h[i_t]`
  store and the cumulative `v_new[i_t]` store materialized for chunks `< i_t`;
* the inner **dynamic** `forRangeDyn "i_c" 0 1 1` (single inner chunk, `BC=BT`)
  loading `b_k`/`b_d`/`b_v`, correcting `b_v ← b_v − dot(b_d,b_h)`, storing
  `v_new`, and accumulating `b_h_cumsum += dot(b_k,b_v)`;
* the `STORE_FINAL_STATE` flush of `H_4`.

Everything is at `i_k = 0` (the `NK=1` regime). `pids 0 = i_k`, `pids 1 = i_v`,
`pids 2 = i_bh`. We keep the `BlockState` symbolic and peel `setReg` chains by
name inequality. -/

/-- The carried `[64,64]` state tile of chunk `i_t`: data `(e,p) ↦ stateValue`. -/
noncomputable def cdfStateTile (s : BlockState) (k v d initial_state : RegionName)
    (USE_INITIAL_STATE : Bool) (i_t : Nat) : Tile .real [64, 64] :=
  ⟨fun idx => some (stateValue s k v d initial_state 8192 128 1 8192 64 1 64 64 32 64 64
    USE_INITIAL_STATE i_t idx.1.val idx.2.1.val)⟩

/-- The loaded `b_k` tile `[64,32]` of chunk `i_t` (inner chunk `i_c=0`, `i_k=0`).
Cell `(e,c)` reads `k` at `i_bh·8192 + e·1 + (i_t·32+c)·128 = kElem`. -/
noncomputable def cdfBkTile (s : BlockState) (k : RegionName) (i_t : Nat) :
    Tile .real [64, 32] :=
  ⟨fun idx => if (idx.1.val < 64 ∧ i_t * 32 + idx.2.1.val < 128) then
      some (kElem s k 8192 128 1 32 i_t idx.1.val idx.2.1.val) else some 0⟩

/-- The loaded `b_d` tile `[32,64]` of chunk `i_t`. Cell `(c,e)` reads `dElem`. -/
noncomputable def cdfBdTile (s : BlockState) (d : RegionName) (i_t : Nat) :
    Tile .real [32, 64] :=
  ⟨fun idx => if (i_t * 32 + idx.1.val < 128 ∧ idx.2.1.val < 64) then
      some (dElem s d 8192 128 1 32 i_t idx.1.val idx.2.1.val) else some 0⟩

/-- The loaded `b_v` tile `[32,64]` of chunk `i_t`. Cell `(c,p)` reads `vElem`. -/
noncomputable def cdfBvTile (s : BlockState) (v : RegionName) (i_t : Nat) :
    Tile .real [32, 64] :=
  ⟨fun idx => if (i_t * 32 + idx.1.val < 128 ∧ s.pids 1 * 64 + idx.2.1.val < 64) then
      some (vElem s v 8192 64 1 32 64 i_t idx.1.val idx.2.1.val) else some 0⟩

/-- Cell-level `dot(ref a, ref b)` eval recipe (returns the closed-form cell tile). -/
theorem cdfDot_op_eval {M Kd N : Nat} (s' : BlockState) (aName bName : RegName)
    (atile : Tile .real [M, Kd]) (btile : Tile .real [Kd, N])
    (fa : Fin M → Fin Kd → ℝ) (fb : Fin Kd → Fin N → ℝ)
    (ha : s'.regs .real [M, Kd] aName = some atile)
    (hb : s'.regs .real [Kd, N] bName = some btile)
    (haf : ∀ m e, atile.data (m, e, PUnit.unit) = some (fa m e))
    (hbf : ∀ e n, btile.data (e, n, PUnit.unit) = some (fb e n)) :
    evalOp (@Op.dot [] M Kd N (Op.ref .real [M, Kd] aName) (Op.ref .real [Kd, N] bName)) s'
      = some (⟨fun idx : TileIndex [M, N] =>
          some (Finset.univ.sum fun e : Fin Kd => fa idx.1 e * fb e idx.2.1)⟩
          : Tile .real [M, N]) := by
  have hev : evalOp (@Op.dot [] M Kd N (Op.ref .real [M, Kd] aName)
      (Op.ref .real [Kd, N] bName)) s' = some (Tile.dot [] atile btile) := by
    rw [evalOp_dot]; simp [ha, hb]
  rw [hev]
  refine congrArg some ?_
  ext idx; obtain ⟨m, n, u⟩ := idx
  exact dot2d_elem atile btile m n (fa m) (fun e => fb e n)
    (fun e => haf m e) (fun e => hbf e n)

/-- Cell-level accumulating `acc + dot(ref a, ref b)` eval recipe. -/
theorem cdfAccDot_op_eval {M Kd N : Nat} (s' : BlockState)
    (accName aName bName : RegName)
    (acctile : Tile .real [M, N]) (atile : Tile .real [M, Kd]) (btile : Tile .real [Kd, N])
    (facc : Fin M → Fin N → ℝ) (fa : Fin M → Fin Kd → ℝ) (fb : Fin Kd → Fin N → ℝ)
    (hacc : s'.regs .real [M, N] accName = some acctile)
    (ha : s'.regs .real [M, Kd] aName = some atile)
    (hb : s'.regs .real [Kd, N] bName = some btile)
    (haccf : ∀ m n, acctile.data (m, n, PUnit.unit) = some (facc m n))
    (haf : ∀ m e, atile.data (m, e, PUnit.unit) = some (fa m e))
    (hbf : ∀ e n, btile.data (e, n, PUnit.unit) = some (fb e n)) :
    evalOp (Op.add .real Broadcast.nil.consSame.consSame (Op.ref .real [M, N] accName)
        (@Op.dot [] M Kd N (Op.ref .real [M, Kd] aName) (Op.ref .real [Kd, N] bName))) s'
      = some (⟨fun idx : TileIndex [M, N] =>
          some (facc idx.1 idx.2.1
            + Finset.univ.sum fun e : Fin Kd => fa idx.1 e * fb e idx.2.1)⟩
          : Tile .real [M, N]) := by
  have hdotev := cdfDot_op_eval s' aName bName atile btile fa fb ha hb haf hbf
  set dottile : Tile .real [M, N] :=
    ⟨fun idx => some (Finset.univ.sum fun e : Fin Kd => fa idx.1 e * fb e idx.2.1)⟩ with hdt
  have hfull : evalOp (Op.add .real Broadcast.nil.consSame.consSame (Op.ref .real [M, N] accName)
      (@Op.dot [] M Kd N (Op.ref .real [M, Kd] aName) (Op.ref .real [Kd, N] bName))) s'
      = some (Tile.bop NumericDType.real.add Broadcast.nil.consSame.consSame acctile dottile) := by
    rw [evalOp_add]
    rw [show evalOp (Op.ref .real [M, N] accName) s' = some acctile from by rw [evalOp_ref]; exact hacc]
    show (evalOp (@Op.dot [] M Kd N (Op.ref .real [M, Kd] aName)
        (Op.ref .real [Kd, N] bName)) s').bind
        (fun vy => some (Tile.bop NumericDType.real.add Broadcast.nil.consSame.consSame acctile vy))
      = _
    rw [hdotev]
    rfl
  rw [hfull]
  refine congrArg some ?_
  ext idx; obtain ⟨m, n, u⟩ := idx
  simp only [hdt, Tile.bop, Broadcast.consSame, Broadcast.leftIndex, Broadcast.rightIndex,
    haccf m n, NumericDType.add, WithBot.realAdd, Option.map₂, Option.bind, Option.map]

/-- Cell-level subtracting `acc − dot(ref a, ref b)` eval recipe (the `b_v` correction). -/
theorem cdfSubDot_op_eval {M Kd N : Nat} (s' : BlockState)
    (accName aName bName : RegName)
    (acctile : Tile .real [M, N]) (atile : Tile .real [M, Kd]) (btile : Tile .real [Kd, N])
    (facc : Fin M → Fin N → ℝ) (fa : Fin M → Fin Kd → ℝ) (fb : Fin Kd → Fin N → ℝ)
    (hacc : s'.regs .real [M, N] accName = some acctile)
    (ha : s'.regs .real [M, Kd] aName = some atile)
    (hb : s'.regs .real [Kd, N] bName = some btile)
    (haccf : ∀ m n, acctile.data (m, n, PUnit.unit) = some (facc m n))
    (haf : ∀ m e, atile.data (m, e, PUnit.unit) = some (fa m e))
    (hbf : ∀ e n, btile.data (e, n, PUnit.unit) = some (fb e n)) :
    evalOp (Op.sub .real Broadcast.nil.consSame.consSame (Op.ref .real [M, N] accName)
        (@Op.dot [] M Kd N (Op.ref .real [M, Kd] aName) (Op.ref .real [Kd, N] bName))) s'
      = some (⟨fun idx : TileIndex [M, N] =>
          some (facc idx.1 idx.2.1
            - Finset.univ.sum fun e : Fin Kd => fa idx.1 e * fb e idx.2.1)⟩
          : Tile .real [M, N]) := by
  have hdotev := cdfDot_op_eval s' aName bName atile btile fa fb ha hb haf hbf
  set dottile : Tile .real [M, N] :=
    ⟨fun idx => some (Finset.univ.sum fun e : Fin Kd => fa idx.1 e * fb e idx.2.1)⟩ with hdt
  have hfull : evalOp (Op.sub .real Broadcast.nil.consSame.consSame (Op.ref .real [M, N] accName)
      (@Op.dot [] M Kd N (Op.ref .real [M, Kd] aName) (Op.ref .real [Kd, N] bName))) s'
      = some (Tile.bop NumericDType.real.sub Broadcast.nil.consSame.consSame acctile dottile) := by
    rw [evalOp_sub]
    rw [show evalOp (Op.ref .real [M, N] accName) s' = some acctile from by rw [evalOp_ref]; exact hacc]
    show (evalOp (@Op.dot [] M Kd N (Op.ref .real [M, Kd] aName)
        (Op.ref .real [Kd, N] bName)) s').bind
        (fun vy => some (Tile.bop NumericDType.real.sub Broadcast.nil.consSame.consSame acctile vy))
      = _
    rw [hdotev]
    rfl
  rw [hfull]
  refine congrArg some ?_
  ext idx; obtain ⟨m, n, u⟩ := idx
  simp only [hdt, Tile.bop, Broadcast.consSame, Broadcast.leftIndex, Broadcast.rightIndex,
    haccf m n, NumericDType.sub, WithBot.realSub, Option.map₂, Option.bind, Option.map]

/-- Corrected `b_v` cell with an abstract chunk-start state `fbh` for `b_h`:
`vElem − Σ_e dElem·fbh(e,p)` on the active region (`0` off it). -/
noncomputable def cdfBvNewCell (s : BlockState) (v d : RegionName)
    (fbh : Nat → Nat → ℝ) (i_t : Nat) (c p : Nat) : ℝ :=
  (if (i_t * 32 + c < 128 ∧ s.pids 1 * 64 + p < 64) then
      vElem s v 8192 64 1 32 64 i_t c p else 0)
    - Finset.univ.sum (fun e : Fin 64 =>
        (if (i_t * 32 + c < 128 ∧ e.val < 64) then
            dElem s d 8192 128 1 32 i_t c e.val else 0) * fbh e.val p)

/-- `b_k` load equals `cdfBkTile` (memory matched to `s`). -/
theorem cdfLoad_bk_eq (s sin : BlockState) (k : RegionName) (i_t : Nat)
    (hmem : ∀ off, sin.readMem k off = s.readMem k off)
    (hpk : sin.regs .blockPtr [64, 32] "p_k" = some
      ⟨fun _ => BlockPtr.mk k (s.pids 2 * 8192) [64, 128] [64, 32] [1, 128]
        [0 * 64, i_t * 32 + 0 * 32]⟩) :
    evalOp (Op.load .real (.blockPtr (Op.ref .blockPtr [64, 32] "p_k") [0, 1]) .none) sin
      = some (cdfBkTile s k i_t) := by
  rw [load_bp_2d_ref k sin "p_k" (s.pids 2 * 8192) 64 128 64 32 1 128 (0 * 64) (i_t * 32 + 0 * 32) hpk]
  refine congrArg some ?_
  ext idx; obtain ⟨e, c, u⟩ := idx
  simp only [cdfBkTile, kElem, hmem, Nat.add_zero, Nat.zero_add, Nat.mul_one, Nat.zero_mul]

/-- `b_d` load equals `cdfBdTile`. -/
theorem cdfLoad_bd_eq (s sin : BlockState) (d : RegionName) (i_t : Nat)
    (hmem : ∀ off, sin.readMem d off = s.readMem d off)
    (hpd : sin.regs .blockPtr [32, 64] "p_d" = some
      ⟨fun _ => BlockPtr.mk d (s.pids 2 * 8192) [128, 64] [32, 64] [128, 1]
        [i_t * 32 + 0 * 32, 0 * 64]⟩) :
    evalOp (Op.load .real (.blockPtr (Op.ref .blockPtr [32, 64] "p_d") [0, 1]) .none) sin
      = some (cdfBdTile s d i_t) := by
  rw [load_bp_2d_ref d sin "p_d" (s.pids 2 * 8192) 128 64 32 64 128 1 (i_t * 32 + 0 * 32) (0 * 64) hpd]
  refine congrArg some ?_
  ext idx; obtain ⟨c, e, u⟩ := idx
  simp only [cdfBdTile, dElem, hmem, Nat.add_zero, Nat.zero_add, Nat.mul_one, Nat.zero_mul]

/-- `b_v` load equals `cdfBvTile`. -/
theorem cdfLoad_bv_eq (s sin : BlockState) (v : RegionName) (i_t : Nat)
    (hmem : ∀ off, sin.readMem v off = s.readMem v off)
    (hpv : sin.regs .blockPtr [32, 64] "p_v" = some
      ⟨fun _ => BlockPtr.mk v (s.pids 2 * 8192) [128, 64] [32, 64] [64, 1]
        [i_t * 32 + 0 * 32, s.pids 1 * 64]⟩) :
    evalOp (Op.load .real (.blockPtr (Op.ref .blockPtr [32, 64] "p_v") [0, 1]) .none) sin
      = some (cdfBvTile s v i_t) := by
  rw [load_bp_2d_ref v sin "p_v" (s.pids 2 * 8192) 128 64 32 64 64 1 (i_t * 32 + 0 * 32) (s.pids 1 * 64) hpv]
  refine congrArg some ?_
  ext idx; obtain ⟨c, p, u⟩ := idx
  simp only [cdfBvTile, vElem, hmem, Nat.add_zero, Nat.zero_add, Nat.mul_one, Nat.zero_mul]

/-- The corrected `b_v` tile `[32,64]`: data `(c,p) ↦ cdfBvNewCell`. -/
noncomputable def cdfBvNewTile (s : BlockState) (v d : RegionName)
    (fbh : Nat → Nat → ℝ) (i_t : Nat) : Tile .real [32, 64] :=
  ⟨fun idx => some (cdfBvNewCell s v d fbh i_t idx.1.val idx.2.1.val)⟩

/-- Eval of `b_v - dot(b_d, b_h)` = `cdfBvNewTile`. -/
theorem cdfCorrect_bv_eq (s sin : BlockState) (v d : RegionName) (i_t : Nat)
    (fbh : Nat → Nat → ℝ) (bhT : Tile .real [64, 64])
    (hbhf : ∀ e p, bhT.data (e, p, PUnit.unit) = some (fbh e.val p.val))
    (hbv : sin.regs .real [32, 64] "b_v" = some (cdfBvTile s v i_t))
    (hbd : sin.regs .real [32, 64] "b_d" = some (cdfBdTile s d i_t))
    (hbh : sin.regs .real [64, 64] "b_h" = some bhT) :
    evalOp (Op.sub .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [32, 64] "b_v")
        (Op.dot (batch := []) (Op.ref .real [32, 64] "b_d") (Op.ref .real [64, 64] "b_h"))) sin
      = some (cdfBvNewTile s v d fbh i_t) := by
  rw [cdfSubDot_op_eval sin "b_v" "b_d" "b_h" (cdfBvTile s v i_t) (cdfBdTile s d i_t) bhT
    (fun c p => if (i_t * 32 + c.val < 128 ∧ s.pids 1 * 64 + p.val < 64) then
        vElem s v 8192 64 1 32 64 i_t c.val p.val else 0)
    (fun c e => if (i_t * 32 + c.val < 128 ∧ e.val < 64) then
        dElem s d 8192 128 1 32 i_t c.val e.val else 0)
    (fun e p => fbh e.val p.val)
    hbv hbd hbh
    (fun c p => by simp only [cdfBvTile]; split <;> rfl)
    (fun c e => by simp only [cdfBdTile]; split <;> rfl)
    (fun e p => hbhf e p)]
  refine congrArg some ?_
  ext idx; obtain ⟨c, p, u⟩ := idx
  simp only [cdfBvNewTile, cdfBvNewCell]

/-- The accumulated `b_h_cumsum` tile `[64,64]`: data `(e,p) ↦ Σ_c kElem·vNewCell`. -/
noncomputable def cdfCumsumTile (s : BlockState) (k v d : RegionName)
    (fbh : Nat → Nat → ℝ) (i_t : Nat) : Tile .real [64, 64] :=
  ⟨fun idx : TileIndex [64, 64] =>
    some (Finset.univ.sum fun c : Fin 32 =>
      (if (idx.1.val < 64 ∧ i_t * 32 + c.val < 128) then
          kElem s k 8192 128 1 32 i_t idx.1.val c.val else 0)
        * cdfBvNewCell s v d fbh i_t c.val idx.2.1.val)⟩

/-- Eval of `b_h_cumsum + dot(b_k, b_v)` (from a zero `b_h_cumsum`) = `cdfCumsumTile`. -/
theorem cdfAccum_cumsum_eq (s sin : BlockState) (k v d : RegionName) (i_t : Nat)
    (fbh : Nat → Nat → ℝ)
    (hcs : sin.regs .real [64, 64] "b_h_cumsum"
        = some (⟨fun _ => some (0:ℝ)⟩ : Tile .real [64, 64]))
    (hbk : sin.regs .real [64, 32] "b_k" = some (cdfBkTile s k i_t))
    (hbv : sin.regs .real [32, 64] "b_v" = some (cdfBvNewTile s v d fbh i_t)) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [64, 64] "b_h_cumsum")
        (Op.dot (batch := []) (Op.ref .real [64, 32] "b_k") (Op.ref .real [32, 64] "b_v"))) sin
      = some (cdfCumsumTile s k v d fbh i_t) := by
  rw [cdfAccDot_op_eval sin "b_h_cumsum" "b_k" "b_v"
    (⟨fun _ => some (0:ℝ)⟩ : Tile .real [64, 64]) (cdfBkTile s k i_t) (cdfBvNewTile s v d fbh i_t)
    (fun _ _ => (0:ℝ))
    (fun e c => if (e.val < 64 ∧ i_t * 32 + c.val < 128) then
        kElem s k 8192 128 1 32 i_t e.val c.val else 0)
    (fun c p => cdfBvNewCell s v d fbh i_t c.val p.val)
    hcs hbk hbv
    (fun e p => rfl)
    (fun e c => by simp only [cdfBkTile]; split <;> rfl)
    (fun c p => by simp only [cdfBvNewTile])]
  refine congrArg some ?_
  ext idx; obtain ⟨e, p, u⟩ := idx
  simp only [cdfCumsumTile, zero_add]

/-- The block-ptr `v_new` store offset at lane `(c,p)` (inner chunk `i_c=0`). -/
def cdfVNewAddr (s : BlockState) (i_t : Nat) (idx : TileIndex [32, 64]) : Nat :=
  s.pids 2 * 8192 + (i_t * 32 + 0 * 32 + idx.1.val) * 64 + (s.pids 1 * 64 + idx.2.1.val) * 1

/-- The explicit post-store state of the `v_new` block-ptr store: the masked
writeMem foldl of the corrected value (built from the reference state `s`) over
the input state `sin`, at chunk `i_t`. -/
noncomputable def cdfVNewStoreState (s sin : BlockState) (v d v_new : RegionName)
    (i_t : Nat) (fbh : Nat → Nat → ℝ) : BlockState :=
  (TileShape.allIndices [32, 64]).foldl
    (fun acc i => if (i_t * 32 + 0 * 32 + i.1.val < 128 ∧ s.pids 1 * 64 + i.2.1.val < 64)
        then acc.writeMem v_new (cdfVNewAddr s i_t i)
          (cdfBvNewCell s v d fbh i_t i.1.val i.2.1.val) else acc) sin

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **`v_new` block-ptr store step (eq).** Stepping the masked block-ptr store of
`b_v` (= `cdfBvNewTile`) through `p_v_new` yields `cdfVNewStoreState`. -/
theorem cdfStore_vnew_step_eq (s sin : BlockState) (v d v_new : RegionName) (i_t : Nat)
    (fbh : Nat → Nat → ℝ)
    (hbv : sin.regs .real [32, 64] "b_v" = some (cdfBvNewTile s v d fbh i_t))
    (hpvn : sin.regs .blockPtr [32, 64] "p_v_new" = some
      ⟨fun _ => BlockPtr.mk v_new (s.pids 2 * 8192) [128, 64] [32, 64] [64, 1]
        [i_t * 32 + 0 * 32, s.pids 1 * 64]⟩) :
    stepStmt (Stmt.store .real [32, 64]
        (.blockPtr (Op.ref .blockPtr [32, 64] "p_v_new") [0, 1])
        (Op.ref .real [32, 64] "b_v") .none) sin
      = some (cdfVNewStoreState s sin v d v_new i_t fbh) := by
  unfold stepStmt cdfVNewStoreState
  simp only [evalOp_ref, hbv, hpvn, Option.bind, Option.map]
  refine congrArg some (congrArg (fun f => List.foldl f sin (TileShape.allIndices [32, 64])) ?_)
  funext acc i
  obtain ⟨c, p, u⟩ := i
  simp only [TileShape.indexToList, BlockPtr.address_2d_offsets, BlockPtr.inBounds_2d_offsets,
    Bool.true_and, cdfVNewAddr]
  by_cases hb : i_t * 32 + 0 * 32 + c.val < 128 ∧ s.pids 1 * 64 + p.val < 64
  · simp only [hb, decide_true, if_true, BlockState.writeMemTyped_real]; rfl
  · simp only [hb, decide_false, Bool.false_eq_true, if_false]

set_option maxHeartbeats 4000000 in
/-- **`v_new` store readback properties.** `cdfVNewStoreState` writes
`cdfBvNewCell` at every active lane and leaves all other addresses unchanged. -/
theorem cdfStore_vnew_step_props (s sin : BlockState) (v d v_new : RegionName) (i_t : Nat)
    (fbh : Nat → Nat → ℝ)
    (hInj : Function.Injective (fun idx : TileIndex [32, 64] => cdfVNewAddr s i_t idx)) :
    (cdfVNewStoreState s sin v d v_new i_t fbh).pids = sin.pids
      ∧ (cdfVNewStoreState s sin v d v_new i_t fbh).regs = sin.regs
      ∧ (∀ idx : TileIndex [32, 64],
          (i_t * 32 + 0 * 32 + idx.1.val < 128 ∧ s.pids 1 * 64 + idx.2.1.val < 64) →
          (cdfVNewStoreState s sin v d v_new i_t fbh).readMem v_new (cdfVNewAddr s i_t idx)
            = cdfBvNewCell s v d fbh i_t idx.1.val idx.2.1.val)
      ∧ (∀ rg off, rg ≠ v_new →
          (cdfVNewStoreState s sin v d v_new i_t fbh).readMem rg off = sin.readMem rg off)
      ∧ (∀ off, (∀ idx : TileIndex [32, 64], off ≠ cdfVNewAddr s i_t idx) →
          (cdfVNewStoreState s sin v d v_new i_t fbh).readMem v_new off = sin.readMem v_new off) := by
  classical
  unfold cdfVNewStoreState
  set offFn : TileIndex [32, 64] → Nat := fun idx => cdfVNewAddr s i_t idx with hoffFn
  set Pmask : TileIndex [32, 64] → Prop :=
    fun idx => i_t * 32 + 0 * 32 + idx.1.val < 128 ∧ s.pids 1 * 64 + idx.2.1.val < 64 with hPmask
  set valFn : TileIndex [32, 64] → ℝ :=
    fun idx => cdfBvNewCell s v d fbh i_t idx.1.val idx.2.1.val with hvalFn
  have hinj : Function.Injective offFn := hInj
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · rw [BlockState.foldl_writeMem_prop_masked_pids]
  · funext dtype shape name; rw [BlockState.foldl_writeMem_prop_masked_regs]
  · intro idx hidx
    have h := BlockState.scatter_readback_prop_masked_nd (region := v_new) sin offFn valFn Pmask hinj idx
    simp only [hoffFn] at h ⊢
    rw [h, if_pos (show Pmask idx from hidx)]
  · intro rg off hrg
    exact BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
      v_new offFn valFn Pmask _ sin rg off hrg
  · intro off hoff
    exact BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
      v_new offFn valFn Pmask _ sin off (fun i _ _ => hoff i)

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **Inner-loop body step (single inner chunk).** Stepping `chunkDeltaInnerBody`
from an entry state `sin` carrying `i_k=0`, `i_t`, `i_c=0`, an abstract `b_h` tile
(cell fn `fbh`), and `b_h_cumsum = 0`, with memory matching `s`, advances
`b_h_cumsum` to the cumulative `Σ_c b_k·b_vNew`, writes the corrected `v_new[i_t]`
block, and preserves `b_h`/pids/the loop registers and the `k`/`v`/`d` regions. -/
theorem chunkDeltaInnerBody_step
    (k v d v_new : RegionName) (s sin : BlockState) (i_t : Nat) (fbh : Nat → Nat → ℝ)
    (bhT : Tile .real [64, 64]) (hbhf : ∀ e p, bhT.data (e, p, PUnit.unit) = some (fbh e.val p.val))
    (hVk : v_new ≠ k) (hVv : v_new ≠ v) (hVd : v_new ≠ d)
    (hInj : Function.Injective (fun idx : TileIndex [32, 64] => cdfVNewAddr s i_t idx))
    (hmemK : ∀ off, sin.readMem k off = s.readMem k off)
    (hmemV : ∀ off, sin.readMem v off = s.readMem v off)
    (hmemD : ∀ off, sin.readMem d off = s.readMem d off)
    (_hpids : sin.pids = s.pids)
    (hik : sin.regs .nat [] "i_k" = some (Tile.scalar 0))
    (hiv : sin.regs .nat [] "i_v" = some (Tile.scalar (s.pids 1)))
    (hibh : sin.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)))
    (hit : sin.regs .nat [] "i_t" = some (Tile.scalar i_t))
    (hic : sin.regs .nat [] "i_c" = some (Tile.scalar 0))
    (hbh : sin.regs .real [64, 64] "b_h" = some bhT)
    (hcs : sin.regs .real [64, 64] "b_h_cumsum"
        = some (⟨fun _ => some (0:ℝ)⟩ : Tile .real [64, 64])) :
    ∃ s', stepStmts (chunkDeltaInnerBody k v d v_new 8192 128 1 8192 64 1 128 64 64 32 32 64 64) sin
        = some s'
      ∧ s'.pids = sin.pids
      ∧ s'.regs .nat [] "i_k" = some (Tile.scalar 0)
      ∧ s'.regs .nat [] "i_v" = sin.regs .nat [] "i_v"
      ∧ s'.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2))
      ∧ s'.regs .nat [] "i_t" = some (Tile.scalar i_t)
      ∧ s'.regs .real [64, 64] "b_h" = some bhT
      ∧ s'.regs .real [64, 64] "b_h_cumsum" = some (cdfCumsumTile s k v d fbh i_t)
      ∧ (∀ off, s'.readMem k off = s.readMem k off)
      ∧ (∀ off, s'.readMem v off = s.readMem v off)
      ∧ (∀ off, s'.readMem d off = s.readMem d off)
      ∧ (∀ rg off, rg ≠ v_new → s'.readMem rg off = sin.readMem rg off)
      ∧ (∀ idx : TileIndex [32, 64],
          (i_t * 32 + 0 * 32 + idx.1.val < 128 ∧ s.pids 1 * 64 + idx.2.1.val < 64) →
          s'.readMem v_new (cdfVNewAddr s i_t idx)
            = cdfBvNewCell s v d fbh i_t idx.1.val idx.2.1.val)
      ∧ (∀ off, (∀ idx : TileIndex [32, 64], off ≠ cdfVNewAddr s i_t idx) →
          s'.readMem v_new off = sin.readMem v_new off) := by
  unfold chunkDeltaInnerBody
  -- p_k
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (makeBlockPtr_2d_eval k sin _ _ _ [64, 128] [64, 32] [1, 128]
      (s.pids 2 * 8192) (0 * 64) (i_t * 32 + 0 * 32)
      (mulConst_eval sin "i_bh" (s.pids 2) 8192 hibh)
      (mulConst_eval sin "i_k" 0 64 hik)
      (addMulMul_eval sin "i_t" "i_c" i_t 32 0 32 hit hic)))]
  -- p_d
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (makeBlockPtr_2d_eval d _ _ _ _ [128, 64] [32, 64] [128, 1]
      (s.pids 2 * 8192) (i_t * 32 + 0 * 32) (0 * 64)
      (mulConst_eval _ "i_bh" (s.pids 2) 8192 (by simp [hibh]))
      (addMulMul_eval _ "i_t" "i_c" i_t 32 0 32 (by simp [hit]) (by simp [hic]))
      (mulConst_eval _ "i_k" 0 64 (by simp [hik]))))]
  -- p_v
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (makeBlockPtr_2d_eval v _ _ _ _ [128, 64] [32, 64] [64, 1]
      (s.pids 2 * 8192) (i_t * 32 + 0 * 32) (s.pids 1 * 64)
      (mulConst_eval _ "i_bh" (s.pids 2) 8192 (by simp [hibh]))
      (addMulMul_eval _ "i_t" "i_c" i_t 32 0 32 (by simp [hit]) (by simp [hic]))
      (mulConst_eval _ "i_v" (s.pids 1) 64 (by simp [hiv]))))]
  -- p_v_new
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (makeBlockPtr_2d_eval v_new _ _ _ _ [128, 64] [32, 64] [64, 1]
      (s.pids 2 * 8192) (i_t * 32 + 0 * 32) (s.pids 1 * 64)
      (mulConst_eval _ "i_bh" (s.pids 2) 8192 (by simp [hibh]))
      (addMulMul_eval _ "i_t" "i_c" i_t 32 0 32 (by simp [hit]) (by simp [hic]))
      (mulConst_eval _ "i_v" (s.pids 1) 64 (by simp [hiv]))))]
  -- b_k load
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cdfLoad_bk_eq s _ k i_t (by intro off; simp [hmemK]) (by simp)))]
  -- b_d load
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cdfLoad_bd_eq s _ d i_t (by intro off; simp [hmemD]) (by simp)))]
  -- b_v load
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cdfLoad_bv_eq s _ v i_t (by intro off; simp [hmemV]) (by simp)))]
  -- b_v correction
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cdfCorrect_bv_eq s _ v d i_t fbh bhT (fun e p => by simp [hbhf])
      (by simp) (by simp) (by simp [hbh])))]
  -- v_new store
  rw [stepStmts.cons_some (cdfStore_vnew_step_eq s _ v d v_new i_t fbh (by simp) (by simp))]
  obtain ⟨hpids9, hregs9, hvnew9, hother9, hoff9⟩ :=
    cdfStore_vnew_step_props s _ v d v_new i_t fbh hInj
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cdfAccum_cumsum_eq s _ k v d i_t fbh ?hcs9 ?hbk9 ?hbv9))]
  case hbk9 => rw [hregs9]; simp [BlockState.setReg_ne_name]
  case hbv9 => rw [hregs9]; simp [BlockState.setReg_same]
  case hcs9 => rw [hregs9]; simp [BlockState.setReg_ne_name, hcs]
  rw [stepStmts.nil]
  -- assemble
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · -- pids
    simp only [BlockState.setReg_pids, hpids9]
  · -- i_k
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hregs9]
    simpa using hik
  · -- i_v (unchanged; regs after setReg = vnew-store regs = sin.regs)
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hregs9]
    rfl
  · -- i_bh
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hregs9]
    simpa using hibh
  · -- i_t
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hregs9]
    simpa using hit
  · -- b_h
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hregs9]
    simpa using hbh
  · -- b_h_cumsum
    simp only [BlockState.setReg_same]
  · -- k unchanged
    intro off; simp only [BlockState.setReg_readMem]
    rw [hother9 k off hVk.symm]; simp [hmemK]
  · -- v unchanged
    intro off; simp only [BlockState.setReg_readMem]
    rw [hother9 v off hVv.symm]; simp [hmemV]
  · -- d unchanged
    intro off; simp only [BlockState.setReg_readMem]
    rw [hother9 d off hVd.symm]; simp [hmemD]
  · -- generic non-v_new region preserved vs sin
    intro rg off hrg; simp only [BlockState.setReg_readMem]
    exact hother9 rg off hrg
  · -- v_new active readback
    intro idx hidx; simp only [BlockState.setReg_readMem]; exact hvnew9 idx hidx
  · -- v_new other offset
    intro off hoff; simp only [BlockState.setReg_readMem]
    rw [hoff9 off hoff]; simp [BlockState.setReg_readMem]

/-- The explicit post-store state of the `h[i_t]` block-ptr store of a `[64,64]`
tile `bhT` (cell fn `fbh`) over input state `sin`, value built from `s`. -/
noncomputable def cdfHStoreState (s sin : BlockState) (h : RegionName)
    (i_t : Nat) (fbh : Nat → Nat → ℝ) : BlockState :=
  (TileShape.allIndices [64, 64]).foldl
    (fun acc i => if (s.pids 0 * 64 + i.1.val < 64 ∧ s.pids 1 * 64 + i.2.1.val < 64)
        then acc.writeMem h (hOffset s i_t 16384 64 64 64 64 64 i)
          (fbh i.1.val i.2.1.val) else acc) sin

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **`h[i_t]` block-ptr store step (eq).** Stepping the masked block-ptr store of
the carry tile `bhT` (cell fn `fbh`) through `p_h` yields `cdfHStoreState`. -/
theorem cdfStore_h_step_eq (s sin : BlockState) (h : RegionName) (i_t : Nat)
    (fbh : Nat → Nat → ℝ) (bhT : Tile .real [64, 64])
    (hbhf : ∀ e p, bhT.data (e, p, PUnit.unit) = some (fbh e.val p.val))
    (hbh : sin.regs .real [64, 64] "b_h" = some bhT)
    (hph : sin.regs .blockPtr [64, 64] "p_h" = some
      ⟨fun _ => BlockPtr.mk h (s.pids 2 * 16384 + i_t * 64 * 64) [64, 64] [64, 64] [64, 1]
        [s.pids 0 * 64, s.pids 1 * 64]⟩) :
    stepStmt (Stmt.store .real [64, 64]
        (.blockPtr (Op.ref .blockPtr [64, 64] "p_h") [0, 1])
        (Op.ref .real [64, 64] "b_h") .none) sin
      = some (cdfHStoreState s sin h i_t fbh) := by
  unfold stepStmt cdfHStoreState
  simp only [evalOp_ref, hbh, hph, Option.bind, Option.map]
  refine congrArg some (congrArg (fun f => List.foldl f sin (TileShape.allIndices [64, 64])) ?_)
  funext acc i
  obtain ⟨e, p, u⟩ := i
  simp only [TileShape.indexToList, BlockPtr.address_2d_offsets, BlockPtr.inBounds_2d_offsets,
    Bool.true_and, hOffset, kIndex, vIndex]
  by_cases hb : s.pids 0 * 64 + e.val < 64 ∧ s.pids 1 * 64 + p.val < 64
  · simp only [hb, decide_true, if_true, BlockState.writeMemTyped_real, hbhf, Nat.mul_one]
    rfl
  · simp only [hb, decide_false, Bool.false_eq_true, if_false, if_neg hb]

set_option maxHeartbeats 4000000 in
/-- **`h[i_t]` store readback properties.** `cdfHStoreState` writes `fbh` at every
active lane and leaves all other addresses unchanged. -/
theorem cdfStore_h_step_props (s sin : BlockState) (h : RegionName) (i_t : Nat)
    (fbh : Nat → Nat → ℝ)
    (hInj : Function.Injective
      (fun idx : TileIndex [64, 64] => hOffset s i_t 16384 64 64 64 64 64 idx)) :
    (cdfHStoreState s sin h i_t fbh).pids = sin.pids
      ∧ (cdfHStoreState s sin h i_t fbh).regs = sin.regs
      ∧ (∀ idx : TileIndex [64, 64], active s 64 64 64 64 idx →
          (cdfHStoreState s sin h i_t fbh).readMem h (hOffset s i_t 16384 64 64 64 64 64 idx)
            = fbh idx.1.val idx.2.1.val)
      ∧ (∀ rg off, rg ≠ h →
          (cdfHStoreState s sin h i_t fbh).readMem rg off = sin.readMem rg off)
      ∧ (∀ off, (∀ idx : TileIndex [64, 64], off ≠ hOffset s i_t 16384 64 64 64 64 64 idx) →
          (cdfHStoreState s sin h i_t fbh).readMem h off = sin.readMem h off) := by
  classical
  unfold cdfHStoreState
  set offFn : TileIndex [64, 64] → Nat := fun idx => hOffset s i_t 16384 64 64 64 64 64 idx with hoffFn
  set Pmask : TileIndex [64, 64] → Prop :=
    fun idx => s.pids 0 * 64 + idx.1.val < 64 ∧ s.pids 1 * 64 + idx.2.1.val < 64 with hPmask
  set valFn : TileIndex [64, 64] → ℝ := fun idx => fbh idx.1.val idx.2.1.val with hvalFn
  have hinj : Function.Injective offFn := hInj
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · rw [BlockState.foldl_writeMem_prop_masked_pids]
  · funext dtype shape name; rw [BlockState.foldl_writeMem_prop_masked_regs]
  · intro idx hidx
    have h := BlockState.scatter_readback_prop_masked_nd (region := h) sin offFn valFn Pmask hinj idx
    simp only [hoffFn] at h ⊢
    rw [h, if_pos (show Pmask idx from by
      obtain ⟨ka, va⟩ := hidx; exact ⟨by simpa [kIndex] using ka, by simpa [vIndex] using va⟩)]
  · intro rg off hrg
    exact BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
      h offFn valFn Pmask _ sin rg off hrg
  · intro off hoff
    exact BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
      h offFn valFn Pmask _ sin off (fun i _ _ => hoff i)

/-! ## Per-Python-shape offset injectivity

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

/-! ## Producer hypotheses (cross-chunk fold summary)

`producesState`/`producesVNew`/`producesFinal` say the within-kernel
accumulation materialized the genuine delta-rule closed form into the producer
buffer at every active lane. They summarize the `NT`/`ceil(BT/BC)` carry fold
(the two `tl.dot` matmuls, the `b_v ← b_v − b_d·b_h` correction, the
`b_h_cumsum += b_k·b_v` accumulation, and the `b_h += b_h_cumsum` advance) — the
analogue of the `chunk_cumsum` carry invariant and the `chunk_gla_simple`
producer hypothesis. -/

def producesState (s : BlockState) (BH k v d initial_state : RegionName)
    (i_t s_h_h s_h_t s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
      K V BT BV BK : Nat) (USE_INITIAL_STATE : Bool) : Prop :=
  ∀ idx : TileIndex [BK, BV], active s K V BK BV idx →
    s.readMem BH (hOffset s i_t s_h_h s_h_t K V BK BV idx)
      = hValue s k v d initial_state s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
          K V BT BV BK USE_INITIAL_STATE i_t idx

def producesVNew (s : BlockState) (BVN k v d initial_state : RegionName)
    (i_t s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
      T K V BT BV BK : Nat) (USE_INITIAL_STATE : Bool) : Prop :=
  ∀ idx : TileIndex [BT, BV], vNewActive s i_t 0 T V BT BT BV idx →
    s.readMem BVN (vNewOffset s i_t 0 s_vo_h s_vo_t s_vo_d BT BT BV idx)
      = vNewSpec s k v d initial_state s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
          K V BT BV BK USE_INITIAL_STATE i_t idx

def producesFinal (s : BlockState) (BHFinal k v d initial_state : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
      K V BT BV BK NT : Nat) (USE_INITIAL_STATE : Bool) : Prop :=
  ∀ idx : TileIndex [BK, BV], active s K V BK BV idx →
    s.readMem BHFinal (finalStateOffset s K V BK BV idx)
      = finalValue s k v d initial_state s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
          K V BT BV BK USE_INITIAL_STATE NT idx

/-! ## Public Python-case coverage summaries

Each summary certifies that (i) the full chunk-delta producer surface lowers to
the algorithm layer, and (ii) under the producer hypotheses the kernel store
faces realize the genuine delta-rule recurrence closed forms `hValue`,
`vNewSpec`, and `finalValue` at the case's exact shape. -/

/-- Public Python case 1 summary: no initial state, no final-state output.
`B=2,H=4,T=128,K=V=64,BT=BC=32,BK=BV=64,NT=4`. The full producer surface lowers,
and under the producer hypotheses the `h`/`v_new` store faces realize the genuine
recurrence. -/
theorem chunk_delta_fwd_python_case1_output_summary
    (k v d v_new h initial_state final_state BH BVN : RegionName)
    (i_t : Fin 4) (s : BlockState)
    (hBH : producesState s BH k v d initial_state
        i_t.val 16384 64 8192 128 1 8192 64 1 64 64 32 64 64 Bool.false)
    (hBVN : producesVNew s BVN k v d initial_state
        i_t.val 8192 128 1 8192 64 1 128 64 64 32 64 64 Bool.false) :
    (∃ alg, (chunk_delta_rule_fwd_h_surface k v d v_new h initial_state
      final_state 8192 128 1 8192 64 1 16384 64
      4 128 64 64 32 32 64 64 4 Bool.false Bool.false).toAlgorithm?
        = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := chunk_delta_fwd_h_store_slice BH h i_t.val 16384 64 64 64 64 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 64] => active s 64 64 64 64 idx)
        (fun idx : TileIndex [64, 64] => (h, hOffset s i_t.val 16384 64 64 64 64 64 idx)))
      (expected := fun idx : TileIndex [64, 64] =>
        hValue s k v d initial_state 8192 128 1 8192 64 1 64 64 32 64 64
          Bool.false i_t.val idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := chunk_delta_fwd_v_new_store_slice BVN v_new i_t.val 0
        8192 64 1 128 64 32 32 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [32, 64] => vNewActive s i_t.val 0 128 64 32 32 64 idx)
        (fun idx : TileIndex [32, 64] =>
          (v_new, vNewOffset s i_t.val 0 8192 64 1 32 32 64 idx)))
      (expected := fun idx : TileIndex [32, 64] =>
        vNewSpec s k v d initial_state 8192 128 1 8192 64 1 64 64 32 64 64
          Bool.false i_t.val idx)) := by
  refine ⟨chunk_delta_rule_fwd_h_surface_toAlgorithm_supported k v d v_new h
    initial_state final_state 8192 128 1 8192 64 1 16384 64
    4 128 64 64 32 32 64 64 4 Bool.false Bool.false, ?_, ?_⟩
  · exact chunk_delta_fwd_h_store_slice_realizes_state BH h k v d initial_state
      i_t.val 16384 64 8192 128 1 8192 64 1 64 64 32 64 64 Bool.false s
      (chunk_delta_fwd_h_python_test_shape_offset_injective s i_t) hBH
  · exact chunk_delta_fwd_v_new_store_slice_realizes_vNew BVN v_new k v d
      initial_state i_t.val 8192 128 1 8192 64 1 128 64 64 32 64 64 Bool.false s
      (chunk_delta_fwd_v_new_python_test_shape_offset_injective s i_t) hBVN

/-- Public Python case 2 summary: initial state and final-state output enabled.
Same shape as case 1. Under the producer hypotheses the `h`, `v_new`, and
`final_state` store faces realize the genuine recurrence (`H_{NT}` for the final
state). -/
theorem chunk_delta_fwd_python_case2_output_summary
    (k v d v_new h initial_state final_state BH BVN BHFinal : RegionName)
    (i_t : Fin 4) (s : BlockState)
    (hBH : producesState s BH k v d initial_state
        i_t.val 16384 64 8192 128 1 8192 64 1 64 64 32 64 64 Bool.true)
    (hBVN : producesVNew s BVN k v d initial_state
        i_t.val 8192 128 1 8192 64 1 128 64 64 32 64 64 Bool.true)
    (hBHF : producesFinal s BHFinal k v d initial_state
        8192 128 1 8192 64 1 64 64 32 64 64 4 Bool.true) :
    (∃ alg, (chunk_delta_rule_fwd_h_surface k v d v_new h initial_state
      final_state 8192 128 1 8192 64 1 16384 64
      4 128 64 64 32 32 64 64 4 Bool.true Bool.true).toAlgorithm?
        = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := chunk_delta_fwd_h_store_slice BH h i_t.val 16384 64 64 64 64 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 64] => active s 64 64 64 64 idx)
        (fun idx : TileIndex [64, 64] => (h, hOffset s i_t.val 16384 64 64 64 64 64 idx)))
      (expected := fun idx : TileIndex [64, 64] =>
        hValue s k v d initial_state 8192 128 1 8192 64 1 64 64 32 64 64
          Bool.true i_t.val idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := chunk_delta_fwd_v_new_store_slice BVN v_new i_t.val 0
        8192 64 1 128 64 32 32 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [32, 64] => vNewActive s i_t.val 0 128 64 32 32 64 idx)
        (fun idx : TileIndex [32, 64] =>
          (v_new, vNewOffset s i_t.val 0 8192 64 1 32 32 64 idx)))
      (expected := fun idx : TileIndex [32, 64] =>
        vNewSpec s k v d initial_state 8192 128 1 8192 64 1 64 64 32 64 64
          Bool.true i_t.val idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := chunk_delta_fwd_final_state_store_slice BHFinal final_state
        64 64 64 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 64] => active s 64 64 64 64 idx)
        (fun idx : TileIndex [64, 64] => (final_state, finalStateOffset s 64 64 64 64 idx)))
      (expected := fun idx : TileIndex [64, 64] =>
        finalValue s k v d initial_state 8192 128 1 8192 64 1 64 64 32 64 64
          Bool.true 4 idx)) := by
  refine ⟨chunk_delta_rule_fwd_h_surface_toAlgorithm_supported k v d v_new h
    initial_state final_state 8192 128 1 8192 64 1 16384 64
    4 128 64 64 32 32 64 64 4 Bool.true Bool.true, ?_, ?_, ?_⟩
  · exact chunk_delta_fwd_h_store_slice_realizes_state BH h k v d initial_state
      i_t.val 16384 64 8192 128 1 8192 64 1 64 64 32 64 64 Bool.true s
      (chunk_delta_fwd_h_python_test_shape_offset_injective s i_t) hBH
  · exact chunk_delta_fwd_v_new_store_slice_realizes_vNew BVN v_new k v d
      initial_state i_t.val 8192 128 1 8192 64 1 128 64 64 32 64 64 Bool.true s
      (chunk_delta_fwd_v_new_python_test_shape_offset_injective s i_t) hBVN
  · exact chunk_delta_fwd_final_state_store_slice_realizes_final BHFinal
      final_state k v d initial_state 8192 128 1 8192 64 1 64 64 32 64 64 4
      Bool.true s (chunk_delta_fwd_final_state_python_test_shape_offset_injective s) hBHF

/-! ## Cross-chunk exec carry-fold (genuine producer derivation)

We step the lowered (algorithm-layer, float-erased) surface body of the checked
Python shape directly, eliminating the producer hypotheses `hBH`/`hBVN`/`hBHF`.

* `cdfStopOp` — the inner `tl.cdiv(32, 32)` loop bound (evaluates to `1`).
* `cdfInnerBody` — the inner `forRangeDyn "i_c"` body (= `chunkDeltaInnerBody`).
* `cdfOuterBody` — the outer `forRange "i_t" 0 4 1` body: `p_h` make + `h` store
  + `b_h_cumsum = 0` + the inner dynamic loop + the `b_h += b_h_cumsum` advance.
* `cdfPrologue USE` — the prologue (`i_k`/`i_v`/`i_bh`, `b_h = 0`, the
  `USE_INITIAL_STATE` `ifThen` seed).
* `cdfEpilogue STORE` — the `STORE_FINAL_STATE` `ifThen` flush.

`chunk_delta_fwd_body_split` proves (by `rfl`) the full algorithm body equals
`cdfPrologue ++ [forRange "i_t" 0 4 1 cdfOuterBody] ++ cdfEpilogue`. -/

/-- The inner `forRangeDyn "i_c"` stop op `tl.cdiv(32, 32)` (= `(32+32-1)/32 = 1`). -/
def cdfStopOp : Op .nat [] :=
  Op.div .nat Broadcast.nil
    (Op.sub .nat Broadcast.nil
      (Op.add .nat Broadcast.nil (Op.constNat 32) (Op.constNat 32)) (Op.constNat 1))
    (Op.constNat 32)

theorem cdfStopOp_eval (s : BlockState) : evalOp cdfStopOp s = some (Tile.scalar 1) := by
  simp only [cdfStopOp, evalOp_div, evalOp_sub, evalOp_add, evalOp_constNat,
    Option.bind_some, Option.bind_eq_bind]
  refine congrArg some ?_
  apply Tile.ext
  intro z
  simp only [Tile.bop_data, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.div, NumericDType.sub, NumericDType.add]

/-- The outer `forRange "i_t" 0 4 1` body (float-erased, algorithm layer). -/
def cdfOuterBody (k v d v_new h : RegionName) : List Stmt :=
  [ Stmt.assign .blockPtr [64, 64] "p_h"
      (Op.makeBlockPtrDynOffsets h
        (Op.add .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat 16384))
          (Op.mul .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat 64))
            (Op.constNat 64)))
        [64, 64] [64, 64] [64, 1]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat 64),
          Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_v") (Op.constNat 64)]),
    Stmt.store .real [64, 64]
      (.blockPtr (Op.ref .blockPtr [64, 64] "p_h") [0, 1])
      (Op.ref .real [64, 64] "b_h") .none,
    Stmt.assign .real [64, 64] "b_h_cumsum"
      (Op.full [64, 64] (Op.const 0)),
    Stmt.forRangeDyn "i_c" (Op.constNat 0) cdfStopOp (Op.constNat 1)
      (chunkDeltaInnerBody k v d v_new 8192 128 1 8192 64 1 128 64 64 32 32 64 64),
    Stmt.assign .real [64, 64] "b_h"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [64, 64] "b_h")
        (Op.ref .real [64, 64] "b_h_cumsum")) ]

/-- The prologue (algorithm layer): program ids, `b_h = 0`, and the
`USE_INITIAL_STATE` `ifThen` seed (`p_h0` make + `b_h` load). -/
def cdfPrologue (initial_state : RegionName) (USE_INITIAL_STATE : Bool) : List Stmt :=
  [ Stmt.assign .nat [] "i_k" (Op.programId 0),
    Stmt.assign .nat [] "i_v" (Op.programId 1),
    Stmt.assign .nat [] "i_bh" (Op.programId 2),
    Stmt.assign .real [64, 64] "b_h" (Op.full [64, 64] (Op.const 0)),
    Stmt.ifThen (Op.constBool USE_INITIAL_STATE)
      [ Stmt.assign .blockPtr [64, 64] "p_h0"
          (Op.makeBlockPtrDynOffsets initial_state
            (Op.mul .nat Broadcast.nil
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat 64))
              (Op.constNat 64))
            [64, 64] [64, 64] [64, 1]
            [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat 64),
              Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_v") (Op.constNat 64)]),
        Stmt.assign .real [64, 64] "b_h"
          (Op.load .real (.blockPtr (Op.ref .blockPtr [64, 64] "p_h0") [0, 1]) .none) ] ]

/-- The epilogue (algorithm layer): the `STORE_FINAL_STATE` `ifThen` flush
(`p_ht` make + `final_state` store). -/
def cdfEpilogue (final_state : RegionName) (STORE_FINAL_STATE : Bool) : List Stmt :=
  [ Stmt.ifThen (Op.constBool STORE_FINAL_STATE)
      [ Stmt.assign .blockPtr [64, 64] "p_ht"
          (Op.makeBlockPtrDynOffsets final_state
            (Op.mul .nat Broadcast.nil
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat 64))
              (Op.constNat 64))
            [64, 64] [64, 64] [64, 1]
            [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat 64),
              Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_v") (Op.constNat 64)]),
        Stmt.store .real [64, 64]
          (.blockPtr (Op.ref .blockPtr [64, 64] "p_ht") [0, 1])
          (Op.ref .real [64, 64] "b_h") .none ] ]

set_option maxRecDepth 8000 in
/-- **Body split (by `rfl`).** The Python-shape surface lowers (float-erased) to
the prologue, the single outer `forRange "i_t" 0 4 1` carrying `cdfOuterBody`, and
the epilogue. -/
theorem chunk_delta_fwd_body_split
    (k v d v_new h initial_state final_state : RegionName)
    (USE_INITIAL_STATE STORE_FINAL_STATE : Bool) :
    (chunk_delta_rule_fwd_h_surface k v d v_new h initial_state final_state
      8192 128 1 8192 64 1 16384 64 4 128 64 64 32 32 64 64 4
      USE_INITIAL_STATE STORE_FINAL_STATE).toAlgKernel.body
      = cdfPrologue initial_state USE_INITIAL_STATE
        ++ [Stmt.forRange "i_t" 0 4 1 (cdfOuterBody k v d v_new h)]
        ++ cdfEpilogue final_state STORE_FINAL_STATE := by
  cases USE_INITIAL_STATE <;> cases STORE_FINAL_STATE <;> rfl

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **Inner-loop wrapper (single inner chunk).** Driving the inner
`forRangeDyn "i_c" 0 cdfStopOp 1 chunkDeltaInnerBody` (a single iteration, `BC=BT`)
from an entry state `sin` advances `b_h_cumsum` to `cdfCumsumTile`, writes the
corrected `v_new[i_t]` block, and preserves `b_h`/pids/the loop registers and the
`k`/`v`/`d` regions — the inner-loop face of one outer chunk. -/
theorem cdfInnerLoop_run
    (k v d v_new : RegionName) (s sin : BlockState) (i_t : Nat) (fbh : Nat → Nat → ℝ)
    (bhT : Tile .real [64, 64]) (hbhf : ∀ e p, bhT.data (e, p, PUnit.unit) = some (fbh e.val p.val))
    (hVk : v_new ≠ k) (hVv : v_new ≠ v) (hVd : v_new ≠ d)
    (hInj : Function.Injective (fun idx : TileIndex [32, 64] => cdfVNewAddr s i_t idx))
    (hmemK : ∀ off, sin.readMem k off = s.readMem k off)
    (hmemV : ∀ off, sin.readMem v off = s.readMem v off)
    (hmemD : ∀ off, sin.readMem d off = s.readMem d off)
    (hpids : sin.pids = s.pids)
    (hik : sin.regs .nat [] "i_k" = some (Tile.scalar 0))
    (hiv : sin.regs .nat [] "i_v" = some (Tile.scalar (s.pids 1)))
    (hibh : sin.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)))
    (hit : sin.regs .nat [] "i_t" = some (Tile.scalar i_t))
    (hbh : sin.regs .real [64, 64] "b_h" = some bhT)
    (hcs : sin.regs .real [64, 64] "b_h_cumsum"
        = some (⟨fun _ => some (0:ℝ)⟩ : Tile .real [64, 64])) :
    ∃ s', stepStmt (Stmt.forRangeDyn "i_c" (Op.constNat 0) cdfStopOp (Op.constNat 1)
        (chunkDeltaInnerBody k v d v_new 8192 128 1 8192 64 1 128 64 64 32 32 64 64)) sin
        = some s'
      ∧ s'.pids = sin.pids
      ∧ s'.regs .nat [] "i_k" = some (Tile.scalar 0)
      ∧ s'.regs .nat [] "i_v" = sin.regs .nat [] "i_v"
      ∧ s'.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2))
      ∧ s'.regs .nat [] "i_t" = some (Tile.scalar i_t)
      ∧ s'.regs .real [64, 64] "b_h" = some bhT
      ∧ s'.regs .real [64, 64] "b_h_cumsum" = some (cdfCumsumTile s k v d fbh i_t)
      ∧ (∀ off, s'.readMem k off = s.readMem k off)
      ∧ (∀ off, s'.readMem v off = s.readMem v off)
      ∧ (∀ off, s'.readMem d off = s.readMem d off)
      ∧ (∀ rg off, rg ≠ v_new → s'.readMem rg off = sin.readMem rg off)
      ∧ (∀ idx : TileIndex [32, 64],
          (i_t * 32 + 0 * 32 + idx.1.val < 128 ∧ s.pids 1 * 64 + idx.2.1.val < 64) →
          s'.readMem v_new (cdfVNewAddr s i_t idx)
            = cdfBvNewCell s v d fbh i_t idx.1.val idx.2.1.val)
      ∧ (∀ off, (∀ idx : TileIndex [32, 64], off ≠ cdfVNewAddr s i_t idx) →
          s'.readMem v_new off = sin.readMem v_new off) := by
  -- carry invariant: at counter 0, the entry state; at counter 1, the post-conditions.
  obtain ⟨final, s', hstep, hfinal, hP⟩ :=
    forRangeDyn_inv (idx := "i_c") (start := 0) (stop := 1) (step := 1)
      (startOp := Op.constNat 0) (stopOp := cdfStopOp) (stepOp := Op.constNat 1)
      (s_init := sin)
      (P := fun n st =>
        (n = 0 →
          (∀ off, st.readMem k off = s.readMem k off)
          ∧ (∀ off, st.readMem v off = s.readMem v off)
          ∧ (∀ off, st.readMem d off = s.readMem d off)
          ∧ st.pids = s.pids
          ∧ st.regs .nat [] "i_k" = some (Tile.scalar 0)
          ∧ st.regs .nat [] "i_v" = some (Tile.scalar (s.pids 1))
          ∧ st.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2))
          ∧ st.regs .nat [] "i_t" = some (Tile.scalar i_t)
          ∧ st.regs .real [64, 64] "b_h" = some bhT
          ∧ st.regs .real [64, 64] "b_h_cumsum" = some (⟨fun _ => some (0:ℝ)⟩ : Tile .real [64, 64])
          ∧ (∀ off, st.readMem v_new off = sin.readMem v_new off)
          ∧ (∀ rg off, st.readMem rg off = sin.readMem rg off)) ∧
        (n ≥ 1 →
          st.pids = sin.pids
          ∧ st.regs .nat [] "i_k" = some (Tile.scalar 0)
          ∧ st.regs .nat [] "i_v" = sin.regs .nat [] "i_v"
          ∧ st.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2))
          ∧ st.regs .nat [] "i_t" = some (Tile.scalar i_t)
          ∧ st.regs .real [64, 64] "b_h" = some bhT
          ∧ st.regs .real [64, 64] "b_h_cumsum" = some (cdfCumsumTile s k v d fbh i_t)
          ∧ (∀ off, st.readMem k off = s.readMem k off)
          ∧ (∀ off, st.readMem v off = s.readMem v off)
          ∧ (∀ off, st.readMem d off = s.readMem d off)
          ∧ (∀ rg off, rg ≠ v_new → st.readMem rg off = sin.readMem rg off)
          ∧ (∀ idx : TileIndex [32, 64],
              (i_t * 32 + 0 * 32 + idx.1.val < 128 ∧ s.pids 1 * 64 + idx.2.1.val < 64) →
              st.readMem v_new (cdfVNewAddr s i_t idx)
                = cdfBvNewCell s v d fbh i_t idx.1.val idx.2.1.val)
          ∧ (∀ off, (∀ idx : TileIndex [32, 64], off ≠ cdfVNewAddr s i_t idx) →
              st.readMem v_new off = sin.readMem v_new off)))
      (by simp [evalOp]) (cdfStopOp_eval sin) (by simp [evalOp]) (by norm_num)
      ⟨fun _ => ⟨hmemK, hmemV, hmemD, hpids, hik, hiv, hibh, hit, hbh, hcs,
          fun _ => rfl, fun _ _ => rfl⟩,
        fun h => absurd h (by norm_num)⟩
      (fun i st hlt hPi => by
        -- only i = 0 is reachable (stop = 1)
        interval_cases i
        obtain ⟨h0, _⟩ := hPi
        obtain ⟨hmemKSt, hmemVSt, hmemDSt, hpidsSt, hikSt, hivSt, hibhSt, hitSt,
          hbhSt, hcsSt, hvnSt, hsinSt⟩ := h0 rfl
        -- set i_c = 0, then run the inner body
        set sc := st.setReg "i_c" .nat [] (Tile.scalar 0) with hsc
        obtain ⟨s'', hbody, hp'', hik'', hiv'', hibh'', hit'', hbh'', hcs'',
          hk'', hv'', hd'', hgen'', hvn'', hoth''⟩ :=
          chunkDeltaInnerBody_step k v d v_new s sc i_t fbh bhT hbhf hVk hVv hVd hInj
            (by rw [hsc]; intro off; simp [BlockState.setReg_readMem, hmemKSt])
            (by rw [hsc]; intro off; simp [BlockState.setReg_readMem, hmemVSt])
            (by rw [hsc]; intro off; simp [BlockState.setReg_readMem, hmemDSt])
            (by rw [hsc]; simp [BlockState.setReg_pids, hpidsSt])
            (by rw [hsc]; simp [BlockState.setReg_ne_name, hikSt])
            (by rw [hsc]; simp [BlockState.setReg_ne_name, hivSt])
            (by rw [hsc]; simp [BlockState.setReg_ne_name, hibhSt])
            (by rw [hsc]; simp [BlockState.setReg_ne_name, hitSt])
            (by rw [hsc]; simp [BlockState.setReg_same])
            (by rw [hsc]; simp [BlockState.setReg_ne_name, hbhSt])
            (by rw [hsc]; simp [BlockState.setReg_ne_name, hcsSt])
        refine ⟨s'', hbody, ?_, ?_⟩
        · intro h; exact absurd h (by norm_num)
        · intro _
          refine ⟨?_, hik'', ?_, hibh'', hit'', hbh'', hcs'', ?_, ?_, ?_, ?_, ?_, ?_⟩
          · rw [hp'', hsc]; simp [BlockState.setReg_pids, hpidsSt, hpids]
          · -- i_v: body preserves it (= sc.regs i_v = st.regs i_v = sin.regs i_v)
            rw [hiv'', hsc, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hivSt, hiv]
          · intro off; rw [hk'']
          · intro off; rw [hv'']
          · intro off; rw [hd'']
          · -- generic non-v_new region preserved vs sin
            intro rg off hrg
            rw [hgen'' rg off hrg, hsc]; simp only [BlockState.setReg_readMem]
            exact hsinSt rg off
          · intro idx hidx; exact hvn'' idx hidx
          · intro off hoff
            rw [hoth'' off hoff, hsc]; simp only [BlockState.setReg_readMem]; exact hvnSt off)
  -- extract the post-conditions at the final counter (≥ 1)
  obtain ⟨_, hpost⟩ := hP
  obtain ⟨hp, hik', hiv', hibh', hit', hbh', hcs', hk', hv', hd', hgen', hvn', hoth'⟩ :=
    hpost hfinal
  exact ⟨s', hstep, hp, hik', hiv', hibh', hit', hbh', hcs', hk', hv', hd', hgen', hvn', hoth'⟩

/-- Abbreviation for the Python-shape genuine state value (`i_k = 0` regime). -/
noncomputable def cdfState (s : BlockState) (k v d initial_state : RegionName)
    (USE_INITIAL_STATE : Bool) (i_t e p : Nat) : ℝ :=
  stateValue s k v d initial_state 8192 128 1 8192 64 1 64 64 32 64 64
    USE_INITIAL_STATE i_t e p

set_option maxHeartbeats 4000000 in
/-- **State-advance linchpin.** On the active region (`p` such that
`s.pids 1 * 64 + p < 64`, i.e. `s.pids 1 = 0`, and `e < 64`) and for a time chunk
`i_t < 4`, the `cdfCumsumTile` cell built from the chunk-start state
`fbh = cdfState … i_t` equals exactly the closed-form advance increment
`stateValue (i_t+1) − stateValue i_t`. Hence
`stateValue (i_t+1) e p = stateValue i_t e p + cdfCumsumTile cell`. -/
theorem cdfAdvance_active
    (s : BlockState) (k v d initial_state : RegionName) (USE_INITIAL_STATE : Bool)
    (i_t : Nat) (hit : i_t < 4) (e p : Nat) (he : e < 64) (hp : s.pids 1 * 64 + p < 64) :
    (cdfCumsumTile s k v d
        (fun e' p' => cdfState s k v d initial_state USE_INITIAL_STATE i_t e' p') i_t).data
        (⟨e, by omega⟩, ⟨p, by omega⟩, PUnit.unit)
      = some (cdfState s k v d initial_state USE_INITIAL_STATE (i_t + 1) e p
          - cdfState s k v d initial_state USE_INITIAL_STATE i_t e p) := by
  simp only [cdfCumsumTile, cdfState, stateValue]
  refine congrArg some ?_
  -- the (i_t+1) state minus i_t state = Σ_c kElem · (vElem − Σ_e' dElem·state)
  rw [add_sub_cancel_left]
  -- both sides are Σ over c : Fin 32; match cell-wise
  refine Finset.sum_congr rfl (fun c _ => ?_)
  have hc : i_t * 32 + c.val < 128 := by have := c.isLt; omega
  -- left guard `e < 64 ∧ i_t*32+c<128` is True
  rw [if_pos ⟨he, hc⟩]
  -- expand cdfBvNewCell, collapse its guards
  simp only [cdfBvNewCell]
  rw [if_pos ⟨hc, hp⟩]
  -- the inner d-sum: collapse guard `i_t*32+c<128 ∧ e'<64`
  have hdsum :
      (Finset.univ.sum (fun e' : Fin 64 =>
        (if (i_t * 32 + c.val < 128 ∧ e'.val < 64) then
            dElem s d 8192 128 1 32 i_t c.val e'.val else 0)
          * stateValue s k v d initial_state 8192 128 1 8192 64 1 64 64 32 64 64
              USE_INITIAL_STATE i_t e'.val p))
      = Finset.univ.sum (fun e' : Fin 64 =>
          dElem s d 8192 128 1 32 i_t c.val e'.val
            * stateValue s k v d initial_state 8192 128 1 8192 64 1 64 64 32 64 64
                USE_INITIAL_STATE i_t e'.val p) := by
    refine Finset.sum_congr rfl (fun e' _ => ?_)
    rw [if_pos ⟨hc, e'.isLt⟩]
  rw [hdsum]

set_option maxHeartbeats 4000000 in
/-- **State-advance linchpin (general `fbh`).** The `cdfCumsumTile` cell at active
lane `(e,p)` built from any `fbh` that agrees with the chunk-start state on column
`p` (all `e'`) equals the closed-form advance increment. -/
theorem cdfAdvance_active_of
    (s : BlockState) (k v d initial_state : RegionName) (USE_INITIAL_STATE : Bool)
    (fbh : Nat → Nat → ℝ)
    (i_t : Nat) (hit : i_t < 4) (e p : Nat) (he : e < 64) (hp : s.pids 1 * 64 + p < 64)
    (hcol : ∀ e' : Nat, e' < 64 →
      fbh e' p = cdfState s k v d initial_state USE_INITIAL_STATE i_t e' p) :
    (cdfCumsumTile s k v d fbh i_t).data (⟨e, by omega⟩, ⟨p, by omega⟩, PUnit.unit)
      = some (cdfState s k v d initial_state USE_INITIAL_STATE (i_t + 1) e p
          - cdfState s k v d initial_state USE_INITIAL_STATE i_t e p) := by
  simp only [cdfCumsumTile, cdfState, stateValue]
  refine congrArg some ?_
  rw [add_sub_cancel_left]
  refine Finset.sum_congr rfl (fun c _ => ?_)
  have hc : i_t * 32 + c.val < 128 := by have := c.isLt; omega
  rw [if_pos ⟨he, hc⟩]
  simp only [cdfBvNewCell]
  rw [if_pos ⟨hc, hp⟩]
  have hdsum :
      (Finset.univ.sum (fun e' : Fin 64 =>
        (if (i_t * 32 + c.val < 128 ∧ e'.val < 64) then
            dElem s d 8192 128 1 32 i_t c.val e'.val else 0) * fbh e'.val p))
      = Finset.univ.sum (fun e' : Fin 64 =>
          dElem s d 8192 128 1 32 i_t c.val e'.val
            * stateValue s k v d initial_state 8192 128 1 8192 64 1 64 64 32 64 64
                USE_INITIAL_STATE i_t e'.val p) := by
    refine Finset.sum_congr rfl (fun e' _ => ?_)
    rw [if_pos ⟨hc, e'.isLt⟩, hcol e'.val e'.isLt]; rfl
  rw [hdsum]

/-! ### Outer-loop carry tile

The kernel's `b_h` register holds, after `c` chunks, a *concrete* `[64,64]` tile
`cdfCarryTile c` — the actual fold the kernel computes, defined recursively as the
seed plus the `cdfCumsumTile` increments. On the active region it agrees with the
genuine `stateValue` (`cdfCarry_active`), but off the active region it is whatever
the masked update produced (the kernel never reads those lanes). This separation
is the linchpin: the advance is exact *by construction*, while the genuine
recurrence match holds only on active lanes (where the stores read). -/

/-- The seed cell `(e,p)` of `b_h` after the prologue: the *masked* initial-state
block-ptr load (`initElem` on the active region when `USE_INITIAL_STATE`, else `0`).
The boundary check matches the kernel's `tl.make_block_ptr` load mask exactly. -/
noncomputable def cdfSeedCell (s : BlockState) (initial_state : RegionName)
    (USE_INITIAL_STATE : Bool) (e p : Nat) : ℝ :=
  if (s.pids 0 * 64 + e < 64 ∧ s.pids 1 * 64 + p < 64) then
    (if USE_INITIAL_STATE then initElem s initial_state 64 64 64 e p else 0)
  else 0

/-- The concrete carry cell `(e,p)` after `c` chunks: the actual fold value
`b_h` holds (seed + Σ of `cdfCumsumTile` increments). Total over all lanes. -/
noncomputable def cdfCarryCell (s : BlockState) (k v d initial_state : RegionName)
    (USE_INITIAL_STATE : Bool) : Nat → Nat → Nat → ℝ
  | 0, e, p => cdfSeedCell s initial_state USE_INITIAL_STATE e p
  | c + 1, e, p =>
      cdfCarryCell s k v d initial_state USE_INITIAL_STATE c e p
        + WithBot.unbotD 0
            ((cdfCumsumTile s k v d
                (fun e' p' => cdfCarryCell s k v d initial_state USE_INITIAL_STATE c e' p') c).data
              (⟨e % 64, Nat.mod_lt _ (by norm_num)⟩, ⟨p % 64, Nat.mod_lt _ (by norm_num)⟩, PUnit.unit))

/-- The concrete `[64,64]` carry tile after `c` chunks (data `(e,p) ↦ cdfCarryCell`). -/
noncomputable def cdfCarryTile (s : BlockState) (k v d initial_state : RegionName)
    (USE_INITIAL_STATE : Bool) (c : Nat) : Tile .real [64, 64] :=
  ⟨fun idx => some (cdfCarryCell s k v d initial_state USE_INITIAL_STATE c idx.1.val idx.2.1.val)⟩

theorem cdfCarryTile_data (s : BlockState) (k v d initial_state : RegionName)
    (USE_INITIAL_STATE : Bool) (c : Nat) (e p : Fin 64) :
    (cdfCarryTile s k v d initial_state USE_INITIAL_STATE c).data (e, p, PUnit.unit)
      = some (cdfCarryCell s k v d initial_state USE_INITIAL_STATE c e.val p.val) := rfl

set_option maxHeartbeats 4000000 in
/-- **Advance eval.** `b_h + b_h_cumsum`, with `b_h = cdfCarryTile c` and
`b_h_cumsum = cdfCumsumTile` (built from the `cdfCarryCell c` fn), evaluates to
`cdfCarryTile (c+1)`. -/
theorem cdfAdvance_eval (s sin : BlockState) (k v d initial_state : RegionName)
    (USE_INITIAL_STATE : Bool) (c : Nat)
    (hbh : sin.regs .real [64, 64] "b_h"
        = some (cdfCarryTile s k v d initial_state USE_INITIAL_STATE c))
    (hcs : sin.regs .real [64, 64] "b_h_cumsum"
        = some (cdfCumsumTile s k v d
            (fun e' p' => cdfCarryCell s k v d initial_state USE_INITIAL_STATE c e' p') c)) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [64, 64] "b_h")
        (Op.ref .real [64, 64] "b_h_cumsum")) sin
      = some (cdfCarryTile s k v d initial_state USE_INITIAL_STATE (c + 1)) := by
  rw [evalOp_add]
  simp only [evalOp_ref, hbh, hcs, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx; obtain ⟨e, p, u⟩ := idx
  simp only [Tile.bop, Broadcast.consSame, Broadcast.leftIndex, Broadcast.rightIndex,
    cdfCarryTile, NumericDType.add, WithBot.realAdd, Option.map₂, Option.bind, Option.map,
    cdfCarryCell, cdfCumsumTile, WithBot.unbotD_some]
  rw [Nat.mod_eq_of_lt e.isLt, Nat.mod_eq_of_lt p.isLt]

/-- **Carry-active match.** On the active region (`s.pids 0 * 64 + e < 64`,
`s.pids 1 * 64 + p < 64`) and for `c ≤ 4`, the concrete carry cell `cdfCarryCell c`
agrees with the genuine `stateValue c`. Proven by induction on `c` via
`cdfAdvance_active_of`. -/
theorem cdfCarry_active
    (s : BlockState) (k v d initial_state : RegionName) (USE_INITIAL_STATE : Bool)
    (hpids0 : s.pids 0 = 0)
    (c : Nat) (hc : c ≤ 4) (e p : Nat) (he : e < 64) (hp : s.pids 1 * 64 + p < 64) :
    cdfCarryCell s k v d initial_state USE_INITIAL_STATE c e p
      = cdfState s k v d initial_state USE_INITIAL_STATE c e p := by
  induction c generalizing e with
  | zero =>
      simp only [cdfCarryCell, cdfSeedCell, cdfState, stateValue, initElem,
        if_pos (show s.pids 0 * 64 + e < 64 ∧ s.pids 1 * 64 + p < 64 from
          ⟨by rw [hpids0]; omega, hp⟩)]
  | succ n ih =>
      have hn4 : n < 4 := by omega
      have hnle : n ≤ 4 := by omega
      simp only [cdfCarryCell]
      have hcell := cdfAdvance_active_of s k v d initial_state USE_INITIAL_STATE
        (fun e' p' => cdfCarryCell s k v d initial_state USE_INITIAL_STATE n e' p')
        n hn4 e p he hp
        (fun e' he'' => ih hnle e' he'')
      rw [show (⟨e % 64, Nat.mod_lt _ (by norm_num)⟩ : Fin 64) = (⟨e, by omega⟩ : Fin 64) from
            by simp [Nat.mod_eq_of_lt he],
          show (⟨p % 64, Nat.mod_lt _ (by norm_num)⟩ : Fin 64) = (⟨p, by omega⟩ : Fin 64) from
            by simp [Nat.mod_eq_of_lt (by omega : p < 64)]]
      rw [hcell, WithBot.unbotD_some]
      rw [ih (by omega) e he]
      -- cdfState n + (cdfState (n+1) - cdfState n) = cdfState (n+1)
      ring

/-! ### Outer-loop body step and carry invariant -/

/-- The `h[i_t]` block-ptr store offset at lane `(e,p)` (Python shape). -/
abbrev cdfHAddr (s : BlockState) (i_t : Nat) (idx : TileIndex [64, 64]) : Nat :=
  hOffset s i_t 16384 64 64 64 64 64 idx

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **Outer-loop body step.** Stepping `cdfOuterBody` from an entry carry state
`sin` (b_h = `cdfCarryTile c`, i_k=0, i_v/i_bh/i_t set, memory matching `s`)
advances `b_h` to `cdfCarryTile (c+1)`, writes the chunk-start state `cdfCarryCell c`
into `h[c]` at every active lane, writes the corrected `v_new[c]` block, and
preserves pids / `i_k`/`i_bh` / the `k`/`v`/`d` regions. -/
theorem cdfOuterBody_step
    (k v d v_new h initial_state : RegionName) (USE_INITIAL_STATE : Bool)
    (s sin : BlockState) (i_t : Nat) (hit : i_t < 4) (hpids0 : s.pids 0 = 0)
    (hVk : v_new ≠ k) (hVv : v_new ≠ v) (hVd : v_new ≠ d) (hHv : h ≠ v_new)
    (hHk : h ≠ k) (hHv2 : h ≠ v) (hHd : h ≠ d)
    (hInjV : Function.Injective (fun idx : TileIndex [32, 64] => cdfVNewAddr s i_t idx))
    (hInjH : Function.Injective (fun idx : TileIndex [64, 64] => cdfHAddr s i_t idx))
    (hmemK : ∀ off, sin.readMem k off = s.readMem k off)
    (hmemV : ∀ off, sin.readMem v off = s.readMem v off)
    (hmemD : ∀ off, sin.readMem d off = s.readMem d off)
    (hpids : sin.pids = s.pids)
    (hik : sin.regs .nat [] "i_k" = some (Tile.scalar 0))
    (hiv : sin.regs .nat [] "i_v" = some (Tile.scalar (s.pids 1)))
    (hibh : sin.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)))
    (hit2 : sin.regs .nat [] "i_t" = some (Tile.scalar i_t))
    (hbh : sin.regs .real [64, 64] "b_h"
        = some (cdfCarryTile s k v d initial_state USE_INITIAL_STATE i_t)) :
    ∃ s', stepStmts (cdfOuterBody k v d v_new h) sin = some s'
      ∧ s'.pids = sin.pids
      ∧ s'.regs .nat [] "i_k" = some (Tile.scalar 0)
      ∧ s'.regs .nat [] "i_v" = some (Tile.scalar (s.pids 1))
      ∧ s'.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2))
      ∧ s'.regs .real [64, 64] "b_h"
          = some (cdfCarryTile s k v d initial_state USE_INITIAL_STATE (i_t + 1))
      ∧ (∀ off, s'.readMem k off = s.readMem k off)
      ∧ (∀ off, s'.readMem v off = s.readMem v off)
      ∧ (∀ off, s'.readMem d off = s.readMem d off)
      -- h[i_t] active lanes hold the chunk-start carry cell
      ∧ (∀ idx : TileIndex [64, 64], active s 64 64 64 64 idx →
          s'.readMem h (cdfHAddr s i_t idx)
            = cdfCarryCell s k v d initial_state USE_INITIAL_STATE i_t idx.1.val idx.2.1.val)
      -- h off chunk-i_t block is unchanged
      ∧ (∀ off, (∀ idx : TileIndex [64, 64], off ≠ cdfHAddr s i_t idx) →
          s'.readMem h off = sin.readMem h off)
      -- v_new[i_t] active lanes hold the corrected value
      ∧ (∀ idx : TileIndex [32, 64],
          (i_t * 32 + 0 * 32 + idx.1.val < 128 ∧ s.pids 1 * 64 + idx.2.1.val < 64) →
          s'.readMem v_new (cdfVNewAddr s i_t idx)
            = cdfBvNewCell s v d
                (fun e' p' => cdfCarryCell s k v d initial_state USE_INITIAL_STATE i_t e' p')
                i_t idx.1.val idx.2.1.val)
      ∧ (∀ off, (∀ idx : TileIndex [32, 64], off ≠ cdfVNewAddr s i_t idx) →
          s'.readMem v_new off = sin.readMem v_new off) := by
  set fbh := fun e' p' => cdfCarryCell s k v d initial_state USE_INITIAL_STATE i_t e' p' with hfbh
  unfold cdfOuterBody
  -- p_h make
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (makeBlockPtr_2d_eval h sin _ _ _ [64, 64] [64, 64] [64, 1]
      (s.pids 2 * 16384 + i_t * 64 * 64) (0 * 64) (s.pids 1 * 64)
      (addMulMulMul_eval sin "i_bh" "i_t" (s.pids 2) 16384 i_t 64 64 hibh hit2)
      (mulConst_eval sin "i_k" 0 64 hik)
      (mulConst_eval sin "i_v" (s.pids 1) 64 hiv)))]
  -- h store (offsets [0, pids1*64] = [pids0*64, pids1*64] since pids0 = 0)
  rw [stepStmts.cons_some (cdfStore_h_step_eq s _ h i_t fbh
      (cdfCarryTile s k v d initial_state USE_INITIAL_STATE i_t)
      (fun e p => by rw [cdfCarryTile_data])
      (by simp [BlockState.setReg_same, hbh])
      (by simp [BlockState.setReg_same, hpids0]))]
  obtain ⟨hpidsH, hregsH, hvnewH, hotherH, hoffH⟩ :=
    cdfStore_h_step_props s
      (sin.setReg "p_h" .blockPtr [64, 64]
        (⟨fun _ => BlockPtr.mk h (s.pids 2 * 16384 + i_t * 64 * 64) [64, 64] [64, 64] [64, 1]
          [0 * 64, s.pids 1 * 64]⟩ : Tile .blockPtr [64, 64]))
      h i_t fbh hInjH
  -- b_h_cumsum = 0
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [64, 64] (Op.const 0)) _
        = some (⟨fun _ => some (0 : ℝ)⟩ : Tile .real [64, 64]) from by
      simp [evalOp_full, evalOp_const]))]
  -- name the p_h-setReg state by generalizing the block-ptr tile it stores
  generalize hsP : sin.setReg "p_h" .blockPtr [64, 64]
      (⟨fun _ => BlockPtr.mk h (s.pids 2 * 16384 + i_t * 64 * 64) [64, 64] [64, 64] [64, 1]
        [0 * 64, s.pids 1 * 64]⟩ : Tile .blockPtr [64, 64]) = sP at hpidsH hregsH hvnewH hotherH hoffH ⊢
  set sH := cdfHStoreState s sP h i_t fbh with hsH
  set sIn := sH.setReg "b_h_cumsum" .real [64, 64]
      (⟨fun _ => some (0 : ℝ)⟩ : Tile .real [64, 64]) with hsIn
  -- relate sP's registers/pids/memory to sin
  have hsP_pids : sP.pids = sin.pids := by rw [← hsP]; simp [BlockState.setReg_pids]
  have hsP_regs : ∀ {dt sh} (nm : RegName), nm ≠ "p_h" →
      sP.regs dt sh nm = sin.regs dt sh nm := by
    intro dt sh nm hnm; rw [← hsP, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hnm]
  have hsP_mem : ∀ rg off, sP.readMem rg off = sin.readMem rg off := by
    intro rg off; rw [← hsP, BlockState.setReg_readMem]
  -- memory of sIn: regions ≠ h match s (h-store only touches h)
  have hmemIn : ∀ rg, rg ≠ h → (∀ off, sin.readMem rg off = s.readMem rg off) →
      ∀ off, sIn.readMem rg off = s.readMem rg off := by
    intro rg hrg hrgmem off
    rw [hsIn, BlockState.setReg_readMem, hotherH rg off hrg, hsP_mem]
    exact hrgmem off
  -- sIn register/pids facts (peel b_h_cumsum setReg, h-store regs = sP regs, p_h setReg)
  have hsIn_pids : sIn.pids = s.pids := by
    rw [hsIn, BlockState.setReg_pids, hpidsH, hsP_pids]; exact hpids
  have hsIn_regs : ∀ {dt sh} (nm : RegName), nm ≠ "b_h_cumsum" → nm ≠ "p_h" →
      sIn.regs dt sh nm = sin.regs dt sh nm := by
    intro dt sh nm hnm hnm2
    rw [hsIn, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hnm, hregsH, hsP_regs nm hnm2]
  -- inner loop
  obtain ⟨sInner, hInnerStep, hpidsI, hikI, hivI, hibhI, hitI, hbhI, hcsI,
    hkI, hvI, hdI, hgenI, hvnewI, hothI⟩ :=
    cdfInnerLoop_run k v d v_new s sIn
      i_t fbh (cdfCarryTile s k v d initial_state USE_INITIAL_STATE i_t)
      (fun e p => by rw [cdfCarryTile_data])
      hVk hVv hVd hInjV
      (hmemIn k hHk.symm hmemK) (hmemIn v hHv2.symm hmemV) (hmemIn d hHd.symm hmemD)
      hsIn_pids
      (by rw [hsIn_regs "i_k" (by decide) (by decide)]; exact hik)
      (by rw [hsIn_regs "i_v" (by decide) (by decide)]; exact hiv)
      (by rw [hsIn_regs "i_bh" (by decide) (by decide)]; exact hibh)
      (by rw [hsIn_regs "i_t" (by decide) (by decide)]; exact hit2)
      (by rw [hsIn_regs "b_h" (by decide) (by decide)]; exact hbh)
      (by rw [hsIn, BlockState.setReg_same])
  rw [stepStmts.cons_some hInnerStep]
  -- advance
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cdfAdvance_eval s sInner k v d initial_state USE_INITIAL_STATE i_t hbhI hcsI))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · -- pids
    rw [BlockState.setReg_pids, hpidsI, hsIn_pids, hpids]
  · -- i_k
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hikI
  · -- i_v: preserved by inner loop, = sIn.regs i_v = sin.regs i_v
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hivI,
      hsIn_regs "i_v" (by decide) (by decide)]; exact hiv
  · -- i_bh
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hibhI
  · -- b_h advance
    simp only [BlockState.setReg_same]
  · -- k unchanged
    intro off; rw [BlockState.setReg_readMem]; exact hkI off
  · -- v unchanged
    intro off; rw [BlockState.setReg_readMem]; exact hvI off
  · -- d unchanged
    intro off; rw [BlockState.setReg_readMem]; exact hdI off
  · -- h active readback
    intro idx hidx
    rw [BlockState.setReg_readMem, hgenI h (cdfHAddr s i_t idx) hHv,
      hsIn, BlockState.setReg_readMem, hvnewH idx hidx]
  · -- h off-block unchanged
    intro off hoff
    rw [BlockState.setReg_readMem, hgenI h off hHv, hsIn, BlockState.setReg_readMem,
      hoffH off hoff, hsP_mem]
  · -- v_new active readback
    intro idx hidx; rw [BlockState.setReg_readMem]; exact hvnewI idx hidx
  · -- v_new off unchanged
    intro off hoff; rw [BlockState.setReg_readMem, hothI off hoff, hsIn,
      BlockState.setReg_readMem, hotherH v_new off hHv.symm, hsP_mem]

/-! ### Cross-chunk block disjointness

Distinct time chunks `j ≠ c` write disjoint `[64,64]` blocks of `h` (block size
`64*64 = 4096`) and disjoint `[32,64]` blocks of `v_new` (block size `32*64 =
2048`, stride `64`), so chunk `c`'s store leaves chunk `j`'s readback intact. -/

theorem cdfHAddr_chunk_disjoint (s : BlockState) (hpids0 : s.pids 0 = 0)
    (j c : Nat) (hjc : j ≠ c) (idxj idxc : TileIndex [64, 64]) :
    cdfHAddr s j idxj ≠ cdfHAddr s c idxc := by
  obtain ⟨⟨ej, hej⟩, ⟨pj, hpj⟩, _⟩ := idxj
  obtain ⟨⟨ec, hec⟩, ⟨pc, hpc⟩, _⟩ := idxc
  simp only [cdfHAddr, hOffset, kIndex, vIndex, hpids0]
  intro heq
  -- pids2*16384 + j*4096 + (0*64+ej)*64 + (pids1*64+pj) = ... c ...
  -- in-block offset ej*64+pj < 4096; blocks at j*4096 vs c*4096 are disjoint
  apply hjc
  nlinarith [hej, hpj, hec, hpc]

theorem cdfVNewAddr_chunk_disjoint (s : BlockState)
    (j c : Nat) (hjc : j ≠ c) (idxj idxc : TileIndex [32, 64]) :
    cdfVNewAddr s j idxj ≠ cdfVNewAddr s c idxc := by
  obtain ⟨⟨cj, hcj⟩, ⟨pj, hpj⟩, _⟩ := idxj
  obtain ⟨⟨cc, hcc⟩, ⟨pc, hpc⟩, _⟩ := idxc
  simp only [cdfVNewAddr]
  intro heq
  apply hjc
  -- (j*32+cj)*64 + (pids1*64+pj) = (c*32+cc)*64 + (pids1*64+pc); cj,cc<32, pj,pc<64
  nlinarith [hcj, hpj, hcc, hpc]

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **Outer-loop carry.** Driving the static `forRange "i_t" 0 4 1 cdfOuterBody`
from the post-prologue state `s0` (b_h = seed = `cdfCarryTile 0`) folds the state
across all 4 chunks: every chunk `j < 4` has its chunk-start carry cell stored into
`h[j]` and its corrected value into `v_new[j]` (at active lanes), with `b_h`
holding `cdfCarryTile 4` at the end, and the `k`/`v`/`d` regions preserved. -/
theorem cdfOuterLoop_run
    (k v d v_new h initial_state : RegionName) (USE_INITIAL_STATE : Bool)
    (s s0 : BlockState) (hpids0 : s.pids 0 = 0)
    (hVk : v_new ≠ k) (hVv : v_new ≠ v) (hVd : v_new ≠ d) (hHv : h ≠ v_new)
    (hHk : h ≠ k) (hHv2 : h ≠ v) (hHd : h ≠ d)
    (hInjV : ∀ i_t : Fin 4,
      Function.Injective (fun idx : TileIndex [32, 64] => cdfVNewAddr s i_t.val idx))
    (hInjH : ∀ i_t : Fin 4,
      Function.Injective (fun idx : TileIndex [64, 64] => cdfHAddr s i_t.val idx))
    (hmem : ∀ rg off, s0.readMem rg off = s.readMem rg off)
    (hpids : s0.pids = s.pids)
    (hik : s0.regs .nat [] "i_k" = some (Tile.scalar 0))
    (hiv : s0.regs .nat [] "i_v" = some (Tile.scalar (s.pids 1)))
    (hibh : s0.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)))
    (hbh : s0.regs .real [64, 64] "b_h"
        = some (cdfCarryTile s k v d initial_state USE_INITIAL_STATE 0)) :
    ∃ s', stepStmt (Stmt.forRange "i_t" 0 4 1 (cdfOuterBody k v d v_new h)) s0 = some s'
      ∧ s'.pids = s.pids
      ∧ s'.regs .real [64, 64] "b_h"
          = some (cdfCarryTile s k v d initial_state USE_INITIAL_STATE 4)
      ∧ (∀ off, s'.readMem k off = s.readMem k off)
      ∧ (∀ off, s'.readMem v off = s.readMem v off)
      ∧ (∀ off, s'.readMem d off = s.readMem d off)
      ∧ (∀ j : Fin 4, ∀ idx : TileIndex [64, 64], active s 64 64 64 64 idx →
          s'.readMem h (cdfHAddr s j.val idx)
            = cdfCarryCell s k v d initial_state USE_INITIAL_STATE j.val idx.1.val idx.2.1.val)
      ∧ (∀ j : Fin 4, ∀ idx : TileIndex [32, 64],
          (j.val * 32 + 0 * 32 + idx.1.val < 128 ∧ s.pids 1 * 64 + idx.2.1.val < 64) →
          s'.readMem v_new (cdfVNewAddr s j.val idx)
            = cdfBvNewCell s v d
                (fun e' p' => cdfCarryCell s k v d initial_state USE_INITIAL_STATE j.val e' p')
                j.val idx.1.val idx.2.1.val) := by
  obtain ⟨final, s', hstep, hfinal, hP⟩ :=
    forRange_inv (idx := "i_t") (start := 0) (stop := 4) (step := 1)
      (s_init := s0)
      (P := fun c st =>
        c ≤ 4
        ∧ st.pids = s.pids
        ∧ st.regs .nat [] "i_k" = some (Tile.scalar 0)
        ∧ st.regs .nat [] "i_v" = some (Tile.scalar (s.pids 1))
        ∧ st.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2))
        ∧ st.regs .real [64, 64] "b_h"
            = some (cdfCarryTile s k v d initial_state USE_INITIAL_STATE c)
        ∧ (∀ off, st.readMem k off = s.readMem k off)
        ∧ (∀ off, st.readMem v off = s.readMem v off)
        ∧ (∀ off, st.readMem d off = s.readMem d off)
        ∧ (∀ j, j < c → ∀ idx : TileIndex [64, 64], active s 64 64 64 64 idx →
            st.readMem h (cdfHAddr s j idx)
              = cdfCarryCell s k v d initial_state USE_INITIAL_STATE j idx.1.val idx.2.1.val)
        ∧ (∀ j, j < c → ∀ idx : TileIndex [32, 64],
            (j * 32 + 0 * 32 + idx.1.val < 128 ∧ s.pids 1 * 64 + idx.2.1.val < 64) →
            st.readMem v_new (cdfVNewAddr s j idx)
              = cdfBvNewCell s v d
                  (fun e' p' => cdfCarryCell s k v d initial_state USE_INITIAL_STATE j e' p')
                  j idx.1.val idx.2.1.val))
      (by norm_num)
      ⟨by norm_num, hpids, hik, hiv, hibh, hbh, hmem k, hmem v, hmem d,
        fun j hj => absurd hj (Nat.not_lt_zero j), fun j hj => absurd hj (Nat.not_lt_zero j)⟩
      (fun c st hlt hPc => by
        obtain ⟨hcle, hpidsC, hikC, hivC, hibhC, hbhC, hkC, hvC, hdC, hhC, hvnC⟩ := hPc
        have hc4 : c < 4 := hlt
        -- step the outer body for chunk c
        set sc := st.setReg "i_t" .nat [] (Tile.scalar c) with hsc
        obtain ⟨s', hbody, hpidsB, hikB, hivB, hibhB, hbhB,
          hkB, hvB, hdB, hhActB, hhOffB, hvnActB, hvnOffB⟩ :=
          cdfOuterBody_step k v d v_new h initial_state USE_INITIAL_STATE s sc c hc4 hpids0
            hVk hVv hVd hHv hHk hHv2 hHd (hInjV ⟨c, hc4⟩) (hInjH ⟨c, hc4⟩)
            (by intro off; rw [hsc, BlockState.setReg_readMem]; exact hkC off)
            (by intro off; rw [hsc, BlockState.setReg_readMem]; exact hvC off)
            (by intro off; rw [hsc, BlockState.setReg_readMem]; exact hdC off)
            (by rw [hsc, BlockState.setReg_pids]; exact hpidsC)
            (by rw [hsc, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hikC)
            (by rw [hsc, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hivC)
            (by rw [hsc, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hibhC)
            (by rw [hsc, BlockState.setReg_same])
            (by rw [hsc, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hbhC)
        refine ⟨s', hbody, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
        · omega
        · rw [hpidsB, hsc, BlockState.setReg_pids]; exact hpidsC
        · exact hikB
        · exact hivB
        · exact hibhB
        · exact hbhB
        · intro off; rw [hkB]
        · intro off; rw [hvB]
        · intro off; rw [hdB]
        · -- h readback for chunks < c+1
          intro j hj idx hact
          rcases Nat.lt_succ_iff_lt_or_eq.mp hj with hjlt | hjeq
          · -- j < c: preserved (chunk c's store off chunk j's block)
            rw [hhOffB (cdfHAddr s j idx) (fun idxc =>
              cdfHAddr_chunk_disjoint s hpids0 j c (by omega) idx idxc)]
            rw [hsc, BlockState.setReg_readMem]
            exact hhC j hjlt idx hact
          · subst hjeq; exact hhActB idx hact
        · -- v_new readback for chunks < c+1
          intro j hj idx hact
          rcases Nat.lt_succ_iff_lt_or_eq.mp hj with hjlt | hjeq
          · rw [hvnOffB (cdfVNewAddr s j idx) (fun idxc =>
              cdfVNewAddr_chunk_disjoint s j c (by omega) idx idxc)]
            rw [hsc, BlockState.setReg_readMem]
            exact hvnC j hjlt idx hact
          · subst hjeq; exact hvnActB idx hact)
  obtain ⟨hcleF, hpidsF, _, _, _, hbhF, hkF, hvF, hdF, hhF, hvnF⟩ := hP
  have hfinalEq : final = 4 := le_antisymm hcleF hfinal
  subst hfinalEq
  exact ⟨s', hstep, hpidsF, hbhF, hkF, hvF, hdF,
    fun j idx hact => hhF j.val j.isLt idx hact,
    fun j idx hact => hvnF j.val j.isLt idx hact⟩

end VeriTile.Bench.TritonBenchG.ChunkDeltaFwd
