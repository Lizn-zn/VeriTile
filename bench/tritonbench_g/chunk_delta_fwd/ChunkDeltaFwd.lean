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

end VeriTile.Bench.TritonBenchG.ChunkDeltaFwd
