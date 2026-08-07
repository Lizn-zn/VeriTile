import VeriTile.Triton

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

This file executes the kernel and proves its stored outputs equal a **genuine
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

## Scope — genuine end-to-end, but shape-pinned

The headline `chunk_delta_fwd_exec_genuine` verifies the per-program
`@triton.jit` body **end-to-end from `exec`**: it runs the whole surface
(prologue + outer `NT` loop + epilogue) and reads the genuine closed form back
out of the output buffers, with **no producer hypotheses**. The price is that it
is pinned to the checked Python shape (all dimensions and strides are concrete
literals) and to `s.pids 0 = 0` (the `i_k = 0` regime the host's `NK == 1`
assertion creates). Per `bench/MAIN_THEOREM_CONVENTIONS.md` §6 that trade —
genuine-but-scoped over general-but-assumed — is the intended one.

The host launch (`chunk_delta_rule_fwd_kernel_h[(NK, NV, B*H)]`, the 3-D grid,
the autotuned warp counts, the host-computed `BK/BV/BC/NT` and the `NK == 1`
assertion) is the *trusted boundary*. Program ids 1 and 2 remain universally
quantified, so the statement covers every program with `i_k = 0`.

## Proof architecture

```
chunk_delta_fwd_exec_genuine                       ← TOP THEOREM (exec-level, shape-pinned)
  ├─ cdfPrologue_run          seed / block-ptr prologue
  ├─ cdfOuterLoop_run         the NT-chunk carry fold, derived — not assumed
  │    ├─ cdfCarry_active     concrete carry cell = genuine `stateValue`
  │    └─ cdfBvNew_eq_vNewValue   concrete corrected value = genuine `vNewValue`
  └─ cdfEpilogue_run          the `STORE_FINAL_STATE` flush
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune`
(`num_warps`) is not modeled. The headline fixes the checked Python shape
(`B,H,T,K,V = 2,4,128,64,64`, `BT = BC = 32`, `BK = BV = 64`, `NT = 4`) and
covers both flag settings symbolically, so the benchmark's case 1 (no
initial/final state) and case 2 (with) are both instances. The dynamic `.to(...)`
casts erase to the identity at the algorithm layer.

Region distinctness is a **real** precondition here and is stated as twelve
explicit `≠` hypotheses: the fold performs many stores interleaved with loads, so
`v_new`, `h` and `final_state` must not alias each other or the inputs. Three
output-offset injectivity hypotheses complete the side conditions.

Deleted rather than kept: three `*_store_slice_realizes_*` faces and the
dimension-general `chunk_delta_fwd_output_summary_general` headline built on them.
Each of those slices was a masked memcpy whose load and store addresses were
character-identical, so each face's entire content was its own producer
hypothesis `hBH`/`hBVN`/`hBHF`. The old prose claimed those hypotheses were
"discharged end-to-end from the kernel `exec` … by `chunk_delta_fwd_exec_genuine`";
they were not, and could not be — `exec_genuine` concludes about `sF.readMem h`
(final state, real region) while `hBH` constrains `s.readMem BH` (initial state,
fiction region), and it is additionally shape-pinned and needs `pids 0 = 0`.
There was no Lean term linking them.

Still not covered: the multi-inner-chunk regime `BC < BT` (the benchmark has
`BC = BT = 32`, hence `ceil(BT/BC) = 1`), and any shape other than the checked
one.
-/

namespace VeriTile.Bench.TritonBenchG.ChunkDeltaFwd

open VeriTile.Triton
open scoped VeriTile.Triton.Masked3DTileKernelIO₁

set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

/-! **★ Main theorem:** `chunk_delta_fwd_exec_genuine` — exec-level and genuine
end-to-end (no producer hypotheses), scoped to the checked Python shape and to
`s.pids 0 = 0`. -/

/-! # ══════════ CORRECT — genuine / dimension-general (review this) ══════════ -/

section Correct_without_Rounding

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

def hOffset (s : BlockState) (i_t s_h_h s_h_t K V BK BV : Nat)
    (idx : TileIndex [BK, BV]) : Nat :=
  s.pids 2 * s_h_h + i_t * K * V +
    kIndex s BK idx.1 * s_h_t + vIndex s BV idx.2.1

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
def finalStateOffset (s : BlockState) (K V BK BV : Nat)
    (idx : TileIndex [BK, BV]) : Nat :=
  s.pids 2 * K * V + kIndex s BK idx.1 * V + vIndex s BV idx.2.1

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
      ∧ s'.regs .nat [] "i_k" = some (Tile.scalar 0)
      ∧ s'.regs .nat [] "i_v" = some (Tile.scalar (s.pids 1))
      ∧ s'.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2))
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
  obtain ⟨hcleF, hpidsF, hikF, hivF, hibhF, hbhF, hkF, hvF, hdF, hhF, hvnF⟩ := hP
  have hfinalEq : final = 4 := le_antisymm hcleF hfinal
  subst hfinalEq
  exact ⟨s', hstep, hpidsF, hikF, hivF, hibhF, hbhF, hkF, hvF, hdF,
    fun j idx hact => hhF j.val j.isLt idx hact,
    fun j idx hact => hvnF j.val j.isLt idx hact⟩

/-! ### Prologue (`USE_INITIAL_STATE` seed) and epilogue (`STORE_FINAL_STATE` flush) -/

/-- An `ifThen` with a `false` constexpr condition is a no-op. -/
theorem cdf_ifThen_false_noop (body : List Stmt) (X : BlockState) :
    stepStmt (Stmt.ifThen (Op.constBool Bool.false) body) X = some X := by
  simp [stepStmt, evalOp]

/-- An `ifThen` with a `true` constexpr condition runs its body. -/
theorem cdf_ifThen_true_run (body : List Stmt) (X : BlockState) :
    stepStmt (Stmt.ifThen (Op.constBool Bool.true) body) X = stepStmts body X := by
  simp [stepStmt, evalOp]

/-- The all-zero `[64,64]` tile equals `cdfCarryTile 0` when `USE_INITIAL_STATE`
is `false` (the seed is `0` on every lane). -/
theorem cdfCarryTile_zero_false (s : BlockState) (k v d initial_state : RegionName) :
    (⟨fun _ => some (0 : ℝ)⟩ : Tile .real [64, 64])
      = cdfCarryTile s k v d initial_state Bool.false 0 := by
  refine congrArg _ ?_
  funext idx; obtain ⟨e, p, u⟩ := idx
  simp only [cdfCarryTile, cdfCarryCell, cdfSeedCell, Bool.false_eq_true, if_false, ite_self]

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **Prologue.** Stepping `cdfPrologue` from `s` (`NK = 1` regime, `s.pids 0 = 0`)
reaches a state `s0` whose program-id registers are set, `b_h` holds the seed
`cdfCarryTile 0`, memory and pids are preserved. Both the `USE_INITIAL_STATE = true`
(masked `initial_state` load) and `false` (zero seed) branches are covered. -/
theorem cdfPrologue_run
    (k v d v_new h initial_state final_state : RegionName) (USE_INITIAL_STATE : Bool)
    (s : BlockState) (hpids0 : s.pids 0 = 0) :
    ∃ s0, stepStmts (cdfPrologue initial_state USE_INITIAL_STATE) s = some s0
      ∧ s0.pids = s.pids
      ∧ (∀ rg off, s0.readMem rg off = s.readMem rg off)
      ∧ s0.regs .nat [] "i_k" = some (Tile.scalar 0)
      ∧ s0.regs .nat [] "i_v" = some (Tile.scalar (s.pids 1))
      ∧ s0.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2))
      ∧ s0.regs .real [64, 64] "b_h"
          = some (cdfCarryTile s k v d initial_state USE_INITIAL_STATE 0) := by
  unfold cdfPrologue
  -- i_k = program_id 0 = pids 0 = 0
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.programId 0) s = some (Tile.scalar 0) from by simp [hpids0]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.programId 1) _ = some (Tile.scalar (s.pids 1)) from by simp))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.programId 2) _ = some (Tile.scalar (s.pids 2)) from by simp))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [64, 64] (Op.const 0)) _
        = some (⟨fun _ => some (0 : ℝ)⟩ : Tile .real [64, 64]) from by
      simp [evalOp_full, evalOp_const]))]
  cases USE_INITIAL_STATE
  · -- false: ifThen false = no-op
    rw [stepStmts.cons_some (cdf_ifThen_false_noop _ _)]
    rw [stepStmts.nil]
    refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · simp [BlockState.setReg_pids]
    · intro rg off; simp [BlockState.setReg_readMem]
    · simp [BlockState.setReg_ne_name, BlockState.setReg_same]
    · simp [BlockState.setReg_ne_name, BlockState.setReg_same]
    · simp [BlockState.setReg_ne_name, BlockState.setReg_same]
    · rw [BlockState.setReg_same, cdfCarryTile_zero_false s k v d initial_state]
  · -- true: ifThen true: run the seed body (p_h0 make + masked load)
    rw [stepStmts.cons_some (st := Stmt.ifThen (Op.constBool Bool.true) _)
      (by
        rw [cdf_ifThen_true_run]
        -- p_h0 make
        rw [stepStmts.cons_some (stepStmt_assign_eq_some
          (makeBlockPtr_2d_eval initial_state _ _ _ _ [64, 64] [64, 64] [64, 1]
            (s.pids 2 * 64 * 64) (0 * 64) (s.pids 1 * 64)
            (mulMulConst_eval _ "i_bh" (s.pids 2) 64 64 (by simp [BlockState.setReg_ne_name]))
            (mulConst_eval _ "i_k" 0 64 (by simp [BlockState.setReg_ne_name]))
            (mulConst_eval _ "i_v" (s.pids 1) 64 (by simp [BlockState.setReg_ne_name]))))]
        -- b_h masked load
        rw [stepStmts.cons_some (stepStmt_assign_eq_some
          (load_bp_2d_ref initial_state _ "p_h0" (s.pids 2 * 64 * 64) 64 64 64 64 64 1
            (0 * 64) (s.pids 1 * 64) (by simp)))]
        rw [stepStmts.nil])]
    rw [stepStmts.nil]
    refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · simp [BlockState.setReg_pids]
    · intro rg off; simp [BlockState.setReg_readMem]
    · simp [BlockState.setReg_ne_name, BlockState.setReg_same]
    · simp [BlockState.setReg_ne_name, BlockState.setReg_same]
    · simp [BlockState.setReg_ne_name, BlockState.setReg_same]
    · -- b_h = masked load tile = cdfCarryTile 0 (USE = true)
      rw [BlockState.setReg_same]
      refine congrArg some ?_
      refine congrArg _ ?_
      funext idx; obtain ⟨e, p, u⟩ := idx
      simp only [cdfCarryTile, cdfCarryCell, cdfSeedCell, initElem, hpids0,
        Nat.zero_mul, Nat.zero_add, Nat.mul_one, BlockState.setReg_readMem]
      by_cases hb : e.val < 64 ∧ s.pids 1 * 64 + p.val < 64
      · rw [if_pos hb, if_pos hb]; simp
      · rw [if_neg hb, if_neg hb]

/-- The explicit post-store state of the `final_state` block-ptr store of the
carry tile `bhT` (cell fn `fbh`) over input state `sin`, value built from `s`. -/
noncomputable def cdfFinalStoreState (s sin : BlockState) (final_state : RegionName)
    (fbh : Nat → Nat → ℝ) : BlockState :=
  (TileShape.allIndices [64, 64]).foldl
    (fun acc i => if (s.pids 0 * 64 + i.1.val < 64 ∧ s.pids 1 * 64 + i.2.1.val < 64)
        then acc.writeMem final_state (finalStateOffset s 64 64 64 64 i)
          (fbh i.1.val i.2.1.val) else acc) sin

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **`final_state` block-ptr store step (eq).** Stepping the masked block-ptr
store of the carry tile `bhT` (cell fn `fbh`) through `p_ht` yields
`cdfFinalStoreState`. -/
theorem cdfStoreFinal_step_eq (s sin : BlockState) (final_state : RegionName)
    (fbh : Nat → Nat → ℝ) (bhT : Tile .real [64, 64])
    (hbhf : ∀ e p, bhT.data (e, p, PUnit.unit) = some (fbh e.val p.val))
    (hbh : sin.regs .real [64, 64] "b_h" = some bhT)
    (hph : sin.regs .blockPtr [64, 64] "p_ht" = some
      ⟨fun _ => BlockPtr.mk final_state (s.pids 2 * 64 * 64) [64, 64] [64, 64] [64, 1]
        [s.pids 0 * 64, s.pids 1 * 64]⟩) :
    stepStmt (Stmt.store .real [64, 64]
        (.blockPtr (Op.ref .blockPtr [64, 64] "p_ht") [0, 1])
        (Op.ref .real [64, 64] "b_h") .none) sin
      = some (cdfFinalStoreState s sin final_state fbh) := by
  unfold stepStmt cdfFinalStoreState
  simp only [evalOp_ref, hbh, hph, Option.bind, Option.map]
  refine congrArg some (congrArg (fun f => List.foldl f sin (TileShape.allIndices [64, 64])) ?_)
  funext acc i
  obtain ⟨e, p, u⟩ := i
  simp only [TileShape.indexToList, BlockPtr.address_2d_offsets, BlockPtr.inBounds_2d_offsets,
    Bool.true_and, finalStateOffset, kIndex, vIndex]
  by_cases hb : s.pids 0 * 64 + e.val < 64 ∧ s.pids 1 * 64 + p.val < 64
  · simp only [hb, decide_true, if_true, BlockState.writeMemTyped_real, hbhf, Nat.mul_one]
    rfl
  · simp only [hb, decide_false, Bool.false_eq_true, if_false, if_neg hb]

set_option maxHeartbeats 4000000 in
/-- **`final_state` store readback (active lanes).** -/
theorem cdfStoreFinal_step_active (s sin : BlockState) (final_state : RegionName)
    (fbh : Nat → Nat → ℝ)
    (hInj : Function.Injective (fun idx : TileIndex [64, 64] => finalStateOffset s 64 64 64 64 idx)) :
    ∀ idx : TileIndex [64, 64], active s 64 64 64 64 idx →
      (cdfFinalStoreState s sin final_state fbh).readMem final_state
          (finalStateOffset s 64 64 64 64 idx)
        = fbh idx.1.val idx.2.1.val := by
  classical
  intro idx hidx
  unfold cdfFinalStoreState
  have h := BlockState.scatter_readback_prop_masked_nd (region := final_state) sin
    (fun idx : TileIndex [64, 64] => finalStateOffset s 64 64 64 64 idx)
    (fun idx => fbh idx.1.val idx.2.1.val)
    (fun idx => s.pids 0 * 64 + idx.1.val < 64 ∧ s.pids 1 * 64 + idx.2.1.val < 64)
    hInj idx
  rw [h, if_pos (show s.pids 0 * 64 + idx.1.val < 64 ∧ s.pids 1 * 64 + idx.2.1.val < 64 from
    by obtain ⟨ka, va⟩ := hidx; exact ⟨by simpa [kIndex] using ka, by simpa [vIndex] using va⟩)]

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **Epilogue.** Stepping `cdfEpilogue` from the post-loop state `sF`
(`b_h = cdfCarryTile 4`, `s.pids 0 = 0`) either leaves memory untouched
(`STORE_FINAL_STATE = false`) or stores `H_4 = cdfCarryCell 4` into `final_state`
at every active lane (`STORE_FINAL_STATE = true`). The `k`/`v`/`d`/`h`/`v_new`
regions and pids are preserved in both cases. -/
theorem cdfEpilogue_run
    (k v d v_new h initial_state final_state : RegionName) (USE_INITIAL_STATE STORE_FINAL_STATE : Bool)
    (s sF : BlockState) (hpids0 : s.pids 0 = 0)
    (hFh : final_state ≠ h) (hFv : final_state ≠ v_new) (hFk : final_state ≠ k)
    (hFv2 : final_state ≠ v) (hFd : final_state ≠ d)
    (hInjF : Function.Injective (fun idx : TileIndex [64, 64] => finalStateOffset s 64 64 64 64 idx))
    (hpids : sF.pids = s.pids)
    (hik : sF.regs .nat [] "i_k" = some (Tile.scalar 0))
    (hiv : sF.regs .nat [] "i_v" = some (Tile.scalar (s.pids 1)))
    (hibh : sF.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)))
    (hbh : sF.regs .real [64, 64] "b_h"
        = some (cdfCarryTile s k v d initial_state USE_INITIAL_STATE 4)) :
    ∃ s', stepStmts (cdfEpilogue final_state STORE_FINAL_STATE) sF = some s'
      ∧ (∀ rg off, rg ≠ final_state → s'.readMem rg off = sF.readMem rg off)
      ∧ (STORE_FINAL_STATE = Bool.true →
          ∀ idx : TileIndex [64, 64], active s 64 64 64 64 idx →
            s'.readMem final_state (finalStateOffset s 64 64 64 64 idx)
              = cdfCarryCell s k v d initial_state USE_INITIAL_STATE 4 idx.1.val idx.2.1.val) := by
  unfold cdfEpilogue
  cases STORE_FINAL_STATE
  · -- false: no-op
    rw [stepStmts.cons_some (cdf_ifThen_false_noop _ _), stepStmts.nil]
    exact ⟨_, rfl, fun rg off _ => rfl, fun h => absurd h (by simp)⟩
  · -- true: store H_4 into final_state
    rw [stepStmts.cons_some (st := Stmt.ifThen (Op.constBool Bool.true) _)
      (by
        rw [cdf_ifThen_true_run]
        rw [stepStmts.cons_some (stepStmt_assign_eq_some
          (makeBlockPtr_2d_eval final_state _ _ _ _ [64, 64] [64, 64] [64, 1]
            (s.pids 2 * 64 * 64) (0 * 64) (s.pids 1 * 64)
            (mulMulConst_eval _ "i_bh" (s.pids 2) 64 64 (by simp [BlockState.setReg_ne_name, hibh]))
            (mulConst_eval _ "i_k" 0 64 (by simp [BlockState.setReg_ne_name, hik]))
            (mulConst_eval _ "i_v" (s.pids 1) 64 (by simp [BlockState.setReg_ne_name, hiv]))))]
        rw [stepStmts.cons_some (cdfStoreFinal_step_eq s _ final_state
          (fun e' p' => cdfCarryCell s k v d initial_state USE_INITIAL_STATE 4 e' p')
          (cdfCarryTile s k v d initial_state USE_INITIAL_STATE 4)
          (fun e p => by rw [cdfCarryTile_data])
          (by simp [BlockState.setReg_ne_name, BlockState.setReg_same, hbh])
          (by simp [BlockState.setReg_same, hpids0]))]
        rw [stepStmts.nil])]
    rw [stepStmts.nil]
    -- name the post-store state
    generalize hsFin : sF.setReg "p_ht" .blockPtr [64, 64]
      (⟨fun _ => BlockPtr.mk final_state (s.pids 2 * 64 * 64) [64, 64] [64, 64] [64, 1]
        [0 * 64, s.pids 1 * 64]⟩ : Tile .blockPtr [64, 64]) = sPt
    refine ⟨_, rfl, ?_, ?_⟩
    · intro rg off hrg
      unfold cdfFinalStoreState
      rw [BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
        final_state _ _ _ _ _ rg off hrg, ← hsFin, BlockState.setReg_readMem]
    · intro _ idx hidx
      exact cdfStoreFinal_step_active s sPt final_state
        (fun e' p' => cdfCarryCell s k v d initial_state USE_INITIAL_STATE 4 e' p')
        hInjF idx hidx

/-! ### Bridges: concrete carry cells = genuine closed forms (active lanes) -/

/-- On the active region, the concrete corrected-value cell `cdfBvNewCell` built
from the carry equals the genuine `vNewValue` (the `i_c = 0`, `BC = BT` regime). -/
theorem cdfBvNew_eq_vNewValue
    (s : BlockState) (k v d initial_state : RegionName) (USE_INITIAL_STATE : Bool)
    (hpids0 : s.pids 0 = 0) (i_t : Nat) (hit : i_t < 4)
    (c p : Nat) (hc : i_t * 32 + c < 128) (hp : s.pids 1 * 64 + p < 64) :
    cdfBvNewCell s v d
        (fun e' p' => cdfCarryCell s k v d initial_state USE_INITIAL_STATE i_t e' p') i_t c p
      = vNewValue s k v d initial_state 8192 128 1 8192 64 1 64 64 32 64 64
          USE_INITIAL_STATE i_t c p := by
  simp only [cdfBvNewCell, vNewValue]
  rw [if_pos (⟨hc, hp⟩ : i_t * 32 + c < 128 ∧ s.pids 1 * 64 + p < 64)]
  refine congrArg (vElem s v 8192 64 1 32 64 i_t c p - ·) ?_
  refine Finset.sum_congr rfl (fun e' _ => ?_)
  rw [if_pos ⟨hc, e'.isLt⟩,
    cdfCarry_active s k v d initial_state USE_INITIAL_STATE hpids0 i_t (by omega) e'.val p
      e'.isLt hp]
  rfl

/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **SCOPE — genuine and end-to-end over the launched surface, but pinned to the
checked Python shape and to `i_k = 0`.** Read the two scope conditions first:

* every dimension and stride is a **concrete literal**
  (`s_qk_h = 8192, T = 128, K = V = 64, BT = BC = 32, BK = BV = 64, NT = 4`, …),
  matching the benchmark's `B,H,T,K,V = 2,4,128,64,64`, `BT = 32`;
* `hpids0 : s.pids 0 = 0` pins the key-axis program id, which the host guarantees
  via its `NK == 1` assertion but which is *not* proven here.

Per `bench/MAIN_THEOREM_CONVENTIONS.md` §6 this is the blessed trade: genuine and
scoped beats dimension-general and assumed. What it buys, and what the previous
headline did not have: **no producer hypotheses at all.** Executing the entire
`chunk_delta_rule_fwd_h_surface` (prologue + outer `forRange` over `NT = 4` +
epilogue) writes the genuine delta-rule closed form into the output buffers —
`h[j]` holds the chunk-start state `hValue j`, `v_new[j]` the corrected
`vNewSpec j`, and, when `STORE_FINAL_STATE`, `final_state` holds `finalValue 4` —
at every active lane, with the whole cross-chunk carry fold *derived from the
kernel `exec`* rather than assumed. The closed forms `hValue` / `vNewSpec` /
`finalValue` are over the input memory `k`/`v`/`d`/`initial_state`; none is an
exec read-back.

Honest hypotheses: the two scope pins above, twelve **region-distinctness**
(`≠`) hypotheses — the outputs `v_new`, `h`, `final_state` must not alias each
other or the inputs, which the store-order-sensitive multi-store fold genuinely
needs — and three output-offset injectivity hypotheses. The `USE_INITIAL_STATE`
and `STORE_FINAL_STATE` flags flow through symbolically.

Not covered: the host launch (the 3-D grid, autotuned warp counts, host-computed
`BK/BV/BC/NT`, and the `NK == 1` assertion), and the multi-inner-chunk regime
`BC < BT` (here `BC = BT = 32`, so `ceil(BT/BC) = 1`). -/
specification chunk_delta_fwd_exec_genuine
    (k v d v_new h initial_state final_state : RegionName)
    (USE_INITIAL_STATE STORE_FINAL_STATE : Bool) (s : BlockState) (hpids0 : s.pids 0 = 0)
    (hVk : v_new ≠ k) (hVv : v_new ≠ v) (hVd : v_new ≠ d) (hHv : h ≠ v_new)
    (hHk : h ≠ k) (hHv2 : h ≠ v) (hHd : h ≠ d)
    (hFh : final_state ≠ h) (hFv : final_state ≠ v_new) (hFk : final_state ≠ k)
    (hFv2 : final_state ≠ v) (hFd : final_state ≠ d)
    (hInjV : ∀ i_t : Fin 4,
      Function.Injective (fun idx : TileIndex [32, 64] => cdfVNewAddr s i_t.val idx))
    (hInjH : ∀ i_t : Fin 4,
      Function.Injective (fun idx : TileIndex [64, 64] => cdfHAddr s i_t.val idx))
    (hInjF : Function.Injective
      (fun idx : TileIndex [64, 64] => finalStateOffset s 64 64 64 64 idx)) :
    ∃ sF, exec (chunk_delta_rule_fwd_h_surface k v d v_new h initial_state final_state
        8192 128 1 8192 64 1 16384 64 4 128 64 64 32 32 64 64 4
        USE_INITIAL_STATE STORE_FINAL_STATE).toAlgKernel s = some sF
      ∧ (∀ j : Fin 4, ∀ idx : TileIndex [64, 64], active s 64 64 64 64 idx →
          sF.readMem h (cdfHAddr s j.val idx)
            = hValue s k v d initial_state 8192 128 1 8192 64 1 64 64 32 64 64
                USE_INITIAL_STATE j.val idx)
      ∧ (∀ j : Fin 4, ∀ idx : TileIndex [32, 64],
          vNewActive s j.val 0 128 64 32 32 64 idx →
          sF.readMem v_new (cdfVNewAddr s j.val idx)
            = vNewSpec s k v d initial_state 8192 128 1 8192 64 1 64 64 32 64 64
                USE_INITIAL_STATE j.val idx)
      ∧ (STORE_FINAL_STATE = Bool.true →
          ∀ idx : TileIndex [64, 64], active s 64 64 64 64 idx →
            sF.readMem final_state (finalStateOffset s 64 64 64 64 idx)
              = finalValue s k v d initial_state 8192 128 1 8192 64 1 64 64 32 64 64
                  USE_INITIAL_STATE 4 idx) := by
  rw [exec, chunk_delta_fwd_body_split]
  -- prologue
  obtain ⟨s0, hpre, hpidsP, hmemP, hikP, hivP, hibhP, hbhP⟩ :=
    cdfPrologue_run k v d v_new h initial_state final_state USE_INITIAL_STATE s hpids0
  -- outer loop
  obtain ⟨sL, hloop, hpidsL, hikL, hivL, hibhL, hbhL, hkL, hvL, hdL, hhL, hvnL⟩ :=
    cdfOuterLoop_run k v d v_new h initial_state USE_INITIAL_STATE s s0 hpids0
      hVk hVv hVd hHv hHk hHv2 hHd hInjV hInjH hmemP hpidsP hikP hivP hibhP hbhP
  -- epilogue
  obtain ⟨sE, hepi, hmemE, hfinE⟩ :=
    cdfEpilogue_run k v d v_new h initial_state final_state USE_INITIAL_STATE STORE_FINAL_STATE
      s sL hpids0 hFh hFv hFk hFv2 hFd hInjF hpidsL hikL hivL hibhL hbhL
  -- assemble the full body: (prologue) ++ (forRange :: epilogue)
  rw [List.append_assoc, stepStmts.append_some hpre]
  rw [show [Stmt.forRange "i_t" 0 4 1 (cdfOuterBody k v d v_new h)]
        ++ cdfEpilogue final_state STORE_FINAL_STATE
      = Stmt.forRange "i_t" 0 4 1 (cdfOuterBody k v d v_new h)
        :: cdfEpilogue final_state STORE_FINAL_STATE from rfl,
    stepStmts.cons_some hloop, hepi]
  refine ⟨sE, rfl, ?_, ?_, ?_⟩
  · -- h readback: h reads of sE = sL (epilogue only writes final_state ≠ h) = cdfCarryCell = hValue
    intro j idx hact
    rw [hmemE h _ hFh.symm, hhL j idx hact]
    -- cdfCarryCell j = stateValue j = hValue j
    obtain ⟨e, p, u⟩ := idx
    simp only [hValue, kIndex, hpids0, Nat.zero_mul, Nat.zero_add]
    rw [cdfCarry_active s k v d initial_state USE_INITIAL_STATE hpids0 j.val (by omega)
      e.val p.val e.isLt (by
        obtain ⟨_, hv⟩ := hact; simpa [vIndex] using hv)]
    rfl
  · -- v_new readback
    intro j idx hact
    obtain ⟨hcT, hvV⟩ := hact
    rw [hmemE v_new _ hFv.symm, hvnL j idx (by exact ⟨by simpa using hcT, by simpa [vIndex] using hvV⟩)]
    obtain ⟨c, p, u⟩ := idx
    simp only [vNewSpec, vNewActive, cIndex, vIndex] at *
    rw [cdfBvNew_eq_vNewValue s k v d initial_state USE_INITIAL_STATE hpids0 j.val (by omega)
      c.val p.val (by simpa using hcT) (by simpa using hvV)]
  · -- final_state readback (STORE_FINAL_STATE = true)
    intro hStore idx hact
    rw [hfinE hStore idx hact]
    obtain ⟨e, p, u⟩ := idx
    simp only [finalValue, kIndex, hpids0, Nat.zero_mul, Nat.zero_add]
    rw [cdfCarry_active s k v d initial_state USE_INITIAL_STATE hpids0 4 (by omega)
      e.val p.val e.isLt (by obtain ⟨_, hv⟩ := hact; simpa [vIndex] using hv)]
    rfl

end Correct_without_Rounding

/-! ## ════════ `⊨` IO face for the per-chunk `h` writeback ════════

The launched kernel is a two-level loop (`i_t` over chunks, `i_c` inside), so its
own `⊨` face would need a full nested-`forRange` safety walk. Its per-chunk state
writeback

```python
        p_h = tl.make_block_ptr(base=h + i_bh*s_h_h + i_t*K*V, shape=(K, V),
                                strides=(s_h_t, 1), offsets=(i_k*BK, i_v*BV),
                                block_shape=(BK, BV), order=(1, 0))
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=[0, 1])
```

is a masked `[BK, BV]` memcpy of the carry register at the flat window `hOffset`,
so `chunk_delta_h_state_store_slice` transcribes it with the loop variable `i_t`
as a `Nat` parameter and the carry register materialized into a scratch region
`HPre`. Two documented modeling choices, both already used by the twin surface in
`chunk_gated_attention`: the block pointer's `boundary_check = [0, 1]` is written
as the equivalent lane mask `(offs_k < K) & (offs_v < V)`, and **`HPre` is a
fiction region** — it stands for the register `b_h`, not for a Python tensor, so
this is a statement about the writeback, not about the recurrence (whose math the
port's own closed-form summary carries).

That shape is `Masked3DTileKernelIO₁` — no new library surface. -/

section IOFace

/-- Cell-level frame of a masked scatter (private copy — `bench` files are
standalone). -/
private theorem foldl_writeMem_frame {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → ℝ) (P : α → Prop) [DecidablePred P]
    (R : RegionName) (off : Nat) :
    ∀ l : List α, (R ≠ region ∨ ∀ k ∈ l, P k → offsetFn k ≠ off) →
      ∀ s : BlockState,
        ((l.foldl (fun acc k =>
            if P k then acc.writeMem region (offsetFn k) (valueFn k) else acc)
            s).mem R off) = s.mem R off := by
  intro l
  induction l with
  | nil => intro _ s; rfl
  | cons hd tl ih =>
      intro hc s
      have htl : R ≠ region ∨ ∀ k ∈ tl, P k → offsetFn k ≠ off := by
        rcases hc with h | h
        · exact Or.inl h
        · exact Or.inr fun k hk => h k (List.mem_cons_of_mem hd hk)
      rw [List.foldl_cons, ih htl]
      by_cases hP : P hd
      · rw [if_pos hP, BlockState.writeMem_mem, if_neg ?_]
        rintro ⟨h1, h2⟩
        rcases hc with h | h
        · exact h h1
        · exact h hd List.mem_cons_self hP h2.symm
      · rw [if_neg hP]

/-- The per-chunk `h` writeback of `chunk_delta_rule_fwd_h_surface`, with the
carry register materialized into `HPre` and the loop variable `i_t` a parameter. -/
def chunk_delta_h_state_store_slice
    (HPre h : RegionName) (i_t s_h_h s_h_t K V BK BV : Nat) :
    ComputeKernel := triton {
  i_k = tl.program_id(0)
  i_v = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = i_k * $(BK) + tl.arange(0, $(BK))
  offs_v = i_v * $(BV) + tl.arange(0, $(BV))
  mask = (offs_k[:, None] < $(K)) & (offs_v[None, :] < $(V))
  b_h = tl.load(HPre + i_bh * $(s_h_h) + $(i_t) * $(K) * $(V) +
      offs_k[:, None] * $(s_h_t) + offs_v[None, :],
    mask=mask, other=0.0)
  tl.store(h + i_bh * $(s_h_h) + $(i_t) * $(K) * $(V) +
      offs_k[:, None] * $(s_h_t) + offs_v[None, :],
    (b_h).to(h.dtype.element_ty), mask=mask)
}

/-- The per-lane value the writeback stores: the materialized carry on an active
lane, `0` elsewhere (the `other = 0.0` of the load). -/
noncomputable def hStoreValue (s : BlockState) (HPre : RegionName)
    (i_t s_h_h s_h_t K V BK BV : Nat) (idx : TileIndex [BK, BV]) : ℝ :=
  WithBot.unbotD 0
    (if active s K V BK BV idx then
      some (s.readMem HPre (hOffset s i_t s_h_h s_h_t K V BK BV idx))
    else some (0.0 : ℝ))

theorem chunk_delta_h_state_store_slice_correct
    (HPre h : RegionName) (i_t s_h_h s_h_t K V BK BV : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BK, BV] =>
        hOffset s i_t s_h_h s_h_t K V BK BV idx)) :
    ∀ idx : TileIndex [BK, BV],
      let outAddr := hOffset s i_t s_h_h s_h_t K V BK BV idx
      (exec (chunk_delta_h_state_store_slice HPre h i_t s_h_h s_h_t K V BK BV)
          s).map (·.readMem h outAddr)
        = some (if active s K V BK BV idx then
            hStoreValue s HPre i_t s_h_h s_h_t K V BK BV idx
          else s.readMem h outAddr) := by
  intro idx
  simp [exec, chunk_delta_h_state_store_slice, stepStmts, stepStmt, evalOp,
    evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
    Tile.ptrAdd, NumericDType.add, NumericDType.mul, ComparableDType.lt,
    kIndex, vIndex, active, hOffset, TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BK, BV] → Nat :=
    fun idx => s.pids 2 * s_h_h + i_t * K * V +
      (s.pids 0 * BK + idx.1.val) * s_h_t + (s.pids 1 * BV + idx.2.1.val)
  let valueFn : TileIndex [BK, BV] → ℝ :=
    fun idx => WithBot.unbotD 0
      (if s.pids 0 * BK + idx.1.val < K ∧ s.pids 1 * BV + idx.2.1.val < V then
        some (s.readMem HPre (offsetFn idx))
      else some (0.0 : ℝ))
  let P : TileIndex [BK, BV] → Prop :=
    fun idx => s.pids 0 * BK + idx.1.val < K ∧ s.pids 1 * BV + idx.2.1.val < V
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, hOffset, kIndex, vIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i =>
        if P i then acc.writeMem h (offsetFn i) (valueFn i) else acc)
      _ (TileShape.allIndices [BK, BV])).readMem h (offsetFn idx) =
    if P idx then hStoreValue s HPre i_t s_h_h s_h_t K V BK BV idx
    else s.readMem h (offsetFn idx)
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive : s.pids 0 * BK + idx.1.val < K ∧
      s.pids 1 * BV + idx.2.1.val < V
  · rfl
  · rfl

theorem h_state_flattenOk (HPre h : RegionName)
    (i_t s_h_h s_h_t K V BK BV : Nat) :
    ((chunk_delta_h_state_store_slice HPre h i_t s_h_h s_h_t K V
      BK BV).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [chunk_delta_h_state_store_slice, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, StmtList.FlattenOk,
    Stmt.FlattenOk, Op.FlattenOk]
  and_intros <;> simp [Op.FlattenOk.eq_def]

theorem h_state_terminates (HPre h : RegionName)
    (i_t s_h_h s_h_t K V BK BV : Nat) (s : BlockState) :
    ∃ s1, exec (chunk_delta_h_state_store_slice HPre h i_t s_h_h s_h_t K V BK BV)
      s = some s1 := by
  simp [exec, chunk_delta_h_state_store_slice, stepStmts, stepStmt,
    evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
    Tile.ptrAdd, NumericDType.add, NumericDType.mul, ComparableDType.lt,
    TileShape.dropInsertedIndex]

theorem h_state_frame (HPre h : RegionName) (i_t s_h_h s_h_t K V BK BV : Nat)
    (s s' : BlockState)
    (hExec : exec (chunk_delta_h_state_store_slice HPre h i_t s_h_h s_h_t K V BK
      BV) s = some s') :
    ∀ (r : RegionName) (o : Nat),
      (r ≠ h ∨ ∀ idx : TileIndex [BK, BV], active s K V BK BV idx →
        o ≠ hOffset s i_t s_h_h s_h_t K V BK BV idx) →
      s'.mem r o = s.mem r o := by
  intro r o hcond
  simp only [active, kIndex, vIndex, hOffset] at hcond
  simp [exec, chunk_delta_h_state_store_slice, stepStmts, stepStmt,
    evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
    Tile.ptrAdd, NumericDType.add, NumericDType.mul, ComparableDType.lt,
    TileShape.dropInsertedIndex] at hExec
  subst hExec
  rw [foldl_writeMem_frame (region := h)
    (fun idx : TileIndex [BK, BV] => s.pids 2 * s_h_h + i_t * K * V
      + (s.pids 0 * BK + idx.1.val) * s_h_t + (s.pids 1 * BV + idx.2.1.val))
    _ (fun idx : TileIndex [BK, BV] =>
      s.pids 0 * BK + idx.1.val < K ∧ s.pids 1 * BV + idx.2.1.val < V)
    r o (TileShape.allIndices [BK, BV]) ?_]
  · simp
  · rcases hcond with hc | hc
    · exact Or.inl hc
    · exact Or.inr fun idx _ hidx => Ne.symm (hc idx hidx)

theorem h_state_traceSafe (HPre h : RegionName)
    (i_t s_h_h s_h_t K V BK BV : Nat) (bounds : RegionBounds) (s : BlockState)
    (hin : ∀ idx : TileIndex [BK, BV], active s K V BK BV idx →
      hOffset s i_t s_h_h s_h_t K V BK BV idx < bounds HPre)
    (hout : ∀ idx : TileIndex [BK, BV], active s K V BK BV idx →
      hOffset s i_t s_h_h s_h_t K V BK BV idx < bounds h) :
    ((chunk_delta_h_state_store_slice HPre h i_t s_h_h s_h_t K V
      BK BV).toAlgKernel).TraceSafe bounds s := by
  simp only [active, kIndex, vIndex, hOffset] at hin hout
  simp [Kernel.TraceSafe, chunk_delta_h_state_store_slice, Stmt.TraceSafeList,
    Stmt.TraceSafe, Op.SafeAt, MaskOpt.SafeAt, MaskOpt.Active,
    MaskOpt.MemorySafe, MemAccess.SafeAt, MemAccess.MemorySafe,
    memAccessMemorySafe, MemAccess.ActiveAddressSafe,
    memAccessActiveAddressSafe, Op.PointerAddressesSafeOn, Op.MemorySafe,
    stepStmts, stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map,
    Tile.bop, Tile.cop, Tile.expandDim, Tile.ptrAdd, NumericDType.add,
    NumericDType.mul, ComparableDType.lt, TileShape.dropInsertedIndex]
  and_intros
  all_goals try exact fun a b ha hb => hin (a, b, PUnit.unit) ⟨ha, hb⟩
  all_goals try exact fun a b ha hb => hout (a, b, PUnit.unit) ⟨ha, hb⟩
  all_goals try simp [Op.SafeAt.eq_def]

/-- Region-model run of the per-chunk `h` writeback. -/
theorem h_state_region_run (HPre h : RegionName)
    (i_t s_h_h s_h_t K V BK BV : Nat) (s₀ : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BK, BV] =>
        hOffset s₀ i_t s_h_h s_h_t K V BK BV idx))
    (xs : TileIndex [BK, BV] → ℝ)
    (hx : ∀ idx : TileIndex [BK, BV], active s₀ K V BK BV idx →
      s₀.readMem HPre (hOffset s₀ i_t s_h_h s_h_t K V BK BV idx) = xs idx) :
    ∃ s1, exec (chunk_delta_h_state_store_slice HPre h i_t s_h_h s_h_t K V BK BV)
        s₀ = some s1
      ∧ (∀ idx : TileIndex [BK, BV], active s₀ K V BK BV idx →
          s1.readMem h (hOffset s₀ i_t s_h_h s_h_t K V BK BV idx) = xs idx)
      ∧ (∀ (r : RegionName) (o : Nat),
          (r ≠ h ∨ ∀ idx : TileIndex [BK, BV], active s₀ K V BK BV idx →
            o ≠ hOffset s₀ i_t s_h_h s_h_t K V BK BV idx) →
          s1.mem r o = s₀.mem r o) := by
  obtain ⟨s1, hexec⟩ := h_state_terminates HPre h i_t s_h_h s_h_t K V BK BV s₀
  refine ⟨s1, hexec, ?_,
    h_state_frame HPre h i_t s_h_h s_h_t K V BK BV s₀ s1 hexec⟩
  intro idx hact
  have hc := chunk_delta_h_state_store_slice_correct HPre h i_t s_h_h s_h_t K V
    BK BV s₀ hOutInj idx
  have hval : s1.readMem h (hOffset s₀ i_t s_h_h s_h_t K V BK BV idx)
      = if active s₀ K V BK BV idx then
          hStoreValue s₀ HPre i_t s_h_h s_h_t K V BK BV idx
        else s₀.readMem h (hOffset s₀ i_t s_h_h s_h_t K V BK BV idx) := by
    simpa [hexec] using hc
  rw [hval, if_pos hact, hStoreValue, if_pos hact]
  simpa using hx idx hact

/-- IO signature of the per-chunk `h` writeback on the three-axis tile surface. -/
def h_stateIO (HPre h : RegionName) (i_t s_h_h s_h_t K V BK BV : Nat) :
    Masked3DTileKernelIO₁ where
  kernel := chunk_delta_h_state_store_slice HPre h i_t s_h_h s_h_t K V BK BV
  inp := HPre
  out := h
  shape := [BK, BV]
  read := fun p₀ p₁ p₂ idx =>
    p₂ * s_h_h + i_t * K * V + (p₀ * BK + idx.1.val) * s_h_t
      + (p₁ * BV + idx.2.1.val)
  write := fun p₀ p₁ p₂ idx =>
    p₂ * s_h_h + i_t * K * V + (p₀ * BK + idx.1.val) * s_h_t
      + (p₁ * BV + idx.2.1.val)
  mask := fun p₀ p₁ _p₂ idx =>
    p₀ * BK + idx.1.val < K ∧ p₁ * BV + idx.2.1.val < V

/-- **Where the writeback face meets the recurrence.** Instantiating the `⊨`
face's channel at the port's own chunk-start state: if the materialized carry
holds `hValue` — the genuine `H_{i_t}` of the delta-rule closed form — on the
kept lanes, then the writeback lands exactly `hValue` in `h` and touches nothing
else. So "`HPre` is a fiction region standing for `b_h`" is a theorem, not a
comment. -/
theorem h_state_io_at_hValue
    (HPre h k v d initial_state : RegionName)
    (i_t s_h_h s_h_t s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d K V BT BV BK : Nat)
    (USE_INITIAL_STATE : Bool) (s₀ : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BK, BV] =>
        hOffset s₀ i_t s_h_h s_h_t K V BK BV idx))
    (hcarry : ∀ idx : TileIndex [BK, BV], active s₀ K V BK BV idx →
      s₀.readMem HPre (hOffset s₀ i_t s_h_h s_h_t K V BK BV idx)
        = hValue s₀ k v d initial_state s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t
            s_vo_d K V BT BV BK USE_INITIAL_STATE i_t idx) :
    ∃ s1, exec (chunk_delta_h_state_store_slice HPre h i_t s_h_h s_h_t K V BK BV)
        s₀ = some s1
      ∧ (∀ idx : TileIndex [BK, BV], active s₀ K V BK BV idx →
          s1.readMem h (hOffset s₀ i_t s_h_h s_h_t K V BK BV idx)
            = hValue s₀ k v d initial_state s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t
                s_vo_d K V BT BV BK USE_INITIAL_STATE i_t idx)
      ∧ (∀ (r : RegionName) (o : Nat),
          (r ≠ h ∨ ∀ idx : TileIndex [BK, BV], active s₀ K V BK BV idx →
            o ≠ hOffset s₀ i_t s_h_h s_h_t K V BK BV idx) →
          s1.mem r o = s₀.mem r o) :=
  h_state_region_run HPre h i_t s_h_h s_h_t K V BK BV s₀ hOutInj _ hcarry

/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/

/-- **The headline on the IO surface** for `chunk_delta_rule_fwd_h_surface`'s
per-chunk `h` writeback: for every disjoint flat placement of `HPre` / `h`, every
program coordinate whose active lanes are in bounds, and every launch state whose
`HPre` block holds `xs` at the active lanes, the writeback terminates, every
active lane of `h` holds `xs idx`, and every other memory cell is unchanged.

**Scope.** `HPre` is a **fiction region**: it stands for the kernel's carry
register `b_h`, not for a Python tensor — the same idiom as
`fused_recurrent_retention`'s `HSeed`. The delta-rule recurrence that produces the
carry keeps its own closed-form summary above; this face is the writeback's
contract, and the two meet at the carry. The block pointer's
`boundary_check = [0, 1]` is transcribed as the equivalent lane mask, exactly as
in the twin `chunk_gated_attention` surface.

Dimension-general in `i_t`, `s_h_h`, `s_h_t`, `K`, `V`, `BK` and `BV`. Honest
side-condition: address injectivity at every program coordinate, the same
hypothesis the per-write-map summaries take. -/
specification chunk_delta_h_state_store_io_correctness
    (HPre h : RegionName) (i_t s_h_h s_h_t K V BK BV : Nat)
    (hInj : ∀ p₀ p₁ p₂ : Nat, Function.Injective
      (fun idx : TileIndex [BK, BV] =>
        p₂ * s_h_h + i_t * K * V + (p₀ * BK + idx.1.val) * s_h_t
          + (p₁ * BV + idx.2.1.val))) :
    h_stateIO HPre h i_t s_h_h s_h_t K V BK BV
      ⊨ fun _p₀ _p₁ xs idx => xs idx := by
  refine Masked3DTileKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact h_state_flattenOk HPre h i_t s_h_h s_h_t K V BK BV
  · intro bounds s h1 h2
    exact h_state_traceSafe HPre h i_t s_h_h s_h_t K V BK BV bounds s
      (fun idx hact => h1 idx hact) (fun idx hact => h2 idx hact)
  · intro s₀ xs hin
    exact h_state_region_run HPre h i_t s_h_h s_h_t K V BK BV s₀
      (hInj (s₀.pids 0) (s₀.pids 1) (s₀.pids 2)) xs
      (fun idx hact => hin idx hact)

/-! ### The rounding face

The writeback is a pure copy: no arithmetic, and the store's `.to(...)` erases to
`.real`. So the slice is **cast-free** — every statement steps identically under
`stepStmtsR R` and `stepStmts` — and the exact run transports to `execR R` for
every rounding model. -/

/-- The slice is cast-free: `execR R` is the exact stepper. -/
private theorem h_state_castFree (R : RoundingModel) (HPre h : RegionName)
    (i_t s_h_h s_h_t K V BK BV : Nat) (s : BlockState) :
    execR R ((chunk_delta_h_state_store_slice HPre h i_t s_h_h s_h_t K V BK
        BV).toAlgKernel) s
      = exec ((chunk_delta_h_state_store_slice HPre h i_t s_h_h s_h_t K V BK
        BV).toAlgKernel) s := by
  simp [execR, exec, chunk_delta_h_state_store_slice, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, stepStmtsR, stepStmts,
    stepStmtR, stepStmt, evalOpR, evalOpR.eq_def, evalOp, evalOp.eq_def,
    BlockState.writeMemTypedR, BlockState.writeMemAsR, Option.bind, Option.map,
    Tile.bop, Tile.cop, Tile.expandDim, Tile.ptrAdd, Tile.uop, NumericDType.add,
    NumericDType.mul, ComparableDType.lt, FloatDType.cast,
    TileShape.dropInsertedIndex]

/-- Per-execution safety walk **under the rounding model**. -/
theorem h_state_traceSafeR (R : RoundingModel) (HPre h : RegionName)
    (i_t s_h_h s_h_t K V BK BV : Nat) (bounds : RegionBounds) (s : BlockState)
    (hin : ∀ idx : TileIndex [BK, BV], active s K V BK BV idx →
      hOffset s i_t s_h_h s_h_t K V BK BV idx < bounds HPre)
    (hout : ∀ idx : TileIndex [BK, BV], active s K V BK BV idx →
      hOffset s i_t s_h_h s_h_t K V BK BV idx < bounds h) :
    Kernel.TraceSafeR R bounds
      ((chunk_delta_h_state_store_slice HPre h i_t s_h_h s_h_t K V BK
        BV).toAlgKernel) s := by
  simp only [active, kIndex, vIndex, hOffset] at hin hout
  unfold Kernel.TraceSafeR
  simp [chunk_delta_h_state_store_slice, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Stmt.TraceSafeListR,
    Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.SafeAtR, MaskOpt.ActiveR,
    MemAccess.SafeAtR, MemAccess.ActiveAddressSafeR,
    memAccessActiveAddressSafeR, stepStmtR, evalOpR, evalOpR.eq_def,
    Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim, Tile.ptrAdd,
    Tile.uop, NumericDType.add, NumericDType.mul, ComparableDType.lt,
    FloatDType.cast, TileShape.dropInsertedIndex]
  and_intros
  all_goals try exact fun a b ha hb => hin (a, b, PUnit.unit) ⟨ha, hb⟩
  all_goals try exact fun a b ha hb => hout (a, b, PUnit.unit) ⟨ha, hb⟩

/-! ### ════════ ★ MAIN THEOREM (rounding face) ★ ════════ -/

/-- **The `⊨[R]` headline** for the per-chunk `h` writeback: for **every**
rounding model `R`, the same masked Hoare triple as
`chunk_delta_h_state_store_io_correctness`, run under `execR R` and read back as
`.real`-typed cells holding `R.round .real (xs idx)`.

The writeback is a pure copy — no arithmetic, and the store's `.to(...)` erases to
`.real` — so the slice is cast-free and the exact run transports verbatim. The
content of the rounding face is exactly that: *the writeback introduces no
rounding event of its own*, at any `R`. Same scope caveat as the exact face:
`HPre` is a fiction region standing for the carry register `b_h`. -/
specification chunk_delta_h_state_store_io_correctnessR
    (R : RoundingModel) (HPre h : RegionName) (i_t s_h_h s_h_t K V BK BV : Nat)
    (hInj : ∀ p₀ p₁ p₂ : Nat, Function.Injective
      (fun idx : TileIndex [BK, BV] =>
        p₂ * s_h_h + i_t * K * V + (p₀ * BK + idx.1.val) * s_h_t
          + (p₁ * BV + idx.2.1.val))) :
    h_stateIO HPre h i_t s_h_h s_h_t K V BK BV
      ⊨[R, FloatDType.real] fun _p₀ _p₁ xs idx => xs idx := by
  refine Masked3DTileKernelIO₁.ImplementsR.intro _ ?_ ?_ ?_
  · exact h_state_flattenOk HPre h i_t s_h_h s_h_t K V BK BV
  · intro bounds s h1 h2
    exact h_state_traceSafeR R HPre h i_t s_h_h s_h_t K V BK BV bounds s
      (fun idx hact => h1 idx hact) (fun idx hact => h2 idx hact)
  · intro s₀ xs hx
    obtain ⟨s1, hexec, hval, hframe⟩ :=
      h_state_region_run HPre h i_t s_h_h s_h_t K V BK BV s₀
        (hInj (s₀.pids 0) (s₀.pids 1) (s₀.pids 2)) xs
        (fun idx hact => hx idx hact)
    have hval' : ∀ idx : TileIndex [BK, BV], active s₀ K V BK BV idx →
        s1.readMem h (s₀.pids 2 * s_h_h + i_t * K * V
            + (s₀.pids 0 * BK + idx.1.val) * s_h_t
            + (s₀.pids 1 * BV + idx.2.1.val)) = xs idx := by
      simpa [hOffset, kIndex, vIndex] using hval
    refine ⟨s1, ?_, ?_, hframe⟩
    · simp only [h_stateIO]
      rw [h_state_castFree R HPre h i_t s_h_h s_h_t K V BK BV s₀]
      exact hexec
    · intro idx hidx
      simp only [h_stateIO]
      rw [BlockState.readMemAs_real, hval' idx hidx]
      simp

end IOFace

end VeriTile.Bench.TritonBenchG.ChunkDeltaFwd

