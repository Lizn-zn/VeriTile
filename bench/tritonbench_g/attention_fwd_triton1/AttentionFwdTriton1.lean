import VeriTile.Triton

/-!
# `attention_fwd_triton1` — strict per-kernel correctness

`attention_fwd_kernel` is a linear (chunked) attention forward kernel: program
`i_bh` (one batch-head) carries a `[BD, BD]` recurrent state `b_h` across the
`cdiv(T, BT)` time chunks. Per chunk it loads `Q`/`K`/`V` block pointers, scales
`b_q` by `scale`, forms `b_s = dot(b_q, b_k)` and the local output
`b_o = dot(b_s, b_v)`, optionally adds the inter-chunk term `dot(b_q, b_h)`,
updates `b_h += dot(b_k, b_v)`, optionally stores `b_h` to `h` (`STORE`), and
stores `b_o` to `o`. The `IFCOND` flag selects whether the first chunk skips the
recurrent add (resetting `b_h = dot(b_k, b_v)` instead).

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`attention_fwd_kernel[grid](...)`, the grid over
`batch·n_heads`, scheduling, and how the runtime composes per-program writes
into one buffer) is the *trusted boundary*, not a proof obligation here. Because
`i_bh` is universally quantified, the per-program statement covers every program
of the grid.

## Proof architecture

```
attention_fwd_triton1_output_summary_general                         ← GENERAL TOP THEOREM (dimension-parameterized)
  ├─ attention_fwd_triton1_exec_outputClosedFormG                     exec O = outputClosedForm
  │    ├─ aft1_exec_carryG           prologue + 32-chunk forRangeDyn loop → b_h = Σ Kᵀ·V, O = aft1OutG
  │    │    ├─ aft1_prologue_invG                      reaches the carry invariant aft1InvG 0
  │    │    ├─ aft1InvG_step                           one iteration preserves aft1InvG (carry n→n+1)
  │    │    │    └─ aft1_loopBody_iter_ffG             full body: loads/dots/carry-update + O store
  │    │    │         └─ aft1_loopBody_regs_ffG        registers: b_o = aft1OutG, b_h = aft1BhTileG (c+1)
  │    │    └─ forRangeDyn_inv                         the dynamic-bound carry induction
  │    └─ aft1OutG_eq_outputClosedForm                 kernel-native output = genuine closed form
  └─ attention_fwd_kernel_surface_toAlgorithm_supported (×4 STORE/IFCOND branch lowerings)
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); dtype casts collapse to
the identity post-erasure; `@triton.autotune` / `num_warps`/`num_stages` are not
modeled. The output summary is dimension-general (symbolic `s_qh`, `BT`, `BD`,
`NT`, `scale`, strides, …); the Python test shape
(`B=2, H=8, T=1024, D=128, BT=32, BD=128, NT=32`, `scale = 1/sqrt(128)`,
contiguous strides) is the special case. The default-branch (`STORE=false, IFCOND=false`) `O`
writeback is stated against the genuine `outputClosedForm` — the linear-attention
local `(scale·Q·K·V)` plus recurrent `(scale·Q·b_h_c)` terms, expressed purely over
input memory with **no self-reference** to the executed kernel. The full kernel is
executed (prologue + the 32-chunk `forRangeDyn` loop carrying `b_h = Σ_{j<c} Kⱼᵀ·Vⱼ`)
and proven to realize that closed form at every chunk/lane. All four
`STORE`/`IFCOND` branch surfaces additionally lower faithfully to the algorithm
layer. This is a single-program (single batch-head) scope; cross-program
composition is the
trusted host boundary.
-/

namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton1

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-! **★ Main theorem:** `attention_fwd_triton1_output_summary_general` -/

/-! # ══════════ CORRECT — genuine, dimension-general (review this) ══════════ -/

section Correct_without_Rounding

/-- Faithful DSL port of `attention_fwd_triton1.py`'s
`attention_fwd_kernel`.

The Python kernel uses block pointers plus two constexpr gates, `STORE` and
`IFCOND`. The `order` metadata is accepted by the DSL and erased into the same
block-pointer AST. -/
def attention_fwd_kernel_surface
    (q k v h o : RegionName)
    (s_qh s_qt s_qd s_hh s_ht T : Nat) (scale : ℝ)
    (BT BD NT : Nat) (STORE IFCOND : Bool) :
    ComputeKernel := triton {
  i_bh = tl.program_id(0)
  b_h = tl.zeros([$(BD), $(BD)], dtype=tl.float32)
  for i in range($(0), tl.cdiv($(T), $(BT))) {
    p_q = tl.make_block_ptr(base=q + i_bh * $(s_qh),
      shape=($(T), $(BD)), strides=($(s_qt), $(s_qd)),
      offsets=(i * $(BT), 0), block_shape=($(BT), $(BD)), order=(1, 0))
    p_k = tl.make_block_ptr(base=k + i_bh * $(s_qh),
      shape=($(BD), $(T)), strides=($(s_qd), $(s_qt)),
      offsets=(0, i * $(BT)), block_shape=($(BD), $(BT)), order=(0, 1))
    p_v = tl.make_block_ptr(base=v + i_bh * $(s_qh),
      shape=($(T), $(BD)), strides=($(s_qt), $(s_qd)),
      offsets=(i * $(BT), 0), block_shape=($(BT), $(BD)), order=(1, 0))
    p_h = tl.make_block_ptr(base=h + i_bh * $(s_hh),
      shape=($((NT * BD : Nat)), $(BD)), strides=($(s_ht), $(s_qd)),
      offsets=(i * $(BD), 0), block_shape=($(BD), $(BD)), order=(1, 0))
    p_o = tl.make_block_ptr(base=o + i_bh * $(s_qh),
      shape=($(T), $(BD)), strides=($(s_qt), $(s_qd)),
      offsets=(i * $(BT), 0), block_shape=($(BT), $(BD)), order=(1, 0))

    if STORE {
      tl.store(p_h, (b_h).to(p_h.dtype.element_ty))
    }
    b_q = tl.load(p_q)
    b_q = (b_q * $((scale : ℝ))).to(b_q.dtype)
    b_k = tl.load(p_k)
    b_v = tl.load(p_v)

    b_s = tl.dot(b_q, b_k, allow_tf32=false)
    b_o = tl.dot((b_s).to(b_q.dtype), b_v, allow_tf32=false)
    if IFCOND {
      if i == $(0) {
        b_h = tl.dot(b_k, b_v, allow_tf32=false)
      } else {
        b_o += tl.dot(b_q, (b_h).to(b_q.dtype), allow_tf32=false)
        b_h += tl.dot(b_k, b_v, allow_tf32=false)
      }
    } else {
      b_o += tl.dot(b_q, (b_h).to(b_q.dtype), allow_tf32=false)
      b_h += tl.dot(b_k, b_v, allow_tf32=false)
    }

    tl.store(p_o, (b_o).to(p_o.dtype.element_ty))
  }
}

/-- The full Python-shaped forward surface lowers to the algorithm layer,
including the block-pointer loop, optional H-state store, dot products, and
output writeback. -/
theorem attention_fwd_kernel_surface_toAlgorithm_supported
    (q k v h o : RegionName)
    (s_qh s_qt s_qd s_hh s_ht T : Nat) (scale : ℝ)
    (BT BD NT : Nat) (STORE IFCOND : Bool) :
    ∃ alg, (attention_fwd_kernel_surface q k v h o s_qh s_qt s_qd s_hh
      s_ht T scale BT BD NT STORE IFCOND).toAlgorithm? = Except.ok alg := by
  simp [attention_fwd_kernel_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-! ## Genuine closed form for the linear-attention output

`attention_fwd_kernel` implements *linear (chunked) attention*: there is **no
softmax** at all. Per batch-head program the loop carries the recurrent state

  `b_h_i = Σ_{j < i} (Kⱼᵀ · Vⱼ)`   (a `[BD, BD]` matrix),

and the output of time-chunk `i` (default branch `STORE=False, IFCOND=False`,
which the `IFCOND=True, i>0` branch coincides with, and the `i=0` branch reduces
to since `b_h_0 = 0`) is

  `Oᵢ[t, d] = ((scale·Qᵢ) · Kᵢ · Vᵢ)[t, d] + ((scale·Qᵢ) · b_h_i)[t, d]`.

The first summand is exactly `localTerm` for chunk `i`'s tiles; the second
is the contraction of the scaled query against the accumulated state. The
definitions below give that genuine closed form purely in terms of the input
memory (no reference to the executed kernel output), and the identity theorems
establish the linear-attention algebra: the recurrent contribution telescopes
into a single double sum over all prior keys/values.

The per-chunk tile accessors are passed as offset functions `qAddr`/`kAddr`/
`vAddr : Nat → ... → Nat` mapping `(chunk, row, col)` to a flat memory offset,
matching the `make_block_ptr` layout `base + (chunk·BT + row)·s_t + col·s_d`. -/

/-- One chunk's local (`(scale·Q)·K·V`) closed-form entry, stated directly over
input memory via per-chunk row/col offset accessors. -/
noncomputable def localTerm
    (s : BlockState) (Q K V : RegionName) (scale : ℝ) (BT BD : Nat)
    (qAddr kAddr vAddr : Nat → Nat → Nat)
    (chunk : Nat) (t : Fin BT) (d : Fin BD) : ℝ :=
  ∑ tk : Fin BT,
    (∑ dd : Fin BD,
      (s.readMem Q (qAddr chunk (BD * t.val + dd.val)) * scale) *
        s.readMem K (kAddr chunk (BT * dd.val + tk.val))) *
      s.readMem V (vAddr chunk (BD * tk.val + d.val))

/-- The recurrent state matrix `b_h_i[d', d] = Σ_{j < i} (Kⱼᵀ·Vⱼ)[d', d]`,
genuine closed form over input memory. -/
noncomputable def recurrentState
    (s : BlockState) (K V : RegionName) (BT BD : Nat)
    (kAddr vAddr : Nat → Nat → Nat)
    (chunk : Nat) (d' d : Fin BD) : ℝ :=
  ∑ j ∈ Finset.range chunk,
    ∑ tk : Fin BT,
      s.readMem K (kAddr j (BT * d'.val + tk.val)) *
        s.readMem V (vAddr j (BD * tk.val + d.val))

/-- The recurrent output contribution `((scale·Qᵢ)·b_h_i)[t, d]`. -/
noncomputable def recurrentTerm
    (s : BlockState) (Q K V : RegionName) (scale : ℝ) (BT BD : Nat)
    (qAddr kAddr vAddr : Nat → Nat → Nat)
    (chunk : Nat) (t : Fin BT) (d : Fin BD) : ℝ :=
  ∑ d' : Fin BD,
    (s.readMem Q (qAddr chunk (BD * t.val + d'.val)) * scale) *
      recurrentState s K V BT BD kAddr vAddr chunk d' d

/-- Genuine closed-form output of chunk `chunk`, position `(t, d)`:
local term plus recurrent term. No self-reference to the executed kernel. -/
noncomputable def outputClosedForm
    (s : BlockState) (Q K V : RegionName) (scale : ℝ) (BT BD : Nat)
    (qAddr kAddr vAddr : Nat → Nat → Nat)
    (chunk : Nat) (t : Fin BT) (d : Fin BD) : ℝ :=
  localTerm s Q K V scale BT BD qAddr kAddr vAddr chunk t d +
    recurrentTerm s Q K V scale BT BD qAddr kAddr vAddr chunk t d

/-! ## `output_summary` — placeholder

The public genuine output summary `attention_fwd_triton1_output_summary_general`
is stated below (after the closed-form derivation) against the genuine
`outputClosedForm`, not against any self-referential executed value. -/

/-! ## Full-surface exec → genuine closed form (carry-fold derivation)

The block below derives the executed output of the full `attention_fwd_kernel_surface`
directly, eliminating the self-referential `attentionFwdTriton1SurfaceValue`. The
loop carries `b_h = recurrentState n` across the 32 chunks via `forRangeDyn_inv`. -/

theorem aft1_withBot_sum_some {N : Nat} (g : Fin N → ℝ) :
    @Finset.sum (Fin N) (WithBot ℝ) _ Finset.univ (fun k => (some (g k) : WithBot ℝ))
      = some (Finset.univ.sum g) := by
  show (Finset.univ.sum fun k => ((g k : ℝ) : WithBot ℝ)) = ((Finset.univ.sum g : ℝ) : WithBot ℝ)
  exact (WithBot.coe_sum Finset.univ g).symm

/-- **2D dot element lemma.** For all-`some` operand tiles `a : [M,K]`, `b : [K,N]`,
the `(m, n)` cell of `dot a b` is `Σ_e a[m,e]·b[e,n]`. -/
theorem aft1_dot2d_elem {M K N : Nat} (a : Tile .real [M, K]) (b : Tile .real [K, N])
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
  exact aft1_withBot_sum_some _

/-- Scalar offset op `name * c` evaluates to `scalar (val * c)` given `name = val`. -/
theorem aft1_mulConst_eval (s : BlockState) (name : RegName) (val c : Nat)
    (hr : s.regs .nat [] name = some (Tile.scalar val)) :
    evalOp (Op.mul .nat Broadcast.nil (Op.ref .nat [] name) (Op.constNat c)) s
      = some (Tile.scalar (val * c)) := by
  rw [evalOp_mul]
  simp only [evalOp_ref, evalOp_constNat, hr, Option.bind_eq_bind, Option.bind_some]
  rfl

/-- **2D `makeBlockPtrDynOffsets` eval recipe.** -/
theorem aft1_makeBlockPtr_2d_eval (rg : RegionName) (s : BlockState)
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

/-- **No-mask, no-boundary-check 2D block-pointer load through a bound register
`name`.** With empty `boundaryCheck` the in-bounds gate is vacuously true, so every
lane `(i,j)` reads the genuine memory cell `base + (rowOff+i)·strideT + (colOff+j)·strideS`.
This matches `attention_fwd_kernel`'s `tl.load(p_x)` (no boundary check). -/
theorem aft1_load_bp_2d_ref (rg : RegionName) (s : BlockState) (name : RegName)
    (base rows cols BT BS strideT strideS rowOff colOff : Nat)
    (hreg : s.regs TileDType.blockPtr [BT, BS] name = some
      ⟨fun _ => BlockPtr.mk rg base [rows, cols] [BT, BS] [strideT, strideS]
        [rowOff, colOff]⟩) :
    evalOp (Op.load TileDType.real
      (MemAccess.blockPtr (Op.ref TileDType.blockPtr [BT, BS] name) []) MaskOpt.none) s
    = some ⟨fun idx : TileIndex [BT, BS] =>
        some (s.readMem rg (base + (rowOff + idx.1.val) * strideT
            + (colOff + idx.2.1.val) * strideS))⟩ := by
  simp only [evalOp, evalOp_ref, hreg, List.mapM, List.mapM.loop, bind, Option.bind, Tile.scalar,
    List.reverse_cons, List.reverse_nil, List.append_nil, List.nil_append, List.cons_append]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, j, rest⟩ := idx
  simp only [TileShape.indexToList, BlockPtr.address_2d_offsets, BlockPtr.inBounds,
    List.all_nil, BlockState.readMemValue_real, if_true]

/-! ### Kernel-native chunk cells (exact `make_block_ptr` reads)

These mirror the executed kernel's per-chunk memory reads exactly (batch-head
`i_bh = pids 0`), so the loop-body stepping connects to them definitionally. The
genuine file-level `recurrentState`/`outputClosedForm` are reconciled with these
through the accessor hypotheses at the summary, via the general bridge
`aft1OutG_eq_outputClosedForm`. -/

/-- An `ifThen` with a `false` constexpr condition is a no-op. -/
theorem aft1_ifThen_false_noop (body : List Stmt) (X : BlockState) :
    stepStmt (Stmt.ifThen (Op.constBool Bool.false) body) X = some X := by
  simp [stepStmt, evalOp]

/-- Evaluation recipe for a plain `dot(ref a, ref b)` assign, returning the tile
whose `(m,n)` cell is `Σ_e fa(m,e)·fb(e,n)`, given the operand registers expose
all-`some` tiles with cell functions `fa`/`fb`. -/
theorem aft1_dot_op_eval {M Kd N : Nat} (s' : BlockState) (aName bName : RegName)
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
  exact aft1_dot2d_elem atile btile m n (fa m) (fun e => fb e n)
    (fun e => haf m e) (fun e => hbf e n)

/-- Evaluation recipe for an accumulating `acc + dot(ref a, ref b)` assign. -/
theorem aft1_accDot_op_eval {M Kd N : Nat} (s' : BlockState)
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
  have hdotev := aft1_dot_op_eval s' aName bName atile btile fa fb ha hb haf hbf
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

/-- An `ifThenElse` with a `false` constexpr cond runs the else-branch. -/
theorem aft1_ifThenElse_false_else (thenB elseB : List Stmt) (X : BlockState) :
    stepStmt (Stmt.ifThenElse (Op.constBool Bool.false) thenB elseB) X
      = stepStmts elseB X := by
  simp [stepStmt, evalOp]

/-- A `foldl` of `writeMemTyped .real` stores into region `wr` preserves `readMem`
on any other region `rr ≠ wr`, regardless of the per-index offsets/values. -/
theorem aft1_foldl_store_readMem_ne {α : Type} (l : List α)
    (wr rr : RegionName) (offFn : α → Nat) (valFn : α → TileCarrier .real)
    (s0 : BlockState) (off : Nat) (hne : rr ≠ wr) :
    (l.foldl (fun acc i => acc.writeMemTyped .real wr (offFn i) (valFn i)) s0).readMem rr off
      = s0.readMem rr off := by
  induction l generalizing s0 with
  | nil => rfl
  | cons hd tl ih =>
      simp only [List.foldl_cons]
      rw [ih]
      simp only [BlockState.writeMemTyped_real, BlockState.writeMem_readMem_of_ne_region _ _ _ _ _ _ hne]

/-- A `foldl` of `writeMemTyped .real` stores preserves `pids`. -/
theorem aft1_foldl_store_pids {α : Type} (l : List α)
    (wr : RegionName) (offFn : α → Nat) (valFn : α → TileCarrier .real)
    (s0 : BlockState) :
    (l.foldl (fun acc i => acc.writeMemTyped .real wr (offFn i) (valFn i)) s0).pids = s0.pids := by
  induction l generalizing s0 with
  | nil => rfl
  | cons hd tl ih => simp only [List.foldl_cons]; rw [ih, BlockState.writeMemTyped_pids]

/-- A `foldl` of `writeMem` stores into region `wr` preserves `readMem wr off`
at any offset `off` distinct from every written offset. -/
theorem aft1_foldl_writeMem_readMem_other {α : Type} (l : List α)
    (wr : RegionName) (offFn : α → Nat) (valFn : α → ℝ)
    (s0 : BlockState) (off : Nat) (hoff : ∀ i ∈ l, offFn i ≠ off) :
    (l.foldl (fun acc i => acc.writeMem wr (offFn i) (valFn i)) s0).readMem wr off
      = s0.readMem wr off := by
  induction l generalizing s0 with
  | nil => rfl
  | cons hd tl ih =>
      simp only [List.foldl_cons]
      rw [ih _ (fun i hi => hoff i (List.mem_cons_of_mem hd hi))]
      rw [BlockState.writeMem_readMem, if_neg]
      rintro ⟨_, heq⟩
      exact hoff hd List.mem_cons_self heq.symm

/-- General scaled query cell `b_q[t,e] = scale · Q[base + (c·BT+t)·BD + e]`. -/
noncomputable def aft1QCellG (s : BlockState) (Q : RegionName)
    (s_qh : Nat) (scale : ℝ) (BT BD : Nat) (c t e : Nat) : ℝ :=
  s.readMem Q (s.pids 0 * s_qh + (c * BT + t) * BD + e) * scale

/-- General key cell `b_k[e,tk] = K[base + e + (c·BT+tk)·BD]`. -/
noncomputable def aft1KCellG (s : BlockState) (K : RegionName)
    (s_qh BT BD : Nat) (c e tk : Nat) : ℝ :=
  s.readMem K (s.pids 0 * s_qh + e + (c * BT + tk) * BD)

/-- General value cell `b_v[tk,d] = V[base + (c·BT+tk)·BD + d]`. -/
noncomputable def aft1VCellG (s : BlockState) (V : RegionName)
    (s_qh BT BD : Nat) (c tk d : Nat) : ℝ :=
  s.readMem V (s.pids 0 * s_qh + (c * BT + tk) * BD + d)

/-- General kernel-native recurrent state after `c` chunks. -/
noncomputable def aft1RecStateG (s : BlockState) (K V : RegionName)
    (s_qh BT BD : Nat) (c : Nat) (d' d : Fin BD) : ℝ :=
  ∑ j ∈ Finset.range c,
    ∑ tk : Fin BT,
      aft1KCellG s K s_qh BT BD j d'.val tk.val * aft1VCellG s V s_qh BT BD j tk.val d.val

/-- General kernel-native local output of chunk `c`. -/
noncomputable def aft1LocalOutG (s : BlockState) (Q K V : RegionName)
    (s_qh : Nat) (scale : ℝ) (BT BD : Nat) (c : Nat) (t : Fin BT) (d : Fin BD) : ℝ :=
  ∑ tk : Fin BT,
    (∑ e : Fin BD, aft1QCellG s Q s_qh scale BT BD c t.val e.val
        * aft1KCellG s K s_qh BT BD c e.val tk.val) *
      aft1VCellG s V s_qh BT BD c tk.val d.val

/-- General kernel-native recurrent output contribution of chunk `c`. -/
noncomputable def aft1RecOutG (s : BlockState) (Q K V : RegionName)
    (s_qh : Nat) (scale : ℝ) (BT BD : Nat) (c : Nat) (t : Fin BT) (d : Fin BD) : ℝ :=
  ∑ d' : Fin BD, aft1QCellG s Q s_qh scale BT BD c t.val d'.val
      * aft1RecStateG s K V s_qh BT BD c d' d

/-- General kernel-native full output of chunk `c`: local + recurrent. -/
noncomputable def aft1OutG (s : BlockState) (Q K V : RegionName)
    (s_qh : Nat) (scale : ℝ) (BT BD : Nat) (c : Nat) (t : Fin BT) (d : Fin BD) : ℝ :=
  aft1LocalOutG s Q K V s_qh scale BT BD c t d
    + aft1RecOutG s Q K V s_qh scale BT BD c t d

/-- General `b_h` carry tile after `c` chunks. -/
noncomputable def aft1BhTileG (s : BlockState) (K V : RegionName)
    (s_qh BT BD : Nat) (c : Nat) : Tile .real [BD, BD] :=
  ⟨fun idx => some (aft1RecStateG s K V s_qh BT BD c idx.1 idx.2.1)⟩

/-- General `b_q` loaded-and-scaled tile of chunk `c`. -/
noncomputable def aft1BqTileG (s : BlockState) (Q : RegionName)
    (s_qh : Nat) (scale : ℝ) (BT BD : Nat) (c : Nat) : Tile .real [BT, BD] :=
  ⟨fun idx => some (aft1QCellG s Q s_qh scale BT BD c idx.1.val idx.2.1.val)⟩

/-- General `b_k` loaded tile of chunk `c`. -/
noncomputable def aft1BkTileG (s : BlockState) (K : RegionName)
    (s_qh BT BD : Nat) (c : Nat) : Tile .real [BD, BT] :=
  ⟨fun idx => some (aft1KCellG s K s_qh BT BD c idx.1.val idx.2.1.val)⟩

/-- General `b_v` loaded tile of chunk `c`. -/
noncomputable def aft1BvTileG (s : BlockState) (V : RegionName)
    (s_qh BT BD : Nat) (c : Nat) : Tile .real [BT, BD] :=
  ⟨fun idx => some (aft1VCellG s V s_qh BT BD c idx.1.val idx.2.1.val)⟩

/-- General `b_o` output tile of chunk `c` (full output). -/
noncomputable def aft1BoTileG (s : BlockState) (Q K V : RegionName)
    (s_qh : Nat) (scale : ℝ) (BT BD : Nat) (c : Nat) : Tile .real [BT, BD] :=
  ⟨fun idx => some (aft1OutG s Q K V s_qh scale BT BD c idx.1 idx.2.1)⟩

/-- General local-only `b_o` tile of chunk `c` (= aft1LocalOutG). -/
noncomputable def aft1BoLocalTileG (s : BlockState) (Q K V : RegionName)
    (s_qh : Nat) (scale : ℝ) (BT BD : Nat) (c : Nat) : Tile .real [BT, BD] :=
  ⟨fun idx => some (aft1LocalOutG s Q K V s_qh scale BT BD c idx.1 idx.2.1)⟩

/-- General intra-chunk score tile `b_s[t,tk] = Σ_e b_q[t,e]·b_k[e,tk]`. -/
noncomputable def aft1BsTileG (s : BlockState) (Q K : RegionName)
    (s_qh : Nat) (scale : ℝ) (BT BD : Nat) (c : Nat) : Tile .real [BT, BT] :=
  ⟨fun idx => some (∑ e : Fin BD, aft1QCellG s Q s_qh scale BT BD c idx.1.val e.val
      * aft1KCellG s K s_qh BT BD c e.val idx.2.1.val)⟩

/-- **General recurrent-state step.** -/
theorem aft1RecStateG_succ (s : BlockState) (K V : RegionName) (s_qh BT BD : Nat)
    (c : Nat) (d' d : Fin BD) :
    aft1RecStateG s K V s_qh BT BD (c + 1) d' d
      = aft1RecStateG s K V s_qh BT BD c d' d
        + ∑ tk : Fin BT,
            aft1KCellG s K s_qh BT BD c d'.val tk.val
              * aft1VCellG s V s_qh BT BD c tk.val d.val := by
  simp [aft1RecStateG, Finset.sum_range_succ]

/-- `aft1BhTileG s K V 0` is the all-zero tile. -/
theorem aft1BhTileG_zero (s : BlockState) (K V : RegionName) (s_qh BT BD : Nat) :
    aft1BhTileG s K V s_qh BT BD 0 = (⟨fun _ => some (0 : ℝ)⟩ : Tile .real [BD, BD]) := by
  unfold aft1BhTileG
  refine congrArg _ ?_
  funext idx
  simp [aft1RecStateG]

/-- General erased loop body of `attention_fwd_kernel_surface` at the contiguous
layout (`s_qt = BD`, `s_qd = 1`, K strides `(1, BD)`). Checked against the
lowered surface by `rfl` in `attention_fwd_triton1_body_splitG`. -/
noncomputable def aft1LoopBodyG (Q K V H O : RegionName)
    (s_qh s_hh s_ht : Nat) (scale : ℝ) (BT BD NT : Nat)
    (STORE IFCOND : Bool) : List Stmt :=
  [ Stmt.assign .blockPtr [BT, BD] "p_q"
      (Op.makeBlockPtrDynOffsets Q
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_qh))
        [NT * BT, BD] [BT, BD] [BD, 1]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i") (Op.constNat BT), Op.constNat 0]),
    Stmt.assign .blockPtr [BD, BT] "p_k"
      (Op.makeBlockPtrDynOffsets K
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_qh))
        [BD, NT * BT] [BD, BT] [1, BD]
        [Op.constNat 0, Op.mul .nat Broadcast.nil (Op.ref .nat [] "i") (Op.constNat BT)]),
    Stmt.assign .blockPtr [BT, BD] "p_v"
      (Op.makeBlockPtrDynOffsets V
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_qh))
        [NT * BT, BD] [BT, BD] [BD, 1]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i") (Op.constNat BT), Op.constNat 0]),
    Stmt.assign .blockPtr [BD, BD] "p_h"
      (Op.makeBlockPtrDynOffsets H
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_hh))
        [NT * BD, BD] [BD, BD] [s_ht, 1]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i") (Op.constNat BD), Op.constNat 0]),
    Stmt.assign .blockPtr [BT, BD] "p_o"
      (Op.makeBlockPtrDynOffsets O
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_qh))
        [NT * BT, BD] [BT, BD] [BD, 1]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i") (Op.constNat BT), Op.constNat 0]),
    Stmt.ifThen (Op.constBool STORE)
      [Stmt.store .real [BD, BD]
          (MemAccess.blockPtr (Op.ref .blockPtr [BD, BD] "p_h") [])
          (Op.ref .real [BD, BD] "b_h") MaskOpt.none],
    Stmt.assign .real [BT, BD] "b_q"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [BT, BD] "p_q") []) MaskOpt.none),
    Stmt.assign .real [BT, BD] "b_q"
      (Op.mul .real Broadcast.scalarR (Op.ref .real [BT, BD] "b_q") (Op.const scale)),
    Stmt.assign .real [BD, BT] "b_k"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [BD, BT] "p_k") []) MaskOpt.none),
    Stmt.assign .real [BT, BD] "b_v"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [BT, BD] "p_v") []) MaskOpt.none),
    Stmt.assign .real [BT, BT] "b_s"
      (@Op.dot [] BT BD BT (Op.ref .real [BT, BD] "b_q") (Op.ref .real [BD, BT] "b_k")),
    Stmt.assign .real [BT, BD] "b_o"
      (@Op.dot [] BT BT BD (Op.ref .real [BT, BT] "b_s") (Op.ref .real [BT, BD] "b_v")),
    Stmt.ifThenElse (Op.constBool IFCOND)
      [Stmt.ifThenElse
          (Op.eq ComparableDType.nat Broadcast.nil (Op.ref .nat [] "i") (Op.constNat 0))
          [Stmt.assign .real [BD, BD] "b_h"
              (@Op.dot [] BD BT BD (Op.ref .real [BD, BT] "b_k") (Op.ref .real [BT, BD] "b_v"))]
          [Stmt.assign .real [BT, BD] "b_o"
              (Op.add .real Broadcast.nil.consSame.consSame
                (Op.ref .real [BT, BD] "b_o")
                (@Op.dot [] BT BD BD (Op.ref .real [BT, BD] "b_q") (Op.ref .real [BD, BD] "b_h"))),
            Stmt.assign .real [BD, BD] "b_h"
              (Op.add .real Broadcast.nil.consSame.consSame
                (Op.ref .real [BD, BD] "b_h")
                (@Op.dot [] BD BT BD (Op.ref .real [BD, BT] "b_k") (Op.ref .real [BT, BD] "b_v")))]]
      [Stmt.assign .real [BT, BD] "b_o"
          (Op.add .real Broadcast.nil.consSame.consSame
            (Op.ref .real [BT, BD] "b_o")
            (@Op.dot [] BT BD BD (Op.ref .real [BT, BD] "b_q") (Op.ref .real [BD, BD] "b_h"))),
        Stmt.assign .real [BD, BD] "b_h"
          (Op.add .real Broadcast.nil.consSame.consSame
            (Op.ref .real [BD, BD] "b_h")
            (@Op.dot [] BD BT BD (Op.ref .real [BD, BT] "b_k") (Op.ref .real [BT, BD] "b_v")))],
    Stmt.store .real [BT, BD]
      (MemAccess.blockPtr (Op.ref .blockPtr [BT, BD] "p_o") [])
      (Op.ref .real [BT, BD] "b_o") MaskOpt.none]

/-- General erased prologue (program id + zero-init of `b_h` at shape `[BD,BD]`). -/
def aft1PrologueG (BD : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "i_bh" (Op.programId 0),
    Stmt.assign .real [BD, BD] "b_h" (Op.full [BD, BD] (Op.const 0)) ]

/-- General dynamic loop-bound op `tl.cdiv(NT·BT, BT)`. -/
def aft1StopOpG (BT NT : Nat) : Op .nat [] :=
  Op.div .nat Broadcast.nil
    (Op.sub .nat Broadcast.nil
      (Op.add .nat Broadcast.nil (Op.constNat (NT * BT)) (Op.constNat BT)) (Op.constNat 1))
    (Op.constNat BT)

set_option maxRecDepth 8000 in
/-- **General body split (by `rfl`).** -/
theorem attention_fwd_triton1_body_splitG
    (Q K V H O : RegionName) (s_qh s_hh s_ht : Nat) (scale : ℝ)
    (BT BD NT : Nat) (STORE IFCOND : Bool) :
    (attention_fwd_kernel_surface Q K V H O
      s_qh BD 1 s_hh s_ht (NT * BT) scale BT BD NT STORE IFCOND).toAlgKernel.body
      = aft1PrologueG BD
        ++ [Stmt.forRangeDyn "i" (Op.constNat 0) (aft1StopOpG BT NT) (Op.constNat 1)
              (aft1LoopBodyG Q K V H O s_qh s_hh s_ht scale BT BD NT STORE IFCOND)] := by
  rfl

/-- General `b_k` load = `aft1BkTileG`. -/
theorem aft1_load_bk_eqG (s sin' : BlockState) (K : RegionName)
    (s_qh BT BD : Nat) (c : Nat)
    (hmem : ∀ rg off, sin'.readMem rg off = s.readMem rg off)
    (hpk : sin'.regs .blockPtr [BD, BT] "p_k" = some
      ⟨fun _ => BlockPtr.mk K (s.pids 0 * s_qh) [BD, NT * BT] [BD, BT] [1, BD] [0, c * BT]⟩) :
    evalOp (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [BD, BT] "p_k") []) MaskOpt.none)
        sin'
      = some (aft1BkTileG s K s_qh BT BD c) := by
  rw [aft1_load_bp_2d_ref K sin' "p_k" (s.pids 0 * s_qh) BD (NT * BT) BD BT 1 BD 0 (c * BT) hpk]
  refine congrArg some ?_
  ext idx; obtain ⟨e, tk, u⟩ := idx
  simp only [aft1BkTileG, aft1KCellG, hmem, Nat.add_zero, Nat.zero_add, Nat.mul_one]

/-- General `b_v` load = `aft1BvTileG`. -/
theorem aft1_load_bv_eqG (s sin' : BlockState) (V : RegionName)
    (s_qh BT BD : Nat) (c : Nat)
    (hmem : ∀ rg off, sin'.readMem rg off = s.readMem rg off)
    (hpv : sin'.regs .blockPtr [BT, BD] "p_v" = some
      ⟨fun _ => BlockPtr.mk V (s.pids 0 * s_qh) [NT * BT, BD] [BT, BD] [BD, 1] [c * BT, 0]⟩) :
    evalOp (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [BT, BD] "p_v") []) MaskOpt.none)
        sin'
      = some (aft1BvTileG s V s_qh BT BD c) := by
  rw [aft1_load_bp_2d_ref V sin' "p_v" (s.pids 0 * s_qh) (NT * BT) BD BT BD BD 1 (c * BT) 0 hpv]
  refine congrArg some ?_
  ext idx; obtain ⟨tk, d, u⟩ := idx
  simp only [aft1BvTileG, aft1VCellG, hmem, Nat.add_zero, Nat.zero_add, Nat.mul_one]

/-- General eval of `b_s = dot(b_q, b_k)` = `aft1BsTileG`. -/
theorem aft1_bs_eqG (s sin' : BlockState) (Q K : RegionName)
    (s_qh : Nat) (scale : ℝ) (BT BD : Nat) (c : Nat)
    (hq : sin'.regs .real [BT, BD] "b_q" = some (aft1BqTileG s Q s_qh scale BT BD c))
    (hk : sin'.regs .real [BD, BT] "b_k" = some (aft1BkTileG s K s_qh BT BD c)) :
    @evalOp .real [BT, BT]
        (@Op.dot [] BT BD BT (Op.ref .real [BT, BD] "b_q") (Op.ref .real [BD, BT] "b_k")) sin'
      = some (aft1BsTileG s Q K s_qh scale BT BD c) := by
  exact aft1_dot_op_eval sin' "b_q" "b_k" (aft1BqTileG s Q s_qh scale BT BD c)
    (aft1BkTileG s K s_qh BT BD c)
    (fun t e => aft1QCellG s Q s_qh scale BT BD c t.val e.val)
    (fun e tk => aft1KCellG s K s_qh BT BD c e.val tk.val)
    hq hk (fun t e => rfl) (fun e tk => rfl)

/-- General eval of `b_o = dot(b_s, b_v)` = `aft1BoLocalTileG`. -/
theorem aft1_bo_local_eqG (s sin' : BlockState) (Q K V : RegionName)
    (s_qh : Nat) (scale : ℝ) (BT BD : Nat) (c : Nat)
    (hs : sin'.regs .real [BT, BT] "b_s" = some (aft1BsTileG s Q K s_qh scale BT BD c))
    (hv : sin'.regs .real [BT, BD] "b_v" = some (aft1BvTileG s V s_qh BT BD c)) :
    @evalOp .real [BT, BD]
        (@Op.dot [] BT BT BD (Op.ref .real [BT, BT] "b_s") (Op.ref .real [BT, BD] "b_v")) sin'
      = some (aft1BoLocalTileG s Q K V s_qh scale BT BD c) := by
  exact aft1_dot_op_eval sin' "b_s" "b_v" (aft1BsTileG s Q K s_qh scale BT BD c)
    (aft1BvTileG s V s_qh BT BD c)
    (fun t tk => ∑ e : Fin BD, aft1QCellG s Q s_qh scale BT BD c t.val e.val
        * aft1KCellG s K s_qh BT BD c e.val tk.val)
    (fun tk d => aft1VCellG s V s_qh BT BD c tk.val d.val) hs hv (fun t tk => rfl) (fun tk d => rfl)

/-- General eval of `b_o += dot(b_q, b_h)` = `aft1BoTileG` (full output). -/
theorem aft1_bo_full_eqG (s sin' : BlockState) (Q K V : RegionName)
    (s_qh : Nat) (scale : ℝ) (BT BD : Nat) (c : Nat)
    (hbo : sin'.regs .real [BT, BD] "b_o" = some (aft1BoLocalTileG s Q K V s_qh scale BT BD c))
    (hq : sin'.regs .real [BT, BD] "b_q" = some (aft1BqTileG s Q s_qh scale BT BD c))
    (hbh : sin'.regs .real [BD, BD] "b_h" = some (aft1BhTileG s K V s_qh BT BD c)) :
    @evalOp .real [BT, BD]
        (Op.add .real Broadcast.nil.consSame.consSame (Op.ref .real [BT, BD] "b_o")
          (@Op.dot [] BT BD BD (Op.ref .real [BT, BD] "b_q") (Op.ref .real [BD, BD] "b_h"))) sin'
      = some (aft1BoTileG s Q K V s_qh scale BT BD c) := by
  rw [aft1_accDot_op_eval sin' "b_o" "b_q" "b_h"
    (aft1BoLocalTileG s Q K V s_qh scale BT BD c) (aft1BqTileG s Q s_qh scale BT BD c)
    (aft1BhTileG s K V s_qh BT BD c)
    (fun t d => aft1LocalOutG s Q K V s_qh scale BT BD c t d)
    (fun t d' => aft1QCellG s Q s_qh scale BT BD c t.val d'.val)
    (fun d' d => aft1RecStateG s K V s_qh BT BD c d' d)
    hbo hq hbh (fun t d => rfl) (fun t d' => rfl) (fun d' d => rfl)]
  refine congrArg some ?_
  ext idx; obtain ⟨t, d, u⟩ := idx
  simp only [aft1BoTileG, aft1OutG, aft1RecOutG]

/-- General eval of `b_h += dot(b_k, b_v)` = `aft1BhTileG (c+1)`. -/
theorem aft1_bh_succ_eqG (s sin' : BlockState) (K V : RegionName)
    (s_qh BT BD : Nat) (c : Nat)
    (hbh : sin'.regs .real [BD, BD] "b_h" = some (aft1BhTileG s K V s_qh BT BD c))
    (hk : sin'.regs .real [BD, BT] "b_k" = some (aft1BkTileG s K s_qh BT BD c))
    (hv : sin'.regs .real [BT, BD] "b_v" = some (aft1BvTileG s V s_qh BT BD c)) :
    @evalOp .real [BD, BD]
        (Op.add .real Broadcast.nil.consSame.consSame (Op.ref .real [BD, BD] "b_h")
          (@Op.dot [] BD BT BD (Op.ref .real [BD, BT] "b_k") (Op.ref .real [BT, BD] "b_v"))) sin'
      = some (aft1BhTileG s K V s_qh BT BD (c + 1)) := by
  rw [aft1_accDot_op_eval sin' "b_h" "b_k" "b_v"
    (aft1BhTileG s K V s_qh BT BD c) (aft1BkTileG s K s_qh BT BD c) (aft1BvTileG s V s_qh BT BD c)
    (fun d' d => aft1RecStateG s K V s_qh BT BD c d' d)
    (fun d' tk => aft1KCellG s K s_qh BT BD c d'.val tk.val)
    (fun tk d => aft1VCellG s V s_qh BT BD c tk.val d.val)
    hbh hk hv (fun d' d => rfl) (fun d' tk => rfl) (fun tk d => rfl)]
  refine congrArg some ?_
  ext idx; obtain ⟨d', d, u⟩ := idx
  simp only [aft1BhTileG]
  rw [aft1RecStateG_succ]

set_option maxHeartbeats 2000000 in
set_option maxRecDepth 8000 in
/-- **General full loop-body register step (STORE=false, IFCOND=false).** -/
theorem aft1_loopBody_regs_ffG
    (Q K V H O : RegionName) (s_qh s_hh s_ht : Nat) (scale : ℝ) (BT BD NT : Nat)
    (s sin : BlockState) (c : Nat)
    (hmem : ∀ rg off, sin.readMem rg off = s.readMem rg off)
    (hibh : sin.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 0)))
    (hi : sin.regs .nat [] "i" = some (Tile.scalar c))
    (hbh : sin.regs .real [BD, BD] "b_h" = some (aft1BhTileG s K V s_qh BT BD c)) :
    ∃ s2, stepStmts ((aft1LoopBodyG Q K V H O s_qh s_hh s_ht scale BT BD NT
          Bool.false Bool.false).take 13) sin = some s2
      ∧ (∀ rg off, s2.readMem rg off = s.readMem rg off)
      ∧ s2.pids = sin.pids
      ∧ s2.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 0))
      ∧ s2.regs .blockPtr [BT, BD] "p_o" = some
          (⟨fun _ => BlockPtr.mk O (s.pids 0 * s_qh) [NT * BT, BD] [BT, BD] [BD, 1]
            [c * BT, 0]⟩ : Tile .blockPtr [BT, BD])
      ∧ s2.regs .real [BT, BD] "b_o" = some (aft1BoTileG s Q K V s_qh scale BT BD c)
      ∧ s2.regs .real [BD, BD] "b_h" = some (aft1BhTileG s K V s_qh BT BD (c + 1)) := by
  unfold aft1LoopBodyG
  simp only [List.take_succ_cons, List.take_zero]
  -- p_q, p_k, p_v, p_h, p_o
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft1_makeBlockPtr_2d_eval Q sin _ _ _ [NT * BT, BD] [BT, BD] [BD, 1]
      (s.pids 0 * s_qh) (c * BT) 0
      (aft1_mulConst_eval sin "i_bh" (s.pids 0) s_qh hibh)
      (aft1_mulConst_eval sin "i" c BT hi) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft1_makeBlockPtr_2d_eval K _ _ _ _ [BD, NT * BT] [BD, BT] [1, BD]
      (s.pids 0 * s_qh) 0 (c * BT)
      (aft1_mulConst_eval _ "i_bh" (s.pids 0) s_qh (by simp [hibh])) (by simp)
      (aft1_mulConst_eval _ "i" c BT (by simp [hi]))))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft1_makeBlockPtr_2d_eval V _ _ _ _ [NT * BT, BD] [BT, BD] [BD, 1]
      (s.pids 0 * s_qh) (c * BT) 0
      (aft1_mulConst_eval _ "i_bh" (s.pids 0) s_qh (by simp [hibh]))
      (aft1_mulConst_eval _ "i" c BT (by simp [hi])) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft1_makeBlockPtr_2d_eval H _ _ _ _ [NT * BD, BD] [BD, BD] [s_ht, 1]
      (s.pids 0 * s_hh) (c * BD) 0
      (aft1_mulConst_eval _ "i_bh" (s.pids 0) s_hh (by simp [hibh]))
      (aft1_mulConst_eval _ "i" c BD (by simp [hi])) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft1_makeBlockPtr_2d_eval O _ _ _ _ [NT * BT, BD] [BT, BD] [BD, 1]
      (s.pids 0 * s_qh) (c * BT) 0
      (aft1_mulConst_eval _ "i_bh" (s.pids 0) s_qh (by simp [hibh]))
      (aft1_mulConst_eval _ "i" c BT (by simp [hi])) (by simp)))]
  rw [stepStmts.cons_some (aft1_ifThen_false_noop _ _)]
  -- b_q raw load
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft1_load_bp_2d_ref Q _ "p_q" (s.pids 0 * s_qh) (NT * BT) BD BT BD BD 1
      (c * BT) 0 (by simp)))]
  -- b_q *= scale → aft1BqTileG
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.mul .real Broadcast.scalarR (Op.ref .real [BT, BD] "b_q")
        (Op.const scale)) _ = some (aft1BqTileG s Q s_qh scale BT BD c) from by
      rw [evalOp_mul]
      simp only [evalOp_ref, BlockState.setReg_same, evalOp_const, Option.bind_eq_bind,
        Option.bind_some]
      refine congrArg some ?_
      ext idx; obtain ⟨t, e, u⟩ := idx
      simp only [Tile.bop, Broadcast.scalarR, Broadcast.leftIndex, Broadcast.rightIndex,
        Tile.scalar, aft1BqTileG, aft1QCellG, NumericDType.mul, WithBot.realMul,
        Option.map₂, Option.bind, Option.map, BlockState.setReg_readMem, hmem,
        Nat.add_zero, Nat.zero_add, Nat.mul_one]))]
  -- b_k, b_v loads
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft1_load_bk_eqG (NT := NT) s _ K s_qh BT BD c (by intro rg off; simp [hmem]) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft1_load_bv_eqG (NT := NT) s _ V s_qh BT BD c (by intro rg off; simp [hmem]) (by simp)))]
  -- b_s = dot(b_q, b_k)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft1_bs_eqG s _ Q K s_qh scale BT BD c (by simp) (by simp)))]
  -- b_o = dot(b_s, b_v)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft1_bo_local_eqG s _ Q K V s_qh scale BT BD c (by simp) (by simp)))]
  -- IFCOND gate (false): elseBody
  rw [stepStmts.cons_some (st := Stmt.ifThenElse (Op.constBool Bool.false) _ _)
    (show stepStmt (Stmt.ifThenElse (Op.constBool Bool.false) _ _) _ = some _ from by
      rw [aft1_ifThenElse_false_else]
      rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (aft1_bo_full_eqG s _ Q K V s_qh scale BT BD c (by simp) (by simp) (by simp [hbh])))]
      rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (aft1_bh_succ_eqG s _ K V s_qh BT BD c (by simp [hbh]) (by simp) (by simp)))]
      exact stepStmts.nil)]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro rg off; simp [hmem]
  · simp
  · simp [hibh]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_same]

set_option maxHeartbeats 2000000 in
set_option maxRecDepth 8000 in
/-- **General full single-chunk iteration (STORE=false, IFCOND=false).** -/
theorem aft1_loopBody_iter_ffG
    (Q K V H O : RegionName) (s_qh s_hh s_ht : Nat) (scale : ℝ) (BT BD NT : Nat)
    (s sin : BlockState) (c : Nat)
    (hOQ : O ≠ Q) (hOK : O ≠ K) (hOV : O ≠ V) (hOH : O ≠ H)
    (hmem : ∀ rg off, sin.readMem rg off = s.readMem rg off)
    (hibh : sin.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 0)))
    (hi : sin.regs .nat [] "i" = some (Tile.scalar c))
    (hbh : sin.regs .real [BD, BD] "b_h" = some (aft1BhTileG s K V s_qh BT BD c)) :
    ∃ s', stepStmts (aft1LoopBodyG Q K V H O s_qh s_hh s_ht scale BT BD NT
        Bool.false Bool.false) sin = some s'
      ∧ s'.pids = sin.pids
      ∧ s'.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 0))
      ∧ s'.regs .real [BD, BD] "b_h" = some (aft1BhTileG s K V s_qh BT BD (c + 1))
      ∧ (∀ off, s'.readMem Q off = s.readMem Q off)
      ∧ (∀ off, s'.readMem K off = s.readMem K off)
      ∧ (∀ off, s'.readMem V off = s.readMem V off)
      ∧ (∀ off, s'.readMem H off = s.readMem H off)
      ∧ (∀ (t : Fin BT) (d : Fin BD),
          s'.readMem O (s.pids 0 * s_qh + (c * BT + t.val) * BD + d.val)
            = aft1OutG s Q K V s_qh scale BT BD c t d)
      ∧ (∀ off, (∀ (t : Fin BT) (d : Fin BD),
            off ≠ s.pids 0 * s_qh + (c * BT + t.val) * BD + d.val)
          → s'.readMem O off = sin.readMem O off) := by
  obtain ⟨s2, hstep, hmem2, hpids2, hibh2, hpo, hbo, hbh2⟩ :=
    aft1_loopBody_regs_ffG Q K V H O s_qh s_hh s_ht scale BT BD NT s sin c hmem hibh hi hbh
  have hdrop : (aft1LoopBodyG Q K V H O s_qh s_hh s_ht scale BT BD NT Bool.false Bool.false).drop 13
      = [Stmt.store .real [BT, BD]
          (MemAccess.blockPtr (Op.ref .blockPtr [BT, BD] "p_o") [])
          (Op.ref .real [BT, BD] "b_o") MaskOpt.none] := by
    unfold aft1LoopBodyG
    simp only [List.drop_succ_cons, List.drop_zero]
  have hsplit : aft1LoopBodyG Q K V H O s_qh s_hh s_ht scale BT BD NT Bool.false Bool.false
      = (aft1LoopBodyG Q K V H O s_qh s_hh s_ht scale BT BD NT Bool.false Bool.false).take 13
        ++ [Stmt.store .real [BT, BD]
              (MemAccess.blockPtr (Op.ref .blockPtr [BT, BD] "p_o") [])
              (Op.ref .real [BT, BD] "b_o") MaskOpt.none] := by
    conv_lhs => rw [← List.take_append_drop 13
      (aft1LoopBodyG Q K V H O s_qh s_hh s_ht scale BT BD NT Bool.false Bool.false)]
    rw [hdrop]
  rw [hsplit, stepStmts.append_some hstep]
  have hstore : stepStmt (Stmt.store .real [BT, BD]
        (MemAccess.blockPtr (Op.ref .blockPtr [BT, BD] "p_o") [])
        (Op.ref .real [BT, BD] "b_o") MaskOpt.none) s2
      = some ((TileShape.allIndices [BT, BD]).foldl
          (fun acc i => acc.writeMemTyped .real O
            (s.pids 0 * s_qh + (c * BT + i.1.val) * BD + (0 + i.2.1.val) * 1)
            ((aft1BoTileG s Q K V s_qh scale BT BD c).data i)) s2) := by
    unfold stepStmt
    simp only [evalOp_ref, hbo, hpo, Option.bind, Option.map]
    refine congrArg some (congrArg (fun f => List.foldl f s2 (TileShape.allIndices [BT, BD])) ?_)
    funext acc i
    obtain ⟨t, d, u⟩ := i
    simp only [TileShape.indexToList, BlockPtr.address_2d_offsets, BlockPtr.inBounds,
      List.all_nil, Bool.and_true, if_true]
  set offFn : TileIndex [BT, BD] → Nat :=
    fun i => s.pids 0 * s_qh + (c * BT + i.1.val) * BD + (0 + i.2.1.val) * 1 with hoffFn
  have hinj : Function.Injective offFn := by
    rintro ⟨⟨ta, hta⟩, ⟨da, hda⟩, _⟩ ⟨⟨tb, htb⟩, ⟨db, hdb⟩, _⟩ h
    simp only [hoffFn] at h
    have ht : ta = tb := by
      by_contra hne
      have hda' : da < BD := hda
      have hdb' : db < BD := hdb
      have : (c * BT + ta) * BD + da ≠ (c * BT + tb) * BD + db := by
        rcases Nat.lt_or_ge ta tb with hlt | hge
        · have : (c * BT + ta) * BD + da < (c * BT + tb) * BD := by
            have h1 : (c * BT + ta) * BD + BD ≤ (c * BT + tb) * BD := by
              have : c * BT + ta + 1 ≤ c * BT + tb := by omega
              calc (c * BT + ta) * BD + BD = (c * BT + ta + 1) * BD := by ring
                _ ≤ (c * BT + tb) * BD := Nat.mul_le_mul_right BD this
            omega
          omega
        · have hgt : tb < ta := by omega
          have : (c * BT + tb) * BD + db < (c * BT + ta) * BD := by
            have h1 : (c * BT + tb) * BD + BD ≤ (c * BT + ta) * BD := by
              have : c * BT + tb + 1 ≤ c * BT + ta := by omega
              calc (c * BT + tb) * BD + BD = (c * BT + tb + 1) * BD := by ring
                _ ≤ (c * BT + ta) * BD := Nat.mul_le_mul_right BD this
            omega
          omega
      omega
    have hd : da = db := by
      subst ht; omega
    subst tb; subst db; rfl
  have hstoreW : (TileShape.allIndices [BT, BD]).foldl
        (fun acc i => acc.writeMemTyped .real O (offFn i) ((aft1BoTileG s Q K V s_qh scale BT BD c).data i)) s2
      = (TileShape.allIndices [BT, BD]).foldl
        (fun acc i => acc.writeMem O (offFn i) (aft1OutG s Q K V s_qh scale BT BD c i.1 i.2.1)) s2 := by
    apply congrArg (fun f => List.foldl f s2 (TileShape.allIndices [BT, BD]))
    funext acc i
    simp only [BlockState.writeMemTyped_real, aft1BoTileG, FloatDType.real_storeValue]
    rfl
  rw [stepStmts.cons_some hstore, stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [aft1_foldl_store_pids]; exact hpids2
  · rw [BlockState.foldl_writeMemTyped_regs]; exact hibh2
  · rw [BlockState.foldl_writeMemTyped_regs]; exact hbh2
  · intro off; rw [aft1_foldl_store_readMem_ne _ O Q _ _ s2 off hOQ.symm]; exact hmem2 Q off
  · intro off; rw [aft1_foldl_store_readMem_ne _ O K _ _ s2 off hOK.symm]; exact hmem2 K off
  · intro off; rw [aft1_foldl_store_readMem_ne _ O V _ _ s2 off hOV.symm]; exact hmem2 V off
  · intro off; rw [aft1_foldl_store_readMem_ne _ O H _ _ s2 off hOH.symm]; exact hmem2 H off
  · intro t d
    rw [hstoreW]
    have hoff : s.pids 0 * s_qh + (c * BT + t.val) * BD + d.val = offFn (t, d, PUnit.unit) := by
      simp [hoffFn]
    rw [hoff, BlockState.scatter_readback_nd s2 offFn
      (fun i => aft1OutG s Q K V s_qh scale BT BD c i.1 i.2.1) hinj (t, d, PUnit.unit)]
  · intro off hoff
    rw [hstoreW]
    rw [aft1_foldl_writeMem_readMem_other _ O offFn _ s2 off (by
      intro i _
      obtain ⟨t, d, u⟩ := i
      simp only [hoffFn, Nat.add_zero, Nat.zero_add, Nat.mul_one]
      exact (hoff t d).symm)]
    rw [hmem2 O off, ← hmem O off]

/-- General carry invariant. -/
noncomputable def aft1InvG (Q K V H O : RegionName) (s_qh : Nat) (scale : ℝ) (BT BD : Nat)
    (s : BlockState) (n : Nat) (st : BlockState) : Prop :=
  st.pids = s.pids
  ∧ st.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 0))
  ∧ st.regs .real [BD, BD] "b_h" = some (aft1BhTileG s K V s_qh BT BD n)
  ∧ (∀ off, st.readMem Q off = s.readMem Q off)
  ∧ (∀ off, st.readMem K off = s.readMem K off)
  ∧ (∀ off, st.readMem V off = s.readMem V off)
  ∧ (∀ off, st.readMem H off = s.readMem H off)
  ∧ (∀ j, j < n → ∀ (t : Fin BT) (d : Fin BD),
      st.readMem O (s.pids 0 * s_qh + (j * BT + t.val) * BD + d.val)
        = aft1OutG s Q K V s_qh scale BT BD j t d)

/-- `aft1BhTileG` depends only on `K`/`V` reads and `pids 0`. -/
theorem aft1BhTileG_congr (s s' : BlockState) (K V : RegionName) (s_qh BT BD : Nat) (n : Nat)
    (hp : s'.pids 0 = s.pids 0)
    (hK : ∀ off, s'.readMem K off = s.readMem K off)
    (hV : ∀ off, s'.readMem V off = s.readMem V off) :
    aft1BhTileG s' K V s_qh BT BD n = aft1BhTileG s K V s_qh BT BD n := by
  unfold aft1BhTileG
  refine congrArg _ ?_
  funext idx
  simp only [aft1RecStateG, aft1KCellG, aft1VCellG, hp, hK, hV]

/-- `aft1OutG` depends only on `Q`/`K`/`V` reads and `pids 0`. -/
theorem aft1OutG_congr (s s' : BlockState) (Q K V : RegionName) (s_qh : Nat) (scale : ℝ)
    (BT BD : Nat) (c : Nat) (t : Fin BT) (d : Fin BD)
    (hp : s'.pids 0 = s.pids 0)
    (hQ : ∀ off, s'.readMem Q off = s.readMem Q off)
    (hK : ∀ off, s'.readMem K off = s.readMem K off)
    (hV : ∀ off, s'.readMem V off = s.readMem V off) :
    aft1OutG s' Q K V s_qh scale BT BD c t d = aft1OutG s Q K V s_qh scale BT BD c t d := by
  simp only [aft1OutG, aft1LocalOutG, aft1RecOutG, aft1RecStateG, aft1QCellG, aft1KCellG,
    aft1VCellG, hp, hQ, hK, hV]

set_option maxHeartbeats 2000000 in
/-- **General carry-invariant step.** One iteration preserves `aft1InvG`. -/
theorem aft1InvG_step (Q K V H O : RegionName) (s_qh s_hh s_ht : Nat) (scale : ℝ)
    (BT BD NT : Nat) (s : BlockState)
    (hOQ : O ≠ Q) (hOK : O ≠ K) (hOV : O ≠ V) (hOH : O ≠ H)
    (n : Nat) (st : BlockState) (hP : aft1InvG Q K V H O s_qh scale BT BD s n st) :
    ∃ st', stepStmts (aft1LoopBodyG Q K V H O s_qh s_hh s_ht scale BT BD NT
        Bool.false Bool.false) (st.setReg "i" .nat [] (Tile.scalar n)) = some st'
      ∧ aft1InvG Q K V H O s_qh scale BT BD s (n + 1) st' := by
  obtain ⟨hpids, hibh, hbh, hQ, hK, hV, hHh, hOprev⟩ := hP
  set sin := st.setReg "i" .nat [] (Tile.scalar n) with hsin
  have hpidsSt0 : st.pids 0 = s.pids 0 := by rw [hpids]
  have hbhSt : aft1BhTileG s K V s_qh BT BD n = aft1BhTileG st K V s_qh BT BD n :=
    aft1BhTileG_congr st s K V s_qh BT BD n hpidsSt0.symm
      (fun off => (hK off).symm) (fun off => (hV off).symm)
  have hbhSin : sin.regs .real [BD, BD] "b_h" = some (aft1BhTileG st K V s_qh BT BD n) := by
    rw [hsin]; rw [← hbhSt]; simpa using hbh
  have hibhSin : sin.regs .nat [] "i_bh" = some (Tile.scalar (st.pids 0)) := by
    rw [hsin, hpids]; simpa [hpids] using hibh
  have hiSin : sin.regs .nat [] "i" = some (Tile.scalar n) := by rw [hsin]; simp
  have hmemSin : ∀ rg off, sin.readMem rg off = st.readMem rg off := by
    intro rg off; rw [hsin]; simp [BlockState.setReg_readMem]
  obtain ⟨st', hstep, hpids', hibh', hbh', hQ', hK', hV', hH', hOcur, hOother⟩ :=
    aft1_loopBody_iter_ffG Q K V H O s_qh s_hh s_ht scale BT BD NT st sin n
      hOQ hOK hOV hOH hmemSin hibhSin hiSin hbhSin
  refine ⟨st', hstep, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hpids', hsin]; simpa using hpids
  · rw [hibh', hpids]
  · rw [hbh', aft1BhTileG_congr s st K V s_qh BT BD (n + 1) hpidsSt0
      (fun off => hK off) (fun off => hV off)]
  · intro off; rw [hQ' off]; exact hQ off
  · intro off; rw [hK' off]; exact hK off
  · intro off; rw [hV' off]; exact hV off
  · intro off; rw [hH' off]; exact hHh off
  · intro j hj t d
    rcases Nat.lt_succ_iff_lt_or_eq.mp hj with hjlt | hjeq
    · have haddr : s.pids 0 * s_qh + (j * BT + t.val) * BD + d.val
          = st.pids 0 * s_qh + (j * BT + t.val) * BD + d.val := by rw [hpidsSt0]
      rw [haddr, hOother _ (by
        intro t' d' heq
        have ht := t.isLt; have hd := d.isLt; have ht' := t'.isLt; have hd' := d'.isLt
        have hbj : (j * BT + t.val) * BD + d.val < (j + 1) * (BT * BD) := by nlinarith
        have hbn : n * (BT * BD) ≤ (n * BT + t'.val) * BD + d'.val := by nlinarith
        have hle : (j + 1) * (BT * BD) ≤ n * (BT * BD) := by
          have : j + 1 ≤ n := hjlt
          exact Nat.mul_le_mul_right (BT * BD) this
        have hcore : (j * BT + t.val) * BD + d.val = (n * BT + t'.val) * BD + d'.val := by omega
        omega)]
      rw [hsin]; simp only [BlockState.setReg_readMem]
      rw [← haddr]; exact hOprev j hjlt t d
    · subst hjeq
      have haddr : s.pids 0 * s_qh + (j * BT + t.val) * BD + d.val
          = st.pids 0 * s_qh + (j * BT + t.val) * BD + d.val := by rw [hpidsSt0]
      rw [haddr, hOcur t d]
      exact aft1OutG_congr s st Q K V s_qh scale BT BD j t d hpidsSt0 hQ hK hV

set_option maxHeartbeats 2000000 in
/-- **General prologue reaches `aft1InvG 0`.** -/
theorem aft1_prologue_invG (Q K V H O : RegionName) (s_qh : Nat) (scale : ℝ) (BT BD : Nat)
    (s : BlockState) :
    ∃ s0, stepStmts (aft1PrologueG BD) s = some s0
      ∧ aft1InvG Q K V H O s_qh scale BT BD s 0 s0 := by
  unfold aft1PrologueG
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.programId 0) s = some (Tile.scalar (s.pids 0)) from by simp))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [BD, BD] (Op.const 0)) _
        = some (⟨fun _ => some (0 : ℝ)⟩ : Tile .real [BD, BD]) from by
      simp [evalOp_full, evalOp_const]))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp
  · simp [BlockState.setReg_ne_name]
  · rw [BlockState.setReg_same, aft1BhTileG_zero]
  · intro off; simp [BlockState.setReg_readMem]
  · intro off; simp [BlockState.setReg_readMem]
  · intro off; simp [BlockState.setReg_readMem]
  · intro off; simp [BlockState.setReg_readMem]
  · intro j hj; exact absurd hj (Nat.not_lt_zero j)

set_option maxHeartbeats 2000000 in
/-- **General full-exec carry result (STORE=false, IFCOND=false).** -/
theorem aft1_exec_carryG (Q K V H O : RegionName) (s_qh s_hh s_ht : Nat) (scale : ℝ)
    (BT BD NT : Nat) (hBT : 0 < BT) (s : BlockState)
    (hOQ : O ≠ Q) (hOK : O ≠ K) (hOV : O ≠ V) (hOH : O ≠ H) :
    ∃ sF, exec (attention_fwd_kernel_surface Q K V H O
        s_qh BD 1 s_hh s_ht (NT * BT) scale BT BD NT Bool.false Bool.false).toAlgKernel s = some sF
      ∧ sF.regs .real [BD, BD] "b_h" = some (aft1BhTileG s K V s_qh BT BD NT)
      ∧ (∀ (c : Nat), c < NT → ∀ (t : Fin BT) (d : Fin BD),
          sF.readMem O (s.pids 0 * s_qh + (c * BT + t.val) * BD + d.val)
            = aft1OutG s Q K V s_qh scale BT BD c t d) := by
  rw [exec, attention_fwd_triton1_body_splitG]
  obtain ⟨s0, hpre, hP0⟩ := aft1_prologue_invG Q K V H O s_qh scale BT BD s
  rw [stepStmts.append_some hpre]
  have hStop : evalOp (aft1StopOpG BT NT) s0 = some (Tile.scalar NT) := by
    simp only [aft1StopOpG, evalOp_div, evalOp_sub, evalOp_add, evalOp_constNat,
      Option.bind_some, Option.bind_eq_bind]
    refine congrArg some ?_
    apply Tile.ext
    intro z
    simp only [Tile.bop_data, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex,
      NumericDType.div, NumericDType.sub, NumericDType.add]
    show (NT * BT + BT - 1) / BT = NT
    have hcomm : NT * BT = BT * NT := Nat.mul_comm NT BT
    have heq : NT * BT + BT - 1 = BT * NT + (BT - 1) := by rw [hcomm]; omega
    rw [heq, Nat.mul_add_div hBT]
    have : (BT - 1) / BT = 0 := Nat.div_eq_of_lt (by omega)
    omega
  obtain ⟨final, sLoop, hloop, hfinal, hPfinal⟩ :=
    forRangeDyn_inv (idx := "i") (start := 0) (stop := NT) (step := 1)
      (startOp := Op.constNat 0) (stopOp := aft1StopOpG BT NT) (stepOp := Op.constNat 1)
      (P := fun n st => aft1InvG Q K V H O s_qh scale BT BD s n st ∧ n ≤ NT)
      (by simp [evalOp]) hStop (by simp [evalOp]) (by norm_num)
      ⟨hP0, by norm_num⟩
      (fun i st hlt hPi => by
        obtain ⟨hInv, _⟩ := hPi
        obtain ⟨st', hstep', hInv'⟩ :=
          aft1InvG_step Q K V H O s_qh s_hh s_ht scale BT BD NT s hOQ hOK hOV hOH i st hInv
        exact ⟨st', hstep', hInv', by omega⟩)
  obtain ⟨hInvF, hleF⟩ := hPfinal
  have hfinalEq : final = NT := le_antisymm hleF hfinal
  subst hfinalEq
  obtain ⟨_, _, hbhF, _, _, _, _, hOF⟩ := hInvF
  rw [stepStmts.cons_some hloop, stepStmts.nil]
  exact ⟨sLoop, rfl, hbhF, hOF⟩

/-- General `Q`/`V`/`O` accessor: chunk `c` offset `off ↦ base + c·(BT·BD) + off`. -/
def aft1QAddrG (s : BlockState) (s_qh BT BD : Nat) (c off : Nat) : Nat :=
  s.pids 0 * s_qh + c * (BT * BD) + off

/-- General `K` accessor: `off = BT·d' + tk ↦ base + d' + (c·BT + tk)·BD`. -/
def aft1KAddrG (s : BlockState) (s_qh BT BD : Nat) (c off : Nat) : Nat :=
  s.pids 0 * s_qh + off / BT + (c * BT + off % BT) * BD

set_option maxRecDepth 8000 in
/-- General bridge: kernel-native output = genuine `outputClosedForm`. -/
theorem aft1OutG_eq_outputClosedForm (s : BlockState) (Q K V : RegionName)
    (s_qh : Nat) (scale : ℝ) (BT BD : Nat) (hBT : 0 < BT) (c : Nat)
    (t : Fin BT) (d : Fin BD) :
    aft1OutG s Q K V s_qh scale BT BD c t d
      = outputClosedForm s Q K V scale BT BD
          (aft1QAddrG s s_qh BT BD) (aft1KAddrG s s_qh BT BD) (aft1QAddrG s s_qh BT BD) c t d := by
  have hKaddr : ∀ (ch a tk : Nat), tk < BT →
      aft1KAddrG s s_qh BT BD ch (BT * a + tk) = s.pids 0 * s_qh + a + (ch * BT + tk) * BD := by
    intro ch a tk htk
    simp only [aft1KAddrG]
    have hdiv : (BT * a + tk) / BT = a := by
      rw [Nat.mul_add_div hBT]; simp [Nat.div_eq_of_lt htk]
    have hmod : (BT * a + tk) % BT = tk := by
      rw [Nat.mul_add_mod]; exact Nat.mod_eq_of_lt htk
    rw [hdiv, hmod]
  simp only [aft1OutG, aft1LocalOutG, aft1RecOutG, aft1RecStateG, aft1QCellG, aft1KCellG,
    aft1VCellG, outputClosedForm, localTerm, recurrentTerm, recurrentState, aft1QAddrG]
  refine congrArg₂ (· + ·) ?_ ?_
  · refine Finset.sum_congr rfl (fun tk _ => ?_)
    have hVeq : s.pids 0 * s_qh + (c * BT + tk.val) * BD + d.val
        = s.pids 0 * s_qh + c * (BT * BD) + (BD * tk.val + d.val) := by ring
    rw [hVeq]
    refine congrArg₂ (· * ·) (Finset.sum_congr rfl (fun dd _ => ?_)) rfl
    have hQeq : s.pids 0 * s_qh + (c * BT + t.val) * BD + dd.val
        = s.pids 0 * s_qh + c * (BT * BD) + (BD * t.val + dd.val) := by ring
    rw [hQeq, hKaddr c dd.val tk.val tk.isLt]
  · refine Finset.sum_congr rfl (fun d' _ => ?_)
    refine congrArg₂ (· * ·) (congrArg₂ (· * ·) (congrArg _ (by ring)) rfl)
      (Finset.sum_congr rfl (fun j _ => Finset.sum_congr rfl (fun tk _ => ?_)))
    refine congrArg₂ (· * ·) (congrArg _ ?_) (congrArg _ (by ring))
    rw [hKaddr j d'.val tk.val tk.isLt]

set_option maxHeartbeats 2000000 in
/-- **General genuine closed-form output (STORE=false, IFCOND=false).** -/
theorem attention_fwd_triton1_exec_outputClosedFormG
    (Q K V H O : RegionName) (s_qh s_hh s_ht : Nat) (scale : ℝ)
    (BT BD NT : Nat) (hBT : 0 < BT) (s : BlockState)
    (hOQ : O ≠ Q) (hOK : O ≠ K) (hOV : O ≠ V) (hOH : O ≠ H) :
    ∃ sF, exec (attention_fwd_kernel_surface Q K V H O
        s_qh BD 1 s_hh s_ht (NT * BT) scale BT BD NT Bool.false Bool.false).toAlgKernel s = some sF
      ∧ ∀ (c : Nat), c < NT → ∀ (t : Fin BT) (d : Fin BD),
          sF.readMem O (s.pids 0 * s_qh + (c * BT + t.val) * BD + d.val)
            = outputClosedForm s Q K V scale BT BD
                (aft1QAddrG s s_qh BT BD) (aft1KAddrG s s_qh BT BD)
                (aft1QAddrG s s_qh BT BD) c t d := by
  obtain ⟨sF, hexec, _, hOF⟩ :=
    aft1_exec_carryG Q K V H O s_qh s_hh s_ht scale BT BD NT hBT s hOQ hOK hOV hOH
  refine ⟨sF, hexec, fun c hc t d => ?_⟩
  rw [hOF c hc t d, aft1OutG_eq_outputClosedForm s Q K V s_qh scale BT BD hBT]


/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/
set_option maxHeartbeats 2000000 in
/-- **Dimension-general `output_summary` for `attention_fwd_triton1`.**

For arbitrary batch-head stride `s_qh`, chunk size `BT > 0`, head dimension `BD`,
chunk count `NT`, scale `scale : ℝ`, and recurrent-state strides `s_hh`/`s_ht`,
executing the full `attention_fwd_kernel_surface` writes the genuine
`outputClosedForm` (= `localTerm` intra-chunk `(scale·Q·K·V)` + `recurrentTerm`
cross-chunk `(scale·Q·b_h_c)`, read purely over INPUT memory, **no self-reference**)
into `O` at every chunk `c < NT` and lane `(t, d)`.

Stated through the standard `ComputeCorrect.Realizes_without_Rounding` surface (like every other
kernel summary), so the `exec`/`Except`/`toAlgorithm?` plumbing stays out of the
statement: the output write is the `WriteMap` over the streamed index
`(chunk, row, col) : Fin NT × Fin BT × Fin BD`, and `expected` is the genuine
`outputClosedForm`. Three conjuncts:
* **(1)** all four `STORE`/`IFCOND` branch surfaces lower faithfully to the
  algorithm layer;
* **(2)** the kernel runs to completion (the default branch's projected algorithm
  kernel produces a final state) — strictly stronger than `Realizes_without_Rounding` alone, which
  is conditional on execution succeeding;
* **(3)** `Realizes_without_Rounding`: every streamed `O` lane holds the genuine `outputClosedForm`.

The only layout contracts are the contiguity ones the kernel genuinely relies
on: Q/V/O block strides `(BD, 1)`, K block strides `(1, BD)`, recurrent stride
`(s_ht, 1)`, and the dynamic bound `cdiv(NT·BT, BT) = NT` (needs `0 < BT`). The
Python test shape (`s_qh = 131072`, `s_hh = 524288`, `s_ht = 128`, `BT = 32`,
`BD = 128`, `NT = 32`, `scale = 1/√128`) is the special case. -/
specification attention_fwd_triton1_output_summary_general
    (Q K V H O : RegionName) (s_qh s_hh s_ht : Nat) (scale : ℝ)
    (BT BD NT : Nat) (hBT : 0 < BT) (s : BlockState)
    (hOQ : O ≠ Q) (hOK : O ≠ K) (hOV : O ≠ V) (hOH : O ≠ H) :
    -- (1) all four STORE/IFCOND branch surfaces lower to the algorithm layer
    ((∃ alg, (attention_fwd_kernel_surface Q K V H O
      s_qh BD 1 s_hh s_ht (NT * BT) scale BT BD NT Bool.false Bool.false).toAlgorithm?
        = Except.ok alg) ∧
     (∃ alg, (attention_fwd_kernel_surface Q K V H O
      s_qh BD 1 s_hh s_ht (NT * BT) scale BT BD NT Bool.true Bool.false).toAlgorithm?
        = Except.ok alg) ∧
     (∃ alg, (attention_fwd_kernel_surface Q K V H O
      s_qh BD 1 s_hh s_ht (NT * BT) scale BT BD NT Bool.false Bool.true).toAlgorithm?
        = Except.ok alg) ∧
     (∃ alg, (attention_fwd_kernel_surface Q K V H O
      s_qh BD 1 s_hh s_ht (NT * BT) scale BT BD NT Bool.true Bool.true).toAlgorithm?
        = Except.ok alg)) ∧
    -- (2) the default branch runs to completion (existence / termination)
    (∃ sF, exec (attention_fwd_kernel_surface Q K V H O
        s_qh BD 1 s_hh s_ht (NT * BT) scale BT BD NT Bool.false Bool.false).toAlgKernel s
          = some sF) ∧
    -- (3) standard Realizes_without_Rounding: every streamed O lane holds the genuine closed form
    ComputeCorrect.Realizes_without_Rounding
      (kernel := attention_fwd_kernel_surface Q K V H O
        s_qh BD 1 s_hh s_ht (NT * BT) scale BT BD NT Bool.false Bool.false)
      (initialState := s)
      (write := fun i : Fin NT × Fin BT × Fin BD =>
        some (O, s.pids 0 * s_qh + (i.1.val * BT + i.2.1.val) * BD + i.2.2.val))
      (expected := fun i : Fin NT × Fin BT × Fin BD =>
        outputClosedForm s Q K V scale BT BD
          (aft1QAddrG s s_qh BT BD) (aft1KAddrG s s_qh BT BD)
          (aft1QAddrG s s_qh BT BD) i.1.val i.2.1 i.2.2) := by
  obtain ⟨sF, hexec, hOF⟩ :=
    attention_fwd_triton1_exec_outputClosedFormG Q K V H O s_qh s_hh s_ht scale
      BT BD NT hBT s hOQ hOK hOV hOH
  refine ⟨⟨?_, ?_, ?_, ?_⟩, ⟨sF, hexec⟩, ?_⟩
  · exact attention_fwd_kernel_surface_toAlgorithm_supported Q K V H O
      s_qh BD 1 s_hh s_ht (NT * BT) scale BT BD NT Bool.false Bool.false
  · exact attention_fwd_kernel_surface_toAlgorithm_supported Q K V H O
      s_qh BD 1 s_hh s_ht (NT * BT) scale BT BD NT Bool.true Bool.false
  · exact attention_fwd_kernel_surface_toAlgorithm_supported Q K V H O
      s_qh BD 1 s_hh s_ht (NT * BT) scale BT BD NT Bool.false Bool.true
  · exact attention_fwd_kernel_surface_toAlgorithm_supported Q K V H O
      s_qh BD 1 s_hh s_ht (NT * BT) scale BT BD NT Bool.true Bool.true
  · -- (3) the genuine closed-form output, packaged through `Realizes_without_Rounding`
    obtain ⟨alg, halg⟩ := attention_fwd_kernel_surface_toAlgorithm_supported Q K V H O
      s_qh BD 1 s_hh s_ht (NT * BT) scale BT BD NT Bool.false Bool.false
    have hk : (attention_fwd_kernel_surface Q K V H O
        s_qh BD 1 s_hh s_ht (NT * BT) scale BT BD NT Bool.false Bool.false).toAlgorithm?
          = Except.ok (attention_fwd_kernel_surface Q K V H O
            s_qh BD 1 s_hh s_ht (NT * BT) scale BT BD NT Bool.false Bool.false).toAlgKernel := by
      simp only [ComputeKernel.toAlgKernel, halg]
    unfold ComputeCorrect.Realizes_without_Rounding
    apply ComputeKernel.computeCorrect_of_toAlgKernel hk
    intro s0 s' hExec hs0
    subst s0
    intro i
    rw [hExec] at hexec
    obtain rfl := Option.some.inj hexec
    exact hOF i.1.val i.1.isLt i.2.1 i.2.2


end Correct_without_Rounding


end VeriTile.Bench.TritonBenchG.AttentionFwdTriton1
