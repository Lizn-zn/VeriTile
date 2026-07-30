import VeriTile.Triton

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

prepare kernel:
  prepare_qg_kg_full_surface_qg_closed_general             ← TOP THEOREM (prepare, qg)
  prepare_qg_kg_full_surface_kg_closed_general             ← TOP THEOREM (prepare, kg)

backward kernel:
  decay_cumsum_backward_closed_output_summary_general  ← TOP THEOREM (backward)
       (realizes genuine closed forms bwdDQInterClosed / bwdDKInterClosed / bwdDGClosed)

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

section Correct_without_Rounding

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
the dimension-parameterized forward closed form (superseding an earlier
`BT=2`-pinned variant); the cross-step `range(BT)` cumulative
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
compute-correctness (superseding an earlier `BT=2`-pinned variant). -/
theorem fwd_decay_cumsum_surface_closed_compute_correct_general
    (G GO : RegionName)
    (s_qk_h s_qk_t s_qk_d B H T : Nat) (scale : ℝ)
    (BT BK DK : Nat) (s : BlockState) (t_rel : Fin BT)
    (hne : G ≠ GO) (hBK : BK ≤ DK) :
    ComputeCorrect.Realizes_without_Rounding
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
`exp2(x) = Real.exp (x * Real.log 2)`. This is the honest pointwise specification
of the `q *= exp2(g) * scale` map (here `scale = 1`). -/
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
`QG` unchanged on inactive lanes). Dimension-parameterized `qg` closed form
(superseding an earlier `BT=2`-pinned variant). -/
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
of loop row `t_rel`. Dimension-parameterized `qg` compute-correctness
(superseding an earlier `BT=2`-pinned variant). -/
theorem prepare_qg_kg_surface_qg_closed_compute_correct_general
    (Q K G QG KG : RegionName)
    (s_qk_h DK BT BK : Nat) (scale : ℝ) (s : BlockState) (t_rel : Fin BT)
    (hQ_QG : Q ≠ QG) (hQ_KG : Q ≠ KG) (hK_QG : K ≠ QG) (hK_KG : K ≠ KG)
    (hG_QG : G ≠ QG) (hG_KG : G ≠ KG) (hQG_KG : QG ≠ KG) (hBK : BK ≤ DK) :
    ComputeCorrect.Realizes_without_Rounding
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
    ComputeCorrect.Realizes_without_Rounding
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
the same closed-form recipe as the forward/prepare general stacks,
but the backward
loop body is ~25 statements with a conditional `last_g` capture and three masked
stores per iteration, traversed over the reverse `range(BT-1,-1,-1)` rows (lowered to a
forward `forRangeDyn "__rev_t" 0 BT 1` with `t := BT-1 - __rev_t`). A single
`simp [exec, …, evalOp.eq_def, stepForRangeAux.*]` blast does *not* scale to this
body (it does not terminate within ~9 min even at 8M heartbeats), so the
mandated per-statement architecture is required:

1. `exec → stepStmts toAlgKernel.body`, with the surface body decomposed by
   `bwd_body_decomp_general` into the 15-stmt prologue + the `forRangeDyn` reverse loop.
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
5. Bridge to `ComputeCorrect.Realizes_without_Rounding` via `realizes_writeIf_iff` +
   `computeCorrect_of_toAlgKernel` (done; `decayBackwardSurfaceValue` deleted).

This plan is now fully realized dimension-generally: `bwd_prologue_eval_general`
runs the 15-stmt prologue, `bwd_decay_cumsum_step_general` advances the reverse-loop
invariant `bwdInvG` (one iteration, head + conditional `last_g` capture + three
masked stores), `bwd_loop_drive_general` assembles prologue + the full `range(BT)`
reverse loop, and the three `_general` readback theorems certify the closed forms
(the `dg` face uses the genuine reverse cumsum via `bwdCumPartialG`).

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
`forRangeAux` invariants, exactly as `FusedLayerNorm` chains its per-stmt
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

/-! ### Full reverse-loop assembly

We chain the 25-statement `bwdIterBodyG` per the validated per-statement template
(`stepStmts.cons_some` + explicit `set`-bound state threading), driving the whole
`range(BT)` reverse loop by the `bwdInvG` invariant (`bwd_decay_cumsum_step_general`),
then read back the three output regions (`DQInter`, `DKInter`, `DG`) against the
genuine closed forms. -/

/-- Per-lane masked load value from `region` at the iteration's row offset `R`.
The mask is the active-lane decision; inactive lanes read `0`. -/
private noncomputable def ldVal (s : BlockState) (region : RegionName) (R : Nat)
    (i : TileIndex [4]) : WithBot ℝ :=
  if decide (s.pids 0 * 4 + i.1.val < 8) then
    some (s.readMem region (s.pids 2 * 64 + s.pids 0 * 4 + i.1.val + R))
  else (0 : WithBot ℝ)

/-- The post-`t`-assign, post-`g_val`-load state after the first two body
statements of `bwdIterBodyG`, given the iteration's input pointer/register
readbacks. The `t` index is `1 - rt` and `g_val` holds the masked load of `G`
at the iteration's pointer row offset `R`. -/
private noncomputable def bwdIterHeadState
    (G : RegionName) (sin : BlockState) (rt R : Nat) : BlockState :=
  (sin.setReg "t" .nat [] (Tile.scalar (2 - 1 - rt * 1))).setReg
    "g_val" .real [4] ⟨fun i => ldVal sin G R i⟩

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
    ComputeCorrect.Realizes_without_Rounding
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
    ComputeCorrect.Realizes_without_Rounding
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
    ComputeCorrect.Realizes_without_Rounding
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
specification decay_cumsum_backward_closed_output_summary_general :
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := bwd_decay_global_cumsum_surface DQInner DQInter DKInner DKInter
        Q K G DG s_qk_h DK BT BK)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf (active s DK BK)
        (fun i => (DQInter, offset s s_qk_h DK t_rel.val BT BK i)))
      (expected := fun i : Fin BK =>
        bwdDQInterClosed s DQInner DQInter G s_qk_h DK BT BK t_rel i)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := bwd_decay_global_cumsum_surface DQInner DQInter DKInner DKInter
        Q K G DG s_qk_h DK BT BK)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf (active s DK BK)
        (fun i => (DKInter, offset s s_qk_h DK t_rel.val BT BK i)))
      (expected := fun i : Fin BK =>
        bwdDKInterClosed s DKInner DKInter G s_qk_h DK BT BK t_rel i)) ∧
    (ComputeCorrect.Realizes_without_Rounding
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

end Correct_without_Rounding

/-! ## The `⊨[R]` streaming headline for `fwd_decay_cumsum`
(wave-5 S3 per-step emit genre, 3-D grid)

Everything below is purely additive; the exact surfaces above (all three
kernels) are untouched. This is the first consumer of the three-pid
single-stream per-step emit skin `StreamEmitMasked3DKernelIO₁`: the store
sits **inside** the `range(BT)` loop — step `t` stores the `BK`-lane
`cum_decay` window at the `t`-th row — and the kernel's spec `f t j` is the
genre's *scan* shape, the prefix fold
`1.44269504 · Σ_{u ≤ t} g[u, j]` over the streamed `g` rows.

Structure of the `execR R` story: this kernel has **zero rounding events**.
The masked load and the in-loop masked store are at `.real`, and the only
`castFloat`s are the erased `.to(tl.float32)` / `.to(p_go.dtype.element_ty)`
casts, i.e. `.real → .real` — exact under every `R` by the model's defining
`round_real` (`Rcast_real_real` below). The loop body therefore collapses
verbatim onto the exact stepper (`stepForRangeAuxR_castFree`), and the whole
proven `fwdInv` invariant stack above is reused unchanged; the `⊨[R]` face
adds only the `TraceSafeR` walk, the per-cell memory frame, and the
stream-lane spec bridge. The skin's readback contract at the default
`outDType := .real` grid carries `R.round .real`, the identity by
`round_real` — the ∀-`R` face is the exact streaming contract via the
model's `.real` identity fields, not a `.triv` special case. -/

section IOFace

open scoped VeriTile.Triton.StreamEmitMasked3DKernelIO₁

set_option maxHeartbeats 4000000
set_option linter.unusedVariables false

/-! ### The lowered program: prologue + loop -/

/-- The lowered pre-loop prologue of `fwd_decay_cumsum_surface`: the three
program ids, the `p_g`/`p_go` row base pointers, the zeroed `cum_decay`
accumulator and the `BK`-lane bound mask. -/
private def fwdPrologue (G GO : RegionName) (s_qk_h BT BK DK : Nat) : List Stmt :=
  [Stmt.assign TileDType.nat [] "i_k" (Op.programId 0),
   Stmt.assign TileDType.nat [] "i_c" (Op.programId 1),
   Stmt.assign TileDType.nat [] "i_bh" (Op.programId 2),
   Stmt.assign TileDType.ptr [BK] "p_g"
     (Op.ptrAdd Broadcast.scalarL (Op.ptrBase G)
       (Op.add NumericDType.nat Broadcast.scalarL
         (Op.add NumericDType.nat Broadcast.nil
           (Op.add NumericDType.nat Broadcast.nil
             (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_bh")
               (Op.constNat s_qk_h))
             (Op.mul NumericDType.nat Broadcast.nil
               (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_c")
                 (Op.constNat BT))
               (Op.constNat DK)))
           (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k")
             (Op.constNat BK)))
         (Op.arange BK))),
   Stmt.assign TileDType.ptr [BK] "p_go"
     (Op.ptrAdd Broadcast.scalarL (Op.ptrBase GO)
       (Op.add NumericDType.nat Broadcast.scalarL
         (Op.add NumericDType.nat Broadcast.nil
           (Op.add NumericDType.nat Broadcast.nil
             (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_bh")
               (Op.constNat s_qk_h))
             (Op.mul NumericDType.nat Broadcast.nil
               (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_c")
                 (Op.constNat BT))
               (Op.constNat DK)))
           (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k")
             (Op.constNat BK)))
         (Op.arange BK))),
   Stmt.assign TileDType.real [BK] "cum_decay" (Op.full [BK] (Op.const 0)),
   Stmt.assign TileDType.bool [BK] "mask"
     (Op.lt ComparableDType.nat Broadcast.scalarR
       (Op.add NumericDType.nat Broadcast.scalarL
         (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k")
           (Op.constNat BK))
         (Op.arange BK))
       (Op.constNat DK))]

/-- The `toAlgKernel` body decomposes into the prologue followed by the
single `range(BT)` storing loop (whose lowered body is the proven
`fwdBody`). -/
private theorem fwd_body_decomp (G GO : RegionName)
    (s_qk_h s_qk_t s_qk_d B H T : Nat) (scale : ℝ) (BT BK DK : Nat) :
    (fwd_decay_cumsum_surface G GO s_qk_h s_qk_t s_qk_d B H T scale BT BK DK).toAlgKernel.body
      = fwdPrologue G GO s_qk_h BT BK DK
        ++ [Stmt.forRange "_i" 0 BT 1 (fwdBody G GO s_qk_h BT BK DK)] := by
  rfl

/-! ### IO signature -/

/-- **Streaming IO signature** of `fwd_decay_cumsum` on the three-pid
single-stream per-step emit skin (S3: in-loop store). Step `t` of the
`range(BT)` loop reads the `BK`-lane `g` row (`read1`) and stores the
`BK`-lane `cum_decay` window (`write`) at the **`.real`** grid (`outDType`
default — the store's `.to(p_go.dtype.element_ty)` cast erases to `.real`,
so the per-step stores have no quantization event). The windows transcribe
the surface's effective addresses verbatim (the surface models the `+= DK`
pointer advance, so row `t`'s address is the base plus `t·DK`):

* `read1` step `t`, lane `j`:
  `i_bh·s_qk_h + i_c·BT·DK + i_k·BK + j + t·DK` — the kernel's `p_g` lane
  after `t` advances (`pid₀ = i_k`, `pid₁ = i_c`, `pid₂ = i_bh`).
* `write` step `t`, lane `j`: the same window into `g_o` — the kernel's
  `p_go` lane.

Both masks are the kernel's single, `t`-independent bound mask
`i_k·BK + j < DK`. -/
def fwdDecayCumsumKernelIO (G GO : RegionName)
    (s_qk_h s_qk_t s_qk_d B H T : Nat) (scale : ℝ) (BT BK DK : Nat) :
    StreamEmitMasked3DKernelIO₁ where
  kernel := fwd_decay_cumsum_surface G GO s_qk_h s_qk_t s_qk_d B H T scale BT BK DK
  inp1 := G
  out := GO
  T := BT
  B1 := BK
  C := BK
  read1 := fun p₀ p₁ p₂ t j => p₂ * s_qk_h + p₁ * BT * DK + p₀ * BK + j.val + t.val * DK
  write := fun p₀ p₁ p₂ t j => p₂ * s_qk_h + p₁ * BT * DK + p₀ * BK + j.val + t.val * DK
  mask1 := fun p₀ _ _ _ j => p₀ * BK + j.val < DK
  writeMask := fun p₀ _ _ _ j => p₀ * BK + j.val < DK

/-! ### The stream-level spec -/

/-- The stream-level decay-cumsum spec (the genre's *scan* shape): output
window `(t, j)` holds the `inv_ln2`-scaled prefix sum of lane `j`'s streamed
`g` values through step `t`. Since the kernel's mask is `t`-independent, the
sum only ever touches lane `j` itself — on a write-active lane every
summand is mask-pinned, so no guard is needed inside the sum.
Algebraically `fwdDecayClosed` with the row reads re-indexed to the curried
stream. -/
noncomputable def fwdDecayStreamSpec (BT BK : Nat)
    (gs : Fin BT → Fin BK → ℝ) (t : Fin BT) (j : Fin BK) : ℝ :=
  1.44269504 * ∑ u : Fin (t.val + 1), gs (Fin.castLE t.isLt u) j

/-- The skin's window addresses in `offset` form: the io windows above are
the proven stack's `offset s s_qk_h DK t BT BK j` (pure `ring` on ℕ). -/
private theorem fwd_addr_eq (s : BlockState) (s_qk_h BT BK DK m : Nat) (i : Fin BK) :
    s.pids 2 * s_qk_h + s.pids 1 * BT * DK + s.pids 0 * BK + i.val + m * DK
      = offset s s_qk_h DK m BT BK i := by
  simp only [offset, baseOffset]
  ring

/-- Per-lane spec bridge: at a masked window `(t, j)` the stream spec **is**
the exact stack's closed form `fwdDecayClosed` — the sum only involves lane
`j`, which is mask-pinned at every row because the mask is
`t`-independent. -/
private theorem fwdDecayStreamSpec_eq_closed (G : RegionName) (s₀ : BlockState)
    (s_qk_h BT BK DK : Nat) (xs : Fin BT → Fin BK → ℝ)
    (hx : ∀ (t : Fin BT) (j : Fin BK), s₀.pids 0 * BK + j.val < DK →
      s₀.readMem G
          (s₀.pids 2 * s_qk_h + s₀.pids 1 * BT * DK + s₀.pids 0 * BK + j.val + t.val * DK)
        = xs t j)
    (t : Fin BT) (j : Fin BK) (hj : active s₀ DK BK j) :
    fwdDecayStreamSpec BT BK xs t j = fwdDecayClosed s₀ G s_qk_h DK BT BK t j := by
  unfold fwdDecayStreamSpec fwdDecayClosed
  congr 1
  refine Finset.sum_congr rfl fun u _ => ?_
  rw [← hx (Fin.castLE t.isLt u) j hj]
  congr 1
  simp only [offset, baseOffset, Fin.val_castLE]
  ring

/-! ### Cast-free collapses and the covered fragment -/

/-- The erased `.to(tl.float32)` / `.to(p_go.dtype.element_ty)` are
`.real → .real` casts: exact under every `R` — `R.roundW .real` is the
identity by the model's defining `round_real`. -/
private theorem Rcast_real_real (R : RoundingModel) :
    R.cast .real .real = FloatDType.cast .real .real := by
  funext v
  simp [RoundingModel.cast, FloatDType.cast]

/-- The prologue is cast-free: it steps identically under `stepStmtsR R`. -/
private theorem fwdPrologue_castFree (R : RoundingModel) (G GO : RegionName)
    (s_qk_h BT BK DK : Nat) (t : BlockState) :
    stepStmtsR R (fwdPrologue G GO s_qk_h BT BK DK) t
      = stepStmts (fwdPrologue G GO s_qk_h BT BK DK) t := by
  simp only [fwdPrologue, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def]
  rfl

/-- The loop body is cast-free **including its in-loop masked `.real`
store**: the masked `.real` load with erased fp32 cast is exact under `R`,
and `stepStmtR` delegates a `.real`-typed store to the exact
`writeMemTyped`, so the whole storing loop steps identically under
`stepStmtsR R` and the exact `fwdInv` stack transports to `execR`. -/
private theorem fwdBody_castFree (R : RoundingModel) (G GO : RegionName)
    (s_qk_h BT BK DK : Nat) (t : BlockState) :
    stepStmtsR R (fwdBody G GO s_qk_h BT BK DK) t
      = stepStmts (fwdBody G GO s_qk_h BT BK DK) t := by
  simp only [fwdBody, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def, Rcast_real_real, BlockState.writeMemTypedR]
  rfl

/-- The full forward surface sits inside the flat-memory bridge's covered
fragment (`FlattenOk`; the `forRange` clause recurses into the cast-free
body). -/
private theorem fwd_flattenOk (G GO : RegionName)
    (s_qk_h s_qk_t s_qk_d B H T : Nat) (scale : ℝ) (BT BK DK : Nat) :
    ((fwd_decay_cumsum_surface G GO s_qk_h s_qk_t s_qk_d B H T scale BT BK DK).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  rw [fwd_body_decomp]
  simp [fwdPrologue, fwdBody, StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-! ### The prologue walk (private copy of the exact headline's inline
prefix walk, packaged as a reusable lemma with the memory frame) -/

/-- The prologue establishes `fwdInv … 0` and never touches memory. -/
private theorem fwd_preLoop (G GO : RegionName) (s_qk_h BT BK DK : Nat)
    (s : BlockState) :
    ∃ s6, stepStmts (fwdPrologue G GO s_qk_h BT BK DK) s = some s6
      ∧ fwdInv G GO s s_qk_h DK BT BK 0 s6 ∧ s6.mem = s.mem := by
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
  have hcumeval : evalOp (Op.full [BK] (Op.const 0)) s_pgo
      = some (fwdCumTile s G s_qk_h DK BT BK 0) := by
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
  -- chain the 7 prologue assigns
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
  unfold fwdPrologue
  rw [stepStmts.cons_some hstepK, stepStmts.cons_some hstepC, stepStmts.cons_some hstepB,
    stepStmts.cons_some hstepPg, stepStmts.cons_some hstepPgo, stepStmts.cons_some hstepCum,
    stepStmts.cons_some hstepMask, stepStmts.nil]
  refine ⟨_, rfl, ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩, ?_⟩
  · -- pids
    rw [hs_cum, hs_pgo, hs_pg, hs_ibh, hs_ic, hs_ik]
    simp only [BlockState.setReg_pids]
  · -- G untouched
    intro a
    rw [hs_cum, hs_pgo, hs_pg, hs_ibh, hs_ic, hs_ik]
    simp only [BlockState.setReg_readMem]
  · -- cum_decay
    rw [BlockState.setReg_ne_name (h := by decide), hs_cum, BlockState.setReg_same]
  · -- p_g
    rw [BlockState.setReg_ne_name (h := by decide), hs_cum,
      BlockState.setReg_ne_name (h := by decide), hs_pgo,
      BlockState.setReg_ne_name (h := by decide), hs_pg, BlockState.setReg_same]
  · -- p_go
    rw [BlockState.setReg_ne_name (h := by decide), hs_cum,
      BlockState.setReg_ne_name (h := by decide), hs_pgo, BlockState.setReg_same]
  · -- mask
    rw [BlockState.setReg_same]
  · -- untouched GO at rows ≥ 0
    intro j hj i
    rw [hs_cum, hs_pgo, hs_pg, hs_ibh, hs_ic, hs_ik]
    simp only [BlockState.setReg_readMem]
  · -- readback GO at rows < 0: vacuous
    intro j hj i
    omega
  · -- memory frame
    rw [hs_cum, hs_pgo, hs_pg, hs_ibh, hs_ic, hs_ik]
    rfl

/-! ### Cell-level memory frame of one loop iteration -/

/-- Exact-side inversion of a leading assign in a `stepStmts` chain (the
exact mirror of `stepStmtR_assign_inv`): threads successor states of
register-only statements without computing their values. -/
private theorem stepStmts_cons_assign_inv {dt : TileDType} {sh : TileShape}
    {nm : RegName} {e : Op dt sh} {rest : List Stmt} {s s' : BlockState}
    (h : stepStmts (Stmt.assign dt sh nm e :: rest) s = some s') :
    ∃ v, evalOp e s = some v ∧ stepStmts rest (s.setReg nm dt sh v) = some s' := by
  cases hv : evalOp e s with
  | none => simp [stepStmts, stepStmt, hv] at h
  | some v =>
      rw [stepStmts.cons_some (stepStmt_assign_eq_some hv)] at h
      exact ⟨v, rfl, h⟩

/-- Cell-level frame of a `Prop`-masked exact `writeMem` scatter `foldl`:
every cell not hit by an active lane is untouched (the cell-level sibling of
`BlockState.scatter_prop_masked_preserves_other_offset`, phrased on raw
`mem`). -/
private theorem foldl_writeMem_prop_preserve_cell {α : Type}
    {region : RegionName} (ofn : α → Nat) (vfn : α → ℝ)
    (P : α → Prop) [DecidablePred P]
    (r : RegionName) (oo : Nat) (l : List α) (s : BlockState)
    (hnot : ∀ k ∈ l, P k → ¬(r = region ∧ oo = ofn k)) :
    (l.foldl (fun acc k =>
        if P k then acc.writeMem region (ofn k) (vfn k) else acc) s).mem r oo
      = s.mem r oo := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons]
      by_cases hm : P hd
      · rw [if_pos hm,
          ih _ (fun k hk hmk => hnot k (List.mem_cons_of_mem hd hk) hmk),
          BlockState.writeMem_mem]
        exact if_neg (hnot hd List.mem_cons_self hm)
      · rw [if_neg hm]
        exact ih _ fun k hk hmk => hnot k (List.mem_cons_of_mem hd hk) hmk

/-- **Cell-level frame of one loop iteration** (the `mem` twin of
`fwd_decay_cumsum_step`, same walk with opaque register values): from the
`fwdInv` `p_go`/`mask` pins, one storing body iteration leaves every cell
off the row-`m` write window `{(GO, offset … m … i) : active i}`
untouched. -/
private theorem fwd_body_step_frame (G GO : RegionName) (s_qk_h BT BK DK : Nat)
    (s sc s' : BlockState) (m : Nat)
    (hpgo : sc.regs .ptr [BK] "p_go" = some (fwdPgoTile s GO s_qk_h DK BT BK m))
    (hmask : sc.regs .bool [BK] "mask" = some (fwdMaskTile s DK BK))
    (hstep : stepStmts (fwdBody G GO s_qk_h BT BK DK)
      (sc.setReg "_i" .nat [] (Tile.scalar m)) = some s')
    (r : RegionName) (oo : Nat)
    (hcond : r ≠ GO ∨ ∀ i : Fin BK, active s DK BK i →
      oo ≠ offset s s_qk_h DK m BT BK i) :
    s'.mem r oo = sc.mem r oo := by
  set sloop := sc.setReg "_i" .nat [] (Tile.scalar m) with hsloop
  unfold fwdBody at hstep
  obtain ⟨v1, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  obtain ⟨v2, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  set s2 := (sloop.setReg "_g" .real [BK] v1).setReg "cum_decay" .real [BK] v2 with hs2
  have hcum2 : s2.regs .real [BK] "cum_decay" = some v2 := by
    rw [hs2]; exact BlockState.setReg_same _ _ _ _ _
  have hpgo2 : s2.regs .ptr [BK] "p_go" = some (fwdPgoTile s GO s_qk_h DK BT BK m) := by
    rw [hs2, BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide), hsloop,
      BlockState.setReg_ne_name (h := by decide)]
    exact hpgo
  have hmask2 : s2.regs .bool [BK] "mask" = some (fwdMaskTile s DK BK) := by
    rw [hs2, BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide), hsloop,
      BlockState.setReg_ne_name (h := by decide)]
    exact hmask
  set vfun : TileIndex [BK] → ℝ := fun k => (v2.data k).unbotD 0 with hvfun
  set sst := (TileShape.allIndices [BK]).foldl
    (fun (acc : BlockState) k =>
      if active s DK BK k.1 then
        acc.writeMem GO (offset s s_qk_h DK m BT BK k.1) (vfun k)
      else acc) s2 with hsst
  have hstore : stepStmt (Stmt.store TileDType.real [BK]
      (MemAccess.ptr (Op.ref TileDType.ptr [BK] "p_go"))
      (Op.ref TileDType.real [BK] "cum_decay")
      (MaskOpt.mask (Op.ref TileDType.bool [BK] "mask"))) s2 = some sst := by
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
  rw [stepStmts.cons_some hstore] at hstep
  obtain ⟨v4, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  obtain ⟨v5, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  rw [stepStmts.nil] at hstep
  obtain rfl := Option.some.inj hstep
  simp only [BlockState.setReg_mem]
  rw [hsst]
  refine Eq.trans (foldl_writeMem_prop_preserve_cell _ _ _ r oo _ _ ?_) ?_
  · intro k hk hmk hbad
    rcases hcond with hne' | hno
    · exact hne' hbad.1
    · exact hno k.1 hmk hbad.2
  · rw [hs2, hsloop]
    simp only [BlockState.setReg_mem]

/-! ### The `TraceSafeR` walk -/

/-- Per-iteration `TraceSafeListR` for the loop body: the accumulate and the
two pointer advances are register-only; the masked `g` load's and the
masked store's **active** lanes are exactly the skin's (`t`-independent)
`mask1`/`writeMask` window, in bounds by the corresponding window bound
(instantiated at row `m`). -/
private theorem fwd_bodySafeR (R : RoundingModel) (bounds : RegionBounds)
    (G GO : RegionName) (s_qk_h BT BK DK : Nat)
    (s stt : BlockState) (m : Nat)
    (hpg : stt.regs .ptr [BK] "p_g" = some (fwdPgTile s G s_qk_h DK BT BK m))
    (hpgo : stt.regs .ptr [BK] "p_go" = some (fwdPgoTile s GO s_qk_h DK BT BK m))
    (hmask : stt.regs .bool [BK] "mask" = some (fwdMaskTile s DK BK))
    (hbG : ∀ i : Fin BK, active s DK BK i → offset s s_qk_h DK m BT BK i < bounds G)
    (hbGO : ∀ i : Fin BK, active s DK BK i → offset s s_qk_h DK m BT BK i < bounds GO) :
    Stmt.TraceSafeListR R bounds (fwdBody G GO s_qk_h BT BK DK)
      (stt.setReg "_i" .nat [] (Tile.scalar m)) := by
  set sloop := stt.setReg "_i" .nat [] (Tile.scalar m) with hsloop
  have hpg' : sloop.regs .ptr [BK] "p_g" = some (fwdPgTile s G s_qk_h DK BT BK m) := by
    rw [hsloop]; simpa using hpg
  have hpgo' : sloop.regs .ptr [BK] "p_go" = some (fwdPgoTile s GO s_qk_h DK BT BK m) := by
    rw [hsloop]; simpa using hpgo
  have hmask' : sloop.regs .bool [BK] "mask" = some (fwdMaskTile s DK BK) := by
    rw [hsloop]; simpa using hmask
  unfold fwdBody
  -- (1) the masked `_g` load
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun t1 ht1 => ?_)
  · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
    refine ⟨by simp [Op.SafeAtR.eq_def], by simp [Op.SafeAtR.eq_def], ?_⟩
    intro ptrs hptrs idx hactive
    rw [evalOpR_ref, hpg'] at hptrs
    obtain rfl := Option.some.inj hptrs
    obtain ⟨masks, hm, hmi⟩ := hactive
    rw [evalOpR_ref, hmask'] at hm
    obtain rfl := Option.some.inj hm
    have ha : active s DK BK idx.1 := by simpa [fwdMaskTile] using hmi
    simpa [fwdPgTile] using hbG idx.1 ha
  · obtain ⟨v1, -, rfl⟩ := stepStmtR_assign_inv ht1
    -- (2) cum_decay accumulate: register-only
    refine Stmt.TraceSafeListR.cons_intro
      (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun t2 ht2 => ?_)
    obtain ⟨v2, -, rfl⟩ := stepStmtR_assign_inv ht2
    have hpgo3 : ((sloop.setReg "_g" .real [BK] v1).setReg
        "cum_decay" .real [BK] v2).regs .ptr [BK] "p_go"
        = some (fwdPgoTile s GO s_qk_h DK BT BK m) := by
      rw [BlockState.setReg_ne_name (h := by decide),
        BlockState.setReg_ne_name (h := by decide)]
      exact hpgo'
    have hmask3 : ((sloop.setReg "_g" .real [BK] v1).setReg
        "cum_decay" .real [BK] v2).regs .bool [BK] "mask"
        = some (fwdMaskTile s DK BK) := by
      rw [BlockState.setReg_ne_name (h := by decide),
        BlockState.setReg_ne_name (h := by decide)]
      exact hmask'
    -- (3) the masked store
    refine Stmt.TraceSafeListR.cons_intro ?_ (fun t3 ht3 => ?_)
    · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MemAccess.SafeAtR,
        MaskOpt.SafeAtR, MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR,
        memAccessActiveAddressSafeR]
      refine ⟨by simp [Op.SafeAtR.eq_def], by simp [Op.SafeAtR.eq_def],
        by simp [Op.SafeAtR.eq_def], ?_⟩
      intro ptrs hptrs idx hactive
      rw [evalOpR_ref, hpgo3] at hptrs
      obtain rfl := Option.some.inj hptrs
      obtain ⟨masks, hm, hmi⟩ := hactive
      rw [evalOpR_ref, hmask3] at hm
      obtain rfl := Option.some.inj hm
      have ha : active s DK BK idx.1 := by simpa [fwdMaskTile] using hmi
      simpa [fwdPgoTile] using hbGO idx.1 ha
    · -- (4)(5) pointer advances: register-only
      refine Stmt.TraceSafeListR.cons_intro
        (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun t4 ht4 => ?_)
      exact Stmt.TraceSafeListR.cons_intro
        (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
        (fun _ _ => Stmt.TraceSafeListR.nil_intro)

/-- **The `TraceSafeR` walk for the whole forward kernel** — driven by
`Stmt.forRangeTraceSafeR_inv` over the proven `fwdInv`, with the counter
advancing by the loop's stride `1`. The two bound groups are the skin's
`read1`/`write` windows (identical addresses into `G`/`GO`); `hne`/`hBK`
feed the reused `fwd_decay_cumsum_step` (see the headline's provenance
notes). -/
private theorem fwd_traceSafeR (R : RoundingModel) (bounds : RegionBounds)
    (G GO : RegionName) (s_qk_h s_qk_t s_qk_d B H T : Nat) (scale : ℝ)
    (BT BK DK : Nat) (hne : G ≠ GO) (hBK : BK ≤ DK) (s : BlockState)
    (hbG : ∀ (t : Fin BT) (j : Fin BK), s.pids 0 * BK + j.val < DK →
      s.pids 2 * s_qk_h + s.pids 1 * BT * DK + s.pids 0 * BK + j.val + t.val * DK
        < bounds G)
    (hbGO : ∀ (t : Fin BT) (j : Fin BK), s.pids 0 * BK + j.val < DK →
      s.pids 2 * s_qk_h + s.pids 1 * BT * DK + s.pids 0 * BK + j.val + t.val * DK
        < bounds GO) :
    ((fwd_decay_cumsum_surface G GO s_qk_h s_qk_t s_qk_d B H T scale BT BK DK).toAlgKernel).TraceSafeR R bounds s := by
  -- window instantiators at `offset`-form addresses
  have hbG' : ∀ m : Nat, m < BT → ∀ i : Fin BK, active s DK BK i →
      offset s s_qk_h DK m BT BK i < bounds G := by
    intro m hm i ha
    have h : s.pids 2 * s_qk_h + s.pids 1 * BT * DK + s.pids 0 * BK + i.val + m * DK
        < bounds G := hbG ⟨m, hm⟩ i ha
    rwa [fwd_addr_eq] at h
  have hbGO' : ∀ m : Nat, m < BT → ∀ i : Fin BK, active s DK BK i →
      offset s s_qk_h DK m BT BK i < bounds GO := by
    intro m hm i ha
    have h : s.pids 2 * s_qk_h + s.pids 1 * BT * DK + s.pids 0 * BK + i.val + m * DK
        < bounds GO := hbGO ⟨m, hm⟩ i ha
    rwa [fwd_addr_eq] at h
  unfold Kernel.TraceSafeR
  rw [fwd_body_decomp]
  refine Stmt.TraceSafeListR.append_intro _ _ ?_ ?_
  · -- prologue: register-only assigns, safe at every state
    refine Stmt.TraceSafeListR.of_forall _ _ ?_
    intro stmt hst s'
    simp only [fwdPrologue, List.mem_cons, List.not_mem_nil, or_false] at hst
    rcases hst with rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
      simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]
  · intro s1 hs1
    obtain ⟨s6, hsp, hP0, hmem6⟩ := fwd_preLoop G GO s_qk_h BT BK DK s
    rw [fwdPrologue_castFree R G GO s_qk_h BT BK DK s, hsp] at hs1
    obtain rfl := Option.some.inj hs1
    refine Stmt.TraceSafeListR.cons_intro ?_ (fun _ _ => Stmt.TraceSafeListR.nil_intro)
    simp only [Stmt.TraceSafeR]
    refine Stmt.forRangeTraceSafeR_inv R bounds "_i" BT 1
      (fwdBody G GO s_qk_h BT BK DK) (fwdInv G GO s s_qk_h DK BT BK) ?_ 0 s6 hP0
    intro c stt hc hP
    have hPd := hP
    obtain ⟨hpids, hG, hcum, hpg, hpgo, hmask, hUnt, hRead⟩ := hPd
    refine ⟨fwd_bodySafeR R bounds G GO s_qk_h BT BK DK s stt c hpg hpgo hmask
      (hbG' c hc) (hbGO' c hc), ?_⟩
    obtain ⟨st', hstep, hP'⟩ :=
      fwd_decay_cumsum_step G GO s s_qk_h BT BK DK hne hBK c stt hP
    exact ⟨st', by rw [fwdBody_castFree]; exact hstep, hP'⟩

/-! ### The rounded Hoare triple (`hrun`) -/

/-- Termination, per-window values and the per-cell frame of the whole
forward kernel under `execR R`, from an **arbitrary** launch state: the
exact `fwd_preLoop` / `fwdInv` stack runs verbatim (the prologue and the
loop body are cast-free, so `execR R` collapses onto the exact stepper),
extended with the per-cell memory frame carried alongside `fwdInv`. -/
private theorem fwd_runR (R : RoundingModel) (G GO : RegionName)
    (s_qk_h s_qk_t s_qk_d B H T : Nat) (scale : ℝ) (BT BK DK : Nat)
    (hne : G ≠ GO) (hBK : BK ≤ DK) (s₀ : BlockState) :
    ∃ sfin,
      execR R (fwd_decay_cumsum_surface G GO s_qk_h s_qk_t s_qk_d B H T scale BT BK DK).toAlgKernel s₀
        = some sfin
      ∧ (∀ (t : Fin BT) (i : Fin BK), active s₀ DK BK i →
          sfin.readMem GO (offset s₀ s_qk_h DK t.val BT BK i)
            = fwdDecayClosed s₀ G s_qk_h DK BT BK t i)
      ∧ (∀ r oo, (r ≠ GO ∨ ∀ (t : Fin BT) (i : Fin BK), active s₀ DK BK i →
            oo ≠ offset s₀ s_qk_h DK t.val BT BK i) →
          sfin.mem r oo = s₀.mem r oo) := by
  obtain ⟨s6, hsp, hP0, hmem6⟩ := fwd_preLoop G GO s_qk_h BT BK DK s₀
  -- the loop with the per-cell frame carried alongside `fwdInv`
  obtain ⟨final, sfinal, hLoop, hfinal, hPfinal⟩ :=
    forRange_inv (idx := "_i") (start := 0) (stop := BT) (step := 1)
      (body := fwdBody G GO s_qk_h BT BK DK)
      (P := fun m stt => fwdInv G GO s₀ s_qk_h DK BT BK m stt
        ∧ ∀ r oo, (r ≠ GO ∨ ∀ (t : Fin BT) (i : Fin BK), active s₀ DK BK i →
            oo ≠ offset s₀ s_qk_h DK t.val BT BK i) →
          stt.mem r oo = s₀.mem r oo)
      Nat.one_ne_zero
      ⟨hP0, fun r oo _ => by rw [hmem6]⟩
      (fun m stt hm hP => by
        obtain ⟨st', hstep, hP'⟩ :=
          fwd_decay_cumsum_step G GO s₀ s_qk_h BT BK DK hne hBK m stt hP.1
        refine ⟨st', hstep, hP', ?_⟩
        intro r oo hcond
        obtain ⟨hpids, hG, hcum, hpg, hpgo, hmask, hUnt, hRead⟩ := hP.1
        rw [fwd_body_step_frame G GO s_qk_h BT BK DK s₀ stt st' m hpgo hmask hstep r oo ?_]
        · exact hP.2 r oo hcond
        · rcases hcond with hne' | hno
          · exact Or.inl hne'
          · exact Or.inr fun i ha => hno ⟨m, hm⟩ i ha)
  obtain ⟨⟨_, _, _, _, _, _, _, hreadGO⟩, hframe⟩ := hPfinal
  -- assemble the `execR` run through the cast-free collapses
  have hLoopR : stepStmtR R (Stmt.forRange "_i" 0 BT 1 (fwdBody G GO s_qk_h BT BK DK)) s6
      = some sfinal := by
    rw [stepStmtR_forRange,
      stepForRangeAuxR_castFree R _ (fwdBody_castFree R G GO s_qk_h BT BK DK) "_i",
      ← stepForRangeAux.forRange_unfold]
    exact hLoop
  refine ⟨sfinal, ?_, ?_, ?_⟩
  · unfold execR
    rw [fwd_body_decomp,
      stepStmtsR_append R (fwdPrologue G GO s_qk_h BT BK DK) _ s₀,
      fwdPrologue_castFree R G GO s_qk_h BT BK DK s₀, hsp, Option.bind_some,
      stepStmtsR_cons_some hLoopR, stepStmtsR_nil]
  · intro t i ha
    have ht : t.val < final := lt_of_lt_of_le t.isLt hfinal
    rw [hreadGO t.val ht i, if_pos ha]
    simp only [fwdDecayClosed, fwdPartial]
  · exact hframe

/-! ### The headline -/

/-- **The `⊨[R]` streaming headline (wave-5 S3 per-step emit genre, 3-D
grid).** For every rounding model `R`, the faithful `fwd_decay_cumsum`
surface implements, on its `StreamEmitMasked3DKernelIO₁` signature, the
**ideal ℝ decay cumulative sum** over the streamed `g` rows: emitted window
`(t, j)` holds the scan `1.44269504 · Σ_{u ≤ t} g[u, j]` — the spec `f` is
exact real arithmetic. The kernel has **zero rounding events** (the masked
load, the accumulate and the per-step masked stores are all at `.real`; the
erased `.to(tl.float32)` / `.to(p_go.dtype.element_ty)` casts are
`.real → .real`), so the skin's boundary quantization degenerates: the
readback contract's `R.round .real` is the identity by the model's defining
`round_real` — the ∀-`R` face holds via the `RoundingModel` `.real`
identity fields, not as a `.triv` special case.

Layer map: the loop body is cast-free, so under `execR R` it collapses
verbatim onto the exact stepper and the proven `fwdInv` invariant stack
above (`fwd_decay_cumsum_step` / `forRange_inv`) is reused unchanged; the
`⊨[R]` face adds the `TraceSafeR` walk, the per-cell memory frame
(`fwd_body_step_frame`, the `mem` twin of the step lemma), and the
stream-lane spec bridge (`fwdDecayStreamSpec_eq_closed` — the mask is
`t`-independent, so on a write-active lane every summand read is
mask-pinned and the guarded form needs no in-sum guard).

Both hypotheses are inherited from the exact headline
`fwd_decay_cumsum_full_surface_closed_general`'s side conditions:

* `hne : G ≠ GO` — the loop stores into `g_o` **between** its per-row
  re-reads of `g`; the invariant's whole-`G` frame (and hence the closed
  form over the *initial* `g` values) requires the output buffer not to
  alias the input. The launch allocates `g` and `g_o` as distinct tensors.
* `hBK : BK ≤ DK` — row separation: the row-`m` scatter must not collide
  with other rows' windows, which needs every lane index `< BK` to stay
  inside one `DK`-wide row. The launch sets `BK = min(DK, 64)`, so this
  holds for every real launch.

Relation to the exact surface: the exact headline
`fwd_decay_cumsum_full_surface_closed_general` above is retained unchanged;
this `⊨[R]` face restates the same scan content on the streaming emit skin,
for every `R` at once (at the `.real` grid the two faces carry the same
exact cell). Both faces are kept per the rounding-as-default doctrine. -/
specification fwd_decay_cumsum_io_correctness (R : RoundingModel)
    (G GO : RegionName) (s_qk_h s_qk_t s_qk_d B H T : Nat) (scale : ℝ)
    (BT BK DK : Nat) (hne : G ≠ GO) (hBK : BK ≤ DK) :
    fwdDecayCumsumKernelIO G GO s_qk_h s_qk_t s_qk_d B H T scale BT BK DK ⊨[R]
      fun _ _ _ gs t j => fwdDecayStreamSpec BT BK gs t j := by
  refine StreamEmitMasked3DKernelIO₁.ImplementsR.intro _ ?_ ?_ ?_
  · exact fwd_flattenOk G GO s_qk_h s_qk_t s_qk_d B H T scale BT BK DK
  · -- safety walk
    intro bounds s xs _hx hbr1 hbw
    simp only [fwdDecayCumsumKernelIO] at hbr1 hbw ⊢
    exact fwd_traceSafeR R bounds G GO s_qk_h s_qk_t s_qk_d B H T scale BT BK DK
      hne hBK s hbr1 hbw
  · -- the rounded Hoare triple
    intro s₀ xs _hundef hx
    simp only [fwdDecayCumsumKernelIO] at hx ⊢
    obtain ⟨sfin, hexec, hval, hframe⟩ :=
      fwd_runR R G GO s_qk_h s_qk_t s_qk_d B H T scale BT BK DK hne hBK s₀
    refine ⟨sfin, hexec, ?_, ?_⟩
    · intro t j hj
      have hval' : sfin.readMem GO (offset s₀ s_qk_h DK t.val BT BK j)
          = fwdDecayClosed s₀ G s_qk_h DK BT BK t j := hval t j hj
      rw [BlockState.readMemAs_real, fwd_addr_eq, hval',
        ← fwdDecayStreamSpec_eq_closed G s₀ s_qk_h BT BK DK xs hx t j hj]
      simp [FloatDType.ofReal]
    · intro r oo hcond
      refine hframe r oo ?_
      rcases hcond with hne' | hno
      · exact Or.inl hne'
      · exact Or.inr fun t i ha hoeq =>
          hno t i ha (hoeq.trans (fwd_addr_eq s₀ s_qk_h BT BK DK t.val i).symm)

end IOFace

/-! ## The `⊨[R]` streaming headline for `prepare_qg_kg`
(wave-5 S3 grouped per-step emit genre, 3-D grid)

Everything below is purely additive; the exact surfaces above (all three
kernels) and the `fwd_decay_cumsum` io section are untouched. This is a
consumer of the grouped multi-channel per-step emit skin
`StreamGroupedEmitMasked3DKernelIO`: three input streams (`q`/`k`/`g` as
channels `0`/`1`/`2` over the shared row addressing), two per-step emit
windows (`qg`/`kg` as output channels `0`/`1`), both stores **inside** the
`range(BT)` loop, and the spec `f o t j` is the genre's *emit* shape — a
per-step map with no cross-step carry (`qg = q·exp2(g)·scale`,
`kg = k·exp2(last_decay − g)`).

The pre-loop `last_decay` tile needs **no extra channel**: its (unmasked!)
load reads exactly the `g` stream's step-`BT−1` row cells, so the spec
reads `xs 2 ⟨BT−1, _⟩ j` directly, and the `g` channel's `readMask`
carries an extra `t.val = BT − 1` disjunct — the kernel genuinely reads
**all** `BK` lanes of the last `g` row (the Python `tl.load` on line 65
has no `mask=`), so the honest read window is "active lanes at every row,
plus every lane at row `BT−1`", and that widened window is what both
bounds the unmasked load in the `TraceSafeR` walk and pins the
`last_decay` inputs for the spec.

Rounding structure: like the forward kernel this surface has **zero
rounding events** — the loads, the `exp2` scalings and the two per-step
masked stores are all at `.real` (the `.to(tl.float32)` /
`.to(*.dtype.element_ty)` casts erase to `.real → .real`), so the body
collapses verbatim onto the exact stepper (`Rcast_real_real`) and the
whole proven `prepInv` invariant stack above (`prepare_qg_kg_step` /
`forRange_inv`) is reused unchanged; the `⊨[R]` face adds only the
`TraceSafeR` walk (with the two-store body threading register pins across
the first store), the per-cell memory frame, and the per-channel
stream-lane spec bridges. -/

section PrepareIOFace

open scoped VeriTile.Triton.StreamGroupedEmitMasked3DKernelIO

set_option maxHeartbeats 4000000
set_option linter.unusedVariables false

/-! ### The lowered program: 10-statement prefix + `last_decay` load + loop -/

/-- The first ten lowered pre-loop statements of `prepare_qg_kg_surface`:
the three program ids, the `offs` lane index, the five row base pointers
and the `BK`-lane bound mask (everything before the `last_decay` load). -/
private def prepPre10 (Q K G QG KG : RegionName)
    (s_qk_h BT BK DK : Nat) : List Stmt :=
  [Stmt.assign TileDType.nat [] "i_k" (Op.programId 0),
   Stmt.assign TileDType.nat [] "i_c" (Op.programId 1),
   Stmt.assign TileDType.nat [] "i_bh" (Op.programId 2),
   Stmt.assign TileDType.nat [BK] "offs" (Op.arange BK),
   Stmt.assign TileDType.ptr [BK] "p_q"
     (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Q)
       (Op.add NumericDType.nat Broadcast.scalarL
         (Op.add NumericDType.nat Broadcast.nil
           (Op.add NumericDType.nat Broadcast.nil
             (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_bh")
               (Op.constNat s_qk_h))
             (Op.mul NumericDType.nat Broadcast.nil
               (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_c")
                 (Op.constNat BT))
               (Op.constNat DK)))
           (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k")
             (Op.constNat BK)))
         (Op.ref TileDType.nat [BK] "offs"))),
   Stmt.assign TileDType.ptr [BK] "p_g"
     (Op.ptrAdd Broadcast.scalarL (Op.ptrBase G)
       (Op.add NumericDType.nat Broadcast.scalarL
         (Op.add NumericDType.nat Broadcast.nil
           (Op.add NumericDType.nat Broadcast.nil
             (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_bh")
               (Op.constNat s_qk_h))
             (Op.mul NumericDType.nat Broadcast.nil
               (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_c")
                 (Op.constNat BT))
               (Op.constNat DK)))
           (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k")
             (Op.constNat BK)))
         (Op.ref TileDType.nat [BK] "offs"))),
   Stmt.assign TileDType.ptr [BK] "p_k"
     (Op.ptrAdd Broadcast.scalarL (Op.ptrBase K)
       (Op.add NumericDType.nat Broadcast.scalarL
         (Op.add NumericDType.nat Broadcast.nil
           (Op.add NumericDType.nat Broadcast.nil
             (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_bh")
               (Op.constNat s_qk_h))
             (Op.mul NumericDType.nat Broadcast.nil
               (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_c")
                 (Op.constNat BT))
               (Op.constNat DK)))
           (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k")
             (Op.constNat BK)))
         (Op.ref TileDType.nat [BK] "offs"))),
   Stmt.assign TileDType.ptr [BK] "p_qg"
     (Op.ptrAdd Broadcast.scalarL (Op.ptrBase QG)
       (Op.add NumericDType.nat Broadcast.scalarL
         (Op.add NumericDType.nat Broadcast.nil
           (Op.add NumericDType.nat Broadcast.nil
             (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_bh")
               (Op.constNat s_qk_h))
             (Op.mul NumericDType.nat Broadcast.nil
               (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_c")
                 (Op.constNat BT))
               (Op.constNat DK)))
           (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k")
             (Op.constNat BK)))
         (Op.ref TileDType.nat [BK] "offs"))),
   Stmt.assign TileDType.ptr [BK] "p_kg"
     (Op.ptrAdd Broadcast.scalarL (Op.ptrBase KG)
       (Op.add NumericDType.nat Broadcast.scalarL
         (Op.add NumericDType.nat Broadcast.nil
           (Op.add NumericDType.nat Broadcast.nil
             (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_bh")
               (Op.constNat s_qk_h))
             (Op.mul NumericDType.nat Broadcast.nil
               (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_c")
                 (Op.constNat BT))
               (Op.constNat DK)))
           (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k")
             (Op.constNat BK)))
         (Op.ref TileDType.nat [BK] "offs"))),
   Stmt.assign TileDType.bool [BK] "mask"
     (Op.lt ComparableDType.nat Broadcast.scalarR
       (Op.add NumericDType.nat Broadcast.scalarL
         (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k")
           (Op.constNat BK))
         (Op.ref TileDType.nat [BK] "offs"))
       (Op.constNat DK))]

/-- The `last_decay` load's address op — the `g` row at chunk row
`i_c·BT + BT − 1` (the last row of this program's `g` chunk). -/
private def prepLdAddrOp (s_qk_h BT BK DK : Nat) : Op TileDType.nat [BK] :=
  Op.add NumericDType.nat Broadcast.scalarL
    (Op.add NumericDType.nat Broadcast.nil
      (Op.add NumericDType.nat Broadcast.nil
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_bh")
          (Op.constNat s_qk_h))
        (Op.mul NumericDType.nat Broadcast.nil
          (Op.sub NumericDType.nat Broadcast.nil
            (Op.add NumericDType.nat Broadcast.nil
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_c")
                (Op.constNat BT))
              (Op.constNat BT))
            (Op.constNat 1))
          (Op.constNat DK)))
      (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k")
        (Op.constNat BK)))
    (Op.ref TileDType.nat [BK] "offs")

/-- The lowered `last_decay` statement: an **unmasked** region load of the
last `g` row (the Python `tl.load` carries no `mask=`). -/
private def prepLdStmt (G : RegionName) (s_qk_h BT BK DK : Nat) : Stmt :=
  Stmt.assign TileDType.real [BK] "last_decay"
    (Op.load TileDType.real
      (MemAccess.region G (prepLdAddrOp s_qk_h BT BK DK))
      MaskOpt.none)

/-- The `toAlgKernel` body decomposes into the 10-statement prefix, the
`last_decay` load, and the single `range(BT)` two-store loop (whose
lowered body is the proven `prepBody`). -/
private theorem prep_body_decomp (Q K G QG KG : RegionName)
    (s_qk_h DK BT BK : Nat) (scale : ℝ) :
    (prepare_qg_kg_surface Q K G QG KG s_qk_h DK BT BK scale).toAlgKernel.body
      = prepPre10 Q K G QG KG s_qk_h BT BK DK
        ++ (prepLdStmt G s_qk_h BT BK DK
            :: [Stmt.forRange "_i" 0 BT 1 (prepBody Q K G QG KG s_qk_h DK BT BK scale)]) := by
  rfl

/-! ### IO signature -/

/-- **Streaming IO signature** of `prepare_qg_kg` on the grouped
multi-channel per-step emit skin (S3: two in-loop stores). Step `t` of the
`range(BT)` loop reads the `BK`-lane `q`/`k`/`g` rows (input channels
`0`/`1`/`2`) and stores the `BK`-lane `qg`/`kg` windows (output channels
`0`/`1`) at the **`.real`** grid (`outDType` default — the stores'
`.to(*.dtype.element_ty)` casts erase to `.real`, so the per-step stores
have no quantization event). All five channels share the surface's
effective row address, transcribed verbatim (`pid₀ = i_k`, `pid₁ = i_c`,
`pid₂ = i_bh`; the surface models the `+= DK` pointer advance, so row
`t`'s address is the base plus `t·DK`):

`i_bh·s_qk_h + i_c·BT·DK + i_k·BK + j + t·DK`.

Masks: the `q`/`k` read masks and both write masks are the kernel's
single, `t`-independent bound mask `i_k·BK + j < DK`. The `g` channel's
read mask carries the extra disjunct `t = BT − 1`: the pre-loop
`last_decay` load is **unmasked** in the Python (`tl.load` without
`mask=`) and reads every lane of the last `g` row — its addresses are
exactly the `g` stream's step-`BT−1` cells, so no extra channel is
needed, only the honestly widened last-row read window. -/
def prepareQgKgKernelIO (Q K G QG KG : RegionName)
    (s_qk_h DK BT BK : Nat) (scale : ℝ) :
    StreamGroupedEmitMasked3DKernelIO where
  kernel := prepare_qg_kg_surface Q K G QG KG s_qk_h DK BT BK scale
  nIn := 3
  nOut := 2
  bufs := [Q, K, G, QG, KG]
  inp := fun i => match i with
    | ⟨0, _⟩ => Q
    | ⟨1, _⟩ => K
    | ⟨2, _⟩ => G
    | ⟨n + 3, h⟩ => absurd h (by omega)
  out := fun o => match o with
    | ⟨0, _⟩ => QG
    | ⟨1, _⟩ => KG
    | ⟨n + 2, h⟩ => absurd h (by omega)
  T := BT
  B := BK
  read := fun _ p₀ p₁ p₂ t j =>
    p₂ * s_qk_h + p₁ * BT * DK + p₀ * BK + j.val + t.val * DK
  readMask := fun i p₀ _ _ t j => match i with
    | ⟨0, _⟩ => p₀ * BK + j.val < DK
    | ⟨1, _⟩ => p₀ * BK + j.val < DK
    | ⟨2, _⟩ => p₀ * BK + j.val < DK ∨ t.val = BT - 1
    | ⟨n + 3, h⟩ => absurd h (by omega)
  write := fun _ p₀ p₁ p₂ t j =>
    p₂ * s_qk_h + p₁ * BT * DK + p₀ * BK + j.val + t.val * DK
  writeMask := fun _ p₀ _ _ _ j => p₀ * BK + j.val < DK

/-! ### The stream-level spec -/

/-- The stream-level `prepare_qg_kg` spec (the genre's *emit* shape, one
clause per output channel): window `(t, j)` of channel `0` (`qg`) holds
`q[t,j] · exp2(g[t,j]) · scale`, of channel `1` (`kg`) holds
`k[t,j] · exp2(g[BT−1,j] − g[t,j])`, with `exp2(x) = exp(x·log 2)` — the
exact spellings of the proven closed forms `prepareQgClosed` /
`prepareKgClosed`, re-indexed to the curried streams. The `last_decay` row
is the `g` stream's step-`BT−1` cells (`xs 2 ⟨BT−1, _⟩ j`); the `dite`
keeps `f` total in `BT` — its `else` branch is unreachable under the
headline's `hBT : 0 < BT`. The channel `match` lives in this single shared
def (inline per-site matchers block cross-declaration `exact`). -/
noncomputable def prepStreamSpec (BT BK : Nat) (scale : ℝ)
    (xs : Fin 3 → Fin BT → Fin BK → ℝ) (o : Fin 2) (t : Fin BT) (j : Fin BK) : ℝ :=
  match o with
  | ⟨0, _⟩ =>
      xs (⟨0, by omega⟩ : Fin 3) t j
        * Real.exp (xs (⟨2, by omega⟩ : Fin 3) t j * Real.log 2) * scale
  | ⟨1, _⟩ =>
      if h : 0 < BT then
        xs (⟨1, by omega⟩ : Fin 3) t j
          * Real.exp ((xs (⟨2, by omega⟩ : Fin 3) ⟨BT - 1, Nat.sub_lt h Nat.one_pos⟩ j
              - xs (⟨2, by omega⟩ : Fin 3) t j) * Real.log 2)
      else 0
  | ⟨n + 2, h2⟩ => absurd h2 (by omega)

/-- The `last_decay` load address in stream form: the kernel's
`(i_c·BT + BT − 1)·DK` row spelling is the `g` stream's step-`BT−1`
address (needs `0 < BT` to move the `− 1` inside the sum). -/
private theorem prep_ld_addr_eq (s : BlockState) (s_qk_h BT BK DK : Nat)
    (hBT : 0 < BT) (i : Fin BK) :
    s.pids 2 * s_qk_h + (s.pids 1 * BT + BT - 1) * DK + s.pids 0 * BK + i.val
      = s.pids 2 * s_qk_h + s.pids 1 * BT * DK + s.pids 0 * BK + i.val
        + (BT - 1) * DK := by
  rw [show s.pids 1 * BT + BT - 1 = s.pids 1 * BT + (BT - 1) from by omega,
    Nat.add_mul]
  ring

/-- Per-lane `qg` spec bridge: at a write-active window `(t, j)` the
stream spec's channel-`0` clause **is** the exact stack's closed form
`prepareQgClosed` — every stream cell it mentions is mask-pinned (the
masks are `t`-independent and lane `j` is active). -/
private theorem prepStreamSpec_qg_eq_closed (Q G : RegionName) (s₀ : BlockState)
    (s_qk_h DK BT BK : Nat) (scale : ℝ) (xs : Fin 3 → Fin BT → Fin BK → ℝ)
    (h0 : (0 : Nat) < 2)
    (hxQ : ∀ (t : Fin BT) (j : Fin BK), s₀.pids 0 * BK + j.val < DK →
      s₀.readMem Q
          (s₀.pids 2 * s_qk_h + s₀.pids 1 * BT * DK + s₀.pids 0 * BK + j.val + t.val * DK)
        = xs (⟨0, by omega⟩ : Fin 3) t j)
    (hxG : ∀ (t : Fin BT) (j : Fin BK), s₀.pids 0 * BK + j.val < DK →
      s₀.readMem G
          (s₀.pids 2 * s_qk_h + s₀.pids 1 * BT * DK + s₀.pids 0 * BK + j.val + t.val * DK)
        = xs (⟨2, by omega⟩ : Fin 3) t j)
    (t : Fin BT) (j : Fin BK) (hj : active s₀ DK BK j) :
    prepStreamSpec BT BK scale xs ⟨0, h0⟩ t j
      = prepareQgClosed s₀ Q G s_qk_h DK BT BK t j scale := by
  have hq := hxQ t j hj
  have hg := hxG t j hj
  rw [fwd_addr_eq] at hq hg
  show xs (⟨0, by omega⟩ : Fin 3) t j
      * Real.exp (xs (⟨2, by omega⟩ : Fin 3) t j * Real.log 2) * scale = _
  rw [← hq, ← hg]
  simp only [prepareQgClosed]

/-- Per-lane `kg` spec bridge: at a write-active window `(t, j)` the
stream spec's channel-`1` clause **is** `prepareKgClosed` — the `k`/`g`
cells at row `t` are mask-pinned at the active lane, and the `last_decay`
cell `xs 2 ⟨BT−1, _⟩ j` is pinned by the `g` channel's widened last-row
read window (the `Or.inr` disjunct). -/
private theorem prepStreamSpec_kg_eq_closed (K G : RegionName) (s₀ : BlockState)
    (s_qk_h DK BT BK : Nat) (scale : ℝ) (xs : Fin 3 → Fin BT → Fin BK → ℝ)
    (hBT : 0 < BT) (h1 : (1 : Nat) < 2)
    (hxK : ∀ (t : Fin BT) (j : Fin BK), s₀.pids 0 * BK + j.val < DK →
      s₀.readMem K
          (s₀.pids 2 * s_qk_h + s₀.pids 1 * BT * DK + s₀.pids 0 * BK + j.val + t.val * DK)
        = xs (⟨1, by omega⟩ : Fin 3) t j)
    (hxG : ∀ (t : Fin BT) (j : Fin BK),
      (s₀.pids 0 * BK + j.val < DK ∨ t.val = BT - 1) →
      s₀.readMem G
          (s₀.pids 2 * s_qk_h + s₀.pids 1 * BT * DK + s₀.pids 0 * BK + j.val + t.val * DK)
        = xs (⟨2, by omega⟩ : Fin 3) t j)
    (t : Fin BT) (j : Fin BK) (hj : active s₀ DK BK j) :
    prepStreamSpec BT BK scale xs ⟨1, h1⟩ t j
      = prepareKgClosed s₀ K G s_qk_h DK BT BK t j := by
  have hk := hxK t j hj
  have hg := hxG t j (Or.inl hj)
  rw [fwd_addr_eq] at hk hg
  have hglast : s₀.readMem G (offset s₀ s_qk_h DK (BT - 1) BT BK j)
      = xs (⟨2, by omega⟩ : Fin 3) ⟨BT - 1, Nat.sub_lt hBT Nat.one_pos⟩ j := by
    have h := hxG ⟨BT - 1, Nat.sub_lt hBT Nat.one_pos⟩ j (Or.inr rfl)
    rwa [fwd_addr_eq] at h
  show (if h : 0 < BT then
      xs (⟨1, by omega⟩ : Fin 3) t j
        * Real.exp ((xs (⟨2, by omega⟩ : Fin 3) ⟨BT - 1, Nat.sub_lt h Nat.one_pos⟩ j
            - xs (⟨2, by omega⟩ : Fin 3) t j) * Real.log 2)
    else 0) = _
  rw [dif_pos hBT, ← hk, ← hg, ← hglast]
  simp only [prepareKgClosed]

/-! ### Cast-free collapses and the covered fragment -/

/-- The 10-statement prefix is cast-free: it steps identically under
`stepStmtsR R`. -/
private theorem prepPre10_castFree (R : RoundingModel) (Q K G QG KG : RegionName)
    (s_qk_h BT BK DK : Nat) (t : BlockState) :
    stepStmtsR R (prepPre10 Q K G QG KG s_qk_h BT BK DK) t
      = stepStmts (prepPre10 Q K G QG KG s_qk_h BT BK DK) t := by
  simp only [prepPre10, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def]
  rfl

/-- The `last_decay` load is cast-free: an unmasked `.real` region load
(nat address arithmetic, no rounding event) steps identically under
`stepStmtR R`. -/
private theorem prepLd_castFree (R : RoundingModel) (G : RegionName)
    (s_qk_h BT BK DK : Nat) (t : BlockState) :
    stepStmtR R (prepLdStmt G s_qk_h BT BK DK) t
      = stepStmt (prepLdStmt G s_qk_h BT BK DK) t := by
  simp only [prepLdStmt, prepLdAddrOp, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def, Rcast_real_real]

/-- The loop body is cast-free **including its two in-loop masked `.real`
stores**: the masked `.real` loads with erased fp32 casts are exact under
every `R`, the `exp2` scalings are real arithmetic, and `stepStmtR`
delegates `.real`-typed stores to the exact `writeMemTyped`, so the whole
two-store loop steps identically under `stepStmtsR R` and the exact
`prepInv` stack transports to `execR`. -/
private theorem prepBody_castFree (R : RoundingModel) (Q K G QG KG : RegionName)
    (s_qk_h DK BT BK : Nat) (scale : ℝ) (t : BlockState) :
    stepStmtsR R (prepBody Q K G QG KG s_qk_h DK BT BK scale) t
      = stepStmts (prepBody Q K G QG KG s_qk_h DK BT BK scale) t := by
  simp only [prepBody, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def, Rcast_real_real, BlockState.writeMemTypedR]
  rfl

/-- The full prepare surface sits inside the flat-memory bridge's covered
fragment (`FlattenOk`; the `forRange` clause recurses into the cast-free
body). -/
private theorem prep_flattenOk (Q K G QG KG : RegionName)
    (s_qk_h DK BT BK : Nat) (scale : ℝ) :
    ((prepare_qg_kg_surface Q K G QG KG s_qk_h DK BT BK scale).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  rw [prep_body_decomp]
  simp [prepPre10, prepLdStmt, prepLdAddrOp, prepBody, StmtList.FlattenOk,
    Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-! ### The prologue walk (private copy of `prepare_qg_kg_loop_drive`'s
prefix walk, packaged to expose the pre-load state and the memory frame) -/

/-- The 10-statement prefix walk plus the `last_decay` load: establishes
`prepInv … 0` at the loop entry. Exposes the intermediate state `s10`
(whose pinned `i_*`/`offs` registers the `TraceSafeR` walk needs to bound
the **unmasked** pre-loop load) and the memory frame (`s11.mem = s.mem` —
the prologue never touches memory). `hBT` feeds the `last_decay` row
arithmetic (`i_c·BT + BT − 1 = i_c·BT + (BT − 1)`). -/
private theorem prep_preLoop (Q K G QG KG : RegionName)
    (s_qk_h DK BT BK : Nat) (scale : ℝ) (hBT : 0 < BT) (s : BlockState) :
    ∃ s10 s11,
      stepStmts (prepPre10 Q K G QG KG s_qk_h BT BK DK) s = some s10
      ∧ stepStmt (prepLdStmt G s_qk_h BT BK DK) s10 = some s11
      ∧ s10.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2))
      ∧ s10.regs .nat [] "i_c" = some (Tile.scalar (s.pids 1))
      ∧ s10.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0))
      ∧ s10.regs .nat [BK] "offs" = some (prepOffsTile BK)
      ∧ prepInv Q K G QG KG s s_qk_h DK BT BK scale 0 s11
      ∧ s11.mem = s.mem := by
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
  -- last_decay eval (unmasked region load) — uses BT-1 with 0 < BT
  have hldeval : evalOp (Op.load TileDType.real
      (MemAccess.region G (prepLdAddrOp s_qk_h BT BK DK))
      MaskOpt.none) s_mask = some (prepLastDecayTile s G s_qk_h DK BT BK) := by
    simp only [prepLdAddrOp, evalOp, Option.bind, hsmask_bh, hsmask_c, hsmask_k, hsmask_o,
      evalOp_constNat, Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add,
      NumericDType.mul, NumericDType.sub, Tile.bop, Tile.scalar]
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
  have hstepLd : stepStmt (prepLdStmt G s_qk_h BT BK DK) s_mask = some s11 := by
    rw [hs11]; exact stepStmt_assign_eq_some hldeval
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
  refine ⟨s_mask, s11, ?_, hstepLd, hsmask_bh, hsmask_c, hsmask_k, hsmask_o, hP0, ?_⟩
  · unfold prepPre10
    rw [stepStmts.cons_some hstepK, stepStmts.cons_some hstepC, stepStmts.cons_some hstepB,
      stepStmts.cons_some hstepOffs, stepStmts.cons_some hstepPq, stepStmts.cons_some hstepPg,
      stepStmts.cons_some hstepPk, stepStmts.cons_some hstepPqg, stepStmts.cons_some hstepPkg,
      stepStmts.cons_some hstepMask, stepStmts.nil]
  · rw [hs11, hs_mask, hs_pkg, hs_pqg, hs_pk, hs_pg, hs_pq, hs_offs, hs_ibh, hs_ic, hs_ik]
    rfl

/-! ### Cell-level memory frame of one loop iteration -/

/-- **Cell-level frame of one loop iteration** (the `mem` twin of
`prepare_qg_kg_step`, same walk with opaque register values): from the
`prepInv` `p_qg`/`p_kg`/`mask` pins, one two-store body iteration leaves
every cell off the row-`m` write windows of **both** output buffers
untouched. -/
private theorem prep_body_step_frame (Q K G QG KG : RegionName)
    (s_qk_h DK BT BK : Nat) (scale : ℝ) (s sc s' : BlockState) (m : Nat)
    (hpqg : sc.regs .ptr [BK] "p_qg" = some (prepPtrTile s QG s_qk_h DK BT BK m))
    (hpkg : sc.regs .ptr [BK] "p_kg" = some (prepPtrTile s KG s_qk_h DK BT BK m))
    (hmask : sc.regs .bool [BK] "mask" = some (fwdMaskTile s DK BK))
    (hstep : stepStmts (prepBody Q K G QG KG s_qk_h DK BT BK scale)
      (sc.setReg "_i" .nat [] (Tile.scalar m)) = some s')
    (r : RegionName) (oo : Nat)
    (hcQG : r ≠ QG ∨ ∀ i : Fin BK, active s DK BK i →
      oo ≠ offset s s_qk_h DK m BT BK i)
    (hcKG : r ≠ KG ∨ ∀ i : Fin BK, active s DK BK i →
      oo ≠ offset s s_qk_h DK m BT BK i) :
    s'.mem r oo = sc.mem r oo := by
  set sloop := sc.setReg "_i" .nat [] (Tile.scalar m) with hsloop
  unfold prepBody at hstep
  obtain ⟨v1, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  obtain ⟨v2, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  obtain ⟨v3, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  obtain ⟨v4, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  obtain ⟨v5, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  set s5 := ((((sloop.setReg "q_val" .real [BK] v1).setReg "k_val" .real [BK] v2).setReg
    "g_val" .real [BK] v3).setReg "q_val" .real [BK] v4).setReg "k_val" .real [BK] v5 with hs5
  have hkval5 : s5.regs .real [BK] "k_val" = some v5 := by
    rw [hs5]; exact BlockState.setReg_same _ _ _ _ _
  have hqval5 : s5.regs .real [BK] "q_val" = some v4 := by
    rw [hs5, BlockState.setReg_ne_name (h := by decide)]
    exact BlockState.setReg_same _ _ _ _ _
  have hpkg5 : s5.regs .ptr [BK] "p_kg" = some (prepPtrTile s KG s_qk_h DK BT BK m) := by
    rw [hs5, BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide), hsloop,
      BlockState.setReg_ne_name (h := by decide)]
    exact hpkg
  have hpqg5 : s5.regs .ptr [BK] "p_qg" = some (prepPtrTile s QG s_qk_h DK BT BK m) := by
    rw [hs5, BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide), hsloop,
      BlockState.setReg_ne_name (h := by decide)]
    exact hpqg
  have hmask5 : s5.regs .bool [BK] "mask" = some (fwdMaskTile s DK BK) := by
    rw [hs5, BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide), hsloop,
      BlockState.setReg_ne_name (h := by decide)]
    exact hmask
  -- the first (KG) store
  set vfun1 : TileIndex [BK] → ℝ := fun k => (v5.data k).unbotD 0 with hvfun1
  set sst1 := (TileShape.allIndices [BK]).foldl
    (fun (acc : BlockState) k =>
      if active s DK BK k.1 then
        acc.writeMem KG (offset s s_qk_h DK m BT BK k.1) (vfun1 k)
      else acc) s5 with hsst1
  have hstore1 : stepStmt (Stmt.store TileDType.real [BK]
      (MemAccess.ptr (Op.ref TileDType.ptr [BK] "p_kg"))
      (Op.ref TileDType.real [BK] "k_val")
      (MaskOpt.mask (Op.ref TileDType.bool [BK] "mask"))) s5 = some sst1 := by
    simp only [stepStmt, evalOp, Option.bind, Option.map, hkval5, hpkg5, hmask5]
    apply congrArg some
    rw [hsst1]
    congr 1
    funext acc k
    simp only [prepPtrTile, fwdMaskTile]
    by_cases ha : active s DK BK k.1
    · rw [if_pos (by simpa using ha), if_pos ha, BlockState.writeMemTyped_real]
      rfl
    · rw [if_neg (by simpa using ha), if_neg ha]
  rw [stepStmts.cons_some hstore1] at hstep
  -- the second (QG) store: the scatter fold preserves registers
  have hqval1 : sst1.regs .real [BK] "q_val" = some v4 := by
    rw [hsst1, BlockState.foldl_writeMem_prop_masked_regs]
    exact hqval5
  have hpqg1 : sst1.regs .ptr [BK] "p_qg" = some (prepPtrTile s QG s_qk_h DK BT BK m) := by
    rw [hsst1, BlockState.foldl_writeMem_prop_masked_regs]
    exact hpqg5
  have hmask1 : sst1.regs .bool [BK] "mask" = some (fwdMaskTile s DK BK) := by
    rw [hsst1, BlockState.foldl_writeMem_prop_masked_regs]
    exact hmask5
  set vfun2 : TileIndex [BK] → ℝ := fun k => (v4.data k).unbotD 0 with hvfun2
  set sst2 := (TileShape.allIndices [BK]).foldl
    (fun (acc : BlockState) k =>
      if active s DK BK k.1 then
        acc.writeMem QG (offset s s_qk_h DK m BT BK k.1) (vfun2 k)
      else acc) sst1 with hsst2
  have hstore2 : stepStmt (Stmt.store TileDType.real [BK]
      (MemAccess.ptr (Op.ref TileDType.ptr [BK] "p_qg"))
      (Op.ref TileDType.real [BK] "q_val")
      (MaskOpt.mask (Op.ref TileDType.bool [BK] "mask"))) sst1 = some sst2 := by
    simp only [stepStmt, evalOp, Option.bind, Option.map, hqval1, hpqg1, hmask1]
    apply congrArg some
    rw [hsst2]
    congr 1
    funext acc k
    simp only [prepPtrTile, fwdMaskTile]
    by_cases ha : active s DK BK k.1
    · rw [if_pos (by simpa using ha), if_pos ha, BlockState.writeMemTyped_real]
      rfl
    · rw [if_neg (by simpa using ha), if_neg ha]
  rw [stepStmts.cons_some hstore2] at hstep
  obtain ⟨w1, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  obtain ⟨w2, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  obtain ⟨w3, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  obtain ⟨w4, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  obtain ⟨w5, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  rw [stepStmts.nil] at hstep
  obtain rfl := Option.some.inj hstep
  simp only [BlockState.setReg_mem]
  rw [hsst2]
  refine Eq.trans (foldl_writeMem_prop_preserve_cell _ _ _ r oo _ _ ?_) ?_
  · intro k hk hmk hbad
    rcases hcQG with hne' | hno
    · exact hne' hbad.1
    · exact hno k.1 hmk hbad.2
  · rw [hsst1]
    refine Eq.trans (foldl_writeMem_prop_preserve_cell _ _ _ r oo _ _ ?_) ?_
    · intro k hk hmk hbad
      rcases hcKG with hne' | hno
      · exact hne' hbad.1
      · exact hno k.1 hmk hbad.2
    · rw [hs5, hsloop]
      simp only [BlockState.setReg_mem]

/-! ### The `TraceSafeR` walk -/

/-- `regs` pass through any `foldl` whose step preserves the register
file. -/
private theorem foldl_regs_of_preserve {α : Type}
    (f : BlockState → α → BlockState)
    (hf : ∀ s a, (f s a).regs = s.regs) :
    ∀ (l : List α) (s : BlockState), (l.foldl f s).regs = s.regs
  | [], _ => rfl
  | a :: l, s => by
      rw [List.foldl_cons, foldl_regs_of_preserve f hf l (f s a), hf]

/-- A successful `stepStmtR R` step of a `.real` masked `.ptr`-form store
leaves the register file unchanged — a store only folds memory writes.
Lets the `TraceSafeR` walk thread register pins across the body's first
store to reach the second (the two-store novelty of this kernel). -/
private theorem stepStmtR_store_real_regs {R : RoundingModel} {sh : TileShape}
    {pe : Op TileDType.ptr sh} {ve : Op TileDType.real sh}
    {me : Op TileDType.bool sh} {s s' : BlockState}
    (h : stepStmtR R (Stmt.store TileDType.real sh (MemAccess.ptr pe) ve
      (MaskOpt.mask me)) s = some s') :
    s'.regs = s.regs := by
  cases hv : evalOpR R ve s with
  | none => simp [stepStmtR, hv] at h
  | some vs =>
      cases hm : evalOpR R me s with
      | none => simp [stepStmtR, hv, hm] at h
      | some ms =>
          cases hp : evalOpR R pe s with
          | none => simp [stepStmtR, hv, hm, hp] at h
          | some ps =>
              simp only [stepStmtR, hv, hm, hp, Option.map_some, bind,
                Option.bind_some, Option.some.injEq] at h
              rw [← h]
              refine foldl_regs_of_preserve _ ?_ _ _
              intro acc i
              by_cases hai : ms.data i = Bool.true
              · rw [if_pos hai]; rfl
              · rw [if_neg hai]

/-- `evalOpR` = `evalOp` on the cast-free nat address op of the
`last_decay` load (register refs and nat arithmetic — `R` never enters). -/
private theorem prepLdAddrR_eq (R : RoundingModel) (s_qk_h BT BK DK : Nat)
    (st : BlockState) :
    evalOpR R (prepLdAddrOp s_qk_h BT BK DK) st
      = evalOp (prepLdAddrOp s_qk_h BT BK DK) st := by
  simp only [prepLdAddrOp, evalOpR.eq_def, evalOp.eq_def]

/-- The per-lane value of the `last_decay` load's address op at any state
carrying the prologue's pinned `i_*`/`offs` registers. -/
private theorem prepLdAddr_eval (s_qk_h BT BK DK : Nat) (s st : BlockState)
    (hbh : st.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)))
    (hc : st.regs .nat [] "i_c" = some (Tile.scalar (s.pids 1)))
    (hk : st.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)))
    (ho : st.regs .nat [BK] "offs" = some (prepOffsTile BK)) :
    evalOp (prepLdAddrOp s_qk_h BT BK DK) st
      = some (Tile.vec (fun i : Fin BK =>
          s.pids 2 * s_qk_h + (s.pids 1 * BT + BT - 1) * DK + s.pids 0 * BK + i.val)) := by
  simp only [prepLdAddrOp, evalOp, Option.bind, hbh, hc, hk, ho, evalOp_constNat,
    Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul,
    NumericDType.sub, Tile.bop, Tile.scalar]
  apply congrArg some
  apply Tile.ext
  intro idx
  obtain ⟨i, u⟩ := idx
  simp only [Tile.vec, prepOffsTile,
    Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL,
    Broadcast.leftIndex_nil, Broadcast.rightIndex_nil]

set_option maxHeartbeats 8000000 in
/-- Per-iteration `TraceSafeListR` for the `prepare_qg_kg` loop body: the
two scalings and the five pointer advances are register-only; the three
masked loads' and the two masked stores' **active** lanes are exactly the
skin's (`t`-independent) masked windows, in bounds by the corresponding
window bound (instantiated at row `m`). The pins survive the first store
via `stepStmtR_store_real_regs`. -/
private theorem prep_bodySafeR (R : RoundingModel) (bounds : RegionBounds)
    (Q K G QG KG : RegionName) (s_qk_h DK BT BK : Nat) (scale : ℝ)
    (s stt : BlockState) (m : Nat)
    (hpq : stt.regs .ptr [BK] "p_q" = some (prepPtrTile s Q s_qk_h DK BT BK m))
    (hpg : stt.regs .ptr [BK] "p_g" = some (prepPtrTile s G s_qk_h DK BT BK m))
    (hpk : stt.regs .ptr [BK] "p_k" = some (prepPtrTile s K s_qk_h DK BT BK m))
    (hpqg : stt.regs .ptr [BK] "p_qg" = some (prepPtrTile s QG s_qk_h DK BT BK m))
    (hpkg : stt.regs .ptr [BK] "p_kg" = some (prepPtrTile s KG s_qk_h DK BT BK m))
    (hmask : stt.regs .bool [BK] "mask" = some (fwdMaskTile s DK BK))
    (hbQ : ∀ i : Fin BK, active s DK BK i → offset s s_qk_h DK m BT BK i < bounds Q)
    (hbK : ∀ i : Fin BK, active s DK BK i → offset s s_qk_h DK m BT BK i < bounds K)
    (hbG : ∀ i : Fin BK, active s DK BK i → offset s s_qk_h DK m BT BK i < bounds G)
    (hbQG : ∀ i : Fin BK, active s DK BK i → offset s s_qk_h DK m BT BK i < bounds QG)
    (hbKG : ∀ i : Fin BK, active s DK BK i → offset s s_qk_h DK m BT BK i < bounds KG) :
    Stmt.TraceSafeListR R bounds (prepBody Q K G QG KG s_qk_h DK BT BK scale)
      (stt.setReg "_i" .nat [] (Tile.scalar m)) := by
  set sloop := stt.setReg "_i" .nat [] (Tile.scalar m) with hsloop
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
  unfold prepBody
  -- (1) the masked `q_val` load
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun t1 ht1 => ?_)
  · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
    refine ⟨by simp [Op.SafeAtR.eq_def], by simp [Op.SafeAtR.eq_def], ?_⟩
    intro ptrs hptrs idx hactive
    rw [evalOpR_ref, hpq'] at hptrs
    obtain rfl := Option.some.inj hptrs
    obtain ⟨masks, hm, hmi⟩ := hactive
    rw [evalOpR_ref, hmask'] at hm
    obtain rfl := Option.some.inj hm
    have ha : active s DK BK idx.1 := by simpa [fwdMaskTile] using hmi
    simpa [prepPtrTile] using hbQ idx.1 ha
  · obtain ⟨v1, -, rfl⟩ := stepStmtR_assign_inv ht1
    have hpk1 : (sloop.setReg "q_val" .real [BK] v1).regs .ptr [BK] "p_k"
        = some (prepPtrTile s K s_qk_h DK BT BK m) := by
      rw [BlockState.setReg_ne_name (h := by decide)]
      exact hpk'
    have hmask1 : (sloop.setReg "q_val" .real [BK] v1).regs .bool [BK] "mask"
        = some (fwdMaskTile s DK BK) := by
      rw [BlockState.setReg_ne_name (h := by decide)]
      exact hmask'
    -- (2) the masked `k_val` load
    refine Stmt.TraceSafeListR.cons_intro ?_ (fun t2 ht2 => ?_)
    · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
        MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
      refine ⟨by simp [Op.SafeAtR.eq_def], by simp [Op.SafeAtR.eq_def], ?_⟩
      intro ptrs hptrs idx hactive
      rw [evalOpR_ref, hpk1] at hptrs
      obtain rfl := Option.some.inj hptrs
      obtain ⟨masks, hm, hmi⟩ := hactive
      rw [evalOpR_ref, hmask1] at hm
      obtain rfl := Option.some.inj hm
      have ha : active s DK BK idx.1 := by simpa [fwdMaskTile] using hmi
      simpa [prepPtrTile] using hbK idx.1 ha
    · obtain ⟨v2, -, rfl⟩ := stepStmtR_assign_inv ht2
      have hpg2 : ((sloop.setReg "q_val" .real [BK] v1).setReg
          "k_val" .real [BK] v2).regs .ptr [BK] "p_g"
          = some (prepPtrTile s G s_qk_h DK BT BK m) := by
        rw [BlockState.setReg_ne_name (h := by decide),
          BlockState.setReg_ne_name (h := by decide)]
        exact hpg'
      have hmask2 : ((sloop.setReg "q_val" .real [BK] v1).setReg
          "k_val" .real [BK] v2).regs .bool [BK] "mask"
          = some (fwdMaskTile s DK BK) := by
        rw [BlockState.setReg_ne_name (h := by decide),
          BlockState.setReg_ne_name (h := by decide)]
        exact hmask'
      -- (3) the masked `g_val` load
      refine Stmt.TraceSafeListR.cons_intro ?_ (fun t3 ht3 => ?_)
      · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
          MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
        refine ⟨by simp [Op.SafeAtR.eq_def], by simp [Op.SafeAtR.eq_def], ?_⟩
        intro ptrs hptrs idx hactive
        rw [evalOpR_ref, hpg2] at hptrs
        obtain rfl := Option.some.inj hptrs
        obtain ⟨masks, hm, hmi⟩ := hactive
        rw [evalOpR_ref, hmask2] at hm
        obtain rfl := Option.some.inj hm
        have ha : active s DK BK idx.1 := by simpa [fwdMaskTile] using hmi
        simpa [prepPtrTile] using hbG idx.1 ha
      · obtain ⟨v3, -, rfl⟩ := stepStmtR_assign_inv ht3
        -- (4) `q_val` scaling: register-only
        refine Stmt.TraceSafeListR.cons_intro
          (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun t4 ht4 => ?_)
        obtain ⟨v4, -, rfl⟩ := stepStmtR_assign_inv ht4
        -- (5) `k_val` scaling: register-only
        refine Stmt.TraceSafeListR.cons_intro
          (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun t5 ht5 => ?_)
        obtain ⟨v5, -, rfl⟩ := stepStmtR_assign_inv ht5
        have hpkg5 : (((((sloop.setReg "q_val" .real [BK] v1).setReg
            "k_val" .real [BK] v2).setReg "g_val" .real [BK] v3).setReg
            "q_val" .real [BK] v4).setReg "k_val" .real [BK] v5).regs .ptr [BK] "p_kg"
            = some (prepPtrTile s KG s_qk_h DK BT BK m) := by
          rw [BlockState.setReg_ne_name (h := by decide),
            BlockState.setReg_ne_name (h := by decide),
            BlockState.setReg_ne_name (h := by decide),
            BlockState.setReg_ne_name (h := by decide),
            BlockState.setReg_ne_name (h := by decide)]
          exact hpkg'
        have hpqg5 : (((((sloop.setReg "q_val" .real [BK] v1).setReg
            "k_val" .real [BK] v2).setReg "g_val" .real [BK] v3).setReg
            "q_val" .real [BK] v4).setReg "k_val" .real [BK] v5).regs .ptr [BK] "p_qg"
            = some (prepPtrTile s QG s_qk_h DK BT BK m) := by
          rw [BlockState.setReg_ne_name (h := by decide),
            BlockState.setReg_ne_name (h := by decide),
            BlockState.setReg_ne_name (h := by decide),
            BlockState.setReg_ne_name (h := by decide),
            BlockState.setReg_ne_name (h := by decide)]
          exact hpqg'
        have hmask5 : (((((sloop.setReg "q_val" .real [BK] v1).setReg
            "k_val" .real [BK] v2).setReg "g_val" .real [BK] v3).setReg
            "q_val" .real [BK] v4).setReg "k_val" .real [BK] v5).regs .bool [BK] "mask"
            = some (fwdMaskTile s DK BK) := by
          rw [BlockState.setReg_ne_name (h := by decide),
            BlockState.setReg_ne_name (h := by decide),
            BlockState.setReg_ne_name (h := by decide),
            BlockState.setReg_ne_name (h := by decide),
            BlockState.setReg_ne_name (h := by decide)]
          exact hmask'
        -- (6) the masked `kg` store
        refine Stmt.TraceSafeListR.cons_intro ?_ (fun t6 ht6 => ?_)
        · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MemAccess.SafeAtR,
            MaskOpt.SafeAtR, MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR,
            memAccessActiveAddressSafeR]
          refine ⟨by simp [Op.SafeAtR.eq_def], by simp [Op.SafeAtR.eq_def],
            by simp [Op.SafeAtR.eq_def], ?_⟩
          intro ptrs hptrs idx hactive
          rw [evalOpR_ref, hpkg5] at hptrs
          obtain rfl := Option.some.inj hptrs
          obtain ⟨masks, hm, hmi⟩ := hactive
          rw [evalOpR_ref, hmask5] at hm
          obtain rfl := Option.some.inj hm
          have ha : active s DK BK idx.1 := by simpa [fwdMaskTile] using hmi
          simpa [prepPtrTile] using hbKG idx.1 ha
        · have hregs6 := stepStmtR_store_real_regs ht6
          have hpqg6 : t6.regs .ptr [BK] "p_qg"
              = some (prepPtrTile s QG s_qk_h DK BT BK m) := by
            rw [hregs6]; exact hpqg5
          have hmask6 : t6.regs .bool [BK] "mask" = some (fwdMaskTile s DK BK) := by
            rw [hregs6]; exact hmask5
          -- (7) the masked `qg` store
          refine Stmt.TraceSafeListR.cons_intro ?_ (fun t7 ht7 => ?_)
          · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MemAccess.SafeAtR,
              MaskOpt.SafeAtR, MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR,
              memAccessActiveAddressSafeR]
            refine ⟨by simp [Op.SafeAtR.eq_def], by simp [Op.SafeAtR.eq_def],
              by simp [Op.SafeAtR.eq_def], ?_⟩
            intro ptrs hptrs idx hactive
            rw [evalOpR_ref, hpqg6] at hptrs
            obtain rfl := Option.some.inj hptrs
            obtain ⟨masks, hm, hmi⟩ := hactive
            rw [evalOpR_ref, hmask6] at hm
            obtain rfl := Option.some.inj hm
            have ha : active s DK BK idx.1 := by simpa [fwdMaskTile] using hmi
            simpa [prepPtrTile] using hbQG idx.1 ha
          · -- (8)–(12) pointer advances: register-only
            refine Stmt.TraceSafeListR.cons_intro
              (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun _ _ => ?_)
            refine Stmt.TraceSafeListR.cons_intro
              (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun _ _ => ?_)
            refine Stmt.TraceSafeListR.cons_intro
              (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun _ _ => ?_)
            refine Stmt.TraceSafeListR.cons_intro
              (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun _ _ => ?_)
            exact Stmt.TraceSafeListR.cons_intro
              (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
              (fun _ _ => Stmt.TraceSafeListR.nil_intro)

set_option maxHeartbeats 8000000 in
/-- **The `TraceSafeR` walk for the whole prepare kernel** — the ten
prefix assigns are register-only; the **unmasked** `last_decay` load is
bounded by the `g` window bound at step `BT − 1` (its addresses are the
`g` stream's last-row cells, all lanes — the `Or.inr` disjunct of the
widened `g` read mask); the loop is driven by
`Stmt.forRangeTraceSafeR_inv` over the proven `prepInv`. The aliasing
hypotheses and `hBK`/`hBT` feed the reused `prepare_qg_kg_step` (see the
headline's provenance notes). -/
private theorem prep_traceSafeR (R : RoundingModel) (bounds : RegionBounds)
    (Q K G QG KG : RegionName) (s_qk_h DK BT BK : Nat) (scale : ℝ)
    (hQ_QG : Q ≠ QG) (hQ_KG : Q ≠ KG) (hK_QG : K ≠ QG) (hK_KG : K ≠ KG)
    (hG_QG : G ≠ QG) (hG_KG : G ≠ KG) (hQG_KG : QG ≠ KG)
    (hBK : BK ≤ DK) (hBT : 0 < BT) (s : BlockState)
    (hbQ : ∀ (t : Fin BT) (j : Fin BK), s.pids 0 * BK + j.val < DK →
      s.pids 2 * s_qk_h + s.pids 1 * BT * DK + s.pids 0 * BK + j.val + t.val * DK
        < bounds Q)
    (hbK : ∀ (t : Fin BT) (j : Fin BK), s.pids 0 * BK + j.val < DK →
      s.pids 2 * s_qk_h + s.pids 1 * BT * DK + s.pids 0 * BK + j.val + t.val * DK
        < bounds K)
    (hbG : ∀ (t : Fin BT) (j : Fin BK),
      (s.pids 0 * BK + j.val < DK ∨ t.val = BT - 1) →
      s.pids 2 * s_qk_h + s.pids 1 * BT * DK + s.pids 0 * BK + j.val + t.val * DK
        < bounds G)
    (hbQG : ∀ (t : Fin BT) (j : Fin BK), s.pids 0 * BK + j.val < DK →
      s.pids 2 * s_qk_h + s.pids 1 * BT * DK + s.pids 0 * BK + j.val + t.val * DK
        < bounds QG)
    (hbKG : ∀ (t : Fin BT) (j : Fin BK), s.pids 0 * BK + j.val < DK →
      s.pids 2 * s_qk_h + s.pids 1 * BT * DK + s.pids 0 * BK + j.val + t.val * DK
        < bounds KG) :
    ((prepare_qg_kg_surface Q K G QG KG s_qk_h DK BT BK scale).toAlgKernel).TraceSafeR R bounds s := by
  -- window instantiators at `offset`-form addresses
  have hbQ' : ∀ m : Nat, m < BT → ∀ i : Fin BK, active s DK BK i →
      offset s s_qk_h DK m BT BK i < bounds Q := by
    intro m hm i ha
    have h : s.pids 2 * s_qk_h + s.pids 1 * BT * DK + s.pids 0 * BK + i.val + m * DK
        < bounds Q := hbQ ⟨m, hm⟩ i ha
    rwa [fwd_addr_eq] at h
  have hbK' : ∀ m : Nat, m < BT → ∀ i : Fin BK, active s DK BK i →
      offset s s_qk_h DK m BT BK i < bounds K := by
    intro m hm i ha
    have h : s.pids 2 * s_qk_h + s.pids 1 * BT * DK + s.pids 0 * BK + i.val + m * DK
        < bounds K := hbK ⟨m, hm⟩ i ha
    rwa [fwd_addr_eq] at h
  have hbG' : ∀ m : Nat, m < BT → ∀ i : Fin BK, active s DK BK i →
      offset s s_qk_h DK m BT BK i < bounds G := by
    intro m hm i ha
    have h : s.pids 2 * s_qk_h + s.pids 1 * BT * DK + s.pids 0 * BK + i.val + m * DK
        < bounds G := hbG ⟨m, hm⟩ i (Or.inl ha)
    rwa [fwd_addr_eq] at h
  have hbQG' : ∀ m : Nat, m < BT → ∀ i : Fin BK, active s DK BK i →
      offset s s_qk_h DK m BT BK i < bounds QG := by
    intro m hm i ha
    have h : s.pids 2 * s_qk_h + s.pids 1 * BT * DK + s.pids 0 * BK + i.val + m * DK
        < bounds QG := hbQG ⟨m, hm⟩ i ha
    rwa [fwd_addr_eq] at h
  have hbKG' : ∀ m : Nat, m < BT → ∀ i : Fin BK, active s DK BK i →
      offset s s_qk_h DK m BT BK i < bounds KG := by
    intro m hm i ha
    have h : s.pids 2 * s_qk_h + s.pids 1 * BT * DK + s.pids 0 * BK + i.val + m * DK
        < bounds KG := hbKG ⟨m, hm⟩ i ha
    rwa [fwd_addr_eq] at h
  unfold Kernel.TraceSafeR
  rw [prep_body_decomp]
  refine Stmt.TraceSafeListR.append_intro _ _ ?_ ?_
  · -- the ten prefix assigns: register-only, safe at every state
    refine Stmt.TraceSafeListR.of_forall _ _ ?_
    intro stmt hst s'
    simp only [prepPre10, List.mem_cons, List.not_mem_nil, or_false] at hst
    rcases hst with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
      simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]
  · intro s1 hs1
    obtain ⟨s10, s11, hsp10, hld, hbh10, hc10, hk10, ho10, hP0, hmem11⟩ :=
      prep_preLoop Q K G QG KG s_qk_h DK BT BK scale hBT s
    rw [prepPre10_castFree R Q K G QG KG s_qk_h BT BK DK s, hsp10] at hs1
    obtain rfl := Option.some.inj hs1
    -- the unmasked `last_decay` load
    refine Stmt.TraceSafeListR.cons_intro ?_ (fun s11' hs11 => ?_)
    · simp only [prepLdStmt, Stmt.TraceSafeR, Op.SafeAtR, MaskOpt.ActiveR,
        MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
      refine ⟨?_, trivial, ?_⟩
      · simp only [prepLdAddrOp, Op.SafeAtR, and_self]
      · intro offsets hoffsets idx _
        rw [prepLdAddrR_eq, prepLdAddr_eval s_qk_h BT BK DK s s10 hbh10 hc10 hk10 ho10]
          at hoffsets
        obtain rfl := Option.some.inj hoffsets
        have hb : s.pids 2 * s_qk_h + s.pids 1 * BT * DK + s.pids 0 * BK + idx.1.val
            + (BT - 1) * DK < bounds G :=
          hbG ⟨BT - 1, Nat.sub_lt hBT Nat.one_pos⟩ idx.1 (Or.inr rfl)
        simpa [Region.cast_id, Tile.vec,
          prep_ld_addr_eq s s_qk_h BT BK DK hBT idx.1] using hb
    · rw [prepLd_castFree R G s_qk_h BT BK DK s10, hld] at hs11
      obtain rfl := Option.some.inj hs11
      -- the two-store loop over the proven invariant
      refine Stmt.TraceSafeListR.cons_intro ?_
        (fun _ _ => Stmt.TraceSafeListR.nil_intro)
      simp only [Stmt.TraceSafeR]
      refine Stmt.forRangeTraceSafeR_inv R bounds "_i" BT 1
        (prepBody Q K G QG KG s_qk_h DK BT BK scale)
        (prepInv Q K G QG KG s s_qk_h DK BT BK scale) ?_ 0 s11 hP0
      intro c stt hc hP
      have hPd := hP
      obtain ⟨hpids, hQm, hKm, hGm, hpq, hpg, hpk, hpqg, hpkg, hmask, hldreg,
        hUQG, hUKG, hrQG, hrKG⟩ := hPd
      refine ⟨prep_bodySafeR R bounds Q K G QG KG s_qk_h DK BT BK scale s stt c
        hpq hpg hpk hpqg hpkg hmask (hbQ' c hc) (hbK' c hc) (hbG' c hc)
        (hbQG' c hc) (hbKG' c hc), ?_⟩
      obtain ⟨st', hstep, hP'⟩ := prepare_qg_kg_step Q K G QG KG s s_qk_h DK BT BK scale
        hQ_QG hQ_KG hK_QG hK_KG hG_QG hG_KG hQG_KG hBK hBT c hc stt hP
      exact ⟨st', by rw [prepBody_castFree]; exact hstep, hP'⟩

/-! ### The rounded Hoare triple (`hrun`) -/

set_option maxHeartbeats 8000000 in
/-- Termination, per-window values and the per-cell frame of the whole
prepare kernel under `execR R`, from an **arbitrary** launch state: the
exact `prep_preLoop` / `prepInv` stack runs verbatim (the prologue, the
`last_decay` load and the two-store loop body are cast-free, so `execR R`
collapses onto the exact stepper), extended with the per-cell memory frame
carried alongside `prepInv` (`prep_body_step_frame`). -/
private theorem prep_runR (R : RoundingModel) (Q K G QG KG : RegionName)
    (s_qk_h DK BT BK : Nat) (scale : ℝ)
    (hQ_QG : Q ≠ QG) (hQ_KG : Q ≠ KG) (hK_QG : K ≠ QG) (hK_KG : K ≠ KG)
    (hG_QG : G ≠ QG) (hG_KG : G ≠ KG) (hQG_KG : QG ≠ KG)
    (hBK : BK ≤ DK) (hBT : 0 < BT) (s₀ : BlockState) :
    ∃ sfin,
      execR R (prepare_qg_kg_surface Q K G QG KG s_qk_h DK BT BK scale).toAlgKernel s₀
        = some sfin
      ∧ (∀ (t : Fin BT) (i : Fin BK), active s₀ DK BK i →
          sfin.readMem QG (offset s₀ s_qk_h DK t.val BT BK i)
            = prepareQgClosed s₀ Q G s_qk_h DK BT BK t i scale)
      ∧ (∀ (t : Fin BT) (i : Fin BK), active s₀ DK BK i →
          sfin.readMem KG (offset s₀ s_qk_h DK t.val BT BK i)
            = prepareKgClosed s₀ K G s_qk_h DK BT BK t i)
      ∧ (∀ r oo,
          (r ≠ QG ∨ ∀ (t : Fin BT) (i : Fin BK), active s₀ DK BK i →
            oo ≠ offset s₀ s_qk_h DK t.val BT BK i) →
          (r ≠ KG ∨ ∀ (t : Fin BT) (i : Fin BK), active s₀ DK BK i →
            oo ≠ offset s₀ s_qk_h DK t.val BT BK i) →
          sfin.mem r oo = s₀.mem r oo) := by
  obtain ⟨s10, s11, hsp10, hld, hbh10, hc10, hk10, ho10, hP0, hmem11⟩ :=
    prep_preLoop Q K G QG KG s_qk_h DK BT BK scale hBT s₀
  -- the loop with the per-cell frame carried alongside `prepInv`
  obtain ⟨final, sfinal, hLoop, hfinal, hPfinal⟩ :=
    forRange_inv (idx := "_i") (start := 0) (stop := BT) (step := 1)
      (body := prepBody Q K G QG KG s_qk_h DK BT BK scale)
      (P := fun m stt => prepInv Q K G QG KG s₀ s_qk_h DK BT BK scale m stt
        ∧ ∀ r oo,
          (r ≠ QG ∨ ∀ (t : Fin BT) (i : Fin BK), active s₀ DK BK i →
            oo ≠ offset s₀ s_qk_h DK t.val BT BK i) →
          (r ≠ KG ∨ ∀ (t : Fin BT) (i : Fin BK), active s₀ DK BK i →
            oo ≠ offset s₀ s_qk_h DK t.val BT BK i) →
          stt.mem r oo = s₀.mem r oo)
      Nat.one_ne_zero
      ⟨hP0, fun r oo _ _ => by rw [hmem11]⟩
      (fun m stt hm hP => by
        obtain ⟨st', hstep, hP'⟩ := prepare_qg_kg_step Q K G QG KG s₀ s_qk_h DK BT BK scale
          hQ_QG hQ_KG hK_QG hK_KG hG_QG hG_KG hQG_KG hBK hBT m hm stt hP.1
        refine ⟨st', hstep, hP', ?_⟩
        intro r oo hcQG hcKG
        obtain ⟨hpids, hQm, hKm, hGm, hpq, hpg, hpk, hpqg, hpkg, hmask, hldreg,
          hUQG, hUKG, hrQG, hrKG⟩ := hP.1
        rw [prep_body_step_frame Q K G QG KG s_qk_h DK BT BK scale s₀ stt st' m
          hpqg hpkg hmask hstep r oo ?_ ?_]
        · exact hP.2 r oo hcQG hcKG
        · rcases hcQG with hne' | hno
          · exact Or.inl hne'
          · exact Or.inr fun i ha => hno ⟨m, hm⟩ i ha
        · rcases hcKG with hne' | hno
          · exact Or.inl hne'
          · exact Or.inr fun i ha => hno ⟨m, hm⟩ i ha)
  obtain ⟨⟨_, _, _, _, _, _, _, _, _, _, _, _, _, hreadQG, hreadKG⟩, hframe⟩ := hPfinal
  -- assemble the `execR` run through the cast-free collapses
  have hLdR : stepStmtR R (prepLdStmt G s_qk_h BT BK DK) s10 = some s11 := by
    rw [prepLd_castFree]; exact hld
  have hLoopR : stepStmtR R
      (Stmt.forRange "_i" 0 BT 1 (prepBody Q K G QG KG s_qk_h DK BT BK scale)) s11
      = some sfinal := by
    rw [stepStmtR_forRange,
      stepForRangeAuxR_castFree R _
        (prepBody_castFree R Q K G QG KG s_qk_h DK BT BK scale) "_i",
      ← stepForRangeAux.forRange_unfold]
    exact hLoop
  refine ⟨sfinal, ?_, ?_, ?_, ?_⟩
  · unfold execR
    rw [prep_body_decomp,
      stepStmtsR_append R (prepPre10 Q K G QG KG s_qk_h BT BK DK) _ s₀,
      prepPre10_castFree R Q K G QG KG s_qk_h BT BK DK s₀, hsp10, Option.bind_some,
      stepStmtsR_cons_some hLdR, stepStmtsR_cons_some hLoopR, stepStmtsR_nil]
  · intro t i ha
    have ht : t.val < final := lt_of_lt_of_le t.isLt hfinal
    rw [hreadQG t.val ht i t.isLt, if_pos ha]
  · intro t i ha
    have ht : t.val < final := lt_of_lt_of_le t.isLt hfinal
    rw [hreadKG t.val ht i t.isLt, if_pos ha]
  · exact hframe

/-! ### The headline -/

set_option maxHeartbeats 8000000 in
/-- **The `⊨[R]` streaming headline (wave-5 S3 grouped per-step emit
genre, 3-D grid).** For every rounding model `R`, the faithful
`prepare_qg_kg` surface implements, on its grouped
`StreamGroupedEmitMasked3DKernelIO` signature, the **ideal ℝ decay
scaling** of the streamed `q`/`k` rows: emitted window `(t, j)` of channel
`qg` holds `q[t,j]·exp2(g[t,j])·scale`, of channel `kg` holds
`k[t,j]·exp2(g[BT−1,j] − g[t,j])` — the spec `f` is exact real
arithmetic, and the `last_decay` row is the `g` stream's own step-`BT−1`
cells (no extra channel). The kernel has **zero rounding events** (all
loads, the `exp2` scalings and both per-step masked stores are at `.real`;
the erased `.to(...)` casts are `.real → .real`), so the skin's boundary
quantization degenerates: the readback contract's `R.round .real` is the
identity by the model's defining `round_real` — the ∀-`R` face holds via
the `RoundingModel` `.real` identity fields, not as a `.triv` special
case.

Layer map: the prologue, the unmasked `last_decay` load and the two-store
loop body are cast-free, so under `execR R` they collapse verbatim onto
the exact stepper and the proven `prepInv` invariant stack above
(`prepare_qg_kg_step` / `forRange_inv`, via `prepare_qg_kg_loop_drive`'s
prefix walk re-packaged as `prep_preLoop`) is reused unchanged; the
`⊨[R]` face adds the `TraceSafeR` walk (threading register pins across
the body's first store — `stepStmtR_store_real_regs`), the per-cell
memory frame (`prep_body_step_frame`, the `mem` twin of the step lemma),
and the per-channel stream-lane spec bridges
(`prepStreamSpec_qg_eq_closed` / `prepStreamSpec_kg_eq_closed`).

All hypotheses are truth-forced:

* `hQ_QG`/`hQ_KG`/`hK_QG`/`hK_KG`/`hG_QG`/`hG_KG` — inherited from the
  exact headlines `prepare_qg_kg_full_surface_{qg,kg}_closed_general`:
  the loop stores into `qg`/`kg` **between** its per-row re-reads of
  `q`/`k`/`g`, so the invariant's whole-input frames (and hence the
  closed forms over the *initial* input values) require the output
  buffers not to alias any input. The launch allocates all five tensors
  separately.
* `hQG_KG : QG ≠ KG` — also from the exact headlines: the two per-step
  stores land at identical offsets of their buffers; if `qg` and `kg`
  aliased, the second (qg) store would clobber the kg row just written.
* `hBK : BK ≤ DK` — row separation, as in the exact headlines: the
  row-`m` scatters must not collide with other rows' windows, which needs
  every lane index `< BK` to stay inside one `DK`-wide row. The launch
  sets `BK = min(DK, 64)`.
* `hBT : 0 < BT` — truth-forced twice over: (a) the pre-loop
  `last_decay` load needs a nonempty stream — its addresses are bounded
  by the `g` window bound *at step `BT − 1`*, and its row arithmetic
  `(i_c·BT + BT − 1)·DK` only matches the stream address at `BT ≥ 1`
  (`prep_ld_addr_eq`); (b) the spec's last-row index `⟨BT − 1, _⟩ : Fin
  BT` needs `Fin BT` inhabited (the spec's `dite` else-branch is
  unreachable under this hypothesis). The exact headlines carry the same
  constraint implicitly via `t_rel : Fin BT`, and the launch sets
  `BT = 16` (`prepare_qg_kg_step` also consumes `hBT` for the
  `last_decay` register pin).

Relation to the exact surface: the exact headlines above are retained
unchanged; this `⊨[R]` face restates both closed forms at once on the
grouped streaming emit skin, for every `R` (at the `.real` grid the two
faces carry the same exact cells). Both faces are kept per the
rounding-as-default doctrine. -/
specification prepare_qg_kg_io_correctness (R : RoundingModel)
    (Q K G QG KG : RegionName) (s_qk_h DK BT BK : Nat) (scale : ℝ)
    (hQ_QG : Q ≠ QG) (hQ_KG : Q ≠ KG) (hK_QG : K ≠ QG) (hK_KG : K ≠ KG)
    (hG_QG : G ≠ QG) (hG_KG : G ≠ KG) (hQG_KG : QG ≠ KG)
    (hBK : BK ≤ DK) (hBT : 0 < BT) :
    prepareQgKgKernelIO Q K G QG KG s_qk_h DK BT BK scale ⊨[R]
      fun _ _ _ xs o t j => prepStreamSpec BT BK scale xs o t j := by
  refine StreamGroupedEmitMasked3DKernelIO.ImplementsR.intro _ ?_ ?_ ?_ ?_
  · -- every output channel's buffer is in the declared allocation list
    intro o
    simp only [prepareQgKgKernelIO]
    match o with
    | ⟨0, _⟩ => show QG ∈ [Q, K, G, QG, KG]; simp
    | ⟨1, _⟩ => show KG ∈ [Q, K, G, QG, KG]; simp
    | ⟨n + 2, h2⟩ => exact absurd h2 (show ¬(n + 2 < 2) by omega)
  · exact prep_flattenOk Q K G QG KG s_qk_h DK BT BK scale
  · -- safety walk
    intro bounds s xs _hx hbr hbw
    simp only [prepareQgKgKernelIO] at hbr hbw ⊢
    refine prep_traceSafeR R bounds Q K G QG KG s_qk_h DK BT BK scale
      hQ_QG hQ_KG hK_QG hK_KG hG_QG hG_KG hQG_KG hBK hBT s ?_ ?_ ?_ ?_ ?_
    · exact fun t j hj => hbr ⟨0, by omega⟩ t j hj
    · exact fun t j hj => hbr ⟨1, by omega⟩ t j hj
    · exact fun t j hj => hbr ⟨2, by omega⟩ t j hj
    · exact fun t j hj => hbw ⟨0, by omega⟩ t j hj
    · exact fun t j hj => hbw ⟨1, by omega⟩ t j hj
  · -- the rounded Hoare triple
    intro s₀ xs _hundef hx
    simp only [prepareQgKgKernelIO] at hx ⊢
    obtain ⟨sfin, hexec, hqg, hkg, hframe⟩ :=
      prep_runR R Q K G QG KG s_qk_h DK BT BK scale
        hQ_QG hQ_KG hK_QG hK_KG hG_QG hG_KG hQG_KG hBK hBT s₀
    refine ⟨sfin, hexec, ?_, ?_⟩
    · intro o t j hj
      match o with
      | ⟨0, h0⟩ =>
          rw [BlockState.readMemAs_real, fwd_addr_eq]
          have hval : sfin.readMem QG (offset s₀ s_qk_h DK t.val BT BK j)
              = prepStreamSpec BT BK scale xs ⟨0, h0⟩ t j := by
            rw [hqg t j hj,
              ← prepStreamSpec_qg_eq_closed Q G s₀ s_qk_h DK BT BK scale xs h0
                (fun t' j' hj' => hx ⟨0, by omega⟩ t' j' hj')
                (fun t' j' hj' => hx ⟨2, by omega⟩ t' j' (Or.inl hj')) t j hj]
          simp only [RoundingModel.round_real_apply]
          exact congrArg some hval
      | ⟨1, h1⟩ =>
          rw [BlockState.readMemAs_real, fwd_addr_eq]
          have hval : sfin.readMem KG (offset s₀ s_qk_h DK t.val BT BK j)
              = prepStreamSpec BT BK scale xs ⟨1, h1⟩ t j := by
            rw [hkg t j hj,
              ← prepStreamSpec_kg_eq_closed K G s₀ s_qk_h DK BT BK scale xs hBT h1
                (fun t' j' hj' => hx ⟨1, by omega⟩ t' j' hj')
                (fun t' j' hj' => hx ⟨2, by omega⟩ t' j' hj') t j hj]
          simp only [RoundingModel.round_real_apply]
          exact congrArg some hval
      | ⟨n + 2, h2⟩ => exact absurd h2 (show ¬(n + 2 < 2) by omega)
    · intro r oo hcond
      refine hframe r oo ?_ ?_
      · by_cases hr : r = QG
        · refine Or.inr fun t i ha hoeq => ?_
          exact hcond ⟨0, by omega⟩ t i ha hr
            (hoeq.trans (fwd_addr_eq s₀ s_qk_h BT BK DK t.val i).symm)
        · exact Or.inl hr
      · by_cases hr : r = KG
        · refine Or.inr fun t i ha hoeq => ?_
          exact hcond ⟨1, by omega⟩ t i ha hr
            (hoeq.trans (fwd_addr_eq s₀ s_qk_h BT BK DK t.val i).symm)
        · exact Or.inl hr

end PrepareIOFace

/-! ## The `⊨[R]` streaming headline for `bwd_decay_global_cumsum`
(wave-6 S3 grouped per-step emit genre, 3-D grid — **first consumer of the
dead-write-tolerant flat bridge**)

Everything below is purely additive; the exact surfaces above (all three
kernels) and the `fwd_decay_cumsum` / `prepare_qg_kg` io sections are
untouched. This is the second consumer of the grouped multi-channel
per-step emit skin `StreamGroupedEmitMasked3DKernelIO`: seven input
streams over the shared row addressing (`g`/`dq_inner`/`dq_inter`/
`dk_inner`/`dk_inter`/`q`/`k` as channels `0`–`6`), three per-step emit
windows (`dq_inter`/`dk_inter`/`dg` as output channels `0`–`2`, the first
two **in place** over their input channels), all three stores inside the
reverse `range(BT-1,-1,-1)` loop.

**Why this headline cannot use the skin's `ImplementsR.intro`**: the loop
body ends in 8 `Op.ptrSub` self-decrements (`p_x -= DK`), and in the
*final* iteration the pointers sit at row `0` — at `pid = (0,·,0)` the
lane addresses are `< BK ≤ DK`, so the truncated `Nat` subtraction
genuinely underflows and `Stmt.TraceSafeR`'s per-state no-underflow
`ptrSub` clause is falsifiable: the kernel-wide `TraceSafeR` walk that
`ImplementsR.intro` demands is **unprovable**. Those 8 writes are dead
(the loop is the kernel's last statement and the registers are never read
again), so the `⊨[R]` triple is proved **directly**, mirroring `intro`'s
assembly but routing the flat leg through the dead-write-tolerant bridge
`FlatAlloc.execR_flatten_deadPtrTail` (strict commutation for the
prologue, the loop core and every non-final tail; observational agreement
`ObsAgree` instead of state equality across the final, underflowing
tail — which is all the memory-level readback and frame legs consume).

Rounding structure: as in the sibling sections this surface has **zero
rounding events** — every load, `exp2` combination and per-step masked
store is at `.real` (the `.to(tl.float32)` / `.to(*.dtype.element_ty)`
casts erase to `.real → .real`), so the body collapses verbatim onto the
exact stepper (`Rcast_real_real`) and the whole proven `bwdInvG`
invariant stack above (`bwd_decay_cumsum_step_general` /
`forRangeDyn_inv`) is reused unchanged. -/

section BwdIOFace

open scoped VeriTile.Triton.StreamGroupedEmitMasked3DKernelIO

set_option maxHeartbeats 4000000
set_option linter.unusedVariables false

/-! ### The lowered program: 15-stmt prologue + reverse loop = core ++ dead pointer tail -/

/-- The lowered reverse loop's stop bound `(BT-1)/1 + 1` (= `BT`), as one
named op (the raw tree of `bwd_body_decomp_general`). -/
private def bwdStopOp (BT : Nat) : Op TileDType.nat [] :=
  Op.add .nat Broadcast.nil
    (Op.div .nat Broadcast.nil
      (Op.sub .nat Broadcast.nil (Op.constNat BT) (Op.constNat 1)) (Op.constNat 1))
    (Op.constNat 1)

/-- The reverse loop body's 18-statement **core**: everything up to and
including the third masked store (`p_dg`). All memory traffic of an
iteration lives here; the core never touches a pointer register. -/
private def bwdCoreG (DK BT BK : Nat) : List Stmt :=
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
      (Op.ref .real [BK] "cum_grad_dg") (MaskOpt.mask (Op.ref .bool [BK] "mask")) ]

/-- The reverse loop body's 8-statement **dead pointer tail**: the eight
`p_x -= DK` self-decrements, register-only, never read after the final
iteration — exactly the `StmtList.PtrDecTail` shape the tolerant bridge
absorbs. -/
private def bwdTailG (DK BK : Nat) : List Stmt :=
  [ Stmt.assign .ptr [BK] "p_g"
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

/-- The proven loop body **is** core ++ tail. -/
private theorem bwdIterBodyG_decomp (DK BT BK : Nat) :
    bwdIterBodyG DK BT BK = bwdCoreG DK BT BK ++ bwdTailG DK BK := rfl

/-- The lowered backward body in the tolerant bridge's shape:
`prologue ++ [forRangeDyn … (core ++ tail)]` — the loop is the kernel's
last statement, so the final iteration's underflowing tail is dead. -/
private theorem bwd_body_decomp_io
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (s_qk_h DK BT BK : Nat) :
    (bwd_decay_global_cumsum_surface DQInner DQInter DKInner DKInter Q K G DG
        s_qk_h DK BT BK).toAlgKernel.body
      = bwdPrologueG DQInner DQInter DKInner DKInter Q K G DG s_qk_h DK BT BK
        ++ [Stmt.forRangeDyn "__rev_t" (Op.constNat 0) (bwdStopOp BT)
              (Op.constNat 1) (bwdCoreG DK BT BK ++ bwdTailG DK BK)] := by
  rw [bwd_body_decomp_general, bwd_prologue_take_general, ← bwdIterBodyG_decomp]
  rfl

/-- The tail is a `PtrDecTail`: every statement is a pointer
self-decrement by the constant `DK`. -/
private theorem bwdTailG_ptrDecTail (DK BK : Nat) :
    StmtList.PtrDecTail (bwdTailG DK BK) := by
  intro st hst
  simp only [bwdTailG, List.mem_cons, List.not_mem_nil, or_false] at hst
  rcases hst with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    exact ⟨[BK], _, Broadcast.scalarR, DK, rfl⟩

/-! ### Covered-fragment membership (`FlattenOkR`) and cast-free collapses -/

/-- The 15-statement prologue sits in the `R`-bridge fragment. -/
private theorem bwd_prologue_flattenOkR
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (s_qk_h DK BT BK : Nat) :
    StmtList.FlattenOkR
      (bwdPrologueG DQInner DQInter DKInner DKInter Q K G DG s_qk_h DK BT BK) := by
  simp [bwdPrologueG, StmtList.FlattenOkR, Stmt.FlattenOkR, Op.FlattenOkR.eq_def]

/-- The 18-statement core sits in the `R`-bridge fragment. -/
private theorem bwd_core_flattenOkR (DK BT BK : Nat) :
    StmtList.FlattenOkR (bwdCoreG DK BT BK) := by
  simp [bwdCoreG, StmtList.FlattenOkR, Stmt.FlattenOkR, Op.FlattenOkR.eq_def]

/-- The stop bound op sits in the `R`-bridge fragment. -/
private theorem bwd_stopOp_flattenOkR (BT : Nat) : (bwdStopOp BT).FlattenOkR := by
  simp [bwdStopOp, Op.FlattenOkR.eq_def]

/-- The prologue is cast-free: it steps identically under `stepStmtsR R`. -/
private theorem bwdPrologue_castFree (R : RoundingModel)
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (s_qk_h DK BT BK : Nat) (t : BlockState) :
    stepStmtsR R (bwdPrologueG DQInner DQInter DKInner DKInter Q K G DG s_qk_h DK BT BK) t
      = stepStmts (bwdPrologueG DQInner DQInter DKInner DKInter Q K G DG s_qk_h DK BT BK) t := by
  simp only [bwdPrologueG, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def]
  rfl

/-- The whole loop body (core ++ tail) is cast-free **including its three
in-loop masked `.real` stores and the `last_g` conditional capture**: the
masked `.real` loads with erased fp32 casts are exact under every `R`, the
`exp2` combinations are real arithmetic, `stepStmtR` delegates
`.real`-typed stores to the exact `writeMemTyped`, and the `ifThen` guard
is nat arithmetic — so the body steps identically under `stepStmtsR R`
and the exact `bwdInvG` stack transports to `execR`. -/
private theorem bwdIterBody_castFree (R : RoundingModel) (DK BT BK : Nat)
    (t : BlockState) :
    stepStmtsR R (bwdIterBodyG DK BT BK) t = stepStmts (bwdIterBodyG DK BT BK) t := by
  simp only [bwdIterBodyG, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def, Rcast_real_real, BlockState.writeMemTypedR]
  rfl

/-- The core alone is cast-free (the mid-body state of the tolerant
bridge's tail obligation is reached under `stepStmtsR R`; the pin walk
below rides the exact stepper through this collapse). -/
private theorem bwdCore_castFree (R : RoundingModel) (DK BT BK : Nat)
    (t : BlockState) :
    stepStmtsR R (bwdCoreG DK BT BK) t = stepStmts (bwdCoreG DK BT BK) t := by
  simp only [bwdCoreG, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def, Rcast_real_real, BlockState.writeMemTypedR]
  rfl

/-- `evalOpR` of the stop bound: `(BT-1)/1 + 1 = BT` (nat arithmetic —
`R` never enters; needs `0 < BT`). -/
private theorem bwd_stopOpR_eval (R : RoundingModel) (c : BlockState)
    (BT : Nat) (hBT : 0 < BT) :
    evalOpR R (bwdStopOp BT) c = some (Tile.scalar BT) := by
  have h : evalOpR R (bwdStopOp BT) c = evalOp (bwdStopOp BT) c := by
    simp only [bwdStopOp, evalOpR.eq_def, evalOp.eq_def]
  rw [h]
  exact bwd_stopOp_eval c BT hBT

/-- `evalOpR` of a `constNat` (R-independent). -/
private theorem bwd_evalOpR_constNat (R : RoundingModel) (n : Nat) (s : BlockState) :
    evalOpR R (Op.constNat n) s = some (Tile.scalar n) := by
  simp [evalOpR]

/-! ### IO signature -/

/-- **Streaming IO signature** of `bwd_decay_global_cumsum` on the grouped
multi-channel per-step emit skin (S3: three in-loop stores, two of them
**in place**). Loop counter `m` processes PYTHON row `t = BT − 1 − m`;
the io is indexed by the *row* `t`, so step `t` of the signature is the
`(BT − 1 − t)`-th executed iteration. Step `t` reads the `BK`-lane rows
of the seven input channels (`0` = `g`, `1` = `dq_inner`, `2` =
`dq_inter` (old contents), `3` = `dk_inner`, `4` = `dk_inter` (old
contents), `5` = `q`, `6` = `k`) and stores the `BK`-lane
`dq_inter`/`dk_inter`/`dg` windows (output channels `0`/`1`/`2`) at the
**`.real`** grid (`outDType` default — the store's
`.to(p_dg.dtype.element_ty)` cast erases to `.real`).

**In-place channels**: output channels `0`/`1` name the same buffers as
input channels `2`/`4` — the skin's decoupled `bufs` allocation list
carries each region exactly once. Pinning the *old* contents on the
launch state is sound because the reverse loop visits each row exactly
once and reads its `dq_inter`/`dk_inter` cells *before* rewriting them in
the same iteration, so every read of those channels observes the launch
value (the invariant's untouched-rows conjunct `hDQIunt`/`hDKIunt` is
precisely this argument).

All ten windows share the surface's effective row address, transcribed
verbatim (`pid₀ = i_k`, `pid₁ = i_c`, `pid₂ = i_bh`; the surface's
pointers start at row `BT−1` and step `-DK`, so row `t`'s address is the
base plus `t·DK`):

`i_bh·s_qk_h + i_c·BT·DK + i_k·BK + j + t·DK`.

All masks are the kernel's single, `t`-independent bound mask
`i_k·BK + j < DK` — the `last_g` capture needs **no widening**: it is the
`g` stream's own step-`BT−1` row, loaded (masked, like every `g` row) in
the first executed iteration. -/
def bwdDecayCumsumKernelIO
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (s_qk_h DK BT BK : Nat) : StreamGroupedEmitMasked3DKernelIO where
  kernel := bwd_decay_global_cumsum_surface DQInner DQInter DKInner DKInter Q K G DG
    s_qk_h DK BT BK
  nIn := 7
  nOut := 3
  bufs := [DQInner, DQInter, DKInner, DKInter, Q, K, G, DG]
  inp := fun i => match i with
    | ⟨0, _⟩ => G
    | ⟨1, _⟩ => DQInner
    | ⟨2, _⟩ => DQInter
    | ⟨3, _⟩ => DKInner
    | ⟨4, _⟩ => DKInter
    | ⟨5, _⟩ => Q
    | ⟨6, _⟩ => K
    | ⟨n + 7, h⟩ => absurd h (by omega)
  out := fun o => match o with
    | ⟨0, _⟩ => DQInter
    | ⟨1, _⟩ => DKInter
    | ⟨2, _⟩ => DG
    | ⟨n + 3, h⟩ => absurd h (by omega)
  T := BT
  B := BK
  read := fun _ p₀ p₁ p₂ t j =>
    p₂ * s_qk_h + p₁ * BT * DK + p₀ * BK + j.val + t.val * DK
  readMask := fun _ p₀ _ _ _ j => p₀ * BK + j.val < DK
  write := fun _ p₀ p₁ p₂ t j =>
    p₂ * s_qk_h + p₁ * BT * DK + p₀ * BK + j.val + t.val * DK
  writeMask := fun _ p₀ _ _ _ j => p₀ * BK + j.val < DK

/-! ### The stream-level spec -/

/-- Per-row `dq` stream value: `dq_inner + dq_inter_old · exp2(g)` (with
`exp2(x) = exp(x·log 2)`) — channel `0`'s clause and the `dg` summand's
first factor. -/
private noncomputable def bwdDqStream (BT BK : Nat)
    (xs : Fin 7 → Fin BT → Fin BK → ℝ) (t : Fin BT) (j : Fin BK) : ℝ :=
  xs (⟨1, by omega⟩ : Fin 7) t j
    + xs (⟨2, by omega⟩ : Fin 7) t j
      * Real.exp (xs (⟨0, by omega⟩ : Fin 7) t j * Real.log 2)

/-- Per-row `dk` stream value:
`dk_inner + dk_inter_old · exp2(last_g − g)` where `last_g` is the `g`
stream's step-`BT−1` row (the first processed row — no extra channel).
The `dite` keeps the definition total in `BT`; its `else` branch is
unreachable under the headline's `hBT : 0 < BT`. -/
private noncomputable def bwdDkStream (BT BK : Nat)
    (xs : Fin 7 → Fin BT → Fin BK → ℝ) (t : Fin BT) (j : Fin BK) : ℝ :=
  if h : 0 < BT then
    xs (⟨3, by omega⟩ : Fin 7) t j
      + xs (⟨4, by omega⟩ : Fin 7) t j
        * Real.exp
            ((xs (⟨0, by omega⟩ : Fin 7) ⟨BT - 1, Nat.sub_lt h Nat.one_pos⟩ j
                - xs (⟨0, by omega⟩ : Fin 7) t j) * Real.log 2)
  else 0

/-- Per-row `dg` stream value: the reverse cumulative sum written at row
`t` is the **suffix sum** `Σ_{u = t}^{BT−1} (dq[u]·q[u] − dk[u]·k[u])`
over the per-row `dq`/`dk` values above (the executed loop folds rows
`BT−1, …, t` before storing row `t`). -/
private noncomputable def bwdDgStream (BT BK : Nat)
    (xs : Fin 7 → Fin BT → Fin BK → ℝ) (t : Fin BT) (j : Fin BK) : ℝ :=
  ∑ d : Fin (BT - t.val),
    (bwdDqStream BT BK xs ⟨t.val + d.val, by omega⟩ j
        * xs (⟨5, by omega⟩ : Fin 7) ⟨t.val + d.val, by omega⟩ j
      - bwdDkStream BT BK xs ⟨t.val + d.val, by omega⟩ j
        * xs (⟨6, by omega⟩ : Fin 7) ⟨t.val + d.val, by omega⟩ j)

/-- The stream-level `bwd_decay_global_cumsum` spec (the genre's *emit* +
*suffix-scan* shape, one clause per output channel): window `(t, j)` of
channel `0` (`dq_inter`) holds `dq_inner + dq_inter_old·exp2(g)`, of
channel `1` (`dk_inter`) holds `dk_inner + dk_inter_old·exp2(last_g − g)`
(with `last_g = g[BT−1]`, the `g` stream's own last row), of channel `2`
(`dg`) holds the suffix sum `Σ_{u ≥ t} (dq[u]·q[u] − dk[u]·k[u])` — the
exact spellings of the proven closed forms `bwdDQInterClosed` /
`bwdDKInterClosed` / `bwdDGClosed`, re-indexed to the curried streams.
The channel `match` lives in this single shared def (inline per-site
matchers block cross-declaration `exact`). -/
noncomputable def bwdDecayStreamSpec (BT BK : Nat)
    (xs : Fin 7 → Fin BT → Fin BK → ℝ) (o : Fin 3) (t : Fin BT) (j : Fin BK) : ℝ :=
  match o with
  | ⟨0, _⟩ => bwdDqStream BT BK xs t j
  | ⟨1, _⟩ => bwdDkStream BT BK xs t j
  | ⟨2, _⟩ => bwdDgStream BT BK xs t j
  | ⟨n + 3, h3⟩ => absurd h3 (by omega)

/-- Per-lane `dq_inter` spec bridge: at a write-active window `(t, j)`
the stream `dq` value **is** the exact stack's closed form
`bwdDQInterClosed` — every stream cell it mentions is mask-pinned (the
masks are `t`-independent and lane `j` is active). -/
private theorem bwdDqStream_eq_closed (DQInner DQInter G : RegionName)
    (s₀ : BlockState) (s_qk_h DK BT BK : Nat) (xs : Fin 7 → Fin BT → Fin BK → ℝ)
    (hxG : ∀ (t : Fin BT) (j : Fin BK), s₀.pids 0 * BK + j.val < DK →
      s₀.readMem G
          (s₀.pids 2 * s_qk_h + s₀.pids 1 * BT * DK + s₀.pids 0 * BK + j.val + t.val * DK)
        = xs (⟨0, by omega⟩ : Fin 7) t j)
    (hxDQi : ∀ (t : Fin BT) (j : Fin BK), s₀.pids 0 * BK + j.val < DK →
      s₀.readMem DQInner
          (s₀.pids 2 * s_qk_h + s₀.pids 1 * BT * DK + s₀.pids 0 * BK + j.val + t.val * DK)
        = xs (⟨1, by omega⟩ : Fin 7) t j)
    (hxDQt : ∀ (t : Fin BT) (j : Fin BK), s₀.pids 0 * BK + j.val < DK →
      s₀.readMem DQInter
          (s₀.pids 2 * s_qk_h + s₀.pids 1 * BT * DK + s₀.pids 0 * BK + j.val + t.val * DK)
        = xs (⟨2, by omega⟩ : Fin 7) t j)
    (t : Fin BT) (j : Fin BK) (hj : active s₀ DK BK j) :
    bwdDqStream BT BK xs t j
      = bwdDQInterClosed s₀ DQInner DQInter G s_qk_h DK BT BK t j := by
  have hg := hxG t j hj
  have h1 := hxDQi t j hj
  have h2 := hxDQt t j hj
  rw [fwd_addr_eq] at hg h1 h2
  show xs (⟨1, by omega⟩ : Fin 7) t j
      + xs (⟨2, by omega⟩ : Fin 7) t j
        * Real.exp (xs (⟨0, by omega⟩ : Fin 7) t j * Real.log 2) = _
  rw [← hg, ← h1, ← h2]
  simp only [bwdDQInterClosed]

/-- Per-lane `dk_inter` spec bridge: at a write-active window `(t, j)`
the stream `dk` value **is** `bwdDKInterClosed` — the `last_g` cell
`xs 0 ⟨BT−1, _⟩ j` is the `g` stream's own last row, pinned at the active
lane because the `g` read mask is `t`-independent. -/
private theorem bwdDkStream_eq_closed (DKInner DKInter G : RegionName)
    (s₀ : BlockState) (s_qk_h DK BT BK : Nat) (xs : Fin 7 → Fin BT → Fin BK → ℝ)
    (hBT : 0 < BT)
    (hxG : ∀ (t : Fin BT) (j : Fin BK), s₀.pids 0 * BK + j.val < DK →
      s₀.readMem G
          (s₀.pids 2 * s_qk_h + s₀.pids 1 * BT * DK + s₀.pids 0 * BK + j.val + t.val * DK)
        = xs (⟨0, by omega⟩ : Fin 7) t j)
    (hxDKi : ∀ (t : Fin BT) (j : Fin BK), s₀.pids 0 * BK + j.val < DK →
      s₀.readMem DKInner
          (s₀.pids 2 * s_qk_h + s₀.pids 1 * BT * DK + s₀.pids 0 * BK + j.val + t.val * DK)
        = xs (⟨3, by omega⟩ : Fin 7) t j)
    (hxDKt : ∀ (t : Fin BT) (j : Fin BK), s₀.pids 0 * BK + j.val < DK →
      s₀.readMem DKInter
          (s₀.pids 2 * s_qk_h + s₀.pids 1 * BT * DK + s₀.pids 0 * BK + j.val + t.val * DK)
        = xs (⟨4, by omega⟩ : Fin 7) t j)
    (t : Fin BT) (j : Fin BK) (hj : active s₀ DK BK j) :
    bwdDkStream BT BK xs t j
      = bwdDKInterClosed s₀ DKInner DKInter G s_qk_h DK BT BK t j := by
  have hg := hxG t j hj
  have h3 := hxDKi t j hj
  have h4 := hxDKt t j hj
  have hglast := hxG ⟨BT - 1, Nat.sub_lt hBT Nat.one_pos⟩ j hj
  rw [fwd_addr_eq] at hg h3 h4 hglast
  show (if h : 0 < BT then
      xs (⟨3, by omega⟩ : Fin 7) t j
        + xs (⟨4, by omega⟩ : Fin 7) t j
          * Real.exp
              ((xs (⟨0, by omega⟩ : Fin 7) ⟨BT - 1, Nat.sub_lt h Nat.one_pos⟩ j
                  - xs (⟨0, by omega⟩ : Fin 7) t j) * Real.log 2)
    else 0) = _
  rw [dif_pos hBT, ← hg, ← h3, ← h4, ← hglast]
  simp only [bwdDKInterClosed]

/-- Per-lane `dg` spec bridge: at a write-active window `(t, j)` the
stream suffix sum **is** `bwdDGClosed` — every summand row `t + d` reads
mask-pinned cells (`t`-independent masks), and the per-row `dq`/`dk`
factors bridge by the two lemmas above. -/
private theorem bwdDgStream_eq_closed
    (DQInner DQInter DKInner DKInter Q K G : RegionName)
    (s₀ : BlockState) (s_qk_h DK BT BK : Nat) (xs : Fin 7 → Fin BT → Fin BK → ℝ)
    (hBT : 0 < BT)
    (hxG : ∀ (t : Fin BT) (j : Fin BK), s₀.pids 0 * BK + j.val < DK →
      s₀.readMem G
          (s₀.pids 2 * s_qk_h + s₀.pids 1 * BT * DK + s₀.pids 0 * BK + j.val + t.val * DK)
        = xs (⟨0, by omega⟩ : Fin 7) t j)
    (hxDQi : ∀ (t : Fin BT) (j : Fin BK), s₀.pids 0 * BK + j.val < DK →
      s₀.readMem DQInner
          (s₀.pids 2 * s_qk_h + s₀.pids 1 * BT * DK + s₀.pids 0 * BK + j.val + t.val * DK)
        = xs (⟨1, by omega⟩ : Fin 7) t j)
    (hxDQt : ∀ (t : Fin BT) (j : Fin BK), s₀.pids 0 * BK + j.val < DK →
      s₀.readMem DQInter
          (s₀.pids 2 * s_qk_h + s₀.pids 1 * BT * DK + s₀.pids 0 * BK + j.val + t.val * DK)
        = xs (⟨2, by omega⟩ : Fin 7) t j)
    (hxDKi : ∀ (t : Fin BT) (j : Fin BK), s₀.pids 0 * BK + j.val < DK →
      s₀.readMem DKInner
          (s₀.pids 2 * s_qk_h + s₀.pids 1 * BT * DK + s₀.pids 0 * BK + j.val + t.val * DK)
        = xs (⟨3, by omega⟩ : Fin 7) t j)
    (hxDKt : ∀ (t : Fin BT) (j : Fin BK), s₀.pids 0 * BK + j.val < DK →
      s₀.readMem DKInter
          (s₀.pids 2 * s_qk_h + s₀.pids 1 * BT * DK + s₀.pids 0 * BK + j.val + t.val * DK)
        = xs (⟨4, by omega⟩ : Fin 7) t j)
    (hxQ : ∀ (t : Fin BT) (j : Fin BK), s₀.pids 0 * BK + j.val < DK →
      s₀.readMem Q
          (s₀.pids 2 * s_qk_h + s₀.pids 1 * BT * DK + s₀.pids 0 * BK + j.val + t.val * DK)
        = xs (⟨5, by omega⟩ : Fin 7) t j)
    (hxK : ∀ (t : Fin BT) (j : Fin BK), s₀.pids 0 * BK + j.val < DK →
      s₀.readMem K
          (s₀.pids 2 * s_qk_h + s₀.pids 1 * BT * DK + s₀.pids 0 * BK + j.val + t.val * DK)
        = xs (⟨6, by omega⟩ : Fin 7) t j)
    (t : Fin BT) (j : Fin BK) (hj : active s₀ DK BK j) :
    bwdDgStream BT BK xs t j
      = bwdDGClosed s₀ DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK t j := by
  simp only [bwdDgStream, bwdDGClosed]
  refine Finset.sum_congr rfl fun d _ => ?_
  have hq := hxQ ⟨t.val + d.val, by omega⟩ j hj
  have hk := hxK ⟨t.val + d.val, by omega⟩ j hj
  rw [fwd_addr_eq] at hq hk
  simp only [bwdDGSummand]
  rw [bwdDqStream_eq_closed DQInner DQInter G s₀ s_qk_h DK BT BK xs hxG hxDQi hxDQt
      ⟨t.val + d.val, by omega⟩ j hj,
    bwdDkStream_eq_closed DKInner DKInter G s₀ s_qk_h DK BT BK xs hBT hxG hxDKi hxDKt
      ⟨t.val + d.val, by omega⟩ j hj,
    ← hq, ← hk]

/-! ### Stepping inversions (exact and `R`-side)

The walks below thread successor states through the 26-statement body.
Assigns invert to `setReg`s with opaque values (`stepStmts_cons_assign_inv`
above / its `R` mirror here), `.real` masked stores preserve the register
file (`stepStmtR_store_real_regs` above), and the `last_g` `ifThen` gets a
dedicated single-assign inversion on both sides. -/

/-- Exact-side inversion of a leading `ifThen [assign]` in a `stepStmts`
chain: whatever the guard decides, memory survives and every register
except the assign target survives. -/
private theorem stepStmts_cons_ifThen_assign_inv {cond : Op TileDType.bool []}
    {dt : TileDType} {sh : TileShape} {nm : RegName} {e : Op dt sh}
    {rest : List Stmt} {s s' : BlockState}
    (h : stepStmts (Stmt.ifThen cond [Stmt.assign dt sh nm e] :: rest) s = some s') :
    ∃ s₂, stepStmts rest s₂ = some s' ∧ s₂.mem = s.mem ∧
      ∀ (d' : TileDType) (sh' : TileShape) (n' : RegName), n' ≠ nm →
        s₂.regs d' sh' n' = s.regs d' sh' n' := by
  cases hst : stepStmt (Stmt.ifThen cond [Stmt.assign dt sh nm e]) s with
  | none => simp [stepStmts, hst] at h
  | some s₂ =>
      rw [stepStmts.cons_some hst] at h
      refine ⟨s₂, h, ?_⟩
      simp only [stepStmt] at hst
      cases hc : evalOp cond s with
      | none => rw [hc] at hst; exact absurd hst (by simp)
      | some vc =>
          rw [hc] at hst
          replace hst : (if vc.data PUnit.unit = Bool.true then
              stepStmts [Stmt.assign dt sh nm e] s else some s) = some s₂ := hst
          by_cases hb : vc.data PUnit.unit = Bool.true
          · rw [if_pos hb] at hst
            obtain ⟨v, hv, hnil⟩ := stepStmts_cons_assign_inv hst
            rw [stepStmts.nil] at hnil
            obtain rfl := Option.some.inj hnil
            exact ⟨rfl, fun d' sh' n' hne => by
              rw [BlockState.setReg_ne_name (h := hne)]⟩
          · rw [if_neg hb] at hst
            obtain rfl := Option.some.inj hst
            exact ⟨rfl, fun _ _ _ _ => rfl⟩

/-- `R`-side cons inversion: a successful list step factors through a
successful head step. -/
private theorem stepStmtsR_cons_inv {R : RoundingModel} {st : Stmt}
    {rest : List Stmt} {s s' : BlockState}
    (h : stepStmtsR R (st :: rest) s = some s') :
    ∃ s₂, stepStmtR R st s = some s₂ ∧ stepStmtsR R rest s₂ = some s' := by
  simp only [stepStmtsR] at h
  cases hst : stepStmtR R st s with
  | none => rw [hst] at h; exact absurd h (by simp)
  | some s₂ =>
      rw [hst] at h
      exact ⟨s₂, rfl, h⟩

/-- `R`-side mirror of `stepStmts_cons_assign_inv`. -/
private theorem stepStmtsR_cons_assign_inv {R : RoundingModel} {dt : TileDType}
    {sh : TileShape} {nm : RegName} {e : Op dt sh} {rest : List Stmt}
    {s s' : BlockState}
    (h : stepStmtsR R (Stmt.assign dt sh nm e :: rest) s = some s') :
    ∃ v, evalOpR R e s = some v ∧
      stepStmtsR R rest (s.setReg nm dt sh v) = some s' := by
  obtain ⟨s₂, hst, h⟩ := stepStmtsR_cons_inv h
  obtain ⟨v, hv, rfl⟩ := stepStmtR_assign_inv hst
  exact ⟨v, hv, h⟩

/-- `R`-side cons inversion of a `.real` masked ptr store: the successor
carries the same register file. -/
private theorem stepStmtsR_cons_store_regs {R : RoundingModel} {sh : TileShape}
    {pe : Op TileDType.ptr sh} {ve : Op TileDType.real sh}
    {me : Op TileDType.bool sh} {rest : List Stmt} {s s' : BlockState}
    (h : stepStmtsR R (Stmt.store TileDType.real sh (MemAccess.ptr pe) ve
        (MaskOpt.mask me) :: rest) s = some s') :
    ∃ s₂, stepStmtsR R rest s₂ = some s' ∧ s₂.regs = s.regs := by
  obtain ⟨s₂, hst, h⟩ := stepStmtsR_cons_inv h
  exact ⟨s₂, h, stepStmtR_store_real_regs hst⟩

/-- `R`-side register effect of the `last_g` `ifThen`: every register
except the assign target survives. -/
private theorem stepStmtR_ifThen_assign_regs {R : RoundingModel}
    {cond : Op TileDType.bool []} {dt : TileDType} {sh : TileShape}
    {nm : RegName} {e : Op dt sh} {s s' : BlockState}
    (h : stepStmtR R (Stmt.ifThen cond [Stmt.assign dt sh nm e]) s = some s') :
    ∀ (d' : TileDType) (sh' : TileShape) (n' : RegName), n' ≠ nm →
      s'.regs d' sh' n' = s.regs d' sh' n' := by
  simp only [stepStmtR] at h
  cases hc : evalOpR R cond s with
  | none => rw [hc] at h; exact absurd h (by simp)
  | some vc =>
      rw [hc] at h
      replace h : (if vc.data PUnit.unit = Bool.true then
          stepStmtsR R [Stmt.assign dt sh nm e] s else some s) = some s' := h
      by_cases hb : vc.data PUnit.unit = Bool.true
      · rw [if_pos hb] at h
        obtain ⟨v, hv, hnil⟩ := stepStmtsR_cons_assign_inv h
        rw [stepStmtsR_nil] at hnil
        obtain rfl := Option.some.inj hnil
        exact fun d' sh' n' hne => by rw [BlockState.setReg_ne_name (h := hne)]
      · rw [if_neg hb] at h
        obtain rfl := Option.some.inj h
        exact fun _ _ _ _ => rfl

/-! ### Cell-level memory frame of one loop iteration -/

set_option maxHeartbeats 8000000 in
/-- **Cell-level frame of one reverse-loop iteration** (the `mem` twin of
`bwd_decay_cumsum_step_general`, same walk with opaque register values):
from the `bwdInvG` `p_dq_inter`/`p_dk_inter`/`p_dg`/`mask` pins (row-`rr`
form), one three-store body iteration leaves every cell off the row-`rr`
write windows of **all three** output buffers untouched. The 8-statement
pointer tail is register-only. -/
private theorem bwd_body_step_frame
    (DQInter DKInter DG : RegionName) (s sc s' : BlockState)
    (s_qk_h DK BT BK rr m : Nat)
    (hpdqt : sc.regs .ptr [BK] "p_dq_inter"
      = some (bwdPtrTileG s DQInter s_qk_h DK BT BK rr))
    (hpdkt : sc.regs .ptr [BK] "p_dk_inter"
      = some (bwdPtrTileG s DKInter s_qk_h DK BT BK rr))
    (hpdg : sc.regs .ptr [BK] "p_dg"
      = some (bwdPtrTileG s DG s_qk_h DK BT BK rr))
    (hmask : sc.regs .bool [BK] "mask" = some (bwdMaskTileG s DK BK))
    (hstep : stepStmts (bwdIterBodyG DK BT BK)
      (sc.setReg "__rev_t" .nat [] (Tile.scalar m)) = some s')
    (r : RegionName) (oo : Nat)
    (hcDQ : r ≠ DQInter ∨ ∀ i : Fin BK, active s DK BK i →
      oo ≠ offset s s_qk_h DK rr BT BK i)
    (hcDK : r ≠ DKInter ∨ ∀ i : Fin BK, active s DK BK i →
      oo ≠ offset s s_qk_h DK rr BT BK i)
    (hcDG : r ≠ DG ∨ ∀ i : Fin BK, active s DK BK i →
      oo ≠ offset s s_qk_h DK rr BT BK i) :
    s'.mem r oo = sc.mem r oo := by
  set sloop := sc.setReg "__rev_t" .nat [] (Tile.scalar m) with hsloop
  unfold bwdIterBodyG at hstep
  -- 0: `t` assign, 1: `g_val` load
  obtain ⟨v0, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  obtain ⟨v1, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  -- 2: the `last_g` capture (memory and non-`last_g` registers survive)
  obtain ⟨s₂, hstep, hmem₂, hregs₂⟩ := stepStmts_cons_ifThen_assign_inv hstep
  -- 3–6: `dq1`/`dq2`/`dq2`/`dq` assigns
  obtain ⟨v3, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  obtain ⟨v4, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  obtain ⟨v5, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  obtain ⟨v6, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  set s6 := (((s₂.setReg "dq1" .real [BK] v3).setReg "dq2" .real [BK] v4).setReg
    "dq2" .real [BK] v5).setReg "dq" .real [BK] v6 with hs6
  have hdq6 : s6.regs .real [BK] "dq" = some v6 := by
    rw [hs6]; exact BlockState.setReg_same _ _ _ _ _
  have hpdqt6 : s6.regs .ptr [BK] "p_dq_inter"
      = some (bwdPtrTileG s DQInter s_qk_h DK BT BK rr) := by
    rw [hs6, BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      hregs₂ _ _ _ (by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide), hsloop,
      BlockState.setReg_ne_name (h := by decide)]
    exact hpdqt
  have hmask6 : s6.regs .bool [BK] "mask" = some (bwdMaskTileG s DK BK) := by
    rw [hs6, BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      hregs₂ _ _ _ (by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide), hsloop,
      BlockState.setReg_ne_name (h := by decide)]
    exact hmask
  -- 7: the `dq_inter` store
  set vfun1 : TileIndex [BK] → ℝ := fun k => (v6.data k).unbotD 0 with hvfun1
  set sst1 := (TileShape.allIndices [BK]).foldl
    (fun (acc : BlockState) k =>
      if active s DK BK k.1 then
        acc.writeMem DQInter (offset s s_qk_h DK rr BT BK k.1) (vfun1 k)
      else acc) s6 with hsst1
  have hstore1 : stepStmt (Stmt.store TileDType.real [BK]
      (MemAccess.ptr (Op.ref TileDType.ptr [BK] "p_dq_inter"))
      (Op.ref TileDType.real [BK] "dq")
      (MaskOpt.mask (Op.ref TileDType.bool [BK] "mask"))) s6 = some sst1 := by
    simp only [stepStmt, evalOp, Option.bind, Option.map, hdq6, hpdqt6, hmask6]
    apply congrArg some
    rw [hsst1]
    congr 1
    funext acc k
    simp only [bwdPtrTileG, bwdMaskTileG]
    by_cases ha : active s DK BK k.1
    · rw [if_pos (by simpa using ha), if_pos ha, BlockState.writeMemTyped_real]
      rfl
    · rw [if_neg (by simpa using ha), if_neg ha]
  rw [stepStmts.cons_some hstore1] at hstep
  -- 8–11: `dk1`/`dk2`/`dk2`/`dk` assigns
  obtain ⟨w1, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  obtain ⟨w2, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  obtain ⟨w3, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  obtain ⟨w4, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  set s11 := (((sst1.setReg "dk1" .real [BK] w1).setReg "dk2" .real [BK] w2).setReg
    "dk2" .real [BK] w3).setReg "dk" .real [BK] w4 with hs11
  have hdk11 : s11.regs .real [BK] "dk" = some w4 := by
    rw [hs11]; exact BlockState.setReg_same _ _ _ _ _
  have hpdkt11 : s11.regs .ptr [BK] "p_dk_inter"
      = some (bwdPtrTileG s DKInter s_qk_h DK BT BK rr) := by
    rw [hs11, BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      hsst1, BlockState.foldl_writeMem_prop_masked_regs,
      hs6, BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      hregs₂ _ _ _ (by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide), hsloop,
      BlockState.setReg_ne_name (h := by decide)]
    exact hpdkt
  have hmask11 : s11.regs .bool [BK] "mask" = some (bwdMaskTileG s DK BK) := by
    rw [hs11, BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      hsst1, BlockState.foldl_writeMem_prop_masked_regs]
    exact hmask6
  -- 12: the `dk_inter` store
  set vfun2 : TileIndex [BK] → ℝ := fun k => (w4.data k).unbotD 0 with hvfun2
  set sst2 := (TileShape.allIndices [BK]).foldl
    (fun (acc : BlockState) k =>
      if active s DK BK k.1 then
        acc.writeMem DKInter (offset s s_qk_h DK rr BT BK k.1) (vfun2 k)
      else acc) s11 with hsst2
  have hstore2 : stepStmt (Stmt.store TileDType.real [BK]
      (MemAccess.ptr (Op.ref TileDType.ptr [BK] "p_dk_inter"))
      (Op.ref TileDType.real [BK] "dk")
      (MaskOpt.mask (Op.ref TileDType.bool [BK] "mask"))) s11 = some sst2 := by
    simp only [stepStmt, evalOp, Option.bind, Option.map, hdk11, hpdkt11, hmask11]
    apply congrArg some
    rw [hsst2]
    congr 1
    funext acc k
    simp only [bwdPtrTileG, bwdMaskTileG]
    by_cases ha : active s DK BK k.1
    · rw [if_pos (by simpa using ha), if_pos ha, BlockState.writeMemTyped_real]
      rfl
    · rw [if_neg (by simpa using ha), if_neg ha]
  rw [stepStmts.cons_some hstore2] at hstep
  -- 13–16: `q_val`/`k_val`/`dg_val`/`cum_grad_dg` assigns
  obtain ⟨x1, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  obtain ⟨x2, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  obtain ⟨x3, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  obtain ⟨x4, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  set s16 := (((sst2.setReg "q_val" .real [BK] x1).setReg "k_val" .real [BK] x2).setReg
    "dg_val" .real [BK] x3).setReg "cum_grad_dg" .real [BK] x4 with hs16
  have hcum16 : s16.regs .real [BK] "cum_grad_dg" = some x4 := by
    rw [hs16]; exact BlockState.setReg_same _ _ _ _ _
  have hpdg16 : s16.regs .ptr [BK] "p_dg"
      = some (bwdPtrTileG s DG s_qk_h DK BT BK rr) := by
    rw [hs16, BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      hsst2, BlockState.foldl_writeMem_prop_masked_regs,
      hs11, BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      hsst1, BlockState.foldl_writeMem_prop_masked_regs,
      hs6, BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      hregs₂ _ _ _ (by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide), hsloop,
      BlockState.setReg_ne_name (h := by decide)]
    exact hpdg
  have hmask16 : s16.regs .bool [BK] "mask" = some (bwdMaskTileG s DK BK) := by
    rw [hs16, BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      hsst2, BlockState.foldl_writeMem_prop_masked_regs]
    exact hmask11
  -- 17: the `dg` store
  set vfun3 : TileIndex [BK] → ℝ := fun k => (x4.data k).unbotD 0 with hvfun3
  set sst3 := (TileShape.allIndices [BK]).foldl
    (fun (acc : BlockState) k =>
      if active s DK BK k.1 then
        acc.writeMem DG (offset s s_qk_h DK rr BT BK k.1) (vfun3 k)
      else acc) s16 with hsst3
  have hstore3 : stepStmt (Stmt.store TileDType.real [BK]
      (MemAccess.ptr (Op.ref TileDType.ptr [BK] "p_dg"))
      (Op.ref TileDType.real [BK] "cum_grad_dg")
      (MaskOpt.mask (Op.ref TileDType.bool [BK] "mask"))) s16 = some sst3 := by
    simp only [stepStmt, evalOp, Option.bind, Option.map, hcum16, hpdg16, hmask16]
    apply congrArg some
    rw [hsst3]
    congr 1
    funext acc k
    simp only [bwdPtrTileG, bwdMaskTileG]
    by_cases ha : active s DK BK k.1
    · rw [if_pos (by simpa using ha), if_pos ha, BlockState.writeMemTyped_real]
      rfl
    · rw [if_neg (by simpa using ha), if_neg ha]
  rw [stepStmts.cons_some hstore3] at hstep
  -- 18–25: the register-only pointer tail
  obtain ⟨u1, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  obtain ⟨u2, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  obtain ⟨u3, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  obtain ⟨u4, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  obtain ⟨u5, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  obtain ⟨u6, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  obtain ⟨u7, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  obtain ⟨u8, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  rw [stepStmts.nil] at hstep
  obtain rfl := Option.some.inj hstep
  simp only [BlockState.setReg_mem]
  rw [hsst3]
  refine Eq.trans (foldl_writeMem_prop_preserve_cell _ _ _ r oo _ _ ?_) ?_
  · intro k hk hmk hbad
    rcases hcDG with hne' | hno
    · exact hne' hbad.1
    · exact hno k.1 hmk hbad.2
  · rw [hs16]
    simp only [BlockState.setReg_mem]
    rw [hsst2]
    refine Eq.trans (foldl_writeMem_prop_preserve_cell _ _ _ r oo _ _ ?_) ?_
    · intro k hk hmk hbad
      rcases hcDK with hne' | hno
      · exact hne' hbad.1
      · exact hno k.1 hmk hbad.2
    · rw [hs11]
      simp only [BlockState.setReg_mem]
      rw [hsst1]
      refine Eq.trans (foldl_writeMem_prop_preserve_cell _ _ _ r oo _ _ ?_) ?_
      · intro k hk hmk hbad
        rcases hcDQ with hne' | hno
        · exact hne' hbad.1
        · exact hno k.1 hmk hbad.2
      · rw [hs6]
        simp only [BlockState.setReg_mem]
        have h₂ : s₂.mem r oo = sloop.mem r oo := by
          rw [hmem₂]
          simp only [BlockState.setReg_mem]
        rw [h₂, hsloop]
        simp only [BlockState.setReg_mem]

/-! ### The core's register-pin walk (`R`-side)

The tolerant bridge's tail-safety obligation runs from *whatever* state
the core reaches (`sMid`). The core never assigns a pointer register (its
targets are `t`/`g_val`/`last_g`/`dq*`/`dk*`/`q_val`/`k_val`/`dg_val`/
`cum_grad_dg`, and its stores are register-transparent), so every pointer
pin survives verbatim. -/

/-- Registers off the core's 13 assign targets pass through a successful
core run unchanged. -/
private theorem bwd_core_regsR (R : RoundingModel) (DK BT BK : Nat)
    (sin sMid : BlockState)
    (h : stepStmtsR R (bwdCoreG DK BT BK) sin = some sMid)
    (dt : TileDType) (sh : TileShape) (nm : RegName)
    (h1 : nm ≠ "t") (h2 : nm ≠ "g_val") (h3 : nm ≠ "last_g") (h4 : nm ≠ "dq1")
    (h5 : nm ≠ "dq2") (h6 : nm ≠ "dq") (h7 : nm ≠ "dk1") (h8 : nm ≠ "dk2")
    (h9 : nm ≠ "dk") (h10 : nm ≠ "q_val") (h11 : nm ≠ "k_val")
    (h12 : nm ≠ "dg_val") (h13 : nm ≠ "cum_grad_dg") :
    sMid.regs dt sh nm = sin.regs dt sh nm := by
  unfold bwdCoreG at h
  obtain ⟨v0, -, h⟩ := stepStmtsR_cons_assign_inv h
  obtain ⟨v1, -, h⟩ := stepStmtsR_cons_assign_inv h
  obtain ⟨s₂, hif, h⟩ := stepStmtsR_cons_inv h
  have hregs₂ := stepStmtR_ifThen_assign_regs hif
  obtain ⟨v3, -, h⟩ := stepStmtsR_cons_assign_inv h
  obtain ⟨v4, -, h⟩ := stepStmtsR_cons_assign_inv h
  obtain ⟨v5, -, h⟩ := stepStmtsR_cons_assign_inv h
  obtain ⟨v6, -, h⟩ := stepStmtsR_cons_assign_inv h
  obtain ⟨s7, h, hregs7⟩ := stepStmtsR_cons_store_regs h
  obtain ⟨w1, -, h⟩ := stepStmtsR_cons_assign_inv h
  obtain ⟨w2, -, h⟩ := stepStmtsR_cons_assign_inv h
  obtain ⟨w3, -, h⟩ := stepStmtsR_cons_assign_inv h
  obtain ⟨w4, -, h⟩ := stepStmtsR_cons_assign_inv h
  obtain ⟨s12, h, hregs12⟩ := stepStmtsR_cons_store_regs h
  obtain ⟨x1, -, h⟩ := stepStmtsR_cons_assign_inv h
  obtain ⟨x2, -, h⟩ := stepStmtsR_cons_assign_inv h
  obtain ⟨x3, -, h⟩ := stepStmtsR_cons_assign_inv h
  obtain ⟨x4, -, h⟩ := stepStmtsR_cons_assign_inv h
  obtain ⟨s17, h, hregs17⟩ := stepStmtsR_cons_store_regs h
  rw [stepStmtsR_nil] at h
  obtain rfl := Option.some.inj h
  rw [hregs17, BlockState.setReg_ne_name (h := h13),
    BlockState.setReg_ne_name (h := h12), BlockState.setReg_ne_name (h := h11),
    BlockState.setReg_ne_name (h := h10), hregs12,
    BlockState.setReg_ne_name (h := h9), BlockState.setReg_ne_name (h := h8),
    BlockState.setReg_ne_name (h := h8), BlockState.setReg_ne_name (h := h7),
    hregs7, BlockState.setReg_ne_name (h := h6),
    BlockState.setReg_ne_name (h := h5), BlockState.setReg_ne_name (h := h5),
    BlockState.setReg_ne_name (h := h4), hregs₂ _ _ _ h3,
    BlockState.setReg_ne_name (h := h2), BlockState.setReg_ne_name (h := h1)]

/-! ### The dead tail's `TraceSafeR` walk (non-final iterations only) -/

/-- **Tail safety at a non-final iteration.** With the pointers pinned at
row `rr ≥ 1`, every lane address is `≥ DK`, so the eight `-= DK`
self-decrements satisfy `Op.SafeAtR`'s per-state no-underflow clause. (In
the *final* iteration `rr = 0` this is false at `pid₀ = pid₂ = 0` — that
is precisely the underflow the tolerant bridge absorbs instead.) -/
private theorem bwd_tailSafeR (R : RoundingModel) (bounds : RegionBounds)
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (s sMid : BlockState) (s_qk_h DK BT BK rr : Nat) (hrr : 1 ≤ rr)
    (hpg : sMid.regs .ptr [BK] "p_g" = some (bwdPtrTileG s G s_qk_h DK BT BK rr))
    (hpk : sMid.regs .ptr [BK] "p_k" = some (bwdPtrTileG s K s_qk_h DK BT BK rr))
    (hpq : sMid.regs .ptr [BK] "p_q" = some (bwdPtrTileG s Q s_qk_h DK BT BK rr))
    (hpdqi : sMid.regs .ptr [BK] "p_dq_inner"
      = some (bwdPtrTileG s DQInner s_qk_h DK BT BK rr))
    (hpdki : sMid.regs .ptr [BK] "p_dk_inner"
      = some (bwdPtrTileG s DKInner s_qk_h DK BT BK rr))
    (hpdqt : sMid.regs .ptr [BK] "p_dq_inter"
      = some (bwdPtrTileG s DQInter s_qk_h DK BT BK rr))
    (hpdkt : sMid.regs .ptr [BK] "p_dk_inter"
      = some (bwdPtrTileG s DKInter s_qk_h DK BT BK rr))
    (hpdg : sMid.regs .ptr [BK] "p_dg"
      = some (bwdPtrTileG s DG s_qk_h DK BT BK rr)) :
    Stmt.TraceSafeListR R bounds (bwdTailG DK BK) sMid := by
  have hDKle : ∀ i : Fin BK, DK ≤ offset s s_qk_h DK rr BT BK i := by
    intro i
    have h1 : DK ≤ (s.pids 1 * BT + rr) * DK :=
      Nat.le_mul_of_pos_left DK (by omega)
    simp only [offset, baseOffset]
    omega
  have hsafe : ∀ (pname : RegName) (reg : RegionName) (st : BlockState),
      st.regs .ptr [BK] pname = some (bwdPtrTileG s reg s_qk_h DK BT BK rr) →
      Stmt.TraceSafeR R bounds (Stmt.assign .ptr [BK] pname
        (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [BK] pname)
          (Op.constNat DK))) st := by
    intro pname reg st hpin
    simp only [Stmt.TraceSafeR, Op.SafeAtR]
    refine ⟨trivial, trivial, ?_⟩
    intro ptrs offs hp ho i
    rw [evalOpR_ref, hpin] at hp
    obtain rfl := Option.some.inj hp
    rw [bwd_evalOpR_constNat] at ho
    obtain rfl := Option.some.inj ho
    exact hDKle (Broadcast.scalarR.leftIndex i).1
  unfold bwdTailG
  refine Stmt.TraceSafeListR.cons_intro (hsafe "p_g" G sMid hpg)
    (fun t1 ht1 => ?_)
  obtain ⟨q1, -, rfl⟩ := stepStmtR_assign_inv ht1
  refine Stmt.TraceSafeListR.cons_intro (hsafe "p_k" K _ (by
    rw [BlockState.setReg_ne_name (h := by decide)]; exact hpk))
    (fun t2 ht2 => ?_)
  obtain ⟨q2, -, rfl⟩ := stepStmtR_assign_inv ht2
  refine Stmt.TraceSafeListR.cons_intro (hsafe "p_q" Q _ (by
    rw [BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide)]; exact hpq))
    (fun t3 ht3 => ?_)
  obtain ⟨q3, -, rfl⟩ := stepStmtR_assign_inv ht3
  refine Stmt.TraceSafeListR.cons_intro (hsafe "p_dq_inner" DQInner _ (by
    rw [BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide)]; exact hpdqi))
    (fun t4 ht4 => ?_)
  obtain ⟨q4, -, rfl⟩ := stepStmtR_assign_inv ht4
  refine Stmt.TraceSafeListR.cons_intro (hsafe "p_dk_inner" DKInner _ (by
    rw [BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide)]; exact hpdki))
    (fun t5 ht5 => ?_)
  obtain ⟨q5, -, rfl⟩ := stepStmtR_assign_inv ht5
  refine Stmt.TraceSafeListR.cons_intro (hsafe "p_dq_inter" DQInter _ (by
    rw [BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide)]; exact hpdqt))
    (fun t6 ht6 => ?_)
  obtain ⟨q6, -, rfl⟩ := stepStmtR_assign_inv ht6
  refine Stmt.TraceSafeListR.cons_intro (hsafe "p_dk_inter" DKInter _ (by
    rw [BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide)]; exact hpdkt))
    (fun t7 ht7 => ?_)
  obtain ⟨q7, -, rfl⟩ := stepStmtR_assign_inv ht7
  refine Stmt.TraceSafeListR.cons_intro (hsafe "p_dg" DG _ (by
    rw [BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide)]; exact hpdg))
    (fun _ _ => Stmt.TraceSafeListR.nil_intro)

/-! ### The core's `TraceSafeR` walk -/

set_option maxHeartbeats 8000000 in
/-- Per-iteration `TraceSafeListR` for the reverse-loop **core**: the
seven masked loads' and three masked stores' active lanes are exactly the
skin's (`t`-independent) masked windows at row `rr`, in bounds by the
corresponding window bound; the `t` recompute, the `last_g` capture and
the `exp2` combinations are register-only. Pins thread across the two
mid-body stores via `stepStmtR_store_real_regs` and across the capture
via `stepStmtR_ifThen_assign_regs`. -/
private theorem bwd_bodySafeR (R : RoundingModel) (bounds : RegionBounds)
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (s stt : BlockState) (s_qk_h DK BT BK rr m : Nat)
    (hpg : stt.regs .ptr [BK] "p_g" = some (bwdPtrTileG s G s_qk_h DK BT BK rr))
    (hpk : stt.regs .ptr [BK] "p_k" = some (bwdPtrTileG s K s_qk_h DK BT BK rr))
    (hpq : stt.regs .ptr [BK] "p_q" = some (bwdPtrTileG s Q s_qk_h DK BT BK rr))
    (hpdqi : stt.regs .ptr [BK] "p_dq_inner"
      = some (bwdPtrTileG s DQInner s_qk_h DK BT BK rr))
    (hpdki : stt.regs .ptr [BK] "p_dk_inner"
      = some (bwdPtrTileG s DKInner s_qk_h DK BT BK rr))
    (hpdqt : stt.regs .ptr [BK] "p_dq_inter"
      = some (bwdPtrTileG s DQInter s_qk_h DK BT BK rr))
    (hpdkt : stt.regs .ptr [BK] "p_dk_inter"
      = some (bwdPtrTileG s DKInter s_qk_h DK BT BK rr))
    (hpdg : stt.regs .ptr [BK] "p_dg"
      = some (bwdPtrTileG s DG s_qk_h DK BT BK rr))
    (hmask : stt.regs .bool [BK] "mask" = some (bwdMaskTileG s DK BK))
    (hbG : ∀ i : Fin BK, active s DK BK i → offset s s_qk_h DK rr BT BK i < bounds G)
    (hbQ : ∀ i : Fin BK, active s DK BK i → offset s s_qk_h DK rr BT BK i < bounds Q)
    (hbK : ∀ i : Fin BK, active s DK BK i → offset s s_qk_h DK rr BT BK i < bounds K)
    (hbDQi : ∀ i : Fin BK, active s DK BK i →
      offset s s_qk_h DK rr BT BK i < bounds DQInner)
    (hbDKi : ∀ i : Fin BK, active s DK BK i →
      offset s s_qk_h DK rr BT BK i < bounds DKInner)
    (hbDQt : ∀ i : Fin BK, active s DK BK i →
      offset s s_qk_h DK rr BT BK i < bounds DQInter)
    (hbDKt : ∀ i : Fin BK, active s DK BK i →
      offset s s_qk_h DK rr BT BK i < bounds DKInter)
    (hbDG : ∀ i : Fin BK, active s DK BK i →
      offset s s_qk_h DK rr BT BK i < bounds DG) :
    Stmt.TraceSafeListR R bounds (bwdCoreG DK BT BK)
      (stt.setReg "__rev_t" .nat [] (Tile.scalar m)) := by
  set sloop := stt.setReg "__rev_t" .nat [] (Tile.scalar m) with hsloop
  have hpg' : sloop.regs .ptr [BK] "p_g"
      = some (bwdPtrTileG s G s_qk_h DK BT BK rr) := by
    rw [hsloop]; simpa using hpg
  have hpk' : sloop.regs .ptr [BK] "p_k"
      = some (bwdPtrTileG s K s_qk_h DK BT BK rr) := by
    rw [hsloop]; simpa using hpk
  have hpq' : sloop.regs .ptr [BK] "p_q"
      = some (bwdPtrTileG s Q s_qk_h DK BT BK rr) := by
    rw [hsloop]; simpa using hpq
  have hpdqi' : sloop.regs .ptr [BK] "p_dq_inner"
      = some (bwdPtrTileG s DQInner s_qk_h DK BT BK rr) := by
    rw [hsloop]; simpa using hpdqi
  have hpdki' : sloop.regs .ptr [BK] "p_dk_inner"
      = some (bwdPtrTileG s DKInner s_qk_h DK BT BK rr) := by
    rw [hsloop]; simpa using hpdki
  have hpdqt' : sloop.regs .ptr [BK] "p_dq_inter"
      = some (bwdPtrTileG s DQInter s_qk_h DK BT BK rr) := by
    rw [hsloop]; simpa using hpdqt
  have hpdkt' : sloop.regs .ptr [BK] "p_dk_inter"
      = some (bwdPtrTileG s DKInter s_qk_h DK BT BK rr) := by
    rw [hsloop]; simpa using hpdkt
  have hpdg' : sloop.regs .ptr [BK] "p_dg"
      = some (bwdPtrTileG s DG s_qk_h DK BT BK rr) := by
    rw [hsloop]; simpa using hpdg
  have hmask' : sloop.regs .bool [BK] "mask" = some (bwdMaskTileG s DK BK) := by
    rw [hsloop]; simpa using hmask
  unfold bwdCoreG
  -- (0) the `t` recompute: register-only
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun t0 ht0 => ?_)
  obtain ⟨v0, -, rfl⟩ := stepStmtR_assign_inv ht0
  have hpg0 : (sloop.setReg "t" .nat [] v0).regs .ptr [BK] "p_g"
      = some (bwdPtrTileG s G s_qk_h DK BT BK rr) := by
    rw [BlockState.setReg_ne_name (h := by decide)]; exact hpg'
  have hmask0 : (sloop.setReg "t" .nat [] v0).regs .bool [BK] "mask"
      = some (bwdMaskTileG s DK BK) := by
    rw [BlockState.setReg_ne_name (h := by decide)]; exact hmask'
  -- (1) the masked `g_val` load
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun t1 ht1 => ?_)
  · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
    refine ⟨by simp [Op.SafeAtR.eq_def], by simp [Op.SafeAtR.eq_def], ?_⟩
    intro ptrs hptrs idx hactive
    rw [evalOpR_ref, hpg0] at hptrs
    obtain rfl := Option.some.inj hptrs
    obtain ⟨masks, hm, hmi⟩ := hactive
    rw [evalOpR_ref, hmask0] at hm
    obtain rfl := Option.some.inj hm
    have ha : active s DK BK idx.1 := by simpa [bwdMaskTileG] using hmi
    simpa [bwdPtrTileG] using hbG idx.1 ha
  · obtain ⟨v1, -, rfl⟩ := stepStmtR_assign_inv ht1
    -- (2) the `last_g` capture: guard is nat arithmetic, body register-only
    refine Stmt.TraceSafeListR.cons_intro ?_ (fun t2 ht2 => ?_)
    · simp only [Stmt.TraceSafeR]
      refine ⟨by simp [Op.SafeAtR.eq_def], ?_⟩
      split
      · split
        · refine Stmt.TraceSafeListR.of_forall _ _ ?_
          intro st hst s'
          simp only [List.mem_cons, List.not_mem_nil, or_false] at hst
          subst hst
          simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]
        · trivial
      · trivial
    · have hregs₂ := stepStmtR_ifThen_assign_regs ht2
      -- pins at the post-capture state
      have h2pdqi : t2.regs .ptr [BK] "p_dq_inner"
          = some (bwdPtrTileG s DQInner s_qk_h DK BT BK rr) := by
        rw [hregs₂ _ _ _ (by decide), BlockState.setReg_ne_name (h := by decide),
          BlockState.setReg_ne_name (h := by decide)]
        exact hpdqi'
      have h2pdqt : t2.regs .ptr [BK] "p_dq_inter"
          = some (bwdPtrTileG s DQInter s_qk_h DK BT BK rr) := by
        rw [hregs₂ _ _ _ (by decide), BlockState.setReg_ne_name (h := by decide),
          BlockState.setReg_ne_name (h := by decide)]
        exact hpdqt'
      have h2pdki : t2.regs .ptr [BK] "p_dk_inner"
          = some (bwdPtrTileG s DKInner s_qk_h DK BT BK rr) := by
        rw [hregs₂ _ _ _ (by decide), BlockState.setReg_ne_name (h := by decide),
          BlockState.setReg_ne_name (h := by decide)]
        exact hpdki'
      have h2pdkt : t2.regs .ptr [BK] "p_dk_inter"
          = some (bwdPtrTileG s DKInter s_qk_h DK BT BK rr) := by
        rw [hregs₂ _ _ _ (by decide), BlockState.setReg_ne_name (h := by decide),
          BlockState.setReg_ne_name (h := by decide)]
        exact hpdkt'
      have h2pq : t2.regs .ptr [BK] "p_q"
          = some (bwdPtrTileG s Q s_qk_h DK BT BK rr) := by
        rw [hregs₂ _ _ _ (by decide), BlockState.setReg_ne_name (h := by decide),
          BlockState.setReg_ne_name (h := by decide)]
        exact hpq'
      have h2pk : t2.regs .ptr [BK] "p_k"
          = some (bwdPtrTileG s K s_qk_h DK BT BK rr) := by
        rw [hregs₂ _ _ _ (by decide), BlockState.setReg_ne_name (h := by decide),
          BlockState.setReg_ne_name (h := by decide)]
        exact hpk'
      have h2pdg : t2.regs .ptr [BK] "p_dg"
          = some (bwdPtrTileG s DG s_qk_h DK BT BK rr) := by
        rw [hregs₂ _ _ _ (by decide), BlockState.setReg_ne_name (h := by decide),
          BlockState.setReg_ne_name (h := by decide)]
        exact hpdg'
      have h2mask : t2.regs .bool [BK] "mask" = some (bwdMaskTileG s DK BK) := by
        rw [hregs₂ _ _ _ (by decide), BlockState.setReg_ne_name (h := by decide),
          BlockState.setReg_ne_name (h := by decide)]
        exact hmask'
      -- (3) the masked `dq1` load
      refine Stmt.TraceSafeListR.cons_intro ?_ (fun t3 ht3 => ?_)
      · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
          MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
        refine ⟨by simp [Op.SafeAtR.eq_def], by simp [Op.SafeAtR.eq_def], ?_⟩
        intro ptrs hptrs idx hactive
        rw [evalOpR_ref, h2pdqi] at hptrs
        obtain rfl := Option.some.inj hptrs
        obtain ⟨masks, hm, hmi⟩ := hactive
        rw [evalOpR_ref, h2mask] at hm
        obtain rfl := Option.some.inj hm
        have ha : active s DK BK idx.1 := by simpa [bwdMaskTileG] using hmi
        simpa [bwdPtrTileG] using hbDQi idx.1 ha
      · obtain ⟨v3, -, rfl⟩ := stepStmtR_assign_inv ht3
        have h3pdqt : (t2.setReg "dq1" .real [BK] v3).regs .ptr [BK] "p_dq_inter"
            = some (bwdPtrTileG s DQInter s_qk_h DK BT BK rr) := by
          rw [BlockState.setReg_ne_name (h := by decide)]; exact h2pdqt
        have h3mask : (t2.setReg "dq1" .real [BK] v3).regs .bool [BK] "mask"
            = some (bwdMaskTileG s DK BK) := by
          rw [BlockState.setReg_ne_name (h := by decide)]; exact h2mask
        -- (4) the masked `dq2` load (in-place input channel)
        refine Stmt.TraceSafeListR.cons_intro ?_ (fun t4 ht4 => ?_)
        · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
            MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
          refine ⟨by simp [Op.SafeAtR.eq_def], by simp [Op.SafeAtR.eq_def], ?_⟩
          intro ptrs hptrs idx hactive
          rw [evalOpR_ref, h3pdqt] at hptrs
          obtain rfl := Option.some.inj hptrs
          obtain ⟨masks, hm, hmi⟩ := hactive
          rw [evalOpR_ref, h3mask] at hm
          obtain rfl := Option.some.inj hm
          have ha : active s DK BK idx.1 := by simpa [bwdMaskTileG] using hmi
          simpa [bwdPtrTileG] using hbDQt idx.1 ha
        · obtain ⟨v4, -, rfl⟩ := stepStmtR_assign_inv ht4
          -- (5)(6) `dq2` scaling and `dq` sum: register-only
          refine Stmt.TraceSafeListR.cons_intro
            (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun t5 ht5 => ?_)
          obtain ⟨v5, -, rfl⟩ := stepStmtR_assign_inv ht5
          refine Stmt.TraceSafeListR.cons_intro
            (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun t6 ht6 => ?_)
          obtain ⟨v6, -, rfl⟩ := stepStmtR_assign_inv ht6
          have h6pdqt : ((((t2.setReg "dq1" .real [BK] v3).setReg
              "dq2" .real [BK] v4).setReg "dq2" .real [BK] v5).setReg
              "dq" .real [BK] v6).regs .ptr [BK] "p_dq_inter"
              = some (bwdPtrTileG s DQInter s_qk_h DK BT BK rr) := by
            rw [BlockState.setReg_ne_name (h := by decide),
              BlockState.setReg_ne_name (h := by decide),
              BlockState.setReg_ne_name (h := by decide),
              BlockState.setReg_ne_name (h := by decide)]
            exact h2pdqt
          have h6mask : ((((t2.setReg "dq1" .real [BK] v3).setReg
              "dq2" .real [BK] v4).setReg "dq2" .real [BK] v5).setReg
              "dq" .real [BK] v6).regs .bool [BK] "mask"
              = some (bwdMaskTileG s DK BK) := by
            rw [BlockState.setReg_ne_name (h := by decide),
              BlockState.setReg_ne_name (h := by decide),
              BlockState.setReg_ne_name (h := by decide),
              BlockState.setReg_ne_name (h := by decide)]
            exact h2mask
          -- (7) the masked `dq_inter` store
          refine Stmt.TraceSafeListR.cons_intro ?_ (fun t7 ht7 => ?_)
          · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MemAccess.SafeAtR,
              MaskOpt.SafeAtR, MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR,
              memAccessActiveAddressSafeR]
            refine ⟨by simp [Op.SafeAtR.eq_def], by simp [Op.SafeAtR.eq_def],
              by simp [Op.SafeAtR.eq_def], ?_⟩
            intro ptrs hptrs idx hactive
            rw [evalOpR_ref, h6pdqt] at hptrs
            obtain rfl := Option.some.inj hptrs
            obtain ⟨masks, hm, hmi⟩ := hactive
            rw [evalOpR_ref, h6mask] at hm
            obtain rfl := Option.some.inj hm
            have ha : active s DK BK idx.1 := by simpa [bwdMaskTileG] using hmi
            simpa [bwdPtrTileG] using hbDQt idx.1 ha
          · have hregs7 := stepStmtR_store_real_regs ht7
            have h7pdki : t7.regs .ptr [BK] "p_dk_inner"
                = some (bwdPtrTileG s DKInner s_qk_h DK BT BK rr) := by
              rw [hregs7, BlockState.setReg_ne_name (h := by decide),
                BlockState.setReg_ne_name (h := by decide),
                BlockState.setReg_ne_name (h := by decide),
                BlockState.setReg_ne_name (h := by decide)]
              exact h2pdki
            have h7pdkt : t7.regs .ptr [BK] "p_dk_inter"
                = some (bwdPtrTileG s DKInter s_qk_h DK BT BK rr) := by
              rw [hregs7, BlockState.setReg_ne_name (h := by decide),
                BlockState.setReg_ne_name (h := by decide),
                BlockState.setReg_ne_name (h := by decide),
                BlockState.setReg_ne_name (h := by decide)]
              exact h2pdkt
            have h7pq : t7.regs .ptr [BK] "p_q"
                = some (bwdPtrTileG s Q s_qk_h DK BT BK rr) := by
              rw [hregs7, BlockState.setReg_ne_name (h := by decide),
                BlockState.setReg_ne_name (h := by decide),
                BlockState.setReg_ne_name (h := by decide),
                BlockState.setReg_ne_name (h := by decide)]
              exact h2pq
            have h7pk : t7.regs .ptr [BK] "p_k"
                = some (bwdPtrTileG s K s_qk_h DK BT BK rr) := by
              rw [hregs7, BlockState.setReg_ne_name (h := by decide),
                BlockState.setReg_ne_name (h := by decide),
                BlockState.setReg_ne_name (h := by decide),
                BlockState.setReg_ne_name (h := by decide)]
              exact h2pk
            have h7pdg : t7.regs .ptr [BK] "p_dg"
                = some (bwdPtrTileG s DG s_qk_h DK BT BK rr) := by
              rw [hregs7, BlockState.setReg_ne_name (h := by decide),
                BlockState.setReg_ne_name (h := by decide),
                BlockState.setReg_ne_name (h := by decide),
                BlockState.setReg_ne_name (h := by decide)]
              exact h2pdg
            have h7mask : t7.regs .bool [BK] "mask"
                = some (bwdMaskTileG s DK BK) := by
              rw [hregs7, BlockState.setReg_ne_name (h := by decide),
                BlockState.setReg_ne_name (h := by decide),
                BlockState.setReg_ne_name (h := by decide),
                BlockState.setReg_ne_name (h := by decide)]
              exact h2mask
            -- (8) the masked `dk1` load
            refine Stmt.TraceSafeListR.cons_intro ?_ (fun t8 ht8 => ?_)
            · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
                MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
              refine ⟨by simp [Op.SafeAtR.eq_def], by simp [Op.SafeAtR.eq_def], ?_⟩
              intro ptrs hptrs idx hactive
              rw [evalOpR_ref, h7pdki] at hptrs
              obtain rfl := Option.some.inj hptrs
              obtain ⟨masks, hm, hmi⟩ := hactive
              rw [evalOpR_ref, h7mask] at hm
              obtain rfl := Option.some.inj hm
              have ha : active s DK BK idx.1 := by simpa [bwdMaskTileG] using hmi
              simpa [bwdPtrTileG] using hbDKi idx.1 ha
            · obtain ⟨w1, -, rfl⟩ := stepStmtR_assign_inv ht8
              have h8pdkt : (t7.setReg "dk1" .real [BK] w1).regs .ptr [BK]
                  "p_dk_inter"
                  = some (bwdPtrTileG s DKInter s_qk_h DK BT BK rr) := by
                rw [BlockState.setReg_ne_name (h := by decide)]; exact h7pdkt
              have h8mask : (t7.setReg "dk1" .real [BK] w1).regs .bool [BK] "mask"
                  = some (bwdMaskTileG s DK BK) := by
                rw [BlockState.setReg_ne_name (h := by decide)]; exact h7mask
              -- (9) the masked `dk2` load (in-place input channel)
              refine Stmt.TraceSafeListR.cons_intro ?_ (fun t9 ht9 => ?_)
              · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
                  MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
                refine ⟨by simp [Op.SafeAtR.eq_def],
                  by simp [Op.SafeAtR.eq_def], ?_⟩
                intro ptrs hptrs idx hactive
                rw [evalOpR_ref, h8pdkt] at hptrs
                obtain rfl := Option.some.inj hptrs
                obtain ⟨masks, hm, hmi⟩ := hactive
                rw [evalOpR_ref, h8mask] at hm
                obtain rfl := Option.some.inj hm
                have ha : active s DK BK idx.1 := by
                  simpa [bwdMaskTileG] using hmi
                simpa [bwdPtrTileG] using hbDKt idx.1 ha
              · obtain ⟨w2, -, rfl⟩ := stepStmtR_assign_inv ht9
                -- (10)(11) `dk2` scaling and `dk` sum: register-only
                refine Stmt.TraceSafeListR.cons_intro
                  (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
                  (fun t10 ht10 => ?_)
                obtain ⟨w3, -, rfl⟩ := stepStmtR_assign_inv ht10
                refine Stmt.TraceSafeListR.cons_intro
                  (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
                  (fun t11 ht11 => ?_)
                obtain ⟨w4, -, rfl⟩ := stepStmtR_assign_inv ht11
                have h11pdkt : ((((t7.setReg "dk1" .real [BK] w1).setReg
                    "dk2" .real [BK] w2).setReg "dk2" .real [BK] w3).setReg
                    "dk" .real [BK] w4).regs .ptr [BK] "p_dk_inter"
                    = some (bwdPtrTileG s DKInter s_qk_h DK BT BK rr) := by
                  rw [BlockState.setReg_ne_name (h := by decide),
                    BlockState.setReg_ne_name (h := by decide),
                    BlockState.setReg_ne_name (h := by decide),
                    BlockState.setReg_ne_name (h := by decide)]
                  exact h7pdkt
                have h11mask : ((((t7.setReg "dk1" .real [BK] w1).setReg
                    "dk2" .real [BK] w2).setReg "dk2" .real [BK] w3).setReg
                    "dk" .real [BK] w4).regs .bool [BK] "mask"
                    = some (bwdMaskTileG s DK BK) := by
                  rw [BlockState.setReg_ne_name (h := by decide),
                    BlockState.setReg_ne_name (h := by decide),
                    BlockState.setReg_ne_name (h := by decide),
                    BlockState.setReg_ne_name (h := by decide)]
                  exact h7mask
                -- (12) the masked `dk_inter` store
                refine Stmt.TraceSafeListR.cons_intro ?_ (fun t12 ht12 => ?_)
                · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def,
                    MemAccess.SafeAtR, MaskOpt.SafeAtR, MaskOpt.ActiveR,
                    MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
                  refine ⟨by simp [Op.SafeAtR.eq_def],
                    by simp [Op.SafeAtR.eq_def],
                    by simp [Op.SafeAtR.eq_def], ?_⟩
                  intro ptrs hptrs idx hactive
                  rw [evalOpR_ref, h11pdkt] at hptrs
                  obtain rfl := Option.some.inj hptrs
                  obtain ⟨masks, hm, hmi⟩ := hactive
                  rw [evalOpR_ref, h11mask] at hm
                  obtain rfl := Option.some.inj hm
                  have ha : active s DK BK idx.1 := by
                    simpa [bwdMaskTileG] using hmi
                  simpa [bwdPtrTileG] using hbDKt idx.1 ha
                · have hregs12 := stepStmtR_store_real_regs ht12
                  have h12pq : t12.regs .ptr [BK] "p_q"
                      = some (bwdPtrTileG s Q s_qk_h DK BT BK rr) := by
                    rw [hregs12, BlockState.setReg_ne_name (h := by decide),
                      BlockState.setReg_ne_name (h := by decide),
                      BlockState.setReg_ne_name (h := by decide),
                      BlockState.setReg_ne_name (h := by decide)]
                    exact h7pq
                  have h12pk : t12.regs .ptr [BK] "p_k"
                      = some (bwdPtrTileG s K s_qk_h DK BT BK rr) := by
                    rw [hregs12, BlockState.setReg_ne_name (h := by decide),
                      BlockState.setReg_ne_name (h := by decide),
                      BlockState.setReg_ne_name (h := by decide),
                      BlockState.setReg_ne_name (h := by decide)]
                    exact h7pk
                  have h12pdg : t12.regs .ptr [BK] "p_dg"
                      = some (bwdPtrTileG s DG s_qk_h DK BT BK rr) := by
                    rw [hregs12, BlockState.setReg_ne_name (h := by decide),
                      BlockState.setReg_ne_name (h := by decide),
                      BlockState.setReg_ne_name (h := by decide),
                      BlockState.setReg_ne_name (h := by decide)]
                    exact h7pdg
                  have h12mask : t12.regs .bool [BK] "mask"
                      = some (bwdMaskTileG s DK BK) := by
                    rw [hregs12, BlockState.setReg_ne_name (h := by decide),
                      BlockState.setReg_ne_name (h := by decide),
                      BlockState.setReg_ne_name (h := by decide),
                      BlockState.setReg_ne_name (h := by decide)]
                    exact h7mask
                  -- (13) the masked `q_val` load
                  refine Stmt.TraceSafeListR.cons_intro ?_ (fun t13 ht13 => ?_)
                  · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def,
                      MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR,
                      memAccessActiveAddressSafeR]
                    refine ⟨by simp [Op.SafeAtR.eq_def],
                      by simp [Op.SafeAtR.eq_def], ?_⟩
                    intro ptrs hptrs idx hactive
                    rw [evalOpR_ref, h12pq] at hptrs
                    obtain rfl := Option.some.inj hptrs
                    obtain ⟨masks, hm, hmi⟩ := hactive
                    rw [evalOpR_ref, h12mask] at hm
                    obtain rfl := Option.some.inj hm
                    have ha : active s DK BK idx.1 := by
                      simpa [bwdMaskTileG] using hmi
                    simpa [bwdPtrTileG] using hbQ idx.1 ha
                  · obtain ⟨x1, -, rfl⟩ := stepStmtR_assign_inv ht13
                    have h13pk : (t12.setReg "q_val" .real [BK] x1).regs
                        .ptr [BK] "p_k"
                        = some (bwdPtrTileG s K s_qk_h DK BT BK rr) := by
                      rw [BlockState.setReg_ne_name (h := by decide)]
                      exact h12pk
                    have h13mask : (t12.setReg "q_val" .real [BK] x1).regs
                        .bool [BK] "mask" = some (bwdMaskTileG s DK BK) := by
                      rw [BlockState.setReg_ne_name (h := by decide)]
                      exact h12mask
                    -- (14) the masked `k_val` load
                    refine Stmt.TraceSafeListR.cons_intro ?_
                      (fun t14 ht14 => ?_)
                    · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def,
                        MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR,
                        memAccessActiveAddressSafeR]
                      refine ⟨by simp [Op.SafeAtR.eq_def],
                        by simp [Op.SafeAtR.eq_def], ?_⟩
                      intro ptrs hptrs idx hactive
                      rw [evalOpR_ref, h13pk] at hptrs
                      obtain rfl := Option.some.inj hptrs
                      obtain ⟨masks, hm, hmi⟩ := hactive
                      rw [evalOpR_ref, h13mask] at hm
                      obtain rfl := Option.some.inj hm
                      have ha : active s DK BK idx.1 := by
                        simpa [bwdMaskTileG] using hmi
                      simpa [bwdPtrTileG] using hbK idx.1 ha
                    · obtain ⟨x2, -, rfl⟩ := stepStmtR_assign_inv ht14
                      -- (15)(16) `dg_val` and `cum_grad_dg`: register-only
                      refine Stmt.TraceSafeListR.cons_intro
                        (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
                        (fun t15 ht15 => ?_)
                      obtain ⟨x3, -, rfl⟩ := stepStmtR_assign_inv ht15
                      refine Stmt.TraceSafeListR.cons_intro
                        (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
                        (fun t16 ht16 => ?_)
                      obtain ⟨x4, -, rfl⟩ := stepStmtR_assign_inv ht16
                      have h16pdg : ((((t12.setReg "q_val" .real [BK] x1).setReg
                          "k_val" .real [BK] x2).setReg
                          "dg_val" .real [BK] x3).setReg
                          "cum_grad_dg" .real [BK] x4).regs .ptr [BK] "p_dg"
                          = some (bwdPtrTileG s DG s_qk_h DK BT BK rr) := by
                        rw [BlockState.setReg_ne_name (h := by decide),
                          BlockState.setReg_ne_name (h := by decide),
                          BlockState.setReg_ne_name (h := by decide),
                          BlockState.setReg_ne_name (h := by decide)]
                        exact h12pdg
                      have h16mask : ((((t12.setReg
                          "q_val" .real [BK] x1).setReg
                          "k_val" .real [BK] x2).setReg
                          "dg_val" .real [BK] x3).setReg
                          "cum_grad_dg" .real [BK] x4).regs .bool [BK] "mask"
                          = some (bwdMaskTileG s DK BK) := by
                        rw [BlockState.setReg_ne_name (h := by decide),
                          BlockState.setReg_ne_name (h := by decide),
                          BlockState.setReg_ne_name (h := by decide),
                          BlockState.setReg_ne_name (h := by decide)]
                        exact h12mask
                      -- (17) the masked `dg` store
                      refine Stmt.TraceSafeListR.cons_intro ?_
                        (fun _ _ => Stmt.TraceSafeListR.nil_intro)
                      simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def,
                        MemAccess.SafeAtR, MaskOpt.SafeAtR, MaskOpt.ActiveR,
                        MemAccess.ActiveAddressSafeR,
                        memAccessActiveAddressSafeR]
                      refine ⟨by simp [Op.SafeAtR.eq_def],
                        by simp [Op.SafeAtR.eq_def],
                        by simp [Op.SafeAtR.eq_def], ?_⟩
                      intro ptrs hptrs idx hactive
                      rw [evalOpR_ref, h16pdg] at hptrs
                      obtain rfl := Option.some.inj hptrs
                      obtain ⟨masks, hm, hmi⟩ := hactive
                      rw [evalOpR_ref, h16mask] at hm
                      obtain rfl := Option.some.inj hm
                      have ha : active s DK BK idx.1 := by
                        simpa [bwdMaskTileG] using hmi
                      simpa [bwdPtrTileG] using hbDG idx.1 ha

/-! ### The rounded Hoare triple (region-model `execR` run) -/

set_option maxHeartbeats 8000000 in
/-- Termination, per-window values and the per-cell frame of the whole
backward kernel under `execR R`, from an **arbitrary** launch state: the
exact `bwd_prologue_eval_general` / `bwdInvG` stack runs verbatim (the
prologue and the loop body — pointer tail included — are cast-free, so
`execR R` collapses onto the exact stepper; the exact stepper's truncated
`ptrSub` is total, so the final iteration's underflow is harmless *here*
— only the flat bridge needs the tolerant treatment), extended with the
per-cell memory frame carried alongside `bwdInvG`
(`bwd_body_step_frame`). -/
private theorem bwd_runR (R : RoundingModel)
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (s_qk_h DK BT BK : Nat) (hBT : 0 < BT) (hBK : BK ≤ DK)
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
    (s₀ : BlockState) :
    ∃ sfin,
      execR R (bwd_decay_global_cumsum_surface DQInner DQInter DKInner DKInter
        Q K G DG s_qk_h DK BT BK).toAlgKernel s₀ = some sfin
      ∧ (∀ (t : Fin BT) (i : Fin BK), active s₀ DK BK i →
          sfin.readMem DQInter (offset s₀ s_qk_h DK t.val BT BK i)
            = bwdDQInterClosed s₀ DQInner DQInter G s_qk_h DK BT BK t i)
      ∧ (∀ (t : Fin BT) (i : Fin BK), active s₀ DK BK i →
          sfin.readMem DKInter (offset s₀ s_qk_h DK t.val BT BK i)
            = bwdDKInterClosed s₀ DKInner DKInter G s_qk_h DK BT BK t i)
      ∧ (∀ (t : Fin BT) (i : Fin BK), active s₀ DK BK i →
          sfin.readMem DG (offset s₀ s_qk_h DK t.val BT BK i)
            = bwdDGClosed s₀ DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK t i)
      ∧ (∀ r oo,
          (r ≠ DQInter ∨ ∀ (t : Fin BT) (i : Fin BK), active s₀ DK BK i →
            oo ≠ offset s₀ s_qk_h DK t.val BT BK i) →
          (r ≠ DKInter ∨ ∀ (t : Fin BT) (i : Fin BK), active s₀ DK BK i →
            oo ≠ offset s₀ s_qk_h DK t.val BT BK i) →
          (r ≠ DG ∨ ∀ (t : Fin BT) (i : Fin BK), active s₀ DK BK i →
            oo ≠ offset s₀ s_qk_h DK t.val BT BK i) →
          sfin.mem r oo = s₀.mem r oo) := by
  obtain ⟨s0, hpro, hsh⟩ :=
    bwd_prologue_eval_general DQInner DQInter DKInner DKInter Q K G DG
      s_qk_h DK BT BK hBT s₀
  have hP0 : bwdInvG DQInner DQInter DKInner DKInter Q K G DG s₀ s_qk_h DK BT BK 0 s0 :=
    bwdInvG_entry DQInner DQInter DKInner DKInter Q K G DG s₀ s0 s_qk_h DK BT BK hsh
  -- the reverse loop with the per-cell frame carried alongside `bwdInvG`
  obtain ⟨final, sfinal, haux, hge, hPfinal⟩ :=
    forRangeAux_inv (idx := "__rev_t") (stop := BT) (step := 1)
      (body := bwdIterBodyG DK BT BK)
      (P := fun m sc =>
        bwdInvG DQInner DQInter DKInner DKInter Q K G DG s₀ s_qk_h DK BT BK m sc
        ∧ ∀ r oo,
          (r ≠ DQInter ∨ ∀ (t : Fin BT) (i : Fin BK), active s₀ DK BK i →
            oo ≠ offset s₀ s_qk_h DK t.val BT BK i) →
          (r ≠ DKInter ∨ ∀ (t : Fin BT) (i : Fin BK), active s₀ DK BK i →
            oo ≠ offset s₀ s_qk_h DK t.val BT BK i) →
          (r ≠ DG ∨ ∀ (t : Fin BT) (i : Fin BK), active s₀ DK BK i →
            oo ≠ offset s₀ s_qk_h DK t.val BT BK i) →
          sc.mem r oo = s₀.mem r oo)
      one_ne_zero
      (fun m sc hm hP => by
        obtain ⟨s', hstep, hP'⟩ :=
          bwd_decay_cumsum_step_general DQInner DQInter DKInner DKInter Q K G DG s₀
            s_qk_h DK BT BK hBK hDKInner_DQInter hDKInter_DQInter hQ_DQInter
            hQ_DKInter hK_DQInter hK_DKInter hDQInter_DKInter hDQInter_DG
            hDKInter_DG hG_DQInter hG_DKInter hG_DG hQ_DG hK_DG hDQInner_DQInter
            hDQInner_DKInter hDQInner_DG hDKInner_DKInter hDKInner_DG m hm sc hP.1
        refine ⟨s', hstep, hP', ?_⟩
        intro r oo hcDQ hcDK hcDG
        obtain ⟨hpid, hGrd, hQrd, hKrd, hDQird, hDKird, hmaskP, hcumP, hlastgP,
          hpgP, hpkP, hpqP, hpdqiP, hpdkiP, hpdqtP, hpdktP, hpdgP,
          hDQIunt, hDKIunt, hDGunt, hDQIwr, hDKIwr, hDGwr⟩ := hP.1
        rw [bwdPtrDecTileG_row s₀ DQInter s_qk_h DK BT BK m hm] at hpdqtP
        rw [bwdPtrDecTileG_row s₀ DKInter s_qk_h DK BT BK m hm] at hpdktP
        rw [bwdPtrDecTileG_row s₀ DG s_qk_h DK BT BK m hm] at hpdgP
        rw [bwd_body_step_frame DQInter DKInter DG s₀ sc s' s_qk_h DK BT BK
          (BT - 1 - m) m hpdqtP hpdktP hpdgP hmaskP hstep r oo ?_ ?_ ?_]
        · exact hP.2 r oo hcDQ hcDK hcDG
        · rcases hcDQ with hne' | hno
          · exact Or.inl hne'
          · exact Or.inr fun i ha => hno ⟨BT - 1 - m, by omega⟩ i ha
        · rcases hcDK with hne' | hno
          · exact Or.inl hne'
          · exact Or.inr fun i ha => hno ⟨BT - 1 - m, by omega⟩ i ha
        · rcases hcDG with hne' | hno
          · exact Or.inl hne'
          · exact Or.inr fun i ha => hno ⟨BT - 1 - m, by omega⟩ i ha)
      0 s0 ⟨hP0, fun r oo _ _ _ => by rw [hsh.mem]⟩
  obtain ⟨⟨_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    hDQIwr, hDKIwr, hDGwr⟩, hframe⟩ := hPfinal
  -- assemble the `execR` run through the cast-free collapses
  have hLoopR : stepStmtR R (Stmt.forRangeDyn "__rev_t" (Op.constNat 0)
      (bwdStopOp BT) (Op.constNat 1) (bwdIterBodyG DK BT BK)) s0
      = some sfinal := by
    have h1 : stepStmtR R (Stmt.forRangeDyn "__rev_t" (Op.constNat 0)
        (bwdStopOp BT) (Op.constNat 1) (bwdIterBodyG DK BT BK)) s0
        = stepForRangeAuxR R "__rev_t" 0 BT 1 (bwdIterBodyG DK BT BK) s0 := by
      simp only [stepStmtR, bwd_evalOpR_constNat R 0 s0,
        bwd_stopOpR_eval R s0 BT hBT, bwd_evalOpR_constNat R 1 s0,
        Option.bind_some]
      rfl
    rw [h1, stepForRangeAuxR_castFree R _ (bwdIterBody_castFree R DK BT BK)
      "__rev_t"]
    exact haux
  have hproG : stepStmts (bwdPrologueG DQInner DQInter DKInner DKInter Q K G DG
      s_qk_h DK BT BK) s₀ = some s0 := by
    rw [← bwd_prologue_take_general]
    exact hpro
  refine ⟨sfinal, ?_, ?_, ?_, ?_, ?_⟩
  · unfold execR
    rw [bwd_body_decomp_io DQInner DQInter DKInner DKInter Q K G DG s_qk_h DK BT BK,
      stepStmtsR_append R
        (bwdPrologueG DQInner DQInter DKInner DKInter Q K G DG s_qk_h DK BT BK) _ s₀,
      bwdPrologue_castFree R DQInner DQInter DKInner DKInter Q K G DG
        s_qk_h DK BT BK s₀,
      hproG, Option.bind_some,
      stepStmtsR_cons_some (show stepStmtR R (Stmt.forRangeDyn "__rev_t"
        (Op.constNat 0) (bwdStopOp BT) (Op.constNat 1)
        (bwdCoreG DK BT BK ++ bwdTailG DK BK)) s0 = some sfinal from by
          rw [← bwdIterBodyG_decomp]; exact hLoopR),
      stepStmtsR_nil]
  · intro t i ha
    rw [hDQIwr t.val (by omega) t.isLt i, if_pos ha]
    exact bwdDqOutG_eq_closed s₀ DQInner DQInter G s_qk_h DK BT BK t i
  · intro t i ha
    rw [hDKIwr t.val (by omega) t.isLt i, if_pos ha]
    exact bwdDkOutG_eq_closed s₀ DKInner DKInter G s_qk_h DK BT BK t i
  · intro t i ha
    rw [hDGwr t.val (by omega) t.isLt i, if_pos ha]
    exact bwdCumPartialG_eq_DGClosed s₀ DQInner DQInter DKInner DKInter Q K G
      s_qk_h DK BT BK t i
  · exact hframe

/-! ### The headline -/

set_option maxHeartbeats 8000000 in
/-- **The `⊨[R]` streaming headline (wave-6 S3 grouped per-step emit
genre, 3-D grid) — the first consumer of the dead-write-tolerant flat
bridge.** For every rounding model `R`, the faithful
`bwd_decay_global_cumsum` surface implements, on its grouped
`StreamGroupedEmitMasked3DKernelIO` signature, the **ideal ℝ backward
decay combination** of the seven streamed rows: emitted window `(t, j)`
of channel `dq_inter` holds `dq_inner + dq_inter_old·exp2(g)`, of channel
`dk_inter` holds `dk_inner + dk_inter_old·exp2(g[BT−1] − g)`, and of
channel `dg` the reverse-scan suffix sum
`Σ_{u ≥ t} (dq[u]·q[u] − dk[u]·k[u])` — the spec `f` is exact real
arithmetic, `last_g` is the `g` stream's own step-`BT−1` row (no extra
channel, no mask widening), and the `dq_inter`/`dk_inter` channels are
**in place** (the pinned inputs are the launch-state contents; sound
because each row is read before it is rewritten, once). The kernel has
**zero rounding events** (all loads, `exp2` combinations and the three
per-step masked stores are at `.real`), so the skin's boundary
quantization degenerates: the readback contract's `R.round .real` is the
identity by the model's defining `round_real`.

**Why this is a direct `ImplementsR` proof and not
`ImplementsR.intro`**: the loop body's 8 trailing `p_x -= DK`
self-decrements genuinely underflow in the final iteration (row 0,
`pid₀ = pid₂ = 0` lanes have addresses `< DK`), so the kernel-wide
`TraceSafeR` walk that `intro` requires is **falsifiable** — machine
checked, not a proof gap. The flat leg instead goes through
`FlatAlloc.execR_flatten_deadPtrTail`: strict commutation for the
prologue + loop core + every non-final tail (`bwd_bodySafeR` /
`bwd_tailSafeR` at rows `≥ 1`), and observational agreement (`ObsAgree`)
across the final, dead, underflowing tail — the readback leg transports
along `ObsAgree.readMemAs_eq` and the frame leg along `ObsAgree.mem`, so
both reduce to the strict case after one rewrite.

Layer map: the region-model triple rides the exact `bwdInvG` stack
verbatim (`bwd_runR`: cast-free collapses + `forRangeAux_inv` +
`bwd_decay_cumsum_step_general`, with the per-cell frame
`bwd_body_step_frame` carried alongside); the flat leg is the tolerant
bridge fed the `bwdInvG` invariant itself as `P` (its pointer pins give
core-load/store bounds at row `BT−1−c` and tail no-underflow at rows
`≥ 1` via `bwd_core_regsR`); the spec legs are the per-channel stream
bridges (`bwdDqStream_eq_closed` / `bwdDkStream_eq_closed` /
`bwdDgStream_eq_closed`).

All hypotheses are truth-forced:

* the 19 region-distinctness hypotheses are **exactly** the exact step
  lemma `bwd_decay_cumsum_step_general`'s side-condition set (the loop
  stores into `dq_inter`/`dk_inter`/`dg` *between* its per-row re-reads
  of `g`/`q`/`k`/`dq_inner`/`dk_inner` and its own earlier outputs, so
  the invariant's whole-input frames and untouched-row conjuncts require
  the three output buffers not to alias any other buffer or each other;
  the launch allocates all eight tensors separately). The
  `dq_inter`-with-`dq_inter` / `dk_inter`-with-`dk_inter` *self*-aliasing
  is the in-place design, **not** a hypothesis.
* `hBK : BK ≤ DK` — row separation, as in the exact stack: the row-`m`
  scatters must not collide with other rows' windows, which needs every
  lane index `< BK` to stay inside one `DK`-wide row. The launch sets
  `BK = min(DK, 64)`.
* `hBT : 0 < BT` — the reverse loop's designated first row `BT − 1`
  (the `last_g` capture and the spec's `⟨BT − 1, _⟩ : Fin BT` index)
  needs a nonempty stream; the exact stack carries the same constraint
  (`bwd_prologue_eval_general`'s `hBT`), and the launch sets `BT = 16`.

Relation to the exact surface: the exact headline
`decay_cumsum_backward_closed_output_summary_general` above is retained
unchanged; this `⊨[R]` face restates all three closed forms at once on
the grouped streaming emit skin, for every `R` (at the `.real` grid the
two faces carry the same exact cells). Both faces are kept per the
rounding-as-default doctrine. -/
specification bwd_decay_global_cumsum_io_correctness (R : RoundingModel)
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (s_qk_h DK BT BK : Nat)
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
    (hBK : BK ≤ DK) (hBT : 0 < BT) :
    bwdDecayCumsumKernelIO DQInner DQInter DKInner DKInter Q K G DG
        s_qk_h DK BT BK ⊨[R]
      fun _ _ _ xs o t j => bwdDecayStreamSpec BT BK xs o t j := by
  intro A hd hregs hcov pid₀ pid₁ pid₂ xs s₀ hpid₀ hpid₁ hpid₂ hu hbr hbw hx
  subst hpid₀
  subst hpid₁
  subst hpid₂
  simp only [bwdDecayCumsumKernelIO] at hbr hbw hx ⊢
  -- the region-model run (exact stack ridden under `execR R`)
  obtain ⟨s1, hexec, hdqv, hdkv, hdgv, hframe⟩ :=
    bwd_runR R DQInner DQInter DKInner DKInter Q K G DG s_qk_h DK BT BK hBT hBK
      hDKInner_DQInter hDKInter_DQInter hQ_DQInter hQ_DKInter hK_DQInter
      hK_DKInter hDQInter_DKInter hDQInter_DG hDKInter_DG hG_DQInter hG_DKInter
      hG_DG hQ_DG hK_DG hDQInner_DQInter hDQInner_DKInter hDQInner_DG
      hDKInner_DKInter hDKInner_DG s₀
  -- prologue trace safety (register-only assigns)
  have hmsPre : Stmt.TraceSafeListR R A.extent
      (bwdPrologueG DQInner DQInter DKInner DKInter Q K G DG s_qk_h DK BT BK)
      s₀ := by
    refine Stmt.TraceSafeListR.of_forall _ _ ?_
    intro stmt hst s'
    simp only [bwdPrologueG, List.mem_cons, List.not_mem_nil, or_false] at hst
    rcases hst with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
      rfl | rfl | rfl | rfl | rfl <;>
      simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]
  -- the tolerant bridge's loop obligations, driven by `bwdInvG` itself
  have hloopOb : ∀ sP, stepStmtsR R
      (bwdPrologueG DQInner DQInter DKInner DKInter Q K G DG s_qk_h DK BT BK) s₀
        = some sP →
      ((Op.constNat 0).SafeAtR R A.extent sP ∧
        (bwdStopOp BT).SafeAtR R A.extent sP ∧
        (Op.constNat 1).SafeAtR R A.extent sP) ∧
      ∀ bv ev sv, evalOpR R (Op.constNat 0) sP = some bv →
        evalOpR R (bwdStopOp BT) sP = some ev →
        evalOpR R (Op.constNat 1) sP = some sv →
        bwdInvG DQInner DQInter DKInner DKInter Q K G DG s₀ s_qk_h DK BT BK
          (bv.data PUnit.unit) sP ∧
        ∀ (c : Nat) (s' : BlockState), c < ev.data PUnit.unit →
          bwdInvG DQInner DQInter DKInner DKInter Q K G DG s₀ s_qk_h DK BT BK c s' →
          Stmt.TraceSafeListR R A.extent (bwdCoreG DK BT BK)
            (s'.setReg "__rev_t" .nat [] (Tile.scalar c)) ∧
          (∃ s'', stepStmtsR R (bwdCoreG DK BT BK ++ bwdTailG DK BK)
              (s'.setReg "__rev_t" .nat [] (Tile.scalar c)) = some s'' ∧
            bwdInvG DQInner DQInter DKInner DKInter Q K G DG s₀ s_qk_h DK BT BK
              (c + sv.data PUnit.unit) s'') ∧
          (c + sv.data PUnit.unit < ev.data PUnit.unit → ∀ sMid,
            stepStmtsR R (bwdCoreG DK BT BK)
              (s'.setReg "__rev_t" .nat [] (Tile.scalar c)) = some sMid →
            Stmt.TraceSafeListR R A.extent (bwdTailG DK BK) sMid) := by
    intro sP hsP
    refine ⟨⟨by simp [Op.SafeAtR.eq_def], by simp [bwdStopOp, Op.SafeAtR.eq_def],
      by simp [Op.SafeAtR.eq_def]⟩, ?_⟩
    intro bv ev sv hbv hev hsv
    rw [bwd_evalOpR_constNat] at hbv
    obtain rfl := Option.some.inj hbv
    rw [bwd_stopOpR_eval R sP BT hBT] at hev
    obtain rfl := Option.some.inj hev
    rw [bwd_evalOpR_constNat] at hsv
    obtain rfl := Option.some.inj hsv
    obtain ⟨s0, hpro, hsh⟩ :=
      bwd_prologue_eval_general DQInner DQInter DKInner DKInter Q K G DG
        s_qk_h DK BT BK hBT s₀
    rw [bwdPrologue_castFree R DQInner DQInter DKInner DKInter Q K G DG
        s_qk_h DK BT BK s₀,
      ← bwd_prologue_take_general, hpro] at hsP
    obtain rfl := Option.some.inj hsP
    refine ⟨bwdInvG_entry DQInner DQInter DKInner DKInter Q K G DG s₀ s0
      s_qk_h DK BT BK hsh, ?_⟩
    intro c s' hc hP
    have hcBT : c < BT := hc
    have hPd := hP
    obtain ⟨hpid, hGrd, hQrd, hKrd, hDQird, hDKird, hmaskP, hcumP, hlastgP,
      hpgP, hpkP, hpqP, hpdqiP, hpdkiP, hpdqtP, hpdktP, hpdgP,
      hDQIunt, hDKIunt, hDGunt, hDQIwr, hDKIwr, hDGwr⟩ := hPd
    rw [bwdPtrDecTileG_row s₀ G s_qk_h DK BT BK c hcBT] at hpgP
    rw [bwdPtrDecTileG_row s₀ K s_qk_h DK BT BK c hcBT] at hpkP
    rw [bwdPtrDecTileG_row s₀ Q s_qk_h DK BT BK c hcBT] at hpqP
    rw [bwdPtrDecTileG_row s₀ DQInner s_qk_h DK BT BK c hcBT] at hpdqiP
    rw [bwdPtrDecTileG_row s₀ DKInner s_qk_h DK BT BK c hcBT] at hpdkiP
    rw [bwdPtrDecTileG_row s₀ DQInter s_qk_h DK BT BK c hcBT] at hpdqtP
    rw [bwdPtrDecTileG_row s₀ DKInter s_qk_h DK BT BK c hcBT] at hpdktP
    rw [bwdPtrDecTileG_row s₀ DG s_qk_h DK BT BK c hcBT] at hpdgP
    have hrow : BT - 1 - c < BT := by omega
    -- window bounds at row `BT-1-c` from the skin's read/write bounds
    have hbG' : ∀ i : Fin BK, active s₀ DK BK i →
        offset s₀ s_qk_h DK (BT - 1 - c) BT BK i < A.extent G := by
      intro i ha
      have h := hbr ⟨0, by omega⟩ ⟨BT - 1 - c, hrow⟩ i ha
      rw [fwd_addr_eq] at h
      exact h
    have hbDQi' : ∀ i : Fin BK, active s₀ DK BK i →
        offset s₀ s_qk_h DK (BT - 1 - c) BT BK i < A.extent DQInner := by
      intro i ha
      have h := hbr ⟨1, by omega⟩ ⟨BT - 1 - c, hrow⟩ i ha
      rw [fwd_addr_eq] at h
      exact h
    have hbDQt' : ∀ i : Fin BK, active s₀ DK BK i →
        offset s₀ s_qk_h DK (BT - 1 - c) BT BK i < A.extent DQInter := by
      intro i ha
      have h := hbr ⟨2, by omega⟩ ⟨BT - 1 - c, hrow⟩ i ha
      rw [fwd_addr_eq] at h
      exact h
    have hbDKi' : ∀ i : Fin BK, active s₀ DK BK i →
        offset s₀ s_qk_h DK (BT - 1 - c) BT BK i < A.extent DKInner := by
      intro i ha
      have h := hbr ⟨3, by omega⟩ ⟨BT - 1 - c, hrow⟩ i ha
      rw [fwd_addr_eq] at h
      exact h
    have hbDKt' : ∀ i : Fin BK, active s₀ DK BK i →
        offset s₀ s_qk_h DK (BT - 1 - c) BT BK i < A.extent DKInter := by
      intro i ha
      have h := hbr ⟨4, by omega⟩ ⟨BT - 1 - c, hrow⟩ i ha
      rw [fwd_addr_eq] at h
      exact h
    have hbQ' : ∀ i : Fin BK, active s₀ DK BK i →
        offset s₀ s_qk_h DK (BT - 1 - c) BT BK i < A.extent Q := by
      intro i ha
      have h := hbr ⟨5, by omega⟩ ⟨BT - 1 - c, hrow⟩ i ha
      rw [fwd_addr_eq] at h
      exact h
    have hbK' : ∀ i : Fin BK, active s₀ DK BK i →
        offset s₀ s_qk_h DK (BT - 1 - c) BT BK i < A.extent K := by
      intro i ha
      have h := hbr ⟨6, by omega⟩ ⟨BT - 1 - c, hrow⟩ i ha
      rw [fwd_addr_eq] at h
      exact h
    have hbDG' : ∀ i : Fin BK, active s₀ DK BK i →
        offset s₀ s_qk_h DK (BT - 1 - c) BT BK i < A.extent DG := by
      intro i ha
      have h := hbw ⟨2, by omega⟩ ⟨BT - 1 - c, hrow⟩ i ha
      rw [fwd_addr_eq] at h
      exact h
    refine ⟨?_, ?_, ?_⟩
    · exact bwd_bodySafeR R A.extent DQInner DQInter DKInner DKInter Q K G DG
        s₀ s' s_qk_h DK BT BK (BT - 1 - c) c
        hpgP hpkP hpqP hpdqiP hpdkiP hpdqtP hpdktP hpdgP hmaskP
        hbG' hbQ' hbK' hbDQi' hbDKi' hbDQt' hbDKt' hbDG'
    · obtain ⟨s'', hstep, hP'⟩ :=
        bwd_decay_cumsum_step_general DQInner DQInter DKInner DKInter Q K G DG s₀
          s_qk_h DK BT BK hBK hDKInner_DQInter hDKInter_DQInter hQ_DQInter
          hQ_DKInter hK_DQInter hK_DKInter hDQInter_DKInter hDQInter_DG
          hDKInter_DG hG_DQInter hG_DKInter hG_DG hQ_DG hK_DG hDQInner_DQInter
          hDQInner_DKInter hDQInner_DG hDKInner_DKInter hDKInner_DG c hcBT s' hP
      refine ⟨s'', ?_, hP'⟩
      rw [← bwdIterBodyG_decomp, bwdIterBody_castFree]
      exact hstep
    · intro hfin sMid hsMid
      have hfin' : c + 1 < BT := hfin
      have hrr : 1 ≤ BT - 1 - c := by omega
      have hMg : sMid.regs .ptr [BK] "p_g"
          = some (bwdPtrTileG s₀ G s_qk_h DK BT BK (BT - 1 - c)) := by
        rw [bwd_core_regsR R DK BT BK _ sMid hsMid .ptr [BK] "p_g"
            (by decide) (by decide) (by decide) (by decide) (by decide)
            (by decide) (by decide) (by decide) (by decide) (by decide)
            (by decide) (by decide) (by decide),
          BlockState.setReg_ne_name (h := by decide)]
        exact hpgP
      have hMk : sMid.regs .ptr [BK] "p_k"
          = some (bwdPtrTileG s₀ K s_qk_h DK BT BK (BT - 1 - c)) := by
        rw [bwd_core_regsR R DK BT BK _ sMid hsMid .ptr [BK] "p_k"
            (by decide) (by decide) (by decide) (by decide) (by decide)
            (by decide) (by decide) (by decide) (by decide) (by decide)
            (by decide) (by decide) (by decide),
          BlockState.setReg_ne_name (h := by decide)]
        exact hpkP
      have hMq : sMid.regs .ptr [BK] "p_q"
          = some (bwdPtrTileG s₀ Q s_qk_h DK BT BK (BT - 1 - c)) := by
        rw [bwd_core_regsR R DK BT BK _ sMid hsMid .ptr [BK] "p_q"
            (by decide) (by decide) (by decide) (by decide) (by decide)
            (by decide) (by decide) (by decide) (by decide) (by decide)
            (by decide) (by decide) (by decide),
          BlockState.setReg_ne_name (h := by decide)]
        exact hpqP
      have hMdqi : sMid.regs .ptr [BK] "p_dq_inner"
          = some (bwdPtrTileG s₀ DQInner s_qk_h DK BT BK (BT - 1 - c)) := by
        rw [bwd_core_regsR R DK BT BK _ sMid hsMid .ptr [BK] "p_dq_inner"
            (by decide) (by decide) (by decide) (by decide) (by decide)
            (by decide) (by decide) (by decide) (by decide) (by decide)
            (by decide) (by decide) (by decide),
          BlockState.setReg_ne_name (h := by decide)]
        exact hpdqiP
      have hMdki : sMid.regs .ptr [BK] "p_dk_inner"
          = some (bwdPtrTileG s₀ DKInner s_qk_h DK BT BK (BT - 1 - c)) := by
        rw [bwd_core_regsR R DK BT BK _ sMid hsMid .ptr [BK] "p_dk_inner"
            (by decide) (by decide) (by decide) (by decide) (by decide)
            (by decide) (by decide) (by decide) (by decide) (by decide)
            (by decide) (by decide) (by decide),
          BlockState.setReg_ne_name (h := by decide)]
        exact hpdkiP
      have hMdqt : sMid.regs .ptr [BK] "p_dq_inter"
          = some (bwdPtrTileG s₀ DQInter s_qk_h DK BT BK (BT - 1 - c)) := by
        rw [bwd_core_regsR R DK BT BK _ sMid hsMid .ptr [BK] "p_dq_inter"
            (by decide) (by decide) (by decide) (by decide) (by decide)
            (by decide) (by decide) (by decide) (by decide) (by decide)
            (by decide) (by decide) (by decide),
          BlockState.setReg_ne_name (h := by decide)]
        exact hpdqtP
      have hMdkt : sMid.regs .ptr [BK] "p_dk_inter"
          = some (bwdPtrTileG s₀ DKInter s_qk_h DK BT BK (BT - 1 - c)) := by
        rw [bwd_core_regsR R DK BT BK _ sMid hsMid .ptr [BK] "p_dk_inter"
            (by decide) (by decide) (by decide) (by decide) (by decide)
            (by decide) (by decide) (by decide) (by decide) (by decide)
            (by decide) (by decide) (by decide),
          BlockState.setReg_ne_name (h := by decide)]
        exact hpdktP
      have hMdg : sMid.regs .ptr [BK] "p_dg"
          = some (bwdPtrTileG s₀ DG s_qk_h DK BT BK (BT - 1 - c)) := by
        rw [bwd_core_regsR R DK BT BK _ sMid hsMid .ptr [BK] "p_dg"
            (by decide) (by decide) (by decide) (by decide) (by decide)
            (by decide) (by decide) (by decide) (by decide) (by decide)
            (by decide) (by decide) (by decide),
          BlockState.setReg_ne_name (h := by decide)]
        exact hpdgP
      exact bwd_tailSafeR R A.extent DQInner DQInter DKInner DKInter Q K G DG
        s₀ sMid s_qk_h DK BT BK (BT - 1 - c) hrr
        hMg hMk hMq hMdqi hMdki hMdqt hMdkt hMdg
  -- the dead-write-tolerant flat bridge
  obtain ⟨s', hexecF, hobs⟩ :=
    A.execR_flatten_deadPtrTail hd hcov R
      (bwd_decay_global_cumsum_surface DQInner DQInter DKInner DKInter Q K G DG
        s_qk_h DK BT BK).toAlgKernel
      (bwdPrologueG DQInner DQInter DKInner DKInter Q K G DG s_qk_h DK BT BK)
      "__rev_t" (Op.constNat 0) (bwdStopOp BT) (Op.constNat 1)
      (bwdCoreG DK BT BK) (bwdTailG DK BK)
      (bwd_body_decomp_io DQInner DQInter DKInner DKInter Q K G DG s_qk_h DK BT BK)
      (bwdTailG_ptrDecTail DK BK)
      (bwd_prologue_flattenOkR DQInner DQInter DKInner DKInter Q K G DG s_qk_h DK BT BK)
      (bwd_core_flattenOkR DK BT BK)
      (by simp [Op.FlattenOkR.eq_def]) (bwd_stopOp_flattenOkR BT)
      (by simp [Op.FlattenOkR.eq_def])
      s₀ hu hmsPre _ hloopOb s1 hexec
  have hmem0 : DQInter ∈ A.regions := by
    rw [hregs]
    show DQInter ∈ [DQInner, DQInter, DKInner, DKInter, Q, K, G, DG]
    simp
  have hmem1 : DKInter ∈ A.regions := by
    rw [hregs]
    show DKInter ∈ [DQInner, DQInter, DKInner, DKInter, Q, K, G, DG]
    simp
  have hmem2 : DG ∈ A.regions := by
    rw [hregs]
    show DG ∈ [DQInner, DQInter, DKInner, DKInter, Q, K, G, DG]
    simp
  refine ⟨s', hexecF, ?_, ?_⟩
  · -- readback leg: `ObsAgree.readMemAs_eq` → strict flat readback →
    -- region readback → per-channel stream bridge
    intro o t j hj
    have hlt := hbw o t j hj
    match o with
    | ⟨0, h0⟩ =>
        rw [hobs.readMemAs_eq, A.flattenState_readMemAs hd s1 hmem0 hlt .real,
          BlockState.readMemAs_real, fwd_addr_eq]
        have hval : s1.readMem DQInter (offset s₀ s_qk_h DK t.val BT BK j)
            = bwdDecayStreamSpec BT BK xs ⟨0, h0⟩ t j := by
          rw [hdqv t j hj]
          show bwdDQInterClosed s₀ DQInner DQInter G s_qk_h DK BT BK t j
            = bwdDqStream BT BK xs t j
          exact (bwdDqStream_eq_closed DQInner DQInter G s₀ s_qk_h DK BT BK xs
            (fun t' j' hj' => hx ⟨0, by omega⟩ t' j' hj')
            (fun t' j' hj' => hx ⟨1, by omega⟩ t' j' hj')
            (fun t' j' hj' => hx ⟨2, by omega⟩ t' j' hj') t j hj).symm
        simp only [RoundingModel.round_real_apply]
        exact congrArg some hval
    | ⟨1, h1⟩ =>
        rw [hobs.readMemAs_eq, A.flattenState_readMemAs hd s1 hmem1 hlt .real,
          BlockState.readMemAs_real, fwd_addr_eq]
        have hval : s1.readMem DKInter (offset s₀ s_qk_h DK t.val BT BK j)
            = bwdDecayStreamSpec BT BK xs ⟨1, h1⟩ t j := by
          rw [hdkv t j hj]
          show bwdDKInterClosed s₀ DKInner DKInter G s_qk_h DK BT BK t j
            = bwdDkStream BT BK xs t j
          exact (bwdDkStream_eq_closed DKInner DKInter G s₀ s_qk_h DK BT BK xs hBT
            (fun t' j' hj' => hx ⟨0, by omega⟩ t' j' hj')
            (fun t' j' hj' => hx ⟨3, by omega⟩ t' j' hj')
            (fun t' j' hj' => hx ⟨4, by omega⟩ t' j' hj') t j hj).symm
        simp only [RoundingModel.round_real_apply]
        exact congrArg some hval
    | ⟨2, h2⟩ =>
        rw [hobs.readMemAs_eq, A.flattenState_readMemAs hd s1 hmem2 hlt .real,
          BlockState.readMemAs_real, fwd_addr_eq]
        have hval : s1.readMem DG (offset s₀ s_qk_h DK t.val BT BK j)
            = bwdDecayStreamSpec BT BK xs ⟨2, h2⟩ t j := by
          rw [hdgv t j hj]
          show bwdDGClosed s₀ DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK t j
            = bwdDgStream BT BK xs t j
          exact (bwdDgStream_eq_closed DQInner DQInter DKInner DKInter Q K G s₀
            s_qk_h DK BT BK xs hBT
            (fun t' j' hj' => hx ⟨0, by omega⟩ t' j' hj')
            (fun t' j' hj' => hx ⟨1, by omega⟩ t' j' hj')
            (fun t' j' hj' => hx ⟨2, by omega⟩ t' j' hj')
            (fun t' j' hj' => hx ⟨3, by omega⟩ t' j' hj')
            (fun t' j' hj' => hx ⟨4, by omega⟩ t' j' hj')
            (fun t' j' hj' => hx ⟨5, by omega⟩ t' j' hj')
            (fun t' j' hj' => hx ⟨6, by omega⟩ t' j' hj') t j hj).symm
        simp only [RoundingModel.round_real_apply]
        exact congrArg some hval
    | ⟨n + 3, h3⟩ => exact absurd h3 (show ¬(n + 3 < 3) by omega)
  · -- frame leg: `ObsAgree.mem` → the strict flattenState mem analysis
    intro r' o' hcond
    rw [hobs.mem]
    by_cases hr : r' = A.flat
    · subst hr
      show (A.flattenState s1).mem A.flat o'
          = (A.flattenState s₀).mem A.flat o'
      simp only [FlatAlloc.flattenState]
      unfold FlatAlloc.readFlat
      cases hdec : A.decode o' with
      | none => rfl
      | some p =>
          obtain ⟨r, o⟩ := p
          obtain ⟨hrmem, hoeq, holt⟩ := A.decode_sound hdec
          show A.trCell (s1.mem r o) = A.trCell (s₀.mem r o)
          refine congrArg A.trCell (hframe r o ?_ ?_ ?_)
          · by_cases hrX : r = DQInter
            · refine Or.inr fun t i ha hoo => ?_
              rcases hcond with hflat | hn
              · exact absurd rfl hflat
              · exact hn ⟨0, by omega⟩ t i ha
                  (by rw [hoeq, hrX, hoo, fwd_addr_eq])
            · exact Or.inl hrX
          · by_cases hrX : r = DKInter
            · refine Or.inr fun t i ha hoo => ?_
              rcases hcond with hflat | hn
              · exact absurd rfl hflat
              · exact hn ⟨1, by omega⟩ t i ha
                  (by rw [hoeq, hrX, hoo, fwd_addr_eq])
            · exact Or.inl hrX
          · by_cases hrX : r = DG
            · refine Or.inr fun t i ha hoo => ?_
              rcases hcond with hflat | hn
              · exact absurd rfl hflat
              · exact hn ⟨2, by omega⟩ t i ha
                  (by rw [hoeq, hrX, hoo, fwd_addr_eq])
            · exact Or.inl hrX
    · simp only [FlatAlloc.flattenState, if_neg hr]

end BwdIOFace

end VeriTile.Bench.TritonBenchG.DecayCumsum
