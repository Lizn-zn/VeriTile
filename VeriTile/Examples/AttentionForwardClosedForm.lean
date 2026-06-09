import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.Math.Attention
import VeriTile.Triton.Kernel

/-!
# `attention_forward_triton` — closed-form correctness (WIP scaffold)

Scaffold for replacing the *self-referential* output summary in
`bench/tritonbench_g/attention_forward_triton` (whose `expected` is the kernel's
own executed output, hence tautological) with a genuine closed-form claim:
the quantized flash-attention forward kernel computes
`VeriTile.Triton.attentionRealBase2PerKeyScale` (base-2 softmax, per-block
key-scale) of the loaded Q/K/V tiles.

The surface kernel is copied verbatim from the bench port (bench ports are not
importable as modules). The correctness theorem below is **general**: it is
stated over arbitrary batch/head strides `stride_qz, stride_qh`, head count `H`,
block sizes `BLOCK_M, BLOCK_N`, KV-block count `numKVBlocks` (so
`N_CTX = BLOCK_N · numKVBlocks`), head dimension `HEAD_DIM`, tile head width
`BLOCK_DMODEL`, and active head width `HEAD_ACTIVE`, with arbitrary per-program
`q_scale` / per-block `k_scale`. The Python test case
(`B=2, H=4, N_CTX=128, HEAD_DIM=128, BLOCK_M=128, BLOCK_N=64, HEAD_ACTIVE=96`,
`q_scale = k_scale = 1`) is the special case of this statement.

The only layout assumptions are the usual contiguity contracts the kernel relies
on: `stride_qm = stride_kn = HEAD_DIM` (row stride = head dimension) and head
stride `1`, so the block-pointer advance `BLOCK_N · HEAD_DIM` composes into a
clean per-key address.

Proof: the multi-phase online-softmax recurrence argument. The pure-math heart
(`attentionRealBase2PerKeyScale_eq_streaming`, `osStep_foldl_eq_batch` in
`Math/Attention.lean`) and the full `exec`-side loop unfolding (Phase 3) are both
complete and sorry-free.
-/

namespace VeriTile.Examples.AttentionForwardClosedForm

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-! ## exec-stepping helpers

`evalOp` has `@[simp]` lemmas for `add`/`mul`/`sub`/`div` but none for the
integral `floorDiv`/`mod` cases (both appear in this kernel's prefix:
`off_z = off_hz // H`, `off_h = off_hz % H`, `vk_offset = qvk_offset // HEAD_DIM`).
These fill that gap so the deterministic prefix steps under `simp`.

**Validated stepping recipe** (proven on the 8-statement scalar prefix): the
compiled body reduces by `rfl`, so
`rw [show body.take N = [⟨concrete stmts⟩] from rfl]` exposes the statement list,
then `simp [stepStmts, stepStmt, evalOp_floorDiv, evalOp_mod, Option.bind]` steps
all assigns (the `@[simp] evalOp_*` and `setReg` lemmas thread register lookups
through the accumulated `setReg` chain automatically). -/
-- `evalOp_floorDiv` and `evalOp_mod` now live in `VeriTile.Triton.Kernel`
-- (EvalHelpers) and are reused from there via the `import` above. The former
-- local copies (byte-identical) were removed to avoid an ambiguous-name clash
-- for downstream files that import both this module and the Kernel lib.

/-- Eval helper for `boolAnd` (the `k_mask` boundary∧head masks): no `@[simp]`
form exists, like `floorDiv`/`mod`. -/
theorem evalOp_boolAnd {a b shape} (bc : Broadcast a b shape)
    (x y : Op .bool _) (s : BlockState) :
    evalOp (.boolAnd bc x y) s = (do
      let vx ← evalOp x s; let vy ← evalOp y s;
      some (Tile.bop (fun p q : Bool => p && q) bc vx vy)) := by
  simp [evalOp]

/-- Specialized `expandDim` axis-0 eval over a `nat` register (canonical axis,
explicit `[1,D]` result shape — so `rw`/`conv` match it without the
proof-term/`insertAxis`-unification friction of the generic lemma). The
exec-stepping **recipe** that finally works on the `triton{}`-elaborated surface:
`conv` to focus the receiver, `rw` this (or `unfold evalOp` for concrete inner
ops), collapse with `Option.bind_eq_bind/bind_some/pure_def`, then `ext`. -/
@[simp] theorem evalOp_expandDim_zero_nat {D : Nat} (name : RegName) (s : BlockState) :
    @evalOp .nat [1, D] (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] name)) s =
      (s.regs .nat [D] name).bind (fun v =>
        some ({ data := fun i : TileIndex [1, D] => v.data (i.2.1, PUnit.unit) } : Tile .nat [1, D])) := by
  unfold evalOp; simp [Tile.expandDim]; rfl

/-- Axis-1 `expandDim` over a `nat` register (the `offs_m[:, None]` column
broadcast in the `Q_ptrs`/`O_block_ptr` pointer offsets). -/
@[simp] theorem evalOp_expandDim_one_nat {M : Nat} (name : RegName) (s : BlockState) :
    @evalOp .nat [M, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] name)) s =
      (s.regs .nat [M] name).bind (fun v =>
        some ({ data := fun i : TileIndex [M, 1] => v.data (i.1, PUnit.unit) } : Tile .nat [M, 1])) := by
  unfold evalOp; simp [Tile.expandDim]; rfl

-- `evalOp_ptrAdd` and `evalOp_ptrBase` also now live in
-- `VeriTile.Triton.Kernel` (EvalHelpers) and are reused from there via the
-- `import` above (former byte-identical local copies removed; see note above).

/-- Eval helper for `exp2` (`tl.math.exp2`): `Tile.uop realExp2` over the operand. -/
theorem evalOp_exp2 {shape : TileShape} (a : Op .real shape) (s : BlockState) :
    evalOp (.exp2 a) s = (do let va ← evalOp a s; some (Tile.uop WithBot.realExp2 va)) := by
  simp [evalOp]

/-- **Worked pointer-eval template** (the `Q_ptrs`/`O_block_ptr` shape):
`ptrAdd (ptrBase Q) (qvk_offset + offs_m[:,None]·HEAD_DIM + offs_k[None,:]·1)`
evaluates, lane `(i,e)`, to the address `qvk + i·HEAD_DIM + e`. Same recipe as
`kmask_eval`: `simp only` over the arithmetic + canonical-axis expandDim lemmas
(which fire after the `Option.bind` collapse), then `ext` for the `ptrAdd` data. -/
theorem qptrs_eval (s : BlockState) (Q : RegionName) (BM BD HEAD_DIM qvk : Nat) (g : Fin BM → Nat)
    (hqvk : s.regs .nat [] "qvk_offset" = some (Tile.scalar qvk))
    (hm : s.regs .nat [BM] "offs_m" = some (Tile.vec g))
    (hk : s.regs .nat [BD] "offs_k" = some (Tile.vec (fun e : Fin BD => e.val))) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Q)
        (Op.add .nat Broadcast.nil.consL.consR
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qvk_offset")
            (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m"))
              (Op.constNat HEAD_DIM)))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BD] "offs_k"))
            (Op.constNat 1)))) s
      = some (⟨fun idx : TileIndex [BM, BD] =>
          (Q.cast, qvk + g idx.1 * HEAD_DIM + idx.2.1.val * 1)⟩ : Tile .ptr [BM, BD]) := by
  rw [evalOp_ptrAdd, evalOp_ptrBase]
  simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref, evalOp_expandDim_one_nat,
    evalOp_expandDim_zero_nat, hqvk, hm, hk, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec]
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec, NumericDType.add, NumericDType.mul]

/-- **Worked q-load mask eval** (`(offs_m[:,None] < N_CTX) ∧ (arange[None,:] <
HEAD_ACTIVE)`, shape `[BLOCK_M, BLOCK_DMODEL]`): `q_mask[r,e] = (r < N_CTX) ∧
(e < HEAD_ACTIVE)`. The `q` analogue of `kmask_eval` (axes swapped: row mask on
`offs_m`, column mask on the head arange). -/
theorem qmask_eval (s : BlockState) (BM BD NC HA : Nat) (g : Fin BM → Nat)
    (hm : s.regs .nat [BM] "offs_m" = some (Tile.vec g)) :
    evalOp (Op.boolAnd Broadcast.nil.consL.consR
        (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m"))
          (Op.constNat NC))
        (Op.expandDim ⟨0, by simp⟩ (Op.lt .nat Broadcast.scalarR (Op.arange BD) (Op.constNat HA)))) s
      = some ⟨fun idx : TileIndex [BM, BD] =>
          (decide (g idx.1 < NC) && decide (idx.2.1.val < HA))⟩ := by
  rw [evalOp_boolAnd]
  conv_lhs => arg 1; rw [evalOp_lt]; arg 1; rw [evalOp_expandDim_one_nat, hm]
  simp only [Option.bind_eq_bind, Option.bind_some, evalOp_constNat]
  conv_lhs => arg 1; unfold evalOp
  simp only [Option.bind_eq_bind, Option.bind_some, Option.pure_def, evalOp_lt, evalOp_arange,
    evalOp_constNat]
  refine congrArg some ?_
  ext idx
  simp [Tile.bop, Tile.cop, Tile.expandDim, Tile.vec, ComparableDType.lt]

/-- `K_ptrs` eval: `ptrAdd (ptrBase K) (qvk + offs_k[:,None] + offs_n[None,:]·HEAD_DIM)`,
shape `[BLOCK_DMODEL, BLOCK_N]`, address `qvk + e + j·HEAD_DIM` (note `offs_k` term
has no stride multiply). -/
theorem kptrs_eval (s : BlockState) (K : RegionName) (BD BN HEAD_DIM qvk : Nat)
    (hqvk : s.regs .nat [] "qvk_offset" = some (Tile.scalar qvk))
    (hk : s.regs .nat [BD] "offs_k" = some (Tile.vec (fun e : Fin BD => e.val)))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val))) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase K)
        (Op.add .nat Broadcast.nil.consL.consR
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qvk_offset")
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BD] "offs_k")))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n"))
            (Op.constNat HEAD_DIM)))) s
      = some (⟨fun idx : TileIndex [BD, BN] => (K.cast, qvk + idx.1.val + idx.2.1.val * HEAD_DIM)⟩ : Tile .ptr [BD, BN]) := by
  rw [evalOp_ptrAdd, evalOp_ptrBase]
  simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref, evalOp_expandDim_one_nat,
    evalOp_expandDim_zero_nat, hqvk, hk, hn, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec]
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec, NumericDType.add, NumericDType.mul]

/-- `V_ptrs` eval: `ptrAdd (ptrBase V) (qvk + offs_n[:,None]·HEAD_DIM + offs_k[None,:]·1)`,
shape `[BLOCK_N, BLOCK_DMODEL]`, address `qvk + j·HEAD_DIM + e`. -/
theorem vptrs_eval (s : BlockState) (V : RegionName) (BN BD HEAD_DIM qvk : Nat)
    (hqvk : s.regs .nat [] "qvk_offset" = some (Tile.scalar qvk))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val)))
    (hk : s.regs .nat [BD] "offs_k" = some (Tile.vec (fun e : Fin BD => e.val))) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase V)
        (Op.add .nat Broadcast.nil.consL.consR
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qvk_offset")
            (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BN] "offs_n")) (Op.constNat HEAD_DIM)))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BD] "offs_k")) (Op.constNat 1)))) s
      = some (⟨fun idx : TileIndex [BN, BD] => (V.cast, qvk + idx.1.val * HEAD_DIM + idx.2.1.val * 1)⟩ : Tile .ptr [BN, BD]) := by
  rw [evalOp_ptrAdd, evalOp_ptrBase]
  simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref, evalOp_expandDim_one_nat,
    evalOp_expandDim_zero_nat, hqvk, hn, hk, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec]
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec, NumericDType.add, NumericDType.mul]

/-- `Q_scale_ptr` eval (scalar): `ptrBase Q_scale + (q_scale_offset + start_m)`. -/
theorem qscaleptr_eval (s : BlockState) (Q_scale : RegionName) (qso sm : Nat)
    (hq : s.regs .nat [] "q_scale_offset" = some (Tile.scalar qso))
    (hsm : s.regs .nat [] "start_m" = some (Tile.scalar sm)) :
    evalOp (Op.ptrAdd Broadcast.nil (Op.ptrBase Q_scale)
        (Op.add .nat Broadcast.nil (Op.ref .nat [] "q_scale_offset") (Op.ref .nat [] "start_m"))) s
      = some (Tile.scalar (Q_scale.cast, qso + sm)) := by
  rw [evalOp_ptrAdd, evalOp_ptrBase]
  simp only [evalOp_add, evalOp_ref, hq, hsm, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  · simp [Tile.ptrAdd, Tile.bop]
  · simp [Tile.ptrAdd, Tile.bop, NumericDType.add]

/-- `K_scale_ptr` eval (scalar): `ptrBase K_scale + k_scale_offset`. -/
theorem kscaleptr_eval (s : BlockState) (K_scale : RegionName) (kso : Nat)
    (hk : s.regs .nat [] "k_scale_offset" = some (Tile.scalar kso)) :
    evalOp (Op.ptrAdd Broadcast.nil (Op.ptrBase K_scale) (Op.ref .nat [] "k_scale_offset")) s
      = some (Tile.scalar (K_scale.cast, kso)) := by
  rw [evalOp_ptrAdd, evalOp_ptrBase]
  simp only [evalOp_ref, hk, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  · simp [Tile.ptrAdd]
  · simp [Tile.ptrAdd]

/-- `m_i` init eval: `tl.zeros − inf = full 0 + (−∞)` → the all-`⊥` tile (matches
`mP … 0`). -/
theorem mi_init_eval (s : BlockState) (BM : Nat) :
    evalOp (Op.add .real Broadcast.scalarR (Op.full [BM] (Op.const 0)) Op.negInf) s
      = some (⟨fun _ : TileIndex [BM] => (⊥ : WithBot ℝ)⟩ : Tile .real [BM]) := by
  simp only [evalOp_add, evalOp_full, evalOp_negInf, evalOp_const, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext r
  simp only [Tile.bop_data, NumericDType.add]
  rfl

/-- `l_i` init eval: `tl.zeros + 1.0` → the all-`1` tile (matches `lP … 0`). -/
theorem li_init_eval (s : BlockState) (BM : Nat) :
    evalOp (Op.add .real Broadcast.scalarR (Op.full [BM] (Op.const 0)) (Op.const 1.0)) s
      = some (⟨fun _ : TileIndex [BM] => some (1 : ℝ)⟩ : Tile .real [BM]) := by
  simp only [evalOp_add, evalOp_full, evalOp_const, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext r
  simp [Tile.bop_data, NumericDType.add]
  norm_num

/-- `acc` init eval: `tl.zeros` → the all-`0` tile (matches `oP … 0`). -/
theorem acc_init_eval (s : BlockState) (BM BD : Nat) :
    evalOp (Op.full [BM, BD] (Op.const 0)) s
      = some (⟨fun _ : TileIndex [BM, BD] => some (0 : ℝ)⟩ : Tile .real [BM, BD]) := by
  simp [evalOp_full, evalOp_const, Option.bind]

/-- **Worked statement-eval template** (the `k_mask` boundary∧head boolean), for a
block with `offs_n[j] = j`, `start_n = SN`. Demonstrates the end-to-end recipe on
the surface's actual `boolAnd`/`lt`/`expandDim`/`arange`/`sub` Op shapes:
`k_mask[e,j] = (j < N_CTX − start_n) ∧ (e < HEAD_ACTIVE)`. -/
theorem kmask_eval (s : BlockState) (BD BN NC SN HA : Nat)
    (hoffs : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val)))
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN)) :
    evalOp (Op.boolAnd Broadcast.nil.consR.consL
        (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n"))
          (Op.sub .nat Broadcast.nil (Op.constNat NC) (Op.ref .nat [] "start_n")))
        (Op.expandDim ⟨1, by simp⟩ (Op.lt .nat Broadcast.scalarR (Op.arange BD) (Op.constNat HA)))) s
      = some ⟨fun idx : TileIndex [BD, BN] =>
          (decide (idx.2.1.val < NC - SN) && decide (idx.1.val < HA))⟩ := by
  rw [evalOp_boolAnd]
  conv_lhs => rw [evalOp_lt]; arg 1; rw [evalOp_expandDim_zero_nat, hoffs]
  simp only [Option.bind_eq_bind, Option.bind_some, evalOp_sub, evalOp_constNat, evalOp_ref, hsn]
  conv_lhs => arg 1; unfold evalOp
  simp only [Option.bind_eq_bind, Option.bind_some, Option.pure_def, evalOp_lt, evalOp_arange,
    evalOp_constNat]
  refine congrArg some ?_
  ext idx
  simp [Tile.bop, Tile.cop, Tile.expandDim, Tile.vec, ComparableDType.lt, NumericDType.sub]

/-- **`v`-load mask eval** (`(offs_n[:,None] < N_CTX − start_n) ∧ (arange < HEAD_ACTIVE)[None,:]`,
shape `[BLOCK_N, BLOCK_DMODEL]`): `v_mask[j,d] = (j < N_CTX − start_n) ∧ (d < HEAD_ACTIVE)`.
The `v` analogue of `kmask_eval` (axes swapped: boundary on `offs_n` rows, head on cols). -/
theorem vmask_eval (s : BlockState) (BN BD NC SN HA : Nat)
    (hoffs : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val)))
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN)) :
    evalOp (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BN] "offs_n"))
          (Op.sub .nat Broadcast.nil (Op.constNat NC) (Op.ref .nat [] "start_n")))
        (Op.expandDim ⟨0, by simp⟩ (Op.lt .nat Broadcast.scalarR (Op.arange BD) (Op.constNat HA)))) s
      = some ⟨fun idx : TileIndex [BN, BD] =>
          (decide (idx.1.val < NC - SN) && decide (idx.2.1.val < HA))⟩ := by
  rw [evalOp_boolAnd]
  conv_lhs => rw [evalOp_lt]; arg 1; rw [evalOp_expandDim_one_nat, hoffs]
  simp only [Option.bind_eq_bind, Option.bind_some, evalOp_sub, evalOp_constNat, evalOp_ref, hsn]
  conv_lhs => arg 1; unfold evalOp
  simp only [Option.bind_eq_bind, Option.bind_some, Option.pure_def, evalOp_lt, evalOp_arange,
    evalOp_constNat]
  refine congrArg some ?_
  ext idx
  simp [Tile.bop, Tile.cop, Tile.expandDim, Tile.vec, ComparableDType.lt, NumericDType.sub]

/-- **Head-masked dot reduction** (the crux tile-op of the step lemma). When the
contraction-axis lanes `e ≥ HEAD_ACTIVE` of both operands are masked to `some 0`
(which is what the kernel's `q`/`k` masked loads produce on a clean state,
`undef = 0`), `tl.dot q k` at `(r, jL)` collapses from the full `Fin BD` sum to
the active `Fin HA` sum — i.e. the unscaled score `rawScore`. The `WithBot`
products are all `some`, so the sum coerces out; `sum_padHeadD_eq` drops the
inactive lanes. -/
theorem dot_headMasked {BD HA N M : Nat} (hHA : HA ≤ BD)
    (f : Fin M → Fin BD → ℝ) (g : Fin BD → Fin N → ℝ)
    (r : Fin M) (jL : Fin N)
    (q : Tile .real [M, BD]) (k : Tile .real [BD, N])
    (hq : ∀ e : Fin BD, q.data (r, e, PUnit.unit) = some (if e.val < HA then f r e else 0))
    (hk : ∀ e : Fin BD, k.data (e, jL, PUnit.unit) = some (if e.val < HA then g e jL else 0)) :
    (Tile.dot [] q k).data (r, jL, PUnit.unit)
      = some (Finset.univ.sum (fun e : Fin HA =>
          f r (Fin.castLE hHA e) * g (Fin.castLE hHA e) jL)) := by
  rw [Tile.dot_nil_data]
  have hterm : (fun e : Fin BD =>
        Option.map₂ (· * ·) (q.data (r, e, PUnit.unit)) (k.data (e, jL, PUnit.unit)))
      = (fun e : Fin BD => ((if e.val < HA then f r e * g e jL else 0 : ℝ) : WithBot ℝ)) := by
    funext e
    rw [hq e, hk e]
    by_cases he : e.val < HA
    · simp only [if_pos he]; rfl
    · simp only [if_neg he, Option.map₂, Option.bind, Option.map, mul_zero]; rfl
  rw [hterm, ← WithBot.coe_sum]
  congr 1
  rw [show (fun e : Fin BD => if e.val < HA then f r e * g e jL else 0)
        = (fun e : Fin BD => if h : e.val < HA then
            (fun d : Fin HA => f r (Fin.castLE hHA d) * g (Fin.castLE hHA d) jL) ⟨e.val, h⟩ else 0) from ?_]
  · exact sum_padHeadD_eq hHA (fun d : Fin HA => f r (Fin.castLE hHA d) * g (Fin.castLE hHA d) jL)
  · funext e
    by_cases he : e.val < HA
    · have hcast : Fin.castLE hHA (⟨e.val, he⟩ : Fin HA) = e := Fin.ext (by simp)
      simp only [if_pos he, dif_pos he, hcast]
    · simp [he]

/-- **reduceSum over axis 1** (the `l_ij = tl.sum(p, 1)` / accumulator column
reductions). An all-`some` `[M,N]` tile row-reduces to `some` of the real row
sum. -/
theorem reduceSum_some {M N : Nat} (h : Fin M → Fin N → ℝ) (x : Tile .real [M, N]) (r : Fin M)
    (hx : ∀ jL : Fin N, x.data (r, jL, PUnit.unit) = some (h r jL)) :
    (Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [M,N].length) x).data (r, PUnit.unit)
      = some (Finset.univ.sum (fun jL : Fin N => h r jL)) := by
  rw [Tile.reduceSumDrop_data]
  have key : ∀ k : Fin N, x.data
      (TileShape.insertAxisIndex [M, N] (⟨1, by simp⟩ : Fin [M,N].length) (r, PUnit.unit) k)
      = ((h r k : ℝ) : WithBot ℝ) := fun k => by
    rw [show (TileShape.insertAxisIndex [M, N] (⟨1, by simp⟩ : Fin [M,N].length) (r, PUnit.unit) k)
          = (r, k, PUnit.unit) from rfl]; exact hx k
  calc (Finset.univ.sum (fun k : Fin N => x.data
          (TileShape.insertAxisIndex [M, N] (⟨1, by simp⟩ : Fin [M,N].length) (r, PUnit.unit) k)))
      = Finset.univ.sum (fun k : Fin N => ((h r k : ℝ) : WithBot ℝ)) :=
        Finset.sum_congr rfl (fun k _ => key k)
    _ = some (Finset.univ.sum (fun jL : Fin N => h r jL)) := by rw [← WithBot.coe_sum]; rfl

/-- **exp2 elementwise** (`p = tl.math.exp2(qk)`, `alpha = exp2(m_i - m_ij)`).
`WithBot.realExp2` on a `some` cell is `some (pow2 ·)`. -/
theorem exp2_some {M N : Nat} (h : Fin M → Fin N → ℝ) (x : Tile .real [M, N])
    (r : Fin M) (jL : Fin N) (hx : x.data (r, jL, PUnit.unit) = some (h r jL)) :
    (Tile.uop WithBot.realExp2 x).data (r, jL, PUnit.unit) = some (pow2 (h r jL)) := by
  show WithBot.realExp2 (x.data (r, jL, PUnit.unit)) = _
  rw [hx]; simp [WithBot.realExp2, pow2, mul_comm]

/-- `WithBot` sup of coerced reals over a nonempty index = `some` of the real
`sup'`. Bridges the kernel's `tl.max(qk, 1)` (`reduceMaxDrop` uses `Finset.sup'`)
to `mP`'s block-max (`Finset.sup` of coerced per-key scores). -/
theorem sup_coe_eq {N : Nat} (hN : 0 < N) (g : Fin N → ℝ) :
    Finset.univ.sup (fun i : Fin N => ((g i : ℝ) : WithBot ℝ))
      = some (Finset.univ.sup' ⟨⟨0, hN⟩, Finset.mem_univ _⟩ g) :=
  (Finset.coe_sup' ⟨⟨0, hN⟩, Finset.mem_univ _⟩ g).symm

/-- **Masked pointer-load characterization** (the `k`/`v`/`q` loads). On a clean
state (`undef = 0`), a masked `.ptr` load yields, per lane, `some (readMem …)`
where the mask holds and `some 0` where it doesn't — i.e.
`some (if mask then readMem else 0)`. This is what feeds the `hq`/`hk`
hypotheses of `dot_headMasked` (masked-off head lanes become `some 0`). -/
theorem load_ptr_mask_real {shape : TileShape}
    (ptrOp : Op .ptr shape) (maskOp : Op .bool shape) (s : BlockState)
    (ptrs : Tile .ptr shape) (masks : Tile .bool shape)
    (hp : evalOp ptrOp s = some ptrs) (hm : evalOp maskOp s = some masks)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    evalOp (.load .real (.ptr ptrOp) (.mask maskOp)) s
      = some ⟨fun i => some (if masks.data i then
          s.readMem (ptrs.data i).1 (ptrs.data i).2 else 0)⟩ := by
  simp only [evalOp, hp, hm, Option.bind]
  refine congrArg some ?_
  ext i
  simp only [BlockState.readMemValue_real, hundef]
  cases hmi : masks.data i <;> simp [hmi]

/-- No-mask `.ptr` load (used for `q_scale`): reads `readMem` at each pointer. -/
theorem load_ptr_none_real {shape : TileShape}
    (ptrOp : Op .ptr shape) (s : BlockState) (ptrs : Tile .ptr shape)
    (hp : evalOp ptrOp s = some ptrs) :
    evalOp (.load .real (.ptr ptrOp) .none) s
      = some ⟨fun i => some (s.readMem (ptrs.data i).1 (ptrs.data i).2)⟩ := by
  simp only [evalOp, hp]
  refine congrArg some ?_
  ext i
  simp [BlockState.readMemValue_real]

/-- `castFloat` eval: applies the (model-identity) float cast pointwise. Used for
the `qk` dot-cast and the `p` fp16 round-trip in the loop body. -/
theorem evalOp_castFloat_real {shape : TileShape} (src dst : FloatDType)
    (a : Op src.toTileDType shape) (s : BlockState) (t : Tile src.toTileDType shape)
    (h : evalOp a s = some t) :
    evalOp (.castFloat src dst a) s = some ⟨fun i => src.cast dst (t.data i)⟩ := by
  simp only [evalOp, h]; rfl

/-- **`qk` statement eval** (`(tl.dot q k).to(fp32) · q_scale · k_scale`): given
the `q`/`k` tiles and the scalar `q_scale`/`k_scale`, the `qk` register evaluates
to the scaled (cast-wrapped) dot. The `castFloat .real .real` reduces via a
natural sub-proof then a defeq dtype-coercion (`FloatDType.real.toTileDType`
vs `TileDType.real`) so it matches the `mul`-normalized goal. -/
theorem qk_op_eval (s : BlockState) (BM BN BD : Nat)
    (qtile : Tile .real [BM, BD]) (ktile : Tile .real [BD, BN]) (qsv ksv : ℝ)
    (hq : s.regs .real [BM, BD] "q" = some qtile) (hk : s.regs .real [BD, BN] "k" = some ktile)
    (hqs : s.regs .real [] "q_scale" = some (Tile.scalar (some qsv)))
    (hks : s.regs .real [] "k_scale" = some (Tile.scalar (some ksv))) :
    evalOp (Op.mul .real Broadcast.scalarR
        (Op.mul .real Broadcast.scalarR
          (Op.castFloat .real .real (Op.dot (batch := []) (Op.ref .real [BM, BD] "q") (Op.ref .real [BD, BN] "k")))
          (Op.ref .real [] "q_scale")) (Op.ref .real [] "k_scale")) s
      = some (Tile.bop NumericDType.real.mul Broadcast.scalarR
         (Tile.bop NumericDType.real.mul Broadcast.scalarR
           (⟨fun i => FloatDType.real.cast FloatDType.real ((Tile.dot [] qtile ktile).data i)⟩ : Tile .real [BM,BN])
           (Tile.scalar (some qsv : WithBot ℝ))) (Tile.scalar (some ksv : WithBot ℝ))) := by
  have hdot : evalOp (Op.dot (batch := []) (Op.ref .real [BM, BD] "q") (Op.ref .real [BD, BN] "k")) s
      = some (Tile.dot [] qtile ktile) := by rw [evalOp_dot]; simp [hq, hk]
  have hcast : evalOp (Op.castFloat .real .real (Op.dot (batch := []) (Op.ref .real [BM, BD] "q") (Op.ref .real [BD, BN] "k"))) s
      = some (⟨fun i => FloatDType.real.cast FloatDType.real ((Tile.dot [] qtile ktile).data i)⟩ : Tile .real [BM,BN]) := by
    rw [evalOp_castFloat]; simp only [FloatDType.toTileDType_real]; rw [hdot]; rfl
  have hcast2 : @evalOp TileDType.real [BM, BN]
        (Op.castFloat .real .real (Op.dot (batch := []) (Op.ref .real [BM, BD] "q") (Op.ref .real [BD, BN] "k"))) s
      = some (⟨fun i => FloatDType.real.cast FloatDType.real ((Tile.dot [] qtile ktile).data i)⟩ : Tile .real [BM,BN]) := hcast
  rw [evalOp_mul, evalOp_mul, hcast2]; simp [hqs, hks]

/-- **`m_ij` statement eval** (`tl.maximum(m_i, tl.max(qk, 1))`): the `where`/`gt`
over `m_i` and the `tl.max(qk,1)` row reduction. `reduceMax` has the same
shape-discrimination issue as `castFloat` (its result shape `eraseAxis` blocks
`simp`/`rw` matching), so the reduced row is proven naturally then defeq-coerced
to `[BLOCK_M]`. -/
theorem mij_op_eval (s : BlockState) (BM BN : Nat)
    (mtile : Tile .real [BM]) (qktile : Tile .real [BM, BN]) (rmaxT : Tile .real [BM])
    (hmi : s.regs .real [BM] "m_i" = some mtile)
    (hqk : s.regs .real [BM, BN] "qk" = some qktile)
    (hrm : Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [BM,BN].length) qktile = some rmaxT) :
    evalOp (Op.where
        (Op.gt .real (Broadcast.consSame Broadcast.nil) (Op.ref .real [BM] "m_i")
          (Op.reduceMax (⟨1, by simp⟩ : Fin [BM,BN].length) Bool.false (Op.ref .real [BM, BN] "qk")))
        (Op.ref .real [BM] "m_i")
        (Op.reduceMax (⟨1, by simp⟩ : Fin [BM,BN].length) Bool.false (Op.ref .real [BM, BN] "qk"))) s
      = some (Tile.select (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) mtile rmaxT) mtile rmaxT) := by
  have hrmaxN : evalOp (Op.reduceMax (⟨1, by simp⟩ : Fin [BM,BN].length) Bool.false (Op.ref .real [BM, BN] "qk")) s = some rmaxT := by
    rw [evalOp_reduceMax]; simp only [evalOp_ref, hqk]; exact hrm
  have hrmax : @evalOp TileDType.real [BM]
        (Op.reduceMax (⟨1, by simp⟩ : Fin [BM,BN].length) Bool.false (Op.ref .real [BM, BN] "qk")) s = some rmaxT := hrmaxN
  rw [evalOp_where]
  simp only [evalOp_gt, evalOp_ref, hmi, hrmax, Option.bind_eq_bind, Option.bind_some]

/-- **`p` statement eval** (`tl.math.exp2(qk)`): `Tile.uop realExp2` over `qk`. -/
theorem p_op_eval (s : BlockState) (BM BN : Nat) (qk2tile : Tile .real [BM,BN])
    (hqk : s.regs .real [BM,BN] "qk" = some qk2tile) :
    evalOp (Op.exp2 (Op.ref .real [BM,BN] "qk")) s = some (Tile.uop WithBot.realExp2 qk2tile) := by
  rw [evalOp_exp2]; simp [hqk]

/-- **`alpha` statement eval** (`tl.math.exp2(m_i − m_ij)`). -/
theorem alpha_op_eval (s : BlockState) (BM : Nat) (mi mij : Tile .real [BM])
    (hmi : s.regs .real [BM] "m_i" = some mi) (hmij : s.regs .real [BM] "m_ij" = some mij) :
    evalOp (Op.exp2 (Op.sub .real (Broadcast.consSame Broadcast.nil) (Op.ref .real [BM] "m_i") (Op.ref .real [BM] "m_ij"))) s
      = some (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mi mij)) := by
  rw [evalOp_exp2]; simp [evalOp_sub, hmi, hmij]

/-- **`l_i` statement eval** (`l_i · alpha + l_ij`). -/
theorem li_op_eval (s : BlockState) (BM : Nat) (li alpha lij : Tile .real [BM])
    (hli : s.regs .real [BM] "l_i" = some li) (ha : s.regs .real [BM] "alpha" = some alpha) (hlij : s.regs .real [BM] "l_ij" = some lij) :
    evalOp (Op.add .real (Broadcast.consSame Broadcast.nil)
        (Op.mul .real (Broadcast.consSame Broadcast.nil) (Op.ref .real [BM] "l_i") (Op.ref .real [BM] "alpha"))
        (Op.ref .real [BM] "l_ij")) s
      = some (Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
          (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) li alpha) lij) := by
  rw [evalOp_add]; simp [evalOp_mul, hli, ha, hlij]

/-- **`qk2` statement eval** (`qk − m_ij[:, None]`). `expandDim` has the same
shape-discrimination issue (its result `insertAxis` shape gets normalized to
`[BM,1]` by the `sub`); the operand eval is proven naturally then defeq-coerced.
The axis proof `hax` is a parameter so the recipe unifies with the loop body. -/
theorem qk2_op_eval (s : BlockState) (BM BN : Nat) (hax : 1 < [BM].length.succ)
    (qktile : Tile .real [BM,BN]) (mij : Tile .real [BM])
    (hqk : s.regs .real [BM,BN] "qk" = some qktile) (hmij : s.regs .real [BM] "m_ij" = some mij) :
    evalOp (Op.sub .real (Broadcast.consSame (Broadcast.consR Broadcast.nil)) (Op.ref .real [BM,BN] "qk")
        (Op.expandDim ⟨1, hax⟩ (Op.ref .real [BM] "m_ij"))) s
      = some (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil)) qktile
          (Tile.expandDim ⟨1, hax⟩ mij)) := by
  have hexpN : evalOp (Op.expandDim ⟨1, hax⟩ (Op.ref .real [BM] "m_ij")) s = some (Tile.expandDim ⟨1, hax⟩ mij) := by
    rw [evalOp_expandDim]; simp [hmij]
  have hexp2 : @evalOp TileDType.real [BM, 1] (Op.expandDim ⟨1, hax⟩ (Op.ref .real [BM] "m_ij")) s
      = some (Tile.expandDim ⟨1, hax⟩ mij) := hexpN
  rw [evalOp_sub]
  simp only [evalOp_ref, hqk, hexp2, Option.bind_eq_bind, Option.bind_some]
  rfl

/-- **`l_ij` statement eval** (`tl.sum(p, 1)`): the `reduceSum` row reduction. -/
theorem lij_op_eval (s : BlockState) (BM BN : Nat) (ptile : Tile .real [BM,BN])
    (hp : s.regs .real [BM,BN] "p" = some ptile) :
    evalOp (Op.reduceSum (⟨1, by simp⟩ : Fin [BM,BN].length) Bool.false (Op.ref .real [BM,BN] "p")) s
      = some (Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [BM,BN].length) ptile) := by
  rw [evalOp_reduceSum]; simp only [evalOp_ref, hp, Option.bind_some]; rfl

/-- **`acc`-rescale statement eval** (`acc · alpha[:, None]`). Mirrors `qk2`'s
expandDim coercion. -/
theorem acc1_op_eval (s : BlockState) (BM BD : Nat) (hax : 1 < [BM].length.succ)
    (acctile : Tile .real [BM,BD]) (alpha : Tile .real [BM])
    (hacc : s.regs .real [BM,BD] "acc" = some acctile) (ha : s.regs .real [BM] "alpha" = some alpha) :
    evalOp (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil)) (Op.ref .real [BM,BD] "acc")
        (Op.expandDim ⟨1, hax⟩ (Op.ref .real [BM] "alpha"))) s
      = some (Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil)) acctile
          (Tile.expandDim ⟨1, hax⟩ alpha)) := by
  have hexpN : evalOp (Op.expandDim ⟨1, hax⟩ (Op.ref .real [BM] "alpha")) s = some (Tile.expandDim ⟨1, hax⟩ alpha) := by
    rw [evalOp_expandDim]; simp [ha]
  have hexp2 : @evalOp TileDType.real [BM, 1] (Op.expandDim ⟨1, hax⟩ (Op.ref .real [BM] "alpha")) s
      = some (Tile.expandDim ⟨1, hax⟩ alpha) := hexpN
  rw [evalOp_mul]; simp only [evalOp_ref, hacc, hexp2, Option.bind_eq_bind, Option.bind_some]; rfl

/-- **`p`-to-fp16 statement eval** (`(p).to(tl.float16)`): pointwise float cast. -/
theorem pfp16_op_eval (s : BlockState) (BM BN : Nat) (ptile : Tile .real [BM,BN])
    (hp : s.regs .real [BM,BN] "p" = some ptile) :
    evalOp (Op.castFloat .real .fp16 (Op.ref .real [BM,BN] "p")) s
      = some (⟨fun i => FloatDType.real.cast FloatDType.fp16 (ptile.data i)⟩ : Tile .fp16 [BM,BN]) := by
  rw [evalOp_castFloat]; simp [hp]

/-- **Pointer-advance statement eval** (`K_ptrs += BLOCK_N · HEAD_DIM`, also `V_ptrs`):
`ptrAdd` of the register by the scalar `BLOCK_N · HEAD_DIM`. -/
theorem kptr_adv_eval (s : BlockState) (a b : Nat) (BNv HD : Nat) (kptr : Tile .ptr [a, b])
    (name : RegName) (hkp : s.regs .ptr [a, b] name = some kptr) :
    evalOp (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [a, b] name)
        (Op.mul .nat Broadcast.nil (Op.constNat BNv) (Op.constNat HD))) s
      = some (Tile.ptrAdd Broadcast.scalarR kptr (Tile.scalar (BNv * HD))) := by
  rw [evalOp_ptrAdd]
  simp [evalOp_ref, hkp, evalOp_mul, evalOp_constNat, NumericDType.mul, Tile.bop]

/-- **`K_scale_ptr += 1` statement eval.** -/
theorem ksptr_adv_eval (s : BlockState) (ksptr : Tile .ptr []) (name : RegName)
    (hks : s.regs .ptr [] name = some ksptr) :
    evalOp (Op.ptrAdd Broadcast.nil (Op.ref .ptr [] name) (Op.constNat 1)) s
      = some (Tile.ptrAdd Broadcast.nil ksptr (Tile.scalar 1)) := by
  rw [evalOp_ptrAdd]; simp [evalOp_ref, hks, evalOp_constNat]

/-- **`acc += tl.dot(p, v)` statement eval** (the numerator accumulation). The `p`
fp16 round-trip `castFloat fp16 (castFloat real fp16 p)` reduces to `p` (cast
identity); the `dot` is coerced like `qk`. -/
theorem acc2_op_eval (s : BlockState) (BM BN BD : Nat)
    (acc1tile : Tile .real [BM,BD]) (ptile : Tile .real [BM,BN]) (vtile : Tile .real [BN,BD])
    (hacc : s.regs .real [BM,BD] "acc" = some acc1tile)
    (hpf16 : s.regs .fp16 [BM,BN] "p" = some (⟨fun i => FloatDType.real.cast FloatDType.fp16 (ptile.data i)⟩ : Tile .fp16 [BM,BN]))
    (hv : s.regs .real [BN,BD] "v" = some vtile) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) (Op.ref .real [BM,BD] "acc")
        (Op.dot (batch := []) (Op.castFloat .fp16 .real (Op.ref .fp16 [BM,BN] "p")) (Op.ref .real [BN,BD] "v"))) s
      = some (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) acc1tile
          (Tile.dot [] ptile vtile)) := by
  have hcb : evalOp (Op.castFloat .fp16 .real (Op.ref .fp16 [BM,BN] "p")) s = some ptile := by
    rw [evalOp_castFloat]; simp [hpf16]; ext i; simp [FloatDType.cast]
  have hcb2 : @evalOp TileDType.real [BM,BN] (Op.castFloat .fp16 .real (Op.ref .fp16 [BM,BN] "p")) s = some ptile := hcb
  have hdotN : evalOp (Op.dot (batch := []) (Op.castFloat .fp16 .real (Op.ref .fp16 [BM,BN] "p")) (Op.ref .real [BN,BD] "v")) s
      = some (Tile.dot [] ptile vtile) := by rw [evalOp_dot]; simp [hcb2, hv]
  have hdotN2 : @evalOp TileDType.real [BM,BD]
      (Op.dot (batch := []) (Op.castFloat .fp16 .real (Op.ref .fp16 [BM,BN] "p")) (Op.ref .real [BN,BD] "v")) s
      = some (Tile.dot [] ptile vtile) := hdotN
  rw [evalOp_add]; simp only [evalOp_ref, hacc, hdotN2, Option.bind_eq_bind, Option.bind_some]; rfl

def attention_forward_triton_surface
    (Q K V Q_scale K_scale Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      _stride_kz _stride_kh stride_kn _stride_kk
      _stride_vz _stride_vh _stride_vk _stride_vn
      _stride_oz _stride_oh _stride_om _stride_on
      _Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE _STAGE : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)

  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  qvk_offset = (off_z).to(tl.int64) * $(stride_qz) + (off_h).to(tl.int64) * $(stride_qh)
  vk_offset = qvk_offset // $(stride_qm)
  q_scale_offset = off_hz * tl.cdiv($(N_CTX), $(BLOCK_M))
  k_scale_offset = off_hz * tl.cdiv($(N_CTX), $(BLOCK_N))

  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_n = tl.arange(0, $(BLOCK_N))
  offs_k = tl.arange(0, $(BLOCK_DMODEL))
  Q_ptrs = Q + qvk_offset + offs_m[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk)
  Q_scale_ptr = Q_scale + q_scale_offset + start_m
  K_ptrs = K + qvk_offset + offs_k[:, None] + offs_n[None, :] * $(stride_kn)
  K_scale_ptr = K_scale + k_scale_offset
  V_ptrs = V + qvk_offset + offs_n[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk)
  O_block_ptr = Out + qvk_offset + offs_m[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk)
  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) + 1.0
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  q = tl.load(Q_ptrs,
    mask=(offs_m[:, None] < $(N_CTX)) & (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[None, :])
  q_scale = tl.load(Q_scale_ptr)
  for start_n in range(0, $(N_CTX), $(BLOCK_N)) {
    start_n = tl.multiple_of(start_n, $(BLOCK_N))
    k_mask = (offs_n[None, :] < ($(N_CTX) - start_n)) &
      (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[:, None]
    k = tl.load(K_ptrs, mask=k_mask)
    k_scale = tl.load(K_scale_ptr)
    qk = (tl.dot(q, k)).to(tl.float32) * q_scale * k_scale
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk = qk - m_ij[:, None]
    p = tl.math.exp2(qk)
    l_ij = tl.sum(p, 1)
    alpha = tl.math.exp2(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    v = tl.load(V_ptrs,
      mask=(offs_n[:, None] < ($(N_CTX) - start_n)) &
        (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[None, :])
    p = (p).to(tl.float16)
    acc += tl.dot(p, v, out_dtype=tl.float16)
    m_i = m_ij
    K_ptrs += $(BLOCK_N) * $(HEAD_DIM)
    K_scale_ptr += $(1)
    V_ptrs += $(BLOCK_N) * $(HEAD_DIM)
  }
  acc = acc / l_i[:, None]
  tl.store(O_block_ptr, (acc).to(Out.type.element_ty),
    mask=(offs_m[:, None] < $(N_CTX)) & (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[None, :])
}

/-! ## Layout helpers (general strides; mirroring the bench port's decomposition). -/

/-- Ceiling division `⌈a / b⌉`, matching Triton's `tl.cdiv`. -/
def cdiv (a b : Nat) : Nat := (a + b - 1) / b

/-- Batch/head base offset `qvk_offset = off_z · stride_qz + off_h · stride_qh`,
with `off_z = pid₁ / H`, `off_h = pid₁ % H`. -/
def baseOffset (s : BlockState) (H stride_qz stride_qh : Nat) : Nat :=
  (s.pids 1 / H) * stride_qz + (s.pids 1 % H) * stride_qh

/-- Global query row of local lane `i`: `pid₀ · BLOCK_M + i`. -/
def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val

/-- Output store address (general strides), matching `O_block_ptr`. -/
def outOffset
    (s : BlockState)
    (H stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  baseOffset s H stride_qz stride_qh +
    mIndex s BLOCK_M idx.1 * stride_qm + idx.2.1.val * stride_qk

/-! ## Loaded tiles as functions of memory (general layout).

Contraction / head axis is the `HEAD_ACTIVE` active lanes (masked-off lanes load
`0` and contribute `0` to the dot, so the active sum is the full sum). Under the
contiguity contracts `stride_qm = stride_kn = HEAD_DIM`, head stride `1`, every
loaded element sits at `base + row · HEAD_DIM + col`. -/

noncomputable def qTile (s : BlockState) (Q : RegionName)
    (H stride_qz stride_qh HEAD_DIM BLOCK_M HEAD_ACTIVE : Nat) :
    TileIndex [BLOCK_M, HEAD_ACTIVE] → ℝ :=
  fun (i, e, _) =>
    s.readMem Q (baseOffset s H stride_qz stride_qh + mIndex s BLOCK_M i * HEAD_DIM + e.val)

/-- Row-masked query tile: rows whose global index `mIndex ≥ N_CTX` load as `0`
(the kernel's `q` row-mask `offs_m < N_CTX`). The loop invariant tracks
`mP`/`lP`/`oP` over THIS tile, so the partials match the kernel on every row —
active rows (`mIndex < N_CTX`) coincide with `qTile`, masked rows degrade to the
all-`0`-score values the kernel actually computes. -/
noncomputable def qTileMasked (s : BlockState) (Q : RegionName)
    (H stride_qz stride_qh HEAD_DIM BLOCK_M HEAD_ACTIVE N_CTX : Nat) :
    TileIndex [BLOCK_M, HEAD_ACTIVE] → ℝ :=
  fun (i, e, _) =>
    if mIndex s BLOCK_M i < N_CTX then
      s.readMem Q (baseOffset s H stride_qz stride_qh + mIndex s BLOCK_M i * HEAD_DIM + e.val)
    else 0

/-- On active rows the row-mask is vacuous, so `qTileMasked = qTile` pointwise. -/
theorem qTileMasked_active (s : BlockState) (Q : RegionName)
    (H stride_qz stride_qh HEAD_DIM BLOCK_M HEAD_ACTIVE N_CTX : Nat)
    (idx : TileIndex [BLOCK_M, HEAD_ACTIVE]) (h : mIndex s BLOCK_M idx.1 < N_CTX) :
    qTileMasked s Q H stride_qz stride_qh HEAD_DIM BLOCK_M HEAD_ACTIVE N_CTX idx
      = qTile s Q H stride_qz stride_qh HEAD_DIM BLOCK_M HEAD_ACTIVE idx := by
  obtain ⟨i, e, u⟩ := idx
  simp only [qTileMasked, qTile, if_pos h]

noncomputable def kTile (s : BlockState) (K : RegionName)
    (H stride_qz stride_qh HEAD_DIM S HEAD_ACTIVE : Nat) :
    TileIndex [S, HEAD_ACTIVE] → ℝ :=
  fun (j, e, _) =>
    s.readMem K (baseOffset s H stride_qz stride_qh + j.val * HEAD_DIM + e.val)

noncomputable def vTile (s : BlockState) (V : RegionName)
    (H stride_qz stride_qh HEAD_DIM S HEAD_ACTIVE : Nat) :
    TileIndex [S, HEAD_ACTIVE] → ℝ :=
  fun (j, d, _) =>
    s.readMem V (baseOffset s H stride_qz stride_qh + j.val * HEAD_DIM + d.val)

/-- Per-key scale `q_scale · k_scale[block(j)]`, `block(j) = j / BLOCK_N`.
`q_scale` is read at `off_hz · cdiv(N_CTX, BLOCK_M) + pid₀`; `k_scale[b]` at
`off_hz · cdiv(N_CTX, BLOCK_N) + b`. -/
noncomputable def keyScale (s : BlockState) (Q_scale K_scale : RegionName)
    (N_CTX BLOCK_M BLOCK_N S : Nat) :
    Fin S → ℝ :=
  fun j =>
    s.readMem Q_scale (s.pids 1 * cdiv N_CTX BLOCK_M + s.pids 0) *
      s.readMem K_scale (s.pids 1 * cdiv N_CTX BLOCK_N + j.val / BLOCK_N)

/-! ## Seeded exp2 / per-key-scale streaming partials (WIP foundation)

The exec-side loop invariant tracks the kernel's running `(m_i, l_i, acc)` after
`b` processed key-blocks. The kernel seeds `m_i = ⊥ (-inf)`, `l_i = 1`, `acc = 0`
and updates per block. These are the base-2, per-key-scale analogue of
`VeriTile/Triton/Semantics/StreamingAccumulator.lean`'s base-`e` partials.

Note the `l_i = 1` seed (this kernel, unlike FA-1's `l_i = 0`): it is zeroed by
block 0's `α = pow2(⊥ − m₁) = 0`, so for `b ≥ 1` the partials match a clean
`l`-seed-`0` recurrence. `mP_ne_bot` records that the running max is finite once
`b ≥ 1`, which is what makes the `α`-rescale algebra
`pow2(m_b − m_{b+1}) · pow2(−m_b) = pow2(−m_{b+1})` valid. -/

/-- Unscaled score `Σ_e Q[i,e]·K[j,e]` (contraction over the active head dim). -/
def rawScore {Mq Sk Dh : Nat} (Q : TileIndex [Mq, Dh] → ℝ) (K : TileIndex [Sk, Dh] → ℝ)
    (i : Fin Mq) (j : Fin Sk) : ℝ :=
  Finset.univ.sum (fun e : Fin Dh => Q (i, e, PUnit.unit) * K (j, e, PUnit.unit))

/-- Global key index of local lane `jL` in block `b` (`b < numKVBlocks`). -/
def gkey (BN nB : Nat) (b : Nat) (hb : b < nB) (jL : Fin BN) : Fin (BN * nB) :=
  ⟨b * BN + jL.val, by
    calc b * BN + jL.val < (b + 1) * BN := by ring_nf; omega
      _ ≤ nB * BN := Nat.mul_le_mul_right _ hb
      _ = BN * nB := by ring⟩

/-- Per-key score for output row `i`, key `(b, jL)`: `keyScale · rawScore`. -/
def kscore {Mq Dh : Nat} (BN nB : Nat)
    (Q : TileIndex [Mq, Dh] → ℝ) (K : TileIndex [BN * nB, Dh] → ℝ)
    (keyScale : Fin (BN * nB) → ℝ)
    (i : Fin Mq) (b : Nat) (hb : b < nB) (jL : Fin BN) : ℝ :=
  keyScale (gkey BN nB b hb jL) * rawScore Q K i (gkey BN nB b hb jL)

/-- Running per-row max of per-key scores over the first `c` blocks, seeded `⊥`. -/
noncomputable def mP {Mq Dh : Nat} (BN nB : Nat)
    (Q : TileIndex [Mq, Dh] → ℝ) (K : TileIndex [BN * nB, Dh] → ℝ)
    (keyScale : Fin (BN * nB) → ℝ) (i : Fin Mq) : Nat → WithBot ℝ
  | 0 => (⊥ : WithBot ℝ)
  | c + 1 =>
      if h : c + 1 ≤ nB then
        max (mP BN nB Q K keyScale i c)
          (Finset.univ.sup (fun jL : Fin BN =>
            ((kscore BN nB Q K keyScale i c (by omega) jL : ℝ) : WithBot ℝ)))
      else
        mP BN nB Q K keyScale i c

/-- The running max is finite (`≠ ⊥`) once at least one nonempty block is seen. -/
theorem mP_ne_bot {Mq Dh : Nat} {BN nB : Nat}
    (Q : TileIndex [Mq, Dh] → ℝ) (K : TileIndex [BN * nB, Dh] → ℝ)
    (keyScale : Fin (BN * nB) → ℝ) (hBN : 0 < BN) (i : Fin Mq)
    (c : Nat) (hc1 : 1 ≤ c) (hc : c ≤ nB) :
    mP BN nB Q K keyScale i c ≠ (⊥ : WithBot ℝ) := by
  obtain ⟨c, rfl⟩ : ∃ c', c = c' + 1 := ⟨c - 1, by omega⟩
  simp only [mP, dif_pos hc]
  have hle := Finset.le_sup (f := fun jL : Fin BN =>
      ((kscore BN nB Q K keyScale i c (by omega) jL : ℝ) : WithBot ℝ))
      (Finset.mem_univ (⟨0, hBN⟩ : Fin BN))
  intro hmax
  rw [max_eq_bot] at hmax
  rw [hmax.2] at hle
  exact absurd (le_bot_iff.mp hle) (WithBot.coe_ne_bot)

/-- Real running max (the kernel's `m_i`, with `⊥` read as `0` — only used at
`b ≥ 1` where it is genuinely finite, see `mP_ne_bot`). -/
noncomputable def mR {Mq Dh : Nat} {BN nB : Nat}
    (Q : TileIndex [Mq, Dh] → ℝ) (K : TileIndex [BN * nB, Dh] → ℝ)
    (keyScale : Fin (BN * nB) → ℝ) (i : Fin Mq) (c : Nat) : ℝ :=
  (mP BN nB Q K keyScale i c).unbotD 0

/-- `m`-free running denominator `Σ_{b<c} Σ_jL pow2(kscore b jL)` (no max shift). -/
noncomputable def lFree2 {Mq Dh : Nat} {BN nB : Nat}
    (Q : TileIndex [Mq, Dh] → ℝ) (K : TileIndex [BN * nB, Dh] → ℝ)
    (keyScale : Fin (BN * nB) → ℝ) (i : Fin Mq) : Nat → ℝ
  | 0 => 0
  | c + 1 =>
      if h : c + 1 ≤ nB then
        lFree2 Q K keyScale i c +
          Finset.univ.sum (fun jL : Fin BN => pow2 (kscore BN nB Q K keyScale i c (by omega) jL))
      else lFree2 Q K keyScale i c

/-- Online-softmax rescale multiplier `α = exp2(m_b − m_{b+1})`. -/
noncomputable def alphaP {Mq Dh : Nat} {BN nB : Nat}
    (Q : TileIndex [Mq, Dh] → ℝ) (K : TileIndex [BN * nB, Dh] → ℝ)
    (keyScale : Fin (BN * nB) → ℝ) (i : Fin Mq) (c : Nat) : ℝ :=
  (WithBot.realExp2 (WithBot.realSub (mP BN nB Q K keyScale i c)
    (mP BN nB Q K keyScale i (c + 1)))).unbotD 0

/-- Running per-row denominator `l_i` after `c` blocks (kernel seed `l = 1`). -/
noncomputable def lP {Mq Dh : Nat} {BN nB : Nat}
    (Q : TileIndex [Mq, Dh] → ℝ) (K : TileIndex [BN * nB, Dh] → ℝ)
    (keyScale : Fin (BN * nB) → ℝ) (i : Fin Mq) : Nat → ℝ
  | 0 => 1
  | c + 1 =>
      if h : c + 1 ≤ nB then
        alphaP Q K keyScale i c * lP Q K keyScale i c +
          Finset.univ.sum (fun jL : Fin BN =>
            pow2 (kscore BN nB Q K keyScale i c (by omega) jL - mR Q K keyScale i (c + 1)))
      else lP Q K keyScale i c

theorem mP_eq_coe {Mq Dh : Nat} {BN nB : Nat}
    (Q : TileIndex [Mq, Dh] → ℝ) (K : TileIndex [BN * nB, Dh] → ℝ)
    (keyScale : Fin (BN * nB) → ℝ) (hBN : 0 < BN) (i : Fin Mq)
    (c : Nat) (hc1 : 1 ≤ c) (hc : c ≤ nB) :
    mP BN nB Q K keyScale i c = ((mR Q K keyScale i c : ℝ) : WithBot ℝ) := by
  have hne := mP_ne_bot Q K keyScale hBN i c hc1 hc
  unfold mR
  cases h : mP BN nB Q K keyScale i c with
  | bot => exact absurd h hne
  | coe r => simp

theorem realExp2_unbotD_coe (r : ℝ) :
    (WithBot.realExp2 ((r : ℝ) : WithBot ℝ)).unbotD 0 = pow2 r := by
  simp [pow2, mul_comm]

theorem alphaP_zero {Mq Dh : Nat} {BN nB : Nat}
    (Q : TileIndex [Mq, Dh] → ℝ) (K : TileIndex [BN * nB, Dh] → ℝ)
    (keyScale : Fin (BN * nB) → ℝ) (i : Fin Mq) :
    alphaP Q K keyScale i 0 = 0 := by
  unfold alphaP
  rw [show mP BN nB Q K keyScale i 0 = (⊥ : WithBot ℝ) from rfl,
      WithBot.realSub_bot_left, WithBot.realExp2_bot]
  rfl

theorem alphaP_succ {Mq Dh : Nat} {BN nB : Nat}
    (Q : TileIndex [Mq, Dh] → ℝ) (K : TileIndex [BN * nB, Dh] → ℝ)
    (keyScale : Fin (BN * nB) → ℝ) (hBN : 0 < BN) (i : Fin Mq)
    (c : Nat) (hc1 : 1 ≤ c) (hc : c + 1 ≤ nB) :
    alphaP Q K keyScale i c = pow2 (mR Q K keyScale i c - mR Q K keyScale i (c + 1)) := by
  unfold alphaP
  rw [mP_eq_coe Q K keyScale hBN i c hc1 (by omega),
      mP_eq_coe Q K keyScale hBN i (c + 1) (by omega) hc]
  rw [WithBot.realSub_coe_coe, realExp2_unbotD_coe]

/-- Factor the max-shift out of a per-block denominator sum. -/
theorem block_sum_shift {Mq Dh : Nat} {BN nB : Nat}
    (Q : TileIndex [Mq, Dh] → ℝ) (K : TileIndex [BN * nB, Dh] → ℝ)
    (keyScale : Fin (BN * nB) → ℝ) (i : Fin Mq) (c : Nat) (hc : c < nB) (m : ℝ) :
    Finset.univ.sum (fun jL : Fin BN =>
        pow2 (kscore BN nB Q K keyScale i c (by omega) jL - m))
      = pow2 (-m) * Finset.univ.sum (fun jL : Fin BN =>
          pow2 (kscore BN nB Q K keyScale i c (by omega) jL)) := by
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro jL _
  rw [show kscore BN nB Q K keyScale i c (by omega) jL - m
        = (-m) + kscore BN nB Q K keyScale i c (by omega) jL by ring, pow2_add]

/-- **Denominator consistency.** For `1 ≤ c ≤ nB`, the kernel's running
denominator `l_i` equals `pow2(−m_i)` times the max-free reference sum — the
exp2/per-key-scale analogue of `StreamingAccumulator`'s base-`e` consistency.
The `α`-rescale telescopes the max shift; the `l = 1` seed is killed at `c = 0`
by `alphaP_zero`. -/
theorem lP_eq {Mq Dh : Nat} {BN nB : Nat}
    (Q : TileIndex [Mq, Dh] → ℝ) (K : TileIndex [BN * nB, Dh] → ℝ)
    (keyScale : Fin (BN * nB) → ℝ) (hBN : 0 < BN) (i : Fin Mq) :
    ∀ c, 1 ≤ c → c ≤ nB →
      lP Q K keyScale i c = pow2 (-(mR Q K keyScale i c)) * lFree2 Q K keyScale i c := by
  intro c hc1 hc
  induction c, hc1 using Nat.le_induction with
  | base =>
    have h1 : (1 : Nat) ≤ nB := hc
    simp only [lP, dif_pos h1, alphaP_zero Q K keyScale i, zero_mul, zero_add]
    rw [block_sum_shift Q K keyScale i 0 (by omega)]
    simp only [lFree2, dif_pos h1]
    ring
  | succ c hc1 ih =>
    have hc' : c ≤ nB := by omega
    have ihc := ih hc'
    simp only [lP, dif_pos hc]
    rw [ihc, alphaP_succ Q K keyScale hBN i c hc1 hc,
        block_sum_shift Q K keyScale i c (by omega)]
    simp only [lFree2, dif_pos hc]
    rw [show pow2 (mR Q K keyScale i c - mR Q K keyScale i (c+1))
            * (pow2 (-(mR Q K keyScale i c)) * lFree2 Q K keyScale i c)
          = pow2 ((mR Q K keyScale i c - mR Q K keyScale i (c+1)) + (-(mR Q K keyScale i c)))
              * lFree2 Q K keyScale i c by rw [pow2_add]; ring]
    rw [show (mR Q K keyScale i c - mR Q K keyScale i (c+1)) + (-(mR Q K keyScale i c))
          = -(mR Q K keyScale i (c+1)) by ring]
    ring

/-- `m`-free running numerator `Σ_{b<c} Σ_jL pow2(kscore)·V[gkey, d]`. -/
noncomputable def oFree2 {Mq Dh : Nat} {BN nB : Nat}
    (Q : TileIndex [Mq, Dh] → ℝ) (K V : TileIndex [BN * nB, Dh] → ℝ)
    (keyScale : Fin (BN * nB) → ℝ) (i : Fin Mq) (d : Fin Dh) : Nat → ℝ
  | 0 => 0
  | c + 1 =>
      if h : c + 1 ≤ nB then
        oFree2 Q K V keyScale i d c +
          Finset.univ.sum (fun jL : Fin BN =>
            pow2 (kscore BN nB Q K keyScale i c (by omega) jL)
              * V (gkey BN nB c (by omega) jL, d, PUnit.unit))
      else oFree2 Q K V keyScale i d c

/-- Running per-row unnormalized output `acc[i,d]` after `c` blocks. -/
noncomputable def oP {Mq Dh : Nat} {BN nB : Nat}
    (Q : TileIndex [Mq, Dh] → ℝ) (K V : TileIndex [BN * nB, Dh] → ℝ)
    (keyScale : Fin (BN * nB) → ℝ) (i : Fin Mq) (d : Fin Dh) : Nat → ℝ
  | 0 => 0
  | c + 1 =>
      if h : c + 1 ≤ nB then
        alphaP Q K keyScale i c * oP Q K V keyScale i d c +
          Finset.univ.sum (fun jL : Fin BN =>
            pow2 (kscore BN nB Q K keyScale i c (by omega) jL - mR Q K keyScale i (c + 1))
              * V (gkey BN nB c (by omega) jL, d, PUnit.unit))
      else oP Q K V keyScale i d c

/-- Numerator analogue of `block_sum_shift`. -/
theorem block_sum_shift_v {Mq Dh : Nat} {BN nB : Nat}
    (Q : TileIndex [Mq, Dh] → ℝ) (K V : TileIndex [BN * nB, Dh] → ℝ)
    (keyScale : Fin (BN * nB) → ℝ) (i : Fin Mq) (d : Fin Dh) (c : Nat) (hc : c < nB) (m : ℝ) :
    Finset.univ.sum (fun jL : Fin BN =>
        pow2 (kscore BN nB Q K keyScale i c (by omega) jL - m)
          * V (gkey BN nB c (by omega) jL, d, PUnit.unit))
      = pow2 (-m) * Finset.univ.sum (fun jL : Fin BN =>
          pow2 (kscore BN nB Q K keyScale i c (by omega) jL)
            * V (gkey BN nB c (by omega) jL, d, PUnit.unit)) := by
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro jL _
  rw [show kscore BN nB Q K keyScale i c (by omega) jL - m
        = (-m) + kscore BN nB Q K keyScale i c (by omega) jL by ring, pow2_add]
  ring

/-- **Numerator consistency** (the `acc` analogue of `lP_eq`). -/
theorem oP_eq {Mq Dh : Nat} {BN nB : Nat}
    (Q : TileIndex [Mq, Dh] → ℝ) (K V : TileIndex [BN * nB, Dh] → ℝ)
    (keyScale : Fin (BN * nB) → ℝ) (hBN : 0 < BN) (i : Fin Mq) (d : Fin Dh) :
    ∀ c, 1 ≤ c → c ≤ nB →
      oP Q K V keyScale i d c = pow2 (-(mR Q K keyScale i c)) * oFree2 Q K V keyScale i d c := by
  intro c hc1 hc
  induction c, hc1 using Nat.le_induction with
  | base =>
    have h1 : (1 : Nat) ≤ nB := hc
    simp only [oP, dif_pos h1, alphaP_zero Q K keyScale i, zero_mul, zero_add]
    rw [block_sum_shift_v Q K V keyScale i d 0 (by omega)]
    simp only [oFree2, dif_pos h1]
    ring
  | succ c hc1 ih =>
    have hc' : c ≤ nB := by omega
    have ihc := ih hc'
    simp only [oP, dif_pos hc]
    rw [ihc, alphaP_succ Q K keyScale hBN i c hc1 hc,
        block_sum_shift_v Q K V keyScale i d c (by omega)]
    simp only [oFree2, dif_pos hc]
    rw [show pow2 (mR Q K keyScale i c - mR Q K keyScale i (c+1))
            * (pow2 (-(mR Q K keyScale i c)) * oFree2 Q K V keyScale i d c)
          = pow2 ((mR Q K keyScale i c - mR Q K keyScale i (c+1)) + (-(mR Q K keyScale i c)))
              * oFree2 Q K V keyScale i d c by rw [pow2_add]; ring]
    rw [show (mR Q K keyScale i c - mR Q K keyScale i (c+1)) + (-(mR Q K keyScale i c))
          = -(mR Q K keyScale i (c+1)) by ring]
    ring

/-- **Ratio invariance.** The running max shift cancels in `acc / l_i`
(`pow2` is never zero), so the kernel's normalized output after `c ≥ 1` blocks
equals the max-free `oFree2 / lFree2`. -/
theorem ratioP {Mq Dh : Nat} {BN nB : Nat}
    (Q : TileIndex [Mq, Dh] → ℝ) (K V : TileIndex [BN * nB, Dh] → ℝ)
    (keyScale : Fin (BN * nB) → ℝ) (hBN : 0 < BN) (i : Fin Mq) (d : Fin Dh)
    (c : Nat) (hc1 : 1 ≤ c) (hc : c ≤ nB) :
    oP Q K V keyScale i d c / lP Q K keyScale i c
      = oFree2 Q K V keyScale i d c / lFree2 Q K keyScale i c := by
  rw [oP_eq Q K V keyScale hBN i d c hc1 hc, lP_eq Q K keyScale hBN i c hc1 hc,
      mul_div_mul_left _ _ (ne_of_gt (pow2_pos _))]

/-! ### Closed-form bridge: the all-blocks ratio IS `attentionRealBase2PerKeyScale`

The kernel's loop covers keys block-by-block; `lFree2`/`oFree2` are block-grid
sums. Reindexing the grid to the flat per-key domain (`sum_fin_eq_block_grid`,
with `gkey = cast ∘ finProdFinEquiv`) turns them into the closed-form
denominator/numerator, whose ratio is exactly `attentionRealBase2PerKeyScale`. -/

/-- `gkey b a` (global key `b·BN + a`) is the block-grid reindex element. -/
theorem gkey_eq_cast {BN nB : Nat} (b : Fin nB) (a : Fin BN) :
    gkey BN nB b.val b.isLt a = Fin.cast (Nat.mul_comm nB BN) (finProdFinEquiv (b, a)) := by
  apply Fin.ext; simp [gkey, finProdFinEquiv_apply_val]; ring

theorem lFree2_range {Mq Dh : Nat} {BN nB : Nat}
    (Q : TileIndex [Mq, Dh] → ℝ) (K : TileIndex [BN * nB, Dh] → ℝ)
    (keyScale : Fin (BN * nB) → ℝ) (i : Fin Mq) : ∀ c,
    lFree2 Q K keyScale i c
      = Finset.sum (Finset.range c) (fun b =>
          if h : b < nB then
            Finset.univ.sum (fun jL : Fin BN => pow2 (kscore BN nB Q K keyScale i b h jL))
          else 0) := by
  intro c
  induction c with
  | zero => simp [lFree2]
  | succ c ih =>
    rw [Finset.sum_range_succ, ← ih]
    by_cases h : c + 1 ≤ nB
    · simp only [lFree2, dif_pos h, dif_pos (show c < nB by omega)]
    · simp only [lFree2, dif_neg h, dif_neg (show ¬ c < nB by omega), add_zero]

theorem lFree2_flat {Mq Dh : Nat} {BN nB : Nat}
    (Q : TileIndex [Mq, Dh] → ℝ) (K : TileIndex [BN * nB, Dh] → ℝ)
    (keyScale : Fin (BN * nB) → ℝ) (i : Fin Mq) :
    lFree2 Q K keyScale i nB
      = Finset.univ.sum (fun j : Fin (BN * nB) => pow2 (keyScale j * rawScore Q K i j)) := by
  rw [lFree2_range Q K keyScale i nB,
      ← Fin.sum_univ_eq_sum_range (fun b => if h : b < nB then
            Finset.univ.sum (fun jL : Fin BN => pow2 (kscore BN nB Q K keyScale i b h jL)) else 0) nB,
      sum_fin_eq_block_grid BN nB (fun j => pow2 (keyScale j * rawScore Q K i j))]
  apply Finset.sum_congr rfl; intro b _
  rw [dif_pos b.isLt]; apply Finset.sum_congr rfl; intro a _; rw [← gkey_eq_cast]; rfl

theorem oFree2_range {Mq Dh : Nat} {BN nB : Nat}
    (Q : TileIndex [Mq, Dh] → ℝ) (K V : TileIndex [BN * nB, Dh] → ℝ)
    (keyScale : Fin (BN * nB) → ℝ) (i : Fin Mq) (d : Fin Dh) : ∀ c,
    oFree2 Q K V keyScale i d c
      = Finset.sum (Finset.range c) (fun b =>
          if h : b < nB then
            Finset.univ.sum (fun jL : Fin BN =>
              pow2 (kscore BN nB Q K keyScale i b h jL) * V (gkey BN nB b h jL, d, PUnit.unit))
          else 0) := by
  intro c
  induction c with
  | zero => simp [oFree2]
  | succ c ih =>
    rw [Finset.sum_range_succ, ← ih]
    by_cases h : c + 1 ≤ nB
    · simp only [oFree2, dif_pos h, dif_pos (show c < nB by omega)]
    · simp only [oFree2, dif_neg h, dif_neg (show ¬ c < nB by omega), add_zero]

theorem oFree2_flat {Mq Dh : Nat} {BN nB : Nat}
    (Q : TileIndex [Mq, Dh] → ℝ) (K V : TileIndex [BN * nB, Dh] → ℝ)
    (keyScale : Fin (BN * nB) → ℝ) (i : Fin Mq) (d : Fin Dh) :
    oFree2 Q K V keyScale i d nB
      = Finset.univ.sum (fun j : Fin (BN * nB) =>
          pow2 (keyScale j * rawScore Q K i j) * V (j, d, PUnit.unit)) := by
  rw [oFree2_range Q K V keyScale i d nB,
      ← Fin.sum_univ_eq_sum_range (fun b => if h : b < nB then
            Finset.univ.sum (fun jL : Fin BN =>
              pow2 (kscore BN nB Q K keyScale i b h jL) * V (gkey BN nB b h jL, d, PUnit.unit))
          else 0) nB,
      sum_fin_eq_block_grid BN nB
        (fun j => pow2 (keyScale j * rawScore Q K i j) * V (j, d, PUnit.unit))]
  apply Finset.sum_congr rfl; intro b _
  rw [dif_pos b.isLt]; apply Finset.sum_congr rfl; intro a _; rw [← gkey_eq_cast]; rfl

/-- **The math heart of the closed-form proof.** The kernel's normalized output
`acc / l_i` after all `nB` blocks (`oP / lP`) equals the closed-form base-2,
per-key-scaled attention `attentionRealBase2PerKeyScale`. Combines ratio
invariance (`ratioP`) with the block-grid→flat reindex of `lFree2`/`oFree2`.
Sorry-free; the remaining work is purely the `exec`-side production of `oP`/`lP`
in the `m_i`/`l_i`/`acc` registers (preLoop + step + postLoop). -/
theorem closed_form_eq {Mq Dh : Nat} {BN nB : Nat}
    (Q : TileIndex [Mq, Dh] → ℝ) (K V : TileIndex [BN * nB, Dh] → ℝ)
    (keyScale : Fin (BN * nB) → ℝ) (hBN : 0 < BN) (hnB : 1 ≤ nB) (i : Fin Mq) (d : Fin Dh) :
    oP Q K V keyScale i d nB / lP Q K keyScale i nB
      = attentionRealBase2PerKeyScale Q K V keyScale (i, d, PUnit.unit) := by
  rw [ratioP Q K V keyScale hBN i d nB hnB le_rfl, oFree2_flat, lFree2_flat]
  simp only [attentionRealBase2PerKeyScale, pow2, rawScore]

/-! ## Exec-proof roadmap (Phase 3, multi-session grind)

The compiled body (`(surface …).toAlgKernel.body`, verified via dump) is:

```
preLoop (22 stmts) ++ [Stmt.forRange "start_n" 0 (BLOCK_N*numKVBlocks) BLOCK_N loopBody] ++
  [Stmt.assign … "acc" (acc / l_i), Stmt.store … O_block_ptr (mask)]
```

* **preLoop (22):** start_m, off_hz, off_z(=÷H), off_h(=%H), qvk_offset
  (=off_z·stride_qz+off_h·stride_qh), vk_offset, q_scale_offset(=off_hz·cdiv(N,BLOCK_M)),
  k_scale_offset(=off_hz·cdiv(N,BLOCK_N)), offs_m, offs_n, offs_k, Q_ptrs, Q_scale_ptr,
  K_ptrs, K_scale_ptr, V_ptrs, O_block_ptr, m_i(=full 0 + (-∞)), l_i(=full 0 + 1),
  acc(=full 0), q(=masked load Q_ptrs), q_scale(=load Q_scale_ptr).
* **loopBody (19):** start_n, k_mask, k(=masked load K_ptrs), k_scale(=load K_scale_ptr),
  qk(=castFloat(q·k)·q_scale·k_scale), m_ij(=where(m_i > reduceMax qk, m_i, reduceMax qk)),
  qk(=qk - m_ij), p(=exp2 qk), l_ij(=reduceSum p), alpha(=exp2(m_i - m_ij)),
  l_i(=l_i·alpha + l_ij), acc(=acc·alpha), v(=masked load V_ptrs), p(=castFloat fp16 p),
  acc(=acc + castFloat(p)·v), m_i(=m_ij), K_ptrs += BLOCK_N·HEAD_DIM,
  K_scale_ptr += 1, V_ptrs += BLOCK_N·HEAD_DIM.

**KEY MECHANISM (verified):** `(surface …).toAlgKernel.body` reduces to a literal
25-element list by `rfl` (`body.length = 25` is `rfl`; proof-irrelevance absorbs the
`⟨i,⋯⟩` Fin proofs). So NO 250-line `Stmt` transcription is needed — decompose with
`List.take_append_drop`: `body = body.take 22 ++ body.drop 22` (rfl), and
`body.drop 22` reduces to `Stmt.forRange "start_n" 0 (BLOCK_N*numKVBlocks) BLOCK_N
loopBody :: [accAssign, store]`. Use `stepStmts.append_some` to split; `forRange_inv`'s
implicit `{body}` unifies with the concrete `loopBody` automatically. Step the 22
computed prefix assigns via `stepStmt_assign_eq_some` + the `evalOp_*` simp lemmas.

Plan (mirrors `Examples/FlashAttention1/Core` — Forward/PreLoop/Steps):
1. **Body decomposition** via `take 22`/`drop 22` + `stepStmts.append_some` (NO
   transcription — body is rfl-computable, see KEY MECHANISM above).
2. **Invariant `P (k) (s)`** (k = forRange counter = blocks·BLOCK_N): pids + the 14
   loop-invariant regs (start_m … q_scale) fixed; m_i/l_i/acc = `osBlockStep` fold over
   the first `k/BLOCK_N` key-blocks; K_ptrs/K_scale_ptr/V_ptrs advanced `k/BLOCK_N` steps.
3. **preLoop lemma** establishes `P 0` (FA-1 `fa1_preLoopStrided_step` style: `s1…s22` via
   `stepStmt_assign_eq_some`). m_i init -∞, l_i init 1 are zeroed by first block's α (see
   note in `osStep_foldl_eq_batch` — ratio robust to init).
4. **step lemma** `P i s → P (i+BLOCK_N) s'`: ONE loopBody iteration = one `osBlockStep`.
   The monster (cf. `fa1_step_strided`, 2422 lines): unfold tl.dot (`Tile.dot`=exact
   `↑Σ`), reduceMax/reduceSum, exp2 (`WithBot.realExp2`=`pow2`), castFloat (identity, see
   [[floats-modeled-as-exact-reals]]), masked loads → match `osBlockStep`'s block-max +
   single rescale + block sums. Drive loop via `forRange_inv`.
5. **postLoop** `acc/l_i` + masked store; for active lane, `osBlockStep_foldl_eq_batch`
   then `sum_flatten_ofFn_ofFn` + `sum_fin_eq_block_grid` +
   `attentionRealBase2PerKeyScale_eq_streaming` ⟹ the closed form.

ALL math lemmas named above are proved sorry-free in `Math/Attention.lean`.

**VALIDATED STEPPING RECIPE (tested in scratch through 11 prefix stmts):**
Step each prefix assign with an *explicit* threaded state (metavar `s'` confuses simp):
```
rw [stepStmts.cons_some (s' := s1) (by apply stepStmt_assign_eq_some;
      simp [evalOp, tile_elementwise, IntegralDType.floorDiv, IntegralDType.mod,
            NumericDType.add, NumericDType.mul, NumericDType.div, NumericDType.sub, e1]),
    stepStmts.cons_some (s' := s2) (by apply stepStmt_assign_eq_some; simp [evalOp, tile_elementwise, e2, e1]),
    … ]
exact stepStmts.nil
```
Key facts: `(body).take N` defeq-reduces so `show stepStmts [_,_,…] s` exposes the cons
list (no transcription); each `sᵢ` is an explicit nested `setReg` with the value I know
(off_z=pids1/H, qvk_offset=pids1/H·stride_qz+pids1%H·stride_qh, offs_m=vec(pids0·BLOCK_M+i),
…); `simp [evalOp, tile_elementwise]` (the `tile_elementwise` attr bundles
bop/ptrAdd/expandDim/dot/reduceSum/…) resolves refs (setReg_* @[simp]) + reduces tile ops;
scalar results close by defeq (`congr 1`). ptr stmts (Q_ptrs…) use `evalOp_expandDim` +
`Tile.ptrAdd_data`; masked loads use the load clause (`evalOp_load_region_none` + mask). -/

/-- **Phase-3 top-level exec reduction (sorry-free).**

Reduces the closed-form `exec` obligation to the three FA-1-style phase
obligations against an arbitrary loop invariant `P : Nat → BlockState → Prop`
(the counter is the key offset `i = block · BLOCK_N`):

* `hpre`  — the 22-statement prefix runs to some `s0` with `P 0 s0` (preLoop);
* `hPle` + `hstep` — `P` is bounded by the loop `stop` and each `loopBody`
  iteration advances the counter by `BLOCK_N` preserving `P` (the `osBlockStep`
  loop step);
* `hpost` — from `P (BLOCK_N · numKVBlocks)` the `acc /= l_i` + masked store tail
  runs to some final state satisfying the user's `post` (postLoop + math bridge).

The body decomposition (`body = take 22 ++ forRange :: accAssign :: store :: []`,
which holds by `rfl`), the `forRange_inv` loop induction, and the loop-exit
alignment `final = stop` (from the bound `hPle` and `forRange_inv`'s
`stop ≤ final`) are all discharged here. Only `hpre`/`hstep`/`hpost` remain —
those are the genuine multi-thousand-line FA-1 analogues (preLoop, the
`osBlockStep` step, postLoop), tracked in the roadmap above. -/
theorem attention_forward_triton_exec_reduction
    (Q K V Q_scale K_scale Out : RegionName) (s : BlockState)
    (stride_qz stride_qh Z H BLOCK_M BLOCK_N numKVBlocks
      HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE STAGE : Nat)
    (hBN : 0 < BLOCK_N)
    (P : Nat → BlockState → Prop)
    (loopBody : List Stmt) (accAssign storeStmt : Stmt)
    (hbody : (attention_forward_triton_surface Q K V Q_scale K_scale Out
        stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
        stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
        Z H (BLOCK_N * numKVBlocks) HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL
        HEAD_ACTIVE STAGE).toAlgKernel.body
      = (attention_forward_triton_surface Q K V Q_scale K_scale Out
          stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
          stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
          Z H (BLOCK_N * numKVBlocks) HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL
          HEAD_ACTIVE STAGE).toAlgKernel.body.take 22
        ++ (Stmt.forRange "start_n" 0 (BLOCK_N * numKVBlocks) BLOCK_N loopBody
            :: accAssign :: storeStmt :: []))
    (hpre : ∃ s0, stepStmts ((attention_forward_triton_surface Q K V Q_scale K_scale Out
          stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
          stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
          Z H (BLOCK_N * numKVBlocks) HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL
          HEAD_ACTIVE STAGE).toAlgKernel.body.take 22) s = some s0 ∧ P 0 s0)
    (hPle : ∀ i st, P i st → i ≤ BLOCK_N * numKVBlocks)
    (hstep : ∀ i st, i < BLOCK_N * numKVBlocks → P i st →
        ∃ st', stepStmts loopBody (st.setReg "start_n" .nat [] (Tile.scalar i)) = some st'
          ∧ P (i + BLOCK_N) st')
    (post : BlockState → Prop)
    (hpost : ∀ st, P (BLOCK_N * numKVBlocks) st →
        ∃ sfin, stepStmts (accAssign :: storeStmt :: []) st = some sfin ∧ post sfin) :
    ∃ sfin, exec (attention_forward_triton_surface Q K V Q_scale K_scale Out
        stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
        stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
        Z H (BLOCK_N * numKVBlocks) HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL
        HEAD_ACTIVE STAGE).toAlgKernel s = some sfin ∧ post sfin := by
  obtain ⟨s0, hpre_eq, hP0⟩ := hpre
  obtain ⟨final, sLoop, hLoopStmt, hfinal, hPLoop⟩ :=
    forRange_inv (idx := "start_n") (start := 0) (stop := BLOCK_N * numKVBlocks)
      (step := BLOCK_N) (by omega) hP0 hstep
  have hfinalEq : final = BLOCK_N * numKVBlocks :=
    le_antisymm (hPle _ _ hPLoop) hfinal
  subst hfinalEq
  obtain ⟨sfin, hTail, hpostFin⟩ := hpost sLoop hPLoop
  refine ⟨sfin, ?_, hpostFin⟩
  rw [exec, hbody, stepStmts.append_some hpre_eq, stepStmts.cons_some hLoopStmt, hTail]

/-- **Step-lemma math heart**: the masked dot of the loaded `q`/`k` tiles equals
`rawScore` over the row-masked query / key tiles, evaluated at the global key
`gkey(c, jL)`. The `q` row-mask folds into `qTileMasked`; the `k` boundary mask
`jL < N_CTX − c·BLOCK_N` is always satisfied (`gkey < N_CTX`); the head mask is
discharged by `dot_headMasked`. -/
theorem dot_qk_eq (s : BlockState) (Q K : RegionName)
    (H sqz sqh HD BM BN BD HA nB c : Nat)
    (hHA : HA ≤ BD) (hc : c < nB)
    (r : Fin BM) (jL : Fin BN)
    (q : Tile .real [BM, BD]) (k : Tile .real [BD, BN])
    (hq : ∀ (i : Fin BM) (e : Fin BD), q.data (i, e, PUnit.unit)
        = some (if (mIndex s BM i < BN * nB ∧ e.val < HA)
            then s.readMem Q (baseOffset s H sqz sqh + mIndex s BM i * HD + e.val) else 0))
    (hk : ∀ (e : Fin BD) (j : Fin BN), k.data (e, j, PUnit.unit)
        = some (if (j.val < BN * nB - c * BN ∧ e.val < HA)
            then s.readMem K (baseOffset s H sqz sqh + e.val + j.val * HD + c * BN * HD) else 0)) :
    (Tile.dot [] q k).data (r, jL, PUnit.unit)
      = some (rawScore (qTileMasked s Q H sqz sqh HD BM HA (BN * nB))
          (kTile s K H sqz sqh HD (BN * nB) HA) r (gkey BN nB c hc jL)) := by
  have hjLlt : jL.val < BN * nB - c * BN := by
    have : c * BN + BN ≤ BN * nB := by
      calc c * BN + BN = (c + 1) * BN := by ring
        _ ≤ nB * BN := Nat.mul_le_mul_right _ hc
        _ = BN * nB := by ring
    omega
  rw [dot_headMasked hHA
      (fun i e => if mIndex s BM i < BN * nB then s.readMem Q (baseOffset s H sqz sqh + mIndex s BM i * HD + e.val) else 0)
      (fun e j => s.readMem K (baseOffset s H sqz sqh + e.val + j.val * HD + c * BN * HD))
      r jL q k
      (by intro e; rw [hq r e]; by_cases he : e.val < HA <;> by_cases hr : mIndex s BM r < BN * nB <;>
            simp [he, hr])
      (by intro e; rw [hk e jL]; simp [hjLlt])]
  refine congrArg some (Finset.sum_congr rfl (fun e _ => ?_))
  simp only [qTileMasked, kTile, gkey, Fin.val_castLE]
  congr 2
  ring

/-- **`qk = kscore`**: the loop's `qk = (dot q k)·q_scale·k_scale` tile equals the
per-key score `kscore` at `(r, c, jL)`. Combines `dot_qk_eq` with the scale
algebra `keyScale(gkey) = q_scale · k_scale` (using `gkey/BLOCK_N = c`). The
`castFloat .real .real` and the scalar broadcasts are identities. -/
theorem qk_eq (s : BlockState) (Q K Q_scale K_scale : RegionName)
    (H sqz sqh HD BM BN BD HA nB c : Nat)
    (hHA : HA ≤ BD) (hc : c < nB) (hBN : 0 < BN)
    (r : Fin BM) (jL : Fin BN)
    (q : Tile .real [BM, BD]) (k : Tile .real [BD, BN])
    (hq : ∀ (i : Fin BM) (e : Fin BD), q.data (i, e, PUnit.unit)
        = some (if (mIndex s BM i < BN * nB ∧ e.val < HA)
            then s.readMem Q (baseOffset s H sqz sqh + mIndex s BM i * HD + e.val) else 0))
    (hk : ∀ (e : Fin BD) (j : Fin BN), k.data (e, j, PUnit.unit)
        = some (if (j.val < BN * nB - c * BN ∧ e.val < HA)
            then s.readMem K (baseOffset s H sqz sqh + e.val + j.val * HD + c * BN * HD) else 0)) :
    (Tile.bop NumericDType.real.mul Broadcast.scalarR
       (Tile.bop NumericDType.real.mul Broadcast.scalarR
         (⟨fun i => FloatDType.real.cast FloatDType.real ((Tile.dot [] q k).data i)⟩ : Tile .real [BM,BN])
         (Tile.scalar (some (s.readMem Q_scale (s.pids 1 * cdiv (BN * nB) BM + s.pids 0)) : WithBot ℝ)))
       (Tile.scalar (some (s.readMem K_scale (s.pids 1 * cdiv (BN * nB) BN + c)) : WithBot ℝ))).data (r, jL, PUnit.unit)
      = some (kscore BN nB (qTileMasked s Q H sqz sqh HD BM HA (BN * nB))
          (kTile s K H sqz sqh HD (BN * nB) HA)
          (keyScale s Q_scale K_scale (BN * nB) BM BN (BN * nB)) r c hc jL) := by
  have hdot := dot_qk_eq s Q K H sqz sqh HD BM BN BD HA nB c hHA hc r jL q k hq hk
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex,
    FloatDType.cast, FloatDType.real_ofWithBot, FloatDType.real_toWithBot, hdot,
    NumericDType.mul, WithBot.realMul, Tile.scalar_data, Option.map₂]
  refine congrArg some ?_
  have hgkeydiv : (c * BN + jL.val) / BN = c := by
    rw [Nat.mul_comm, Nat.add_comm, Nat.add_mul_div_left _ _ hBN, Nat.div_eq_of_lt jL.isLt, Nat.zero_add]
  unfold kscore keyScale gkey
  simp only [hgkeydiv]
  ring

/-- `reduceMaxDrop` of a row whose cells are `g jL` = `Finset.sup` of the `g` —
the `tl.max(qk, 1)` row reduction as a `WithBot` `sup`. -/
theorem reduceMaxDrop_data_row (BM BN : Nat) (hBN : 0 < BN) (qk : Tile .real [BM, BN])
    (rmaxT : Tile .real [BM]) (hrm : Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [BM,BN].length) qk = some rmaxT)
    (r : Fin BM) (g : Fin BN → WithBot ℝ) (hqk : ∀ jL : Fin BN, qk.data (r, jL, PUnit.unit) = g jL) :
    rmaxT.data (r, PUnit.unit) = Finset.univ.sup g := by
  unfold Tile.reduceMaxDrop at hrm
  rw [dif_pos (show 0 < TileShape.axisDim [BM,BN] (⟨1, by simp⟩ : Fin [BM,BN].length) from hBN)] at hrm
  rw [← Option.some.inj hrm]
  simp only [Finset.sup'_eq_sup]
  exact Finset.sup_congr rfl (fun jL _ => hqk jL)

/-- **`m_ij = mP(c+1)`**: the `tl.maximum(m_i, tl.max(qk,1))` register, given
`m_i = mP c` and the reduced row = `Finset.sup` of the block's `kscore`s, equals
`mP` at `c+1` (the `mP`-succ recurrence; `where`/`gt` collapse to WithBot `max`). -/
theorem mij_eq (BM BN nB c : Nat) (hc : c < nB)
    {Dh : Nat} (qT : TileIndex [BM,Dh]→ℝ) (kT : TileIndex [BN*nB,Dh]→ℝ) (kS : Fin (BN*nB)→ℝ)
    (m_i rmaxT : Tile .real [BM]) (r : Fin BM)
    (hmi : m_i.data (r, PUnit.unit) = mP BN nB qT kT kS r c)
    (hrmax : rmaxT.data (r, PUnit.unit)
        = Finset.univ.sup (fun jL : Fin BN => ((kscore BN nB qT kT kS r c hc jL : ℝ) : WithBot ℝ))) :
    (Tile.select (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) m_i rmaxT) m_i rmaxT).data (r, PUnit.unit)
      = mP BN nB qT kT kS r (c+1) := by
  rw [Tile.select_data, Tile.cop_data]
  simp only [Broadcast.leftIndex, Broadcast.rightIndex, ComparableDType.gt, hmi, hrmax]
  rw [mP, dif_pos (by omega : c + 1 ≤ nB)]
  by_cases h : mP BN nB qT kT kS r c ≤ Finset.univ.sup (fun jL : Fin BN => ((kscore BN nB qT kT kS r c hc jL : ℝ) : WithBot ℝ))
  · rw [if_neg (by simp [not_lt.mpr h]), max_eq_right h]
  · rw [if_pos (by simpa using not_le.mp h), max_eq_left (le_of_lt (not_le.mp h))]

/-- `WithBot.realExp2` is total: it never returns `⊥` (`exp2(-inf) = 0`). -/
theorem realExp2_eq_some_unbotD (z : WithBot ℝ) :
    WithBot.realExp2 z = some ((WithBot.realExp2 z).unbotD 0) := by
  cases z <;> rfl

/-- **`alpha = some(alphaP c)`**: the loop's `alpha = exp2(m_i − m_ij)` register
equals the online-softmax rescale `alphaP` at block `c` (`realExp2` total, so the
`unbotD 0` in `alphaP` is exact). -/
theorem alpha_eq (BM BN nB c : Nat)
    {Dh : Nat} (qT : TileIndex [BM,Dh]→ℝ) (kT : TileIndex [BN*nB,Dh]→ℝ) (kS : Fin (BN*nB)→ℝ)
    (m_i m_ij : Tile .real [BM]) (r : Fin BM)
    (hmi : m_i.data (r, PUnit.unit) = mP BN nB qT kT kS r c)
    (hmij : m_ij.data (r, PUnit.unit) = mP BN nB qT kT kS r (c+1)) :
    (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) m_i m_ij)).data (r, PUnit.unit)
      = some (alphaP qT kT kS r c) := by
  show WithBot.realExp2 _ = _
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, hmi, hmij, NumericDType.sub]
  rw [alphaP]
  exact realExp2_eq_some_unbotD _

/-- A `WithBot ℝ` sum of `some`-valued cells is `some` of the real sum. -/
theorem withBot_sum_some {N : Nat} (g : Fin N → ℝ) :
    @Finset.sum (Fin N) (WithBot ℝ) _ Finset.univ (fun k => (some (g k) : WithBot ℝ))
      = some (Finset.univ.sum g) := by
  show (Finset.univ.sum fun k => ((g k : ℝ) : WithBot ℝ)) = ((Finset.univ.sum g : ℝ) : WithBot ℝ)
  exact (WithBot.coe_sum Finset.univ g).symm

/-- The `p · v` dot row (no head mask — contraction over all `BLOCK_N` keys):
`(dot p v)[r,d] = Σ_jL p[r,jL]·v[jL,d]` when both are all-`some`. -/
theorem dot_pv (BM BN BD : Nat) (p : Tile .real [BM,BN]) (v : Tile .real [BN,BD])
    (r : Fin BM) (d : Fin BD) (fp fv : Fin BN → ℝ)
    (hp : ∀ jL : Fin BN, p.data (r, jL, PUnit.unit) = some (fp jL))
    (hv : ∀ jL : Fin BN, v.data (jL, d, PUnit.unit) = some (fv jL)) :
    (Tile.dot [] p v).data (r, d, PUnit.unit) = some (Finset.univ.sum fun jL : Fin BN => fp jL * fv jL) := by
  rw [Tile.dot_nil_data]
  rw [show (@Finset.sum (Fin BN) (WithBot ℝ) _ Finset.univ
        (fun k => Option.map₂ (· * ·) (p.data (r, k, PUnit.unit)) (v.data (k, d, PUnit.unit))))
      = @Finset.sum (Fin BN) (WithBot ℝ) _ Finset.univ (fun k => (some (fp k * fv k) : WithBot ℝ))
      from Finset.sum_congr rfl (fun k _ => by rw [hp k, hv k]; rfl)]
  exact withBot_sum_some _

set_option maxHeartbeats 1000000 in
/-- **`l_i = lP(c+1)`**: the loop's `l_i = l_i·alpha + tl.sum(exp2(qk − m_ij), 1)`
register equals the online denominator `lP` at `c+1` (the `lP`-succ recurrence):
`alpha = alphaP c`, the summed `p`-row = `Σ pow2(kscore − mR(c+1))`. -/
theorem li_eq (BM BN nB c : Nat) (hBN : 0 < BN) (hc : c < nB)
    {Dh : Nat} (qT : TileIndex [BM,Dh]→ℝ) (kT : TileIndex [BN*nB,Dh]→ℝ) (kS : Fin (BN*nB)→ℝ)
    (m_i m_ij l_i : Tile .real [BM]) (qk : Tile .real [BM, BN]) (r : Fin BM)
    (hmi : m_i.data (r, PUnit.unit) = mP BN nB qT kT kS r c)
    (hmij : m_ij.data (r, PUnit.unit) = mP BN nB qT kT kS r (c+1))
    (hli : l_i.data (r, PUnit.unit) = some (lP qT kT kS r c))
    (hqk : ∀ jL : Fin BN, qk.data (r, jL, PUnit.unit) = some (kscore BN nB qT kT kS r c hc jL)) :
    (Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
       (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) l_i
         (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) m_i m_ij)))
       (Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [BM,BN].length)
         (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil)) qk
           (Tile.expandDim (⟨1, by simp⟩ : Fin [BM].length.succ) m_ij))))).data (r, PUnit.unit)
      = some (lP qT kT kS r (c+1)) := by
  have hmijReal : m_ij.data (r, PUnit.unit) = some (mR qT kT kS r (c+1)) := by
    rw [hmij, mP_eq_coe qT kT kS hBN r (c+1) (by omega) (by omega)]; rfl
  have halpha := alpha_eq BM BN nB c qT kT kS m_i m_ij r hmi hmij
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex]
  rw [halpha, hli]
  -- reduce the l_ij reduceSum row on the goal's OWN term (variable-axis @[simp] lemmas)
  simp only [Tile.reduceSumDrop_data, Tile.uop_data, Tile.bop_data, Broadcast.leftIndex,
    Broadcast.rightIndex, Tile.expandDim_data, TileShape.dropInsertedIndex,
    TileShape.insertAxisIndex, hqk, hmijReal, NumericDType.sub, WithBot.realSub,
    Option.map₂, Option.bind, Option.map, WithBot.realExp2_some]
  simp only [withBot_sum_some]
  rw [lP, dif_pos (by omega : c + 1 ≤ nB)]
  simp only [NumericDType.add, NumericDType.mul, WithBot.realAdd, WithBot.realMul,
    Option.map₂, Option.bind, Option.map]
  refine congrArg some ?_
  rw [mul_comm (lP qT kT kS r c)]
  congr 1
  exact Finset.sum_congr rfl (fun k _ => by rw [pow2, mul_comm])

set_option maxHeartbeats 1000000 in
/-- **`acc = oP(c+1)`**: the loop's `acc = acc·alpha + tl.dot(p, v)` register equals
the online numerator `oP` at `c+1` (the `oP`-succ recurrence). Head-active lanes
(`d < HEAD_ACTIVE`) get the real recurrence; inactive lanes stay `0` (the `v`
head-mask zeroes the column). The `p` fp16 round-trip is identity in the model. -/
theorem acc_eq (BM BN BD nB c HA : Nat) (hc : c < nB)
    (qT : TileIndex [BM,HA]→ℝ) (kT vT : TileIndex [BN*nB,HA]→ℝ) (kS : Fin (BN*nB)→ℝ)
    (acc : Tile .real [BM,BD]) (alpha : Tile .real [BM]) (p : Tile .real [BM,BN]) (v : Tile .real [BN,BD])
    (r : Fin BM) (d : Fin BD)
    (hacc : acc.data (r, d, PUnit.unit) = if h : d.val < HA then some (oP qT kT vT kS r ⟨d.val,h⟩ c) else some 0)
    (halpha : alpha.data (r, PUnit.unit) = some (alphaP qT kT kS r c))
    (hp : ∀ jL : Fin BN, p.data (r, jL, PUnit.unit) = some (pow2 (kscore BN nB qT kT kS r c hc jL - mR qT kT kS r (c+1))))
    (hv : ∀ jL : Fin BN, v.data (jL, d, PUnit.unit)
        = some (if h : d.val < HA then vT (gkey BN nB c hc jL, ⟨d.val,h⟩, PUnit.unit) else 0)) :
    (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
       (Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil)) acc
         (Tile.expandDim (⟨1, by simp⟩ : Fin [BM].length.succ) alpha))
       (Tile.dot [] p v)).data (r, d, PUnit.unit)
      = if h : d.val < HA then some (oP qT kT vT kS r ⟨d.val,h⟩ (c+1)) else some 0 := by
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, Tile.expandDim_data,
    TileShape.dropInsertedIndex, halpha]
  by_cases h : d.val < HA
  · rw [dot_pv BM BN BD p v r d (fun jL => pow2 (kscore BN nB qT kT kS r c hc jL - mR qT kT kS r (c+1)))
        (fun jL => vT (gkey BN nB c hc jL, ⟨d.val,h⟩, PUnit.unit)) hp (fun jL => by rw [hv jL, dif_pos h])]
    rw [hacc, dif_pos h, dif_pos h]
    rw [oP, dif_pos (by omega : c + 1 ≤ nB)]
    simp only [NumericDType.add, NumericDType.mul, WithBot.realAdd, WithBot.realMul,
      Option.map₂, Option.bind, Option.map]
    refine congrArg some ?_
    rw [mul_comm (oP qT kT vT kS r ⟨d.val,h⟩ c)]
  · rw [dot_pv BM BN BD p v r d (fun jL => pow2 (kscore BN nB qT kT kS r c hc jL - mR qT kT kS r (c+1)))
        (fun _ => 0) hp (fun jL => by rw [hv jL, dif_neg h])]
    rw [hacc, dif_neg h, dif_neg h]
    simp [NumericDType.add, NumericDType.mul, WithBot.realAdd, WithBot.realMul]

/-- **Loop invariant for the exec proof** (counter `i = block c · BLOCK_N`).

Captures the full register state after the `c`-th key block: program ids and the
loop-invariant registers (`q` loaded under the head/row mask, `q_scale`, `offs_*`,
`O_block_ptr`) fixed; `m_i`/`l_i`/`acc` equal the seeded streaming partials
`mP`/`lP`/`oP` over the first `c` blocks; and the `K_ptrs`/`K_scale_ptr`/`V_ptrs`
pointers advanced by `c` steps. `preLoop` establishes `P 0`, `step` advances `P`
by one block, and `postLoop` reads the closed form off `P numKVBlocks` via
`closed_form_eq`. (Strides specialized to the closed-form contract:
`stride_qm = stride_kn = HEAD_DIM`, head stride `1`.) -/
noncomputable def attnInvariant
    (Q K V Q_scale K_scale Out : RegionName) (s0 : BlockState)
    (stride_qz stride_qh H BLOCK_M BLOCK_N numKVBlocks HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE : Nat)
    (i : Nat) (s : BlockState) : Prop :=
  let nB := numKVBlocks; let c := i / BLOCK_N; let N_CTX := BLOCK_N * nB
  let base := baseOffset s0 H stride_qz stride_qh
  let qT := qTileMasked s0 Q H stride_qz stride_qh HEAD_DIM BLOCK_M HEAD_ACTIVE N_CTX
  let kT := kTile s0 K H stride_qz stride_qh HEAD_DIM N_CTX HEAD_ACTIVE
  let vT := vTile s0 V H stride_qz stride_qh HEAD_DIM N_CTX HEAD_ACTIVE
  let kS := keyScale s0 Q_scale K_scale N_CTX BLOCK_M BLOCK_N N_CTX
  s.pids = s0.pids ∧ i = c * BLOCK_N ∧ c ≤ nB ∧
  (s.regs .real [BLOCK_M] "m_i" = some ⟨fun r : TileIndex [BLOCK_M] => mP BLOCK_N nB qT kT kS r.1 c⟩) ∧
  (s.regs .real [BLOCK_M] "l_i" = some ⟨fun r : TileIndex [BLOCK_M] => ((lP qT kT kS r.1 c : ℝ) : WithBot ℝ)⟩) ∧
  (s.regs .real [BLOCK_M, BLOCK_DMODEL] "acc" = some ⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        if h : idx.2.1.val < HEAD_ACTIVE then ((oP qT kT vT kS idx.1 ⟨idx.2.1.val, h⟩ c : ℝ) : WithBot ℝ) else some 0⟩) ∧
  (s.regs .real [BLOCK_M, BLOCK_DMODEL] "q" = some ⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        some (if (mIndex s0 BLOCK_M idx.1 < N_CTX ∧ idx.2.1.val < HEAD_ACTIVE)
          then s0.readMem Q (base + mIndex s0 BLOCK_M idx.1 * HEAD_DIM + idx.2.1.val) else 0)⟩) ∧
  (s.regs .real [] "q_scale" = some (Tile.scalar (some (s0.readMem Q_scale (s0.pids 1 * cdiv N_CTX BLOCK_M + s0.pids 0))))) ∧
  (s.regs .nat [BLOCK_N] "offs_n" = some (Tile.vec (fun j : Fin BLOCK_N => j.val))) ∧
  (s.regs .nat [BLOCK_M] "offs_m" = some (Tile.vec (fun r : Fin BLOCK_M => mIndex s0 BLOCK_M r))) ∧
  (s.regs .nat [BLOCK_DMODEL] "offs_k" = some (Tile.vec (fun e : Fin BLOCK_DMODEL => e.val))) ∧
  (s.regs .ptr [BLOCK_M, BLOCK_DMODEL] "O_block_ptr" = some ⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        (Out.cast, base + mIndex s0 BLOCK_M idx.1 * HEAD_DIM + idx.2.1.val)⟩) ∧
  (s.regs .ptr [BLOCK_DMODEL, BLOCK_N] "K_ptrs" = some ⟨fun idx : TileIndex [BLOCK_DMODEL, BLOCK_N] =>
        (K.cast, base + idx.1.val + idx.2.1.val * HEAD_DIM + c * BLOCK_N * HEAD_DIM)⟩) ∧
  (s.regs .ptr [] "K_scale_ptr" = some (Tile.scalar (K_scale.cast, s0.pids 1 * cdiv N_CTX BLOCK_N + c))) ∧
  (s.regs .ptr [BLOCK_N, BLOCK_DMODEL] "V_ptrs" = some ⟨fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
        (V.cast, base + idx.1.val * HEAD_DIM + idx.2.1.val + c * BLOCK_N * HEAD_DIM)⟩) ∧
  (∀ rg o, s.undef rg o = 0) ∧ (s.mem = s0.mem)

/-- **preLoop chunk 1** (statements 0–10): the 8 scalar offsets + the 3 index
vectors. Steps to a state with all the register readbacks the pointer/load/init
chunks need downstream. -/
theorem preLoop_scalars (Q K V Q_scale K_scale Out : RegionName) (s : BlockState)
    (stride_qz stride_qh Z H BLOCK_M BLOCK_N numKVBlocks HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE STAGE : Nat) :
    ∃ s11, stepStmts ((attention_forward_triton_surface Q K V Q_scale K_scale Out
        stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
        stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
        Z H (BLOCK_N * numKVBlocks) HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL
        HEAD_ACTIVE STAGE).toAlgKernel.body.take 11) s = some s11
      ∧ s11.pids = s.pids
      ∧ s11.regs .nat [] "start_m" = some (Tile.scalar (s.pids 0))
      ∧ s11.regs .nat [] "qvk_offset" = some (Tile.scalar (s.pids 1 / H * stride_qz + s.pids 1 % H * stride_qh))
      ∧ s11.regs .nat [] "q_scale_offset" = some (Tile.scalar (s.pids 1 * cdiv (BLOCK_N * numKVBlocks) BLOCK_M))
      ∧ s11.regs .nat [] "k_scale_offset" = some (Tile.scalar (s.pids 1 * cdiv (BLOCK_N * numKVBlocks) BLOCK_N))
      ∧ s11.regs .nat [BLOCK_M] "offs_m" = some (Tile.vec (fun r : Fin BLOCK_M => s.pids 0 * BLOCK_M + r.val))
      ∧ s11.regs .nat [BLOCK_N] "offs_n" = some (Tile.vec (fun j : Fin BLOCK_N => j.val))
      ∧ s11.regs .nat [BLOCK_DMODEL] "offs_k" = some (Tile.vec (fun e : Fin BLOCK_DMODEL => e.val))
      ∧ s11.undef = s.undef
      ∧ s11.mem = s.mem := by
  rw [show ((attention_forward_triton_surface Q K V Q_scale K_scale Out
        stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
        stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
        Z H (BLOCK_N * numKVBlocks) HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL
        HEAD_ACTIVE STAGE).toAlgKernel.body.take 11)
      = [ Stmt.assign .nat [] "start_m" (Op.programId 0),
          Stmt.assign .nat [] "off_hz" (Op.programId 1),
          Stmt.assign .nat [] "off_z" (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat H)),
          Stmt.assign .nat [] "off_h" (Op.mod .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat H)),
          Stmt.assign .nat [] "qvk_offset" (Op.add .nat Broadcast.nil
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_z") (Op.constNat stride_qz))
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_h") (Op.constNat stride_qh))),
          Stmt.assign .nat [] "vk_offset" (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "qvk_offset") (Op.constNat HEAD_DIM)),
          Stmt.assign .nat [] "q_scale_offset" (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz")
              (Op.div .nat Broadcast.nil (Op.sub .nat Broadcast.nil
                (Op.add .nat Broadcast.nil (Op.constNat (BLOCK_N * numKVBlocks)) (Op.constNat BLOCK_M)) (Op.constNat 1)) (Op.constNat BLOCK_M))),
          Stmt.assign .nat [] "k_scale_offset" (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz")
              (Op.div .nat Broadcast.nil (Op.sub .nat Broadcast.nil
                (Op.add .nat Broadcast.nil (Op.constNat (BLOCK_N * numKVBlocks)) (Op.constNat BLOCK_N)) (Op.constNat 1)) (Op.constNat BLOCK_N))),
          Stmt.assign .nat [BLOCK_M] "offs_m" (Op.add .nat Broadcast.scalarL
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat BLOCK_M)) (Op.arange BLOCK_M)),
          Stmt.assign .nat [BLOCK_N] "offs_n" (Op.arange BLOCK_N),
          Stmt.assign .nat [BLOCK_DMODEL] "offs_k" (Op.arange BLOCK_DMODEL) ] from rfl]
  simp [stepStmts, stepStmt, evalOp_floorDiv, evalOp_mod, Option.bind, BlockState.setReg,
    Tile.bop, Tile.vec, NumericDType.add, NumericDType.mul, NumericDType.div, NumericDType.sub, cdiv]

set_option maxHeartbeats 1000000 in
/-- **preLoop** (statements 0–21): from a clean input state (`undef = 0`), the
prologue steps to a state satisfying `attnInvariant … 0` — the base case of the
loop induction (`m_i = ⊥`, `l_i = 1`, `acc = 0`, `q`/`q_scale` loaded, pointers
seeded). -/
theorem preLoop (Q K V Q_scale K_scale Out : RegionName) (s : BlockState)
    (stride_qz stride_qh H BLOCK_M BLOCK_N numKVBlocks HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE STAGE : Nat)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ s', stepStmts ((attention_forward_triton_surface Q K V Q_scale K_scale Out
        stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
        stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
        stride_qz H (BLOCK_N * numKVBlocks) HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL
        HEAD_ACTIVE STAGE).toAlgKernel.body.take 22) s = some s'
      ∧ attnInvariant Q K V Q_scale K_scale Out s
          stride_qz stride_qh H BLOCK_M BLOCK_N numKVBlocks HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE 0 s' := by
  obtain ⟨s11, h11, hpids, hstart, hqvk, hqso, hkso, hm, hn, hk, huf, hmem⟩ :=
    preLoop_scalars Q K V Q_scale K_scale Out s
      stride_qz stride_qh stride_qz H BLOCK_M BLOCK_N numKVBlocks HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE STAGE
  set qvk : Nat := s.pids 1 / H * stride_qz + s.pids 1 % H * stride_qh with hqvk_def
  set gm : Fin BLOCK_M → Nat := fun r => s.pids 0 * BLOCK_M + (r : Nat) with hgm_def
  have hrm : s11.readMem = s.readMem := by funext rg o; simp [BlockState.readMem, hmem]
  rw [show ((attention_forward_triton_surface Q K V Q_scale K_scale Out
        stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
        stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
        stride_qz H (BLOCK_N * numKVBlocks) HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL
        HEAD_ACTIVE STAGE).toAlgKernel.body.take 22)
      = ((attention_forward_triton_surface Q K V Q_scale K_scale Out
        stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
        stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
        stride_qz H (BLOCK_N * numKVBlocks) HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL
        HEAD_ACTIVE STAGE).toAlgKernel.body.take 11) ++
      [ Stmt.assign .ptr [BLOCK_M, BLOCK_DMODEL] "Q_ptrs"
          (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Q)
            (Op.add .nat Broadcast.nil.consL.consR
              (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qvk_offset")
                (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m"))
                  (Op.constNat HEAD_DIM)))
              (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_k"))
                (Op.constNat 1)))),
        Stmt.assign .ptr [] "Q_scale_ptr"
          (Op.ptrAdd Broadcast.nil (Op.ptrBase Q_scale)
            (Op.add .nat Broadcast.nil (Op.ref .nat [] "q_scale_offset") (Op.ref .nat [] "start_m"))),
        Stmt.assign .ptr [BLOCK_DMODEL, BLOCK_N] "K_ptrs"
          (Op.ptrAdd Broadcast.scalarL (Op.ptrBase K)
            (Op.add .nat Broadcast.nil.consL.consR
              (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qvk_offset")
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_k")))
              (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n"))
                (Op.constNat HEAD_DIM)))),
        Stmt.assign .ptr [] "K_scale_ptr"
          (Op.ptrAdd Broadcast.nil (Op.ptrBase K_scale) (Op.ref .nat [] "k_scale_offset")),
        Stmt.assign .ptr [BLOCK_N, BLOCK_DMODEL] "V_ptrs"
          (Op.ptrAdd Broadcast.scalarL (Op.ptrBase V)
            (Op.add .nat Broadcast.nil.consL.consR
              (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qvk_offset")
                (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n")) (Op.constNat HEAD_DIM)))
              (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_k")) (Op.constNat 1)))),
        Stmt.assign .ptr [BLOCK_M, BLOCK_DMODEL] "O_block_ptr"
          (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Out)
            (Op.add .nat Broadcast.nil.consL.consR
              (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qvk_offset")
                (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m"))
                  (Op.constNat HEAD_DIM)))
              (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_k"))
                (Op.constNat 1)))),
        Stmt.assign .real [BLOCK_M] "m_i"
          (Op.add .real Broadcast.scalarR (Op.full [BLOCK_M] (Op.const 0)) Op.negInf),
        Stmt.assign .real [BLOCK_M] "l_i"
          (Op.add .real Broadcast.scalarR (Op.full [BLOCK_M] (Op.const 0)) (Op.const 1.0)),
        Stmt.assign .real [BLOCK_M, BLOCK_DMODEL] "acc" (Op.full [BLOCK_M, BLOCK_DMODEL] (Op.const 0)),
        Stmt.assign .real [BLOCK_M, BLOCK_DMODEL] "q"
          (Op.load .real (.ptr (Op.ref .ptr [BLOCK_M, BLOCK_DMODEL] "Q_ptrs"))
            (.mask (Op.boolAnd Broadcast.nil.consL.consR
              (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m"))
                (Op.constNat (BLOCK_N * numKVBlocks)))
              (Op.expandDim ⟨0, by simp⟩ (Op.lt .nat Broadcast.scalarR (Op.arange BLOCK_DMODEL) (Op.constNat HEAD_ACTIVE)))))),
        Stmt.assign .real [] "q_scale"
          (Op.load .real (.ptr (Op.ref .ptr [] "Q_scale_ptr")) .none) ] from rfl,
      stepStmts.append_some h11,
    stepStmts.cons_some (stepStmt_assign_eq_some
      (qptrs_eval s11 Q BLOCK_M BLOCK_DMODEL HEAD_DIM qvk gm hqvk hm hk)),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (qscaleptr_eval _ Q_scale (s.pids 1 * cdiv (BLOCK_N * numKVBlocks) BLOCK_M) (s.pids 0)
        (by simp [hqso]) (by simp [hstart]))),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (kptrs_eval _ K BLOCK_DMODEL BLOCK_N HEAD_DIM qvk (by simp [hqvk]) (by simp [hk]) (by simp [hn]))),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (kscaleptr_eval _ K_scale (s.pids 1 * cdiv (BLOCK_N * numKVBlocks) BLOCK_N) (by simp [hkso]))),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (vptrs_eval _ V BLOCK_N BLOCK_DMODEL HEAD_DIM qvk (by simp [hqvk]) (by simp [hn]) (by simp [hk]))),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (qptrs_eval _ Out BLOCK_M BLOCK_DMODEL HEAD_DIM qvk gm (by simp [hqvk]) (by simp [hm]) (by simp [hk]))),
    stepStmts.cons_some (stepStmt_assign_eq_some (mi_init_eval _ BLOCK_M)),
    stepStmts.cons_some (stepStmt_assign_eq_some (li_init_eval _ BLOCK_M)),
    stepStmts.cons_some (stepStmt_assign_eq_some (acc_init_eval _ BLOCK_M BLOCK_DMODEL)),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (load_ptr_mask_real (Op.ref .ptr [BLOCK_M, BLOCK_DMODEL] "Q_ptrs") _ _
        (⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => (Q.cast, qvk + gm idx.1 * HEAD_DIM + idx.2.1.val * 1)⟩ : Tile .ptr [BLOCK_M, BLOCK_DMODEL])
        _ (by simp) (qmask_eval _ BLOCK_M BLOCK_DMODEL (BLOCK_N * numKVBlocks) HEAD_ACTIVE gm (by simp [hm]))
        (by intro rg o; simp [huf, hundef]))),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (load_ptr_none_real (Op.ref .ptr [] "Q_scale_ptr")
        _ (Tile.scalar (Q_scale.cast, s.pids 1 * cdiv (BLOCK_N * numKVBlocks) BLOCK_M + s.pids 0))
        (by simp))),
    stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  refine ⟨by simp [hpids], by simp, by simp, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · -- m_i = mP … 0 = ⊥
    simp only [Nat.zero_div, mP]
    simp
  · -- l_i = lP … 0 = 1
    simp only [Nat.zero_div, lP]
    simp
    rfl
  · -- acc = oP … 0 = 0
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx
    by_cases h : idx.2.1.val < HEAD_ACTIVE
    · simp only [dif_pos h, oP, Nat.zero_div]; rfl
    · simp [h]
  · -- q (masked load)
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx
    simp [hrm, baseOffset, hqvk_def, mIndex, hgm_def, Bool.and_eq_true, decide_eq_true_eq, mul_one]
  · -- q_scale
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    simp [hrm]
  · -- offs_n
    simp [hn]
  · -- offs_m
    simp only [BlockState.setReg]
    rw [hm]
    refine congrArg some ?_
    ext r
    simp [Tile.vec, mIndex, hgm_def]
  · -- offs_k
    simp [hk]
  · -- O_block_ptr
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx <;> simp [baseOffset, hqvk_def, mIndex, hgm_def, mul_one]
  · -- K_ptrs
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx <;> simp [baseOffset, hqvk_def, Nat.zero_div]
  · -- K_scale_ptr
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    simp [Nat.zero_div]
  · -- V_ptrs
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx <;> simp [baseOffset, hqvk_def, Nat.zero_div, mul_one]
  · -- undef = 0
    intro rg o; simp [huf, hundef]
  · -- mem = s0.mem
    exact hmem

/-- The 19-statement loop body (`for start_n …`), transcribed. Independent of the
region names (all loads/stores go through register pointers). -/
def attnLoopBody (BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE HEAD_DIM numKVBlocks : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "start_n" (Op.ref .nat [] "start_n"),
    Stmt.assign .bool [BLOCK_DMODEL, BLOCK_N] "k_mask"
      (Op.boolAnd (.consL (.consR .nil))
        (Op.lt .nat .scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n"))
          (Op.sub .nat .nil (Op.constNat (BLOCK_N * numKVBlocks)) (Op.ref .nat [] "start_n")))
        (Op.expandDim ⟨1, by simp⟩ (Op.lt .nat .scalarR (Op.arange BLOCK_DMODEL) (Op.constNat HEAD_ACTIVE)))),
    Stmt.assign .real [BLOCK_DMODEL, BLOCK_N] "k"
      (Op.load .real (.ptr (Op.ref .ptr [BLOCK_DMODEL, BLOCK_N] "K_ptrs"))
        (.mask (Op.ref .bool [BLOCK_DMODEL, BLOCK_N] "k_mask"))),
    Stmt.assign .real [] "k_scale"
      (Op.load .real (.ptr (Op.ref .ptr [] "K_scale_ptr")) .none),
    Stmt.assign .real [BLOCK_M, BLOCK_N] "qk"
      (Op.mul .real .scalarR
        (Op.mul .real .scalarR
          (Op.castFloat .real .real (Op.dot (batch := []) (Op.ref .real [BLOCK_M, BLOCK_DMODEL] "q") (Op.ref .real [BLOCK_DMODEL, BLOCK_N] "k")))
          (Op.ref .real [] "q_scale"))
        (Op.ref .real [] "k_scale")),
    Stmt.assign .real [BLOCK_M] "m_ij"
      (Op.where
        (Op.gt .real (.consSame .nil) (Op.ref .real [BLOCK_M] "m_i")
          (Op.reduceMax ⟨1, by simp⟩ Bool.false (Op.ref .real [BLOCK_M, BLOCK_N] "qk")))
        (Op.ref .real [BLOCK_M] "m_i")
        (Op.reduceMax ⟨1, by simp⟩ Bool.false (Op.ref .real [BLOCK_M, BLOCK_N] "qk"))),
    Stmt.assign .real [BLOCK_M, BLOCK_N] "qk"
      (Op.sub .real (.consSame (.consR .nil)) (Op.ref .real [BLOCK_M, BLOCK_N] "qk")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLOCK_M] "m_ij"))),
    Stmt.assign .real [BLOCK_M, BLOCK_N] "p" (Op.exp2 (Op.ref .real [BLOCK_M, BLOCK_N] "qk")),
    Stmt.assign .real [BLOCK_M] "l_ij" (Op.reduceSum ⟨1, by simp⟩ Bool.false (Op.ref .real [BLOCK_M, BLOCK_N] "p")),
    Stmt.assign .real [BLOCK_M] "alpha"
      (Op.exp2 (Op.sub .real (.consSame .nil) (Op.ref .real [BLOCK_M] "m_i") (Op.ref .real [BLOCK_M] "m_ij"))),
    Stmt.assign .real [BLOCK_M] "l_i"
      (Op.add .real (.consSame .nil)
        (Op.mul .real (.consSame .nil) (Op.ref .real [BLOCK_M] "l_i") (Op.ref .real [BLOCK_M] "alpha"))
        (Op.ref .real [BLOCK_M] "l_ij")),
    Stmt.assign .real [BLOCK_M, BLOCK_DMODEL] "acc"
      (Op.mul .real (.consSame (.consR .nil)) (Op.ref .real [BLOCK_M, BLOCK_DMODEL] "acc")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLOCK_M] "alpha"))),
    Stmt.assign .real [BLOCK_N, BLOCK_DMODEL] "v"
      (Op.load .real (.ptr (Op.ref .ptr [BLOCK_N, BLOCK_DMODEL] "V_ptrs"))
        (.mask (Op.boolAnd (.consR (.consL .nil))
          (Op.lt .nat .scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n"))
            (Op.sub .nat .nil (Op.constNat (BLOCK_N * numKVBlocks)) (Op.ref .nat [] "start_n")))
          (Op.expandDim ⟨0, by simp⟩ (Op.lt .nat .scalarR (Op.arange BLOCK_DMODEL) (Op.constNat HEAD_ACTIVE)))))),
    Stmt.assign .fp16 [BLOCK_M, BLOCK_N] "p" (Op.castFloat .real .fp16 (Op.ref .real [BLOCK_M, BLOCK_N] "p")),
    Stmt.assign .real [BLOCK_M, BLOCK_DMODEL] "acc"
      (Op.add .real (.consSame (.consSame .nil)) (Op.ref .real [BLOCK_M, BLOCK_DMODEL] "acc")
        (Op.dot (batch := []) (Op.castFloat .fp16 .real (Op.ref .fp16 [BLOCK_M, BLOCK_N] "p")) (Op.ref .real [BLOCK_N, BLOCK_DMODEL] "v"))),
    Stmt.assign .real [BLOCK_M] "m_i" (Op.ref .real [BLOCK_M] "m_ij"),
    Stmt.assign .ptr [BLOCK_DMODEL, BLOCK_N] "K_ptrs"
      (Op.ptrAdd .scalarR (Op.ref .ptr [BLOCK_DMODEL, BLOCK_N] "K_ptrs")
        (Op.mul .nat .nil (Op.constNat BLOCK_N) (Op.constNat HEAD_DIM))),
    Stmt.assign .ptr [] "K_scale_ptr"
      (Op.ptrAdd .nil (Op.ref .ptr [] "K_scale_ptr") (Op.constNat 1)),
    Stmt.assign .ptr [BLOCK_N, BLOCK_DMODEL] "V_ptrs"
      (Op.ptrAdd .scalarR (Op.ref .ptr [BLOCK_N, BLOCK_DMODEL] "V_ptrs")
        (Op.mul .nat .nil (Op.constNat BLOCK_N) (Op.constNat HEAD_DIM))) ]

/-- Post-loop `acc /= l_i[:, None]`. -/
def attnAccAssign (BLOCK_M BLOCK_DMODEL : Nat) : Stmt :=
  Stmt.assign .real [BLOCK_M, BLOCK_DMODEL] "acc"
    (Op.div .real (.consSame (.consR .nil)) (Op.ref .real [BLOCK_M, BLOCK_DMODEL] "acc")
      (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLOCK_M] "l_i")))

/-- Post-loop masked store of `acc` to `O_block_ptr`. -/
def attnStoreStmt (BLOCK_M BLOCK_DMODEL BLOCK_N numKVBlocks HEAD_ACTIVE : Nat) : Stmt :=
  Stmt.store .real [BLOCK_M, BLOCK_DMODEL] (.ptr (Op.ref .ptr [BLOCK_M, BLOCK_DMODEL] "O_block_ptr"))
    (Op.ref .real [BLOCK_M, BLOCK_DMODEL] "acc")
    (.mask (Op.boolAnd (.consR (.consL .nil))
      (Op.lt .nat .scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m"))
        (Op.constNat (BLOCK_N * numKVBlocks)))
      (Op.expandDim ⟨0, by simp⟩ (Op.lt .nat .scalarR (Op.arange BLOCK_DMODEL) (Op.constNat HEAD_ACTIVE)))))

/-- Body decomposition: prologue (22) ++ [for-loop, acc/=l_i, store]. By `rfl`. -/
theorem attn_body_split (Q K V Q_scale K_scale Out : RegionName)
    (stride_qz stride_qh Z H BLOCK_M BLOCK_N numKVBlocks HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE STAGE : Nat) :
    (attention_forward_triton_surface Q K V Q_scale K_scale Out
        stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
        stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
        Z H (BLOCK_N * numKVBlocks) HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL
        HEAD_ACTIVE STAGE).toAlgKernel.body
      = (attention_forward_triton_surface Q K V Q_scale K_scale Out
          stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
          stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
          Z H (BLOCK_N * numKVBlocks) HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL
          HEAD_ACTIVE STAGE).toAlgKernel.body.take 22
        ++ (Stmt.forRange "start_n" 0 (BLOCK_N * numKVBlocks) BLOCK_N
              (attnLoopBody BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE HEAD_DIM numKVBlocks)
            :: attnAccAssign BLOCK_M BLOCK_DMODEL
            :: attnStoreStmt BLOCK_M BLOCK_DMODEL BLOCK_N numKVBlocks HEAD_ACTIVE :: []) := by
  rfl

set_option maxHeartbeats 8000000 in
/-- **Loop-body execution chain (validated).** Given the register readbacks the
loop invariant supplies on the iteration-entry state `sin` (with `start_n` already
set to the block offset `SN`), the full 19-statement `attnLoopBody` executes to
some final state. This is the mechanical core of the step lemma: every statement's
`evalOp` is discharged by its dedicated recipe (`kmask_eval`/`load_ptr_*`/
`qk_op_eval`/`mij_op_eval`/`qk2_op_eval`/`p_op_eval`/`lij_op_eval`/`alpha_op_eval`/
`li_op_eval`/`acc1_op_eval`/`vmask_eval`/`pfp16_op_eval`/`acc2_op_eval`/
`kptr_adv_eval`/`ksptr_adv_eval`), threaded through `stepStmts.cons_some`. The
`reduceSum`→`[BM]` and `castFloat`→`fp16` statements need an explicit
`@stepStmt_assign_eq_some` shape/dtype instantiation (their `Op` type indices are
`reduceShape`/`toTileDType`, defeq-but-not-syntactic to the body's literal forms);
`rmaxT` is obtained from `0 < BN` (`reduceMaxDrop` is total on a positive axis). -/
theorem attn_loopBody_steps (BM BN BD NC HA HD : Nat) (hBN : 0 < BN)
    (hax : 1 < [BM].length.succ) (sin : BlockState)
    (SN : TileCarrier .nat) (qsv : ℝ)
    (Kptr : Tile .ptr [BD, BN]) (Ksptr : Tile .ptr []) (Vptr : Tile .ptr [BN, BD])
    (qtile : Tile .real [BM, BD]) (mtile ltile : Tile .real [BM]) (acctile : Tile .real [BM, BD])
    (hoffs : sin.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val)))
    (hsn : sin.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hKp : sin.regs .ptr [BD, BN] "K_ptrs" = some Kptr) (hKs : sin.regs .ptr [] "K_scale_ptr" = some Ksptr)
    (hVp : sin.regs .ptr [BN, BD] "V_ptrs" = some Vptr) (hq : sin.regs .real [BM, BD] "q" = some qtile)
    (hqs : sin.regs .real [] "q_scale" = some (Tile.scalar (some qsv))) (hmi : sin.regs .real [BM] "m_i" = some mtile)
    (hli : sin.regs .real [BM] "l_i" = some ltile) (hacc : sin.regs .real [BM, BD] "acc" = some acctile)
    (hundef : ∀ rg o, sin.undef rg o = 0) :
    ∃ sF, stepStmts (attnLoopBody BM BN BD HA HD NC) (sin) = some sF
      ∧ sF.pids = sin.pids ∧ sF.mem = sin.mem ∧ (∀ rg o, sF.undef rg o = 0)
      ∧ sF.regs .ptr [BD, BN] "K_ptrs" = some (Tile.ptrAdd Broadcast.scalarR Kptr (Tile.scalar (BN * HD)))
      ∧ sF.regs .ptr [] "K_scale_ptr" = some (Tile.ptrAdd Broadcast.nil Ksptr (Tile.scalar 1))
      ∧ sF.regs .ptr [BN, BD] "V_ptrs" = some (Tile.ptrAdd Broadcast.scalarR Vptr (Tile.scalar (BN * HD)))
      ∧ sF.regs .real [BM, BD] "q" = some qtile
      ∧ sF.regs .real [] "q_scale" = some (Tile.scalar (some qsv))
      ∧ sF.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val))
      ∧ sF.regs .nat [BM] "offs_m" = sin.regs .nat [BM] "offs_m"
      ∧ sF.regs .nat [BD] "offs_k" = sin.regs .nat [BD] "offs_k"
      ∧ sF.regs .ptr [BM, BD] "O_block_ptr" = sin.regs .ptr [BM, BD] "O_block_ptr"
      ∧ ∃ (rmaxT mijT alphaT : Tile .real [BM]) (qkT pT : Tile .real [BM, BN]) (lijT : Tile .real [BM])
            (acc1T : Tile .real [BM, BD]) (vloadT : Tile .real [BN, BD]),
          qkT = (Tile.bop NumericDType.real.mul Broadcast.scalarR
                (Tile.bop NumericDType.real.mul Broadcast.scalarR
                  (⟨fun i => FloatDType.real.cast FloatDType.real ((Tile.dot [] qtile
                      (⟨fun i => some (if (⟨fun idx => (decide (idx.2.1.val < BN * NC - SN) && decide (idx.1.val < HA))⟩ : Tile .bool [BD, BN]).data i then sin.readMem (Kptr.data i).1 (Kptr.data i).2 else 0)⟩ : Tile .real [BD, BN])).data i)⟩ : Tile .real [BM,BN])
                  (Tile.scalar (some qsv : WithBot ℝ)))
                (Tile.scalar (some (sin.readMem (Ksptr.data PUnit.unit).1 (Ksptr.data PUnit.unit).2) : WithBot ℝ)))
          ∧ Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [BM,BN].length) qkT = some rmaxT
          ∧ mijT = Tile.select (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) mtile rmaxT) mtile rmaxT
          ∧ alphaT = Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mtile mijT)
          ∧ pT = Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil))
              qkT (Tile.expandDim ⟨1, hax⟩ mijT))
          ∧ lijT = Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [BM,BN].length) pT
          ∧ acc1T = Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil)) acctile (Tile.expandDim ⟨1, hax⟩ alphaT)
          ∧ vloadT = (⟨fun i => some (if (⟨fun idx => (decide (idx.1.val < BN * NC - SN) && decide (idx.2.1.val < HA))⟩ : Tile .bool [BN, BD]).data i then sin.readMem (Vptr.data i).1 (Vptr.data i).2 else 0)⟩ : Tile .real [BN, BD])
          ∧ sF.regs .real [BM] "m_i" = some mijT
          ∧ sF.regs .real [BM] "l_i" = some (Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
              (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) ltile alphaT) lijT)
          ∧ sF.regs .real [BM, BD] "acc" = some (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
              acc1T (Tile.dot [] pT vloadT)) := by
  set kmaskT : Tile .bool [BD, BN] := ⟨fun idx => (decide (idx.2.1.val < BN * NC - SN) && decide (idx.1.val < HA))⟩ with hkm
  set kloadT : Tile .real [BD, BN] := ⟨fun i => some (if kmaskT.data i then sin.readMem (Kptr.data i).1 (Kptr.data i).2 else 0)⟩ with hkl
  set ksv : ℝ := sin.readMem (Ksptr.data PUnit.unit).1 (Ksptr.data PUnit.unit).2 with hksv
  set qkT : Tile .real [BM, BN] := Tile.bop NumericDType.real.mul Broadcast.scalarR
      (Tile.bop NumericDType.real.mul Broadcast.scalarR
        (⟨fun i => FloatDType.real.cast FloatDType.real ((Tile.dot [] qtile kloadT).data i)⟩ : Tile .real [BM,BN])
        (Tile.scalar (some qsv : WithBot ℝ))) (Tile.scalar (some ksv : WithBot ℝ)) with hqk
  obtain ⟨rmaxT, hrm⟩ : ∃ t, Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [BM,BN].length) qkT = some t :=
    ⟨_, by unfold Tile.reduceMaxDrop; rw [dif_pos (show 0 < TileShape.axisDim [BM,BN] (⟨1, by simp⟩ : Fin [BM,BN].length) from hBN)]⟩
  set mijT : Tile .real [BM] := Tile.select (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) mtile rmaxT) mtile rmaxT with hmij
  set qk2T : Tile .real [BM, BN] := Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil)) qkT (Tile.expandDim ⟨1, hax⟩ mijT) with hqk2
  set pT : Tile .real [BM, BN] := Tile.uop WithBot.realExp2 qk2T with hpT
  set lijT : Tile .real [BM] := Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [BM,BN].length) pT with hlij
  set alphaT : Tile .real [BM] := Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mtile mijT) with hal
  set acc1T : Tile .real [BM, BD] := Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil)) acctile (Tile.expandDim ⟨1, hax⟩ alphaT) with hacc1
  set vmaskT : Tile .bool [BN, BD] := ⟨fun idx => (decide (idx.1.val < BN * NC - SN) && decide (idx.2.1.val < HA))⟩ with hvm
  set vloadT : Tile .real [BN, BD] := ⟨fun i => some (if vmaskT.data i then sin.readMem (Vptr.data i).1 (Vptr.data i).2 else 0)⟩ with hvl
  unfold attnLoopBody
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (Op.ref .nat [] "start_n") sin = some (Tile.scalar SN) from by rw [evalOp_ref]; exact hsn))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (kmask_eval _ BD BN (BN * NC) SN HA (by simp [hoffs]) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (load_ptr_mask_real (Op.ref .ptr [BD, BN] "K_ptrs") _ _ Kptr kmaskT (by rw [evalOp_ref]; simp [hKp]) (by rw [evalOp_ref]; simp [hkm]) (by intro rg o; simp [hundef])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (load_ptr_none_real (Op.ref .ptr [] "K_scale_ptr") _ Ksptr (by rw [evalOp_ref]; simp [hKs])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (qk_op_eval _ BM BN BD qtile kloadT qsv ksv (by simp [hq]) (by simp [hkl]) (by simp [hqs]) (by simp [hksv]; ext z; rfl)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (mij_op_eval _ BM BN mtile qkT rmaxT (by simp [hmi]) (by simp [hqk]) hrm))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (qk2_op_eval _ BM BN hax qkT mijT (by simp [hqk]) (by simp [hmij])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (p_op_eval _ BM BN qk2T (by simp [hqk2])))]
  rw [stepStmts.cons_some (@stepStmt_assign_eq_some .real [BM] "l_ij" (Op.reduceSum (⟨1, by simp⟩ : Fin [BM,BN].length) Bool.false (Op.ref .real [BM,BN] "p")) _ lijT (lij_op_eval _ BM BN pT (by simp [hpT])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (alpha_op_eval _ BM mtile mijT (by simp [hmi]) (by simp [hmij])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (li_op_eval _ BM ltile alphaT lijT (by simp [hli]) (by simp [hal]) (by simp [hlij])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (acc1_op_eval _ BM BD hax acctile alphaT (by simp [hacc]) (by simp [hal])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (load_ptr_mask_real (Op.ref .ptr [BN, BD] "V_ptrs") _ _ Vptr vmaskT (by rw [evalOp_ref]; simp [hVp]) (vmask_eval _ BN BD (BN * NC) SN HA (by simp [hoffs]) (by simp)) (by intro rg o; simp [hundef])))]
  rw [stepStmts.cons_some (@stepStmt_assign_eq_some TileDType.fp16 [BM,BN] "p" (Op.castFloat .real .fp16 (Op.ref .real [BM,BN] "p")) _ (⟨fun i => FloatDType.real.cast FloatDType.fp16 (pT.data i)⟩ : Tile .fp16 [BM,BN]) (pfp16_op_eval _ BM BN pT (by simp [hpT])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (acc2_op_eval _ BM BN BD acc1T pT vloadT (by simp [hacc1]) (by simp [hpT]) (by simp [hvl])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (Op.ref .real [BM] "m_ij") _ = some mijT from by rw [evalOp_ref]; simp [hmij]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (kptr_adv_eval _ BD BN BN HD Kptr "K_ptrs" (by simp [hKp])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (ksptr_adv_eval _ Ksptr "K_scale_ptr" (by simp [hKs])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (kptr_adv_eval _ BN BD BN HD Vptr "V_ptrs" (by simp [hVp])))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp
  · funext region offset; simp
  · intro rg o; simp [hundef]
  · simp
  · simp
  · simp
  · simp [hq]
  · simp [hqs]
  · simp [hoffs]
  · simp
  · simp
  · simp
  · exact ⟨rmaxT, mijT, alphaT, qkT, pT, lijT, acc1T, vloadT, rfl, hrm, rfl, rfl, rfl, rfl, rfl, rfl,
      by simp, by simp, by simp⟩

set_option maxHeartbeats 8000000 in
/-- **Step lemma**: the loop body advances the invariant by one key block. Combines
the validated execution chain (`attn_loopBody_steps`) with the streaming-softmax
recurrence bridges (`mij_eq`/`li_eq`/`acc_eq`, fed by `qk_eq` and
`reduceMaxDrop_data_row`). The pointer registers advance `c → c+1`
(`Nat.add_div_right`); the boundary `v`-mask is always satisfied for in-range
blocks (`jL < BN ≤ BN·(nB−c)`), so the `v` load matches `vTile` at the global key. -/
theorem attn_step (Q K V Q_scale K_scale Out : RegionName) (s0 : BlockState)
    (stride_qz stride_qh H BLOCK_M BLOCK_N numKVBlocks HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE : Nat)
    (hBN : 0 < BLOCK_N) (hHA : HEAD_ACTIVE ≤ BLOCK_DMODEL)
    (i : Nat) (s : BlockState) (hilt : i < BLOCK_N * numKVBlocks)
    (hinv : attnInvariant Q K V Q_scale K_scale Out s0 stride_qz stride_qh H BLOCK_M BLOCK_N
        numKVBlocks HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE i s) :
    ∃ s', stepStmts (attnLoopBody BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE HEAD_DIM numKVBlocks)
        (s.setReg "start_n" .nat [] (Tile.scalar i)) = some s'
      ∧ attnInvariant Q K V Q_scale K_scale Out s0 stride_qz stride_qh H BLOCK_M BLOCK_N
          numKVBlocks HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE (i + BLOCK_N) s' := by
  have hax : 1 < [BLOCK_M].length.succ := by simp
  have hc : i / BLOCK_N < numKVBlocks := (Nat.div_lt_iff_lt_mul hBN).mpr (by rw [Nat.mul_comm]; exact hilt)
  have hc1 : (i + BLOCK_N) / BLOCK_N = i / BLOCK_N + 1 := Nat.add_div_right i hBN
  simp only [attnInvariant] at hinv
  obtain ⟨hpids, hi, hcle, hmi, hli, hacc, hq, hqs, hn, hm, hk, hO, hKp, hKs, hVp, hundef, hmem⟩ := hinv
  -- streaming functions (match the invariant's lets)
  set qT := qTileMasked s0 Q H stride_qz stride_qh HEAD_DIM BLOCK_M HEAD_ACTIVE (BLOCK_N * numKVBlocks) with hqTdef
  set kT := kTile s0 K H stride_qz stride_qh HEAD_DIM (BLOCK_N * numKVBlocks) HEAD_ACTIVE with hkTdef
  set vT := vTile s0 V H stride_qz stride_qh HEAD_DIM (BLOCK_N * numKVBlocks) HEAD_ACTIVE with hvTdef
  set kS := keyScale s0 Q_scale K_scale (BLOCK_N * numKVBlocks) BLOCK_M BLOCK_N (BLOCK_N * numKVBlocks) with hkSdef
  obtain ⟨sF, hchain, hpidsF, hmemF, hundefF, hKpF, hKsF, hVpF, hqF, hqsF, hnF, hmF, hkF, hOF,
      rmaxT, mijT, alphaT, qkT, pT, lijT, acc1T, vloadT, hqkTd, hrm, hmijd, halphad, hpTd, hlijd, hacc1d, hvloadd,
      hm_iF, hl_iF, haccF⟩ :=
    attn_loopBody_steps BLOCK_M BLOCK_N BLOCK_DMODEL numKVBlocks HEAD_ACTIVE HEAD_DIM hBN hax
      (s.setReg "start_n" .nat [] (Tile.scalar i)) i
      (s0.readMem Q_scale (s0.pids 1 * cdiv (BLOCK_N * numKVBlocks) BLOCK_M + s0.pids 0))
      (⟨fun idx : TileIndex [BLOCK_DMODEL, BLOCK_N] => (K.cast, baseOffset s0 H stride_qz stride_qh + idx.1.val + idx.2.1.val * HEAD_DIM + i / BLOCK_N * BLOCK_N * HEAD_DIM)⟩)
      (Tile.scalar (K_scale.cast, s0.pids 1 * cdiv (BLOCK_N * numKVBlocks) BLOCK_N + i / BLOCK_N))
      (⟨fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] => (V.cast, baseOffset s0 H stride_qz stride_qh + idx.1.val * HEAD_DIM + idx.2.1.val + i / BLOCK_N * BLOCK_N * HEAD_DIM)⟩)
      (⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => some (if (mIndex s0 BLOCK_M idx.1 < BLOCK_N * numKVBlocks ∧ idx.2.1.val < HEAD_ACTIVE) then s0.readMem Q (baseOffset s0 H stride_qz stride_qh + mIndex s0 BLOCK_M idx.1 * HEAD_DIM + idx.2.1.val) else 0)⟩)
      (⟨fun r : TileIndex [BLOCK_M] => mP BLOCK_N numKVBlocks qT kT kS r.1 (i / BLOCK_N)⟩)
      (⟨fun r : TileIndex [BLOCK_M] => ((lP qT kT kS r.1 (i / BLOCK_N) : ℝ) : WithBot ℝ)⟩)
      (⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => if h : idx.2.1.val < HEAD_ACTIVE then ((oP qT kT vT kS idx.1 ⟨idx.2.1.val, h⟩ (i / BLOCK_N) : ℝ) : WithBot ℝ) else some 0⟩)
      (by simp [hn]) (by simp) (by simp [hKp]) (by simp [hKs]) (by simp [hVp]) (by simp [hq])
      (by simp [hqs]) (by simp [hmi]) (by simp [hli]) (by simp [hacc]) (by intro rg o; simp [hundef])
  refine ⟨sF, hchain, ?_⟩
  -- the loop body reads the (unchanged) memory of `s0`
  have hrmem : ∀ (R : RegionName) (o : Nat),
      (s.setReg "start_n" .nat [] (Tile.scalar i)).readMem R o = s0.readMem R o := by
    intro R o; simp only [BlockState.setReg_readMem]; unfold BlockState.readMem; rw [hmem]
  simp only [hrmem] at hqkTd hvloadd
  -- per-cell `qk = kscore` bridge (feeds m_ij / l_i)
  have hqk : ∀ (r : Fin BLOCK_M) (jL : Fin BLOCK_N),
      qkT.data (r, jL, PUnit.unit) = some (kscore BLOCK_N numKVBlocks qT kT kS r (i / BLOCK_N) hc jL) := by
    intro r jL
    rw [hqkTd]
    refine qk_eq s0 Q K Q_scale K_scale H stride_qz stride_qh HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL
      HEAD_ACTIVE numKVBlocks (i / BLOCK_N) hHA hc hBN r jL _ _ (fun a e => rfl) ?_
    intro e j
    simp only [← hi]
    by_cases hcond : (j.val < BLOCK_N * numKVBlocks - i ∧ e.val < HEAD_ACTIVE)
    · simp only [if_pos hcond, Region.cast_id]
      refine congrArg some ?_
      rw [if_pos (by simp only [Bool.and_eq_true, decide_eq_true_eq]; exact hcond)]
    · simp only [if_neg hcond]
      refine congrArg some ?_
      rw [if_neg (by simp only [Bool.and_eq_true, decide_eq_true_eq]; exact hcond)]
  have hmij_eq : ∀ idx : TileIndex [BLOCK_M], mijT.data idx
      = mP BLOCK_N numKVBlocks qT kT kS idx.1 (i / BLOCK_N + 1) := by
    intro idx
    rw [hmijd]
    refine mij_eq BLOCK_M BLOCK_N numKVBlocks (i / BLOCK_N) hc qT kT kS _ rmaxT idx.1 ?_ ?_
    · rfl
    · exact reduceMaxDrop_data_row BLOCK_M BLOCK_N hBN _ rmaxT hrm idx.1 _ (fun jL => hqk idx.1 jL)
  simp only [attnInvariant]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hpidsF]; exact hpids
  · rw [hc1, Nat.add_one_mul, ← hi]
  · rw [hc1]; omega
  · -- m_i = mP (c+1)
    rw [hm_iF, hc1]; refine congrArg some ?_; ext idx; exact hmij_eq idx
  · -- l_i = lP (c+1)
    rw [hl_iF, hc1]; refine congrArg some ?_; ext idx
    rw [halphad, hlijd, hpTd]
    exact li_eq BLOCK_M BLOCK_N numKVBlocks (i / BLOCK_N) hBN hc qT kT kS _ mijT _ _ idx.1
      rfl (hmij_eq idx) rfl (fun jL => hqk idx.1 jL)
  · -- acc = oP (c+1)
    rw [haccF, hc1]; refine congrArg some ?_; ext idx
    rw [hacc1d, halphad, hpTd, hvloadd]
    refine acc_eq BLOCK_M BLOCK_N BLOCK_DMODEL numKVBlocks (i / BLOCK_N) HEAD_ACTIVE hc qT kT vT kS
      _ _ _ _ idx.1 idx.2.1 ?_ ?_ ?_ ?_
    · rfl
    · exact alpha_eq BLOCK_M BLOCK_N numKVBlocks (i / BLOCK_N) qT kT kS _ mijT idx.1 rfl
        (hmij_eq (idx.1, PUnit.unit))
    · intro jL
      refine exp2_some (fun a b => kscore BLOCK_N numKVBlocks qT kT kS a (i / BLOCK_N) hc b
        - mR qT kT kS a (i / BLOCK_N + 1)) _ idx.1 jL ?_
      simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, Tile.expandDim_data,
        TileShape.dropInsertedIndex, TileShape.insertAxisIndex, hqk idx.1 jL,
        hmij_eq (idx.1, PUnit.unit),
        mP_eq_coe qT kT kS hBN idx.1 (i / BLOCK_N + 1) (Nat.le_add_left 1 (i / BLOCK_N)) hc,
        NumericDType.sub, WithBot.realSub, Option.map₂, Option.bind, Option.map]
    · intro jL
      have hjL : jL.val < BLOCK_N * numKVBlocks - i := by
        have hle : i + BLOCK_N ≤ BLOCK_N * numKVBlocks := by
          rw [hi]
          calc i / BLOCK_N * BLOCK_N + BLOCK_N = (i / BLOCK_N + 1) * BLOCK_N := by ring
            _ ≤ numKVBlocks * BLOCK_N := Nat.mul_le_mul_right _ hc
            _ = BLOCK_N * numKVBlocks := by ring
        have := jL.isLt; omega
      simp only [decide_eq_true_eq.mpr hjL, Bool.true_and, Region.cast_id]
      by_cases hd : idx.2.1.val < HEAD_ACTIVE
      · rw [if_pos (decide_eq_true_eq.mpr hd), dif_pos hd]
        refine congrArg some ?_
        show s0.readMem V _ = vTile s0 V H stride_qz stride_qh HEAD_DIM (BLOCK_N * numKVBlocks)
            HEAD_ACTIVE (gkey BLOCK_N numKVBlocks (i / BLOCK_N) hc jL, ⟨idx.2.1.val, hd⟩, PUnit.unit)
        unfold vTile gkey
        congr 1
        ring
      · rw [if_neg (by simp only [decide_eq_true_eq]; exact hd), dif_neg hd]
  · rw [hqF]
  · rw [hqsF]
  · rw [hnF]
  · rw [hmF]; exact hm
  · rw [hkF]; exact hk
  · rw [hOF]; exact hO
  · -- K_ptrs advances c → c+1
    rw [hKpF, hc1]; refine congrArg some ?_
    ext idx <;> simp only [Tile.ptrAdd_data, Broadcast.leftIndex, Broadcast.rightIndex, Tile.scalar] <;> ring
  · -- K_scale_ptr advances
    rw [hKsF, hc1]; refine congrArg some ?_
    ext idx <;> simp only [Tile.ptrAdd_data, Broadcast.leftIndex, Broadcast.rightIndex, Tile.scalar] <;> ring
  · -- V_ptrs advances
    rw [hVpF, hc1]; refine congrArg some ?_
    ext idx <;> simp only [Tile.ptrAdd_data, Broadcast.leftIndex, Broadcast.rightIndex, Tile.scalar] <;> ring
  · exact hundefF
  · rw [hmemF]; exact hmem

set_option maxHeartbeats 4000000 in
/-- **Post-loop**: after the streaming loop (invariant at `numKVBlocks`), the
`acc /= l_i[:, None]` rescale + masked store writes the normalized accumulator
`oP / lP = attentionRealBase2PerKeyScale` to `Out` at every active lane. The store
is a masked scatter (`scatter_readback_prop_masked_nd_of_true`) over an injective
offset map (active columns `< HEAD_ACTIVE ≤ HEAD_DIM`, so `row·HEAD_DIM + col` is
injective); the `qTileMasked → qTile` swap on the active row is `qTileMasked_active`. -/
theorem attn_postLoop (Q K V Q_scale K_scale Out : RegionName) (s : BlockState)
    (stride_qz stride_qh H BLOCK_M BLOCK_N numKVBlocks HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE : Nat)
    (hBN : 0 < BLOCK_N) (hnB : 1 ≤ numKVBlocks) (hAD : HEAD_ACTIVE ≤ HEAD_DIM)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL])
    (hActive : mIndex s BLOCK_M idx.1 < BLOCK_N * numKVBlocks ∧ idx.2.1.val < HEAD_ACTIVE)
    (st : BlockState)
    (hinv : attnInvariant Q K V Q_scale K_scale Out s stride_qz stride_qh H BLOCK_M BLOCK_N
        numKVBlocks HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE (BLOCK_N * numKVBlocks) st) :
    ∃ sfin, stepStmts (attnAccAssign BLOCK_M BLOCK_DMODEL
        :: attnStoreStmt BLOCK_M BLOCK_DMODEL BLOCK_N numKVBlocks HEAD_ACTIVE :: []) st = some sfin
      ∧ sfin.readMem Out (outOffset s H stride_qz stride_qh HEAD_DIM 1 BLOCK_M BLOCK_DMODEL idx)
        = attentionRealBase2PerKeyScale
            (qTile s Q H stride_qz stride_qh HEAD_DIM BLOCK_M HEAD_ACTIVE)
            (kTile s K H stride_qz stride_qh HEAD_DIM (BLOCK_N * numKVBlocks) HEAD_ACTIVE)
            (vTile s V H stride_qz stride_qh HEAD_DIM (BLOCK_N * numKVBlocks) HEAD_ACTIVE)
            (keyScale s Q_scale K_scale (BLOCK_N * numKVBlocks) BLOCK_M BLOCK_N (BLOCK_N * numKVBlocks))
            (idx.1, ⟨idx.2.1.val, hActive.2⟩, PUnit.unit) := by
  have hax : 1 < [BLOCK_M].length.succ := by simp
  have hcnB : (BLOCK_N * numKVBlocks) / BLOCK_N = numKVBlocks := by
    rw [Nat.mul_comm, Nat.mul_div_cancel _ hBN]
  simp only [attnInvariant, hcnB] at hinv
  obtain ⟨hpids, hi, hcle, hmi, hli, hacc, hq, hqs, hn, hm, hk, hO, hKp, hKs, hVp, hundef, hmem⟩ := hinv
  set qT := qTileMasked s Q H stride_qz stride_qh HEAD_DIM BLOCK_M HEAD_ACTIVE (BLOCK_N * numKVBlocks) with hqTdef
  set kT := kTile s K H stride_qz stride_qh HEAD_DIM (BLOCK_N * numKVBlocks) HEAD_ACTIVE with hkTdef
  set vT := vTile s V H stride_qz stride_qh HEAD_DIM (BLOCK_N * numKVBlocks) HEAD_ACTIVE with hvTdef
  set kS := keyScale s Q_scale K_scale (BLOCK_N * numKVBlocks) BLOCK_M BLOCK_N (BLOCK_N * numKVBlocks) with hkSdef
  -- div result tile (acc / l_i[:, None])
  set ltile : Tile .real [BLOCK_M] := ⟨fun r => ((lP qT kT kS r.1 numKVBlocks : ℝ) : WithBot ℝ)⟩ with hltile
  set acctile : Tile .real [BLOCK_M, BLOCK_DMODEL] :=
    ⟨fun i => if h : i.2.1.val < HEAD_ACTIVE then ((oP qT kT vT kS i.1 ⟨i.2.1.val, h⟩ numKVBlocks : ℝ) : WithBot ℝ) else some 0⟩ with hacctile
  set acc' : Tile .real [BLOCK_M, BLOCK_DMODEL] :=
    Tile.bop NumericDType.real.div (Broadcast.consSame (Broadcast.consR Broadcast.nil))
      acctile (Tile.expandDim ⟨1, by simp⟩ ltile) with hacc'def
  -- div step
  have hAccEval : evalOp (Op.div .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
      (Op.ref .real [BLOCK_M, BLOCK_DMODEL] "acc") (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLOCK_M] "l_i"))) st
        = some acc' := by
    have hexpN : evalOp (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLOCK_M] "l_i")) st
        = some (Tile.expandDim ⟨1, by simp⟩ ltile) := by
      rw [evalOp_expandDim]; simp [hli, hltile]
    have hexp2 : @evalOp TileDType.real [BLOCK_M, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLOCK_M] "l_i")) st
        = some (Tile.expandDim ⟨1, by simp⟩ ltile) := hexpN
    rw [evalOp_div]; simp only [evalOp_ref, hacc, hexp2, Option.bind_eq_bind, Option.bind_some]; rfl
  set st1 := st.setReg "acc" .real [BLOCK_M, BLOCK_DMODEL] acc' with hst1
  -- store readback ingredients
  set offsetFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → Nat :=
    fun i => baseOffset s H stride_qz stride_qh + mIndex s BLOCK_M i.1 * HEAD_DIM + i.2.1.val with hoff
  have hOpt : evalOp (Op.ref .ptr [BLOCK_M, BLOCK_DMODEL] "O_block_ptr") st1
      = some (⟨fun i => (Out.cast, offsetFn i)⟩ : Tile .ptr [BLOCK_M, BLOCK_DMODEL]) := by
    rw [evalOp_ref]; simp [hst1, hO, hoff]
  set maskTile : Tile .bool [BLOCK_M, BLOCK_DMODEL] :=
    ⟨fun i => decide (mIndex s BLOCK_M i.1 < BLOCK_N * numKVBlocks) && decide (i.2.1.val < HEAD_ACTIVE)⟩ with hmaskTile
  have hmaskEval : evalOp (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
      (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m"))
        (Op.constNat (BLOCK_N * numKVBlocks)))
      (Op.expandDim ⟨0, by simp⟩ (Op.lt .nat Broadcast.scalarR (Op.arange BLOCK_DMODEL) (Op.constNat HEAD_ACTIVE)))) st1
      = some maskTile :=
    qmask_eval st1 BLOCK_M BLOCK_DMODEL (BLOCK_N * numKVBlocks) HEAD_ACTIVE (fun r => mIndex s BLOCK_M r) (by simp [hst1, hm])
  have haccEval2 : evalOp (Op.ref .real [BLOCK_M, BLOCK_DMODEL] "acc") st1 = some acc' := by
    rw [evalOp_ref]; simp [hst1]
  -- offset injectivity on active cells
  have hcolinj : ∀ a b c d : Nat, b < HEAD_DIM → d < HEAD_DIM →
      a * HEAD_DIM + b = c * HEAD_DIM + d → a = c ∧ b = d := by
    intro a b c d hb hd h
    have hHD : 0 < HEAD_DIM := lt_of_le_of_lt (Nat.zero_le _) hb
    have hbd : b = d := by
      have h1 : (a * HEAD_DIM + b) % HEAD_DIM = (c * HEAD_DIM + d) % HEAD_DIM := by rw [h]
      rw [Nat.mul_comm a HEAD_DIM, Nat.mul_comm c HEAD_DIM] at h1
      rwa [Nat.mul_add_mod, Nat.mul_add_mod, Nat.mod_eq_of_lt hb, Nat.mod_eq_of_lt hd] at h1
    subst hbd
    have hac : a * HEAD_DIM = c * HEAD_DIM := by omega
    exact ⟨Nat.eq_of_mul_eq_mul_right hHD hac, rfl⟩
  have hinj : ∀ k, (mIndex s BLOCK_M k.1 < BLOCK_N * numKVBlocks ∧ k.2.1.val < HEAD_ACTIVE) →
      offsetFn k = offsetFn idx → k = idx := by
    rintro ⟨kr, kc, ku⟩ ⟨_, hkc⟩ heq
    obtain ⟨ir, ic, iu⟩ := idx
    simp only [hoff, mIndex] at heq
    simp only [mIndex] at hActive
    have hkcHD : kc.val < HEAD_DIM := lt_of_lt_of_le hkc hAD
    have hicHD : ic.val < HEAD_DIM := lt_of_lt_of_le hActive.2 hAD
    have hkey := hcolinj (s.pids 0 * BLOCK_M + kr.val) kc.val (s.pids 0 * BLOCK_M + ir.val) ic.val
      hkcHD hicHD (by omega)
    obtain ⟨hr, hcc⟩ := hkey
    have hkrir : kr = ir := Fin.ext (by omega)
    have hkcic : kc = ic := Fin.ext hcc
    subst hkrir; subst hkcic; rfl
  unfold attnAccAssign
  rw [stepStmts.cons_some (stepStmt_assign_eq_some hAccEval), stepStmts.cons_some
      (show stepStmt (attnStoreStmt BLOCK_M BLOCK_DMODEL BLOCK_N numKVBlocks HEAD_ACTIVE) st1 = some _ from by
        simp only [attnStoreStmt, stepStmt, haccEval2, hmaskEval, hOpt]; rfl),
      stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  -- readback on the store result
  · rw [show outOffset s H stride_qz stride_qh HEAD_DIM 1 BLOCK_M BLOCK_DMODEL idx = offsetFn idx from by
      simp [outOffset, hoff, mul_one]]
    simp only [BlockState.writeMemTyped_real, Region.cast_id]
    rw [BlockState.scatter_readback_prop_masked_nd_of_true (region := Out) st1 offsetFn
      (fun i => FloatDType.real.storeValue (acc'.data i))
      (fun i => maskTile.data i)
      idx (by simp [hmaskTile, hActive.1, hActive.2]) (fun k hk heq => hinj k (by simpa [hmaskTile] using hk) heq)]
    -- value at idx = oP/lP = closed form
    simp only [hacc'def, hacctile, hltile, Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex,
      Tile.expandDim_data, TileShape.dropInsertedIndex, dif_pos hActive.2, NumericDType.div,
      WithBot.realDiv, Option.map₂, Option.bind, Option.map, FloatDType.storeValue,
      FloatDType.real_toWithBot, WithBot.unbotD_coe]
    rw [closed_form_eq qT kT vT kS hBN hnB idx.1 ⟨idx.2.1.val, hActive.2⟩]
    have hQrow : ∀ e : Fin HEAD_ACTIVE,
        qTileMasked s Q H stride_qz stride_qh HEAD_DIM BLOCK_M HEAD_ACTIVE (BLOCK_N * numKVBlocks) (idx.1, e, PUnit.unit)
          = qTile s Q H stride_qz stride_qh HEAD_DIM BLOCK_M HEAD_ACTIVE (idx.1, e, PUnit.unit) :=
      fun e => qTileMasked_active s Q H stride_qz stride_qh HEAD_DIM BLOCK_M HEAD_ACTIVE
        (BLOCK_N * numKVBlocks) (idx.1, e, PUnit.unit) hActive.1
    simp only [WithBot.unbotD_some, hqTdef, hkTdef, hvTdef, hkSdef,
      attentionRealBase2PerKeyScale, hQrow]

/-- **Closed-form correctness for `attention_forward_triton` (general statement).**

For arbitrary batch/head strides, head count, block sizes, KV-block count,
head/active dimensions and arbitrary `q_scale`/`k_scale`, every active output
lane of `Out` (`mIndex < N_CTX ∧ head < HEAD_ACTIVE`) equals
`attentionRealBase2PerKeyScale` of the loaded Q/K/V tiles under the per-block key
scale — i.e. the genuine base-2, per-key-scaled attention output, NOT the
kernel's own executed value.

Layout contracts: `N_CTX = BLOCK_N · numKVBlocks`, `stride_qm = stride_kn =
HEAD_DIM` and head stride `1` (so the per-block pointer advance composes into a
per-key address), `0 < BLOCK_N`, `HEAD_ACTIVE ≤ BLOCK_DMODEL`,
`HEAD_ACTIVE ≤ HEAD_DIM` (store-offset injectivity), clean initial `undef`.

**Proven sorry-free** via the full `exec`-side loop unfolding: `preLoop`
(`P 0`) + `attn_step` (one-block invariant advance, the 19-statement loop body) +
`attn_postLoop` (`acc /= l_i` + masked store = closed form), composed by
`attention_forward_triton_exec_reduction`/`forRange_inv`. -/
theorem attention_forward_triton_closed_form_correct
    (Q K V Q_scale K_scale Out : RegionName) (s : BlockState)
    (stride_qz stride_qh Z H BLOCK_M BLOCK_N numKVBlocks
      HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE STAGE : Nat)
    (hBN : 0 < BLOCK_N) (hActiveLe : HEAD_ACTIVE ≤ BLOCK_DMODEL)
    (hHD : HEAD_ACTIVE ≤ HEAD_DIM) (hundef : ∀ rg o, s.undef rg o = 0)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL])
    (hActive : mIndex s BLOCK_M idx.1 < BLOCK_N * numKVBlocks
      ∧ idx.2.1.val < HEAD_ACTIVE) :
    (match exec (attention_forward_triton_surface Q K V Q_scale K_scale Out
        stride_qz stride_qh HEAD_DIM 1
        stride_qz stride_qh HEAD_DIM 1
        stride_qz stride_qh HEAD_DIM 1
        stride_qz stride_qh HEAD_DIM 1
        Z H (BLOCK_N * numKVBlocks) HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL
        HEAD_ACTIVE STAGE) s with
      | some s' => s'.readMem Out
          (outOffset s H stride_qz stride_qh HEAD_DIM 1 BLOCK_M BLOCK_DMODEL idx)
      | none => (0.0 : ℝ)) =
      attentionRealBase2PerKeyScale
        (qTile s Q H stride_qz stride_qh HEAD_DIM BLOCK_M HEAD_ACTIVE)
        (kTile s K H stride_qz stride_qh HEAD_DIM (BLOCK_N * numKVBlocks) HEAD_ACTIVE)
        (vTile s V H stride_qz stride_qh HEAD_DIM (BLOCK_N * numKVBlocks) HEAD_ACTIVE)
        (keyScale s Q_scale K_scale (BLOCK_N * numKVBlocks) BLOCK_M BLOCK_N
          (BLOCK_N * numKVBlocks))
        (idx.1, ⟨idx.2.1.val, hActive.2⟩, PUnit.unit) := by
  have hnB : 1 ≤ numKVBlocks := by
    rcases numKVBlocks with _ | n
    · exact absurd hActive.1 (by simp)
    · omega
  obtain ⟨sfin, hexec, hpost⟩ := attention_forward_triton_exec_reduction Q K V Q_scale K_scale Out s
    stride_qz stride_qh Z H BLOCK_M BLOCK_N numKVBlocks HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE STAGE hBN
    (attnInvariant Q K V Q_scale K_scale Out s stride_qz stride_qh H BLOCK_M BLOCK_N numKVBlocks
      HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE)
    (attnLoopBody BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE HEAD_DIM numKVBlocks)
    (attnAccAssign BLOCK_M BLOCK_DMODEL)
    (attnStoreStmt BLOCK_M BLOCK_DMODEL BLOCK_N numKVBlocks HEAD_ACTIVE)
    (attn_body_split Q K V Q_scale K_scale Out stride_qz stride_qh Z H BLOCK_M BLOCK_N numKVBlocks
      HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE STAGE)
    (preLoop Q K V Q_scale K_scale Out s stride_qz stride_qh H BLOCK_M BLOCK_N numKVBlocks
      HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE STAGE hundef)
    (fun i st hinv => by
      simp only [attnInvariant] at hinv
      obtain ⟨_, hieq, hcle, _⟩ := hinv
      calc i = i / BLOCK_N * BLOCK_N := hieq
        _ ≤ numKVBlocks * BLOCK_N := Nat.mul_le_mul_right _ hcle
        _ = BLOCK_N * numKVBlocks := Nat.mul_comm _ _)
    (fun i st hlt hinv => attn_step Q K V Q_scale K_scale Out s stride_qz stride_qh H BLOCK_M BLOCK_N
      numKVBlocks HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE hBN hActiveLe i st hlt hinv)
    (fun sfin => sfin.readMem Out (outOffset s H stride_qz stride_qh HEAD_DIM 1 BLOCK_M BLOCK_DMODEL idx)
      = attentionRealBase2PerKeyScale
          (qTile s Q H stride_qz stride_qh HEAD_DIM BLOCK_M HEAD_ACTIVE)
          (kTile s K H stride_qz stride_qh HEAD_DIM (BLOCK_N * numKVBlocks) HEAD_ACTIVE)
          (vTile s V H stride_qz stride_qh HEAD_DIM (BLOCK_N * numKVBlocks) HEAD_ACTIVE)
          (keyScale s Q_scale K_scale (BLOCK_N * numKVBlocks) BLOCK_M BLOCK_N (BLOCK_N * numKVBlocks))
          (idx.1, ⟨idx.2.1.val, hActive.2⟩, PUnit.unit))
    (fun st hinv => attn_postLoop Q K V Q_scale K_scale Out s stride_qz stride_qh H BLOCK_M BLOCK_N
      numKVBlocks HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE hBN hnB hHD idx hActive st hinv)
  rw [hexec]
  exact hpost

end VeriTile.Examples.AttentionForwardClosedForm
