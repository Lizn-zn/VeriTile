import VeriTile.Triton

/-!
# `chunk_linear_attn` — strict per-kernel correctness

The upstream file holds **four** `@triton.jit` kernels: the state-recurrence
forward `chunk_linear_attn_fwd_kernel_h`, the output forward
`chunk_linear_attn_fwd_kernel_o`, the state-recurrence backward
`chunk_linear_attn_bwd_kernel_dh`, and the fused gradient backward
`chunk_linear_attn_bwd_kernel_dqkv`. This file covers the two **state-recurrence**
kernels — the file's first kernel, `chunk_linear_attn.py`'s
`chunk_linear_attn_fwd_kernel_h`, and its descending mirror `bwd_kernel_dh` —
which share one proof skeleton: a `[BK, BV]` running state is **stored to memory
at every chunk** and then advanced by one `tl.dot`.

```
fwd_h:  h[·,·,t] = H_t,   H_0 = (h0 or 0),  H_{t+1} = H_t + k_tᵀ · v_t
        (and ht = H_NT when STORE_FINAL_STATE)
bwd_dh: dh[·,·,t] = D_t,  D_NT = 0,         D_{t-1} = D_t + (scale·q_t) · do_t
```

The store-then-accumulate order matters and is carried by the spec: chunk `t`
receives the state *before* chunk `t`'s own contribution.

## The descending loop, spelled ascending

`bwd_kernel_dh` iterates `for i_t in range(NT - 1, -1, -1)`. The DSL's `forRange`
counts up (`stepForRangeAux` advances while `cur < stop`), so the faithful surface
spells the change of variable explicitly:

    for j in range(0, NT, 1) { i_t = NT - $(1) - j; … }

— the identical iteration sequence, with the descending index computed from the
ascending counter as the body's first statement. This is the established
respelling (`triton_linear_activation`'s "descending K-loop as antiquoted
ascending trip count", `completion_audit.md`), and it is observable in the
surface rather than hidden in a spec.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); the host launch (the 3-D
grid `(NK, NV, B*H)` and the host-computed `NT`) is the *trusted boundary*. Every
dimension and stride stays a symbolic parameter, as do the two `constexpr` gates
`USE_INITIAL_STATE` / `STORE_FINAL_STATE` — one theorem covers all four
configurations. The `.to(...)` dtype round-trips erase to the identity at the
algorithm layer.

Spelling notes, per `bench/MAIN_THEOREM_CONVENTIONS.md`, all surface syntax
rather than semantics: integer literals inside index arithmetic are written
`$(n)`; `boundary_check=(0, 1)` is written `boundary_check=([0, 1] : List Nat)`.
-/

namespace VeriTile.Bench.TritonBenchG.ChunkLinearAttn

open VeriTile.Triton

section Correct_without_Rounding

/-- Faithful transcription of `chunk_linear_attn_fwd_kernel_h`. -/
def cla_fwd_h_surface
    (k v h h0 ht : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat)
    (T K V BT BK BV NT : Nat)
    (USE_INITIAL_STATE STORE_FINAL_STATE : Bool) :
    ComputeKernel := triton {
  i_k = tl.program_id(0)
  i_v = tl.program_id(1)
  i_bh = tl.program_id(2)
  b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
  if USE_INITIAL_STATE {
    p_h0 = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V),
      shape=($(K), $(V)), strides=($(V), $(1)),
      offsets=(i_k * $(BK), i_v * $(BV)), block_shape=($(BK), $(BV)), order=(1, 0))
    b_h = (tl.load(p_h0, boundary_check=([0, 1] : List Nat))).to(tl.float32)
  }
  for i_t in range($(0), $(NT), $(1)) {
    p_k = tl.make_block_ptr(base=k + i_bh * $(s_qk_h),
      shape=($(K), $(T)), strides=($(s_qk_d), $(s_qk_t)),
      offsets=(i_k * $(BK), i_t * $(BT)), block_shape=($(BK), $(BT)), order=(0, 1))
    p_v = tl.make_block_ptr(base=v + i_bh * $(s_vo_h),
      shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)),
      offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
    p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V),
      shape=($(K), $(V)), strides=($(s_h_t), $(1)),
      offsets=(i_k * $(BK), i_v * $(BV)), block_shape=($(BK), $(BV)), order=(1, 0))
    tl.store(p_h, (b_h).to(p_h.dtype.element_ty), boundary_check=([0, 1] : List Nat))
    b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
    b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
    b_h += tl.dot(b_k, b_v, allow_tf32=false)
  }
  if STORE_FINAL_STATE {
    p_ht = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V),
      shape=($(K), $(V)), strides=($(V), $(1)),
      offsets=(i_k * $(BK), i_v * $(BV)), block_shape=($(BK), $(BV)), order=(1, 0))
    tl.store(p_ht, (b_h).to(p_ht.dtype.element_ty), boundary_check=([0, 1] : List Nat))
  }
}

/-- The forward state surface lowers to the algorithm layer. -/
theorem cla_fwd_h_surface_toAlgorithm_supported
    (k v h h0 ht : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat)
    (T K V BT BK BV NT : Nat) (USE_INITIAL_STATE STORE_FINAL_STATE : Bool) :
    ∃ alg, (cla_fwd_h_surface k v h h0 ht s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
      s_h_h s_h_t T K V BT BK BV NT
      USE_INITIAL_STATE STORE_FINAL_STATE).toAlgorithm? = Except.ok alg := by
  simp [cla_fwd_h_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- Faithful transcription of `chunk_linear_attn_bwd_kernel_dh`, with the
descending loop spelled as its ascending change of variable (see the preamble). -/
def cla_bwd_dh_surface
    (q do_ dh : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV NT : Nat) :
    ComputeKernel := triton {
  i_k = tl.program_id(0)
  i_v = tl.program_id(1)
  i_bh = tl.program_id(2)
  b_dh = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
  for j in range($(0), $(NT), $(1)) {
    i_t = $(NT) - $(1) - j
    p_q = tl.make_block_ptr(base=q + i_bh * $(s_qk_h),
      shape=($(K), $(T)), strides=($(s_qk_d), $(s_qk_t)),
      offsets=(i_k * $(BK), i_t * $(BT)), block_shape=($(BK), $(BT)), order=(0, 1))
    p_do = tl.make_block_ptr(base=do_ + i_bh * $(s_vo_h),
      shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)),
      offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
    p_dh = tl.make_block_ptr(base=dh + i_bh * $(s_h_h) + i_t * $(K) * $(V),
      shape=($(K), $(V)), strides=($(s_h_t), $(1)),
      offsets=(i_k * $(BK), i_v * $(BV)), block_shape=($(BK), $(BV)), order=(1, 0))
    tl.store(p_dh, (b_dh).to(p_dh.dtype.element_ty), boundary_check=([0, 1] : List Nat))
    b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat))
    b_q = (b_q * $(scale)).to(b_q.dtype)
    b_do = tl.load(p_do, boundary_check=([0, 1] : List Nat))
    b_dh += tl.dot(b_q, (b_do).to(b_q.dtype), allow_tf32=false)
  }
}

/-- The backward state surface lowers to the algorithm layer. -/
theorem cla_bwd_dh_surface_toAlgorithm_supported
    (q do_ dh : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV NT : Nat) :
    ∃ alg, (cla_bwd_dh_surface q do_ dh s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
      s_h_h s_h_t scale T K V BT BK BV NT).toAlgorithm? = Except.ok alg := by
  simp [cla_bwd_dh_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-! ## Compiled body decomposition

The algorithm-lowered statement lists, checked against the macro output by `rfl`.
One lowering worth naming: an assignment-position `.to(tl.float32)` on a load
lowers to a real `Op.castFloat FloatDType.real FloatDType.real` (not to erasure —
only *expression-interior* casts erase), whose `.real → .real` semantics is the
identity on lanes. -/

/-- The `USE_INITIAL_STATE` branch: the `h0` block pointer and the seeded load. -/
def claInitBranch (h0 : RegionName) (K V BK BV : Nat) : List Stmt :=
  [ Stmt.assign .blockPtr [BK, BV] "p_h0"
      (Op.makeBlockPtrDynOffsets h0
        (Op.mul .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat K))
          (Op.constNat V))
        [K, V] [BK, BV] [V, 1]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BK),
          Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_v") (Op.constNat BV)]),
    Stmt.assign .real [BK, BV] "b_h"
      (Op.castFloat FloatDType.real FloatDType.real
        (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [BK, BV] "p_h0") [0, 1])
          MaskOpt.none)) ]

/-- The forward chunk body: three block pointers, the state **store**, the `k` and
`v` loads, and the recurrence `b_h += tl.dot(b_k, b_v)`. The store-before-load
order is the whole content of the spec's "state *before* chunk `t`". -/
def claFwdBody (k v h : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat)
    (T K V BT BK BV : Nat) : List Stmt :=
  [ Stmt.assign .blockPtr [BK, BT] "p_k"
      (Op.makeBlockPtrDynOffsets k
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_qk_h))
        [K, T] [BK, BT] [s_qk_d, s_qk_t]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BK),
          Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT)]),
    Stmt.assign .blockPtr [BT, BV] "p_v"
      (Op.makeBlockPtrDynOffsets v
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_vo_h))
        [T, V] [BT, BV] [s_vo_t, s_vo_d]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT),
          Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_v") (Op.constNat BV)]),
    Stmt.assign .blockPtr [BK, BV] "p_h"
      (Op.makeBlockPtrDynOffsets h
        (Op.add .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_h_h))
          (Op.mul .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat K))
            (Op.constNat V)))
        [K, V] [BK, BV] [s_h_t, 1]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BK),
          Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_v") (Op.constNat BV)]),
    Stmt.store .real [BK, BV]
      (MemAccess.blockPtr (Op.ref .blockPtr [BK, BV] "p_h") [0, 1])
      (Op.ref .real [BK, BV] "b_h") MaskOpt.none,
    Stmt.assign .real [BK, BT] "b_k"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [BK, BT] "p_k") [0, 1])
        MaskOpt.none),
    Stmt.assign .real [BT, BV] "b_v"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [BT, BV] "p_v") [0, 1])
        MaskOpt.none),
    Stmt.assign .real [BK, BV] "b_h"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BK, BV] "b_h")
        (Op.dot (batch := []) (Op.ref .real [BK, BT] "b_k")
          (Op.ref .real [BT, BV] "b_v"))) ]

/-- The `STORE_FINAL_STATE` branch: the `ht` block pointer and the final store. -/
def claFinalBranch (ht : RegionName) (K V BK BV : Nat) : List Stmt :=
  [ Stmt.assign .blockPtr [BK, BV] "p_ht"
      (Op.makeBlockPtrDynOffsets ht
        (Op.mul .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat K))
          (Op.constNat V))
        [K, V] [BK, BV] [V, 1]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BK),
          Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_v") (Op.constNat BV)]),
    Stmt.store .real [BK, BV]
      (MemAccess.blockPtr (Op.ref .blockPtr [BK, BV] "p_ht") [0, 1])
      (Op.ref .real [BK, BV] "b_h") MaskOpt.none ]

set_option maxRecDepth 8000 in
/-- **Forward body split (by `rfl`).** Seven top-level statements. -/
theorem cla_fwd_body_eq (k v h h0 ht : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat)
    (T K V BT BK BV NT : Nat) (UIS SFS : Bool) :
    (cla_fwd_h_surface k v h h0 ht s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
        s_h_h s_h_t T K V BT BK BV NT UIS SFS).toAlgKernel.body
      = [ Stmt.assign .nat [] "i_k" (Op.programId 0),
          Stmt.assign .nat [] "i_v" (Op.programId 1),
          Stmt.assign .nat [] "i_bh" (Op.programId 2),
          Stmt.assign .real [BK, BV] "b_h" (Op.full [BK, BV] (Op.const 0)),
          Stmt.ifThen (Op.constBool UIS) (claInitBranch h0 K V BK BV),
          Stmt.forRange "i_t" 0 NT 1
            (claFwdBody k v h s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t
              T K V BT BK BV),
          Stmt.ifThen (Op.constBool SFS) (claFinalBranch ht K V BK BV) ] := by
  rfl

/-- The backward chunk body: the descending index as its first statement, then the
same store-then-accumulate skeleton over `q` / `do`. -/
def claBwdBody (q do_ dh : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat) (scale : ℝ)
    (T K V BT BK BV NT : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "i_t"
      (Op.sub .nat Broadcast.nil
        (Op.sub .nat Broadcast.nil (Op.constNat NT) (Op.constNat 1))
        (Op.ref .nat [] "j")),
    Stmt.assign .blockPtr [BK, BT] "p_q"
      (Op.makeBlockPtrDynOffsets q
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_qk_h))
        [K, T] [BK, BT] [s_qk_d, s_qk_t]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BK),
          Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT)]),
    Stmt.assign .blockPtr [BT, BV] "p_do"
      (Op.makeBlockPtrDynOffsets do_
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_vo_h))
        [T, V] [BT, BV] [s_vo_t, s_vo_d]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT),
          Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_v") (Op.constNat BV)]),
    Stmt.assign .blockPtr [BK, BV] "p_dh"
      (Op.makeBlockPtrDynOffsets dh
        (Op.add .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_h_h))
          (Op.mul .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat K))
            (Op.constNat V)))
        [K, V] [BK, BV] [s_h_t, 1]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BK),
          Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_v") (Op.constNat BV)]),
    Stmt.store .real [BK, BV]
      (MemAccess.blockPtr (Op.ref .blockPtr [BK, BV] "p_dh") [0, 1])
      (Op.ref .real [BK, BV] "b_dh") MaskOpt.none,
    Stmt.assign .real [BK, BT] "b_q"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [BK, BT] "p_q") [0, 1])
        MaskOpt.none),
    Stmt.assign .real [BK, BT] "b_q"
      (Op.mul .real Broadcast.scalarR (Op.ref .real [BK, BT] "b_q") (Op.const scale)),
    Stmt.assign .real [BT, BV] "b_do"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [BT, BV] "p_do") [0, 1])
        MaskOpt.none),
    Stmt.assign .real [BK, BV] "b_dh"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BK, BV] "b_dh")
        (Op.dot (batch := []) (Op.ref .real [BK, BT] "b_q")
          (Op.ref .real [BT, BV] "b_do"))) ]

set_option maxRecDepth 8000 in
/-- **Backward body split (by `rfl`).** Five top-level statements. -/
theorem cla_bwd_body_eq (q do_ dh : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV NT : Nat) :
    (cla_bwd_dh_surface q do_ dh s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
        s_h_h s_h_t scale T K V BT BK BV NT).toAlgKernel.body
      = [ Stmt.assign .nat [] "i_k" (Op.programId 0),
          Stmt.assign .nat [] "i_v" (Op.programId 1),
          Stmt.assign .nat [] "i_bh" (Op.programId 2),
          Stmt.assign .real [BK, BV] "b_dh" (Op.full [BK, BV] (Op.const 0)),
          Stmt.forRange "j" 0 NT 1
            (claBwdBody q do_ dh s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h
              s_h_t scale T K V BT BK BV NT) ] := by
  rfl

/-! ## Closed-form specification

All accessors read the **launch** state's memory at the addresses the block
pointers compute, with the boundary checks factored into guarded forms. The chunk
index is `t`, the K lane `e`, the T lane `c`, the V lane `p`. -/

/-- `k[i_k·BK + e, t·BT + c]` (parent `(K, T)`, strides `(s_qk_d, s_qk_t)`). -/
noncomputable def kElem (s : BlockState) (k : RegionName)
    (s_qk_h s_qk_t s_qk_d BT BK : Nat) (t c e : Nat) : ℝ :=
  s.readMem k (s.pids 2 * s_qk_h + (s.pids 0 * BK + e) * s_qk_d
    + (t * BT + c) * s_qk_t)

/-- `v[t·BT + c, i_v·BV + p]` (parent `(T, V)`, strides `(s_vo_t, s_vo_d)`). -/
noncomputable def vElem (s : BlockState) (v : RegionName)
    (s_vo_h s_vo_t s_vo_d BT BV : Nat) (t c p : Nat) : ℝ :=
  s.readMem v (s.pids 2 * s_vo_h + (t * BT + c) * s_vo_t
    + (s.pids 1 * BV + p) * s_vo_d)

/-- `h0[i_k·BK + e, i_v·BV + p]` (parent `(K, V)`, strides `(V, 1)`). -/
noncomputable def h0Elem (s : BlockState) (h0 : RegionName)
    (K V BK BV : Nat) (e p : Nat) : ℝ :=
  s.readMem h0 (s.pids 2 * K * V + (s.pids 0 * BK + e) * V
    + (s.pids 1 * BV + p) * 1)

/-- The guarded `k` lane, as `b_k` holds it. -/
noncomputable def kGuarded (s : BlockState) (k : RegionName)
    (s_qk_h s_qk_t s_qk_d : Nat) (T K BT BK : Nat) (t c e : Nat) : ℝ :=
  if s.pids 0 * BK + e < K ∧ t * BT + c < T then
    kElem s k s_qk_h s_qk_t s_qk_d BT BK t c e
  else 0

/-- The guarded `v` lane, as `b_v` holds it. -/
noncomputable def vGuarded (s : BlockState) (v : RegionName)
    (s_vo_h s_vo_t s_vo_d : Nat) (T V BT BV : Nat) (t c p : Nat) : ℝ :=
  if t * BT + c < T ∧ s.pids 1 * BV + p < V then
    vElem s v s_vo_h s_vo_t s_vo_d BT BV t c p
  else 0

/-- The guarded initial state, as the `USE_INITIAL_STATE` branch leaves `b_h`. -/
noncomputable def h0Guarded (s : BlockState) (h0 : RegionName)
    (K V BK BV : Nat) (e p : Nat) : ℝ :=
  if s.pids 0 * BK + e < K ∧ s.pids 1 * BV + p < V then
    h0Elem s h0 K V BK BV e p
  else 0

/-- One chunk's contribution to state lane `(e, p)`: `Σ_c k[e, c] · v[c, p]`. -/
noncomputable def claContrib (s : BlockState) (k v : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat)
    (T K V BT BK BV : Nat) (t e p : Nat) : ℝ :=
  ∑ c : Fin BT,
    kGuarded s k s_qk_h s_qk_t s_qk_d T K BT BK t c.val e
      * vGuarded s v s_vo_h s_vo_t s_vo_d T V BT BV t c.val p

/-- **The forward state at chunk `t`** — what the kernel stores into `h[·,·,t]`:
the (gated) initial state plus every chunk *before* `t`. -/
noncomputable def claHState (s : BlockState) (k v h0 : RegionName)
    (UIS : Bool) (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat)
    (T K V BT BK BV : Nat) (t e p : Nat) : ℝ :=
  (if UIS then h0Guarded s h0 K V BK BV e p else 0)
    + ∑ u : Fin t, claContrib s k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
        T K V BT BK BV u.val e p

/-- One chunk's contribution to the backward state lane `(e, p)`:
`Σ_c (scale · q)[e, c] · do[c, p]` — `q` shares `k`'s layout, `do` shares `v`'s. -/
noncomputable def claBContrib (s : BlockState) (q do_ : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (T K V BT BK BV : Nat) (t e p : Nat) : ℝ :=
  ∑ c : Fin BT,
    (kGuarded s q s_qk_h s_qk_t s_qk_d T K BT BK t c.val e * scale)
      * vGuarded s do_ s_vo_h s_vo_t s_vo_d T V BT BV t c.val p

/-- **The backward state at chunk `t`** — what the kernel stores into `dh[·,·,t]`:
every chunk *after* `t` (the loop descends, and each chunk is stored before it is
accumulated). -/
noncomputable def claDhState (s : BlockState) (q do_ : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (T K V BT BK BV NT : Nat) (t e p : Nat) : ℝ :=
  ∑ u : Fin NT, if t < u.val then
    claBContrib s q do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
      T K V BT BK BV u.val e p
  else 0

/-- The `h` / `dh` chunk-`t` store address for lane `(e, p)`. -/
def claHOffset (s : BlockState) (s_h_h s_h_t K V BK BV : Nat) (t : Nat)
    (idx : TileIndex [BK, BV]) : Nat :=
  s.pids 2 * s_h_h + t * K * V + (s.pids 0 * BK + idx.1.val) * s_h_t
    + (s.pids 1 * BV + idx.2.1.val) * 1

/-- The `ht` store address for lane `(e, p)`. -/
def claHtOffset (s : BlockState) (K V BK BV : Nat)
    (idx : TileIndex [BK, BV]) : Nat :=
  s.pids 2 * K * V + (s.pids 0 * BK + idx.1.val) * V
    + (s.pids 1 * BV + idx.2.1.val) * 1

/-- A state lane is *active* when it maps inside the `K × V` window. -/
def claActive (s : BlockState) (K V BK BV : Nat)
    (idx : TileIndex [BK, BV]) : Prop :=
  s.pids 0 * BK + idx.1.val < K ∧ s.pids 1 * BV + idx.2.1.val < V

end Correct_without_Rounding

end VeriTile.Bench.TritonBenchG.ChunkLinearAttn
