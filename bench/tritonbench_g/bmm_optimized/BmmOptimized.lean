import VeriTile.Triton

/-!
# `bmm_optimized` — strict per-kernel correctness

`bmm_optimized.py` holds **one** `@triton.jit` kernel: `bmm_kernel`, a
batched GEMM (`O[b] = A[b] · B[b]` over a 3-D grid
`(cdiv(M,TILE_M), cdiv(N,TILE_N), batch)`) with a `GROUP_M` CTA-reorder
swizzle and `DIVISIBLE_M/N/K` heuristic mask specialization.

## Scope and constexpr specialization

Translation-surface blocker: the kernel's `DIVISIBLE_M` / `DIVISIBLE_N` /
`DIVISIBLE_K` `triton.heuristics` constexprs select between arms in which
the load/store *mask arguments* are `None` or mask tiles — a register
holding "`None` or a tile" has no DSL analogue (`MaskOpt` is syntax-level),
so this port fixes the constexpr assignment
`DIVISIBLE_M = DIVISIBLE_N = DIVISIBLE_K = False` (the fully-masked arm —
the only arm that is total for **arbitrary** `M, N, K`; the other arms are
mask-elision optimizations the host may select when a dimension happens to
be divisible, and they compute the same values on their domains). The
`GROUP_M == 1` / `else` CTA-reorder branch is runtime-evaluable on spliced
constants and is transcribed **in full** (both arms; the `GROUP_SIZE`
boundary gate is a runtime `if` inside the else arm). The three
batch-offset parameter reassignments (`A += pid_b*M*K`, `B += pid_b*K*N`,
`O += pid_b*M*N`) are folded into the three pointer-tile constructions
(the DSL cannot rebind a region parameter), the tuple assignment
`pid_m, pid_n = pidx, pidy` is split into one statement each, and the
single-argument `range(num_iters)` is spelled `range($(0), num_iters,
$(1))`. The textual py↔lean scans in `bench/audit_tritonbench_g.sh` exempt
this port on this marker (registered in `proof_blockers.md`).

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float);
`num_warps`/`num_stages` and the `triton.autotune` config sweep are not
modeled (`TILE_M/TILE_N/TILE_K/GROUP_M` stay symbolic binders). The masked
plain-pointer loads carry no `other`, so masked-off lanes read the
`s.undef` channel; the headline carries the standard clean-input hypothesis
`hundef : ∀ rg off, s.undef rg off = 0` (the `bmm_chunk_fwd` /
`attn_fwd_triton` convention). The store mask guards every lane back into
the `M × N` window, so the headline holds for **arbitrary ragged** `M, N,
K` with no divisibility hypotheses at all.

The port targets `bmm_optimized.py`'s `bmm_kernel`.
-/

namespace VeriTile.Bench.TritonBenchG.BmmOptimized

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-! **★ Main theorem:** `bmm_o_exec_genuine` -/

section Correct_without_Rounding

/-- Faithful transcription of `bmm_kernel` at
`DIVISIBLE_M = DIVISIBLE_N = DIVISIBLE_K = False` (see the preamble), with
both `GROUP_M` CTA-reorder arms. -/
def bmm_surface
    (A B O : RegionName)
    (M N K TILE_M TILE_N TILE_K GROUP_M : Nat) :
    ComputeKernel := triton {
  pid_b = tl.program_id(2)
  pidx = tl.program_id(0)
  pidy = tl.program_id(1)
  if $(GROUP_M) == $(1) {
    pid_m = pidx
    pid_n = pidy
  } else {
    gridx = tl.num_programs(0)
    gridy = tl.num_programs(1)
    pid = pidx + pidy * gridx
    num_CTA_per_group = gridy * $(GROUP_M)
    group_id = pid // num_CTA_per_group
    inner_group_id = pid % num_CTA_per_group
    if (group_id * $(GROUP_M) + $(GROUP_M)) > gridx {
      GROUP_SIZE = gridx % $(GROUP_M)
    } else {
      GROUP_SIZE = $(GROUP_M)
    }
    pid_m = group_id * $(GROUP_M) + inner_group_id % GROUP_SIZE
    pid_n = inner_group_id // GROUP_SIZE
  }
  offs_m = pid_m * $(TILE_M) + tl.arange(0, $(TILE_M))
  offs_n = pid_n * $(TILE_N) + tl.arange(0, $(TILE_N))
  offs_k = tl.arange(0, $(TILE_K))
  mask_m = offs_m < $(M)
  mask_n = offs_n < $(N)
  a_ptrs = A + (pid_b * $(M) * $(K) + offs_m[:, None] * $(K) + offs_k[None, :])
  b_ptrs = B + (pid_b * $(K) * $(N) + offs_k[:, None] * $(N) + offs_n[None, :])
  o_ptrs = O + (pid_b * $(M) * $(N) + offs_m[:, None] * $(N) + offs_n[None, :])
  num_iters = tl.cdiv($(K), $(TILE_K))
  o = tl.zeros([$(TILE_M), $(TILE_N)], dtype=tl.float32)
  for _i in range($(0), num_iters, $(1)) {
    mask_k = offs_k < $(K)
    mask_a = mask_m[:, None] & mask_k[None, :]
    mask_b = mask_k[:, None] & mask_n[None, :]
    a = tl.load(a_ptrs, mask=mask_a)
    b = tl.load(b_ptrs, mask=mask_b)
    offs_k += $(TILE_K)
    a_ptrs += $(TILE_K)
    b_ptrs += $(TILE_K) * $(N)
    o += tl.dot(a, b, allow_tf32=false)
  }
  mask_c = mask_m[:, None] & mask_n[None, :]
  tl.store(o_ptrs, o, mask=mask_c)
}

/-- The surface lowers to the algorithm layer. -/
theorem bmm_surface_toAlgorithm_supported
    (A B O : RegionName)
    (M N K TILE_M TILE_N TILE_K GROUP_M : Nat) :
    ∃ alg, (bmm_surface A B O M N K TILE_M TILE_N TILE_K
      GROUP_M).toAlgorithm? = Except.ok alg := by
  simp [bmm_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-! ## Compiled body decomposition

The algorithm-lowered statement list, checked against the macro output by
`rfl`. The `GROUP_M == 1` constexpr gate lowers to `Stmt.ifThenElse` on
`Op.eq` of spliced constants (with the runtime `GROUP_SIZE` boundary gate
as a nested `Stmt.ifThenElse` on `Op.gt`); `tl.num_programs` is
`Op.numPrograms`; the masked plain loads/stores are `MemAccess.ptr` with
`MaskOpt.mask`. -/

/-- The `GROUP_M == 1` arm: identity CTA map. -/
def bmmGate1Body : List Stmt :=
  [ Stmt.assign .nat [] "pid_m" (Op.ref .nat [] "pidx"),
    Stmt.assign .nat [] "pid_n" (Op.ref .nat [] "pidy") ]

/-- The `else` arm: the grouped CTA-reorder swizzle with the runtime
`GROUP_SIZE` boundary gate. -/
def bmmGateGrpBody (GROUP_M : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "gridx" (Op.numPrograms 0),
    Stmt.assign .nat [] "gridy" (Op.numPrograms 1),
    Stmt.assign .nat [] "pid"
      (Op.add .nat Broadcast.nil (Op.ref .nat [] "pidx")
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pidy")
          (Op.ref .nat [] "gridx"))),
    Stmt.assign .nat [] "num_CTA_per_group"
      (Op.mul .nat Broadcast.nil (Op.ref .nat [] "gridy") (Op.constNat GROUP_M)),
    Stmt.assign .nat [] "group_id"
      (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref .nat [] "pid")
        (Op.ref .nat [] "num_CTA_per_group")),
    Stmt.assign .nat [] "inner_group_id"
      (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "pid")
        (Op.ref .nat [] "num_CTA_per_group")),
    Stmt.ifThenElse
      (Op.gt ComparableDType.nat Broadcast.nil
        (Op.add .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "group_id")
            (Op.constNat GROUP_M))
          (Op.constNat GROUP_M))
        (Op.ref .nat [] "gridx"))
      [ Stmt.assign .nat [] "GROUP_SIZE"
          (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "gridx")
            (Op.constNat GROUP_M)) ]
      [ Stmt.assign .nat [] "GROUP_SIZE" (Op.constNat GROUP_M) ],
    Stmt.assign .nat [] "pid_m"
      (Op.add .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "group_id")
          (Op.constNat GROUP_M))
        (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "inner_group_id")
          (Op.ref .nat [] "GROUP_SIZE"))),
    Stmt.assign .nat [] "pid_n"
      (Op.floorDiv IntegralDType.nat Broadcast.nil
        (Op.ref .nat [] "inner_group_id") (Op.ref .nat [] "GROUP_SIZE")) ]

/-- The K-streaming loop body: per-block masks, the two masked plain loads,
the offset/pointer advances, and the accumulator. -/
def bmmLoopBody (N K TILE_M TILE_N TILE_K : Nat) : List Stmt :=
  [ Stmt.assign .bool [TILE_K] "mask_k"
      (Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref .nat [TILE_K] "offs_k")
        (Op.constNat K)),
    Stmt.assign .bool [TILE_M, TILE_K] "mask_a"
      (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .bool [TILE_M] "mask_m"))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .bool [TILE_K] "mask_k"))),
    Stmt.assign .bool [TILE_K, TILE_N] "mask_b"
      (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .bool [TILE_K] "mask_k"))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .bool [TILE_N] "mask_n"))),
    Stmt.assign .real [TILE_M, TILE_K] "a"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [TILE_M, TILE_K] "a_ptrs"))
        (MaskOpt.mask (Op.ref .bool [TILE_M, TILE_K] "mask_a"))),
    Stmt.assign .real [TILE_K, TILE_N] "b"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [TILE_K, TILE_N] "b_ptrs"))
        (MaskOpt.mask (Op.ref .bool [TILE_K, TILE_N] "mask_b"))),
    Stmt.assign .nat [TILE_K] "offs_k"
      (Op.add .nat Broadcast.scalarR (Op.ref .nat [TILE_K] "offs_k")
        (Op.constNat TILE_K)),
    Stmt.assign .ptr [TILE_M, TILE_K] "a_ptrs"
      (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [TILE_M, TILE_K] "a_ptrs")
        (Op.constNat TILE_K)),
    Stmt.assign .ptr [TILE_K, TILE_N] "b_ptrs"
      (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [TILE_K, TILE_N] "b_ptrs")
        (Op.mul .nat Broadcast.nil (Op.constNat TILE_K) (Op.constNat N))),
    Stmt.assign .real [TILE_M, TILE_N] "o"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [TILE_M, TILE_N] "o")
        (Op.dot (batch := []) (Op.ref .real [TILE_M, TILE_K] "a")
          (Op.ref .real [TILE_K, TILE_N] "b"))) ]

set_option maxRecDepth 8000 in
/-- **Body split (by `rfl`).** Sixteen top-level statements. -/
theorem bmm_body_eq (A B O : RegionName)
    (M N K TILE_M TILE_N TILE_K GROUP_M : Nat) :
    (bmm_surface A B O M N K TILE_M TILE_N TILE_K GROUP_M).toAlgKernel.body
      = [ Stmt.assign .nat [] "pid_b" (Op.programId 2),
          Stmt.assign .nat [] "pidx" (Op.programId 0),
          Stmt.assign .nat [] "pidy" (Op.programId 1),
          Stmt.ifThenElse
            (Op.eq ComparableDType.nat Broadcast.nil (Op.constNat GROUP_M)
              (Op.constNat 1))
            bmmGate1Body (bmmGateGrpBody GROUP_M),
          Stmt.assign .nat [TILE_M] "offs_m"
            (Op.add .nat Broadcast.scalarL
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m")
                (Op.constNat TILE_M))
              (Op.arange TILE_M)),
          Stmt.assign .nat [TILE_N] "offs_n"
            (Op.add .nat Broadcast.scalarL
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_n")
                (Op.constNat TILE_N))
              (Op.arange TILE_N)),
          Stmt.assign .nat [TILE_K] "offs_k" (Op.arange TILE_K),
          Stmt.assign .bool [TILE_M] "mask_m"
            (Op.lt ComparableDType.nat Broadcast.scalarR
              (Op.ref .nat [TILE_M] "offs_m") (Op.constNat M)),
          Stmt.assign .bool [TILE_N] "mask_n"
            (Op.lt ComparableDType.nat Broadcast.scalarR
              (Op.ref .nat [TILE_N] "offs_n") (Op.constNat N)),
          Stmt.assign .ptr [TILE_M, TILE_K] "a_ptrs"
            (Op.ptrAdd Broadcast.scalarL (Op.ptrBase A)
              (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
                (Op.add .nat Broadcast.scalarL
                  (Op.mul .nat Broadcast.nil
                    (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_b")
                      (Op.constNat M))
                    (Op.constNat K))
                  (Op.mul .nat Broadcast.scalarR
                    (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [TILE_M] "offs_m"))
                    (Op.constNat K)))
                (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [TILE_K] "offs_k")))),
          Stmt.assign .ptr [TILE_K, TILE_N] "b_ptrs"
            (Op.ptrAdd Broadcast.scalarL (Op.ptrBase B)
              (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
                (Op.add .nat Broadcast.scalarL
                  (Op.mul .nat Broadcast.nil
                    (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_b")
                      (Op.constNat K))
                    (Op.constNat N))
                  (Op.mul .nat Broadcast.scalarR
                    (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [TILE_K] "offs_k"))
                    (Op.constNat N)))
                (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [TILE_N] "offs_n")))),
          Stmt.assign .ptr [TILE_M, TILE_N] "o_ptrs"
            (Op.ptrAdd Broadcast.scalarL (Op.ptrBase O)
              (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
                (Op.add .nat Broadcast.scalarL
                  (Op.mul .nat Broadcast.nil
                    (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_b")
                      (Op.constNat M))
                    (Op.constNat N))
                  (Op.mul .nat Broadcast.scalarR
                    (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [TILE_M] "offs_m"))
                    (Op.constNat N)))
                (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [TILE_N] "offs_n")))),
          Stmt.assign .nat [] "num_iters"
            (Op.div .nat Broadcast.nil
              (Op.sub .nat Broadcast.nil
                (Op.add .nat Broadcast.nil (Op.constNat K) (Op.constNat TILE_K))
                (Op.constNat 1))
              (Op.constNat TILE_K)),
          Stmt.assign .real [TILE_M, TILE_N] "o"
            (Op.full [TILE_M, TILE_N] (Op.const 0)),
          Stmt.forRangeDyn "_i" (Op.constNat 0) (Op.ref .nat [] "num_iters")
            (Op.constNat 1) (bmmLoopBody N K TILE_M TILE_N TILE_K),
          Stmt.assign .bool [TILE_M, TILE_N] "mask_c"
            (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .bool [TILE_M] "mask_m"))
              (Op.expandDim ⟨0, by simp⟩ (Op.ref .bool [TILE_N] "mask_n"))),
          Stmt.store .real [TILE_M, TILE_N]
            (MemAccess.ptr (Op.ref .ptr [TILE_M, TILE_N] "o_ptrs"))
            (Op.ref .real [TILE_M, TILE_N] "o")
            (MaskOpt.mask (Op.ref .bool [TILE_M, TILE_N] "mask_c")) ] := by
  rfl

/-! ## CTA-swizzle closed forms and the GEMM specification -/

/-- The runtime `GROUP_SIZE` boundary gate: `gridx % GROUP_M` on a ragged
final group, else `GROUP_M`. -/
def bmmGroupSize (gridx gid GROUP_M : Nat) : Nat :=
  if gid * GROUP_M + GROUP_M > gridx then gridx % GROUP_M else GROUP_M

/-- `pid_m` under the CTA reorder (identity when `GROUP_M = 1`). -/
def bmmPidM (pidx pidy gridx gridy GROUP_M : Nat) : Nat :=
  if GROUP_M = 1 then pidx
  else
    (pidx + pidy * gridx) / (gridy * GROUP_M) * GROUP_M
      + (pidx + pidy * gridx) % (gridy * GROUP_M)
          % bmmGroupSize gridx ((pidx + pidy * gridx) / (gridy * GROUP_M)) GROUP_M

/-- `pid_n` under the CTA reorder (identity when `GROUP_M = 1`). -/
def bmmPidN (pidx pidy gridx gridy GROUP_M : Nat) : Nat :=
  if GROUP_M = 1 then pidy
  else
    (pidx + pidy * gridx) % (gridy * GROUP_M)
      / bmmGroupSize gridx ((pidx + pidy * gridx) / (gridy * GROUP_M)) GROUP_M

/-- The global output row of tile lane `i`. -/
def bmmRowG (s : BlockState) (TILE_M GROUP_M : Nat) (i : Nat) : Nat :=
  bmmPidM (s.pids 0) (s.pids 1) (s.numPids 0) (s.numPids 1) GROUP_M * TILE_M + i

/-- The global output column of tile lane `j`. -/
def bmmColG (s : BlockState) (TILE_N GROUP_M : Nat) (j : Nat) : Nat :=
  bmmPidN (s.pids 0) (s.pids 1) (s.numPids 0) (s.numPids 1) GROUP_M * TILE_N + j

/-- `A[pid_b, m, t]` (row-major `M × K` batch slab). -/
noncomputable def bmmAVal (s : BlockState) (A : RegionName) (M K m t : Nat) : ℝ :=
  s.readMem A (s.pids 2 * M * K + m * K + t)

/-- `B[pid_b, t, n]` (row-major `K × N` batch slab). -/
noncomputable def bmmBVal (s : BlockState) (B : RegionName) (K N t n : Nat) : ℝ :=
  s.readMem B (s.pids 2 * K * N + t * N + n)

/-- The `mask_a`-guarded `a` lane. -/
noncomputable def bmmAGuarded (s : BlockState) (A : RegionName)
    (M K m t : Nat) : ℝ :=
  if m < M ∧ t < K then bmmAVal s A M K m t else 0

/-- The `mask_b`-guarded `b` lane. -/
noncomputable def bmmBGuarded (s : BlockState) (B : RegionName)
    (K N t n : Nat) : ℝ :=
  if t < K ∧ n < N then bmmBVal s B K N t n else 0

/-- `num_iters = tl.cdiv(K, TILE_K)`. -/
def bmmNI (K TILE_K : Nat) : Nat := (K + TILE_K - 1) / TILE_K

/-- The accumulator after `c` K-blocks: the guarded partial GEMM over keys
`[0, c·TILE_K)` at global lanes `(m, n)`. -/
noncomputable def bmmAcc (s : BlockState) (A B : RegionName)
    (M N K TILE_K : Nat) (c m n : Nat) : ℝ :=
  ∑ t ∈ Finset.range (c * TILE_K),
    bmmAGuarded s A M K m t * bmmBGuarded s B K N t n

/-- **The stored `o` lane** — the batched GEMM closed form
`Σ_{t < K} A[pid_b, m, t] · B[pid_b, t, n]` (raw reads; the store readback
is claimed at in-window lanes only). -/
noncomputable def bmmOOut (s : BlockState) (A B : RegionName)
    (M N K : Nat) (m n : Nat) : ℝ :=
  ∑ t ∈ Finset.range K, bmmAVal s A M K m t * bmmBVal s B K N t n

private theorem bmm_sum_range_add_block (n BTS : Nat) (f : Nat → ℝ) :
    ∑ t ∈ Finset.range (n + BTS), f t
      = (∑ t ∈ Finset.range n, f t) + ∑ c : Fin BTS, f (n + c.val) := by
  rw [Finset.sum_range_add]
  congr 1
  rw [Fin.sum_univ_eq_sum_range (fun c => f (n + c))]

/-- One K-block appended to the accumulator. -/
private theorem bmmAcc_step (s : BlockState) (A B : RegionName)
    (M N K TILE_K : Nat) (c m n : Nat) :
    bmmAcc s A B M N K TILE_K c m n
      + ∑ t : Fin TILE_K,
          bmmAGuarded s A M K m (c * TILE_K + t.val)
            * bmmBGuarded s B K N (c * TILE_K + t.val) n
      = bmmAcc s A B M N K TILE_K (c + 1) m n := by
  unfold bmmAcc
  rw [show (c + 1) * TILE_K = c * TILE_K + TILE_K from by ring,
    bmm_sum_range_add_block]

/-- After the full loop, the guarded accumulator collapses to the raw GEMM
closed form at every in-window lane. -/
private theorem bmmAcc_final (s : BlockState) (A B : RegionName)
    (M N K TILE_K : Nat) (m n : Nat)
    (hTK : 0 < TILE_K) (hm : m < M) (hn : n < N) :
    bmmAcc s A B M N K TILE_K (bmmNI K TILE_K) m n
      = bmmOOut s A B M N K m n := by
  unfold bmmAcc bmmOOut
  have hcov : K ≤ bmmNI K TILE_K * TILE_K := by
    unfold bmmNI
    have hdm := Nat.div_add_mod (K + TILE_K - 1) TILE_K
    have hlt := Nat.mod_lt (K + TILE_K - 1) hTK
    rw [Nat.mul_comm]
    omega
  rw [← Finset.sum_subset
    (fun t ht => Finset.mem_range.mpr
      (lt_of_lt_of_le (Finset.mem_range.mp ht) hcov))
    (fun t _ ht => by
      have : ¬ t < K := fun h => ht (Finset.mem_range.mpr h)
      simp [bmmAGuarded, this])]
  refine Finset.sum_congr rfl fun t ht => ?_
  have htK : t < K := Finset.mem_range.mp ht
  rw [bmmAGuarded, if_pos ⟨hm, htK⟩, bmmBGuarded, if_pos ⟨htK, hn⟩]

/-- The `o` store address at lane `(i, j)`. -/
def bmmOOffset (s : BlockState) (M N TILE_M TILE_N GROUP_M : Nat)
    (idx : TileIndex [TILE_M, TILE_N]) : Nat :=
  s.pids 2 * M * N + bmmRowG s TILE_M GROUP_M idx.1.val * N
    + bmmColG s TILE_N GROUP_M idx.2.1.val

/-- An `o` store lane is *active* when it maps inside the `M × N` window. -/
def bmmOActive (s : BlockState) (M N TILE_M TILE_N GROUP_M : Nat)
    (idx : TileIndex [TILE_M, TILE_N]) : Prop :=
  bmmRowG s TILE_M GROUP_M idx.1.val < M ∧ bmmColG s TILE_N GROUP_M idx.2.1.val < N

instance bmmOActiveDecidable (s : BlockState) (M N TILE_M TILE_N GROUP_M : Nat)
    (idx : TileIndex [TILE_M, TILE_N]) :
    Decidable (bmmOActive s M N TILE_M TILE_N GROUP_M idx) := by
  unfold bmmOActive
  infer_instance

/-! ## Eval recipes -/

private theorem bmm_ifThenElse_step_bool (b : Bool) (cond : Op .bool [])
    (thenB elseB : List Stmt) (t : BlockState)
    (hcond : evalOp cond t = some (Tile.scalar b)) :
    stepStmt (Stmt.ifThenElse cond thenB elseB) t
      = if b then stepStmts thenB t else stepStmts elseB t := by
  have h : stepStmt (Stmt.ifThenElse cond thenB elseB) t
      = (evalOp cond t).bind
          (fun c => if c.data PUnit.unit then stepStmts thenB t
            else stepStmts elseB t) := by
    unfold stepStmt
    cases evalOp cond t <;> rfl
  rw [h, hcond]
  rfl

private theorem bmm_gateCond_eval (s : BlockState) (GROUP_M : Nat) :
    evalOp (Op.eq ComparableDType.nat Broadcast.nil (Op.constNat GROUP_M)
      (Op.constNat 1)) s = some (Tile.scalar (decide (GROUP_M = 1))) := by
  simp only [evalOp, evalOp_constNat, bind, Option.bind]
  rfl

private theorem bmm_innerCond_eval (s : BlockState) (GROUP_M g gx : Nat)
    (hg : s.regs .nat [] "group_id" = some (Tile.scalar g))
    (hgx : s.regs .nat [] "gridx" = some (Tile.scalar gx)) :
    evalOp (Op.gt ComparableDType.nat Broadcast.nil
        (Op.add .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "group_id")
            (Op.constNat GROUP_M))
          (Op.constNat GROUP_M))
        (Op.ref .nat [] "gridx")) s
      = some (Tile.scalar (decide (g * GROUP_M + GROUP_M > gx))) := by
  simp only [evalOp, evalOp_ref, hg, hgx, evalOp_constNat, bind, Option.bind]
  rfl

private theorem bmm_mulConst_eval (s : BlockState) (name : RegName) (val c : Nat)
    (hr : s.regs .nat [] name = some (Tile.scalar val)) :
    evalOp (Op.mul .nat Broadcast.nil (Op.ref .nat [] name) (Op.constNat c)) s
      = some (Tile.scalar (val * c)) := by
  rw [evalOp_mul]
  simp only [evalOp_ref, evalOp_constNat, hr, Option.bind_eq_bind, Option.bind_some]
  rfl

private theorem bmm_pid_eval (s : BlockState) (px py gx : Nat)
    (hx : s.regs .nat [] "pidx" = some (Tile.scalar px))
    (hy : s.regs .nat [] "pidy" = some (Tile.scalar py))
    (hgx : s.regs .nat [] "gridx" = some (Tile.scalar gx)) :
    evalOp (Op.add .nat Broadcast.nil (Op.ref .nat [] "pidx")
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pidy")
          (Op.ref .nat [] "gridx"))) s
      = some (Tile.scalar (px + py * gx)) := by
  simp only [evalOp, evalOp_ref, hx, hy, hgx, bind, Option.bind]
  rfl

private theorem bmm_floorDiv_refs_eval (s : BlockState) (na nb : RegName)
    (va vb : Nat)
    (ha : s.regs .nat [] na = some (Tile.scalar va))
    (hb : s.regs .nat [] nb = some (Tile.scalar vb)) :
    evalOp (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref .nat [] na)
        (Op.ref .nat [] nb)) s
      = some (Tile.scalar (va / vb)) := by
  rw [evalOp_floorDiv]
  simp only [evalOp_ref, ha, hb, Option.bind_eq_bind, Option.bind_some]
  rfl

private theorem bmm_mod_refs_eval (s : BlockState) (na nb : RegName)
    (va vb : Nat)
    (ha : s.regs .nat [] na = some (Tile.scalar va))
    (hb : s.regs .nat [] nb = some (Tile.scalar vb)) :
    evalOp (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] na)
        (Op.ref .nat [] nb)) s
      = some (Tile.scalar (va % vb)) := by
  rw [evalOp_mod]
  simp only [evalOp_ref, ha, hb, Option.bind_eq_bind, Option.bind_some]
  rfl

private theorem bmm_mod_ref_const_eval (s : BlockState) (na : RegName)
    (va c : Nat)
    (ha : s.regs .nat [] na = some (Tile.scalar va)) :
    evalOp (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] na)
        (Op.constNat c)) s
      = some (Tile.scalar (va % c)) := by
  rw [evalOp_mod]
  simp only [evalOp_ref, evalOp_constNat, ha, Option.bind_eq_bind, Option.bind_some]
  rfl

private theorem bmm_pidm_eval (s : BlockState) (GROUP_M g ig gs : Nat)
    (hg : s.regs .nat [] "group_id" = some (Tile.scalar g))
    (hig : s.regs .nat [] "inner_group_id" = some (Tile.scalar ig))
    (hgs : s.regs .nat [] "GROUP_SIZE" = some (Tile.scalar gs)) :
    evalOp (Op.add .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "group_id")
          (Op.constNat GROUP_M))
        (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "inner_group_id")
          (Op.ref .nat [] "GROUP_SIZE"))) s
      = some (Tile.scalar (g * GROUP_M + ig % gs)) := by
  simp only [evalOp, evalOp_ref, hg, hig, hgs, evalOp_constNat, bind, Option.bind]
  rfl

/-- `offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)` (generic in the pid
register name). -/
private theorem bmm_offs_eval (s : BlockState) (TM : Nat) (pm : Nat)
    (pidReg : RegName)
    (hpm : s.regs .nat [] pidReg = some (Tile.scalar pm)) :
    evalOp (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] pidReg) (Op.constNat TM))
        (Op.arange TM)) s
      = some (Tile.vec (fun i : Fin TM => pm * TM + i.val)) := by
  simp only [evalOp_add, evalOp_mul, evalOp_ref, evalOp_constNat, evalOp_arange,
    hpm, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext i
  simp only [Tile.bop_data, Tile.scalar_data, Tile.vec_data, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add, NumericDType.mul]

/-- A `< const` mask over a `Tile.vec` register. -/
private theorem bmm_maskLt_eval (s : BlockState) (n c : Nat) (name : RegName)
    (g : Fin n → Nat)
    (hr : s.regs .nat [n] name = some (Tile.vec g)) :
    evalOp (Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref .nat [n] name)
        (Op.constNat c)) s
      = some (⟨fun i => decide (g i.1 < c)⟩ : Tile .bool [n]) := by
  rw [evalOp_lt]
  simp only [evalOp_ref, hr, evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext i
  simp [Tile.cop_data, Tile.vec, Tile.scalar_data, ComparableDType.lt,
    Broadcast.leftIndex, Broadcast.rightIndex]

private theorem bmm_evalOp_boolAnd {a b shape} (bc : Broadcast a b shape)
    (x : Op .bool a) (y : Op .bool b) (s : BlockState) :
    evalOp (.boolAnd bc x y) s = (do
      let vx ← evalOp x s; let vy ← evalOp y s; some (Tile.bop (· && ·) bc vx vy)) := by
  simp [evalOp]

/-- The 2-D `&` of two expanded 1-D masks. -/
private theorem bmm_mask2d_eval (s : BlockState) (mR nR : RegName)
    (m n : Nat) (P : Fin m → Prop) (Q : Fin n → Prop)
    [DecidablePred P] [DecidablePred Q]
    (hm : s.regs .bool [m] mR = some ⟨fun i => decide (P i.1)⟩)
    (hn : s.regs .bool [n] nR = some ⟨fun j => decide (Q j.1)⟩) :
    evalOp (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .bool [m] mR))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .bool [n] nR))) s
      = some (⟨fun idx : TileIndex [m, n] =>
          decide (P idx.1) && decide (Q idx.2.1)⟩ : Tile .bool [m, n]) := by
  rw [bmm_evalOp_boolAnd]
  erw [evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hm,
    evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hn]
  simp only [Option.bind_eq_bind, Option.bind_some]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨i, j, u⟩ := idx
  rfl

/-- The batched pointer-tile construction
`R + (pid_b·D₁·D₂ + rows[:,None]·σ + cols[None,:])` (generic in the two
`Tile.vec` index registers). -/
private theorem bmm_ptrs_eval (s : BlockState) (R : RegionName)
    (rowsR colsR : RegName) (m n : Nat) (base σ : Nat)
    (baseOp : Op .nat []) (gr : Fin m → Nat) (gc : Fin n → Nat)
    (hbase : evalOp baseOp s = some (Tile.scalar base))
    (hr : s.regs .nat [m] rowsR = some (Tile.vec gr))
    (hc : s.regs .nat [n] colsR = some (Tile.vec gc)) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase R)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.add .nat Broadcast.scalarL baseOp
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [m] rowsR))
              (Op.constNat σ)))
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [n] colsR)))) s
      = some (⟨fun idx : TileIndex [m, n] =>
          (R, base + gr idx.1 * σ + gc idx.2.1)⟩ : Tile .ptr [m, n]) := by
  rw [evalOp_ptrAdd, evalOp_ptrBase]
  rw [evalOp_add, evalOp_add, evalOp_mul, hbase]
  erw [evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hr,
    evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hc]
  simp only [evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨i, j, u⟩ := idx
  show ((R : RegionName), 0 + (base + gr i * σ + gc j))
    = ((R : RegionName), base + gr i * σ + gc j)
  rw [Nat.zero_add]

/-- Advance a pointer tile by a scalar constant. -/
private theorem bmm_ptrAdv_const_eval (s : BlockState) (name : RegName)
    {m n : Nat} (f : TileIndex [m, n] → RegionName × Nat) (d : Nat)
    (hp : s.regs .ptr [m, n] name = some ⟨f⟩) :
    evalOp (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [m, n] name)
        (Op.constNat d)) s
      = some (⟨fun idx => ((f idx).1, (f idx).2 + d)⟩ : Tile .ptr [m, n]) := by
  rw [evalOp_ptrAdd]
  simp only [evalOp_ref, hp, evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
  rfl

/-- Advance a pointer tile by a product of spliced constants. -/
private theorem bmm_ptrAdv_mul_eval (s : BlockState) (name : RegName)
    {m n : Nat} (f : TileIndex [m, n] → RegionName × Nat) (d e : Nat)
    (hp : s.regs .ptr [m, n] name = some ⟨f⟩) :
    evalOp (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [m, n] name)
        (Op.mul .nat Broadcast.nil (Op.constNat d) (Op.constNat e))) s
      = some (⟨fun idx => ((f idx).1, (f idx).2 + d * e)⟩ : Tile .ptr [m, n]) := by
  rw [evalOp_ptrAdd, evalOp_mul]
  simp only [evalOp_ref, hp, evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
  rfl

private theorem bmm_NI_eval (s : BlockState) (K TILE_K : Nat) :
    evalOp (Op.div .nat Broadcast.nil
        (Op.sub .nat Broadcast.nil
          (Op.add .nat Broadcast.nil (Op.constNat K) (Op.constNat TILE_K))
          (Op.constNat 1))
        (Op.constNat TILE_K)) s
      = some (Tile.scalar (bmmNI K TILE_K)) := by
  simp only [evalOp, evalOp_constNat, bind, Option.bind]
  rfl

private theorem bmm_zeros_eval (sh : TileShape) (t : BlockState) :
    evalOp (Op.full sh (Op.const 0)) t
      = some (⟨fun _ => some (0 : ℝ)⟩ : Tile .real sh) := by
  simp [evalOp_full, evalOp_const]

private theorem bmm_okAdd_eval (sin : BlockState) (TK off : Nat)
    (hok : sin.regs .nat [TK] "offs_k" = some (Tile.vec fun r => off + r.val)) :
    evalOp (Op.add .nat Broadcast.scalarR (Op.ref .nat [TK] "offs_k")
        (Op.constNat TK)) sin
      = some (Tile.vec fun r : Fin TK => (off + TK) + r.val) := by
  rw [evalOp_add]
  simp only [evalOp_ref, hok, evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨r, u⟩ := idx
  show off + r.val + TK = off + TK + r.val
  omega

/-- Masked plain-pointer 2-D load on a clean-`undef` state: kept lanes read
memory (bridged to the launch state via `hmem`), masked-off lanes read the
(zero) `undef` channel. -/
private theorem bmm_maskedLoad_eval (sin s0 : BlockState) (pR mR : RegName)
    (rg : RegionName) {m n : Nat} (off : TileIndex [m, n] → Nat)
    (P : TileIndex [m, n] → Prop) [DecidablePred P]
    (hundef : ∀ r o, sin.undef r o = 0)
    (hmem : sin.mem = s0.mem)
    (hp : sin.regs .ptr [m, n] pR = some ⟨fun idx => (rg, off idx)⟩)
    (hm : sin.regs .bool [m, n] mR = some ⟨fun idx => decide (P idx)⟩) :
    evalOp (Op.load .real (MemAccess.ptr (Op.ref .ptr [m, n] pR))
        (MaskOpt.mask (Op.ref .bool [m, n] mR))) sin
      = some ⟨fun idx => some (if P idx then s0.readMem rg (off idx) else 0)⟩ := by
  simp only [evalOp, evalOp_ref, hp, hm, bind, Option.bind]
  refine congrArg some (Tile.ext fun idx => ?_)
  have hread : ∀ o, sin.readMem rg o = s0.readMem rg o := by
    intro o
    unfold BlockState.readMem
    rw [hmem]
  by_cases h : P idx
  · simp [h, BlockState.readMemValue_real, hread]
  · simp [h, hundef]

/-- `o (+)= tl.dot(a, b)` on all-`some` tiles: lanewise contraction. -/
private theorem bmm_oAdd_eval (sin : BlockState) (TM TK TN : Nat)
    (g : Nat → Nat → ℝ) (f : Nat → Nat → ℝ) (w : Nat → Nat → ℝ)
    (ho : sin.regs .real [TM, TN] "o"
      = some ⟨fun idx => some (g idx.1.val idx.2.1.val)⟩)
    (ha : sin.regs .real [TM, TK] "a"
      = some ⟨fun idx => some (f idx.1.val idx.2.1.val)⟩)
    (hb : sin.regs .real [TK, TN] "b"
      = some ⟨fun idx => some (w idx.1.val idx.2.1.val)⟩) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [TM, TN] "o")
        (Op.dot (batch := []) (Op.ref .real [TM, TK] "a")
          (Op.ref .real [TK, TN] "b"))) sin
      = some ⟨fun idx : TileIndex [TM, TN] =>
          some (g idx.1.val idx.2.1.val
            + ∑ c : Fin TK, f idx.1.val c.val * w c.val idx.2.1.val)⟩ := by
  have hdot : evalOp (Op.dot (batch := []) (Op.ref .real [TM, TK] "a")
      (Op.ref .real [TK, TN] "b")) sin
      = some (Tile.dot [] (⟨fun idx => some (f idx.1.val idx.2.1.val)⟩ : Tile .real [TM, TK])
          (⟨fun idx => some (w idx.1.val idx.2.1.val)⟩ : Tile .real [TK, TN])) := by
    erw [evalOp_dot]
    simp [evalOp_ref, ha, hb]
  have hadd : evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
      (Op.ref .real [TM, TN] "o")
      (Op.dot (batch := []) (Op.ref .real [TM, TK] "a")
        (Op.ref .real [TK, TN] "b"))) sin
      = some (Tile.bop NumericDType.real.add
          (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          (⟨fun idx => some (g idx.1.val idx.2.1.val)⟩ : Tile .real [TM, TN])
          (Tile.dot [] (⟨fun idx => some (f idx.1.val idx.2.1.val)⟩ : Tile .real [TM, TK])
            (⟨fun idx => some (w idx.1.val idx.2.1.val)⟩ : Tile .real [TK, TN]))) := by
    rw [evalOp_add]
    simp only [evalOp_ref, ho, Option.bind_eq_bind, Option.bind_some]
    erw [hdot]
    rfl
  erw [hadd]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨i, j, u⟩ := idx
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex]
  rw [Tile.dot_nil_data]
  rw [show (@Finset.sum (Fin TK) (WithBot ℝ) _ Finset.univ
        (fun e => Option.map₂ (· * ·)
          ((⟨fun idx => some (f idx.1.val idx.2.1.val)⟩ : Tile .real [TM, TK]).data (i, e, PUnit.unit))
          ((⟨fun idx => some (w idx.1.val idx.2.1.val)⟩ : Tile .real [TK, TN]).data (e, j, PUnit.unit))))
      = @Finset.sum (Fin TK) (WithBot ℝ) _ Finset.univ
          (fun e => (some (f i.val e.val * w e.val j.val) : WithBot ℝ))
      from Finset.sum_congr rfl (fun e _ => rfl)]
  rw [withBot_sum_some]
  rfl

/-! ## The CTA-reorder gate -/

/-- The runtime `GROUP_SIZE` boundary gate lands on `bmmGroupSize`. -/
private theorem bmm_groupSize_step (u : BlockState) (GROUP_M g gx : Nat)
    (hg : u.regs .nat [] "group_id" = some (Tile.scalar g))
    (hgx : u.regs .nat [] "gridx" = some (Tile.scalar gx)) :
    stepStmt (Stmt.ifThenElse
        (Op.gt ComparableDType.nat Broadcast.nil
          (Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "group_id")
              (Op.constNat GROUP_M))
            (Op.constNat GROUP_M))
          (Op.ref .nat [] "gridx"))
        [ Stmt.assign .nat [] "GROUP_SIZE"
            (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "gridx")
              (Op.constNat GROUP_M)) ]
        [ Stmt.assign .nat [] "GROUP_SIZE" (Op.constNat GROUP_M) ]) u
      = some (u.setReg "GROUP_SIZE" .nat []
          (Tile.scalar (bmmGroupSize gx g GROUP_M))) := by
  rw [bmm_ifThenElse_step_bool (decide (g * GROUP_M + GROUP_M > gx)) _ _ _ u
    (bmm_innerCond_eval u GROUP_M g gx hg hgx)]
  by_cases hb : g * GROUP_M + GROUP_M > gx
  · rw [if_pos (by simpa using hb)]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (bmm_mod_ref_const_eval u "gridx" gx GROUP_M hgx))]
    rw [stepStmts.nil]
    rw [show bmmGroupSize gx g GROUP_M = gx % GROUP_M from by
      rw [bmmGroupSize, if_pos hb]]
  · rw [if_neg (by simpa using hb)]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_constNat GROUP_M u))]
    rw [stepStmts.nil]
    rw [show bmmGroupSize gx g GROUP_M = GROUP_M from by
      rw [bmmGroupSize, if_neg hb]]

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **The `GROUP_M` gate**: both arms land `pid_m`/`pid_n` on the closed
forms `bmmPidM`/`bmmPidN`, preserve `pids`/`mem`/`undef`/`numPids`, and do
not touch the `pid_b` register. -/
theorem bmmGate_step (t : BlockState) (GROUP_M : Nat) (px py : Nat)
    (hx : t.regs .nat [] "pidx" = some (Tile.scalar px))
    (hy : t.regs .nat [] "pidy" = some (Tile.scalar py)) :
    ∃ t', stepStmt (Stmt.ifThenElse
        (Op.eq ComparableDType.nat Broadcast.nil (Op.constNat GROUP_M)
          (Op.constNat 1))
        bmmGate1Body (bmmGateGrpBody GROUP_M)) t = some t'
      ∧ t'.pids = t.pids ∧ t'.mem = t.mem ∧ t'.undef = t.undef
      ∧ t'.regs .nat [] "pid_b" = t.regs .nat [] "pid_b"
      ∧ t'.regs .nat [] "pid_m"
          = some (Tile.scalar (bmmPidM px py (t.numPids 0) (t.numPids 1) GROUP_M))
      ∧ t'.regs .nat [] "pid_n"
          = some (Tile.scalar (bmmPidN px py (t.numPids 0) (t.numPids 1) GROUP_M)) := by
  rw [bmm_ifThenElse_step_bool (decide (GROUP_M = 1)) _ _ _ t
    (bmm_gateCond_eval t GROUP_M)]
  by_cases hg : GROUP_M = 1
  · rw [if_pos (by simp [hg])]
    unfold bmmGate1Body
    rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (Op.ref .nat [] "pidx") t = some (Tile.scalar px) from by
        rw [evalOp_ref, hx]))]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (Op.ref .nat [] "pidy") _ = some (Tile.scalar py) from by
        rw [evalOp_ref]; simpa using hy))]
    rw [stepStmts.nil]
    exact ⟨_, rfl, by first | rfl | simp, by first | rfl | simp,
      by first | rfl | simp, by first | rfl | simp,
      by simp [bmmPidM, hg], by simp [bmmPidN, hg]⟩
  · rw [if_neg (by simp [hg])]
    unfold bmmGateGrpBody
    rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_numPrograms 0 t))]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_numPrograms 1 _))]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (bmm_pid_eval _ px py (t.numPids 0)
        (by simpa using hx) (by simpa using hy) (by simp)))]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (bmm_mulConst_eval _ "gridy" (t.numPids 1) GROUP_M (by simp)))]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (bmm_floorDiv_refs_eval _ "pid" "num_CTA_per_group"
        (px + py * t.numPids 0) (t.numPids 1 * GROUP_M) (by simp) (by simp)))]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (bmm_mod_refs_eval _ "pid" "num_CTA_per_group"
        (px + py * t.numPids 0) (t.numPids 1 * GROUP_M) (by simp) (by simp)))]
    rw [stepStmts.cons_some (bmm_groupSize_step _ GROUP_M
      ((px + py * t.numPids 0) / (t.numPids 1 * GROUP_M)) (t.numPids 0)
      (by simp) (by simp))]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (bmm_pidm_eval _ GROUP_M
        ((px + py * t.numPids 0) / (t.numPids 1 * GROUP_M))
        ((px + py * t.numPids 0) % (t.numPids 1 * GROUP_M))
        (bmmGroupSize (t.numPids 0)
          ((px + py * t.numPids 0) / (t.numPids 1 * GROUP_M)) GROUP_M)
        (by simp) (by simp) (by simp)))]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (bmm_floorDiv_refs_eval _ "inner_group_id" "GROUP_SIZE"
        ((px + py * t.numPids 0) % (t.numPids 1 * GROUP_M))
        (bmmGroupSize (t.numPids 0)
          ((px + py * t.numPids 0) / (t.numPids 1 * GROUP_M)) GROUP_M)
        (by simp) (by simp)))]
    rw [stepStmts.nil]
    exact ⟨_, rfl, by first | rfl | simp, by first | rfl | simp,
      by first | rfl | simp, by first | rfl | simp,
      by simp [bmmPidM, hg],
      by simp [bmmPidN, hg]⟩

/-! ## Loop invariant, prologue, and the K loop -/

/-- The loop invariant after `c` K-blocks. -/
def bmmInv (s0 : BlockState) (A B O : RegionName)
    (M N K TILE_M TILE_N TILE_K GROUP_M : Nat) (c : Nat) (s : BlockState) : Prop :=
  s.pids = s0.pids ∧ s.mem = s0.mem ∧ (∀ rg o, s.undef rg o = 0) ∧
  c ≤ bmmNI K TILE_K ∧
  s.regs .nat [] "num_iters" = some (Tile.scalar (bmmNI K TILE_K)) ∧
  s.regs .bool [TILE_M] "mask_m"
    = some ⟨fun i => decide (bmmRowG s0 TILE_M GROUP_M i.1.val < M)⟩ ∧
  s.regs .bool [TILE_N] "mask_n"
    = some ⟨fun j => decide (bmmColG s0 TILE_N GROUP_M j.1.val < N)⟩ ∧
  s.regs .nat [TILE_K] "offs_k"
    = some (Tile.vec fun r => c * TILE_K + r.val) ∧
  s.regs .ptr [TILE_M, TILE_K] "a_ptrs"
    = some ⟨fun idx => (A, s0.pids 2 * M * K
        + bmmRowG s0 TILE_M GROUP_M idx.1.val * K
        + (c * TILE_K + idx.2.1.val))⟩ ∧
  s.regs .ptr [TILE_K, TILE_N] "b_ptrs"
    = some ⟨fun idx => (B, s0.pids 2 * K * N
        + (c * TILE_K + idx.1.val) * N
        + bmmColG s0 TILE_N GROUP_M idx.2.1.val)⟩ ∧
  s.regs .ptr [TILE_M, TILE_N] "o_ptrs"
    = some ⟨fun idx => (O, bmmOOffset s0 M N TILE_M TILE_N GROUP_M idx)⟩ ∧
  s.regs .real [TILE_M, TILE_N] "o"
    = some ⟨fun idx => some (bmmAcc s0 A B M N K TILE_K c
        (bmmRowG s0 TILE_M GROUP_M idx.1.val)
        (bmmColG s0 TILE_N GROUP_M idx.2.1.val))⟩

private theorem bmm_base_eval (s : BlockState) (v D1 D2 : Nat)
    (hb : s.regs .nat [] "pid_b" = some (Tile.scalar v)) :
    evalOp (Op.mul .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_b") (Op.constNat D1))
        (Op.constNat D2)) s
      = some (Tile.scalar (v * D1 * D2)) := by
  simp only [evalOp, evalOp_ref, hb, evalOp_constNat, bind, Option.bind]
  rfl

set_option maxHeartbeats 8000000 in
set_option maxRecDepth 8000 in
/-- **The prologue** (statements 1–14): program ids, the CTA-reorder gate,
offsets, boundary masks, the three batched pointer tiles, `num_iters`, and
the zero accumulator establish `bmmInv` at `0`. -/
theorem bmmPrologue_run (s : BlockState) (A B O : RegionName)
    (M N K TILE_M TILE_N TILE_K GROUP_M : Nat)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ sP, stepStmts
        [ Stmt.assign .nat [] "pid_b" (Op.programId 2),
          Stmt.assign .nat [] "pidx" (Op.programId 0),
          Stmt.assign .nat [] "pidy" (Op.programId 1),
          Stmt.ifThenElse
            (Op.eq ComparableDType.nat Broadcast.nil (Op.constNat GROUP_M)
              (Op.constNat 1))
            bmmGate1Body (bmmGateGrpBody GROUP_M),
          Stmt.assign .nat [TILE_M] "offs_m"
            (Op.add .nat Broadcast.scalarL
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m")
                (Op.constNat TILE_M))
              (Op.arange TILE_M)),
          Stmt.assign .nat [TILE_N] "offs_n"
            (Op.add .nat Broadcast.scalarL
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_n")
                (Op.constNat TILE_N))
              (Op.arange TILE_N)),
          Stmt.assign .nat [TILE_K] "offs_k" (Op.arange TILE_K),
          Stmt.assign .bool [TILE_M] "mask_m"
            (Op.lt ComparableDType.nat Broadcast.scalarR
              (Op.ref .nat [TILE_M] "offs_m") (Op.constNat M)),
          Stmt.assign .bool [TILE_N] "mask_n"
            (Op.lt ComparableDType.nat Broadcast.scalarR
              (Op.ref .nat [TILE_N] "offs_n") (Op.constNat N)),
          Stmt.assign .ptr [TILE_M, TILE_K] "a_ptrs"
            (Op.ptrAdd Broadcast.scalarL (Op.ptrBase A)
              (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
                (Op.add .nat Broadcast.scalarL
                  (Op.mul .nat Broadcast.nil
                    (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_b")
                      (Op.constNat M))
                    (Op.constNat K))
                  (Op.mul .nat Broadcast.scalarR
                    (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [TILE_M] "offs_m"))
                    (Op.constNat K)))
                (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [TILE_K] "offs_k")))),
          Stmt.assign .ptr [TILE_K, TILE_N] "b_ptrs"
            (Op.ptrAdd Broadcast.scalarL (Op.ptrBase B)
              (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
                (Op.add .nat Broadcast.scalarL
                  (Op.mul .nat Broadcast.nil
                    (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_b")
                      (Op.constNat K))
                    (Op.constNat N))
                  (Op.mul .nat Broadcast.scalarR
                    (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [TILE_K] "offs_k"))
                    (Op.constNat N)))
                (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [TILE_N] "offs_n")))),
          Stmt.assign .ptr [TILE_M, TILE_N] "o_ptrs"
            (Op.ptrAdd Broadcast.scalarL (Op.ptrBase O)
              (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
                (Op.add .nat Broadcast.scalarL
                  (Op.mul .nat Broadcast.nil
                    (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_b")
                      (Op.constNat M))
                    (Op.constNat N))
                  (Op.mul .nat Broadcast.scalarR
                    (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [TILE_M] "offs_m"))
                    (Op.constNat N)))
                (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [TILE_N] "offs_n")))),
          Stmt.assign .nat [] "num_iters"
            (Op.div .nat Broadcast.nil
              (Op.sub .nat Broadcast.nil
                (Op.add .nat Broadcast.nil (Op.constNat K) (Op.constNat TILE_K))
                (Op.constNat 1))
              (Op.constNat TILE_K)),
          Stmt.assign .real [TILE_M, TILE_N] "o"
            (Op.full [TILE_M, TILE_N] (Op.const 0)) ] s = some sP
      ∧ bmmInv s A B O M N K TILE_M TILE_N TILE_K GROUP_M 0 sP := by
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 2 s))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 1 _))]
  simp only [BlockState.setReg_pids]
  obtain ⟨sG, hG, hGpids, hGmem, hGundef, hGpidb, hGpm, hGpn⟩ :=
    bmmGate_step (((s.setReg "pid_b" .nat [] (Tile.scalar (s.pids 2))).setReg
        "pidx" .nat [] (Tile.scalar (s.pids 0))).setReg
        "pidy" .nat [] (Tile.scalar (s.pids 1)))
      GROUP_M (s.pids 0) (s.pids 1) (by simp) (by simp)
  rw [stepStmts.cons_some hG]
  have hGpm' : sG.regs .nat [] "pid_m"
      = some (Tile.scalar (bmmPidM (s.pids 0) (s.pids 1) (s.numPids 0)
          (s.numPids 1) GROUP_M)) := by simpa using hGpm
  have hGpn' : sG.regs .nat [] "pid_n"
      = some (Tile.scalar (bmmPidN (s.pids 0) (s.pids 1) (s.numPids 0)
          (s.numPids 1) GROUP_M)) := by simpa using hGpn
  have hpidb : sG.regs .nat [] "pid_b" = some (Tile.scalar (s.pids 2)) := by
    rw [hGpidb]; simp
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bmm_offs_eval _ TILE_M _ "pid_m" hGpm'))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bmm_offs_eval _ TILE_N _ "pid_n" (by simpa using hGpn')))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_arange TILE_K _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bmm_maskLt_eval _ TILE_M M "offs_m"
      (fun i : Fin TILE_M => bmmPidM (s.pids 0) (s.pids 1) (s.numPids 0)
        (s.numPids 1) GROUP_M * TILE_M + i.val)
      (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bmm_maskLt_eval _ TILE_N N "offs_n"
      (fun j : Fin TILE_N => bmmPidN (s.pids 0) (s.pids 1) (s.numPids 0)
        (s.numPids 1) GROUP_M * TILE_N + j.val)
      (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bmm_ptrs_eval _ A "offs_m" "offs_k" TILE_M TILE_K
      (s.pids 2 * M * K) K _
      (fun i : Fin TILE_M => bmmPidM (s.pids 0) (s.pids 1) (s.numPids 0)
        (s.numPids 1) GROUP_M * TILE_M + i.val)
      (fun j : Fin TILE_K => j.val)
      (bmm_base_eval _ (s.pids 2) M K (by simpa using hpidb))
      (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bmm_ptrs_eval _ B "offs_k" "offs_n" TILE_K TILE_N
      (s.pids 2 * K * N) N _
      (fun i : Fin TILE_K => i.val)
      (fun j : Fin TILE_N => bmmPidN (s.pids 0) (s.pids 1) (s.numPids 0)
        (s.numPids 1) GROUP_M * TILE_N + j.val)
      (bmm_base_eval _ (s.pids 2) K N (by simpa using hpidb))
      (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bmm_ptrs_eval _ O "offs_m" "offs_n" TILE_M TILE_N
      (s.pids 2 * M * N) N _
      (fun i : Fin TILE_M => bmmPidM (s.pids 0) (s.pids 1) (s.numPids 0)
        (s.numPids 1) GROUP_M * TILE_M + i.val)
      (fun j : Fin TILE_N => bmmPidN (s.pids 0) (s.pids 1) (s.numPids 0)
        (s.numPids 1) GROUP_M * TILE_N + j.val)
      (bmm_base_eval _ (s.pids 2) M N (by simpa using hpidb))
      (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (bmm_NI_eval _ K TILE_K))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bmm_zeros_eval [TILE_M, TILE_N] _))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, Nat.zero_le _, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp [hGpids]
  · exact hGmem
  · intro rg o
    simp only [BlockState.setReg_undef]
    rw [hGundef]
    simpa using hundef rg o
  · simp
  · rw [show (⟨fun i => decide (bmmRowG s TILE_M GROUP_M i.1.val < M)⟩
        : Tile .bool [TILE_M])
      = ⟨fun i => decide (bmmPidM (s.pids 0) (s.pids 1) (s.numPids 0)
          (s.numPids 1) GROUP_M * TILE_M + i.1.val < M)⟩
      from Tile.ext fun i => by simp [bmmRowG]]
    simp
  · rw [show (⟨fun j => decide (bmmColG s TILE_N GROUP_M j.1.val < N)⟩
        : Tile .bool [TILE_N])
      = ⟨fun j => decide (bmmPidN (s.pids 0) (s.pids 1) (s.numPids 0)
          (s.numPids 1) GROUP_M * TILE_N + j.1.val < N)⟩
      from Tile.ext fun j => by simp [bmmColG]]
    simp
  · rw [show ((Tile.vec fun r : Fin TILE_K => 0 * TILE_K + r.val)
        : Tile .nat [TILE_K])
      = ((Tile.vec fun r : Fin TILE_K => r.val) : Tile .nat [TILE_K])
      from Tile.ext fun r => by
        show 0 * TILE_K + r.1.val = r.1.val
        omega]
    simp
  · rw [show (⟨fun idx : TileIndex [TILE_M, TILE_K] =>
        ((A : RegionName), s.pids 2 * M * K
          + bmmRowG s TILE_M GROUP_M idx.1.val * K
          + (0 * TILE_K + idx.2.1.val))⟩ : Tile .ptr [TILE_M, TILE_K])
      = ⟨fun idx => ((A : RegionName), s.pids 2 * M * K
          + (bmmPidM (s.pids 0) (s.pids 1) (s.numPids 0) (s.numPids 1) GROUP_M
              * TILE_M + idx.1.val) * K
          + idx.2.1.val)⟩
      from Tile.ext fun idx => by
        simp only [bmmRowG]
        refine congrArg _ ?_
        omega]
    simp
  · rw [show (⟨fun idx : TileIndex [TILE_K, TILE_N] =>
        ((B : RegionName), s.pids 2 * K * N
          + (0 * TILE_K + idx.1.val) * N
          + bmmColG s TILE_N GROUP_M idx.2.1.val)⟩ : Tile .ptr [TILE_K, TILE_N])
      = ⟨fun idx => ((B : RegionName), s.pids 2 * K * N
          + idx.1.val * N
          + (bmmPidN (s.pids 0) (s.pids 1) (s.numPids 0) (s.numPids 1) GROUP_M
              * TILE_N + idx.2.1.val))⟩
      from Tile.ext fun idx => by
        simp only [bmmColG]
        refine congrArg _ ?_
        have : (0 * TILE_K + idx.1.val) * N = idx.1.val * N := by
          rw [Nat.zero_mul, Nat.zero_add]
        omega]
    simp
  · rw [show (⟨fun idx : TileIndex [TILE_M, TILE_N] =>
        ((O : RegionName), bmmOOffset s M N TILE_M TILE_N GROUP_M idx)⟩
          : Tile .ptr [TILE_M, TILE_N])
      = ⟨fun idx => ((O : RegionName), s.pids 2 * M * N
          + (bmmPidM (s.pids 0) (s.pids 1) (s.numPids 0) (s.numPids 1) GROUP_M
              * TILE_M + idx.1.val) * N
          + (bmmPidN (s.pids 0) (s.pids 1) (s.numPids 0) (s.numPids 1) GROUP_M
              * TILE_N + idx.2.1.val))⟩
      from Tile.ext fun idx => by
        simp [bmmOOffset, bmmRowG, bmmColG]]
    simp
  · rw [show (⟨fun idx : TileIndex [TILE_M, TILE_N] =>
        some (bmmAcc s A B M N K TILE_K 0
          (bmmRowG s TILE_M GROUP_M idx.1.val)
          (bmmColG s TILE_N GROUP_M idx.2.1.val))⟩ : Tile .real [TILE_M, TILE_N])
      = ⟨fun _ => some (0 : ℝ)⟩
      from Tile.ext fun idx => by simp [bmmAcc]]
    simp

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **One K-block.** -/
theorem bmmLoop_step (s0 sin : BlockState) (A B O : RegionName)
    (M N K TILE_M TILE_N TILE_K GROUP_M : Nat) (c : Nat)
    (hlt : c < bmmNI K TILE_K)
    (hInv : bmmInv s0 A B O M N K TILE_M TILE_N TILE_K GROUP_M c sin) :
    ∃ s', stepStmts (bmmLoopBody N K TILE_M TILE_N TILE_K)
        (sin.setReg "_i" .nat [] (Tile.scalar c)) = some s'
      ∧ bmmInv s0 A B O M N K TILE_M TILE_N TILE_K GROUP_M (c + 1) s' := by
  obtain ⟨hpids, hmem, hund, hle, hni, hmm, hmn, hok, hap, hbp, hop, ho⟩ := hInv
  unfold bmmLoopBody
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bmm_maskLt_eval _ TILE_K K "offs_k" (fun r => c * TILE_K + r.val)
      (by simpa using hok)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bmm_mask2d_eval _ "mask_m" "mask_k" TILE_M TILE_K
      (fun i => bmmRowG s0 TILE_M GROUP_M i.val < M)
      (fun j => c * TILE_K + j.val < K)
      (by simpa using hmm) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bmm_mask2d_eval _ "mask_k" "mask_n" TILE_K TILE_N
      (fun i => c * TILE_K + i.val < K)
      (fun j => bmmColG s0 TILE_N GROUP_M j.val < N)
      (by simp) (by simpa using hmn)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bmm_maskedLoad_eval _ s0 "a_ptrs" "mask_a" A
      (fun idx : TileIndex [TILE_M, TILE_K] => s0.pids 2 * M * K
        + bmmRowG s0 TILE_M GROUP_M idx.1.val * K + (c * TILE_K + idx.2.1.val))
      (fun idx => bmmRowG s0 TILE_M GROUP_M idx.1.val < M
        ∧ c * TILE_K + idx.2.1.val < K)
      (fun r o => by simpa using hund r o)
      (by simpa using hmem)
      (by simpa using hap)
      (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bmm_maskedLoad_eval _ s0 "b_ptrs" "mask_b" B
      (fun idx : TileIndex [TILE_K, TILE_N] => s0.pids 2 * K * N
        + (c * TILE_K + idx.1.val) * N + bmmColG s0 TILE_N GROUP_M idx.2.1.val)
      (fun idx => c * TILE_K + idx.1.val < K
        ∧ bmmColG s0 TILE_N GROUP_M idx.2.1.val < N)
      (fun r o => by simpa using hund r o)
      (by simpa using hmem)
      (by simpa using hbp)
      (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bmm_okAdd_eval _ TILE_K (c * TILE_K) (by simpa using hok)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bmm_ptrAdv_const_eval _ "a_ptrs"
      (fun idx : TileIndex [TILE_M, TILE_K] => ((A : RegionName),
        s0.pids 2 * M * K + bmmRowG s0 TILE_M GROUP_M idx.1.val * K
          + (c * TILE_K + idx.2.1.val)))
      TILE_K (by simpa using hap)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bmm_ptrAdv_mul_eval _ "b_ptrs"
      (fun idx : TileIndex [TILE_K, TILE_N] => ((B : RegionName),
        s0.pids 2 * K * N + (c * TILE_K + idx.1.val) * N
          + bmmColG s0 TILE_N GROUP_M idx.2.1.val))
      TILE_K N (by simpa using hbp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bmm_oAdd_eval _ TILE_M TILE_K TILE_N
      (fun i j => bmmAcc s0 A B M N K TILE_K c
        (bmmRowG s0 TILE_M GROUP_M i) (bmmColG s0 TILE_N GROUP_M j))
      (fun i t => bmmAGuarded s0 A M K (bmmRowG s0 TILE_M GROUP_M i)
        (c * TILE_K + t))
      (fun t j => bmmBGuarded s0 B K N (c * TILE_K + t)
        (bmmColG s0 TILE_N GROUP_M j))
      (by simpa using ho)
      (by
        rw [show (⟨fun idx : TileIndex [TILE_M, TILE_K] =>
            some (bmmAGuarded s0 A M K (bmmRowG s0 TILE_M GROUP_M idx.1.val)
              (c * TILE_K + idx.2.1.val))⟩ : Tile .real [TILE_M, TILE_K])
          = ⟨fun idx => some (if bmmRowG s0 TILE_M GROUP_M idx.1.val < M
              ∧ c * TILE_K + idx.2.1.val < K
            then s0.readMem A (s0.pids 2 * M * K
              + bmmRowG s0 TILE_M GROUP_M idx.1.val * K
              + (c * TILE_K + idx.2.1.val))
            else 0)⟩
          from Tile.ext fun idx => by
            simp only [bmmAGuarded, bmmAVal]]
        simp)
      (by
        rw [show (⟨fun idx : TileIndex [TILE_K, TILE_N] =>
            some (bmmBGuarded s0 B K N (c * TILE_K + idx.1.val)
              (bmmColG s0 TILE_N GROUP_M idx.2.1.val))⟩ : Tile .real [TILE_K, TILE_N])
          = ⟨fun idx => some (if c * TILE_K + idx.1.val < K
              ∧ bmmColG s0 TILE_N GROUP_M idx.2.1.val < N
            then s0.readMem B (s0.pids 2 * K * N
              + (c * TILE_K + idx.1.val) * N
              + bmmColG s0 TILE_N GROUP_M idx.2.1.val)
            else 0)⟩
          from Tile.ext fun idx => by
            simp only [bmmBGuarded, bmmBVal]]
        simp)))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simpa using hpids
  · simpa using hmem
  · intro rg o
    simpa using hund rg o
  · omega
  · simpa using hni
  · simpa using hmm
  · simpa using hmn
  · rw [show ((Tile.vec fun r : Fin TILE_K => (c + 1) * TILE_K + r.val)
        : Tile .nat [TILE_K])
      = ((Tile.vec fun r : Fin TILE_K => c * TILE_K + TILE_K + r.val)
        : Tile .nat [TILE_K])
      from Tile.ext fun r => by
        show (c + 1) * TILE_K + r.1.val = c * TILE_K + TILE_K + r.1.val
        have : (c + 1) * TILE_K = c * TILE_K + TILE_K := by ring
        omega]
    simp
  · rw [show (⟨fun idx : TileIndex [TILE_M, TILE_K] =>
        ((A : RegionName), s0.pids 2 * M * K
          + bmmRowG s0 TILE_M GROUP_M idx.1.val * K
          + ((c + 1) * TILE_K + idx.2.1.val))⟩ : Tile .ptr [TILE_M, TILE_K])
      = ⟨fun idx => ((A : RegionName), s0.pids 2 * M * K
          + bmmRowG s0 TILE_M GROUP_M idx.1.val * K
          + (c * TILE_K + idx.2.1.val) + TILE_K)⟩
      from Tile.ext fun idx => by
        show ((A : RegionName), _) = ((A : RegionName), _)
        refine congrArg _ ?_
        have : (c + 1) * TILE_K = c * TILE_K + TILE_K := by ring
        omega]
    simp
  · rw [show (⟨fun idx : TileIndex [TILE_K, TILE_N] =>
        ((B : RegionName), s0.pids 2 * K * N
          + ((c + 1) * TILE_K + idx.1.val) * N
          + bmmColG s0 TILE_N GROUP_M idx.2.1.val)⟩ : Tile .ptr [TILE_K, TILE_N])
      = ⟨fun idx => ((B : RegionName), s0.pids 2 * K * N
          + (c * TILE_K + idx.1.val) * N
          + bmmColG s0 TILE_N GROUP_M idx.2.1.val + TILE_K * N)⟩
      from Tile.ext fun idx => by
        show ((B : RegionName), _) = ((B : RegionName), _)
        refine congrArg _ ?_
        have : ((c + 1) * TILE_K + idx.1.val) * N
            = (c * TILE_K + idx.1.val) * N + TILE_K * N := by ring
        omega]
    simp
  · simpa using hop
  · rw [show (⟨fun idx : TileIndex [TILE_M, TILE_N] =>
        some (bmmAcc s0 A B M N K TILE_K (c + 1)
          (bmmRowG s0 TILE_M GROUP_M idx.1.val)
          (bmmColG s0 TILE_N GROUP_M idx.2.1.val))⟩ : Tile .real [TILE_M, TILE_N])
      = ⟨fun idx => some (bmmAcc s0 A B M N K TILE_K c
          (bmmRowG s0 TILE_M GROUP_M idx.1.val)
          (bmmColG s0 TILE_N GROUP_M idx.2.1.val)
        + ∑ t : Fin TILE_K,
            bmmAGuarded s0 A M K (bmmRowG s0 TILE_M GROUP_M idx.1.val)
                (c * TILE_K + t.val)
              * bmmBGuarded s0 B K N (c * TILE_K + t.val)
                (bmmColG s0 TILE_N GROUP_M idx.2.1.val))⟩
      from Tile.ext fun idx =>
        congrArg some (bmmAcc_step s0 A B M N K TILE_K c
          (bmmRowG s0 TILE_M GROUP_M idx.1.val)
          (bmmColG s0 TILE_N GROUP_M idx.2.1.val)).symm]
    simp

set_option maxHeartbeats 4000000 in
/-- **The full K loop.** -/
theorem bmmLoop_run (s0 sPre : BlockState) (A B O : RegionName)
    (M N K TILE_M TILE_N TILE_K GROUP_M : Nat)
    (hInv : bmmInv s0 A B O M N K TILE_M TILE_N TILE_K GROUP_M 0 sPre) :
    ∃ sL, stepStmt (Stmt.forRangeDyn "_i" (Op.constNat 0)
        (Op.ref .nat [] "num_iters") (Op.constNat 1)
        (bmmLoopBody N K TILE_M TILE_N TILE_K)) sPre = some sL
      ∧ bmmInv s0 A B O M N K TILE_M TILE_N TILE_K GROUP_M
          (bmmNI K TILE_K) sL := by
  have hni := hInv.2.2.2.2.1
  obtain ⟨final, sL, hrun, hge, hP⟩ :=
    forRangeDyn_inv (idx := "_i")
      (P := fun c s => bmmInv s0 A B O M N K TILE_M TILE_N TILE_K GROUP_M c s)
      (evalOp_constNat 0 sPre)
      (show evalOp (Op.ref .nat [] "num_iters") sPre
        = some (Tile.scalar (bmmNI K TILE_K)) from by rw [evalOp_ref, hni])
      (evalOp_constNat 1 sPre)
      one_ne_zero
      hInv
      (fun c s hc hPc =>
        bmmLoop_step s0 s A B O M N K TILE_M TILE_N TILE_K GROUP_M c hc hPc)
  have hfin : final = bmmNI K TILE_K := le_antisymm hP.2.2.2.1 hge
  exact ⟨sL, hrun, hfin ▸ hP⟩

/-! ## The masked store and ★ main theorem -/

private theorem bmm_block_index_inj {Q j c A' B' : Nat} (hA : A' < Q) (hB : B' < Q)
    (heq : j * Q + A' = c * Q + B') : j = c := by
  have hQ : 0 < Q := Nat.lt_of_le_of_lt (Nat.zero_le A') hA
  have hj : (j * Q + A') / Q = j := by
    rw [Nat.mul_comm, Nat.mul_add_div hQ, Nat.div_eq_of_lt hA, Nat.add_zero]
  have hc : (c * Q + B') / Q = c := by
    rw [Nat.mul_comm, Nat.mul_add_div hQ, Nat.div_eq_of_lt hB, Nat.add_zero]
  rw [← hj, heq, hc]

/-- The `o` store lanes are pairwise distinct once the tile column fits under
the row stride (`TILE_N ≤ N`). -/
theorem bmmOOffset_injective (s : BlockState)
    (M N TILE_M TILE_N GROUP_M : Nat) (hTN : TILE_N ≤ N) :
    Function.Injective (bmmOOffset s M N TILE_M TILE_N GROUP_M) := by
  rintro ⟨i, j, u⟩ ⟨i', j', u'⟩ heq
  simp only [bmmOOffset, bmmRowG, bmmColG] at heq
  have hj := j.isLt
  have hj' := j'.isLt
  set PM := bmmPidM (s.pids 0) (s.pids 1) (s.numPids 0) (s.numPids 1) GROUP_M
  set PN := bmmPidN (s.pids 0) (s.pids 1) (s.numPids 0) (s.numPids 1) GROUP_M
  have h2 : (PM * TILE_M + i.val) * N + (PN * TILE_N + j.val)
      = (PM * TILE_M + i'.val) * N + (PN * TILE_N + j'.val) := by omega
  have h3 : (PM * TILE_M + i.val) * N + (PN * TILE_N + j.val) <
      ((PM * TILE_M + i.val) + 1) * N + (PN * TILE_N) := by
    have : (PM * TILE_M + i.val + 1) * N
        = (PM * TILE_M + i.val) * N + N := by ring
    omega
  have hrow : PM * TILE_M + i.val = PM * TILE_M + i'.val := by
    by_contra hne
    rcases Nat.lt_or_ge (PM * TILE_M + i.val) (PM * TILE_M + i'.val) with hlt | hge
    · have : ((PM * TILE_M + i.val) + 1) * N ≤ (PM * TILE_M + i'.val) * N :=
        Nat.mul_le_mul_right N hlt
      omega
    · have hlt' : PM * TILE_M + i'.val < PM * TILE_M + i.val := by omega
      have : ((PM * TILE_M + i'.val) + 1) * N ≤ (PM * TILE_M + i.val) * N :=
        Nat.mul_le_mul_right N hlt'
      have hstep : (PM * TILE_M + i'.val + 1) * N
          = (PM * TILE_M + i'.val) * N + N := by ring
      omega
  have hi : i = i' := Fin.ext (by omega)
  have hjv : j = j' := Fin.ext (by
    have : (PM * TILE_M + i.val) * N = (PM * TILE_M + i'.val) * N := by
      rw [hrow]
    omega)
  subst hi
  subst hjv
  rfl

/-- The post-store state: every active lane written. -/
noncomputable def bmmStoreState (sin : BlockState) (rg : RegionName)
    (s0 : BlockState) (M N TILE_M TILE_N GROUP_M : Nat)
    (f : TileIndex [TILE_M, TILE_N] → ℝ) : BlockState :=
  (TileShape.allIndices [TILE_M, TILE_N]).foldl
    (fun acc i => if bmmOActive s0 M N TILE_M TILE_N GROUP_M i
        then acc.writeMem rg (bmmOOffset s0 M N TILE_M TILE_N GROUP_M i) (f i)
        else acc) sin

set_option maxHeartbeats 4000000 in
/-- **Masked store step (eq).** -/
theorem bmmStore_step_eq (sin s0 : BlockState) (rg : RegionName)
    (M N TILE_M TILE_N GROUP_M : Nat)
    (f : TileIndex [TILE_M, TILE_N] → ℝ)
    (ho : sin.regs .real [TILE_M, TILE_N] "o"
      = some ⟨fun idx => some (f idx)⟩)
    (hp : sin.regs .ptr [TILE_M, TILE_N] "o_ptrs"
      = some ⟨fun idx => (rg, bmmOOffset s0 M N TILE_M TILE_N GROUP_M idx)⟩)
    (hm : sin.regs .bool [TILE_M, TILE_N] "mask_c"
      = some ⟨fun idx => decide (bmmOActive s0 M N TILE_M TILE_N GROUP_M idx)⟩) :
    stepStmt (Stmt.store .real [TILE_M, TILE_N]
        (MemAccess.ptr (Op.ref .ptr [TILE_M, TILE_N] "o_ptrs"))
        (Op.ref .real [TILE_M, TILE_N] "o")
        (MaskOpt.mask (Op.ref .bool [TILE_M, TILE_N] "mask_c"))) sin
      = some (bmmStoreState sin rg s0 M N TILE_M TILE_N GROUP_M f) := by
  unfold stepStmt bmmStoreState
  simp only [evalOp_ref, ho, hp, hm]
  refine congrArg some
    (congrArg (fun g => List.foldl g sin (TileShape.allIndices [TILE_M, TILE_N])) ?_)
  funext acc i
  by_cases hact : bmmOActive s0 M N TILE_M TILE_N GROUP_M i
  · simp only [hact, decide_true, if_true, BlockState.writeMemTyped_real]
    rfl
  · simp only [hact, decide_false, Bool.false_eq_true, if_false]

set_option maxHeartbeats 4000000 in
/-- **Masked store readback.** -/
theorem bmmStore_props (sin s0 : BlockState) (rg : RegionName)
    (M N TILE_M TILE_N GROUP_M : Nat)
    (f : TileIndex [TILE_M, TILE_N] → ℝ)
    (hInj : Function.Injective (bmmOOffset s0 M N TILE_M TILE_N GROUP_M)) :
    (∀ idx : TileIndex [TILE_M, TILE_N],
        bmmOActive s0 M N TILE_M TILE_N GROUP_M idx →
        (bmmStoreState sin rg s0 M N TILE_M TILE_N GROUP_M f).readMem rg
            (bmmOOffset s0 M N TILE_M TILE_N GROUP_M idx)
          = f idx) := by
  classical
  intro idx hidx
  unfold bmmStoreState
  have h := BlockState.scatter_readback_prop_masked_nd (region := rg) sin
    (bmmOOffset s0 M N TILE_M TILE_N GROUP_M) f
    (fun i => bmmOActive s0 M N TILE_M TILE_N GROUP_M i)
    hInj idx
  rw [h, if_pos hidx]

set_option maxHeartbeats 8000000 in
set_option maxRecDepth 8000 in
/-- **The epilogue**: `mask_c` and the masked store, from the loop-exit
invariant, with the readback collapsed to the raw GEMM closed form. -/
theorem bmmStores_run (s0 sL : BlockState) (A B O : RegionName)
    (M N K TILE_M TILE_N TILE_K GROUP_M : Nat)
    (hTN : TILE_N ≤ N) (hTK : 0 < TILE_K)
    (hInv : bmmInv s0 A B O M N K TILE_M TILE_N TILE_K GROUP_M
      (bmmNI K TILE_K) sL) :
    ∃ sF, stepStmts
        [ Stmt.assign .bool [TILE_M, TILE_N] "mask_c"
            (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .bool [TILE_M] "mask_m"))
              (Op.expandDim ⟨0, by simp⟩ (Op.ref .bool [TILE_N] "mask_n"))),
          Stmt.store .real [TILE_M, TILE_N]
            (MemAccess.ptr (Op.ref .ptr [TILE_M, TILE_N] "o_ptrs"))
            (Op.ref .real [TILE_M, TILE_N] "o")
            (MaskOpt.mask (Op.ref .bool [TILE_M, TILE_N] "mask_c")) ] sL = some sF
      ∧ (∀ idx : TileIndex [TILE_M, TILE_N],
          bmmOActive s0 M N TILE_M TILE_N GROUP_M idx →
          sF.readMem O (bmmOOffset s0 M N TILE_M TILE_N GROUP_M idx)
            = bmmOOut s0 A B M N K
                (bmmRowG s0 TILE_M GROUP_M idx.1.val)
                (bmmColG s0 TILE_N GROUP_M idx.2.1.val)) := by
  obtain ⟨hpids, hmem, hund, hle, hni, hmm, hmn, hok, hap, hbp, hop, ho⟩ := hInv
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bmm_mask2d_eval _ "mask_m" "mask_n" TILE_M TILE_N
      (fun i => bmmRowG s0 TILE_M GROUP_M i.val < M)
      (fun j => bmmColG s0 TILE_N GROUP_M j.val < N)
      (by simpa using hmm) (by simpa using hmn)))]
  rw [stepStmts.cons_some (bmmStore_step_eq _ s0 O M N TILE_M TILE_N GROUP_M
    (fun idx => bmmAcc s0 A B M N K TILE_K (bmmNI K TILE_K)
      (bmmRowG s0 TILE_M GROUP_M idx.1.val)
      (bmmColG s0 TILE_N GROUP_M idx.2.1.val))
    (by simpa using ho)
    (by simpa using hop)
    (by
      rw [show (⟨fun idx : TileIndex [TILE_M, TILE_N] =>
          decide (bmmOActive s0 M N TILE_M TILE_N GROUP_M idx)⟩
            : Tile .bool [TILE_M, TILE_N])
        = ⟨fun idx => decide (bmmRowG s0 TILE_M GROUP_M idx.1.val < M
            ∧ bmmColG s0 TILE_N GROUP_M idx.2.1.val < N)⟩
        from Tile.ext fun idx => by simp [bmmOActive]]
      simp))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  intro idx hact
  rw [bmmStore_props _ s0 O M N TILE_M TILE_N GROUP_M _
    (bmmOOffset_injective s0 M N TILE_M TILE_N GROUP_M hTN) idx hact]
  exact bmmAcc_final s0 A B M N K TILE_K
    (bmmRowG s0 TILE_M GROUP_M idx.1.val)
    (bmmColG s0 TILE_N GROUP_M idx.2.1.val) hTK hact.1 hact.2

/-! ## ★ Main theorem -/

set_option maxHeartbeats 8000000 in
set_option maxRecDepth 8000 in
/-- **★ Main theorem: the `o` store is the genuine batched GEMM closed
form.**

For every program `(pidx, pidy, pid_b)` on any launch grid (both quantified
through `s.pids` / `s.numPids`), executing the full surface succeeds and the
masked `o` store holds `Σ_{t < K} A[pid_b, m, t] · B[pid_b, t, n]` at every
in-window lane, where `(m, n)` are the global row/column of the
CTA-reordered tile (`bmmPidM`/`bmmPidN` — the `GROUP_M = 1` identity map or
the grouped swizzle with its runtime `GROUP_SIZE` boundary gate, both arms
proven).

Side conditions: `TILE_N ≤ N` (store-lane injectivity under the row-major
stride), `0 < TILE_K` (the kernel's own `tl.cdiv` trip count), and the
clean-input `hundef` (the masked loads carry no `other`, so masked-off
lanes read the `undef` channel — the `bmm_chunk_fwd` convention). **No**
divisibility hypotheses on `M`, `N`, or `K`: this is the fully-masked
`DIVISIBLE_* = False` arm, exact on arbitrary ragged shapes. -/
specification bmm_o_exec_genuine
    (s : BlockState) (A B O : RegionName)
    (M N K TILE_M TILE_N TILE_K GROUP_M : Nat)
    (hTN : TILE_N ≤ N) (hTK : 0 < TILE_K)
    (hundef : ∀ rg off, s.undef rg off = 0) :
    ∃ sF, exec (bmm_surface A B O M N K TILE_M TILE_N TILE_K
        GROUP_M).toAlgKernel s = some sF
      ∧ (∀ idx : TileIndex [TILE_M, TILE_N],
          bmmOActive s M N TILE_M TILE_N GROUP_M idx →
          sF.readMem O (bmmOOffset s M N TILE_M TILE_N GROUP_M idx)
            = bmmOOut s A B M N K
                (bmmRowG s TILE_M GROUP_M idx.1.val)
                (bmmColG s TILE_N GROUP_M idx.2.1.val)) := by
  obtain ⟨sP, hPro, hInvP⟩ := bmmPrologue_run s A B O M N K TILE_M TILE_N
    TILE_K GROUP_M hundef
  obtain ⟨sL, hLoop, hInvL⟩ := bmmLoop_run s sP A B O M N K TILE_M TILE_N
    TILE_K GROUP_M hInvP
  rw [exec, bmm_body_eq]
  rw [show ([ Stmt.assign .nat [] "pid_b" (Op.programId 2),
      Stmt.assign .nat [] "pidx" (Op.programId 0),
      Stmt.assign .nat [] "pidy" (Op.programId 1),
      Stmt.ifThenElse
        (Op.eq ComparableDType.nat Broadcast.nil (Op.constNat GROUP_M)
          (Op.constNat 1))
        bmmGate1Body (bmmGateGrpBody GROUP_M),
      Stmt.assign .nat [TILE_M] "offs_m"
        (Op.add .nat Broadcast.scalarL
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m")
            (Op.constNat TILE_M))
          (Op.arange TILE_M)),
      Stmt.assign .nat [TILE_N] "offs_n"
        (Op.add .nat Broadcast.scalarL
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_n")
            (Op.constNat TILE_N))
          (Op.arange TILE_N)),
      Stmt.assign .nat [TILE_K] "offs_k" (Op.arange TILE_K),
      Stmt.assign .bool [TILE_M] "mask_m"
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.ref .nat [TILE_M] "offs_m") (Op.constNat M)),
      Stmt.assign .bool [TILE_N] "mask_n"
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.ref .nat [TILE_N] "offs_n") (Op.constNat N)),
      Stmt.assign .ptr [TILE_M, TILE_K] "a_ptrs"
        (Op.ptrAdd Broadcast.scalarL (Op.ptrBase A)
          (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
            (Op.add .nat Broadcast.scalarL
              (Op.mul .nat Broadcast.nil
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_b")
                  (Op.constNat M))
                (Op.constNat K))
              (Op.mul .nat Broadcast.scalarR
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [TILE_M] "offs_m"))
                (Op.constNat K)))
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [TILE_K] "offs_k")))),
      Stmt.assign .ptr [TILE_K, TILE_N] "b_ptrs"
        (Op.ptrAdd Broadcast.scalarL (Op.ptrBase B)
          (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
            (Op.add .nat Broadcast.scalarL
              (Op.mul .nat Broadcast.nil
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_b")
                  (Op.constNat K))
                (Op.constNat N))
              (Op.mul .nat Broadcast.scalarR
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [TILE_K] "offs_k"))
                (Op.constNat N)))
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [TILE_N] "offs_n")))),
      Stmt.assign .ptr [TILE_M, TILE_N] "o_ptrs"
        (Op.ptrAdd Broadcast.scalarL (Op.ptrBase O)
          (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
            (Op.add .nat Broadcast.scalarL
              (Op.mul .nat Broadcast.nil
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_b")
                  (Op.constNat M))
                (Op.constNat N))
              (Op.mul .nat Broadcast.scalarR
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [TILE_M] "offs_m"))
                (Op.constNat N)))
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [TILE_N] "offs_n")))),
      Stmt.assign .nat [] "num_iters"
        (Op.div .nat Broadcast.nil
          (Op.sub .nat Broadcast.nil
            (Op.add .nat Broadcast.nil (Op.constNat K) (Op.constNat TILE_K))
            (Op.constNat 1))
          (Op.constNat TILE_K)),
      Stmt.assign .real [TILE_M, TILE_N] "o"
        (Op.full [TILE_M, TILE_N] (Op.const 0)),
      Stmt.forRangeDyn "_i" (Op.constNat 0) (Op.ref .nat [] "num_iters")
        (Op.constNat 1) (bmmLoopBody N K TILE_M TILE_N TILE_K),
      Stmt.assign .bool [TILE_M, TILE_N] "mask_c"
        (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .bool [TILE_M] "mask_m"))
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .bool [TILE_N] "mask_n"))),
      Stmt.store .real [TILE_M, TILE_N]
        (MemAccess.ptr (Op.ref .ptr [TILE_M, TILE_N] "o_ptrs"))
        (Op.ref .real [TILE_M, TILE_N] "o")
        (MaskOpt.mask (Op.ref .bool [TILE_M, TILE_N] "mask_c")) ] : List Stmt)
      = [ Stmt.assign .nat [] "pid_b" (Op.programId 2),
          Stmt.assign .nat [] "pidx" (Op.programId 0),
          Stmt.assign .nat [] "pidy" (Op.programId 1),
          Stmt.ifThenElse
            (Op.eq ComparableDType.nat Broadcast.nil (Op.constNat GROUP_M)
              (Op.constNat 1))
            bmmGate1Body (bmmGateGrpBody GROUP_M),
          Stmt.assign .nat [TILE_M] "offs_m"
            (Op.add .nat Broadcast.scalarL
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m")
                (Op.constNat TILE_M))
              (Op.arange TILE_M)),
          Stmt.assign .nat [TILE_N] "offs_n"
            (Op.add .nat Broadcast.scalarL
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_n")
                (Op.constNat TILE_N))
              (Op.arange TILE_N)),
          Stmt.assign .nat [TILE_K] "offs_k" (Op.arange TILE_K),
          Stmt.assign .bool [TILE_M] "mask_m"
            (Op.lt ComparableDType.nat Broadcast.scalarR
              (Op.ref .nat [TILE_M] "offs_m") (Op.constNat M)),
          Stmt.assign .bool [TILE_N] "mask_n"
            (Op.lt ComparableDType.nat Broadcast.scalarR
              (Op.ref .nat [TILE_N] "offs_n") (Op.constNat N)),
          Stmt.assign .ptr [TILE_M, TILE_K] "a_ptrs"
            (Op.ptrAdd Broadcast.scalarL (Op.ptrBase A)
              (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
                (Op.add .nat Broadcast.scalarL
                  (Op.mul .nat Broadcast.nil
                    (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_b")
                      (Op.constNat M))
                    (Op.constNat K))
                  (Op.mul .nat Broadcast.scalarR
                    (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [TILE_M] "offs_m"))
                    (Op.constNat K)))
                (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [TILE_K] "offs_k")))),
          Stmt.assign .ptr [TILE_K, TILE_N] "b_ptrs"
            (Op.ptrAdd Broadcast.scalarL (Op.ptrBase B)
              (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
                (Op.add .nat Broadcast.scalarL
                  (Op.mul .nat Broadcast.nil
                    (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_b")
                      (Op.constNat K))
                    (Op.constNat N))
                  (Op.mul .nat Broadcast.scalarR
                    (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [TILE_K] "offs_k"))
                    (Op.constNat N)))
                (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [TILE_N] "offs_n")))),
          Stmt.assign .ptr [TILE_M, TILE_N] "o_ptrs"
            (Op.ptrAdd Broadcast.scalarL (Op.ptrBase O)
              (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
                (Op.add .nat Broadcast.scalarL
                  (Op.mul .nat Broadcast.nil
                    (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_b")
                      (Op.constNat M))
                    (Op.constNat N))
                  (Op.mul .nat Broadcast.scalarR
                    (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [TILE_M] "offs_m"))
                    (Op.constNat N)))
                (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [TILE_N] "offs_n")))),
          Stmt.assign .nat [] "num_iters"
            (Op.div .nat Broadcast.nil
              (Op.sub .nat Broadcast.nil
                (Op.add .nat Broadcast.nil (Op.constNat K) (Op.constNat TILE_K))
                (Op.constNat 1))
              (Op.constNat TILE_K)),
          Stmt.assign .real [TILE_M, TILE_N] "o"
            (Op.full [TILE_M, TILE_N] (Op.const 0)) ]
        ++ (Stmt.forRangeDyn "_i" (Op.constNat 0) (Op.ref .nat [] "num_iters")
            (Op.constNat 1) (bmmLoopBody N K TILE_M TILE_N TILE_K)
          :: [ Stmt.assign .bool [TILE_M, TILE_N] "mask_c"
                (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
                  (Op.expandDim ⟨1, by simp⟩ (Op.ref .bool [TILE_M] "mask_m"))
                  (Op.expandDim ⟨0, by simp⟩ (Op.ref .bool [TILE_N] "mask_n"))),
              Stmt.store .real [TILE_M, TILE_N]
                (MemAccess.ptr (Op.ref .ptr [TILE_M, TILE_N] "o_ptrs"))
                (Op.ref .real [TILE_M, TILE_N] "o")
                (MaskOpt.mask (Op.ref .bool [TILE_M, TILE_N] "mask_c")) ])
      from rfl]
  rw [stepStmts.append_some hPro, stepStmts.cons_some hLoop]
  obtain ⟨sF, hSt, hRead⟩ := bmmStores_run s sL A B O M N K TILE_M TILE_N
    TILE_K GROUP_M hTN hTK hInvL
  exact ⟨sF, hSt, hRead⟩

end Correct_without_Rounding

end VeriTile.Bench.TritonBenchG.BmmOptimized
