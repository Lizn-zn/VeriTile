import VeriTile.Triton
import VeriTile.Triton.Memory.Flatten
import VeriTile.Examples.Common

/-!
# `mul_exponent_compensator` — strict per-kernel correctness

`mul_kernel` scales a vector by a constant: program `pid` loads block
`[pid·BLOCK_SIZE, (pid+1)·BLOCK_SIZE)` of `src` (no mask), multiplies each lane
by the compile-time constant `exponent_compensator = 2.0 ** (127 - 15)`, and
stores the result to `dst`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`mul_kernel[(shape[0] // BLOCK_SIZE,)](...)`, the grid
size, and how the runtime composes per-program writes into one buffer) is the
*trusted boundary*, not a proof obligation here. Because the headline
quantifies over every program id whose windows are in bounds, the per-program
statement covers every program of the grid. Note the kernel uses an unmasked
load/store, so the host is trusted to supply a buffer that is an exact
multiple of `BLOCK_SIZE`.

## Proof architecture

```
mul_kernel_correctness                    ← THE SPEC (⊨ headline, KernelIO₁)
  ├─ mul_kernel_flattenOk                 bridge fragment membership
  ├─ mul_kernel_traceSafe                 per-execution safety walk
  └─ mul_kernel_region_run                region-model Hoare triple
       ├─ mul_kernel_correct              value: per-lane readback
       └─ mul_kernel_frame                frame: cells off the output window
```

The spec is plain elementwise `xs i * exponentCompensator` — no
optimizer/reduction oracle applies. `mulIO` is the kernel's IO signature
(1-in/1-out, `KernelIO₁`): program `pid` reads its `BLOCK_SIZE`-tile at
`pid * BLOCK_SIZE` of `src` and writes the same window of `dst`.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float). The Python compile-time literal `2.0 ** (127 - 15)` is modeled by the
noncomputable real constant `exponentCompensator` and injected as a real scalar
antiquote — its bit pattern (a float32 exponent-bias compensator) is not
otherwise interpreted. No output/input disjointness is assumed by the
region-model triple: the input is read into registers before the scatter, so
the result is correct even if `dst` aliases `src` (the flat-memory headline
additionally quantifies over disjoint placements of the two buffers).
-/

namespace VeriTile.Bench.TritonBenchG.MulExponentCompensator

open VeriTile.Triton
open scoped VeriTile.Triton.KernelIO₁
open VeriTile.Examples

/-- The constexpr multiplier from `mul_exponent_compensator.py`. -/
noncomputable def exponentCompensator : ℝ :=
  (2 : ℝ) ^ (127 - 15)

/-- Faithful 1:1 transcription of `mul_exponent_compensator.py`'s
`mul_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python local constexpr literal `2.0 ** (127 - 15)` is represented by the
  Lean constant `exponentCompensator` and injected as a real scalar antiquote.
- Python `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameter. -/
noncomputable def mul_kernel
    (src dst : RegionName)
    (BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  exponent_compensator = $((exponentCompensator : ℝ))
  idxs = tl.program_id(0) * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  x = tl.load(src + idxs)
  y = x * exponent_compensator
  tl.store(dst + idxs, y)
}

/-! ## Region-model Hoare triple

The mathematical core, proved against the region model (named buffers, no
pointer arithmetic): the value lemma `mul_kernel_correct`, a frame lemma
(`mul_kernel_frame`, via the scatter-store `foldl`), and their package
`mul_kernel_region_run` — exactly the `hrun` obligation of
`KernelIO₁.Implements.intro`. -/

/-- Algorithm-layer correctness for `mul_kernel`: from any state with the
input tile loaded, every output lane holds `xs i * exponentCompensator`. -/
theorem mul_kernel_correct
    (src dst : RegionName)
    (BLOCK_SIZE : Nat) (_hBlockSize : 0 < BLOCK_SIZE)
    (s : BlockState) (xs : Fin BLOCK_SIZE → ℝ)
    (h_x : InputLoadedAt s src BLOCK_SIZE xs) :
    ∀ i : Fin BLOCK_SIZE,
      observeAt (exec (mul_kernel src dst BLOCK_SIZE) s)
          dst BLOCK_SIZE s.pid i
        = some (xs i * exponentCompensator) := by
  intro i
  have h_inj := injective_offset_singleton (n := BLOCK_SIZE) (s.pid * BLOCK_SIZE)
  simp [observeAt, exec, mul_kernel, stepStmts, stepStmt, evalOp.eq_def,
        Tile.bop, NumericDType.add, NumericDType.mul,
        exponentCompensator]
  unfold InputLoadedAt at h_x
  rw [BlockState.scatter_readback_nd _ _ _ h_inj (i, PUnit.unit)]
  simp [h_x]

/-- A scatter-store `foldl` leaves every memory cell it does not hit
unchanged (cell-level frame for the unmasked store). -/
private theorem foldl_store_preserve_cell {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → ℝ)
    (r : RegionName) (o : Nat) (l : List α) (s : BlockState)
    (hnot : ∀ k ∈ l, ¬(region = r ∧ offsetFn k = o)) :
    (l.foldl (fun acc k =>
        acc.writeMem region (offsetFn k) (valueFn k)) s).mem r o
      = s.mem r o := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons,
        ih _ (fun k hk => hnot k (List.mem_cons_of_mem hd hk)),
        BlockState.writeMem_mem]
      exact if_neg (fun hc =>
        hnot hd List.mem_cons_self ⟨hc.1.symm, hc.2.symm⟩)

/-- Frame half: every memory cell other than the output window is preserved
by the run. -/
private theorem mul_kernel_frame (src dst : RegionName)
    (B : Nat) (s s1 : BlockState)
    (hExec : exec ((mul_kernel src dst B).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ i : Fin B, ¬(dst = r ∧ s.pid * B + i.val = o)) :
    s1.mem r o = s.mem r o := by
  simp [exec, mul_kernel, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
    evalOp.eq_def, Tile.bop, NumericDType.add, NumericDType.mul]
    at hExec
  subst hExec
  refine Eq.trans (foldl_store_preserve_cell _ _ r o _ _ ?_) rfl
  intro k _ hc
  exact hmiss k.1 hc

/-- **The region-model Hoare triple** — termination, output-window values,
and frame, from any launch state whose input window is loaded. This is what
the `⊨` headline transports to flat memory. -/
theorem mul_kernel_region_run (src dst : RegionName) (B : Nat) (hB : 0 < B)
    (s₀ : BlockState) (xs : Fin B → ℝ)
    (hx : ∀ j : Fin B, s₀.readMem src (s₀.pid * B + j.val) = xs j) :
    ∃ s1, exec ((mul_kernel src dst B).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin B,
          s1.readMem dst (s₀.pid * B + j.val)
            = xs j * exponentCompensator)
      ∧ (∀ r o,
          (r ≠ dst ∨ ∀ j : Fin B, o ≠ s₀.pid * B + j.val) →
          s1.mem r o = s₀.mem r o) := by
  have hobs := mul_kernel_correct src dst B hB s₀ xs hx
  rw [show exec (mul_kernel src dst B) s₀
      = exec ((mul_kernel src dst B).toAlgKernel) s₀ from rfl]
    at hobs
  cases hsrc : exec ((mul_kernel src dst B).toAlgKernel) s₀ with
  | none =>
      have := hobs ⟨0, hB⟩
      rw [hsrc] at this
      simp [observeAt] at this
  | some s1 =>
      refine ⟨s1, rfl, fun j => ?_, fun r o hcond => ?_⟩
      · have := hobs j
        rw [hsrc] at this
        simpa [observeAt] using this
      · refine mul_kernel_frame src dst B s₀ s1 hsrc r o
          (fun i ⟨hr, ho⟩ => ?_)
        rcases hcond with hne | hno
        · exact hne hr.symm
        · exact hno i ho.symm

/-! ## Flat-memory bridge side conditions

`mul_kernel` is register-indirect (`idxs = ...; tl.load(src + idxs)`), so no
∀-state contract covers it; the flat-memory bridge (v1.2) takes the
per-execution `Kernel.TraceSafe` contract instead, discharged below by
walking the actual five-statement execution. Because the kernel is
**unmasked**, `MaskOpt.none.Active` makes every lane active, and the bounds
obligation is the aligned contract `pid * B + B ≤ bounds reg` — the whole
program tile in bounds — for each of the two regions. -/

/-- Inversion for a successful `assign` step. -/
private theorem stepStmt_assign_inv {d : TileDType} {sh : TileShape}
    {nm : RegName} {e : Op d sh} {s s' : BlockState}
    (h : stepStmt (.assign d sh nm e) s = some s') :
    ∃ v, evalOp e s = some v ∧ s' = s.setReg nm d sh v := by
  simp only [stepStmt] at h
  cases hv : evalOp e s with
  | none => rw [hv] at h; exact absurd h (by simp)
  | some v =>
      rw [hv] at h
      replace h : some (s.setReg nm d sh v) = some s' := h
      exact ⟨v, rfl, (Option.some_inj.mp h).symm⟩

set_option maxHeartbeats 1600000 in
/-- The kernel is trace-safe: each of its five statements is safe in the
state the execution actually reaches. The unmasked load/store touch all `B`
lanes of the program tile, so each region's bound must cover the whole
tile: `s.pid * B + B ≤ bounds reg`. -/
theorem mul_kernel_traceSafe (src dst : RegionName)
    (B : Nat) (bounds : RegionBounds) (s : BlockState)
    (hsrcB : s.pid * B + B ≤ bounds src)
    (hdstB : s.pid * B + B ≤ bounds dst) :
    Kernel.TraceSafe bounds
      ((mul_kernel src dst B).toAlgKernel)
      s := by
  unfold Kernel.TraceSafe
  -- statement 1: exponent_compensator = <real constant>
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s1 hs1
  obtain ⟨v1, hv1, rfl⟩ := stepStmt_assign_inv hs1
  -- statement 2: idxs = program_id(0) * B + arange B
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s2 hs2
  obtain ⟨v2, hv2, rfl⟩ := stepStmt_assign_inv hs2
  rw [show evalOp (Op.add .nat .scalarL
      (Op.mul .nat .nil (Op.programId 0) (Op.constNat B))
      (Op.arange B))
      (s.setReg "exponent_compensator" .real [] v1)
      = some ⟨fun i => s.pids 0 * B + i.1.val⟩ from by
    simp [Tile.bop, NumericDType.nat_add, NumericDType.nat_mul,
      Tile.vec]] at hv2
  obtain rfl := Option.some_inj.mp hv2
  -- shared register-readback fact at the state before the load
  set sm := (s.setReg "exponent_compensator" .real [] v1).setReg "idxs"
      .nat [B] ⟨fun i => s.pids 0 * B + i.1.val⟩
      with hsm
  have hAAS : ∀ (t : BlockState),
      (∀ dt sh nm, nm = "idxs" → t.regs dt sh nm = sm.regs dt sh nm) →
      ∀ (reg : RegionName), s.pid * B + B ≤ bounds reg →
      MemAccess.ActiveAddressSafe bounds
        (MemAccess.region reg (Op.ref .nat [B] "idxs")) t
        ((MaskOpt.none (dtype := .real)).Active t) := by
    intro t hframe reg hreg
    simp only [MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe]
    intro offsets hoffs i _
    rw [show evalOp (Op.ref .nat [B] "idxs") t
        = some ⟨fun i => s.pids 0 * B + i.1.val⟩ from by
      simp [hframe .nat [B] "idxs" rfl, hsm, BlockState.setReg]] at hoffs
    obtain rfl := Option.some_inj.mp hoffs
    simp only [Region.cast_self]
    exact lt_of_lt_of_le (Nat.add_lt_add_left i.1.isLt _) hreg
  -- statement 3: x = load(src + idxs)   (unmasked: all lanes active)
  refine Stmt.TraceSafeList.cons_intro ?_ ?_
  · simp only [Stmt.TraceSafe, Op.SafeAt]
    exact ⟨by simp, trivial, hAAS sm (fun _ _ _ _ => rfl) src hsrcB⟩
  intro s4 hs4
  obtain ⟨v4, hv4, rfl⟩ := stepStmt_assign_inv hs4
  -- statement 4: y = x * exponent_compensator   (register op)
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s5 hs5
  obtain ⟨v5, hv5, rfl⟩ := stepStmt_assign_inv hs5
  -- statement 5: store(dst + idxs, y)   (unmasked)
  refine Stmt.TraceSafeList.cons_intro ?_ (fun _ _ => .nil_intro)
  simp only [Stmt.TraceSafe, MemAccess.SafeAt, MaskOpt.SafeAt]
  refine ⟨by simp [Op.SafeAt], by simp [Op.SafeAt], trivial,
    hAAS _ (fun dt sh nm hnm => ?_) dst hdstB⟩
  subst hnm; simp [BlockState.setReg]

/-- The kernel sits inside the bridge's covered fragment. -/
theorem mul_kernel_flattenOk (src dst : RegionName) (B : Nat) :
    ((mul_kernel src dst B).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [mul_kernel, ComputeKernel.toAlgKernel,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]

/-! ## The spec: `mulIO ⊨` the constant scaling -/

/-- `mul_kernel`'s **IO signature** — the whole kernel-specific audit
surface of the headline: `src` in, `dst` out; tile lengths
`Bin = Bout = B`; program `pid` reads its tile at `pid * B` of `src` and
writes the same window of `dst`. The windows are declared, not parsed from
the kernel: they formalize the host-side launch convention
(`idxs = pid * BLOCK_SIZE + arange`), and the headline **proves** the
kernel's actual addressing matches them. Buffer sizes are not signature
content: the headline quantifies over every allocation large enough for the
windows. -/
noncomputable def mulIO (src dst : RegionName) (B : Nat) : KernelIO₁ where
  kernel := mul_kernel src dst B
  inp := src
  out := dst
  Bin := B
  Bout := B
  read := fun pid => pid * B
  write := fun pid => pid * B

/-- **The headline**: `mul_kernel` implements the elementwise constant
scaling `xs i * exponentCompensator` on its IO signature. Spelled out (see
`KernelIO₁.Implements`): for every disjoint flat placement of the two
buffers, every program id whose windows are in bounds, and every launch
state whose input window holds `xs` — everything else arbitrary — the
translated pointer kernel terminates, its output window holds
`xs i * exponentCompensator`, and every other memory cell is unchanged.
Proof: `Implements.intro` assembles the region-model triple with the bridge
side conditions. -/
specification mul_kernel_correctness (src dst : RegionName) (B : Nat)
    (hB : 0 < B) :
    mulIO src dst B ⊨ fun xs i => xs i * exponentCompensator := by
  refine KernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact mul_kernel_flattenOk src dst B
  · intro bounds s h1 h2 _
    exact mul_kernel_traceSafe src dst B bounds s h1 h2
  · intro s₀ xs hx
    obtain ⟨s1, hexec, hval, hframe⟩ := mul_kernel_region_run src dst B hB s₀ xs hx
    -- scratch is empty, so its frame side condition is vacuous
    exact ⟨s1, hexec, hval, fun r o hout _ => hframe r o hout⟩

end VeriTile.Bench.TritonBenchG.MulExponentCompensator
