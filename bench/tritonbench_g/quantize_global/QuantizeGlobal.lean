import VeriTile.Triton

/-!
# `quantize_global` — strict per-kernel correctness (blocked tail)

`_quantize_global` quantizes a flat tensor to int8: program `pid` loads its block
`[pid·BLOCK_SIZE, (pid+1)·BLOCK_SIZE)` of `x`, loads the scalar `absmax_inv`, and
stores `llrint(127.0 * (x * absmax_inv))` to `output_ptr`, masked by
`offsets < n_elements`. The host computes `absmax = |x|.max()` and
`absmax_inv = 1/absmax` before the launch.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`_quantize_global[grid](...)`, the grid
`cdiv(n_elements, BLOCK_SIZE)`, the PyTorch-side `absmax` reduction returned
alongside the output, and the runtime composition of per-program writes) is the
*trusted boundary*. The program id is universally quantified, so the
per-program statement covers every program of the grid.

## Proof architecture

```
quantize_global_correctness                     ← TOP THEOREM (blocked surface ∧ quantizeGlobalIO ⊨ scaled quantize)
  ├─ quantize_global_surface_toAlgorithm?_blocked      full surface does NOT lower
  ├─ quantize_global_scaled_store_slice_flattenOk      bridge fragment membership
  ├─ quantize_global_scaled_store_slice_traceSafe      per-execution lane-wise safety walk
  └─ quantize_global_scaled_store_slice_region_run     region-model masked Hoare triple
       ├─ quantize_global_scaled_store_slice_correct   algorithm-layer readback per lane
       └─ quantize_global_scaled_store_slice_frame     masked scatter-store cell frame
```

The verified half of the headline is stated on the checked store slice's masked
**IO signature** `quantizeGlobalIO` (`Masked2DKernelIO₂`, the general-window
family): program `pid` reads the contiguous data window
`x_ptr[pid·BLOCK_SIZE + j]` under the tail mask `pid·BLOCK_SIZE + j <
n_elements`, while the scale is the **single cell** `absmax_inv_ptr[0]` read
**unmasked** — declared by `read2 := 0` with `read2Mask := True`, which the 1D
contiguous-window family cannot express. The kernel's grid is 1D; the signature
simply never mentions the second program-id axis. `⊨` is the audit-once masked
Hoare-triple combinator (`Masked2DKernelIO₂.Implements`): for **every** disjoint
placement of the three buffers in flat memory, **every** program id whose
declared reads are in bounds, and **every** launch state whose input windows
hold `xs`/`ys` at the declared lanes, the translated pointer kernel terminates,
every active output lane holds `scale127 * (xs i * ys i)`, and every other
memory cell is unchanged.

The headline carries one honest side condition, `0 < BLOCK_SIZE`: the scalar
load `tl.load(absmax_inv_ptr)` is **unmasked** while the active mask
`pid·BLOCK_SIZE + j < n_elements` is *pid-dependent* — a tail program past the
data (`pid·BLOCK_SIZE ≥ n_elements`) has **no** active lanes yet still executes
the scalar load, so its safety bound `0 < extent absmax_inv_ptr` cannot come
from the lane-gated data contract. The unconditional `read2Mask = True` clause
supplies it given any lane witness, i.e. `0 < BLOCK_SIZE`, which holds for
every real launch (the autotune configs use `BLOCK_SIZE ∈ {1024, 2048, 4096}`).

## Modeling boundary

Arithmetic is over `ℝ`, not bit-accurate IEEE float. This is the **honestly
blocked** quantization case: the faithful Python surface contains CUDA
`tl.extra.cuda.libdevice.llrint` plus the int8 store, and that surface is proven
*not* to lower through algorithm projection
(`quantize_global_surface_toAlgorithm?_blocked` returns `Except.error`) — the
first conjunct of the headline records exactly that. What is verified is the
real-valued pre-rounding store slice — the masked addressing, the scalar
`absmax_inv` load, and the value `127.0 * (x * absmax_inv)` — exactly up to the
`llrint` rounding / int8 cast, which are left as the blocker and are **not**
numerically modeled. `@triton.autotune` is not modeled; the statement is
dimension-general, covering every config.
-/

namespace VeriTile.Bench.TritonBenchG.QuantizeGlobal

open VeriTile.Triton
open scoped VeriTile.Triton.Masked2DKernelIO₂

/-- Faithful transcription of `quantize_global.py`'s `_quantize_global`.

The CUDA `llrint` operation is preserved as a surface operation; the algorithm
carrier records the pre-cast real value. -/
def quantize_global_surface
    (x_ptr absmax_inv_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(x_ptr + offsets, mask=mask)
  absmax_inv = tl.load(absmax_inv_ptr)
  output = tl.extra.cuda.libdevice.llrint(127.0 * (x * absmax_inv))
  tl.store(output_ptr + offsets, output, mask=mask)
}

/-- Proof-oriented arithmetic store slice of `quantize_global.py`'s
`_quantize_global`.

This slice proves the masked vector addressing and scaled value before the
backend-specific rounding/cast step. -/
def quantize_global_scaled_store_slice
    (x_ptr absmax_inv_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) (scale127 : ℝ) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(x_ptr + offsets, mask=mask)
  absmax_inv = tl.load(absmax_inv_ptr)
  output = $(scale127) * (x * absmax_inv)
  tl.store(output_ptr + offsets, output, mask=mask)
}

def offset (s : BlockState) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * BLOCK_SIZE + i.val

noncomputable def quantizeGlobalScaledSpec
    (s : BlockState) (x_ptr absmax_inv_ptr : RegionName)
    (BLOCK_SIZE : Nat) (scale127 : ℝ) (i : Fin BLOCK_SIZE) : ℝ :=
  scale127 * (s.readMem x_ptr (offset s BLOCK_SIZE i) *
    s.readMem absmax_inv_ptr 0)

/-- Algorithm-layer correctness for the masked global-quantization store slice. -/
theorem quantize_global_scaled_store_slice_correct
    (x_ptr absmax_inv_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) (scale127 : ℝ)
    (s : BlockState) :
    ∀ i : Fin BLOCK_SIZE,
      let outAddr := offset s BLOCK_SIZE i
      (exec (quantize_global_scaled_store_slice x_ptr absmax_inv_ptr output_ptr
            n_elements BLOCK_SIZE scale127) s).map (·.readMem output_ptr outAddr)
        = some (if outAddr < n_elements then
            quantizeGlobalScaledSpec s x_ptr absmax_inv_ptr BLOCK_SIZE scale127 i
          else s.readMem output_ptr outAddr) := by
  intro i
  simp [exec, quantize_global_scaled_store_slice, stepStmts, stepStmt, evalOp.eq_def,
        Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul,
        ComparableDType.lt, offset]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
        (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
  by_cases hi : s.pid * BLOCK_SIZE + i.val < n_elements
  · simp [hi, quantizeGlobalScaledSpec, offset]
  · simp [hi]

/-- The faithful full surface is intentionally blocked at compute projection:
the tested Python kernel stores CUDA `llrint`/int8 results, not the real-valued
pre-rounding expression proved by `quantize_global_scaled_store_slice`. -/
theorem quantize_global_surface_toAlgorithm?_blocked
    (x_ptr absmax_inv_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ∃ err,
      (quantize_global_surface x_ptr absmax_inv_ptr output_ptr
        n_elements BLOCK_SIZE).toAlgorithm? = Except.error err := by
  simp [quantize_global_surface, ComputeExpr.toAlgorithm?]

/-- A masked scatter-store `foldl` leaves every memory cell it does not
actively hit unchanged (cell-level frame for the masked store). -/
private theorem foldl_store_preserve_cell {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → ℝ) (P : α → Prop) [DecidablePred P]
    (r : RegionName) (o : Nat) (l : List α) (s : BlockState)
    (hnot : ∀ k ∈ l, P k → ¬(region = r ∧ offsetFn k = o)) :
    (l.foldl (fun acc k =>
        if P k then acc.writeMem region (offsetFn k) (valueFn k) else acc)
      s).mem r o = s.mem r o := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons]
      by_cases hP : P hd
      · rw [if_pos hP,
          ih _ (fun k hk => hnot k (List.mem_cons_of_mem hd hk)),
          BlockState.writeMem_mem]
        exact if_neg (fun hc =>
          hnot hd List.mem_cons_self hP ⟨hc.1.symm, hc.2.symm⟩)
      · rw [if_neg hP]
        exact ih _ (fun k hk => hnot k (List.mem_cons_of_mem hd hk))

set_option maxHeartbeats 1600000 in
/-- Frame half: every memory cell not actively written by the masked output
store is preserved by the run — in particular every cell of every region other
than `output_ptr`, and the *inactive* lanes of the output window itself. -/
private theorem quantize_global_scaled_store_slice_frame
    (x_ptr absmax_inv_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) (scale127 : ℝ) (s s1 : BlockState)
    (hExec : exec ((quantize_global_scaled_store_slice x_ptr absmax_inv_ptr
      output_ptr n_elements BLOCK_SIZE scale127).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ i : Fin BLOCK_SIZE, s.pid * BLOCK_SIZE + i.val < n_elements →
      ¬(output_ptr = r ∧ s.pid * BLOCK_SIZE + i.val = o)) :
    s1.mem r o = s.mem r o := by
  simp [exec, quantize_global_scaled_store_slice, ComputeKernel.toAlgKernel,
    stepStmts, stepStmt, evalOp.eq_def, Tile.bop, Tile.cop,
    NumericDType.add, NumericDType.mul, ComparableDType.lt] at hExec
  subst hExec
  refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
  intro k _ hmk hc
  exact hmiss k.1 (by simpa using hmk) hc

/-- **The region-model masked Hoare triple** — termination, active-lane output
values, and frame off the active output lanes, from any launch state whose data
window holds `xs` and whose scale cell holds `ys`. This is the `hrun`
obligation of `Masked2DKernelIO₂.Implements.intro`; the value half reuses
`quantize_global_scaled_store_slice_correct` (instantiated at the values the
state actually holds). -/
theorem quantize_global_scaled_store_slice_region_run
    (x_ptr absmax_inv_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) (scale127 : ℝ)
    (s₀ : BlockState) (xs ys : Fin BLOCK_SIZE → ℝ)
    (hx : ∀ j : Fin BLOCK_SIZE, s₀.pid * BLOCK_SIZE + j.val < n_elements →
      s₀.readMem x_ptr (s₀.pid * BLOCK_SIZE + j.val) = xs j)
    (hy : ∀ j : Fin BLOCK_SIZE, s₀.readMem absmax_inv_ptr 0 = ys j) :
    ∃ s1, exec ((quantize_global_scaled_store_slice x_ptr absmax_inv_ptr
        output_ptr n_elements BLOCK_SIZE scale127).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin BLOCK_SIZE, s₀.pid * BLOCK_SIZE + j.val < n_elements →
          s1.readMem output_ptr (s₀.pid * BLOCK_SIZE + j.val)
            = scale127 * (xs j * ys j))
      ∧ (∀ r o,
          (r ≠ output_ptr ∨ ∀ j : Fin BLOCK_SIZE,
            s₀.pid * BLOCK_SIZE + j.val < n_elements →
            o ≠ s₀.pid * BLOCK_SIZE + j.val) →
          s1.mem r o = s₀.mem r o) := by
  have hobs := quantize_global_scaled_store_slice_correct x_ptr absmax_inv_ptr
    output_ptr n_elements BLOCK_SIZE scale127 s₀
  rw [show exec (quantize_global_scaled_store_slice x_ptr absmax_inv_ptr
      output_ptr n_elements BLOCK_SIZE scale127) s₀
      = exec ((quantize_global_scaled_store_slice x_ptr absmax_inv_ptr
          output_ptr n_elements BLOCK_SIZE scale127).toAlgKernel) s₀
      from rfl] at hobs
  cases hsrc : exec ((quantize_global_scaled_store_slice x_ptr absmax_inv_ptr
      output_ptr n_elements BLOCK_SIZE scale127).toAlgKernel) s₀ with
  | none =>
      exact absurd hsrc (by
        simp [exec, quantize_global_scaled_store_slice, ComputeKernel.toAlgKernel,
          stepStmts, stepStmt, evalOp.eq_def, Tile.bop, Tile.cop,
          NumericDType.add, NumericDType.mul, ComparableDType.lt])
  | some s1 =>
      refine ⟨s1, rfl, fun j hj => ?_, fun r o hcond => ?_⟩
      · have hje := hobs j
        rw [hsrc] at hje
        simp only [offset, Option.map_some, Option.some_inj, if_pos hj] at hje
        rw [hje]
        simp only [quantizeGlobalScaledSpec, offset]
        rw [hx j hj, hy j]
      · refine quantize_global_scaled_store_slice_frame x_ptr absmax_inv_ptr
          output_ptr n_elements BLOCK_SIZE scale127 s₀ s1 hsrc r o
          (fun i hi ⟨hr, ho⟩ => ?_)
        rcases hcond with hne | hno
        · exact hne hr.symm
        · exact hno i hi ho.symm

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk: the masked data load and the masked store
address the window `pid * BLOCK_SIZE + j`, active only when
`pid * BLOCK_SIZE + j < n_elements`; the **unmasked** scale load reads the
single cell `absmax_inv_ptr[0]`, whose bound is taken as the explicit
hypothesis `hscale` (a tail program has no active lane to derive it from). -/
theorem quantize_global_scaled_store_slice_traceSafe
    (x_ptr absmax_inv_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) (scale127 : ℝ)
    (bounds : RegionBounds) (s : BlockState)
    (hx : ∀ j : Fin BLOCK_SIZE, s.pid * BLOCK_SIZE + j.val < n_elements →
      s.pid * BLOCK_SIZE + j.val < bounds x_ptr)
    (hscale : 0 < bounds absmax_inv_ptr)
    (hout : ∀ j : Fin BLOCK_SIZE, s.pid * BLOCK_SIZE + j.val < n_elements →
      s.pid * BLOCK_SIZE + j.val < bounds output_ptr) :
    Kernel.TraceSafe bounds
      ((quantize_global_scaled_store_slice x_ptr absmax_inv_ptr output_ptr
        n_elements BLOCK_SIZE scale127).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  -- Computational unroll: walks all eight statements, discharging every
  -- load-free `SafeAt` and reducing the three memory accesses' lane-wise
  -- address obligations to the bounds hypotheses below.
  simp [quantize_global_scaled_store_slice, ComputeKernel.toAlgKernel,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt, MaskOpt.SafeAt,
    stepStmt, evalOp.eq_def,
    Tile.bop, Tile.cop,
    NumericDType.add, NumericDType.mul,
    ComparableDType.lt,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe, MemAccess.SafeAt,
    MaskOpt.Active, BlockState.setReg]
  exact ⟨fun a ha => hx a ha, hscale, fun a ha => hout a ha⟩

/-- The store slice sits inside the flat-memory bridge's covered fragment. -/
theorem quantize_global_scaled_store_slice_flattenOk
    (x_ptr absmax_inv_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) (scale127 : ℝ) :
    ((quantize_global_scaled_store_slice x_ptr absmax_inv_ptr output_ptr
      n_elements BLOCK_SIZE scale127).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [quantize_global_scaled_store_slice, ComputeKernel.toAlgKernel,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]

/-- The checked store slice's masked **IO signature** — the whole
kernel-specific audit surface of the headline's verified half: which buffer is
which argument (the wiring), where program `pid` reads its `BLOCK_SIZE`-lane
data window (`x_ptr[pid·BLOCK_SIZE + j]`), where it reads its scale — the
**single scalar cell** `absmax_inv_ptr[0]`, the same address at every lane,
read **unmasked** (`read2Mask := True`: the load executes even for a tail
program with no active data lanes) — where it writes its output window, and
the active-lane predicate `pid·BLOCK_SIZE + j < n_elements`. The kernel's grid
is 1D, so no field mentions the second program-id axis. The windows and masks
are declared, not parsed from the kernel: they formalize the host-side launch
convention (`offsets = pid * BLOCK_SIZE + arange; mask = offsets <
n_elements`, one global scale), and the headline **proves** the kernel's
actual addressing and masking match them. Buffer sizes are not signature
content: the headline quantifies over every allocation whose extents cover the
declared reads and writes. -/
def quantizeGlobalIO (x_ptr absmax_inv_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) (scale127 : ℝ) : Masked2DKernelIO₂ where
  kernel := quantize_global_scaled_store_slice x_ptr absmax_inv_ptr output_ptr
    n_elements BLOCK_SIZE scale127
  in1 := x_ptr
  in2 := absmax_inv_ptr
  out := output_ptr
  B := BLOCK_SIZE
  read1 := fun pid _ j => pid * BLOCK_SIZE + j.val
  read2 := fun _ _ _ => 0
  write := fun pid _ j => pid * BLOCK_SIZE + j.val
  mask := fun pid _ j => pid * BLOCK_SIZE + j.val < n_elements
  read2Mask := fun _ _ _ => True

/-- **The headline** (blocked tail): the faithful full surface is recorded as
**blocked** at algorithm projection (it stores CUDA `llrint`/int8 results — the
honest, unmodeled blocker), while the checked store slice implements the
pre-rounding scaled quantization on its masked IO signature — for every
disjoint flat placement of the three buffers, every program id whose declared
reads are in bounds, and every launch state whose data window holds `xs` at
the active lanes and whose scale cell holds `ys`, the translated pointer
kernel terminates, every active output lane holds `scale127 * (xs i * ys i)`
(`ys` is constant across the lanes since every lane reads the same cell;
concrete launches use `scale127 = 127.0`), and every other memory cell is
unchanged.

`0 < BLOCK_SIZE` is genuinely forced: the scale load is unmasked and the
active mask is pid-dependent, so a tail program with no active lanes still
reads `absmax_inv_ptr[0]`, whose bound only the unconditional `read2Mask`
clause supplies — via a lane witness, hence `0 < BLOCK_SIZE`. It holds for
every real launch. Proof: `Masked2DKernelIO₂.Implements.intro` assembles the
region-model masked triple with the flat-memory bridge side conditions. -/
specification quantize_global_correctness
    (x_ptr absmax_inv_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) (scale127 : ℝ)
    (hBS : 0 < BLOCK_SIZE) :
    (∃ err, (quantize_global_surface x_ptr absmax_inv_ptr output_ptr
        n_elements BLOCK_SIZE).toAlgorithm? = Except.error err) ∧
    (quantizeGlobalIO x_ptr absmax_inv_ptr output_ptr n_elements BLOCK_SIZE
        scale127 ⊨ fun _ _ xs ys i => scale127 * (xs i * ys i)) := by
  refine ⟨quantize_global_surface_toAlgorithm?_blocked x_ptr absmax_inv_ptr
      output_ptr n_elements BLOCK_SIZE, ?_⟩
  refine Masked2DKernelIO₂.Implements.intro _ ?_ ?_ ?_
  · exact quantize_global_scaled_store_slice_flattenOk x_ptr absmax_inv_ptr
      output_ptr n_elements BLOCK_SIZE scale127
  · intro bounds s h1 h2 h3 _
    exact quantize_global_scaled_store_slice_traceSafe x_ptr absmax_inv_ptr
      output_ptr n_elements BLOCK_SIZE scale127 bounds s h1
      (h2 ⟨0, hBS⟩ trivial) h3
  · intro s₀ xs ys hx hy
    obtain ⟨s1, hexec, hval, hframe⟩ := quantize_global_scaled_store_slice_region_run
      x_ptr absmax_inv_ptr output_ptr n_elements BLOCK_SIZE scale127 s₀ xs ys
      hx (fun j => hy j trivial)
    -- scratch is empty, so its frame side condition is vacuous
    exact ⟨s1, hexec, hval, fun r o hout _ => hframe r o hout⟩

end VeriTile.Bench.TritonBenchG.QuantizeGlobal
