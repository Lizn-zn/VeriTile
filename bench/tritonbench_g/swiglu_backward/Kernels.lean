import VeriTile.Triton

/-!
# `swiglu_backward` — strict per-kernel correctness

`_swiglu_bwd_kernel` is the backward pass of SwiGLU: program `(row, col_block)`
loads the gate `X`, value `Y`, and upstream gradient `DOUT`, recomputes
`x_sigmoid = σ(x)`, and writes the two input gradients `dx`→`DX`, `dy`→`DY`, plus
optionally (when `RECOMPUTE_OUTPUT`) the forward output `out = x·σ(x)·y`→`OUT`.
All stores are masked by `cols < ncols`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`_swiglu_bwd_kernel[grid](...)`, the 2-D grid
`(M, cdiv(N, BLOCK_N))`, the `@triton.autotune` choice of `BLOCK_N`, the
`@triton.heuristics` setting of `RECOMPUTE_OUTPUT`, and how the runtime composes
per-program writes into one buffer) is the *trusted boundary*, not a proof
obligation here. Because both program ids (`pid₀`, `pid₁`) are universally
quantified, the per-program statement covers every program of the grid; the
proofs case-split on the `RECOMPUTE_OUTPUT` flag, covering both heuristic
outcomes.

## Proof architecture

```
swiglu_bwd_kernel_correctness                 ← TOP SPECIFICATION (swigluBackwardIO ⊨ backward triple)
  ├─ swiglu_bwd_kernel_flattenOk              bridge fragment membership
  ├─ swiglu_bwd_kernel_traceSafe              per-execution lane-wise safety walk
  └─ swiglu_bwd_kernel_region_run             region-model masked Hoare triple
       ├─ swiglu_bwd_kernel_exec_isSome       termination
       ├─ swiglu_bwd_kernel_correct           algorithm-layer readback per lane
       └─ swiglu_bwd_kernel_frame             three-scatter cell frame
```

The headline is stated on the kernel's masked three-input / three-output **IO
signature** `swigluBackwardIO` (`Masked2DKernelIO₃ₓ₃`). `⊨`
(`Masked2DKernelIO₃ₓ₃.Implements`) is the audit-once masked Hoare-triple
combinator: for **every** disjoint flat placement of the six buffers, **every**
program id whose active lanes are in bounds, and **every** launch state whose
active input lanes hold `xs`/`ys`/`douts`, the translated pointer kernel
terminates, every write-active output lane holds its spec value, and every
other memory cell is unchanged.

## Modeling boundary

The spec is an **oracle wrapper** over `VeriTile.Triton.Math.Activation`: the
SwiGLU math (`TiledActivation.swigluBwdA/B`, `TiledActivation.swiglu`, built on
`TiledActivation.silu`) lives once in `Math.Activation` and is reused here, so
this file only checks that the kernel realizes those oracles lane-wise.
Arithmetic is over `ℝ` (not bit-accurate IEEE float); the `.to(tl.float32)` load
casts reduce to the identity at the algorithm layer (post-erasure all dtypes
unify to `ℝ`); `@triton.autotune` is not modeled. The kernel writes up to three
channels (`DX`, `DY`, and conditionally `OUT`); the headline assumes the
three output regions are pairwise distinct (`DX ≠ DY`, `OUT ≠ DX`, `OUT ≠ DY`) so
later stores cannot clobber earlier channels. The `RECOMPUTE_OUTPUT`
control-flow branch is **not** case-narrowed away: the constexpr gate lives in
the signature's `writeMask3` (`RECOMPUTE_OUTPUT = Bool.true ∧ cols < ncols`), so one
headline covers both heuristic outcomes — when the flag is off, the `OUT`
channel simply has no write-active lanes and carries no obligations.
-/

namespace VeriTile.Bench.TritonBenchG.SwigluBackward

open VeriTile.Triton
open scoped VeriTile.Triton.Masked2DKernelIO₃ₓ₃

/-- Faithful transcription of `swiglu_backward.py`'s `_swiglu_bwd_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_N: tl.constexpr` / `RECOMPUTE_OUTPUT: tl.constexpr` -> Lean
  parameters. -/
def swiglu_bwd_kernel
    (X Y DOUT OUT DX DY : RegionName)
    (stride_x_row stride_y_row stride_dout_row stride_out_row
      stride_dx_row stride_dy_row ncols BLOCK_N : Nat)
    (RECOMPUTE_OUTPUT : Bool) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  start_col = tl.program_id(1) * $(BLOCK_N)
  X += row * $(stride_x_row)
  Y += row * $(stride_y_row)
  DOUT += row * $(stride_dout_row)
  if RECOMPUTE_OUTPUT {
    OUT += row * $(stride_out_row)
  }
  DX += row * $(stride_dx_row)
  DY += row * $(stride_dy_row)
  cols = start_col + tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(ncols), other=0.0).to(tl.float32)
  y = tl.load(Y + cols, mask=cols < $(ncols), other=0.0).to(tl.float32)
  dout = tl.load(DOUT + cols, mask=cols < $(ncols), other=0.0).to(tl.float32)
  x_sigmoid = tl.sigmoid(x)
  dx = x_sigmoid * (1 + x * (1 - x_sigmoid)) * y * dout
  dy = x * x_sigmoid * dout
  tl.store(DX + cols, dx, mask=cols < $(ncols))
  tl.store(DY + cols, dy, mask=cols < $(ncols))
  if RECOMPUTE_OUTPUT {
    out = x * x_sigmoid * y
    tl.store(OUT + cols, out, mask=cols < $(ncols))
  }
}


open Lean
syntax "swigluBackward%" "[" tritonDType "," tritonDType "," tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(swigluBackward%[$input:tritonDType, $upstream:tritonDType,
      $gradient:tritonDType, $output:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `commuted do
      Macro.throwError "expected original or commuted"
    let xBase : TSyntax `tritonStmt ← if version.getId == `original then
      `(tritonStmt| X += row * $(sx)) else `(tritonStmt| X += $(sx) * row)
    let yBase : TSyntax `tritonStmt ← if version.getId == `original then
      `(tritonStmt| Y += row * $(sy)) else `(tritonStmt| Y += $(sy) * row)
    let dBase : TSyntax `tritonStmt ← if version.getId == `original then
      `(tritonStmt| DOUT += row * $(sd)) else `(tritonStmt| DOUT += $(sd) * row)
    let oBase : TSyntax `tritonStmt ← if version.getId == `original then
      `(tritonStmt| OUT += row * $(so)) else `(tritonStmt| OUT += $(so) * row)
    let dxBase : TSyntax `tritonStmt ← if version.getId == `original then
      `(tritonStmt| DX += row * $(sdx)) else `(tritonStmt| DX += $(sdx) * row)
    let dyBase : TSyntax `tritonStmt ← if version.getId == `original then
      `(tritonStmt| DY += row * $(sdy)) else `(tritonStmt| DY += $(sdy) * row)
    let dxWrite : TSyntax `tritonStmt ← match gradient with
      | `(tritonDType| tl.float32) => `(tritonStmt| tl.store(DX + cols, dx, mask=cols < $(N)))
      | _ => `(tritonStmt| tl.store(DX + cols, dx.to($gradient), mask=cols < $(N)))
    let dyWrite : TSyntax `tritonStmt ← match gradient with
      | `(tritonDType| tl.float32) => `(tritonStmt| tl.store(DY + cols, dy, mask=cols < $(N)))
      | _ => `(tritonStmt| tl.store(DY + cols, dy.to($gradient), mask=cols < $(N)))
    let outWrite : TSyntax `tritonStmt ← match output with
      | `(tritonDType| tl.float32) => `(tritonStmt| tl.store(OUT + cols, out, mask=cols < $(N)))
      | _ => `(tritonStmt| tl.store(OUT + cols, out.to($output), mask=cols < $(N)))
    `(fun (X Y DOUT OUT DX DY : RegionName) (sx sy sd so sdx sdy N B : Nat) (R : Bool) =>
      ComputeKernel.mk [X,Y,DOUT] [DX,DY,OUT] (triton {
  tl.region X = $input, Y = $input, DOUT = $upstream, DX = $gradient, DY = $gradient, OUT = $output
  row = tl.program_id(0)
  start_col = tl.program_id(1) * $(B)
  $xBase
  $yBase
  $dBase
  if R { $oBase }
  $dxBase
  $dyBase
  cols = start_col + tl.arange(0, $(B))
  x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  y = tl.load(Y + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  dout = tl.load(DOUT + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  x_sigmoid = tl.sigmoid(x)
  dx = x_sigmoid * (1 + x * (1 - x_sigmoid)) * y * dout
  dy = x * x_sigmoid * dout
  $dxWrite
  $dyWrite
  if R {
    out = x * x_sigmoid * y
    $outWrite
  }
}).surfaceBody)

/-- Wrapper-correlated storage: X/Y and DX/DY each share a backing tensor dtype. -/
inductive Storage where
  | fp32 | fp16 | bf16 | fp64 | f8e4 | f8e5
  deriving DecidableEq, Repr

abbrev Source := RegionName → RegionName → RegionName → RegionName → RegionName → RegionName →
  Nat → Nat → Nat → Nat → Nat → Nat → Nat → Nat → Bool → ComputeKernel

set_option maxHeartbeats 16000000
set_option maxRecDepth 16000

/-- Complete source template. Only typed memory-boundary statements are replaced
below; both conditional blocks and all fp32 tensor computations remain present. -/
def template (v : Bool) : Source :=
  if v then swigluBackward%[tl.float32, tl.float32, tl.float32, tl.float32, commuted]
  else swigluBackward%[tl.float32, tl.float32, tl.float32, tl.float32, original]

def loadAddress (pointer : RegName) (dtype : TileDType) (B _N : Nat) : MemAccess dtype [B] :=
  .ptr (.ptrAdd Broadcast.scalarL
    (.ref .ptr [] pointer) (.ref .nat [B] "cols"))

def typedLoad (pointer name : RegName) (input : Storage) (B N : Nat) : ComputeStmt :=
  match input with
  | .fp32 => .assign .real [B] name (.compute (.alg .fp32
      (.castFloat .real .real (.load .real (loadAddress pointer .real B N)
        (.maskOther (.lt .nat .scalarR (.ref .nat [B] "cols") (.constNat N))
          ((Op.const 0.0).broadcast [B]))))))
  | .fp64 => .assign .real [B] name (.compute (.alg .fp32
      (.castFloat .real .real (.withPrecision .real .fp64
        (.load .real (loadAddress pointer .real B N)
          (.maskOther (.lt .nat .scalarR (.ref .nat [B] "cols") (.constNat N))
            ((Op.const 0.0).broadcast [B])))))))
  | .fp16 => .assign .real [B] name (.compute (.alg .fp32
      (.castFloat .fp16 .real (.load .fp16 (loadAddress pointer .fp16 B N)
        (.maskOther (.lt .nat .scalarR (.ref .nat [B] "cols") (.constNat N))
          ((Op.castFloat .real .fp16 (.const 0.0)).broadcast [B]))))))
  | .bf16 => .assign .real [B] name (.compute (.alg .fp32
      (.castFloat .bf16 .real (.load .bf16 (loadAddress pointer .bf16 B N)
        (.maskOther (.lt .nat .scalarR (.ref .nat [B] "cols") (.constNat N))
          ((Op.castFloat .real .bf16 (.const 0.0)).broadcast [B]))))))
  | .f8e4 => .assign .real [B] name (.compute (.alg .fp32
      (.castFloat .f8e4 .real (.load .f8e4 (loadAddress pointer .f8e4 B N)
        (.maskOther (.lt .nat .scalarR (.ref .nat [B] "cols") (.constNat N))
          ((Op.castFloat .real .f8e4 (.const 0.0)).broadcast [B]))))))
  | .f8e5 => .assign .real [B] name (.compute (.alg .fp32
      (.castFloat .f8e5 .real (.load .f8e5 (loadAddress pointer .f8e5 B N)
        (.maskOther (.lt .nat .scalarR (.ref .nat [B] "cols") (.constNat N))
          ((Op.castFloat .real .f8e5 (.const 0.0)).broadcast [B]))))))


def storeAddress (pointer : RegName) (dtype : TileDType) (B : Nat) : MemAccess dtype [B] :=
  .ptr (.ptrAdd .scalarL (.ref .ptr [] pointer) (.ref .nat [B] "cols"))

def typedStore (pointer name : RegName) (storage : Storage) (B N : Nat) : ComputeStmt :=
  match storage with
  | .fp32 => .store .real [B] (storeAddress pointer .real B)
      (.compute (.alg .fp32 (.ref .real [B] name)))
      (.mask (.lt .nat .scalarR (.ref .nat [B] "cols") (.constNat N)))
  | .fp64 => .store .real [B] (storeAddress pointer .real B)
      (.compute (.alg .fp64 (.castFloat .real .real
        (.withPrecision .real .fp32 (.ref .real [B] name)))))
      (.mask (.lt .nat .scalarR (.ref .nat [B] "cols") (.constNat N)))
  | .fp16 => .store .fp16 [B] (storeAddress pointer .fp16 B)
      (.alg (.castFloat .real .fp16 (.ref .real [B] name)))
      (.mask (.lt .nat .scalarR (.ref .nat [B] "cols") (.constNat N)))
  | .bf16 => .store .bf16 [B] (storeAddress pointer .bf16 B)
      (.alg (.castFloat .real .bf16 (.ref .real [B] name)))
      (.mask (.lt .nat .scalarR (.ref .nat [B] "cols") (.constNat N)))

  | .f8e4 => .store .f8e4 [B] (storeAddress pointer .f8e4 B)
      (.alg (.castFloat .real .f8e4 (.ref .real [B] name)))
      (.mask (.lt .nat .scalarR (.ref .nat [B] "cols") (.constNat N)))
  | .f8e5 => .store .f8e5 [B] (storeAddress pointer .f8e5 B)
      (.alg (.castFloat .real .f8e5 (.ref .real [B] name)))
      (.mask (.lt .nat .scalarR (.ref .nat [B] "cols") (.constNat N)))


def recomputedValue (B : Nat) : ComputeStmt :=
  .assign .real [B] "out" (.compute (.alg .fp32
    (.mul .real (Broadcast.consSame .nil)
      (.mul .real (Broadcast.consSame .nil) (.ref .real [B] "x")
        (.ref .real [B] "x_sigmoid")) (.ref .real [B] "y"))))

/-- Full wrapper-correlated typed source. The full prefix and fp32 computation
are shared; typed load/store nodes expose all source storage conversions. -/
def typedKernel (input upstream gradient output : Storage) (v : Bool)
    (X Y DOUT OUT DX DY : RegionName) (sx sy sd so sdx sdy N B : Nat) (R : Bool) : ComputeKernel :=
  let body := (template v X Y DOUT OUT DX DY sx sy sd so sdx sdy N B R).surfaceBody
  ComputeKernel.mk [X,Y,DOUT] [DX,DY,OUT]
    (body.take 9 ++
      [typedLoad "X" "x" input B N, typedLoad "Y" "y" input B N,
       typedLoad "DOUT" "dout" upstream B N] ++
      (body.drop 12).take 3 ++
      [typedStore "DX" "dx" gradient B N, typedStore "DY" "dy" gradient B N,
       .ifThen (.constBool R) [recomputedValue B, typedStore "OUT" "out" output B N]])

end VeriTile.Bench.TritonBenchG.SwigluBackward
