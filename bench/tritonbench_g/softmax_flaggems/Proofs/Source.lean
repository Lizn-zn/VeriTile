import bench.tritonbench_g.softmax_flaggems.Kernels

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.SourceChecks
open Lean VeriTile Triton FullSource

/-- Regression for the real mixed-type maximum used by both forward fallback loops. -/
def mixedMaximum (x : RegionName) (B : Nat) := triton {
  tl.region x = tl.bfloat16
  m = tl.full([$(B)], value=-float("inf"), dtype=tl.float32)
  inp = tl.load(x + tl.arange(0, $(B)))
  m_new = tl.maximum(m, inp)
}

theorem mixedMaximum_expansion (x : RegionName) (B : Nat) :
    (mixedMaximum x B).surfaceBody =
      [.assign .real [B] "m" (.compute (.full [B] (.alg .fp32 .negInf))),
       .assign .bf16 [B] "inp" (.alg (.load .bf16 (.region x (.arange B)) .none)),
       .assign .real [B] "m_new" (.compute (.alg .fp32
         (.where (.gt .real (.consSame .nil) (.ref .real [B] "m")
           (.castFloat .bf16 .real (.ref .bf16 [B] "inp")))
           (.ref .real [B] "m")
           (.castFloat .bf16 .real (.ref .bf16 [B] "inp")))))] := rfl

/-- Collect actual assignment channels through both branches and every loop body. -/
def assignmentTypes (name : RegName) : List ComputeStmt → List TileDType
  | [] => []
  | .assign dtype _ reg _ :: rest =>
      (if reg = name then [dtype] else []) ++ assignmentTypes name rest
  | .ifThenElse _ yes no :: rest =>
      assignmentTypes name yes ++ assignmentTypes name no ++ assignmentTypes name rest
  | .ifThen _ body :: rest => assignmentTypes name body ++ assignmentTypes name rest
  | .forRangeDyn _ _ _ _ body :: rest =>
      assignmentTypes name body ++ assignmentTypes name rest
  | .forRange _ _ _ _ body :: rest =>
      assignmentTypes name body ++ assignmentTypes name rest
  | .forLoop _ _ body :: rest => assignmentTypes name body ++ assignmentTypes name rest
  | _ :: rest => assignmentTypes name rest

/-- Pointer rebinding in the fallback retains the actual fp16 load channel. -/
theorem inner_load_channels (M N B : Nat) (oneTile : Bool) :
    assignmentTypes "inp"
      (softmax_kernel_inner .fp16 .fp32 "Y" "X" M N B oneTile).surfaceBody =
      [.real, .fp16, .fp16, .fp16, .fp16] := by
  simp [softmax_kernel_inner, ComputeKernel.surfaceBody, assignmentTypes]

/-- All three non-inner load sites retain bf16 storage before arithmetic promotion. -/
theorem non_inner_load_channels (M N K BN BK : Nat) (oneTile : Bool) :
    assignmentTypes "inp"
      (softmax_kernel_non_inner .bf16 .fp32 "Y" "X" M N K BN BK oneTile).surfaceBody =
      [.bf16, .bf16, .bf16] := by
  simp [softmax_kernel_non_inner, ComputeKernel.surfaceBody, assignmentTypes]

syntax "flaggemsMaxProbe%" "[" tritonDType "]" : term
set_option hygiene false in
macro_rules
  | `(flaggemsMaxProbe%[$dtype:tritonDType]) =>
    `(fun (x : RegionName) (B : Nat) => triton {
      tl.region x = $dtype
      inp = tl.load(x + tl.arange(0, $(B)))
      m = tl.max(inp, 0)
      delta = inp - m
      e = tl.exp(delta)
    })

def fp16Maximum := flaggemsMaxProbe%[tl.float16]
def bf16Maximum := flaggemsMaxProbe%[tl.bfloat16]
def fp32Maximum := flaggemsMaxProbe%[tl.float32]

/-- Triton v3.5.1 standard.max promotes sub-32-bit floats before reduction. -/
theorem fp16_maximum_promoted (x : RegionName) (B : Nat) :
    (fp16Maximum x B).surfaceBody[1]? = some
      (.assign .real [] "m" (.compute (.alg .fp32
        (.reduceMax ⟨0, by simp⟩ Bool.false
          (.castFloat .fp16 .real (.ref .fp16 [B] "inp")))))) := rfl

theorem bf16_maximum_promoted (x : RegionName) (B : Nat) :
    (bf16Maximum x B).surfaceBody[1]? = some
      (.assign .real [] "m" (.compute (.alg .fp32
        (.reduceMax ⟨0, by simp⟩ Bool.false
          (.castFloat .bf16 .real (.ref .bf16 [B] "inp")))))) := rfl

theorem fp16_subtraction_after_promotion (x : RegionName) (B : Nat) :
    (fp16Maximum x B).surfaceBody[2]? = some
      (.assign .real [B] "delta" (.compute (.alg .fp32
        (.sub .real .scalarR (.castFloat .fp16 .real (.ref .fp16 [B] "inp"))
          (.ref .real [] "m"))))) ∧
    (fp16Maximum x B).surfaceBody[3]? = some
      (.assign .real [B] "e" (.compute (.alg .fp32
        (.exp (.ref .real [B] "delta"))))) := ⟨rfl, rfl⟩

theorem bf16_subtraction_after_promotion (x : RegionName) (B : Nat) :
    (bf16Maximum x B).surfaceBody[2]? = some
      (.assign .real [B] "delta" (.compute (.alg .fp32
        (.sub .real .scalarR (.castFloat .bf16 .real (.ref .bf16 [B] "inp"))
          (.ref .real [] "m"))))) ∧
    (bf16Maximum x B).surfaceBody[3]? = some
      (.assign .real [B] "e" (.compute (.alg .fp32
        (.exp (.ref .real [B] "delta"))))) := ⟨rfl, rfl⟩

/-- The existing fp32 reduction constructor and carrier remain unchanged. -/
theorem fp32_maximum_unchanged (x : RegionName) (B : Nat) :
    (fp32Maximum x B).surfaceBody[1]? = some
      (.assign .real [] "m" (.compute (.alg .fp32
        (.reduceMax ⟨0, by simp⟩ Bool.false (.ref .real [B] "inp"))))) := rfl

def fp64Maximum := flaggemsMaxProbe%[tl.float64]

/-- One-tile fp64 load/max/subtract/exp retains fp64 precision. This does not
assert legality of the separate fp32-initialized fallback accumulator loops. -/
theorem fp64_one_tile_precision (x : RegionName) (B : Nat) :
    (fp64Maximum x B).surfaceBody =
      [.assign .real [B] "inp" (.compute (.alg .fp64 (.load .real (.region x (.arange B)) .none))),
       .assign .real [] "m" (.compute (.alg .fp64
         (.reduceMax ⟨0, by simp⟩ Bool.false (.ref .real [B] "inp")))),
       .assign .real [B] "delta" (.compute (.alg .fp64
         (.sub .real .scalarR (.ref .real [B] "inp") (.ref .real [] "m")))),
       .assign .real [B] "e" (.compute (.alg .fp64 (.exp (.ref .real [B] "delta"))))] := rfl

def exprAlgPrecision {dtype : AlgDType} {shape : TileShape} (e : ComputeExpr dtype shape) : Option ComputeDType :=
  match e with
  | .compute op => match op with
    | .alg precision _ => some precision
    | _ => none
  | _ => none

def stmtAlgPrecision (name : RegName) (stmt : ComputeStmt) : Option ComputeDType :=
  match stmt with
  | .assign _ _ reg e => if reg = name then exprAlgPrecision e else none
  | _ => none

def topAlgPrecisions (name : RegName) (body : List ComputeStmt) : List ComputeDType :=
  body.filterMap (stmtAlgPrecision name)

def oneTileSurface (body : List ComputeStmt) (branchIndex : Nat) : List ComputeStmt :=
  match body[branchIndex]? with
  | some (.ifThenElse _ yes _) => yes
  | _ => []

/-- Precision provenance in the actual complete non-inner one-tile source. -/
theorem fp64_non_inner_source_precision (Y X : RegionName) (M N K B C : Nat) :
    let body := oneTileSurface (softmax_kernel_non_inner .fp64 .fp64 Y X M N K B C Bool.true).surfaceBody 3
    topAlgPrecisions "inp" body = [.fp64] ∧ topAlgPrecisions "m" body = [.fp64] ∧
      topAlgPrecisions "e" body = [.fp64] ∧ topAlgPrecisions "out" body = [.fp64] := by
  simp [oneTileSurface, softmax_kernel_non_inner, ComputeKernel.surfaceBody, topAlgPrecisions, stmtAlgPrecision, exprAlgPrecision]

/-- Precision provenance in the actual complete inner one-tile source. -/
theorem fp64_inner_source_precision (Y X : RegionName) (M N B : Nat) :
    let body := oneTileSurface (softmax_kernel_inner .fp64 .fp64 Y X M N B Bool.true).surfaceBody 1
    topAlgPrecisions "inp" body = [.fp64] ∧ topAlgPrecisions "m" body = [.fp64] ∧
      topAlgPrecisions "e" body = [.fp64] ∧ topAlgPrecisions "out" body = [.fp64] := by
  simp [oneTileSurface, softmax_kernel_inner, ComputeKernel.surfaceBody, topAlgPrecisions, stmtAlgPrecision, exprAlgPrecision]

/-- Both backward input pointers share the declared fp64 storage profile. -/
theorem fp64_backward_non_inner_source_precision (P G Y : RegionName) (M N K B C : Nat) :
    let body := oneTileSurface (softmax_backward_kernel_non_inner .fp64 P G Y M N K B C Bool.true).surfaceBody 3
    topAlgPrecisions "out_tile" body = [.fp64] ∧ topAlgPrecisions "out_grad_tile" body = [.fp64] ∧
      topAlgPrecisions "scale" body = [.fp64] ∧ topAlgPrecisions "in_grad_tile" body = [.fp64] := by
  simp [oneTileSurface, softmax_backward_kernel_non_inner, ComputeKernel.surfaceBody, topAlgPrecisions, stmtAlgPrecision, exprAlgPrecision]

theorem fp64_backward_inner_source_precision (P G Y : RegionName) (M N B C : Nat) :
    let body := oneTileSurface (softmax_backward_kernel_inner .fp64 P G Y M N C B Bool.true).surfaceBody 2
    topAlgPrecisions "out_tile" body = [.fp64] ∧ topAlgPrecisions "out_grad_tile" body = [.fp64] ∧
      topAlgPrecisions "scale" body = [.fp64] ∧ topAlgPrecisions "in_grad_tile" body = [.fp64] := by
  simp [oneTileSurface, softmax_backward_kernel_inner, ComputeKernel.surfaceBody, topAlgPrecisions, stmtAlgPrecision, exprAlgPrecision]

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.SourceChecks
