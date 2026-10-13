import bench.tritonbench_g.flash_attn.Proofs.FP
import VeriTile.Triton.Float.ProfiledSoundness
import VeriTile.Triton.Semantics.TileOps

/-! A real-arithmetic backend with a genuine negative-infinity sentinel.
Casts are exact, reductions/dots are the actual finite sums/maxima, and exp2 /
log2 retain their mathematical meanings. This is an execution witness backend,
not an assertion that IEEE rounding equals real arithmetic. -/
namespace VeriTile.Bench.TritonBenchG.FlashAttn.NumericTypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite TypedSource
noncomputable section
open scoped Classical

def model : Algebra (WithBot ℝ) where
  literal := fun _ _ r => some r
  negInf := none
  binary := fun _ _ op a b => match op with
    | .add => WithBot.realAdd a b
    | .sub => WithBot.realSub a b
    | .mul => WithBot.realMul a b
    | .div => WithBot.realDiv a b
    | .max => max a b
    | .pow => some 0
  atomicAdd := fun _ => some WithBot.realAdd
  unary := fun _ op a => match op with
    | .exp2 => WithBot.realExp2 a
    | .log2 => WithBot.realLog2 a
    | _ => some 0
  compareLe := fun _ _ => some (fun a b => decide (a ≤ b))
  compareLt := fun _ _ => some (fun a b => decide (a < b))
  compareEq := fun _ _ => some (fun a b => decide (a = b))
  cast := fun _ _ _ x => x
  fromNat := fun _ n => some n
  fromInt := fun _ n => some n
  fp32Bits := fun _ => some 0
  fp32Load := id
  reduceMax := fun _ {_} ax keep v => ((Tile.reduceMax ax keep ⟨v⟩).getD ⟨fun _ => none⟩).data
  reduceSum := fun _ {_} ax keep v => (Tile.reduceSum ax keep ⟨v⟩).data
  reduceSumTyped := fun _ d {_} ax keep v =>
    some (fun i => ofFloat d ((Tile.reduceSum ax keep ⟨fun j => toFloat d (v j)⟩).data i))
  reduceMaxTyped := fun _ d {_} ax keep v =>
    some (fun i => ofFloat d (((Tile.reduceMax ax keep ⟨fun j => toFloat d (v j)⟩).getD ⟨fun _ => none⟩).data i))
  dot := fun _ {batch} {_} {_} {_} a b => some (Tile.dot batch ⟨a⟩ ⟨b⟩).data
  dotTyped := fun _ l r out _ {batch} {_} {_} {_} a b =>
    some (fun i => ofFloat out ((Tile.dot batch ⟨fun j => toFloat l (a j)⟩
      ⟨fun j => toFloat r (b j)⟩).data i))

def storageType : Storage → FloatDType
  | .fp16 => .fp16 | .bf16 => .bf16 | .fp32 => .real

def initial (input output : Storage) : State (WithBot ℝ) where
  mem := fun r o => if r="Q" then .mk (storageType input).toTileDType (ofFloat (storageType input) (some 0))
    else if r="K" then .mk .fp16 (ofFloat .fp16 (some 0))
    else if r="V" then .mk .fp16 (ofFloat .fp16 (some (if o=41 ∨ o=44 then 4 else 0)))
    else if r="O" then .mk (storageType output).toTileDType (ofFloat (storageType output) (some 99))
    else .mk .real (ofFloat .real (some 98))
  regs := fun _ _ _ => none
  pids := fun ax => if ax=1 then 1 else 0
  numPids := fun _ => 2
  undef := fun d _ _ => defaultValue d

@[simp] theorem masked_scatter_regs {ι : Type} (l : List ι) (s : State (WithBot ℝ))
    (region : ι → RegionName) (offset : ι → Nat) (value : ι → Cell (WithBot ℝ))
    (active : ι → Prop) [DecidablePred active] :
    (l.foldl (fun t i => if active i then t.write (region i) (offset i) (value i) else t) s).regs=s.regs := by
  induction l generalizing s with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons,ih]
    split <;> rfl

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.FlashAttn.NumericTypedExecution
