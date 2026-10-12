import bench.tritonbench_g.chunk_cumsum_kernel.Kernels
import VeriTile.Meta.StatementAudit

/-! Regressions for the complete typed source: explicit fp32 load conversion,
scan input, fp32 block total and the independent output storage conversion. -/
namespace FPChunkCumsumSourceTests
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkCumsumKernel

def loop (dtype out_dtype : FloatDType) (T BT : Nat) : Source.DynamicRange :=
  (Source.dynamicRange? (TypedSource.original dtype out_dtype "s" "o" T BT).surfaceBody
    "i_t").getD ⟨[], .constNat 0, .constNat 0, .constNat 0, [], []⟩

theorem load_explicitly_promotes (dtype out_dtype : FloatDType) (T BT : Nat) :
    (loop dtype out_dtype T BT).body[2]? = some (.assign .real [BT] "b_s"
      (.compute (.alg .fp32 (.castFloat dtype .real
        (.load dtype.toTileDType (.blockPtr (.ref .blockPtr [BT] "p_s") [0]) .none .unspecified))))) := rfl

theorem scan_keeps_input_axis_and_direction (dtype out_dtype : FloatDType) (T BT : Nat) :
    (loop dtype out_dtype T BT).body[3]? = some (.assign .real [BT] "b_o"
      (.alg (.add .real Broadcast.nil.consR
        (.scan .sum ⟨0, by simp⟩ .forward (.ref .real [BT] "b_s"))
        (.expandDim ⟨0, by simp⟩ (.ref .real [] "b_z"))))) := rfl

theorem total_is_fp32 (dtype out_dtype : FloatDType) (T BT : Nat) :
    (loop dtype out_dtype T BT).body[4]? = some (.assign .real [] "b_zz"
      (.compute (.alg .fp32 (.reduceSum ⟨0, by simp⟩ Bool.false (.ref .real [BT] "b_s"))))) := rfl

theorem output_cast_is_retained (dtype out_dtype : FloatDType) (T BT : Nat) :
    (loop dtype out_dtype T BT).body[6]? = some (.store out_dtype.toTileDType [BT]
      (.blockPtr (.ref .blockPtr [BT] "p_o") [0])
      (.alg (.castFloat .real out_dtype (.ref .real [BT] "b_o"))) .none) := rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end FPChunkCumsumSourceTests
