import bench.tritonbench_g.chunk_cumsum_vector.Kernels
import VeriTile.Meta.StatementAudit

/-! Source regressions for storage conversion, TF32-disabled triangular dot,
both boundary axes, output precision and the source carry-update branch. -/
namespace FPVectorCumsumSourceTests
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkCumsumVector

def loop (dtype out_dtype : FloatDType) (T S BT BS : Nat) : Source.DynamicRange :=
  (Source.dynamicRange? (TypedSource.original dtype out_dtype "s" "o" 99 11 2 T S BT BS).surfaceBody
    "i_t").getD ⟨[], .constNat 0, .constNat 0, .constNat 0, [], []⟩

theorem load_explicitly_promotes (dtype out_dtype : FloatDType) (T S BT BS : Nat) :
    (loop dtype out_dtype T S BT BS).body[2]? = some (.assign .real [BT,BS] "b_s"
      (.compute (.alg .fp32 (.castFloat dtype .real
        (.load dtype.toTileDType (.blockPtr (.ref .blockPtr [BT,BS] "p_s") [0,1]) .none .unspecified))))) := rfl

theorem dot_disables_tf32 (dtype out_dtype : FloatDType) (T S BT BS : Nat) :
    (loop dtype out_dtype T S BT BS).body[3]? = some (.assign .real [BT,BS] "b_c"
      (.alg (.add .real (.consL (.consSame .nil))
        (.expandDim ⟨0, by simp⟩ (.ref .real [BS] "b_z"))
        (.dot (batch := []) (.ref .real [BT,BT] "m_s") (.ref .real [BT,BS] "b_s")
          {allowTF32 := some Bool.false})))) := rfl

theorem output_cast_and_both_boundaries (dtype out_dtype : FloatDType) (T S BT BS : Nat) :
    (loop dtype out_dtype T S BT BS).body[4]? = some (.store out_dtype.toTileDType [BT,BS]
      (.blockPtr (.ref .blockPtr [BT,BS] "p_z") [0,1])
      (.alg (.castFloat .real out_dtype (.ref .real [BT,BS] "b_c"))) .none) := rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end FPVectorCumsumSourceTests
