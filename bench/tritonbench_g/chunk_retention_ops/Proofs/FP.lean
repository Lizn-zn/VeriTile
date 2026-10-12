import bench.tritonbench_g.chunk_retention_ops.Kernels
import VeriTile.Triton.Float.RewriteTactics

namespace VeriTile.Bench.TritonBenchG.ChunkRetentionOps.FPProof
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec
set_option maxRecDepth 24000
set_option maxHeartbeats 12000000

def storageDType : Storage → FloatDType
  | .fp32 => .real | .fp16 => .fp16 | .bf16 => .bf16

def dotResult (storage : Storage) (M N K : Nat) (a b : RegName) : Op .real [M,N] :=
  match storage with
  | .fp32 => .dot (batch := []) (.ref .real [M,K] a) (.ref .real [K,N] b)
      {allowTF32 := some Bool.false}
  | .fp16 => .castFloat .fp32 .real (.dotTyped .fp16 .fp16 .fp32 (batch := [])
      (.ref .fp16 [M,K] a) (.ref .fp16 [K,N] b) {allowTF32 := some Bool.false})
  | .bf16 => .castFloat .fp32 .real (.dotTyped .bf16 .bf16 .fp32 (batch := [])
      (.ref .bf16 [M,K] a) (.ref .bf16 [K,N] b) {allowTF32 := some Bool.false})

def castLeftDot (storage : Storage) (M N K : Nat) (a b : RegName) : Op .real [M,N] :=
  match storage with
  | .fp32 => .dot (batch := []) (.castFloat .real .real (.ref .real [M,K] a))
      (.ref .real [K,N] b) {allowTF32 := some Bool.false}
  | .fp16 => .castFloat .fp32 .real (.dotTyped .fp16 .fp16 .fp32 (batch := [])
      (.castFloat .real .fp16 (.ref .real [M,K] a)) (.ref .fp16 [K,N] b)
      {allowTF32 := some Bool.false})
  | .bf16 => .castFloat .fp32 .real (.dotTyped .bf16 .bf16 .fp32 (batch := [])
      (.castFloat .real .bf16 (.ref .real [M,K] a)) (.ref .bf16 [K,N] b)
      {allowTF32 := some Bool.false})

def weighted (storage : Storage) (M N : Nat) (a weight : RegName) : Op .real [M,N] :=
  let v : Op .real [M,N] := match storage with
    | .fp32 => .ref .real [M,N] a
    | .fp16 => .castFloat .fp16 .real (.ref .fp16 [M,N] a)
    | .bf16 => .castFloat .bf16 .real (.ref .bf16 [M,N] a)
  .mul .real (.consSame (.consR .nil)) v
    (.expandDim ⟨1,by simp⟩ (.ref .real [M] weight))

def weightedRightDot (storage : Storage) (M N K : Nat) : Op .real [M,N] :=
  match storage with
  | .fp32 => .dot (batch := []) (.ref .real [M,K] "b_k")
      (.castFloat .real .real (weighted storage K N "b_v" "d_i")) {allowTF32 := some Bool.false}
  | .fp16 => .castFloat .fp32 .real (.dotTyped .fp16 .fp16 .fp32 (batch := [])
      (.ref .fp16 [M,K] "b_k") (.castFloat .real .fp16 (weighted storage K N "b_v" "d_i"))
      {allowTF32 := some Bool.false})
  | .bf16 => .castFloat .fp32 .real (.dotTyped .bf16 .bf16 .fp32 (batch := [])
      (.ref .bf16 [M,K] "b_k") (.castFloat .real .bf16 (weighted storage K N "b_v" "d_i"))
      {allowTF32 := some Bool.false})

def weightedGradientDot (storage : Storage) (M N K : Nat) : Op .real [M,N] :=
  match storage with
  | .fp32 => .dot (batch := []) (.ref .real [M,K] "b_q")
      (.castFloat .real .real (weighted storage K N "b_do" "d_i")) {allowTF32 := some Bool.false}
  | .fp16 => .castFloat .fp32 .real (.dotTyped .fp16 .fp16 .fp32 (batch := [])
      (.ref .fp16 [M,K] "b_q") (.castFloat .real .fp16 (weighted storage K N "b_do" "d_i"))
      {allowTF32 := some Bool.false})
  | .bf16 => .castFloat .fp32 .real (.dotTyped .bf16 .bf16 .fp32 (batch := [])
      (.ref .bf16 [M,K] "b_q") (.castFloat .real .bf16 (weighted storage K N "b_do" "d_i"))
      {allowTF32 := some Bool.false})

def forwardHRequirements (kernel : ComputeKernel) (storage : Storage) (BK BV BT : Nat) : List Site :=
  (do
    let loop ← Source.dynamicRange? kernel.surfaceBody "i_t"
    return Site.range loop.before "i_t" loop.start loop.stop loop.stride loop.body
      (.addFP32 (Source.prefixBefore loop.body "b_h") [BK,BV]
        (.mul .real .scalarL (.ref .real [] "d_b") (.ref .real [BK,BV] "b_h")) (weightedRightDot storage BK BV BT))).toList

def backwardHRequirements (kernel : ComputeKernel) (storage : Storage) (BK BV BT : Nat) : List Site :=
  (do
    let loop ← Source.dynamicRange? kernel.surfaceBody "j"
    return Site.range loop.before "j" loop.start loop.stop loop.stride loop.body
      (.addFP32 (Source.prefixBefore loop.body "b_dh") [BK,BV]
        (.mul .real .scalarL (.ref .real [] "d_b") (.ref .real [BK,BV] "b_dh"))
        (weightedGradientDot storage BK BV BT))).toList

def forwardORequirements (kernel : ComputeKernel) (storage : Storage) (BT BV : Nat) (scale : ℝ) : List Site :=
  [.mulScalar (Source.prefixBefore kernel.surfaceBody "b_o" 1) BT [BV]
    (.add .real (.consSame (.consSame .nil)) (.ref .real [BT,BV] "b_o")
      (castLeftDot storage BT BV BT "b_s" "b_v")) (.const scale)]

def backwardQKVRequirements (kernel : ComputeKernel) (storage : Storage) (BT BK : Nat) : List Site :=
  [.addFP32 (Source.prefixBefore kernel.surfaceBody "b_dq" 1) [BT,BK]
    (weighted .fp32 BT BK "b_dq" "d_q") (dotResult storage BT BK BT "b_ds" "b_k")]

namespace forwardH
section
variable (dotStorage : Storage) (dtype out_dtype : FloatDType) (k v h h0 ht : RegionName)
  (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat) (H T K V BT BK BV NT : Nat) (USE_INITIAL_STATE STORE_FINAL_STATE : Bool)
def original : Program where
  kernel := VeriTile.Bench.TritonBenchG.ChunkRetentionOps.forwardH dotStorage dtype out_dtype k v h h0 ht s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t H T K V BT BK BV NT USE_INITIAL_STATE STORE_FINAL_STATE
  domain := forwardHRequirements (VeriTile.Bench.TritonBenchG.ChunkRetentionOps.forwardH dotStorage dtype out_dtype k v h h0 ht s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t H T K V BT BK BV NT USE_INITIAL_STATE STORE_FINAL_STATE) dotStorage BK BV BT
def dotLeft : Program :=
  { original dotStorage dtype out_dtype k v h h0 ht s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t H T K V BT BK BV NT USE_INITIAL_STATE STORE_FINAL_STATE with kernel := VeriTile.Bench.TritonBenchG.ChunkRetentionOps.forwardHDotLeft dotStorage dtype out_dtype k v h h0 ht s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t H T K V BT BK BV NT USE_INITIAL_STATE STORE_FINAL_STATE }
theorem equivalent (R : FP.ScalarArithmetic.Rules) :
    original dotStorage dtype out_dtype k v h h0 ht s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t H T K V BT BK BV NT USE_INITIAL_STATE STORE_FINAL_STATE ≡[R] dotLeft dotStorage dtype out_dtype k v h h0 ht s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t H T K V BT BK BV NT USE_INITIAL_STATE STORE_FINAL_STATE := by
  cases dotStorage <;> equiv_decompose <;> fp_prove
end
#print_fp_assumptions equivalent
end forwardH

namespace forwardO
section
variable (dotStorage valueStorage : Storage) (out_dtype : FloatDType) (q k v h o : RegionName)
  (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat) (scale : ℝ) (H T K V BT BK BV : Nat)
def original : Program where
  kernel := VeriTile.Bench.TritonBenchG.ChunkRetentionOps.forwardO dotStorage valueStorage out_dtype q k v h o s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t scale H T K V BT BK BV
  domain := forwardORequirements (VeriTile.Bench.TritonBenchG.ChunkRetentionOps.forwardO dotStorage valueStorage out_dtype q k v h o s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t scale H T K V BT BK BV ) valueStorage BT BV scale
def scaleFirst : Program :=
  { original dotStorage valueStorage out_dtype q k v h o s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t scale H T K V BT BK BV  with kernel := VeriTile.Bench.TritonBenchG.ChunkRetentionOps.forwardOScaleFirst dotStorage valueStorage out_dtype q k v h o s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t scale H T K V BT BK BV  }
theorem equivalent (R : FP.ScalarArithmetic.Rules) :
    original dotStorage valueStorage out_dtype q k v h o s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t scale H T K V BT BK BV  ≡[R] scaleFirst dotStorage valueStorage out_dtype q k v h o s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t scale H T K V BT BK BV  := by
  cases dotStorage <;> cases valueStorage <;> equiv_decompose <;> fp_prove
end
#print_fp_assumptions equivalent
end forwardO

namespace backwardH
section
variable (dotStorage : Storage) (out_dtype : FloatDType) (q do_ dh : RegionName)
  (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat) (scale : ℝ) (H T K V BT BK BV NT : Nat)
def original : Program where
  kernel := VeriTile.Bench.TritonBenchG.ChunkRetentionOps.backwardH dotStorage out_dtype q do_ dh s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t scale H T K V BT BK BV NT
  domain := backwardHRequirements (VeriTile.Bench.TritonBenchG.ChunkRetentionOps.backwardH dotStorage out_dtype q do_ dh s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t scale H T K V BT BK BV NT ) dotStorage BK BV BT
def dotLeft : Program :=
  { original dotStorage out_dtype q do_ dh s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t scale H T K V BT BK BV NT  with kernel := VeriTile.Bench.TritonBenchG.ChunkRetentionOps.backwardHDotLeft dotStorage out_dtype q do_ dh s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t scale H T K V BT BK BV NT  }
theorem equivalent (R : FP.ScalarArithmetic.Rules) :
    original dotStorage out_dtype q do_ dh s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t scale H T K V BT BK BV NT  ≡[R] dotLeft dotStorage out_dtype q do_ dh s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t scale H T K V BT BK BV NT  := by
  cases dotStorage <;> equiv_decompose <;> fp_prove
end
#print_fp_assumptions equivalent
end backwardH

namespace backwardQKV
section
variable (dotStorage : Storage)  (q k v h do_ dh dq dk dv : RegionName)
  (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat) (scale : ℝ) (H T K V BT BK BV NT : Nat)
def original : Program where
  kernel := VeriTile.Bench.TritonBenchG.ChunkRetentionOps.backwardQKV dotStorage  q k v h do_ dh dq dk dv s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t scale H T K V BT BK BV NT
  domain := backwardQKVRequirements (VeriTile.Bench.TritonBenchG.ChunkRetentionOps.backwardQKV dotStorage  q k v h do_ dh dq dk dv s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t scale H T K V BT BK BV NT ) dotStorage BT BK
def dotLeft : Program :=
  { original dotStorage  q k v h do_ dh dq dk dv s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t scale H T K V BT BK BV NT  with kernel := VeriTile.Bench.TritonBenchG.ChunkRetentionOps.backwardQKVDotLeft dotStorage  q k v h do_ dh dq dk dv s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t scale H T K V BT BK BV NT  }
theorem equivalent (R : FP.ScalarArithmetic.Rules) :
    original dotStorage  q k v h do_ dh dq dk dv s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t scale H T K V BT BK BV NT  ≡[R] dotLeft dotStorage  q k v h do_ dh dq dk dv s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t scale H T K V BT BK BV NT  := by
  cases dotStorage <;> equiv_decompose <;> fp_prove
end
#print_fp_assumptions equivalent
end backwardQKV

end VeriTile.Bench.TritonBenchG.ChunkRetentionOps.FPProof

#guard_msgs (drop info) in
#auditModuleAxioms
