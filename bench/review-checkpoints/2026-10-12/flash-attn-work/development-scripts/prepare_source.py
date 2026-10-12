from pathlib import Path
import hashlib,itertools,json,sys
root=Path('/home/argustest/lizn/VeriTile');work=Path(__file__).resolve().parent;sys.path.insert(0,str(root));from bench.audit_source import lean_triton_bodies
legacy=root/'bench/tritonbench_g/flash_attn/FlashAttn.lean';body=lean_triton_bodies(legacy.read_text())[0]
assert body.count('qkv_base_offset = off_bs_head * $(stride_q_head)')==1
body=body.replace('qkv_base_offset = off_bs_head * $(stride_q_head)','$baseOffset')
body=body.replace('tl.store(l_ptr, max + tl.math.log2(denom))','tl.store(l_ptr, (max + tl.math.log2(denom)).to(tl.float32))')
body=body.replace('tl.store(O_block_ptr, (out_buffer).to(tl.float16))','tl.store(O_block_ptr, ((out_buffer).to(tl.float16)).to($outputTy:tritonDType))')
header='''import VeriTile.Triton.DSL
import VeriTile.Triton.Source
/-! Source: flash_attn.py, _fwd_kernel.
The full source and candidate differ only in the natural-number head-base
product. Q and O storage are independently fp16, bf16 or fp32. K and V are
fp16 because both dot left operands are explicitly converted to fp16. L is
fp32, as allocated by the host. Source stride choices, unchecked block loads
and stores, the literal 1.44269504 and denom * 0 + alpha are retained.
No host-layout correction or replacement with ordinary softmax is made. -/
namespace VeriTile.Bench.TritonBenchG.FlashAttn.TypedSource
open Lean VeriTile.Triton
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
inductive Storage where
  | fp16 | bf16 | fp32
  deriving DecidableEq
syntax "flashAttnSource%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(flashAttnSource%[$inputTy:tritonDType,$outputTy:tritonDType,$version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let baseOffset : TSyntax `tritonStmt ←
      if version.getId == `indexCommuted then
        `(tritonStmt| qkv_base_offset = $(stride_q_head) * off_bs_head)
      else `(tritonStmt| qkv_base_offset = off_bs_head * $(stride_q_head))
    `(fun (Q K V L O : RegionName) (sm_scale : ℝ)
      (stride_q_bs stride_q_head stride_q_seqlen stride_q_dim
        stride_k_bs stride_k_head stride_k_seqlen stride_k_dim
        stride_v_bs stride_v_head stride_v_seqlen stride_v_dim
        stride_o_bs stride_o_head stride_o_seqlen stride_o_dim
        BS HEAD SEQLEN BLOCK_M DIM BLOCK_N : Nat) (IS_CAUSAL : Bool) =>
      ComputeKernel.mk [Q,K,V] [L,O] (triton {
  tl.region Q = $inputTy, K = tl.float16, V = tl.float16, L = tl.float32, O = $outputTy
'''
s=header+body+'\n}).surfaceBody)\n\n'
types={'fp16':'tl.float16','bf16':'tl.bfloat16','fp32':'tl.float32'}
for a,b in itertools.product(types,repeat=2):
 s+=f'abbrev kernel_{a}_{b} (swapped : Bool) := if swapped then flashAttnSource%[{types[a]},{types[b]},indexCommuted] else flashAttnSource%[{types[a]},{types[b]},original]\n'
s+='\ndef kernel (input output : Storage) (swapped : Bool) := match input,output with\n'
for a,b in itertools.product(types,repeat=2):s+=f'  | .{a},.{b} => kernel_{a}_{b} swapped\n'
s+='end\nend VeriTile.Bench.TritonBenchG.FlashAttn.TypedSource\n'
(work/'FlashAttnDraft/Kernels.lean').write_text(s)
p=root/'bench/tritonbench_g/flash_attn/flash_attn.py'
record={'accepted':False,'repository_integrated':False,'status':'Typed source development; no compilation or proof acceptance yet','python_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(s.encode()).hexdigest(),'precision_reference':'https://github.com/triton-lang/triton/blob/v3.5.1/python/triton/language/semantic.py#L1328-L1340','findings':['All Q/K/V/O pointers use off_bs_head * stride_q_head as their common base.','V row stride uses stride_k_seqlen.','Both dot left operands convert to fp16; Triton dot requires matching input dtypes, so K/V are fp16.','All block loads and stores omit boundary_check: retain unchecked access.','The loop visits ceil(hi/BLOCK_N) full key tiles, even if hi is not divisible by BLOCK_N. Noncausal padded keys are included. Causal keys are filtered by the source query-row comparison.','Source scaling uses the exact decimal 1.44269504; do not replace it by exact log2(e).','Both L and O outputs and their frame require proof; source out_scale=denom*0+alpha remains in FP.'],'pending':['Complete Python/Lean AST comparison and typed elaboration','Independent source-specific real output/LSE formulas with symbolic strides and both causal modes','Whole typed state/failure equivalence and nonempty execution witnesses','Repository integration and independent comparator/trust replay']}
(work/'review.json').write_text(json.dumps(record,indent=2)+'\n')
print('Prepared full flash_attn typed source in persistent staging; not accepted')
