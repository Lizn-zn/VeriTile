from pathlib import Path
import ast,itertools,re
root=Path('/home/argustest/lizn/VeriTile');text=(root/'bench/tritonbench_g/fifth_order_sph_harmonics/fifth_order_sph_harmonics.py').read_text();tree=ast.parse(text)
fns=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['fifth_order_fwd','fifth_order_bwd']]
header='''import VeriTile.Triton.DSL
import VeriTile.Triton.Source
/-! Source: fifth_order_sph_harmonics.py, fifth_order_fwd.
Both fifth_order_fwd and fifth_order_bwd are transcribed completely.
Named decimal coefficient assignments become fp32 scalar tensors, following
ordinary Triton assignment semantics. They promote polynomial arithmetic to
fp32; powers formed before the coefficient multiplication retain input type.
Negative decimal literals are staged as single constants; negating a named
tensor coefficient remains an arithmetic operation.
The candidate only commutes the natural-number coordinate block product.
All implicit destination conversions are explicit. Missing load padding stays
unspecified; it is not translated as zero. Backward adds to the initial dX.
-/
namespace VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.TypedSource
open Lean VeriTile.Triton
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000
inductive Storage where
  | fp16 | bf16 | fp32
  deriving DecidableEq

'''
body=header
for fn in fns:
 forward=fn.name.endswith('fwd');macro='fifthFwd' if forward else 'fifthBwd';name='forward' if forward else 'backward';pointers=['coord_ptr','output_ptr'] if forward else ['coord_ptr','coord_grad_ptr','sph_grad_ptr'];params=[a.arg for a in fn.args.args if a.arg not in pointers];constants={n.targets[0].id:ast.get_source_segment(text,n.value) for n in fn.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id.startswith('CONST')}
 def pointer(n):
  result=ast.unparse(n)
  for p in params:result=re.sub(r'\b'+re.escape(p)+r'\b',lambda _: '$('+p+')',result)
  result=re.sub(r'\b[0-9]+\b',lambda m:'$('+m[0]+')',result)
  return result
 def expr(n):
  if isinstance(n,ast.Name):
   return '$('+n.id+')' if n.id in params else n.id
  if isinstance(n,ast.Constant):
   if isinstance(n.value,int):return '$('+str(n.value)+')'
   return ast.get_source_segment(text,n)
  if isinstance(n,ast.UnaryOp) and isinstance(n.op,ast.USub):
   if isinstance(n.operand,ast.Constant) and isinstance(n.operand.value,float):return '$(('+ast.get_source_segment(text,n)+' : ℝ))'
   return '(-'+expr(n.operand)+')'
  if isinstance(n,ast.BinOp):return '('+expr(n.left)+' '+{ast.Add:'+',ast.Sub:'-',ast.Mult:'*'}[type(n.op)]+' '+expr(n.right)+')'
  if isinstance(n,ast.Compare):assert len(n.ops)==1;return expr(n.left)+' < '+expr(n.comparators[0])
  if isinstance(n,ast.Call) and ast.unparse(n.func)=='tl.load':return 'tl.load('+', '.join([pointer(n.args[0])]+[expr(a) for a in n.args[1:]]+[k.arg+'='+expr(k.value) for k in n.keywords])+')'
  if isinstance(n,ast.Call):return expr(n.func)+'('+', '.join([expr(a) for a in n.args]+[k.arg+'='+expr(k.value) for k in n.keywords])+')'
  if isinstance(n,ast.Attribute):return expr(n.value)+'.'+n.attr
  raise ValueError(ast.dump(n))
 lines=[]
 for n in fn.body:
  if isinstance(n,ast.Assign):
   lhs=n.targets[0].id
   if lhs=='coord_row_offset':lines.append('  $coordinateOffset');continue
   val=expr(n.value)
   lines.append('  '+lhs+' = '+val)
  elif isinstance(n,ast.AugAssign):lines.append('  '+n.target.id+' += '+expr(n.value))
  elif isinstance(n,ast.Expr) and isinstance(n.value,ast.Call):
   call=n.value;assert expr(call.func)=='tl.store';dstType='$outTy' if forward else '$gradTy';val='('+expr(call.args[1])+').to('+dstType+':tritonDType)';lines.append('  tl.store('+pointer(call.args[0])+', '+val+', '+', '.join(k.arg+'='+expr(k.value) for k in call.keywords)+')')
  else:raise ValueError(ast.dump(n))
 types=['inputTy','outTy'] if forward else ['inputTy','gradTy','upstreamTy']
 body+='syntax "'+macro+'%" "[" '+' "," '.join('tritonDType' for t in types)+' "," ident "]" : term\nset_option hygiene false in\nmacro_rules\n'
 body+='  | `('+macro+'%['+','.join('$'+t+':tritonDType' for t in types)+',$version:ident]) => do\n'
 body+='''    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let coordinateOffset : TSyntax `tritonStmt ←
      if version.getId == `indexCommuted then
        `(tritonStmt| coord_row_offset = coord_striding + block_id * ($(block_size) * coord_stride))
      else
        `(tritonStmt| coord_row_offset = coord_striding + ($(block_size) * coord_stride) * block_id)
'''
 body+='    `(fun ('+' '.join(pointers)+' : RegionName) ('+' '.join(params)+' : Nat) =>\n'
 body+='      ComputeKernel.mk ['+','.join(pointers if forward else ['coord_ptr','coord_grad_ptr','sph_grad_ptr'])+'] ['+('output_ptr' if forward else 'coord_grad_ptr')+'] (triton {\n'
 body+='  tl.region '+', '.join(p+' = $'+t for p,t in zip(pointers,types))+'\n'+'\n'.join(lines)+'\n}).surfaceBody)\n\n'
 storages=['fp16','bf16','fp32'];dtypes={'fp16':'tl.float16','bf16':'tl.bfloat16','fp32':'tl.float32'}
 labels=['input','output'] if forward else ['input','gradient','upstream']
 for combo in itertools.product(storages,repeat=len(types)):
  ts=','.join(dtypes[x] for x in combo);variant=name+'_'+'_'.join(combo)
  body+='abbrev '+variant+' (swapped : Bool) := if swapped then '+macro+'%['+ts+',indexCommuted] else '+macro+'%['+ts+',original]\n'
 body+='\ndef '+name+' ('+' '.join(labels)+' : Storage) (swapped : Bool) := match '+','.join(labels)+' with\n'
 for combo in itertools.product(storages,repeat=len(types)):
  variant=name+'_'+'_'.join(combo);body+='  | '+','.join('.'+x for x in combo)+' => '+variant+' swapped\n'
 body+='\n'
body+='end\nend VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.TypedSource\n';Path('/tmp/FifthSphDraft/Kernels.lean').write_text(body)
