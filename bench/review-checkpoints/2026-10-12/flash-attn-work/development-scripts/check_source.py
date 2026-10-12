from pathlib import Path
import ast,hashlib,json,re,sys,textwrap
root=Path('/home/argustest/lizn/VeriTile');work=Path(__file__).resolve().parent;sys.path.insert(0,str(root))
from bench.audit_source import lean_triton_bodies,python_named_jit_body,strip_region_declarations
lp=work/'FlashAttnDraft/Kernels.lean';pp=root/'bench/tritonbench_g/flash_attn/flash_attn.py';text=lp.read_text();bodies=lean_triton_bodies(text);assert len(bodies)==1
s=strip_region_declarations(bodies[0]);s=re.sub(r'\$\(\(([^()]+?)\s*:\s*(?:ℝ|Nat|Int)\)\)',r'\1',s);s=re.sub(r'\$\(([^()]+)\)',r'\1',s);s=s.replace('$outputTy:tritonDType','OUTPUT')
lines=[];indent=0
for line in s.splitlines():
 line=line.strip()
 if not line:continue
 if line=='}':indent-=1;continue
 opens=line.endswith('{')
 if opens:line=line[:-1]+':'
 lines.append('    '*indent+line)
 if opens:indent+=1
assert indent==0
lean=ast.parse(textwrap.dedent('\n'.join(lines)));python=ast.parse(textwrap.dedent(python_named_jit_body(pp.read_text(),'_fwd_kernel')))
for node in ast.walk(lean):
 if isinstance(node,ast.Call) and ast.unparse(node.func)=='tl.advance':
  assert isinstance(node.args[1],ast.List) and len(node.args[1].elts)==2
  node.args[1]=ast.Tuple(elts=node.args[1].elts,ctx=ast.Load())
ps=[n for n in ast.walk(python) if isinstance(n,ast.Call) and ast.unparse(n.func)=='tl.store'];ls=[n for n in ast.walk(lean) if isinstance(n,ast.Call) and ast.unparse(n.func)=='tl.store'];assert len(ps)==len(ls)==2
for p,l,dtype in zip(ps,ls,['tl.float32','OUTPUT']):
 assert ast.dump(p.args[0])==ast.dump(l.args[0]),'store address'
 cast=l.args[1];assert isinstance(cast,ast.Call) and isinstance(cast.func,ast.Attribute) and cast.func.attr=='to'
 assert ast.dump(cast.func.value)==ast.dump(p.args[1]),'store value'
 assert len(cast.args)==1 and ast.unparse(cast.args[0])==dtype and not cast.keywords,'destination dtype'
 p.args[1]=cast
assert ast.dump(python)==ast.dump(lean), (ast.dump(python),ast.dump(lean))
result={'python_sha256':hashlib.sha256(pp.read_bytes()).hexdigest(),'lean_sha256':hashlib.sha256(lp.read_bytes()).hexdigest(),'complete_statement_ast':True,'loads':3,'stores':2,'block_pointer_constructions':4,'both_causal_branches':True,'normalizations':['Lean Nat/real/type antiquotations and braces','Two fixed 2D tl.advance delta lists to Python tuples','Independent validation of both explicit destination casts']}
(work/'source-comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
