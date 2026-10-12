from pathlib import Path
from fractions import Fraction
import ast,hashlib,json,re,sys,textwrap
root=Path('/home/argustest/lizn/VeriTile');sys.path.insert(0,str(root))
from bench.audit_source import lean_triton_bodies,python_named_jit_body,strip_region_declarations
lp=Path(sys.argv[1]) if len(sys.argv)>1 else Path('/tmp/FifthSphDraft/Kernels.lean');pp=root/'bench/tritonbench_g/fifth_order_sph_harmonics/fifth_order_sph_harmonics.py';lean=lp.read_text();python=pp.read_text()
bs=lean_triton_bodies(lean);assert len(bs)==2

def clean(s):
 s=re.sub(r'\$\(\(([^()]+?)\s*:\s*(?:ℝ|Nat)\)\)',r'\1',s)
 s=re.sub(r'\$\(([^()]+)\)',r'\1',s)
 for name in ['inputTy','outTy','gradTy','upstreamTy']:s=s.replace('$'+name+':tritonDType',name.upper()).replace('$'+name,name.upper())
 return s
checks=[]
for i,(body,name,storage) in enumerate(zip(bs,['fifth_order_fwd','fifth_order_bwd'],['OUTTY','GRADTY'])):
 py=textwrap.dedent(python_named_jit_body(python,name));pst=ast.parse(py)
 lst=ast.parse('\n'.join(line.strip() for line in clean(strip_region_declarations(body)).splitlines()))
 pc=[n for n in ast.walk(pst) if isinstance(n,ast.Call) and ast.unparse(n.func)=='tl.store']
 lc=[n for n in ast.walk(lst) if isinstance(n,ast.Call) and ast.unparse(n.func)=='tl.store']
 assert len(pc)==len(lc)==(11 if i==0 else 3)
 for p,l in zip(pc,lc):
  assert ast.dump(p.args[0])==ast.dump(l.args[0]),(name,'changed store pointer')
  cast=l.args[1];assert isinstance(cast,ast.Call) and isinstance(cast.func,ast.Attribute) and cast.func.attr=='to'
  assert ast.dump(cast.func.value)==ast.dump(p.args[1]),(name,'changed store value')
  assert len(cast.args)==1 and isinstance(cast.args[0],ast.Name) and cast.args[0].id==storage and not cast.keywords,(name,'changed store precision')
  p.args[1]=cast
 assert ast.dump(pst)==ast.dump(lst),(name,ast.dump(pst),ast.dump(lst))
 pyconstants={n.targets[0].id:Fraction(ast.get_source_segment(py,n.value)) for n in pst.body if isinstance(n,ast.Assign) and n.targets[0].id.startswith('CONST')}
 checks.append(dict(kernel=name,complete_statement_ast=True,all_load_masks=True,all_store_values_and_destination_casts=True,ordinary_fp32_constant_assignments=True,constants=len(pyconstants)))
r=dict(lean_sha256=hashlib.sha256(lp.read_bytes()).hexdigest(),python_sha256=hashlib.sha256(pp.read_bytes()).hexdigest(),checks=checks,normalizations=['Lean syntax and Nat/real antiquotation','Explicit destination cast checked independently for every store'])
Path('/tmp/fifth-sph-source-check.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
