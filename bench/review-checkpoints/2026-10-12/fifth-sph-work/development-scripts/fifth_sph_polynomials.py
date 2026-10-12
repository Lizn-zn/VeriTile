from pathlib import Path
import ast,collections,json
from fractions import Fraction
src=Path('/home/argustest/lizn/VeriTile/bench/tritonbench_g/fifth_order_sph_harmonics/fifth_order_sph_harmonics.py').read_text();tree=ast.parse(src)
# Exponents: x,y,z and the eleven incoming spherical-gradient channels.
Z=(0,)*14
const=lambda q:{Z:Fraction(q)}
def var(k):
 t=list(Z);t[k]=1;return {tuple(t):Fraction(1)}
def add(a,b):
 out=collections.defaultdict(Fraction,a)
 for k,v in b.items():out[k]+=v
 return {k:v for k,v in out.items() if v}
def neg(a):return {k:-v for k,v in a.items()}
def mul(a,b):
 out=collections.defaultdict(Fraction)
 for ka,va in a.items():
  for kb,vb in b.items():out[tuple(x+y for x,y in zip(ka,kb))]+=va*vb
 return {k:v for k,v in out.items() if v}
def ev(n,env):
 if isinstance(n,ast.Name):return env[n.id]
 if isinstance(n,ast.Constant):return const(ast.get_source_segment(src,n))
 if isinstance(n,ast.UnaryOp):assert isinstance(n.op,ast.USub);return neg(ev(n.operand,env))
 if isinstance(n,ast.BinOp):
  a,b=ev(n.left,env),ev(n.right,env)
  if isinstance(n.op,ast.Add):return add(a,b)
  if isinstance(n.op,ast.Sub):return add(a,neg(b))
  if isinstance(n.op,ast.Mult):return mul(a,b)
 raise ValueError(ast.dump(n))
def render(poly,lean=False):
 terms=[]
 for exps,c in sorted(poly.items(),reverse=True):
  coeff=f'({c.numerator} / {c.denominator} : ℝ)' if lean else str(c)
  mon=[f'{v}^{e}' if e>1 else v for v,e in zip(['x','y','z']+[f'g {i}' for i in range(11)],exps) if e]
  terms.append(coeff+(' * '+' * '.join(mon) if mon else ''))
 return ' + '.join(terms) if terms else '0'
envs={}
for fn in [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['fifth_order_fwd','fifth_order_bwd']]:
 env={name:var(i) for i,name in enumerate(['x','y','z']+[f'g_{j}' for j in range(11)])}
 for st in fn.body:
  if isinstance(st,ast.Assign) and st.targets[0].id.startswith(('CONST','VAR','Y')):env[st.targets[0].id]=ev(st.value,env)
  if isinstance(st,ast.AugAssign):env[st.target.id]=ev(st.value,env)
 envs[fn.name]=env
forward=[envs['fifth_order_fwd'][f'Y{i:02}'] for i in range(11)];backward=[envs['fifth_order_bwd'][f'g_{s}'] for s in 'xyz']
differences=[]
for a in range(3):
 for i in range(11):
  actual={tuple([*k[:3],*[0]*11]):v for k,v in backward[a].items() if k[3+i]==1}
  derivative={tuple(e-1 if d==a else e for d,e in enumerate(k)):v*k[a] for k,v in forward[i].items() if k[a]}
  diff=add(actual,neg(derivative))
  if diff:differences.append(dict(axis='xyz'[a],channel=i,difference=render(diff)))
p=Path('/tmp/FifthSphDraft/polynomial-review.json');p.write_text(json.dumps(dict(coefficient_model='Exact source decimal rationals, independent of physical FP rounding',forward=[render(x) for x in forward],backward_update=[render(x) for x in backward],backward_minus_derivative=differences),indent=2)+'\n')
print('forward channels',len(forward),'backward components',len(backward),'coefficient mismatches',len(differences))
print(json.dumps(differences,indent=2))
# The specification is a polynomial coefficient table, independent of execution.
lean='''import VeriTile.Triton
import Mathlib.Analysis.Calculus.Deriv.Pow
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.RealMath
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 1000000

def output (c : Fin 11) (x y z : ℝ) : ℝ := match c.val with
'''
for i,poly in enumerate(forward):lean+='  | '+str(i)+' => '+render(poly,True)+'\n'
lean+='  | _ => 0\n\n'
lean+='''/-- Decimal coefficients of the source backward update. They are not
asserted to be the exact derivatives of the source's rounded forward table. -/
def backwardWeight (component : Fin 3) (c : Fin 11) (x y z : ℝ) : ℝ := match component.val,c.val with
'''
for a in range(3):
 for i in range(11):
  poly={tuple([*k[:3],*[0]*11]):v for k,v in backward[a].items() if k[3+i]==1}
  if poly:lean+='  | '+str(a)+','+str(i)+' => '+render(poly,True)+'\n'
lean+='''  | _,_ => 0

def gradient (component : Fin 3) (x y z old : ℝ) (g : Fin 11 → ℝ) : ℝ :=
  old+∑ c,g c*backwardWeight component c x y z

/-- At (1,0,0), the x derivative of channel 2 differs from its backward
coefficient by 10^-14 in the exact-decimal real model. This is a coefficient
identity, not a claim about the magnitude of physical floating-point error. -/
theorem channel2_derivative :
    HasDerivAt (fun x : ℝ => output 2 x 0 0) (867152307844475/100000000000000 : ℝ) 1 := by
  have h : (fun x : ℝ => output 2 x 0 0)=(fun x : ℝ => (34686092313779/20000000000000 : ℝ)*x^5) := by
    funext x
    norm_num [output]
  rw [h]
  convert ((hasDerivAt_id (1 : ℝ)).pow 5).const_mul (34686092313779/20000000000000 : ℝ) using 1 <;> norm_num

theorem backward_channel2_coefficient_differs :
    backwardWeight 0 2 1 0 0-(867152307844475/100000000000000 : ℝ)=1/100000000000000 := by
  norm_num [backwardWeight]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.RealMath
'''
Path('/tmp/FifthSphDraft/RealMath.lean').write_text(lean)
