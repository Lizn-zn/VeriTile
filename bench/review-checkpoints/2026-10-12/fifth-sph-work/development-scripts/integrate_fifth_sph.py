from pathlib import Path
import ast,collections,csv,hashlib,json,re,subprocess,sys
root=Path('/home/argustest/lizn/VeriTile');work=Path(__file__).resolve().parent;plan=json.loads((work/'migration.json').read_text());out=Path(plan['prepared']);assert not plan['repository_integrated']
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for f,h in plan['files'].items():assert digest(out/f)==h,f
for f,h in plan['oldfiles'].items():
 assert digest(root/f)==h,f
 hits=subprocess.run(['rg','-l','^import '+re.escape(f[:-5].replace('/','.'))+'$','--glob','*.lean','bench','VeriTile'],cwd=root,capture_output=True,text=True);assert hits.returncode==1,(hits.stdout,hits.stderr)
for f in plan['files']:
 p=root/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((out/f).read_bytes())
for f in plan['oldfiles']:(root/f).unlink()
p=root/'lakefile.toml';s=p.read_text();needle='"bench.tritonbench_g.fast_rope_embedding.+"';assert s.count(needle)==1;s=s.replace(needle,needle+', "bench.tritonbench_g.fifth_order_sph_harmonics.+"');p.write_text(s)
case='fifth_order_sph_harmonics';base=root/'bench/tritonbench_g'/case;py=base/(case+'.py');pynames={n.name:n.lineno for n in ast.parse(py.read_text()).body if isinstance(n,ast.FunctionDef)}
common='Complete original/candidate JIT with symbolic dimensions/strides, independently expanded decimal-coefficient real polynomials, successful finite flat execution, actual bounds for all loads/stores, all outputs and the complete tagged-cell frame. Finite disjoint allocation, coverage and zero undefined values are explicit. '
precision='All nine coordinate/output fp16-bf16-fp32 combinations forward and all 27 coordinate/gradient/upstream combinations backward. Named decimal assignments materialize fp32 scalar tensors, pre-coefficient powers retain input precision, coefficient arithmetic promotes to fp32, and destination conversions are explicit. Unspecified masked padding is preserved. '
limits=' Per-program mathematical/typed model; no host scheduling/composition, race, fixed-width-overflow or physical IEEE-rounding claim.'
scopes={'forward':'Eleven expanded fifth-degree coordinate polynomials. Output stride is greater than ten, with complete xyz input rows for active output rows. Every active channel and all unwritten cells are specified.', 'backward':'Three initial-gradient-plus-weighted-update polynomials with an independent 3-by-11 coefficient table. Active coordinate rows require complete xyz and eleven upstream channels. All loads precede stores, so initial values are used even under named-region aliasing. This rounded decimal table is not the exact derivative of the forward table: a checked channel-2 x-derivative example differs by 1e-14.'}
p=root/'bench/tritonbench_g/coverage_review.json';review=json.loads(p.read_text());review['ports']=[r for r in review['ports'] if r['file'] not in plan['oldfiles']]
sys.path.insert(0,str(root/'bench'));from port_layout import proof_files
for entry in ['Correct','FPEquiv']:
 lean=base/(entry+'.lean');decls=[]
 for name in re.findall(r'^specification\s+(\w+)',lean.read_text(),re.M):
  kind=name.split('_')[0];assert kind in scopes
  scope=common+precision+scopes[kind]+limits if entry=='Correct' else 'Whole typed-state/failure equivalence; empty input domain and no numerical FP atoms. Only the natural-number coordinate-block product is commuted. All source assignments, casts, masks, loads, old-gradient additions and stores are retained. '+precision+limits
  fn='fifth_order_'+('fwd' if kind=='forward' else 'bwd')
  decls.append(dict(declaration=name,coverage_level='full_value_candidate',numeric_model='mathematical' if entry=='Correct' else 'admitted_fp',python_functions=[dict(name=fn,line=pynames[fn])],scope=scope))
 review['ports'].append(dict(file=str(lean.relative_to(root)),python_file=str(py.relative_to(root)),lean_sha256=digest(lean),python_sha256=digest(py),support_sha256={str(p.relative_to(root)):digest(p) for p in proof_files(base) if p!=lean},declarations=decls))
review['ports'].sort(key=lambda r:r['file']);p.write_text(json.dumps(review,indent=2,ensure_ascii=False)+'\n')
subprocess.run(['python3','bench/check_proof_gap_manifest.py','--write'],cwd=root,check=True)
rows=list(csv.DictReader((root/'bench/tritonbench_g/proof_gap_manifest.tsv').open(),delimiter='\t'));counts=collections.Counter(r['coverage_level'] for r in rows);files=len(set(r['file'] for r in rows))
p=root/'bench/tritonbench_g/completion_audit.md';s=p.read_text();s=re.sub(r'\d+ headline \(`specification`\) declarations in \d+ specification files',f'{len(rows)} headline (`specification`) declarations in {files} specification files',s,count=1)
for level,count in counts.items():s=re.sub(r'(\| `'+level+r'` \| )\d+( \|)',lambda m:m[1]+str(count)+m[2],s,count=1)
s=re.sub(r'\| \*\*Total\*\* \| \*\*\d+\*\* \| \*\*Across \d+ specification files in 173 ports\.\*\* \|',f'| **Total** | **{len(rows)}** | **Across {files} specification files in 173 ports.** |',s,count=1)
s+='''\n\n## Fifth-order spherical harmonics source migration

The complete forward/backward sources and integer-index candidates expose four
independent finite-flat real contracts and two whole typed FP equivalences.
The real tables retain the source decimal coefficients, including the checked
mismatch between backward weights and exact derivatives of the forward table.
All source storage choices, casts, masks and initial-gradient accumulation are
retained. Typed and finite real execution witnesses accompany the public
contracts. Acceptance awaits repository gates and independent trust replay;
source migration and scope classification do not change the accepted count.
''';p.write_text(s)
for f in plan['oldfiles']:
 p=Path(f);sheet=root/'bench/tritonbench_g/_spec-sheets'/(p.parent.name+'__'+p.stem+'.md')
 if sheet.exists():sheet.unlink()
subprocess.run(['python3','scripts/spec_sheet.py'],cwd=root,check=True)
p=root/'bench/tritonbench_g/verification_round.md';s=p.read_text();needle='| `fifth_order_sph_harmonics` | pending review |';assert needle in s;s=s.replace(needle,'| `fifth_order_sph_harmonics` | complete source/real/FP development and execution witnesses integrated; repository gates and independent replay pending |',1);p.write_text(s)
plan.update(repository_integrated=True,accepted=False,root_frozen=True,manifest_counts=dict(counts),headlines=len(rows),specification_files=files)
(work/'migration.json').write_text(json.dumps(plan,indent=2)+'\n')
print('Integrated fifth_order_sph_harmonics for repository verification; acceptance remains 59 / 184')
