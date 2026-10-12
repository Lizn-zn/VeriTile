from pathlib import Path
import hashlib,json,re
root=Path('/home/argustest/lizn/VeriTile');work=Path(__file__).resolve().parent;src=work/'FifthSphDraft';out=work/'final-files';case='fifth_order_sph_harmonics'
assert 'Current accepted count is **59 complete, 125 pending**' in (root/'bench/tritonbench_g/verification_round.md').read_text()
verified=json.loads((work/'validation/complete.json').read_text());assert verified['modules']==24
for row in verified['results']:
 assert row['code']==0,row
 assert hashlib.sha256((src/(row['module']+'.lean')).read_bytes()).hexdigest()==row['source_sha256'],row['module']
check=json.loads((work/'source-comparison.json').read_text());assert check['lean_sha256']==hashlib.sha256((src/'Kernels.lean').read_bytes()).hexdigest()
assert check['python_sha256']==hashlib.sha256((root/'bench/tritonbench_g/fifth_order_sph_harmonics/fifth_order_sph_harmonics.py').read_bytes()).hexdigest()
names=['Kernels','Correct','FPEquiv','FP','Erasure','Candidate','RealMath','NormalizedSource','ForwardReal','ForwardIO','BackwardSupport','BackwardBase','BackwardReadbackCommon','BackwardReadback0','BackwardReadback1','BackwardReadback2','BackwardReadback','BackwardIO','TypedModel','TypedForwardExecution','TypedBackwardExecution','TypedExecution','Execution','SourceChecks']
files={}
for name in names:
 dest=(f'bench/tests/FifthOrderSphHarmonicsExecution.lean' if name=='Execution' else f'bench/tests/FPFifthOrderSphHarmonicsSource.lean' if name=='SourceChecks' else f'bench/tritonbench_g/{case}/'+('' if name in ['Kernels','Correct','FPEquiv'] else 'Proofs/')+name+'.lean')
 files[str(src/(name+'.lean'))]=dest
mapping={'FifthSphDraft.'+Path(s).stem:d[:-5].replace('/','.') for s,d in files.items()};hashes={}
for path,dest in files.items():
 s=Path(path).read_text()
 # Development progress reporting is not part of the proof and is omitted.
 s=re.sub(r'^  run_tac do IO.FS.writeFile[^\n]+\n','',s,flags=re.M)
 for a,b in sorted(mapping.items(),key=lambda p:-len(p[0])):s=re.sub(r'\b'+re.escape(a)+r'\b',lambda _:b,s)
 assert 'FifthSphDraft' not in s and '/tmp/' not in s,path
 target=out/dest;target.parent.mkdir(parents=True,exist_ok=True);assert not (root/dest).exists(),dest;target.write_text(s);hashes[dest]=hashlib.sha256(target.read_bytes()).hexdigest()
dest=f'bench/tritonbench_g/{case}/README.md';s=(src/'README.md').read_text();s=s.replace('Status: development in a separate staging directory; repository acceptance is\npending. No GPU experiment has been run for this review.','Status: complete source and proof development is integrated; repository build,\nsource/layout/test gates and independent comparator/trust replay remain pending.\nNo GPU experiment has been run for this review.')
s=s.replace('Polynomial readback and complete finite witnesses,\nall-storage typed execution witnesses, repository integration/gates and fresh\nindependent comparator/trust replay remain pending.','All three polynomial readbacks, public finite execution witnesses and all\n9+27 typed execution choices pass local checks. Repository gates and fresh\nindependent comparator/trust replay remain pending.')
s=s.replace('The staged finite execution regression connects this coefficient fact\nto the complete backward kernel once its checks pass.','The finite execution regression connects this coefficient fact to the complete\nbackward kernel.')
s=s.split('\n\nThe development sources were recovered')[0]+'\n';p=out/dest;p.write_text(s);hashes[dest]=hashlib.sha256(p.read_bytes()).hexdigest()
legacy=f'bench/tritonbench_g/{case}/FifthOrderSphHarmonics.lean';p=root/legacy;backup=work/'legacy-before-migration.lean';backup.write_bytes(p.read_bytes())
plan=dict(prepared=str(out),files=hashes,oldfiles={legacy:hashlib.sha256(p.read_bytes()).hexdigest()},mapping=files,accepted=False,repository_integrated=False,root_frozen=False,pending=['Fresh repository source/proof build and finite/type regressions','Source/layout/audit gates and independent comparator/trust replay'],source7_note='Fresh root and independent-snapshot compilation are mandatory after staging formatting changes.')
(work/'migration.json').write_text(json.dumps(plan,indent=2)+'\n')
print('Prepared 24 Lean files/tests and README; root case unchanged')
