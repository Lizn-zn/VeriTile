from pathlib import Path
import json,subprocess,time
p=Path('/tmp/fifth-sph-source7.exit')
while not p.exists():time.sleep(5)
assert p.read_text().strip()=='0','source7 failed'
rows=[]
for module,label in [('SourceChecks','source-types2'),('Erasure','erasure1'),('Candidate','candidate1'),('ForwardReal','forward-real1'),('ForwardIO','forward-io1'),('FP','fp1')]:
 print('Starting '+module,flush=True)
 r=subprocess.run(['python3','/tmp/check_fifth_sph_module.py',module,label])
 rows.append(dict(module=module,label=label,code=r.returncode));Path('/tmp/fifth-sph-staged-results.json').write_text(json.dumps(rows,indent=2)+'\n')
 assert r.returncode==0,module
print('PASS all initial fifth-order staging checks',flush=True)
