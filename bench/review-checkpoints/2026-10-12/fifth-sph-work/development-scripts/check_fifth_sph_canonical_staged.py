from pathlib import Path
import json,subprocess,time
p=Path('/tmp/fifth-sph-canonical-source1.exit')
while not p.exists():time.sleep(5)
assert p.read_text().strip()=='0'
rows=[]
for module,label in [('BackwardBase','back-base1'),('BackwardReadback','back-readback1'),('ForwardReal','forward-real1')]:
 print('Starting '+module,flush=True)
 r=subprocess.run(['python3','/tmp/check_fifth_sph_canonical.py',module,label])
 rows.append(dict(module=module,label=label,code=r.returncode));Path('/tmp/fifth-sph-canonical-results.json').write_text(json.dumps(rows,indent=2)+'\n')
 assert r.returncode==0,module
