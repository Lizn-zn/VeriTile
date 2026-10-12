from pathlib import Path
import json,subprocess,sys,time
kind=sys.argv[1]
if kind=='real':
 prerequisites=[f'/tmp/fifth-sph-backward-readback{c}-third.exit' for c in range(3)]
 modules=[('BackwardReadback','backward-readback-aggregate1'),('BackwardIO','backward-io1'),('Correct','correct1'),('Execution','execution1')]
else:
 assert kind=='typed'
 prerequisites=['/tmp/fifth-sph-typed-forward1.exit','/tmp/fifth-sph-typed-backward1.exit']
 modules=[('TypedExecution','typed-execution2')]
for name in prerequisites:
 p=Path(name)
 while not p.exists():time.sleep(5)
 assert p.read_text().strip()=='0',name
rows=[]
for module,label in modules:
 print('Checking '+module,flush=True)
 r=subprocess.run(['python3','/tmp/check_fifth_sph_module.py',module,label])
 rows.append(dict(module=module,label=label,code=r.returncode));Path('/tmp/fifth-sph-'+kind+'-closure-results.json').write_text(json.dumps(rows,indent=2)+'\n')
 assert r.returncode==0,module
print('PASS '+kind+' closure',flush=True)
