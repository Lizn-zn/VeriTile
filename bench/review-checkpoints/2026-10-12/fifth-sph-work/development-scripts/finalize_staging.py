from pathlib import Path
import datetime,hashlib,json,subprocess,time
w=Path(__file__).resolve().parent;src=w/'FifthSphDraft';logs=w/'validation'
r=json.loads((src/'review.json').read_text());names=[Path(f).stem for f in r['files']]
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
# Source snapshot for the uninterrupted DAG; all dependencies except the
# leaf Execution proof remain unchanged. New leaf checks have actual receipts.
for n in names:
 if n!='Execution':assert digest(src/(n+'.lean'))==r['files'][n+'.lean'],n
for n in ['TypedForwardExecution','TypedBackwardExecution']:
 while not (logs/(n+'.result.json')).exists():time.sleep(10)
 result=json.loads((logs/(n+'.result.json')).read_text());assert result['code']==0,result
 assert result['source_sha256']==digest(src/(n+'.lean')),n
 print('Confirmed',n,flush=True)
subprocess.run(['python3',str(w/'check_module.py'),'TypedExecution'],check=True)
rows=[]
for n in names:
 result=json.loads((logs/(n+'.result.json')).read_text())
 assert result['code']==0,result
 assert result['source_sha256']==digest(src/(n+'.lean')),n
 assert (src/(n+'.olean')).exists(),n
 rows.append(result)
(logs/'complete.json').write_text(json.dumps(dict(modules=len(rows),results=rows,checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),note='Actual module receipts; original dependency snapshot checked unchanged except the independently rechecked Execution leaf.'),indent=2)+'\n')
r.update(status='All 24 recovered staging modules now pass fresh local validation. Repository integration and independent replay remain pending.',files={n+'.lean':digest(src/(n+'.lean')) for n in names},accepted=False)
(src/'review.json').write_text(json.dumps(r,indent=2)+'\n')
subprocess.run(['python3',str(w/'prepare_fifth_sph_migration.py')],check=True)
print('All staging checks passed; migration prepared for review.',flush=True)
