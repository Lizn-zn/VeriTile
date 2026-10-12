from pathlib import Path
import concurrent.futures,datetime,hashlib,json,os,re,subprocess,time
root=Path('/home/argustest/lizn/VeriTile');work=Path(__file__).resolve().parent;src=work/'FifthSphDraft';logs=work/'validation';logs.mkdir(exist_ok=True)
record=json.loads((src/'review.json').read_text());names=[Path(n).stem for n in record['files']]
imports={n:set(re.findall(r'^import FifthSphDraft\.(\w+)',(src/(n+'.lean')).read_text(),re.M)) for n in names}
assert all(d<=set(names) for d in imports.values()),imports
lean=subprocess.run(['lake','env','which','lean'],cwd=root,capture_output=True,text=True,check=True).stdout.strip()
path=subprocess.run(['lake','env','printenv','LEAN_PATH'],cwd=root,capture_output=True,text=True,check=True).stdout.strip()
env=os.environ.copy();env['LEAN_PATH']=str(work)+':'+':'.join(str(root/p) if p and not p.startswith('/') else p for p in path.split(':'))
def closure(n):
 result={n};todo=list(imports[n])
 while todo:
  m=todo.pop()
  if m not in result:result.add(m);todo.extend(imports[m])
 return {m:hashlib.sha256((src/(m+'.lean')).read_bytes()).hexdigest() for m in sorted(result)}
def run(n):
 source=src/(n+'.lean');digest=hashlib.sha256(source.read_bytes()).hexdigest();start=time.time();log=logs/(n+'.log')
 with log.open('w') as out:
  p=subprocess.Popen([lean,'-R',str(work),'-o',str(src/(n+'.olean')),str(source)],cwd=root,env=env,stdout=out,stderr=subprocess.STDOUT)
  (logs/(n+'.running.json')).write_text(json.dumps({'pid':p.pid,'source_sha256':digest,'started':start,'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()},indent=2)+'\n')
  code=p.wait()
 result={'module':n,'code':code,'dependency_sources':closure(n),'source_sha256':digest,'seconds':time.time()-start,'log':str(log),'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 (logs/(n+'.result.json')).write_text(json.dumps(result,indent=2)+'\n');return result
done=set();pending=set(names);running={};results=[]
for n in names:
 receipt=logs/(n+'.result.json')
 if receipt.exists():
  r=json.loads(receipt.read_text())
  if r.get('code')==0 and r.get('dependency_sources')==closure(n) and (src/(n+'.olean')).exists():
   done.add(n);pending.remove(n);results.append(r);print('REUSE',n,flush=True)

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 while pending or running:
  ready=sorted(n for n in pending if imports[n]<=done)
  for n in ready[:max(0,2-len(running))]:
   pending.remove(n);running[pool.submit(run,n)]=n;print('START',n,flush=True)
  assert running,('Unsatisfied dependencies',pending,done)
  completed,_=concurrent.futures.wait(running,return_when=concurrent.futures.FIRST_COMPLETED)
  for f in completed:
   n=running.pop(f);r=f.result();results.append(r);(logs/'results.json').write_text(json.dumps(results,indent=2)+'\n');print('END',n,r['code'],round(r['seconds'],1),flush=True)
   if r['code']!=0:
    (logs/'failure.json').write_text(json.dumps(r,indent=2)+'\n');raise SystemExit('Failed '+n)
   done.add(n)
(logs/'complete.json').write_text(json.dumps({'modules':len(done),'results':results},indent=2)+'\n')
print('PASS all',len(done),'restored source/proof/test modules',flush=True)
