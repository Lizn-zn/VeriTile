from pathlib import Path
import hashlib,json,os,subprocess,sys,time
root=Path('/home/argustest/lizn/VeriTile');work=Path('/home/argustest/lizn/VeriTile/Logs/fifth-sph-work');module=sys.argv[1];assert module.isidentifier();source=work/'FifthSphDraft'/(module+'.lean');logs=work/'validation';logs.mkdir(exist_ok=True)
lean=subprocess.run(['lake','env','which','lean'],cwd=root,capture_output=True,text=True,check=True).stdout.strip();path=subprocess.run(['lake','env','printenv','LEAN_PATH'],cwd=root,capture_output=True,text=True,check=True).stdout.strip();env=os.environ.copy();env['LEAN_PATH']=str(work)+':'+':'.join(str(root/p) if p and not p.startswith('/') else p for p in path.split(':'))
start=time.time();sha=hashlib.sha256(source.read_bytes()).hexdigest()
with (logs/(module+'.log')).open('w') as out:
 p=subprocess.Popen([lean,'-R',str(work),'-o',str(source.with_suffix('.olean')),str(source)],cwd=root,env=env,stdout=out,stderr=subprocess.STDOUT)
 (logs/(module+'.running.json')).write_text(json.dumps({'pid':p.pid,'source_sha256':sha,'started':start,'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()},indent=2)+'\n');code=p.wait()
result={'module':module,'code':code,'source_sha256':sha,'seconds':time.time()-start};(logs/(module+'.result.json')).write_text(json.dumps(result,indent=2)+'\n');print(result);raise SystemExit(code)
