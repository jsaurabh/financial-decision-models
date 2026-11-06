import os,re,subprocess,sys,time
env=dict(os.environ);env["GOOGLE_APPLICATION_CREDENTIALS"]="/workspace/.credentials/public-eval-colab-adc.json"
role=sys.argv[1];args=sys.argv[2:]
cmd=["/workspace/.venv/bin/colab","--auth","adc","--config",f"/workspace/.credentials/public-{role}-sessions.json"]+args
for attempt in range(4):
 r=subprocess.run(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 clean=re.sub(r"(colab-runtime-proxy-token=)[^&\s]+",r"\1[REDACTED]",r.stdout)
 print(clean,end="",flush=True)
 if r.returncode==0:break
 if not any(x in r.stdout for x in ["502", "503", "504"]):break
 time.sleep(5*(attempt+1))
sys.exit(r.returncode)
