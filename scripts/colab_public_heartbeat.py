"""Keep the owned kernel active and inspect only safe worker liveness metadata."""
import json,time
from pathlib import Path
import psutil
status=Path('/content/public-status.json')
workers=[]
for proc in psutil.process_iter(['pid','cmdline']):
 args=proc.info['cmdline']or[]
 if any(x in args for x in ['/content/public-driver.py','finance_decisions.public_eval']):workers.append(proc.pid)
print(json.dumps({'time':time.time(),'status_file_exists':status.exists(),'worker_pids':workers,'boot_time':psutil.boot_time()}))
