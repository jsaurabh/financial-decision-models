import json,time
from pathlib import Path

print(json.dumps({'time':time.time(),'status_file_exists':Path('/content/phase3-status.json').exists()}))
