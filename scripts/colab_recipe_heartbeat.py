import json
import os
import time
from pathlib import Path

p=Path('/content/recipe-driver.pid');alive=False
if p.exists():
    try:os.kill(int(p.read_text()),0);alive=True
    except ProcessLookupError:pass
print(json.dumps({'time':time.time(),'driver_alive':alive,
                 'status_file_exists':Path('/content/recipe-status.json').exists()}))
