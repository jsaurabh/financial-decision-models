"""A service outage is not evidence of runtime deletion or preemption."""
import re

def confirmed_runtime_loss(returncode,detail):
    if returncode==0:return False
    text=' '.join(detail.lower().split())
    if any(word in text for word in ['502','503','504','401','403','timeout','timed out','connection','gateway','service unavailable','credential']):return False
    return bool(re.search(r"(?:session|runtime)\s+(?:(?:['\"][^'\"]+['\"]|[a-z0-9_-]+)\s+)?(?:was |is |has been )?(?:not found|deleted|expired|terminated)",text))
