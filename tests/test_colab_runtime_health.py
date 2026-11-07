import pytest
from scripts.colab_runtime_health import confirmed_runtime_loss

@pytest.mark.parametrize('detail',['503 Service Unavailable','Download failed: 502 Bad Gateway','Connection reset','401 credential refresh failed',"Session 'example' not found; upstream returned 503"])
def test_outage_and_auth_failure_do_not_prove_vm_loss(detail):
    assert not confirmed_runtime_loss(1,detail)

@pytest.mark.parametrize('detail',["Session 'example' not found",'Runtime has been deleted','runtime expired'])
def test_explicit_loss_can_trigger_bounded_recovery(detail):
    assert confirmed_runtime_loss(1,detail)

def test_success_never_triggers_recovery():
    assert not confirmed_runtime_loss(0,'runtime expired')
