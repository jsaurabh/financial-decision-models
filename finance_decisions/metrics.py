import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.metrics import f1_score


def fit_temperature(logits, targets):
    def nll(log_t):
        p = softmax(np.asarray(logits) / np.exp(log_t), axis=-1)
        return -np.log(p[np.arange(len(p)), targets].clip(1e-12)).mean()
    result = minimize_scalar(nll, bounds=(-2.3, 2.3), method='bounded')
    return float(np.exp(result.x))


def metrics(logits, targets, temperature=1):
    p = softmax(np.asarray(logits) / temperature, axis=-1)
    y = np.asarray(targets); pred = p.argmax(-1); confidence = p.max(-1)
    correct = pred == y
    ece = 0
    for low in np.arange(0, 1, .1):
        mask = (confidence > low) & (confidence <= low + .1 + 1e-10)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    result = {'n': len(y), 'accuracy': float(correct.mean()),
              'position_macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
              'nll': float(-np.log(p[np.arange(len(y)), y].clip(1e-12)).mean()),
              'brier': float(((p - np.eye(p.shape[1])[y])**2).sum(-1).mean()), 'ece_10bin': float(ece)}
    result['selective'] = {}
    for t in [.5, .7, .8, .9, .95]:
        mask = confidence >= t
        result['selective'][str(t)] = {'coverage': float(mask.mean()),
                                      'accuracy': float(correct[mask].mean()) if mask.any() else None}
    return result
