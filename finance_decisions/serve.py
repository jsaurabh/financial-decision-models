"""Single-pass Gemma API. Run one process per replica; bound inference concurrency."""
import os
import threading

import torch
from huggingface_hub import snapshot_download

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .model import DecisionModel
from .schema import answer, options

app = FastAPI(title='Financial Decision Classifier')
artifact = os.environ.get('DECISION_ARTIFACT')
if artifact is None:
    artifact = snapshot_download('jsaurabh/gemma4-e2b-finance-decisions-v1',
        allow_patterns=['decision_config.json', 'readout.safetensors', 'backbone/*', 'tokenizer/*'])
device = os.environ.get('DECISION_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
model = DecisionModel.load(artifact, device=device)
capacity = threading.BoundedSemaphore(1)


class Request(BaseModel):
    state: str | dict | list
    questions: dict[str, dict] = Field(min_length=1, max_length=8)
    review_threshold: float = Field(default=.8, ge=0, le=1)


@app.get('/health')
def health():
    return {'status': 'ready', 'runtime': 'pytorch', 'device': device, 'max_length': model.max_length,
            'max_choices': 8, 'modalities': ['text'],
            'probabilities': 'conditional on supplied choices; threshold requires in-domain validation'}


@app.post('/v1/decisions')
@app.post('/v1/systemone')
def decisions(request: Request):
    if not capacity.acquire(blocking=False):
        raise HTTPException(503, 'Classifier busy; retry or scale replicas')
    try:
        answers = {}
        with torch.inference_mode():
            for key, question in request.questions.items():
                labels, _ = options(question)
                row = {'state': request.state, 'question': question}
                logits = model(model.prepare([row]), [len(labels)])[0, :len(labels)]
                probabilities = (logits / model.temperature).softmax(-1).cpu().tolist()
                answers[key] = answer(question, probabilities, request.review_threshold)
        return {'answers': answers, 'model': 'gemma4-e2b-finance-decisions-v1', 'shared_prefill': False}
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        capacity.release()
