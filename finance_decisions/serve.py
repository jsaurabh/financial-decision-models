"""Local Gemma server with TypeSafe and OpenAI decision response formats."""
import logging
import math
import os
import threading
from contextlib import asynccontextmanager

from fastapi import Body, FastAPI, HTTPException

from .api_protocol import decisions_request, decisions_response, systemone_answer, systemone_request
from .schema import answer, options

DEFAULT_MODEL_ID = 'jsaurabh/gemma4-e2b-finance-decisions-v4'


class GemmaBackend:
    def __init__(self, model_id, artifact=None):
        import torch
        from huggingface_hub import snapshot_download
        from .model import DecisionModel

        if artifact is None:
            artifact = snapshot_download(model_id, revision=os.environ.get('DECISION_REVISION'),
                allow_patterns=['decision_config.json', 'readout.safetensors', 'backbone/*', 'tokenizer/*'])
        self.device = os.environ.get('DECISION_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DecisionModel.load(artifact, device=self.device)
        self.max_length = self.model.max_length
        if not math.isfinite(self.model.temperature) or self.model.temperature <= 0:
            raise ValueError('Saved calibration temperature must be finite and positive')

    def predict(self, rows):
        import torch

        distributions, input_tokens = [], 0
        with torch.inference_mode():
            for row in rows:
                batch = self.model.prepare([row])
                # Count actual packed computation, once per question; no shared prefill.
                input_tokens += int(batch['input_ids'].numel())
                count = len(options(row['question'])[0])
                logits = self.model(batch, [count])[0, :count]
                distributions.append((logits / self.model.temperature).softmax(-1).cpu().tolist())
        return distributions, input_tokens


def create_app(backend=None, *, model_id=None):
    model_id = model_id or os.environ.get('DECISION_MODEL_ID', DEFAULT_MODEL_ID)
    capacity = threading.BoundedSemaphore(1)

    @asynccontextmanager
    async def lifespan(app):
        if app.state.backend is None:
            app.state.backend = GemmaBackend(model_id, os.environ.get('DECISION_ARTIFACT'))
        yield

    app = FastAPI(title='Financial Decision Classifier', lifespan=lifespan)
    app.state.backend = backend

    def evaluate(body, *, legacy_threshold=None):
        if body.get('model', model_id) != model_id:
            raise HTTPException(404, 'Unknown model; supply the configured local Gemma model ID')
        if not capacity.acquire(blocking=False):
            raise HTTPException(503, 'Classifier busy; retry or scale replicas')
        try:
            rows = [{'state': body['state'], 'question': q} for q in body['questions'].values()]
            distributions, tokens = app.state.backend.predict(rows)
            if len(distributions) != len(rows) or type(tokens) is not int or tokens < 0:
                raise RuntimeError('Backend returned invalid answer count or token usage')
            answers = {key: systemone_answer(q, p) for (key, q), p in zip(body['questions'].items(), distributions, strict=True)}
            if legacy_threshold is not None:
                answers = {key: answer(q, p, legacy_threshold) for (key, q), p in zip(body['questions'].items(), distributions, strict=True)}
                return {'model': model_id, 'answers': answers, 'shared_prefill': False}
            return {'model': model_id, 'answers': answers, 'usage': {'input_tokens': tokens, 'output_tokens': 0}}
        finally:
            capacity.release()

    def run(body, mode):
        try:
            if mode == 'decisions':
                normalized, mapping = decisions_request(body)
                result = evaluate(normalized)
                tokens = result['usage']['input_tokens']
                result['usage'].update(input_tokens_details={'cached_tokens': 0, 'cache_write_tokens': 0},
                                       output_tokens_details={'reasoning_tokens': 0}, total_tokens=tokens)
                return decisions_response(result, mapping)
            if mode == 'legacy':
                body = dict(body)
                threshold = body.pop('review_threshold', .8)
                if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
                    raise ValueError('review_threshold must be between 0 and 1')
                return evaluate(systemone_request(body), legacy_threshold=threshold)
            return evaluate(systemone_request(body))
        except (ValueError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from exc
        except HTTPException:
            raise
        except Exception:
            logging.getLogger(__name__).exception('Decision inference failed')
            raise HTTPException(500, 'Decision inference failed') from None

    @app.get('/health')
    def health():
        current = app.state.backend
        return {'status': 'ready' if current is not None else 'starting', 'model': model_id,
                'runtime': 'pytorch', 'device': getattr(current, 'device', None),
                'max_length': getattr(current, 'max_length', None), 'max_choices': 8,
                'max_questions': 8, 'modalities': ['text'], 'shared_prefill': False,
                'confidence': 'TypeSafe distribution formulas in both formats; not OpenAI hosted confidence',
                'probabilities': 'conditional on supplied choices; thresholds require in-domain validation'}

    @app.post('/v1/systemone')
    def systemone(body: dict = Body(...)):
        return run(body, 'systemone')

    @app.post('/v1/decisions')
    def decisions(body: dict = Body(...)):
        return run(body, 'decisions')

    @app.post('/v1/classify', deprecated=True)
    def legacy(body: dict = Body(...)):
        return run(body, 'legacy')

    return app


app = create_app()
