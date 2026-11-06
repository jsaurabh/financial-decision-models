# Decision API formats

The local Gemma server supports TypeSafe System One and OpenAI Decisions request/response formats for a bounded text/JSON subset. No provider API is called. The model weights, saved temperature, prompt renderer and training/evaluation code are unchanged.

Run `uvicorn finance_decisions.serve:app --host 127.0.0.1 --port 8000 --workers 1` with `DECISION_ARTIFACT` pointing to the saved model. Set `DECISION_MODEL_ID` to its actual ID; the default remains the earlier v4 release. The server loads weights during startup. `/health` reports the configured ID, model token limit and supported scope. These examples use the default ID; substitute your configured ID when serving a phase-2 artifact.

## TypeSafe System One

`POST /v1/systemone` accepts `state` and a map of named questions. `model` is optional and defaults to the configured local artifact; a different ID returns 404.

```json
{
  "model": "jsaurabh/gemma4-e2b-finance-decisions-v4",
  "state": "The customer reports an unauthorized card payment.",
  "questions": {
    "review": {"type": "noul", "instructions": "Does this report suspected fraud?"},
    "route": {
      "type": "choice", "instructions": "Choose the support team.",
      "criteria": {"payments": "Payment processing issue", "security": "Suspected fraud"}
    },
    "severity": {
      "type": "score", "instructions": "Rate urgency.",
      "criteria": ["Low", "Medium", "High"]
    }
  }
}
```

An illustrative response, not model benchmark output:

```json
{
  "model": "jsaurabh/gemma4-e2b-finance-decisions-v4",
  "answers": {
    "review": {"type": "noul", "noul": 0.9},
    "route": {
      "type": "choice", "choice": "security",
      "probabilities": {"payments": 0.1, "security": 0.9}, "confidence": 0.8
    },
    "severity": {
      "type": "score", "score": 1.6,
      "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7},
      "legend": {"0": "Low", "1": "Medium", "2": "High"}, "confidence": 0.25
    }
  },
  "usage": {"input_tokens": 240, "output_tokens": 0}
}
```

Binary answers return the probability of true, not a hard boolean. Scores return the expected zero-based level, which can be fractional. Ties choose the first supplied maximum. Structured state/instructions/criteria are serialized as JSON text for the model. Null choice descriptions fall back to the label; missing binary criteria get explicit true/false defaults.

## OpenAI Decisions

`POST /v1/decisions` requires `model`, `input` and an ordered question array. The model ID must identify the configured local artifact. It does not route to GPT-6 Luna or impersonate its model ID.

```json
{
  "model": "jsaurabh/gemma4-e2b-finance-decisions-v4",
  "input": "The customer reports an unauthorized card payment.",
  "questions": [
    {"type": "predicate", "name": "review", "instructions": "Does this report suspected fraud?"},
    {
      "type": "choice", "name": "route", "instructions": "Choose the support team.",
      "choices": [
        {"value": "payments", "description": "Payment processing issue"},
        {"value": "security", "description": "Suspected fraud"}
      ]
    },
    {
      "type": "score", "name": "severity", "instructions": "Rate urgency.",
      "levels": [{"label": "Low"}, {"label": "Medium"}, {"label": "High"}]
    }
  ]
}
```

The illustrative answer portion for the same distributions:

```json
{
  "answers": [
    {"type": "predicate", "name": "review", "probability": 0.9},
    {
      "type": "choice", "name": "route", "choice": "security", "confidence": 0.8,
      "probabilities": [{"value": "payments", "probability": 0.1}, {"value": "security", "probability": 0.9}]
    },
    {
      "type": "score", "name": "severity", "score": 1.6, "confidence": 0.25,
      "probabilities": [
        {"value": 0, "label": "Low", "probability": 0.1},
        {"value": 1, "label": "Medium", "probability": 0.2},
        {"value": 2, "label": "High", "probability": 0.7}
      ]
    }
  ]
}
```

The full response also contains `model` and `usage`, including `input_tokens_details`, `output_tokens_details` and `total_tokens`. Names are optional and echoed as null when omitted. Supplied names must be unique. String and boolean choice values preserve their types; `true` and `"true"` are distinct candidates. Boolean/mixed values use unique internal labels, so their model prompts differ from native string-label prompts. Level descriptions are optional; scores use ordered indices, not arbitrary numeric values.

Input can be a string or an array of user messages with string content or `input_text`/`input_json` content parts. Question instructions also accept these text/JSON parts. One message becomes its rendered content; multiple messages become a JSON list retaining their boundaries. These inputs are rendered as evidence for the existing classifier, not processed with OpenAI's chat template.

## Confidence, usage and scope

Choice confidence rescales the maximum probability above a uniform distribution. Score confidence measures probability-weighted distance from the most likely level, normalized by uniform mean absolute deviation. Both follow [TypeSafe's published formulas](https://docs.typesafe.ai/confidence). This is a distribution statistic, not a separately learned correctness estimate or a reproduction of OpenAI's undocumented hosted confidence calculation. Binary/predicate answers have no separate confidence field.

Token usage counts actual model input positions, including repeated state/question prefixes and packed candidate blocks. Each question runs separately; there is no shared prefill or prompt cache. These local counts are not either provider's billable tokenizer counts. Output/reasoning/cache tokens are zero because the classifier does not generate tokens or cache prefixes. The saved calibration temperature is applied once.

Limits remain 1–8 questions, 2–8 choices/score levels, the existing 12,000-character per-question budget, and the loaded artifact's token limit. There is no silent truncation or candidate reduction. Images/assets, box questions, tool/conversation items other than user messages, provider cache controls and other unknown fields return 422. Model mismatch returns 404; busy inference returns 503. Model failures return 500. The local model does not implement a learned refusal behavior; the response adapter can preserve a backend refusal but does not fabricate one.

This is schema compatibility for the supported subset, not full provider capability parity. Contracts were checked on November 6, 2026 against the [TypeSafe API](https://docs.typesafe.ai/api/system-one), [official TypeSafe response types](https://github.com/typesafe-ai/typesafe-sdk-js/blob/main/src/types.ts), and [OpenAI Decisions reference](https://developers.openai.com/api/reference/resources/decisions/methods/create).

## Existing custom clients

The old `state`/named-`questions` payload with optional `review_threshold` now belongs at `POST /v1/classify`. It retains `value`, `probabilities`, `max_probability`, `needs_review`, score `most_likely_level`, and top-level `shared_prefill: false`. The previous `/v1/decisions` custom shape is replaced by the OpenAI format; `/v1/systemone` now returns the Jev format. Provider-compatible responses omit local review metadata. Set review thresholds in your application using reviewed in-domain labels.

Run `python -m pytest tests/test_api_contracts.py` for deterministic HTTP contract checks without loading weights or calling providers. Run `python scripts/api_smoke.py` with a local saved artifact for the real-model integration smoke. Historical smoke artifacts describe the earlier custom interface and are retained unchanged.
