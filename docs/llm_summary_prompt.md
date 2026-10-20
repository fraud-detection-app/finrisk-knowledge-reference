---
title: LLM Summary Prompt and Output Contract
owner: ML Engineering
effective: 2026-09-16
---

# LLM Summary Prompt

The prompt, the output contract and the three rules behind the risk explainer in
`src/financial_risk/llm/bedrock_client.py`. Week 8 builds this; week 10 exposes it as an API.

## Where the LLM sits

```
features -> classifier (score, threshold) -> SHAP local reasons -> LLM -> reviewer-ready insight
```

The classifier **decides** (`risk_level`), SHAP **attributes** (`top_reasons`), the LLM
**narrates**. The LLM never sees the raw feature row and never changes the score.

## The prompt

Chat form (`build_messages`) -- the shape every chat API takes:

```text
[system]
You are a financial risk analytics assistant for an educational data science project.
A machine-learning model has already scored the transaction; you explain that score, you do not change it.
Return strict JSON with exactly these keys: risk_level, summary, key_reasons, recommended_actions, disclaimer.
risk_level must repeat the model's risk_level from the context. key_reasons must be grounded in the context's top_reasons.
Do not provide investment, lending, or legal advice. Mention this is not a production decision system.

[user]
Transaction/model context:
{
  "row_id": 1,
  "transaction_type": "TRANSFER",
  "amount": 250000.0,
  "risk_score": 0.82,
  "risk_level": "high",
  "threshold": 0.5,
  "top_reasons": ["origin_new_balance = 0 after the transfer", "amount is 98% of the balance"]
}
```

`build_risk_prompt` flattens the same two parts into one string for completion-style
endpoints (DeepSeek v3.2 on Bedrock, the course default).

### Message types

| role | what it is | in this project |
|---|---|---|
| `system` | standing policy: who the model is, what to return, what never to do | `SYSTEM_PROMPT`, sent with every call |
| `user` | the request for this turn | the curated context as JSON |
| `assistant` | a previous model turn -- or one we *write* as a demonstration | `FEW_SHOT_EXAMPLE` via `build_messages(..., examples=[...])` |
| `tool` / function | a tool's result fed back to the model | week 10's API, not used here |

`build_messages(context, examples=..., extra_instructions=...)` composes these: extra
instructions extend the system message (e.g. a reasoning hint), examples become
user/assistant pairs before the final user turn.

### Prompt security and the harness

The prompt is not a security boundary: any text the model reads can try to instruct it
(prompt injection), and a system prompt only *asks* the model to resist. The harness -- the
code around the model -- is what actually holds:

- **allow-listed context**: `CONTEXT_KEYS` -- free-text fields the narrative does not need never reach the prompt;
- **injection scan**: `scan_for_injection()` flags strings matching `INJECTION_PATTERNS`; `curate_context()` withholds them and caps every string at `MAX_FIELD_CHARS`;
- **output gates**: extract / validate / enforce (below) -- an obeyed injection still cannot change `risk_level`;
- **no tools, no side effects**: the explainer can only return JSON; it cannot query, transfer, or email;
- **fallback + timeout**: a failed or refused call degrades to the template, never to an error page;
- **logging and evals**: every withheld field and every gate failure is a metric to watch.

### Context control

Only `CONTEXT_KEYS` reach the prompt: `row_id, transaction_type, amount, risk_score,
risk_level, threshold, top_reasons`. `curate_context()` drops everything else -- account
identifiers, balances, the customer's history. Less leakage surface, cheaper calls, and the
model cannot be "helpful" with facts it was never given.

## The output contract (`INSIGHT_SCHEMA`)

```json
{
  "type": "object",
  "required": ["risk_level", "summary", "key_reasons", "recommended_actions", "disclaimer"],
  "additionalProperties": false,
  "properties": {
    "risk_level":          {"type": "string", "enum": ["low", "medium", "high"]},
    "summary":             {"type": "string", "minLength": 1, "maxLength": 600},
    "key_reasons":         {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 6},
    "recommended_actions": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 6},
    "disclaimer":          {"type": "string", "minLength": 1}
  }
}
```

The reply goes through three gates before anyone downstream sees it:

| gate | function | on failure |
|---|---|---|
| 1. extract | `extract_json(raw)` -- first `{ ... }` in the reply, prose/code fences tolerated | fallback |
| 2. validate | `validate_insight(d)` -- required keys, types, enum, list sizes | fallback |
| 3. enforce | `enforce_model_decision(d, context)` -- `risk_level` reset to the model's; disagreement noted in `summary` | never fails |

`parse_insight(raw, context)` runs all three and returns `None` when the fallback should be
used.

## The fallback

`generate_risk_insight(context, use_bedrock=False)` -- and any exception from the cloud call,
and any reply that fails validation -- returns `fallback_risk_insight(context)`: a
deterministic template that satisfies the same schema. The class runs on it offline; a
production system needs it anyway, because an LLM outage must not stop scoring.

## Example output

```json
{
  "risk_level": "high",
  "summary": "The transaction is classified as high risk with score 0.820.",
  "key_reasons": ["origin_new_balance = 0 after the transfer", "amount is 98% of the balance"],
  "recommended_actions": [
    "Review the transaction in an analyst workflow.",
    "Check whether the pattern is consistent with the customer's recent activity.",
    "Do not make automated financial decisions from this educational model alone."
  ],
  "disclaimer": "Educational demo only. This is not financial, lending, legal, or investment advice."
}
```
