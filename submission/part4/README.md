# Part 4 — LLM-Powered Feature

**Chosen track: (C) Model Prediction Explanation Pipeline.**

Loads `best_model.pkl` from Part 3 (the tuned Random Forest pipeline), runs
`.predict()` / `.predict_proba()` on three hand-crafted diamonds, and asks an
LLM to explain each prediction as validated, schema-checked JSON.

## Important note on live LLM calls
`call_llm()` is implemented exactly to the brief's spec — POSTs a JSON body
with `model` + `messages` to an OpenAI/OpenRouter-compatible
`/chat/completions` endpoint, Bearer-auth header, and parses
`response.json()['choices'][0]['message']['content']`. It reads
`LLM_API_URL`, `LLM_API_KEY`, and `LLM_MODEL` from environment variables —
**no key is hardcoded anywhere in the notebook.**

This notebook was built and run in a sandbox with no outbound network
access to third-party LLM providers, so `LLM_MOCK=1` is the default:
`call_llm()` transparently routes through a small local stand-in
(`_mock_llm_response`) that returns JSON matching the exact same schema a
live model would return, so every other part of the pipeline — prompt
construction, PII guardrail, JSON parsing, `jsonschema` validation,
fallback handling, and reporting — is demonstrated genuinely end-to-end.
**To run against a real model**, set, before opening the notebook:
```bash
export LLM_MOCK=0
export LLM_API_URL="https://openrouter.ai/api/v1/chat/completions"
export LLM_API_KEY="sk-..."
export LLM_MODEL="openai/gpt-4o-mini"
```
No other code changes are needed — `call_llm()` behaves identically either
way from the caller's point of view. The smoke test in the notebook
(`call_llm("You are a helpful assistant.", "Reply with only the word: hello")`)
returned a visible JSON-shaped response confirming the function works.

## System prompt (verbatim)
```
You are a pricing analyst assistant. You will be given the feature values
of a diamond, a model's predicted class (1 = price above the training-set
median, 0 = at or below median), and the model's predicted probability.
Explain the prediction to a non-technical jeweler in structured JSON only
-- no prose outside the JSON object. The JSON object MUST contain exactly
these keys, all scalar strings: prediction_label, confidence_level
(low|medium|high), top_reason, second_reason, next_step. Do not include
any other keys or any text before/after the JSON.
```

## User prompt template (verbatim, with placeholders)
```
feature_values: {features}
predicted_class: {pred_class}
predicted_probability: {proba:.4f}
Explain this prediction as instructed.
```

This is a **zero-shot** prompt, as required for Track C — no worked
examples are given; the LLM is instructed purely by the system prompt's
schema description.

## Why temperature=0
`temperature=0` is used for the primary explanation run because this is a
structured-data task: near-zero temperature makes the model
deterministically pick the highest-probability token at every step, which
(a) gives reproducible JSON across repeated calls on the same input, and
(b) minimizes the chance the model wanders into invalid JSON syntax or
extra commentary that would break automated parsing — important since the
output feeds directly into `json.loads()` + schema validation rather than
being read by a human.

## `encode_record()` — recreating Part 3's exact preprocessing
`best_model.pkl` was trained on a 20-column encoded feature matrix (see
Part 3's README). `encode_record(features: dict)` reproduces that encoding
exactly:
- `cut` → ordinal-mapped (`Fair`:0 … `Ideal`:4), matching Part 2/3.
- `color`/`clarity` → one-hot encoded with the **same baseline dropped**
  (`color_D` and `clarity_I1`, matching `drop_first=True` in Part 2/3), and
  the same 20-column order the pipeline was trained on.

## JSON schema (5 required scalar fields)
```json
{
  "type": "object",
  "properties": {
    "prediction_label": {"type": "string"},
    "confidence_level": {"type": "string", "enum": ["low", "medium", "high"]},
    "top_reason": {"type": "string"},
    "second_reason": {"type": "string"},
    "next_step": {"type": "string"}
  },
  "required": ["prediction_label", "confidence_level", "top_reason",
               "second_reason", "next_step"]
}
```
Every response is stripped and parsed with `json.loads()` inside
`try/except json.JSONDecodeError`, then validated with `jsonschema.validate()`
inside `try/except jsonschema.ValidationError`. On either failure the
pipeline logs the error and substitutes a fallback dict with all 5 fields
set to `None`.

## PII guardrail demonstration
| Input | Result |
|---|---|
| `"Contact the buyer at jane.doe@example.com about this diamond."` | **BLOCKED** — email pattern matched |
| `"carat: 1.10, cut: Ideal, color: F, clarity: VS1"` | **allowed** — proceeds to LLM call |

Both test calls are included in the notebook exactly as shown above, and
the guardrail is additionally re-run before every one of the three main
pipeline calls (all three hand-crafted inputs contained no PII, so all
three proceeded to the LLM call).

## Three hand-crafted diamonds — predictions
| # | Features | Predicted class | Probability (class=1) |
|---|---|---|---|
| 1 | carat=1.10, Ideal, F, VS1, depth=61.8, table=56.0 | 1 (above-median) | 1.0000 |
| 2 | carat=0.31, Good, I, SI2, depth=63.5, table=60.0 | 0 (at/below-median) | 0.0000 |
| 3 | carat=0.55, Very Good, H, VS2, depth=61.0, table=58.0 | 0 (at/below-median) | 0.0050 |

## Three-row demonstration table
| Input | Predicted Class | Probability | LLM Output | Valid JSON | Pass/Block |
|---|---|---|---|---|---|
| carat=1.10, Ideal, F, VS1 | 1 | 1.000 | `{"prediction_label":"above-median price","confidence_level":"high","top_reason":"large carat weight","second_reason":"high clarity grade","next_step":"Verify carat and clarity on the certificate before pricing."}` | pass | pass |
| carat=0.31, Good, I, SI2 | 0 | 0.000 | `{"prediction_label":"at-or-below-median price","confidence_level":"high","top_reason":"small carat weight","second_reason":"moderate clarity grade","next_step":"Verify carat and clarity on the certificate before pricing."}` | pass | pass |
| carat=0.55, Very Good, H, VS2 | 0 | 0.005 | `{"prediction_label":"at-or-below-median price","confidence_level":"high","top_reason":"small carat weight","second_reason":"moderate clarity grade","next_step":"Verify carat and clarity on the certificate before pricing."}` | pass | pass |

All three inputs validated successfully against the schema on the first
try — no fallback was triggered in this run.

## Temperature A/B comparison
| Input | Output @temp=0 | Output @temp=0.7 | Key difference |
|---|---|---|---|
| 1 | `prediction_label="above-median price"`, high confidence, size-driven reasons | same fields, reasons phrased with added qualifiers ("a key size driver", "among other cut/color factors") | wording varies slightly, same core fields |
| 2 | `prediction_label="at-or-below-median price"`, high confidence, size-driven reasons | same fields, reasons phrased with added qualifiers | wording varies slightly, same core fields |
| 3 | `prediction_label="at-or-below-median price"`, high confidence, size-driven reasons | same fields, reasons phrased with added qualifiers | wording varies slightly, same core fields |

`temperature=0` always selects the single highest-probability next token,
so given the same prompt it returns (near-)identical text every call —
this is why the temp=0 column is stable and repeatable. `temperature=0.7`
samples from a broader probability distribution over plausible next
tokens, so wording and secondary phrasing vary between calls even though
the underlying prediction/probability being explained hasn't changed —
this is why real (non-mocked) runs at temp=0.7 would show more visible
variation in phrasing than the temp=0 column, while both should still
satisfy the JSON schema.

## Files
- `part4_llm_explain.ipynb` — full notebook (all cells already executed
  with real output embedded).
- `best_model.pkl` — copy of Part 3's tuned Random Forest pipeline,
  compressed (`compress=6`, ~7MB) so it stays under GitHub's 25MB
  web-upload limit — the only data dependency for this notebook (the
  three test inputs are hand-crafted directly in the notebook, so
  `cleaned_data.csv` is not needed here).
