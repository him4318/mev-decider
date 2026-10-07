# mev-decider

Run [**mev-decider**](https://huggingface.co/metalkeys/mev), a ModernBERT-based System One decision model. It reads a state (text or JSON) and typed questions about it, returning calibrated probabilities in a single forward pass. It uses TypeSafe Jev's question and answer format, allowing a local server to stand in for the `/v1/systemone` API.

- **Bidirectional ModernBERT Backbone (~149M Base / ~395M Large).** Highly efficient transformer encoder running on laptop CPU, Apple Silicon, or GPU.
- **Choice-order invariant.** Every candidate choice restarts at the identical starting position index $Q$, ensuring equidistant geometric relation to the prompt and state under RoPE coordinates. Native bidirectional attention allows all choices to mutually cross-attend, while configuring `local_attention: 16384` eliminates sliding-window artifacts and guarantees complete sequence-wide global attention. This is exact in fp32: reordering choices yields zero numerical drift. On GPU or Apple Silicon, default `precision="bf16"` inference runs at maximum throughput; pass `precision="fp32"` if perfect machine-precision invariance is required.
- **Typed answers:** `choice` (a key and probabilities), `noul` (P(yes)), and `score` (an expected level and probabilities).

---

## Install

```bash
pip install mev-decider            # library
pip install "mev-decider[serve]"   # + local /v1/systemone server
```

The model weights are downloaded from the Hugging Face Hub on first use.

---

## Python

```python
from mev_decider import load

decider = load()  # loads metalkeys/mev or local checkpoint directory

state = {
    "message": (
        "URGENT: you charged my card twice this month. "
        "Refund the duplicate within 24 hours or I'm disputing it with my bank."
    )
}

questions = {
    "intent": {
        "type": "choice",
        "instructions": "What does the customer want?",
        "criteria": {
            "refund": "money returned or a duplicate charge reversed",
            "technical_help": "a bug, outage or integration problem",
            "cancellation": "wants to cancel or downgrade",
        },
    },
    "urgent": {
        "type": "noul",
        "instructions": "Does the message communicate time pressure or a deadline?",
    },
    "anger": {
        "type": "score",
        "instructions": "How angry is the customer?",
        "criteria": ["calm", "mildly annoyed", "frustrated", "furious"],
    },
}

answers = decider.decide(state, questions)
```

Example output:

```json
{
  "intent": {
    "type": "choice",
    "choice": "refund",
    "probabilities": {"refund": 1.0, "technical_help": 0.0, "cancellation": 0.0}
  },
  "urgent": {"type": "noul", "noul": 0.99},
  "anger": {
    "type": "score",
    "score": 1.90,
    "probabilities": {"0": 0.09, "1": 0.10, "2": 0.65, "3": 0.17}
  }
}
```

To choose a device, pass `load(device="cpu")`, `"mps"`, or `"cuda"`. On GPU and Apple Silicon, inference runs in bf16 by default for speed; pass `load(precision="fp32")` for exact probabilities and perfect option-order invariance (the CPU always runs fp32).

Question types:

| type | criteria | answer |
|---|---|---|
| `choice` | `{key: description}` (a description may be empty) | `choice`: the most likely key, and `probabilities` per key |
| `noul` | optional `{"true": ..., "false": ...}` | `noul`: P(yes) |
| `score` | a list of level descriptions, lowest first | `score`: the expected level, and `probabilities` per level |

Questions about the same state are batched together. States longer than 1,024 tokens are truncated; change the limit with `load(max_state_tokens=...)`.

---

## Server

```bash
mev-decider serve --port 8008          # add --precision fp32 for exact results
curl -s localhost:8008/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Order 1182 arrived with a cracked screen.",
  "questions": {
    "damaged": {"type": "noul", "instructions": "Was the item damaged on arrival?"}
  }
}'
```

`POST /v1/systemone` accepts `{"state", "questions", "model"?}` and returns `{"model", "answers", "latency_ms"}`. `GET /v1/models` describes the loaded model. The server has no authentication and binds to `127.0.0.1` by default.

---

## Command Line

```bash
echo '{"state": "...", "questions": {...}}' | mev-decider decide
```

---

## Tests

```bash
uv run pytest
```

---

## License

The code is Apache-2.0.
