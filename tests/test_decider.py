"""Tests for mev-decider: batching consistency, permutation invariance, and server endpoints."""

import json
from pathlib import Path

import pytest

from mev_decider import load

PARITY_FILE = Path(__file__).parent / "parity_cases.json"
CASES = json.loads(PARITY_FILE.read_text()) if PARITY_FILE.exists() else {"cases": []}


@pytest.fixture(scope="module")
def decider():
    return load(device="cpu")


def probs(answer):
    return list(answer["probabilities"].values()) if "probabilities" in answer else [1 - answer["noul"], answer["noul"]]


def test_batched_equals_single(decider):
    if not CASES.get("cases"):
        state = "Traffic scenario at a 4-way stop."
        questions = {
            "q1": {"type": "noul", "instructions": "Should ego yield?"},
            "q2": {
                "type": "choice",
                "instructions": "Next action?",
                "criteria": {"stop": "Stop", "go": "Go", "turn": "Turn"},
            },
        }
    else:
        cases = CASES["cases"][:6]
        state = cases[0]["state"]
        questions = {str(i): c["question"] for i, c in enumerate(cases)}

    batched = decider.decide(state, questions, batch_size=len(questions))
    for qid, question in questions.items():
        single = decider.decide(state, {qid: question})[qid]
        assert probs(batched[qid]) == pytest.approx(probs(single), abs=1e-4)


def test_choice_order_invariant(decider):
    state = "Ego vehicle is approaching a yellow traffic light 40 meters ahead."
    question = {
        "type": "choice",
        "instructions": "What is the safest action?",
        "criteria": {
            "stop": "Come to a complete stop",
            "lane": "Stay in current lane",
            "intersection": "Proceed through intersection",
            "honk": "Honk horn",
        },
    }
    reversed_question = {**question, "criteria": dict(reversed(list(question["criteria"].items())))}
    a = decider.decide(state, {"q": question})["q"]
    b = decider.decide(state, {"q": reversed_question})["q"]
    assert a["choice"] == b["choice"]
    for key, p in a["probabilities"].items():
        assert b["probabilities"][key] == pytest.approx(p, abs=1e-4)


def test_server(decider):
    from fastapi.testclient import TestClient

    from mev_decider.serve import create_app

    client = TestClient(create_app(decider))
    assert client.get("/v1/models").status_code == 200
    body = {
        "state": {"message": "Refund the duplicate charge today or we cancel."},
        "questions": {
            "urgent": {"type": "noul", "instructions": "Is there time pressure?"},
            "intent": {
                "type": "choice",
                "instructions": "What does the customer want?",
                "criteria": {"refund": "money back", "technical_help": "a bug"},
            },
        },
    }
    r = client.post("/v1/systemone", json=body)
    assert r.status_code == 200
    answers = r.json()["answers"]
    direct = decider.decide(body["state"], body["questions"])
    assert answers["intent"]["choice"] == direct["intent"]["choice"]
    assert answers["urgent"]["noul"] == pytest.approx(direct["urgent"]["noul"], abs=1e-6)
    assert client.post("/v1/systemone", json={"state": "x", "questions": {"q": {"type": "bad"}}}).status_code == 400


def test_invalid_precision():
    with pytest.raises(ValueError):
        load(precision="fp16")
