"""Loading mev-decider from the Hugging Face Hub and answering typed questions."""

import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoTokenizer, ModernBertConfig, ModernBertModel

from .encode import Encoder
from .model import ChoiceHead, DeciderNetwork

DEFAULT_MODEL = "metalkeys/mev"


def default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Decider:
    def __init__(self, network, encoder, config, device, name, precision="bf16"):
        self.network, self.encoder, self.config, self.device, self.name = network, encoder, config, device, name
        self.precision = precision

    @torch.no_grad()
    def probabilities(self, examples):
        batch = {k: v.to(self.device) for k, v in self.encoder.collate(examples).items()}
        # bf16 autocast on GPU / Apple Silicon (as in training); the CPU always runs fp32
        use_bf16 = self.precision == "bf16" and self.device.type in ("cuda", "mps")
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=use_bf16):
            logits = self.network(**batch)
        return torch.softmax(logits.float(), dim=-1).cpu()

    def decide(self, state, questions, batch_size=16):
        """Answers Jev-format questions about one state.

        state: a string, or any JSON value (rendered as indented JSON).
        questions: {id: {"type": "choice" | "noul" | "score", "instructions": ..., "criteria": ...}}
        Returns {id: answer}, where answer is
          choice: {"type": "choice", "choice": <best key>, "probabilities": {key: p}}
          noul:   {"type": "noul", "noul": P(yes)}
          score:  {"type": "score", "score": <expected level>, "probabilities": {"0": p, ...}}
        """
        ids = list(questions)
        examples = [self.encoder.encode(state, questions[qid]) for qid in ids]
        answers = {}
        for start in range(0, len(examples), batch_size):
            chunk = examples[start:start + batch_size]
            probs = self.probabilities(chunk)
            for qid, ex, p in zip(ids[start:start + batch_size], chunk, probs):
                p = p[:len(ex["keys"])].tolist()
                kind = questions[qid]["type"]
                if kind == "noul":
                    answers[qid] = {"type": "noul", "noul": p[1]}
                elif kind == "choice":
                    best = max(range(len(p)), key=p.__getitem__)
                    answers[qid] = {"type": "choice", "choice": ex["keys"][best], "probabilities": dict(zip(ex["keys"], p))}
                else:
                    answers[qid] = {"type": "score", "score": sum(i * x for i, x in enumerate(p)),
                                    "probabilities": dict(zip(ex["keys"], p))}
        return answers


def load(model=DEFAULT_MODEL, revision=None, device=None, max_state_tokens=None, precision="bf16"):
    """Loads mev-decider from a Hub repo id or a local folder.

    precision: "bf16" (default) runs inference under bf16 autocast on CUDA / Apple Silicon, which is faster;
    "fp32" is exact. The CPU always runs fp32.
    """
    if precision not in ("bf16", "fp32"):
        raise ValueError(f"precision must be 'bf16' or 'fp32', not {precision!r}")
    path = Path(model) if Path(model).is_dir() else Path(snapshot_download(model, revision=revision))
    device = torch.device(device) if device else default_device()

    config = {}
    if (path / "config.json").exists():
        try:
            config = json.loads((path / "config.json").read_text())
        except Exception:
            pass
    if not config and (path / "meta.json").exists():
        try:
            config = json.loads((path / "meta.json").read_text())
        except Exception:
            pass

    safetensors_path = path / "model.safetensors"
    if safetensors_path.exists():
        try:
            with safe_open(safetensors_path, framework="pt", device="cpu") as f:
                header = f.metadata()
                if header and "meta" in header:
                    embedded = json.loads(header["meta"])
                    for k, v in embedded.items():
                        if k not in config:
                            config[k] = v
        except Exception:
            pass

    model_name = config.get("model_name") or config.get("model", {}).get("name") or "answerdotai/ModernBERT-large"
    local_attention = config.get("local_attention", 16384)
    attn_impl = config.get("attn_implementation", "sdpa")
    head_config = config.get("head_config") or config.get("head") or {"new_dim": 512, "num_layers": 2, "num_task_types": 3}

    # Instantiate backbone
    if (path / "backbone").is_dir():
        backbone = ModernBertModel.from_pretrained(path / "backbone", torch_dtype=torch.float32)
    else:
        if (path / "backbone_config.json").exists():
            backbone_config = ModernBertConfig.from_pretrained(path / "backbone_config.json")
        elif (path / "config.json").exists() and "architectures" in (path / "config.json").read_text():
            backbone_config = ModernBertConfig.from_pretrained(path / "config.json")
        else:
            backbone_config = ModernBertConfig.from_pretrained(model_name)
        if local_attention is not None:
            backbone_config.local_attention = local_attention
        if attn_impl:
            backbone_config._attn_implementation = attn_impl
        backbone = ModernBertModel(backbone_config)

    head = ChoiceHead(hidden_dim=backbone.config.hidden_size, **head_config)

    # Load weights
    if safetensors_path.exists():
        weights = load_file(safetensors_path)
        backbone_weights = {k[len("backbone."):]: v.float() for k, v in weights.items() if k.startswith("backbone.")}
        if backbone_weights:
            backbone.load_state_dict(backbone_weights)
        head_weights = {k[len("head."):]: v.float() for k, v in weights.items() if k.startswith("head.")}
        if head_weights:
            head.load_state_dict(head_weights)
    elif (path / "head.pt").exists():
        head.load_state_dict(torch.load(path / "head.pt", map_location="cpu"))

    network = DeciderNetwork(backbone.float(), head).to(device).eval()

    try:
        tokenizer = AutoTokenizer.from_pretrained(path)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    encoder = Encoder(tokenizer, max_state_tokens or config.get("max_state_tokens", 1024), config.get("max_choice_tokens", 64))
    return Decider(network, encoder, config, device, name=str(model), precision=precision)
