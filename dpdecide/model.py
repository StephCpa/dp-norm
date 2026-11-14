"""A causal LM read out as a typed decision model: one forward pass, logits over the 26 letters
' A'..' Z' at the last prompt token, masked beyond each row's option count. Encoding follows
eve-rlcd exactly (right padding, left truncation that keeps BOS and the tail), so eve-rlcd warmup
checkpoints load here unchanged."""
from __future__ import annotations

import json
import os
from contextlib import nullcontext
from pathlib import Path

import torch

from dpdecide.schema import LETTERS, MAX_CHOICES, NEG, Question, render_prompt


def letter_token_ids(tok) -> list[int]:
    ids = []
    for letter in LETTERS:
        t = tok.encode(" " + letter, add_special_tokens=False)
        if len(t) != 1:
            raise ValueError(f"' {letter}' is not a single token: {t}")
        ids.append(t[0])
    if len(set(ids)) != MAX_CHOICES:
        raise ValueError("letter token ids collide")
    return ids


def adds_bos(tok) -> bool:
    bos = getattr(tok, "bos_token_id", None)
    if bos is None:
        return False
    ids = tok.encode("x", add_special_tokens=True)
    return len(ids) > 0 and ids[0] == bos


def mask_beyond_k(logits: torch.Tensor, ks) -> torch.Tensor:
    k = torch.as_tensor(ks, dtype=torch.long, device=logits.device)
    pos = torch.arange(MAX_CHOICES, device=logits.device)[None, :]
    return logits.masked_fill(pos >= k[:, None], NEG)


class DecisionLM:
    def __init__(self, model, tok, lora: bool, base: str | None, device: str):
        self.model, self.tok, self.lora, self.base, self.device = model, tok, lora, base, device
        self.letters = letter_token_ids(tok)
        self.prepend_bos = adds_bos(tok)
        pad = getattr(tok, "pad_token_id", None)
        self.pad_id = pad if pad is not None else tok.eos_token_id
        if self.pad_id is None:
            raise ValueError("tokenizer has neither a pad nor an eos token")

    # ------------------------------------------------------------------ loading / saving
    @classmethod
    def load(cls, path_or_id: str, device: str = "cuda", lora: bool = False, lora_r: int = 32,
             grad_checkpointing: bool = False) -> "DecisionLM":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        path_or_id = os.fspath(path_or_id)
        adapter = os.path.isfile(os.path.join(path_or_id, "adapter_config.json"))
        record = {}
        if os.path.isfile(os.path.join(path_or_id, "policy.json")):
            record = json.loads(Path(path_or_id, "policy.json").read_text())
        base = record.get("base") if adapter else path_or_id
        tok = AutoTokenizer.from_pretrained(path_or_id)
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32)
        if any(p.dtype != torch.float32 for p in model.parameters()):
            raise RuntimeError("master weights must be fp32")
        if grad_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.config.use_cache = False
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, path_or_id, is_trainable=True)
        elif lora:
            from peft import LoraConfig, get_peft_model
            model = get_peft_model(model, LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.0,
                                                     target_modules="all-linear"))
        model.to(device)
        return cls(model, tok, lora=adapter or lora, base=base if (adapter or lora) else None, device=device)

    def save(self, out_dir: str, meta: dict) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(out, safe_serialization=True)
        self.tok.save_pretrained(out)
        (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
        # eve-rlcd-compatible record, so its tools can read a full fine-tune saved here
        (out / "policy.json").write_text(json.dumps({"backend": "hf-decoder", "lora": self.lora, "base": self.base,
                                                     "prepend_bos": self.prepend_bos}, indent=2))

    # ------------------------------------------------------------------ forward
    def hf_model(self):
        return self.model.get_base_model() if self.lora else self.model

    def body(self):
        m = self.hf_model()
        get = getattr(m, "get_decoder", None)
        b = get() if callable(get) else None
        return b if b is not None and b is not m else m.model

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [p for p in self.model.parameters() if p.requires_grad]

    def encode(self, questions: list[Question], max_len: int):
        prefix = [self.tok.bos_token_id] if self.prepend_bos else []
        room = max_len - len(prefix)
        if room <= 0:
            raise ValueError("max_len leaves no room for the prompt")
        seqs = [prefix + self.tok.encode(render_prompt(q), add_special_tokens=False)[-room:] for q in questions]
        width = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        last = torch.zeros(len(seqs), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
            mask[i, :len(s)] = 1
            last[i] = len(s) - 1
        return ids.to(self.device), mask.to(self.device), last.to(self.device)

    def logits(self, questions: list[Question], max_len: int = 512) -> torch.Tensor:
        """fp32 decision logits (B, 26), NEG at and beyond each row's option count."""
        on_cuda = torch.device(self.device).type == "cuda"
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if on_cuda else nullcontext()
        ids, mask, last = self.encode(questions, max_len)
        with ctx:
            hidden = self.body()(input_ids=ids, attention_mask=mask, use_cache=False)[0]
        rows = hidden[torch.arange(hidden.size(0), device=hidden.device), last]
        with torch.autocast(device_type=rows.device.type, enabled=False):
            head = self.hf_model().get_output_embeddings()
            idx = torch.as_tensor(self.letters, device=head.weight.device)
            out = rows.float() @ head.weight[idx].float().t()
            bias = getattr(head, "bias", None)
            if bias is not None:
                out = out + bias[idx].float()
        return mask_beyond_k(out, [q.k for q in questions])

    def train(self, mode: bool = True):
        self.model.train(mode)
        return self

    def eval(self):
        return self.train(False)

    @torch.no_grad()
    def predict(self, questions: list[Question], max_len: int = 512, batch_size: int = 32) -> list[list[float]]:
        was = self.model.training
        self.eval()
        rows = []
        try:
            for i in range(0, len(questions), batch_size):
                b = questions[i:i + batch_size]
                for q, r in zip(b, self.logits(b, max_len).cpu()):
                    rows.append(r[:q.k].tolist())
        finally:
            self.train(was)
        return rows
