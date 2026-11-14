"""Build a tiny random Qwen3 decision model plus a word-level tokenizer, entirely offline, for CPU
smoke tests of the training/eval/audit pipeline. Not a real model: it only checks that the code runs
and that the privacy plumbing behaves.

  python scripts/make_tiny_model.py --out /tmp/tiny
"""
import argparse
import re

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from dpdecide.schema import LETTERS, render_prompt
from dpdecide.synth import tickets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    corpus = " ".join(render_prompt(q) for q in tickets(400, seed=123))
    words = sorted(set(re.findall(r"\w+|[^\w\s]", corpus)) | set(LETTERS))
    specials = ["<pad>", "<bos>", "<eos>", "<unk>"]
    vocab = {w: i for i, w in enumerate(specials + words)}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()])
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", bos_token="<bos>", eos_token="<eos>",
                                  unk_token="<unk>")
    cfg = Qwen3Config(vocab_size=len(vocab), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=1024,
                      tie_word_embeddings=True, pad_token_id=0, bos_token_id=1, eos_token_id=2)
    model = Qwen3ForCausalLM(cfg)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"tiny model: vocab {len(vocab)}, params {sum(p.numel() for p in model.parameters()):,} -> {args.out}")


if __name__ == "__main__":
    main()
