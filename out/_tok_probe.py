# -*- coding: utf-8 -*-
"""检查分词器底层词表与 added_tokens 映射，定位编码越界根因"""
from transformers import AutoTokenizer

path = r"D:\workspace\xllm\dataset\internlm2-1_8b-reward"
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
print("tokenizer class:", type(tok).__name__)

sp = getattr(tok, "sp_model", None)
if sp is not None:
    print("sp_model vocab size:", sp.GetPieceSize())
else:
    print("no sp_model (Fast tokenizer); vocab size:", tok.vocab_size)
    vocab = tok.get_vocab()
    print("get_vocab() len:", len(vocab))
    print("词表尾部 12 项（按 id 排序）:")
    for name, i in sorted(vocab.items(), key=lambda kv: kv[1])[-12:]:
        print("  ", i, "->", repr(name))

print("\nadded_tokens_encoder:", getattr(tok, "added_tokens_encoder", None))

print("\nconvert_tokens_to_ids:")
tokens = ["<unk>", "<s>", "</s>", "<|reward|>", "<|im_start|>", "<|im_end|>"]
for t in tokens:
    print("  ", t, "->", tok.convert_tokens_to_ids(t))
print("\ngetattr reward_token_id:", getattr(tok, "reward_token_id", None))
