# -*- coding: utf-8 -*-
"""检查慢速分词器与底层 sp 模型的真实编码"""
from transformers import AutoTokenizer

path = r"D:\workspace\xllm\dataset\internlm2-1_8b-reward"
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
print("class:", type(tok).__name__)
print("vocab_size:", tok.vocab_size, "len:", len(tok))
enc = tok.added_tokens_encoder
print("added_tokens_encoder:", enc)

targets = ["<unk>", "<s>", "</s>", "<|reward|>", "