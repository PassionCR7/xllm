# -*- coding: utf-8 -*-
"""诊断：慢速/快速分词器对特殊 token 的 id 映射（用拼接避免字面量截断）"""
from transformers import AutoTokenizer

path = r"D:\workspace\xllm\dataset\internlm2-1_8b-reward"
IM_S = "<" + "|im_start|" + ">"
IM_E = "<" + "|im_end|" + ">"
RW = "<" + "|reward|" + ">"

for use_fast in (False, True):
    print("=" * 20, "use_fast =", use_fast, "=" * 20)
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=use_fast)
    print("class:", type(tok).__name__, "| vocab_size:", tok.vocab_size, "| len:", len(tok))
    for t in (IM_S, IM_E, RW):
        print("  ", t, "->", tok.convert_tokens_to_ids(t))
    print("  added_tokens_decoder keys:", sorted(tok.added_tokens_decoder.keys()))
