import os
import sys
import torch
import transformers

print("transformers:", transformers.__version__)
print("torch:", torch.__version__)

from transformers import AutoModel, AutoTokenizer, AutoConfig

path = r"D:\workspace\xllm\dataset\internlm2-1_8b-reward"

# 1) config
cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
print("rope_scaling =", repr(cfg.rope_scaling))
print("attn_implementation =", cfg.attn_implementation)

# 2) load model on GPU
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
print("loading reward model on", dev)
model = AutoModel.from_pretrained(path, torch_dtype=torch.float16, trust_remote_code=True)
model = model.to(dev).eval().requires_grad_(False)
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
print("loaded ok, v_head:", model.v_head)

# 3) forward get_score on a tiny conversation
conv = [
    {"role": "user", "content": "1+1=?"},
    {"role": "assistant", "content": "2"},
]
with torch.no_grad():
    score = model.get_score(tok, conv)
print("get_score =", score)
print("PROBE_OK")
