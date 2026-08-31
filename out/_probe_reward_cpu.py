import os
import sys
import torch
import transformers

print("transformers:", transformers.__version__)

from transformers import AutoModel, AutoTokenizer, AutoConfig

path = r"D:\workspace\xllm\dataset\internlm2-1_8b-reward"
dev = "cpu"
print("loading reward model on", dev)
model = AutoModel.from_pretrained(path, torch_dtype=torch.float32, trust_remote_code=True)
model = model.to(dev).eval().requires_grad_(False)
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

conv = [
    {"role": "user", "content": "1+1=?"},
    {"role": "assistant", "content": "2"},
]
conversation_str = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
print("conversation_str =", repr(conversation_str))
input_ids = tok.encode(conversation_str, return_tensors="pt", add_special_tokens=False)
print("input_ids shape:", input_ids.shape)
print("input_ids max:", input_ids.max().item(), "min:", input_ids.min().item())
print("vocab_size:", model.config.vocab_size, "reward_token_id:", model.config.reward_token_id)

# check for out-of-range ids
oob = (input_ids >= model.config.vocab_size).nonzero()
print("out-of-range token ids:", oob.tolist())
ids = input_ids[0].tolist()
print("ids tail:", ids[-5:])
print("last is reward?", ids[-1] == model.config.reward_token_id)

with torch.no_grad():
    score = model.get_score(tok, conv)
print("get_score =", score)
print("PROBE_OK")
