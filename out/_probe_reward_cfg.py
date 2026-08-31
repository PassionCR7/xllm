import sys
import transformers
print("transformers version:", transformers.__version__)
print("python:", sys.version)
print("HF cache modules:", transformers.utils.hub.TRANSFORMERS_CACHE if hasattr(transformers.utils.hub, "TRANSFORMERS_CACHE") else "n/a")

from transformers import AutoConfig

path = r"D:\workspace\xllm\dataset\internlm2-1_8b-reward"
cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
print("rope_scaling =", repr(cfg.rope_scaling), "type:", type(cfg.rope_scaling))
print("attn_implementation =", cfg.attn_implementation)
print("rope_theta =", cfg.rope_theta)
print("max_position_embeddings =", cfg.max_position_embeddings)
print("torch_dtype =", getattr(cfg, "torch_dtype", None))
