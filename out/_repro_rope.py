# -*- coding: utf-8 -*-
"""定位越界 token：打印完整编码序列与每个 id 对应的 token"""
from transformers import AutoTokenizer

path = r"D:\workspace\xllm\dataset\internlm2-1_8b-reward"
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

chat = [
    {"role": "user", "content": "1+1等于几？"},
    {"role": "assistant", "content": "1+1等于2。"},
]
s = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
ids = tok.encode(s, add_special_tokens=False)
print("共", len(ids), "个 token")
for i in ids:
    try:
        t = tok.convert_ids_to_tokens(i)
    except Exception as e:
        t = f"<转换失败: {e}>"
    flag = " <<<越界" if i >= 92544 else ""
    print(i, repr(t), flag)

# 单独测试 <|reward|> 的编码
print("\n<|reward|> 单独编码:", tok.encode("<|reward|>", add_special_tokens=False))
print("reward_token_id 应为: 92527")
