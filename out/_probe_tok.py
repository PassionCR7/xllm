from transformers import AutoTokenizer

path = r"D:\workspace\xllm\dataset\internlm2-1_8b-reward"
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
print("tokenizer class:", type(tok).__name__)
print("vocab_size:", tok.vocab_size)
v = tok.get_vocab()
print("len(get_vocab()):", len(v))
print("max id in vocab:", max(v.values()) if v else None)

for s in ["<s>", "</s>", "<|im_start|>", "<|im_end|>", "<|reward|>", "<|im_sep|>", "<|User|>", "<|Assistant|>"]:
    print(f"  {s!r:20} -> id {tok.convert_tokens_to_ids(s)}")

print("all_special_tokens:", tok.all_special_tokens)
print("all_special_ids:", tok.all_special_ids)

# what do we get when encoding the conversation
conversation_str = '<s><|im_start|>user\n1+1=?<|im_end|>\n<|im_start|>assistant\n2<|im_end|>\n<|reward|>'
ids = tok.encode(conversation_str, add_special_tokens=False)
print("encoded ids:", ids)
print("decoded:", tok.decode(ids))
