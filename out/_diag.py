# -*- coding: utf-8 -*-
"""诊断：1) 显卡真实可分配显存上限；2) 真实数据管线下训练循环的崩溃点"""
import os, sys
sys.path.insert(0, r'D:\workspace\xllm')
os.chdir(r'D:\workspace\xllm\trainer')
import torch

print("=" * 30, "阶段1：显存可分配上限测试", "=" * 30)
free, total = torch.cuda.mem_get_info()
print(f"驱动报告: total={total/2**30:.2f} GiB, free={free/2**30:.2f} GiB")
blocks = []
try:
    while True:
        blocks.append(torch.empty(64 * 1024 * 1024 // 4, dtype=torch.float32, device='cuda'))
except torch.cuda.OutOfMemoryError:
    pass
alloc_total = len(blocks) * 64 / 1024
print(f"实际可分配(64MiB块): {alloc_total:.2f} GiB, 块数={len(blocks)}")
del blocks
torch.cuda.empty_cache()
free2, _ = torch.cuda.mem_get_info()
print(f"释放后 free={free2/2**30:.2f} GiB")

print("=" * 30, "阶段2：真实数据管线训练循环", "=" * 30)
from torch.utils.data import DataLoader
from model.XLLMModel import XLLMConfig, XLLMForCausalLM
from dataset.lm_dataset import PretrainDataset
from transformers import AutoTokenizer

device = 'cuda:0'
tokenizer = AutoTokenizer.from_pretrained(r'D:\workspace\xllm\model')
config = XLLMConfig(hidden_size=512, num_hidden_layers=8, use_moe=False)
model = XLLMForCausalLM(config).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=5e-4)

ds = PretrainDataset('../dataset/pretrain_t2t_mini.jsonl', tokenizer, max_length=512)
print(f"数据集大小: {len(ds)}")
loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=1, pin_memory=True)

autocast_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16)
it = iter(loader)
for step in range(1, 301):
    try:
        batch = next(it)
    except StopIteration:
        print("数据集遍历完毕"); break
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    try:
        with autocast_ctx:
            res = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = res.loss + res.aux_loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    except Exception as e:
        free3, total3 = torch.cuda.mem_get_info()
        print(f"!!! step {step} 崩溃: {type(e).__name__}: {e}")
        print(f"崩溃时: alloc={torch.cuda.memory_allocated()/2**30:.2f} GiB, "
              f"reserved={torch.cuda.memory_reserved()/2**30:.2f} GiB, free={free3/2**30:.2f} GiB")
        sys.exit(1)
    if step % 25 == 0:
        free3, _ = torch.cuda.mem_get_info()
        print(f"step {step}: loss={loss.item():.4f} alloc={torch.cuda.memory_allocated()/2**30:.2f} GiB "
              f"reserved={torch.cuda.memory_reserved()/2**30:.2f} GiB free={free3/2**30:.2f} GiB")
print("300 步全部完成，训练管线稳定")
