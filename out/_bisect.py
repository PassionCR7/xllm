# -*- coding: utf-8 -*-
"""单配置复现（每次独立进程运行）：真实数据管线下定位崩溃原因
用法: python _bisect.py <A|B|C|D> [steps]
A: bsz=16 workers=1 pin=True 默认后端
B: bsz=16 workers=1 pin=True 仅math后端
C: bsz=16 workers=0 pin=False 仅math后端
D: bsz=8  workers=0 pin=False 仅math后端
"""
import os, sys
sys.path.insert(0, r'D:\workspace\xllm')
os.chdir(r'D:\workspace\xllm\trainer')
import torch

CONF = {
    "A": dict(bsz=16, num_workers=1, pin=True,  math_only=False),
    "B": dict(bsz=16, num_workers=1, pin=True,  math_only=True),
    "C": dict(bsz=16, num_workers=0, pin=False, math_only=True),
    "D": dict(bsz=8,  num_workers=0, pin=False, math_only=True),
}


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else "A"
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    cfg = CONF[key]
    if cfg["math_only"]:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    print(f"=== 配置{key}: {cfg} ===")

    from transformers import AutoTokenizer
    from model.XLLMModel import XLLMConfig, XLLMForCausalLM
    from dataset.lm_dataset import PretrainDataset

    tokenizer = AutoTokenizer.from_pretrained(r'D:\workspace\xllm\model')
    config = XLLMConfig(hidden_size=512, num_hidden_layers=8, use_moe=False)
    model = XLLMForCausalLM(config).to('cuda:0')
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
    ds = PretrainDataset('../dataset/pretrain_t2t_mini.jsonl', tokenizer, max_length=512)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=cfg["bsz"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=cfg["pin"])

    it = iter(loader)
    for step in range(1, steps + 1):
        batch = next(it)
        input_ids = batch["input_ids"].to('cuda:0', non_blocking=True)
        attention_mask = batch["attention_mask"].to('cuda:0', non_blocking=True)
        labels = batch["labels"].to('cuda:0', non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            res = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = res.loss + res.aux_loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 25 == 0:
            free, _ = torch.cuda.mem_get_info()
            print(f"step {step}: loss={loss.item():.4f} "
                  f"alloc={torch.cuda.memory_allocated()/2**30:.2f} GiB 驱动free={free/2**30:.2f} GiB", flush=True)
    print(f"配置{key}: 完成 {steps} 步，稳定")


if __name__ == "__main__":
    main()
