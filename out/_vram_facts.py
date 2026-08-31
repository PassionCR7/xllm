# -*- coding: utf-8 -*-
"""事实收集：逐级分配显存，同时对照驱动层显存计数，定位可用显存上限及去向"""
import subprocess, sys
import torch

def smi():
    out = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.used,memory.free', '--format=csv,noheader,nounits'],
        capture_output=True, text=True)
    return out.stdout.strip()

print(f"torch 报告 total_memory = {torch.cuda.get_device_properties(0).total_memory/2**30:.2f} GiB")
x = torch.zeros(1, device='cuda')
print(f"[阶段0 上下文初始化后] torch_reserved={torch.cuda.memory_reserved()/2**30:.2f} GiB | 驱动: used/free(MiB)={smi()}")

held = []
targets = [512, 1024, 1536, 2048, 2560, 3072]  # MiB 累计目标
try:
    for t in targets:
        cur = len(held) * 64
        while len(held) * 64 < t:
            held.append(torch.empty(64 * 1024 * 1024 // 4, dtype=torch.float32, device='cuda'))
        print(f"[累计 {len(held)*64} MiB] torch_reserved={torch.cuda.memory_reserved()/2**30:.2f} GiB | 驱动: {smi()}")
except torch.cuda.OutOfMemoryError as e:
    print(f"[分配失败于累计 {len(held)*64} MiB] | 驱动: {smi()}")
    print("OOM:", str(e)[:200])

print("释放全部块...")
del held
torch.cuda.empty_cache()
print(f"[释放后] 驱动: {smi()}")
