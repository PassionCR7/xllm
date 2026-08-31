# -*- coding: utf-8 -*-
"""决定性测试：单块连续分配上限（区分硬性上限 vs 碎片）"""
import subprocess
import torch

def smi_free():
    out = subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],
                         capture_output=True, text=True)
    return out.stdout.strip()

print(f"初始驱动 free = {smi_free()} MiB")
print(f"torch mem_get_info free = {torch.cuda.mem_get_info()[0]/2**30:.2f} GiB\n")

# 单块连续分配：从 1GiB 逐步增大，找到最大可分配的单个连续张量
print("--- 单块连续分配测试（无碎片干扰）---")
for gib in [1, 2, 3, 4, 5, 6]:
    torch.cuda.empty_cache()
    try:
        t = torch.empty(int(gib * 2**30) // 4, dtype=torch.float32, device='cuda')
        print(f"单块 {gib} GiB: 成功")
        del t
    except torch.cuda.OutOfMemoryError:
        print(f"单块 {gib} GiB: 失败（驱动 free={smi_free()} MiB）")
        break

# 释放后再测一次，确认无残留
torch.cuda.empty_cache()
print(f"\n结束 驱动 free = {smi_free()} MiB")
