import re, os

path = r"D:\workspace\xllm\out\_ppo_log_full.txt"
with open(path, encoding="utf-8", errors="replace") as f:
    content = f.read()

print("file bytes:", os.path.getsize(path))
print("contains 'Task has finished':", "Task has finished" in content)

# find all step logs
steps = re.findall(r"Step: (\d+)/9751", content)
if steps:
    steps = [int(s) for s in steps]
    print("max step reached:", max(steps))
    print("num step lines:", len(steps))
else:
    print("no step logs found")

# check for completion / epoch-end markers
for marker in ["Epoch 1 finished", "Epoch 2", "epochs done", "Training complete", "Traceback", "Error", "CUDA error", "OutOfMemory", "saved"]:
    idx = content.rfind(marker)
    if idx != -1:
        print(f"marker '{marker}' found at tail-offset {len(content)-idx}")

# last 800 chars
print("===== LAST 800 CHARS =====")
print(content[-800:])
