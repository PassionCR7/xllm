import torch

path = r"D:\workspace\xllm\checkpoints\ppo_actor_512_resume.pth"
data = torch.load(path, map_location="cpu", weights_only=False)
print("keys:", list(data.keys()))
print("epoch:", data.get("epoch"))
print("step:", data.get("step"))
print("world_size:", data.get("world_size"))
print("model state_dict count:", len(data.get("model", {})))
