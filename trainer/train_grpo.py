import os
import sys
import re
import gc
import argparse
import warnings
import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer, AutoModel

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.XLLMModel import XLLMConfig
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import (
    Logger,
    is_main_process,
    lm_checkpoint,
    init_distributed_mode,
    setup_seed,
    SkipBatchSampler,
    init_model,
)

warnings.filterwarnings("ignore")


def calculate_rewards(prompts, responses, reward_model, reward_tokenizer):
    def reasoning_model_reward(rewards_tensor):
        pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
        pattern2 = r"^<think>\n.*?\n</think>\n\n<answer>\n.*?\n</answer>$"

        format_rewards = []
        for response in responses:
            matched = re.match(pattern, response, re.S) or re.match(
                pattern2, response, re.S
            )
            format_rewards.append(0.5 if matched else 0.0)
        rewards_tensor += torch.tensor(format_rewards, device=args.device)

        def mark_num(text):
            reward = 0.0
            if text.count("<think>") == 1:
                reward += 0.25
            if text.count("</think>") == 1:
                reward += 0.25
            if text.count("<answer>") == 1:
                reward += 0.25
            if text.count("</answer>") == 1:
                reward += 0.25
            return reward

        rewards_tensor += torch.tensor(
            [mark_num(response) for response in responses], device=args.device
        )
        return rewards_tensor

    rewards = torch.zeros(len(responses), device=args.device)
    if args.reasoning == 1:
        rewards = reasoning_model_reward(rewards)

    with torch.no_grad():
        reward_model_scores = []
        scale = 3.0

        for i, prompt in enumerate(prompts):
            pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
            matches = re.findall(pattern, prompt, re.DOTALL)
            messages = [
                {"role": role, "content": content.strip()} for role, content in matches
            ]

            for j in range(args.num_generations):
                response_idx = i * args.num_generations + j
                response = responses[response_idx]
                tmp_chat = messages + [{"role": "assistant", "content": response}]
                score = reward_model.get_score(reward_tokenizer, tmp_chat)
                score = max(min(score, scale), -scale)

                if args.reasoning == 1:
                    answer_match = re.search(
                        r"<answer>(.*?)</answer>", response, re.DOTALL
                    )
                    if answer_match:
                        answer_content = answer_match.group(1).strip()
                        answer_chat = messages + [
                            {"role": "assistant", "content": answer_content}
                        ]
                        answer_score = reward_model.get_score(
                            reward_tokenizer, answer_chat
                        )
                        answer_score = max(min(answer_score, scale), -scale)
                        score = score * 0.4 + answer_score * 0.6

                reward_model_scores.append(score)

        rewards += torch.tensor(reward_model_scores, device=args.device)

    return rewards


def grpo_train_epoch(
    epoch,
    loader,
    iters,
    ref_model,
    reward_model,
    reward_tokenizer,
    start_step=0,
    wandb=None,
):
    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch["prompt"]

        prompt_inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
            add_special_tokens=False,
        ).to(args.device)

        if args.max_seq_len:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][
                :, -args.max_seq_len :
            ]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][
                :, -args.max_seq_len :
            ]

        with torch.no_grad():
            model_for_gen = (
                model.module if isinstance(model, DistributedDataParallel) else model
            )
            outputs = model_for_gen.generate(
                **prompt_inputs,
                max_new_tokens=args.max_gen_len,
                do_sample=True,
                temperature=0.8,
                num_return_sequences=args.num_generations,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        completion_ids = outputs[:, prompt_inputs["input_ids"].size(1) :]
        # 生成结束后及时回收 KV cache 占用的显存，为后续分块前向留出空间
        torch.cuda.empty_cache()

        def get_per_token_logps(mdl, input_ids, n_keep, chunk_size=4):
            # 分块前向以降低峰值显存：注意力分数矩阵大小为 O(B * seq_len^2)，
            # GRPO 的 batch 被 num_generations 放大（如 2*8=16 条序列），一次性全量前向
            # 在 8GB 显存上会 OOM（16*1602^2*8层≈5GB，另有常驻的 1.8B reward 模型）。
            # 按 chunk_size 分批计算每行 logps 后拼接，逐行语义与原实现完全一致。
            input_ids = (
                input_ids.detach().clone() if input_ids.is_inference() else input_ids
            )
            all_per_token_logps = []
            n_rows = input_ids.size(0)
            for start in range(0, n_rows, chunk_size):
                chunk_ids = input_ids[start : start + chunk_size]
                logits = mdl(
                    input_ids=chunk_ids, logits_to_keep=n_keep + 1
                ).logits[:, :-1, :]
                for logits_row, ids_row in zip(logits, chunk_ids[:, -n_keep:]):
                    ids_row = (
                        ids_row.detach().clone() if ids_row.is_inference() else ids_row
                    )
                    token_logps = torch.gather(
                        logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)
                    ).squeeze(1)
                    all_per_token_logps.append(token_logps)
            return torch.stack(all_per_token_logps)

        # 记录 rollout 时的行为策略（旧策略）logps，用于 GRPO 重要性比率 exp(new - old)
        with torch.no_grad():
            old_per_token_logps = get_per_token_logps(
                model, outputs, completion_ids.size(1)
            )
        per_token_logps = get_per_token_logps(model, outputs, completion_ids.size(1))
        with torch.no_grad():
            ref_per_token_logps = get_per_token_logps(
                ref_model, outputs, completion_ids.size(1)
            )

        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        rewards = calculate_rewards(
            prompts, completions, reward_model, reward_tokenizer
        ).to(args.device)

        grouped_rewards = rewards.view(-1, args.num_generations)
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)
        std_r = grouped_rewards.std(dim=1).repeat_interleave(args.num_generations)
        advantages = torch.clamp((rewards - mean_r) / (std_r + 1e-4), -10, 10)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        is_eos = completion_ids == tokenizer.eos_token_id
        eos_idx = torch.full(
            (is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=args.device
        )
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        completion_mask = (
            torch.arange(is_eos.size(1), device=args.device).expand(is_eos.size(0), -1)
            <= eos_idx.unsqueeze(1)
        ).int()

        kl_div = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(kl_div) - kl_div - 1
        # 重要性比率：exp(新策略logp - 旧策略logp)，旧策略logp在rollout时捕获
        ratio = torch.exp(per_token_logps - old_per_token_logps)
        if args.loss_type == "cispo":
            # ================= CISPO (Clipped Importance Sampling-weight Policy Optimization) =================
            # 出处：MiniMax-M1 (arXiv:2506.13585)，GRPO 系变体，用于解决 PPO-clip 硬裁剪"丢弃 token"的问题。
            # 核心思想：不裁剪 token 的更新（即不 mask 梯度），只裁剪"重要性采样权重"本身；
            #           且裁剪后 .detach()（等价于论文中的 stop-gradient sg()），使梯度只从 logp 流出。
            # 论文公式（token 级 + 组内相对 advantage Â，无价值模型）：
            #   r_t    = π_θ(o_t|q,o_<t) / π_θ_old(o_t|q,o_<t) = exp(new_logp − old_logp)   ← 即上面的 ratio
            #   r̂_t    = clip(r_t, 1−ε_low^IS, 1+ε_high^IS)     ← 实现上只封上界 max=epsilon_high（论文实践不设下界）
            #   L_CISPO= − r̂_t(固定/stop-grad) · Â_t · log π_θ(o_t|q,o_<t) + β·KL
            # 与 PPO-clip 的本质区别：任何 token 的梯度都不会被清零，只限制权重幅度；
            #   低概率的"反思 token"（However/Recheck/Wait 等）不会被丢弃 → 长 CoT 推理更易涌现、熵更稳定。
            clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
            # ① 只对上界截断：clamp(ratio, max=epsilon_high) 等价于 clip(r, −∞, 1+ε_high)
            #   ε_high 即 args.epsilon_high（默认 5.0）；下界不设，符合论文"只调 ε_high^IS"的实践。
            # ② .detach() = stop-gradient：clamped_ratio 被当作常数系数、不回传梯度，
            #   因此下方 per_token_loss 中只有 per_token_logps 参与求导（梯度恒保留，只是幅度受限）。
            per_token_loss = -(
                clamped_ratio * advantages.unsqueeze(1) * per_token_logps
                - args.beta * per_token_kl
            )

            # 逐 token 损失 = -( 固定权重 * advantage * logp - beta*KL )
            #   advantages.unsqueeze(1)：把 [B] 扩成 [B,1]，与 token 维度 [B,T] 广播相乘
            #   外层负号把 -beta*KL 变成 +beta*KL：即对偏离参考策略 ref_model 的行为施加惩罚
            #   注：原论文 CISPO 不含 KL 项；此处保留 beta*KL 是复现代码的工程取舍，可传 --beta 0 关闭。
            # -------- 对照参考：若此处改为手动写 PPO-clip（而不走下方 else 分支），对应代码如下 --------
            # # PPO-clip 目标（DeepSeekMath GRPO 的核心）：
            # #   L = min( r*Â,  clip(r, 1-eps, 1+eps)*Â )  - beta*KL
            # # min 双分支的隐式 mask：当 Â>0 且 r>1+eps（或 Â<0 且 r<1-eps）时，min 取到不依赖 θ 的常数分支，
            # #   该 token 的梯度被清零（被"丢弃"）——这正是 CISPO 要避免的：CISPO 只封顶权重、从不把梯度归零。
            # clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
            # per_token_loss1 = ratio * advantages.unsqueeze(1)          # 未裁剪项 r*Â
            # per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)  # 裁剪项 clip(r)*Â
            # per_token_loss = -(torch.min(per_token_loss1, per_token_loss2) - args.beta * per_token_kl)
        else:
            # GRPO（PPO-clip 风格，与上方"注释对照代码"逻辑等价，通过 --loss_type grpo 启用）：
            #   min(ratio, clipped(ratio))*Â - beta*KL；超界 token 的梯度被 mask 清零（与 CISPO 的关键差异）
            clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
            per_token_loss1 = ratio * advantages.unsqueeze(1)
            per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
            per_token_loss = -(
                torch.min(per_token_loss1, per_token_loss2) - args.beta * per_token_kl
            )

        loss = (
            (per_token_loss * completion_mask).sum(dim=1)
            / completion_mask.sum(dim=1).clamp(min=1)
        ).mean() / args.accumulation_steps
        loss.backward()

        if step % args.accumulation_steps == 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if step % args.log_interval == 0 or step == iters:
            policy_loss_val = loss.item() * args.accumulation_steps
            avg_reward_val = rewards.mean().item()
            avg_len_val = completion_mask.sum(dim=1).float().mean().item()
            current_lr = optimizer.param_groups[0]["lr"]

            Logger(
                f"Epoch: {epoch + 1}, Step: {step}/{iters}, "
                f"Actor Loss: {policy_loss_val:.6f}, Reward: {avg_reward_val:.6f}, "
                f"Avg Response Len: {avg_len_val:.2f}, LR: {current_lr:.2e}"
            )

            if wandb and is_main_process():
                wandb.log(
                    {
                        "policy_loss": policy_loss_val,
                        "reward": avg_reward_val,
                        "avg_response_len": avg_len_val,
                        "advantages_mean": advantages.mean().item(),
                        "learning_rate": current_lr,
                    }
                )

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = "_moe" if lm_config.use_moe else ""
            ckp = f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth"
            state_dict = (
                model.module.state_dict()
                if isinstance(model, DistributedDataParallel)
                else model.state_dict()
            )
            torch.save({k: v.half() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(
                lm_config,
                weight=args.save_weight,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=step,
                wandb=wandb,
                save_dir="../checkpoints",
                scheduler=scheduler,
            )
            model.train()

        del (
            prompt_inputs,
            outputs,
            completion_ids,
            per_token_logps,
            ref_per_token_logps,
            old_per_token_logps,
        )
        del (
            completions,
            rewards,
            grouped_rewards,
            mean_r,
            std_r,
            advantages,
            completion_mask,
        )
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="XLLM GRPO (Group Relative Policy Optimization)"
    )

    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument(
        "--save_weight", default="grpo", type=str, help="保存权重的前缀名"
    )
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=8e-8, help="初始学习率")

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="训练设备",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=1, help="数据加载线程数")

    parser.add_argument(
        "--accumulation_steps", type=int, default=1, help="梯度累积步数"
    )
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")

    parser.add_argument("--hidden_size", default=512, type=int, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", default=8, type=int, help="隐藏层数量")
    parser.add_argument(
        "--use_moe",
        default=0,
        type=int,
        choices=[0, 1],
        help="是否使用MoE架构（0=否，1=是）",
    )

    parser.add_argument("--max_seq_len", default=66, type=int, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1536, help="生成的最大长度")

    parser.add_argument(
        "--data_path",
        type=str,
        default="../dataset/rlaif.jsonl",
        help="RLAIF数据路径",
    )
    parser.add_argument(
        "--num_generations", type=int, default=8, help="每个prompt生成的样本数"
    )
    parser.add_argument("--beta", type=float, default=0.02, help="KL惩罚系数")
    parser.add_argument(
        "--loss_type",
        type=str,
        default="cispo",
        choices=["grpo", "cispo"],
        help="loss类型（cispo=minimind默认，grpo=PPO-clip风格）",
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.2, help="GRPO的PPO clip epsilon"
    )
    parser.add_argument(
        "--epsilon_high", type=float, default=5.0, help="epsilon上界"
    )
    parser.add_argument(
        "--reasoning",
        type=int,
        default=1,
        choices=[0, 1],
        help="推理模型类型（0=普通模型，1=推理模型）",
    )
    parser.add_argument(
        "--reward_model_path",
        type=str,
        default="../dataset/internlm2-1_8b-reward",
        help="Reward模型路径",
    )
    parser.add_argument(
        "--from_resume",
        default=0,
        type=int,
        choices=[0, 1],
        help="是否自动检测&续训（0=否，1=是）",
    )

    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument(
        "--wandb_project", type=str, default="XLLM-GRPO", help="wandb项目名"
    )
    args = parser.parse_args()

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = XLLMConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_position_embeddings=args.max_seq_len + args.max_gen_len,
        use_moe=bool(args.use_moe),
    )
    ckp_data = (
        lm_checkpoint(lm_config, weight=args.save_weight, save_dir="../checkpoints")
        if args.from_resume == 1
        else None
    )

    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb

        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        resume = "must" if wandb_id else None
        wandb_run_name = f"XLLM-GRPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(
            project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume
        )

    base_weight = "reason" if args.reasoning == 1 else "full_sft"

    model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    tokenizer.padding_side = "left"  # GRPO 在线生成需要左侧 padding，否则右 padding 会导致生成错位
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)

    reward_model = AutoModel.from_pretrained(
        args.reward_model_path, torch_dtype=torch.float16, trust_remote_code=True
    )
    reward_model = reward_model.to(args.device).eval().requires_grad_(False)
    reward_tokenizer = AutoTokenizer.from_pretrained(
        args.reward_model_path, trust_remote_code=True
    )

    train_ds = RLAIFDataset(
        args.data_path, tokenizer, max_length=lm_config.max_position_embeddings
    )
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    loader_for_count = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler
    )
    iters = len(loader_for_count)
    total_optimizer_steps = max(1, (iters // args.accumulation_steps) * args.epochs)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10
    )

    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data["model"])
        optimizer.load_state_dict(ckp_data["optimizer"])
        scheduler.load_state_dict(ckp_data["scheduler"])
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)

    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)

        if epoch == start_epoch and start_step > 0:
            batch_sampler = SkipBatchSampler(
                train_sampler or range(len(train_ds)), args.batch_size, start_step
            )
            loader = DataLoader(
                train_ds,
                batch_sampler=batch_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            Logger(
                f"Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始"
            )
            grpo_train_epoch(
                epoch,
                loader,
                len(loader) + start_step,
                ref_model,
                reward_model,
                reward_tokenizer,
                start_step,
                wandb,
            )
        else:
            loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                pin_memory=True,
                drop_last=False,
                shuffle=(train_sampler is None),
                num_workers=args.num_workers,
                sampler=train_sampler,
            )
            grpo_train_epoch(
                epoch,
                loader,
                len(loader),
                ref_model,
                reward_model,
                reward_tokenizer,
                0,
                wandb,
            )
