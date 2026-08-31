# 本代码实现：PPO‑Clip，RLAIF范式，没有 GAE，使用整条序列终点单标量奖励

# PPO 和 SFT/DPO 最大区别：PPO 是在线生成，不能预先 tokenize 答案
# SFT/DPO 是离线已经有正确回答，直接 tokenize；PPO 只有 prompt，回答由 Actor 实时 rollout 生成
#
#
import os
import sys

# 📚 Python模块系统
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse  # 命令行参数解析
import re  # 正则表达式，用于奖励计算
import warnings  # 警告控制
import torch  # PyTorch深度学习框架
import torch.distributed as dist  # 分布式训练支持
import torch.nn.functional as F  # 神经网络函数
from transformers import AutoTokenizer  # HuggingFace分词器
from contextlib import nullcontext  # 上下文管理器
from torch import optim, nn  # 优化器和神经网络
from torch.nn.parallel import DistributedDataParallel  # 分布式并行
from torch.utils.data import DataLoader, DistributedSampler  # 数据加载
from torch.nn.utils import clip_grad_norm_  # 梯度裁剪
from torch.optim.lr_scheduler import CosineAnnealingLR  # 余弦退火学习率调度
from transformers import AutoModel  # HuggingFace模型加载
from model.XLLMModel import XLLMConfig, XLLMForCausalLM  # MiniMind模型
from dataset.lm_dataset import RLAIFDataset  # RL数据集
from trainer.trainer_utils import (  # 训练工具函数
    Logger,
    is_main_process,
    lm_checkpoint,
    init_distributed_mode,
    setup_seed,
    SkipBatchSampler,
    init_model,
)

warnings.filterwarnings("ignore")
# ==========Critic Model部分==========


class CriticModel(XLLMForCausalLM):
    def __init__(self, params):
        super().__init__(params)
        # 价值头，用于输出每个token位置的状态价值
        self.value_head = nn.Linear(params.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )
        hidden_states = self.model.norm(outputs[0])

        values = self.value_head(hidden_states).squeeze(-1)
        return values


# ==========奖励计算部分==========
def calculate_rewards(prompts, responses, reward_model, reward_tokenizer):
    def reasoning_model_reward(rewards):
        # 使用正则表达式匹配思考-回答格式
        pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
        # 多了一个\n，考虑到think和answer之间有空行的情况
        pattern2 = r"^<think>\n.*?\n</think>\n\n<answer>\n.*?\n</answer>$"
        # 通过正则表达式计算奖励，如果回答符合格式则奖励0.5，否则0.0
        matches_pattern = [re.match(pattern, response, re.S) for response in responses]
        matches_pattern2 = [
            re.match(pattern2, response, re.S) for response in responses
        ]

        format_rewards = []
        for match_pattern, match_pattern2 in zip(matches_pattern, matches_pattern2):
            if match_pattern:
                format_rewards.append(0.5)
            elif match_pattern2:
                format_rewards.append(0.5)
            else:
                format_rewards.append(0.0)
        rewards += torch.tensor(format_rewards, device=args.device)

        def mark_num(text):
            reward = 0
            if text.count("<think>") == 1:
                reward += 0.25
            if text.count("</think>") == 1:
                reward += 0.25
            if text.count("<answer>") == 1:
                reward += 0.25
            if text.count("</answer>") == 1:
                reward += 0.25
            return reward

        mark_rewards = [mark_num(response) for response in responses]
        rewards += torch.tensor(mark_rewards, device=args.device)
        return rewards

    rewards = torch.zeros(len(responses), device=args.device)

    if args.reasoning == 1:
        rewards = reasoning_model_reward(rewards)
    # ==========Reward模型评分部分==========
    with torch.no_grad():
        reward_model_scores = []
        for prompt, response in zip(prompts, responses):
            pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
            matches = re.findall(pattern, prompt, re.DOTALL)
            messages = [
                {"role": role, "content": content.strip()} for role, content in matches
            ]

            tmp_chat = messages + [{"role": "assistant", "content": response}]
            score = reward_model.get_score(
                reward_tokenizer, tmp_chat
            )  # ！修正：原get_reward(tmp_chat, reward_tokenizer)方法名和参数顺序错误

            scale = 3.0
            score = max(min(score, scale), -scale)

            if args.reasoning == 1:
                answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
                if answer_match:
                    answer_content = answer_match.group(1).strip()
                    # 对answer内容单独计算reward
                    tmp_chat = messages + [
                        {"role": "assistant", "content": answer_content}
                    ]
                    answer_score = reward_model.get_score(reward_tokenizer, tmp_chat)
                    answer_score = max(min(answer_score, scale), -scale)
                    # 📚 加权组合
                    score = score * 0.4 + answer_score * 0.6
            reward_model_scores.append(score)

        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores

    return rewards


# ==========PPO训练一个Epoch部分==========
def ppo_train_epoch(
    epoch,
    loader,
    iters,
    old_actor_model,
    ref_model,
    actor_scheduler,
    critic_scheduler,
    reward_model,
    reward_tokenizer,
    start_step=0,
    wandb=None,
):
    # 切换actor和critic模型到训练模式
    actor_model.train()
    critic_model.train()

    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch["prompt"]
        #将 prompt 字符串转 input_ids、attention_mask；这里 padding_side 是 left（左 padding，生成任务要求）
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_seq_len,
        ).to(args.device)
        #得到这条 prompt 的 token 数量 N_prompt
        #前 N_prompt 个 token 是上下文，后面是模型将要生成的部分
        prompt_lengths = enc.attention_mask.sum(dim=1)

        with torch.no_grad():
            model_for_gen = (
                actor_model.module
                if isinstance(actor_model, DistributedDataParallel)
                else actor_model
            )

#————————————————————————————————————————————————————————————————————————————————————————————————————
# gen_out完整序列(prompt+response)
#         ├─送入old_actor(no_grad) → old_logp（response片段）
#         ├─送入ref_model(no_grad) → ref_logp（response片段）
#         └─送入actor(可训练) → actor_logp（response片段，带梯度）
#                 ↓
# ratio = exp(actor_logp‑old_logp)
#                 ↓
# PPO‑clip policy_loss
#                 ↓
# value_loss(MSE(values,rewards)) + kl_ref惩罚
#                 ↓
# total loss.backward()
#                 ↓
# 仅更新 actor_model、critic_model #old_actor_model / ref_model 参数完全不变
#————————————————————————————————————————————————————————————————————————————————————————————————————            
            #Actor 模型在线 generate（rollout），gen_out.shape = [1, N_prompt + N_gen]
            #前N_prompt：原始 prompt tokens（多轮对话上下文）；
            #后面N_gen：Actor 模型采样生成出来的 token 序列，就是对 LBC 那道 abcd 大题的模型输出。
            #因为temperature=0.8，每次运行生成的回答文本不一样，存在探索性，这是强化学习 rollout。
            gen_out = model_for_gen.generate(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                max_new_tokens=args.max_gen_len,
                do_sample=True,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # 解码得到 response_text，只截取prompt_lengths[i]:之后的部分，丢掉 prompt 上下文
        responses_text = [
            tokenizer.decode(gen_out[i, prompt_lengths[i] :], skip_special_tokens=True)
            for i in range(len(prompts))
        ]

        # 计算这条样本的总 reward：分两部分计算奖励（args.reasoning=1开启推理格式奖励）
        #（1）格式规则奖励
        # 正则匹配是否完整\n…\n…格式，匹配到 + 0.5；
        # 统计 4 个标签'   '每个恰好出现一次，每个 + 0.25，最多 + 1.0；
        # 如果模型输出标签乱、缺失，这部分奖励直接降低。
        #（2）Reward 模型打分
        #把原始 prompt 上下文 + 模型刚刚生成的 response_text 组装成 chat 对话送入 reward_model；
        #推理模式：提取''内部的解答文本单独打分；总 score = 完整回复分数*0.4 + answer 内容分数*0.6；
        #score 截断 [-3, 3]；

        #总奖励 = 格式奖励 + reward_model 分数 → 得到一个标量 reward（单条样本，1 个数值）
        rewards = calculate_rewards(
            prompts, responses_text, reward_model, reward_tokenizer
        )

        # 创建一个mask，用于标记哪些位置上是有效token
        full_mask = (gen_out != tokenizer.pad_token_id).long()
        # Critic 对完整序列（prompt + 生成回答）每个 token 输出价值
        value_seq = critic_model(input_ids=gen_out, attention_mask=full_mask)
        # 取序列最后一个有效 token 的 value作为整条样本的状态价值 V
        last_indices = full_mask.sum(dim=1) - 1
        values = value_seq[torch.arange(len(last_indices)), last_indices]
        # 优势函数 A = R - V；无 GAE 简化版本。
        advantages = rewards - values.detach()  # [B]

        # 计算actor log，表示actor对这个答案的“信心”
        # 先生成logits
        # 计算 log 概率，只对生成 response 部分计算
        logits = actor_model(
            input_ids=gen_out, attention_mask=full_mask
        ).logits  # [B, L, V]
        # label是生成的token序列，去掉第一个token（因为logits是预测下一个token的概率）
        labels = gen_out[:, 1:].clone()
        # 使用log_softmax计算log概率
        logp_tokens = (
            F.log_softmax(logits[:, :-1, :], dim=-1)
            .gather(2, labels.unsqueeze(-1))
            .squeeze(-1)
        )  # [B, L-1]
        seq_len = gen_out.size(1) - 1
        #屏蔽前面多轮历史对话部分，只对模型自己生成出来的LBC解答那一段token计算log概率,把prompts部分的mask掉
        resp_mask = torch.arange(seq_len, device=gen_out.device).unsqueeze(
            0
        ) >= prompt_lengths.unsqueeze(1)

        final_mask = resp_mask & (~labels.eq(tokenizer.pad_token_id))
        # 把所有回答部分的log概率加起来，得到每条序列的总log概率
        actor_logp = (logp_tokens * final_mask).sum(dim=1)

        # 计算old和ref log的概率
        # old用于防止策略更新过大，ref用于计算KL惩罚，防止模型忘本
        with torch.no_grad():
            #old_actor：上一版本的 actor 权重，计算旧策略下该生成序列的 logp，用于 PPO‑clip 重要性采样
            #返回logits：[B,序列长度,词表大小V]。每个位置是模型对下一个 token 的原始得分，还没有做 softmax
            #(用 log 概率：避免连乘小数下溢，乘法转为加法)
            old_logits = old_actor_model(
                input_ids=gen_out, attention_mask=full_mask
            ).logits  # [B, P+R, V]
            old_logp_tokens = (
                F.log_softmax(old_logits[:, :-1], dim=-1)
                .gather(2, labels.unsqueeze(-1))
                .squeeze(-1)
            )  # [B, P+R-1]
            old_logp = (old_logp_tokens * final_mask).sum(dim=1)  # [B]

            #ref_model：原始 SFT 基座，计算 ref_logp，用于 KL 惩罚，防止模型学偏，彻底忘掉 SFT 知识
            #计算 SFT 参考模型，对同一份 rollout 生成回答 y的联合对数概率
            #依靠final_mask屏蔽 prompt 部分，只累加 response 片段
            ref_logits = ref_model(
                input_ids=gen_out, attention_mask=full_mask
            ).logits  # [B, P+R, V]
            ref_logp_tokens = (
                F.log_softmax(ref_logits[:, :-1], dim=-1)
                .gather(2, labels.unsqueeze(-1))
                .squeeze(-1)
            )  # [B, P+R-1]
            ref_logp = (ref_logp_tokens * final_mask).sum(dim=1)  # [B]

        # 计算KL散度和ratio
        # 近似KL，用来监控新旧策略差异
        kl = (actor_logp - old_logp).mean()
        #kl_ref参与 loss 的惩罚项
        kl_ref = (actor_logp - ref_logp).mean()
        #ratio = torch.exp(actor_logp‑old_logp)，概率比值
        #新策略相比旧策略，有多偏好当前这条 rollout 样本
        # ratio>1：新策略更喜欢输出这条回答；
        # ratio<1：新策略更不喜欢输出这条回答。
        ratio = torch.exp(actor_logp - old_logp)  # [B]

        # PPO‑Clip 代理目标公式(policy_loss只对 actor 模型产生梯度，old_actor 没有梯度)
        #surr1是原始未裁剪代理目标 
            #- 如果advantages>0(这条回答奖励很高)，希望增大 ratio，提高新策略生成该样本概率；
            # -如果advantages<0(坏样本),希望减小ratio，降低生成该样本概率。
        surr1 = ratio * advantages  # [B] 
        #surr2是裁剪后的代理目标
        surr2 = (
            #把 ratio 强行限制在区间[1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon]
            #防止单次梯度更新，新旧策略差距爆炸，保证 “近端”
            torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon)
            * advantages
        )  # [B]
        #当更新会超出允许区间时，用裁剪后的值抑制梯度；在区间内使用原始 ratio
        policy_loss = -torch.min(surr1, surr2).mean()

        # 价值函数损失（MSE 损失，训练 Critic，让预测 value 逼近真实 reward）
        #values是Critic 网络预测整条序列的状态价值
        #rewards是rollout 结束拿到的真实标量奖励
        value_loss = F.mse_loss(values, rewards)
        # 总损失
        #policy_loss：PPO‑clip（Actor） 策略损失，目标：提升高 advantage 回答的概率，抑制低 advantage
        #args.vf_coef *value_loss：Critic 的 MSE，让 Critic 预测的 value 尽量逼近真实拿到的 reward
        #args.kl_coef * kl_ref：若actor策略偏离原始SFT基座，actor_logp远大于ref_logp，kl_ref会变大，给总loss增加惩罚
        loss = policy_loss + args.vf_coef * value_loss + args.kl_coef * kl_ref  # scalar
        #只有actor_model、critic_model产生梯度
        loss.backward()

        # 更新参数
        if step % args.accumulation_steps == 0:
            clip_grad_norm_(actor_model.parameters(), args.grad_clip)
            clip_grad_norm_(critic_model.parameters(), args.grad_clip)
            actor_optimizer.step()
            critic_optimizer.step()
            actor_scheduler.step()
            critic_scheduler.step()
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()

        # 📚 日志记录
        if is_main_process() and (step % args.log_interval == 0 or step == iters):
            response_ids = gen_out[:, enc.input_ids.shape[1] :]
            is_eos = response_ids == tokenizer.eos_token_id
            eos_indices = torch.argmax(is_eos.int(), dim=1)
            has_eos = is_eos.any(dim=1)
            lengths = torch.where(
                has_eos,
                eos_indices + 1,
                torch.tensor(response_ids.shape[1], device=is_eos.device),
            )
            avg_len = lengths.float().mean()

            actor_loss_val = policy_loss.item()
            critic_loss_val = value_loss.item()
            reward_val = rewards.mean().item()
            kl_val = kl.item()
            kl_ref_val = kl_ref.item()
            avg_len_val = avg_len.item()
            actor_lr = actor_optimizer.param_groups[0]["lr"]
            critic_lr = critic_optimizer.param_groups[0]["lr"]

            if wandb is not None:
                wandb.log(
                    {
                        "actor_loss": actor_loss_val,
                        "critic_loss": critic_loss_val,
                        "reward": reward_val,
                        "kl": kl_val,
                        "kl_ref": kl_ref_val,
                        "avg_response_len": avg_len_val,
                        "actor_lr": actor_lr,
                    }
                )

            Logger(
                f"Epoch: {epoch + 1}, Step: {step}/{iters}, "
                f"Actor Loss: {actor_loss_val:.6f}, Critic Loss: {critic_loss_val:.6f}, "
                f"Reward: {reward_val:.6f}, KL: {kl_val:.6f}, KL_ref: {kl_ref_val:.6f}, "
                f"Avg Response Len: {avg_len_val:.2f}, Actor LR: {actor_lr:.2e}, Critic LR: {critic_lr:.2e}"
            )

        # 📚 更新old actor
        if step % args.update_old_actor_freq == 0:
            state_dict = (
                actor_model.module.state_dict()
                if isinstance(actor_model, DistributedDataParallel)
                else actor_model.state_dict()
            )
            old_actor_model.load_state_dict(
                {k: v.detach().cpu() for k, v in state_dict.items()}
            )
            old_actor_model.to(args.device)

        # 📚 模型保存
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            actor_model.eval()
            moe_suffix = "_moe" if lm_config.use_moe else ""
            ckp = f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth"
            actor_state = (
                actor_model.module.state_dict()
                if isinstance(actor_model, DistributedDataParallel)
                else actor_model.state_dict()
            )
            torch.save({k: v.half() for k, v in actor_state.items()}, ckp)

            # 使用 lm_checkpoint 保存完整状态（包括 critic）
            lm_checkpoint(
                lm_config,
                weight=args.save_weight,
                model=actor_model,
                optimizer=actor_optimizer,
                epoch=epoch,
                step=step,
                wandb=wandb,
                save_dir="../checkpoints",
                scheduler=actor_scheduler,
                critic_model=critic_model,
                critic_optimizer=critic_optimizer,
                critic_scheduler=critic_scheduler,
            )
            actor_model.train()


if __name__ == "__main__":
    """
    PPO主函数：近端策略优化脚本的入口点
    
    📚 PPO训练架构：
    1. Actor模型：生成策略，输出动作概率
    2. Critic模型：价值函数，估计状态价值
    3. Reward模型：奖励函数，评估生成质量
    4. Old Actor：用于重要性采样的旧策略
    5. Reference：用于KL惩罚的参考策略
    """

    # 📚 命令行参数解析
    parser = argparse.ArgumentParser(
        description="XLLM PPO (Proximal Policy Optimization)"
    )

    # ========== 基础训练参数 ==========
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument(
        "--save_weight", default="ppo_actor", type=str, help="保存权重的前缀名"
    )
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument(
        "--batch_size", type=int, default=2, help="batch size（PPO batch较小）"
    )

    # 📚 PPO学习率设置
    # PPO学习率通常很小，避免策略剧烈变化
    parser.add_argument("--learning_rate", type=float, default=8e-8, help="Actor学习率")
    parser.add_argument(
        "--critic_learning_rate", type=float, default=8e-8, help="Critic学习率"
    )

    # ========== 硬件配置 ==========
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="训练设备",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=1, help="数据加载线程数")

    # ========== 训练策略 ==========
    parser.add_argument(
        "--accumulation_steps", type=int, default=1, help="梯度累积步数"
    )
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")

    # ========== 模型架构参数 ==========
    parser.add_argument("--hidden_size", default=512, type=int, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", default=8, type=int, help="隐藏层数量")
    parser.add_argument(
        "--use_moe",
        default=0,
        type=int,
        choices=[0, 1],
        help="是否使用MoE架构（0=否，1=是）",
    )

    # ========== PPO生成参数 ==========
    parser.add_argument("--max_seq_len", default=66, type=int, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1536, help="生成的最大长度")

    # ========== 数据和模型参数 ==========
    parser.add_argument(
        "--data_path",
        type=str,
        default="../dataset/rlaif.jsonl",
        help="RLAIF数据路径",
    )

    # 📚 PPO超参数
    parser.add_argument(
        "--clip_epsilon",
        type=float,
        default=0.1,
        help="PPO裁剪参数（控制策略更新幅度）",
    )
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function系数")
    parser.add_argument("--kl_coef", type=float, default=0.02, help="KL散度惩罚系数")

    # 📚 推理模型配置
    parser.add_argument(
        "--reasoning",
        type=int,
        default=1,
        choices=[0, 1],
        help="推理模型类型（0=普通模型，1=推理模型）",
    )
    parser.add_argument(
        "--update_old_actor_freq", type=int, default=4, help="更新old_actor_model的频率"
    )

    # 📚 Reward模型路径
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

    # ========== 实验跟踪 ==========
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument(
        "--wandb_project", type=str, default="XLLM-PPO", help="wandb项目名"
    )

    args = parser.parse_args()
    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = XLLMConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )
    ckp_data = (
        lm_checkpoint(lm_config, weight=args.save_weight, save_dir="../checkpoints")
        if args.from_resume == 1
        else None
    )

    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = (
        nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    )

    # ========== 4. 配置wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb

        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        resume = "must" if wandb_id else None
        wandb_run_name = f"XLLM-PPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(
            project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume
        )
    # ========== 5. 初始化模型和数据 ==========
    # 📚 PPO模型架构
    base_weight = "reason" if args.reasoning == 1 else "full_sft"

    # 📚 Actor模型（策略模型）
    actor_model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    tokenizer.padding_side = "left"  # PPO需要左侧padding
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 📚 Old Actor模型（用于重要性采样）
    old_actor_model, _ = init_model(lm_config, base_weight, device=args.device)
    old_actor_model = old_actor_model.eval().requires_grad_(False)

    # 📚 Reference模型（用于KL惩罚）
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)

    # 📚 Critic模型（价值函数）
    moe_suffix = "_moe" if lm_config.use_moe else ""
    ckp = f"{args.save_dir}/{base_weight}_{lm_config.hidden_size}{moe_suffix}.pth"
    state_dict = torch.load(ckp, map_location=args.device)
    critic_model = CriticModel(lm_config)
    critic_model.load_state_dict(state_dict, strict=False)
    critic_model = critic_model.to(args.device)

    # 📚 Reward模型（奖励函数）
    reward_model = AutoModel.from_pretrained(
        args.reward_model_path, torch_dtype=torch.float16, trust_remote_code=True
    )
    reward_model = reward_model.to(args.device).eval().requires_grad_(False)
    reward_tokenizer = AutoTokenizer.from_pretrained(
        args.reward_model_path, trust_remote_code=True
    )

    # 📚 数据和优化器
    train_ds = RLAIFDataset(
        args.data_path, tokenizer, max_length=(args.max_seq_len + args.max_gen_len)
    )
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    actor_optimizer = optim.AdamW(actor_model.parameters(), lr=args.learning_rate)
    critic_optimizer = optim.AdamW(
        critic_model.parameters(), lr=args.critic_learning_rate
    )
    loader_for_count = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler
    )
    iters = len(loader_for_count)
    total_optimizer_steps = max(1, (iters // args.accumulation_steps) * args.epochs)
    actor_scheduler = CosineAnnealingLR(
        actor_optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10
    )
    critic_scheduler = CosineAnnealingLR(
        critic_optimizer,
        T_max=total_optimizer_steps,
        eta_min=args.critic_learning_rate / 10,
    )

    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        actor_model.load_state_dict(ckp_data["model"])
        critic_model.load_state_dict(ckp_data["critic_model"])
        actor_optimizer.load_state_dict(ckp_data["optimizer"])
        critic_optimizer.load_state_dict(ckp_data["critic_optimizer"])
        actor_scheduler.load_state_dict(ckp_data["scheduler"])
        critic_scheduler.load_state_dict(ckp_data["critic_scheduler"])
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)

    # ========== 7. DDP包装模型 ==========
    if dist.is_initialized():
        actor_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        critic_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        actor_model = DistributedDataParallel(actor_model, device_ids=[local_rank])
        critic_model = DistributedDataParallel(critic_model, device_ids=[local_rank])
        old_actor_model.to(args.device)

    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        if epoch == start_epoch and start_step > 0:  # 第一个epoch且存在检查点
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
            ppo_train_epoch(
                epoch,
                loader,
                len(loader) + start_step,
                old_actor_model,
                ref_model,
                actor_scheduler,
                critic_scheduler,
                reward_model,
                reward_tokenizer,
                start_step,
                wandb,
            )
        else:  # 默认从头开始
            loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            ppo_train_epoch(
                epoch,
                loader,
                len(loader),
                old_actor_model,
                ref_model,
                actor_scheduler,
                critic_scheduler,
                reward_model,
                reward_tokenizer,
                0,
                wandb,
            )
# jsonl一行conversations（4轮对话，最后assistant为空占位）
#         ↓RLAIFDataset
# messages[:‑1]切掉最后空assistant，保留全部历史上下文
#         ↓apply_chat_template(add_generation_prompt=True)
# prompt字符串：多轮对话 + 末尾<|im_start|>assistant续写标记
#         ↓DataLoader返回{"prompt":字符串,"answer":""}
#         ↓train_ppo.py循环
# tokenizer编码prompt得到input_ids
#         ↓actor.generate()在线采样生成LBC题的解答（带）
#         ↓decode得到response_text
#         ↓calculate_rewards()：格式奖励 + reward_model打分 → 单标量reward
#         ↓Critic输出序列末尾value，计算advantages
#         ↓actor / old_actor / ref_model计算response部分log概率
#         ↓PPO‑clip policy loss + critic MSE loss + KL惩罚 → total loss
#         ↓backward，梯度累积更新Actor、Critic
#         ↓定期同步old‑actor，打印日志，保存checkpoint


#梯度裁剪/梯度累积/学习率调度器更新