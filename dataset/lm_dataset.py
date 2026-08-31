import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from torch.utils.data import Dataset
import torch
import random
from datasets import load_dataset

# 禁用 HuggingFace tokenizer 的多进程并行，避免在 DataLoader 多进程环境中产生死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def pre_processing_chat(conversations, add_system_ratio=0.2, use_cot=0):
    """
    对话前处理：以一定概率随机插入 system 消息。
    :param use_cot: 0 丢弃reasoning_content；1 将reasoning_content拼接
    特点：
    - 只有当首条消息不是 system 角色时才可能插入。
    - add_system_ratio 控制插入概率（默认 20%），引入随机性可提升模型
      对有/无 system prompt 两种情况的泛化能力。
    - system 内容从预定义的中英文 prompt 池中随机抽取，覆盖不同表达风格。
    """
    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model.",
    ]
    cleaned_conversations = []
    for msg in conversations:
        role = msg["role"]
        content = msg["content"]

        if role == "assistant":
            reasoning_content = msg.get("reasoning_content", "")
            if use_cot == 1 and reasoning_content.strip():
                # 开启CoT：拼接 思考内容
                content = f"{reasoning_content}{content}"
            # use_cot=0：直接忽略reasoning_content，不做任何拼接

        new_msg = {"role": role, "content": content}
        cleaned_conversations.append(new_msg)

    conversations = cleaned_conversations

    if conversations and conversations[0].get("role") != "system":
        if random.random() < add_system_ratio:
            return [
                {"role": "system", "content": random.choice(SYSTEM_PROMPTS)}
            ] + conversations
    return conversations


def post_processing_chat(prompt_content, empty_think_ratio=0.05):
    """
    对话后处理：清理模板渲染后多余的空 <think> 块。

    特点：
    - 针对带 CoT（chain-of-thought）格式的模型，apply_chat_template 有时会
      渲染出 "<think>\n\n</think>\n\n" 这样的空思考块占位符。
    - 大部分情况下（概率 1 - empty_think_ratio = 95%）直接删除该空块，
      防止模型学到"无意义思考"的坏习惯。
    - 保留少量空思考块（empty_think_ratio = 5%），让模型也能处理该边界情况。
    """
    if (
        "<think>\n\n</think>\n\n" in prompt_content
        and random.random() > empty_think_ratio
    ):
        prompt_content = prompt_content.replace("<think>\n\n</think>\n\n", "")
    return prompt_content


# ──────────────────────────────────────────────────────────────────────────────
# 1. PretrainDataset —— 自回归预训练数据集
# ──────────────────────────────────────────────────────────────────────────────
# 训练目标：Next-Token Prediction（下一个 token 预测）
# 数据格式：{"text": "一段原始文本"}
# 训练特点：
#   - 模型对整段文本的每个位置都进行预测，没有"只学回复"的区分。
#   - 使用 BOS/EOS 标记文本边界，让模型学会文本的起止。
#   - PAD token 对应的 label 置 -100，不参与 loss 计算，节省无效梯度。
#   - labels 直接 clone 自 input_ids（即 X 和 Y 错位一格：Y[t] = X[t+1]）。
# ──────────────────────────────────────────────────────────────────────────────
class PretrainDataset(Dataset):
    #init
    def __init__(self,data_path,tokenizer,max_length=512):
        super().__init__()
        self.tokenizer=tokenizer
        self.max_length=max_length #输入给GPU的最大长度
        self.samples=load_dataset("json",data_files=data_path,split="train")
    #__len__
    def __len__(self):
        return len(self.samples)
    #__getitem__,拿到json每一行，

    def __getitem__(self,index):
        sample=self.samples[index]
        #tokenizer把文本转化为input_id，首尾各留一个token的位置给BOS、EOS
        tokens=self.tokenizer(
            # jsonl 文件必须每行是{"text":"xxx"}
            str(sample["text"]), #这里假设jsonl中有一个“text”字段，包含了文本内容 
            add_special_tokens=False,
            max_length=self.max_length-2, #留位置给EOS，BOS
            truncation=True, #长度超过max_length时，剪切，丢掉后面多余 token
        ).input_ids
        # 拼接 BOS + token序列 + EOS，构成完整序列
        tokens=[self.tokenizer.bos_token_id]+tokens+[self.tokenizer.eos_token_id]
        # 右侧用 PAD 补齐到 max_length，保证 batch 内等长
        input_ids=tokens+[self.tokenizer.pad_token_id]*(self.max_length-len(tokens)) #填充到max_length
        input_ids=torch.tensor(input_ids,dtype=torch.long) #转为tensor
        # 需要编写labels，防止PAD参与loss计算
        # labels 与 input_ids 完全相同，但 PAD 位置置 -100，CrossEntropyLoss 会自动忽略 -100，不计入 loss
        labels=input_ids.clone()
        labels[labels==self.tokenizer.pad_token_id]=-100 #PAD位置设为-100，忽略这些位置的loss计算

        # 编写attention_mask,告诉model哪些位置有效，哪些位置是PAD
        # 返回 attention_mask，使 attention 层能屏蔽 padding token
        attention_mask=(input_ids!=self.tokenizer.pad_token_id).long() #PAD位置为0，不是PAD位置为1
        #需要输出：input_ids,attention_mask,labels
        return input_ids, labels, attention_mask


# ──────────────────────────────────────────────────────────────────────────────
# 2. SFTDataset —— 有监督微调（Supervised Fine-Tuning）数据集
# ──────────────────────────────────────────────────────────────────────────────
# 训练目标：让模型学会"只预测 assistant 回复"，忽略 user/system 输入
# 数据格式：{"conversations": [{"role": "user"/"assistant"/"system", "content": "..."}]}
# 训练特点：
#   - 通过 generate_labels 扫描 bos_id（assistant 回复起始标记）定位每段回复，
#     仅将 assistant 回复的 token 位置设为有效 label，其余全部为 -100。
#   - 这样做的意义：让 loss 只反映模型对"正确回答"的拟合，不浪费梯度在
#     用户输入的复现上（用户输入只作为 context，不是预测目标）。
#   - 支持 function calling：若 system 消息携带 "functions" 字段，
#     会透传给 apply_chat_template，生成带工具描述的提示词。
#   - 与 PretrainDataset 的关键区别：标签是"稀疏"的，只有 assistant 部分非 -100。
# ──────────────────────────────────────────────────────────────────────────────
class SFTDataset(Dataset):
    def __init__(self,jsonl_path,tokenizer,max_length=1024, use_cot=0):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_cot = use_cot   # 保存cot开关
        self.samples = load_dataset("json", data_files=jsonl_path, split="train")
        # 预先 tokenize assistant 回复的起始标记（BOS + "assistant\n"）
        # 用于在 generate_labels 中定位每段 assistant 回复的开始位置
        
        # 把模板中bos + assistant\n这一段预先 tokenize 成 id 列表
        # 训练时扫描 input_ids，匹配这个 id 片段，就代表assistant 回答正式开始
        self.bos_id = tokenizer(
            f"{tokenizer.bos_token}assistant\n", add_special_tokens=False
        ).input_ids
        # 预先 tokenize assistant 回复的结束标记（EOS + "\n"）
        # 用于在 generate_labels 中定位每段 assistant 回复的结束位置
        self.eos_id = tokenizer(
            f"{tokenizer.eos_token}\n", add_special_tokens=False
        ).input_ids

    def __len__(self):
        return len(self.samples)


    def create_chat_prompt(self, conversations):
        """
        将多轮对话转换为模型输入的字符串。

        特点：
        - 复制原始 conversations,防止修改原始数据
        - 检测 system 消息中是否携带 functions 字段(function calling 场景),
          若有则透传给 apply_chat_template,生成标准 tool-use 格式的提示词
        - add_generation_prompt=False:不在末尾追加"请模型续写"的 prompt,
          因为训练时需要完整的 input+output 序列，而非开放续写。
        """
        messages = conversations.copy()
        tools = (
            conversations[0]["functions"]
            if (
                conversations
                and conversations[0]["role"] == "system"
                and conversations[0].get("functions")
            )
            else None
        )
        #add_generation_prompt=False:训练，数据是完整的user输入 + assistant完整回答，不需要额外追加续写标记
        # True：会在末尾追加 assistant 前缀，用于推理阶段，告诉模型 “现在该你输出了”
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, tools=tools
        )


    def generate_labels(self, input_ids):
        # 绝大多数位置 label=-100，只有 assistant 回答片段的 token 才是真实 token id，参与 loss 计算
        # user、system 输入部分全部‑100，不计算 loss
        """
        生成 SFT 训练所需的稀疏标签序列。

        算法逻辑（滑动窗口扫描）：
        1. 初始化全 -100 的 labels，默认所有位置不计算 loss。
        2. 逐位扫描 input_ids，检测是否匹配 bos_id（assistant 回复起始）。
        3. 匹配到 bos_id 后，向后扫描直到找到 eos_id（回复结束）。
        4. 将 [start, end+len(eos_id)) 区间内的 label 设为对应的 input_ids 值，
           即这段 assistant 回复参与 loss 计算。
        5. EOS token 本身也计入 label，让模型学会何时停止生成。
        6. 跳过已处理区间，继续扫描下一段 assistant 回复（支持多轮对话）。
        """
        #labels 数组初始全部 -100。CrossEntropyLoss 的特性：label=-100 会直接忽略该位置 loss
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i : i + len(self.bos_id)] == self.bos_id:
                # 跳过 bos_id 本身，从 assistant 实际内容开始
                start = i + len(self.bos_id)
                end = start
                # 向后扫描，找到 eos_id 的位置
                while end < len(input_ids):
                    if input_ids[end : end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # 将 assistant 回复（含 EOS）区间的 label 设为真实 token id
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels


    def __getitem__(self, index):
        sample = self.samples[index]

        # Step 1：随机决定是否插入 system prompt（数据增强）
        #20% 概率随机往对话最前面插入一条 system 角色消息
        conversations = pre_processing_chat(sample["conversations"], use_cot=self.use_cot)

        # Step 2：用 chat template 渲染完整对话字符串
        prompt = self.create_chat_prompt(conversations)

        # Step 3：清理可能出现的空 <think> 块
        # 针对 CoT 模型，模板容易生成空块：\n\n\n\n
        #95% 概率删掉这个空思考块，5% 保留，增加边界鲁棒性
        prompt = post_processing_chat(prompt)

        # Step 4：tokenize 并截断到 max_length，不足则右侧 PAD 补齐
        input_ids = self.tokenizer(prompt).input_ids[: self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        # Step 5：生成稀疏标签，只有 assistant 回复部分有有效 label
        labels = self.generate_labels(input_ids)
        # # === 调试打印 ===
        # print(f"\n--- Sample {index} ---")
        # for i, (x, y) in enumerate(zip(input_ids[:-1], labels[1:])):
        #     print(f"{i:3d}: X={self.tokenizer.decode([x])!r:16s} ---> Y={self.tokenizer.decode([input_ids[i+1]])!r:16s} label={y}")
        # # ================

        # Step6：构造attention_mask，pad位置为0，真实token为1，attention 层屏蔽 padding token
        attention_mask = (
            torch.tensor(input_ids, dtype=torch.long) != self.tokenizer.pad_token_id
        ).long()
        #返回三元组(input_ids, labels, attention_mask)，
        # 和 PretrainDataset 输出格式保持一致，训练脚本可以复用同一套训练循环
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            attention_mask,
        )



# ──────────────────────────────────────────────────────────────────────────────
# 3. DPODataset —— 直接偏好优化（Direct Preference Optimization）数据集
# ──────────────────────────────────────────────────────────────────────────────
# 训练目标：让模型学会"偏好好回答、远离坏回答"，使输出更符合人类偏好
# 数据格式：{"chosen": [{role, content}...], "rejected": [{role, content}...]}
#   - chosen：人类标注的更优回答对话
#   - rejected：人类标注的较差回答对话
# 训练特点：
#   - 每条样本同时返回 chosen 和 rejected 两份 tokenized 序列，
#     训练时 DPO loss 会最大化 chosen 回复的对数似然、最小化 rejected 的。
#   - loss_mask 的设计与 SFT 一致：只有 assistant 回复部分为 1，
#     其余为 0，保证对比信号仅来自模型的实际输出部分。
#   - 采用"错位"方式构造输入输出对：x 取 [:-1]，y 取 [1:]，
#     即 x[t] 预测 y[t] = input[t+1]，标准自回归格式。
#   - mask 同样错位取 [1:]，与 y 对齐，方便在训练时直接做 masked loss。
#   - max_length 默认 4096，比 SFT 更长，因为 DPO 数据通常包含完整对话上下文。
# ──────────────────────────────────────────────────────────────────────────────
class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # pad_token_id 若不存在则回退到 0，保证补齐操作不会崩溃
        self.padding = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        )
        # 与 SFTDataset 相同：预先 tokenize assistant 回复的起止标记，
        # 用于 generate_loss_mask 中精准定位 assistant 回复区间
        self.bos_id = tokenizer(
            f"{tokenizer.bos_token}assistant\n", add_special_tokens=False
        ).input_ids
        self.eos_id = tokenizer(
            f"{tokenizer.eos_token}\n", add_special_tokens=False
        ).input_ids
        self.samples = load_dataset("json", data_files=file_path, split="train")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        chosen = sample["chosen"]  # 优质回答对话列表，格式：[{role, content}, ...]
        rejected = sample["rejected"]  # 劣质回答对话列表，格式同上

        # Step 1：将 chosen / rejected 对话分别渲染为字符串
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )
        chosen_prompt = post_processing_chat(chosen_prompt)

        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        rejected_prompt = post_processing_chat(rejected_prompt)

        # Step 2：tokenize 并 padding 到 max_length（统一序列长度，方便 batch）
        chosen_encoding = self.tokenizer(
            chosen_prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )

        chosen_input_ids = chosen_encoding["input_ids"]
        # Step 3：生成 loss mask，只有 assistant 回复部分为 1
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding["input_ids"]
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)

        # Step 4：构造自回归训练对，x=[:-1] 作为输入，y=[1:] 作为目标
        #         mask=[1:] 与 y 对齐，决定哪些位置的 loss 计入梯度
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        # ！修正：返回 attention_mask，使 attention 层能屏蔽 padding token
        attention_mask_chosen = (
            torch.tensor(chosen_input_ids[:-1], dtype=torch.long) != self.padding
        ).long()
        attention_mask_rejected = (
            torch.tensor(rejected_input_ids[:-1], dtype=torch.long) != self.padding
        ).long()

        return {
            "x_chosen": x_chosen,
            "y_chosen": y_chosen,
            "mask_chosen": mask_chosen,
            "x_rejected": x_rejected,
            "y_rejected": y_rejected,
            "mask_rejected": mask_rejected,
            "attention_mask_chosen": attention_mask_chosen,
            "attention_mask_rejected": attention_mask_rejected,
        }

    def generate_loss_mask(self, input_ids):
        """
        生成 DPO 训练所需的 loss mask（0/1 二值序列）。

        与 SFTDataset.generate_labels 逻辑完全相同，区别在于：
        - SFT 返回的是具体的 token id（用于 CE loss）
        - DPO 返回的是 0/1 掩码（用于 masked 对数似然计算）
        算法：扫描 bos_id → 找到 eos_id → 区间内置 1，其余置 0。
        """
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i : i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end : end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # 将 assistant 回复（含 EOS）区间的 mask 置 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    

# ──────────────────────────────────────────────────────────────────────────────
# 4. RLAIFDataset —— 基于 AI 反馈的强化学习数据集（用于 PPO / GRPO）
# ──────────────────────────────────────────────────────────────────────────────
# 训练目标：为 RL 训练提供"问题-参考答案"对，只给对话上下文，不给参考答案，让模型在线生成回答，
# 由 actor 在线采样生成回复，再由 reward model 或规则函数打分优化      
# 数据格式：{"conversations": [{"content": "..."}, {"content": "..."}]}
#   - 奇数索引 (0,2,4...) 为 user 发言
#   - 偶数索引 (1,3,5...) 为 assistant 发言（最后一条为参考答案）
# 训练特点（与前三个 Dataset 的核心区别）：
#   - **不做离线 tokenize**：只返回原始字符串 prompt 和 answer，
#     让 RL trainer（PPO/GRPO）在线 rollout 时自行 tokenize，
#     因为 RL 需要动态生成回复并实时打分，无法预先固定 token 序列。
#   - create_chat_prompt 会剥离最后一条 assistant 消息，
#     将其余对话渲染为带 add_generation_prompt=True 的 prompt，
#     供 actor 模型续写；answer 保存为参考答案用于奖励计算。
#   - bos_id / eos_id 在此类中被定义但目前未用于 mask 计算，
#     保留以备后续扩展（如 reward shaping）需要。
#   - 返回值是 dict{"prompt": str, "answer": str}，而非 tensor，
#     这是 RL 数据集与 SL 数据集（返回 tensor）的最显著差异。
# ──────────────────────────────────────────────────────────────────────────────
# 以rlaif.jsonl中的如下一个样本为例：
# {"conversations": [{"role": "user", "content": "如果给定生成矩阵为 G = [1 0 0 0 1 0 0 0 1 1 1 0 0 1 1 1 0 1]，请找出用于解码线性块码的奇偶校验矩阵。"}, 
# {"role": "assistant", "content": "奇偶校验矩阵用于通过将其与码字相乘并检查结果是否为零向量来检查码字的有效性。\n要找到给定生成矩阵的奇偶校验矩阵，需要找到其行最简阶梯形式（RREF）。\n给定生成矩阵 G 如下：\nG = [1 0 0 0 1 0 0 0 1 1 1 0 0 1 1 1 0 1]\n为了找到 G 的 RREF，我们将使用以下步骤：\n1. 通过将整行除以该元素，将第一列中的第一个非零元素变为 1。\n2.接下来，使用第一行通过行操作消除第一列中的其他行中的非零元素。\n3. 重复第二列等同样的过程，直到矩阵处于 RREF 形式。\n[1 0 0 0 1 0 0 0 1 1 1 0 0 1 1 1 0 1]\n[0 1 1 1 0 1 1 1 0 0 0 1 1 0 0 0 1 0]\n[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]\n最后一个矩阵是 G 的 RREF，因此奇偶校验矩阵是其转置。\nH = [1 0 0 1 1 1 1 0 0 0 1 1 1 0 1 1 0 1]\n因此，给定生成矩阵的奇偶校验矩阵为 H = [1 0 0 1 1 1 1 0 0 0 1 1 1 0 1 1 0 1]。"}, 
# {"role": "user", "content": "对于系统化 LBC，奇偶校验位为\nC1 = M1 ⊕ M2 ⊕ M3\nC2 = M2 ⊕ M3 ⊕ M4\nC3 = M1 ⊕ M2 ⊕ M4\n请找出：\na. 生成矩阵。\nb. 错误检测和纠正能力。\nc. 奇偶校验矩阵。\nd. 接收到的码字 [1101001] 的纠正码字。"}, 
# {"role": "assistant", "content": ""}]}

# conversations共 4 条；第 0 user，第 1 assistant，第 2 user，第 3 条 assistant content 为空字符串，
# 这是 PPO‑RLAIF 样本特点：只给对话上下文，不给参考答案，让模型在线生成回答。
class RLAIFDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset("json", data_files=jsonl_path, split="train")
        # 保留 bos_id / eos_id 以兼容未来可能的 mask 扩展
        self.bos_id = tokenizer(
            f"{tokenizer.bos_token}assistant", add_special_tokens=False
        ).input_ids
        self.eos_id = tokenizer(
            f"{tokenizer.eos_token}", add_special_tokens=False
        ).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        messages = []
        answer = ""
        #循环遍历数据中4个turn的user、assistant
        # - i=0：turn=user 问题 → role=user，append；answer = 用户问题文本
        # - i=1：turn=assistant 回答 → role=assistant，append；answer = 第一段解答文本
        # - i=2：turn = 第二个 user 问题 → role=user，append；answer = 第二个用户的LBC编码大题
        # - i=3：turn=assistant，content 是空字符串 → role=assistant，append；answer = ""
        for i, turn in enumerate(conversations):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": turn["content"]})
            #得到：
            # messages = [
            #  {"role":"user", "content":"G矩阵求奇偶校验矩阵问题"},
            #  {"role":"assistant", "content":"第一段解答……"},
            #  {"role":"user", "content":"系统化LBC那道abcd大题"},
            #  {"role":"assistant", "content":""}
            # ]
            # answer = ""
            answer = turn["content"]  
        # apply_chat_template(..., add_generation_prompt=True)把上面 3 条上下文渲染成字符串 prompt（如下）：
                # <|im_start|>user
                # 如果给定生成矩阵为 G = [1 0 0 0 1 0 0 0 1 1 1 0 0 1 1 1 0 1]，请找出用于解码线性块码的奇偶校验矩阵。
                # <|im_end|>
                # <|im_start|>assistant
                # 奇偶校验矩阵用于通过将其与码字相乘并检查结果是否为零向量来检查码字的有效性。……
                # <|im_end|>
                # <|im_start|>user
                # 对于系统化 LBC，奇偶校验位为
                # C1 = M1 ⊕ M2 ⊕ M3
                # C2 = M2 ⊕ M3 ⊕ M4
                # C3 = M1 ⊕ M2 ⊕ M4
                # 请找出：
                # a. 生成矩阵。
                # b. 错误检测和纠正能力。
                # c. 奇偶校验矩阵。
                # d. 接收到的码字 [1101001] 的纠正码字。
                # <|im_end|>
                # <|im_start|>assistant
        # 注意：末尾多出<|im_start|>assistant续写标记，告诉模型：现在该 assistant 输出回答（LBC编码大题abcd 的解答）
        prompt = self.tokenizer.apply_chat_template(
            # 把message中最后这个{"role":"assistant", "content":""}整条删掉，不会进入 prompt；
            # 这一条是待生成的回复位置，不能放进 prompt；
            # 如果不切掉，模板会把空 assistant 拼进 prompt，generate 就没有地方续写。
            messages[:-1],
            tokenize=False,
            # add_generation_prompt=True：在末尾追加续写引导 token，告诉模型"现在开始生成"
            add_generation_prompt=True,
        )
        prompt = post_processing_chat(prompt)
        # 数据集输出是字符串，不是 tensor；tokenize 放到 ppo 训练循环内部做
        return prompt, answer

    def __getitem__(self, index):
        sample = self.samples[index]
        # 返回原始字符串，不做 tokenize，由 RL trainer 在线处理
        prompt, answer = self.create_chat_prompt(sample["conversations"])
        #  返回原始字符串，不返回 tensor；answer只是参考答案，
        # 本代码的奖励计算并没有使用这个 answer 字段，奖励来自规则+外部 reward model
        return {"prompt": prompt, "answer": answer}
        #输出 batch：`batch["prompt"]`是字符串列表，送入 ppo 脚本做在线 tokenize

if __name__ == "__main__":
    pass