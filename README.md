# XLLM
**基于轻量小模型从零复现的大语言模型项目，低成本、全流程可落地、适合 LLM 入门实战学习**

⭐ **本项目为个人学习复现项目，基于 minimind、MokioMind 开源项目二次学习重构实现，完整复刻小参数 LLM 全链路训练、微调、强化学习、工具调用能力**

## 📖 项目介绍
XLLM 是一个**从零基于 PyTorch 原生实现的超轻量大语言模型项目**，专为 AI 入门学习者设计，以极低的算力成本、极简的代码结构，完整复现工业级 LLM 的全生命周期技术链路。

本项目核心参考 **jingyaogong/minimind**、**Wood-Q/MokioMind** 两大优质开源项目学习复现，摒弃各类第三方框架高层封装，所有核心算法、模型结构、训练逻辑均手写实现，无黑盒依赖，完美适配个人开发者学习、调试、二次开发。

区别于传统大模型动辄百亿参数、超高算力门槛的痛点，XLLM 延续轻量模型设计理念，核心优势如下：
- **超低成本**：单卡 3090 即可完成全流程训练，SFT 阶段仅需 2 小时、GPU 租用成本低至 3 元左右，个人电脑/低配服务器均可上手
- **全链路开源**：覆盖预训练、监督微调、LoRA 微调、DPO/PPO/GRPO/CISPO 强化学习、Agent 工具调用、模型蒸馏、自适应思考等全场景能力
- **纯原生实现**：不依赖 transformers、trl、peft 等框架封装，每一行核心代码均可追溯、可理解、可修改
- **高兼容性**：支持主流推理框架（llama.cpp、vllm、ollama）、OpenAI API 协议、Streamlit WebUI 可视化对话
- **轻量化高性能**：核心模型沿用 64M 轻量参数设计，结构对齐 Qwen3 生态，兼顾推理速度与基础对话、推理、工具调用能力

## ✨ 核心功能特性
本项目完整复刻主流小模型全能力，覆盖 LLM 入门到进阶所有核心技术点：

### 1. 模型结构
- 支持 Dense 基础模型 &amp; MoE 混合专家模型双架构
- 采用 Pre-Norm + RMSNorm、SwiGLU 激活函数、RoPE 旋转位置编码，支持 YaRN 长文本外推
- 自定义极简 Tokenizer（6400 词表），适配思考标签、工具调用专属标记

### 2. 完整训练链路
- 预训练（Pretrain）：通用文本语义学习，构建基础语言能力
- 监督微调（SFT）：多轮对话能力、指令跟随能力优化
- LoRA 参数高效微调：低成本垂直领域适配（医疗、专属问答、自我认知等）
- 偏好对齐（DPO）：基于人类偏好的模型回复优化
- AI 强化学习（RLAIF）：原生实现 PPO / GRPO / CISPO 前沿 RL 算法
- Agentic RL：多轮工具调用、环境交互、任务闭环强化学习
- 模型蒸馏：支持黑盒/白盒蒸馏，适配大模型能力迁移

### 3. 拓展能力
- 自适应思考：可动态开关模型显式思维链输出
- 原生 Tool Call：支持数学计算、时间查询、随机数生成等工具调用
- 断点续训：支持训练中断恢复、跨 GPU 设备续训、训练日志连续性保留
- 多卡训练：支持 DDP、DeepSpeed 分布式训练
- 可视化训练：适配 SwanLab / Wandb 训练指标监控
- 标准化部署：兼容 OpenAI API 协议、第三方 Chat UI、轻量化 Web 对话演示

## 📊 模型参数规格
项目默认主推轻量化模型，适配个人设备快速训练复现，核心参数如下：

| 模型版本 | 参数量 | 词表大小 | 上下文长度 | 核心特性 |
|--------|--------|----------|------------|----------|
| xllm-base（Dense） | 64M | 6400 | 32768 | 基础对话、推理、工具调用，极致低成本 |
| xllm-moe（MoE） | 198M | 6400 | 32768 | 4 专家混合架构，更高模型容量、更强泛化能力 |

## 💻 环境依赖配置
### 1. 基础软硬件环境（推荐）
- 系统：Ubuntu 20.04+ / Windows 10+ / macOS
- GPU：NVIDIA 3090/4090（24G 显存及以上，单卡即可）
- CUDA：12.2
- Python：3.10+
- 内存：≥32G

### 2. 依赖安装
```bash
# 克隆仓库
git clone https://github.com/PassionCR7/xllm.git
cd xllm

# 安装依赖（阿里云镜像加速）
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
```

## 📂 项目目录结构
```
xllm/
├── dataset/          # 训练数据集存放目录
├── images/           # 项目资源图片
├── model/            # 模型核心结构代码（Dense/MoE/Tokenizer）
├── scripts/          # 推理、权重转换、WebUI 脚本
├── trainer/          # 全流程训练脚本（预训练/SFT/RL/蒸馏等）
├── checkpoints/      # 训练断点权重保存目录
├── out/              # 最终模型权重输出目录
├── requirements.txt  # 环境依赖清单
└── README.md         # 项目说明文档
```

## 📥 数据集下载与配置
本项目沿用 minimind 开源标准化数据集，无需自行预处理，分为**轻量化快速复现版**和**完整版训练数据集**，按需下载即可。

### 1. 数据集清单
**核心必备数据集（新手推荐）**
- `pretrain_t2t_mini.jsonl`（1.2GB）：轻量化预训练数据，快速搭建基础语言能力
- `sft_t2t_mini.jsonl`（1.6GB）：轻量化 SFT 对话数据，包含基础工具调用样本
- `rlaif.jsonl`（24MB）：RL 强化学习训练数据

**进阶完整版数据集（效果最优）**
- `pretrain_t2t.jsonl`（10GB）：全量预训练数据
- `sft_t2t.jsonl`（14GB）：全量 SFT 对话&amp;工具调用数据
- `dpo.jsonl`（53MB）：DPO 偏好对齐数据
- `agent_rl.jsonl`（86MB）：Agent 多轮工具交互训练数据
- `agent_rl_math.jsonl`（18MB）：数学推理强化学习数据

### 2. 下载方式
所有数据集、预训练权重均来自 **minimind 官方开源仓库**：
- ModelScope（国内高速）：`gongjy/minimind-3`
- HuggingFace：`jingyaogong/minimind-3`

### 3. 数据集配置
将下载后的所有 `.jsonl` 数据集文件，统一放入项目根目录的 `./dataset` 文件夹下，无需二次处理，训练脚本可直接读取。

## 🚀 快速运行教程
> 本项目**不自带权重**，所有预训练权重均从 minimind 官方下载

### 一、模型推理（快速体验）
#### 1. 下载官方预训练权重
```bash
# ModelScope 国内高速下载
modelscope download --model gongjy/minimind-3 --local_dir ./minimind-3

# 或 HuggingFace 官方下载
git clone https://huggingface.co/jingyaogong/minimind-3
```

#### 2. 命令行推理
```bash
# Transformers 格式权重推理
python eval_llm.py --load_from ./minimind-3

# 原生 PyTorch 权重推理
python eval_llm.py --load_from ./model --weight full_sft
```

#### 3. Web 可视化对话
```bash
pip install streamlit
cd scripts
streamlit run web_demo.py
```

#### 4. 工具调用推理
```bash
python eval_toolcall.py --weight full_sft
```

### 二、从零训练全流程
所有训练脚本均在 `./trainer` 目录下执行，支持单卡/多卡训练、断点续训。

#### 1. 预训练（必备）
```bash
cd trainer
python train_pretrain.py
# 断点续训
python train_pretrain.py --from_resume 1
```

#### 2. 监督微调 SFT（必备）
```bash
cd trainer
python train_full_sft.py
# 断点续训
python train_full_sft.py --from_resume 1
```

#### 3. 进阶训练（可选）
```bash
# LoRA 微调
python train_lora.py

# DPO 偏好对齐
python train_dpo.py

# GRPO 强化学习
python train_grpo.py

# Agent 工具强化学习
python train_agent.py

# 模型蒸馏
python train_distillation.py
```

### 三、权重合并与导出
支持 LoRA 权重与基础模型合并，导出完整部署权重：
```bash
cd scripts
python convert_model.py
```

## ⚙️ 核心特性使用说明
### 1. 自适应思考开关
```bash
# 开启显式思维链输出
python eval_llm.py --load_from ./minimind-3 --open_thinking 1
```

### 2. OpenAI API 部署
兼容标准 OpenAI 协议，可接入任意第三方对话 UI：
```bash
cd scripts
python serve_openai_api.py
```

## 📈 训练成本与效率
基于单卡 NVIDIA 3090 实测，极致低成本复现：
- xllm-base(64M) 预训练+SFT 全流程：耗时 ≈ 2.3 小时，总成本 ≈ 3 元
- xllm-moe(198M) 全流程训练：耗时 ≈ 3.2 小时，总成本 ≈ 4.2 元

真正实现**个人零门槛、低成本吃透大模型全链路技术**。

## 📝 项目声明
- **学习复现说明**：本项目为个人开源学习项目，基于 **jingyaogong/minimind**、**Wood-Q/MokioMind** 开源项目学习、理解、重构实现，仅用于技术学习与交流。
- **开源协议**：沿用 Apache 2.0 开源协议，免费开源，可自由学习、二次开发。
- **资源说明**：本仓库**不存放任何权重与数据集**，所有权重、数据集均来自原 minimind 官方开源地址。

## 🌟 欢迎 Star &amp; Fork
如果本项目对你的 LLM 学习、实战开发有帮助，欢迎 **Star** 收藏、**Fork** 二次开发！
```
