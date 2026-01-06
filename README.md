# 项目说明

本项目为一个**极简、可在 CPU 上复现**的深度强化学习（DRL）代码生成实验。
智能体以字符级方式生成 Python 代码，通过 pytest 测试获得奖励，用于本科机器学习课程大作业。

## 已完成的工作
- 构建了小规模任务集：每个任务包含 prompt、参考解与 pytest 测试。
- 实现 Gym 风格代码生成环境，基于 pytest 运行结果给奖励。
- 提供基线（random/template）、监督预热（teacher forcing）与 PPO 训练脚本。
- 提供统一评估脚本，用于对比不同方法的通过率。
- 增加训练/验证划分（默认 8:2，seed=42）。
- 补充小组计划与完整运行流程文档。

## 结果摘要与使用建议
- 建议主实验使用 **不划分**（`--no-split`，训练=评估）配置，PPO 可复现约 11–12/13 成功率（本地任务）；将其作为报告的主要结果。结果在 example_output.txt
- **划分模式**（train/val=10/3）目前验证集成功率极低（有实验 0/3），可在报告的“局限/改进”部分说明“泛化不足”。
- **MBPP** 训练/评估表现很差（验证集可到 0/90），字符级小模型难以泛化到公开题库，同样建议作为不足与未来改进方向记录。
- 日志与结果示例见 `results/run_medium_local_*.txt`，可在报告引用作为证据。

## 项目结构
- `tasks/`：任务定义（prompt、参考解、测试代码）。
- `env/`：代码生成环境（运行 pytest 并给奖励）。
- `models/`：字符级 GRU 模型。
- `train.py`：基线 + 预热训练 + 评估入口。
- `ppo_train.py`：PPO 训练脚本（含行为克隆可选项）。
- `benchmark.py`：统一评估脚本（random/template/model）。
- `mbpp_eval.py`：在 MBPP 数据集上的额外评估脚本（外部评估集）。
- `tests/`：参考解的 pytest 校验。
- `report.md`：工作日志与原始实验记录输出参考。（比较乱，建议直接给ai看就好）
- `PLAN.md`：初始计划。
- `RUN.md`：完整运行流程（下方也包含）。

## 工作流程概览
1. 任务提供自然语言 prompt 与 pytest 测试。
2. 智能体逐字符生成代码，直到 `<EOS>` 或达到最大长度。
3. 环境运行 pytest 计算奖励（通过比例 + 语法奖励）。
4. 在验证集上评估成功率与平均奖励。

## 数据与任务
任务为小规模函数题（数学/字符串/列表处理等），保持简单可行。
任务列表在 `tasks/tasks.py` 中定义。

## 运行指令
默认使用训练/验证划分（8:2，seed=42），数据集为 `local`（自建任务集）。  
若要全量训练与评估，请在命令末尾加 `--no-split`。  
若要使用 MBPP 数据集训练/评估，请添加 `--dataset mbpp` 及对应参数。

### 0) 环境准备
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 一键运行全流程脚本

注意！这里只建议使用 `medium+不划分` 运行，也就是 `./scripts/run_all_medium.sh --dataset local --no-split`

如果需要验证环境是否可行可以先运行 `./scripts/run_all_fast.sh --dataset local --no-split` 看能否正常结束

```bash
./scripts/run_all.sh --dataset local --no-split
./scripts/run_all.sh --dataset mbpp --mbpp-max-samples 200
```

### 一键运行全流程脚本（更快的 PPO 配置）
```bash
./scripts/run_all_fast.sh --dataset local --no-split
./scripts/run_all_fast.sh --dataset mbpp --mbpp-max-samples 200
```

### 一键运行全流程脚本（中等 PPO 配置）
```bash
./scripts/run_all_medium.sh --dataset local --no-split
./scripts/run_all_medium.sh --dataset mbpp --mbpp-max-samples 200
```

### 1) 任务与测试自检
```bash
pytest tests/test_tasks.py
```

### 2) 基线实验（验证集，本地任务集）
随机基线：
```bash
python train.py --mode random --episodes 5 --max-steps 120
```

模板基线：
```bash
python train.py --mode template --episodes 5 --max-steps 120
```

### 3) 预热训练 + 评估（本地任务集，划分）
预热训练（训练集）：
```bash
python train.py --mode pretrain --epochs 40 --batch-size 4 --embed-dim 128 --hidden-dim 256 --save-path checkpoints/pretrain.pt
```

预热评估（验证集）：
```bash
python train.py --mode eval --load-path checkpoints/pretrain.pt --max-gen-len 200 --max-steps 200
```

### 4) PPO 训练 + 评估（本地任务集，划分）
PPO 训练（训练集）：
```bash
python ppo_train.py --updates 30 --episodes-per-update 12 --teacher-episodes 4 --max-steps 150 --max-seq-len 150 --save-path checkpoints/ppo.pt
```

PPO 评估（验证集）：
```bash
python benchmark.py --mode model --ckpt checkpoints/ppo.pt --max-gen-len 200 --max-steps 200
```

### 5) 统一评估脚本（本地任务集，划分）
```bash
python benchmark.py --mode random --episodes 5
python benchmark.py --mode template --episodes 5
python benchmark.py --mode model --ckpt checkpoints/pretrain.pt
python benchmark.py --mode model --ckpt checkpoints/ppo.pt
```

### 6) 本地任务集（不划分，训练=评估）
```bash
python train.py --mode random --episodes 5 --max-steps 120 --no-split
python train.py --mode template --episodes 5 --max-steps 120 --no-split
python train.py --mode pretrain --epochs 40 --batch-size 4 --save-path checkpoints/pretrain.pt --no-split
python train.py --mode eval --load-path checkpoints/pretrain.pt --max-gen-len 200 --max-steps 200 --no-split
```

### 7) MBPP 外部评估（仅评估）
```bash
python mbpp_eval.py --ckpt checkpoints/pretrain.pt --split validation --max-samples 50
python mbpp_eval.py --ckpt checkpoints/ppo.pt --split validation --max-samples 50
```

### 8) 使用 MBPP 进行训练与评估
预热训练（MBPP）：
```bash
python train.py --dataset mbpp --mode pretrain --mbpp-train-split train --mbpp-eval-split validation --mbpp-max-samples 200 --save-path checkpoints/pretrain_mbpp.pt
```

评估（MBPP）：
```bash
python train.py --dataset mbpp --mode eval --mbpp-eval-split validation --mbpp-max-samples 200 --load-path checkpoints/pretrain_mbpp.pt
```

PPO 训练（MBPP）：
```bash
python ppo_train.py --dataset mbpp --mbpp-train-split train --mbpp-eval-split validation --mbpp-max-samples 200 --save-path checkpoints/ppo_mbpp.pt
```

评估（MBPP）：
```bash
python mbpp_eval.py --ckpt checkpoints/ppo_mbpp.pt --split validation --max-samples 200
```

## 说明
- 奖励基于 pytest 通过比例，并带有小额语法奖励。
- 词表覆盖所有 prompt 与参考解字符，避免 OOV。
- 训练参数保持轻量，适合 CPU 快速验证。
