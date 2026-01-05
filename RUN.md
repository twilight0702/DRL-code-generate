# 运行流程（全套命令）

以下命令按“训练/验证划分”执行，默认训练集:验证集=8:2，随机种子=42。  
如需用全量任务训练/评估，请在命令末尾加 `--no-split`。  
如需使用 MBPP 数据集训练/评估，请添加 `--dataset mbpp`。

## 0. 环境准备

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 一键运行全流程脚本
```bash
./scripts/run_all.sh --dataset local --no-split
./scripts/run_all.sh --dataset mbpp --mbpp-max-samples 200
```

## 一键运行全流程脚本（更快的 PPO 配置）
```bash
./scripts/run_all_fast.sh --dataset local --no-split
./scripts/run_all_fast.sh --dataset mbpp --mbpp-max-samples 200
```

## 一键运行全流程脚本（中等 PPO 配置）
```bash
./scripts/run_all_medium.sh --dataset local --no-split
./scripts/run_all_medium.sh --dataset mbpp --mbpp-max-samples 200
```

## 1. 任务与测试自检

```bash
pytest tests/test_tasks.py
```

## 2. 基线实验（本地任务集，划分）

随机策略（验证集）
```bash
python train.py --mode random --episodes 5 --max-steps 120
```

random 作为下限，通常接近 0；如果 random 也很高，说明测试或奖励设计有问题。

模板基线（验证集）
```bash
python train.py --mode template --episodes 5 --max-steps 120
```

template 直接输出标准答案，给一个“理论上限”或“最简单可行”的成功率，方便写报告对比。

## 3. 预热训练 + 评估（本地任务集，划分）

预热训练（训练集）
```bash
python train.py --mode pretrain --epochs 40 --batch-size 4 --embed-dim 128 --hidden-dim 256 --save-path checkpoints/pretrain.pt
```

预热评估（验证集）
```bash
python train.py --mode eval --load-path checkpoints/pretrain.pt --max-gen-len 200 --max-steps 200
```

## 4. PPO 训练 + 评估（本地任务集，划分）

PPO 训练（训练集）
```bash
python ppo_train.py --updates 30 --episodes-per-update 12 --teacher-episodes 4 --max-steps 150 --max-seq-len 150 --save-path checkpoints/ppo.pt
```

PPO 模型评估（验证集）
```bash
python benchmark.py --mode model --ckpt checkpoints/ppo.pt --max-gen-len 200 --max-steps 200
```

## 5. 统一评估脚本（本地任务集，划分）

随机/模板/模型三种对比（验证集）
```bash
python benchmark.py --mode random --episodes 5
python benchmark.py --mode template --episodes 5
python benchmark.py --mode model --ckpt checkpoints/pretrain.pt
python benchmark.py --mode model --ckpt checkpoints/ppo.pt
```

## 6. 本地任务集（不划分，训练=评估）
```bash
python train.py --mode random --episodes 5 --max-steps 120 --no-split
python train.py --mode template --episodes 5 --max-steps 120 --no-split
python train.py --mode pretrain --epochs 40 --batch-size 4 --save-path checkpoints/pretrain.pt --no-split
python train.py --mode eval --load-path checkpoints/pretrain.pt --max-gen-len 200 --max-steps 200 --no-split
```

## 7. MBPP 外部评估（仅评估）
```bash
python mbpp_eval.py --ckpt checkpoints/pretrain.pt --split validation --max-samples 50
python mbpp_eval.py --ckpt checkpoints/ppo.pt --split validation --max-samples 50
```

## 8. 使用 MBPP 进行训练与评估
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

## 9. 结果文件（建议）

将关键结果手动整理写入：
- `results/exp_results.txt`
  
包含：成功率、平均奖励、样例生成代码（至少 1–2 个）。
