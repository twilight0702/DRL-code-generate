# 运行流程（全套命令）

以下命令按“训练/验证划分”执行，默认训练集:验证集=8:2，随机种子=42。  
如需用全量任务训练/评估，请在命令末尾加 `--no-split`。

## 0. 环境准备

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1. 任务与测试自检

```bash
pytest tests/test_tasks.py
```

## 2. 基线实验

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

## 3. 预热训练 + 评估

预热训练（训练集）
```bash
python train.py --mode pretrain --epochs 40 --batch-size 4 --embed-dim 128 --hidden-dim 256 --save-path checkpoints/pretrain.pt
```

预热评估（验证集）
```bash
python train.py --mode eval --load-path checkpoints/pretrain.pt --max-gen-len 200 --max-steps 200
```

## 4. PPO 训练 + 评估

PPO 训练（训练集）
```bash
python ppo_train.py --updates 30 --episodes-per-update 12 --teacher-episodes 4 --max-steps 150 --max-seq-len 150 --save-path checkpoints/ppo.pt
```

PPO 模型评估（验证集）
```bash
python benchmark.py --mode model --ckpt checkpoints/ppo.pt --max-gen-len 200 --max-steps 200
```

## 5. 统一评估脚本（可选）

随机/模板/模型三种对比（验证集）
```bash
python benchmark.py --mode random --episodes 5
python benchmark.py --mode template --episodes 5
python benchmark.py --mode model --ckpt checkpoints/pretrain.pt
python benchmark.py --mode model --ckpt checkpoints/ppo.pt
```

## 6. 结果文件（建议）

将关键结果手动整理写入：
- `results/exp_results.txt`
  
包含：成功率、平均奖励、样例生成代码（至少 1–2 个）。
