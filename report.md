# 深度强化学习代码生成实验工作日志（含问题与解决方案）

> 项目目录：`/home/twilight/machine-learning/my-test`  
> 主题：极简深度强化学习（PPO）用于字符级代码生成，任务为通过简单函数的 pytest 测试。  
> 目标：零基础可复现，CPU 可运行，包含预热、强化训练、评估与基线。

## 时间线与操作步骤

### 1. 初始规划与脚手架
- 编写了 `PLAN.md`，明确任务设定、数据、模型、算法、流程、目录和时间表。
- 创建基础文件和依赖：
  - `requirements.txt`：torch、gymnasium、numpy、pytest、tqdm、rich。
  - 目录：`tasks/`、`env/`、`models/`、`tests/`。
  - `tasks/tasks.py`：5 个小任务（abs、reverse_string、factorial、is_palindrome、count_vowels），含 prompt、参考解、pytest 测试。
  - `env/code_env.py`：字符级代码生成 Gym 环境，step 生成字符，结束时运行 pytest 得到奖励。
  - `tests/test_tasks.py`：验证每个任务的参考解能通过测试。
  - `train.py`：随机策略/模板基线。
- 补充包初始化文件 `tasks/__init__.py`、`env/__init__.py`、`models/__init__.py`，解决 pytest 导入问题，并在 `tests/conftest.py` 加入仓库根目录到 sys.path。
- 运行 `pytest tests/test_tasks.py` 通过。

```bash
(.venv) twilight@TWILIGHT:~/machine-learning/my-test$ pytest tests/test_tasks.py
========================================================== test session starts ==========================================================
platform linux -- Python 3.12.3, pytest-9.0.2, pluggy-1.6.0
rootdir: /home/twilight/machine-learning/my-test
collected 1 item                                                                                                                        

tests/test_tasks.py .                                                                                                             [100%]

=========================================================== 1 passed in 1.12s ===========================================================
(.venv) twilight@TWILIGHT:~/machine-learning/my-test$ python train.py --episodes 2 --max-steps 120
[ep 0] task=is_palindrome passed=False reward=0.0 steps=40
[ep 1] task=count_vowels passed=False reward=0.0 steps=11
(.venv) twilight@TWILIGHT:~/machine-learning/my-test$ python train.py --episodes 3 --mode random
[random ep 0] task=abs_value passed=False reward=0.0 steps=116
[random ep 1] task=reverse_string passed=False reward=0.0 steps=120
[random ep 2] task=reverse_string passed=False reward=0.0 steps=3
Success rate: 0/3 = 0.00
(.venv) twilight@TWILIGHT:~/machine-learning/my-test$ python train.py --episodes 3 --mode template
[template ep 0] task=factorial passed=True reward=1.0 steps~179
[template ep 1] task=abs_value passed=True reward=1.0 steps~50
[template ep 2] task=factorial passed=True reward=1.0 steps~179
Success rate: 3/3 = 1.00
```

### 2. 基线与预热
- `train.py` 增加模式：
  - `random`：随机策略（奖励 0）。
  - `template`：直接输出 canonical_solution（奖励 1）。
  - `pretrain`：字符级 GRU 模型（`models/net.py` CharPolicy）做 teacher forcing 预热。
- 预热流程：
  - Tokenizer/TaskDataset/Collate 处理 padding，CrossEntropy 忽略 padding。
  - 保存 checkpoint（含 vocab、eos_token、embed_dim、hidden_dim）。
  - 贪心解码支持任务前缀 prompt。
- 运行示例：
  - 预热：`python train.py --mode pretrain --epochs 60 --hidden-dim 512 --save-path checkpoints/pretrain.pt`
  - 评估预热：`python train.py --mode eval --load-path checkpoints/pretrain.pt --max-gen-len 200`
- 结果：预热后贪心成功率 2/5（abs/reverse），其余未过。

```bash
(.venv) twilight@TWILIGHT:~/machine-learning/my-test$ python train.py --mode pretrain --epochs 30 --save-path checkpoints/pretrain.pt
Pretrain on 5 samples, device=cuda
[pretrain] epoch=1 loss/token=4.4566
[pretrain] epoch=2 loss/token=4.1604
[pretrain] epoch=3 loss/token=3.8723
[pretrain] epoch=4 loss/token=3.5616
[pretrain] epoch=5 loss/token=3.2679
[pretrain] epoch=6 loss/token=3.1066
[pretrain] epoch=7 loss/token=2.9714
[pretrain] epoch=8 loss/token=2.8605
[pretrain] epoch=9 loss/token=2.7553
[pretrain] epoch=10 loss/token=2.7150
[pretrain] epoch=11 loss/token=2.6353
[pretrain] epoch=12 loss/token=2.5430
[pretrain] epoch=13 loss/token=2.4638
[pretrain] epoch=14 loss/token=2.3636
[pretrain] epoch=15 loss/token=2.3364
[pretrain] epoch=16 loss/token=2.2623
[pretrain] epoch=17 loss/token=2.2004
[pretrain] epoch=18 loss/token=2.1116
[pretrain] epoch=19 loss/token=2.0650
[pretrain] epoch=20 loss/token=1.9818
[pretrain] epoch=21 loss/token=1.9204
[pretrain] epoch=22 loss/token=1.8916
[pretrain] epoch=23 loss/token=1.8428
[pretrain] epoch=24 loss/token=1.7865
[pretrain] epoch=25 loss/token=1.7289
[pretrain] epoch=26 loss/token=1.6852
[pretrain] epoch=27 loss/token=1.6235
[pretrain] epoch=28 loss/token=1.5621
[pretrain] epoch=29 loss/token=1.4936
[pretrain] epoch=30 loss/token=1.4947
Saved checkpoint to checkpoints/pretrain.pt
(.venv) twilight@TWILIGHT:~/machine-learning/my-test$ python train.py --mode eval --load-path checkpoints/pretrain.pt --max-gen-len 256 --max-steps 200
[eval] task=abs_value passed=False reward=0.0 code_len=256
[eval] task=reverse_string passed=False reward=0.0 code_len=256
[eval] task=factorial passed=False reward=0.0 code_len=256
[eval] task=is_palindrome passed=False reward=0.0 code_len=256
[eval] task=count_vowels passed=False reward=0.0 code_len=256
Success rate: 0/5 = 0.00

(.venv) twilight@TWILIGHT:~/machine-learning/my-test$ python train.py --mode pretrain --epochs 60 --hidden-dim 512  --save-path checkpoints/pretrain.pt
Pretrain on 5 samples, device=cuda
[pretrain] epoch=1 loss/token=4.8463
[pretrain] epoch=2 loss/token=4.4408
[pretrain] epoch=3 loss/token=3.9611
[pretrain] epoch=4 loss/token=3.6246
[pretrain] epoch=5 loss/token=3.2217
[pretrain] epoch=6 loss/token=3.1525
[pretrain] epoch=7 loss/token=3.0484
[pretrain] epoch=8 loss/token=2.8761
[pretrain] epoch=9 loss/token=2.7271
[pretrain] epoch=10 loss/token=2.5900
[pretrain] epoch=11 loss/token=2.4848
[pretrain] epoch=12 loss/token=2.3799
[pretrain] epoch=13 loss/token=2.2823
[pretrain] epoch=14 loss/token=2.1670
[pretrain] epoch=15 loss/token=2.0804
[pretrain] epoch=16 loss/token=1.9705
[pretrain] epoch=17 loss/token=1.8792
[pretrain] epoch=18 loss/token=1.7938
[pretrain] epoch=19 loss/token=1.7171
[pretrain] epoch=20 loss/token=1.6210
[pretrain] epoch=21 loss/token=1.5361
[pretrain] epoch=22 loss/token=1.4490
[pretrain] epoch=23 loss/token=1.3344
[pretrain] epoch=24 loss/token=1.3103
[pretrain] epoch=25 loss/token=1.1805
[pretrain] epoch=26 loss/token=1.1435
[pretrain] epoch=27 loss/token=1.0388
[pretrain] epoch=28 loss/token=1.0045
[pretrain] epoch=29 loss/token=0.9345
[pretrain] epoch=30 loss/token=0.8855
[pretrain] epoch=31 loss/token=0.8186
[pretrain] epoch=32 loss/token=0.7630
[pretrain] epoch=33 loss/token=0.7207
[pretrain] epoch=34 loss/token=0.6445
[pretrain] epoch=35 loss/token=0.6313
[pretrain] epoch=36 loss/token=0.5841
[pretrain] epoch=37 loss/token=0.5193
[pretrain] epoch=38 loss/token=0.5095
[pretrain] epoch=39 loss/token=0.4708
[pretrain] epoch=40 loss/token=0.4456
[pretrain] epoch=41 loss/token=0.4305
[pretrain] epoch=42 loss/token=0.4120
[pretrain] epoch=43 loss/token=0.3779
[pretrain] epoch=44 loss/token=0.3624
[pretrain] epoch=45 loss/token=0.3450
[pretrain] epoch=46 loss/token=0.3158
[pretrain] epoch=47 loss/token=0.2972
[pretrain] epoch=48 loss/token=0.2843
[pretrain] epoch=49 loss/token=0.2698
[pretrain] epoch=50 loss/token=0.2514
[pretrain] epoch=51 loss/token=0.2449
[pretrain] epoch=52 loss/token=0.2308
[pretrain] epoch=53 loss/token=0.2179
[pretrain] epoch=54 loss/token=0.2112
[pretrain] epoch=55 loss/token=0.1976
[pretrain] epoch=56 loss/token=0.1803
[pretrain] epoch=57 loss/token=0.1716
[pretrain] epoch=58 loss/token=0.1736
[pretrain] epoch=59 loss/token=0.1602
[pretrain] epoch=60 loss/token=0.1546
Saved checkpoint to checkpoints/pretrain.pt

(.venv) twilight@TWILIGHT:~/machine-learning/my-test$ python train.py --mode eval --load-path checkpoints/pretrain.pt --max-gen-len 200 --max-steps 200
[eval] inferred embed_dim=128, hidden_dim=512 from checkpoint
[eval] task=abs_value passed=True reward=1.0 code_len=49
[eval] task=reverse_string passed=True reward=1.0 code_len=54
[eval] task=factorial passed=False reward=0.0 code_len=65
[eval] task=is_palindrome passed=False reward=0.0 code_len=56
[eval] task=count_vowels passed=False reward=0.0 code_len=54
Success rate: 2/5 = 0.40
```

### 3. PPO 训练脚本（独立于 train.py）
- 新建 `ppo_train.py`：
  - 模型 ActorCritic（嵌入+GRU+策略/价值头）。
  - 从预热权重加载：先用 CharPolicy 恢复，再拷贝到 ActorCritic。
  - 采样与更新：PPO+GAE，优势归一化，梯度裁剪。
  - 动作掩码：仅允许任务字符集 + `<EOS>`。
  - 教师轨迹混合：每轮除随机采样外加入若干贪心轨迹，提供正样本。
  - 行为克隆混合：对 prompt+参考解的 teacher forcing 增加 CE 损失（`--bc-coef`）。
  - 默认较激进超参：episodes-per-update=16，teacher-episodes=8，max-steps/max-seq-len=150。
- 新建 `benchmark.py`：
  - 支持随机/模板/任意 checkpoint 的评估，兼容 CharPolicy 与 ActorCritic 的加载与解码（只输出生成代码，不含 prompt）。

  
### 4. 奖励稀疏与字符集问题
- 发现中文 prompt 导致编码错误，修复：`env.code_env` 的 vocab 包含任务 prompt 中的所有字符。
- 初期评估 0/5，因解码没带 prompt，修复解码仅返回生成部分。
- 奖励稀疏：最初仅 0/1 通过，后加入部分通过奖励（通过用例比例），再加入语法奖励（`py_compile` 成功 +0.1，且代码非空才给）。

### 5. 训练/评估中的问题与解决
- 预热加载维度不匹配：checkpoint 中 embed/hidden 维度记录缺失，通过 state_dict 形状推断并重命名 head→policy_head。
- benchmark 报错：同样处理维度推断与命名兼容。
- PPO 奖励长期为 0：
  - 加入部分通过奖励 + 语法奖励。
  - 收紧动作集，缩短生成长度。
  - 增加教师轨迹（贪心）和行为克隆项。
  - 提高 entropy/lr/更新轮数建议。
- 空代码拿奖励：语法奖励仅在代码非空时给，抑制空生成。
- 仍未取得 >0 的成功率：当前 PPO 仍只得到极低正奖励（语法奖励/部分通过少量），未收敛到可通过用例的策略。

最初版本：
```bash
(.venv) twilight@TWILIGHT:~/machine-learning/my-test$ python ppo_train.py --updates 30 --episodes-per-update 8 --max-steps 200 --max-seq-len 256 --load-pretrain checkpoints/pretrain.pt --save-path checkpoints/ppo.pt
[init] loaded 2 keys from pretrain checkpoint checkpoints/pretrain.pt
[update 1] avg_reward=0.000 last_info=count_vowels passed=False
[update 2] avg_reward=0.000 last_info=is_palindrome passed=False
[update 3] avg_reward=0.000 last_info=count_vowels passed=False
[update 4] avg_reward=0.000 last_info=reverse_string passed=False
[update 5] avg_reward=0.000 last_info=reverse_string passed=False
[update 6] avg_reward=0.000 last_info=count_vowels passed=False
[update 7] avg_reward=0.000 last_info=abs_value passed=False
[update 8] avg_reward=0.000 last_info=factorial passed=False
[update 9] avg_reward=0.000 last_info=is_palindrome passed=False
[update 10] avg_reward=0.000 last_info=abs_value passed=False
[update 11] avg_reward=0.000 last_info=abs_value passed=False
[update 12] avg_reward=0.000 last_info=factorial passed=False
[update 13] avg_reward=0.000 last_info=reverse_string passed=False
[update 14] avg_reward=0.000 last_info=factorial passed=False
[update 15] avg_reward=0.000 last_info=factorial passed=False
[update 16] avg_reward=0.000 last_info=reverse_string passed=False
[update 17] avg_reward=0.000 last_info=count_vowels passed=False
[update 18] avg_reward=0.000 last_info=is_palindrome passed=False
[update 19] avg_reward=0.000 last_info=factorial passed=False
[update 20] avg_reward=0.000 last_info=count_vowels passed=False
[update 21] avg_reward=0.000 last_info=count_vowels passed=False
[update 22] avg_reward=0.000 last_info=is_palindrome passed=False
[update 23] avg_reward=0.000 last_info=count_vowels passed=False
[update 24] avg_reward=0.000 last_info=is_palindrome passed=False
[update 25] avg_reward=0.000 last_info=reverse_string passed=False
[update 26] avg_reward=0.000 last_info=factorial passed=False
[update 27] avg_reward=0.000 last_info=factorial passed=False
[update 28] avg_reward=0.000 last_info=count_vowels passed=False
[update 29] avg_reward=0.000 last_info=is_palindrome passed=False
[update 30] avg_reward=0.000 last_info=factorial passed=False
Saved PPO checkpoint to checkpoints/ppo.pt
```

加入“能运行”的正反馈：
```bash
(.venv) twilight@TWILIGHT:~/machine-learning/my-test$ python ppo_train.py --updates 30 --episodes-per-update 8 --max-steps 150 --max-seq-len 200 --load-pretrain checkpoints/pretrain.pt --save-path checkpoint
s/ppo.pt --entropy-coef 0.1 --lr 3e-4
[init] loaded pretrain weights from checkpoints/pretrain.pt
[update 1] avg_reward=0.010 last_info=reverse_string passed=False
[update 2] avg_reward=0.020 last_info=count_vowels passed=False
[update 3] avg_reward=0.030 last_info=reverse_string passed=False
[update 4] avg_reward=0.020 last_info=factorial passed=False
[update 5] avg_reward=0.040 last_info=abs_value passed=False
[update 6] avg_reward=0.030 last_info=reverse_string passed=False
[update 7] avg_reward=0.000 last_info=reverse_string passed=False
[update 8] avg_reward=0.000 last_info=is_palindrome passed=False
[update 9] avg_reward=0.000 last_info=abs_value passed=False
[update 10] avg_reward=0.020 last_info=factorial passed=False
[update 11] avg_reward=0.000 last_info=abs_value passed=False
[update 12] avg_reward=0.000 last_info=is_palindrome passed=False
[update 13] avg_reward=0.000 last_info=is_palindrome passed=False
[update 14] avg_reward=0.000 last_info=reverse_string passed=False
[update 15] avg_reward=0.000 last_info=count_vowels passed=False
[update 16] avg_reward=0.000 last_info=factorial passed=False
[update 17] avg_reward=0.000 last_info=reverse_string passed=False
[update 18] avg_reward=0.000 last_info=abs_value passed=False
[update 19] avg_reward=0.010 last_info=abs_value passed=False
[update 20] avg_reward=0.020 last_info=reverse_string passed=False
[update 21] avg_reward=0.040 last_info=reverse_string passed=False
[update 22] avg_reward=0.000 last_info=is_palindrome passed=False
[update 23] avg_reward=0.030 last_info=reverse_string passed=False
[update 24] avg_reward=0.020 last_info=is_palindrome passed=False
[update 25] avg_reward=0.010 last_info=factorial passed=False
[update 26] avg_reward=0.000 last_info=reverse_string passed=False
[update 27] avg_reward=0.010 last_info=reverse_string passed=False
[update 28] avg_reward=0.030 last_info=factorial passed=False
[update 29] avg_reward=0.040 last_info=reverse_string passed=False
[update 30] avg_reward=0.020 last_info=factorial passed=False
Saved PPO checkpoint to checkpoints/ppo.pt
```

加入教师：
```
(.venv) twilight@TWILIGHT:~/machine-learning/my-test$  python ppo_train.py --updates 50 --episodes-per-update 16 --teacher-episodes 8 --max-steps 150 --max-seq-len 150 --load-pretrain checkpoints/pretrain.pt
 --save-path checkpoints/ppo.pt --entropy-coef 0.1 --lr 3e-4
[init] loaded pretrain weights from checkpoints/pretrain.pt
[update 1] avg_reward=0.000 last_info=abs_value passed=False
[update 2] avg_reward=0.004 last_info=abs_value passed=False
[update 3] avg_reward=0.008 last_info=count_vowels passed=False
[update 4] avg_reward=0.004 last_info=count_vowels passed=False
[update 5] avg_reward=0.000 last_info=reverse_string passed=False
[update 6] avg_reward=0.000 last_info=is_palindrome passed=False
[update 7] avg_reward=0.000 last_info=abs_value passed=False
[update 8] avg_reward=0.000 last_info=abs_value passed=False
[update 9] avg_reward=0.004 last_info=count_vowels passed=False
[update 10] avg_reward=0.000 last_info=count_vowels passed=False
[update 11] avg_reward=0.004 last_info=abs_value passed=False
[update 12] avg_reward=0.004 last_info=reverse_string passed=False
[update 13] avg_reward=0.004 last_info=reverse_string passed=False
[update 14] avg_reward=0.000 last_info=is_palindrome passed=False
[update 15] avg_reward=0.000 last_info=abs_value passed=False
[update 16] avg_reward=0.000 last_info=factorial passed=False
[update 17] avg_reward=0.000 last_info=factorial passed=False
[update 18] avg_reward=0.004 last_info=abs_value passed=False
[update 19] avg_reward=0.000 last_info=count_vowels passed=False
[update 20] avg_reward=0.000 last_info=count_vowels passed=False
[update 21] avg_reward=0.000 last_info=is_palindrome passed=False
[update 22] avg_reward=0.000 last_info=reverse_string passed=False
[update 23] avg_reward=0.000 last_info=abs_value passed=False
[update 24] avg_reward=0.000 last_info=abs_value passed=False
[update 25] avg_reward=0.008 last_info=factorial passed=False
[update 26] avg_reward=0.004 last_info=factorial passed=False
[update 27] avg_reward=0.013 last_info=factorial passed=False
[update 28] avg_reward=0.025 last_info=count_vowels passed=False
[update 29] avg_reward=0.025 last_info=is_palindrome passed=False
[update 30] avg_reward=0.029 last_info=factorial passed=False
[update 31] avg_reward=0.008 last_info=count_vowels passed=False
[update 32] avg_reward=0.008 last_info=factorial passed=False
[update 33] avg_reward=0.013 last_info=reverse_string passed=False
[update 34] avg_reward=0.017 last_info=is_palindrome passed=False
[update 35] avg_reward=0.017 last_info=factorial passed=False
```

## 现状与下一步建议
- 现状：预热贪心 2/5 成功；PPO 在多次尝试后仍未显著提升，奖励仅来自语法/极少部分通过。
- 下一步可尝试：
  1) 提高教师权重与数量：增大 `--bc-coef`，增加 `--teacher-episodes`，提高 updates。
  2) 进一步收紧动作集/长度，或在 env 中对通过 1 个用例给更高 shaping。
  3) 调 entropy/lr，分阶段降低 entropy。
  4) 若允许，可切换到 token 级（词汇更小）或直接用模板初始化输出，以降低搜索空间。

## 最近修改说明（PPO）
- 观测编码去掉了额外的 `<EOS>`（`encode_obs` 不再强行 `add_eos=True`），避免模型在每步都看到 EOS 从而倾向立即输出 EOS、获得零奖励。
- PPO 计算 logprob/value 时改为使用真实序列长度（非 padding 位置）取最后一个 token 的 logits/values，确保优势/价值对应的是实际动作而不是填充 token。

```powershell
(.venv) twilight@TWILIGHT:~/machine-learning/my-test$ python ppo_train.py --updates 30 --episodes-per-update 16 --teacher-episodes 8 --max-steps 150 --max-seq-len 150 --load-pretrain checkpoints/pretrain.pt --save-path checkpoints/ppo.pt --entropy-coef 0.1 --lr 3e-4
[init] loaded pretrain weights from checkpoints/pretrain.pt
[update 1] avg_reward=0.183 last_info=reverse_string passed=True
[update 2] avg_reward=0.167 last_info=reverse_string passed=True
[update 3] avg_reward=0.175 last_info=is_palindrome passed=False
[update 4] avg_reward=0.171 last_info=abs_value passed=True
[update 5] avg_reward=0.183 last_info=count_vowels passed=False
[update 6] avg_reward=0.238 last_info=factorial passed=False
[update 7] avg_reward=0.138 last_info=count_vowels passed=False
[update 8] avg_reward=0.208 last_info=reverse_string passed=True
[update 9] avg_reward=0.179 last_info=reverse_string passed=True
[update 10] avg_reward=0.263 last_info=factorial passed=False
[update 11] avg_reward=0.113 last_info=abs_value passed=True
[update 12] avg_reward=0.138 last_info=reverse_string passed=True
[update 13] avg_reward=0.254 last_info=reverse_string passed=True
[update 14] avg_reward=0.092 last_info=count_vowels passed=False
[update 15] avg_reward=0.296 last_info=reverse_string passed=True
[update 16] avg_reward=0.263 last_info=abs_value passed=True
[update 17] avg_reward=0.421 last_info=abs_value passed=True
[update 18] avg_reward=0.300 last_info=reverse_string passed=True
[update 19] avg_reward=0.213 last_info=count_vowels passed=False
[update 20] avg_reward=0.092 last_info=count_vowels passed=False
[update 21] avg_reward=0.221 last_info=count_vowels passed=False
[update 22] avg_reward=0.300 last_info=factorial passed=False
[update 23] avg_reward=0.221 last_info=factorial passed=False
[update 24] avg_reward=0.150 last_info=abs_value passed=True
[update 25] avg_reward=0.129 last_info=is_palindrome passed=False
[update 26] avg_reward=0.254 last_info=reverse_string passed=True
[update 27] avg_reward=0.183 last_info=count_vowels passed=False
[update 28] avg_reward=0.250 last_info=count_vowels passed=False
[update 29] avg_reward=0.217 last_info=abs_value passed=True
[update 30] avg_reward=0.138 last_info=reverse_string passed=True
Saved PPO checkpoint to checkpoints/ppo.pt
(.venv) twilight@TWILIGHT:~/machine-learning/my-test$ python benchmark.py --mode model --ckpt checkpoints/ppo.pt --max-gen-len 200
[model] task=abs_value passed=True reward=1.0 code_len=50
[model] task=reverse_string passed=True reward=1.0 code_len=55
[model] task=factorial passed=False reward=0.0 code_len=136
[model] task=is_palindrome passed=False reward=0.0 code_len=154
[model] task=count_vowels passed=False reward=0.0 code_len=153
[model] success=2/5 = 0.40
(.venv) twilight@TWILIGHT:~/machine-learning/my-test$ python benchmark.py --mode model --ckpt checkpoints/ppo.pt --max-gen-len 150
[model] task=abs_value passed=True reward=1.0 code_len=50
[model] task=reverse_string passed=True reward=1.0 code_len=55
[model] task=factorial passed=False reward=0.0 code_len=120
[model] task=is_palindrome passed=False reward=0.1 code_len=116
[model] task=count_vowels passed=False reward=0.0 code_len=103
[model] success=2/5 = 0.40
```

调整参数后：
```bash
(.venv) twilight@TWILIGHT:~/machine-learning/my-test$ python ppo_train.py --updates 30 --episodes-per-update 16 --teacher-episodes 8 --max-steps 150 --max-seq-len 150 --load-pretrain checkpoints/pretrain.pt 
--save-path checkpoints/ppo.pt --entropy-coef 0.1 --lr 3e-4
[init] loaded pretrain weights from checkpoints/pretrain.pt
[update 1] avg_reward=0.108 last_info=is_palindrome passed=False
[update 2] avg_reward=0.004 last_info=abs_value passed=False
[update 3] avg_reward=0.029 last_info=is_palindrome passed=False
[update 4] avg_reward=0.342 last_info=reverse_string passed=True
[update 5] avg_reward=0.175 last_info=abs_value passed=True
[update 6] avg_reward=0.183 last_info=abs_value passed=True
[update 7] avg_reward=0.263 last_info=reverse_string passed=True
[update 8] avg_reward=0.296 last_info=abs_value passed=True
[update 9] avg_reward=0.308 last_info=reverse_string passed=True
[update 10] avg_reward=0.142 last_info=reverse_string passed=True
[update 11] avg_reward=0.213 last_info=factorial passed=False
[update 12] avg_reward=0.346 last_info=is_palindrome passed=False
[update 13] avg_reward=0.046 last_info=count_vowels passed=False
[update 14] avg_reward=0.383 last_info=abs_value passed=True
[update 15] avg_reward=0.133 last_info=is_palindrome passed=False
[update 16] avg_reward=0.133 last_info=count_vowels passed=False
[update 17] avg_reward=0.125 last_info=reverse_string passed=True
[update 18] avg_reward=0.338 last_info=count_vowels passed=False
[update 19] avg_reward=0.133 last_info=count_vowels passed=False
[update 20] avg_reward=0.183 last_info=factorial passed=False
[update 21] avg_reward=0.258 last_info=reverse_string passed=True
[update 22] avg_reward=0.179 last_info=is_palindrome passed=False
[update 23] avg_reward=0.258 last_info=is_palindrome passed=False
[update 24] avg_reward=0.183 last_info=is_palindrome passed=False
[update 25] avg_reward=0.304 last_info=factorial passed=False
[update 26] avg_reward=0.171 last_info=abs_value passed=True
[update 27] avg_reward=0.138 last_info=abs_value passed=True
[update 28] avg_reward=0.054 last_info=is_palindrome passed=False
[update 29] avg_reward=0.004 last_info=factorial passed=False
[update 30] avg_reward=0.258 last_info=abs_value passed=True
Saved PPO checkpoint to checkpoints/ppo.pt
(.venv) twilight@TWILIGHT:~/machine-learning/my-test$ python benchmark.py --mode model --ckpt checkpoints/ppo.pt --max-gen-len 150
[model] task=abs_value passed=True reward=1.0 code_len=50
[model] task=reverse_string passed=True reward=1.0 code_len=55
[model] task=factorial passed=False reward=0.0 code_len=120
[model] task=is_palindrome passed=False reward=0.1 code_len=116
[model] task=count_vowels passed=False reward=0.0 code_len=103
[model] success=2/5 = 0.40
```


2026.1.5

### 6. 近期调参与实验记录（本次对话概览）
- 任务与脚本：任务扩展到 13；支持 train/val 划分，`--no-split` 为全量训练评估。新增一键脚本 `run_all.sh`（标准）、`run_all_fast.sh`（极简）、`run_all_medium.sh`（可调参），日志统一写到 `results/` 单一 txt。
- 预热调参：hidden_dim 提升到 512，epochs 提升到 80，长度上限收紧到 120/150；无划分时预热成功率提升到约 4–5/13（见 results/run_medium_local_*），但在最新强配置下预热评估仍可能 0/13（依赖生成长度）。
- PPO 调参（本地无划分）：逐步提高教师信号（teacher-episodes 4→6→8→10），`bc-coef 1.0→2.0→3.0`，更新步数 20→30→50，长度 150→180→200/220，学习率 1e-4→5e-4→7e-4；最佳 run 达到约 12/13 成功（results/run_medium_local_20260106_004453.txt，仅 fizzbuzz_str 未过），此前有 9/13 与 11/13 的中间版本。
- 划分模式（10/3）：最佳仍 0/3（results/run_medium_local_20260105_143903.txt），说明模型记忆训练任务但对验证任务泛化弱。
- MBPP 训练/评估：使用更强预热+PPO（30 updates、长序列、强教师）验证集仍 0/90（results/run_medium_mbpp_20260105_150151.txt），字符级小模型对公开题库几乎无泛化。
- 典型问题与处理：
  - vocab mismatch：fast 脚本在 MBPP 模式下加载了本地预热权重，shape 不符；需使用对应数据集的预热/ppo 权重，或改用 medium/full 脚本。
  - HF Hub 警告/SSL 抖动：网络不稳或未设置 `HF_TOKEN`，可忽略或配置 `HF_TOKEN`。
- 交付建议：用本地任务无划分的高成功率（如 11–12/13）作为主实验，附带划分/MBPP 的 0 成绩作为“泛化不足”证据，在报告中说明限制与改进方向（更多任务、token 级建模、更大模型）。

### 7. 后续额外调参记录
- 更强一版本地无划分：预热 embed=256/hidden=512，PPO updates=50、teacher-episodes=10、max-steps/seq-len/gen-len=220、lr=7e-4，达到 12/13（fizzbuzz_str 未过），见 results/run_medium_local_20260106_004453.txt。
- 若再尝试：可继续放宽长度至 240，或加 teacher-episodes=12、bc-coef=4.0，但耗时增加，收益不确定。
