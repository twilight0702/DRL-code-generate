"""
极简驱动脚本：
- 随机策略：验证环境能跑通，奖励应长期为 0。
- greedy-template：直接输出 canonical_solution，代表上限基线（奖励应为 1）。
- 监督预热：teacher forcing 训练字符级模型，降低 RL 冷启动难度。
后续可替换/叠加 PPO/A2C。
"""

import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from env.code_env import CodeGenEnv
from models.net import CharPolicy
from tasks.tasks import TASKS, split_tasks
from mbpp_tasks import load_mbpp_tasks


@dataclass
class Tokenizer:
    vocab: List[str]
    eos_token: str

    def __post_init__(self):
        self.token_to_id = {tok: i for i, tok in enumerate(self.vocab)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}

    def encode(self, text: str, add_eos: bool = True) -> List[int]:
        ids = []
        for ch in text:
            if ch not in self.token_to_id:
                raise ValueError(f"Character {repr(ch)} not in vocabulary")
            ids.append(self.token_to_id[ch])
        if add_eos:
            ids.append(self.token_to_id[self.eos_token])
        return ids

    def decode(self, ids: List[int]) -> str:
        toks = []
        for i in ids:
            tok = self.id_to_token.get(i, "")
            if tok == self.eos_token:
                break
            toks.append(tok)
        return "".join(toks)


class TaskDataset(Dataset):
    def __init__(self, tasks, tokenizer: Tokenizer, eos_as_pad: bool = True):
        self.samples = []
        self.pad_id = tokenizer.token_to_id[tokenizer.eos_token] if eos_as_pad else 0
        for task in tasks:
            # 将 prompt 作为条件前缀，帮助模型区分任务
            seq = f"{task.prompt}\n{task.canonical_solution}"
            token_ids = tokenizer.encode(seq, add_eos=True)
            # 输入/目标，长度相同
            self.samples.append((token_ids[:-1], token_ids[1:]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def make_collate_fn(pad_id: int):
    def collate_batch(batch):
        max_len = max(len(inp) for inp, _ in batch)
        inputs, targets = [], []
        for inp, tgt in batch:
            pad_len = max_len - len(inp)
            inputs.append(inp + [pad_id] * pad_len)
            targets.append(tgt + [-100] * pad_len)  # -100 会被 CE 忽略
        inputs_tensor = torch.tensor(inputs, dtype=torch.long)
        targets_tensor = torch.tensor(targets, dtype=torch.long)
        return inputs_tensor, targets_tensor

    return collate_batch


def random_rollout(episodes: int, max_steps: int, tasks: List, vocab_tasks: List) -> Tuple[int, int]:
    env = CodeGenEnv(tasks=tasks, max_steps=max_steps, vocab_tasks=vocab_tasks)
    successes = 0
    for ep in range(episodes):
        _, info = env.reset()
        done = False
        truncated = False
        total_reward = 0.0
        steps = 0
        while not (done or truncated):
            action = env.action_space.sample()
            _, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
        passed = info.get("passed", False)
        successes += int(passed)
        print(
            f"[random ep {ep}] task={info.get('task')} passed={passed} "
            f"reward={total_reward} steps={steps}"
        )
    return successes, episodes


def greedy_template(episodes: int, max_steps: int, tasks: List, vocab_tasks: List) -> Tuple[int, int]:
    """
    直接输出 canonical_solution，作为上限基线。
    不通过环境 step，而是复用 env 的评估逻辑。
    """
    env = CodeGenEnv(tasks=tasks, max_steps=max_steps, vocab_tasks=vocab_tasks)
    successes = 0
    for ep in range(episodes):
        _, info = env.reset()
        task_name = info["task"]
        task = env._task  # noqa: SLF001 - 直接读取当前任务
        code = task.canonical_solution
        reward, passed = env._evaluate_code(code, task)
        successes += int(passed)
        steps = len(code)
        print(
            f"[template ep {ep}] task={task_name} passed={passed} "
            f"reward={reward} steps~{steps}"
        )
    return successes, episodes


def pretrain_teacher_forcing(
    epochs: int,
    batch_size: int,
    lr: float,
    embed_dim: int,
    hidden_dim: int,
    save_path: Path | None,
    tasks: List,
    vocab_tasks: List,
) -> CharPolicy:
    env = CodeGenEnv(tasks=tasks, vocab_tasks=vocab_tasks)
    tokenizer = Tokenizer(vocab=env.vocab, eos_token=env.eos_token)
    dataset = TaskDataset(tasks, tokenizer)
    collate_fn = make_collate_fn(dataset.pad_id)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CharPolicy(len(env.vocab), embed_dim=embed_dim, hidden_dim=hidden_dim).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"Pretrain on {len(dataset)} samples, device={device}")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits, _ = model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()
        avg_loss = total_loss / total_tokens
        print(f"[pretrain] epoch={epoch} loss/token={avg_loss:.4f}")
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "vocab": env.vocab,
                "eos_token": env.eos_token,
                "embed_dim": embed_dim,
                "hidden_dim": hidden_dim,
            },
            save_path,
        )
        print(f"Saved checkpoint to {save_path}")
    return model


def greedy_decode(
    model: CharPolicy, tokenizer: Tokenizer, prefix: str = "", max_len: int = 256
) -> str:
    """
    带任务前缀的贪心解码：先喂入 prefix，继续自回归生成直到 <EOS> 或长度上限。
    """
    model.eval()
    device = next(model.parameters()).device
    generated = []
    hidden = None

    # 先跑一遍前缀，设置 hidden state
    if prefix:
        prefix_ids = tokenizer.encode(prefix, add_eos=False)
        for tid in prefix_ids:
            input_id = torch.tensor([[tid]], dtype=torch.long, device=device)
            _, hidden = model.step(input_id, hidden)
        last_id = prefix_ids[-1]
    else:
        last_id = tokenizer.token_to_id[tokenizer.eos_token]

    input_id = torch.tensor([[last_id]], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_len):
            logits, hidden = model.step(input_id, hidden)
            next_id = int(torch.argmax(logits[0, -1]).item())
            if tokenizer.id_to_token.get(next_id) == tokenizer.eos_token:
                break
            generated.append(next_id)
            input_id = torch.tensor([[next_id]], dtype=torch.long, device=device)
    return tokenizer.decode(generated)


def evaluate_checkpoint(
    ckpt_path: Path,
    tasks: List,
    max_len: int = 256,
    max_steps: int = 200,
) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    vocab = ckpt["vocab"]
    eos_token = ckpt["eos_token"]
    # 兼容老的 checkpoint：如无 embed/hidden 维度记录则从参数形状推断
    if "embed_dim" in ckpt and "hidden_dim" in ckpt:
        embed_dim = ckpt["embed_dim"]
        hidden_dim = ckpt["hidden_dim"]
    else:
        state = ckpt["state_dict"]
        embed_dim = state["embed.weight"].shape[1]
        hidden_dim = state["gru.weight_hh_l0"].shape[0] // 3
        print(f"[eval] inferred embed_dim={embed_dim}, hidden_dim={hidden_dim} from checkpoint")
    tokenizer = Tokenizer(vocab=vocab, eos_token=eos_token)

    model = CharPolicy(len(vocab), embed_dim=embed_dim, hidden_dim=hidden_dim)
    model.load_state_dict(ckpt["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    env = CodeGenEnv(tasks=tasks, max_steps=max_steps, vocab_tasks=TASKS)
    successes = 0
    for task in tasks:
        prefix = f"{task.prompt}\n"
        code = greedy_decode(model, tokenizer, prefix=prefix, max_len=max_len)
        reward, passed = env._evaluate_code(code, task)
        successes += int(passed)
        print(f"[eval] task={task.name} passed={passed} reward={reward} code_len={len(code)}")
    print(f"Success rate: {successes}/{len(tasks)} = {successes/len(tasks):.2f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument(
        "--dataset",
        choices=["local", "mbpp"],
        default="local",
        help="训练/评估数据集来源",
    )
    parser.add_argument(
        "--mode",
        choices=["random", "template", "pretrain", "eval"],
        default="random",
        help="运行模式",
    )
    parser.add_argument("--epochs", type=int, default=20, help="pretrain 轮数")
    parser.add_argument("--batch-size", type=int, default=4, help="pretrain batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="pretrain 学习率")
    parser.add_argument("--embed-dim", type=int, default=128, help="字符嵌入维度")
    parser.add_argument("--hidden-dim", type=int, default=256, help="GRU 隐藏维度")
    parser.add_argument(
        "--save-path",
        type=Path,
        default=Path("checkpoints/pretrain.pt"),
        help="pretrain 模型保存路径",
    )
    parser.add_argument(
        "--load-path",
        type=Path,
        default=None,
        help="评估时加载的 checkpoint 路径",
    )
    parser.add_argument("--max-gen-len", type=int, default=256, help="贪心解码最大长度")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--split-seed", type=int, default=42, help="训练/验证划分随机种子")
    parser.add_argument("--no-split", action="store_true", help="不做训练/验证划分")
    parser.add_argument("--mbpp-train-split", type=str, default="train")
    parser.add_argument("--mbpp-eval-split", type=str, default="validation")
    parser.add_argument("--mbpp-max-samples", type=int, default=0)
    parser.add_argument("--mbpp-seed", type=int, default=42)
    parser.add_argument("--mbpp-use-challenge-tests", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.dataset == "mbpp":
        train_tasks = load_mbpp_tasks(
            split=args.mbpp_train_split,
            max_samples=args.mbpp_max_samples,
            seed=args.mbpp_seed,
            use_challenge_tests=args.mbpp_use_challenge_tests,
        )
        val_tasks = load_mbpp_tasks(
            split=args.mbpp_eval_split,
            max_samples=args.mbpp_max_samples,
            seed=args.mbpp_seed,
            use_challenge_tests=args.mbpp_use_challenge_tests,
        )
        vocab_tasks = train_tasks + val_tasks
    else:
        if args.no_split:
            train_tasks, val_tasks = TASKS, TASKS
        else:
            train_tasks, val_tasks = split_tasks(
                TASKS, train_ratio=args.train_ratio, seed=args.split_seed
            )
        vocab_tasks = TASKS
    if args.mode == "random":
        success, total = random_rollout(
            episodes=args.episodes,
            max_steps=args.max_steps,
            tasks=val_tasks,
            vocab_tasks=vocab_tasks,
        )
        print(f"Success rate: {success}/{total} = {success/total:.2f}")
    elif args.mode == "template":
        success, total = greedy_template(
            episodes=args.episodes,
            max_steps=args.max_steps,
            tasks=val_tasks,
            vocab_tasks=vocab_tasks,
        )
        print(f"Success rate: {success}/{total} = {success/total:.2f}")
    elif args.mode == "pretrain":
        pretrain_teacher_forcing(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            save_path=args.save_path,
            tasks=train_tasks,
            vocab_tasks=vocab_tasks,
        )
    else:  # eval
        if args.load_path is None:
            raise ValueError("--load-path is required in eval mode")
        evaluate_checkpoint(
            ckpt_path=args.load_path,
            tasks=val_tasks,
            max_len=args.max_gen_len,
            max_steps=args.max_steps,
        )
