"""
最小 PPO 训练脚本（字符级代码生成，基于 env/code_env.py）。
不修改 train.py，独立文件便于与基线对比。
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical

from env.code_env import CodeGenEnv
from models.net import CharPolicy
from tasks.tasks import TASKS


@dataclass
class Tokenizer:
    vocab: List[str]
    eos_token: str

    def __post_init__(self):
        self.token_to_id = {tok: i for i, tok in enumerate(self.vocab)}

    def encode(self, text: str, add_eos: bool = True, max_len: int | None = None) -> List[int]:
        ids = []
        for ch in text:
            if ch not in self.token_to_id:
                continue  # 忽略 OOV 字符（按 vocab 已收录 prompt 字符，通常不会发生）
            ids.append(self.token_to_id[ch])
            if max_len and len(ids) >= max_len:
                break
        if add_eos and (max_len is None or len(ids) < max_len):
            ids.append(self.token_to_id[self.eos_token])
        return ids


class ActorCritic(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.policy_head = nn.Linear(hidden_dim, vocab_size)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids: torch.Tensor):
        # input_ids: (batch, seq_len)
        x = self.embed(input_ids)
        h, _ = self.gru(x)
        logits = self.policy_head(h)
        values = self.value_head(h)
        return logits, values


def build_model_from_pretrain(ckpt_path: Path, vocab_size: int):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["state_dict"]
    # 从权重推断维度，避免缺少元数据时的 mismatch
    embed_dim = state["embed.weight"].shape[1]
    hidden_dim = state["gru.weight_hh_l0"].shape[0] // 3

    # 先用 CharPolicy 恢复完整权重，再映射到 ActorCritic。严格按 checkpoint 中的维度创建模型，避免不匹配。
    vocab_size_ckpt = len(ckpt["vocab"])
    if vocab_size_ckpt != vocab_size:
        print(f"[init] warning: vocab size mismatch ckpt={vocab_size_ckpt} current={vocab_size}")
    char_model = CharPolicy(vocab_size_ckpt, embed_dim=embed_dim, hidden_dim=hidden_dim)
    char_model.load_state_dict(state)

    model = ActorCritic(vocab_size_ckpt, embed_dim=embed_dim, hidden_dim=hidden_dim)
    # 拷贝参数
    model.embed.load_state_dict(char_model.embed.state_dict())
    model.gru.load_state_dict(char_model.gru.state_dict())
    model.policy_head.load_state_dict(char_model.head.state_dict())
    print(f"[init] loaded pretrain weights from {ckpt_path}")
    return model


def encode_obs(tokenizer: Tokenizer, obs: dict, max_seq_len: int) -> torch.Tensor:
    seq = f"{obs['prompt']}\n{obs['code']}"
    # 当前状态用于预测下一个 token，不需要额外的 EOS，否则模型会在 EOS 上预测下一步而偏向提前结束
    ids = tokenizer.encode(seq, add_eos=False, max_len=max_seq_len)
    return torch.tensor([ids], dtype=torch.long)


def collect_trajectory(
    env: CodeGenEnv,
    model: ActorCritic,
    tokenizer: Tokenizer,
    device: torch.device,
    max_seq_len: int,
    allowed_ids: set[int],
    greedy: bool = False,
):
    obs, info = env.reset()
    done = False
    truncated = False
    trajectory = {
        "states": [],
        "actions": [],
        "logprobs": [],
        "values": [],
        "rewards": [],
        "masks": [],
    }
    while not (done or truncated):
        ids = encode_obs(tokenizer, obs, max_seq_len=max_seq_len).to(device)
        logits, values = model(ids)
        logits_last = mask_logits(logits[0, -1], allowed_ids)
        value_last = values[0, -1]
        dist = Categorical(logits=logits_last)
        if greedy:
            action = torch.argmax(logits_last)
            logprob = dist.log_prob(action)
        else:
            action = dist.sample()
            logprob = dist.log_prob(action)

        obs, reward, done, truncated, info = env.step(action.item())

        trajectory["states"].append(ids.squeeze(0).cpu().tolist())
        trajectory["actions"].append(action)
        trajectory["logprobs"].append(logprob)
        trajectory["values"].append(value_last)
        trajectory["rewards"].append(torch.tensor([reward], device=device))
        mask = 0.0 if (done or truncated) else 1.0
        trajectory["masks"].append(torch.tensor([mask], device=device))
    return trajectory, info


def compute_gae(rewards, values, masks, gamma: float, lam: float):
    gae = 0
    returns = []
    values = [v.detach() for v in values]
    values = values + [torch.zeros_like(values[0])]
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * lam * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns


def mask_logits(logits: torch.Tensor, allowed_ids: set[int]) -> torch.Tensor:
    if len(allowed_ids) == 0:
        return logits
    mask = torch.full_like(logits, -1e9)
    idx = torch.tensor(list(allowed_ids), dtype=torch.long, device=logits.device)
    mask[idx] = 0.0
    return logits + mask


def ppo_update(
    model,
    optimizer,
    batch_logprobs,
    batch_actions,
    batch_states,
    batch_state_lens,
    batch_returns,
    batch_advantages,
    clip_coef: float,
    value_coef: float,
    entropy_coef: float,
    epochs: int,
    minibatch_size: int,
    device: torch.device,
    bc_inputs: torch.Tensor | None = None,
    bc_targets: torch.Tensor | None = None,
    bc_coef: float = 0.0,
):
    total_steps = batch_states.size(0)
    for _ in range(epochs):
        indices = torch.randperm(total_steps)
        for start in range(0, total_steps, minibatch_size):
            end = start + minibatch_size
            mb_idx = indices[start:end]
            states = batch_states[mb_idx].to(device)
            state_lens = batch_state_lens[mb_idx].to(device)
            actions = batch_actions[mb_idx].to(device)
            old_logprobs = batch_logprobs[mb_idx].to(device)
            returns = batch_returns[mb_idx].to(device)
            advantages = batch_advantages[mb_idx].to(device)

            logits, values = model(states)
            batch_indices = torch.arange(states.size(0), device=device)
            last_token_idx = state_lens.to(device) - 1  # gather logits at each sequence's true end
            logits_last = logits[batch_indices, last_token_idx, :]
            values_last = values[batch_indices, last_token_idx, 0]

            dist = Categorical(logits=logits_last)
            logprobs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratios = (logprobs - old_logprobs).exp()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - clip_coef, 1.0 + clip_coef) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = (returns - values_last).pow(2).mean()

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            # 可选的行为克隆项：用教师数据的下一 token 监督指导策略
            if bc_inputs is not None and bc_targets is not None and bc_coef > 0:
                logits_bc, _ = model(bc_inputs.to(device))
                bc_loss = nn.CrossEntropyLoss(ignore_index=-100)(
                    logits_bc.view(-1, logits_bc.size(-1)), bc_targets.to(device).view(-1)
                )
                loss = loss + bc_coef * bc_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()


def flatten_sequences(seqs: List[List[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length sequences and return their true lengths."""
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    max_len = int(lengths.max().item())
    padded = []
    for s in seqs:
        pad_len = max_len - len(s)
        padded.append(s + [pad_id] * pad_len)
    return torch.tensor(padded, dtype=torch.long), lengths


def build_teacher_batch(tokenizer: Tokenizer, max_seq_len: int, pad_id: int):
    """
    使用任务的 prompt+canonical_solution 构建一个 teacher forcing 批次，提供监督信号。
    """
    inputs = []
    targets = []
    for task in TASKS:
        seq = f"{task.prompt}\n{task.canonical_solution}"
        token_ids = tokenizer.encode(seq, add_eos=True, max_len=max_seq_len)
        inp = token_ids[:-1]
        tgt = token_ids[1:]
        inputs.append(inp)
        targets.append(tgt)
    max_len = max(len(x) for x in inputs)
    padded_inp = []
    padded_tgt = []
    for inp, tgt in zip(inputs, targets):
        pad_len = max_len - len(inp)
        padded_inp.append(inp + [pad_id] * pad_len)
        padded_tgt.append(tgt + [-100] * pad_len)  # -100 供 CE ignore
    return torch.tensor(padded_inp, dtype=torch.long), torch.tensor(padded_tgt, dtype=torch.long)


def train_ppo(args):
    env = CodeGenEnv(max_steps=args.max_steps)
    tokenizer = Tokenizer(env.vocab, env.eos_token)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 构建受限动作集：仅允许任务 prompt 和参考解中出现过的字符
    char_set = set()
    for t in TASKS:
        char_set.update(t.prompt)
        char_set.update(t.canonical_solution)
    allowed_ids = {tokenizer.token_to_id[ch] for ch in char_set if ch in tokenizer.token_to_id}
    allowed_ids.add(tokenizer.token_to_id[env.eos_token])

    if args.load_pretrain and Path(args.load_pretrain).exists():
        model = build_model_from_pretrain(Path(args.load_pretrain), vocab_size=len(env.vocab))
    else:
        model = ActorCritic(len(env.vocab), embed_dim=args.embed_dim, hidden_dim=args.hidden_dim)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for update in range(1, args.updates + 1):
        trajectories = []
        infos = []
        for _ in range(args.episodes_per_update):
            traj, info = collect_trajectory(
                env,
                model,
                tokenizer,
                device=device,
                max_seq_len=args.max_seq_len,
                allowed_ids=allowed_ids,
                greedy=False,
            )
            trajectories.append(traj)
            infos.append(info)

        # 教师轨迹（贪心）用于提供正奖励样本
        for _ in range(args.teacher_episodes):
            traj, info = collect_trajectory(
                env,
                model,
                tokenizer,
                device=device,
                max_seq_len=args.max_seq_len,
                allowed_ids=allowed_ids,
                greedy=True,
            )
            trajectories.append(traj)
            infos.append(info)

        # 展平
        states = []
        actions = []
        logprobs = []
        rewards = []
        values = []
        masks = []
        for traj in trajectories:
            rewards.extend(traj["rewards"])
            values.extend(traj["values"])
            logprobs.extend(traj["logprobs"])
            masks.extend(traj["masks"])
            states.extend(traj["states"])
            actions.extend(traj["actions"])

        # 计算 returns / advantages
        values_detached = [v.detach() for v in values]
        returns = compute_gae(rewards, values_detached, masks, gamma=args.gamma, lam=args.lam)
        advantages = [ret - val for ret, val in zip(returns, values_detached)]
        advantages = torch.stack(advantages).squeeze(-1)
        returns = torch.stack(returns).squeeze(-1).detach()
        logprobs = torch.stack(logprobs).detach()
        actions = torch.stack(actions).detach()
        pad_id = tokenizer.token_to_id[tokenizer.eos_token]
        states, state_lens = flatten_sequences(states, pad_id=pad_id)

        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 构建 teacher forcing 批次
        bc_inputs, bc_targets = build_teacher_batch(
            tokenizer, max_seq_len=args.max_seq_len, pad_id=pad_id
        )

        ppo_update(
            model,
            optimizer,
            batch_logprobs=logprobs,
            batch_actions=actions,
            batch_states=states,
            batch_state_lens=state_lens,
            batch_returns=returns,
            batch_advantages=advantages,
            clip_coef=args.clip,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
            device=device,
            bc_inputs=bc_inputs.to(device),
            bc_targets=bc_targets.to(device),
            bc_coef=args.bc_coef,
        )

        # 简单日志：平均 episodic reward
        ep_rewards = [sum([r.item() for r in traj["rewards"]]) for traj in trajectories]
        avg_reward = sum(ep_rewards) / len(ep_rewards)
        print(
            f"[update {update}] avg_reward={avg_reward:.3f} "
            f"last_info={infos[-1].get('task')} passed={infos[-1].get('passed', False)}"
        )

    # 训练结束保存
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    embed_dim = model.embed.embedding_dim
    hidden_dim = model.gru.hidden_size
    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocab": env.vocab,
            "eos_token": env.eos_token,
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
        },
        args.save_path,
    )
    print(f"Saved PPO checkpoint to {args.save_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=50)
    parser.add_argument("--episodes-per-update", type=int, default=16)
    parser.add_argument("--teacher-episodes", type=int, default=8, help="每轮额外的贪心教师轨迹数量")
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--max-seq-len", type=int, default=150)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=32)
    parser.add_argument("--load-pretrain", type=str, default="checkpoints/pretrain.pt")
    parser.add_argument("--save-path", type=str, default="checkpoints/ppo.pt")
    parser.add_argument("--bc-coef", type=float, default=0.5, help="行为克隆损失系数")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_ppo(args)
