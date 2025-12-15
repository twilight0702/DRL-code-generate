"""
基准评测脚本：随机/模板/预热或 PPO 模型的通过率比较。
不修改 train.py，方便独立对比。
"""

import argparse
from pathlib import Path

import torch

from env.code_env import CodeGenEnv
from ppo_train import ActorCritic, Tokenizer
from models.net import CharPolicy
from tasks.tasks import TASKS


def greedy_decode(model, tokenizer: Tokenizer, prompt: str, max_len: int = 256):
    """
    兼容 CharPolicy（预热）与 ActorCritic（PPO），使用贪心解码。
    """
    model.eval()
    device = next(model.parameters()).device
    ids = tokenizer.encode(prompt, add_eos=False, max_len=max_len)
    prefix_len = len(ids)
    with torch.no_grad():
        for _ in range(max_len - len(ids)):
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            out = model(input_ids)
            if isinstance(out, tuple) and len(out) == 2:
                logits = out[0]
            else:
                logits = out
            next_id = int(torch.argmax(logits[0, -1]).item())
            if tokenizer.vocab[next_id] == tokenizer.eos_token:
                break
            ids.append(next_id)
    # 仅返回生成部分（不含 prompt），并剔除 eos
    gen_ids = [i for i in ids[prefix_len:] if tokenizer.vocab[i] != tokenizer.eos_token]
    return "".join(tokenizer.vocab[i] for i in gen_ids)


def load_model_checkpoint(ckpt_path: Path) -> tuple[object, Tokenizer]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["state_dict"]
    vocab = ckpt["vocab"]
    eos_token = ckpt["eos_token"]
    tokenizer = Tokenizer(vocab=vocab, eos_token=eos_token)

    # 判断是预热（CharPolicy）还是 PPO（ActorCritic）
    if any(k.startswith("value_head") for k in state.keys()):
        embed_dim = state["embed.weight"].shape[1]
        hidden_dim = state["gru.weight_hh_l0"].shape[0] // 3
        model = ActorCritic(len(vocab), embed_dim=embed_dim, hidden_dim=hidden_dim)
        # 兼容 head.* -> policy_head.* 的命名
        rename_map = {"head.weight": "policy_head.weight", "head.bias": "policy_head.bias"}
        for src, dst in rename_map.items():
            if src in state:
                state[dst] = state[src]
    else:
        embed_dim = state["embed.weight"].shape[1]
        hidden_dim = state["gru.weight_hh_l0"].shape[0] // 3
        model = CharPolicy(len(vocab), embed_dim=embed_dim, hidden_dim=hidden_dim)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] missing keys: {missing}")
    if unexpected:
        print(f"[warn] unexpected keys: {unexpected}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer


def eval_random(episodes: int, max_steps: int):
    env = CodeGenEnv(max_steps=max_steps)
    successes = 0
    for ep in range(episodes):
        obs, info = env.reset()
        done = truncated = False
        while not (done or truncated):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
        successes += int(info.get("passed", False))
    print(f"[random] success={successes}/{episodes} = {successes/episodes:.2f}")


def eval_template(episodes: int, max_steps: int):
    env = CodeGenEnv(max_steps=max_steps)
    successes = 0
    for ep in range(episodes):
        _, info = env.reset()
        task = env._task  # noqa: SLF001 - 直接读取当前任务
        code = task.canonical_solution
        reward, passed = env._evaluate_code(code, task)
        successes += int(passed)
        print(f"[template ep {ep}] task={task.name} passed={passed} reward={reward}")
    print(f"[template] success={successes}/{episodes} = {successes/episodes:.2f}")


def eval_model(ckpt_path: Path, max_len: int, max_steps: int):
    model, tokenizer = load_model_checkpoint(ckpt_path)
    env = CodeGenEnv(max_steps=max_steps)
    successes = 0
    for task in TASKS:
        prompt = f"{task.prompt}\n"
        code = greedy_decode(model, tokenizer, prompt=prompt, max_len=max_len)
        reward, passed = env._evaluate_code(code, task)
        successes += int(passed)
        print(f"[model] task={task.name} passed={passed} reward={reward} code_len={len(code)}")
    print(f"[model] success={successes}/{len(TASKS)} = {successes/len(TASKS):.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["random", "template", "model"], default="random")
    parser.add_argument("--episodes", type=int, default=5, help="random/template 模式的运行回合数")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-gen-len", type=int, default=256)
    parser.add_argument("--ckpt", type=Path, default=Path("checkpoints/pretrain.pt"))
    args = parser.parse_args()

    if args.mode == "random":
        eval_random(args.episodes, args.max_steps)
    elif args.mode == "template":
        eval_template(args.episodes, args.max_steps)
    else:
        eval_model(args.ckpt, args.max_gen_len, args.max_steps)


if __name__ == "__main__":
    main()
