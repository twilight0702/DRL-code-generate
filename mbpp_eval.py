"""
Evaluate a checkpoint on the MBPP dataset (external evaluation set).
This does not affect training; it only measures generalization.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from env.code_env import CodeGenEnv
from models.net import CharPolicy
from ppo_train import ActorCritic, Tokenizer
from tasks.tasks import TASKS
from mbpp_tasks import load_mbpp_tasks


def greedy_decode(model, tokenizer: Tokenizer, prompt: str, max_len: int = 256) -> str:
    model.eval()
    device = next(model.parameters()).device
    ids = tokenizer.encode(prompt, add_eos=False, max_len=max_len)
    prefix_len = len(ids)
    with torch.no_grad():
        for _ in range(max_len - len(ids)):
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            out = model(input_ids)
            logits = out[0] if isinstance(out, tuple) else out
            next_id = int(torch.argmax(logits[0, -1]).item())
            if tokenizer.vocab[next_id] == tokenizer.eos_token:
                break
            ids.append(next_id)
    gen_ids = [i for i in ids[prefix_len:] if tokenizer.vocab[i] != tokenizer.eos_token]
    return "".join(tokenizer.vocab[i] for i in gen_ids)


def load_model_checkpoint(ckpt_path: Path) -> tuple[object, Tokenizer]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["state_dict"]
    vocab = ckpt["vocab"]
    eos_token = ckpt["eos_token"]
    tokenizer = Tokenizer(vocab=vocab, eos_token=eos_token)

    if any(k.startswith("value_head") for k in state.keys()):
        embed_dim = state["embed.weight"].shape[1]
        hidden_dim = state["gru.weight_hh_l0"].shape[0] // 3
        model = ActorCritic(len(vocab), embed_dim=embed_dim, hidden_dim=hidden_dim)
        rename_map = {"head.weight": "policy_head.weight", "head.bias": "policy_head.bias"}
        for src, dst in rename_map.items():
            if src in state:
                state[dst] = state[src]
    else:
        embed_dim = state["embed.weight"].shape[1]
        hidden_dim = state["gru.weight_hh_l0"].shape[0] // 3
        model = CharPolicy(len(vocab), embed_dim=embed_dim, hidden_dim=hidden_dim)

    model.load_state_dict(state, strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=Path("checkpoints/pretrain.pt"))
    parser.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--max-gen-len", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-challenge-tests", action="store_true")
    args = parser.parse_args()

    tasks = load_mbpp_tasks(
        split=args.split,
        max_samples=args.max_samples,
        seed=args.seed,
        use_challenge_tests=args.use_challenge_tests,
    )

    model, tokenizer = load_model_checkpoint(args.ckpt)
    env = CodeGenEnv(max_steps=args.max_steps, tasks=TASKS, vocab_tasks=TASKS)

    successes = 0
    total_reward = 0.0
    for task in tasks:
        code = greedy_decode(model, tokenizer, prompt=task.prompt + "\n", max_len=args.max_gen_len)
        reward, passed = env._evaluate_code(code, task)
        successes += int(passed)
        total_reward += reward
        print(
            f"[mbpp] id={task.name} passed={passed} reward={reward:.2f} "
            f"code_len={len(code)}"
        )

    total = len(tasks)
    avg_reward = total_reward / total if total else 0.0
    print(f"[mbpp] success={successes}/{total} = {successes/total:.2f}")
    print(f"[mbpp] avg_reward={avg_reward:.2f}")


if __name__ == "__main__":
    main()
