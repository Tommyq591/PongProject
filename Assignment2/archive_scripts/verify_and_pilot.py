"""Meaningful correctness checks and separate exploratory throughput pilots."""
import copy
import json
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from pong_dqn import (Config, PongEnv, QNetwork, ReplayBuffer, Trainer,
                      bootstrap_values, learning_update, seed_everything,
                      system_information, write_json)

ROOT = Path(__file__).resolve().parent


def check_targets():
    # Online prefers action 1; target prefers action 0. The methods must differ.
    online = torch.tensor([[1., 4., 2.], [1., 4., 2.]])
    target = torch.tensor([[3., 2., 1.], [3., 2., 1.]])
    assert torch.equal(bootstrap_values(online, target, "dqn"), torch.tensor([3., 3.]))
    assert torch.equal(bootstrap_values(online, target, "double_dqn"), torch.tensor([2., 2.]))
    # A time cutoff must retain the future value; a true terminal must remove it.
    rewards = torch.tensor([1., 1.])
    terminated = torch.tensor([1., 0.])
    computed = rewards + .9 * (1 - terminated) * bootstrap_values(online, target, "double_dqn")
    assert torch.allclose(computed, torch.tensor([1., 2.8]))
    return {"dqn_next_value": 3., "double_dqn_next_value": 2.,
            "terminal_target": 1., "truncation_target": 2.8}


def check_replay():
    # Deliberately tiny ring: exercise overwrite boundaries and frequent resets.
    replay = ReplayBuffer(13, 123)
    expected = {}
    state = None
    for index in range(60):
        t = index % 7
        frame = np.full((84, 84), index, np.uint8)
        state = np.repeat(frame[None], 4, axis=0) if t == 0 else next_state
        next_frame = np.full((84, 84), index + 1, np.uint8)
        next_state = np.concatenate([state[1:], next_frame[None]])
        # Use action to identify the sampled transition in this synthetic check.
        replay.add(state, index, 1., next_state, t == 6, False, t)
        expected[index] = (state.copy(), next_state.copy())
        batch = replay.sample(20)
        for j, action in enumerate(batch["actions"]):
            assert np.array_equal(batch["states"][j], expected[action][0])
            assert np.array_equal(batch["next_states"][j], expected[action][1])
    return "60 insertions; exact reconstructed stacks after resets and ring overwrite"


def check_weight_update():
    config = replace(Config(), replay_capacity=128)
    seed_everything(9)
    online = QNetwork()
    target = copy.deepcopy(online).eval()
    target.requires_grad_(False)
    optimizer = torch.optim.Adam(online.parameters(), lr=config.learning_rate)
    rng = np.random.default_rng(9)
    batch = {"states": rng.integers(0, 256, (4, 4, 84, 84), dtype=np.uint8),
             "next_states": rng.integers(0, 256, (4, 4, 84, 84), dtype=np.uint8),
             "actions": np.array([0, 1, 2, 3]), "rewards": np.array([1., -1., 0., 1.]),
             "terminated": np.array([True, False, False, True]),
             "truncated": np.array([False, True, False, False])}
    before = {k: v.clone() for k, v in online.state_dict().items()}
    target_before = {k: v.clone() for k, v in target.state_dict().items()}
    metrics = learning_update(online, target, optimizer, batch, config)
    assert np.isfinite(metrics["loss"])
    assert any(not torch.equal(v, before[k]) for k, v in online.state_dict().items())
    assert all(torch.equal(v, target_before[k]) for k, v in target.state_dict().items())
    assert all(p.grad is None for p in target.parameters())
    return metrics


def check_environment():
    config = Config()
    env = PongEnv(config)
    try:
        state, _ = env.reset(seed=7)
        assert state.shape == (4, 84, 84) and state.dtype == np.uint8
        for _ in range(30):
            env.step(2)
        saved = env.state_dict()
        continuation = [env.step(a) for a in [2, 3, 1, 0, 2, 3]]
        env.load_state_dict(saved)
        repeated = [env.step(a) for a in [2, 3, 1, 0, 2, 3]]
        for left, right in zip(continuation, repeated):
            assert np.array_equal(left[0], right[0])
            assert left[1:4] == right[1:4]
        # Confirm our external limit returns truncated rather than terminated.
        limited = PongEnv(replace(config, max_episode_steps=2))
        limited.reset(seed=7)
        limited.step(0)
        _, _, terminated, truncated, _ = limited.step(0)
        limited.close()
        assert truncated and not terminated
        return {"shape": list(state.shape), "dtype": str(state.dtype),
                "actions": env.raw.unwrapped.get_action_meanings(),
                "base_frameskip": env.raw.unwrapped._frameskip,
                "wrapper_frameskip": env.pre.frame_skip,
                "exact_environment_recovery": True, "external_truncation": True}
    finally:
        env.close()


def check_full_recovery():
    config = replace(Config(), total_steps=320, replay_capacity=256,
                     learning_starts=64, epsilon_decay_steps=300)
    folder = ROOT / "runs" / "pilot" / "recovery"
    trainer = Trainer(config, folder)
    trainer.advance(160)
    trainer.save()
    trainer.advance(80)
    expected = {k: v.clone() for k, v in trainer.online.state_dict().items()}
    observation = trainer.observation.copy()
    rewards = trainer.episode_return
    updates = trainer.updates
    trainer.close()
    restored = Trainer(config, folder)
    try:
        restored.load()
        assert restored.step == 160
        restored.advance(80)
        assert restored.updates == updates
        assert restored.episode_return == rewards
        assert np.array_equal(restored.observation, observation)
        assert all(torch.equal(v, expected[k]) for k, v in restored.online.state_dict().items())
        restored.save()
        return {"saved_step": 160, "compared_step": 240,
                "identical_pixels_rewards_and_weights": True, "updates": updates}
    finally:
        restored.close()


def check_mid_repeat_truncation():
    """The future-value input must be the real final screen at a raw-frame cutoff."""
    import cv2
    env = PongEnv(Config())
    try:
        env.reset(seed=77)
        for i in range(120):
            env.step(i % 6)
        original = env.sticky.step

        def forced_cutoff(action):
            observation, reward, terminated, truncated, info = original(action)
            return observation, reward, terminated, True, info

        env.sticky.step = forced_cutoff
        state, _, terminated, truncated, _ = env.step(2)
        expected = cv2.resize(env.raw.unwrapped.ale.getScreenGrayscale(), (84, 84),
                              interpolation=cv2.INTER_AREA)
        assert truncated and not terminated
        assert np.array_equal(state[-1], expected)
        return "Raw cutoff during action repeat bootstraps from exact final screen"
    finally:
        env.close()


def benchmark():
    rows = []
    # A hardware choice, not performance tuning: no game scores select settings.
    for threads in (1, 2, 4):
        config = replace(Config(), total_steps=1400, learning_starts=128,
                         replay_capacity=2000, torch_threads=threads)
        trainer = Trainer(config, ROOT / "runs" / "pilot" / f"throughput_threads_{threads}")
        try:
            trainer.advance(400)  # Warm up CNN and optimiser before measuring.
            before_updates = trainer.updates
            started = time.perf_counter()
            trainer.advance(1000)
            seconds = time.perf_counter() - started
            trainer.save()
            row = {"threads": threads, "measured_decisions": 1000,
                   "measured_updates": trainer.updates - before_updates,
                   "elapsed_s": seconds, "decisions_per_s": 1000 / seconds,
                   "note": "Pilot only; includes actual CNN updates and environment stepping."}
            rows.append(row)
            write_json(ROOT / "results" / "throughput.json", rows)
            print(json.dumps(row), flush=True)
        finally:
            trainer.close()
    return rows


def main():
    (ROOT / "results").mkdir(exist_ok=True)
    write_json(ROOT / "results" / "system.json", system_information())
    report = {}
    checks = [("target_equations", check_targets), ("replay", check_replay),
              ("learning_update", check_weight_update), ("environment", check_environment),
              ("full_recovery", check_full_recovery),
              ("mid_repeat_truncation", check_mid_repeat_truncation)]
    for name, check in checks:
        try:
            report[name] = check()
        except BaseException:
            with (ROOT / "results" / "verification_failures.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"check": name, "traceback": traceback.format_exc()}) + "\n")
            raise
        write_json(ROOT / "results" / "verification.json", report)
        print("PASS", name, report[name], flush=True)
    benchmark()


if __name__ == "__main__":
    main()
