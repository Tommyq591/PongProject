"""Verbatim training definitions extracted from
CS4287-Assign2_Atari_ID1_ID2_ID3.ipynb
cells [5, 6, 10, 13, 16, 18, 21, 27, 28, 29]. Do not edit; overrides are applied in run_one.py."""

# ===== notebook cell 5 =====
from __future__ import annotations
import argparse
import copy
import csv
import dataclasses
import gzip
import hashlib
import importlib.metadata
import json
import os
import pickle
import platform
import random
import shutil
import sys
import time
import traceback
from collections import deque
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
import ale_py
import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

gym.register_envs(ale_py)
def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    """Replace a small JSON file atomically, so a crash cannot leave half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def append_csv(path, row):
    """Keep every measurement on disk rather than only in notebook memory."""
    path = Path(path)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def append_event(folder, event, **details):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"time": utc_now(), "event": event, **details}) + "\n")

# ===== notebook cell 6 =====
@dataclass
class Config:
    # A step is ONE agent decision, usually FOUR emulator frames.
    method: str = "dqn"
    seed: int = 17
    total_steps: int = 100_000
    env_id: str = "ALE/Pong-v5"
    frame_skip: int = 4
    screen_size: int = 84
    stack_size: int = 4
    sticky_probability: float = 0.25
    noop_max: int = 30
    max_episode_steps: int = 27_000
    replay_capacity: int = 20_000
    learning_starts: int = 2_000
    batch_size: int = 32
    train_frequency: int = 4
    gamma: float = 0.99
    learning_rate: float = 0.0001
    target_update_interval: int = 2_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 80_000
    clip_rewards: bool = True
    max_grad_norm: float = 10.0
    eval_interval: int = 20_000
    eval_episodes: int = 5
    final_eval_episodes: int = 10
    eval_epsilon: float = 0.01
    checkpoint_interval: int = 10_000
    log_interval: int = 1_000
    torch_threads: int = 2
    device: str = "cpu"

    def validate(self):
        assert self.method in {"dqn", "double_dqn"}
        assert self.stack_size == 4 and self.screen_size == 84
        assert self.replay_capacity > self.stack_size + self.batch_size
        assert self.total_steps >= 1 and self.train_frequency >= 1
        assert self.eval_interval >= 1 and self.checkpoint_interval >= 1
        assert 0 <= self.sticky_probability <= 1


def seed_everything(seed, threads=2):
    """Matching seeds control initial weights; separate RNGs control sampling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    # We use CPU. Request deterministic operations for repeatable local recovery.
    torch.use_deterministic_algorithms(True)


def epsilon_at(step, config):
    # Express decay in game decisions, so pausing does not restart exploration.
    fraction = min(step / config.epsilon_decay_steps, 1.0)
    return config.epsilon_start + fraction * (config.epsilon_end - config.epsilon_start)

# ===== notebook cell 10 =====
class StickyFrameActions(gym.Wrapper):
    """Repeat the previous action with probability p on EACH emulator frame.

    ALE 0.12.1's snapshot omits its previous sticky action. Keeping that small
    piece of state here makes mid-episode checkpoint recovery exact and visible.
    ALE's own sticky probability is therefore zero; this wrapper applies it once.
    """

    def __init__(self, env, probability):
        super().__init__(env)
        self.probability = probability
        self.rng = np.random.default_rng(0)
        self.previous_action = 0

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed + 3000)
        self.previous_action = 0
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        # This wrapper is INSIDE action repetition: sample once per raw frame.
        if self.rng.random() < self.probability:
            action = self.previous_action
        self.previous_action = int(action)
        return self.env.step(self.previous_action)


class PongEnv:
    """Official Atari preprocessing plus a transparent four-frame stack.

    The base ALE has frameskip=1. ONLY AtariPreprocessing repeats the action.
    No window is opened. rgb_array is enabled only when recording a video.
    """

    def __init__(self, config, video=False):
        self.config = config
        self.raw = gym.make(
            config.env_id,
            frameskip=1,
            repeat_action_probability=0.0,  # StickyFrameActions applies p once.
            full_action_space=False,
            render_mode="rgb_array" if video else None,
            max_episode_steps=-1,  # We apply our decision-based limit below.
        )
        self.sticky = StickyFrameActions(self.raw, config.sticky_probability)
        self.pre = gym.wrappers.AtariPreprocessing(
            self.sticky,
            noop_max=config.noop_max,
            frame_skip=config.frame_skip,
            screen_size=config.screen_size,
            terminal_on_life_loss=False,  # A point/life loss is not a whole match.
            grayscale_obs=True,
            scale_obs=False,  # Save uint8; convert to [0,1] inside the CNN.
        )
        assert self.raw.unwrapped._frameskip == 1
        self.frames = deque(maxlen=config.stack_size)
        # The deque automatically drops the oldest image when a new one arrives.
        self.episode_steps = 0
        self.action_count = self.raw.action_space.n

    def reset(self, seed=None):
        image, info = self.pre.reset(seed=seed)
        # Initially no history exists: repeat the first frame four times.
        self.frames.clear()
        self.frames.extend(image.copy() for _ in range(self.config.stack_size))
        self.episode_steps = 0
        return np.stack(self.frames), info

    def step(self, action):
        # Count actual emulator frames, including shortened terminal repeats.
        before = self.raw.unwrapped.ale.getFrameNumber()
        image, reward, terminated, truncated, info = self.pre.step(int(action))
        if truncated and not terminated:
            # AtariPreprocessing can break before refreshing its pooling buffers
            # when ALE itself truncates mid-repeat. Bootstrap from the actual
            # final screen in that rare case, rather than a stale earlier image.
            import cv2
            final_screen = self.raw.unwrapped.ale.getScreenGrayscale()
            image = cv2.resize(final_screen, (self.config.screen_size, self.config.screen_size),
                               interpolation=cv2.INTER_AREA)
        self.episode_steps += 1
        # Our limit counts decisions; the engine's own limit counts raw frames.
        truncated = bool(truncated or self.episode_steps >= self.config.max_episode_steps)
        self.frames.append(image.copy())
        info["actual_game_frames"] = self.raw.unwrapped.ale.getFrameNumber() - before
        # Return the final observation BEFORE resetting, including on truncation.
        return np.stack(self.frames), float(reward), bool(terminated), truncated, info

    def state_dict(self):
        """Save the emulator AND wrapper state for exact mid-episode recovery."""
        base = self.raw.unwrapped
        return {
            "ale": pickle.dumps(base.ale.cloneSystemState()),
            "env_rng": copy.deepcopy(base.np_random.bit_generator.state),
            "frames": list(self.frames),
            "episode_steps": self.episode_steps,
            "obs_buffer": [x.copy() for x in self.pre.obs_buffer],
            "lives": self.pre.lives,
            "game_over": self.pre.game_over,
            "sticky_rng": copy.deepcopy(self.sticky.rng.bit_generator.state),
            "previous_action": self.sticky.previous_action,
        }

    def load_state_dict(self, state):
        # Reset first so Gymnasium's OrderEnforcing wrapper knows reset occurred.
        self.reset(seed=0)
        base = self.raw.unwrapped
        base.ale.restoreSystemState(pickle.loads(state["ale"]))
        base.np_random.bit_generator.state = state["env_rng"]
        self.frames = deque((x.copy() for x in state["frames"]), maxlen=self.config.stack_size)
        self.episode_steps = state["episode_steps"]
        self.pre.obs_buffer = [x.copy() for x in state["obs_buffer"]]
        self.pre.lives = state["lives"]
        self.pre.game_over = state["game_over"]
        self.sticky.rng.bit_generator.state = state["sticky_rng"]
        self.sticky.previous_action = state["previous_action"]

    def close(self):
        self.pre.close()

# ===== notebook cell 13 =====
class QNetwork(nn.Module):
    """Small CPU-friendly CNN. Both methods use this exact architecture."""

    def __init__(self, action_count=6):
        super().__init__()
        # Four chronological images enter as channels, not as four batch items.
        self.features = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=8, stride=4), nn.ReLU(),  # 16 x 20 x 20
            nn.Conv2d(16, 32, kernel_size=4, stride=2), nn.ReLU(),  # 32 x 9 x 9
            nn.Conv2d(32, 32, kernel_size=3, stride=1), nn.ReLU(),  # 32 x 7 x 7
            nn.Flatten(),
        )
        self.values = nn.Sequential(
            # Flattening preserves every feature; the dense layer combines them.
            nn.Linear(32 * 7 * 7, 256), nn.ReLU(),
            nn.Linear(256, action_count),  # One value for each action; no softmax.
        )

    def forward(self, images):
        images = images.float() / 255.0
        return self.values(self.features(images))

# ===== notebook cell 16 =====
class ReplayBuffer:
    """Uniform replay storing two uint8 frames per transition instead of 8.

    Consecutive states share most images. We reconstruct a current stack from
    the previous three transitions; next_stack = current_stack[1:] + next_frame.
    episode_steps prevents histories crossing resets. Absolute IDs prevent
    accidentally reading overwritten entries after the circular buffer wraps.
    """

    def __init__(self, capacity, seed, screen_size=84):
        self.capacity = capacity
        # Byte images take one quarter the space of float32 images.
        self.frames = np.zeros((capacity, screen_size, screen_size), dtype=np.uint8)
        self.next_frames = np.zeros_like(self.frames)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        # Store both ending flags: they have different meanings for bootstrapping.
        self.terminated = np.zeros(capacity, dtype=np.bool_)
        self.truncated = np.zeros(capacity, dtype=np.bool_)
        self.episode_steps = np.zeros(capacity, dtype=np.int64)
        self.ids = np.full(capacity, -1, dtype=np.int64)
        self.count = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return min(self.count, self.capacity)

    def add(self, state, action, reward, next_state, terminated, truncated, episode_step):
        # Modulo implements a ring: new transitions replace the oldest slots.
        i = self.count % self.capacity
        self.frames[i] = state[-1]
        self.next_frames[i] = next_state[-1]
        self.actions[i] = action
        self.rewards[i] = reward
        self.terminated[i] = terminated
        self.truncated[i] = truncated
        self.episode_steps[i] = episode_step
        self.ids[i] = self.count
        # IDs keep increasing even after the physical storage index wraps.
        self.count += 1

    def sample(self, batch_size):
        # Once full, omit the oldest three entries whose history was overwritten.
        oldest = max(0, self.count - self.capacity)
        start = oldest + (3 if oldest > 0 else 0)
        absolute_ids = self.rng.integers(start, self.count, size=batch_size)
        # Sampling with replacement allows a transition to occur twice in a batch.
        slots = absolute_ids % self.capacity
        stacks = []
        for offset in (3, 2, 1, 0):
            # At the beginning of an episode, pad with that episode's first frame.
            previous_ids = absolute_ids - np.minimum(offset, self.episode_steps[slots])
            history_slots = previous_ids % self.capacity
            assert np.array_equal(self.ids[history_slots], previous_ids)
            stacks.append(self.frames[history_slots])
        states = np.stack(stacks, axis=1)
        # Shift history by one image, then append the actual next image.
        next_states = np.concatenate((states[:, 1:], self.next_frames[slots, None]), axis=1)
        return {
            "states": states, "next_states": next_states, "slots": slots,
            "actions": self.actions[slots], "rewards": self.rewards[slots],
            "terminated": self.terminated[slots], "truncated": self.truncated[slots],
        }

    def state_dict(self):
        # Copy only populated slots until the circular buffer is full.
        n = len(self)
        names = ("frames", "next_frames", "actions", "rewards", "terminated",
                 "truncated", "episode_steps", "ids")
        return {"count": self.count, "rng": self.rng.bit_generator.state,
                "arrays": {name: getattr(self, name)[:n] for name in names}}

    def load_state_dict(self, state):
        self.count = state["count"]
        # Restoring the generator state restores the future sequence of batches.
        self.rng.bit_generator.state = state["rng"]
        for name, array in state["arrays"].items():
            getattr(self, name)[:len(array)] = array


@torch.inference_mode()
def choose_action(network, observation, epsilon, rng, action_count, device="cpu"):
    # Use our saved generator, not action_space.sample's independent hidden RNG.
    if rng.random() < epsilon:
        return int(rng.integers(action_count))
    image_tensor = torch.as_tensor(observation[None], device=device)
    # Add a batch dimension: [4,84,84] becomes [1,4,84,84].
    return int(network(image_tensor).argmax(dim=1).item())

# ===== notebook cell 18 =====
def bootstrap_values(next_online, next_target, method):
    """The ONLY algorithmic difference between DQN and Double DQN."""
    if method == "double_dqn":
        # Online selects the action; target evaluates that selected action.
        selected_actions = next_online.argmax(dim=1, keepdim=True)
        return next_target.gather(1, selected_actions).squeeze(1)
    # DQN uses target values both to select and to evaluate the action.
    return next_target.max(dim=1).values

# ===== notebook cell 21 =====
def learning_update(online, target, optimizer, batch, config):
    """One complete Q-learning weight update; explained line by line in notebook."""
    device = torch.device(config.device)
    states = torch.as_tensor(batch["states"], device=device)
    next_states = torch.as_tensor(batch["next_states"], device=device)
    actions = torch.as_tensor(batch["actions"], device=device).long()
    # Float rewards and masks can participate in the target arithmetic.
    rewards = torch.as_tensor(batch["rewards"], device=device).float()
    terminated = torch.as_tensor(batch["terminated"], device=device).float()

    # 1. Predict Q(s,a) for the actions that were actually taken in the batch.
    predicted = online(states).gather(1, actions[:, None]).squeeze(1)
    # 2. Targets are fixed numbers during this update: do not differentiate them.
    with torch.no_grad():
        next_target = target(next_states)
        next_online = online(next_states) if config.method == "double_dqn" else None
        next_values = bootstrap_values(next_online, next_target, config.method)
        # Reset after termination OR truncation, but bootstrap after truncation.
        # A real terminal game has no future return. A time cut still has one.
        expected = rewards + config.gamma * (1.0 - terminated) * next_values

    # 3. Huber loss: quadratic near zero error, linear for unusually large errors.
    loss = F.smooth_l1_loss(predicted, expected)
    # The default reduction averages the 32 sample losses into one scalar.
    # 4. Remove gradients left over from the previous mini-batch.
    optimizer.zero_grad(set_to_none=True)
    # 5. Chain rule: calculate how each weight contributed to the prediction error.
    loss.backward()
    # 6. Limit unusually large gradients, identically for both methods.
    gradient_norm = nn.utils.clip_grad_norm_(online.parameters(), config.max_grad_norm)
    # 7. Adam uses those gradients to change ONLINE weights; target stays fixed.
    optimizer.step()
    return {"loss": float(loss.detach()), "mean_q": float(predicted.detach().mean()),
            "mean_target": float(expected.mean()),
            "mean_abs_td_error": float((predicted.detach() - expected).abs().mean()),
            "gradient_norm": float(gradient_norm)}

# ===== notebook cell 27 =====
@torch.inference_mode()
def evaluate(network, config, step, episodes, train_elapsed_s=0.0):
    """Separate environment/RNG; raw rewards, no replay writes or weight updates."""
    env = PongEnv(config)
    # Evaluation never advances the training environment or its random streams.
    was_training = network.training
    network.eval()
    rows = []
    try:
        for index in range(episodes):
            # Reuse the same held-out starts at every checkpoint and across methods.
            eval_seed = 100_000 + config.seed * 100 + index
            rng = np.random.default_rng(eval_seed + 1_000_000)
            observation, _ = env.reset(seed=eval_seed)
            score, length, frames = 0.0, 0, 0
            started = time.perf_counter()
            terminated = truncated = False
            while not (terminated or truncated):
                action = choose_action(network, observation, config.eval_epsilon,
                                       rng, env.action_count, config.device)
                observation, reward, terminated, truncated, info = env.step(action)
                score += reward  # Original game score; NO reward clipping here.
                length += 1
                frames += info["actual_game_frames"]
            rows.append({"step": step, "eval_index": index, "eval_seed": eval_seed,
                         "score": score, "length": length, "game_frames": frames,
                         "terminated": terminated, "truncated": truncated,
                         "eval_elapsed_s": time.perf_counter() - started,
                         "train_elapsed_s": train_elapsed_s})
    finally:
        env.close()
        # Return the model to its previous mode for any subsequent training.
        network.train(was_training)
    return rows

# ===== notebook cell 28 =====
class Trainer:
    """Shared training loop. Configuration differs only in method and run seed."""

    def __init__(self, config, folder):
        config.validate()
        self.config, self.folder = config, Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        seed_everything(config.seed, config.torch_threads)
        # Seed once at run start. Re-seeding each episode would repeat starts.
        self.env = PongEnv(config)
        self.observation, _ = self.env.reset(seed=config.seed)
        self.online = QNetwork(self.env.action_count).to(config.device)
        self.target = copy.deepcopy(self.online).eval()
        # A copy starts with identical values, but has independent tensor storage.
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        # Only online parameters are registered with Adam; target cannot be fitted.
        self.replay = ReplayBuffer(config.replay_capacity, config.seed + 1000)
        self.action_rng = np.random.default_rng(config.seed + 2000)
        self.step = self.updates = self.episode = self.episode_length = 0
        self.episode_return = self.elapsed_s = self.eval_elapsed_s = 0.0
        self.game_frames = self.reset_frames = 0
        self.initial_evaluated = False
        self.last_eval_step = -1
        self.last_eval_episodes = 0
        write_json(self.folder / "config.json", asdict(config))

    def advance(self, decisions):
        """Collect real transitions and update every train_frequency decisions."""
        started = time.perf_counter()
        latest = None
        for _ in range(decisions):
            action = choose_action(self.online, self.observation,
                                   epsilon_at(self.step, self.config), self.action_rng,
                                   self.env.action_count, self.config.device)
            next_observation, reward, terminated, truncated, info = self.env.step(action)
            stored_reward = float(np.sign(reward)) if self.config.clip_rewards else reward
            # Keep raw reward below for episode reporting; clip only replay reward.
            self.replay.add(self.observation, action, stored_reward, next_observation,
                            terminated, truncated, self.episode_length)
            self.observation = next_observation
            self.step += 1
            self.game_frames += info["actual_game_frames"]
            self.episode_length += 1
            self.episode_return += reward
            current_elapsed = self.elapsed_s + time.perf_counter() - started
            if terminated or truncated:
                # Store the final transition above BEFORE resetting the environment.
                append_csv(self.folder / "episodes.csv", {
                    "step": self.step, "episode": self.episode,
                    "score": self.episode_return, "length": self.episode_length,
                    "terminated": terminated, "truncated": truncated,
                    "train_elapsed_s": current_elapsed, "game_frames": self.game_frames})
                self.episode += 1
                self.episode_length, self.episode_return = 0, 0.0
                self.observation, _ = self.env.reset()
                self.reset_frames += self.env.raw.unwrapped.ale.getEpisodeFrameNumber()
            if self.step >= self.config.learning_starts and self.step % self.config.train_frequency == 0:
                # Reuse experience through replay instead of learning only this step.
                batch = self.replay.sample(self.config.batch_size)
                latest = learning_update(self.online, self.target, self.optimizer, batch, self.config)
                self.updates += 1
                if self.updates % 100 == 0:
                    append_csv(self.folder / "updates.csv", {"step": self.step,
                               "update": self.updates, **latest})
            if self.step % self.config.target_update_interval == 0:
                # A hard copy, NOT a gradient update to the target network.
                self.target.load_state_dict(self.online.state_dict())
        self.elapsed_s += time.perf_counter() - started
        # Evaluation and checkpoint I/O are measured separately from training time.
        return latest

    def state_dict(self):
        # Model-only saving would lose replay, momentum, exploration and game state.
        return {"config": asdict(self.config), "step": self.step, "updates": self.updates,
                "episode": self.episode, "episode_length": self.episode_length,
                "episode_return": self.episode_return, "elapsed_s": self.elapsed_s,
                "eval_elapsed_s": self.eval_elapsed_s, "game_frames": self.game_frames,
                "reset_frames": self.reset_frames,
                "initial_evaluated": self.initial_evaluated, "last_eval_step": self.last_eval_step,
                "last_eval_episodes": self.last_eval_episodes,
                "online": self.online.state_dict(), "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(), "replay": self.replay.state_dict(),
                "action_rng": self.action_rng.bit_generator.state,
                "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(), "env": self.env.state_dict(),
                "observation": self.observation}

    def load_state_dict(self, state):
        # Budget may be extended explicitly; all other shared settings must match.
        old, new = dict(state["config"]), asdict(self.config)
        old.pop("total_steps"); new.pop("total_steps")
        if old != new:
            raise ValueError("Checkpoint settings differ. Use the saved configuration.")
        for name in ("step", "updates", "episode", "episode_length", "episode_return",
                     "elapsed_s", "eval_elapsed_s", "game_frames", "reset_frames",
                     "initial_evaluated", "last_eval_step", "last_eval_episodes"):
            setattr(self, name, state[name])
        self.online.load_state_dict(state["online"])
        self.target.load_state_dict(state["target"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.replay.load_state_dict(state["replay"])
        self.action_rng.bit_generator.state = state["action_rng"]
        self.env.load_state_dict(state["env"])
        self.observation = state["observation"].copy()
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])

    def save(self):
        """Full replay + optimiser + RNGs + emulator, compressed and atomic."""
        started = time.perf_counter()
        temporary = self.folder / "checkpoint.tmp.gz"
        with gzip.open(temporary, "wb", compresslevel=1) as stream:
            # Compression is effective because Pong images contain large flat areas.
            torch.save(self.state_dict(), stream)
        os.replace(temporary, self.folder / "checkpoint.pt.gz")
        torch.save({"config": asdict(self.config), "step": self.step,
                    "online": self.online.state_dict()}, self.folder / "policy.pt")
        append_event(self.folder, "checkpoint_saved", step=self.step,
                     seconds=time.perf_counter() - started)

    def load(self):
        # weights_only=False is needed for our own trusted replay/RNG checkpoint.
        # Never load an unknown downloaded pickle/checkpoint with this setting.
        with gzip.open(self.folder / "checkpoint.pt.gz", "rb") as stream:
            state = torch.load(stream, map_location=self.config.device, weights_only=False)
        self.load_state_dict(state)
        # A process may die after logging but before saving. Preserve those rows
        # in an archive, then roll active logs back to the saved step.
        for name in ("episodes.csv", "updates.csv", "progress.csv", "evaluations.csv"):
            path = self.folder / name
            if path.exists():
                with path.open(newline="", encoding="utf-8") as stream:
                    rows = list(csv.DictReader(stream))
                kept = [r for r in rows if int(r["step"]) <= self.step]
                if len(kept) != len(rows):
                    shutil.copy2(path, path.with_name(path.stem + f"_before_resume_{time.time_ns()}.csv"))
                    path.unlink()
                    for row in kept:
                        append_csv(path, row)
        append_event(self.folder, "checkpoint_resumed", step=self.step)

    def evaluate_now(self, episodes):
        started = time.perf_counter()
        # Evaluation is outside advance(): there are no replay writes or updates.
        rows = evaluate(self.online, self.config, self.step, episodes, self.elapsed_s)
        # If final evaluation needs more episodes than periodic evaluation, keep
        # its first episodes and append only the extra held-out episodes.
        already = self.last_eval_episodes if self.last_eval_step == self.step else 0
        for row in rows[already:]:
            append_csv(self.folder / "evaluations.csv", row)
        self.eval_elapsed_s += time.perf_counter() - started
        self.last_eval_step, self.last_eval_episodes = self.step, episodes
        self.initial_evaluated = True
        print(f"{self.config.method} seed={self.config.seed} step={self.step:,} "
              f"evaluation={np.mean([r['score'] for r in rows]):.2f} ({episodes} games)", flush=True)

    def close(self):
        self.env.close()

# ===== notebook cell 29 =====
def run_training(config, folder, stop_at=None, trainer_factory=Trainer):
    """Resume automatically. stop_at pauses work without changing the planned budget."""
    folder = Path(folder)
    saved_config = folder / "config.json"
    # A mismatched configuration requires a new run folder, protecting comparisons.
    if saved_config.exists():
        previous = json.loads(saved_config.read_text(encoding="utf-8"))
        if previous != asdict(config):
            raise ValueError("Run folder already has another config; do not mix experiments.")
    trainer = trainer_factory(config, folder)
    end = min(config.total_steps, stop_at if stop_at is not None else config.total_steps)
    status = "running"
    append_event(folder, "run_started", planned_steps=config.total_steps, session_stop=end)
    try:
        if (folder / "checkpoint.pt.gz").exists():
            # Resume absolute step counters, so schedules continue where they stopped.
            trainer.load()
        if not trainer.initial_evaluated:
            trainer.evaluate_now(config.eval_episodes)
            trainer.save()
        write_json(folder / "status.json", {"status": status, "step": trainer.step,
                                             "planned_steps": config.total_steps, "time": utc_now()})
        while trainer.step < end:
            # Stop exactly at every evaluation/checkpoint/log boundary.
            next_events = [end]
            for interval in (config.log_interval, config.eval_interval, config.checkpoint_interval):
                next_events.append((trainer.step // interval + 1) * interval)
            next_step = min(next_events)
            before, started = trainer.step, time.perf_counter()
            latest = trainer.advance(next_step - trainer.step)
            segment_seconds = time.perf_counter() - started
            append_csv(folder / "progress.csv", {
                "step": trainer.step, "updates": trainer.updates,
                "epsilon": epsilon_at(trainer.step, config),
                "train_elapsed_s": trainer.elapsed_s, "eval_elapsed_s": trainer.eval_elapsed_s,
                "decisions_per_s": (trainer.step - before) / segment_seconds,
                "game_frames": trainer.game_frames, "reset_frames": trainer.reset_frames,
                "replay_size": len(trainer.replay)})
            if trainer.step % config.eval_interval == 0 or trainer.step == config.total_steps:
                # Evaluate every planned checkpoint, including a nonmultiple budget.
                count = config.final_eval_episodes if trainer.step == config.total_steps else config.eval_episodes
                trainer.evaluate_now(count)
            if trainer.step % config.checkpoint_interval == 0 or trainer.step == end:
                trainer.save()
            write_json(folder / "status.json", {"status": "running", "step": trainer.step,
                                                 "planned_steps": config.total_steps, "time": utc_now()})
            print(f"{config.method} seed={config.seed} {trainer.step:,}/{config.total_steps:,} "
                  f"{(trainer.step-before)/segment_seconds:.1f} decisions/s "
                  f"updates={trainer.updates}", flush=True)
        if trainer.step == config.total_steps and (trainer.last_eval_step != trainer.step or
                                                   trainer.last_eval_episodes < config.final_eval_episodes):
            trainer.evaluate_now(config.final_eval_episodes)
            trainer.save()
        status = "complete" if trainer.step >= config.total_steps else "paused"
        return {"status": status, "step": trainer.step, "updates": trainer.updates,
                "train_elapsed_s": trainer.elapsed_s, "eval_elapsed_s": trainer.eval_elapsed_s}
    except BaseException as error:
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        append_event(folder, status, step=trainer.step, error=repr(error), traceback=traceback.format_exc())
        # Preserve current in-memory training when a normal Python error/interrupt occurs.
        try:
            trainer.save()
        except BaseException:
            append_event(folder, "emergency_save_failed", traceback=traceback.format_exc())
        raise
    finally:
        write_json(folder / "status.json", {"status": status, "step": trainer.step,
                                             "planned_steps": config.total_steps, "time": utc_now()})
        trainer.close()
