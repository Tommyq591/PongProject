"""An image and motion guided follow-up to the original Atari comparison."""

from original_core import *
from visual_coach import VisualCoach, demonstration_action


@dataclass
class GuidedConfig(Config):
    total_steps: int = 20_000
    eval_interval: int = 10_000
    checkpoint_interval: int = 5_000
    torch_threads: int = 4
    aim_offset: float = 5.0
    coach_margin: float = 0.10
    coach_weight: float = 0.25


class GuidedPongEnv(PongEnv):
    """Reuse the original pixel stack and attach eight features from RGB frames."""

    def __init__(self, config):
        self.vision = VisualCoach(config.aim_offset)
        # rgb_array gives the screen as data without opening a window.
        super().__init__(config, video=True)

    def reset(self, seed=None):
        self.vision.reset()
        images, info = super().reset(seed)
        return {"images": images, "cues": self.vision.observe(self.raw.render())}, info

    def step(self, action):
        images, reward, terminated, truncated, info = super().step(action)
        if reward != 0:
            # A new rally invalidates the previous ball position.
            self.vision.reset()
        observation = {"images": images, "cues": self.vision.observe(self.raw.render())}
        return observation, reward, terminated, truncated, info

    def state_dict(self):
        state = super().state_dict()
        state["previous_ball"] = self.vision.previous_ball
        return state

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.vision.previous_ball = state["previous_ball"]


class GuidedQNetwork(nn.Module):
    """Add the visual CNN's six values to six values learned from motion cues."""

    def __init__(self, actions=6):
        super().__init__()
        self.image_values = QNetwork(actions)
        self.motion_values = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, actions))

    def forward(self, images, cues):
        return self.image_values(images) + self.motion_values(cues)


class GuidedReplay(ReplayBuffer):
    """Store cue vectors and a coach label alongside the original replay arrays."""

    def __init__(self, capacity, seed):
        super().__init__(capacity, seed)
        self.cues = np.zeros((capacity, 8), dtype=np.float32)
        self.next_cues = np.zeros_like(self.cues)
        self.coach_actions = np.zeros(capacity, dtype=np.int64)

    def add(self, state, action, reward, next_state, terminated, truncated, episode_step, coach_action):
        slot = self.count % self.capacity
        self.cues[slot] = state["cues"]
        self.next_cues[slot] = next_state["cues"]
        self.coach_actions[slot] = coach_action
        super().add(state["images"], action, reward, next_state["images"], terminated, truncated, episode_step)

    def sample(self, batch_size):
        batch = super().sample(batch_size)
        slots = batch.pop("slots")
        batch["cues"] = self.cues[slots]
        batch["next_cues"] = self.next_cues[slots]
        batch["coach_actions"] = self.coach_actions[slots]
        return batch

    def state_dict(self):
        state = super().state_dict()
        for name in ("cues", "next_cues", "coach_actions"):
            state["arrays"][name] = getattr(self, name)[:len(self)]
        return state


def guided_update(online, target, optimizer, batch, config):
    """Fit reward targets and use a second loss to favour the coach's action."""
    images = torch.as_tensor(batch["states"])
    cues = torch.as_tensor(batch["cues"])
    next_images = torch.as_tensor(batch["next_states"])
    next_cues = torch.as_tensor(batch["next_cues"])
    actions = torch.as_tensor(batch["actions"]).long()
    rewards = torch.as_tensor(batch["rewards"]).float()
    terminated = torch.as_tensor(batch["terminated"]).float()
    coach_actions = torch.as_tensor(batch["coach_actions"]).long()

    # TD loss learns about the action actually taken and the real game reward.
    all_values = online(images, cues)
    chosen_values = all_values.gather(1, actions[:, None]).squeeze(1)
    with torch.no_grad():
        future_target = target(next_images, next_cues)
        future_online = online(next_images, next_cues) if config.method == "double_dqn" else None
        next_values = bootstrap_values(future_online, future_target, config.method)
        expected = rewards + config.gamma * (1 - terminated) * next_values
    td_loss = F.smooth_l1_loss(chosen_values, expected)

    # Margin loss asks that the coach action rank at least 0.10 above alternatives.
    # Its contribution is zero when that ranking is already satisfied.
    margin = torch.full_like(all_values, config.coach_margin)
    margin.scatter_(1, coach_actions[:, None], 0.0)
    coach_values = all_values.gather(1, coach_actions[:, None]).squeeze(1)
    coach_loss = ((all_values + margin).max(dim=1).values - coach_values).mean()
    loss = td_loss + config.coach_weight * coach_loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = nn.utils.clip_grad_norm_(online.parameters(), config.max_grad_norm)
    optimizer.step()
    return {"loss": float(loss.detach()), "td_loss": float(td_loss.detach()),
            "coach_loss": float(coach_loss.detach()), "gradient_norm": float(gradient_norm),
            "mean_q": float(chosen_values.detach().mean()), "mean_target": float(expected.mean()),
            "coach_agreement": float((all_values.detach().argmax(1) == coach_actions).float().mean())}


@torch.inference_mode()
def guided_action(network, observation, epsilon, rng, action_count):
    if rng.random() < epsilon:
        return int(rng.integers(action_count))
    images = torch.as_tensor(observation["images"][None])
    cues = torch.as_tensor(observation["cues"][None])
    return int(network(images, cues).argmax(dim=1).item())


@torch.inference_mode()
def evaluate_guided(network, config, step, episodes, train_elapsed_s=0.0, video_path=None):
    """Evaluate raw reward. Passing None measures the scripted coach itself."""
    import imageio.v2 as imageio
    env = GuidedPongEnv(config)
    was_training = network.training if network is not None else False
    if network is not None:
        network.eval()
    writer = None
    rows = []
    try:
        if video_path is not None:
            Path(video_path).parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(video_path, fps=15, codec="libx264", macro_block_size=2, quality=7)
        for index in range(episodes):
            eval_seed = 100_000 + config.seed * 100 + index
            rng = np.random.default_rng(eval_seed + 1_000_000)
            observation, _ = env.reset(seed=eval_seed)
            score, length, frames = 0.0, 0, 0
            started = time.perf_counter()
            terminated = truncated = False
            while not (terminated or truncated):
                if writer is not None and index == 0 and length < 900:
                    writer.append_data(env.raw.render())
                if network is None:
                    action = demonstration_action(observation["cues"])
                    if rng.random() < config.eval_epsilon:
                        action = int(rng.integers(env.action_count))
                else:
                    # The evaluation policy only uses the network's values.
                    action = guided_action(network, observation, config.eval_epsilon, rng, env.action_count)
                observation, reward, terminated, truncated, info = env.step(action)
                score += reward
                length += 1
                frames += info["actual_game_frames"]
            rows.append({"step": step, "eval_index": index, "eval_seed": eval_seed, "score": score,
                         "length": length, "game_frames": frames, "terminated": terminated,
                         "truncated": truncated, "eval_elapsed_s": time.perf_counter() - started,
                         "train_elapsed_s": train_elapsed_s})
    finally:
        if writer is not None:
            writer.close()
        env.close()
        if network is not None:
            network.train(was_training)
    return rows


class GuidedTrainer(Trainer):
    """Reuse checkpoint format and recovery, with the coach added to training."""

    def __init__(self, config, folder):
        config.validate()
        self.config, self.folder = config, Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        seed_everything(config.seed, config.torch_threads)
        self.env = GuidedPongEnv(config)
        self.observation, _ = self.env.reset(seed=config.seed)
        self.online = GuidedQNetwork(self.env.action_count)
        self.target = copy.deepcopy(self.online).eval()
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.replay = GuidedReplay(config.replay_capacity, config.seed + 1000)
        self.action_rng = np.random.default_rng(config.seed + 2000)
        self.step = self.updates = self.episode = self.episode_length = 0
        self.episode_return = self.elapsed_s = self.eval_elapsed_s = 0.0
        self.game_frames = self.reset_frames = 0
        self.initial_evaluated = False
        self.last_eval_step = -1
        self.last_eval_episodes = 0
        write_json(self.folder / "config.json", asdict(config))

    def advance(self, decisions):
        started = time.perf_counter()
        latest = None
        for _ in range(decisions):
            coach_action = demonstration_action(self.observation["cues"])
            # Coach fills replay during the existing 2,000-decision warm-up.
            # Afterwards the agent follows its normal epsilon-greedy policy.
            if self.step < self.config.learning_starts:
                action = coach_action
            else:
                action = guided_action(self.online, self.observation, epsilon_at(self.step, self.config),
                                       self.action_rng, self.env.action_count)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            stored_reward = float(np.sign(reward)) if self.config.clip_rewards else reward
            self.replay.add(self.observation, action, stored_reward, next_obs,
                            terminated, truncated, self.episode_length, coach_action)
            self.observation = next_obs
            self.step += 1
            self.game_frames += info["actual_game_frames"]
            self.episode_length += 1
            self.episode_return += reward
            if terminated or truncated:
                append_csv(self.folder / "episodes.csv", {"step": self.step, "episode": self.episode,
                           "score": self.episode_return, "length": self.episode_length,
                           "terminated": terminated, "truncated": truncated,
                           "train_elapsed_s": self.elapsed_s + time.perf_counter() - started,
                           "game_frames": self.game_frames})
                self.episode += 1
                self.episode_length, self.episode_return = 0, 0.0
                self.observation, _ = self.env.reset()
                self.reset_frames += self.env.raw.unwrapped.ale.getEpisodeFrameNumber()
            if self.step >= self.config.learning_starts and self.step % self.config.train_frequency == 0:
                batch = self.replay.sample(self.config.batch_size)
                latest = guided_update(self.online, self.target, self.optimizer, batch, self.config)
                self.updates += 1
                if self.updates % 100 == 0:
                    append_csv(self.folder / "updates.csv", {"step": self.step, "update": self.updates, **latest})
            if self.step % self.config.target_update_interval == 0:
                self.target.load_state_dict(self.online.state_dict())
        self.elapsed_s += time.perf_counter() - started
        return latest

    def evaluate_now(self, episodes):
        started = time.perf_counter()
        rows = evaluate_guided(self.online, self.config, self.step, episodes, self.elapsed_s)
        already = self.last_eval_episodes if self.last_eval_step == self.step else 0
        for row in rows[already:]:
            append_csv(self.folder / "evaluations.csv", row)
        self.eval_elapsed_s += time.perf_counter() - started
        self.last_eval_step, self.last_eval_episodes = self.step, episodes
        self.initial_evaluated = True
        print(f"{self.config.method} seed={self.config.seed} step={self.step:,} "
              f"evaluation={np.mean([row['score'] for row in rows]):.2f} ({episodes} games)", flush=True)


def run_guided(config, folder, until=None):
    return run_training(config, folder, stop_at=until, trainer_factory=GuidedTrainer)
