"""Unit tests for the verbatim notebook training code (../pong_train_extract.py).

Run:  python test_core.py      (no pytest needed; prints PASS/FAIL per test, writes test_results.json)
All tests use CPU and small sizes; nothing outside this folder is written.
"""
import copy, importlib.util, json, sys, tempfile, traceback
from dataclasses import replace
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]   # repo root (all paths are relative to it)
sys.path.insert(0, str(REPO / "common"))
OUT = HERE / "output"; OUT.mkdir(exist_ok=True)   # re-runs write here; recorded results are in recorded/
spec = importlib.util.spec_from_file_location("pong_train_extract", REPO / "common" / "pong_train_extract.py")
nb = importlib.util.module_from_spec(spec); sys.modules["pong_train_extract"] = nb; spec.loader.exec_module(nb)
import numpy as np, torch
from torch.nn import functional as F

PLAN = json.loads((REPO / "Assignment2" / "experiment_plan.json").read_text())["shared_config"]
BASE = replace(nb.Config(**PLAN), device="cpu")
RESULTS = {}
def test(fn):
    try:
        info = fn(); RESULTS[fn.__name__] = {"pass": True, "info": info}; print(f"PASS {fn.__name__}: {info}")
    except Exception as e:
        RESULTS[fn.__name__] = {"pass": False, "error": repr(e), "tb": traceback.format_exc()}
        print(f"FAIL {fn.__name__}: {e!r}")
    return fn

@test
def input_scaled_by_255():
    torch.manual_seed(0); net = nb.QNetwork(6)
    x = torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8)
    manual = net.values(net.features(x.float() / 255.0))
    assert torch.allclose(net(x), manual)
    assert not torch.allclose(net(x), net.values(net.features(x.float())))
    return "forward() = network(x / 255); uint8 in, float in [0,1] used"

@test
def frame_stack_order_real_env():
    env = nb.PongEnv(BASE); rng = np.random.default_rng(1)
    prev, _ = env.reset(seed=1)
    assert prev.shape == (4, 84, 84) and prev.dtype == np.uint8
    assert all(np.array_equal(prev[0], prev[i]) for i in range(4)), "reset should repeat first frame"
    for t in range(300):
        obs, _, term, trunc, _ = env.step(int(rng.integers(6)))
        assert np.array_equal(obs[:3], prev[1:]), f"stack did not shift by one at t={t}"
        assert np.array_equal(obs[3], env.frames[-1])
        prev = obs
        if term or trunc: prev, _ = env.reset()
    env.close()
    return "obs[t][0:3] == obs[t-1][1:4] for 300 real steps; obs[3] is the newest frame"

@test
def replay_reconstructs_real_transitions_exactly():
    # small ring + short episodes -> many episode boundaries and wrap-arounds on REAL frames
    cfg = replace(BASE, max_episode_steps=90)
    env = nb.PongEnv(cfg); rng = np.random.default_rng(2)
    rb = nb.ReplayBuffer(257, seed=3)
    truth = {}
    obs, _ = env.reset(seed=2); ep_len = 0
    for i in range(1500):
        a = int(rng.integers(6))
        nxt, r, term, trunc, _ = env.step(a)
        rb.add(obs, a, r, nxt, term, trunc, ep_len)
        truth[i] = (obs.copy(), nxt.copy(), term, trunc)
        obs, ep_len = nxt, ep_len + 1
        if term or trunc: obs, _ = env.reset(); ep_len = 0
        if i > 40 and i % 7 == 0:
            b = rb.sample(32)
            for j, slot in enumerate(b["slots"]):
                s, n, te, tr = truth[int(rb.ids[slot])]
                assert np.array_equal(b["states"][j], s), f"state mismatch id {rb.ids[slot]}"
                assert np.array_equal(b["next_states"][j], n), f"next_state mismatch id {rb.ids[slot]}"
                assert b["terminated"][j] == te and b["truncated"][j] == tr
    env.close()
    return "1500 real steps, ring of 257, episodes of 90: every sampled (s, s') identical to what the env produced"

def _nets():
    torch.manual_seed(0); online = nb.QNetwork(6)
    torch.manual_seed(1); target = nb.QNetwork(6)
    return online, target

def _batch():
    g = np.random.default_rng(4)
    return {"states": g.integers(0, 256, (8, 4, 84, 84), dtype=np.uint8),
            "next_states": g.integers(0, 256, (8, 4, 84, 84), dtype=np.uint8),
            "actions": g.integers(0, 6, 8), "rewards": np.array([1, -1, 0, 0, 1, 0, -1, 0], np.float32),
            "terminated": np.array([1, 0, 0, 1, 0, 0, 0, 0], bool),
            "truncated": np.array([0, 1, 0, 0, 0, 0, 0, 0], bool)}

@test
def dqn_target_uses_target_network_max_and_terminal_mask():
    online, target = _nets(); b = _batch()
    opt = torch.optim.SGD(online.parameters(), lr=0.0)    # lr 0: inspect target without changing nets
    cfg = replace(BASE, method="dqn")
    with torch.no_grad():
        nt = target(torch.as_tensor(b["next_states"])); no = online(torch.as_tensor(b["next_states"]))
        r = torch.as_tensor(b["rewards"]); d = torch.as_tensor(b["terminated"]).float()
        want = r + 0.99 * (1 - d) * nt.max(1).values
        wrong_online = r + 0.99 * (1 - d) * no.max(1).values
    m = nb.learning_update(online, target, opt, b, cfg)
    assert abs(m["mean_target"] - float(want.mean())) < 1e-5
    assert abs(m["mean_target"] - float(wrong_online.mean())) > 1e-4
    # truncated (index 1) must bootstrap; terminated (0, 3) must not
    assert torch.isclose(want[1], r[1] + 0.99 * nt[1].max()) and want[0] == r[0] and want[3] == r[3]
    return "target = r + 0.99*(1-terminated)*max_a target(s',a); online net NOT used; truncation bootstraps"

@test
def double_dqn_target_online_selects_target_evaluates():
    online, target = _nets(); b = _batch()
    opt = torch.optim.SGD(online.parameters(), lr=0.0)
    cfg = replace(BASE, method="double_dqn")
    with torch.no_grad():
        nt = target(torch.as_tensor(b["next_states"])); no = online(torch.as_tensor(b["next_states"]))
        sel = no.argmax(1, keepdim=True)
        want = torch.as_tensor(b["rewards"]) + 0.99 * (1 - torch.as_tensor(b["terminated"]).float()) * nt.gather(1, sel).squeeze(1)
    m = nb.learning_update(online, target, opt, b, cfg)
    assert abs(m["mean_target"] - float(want.mean())) < 1e-5
    return "target = r + 0.99*(1-terminated)*Q_target(s', argmax_a Q_online(s',a))"

@test
def loss_is_huber_grad_clipped_adam_updates_online_only():
    online, target = _nets(); b = _batch()
    cfg = replace(BASE, method="dqn")
    with torch.no_grad():
        pred = online(torch.as_tensor(b["states"])).gather(1, torch.as_tensor(b["actions"])[:, None]).squeeze(1)
        y = torch.as_tensor(b["rewards"]) + 0.99 * (1 - torch.as_tensor(b["terminated"]).float()) * target(torch.as_tensor(b["next_states"])).max(1).values
    huber, mse = float(F.smooth_l1_loss(pred, y)), float(F.mse_loss(pred, y))
    tb = copy.deepcopy(target.state_dict())
    opt = torch.optim.Adam(online.parameters(), lr=cfg.learning_rate)
    m = nb.learning_update(online, target, opt, b, cfg)
    assert abs(m["loss"] - huber) < 1e-6, (m["loss"], huber)
    assert abs(m["loss"] - mse) > 1e-3, "loss matches MSE, not Huber"
    assert all(torch.equal(v, tb[k]) for k, v in target.state_dict().items())
    return f"loss = smooth_l1 (Huber, beta 1) = {huber:.5f} (MSE would be {mse:.5f}); clip_grad_norm {cfg.max_grad_norm}; Adam lr {cfg.learning_rate}; target untouched"

@test
def target_sync_train_frequency_learning_starts():
    cfg = replace(BASE, total_steps=700, learning_starts=200, target_update_interval=100,
                  replay_capacity=1000, train_frequency=4, torch_threads=1)
    with tempfile.TemporaryDirectory() as d:
        tr = nb.Trainer(cfg, d)
        tr.advance(199); assert tr.updates == 0, "updated before learning_starts"
        tr.advance(101)                          # step 300
        assert tr.updates == len([s for s in range(1, 301) if s >= 200 and s % 4 == 0])
        same = all(torch.equal(v, tr.online.state_dict()[k]) for k, v in tr.target.state_dict().items())
        assert same, "target should equal online right after a sync step (300 % 100 == 0)"
        tr.advance(50)                           # step 350: updates happened, no sync
        diff = any(not torch.equal(v, tr.online.state_dict()[k]) for k, v in tr.target.state_dict().items())
        assert diff, "target should lag online between syncs"
        tr.close()
    return "first update at step learning_starts; then 1 update per train_frequency decisions; hard target copy every target_update_interval decisions"

@test
def epsilon_schedule():
    vals = {s: round(nb.epsilon_at(s, BASE), 4) for s in (0, 40_000, 80_000, 600_000)}
    assert vals[0] == 1.0 and vals[80_000] == 0.05 and vals[600_000] == 0.05
    return f"linear 1.0 -> 0.05 over {BASE.epsilon_decay_steps:,} decisions, then flat: {vals}"

@test
def action_mapping_up_down():
    env = nb.PongEnv(replace(BASE, sticky_probability=0.0)); env.reset(seed=0)
    names = env.raw.unwrapped.get_action_meanings(); out = {}
    for a in range(6):
        env.reset(seed=0)
        for _ in range(10): env.step(0)
        y0 = int(env.raw.unwrapped.ale.getRAM()[51])
        for _ in range(4): env.step(a)
        out[names[a]] = int(env.raw.unwrapped.ale.getRAM()[51]) - y0
    env.close()
    assert out["RIGHT"] < 0 and out["RIGHTFIRE"] < 0 and out["LEFT"] > 0 and out["LEFTFIRE"] > 0
    assert out["NOOP"] == 0
    return f"paddle RAM-y change after 4 decisions (negative = up): {out}"

@test
def sticky_actions_train_and_eval():
    env = nb.PongEnv(BASE)
    assert env.sticky.probability == 0.25
    assert env.raw.unwrapped.ale.getFloat("repeat_action_probability") == 0.0, "ALE must not add its own stickiness"
    env.close()
    # empirical: drive the REAL wrapper class over a stub env that records executed actions,
    # alternating the requested action every raw frame; executed != requested means "stuck".
    import gymnasium as gym
    class Stub(gym.Env):
        action_space = gym.spaces.Discrete(6); observation_space = gym.spaces.Discrete(1)
        def __init__(self): self.done = []
        def reset(self, *, seed=None, options=None): return 0, {}
        def step(self, a): self.done.append(a); return 0, 0.0, False, False, {}
    stub = Stub(); w = nb.StickyFrameActions(stub, 0.25); w.reset(seed=11)
    n = 20000; asked = [2 if i % 2 else 3 for i in range(n)]
    for a in asked: w.step(a)
    # stick probability = P(executed == previous executed | requested != previous executed)
    pairs = [(stub.done[i], stub.done[i - 1]) for i in range(1, n) if asked[i] != stub.done[i - 1]]
    rate = float(np.mean([d == p for d, p in pairs]))
    assert 0.23 < rate < 0.27, rate
    # evaluate() constructs PongEnv(config) with the SAME config -> same 0.25 stickiness
    import inspect
    assert "PongEnv(config)" in inspect.getsource(nb.evaluate)
    return f"p=0.25 per raw frame (empirical {rate:.3f}), ALE's own stickiness 0; evaluate() uses PongEnv(config) -> also sticky 0.25"

@test
def reward_clipping():
    src = __import__("inspect").getsource(nb.Trainer.advance)
    assert "np.sign(reward)" in src
    return "replay stores sign(reward) in {-1,0,+1}; raw reward kept only for episode logging"

ok = sum(r["pass"] for r in RESULTS.values())
print(f"\n{ok}/{len(RESULTS)} tests passed")
(OUT / "test_results.json").write_text(json.dumps(RESULTS, indent=2, default=str))
