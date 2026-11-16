"""Build a teaching notebook with the exact runnable implementation in code cells."""
import ast
import json
import textwrap
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent
NAME = "CS4287-Assign2_Atari_ID1_ID2_ID3.ipynb"
SOURCE = (ROOT / "pong_dqn.py").read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)
LINES = SOURCE.splitlines()
NODES = {node.name: node for node in TREE.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}


def implementation(*names):
    parts = []
    for name in names:
        node = NODES[name]
        start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
        parts.append("\n".join(LINES[start:node.end_lineno]))
    return "\n\n\n".join(parts)


def build():
    cells = []
    md = lambda text: cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip()))
    code = lambda text: cells.append(nbf.v4.new_code_cell(textwrap.dedent(text).strip()))
    code('''
    # Team: PENDING actual team names and student IDs; do not submit these placeholders.
    # Execution: not yet verified; see the executed status and run inventory below.
    # Third-party implementations: https://github.com/Farama-Foundation/Gymnasium/blob/v1.4.0/gymnasium/wrappers/atari_preprocessing.py ; guidance: https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    ''')
    md('''
    # CS4287 Assignment 2: Atari Pong — DQN versus Double DQN

    **Question:** under the same amount of game experience, does Double DQN improve
    learning speed, final score, or consistency compared with DQN?

    This notebook follows **Option 2** of `CS4287-Assign2-RL_DQN-S1AY2627-v1.docx`
    (4 November 2026). The entire brief was read. It explicitly requires visible
    code outputs, extensive comments, diagrams, videos, references, one contribution
    paragraph per member, and an AI-use record. It permits sourced/AI-assisted code
    if it is acknowledged and understood. The brief specifies groups of two or three
    with different membership from Assignment 1; confirm this yourselves.

    **Honesty:** the cells below read actual saved measurements. A verification run
    checks engineering; it does not demonstrate learned Pong skill. A partial main
    run is never labelled a completed experiment. No result is filled in for a
    missing run. Team identities and human contributions are intentionally pending.
    The filename contains placeholder IDs and must be adjusted to the actual team.

    **How to use:** select the project's `.venv` Python 3.12 kernel and Run All.
    By default this rechecks a small weight update and regenerates the analysis
    from saved runs. The training cell has an explicit switch for resuming the
    longer experiments. Everything needed to understand the algorithm is visible
    here; `pong_dqn.py` contains the same implementation for command-line training.
    ''')
    code('''
    # Locate supporting results whether PyCharm starts in Assignment2 or its parent.
    from pathlib import Path
    import json
    import sys
    import os
    from IPython.display import display, Markdown, Image, Video, SVG
    candidates = [Path.cwd(), Path.cwd() / "Assignment2", Path.cwd() / "Pong_Assignment2"]
    ROOT = next((p for p in candidates if (p / "pong_dqn.py").exists()), None)
    if ROOT is None:
        raise FileNotFoundError("Open this notebook beside pong_dqn.py and its runs folder.")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    os.environ["MPLCONFIGDIR"] = str(ROOT / ".matplotlib")
    print("Supporting files:", ROOT)
    print("Notebook Python:", sys.executable)
    plan = json.loads((ROOT / "experiment_plan.json").read_text(encoding="utf-8"))
    print("Declared main plan:", plan["methods"], "seeds", plan["seeds"],
          "decisions per run", plan["steps_per_run"])
    ''')
    md('''
    ## 1. Why reinforcement learning suits Pong

    An **agent** is the player controlled by our program. An **environment** is the
    game it interacts with. An **action** is a joystick/button choice. A **reward**
    is feedback: scoring a point gives +1, conceding gives -1, and most frames give 0.
    A **policy** is the rule used to choose actions from observations.

    Pong is a sequence of decisions. Moving the paddle now changes what can happen
    later. We have no labelled dataset saying which action is correct for every
    image. Reinforcement learning lets the agent collect experience and learn from
    consequences. Its objective is high total future reward, rather than high
    classification accuracy on a set of labelled frames.

    **Q(s,a)** estimates the discounted future reward from taking action `a` in
    observed state `s`, then choosing good actions. Here `s` is four recent images.
    A **discount factor**, gamma, gives less weight to rewards further away. Our
    gamma is 0.99 per agent decision. A **Deep Q Network (DQN)** approximates Q using
    a neural network. Its six outputs are values, not action probabilities.
    ''')
    md('''
    ## 2. Reproducible setup and shared settings

    A **seed** initialises a random generator. Matching seeds give the two methods
    the same initial network weights and reproducible random streams; they do not
    force identical later trajectories once the policies differ. We use training
    seeds 17, 29 and 43. Evaluation uses a separate, documented set of seeds.

    The following cells define the code used in the experiments. Comments explain
    the purpose of the important operations. Configuration files are saved per run
    so settings cannot silently change when a run resumes. `requirements-lock.txt`
    records the full installed environment; `requirements.txt` lists direct packages.
    ''')
    imports = []
    for node in TREE.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append("\n".join(LINES[node.lineno-1:node.end_lineno]))
    code("\n".join(imports) + "\n\ngym.register_envs(ale_py)\n" + implementation("utc_now", "write_json", "append_csv", "append_event"))
    code(implementation("Config", "seed_everything", "epsilon_at"))
    code(implementation("system_information") + textwrap.dedent('''

    # Show the machine used for training, rather than guessing its capabilities.
    system_record = json.loads((ROOT / "results/system.json").read_text(encoding="utf-8"))
    display(system_record)
    shared_config = Config(**plan["shared_config"])
    print("Shared training settings:")
    display(asdict(shared_config))
    '''))
    md('''
    The inspected machine has an Intel i7-1360P CPU, 16 logical processors and
    about 15.6 GiB RAM. Only about 2 GiB was available initially. CUDA and XPU
    availability are reported above; training uses CPU. The physical graphics
    adapter is Intel Iris Xe integrated graphics, with no NVIDIA CUDA device.
    Disk space was ample.
    Four PyTorch threads were chosen from an actual learning-throughput pilot,
    not from game scores. A smaller CNN and a 20,000-transition replay memory keep
    the experiment practical. This is a resource-limited DQN comparison, not a
    reproduction of the original large-scale Atari benchmark.
    ''')
    md('''
    ## 3. The environment and image preprocessing

    The [official ALE Pong documentation](https://ale.farama.org/environments/pong/)
    specifies RGB images of shape **210 × 160 × 3**, with byte values 0–255, and a
    minimal set of six actions. We print their installed-library names below.
    Pong's sideways-controller names `RIGHT`/`LEFT` correspond to paddle movement
    in this game; we preserve the exact labels rather than renaming their indices.
    A whole match normally ends when a player reaches 21 points. The raw match
    return is points scored minus points conceded, ranging from -21 to +21.

    Processing follows Gymnasium's
    [AtariPreprocessing implementation](https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.AtariPreprocessing):
    random initial no-op actions (1–30), action repetition for four emulator frames,
    maximum over the latest two frames to reduce flicker, greyscale conversion,
    and resizing to 84 × 84. We retain the complete resized screen. We stack four
    observations in chronological order, giving **4 × 84 × 84** bytes. The network
    divides these bytes by 255 to obtain input values between 0 and 1.

    One image shows where the ball is. Several images can show where it came from,
    helping the network distinguish a ball travelling towards the paddle from one
    travelling away. This makes the observed state more informative, although it
    does not guarantee perfect knowledge of the emulator's hidden state.

    **Timing:** the base environment has `frameskip=1`. Only AtariPreprocessing
    repeats an action four times. One training step means one agent decision,
    usually four emulator frames; the logs also record actual emulator frames.
    Reset no-ops and evaluation interactions are recorded separately or excluded
    from the training-decision budget. Both methods have the same decision budget.

    **Sticky actions:** with probability 0.25 on each emulator frame, execute the
    previously executed action. This introduces uncertainty, following the
    evaluation motivation in [Machado et al.](https://arxiv.org/abs/1709.06009).
    Our practice-game settings were smooth and deterministic; the experiment uses
    this explicitly stated stochastic setting instead.

    During pilot recovery tests, ALE 0.12.1's native sticky-action state did not
    recover exactly in a new instance. Inspection of its
    [source](https://raw.githubusercontent.com/Farama-Foundation/Arcade-Learning-Environment/v0.12.1/src/ale/environment/stella_environment.cpp)
    showed that the last sticky action is outside the saved ALEState. We therefore
    set native `repeat_action_probability=0.0` and implement the same per-frame
    probability in the small wrapper below, saving its RNG and previous action.
    This is shared by both methods. It is a custom protocol, so scores should not
    be compared directly with papers using a different ALE setup.

    **Episode endings:** `terminated` means the game truly finished. `truncated`
    means an external limit cut it off. We reset for either, but remove the future
    value only for termination. A time cutoff uses the final observed next image,
    not the first image of the next match. We do not end training episodes at each
    point/life loss. The decision limit is 27,000; ALE may also report its own cap.
    If ALE cuts off inside an action repeat, we explicitly read its actual final
    screen because the preprocessing wrapper can otherwise return its old pooling
    buffer. This rare final observation uses direct greyscale resizing. An
    additional check forces such a cutoff and verifies the returned screen.
    See [Gymnasium's time-limit guidance](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/).
    ''')
    code(implementation("StickyFrameActions", "PongEnv"))
    code('''
    # Inspect real processed images without opening a game window.
    import matplotlib.pyplot as plt
    demonstration_env = PongEnv(shared_config)
    observation, info = demonstration_env.reset(seed=808)
    for _ in range(90):
        observation, _, ended, cut_off, _ = demonstration_env.step(2)
        if ended or cut_off:
            observation, info = demonstration_env.reset()
    print("Processed shape and dtype:", observation.shape, observation.dtype)
    print("Installed action names:", demonstration_env.raw.unwrapped.get_action_meanings())
    print("Base frame skip:", demonstration_env.raw.unwrapped._frameskip,
          "; preprocessing frame skip:", demonstration_env.pre.frame_skip)
    fig, axes = plt.subplots(1, 4, figsize=(10, 3), constrained_layout=True)
    for i, ax in enumerate(axes):
        ax.imshow(observation[i], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"Frame {i+1} ({'newest' if i==3 else 'older'})")
        ax.axis("off")
    plt.show()
    demonstration_env.close()
    print("These images illustrate preprocessing; this fixed-action example is not a trained agent.")
    ''')
    md('''
    ## 4. CNN architecture

    A **convolution** learns small image filters and reuses them across the screen.
    This is useful for recognising the ball and paddles in different positions.
    A **ReLU** keeps positive activations and replaces negative ones with zero,
    allowing the model to learn nonlinear relationships. A **fully connected**
    layer combines the extracted features into action values.

    Our CNN has three convolution layers, followed by 256 hidden units and six
    output units. The output has no softmax: a Q value can be negative and is not
    a probability. The channels and hidden width are smaller than the canonical
    Nature DQN to reduce CPU cost. Both methods use identical architecture,
    optimiser and initialisation. There is no dropout, weight decay, prioritised
    replay, duelling head or other modification that could confuse the comparison.
    The DQN background is [Mnih et al. (2015)](https://doi.org/10.1038/nature14236).
    ''')
    code(implementation("QNetwork"))
    code('''
    # An original diagram of THIS network, with tensor sizes after every layer.
    from matplotlib.patches import FancyBboxPatch
    diagram_labels = ["4 × 84 × 84\\nstack + /255", "Conv 16\\n8×8, stride 4\\n16 × 20 × 20",
                      "Conv 32\\n4×4, stride 2\\n32 × 9 × 9", "Conv 32\\n3×3, stride 1\\n32 × 7 × 7",
                      "Flatten\\n1,568", "Dense 256\\nReLU", "6 Q values\\n(no softmax)"]
    fig, ax = plt.subplots(figsize=(13, 3), constrained_layout=True)
    for i, label in enumerate(diagram_labels):
        x = i * 2.0
        ax.add_patch(FancyBboxPatch((x, .45), 1.55, 1.5, boxstyle="round,pad=.08",
                                    facecolor="#e7f0fa" if i < 4 else "#fff0df",
                                    edgecolor="#38546e"))
        ax.text(x+.775, 1.2, label, ha="center", va="center", fontsize=10)
        if i < len(diagram_labels)-1:
            ax.annotate("", xy=(x+1.87, 1.2), xytext=(x+1.65, 1.2),
                        arrowprops={"arrowstyle": "->", "color": "#38546e"})
    ax.set(xlim=(-.2, 13.9), ylim=(0, 2.4))
    ax.axis("off")
    ax.set_title("Shared CNN: ReLU follows each convolution and the hidden dense layer")
    fig.savefig(ROOT / "results/network_diagram.png", dpi=170)
    plt.show()
    example_network = QNetwork(6)
    print(example_network)
    print("Trainable parameters:", sum(p.numel() for p in example_network.parameters()))
    print("Batch-of-one output shape:", tuple(example_network(torch.zeros(1,4,84,84,dtype=torch.uint8)).shape))
    ''')
    md('''
    ## 5. Collecting and sampling experiences

    An **experience** or **transition** is `(state, action, reward, next_state,
    terminated, truncated)`. The agent collects one transition per decision.
    **Experience replay** stores recent transitions and samples a random
    **mini-batch** (a small group) for each learning update. This reuses experience
    and reduces the strong similarity between immediately adjacent training samples.

    Our replay buffer holds 20,000 transitions, overwriting the oldest when full.
    Naively storing both four-frame stacks would take about 1.05 GiB. We instead
    save the current last frame and the next frame per transition: about 269 MiB
    for image arrays. Previous frames reconstruct the rest of the stack. Episode
    counters prevent joining frames from different matches; absolute IDs detect
    overwritten history. The oldest three transitions are excluded after the
    ring wraps because their earlier history may have been overwritten. This is
    uniform sampling with replacement among the remaining valid entries.

    **Epsilon-greedy exploration** chooses a uniformly random action with
    probability epsilon; otherwise it chooses the action with largest online Q.
    Epsilon falls linearly from 1.0 to 0.05 over 80,000 training decisions. Random
    actions help collect varied experience. The first 2,000 decisions fill replay
    before learning begins; thereafter one batch of 32 is learned every four
    decisions. Rewards are clipped to their sign for training. Pong already has
    unit point rewards, so this normally does not change them. Evaluation always
    sums original rewards over a whole match.
    ''')
    code(implementation("ReplayBuffer", "choose_action"))
    md(r'''
    ## 6. Q-learning, maximisation bias and Double DQN

    There are two networks in **both** methods. The **online network** chooses
    actions and has its weights changed by learning. The **target network** is a
    frozen copy used to calculate targets. Every 2,000 training decisions we copy
    online weights into it. Frozen targets change less rapidly, making the
    prediction problem more stable. Double DQN does not introduce a third network.

    **Maximisation bias** means that choosing the largest of several noisy value
    estimates tends to select an overly optimistic estimate. Even zero-mean errors
    can create an upward bias after taking their maximum. This motivated
    [Double Q-learning (van Hasselt, 2010)](https://proceedings.neurips.cc/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html).
    [Double DQN (van Hasselt, Guez and Silver, 2016)](https://arxiv.org/abs/1509.06461)
    applies action selection and evaluation using the two existing DQN networks.
    It can reduce overestimation; it does not guarantee a higher score in Pong.

    Let `r` be reward, `gamma` the discount, `T` the true-termination flag,
    `theta` the online weights and `theta_minus` the frozen target weights.

    **DQN target:**
    $$y = r + \gamma(1-T)\max_{a'}Q_{\theta^-}(s',a').$$

    **Double DQN target:** first select
    $$a^* = \operatorname{argmax}_{a'}Q_\theta(s',a'),$$
    then evaluate
    $$y = r + \gamma(1-T)Q_{\theta^-}(s',a^*).$$

    This target calculation is the controlled modification. Double DQN uses an
    extra online forward pass when calculating targets, so equal game experience
    need not cost equal wall-clock time. The networks remain correlated because
    one is periodically copied from the other. That limits the independence of
    their estimation errors and means residual bias can remain.
    ''')
    code(implementation("bootstrap_values"))
    code('''
    # Synthetic illustration of maximisation bias; these are NOT Pong results.
    rng = np.random.default_rng(2026)
    errors_for_selection = rng.normal(0, 1, size=(50000, 6))
    independent_errors = rng.normal(0, 1, size=(50000, 6))
    selected = errors_for_selection.argmax(axis=1)
    single_estimator = errors_for_selection.max(axis=1)
    double_estimator = independent_errors[np.arange(len(selected)), selected]
    print("True value of every synthetic action: 0")
    print("Mean after taking max of one noisy estimate:", round(single_estimator.mean(), 3))
    print("Mean after independent selection/evaluation:", round(double_estimator.mean(), 3))
    print("Our real online/target networks are correlated; this toy calculation does not prove Pong improvement.")
    ''')
    md(r'''
    ## 7. The weight update, in detail

    The next cell is the exact function used during training. Read it in this order:

    1. **Prepare the batch.** Each state batch has shape `[32, 4, 84, 84]`.
       Actions and rewards have shape `[32]`. Image bytes become floats and are
       scaled inside `forward`. The action indices must be integers.
    2. **Make a prediction.** `online(states)` returns `[32, 6]`: six Q values for
       each experience. `gather(1, actions[:, None])` picks the value for the action
       actually recorded in each transition. `squeeze(1)` gives `[32]` predictions.
       The agent must learn about the action it took, not whichever action is now
       its favourite.
    3. **Build a target.** Inside `torch.no_grad()`, compute the next-state value
       using the DQN or Double DQN rule above. Add the observed reward and the
       discounted future value. For true terminal states the multiplier is zero,
       so `y = r`. For time-limit truncation it stays one; the next state is the
       final frame before reset. A target is a temporary estimate, not a known
       correct answer supplied by a teacher.
    4. **Calculate the error.** The temporal-difference (**TD**) error is
       `delta = target - prediction`. We use PyTorch's
       [SmoothL1Loss](https://docs.pytorch.org/docs/2.15/generated/torch.nn.SmoothL1Loss.html)
       with its default beta of 1. This is the Huber loss for these settings.
       For error `e = prediction - target`, the per-sample loss is
       `0.5 * e**2` when `abs(e) < 1`; otherwise it is `abs(e) - 0.5`.
       We average over the mini-batch. Large errors have a bounded slope before
       gradient clipping, making them less dominant than with squared error.
    5. **Clear old gradients.** `optimizer.zero_grad(set_to_none=True)` prevents
       gradients from earlier batches accumulating unintentionally.
    6. **Backpropagate.** `loss.backward()` applies the chain rule through the
       selected output, dense layers, ReLUs and convolutions. For a weight `w`,
       `dL/dw = (dL/dq) * (dq/dw)`.
       In the quadratic part, `dL/dq = prediction - target` for one sample;
       in the linear part it is +1 or -1. Batch averaging divides each sample's
       contribution by the batch size. Shared hidden weights can affect all action
       predictions, even though the loss uses only each recorded action.
    7. **Change the online weights.** Clip the total gradient norm to 10, then
       `optimizer.step()` applies [Adam](https://docs.pytorch.org/docs/2.15/generated/torch.optim.Adam.html)
       at learning rate 0.0001. Adam tracks moving averages of gradients and squared
       gradients to scale updates. Its update is not simply `w -= alpha*gradient`.
       The target network is excluded from the optimiser, has gradients disabled,
       and changes only at the separate hard-copy step.

    `no_grad()` is crucial: the target must stay fixed while this batch is fitted.
    Otherwise the optimiser could change both sides of the equation and reduce
    the reported loss without making the intended prediction correction. This is
    called a **semi-gradient** update: differentiate the prediction, not the
    bootstrapped target. It is still possible for Q-learning with neural networks
    to be unstable. A falling loss alone does not show that the agent plays well.
    ''')
    code(implementation("learning_update"))
    md('''
    ### A small numerical example

    Use only three actions for clarity. Suppose the next-state online values are
    `[1, 4, 2]` and target values are `[3, 2, 1]`. Let reward = 1, gamma = 0.9 and
    current prediction = 2.2. This is a nonterminal transition.

    DQN chooses the largest target value, 3: its target is `1 + 0.9*3 = 3.7`.
    Double DQN selects action index 1 using online value 4, then reads target
    value 2 for that action: its target is `1 + 0.9*2 = 2.8`.
    The DQN error is `2.2-3.7 = -1.5`, giving Huber loss `1.5-0.5 = 1.0`.
    The Double DQN error is `2.2-2.8 = -0.6`, giving loss `0.5*0.6² = 0.18`.
    A lower target/loss in this example is not evidence that Double DQN is correct
    about the true return; it illustrates how the formulas differ.

    To see an actual weight change simply, replace the CNN by one scalar weight:
    `q = w*x`, with `w=1.1`, `x=2`. For the Double DQN target, `dL/dq=-0.6` and
    `dq/dw=2`, so `dL/dw=-1.2`. One **SGD** update with rate 0.1 changes `w` to
    `1.1-0.1*(-1.2)=1.22`, increasing the prediction to 2.44, closer to 2.8.
    This scalar example uses SGD to expose the arithmetic. The real CNN uses Adam,
    a batch mean, and optional gradient clipping as shown in the implementation.
    ''')
    code('''
    # Execute the arithmetic, including autograd and a real scalar optimiser step.
    next_online = torch.tensor([[1., 4., 2.]])
    next_target = torch.tensor([[3., 2., 1.]])
    for method in ("dqn", "double_dqn"):
        y = 1. + .9 * bootstrap_values(next_online, next_target, method)
        loss = F.smooth_l1_loss(torch.tensor([2.2]), y)
        print(method, "target =", round(y.item(), 3), "Huber loss =", round(loss.item(), 3))
    w = nn.Parameter(torch.tensor(1.1))
    scalar_optimizer = torch.optim.SGD([w], lr=.1)
    q_before = w * 2.
    scalar_loss = F.smooth_l1_loss(q_before, torch.tensor(2.8))
    scalar_optimizer.zero_grad()
    scalar_loss.backward()
    print("Scalar gradient:", round(w.grad.item(), 3))
    scalar_optimizer.step()
    print("Updated weight:", round(w.item(), 3), "; updated prediction:", round((w*2.).item(), 3))
    # The future value vanishes only for real termination, not for a time cutoff.
    print("True terminal target:", 1.)
    print("Time-limit truncation target:", 1. + .9 * 2.)
    ''')
    code('''
    # Run a genuine CNN/Adam update and check that only online weights change.
    # This is a learning-update verification, not a Pong performance experiment.
    from verify_and_pilot import check_targets, check_replay, check_weight_update
    print("Equation check:", check_targets())
    print("Replay reset/overwrite check:", check_replay())
    display(check_weight_update())
    verification = json.loads((ROOT / "results/verification.json").read_text(encoding="utf-8"))
    print("Saved pilot verification (including exact recovery):")
    display(verification)
    ''')
    md('''
    ## 8. Training, evaluation and recovery

    The loop chooses an action, steps the game, saves the transition and sometimes
    learns a mini-batch. We preserve raw episode scores in `episodes.csv`, sampled
    update diagnostics in `updates.csv`, throughput in `progress.csv`, and each
    evaluation game in `evaluations.csv`. Configuration and status JSON files
    accompany every planned run, including a failed or unstarted run. JSONL event
    files record exceptions and recovery. Pilot/debug runs live under `runs/pilot`
    and are excluded from main performance plots.

    Evaluation uses a separate environment, the online network, and fixed
    **epsilon = 0.01**: 99% greedy decisions and 1% uniformly random decisions.
    The RNG and starting seeds are separate from training. Periodic evaluation
    uses five full games at step 0 and every 20,000 decisions; final evaluation
    uses ten games at 100,000. The first five final starts match the periodic starts.
    There are no optimiser steps, replay writes or training-epsilon changes during
    evaluation. We report original scores and episode lengths, plus truncation flags.

    Full checkpoints include both networks, Adam state, replay, current stacked
    observation, in-progress episode score/length, Python/NumPy/PyTorch/action/replay
    RNG states, ALE state, preprocessing buffers, and the explicit sticky-action
    state. They are compressed and replaced atomically. `policy.pt` is a small
    inference-only copy. Recovery tests compare continued versus restored pixels,
    rewards and weights, rather than merely checking that a file can be opened.
    Log rows beyond the last saved checkpoint are archived before rollback, so
    a crash cannot silently count transitions twice. At most the work since the
    latest checkpoint may need to be repeated after an abrupt process/power loss.
    ''')
    code(implementation("evaluate"))
    code(implementation("Trainer"))
    code(implementation("run_training"))
    md('''
    ## 9. Pilot, declared experiment and actual execution status

    The pilot tested actual environment interaction, preprocessing, CNN learning,
    logs and exact checkpoint recovery. It also tested 1, 2 and 4 CPU threads,
    measuring 1,000 decisions **with 250 learning updates**, after 400 warm-up
    decisions. Hardware throughput selected the thread setting; no main score was
    used. All pilot records and the early recovery failure are retained.

    The fixed main plan uses three independent training seeds per method,
    equal 100,000-decision budgets and otherwise identical settings. Runs alternate
    methods across seeds and proceed in rounds to limit differences caused by
    running one method only while the laptop is cool. The training estimate is
    approximate: Double DQN's extra forward pass, evaluation, saving and thermal
    slowdown add time. 100,000 decisions is much smaller than full Atari research
    training and may be insufficient to learn Pong. Insufficient improvement is
    therefore interpreted cautiously.

    The primary measure is the mean raw final score for each training seed, then
    the mean of these three seed means. Learning speed is the area under the
    evaluation curve divided by decisions and the first evaluation checkpoint
    with mean score at least zero. This threshold was fixed before the main runs.
    Variation is the sample standard deviation across the three seed means.
    Individual game variation is also shown. Three seeds support a descriptive
    comparison, not a strong statistical significance claim.
    ''')
    code('''
    import pandas as pd
    throughput = pd.DataFrame(json.loads((ROOT / "results/throughput.json").read_text()))
    display(throughput)
    print("Declared plan created:", plan["created_utc"])
    print("Pilot-based training-only estimate for all six runs (hours):",
          round(plan["estimated_training_hours_all_runs"], 2))
    display(json.loads((ROOT / "results/pilot_development_notes.json").read_text()))
    # Safe default: inspecting/Run All regenerates analysis without silently
    # starting several hours of training. Set True to finish/resume the saved plan.
    RUN_MAIN_TRAINING = False
    if RUN_MAIN_TRAINING:
        from run_experiments import execute
        execute()  # Resumes all six runs using their saved configurations.
    else:
        print("Long training not relaunched by Run All; saved run inventory follows.")
    ''')
    # Include the analysis implementation, so statistics are reviewable in notebook.
    md('''
    ## 10. Actual results and comparison

    The analysis below reads existing CSV files only. It keeps within-run game
    variation separate from variation between independently trained seeds. When
    runs are unfinished, it compares only the largest evaluated horizon shared by
    all six runs. A partially trained policy's latest evaluation is labelled as
    such; it is not substituted for the declared final 100,000-step result.

    The score curve displays each seed and the method mean; shading is **one
    across-seed standard deviation**, not a confidence interval. The area under
    this sparsely measured curve uses straight-line interpolation, so it is only
    a coarse learning-speed measure. A threshold time is interval-censored by the
    evaluation spacing: we know it was observed at that checkpoint, not the exact
    decision at which skill changed. `NaN` in threshold time means not observed.
    ''')
    analysis_source = (ROOT / "analyse_results.py").read_text(encoding="utf-8")
    analysis_tree = ast.parse(analysis_source)
    analysis_lines = analysis_source.splitlines()
    functions = [node for node in analysis_tree.body if isinstance(node, ast.FunctionDef)]
    analysis_impl = "\n\n".join("\n".join(analysis_lines[n.lineno-1:n.end_lineno]) for n in functions)
    code(textwrap.dedent('''
    COLORS = {"dqn": "#2171b5", "double_dqn": "#d94801"}
    LABELS = {"dqn": "DQN", "double_dqn": "Double DQN"}
    ''') + analysis_impl + textwrap.dedent('''

    findings, per_seed, method_summary = make_analysis(ROOT)
    display(per_seed)
    display(method_summary)
    display(Image(filename=str(ROOT / "results/learning_curves.png")))
    display(Image(filename=str(ROOT / "results/scores_and_loss.png")))
    '''))
    code('''
    # Report status from artifacts rather than assuming training finished.
    all_complete = findings["completed_main_runs"] == findings["planned_runs"]
    horizon = findings["common_evaluated_horizon"]
    status_text = "Completed declared main training" if all_complete else "Main training incomplete"
    display(Markdown(f"**{status_text}.** {findings['completed_main_runs']}/{findings['planned_runs']} runs "
        f"have the declared final evaluation. The shared evaluated budget is **{horizon:,} decisions per run**; "
        f"the plan is {findings['planned_steps_per_run']:,}. Actual saved evaluation games: "
        f"{findings['actual_evaluation_games']}."))
    if len(method_summary) == 2:
        rows = method_summary.set_index("method")
        difference = findings["paired_score_difference_double_minus_dqn"]
        display(Markdown(f"At the equal observed budget, mean scores are **DQN {rows.loc['dqn','mean_score']:.2f}** "
            f"and **Double DQN {rows.loc['double_dqn','mean_score']:.2f}**. The mean paired difference "
            f"(Double minus DQN) is **{difference:+.2f} points**. "
            f"This is {'an interim' if not all_complete else 'a budget-limited final'} observation, "
            "not evidence about what unlimited training would achieve."))
        if not per_seed["first_nonlosing_step"].notna().any():
            display(Markdown("No run has reached the declared non-losing threshold in the observed evaluations. "
                "There is no observed threshold-learning-speed winner; this does not prove equal learning ability."))
        if np.allclose(rows["mean_score"].to_numpy(), -21.):
            display(Markdown("**Both methods are at the losing-score floor (-21).** "
                "The evaluations have not demonstrated improved play. Any zero score variation here "
                "means consistently losing, rather than consistently skilled agents. "
                "Equal floor-level curves cannot establish a learning-speed or final-performance advantage."))
    ''')
    md('''
    ## 11. Critical evaluation

    **What the measurements support.** Original game return is directly relevant:
    a higher return means scoring more points relative to the opponent. Compare
    both the numerical difference and seed-by-seed behaviour in the table. Similar
    final scores would make earlier curve values, area under the curve and threshold
    times useful, provided there was meaningful improvement. If both stay near
    -21, similar results mean both performed poorly within this budget; that is
    not proof that their learning speeds are equal.

    **What they do not support.** Weight changes, a running game or low training
    loss do not establish learned skill. If the main budget is unfinished, final
    performance remains unmeasured. If final scores differ slightly across only
    three seeds and ten games, the difference may reflect chance and varied starts.
    No claim of statistical significance is made. A negative or inconclusive
    comparison is acceptable, and must not be rewritten as a Double DQN success.

    **Why familiar supervised metrics are insufficient.** Accuracy and a confusion
    matrix require labelled correct actions, which we do not have. Training loss
    compares one network estimate with another bootstrapped estimate. It diagnoses
    the update but is not an external test of playing ability. Return, learning
    curves, episode lengths and variation across seeds are more informative here.
    A long episode might mean good rallies, slow loss or inactivity, so interpret
    length alongside the score and video rather than rewarding length alone.

    **Bias is not directly measured here.** Return comparison tests the practical
    efficacy of Double DQN in this setup. Neither a lower predicted Q nor a higher
    game score alone proves reduced maximisation bias. Direct bias measurement
    would require suitable estimates of true returns for the same state-action
    pairs, with a carefully defined future policy and uncertainty analysis. Our
    synthetic example only demonstrates the mechanism under known assumptions.

    **Limitations.** CPU training, small CNN, short budget and a 20,000-transition
    replay buffer can constrain learning and increase forgetting. Four resized
    greyscale frames lose detail. There are only three independent training seeds;
    evaluation games within one seed are not independent training replications.
    Repeated fixed evaluation starts improve comparisons but do not test all
    possible Pong states. Final starts overlap periodic ones and no separate
    validation-based tuning is performed. Sparse checkpoints can miss transient
    improvements. Sticky actions are implemented explicitly for recoverability,
    so this protocol differs from an untouched default ALE environment.

    **Next steps.** First finish all declared runs using the exact resume command
    below. Preserve these results. If further compute time is available, declare a
    new longer-budget experiment for both methods before looking at its outcomes;
    do not extend only the method that currently looks better. More training seeds
    and an additional held-out evaluation seed set would improve uncertainty
    estimates. Any new hyperparameter tuning should remain separate from final runs.
    ''')
    md('''
    ## 12. Short gameplay videos

    Each video below loads a real saved policy, uses the evaluation action rule
    (epsilon 0.01) and records at 15 frames per second because each displayed image
    follows a four-frame decision. There is no display window. Videos are limited
    to 900 decisions (about 60 seconds of game time) or the end of a match.
    A clip's partial score is not a full-match evaluation score. A poor checkpoint
    is labelled by its actual training step, without claiming it learned to win.
    The separate fixed-action preprocessing illustration was not a policy video.
    ''')
    code(implementation("record_video"))
    code('''
    # Clips are generated from saved weights by make_videos.py, then embedded here.
    # Keeping the MP4 data embedded allows playback without an external URL.
    for method in ("dqn", "double_dqn"):
        path = ROOT / "videos" / f"{method}_seed_17.mp4"
        metadata_path = path.with_suffix(".json")
        if path.exists() and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            display(Markdown(f"**{method}, seed 17 — checkpoint after "
                f"{metadata['training_steps']:,} decisions.** "
                f"This clip shows {metadata['recorded_decisions']} decisions; "
                f"partial score {metadata['partial_score']:+.0f}. "
                "Observe ball returns, paddle movement and missed points; the evaluation table, "
                "not this selected clip, determines the comparison."))
            display(Video(filename=str(path), embed=True, html_attributes="controls muted"))
        else:
            display(Markdown(f"**Video missing for {method}.** Run make_videos.py after its checkpoint exists."))
    ''')
    md('''
    ## 13. Resume and reproduce

    In PowerShell, from `C:\\Users\\tyqnn\\PycharmProjects\\PongProject`:

    ```powershell
    .\\.venv\\Scripts\\python.exe .\\Assignment2\\run_experiments.py
    .\\.venv\\Scripts\\python.exe .\\Assignment2\\analyse_results.py
    .\\.venv\\Scripts\\python.exe .\\Assignment2\\make_videos.py
    .\\.venv\\Scripts\\python.exe .\\Assignment2\\execute_notebook.py
    ```

    The first command resumes each saved run to its declared 100,000-step budget;
    it skips completed milestones and preserves the same exploration schedule.
    To do a bounded session, add `--until 40000`, for example. Ctrl+C saves the
    current run through Python's interruption handler; an abrupt process kill can
    only recover the latest already-written checkpoint. Keep the laptop plugged
    in and awake for the training interval. No paid services or external compute
    are used. The remaining commands refresh plots, clips and visible notebook outputs.

    To reproduce from scratch, make a **new copy** of this assignment folder,
    keep `experiment_plan.json` and the configs, and place the existing `runs`
    directory in an archive before starting new runs. Do not overwrite these
    observed logs. Install Python 3.12 and direct dependencies from
    `requirements.txt`; the exact original environment is in `requirements-lock.txt`.
    Reproducibility across different hardware/library builds is not guaranteed;
    exact recovery was checked on this machine and environment.
    ''')
    md('''
    ## 14. Contribution statements and use of generative AI

    **Required before submission:** enter the actual team names and IDs in the
    first notebook line, adjust the filename for two or three members, and replace
    the statements below with one honest paragraph per person. The user requested
    the notebook only and did not provide these details. No identities from other
    assignments have been reused.

    **Member 1 — NAME / ID PENDING:** describe your actual implementation,
    experiments, analysis, writing, checking and understanding contributions.
    Do not claim AI-generated work as unaided authorship.

    **Member 2 — NAME / ID PENDING:** actual contribution paragraph required.

    **Member 3 — NAME / ID PENDING, only if a three-person team:** actual
    contribution paragraph required; remove this paragraph for a two-person team.

    **AI contribution:** OpenAI Codex read the brief and existing practice code,
    inspected the environment, generated the training/analysis code and educational
    explanations, ran verification and the saved experiment work, and assembled
    this notebook. The human team must inspect, understand, check and acknowledge
    this assistance. Human contributions have not been inferred.

    **Sourced material:** Gymnasium/ALE and PyTorch are third-party libraries.
    AtariPreprocessing is reused directly. The network/replay/training code was
    generated for this project using the cited algorithm ideas and PyTorch API
    guidance; no external DQN repository or pretrained policy was copied. Our
    implementation choices include the smaller CNN, compact replay, explicit
    recoverable sticky-action wrapper, six-run comparison, logging and evaluation.

    `AI_USE_LOG.json` records this session's user prompts verbatim, a one-line
    response summary, and a one-line explanation of use. **You must also add all
    earlier assistance outside this session**, including earlier manual-play help;
    it cannot be recovered or invented by this notebook. Add future prompts too.
    ''')
    code('''
    # Print the full prompts and required one-line summaries inside the notebook.
    ai_log = json.loads((ROOT / "AI_USE_LOG.json").read_text(encoding="utf-8"))
    print(ai_log["outside_session_assistance"])
    for index, item in enumerate(ai_log["entries"], start=1):
        print(f"\\nAI interaction {index} — {item['date']} — {item['tool']}")
        print("PROMPT:\\n" + item["prompt"])
        print("RESPONSE SUMMARY: " + item["response_summary"])
        print("HOW USED: " + item["use"])
    ''')
    md('''
    ## 15. Primary references

    References were opened and checked on 10 November 2026. Explanations and
    diagrams are written for this project rather than reproduced verbatim.

    1. Mnih, V. et al. (2015). *Human-level control through deep reinforcement
       learning*. Nature, 518, 529–533.
       [Original paper and DOI](https://doi.org/10.1038/nature14236).
       Background for DQN, replay, target networks and image-based control.
    2. van Hasselt, H. (2010). *Double Q-learning*. Advances in Neural Information
       Processing Systems 23.
       [Original proceedings](https://proceedings.neurips.cc/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html).
       Motivation: maximisation bias and separate estimators.
    3. van Hasselt, H., Guez, A. and Silver, D. (2016). *Deep Reinforcement
       Learning with Double Q-learning*. AAAI 2016; arXiv preprint first posted 2015.
       [Original paper](https://arxiv.org/abs/1509.06461).
       The Double DQN selection/evaluation rule implemented here.
    4. Machado, M. C. et al. (2018). *Revisiting the Arcade Learning Environment:
       Evaluation Protocols and Open Problems for General Agents*. Journal of
       Artificial Intelligence Research, 61, 523–562; preprint 2017.
       [Authors' preprint](https://arxiv.org/abs/1709.06009).
       Motivation for clear evaluation protocols and sticky actions.
    5. Farama Foundation. *ALE Pong documentation*.
       [Official environment specification](https://ale.farama.org/environments/pong/).
       Observation/action specification and variants.
    6. Farama Foundation. *Gymnasium AtariPreprocessing*.
       [Official wrapper documentation](https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.AtariPreprocessing).
       [Versioned implementation reused](https://github.com/Farama-Foundation/Gymnasium/blob/v1.4.0/gymnasium/wrappers/atari_preprocessing.py).
    7. Farama Foundation. *Handling Time Limits*.
       [Official tutorial](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/).
       Correct reset versus bootstrap handling for termination/truncation.
    8. PyTorch. *Reinforcement Learning (DQN) Tutorial*.
       [Official implementation guidance](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html).
       Consulted for API/update guidance; this project's CNN/replay/experiment is custom.
    9. PyTorch. *SmoothL1Loss* and *Adam* API documentation.
       [Loss](https://docs.pytorch.org/docs/2.15/generated/torch.nn.SmoothL1Loss.html),
       [optimiser](https://docs.pytorch.org/docs/2.15/generated/torch.optim.Adam.html).
    10. Farama Foundation. *ALE 0.12.1 source*.
        [Sticky-action and state restoration implementation](https://raw.githubusercontent.com/Farama-Foundation/Arcade-Learning-Environment/v0.12.1/src/ale/environment/stella_environment.cpp),
        [ALEState serialisation](https://raw.githubusercontent.com/Farama-Foundation/Arcade-Learning-Environment/v0.12.1/src/ale/environment/ale_state.cpp).
        Used to diagnose the pilot recovery issue, not copied into our code.
    11. CS4287 teaching team (2026). *Assignment 2: Deep Reinforcement Learning*,
        Sem1 AY26/27 v1, 4 November 2026. Supplied assignment brief; authoritative
        source for submission requirements, deadline and marking criteria.
    ''')
    md('''
    ## 16. Code-inspection preparation and outstanding checks

    **Explain the project in your own words:** the agent sees four small game
    images, predicts the future value of six actions, sometimes explores randomly,
    and stores what happens. It learns from random groups of stored experiences.
    Its online prediction moves towards a reward-plus-future-value target. A frozen
    copy helps stabilise that target. Double DQN changes who chooses the next
    action: online chooses; target evaluates. We compare actual game scores using
    equal training experience and multiple seeds, without assuming a winner.

    Be ready to answer:

    - Why are four images more useful than one? What do 84 × 84 and `[B,4,84,84]` mean?
    - Why does the CNN output six values, and why is there no softmax?
    - What exactly is stored in replay, and why sample randomly?
    - How do episode counters and absolute IDs stop invalid frame histories?
    - What does epsilon control during training and during evaluation?
    - Which line selects Q for the action actually taken? Explain `gather`.
    - Calculate a target, TD error and Huber loss by hand for the numerical example.
    - Why are targets inside `no_grad()`? Which weights change in `backward`/`step`?
    - How does Adam differ from the scalar SGD illustration?
    - How does Double DQN differ from DQN, and why can maximisation create bias?
    - Why do we reset on truncation but still keep the future-value term?
    - Where is frame skip applied? Where is sticky-action probability applied?
    - What is included in a resumable checkpoint beyond network weights?
    - Which runs actually finished, what do their scores support, and what remains uncertain?
    - Why are three training seeds different from three evaluation games?
    - Which code is third-party/AI-assisted, and what did each team member actually do?

    Before submission: finish and refresh any incomplete declared experiments;
    inspect both videos and all plots; replace team/filename placeholders; write
    actual contribution paragraphs; add earlier and later AI prompts; rerun the
    notebook to keep execution status and outputs current. This notebook must not
    be called submission-ready while those requirements are still outstanding.
    ''')
    notebook = nbf.v4.new_notebook(cells=cells)
    # Catch accidental escape characters in prose/equations before delivery.
    for index, cell in enumerate(cells):
        if cell.cell_type == "markdown":
            assert not any(ord(char) < 32 and char not in "\n\r" for char in cell.source), index
    notebook.metadata = {
        "kernelspec": {"display_name": "PongProject (.venv)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.10"},
        "cs4287": {"option": "Atari", "team_details": "pending", "brief_read_fully": True,
                   "training_status_source": "results/findings.json"}}
    nbf.validate(notebook)
    nbf.write(notebook, ROOT / NAME)
    print(f"Built {NAME}: {len(cells)} cells; {sum(c.cell_type=='code' for c in cells)} code cells")


if __name__ == "__main__":
    build()
