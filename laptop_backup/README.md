# Laptop backup (not part of the submission)

Everything from the laptop that wasn't already on GitHub, kept here for safekeeping only.

- `gpu_scratch/` - the original working folder (sits next to the repo on the laptop). Most of
  its scripts were later cleaned up and moved into `common/`, `experiments/` and `demos/`; the
  copies here are the original versions. It also holds the run checkpoints (`checkpoint.pt.gz`),
  replay-buffer metadata and the `explain_output` results from the watch/explain scripts.
- `PongProject_ignored_outputs/` - git-ignored outputs from inside the repo (`demos/explain_output`).

**Not included:** the four replay-buffer image files
`gpu_scratch/diagnose/runs/{A_dqn_s17_2M,A_dqn_s17_ext}/run/replay/{frames,next_frames}.npy`
(1.4 GB each - over GitHub's 100 MB per-file limit). Without them a checkpoint can't be resumed
with its exact replay buffer, but every trained policy and result is here.
