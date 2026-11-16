# Work entirely in the Jupyter notebook

Open CS4287-Assign2_Atari_ID1_ID2_ID3.ipynb with the PongProject .venv kernel.
All project implementation, training, resuming, evaluation, checks, plotting and video code is now in notebook cells. No project .py imports are needed.

Run All with the training and video flags set to False to review the saved results. Section 12 shows the original gameplay. Section 17 explains the visual guidance, shows the old and new results, and contains the guided gameplay clips.

The original six runs reached 100,000 decisions each. Their recorded final means were -21.00 for DQN and -20.97 for Double DQN. The follow-up uses a visual cue that estimates where the ball will arrive, 2,000 coached warm-up decisions, and an extra margin loss. The Q network combines a CNN state value with learned action preferences from the cue. Both methods get the same guidance. Section 17 gives the formula and a numerical example.

To advance the follow-up to a planned checkpoint, set RUN_GUIDED_TRAINING = True in section 17, select GUIDED_UNTIL = 10000 or 20000, and run the notebook. Turn on REFRESH_GUIDED_VIDEOS to record the resulting checkpoints. Return both flags to False and save after the outputs have refreshed. The current guided checkpoint and result are recorded in results/guided/outcome.json.

The optional RUN_FRESH_PILOT switch reruns engineering checks and throughput measurements; these are separate from main results.

Keep experiment_plan.json, guided_plan.json, runs/, results/, videos/, AI_USE_LOG.json and the dependency files alongside the notebook. The original and guided checkpoints can be reopened with their respective notebook settings.

Team names/IDs, the filename, actual contribution paragraphs and earlier AI-use records still require completion.

archive_scripts/ preserves the earlier helper scripts and previous notebook only as historical backups. They are not needed for the revised notebook workflow. Do not use the old builder to overwrite the revised notebook.
